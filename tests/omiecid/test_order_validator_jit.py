"""Tests for P2-L1 — `order_validator` JIT-safe.

Behavioral contract for the JAX-traceable `validate_order`:

1. `jax.jit(validate_order, static_argnums=(1, 2))` compiles cleanly
   (`config` and `contract_open` are static; `order` is the traced input).
2. Return value is a single `jnp.bool_` scalar (not a Python `bool`,
   not a tuple, not a NamedTuple).
3. `jax.tree.leaves(result)` yields only `jnp.ndarray` leaves.
4. Tick-alignment check is performed on integers (no `1e-9` float
   tolerance). The function may encode the input internally; the
   contract is semantic: a value not aligned to its tick is rejected.
5. Semantic parity with the Phase 1 / 1.5 Python validator on every
   input used in `tests/omiecid/test_order_validator.py`.

All expected values are derived from
- docs/omiecid/market-spec.md §2.3, §4.2, §4.5
- docs/omiecid/agile-plan.md §P2-L1
- docs/omiecid/migration-plan.md §Phase 2 "Loader dtype contract"

These tests are black-box: they import only the public interface.
No assumption on how the seven checks are fused internally.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    make_order,
)
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_validator import validate_order

# Module-level marker (Story T1.1): every test in this file traces or
# compiles `validate_order` under JAX.  Marker pinned at module scope.
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Default OMIE product config (market-spec §2.3 values)."""
    return OMIEProductConfig()


def _make_valid_buy(price=45.50, quantity=10.0, **overrides):
    kwargs = dict(
        contract_id=37,
        side=Side.BUY,
        price=price,
        quantity=quantity,
        order_id=1,
        agent_id=0,
        timestamp=100,
    )
    kwargs.update(overrides)
    return make_order(**kwargs)


def _make_valid_sell(price=50.00, quantity=5.0, **overrides):
    kwargs = dict(
        contract_id=37,
        side=Side.SELL,
        price=price,
        quantity=quantity,
        order_id=2,
        agent_id=1,
        timestamp=200,
    )
    kwargs.update(overrides)
    return make_order(**kwargs)


def _as_bool(result) -> bool:
    """Coerce the validator return value to a Python bool for assertion.

    The contract demands the raw return be a `jnp.bool_` scalar; use
    `bool(...)` only after the shape/dtype asserts in the dedicated
    return-type test. This helper is used in semantic-parity tests
    where we only care about the accept/reject decision.
    """
    return bool(result)


# ---------------------------------------------------------------------------
# Section 1 — Return-type contract
# ---------------------------------------------------------------------------


class TestReturnTypeContract:
    """validate_order must return a single `jnp.bool_` scalar."""

    def test_return_is_jax_array(self, config):
        order = _make_valid_buy()
        result = validate_order(order, config, True)
        assert isinstance(result, jnp.ndarray), (
            f"validate_order must return a jnp.ndarray, got {type(result)}"
        )

    def test_return_dtype_is_bool(self, config):
        order = _make_valid_buy()
        result = validate_order(order, config, True)
        assert result.dtype == jnp.bool_, (
            f"validate_order must return dtype jnp.bool_, got {result.dtype}"
        )

    def test_return_is_scalar(self, config):
        order = _make_valid_buy()
        result = validate_order(order, config, True)
        assert result.shape == (), (
            f"validate_order must return a scalar (shape ()); got shape {result.shape}"
        )

    def test_return_not_python_bool(self, config):
        """The return must not be a plain Python bool (would leak host-side)."""
        order = _make_valid_buy()
        result = validate_order(order, config, True)
        assert not isinstance(result, bool), (
            "validate_order must not return a Python bool; use jnp.bool_"
        )

    def test_return_not_tuple_or_namedtuple(self, config):
        """The return must be a single scalar, not a container."""
        order = _make_valid_buy()
        result = validate_order(order, config, True)
        assert not isinstance(result, tuple), (
            "validate_order must return a scalar, not a tuple"
        )

    def test_tree_leaves_only_jax_arrays(self, config):
        """jax.tree.leaves(result) must yield only jnp.ndarray leaves."""
        order = _make_valid_buy()
        result = validate_order(order, config, True)
        leaves = jax.tree.leaves(result)
        assert len(leaves) == 1, (
            f"Expected exactly one leaf, got {len(leaves)}"
        )
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray), (
                f"All leaves must be jnp.ndarray; got {type(leaf)}"
            )

    def test_rejected_return_is_also_jax_bool(self, config):
        """Rejection path must also return jnp.bool_, not Python False."""
        order = _make_valid_buy(price=-20000.0)  # out of range
        result = validate_order(order, config, True)
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.bool_
        assert result.shape == ()


# ---------------------------------------------------------------------------
# Section 2 — JIT compilation contract
# ---------------------------------------------------------------------------


class TestJitCompilation:
    """jax.jit(validate_order, static_argnums=(1, 2)) must compile cleanly."""

    def test_jit_compiles_with_static_config_and_contract_open(self, config):
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        order = _make_valid_buy()
        # Must not raise.
        result = jitted(order, config, True)
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.bool_

    def test_jit_accept_equals_python_accept(self, config):
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        order = _make_valid_buy()
        assert _as_bool(jitted(order, config, True)) is True

    def test_jit_reject_equals_python_reject(self, config):
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        order = _make_valid_buy(price=-20000.0)
        assert _as_bool(jitted(order, config, True)) is False

    def test_jit_reject_on_closed_contract(self, config):
        """contract_open=False is a static arg and must still reject."""
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        order = _make_valid_buy()
        assert _as_bool(jitted(order, config, False)) is False

    def test_jit_recompiles_for_contract_open_flip(self, config):
        """contract_open is static, so True and False are distinct traces."""
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        order = _make_valid_buy()
        # Both traces must succeed and produce the correct answer.
        assert _as_bool(jitted(order, config, True)) is True
        assert _as_bool(jitted(order, config, False)) is False

    def test_jit_no_recompile_across_order_variation(self, config):
        """Varying the traced `order` must not force recompilation."""
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        # Warmup to establish the compiled trace.
        _ = jitted(_make_valid_buy(), config, True)
        trace_count_before = jitted._cache_size() if hasattr(jitted, "_cache_size") else None
        # Many distinct order values with the same pytree shape.
        for p in (45.50, 46.00, 50.10, -10.0, 9999.0):
            _ = jitted(_make_valid_buy(price=p), config, True)
        if trace_count_before is not None:
            assert jitted._cache_size() == trace_count_before, (
                "JIT must not recompile when only traced-arg values change"
            )


# ---------------------------------------------------------------------------
# Section 3 — Tick-alignment uses integer equality, not float tolerance
# ---------------------------------------------------------------------------


class TestTickAlignmentIntegerExact:
    """After P2-F0, the loader emits int32. The validator's tick check must
    be exact-integer (``==`` on int32, not ``abs(ratio - round(ratio)) < 1e-9``).

    As of P2-L1-HOTFIX-tick-precision (Option C), off-tick rejection is the
    encoder's responsibility — sub-half-tick offsets snap to the nearest
    tick via ``jnp.round`` and the validator bounds-checks the resulting
    int32. This class retains on-tick acceptance probes and the
    ``1e-9``-absence structural guardrail; the sub-half-tick off-tick
    probes have moved to ``tests/omiecid/test_encoding_snap_semantics.py``.
    """

    def test_price_on_tick_accepted(self, config):
        order = _make_valid_buy(price=45.51)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_qty_on_tick_accepted(self, config):
        order = _make_valid_buy(quantity=10.3)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_zero_price_accepted_as_on_tick(self, config):
        order = _make_valid_buy(price=0.0)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_tick_check_has_no_float_tolerance_source(self):
        """The module-level implementation must not embed a `1e-9` literal.

        This is an architect-pinned guardrail: the Phase 1 helper
        `_is_on_tick` used `abs(ratio - round(ratio)) < 1e-9`. After
        P2-L1 the tolerance is removed in favor of integer equality.
        """
        import inspect
        from omiecid_simulator.omiecid import order_validator as mod

        src = inspect.getsource(mod)
        assert "1e-9" not in src, (
            "order_validator.py must not contain a 1e-9 float tolerance "
            "after P2-L1; tick-alignment is exact-integer."
        )


# ---------------------------------------------------------------------------
# Section 4 — Semantic parity with Phase 1 / 1.5 behavior
# ---------------------------------------------------------------------------


class TestSemanticParityValid:
    """Every accept-case from test_order_validator.py still accepts."""

    def test_valid_buy_order_accepted(self, config):
        order = _make_valid_buy(price=45.50, quantity=10.0)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_valid_sell_order_accepted(self, config):
        order = _make_valid_sell(price=50.00, quantity=5.0)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_valid_ioc_order_accepted(self, config):
        order = _make_valid_buy(exec_restriction=ExecRestriction.IOC)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_valid_fok_order_accepted(self, config):
        order = _make_valid_sell(exec_restriction=ExecRestriction.FOK)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_price_at_min_accepted(self, config):
        order = _make_valid_buy(price=-9999.0)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_price_at_max_accepted(self, config):
        order = _make_valid_sell(price=9999.0)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_quantity_at_min_accepted(self, config):
        order = _make_valid_buy(quantity=0.1)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_quantity_at_max_accepted(self, config):
        order = _make_valid_sell(quantity=999.0)
        assert _as_bool(validate_order(order, config, True)) is True

    def test_negative_price_on_tick_accepted(self, config):
        order = _make_valid_buy(price=-50.01)
        assert _as_bool(validate_order(order, config, True)) is True


class TestSemanticParityReject:
    """Every reject-case from test_order_validator.py still rejects.

    As of P2-L1-HOTFIX-tick-precision (Option C), quantity-below-min probes at
    sub-qty_tick offsets (e.g. ``0.09``) snap onto ``qty_min`` via the encoder
    and are accepted by the validator. Only quantities that snap to an int
    below ``qty_int_min`` (e.g. ``0.0``, ``0.04``) still reject here. The
    encoder-layer snap contract lives in
    ``tests/omiecid/test_encoding_snap_semantics.py``.
    """

    def test_closed_contract_buy_rejected(self, config):
        order = _make_valid_buy()
        assert _as_bool(validate_order(order, config, False)) is False

    def test_closed_contract_sell_rejected(self, config):
        order = _make_valid_sell()
        assert _as_bool(validate_order(order, config, False)) is False

    def test_price_below_min_rejected(self, config):
        order = _make_valid_buy(price=-9999.01)
        assert _as_bool(validate_order(order, config, True)) is False

    def test_price_far_below_min_rejected(self, config):
        order = _make_valid_buy(price=-20000.0)
        assert _as_bool(validate_order(order, config, True)) is False

    def test_price_above_max_rejected(self, config):
        order = _make_valid_sell(price=9999.01)
        assert _as_bool(validate_order(order, config, True)) is False

    def test_price_far_above_max_rejected(self, config):
        order = _make_valid_sell(price=20000.0)
        assert _as_bool(validate_order(order, config, True)) is False

    def test_qty_zero_rejected(self, config):
        order = _make_valid_buy(quantity=0.0)
        assert _as_bool(validate_order(order, config, True)) is False

    def test_qty_negative_rejected(self, config):
        order = _make_valid_buy(quantity=-1.0)
        assert _as_bool(validate_order(order, config, True)) is False

    def test_qty_above_max_rejected(self, config):
        order = _make_valid_sell(quantity=999.1)
        assert _as_bool(validate_order(order, config, True)) is False

    def test_qty_far_above_max_rejected(self, config):
        order = _make_valid_sell(quantity=5000.0)
        assert _as_bool(validate_order(order, config, True)) is False

    def test_invalid_side_rejected(self, config):
        order = make_order(
            contract_id=37,
            side=99,  # invalid
            price=45.50,
            quantity=10.0,
            order_id=1,
            agent_id=0,
            timestamp=100,
        )
        assert _as_bool(validate_order(order, config, True)) is False

    def test_multiple_violations_rejected(self, config):
        order = _make_valid_buy(price=99999.0, quantity=10.05)
        assert _as_bool(validate_order(order, config, True)) is False


class TestIcebergParity:
    """ICE branch parity with Phase 1.5 behavior (market-spec §4.2)."""

    def test_iceberg_valid_accepted(self, config):
        order = _make_valid_buy(
            exec_restriction=ExecRestriction.ICE,
            peak_qty=5.0,
            ppd=0.0,
        )
        assert _as_bool(validate_order(order, config, True)) is True

    def test_iceberg_sell_valid_accepted(self, config):
        order = _make_valid_sell(
            exec_restriction=ExecRestriction.ICE,
            peak_qty=2.0,
            ppd=0.1,
        )
        assert _as_bool(validate_order(order, config, True)) is True

    def test_iceberg_peak_zero_rejected(self, config):
        order = _make_valid_buy(
            exec_restriction=ExecRestriction.ICE,
            peak_qty=0.0,
            ppd=0.0,
        )
        assert _as_bool(validate_order(order, config, True)) is False

    def test_iceberg_peak_negative_rejected(self, config):
        order = _make_valid_buy(
            exec_restriction=ExecRestriction.ICE,
            peak_qty=-1.0,
            ppd=0.0,
        )
        assert _as_bool(validate_order(order, config, True)) is False

    def test_iceberg_peak_exceeds_quantity_rejected(self, config):
        order = _make_valid_buy(
            quantity=10.0,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=11.0,
            ppd=0.0,
        )
        assert _as_bool(validate_order(order, config, True)) is False

    def test_iceberg_peak_equals_quantity_accepted(self, config):
        order = _make_valid_buy(
            quantity=10.0,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=10.0,
            ppd=0.0,
        )
        assert _as_bool(validate_order(order, config, True)) is True

    def test_iceberg_on_closed_contract_rejected(self, config):
        order = _make_valid_buy(
            exec_restriction=ExecRestriction.ICE,
            peak_qty=5.0,
        )
        assert _as_bool(validate_order(order, config, False)) is False

    def test_non_ice_peak_not_checked(self, config):
        """A NON order with peak_qty=0 is valid; peak_qty check is
        gated on exec_restriction == ICE (market-spec §4.2)."""
        order = _make_valid_buy(
            exec_restriction=ExecRestriction.NON,
            peak_qty=0.0,
        )
        assert _as_bool(validate_order(order, config, True)) is True


# ---------------------------------------------------------------------------
# Section 5 — JIT semantic parity (all of the above under jax.jit)
# ---------------------------------------------------------------------------


@pytest.fixture
def jitted_validate(config):
    return jax.jit(validate_order, static_argnums=(1, 2))


class TestJittedSemanticParity:
    """Every key decision case must agree between Python-mode and JIT."""

    @pytest.mark.parametrize(
        "factory,contract_open,expected",
        [
            (lambda: _make_valid_buy(), True, True),
            (lambda: _make_valid_sell(), True, True),
            (lambda: _make_valid_buy(price=-9999.0), True, True),
            (lambda: _make_valid_sell(price=9999.0), True, True),
            (lambda: _make_valid_buy(quantity=0.1), True, True),
            (lambda: _make_valid_sell(quantity=999.0), True, True),
            (lambda: _make_valid_buy(price=-9999.01), True, False),
            (lambda: _make_valid_sell(price=9999.01), True, False),
            (lambda: _make_valid_sell(quantity=999.1), True, False),
            # sub-half-tick off-tick probes (price=45.501, quantity=10.05)
            # and sub-qty_tick probes (quantity=0.09) removed per
            # P2-L1-HOTFIX-tick-precision (Option C): encoder now snaps these
            # onto the nearest tick; validator accepts the snapped int32
            # (in-range). See test_encoding_snap_semantics.py.
            (lambda: _make_valid_buy(), False, False),
            (
                lambda: _make_valid_buy(
                    exec_restriction=ExecRestriction.ICE,
                    peak_qty=5.0,
                ),
                True,
                True,
            ),
            (
                lambda: _make_valid_buy(
                    exec_restriction=ExecRestriction.ICE,
                    peak_qty=0.0,
                ),
                True,
                False,
            ),
            (
                lambda: _make_valid_buy(
                    quantity=10.0,
                    exec_restriction=ExecRestriction.ICE,
                    peak_qty=11.0,
                ),
                True,
                False,
            ),
        ],
    )
    def test_jit_parity(self, jitted_validate, config, factory, contract_open, expected):
        order = factory()
        result = jitted_validate(order, config, contract_open)
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.bool_
        assert _as_bool(result) is expected


# ---------------------------------------------------------------------------
# Section 6 — Behavioral trait: order fields are treated as traced inputs
# ---------------------------------------------------------------------------


class TestTracedOrderInputs:
    """The order is the traced input; passing jnp-array fields must work.

    After P2-F0, the loader emits int32 arrays and downstream JAX code
    constructs `OMIEOrder` NamedTuples whose fields are `jnp.ndarray`
    scalars (not Python scalars). The validator must accept this input
    shape and return the correct decision.
    """

    def test_order_with_jax_array_fields_accepted(self, config):
        order = make_order(
            contract_id=jnp.int32(37),
            side=jnp.int32(int(Side.BUY)),
            price=jnp.float32(45.50),
            quantity=jnp.float32(10.0),
            order_id=jnp.int32(1),
            agent_id=jnp.int32(0),
            timestamp=jnp.int32(100),
        )
        result = validate_order(order, config, True)
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.bool_
        assert _as_bool(result) is True

    def test_order_with_jax_array_fields_rejected_out_of_range(self, config):
        order = make_order(
            contract_id=jnp.int32(37),
            side=jnp.int32(int(Side.BUY)),
            price=jnp.float32(20000.0),  # out of range
            quantity=jnp.float32(10.0),
            order_id=jnp.int32(1),
            agent_id=jnp.int32(0),
            timestamp=jnp.int32(100),
        )
        result = validate_order(order, config, True)
        assert _as_bool(result) is False

    def test_jit_with_jax_array_order_fields(self, config):
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        order = make_order(
            contract_id=jnp.int32(37),
            side=jnp.int32(int(Side.BUY)),
            price=jnp.float32(45.50),
            quantity=jnp.float32(10.0),
            order_id=jnp.int32(1),
            agent_id=jnp.int32(0),
            timestamp=jnp.int32(100),
        )
        result = jitted(order, config, True)
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.bool_
        assert _as_bool(result) is True
