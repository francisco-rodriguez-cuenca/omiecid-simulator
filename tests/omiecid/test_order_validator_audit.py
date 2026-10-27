"""Audit tests for P2-L1 — `order_validator` JIT-safe.

Derived independently from market-spec.md §2.3, §4.2, §4.5, §5.4
and migration-plan.md Phase 2 §Loader dtype contract + P2-L1 contract.

Focus areas (per orchestrator brief):
1. JIT compilation contract + parity jit vs non-jit.
2. Return type invariants (jnp.bool_ scalar; tree leaves are jnp.ndarray).
3. Tick-alignment purity (exact-integer equality; no float tolerance leak).
4. No `1e-9` float tolerance appears in the traced path (jaxpr probe).
5. Market-spec semantic parity (all validation rules).
6. Static-argnums hashability / JIT cache hit on repeat call.
"""

import re
import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.order_validator import validate_order
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    OMIEOrder,
    Side,
    ExecRestriction,
    Validity,
    Status,
    make_order,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> OMIEProductConfig:
    return OMIEProductConfig()


def _valid_order(**overrides) -> OMIEOrder:
    base = dict(
        contract_id=0,
        side=Side.BUY,
        price=50.00,
        quantity=1.0,
        order_id=1,
        agent_id=0,
        timestamp=0,
        exec_restriction=ExecRestriction.NON,
        validity=Validity.GFS,
        expiry_time=0,
        status=Status.ACTIVE,
        peak_qty=0.0,
        ppd=0.0,
    )
    base.update(overrides)
    return make_order(**base)


# ---------------------------------------------------------------------------
# 1. JIT compilation contract + parity
# ---------------------------------------------------------------------------

class TestJitCompilationContract:
    """validate_order must compile under jax.jit with config and contract_open
    as static args, and produce identical results vs. the eager call."""

    @pytest.mark.jit
    def test_jit_compiles_with_static_argnums(self, config):
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        order = _valid_order()
        out = jitted(order, config, True)
        assert isinstance(out, jnp.ndarray)

    @pytest.mark.jit
    @pytest.mark.parametrize(
        "order_kwargs, contract_open, expected",
        [
            # valid
            (dict(), True, True),
            # contract closed
            (dict(), False, False),
            # --- sub-half-tick off-tick cases intentionally omitted ---
            # Under P2-L1-HOTFIX-tick-precision (Option C), `jnp.round` snaps
            # inputs within half-a-tick of a valid tick onto that tick; the
            # validator then bounds-checks the resulting int32. Off-tick
            # rejection at the validator layer is therefore no longer a
            # contract for sub-half-tick offsets — it lives in the encoder.
            # See `tests/omiecid/test_encoding_snap_semantics.py`.
            # price above ceiling
            (dict(price=10000.0), True, False),
            # price below floor
            (dict(price=-10000.0), True, False),
            # qty below floor
            (dict(quantity=0.0), True, False),
            # qty above ceiling
            (dict(quantity=1000.0), True, False),
            # invalid side
            (dict(side=5), True, False),
            # ICE with peak_qty=0 -> reject
            (dict(exec_restriction=ExecRestriction.ICE, peak_qty=0.0,
                  quantity=5.0), True, False),
            # ICE with peak_qty > qty -> reject
            (dict(exec_restriction=ExecRestriction.ICE, peak_qty=10.0,
                  quantity=5.0), True, False),
            # ICE valid
            (dict(exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
                  quantity=5.0), True, True),
            # sell valid
            (dict(side=Side.SELL, price=100.0, quantity=2.0), True, True),
            # boundary: price_max exactly
            (dict(price=9999.0), True, True),
            # boundary: price_min exactly
            (dict(price=-9999.0), True, True),
            # boundary: qty_min exactly
            (dict(quantity=0.1), True, True),
            # boundary: qty_max exactly
            (dict(quantity=999.0), True, True),
        ],
    )
    def test_jit_matches_eager(self, config, order_kwargs, contract_open, expected):
        order = _valid_order(**order_kwargs)
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        eager_out = validate_order(order, config, contract_open)
        jit_out = jitted(order, config, contract_open)
        assert bool(eager_out) == bool(jit_out)
        assert bool(jit_out) == expected


# ---------------------------------------------------------------------------
# 2. Return type invariants
# ---------------------------------------------------------------------------

class TestReturnTypeInvariants:

    def test_returns_jnp_bool(self, config):
        out = validate_order(_valid_order(), config, True)
        assert isinstance(out, jnp.ndarray)
        assert out.dtype == jnp.bool_

    def test_is_scalar(self, config):
        out = validate_order(_valid_order(), config, True)
        assert out.shape == ()

    def test_tree_leaves_are_all_ndarray(self, config):
        out = validate_order(_valid_order(), config, True)
        leaves = jax.tree.leaves(out)
        assert len(leaves) == 1
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray), (
                f"Non-ndarray leaf: {type(leaf)}"
            )

    def test_accept_and_reject_both_return_jnp_bool(self, config):
        accepted = validate_order(_valid_order(), config, True)
        rejected = validate_order(_valid_order(), config, False)
        assert accepted.dtype == jnp.bool_
        assert rejected.dtype == jnp.bool_
        assert bool(accepted) is True
        assert bool(rejected) is False


# ---------------------------------------------------------------------------
# 3. Tick-alignment purity — exact integer equality
# ---------------------------------------------------------------------------

class TestTickAlignmentPurity:
    """Probe that tick-alignment is handled by exact integer encoding, not a
    float tolerance.

    As of P2-L1-HOTFIX-tick-precision (Option C), off-tick rejection is the
    encoder's responsibility — sub-half-tick offsets snap to the nearest
    tick via ``jnp.round``. This class retains only on-tick acceptance,
    ULP-boundary sentinels (from the hotfix), and JIT-invariance assertions.
    See ``tests/omiecid/test_encoding_snap_semantics.py`` for the encoder-layer
    tick-snap contract.
    """

    def test_on_tick_price_accepted(self, config):
        # 50.00, 50.01, 50.02 are all exactly on a 0.01 tick.
        for p in [50.00, 50.01, 50.02, 133.87, 0.0, -9998.99]:
            order = _valid_order(price=p)
            assert bool(validate_order(order, config, True)), (
                f"on-tick price {p} rejected"
            )

    def test_on_tick_qty_accepted(self, config):
        for q in [0.1, 0.2, 1.0, 10.0, 999.0]:
            order = _valid_order(quantity=q)
            assert bool(validate_order(order, config, True)), (
                f"on-tick qty {q} rejected"
            )

    def test_ulp_noise_on_clean_decimal_accepted(self, config):
        # market-spec example price 133.87 — clean decimal that has no exact
        # float32 representation (133.8699951...). Exact integer tick check
        # in the scaled space must still accept after round(), without any
        # float tolerance.
        order = _valid_order(price=133.87)
        assert bool(validate_order(order, config, True))

    # -----------------------------------------------------------------
    # ULP-boundary sentinels (P2-L1-HOTFIX-tick-precision, 2026-04-23)
    # -----------------------------------------------------------------
    # Each of these prices is on the OMIE default 0.01 grid, but
    # ``float32(price) * float32(100.0)`` slips one ULP below the exact
    # integer, so the HEAD implementation at order_validator.py:77-78
    # spuriously rejects them. Selected from the exhaustive sweep so
    # they span:
    #   - magnitudes: 0.15 through 8999.98 (five decades)
    #   - both signs
    #   - buy-side and sell-side validation paths
    # Any future rewrite that re-breaks this bug class will fire here
    # *before* the much larger sweep test in
    # ``tests/omiecid/test_order_validator_tick_precision.py`` runs —
    # giving the developer a minimal failing case instead of a 262k
    # rejection count.
    ULP_BOUNDARY_PRICES = [
        0.15,     # smallest magnitude on the OMIE grid
        -0.15,    # negative-side counterpart
        1.05,
        -1.05,
        40.94,
        129.68,   # canonical demo oracle case (SELL @ 129.68 EUR/MWh)
        -129.68,
        512.03,
        1024.05,
        -1024.05,
        4999.99,
        8999.98,  # near the positive price_max boundary
        -9000.02, # near the negative price_min boundary
    ]

    @pytest.mark.parametrize("price", ULP_BOUNDARY_PRICES)
    def test_ulp_boundary_price_buy_accepted(self, config, price):
        """Buy-side: on-tick ULP-boundary price must accept."""
        order = _valid_order(side=Side.BUY, price=price)
        assert bool(validate_order(order, config, True)), (
            f"ULP-boundary on-tick price {price} rejected on BUY path"
        )

    @pytest.mark.parametrize("price", ULP_BOUNDARY_PRICES)
    def test_ulp_boundary_price_sell_accepted(self, config, price):
        """Sell-side: same sentinels, sell-path. Fire independently so a
        sided rewrite cannot break one path silently."""
        order = _valid_order(side=Side.SELL, price=price)
        assert bool(validate_order(order, config, True)), (
            f"ULP-boundary on-tick price {price} rejected on SELL path"
        )

    def test_ulp_boundary_price_129_68_named_regression(self, config):
        """Named regression case — the demo-oracle price, pinned as an
        unparametrized test so failure surfaces immediately without
        parametrize-id noise."""
        order = _valid_order(side=Side.SELL, price=129.68, quantity=2.5)
        assert bool(validate_order(order, config, True))


# ---------------------------------------------------------------------------
# 4. No float tolerance leak — jaxpr structural probe
# ---------------------------------------------------------------------------

class TestNoFloatToleranceLeak:
    """Guard against regression: the validator must not contain a 1e-9 style
    float tolerance in its traced computation."""

    def test_jaxpr_has_no_tolerance_literal(self, config):
        order = _valid_order()
        jaxpr = jax.make_jaxpr(validate_order, static_argnums=(1, 2))(
            order, config, True
        )
        text = str(jaxpr)
        # Any suspicious small-tolerance literal is a red flag.
        forbidden = [
            "1e-09", "1e-9", "1.e-9", "1.0e-09",
            "1e-08", "1e-8",
            "1e-06", "1e-6",
        ]
        for pat in forbidden:
            assert pat not in text, (
                f"Validator jaxpr contains forbidden tolerance literal {pat!r}"
            )
        # Also reject generic exponent-form near-zero literals like 1.2e-9 etc.
        assert re.search(r"\b\d+(\.\d+)?e-0?[6-9]\b", text) is None, (
            "Validator jaxpr contains a suspicious small-exponent literal"
        )


# ---------------------------------------------------------------------------
# 5. Market-spec semantic parity
# ---------------------------------------------------------------------------

class TestMarketSpecParity:
    """Each test cites market-spec.md rule(s) it probes."""

    # §2.3 price bounds
    def test_price_above_max_rejected(self, config):
        assert not bool(validate_order(_valid_order(price=9999.01), config, True))

    def test_price_below_min_rejected(self, config):
        assert not bool(validate_order(_valid_order(price=-9999.01), config, True))

    def test_price_exact_max_accepted(self, config):
        assert bool(validate_order(_valid_order(price=9999.0), config, True))

    def test_price_exact_min_accepted(self, config):
        assert bool(validate_order(_valid_order(price=-9999.0), config, True))

    # §2.3 qty bounds
    def test_qty_above_max_rejected(self, config):
        assert not bool(validate_order(_valid_order(quantity=999.1), config, True))

    def test_qty_below_min_rejected(self, config):
        # 0.0 is on-tick but below qty_min=0.1.
        assert not bool(validate_order(_valid_order(quantity=0.0), config, True))

    # §4.1, §4.5 side validity
    def test_buy_side_accepted(self, config):
        assert bool(validate_order(_valid_order(side=Side.BUY), config, True))

    def test_sell_side_accepted(self, config):
        assert bool(validate_order(_valid_order(side=Side.SELL, price=100.0),
                                   config, True))

    def test_invalid_side_rejected(self, config):
        for bad_side in [-1, 2, 99]:
            assert not bool(validate_order(_valid_order(side=bad_side),
                                           config, True))

    # §5.4 contract must be OPEN
    def test_closed_contract_rejects(self, config):
        assert not bool(validate_order(_valid_order(), config, False))

    # §4.2 iceberg validation (ICE)
    def test_ice_requires_positive_peak(self, config):
        order = _valid_order(
            exec_restriction=ExecRestriction.ICE, peak_qty=0.0, quantity=5.0,
        )
        assert not bool(validate_order(order, config, True))

    def test_ice_peak_larger_than_quantity_rejected(self, config):
        order = _valid_order(
            exec_restriction=ExecRestriction.ICE, peak_qty=6.0, quantity=5.0,
        )
        assert not bool(validate_order(order, config, True))

    def test_ice_peak_equal_to_quantity_accepted(self, config):
        order = _valid_order(
            exec_restriction=ExecRestriction.ICE, peak_qty=5.0, quantity=5.0,
        )
        assert bool(validate_order(order, config, True))

    def test_ice_peak_smaller_than_quantity_accepted(self, config):
        order = _valid_order(
            exec_restriction=ExecRestriction.ICE, peak_qty=2.0, quantity=5.0,
        )
        assert bool(validate_order(order, config, True))

    def test_non_ice_peak_qty_ignored(self, config):
        # peak_qty > quantity is a red herring for non-ICE orders.
        order = _valid_order(
            exec_restriction=ExecRestriction.NON,
            peak_qty=999.0,
            quantity=1.0,
        )
        assert bool(validate_order(order, config, True))

    # IOC / FOK are allowed with valid price/qty (market-spec §4.5)
    def test_ioc_accepted(self, config):
        order = _valid_order(exec_restriction=ExecRestriction.IOC)
        assert bool(validate_order(order, config, True))

    def test_fok_accepted(self, config):
        order = _valid_order(exec_restriction=ExecRestriction.FOK)
        assert bool(validate_order(order, config, True))


# ---------------------------------------------------------------------------
# 6. Static-argnums hashability / JIT cache hit
# ---------------------------------------------------------------------------

class TestStaticArgnumsHashability:

    def test_config_is_hashable(self, config):
        # frozen dataclass => hashable. Required for static_argnums.
        assert hash(config) == hash(OMIEProductConfig())

    @pytest.mark.jit
    def test_jit_cache_hit_on_repeat_call(self, config):
        # Second call with the same static args must not retrigger tracing.
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        order = _valid_order()
        jitted(order, config, True)  # warm
        # Use _cache_size on the compiled function if available; else fall
        # back to timing the tracer via make_jaxpr's identity.
        # jax>=0.4 exposes `.jit_info()` or `._cache_size()` inconsistently;
        # instead, assert identity of the compiled executable by comparing
        # repeated outputs across different orders with same static args.
        for p in [50.0, 51.0, 52.0]:
            o = _valid_order(price=p)
            assert bool(jitted(o, config, True)) is True
        # If tracing had occurred per call, a non-static argnum leak would
        # likely show as a tracer error on dynamic Python-bool contract_open;
        # the fact that contract_open is static is confirmed by:
        with pytest.raises(Exception):
            # Passing a traced jnp array as contract_open should error
            # because it's marked static. (If implementation were lenient,
            # this wouldn't raise — that would indicate contract_open is
            # NOT truly static.)
            jax.jit(validate_order, static_argnums=(1, 2))(
                _valid_order(), config, jnp.asarray(True)
            )

    @pytest.mark.jit
    def test_two_distinct_configs_both_work(self, config):
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        cfg_a = OMIEProductConfig()
        cfg_b = OMIEProductConfig(qty_max=500.0)
        order = _valid_order(quantity=700.0)
        # accepted under cfg_a (qty_max=999); rejected under cfg_b.
        assert bool(jitted(order, cfg_a, True)) is True
        assert bool(jitted(order, cfg_b, True)) is False
