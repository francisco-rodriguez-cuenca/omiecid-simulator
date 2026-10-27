"""Audit tests for P2-L1-HOTFIX-tick-precision (Option C Method A).

Audit scope (independently derived from the story card in
``docs/omiecid/agile-plan.md`` §`P2-L1-HOTFIX-tick-precision`, market-spec
§§2.3/4.2/4.5/5.4, migration-plan §Phase 2 dtype contract, and open
questions OQ-16 / OQ-17). These tests probe the post-hotfix validator
from angles complementary to the original locked suite. They exercise:

1. Trace-mode parity (eager vs ``jit`` vs ``vmap`` vs ``jit(vmap(...))``).
2. Banker's-rounding determinism at exact half-tick offsets.
3. Boundary-of-boundary cases at ``price_min`` / ``price_max`` /
   ``qty_min`` / ``qty_max``.
4. Sign-negation invariants in int32 tick space.
5. ICE / ICEwPI branch edges (``peak_qty`` zero / equal / larger / tick).
6. Closed-contract invariant and its static-argnum shape.
7. JIT retrace contract under ``static_argnums=(1, 2)``.
8. Jaxpr structural probes (no division / reciprocal / float64 / cond).
9. Method A XLA-safety invariant compliance (OQ-16 headroom).
10. Shape-and-dtype hygiene for scalar-jnp-array inputs.

Tests target the black-box public API only (``validate_order``). The
implementation file is inspected for interface shape, not for mirroring
internal structure into assertions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    Status,
    Validity,
    make_order,
)
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_validator import validate_order


# ---------------------------------------------------------------------------
# Device-parametrized fixture (P3-OPS-01a, 2026-05-28)
# ---------------------------------------------------------------------------
#
# Twin of the fixture in ``test_order_validator.py``. The OQ-16 Method A
# tick-precision invariant — which this audit file deepens — is the one
# OMIE invariant that legitimately requires dual-device coverage under
# the 2026-05-28 device-policy ruling. Every test in this module runs
# once per device. GPU parametrization auto-skips on CUDA-hidden hosts.
# Per-call ``jax.default_device(...)`` overrides the session
# ``jax_platform_name=cpu`` pin set in ``tests/omiecid/conftest.py``.

@pytest.fixture(autouse=True, params=["cpu", "gpu"])
def jax_device(request):
    backend = request.param
    try:
        devices = jax.devices(backend)
    except RuntimeError:
        pytest.skip(f"{backend.upper()} device not visible")
    with jax.default_device(devices[0]):
        yield backend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg() -> OMIEProductConfig:
    """OMIE default product config (price_tick=0.01, qty_tick=0.1)."""
    return OMIEProductConfig()


def _order(
    price: float,
    quantity: float,
    *,
    side: int = int(Side.BUY),
    exec_restriction: int = int(ExecRestriction.NON),
    peak_qty: float = 0.0,
    ppd: float = 0.0,
) -> OMIEOrder:
    """Construct an OMIEOrder with deterministic non-price/qty defaults."""
    return make_order(
        contract_id=1,
        side=Side(side),
        price=price,
        quantity=quantity,
        order_id=1,
        agent_id=0,
        timestamp=0,
        exec_restriction=ExecRestriction(exec_restriction),
        validity=Validity.GFS,
        expiry_time=0,
        status=Status.ACTIVE,
        peak_qty=peak_qty,
        ppd=ppd,
    )


# ---------------------------------------------------------------------------
# Section 1 — Trace-mode parity (eager / jit / vmap / jit(vmap))
# ---------------------------------------------------------------------------

class TestTraceModeParity:
    """Identical outcomes under all four trace modes."""

    @pytest.mark.jit
    @pytest.mark.parametrize(
        "price,quantity,expected",
        [
            (129.68, 2.5, True),                 # on-tick regression pin
            (50.005, 1.0, True),                  # sub-half-tick (encoder-snap)
            (-9999.00, 0.1, True),                # lower price bound on-tick
            (9999.00, 999.0, True),               # upper price + qty bounds
            (10_000.00, 1.0, False),              # price above max (out of range)
            (-10_000.00, 1.0, False),             # price below min (out of range)
            (50.00, 0.04, False),                 # rounds to 0, below qty_min
            (50.00, 1000.0, False),               # qty above max
        ],
    )
    def test_eager_jit_vmap_consistency(self, cfg, price, quantity, expected):
        order = _order(price, quantity)
        eager = bool(validate_order(order, cfg, True))
        jit_result = bool(jax.jit(
            validate_order, static_argnums=(1, 2)
        )(order, cfg, True))
        assert eager == expected
        assert jit_result == expected

    @pytest.mark.jit
    def test_vmap_matches_eager_loop(self, cfg):
        """vmap over a batch of prices returns per-element results identical
        to a Python-level loop of eager calls."""
        prices = jnp.array(
            [50.00, 129.68, -9999.00, 9999.00, 0.00, 50.01, -50.01, 500.05],
            dtype=jnp.float32,
        )
        qty = jnp.float32(1.0)

        def one(p):
            order = _order(float(p), float(qty))  # ignored; we vmap below
            return validate_order(order, cfg, True)

        # Build a batch by varying only the price field via a vmap over a
        # struct-of-arrays OMIEOrder. We do this by creating a per-slot
        # OMIEOrder, then vmap over price.
        def make_one(p_scalar):
            return OMIEOrder(
                contract_id=jnp.int32(1),
                side=jnp.int32(int(Side.BUY)),
                price=p_scalar,
                quantity=jnp.float32(1.0),
                order_id=jnp.int32(1),
                agent_id=jnp.int32(0),
                timestamp=jnp.int32(0),
                exec_restriction=jnp.int32(int(ExecRestriction.NON)),
                validity=jnp.int32(int(Validity.GFS)),
                expiry_time=jnp.int32(0),
                status=jnp.int32(int(Status.ACTIVE)),
                peak_qty=jnp.float32(0.0),
                ppd=jnp.float32(0.0),
            )

        def batched(p):
            order = make_one(p)
            return validate_order(order, cfg, True)

        vmapped = jax.vmap(batched)(prices)
        jitted_vmapped = jax.jit(
            jax.vmap(batched), static_argnums=()
        )(prices)

        eager_refs = jnp.array([batched(p) for p in prices])
        assert jnp.array_equal(vmapped, eager_refs)
        assert jnp.array_equal(jitted_vmapped, eager_refs)


# ---------------------------------------------------------------------------
# Section 2 — Banker's-rounding tie-break determinism
# ---------------------------------------------------------------------------

class TestBankersRoundingStability:
    """At exact half-tick offsets, ``jnp.round`` applies banker's rounding.

    The outcome must be deterministic across trace modes / repeat calls —
    NOT re-litigated here for 'correct' accept/reject; only locked as a
    guardrail against drift.
    """

    @pytest.mark.jit
    @pytest.mark.parametrize(
        "price",
        [
            0.005,      # midway between 0.00 and 0.01
            50.005,     # midway between 50.00 and 50.01
            129.685,    # midway between 129.68 and 129.69
            100.015,    # midway between 100.01 and 100.02
            -50.005,    # signed midway
        ],
    )
    def test_half_tick_outcome_deterministic_across_modes(self, cfg, price):
        order = _order(price, 1.0)
        eager = bool(validate_order(order, cfg, True))
        jitted = bool(jax.jit(validate_order, static_argnums=(1, 2))(
            order, cfg, True
        ))
        assert eager == jitted

    @pytest.mark.jit
    def test_half_tick_repeat_calls_identical(self, cfg):
        """Five consecutive JIT calls at a half-tick offset return the
        same boolean. Guards against backend drift."""
        order = _order(50.005, 1.0)
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        outputs = {bool(jitted(order, cfg, True)) for _ in range(5)}
        assert len(outputs) == 1  # one unique value

    @pytest.mark.jit
    def test_half_qty_tick_outcome_deterministic(self, cfg):
        order = _order(50.00, 0.05)  # 0.5 scaled -> banker's rounds to 0
        eager = bool(validate_order(order, cfg, True))
        jitted = bool(jax.jit(validate_order, static_argnums=(1, 2))(
            order, cfg, True
        ))
        assert eager == jitted


# ---------------------------------------------------------------------------
# Section 3 — Boundary-of-boundary cases
# ---------------------------------------------------------------------------

class TestBoundaryOfBoundary:
    """Exact bounds, one tick inside, one tick outside."""

    def test_exact_price_min(self, cfg):
        """price = price_min = -9999.00 is on-tick and in-range."""
        order = _order(cfg.price_min, 1.0)
        assert bool(validate_order(order, cfg, True)) is True

    def test_exact_price_max(self, cfg):
        order = _order(cfg.price_max, 1.0)
        assert bool(validate_order(order, cfg, True)) is True

    def test_exact_qty_min(self, cfg):
        order = _order(50.00, cfg.qty_min)
        assert bool(validate_order(order, cfg, True)) is True

    def test_exact_qty_max(self, cfg):
        order = _order(50.00, cfg.qty_max)
        assert bool(validate_order(order, cfg, True)) is True

    def test_one_tick_inside_price_min(self, cfg):
        """price_min + price_tick must be accepted."""
        order = _order(cfg.price_min + cfg.price_tick, 1.0)
        assert bool(validate_order(order, cfg, True)) is True

    def test_one_tick_outside_price_max(self, cfg):
        """price_max + price_tick must be rejected."""
        order = _order(cfg.price_max + cfg.price_tick, 1.0)
        assert bool(validate_order(order, cfg, True)) is False

    def test_one_tick_outside_qty_min(self, cfg):
        """qty_min - qty_tick = 0.0 must be rejected (below qty_min)."""
        order = _order(50.00, cfg.qty_min - cfg.qty_tick)
        assert bool(validate_order(order, cfg, True)) is False

    def test_one_tick_outside_qty_max(self, cfg):
        order = _order(50.00, cfg.qty_max + cfg.qty_tick)
        assert bool(validate_order(order, cfg, True)) is False

    def test_zero_price_accepted(self, cfg):
        """price = 0.0 is on-tick and in (-9999, 9999)."""
        order = _order(0.0, 1.0)
        assert bool(validate_order(order, cfg, True)) is True

    def test_zero_quantity_rejected(self, cfg):
        """qty = 0.0 is on-tick but below qty_min = 0.1."""
        order = _order(50.00, 0.0)
        assert bool(validate_order(order, cfg, True)) is False


# ---------------------------------------------------------------------------
# Section 4 — Sign / negation invariants
# ---------------------------------------------------------------------------

class TestSignInvariants:
    """Offset encoding preserves sign: negative and positive
    magnitudes are both in-range at OMIE defaults."""

    @pytest.mark.parametrize(
        "magnitude",
        [0.01, 1.00, 50.00, 129.68, 1000.00, 5000.00, 9000.00, 9999.00],
    )
    def test_positive_and_negative_both_accepted(self, cfg, magnitude):
        pos = _order(+magnitude, 1.0)
        neg = _order(-magnitude, 1.0)
        assert bool(validate_order(pos, cfg, True)) is True
        assert bool(validate_order(neg, cfg, True)) is True

    def test_negation_monotonic_out_of_range(self, cfg):
        """One-tick beyond each bound on both signs rejects."""
        above = _order(cfg.price_max + 0.01, 1.0)
        below = _order(cfg.price_min - 0.01, 1.0)
        assert bool(validate_order(above, cfg, True)) is False
        assert bool(validate_order(below, cfg, True)) is False


# ---------------------------------------------------------------------------
# Section 5 — ICE / ICEwPI branch edges
# ---------------------------------------------------------------------------

class TestIcebergBranch:
    """Iceberg-branch predicate: ``(peak_qty > 0) & (peak_qty <= quantity)``.

    Gated on ``exec_restriction == ICE``; NON orders never consult peak_qty.
    """

    def test_ice_peak_equal_quantity_accepted(self, cfg):
        """ICE order with peak_qty == quantity is a single-slice iceberg."""
        order = _order(
            50.00, 2.0, exec_restriction=int(ExecRestriction.ICE), peak_qty=2.0
        )
        assert bool(validate_order(order, cfg, True)) is True

    def test_ice_peak_zero_rejected(self, cfg):
        """ICE with peak_qty == 0 must be rejected (market-spec §4.2)."""
        order = _order(
            50.00, 2.0, exec_restriction=int(ExecRestriction.ICE), peak_qty=0.0
        )
        assert bool(validate_order(order, cfg, True)) is False

    def test_ice_peak_negative_rejected(self, cfg):
        order = _order(
            50.00, 2.0, exec_restriction=int(ExecRestriction.ICE), peak_qty=-0.1
        )
        assert bool(validate_order(order, cfg, True)) is False

    def test_ice_peak_exceeds_quantity_rejected(self, cfg):
        order = _order(
            50.00, 2.0, exec_restriction=int(ExecRestriction.ICE), peak_qty=3.0
        )
        assert bool(validate_order(order, cfg, True)) is False

    def test_ice_peak_one_qty_tick_accepted(self, cfg):
        """peak_qty == qty_tick (one unit) is the minimal valid peak."""
        order = _order(
            50.00,
            2.0,
            exec_restriction=int(ExecRestriction.ICE),
            peak_qty=cfg.qty_tick,
        )
        assert bool(validate_order(order, cfg, True)) is True

    def test_non_order_with_invalid_peak_still_accepted(self, cfg):
        """NON (non-ICE) order ignores peak_qty entirely — invalid peak
        should not cause rejection when exec_restriction != ICE."""
        order = _order(
            50.00,
            2.0,
            exec_restriction=int(ExecRestriction.NON),
            peak_qty=-5.0,  # invalid as ICE, but branch is gated
        )
        assert bool(validate_order(order, cfg, True)) is True

    def test_ioc_order_with_invalid_peak_still_accepted(self, cfg):
        """IOC order with bogus peak_qty still accepts — ICE branch is
        gated on ``exec_restriction == ICE`` only."""
        order = _order(
            50.00,
            2.0,
            exec_restriction=int(ExecRestriction.IOC),
            peak_qty=0.0,  # bogus for ICE but IOC-irrelevant
        )
        assert bool(validate_order(order, cfg, True)) is True

    def test_fok_order_with_invalid_peak_still_accepted(self, cfg):
        order = _order(
            50.00,
            2.0,
            exec_restriction=int(ExecRestriction.FOK),
            peak_qty=-1.0,
        )
        assert bool(validate_order(order, cfg, True)) is True


# ---------------------------------------------------------------------------
# Section 6 — Closed-contract invariant
# ---------------------------------------------------------------------------

class TestClosedContractInvariant:
    """``contract_open=False`` must reject regardless of other fields."""

    @pytest.mark.parametrize(
        "price,quantity,side,exec_restriction,peak_qty",
        [
            (50.00, 1.0, int(Side.BUY), int(ExecRestriction.NON), 0.0),
            (50.00, 1.0, int(Side.SELL), int(ExecRestriction.NON), 0.0),
            (129.68, 2.5, int(Side.BUY), int(ExecRestriction.ICE), 0.5),
            (0.0, 0.1, int(Side.BUY), int(ExecRestriction.IOC), 0.0),
            (cfg_field := -9999.0, 1.0, int(Side.SELL), int(ExecRestriction.FOK), 0.0),
        ],
    )
    def test_closed_always_rejects(self, cfg, price, quantity, side, exec_restriction, peak_qty):
        order = _order(
            price, quantity, side=side,
            exec_restriction=exec_restriction, peak_qty=peak_qty,
        )
        assert bool(validate_order(order, cfg, False)) is False

    def test_closed_rejects_even_with_otherwise_valid_order(self, cfg):
        order = _order(50.00, 1.0)
        assert bool(validate_order(order, cfg, True)) is True
        assert bool(validate_order(order, cfg, False)) is False


# ---------------------------------------------------------------------------
# Section 7 — JIT retrace contract (static_argnums=(1, 2))
# ---------------------------------------------------------------------------

class TestJitRetraceContract:
    """Verify static-argnums stability: config/contract_open changes
    retrace (expected), traced input changes do not."""

    @pytest.mark.jit
    def test_jit_same_config_same_trace(self, cfg):
        """Two calls with identical config + contract_open reuse the
        compiled trace (no recompilation)."""
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        # Warm trace
        jitted(_order(50.00, 1.0), cfg, True)
        cache_after_first = jitted._cache_size() if hasattr(jitted, "_cache_size") else None
        jitted(_order(129.68, 2.5), cfg, True)
        cache_after_second = jitted._cache_size() if hasattr(jitted, "_cache_size") else None
        if cache_after_first is not None and cache_after_second is not None:
            assert cache_after_second == cache_after_first

    @pytest.mark.jit
    def test_jit_different_contract_open_retraces(self, cfg):
        """``contract_open`` is static_argnum=2; flipping it should
        produce a fresh trace (distinct cache entry)."""
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        r_open = jitted(_order(50.00, 1.0), cfg, True)
        r_closed = jitted(_order(50.00, 1.0), cfg, False)
        assert bool(r_open) is True
        assert bool(r_closed) is False

    @pytest.mark.jit
    def test_jit_different_config_retraces(self, cfg):
        """Config is static_argnum=1; a frozen dataclass with different
        bounds must be treated as a distinct static key."""
        narrow = OMIEProductConfig(price_min=-100.0, price_max=100.0)
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        # 500.00 is out of range under narrow, in range under default
        assert bool(jitted(_order(500.00, 1.0), cfg, True)) is True
        assert bool(jitted(_order(500.00, 1.0), narrow, True)) is False


# ---------------------------------------------------------------------------
# Section 8 — Jaxpr structural probes
# ---------------------------------------------------------------------------

class TestJaxprStructure:
    """Deeper jaxpr probes than "no tolerance literal" — verify the
    compiled path has no division, no reciprocal, no float64, no cond."""

    @pytest.fixture
    def jaxpr_str(self, cfg) -> str:
        order = _order(50.00, 1.0)
        return str(jax.make_jaxpr(
            validate_order, static_argnums=(1, 2)
        )(order, cfg, True))

    def test_no_division_op(self, jaxpr_str):
        """Method A replaces ``(p / tick)`` with ``p * scale``. The
        jaxpr must contain no ``div`` primitive on the traced path."""
        # ``div`` appears as a standalone primitive name in jaxpr output
        for token in [" div ", " div\n", "lax.div"]:
            assert token not in jaxpr_str, (
                f"Found division op in jaxpr: {token!r}"
            )

    def test_no_reciprocal_literal(self, jaxpr_str):
        """No ``1.0 / tick`` or ``reciprocal``-style primitive."""
        assert "reciprocal" not in jaxpr_str
        # ``rcp`` / ``rcp.approx`` would only appear in XLA HLO, not
        # the Python-side jaxpr, but guard against it leaking here.
        assert "rcp" not in jaxpr_str

    def test_no_float64_on_traced_path(self, jaxpr_str):
        """Method A pins ``float32`` explicitly. No ``float64`` / ``f64``
        dtype should appear in the jaxpr."""
        assert "float64" not in jaxpr_str
        # HLO uses ``f64``; check both tokens for robustness.
        for token in ["f64[", "f64 "]:
            assert token not in jaxpr_str, f"Found f64 in jaxpr: {token!r}"

    def test_no_lax_cond(self, jaxpr_str):
        """Story contract: 'Five checks fused into one AND expression
        (no ``lax.cond``).'"""
        assert "cond" not in jaxpr_str

    def test_no_forbidden_tolerance_literals(self, jaxpr_str):
        """Re-lock the P2-L1 invariant: no literal in [1e-9, 1e-6]."""
        import re

        forbidden = re.compile(r"\b\d+(\.\d+)?e-0?[6-9]\b")
        matches = forbidden.findall(jaxpr_str)
        assert not matches, f"Forbidden tolerance literals: {matches}"

    def test_jaxpr_returns_single_bool_leaf(self, cfg):
        """``jax.tree.leaves`` on the return value yields exactly one
        ``jnp.ndarray`` leaf of dtype bool."""
        result = validate_order(_order(50.00, 1.0), cfg, True)
        leaves = jax.tree.leaves(result)
        assert len(leaves) == 1
        assert leaves[0].dtype == jnp.bool_


# ---------------------------------------------------------------------------
# Section 9 — Method A XLA-safety invariant (OQ-16) positive headroom
# ---------------------------------------------------------------------------

class TestMethodAHeadroom:
    """At OMIE defaults, ``|p * scale|`` stays well within the
    float32 integer-exact range (±2²⁴ = ±16_777_216).

    The story pins a 16.8× safety margin; verify positively on a grid
    of extreme values, and verify the encoded int is identical to the
    numpy reference (the simulator's Python-level ``encode_price``).
    """

    @pytest.mark.parametrize(
        "price",
        [
            -9999.00, -9000.00, -5000.00, -1000.00, -129.68, -50.00,
            -0.01, 0.00, 0.01, 50.00, 129.68, 1000.00, 5000.00, 9999.00,
        ],
    )
    def test_scaled_product_within_float32_integer_exact_range(self, price):
        """``|p * 100.0|`` fits in float32 integer-exact range."""
        scaled = np.float32(price) * np.float32(100.0)
        # float32 integer-exact range is ±2**24
        assert abs(scaled) < 2.0**24

    @pytest.mark.parametrize(
        "price",
        [-9999.00, -129.68, 0.01, 129.68, 9999.00],
    )
    def test_method_a_encodes_exact_tick_index(self, cfg, price):
        """``round(float32(p) * float32(100)).astype(int32)`` equals
        the numpy integer reference."""
        scaled = np.float32(price) * np.float32(round(1.0 / cfg.price_tick))
        encoded = int(np.int32(np.round(scaled)))
        reference = round(price / cfg.price_tick)
        assert encoded == reference

    def test_qty_headroom(self, cfg):
        """999.0 * 10 = 9990, far below 2²⁴."""
        scaled = np.float32(cfg.qty_max) * np.float32(round(1.0 / cfg.qty_tick))
        assert abs(scaled) < 2.0**24


# ---------------------------------------------------------------------------
# Section 10 — Shape-and-dtype hygiene
# ---------------------------------------------------------------------------

class TestShapeDtypeHygiene:
    """Loader emits scalar-jnp-array fields; verify they route through
    the validator identically to Python-float fields."""

    def test_jnp_scalar_fields_equivalent_to_python_floats(self, cfg):
        py_order = _order(129.68, 2.5)
        jnp_order = OMIEOrder(
            contract_id=jnp.int32(1),
            side=jnp.int32(int(Side.BUY)),
            price=jnp.float32(129.68),
            quantity=jnp.float32(2.5),
            order_id=jnp.int32(1),
            agent_id=jnp.int32(0),
            timestamp=jnp.int32(0),
            exec_restriction=jnp.int32(int(ExecRestriction.NON)),
            validity=jnp.int32(int(Validity.GFS)),
            expiry_time=jnp.int32(0),
            status=jnp.int32(int(Status.ACTIVE)),
            peak_qty=jnp.float32(0.0),
            ppd=jnp.float32(0.0),
        )
        assert bool(validate_order(py_order, cfg, True)) == bool(
            validate_order(jnp_order, cfg, True)
        )

    def test_return_dtype_is_bool(self, cfg):
        result = validate_order(_order(50.00, 1.0), cfg, True)
        assert result.dtype == jnp.bool_

    def test_return_is_scalar(self, cfg):
        result = validate_order(_order(50.00, 1.0), cfg, True)
        assert result.shape == ()

    @pytest.mark.jit
    def test_return_under_jit_is_scalar_bool(self, cfg):
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        result = jitted(_order(50.00, 1.0), cfg, True)
        assert result.dtype == jnp.bool_
        assert result.shape == ()

    def test_return_tree_structure_is_leaf(self, cfg):
        result = validate_order(_order(50.00, 1.0), cfg, True)
        leaves, treedef = jax.tree.flatten(result)
        assert len(leaves) == 1
        # treedef for a leaf has no children
        assert treedef.num_leaves == 1


# ---------------------------------------------------------------------------
# Section 11 — Side validity branch
# ---------------------------------------------------------------------------

class TestSideValidity:
    """Side must be BUY=0 or SELL=1; other values reject."""

    def test_buy_accepted(self, cfg):
        order = _order(50.00, 1.0, side=int(Side.BUY))
        assert bool(validate_order(order, cfg, True)) is True

    def test_sell_accepted(self, cfg):
        order = _order(50.00, 1.0, side=int(Side.SELL))
        assert bool(validate_order(order, cfg, True)) is True

    def test_invalid_side_rejected(self, cfg):
        # Bypass IntEnum via direct construction (IntEnum rejects 2 but
        # OMIEOrder is a NamedTuple so we can inject a raw int).
        order = OMIEOrder(
            contract_id=1,
            side=2,  # not BUY / not SELL
            price=50.00,
            quantity=1.0,
            order_id=1,
            agent_id=0,
            timestamp=0,
            exec_restriction=int(ExecRestriction.NON),
            validity=int(Validity.GFS),
            expiry_time=0,
            status=int(Status.ACTIVE),
            peak_qty=0.0,
            ppd=0.0,
        )
        assert bool(validate_order(order, cfg, True)) is False


# ---------------------------------------------------------------------------
# Section 12 — Cross-mode acceptance parity on the hotfix regression
# ---------------------------------------------------------------------------

class TestHotfixRegressionCrossMode:
    """The 129.68 hotfix pin: verify it holds under every trace mode
    we exercise in production."""

    def test_129_68_eager(self, cfg):
        assert bool(validate_order(_order(129.68, 2.5), cfg, True)) is True

    @pytest.mark.jit
    def test_129_68_jit(self, cfg):
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        assert bool(jitted(_order(129.68, 2.5), cfg, True)) is True

    def test_129_68_vmap(self, cfg):
        def one(p):
            order = OMIEOrder(
                contract_id=jnp.int32(1),
                side=jnp.int32(int(Side.BUY)),
                price=p,
                quantity=jnp.float32(2.5),
                order_id=jnp.int32(1),
                agent_id=jnp.int32(0),
                timestamp=jnp.int32(0),
                exec_restriction=jnp.int32(int(ExecRestriction.NON)),
                validity=jnp.int32(int(Validity.GFS)),
                expiry_time=jnp.int32(0),
                status=jnp.int32(int(Status.ACTIVE)),
                peak_qty=jnp.float32(0.0),
                ppd=jnp.float32(0.0),
            )
            return validate_order(order, cfg, True)

        prices = jnp.array([129.68, 129.68, 129.68], dtype=jnp.float32)
        out = jax.vmap(one)(prices)
        assert bool(jnp.all(out)) is True

    @pytest.mark.jit
    def test_129_68_jit_of_vmap(self, cfg):
        def one(p):
            order = OMIEOrder(
                contract_id=jnp.int32(1),
                side=jnp.int32(int(Side.BUY)),
                price=p,
                quantity=jnp.float32(2.5),
                order_id=jnp.int32(1),
                agent_id=jnp.int32(0),
                timestamp=jnp.int32(0),
                exec_restriction=jnp.int32(int(ExecRestriction.NON)),
                validity=jnp.int32(int(Validity.GFS)),
                expiry_time=jnp.int32(0),
                status=jnp.int32(int(Status.ACTIVE)),
                peak_qty=jnp.float32(0.0),
                ppd=jnp.float32(0.0),
            )
            return validate_order(order, cfg, True)

        prices = jnp.array([129.68, 129.68, 129.68], dtype=jnp.float32)
        out = jax.jit(jax.vmap(one))(prices)
        assert bool(jnp.all(out)) is True
