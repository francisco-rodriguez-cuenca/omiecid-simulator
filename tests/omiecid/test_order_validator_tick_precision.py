"""Locked regression tests for P2-L1-HOTFIX-tick-precision.

Pin the fix for the float32 ULP-boundary regression in
``omiecid_simulator/omiecid/order_validator.py`` (the on-tick equality check at
lines 77-78 of the JIT-safe validator).

Story reference: ``docs/omiecid/agile-plan.md`` §``Phase 2 HOTFIX — Order
validator tick precision`` / ``P2-L1-HOTFIX-tick-precision``.

Regression summary
------------------
The P2-L1 validator checks on-tick prices via::

    price_scaled = float32(price) * float32(1.0 / tick)  # tick=0.01 -> scale=100
    price_on_tick = price_scaled == jnp.round(price_scaled)

For on-tick ``price=129.68``, ``float32(129.68) * float32(100.0)`` evaluates
to ``12967.99902344`` — one ULP below the exact integer ``12968.0`` — so the
equality test fails and a valid order is spuriously rejected. An exhaustive
sweep over the OMIE default on-tick price grid ``[-9999.00, 9999.00]`` step
``0.01`` (1,999,801 cases) shows 262,450 false rejections on HEAD. The qty
grid ``[0.1, 999.0]`` step ``0.1`` is latent-clean at OMIE defaults but the
same bug class lives on the same line.

Fix (Option C / C-variant 1-prime, from the re-adjudicated story card):
encode float price/qty to int32 ticks inside the validator via
``jnp.round(p * jnp.float32(round(1.0 / tick))).astype(jnp.int32)``
(Method A) and bounds-check in int32 space only. The on-tick check
collapses to a no-op (``jnp.round`` is the semantic tick-snap); the
forbidden-tolerance jaxpr probe stays green (only the structural
literals ``100.0`` and ``10.0`` remain). See
``docs/omiecid/sprint-status.md`` §``Second fix-mechanism re-adjudication
(Option C)`` for the architect's full brief.

Contract this file pins (story §Test-first contract, surviving items)
----------------------------------------------------------------------
1. Named regression: ``test_price_129_68_accepted`` — the canonical case.
2. Exhaustive on-tick price sweep via ``jax.vmap`` — 1,999,801 cases accept.
3. Exhaustive on-tick qty sweep via ``jax.vmap`` — 9,990 cases accept.
4. *(Migrated)* Off-tick rejection semantics — moved in inverted form to
   ``tests/omiecid/test_encoding_snap_semantics.py`` as the encoder-layer
   tick-snap contract. See the placeholder section 4 below for the
   pointer comment.
5. Jaxpr invariant re-probe — no ``1e-9`` / ``1e-8`` / ``1e-6`` tolerance
   literal appears in the post-fix validator's traced computation.

Items (6) and (7) are not in this file:

- (6) "Full P2-L1 suite still green" is a *gate*, not a new test. It is
  verified by running ``tests/omiecid/test_order_validator.py`` and
  ``tests/omiecid/test_order_validator_jit.py`` and
  ``tests/omiecid/test_order_validator_audit.py`` as part of the fast suite.
- (7) The ULP-boundary sentinel guardrail lives in
  ``tests/omiecid/test_order_validator_audit.py::TestTickAlignmentPurity`` as
  a structural audit extension (so the audit suite would have caught this
  bug had it existed). The sentinels are selected from this file's sweep
  but authored into the audit file, not here.

Item (8) — demo oracle — is NOT a pytest test:
------------------------------------------------
Per the story card, the developer must execute (at implementation time)::

    PYTHONPATH=. .venv/bin/python3 demos/generate_real_data_demo.py

and verify that the regenerated ``demos/20260112_omie_real/demo.html``
step-4 snapshot accepts order #991 (ARACA/ARACC01 SELL NON 129.68 @ 2.50 MW).
This is a developer checklist item and an architecture-audit check, not a
pytest test. Recorded here so future maintainers know where to look.
"""
from __future__ import annotations

import re
import time

import jax
import jax.numpy as jnp
import numpy as np
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
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
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


def _make_vmapped_validator(config: OMIEProductConfig):
    """Build a ``jax.jit(jax.vmap(...))`` wrapper that accepts batched
    (price, quantity) float32 arrays and returns a batched bool array.

    Non-varying OMIEOrder fields are held static-for-the-batch through
    ``jnp.broadcast_to``. The config + contract_open are passed through
    ``validate_order``'s ``static_argnums`` — via a closure so
    ``jax.vmap`` only sees the traced leaves.
    """

    def check(price_f32: jnp.ndarray, qty_f32: jnp.ndarray) -> jnp.ndarray:
        order = OMIEOrder(
            contract_id=jnp.int32(0),
            side=jnp.int32(int(Side.BUY)),
            price=price_f32,
            quantity=qty_f32,
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
        return validate_order(order, config, True)

    return jax.jit(jax.vmap(check))  # noqa: jit-marker  (helper; consuming tests carry their own pytest.mark.jit)


def _price_grid_f32() -> np.ndarray:
    """Return the OMIE default on-tick price grid as a float32 array.

    Uses integer arithmetic in float64 (``(idx - 999900) * 0.01``) then
    casts once to float32 — the exact input shape the loader produces
    (P2-F0 dtype contract).
    """
    n = int(round((9999.0 - (-9999.0)) / 0.01)) + 1  # 1,999,801
    idx = np.arange(n, dtype=np.int64)
    prices_f64 = (idx - 999900).astype(np.float64) * 0.01
    return prices_f64.astype(np.float32)


def _qty_grid_f32() -> np.ndarray:
    """Return the OMIE default on-tick quantity grid as a float32 array:
    ``[0.1, 0.2, ..., 999.0]``, 9,990 cases."""
    n = int(round((999.0 - 0.1) / 0.1)) + 1
    idx = np.arange(n, dtype=np.int64)
    qty_f64 = (idx + 1).astype(np.float64) * 0.1
    return qty_f64.astype(np.float32)


def _known_failing_prices_f32() -> np.ndarray:
    """Fallback subsample (story card §Test-first contract §2): precompute
    the set of on-tick prices for which ``float32(p) * float32(100.0)``
    slips one ULP from the exact integer. These are the 262,450 prices
    the HEAD validator spuriously rejects.

    Offered as a fallback in case the full sweep exceeds CPU budget; on
    the current machine the full sweep runs in ~50 ms so this is unused
    by the main sweep test. It is, however, exposed as a test fixture
    (``test_known_failing_prices_all_accepted``) to pin that every
    single HEAD-failing price is post-fix clean.
    """
    n = int(round((9999.0 - (-9999.0)) / 0.01)) + 1
    idx = np.arange(n, dtype=np.int64)
    prices_f64 = (idx - 999900).astype(np.float64) * 0.01
    prices_f32 = prices_f64.astype(np.float32)
    scaled = prices_f32 * np.float32(100.0)
    rounded = np.round(scaled)
    return prices_f32[scaled != rounded]


# ---------------------------------------------------------------------------
# 1. Named regression — the exact case the story card pins
# ---------------------------------------------------------------------------

class TestNamedRegression:
    """The canonical 129.68 EUR/MWh case from the 2026-01-12 demo oracle
    (order #991 — ARACA/ARACC01 SELL NON 129.68 @ 2.50 MW, step 4)."""

    def test_price_129_68_accepted(self, config):
        """Validator must accept an on-tick SELL @ 129.68 EUR/MWh, 2.50 MW.

        Fails on HEAD (``float32(129.68) * 100.0 = 12967.99902344`` !=
        ``12968.0``). Passes post-fix via the float32 round-trip.
        """
        order = _valid_order(side=Side.SELL, price=129.68, quantity=2.5)
        assert bool(validate_order(order, config, True)), (
            "on-tick price 129.68 EUR/MWh spuriously rejected — "
            "float32 ULP regression in order_validator.py:77-78"
        )

    def test_price_129_68_accepted_buy_side(self, config):
        """Mirror of the SELL case for the BUY validation path.

        The bug is side-independent but we pin both paths explicitly so a
        future sided-rewrite cannot re-break one side silently.
        """
        order = _valid_order(side=Side.BUY, price=129.68, quantity=2.5)
        assert bool(validate_order(order, config, True))

    @pytest.mark.jit
    def test_price_129_68_accepted_under_jit(self, config):
        """Pin the same acceptance under ``jax.jit`` (static_argnums for
        config + contract_open) — this is the path the live environment
        exercises, not the eager path.
        """
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        order = _valid_order(side=Side.SELL, price=129.68, quantity=2.5)
        assert bool(jitted(order, config, True))


# ---------------------------------------------------------------------------
# 2. Exhaustive on-tick price sweep
# ---------------------------------------------------------------------------

class TestExhaustivePriceSweep:
    """All 1,999,801 on-tick prices in ``[-9999.00, 9999.00]`` step ``0.01``
    must be accepted. Batched via ``jax.vmap`` + ``jnp.all`` reduce.

    Budget: the full sweep runs in ~50 ms cold on CPU (well under the 2 s
    target from the story card); no subsampling fallback is needed.
    """

    @pytest.mark.jit
    def test_full_on_tick_price_grid_accepted(self, config):
        vcheck = _make_vmapped_validator(config)
        prices = jnp.asarray(_price_grid_f32())
        qty = jnp.ones_like(prices)  # 1.0 MW — on-tick, in-range
        t0 = time.time()
        results = vcheck(prices, qty)
        results.block_until_ready()
        elapsed = time.time() - t0
        rejections = int(jnp.sum(~results))
        assert rejections == 0, (
            f"{rejections} on-tick prices rejected by validator "
            f"(HEAD baseline: 262450 rejections). Elapsed: {elapsed:.3f} s"
        )
        # Hard runtime ceiling per the story card (4x margin over observed).
        assert elapsed < 8.0, (
            f"Sweep exceeded 8 s budget ({elapsed:.3f} s); investigate "
            "compile-time regression before raising the ceiling."
        )

    @pytest.mark.jit
    def test_known_failing_prices_all_accepted(self, config):
        """Every price that HEAD spuriously rejects (262,450 of them) must
        post-fix be accepted. This is the tightest possible regression
        pin: if *any* HEAD-failing price remains rejected, this fires.
        """
        vcheck = _make_vmapped_validator(config)
        failing = _known_failing_prices_f32()
        # Sanity: the precompute should still identify the expected count;
        # if the float32 arithmetic of the test machine's JAX backend
        # differs from numpy's, the story's empirical 262,450 count may
        # drift — pin it so any divergence is loud.
        assert failing.size == 262450, (
            f"known-failing precompute produced {failing.size} cases; "
            "expected 262450 per the story card (2026-04-23 sweep)"
        )
        prices = jnp.asarray(failing)
        qty = jnp.ones_like(prices)
        results = vcheck(prices, qty)
        assert bool(jnp.all(results)), (
            "Some of the 262450 HEAD-failing prices still rejected post-fix"
        )


# ---------------------------------------------------------------------------
# 3. Exhaustive on-tick qty sweep
# ---------------------------------------------------------------------------

class TestExhaustiveQtySweep:
    """All 9,990 on-tick quantities in ``[0.1, 999.0]`` step ``0.1`` must
    be accepted. Latent-clean at HEAD (qty grid does not hit the ULP
    boundary at OMIE default ``qty_tick=0.1``, per the story card), but
    pinned here so any regression-class re-introduction fires."""

    @pytest.mark.jit
    def test_full_on_tick_qty_grid_accepted(self, config):
        vcheck = _make_vmapped_validator(config)
        qty = jnp.asarray(_qty_grid_f32())
        # Use a known-clean price (50.0 is exact in float32) so rejection
        # can only come from the qty check.
        price = jnp.full_like(qty, 50.0)
        results = vcheck(price, qty)
        rejections = int(jnp.sum(~results))
        assert rejections == 0, (
            f"{rejections} on-tick quantities rejected; qty check has "
            "re-acquired a ULP-boundary regression"
        )


# ---------------------------------------------------------------------------
# 4. Off-tick rejection — MIGRATED to tests/omiecid/test_encoding_snap_semantics.py
# ---------------------------------------------------------------------------
#
# Under P2-L1-HOTFIX-tick-precision (Option C / C-variant 1-prime), the
# validator no longer performs an independent on-tick check. Sub-half-tick
# float offsets are snapped to the nearest tick by the encoder's
# ``jnp.round``; the resulting int32 is what the validator range-checks.
# The semantic content of the old ``TestOffTickRejection`` class (that
# off-tick floats are not silently treated as valid on-tick ticks) has
# been inverted and moved to ``tests/omiecid/test_encoding_snap_semantics.py``:
# that file asserts the positive contract — the encoder snaps sub-half-tick
# offsets onto the nearest tick, produces distinct ints for offsets
# ≥ half-tick, and round-trips on-tick inputs cleanly.
#
# The on-tick acceptance sweeps above (sections 1-3) and the jaxpr
# invariant below (section 5) are the surviving hotfix contract at the
# validator layer.


# ---------------------------------------------------------------------------
# 5. Jaxpr invariant re-probe — post-fix validator must stay tolerance-free
# ---------------------------------------------------------------------------

class TestJaxprNoToleranceLiteral:
    """Mirrors ``test_order_validator_audit.py::TestNoFloatToleranceLeak::
    test_jaxpr_has_no_tolerance_literal`` but is pinned as a post-fix
    contract here too. The fix (Option H, round-trip divide) introduces
    one additional ``div`` op in the jaxpr; no new numeric literals
    should appear in the forbidden range.
    """

    # NOTE: intentionally self-contained — we do not import the audit
    # helper, because this file must stay readable as a single regression
    # pin even if the audit file is reorganised.

    def test_no_forbidden_tolerance_in_jaxpr(self, config):
        order = _valid_order()
        jaxpr = jax.make_jaxpr(validate_order, static_argnums=(1, 2))(
            order, config, True
        )
        text = str(jaxpr)
        forbidden = [
            "1e-09", "1e-9", "1.e-9", "1.0e-09",
            "1e-08", "1e-8",
            "1e-06", "1e-6",
        ]
        for pat in forbidden:
            assert pat not in text, (
                f"post-fix validator jaxpr contains forbidden tolerance "
                f"literal {pat!r}"
            )
        assert re.search(r"\b\d+(\.\d+)?e-0?[6-9]\b", text) is None, (
            "post-fix validator jaxpr contains a suspicious small-exponent literal"
        )

    @pytest.mark.jit
    def test_no_forbidden_tolerance_under_vmap(self, config):
        """Also probe the vmapped path, since that's the one the sweep
        tests exercise and it is a different jaxpr."""
        vcheck = _make_vmapped_validator(config)
        prices = jnp.asarray(np.array([50.0, 129.68], dtype=np.float32))
        qty = jnp.ones_like(prices)
        jaxpr = jax.make_jaxpr(vcheck)(prices, qty)
        text = str(jaxpr)
        forbidden = [
            "1e-09", "1e-9", "1.e-9", "1.0e-09",
            "1e-08", "1e-8",
            "1e-06", "1e-6",
        ]
        for pat in forbidden:
            assert pat not in text, (
                f"post-fix vmapped validator jaxpr contains forbidden "
                f"tolerance literal {pat!r}"
            )
        assert re.search(r"\b\d+(\.\d+)?e-0?[6-9]\b", text) is None
