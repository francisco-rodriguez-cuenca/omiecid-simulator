"""AUDIT tests for P3-PERF-REGEN-QUIESCENCE-1 — cross-slot quiescence re-arm.

Independent post-implementation verification (audit mode, orchestrate.md
Step 5) of the OQ-23 closure in ``IcebergManager.regenerate_slices``:
after ``regenerate_slices`` (and therefore after every ``submit_order``)
returns, NO active iceberg slot in ``meta[:n_active]`` may be left with
a consumed visible peak (``get_order(...)[2] <= 0``) — including the
CROSS-slot case where a higher-index slot's regen slice partially rests
while consuming an already-visited lower-index slot's peak. This
post-condition is what ``contract_book.py``'s ``no_fill_path``
(P2-PERF-8 regen-skip) relies on between aggressors, and what the
``matchable_hidden_volume`` FOK pre-check needs to report correctly
(open-questions.md §OQ-23; agile-plan.md §P3-PERF-REGEN-QUIESCENCE-1).

Derived INDEPENDENTLY from the sprint requirements (the original locked
test file was NOT read, per audit-mode isolation).

Constructor note (market-spec §13.6): the OQ-23 probe's primary
constructor (ICEwPI SELL with ppd = -0.20) violates the HARD sign rule
(sell PPD must be > 0; buy PPD < 0), so all gap shapes here use VALID
signs. The valid-sign gap family used throughout: a flat-PPD
mutual-crossing ICE pair with price separation and ``slice_j > peak_k``
partial-rest sizing —

    k: ICE BUY  @ 60.05, total 2.0 MW, peak 0.4 MW  (slot 0, registered first)
    j: ICE SELL @ 60.00, total 1.2 MW, peak 0.6 MW  (slot 1)

  j's registration peak (0.6 @ 60.00) crosses k's bid (0.4 @ 60.05):
  trades 0.4, rests 0.2. Inside the post-dispatch ``regenerate_slices``
  pass 1: k regenerates (0.4 @ 60.05), consumes j's 0.2 ask and PARTIALLY
  RESTS 0.2 (k's own OID rests -> no self-dirty); then j regenerates
  (0.6 @ 60.00), consumes k's freshly-rested 0.2 peak and PARTIALLY RESTS
  0.4 (no self-dirty). Pass 1 ends with slot 0 (k, already visited)
  ``peak_consumed=True`` and ``dirty=False`` on the self-OID channel —
  the exact OQ-23 shape the cross-slot re-arm must catch. When the
  re-arm works, the cascade drains fully: j sells its whole 1.2 MW,
  is cleaned up, and k ends with visible peak 0.4 MW + hidden 0.4 MW
  (meta: remaining_hidden=4 ticks, slice_count=3). Every constructor is
  validated against ``validate_order`` (on-tick, OQ-01 envelopes,
  positive OIDs, in-window OPEN contract).

Rule-4 framing: a leftover consumed-peak slot is an ENGINEERING-invariant
violation (OQ-23; NOT a §4.2.1 Rule 4 breach — the resting book stays
uncrossed), so these tests also assert the uncrossed-book observable
separately.

These tests assert the CORRECT (spec/story-contracted) behavior. A
failure against the shipped implementation is a real finding and must be
reported, not weakened away.

Device policy: CPU-hermetic (conftest pins the cpu platform).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob.jorderbook import OrderBook
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    _build_state_jit,
    _make_jaxob_cfg,
    _submit_order,
)
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_PEAK_QTY,
    COL_REMAINING_HIDDEN,
    COL_SIDE,
    COL_SLICE_COUNT,
    IcebergManager,
)
from omiecid_simulator.omiecid.jax_constants import MAX_SLICES_PER_REGEN
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.order_validator import validate_order

# The file jit-compiles ``regenerate_slices`` at explicit ``max_iterations``
# values (cap-backstop probes) and drives the module-level jitted
# ``_submit_order`` — T1.1 jit-marker discipline applies file-wide.
pytestmark = [pytest.mark.jit]


# ---------------------------------------------------------------------------
# Shared construction / probe helpers (derived independently from the
# iceberg_meta schema + the jaxob get_order row layout
# [price, qty, oid, tid, sec, nsec]; NOT copied from any locked test).
# ---------------------------------------------------------------------------

_CONTRACT_ID = 37


def _open_contract() -> Contract:
    return Contract(
        contract_id=_CONTRACT_ID,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


@pytest.fixture(scope="module")
def audit_book(product_config, market_config):
    return ContractBook(_open_contract(), product_config, market_config)


@pytest.fixture(scope="module")
def audit_ob(market_config):
    return OrderBook(cfg=_make_jaxob_cfg(market_config))


def _mk_order(pc, side, price, qty, oid, ts,
              er=ExecRestriction.NON, peak=0.0, ppd=0.0):
    """Build an OMIE order and REQUIRE it to pass ``validate_order``.

    §13.6 sign rule is enforced here as a fixture precondition (the
    validator does not check PPD signs): buy PPD <= 0, sell PPD >= 0.
    """
    if ppd != 0.0:
        if side == int(Side.BUY):
            assert ppd < 0.0, "§13.6: ICEwPI BUY requires ppd < 0"
        else:
            assert ppd > 0.0, "§13.6: ICEwPI SELL requires ppd > 0"
    order = make_order(
        contract_id=_CONTRACT_ID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=oid, timestamp=ts,
        exec_restriction=er, peak_qty=peak, ppd=ppd,
    )
    assert bool(validate_order(order, pc, True)), (
        f"fixture bug: constructor failed validate_order: {order}"
    )
    return order


def _consumed_active_slots(state, ob) -> list:
    """The OQ-23 quiescence probe, re-derived from first principles.

    For every packed-prefix slot i in [0, n_active): look the slot's OID
    up on its own resting side (meta side 0=BUY -> jaxob lookup 1=bids)
    and flag the slot when the OID is absent (``row[2] <= 0`` on the
    [price, qty, oid, ...] row) — i.e. the visible peak is consumed while
    the slot is still active. Post-condition: this list is EMPTY after
    every ``regenerate_slices`` / ``submit_order`` return.
    """
    meta = np.asarray(state.iceberg_meta)
    n_active = int(state.n_active)
    violations = []
    for i in range(n_active):
        oid = int(meta[i, COL_ORDER_ID])
        side = int(meta[i, COL_SIDE])
        row = np.asarray(
            ob.get_order(state.lob, jnp.int32(1 - side), jnp.int32(oid))
        )
        if int(row[2]) <= 0:
            violations.append(
                (i, oid,
                 f"remaining_hidden={int(meta[i, COL_REMAINING_HIDDEN])}",
                 f"slice_count={int(meta[i, COL_SLICE_COUNT])}")
            )
    return violations


def _visible_qty_ticks(state, side, oid, ob) -> int:
    """Resting visible quantity (in qty ticks) of ``oid`` on its own side."""
    row = np.asarray(
        ob.get_order(state.lob, jnp.int32(1 - side), jnp.int32(oid))
    )
    return int(row[1]) if int(row[2]) > 0 else 0


def _book_is_uncrossed(state) -> bool:
    """Rule 4's binding observable: best bid strictly below best ask."""
    bids = np.asarray(state.lob.bids)
    asks = np.asarray(state.lob.asks)
    pb = bids[bids[:, 2] > 0][:, 0]
    pa = asks[asks[:, 2] > 0][:, 0]
    if pb.size == 0 or pa.size == 0:
        return True
    return int(pb.max()) < int(pa.min())


def _submit_gap_pair(book, pc, oid_k, oid_j, *, j_ppd=0.0,
                     k_total=2.0, j_total=1.2):
    """Submit the valid-sign OQ-23 gap pair (module docstring) and return
    the post-pair state. ``j_ppd`` optionally makes j an ICEwPI SELL
    (must be >= 0 per §13.6; +0.05 keeps the drifted slice crossing the
    60.05 bid)."""
    state = book.init()
    state, r_k = book.submit_order(
        state,
        _mk_order(pc, int(Side.BUY), 60.05, k_total, oid_k, 1,
                  ExecRestriction.ICE, peak=0.4),
    )
    assert bool(r_k.accepted)
    state, r_j = book.submit_order(
        state,
        _mk_order(pc, int(Side.SELL), 60.00, j_total, oid_j, 2,
                  ExecRestriction.ICE, peak=0.6, ppd=j_ppd),
    )
    assert bool(r_j.accepted)
    return state


# ---------------------------------------------------------------------------
# A. Gap closure on valid-sign constructors (the behavior change itself)
# ---------------------------------------------------------------------------


class TestOQ23GapClosureValidSignConstructors:
    """The post-condition must HOLD on the §13.6-valid gap family.

    Expected drained end-state for the (2.0/0.4 BUY vs 1.2/0.6 SELL)
    pair, derived by hand and confirmed against the fully-drained
    reference cascade: j sells all 1.2 MW and is cleaned up
    (n_active == 1); k ends with visible peak 0.4 MW (4 ticks) resting
    at 60.05, remaining_hidden == 4 ticks, slice_count == 3.
    """

    def test_gap_closed_primary_flat_ppd_constructor(
        self, audit_book, audit_ob, product_config,
    ):
        """Primary valid-sign gap shape (realistic order ids)."""
        state = _submit_gap_pair(audit_book, product_config, 101, 102)
        violations = _consumed_active_slots(state, audit_ob)
        assert violations == [], (
            "OQ-23 post-condition VIOLATED after submit_order returned: "
            f"active slots left peak_consumed=True: {violations}. The "
            "cross-slot re-arm (P3-PERF-REGEN-QUIESCENCE-1) did not fire "
            "on the valid-sign gap constructor."
        )
        assert _book_is_uncrossed(state)
        # Exact drained end-state.
        meta = np.asarray(state.iceberg_meta)
        assert int(state.n_active) == 1, (
            f"j (oid 102) must be fully drained and cleaned up; "
            f"n_active={int(state.n_active)}, meta[:2]={meta[:2].tolist()}"
        )
        assert int(meta[0, COL_ORDER_ID]) == 101
        assert int(meta[0, COL_REMAINING_HIDDEN]) == 4  # 0.4 MW hidden
        assert int(meta[0, COL_SLICE_COUNT]) == 3
        assert _visible_qty_ticks(state, int(Side.BUY), 101, audit_ob) == 4

    def test_gap_closed_mirrored_arrangement(
        self, audit_book, audit_ob, product_config,
    ):
        """Mirrored registration order: the SELL iceberg 201 rests first
        (slot 0). Since OMIE-ICE-INCOMING-HOP-1 the incoming ICE BUY 202
        sweeps its total before registering: it takes 201 whole in two
        trades of 0.6 @ 60.00 (passive 201: the visible peak, then the
        regenerated slice) and its remainder 0.8 registers showing 0.4
        with 0.4 hidden and slice_count 0. This arrangement therefore no
        longer reaches the OQ-23 cross-slot gap (the Step-2 vacuity
        finding); it stays as a quiescence / post-condition guard."""
        state = audit_book.init()
        state, _ = audit_book.submit_order(
            state,
            _mk_order(product_config, int(Side.SELL), 60.00, 1.2, 201, 1,
                      ExecRestriction.ICE, peak=0.6),
        )
        # SELL rests into an empty book — sanity: nothing consumed yet.
        assert _consumed_active_slots(state, audit_ob) == []
        state, _ = audit_book.submit_order(
            state,
            _mk_order(product_config, int(Side.BUY), 60.05, 2.0, 202, 2,
                      ExecRestriction.ICE, peak=0.4),
        )
        violations = _consumed_active_slots(state, audit_ob)
        assert violations == [], (
            "OQ-23 post-condition VIOLATED on the mirrored arrangement: "
            f"{violations}"
        )
        assert _book_is_uncrossed(state)
        meta = np.asarray(state.iceberg_meta)
        assert int(state.n_active) == 1
        assert int(meta[0, COL_ORDER_ID]) == 202  # BUY survivor
        assert int(meta[0, COL_REMAINING_HIDDEN]) == 4
        assert int(meta[0, COL_SLICE_COUNT]) == 0
        assert _visible_qty_ticks(state, int(Side.BUY), 202, audit_ob) == 4

    def test_gap_closed_icewpi_valid_sign_variant(
        self, audit_book, audit_ob, product_config,
    ):
        """ICEwPI j with a VALID sign (SELL, ppd=+0.05): the drifted regen
        slice (60.00 + 0.05 = 60.05) still crosses k's 60.05 bid, so the
        gap shape fires with a genuine ICEwPI participant — unlike the
        OQ-23 probe's forbidden negative-ppd SELL."""
        state = _submit_gap_pair(
            audit_book, product_config, 301, 302, j_ppd=0.05,
        )
        violations = _consumed_active_slots(state, audit_ob)
        assert violations == [], (
            "OQ-23 post-condition VIOLATED on the valid-sign ICEwPI "
            f"variant: {violations}"
        )
        assert _book_is_uncrossed(state)
        meta = np.asarray(state.iceberg_meta)
        assert int(state.n_active) == 1
        assert int(meta[0, COL_ORDER_ID]) == 301
        assert int(meta[0, COL_REMAINING_HIDDEN]) == 4
        assert int(meta[0, COL_SLICE_COUNT]) == 3

    def test_gap_closure_independent_of_order_id_magnitude(
        self, audit_book, audit_ob, product_config,
    ):
        """DIAGNOSTIC PIN: order ids are opaque labels — matching and
        quiescence outcomes must be identical for tiny ids (1, 2) and
        realistic ids (1001, 1002) on the byte-wise-identical order flow.

        Rationale: any predicate inside ``regenerate_slices`` that
        accidentally compares the jaxob row's ORDER-ID column
        (``row[2]``) against a quantity (instead of the qty column
        ``row[1]``) would make the re-arm fire only when
        ``oid < slice_qty_ticks`` — an id-magnitude dependence that this
        A/B pins out.
        """
        outcomes = {}
        for label, (oid_k, oid_j) in (
            ("tiny", (1, 2)), ("realistic", (1001, 1002)),
        ):
            state = _submit_gap_pair(audit_book, product_config, oid_k, oid_j)
            meta = np.asarray(state.iceberg_meta)
            outcomes[label] = {
                "violations": [
                    v[:1] for v in _consumed_active_slots(state, audit_ob)
                ],
                "n_active": int(state.n_active),
                "k_hidden": int(meta[0, COL_REMAINING_HIDDEN]),
                "k_slice_count": int(meta[0, COL_SLICE_COUNT]),
                "k_visible": _visible_qty_ticks(
                    state, int(Side.BUY), oid_k, audit_ob,
                ),
            }
        assert outcomes["tiny"] == outcomes["realistic"], (
            "Quiescence outcome depends on ORDER-ID MAGNITUDE — the "
            "re-arm predicate is reading the wrong get_order column "
            "(oid col 2 vs qty col 1). Outcomes:\n"
            f"  tiny ids:      {outcomes['tiny']}\n"
            f"  realistic ids: {outcomes['realistic']}"
        )


# ---------------------------------------------------------------------------
# B. The no_fill_path reliance made safe (the downstream hazard)
# ---------------------------------------------------------------------------


class TestNoFillPathRelianceSafe:
    def test_hidden_liquidity_visible_across_no_fill_aggressor(
        self, audit_book, audit_ob, product_config,
    ):
        """The P2-PERF-8 ``no_fill_path`` regen-skip relies on the
        post-condition: consumed hidden liquidity must be re-exposed
        BEFORE the next aggressor, because a no-fill aggressor skips
        regen entirely. Gap-shape submit -> non-crossing aggressor
        (no-fill; regen skipped) -> crossing aggressor MUST trade
        against the re-exposed 0.4 MW peak at 60.05."""
        pc = product_config
        state = _submit_gap_pair(audit_book, pc, 401, 402)
        # Non-crossing SELL @ 61.00 — rests; no_fill_path skips regen.
        state, r_nf = audit_book.submit_order(
            state, _mk_order(pc, int(Side.SELL), 61.00, 0.5, 403, 3),
        )
        assert not bool(r_nf.traded)
        assert _consumed_active_slots(state, audit_ob) == [], (
            "a consumed-peak slot survived ACROSS a no-fill aggressor — "
            "the exact OQ-23 downstream hazard (no_fill_path skipped the "
            "regen that would have re-exposed it)"
        )
        # Crossing SELL @ 60.05 for exactly the visible peak.
        state, r_x = audit_book.submit_order(
            state, _mk_order(pc, int(Side.SELL), 60.05, 0.4, 404, 4),
        )
        assert bool(r_x.traded), (
            "crossing aggressor found NO liquidity at 60.05 — the "
            "iceberg's re-exposed peak was not resting (mis-exposed "
            "liquidity across aggressors, OQ-23 hazard live)"
        )
        assert float(r_x.trade_qty) == pytest.approx(0.4, rel=1e-5)
        assert float(r_x.trade_price) == pytest.approx(60.05, rel=1e-5)
        assert _consumed_active_slots(state, audit_ob) == []


# ---------------------------------------------------------------------------
# C. FOK matchable_hidden_volume correctness on a (formerly) gap shape
# ---------------------------------------------------------------------------


class TestFOKPreCheckOnGapShape:
    def test_fok_fills_in_full_against_re_exposed_peak_plus_hidden(
        self, audit_book, audit_ob, product_config,
    ):
        """After the gap pair, k holds visible 0.4 MW + hidden 0.4 MW at
        60.05. An FOK SELL @ 60.05 for 0.8 MW is exactly feasible and
        must fill IN FULL (all-or-none). If the peak were left consumed
        (gap open), the pre-check would count hidden-only volume and the
        fill loop would find no resting bid — a feasibility mis-report."""
        pc = product_config
        state = _submit_gap_pair(audit_book, pc, 501, 502)
        state_f, r_f = audit_book.submit_order(
            state,
            _mk_order(pc, int(Side.SELL), 60.05, 0.8, 503, 5,
                      ExecRestriction.FOK),
        )
        assert bool(r_f.traded) and (
            float(r_f.trade_qty) == pytest.approx(0.8, rel=1e-5)
        ), (
            "FOK SELL 0.8 @ 60.05 must fill in full against the "
            "re-exposed visible peak (0.4) + hidden (0.4); got "
            f"traded={bool(r_f.traded)}, qty={float(r_f.trade_qty)} — "
            "the matchable-volume pre-check / book state mis-reported "
            "the (formerly) gap-consumed liquidity"
        )
        # k fully drained by the FOK -> slot cleaned.
        assert int(state_f.n_active) == 0
        assert _consumed_active_slots(state_f, audit_ob) == []

    def test_fok_above_available_kills_and_leaves_book_untouched(
        self, audit_book, audit_ob, product_config,
    ):
        """FOK SELL 0.9 MW > the 0.8 MW available: must not trade, and the
        resting book must be unchanged (visible peak still 0.4 MW)."""
        pc = product_config
        state = _submit_gap_pair(audit_book, pc, 601, 602)
        state_g, r_g = audit_book.submit_order(
            state,
            _mk_order(pc, int(Side.SELL), 60.05, 0.9, 603, 5,
                      ExecRestriction.FOK),
        )
        assert not bool(r_g.traded)
        assert float(r_g.trade_qty) == pytest.approx(0.0, abs=1e-9)
        assert _visible_qty_ticks(state_g, int(Side.BUY), 601, audit_ob) == 4
        assert _consumed_active_slots(state_g, audit_ob) == []


# ---------------------------------------------------------------------------
# D. The post-condition as a UNIVERSAL invariant (seeded randomized family)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPostConditionUniversalInvariant:
    """slow: ~70 jitted submit_order dispatches + a per-submit eager
    quiescence probe across 6 seeded scenarios (well over the 1 s
    fast-suite budget in aggregate)."""

    @staticmethod
    def _random_order(pc, rng, oid, ts):
        """One random valid OMIE order. All values on-tick, inside the
        OQ-01 envelopes; ICEwPI signs per §13.6; iceberg worst-case
        slice counts kept <= 12 (far below MAX_SLICES_PER_REGEN)."""
        side = int(rng.integers(0, 2))
        price = round(60.00 + 0.05 * int(rng.integers(-4, 5)), 2)
        qty = round(0.1 * int(rng.integers(1, 11)), 1)  # 0.1 .. 1.0 MW
        kind = rng.choice(
            ["NON", "ICE", "ICEWPI", "IOC", "FOK"],
            p=[0.35, 0.30, 0.15, 0.10, 0.10],
        )
        peak, ppd, er = 0.0, 0.0, ExecRestriction.NON
        if kind in ("ICE", "ICEWPI"):
            er = ExecRestriction.ICE
            # peak in [max(0.1, qty/4), qty], on the 0.1 grid.
            min_peak_ticks = max(1, int(round(qty * 10)) // 4)
            peak = round(
                0.1 * int(rng.integers(min_peak_ticks,
                                       int(round(qty * 10)) + 1)),
                1,
            )
            if kind == "ICEWPI":
                mag = 0.05 * int(rng.integers(1, 4))
                ppd = -mag if side == int(Side.BUY) else mag
        elif kind == "IOC":
            er = ExecRestriction.IOC
        elif kind == "FOK":
            er = ExecRestriction.FOK
        return _mk_order(pc, side, price, qty, oid, ts, er, peak, ppd)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_no_consumed_active_slot_after_any_submit(
        self, seed, audit_book, audit_ob, product_config,
    ):
        """After EVERY ``submit_order`` in a mixed random-but-seeded order
        flow (both sides; NON/ICE/ICEwPI/IOC/FOK; overlapping prices;
        realistic order ids), no active slot may be left with a consumed
        visible peak, and the resting book must be uncrossed."""
        rng = np.random.default_rng(seed)
        state = audit_book.init()
        for step in range(12):
            oid = 100 + 100 * seed + step
            order = self._random_order(product_config, rng, oid, step + 1)
            state, result = audit_book.submit_order(state, order)
            assert bool(result.accepted), (
                f"seed={seed} step={step}: fixture bug — valid order "
                f"rejected: {order}"
            )
            violations = _consumed_active_slots(state, audit_ob)
            assert violations == [], (
                f"OQ-23 post-condition VIOLATED (seed={seed}, step={step}) "
                f"after submitting {order.exec_restriction=} "
                f"side={order.side} px={order.price} qty={order.quantity} "
                f"peak={order.peak_qty} ppd={order.ppd} oid={oid}: "
                f"consumed active slots {violations}"
            )
            assert _book_is_uncrossed(state), (
                f"resting book CROSSED (seed={seed}, step={step}) — a "
                "genuine Rule 4 breach, stronger than the OQ-23 "
                "engineering invariant"
            )


# ---------------------------------------------------------------------------
# E. Termination and the loud cap backstop
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTerminationAndCapBackstop:
    """slow: multi-pass cascade through the jitted submit_order plus a
    direct ``regenerate_slices`` compile at two ``max_iterations``
    values."""

    def test_deep_cascade_with_rearm_terminates_quiescent(
        self, audit_book, audit_ob, product_config,
    ):
        """A deep mutual gap-family cascade (k 8.0/0.4 BUY vs j 7.2/0.6
        SELL: 1.2 MW traded per ~3 passes, ~30 passes total) must
        terminate well inside the default caps, end quiescent, and NOT
        trip the truncation backstop (worst-case slice counts 20 and 12,
        far below MAX_SLICES_PER_REGEN=64).

        Small order ids (1, 2) are used deliberately so the cascade
        re-arm path is exercised end-to-end regardless of the
        id-magnitude finding pinned in
        ``test_gap_closure_independent_of_order_id_magnitude``.
        """
        pc = product_config
        t0 = time.monotonic()
        state = _submit_gap_pair(
            audit_book, pc, 1, 2, k_total=8.0, j_total=7.2,
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 120.0, (
            f"deep cascade did not complete promptly ({elapsed:.1f}s) — "
            "the re-arm must not defeat bounded termination"
        )
        assert _consumed_active_slots(state, audit_ob) == []
        assert not bool(state.truncated)
        meta = np.asarray(state.iceberg_meta)
        assert int(state.n_active) == 1
        assert int(meta[0, COL_ORDER_ID]) == 1
        # j sold 7.2 in full; k = 8.0 - 7.2 = 0.8 left: 0.4 visible + 0.4 hidden.
        assert int(meta[0, COL_REMAINING_HIDDEN]) == 4
        assert int(meta[0, COL_SLICE_COUNT]) == 18
        assert _visible_qty_ticks(state, int(Side.BUY), 1, audit_ob) == 4

    def test_cap_bound_exit_is_loud_and_default_cap_drains(
        self, product_config, market_config, audit_ob,
    ):
        """The ONLY permitted non-quiescent exit is the ``max_iterations``
        cap-trip, and it must be LOUD: build the cross-slot gap-entry
        state directly (k BUY 26.8/0.4 @60.05 registered + peak rested;
        j SELL 1.2/0.6 @60.00 registered + peak dispatched, consuming
        k's peak and resting 0.2), then

          * ``max_iterations=1``: only the priming pass runs; the
            re-armed dirty verdict cannot fire another pass -> a consumed
            slot REMAINS and the Path B truncation backstop surfaces
            ``truncated=True`` (k's worst-case slice count 66 > 64);
          * default ``max_iterations``: the same entry state drains to
            quiescence (the re-arm's extra passes are bounded by, and do
            not defeat, the cap).
        """
        pc, mc = product_config, market_config
        ob = audit_ob
        ice_mgr = IcebergManager(ob, pc)
        lob = ob.init()
        meta = IcebergManager.empty_meta(mc.max_icebergs)
        n_active = jnp.int32(0)

        k_order = _mk_order(pc, int(Side.BUY), 60.05, 26.8, 1, 1,
                            ExecRestriction.ICE, peak=0.4)
        assert (26.8 / 0.4) > MAX_SLICES_PER_REGEN  # 67 slices worst-case
        meta, n_active, _ = ice_mgr.register(meta, n_active, k_order)
        lob = ob.process_order_array(lob, jnp.array(
            [1, 1, 4, int(ice_mgr._encode_price_int(60.05)), 1, 1, 1, 0],
            dtype=jnp.int32,
        ))
        j_order = _mk_order(pc, int(Side.SELL), 60.00, 1.2, 2, 2,
                            ExecRestriction.ICE, peak=0.6)
        meta, n_active, _ = ice_mgr.register(meta, n_active, j_order)
        lob = ob.process_order_array(lob, jnp.array(
            [1, -1, 6, int(ice_mgr._encode_price_int(60.00)), 2, 2, 2, 0],
            dtype=jnp.int32,
        ))
        ts = jnp.int32(3)

        def probe(lob_, meta_, n_):
            fake_state = type("S", (), {})()
            fake_state.iceberg_meta = meta_
            fake_state.n_active = n_
            fake_state.lob = lob_
            return _consumed_active_slots(fake_state, ob)

        assert probe(lob, meta, n_active), (
            "fixture bug: entry state has no consumed slot to chase"
        )

        regen = jax.jit(
            ice_mgr.regenerate_slices, static_argnames=("max_iterations",),
        )
        # Capped: terminates, leaves the consumed slot, and is LOUD.
        lob_c, meta_c, na_c, ts_c = regen(
            lob, meta, n_active, ts, max_iterations=1,
        )
        capped_leftover = probe(lob_c, meta_c, na_c)
        assert capped_leftover, (
            "with max_iterations=1 the cross-slot cascade cannot drain in "
            "the single priming pass; a consumed slot must remain"
        )
        state_c = _build_state_jit(
            lob_c, meta_c, na_c, ts_c, jnp.bool_(False), ice_mgr,
            overflow=jnp.bool_(False),
        )
        assert bool(state_c.truncated), (
            "cap-bound non-quiescent exit was SILENT — the Path B "
            "truncation backstop must surface truncated=True"
        )
        # Default cap: the same entry state drains to quiescence.
        lob_d, meta_d, na_d, _ts_d = regen(
            lob, meta, n_active, ts,
            max_iterations=mc.max_match_iterations,
        )
        drained_leftover = probe(lob_d, meta_d, na_d)
        assert drained_leftover == [], (
            "the default-cap run must reach quiescence on the cross-slot "
            f"gap-entry state; leftover consumed slots: {drained_leftover}"
        )


# ---------------------------------------------------------------------------
# F. One-compile invariant across gap variants
# ---------------------------------------------------------------------------


class TestOneCompileInvariant:
    def test_no_new_submit_order_cache_entry_across_gap_variants(
        self, audit_book, product_config,
    ):
        """Warm-first cache-delta pin: after one warm ``submit_order``
        call, replaying every gap-family variant (flat-ppd pair,
        mirrored, valid-sign ICEwPI, FOK drain, no-fill) must add ZERO
        entries to the module-level jitted ``_submit_order`` cache — the
        re-arm is data-dependent control flow inside the one compiled
        artifact, not a new trace."""
        if not hasattr(_submit_order, "_cache_size"):
            pytest.skip("_cache_size probe unavailable on this JAX version")
        pc = product_config
        # Warm with a plain NON submit under the same static configs.
        state = audit_book.init()
        audit_book.submit_order(
            state, _mk_order(pc, int(Side.BUY), 60.00, 0.5, 701, 1),
        )
        warm_size = _submit_order._cache_size()

        _submit_gap_pair(audit_book, pc, 711, 712)
        _submit_gap_pair(audit_book, pc, 721, 722, j_ppd=0.05)
        state = _submit_gap_pair(audit_book, pc, 731, 732)
        state, _ = audit_book.submit_order(
            state, _mk_order(pc, int(Side.SELL), 61.00, 0.5, 733, 3),
        )
        audit_book.submit_order(
            state,
            _mk_order(pc, int(Side.SELL), 60.05, 0.8, 734, 4,
                      ExecRestriction.FOK),
        )

        delta = _submit_order._cache_size() - warm_size
        assert delta == 0, (
            f"gap-family variants added {delta} new _submit_order cache "
            "entries after warm-up — the one-compile invariant is broken"
        )
