"""OMIE-CERT-INVARIANTS-1 — Step-5 fresh-context audit of the invariant battery.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-CERT-INVARIANTS-1" (V1-V16,
incl. the v5.186 amendments). Written without reading the locked test files.

What this file hunts (plan §"Rules the first attempt taught"):

* checks that cannot fail -- every battery statement gets an **independent**
  broken engine run (trade rows / book arrays rewritten consistently in every
  snapshot, so RB still sees an internally consistent record) or record mutation
  that a correct check must reject, next to a real-engine run it must accept;
* record fields re-derived from the tape instead of engine state (plan rule 2);
* a residual-defined "cancelled" term the battery would not notice;
* the I6 tolerance: sound against a float32 re-enactment of the engine's
  settlement chain and real extreme-magnitude runs (not too tight), and tight
  enough that a one-price-tick report error is rejected (not too loose).

Fill rows are decoded here by an independent reader (``_decode_row``) written
from V5's row layout, not by the module's private readers. CPU only.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Dict, Iterable, List

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_tape_synth import CapacityEnvelope, TapeBuilder
from omiecid_simulator.omiecid.contract import ContractStatus
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from tests.omiecid.reference.omie_invariants import (
    Cancellation,
    Fill,
    RunRecord,
    assert_check_fails,
    build_record,
    check_i1_volume_balance,
    check_i2_quantity_conservation,
    check_i3_uncrossed_on_return,
    check_i4_priority,
    check_i5_price_setter,
    check_i6_settlement,
    check_i7_no_manufactured_liquidity,
    check_i8_capacity_headroom,
    check_i9_non_vacuity,
    check_m1_tie_permutation,
    check_m2_monotonicity,
    check_m3_ioc_non_equivalence,
    check_m4_fok_dichotomy,
    check_m5_split_aggressor,
    check_m6_ownership_blindness,
    check_record,
    compare_records,
    i6_tolerance,
    run_battery,
)

B, S = int(Side.BUY), int(Side.SELL)
NON, IOC, FOK, ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                      int(ExecRestriction.FOK), int(ExecRestriction.ICE))
PC = OMIEProductConfig()
O = 999_900                     # round(-price_min / price_tick)
K = Fraction(1, 10) * Fraction(1, 100) * Fraction(1, 4)   # EUR per tick*tick
U = Fraction(1, 2 ** 24)

# Book-side columns (OrderSideFeat) and trade-row columns (TradesFeat), V4/V5.
BP, BQ, BOID, BTID, BSEC, BNSEC = 0, 1, 2, 3, 4, 5
TP, TQ, TPASS, TAGRS, TSEC, TNSEC, TPTID, TATID = 0, 1, 2, 3, 4, 5, 6, 7
# iceberg_meta columns (iceberg_manager.py)
M_OID, M_HIDDEN = 0, 1


# ---------------------------------------------------------------------------
# Independent helpers
# ---------------------------------------------------------------------------
def _decode_row(row, row_index: int, submission: int) -> Fill:
    """V5: Q < 0 means the aggressor bought."""
    p, q, po, ao, sec, nsec, pt, at = (int(x) for x in row[:8])
    if q < 0:
        bo, so, ba, sa = ao, po, at, pt
    else:
        bo, so, ba, sa = po, ao, pt, at
    return Fill(row_index=row_index, submission=submission, price_int=p, qty_int=abs(q),
                qty_signed=q, passive_oid=po, aggressor_oid=ao, passive_agent=pt,
                aggressor_agent=at, buyer_oid=bo, seller_oid=so, buyer_agent=ba,
                seller_agent=sa, sec=sec, nsec=nsec)


def _filled_from(fills: Iterable[Fill]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for f in fills:
        for o in {f.passive_oid, f.aggressor_oid}:
            out[o] = out.get(o, 0) + f.qty_int
    return out


def _rewrite_trade_row(rec: RunRecord, row_index: int, edit) -> RunRecord:
    """A broken engine: trade row ``row_index`` reads ``edit(old_row)`` in every
    snapshot that holds it; fills and ``filled`` follow, so RB stays consistent."""
    fill = next(f for f in rec.fills if f.row_index == row_index)
    old = np.array(rec.post_snapshots[fill.submission].trades[row_index], copy=True)
    new = np.array(edit(old.copy()), dtype=old.dtype)

    def fix(snap):
        t = np.asarray(snap.trades)
        if np.array_equal(t[row_index], old):
            t = t.copy()
            t[row_index] = new
            return dataclasses.replace(snap, trades=t)
        return snap

    fills = tuple(_decode_row(new, row_index, f.submission) if f is fill else f
                  for f in rec.fills)
    filled = {oid: _filled_from(fills).get(oid, 0) for oid in rec.accepted}
    return dataclasses.replace(
        rec,
        pre_snapshots=tuple(fix(s) for s in rec.pre_snapshots),
        post_snapshots=tuple(fix(s) for s in rec.post_snapshots),
        final_snapshot=fix(rec.final_snapshot),
        fills=fills, filled=filled)


def _set_book_qty(snap, side: int, oid: int, qty: int):
    arr = np.array(snap.bids if side == B else snap.asks, copy=True)
    hit = arr[:, BOID] == oid
    assert hit.any(), f"oid {oid} not on side {side}"
    arr[hit, BQ] = qty
    return dataclasses.replace(snap, **{"bids" if side == B else "asks": arr})


def _book_visible(arr, oid: int) -> int:
    arr = np.asarray(arr)
    m = (arr[:, BOID] == oid) & (arr[:, BQ] > 0)
    return int(arr[m, BQ].sum())


def _status(rec, **kw):
    return {r.check_id: r.status for r in run_battery(rec, **kw).results}


def _tape(rows: List[dict], **tb_kw) -> np.ndarray:
    tb = TapeBuilder(PC, **tb_kw)
    for r in rows:
        tb.add(**r)
    return tb.build()


# ---------------------------------------------------------------------------
# Real-engine records (module-scoped; each is a small jitted run)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rec_walk():
    """S1 @50 2.0 MW, S2 @51 2.0 MW, B3 @51 3.0 MW: takes 2.0 of S1, 1.0 of S2."""
    return build_record(_tape([
        dict(side=S, price_eur=50.0, qty_mw=2.0, agent_id=1),
        dict(side=S, price_eur=51.0, qty_mw=2.0, agent_id=2),
        dict(side=B, price_eur=51.0, qty_mw=3.0, agent_id=3),
    ]))


@pytest.fixture(scope="module")
def rec_time():
    """S1 @50 2.0 (earlier), S2 @50 2.0 (later), B3 @50 3.0: S1 then 1.0 of S2."""
    return build_record(_tape([
        dict(side=S, price_eur=50.0, qty_mw=2.0, agent_id=1),
        dict(side=S, price_eur=50.0, qty_mw=2.0, agent_id=2),
        dict(side=B, price_eur=50.0, qty_mw=3.0, agent_id=3),
    ]))


@pytest.fixture(scope="module")
def rec_ice():
    """ICE S1 @50 qty 1.0 peak 0.3, B2 @50 0.4: visible 0.2 + hidden 0.4 remain."""
    return build_record(_tape([
        dict(side=S, price_eur=50.0, qty_mw=1.0, peak_qty_mw=0.3, exec_restriction=ICE,
             agent_id=1),
        dict(side=B, price_eur=50.0, qty_mw=0.4, agent_id=2),
    ]))


@pytest.fixture(scope="module")
def rec_ice_hop_base():
    """ICE S1 @50 qty 1.0 peak 0.3, B2 @52 0.2 (a fill whose setter is the iceberg)."""
    return build_record(_tape([
        dict(side=S, price_eur=50.0, qty_mw=1.0, peak_qty_mw=0.3, exec_restriction=ICE,
             agent_id=1),
        dict(side=B, price_eur=52.0, qty_mw=0.2, agent_id=2),
    ]))


@pytest.fixture(scope="module")
def rec_icewpi():
    """ICEwPI S1 @50 qty 1.0 peak 0.3 ppd +0.5 (4 declared slices), B2 @52 0.2.

    B2's limit 52.00 = base + 4*ppd sits at/above every rung the I5 ladder audit
    exercises (k = -1..4), so that audit isolates the ladder sub-claim from V9's
    limit_cross clause (v5.187, D3-02). The real engine still prints at the
    resting iceberg's 50.00."""
    return build_record(_tape([
        dict(side=S, price_eur=50.0, qty_mw=1.0, peak_qty_mw=0.3, ppd_eur=0.5,
             exec_restriction=ICE, agent_id=1),
        dict(side=B, price_eur=52.0, qty_mw=0.2, agent_id=2),
    ]))


@pytest.fixture(scope="module")
def rec_cancel():
    """S1 @50 1.0; B2 @50 0.3 fills 0.3; S1 cancelled before B3 @40 0.1 (rests)."""
    return build_record(_tape([
        dict(side=S, price_eur=50.0, qty_mw=1.0, agent_id=1),
        dict(side=B, price_eur=50.0, qty_mw=0.3, agent_id=2),
        dict(side=B, price_eur=40.0, qty_mw=0.1, agent_id=3),
    ]), cancels_before={3: (1,)})


@pytest.fixture(scope="module")
def rec_expire():
    """S1 @50 1.0; B2 @50 0.3; gate expiry removes S1's 0.7."""
    return build_record(_tape([
        dict(side=S, price_eur=50.0, qty_mw=1.0, agent_id=1),
        dict(side=B, price_eur=50.0, qty_mw=0.3, agent_id=2),
    ]), expire_at_gct=True)


@pytest.fixture(scope="module")
def rec_evict():
    """book_depth=4 and seven 1.0-MW sells: the engine wipes levels (raises book_full)."""
    rows = [dict(side=S, price_eur=50.0 + i, qty_mw=1.0, agent_id=1) for i in range(6)]
    rows.append(dict(side=S, price_eur=40.0, qty_mw=1.0, agent_id=1))
    return build_record(_tape(rows), market_config=OMIEMarketConfig(book_depth=4))


def _extreme_walk_tape(sign: int) -> np.ndarray:
    """Eight resting levels at the price extreme, each 124.8 MW; one 999-MW taker."""
    rows = []
    if sign > 0:   # sells near +9999, a buy sweeps them
        for i in range(8):
            rows.append(dict(side=S, price_int=1_999_800 - 7 + i, qty_int=1248,
                             agent_id=1 + i % 3))
        rows.append(dict(side=B, price_int=1_999_800, qty_int=9990, agent_id=7))
    else:          # bids near -9999, a sell sweeps them
        for i in range(8):
            rows.append(dict(side=B, price_int=7 - i, qty_int=1248, agent_id=1 + i % 3))
        rows.append(dict(side=S, price_int=0, qty_int=9990, agent_id=7))
    return _tape(rows)


def _near_zero_tape() -> np.ndarray:
    """Levels straddling 0 EUR, so D_s is a small difference of large numbers."""
    rows = [dict(side=S, price_int=O - 2 + i, qty_int=777 + 13 * i, agent_id=1 + i % 2)
            for i in range(5)]
    rows.append(dict(side=B, price_int=O + 2, qty_int=3999, agent_id=5))
    return _tape(rows)


@pytest.fixture(scope="module")
def rec_extreme_hi():
    return build_record(_extreme_walk_tape(+1))


@pytest.fixture(scope="module")
def rec_extreme_lo():
    return build_record(_extreme_walk_tape(-1))


@pytest.fixture(scope="module")
def rec_near_zero():
    return build_record(_near_zero_tape())


# ---------------------------------------------------------------------------
# Clean real-engine runs the battery must accept (I6 not too tight, no false fails)
# ---------------------------------------------------------------------------
class TestCleanRealEngineRuns:
    @pytest.mark.parametrize("name", ["rec_walk", "rec_time", "rec_ice", "rec_icewpi",
                                      "rec_cancel", "rec_expire", "rec_ice_hop_base"])
    def test_small_real_runs_pass_whole_battery(self, name, request):
        rec = request.getfixturevalue(name)
        report = run_battery(rec, expects_trade=True)
        assert report.passed and report.valid, report.statuses()

    @pytest.mark.parametrize("name", ["rec_extreme_hi", "rec_extreme_lo", "rec_near_zero"])
    def test_extreme_magnitude_walks_pass_whole_battery(self, name, request):
        rec = request.getfixturevalue(name)
        assert len(rec.fills) >= 5
        report = run_battery(rec, expects_trade=True)
        assert report.passed and report.valid, {
            r.check_id: (r.status, [v.message for v in r.violations]) for r in report.results}

    @pytest.mark.parametrize("name", ["rec_extreme_hi", "rec_extreme_lo", "rec_near_zero"])
    def test_extreme_walks_settlement_error_is_inside_tau(self, name, request):
        r = check_i6_settlement(request.getfixturevalue(name))
        assert r.status == "pass", r.violations
        assert 0.0 <= r.evidence["max_err_over_tau"] <= 1.0

    @pytest.mark.slow
    def test_eager_extreme_walk_passes_and_matches_jitted(self, rec_extreme_hi):
        """slow: a whole build under jax.disable_jit() (~1-3 s eager)."""
        eager = build_record(_extreme_walk_tape(+1), jit=False)
        assert run_battery(eager, expects_trade=True).passed
        assert compare_records(eager, rec_extreme_hi) == ()

    def test_eviction_run_is_flagged_by_i2_and_i8_but_consistent_for_rb(self, rec_evict):
        assert check_record(rec_evict).status == "pass"
        assert sum(rec_evict.evicted.values()) > 0
        assert_check_fails(check_i2_quantity_conservation(rec_evict), "I2", field="evicted")
        assert check_i2_quantity_conservation(rec_evict, allow_eviction=True).status == "pass"
        assert_check_fails(check_i8_capacity_headroom(rec_evict), "I8", status="invalid")


# ---------------------------------------------------------------------------
# Plan rule 2: buckets read from engine state, not the tape
# ---------------------------------------------------------------------------
class TestRecordReadsEngineState:
    def test_iceberg_buckets_equal_raw_engine_arrays_not_tape_arithmetic(self, rec_ice):
        fin = rec_ice.final_snapshot
        meta = np.asarray(fin.iceberg_meta)
        hidden_raw = int(meta[meta[:, M_OID] == 1, M_HIDDEN].sum())
        assert rec_ice.still_resting[1] == _book_visible(fin.asks, 1) == 2
        assert rec_ice.hidden_remaining[1] == hidden_raw == 4
        # tape arithmetic would give visible = peak (3) and hidden = qty - peak (7)
        assert (rec_ice.still_resting[1], rec_ice.hidden_remaining[1]) != (3, 7)

    def test_partial_resting_qty_comes_from_final_book(self, rec_walk):
        assert rec_walk.still_resting[2] == _book_visible(rec_walk.final_snapshot.asks, 2) == 10
        assert rec_walk.filled == {1: 20, 2: 10, 3: 30}

    def test_cancellation_qty_equals_book_before_the_cancel(self, rec_cancel):
        (c,) = rec_cancel.cancelled_or_expired[1]
        before = rec_cancel.post_snapshots[1]           # state just before the cancel
        assert c.cause == "cancellation" and c.qty_int == _book_visible(before.asks, 1) == 7

    def test_gate_expiry_qty_equals_book_before_expiry(self, rec_expire):
        (c,) = rec_expire.cancelled_or_expired[1]
        assert c.cause == "gate_expiry"
        assert c.qty_int == _book_visible(rec_expire.post_snapshots[-1].asks, 1) == 7
        assert rec_expire.still_resting[1] == 0

    def test_record_rederived_still_resting_from_tape_is_rejected_by_rb(self, rec_walk):
        bad = dataclasses.replace(rec_walk, still_resting={**rec_walk.still_resting, 2: 20})
        assert_check_fails(check_record(bad), "RB", field="still_resting")

    def test_record_rederived_hidden_from_tape_is_rejected_by_rb(self, rec_ice):
        bad = dataclasses.replace(rec_ice, hidden_remaining={**rec_ice.hidden_remaining, 1: 7})
        assert_check_fails(check_record(bad), "RB", field="hidden_remaining")

    def test_dropped_fill_is_rejected_by_rb(self, rec_walk):
        bad = dataclasses.replace(rec_walk, fills=rec_walk.fills[:1])
        assert_check_fails(check_record(bad), "RB", field="fills")

    def test_fill_mis_tagged_to_another_submission_is_rejected_by_rb(self, rec_walk):
        f0 = rec_walk.fills[0]
        bad = dataclasses.replace(rec_walk, fills=(dataclasses.replace(f0, submission=1),
                                                   *rec_walk.fills[1:]))
        assert_check_fails(check_record(bad), "RB", field="fills")

    def test_compare_records_sees_an_integer_drift(self, rec_walk):
        bad = dataclasses.replace(rec_walk, still_resting={**rec_walk.still_resting, 2: 9})
        assert "still_resting" in compare_records(rec_walk, bad)
        assert compare_records(rec_walk, rec_walk) == ()


# ---------------------------------------------------------------------------
# I1
# ---------------------------------------------------------------------------
class TestI1Audit:
    def test_engine_writes_contradictory_q_sign_rb_passes_i1_fails(self, rec_walk):
        def flip(row):
            row[TQ] = -row[TQ]
            return row
        bad = _rewrite_trade_row(rec_walk, rec_walk.fills[0].row_index, flip)
        assert check_record(bad).status == "pass"      # the record is consistent
        assert_check_fails(check_i1_volume_balance(bad), "I1", field="fill_sides")

    def test_both_legs_named_on_one_side_fails(self, rec_walk):
        def same_side(row):
            row[TPASS] = 2            # passive now the other SELL order
            row[TAGRS] = 1            # aggressor a SELL too
            return row
        bad = _rewrite_trade_row(rec_walk, rec_walk.fills[0].row_index, same_side)
        assert_check_fails(check_i1_volume_balance(bad), "I1", field="fill_sides")

    def test_leg_naming_an_unknown_order_fails(self, rec_walk):
        def ghost(row):
            row[TPASS] = 999
            return row
        bad = _rewrite_trade_row(rec_walk, rec_walk.fills[0].row_index, ghost)
        assert_check_fails(check_i1_volume_balance(bad), "I1")


# ---------------------------------------------------------------------------
# I2
# ---------------------------------------------------------------------------
class TestI2Audit:
    def test_resting_qty_vanishing_without_trade_row_fails_identity(self, rec_walk):
        fin = _set_book_qty(rec_walk.final_snapshot, S, 2, 9)
        bad = dataclasses.replace(rec_walk, final_snapshot=fin,
                                  still_resting={**rec_walk.still_resting, 2: 9})
        assert check_record(bad).status == "pass"
        assert_check_fails(check_i2_quantity_conservation(bad), "I2", field="identity")

    def test_engine_ignoring_cancel_fails_identity(self, rec_cancel):
        # the cancel_order of oid 1 left it on the book
        pre_cancel = rec_cancel.post_snapshots[1]
        fin = dataclasses.replace(rec_cancel.final_snapshot, asks=np.array(pre_cancel.asks))
        bad = dataclasses.replace(rec_cancel, final_snapshot=fin,
                                  still_resting={**rec_cancel.still_resting, 1: 7})
        assert check_record(bad).status == "pass"
        assert_check_fails(check_i2_quantity_conservation(bad), "I2")

    def test_engine_ignoring_gate_expiry_fails_identity(self, rec_expire):
        fin = dataclasses.replace(rec_expire.final_snapshot,
                                  asks=np.array(rec_expire.post_snapshots[-1].asks))
        bad = dataclasses.replace(rec_expire, final_snapshot=fin,
                                  still_resting={**rec_expire.still_resting, 1: 7})
        assert check_record(bad).status == "pass"
        assert_check_fails(check_i2_quantity_conservation(bad), "I2")

    def test_iceberg_hidden_inflated_in_engine_meta_fails_identity(self, rec_ice):
        fin = rec_ice.final_snapshot
        meta = np.array(fin.iceberg_meta, copy=True)
        meta[meta[:, M_OID] == 1, M_HIDDEN] += 1
        bad = dataclasses.replace(rec_ice, final_snapshot=dataclasses.replace(fin, iceberg_meta=meta),
                                  hidden_remaining={**rec_ice.hidden_remaining, 1: 5})
        assert check_record(bad).status == "pass"
        assert_check_fails(check_i2_quantity_conservation(bad), "I2", field="identity")

    def test_residual_cancellation_over_phantom_decrement_is_rejected(self, rec_cancel):
        """The first attempt's defect: the engine silently shrank S1 from 7 to 5
        ticks (no trade row) before the cancel, and a residual-defined builder
        files accepted - filled - resting = 7 as the cancellation. The record holds
        the pre-cancel state (post_snapshots[1]); the battery must notice the
        2 ticks that vanished."""
        post1 = _set_book_qty(rec_cancel.post_snapshots[1], S, 1, 5)
        bad = dataclasses.replace(
            rec_cancel, post_snapshots=(rec_cancel.post_snapshots[0], post1,
                                        *rec_cancel.post_snapshots[2:]))
        assert bad.cancelled_or_expired[1][0].qty_int == 7       # the residual
        assert not run_battery(bad, expects_trade=True).passed, \
            "residual cancellation over a phantom decrement passes the whole battery"

    def test_residual_gate_expiry_over_phantom_decrement_is_rejected(self, rec_expire):
        """As above for gate expiry: the state immediately before expire_at_gct
        (post_snapshots[-1]) holds 5 ticks, the record files 7 as expired."""
        last = _set_book_qty(rec_expire.post_snapshots[-1], S, 1, 5)
        bad = dataclasses.replace(rec_expire,
                                  post_snapshots=(*rec_expire.post_snapshots[:-1], last))
        assert not run_battery(bad, expects_trade=True).passed, \
            "residual gate expiry over a phantom decrement passes the whole battery"

    def test_duplicated_cancellation_entry_fails(self, rec_cancel):
        cs = rec_cancel.cancelled_or_expired[1]
        bad = dataclasses.replace(rec_cancel, cancelled_or_expired={
            **rec_cancel.cancelled_or_expired, 1: cs + cs})
        assert_check_fails(check_i2_quantity_conservation(bad), "I2")

    def test_eviction_refiled_as_gate_expiry_on_unexpired_run_fails(self, rec_evict):
        moved = {o: (Cancellation(o, "gate_expiry", q, rec_evict.n_submissions),)
                 for o, q in rec_evict.evicted.items() if q}
        bad = dataclasses.replace(
            rec_evict, evicted={o: 0 for o in rec_evict.evicted},
            cancelled_or_expired={**rec_evict.cancelled_or_expired, **moved})
        assert_check_fails(check_i2_quantity_conservation(bad), "I2",
                           field="cancelled_or_expired")
        assert_check_fails(check_record(bad), "RB", field="evicted")

    def test_fok_kill_cause_on_a_non_order_fails(self, rec_walk):
        bad = dataclasses.replace(rec_walk, cancelled_or_expired={
            **rec_walk.cancelled_or_expired, 2: (Cancellation(2, "fok_kill", 10, 1),)},
            still_resting={**rec_walk.still_resting, 2: 0})
        assert_check_fails(check_i2_quantity_conservation(bad), "I2",
                           field="cancelled_or_expired")

    def test_non_section_5_1_cause_fails(self, rec_walk):
        bad = dataclasses.replace(rec_walk, cancelled_or_expired={
            **rec_walk.cancelled_or_expired, 2: (Cancellation(2, "residual", 10, 1),)},
            still_resting={**rec_walk.still_resting, 2: 0})
        assert_check_fails(check_i2_quantity_conservation(bad), "I2",
                           field="cancelled_or_expired")


# ---------------------------------------------------------------------------
# I3
# ---------------------------------------------------------------------------
class TestI3Audit:
    def _cross(self, snap, ask_price):
        asks = np.array(snap.asks, copy=True)
        asks[asks[:, BOID] == 2, BP] = ask_price
        return dataclasses.replace(snap, asks=asks)

    def test_crossed_only_in_final_snapshot_fails(self, rec_walk):
        bb = 1004000
        bids = np.array(rec_walk.final_snapshot.bids, copy=True)
        i = int(np.nonzero(bids[:, BOID] <= 0)[0][0])
        bids[i, [BP, BQ, BOID, BTID, BSEC, BNSEC]] = [1005100, 5, 77, 9, 99, 0]
        fin = dataclasses.replace(rec_walk.final_snapshot, bids=bids)
        bad = dataclasses.replace(rec_walk, final_snapshot=fin)
        assert bb  # bid 51.01 > ask 51.00
        assert_check_fails(check_i3_uncrossed_on_return(bad), "I3")

    def test_locked_book_bid_equal_ask_on_last_return_fails(self, rec_walk):
        last = rec_walk.post_snapshots[-1]
        bids = np.array(last.bids, copy=True)
        i = int(np.nonzero(bids[:, BOID] <= 0)[0][0])
        bids[i, [BP, BQ, BOID, BTID, BSEC, BNSEC]] = [1005000, 5, 77, 9, 99, 0]   # == ask 51.00
        bad = dataclasses.replace(rec_walk, post_snapshots=(
            *rec_walk.post_snapshots[:-1], dataclasses.replace(last, bids=bids)))
        assert_check_fails(check_i3_uncrossed_on_return(bad), "I3")


# ---------------------------------------------------------------------------
# I4
# ---------------------------------------------------------------------------
class TestI4Audit:
    def test_worse_priced_maker_filled_while_better_waits_fails(self, rec_walk):
        def worse(row):
            row[TPASS], row[TPTID], row[TP] = 2, 2, 1005000
            return row
        bad = _rewrite_trade_row(rec_walk, rec_walk.fills[0].row_index, worse)
        assert check_i5_price_setter(bad).status == "pass"   # priced at its maker: I5 blind
        assert_check_fails(check_i4_priority(bad), "I4")

    def test_later_equal_price_maker_filled_first_fails(self, rec_time):
        def later(row):
            row[TPASS], row[TPTID] = 2, 2
            return row
        bad = _rewrite_trade_row(rec_time, rec_time.fills[0].row_index, later)
        assert_check_fails(check_i4_priority(bad), "I4")

    def test_maker_overfilled_past_its_depth_while_peer_waits_fails(self, rec_time):
        def again(row):
            row[TPASS], row[TPTID] = 1, 1
            return row
        bad = _rewrite_trade_row(rec_time, rec_time.fills[1].row_index, again)
        assert_check_fails(check_i4_priority(bad), "I4")

    def test_real_depth_walk_across_equal_price_queue_passes(self, rec_time):
        assert [f.passive_oid for f in rec_time.fills] == [1, 2]
        assert check_i4_priority(rec_time).status == "pass"


# ---------------------------------------------------------------------------
# I5
# ---------------------------------------------------------------------------
class TestI5Audit:
    def test_one_tick_mispricing_against_its_maker_fails(self, rec_walk):
        def tick(row):
            row[TP] += 1
            return row
        bad = _rewrite_trade_row(rec_walk, rec_walk.fills[0].row_index, tick)
        assert_check_fails(check_i5_price_setter(bad), "I5")

    def test_rejected_formulation_price_of_another_resting_order_fails(self, rec_walk):
        """A fill taking the 51.00 maker at 50.00 (a price resting on that side)."""
        def other(row):
            row[TP] = 1004900
            return row
        bad = _rewrite_trade_row(rec_walk, rec_walk.fills[1].row_index, other)
        assert_check_fails(check_i5_price_setter(bad), "I5")

    def _hop(self, rec, price):
        def hop(row):
            # slice of ICE oid 1 (seller) aggresses the submitting order's remainder (oid 2)
            row[TPASS], row[TAGRS] = 2, 1
            row[TPTID], row[TATID] = 2, 1
            row[TQ] = abs(row[TQ])       # Q > 0: the passive side bought
            row[TP] = price
            return row
        return _rewrite_trade_row(rec, rec.fills[0].row_index, hop)

    def test_rule3_hop_at_aggressor_limit_fails(self, rec_ice_hop_base):
        bad = self._hop(rec_ice_hop_base, 1005100)          # B2's limit 52.00
        r = assert_check_fails(check_i5_price_setter(bad), "I5", field="rule3_hop")
        assert r.evidence["rule3_hop_fills"] == 1

    def test_rule3_hop_at_iceberg_limit_passes(self, rec_ice_hop_base):
        ok = self._hop(rec_ice_hop_base, 1004900)           # the iceberg's 50.00
        r = check_i5_price_setter(ok)
        assert r.status == "pass" and r.evidence["rule3_hop_fills"] == 1

    def test_unruled_hop_fails(self, rec_walk):
        def third(row):
            row[TPASS], row[TAGRS] = 1, 2
            return row
        bad = _rewrite_trade_row(rec_walk, rec_walk.fills[0].row_index, third)
        assert_check_fails(check_i5_price_setter(bad), "I5", field="unruled_hop")

    def test_icewpi_price_on_declared_ladder_passes_and_beyond_slice_count_fails(self, rec_icewpi):
        base = rec_icewpi.accepted[1].price_int
        ppd = rec_icewpi.accepted[1].ppd_int
        assert ppd == 50
        # The buyer's limit covers every rung exercised below, so k = 4 cannot fail
        # on limit_cross (V9 as amended at v5.187); only the ladder sub-claim can.
        assert rec_icewpi.accepted[2].price_int >= base + 4 * ppd
        assert rec_icewpi.fills[0].passive_oid == 1          # the iceberg sets the price
        def at(k):
            def edit(row):
                row[TP] = base + k * ppd
                return row
            return _rewrite_trade_row(rec_icewpi, rec_icewpi.fills[0].row_index, edit)
        ok = check_i5_price_setter(at(3))                      # k = 3 < ceil(10/3) = 4
        assert ok.status == "pass", ok.violations
        r4 = assert_check_fails(check_i5_price_setter(at(4)), "I5", field="fills")
        assert {v.field for v in r4.violations} == {"fills"}   # off the ladder, not a cross
        # k = -1 (wrong direction) is below the seller's base, so it also crosses the
        # seller's limit; the ladder sub-claim must be raised on its own field regardless.
        rm1 = assert_check_fails(check_i5_price_setter(at(-1)), "I5", field="fills")
        assert {v.field for v in rm1.violations} == {"fills", "limit_cross"}


# ---------------------------------------------------------------------------
# I6 and the derived tolerance
# ---------------------------------------------------------------------------
def _tau_exact(rows):
    W = sum(Fraction(p) * q for p, q in rows)
    Q = sum(Fraction(q) for _p, q in rows)
    D = W - O * Q
    # V10 as amended at Step 5 (v5.187), with c_s = len(rows) (chunks=None).
    return K * U * (len(rows) * W + O * Q + 5 * abs(D)) * (1 + (len(rows) + 8) * U), K * D


def _engine_chain_f32(chunks) -> float:
    """Re-enactment of contract_book's float32 settlement chain: each chunk's int
    weighted price and qty cast to float32, accumulated in float32, then
    (W - O*Q) * fl(0.1) * fl(0.01) * fl(0.25), every op rounded to float32."""
    f = np.float32
    W, Q = f(0.0), f(0.0)
    for rows in chunks:
        W = f(W + f(sum(int(p) * int(q) for p, q in rows)))
        Q = f(Q + f(sum(int(q) for _p, q in rows)))
    v = f(W - f(f(O) * Q))
    return float(f(f(f(v * f(0.1)) * f(0.01)) * f(0.25)))


class TestI6ToleranceAudit:
    @pytest.mark.parametrize("rows", [
        [(1_004_598, 20)],
        [(1_999_800, 9990)],
        [(0, 9990)],
        [(O, 37)],
        [(1_004_900, 20), (1_005_000, 10), (1_005_100, 3)],
        [(1_999_800 - i, 1248) for i in range(8)],
    ])
    def test_tau_and_value_match_exact_rational_formula(self, rows):
        tau, value = i6_tolerance(rows, PC)
        t_exact, v_exact = _tau_exact(rows)
        assert tau == pytest.approx(float(t_exact), rel=1e-12, abs=0.0)
        assert value == pytest.approx(float(v_exact), rel=1e-12, abs=1e-15)

    def test_pinned_card_example(self):
        tau, value = i6_tolerance([(1_004_598, 20)], PC)
        assert tau == pytest.approx(6.043878e-4, rel=1e-6)
        assert value == pytest.approx(23.49, rel=1e-12)

    def test_tau_bounds_float32_chain_worst_case_chunking(self):
        """Not too tight: tau_s covers the float32 chain for random extreme rows,
        both one chunk per row (the bound's worst case) and one chunk overall."""
        rng = np.random.default_rng(20261002)
        worst = 0.0
        for _ in range(400):
            m = int(rng.integers(1, 12))
            hi = bool(rng.integers(0, 2))
            p = rng.integers(1_900_000 if hi else 0, 1_999_801 if hi else 100_000, m)
            if rng.integers(0, 3) == 0:
                p = rng.integers(O - 300, O + 300, m)
            q = rng.integers(1, 9991, m)
            rows = list(zip(p.tolist(), q.tolist()))
            tau, value = i6_tolerance(rows, PC)
            for chunks in ([[r] for r in rows], [rows]):
                err = abs(_engine_chain_f32(chunks) - value)
                worst = max(worst, err / tau)
                assert err <= tau, (rows, chunks is rows, err, tau)
        assert worst > 0.0

    @pytest.mark.parametrize("m", [1, 2, 3, 4])
    def test_one_price_tick_on_whole_submission_exceeds_tau_for_few_rows(self, m):
        """Not too loose: for up to four rows anywhere on the price range, a report
        off by one price tick on its whole quantity (k*Q_s) is outside tau_s."""
        for p in (0, 1, O // 2, O - 1, O, O + 1, 3 * O // 2, 1_999_800):
            for q in (1, 7, 1000, 9990):
                rows = [(p, q)] * m
                tau, _ = i6_tolerance(rows, PC)
                assert float(K) * q * m > tau, (m, p, q, tau)

    def test_one_price_tick_report_error_fails_i6_on_real_run(self, rec_walk):
        s = rec_walk.fills[0].submission
        er = rec_walk.engine_reports[s]
        Q = sum(f.qty_int for f in rec_walk.fills if f.submission == s)
        bad_er = dataclasses.replace(er, settlement_value_eur=er.settlement_value_eur + float(K) * Q)
        reports = tuple(bad_er if i == s else r for i, r in enumerate(rec_walk.engine_reports))
        bad = dataclasses.replace(rec_walk, engine_reports=reports)
        assert_check_fails(check_i6_settlement(bad), "I6", field="engine_reports")

    def test_extreme_sweep_engine_resolves_one_price_tick(self, rec_extreme_hi):
        """Evidence for the test below: on the clean eight-row ~9999 EUR sweep the
        engine's own float32 settlement error is far below one price tick on the
        whole quantity (k*Q_s), so a one-tick report error is resolvable here."""
        r = check_i6_settlement(rec_extreme_hi)
        assert r.status == "pass"
        s = rec_extreme_hi.fills[0].submission
        Q = sum(f.qty_int for f in rec_extreme_hi.fills if f.submission == s)
        err = r.evidence["max_err_over_tau"] * r.evidence["tau_sum"]
        assert err < 0.1 * float(K) * Q

    def test_one_price_tick_report_error_fails_i6_on_extreme_eight_row_walk(self, rec_extreme_hi):
        """Not too loose on a realistic multi-level sweep: eight rows at ~9999 EUR
        swept by ONE taker in one dispatch; the report is off by one price tick on
        the whole quantity (k*Q = 2.4975 EUR)."""
        s = rec_extreme_hi.fills[0].submission
        er = rec_extreme_hi.engine_reports[s]
        Q = sum(f.qty_int for f in rec_extreme_hi.fills if f.submission == s)
        bad_er = dataclasses.replace(er, settlement_value_eur=er.settlement_value_eur - float(K) * Q)
        reports = tuple(bad_er if i == s else r for i, r in enumerate(rec_extreme_hi.engine_reports))
        bad = dataclasses.replace(rec_extreme_hi, engine_reports=reports)
        assert_check_fails(check_i6_settlement(bad), "I6", field="engine_reports")

    def test_one_qty_tick_report_error_fails_i6(self, rec_walk):
        s = rec_walk.fills[0].submission
        er = rec_walk.engine_reports[s]
        bad_er = dataclasses.replace(er, trade_qty_mw=er.trade_qty_mw + 0.1)
        bad = dataclasses.replace(rec_walk, engine_reports=tuple(
            bad_er if i == s else r for i, r in enumerate(rec_walk.engine_reports)))
        assert_check_fails(check_i6_settlement(bad), "I6", field="engine_reports")

    def test_traded_flag_false_with_rows_fails_i6(self, rec_walk):
        s = rec_walk.fills[0].submission
        bad_er = dataclasses.replace(rec_walk.engine_reports[s], traded=False)
        bad = dataclasses.replace(rec_walk, engine_reports=tuple(
            bad_er if i == s else r for i, r in enumerate(rec_walk.engine_reports)))
        assert_check_fails(check_i6_settlement(bad), "I6", field="engine_reports")

    def test_settlement_leg_off_by_a_tenth_of_a_percent_fails_i6a(self, rec_walk):
        es = dict(rec_walk.engine_settlements)
        es[1] = es[1] * 1.001
        bad = dataclasses.replace(rec_walk, engine_settlements=es)
        assert_check_fails(check_i6_settlement(bad), "I6", field="engine_settlements")

    def test_agent_not_declared_for_its_oid_fails_i6(self, rec_walk):
        def relabel(row):
            row[TPTID] = 2           # oid 1 belongs to agent 1, not 2
            return row
        bad = _rewrite_trade_row(rec_walk, rec_walk.fills[0].row_index, relabel)
        assert_check_fails(check_i6_settlement(bad), "I6")


# ---------------------------------------------------------------------------
# I7
# ---------------------------------------------------------------------------
class TestI7Audit:
    def test_traded_volume_beyond_accepted_sells_fails(self, rec_walk):
        def big(row):
            row[TQ] = -41            # sells accepted total 40 ticks
            return row
        bad = _rewrite_trade_row(rec_walk, rec_walk.fills[0].row_index, big)
        assert_check_fails(check_i7_no_manufactured_liquidity(bad), "I7")

    def test_real_run_at_exact_cap_passes(self, rec_walk):
        r = check_i7_no_manufactured_liquidity(rec_walk)
        assert r.status == "pass" and r.evidence["traded"] == 30


# ---------------------------------------------------------------------------
# I8
# ---------------------------------------------------------------------------
class TestI8Audit:
    def test_record_observed_capacity_is_not_trusted(self, rec_walk):
        env = CapacityEnvelope(max_resting_per_side=1)
        bad = dataclasses.replace(rec_walk, observed_capacity={
            "resting_per_side": 0, "trades": 0, "concurrent_icebergs": 0})
        assert_check_fails(check_i8_capacity_headroom(bad, envelope=env), "I8", status="invalid")

    def test_cleared_flags_over_engine_book_full_still_invalid(self, rec_evict):
        """The snapshots hold book_full=True (engine state); a record whose
        ``flags`` lost it must still be ruled INVALID (or be rejected by RB)."""
        assert any(s.book_full for s in rec_evict.post_snapshots)
        bad = dataclasses.replace(rec_evict, flags={k: False for k in rec_evict.flags})
        i8 = check_i8_capacity_headroom(bad)
        rb = check_record(bad)
        assert i8.status == "invalid" or rb.status == "fail", (
            "flags field is trusted by I8 and unvalidated by RB")

    def test_cleared_truncated_flag_over_engine_truncation_still_invalid(self, rec_walk):
        last = dataclasses.replace(rec_walk.post_snapshots[-1], truncated=True)
        bad = dataclasses.replace(rec_walk, post_snapshots=(*rec_walk.post_snapshots[:-1], last))
        assert bad.flags["truncated"] is False
        i8 = check_i8_capacity_headroom(bad)
        rb = check_record(bad)
        assert i8.status == "invalid" or rb.status == "fail", (
            "a truncated snapshot behind a clean flags field passes I8 and RB")

    def test_allow_breach_keeps_evidence(self, rec_evict):
        r = check_i8_capacity_headroom(rec_evict, allow_breach=True)
        assert r.status == "not_applicable" and r.violations and r.evidence["flags"]["book_full"]
        assert "matching_iterations" in r.unobserved


# ---------------------------------------------------------------------------
# I9
# ---------------------------------------------------------------------------
class TestI9Audit:
    def test_closed_contract_run_has_no_accepted_order(self):
        rec = build_record(_tape([dict(side=S, price_eur=50.0, qty_mw=1.0)]),
                           contract_status=ContractStatus.CLOSED)
        assert rec.rejections and rec.rejections[0].reason == "contract_not_open"
        assert_check_fails(check_i9_non_vacuity(rec, expects_trade=False), "I9")

    def test_expected_trade_absent_fails(self, rec_cancel):
        no_fill = dataclasses.replace(rec_cancel, fills=())
        assert_check_fails(check_i9_non_vacuity(no_fill, expects_trade=True), "I9")
        assert check_i9_non_vacuity(no_fill, expects_trade=False).status == "pass"

    def test_expects_trade_has_no_default(self, rec_walk):
        with pytest.raises(TypeError):
            check_i9_non_vacuity(rec_walk)


# ---------------------------------------------------------------------------
# M1-M6: a real passing pair and an independent broken pair each
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def m_tapes():
    def ladder(order):
        tb = TapeBuilder(PC)
        rows = {1: dict(side=S, price_eur=50.0, qty_mw=1.0, agent_id=1, order_id=1, timestamp=5),
                2: dict(side=S, price_eur=51.0, qty_mw=1.0, agent_id=2, order_id=2, timestamp=5)}
        for o in order:
            tb.add(**rows[o])
        tb.add(side=B, price_eur=51.0, qty_mw=1.5, agent_id=3, order_id=3, timestamp=6)
        return tb.build()
    return {
        "tie_a": build_record(ladder((1, 2))),
        "tie_b": build_record(ladder((2, 1))),
    }


class TestMetamorphicAudit:
    def test_m1_real_permutation_passes_and_priced_through_fails(self, m_tapes):
        assert check_m1_tie_permutation(m_tapes["tie_a"], m_tapes["tie_b"],
                                        tie_oids=(1, 2)).status == "pass"
        def worse(row):
            row[TPASS], row[TPTID], row[TP] = 2, 2, 1005000
            return row
        broken = _rewrite_trade_row(m_tapes["tie_b"], m_tapes["tie_b"].fills[0].row_index, worse)
        assert_check_fails(check_m1_tie_permutation(m_tapes["tie_a"], broken, tie_oids=(1, 2)), "M1")

    def test_m2_size_and_limit(self):
        def run(qty, px):
            return build_record(_tape([
                dict(side=S, price_eur=50.0, qty_mw=1.0, agent_id=1),
                dict(side=S, price_eur=51.0, qty_mw=1.0, agent_id=2),
                dict(side=B, price_eur=px, qty_mw=qty, agent_id=3)]))
        small, large, tight = run(1.2, 51.0), run(1.8, 51.0), run(1.8, 50.0)
        assert check_m2_monotonicity(small, large, aggressor_oid=3,
                                     relation="size_increased").status == "pass"
        assert check_m2_monotonicity(large, tight, aggressor_oid=3,
                                     relation="limit_tightened").status == "pass"
        assert_check_fails(check_m2_monotonicity(large, small, aggressor_oid=3,
                                                 relation="size_increased"), "M2")
        assert_check_fails(check_m2_monotonicity(tight, large, aggressor_oid=3,
                                                 relation="limit_tightened"), "M2")

    def _ioc_non(self, er):
        return build_record(_tape([
            dict(side=S, price_eur=50.0, qty_mw=2.0, agent_id=1),
            dict(side=B, price_eur=50.0, qty_mw=3.0, agent_id=2, exec_restriction=er)]))

    def test_m3_real_pair_passes_and_ioc_resting_like_non_fails(self):
        ioc, non = self._ioc_non(IOC), self._ioc_non(NON)
        assert check_m3_ioc_non_equivalence(ioc, non, oid=2).status == "pass"
        # an engine treating IOC as NON: its remainder rests
        assert_check_fails(check_m3_ioc_non_equivalence(non, non, oid=2), "M3")

    def test_m4_kill_with_identical_book_passes_partial_fill_fails(self):
        def run(er, qty):
            return build_record(_tape([
                dict(side=S, price_eur=50.0, qty_mw=2.0, agent_id=1),
                dict(side=B, price_eur=50.0, qty_mw=qty, agent_id=2, exec_restriction=er)]))
        killed, full, partial = run(FOK, 3.0), run(FOK, 2.0), run(NON, 3.0)
        assert not killed.fills
        assert check_m4_fok_dichotomy(killed, fok_oid=2).status == "pass"
        assert check_m4_fok_dichotomy(full, fok_oid=2).status == "pass"
        assert_check_fails(check_m4_fok_dichotomy(partial, fok_oid=2), "M4")
        # killed, but the engine left a mark on the book (the FOK's ts slot)
        s = killed.accepted[2].submission
        post = killed.post_snapshots[s]
        bids = np.array(post.bids, copy=True)
        bids[0, BSEC] = bids[0, BSEC] + 1
        bad = dataclasses.replace(killed, post_snapshots=tuple(
            dataclasses.replace(post, bids=bids) if i == s else p
            for i, p in enumerate(killed.post_snapshots)))
        assert_check_fails(check_m4_fok_dichotomy(bad, fok_oid=2), "M4")

    def test_m5_whole_vs_split_real_passes_and_short_split_fails(self):
        rest = [dict(side=S, price_eur=50.0, qty_mw=1.0, agent_id=1),
                dict(side=S, price_eur=51.0, qty_mw=1.0, agent_id=2)]
        whole = build_record(_tape(rest + [dict(side=B, price_eur=51.0, qty_mw=1.6, agent_id=3)]))
        split = build_record(_tape(rest + [dict(side=B, price_eur=51.0, qty_mw=0.8, agent_id=3),
                                           dict(side=B, price_eur=51.0, qty_mw=0.8, agent_id=3)]))
        short = build_record(_tape(rest + [dict(side=B, price_eur=51.0, qty_mw=0.8, agent_id=3),
                                           dict(side=B, price_eur=50.0, qty_mw=0.8, agent_id=3)]))
        assert check_m5_split_aggressor(whole, split, whole_oid=3,
                                        split_oids=(3, 4)).status == "pass"
        assert_check_fails(check_m5_split_aggressor(whole, short, whole_oid=3,
                                                    split_oids=(3, 4)), "M5")

    def test_m6_self_match_relabel_passes_and_reordered_book_fails(self):
        def run(a1, a2, a3, p2=51.0):
            return build_record(_tape([
                dict(side=S, price_eur=50.0, qty_mw=1.0, agent_id=a1),
                dict(side=S, price_eur=p2, qty_mw=1.0, agent_id=a2),
                dict(side=B, price_eur=51.0, qty_mw=1.5, agent_id=a3)]))
        base = run(1, 2, 3)
        relab = run(3, 2, 3)            # buyer 3 now owns maker 1: a self-match
        assert any(f.buyer_agent == f.seller_agent for f in relab.fills)
        assert check_m6_ownership_blindness(base, relab, agent_map={1: 3}).status == "pass"
        assert_check_fails(check_m6_ownership_blindness(base, run(1, 2, 3, p2=49.0),
                                                        agent_map={}), "M6")
