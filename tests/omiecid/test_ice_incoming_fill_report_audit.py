"""AUDIT tests for OMIE-ICE-INCOMING-FILL-REPORT-1 (Step 5, audit mode).

Independent post-implementation verification that an INCOMING iceberg's
``SubmitResult`` reports EVERY fill written to the trades tape during the
``submit_order`` call — including the fills of its further slices matched
inside ``IcebergManager.regenerate_slices`` — and nothing else.

Derived independently from the story card (agile-plan.md
§OMIE-ICE-INCOMING-FILL-REPORT-1), market-spec §5.2 / §13.6 and the public
``ContractBook.submit_order`` surface. The locked test file for this story
was NOT read (audit-mode isolation).

Probe strategy (the "locked tests that hold a fixture constant mask
defects" lesson): every fixture varies OIDs (3-5 digits), sides,
peak/total ratios (incl. non-multiples), off-round prices, and ppd.

Contract locked here:
  * Conservation: reported ``trade_qty`` / ``settlement_value`` /
    ``trade_price`` equal the aggregates of the tape rows written by the
    call; every such row names the incoming order as aggressor or passive
    (no foreign rows); and the reported quantity equals the incoming
    iceberg's total minus its residual (visible + hidden) in the book.
  * BUY/SELL mirror symmetry.
  * NON / IOC / FOK aggressor paths against resting icebergs unchanged
    (report == tape diff).
  * Incoming ICE / ICEwPI (OMIE-ICE-INCOMING-HOP-1, Rules Annex §4.4):
    the incoming iceberg aggresses with its TOTAL quantity in one sweep,
    one trade per resting counter bid; the remainder R rests showing
    min(peak, R) at its limit with slice_count 0. An incoming ICEwPI is
    swept at its BASE limit (A4 modelling choice; no drift while
    aggressing).
  * Overflowed iceberg table: the whole sweep is reported (it happens
    before registration); the unregistered remainder rests as
    min(peak, R) with its hidden part dropped (A3).

Every tape stays far below NTRADE_CAP=100 (fresh ``book.init()`` per test).
Every constructor passes ``validate_order``; §13.6 PPD sign enforced in
the fixture helper. Settlement tolerance ``abs=0.02`` EUR (float32 decode).

Device policy: CPU-hermetic (conftest pins the cpu platform).
"""

from __future__ import annotations

import numpy as np
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, _submit_order
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_REMAINING_HIDDEN,
    COL_SLICE_COUNT,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.order_validator import validate_order

# ``submit_order`` drives the module-level jitted ``_submit_order``.
pytestmark = [pytest.mark.jit]

_CID = 37
_BUY = int(Side.BUY)
_SELL = int(Side.SELL)
_NON = ExecRestriction.NON
_IOC = ExecRestriction.IOC
_FOK = ExecRestriction.FOK
_ICE = ExecRestriction.ICE

# Trades-tape column layout (jaxob TradesFeat): P, Q, PASS_OID, AGRS_OID, ...
_P, _Q, _PASS, _AGRS = 0, 1, 2, 3
_NTRADE_CAP = 100

_SETTLE_ABS = 0.02
_QTY_ABS = 1e-4
_PRICE_ABS = 0.0051  # decoded VWAP is rounded to the 0.01 tick


def _contract() -> Contract:
    return Contract(
        contract_id=_CID,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


@pytest.fixture(scope="module")
def pc():
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def abook(pc):
    return ContractBook(_contract(), pc, OMIEMarketConfig())


@pytest.fixture(scope="module")
def tiny_table_book(pc):
    return ContractBook(_contract(), pc, OMIEMarketConfig(max_icebergs=1))


def _mk(pc, side, price, qty, oid, ts, er=_NON, peak=0.0, ppd=0.0):
    """Build an order and REQUIRE it to pass ``validate_order``.

    §13.6 PPD sign (not validator-enforced, OQ-23 (a)) is a fixture
    precondition: ICEwPI BUY ppd < 0, ICEwPI SELL ppd > 0.
    """
    if ppd != 0.0:
        if side == _BUY:
            assert ppd < 0.0, "§13.6: ICEwPI BUY requires ppd < 0"
        else:
            assert ppd > 0.0, "§13.6: ICEwPI SELL requires ppd > 0"
    o = make_order(
        contract_id=_CID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=oid, timestamp=ts,
        exec_restriction=er, peak_qty=peak, ppd=ppd,
    )
    assert bool(validate_order(o, pc, True)), f"fixture bug: invalid {o}"
    return o


# ---------------------------------------------------------------------------
# Independent tape / book probes (raw jaxob int domain)
# ---------------------------------------------------------------------------


def _price_offset(pc) -> int:
    return int(round(-pc.price_min / pc.price_tick))


def _new_rows(before, after) -> np.ndarray:
    b = np.asarray(before)
    a = np.asarray(after)
    return a[np.any(b != a, axis=1)]


def _tape_aggregates(rows, pc):
    """(qty MW, settlement EUR, exact VWAP EUR/MWh) of tape rows."""
    q_ticks = np.abs(rows[:, _Q].astype(np.int64))
    p_centered = rows[:, _P].astype(np.int64) - _price_offset(pc)
    qty_mw = float(q_ticks.sum()) * pc.qty_tick
    value = float((p_centered * q_ticks).sum()) * pc.price_tick * pc.qty_tick
    settle = value * pc.delivery_duration_hours
    vwap = value / qty_mw if qty_mw > 0 else 0.0
    return qty_mw, settle, vwap


def _visible_mw(state, side, oid, pc) -> float:
    arr = np.asarray(state.lob.bids if side == _BUY else state.lob.asks)
    hit = arr[arr[:, 2] == oid]
    return float(hit[:, 1].sum()) * pc.qty_tick


def _visible_rows(state, side, oid):
    arr = np.asarray(state.lob.bids if side == _BUY else state.lob.asks)
    return arr[arr[:, 2] == oid]


def _meta_row(state, oid):
    meta = np.asarray(state.iceberg_meta)[: int(state.n_active)]
    hit = meta[meta[:, COL_ORDER_ID] == oid]
    return hit[0] if hit.shape[0] else None


def _hidden_mw(state, oid, pc) -> float:
    row = _meta_row(state, oid)
    return 0.0 if row is None else float(row[COL_REMAINING_HIDDEN]) * pc.qty_tick


def _dec(pc, price_int) -> float:
    return (int(price_int) - _price_offset(pc)) * pc.price_tick


def _submit_and_check_conservation(book, pc, state, order):
    """Submit ``order`` and assert the full conservation contract.

    Returns (new_state, result, new_tape_rows).
    """
    before = state.lob.trades
    state2, r = book.submit_order(state, order)
    rows = _new_rows(before, state2.lob.trades)
    assert rows.shape[0] < _NTRADE_CAP, "fixture bug: tape near saturation"
    assert bool(r.accepted)

    oid = int(order.order_id)
    own = (rows[:, _PASS] == oid) | (rows[:, _AGRS] == oid)
    assert bool(own.all()), (
        f"foreign tape rows written during submit of oid {oid}: {rows[~own]}"
    )

    q, s, vwap = _tape_aggregates(rows, pc)
    assert bool(r.traded) == (rows.shape[0] > 0)
    assert float(r.trade_qty) == pytest.approx(q, abs=_QTY_ABS), (
        f"reported {float(r.trade_qty)} MW vs tape {q} MW ({rows.shape[0]} rows)"
    )
    assert float(r.settlement_value) == pytest.approx(s, abs=_SETTLE_ABS)
    if q > 0:
        assert float(r.trade_price) == pytest.approx(vwap, abs=_PRICE_ABS)
    else:
        assert float(r.trade_price) == 0.0
        assert float(r.settlement_value) == 0.0
    return state2, r, rows


def _assert_incoming_ice_mass_balance(state, order, r, pc):
    """Filled qty == total - residual(visible + hidden) of the incoming ICE."""
    side = int(order.side)
    oid = int(order.order_id)
    residual = _visible_mw(state, side, oid, pc) + _hidden_mw(state, oid, pc)
    assert float(r.trade_qty) == pytest.approx(
        float(order.quantity) - residual, abs=_QTY_ABS
    ), (
        f"mass balance: total {float(order.quantity)} - residual {residual} "
        f"!= reported {float(r.trade_qty)}"
    )


# ---------------------------------------------------------------------------
# A. Incoming-ICE multi-slice reports (hand-derived values)
# ---------------------------------------------------------------------------


class TestIncomingIceMultiSliceReport:

    def test_non_multiple_total_fully_filled_reports_whole_order(self, abook, pc):
        """ICE BUY 2.3 peak 0.6 vs NON SELL 5.0 @ 33.47: the total 2.3
        sweeps against the one counter bid 1893, one row per incoming
        instance (OMIE-ICE-INCOMING-LADDER-1, Rule 2 item (2)); the iceberg
        is fully consumed and cleaned up."""
        s = abook.init()
        s, _ = abook.submit_order(s, _mk(pc, _SELL, 33.47, 5.0, 1893, 1))
        ice = _mk(pc, _BUY, 33.47, 2.3, 55120, 2, _ICE, peak=0.6)
        s, r, rows = _submit_and_check_conservation(abook, pc, s, ice)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): the one counter bid's row
        # of 2.3 splits at the incoming instance boundaries 0.6, 1.2, 1.8 into
        # 0.6 + 0.6 + 0.6 + 0.5 (spec §4.2.1 Rule 2 item (2)).
        assert rows.shape[0] == 4
        assert np.abs(rows[:, _Q]).tolist() == [6, 6, 6, 5]
        assert float(r.trade_qty) == pytest.approx(2.3, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(33.47, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(
            2.3 * 33.47 * 0.25, abs=_SETTLE_ABS)
        assert _meta_row(s, 55120) is None
        assert int(s.n_active) == 0
        assert _visible_rows(s, _BUY, 55120).shape[0] == 0
        assert _visible_mw(s, _SELL, 1893, pc) == pytest.approx(2.7, abs=_QTY_ABS)
        _assert_incoming_ice_mass_balance(s, ice, r, pc)

    def test_multi_level_walk_reports_vwap_across_slices(self, abook, pc):
        """ICE BUY 3.0 @ 41.07 peak 0.5 walks 0.8@41.02 + 1.1@41.05 (1.9 MW
        crossable; 2.0@41.09 out of limit). The total sweeps both levels
        (one trade per (counter bid, incoming instance)); the remainder 1.1
        rests showing v_1 = min(peak - u, 1.1) = 0.1 @ 41.07 with 1.0 hidden
        and slice_count 0 (OMIE-ICE-INCOMING-LADDER-1; u = 1.9 - 3 * 0.5)."""
        s = abook.init()
        s, _ = abook.submit_order(s, _mk(pc, _SELL, 41.02, 0.8, 301, 1))
        s, _ = abook.submit_order(s, _mk(pc, _SELL, 41.05, 1.1, 302, 2))
        s, _ = abook.submit_order(s, _mk(pc, _SELL, 41.09, 2.0, 303, 3))
        ice = _mk(pc, _BUY, 41.07, 3.0, 8842, 4, _ICE, peak=0.5)
        s, r, _ = _submit_and_check_conservation(abook, pc, s, ice)
        value = 0.8 * 41.02 + 1.1 * 41.05
        assert float(r.trade_qty) == pytest.approx(1.9, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(41.04, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(
            value * 0.25, abs=_SETTLE_ABS)
        vis = _visible_rows(s, _BUY, 8842)
        assert vis.shape[0] == 1
        assert _dec(pc, vis[0, 0]) == pytest.approx(41.07, abs=1e-6)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): M = 19, P = 5 -> k_e = 3,
        # u = 4; v_1 = min(5 - 4, 11) = 1 tick visible, 10 hidden (item (4)).
        assert int(vis[0, 1]) == 1
        meta = _meta_row(s, 8842)
        assert meta is not None
        assert int(meta[COL_REMAINING_HIDDEN]) == 10
        assert int(meta[COL_SLICE_COUNT]) == 0
        assert _visible_mw(s, _SELL, 303, pc) == pytest.approx(2.0, abs=_QTY_ABS)
        _assert_incoming_ice_mass_balance(s, ice, r, pc)

    def test_second_incoming_ice_reports_only_its_own_call(self, abook, pc):
        """Two consecutive incoming ICE BUYs against one NON SELL 6.0 @ 62.31:
        A (2.4, peak 0.8) takes 2.4 in one trade; B (5.0, peak 1.3) sweeps
        the remaining 3.6, then its remainder 1.4 rests showing v_1 =
        min(1.3 - 1.0, 1.4) = 0.3 with 1.1 hidden (OMIE-ICE-INCOMING-LADDER-1). B's report excludes A's rows;
        A + B reports sum to the whole tape."""
        s0 = abook.init()
        s, _ = abook.submit_order(s0, _mk(pc, _SELL, 62.31, 6.0, 9001, 1))
        a = _mk(pc, _BUY, 62.31, 2.4, 9101, 2, _ICE, peak=0.8)
        s, ra, _ = _submit_and_check_conservation(abook, pc, s, a)
        b = _mk(pc, _BUY, 62.40, 5.0, 9102, 3, _ICE, peak=1.3)
        s, rb, _ = _submit_and_check_conservation(abook, pc, s, b)
        assert float(ra.trade_qty) == pytest.approx(2.4, abs=_QTY_ABS)
        assert float(rb.trade_qty) == pytest.approx(3.6, abs=_QTY_ABS)
        assert float(rb.trade_price) == pytest.approx(62.31, abs=_PRICE_ABS)
        assert float(rb.settlement_value) == pytest.approx(
            3.6 * 62.31 * 0.25, abs=_SETTLE_ABS)
        whole_q, whole_s, _ = _tape_aggregates(
            _new_rows(s0.lob.trades, s.lob.trades), pc)
        assert float(ra.trade_qty) + float(rb.trade_qty) == pytest.approx(
            whole_q, abs=_QTY_ABS)
        assert float(ra.settlement_value) + float(rb.settlement_value) == (
            pytest.approx(whole_s, abs=_SETTLE_ABS))
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): M = 36, P = 13 -> k_e = 2,
        # u = 10; v_1 = min(13 - 10, 14) = 3 ticks (0.3 MW) visible, 11 hidden.
        assert _visible_mw(s, _BUY, 9102, pc) == pytest.approx(0.3, abs=_QTY_ABS)
        assert int(_meta_row(s, 9102)[COL_REMAINING_HIDDEN]) == 11
        _assert_incoming_ice_mass_balance(s, b, rb, pc)

    def test_incoming_ice_vs_resting_ice_mutual_cascade(self, abook, pc):
        """Incoming ICE SELL 4.4 @ 38.07 peak 0.9 vs resting ICE BUY 3.1 @
        38.12 peak 0.8: both regenerate; the whole 3.1 MW of the resting
        iceberg trades in the one call (mixed passive prices 38.12/38.07)."""
        s = abook.init()
        s, _ = abook.submit_order(
            s, _mk(pc, _BUY, 38.12, 3.1, 22019, 1, _ICE, peak=0.8))
        ice = _mk(pc, _SELL, 38.07, 4.4, 22020, 2, _ICE, peak=0.9)
        s, r, rows = _submit_and_check_conservation(abook, pc, s, ice)
        assert float(r.trade_qty) == pytest.approx(3.1, abs=_QTY_ABS)
        prices = {round(_dec(pc, p), 2) for p in rows[:, _P]}
        assert prices <= {38.12, 38.07}
        assert 38.07 <= float(r.trade_price) <= 38.12
        assert _meta_row(s, 22019) is None
        assert _visible_rows(s, _BUY, 22019).shape[0] == 0
        _assert_incoming_ice_mass_balance(s, ice, r, pc)
        assert float(r.trade_qty) == pytest.approx(3.1, abs=_QTY_ABS)

    def test_peak_equal_total_single_slice(self, abook, pc):
        """Degenerate ICE (peak == total 1.7) vs NON SELL 4.0 @ 27.83."""
        s = abook.init()
        s, _ = abook.submit_order(s, _mk(pc, _SELL, 27.83, 4.0, 4410, 1))
        ice = _mk(pc, _BUY, 27.83, 1.7, 4411, 2, _ICE, peak=1.7)
        s, r, rows = _submit_and_check_conservation(abook, pc, s, ice)
        assert rows.shape[0] == 1
        assert float(r.trade_qty) == pytest.approx(1.7, abs=_QTY_ABS)
        assert float(r.settlement_value) == pytest.approx(
            1.7 * 27.83 * 0.25, abs=_SETTLE_ABS)
        assert int(s.n_active) == 0

    def test_non_crossing_ice_reports_nothing(self, abook, pc):
        s = abook.init()
        s, _ = abook.submit_order(s, _mk(pc, _SELL, 58.44, 3.0, 7120, 1))
        ice = _mk(pc, _BUY, 58.43, 9.0, 7121, 2, _ICE, peak=1.2)
        s, r, rows = _submit_and_check_conservation(abook, pc, s, ice)
        assert rows.shape[0] == 0
        assert not bool(r.traded)
        assert float(r.trade_qty) == 0.0
        _assert_incoming_ice_mass_balance(s, ice, r, pc)


# ---------------------------------------------------------------------------
# B. Incoming ICEwPI (§13.6): pins the A4 modelling choice
# ---------------------------------------------------------------------------
# OMIE-ICE-INCOMING-HOP-1 (A4): an INCOMING ICEwPI is swept at its BASE limit
# with its total quantity; the price drift applies only to the slices it
# shows later while resting. The test names below predate this and are
# kept for node-id stability; they no longer describe the outcome.


class TestIcewpiDrift:

    def test_buy_drift_stops_when_slice_no_longer_crosses(self, abook, pc):
        """OMIE-ICE-INCOMING-LADDER-1 (spec §4.2.1 Rule 2 items (2), (4), (6),
        amended 2026-10-20). ICEwPI BUY 3.5 @ 45.20 peak 0.7 ppd -0.03 vs
        NON SELL 5.0 @ 45.14: instance limits 45.20, 45.17, 45.14, 45.11.
        Instances 0-2 cross 45.14 and each trade 0.7 @ 45.14 (one row per
        instance); instance 3 (45.11) cannot trade, so the sweep stops.
        M = 2.1, k_e = 3, u = 0: the remainder 1.4 rests as instance 3,
        showing 0.7 @ 45.11 with 0.7 hidden; 612 keeps 2.9."""
        s = abook.init()
        s, _ = abook.submit_order(s, _mk(pc, _SELL, 45.14, 5.0, 612, 1))
        ice = _mk(pc, _BUY, 45.20, 3.5, 70411, 2, _ICE, peak=0.7, ppd=-0.03)
        s, r, rows = _submit_and_check_conservation(abook, pc, s, ice)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-A): three instance rows,
        # 2.1 MW traded, remainder rests at b + 3 d = 45.11.
        assert rows.shape[0] == 3
        assert np.abs(rows[:, _Q]).tolist() == [7, 7, 7]
        assert float(r.trade_qty) == pytest.approx(2.1, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(45.14, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(
            2.1 * 45.14 * 0.25, abs=_SETTLE_ABS)  # 23.6985
        vis = _visible_rows(s, _BUY, 70411)
        assert vis.shape[0] == 1
        assert _dec(pc, vis[0, 0]) == pytest.approx(45.11, abs=1e-6)
        assert int(vis[0, 1]) == 7
        assert int(_meta_row(s, 70411)[COL_REMAINING_HIDDEN]) == 7
        assert int(s.n_active) == 1
        assert _visible_mw(s, _SELL, 612, pc) == pytest.approx(2.9, abs=_QTY_ABS)
        _assert_incoming_ice_mass_balance(s, ice, r, pc)

    def test_sell_drift_mirror(self, abook, pc):
        """OMIE-ICE-INCOMING-LADDER-1 mirror: ICEwPI SELL 3.5 @ 54.80 peak
        0.7 ppd +0.03 vs NON BUY 5.0 @ 54.86: instance limits 54.80, 54.83,
        54.86, 54.89. Instances 0-2 each trade 0.7 @ 54.86; instance 3
        (54.89) cannot trade. The remainder 1.4 rests as instance 3: 0.7 @
        54.89, 0.7 hidden; 613 keeps 2.9."""
        s = abook.init()
        s, _ = abook.submit_order(s, _mk(pc, _BUY, 54.86, 5.0, 613, 1))
        ice = _mk(pc, _SELL, 54.80, 3.5, 70412, 2, _ICE, peak=0.7, ppd=0.03)
        s, r, rows = _submit_and_check_conservation(abook, pc, s, ice)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-A): three instance rows,
        # 2.1 MW traded, remainder rests at b + 3 d = 54.89.
        assert rows.shape[0] == 3
        assert np.abs(rows[:, _Q]).tolist() == [7, 7, 7]
        assert float(r.trade_qty) == pytest.approx(2.1, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(54.86, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(
            2.1 * 54.86 * 0.25, abs=_SETTLE_ABS)  # 28.8015
        vis = _visible_rows(s, _SELL, 70412)
        assert vis.shape[0] == 1
        assert _dec(pc, vis[0, 0]) == pytest.approx(54.89, abs=1e-6)
        assert int(vis[0, 1]) == 7
        assert int(_meta_row(s, 70412)[COL_REMAINING_HIDDEN]) == 7
        assert int(s.n_active) == 1
        assert _visible_mw(s, _BUY, 613, pc) == pytest.approx(2.9, abs=_QTY_ABS)
        _assert_incoming_ice_mass_balance(s, ice, r, pc)

    def test_drift_across_ladder_takes_only_crossable_levels(self, abook, pc):
        """OMIE-ICE-INCOMING-LADDER-1 (spec §4.2.1 Rule 2 items (2), (4), (6),
        amended 2026-10-20). ICEwPI SELL 2.4 @ 71.00 peak 0.4 ppd +0.07 vs
        bids 0.6 @ 71.15 (oid 1201) and 1.0 @ 71.06 (oid 1202). Instance 0
        (limit 71.00) trades 0.4 @ 71.15; instance 1 (limit 71.07) trades the
        last 0.2 @ 71.15 and cannot reach 71.06, so the sweep stops inside
        instance 1. M = 0.6, VWAP 71.15, settlement 0.6 * 71.15 * 0.25 =
        10.6725. k_e = 1, u = 0.2: the remainder 1.8 continues instance 1 at
        71.00 + 0.07 = 71.07, showing v_1 = min(0.4 - 0.2, 1.8) = 0.2 with
        1.6 hidden and slice_count 0. 1202 keeps its 1.0."""
        s = abook.init()
        s, _ = abook.submit_order(s, _mk(pc, _BUY, 71.15, 0.6, 1201, 1))
        s, _ = abook.submit_order(s, _mk(pc, _BUY, 71.06, 1.0, 1202, 2))
        ice = _mk(pc, _SELL, 71.00, 2.4, 1299, 3, _ICE, peak=0.4, ppd=0.07)
        s, r, rows = _submit_and_check_conservation(abook, pc, s, ice)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-A): instance limits; two rows
        # against 1201 (one per incoming instance), none against 1202.
        assert rows.shape[0] == 2
        by_row = [
            (int(row[_PASS]), round(_dec(pc, row[_P]), 2), abs(int(row[_Q])))
            for row in rows
        ]
        assert by_row == [(1201, 71.15, 4), (1201, 71.15, 2)]
        assert float(r.trade_qty) == pytest.approx(0.6, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(71.15, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(
            0.6 * 71.15 * 0.25, abs=_SETTLE_ABS)  # 10.6725
        assert _visible_mw(s, _BUY, 1202, pc) == pytest.approx(1.0, abs=_QTY_ABS)
        vis = _visible_rows(s, _SELL, 1299)
        assert vis.shape[0] == 1
        assert _dec(pc, vis[0, 0]) == pytest.approx(71.07, abs=1e-6)
        assert int(vis[0, 1]) == 2
        meta = _meta_row(s, 1299)
        assert int(meta[COL_REMAINING_HIDDEN]) == 16
        assert int(meta[COL_SLICE_COUNT]) == 0
        _assert_incoming_ice_mass_balance(s, ice, r, pc)


# ---------------------------------------------------------------------------
# C. BUY/SELL mirror symmetry (generic scenarios mirrored around 50.00)
# ---------------------------------------------------------------------------

_MIRROR_C = 50.00


def _mirror_price(side, p):
    return p if side == _BUY else round(2 * _MIRROR_C - p, 2)


# Each scenario is expressed in the INCOMING-BUY orientation:
# (resting list [(price, qty, oid)], incoming (price, qty, oid, peak, ppd)).
_SCENARIOS = {
    "flat_three_slices": (
        [(47.61, 4.5, 3301)], (47.61, 6.0, 3399, 1.5, 0.0)),
    "partial_last_slice": (
        [(46.93, 2.2, 610)], (46.95, 8.1, 611, 0.9, 0.0)),
    "two_levels_ppd": (
        [(48.02, 0.9, 20417), (48.10, 1.4, 20418)],
        (48.25, 5.0, 20419, 0.6, -0.04)),
    "odd_ratio_walk": (
        [(49.37, 0.3, 812), (49.41, 0.3, 813), (49.44, 2.0, 814)],
        (49.44, 1.9, 815, 0.7, 0.0)),
}


def _run_scenario(book, pc, key, incoming_side):
    rest_side = _SELL if incoming_side == _BUY else _BUY
    sign = 1.0 if incoming_side == _BUY else -1.0
    resting, (ip, iq, ioid, ipeak, ippd) = _SCENARIOS[key]
    s = book.init()
    ts = 1
    for (p, q, oid) in resting:
        s, _ = book.submit_order(
            s, _mk(pc, rest_side, _mirror_price(incoming_side, p), q, oid, ts))
        ts += 1
    ice = _mk(pc, incoming_side, _mirror_price(incoming_side, ip), iq, ioid,
              ts, _ICE, peak=ipeak, ppd=round(sign * ippd, 2))
    s, r, rows = _submit_and_check_conservation(book, pc, s, ice)
    _assert_incoming_ice_mass_balance(s, ice, r, pc)
    return s, r, rows


class TestBuySellSymmetry:

    @pytest.mark.parametrize("key", sorted(_SCENARIOS))
    def test_mirrored_scenario_reports_mirrored_fills(self, abook, pc, key):
        _, rb, rows_b = _run_scenario(abook, pc, key, _BUY)
        _, rs, rows_s = _run_scenario(abook, pc, key, _SELL)
        assert rows_b.shape[0] == rows_s.shape[0]
        assert float(rb.trade_qty) == pytest.approx(
            float(rs.trade_qty), abs=_QTY_ABS)
        assert float(rb.trade_qty) > 0.0
        # Mirrored per-row prices: p_buy + p_sell == 2C, same quantities.
        pb = sorted(round(_dec(pc, p), 2) for p in rows_b[:, _P])
        ps = sorted((round(2 * _MIRROR_C - _dec(pc, p), 2)
                     for p in rows_s[:, _P]))
        assert pb == ps
        # Settlement mirrors: V_buy + V_sell == 2C * Q * d.
        expect_sum = 2 * _MIRROR_C * float(rb.trade_qty) * 0.25
        assert float(rb.settlement_value) + float(rs.settlement_value) == (
            pytest.approx(expect_sum, abs=2 * _SETTLE_ABS))

    def test_scenario_expected_quantities(self, abook, pc):
        """Hand-derived fill quantities per scenario (BUY orientation)."""
        expected = {
            "flat_three_slices": 4.5,   # 1.5 x 3, rests 1.5 @ 47.61
            "partial_last_slice": 2.2,  # 0.9 + 0.9 + 0.4, rests 0.5
            # peak 0.6@48.25 and s1 0.6@48.21, s2 @48.17, s3 @48.13
            # all cross 48.02/48.10 levels (2.3 MW total) -> 2.3
            "two_levels_ppd": 2.3,
            "odd_ratio_walk": 1.9,      # whole order fills (2.6 available)
        }
        for key, q in expected.items():
            _, r, _ = _run_scenario(abook, pc, key, _BUY)
            assert float(r.trade_qty) == pytest.approx(q, abs=_QTY_ABS), key


# ---------------------------------------------------------------------------
# D. NON / IOC / FOK aggressors against a resting iceberg (unaffected paths)
# ---------------------------------------------------------------------------


class TestNonIcePathsUnaffected:

    def _rest(self, book, pc):
        s = book.init()
        s, _ = book.submit_order(
            s, _mk(pc, _SELL, 71.93, 4.2, 400, 1, _ICE, peak=0.9))
        return s

    def test_non_aggressor_multi_slice(self, abook, pc):
        s = self._rest(abook, pc)
        o = _mk(pc, _BUY, 72.00, 3.0, 401, 2)
        s, r, _ = _submit_and_check_conservation(abook, pc, s, o)
        assert float(r.trade_qty) == pytest.approx(3.0, abs=_QTY_ABS)
        assert float(r.settlement_value) == pytest.approx(
            3.0 * 71.93 * 0.25, abs=_SETTLE_ABS)

    def test_ioc_aggressor_sweeps_whole_iceberg(self, abook, pc):
        s = self._rest(abook, pc)
        o = _mk(pc, _BUY, 72.00, 5.0, 402, 2, _IOC)
        s, r, _ = _submit_and_check_conservation(abook, pc, s, o)
        assert float(r.trade_qty) == pytest.approx(4.2, abs=_QTY_ABS)
        assert float(r.settlement_value) == pytest.approx(
            4.2 * 71.93 * 0.25, abs=_SETTLE_ABS)
        assert _visible_rows(s, _BUY, 402).shape[0] == 0  # IOC never rests

    def test_fok_aggressor_fills_within_hidden(self, abook, pc):
        s = self._rest(abook, pc)
        o = _mk(pc, _BUY, 72.00, 3.9, 403, 2, _FOK)
        s, r, _ = _submit_and_check_conservation(abook, pc, s, o)
        assert float(r.trade_qty) == pytest.approx(3.9, abs=_QTY_ABS)
        assert float(r.settlement_value) == pytest.approx(
            3.9 * 71.93 * 0.25, abs=_SETTLE_ABS)

    def test_fok_aggressor_killed_when_insufficient(self, abook, pc):
        s = self._rest(abook, pc)
        o = _mk(pc, _BUY, 72.00, 4.3, 404, 2, _FOK)
        s, r, rows = _submit_and_check_conservation(abook, pc, s, o)
        assert rows.shape[0] == 0
        assert not bool(r.traded)

    def test_non_hits_resting_remnant_of_previous_incoming_ice(self, abook, pc):
        """After an incoming ICE rests a remnant, a later NON SELL that
        consumes it (and its regen slices) reports the whole cascade."""
        s = abook.init()
        s, _ = abook.submit_order(s, _mk(pc, _SELL, 64.08, 1.0, 5501, 1))
        ice = _mk(pc, _BUY, 64.11, 4.0, 5502, 2, _ICE, peak=0.8)
        s, r1, _ = _submit_and_check_conservation(abook, pc, s, ice)
        assert float(r1.trade_qty) == pytest.approx(1.0, abs=_QTY_ABS)
        o = _mk(pc, _SELL, 64.05, 2.5, 5503, 3)
        s, r2, _ = _submit_and_check_conservation(abook, pc, s, o)
        assert float(r2.trade_qty) == pytest.approx(2.5, abs=_QTY_ABS)
        assert float(r2.trade_price) == pytest.approx(64.11, abs=_PRICE_ABS)
        assert _visible_mw(s, _BUY, 5502, pc) + _hidden_mw(s, 5502, pc) == (
            pytest.approx(0.5, abs=_QTY_ABS))


# ---------------------------------------------------------------------------
# E. Overflowed iceberg table: sweep reported, remainder rests unregistered
# ---------------------------------------------------------------------------


class TestOverflowReportsPeakOnly:

    @pytest.mark.slow  # slow: fresh _submit_order compile for max_icebergs=1 (~11s)
    def test_unregistered_incoming_ice_reports_peak_only(
        self, tiny_table_book, pc,
    ):
        """Name obsolete (OMIE-ICE-INCOMING-HOP-1, A3). max_icebergs=1 is
        occupied by a same-side resting ICE BUY. The incoming ICE BUY 4.0
        peak 1.1 sweeps its total BEFORE registration: 3.3 @ 31.18 drains
        3402, printed as three 1.1 rows (one per incoming instance,
        OMIE-ICE-INCOMING-LADDER-1). Its remainder 0.7 cannot register, so it rests
        unregistered as min(peak, 0.7) = 0.7 @ 31.18 with the hidden part
        dropped; truncated is set."""
        bk = tiny_table_book
        s = bk.init()
        s, _ = bk.submit_order(
            s, _mk(pc, _BUY, 29.50, 2.0, 3401, 1, _ICE, peak=0.5))
        s, _ = bk.submit_order(s, _mk(pc, _SELL, 31.18, 3.3, 3402, 2))
        assert not bool(s.truncated)
        ice = _mk(pc, _BUY, 31.18, 4.0, 3403, 3, _ICE, peak=1.1)
        s, r, rows = _submit_and_check_conservation(bk, pc, s, ice)
        assert bool(s.truncated)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): the 3.3 row splits at the
        # incoming instance boundaries 1.1 and 2.2 into 1.1 + 1.1 + 1.1.
        assert rows.shape[0] == 3
        assert float(r.trade_qty) == pytest.approx(3.3, abs=_QTY_ABS)
        assert float(r.settlement_value) == pytest.approx(
            3.3 * 31.18 * 0.25, abs=_SETTLE_ABS)  # 25.7235
        assert _meta_row(s, 3403) is None
        assert _visible_mw(s, _SELL, 3402, pc) == pytest.approx(0.0, abs=_QTY_ABS)
        vis = _visible_rows(s, _BUY, 3403)
        assert vis.shape[0] == 1
        assert _dec(pc, vis[0, 0]) == pytest.approx(31.18, abs=1e-6)
        assert _visible_mw(s, _BUY, 3403, pc) == pytest.approx(0.7, abs=_QTY_ABS)


# ---------------------------------------------------------------------------
# F. One-compile invariant: varied ICE submits add no jit cache entries
# ---------------------------------------------------------------------------


def test_varied_incoming_ice_submits_do_not_recompile(abook, pc):
    if not hasattr(_submit_order, "_cache_size"):
        pytest.skip("jaxlib does not expose _cache_size")
    s = abook.init()
    s, _ = abook.submit_order(s, _mk(pc, _SELL, 52.17, 9.0, 60001, 1))
    s, _ = abook.submit_order(
        s, _mk(pc, _BUY, 52.17, 1.6, 60002, 2, _ICE, peak=0.4))
    n0 = _submit_order._cache_size()
    s, _ = abook.submit_order(
        s, _mk(pc, _BUY, 52.30, 2.9, 60003, 3, _ICE, peak=0.3, ppd=-0.02))
    s, _ = abook.submit_order(
        s, _mk(pc, _SELL, 52.00, 0.8, 60004, 4, _ICE, peak=0.8))
    assert _submit_order._cache_size() == n0
