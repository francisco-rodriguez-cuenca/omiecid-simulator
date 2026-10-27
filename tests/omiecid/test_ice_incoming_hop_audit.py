"""AUDIT tests for OMIE-ICE-INCOMING-HOP-1 (Step 5, audit mode).

Independent post-implementation verification of the incoming-iceberg
total-quantity sweep, derived from the story card (agile-plan.md v5.196,
"Certification fix pass"), OMIE Rules Annex §4.4 via market-spec §6.2
item 7 / OQ-31 Reading B, and the domain analyst's Step-2 ruling
(sprint-status.md):

  * an incoming ICE aggresses with its TOTAL quantity (not its peak);
  * one trade row per resting counter bid -- each resting NON order and
    each resting iceberg INSTANCE (slice) is a separate counter bid with
    its own creation time; an incoming peak boundary never splits a row;
  * the resting entry is passive and sets the price (a resting ICEwPI's
    drifted price); the incoming OID is the aggressor on every row;
  * the remainder R > 0 rests as an ICE showing min(peak, R) at its limit,
    stamped (order.timestamp, 0), registered with total R, slice_count 0,
    never crossing the book and never treated as IOC (no cancel, it
    behaves as a normal resting iceberg afterwards);
  * an iceberg swept whole takes no table slot and never sets
    ``truncated`` (card A3).

The locked file ``tests/omiecid/test_ice_incoming_hop.py`` was NOT read
(audit-mode isolation). Incidental constants are varied on purpose
(OIDs 1-6 digits, both sides, off-round prices, peak/total non-multiples)
per the "locked tests that hold a fixture constant mask defects" lesson.

Device policy: CPU-hermetic (conftest pins the cpu platform).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import enable_x64
import numpy as np
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, _submit_order
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_PEAK_QTY,
    COL_REMAINING_HIDDEN,
    COL_SLICE_COUNT,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.order_validator import validate_order

# ``submit_order`` drives the module-level jitted ``_submit_order``.
pytestmark = [pytest.mark.jit]

_CID = 52
_BUY = int(Side.BUY)
_SELL = int(Side.SELL)
_NON = ExecRestriction.NON
_ICE = ExecRestriction.ICE

# jaxob layouts: book rows (P, Q, OID, TID, SEC, NSEC);
# trade rows (P, Q, PASS_OID, AGRS_OID, SEC, NSEC, PASS_TID, AGRS_TID).
_BP, _BQ, _BOID, _BTID, _BSEC, _BNSEC = 0, 1, 2, 3, 4, 5
_P, _Q, _PASS, _AGRS, _PASS_TID, _AGRS_TID = 0, 1, 2, 3, 6, 7

_QTY_ABS = 1e-4
_SETTLE_ABS = 0.02


def _contract() -> Contract:
    return Contract(
        contract_id=_CID,
        delivery_start_minutes=14 * 60 + 45,
        delivery_end_minutes=15 * 60,
        gct_minutes=14 * 60 + 45 - 60,
        status=ContractStatus.OPEN,
    )


@pytest.fixture(scope="module")
def pc():
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def book(pc):
    return ContractBook(_contract(), pc, OMIEMarketConfig())


@pytest.fixture(scope="module")
def tiny_book(pc):
    return ContractBook(_contract(), pc, OMIEMarketConfig(max_icebergs=1))


def _mk(pc, side, price, qty, oid, ts, er=_NON, peak=0.0, ppd=0.0, agent=None):
    if ppd != 0.0:
        # §13.6 PPD sign: BUY drifts down, SELL drifts up.
        assert (ppd < 0.0) if side == _BUY else (ppd > 0.0)
    o = make_order(
        contract_id=_CID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=oid if agent is None else agent, timestamp=ts,
        exec_restriction=er, peak_qty=peak, ppd=ppd,
    )
    assert bool(validate_order(o, pc, True)), f"fixture bug: invalid {o}"
    return o


def _off(pc) -> int:
    return int(round(-pc.price_min / pc.price_tick))


def _p(pc, eur) -> int:
    return int(round(eur / pc.price_tick)) + _off(pc)


def _q(pc, mw) -> int:
    return int(round(mw / pc.qty_tick))


def _new_rows(before, after) -> np.ndarray:
    b, a = np.asarray(before), np.asarray(after)
    return a[np.any(b != a, axis=1)]


def _side_arr(state, side):
    return np.asarray(state.lob.bids if side == _BUY else state.lob.asks)


def _rows_of(state, side, oid):
    arr = _side_arr(state, side)
    return arr[arr[:, _BOID] == oid]


def _meta_row(state, oid):
    meta = np.asarray(state.iceberg_meta)[: int(state.n_active)]
    hit = meta[meta[:, COL_ORDER_ID] == oid]
    return hit[0] if hit.shape[0] else None


def _best(state, side):
    arr = _side_arr(state, side)
    live = arr[arr[:, _BOID] > 0]
    if live.shape[0] == 0:
        return None
    return int(live[:, _BP].max() if side == _BUY else live[:, _BP].min())


def _assert_uncrossed(state):
    bb, ba = _best(state, _BUY), _best(state, _SELL)
    if bb is not None and ba is not None:
        assert bb < ba, f"book crossed after submit: bid {bb} >= ask {ba}"


def _submit(book, state, order):
    before = state.lob.trades
    s2, r = book.submit_order(state, order)
    assert bool(r.accepted)
    rows = _new_rows(before, s2.lob.trades)
    return s2, r, rows


def _assert_tape(pc, rows, expected, aggressor_oid, aggressor_tid):
    """``expected`` = list of (price EUR, qty MW, passive oid) in tape order."""
    got = [(int(r[_P]), abs(int(r[_Q])), int(r[_PASS])) for r in rows]
    want = [(_p(pc, p), _q(pc, q), po) for p, q, po in expected]
    assert got == want, f"tape rows {got} != expected {want}"
    assert np.all(rows[:, _AGRS] == aggressor_oid), (
        f"incoming oid {aggressor_oid} not the aggressor on every row: "
        f"{rows[:, _AGRS]}"
    )
    assert np.all(rows[:, _AGRS_TID] == aggressor_tid)
    assert not np.any(rows[:, _PASS] == aggressor_oid)


def _assert_report_matches(pc, r, expected):
    qty = sum(q for _, q, _ in expected)
    value = sum(p * q for p, q, _ in expected)
    assert bool(r.traded) == (qty > 0)
    assert float(r.trade_qty) == pytest.approx(qty, abs=_QTY_ABS)
    assert float(r.settlement_value) == pytest.approx(
        value * pc.delivery_duration_hours, abs=_SETTLE_ABS)


def _assert_remainder(pc, state, order, rest_mw, hidden_mw):
    """R rests as an ICE: one row min(peak, R) at (limit, T, 0), meta total R."""
    side, oid = int(order.side), int(order.order_id)
    rows = _rows_of(state, side, oid)
    assert rows.shape[0] == 1, f"expected one visible slice of {oid}: {rows}"
    row = rows[0]
    assert int(row[_BP]) == _p(pc, float(order.price))
    assert int(row[_BQ]) == _q(pc, rest_mw)
    assert int(row[_BSEC]) == int(order.timestamp)
    assert int(row[_BNSEC]) == 0
    meta = _meta_row(state, oid)
    assert meta is not None, "remainder not registered as an iceberg"
    assert int(meta[COL_REMAINING_HIDDEN]) == _q(pc, hidden_mw)
    assert int(meta[COL_PEAK_QTY]) == _q(pc, float(order.peak_qty))
    assert int(meta[COL_SLICE_COUNT]) == 0


# ---------------------------------------------------------------------------
# A. Total-quantity sweep over a mixed ladder, one row per counter bid
# ---------------------------------------------------------------------------


class TestTotalSweepMixedLadder:

    def test_buy_sweep_mixed_ladder_one_row_per_counter_bid(self, book, pc):
        """Resting asks: NON 0.7@47.31, ICE 2.2/0.9@47.33, NON 1.4@47.36,
        NON 3.0@47.40. Incoming BUY ICE 6.1 peak 1.6 @ 47.38 sweeps its
        TOTAL: 0.7@47.31, the iceberg's three instances 0.9/0.9/0.4@47.33
        (price priority: hidden 47.33 before 47.36), 1.4@47.36 = 4.3 MW, the
        last split 0.3 + 1.1 at the incoming instance boundary 3.2.
        R = 1.8 rests showing v_1 = min(1.6 - 1.1, 1.8) = 0.5 @ 47.38,
        hidden 1.3 (OMIE-ICE-INCOMING-LADDER-1)."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _SELL, 47.31, 0.7, 4401, 1))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 47.33, 2.2, 4402, 2, _ICE, peak=0.9))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 47.36, 1.4, 4403, 3))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 47.40, 3.0, 4404, 4))
        ice = _mk(pc, _BUY, 47.38, 6.1, 91217, 5, _ICE, peak=1.6, agent=318)
        s, r, rows = _submit(book, s, ice)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): the incoming instance
        # boundaries are 1.6 and 3.2; the row 1.4 @ 47.36 (cumulative 2.9 ->
        # 4.3) crosses 3.2 and splits 0.3 + 1.1 (Rule 2 item (2)). M = 4.3,
        # k_e = 2, u = 1.1: v_1 = min(1.6 - 1.1, 1.8) = 0.5, hidden 1.3 (item (4)).
        expected = [(47.31, 0.7, 4401), (47.33, 0.9, 4402), (47.33, 0.9, 4402),
                    (47.33, 0.4, 4402), (47.36, 0.3, 4403), (47.36, 1.1, 4403)]
        _assert_tape(pc, rows, expected, 91217, 318)
        _assert_report_matches(pc, r, expected)
        _assert_remainder(pc, s, ice, rest_mw=0.5, hidden_mw=1.3)
        assert _meta_row(s, 4402) is None
        assert int(s.n_active) == 1
        assert _rows_of(s, _SELL, 4404)[0, _BQ] == _q(pc, 3.0)
        assert not bool(s.truncated)
        _assert_uncrossed(s)

    def test_sell_sweep_mirror_remainder_below_peak(self, book, pc):
        """Mirror with other constants. Resting bids: NON 1.1@62.48, ICE
        1.5/0.4@62.45, NON 0.6@62.41. Incoming SELL ICE 2.9 peak 0.7 @
        62.44: 1.1@62.48 then four instances 0.4/0.4/0.4/0.3@62.45 = 2.6,
        rows split at the incoming instance boundaries 0.7, 1.4, 2.1.
        R = 0.3: shows v_1 = min(0.7 - 0.5, 0.3) = 0.2, hidden 0.1
        (OMIE-ICE-INCOMING-LADDER-1)."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 62.48, 1.1, 77, 10))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 62.45, 1.5, 78, 11, _ICE, peak=0.4))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 62.41, 0.6, 79, 12))
        ice = _mk(pc, _SELL, 62.44, 2.9, 230051, 13, _ICE, peak=0.7, agent=4)
        s, r, rows = _submit(book, s, ice)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): incoming instance
        # boundaries 0.7, 1.4, 2.1 split the rows 1.1 (0.7 + 0.4), the 2nd
        # 0.4 (cumulative 1.1 -> 1.5: 0.3 + 0.1) and the 4th 0.4 (1.9 -> 2.3:
        # 0.2 + 0.2). M = 2.6, k_e = 3, u = 0.5: v_1 = min(0.7 - 0.5, 0.3) =
        # 0.2, hidden 0.1 (item (4)).
        expected = [(62.48, 0.7, 77), (62.48, 0.4, 77), (62.45, 0.3, 78),
                    (62.45, 0.1, 78), (62.45, 0.4, 78), (62.45, 0.2, 78),
                    (62.45, 0.2, 78), (62.45, 0.3, 78)]
        _assert_tape(pc, rows, expected, 230051, 4)
        _assert_report_matches(pc, r, expected)
        _assert_remainder(pc, s, ice, rest_mw=0.2, hidden_mw=0.1)
        assert _rows_of(s, _BUY, 79)[0, _BQ] == _q(pc, 0.6)
        assert int(s.n_active) == 1
        _assert_uncrossed(s)

    def test_incoming_peak_boundary_never_splits_a_row(self, book, pc):
        """Name obsolete since OMIE-ICE-INCOMING-LADDER-1 (kept for node-id
        stability): spec §4.2.1 Rule 2 item (2), amended 2026-10-20, ends a
        trade row where one incoming instance ends. Incoming peak 0.3
        against ONE resting NON of 2.0: the total 1.7 trades as 5 x 0.3 +
        0.2, one row per incoming instance."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 88.88, 2.0, 600001, 1))
        ice = _mk(pc, _SELL, 88.80, 1.7, 9, 2, _ICE, peak=0.3)
        s, r, rows = _submit(book, s, ice)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): the incoming peak boundary
        # now splits the row (instance boundaries 0.3, 0.6, ..., 1.5).
        _assert_tape(pc, rows, [(88.88, 0.3, 600001)] * 5 + [(88.88, 0.2, 600001)],
                     9, 9)
        assert _meta_row(s, 9) is None
        assert _rows_of(s, _SELL, 9).shape[0] == 0
        assert int(s.n_active) == 0
        assert _rows_of(s, _BUY, 600001)[0, _BQ] == _q(pc, 0.3)


# ---------------------------------------------------------------------------
# B. Same-price time priority: each resting instance is its own counter bid
# ---------------------------------------------------------------------------


class TestInstanceTimePriority:

    def test_regenerated_instance_queues_behind_later_non(self, book, pc):
        """Resting SELL ICE 2.0/1.0 @ 50.00 (A, t1) then SELL NON 1.0 @
        50.00 (B, t2). Incoming BUY ICE 3.0/2.0 @ 50.00: A's first instance,
        then B (earlier than A's regenerated instance), then A's second
        instance -- three 1.0 rows, passive A, B, A. Nothing rests."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _SELL, 50.00, 2.0, 1201, 1, _ICE, peak=1.0))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 50.00, 1.0, 1202, 2))
        ice = _mk(pc, _BUY, 50.00, 3.0, 33, 3, _ICE, peak=2.0)
        s, r, rows = _submit(book, s, ice)
        expected = [(50.00, 1.0, 1201), (50.00, 1.0, 1202), (50.00, 1.0, 1201)]
        _assert_tape(pc, rows, expected, 33, 33)
        _assert_report_matches(pc, r, expected)
        assert int(s.n_active) == 0
        assert _rows_of(s, _BUY, 33).shape[0] == 0
        assert _best(s, _SELL) is None


# ---------------------------------------------------------------------------
# C. Resting ICEwPI sets the price of every hop (drifted price), until it
#    drifts out of the incoming limit
# ---------------------------------------------------------------------------


class TestRestingIcewpiPricesHops:

    def test_buy_incoming_stops_when_resting_drifts_out(self, book, pc):
        """Resting SELL ICEwPI 3.0/1.0 @ 50.00 ppd +0.20. Incoming BUY ICE
        5.0/2.0 @ 50.30 trades 1.0@50.00, 1.0@50.20 (resting drifted
        price); the third instance @ 50.40 does not cross. R = 3.0 rests
        2.0 @ 50.30, hidden 1.0; the resting iceberg shows 1.0 @ 50.40."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _SELL, 50.00, 3.0, 31, 1, _ICE, peak=1.0, ppd=0.20))
        ice = _mk(pc, _BUY, 50.30, 5.0, 4732, 2, _ICE, peak=2.0)
        s, r, rows = _submit(book, s, ice)
        expected = [(50.00, 1.0, 31), (50.20, 1.0, 31)]
        _assert_tape(pc, rows, expected, 4732, 4732)
        _assert_report_matches(pc, r, expected)
        _assert_remainder(pc, s, ice, rest_mw=2.0, hidden_mw=1.0)
        res = _rows_of(s, _SELL, 31)
        assert res.shape[0] == 1
        assert (int(res[0, _BP]), int(res[0, _BQ])) == (_p(pc, 50.40), _q(pc, 1.0))
        assert int(_meta_row(s, 31)[COL_REMAINING_HIDDEN]) == 0
        assert int(s.n_active) == 2
        _assert_uncrossed(s)

    def test_sell_incoming_mirror(self, book, pc):
        """Resting BUY ICEwPI 2.4/0.8 @ 70.15 ppd -0.05. Incoming SELL ICE
        3.1/1.2 @ 70.08: 0.8@70.15, 0.8@70.10 (split 0.4 + 0.4 at the
        incoming instance boundary 1.2); 70.05 is below the limit. R = 1.5
        rests v_1 = 0.8 @ 70.08, hidden 0.7 (OMIE-ICE-INCOMING-LADDER-1);
        resting shows 0.8 @ 70.05."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 70.15, 2.4, 3301, 6, _ICE, peak=0.8, ppd=-0.05))
        ice = _mk(pc, _SELL, 70.08, 3.1, 815, 7, _ICE, peak=1.2, agent=27)
        s, r, rows = _submit(book, s, ice)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): the 2nd row (cumulative
        # 0.8 -> 1.6) crosses the incoming instance boundary 1.2 and splits
        # 0.4 + 0.4. M = 1.6, k_e = 1, u = 0.4: v_1 = min(1.2 - 0.4, 1.5) =
        # 0.8, hidden 0.7 (item (4)).
        expected = [(70.15, 0.8, 3301), (70.10, 0.4, 3301), (70.10, 0.4, 3301)]
        _assert_tape(pc, rows, expected, 815, 27)
        _assert_report_matches(pc, r, expected)
        _assert_remainder(pc, s, ice, rest_mw=0.8, hidden_mw=0.7)
        res = _rows_of(s, _BUY, 3301)
        assert (int(res[0, _BP]), int(res[0, _BQ])) == (_p(pc, 70.05), _q(pc, 0.8))
        _assert_uncrossed(s)

    def test_resting_non_multiple_iceberg_last_instance_partial(self, book, pc):
        """Resting SELL ICE 2.5/1.0 @ 50.07; incoming BUY ICE 4.0/3.0 @
        50.07: rows 1.0, 1.0, 0.5 all @ 50.07, passive the resting OID.
        R = 1.5 rests v_1 = min(3.0 - 2.5, 1.5) = 0.5, hidden 1.0
        (OMIE-ICE-INCOMING-LADDER-1)."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _SELL, 50.07, 2.5, 7071, 1, _ICE, peak=1.0))
        ice = _mk(pc, _BUY, 50.07, 4.0, 7072, 2, _ICE, peak=3.0)
        s, r, rows = _submit(book, s, ice)
        expected = [(50.07, 1.0, 7071), (50.07, 1.0, 7071), (50.07, 0.5, 7071)]
        _assert_tape(pc, rows, expected, 7072, 7072)
        _assert_report_matches(pc, r, expected)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): M = 2.5, P = 3.0 -> k_e = 0,
        # u = 2.5; v_1 = min(3.0 - 2.5, 1.5) = 0.5, hidden 1.0 (item (4)).
        _assert_remainder(pc, s, ice, rest_mw=0.5, hidden_mw=1.0)
        assert _meta_row(s, 7071) is None
        assert int(s.n_active) == 1


# ---------------------------------------------------------------------------
# D. The remainder is an ordinary resting ICE of total R (never an IOC)
# ---------------------------------------------------------------------------


class TestRemainderIsRestingIceberg:

    def _seed(self, book, pc):
        """BUY ICE 5.0/2.0 @ 39.95 sweeps 1.3 @ 39.90; R = 3.7 shows
        v_1 = min(2.0 - 1.3, 3.7) = 0.7, hidden 3.0
        (OMIE-ICE-INCOMING-LADDER-1)."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _SELL, 39.90, 1.3, 501, 1))
        ice = _mk(pc, _BUY, 39.95, 5.0, 6020, 2, _ICE, peak=2.0)
        s, r, rows = _submit(book, s, ice)
        _assert_tape(pc, rows, [(39.90, 1.3, 501)], 6020, 6020)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): M = 1.3, P = 2.0 -> k_e = 0,
        # u = 1.3; v_1 = min(0.7, 3.7) = 0.7 visible, 3.0 hidden (item (4)).
        _assert_remainder(pc, s, ice, rest_mw=0.7, hidden_mw=3.0)
        return s

    def test_remainder_total_is_r_and_drains_as_passive(self, book, pc):
        """A later SELL NON 4.2 @ 39.95 takes 0.7 (the shown rest of
        instance 0), then 2.0 and 1.0 (the regenerated instances) with the
        remainder passive: exactly R = 3.7 (not the original 5.0). The NON
        rests its own 0.5; the iceberg is cleaned up."""
        s = self._seed(book, pc)
        s, r, rows = _submit(book, s, _mk(pc, _SELL, 39.95, 4.2, 777, 3))
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): the remainder shows
        # v_1 = 0.7 first, then regenerates full instances of 2.0 and 1.0.
        _assert_tape(pc, rows, [(39.95, 0.7, 6020), (39.95, 2.0, 6020),
                                (39.95, 1.0, 6020)], 777, 777)
        assert float(r.trade_qty) == pytest.approx(3.7, abs=_QTY_ABS)
        assert _meta_row(s, 6020) is None
        assert int(s.n_active) == 0
        assert _rows_of(s, _BUY, 6020).shape[0] == 0
        assert int(_rows_of(s, _SELL, 777)[0, _BQ]) == _q(pc, 0.5)

    def test_remainder_cancellable_like_any_iceberg(self, book, pc):
        """cancel_order removes the visible slice and the meta row."""
        s = self._seed(book, pc)
        s = book.cancel_order(s, 6020, _BUY)
        assert _rows_of(s, _BUY, 6020).shape[0] == 0
        assert _meta_row(s, 6020) is None
        assert int(s.n_active) == 0

    def test_non_crossing_entry_rests_whole_iceberg(self, book, pc):
        """No counter bid crosses: no trade, R = total; shows min(peak, R)
        at (T, 0), hidden total - peak, slice_count 0."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 30.01, 1.0, 12, 1))
        ice = _mk(pc, _SELL, 30.02, 4.4, 123456, 9, _ICE, peak=1.1)
        s, r, rows = _submit(book, s, ice)
        assert rows.shape[0] == 0
        assert not bool(r.traded)
        assert float(r.trade_qty) == 0.0
        _assert_remainder(pc, s, ice, rest_mw=1.1, hidden_mw=3.3)
        _assert_uncrossed(s)


# ---------------------------------------------------------------------------
# E. Iceberg table interaction (card A3)
# ---------------------------------------------------------------------------


class TestTableSlotOnlyForRemainder:

    def _full_table(self, tiny_book, pc):
        s = tiny_book.init()
        s, _ = tiny_book.submit_order(s, _mk(pc, _SELL, 80.00, 3.0, 11, 1, _ICE, peak=1.0))
        s, _ = tiny_book.submit_order(s, _mk(pc, _BUY, 60.00, 2.0, 12, 2))
        assert int(s.n_active) == 1 and not bool(s.truncated)
        return s

    def test_swept_whole_on_full_table_takes_no_slot(self, tiny_book, pc):
        s = self._full_table(tiny_book, pc)
        ice = _mk(pc, _SELL, 59.00, 1.5, 13, 3, _ICE, peak=0.5)
        s, r, rows = _submit(tiny_book, s, ice)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): one row per incoming
        # instance, 3 x 0.5 (boundaries 0.5, 1.0).
        _assert_tape(pc, rows, [(60.00, 0.5, 12)] * 3, 13, 13)
        assert not bool(s.truncated)
        assert int(s.n_active) == 1
        assert _meta_row(s, 11) is not None
        assert _rows_of(s, _SELL, 13).shape[0] == 0

    def test_remainder_on_full_table_flags_and_rests_slice(self, tiny_book, pc):
        """Sweep 2.0 @ 60.00 is reported whole; R = 1.0 cannot register:
        ``truncated`` set; min(0.5, 1.0) = 0.5 rests unregistered @ 59.00."""
        s = self._full_table(tiny_book, pc)
        ice = _mk(pc, _SELL, 59.00, 3.0, 14, 3, _ICE, peak=0.5)
        s, r, rows = _submit(tiny_book, s, ice)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): one row per incoming
        # instance, 4 x 0.5; display unchanged (M = 2.0, u = 0, v_1 = 0.5).
        _assert_tape(pc, rows, [(60.00, 0.5, 12)] * 4, 14, 14)
        assert float(r.trade_qty) == pytest.approx(2.0, abs=_QTY_ABS)
        assert bool(s.truncated)
        assert _meta_row(s, 14) is None
        rest = _rows_of(s, _SELL, 14)
        assert rest.shape[0] == 1
        assert (int(rest[0, _BP]), int(rest[0, _BQ])) == (_p(pc, 59.00), _q(pc, 0.5))
        _assert_uncrossed(s)


# ---------------------------------------------------------------------------
# F. Shared hoisted sweep under vmap (card A1): NON lanes are unaffected
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_vmap_mixed_ice_and_non_lanes_match_scalar_calls(book, pc):
    """slow: fresh vmap compile of _submit_order (all switch arms traced).

    Under vmap the IOC|ICE sweep cond becomes a select, so every lane runs
    the sweep. A NON lane must still produce exactly the NON outcome and an
    ICE lane the sweep outcome, identical to unbatched calls."""
    mc = OMIEMarketConfig()
    s = book.init()
    s, _ = book.submit_order(s, _mk(pc, _SELL, 47.31, 0.7, 4401, 1))
    s, _ = book.submit_order(s, _mk(pc, _SELL, 47.33, 2.2, 4402, 2, _ICE, peak=0.9))
    orders = [
        _mk(pc, _BUY, 47.38, 6.1, 91217, 5, _ICE, peak=1.6),
        _mk(pc, _BUY, 47.38, 6.1, 91217, 5, _NON),
    ]
    scalar = [book.submit_order(s, o) for o in orders]

    stack = lambda *xs: jnp.stack([jnp.asarray(x) for x in xs])  # noqa: E731
    bs = jax.tree.map(stack, s, s)
    bo = jax.tree.map(stack, *orders)
    # The batched env path runs under an outer ``enable_x64`` (extract_fills'
    # scoped int64 sums do not lower under vmap with x64 off; pre-existing,
    # see trade_extraction.py / omie_states.py), so mirror it here.
    with enable_x64():
        bst, bres = jax.vmap(lambda st, o: _submit_order(st, o, pc, mc))(bs, bo)

    for lane, (st1, r1) in enumerate(scalar):
        lane_st = jax.tree.map(lambda x: x[lane], bst)
        lane_r = jax.tree.map(lambda x: x[lane], bres)
        for name in ("bids", "asks", "trades"):
            np.testing.assert_array_equal(
                np.asarray(getattr(lane_st.lob, name)),
                np.asarray(getattr(st1.lob, name)), err_msg=f"lane {lane} {name}")
        np.testing.assert_array_equal(np.asarray(lane_st.iceberg_meta),
                                      np.asarray(st1.iceberg_meta))
        assert int(lane_st.n_active) == int(st1.n_active)
        assert int(lane_st.ts_counter) == int(st1.ts_counter)
        assert float(lane_r.trade_qty) == pytest.approx(float(r1.trade_qty))
    # The two lanes differ: the NON rests 6.1 - 2.9 = 3.2 whole; the ICE
    # sweeps M = 2.9 at P = 1.6 (k_e = 1, u = 1.3) and shows v_1 =
    # min(1.6 - 1.3, 3.2) = 0.3.
    assert int(_rows_of(scalar[1][0], _BUY, 91217)[0, _BQ]) == _q(pc, 3.2)
    # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): display v_1 = 0.3, not 1.6.
    assert int(_rows_of(scalar[0][0], _BUY, 91217)[0, _BQ]) == _q(pc, 0.3)
