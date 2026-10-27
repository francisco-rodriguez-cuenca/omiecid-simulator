"""Locked behavioural tests for OMIE-ICE-INCOMING-FILL-REPORT-1.

Contract (agile-plan.md §Story OMIE-ICE-INCOMING-FILL-REPORT-1, v5.149,
"Test-first contract" row): the ``SubmitResult`` returned by
``ContractBook.submit_order`` for an INCOMING iceberg (ICE / ICEwPI) must
report ALL fills that submit produced in the same call — the peak's fills
AND the fills of every further slice of the same iceberg matched during
the regeneration cascade (and every counterparty-iceberg slice that trades
against the incoming iceberg's resting slice). A submit's fills are
everything the submit wrote to the trades tape.

Every expected value below was re-derived by hand from the regeneration
walk (ascending slot index; ``register`` appends at the tail; an ICEwPI's
k-th regenerated slice is priced at ``base + k * ppd``; trades execute at
the passive/resting price). The book post-states pinned here are the SAME
before and after the fix and double as a unit-level byte-identity guard
(card item (8)).

Settlement tolerance: ``settlement_value`` is decoded in float32
(card v5.149 tolerance note) — every settlement assertion uses
``abs=0.02`` EUR except test (3) (1.2 MW magnitude, ``abs=1e-3``).

Fixtures go through the public ``ContractBook.submit_order`` surface on a
fresh ``book.init()`` per test (every tape stays well below
``NTRADE_CAP``). Every constructor passes ``validate_order``; the §13.6
PPD sign rule (buy ppd < 0, sell ppd > 0) is asserted as a fixture
precondition. Incidental constants (OIDs, sides, prices, peak/total
ratios, ppd) are deliberately varied.

Device policy: CPU default (conftest). ``submit_order`` drives the
module-level jitted ``_submit_order`` — file-wide ``jit`` marker.

RE-BASELINED by OMIE-ICE-INCOMING-HOP-1 (S4, agile-plan v5.195, Deliverable
(2)): an incoming iceberg now aggresses with its TOTAL quantity through the
shared IOC loop before it registers (OMIE Rules Annex §4.4). Every resting
order it meets is passive and sets the price, there is one trade row per
resting counter bid, and only the remainder R registers, showing
``min(peak, R)`` at its entry stamp. The walks and post-states below are
re-derived for that order of operations. The reported totals of (1), (2),
(6b) and (7) are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_REMAINING_HIDDEN,
    COL_SLICE_COUNT,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.order_validator import validate_order

pytestmark = [pytest.mark.jit]


# ---------------------------------------------------------------------------
# Constants / tape layout
# ---------------------------------------------------------------------------

_CONTRACT_ID = 37

# trades-tape columns (jaxob TradesFeat): P=0, Q=1, PASS_OID=2, AGRS_OID=3
_T_Q = 1
_T_PASS_OID = 2
_T_AGRS_OID = 3

# resting-book row layout: [price_int, qty_ticks, oid, tid, sec, nsec]
_B_PRICE = 0
_B_QTY = 1
_B_OID = 2

BUY = int(Side.BUY)
SELL = int(Side.SELL)
ICE = ExecRestriction.ICE
NON = ExecRestriction.NON

_SETTLE_ABS = 0.02
_QTY_ABS = 1e-4
_PRICE_ABS = 1e-3


def _open_contract() -> Contract:
    return Contract(
        contract_id=_CONTRACT_ID,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


_PC = OMIEProductConfig()


@pytest.fixture(scope="module")
def book():
    """Default product (0.25 h, 0.01 EUR tick, 0.1 MW tick) + default market."""
    return ContractBook(_open_contract(), _PC, OMIEMarketConfig())


@pytest.fixture(scope="module")
def tiny_table_book():
    """Iceberg side-table of capacity 2 (test (7): register overflow)."""
    return ContractBook(_open_contract(), _PC, OMIEMarketConfig(max_icebergs=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk(side, price, qty, oid, ts, er=NON, peak=0.0, ppd=0.0):
    """Build an OMIE order and REQUIRE it to pass ``validate_order``.

    §13.6 PPD sign rule enforced here (not validator-enforced; OQ-23 (a)).
    """
    if ppd != 0.0:
        if side == BUY:
            assert ppd < 0.0, "§13.6: ICEwPI BUY requires ppd < 0"
        else:
            assert ppd > 0.0, "§13.6: ICEwPI SELL requires ppd > 0"
    order = make_order(
        contract_id=_CONTRACT_ID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=oid, timestamp=ts,
        exec_restriction=er, peak_qty=peak, ppd=ppd,
    )
    assert bool(validate_order(order, _PC, True)), (
        f"fixture bug: constructor failed validate_order: {order}"
    )
    return order


def _price_int(price_eur: float) -> int:
    return int(round((price_eur - _PC.price_min) / _PC.price_tick))


def _qty_ticks(qty_mw: float) -> int:
    return int(round(qty_mw / _PC.qty_tick))


def _run(book, priors, incoming):
    """Submit ``priors`` (all must be accepted), then ``incoming``.

    Returns ``(state_before_incoming, state_after, result)``.
    """
    state = book.init()
    for o in priors:
        state, r = book.submit_order(state, o)
        assert bool(r.accepted), f"fixture bug: prior order rejected: {o}"
    state_after, result = book.submit_order(state, incoming)
    assert bool(result.accepted), "incoming order must be accepted"
    return state, state_after, result


def _resting(state, side, oid):
    """``(price_int, qty_ticks)`` of ``oid`` resting on ``side``, or None."""
    arr = np.asarray(state.lob.bids if side == BUY else state.lob.asks)
    rows = arr[arr[:, _B_OID] == oid]
    rows = rows[rows[:, _B_QTY] > 0]
    if rows.shape[0] == 0:
        return None
    assert rows.shape[0] == 1, f"oid {oid} rests on >1 row: {rows}"
    return int(rows[0, _B_PRICE]), int(rows[0, _B_QTY])


def _live_oids(state, side):
    arr = np.asarray(state.lob.bids if side == BUY else state.lob.asks)
    return sorted(int(x) for x in arr[(arr[:, _B_OID] > 0) & (arr[:, _B_QTY] > 0), _B_OID])


def _meta_row(state, oid):
    meta = np.asarray(state.iceberg_meta)[: int(state.n_active)]
    rows = meta[meta[:, COL_ORDER_ID] == oid]
    assert rows.shape[0] == 1, f"oid {oid} not uniquely active in meta: {rows}"
    return rows[0]


def _changed_rows(state_before, state_after):
    tb = np.asarray(state_before.lob.trades)
    ta = np.asarray(state_after.lob.trades)
    return ta[np.any(tb != ta, axis=1)]


def _assert_rests(state, side, oid, price_eur, qty_mw):
    got = _resting(state, side, oid)
    assert got == (_price_int(price_eur), _qty_ticks(qty_mw)), (
        f"oid {oid}: expected resting {qty_mw} MW @ {price_eur}, "
        f"got (price_int, qty_ticks)={got}"
    )


# ---------------------------------------------------------------------------
# Scenario constructors (shared by the per-shape tests and the
# parametrized conservation test (5)). Each returns (priors, incoming).
# ---------------------------------------------------------------------------


def _scn_three_slices():
    # (1) NON SELL 30.0 @ 50.00 vs ICE BUY 30.0 @ 50.00 peak 10.0
    return (
        [_mk(SELL, 50.00, 30.0, 4127, 1)],
        _mk(BUY, 50.00, 30.0, 9316, 2, ICE, peak=10.0),
    )


def _scn_partial_last_slice():
    # (2) NON BUY 25.0 @ 61.37 vs ICE SELL 40.0 @ 61.30 peak 10.0
    return (
        [_mk(BUY, 61.37, 25.0, 2203, 3)],
        _mk(SELL, 61.30, 40.0, 7741, 5, ICE, peak=10.0),
    )


def _scn_gap_pair():
    # (3) k = ICE BUY 2.0 @ 60.05 peak 0.4 ; j = ICE SELL 1.2 @ 60.00 peak 0.6
    return (
        [_mk(BUY, 60.05, 2.0, 583201, 11, ICE, peak=0.4)],
        _mk(SELL, 60.00, 1.2, 583377, 12, ICE, peak=0.6),
    )


def _scn_icewpi_drift():
    # (4) NON BUY 1.5 @ 60.06 vs ICEwPI SELL 1.5 @ 60.00 peak 0.5 ppd +0.05
    return (
        [_mk(BUY, 60.06, 1.5, 3310, 7)],
        _mk(SELL, 60.00, 1.5, 8804, 9, ICE, peak=0.5, ppd=0.05),
    )


def _scn_mixed_ladder():
    # (5) ladder: NON SELL 4.0 @ 50.00 (101); ICE SELL 6.0 peak 2.0 @ 50.01
    # (202); NON SELL 3.0 @ 50.02 (303); incoming ICE BUY 12.0 @ 50.02
    # peak 5.0 (404).
    return (
        [
            _mk(SELL, 50.00, 4.0, 101, 1),
            _mk(SELL, 50.01, 6.0, 202, 2, ICE, peak=2.0),
            _mk(SELL, 50.02, 3.0, 303, 3),
        ],
        _mk(BUY, 50.02, 12.0, 404, 4, ICE, peak=5.0),
    )


# ---------------------------------------------------------------------------
# (1) three full slices
# ---------------------------------------------------------------------------


class TestIncomingIceMultiSliceReport:

    def test_incoming_ice_three_slices_reports_total(self, book):
        """The total 30.0 MW sweeps the 30 MW NON in one trade (one row per
        counter bid): the result must report 30.0 MW / 375.00 EUR, not the
        peak's 10."""
        priors, inc = _scn_three_slices()
        _, st, r = _run(book, priors, inc)
        assert bool(r.traded)
        assert float(r.trade_qty) == pytest.approx(30.0, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(50.00, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(375.00, abs=_SETTLE_ABS)

    def test_incoming_ice_three_slices_post_state(self, book):
        """Book-state lock (8): both sides empty, iceberg fully drained."""
        priors, inc = _scn_three_slices()
        _, st, _ = _run(book, priors, inc)
        assert _live_oids(st, BUY) == []
        assert _live_oids(st, SELL) == []
        assert int(st.n_active) == 0
        assert not bool(st.truncated)
        assert not bool(st.book_full)

    # -----------------------------------------------------------------------
    # (2) partial last slice rests
    # -----------------------------------------------------------------------

    def test_incoming_ice_partial_last_slice_rests(self, book):
        """The total 40.0 MW sweeps the 25 MW NON in one trade at the
        resting 61.37: report 25.0 MW @ 61.37, settlement
        25 * 61.37 * 0.25 = 383.5625."""
        priors, inc = _scn_partial_last_slice()
        _, st, r = _run(book, priors, inc)
        assert bool(r.traded)
        assert float(r.trade_qty) == pytest.approx(25.0, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(61.37, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(383.5625, abs=_SETTLE_ABS)

    def test_incoming_ice_partial_last_slice_post_state(self, book):
        """Book-state lock (8): R = 15.0 MW registers. OMIE-ICE-INCOMING-LADDER-1
        re-baseline: 7741 shows v_1 = min(10.0 - 5.0, 15.0) = 5.0 @ 61.30,
        hidden 10.0 MW (S4 had min(peak, R) = 10.0 visible / 5.0 hidden)."""
        priors, inc = _scn_partial_last_slice()
        _, st, _ = _run(book, priors, inc)
        assert _live_oids(st, BUY) == []
        assert _live_oids(st, SELL) == [7741]
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): M = 25.0, P = 10.0 ->
        # k_e = 2, u = 5.0; v_1 = min(10.0 - 5.0, 15.0) = 5.0 @ 61.30, hidden
        # 10.0 MW (spec §4.2.1 Rule 2 item (4)).
        _assert_rests(st, SELL, 7741, 61.30, 5.0)
        assert int(st.n_active) == 1
        assert int(_meta_row(st, 7741)[COL_REMAINING_HIDDEN]) == 100

    # -----------------------------------------------------------------------
    # (3) incoming ICE vs resting iceberg cascade (OQ-23 gap-pair shape)
    # -----------------------------------------------------------------------

    def test_incoming_ice_vs_resting_iceberg_cascade(self, book):
        """S4 re-baseline: j sells its total 1.2 MW against k's instances,
        each its own counter bid at the resting price: 3 x 0.4 @ 60.05
        (OMIE-ICE-INCOMING-LADDER-1: the middle one splits 0.2 + 0.2 at j's
        instance boundary, 4 rows), passive 583201, aggressor 583377 ->
        settlement 1.2 * 60.05 * 0.25 =
        18.015 EUR, VWAP exactly 60.05 (was 18.0075, with two hops printed
        @ 60.00 and the roles swapped)."""
        priors, inc = _scn_gap_pair()
        before, st, r = _run(book, priors, inc)
        assert bool(r.traded)
        assert float(r.trade_qty) == pytest.approx(1.2, abs=_QTY_ABS)
        assert float(r.settlement_value) == pytest.approx(18.015, abs=1e-3)
        assert float(r.trade_price) == pytest.approx(60.05, abs=_PRICE_ABS)
        changed = _changed_rows(before, st)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): j's peak 0.6 makes its
        # 2nd row (cumulative 0.4 -> 0.8) cross the instance boundary 0.6, so
        # it splits 0.2 + 0.2: rows 0.4, 0.2, 0.2, 0.4 (Rule 2 item (2)).
        assert changed[:, _T_PASS_OID].tolist() == [583201] * 4
        assert changed[:, _T_AGRS_OID].tolist() == [583377] * 4
        assert np.abs(changed[:, 1]).tolist() == [4, 2, 2, 4]

    def test_incoming_ice_vs_resting_iceberg_cascade_post_state(self, book):
        """Book-state lock (8): quiescent, uncrossed; k = 0.4 @ 60.05
        visible, 4 ticks hidden, slice_count 3; j cleaned up."""
        priors, inc = _scn_gap_pair()
        _, st, _ = _run(book, priors, inc)
        assert _live_oids(st, SELL) == []
        assert _live_oids(st, BUY) == [583201]
        _assert_rests(st, BUY, 583201, 60.05, 0.4)
        assert int(st.n_active) == 1
        row = _meta_row(st, 583201)
        assert int(row[COL_REMAINING_HIDDEN]) == 4
        assert int(row[COL_SLICE_COUNT]) == 3
        # Quiescence: every active slot has its visible peak resting.
        meta = np.asarray(st.iceberg_meta)[: int(st.n_active)]
        for oid in meta[:, COL_ORDER_ID]:
            assert (_resting(st, BUY, int(oid)) is not None
                    or _resting(st, SELL, int(oid)) is not None)

    # -----------------------------------------------------------------------
    # (4) ICEwPI drift stops crossing
    # -----------------------------------------------------------------------

    def test_incoming_icewpi_drift_stops_crossing(self, book):
        """OMIE-ICE-INCOMING-LADDER-1 re-baseline (spec §4.2.1 Rule 2 item
        (6), amended 2026-10-20): the incoming ICEwPI is a chain of
        instances of 0.5 MW with limits 60.00, 60.05, 60.10. Instances 0 and
        1 cross the bid @ 60.06 (0.5 + 0.5); instance 2 (60.10) cannot trade,
        so the sweep stops. Report 1.0 MW @ 60.06, settlement
        1.0 * 60.06 * 0.25 = 15.015."""
        priors, inc = _scn_icewpi_drift()
        _, st, r = _run(book, priors, inc)
        assert bool(r.traded)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-A): instance limits stop the
        # sweep after 1.0 MW (was 1.5 swept at the base limit).
        assert float(r.trade_qty) == pytest.approx(1.0, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(60.06, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(15.015, abs=_SETTLE_ABS)

    def test_incoming_icewpi_drift_post_state(self, book):
        """Book-state lock (8), OMIE-ICE-INCOMING-LADDER-1 re-baseline: the
        sweep stops at instance 2 (limit 60.10), M = 1.0 MW, so k_e = 2,
        u = 0 and the remainder R = 0.5 continues instance 2 at 60.00 + 2 *
        0.05 = 60.10, showing v_1 = min(0.5 - 0, 0.5) = 0.5 (Rule 2 items (4),
        (6)). 3310 keeps 0.5 @ 60.06; n_active 1 (8804 registers)."""
        priors, inc = _scn_icewpi_drift()
        _, st, _ = _run(book, priors, inc)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-A): 8804 rests 0.5 @ 60.10,
        # 3310 rests 0.5 @ 60.06, 8804 registered (hidden 0).
        assert _live_oids(st, SELL) == [8804]
        _assert_rests(st, SELL, 8804, 60.10, 0.5)
        assert _live_oids(st, BUY) == [3310]
        _assert_rests(st, BUY, 3310, 60.06, 0.5)
        assert int(st.n_active) == 1
        assert int(_meta_row(st, 8804)[COL_REMAINING_HIDDEN]) == 0

    # -----------------------------------------------------------------------
    # (5, ladder) mixed ladder with a resting iceberg re-aggressing
    # -----------------------------------------------------------------------

    def test_mixed_ladder_reports_total(self, book):
        """404 sweeps its total 12.0 MW, one row per counter bid: 4@50.00
        (101), 202's three instances 3x2@50.01, then 2@50.02 (303). Value
        600.10 -> 150.025 EUR, VWAP 50.0083 decodes to 50.01 (S4 walk; was
        150.0325, with two hops where 202 was the aggressor)."""
        priors, inc = _scn_mixed_ladder()
        _, st, r = _run(book, priors, inc)
        assert bool(r.traded)
        assert float(r.trade_qty) == pytest.approx(12.0, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(50.01, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(150.025, abs=_SETTLE_ABS)

    def test_mixed_ladder_post_state(self, book):
        """Book-state lock (8), S4 re-baseline: bids empty; 202 is drained
        (n_active 0); 303 rests 1.0 @ 50.02; 6 changed rows (5 per counter
        bid, one split at 404's instance boundary; OMIE-ICE-INCOMING-LADDER-1)
        with 404 the aggressor on every one, so no (ii) row has 202 as the aggressor
        (was: 202 1.0 @ 50.01 resting, 7 rows, 2 of them (ii))."""
        priors, inc = _scn_mixed_ladder()
        before, st, _ = _run(book, priors, inc)
        assert _live_oids(st, BUY) == []
        assert _live_oids(st, SELL) == [303]
        _assert_rests(st, SELL, 303, 50.02, 1.0)
        assert int(st.n_active) == 0
        changed = _changed_rows(before, st)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): 404's peak 5.0 sets
        # instance boundaries at 5.0 and 10.0; the row 2.0 @ 50.01 (cumulative
        # 4.0 -> 6.0) crosses 5.0 and splits 1.0 + 1.0, so 6 rows, not 5
        # (Rule 2 item (2)). The row 2.0 @ 50.01 ending at 10.0 is not split.
        assert changed.shape[0] == 6
        assert changed[:, _T_AGRS_OID].tolist() == [404] * 6
        assert np.abs(changed[:, 1]).tolist() == [40, 10, 10, 20, 20, 20]
        ii = changed[(changed[:, _T_AGRS_OID] == 202)
                     & (changed[:, _T_PASS_OID] == 404)]
        assert ii.shape[0] == 0


# ---------------------------------------------------------------------------
# (5) conservation: reported qty == own-OID tape sum == whole-tape diff
# ---------------------------------------------------------------------------

_CONSERVATION_SCENARIOS = {
    "three_slices": _scn_three_slices,
    "partial_last_slice": _scn_partial_last_slice,
    "gap_pair_cascade": _scn_gap_pair,
    "icewpi_drift": _scn_icewpi_drift,
    "mixed_ladder": _scn_mixed_ladder,
}


class TestConservation:

    @pytest.mark.parametrize("name", sorted(_CONSERVATION_SCENARIOS))
    def test_conservation_reported_equals_own_oid_tape_sum(self, book, name):
        """(a) sum|Q| over changed rows involving the incoming OID equals
        round(trade_qty / qty_tick); (b) every changed row involves the
        incoming OID (no foreign rows); (c) truncated/book_full False."""
        priors, inc = _CONSERVATION_SCENARIOS[name]()
        oid = int(inc.order_id)
        before, st, r = _run(book, priors, inc)
        changed = _changed_rows(before, st)
        assert changed.shape[0] > 0, "fixture must trade"

        involves = ((changed[:, _T_PASS_OID] == oid)
                    | (changed[:, _T_AGRS_OID] == oid))
        # (b) no foreign rows
        assert bool(np.all(involves)), (
            f"{name}: foreign tape rows (not involving oid {oid}): "
            f"{changed[~involves].tolist()}"
        )
        # (c) preconditions
        assert not bool(st.truncated)
        assert not bool(st.book_full)
        # (a) reported == own-OID tape sum
        own_ticks = int(np.abs(changed[involves, _T_Q]).sum())
        reported_ticks = int(round(float(r.trade_qty) / _PC.qty_tick))
        assert reported_ticks == own_ticks, (
            f"{name}: reported {reported_ticks} qty ticks but the submit wrote "
            f"{own_ticks} ticks of its own fills to the tape"
        )


# ---------------------------------------------------------------------------
# (6) no over-reporting on single-pass / no-cross shapes
# ---------------------------------------------------------------------------


class TestSinglePassUnchanged:

    def test_incoming_ice_no_cross_reports_nothing(self, book):
        """(6a) ICE BUY 5.0 @ 44.90 peak 1.0 vs ask 2.0 @ 45.10: no trade;
        peak rests, 4.0 MW hidden."""
        priors = [_mk(SELL, 45.10, 2.0, 31877, 1)]
        inc = _mk(BUY, 44.90, 5.0, 46602, 2, ICE, peak=1.0)
        _, st, r = _run(book, priors, inc)
        assert not bool(r.traded)
        assert float(r.trade_qty) == pytest.approx(0.0, abs=_QTY_ABS)
        assert float(r.settlement_value) == pytest.approx(0.0, abs=_SETTLE_ABS)
        _assert_rests(st, BUY, 46602, 44.90, 1.0)
        _assert_rests(st, SELL, 31877, 45.10, 2.0)
        assert int(st.n_active) == 1
        assert int(_meta_row(st, 46602)[COL_REMAINING_HIDDEN]) == 40

    def test_incoming_ice_single_pass_unchanged(self, book):
        """(6b) ask 3.0 @ 70.20 vs ICE BUY 20.0 @ 70.20 peak 5.0: the sweep
        fills 3.0; R = 17.0 registers and shows v_1 = min(5.0 - 3.0, 17.0) =
        2.0, hidden 15.0 (OMIE-ICE-INCOMING-LADDER-1) -> report exactly 3.0 MW
        (3 * 70.20 * 0.25 = 52.65 EUR)."""
        priors = [_mk(SELL, 70.20, 3.0, 61452, 4)]
        inc = _mk(BUY, 70.20, 20.0, 73018, 6, ICE, peak=5.0)
        _, st, r = _run(book, priors, inc)
        assert bool(r.traded)
        assert float(r.trade_qty) == pytest.approx(3.0, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(70.20, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(52.65, abs=_SETTLE_ABS)
        assert _live_oids(st, SELL) == []
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): M = 3.0, P = 5.0 ->
        # k_e = 0, u = 3.0; v_1 = min(5.0 - 3.0, 17.0) = 2.0, hidden 15.0 MW.
        _assert_rests(st, BUY, 73018, 70.20, 2.0)
        assert int(st.n_active) == 1
        assert int(_meta_row(st, 73018)[COL_REMAINING_HIDDEN]) == 150


# ---------------------------------------------------------------------------
# (7) register overflow — the unregistered incoming iceberg reports peak only
# ---------------------------------------------------------------------------


class TestOverflowedTable:

    def test_incoming_ice_on_overflowed_table_reports_peak_only(
        self, tiny_table_book,
    ):
        """(7) table of 2 filled by two SAME-SIDE resting icebergs; the
        incoming ICE BUY sweeps 1.5 @ 55.00; its remainder R = 4.5 cannot
        register (truncated) and its first slice v_1 = min(2.0 - 1.5, 4.5) =
        0.5 rests unregistered (OMIE-ICE-INCOMING-LADDER-1). Report 1.5 MW / 20.625 EUR,
        unchanged (no fills are invented for an unregistered iceberg)."""
        priors = [
            _mk(BUY, 40.00, 3.0, 5101, 1, ICE, peak=1.0),
            _mk(BUY, 40.10, 3.0, 5102, 2, ICE, peak=1.0),
            _mk(SELL, 55.00, 1.5, 5150, 3),
        ]
        inc = _mk(BUY, 55.00, 6.0, 5199, 4, ICE, peak=2.0)
        _, st, r = _run(tiny_table_book, priors, inc)
        assert bool(st.truncated)
        assert bool(r.traded)
        assert float(r.trade_qty) == pytest.approx(1.5, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(55.00, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(20.625, abs=_SETTLE_ABS)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): on an overflowed table the
        # visible slice follows the same rule: M = 1.5, P = 2.0, u = 1.5 ->
        # v_1 = min(2.0 - 1.5, 4.5) = 0.5 rests unregistered.
        _assert_rests(st, BUY, 5199, 55.00, 0.5)
        _assert_rests(st, BUY, 5101, 40.00, 1.0)
        _assert_rests(st, BUY, 5102, 40.10, 1.0)
        assert _live_oids(st, SELL) == []
        assert int(st.n_active) == 2

    def test_incoming_ice_on_overflowed_table_opposite_side_counterexample(
        self, tiny_table_book,
    ):
        """(7b) table of 2 filled by two OPPOSITE-side resting icebergs.
        S4 re-target (card v5.195 A3): incoming ICE BUY 6.0 @ 55.00 peak 2.0
        (oid 5199) sweeps its total BEFORE it registers. It takes all three
        1.0 instances of 5101 @ 55.00 (5199 the aggressor on each), which
        drains 5101 and frees its slot; 5102 @ 55.10 does not cross. The
        remainder R = 3.0 then registers into the freed slot, so the table
        no longer overflows: ``truncated`` False (was True). Report 3.0 MW
        @ 55.00 = 41.25 EUR (was 2.0 / 27.50); every changed row involves
        5199 (30 qty ticks, was 20)."""
        priors = [
            _mk(SELL, 55.00, 3.0, 5101, 1, ICE, peak=1.0),
            _mk(SELL, 55.10, 3.0, 5102, 2, ICE, peak=1.0),
        ]
        inc = _mk(BUY, 55.00, 6.0, 5199, 4, ICE, peak=2.0)
        before, st, r = _run(tiny_table_book, priors, inc)
        assert not bool(st.truncated)
        assert bool(r.traded)
        assert float(r.trade_qty) == pytest.approx(3.0, abs=_QTY_ABS)
        assert float(r.trade_price) == pytest.approx(55.00, abs=_PRICE_ABS)
        assert float(r.settlement_value) == pytest.approx(41.25, abs=_SETTLE_ABS)
        changed = _changed_rows(before, st)
        involves = ((changed[:, _T_PASS_OID] == 5199)
                    | (changed[:, _T_AGRS_OID] == 5199))
        assert bool(np.all(involves))
        assert changed[:, _T_AGRS_OID].tolist() == [5199] * 3
        assert int(np.abs(changed[:, _T_Q]).sum()) == 30

    def test_overflowed_table_opposite_side_post_state(self, tiny_table_book):
        """(7b) book-state lock (8), OMIE-ICE-INCOMING-LADDER-1 re-baseline:
        bids 5199 v_1 = 1.0 @ 55.00 (hidden 2.0 MW); asks 5102 only, 1.0 @ 55.10 (hidden 2.0 MW); 5101
        drained; n_active 2 (5102 and 5199). Was: bids empty, asks 5101 and
        5102."""
        priors = [
            _mk(SELL, 55.00, 3.0, 5101, 1, ICE, peak=1.0),
            _mk(SELL, 55.10, 3.0, 5102, 2, ICE, peak=1.0),
        ]
        inc = _mk(BUY, 55.00, 6.0, 5199, 4, ICE, peak=2.0)
        _, st, _ = _run(tiny_table_book, priors, inc)
        assert _live_oids(st, BUY) == [5199]
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): M = 3.0, P = 2.0 ->
        # k_e = 1, u = 1.0; v_1 = min(2.0 - 1.0, 3.0) = 1.0 @ 55.00, hidden 2.0.
        _assert_rests(st, BUY, 5199, 55.00, 1.0)
        assert _live_oids(st, SELL) == [5102]
        _assert_rests(st, SELL, 5102, 55.10, 1.0)
        assert int(st.n_active) == 2
        assert int(_meta_row(st, 5199)[COL_REMAINING_HIDDEN]) == 20
        assert int(_meta_row(st, 5102)[COL_REMAINING_HIDDEN]) == 20
