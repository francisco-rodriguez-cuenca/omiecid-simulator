"""AUDIT tests for OMIE-NON-REMAINDER-TS-1 — a resting NON residual keeps
its ORIGINAL entry timestamp.

Independent post-implementation verification (audit mode, orchestrate.md
Step 5). Derived from ``market-spec.md`` v2.9 §5.2 / §6.1 / §6.2 items 3-7
and the story card in ``agile-plan.md``; the locked test file was NOT read.

Spec under test (§5.2): "Partial execution is NOT a modification and does
NOT assign a new timestamp. The unexecuted residual of a NON incoming order
enters the book with its original entry timestamp ... Likewise a resting
order that is partially executed keeps its timestamp." A new timestamp is
assigned only on modification or on a new iceberg slice.

The fixtures deliberately VARY incidental constants that a single-seed test
would hold fixed: side (BUY/SELL aggressor), T > ts_counter and T < ts_counter,
OIDs from 1 to 10 digits, off-round prices, price-improving limits, one or
several opposite icebergs, and the ``do_resubmit`` truncation branch on the
sell side.

Book rows are read through the jaxob row layout
``[P, Q, OID, TID, SEC, NSEC]`` (``OrderSideFeat``); the trade tape through
``TradesFeat``. Empty rows carry ``OID <= 0``.

Device policy: CPU (conftest pins the cpu platform).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    _restamp_resting_residual,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.order_validator import validate_order

# The file drives the module-level jitted ``_submit_order`` and jit-compiles
# ``_restamp_resting_residual`` directly — T1.1 jit-marker discipline.
pytestmark = [pytest.mark.jit]

_CID = 37
BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, IOC, FOK, ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                      int(ExecRestriction.FOK), int(ExecRestriction.ICE))

_P = cst.OrderSideFeat.P.value
_Q = cst.OrderSideFeat.Q.value
_OID = cst.OrderSideFeat.OID.value
_SEC = cst.OrderSideFeat.SEC.value
_NSEC = cst.OrderSideFeat.NSEC.value
_T_SEC = cst.TradesFeat.SEC.value
_T_PASS = cst.TradesFeat.PASS_OID.value
_T_AGRS = cst.TradesFeat.AGRS_OID.value


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pc():
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def bk(pc):
    contract = Contract(
        contract_id=_CID,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )
    return ContractBook(contract, pc, OMIEMarketConfig())


def _o(pc, side, price, qty, oid, ts, er=NON, peak=0.0, agent=None):
    order = make_order(
        contract_id=_CID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=oid if agent is None else agent,
        timestamp=ts, exec_restriction=er, peak_qty=peak,
    )
    assert bool(validate_order(order, pc, True)), (
        f"fixture bug: order fails validate_order: {order}"
    )
    return order


def _side_arr(state, side):
    return np.asarray(state.lob.bids if side == BUY else state.lob.asks)


def _rows(state, side, oid):
    a = _side_arr(state, side)
    return a[a[:, _OID] == oid]


def _row(state, side, oid):
    r = _rows(state, side, oid)
    assert r.shape[0] == 1, f"expected exactly one row for oid {oid}, got {r}"
    return r[0]


def _qty_mw(row):
    return int(row[_Q]) / 10.0


def _run(bk, orders):
    """Submit sequentially; return (final_state, list of (state, result))."""
    s = bk.init()
    trail = []
    for x in orders:
        s, r = bk.submit_order(s, x)
        trail.append((s, r))
    return s, trail


def _new_trades(before, after):
    """Trade rows present in ``after`` but not in ``before`` (row-set diff)."""
    b = np.asarray(before.lob.trades)
    a = np.asarray(after.lob.trades)
    changed = np.any(a != b, axis=1)
    return a[changed]


# ---------------------------------------------------------------------------
# 1. The residual rests at T — varied side / T-vs-counter / OID / price
# ---------------------------------------------------------------------------

# (label, passive side, passive price, passive qty, passive oid, passive ts,
#  aggressor price, aggressor qty, aggressor oid, aggressor T, expected rest qty)
_PLAIN_CASES = [
    # BUY aggressor, T > counter, price-improving limit, 1-digit OIDs
    ("buy_T_gt_C_improve", SELL, 50.11, 1.0, 3, 4, 52.37, 3.0, 7, 20, 2.0),
    # BUY aggressor, T < counter (prior order stamped far later)
    ("buy_T_lt_C", SELL, 48.73, 0.4, 5_512, 900, 48.73, 1.7, 88_031, 12, 1.3),
    # SELL aggressor, T > counter, off-round price, 6-digit OIDs
    ("sell_T_gt_C", BUY, 61.07, 2.2, 400_001, 30, 60.99, 2.9, 734_219, 71, 0.7),
    # SELL aggressor, T < counter, 10-digit OIDs near int32 max
    ("sell_T_lt_C_bigoid", BUY, 33.33, 0.1, 2_000_000_001, 5_000,
     33.30, 0.5, 1_999_999_999, 77, 0.4),
    # negative prices, BUY aggressor, T == counter boundary (counter=T after ts T-1)
    ("buy_neg_price_T_eq_C", SELL, -12.34, 1.1, 42, 9, -12.30, 2.0, 43, 10, 0.9),
]


@pytest.mark.parametrize(
    "case", _PLAIN_CASES, ids=[c[0] for c in _PLAIN_CASES])
def test_partial_fill_residual_rests_at_entry_timestamp(bk, pc, case):
    """§5.2: the unexecuted NON residual keeps SEC==T, NSEC==0, its own
    limit price and the unfilled quantity, whatever the counter was."""
    (_, pside, pp, pq, poid, pts, ap, aq, aoid, T, rest_q) = case
    aside = 1 - pside
    s, trail = _run(bk, [_o(pc, pside, pp, pq, poid, pts),
                         _o(pc, aside, ap, aq, aoid, T)])
    res = trail[-1][1]
    assert bool(res.traded)
    assert float(res.trade_qty) == pytest.approx(pq, abs=1e-4)
    # maker pricing (§6.2 item 3): trade at the resting price
    assert float(res.trade_price) == pytest.approx(pp, abs=0.02)
    row = _row(s, aside, aoid)
    assert int(row[_SEC]) == T
    assert int(row[_NSEC]) == 0
    assert _qty_mw(row) == pytest.approx(rest_q, abs=1e-6)
    # residual keeps its own limit (not the trade price)
    s_lim, _ = _run(bk, [_o(pc, aside, ap, 0.1, 1, T)])
    assert int(row[_P]) == int(_row(s_lim, aside, 1)[_P])
    # passive side emptied
    assert _rows(s, pside, poid).shape[0] == 0


def test_residual_timestamp_visible_via_get_agent_orders(bk, pc):
    """The public read-back reports the entry timestamp too."""
    agent = 9_091
    s, _ = _run(bk, [
        _o(pc, BUY, 71.19, 0.6, 1_234, 400),
        _o(pc, SELL, 70.01, 1.5, 56_789, 150, agent=agent),
    ])
    mine = bk.get_agent_orders(s, agent)
    assert len(mine) == 1
    assert int(mine[0].order_id) == 56_789
    assert int(mine[0].timestamp) == 150
    assert float(mine[0].quantity) == pytest.approx(0.9, abs=1e-5)


def test_restamp_does_not_rewind_ts_counter(bk, pc):
    """The restamp neither reads nor writes ``ts_counter``: after a
    fill-then-rest at a small T the counter is still past every stamp it
    has issued (it is not pulled back to T)."""
    s0, _ = _run(bk, [_o(pc, SELL, 45.55, 0.3, 17, 800)])
    c0 = int(s0.ts_counter)
    s1, r = bk.submit_order(s0, _o(pc, BUY, 45.55, 1.0, 18, 3))
    assert bool(r.traded)
    assert int(_row(s1, BUY, 18)[_SEC]) == 3
    assert int(s1.ts_counter) > c0 > 3


def test_opposite_side_row_with_same_oid_not_restamped(bk, pc):
    """The restamp is confined to the incoming order's OWN side (traced
    side-select): a non-crossing opposite-side row that happens to share
    the aggressor's OID keeps its own SEC."""
    s, _ = _run(bk, [
        _o(pc, SELL, 70.00, 1.0, 500, 3),   # does not cross 52.37
        _o(pc, SELL, 50.11, 1.0, 9, 4),
        _o(pc, BUY, 52.37, 3.0, 500, 20),
    ])
    assert int(_row(s, BUY, 500)[_SEC]) == 20
    ask = _row(s, SELL, 500)
    assert int(ask[_SEC]) == 3
    assert _qty_mw(ask) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Multi-pass paths: several icebergs, iceberg sweep sell side, resubmit
# ---------------------------------------------------------------------------

def test_sweep_through_two_icebergs_rests_at_T_and_tape_keeps_counter_stamps(
        bk, pc):
    """Two opposite ICE SELLs (different prices/peaks) are swept over many
    passes. The residual rests at T; the trade tape still carries DISTINCT
    counter SEC values for the passes (the saturation fill disambiguator
    is untouched — trade rows are not restamped)."""
    s_pre, _ = _run(bk, [
        _o(pc, SELL, 50.05, 2.0, 31, 5, ICE, 0.5),
        _o(pc, SELL, 50.07, 1.2, 32, 6, ICE, 0.4),
    ])
    T = 30
    s, r = bk.submit_order(s_pre, _o(pc, BUY, 50.13, 5.0, 33, T))
    assert float(r.trade_qty) == pytest.approx(3.2, abs=1e-4)
    assert int(s.n_active) == 0
    row = _row(s, BUY, 33)
    assert int(row[_SEC]) == T and int(row[_NSEC]) == 0
    assert _qty_mw(row) == pytest.approx(1.8, abs=1e-6)
    new = _new_trades(s_pre, s)
    assert new.shape[0] >= 3
    assert np.all(new[:, _T_AGRS] == 33)
    secs = new[:, _T_SEC].astype(int)
    assert secs.min() == T                       # pass 1 at T
    assert secs.max() > T                        # later passes at counter
    assert int(s.ts_counter) > secs.max()


def test_iceberg_sweep_sell_aggressor_T_lt_counter(bk, pc):
    """Mirror: resting ICE BUY at off-round price, NON SELL aggressor whose
    T is far below the counter."""
    s, trail = _run(bk, [
        _o(pc, BUY, 39.87, 2.4, 60_606, 2_000, ICE, 0.8),
        _o(pc, SELL, 39.80, 3.5, 70_707, 15),
    ])
    r = trail[-1][1]
    assert float(r.trade_qty) == pytest.approx(2.4, abs=1e-4)
    assert float(r.trade_price) == pytest.approx(39.87, abs=0.02)
    assert int(s.n_active) == 0
    row = _row(s, SELL, 70_707)
    assert int(row[_SEC]) == 15 and int(row[_NSEC]) == 0
    assert _qty_mw(row) == pytest.approx(1.1, abs=1e-6)


def test_resubmit_truncation_branch_sell_side_rests_at_T(bk, pc):
    """The ``do_resubmit`` branch (loop trip bound exhausted) on the SELL
    side with T < counter: residual rests at T."""
    s, trail = _run(bk, [
        _o(pc, BUY, 61.23, 12.0, 841, 300, ICE, 0.1),
        _o(pc, SELL, 61.23, 11.0, 842, 40),
    ])
    r = trail[-1][1]
    assert bool(s.truncated)
    assert float(r.trade_qty) == pytest.approx(10.1, abs=0.05)
    assert float(r.trade_price) == pytest.approx(61.23, abs=0.02)
    row = _row(s, SELL, 842)
    assert int(row[_SEC]) == 40 and int(row[_NSEC]) == 0
    assert _qty_mw(row) == pytest.approx(0.9, abs=1e-5)


def test_saturated_tape_sell_aggressor_sweep(bk, pc):
    """R2 guard, mirrored: tape pre-filled with 100 trades, then a NON
    SELL sweeps an ICE BUY. Full fill of the iceberg is still detected
    (3 passes), the iceberg is fully consumed, residual rests at T."""
    orders = []
    oid = 10_000
    for k in range(100):
        px = 30.00 + 0.01 * (k % 7)
        orders.append(_o(pc, BUY, px, 0.1, oid, 1))
        orders.append(_o(pc, SELL, px, 0.1, oid + 1, 1))
        oid += 2
    s_pre, trail = _run(bk, orders)
    assert all(bool(r.traded) for _, r in trail[1::2])
    s_pre, _ = _run_from(bk, s_pre, [_o(pc, BUY, 44.44, 2.1, 5_111, 10, ICE, 0.7)])
    s, r = bk.submit_order(s_pre, _o(pc, SELL, 44.40, 3.0, 5_222, 25))
    assert float(r.trade_qty) == pytest.approx(2.1, abs=1e-4)
    assert int(s.n_active) == 0
    assert _rows(s, BUY, 5_111).shape[0] == 0
    row = _row(s, SELL, 5_222)
    assert int(row[_SEC]) == 25
    assert _qty_mw(row) == pytest.approx(0.9, abs=1e-5)


def _run_from(bk, s, orders):
    trail = []
    for x in orders:
        s, r = bk.submit_order(s, x)
        trail.append((s, r))
    return s, trail


# ---------------------------------------------------------------------------
# 3. Price-time priority consequences (§6.1)
# ---------------------------------------------------------------------------

def test_later_second_competitor_behind_multi_pass_residual(bk, pc):
    """A residual that needed several passes (counter stamps up to ~T+8
    under the old behaviour) must still beat a same-price competitor that
    arrived in a LATER second (T+2). Independent of the row tie-break."""
    T = 30
    s, _ = _run(bk, [
        _o(pc, SELL, 50.05, 2.0, 31, 5, ICE, 0.5),
        _o(pc, BUY, 50.13, 3.0, 33, T),            # residual 1.0 @ 50.13
        _o(pc, BUY, 50.13, 1.0, 34, T + 2),        # competitor, later second
    ])
    assert int(_row(s, BUY, 33)[_SEC]) == T
    s1, r = bk.submit_order(s, _o(pc, SELL, 50.13, 0.6, 35, T + 9))
    assert bool(r.traded)
    new = _new_trades(s, s1)
    assert new.shape[0] == 1 and int(new[0, _T_PASS]) == 33
    assert _qty_mw(_row(s1, BUY, 33)) == pytest.approx(0.4, abs=1e-6)
    assert _qty_mw(_row(s1, BUY, 34)) == pytest.approx(1.0, abs=1e-6)


def test_same_second_competitor_sell_side_no_freed_rows(bk, pc):
    """Sell-side same-second competitor. No row on the ask side is freed
    between the residual resting and the competitor arriving, so the
    lowest-row tie-break (O2) coincides with arrival order."""
    T = 77
    s, _ = _run(bk, [
        _o(pc, BUY, 22.22, 0.3, 90_001, 500),
        _o(pc, SELL, 22.19, 1.0, 90_002, T),       # residual 0.7 @ 22.19
        _o(pc, SELL, 22.19, 0.5, 90_003, T),       # same-second competitor
    ])
    s1, r = bk.submit_order(s, _o(pc, BUY, 22.19, 0.2, 90_004, 600))
    new = _new_trades(s, s1)
    assert new.shape[0] == 1 and int(new[0, _T_PASS]) == 90_002
    assert _qty_mw(_row(s1, SELL, 90_003)) == pytest.approx(0.5, abs=1e-6)


def test_residual_partially_filled_as_passive_keeps_T(bk, pc):
    """§5.2 'a resting order that is partially executed keeps its
    timestamp': once rested at T, later passive partial fills must not
    move the stamp, and it keeps priority over a later arrival."""
    T = 40
    s, _ = _run(bk, [
        _o(pc, SELL, 55.55, 1.0, 1_001, 10),
        _o(pc, BUY, 55.60, 4.0, 1_002, T),          # residual 3.0 @ 55.60
        _o(pc, SELL, 55.60, 0.7, 1_003, 60),        # passive partial #1
        _o(pc, SELL, 55.58, 0.9, 1_004, 61),        # passive partial #2
        _o(pc, BUY, 55.60, 1.0, 1_005, 62),         # later competitor
    ])
    row = _row(s, BUY, 1_002)
    assert int(row[_SEC]) == T
    assert _qty_mw(row) == pytest.approx(1.4, abs=1e-6)
    s1, _ = bk.submit_order(s, _o(pc, SELL, 55.60, 1.4, 1_006, 70))
    assert _rows(s1, BUY, 1_002).shape[0] == 0
    assert _qty_mw(_row(s1, BUY, 1_005)) == pytest.approx(1.0, abs=1e-6)


def test_residual_behind_better_priced_competitor(bk, pc):
    """Timestamp restoration never overrides price priority: a later
    competitor at a better price fills first."""
    s, _ = _run(bk, [
        _o(pc, SELL, 47.00, 0.5, 201, 1),
        _o(pc, BUY, 47.00, 2.0, 202, 5),           # residual 1.5 @ 47.00, T=5
        _o(pc, BUY, 47.01, 0.3, 203, 50),          # better price, later
    ])
    s1, _ = bk.submit_order(s, _o(pc, SELL, 46.90, 0.3, 204, 60))
    new = _new_trades(s, s1)
    assert int(new[0, _T_PASS]) == 203
    assert int(_row(s1, BUY, 202)[_SEC]) == 5


# ---------------------------------------------------------------------------
# 4. Paths that must NOT be restamped
# ---------------------------------------------------------------------------

def test_modify_of_rested_residual_gets_replacement_timestamp(bk, pc):
    """§5.3: modifying a rested residual resets its timestamp to the
    replacement's own stamp (and the replacement then loses priority)."""
    s, _ = _run(bk, [
        _o(pc, SELL, 50.50, 0.4, 3_001, 2),
        _o(pc, BUY, 50.50, 1.4, 3_002, 20),        # residual 1.0 @ T=20
        _o(pc, BUY, 50.40, 0.5, 3_003, 25),
    ])
    assert int(_row(s, BUY, 3_002)[_SEC]) == 20
    s, _ = bk.modify_order(s, 3_002, BUY, _o(pc, BUY, 50.40, 0.8, 3_002, 90))
    assert int(_row(s, BUY, 3_002)[_SEC]) == 90
    s1, _ = bk.submit_order(s, _o(pc, SELL, 50.40, 0.5, 3_004, 95))
    new = _new_trades(s, s1)
    assert int(new[0, _T_PASS]) == 3_003


def test_modify_replacement_that_crosses_rests_at_replacement_ts(bk, pc):
    """Card interaction (vi): a crossing replacement rests at ITS stamp."""
    s, _ = _run(bk, [
        _o(pc, BUY, 49.00, 1.0, 4_001, 3),
        _o(pc, SELL, 49.50, 0.3, 4_002, 4),
    ])
    s, r = bk.modify_order(s, 4_001, BUY, _o(pc, BUY, 49.61, 1.0, 4_001, 44))
    assert bool(r.traded)
    row = _row(s, BUY, 4_001)
    assert int(row[_SEC]) == 44
    assert _qty_mw(row) == pytest.approx(0.7, abs=1e-6)


@pytest.mark.parametrize("side", [BUY, SELL], ids=["buy", "sell"])
def test_ioc_residual_never_rests(bk, pc, side):
    s, trail = _run(bk, [
        _o(pc, 1 - side, 50.00, 1.0, 71, 5),
        _o(pc, side, 50.00, 3.0, 72, 30, IOC),
    ])
    assert float(trail[-1][1].trade_qty) == pytest.approx(1.0, abs=1e-4)
    assert not np.any(_side_arr(s, side)[:, _OID] > 0)


def test_fully_filled_non_leaves_no_row(bk, pc):
    s, trail = _run(bk, [
        _o(pc, SELL, 50.01, 2.0, 81, 5),
        _o(pc, BUY, 50.01, 2.0, 82, 30),
    ])
    assert float(trail[-1][1].trade_qty) == pytest.approx(2.0, abs=1e-4)
    assert not np.any(_side_arr(s, BUY)[:, _OID] > 0)
    assert not np.any(_side_arr(s, SELL)[:, _OID] > 0)


def test_non_fill_submit_is_byte_noop_on_book_rows(bk, pc):
    """A NON that does not trade rests at T and leaves every other row
    byte-identical (fill gate)."""
    s0, _ = _run(bk, [
        _o(pc, BUY, 45.00, 1.0, 7_777, 5),
        _o(pc, SELL, 46.10, 1.0, 7_778, 6),
    ])
    s1, r = bk.submit_order(s0, _o(pc, BUY, 44.00, 0.5, 7_779, 300))
    assert not bool(r.traded)
    assert int(_row(s1, BUY, 7_779)[_SEC]) == 300
    assert np.array_equal(_row(s1, BUY, 7_777), _row(s0, BUY, 7_777))
    assert np.array_equal(np.asarray(s1.lob.asks), np.asarray(s0.lob.asks))
    assert np.array_equal(np.asarray(s1.lob.trades), np.asarray(s0.lob.trades))


def test_non_fill_shared_oid_sell_side_older_row_keeps_stamp(bk, pc):
    """Fill gate, sell side, 7-digit OID: a non-crossing NON SELL that
    reuses a resting ask's OID must not restamp the older row."""
    s0, _ = _run(bk, [
        _o(pc, SELL, 58.08, 0.6, 3_141_592, 12),
        _o(pc, BUY, 50.00, 1.0, 11, 13),
    ])
    s1, r = bk.submit_order(s0, _o(pc, SELL, 58.20, 0.4, 3_141_592, 400))
    assert not bool(r.traded)
    secs = sorted(int(x) for x in _rows(s1, SELL, 3_141_592)[:, _SEC])
    assert secs == [12, 400]
    assert np.array_equal(np.asarray(s1.lob.bids), np.asarray(s0.lob.bids))


def test_incoming_iceberg_peak_rest_unaffected(bk, pc):
    """ICE path is out of scope: a partially filled incoming iceberg rests
    at the incoming T (it was never re-dispatched). Since
    OMIE-ICE-INCOMING-HOP-1 it sweeps its total (0.3 fills) and the
    remainder 2.7 rests showing v_1 = min(peak - 0.3, 2.7) = 0.7 MW (20
    ticks hidden), SEC == T, NSEC 0 (OMIE-ICE-INCOMING-LADDER-1)."""
    s, trail = _run(bk, [
        _o(pc, SELL, 50.00, 0.3, 61, 5),
        _o(pc, BUY, 50.00, 3.0, 62, 30, ICE, 1.0),
    ])
    assert float(trail[-1][1].trade_qty) == pytest.approx(0.3, abs=1e-4)
    row = _row(s, BUY, 62)
    assert int(row[_SEC]) == 30
    # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): M = 3, P = 10 -> k_e = 0,
    # u = 3; v_1 = min(10 - 3, 27) = 7 ticks (spec §4.2.1 Rule 2 item (4)).
    assert _qty_mw(row) == pytest.approx(0.7, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Property: randomized mixed sequence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [3, 11, 2026])
def test_random_mixed_sequence_invariants(bk, pc, seed):
    """Across a random NON/IOC/FOK/ICE stream with off-round prices, both
    sides and non-monotone T:
      (a) every NON that traded and still rests has SEC == its T, NSEC == 0;
      (b) every NON/IOC/FOK that did NOT trade leaves pre-existing non-
          iceberg rows' SEC untouched, and every pre-existing non-iceberg
          row that survives a submit keeps its SEC (partial execution never
          restamps a resting order);
      (c) ts_counter never decreases;
      (d) the book is never crossed after a submit.
    """
    rng = np.random.default_rng(seed)
    s = bk.init()
    ice_oids = set()
    oid = int(rng.integers(10, 10_000))
    for _ in range(45):
        oid += int(rng.integers(1, 50_000))
        side = int(rng.integers(0, 2))
        er = int(rng.choice([NON, NON, NON, IOC, FOK, ICE]))
        price = round(50.0 + int(rng.integers(-40, 41)) * 0.01, 2)
        qty = round(int(rng.integers(1, 25)) * 0.1, 1)
        T = int(rng.integers(1, 3_000))
        peak = 0.0
        if er == ICE:
            qty = max(qty, 0.4)
            peak = round(max(0.1, round(qty / 3, 1)), 1)
            if peak >= qty:
                er, peak = NON, 0.0
        x = _o(pc, side, price, qty, oid, T, er, peak)
        pre = {}
        for sd in (BUY, SELL):
            a = _side_arr(s, sd)
            for rrow in a[a[:, _OID] > 0]:
                if int(rrow[_OID]) not in ice_oids:
                    pre[(sd, int(rrow[_OID]))] = int(rrow[_SEC])
        c_before = int(s.ts_counter)
        s, r = bk.submit_order(s, x)
        if er == ICE:
            ice_oids.add(oid)
        assert int(s.ts_counter) >= c_before
        if er == NON:
            here = _rows(s, side, oid)
            if here.shape[0]:
                assert int(here[0, _SEC]) == T, (seed, oid, here, bool(r.traded))
                assert int(here[0, _NSEC]) == 0
        for (sd, o_id), sec in pre.items():
            got = _rows(s, sd, o_id)
            if got.shape[0]:
                assert int(got[0, _SEC]) == sec, (seed, sd, o_id)
        bids, asks = _side_arr(s, BUY), _side_arr(s, SELL)
        lb, la = bids[bids[:, _OID] > 0], asks[asks[:, _OID] > 0]
        if lb.shape[0] and la.shape[0]:
            assert lb[:, _P].max() < la[:, _P].min()


# ---------------------------------------------------------------------------
# 6. The helper itself (jit, traced side, gate, both columns)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_sided_lob(bk, pc):
    s, _ = _run(bk, [
        _o(pc, BUY, 40.00, 1.0, 123_456, 7),
        _o(pc, BUY, 39.00, 1.0, 5, 8),
        _o(pc, SELL, 60.00, 1.0, 123_456, 9),
        _o(pc, SELL, 61.00, 1.0, 6, 10),
    ])
    return s.lob


def test_helper_jitted_single_compile_traced_side_and_gate(two_sided_lob):
    """Under jit with side, oid, T and filled all traced: one compile
    serves both sides and both gate values; only the own-side hit rows'
    SEC/NSEC change; the gate-False call is a byte no-op."""
    lob = two_sided_lob
    f = jax.jit(_restamp_resting_residual)
    args = lambda sd, fl: (lob, jnp.int32(123_456), jnp.int32(sd),
                           jnp.int32(3), jnp.bool_(fl))
    out_b = f(*args(BUY, True))
    out_s = f(*args(SELL, True))
    out_n = f(*args(BUY, False))
    out_n2 = f(*args(SELL, False))
    assert f._cache_size() == 1
    for out in (out_n, out_n2):
        assert np.array_equal(np.asarray(out.bids), np.asarray(lob.bids))
        assert np.array_equal(np.asarray(out.asks), np.asarray(lob.asks))
        assert np.array_equal(np.asarray(out.trades), np.asarray(lob.trades))

    def check(out, own):
        own_new = np.asarray(out.bids if own == BUY else out.asks)
        own_old = np.asarray(lob.bids if own == BUY else lob.asks)
        oth_new = np.asarray(out.asks if own == BUY else out.bids)
        oth_old = np.asarray(lob.asks if own == BUY else lob.bids)
        assert np.array_equal(oth_new, oth_old)
        hit = own_old[:, _OID] == 123_456
        assert hit.sum() == 1
        assert np.all(own_new[hit, _SEC] == 3)
        assert np.all(own_new[hit, _NSEC] == 0)
        keep = [c for c in range(own_old.shape[1]) if c not in (_SEC, _NSEC)]
        assert np.array_equal(own_new[:, keep], own_old[:, keep])
        assert np.array_equal(own_new[~hit], own_old[~hit])
        assert np.array_equal(np.asarray(out.trades), np.asarray(lob.trades))

    check(out_b, BUY)
    check(out_s, SELL)


def test_helper_miss_is_noop(two_sided_lob):
    lob = two_sided_lob
    out = _restamp_resting_residual(lob, 999_999, BUY, 1, True)
    assert np.array_equal(np.asarray(out.bids), np.asarray(lob.bids))
    assert np.array_equal(np.asarray(out.asks), np.asarray(lob.asks))


def test_helper_does_not_touch_empty_rows_with_sentinel_oid(two_sided_lob):
    """Empty rows carry a non-positive sentinel OID; a (spec-invalid,
    defensive) order_id equal to that sentinel must still not corrupt
    empty rows in a way that makes them look live (P/Q/OID untouched)."""
    lob = two_sided_lob
    bids = np.asarray(lob.bids)
    sentinel = int(bids[bids[:, _OID] <= 0][0, _OID])
    out = _restamp_resting_residual(lob, sentinel, BUY, 5, True)
    nb = np.asarray(out.bids)
    for c in (_P, _Q, _OID):
        assert np.array_equal(nb[:, c], bids[:, c])
