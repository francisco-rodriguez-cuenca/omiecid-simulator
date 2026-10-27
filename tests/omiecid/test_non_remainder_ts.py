"""Locked behavioural tests for OMIE-NON-REMAINDER-TS-1.

Contract (agile-plan.md §Story OMIE-NON-REMAINDER-TS-1, v5.151,
"Test-first contract" + "Blast radius" rows; market-spec.md v2.9 §5.2 and
§6.2 item 4):

    Partial execution is NOT a modification and does NOT assign a new
    timestamp. The unexecuted residual of a NON incoming order enters the
    book with its ORIGINAL entry timestamp (``order.timestamp``), with
    ``NSEC == 0``.

The fix is book-visible only: the resting row's ``SEC``/``NSEC`` cells.
Everything else is a byte-identity guard against the values measured on
the unfixed code (``main`` @ 0c91291 / 14b603d, CPU). That covers the
trades tape, ``ts_counter``, ``SubmitResult``, the IOC/FOK/ICE paths, the
no-fill NON path, and the stamps of regenerated iceberg slices. Every
literal below was re-derived by the test-agent on the unfixed code before
it was locked.

RED on unfixed code (they fail by design until the fix lands):
    (1a/1b/1c), (2), (3) + its mirrored sell-side variant, (4) + its
    variants, (6), (7), (11b).
GREEN on unfixed code (guards; they must stay green):
    (5), (5b), (8), (9), (10a-e), (11a).

Row layout (jaxob ``OrderSideFeat``): [P, Q, OID, TID, SEC, NSEC].
Tape layout (jaxob ``TradesFeat``):
    [P, Q, PASS_OID, AGRS_OID, SEC, NSEC, PASS_TID, AGRS_TID].
Every fixture uses ``agent_id == order_id``, so TID == OID.

Every order goes through the public ``ContractBook.submit_order`` on a
fresh ``book.init()``. It must pass ``validate_order``. Incidental
constants vary: T > C and T < C, both sides, 1-6 digit OIDs, and off-round
prices where the seed allows.

Device policy: CPU default (conftest). ``submit_order`` drives the
module-level jitted ``_submit_order``, so the file-wide ``jit`` marker
applies.
"""

from __future__ import annotations

import functools
import hashlib

import numpy as np
import pytest

from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat, TradesFeat
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.order_validator import validate_order

pytestmark = [pytest.mark.jit]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTRACT_ID = 37

BUY = int(Side.BUY)
SELL = int(Side.SELL)
NON = ExecRestriction.NON
IOC = ExecRestriction.IOC
FOK = ExecRestriction.FOK
ICE = ExecRestriction.ICE

_P = OrderSideFeat.P.value
_Q = OrderSideFeat.Q.value
_OID = OrderSideFeat.OID.value
_SEC = OrderSideFeat.SEC.value
_NSEC = OrderSideFeat.NSEC.value

_T_PASS_OID = TradesFeat.PASS_OID.value
_T_PASS_TID = TradesFeat.PASS_TID.value

# SubmitResult tolerances (float32 decode, card (9)).
_SETTLE_ABS = 0.02
_PRICE_ABS = 0.02
_QTY_ABS = 1e-3

# sha256[:16] of an all-zero (empty) book side / empty trades tape at the
# default config. Pinned on the unfixed code.
_EMPTY_SIDE_H = "50ee3bf5d25a296b"
_EMPTY_TAPE_H = "b525f810c4462067"

_PC = OMIEProductConfig()


def _open_contract() -> Contract:
    return Contract(
        contract_id=_CONTRACT_ID,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


_BOOK = ContractBook(_open_contract(), _PC, OMIEMarketConfig())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_order(side, price, qty, oid, ts, er=NON, peak=0.0):
    """Build an OMIE order and REQUIRE it to pass ``validate_order``."""
    order = make_order(
        contract_id=_CONTRACT_ID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=oid, timestamp=ts,
        exec_restriction=er, peak_qty=peak,
    )
    assert bool(validate_order(order, _PC, True)), (
        f"fixture bug: constructor failed validate_order: {order}"
    )
    return order


def _h(arr) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(arr)).tobytes()
    ).hexdigest()[:16]


def _side_arr(state, side):
    return np.asarray(state.lob.bids if side == BUY else state.lob.asks)


def _rows(state, side, oid):
    """``[(row_index, row), ...]`` for live rows of ``oid`` on ``side``."""
    arr = _side_arr(state, side)
    idx = np.where((arr[:, _OID] == oid) & (arr[:, _Q] > 0))[0]
    return [(int(i), arr[i]) for i in idx]


def _single_row(state, side, oid):
    rows = _rows(state, side, oid)
    assert len(rows) == 1, f"oid {oid}: expected exactly 1 live row, got {rows}"
    return rows[0]


def _qty_ticks(qty_mw: float) -> int:
    return int(round(qty_mw / _PC.qty_tick))


def _live(state, side):
    arr = _side_arr(state, side)
    return arr[(arr[:, _OID] > 0) & (arr[:, _Q] > 0)]


class _Run:
    """Per-submit record of a scenario (states are immutable JAX pytrees)."""

    def __init__(self, orders):
        self.orders = orders
        self.states = []   # post-submit states
        self.results = []  # SubmitResult per submit
        state = _BOOK.init()
        self.init_state = state
        for o in orders:
            state, r = _BOOK.submit_order(state, o)
            self.states.append(state)
            self.results.append(r)


# ---------------------------------------------------------------------------
# Scenarios. Each returns the ordered submit list. ``_REST`` gives the index
# of the NON submit whose residual rests after a fill.
# ---------------------------------------------------------------------------


def _saturating_prefix():
    """Exactly 100 trades: 50 SELL/BUY 1.0 @ 40.00 ts1 pairs + 50 @ 41.00 ts2."""
    out, oid = [], 60000
    for price, ts in ((40.00, 1), (41.00, 2)):
        for _ in range(50):
            out.append(_mk_order(SELL, price, 1.0, oid, ts))
            out.append(_mk_order(BUY, price, 1.0, oid + 1, ts))
            oid += 2
    return out


def _scenario(name):
    m = _mk_order
    if name == "1a":   # T=20 > C=11, buy side
        return [m(SELL, 50.00, 1.0, 101, 10), m(BUY, 50.00, 3.0, 202, 20)]
    if name == "1b":   # T=120 > C=102
        return [m(SELL, 50.00, 1.0, 55501, 101),
                m(BUY, 50.00, 3.0, 55502, 120)]
    if name == "1c":   # T=50 < C=101, sell side
        return [m(BUY, 60.00, 1.0, 7001, 100), m(SELL, 60.00, 4.0, 7003, 50)]
    if name == "2":    # multi-level sweep in one pass
        return [m(SELL, 48.00, 1.0, 11, 5), m(SELL, 49.00, 1.0, 12, 6),
                m(SELL, 50.00, 1.0, 13, 7), m(BUY, 50.00, 5.0, 14, 20)]
    if name == "3":    # iceberg sweep, buy aggressor, T > C
        return [m(SELL, 50.00, 3.0, 111, 10, ICE, 1.0),
                m(BUY, 50.00, 5.0, 222, 20)]
    if name == "3m":   # mirrored: ICE BUY, NON SELL aggressor, T=40 < C=301
        return [m(BUY, 55.37, 3.0, 900001, 300, ICE, 1.0),
                m(SELL, 55.37, 5.0, 900002, 40)]
    if name == "6":    # _TRIP_BOUND exhaustion -> do_resubmit branch
        return [m(SELL, 50.00, 12.0, 41, 10, ICE, 0.1),
                m(BUY, 50.00, 11.0, 42, 20)]
    if name == "7":    # saturated tape, then the (3) shape
        return _saturating_prefix() + [
            m(SELL, 50.00, 3.0, 5111, 10, ICE, 1.0),
            m(BUY, 50.00, 5.0, 5222, 20)]
    # (4) same-second competitor + probe
    if name == "4a":
        return _scenario("1a") + [m(BUY, 50.00, 1.0, 303, 20),
                                  m(SELL, 50.00, 1.0, 404, 30)]
    if name == "4b":
        return _scenario("3") + [m(BUY, 50.00, 1.0, 333, 20),
                                 m(SELL, 50.00, 1.0, 444, 30)]
    if name == "4c":
        return _scenario("1c") + [m(SELL, 60.00, 1.0, 7004, 50),
                                  m(BUY, 60.00, 1.0, 7005, 200)]
    # (5) later-second competitor (T+1 / T+2) + probe
    if name == "5a1":
        return _scenario("1a") + [m(BUY, 50.00, 1.0, 313, 21),
                                  m(SELL, 50.00, 1.0, 414, 30)]
    if name == "5a2":
        return _scenario("1a") + [m(BUY, 50.00, 1.0, 323, 22),
                                  m(SELL, 50.00, 1.0, 424, 30)]
    if name == "5s1":
        return _scenario("2") + [m(BUY, 50.00, 1.0, 15, 21),
                                 m(SELL, 49.50, 1.0, 16, 40)]
    if name == "5s2":
        return _scenario("2") + [m(BUY, 50.00, 1.0, 17, 22),
                                 m(SELL, 49.50, 1.0, 18, 40)]
    # (5b) earlier-second competitor + probe
    if name == "5bA":
        return _scenario("1a") + [m(BUY, 50.00, 1.0, 3031, 19),
                                  m(SELL, 50.00, 1.0, 4041, 30)]
    if name == "5bB":
        return _scenario("1c") + [m(SELL, 60.00, 1.0, 70049, 49),
                                  m(BUY, 60.00, 1.0, 70059, 200)]
    # (10) other paths
    if name == "10a":
        return [m(BUY, 61.07, 1.0, 8, 3), m(SELL, 63.21, 2.0, 424242, 77)]
    if name == "10b":
        return [m(SELL, 50.00, 1.0, 101, 10), m(BUY, 50.00, 3.0, 31337, 20, IOC)]
    if name == "10c_acc":
        return [m(SELL, 50.00, 2.0, 6061, 10), m(BUY, 50.00, 2.0, 6062, 20, FOK)]
    if name == "10c_rej":
        return [m(SELL, 50.00, 1.0, 6071, 10), m(BUY, 50.00, 3.0, 6072, 20, FOK)]
    if name == "10d_peak":
        return [m(SELL, 52.00, 1.0, 97, 4), m(BUY, 51.43, 3.0, 818, 33, ICE, 1.0)]
    if name == "10d_regen":
        return [m(SELL, 51.43, 1.5, 97, 4), m(BUY, 51.43, 3.0, 818, 33, ICE, 1.0)]
    if name == "10e":
        return [m(BUY, 45.00, 1.0, 77, 5), m(BUY, 44.00, 1.0, 77, 30)]
    if name == "11":
        return _mixed_sequence()
    raise KeyError(name)


def _mixed_sequence():
    """(11): mixed NON/IOC/FOK/ICE sequence with two NON fill-then-rest events.

    idx3  NON BUY 3.5 @ 51.50 ts9 (T > C) sweeps 51.13 + an ICE SELL and
          rests 0.5 MW.
    idx9  NON SELL 3.0 @ 49.00 ts5 (T < C) sweeps 49.87 + an ICE BUY and
          rests 0.5 MW. idx10 partially fills it passively.
    """
    m = _mk_order
    return [
        m(SELL, 51.13, 2.0, 21, 3),
        m(SELL, 51.50, 2.0, 3322, 4, ICE, 0.5),
        m(BUY, 51.13, 1.0, 44, 8),
        m(BUY, 51.50, 3.5, 55, 9),
        m(SELL, 51.00, 1.0, 66, 12, IOC),
        m(SELL, 50.00, 1.0, 777, 13, FOK),
        m(BUY, 49.87, 1.0, 88, 14),
        m(SELL, 49.87, 0.5, 99, 15, FOK),
        m(BUY, 49.00, 2.0, 1234, 16, ICE, 0.4),
        m(SELL, 49.00, 3.0, 98765, 5),
        m(BUY, 49.00, 0.2, 5, 25),
    ]


@functools.lru_cache(maxsize=None)
def _run(name) -> _Run:
    return _Run(_scenario(name))


# Index of the NON submit whose residual rests after a fill.
_REST = {"1a": 1, "1b": 1, "1c": 1, "2": 3, "3": 1, "3m": 1, "6": 1,
         "7": 201, "4a": 1, "4b": 1, "4c": 1}


# ---------------------------------------------------------------------------
# Pins measured on the unfixed code (card (8)/(9)).
# ---------------------------------------------------------------------------

# name -> (submit index, trades-tape hash, ts_counter) after that submit.
_TAPE_COUNTER_PINS = {
    "1a": (1, "3ff91cf8669aadeb", 22),
    "1b": (1, "84b87f197f007c67", 122),
    "1c": (1, "3f6499e37d31a37d", 102),
    "2": (3, "ba9e50aa897a264b", 22),
    "3": (1, "078cc9d521cd87a8", 26),
    "3m": (1, "6f72d9c8abe8ba9a", 306),
    "6": (1, "bc3ace6ab6bf0832", 222),
    "7": (201, "f1fe2cbe582e2ecb", 107),
    "10a": (1, _EMPTY_TAPE_H, 78),
    "10b": (1, "8b1f4f91b06dbef8", 22),
    "10c_acc": (1, "fb30438f05970a79", 22),
    "10c_rej": (1, _EMPTY_TAPE_H, 11),
    "10d_peak": (1, _EMPTY_TAPE_H, 34),
    # S4 (OMIE-ICE-INCOMING-HOP-1) re-baseline: the incoming ICE sweeps its
    # total, so the 1.5 MW fill is ONE trade row (one per counter bid), not
    # 1.0 + 0.5 (was "4f862945f9cc9134"); ts_counter unchanged.
    # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): the one row crosses the
    # incoming instance boundary 1.0 (peak) and splits into two pieces,
    # Q -10 and -5, every other column copied (Rule 2 item (2)). Hash derived
    # from the S4 tape "3d9c7e94362f0246" (row [1005043, -15, 97, 818, 33, 0,
    # 97, 818]) by that split; ts_counter unchanged (E-C writes only the tape).
    "10d_regen": (1, "bb13afe7bf9671aa", 35),
    "10e": (1, _EMPTY_TAPE_H, 31),
    "11": (10, "3192811ee368a54e", 29),
    # (4) + variants: up to AND including the residual-resting submit, and
    # also after the competitor submit (the competitor writes no trade).
    "4a": (2, "3ff91cf8669aadeb", 22),
    "4b": (2, "078cc9d521cd87a8", 26),
    "4c": (2, "3f6499e37d31a37d", 102),
}

# name -> {submit index: (accepted, traded, trade_qty, trade_price, settlement)}
_SUBMIT_RESULT_PINS = {
    "1a": {1: (True, True, 1.0, 50.00, 12.5)},
    "1b": {1: (True, True, 1.0, 50.00, 12.5)},
    "1c": {1: (True, True, 1.0, 60.00, 15.0)},
    "2": {3: (True, True, 3.0, 49.00, 36.75)},
    "3": {1: (True, True, 3.0, 50.00, 37.5)},
    "3m": {1: (True, True, 3.0, 55.37, 41.5275)},
    "6": {1: (True, True, 10.1, 49.99, 126.217)},
    "7": {201: (True, True, 3.0, 50.00, 37.5)},
    "4a": {1: (True, True, 1.0, 50.00, 12.5), 2: (True, False, 0.0, 0.0, 0.0),
           3: (True, True, 1.0, 50.00, 12.5)},
    "4b": {1: (True, True, 3.0, 50.00, 37.5), 2: (True, False, 0.0, 0.0, 0.0),
           3: (True, True, 1.0, 50.00, 12.5)},
    "4c": {1: (True, True, 1.0, 60.00, 15.0), 2: (True, False, 0.0, 0.0, 0.0),
           3: (True, True, 1.0, 60.00, 15.0)},
    "10a": {1: (True, False, 0.0, 0.0, 0.0)},
    "10b": {1: (True, True, 1.0, 50.00, 12.5)},
    "10c_acc": {1: (True, True, 2.0, 50.00, 25.0)},
    "10c_rej": {1: (True, False, 0.0, 0.0, 0.0)},
    "10d_peak": {1: (True, False, 0.0, 0.0, 0.0)},
    "10d_regen": {1: (True, True, 1.5, 51.43, 19.2863)},
    "10e": {1: (True, False, 0.0, 0.0, 0.0)},
    "11": {
        0: (True, False, 0.0, 0.0, 0.0),
        1: (True, False, 0.0, 0.0, 0.0),
        2: (True, True, 1.0, 51.13, 12.7825),
        3: (True, True, 3.0, 51.38, 38.5325),
        4: (True, True, 0.5, 51.50, 6.4375),
        5: (True, False, 0.0, 0.0, 0.0),   # FOK killed
        6: (True, False, 0.0, 0.0, 0.0),
        7: (True, True, 0.5, 49.87, 6.2338),
        8: (True, False, 0.0, 0.0, 0.0),
        9: (True, True, 2.5, 49.17, 30.734),
        10: (True, True, 0.2, 49.00, 2.45),
    },
}

# (11): ts_counter after each submit, unfixed code.
_MIXED_TS_COUNTER = [4, 5, 10, 17, 18, 18, 18, 19, 19, 28, 29]


def _assert_residual_at_T(state, order, qty_mw):
    """The resting residual of ``order``: ``qty_mw``, SEC == T, NSEC == 0."""
    side, oid, T = int(order.side), int(order.order_id), int(order.timestamp)
    _, row = _single_row(state, side, oid)
    assert int(row[_Q]) == _qty_ticks(qty_mw), (
        f"oid {oid}: residual qty ticks {int(row[_Q])} != {_qty_ticks(qty_mw)}")
    assert int(row[_SEC]) == T, (
        f"oid {oid}: resting residual SEC={int(row[_SEC])}, expected the "
        f"original entry timestamp T={T} (market-spec v2.9 §5.2)")
    assert int(row[_NSEC]) == 0, f"oid {oid}: NSEC={int(row[_NSEC])} != 0"


# ---------------------------------------------------------------------------
# (1) Plain partial fill -> residual rests at T.        RED on unfixed code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, residual_mw", [
    ("1a", 2.0),   # unfixed: SEC 21
    ("1b", 2.0),   # unfixed: SEC 121
    ("1c", 3.0),   # unfixed: SEC 101 (T < C, sell side)
])
def test_plain_partial_fill_rests_at_T(name, residual_mw):
    run = _run(name)
    order = run.orders[_REST[name]]
    state = run.states[_REST[name]]
    _assert_residual_at_T(state, order, residual_mw)
    # The same stamp through the public agent view.
    views = [o for o in _BOOK.get_agent_orders(state, int(order.agent_id))
             if int(o.order_id) == int(order.order_id)]
    assert len(views) == 1
    assert int(views[0].timestamp) == int(order.timestamp)


# ---------------------------------------------------------------------------
# (2) Multi-level sweep in one pass.                    RED on unfixed code.
# ---------------------------------------------------------------------------


def test_multi_level_sweep_rests_at_T():
    run = _run("2")
    r = run.results[3]
    assert float(r.trade_qty) == pytest.approx(3.0, abs=_QTY_ABS)
    assert float(r.trade_price) == pytest.approx(49.00, abs=_PRICE_ABS)
    _assert_residual_at_T(run.states[3], run.orders[3], 2.0)  # unfixed: 21
    assert _live(run.states[3], SELL).shape[0] == 0


# ---------------------------------------------------------------------------
# (3) Iceberg sweep (+ mirrored sell side, T < C).      RED on unfixed code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "3",    # ICE SELL vs NON BUY, T=20 > C; unfixed: SEC 25
    "3m",   # ICE BUY vs NON SELL, T=40 < C=301; unfixed: SEC 305
])
def test_iceberg_sweep_rests_at_T(name):
    run = _run(name)
    state, order, r = run.states[1], run.orders[1], run.results[1]
    assert float(r.trade_qty) == pytest.approx(3.0, abs=_QTY_ABS)
    _assert_residual_at_T(state, order, 2.0)
    assert int(state.n_active) == 0
    assert _live(state, 1 - int(order.side)).shape[0] == 0


# ---------------------------------------------------------------------------
# (4) Same-second competitor does not jump the residual. RED on unfixed code.
# ---------------------------------------------------------------------------

# name -> (residual oid, competitor oid, residual side,
#          residual qty MW after probe, unfixed-code probe tape row)
_SAME_SECOND = {
    "4a": (202, 303, BUY, 1.0, [1004900, 10, 303, 404, 30, 0, 303, 404]),
    "4b": (222, 333, BUY, 1.0, [1004900, 10, 333, 444, 30, 0, 333, 444]),
    "4c": (7003, 7004, SELL, 2.0, [1005900, -10, 7004, 7005, 200, 0, 7004, 7005]),
}


def _assert_fixture_rows_ordered(run, res_oid, comp_oid, side):
    """Fixture condition (O2): residual row index < competitor row index.

    At equal (SEC, NSEC) jaxob breaks the tie by lowest row index. Each
    fixture frees no row on the residual's side between the residual
    resting and the competitor arriving, so the residual holds the lower
    row. This precondition is true on the unfixed code too.
    """
    comp_state = run.states[2]
    (res_idx, _), (comp_idx, comp_row) = (
        _single_row(comp_state, side, res_oid),
        _single_row(comp_state, side, comp_oid))
    assert res_idx < comp_idx, "fixture bug: residual not in the lower row"
    assert int(comp_row[_SEC]) == int(run.orders[1].timestamp), (
        "fixture bug: competitor must carry the residual's entry second T")


@pytest.mark.parametrize("name", ["4a", "4b", "4c"])
def test_same_second_competitor_does_not_jump(name):
    """A same-price same-side competitor at the SAME second T arrives after
    the residual rests. The next opposite order must fill the RESIDUAL.

    Fixture condition: the win at an equal stamp relies on the jaxob
    lowest-row tie-break (O2). No row on the residual's side is freed
    between the residual resting and the competitor arriving (asserted).
    """
    res_oid, comp_oid, side, res_after, _ = _SAME_SECOND[name]
    run = _run(name)
    _assert_fixture_rows_ordered(run, res_oid, comp_oid, side)
    pre, post = run.states[2], run.states[3]
    changed = np.asarray(post.lob.trades)[
        np.any(np.asarray(pre.lob.trades) != np.asarray(post.lob.trades), axis=1)]
    assert changed.shape[0] == 1
    assert int(changed[0, _T_PASS_OID]) == res_oid, (
        f"probe filled oid {int(changed[0, _T_PASS_OID])}, expected the "
        f"residual {res_oid} (price-time priority, market-spec §5.2)")
    _, rrow = _single_row(post, side, res_oid)
    _, crow = _single_row(post, side, comp_oid)
    assert int(rrow[_Q]) == _qty_ticks(res_after)
    assert int(crow[_Q]) == _qty_ticks(1.0)


# ---------------------------------------------------------------------------
# (5) Later-second competitor stays behind.        GREEN on unfixed code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, res_oid, comp_oid", [
    ("5a1", 202, 313),   # (1a), competitor at T+1
    ("5a2", 202, 323),   # (1a), competitor at T+2
    ("5s1", 14, 15),     # (2),  competitor at T+1
    ("5s2", 14, 17),     # (2),  competitor at T+2
])
def test_competitor_at_T_plus_1_stays_behind(name, res_oid, comp_oid):
    run = _run(name)
    pre, post = run.states[-2], run.states[-1]
    changed = np.asarray(post.lob.trades)[
        np.any(np.asarray(pre.lob.trades) != np.asarray(post.lob.trades), axis=1)]
    assert changed.shape[0] == 1
    assert int(changed[0, _T_PASS_OID]) == res_oid
    _, crow = _single_row(post, BUY, comp_oid)
    assert int(crow[_Q]) == _qty_ticks(1.0)


# ---------------------------------------------------------------------------
# (5b) Earlier-second competitor is never overtaken.  GREEN on unfixed code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, side, res_oid, comp_oid", [
    ("5bA", BUY, 202, 3031),     # (1a) T=20, competitor ts19
    ("5bB", SELL, 7003, 70049),  # (1c) T=50, competitor ts49
])
def test_earlier_second_competitor_never_overtaken(name, side, res_oid, comp_oid):
    """F4 / interaction (vii): the restamp never ranks the residual ahead of
    a same-price same-side order stamped in an EARLIER second.

    An earlier same-price order cannot rest BEFORE the aggressor's submit
    without having been crossed by the liquidity the aggressor took. So
    the competitor is submitted after the residual rests, with a stamp one
    second before T.
    """
    run = _run(name)
    pre, post = run.states[-2], run.states[-1]
    changed = np.asarray(post.lob.trades)[
        np.any(np.asarray(pre.lob.trades) != np.asarray(post.lob.trades), axis=1)]
    assert changed.shape[0] == 1
    assert int(changed[0, _T_PASS_OID]) == comp_oid
    assert _rows(post, side, comp_oid) == []
    _, rrow = _single_row(post, side, res_oid)
    assert int(rrow[_SEC]) >= int(run.orders[1].timestamp)


# ---------------------------------------------------------------------------
# (6) do_resubmit truncation branch.                    RED on unfixed code.
# ---------------------------------------------------------------------------


def test_resubmit_truncation_branch_rests_at_T():
    """The loop exhausts _TRIP_BOUND, then ``do_resubmit`` rests the residual.

    Observed and NOT asserted as spec (S1/O5): the iceberg is left with
    ``n_active == 1`` and an empty ask side, because ``do_resubmit`` has no
    following regeneration. Note: ``truncated`` is already True after the
    ICE SELL registration submit on the unfixed code.
    """
    run = _run("6")
    state, r = run.states[1], run.results[1]
    assert bool(state.truncated) is True
    assert float(r.trade_qty) == pytest.approx(10.1, abs=0.05)
    assert float(r.trade_price) == pytest.approx(50.00, abs=0.02)
    assert int(state.ts_counter) == 222
    _assert_residual_at_T(state, run.orders[1], 0.9)   # unfixed: SEC 221


# ---------------------------------------------------------------------------
# (7) Saturated tape (the R2 guard).         RED on unfixed code (stamp only).
# ---------------------------------------------------------------------------


def test_saturated_tape_iceberg_sweep():
    run = _run("7")
    sat_state = run.states[199]
    assert int(np.count_nonzero(np.any(np.asarray(sat_state.lob.trades) != 0,
                                       axis=1))) == 100, "fixture: tape not full"
    state, r = run.states[201], run.results[201]
    assert float(r.trade_qty) == pytest.approx(3.0, abs=_QTY_ABS)
    assert int(state.n_active) == 0
    assert _live(state, SELL).shape[0] == 0
    _assert_residual_at_T(state, run.orders[201], 2.0)  # unfixed: SEC 106


# ---------------------------------------------------------------------------
# (8) Trades tape + ts_counter byte-identical.        GREEN on unfixed code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_TAPE_COUNTER_PINS))
def test_trades_tape_and_counter_byte_identical(name):
    idx, tape_h, ts = _TAPE_COUNTER_PINS[name]
    run = _run(name)
    assert _h(run.states[idx].lob.trades) == tape_h
    assert int(run.states[idx].ts_counter) == ts
    if name in _REST:
        # Up to AND including the submit at which the residual rests.
        assert _h(run.states[_REST[name]].lob.trades) == tape_h
    if name == "11":
        assert [int(s.ts_counter) for s in run.states] == _MIXED_TS_COUNTER


@pytest.mark.parametrize("name", ["4a", "4b", "4c"])
def test_probe_trade_differs_only_in_passive_cells(name):
    """(8) for the (4) probe: the new tape row equals the unfixed-code row
    except PASS_OID / PASS_TID, which name the residual instead of the
    competitor. ``ts_counter`` is unchanged throughout. (Also RED on the
    unfixed code, because the passive side is the fix itself.)"""
    res_oid, _, _, _, main_row = _SAME_SECOND[name]
    run = _run(name)
    rest_idx, tape_h, _ = _TAPE_COUNTER_PINS[name]
    assert _h(run.states[1].lob.trades) == tape_h
    pre, post = np.asarray(run.states[2].lob.trades), np.asarray(run.states[3].lob.trades)
    diff_rows = np.where(np.any(pre != post, axis=1))[0]
    assert diff_rows.shape[0] == 1
    expected = list(main_row)
    expected[_T_PASS_OID] = res_oid
    expected[_T_PASS_TID] = res_oid
    assert post[diff_rows[0]].tolist() == expected
    other = np.ones(pre.shape[0], dtype=bool)
    other[diff_rows[0]] = False
    assert np.array_equal(pre[other], post[other])
    probe_ts = {"4a": 32, "4b": 32, "4c": 202}[name]
    assert int(run.states[3].ts_counter) == probe_ts


# ---------------------------------------------------------------------------
# (9) SubmitResult unchanged.                         GREEN on unfixed code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_SUBMIT_RESULT_PINS))
def test_submit_result_unchanged(name):
    run = _run(name)
    for idx, (acc, traded, q, p, sv) in _SUBMIT_RESULT_PINS[name].items():
        r = run.results[idx]
        tag = f"{name}[{idx}]"
        assert bool(r.accepted) is acc, tag
        assert bool(r.traded) is traded, tag
        assert float(r.trade_qty) == pytest.approx(
            q, abs=0.05 if name == "6" else _QTY_ABS), tag
        assert float(r.trade_price) == pytest.approx(p, abs=_PRICE_ABS), tag
        assert float(r.settlement_value) == pytest.approx(sv, abs=_SETTLE_ABS), tag


# ---------------------------------------------------------------------------
# (10) No-fill rest and other paths unchanged.        GREEN on unfixed code.
# ---------------------------------------------------------------------------


def test_no_fill_rest_and_other_paths_unchanged_10a_non_crossing_rests_at_T():
    run = _run("10a")
    s = run.states[1]
    _assert_residual_at_T(s, run.orders[1], 2.0)
    assert (_h(s.lob.bids), _h(s.lob.asks)) == ("01a5fdfd6f51c1a9", "ff2c37b78aba0dc3")


def test_no_fill_rest_and_other_paths_unchanged_10b_ioc():
    run = _run("10b")
    s = run.states[1]
    assert _live(s, BUY).shape[0] == 0
    assert (_h(s.lob.bids), _h(s.lob.asks), _h(s.lob.trades), int(s.ts_counter)) == (
        _EMPTY_SIDE_H, _EMPTY_SIDE_H, "8b1f4f91b06dbef8", 22)


@pytest.mark.parametrize("name, pins", [
    ("10c_acc", (_EMPTY_SIDE_H, _EMPTY_SIDE_H, "fb30438f05970a79", 22)),
    ("10c_rej", (_EMPTY_SIDE_H, "c82dc179c31c477f", _EMPTY_TAPE_H, 11)),
])
def test_no_fill_rest_and_other_paths_unchanged_10c_fok(name, pins):
    s = _run(name).states[1]
    assert (_h(s.lob.bids), _h(s.lob.asks), _h(s.lob.trades), int(s.ts_counter)) == pins


def test_no_fill_rest_and_other_paths_unchanged_10d_incoming_ice():
    # Peak with no cross rests at T.
    run = _run("10d_peak")
    _assert_residual_at_T(run.states[1], run.orders[1], 1.0)
    assert _h(run.states[1].lob.bids) == "fcc08d782aab5e96"
    # S4 (OMIE-ICE-INCOMING-HOP-1) re-baseline: the incoming ICE sweeps its
    # total 3.0 MW (fills 1.5), then the remainder R = 1.5 registers and shows
    # its FIRST slice at the entry stamp (SEC == T, NSEC 0; Annex §4.4).
    # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): M = 15, P = 10 -> k_e = 1,
    # u = 5; the first slice is v_1 = min(10 - 5, 15) = 0.5 MW (spec §4.2.1
    # Rule 2 item (4)), not min(peak, R) = 1.0.
    run = _run("10d_regen")
    s, T = run.states[1], int(run.orders[1].timestamp)
    _, row = _single_row(s, BUY, 818)
    assert int(row[_Q]) == _qty_ticks(0.5)
    assert int(row[_SEC]) == T
    assert int(row[_NSEC]) == 0
    assert int(s.n_active) == 1
    # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): bids digest of the S4 book
    # "fcc08d782aab5e96" (= the 10d_peak bids above) with only 818's Q cell
    # 10 -> 5 (same slot, P, OID, TID, SEC, NSEC).
    assert (_h(s.lob.bids), _h(s.lob.asks)) == ("f911edbc8879c0e7", _EMPTY_SIDE_H)


def test_no_fill_rest_and_other_paths_unchanged_10e_no_fill_restamp_gate():
    """F5: a NON submit that does NOT fill must not restamp an older resting
    row that shares its order_id. GREEN on the unfixed code; FAILS under an
    ungated restamp write."""
    run = _run("10e")
    s = run.states[1]
    arr = _side_arr(s, BUY)
    old = arr[(arr[:, _OID] == 77) & (arr[:, _P] == 1004400)]
    new = arr[(arr[:, _OID] == 77) & (arr[:, _P] == 1004300)]
    assert old.shape[0] == 1 and new.shape[0] == 1
    assert int(old[0, _SEC]) == 5
    assert int(new[0, _SEC]) == 30
    assert (_h(s.lob.bids), _h(s.lob.asks), _h(s.lob.trades), int(s.ts_counter)) == (
        "a2bf39aff4bb296c", _EMPTY_SIDE_H, _EMPTY_TAPE_H, 31)


# ---------------------------------------------------------------------------
# (11) Mixed sequence.
# ---------------------------------------------------------------------------


def test_mixed_sequence_ts_counter_monotone():
    """(11a) guard: GREEN on unfixed code."""
    run = _run("11")
    seq = [int(run.init_state.ts_counter)] + [int(s.ts_counter) for s in run.states]
    assert all(b >= a for a, b in zip(seq, seq[1:])), seq


def test_mixed_sequence_every_filled_non_residual_rests_at_T():
    """(11b): every NON submit that filled and left a resting residual has
    SEC == its order.timestamp, right after the submit and at the end.
    RED on unfixed code (oid 55: SEC 16 vs T=9; oid 98765: SEC 27 vs T=5)."""
    run = _run("11")
    checked = []
    for o, r, s in zip(run.orders, run.results, run.states):
        if int(o.exec_restriction) != int(NON) or not bool(r.traded):
            continue
        rows = _rows(s, int(o.side), int(o.order_id))
        if not rows:
            continue
        checked.append(int(o.order_id))
        for _, row in rows:
            assert (int(row[_SEC]), int(row[_NSEC])) == (int(o.timestamp), 0), (
                f"oid {int(o.order_id)}: rests at SEC={int(row[_SEC])}, "
                f"expected T={int(o.timestamp)}")
    assert checked == [55, 98765], f"fixture drift: {checked}"
    final = run.states[-1]
    _, row = _single_row(final, SELL, 98765)
    assert int(row[_Q]) == _qty_ticks(0.3)
    assert int(row[_SEC]) == 5
