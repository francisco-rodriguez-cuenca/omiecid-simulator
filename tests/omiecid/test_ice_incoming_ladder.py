"""Locked behavioural-contract tests for Story OMIE-ICE-INCOMING-LADDER-1 (S2).

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-ICE-INCOMING-LADDER-1 — an
incoming iceberg sweeps as a chain of instances (E-A, E-B, E-C) (S2)", its
"Test-first contract" row items (1)-(13), as amended by "Step 2 amendment" and
"Step 2 amendment 2 (2026-10-20, second-architect findings reconciled)"
(amendment 2 wins where they conflict), plus the "Byte-identical" row.

Rule (market-spec §4.2.1 Rule 2 *Incoming iceberg* items (1)-(6), amended
2026-10-20; Rule 3 *ICE aggressor*; OQ-33 revised, OQ-34 resolved). With $b$ the
incoming iceberg's price, $P$ its peak, $d$ its price increment (0 for a
standard iceberg), $M$ the quantity it sweeps on entry and $R$ its remainder:

* The incoming iceberg is a chain of instances of size $P$. Instance $k$ trades
  only where its own limit $b + k d$ crosses; the sweep stops at the first
  instance that cannot trade (even if $b$ would still cross).
* Every trade prices at the resting entry (for a resting ICEwPI, the matched
  slice's drifted price).
* A trade row ends where a resting entry ends OR where one incoming instance
  ends and the next begins.
* The remainder keeps its place in the ladder: with $k_e = \\lfloor M/P \\rfloor$,
  $u = M - k_e P$, it shows $v_1 = \\min(P - u, R)$ at $b + k_e d$ and keeps
  $R - v_1$ hidden; the next regenerated slice is priced $b + (k_e + 1) d$.
* Simulator limit: on a trade ring without room for the split rows the rows
  stay unsplit (all or nothing).

Contract map (card numbering):
  (1)  GESTVD2 replica (SELL ICEwPI, d = +0.75).          RED on main.
  (2)  part-used instance (BUY ICEwPI, d = -0.50, u > 0). RED on main.
  (3)  stop with crossing-at-base liquidity left, both sides.  RED on main.
  (4)  resting iceberg as the counter at a step price.     RED on main.
  (5)  FOK after entry; FOK quantity strictly between the two projections.
                                                           RED on main.
  (6)  standard iceberg row split (u = 0).                 RED on main.
  (7)  standard display after a part-used instance (u > 0). RED on main.
  (8)  split fallback (too little room: unsplit; saturated ring: byte no-op
       against frozen main).        applies-case RED; fallback cases GREEN.
  (9a) NON / IOC / FOK sessions (no ICE, or only non-crossing / aligned u = 0,
       d = 0 ICE): full byte identity with frozen main.    GREEN.
  (9b) standard-ICE sessions with u = 0: every non-tape leaf + SubmitResult
       identical to frozen main; the tape via (13).        GREEN.
  (9c) standard-ICE entries with u > 0: per call, SubmitResult, ts_counter,
       n_active, book_full, truncated and the opposite side identical to frozen
       main.                                               GREEN.
  (10) E-C properties: (i) unit tests of the pure helper
       ``_split_incoming_rows(trades_before, trades_after, peak_int, order_id)``
       (RED on main: the helper does not exist); (ii) under E-A with d != 0 no
       own row spans a multiple of P (RED on main); (ii-b) standard ICE: every
       own piece <= P and every multiple jP < M is a boundary (RED on main);
       (iii) non-tape leaves do not depend on whether the split applied (GREEN).
  (11) eager = jit = each vmap lane, mixed ICEwPI / ICE / NON lanes, incl.
       d != 0, k_e > 0, u > 0.                              slow; GREEN on main.
  (12) an overflowed table (d != 0, k_e > 0, u > 0; and d = 0, u > 0).
                                                            slow; RED on main.
  (13) tape: merging each call's split rows back (every column except Q,
       contiguous runs) reproduces the E-B tape (= frozen main tape on these
       d = 0, u = 0 sessions) byte for byte (RED on main only for the expected-
       tape half); the routine file's hash pins are a REGRESSION TRIPWIRE
       derived from this file's own split walk, not the independent audit pins.

FROZEN VALUES: every ``_MAIN_*`` constant below was frozen by running the
unchanged engine on branch ``omie/OMIE-ICE-INCOMING-LADDER-1-instance-chain-sweep``
@ 2b3818f, whose ``omiecid_simulator/`` is identical to main engine @ 4000bc7
(``git diff 4000bc7 HEAD -- omiecid_simulator/`` empty), CPU, 2026-10-20. They are
"frozen on main engine @ 4000bc7-equivalent". They are never re-baselined by
hand: a moved pin is a byte-identity break.

Units: prices in int ticks (``round((eur + 9999) / 0.01)``; 50.00 EUR ->
1004900), quantities in 0.1 MW ticks. Trade-row Q is signed by jaxob (negative
for a BUY aggressor); this file compares ``|Q|`` except where noted.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from omiecid_simulator.omiecid import contract_book as _cb
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, _submit_order
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_BASE_PRICE_INT,
    COL_ORDER_ID,
    COL_PEAK_QTY,
    COL_REMAINING_HIDDEN,
    COL_SLICE_COUNT,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import OMIEOrder

# Every test here runs the jitted submit path (T1.1 marker lint).
pytestmark = pytest.mark.jit

PC = OMIEProductConfig()
MC = OMIEMarketConfig()
N_RING = MC.nTrades  # 100

BUY, SELL = 0, 1
NON, IOC, FOK, ICE = 0, 1, 2, 3
CID = 37

# Trade-ring columns (jaxob TradesFeat).
T_P, T_Q, T_POID, T_AOID, T_SEC, T_NSEC, T_PTID, T_ATID = range(8)
# Book-side columns (jaxob OrderSideFeat).
S_P, S_Q, S_OID = 0, 1, 2


# ---------------------------------------------------------------------------
# Encoding and construction
# ---------------------------------------------------------------------------
def px(eur: float) -> int:
    """EUR/MWh -> jaxob price tick (offset encoding, market-spec §2.3)."""
    return int(round((eur + 9999.0) / 0.01))


def qt(mw: float) -> int:
    """MW -> 0.1 MW quantity tick."""
    return int(round(mw * 10))


def _o(side, price, qty, oid, ts, ex=NON, peak=0.0, ppd=0.0, agent=1) -> OMIEOrder:
    """A typed OMIEOrder (fixed dtypes: one compile for every order)."""
    return OMIEOrder(
        contract_id=jnp.int32(CID), side=jnp.int32(side),
        price=jnp.float32(price), quantity=jnp.float32(qty),
        order_id=jnp.int32(oid), agent_id=jnp.int32(agent),
        timestamp=jnp.int32(ts), exec_restriction=jnp.int32(ex),
        validity=jnp.int32(0), expiry_time=jnp.int32(0), status=jnp.int32(0),
        peak_qty=jnp.float32(peak), ppd=jnp.float32(ppd))


def _new_book(mc: OMIEMarketConfig = MC) -> ContractBook:
    contract = Contract(contract_id=CID, delivery_start_minutes=555,
                        delivery_end_minutes=570, gct_minutes=495,
                        status=ContractStatus.OPEN)
    return ContractBook(contract, PC, mc)


@pytest.fixture(scope="module")
def book() -> ContractBook:
    return _new_book()


# ---------------------------------------------------------------------------
# Observation helpers (public state only)
# ---------------------------------------------------------------------------
def _call_rows(before, after) -> np.ndarray:
    """Rows the call wrote (whole-row diff of the ring, slot order)."""
    b = np.asarray(before.lob.trades)
    a = np.asarray(after.lob.trades)
    return a[np.any(a != b, axis=1)]


def _fills(rows: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """``(price_tick, |qty| tick, passive_oid, aggressor_oid)`` per row."""
    return [(int(r[T_P]), abs(int(r[T_Q])), int(r[T_POID]), int(r[T_AOID]))
            for r in rows]


def _submit(book, state, order):
    new, res = book.submit_order(state, order)
    return new, res, _call_rows(state, new)


def _side(state, side) -> np.ndarray:
    return np.asarray(state.lob.bids if side == BUY else state.lob.asks)


def _shown(state, side, oid) -> List[Tuple[int, int]]:
    """Visible rows ``(price_tick, qty_tick)`` of ``oid`` on ``side``."""
    arr = _side(state, side)
    return [(int(r[S_P]), int(r[S_Q])) for r in arr
            if int(r[S_OID]) == oid and int(r[S_Q]) > 0 and int(r[S_P]) >= 0]


def _live(state, side) -> List[Tuple[int, int, int]]:
    """All live ``(price, qty, oid)`` on ``side``, sorted by price then oid."""
    arr = _side(state, side)
    rows = [(int(r[S_P]), int(r[S_Q]), int(r[S_OID])) for r in arr
            if int(r[S_P]) >= 0 and int(r[S_Q]) > 0]
    return sorted(rows)


def _assert_uncrossed(state):
    bids, asks = _live(state, BUY), _live(state, SELL)
    if bids and asks:
        assert max(p for p, _, _ in bids) < min(p for p, _, _ in asks), "book crossed"


def _has_meta(state, oid) -> bool:
    meta = np.asarray(state.iceberg_meta)[: int(state.n_active)]
    return bool(np.any(meta[:, COL_ORDER_ID] == oid))


def _h(*arrs) -> str:
    m = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str((a.dtype.str, a.shape)).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def _res_arrays(res):
    return [np.asarray(x) for x in res]


def _leaves(state, res, *, tape: bool, opposite_of=None) -> Dict[str, str]:
    """Per-leaf digests (``LobState.key`` excluded)."""
    d = {
        "result": _h(*_res_arrays(res)),
        "bids": _h(state.lob.bids),
        "asks": _h(state.lob.asks),
        "iceberg_meta": _h(state.iceberg_meta),
        "n_active": _h(state.n_active),
        "ts_counter": _h(state.ts_counter),
        "truncated": _h(state.truncated),
        "book_full": _h(state.book_full),
    }
    if tape:
        d["trades"] = _h(state.lob.trades)
    return d


def _combined(d: Dict[str, str]) -> str:
    return hashlib.sha256(repr(sorted(d.items())).encode()).hexdigest()[:16]


def _c9c_digest(state, res, entry_side) -> str:
    """(9c) per-call leaves: SubmitResult, ts_counter, n_active, book_full,
    truncated and the side opposite the entry."""
    opp = state.lob.asks if entry_side == BUY else state.lob.bids
    return _h(*_res_arrays(res), state.ts_counter, state.n_active,
              state.book_full, state.truncated, opp)


def _run(book, orders, state=None):
    """Submit every order; return [(state_before, state_after, result)]."""
    s = book.init() if state is None else state
    out = []
    for od in orders:
        s2, r = book.submit_order(s, od)
        assert bool(r.accepted), f"order {od.order_id} rejected"
        out.append((s, s2, r))
        s = s2
    return out


def _cum_intervals(qtys: Sequence[int]) -> List[Tuple[int, int]]:
    c, out = 0, []
    for q in qtys:
        out.append((c, c + q))
        c += q
    return out


def _spans_multiple(lo: int, hi: int, p: int) -> bool:
    """True when some multiple of p lies strictly inside (lo, hi)."""
    return hi - lo > 0 and (lo // p) != ((hi - 1) // p)


# ===========================================================================
# (1) GESTVD2 replica (orders_20250610.1:125794)
# ===========================================================================
GEST_OID = 125794
GEST_SETUP = [
    _o(BUY, 112.36, 1.0, 11, 1, agent=2),
    _o(BUY, 112.34, 20.5, 2207, 2, agent=3),
    _o(BUY, 112.00, 11.9, 33, 3, agent=4),
    _o(BUY, 110.00, 10.0, 640017, 4, agent=5),
]
GEST_ICE = _o(SELL, 110.00, 100.0, GEST_OID, 10, ICE, peak=5.0, ppd=0.75, agent=6)


@pytest.fixture(scope="module")
def gest(book):
    s = _run(book, GEST_SETUP)[-1][1]
    return _submit(book, s, GEST_ICE)


class TestGestvd2Replica:
    """SELL ICE 110.00, 100, P = 5, d = +0.75. Instance limits 110.00, 110.75,
    111.50, 112.25, 113.00. Instance 0 takes 1.0 @ 112.36 and 4.0 @ 112.34;
    instances 1-3 take 5.0 each @ 112.34; instance 4 (113.00) cannot trade.
    M = 20, k_e = 4, u = 0, R = 80: shows 5.0 @ 113.00, hidden 75."""

    def test_trades_split_at_instance_boundaries(self, gest):
        _, res, rows = gest
        assert _fills(rows) == [
            (px(112.36), 10, 11, GEST_OID),
            (px(112.34), 40, 2207, GEST_OID),
            (px(112.34), 50, 2207, GEST_OID),
            (px(112.34), 50, 2207, GEST_OID),
            (px(112.34), 50, 2207, GEST_OID),
        ]
        assert float(res.trade_qty) == pytest.approx(20.0, abs=1e-4)

    def test_bids_left(self, gest):
        s, _, _ = gest
        assert _live(s, BUY) == sorted([
            (px(112.34), 15, 2207), (px(112.00), 119, 33), (px(110.00), 100, 640017)])

    def test_remainder_shows_peak_at_instance_4_price(self, gest):
        s, _, _ = gest
        assert _shown(s, SELL, GEST_OID) == [(px(113.00), 50)]
        _assert_uncrossed(s)

    def test_later_buy_prints_rebased_ladder(self, book, gest):
        """BUY 7.0 @ 114.00 trades 5.0 @ 113.00 then 2.0 @ 113.75 (slice k_e+1)."""
        s, _, _ = gest
        s2, res, rows = _submit(book, s, _o(BUY, 114.00, 7.0, 3071, 20, agent=7))
        assert _fills(rows) == [(px(113.00), 50, GEST_OID, 3071),
                                (px(113.75), 20, GEST_OID, 3071)]
        assert _shown(s2, SELL, GEST_OID) == [(px(113.75), 30)]

    def test_remainder_total_and_ladder_drain(self, book, gest):
        """A BUY 200 @ 200.00 drains exactly R = 80: 16 slices of 5.0 priced
        113.00 + 0.75 j, j = 0..15 (hidden 75 = R - v_1)."""
        s, _, _ = gest
        s2, res, rows = _submit(book, s, _o(BUY, 200.00, 200.0, 48, 21, agent=8))
        f = _fills(rows)
        assert [p for p, _, _, _ in f] == [px(113.00 + 0.75 * j) for j in range(16)]
        assert [q for _, q, _, _ in f] == [50] * 16
        assert _shown(s2, SELL, GEST_OID) == []
        assert not _has_meta(s2, GEST_OID)

    def test_remainder_ranks_by_entry_time_at_rebased_price(self, book):
        """§6.1 at the rebased price: a SELL NON @ 113.00 entered before the
        iceberg trades first; one entered after trades after the remainder."""
        s = _run(book, GEST_SETUP + [_o(SELL, 113.00, 1.0, 3, 5, agent=9)])[-1][1]
        s, _, _ = _submit(book, s, GEST_ICE)
        s, _, _ = _submit(book, s, _o(SELL, 113.00, 1.0, 4, 30, agent=9))
        _, _, rows = _submit(book, s, _o(BUY, 113.00, 7.5, 5, 31, agent=7))
        assert _fills(rows) == [(px(113.00), 10, 3, 5), (px(113.00), 50, GEST_OID, 5),
                                (px(113.00), 10, 4, 5)]


# ===========================================================================
# (2) Part-used instance (BUY ICEwPI, d = -0.50)
# ===========================================================================
PART_OID = 4071
PART_SETUP = [_o(SELL, 50.00, 3.0, 61, 1, agent=2)]
PART_ICE = _o(BUY, 51.00, 10.0, PART_OID, 5, ICE, peak=2.0, ppd=-0.50, agent=3)


@pytest.fixture(scope="module")
def part(book):
    s = _run(book, PART_SETUP)[-1][1]
    return _submit(book, s, PART_ICE)


class TestPartUsedInstance:
    """Instance 0 (51.00) takes 2.0 @ 50.00; instance 1 (50.50) takes 1.0 @
    50.00. M = 3, k_e = 1, u = 1, R = 7: shows v_1 = min(2 - 1, 7) = 1.0 @
    50.50; the next slices are priced 50.00, 49.50, 49.00."""

    def test_trades(self, part):
        _, res, rows = part
        assert _fills(rows) == [(px(50.00), 20, 61, PART_OID),
                                (px(50.00), 10, 61, PART_OID)]
        assert float(res.trade_qty) == pytest.approx(3.0, abs=1e-4)

    def test_remainder_shows_unused_part_of_instance_1(self, part):
        s, _, _ = part
        assert _shown(s, BUY, PART_OID) == [(px(50.50), 10)]
        assert _live(s, SELL) == []

    def test_later_sell_prints_ladder(self, book, part):
        s, _, _ = part
        s2, res, rows = _submit(book, s, _o(SELL, 49.00, 7.0, 90210, 9, agent=4))
        assert _fills(rows) == [(px(50.50), 10, PART_OID, 90210),
                                (px(50.00), 20, PART_OID, 90210),
                                (px(49.50), 20, PART_OID, 90210),
                                (px(49.00), 20, PART_OID, 90210)]
        assert _shown(s2, BUY, PART_OID) == []
        assert _live(s2, SELL) == []


# ===========================================================================
# Varied ICEwPI entries (incidental constants: P 0.1-50 MW, |d| 1-300 ticks)
# ===========================================================================
TINY_SETUP = [_o(BUY, 81.40, 0.3, 18, 1, agent=2), _o(BUY, 81.39, 0.4, 900001, 2, agent=3)]
TINY_ICE = _o(SELL, 81.37, 2.3, 52, 3, ICE, peak=0.1, ppd=0.01, agent=4)

BIG_SETUP = [_o(SELL, 115.20, 30.0, 7, 1, agent=2), _o(SELL, 116.00, 45.5, 31415, 2, agent=3),
             _o(SELL, 117.43, 40.0, 880014, 3, agent=4)]
BIG_ICE = _o(BUY, 120.45, 180.0, 6020, 4, ICE, peak=50.0, ppd=-3.00, agent=5)

MID_SETUP = [_o(BUY, 34.50, 0.9, 18, 1, agent=2), _o(BUY, 34.00, 1.0, 4071, 2, agent=3),
             _o(BUY, 33.20, 1.1, 900001, 3, agent=4)]
MID_ICE = _o(SELL, 33.33, 7.7, 512, 4, ICE, peak=1.2, ppd=0.40, agent=5)


class TestVariedIcewpiEntries:

    def test_peak_tenth_mw_one_tick_increment(self, book):
        """P = 0.1, d = +0.01: instances 81.37, 81.38, 81.39 each take 0.1 @
        81.40; instance 3 (81.40) cannot reach the 81.39 bid. M = 0.3, k_e = 3,
        u = 0: shows 0.1 @ 81.40; the 81.39 bid stays whole."""
        s = _run(book, TINY_SETUP)[-1][1]
        s2, _, rows = _submit(book, s, TINY_ICE)
        assert _fills(rows) == [(px(81.40), 1, 18, 52)] * 3
        assert _shown(s2, SELL, 52) == [(px(81.40), 1)]
        assert _live(s2, BUY) == [(px(81.39), 4, 900001)]
        _assert_uncrossed(s2)

    def test_peak_fifty_mw_three_hundred_tick_increment(self, book):
        """P = 50, d = -3.00: instance 0 (120.45) takes 30.0 @ 115.20 + 20.0 @
        116.00; instance 1 (117.45) takes 25.5 @ 116.00 + 24.5 @ 117.43;
        instance 2 (114.45) cannot trade. M = 100, k_e = 2, u = 0, R = 80: shows
        50.0 @ 114.45; 15.5 @ 117.43 stays."""
        s = _run(book, BIG_SETUP)[-1][1]
        s2, res, rows = _submit(book, s, BIG_ICE)
        assert _fills(rows) == [(px(115.20), 300, 7, 6020), (px(116.00), 200, 31415, 6020),
                                (px(116.00), 255, 31415, 6020), (px(117.43), 245, 880014, 6020)]
        assert _shown(s2, BUY, 6020) == [(px(114.45), 500)]
        assert _live(s2, SELL) == [(px(117.43), 155, 880014)]
        assert float(res.trade_qty) == pytest.approx(100.0, abs=1e-3)

    def test_sell_part_used_instance_and_later_prints(self, book):
        """P = 1.2, d = +0.40: instance 0 (33.33) takes 0.9 @ 34.50 + 0.3 @
        34.00; instance 1 (33.73) takes the last 0.7 @ 34.00 and cannot reach
        33.20. M = 1.9, k_e = 1, u = 0.7, R = 5.8: shows 0.5 @ 33.73. A later
        BUY 1.0 @ 34.20 prints 0.5 @ 33.73 then 0.5 @ 34.13 (slice 2)."""
        s = _run(book, MID_SETUP)[-1][1]
        s2, _, rows = _submit(book, s, MID_ICE)
        assert _fills(rows) == [(px(34.50), 9, 18, 512), (px(34.00), 3, 4071, 512),
                                (px(34.00), 7, 4071, 512)]
        assert _shown(s2, SELL, 512) == [(px(33.73), 5)]
        assert _live(s2, BUY) == [(px(33.20), 11, 900001)]
        _, _, rows2 = _submit(book, s2, _o(BUY, 34.20, 1.0, 66, 9, agent=6))
        assert _fills(rows2) == [(px(33.73), 5, 512, 66), (px(34.13), 5, 512, 66)]


# ===========================================================================
# (3) Stop with crossing-at-base liquidity left resting, both sides
# ===========================================================================
class TestStopLeavesBaseCrossingLiquidity:

    def test_sell_stops_at_first_instance_that_cannot_trade(self, book):
        """SELL ICE 60.00, 6.0, P = 2, d = +1.00 vs bids 2.0 @ 61.50, 3.0 @
        60.00: 2.0 @ 61.50, then instance 1 (61.00) cannot trade. The bid 3.0 @
        60.00 stays; the remainder 4.0 shows 2.0 @ 61.00."""
        s = _run(book, [_o(BUY, 61.50, 2.0, 74, 1, agent=2),
                        _o(BUY, 60.00, 3.0, 3301, 2, agent=3)])[-1][1]
        s2, res, rows = _submit(book, s, _o(SELL, 60.00, 6.0, 9, 3, ICE, peak=2.0,
                                            ppd=1.00, agent=4))
        assert _fills(rows) == [(px(61.50), 20, 74, 9)]
        assert _live(s2, BUY) == [(px(60.00), 30, 3301)]
        assert _shown(s2, SELL, 9) == [(px(61.00), 20)]
        assert float(res.trade_qty) == pytest.approx(2.0, abs=1e-4)
        _assert_uncrossed(s2)

    def test_buy_mirror(self, book):
        """BUY ICE 47.37, 7.3, P = 2, d = -2.87 vs asks 2.0 @ 45.87, 3.0 @
        47.37: 2.0 @ 45.87, then instance 1 (44.50) cannot trade. The ask 3.0 @
        47.37 stays; R = 5.3 shows 2.0 @ 44.50."""
        s = _run(book, [_o(SELL, 45.87, 2.0, 270001, 1, agent=2),
                        _o(SELL, 47.37, 3.0, 88, 2, agent=3)])[-1][1]
        s2, _, rows = _submit(book, s, _o(BUY, 47.37, 7.3, 640, 3, ICE, peak=2.0,
                                          ppd=-2.87, agent=4))
        assert _fills(rows) == [(px(45.87), 20, 270001, 640)]
        assert _live(s2, SELL) == [(px(47.37), 30, 88)]
        assert _shown(s2, BUY, 640) == [(px(44.50), 20)]
        _assert_uncrossed(s2)


# ===========================================================================
# (4) A resting iceberg as the counter at a step price
# ===========================================================================
class TestRestingIcebergCounter:

    def test_icewpi_vs_resting_icewpi(self, book):
        """Resting SELL ICE 50.00, 6.0, P_r = 1.5, d_r = +0.10 (slices 50.00,
        50.10, 50.20, 50.30). Incoming BUY ICE 50.60, 5.0, P = 2, d = -0.20
        (limits 50.60, 50.40, 50.20). One row per (resting slice, incoming
        instance), each priced by the resting slice:
        1.5 @ 50.00 (r0,i0), 0.5 @ 50.10 (r1,i0), 1.0 @ 50.10 (r1,i1),
        1.0 @ 50.20 (r2,i1), 0.5 @ 50.20 (r2,i2); slice 3 (50.30) does not
        cross instance 2 (50.20). M = 4.5, k_e = 2, u = 0.5, R = 0.5: shows
        0.5 @ 50.20."""
        s = _run(book, [_o(SELL, 50.00, 6.0, 707, 1, ICE, peak=1.5, ppd=0.10, agent=2)])[-1][1]
        s2, _, rows = _submit(book, s, _o(BUY, 50.60, 5.0, 29, 2, ICE, peak=2.0,
                                          ppd=-0.20, agent=3))
        assert _fills(rows) == [(px(50.00), 15, 707, 29), (px(50.10), 5, 707, 29),
                                (px(50.10), 10, 707, 29), (px(50.20), 10, 707, 29),
                                (px(50.20), 5, 707, 29)]
        assert _shown(s2, BUY, 29) == [(px(50.20), 5)]
        assert _shown(s2, SELL, 707) == [(px(50.30), 15)]
        _assert_uncrossed(s2)

    def test_standard_incoming_vs_resting_icewpi(self, book):
        """Resting SELL ICE 50.00, 4.5, P_r = 1.5, d_r = +0.05 (50.00, 50.05,
        50.10). Incoming BUY ICE 50.40, 4.5, P = 2 (d = 0): rows split at both
        boundary sets {1.5, 3.0} and {2.0, 4.0}: 1.5 @ 50.00, 0.5 @ 50.05,
        1.0 @ 50.05, 1.0 @ 50.10, 0.5 @ 50.10. Nothing rests."""
        s = _run(book, [_o(SELL, 50.00, 4.5, 33012, 1, ICE, peak=1.5, ppd=0.05, agent=2)])[-1][1]
        s2, _, rows = _submit(book, s, _o(BUY, 50.40, 4.5, 6, 2, ICE, peak=2.0, agent=3))
        assert _fills(rows) == [(px(50.00), 15, 33012, 6), (px(50.05), 5, 33012, 6),
                                (px(50.05), 10, 33012, 6), (px(50.10), 10, 33012, 6),
                                (px(50.10), 5, 33012, 6)]
        assert _live(s2, BUY) == [] and _live(s2, SELL) == []


# ===========================================================================
# (5) FOK after entry
# ===========================================================================
FOKE_SETUP = [_o(BUY, 63.00, 4.0, 405, 1, agent=2)]
FOKE_ICE = _o(SELL, 60.00, 10.0, 7777, 2, ICE, peak=2.0, ppd=0.50, agent=3)


@pytest.fixture(scope="module")
def foke(book):
    s = _run(book, FOKE_SETUP)[-1][1]
    return _submit(book, s, FOKE_ICE)


class TestFokAfterEntry:
    """SELL ICE 60.00, 10, P = 2, d = +0.50 vs bid 4.0 @ 63.00: instances 0 and
    1 take 2.0 each; M = 4, k_e = 2, u = 0, R = 6: shows 2.0 @ 61.00, then
    slices 61.50, 62.00. A BUY FOK @ 61.20 can see 2.0 (new projection), where
    a remainder resting at the base would offer 6.0 (60.00, 60.50, 61.00)."""

    def test_entry(self, foke):
        s, _, rows = foke
        assert _fills(rows) == [(px(63.00), 20, 405, 7777)] * 2
        assert _shown(s, SELL, 7777) == [(px(61.00), 20)]

    def test_fok_between_projections_is_killed(self, book, foke):
        """FOK 4.0 @ 61.20 lies strictly between 2.0 and 6.0: accepted, no
        trade, book unchanged."""
        s, _, _ = foke
        s2, res, rows = _submit(book, s, _o(BUY, 61.20, 4.0, 3418, 9, FOK, agent=4))
        assert bool(res.accepted) and not bool(res.traded)
        assert rows.shape[0] == 0
        assert _shown(s2, SELL, 7777) == [(px(61.00), 20)]

    def test_fok_within_new_projection_fills(self, book, foke):
        s, _, _ = foke
        _, res, rows = _submit(book, s, _o(BUY, 61.20, 2.0, 3419, 9, FOK, agent=4))
        assert bool(res.traded)
        assert _fills(rows) == [(px(61.00), 20, 7777, 3419)]

    def test_fok_counts_slices_from_k_e(self, book, foke):
        """FOK 3.0 @ 61.60 sees 61.00 (2.0) + 61.50 (2.0): fills 2.0 + 1.0."""
        s, _, _ = foke
        _, res, rows = _submit(book, s, _o(BUY, 61.60, 3.0, 3420, 9, FOK, agent=4))
        assert _fills(rows) == [(px(61.00), 20, 7777, 3420), (px(61.50), 10, 7777, 3420)]


# ===========================================================================
# (6) Standard iceberg split, (7) standard display (with SELL mirrors)
# ===========================================================================
STD6_SETUP = [_o(SELL, 50.00, 6.0, 21, 1, agent=2), _o(SELL, 50.50, 6.0, 404, 2, agent=3)]
STD6_ICE = _o(BUY, 51.00, 20.0, 55, 3, ICE, peak=4.0, agent=4)


class TestStandardIcebergSplit:

    def test_rows_split_at_instance_boundaries(self, book):
        """BUY ICE 51.00, 20, P = 4 vs SELL 6.0 @ 50.00, 6.0 @ 50.50: rows 4.0,
        2.0 @ 50.00 and 2.0, 4.0 @ 50.50; shows 4.0 @ 51.00."""
        s = _run(book, STD6_SETUP)[-1][1]
        s2, res, rows = _submit(book, s, STD6_ICE)
        assert _fills(rows) == [(px(50.00), 40, 21, 55), (px(50.00), 20, 21, 55),
                                (px(50.50), 20, 404, 55), (px(50.50), 40, 404, 55)]
        assert _shown(s2, BUY, 55) == [(px(51.00), 40)]
        assert float(res.trade_qty) == pytest.approx(12.0, abs=1e-4)

    def test_hidden_four(self, book):
        """R = 8 = 4 visible + 4 hidden: a later SELL 10.0 @ 50.90 takes 4.0 +
        4.0 @ 51.00 and rests 2.0."""
        s = _run(book, STD6_SETUP + [STD6_ICE])[-1][1]
        s2, _, rows = _submit(book, s, _o(SELL, 50.90, 10.0, 812, 9, agent=5))
        assert _fills(rows) == [(px(51.00), 40, 55, 812)] * 2
        assert _shown(s2, SELL, 812) == [(px(50.90), 20)]

    def test_sell_mirror(self, book):
        """SELL ICE 38.07, 13, P = 2.5 vs BUY 4.0 @ 38.50, 3.5 @ 38.21: M =
        7.5 = 3P; rows 2.5, 1.5 @ 38.50 and 1.0, 2.5 @ 38.21; shows 2.5 @ 38.07."""
        s = _run(book, [_o(BUY, 38.50, 4.0, 52, 1, agent=2),
                        _o(BUY, 38.21, 3.5, 640001, 2, agent=3)])[-1][1]
        s2, _, rows = _submit(book, s, _o(SELL, 38.07, 13.0, 3, 3, ICE, peak=2.5, agent=4))
        assert _fills(rows) == [(px(38.50), 25, 52, 3), (px(38.50), 15, 52, 3),
                                (px(38.21), 10, 640001, 3), (px(38.21), 25, 640001, 3)]
        assert _shown(s2, SELL, 3) == [(px(38.07), 25)]


STD7_SETUP = [_o(SELL, 50.00, 6.0, 61, 1, agent=2)]
STD7_ICE = _o(BUY, 50.00, 20.0, 9312, 2, ICE, peak=4.0, agent=3)


class TestStandardDisplay:

    def test_rows_and_display(self, book):
        """BUY ICE 50.00, 20, P = 4 vs SELL 6.0 @ 50.00: rows 4.0, 2.0; M = 6,
        k_e = 1, u = 2: rests v_1 = 2.0 @ 50.00."""
        s = _run(book, STD7_SETUP)[-1][1]
        s2, _, rows = _submit(book, s, STD7_ICE)
        assert _fills(rows) == [(px(50.00), 40, 61, 9312), (px(50.00), 20, 61, 9312)]
        assert _shown(s2, BUY, 9312) == [(px(50.00), 20)]

    def test_later_sell_prints_two_then_one(self, book):
        """A later SELL 3.0 @ 50.00 prints 2.0 (v_1) and 1.0 (next slice)."""
        s = _run(book, STD7_SETUP + [STD7_ICE])[-1][1]
        _, _, rows = _submit(book, s, _o(SELL, 50.00, 3.0, 33, 9, agent=4))
        assert _fills(rows) == [(px(50.00), 20, 9312, 33), (px(50.00), 10, 9312, 33)]

    def test_hidden_twelve(self, book):
        """R = 14 = 2 visible + 12 hidden: a SELL 30 @ 49.00 takes 2, 4, 4, 4."""
        s = _run(book, STD7_SETUP + [STD7_ICE])[-1][1]
        s2, _, rows = _submit(book, s, _o(SELL, 49.00, 30.0, 34, 9, agent=4))
        assert _fills(rows) == [(px(50.00), 20, 9312, 34)] + [(px(50.00), 40, 9312, 34)] * 3
        assert _shown(s2, SELL, 34) == [(px(49.00), 160)]

    def test_sell_mirror(self, book):
        """SELL ICE 38.07, 9.0, P = 2.5 vs BUY 3.2 @ 38.07: rows 2.5, 0.7; u =
        0.7, shows 1.8; a later BUY 2.0 @ 38.07 prints 1.8 then 0.2."""
        s = _run(book, [_o(BUY, 38.07, 3.2, 9, 1, agent=2)])[-1][1]
        s2, _, rows = _submit(book, s, _o(SELL, 38.07, 9.0, 270001, 2, ICE, peak=2.5, agent=3))
        assert _fills(rows) == [(px(38.07), 25, 9, 270001), (px(38.07), 7, 9, 270001)]
        assert _shown(s2, SELL, 270001) == [(px(38.07), 18)]
        _, _, rows2 = _submit(book, s2, _o(BUY, 38.07, 2.0, 42, 3, agent=4))
        assert _fills(rows2) == [(px(38.07), 18, 270001, 42), (px(38.07), 2, 270001, 42)]


# ===========================================================================
# (8) Split fallback on a short ring; (10)(iii) non-tape leaves vs ring room
# ===========================================================================
def _prefilled(state, n0: int):
    """Replace the trade ring with ``n0`` foreign filler rows (no -1 cell).

    The book never reads the ring (card Step 2 (e) harness), so this only sets
    the ring occupancy the call starts from."""
    ring = np.full((N_RING, 8), -1, dtype=np.int32)
    for i in range(n0):
        ring[i] = [px(10.00) + i, 10 + i, 900000 + i, 910000 + i, 1, i, 98, 99]
    lob = state.lob._replace(trades=jnp.asarray(ring, dtype=state.lob.trades.dtype))
    return state._replace(lob=lob)


@pytest.fixture(scope="module")
def std6_pre(book):
    return _run(book, STD6_SETUP)[-1][1]


def _non_tape(state, res) -> Dict[str, str]:
    return _leaves(state, res, tape=False)


class TestSplitFallback:

    def test_exact_fit_applies_split(self, book, std6_pre):
        """n0 = 96: 96 + 4 pieces = 100 = N fits; the pieces fill slots 96-99."""
        s = _prefilled(std6_pre, 96)
        s2, _ = book.submit_order(s, STD6_ICE)
        ring = np.asarray(s2.lob.trades)
        np.testing.assert_array_equal(ring[:96], np.asarray(s.lob.trades)[:96])
        assert _fills(ring[96:]) == [(px(50.00), 40, 21, 55), (px(50.00), 20, 21, 55),
                                     (px(50.50), 20, 404, 55), (px(50.50), 40, 404, 55)]

    def test_too_little_room_keeps_unsplit_rows(self, book, std6_pre):
        """n0 = 97: 97 + 4 > 100, so nothing splits: rows 6.0 @ 50.00 and 6.0 @
        50.50 in slots 97, 98; slot 99 stays empty."""
        s = _prefilled(std6_pre, 97)
        s2, res = book.submit_order(s, STD6_ICE)
        ring = np.asarray(s2.lob.trades)
        np.testing.assert_array_equal(ring[:97], np.asarray(s.lob.trades)[:97])
        assert _fills(ring[97:99]) == [(px(50.00), 60, 21, 55), (px(50.50), 60, 404, 55)]
        assert np.all(ring[99] == -1)
        assert float(res.trade_qty) == pytest.approx(12.0, abs=1e-4)

    @pytest.mark.parametrize("n0", [99, 100])
    def test_saturated_ring_is_byte_no_op_vs_main(self, book, std6_pre, n0):
        """No room even for the unsplit rows (n0 = 99) or a full ring (n0 =
        100). Contract item (8) "byte no-op" means E-C does not split here; it
        does NOT mean the display equals main's (architect ruling, Step 4,
        Reading A).

        OQ-15 / the certificate's saturated-ring gap: on a saturated ring
        jaxob overwrites the last slot, so ``extract_fills`` sees only that
        overwritten last row and the measured sweep M under-counts the true
        12.0 MW. E-B applies the remainder rule to the MEASURED M.

        (1) Leaves that stay equal to frozen main: SubmitResult, asks, trades
            (the E-C byte no-op), n_active, ts_counter, truncated, book_full.
        (2) The measured sweep is 6.0 MW (``res.trade_qty``), the explicit
            dependency of the display on the saturated ring.
        (3) bids / iceberg_meta are not compared with main; the rule on the
            measured M = 6.0 MW, P = 4: u = 2.0, R = 14.0, v_1 = 2.0, so OID 55
            is the only live bid, showing 2.0 @ 51.00; its meta row holds
            remaining hidden 12.0, base 51.00, peak 4.0, slice_count 0."""
        s = _prefilled(std6_pre, n0)
        s2, res = book.submit_order(s, STD6_ICE)
        got = _leaves(s2, res, tape=True)
        for leaf in ("result", "asks", "trades", "n_active", "ts_counter",
                     "truncated", "book_full"):
            assert got[leaf] == _MAIN_SATURATED[n0][leaf], leaf
        assert float(res.trade_qty) == pytest.approx(6.0, abs=1e-4)
        assert _live(s2, BUY) == [(px(51.00), qt(2.0), 55)]
        meta = np.asarray(s2.iceberg_meta)[: int(s2.n_active)]
        row = meta[meta[:, COL_ORDER_ID] == 55]
        assert row.shape[0] == 1
        assert int(row[0, COL_REMAINING_HIDDEN]) == qt(12.0)
        assert int(row[0, COL_BASE_PRICE_INT]) == px(51.00)
        assert int(row[0, COL_PEAK_QTY]) == qt(4.0)
        assert int(row[0, COL_SLICE_COUNT]) == 0

    @pytest.mark.parametrize("case", ["std6", "part", "mid"])
    def test_non_tape_leaves_do_not_depend_on_split(self, book, case):
        """(10)(iii): from the same book, a call on an empty ring and on a ring
        with room for the unsplit rows only (n0 = 97) leaves every non-tape
        leaf and SubmitResult identical."""
        setup, order = {"std6": (STD6_SETUP, STD6_ICE), "part": (PART_SETUP, PART_ICE),
                        "mid": (MID_SETUP, MID_ICE)}[case]
        pre = _run(book, setup)[-1][1]
        a, ra = book.submit_order(pre, order)
        b, rb = book.submit_order(_prefilled(pre, 97), order)
        assert _non_tape(a, ra) == _non_tape(b, rb)


# ===========================================================================
# (9) Guards frozen on main
# ===========================================================================
# (9a) NON / IOC / FOK with no ICE, or only non-crossing ICE / ICEwPI (M = 0)
# and one crossing standard ICE with u = 0 whose rows end on multiples of P.
S9A_BUY = [
    _o(SELL, 47.31, 1.7, 61, 3, agent=1),
    _o(SELL, 47.12, 3.0, 7003, 4, ICE, peak=0.4, agent=2),
    _o(SELL, 47.20, 2.4, 33012, 5, ICE, peak=0.6, ppd=0.07, agent=3),
    _o(BUY, 46.88, 0.9, 512, 6, agent=4),
    _o(BUY, 47.25, 2.3, 880014, 9, agent=5),
    _o(BUY, 47.40, 1.1, 4404, 11, IOC, agent=6),
    _o(BUY, 47.35, 3.3, 90210, 12, FOK, agent=7),
    _o(BUY, 47.33, 50.0, 42, 13, FOK, agent=8),
    _o(SELL, 46.50, 0.5, 77701, 14, IOC, agent=9),
    _o(BUY, 46.95, 2.0, 640, 15, ICE, peak=0.5, ppd=-0.03, agent=2),
    _o(SELL, 46.90, 1.2, 7, 16, agent=3),
    _o(SELL, 46.80, 0.8, 3301, 17, FOK, agent=4),
]
S9A_SELL = [
    _o(BUY, 52.64, 2.0, 95, 11, agent=1),
    _o(BUY, 52.61, 2.0, 9001, 12, agent=2),
    _o(BUY, 52.41, 1.5, 4, 13, agent=3),
    _o(SELL, 52.58, 6.1, 31, 14, ICE, peak=2.0, agent=4),
    _o(BUY, 52.50, 2.4, 2207, 15, ICE, peak=0.6, ppd=-0.05, agent=5),
    _o(SELL, 53.02, 0.9, 777, 16, agent=6),
    _o(BUY, 52.60, 1.0, 51, 17, IOC, agent=7),
    _o(BUY, 52.59, 1.5, 3417, 18, FOK, agent=8),
    _o(BUY, 52.59, 1.1, 3418, 19, FOK, agent=9),
    _o(SELL, 52.40, 1.0, 610, 20, agent=1),
    _o(SELL, 52.30, 3.0, 88, 21, FOK, agent=2),
    _o(SELL, 52.30, 3.0, 89, 22, IOC, agent=3),
]
S9A = {"buy": S9A_BUY, "sell": S9A_SELL}

# (9b) and (13): standard-ICE sessions whose every incoming ICE has u = 0.
S13_BUY = [
    _o(SELL, 50.11, 1.5, 404, 2, agent=3),
    _o(SELL, 50.13, 2.5, 21, 3, agent=4),
    _o(SELL, 50.14, 3.0, 7071, 4, ICE, peak=1.0, agent=5),
    _o(BUY, 50.13, 7.0, 55, 5, ICE, peak=2.0, agent=6),
    _o(SELL, 50.12, 2.6, 3009, 6, agent=7),
    _o(BUY, 50.20, 6.4, 81237, 7, ICE, peak=1.5, agent=8),
    _o(SELL, 50.19, 2.0, 6, 8, IOC, agent=9),
    _o(SELL, 50.10, 0.4, 77, 9, agent=1),
]
S13_SELL = [
    _o(BUY, 61.47, 0.7, 270001, 1, agent=1),
    _o(BUY, 61.40, 3.3, 12, 2, agent=2),
    _o(BUY, 61.05, 1.0, 3456, 3, agent=3),
    _o(SELL, 61.30, 9.6, 6, 4, ICE, peak=2.0, agent=4),
    _o(BUY, 61.35, 5.0, 50021, 5, ICE, peak=2.5, agent=5),
    _o(SELL, 61.00, 1.5, 909, 6, agent=6),
    _o(BUY, 61.32, 1.0, 31, 7, agent=7),
]
S13 = {"buy": S13_BUY, "sell": S13_SELL}

# (9c): standard-ICE entries with u > 0, all on one side; nothing trades their
# remainders later. Indices of the u > 0 entry calls.
S9C_BUY = [
    _o(SELL, 39.90, 1.3, 501, 1, agent=1),
    _o(SELL, 39.97, 2.2, 28, 2, agent=2),
    _o(SELL, 40.03, 0.4, 10442, 3, agent=3),
    _o(SELL, 40.05, 5.0, 888, 4, ICE, peak=0.7, agent=4),
    _o(BUY, 39.95, 5.0, 6020, 5, ICE, peak=2.0, agent=5),     # M 1.3, u 1.3
    _o(BUY, 40.03, 6.0, 314159, 6, ICE, peak=1.5, agent=6),   # M 2.6, u 1.1
    _o(BUY, 40.05, 3.3, 47, 7, ICE, peak=1.0, agent=7),       # M 3.3, u 0.3, R 0
    _o(BUY, 40.06, 9.9, 2, 8, ICE, peak=3.0, agent=8),        # M 1.7, u 1.7
]
S9C_SELL = [
    _o(BUY, 61.47, 3.2, 270001, 2, agent=1),
    _o(BUY, 60.30, 0.5, 812, 9, agent=2),
    _o(BUY, 60.10, 4.0, 3003, 10, ICE, peak=0.9, agent=3),
    _o(SELL, 61.13, 7.0, 61, 18, ICE, peak=2.0, agent=4),     # M 3.2, u 1.2
    _o(SELL, 60.05, 8.0, 33417, 25, ICE, peak=2.5, agent=5),  # M 4.5, u 2.0
]
S9C = {"buy": (S9C_BUY, BUY, (4, 5, 6, 7)), "sell": (S9C_SELL, SELL, (3, 4))}


class TestGuardsFrozenOnMain:

    @pytest.mark.parametrize("name", sorted(S9A))
    def test_9a_full_byte_identity(self, book, name):
        got = [_combined(_leaves(s2, r, tape=True)) for _, s2, r in _run(book, S9A[name])]
        assert got == _MAIN_9A[name]

    @pytest.mark.parametrize("name", sorted(S13))
    def test_9b_non_tape_identity(self, book, name):
        got = [_combined(_leaves(s2, r, tape=False)) for _, s2, r in _run(book, S13[name])]
        assert got == _MAIN_9B[name]

    @pytest.mark.parametrize("name", sorted(S9C))
    def test_9c_per_call_identity(self, book, name):
        orders, side, idx = S9C[name]
        calls = _run(book, orders)
        got = [_c9c_digest(calls[i][1], calls[i][2], side) for i in idx]
        assert got == _MAIN_9C[name]

    @pytest.mark.parametrize("name", sorted(S9C))
    def test_9c_entries_really_have_u_gt_0(self, book, name):
        """Fixture check: each pinned entry sweeps M with M mod P != 0."""
        orders, side, idx = S9C[name]
        calls = _run(book, orders)
        for i in idx:
            m = sum(q for _, q, _, _ in _fills(_call_rows(calls[i][0], calls[i][1])))
            assert m > 0 and m % qt(float(orders[i].peak_qty)) != 0, i


# ===========================================================================
# (10) E-C properties
# ===========================================================================
@pytest.fixture(scope="module")
def split_fn():
    fn = getattr(_cb, "_split_incoming_rows", None)
    if fn is None:
        def _missing(*_args, **_kwargs):
            pytest.fail("contract_book._split_incoming_rows(trades_before, trades_after, "
                        "peak_int, order_id) is not defined (card E-C)")
        return _missing
    return fn


_OWN, _OTHER = 4242, 515


def _row(p, q, poid, aoid, sec=7, nsec=3, ptid=2, atid=5):
    return [p, q, poid, aoid, sec, nsec, ptid, atid]


def _ring(rows) -> np.ndarray:
    r = np.full((N_RING, 8), -1, dtype=np.int32)
    for i, row in enumerate(rows):
        r[i] = row
    return r


def _filler(n):
    return [_row(px(20.00) + i, 10, 800000 + i, 810000 + i, 1, i, 8, 9) for i in range(n)]


def _apply(fn, before_rows, new_rows, peak_int, oid=_OWN) -> np.ndarray:
    before = _ring(before_rows)
    after = _ring(before_rows + new_rows)
    out = fn(jnp.asarray(before), jnp.asarray(after), jnp.int32(peak_int), jnp.int32(oid))
    out = np.asarray(out)
    assert out.shape == (N_RING, 8) and out.dtype == np.int32
    return out


def _used(ring) -> np.ndarray:
    return ring[ring[:, T_SEC] != -1]


class TestSplitHelperUnit:
    """(10)(i): the pure helper on arrays. Returns the new ring (same shape,
    int32). Piece Q = sign(q) * length; every other column copied."""

    def test_single_own_row_splits(self, split_fn):
        pre = _filler(3)
        out = _apply(split_fn, pre, [_row(px(50.0), -60, 21, _OWN)], 40)
        np.testing.assert_array_equal(out[:3], np.asarray(pre, np.int32))
        assert _used(out)[3:].tolist() == [_row(px(50.0), -40, 21, _OWN),
                                            _row(px(50.0), -20, 21, _OWN)]
        assert np.all(out[5:] == -1)

    def test_cumulative_boundaries_across_rows(self, split_fn):
        """|q| 15, 25, 10 with P = 20: c = 15, 40, 50 -> 15 | 5, 20 | 10."""
        new = [_row(101, 15, 1, _OWN, ptid=11), _row(102, 25, 2, _OWN, ptid=12),
               _row(103, 10, 3, _OWN, ptid=13)]
        out = _used(_apply(split_fn, [], new, 20))
        assert out[:, T_Q].tolist() == [15, 5, 20, 10]
        assert out[:, T_P].tolist() == [101, 102, 102, 103]
        assert out[:, T_PTID].tolist() == [11, 12, 12, 13]

    def test_long_row_and_property_checks(self, split_fn):
        """|q| 95, 7, 38 with P = 20: every piece <= P, every multiple of P
        below M = 140 is a cumulative boundary, per-row sums kept."""
        new = [_row(201, 95, 9, _OWN), _row(202, 7, 9, _OWN), _row(203, 38, 9, _OWN)]
        out = _used(_apply(split_fn, [], new, 20))
        q = out[:, T_Q].tolist()
        assert all(0 < x <= 20 for x in q)
        cum = np.cumsum(q).tolist()
        assert all(j * 20 in cum for j in range(1, 7))
        for p, total in ((201, 95), (202, 7), (203, 38)):
            assert int(out[out[:, T_P] == p][:, T_Q].sum()) == total
        assert q == [20, 20, 20, 20, 15, 5, 2, 18, 20]

    def test_foreign_rows_untouched_and_in_sequence(self, split_fn):
        """Own 30, foreign 70, own 30 with P = 20: own cumulative counts own
        rows only -> 20, 10 | foreign 70 | 10, 20."""
        new = [_row(301, 30, 1, _OWN), _row(302, 70, 2, _OTHER, atid=6),
               _row(303, 30, 3, _OWN)]
        out = _used(_apply(split_fn, [], new, 20))
        assert out.tolist() == [_row(301, 20, 1, _OWN), _row(301, 10, 1, _OWN),
                                _row(302, 70, 2, _OTHER, atid=6),
                                _row(303, 10, 3, _OWN), _row(303, 20, 3, _OWN)]

    def test_all_or_nothing_boundary(self, split_fn):
        """One own row of 60 with P = 20 makes 3 pieces: at n0 = 97 they fit
        exactly; at n0 = 98 nothing splits (the output is trades_after)."""
        new = [_row(401, 60, 4, _OWN)]
        fit = _apply(split_fn, _filler(97), new, 20)
        assert fit[97:, T_Q].tolist() == [20, 20, 20]
        pre = _filler(98)
        nofit = _apply(split_fn, pre, new, 20)
        np.testing.assert_array_equal(nofit, _ring(pre + new))

    def test_fit_check_counts_foreign_rows(self, split_fn):
        """Own 40 (2 pieces) + foreign 10: at n0 = 97, 97 + 2 + 1 = 100 fits;
        at n0 = 98 it does not."""
        new = [_row(501, 40, 5, _OWN), _row(502, 10, 6, _OTHER)]
        fit = _used(_apply(split_fn, _filler(97), new, 20))
        assert fit[97:, T_Q].tolist() == [20, 20, 10]
        pre = _filler(98)
        np.testing.assert_array_equal(_apply(split_fn, pre, new, 20), _ring(pre + new))

    def test_full_ring_before_call_is_no_op(self, split_fn):
        before = _ring(_filler(N_RING))
        after = before.copy()
        after[-1] = _row(601, 60, 7, _OWN)  # jaxob's full-ring overwrite slot
        out = np.asarray(split_fn(jnp.asarray(before), jnp.asarray(after),
                                  jnp.int32(20), jnp.int32(_OWN)))
        np.testing.assert_array_equal(out, after)

    def test_no_new_rows_is_no_op(self, split_fn):
        pre = _filler(5)
        np.testing.assert_array_equal(_apply(split_fn, pre, [], 20), _ring(pre))

    def test_rows_on_boundaries_unchanged(self, split_fn):
        new = [_row(701, -20, 1, _OWN), _row(702, -40, 2, _OWN)]
        assert _used(_apply(split_fn, [], new, 20)).tolist() == [
            _row(701, -20, 1, _OWN), _row(702, -20, 2, _OWN), _row(702, -20, 2, _OWN)]

    def test_every_own_row_keeps_a_piece(self, split_fn):
        """A zero-quantity own row on a boundary still yields one piece."""
        new = [_row(801, 20, 1, _OWN), _row(802, 0, 2, _OWN), _row(803, 5, 3, _OWN)]
        out = _used(_apply(split_fn, [], new, 20))
        assert out[:, T_P].tolist() == [801, 802, 803]
        assert out[:, T_Q].tolist() == [20, 0, 5]

    def test_peak_zero_treated_as_one_tick(self, split_fn):
        out = _used(_apply(split_fn, [], [_row(901, 3, 1, _OWN)], 0))
        assert out[:, T_Q].tolist() == [1, 1, 1]


_ICEWPI_CASES = {
    "gest": (GEST_SETUP, GEST_ICE), "part": (PART_SETUP, PART_ICE),
    "tiny": (TINY_SETUP, TINY_ICE), "big": (BIG_SETUP, BIG_ICE), "mid": (MID_SETUP, MID_ICE),
    "resting_icewpi": ([_o(SELL, 50.00, 6.0, 707, 1, ICE, peak=1.5, ppd=0.10, agent=2)],
                       _o(BUY, 50.60, 5.0, 29, 2, ICE, peak=2.0, ppd=-0.20, agent=3)),
}
_STD_CASES = {
    "std6": (STD6_SETUP, STD6_ICE), "std7": (STD7_SETUP, STD7_ICE),
    "s13_buy_55": (S13_BUY[:3], S13_BUY[3]),
    "s13_sell_6": (S13_SELL[:3], S13_SELL[3]),
    "vs_resting_icewpi": ([_o(SELL, 50.00, 4.5, 33012, 1, ICE, peak=1.5, ppd=0.05, agent=2)],
                          _o(BUY, 50.40, 4.5, 6, 2, ICE, peak=2.0, agent=3)),
}


def _own_qtys(book, setup, order):
    pre = _run(book, setup)[-1][1] if setup else book.init()
    _, _, rows = _submit(book, pre, order)
    oid = int(order.order_id)
    return [q for _, q, _, a in _fills(rows) if a == oid], qt(float(order.peak_qty))


class TestSplitOutputProperties:

    @pytest.mark.parametrize("case", sorted(_ICEWPI_CASES))
    def test_icewpi_no_own_row_spans_a_multiple_of_p(self, book, case):
        """(10)(ii): under E-A with d != 0 the passes end at instance
        boundaries, so no own row's cumulative interval spans a multiple of P."""
        qtys, p = _own_qtys(book, *_ICEWPI_CASES[case])
        assert qtys, "fixture must trade"
        bad = [(lo, hi) for lo, hi in _cum_intervals(qtys) if _spans_multiple(lo, hi, p)]
        assert bad == []

    @pytest.mark.parametrize("case", sorted(_STD_CASES))
    def test_standard_pieces_bounded_and_boundaries_present(self, book, case):
        """(10)(i) on engine output: every own piece <= P; every multiple jP < M
        is a cumulative boundary."""
        qtys, p = _own_qtys(book, *_STD_CASES[case])
        m = sum(qtys)
        assert all(0 < q <= p for q in qtys)
        cum = set(np.cumsum(qtys).tolist())
        assert all(j * p in cum for j in range(1, -(-m // p)) if j * p < m)


# ===========================================================================
# (11) Eager = jit = each vmap lane
# ===========================================================================
def _lane_cases(book):
    pre_part = _run(book, PART_SETUP)[-1][1]
    pre_mid = _run(book, MID_SETUP)[-1][1]
    pre_std7 = _run(book, STD7_SETUP)[-1][1]
    pre_std6 = _run(book, STD6_SETUP)[-1][1]
    return [
        (pre_part, PART_ICE),                                   # BUY ICEwPI k_e 1, u > 0
        (pre_mid, MID_ICE),                                     # SELL ICEwPI k_e 1, u > 0
        (pre_std7, STD7_ICE),                                   # standard ICE u > 0
        (pre_std6, STD6_ICE),                                   # standard ICE u = 0, split
        (pre_std7, _o(BUY, 50.00, 20.0, 9312, 2, NON, agent=3)),  # NON lane
    ]


def _assert_same(a_state, a_res, b_state, b_res, msg):
    """Every state leaf bit-identical (``LobState.key`` excluded); the
    SubmitResult flags exact and its float32 sums to 1 ulp-scale tolerance
    (the eager / jit settlement product differs by FMA contraction on main
    already, see OMIE-PRICE-DECODE-FMA-1; not a property of this story)."""
    strip = lambda st: st._replace(lob=st.lob._replace(key=0))  # noqa: E731
    for x, y in zip(jax.tree_util.tree_leaves(strip(a_state)),
                    jax.tree_util.tree_leaves(strip(b_state))):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=msg)
    assert bool(a_res.accepted) == bool(b_res.accepted), msg
    assert bool(a_res.traded) == bool(b_res.traded), msg
    for name in ("trade_qty", "trade_price", "settlement_value"):
        assert float(getattr(a_res, name)) == pytest.approx(
            float(getattr(b_res, name)), rel=1e-6, abs=1e-6), f"{msg} {name}"


class TestEagerJitVmap:

    @pytest.mark.slow
    def test_eager_equals_jit(self, book):
        """slow: every lane case re-run under jax.disable_jit (eager, ~seconds)."""
        for i, (pre, od) in enumerate(_lane_cases(book)):
            js, jr = book.submit_order(pre, od)
            with jax.disable_jit():
                es, er = book.submit_order(pre, od)
            _assert_same(js, jr, es, er, f"lane {i}")

    @pytest.mark.slow
    def test_vmap_lanes_equal_scalar_calls(self, book):
        """slow: fresh vmap compile of _submit_order on mixed lanes."""
        cases = _lane_cases(book)
        scalar = [book.submit_order(pre, od) for pre, od in cases]
        stack = lambda *xs: jnp.stack([jnp.asarray(x) for x in xs])  # noqa: E731
        bs = jax.tree_util.tree_map(stack, *[c[0] for c in cases])
        bo = jax.tree_util.tree_map(stack, *[c[1] for c in cases])
        with enable_x64():
            bst, bres = jax.vmap(lambda st, o: _submit_order(st, o, PC, MC))(bs, bo)
        for lane, (st1, r1) in enumerate(scalar):
            ls = jax.tree_util.tree_map(lambda x: x[lane], bst)
            lr = jax.tree_util.tree_map(lambda x: x[lane], bres)
            for name in ("bids", "asks", "trades"):
                np.testing.assert_array_equal(np.asarray(getattr(ls.lob, name)),
                                              np.asarray(getattr(st1.lob, name)),
                                              err_msg=f"lane {lane} {name}")
            for name in ("iceberg_meta", "n_active", "ts_counter", "truncated", "book_full"):
                np.testing.assert_array_equal(np.asarray(getattr(ls, name)),
                                              np.asarray(getattr(st1, name)),
                                              err_msg=f"lane {lane} {name}")
            assert float(lr.trade_qty) == pytest.approx(float(r1.trade_qty))
            assert float(lr.settlement_value) == pytest.approx(float(r1.settlement_value))


# ===========================================================================
# (12) An overflowed table
# ===========================================================================
@pytest.fixture(scope="module")
def tiny_table_book():
    return _new_book(OMIEMarketConfig(max_icebergs=1))


class TestOverflowedTable:

    @pytest.mark.slow
    def test_icewpi_remainder_unregistered_follows_ladder_rule(self, tiny_table_book):
        """slow: fresh _submit_order compile for max_icebergs = 1.

        The one slot is taken by a resting BUY ICE. SELL ICE 50.00, 10, P = 2,
        d = +0.50 vs bid 3.0 @ 51.00: 2.0 and 1.0 @ 51.00 (M = 3, k_e = 1,
        u = 1). R = 7 cannot register: shows v_1 = 1.0 @ 50.50 unregistered;
        truncated is set."""
        bk = tiny_table_book
        s = _run(bk, [_o(BUY, 29.50, 2.0, 3401, 1, ICE, peak=0.5, agent=1),
                      _o(BUY, 51.00, 3.0, 5120, 2, agent=2)])[-1][1]
        assert not bool(s.truncated) and int(s.n_active) == 1
        s2, res, rows = _submit(bk, s, _o(SELL, 50.00, 10.0, 77, 3, ICE, peak=2.0,
                                          ppd=0.50, agent=3))
        assert _fills(rows) == [(px(51.00), 20, 5120, 77), (px(51.00), 10, 5120, 77)]
        assert _shown(s2, SELL, 77) == [(px(50.50), 10)]
        assert bool(s2.truncated)
        assert int(s2.n_active) == 1 and not _has_meta(s2, 77)
        assert float(res.trade_qty) == pytest.approx(3.0, abs=1e-4)

    @pytest.mark.slow
    def test_standard_remainder_unregistered_shows_v1(self, tiny_table_book):
        """slow: shares the max_icebergs = 1 compile.

        BUY ICE 31.18, 4.0, P = 1.5 vs SELL 2.0 @ 31.10: rows 1.5, 0.5 (M = 2,
        k_e = 1, u = 0.5); shows v_1 = 1.0 @ 31.18 unregistered; truncated."""
        bk = tiny_table_book
        s = _run(bk, [_o(BUY, 29.50, 2.0, 3401, 1, ICE, peak=0.5, agent=1),
                      _o(SELL, 31.10, 2.0, 3402, 2, agent=2)])[-1][1]
        s2, _, rows = _submit(bk, s, _o(BUY, 31.18, 4.0, 3403, 3, ICE, peak=1.5, agent=3))
        assert _fills(rows) == [(px(31.10), 15, 3402, 3403), (px(31.10), 5, 3402, 3403)]
        assert _shown(s2, BUY, 3403) == [(px(31.18), 10)]
        assert bool(s2.truncated) and not _has_meta(s2, 3403)


# ===========================================================================
# (13) Tape: merge-back to the E-B tape; tripwire hashes
# ===========================================================================
def _merge_runs(rows: Sequence[Sequence[int]]) -> List[List[int]]:
    """Merge contiguous rows equal on every column except Q (summing Q)."""
    out: List[List[int]] = []
    for r in rows:
        r = [int(x) for x in r]
        if out and all(out[-1][c] == r[c] for c in range(8) if c != T_Q):
            out[-1][T_Q] += r[T_Q]
        else:
            out.append(r)
    return out


def _walk_split(rows: Sequence[Sequence[int]], peak_int: int, oid: int,
                n0: int) -> List[List[int]]:
    """This file's own split walk of one call's rows (card E-C)."""
    p = max(int(peak_int), 1)
    c, out = 0, []
    for r in rows:
        r = [int(x) for x in r]
        if r[T_AOID] != oid:
            out.append(r)
            continue
        q = abs(r[T_Q])
        sgn = -1 if r[T_Q] < 0 else 1
        lo, hi = c, c + q
        pieces = []
        m = lo // p
        while True:
            a, b = max(lo, m * p), min(hi, (m + 1) * p)
            if a >= b:
                break
            pieces.append(r[:T_Q] + [sgn * (b - a)] + r[T_Q + 1:])
            m += 1
        out.extend(pieces or [r])
        c = hi
    return out if n0 + len(out) <= N_RING else [[int(x) for x in r] for r in rows]


def _session_call_rows(book, orders):
    return [_call_rows(s1, s2) for s1, s2, _ in _run(book, orders)]


class TestTapeMergeBack:

    @pytest.mark.parametrize("name", sorted(S13))
    def test_fixture_main_rows_have_no_mergeable_neighbours(self, name):
        """Fixture check: merge-back is lossless on the frozen main rows."""
        for call in _MAIN_13_ROWS[name]:
            assert _merge_runs(call) == [list(r) for r in call]

    @pytest.mark.parametrize("name", sorted(S13))
    def test_merging_split_rows_back_reproduces_e_b_tape(self, book, name):
        """Per call, merging the new rows over contiguous runs (every column
        except Q) gives the E-B rows, which on these d = 0, u = 0 sessions are
        the frozen main rows, byte for byte."""
        got = [_merge_runs(rows.tolist()) for rows in _session_call_rows(book, S13[name])]
        assert got == [[list(r) for r in call] for call in _MAIN_13_ROWS[name]]

    @pytest.mark.parametrize("name", sorted(S13))
    def test_new_tape_equals_split_walk_of_main_rows(self, book, name):
        orders = S13[name]
        got = _session_call_rows(book, orders)
        n0 = 0
        for i, (od, main_rows) in enumerate(zip(orders, _MAIN_13_ROWS[name])):
            if int(od.exec_restriction) == ICE:
                want = _walk_split(main_rows, qt(float(od.peak_qty)), int(od.order_id), n0)
            else:
                want = [list(r) for r in main_rows]
            assert got[i].tolist() == want, f"call {i}"
            n0 += len(want)

    @pytest.mark.parametrize("name", sorted(S13))
    def test_tape_hash_tripwire(self, book, name):
        """REGRESSION TRIPWIRE (not the independent audit pin): the final ring
        hash, derived from this file's split walk of the frozen main rows."""
        final = _run(book, S13[name])[-1][1]
        assert _h(final.lob.trades) == _TRIPWIRE_13[name]


# ===========================================================================
# Frozen values (main engine @ 4000bc7-equivalent; see module docstring)
# ===========================================================================
# All pins below were frozen on main engine @ 4000bc7-equivalent (branch tip
# 2b3818f, omiecid_simulator/ unchanged since 4000bc7), CPU, 2026-10-20.

# (8) saturated ring (n0 = 99 and n0 = 100), call STD6_ICE on STD6_SETUP.
_FROZEN_SAT = {
    "asks": "1404aa613105d249", "bids": "81bda0035613a938",
    "book_full": "9411fbc9e489c3ee", "iceberg_meta": "cca2a7d333a28f70",
    "n_active": "b61a396bc28eeeb2", "result": "3b695a9ff25d4f61",
    "trades": "dfbd17f24af8692f", "truncated": "9411fbc9e489c3ee",
    "ts_counter": "304e0e5e38e86a65",
}
_MAIN_SATURATED: Dict[int, Dict[str, str]] = {99: dict(_FROZEN_SAT), 100: dict(_FROZEN_SAT)}

# (9a) per call: combined digest of every leaf (tape included) + SubmitResult.
_MAIN_9A: Dict[str, List[str]] = {
    "buy": ["39e7a2f2df46cfb5", "9b3b4ffad935f071", "2d67497f0e101245", "bce40dd80018bd8a",
            "e1f2f5cc4e67713e", "01911957f1c7a84d", "3d7ece8e8479479f", "3d7ece8e8479479f",
            "ad7d74eb0ecb4712", "47d6757cdf62e704", "3f0bce8675b6575f", "c1a272f22dc9a86b"],
    "sell": ["f7b619eb3aeab480", "59ba9b9791b4da7d", "5126261d182df37a", "99fbf4b2ccf93344",
             "3aa1df50a43f01d6", "cdeac12b7e4af641", "1180675cd8a51020", "a4b98cb4a4c87567",
             "8b10b913c8a8809a", "c5f0032fdf9126ec", "2e84bd34a624f672", "425a67184d87889e"],
}

# (9b) per call: combined digest of every NON-tape leaf + SubmitResult.
_MAIN_9B: Dict[str, List[str]] = {
    "buy": ["15e664887a75f333", "d00620758a0d9f74", "6c5740f0cdb35665", "3055127f3e3ad7c1",
            "8a4bd6c5ed90c24c", "4617693a37bd2102", "8a803aaa2f35f074", "5b414918e9da866c"],
    "sell": ["48077ad69f70c24e", "16c778369b12103a", "f868f7d1eba79215", "37cd91a4710f44e4",
             "0591d7f304e45737", "b3428224fe73f300", "43f77b13898439d8"],
}

# (9c) per u > 0 entry call: SubmitResult, ts_counter, n_active, book_full,
# truncated, opposite side.
_MAIN_9C: Dict[str, List[str]] = {
    "buy": ["0218426f502dd648", "8d6cceadba7e5862", "6b857b4e3d68cb3d", "68dd2d1432270b06"],
    "sell": ["02e71dcae14632dd", "11f13893333f8a13"],
}

# (13) main per-call trade rows (P, Q, PASS_OID, AGRS_OID, SEC, NSEC, PASS_TID,
# AGRS_TID). On these d = 0, u = 0 sessions E-B leaves the tape as on main.
_MAIN_13_ROWS: Dict[str, List[List[Tuple[int, ...]]]] = {
    "buy": [
        [], [], [],
        [(1004911, -15, 404, 55, 5, 0, 3, 6), (1004913, -25, 21, 55, 5, 0, 4, 6)],
        [(1004913, 20, 55, 3009, 6, 0, 6, 7), (1004913, 6, 55, 3009, 8, 0, 6, 7)],
        [(1004914, -10, 7071, 81237, 7, 0, 5, 8), (1004914, -10, 7071, 81237, 11, 0, 5, 8),
         (1004914, -10, 7071, 81237, 13, 0, 5, 8)],
        [(1004920, 15, 81237, 6, 8, 0, 8, 9), (1004920, 5, 81237, 6, 16, 0, 8, 9)],
        [(1004920, 4, 81237, 77, 9, 0, 8, 1)],
    ],
    "sell": [
        [], [], [],
        [(1006047, 7, 270001, 6, 4, 0, 1, 4), (1006040, 33, 12, 6, 4, 0, 2, 4)],
        [(1006030, -20, 6, 50021, 5, 0, 4, 5), (1006030, -20, 6, 50021, 7, 0, 4, 5),
         (1006030, -10, 6, 50021, 9, 0, 4, 5)],
        [(1006005, 10, 3456, 909, 6, 0, 3, 6)],
        [(1006000, -5, 909, 31, 7, 0, 6, 7), (1006030, -5, 6, 31, 7, 0, 4, 7)],
    ],
}

# REGRESSION TRIPWIRE (not frozen on main, and not the independent audit pin):
# sha256[:16] of the (100, 8) int32 ring built by applying _walk_split to each
# ICE call of _MAIN_13_ROWS (other calls copied), padded with -1.
_TRIPWIRE_13: Dict[str, str] = {"buy": "e8a759ecd4170466", "sell": "a9bfa7fe393b1b2f"}
