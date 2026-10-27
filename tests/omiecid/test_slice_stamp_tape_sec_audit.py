"""AUDIT tests for OMIE-SLICE-STAMP-TAPE-SEC-1 (audit mode, independent).

Derived from the story card (agile-plan v5.212: Description, DESIGN
DECISION, Correct behaviour E1/E2, Covers) and market-spec §5.2, §6.1,
§4.2, §4.2.1 Rule 2/Rule 4:

  E1  Every stored entry stamp ``[SEC, NSEC]`` is unique per contract,
      across prices AND sides, after every submit. In a tape-only session
      the stamps sort in processing order. Two non-crossing orders in one
      second, one BUY and one SELL, store NSEC 0 and 1.
  E2  A slice activated inside a matching process takes its queue place at
      THAT event: after the order whose process activated it, behind every
      order then resting at its price, ahead of every later arrival. Slices
      exhausted earlier in one process queue earlier. The aggressor's own
      resting remainder (NON) or first slice (ICE) keeps its entry stamp
      and precedes every slice its own process activated. When an agent
      order (stamped from the counter, ahead of the tape clock) already
      rests at the slice's price, the slice stays behind it (card: inner max
      in S_r). Holds at every call site: NON, IOC, FOK, incoming ICE.
  --  No price or volume change: every trade equals a plain price-time walk
      over a processing-order queue (computed here by ``_Ref``, an
      independent reference model, never by reading the engine).
  S5-A2  every live SEC is strictly below ``ts_counter``.

The locked file ``tests/omiecid/test_slice_stamp_tape_sec.py`` was NOT read
(audit isolation). Incidental constants are varied on purpose: OIDs of 1-6
digits, off-round prices, both sides, different seconds, counter running
ahead of the tape by different amounts.

Device policy: CPU-hermetic (tests/omiecid/conftest.py pins the cpu platform).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.encoding import decode_ppd, decode_price, decode_qty
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order

pytestmark = [pytest.mark.jit]

_ROOT = Path(__file__).resolve().parents[2]
_FIX = _ROOT / "notebooks" / "certification" / "fixtures"

_CID = 23
BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, IOC, FOK, ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                      int(ExecRestriction.FOK), int(ExecRestriction.ICE))

# jaxob layouts: book rows (P, Q, OID, TID, SEC, NSEC);
# trade rows (P, Q, PASS_OID, AGRS_OID, SEC, NSEC, PASS_TID, AGRS_TID).
_BP, _BQ, _BOID, _BTID, _BSEC, _BNSEC = 0, 1, 2, 3, 4, 5
_TP, _TQ, _TPASS, _TAGRS = 0, 1, 2, 3

_PC = OMIEProductConfig()
_OFF = int(round(-_PC.price_min / _PC.price_tick))  # price int of 0.00 EUR
_PENDING = 10 ** 9                                  # model: "activated in this process"


def _P(cents: int) -> int:
    """jaxob price int of ``cents`` hundredths of EUR/MWh."""
    return _OFF + cents


def _contract() -> Contract:
    return Contract(contract_id=_CID, delivery_start_minutes=17 * 60 + 15,
                    delivery_end_minutes=17 * 60 + 30,
                    gct_minutes=16 * 60 + 15, status=ContractStatus.OPEN)


@pytest.fixture(scope="module")
def book():
    return ContractBook(_contract(), _PC, OMIEMarketConfig())


def _mk(side, cents, q, oid, ts, er=NON, peak=0, ppd=0, agent=None):
    """Order from integer ticks: price in cents, quantities in 0.1 MW."""
    return make_order(
        contract_id=_CID, side=Side(side), price=round(cents * 0.01, 2),
        quantity=round(q * 0.1, 1), order_id=oid,
        agent_id=oid if agent is None else agent, timestamp=int(ts),
        exec_restriction=ExecRestriction(er),
        peak_qty=round(peak * 0.1, 1), ppd=round(ppd * 0.01, 2),
    )


# ---------------------------------------------------------------------------
# Engine observation helpers (public state only)
# ---------------------------------------------------------------------------

def _live(state):
    """Live rows as (side, P, Q, OID, SEC, NSEC)."""
    out = []
    for s, arr in ((BUY, np.asarray(state.lob.bids)), (SELL, np.asarray(state.lob.asks))):
        for r in arr:
            if int(r[_BQ]) > 0 and int(r[_BOID]) > 0:
                out.append((s, int(r[_BP]), int(r[_BQ]), int(r[_BOID]),
                            int(r[_BSEC]), int(r[_BNSEC])))
    return out


def _stamp(state, side, oid):
    rows = [r for r in _live(state) if r[0] == side and r[3] == oid]
    assert len(rows) == 1, f"expected one live row of oid {oid} side {side}: {rows}"
    return rows[0][4], rows[0][5]


def _new_trades(before, after):
    b, a = np.asarray(before), np.asarray(after)
    rows = a[np.any(b != a, axis=1)]
    return [(int(r[_TP]), abs(int(r[_TQ])), int(r[_TPASS]), int(r[_TAGRS]))
            for r in rows]


def _trade_rows_used(state) -> int:
    return int((np.asarray(state.lob.trades)[:, _TPASS] != -1).sum())


def _assert_invariants(state, where=""):
    rows = _live(state)
    keys = [(r[4], r[5]) for r in rows]
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, (
        f"{where}: stored stamps not unique per contract (E1, §5.2): "
        f"{[r for r in rows if (r[4], r[5]) in dupes]}")
    ctr = int(state.ts_counter)
    bad = [r for r in rows if not r[4] < ctr]
    assert not bad, f"{where}: S5-A2 broken, SEC >= ts_counter {ctr}: {bad}"
    assert all(r[5] >= 0 for r in rows), f"{where}: negative NSEC {rows}"


def _queue(state, side, price_int):
    """OIDs resting at one price, in stored-stamp order (jaxob's queue)."""
    rows = [r for r in _live(state) if r[0] == side and r[1] == price_int]
    return [r[3] for r in sorted(rows, key=lambda r: (r[4], r[5]))]


# ---------------------------------------------------------------------------
# Independent reference model: price-time over a processing-order queue
# ---------------------------------------------------------------------------

# OMIE-ICE-INCOMING-LADDER-1 (licensed oracle change, E-B / E-C): market-spec
# §4.2.1 Rule 2 *Incoming iceberg* items (2) and (4), amended 2026-10-20. An
# incoming iceberg is a chain of instances of size P (PPD = 0 here, so every
# instance has limit b). A trade row also ends where one incoming instance
# ends; a remainder R after a sweep of M continues instance k_e = M // P and
# shows v_1 = min(P - u, R), u = M - k_e P. Both helpers below are the
# reference's incoming-iceberg branch; per-case walks are never edited.

def _split_incoming(fills, peak):
    """Split the price-time walk ``(p, q, passive, aggressor)`` of ONE
    incoming iceberg at the multiples of ``peak`` of its cumulative traded
    quantity (item (2)); prices, OIDs and the order of rows are kept."""
    peak = max(int(peak), 1)
    out, c = [], 0
    for p_, q_, po, ao in fills:
        lo, hi = c, c + q_
        m = lo // peak
        while max(lo, m * peak) < hi:
            out.append((p_, min(hi, (m + 1) * peak) - max(lo, m * peak), po, ao))
            m += 1
        c = hi
    return out


def _incoming_first_slice(total, swept, peak):
    """Visible first slice v_1 = min(P - u, R) of an incoming iceberg's
    remainder (item (4))."""
    peak = max(int(peak), 1)
    rem = total - swept
    return min(peak - (swept - (swept // peak) * peak), rem)


class _Ref:
    """Plain OMIE continuous matching for PPD = 0 orders.

    Queue key of a resting entry = (sec, seq). ``seq`` is the global
    processing order (arrival, or slice activation). For tape-only sessions
    ``sec`` is non-decreasing in ``seq``, so the key is pure processing
    order. A slice activated inside a process ranks behind every entry
    already resting during that process; at process end it is dated
    ``max(e, latest pre-existing sec at its price on its side)`` (card E2,
    which binds only when a counter-stamped agent order rests there).
    """

    def __init__(self):
        self.book = []
        self.g = 0

    def submit(self, side, p, q, oid, e, er=NON, peak=0):
        def crosses(x):
            return x["p"] <= p if side == BUY else x["p"] >= p

        if er == FOK:
            avail = sum(x["vis"] + x["hid"] for x in self.book
                        if x["side"] != side and crosses(x))
            if avail < q:
                return []
        a_seq = self.g
        self.g += 1
        rem, fills = q, []
        while rem > 0:
            cands = [x for x in self.book
                     if x["side"] != side and x["vis"] > 0 and crosses(x)]
            if not cands:
                break
            best = min(cands, key=lambda x: (x["p"] if side == BUY else -x["p"],
                                             x["sec"], x["seq"]))
            t = min(rem, best["vis"])
            fills.append((best["p"], t, best["oid"], oid))
            best["vis"] -= t
            rem -= t
            if best["vis"] == 0:
                if best["hid"] > 0:
                    v = min(best["peak"], best["hid"])
                    best["vis"], best["hid"] = v, best["hid"] - v
                    best["seq"], best["sec"] = self.g, _PENDING
                    self.g += 1
                else:
                    self.book.remove(best)
        pre = [y for y in self.book if y["seq"] < a_seq]
        for y in self.book:
            if y["sec"] == _PENDING:
                y["sec"] = max([e] + [z["sec"] for z in pre
                                      if z["side"] == y["side"] and z["p"] == y["p"]])
        if rem > 0 and er in (NON, ICE):
            # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): v_1 = min(P - u, R)
            # (was min(P, R)).
            v = rem if er == NON else _incoming_first_slice(q, q - rem, peak)
            self.book.append(dict(side=side, p=p, vis=v, hid=rem - v,
                                  peak=peak, oid=oid, seq=a_seq, sec=e))
        if er == ICE:
            # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): rows split at the
            # incoming instance boundaries.
            fills = _split_incoming(fills, peak)
        return fills

    def order(self):
        return [(x["side"], x["oid"]) for x in
                sorted(self.book, key=lambda x: (x["sec"], x["seq"]))]


class _Session:
    """Drives the engine and (optionally) the reference model in lockstep."""

    def __init__(self, book, model=True):
        self.book = book
        self.st = book.init()
        self.ref = _Ref() if model else None

    def submit(self, side, cents, q, oid, ts, er=NON, peak=0, ppd=0,
               agent=None, expect=None):
        before = self.st.lob.trades
        self.st, r = self.book.submit_order(
            self.st, _mk(side, cents, q, oid, ts, er, peak, ppd, agent))
        assert bool(r.accepted), f"fixture bug: oid {oid} rejected"
        got = _new_trades(before, self.st.lob.trades)
        if expect is not None:
            want = [(_P(c), qq, po, oid) for c, qq, po in expect]
            if er == ICE:
                # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): ``expect`` is the
                # per-counter-bid price-time walk; the rows the tape must hold
                # are that walk split by the reference's incoming branch.
                want = _split_incoming(want, peak)
            assert got == want, (
                f"oid {oid} @ t{ts}: tape {got} != price-time walk {want}")
        if self.ref is not None:
            want = self.ref.submit(side, _P(cents), q, oid, int(ts), er, peak)
            assert got == want, (
                f"oid {oid} @ t{ts}: tape {got} != reference walk {want}")
        _assert_invariants(self.st, f"after oid {oid}")
        return got

    def agent_ts(self):
        """A counter-stamped (agent-style) timestamp: the current counter."""
        return int(self.st.ts_counter)


# ---------------------------------------------------------------------------
# E1 — per-contract uniqueness across prices and sides
# ---------------------------------------------------------------------------

class TestE1UniquePerContract:

    @pytest.mark.parametrize("first", [BUY, SELL])
    def test_buy_and_sell_same_second_store_nsec_0_and_1(self, book, first):
        """Card E1: two non-crossing orders in one second, one BUY and one
        SELL, store NSEC 0 and 1 (in processing order)."""
        s = _Session(book)
        second = 17 if first == BUY else 4093
        o1, o2 = (61, 250007) if first == BUY else (8, 3141)
        other = SELL if first == BUY else BUY
        p1, p2 = (5317, 5402) if first == BUY else (8813, 8790)
        s.submit(first, p1, 14, o1, second)
        s.submit(other, p2, 9, o2, second)
        assert _stamp(s.st, first, o1) == (second, 0)
        assert _stamp(s.st, other, o2) == (second, 1)

    def test_same_side_different_prices_same_second_ranked_in_order(self, book):
        """Three BUYs at three prices in one second, then a SELL far away:
        NSEC strictly increasing in processing order across prices/sides."""
        s = _Session(book)
        seq = [(BUY, 4471, 5, 702), (BUY, 4467, 3, 13), (BUY, 4473, 8, 99001),
               (SELL, 4519, 2, 4)]
        for side, c, q, oid in seq:
            s.submit(side, c, q, oid, 33)
        stamps = [_stamp(s.st, side, oid) for side, _c, _q, oid in seq]
        assert all(sec == 33 for sec, _ in stamps)
        nsecs = [n for _, n in stamps]
        assert nsecs == sorted(nsecs) and len(set(nsecs)) == 4, stamps

    def test_same_second_new_order_after_partial_fill_ranks_after_all(self, book):
        """A same-second NON residual is ranked behind every row of that
        second on BOTH sides (its peers include the opposite side)."""
        s = _Session(book)
        s.submit(SELL, 6129, 7, 52, 70)
        s.submit(SELL, 6140, 4, 7710, 70)
        s.submit(BUY, 6101, 6, 3, 70)
        # crossing BUY: takes 7 @ 61.29, residual 5 rests at 61.33
        s.submit(BUY, 6133, 12, 604, 70, expect=[(6129, 7, 52)])
        n_resid = _stamp(s.st, BUY, 604)
        others = [_stamp(s.st, SELL, 7710), _stamp(s.st, BUY, 3)]
        assert n_resid[0] == 70
        assert all(n_resid[1] > o[1] for o in others), (n_resid, others)

    def test_cancel_then_same_second_arrival_stays_unique(self, book):
        """Cancelling a same-second row then adding another order in that
        second: stamps stay unique and the new order ranks last."""
        s = _Session(book, model=False)
        for side, c, oid in ((BUY, 3906, 11), (SELL, 3990, 4205), (BUY, 3911, 87)):
            s.submit(side, c, 5, oid, 9)
        s.st = book.cancel_order(s.st, 4205, SELL)
        _assert_invariants(s.st, "after cancel")
        s.submit(SELL, 3990, 3, 56001, 9)
        _assert_invariants(s.st, "after re-add")
        new = _stamp(s.st, SELL, 56001)
        assert new[0] == 9
        assert new[1] > max(_stamp(s.st, BUY, 11)[1], _stamp(s.st, BUY, 87)[1])

    def test_c03_seed11_same_second_fixture(self, book):
        """C03_seed11_same_second.npz (4 rows): after every submit every live
        stamp is unique; the remaining stamps sort in processing order; the
        tape equals a plain price-time walk."""
        tape, _params = _load_fixture("C03_seed11_same_second.npz")
        s = _Session(book)
        _replay(s, tape)
        assert [(r[0], r[3]) for r in sorted(_live(s.st), key=lambda r: (r[4], r[5]))] \
            == s.ref.order()


# ---------------------------------------------------------------------------
# E2 — call site NON
# ---------------------------------------------------------------------------

_NON_CASES = [
    # rest side, price, oids (A, B, agg1, agg2, agg3), seconds (A, agg1, B, agg2, agg3)
    pytest.param(BUY, 4731, (73, 40512, 9, 618, 200033), (3, 8, 8, 20, 21), id="buy-B-same-second"),
    pytest.param(SELL, 6208, (300117, 58, 4, 77001, 912), (0, 5, 6, 30, 30), id="sell-B-next-second"),
]


class TestE2NonSite:

    @pytest.mark.parametrize("rest, x, oids, ts", _NON_CASES)
    def test_slice_activated_by_non_is_ahead_of_later_arrival(self, book, rest, x, oids, ts):
        a, b, g1, g2, g3 = oids
        ta, t1, tb, t2, t3 = ts
        agg = SELL if rest == BUY else BUY
        s = _Session(book)
        s.submit(rest, x, 50, a, ta, ICE, peak=12)
        # agg1 exhausts two slices: 12, 12, then 7 of the third (activated at t1)
        s.submit(agg, x, 31, g1, t1, expect=[(x, 12, a), (x, 12, a), (x, 7, a)])
        assert _stamp(s.st, rest, a)[0] == t1, "slice not dated at its activation"
        s.submit(rest, x, 6, b, tb)
        assert _queue(s.st, rest, _P(x)) == [a, b], "later arrival jumped the slice"
        s.submit(agg, x, 9, g2, t2, expect=[(x, 5, a), (x, 4, b)])
        # A's next slice was activated at t2, behind B's remainder
        assert _stamp(s.st, rest, a)[0] == t2
        assert _queue(s.st, rest, _P(x)) == [b, a]
        # agg3: B's 2, A's 12 (exhausted), A's last slice 2, residual 4 rests
        s.submit(agg, x, 20, g3, t3, expect=[(x, 2, b), (x, 12, a), (x, 2, a)])
        assert _stamp(s.st, agg, g3)[0] == t3

    def test_counter_far_ahead_of_tape_slice_still_ahead_of_later_tape(self, book):
        """Many slice activations push the counter far ahead of the tape clock;
        a tape order arriving 3 s after the activating process (well below
        the counter) still queues behind the slice."""
        s = _Session(book)
        x = 2589
        s.submit(SELL, x, 60, 4, 1, ICE, peak=4)
        s.submit(BUY, x, 27, 81122, 2, expect=[(x, 4, 4)] * 6 + [(x, 3, 4)])
        assert int(s.st.ts_counter) > 5 + 3, "fixture: counter not ahead"
        s.submit(SELL, x, 10, 935, 5)          # tape, 3 s after activation
        s.submit(BUY, x, 3, 66, 6, expect=[(x, 1, 4), (x, 2, 935)])

    def test_non_remainder_precedes_slices_its_process_activated(self, book):
        """ICEwPI makes the activated slice non-crossing, so the NON
        aggressor's remainder rests in the same process. Its stamp keeps the
        entry second and precedes the slice (both sides, same-second peer)."""
        s = _Session(book, model=False)
        s.submit(SELL, 7012, 40, 31, 12, ICE, peak=11, ppd=4)  # 70.12, +0.04
        s.submit(BUY, 1205, 3, 5555, 40)                       # peer (40, 0)
        s.submit(SELL, 7099, 2, 82, 40)                        # peer (40, 1)
        # BUY NON 3.0 @ 70.14: 11 @ 70.12; slice 1 @ 70.16 does not cross
        s.submit(BUY, 7014, 30, 444, 40, expect=[(7012, 11, 31)])
        rem = _stamp(s.st, BUY, 444)
        sl = _stamp(s.st, SELL, 31)
        assert rem[0] == 40 and sl[0] == 40, (rem, sl)
        assert rem[1] > max(_stamp(s.st, BUY, 5555)[1], _stamp(s.st, SELL, 82)[1])
        assert sl[1] > rem[1], f"slice {sl} not after its activator's remainder {rem}"

    def test_non_remainder_precedes_slice_mirror_buy_iceberg(self, book):
        s = _Session(book, model=False)
        s.submit(BUY, 9387, 25, 120034, 2, ICE, peak=7, ppd=-6)  # 93.87, -0.06
        s.submit(SELL, 9385, 20, 8, 3, expect=[(9387, 7, 120034)])
        rem, sl = _stamp(s.st, SELL, 8), _stamp(s.st, BUY, 120034)
        assert rem[0] == sl[0] == 3
        assert sl[1] > rem[1]
        # the slice now rests at 93.81, below the remainder at 93.85
        assert [r[1] for r in _live(s.st) if r[3] == 120034] == [_P(9381)]


# ---------------------------------------------------------------------------
# E2 — call site IOC
# ---------------------------------------------------------------------------

class TestE2IocSite:

    @pytest.mark.parametrize("rest, x, far, oids, e", [
        pytest.param(BUY, 8344, 8361, (5, 70707, 31, 9902, 2), 14, id="buy"),
        pytest.param(SELL, 1499, 1460, (130021, 6, 444, 81, 70), 3, id="sell"),
    ])
    def test_ioc_slices_ranked_after_same_second_peers_and_before_later(
            self, book, rest, x, far, oids, e):
        """An IOC exhausts slices then cancels its residual. The slice is dated
        at the IOC's second, after a same-second peer on the opposite side,
        and ahead of a later arrival in that same second."""
        a, peer, ioc, b, g = oids
        agg = SELL if rest == BUY else BUY
        s = _Session(book)
        s.submit(rest, x, 50, a, e - 2, ICE, peak=9)
        s.submit(agg, far, 4, peer, e)              # non-crossing, (e, 0)
        # IOC 3.0 with liquidity only at x: 9, 9, 9, 3 -> slice 4 shows 6
        s.submit(agg, x, 30, ioc, e, IOC,
                 expect=[(x, 9, a), (x, 9, a), (x, 9, a), (x, 3, a)])
        assert not [r for r in _live(s.st) if r[3] == ioc], "IOC residual rested"
        sl = _stamp(s.st, rest, a)
        assert sl[0] == e and sl[1] > _stamp(s.st, agg, peer)[1]
        s.submit(rest, x, 5, b, e)
        s.submit(agg, x, 8, g, e + 1, expect=[(x, 6, a), (x, 2, b)])

    def test_ioc_dates_slices_at_its_second_and_rests_nothing(self, book):
        s = _Session(book)
        s.submit(SELL, 5560, 16, 4011, 9, ICE, peak=5)
        s.submit(SELL, 5561, 3, 72, 9)
        s.submit(BUY, 5560, 13, 1, 11, IOC,
                 expect=[(5560, 5, 4011), (5560, 5, 4011), (5560, 3, 4011)])
        assert _stamp(s.st, SELL, 4011)[0] == 11
        assert _queue(s.st, SELL, _P(5560)) == [4011]


# ---------------------------------------------------------------------------
# E2 — call site FOK
# ---------------------------------------------------------------------------

class TestE2FokSite:

    def test_c15_slice_stamp_fixture(self, book):
        """C15_r2_slice_stamp_seed100000254.npz: FOK 8 (t 8) leaves oid 5's
        slice activated at t 8; BUY ICE 13 (t 11) rests at the same price;
        SELL ICE 24 (t 20) trades 2 @ oid 5, then 21 @ oid 13 (rows split at
        24's instance boundaries, OMIE-ICE-INCOMING-LADDER-1)."""
        tape, _ = _load_fixture("C15_r2_slice_stamp_seed100000254.npz")
        s = _Session(book)
        got = _replay(s, tape)
        last = got[-1]
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): the walk 2 @ oid 5, 21 @
        # oid 13 is printed split at the incoming ICE 24's instance boundaries
        # (its peak, read from the tape), via the reference's incoming branch.
        peak24 = int(tape[tape[:, _C_OID] == 24][0, _C_PEAK])
        walk = _split_incoming([(0, 2, 5, 24), (0, 21, 13, 24)], peak24)
        assert [(q, po) for _p, q, po, _a in last] == [(q, po) for _p, q, po, _a in walk], last

    def test_fok_mirror_sell_iceberg_other_constants(self, book):
        s = _Session(book)
        x = 5503
        s.submit(SELL, x, 67, 314, 5, ICE, peak=16)
        s.submit(BUY, x + 27, 62, 2, 8, FOK,
                 expect=[(x, 16, 314), (x, 16, 314), (x, 16, 314), (x, 14, 314)])
        assert _stamp(s.st, SELL, 314)[0] == 8
        s.submit(SELL, x, 74, 13, 9, ICE, peak=24)
        s.submit(BUY, x + 40, 23, 240001, 20, ICE, peak=5,
                 expect=[(x, 2, 314), (x, 21, 13)])

    def test_fok_two_icebergs_same_price_exhaustion_order(self, book):
        """Multi-pass FOK alternates two icebergs at one price; the slices it
        activates queue in exhaustion order, and a same-second later
        arrival queues behind them."""
        s = _Session(book)
        x = 7718
        s.submit(BUY, x, 20, 9001, 1, ICE, peak=5)
        s.submit(BUY, x, 20, 27, 2, ICE, peak=4)
        s.submit(SELL, x - 3, 15, 645, 6, FOK,
                 expect=[(x, 5, 9001), (x, 4, 27), (x, 5, 9001), (x, 1, 27)])
        # activations: 9001#2, 27#2, 9001#3 -> 27's live slice precedes 9001's
        assert _queue(s.st, BUY, _P(x)) == [27, 9001]
        s.submit(BUY, x, 4, 3, 6)
        assert _queue(s.st, BUY, _P(x)) == [27, 9001, 3]

    def test_fok_killed_changes_no_stamp(self, book):
        s = _Session(book)
        s.submit(SELL, 3030, 12, 66, 2, ICE, peak=3)
        s.submit(SELL, 3031, 4, 501, 4)
        rows = sorted(_live(s.st))
        s.submit(BUY, 3030, 13, 77, 9, FOK, expect=[])
        assert sorted(_live(s.st)) == rows


# ---------------------------------------------------------------------------
# E2 — call site incoming ICE
# ---------------------------------------------------------------------------

class TestE2IncomingIceSite:

    @pytest.mark.parametrize("rest, x, oids, ts", [
        pytest.param(BUY, 6650, (2, 50123, 9, 787), (2, 6, 6, 7), id="buy"),
        pytest.param(SELL, 1187, (440, 3, 120120, 61), (10, 31, 32, 32), id="sell"),
    ])
    def test_incoming_ice_sweep_slices_dated_at_its_process(self, book, rest, x, oids, ts):
        a, b, ice, g = oids
        ta, te, tb, tg = ts
        agg = SELL if rest == BUY else BUY
        s = _Session(book)
        s.submit(rest, x, 40, a, ta, ICE, peak=7)
        # incoming ICE total 2.5 (peak 0.6): sweeps 7, 7, 7, 4 -> A shows 3
        s.submit(agg, x, 25, ice, te, ICE, peak=6,
                 expect=[(x, 7, a), (x, 7, a), (x, 7, a), (x, 4, a)])
        assert _stamp(s.st, rest, a)[0] == te
        s.submit(rest, x, 9, b, tb)
        s.submit(agg, x, 5, g, tg, expect=[(x, 3, a), (x, 2, b)])

    @pytest.mark.parametrize("ice_side", [BUY, SELL])
    def test_incoming_ice_first_slice_precedes_slices_it_activated(self, book, ice_side):
        """Incoming ICE sweeps one ICEwPI slice, the next (drifted) slice no
        longer crosses, and the remainder rests as an iceberg at the entry
        second. The remainder's first slice precedes the slice it activated,
        and both follow a same-second peer."""
        s = _Session(book, model=False)
        if ice_side == BUY:
            s.submit(SELL, 7012, 40, 31, 1, ICE, peak=11, ppd=4)
            s.submit(SELL, 7062, 2, 900, 9)                   # peer (9, 0)
            s.submit(BUY, 7014, 30, 5150, 9, ICE, peak=8, expect=[(7012, 11, 31)])
            own, sl, peer = (_stamp(s.st, BUY, 5150), _stamp(s.st, SELL, 31),
                             _stamp(s.st, SELL, 900))
            vis = [r[2] for r in _live(s.st) if r[3] == 5150]
        else:
            s.submit(BUY, 2240, 33, 77777, 4, ICE, peak=10, ppd=-5)
            s.submit(BUY, 2101, 1, 6, 18)                     # peer (18, 0)
            s.submit(SELL, 2237, 21, 12, 18, ICE, peak=4, expect=[(2240, 10, 77777)])
            own, sl, peer = (_stamp(s.st, SELL, 12), _stamp(s.st, BUY, 77777),
                             _stamp(s.st, BUY, 6))
            vis = [r[2] for r in _live(s.st) if r[3] == 12]
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): v_1 = min(P - u, R) from
        # the reference's incoming branch (BUY: 3.0 / 0.8 swept 1.1 -> 0.5;
        # SELL: 2.1 / 0.4 swept 1.0 -> 0.2), not min(P, R).
        assert vis == ([_incoming_first_slice(30, 11, 8)] if ice_side == BUY
                       else [_incoming_first_slice(21, 10, 4)])
        assert own[0] == sl[0] == peer[0]
        assert peer[1] < own[1] < sl[1], (peer, own, sl)


# ---------------------------------------------------------------------------
# E2 — queue placement rules shared by all sites
# ---------------------------------------------------------------------------

class TestE2QueuePlacement:

    @pytest.mark.parametrize("rest", [BUY, SELL])
    def test_slice_queues_behind_orders_then_resting_at_its_price(self, book, rest):
        agg = SELL if rest == BUY else BUY
        x = 4447 if rest == BUY else 9123
        s = _Session(book)
        s.submit(rest, x, 20, 81, 1, ICE, peak=6)
        s.submit(rest, x, 4, 302, 2)
        s.submit(agg, x, 6, 4, 5, expect=[(x, 6, 81)])
        assert _queue(s.st, rest, _P(x)) == [302, 81]
        s.submit(agg, x, 7, 15001, 6, expect=[(x, 4, 302), (x, 3, 81)])

    @pytest.mark.parametrize("rest", [BUY, SELL])
    def test_exhaustion_order_within_one_process_same_price(self, book, rest):
        agg = SELL if rest == BUY else BUY
        x = 3371 if rest == BUY else 2898
        a, c, d = (12, 90210, 5) if rest == BUY else (700001, 44, 3)
        s = _Session(book)
        s.submit(rest, x, 30, a, 1, ICE, peak=5)
        s.submit(rest, x, 30, c, 2, ICE, peak=4)
        s.submit(agg, x, 9, 6060, 7, expect=[(x, 5, a), (x, 4, c)])
        sa, sc = _stamp(s.st, rest, a), _stamp(s.st, rest, c)
        assert sa[0] == sc[0] == 7 and sa[1] < sc[1], (sa, sc)
        s.submit(rest, x, 6, d, 7)
        s.submit(agg, x, 12, 616, 8, expect=[(x, 5, a), (x, 4, c), (x, 3, d)])

    def test_exhaustion_order_across_prices_icewpi(self, book):
        """Slice of A (better price) exhausted first, its drifted next slice
        no longer best; then C's slice at a worse price is exhausted. Both
        are dated at the process second; A's stamp precedes C's."""
        s = _Session(book, model=False)
        s.submit(SELL, 4100, 30, 5, 1, ICE, peak=6, ppd=9)    # 41.00, next 41.09
        s.submit(SELL, 4103, 30, 6118, 2, ICE, peak=5)        # 41.03
        s.submit(BUY, 4103, 11, 70, 4, expect=[(4100, 6, 5), (4103, 5, 6118)])
        sa, sc = _stamp(s.st, SELL, 5), _stamp(s.st, SELL, 6118)
        assert sa[0] == sc[0] == 4
        assert sa[1] < sc[1], (sa, sc)

    @pytest.mark.parametrize("rest", [BUY, SELL])
    def test_slice_stays_behind_counter_stamped_agent_order(self, book, rest):
        """Inner max in S_r: an agent order (stamped from the counter, ahead
        of the tape clock) rests at the iceberg's price; a tape aggressor at
        an EARLIER second exhausts the iceberg's slice. The slice must not
        jump ahead of the agent order, which arrived first."""
        agg = SELL if rest == BUY else BUY
        x = 5099 if rest == BUY else 6301
        a, gid = (40, 11) if rest == BUY else (3, 404404)
        s = _Session(book)
        if rest == SELL:  # push the counter well ahead of the tape first
            s.submit(BUY, 100, 40, 9, 2, ICE, peak=2)
            s.submit(SELL, 100, 39, 10, 2)
        t = 3 if rest == BUY else 2
        s.submit(rest, x, 40, a, t, ICE, peak=8)
        ag_ts = s.agent_ts()
        assert ag_ts > t
        s.submit(rest, x, 5, gid, ag_ts, agent=990)
        s.submit(agg, x, 8, 6, t, expect=[(x, 8, a)])
        assert _stamp(s.st, rest, a)[0] == ag_ts, "slice not dated at the agent's second"
        assert _queue(s.st, rest, _P(x)) == [gid, a]
        s.submit(agg, x, 7, 7, t, expect=[(x, 5, gid), (x, 2, a)])


# ---------------------------------------------------------------------------
# Session-level: random tape-only and agent-mixed sessions vs the model
# ---------------------------------------------------------------------------

def _random_session(book, seed, n, agents=False):
    rng = random.Random(seed)
    s = _Session(book)
    base = rng.randint(-2000, 9000)
    t = rng.randint(0, 3000)
    oid = rng.choice([1, 37, 512, 4021, 61177, 300001])
    for i in range(n):
        if _trade_rows_used(s.st) > 50:   # keep clear of the 100-row tape buffer
            break
        side = rng.randint(0, 1)
        er = rng.choices([NON, IOC, FOK, ICE], [4, 1, 1, 3])[0]
        cents = base + rng.randint(-3, 3)
        q = rng.randint(1, 40)
        peak = rng.randint(2, max(2, q // 2)) if er == ICE else 0
        if er == ICE:
            peak = min(peak, q)
        t += rng.choice([0, 0, 1, 2, 7])
        oid += rng.randint(1, 900)
        ts = s.agent_ts() if (agents and rng.random() < 0.3) else t
        s.submit(side, cents, q, oid, ts, er, peak, agent=4 if ts != t else None)
        if not agents:
            got = [(r[0], r[3]) for r in sorted(_live(s.st), key=lambda r: (r[4], r[5]))]
            assert got == s.ref.order(), (
                f"seed {seed} step {i}: stored stamps not in processing order")
        refrows = {(x["side"], x["oid"]): x for x in s.ref.book}
        for r in _live(s.st):
            x = refrows.get((r[0], r[3]))
            assert x is not None and (x["p"], x["vis"], x["sec"]) == (r[1], r[2], r[4]), (
                f"seed {seed} step {i}: row {r} vs reference {x}")
        assert len(refrows) == len(_live(s.st)), f"seed {seed} step {i}: row count"
    return s


class TestRandomSessions:

    @pytest.mark.parametrize("seed", range(0, 24))
    def test_tape_only_session_matches_processing_order_walk(self, book, seed):
        """Tape-only mixed NON/IOC/FOK/ICE session on a 7-tick band: after
        every submit the tape equals the reference walk, every row's SEC is
        its entry/activation second, the stamps are unique, below the
        counter, and sort in processing order."""
        _random_session(book, 1000 + seed, 28)

    @pytest.mark.parametrize("seed", range(0, 16))
    def test_agent_mixed_session_matches_card_queue(self, book, seed):
        """30% of orders are counter-stamped (agent-style). Trades follow the
        stored queue; slices never jump an order resting at their price."""
        _random_session(book, 5000 + seed, 28, agents=True)

    @pytest.mark.parametrize("seed", range(0, 6))
    def test_small_iceberg_cap_keeps_stamp_invariants(self, seed):
        """Under a tiny iceberg table (eviction regime, engineering limit)
        the stamp invariants (uniqueness, S5-A2) still hold after every
        submit. Trades are not compared (eviction drops hidden volume)."""
        bk = ContractBook(_contract(), _PC, OMIEMarketConfig(max_icebergs=2))
        rng = random.Random(9000 + seed)
        st = bk.init()
        t, oid = rng.randint(0, 40), rng.choice([3, 4021, 300001])
        for i in range(30):
            if _trade_rows_used(st) > 50:
                break
            side = rng.randint(0, 1)
            er = rng.choices([NON, IOC, FOK, ICE], [3, 1, 1, 4])[0]
            q = rng.randint(1, 40)
            peak = min(q, rng.randint(2, max(2, q // 2))) if er == ICE else 0
            t += rng.choice([0, 1, 4])
            oid += rng.randint(1, 300)
            ts = int(st.ts_counter) if rng.random() < 0.2 else t
            st, r = bk.submit_order(st, _mk(side, 4800 + rng.randint(-3, 3), q,
                                            oid, ts, er, peak))
            _assert_invariants(st, f"seed {seed} step {i}")


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

# Certification tape layout (omiecid_simulator/jaxomiecid/omie_tape_synth.py).
_C_SIDE, _C_PRICE, _C_QTY, _C_OID, _C_AGENT, _C_TS = 1, 2, 3, 4, 5, 6
_C_EXEC, _C_PEAK, _C_PPD = 7, 11, 12


def _load_fixture(name):
    path = _FIX / name
    if not path.exists():
        pytest.skip(f"fixture {path} not present")
    z = np.load(path, allow_pickle=False)
    return np.asarray(z["tape"]), json.loads(str(z["params_json"]))


def _replay(session, tape):
    """Replay a certification tape (PPD = 0 rows) through the session."""
    out = []
    for r in tape:
        if int(r[_C_OID]) < 1:
            continue
        assert int(r[_C_PPD]) == 0, "fixture with PPD not supported by the model"
        order = make_order(
            contract_id=_CID, side=Side(int(r[_C_SIDE])),
            price=decode_price(int(r[_C_PRICE]), _PC),
            quantity=decode_qty(int(r[_C_QTY]), _PC),
            order_id=int(r[_C_OID]), agent_id=int(r[_C_AGENT]),
            timestamp=int(r[_C_TS]), exec_restriction=ExecRestriction(int(r[_C_EXEC])),
            peak_qty=decode_qty(int(r[_C_PEAK]), _PC),
            ppd=decode_ppd(int(r[_C_PPD]), _PC),
        )
        before = session.st.lob.trades
        session.st, res = session.book.submit_order(session.st, order)
        assert bool(res.accepted)
        got = _new_trades(before, session.st.lob.trades)
        want = session.ref.submit(int(r[_C_SIDE]), int(r[_C_PRICE]), int(r[_C_QTY]),
                                  int(r[_C_OID]), int(r[_C_TS]), int(r[_C_EXEC]),
                                  int(r[_C_PEAK]))
        assert got == want, f"oid {int(r[_C_OID])}: tape {got} != walk {want}"
        _assert_invariants(session.st, f"after oid {int(r[_C_OID])}")
        out.append(got)
    return out
