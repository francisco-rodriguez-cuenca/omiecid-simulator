"""Audit twin for OMIE-ICE-INCOMING-LADDER-1 (Step 5, mode: audit).

An incoming iceberg sweeps as a chain of instances (market-spec §4.2.1
Rule 2, *Incoming iceberg*, items (1)-(6), amended 2026-10-20; Rule 3,
*ICE aggressor*; OQ-33 / OQ-34 blocks of 2026-10-20; OQ-15).

This file is derived from the rule text only.  It carries its own pure-
Python reference walk (``_Walk``) of a single contract book, in integer
ticks, and compares the engine (public ``ContractBook.submit_order``,
jitted, and under ``jax.disable_jit()``) against it:

* the incoming iceberg (standard, d = 0, and ICEwPI, d != 0) is a chain of
  instances of size P; instance k trades only where its own limit b + k d
  crosses; the sweep stops at the first instance that cannot trade;
* each trade is between ONE resting entry (a NON order, or one instance of a
  resting iceberg) and ONE incoming instance, priced at the resting entry;
  a row ends where the resting entry ends or the incoming instance ends;
* the remainder R rests as instance k_e = floor(M / P) at b + k_e d, shows
  v1 = min(P - u, R) (u = M - k_e P) and hides R - v1; later instances
  continue the ladder at b + (k_e + 1) d, ...;
* a regenerated slice is a new entry (new timestamp) ranked at its own price
  (Rule 2, *Re-ranking*), and a FOK is filled only if the full multi-slice
  walk fills it (Rule 1);
* simulator limit: if the 100-row trade ring has no room for the split rows
  the rows are left unsplit (all or nothing); on a saturated ring (OQ-15)
  the measured sweep M under-counts, and only what the rule implies on the
  measured M is asserted.

Columns of a trade row (jaxob ``TradesFeat``): P, Q, PASS_OID, AGRS_OID,
SEC, NSEC, PASS_TID, AGRS_TID.  The walk predicts every column except the
engine's timestamps SEC / NSEC; Q is signed (negative when the aggressor is
a BUY, jaxob's own convention, which this story does not touch).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat, TradesFeat
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_REMAINING_HIDDEN,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.order_validator import validate_order


# ---------------------------------------------------------------------------
# Constants and engine encodings (public config only)
# ---------------------------------------------------------------------------

_CID = 41
_BUY, _SELL = int(Side.BUY), int(Side.SELL)
_NON, _IOC, _FOK, _ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                          int(ExecRestriction.FOK), int(ExecRestriction.ICE))

_PC = OMIEProductConfig()
_MC = OMIEMarketConfig()
_N_TRADES = _MC.nTrades                      # 100-row trade ring
_PRICE_OFF = int(round(-_PC.price_min / _PC.price_tick))   # cents -> jaxob int

_TP, _TQ = TradesFeat.P.value, TradesFeat.Q.value
_TPASS, _TAGRS = TradesFeat.PASS_OID.value, TradesFeat.AGRS_OID.value
_TSEC, _TNSEC = TradesFeat.SEC.value, TradesFeat.NSEC.value
_TPTID, _TATID = TradesFeat.PASS_TID.value, TradesFeat.AGRS_TID.value
# The walk-predictable columns, in this order.
_CMP_COLS = (_TP, _TQ, _TPASS, _TAGRS, _TPTID, _TATID)
# Merge key: every trade column except Q.
_MERGE_KEY = tuple(c for c in range(8) if c != _TQ)

_SP, _SQ, _SOID = OrderSideFeat.P.value, OrderSideFeat.Q.value, OrderSideFeat.OID.value
_SSEC, _SNSEC = OrderSideFeat.SEC.value, OrderSideFeat.NSEC.value


def _contract() -> Contract:
    return Contract(
        contract_id=_CID,
        delivery_start_minutes=17 * 60 + 30,
        delivery_end_minutes=17 * 60 + 45,
        gct_minutes=16 * 60 + 30,
        status=ContractStatus.OPEN,
    )


_BOOK = ContractBook(_contract(), _PC, _MC)


# ---------------------------------------------------------------------------
# Order spec (integer ticks: price in cents, quantity in 0.1 MW, d in cents)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class O:
    side: int
    price: int          # cents
    qty: int            # 0.1 MW ticks
    oid: int
    agent: int
    er: int = _NON
    peak: int = 0       # ticks
    d: int = 0          # cents per instance

    def to_omie(self, ts: int):
        o = make_order(
            contract_id=_CID, side=Side(self.side), price=self.price / 100.0,
            quantity=self.qty / 10.0, order_id=self.oid, agent_id=self.agent,
            timestamp=ts, exec_restriction=ExecRestriction(self.er),
            peak_qty=self.peak / 10.0, ppd=self.d / 100.0,
        )
        assert bool(validate_order(o, _PC, True)), f"fixture bug: invalid {self}"
        return o


# ---------------------------------------------------------------------------
# Independent reference walk
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    oid: int
    agent: int
    side: int
    price: int
    vis: int
    seq: int
    ice: bool = False
    hidden: int = 0
    peak: int = 0
    d: int = 0
    base: int = 0        # price of instance 0
    inst: int = 0        # absolute instance index of the visible slice


@dataclass
class _Row:
    price: int           # cents
    q: int               # signed ticks
    pass_oid: int
    agrs_oid: int
    pass_tid: int
    agrs_tid: int
    eid: int             # resting-entry instance identity (seq), walk only

    def cmp(self):
        return (self.price + _PRICE_OFF, self.q, self.pass_oid,
                self.agrs_oid, self.pass_tid, self.agrs_tid)


@dataclass
class _Walk:
    entries: list = field(default_factory=list)
    seq: int = 0

    # -- helpers ---------------------------------------------------------
    def _next_seq(self):
        self.seq += 1
        return self.seq

    @staticmethod
    def _better(side, a, b):
        """True if price a ranks strictly better than b on ``side``."""
        return a > b if side == _BUY else a < b

    def ranked(self, side):
        es = [e for e in self.entries if e.side == side]
        key = (lambda e: (-e.price, e.seq)) if side == _BUY else (
            lambda e: (e.price, e.seq))
        return sorted(es, key=key)

    @staticmethod
    def _crosses(agg_side, limit, counter_price):
        return counter_price <= limit if agg_side == _BUY else counter_price >= limit

    def clone(self):
        w = _Walk(seq=self.seq)
        w.entries = [_Entry(**e.__dict__) for e in self.entries]
        return w

    def _consume(self, e, q):
        e.vis -= q
        if e.vis > 0:
            return
        if e.ice and e.hidden > 0:
            # Regenerate the next instance: new entry, new timestamp, own price.
            e.inst += 1
            e.vis = min(e.peak, e.hidden)
            e.hidden -= e.vis
            e.price = e.base + e.inst * e.d
            e.seq = self._next_seq()
        else:
            self.entries.remove(e)

    def _sweep(self, o: O, ladder: bool):
        """Match ``o`` against the opposite side; return (rows, filled)."""
        opp = 1 - o.side
        sgn = -1 if o.side == _BUY else 1
        rows, filled = [], 0
        P = max(o.peak, 1)
        while filled < o.qty:
            if ladder:
                k = filled // P
                limit = o.price + k * o.d
                room = P * (k + 1) - filled
            else:
                limit, room = o.price, o.qty
            book = self.ranked(opp)
            if not book or not self._crosses(o.side, limit, book[0].price):
                break
            e = book[0]
            q = min(o.qty - filled, e.vis, room)
            rows.append(_Row(e.price, sgn * q, e.oid, o.oid, e.agent, o.agent,
                             e.seq))
            filled += q
            self._consume(e, q)
        return rows, filled

    def matchable(self, side, limit):
        """Volume a FOK on ``side`` at ``limit`` could fill (full walk)."""
        w = self.clone()
        big = O(side, limit, 10 ** 9, -1, -1)
        _, filled = w._sweep(big, ladder=False)
        return filled

    def submit(self, o: O):
        """Apply one order; return (rows, filled, accepted_traded)."""
        seq = self._next_seq()
        if o.er == _FOK:
            if self.matchable(o.side, o.price) < o.qty:
                return [], 0
            rows, filled = self._sweep(o, ladder=False)
            assert filled == o.qty
            return rows, filled
        if o.er == _ICE:
            rows, M = self._sweep(o, ladder=True)
            R = o.qty - M
            if R > 0:
                P = o.peak
                k_e, u = M // P, M % P
                v1 = min(P - u, R)
                self.entries.append(_Entry(
                    oid=o.oid, agent=o.agent, side=o.side,
                    price=o.price + k_e * o.d, vis=v1, seq=seq, ice=True,
                    hidden=R - v1, peak=P, d=o.d, base=o.price, inst=k_e))
            return rows, M
        rows, filled = self._sweep(o, ladder=False)
        if o.er == _NON and filled < o.qty:
            self.entries.append(_Entry(o.oid, o.agent, o.side, o.price,
                                       o.qty - filled, seq))
        return rows, filled

    # -- views -------------------------------------------------------------
    def view(self):
        """Non-tape leaves: per side {oid: (price_int, vis)}, priority order,
        hidden per iceberg oid, n_active."""
        out = {}
        for side in (_BUY, _SELL):
            r = self.ranked(side)
            out[side] = ({e.oid: (e.price + _PRICE_OFF, e.vis) for e in r},
                         [e.oid for e in r])
        out["hidden"] = {e.oid: e.hidden for e in self.entries if e.ice}
        out["n_active"] = sum(1 for e in self.entries if e.ice)
        return out


def _unsplit(rows):
    """Merge consecutive rows against the same resting-entry instance."""
    out = []
    for r in rows:
        if out and out[-1].eid == r.eid and out[-1].pass_oid == r.pass_oid:
            out[-1] = _Row(r.price, out[-1].q + r.q, r.pass_oid, r.agrs_oid,
                           r.pass_tid, r.agrs_tid, r.eid)
        else:
            out.append(r)
    return out


def _tape_hash(rows_cmp) -> str:
    a = np.asarray(rows_cmp, dtype=np.int64).reshape(-1, len(_CMP_COLS))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Engine side
# ---------------------------------------------------------------------------

def _n_used(trades) -> int:
    t = np.asarray(trades)
    used = t[:, _TSEC] != -1
    return int(used.sum())


def _engine_view(state):
    out = {}
    for side, arr in ((_BUY, state.lob.bids), (_SELL, state.lob.asks)):
        a = np.asarray(arr)
        live = a[a[:, _SOID] > 0]
        d = {int(r[_SOID]): (int(r[_SP]), int(r[_SQ])) for r in live}
        if side == _BUY:
            order = sorted(live.tolist(), key=lambda r: (-r[_SP], r[_SSEC], r[_SNSEC]))
        else:
            order = sorted(live.tolist(), key=lambda r: (r[_SP], r[_SSEC], r[_SNSEC]))
        out[side] = (d, [int(r[_SOID]) for r in order])
    n = int(state.n_active)
    meta = np.asarray(state.iceberg_meta)[:n]
    out["hidden"] = {int(r[COL_ORDER_ID]): int(r[COL_REMAINING_HIDDEN]) for r in meta}
    out["n_active"] = n
    return out


def _submit(state, o: O, ts: int, eager: bool = False):
    n0 = _n_used(state.lob.trades)
    if eager:
        with jax.disable_jit():
            new, res = _BOOK.submit_order(state, o.to_omie(ts))
    else:
        new, res = _BOOK.submit_order(state, o.to_omie(ts))
    t = np.asarray(new.lob.trades)
    rows = t[n0:][t[n0:, _TSEC] != -1]
    return new, res, rows


def _cmp_rows(rows):
    return [tuple(int(r[c]) for c in _CMP_COLS) for r in rows]


def _merge_engine_rows(rows):
    out = []
    for r in rows:
        key = tuple(int(r[c]) for c in _MERGE_KEY)
        if out and out[-1][0] == key:
            out[-1][1] += int(r[_TQ])
        else:
            out.append([key, int(r[_TQ])])
    merged = []
    for key, q in out:
        full = dict(zip(_MERGE_KEY, key))
        full[_TQ] = q
        merged.append(tuple(full[c] for c in _CMP_COLS))
    return merged


def _run_session(orders, eager_from=None, state=None):
    """Run orders through engine and walk; return per-call records."""
    if state is None:
        state = _BOOK.init()
    walk = _Walk()
    recs = []
    for i, o in enumerate(orders):
        ts = 10 * (i + 1)
        eager = eager_from is not None and i >= eager_from
        state, res, rows = _submit(state, o, ts, eager=eager)
        wrows, wfilled = walk.submit(o)
        recs.append(dict(order=o, res=res, rows=rows, wrows=wrows,
                         wfilled=wfilled, eview=_engine_view(state),
                         wview=walk.view(), state=state))
    return recs, walk, state


def _assert_call(rec, ctx=""):
    o, res = rec["order"], rec["res"]
    eng = _cmp_rows(rec["rows"])
    wlk = [r.cmp() for r in rec["wrows"]]
    assert bool(res.accepted), f"{ctx} rejected {o}"
    assert eng == wlk, f"{ctx} tape mismatch for {o}\n engine={eng}\n walk  ={wlk}"
    assert abs(float(res.trade_qty) - rec["wfilled"] / 10.0) < 1e-3, ctx
    assert bool(res.traded) == (rec["wfilled"] > 0), ctx
    assert rec["eview"] == rec["wview"], (
        f"{ctx} book mismatch after {o}\n engine={rec['eview']}\n walk  ={rec['wview']}")


def _P(eur: float) -> int:
    return int(round(eur * 100))


def _Q(mw: float) -> int:
    return int(round(mw * 10))


# ---------------------------------------------------------------------------
# Walk self-checks against the worked figures of the rule (no engine)
# ---------------------------------------------------------------------------

def _gestvd2_orders():
    return [
        O(_BUY, _P(112.36), _Q(1.0), 4711, 31),
        O(_BUY, _P(112.34), _Q(20.5), 58, 32),
        O(_BUY, _P(112.00), _Q(11.9), 90210, 33),
        O(_BUY, _P(110.00), _Q(10.0), 306, 34),
        O(_SELL, _P(110.00), _Q(100.0), 777001, 35, _ICE, _Q(5.0), 75),
        O(_BUY, _P(114.00), _Q(7.0), 12, 36),
    ]


class TestWalkReproducesRuleFigures:
    """The audit walk itself reproduces the rule's worked examples."""

    def test_gestvd2_replica(self):
        w = _Walk()
        orders = _gestvd2_orders()
        for o in orders[:4]:
            w.submit(o)
        rows, M = w.submit(orders[4])
        assert [(r.price, abs(r.q)) for r in rows] == [
            (11236, 10), (11234, 40), (11234, 50), (11234, 50), (11234, 50)]
        assert M == 200
        v = w.view()
        assert v[_SELL][0][777001] == (_PRICE_OFF + 11300, 50)
        assert v["hidden"][777001] == 750
        assert v[_BUY][0] == {58: (_PRICE_OFF + 11234, 15),
                              90210: (_PRICE_OFF + 11200, 119),
                              306: (_PRICE_OFF + 11000, 100)}
        later = w.submit(orders[5])[0]
        assert [(r.price, abs(r.q)) for r in later] == [(11300, 50), (11375, 20)]

    def test_part_used_instance(self):
        w = _Walk()
        w.submit(O(_SELL, _P(50.00), _Q(3.0), 222, 1))
        rows, M = w.submit(O(_BUY, _P(51.00), _Q(10.0), 3333, 2, _ICE, _Q(2.0), -50))
        assert [(r.price, abs(r.q)) for r in rows] == [(5000, 20), (5000, 10)]
        v = w.view()
        assert v[_BUY][0][3333] == (_PRICE_OFF + 5050, 10)
        assert v["hidden"][3333] == 60
        later, _ = w.submit(O(_SELL, _P(49.00), _Q(7.0), 44, 3))
        assert [(r.price, abs(r.q)) for r in later] == [
            (5050, 10), (5000, 20), (4950, 20), (4900, 20)]

    def test_standard_split_and_display(self):
        w = _Walk()
        w.submit(O(_SELL, _P(50.00), _Q(6.0), 5, 1))
        w.submit(O(_SELL, _P(50.50), _Q(6.0), 66, 2))
        rows, _ = w.submit(O(_BUY, _P(51.00), _Q(20.0), 123456, 3, _ICE, _Q(4.0)))
        assert [(r.price, abs(r.q)) for r in rows] == [
            (5000, 40), (5000, 20), (5050, 20), (5050, 40)]
        v = w.view()
        assert v[_BUY][0][123456] == (_PRICE_OFF + 5100, 40)
        assert v["hidden"][123456] == 40
        assert [(r.price, abs(r.q)) for r in _unsplit(rows)] == [(5000, 60), (5050, 60)]


# ---------------------------------------------------------------------------
# Fixed sessions: engine (jitted) against the walk, plus pinned leaves/hashes
# ---------------------------------------------------------------------------

def _fixed_sessions():
    s = {}
    s["gestvd2"] = _gestvd2_orders()
    s["part_used"] = [
        O(_SELL, _P(50.00), _Q(3.0), 222, 1),
        O(_BUY, _P(51.00), _Q(10.0), 3333, 2, _ICE, _Q(2.0), -50),
        O(_SELL, _P(49.00), _Q(7.0), 44, 3),
    ]
    s["stop_sell"] = [
        O(_BUY, _P(61.50), _Q(2.0), 81, 1),
        O(_BUY, _P(60.00), _Q(3.0), 8282, 2),
        O(_SELL, _P(60.00), _Q(6.0), 64001, 3, _ICE, _Q(2.0), 100),
    ]
    s["stop_buy"] = [
        O(_SELL, _P(38.50), _Q(2.0), 18, 1),
        O(_SELL, _P(40.00), _Q(3.0), 2828, 2),
        O(_BUY, _P(40.00), _Q(6.0), 10046, 3, _ICE, _Q(2.0), -100),
    ]
    s["std_split"] = [
        O(_SELL, _P(50.00), _Q(6.0), 5, 1),
        O(_SELL, _P(50.50), _Q(6.0), 66, 2),
        O(_BUY, _P(51.00), _Q(20.0), 123456, 3, _ICE, _Q(4.0)),
    ]
    s["std_display"] = [
        O(_SELL, _P(50.00), _Q(6.0), 919, 1),
        O(_BUY, _P(50.00), _Q(20.0), 72, 2, _ICE, _Q(4.0)),
        O(_SELL, _P(50.00), _Q(3.0), 50505, 3),
    ]
    # (4) resting ICEwPI as counter at a step price: resting SELL ICE
    # base 70.13, d = +0.37, P = 1.5, Q = 6.0; incoming BUY ICE 71.00,
    # P = 2.0, d = -0.29, Q = 9.0.
    s["resting_ice_counter"] = [
        O(_SELL, _P(70.13), _Q(6.0), 314159, 1, _ICE, _Q(1.5), 37),
        O(_SELL, _P(70.60), _Q(1.3), 27, 2),
        O(_BUY, _P(71.00), _Q(9.0), 4242, 3, _ICE, _Q(2.0), -29),
        O(_SELL, _P(69.00), _Q(12.0), 999, 4),
    ]
    return s


_FIXED = _fixed_sessions()


def _walk_only(orders):
    w = _Walk()
    calls = [w.submit(o) for o in orders]
    return w, calls


# Tape hashes over the walk-predictable columns of every call of the session,
# computed from the audit WALK (not from the engine).
_PINNED_HASH = {
    "gestvd2": "e0a28cece370e5f4",              # 7 rows
    "part_used": "af5b544710028d1e",            # 6 rows
    "stop_sell": "16ad2c72dda5a09c",            # 1 row
    "stop_buy": "0b4c2f98a9c7967d",             # 1 row
    "std_split": "a1da56b607cc354b",            # 4 rows
    "std_display": "1cbbd6fd65ea58d5",          # 4 rows
    "resting_ice_counter": "1acc7da67cc9a25b",  # 7 rows
}


class TestFixedSessions:

    @pytest.mark.parametrize("name", sorted(_FIXED))
    def test_engine_matches_walk_every_call(self, name):
        recs, _, _ = _run_session(_FIXED[name])
        for i, rec in enumerate(recs):
            _assert_call(rec, ctx=f"[{name} call {i}]")

    @pytest.mark.parametrize("name", sorted(_FIXED))
    def test_walk_tape_hash_pinned_and_engine_reproduces_it(self, name):
        _, calls = _walk_only(_FIXED[name])
        wcmp = [r.cmp() for rows, _ in calls for r in rows]
        assert _tape_hash(wcmp) == _PINNED_HASH[name], "walk drifted from pin"
        recs, _, _ = _run_session(_FIXED[name])
        ecmp = [row for rec in recs for row in _cmp_rows(rec["rows"])]
        assert _tape_hash(ecmp) == _PINNED_HASH[name]


# Non-tape leaves at the end of each fixed session, transcribed from the
# audit walk (hand-checked against the rule): per side {oid: (price_int,
# visible ticks)} and priority order, hidden ticks per iceberg, n_active.
def _pv(eur):
    return _P(eur) + _PRICE_OFF


_PINNED_FINAL_VIEW = {
    "gestvd2": {
        _BUY: ({58: (_pv(112.34), 15), 90210: (_pv(112.00), 119),
                306: (_pv(110.00), 100)}, [58, 90210, 306]),
        _SELL: ({777001: (_pv(113.75), 30)}, [777001]),
        "hidden": {777001: 700}, "n_active": 1},
    "part_used": {_BUY: ({}, []), _SELL: ({}, []), "hidden": {}, "n_active": 0},
    "stop_sell": {
        _BUY: ({8282: (_pv(60.00), 30)}, [8282]),
        _SELL: ({64001: (_pv(61.00), 20)}, [64001]),
        "hidden": {64001: 20}, "n_active": 1},
    "stop_buy": {
        _BUY: ({10046: (_pv(39.00), 20)}, [10046]),
        _SELL: ({2828: (_pv(40.00), 30)}, [2828]),
        "hidden": {10046: 20}, "n_active": 1},
    "std_split": {
        _BUY: ({123456: (_pv(51.00), 40)}, [123456]),
        _SELL: ({}, []), "hidden": {123456: 40}, "n_active": 1},
    "std_display": {
        _BUY: ({72: (_pv(50.00), 30)}, [72]),
        _SELL: ({}, []), "hidden": {72: 80}, "n_active": 1},
    "resting_ice_counter": {
        _BUY: ({}, []),
        _SELL: ({999: (_pv(69.00), 70), 27: (_pv(70.60), 3),
                 314159: (_pv(70.87), 15)}, [999, 27, 314159]),
        "hidden": {314159: 15}, "n_active": 1},
}


class TestPinnedNonTapeLeaves:

    @pytest.mark.parametrize("name", sorted(_PINNED_FINAL_VIEW))
    def test_final_book_leaves_pinned_from_walk(self, name):
        w, _ = _walk_only(_FIXED[name])
        assert w.view() == _PINNED_FINAL_VIEW[name], "walk drifted from pin"
        recs, _, _ = _run_session(_FIXED[name])
        assert recs[-1]["eview"] == _PINNED_FINAL_VIEW[name]

    def test_gestvd2_remainder_after_entry(self):
        """Remainder after the sweep: 5.0 @ 113.00 shown, 75.0 hidden."""
        recs, _, _ = _run_session(_FIXED["gestvd2"][:5])
        v = recs[-1]["eview"]
        assert v[_SELL][0] == {777001: (_pv(113.00), 50)}
        assert v["hidden"] == {777001: 750}
        assert v["n_active"] == 1

    def test_part_used_remainder_after_entry(self):
        """M = 3.0, P = 2.0: k_e = 1, u = 1.0, shows 1.0 @ 50.50, hides 6.0."""
        recs, _, _ = _run_session(_FIXED["part_used"][:2])
        v = recs[-1]["eview"]
        assert v[_BUY][0] == {3333: (_pv(50.50), 10)}
        assert v["hidden"] == {3333: 60}


# ---------------------------------------------------------------------------
# FOK after entry: the pre-check counts the remainder from instance k_e
# ---------------------------------------------------------------------------

class TestFokAfterEntry:
    """part_used remainder: 1.0 @ 50.50, then 2.0 @ 50.00, 2.0 @ 49.50, ...
    A SELL FOK at 49.50 can fill 5.0 under the ladder rule.  A restart at
    instance 0 (2.0 @ 51.00, 2.0 @ 50.50, 2.0 @ 50.00, 1.0 @ 49.50) would
    count 7.0; 6.0 lies strictly between."""

    @pytest.mark.parametrize("qty_mw,fills", [(5.0, True), (6.0, False),
                                              (5.1, False), (4.3, True)])
    def test_fok_projection_counts_from_rest_instance(self, qty_mw, fills):
        fok = O(_SELL, _P(49.50), _Q(qty_mw), 6006, 9, _FOK)
        recs, _, _ = _run_session(_FIXED["part_used"][:2] + [fok])
        rec = recs[-1]
        assert (rec["wfilled"] > 0) == fills
        _assert_call(rec, ctx=f"[FOK {qty_mw}]")


# ---------------------------------------------------------------------------
# Randomized sessions
# ---------------------------------------------------------------------------

def _rand_oids(rng, n):
    out = set()
    while len(out) < n:
        nd = int(rng.integers(2, 7))
        out.add(int(rng.integers(10 ** (nd - 1), 10 ** nd)))
    out = sorted(out)
    rng.shuffle(out)
    return out


def _logu(rng, lo, hi):
    return int(round(float(np.exp(rng.uniform(np.log(lo), np.log(hi))))))


def _rand_d(rng, side, p_zero):
    """PPD in cents, |d| in 1..300, sign per §13.6 (BUY down, SELL up)."""
    if rng.uniform() < p_zero:
        return 0
    mag = max(1, min(300, _logu(rng, 1, 300)))
    return -mag if side == _BUY else mag


def _random_session(seed, with_counter_ice=None, d_zero=None):
    """A quiescent book, one crossing incoming iceberg, 1-2 later orders.

    Peaks 0.1-50 MW, OIDs of 2-6 digits, off-round cent prices, both sides,
    resting NON and resting icebergs (standard and ICEwPI) as counters,
    same-side resting liquidity behind the best counter."""
    rng = np.random.default_rng(seed)
    side = int(rng.integers(0, 2))
    opp = 1 - side
    dirn = 1 if side == _BUY else -1          # +: more aggressive for `side`
    P = max(1, min(500, _logu(rng, 1, 500)))
    n_inst = int(rng.integers(1, 11))
    Q = max(P, min(9990, P * n_inst + int(rng.integers(0, P))))
    if d_zero is None:
        d = _rand_d(rng, side, 0.35)
    else:
        d = 0 if d_zero else _rand_d(rng, side, 0.0)
    b = int(rng.integers(2000, 20000))
    span = max(60, abs(d) * n_inst)
    oids = _rand_oids(rng, 20)
    agents = [int(a) for a in rng.choice(np.arange(1, 30000), 20, replace=False)]
    k = iter(range(20))

    def nxt():
        i = next(k)
        return oids[i], agents[i]

    setup, counter_prices = [], []
    for _ in range(int(rng.integers(0, 7))):
        if counter_prices and rng.uniform() < 0.25:
            pr = int(rng.choice(counter_prices))      # time-priority ties
        else:
            pr = b - dirn * int(rng.integers(-40, span + 1))
        counter_prices.append(pr)
        q = max(1, min(9990, int(round(P * rng.uniform(0.1, 3.0)))))
        oid, ag = nxt()
        setup.append(O(opp, pr, q, oid, ag))
    if with_counter_ice is None:
        n_ci = int(rng.integers(0, 3))
    else:
        n_ci = int(rng.integers(1, 3)) if with_counter_ice else 0
    for _ in range(n_ci):
        pr = b - dirn * int(rng.integers(-40, span + 1))
        counter_prices.append(pr)
        pk = max(1, min(500, int(round(P * rng.uniform(0.2, 2.0)))))
        tot = min(9990, pk * int(rng.integers(1, 6)) + int(rng.integers(0, pk)))
        oid, ag = nxt()
        setup.append(O(opp, pr, max(tot, pk), oid, ag, _ICE, pk,
                       _rand_d(rng, opp, 0.5)))
    # Same-side resting liquidity, strictly behind the best counter.
    if counter_prices:
        edge = min(counter_prices) if side == _BUY else max(counter_prices)
    else:
        edge = b + dirn * 50
    for _ in range(int(rng.integers(0, 3))):
        pr = edge - dirn * int(rng.integers(1, 400))
        oid, ag = nxt()
        setup.append(O(side, pr, max(1, int(round(P * rng.uniform(0.2, 2)))),
                       oid, ag))
    if rng.uniform() < 0.4:
        pr = edge - dirn * int(rng.integers(1, 400))
        pk = max(1, min(500, int(round(P * rng.uniform(0.3, 1.5)))))
        oid, ag = nxt()
        setup.append(O(side, pr, pk * int(rng.integers(1, 4)), oid, ag, _ICE,
                       pk, _rand_d(rng, side, 0.5)))
    rng.shuffle(setup)
    oid, ag = nxt()
    incoming = O(side, b, Q, oid, ag, _ICE, P, d)

    # Later orders on the opposite side (they print the remainder's ladder).
    w = _Walk()
    for o in setup + [incoming]:
        w.submit(o)
    later = []
    for _ in range(int(rng.integers(1, 3))):
        lim = b - dirn * int(rng.integers(0, span + 400))
        kind = str(rng.choice(["NON", "NON", "FOK", "ICE", "IOC"]))
        oid, ag = nxt()
        if kind == "FOK":
            m = w.matchable(opp, lim)
            q = m + int(rng.integers(0, 2)) if m > 0 else P
            o = O(opp, lim, max(1, min(9990, q)), oid, ag, _FOK)
        elif kind == "ICE":
            pk = max(1, min(500, int(round(P * rng.uniform(0.3, 2.0)))))
            q = max(pk, min(9990, pk * int(rng.integers(1, 6))))
            o = O(opp, lim, q, oid, ag, _ICE, pk, _rand_d(rng, opp, 0.4))
        else:
            q = max(1, min(9990, int(round(Q * rng.uniform(0.2, 1.2)))))
            o = O(opp, lim, q, oid, ag, _IOC if kind == "IOC" else _NON)
        w.submit(o)
        later.append(o)
    return setup, incoming, later


_FUZZ_SEEDS = list(range(1000, 1060))


def _own_rows_properties(rows, o: O):
    """Rule-implied tape properties of an incoming-ICE call (engine only):
    every own row <= P, every multiple jP < M is a cumulative boundary."""
    own = [r for r in rows if int(r[_TAGRS]) == o.oid]
    cum, bounds = 0, {0}
    for r in own:
        q = abs(int(r[_TQ]))
        assert 0 < q <= o.peak, f"row {q} exceeds peak {o.peak}"
        cum += q
        bounds.add(cum)
    for j in range(1, cum // o.peak + 1):
        if j * o.peak < cum:
            assert j * o.peak in bounds, f"boundary {j * o.peak} inside a row"


@pytest.mark.slow
class TestRandomSessionsJitted:
    """slow: 120 randomized sessions, each a handful of jitted submits."""

    @pytest.mark.parametrize("seed", _FUZZ_SEEDS)
    def test_engine_matches_walk(self, seed):
        setup, inc, later = _random_session(seed)
        recs, _, _ = _run_session(setup + [inc] + later)
        for i, rec in enumerate(recs):
            _assert_call(rec, ctx=f"[seed {seed} call {i}]")
            if rec["order"].er == _ICE:
                _own_rows_properties(rec["rows"], rec["order"])

    @pytest.mark.parametrize("seed", _FUZZ_SEEDS[:30])
    def test_resting_iceberg_counters(self, seed):
        setup, inc, later = _random_session(seed + 5000, with_counter_ice=True)
        recs, _, _ = _run_session(setup + [inc] + later)
        for i, rec in enumerate(recs):
            _assert_call(rec, ctx=f"[seed {seed + 5000} call {i}]")

    @pytest.mark.parametrize("seed", _FUZZ_SEEDS[:30])
    def test_merge_back_reproduces_unsplit_walk_standard_ice(self, seed):
        """d = 0: merging each call's rows on every column except Q over
        contiguous runs gives the walk with one row per resting entry."""
        setup, inc, later = _random_session(seed + 9000, d_zero=True)
        recs, _, _ = _run_session(setup + [inc] + later)
        for i, rec in enumerate(recs):
            o = rec["order"]
            if o.er == _ICE and o.d != 0:
                continue
            merged = _merge_engine_rows(rec["rows"])
            unsplit = [r.cmp() for r in _unsplit(rec["wrows"])]
            assert merged == unsplit, f"[seed {seed + 9000} call {i}]"


_EAGER_SEEDS = [1003, 1011, 1017, 1024, 1032, 1041]


@pytest.mark.slow
class TestRandomSessionsEager:
    """slow: the incoming iceberg and later calls under jax.disable_jit()
    (about 4 s per eager submit)."""

    @pytest.mark.parametrize("seed", _EAGER_SEEDS)
    def test_eager_engine_matches_walk(self, seed):
        setup, inc, later = _random_session(seed, d_zero=(seed % 2 == 0))
        recs, _, _ = _run_session(setup + [inc] + later, eager_from=len(setup))
        for i, rec in enumerate(recs):
            _assert_call(rec, ctx=f"[eager seed {seed} call {i}]")


# ---------------------------------------------------------------------------
# Trade-ring room: all or nothing, and the saturated ring (OQ-15)
# ---------------------------------------------------------------------------

def _prefill(state, n0):
    t = np.full((_N_TRADES, 8), -1, dtype=np.int32)
    for i in range(n0):
        t[i] = [_PRICE_OFF + 1, 1, 3, 4, 1, i, 3, 4]
    lob = state.lob._replace(trades=jnp.asarray(t, dtype=state.lob.trades.dtype))
    return state._replace(lob=lob)


def _side_volume(state, side):
    """Visible + hidden ticks resting on ``side``."""
    v = _engine_view(state)
    vis = sum(q for _, q in v[side][0].values())
    hid = sum(h for oid, h in v["hidden"].items() if oid in v[side][0])
    return vis + hid


def _setup_state(orders):
    s = _BOOK.init()
    for i, o in enumerate(orders):
        s, _ = _BOOK.submit_order(s, o.to_omie(10 * (i + 1)))
    return s


class TestRingRoom:
    """std_split: 2 unsplit rows, 4 split rows (BUY ICE 20, P = 4)."""

    @pytest.mark.parametrize("n0,split", [(0, True), (96, True),
                                          (97, False), (98, False)])
    def test_split_applies_only_when_all_rows_fit(self, n0, split):
        orders = _FIXED["std_split"]
        s = _prefill(_setup_state(orders[:2]), n0)
        w = _Walk()
        for o in orders[:2]:
            w.submit(o)
        s, res, rows = _submit(s, orders[2], 30)
        wrows, M = w.submit(orders[2])
        expect = wrows if split else _unsplit(wrows)
        assert _cmp_rows(rows) == [r.cmp() for r in expect]
        # Non-tape leaves do not depend on the ring room.
        assert _engine_view(s) == w.view()
        assert abs(float(res.trade_qty) - M / 10.0) < 1e-3
        # The prefilled rows are untouched.
        t = np.asarray(s.lob.trades)
        assert (t[:n0, _TNSEC] == np.arange(n0)).all()

    @pytest.mark.parametrize("name", ["std_split", "gestvd2", "part_used",
                                      "resting_ice_counter", "stop_sell"])
    def test_saturated_ring_remainder_follows_measured_sweep(self, name):
        """OQ-15: on a full ring the engine's measured sweep under-counts.

        Assert only what the rule implies on the MEASURED sweep M', i.e. the
        quantity the engine took off the incoming iceberg (Q minus what of it
        rests): the remainder R' = Q - M' rests as instance floor(M'/P) at
        b + floor(M'/P) d, shows min(P - u', R') and hides the rest.  The
        reported trade_qty never exceeds M', M' never exceeds what left the
        counter side, and the ring stays full.  Volume conservation between
        the two sides is NOT asserted (certificate gap)."""
        orders = _FIXED[name]
        i_ice = max(i for i, o in enumerate(orders) if o.er == _ICE)
        s0 = _setup_state(orders[:i_ice])
        o = orders[i_ice]
        opp_before = _side_volume(s0, 1 - o.side)
        s, res, _ = _submit(_prefill(s0, _N_TRADES), o, 10 * (i_ice + 1))
        assert bool(res.accepted)
        v = _engine_view(s)
        assert (np.asarray(s.lob.trades)[:, _TSEC] != -1).all()
        Rp = v[o.side][0].get(o.oid, (0, 0))[1] + v["hidden"].get(o.oid, 0)
        Mp = o.qty - Rp
        assert 0 <= Mp <= o.qty
        assert int(round(float(res.trade_qty) * 10)) <= Mp
        assert Mp <= opp_before - _side_volume(s, 1 - o.side)
        if Rp == 0:
            assert o.oid not in v[o.side][0]
            return
        k_e, u = Mp // o.peak, Mp % o.peak
        v1 = min(o.peak - u, Rp)
        assert v[o.side][0][o.oid] == (o.price + k_e * o.d + _PRICE_OFF, v1)
        assert v["hidden"][o.oid] == Rp - v1


# ---------------------------------------------------------------------------
# vmap lanes equal scalar calls (mixed ICEwPI / ICE / NON incoming)
# ---------------------------------------------------------------------------

class TestVmapLanes:

    @pytest.mark.slow
    def test_vmap_lanes_equal_scalar_calls(self):
        """slow: fresh vmap compile of submit_order (all switch arms)."""
        lanes = []
        for nm in ["gestvd2", "part_used", "std_split", "resting_ice_counter",
                   "std_display"]:
            os_ = _FIXED[nm]
            i = max(i for i, o in enumerate(os_) if o.er == _ICE)
            if nm == "std_display":
                i = 2                              # the later NON print
            st = _setup_state(os_[:i])
            lanes.append((st, os_[i].to_omie(10 * (i + 1))))
        scal = [_BOOK.submit_order(st, om) for st, om in lanes]
        bs = jax.tree.map(lambda *x: jnp.stack(x), *[st for st, _ in lanes])
        bo = jax.tree.map(lambda *x: jnp.stack([jnp.asarray(v) for v in x]),
                          *[om for _, om in lanes])
        # The engine's scoped int64 sums only lower under vmap with x64 on
        # (pre-existing; the batched inputs are built as int32 above).
        with enable_x64():
            vs, vr = jax.vmap(_BOOK.submit_order)(bs, bo)
        for lane, (ss, sr) in enumerate(scal):
            got_s = jax.tree.map(lambda x: x[lane], vs)
            got_r = jax.tree.map(lambda x: x[lane], vr)
            for a, b_ in zip(jax.tree.leaves(got_s), jax.tree.leaves(ss)):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b_))
            for a, b_ in zip(got_r, sr):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b_))
