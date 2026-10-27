"""Locked behavioural-contract tests for Story OMIE-STAMP-UNIQUE-1.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-STAMP-UNIQUE-1 — stored
entry stamps are unique among same-price, same-second resting orders (NSEC
rank on each resting add)" (certification fix pass, S5 / plan D4a; card as
amended at Step 2, v5.198, S5-A1..A3).

Rule (market-spec §5.2 "single global queue, unique, sequential"; §6.1 time
priority; Matching Algorithm §4.3, §5.5; domain ruling 2026-10-06):

* Among resting entries on one side at the same price and the same stored
  second (SEC), the stored stamp (SEC, NSEC) is strictly increasing in the
  simulator's processing order ("processing order" convention; the
  same-second replay convention itself is an open question routed to S9).
* The order stays correct after a cancel frees an earlier array slot: a later
  same-second arrival that lands in the freed slot does NOT jump the queue.
* When an order rests at (P, SEC) and other live rows on its side already
  hold (P, SEC), its NSEC is ``1 + max(NSEC)`` of those rows; otherwise NSEC
  stays 0. SEC, ``ts_counter`` and every dispatch stamp are unchanged.
* Regenerated iceberg slices are stamped from ``ts_counter``, which is
  strictly above every resting SEC after every call (S5-A2). A later tape
  order arriving at the second the counter already gave a resting slice
  ranks BEHIND that slice (probe S5-P2).

Contract (card Test-first contract row, numbering preserved):
  (1)  C05 ``same_second`` fixture: before the take, residual oid 2 stores
       [2, 0] and interloper oid 3 stores [2, 1]; the take fills oid 2.
                                                          RED before the fix.
  (2)  probe A (slot reuse after a cancel): the take fills oid 2, oid 3 keeps
       166 ticks; probe B (two plain same-second NONs + decoy cancel), both
       sides: the earlier order fills first.              RED before the fix.
  (3)  three non-crossing NONs at one (price, second): NSEC 0, 1, 2 in
       submission order (both sides), plus an ICE first slice in the middle
       (site a'), and the "max + 1, not count" edge after a cancel.
                                                          RED before the fix.
  (4)  probe S5-P2: a tape NON at the counter second of a resting regenerated
       slice stores [12, 1] behind the slice [12, 0].     RED before the fix.
       Guard: ``ts_counter`` > every resting SEC after every submit of a
       regen-heavy session; no regenerated slice is re-ranked.
                                                          GREEN before and after.
  (5)  across a mixed NON/IOC/FOK/ICE/cancel session with many same-second
       arrivals, no two live rows on a side share (P, SEC, NSEC).
                                                          RED before the fix.
  (6)  Guard: sessions whose every tape second is >= the pre-submit
       ``ts_counter`` (so no collision is possible) are byte-identical to the
       pre-fix engine (book, tape, iceberg_meta, ts_counter).
                                                          GREEN before and after.
  (7)  Guard: on (1), (3), (4) the SEC cells, ``ts_counter`` and the trade
       tape equal the pre-fix engine; on (2) and (5) only the passive OID/TID
       of queue-order-corrected fills (and NSEC / which row survives) differ.
                                                          GREEN before and after.
  (8)  Guard: ``_submit_order`` compiles once.            GREEN before and after.

Every literal pinned "pre-fix" below was measured by the test-agent on the
S4 tip (``ba2e15c`` tree, CPU). Every "after the fix" value was derived by hand
from the rule above and cross-checked against an out-of-repo scratch
prototype of the card's helper (never committed).

Units: prices in int ticks (``round((eur + 9999) / 0.01)``; 50.00 EUR ->
1004900), quantities in 0.1 MW ticks. Resting-row layout (jaxob
``OrderSideFeat``): [P, Q, OID, TID, SEC, NSEC]. Trade-tape layout (jaxob
``TradesFeat``): [P, Q, PASS_OID, AGRS_OID, SEC, NSEC, PASS_TID, AGRS_TID].

Device policy: CPU default (conftest). ``submit_order`` drives the module-level
jitted ``_submit_order``, so the file-wide ``jit`` marker applies.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat, TradesFeat
from omiecid_simulator.jaxomiecid.omie_data_loader import _row_to_order_traced
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, _submit_order
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig

# Every test here runs the jitted submit path (T1.1 marker lint).
pytestmark = pytest.mark.jit


REPO_ROOT = Path(__file__).resolve().parents[2]
# The ONE fixture-path rule: the C05 fixture lives under
# notebooks/certification/fixtures/; if OMIE-CERT-HARNESS-2 moves it to
# tests/omiecid/fixtures/cert/, the second candidate is used (not a re-baseline).
_FIXTURE_CANDIDATES = (
    REPO_ROOT / "notebooks" / "certification" / "fixtures",
    REPO_ROOT / "tests" / "omiecid" / "fixtures" / "cert",
)
C05_SAME_SECOND = "C05_seed0_same_second.npz"

BUY, SELL = 0, 1
NON, IOC, FOK, ICE = 0, 1, 2, 3
CONTRACT_ID = 37
_DELIVERY_START, _DELIVERY_END, _GCT = 555, 570, 495

_P = OrderSideFeat.P.value
_Q = OrderSideFeat.Q.value
_OID = OrderSideFeat.OID.value
_SEC = OrderSideFeat.SEC.value
_NSEC = OrderSideFeat.NSEC.value

_T_P = TradesFeat.P.value
_T_Q = TradesFeat.Q.value
_T_PASS_OID = TradesFeat.PASS_OID.value
_T_AGRS_OID = TradesFeat.AGRS_OID.value
_T_PASS_TID = TradesFeat.PASS_TID.value

# Tape columns (the loader's 13-wide int32 layout).
_C_SIDE, _C_PRICE, _C_QTY, _C_OID, _C_AGENT, _C_TS, _C_EXEC, _C_PEAK = (
    1, 2, 3, 4, 5, 6, 7, 11)

_PC = OMIEProductConfig()
_MC = OMIEMarketConfig()


# ---------------------------------------------------------------------------
# Tape construction
# ---------------------------------------------------------------------------
def px(eur: float) -> int:
    """EUR/MWh -> jaxob price tick (offset encoding, market-spec §2.3)."""
    return int(round((eur + 9999.0) / 0.01))


def _row(side, price, qty_mw, oid, ts, ex=NON, peak_mw=0.0, agent=None,
         cid=CONTRACT_ID) -> np.ndarray:
    """One loader-layout row. ``price`` is EUR/MWh (float) or an int tick."""
    r = np.zeros(13, dtype=np.int32)
    r[0] = cid
    r[_C_SIDE] = side
    r[_C_PRICE] = price if isinstance(price, (int, np.integer)) else px(price)
    r[_C_QTY] = int(round(qty_mw * 10))
    r[_C_OID] = oid
    r[_C_AGENT] = oid if agent is None else agent
    r[_C_TS] = ts
    r[_C_EXEC] = ex
    r[_C_PEAK] = int(round(peak_mw * 10))
    return r


class Cancel(Tuple[int, int]):
    """A ``ContractBook.cancel_order(oid, side)`` step inside a script."""

    def __new__(cls, oid: int, side: int):
        return super().__new__(cls, (oid, side))


Step = Union[np.ndarray, Cancel]


def _new_book(cid: int = CONTRACT_ID) -> ContractBook:
    contract = Contract(contract_id=cid, delivery_start_minutes=_DELIVERY_START,
                        delivery_end_minutes=_DELIVERY_END, gct_minutes=_GCT,
                        status=ContractStatus.OPEN)
    return ContractBook(contract, _PC, _MC)


class Run:
    """Replay a script on a fresh ``book.init()`` through the public surface.

    ``states[i]`` / ``results[i]`` are the state / SubmitResult after step i
    (``results[i]`` is None for a cancel step). ``pre_counter[i]`` is
    ``ts_counter`` just before step i.
    """

    def __init__(self, script: Sequence[Step], cid: int = CONTRACT_ID):
        self.script = list(script)
        self.book = _new_book(cid)
        state = self.book.init()
        self.states, self.results, self.pre_counter = [], [], []
        for step in self.script:
            self.pre_counter.append(int(state.ts_counter))
            if isinstance(step, Cancel):
                state = self.book.cancel_order(state, step[0], step[1])
                res = None
            else:
                state, res = self.book.submit_order(
                    state, _row_to_order_traced(jnp.asarray(step), _PC))
                assert bool(res.accepted), f"row {step.tolist()} rejected"
            self.states.append(state)
            self.results.append(res)

    @property
    def final(self):
        return self.states[-1]

    def tape_ts(self) -> Dict[int, int]:
        return {int(s[_C_OID]): int(s[_C_TS]) for s in self.script
                if not isinstance(s, Cancel)}


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------
def _side_arr(state, side: int) -> np.ndarray:
    return np.asarray(state.lob.bids if side == BUY else state.lob.asks)


def _live(state, side: int) -> np.ndarray:
    a = _side_arr(state, side)
    return a[(a[:, _OID] > 0) & (a[:, _Q] > 0)]


def _row_of(state, side: int, oid: int) -> np.ndarray:
    a = _live(state, side)
    rows = a[a[:, _OID] == oid]
    assert len(rows) == 1, f"oid {oid}: expected exactly 1 live row, got {rows.tolist()}"
    return rows[0]


def _stamp(state, side: int, oid: int) -> List[int]:
    r = _row_of(state, side, oid)
    return [int(r[_SEC]), int(r[_NSEC])]


def _trades(state) -> np.ndarray:
    t = np.asarray(state.lob.trades)
    return t[t[:, _T_P] > 0]


def _new_trades(before, after) -> List[Tuple[int, int, int, int]]:
    """(P, |Q|, PASS_OID, AGRS_OID) of the tape rows written between two states."""
    n0 = len(_trades(before))
    return [(int(r[_T_P]), abs(int(r[_T_Q])), int(r[_T_PASS_OID]),
             int(r[_T_AGRS_OID])) for r in _trades(after)[n0:]]


def _h(*arrs) -> str:
    m = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str((a.dtype.str, a.shape)).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def _digest(state) -> Dict[str, str]:
    """Book, tape, iceberg side-table (+ n_active) and ts_counter hashes.

    ``LobState.key`` is excluded (only jaxob's random cancel modes read it)."""
    return {
        "bids": _h(state.lob.bids),
        "asks": _h(state.lob.asks),
        "trades": _h(state.lob.trades),
        "iceberg_meta": _h(state.iceberg_meta, np.asarray(state.n_active)),
        "ts_counter": int(state.ts_counter),
    }


def _sec_digest(state) -> str:
    """Hash of the SEC column of both sides (all slots, live or not)."""
    return _h(np.asarray(state.lob.bids)[:, _SEC], np.asarray(state.lob.asks)[:, _SEC])


def _tape_masked_digest(state) -> str:
    """Trade-tape hash with the passive OID and passive TID columns zeroed."""
    t = np.array(np.asarray(state.lob.trades))
    t[:, _T_PASS_OID] = 0
    t[:, _T_PASS_TID] = 0
    return _h(t)


def _duplicate_keys(state, side: int) -> List[Tuple[int, int, int]]:
    keys = [(int(r[_P]), int(r[_SEC]), int(r[_NSEC])) for r in _live(state, side)]
    seen, dups = set(), []
    for k in keys:
        if k in seen:
            dups.append(k)
        seen.add(k)
    return dups


def _fixture(name: str) -> Path:
    for d in _FIXTURE_CANDIDATES:
        if (d / name).is_file():
            return d / name
    raise AssertionError(f"fixture missing: {name} (looked in {_FIXTURE_CANDIDATES})")


# ---------------------------------------------------------------------------
# Scripts (derivations in the comments)
# ---------------------------------------------------------------------------
def c05_same_second_script() -> List[np.ndarray]:
    """C05 seed 0 ``same_second`` twin, verbatim from the frozen fixture:
    BUY r (oid 1, 1004519 x 27, t1) rests; SELL a (oid 2, 1004469 x 52, t2)
    fills r for 27 and its residual 25 rests at [2, 0] (NON-REMAINDER-TS-1);
    SELL interloper (oid 3, 1004469 x 166, t2) rests at the same price and
    second; BUY take (oid 4, 1004519 x 25, t4)."""
    z = np.load(_fixture(C05_SAME_SECOND), allow_pickle=False)
    tape = np.asarray(z["tape"])
    assert tape.dtype == np.int32 and tape.shape[1] == 13
    return [r for r in tape if int(r[_C_OID]) >= 1]


# Probe A (C05 notebook section 9.4 (5)), int ticks verbatim from the card:
# SELL decoy z (oid 5, 1005000 x 10, t1) takes ask slot 0; BUY r (oid 1,
# 1004519 x 27, t2) rests; SELL a (oid 2, 1004469 x 52, t3) fills r for 27 and
# rests 25 in ask slot 1 at [3, 0]; z is cancelled (slot 0 freed); SELL
# interloper (oid 3, 1004469 x 166, t3) lands in slot 0; BUY take (oid 4,
# 1004519 x 25, t4). Time priority: oid 2 (earlier arrival) fills 25 and is
# gone; oid 3 keeps 166. Pre-fix: both store [3, 0], slot 0 wins, oid 3 fills.
PROBE_A: List[Step] = [
    _row(SELL, 1005000, 1.0, 5, 1),
    _row(BUY, 1004519, 2.7, 1, 2),
    _row(SELL, 1004469, 5.2, 2, 3),
    Cancel(5, SELL),
    _row(SELL, 1004469, 16.6, 3, 3),
    _row(BUY, 1004519, 2.5, 4, 4),
]

# Probe B, SELL side: decoy SELL 61.27 x 0.4 (oid 70001, t40) in ask slot 0;
# SELL e1 58.43 x 1.3 (oid 70002, t41) in slot 1; decoy cancelled; SELL e2
# 58.43 x 1.3 (oid 70003, t41) lands in slot 0; BUY take 58.43 x 1.3
# (oid 70004, t42). Fixed: e1 [41, 0], e2 [41, 1] -> the take fills e1.
PROBE_B_SELL: List[Step] = [
    _row(SELL, 61.27, 0.4, 70001, 40),
    _row(SELL, 58.43, 1.3, 70002, 41),
    Cancel(70001, SELL),
    _row(SELL, 58.43, 1.3, 70003, 41),
    _row(BUY, 58.43, 1.3, 70004, 42),
]
# Probe B, BUY side (mirror; 2-digit / 6-digit OIDs): decoy BUY 12.06 x 0.2
# (oid 64, t900) in bid slot 0; BUY e1 13.91 x 0.8 (oid 123456, t903) in slot
# 1; decoy cancelled; BUY e2 13.91 x 0.8 (oid 65, t903) lands in slot 0; SELL
# take 13.91 x 0.8 (oid 66, t904). Fixed: the take fills e1 (oid 123456).
PROBE_B_BUY: List[Step] = [
    _row(BUY, 12.06, 0.2, 64, 900),
    _row(BUY, 13.91, 0.8, 123456, 903),
    Cancel(64, BUY),
    _row(BUY, 13.91, 0.8, 65, 903),
    _row(SELL, 13.91, 0.8, 66, 904),
]

# (3) three non-crossing NONs at one price and one second.
NO_FILL_BUY: List[Step] = [
    _row(BUY, 47.13, 1.0, 301, 7),
    _row(BUY, 47.13, 2.0, 302, 7),
    _row(BUY, 47.13, 0.5, 303, 7),
]
NO_FILL_SELL: List[Step] = [
    _row(SELL, 63.07, 0.9, 88001, 2205),
    _row(SELL, 63.07, 3.3, 51, 2205),
    _row(SELL, 63.07, 1.1, 4242, 2205),
]
# (3, site a') NON, ICE (first slice 1.0 of 4.0), NON at 44.44, second 300:
# the ICE first slice ranks 1, the later NON 2.
NO_FILL_ICE_MIDDLE: List[Step] = [
    _row(SELL, 44.44, 0.6, 9001, 300),
    _row(SELL, 44.44, 4.0, 9002, 300, ICE, 1.0),
    _row(SELL, 44.44, 0.7, 9003, 300),
]
# (3, edge) rank is max + 1, not the count of peers: three BUYs at 39.99 t55
# store 0, 1, 2; cancel the NSEC-1 row (oid 712); a fourth BUY at the same
# (price, second) stores 3 (peers hold {0, 2}; count would give 2 = collision).
NO_FILL_MAX_EDGE: List[Step] = [
    _row(BUY, 39.99, 1.0, 711, 55),
    _row(BUY, 39.99, 1.0, 712, 55),
    _row(BUY, 39.99, 1.0, 713, 55),
    Cancel(712, BUY),
    _row(BUY, 39.99, 1.0, 714, 55),
]

# (4) probe S5-P2 (card, verbatim): SELL ICE 3.0 / peak 1.0 @ 50.00 (OID 11,
# t10): first slice [10, 0], ts_counter 11. BUY NON 1.0 @ 50.00 (OID 22, t11)
# fills the slice; the regenerated slice is stamped from the counter: [12, 0].
# SELL NON 2.0 @ 50.00 (OID 33, t12) rests at the slice's (price, second):
# processing order puts it BEHIND the slice -> [12, 1]. Pre-fix: [12, 0].
PROBE_S5_P2: List[Step] = [
    _row(SELL, 50.00, 3.0, 11, 10, ICE, 1.0),
    _row(BUY, 50.00, 1.0, 22, 11),
    _row(SELL, 50.00, 2.0, 33, 12),
]


def regen_heavy_script() -> List[Step]:
    """Mixed session with many regenerations and same-second arrivals.

    Two resting icebergs per side at off-round prices; repeated small takes
    regenerate their slices (counter stamps); NON / IOC / FOK / cancel steps
    are interleaved, several in one second and several at a second the
    counter already handed to a regenerated slice."""
    s: List[Step] = [
        _row(SELL, 52.37, 6.0, 31, 100, ICE, 1.0),
        _row(SELL, 52.37, 4.5, 32, 100, ICE, 1.5),
        _row(BUY, 48.19, 5.0, 4001, 100, ICE, 1.0),
        _row(BUY, 48.19, 2.0, 4002, 100),
        _row(BUY, 52.37, 0.4, 501, 101),
        _row(BUY, 52.37, 0.6, 502, 101),
        _row(SELL, 48.19, 1.2, 503, 101, IOC),
        _row(SELL, 52.37, 0.8, 504, 103),
        _row(BUY, 52.37, 1.5, 505, 103, FOK),
        Cancel(4002, BUY),
        _row(BUY, 48.19, 0.9, 506, 104),
        _row(BUY, 52.37, 2.0, 507, 105),
        _row(SELL, 48.19, 2.5, 508, 105),
        _row(SELL, 52.37, 1.0, 509, 106),
        _row(BUY, 52.37, 1.7, 510, 107, IOC),
        _row(SELL, 48.19, 0.3, 511, 107),
    ]
    return s


def mixed_same_second_script() -> List[Step]:
    """(5) Mixed NON/IOC/FOK/ICE/cancel session, many same-second arrivals.

    Resting same-second groups use equal per-order quantities and takes are
    whole multiples of them, so a queue-order correction changes only which
    OID is passive on a row, never the row structure (needed by (7))."""
    return [
        # second 200: three SELL NON 1.0 @ 51.37 (equal size), an ICE first
        # slice at the same price/second, and two BUY NON 1.5 @ 49.83.
        _row(SELL, 51.37, 1.0, 4101, 200),
        _row(SELL, 51.37, 1.0, 4102, 200),
        _row(BUY, 49.83, 1.5, 87, 200),
        _row(SELL, 51.37, 3.0, 4103, 200, ICE, 1.0),
        _row(BUY, 49.83, 1.5, 88, 200),
        _row(SELL, 51.37, 1.0, 4104, 200),
        # second 201: free ask slot 0 (oid 4101) and bid slot 0 (oid 87), then
        # same-second interlopers land in the freed slots.
        Cancel(4101, SELL),
        Cancel(87, BUY),
        _row(SELL, 51.37, 1.0, 99001, 200),
        _row(BUY, 49.83, 1.5, 99002, 200),
        # second 202: an IOC take of one queue unit, a FOK of one bid unit,
        # another same-second NON pair.
        _row(BUY, 51.37, 1.0, 610, 202, IOC),
        _row(SELL, 49.83, 1.5, 611, 202, FOK),
        _row(BUY, 47.05, 0.7, 612, 202),
        _row(BUY, 47.05, 0.7, 613, 202),
        # second 203: a NON take of two queue units at 51.37.
        _row(BUY, 51.37, 2.0, 614, 203),
        _row(SELL, 47.05, 0.7, 615, 204),
    ]


# (6) guard sessions: every tape second >= the pre-submit ts_counter (seconds
# spaced wider than the number of regenerations), so no same-(P, SEC) pair can
# ever meet. Mixed NON/IOC/FOK/ICE/cancel, both sides.
GUARD_NO_ICE: List[Step] = [
    _row(SELL, 55.55, 2.0, 801, 10),
    _row(BUY, 53.21, 1.4, 802, 20),
    _row(BUY, 55.55, 0.5, 803, 30),
    _row(SELL, 53.21, 0.6, 804, 40, IOC),
    _row(BUY, 55.55, 2.5, 805, 50),
    Cancel(802, BUY),
    _row(SELL, 54.02, 1.0, 806, 60, FOK),
    _row(SELL, 54.02, 0.9, 807, 70),
]
GUARD_WITH_ICE: List[Step] = [
    _row(SELL, 50.17, 5.0, 901, 100, ICE, 1.0),
    _row(BUY, 49.61, 3.0, 902, 120, ICE, 0.7),
    _row(BUY, 50.17, 2.4, 903, 140),
    _row(SELL, 49.61, 1.5, 904, 160, IOC),
    _row(BUY, 50.17, 0.8, 905, 180, FOK),
    Cancel(902, BUY),
    _row(SELL, 49.90, 1.1, 906, 200),
    _row(BUY, 50.17, 3.0, 907, 220),
    _row(SELL, 51.03, 2.0, 908, 240, ICE, 0.5),
    _row(BUY, 50.88, 1.2, 909, 260, ICE, 0.4),
]
GUARD_SESSIONS = {"no_ice": GUARD_NO_ICE, "with_ice": GUARD_WITH_ICE}


# ---------------------------------------------------------------------------
# Pre-fix pins (measured on the S4 tip, CPU; see module docstring)
# ---------------------------------------------------------------------------
# sha256[:16] of an all-zero book side / iceberg side-table at the defaults.
_EMPTY_SIDE_H = "1404aa613105d249"

# (6) final-state digests of the guard sessions.
PRE_FIX_GUARD_DIGESTS: Dict[str, Dict[str, object]] = {
    "no_ice": {"bids": _EMPTY_SIDE_H, "asks": "d8b8e2eddbbd82d8",
               "trades": "c25303841820c00b",
               "iceberg_meta": "441f30e2a854cd36", "ts_counter": 71},
    "with_ice": {"bids": "455042f3e520b5b8", "asks": "fc8969fda9ed83f0",
                 "trades": "b0c6fce6a09f66b3",
                 "iceberg_meta": "57918bc8bf699431", "ts_counter": 261},
}

# (7) on (1), (3), (4): (SEC-column hash, ts_counter, trade-tape hash).
_EMPTY_TAPE_H = "48d9b7d2a0e91365"
PRE_FIX_SEC_TAPE: Dict[str, Tuple[str, int, str]] = {
    "c05_same_second": ("c2638d4c00164fa3", 6, "4e6b0f34a497fd10"),
    "no_fill_buy": ("6cfc6d67e24c7654", 8, _EMPTY_TAPE_H),
    "no_fill_sell": ("016e71b30f02aac2", 2206, _EMPTY_TAPE_H),
    "no_fill_ice_middle": ("d25cbd79672ca063", 301, _EMPTY_TAPE_H),
    "no_fill_max_edge": ("4c6b59756d73a1fb", 56, _EMPTY_TAPE_H),
    "probe_s5_p2": ("d1636bc108faa2ca", 14, "bc2c77e1d0c8ad1d"),
}

# (7) on (2) and (5): (masked-tape hash [PASS_OID, PASS_TID zeroed],
# ts_counter, per-step (trade_qty MW, trade_price EUR) or None for a cancel).
_Z = (0.0, 0.0)
PRE_FIX_MASKED: Dict[str, Tuple[str, int, list]] = {
    "probe_a": ("30663876d24ee046", 6,
                [_Z, _Z, (2.7, 46.19), None, _Z, (2.5, 45.69)]),
    "probe_b_sell": ("99005436496bbcb5", 44, [_Z, _Z, None, _Z, (1.3, 58.43)]),
    "probe_b_buy": ("618e2a959cc51a10", 906, [_Z, _Z, None, _Z, (0.8, 13.91)]),
    "mixed": ("04469c62997084ef", 208,
              [_Z, _Z, _Z, _Z, _Z, _Z, None, None, _Z, _Z, (1.0, 51.37),
               (1.5, 49.83), _Z, _Z, (2.0, 51.37), (0.7, 49.83)]),
}
# (7) on (5): the final book as a multiset of (P, Q, SEC) per side (OID and
# NSEC dropped: only which same-second row survives may change).
PRE_FIX_MIXED_BOOK_PQS = {
    BUY: [(1004605, 7, 202), (1004605, 7, 202), (1004883, 8, 200)],
    SELL: [(1005037, 10, 200), (1005037, 10, 205)],
}

# OMIE-SLICE-STAMP-TAPE-SEC-1 (E2) re-baselines (licence (10), (11)).
# (10) probe_s5_p2: hand-derived SEC digest — the pre-fix SEC columns with the
# one moved cell, OID 11's regenerated slice, at its activation second 11
# (OID 22's second) instead of the counter value 12; OID 33 keeps SEC 12.
# Computed by the test-agent from the pre-fix final state (main @ 1dc4456).
POST_E2_SEC_DIGEST: Dict[str, str] = {"probe_s5_p2": "76196a8dcddcc60a"}
# (11) mixed: OID 4103's last slice is activated by OID 614 at second 203
# (pre-fix: counter value 205); every other (P, Q, SEC) is unchanged.
POST_E2_MIXED_BOOK_PQS = {
    BUY: [(1004605, 7, 202), (1004605, 7, 202), (1004883, 8, 200)],
    SELL: [(1005037, 10, 200), (1005037, 10, 203)],
}


def _scripts() -> Dict[str, List[Step]]:
    return {
        "c05_same_second": c05_same_second_script(),
        "probe_a": PROBE_A,
        "probe_b_sell": PROBE_B_SELL,
        "probe_b_buy": PROBE_B_BUY,
        "no_fill_buy": NO_FILL_BUY,
        "no_fill_sell": NO_FILL_SELL,
        "no_fill_ice_middle": NO_FILL_ICE_MIDDLE,
        "no_fill_max_edge": NO_FILL_MAX_EDGE,
        "probe_s5_p2": PROBE_S5_P2,
        "regen_heavy": regen_heavy_script(),
        "mixed": mixed_same_second_script(),
        "guard_no_ice": GUARD_NO_ICE,
        "guard_with_ice": GUARD_WITH_ICE,
    }


def _run(name: str) -> Run:
    script = _scripts()[name]
    first = next(s for s in script if not isinstance(s, Cancel))
    return Run(script, cid=int(first[0]))


# ===========================================================================
# (1) C05 same-second twin
# ===========================================================================
class TestC05SameSecond:
    def test_fixture_is_the_documented_same_second_twin(self):
        """Precondition: the frozen tape is the card's construction (oids 1-4,
        residual and interloper both at second 2, same price)."""
        rows = c05_same_second_script()
        assert [int(r[_C_OID]) for r in rows] == [1, 2, 3, 4]
        assert [int(r[_C_TS]) for r in rows] == [1, 2, 2, 4]
        assert int(rows[1][_C_PRICE]) == int(rows[2][_C_PRICE]) == 1004469
        assert int(rows[1][_C_SIDE]) == int(rows[2][_C_SIDE]) == SELL

    def test_c05_same_second_residual_before_interloper(self):
        """Just before the take: residual oid 2 stores [2, 0], interloper oid
        3 stores [2, 1]; the take fills oid 2 (25 ticks) and oid 3 keeps 166."""
        run = _run("c05_same_second")
        before_take = run.states[2]
        assert _stamp(before_take, SELL, 2) == [2, 0]
        assert _stamp(before_take, SELL, 3) == [2, 1]
        assert _new_trades(before_take, run.final) == [(1004469, 25, 2, 4)]
        assert [int(x) for x in _row_of(run.final, SELL, 3)[[_Q, _SEC, _NSEC]]] == [166, 2, 1]
        assert len(_live(run.final, SELL)[_live(run.final, SELL)[:, _OID] == 2]) == 0


# ===========================================================================
# (2) slot reuse after a cancel: no queue jump
# ===========================================================================
class TestSlotReuseNoQueueJump:
    def test_probe_a_slot_reuse_no_queue_jump(self):
        """Probe A: the interloper lands in the decoy's freed slot 0 but ranks
        [3, 1] behind the residual [3, 0]; the take fills oid 2, oid 3 stays 166."""
        run = _run("probe_a")
        before_take = run.states[4]
        assert _stamp(before_take, SELL, 2) == [3, 0]
        assert _stamp(before_take, SELL, 3) == [3, 1]
        assert _new_trades(before_take, run.final) == [(1004469, 25, 2, 4)]
        live = _live(run.final, SELL)
        assert live[:, _OID].tolist() == [3]
        assert int(live[0][_Q]) == 166

    @pytest.mark.parametrize("name, side, e1, e2, take, price", [
        ("probe_b_sell", SELL, 70002, 70003, 70004, px(58.43)),
        ("probe_b_buy", BUY, 123456, 65, 66, px(13.91)),
    ])
    def test_probe_b_earlier_same_second_order_fills_first(
            self, name, side, e1, e2, take, price):
        """Probe B: two plain NONs, same price and second, no partial fill;
        the decoy cancel frees slot 0 for the later one. The earlier order
        (e1) is filled by the take; the later one (e2) rests with NSEC 1."""
        run = _run(name)
        before_take = run.states[3]
        sec = _stamp(before_take, side, e1)[0]
        assert _stamp(before_take, side, e1) == [sec, 0]
        assert _stamp(before_take, side, e2) == [sec, 1]
        qty = int(_row_of(before_take, side, e1)[_Q])
        assert _new_trades(before_take, run.final) == [(price, qty, e1, take)]
        assert _live(run.final, side)[:, _OID].tolist() == [e2]


# ===========================================================================
# (3) no-fill rests are ranked in submission order
# ===========================================================================
class TestNoFillRestRanked:
    @pytest.mark.parametrize("name, side, oids", [
        ("no_fill_buy", BUY, (301, 302, 303)),
        ("no_fill_sell", SELL, (88001, 51, 4242)),
    ])
    def test_no_fill_rest_ranked(self, name, side, oids):
        """Three non-crossing NONs at one (price, second): NSEC 0, 1, 2."""
        run = _run(name)
        sec = int(run.script[0][_C_TS])
        assert [_stamp(run.final, side, o) for o in oids] == [
            [sec, 0], [sec, 1], [sec, 2]]

    def test_ice_first_slice_ranked_between_nons(self):
        """Site (a'): NON, ICE first slice, NON at one (price, second) store
        NSEC 0, 1, 2; the ICE first slice keeps SEC = its entry second."""
        run = _run("no_fill_ice_middle")
        assert [_stamp(run.final, SELL, o) for o in (9001, 9002, 9003)] == [
            [300, 0], [300, 1], [300, 2]]
        assert int(_row_of(run.final, SELL, 9002)[_Q]) == 10  # peak 1.0 MW

    def test_rank_is_max_plus_one_not_peer_count(self):
        """After the NSEC-1 row is cancelled, peers hold {0, 2}: the next
        same-(price, second) arrival stores 3, never a duplicate 2."""
        run = _run("no_fill_max_edge")
        assert [_stamp(run.final, BUY, o) for o in (711, 713, 714)] == [
            [55, 0], [55, 2], [55, 3]]

    def test_distinct_price_or_second_not_ranked(self):
        """Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E1, licence (1)): the
        stamp is unique per contract, across prices and sides, so a row at
        another price or on the other side in the same second IS a peer
        (3102 -> [7, 1], 3104 -> [8, 1]); a row alone in its second keeps
        NSEC 0."""
        run = Run([
            _row(BUY, 47.13, 1.0, 3101, 7),
            _row(BUY, 47.12, 1.0, 3102, 7),   # other price, same second
            _row(BUY, 47.13, 1.0, 3103, 8),   # same price, later second
            _row(SELL, 47.21, 1.0, 3104, 8),  # other side
        ])
        assert _stamp(run.final, BUY, 3101) == [7, 0]
        assert _stamp(run.final, BUY, 3102) == [7, 1]
        assert _stamp(run.final, BUY, 3103) == [8, 0]
        assert _stamp(run.final, SELL, 3104) == [8, 1]


# ===========================================================================
# (4) a tape order at a regenerated slice's counter second ranks behind it
# ===========================================================================
class TestCounterStampedSlice:
    def test_probe_s5_p2_preconditions(self):
        """Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence (7)): the
        regenerated slice of OID 11 is stamped at its activation event, the
        second of OID 22 (11), not from the counter; the counter has still run
        ahead past second 12, at which the tape NON OID 33 arrives."""
        run = _run("probe_s5_p2")
        assert _stamp(run.states[1], SELL, 11)[0] == int(run.script[1][_C_TS]) == 11
        assert int(run.states[1].ts_counter) > int(run.script[2][_C_TS]) == 12

    def test_tape_order_ranked_behind_counter_stamped_slice(self):
        """Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence (8)): the
        slice (activated at second 11) orders before OID 33, which arrives
        later and stores [12, 0] (ordering, not a literal slice NSEC)."""
        run = _run("probe_s5_p2")
        k_slice = tuple(_stamp(run.final, SELL, 11))
        assert k_slice[0] == 11
        assert _stamp(run.final, SELL, 33) == [12, 0]
        assert k_slice < (12, 0)

    @pytest.mark.parametrize("name", ["regen_heavy", "probe_s5_p2", "mixed"])
    def test_counter_exceeds_every_resting_sec(self, name):
        """Guard (S5-A2): after every step, every live resting SEC is strictly
        below ``ts_counter``, and regenerated slices are observed (null guard).

        OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence (9)): the former "every
        regenerated slice keeps NSEC 0" assertion is deleted (slices are now
        ranked at their activation event); the SEC < ``ts_counter`` guard,
        the regenerated-row detection and the null guard are kept."""
        run = _run(name)
        tape_ts = run.tape_ts()
        regen_rows_seen = 0
        for i, st in enumerate(run.states):
            counter = int(st.ts_counter)
            for side in (BUY, SELL):
                for r in _live(st, side):
                    assert int(r[_SEC]) < counter, (
                        f"{name} step {i}: row {r.tolist()} SEC >= ts_counter {counter}")
                    if int(r[_SEC]) != tape_ts[int(r[_OID])]:
                        regen_rows_seen += 1
        assert regen_rows_seen > 0, f"{name}: no regenerated slice observed (null guard)"


# ===========================================================================
# (5) unique keys across a mixed session
# ===========================================================================
class TestUniqueKeysInvariant:
    def test_mixed_session_has_same_second_collisions(self):
        """Precondition (non-null): the session puts >= 3 resting entries on one
        (side, price, second) and lands interlopers in freed slots."""
        rows = [s for s in mixed_same_second_script() if not isinstance(s, Cancel)]
        keys = [(int(r[_C_SIDE]), int(r[_C_PRICE]), int(r[_C_TS])) for r in rows
                if int(r[_C_EXEC]) in (NON, ICE)]
        assert max(keys.count(k) for k in set(keys)) >= 3
        assert any(isinstance(s, Cancel) for s in mixed_same_second_script())

    @pytest.mark.parametrize("name", ["mixed", "regen_heavy"])
    def test_unique_keys_invariant(self, name):
        """After every step, no two live rows on a side share (P, SEC, NSEC)."""
        run = _run(name)
        for i, st in enumerate(run.states):
            for side in (BUY, SELL):
                assert _duplicate_keys(st, side) == [], (
                    f"{name} step {i} side {side}: duplicate stored keys")

    def test_mixed_session_queue_order(self):
        """Hand-derived passive OIDs: the IOC takes oid 4102 (earliest live
        51.37 ask after 4101's cancel), the FOK takes oid 88 (earliest live
        49.83 bid), the NON take 614 takes 4103's slice then 4104, and the
        last SELL fills oid 99002 (ranked behind 88 at second 200)."""
        run = _run("mixed")
        passive = [int(r[_T_PASS_OID]) for r in _trades(run.final)]
        assert passive == [4102, 88, 4103, 4104, 99002]
        # OMIE-SLICE-STAMP-TAPE-SEC-1 (E1, licence (2)): stamps are unique per
        # contract across prices and sides, so each interloper (script steps
        # 8 and 9) ranks behind every live second-200 row of EITHER side that
        # rested when it arrived (ordering, not literal NSEC values).
        def sec200(st):
            return [(int(r[_SEC]), int(r[_NSEC])) for side in (BUY, SELL)
                    for r in _live(st, side) if int(r[_SEC]) == 200]
        k1 = tuple(_stamp(run.final, SELL, 99001))
        k2 = tuple(_stamp(run.final, BUY, 99002))
        assert k1[0] == k2[0] == 200
        assert max(sec200(run.states[7])) < k1 < k2
        assert max(sec200(run.states[8])) < k2


# ===========================================================================
# (6) Guard: no collision -> byte-identical
# ===========================================================================
@pytest.mark.parametrize("name", sorted(GUARD_SESSIONS))
def test_no_collision_byte_identical(name):
    """Sessions whose every tape second is >= the pre-submit ``ts_counter``
    (asserted) leave book, tape, iceberg_meta and ts_counter byte-identical
    to the pre-fix engine."""
    run = Run(GUARD_SESSIONS[name])
    for step, pre in zip(run.script, run.pre_counter):
        if not isinstance(step, Cancel):
            assert int(step[_C_TS]) >= pre, (
                f"guard precondition broken: oid {int(step[_C_OID])} second "
                f"{int(step[_C_TS])} < pre-submit ts_counter {pre}")
    assert any(r is not None and bool(r.traded) for r in run.results)
    assert _digest(run.final) == PRE_FIX_GUARD_DIGESTS[name]


# ===========================================================================
# (7) Guard: SEC, counter and tape unchanged
# ===========================================================================
@pytest.mark.parametrize("name", sorted(PRE_FIX_SEC_TAPE))
def test_sec_counter_tape_unchanged(name):
    """On (1), (3), (4): SEC cells, ts_counter and trade-tape bytes equal the
    pre-fix engine (only NSEC cells move)."""
    run = _run(name)
    sec_h, ts, tape_h = PRE_FIX_SEC_TAPE[name]
    # OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence (10)): SEC digest only.
    sec_h = POST_E2_SEC_DIGEST.get(name, sec_h)
    assert _sec_digest(run.final) == sec_h
    assert int(run.final.ts_counter) == ts
    assert _h(run.final.lob.trades) == tape_h


@pytest.mark.parametrize("name", sorted(PRE_FIX_MASKED))
def test_queue_corrected_runs_differ_only_in_passive_oid(name):
    """On (2) and (5): with the passive OID/TID columns masked the trade tape
    is byte-identical to the pre-fix engine; ts_counter and every
    SubmitResult (qty, price) are unchanged."""
    run = _run(name)
    masked_h, ts, results = PRE_FIX_MASKED[name]
    assert _tape_masked_digest(run.final) == masked_h
    assert int(run.final.ts_counter) == ts
    got = [None if r is None else (float(r.trade_qty), float(r.trade_price))
           for r in run.results]
    assert len(got) == len(results)
    for g, e in zip(got, results):
        if e is None:
            assert g is None
        else:
            assert g == pytest.approx(e, abs=1e-3)


def test_mixed_book_differs_only_in_oid_and_nsec():
    """On (5): the final book, as a multiset of (P, Q, SEC) per side, equals
    the pre-fix engine's; only OIDs (which same-second row survives) and NSEC
    cells differ. Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence
    (11)): one regenerated slice's SEC moves 205 -> 203 (its activation
    second)."""
    run = _run("mixed")
    for side in (BUY, SELL):
        got = sorted((int(r[_P]), int(r[_Q]), int(r[_SEC]))
                     for r in _live(run.final, side))
        assert got == POST_E2_MIXED_BOOK_PQS[side]


# ===========================================================================
# (8) Guard: one compile
# ===========================================================================
@pytest.mark.slow
def test_one_compile_preserved():
    """slow: a fresh _submit_order compile + every script in this file (~5 s).

    The cache is cleared first so entries left by other modules in the same
    (xdist) worker, with other static configs, do not pollute the count."""
    if not hasattr(_submit_order, "_cache_size"):
        pytest.skip("_submit_order does not expose `_cache_size`")
    if hasattr(_submit_order, "clear_cache"):
        _submit_order.clear_cache()
    else:  # pragma: no cover - older jax
        _submit_order._clear_cache()
    for name in _scripts():
        _run(name)
    size = _submit_order._cache_size()
    assert size == 1, f"expected one _submit_order compile, got {size}"
