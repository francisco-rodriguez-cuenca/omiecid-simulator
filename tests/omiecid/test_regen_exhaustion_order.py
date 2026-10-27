"""Locked behavioural-contract tests for Story OMIE-REGEN-EXHAUSTION-ORDER-1.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-REGEN-EXHAUSTION-ORDER-1
— slices regenerated in one pass are stamped in the order their peaks were
exhausted, not in slot order" (filed v5.205, amended at Step 2, v5.206,
R1–R7). Domain confirmation (AC 8): ``docs/omiecid/sprint-status.md`` under the
story heading (CONFIRMED 2026-10-07).

Rule (market-spec v2.14 §4.2.1 Rule 2 *Re-ranking*, Rule 4 *Slice activation
timing*; Matching Algorithm §3.3, §4.3, §4.4, §5.5; Rules Annex §4.4):

* Activation is **eager**: a new iceberg slice is created, with the next
  sequential timestamp of the single global queue, at the trade that
  exhausts the previous peak.
* So among the slices regenerated in one matching pass, time priority is the
  order of the trades that exhausted their peaks. Engine slot / scan order
  has no standing.
* One aggressor trades resting peaks sequentially in book priority
  (best price first, then oldest), so the pre-pass priority of the consumed
  peaks IS the exhaustion order. The ranking is global across the pass, not
  per price level (two ICEwPI slices exhausted at different prices may land
  on one price, AC 4).
* Only the stamps move: prices, per-level quantities, totals and the final
  ``ts_counter`` are unchanged (stamps in one pass are a permutation of the
  pre-fix stamps).

Contract (card AC numbering preserved):
  (1)  Card reproducer: the ts-5 BUY fills oid 2 (main: oid 1).      RED now.
  (2)  Slot-order variation: registration order swapped -> fills oid 1;
       three-iceberg rotation (exhaustion 3, 1, 2) -> next BUY fills oid 3
       (main: oid 1).                                                RED now.
  (2b) Saturated trade tape (``nTrades`` rows pre-filled): the exhaustion
       order still holds (guards against ranking by trade-ring index, R1).
       Carried by the ICEwPI scenario (4), not (1); see the note above
       ``TestSaturatedTape``.                                       RED now.
  (3)  Large single aggressor: passive OIDs A1, B2, A2, B3
       (main: A1, B2, B3, A2).                                       RED now.
  (4)  ICEwPI slices coinciding at 50.00: next BUY fills oid 2
       (main: oid 1).                                                RED now.
  (4b) BUY-side mirror of (1); (1) with the middle aggressor IOC, FOK and
       incoming ICE.                                                 RED now.
  (5)  Guard (restated, R5): sessions in which no pass trades two or more
       iceberg peaks are byte-identical to main (book, tape, meta,
       ``ts_counter``).                                  GREEN before and after.
  (5b) Guard (R6): over a seeded regen-heavy session, after every submit
       ``ts_counter`` > every resting SEC and no two live rows on a side share
       (P, SEC, NSEC); the final ``ts_counter`` equals main's.
                                                         GREEN before and after.
  (D1) Interface: ``IcebergManager.regenerate_slices(lob, iceberg_meta,
       n_active, ts_counter, max_iterations=None, prior_lob=None,
       aggressor_side=None)``. With ``prior_lob`` given, the slots whose peak
       rested in ``prior_lob`` (on the side opposite ``aggressor_side``) and
       is gone from ``lob`` are stamped in their ``prior_lob`` priority; with
       ``prior_lob=None`` the result (and the traced program) is the pre-fix
       one.                                       RED now (except the None arm).
  (C)  ``_submit_order`` compiles once over every script here.  GREEN.

AC (6) (eval-fast golden, fori-parity ``_GOLDEN``, IOC #1689, synth digests)
is guarded by the existing locked files that pin those values; it is not
duplicated here. AC (7) is a process gate (JUnit / test-perf / memory maps).

Every "main" literal below was measured by the test-agent on the branch head
(``62d984c`` tree, CPU). Every "after the fix" value was derived by hand from
the rule above and cross-checked against the architect's out-of-repo scratch
prototype (never committed).

Units: prices in int ticks (``round((eur + 9999) / 0.01)``; 50.00 EUR ->
1004900), quantities in 0.1 MW ticks. Resting-row layout (jaxob
``OrderSideFeat``): [P, Q, OID, TID, SEC, NSEC]. Trade-tape layout (jaxob
``TradesFeat``): [P, Q, PASS_OID, AGRS_OID, SEC, NSEC, PASS_TID, AGRS_TID].
OMIE sides: BUY = 0, SELL = 1.

Device policy: CPU default (conftest). ``submit_order`` drives the
module-level jitted ``_submit_order``, so the file-wide ``jit`` marker
applies.
"""

from __future__ import annotations

import hashlib
import inspect
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat, TradesFeat
from omiecid_simulator.jaxomiecid.omie_data_loader import _row_to_order_traced
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, _submit_order
from omiecid_simulator.omiecid.iceberg_manager import IcebergManager
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig

# Every test here runs the jitted submit path (T1.1 marker lint).
pytestmark = pytest.mark.jit


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

# Tape columns (the loader's 13-wide int32 layout).
_C_SIDE, _C_PRICE, _C_QTY, _C_OID, _C_AGENT, _C_TS, _C_EXEC, _C_PEAK, _C_PPD = (
    1, 2, 3, 4, 5, 6, 7, 11, 12)

_PC = OMIEProductConfig()
_MC = OMIEMarketConfig()


# ---------------------------------------------------------------------------
# Tape construction
# ---------------------------------------------------------------------------
def px(eur: float) -> int:
    """EUR/MWh -> jaxob price tick (offset encoding, market-spec §2.3)."""
    return int(round((eur + 9999.0) / 0.01))


def _row(side, price, qty_mw, oid, ts, ex=NON, peak_mw=0.0, agent=None,
         ppd_eur=0.0) -> np.ndarray:
    """One loader-layout row. ``price`` is EUR/MWh."""
    r = np.zeros(13, dtype=np.int32)
    r[0] = CONTRACT_ID
    r[_C_SIDE] = side
    r[_C_PRICE] = px(price)
    r[_C_QTY] = int(round(qty_mw * 10))
    r[_C_OID] = oid
    r[_C_AGENT] = oid if agent is None else agent
    r[_C_TS] = ts
    r[_C_EXEC] = ex
    r[_C_PEAK] = int(round(peak_mw * 10))
    r[_C_PPD] = int(round(ppd_eur * 100))
    return r


def _ice(side, price, oid, ts, agent, total=10.0, peak=1.0, ppd_eur=0.0):
    return _row(side, price, total, oid, ts, ex=ICE, peak_mw=peak, agent=agent,
                ppd_eur=ppd_eur)


def _new_book() -> ContractBook:
    contract = Contract(contract_id=CONTRACT_ID,
                        delivery_start_minutes=_DELIVERY_START,
                        delivery_end_minutes=_DELIVERY_END, gct_minutes=_GCT,
                        status=ContractStatus.OPEN)
    return ContractBook(contract, _PC, _MC)


class Run:
    """Replay a script on a fresh ``book.init()`` through ``submit_order``.

    ``init_fn`` (optional) maps the initial state to the state the script
    starts from (used for the saturated-tape variant). ``states[i]`` /
    ``results[i]`` are the state / SubmitResult after step i;
    ``pre_states[i]`` is the state just before step i.
    """

    def __init__(self, script: Sequence[np.ndarray],
                 init_fn: Optional[Callable] = None):
        self.script = list(script)
        self.book = _new_book()
        state = self.book.init()
        if init_fn is not None:
            state = init_fn(state)
        self.pre_states, self.states, self.results = [], [], []
        for step in self.script:
            self.pre_states.append(state)
            state, res = self.book.submit_order(
                state, _row_to_order_traced(jnp.asarray(step), _PC))
            assert bool(res.accepted), f"row {step.tolist()} rejected"
            self.states.append(state)
            self.results.append(res)

    @property
    def final(self):
        return self.states[-1]


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


def _sec(state, side: int, oid: int) -> int:
    return int(_row_of(state, side, oid)[_SEC])


def _key(state, side: int, oid: int) -> Tuple[int, int]:
    """Stored entry stamp ``(SEC, NSEC)`` (OMIE-SLICE-STAMP-TAPE-SEC-1: the
    licensed node IDs compare the pair; ``_sec`` itself is unchanged)."""
    r = _row_of(state, side, oid)
    return int(r[_SEC]), int(r[_NSEC])


def _trades(state) -> np.ndarray:
    t = np.asarray(state.lob.trades)
    return t[t[:, _T_P] > 0]


def _passive(run: Run, i: int) -> List[int]:
    """Passive OIDs of the tape rows written by step ``i`` (unsaturated tape)."""
    n0 = len(_trades(run.pre_states[i]))
    return [int(r[_T_PASS_OID]) for r in _trades(run.states[i])[n0:]]


def _h(*arrs) -> str:
    m = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str((a.dtype.str, a.shape)).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def _digest(state) -> Dict[str, object]:
    """Book, tape, iceberg side-table (+ n_active) and ts_counter."""
    return {
        "bids": _h(state.lob.bids),
        "asks": _h(state.lob.asks),
        "trades": _h(state.lob.trades),
        "iceberg_meta": _h(state.iceberg_meta, np.asarray(state.n_active)),
        "ts_counter": int(state.ts_counter),
    }


def _duplicate_keys(state, side: int) -> List[Tuple[int, int, int]]:
    keys = [(int(r[_P]), int(r[_SEC]), int(r[_NSEC])) for r in _live(state, side)]
    return [k for k, c in Counter(keys).items() if c > 1]


def _saturate_tape(state):
    """Pre-fill every row of ``lob.trades`` with a distinct dummy trade.

    Once the tape is full jaxob writes every new trade into the LAST row, so
    trade-ring indices carry no order (card R1). The dummy rows use a price
    and OIDs that no script here uses."""
    t = np.asarray(state.lob.trades).copy()
    n = t.shape[0]
    dummy = np.zeros_like(t)
    dummy[:, _T_P] = px(1.0)
    dummy[:, _T_Q] = 1
    dummy[:, _T_PASS_OID] = 900_000 + np.arange(n)
    dummy[:, _T_AGRS_OID] = 950_000 + np.arange(n)
    lob = state.lob._replace(trades=jnp.asarray(dummy, dtype=state.lob.trades.dtype))
    return state._replace(lob=lob)


# ---------------------------------------------------------------------------
# Scripts (card AC numbering)
# ---------------------------------------------------------------------------
def reproducer(side_ice: int = SELL, middle_ex: int = NON) -> List[np.ndarray]:
    """AC (1): B (oid 1, agent 2, ts 1) and A (oid 2, agent 3, ts 2), both
    10.0/1.0 @ 50.00 on ``side_ice``; a 1.0 take (ts 3) consumes B1 so B2
    rests behind A1; a 2.0 take (ts 4) trades A1 then B2 in ONE pass; the
    1.0 take at ts 5 must fill A's new slice (oid 2)."""
    agg = 1 - side_ice
    peak = 2.0 if middle_ex == ICE else 0.0
    return [
        _ice(side_ice, 50.0, 1, 1, agent=2),
        _ice(side_ice, 50.0, 2, 2, agent=3),
        _row(agg, 50.0, 1.0, 11, 3),
        _row(agg, 50.0, 2.0, 12, 4, ex=middle_ex, peak_mw=peak),
        _row(agg, 50.0, 1.0, 13, 5),
    ]


def reproducer_swapped() -> List[np.ndarray]:
    """AC (2): as (1) with the registration (slot) order swapped: oid 2
    registers first (ts 1), oid 1 second (ts 2). The ts-3 take consumes
    oid 2's peak, the ts-4 pass trades oid 1 then oid 2, so oid 1's new
    slice is older and the ts-5 take fills oid 1."""
    return [
        _ice(SELL, 50.0, 2, 1, agent=3),
        _ice(SELL, 50.0, 1, 2, agent=2),
        _row(BUY, 50.0, 1.0, 11, 3),
        _row(BUY, 50.0, 2.0, 12, 4),
        _row(BUY, 50.0, 1.0, 13, 5),
    ]


def three_rotation() -> List[np.ndarray]:
    """AC (2): SELL ICE oids 1, 2, 3 (ts 1-3); BUY 1.0 (ts 4) and BUY 1.0
    (ts 5) rotate the queue to 3, 1, 2; BUY 3.0 (ts 6) trades 3, 1, 2 in one
    pass; the next BUY 1.0 (ts 7) fills oid 3."""
    return [
        _ice(SELL, 50.0, 1, 1, agent=2),
        _ice(SELL, 50.0, 2, 2, agent=3),
        _ice(SELL, 50.0, 3, 3, agent=4),
        _row(BUY, 50.0, 1.0, 11, 4),
        _row(BUY, 50.0, 1.0, 12, 5),
        _row(BUY, 50.0, 3.0, 13, 6),
        _row(BUY, 50.0, 1.0, 14, 7),
    ]


def single_aggressor() -> List[np.ndarray]:
    """AC (3): the (1) set-up, then one BUY 4.0 (ts 4) that trades four
    slices: A1, B2, then (eager) A2, B3."""
    return [
        _ice(SELL, 50.0, 1, 1, agent=2),
        _ice(SELL, 50.0, 2, 2, agent=3),
        _row(BUY, 50.0, 1.0, 11, 3),
        _row(BUY, 50.0, 4.0, 12, 4),
    ]


def icewpi_coincide() -> List[np.ndarray]:
    """AC (4): SELL ICE A 10.0/1.0 @ 49.00 PPD +1.00 (oid 1, ts 1) and SELL
    ICE B 10.0/1.0 @ 48.00 PPD +2.00 (oid 2, ts 2). BUY 2.0 @ 49.00 (ts 4)
    takes B1 (48.00) then A1 (49.00) in one pass; both next slices land at
    50.00, B's first. BUY 1.0 @ 50.00 (ts 5) fills oid 2."""
    return [
        _ice(SELL, 49.0, 1, 1, agent=2, ppd_eur=1.0),
        _ice(SELL, 48.0, 2, 2, agent=3, ppd_eur=2.0),
        _row(BUY, 49.0, 2.0, 12, 4),
        _row(BUY, 50.0, 1.0, 13, 5),
    ]


# ---------------------------------------------------------------------------
# Pinned main values (measured on the branch head, CPU)
# ---------------------------------------------------------------------------
# AC (1): after the ts-4 pass, main stamps B3 (oid 1) = 6 and A2 (oid 2) = 7.
# The fixed engine permutes these two stamps: A2 = 6, B3 = 7.
REPRO_MAIN_SECS = {1: 6, 2: 7}
REPRO_EAGER_SECS = {1: 7, 2: 6}
# AC (1): final ``ts_counter`` (identical before and after the fix).
REPRO_FINAL_TS_COUNTER = 11


# ===========================================================================
# (1) the card reproducer
# ===========================================================================
class TestReproducer:
    def test_preconditions_b2_rests_behind_a1(self):
        """Non-null set-up: after the ts-3 take, B's second slice rests
        behind A's first slice at 50.00, so the ts-4 pass trades A1 then B2."""
        run = Run(reproducer())
        assert _passive(run, 2) == [1]
        assert _sec(run.states[2], SELL, 2) < _sec(run.states[2], SELL, 1)
        assert _passive(run, 3) == [2, 1]

    def test_regenerated_stamps_follow_exhaustion_order(self):
        """After the ts-4 pass A2 (oid 2) is stamped before B3 (oid 1).

        Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence (13)): both
        slices are activated in the ts-4 matching process, so both store SEC
        4 (the activation event), and the exhaustion order is the
        ``(SEC, NSEC)`` order (was: the counter stamps
        ``REPRO_EAGER_SECS``)."""
        run = Run(reproducer())
        st = run.states[3]
        got = {oid: _key(st, SELL, oid) for oid in (1, 2)}
        assert got[1][0] == got[2][0] == int(run.script[3][_C_TS]) == 4, got
        assert got[2] < got[1], (
            f"expected oid 2 before oid 1 by (SEC, NSEC), got {got} "
            f"(main, slot order: {REPRO_MAIN_SECS})")
        assert int(_row_of(st, SELL, 1)[_P]) == int(_row_of(st, SELL, 2)[_P]) == px(50.0)

    def test_ts5_take_fills_oid2(self):
        """The ts-5 BUY 1.0 fills oid 2 (agent 3), not oid 1 (main)."""
        run = Run(reproducer())
        assert _passive(run, 4) == [2]

    def test_prices_quantities_and_counter_unchanged(self):
        """Prices, per-level quantities and the final ``ts_counter`` are the
        same as main's: (P, Q) multiset per side and every SubmitResult."""
        run = Run(reproducer())
        asks = sorted((int(r[_P]), int(r[_Q])) for r in _live(run.final, SELL))
        assert asks == [(px(50.0), 10), (px(50.0), 10)]
        assert _live(run.final, BUY).shape[0] == 0
        assert [float(r.trade_qty) for r in run.results] == pytest.approx(
            [0.0, 0.0, 1.0, 2.0, 1.0])
        assert int(run.final.ts_counter) == REPRO_FINAL_TS_COUNTER


# ===========================================================================
# (2) slot-order variation
# ===========================================================================
class TestSlotOrderVariation:
    def test_swapped_registration_fills_oid1(self):
        """Registration order swapped: oid 1's peak is exhausted first in the
        ts-4 pass, so the ts-5 take fills oid 1 (main fills oid 2)."""
        run = Run(reproducer_swapped())
        assert _passive(run, 2) == [2]
        assert _passive(run, 3) == [1, 2]
        # OMIE-SLICE-STAMP-TAPE-SEC-1 (licence (14)): compared as (SEC, NSEC).
        assert _key(run.states[3], SELL, 1) < _key(run.states[3], SELL, 2)
        assert _passive(run, 4) == [1]

    def test_three_iceberg_rotation_preconditions(self):
        """Non-null set-up: the ts-6 pass trades oids 3, 1, 2 in that order."""
        run = Run(three_rotation())
        assert _passive(run, 3) == [1]
        assert _passive(run, 4) == [2]
        assert _passive(run, 5) == [3, 1, 2]

    def test_three_iceberg_rotation_next_take_fills_oid3(self):
        """Exhaustion order 3, 1, 2 is a rotation of slot order 1, 2, 3: the
        new slices are stamped 3 < 1 < 2 and the next BUY fills oid 3
        (main: oid 1)."""
        run = Run(three_rotation())
        st = run.states[5]
        # OMIE-SLICE-STAMP-TAPE-SEC-1 (licence (15)): compared as (SEC, NSEC).
        s1, s2, s3 = (_key(st, SELL, o) for o in (1, 2, 3))
        assert s3 < s1 < s2, f"stamps oid1={s1} oid2={s2} oid3={s3}"
        assert _passive(run, 6) == [3]


# ===========================================================================
# (2b) saturated trade tape
# ===========================================================================
#
# Carrier choice. On a saturated tape every trade of one jaxob dispatch is
# written to the LAST row, so the whole-row ``extract_fills`` sees one fill
# per dispatch (the documented write-side limitation pinned in
# ``test_trade_extraction_ring_buffer.py::TestSaturatedBufferRegression``).
# A same-price multi-peak pass is therefore under-counted and the aggressor
# loops into the regenerated slices (over-fill), which confounds (1) run
# literally on a saturated tape: measured on the prototype, the ts-4 BUY then
# trades a third slice and the ts-5 take fills oid 1. The ICEwPI scenario (4)
# avoids the confound: its two regenerated slices land at 50.00, beyond the
# ts-4 aggressor's 49.00 limit, so the extra loop iteration trades nothing.
# It still exhausts two peaks in one dispatch, so ring indices carry no
# order and ranking by trade-ring index fails it.
class TestSaturatedTape:
    def test_tape_is_saturated_throughout(self):
        """Precondition (R1): every tape row is occupied before and after
        every step, so new trades land in the last row only."""
        run = Run(icewpi_coincide(), init_fn=_saturate_tape)
        n = np.asarray(run.final.lob.trades).shape[0]
        for st in [run.pre_states[0], *run.states]:
            t = np.asarray(st.lob.trades)
            assert (t[:, _T_P] > 0).sum() == n

    def test_icewpi_on_saturated_tape(self):
        """(4) with ``lob.trades`` pre-filled to ``nTrades`` rows: B's slice
        (peak exhausted first) is stamped before A's at 50.00, and the ts-5
        take (written to the last tape row) fills oid 2."""
        run = Run(icewpi_coincide(), init_fn=_saturate_tape)
        st = run.states[2]
        assert int(_row_of(st, SELL, 1)[_P]) == int(_row_of(st, SELL, 2)[_P]) == px(50.0)
        # OMIE-SLICE-STAMP-TAPE-SEC-1 (licence (16)): compared as (SEC, NSEC).
        assert _key(st, SELL, 2) < _key(st, SELL, 1)
        last = np.asarray(run.final.lob.trades)[-1]
        assert int(last[_T_AGRS_OID]) == 13
        assert int(last[_T_PASS_OID]) == 2

    def test_saturated_stamps_equal_unsaturated_stamps(self):
        """The tape's fill state does not change the regenerated stamps:
        after the two-peak pass both slices carry the same SEC as on an
        empty tape. GREEN before and after."""
        a = Run(icewpi_coincide(), init_fn=_saturate_tape).states[2]
        b = Run(icewpi_coincide()).states[2]
        for oid in (1, 2):
            assert _sec(a, SELL, oid) == _sec(b, SELL, oid)

    @pytest.mark.skip(reason=(
        "card AC (2b) literal: (1) re-run on a saturated tape. Confounded by "
        "the pre-existing saturated-tape write-side limitation (one fill "
        "counted per dispatch -> the ts-4 BUY over-fills into a regenerated "
        "slice); on the prototype the ts-5 take then fills oid 1. Carried by "
        "test_icewpi_on_saturated_tape instead; routed to the orchestrator."))
    def test_reproducer_on_saturated_tape_literal(self):
        run = Run(reproducer(), init_fn=_saturate_tape)
        last = np.asarray(run.final.lob.trades)[-1]
        assert int(last[_T_PASS_OID]) == 2


# ===========================================================================
# (3) large single aggressor
# ===========================================================================
def test_single_aggressor_trades_a1_b2_a2_b3():
    """One BUY 4.0 trades A1, B2, A2, B3: A2 is created at A1's exhaustion,
    before B3 (main: A1, B2, B3, A2)."""
    run = Run(single_aggressor())
    assert _passive(run, 2) == [1]
    assert _passive(run, 3) == [2, 1, 2, 1]
    assert float(run.results[3].trade_qty) == pytest.approx(4.0)


# ===========================================================================
# (4) ICEwPI slices coinciding on one price (global ranking)
# ===========================================================================
class TestIcewpiCoincide:
    def test_preconditions_both_slices_land_at_50(self):
        """Non-null set-up: the ts-4 pass takes B1 @ 48.00 then A1 @ 49.00,
        and both next slices rest at 50.00."""
        run = Run(icewpi_coincide())
        assert _passive(run, 2) == [2, 1]
        st = run.states[2]
        assert int(_row_of(st, SELL, 1)[_P]) == px(50.0)
        assert int(_row_of(st, SELL, 2)[_P]) == px(50.0)

    def test_ranking_is_global_across_prices(self):
        """B's peak (48.00) was exhausted before A's (49.00): B's slice is
        older at 50.00 and the next BUY fills oid 2 (main: oid 1)."""
        run = Run(icewpi_coincide())
        st = run.states[2]
        # OMIE-SLICE-STAMP-TAPE-SEC-1 (licence (17)): compared as (SEC, NSEC).
        assert _key(st, SELL, 2) < _key(st, SELL, 1)
        assert _passive(run, 3) == [2]


# ===========================================================================
# (4b) BUY-side mirror; IOC / FOK / incoming-ICE middle aggressor
# ===========================================================================
def test_buy_side_mirror():
    """(1) with BUY icebergs and SELL takes: the ts-5 SELL fills oid 2."""
    run = Run(reproducer(side_ice=BUY))
    assert _passive(run, 2) == [1]
    assert _passive(run, 3) == [2, 1]
    # OMIE-SLICE-STAMP-TAPE-SEC-1 (licence (18)): compared as (SEC, NSEC).
    assert _key(run.states[3], BUY, 2) < _key(run.states[3], BUY, 1)
    assert _passive(run, 4) == [2]


@pytest.mark.parametrize("middle_ex,name", [(IOC, "IOC"), (FOK, "FOK"),
                                            (ICE, "ICE")])
def test_middle_aggressor_exec_types(middle_ex, name):
    """(1) with the ts-4 aggressor as IOC, FOK or incoming ICE (2.0 peak):
    it still trades A1 then B2 in one pass and the ts-5 take fills oid 2."""
    run = Run(reproducer(middle_ex=middle_ex))
    assert _passive(run, 3) == [2, 1], name
    # OMIE-SLICE-STAMP-TAPE-SEC-1 (licence (19)-(21)): compared as (SEC, NSEC).
    assert _key(run.states[3], SELL, 2) < _key(run.states[3], SELL, 1), name
    assert _passive(run, 4) == [2], name


# ===========================================================================
# (5) Guard: no pass with two or more traded iceberg peaks -> byte-identical
# ===========================================================================
def _guard_sessions() -> Dict[str, List[np.ndarray]]:
    s = {}
    # Two same-price SELL icebergs taken one peak per aggressor (alternating).
    s["one_peak_per_pass"] = (
        [_ice(SELL, 50.0, 1, 1, agent=2), _ice(SELL, 50.0, 2, 2, agent=3)]
        + [_row(BUY, 50.0, 1.0, 100 + k, 3 + k) for k in range(8)]
    )
    # A BUY iceberg plus resting NON bids; SELL takes cross one iceberg peak
    # and NON rows; partial peak fills; IOC and FOK takes.
    s["ice_plus_nons"] = [
        _ice(BUY, 50.0, 1, 1, agent=2, total=6.0, peak=2.0),
        _row(BUY, 50.0, 1.5, 2, 2),
        _row(BUY, 49.5, 3.0, 3, 3),
        _row(SELL, 50.0, 0.5, 20, 4),
        _row(SELL, 50.0, 2.0, 21, 5),
        _row(SELL, 49.5, 3.0, 22, 6, ex=IOC),
        _row(SELL, 50.0, 1.0, 23, 7, ex=FOK),
        _row(SELL, 51.0, 2.0, 24, 8),
        _row(BUY, 51.0, 1.0, 25, 9),
    ]
    # An incoming ICE aggressor sweeping a NON book (ICE sweep + heal regen)
    # against a single resting iceberg on the other side.
    s["incoming_ice_sweep"] = [
        _row(SELL, 50.0, 1.0, 1, 1),
        _ice(SELL, 50.5, 2, 2, agent=3, total=4.0, peak=1.0),
        _row(SELL, 51.0, 2.0, 3, 3),
        _ice(BUY, 51.0, 10, 4, agent=9, total=6.0, peak=2.0),
        _row(SELL, 51.0, 1.0, 4, 5),
    ]
    # ICEwPI single iceberg drifting across several takes.
    s["icewpi_single"] = (
        [_ice(SELL, 48.0, 1, 1, agent=2, total=5.0, peak=1.0, ppd_eur=1.0)]
        + [_row(BUY, 52.0, 1.0, 100 + k, 2 + k) for k in range(4)]
    )
    return s


GUARD_SESSIONS = _guard_sessions()

# Measured on main (branch head, CPU).
MAIN_GUARD_DIGESTS: Dict[str, Dict[str, object]] = {
    "ice_plus_nons": {"bids": "ba8bc72d9b2c3b7b", "asks": "263016da2f6e19ef",
                      "trades": "cf79de182d4f93c0",
                      "iceberg_meta": "bd97d55fc2aafd2e", "ts_counter": 12},
    "icewpi_single": {"bids": "1404aa613105d249", "asks": "aa51d61675ae4c85",
                      "trades": "93a75ac4901dbf6c",
                      "iceberg_meta": "80f93c3f67e18776", "ts_counter": 11},
    "incoming_ice_sweep": {"bids": "1404aa613105d249",
                           "asks": "22c109ade9331e62",
                           "trades": "1031f981749d1155",
                           "iceberg_meta": "441f30e2a854cd36",
                           "ts_counter": 12},
    "one_peak_per_pass": {"bids": "1404aa613105d249",
                          "asks": "cdc3921813aa896f",
                          "trades": "b2c594ab0219c165",
                          "iceberg_meta": "b28fe47a2daa239a",
                          "ts_counter": 20},
}

# Iceberg OIDs per guard session (for the "at most one iceberg traded per
# step" precondition).
_GUARD_ICE_OIDS = {
    name: {int(r[_C_OID]) for r in rows if int(r[_C_EXEC]) == ICE}
    for name, rows in GUARD_SESSIONS.items()
}


@pytest.mark.parametrize("name", sorted(GUARD_SESSIONS))
def test_guard_session_has_at_most_one_iceberg_per_step(name):
    """Precondition: no step trades two or more distinct resting iceberg
    OIDs (so no pass can have two ranked slots), and the session trades at
    least one iceberg peak (non-null)."""
    run = Run(GUARD_SESSIONS[name])
    ice = _GUARD_ICE_OIDS[name]
    touched = 0
    for i in range(len(run.script)):
        hit = set(_passive(run, i)) & ice
        assert len(hit) <= 1, f"{name} step {i}: icebergs {sorted(hit)} in one step"
        touched += len(hit)
    assert touched > 0


# OMIE-SLICE-STAMP-TAPE-SEC-1 re-baselines (licence (6) E1 ``ice_plus_nons``;
# (22)-(23) E2 ``icewpi_single`` and ``one_peak_per_pass``; book digests only).
# In these sessions the fixed engine moves only the stored [SEC, NSEC] cells
# of one book side (at the E1 + E2 tip: ``bids`` for ice_plus_nons, whose
# ``asks`` returns to its main value; ``asks`` for the other two). Tape,
# iceberg_meta and ts_counter stay equal to main, and so do the P / Q / OID /
# TID columns of the moved side (pinned below, measured on main @ 1dc4456).
# The moved side's digest is frozen below from the E1 + E2 tip; the test
# asserts equality to it, that it DIFFERS from main (non-null), and that every
# other field is unchanged.
_GUARD_MOVED_SIDE: Dict[str, str] = {
    "ice_plus_nons": "bids",
    "icewpi_single": "asks",
    "one_peak_per_pass": "asks",
}
MAIN_GUARD_MOVED_SIDE_PQOT: Dict[str, str] = {
    "ice_plus_nons": "66682ec95a05cde4",
    "icewpi_single": "f296481cac464add",
    "one_peak_per_pass": "76bbc24a99b81cb4",
}
# Tip digests of the moved side, frozen from the E1 + E2 tip (e604788, CPU)
# and cross-checked equal on the Step-2 architect's independent E1 + E2
# prototype (ice_plus_nons also matches its JUnit-reported ``bids`` bf91f0bb).
POST_FIX_GUARD_MOVED_SIDE_DIGEST: Dict[str, str] = {
    "ice_plus_nons": "bf91f0bbd057d50f",
    "icewpi_single": "257f59bad344335f",
    "one_peak_per_pass": "2f1b7769993c3cce",
}


@pytest.mark.parametrize("name", sorted(GUARD_SESSIONS))
def test_guard_session_byte_identical_to_main(name):
    """R5: book, tape, iceberg meta (+ n_active) and ``ts_counter`` equal the
    pre-fix engine's (except the licensed stored-stamp cells, see above)."""
    run = Run(GUARD_SESSIONS[name])
    got, want = _digest(run.final), MAIN_GUARD_DIGESTS[name]
    moved = _GUARD_MOVED_SIDE.get(name)
    if moved is None:
        assert got == want
        return
    assert {k: v for k, v in got.items() if k != moved} == {
        k: v for k, v in want.items() if k != moved}
    side_arr = np.asarray(getattr(run.final.lob, moved))
    assert _h(side_arr[:, [_P, _Q, _OID, OrderSideFeat.TID.value]]) == (
        MAIN_GUARD_MOVED_SIDE_PQOT[name]), "only SEC / NSEC cells may move"
    assert got[moved] != want[moved], (
        f"{name}: the {moved} stamp cells must move (regenerated slice "
        f"stamped at its activation event)")
    assert got[moved] == POST_FIX_GUARD_MOVED_SIDE_DIGEST[name]


# ===========================================================================
# (5b) Guard: counter and uniqueness over a seeded regen-heavy session
# ===========================================================================
def regen_heavy_session(seed: int = 20261007, n_steps: int = 20) -> List[np.ndarray]:
    """Four SELL icebergs (50.00 / 50.50) and three BUY icebergs (49.00 /
    48.50), all with 1.0 MW peaks and totals large enough never to drain,
    then random NON / IOC aggressors of 1-5 whole MW that cross several
    peaks per pass. Seconds increase by 1 per row.

    Every peak is 1.0 MW and every aggressor a whole number of MW, so each
    aggressor exhausts exactly ``qty`` peaks whatever their queue order: the
    number of regenerated slices, and hence the final ``ts_counter``, is
    order-independent, while the queue order (which iceberg trades) is not.
    """
    rng = np.random.default_rng(seed)
    rows: List[np.ndarray] = []
    ts = 1
    oid = 1
    for k in range(4):
        rows.append(_ice(SELL, float(rng.choice([50.0, 50.0, 50.5])), oid, ts,
                         agent=2 + k, total=40.0, peak=1.0))
        oid += 1
        ts += 1
    for k in range(3):
        rows.append(_ice(BUY, float(rng.choice([49.0, 49.0, 48.5])), oid, ts,
                         agent=10 + k, total=40.0, peak=1.0))
        oid += 1
        ts += 1
    oid = 1000
    for _ in range(n_steps):
        side = int(rng.integers(0, 2))
        price = 51.0 if side == BUY else 48.0
        qty = float(rng.integers(1, 6))
        ex = int(rng.choice([NON, NON, IOC]))
        rows.append(_row(side, price, qty, oid, ts, ex=ex))
        oid += 1
        ts += 1
    return rows


# Measured on main (CPU); order-independent by construction (see the docstring).
MAIN_REGEN_HEAVY_FINAL_TS_COUNTER = 107


class TestCounterAndUniqueness:
    def test_session_has_multi_peak_passes(self):
        """Non-null: at least three steps trade two or more distinct
        iceberg OIDs on one side (multi-slot ranked passes)."""
        script = regen_heavy_session()
        run = Run(script)
        ice = {int(r[_C_OID]) for r in script if int(r[_C_EXEC]) == ICE}
        multi = sum(1 for i in range(len(script))
                    if len(set(_passive(run, i)) & ice) >= 2)
        assert multi >= 3, f"only {multi} multi-iceberg steps"
        # The tape must not saturate, or _passive would undercount.
        assert len(_trades(run.final)) < np.asarray(run.final.lob.trades).shape[0]

    def test_counter_above_every_resting_sec_and_keys_unique(self):
        """After every submit: ``ts_counter`` > every live SEC on both sides,
        and no two live rows on a side share (P, SEC, NSEC)."""
        run = Run(regen_heavy_session())
        for i, st in enumerate(run.states):
            counter = int(st.ts_counter)
            for side in (BUY, SELL):
                for r in _live(st, side):
                    assert int(r[_SEC]) < counter, (
                        f"step {i}: row {r.tolist()} SEC >= ts_counter {counter}")
                assert _duplicate_keys(st, side) == [], (
                    f"step {i} side {side}: duplicate stored keys")

    def test_final_counter_equals_main(self):
        """Stamps are a permutation within each pass: the final
        ``ts_counter`` equals main's."""
        run = Run(regen_heavy_session())
        assert int(run.final.ts_counter) == MAIN_REGEN_HEAVY_FINAL_TS_COUNTER


# ===========================================================================
# (D1) IcebergManager.regenerate_slices interface
# ===========================================================================
def _consumed_pass_inputs():
    """State after step 3 of the reproducer (``prior``), and the same book
    with A1 (oid 2) and B2 (oid 1) removed, as a BUY pass that traded both
    leaves it before regeneration."""
    run = Run(reproducer())
    prior = run.states[2]
    asks = np.asarray(prior.lob.asks).copy()
    for oid in (1, 2):
        idx = np.where((asks[:, _OID] == oid) & (asks[:, _Q] > 0))[0]
        assert len(idx) == 1
        asks[idx[0], :] = -1
    lob_c = prior.lob._replace(asks=jnp.asarray(asks, dtype=prior.lob.asks.dtype))
    return run.book, prior, lob_c


def _regen_secs(lob_out) -> Dict[int, int]:
    a = np.asarray(lob_out.asks)
    live = a[(a[:, _OID] > 0) & (a[:, _Q] > 0)]
    return {int(r[_OID]): int(r[_SEC]) for r in live}


class TestRegenerateSlicesInterface:
    def test_signature_has_prior_lob_and_aggressor_side(self):
        """D1: two new keyword arguments, both defaulting to ``None``."""
        params = inspect.signature(IcebergManager.regenerate_slices).parameters
        for name in ("prior_lob", "aggressor_side"):
            assert name in params, f"regenerate_slices lacks `{name}`"
            assert params[name].default is None

    def test_without_prior_lob_stamps_in_slot_order(self):
        """D1, ``prior_lob`` omitted: the pre-fix behaviour (slot order,
        oid 1 first), stamps ``ts`` and ``ts + 1``, counter ``ts + 2``.
        GREEN before and after."""
        book, prior, lob_c = _consumed_pass_inputs()
        mgr = book._iceberg_mgr
        ts = int(prior.ts_counter)
        lob_r, _, n_r, ts_r = mgr.regenerate_slices(
            lob_c, prior.iceberg_meta, prior.n_active, prior.ts_counter)
        assert _regen_secs(lob_r) == {1: ts, 2: ts + 1}
        assert int(n_r) == 2
        assert int(ts_r) == ts + 2

    def test_prior_lob_none_is_byte_identical(self):
        """D1: ``prior_lob=None, aggressor_side=None`` returns exactly what
        the call without the keywords returns."""
        book, prior, lob_c = _consumed_pass_inputs()
        mgr = book._iceberg_mgr
        args = (lob_c, prior.iceberg_meta, prior.n_active, prior.ts_counter)
        a = mgr.regenerate_slices(*args)
        b = mgr.regenerate_slices(*args, prior_lob=None, aggressor_side=None)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_with_prior_lob_stamps_in_prior_priority(self):
        """D3/D4: with ``prior_lob`` (A1 ahead of B2) and a BUY aggressor,
        A's new slice gets the first stamp and B's the second; the stamp set,
        n_active and the returned counter equal the slot-order call's."""
        book, prior, lob_c = _consumed_pass_inputs()
        mgr = book._iceberg_mgr
        args = (lob_c, prior.iceberg_meta, prior.n_active, prior.ts_counter)
        base = mgr.regenerate_slices(*args)
        lob_r, meta_r, n_r, ts_r = mgr.regenerate_slices(
            *args, prior_lob=prior.lob, aggressor_side=jnp.int32(BUY))
        s_base, s_new = _regen_secs(base[0]), _regen_secs(lob_r)
        assert s_new == {1: s_base[2], 2: s_base[1]}
        assert int(n_r) == int(base[2])
        assert int(ts_r) == int(base[3])
        np.testing.assert_array_equal(np.asarray(meta_r), np.asarray(base[1]))

    def test_same_side_aggressor_is_not_ranked(self):
        """D3: slots on the aggressor's own side are not ranked: with
        ``aggressor_side=SELL`` the SELL icebergs keep slot-order stamps."""
        book, prior, lob_c = _consumed_pass_inputs()
        mgr = book._iceberg_mgr
        args = (lob_c, prior.iceberg_meta, prior.n_active, prior.ts_counter)
        base = mgr.regenerate_slices(*args)
        out = mgr.regenerate_slices(*args, prior_lob=prior.lob,
                                    aggressor_side=jnp.int32(SELL))
        assert _regen_secs(out[0]) == _regen_secs(base[0])


# ===========================================================================
# (C) one compile
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
    scripts = [reproducer(), reproducer_swapped(), three_rotation(),
               single_aggressor(), icewpi_coincide(), reproducer(side_ice=BUY),
               *(reproducer(middle_ex=e) for e in (IOC, FOK, ICE)),
               *GUARD_SESSIONS.values(), regen_heavy_session()]
    for s in scripts:
        Run(s)
    Run(reproducer(), init_fn=_saturate_tape)
    size = _submit_order._cache_size()
    assert size == 1, f"expected one _submit_order compile, got {size}"
