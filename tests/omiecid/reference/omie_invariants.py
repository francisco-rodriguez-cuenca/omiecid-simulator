"""The invariant battery (I1-I9, M1-M6) and the record builder that feeds it.

Story ``OMIE-CERT-INVARIANTS-1`` (``docs/omiecid/agile-plan.md``), binding decisions
V1-V16. This module is test-reference code: it is the judge every certification
scenario runs, not part of the simulator.

Two halves:

* **The record builder** (:func:`build_record`, :func:`record_from_draw`) drives
  one simulator run, one real tape row at a time, through the production decode
  (``_row_to_order_traced``) and ``ContractBook.submit_order``. It writes down what
  the **engine** did, read from engine state (plan rule 2): the book before every
  submission and after every return, every trade row the submission wrote, the
  engine's own report, and per-order fates. It never re-derives an engine quantity
  from the tape, and it never raises on engine behaviour; it raises
  :class:`RecordBuildError` only for an unusable script.
* **The checks** (``check_record`` = RB, ``check_i1_*`` ... ``check_m6_*``) each
  read a :class:`RunRecord` (or a pair) and return a :class:`CheckResult`. They
  never raise on content, so a notebook can render the whole battery.

Units: every comparison is in integer ticks, except I6's euro figures, which are
compared within a tolerance derived from the engine's float32 settlement chain
(V10). What the simulator does not expose is reported in
``CheckResult.unobserved``, never as a pass (V13).
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat, TradesFeat
from omiecid_simulator.jaxomiecid.omie_data_loader import _row_to_order_traced
from omiecid_simulator.jaxomiecid.omie_tape_synth import (
    DEFAULT_ENVELOPE,
    CapacityEnvelope,
    Draw,
    tape_capacity_bounds,
)
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_REMAINING_HIDDEN,
    COL_SIDE,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from omiecid_simulator.omiecid.settlement import compute_agent_pnl

__all__ = [
    "AcceptedOrder", "Rejection", "Fill", "BookSnapshot", "Cancellation",
    "EngineReport", "RunRecord", "RecordBuildError", "Violation", "CheckResult",
    "BatteryReport", "build_record", "record_from_draw", "compare_records",
    "check_record", "check_i1_volume_balance", "check_i2_quantity_conservation",
    "check_i3_uncrossed_on_return", "check_i4_priority", "check_i5_price_setter",
    "check_i6_settlement", "check_i7_no_manufactured_liquidity",
    "check_i8_capacity_headroom", "check_i9_non_vacuity", "check_m1_tie_permutation",
    "check_m2_monotonicity", "check_m3_ioc_non_equivalence", "check_m4_fok_dichotomy",
    "check_m5_split_aggressor", "check_m6_ownership_blindness", "CHECKS",
    "ALL_CHECKS", "STATEMENTS", "RULE3_HOP_PRICE_SETTER", "run_battery",
    "assert_check_fails", "i6_tolerance",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, IOC, FOK, ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                      int(ExecRestriction.FOK), int(ExecRestriction.ICE))

# Tape columns (the loader's 13-wide int32 layout).
_T_CONTRACT, _T_SIDE, _T_PRICE, _T_QTY, _T_OID, _T_AGENT, _T_TS = 0, 1, 2, 3, 4, 5, 6
_T_EXEC, _T_PEAK, _T_PPD = 7, 11, 12
_N_COLS = 13

# Book-side row layout (OrderSideFeat) and trade-row layout (TradesFeat).
_P, _Q, _OID, _TID = (OrderSideFeat.P.value, OrderSideFeat.Q.value,
                      OrderSideFeat.OID.value, OrderSideFeat.TID.value)
_SEC, _NSEC = OrderSideFeat.SEC.value, OrderSideFeat.NSEC.value
_TR_P, _TR_Q = TradesFeat.P.value, TradesFeat.Q.value
_TR_PASS_OID, _TR_AGRS_OID = TradesFeat.PASS_OID.value, TradesFeat.AGRS_OID.value
_TR_SEC, _TR_NSEC = TradesFeat.SEC.value, TradesFeat.NSEC.value
_TR_PASS_TID, _TR_AGRS_TID = TradesFeat.PASS_TID.value, TradesFeat.AGRS_TID.value

#: V9: the single switch a re-opened OQ-31 would flip. ``"iceberg"`` is Reading B
#: (OQ-31 RESOLVED 2026-10-02): at the §4.2.1 Rule-3 hop the trade prices at the
#: regenerated slice's own limit. The alternative is ``"resting_remainder"``.
RULE3_HOP_PRICE_SETTER = "iceberg"

#: The §5.1 cancellation causes, the only ones ``cancelled_or_expired`` may carry.
CANCEL_CAUSES = ("ioc_remainder", "fok_kill", "cancellation", "gate_expiry")

#: V13: the fixed vocabulary of unobserved outputs.
UNOBSERVED_VOCABULARY = ("rejection_reason", "queue_timestamp_correctness",
                         "independent_valuation", "matching_iterations")

#: float32 unit roundoff (V10).
_U = 2.0 ** -24

#: V2: the archive's contract (delivery 09:15-09:30, GCT 08:15), minutes from midnight.
_DELIVERY_START, _DELIVERY_END, _GCT = 555, 570, 495


# ---------------------------------------------------------------------------
# Record types (V3)
# ---------------------------------------------------------------------------
class RecordBuildError(ValueError):
    """The script handed to the builder is unusable (never raised on engine behaviour)."""


@dataclass(frozen=True)
class AcceptedOrder:
    """An accepted order's **declaration**, read from its tape row."""

    oid: int
    submission: int
    side: int
    price_int: int
    qty_int: int
    agent: int
    tape_ts: int
    exec_restriction: int
    peak_int: int
    ppd_int: int


@dataclass(frozen=True)
class Rejection:
    """A submission the engine refused. ``reason`` is ``"contract_not_open"`` or
    ``"validator"``; the validator's own reason is unobserved (P2-L1)."""

    submission: int
    oid: int
    row: Tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class Fill:
    """One trade row written by ``submission``, in integer ticks (V5)."""

    row_index: int
    submission: int
    price_int: int
    qty_int: int
    qty_signed: int
    passive_oid: int
    aggressor_oid: int
    passive_agent: int
    aggressor_agent: int
    buyer_oid: int
    seller_oid: int
    buyer_agent: int
    seller_agent: int
    sec: int
    nsec: int


@dataclass(frozen=True, eq=False)
class BookSnapshot:
    """Raw copies of the engine's book arrays at one instant.

    ``bids`` / ``asks`` are ``(book_depth, 6)`` (``OrderSideFeat`` columns),
    ``trades`` is ``(nTrades, 8)`` (``TradesFeat``), ``iceberg_meta`` is the live
    prefix ``iceberg_meta[:n_active]`` (``iceberg_manager`` columns 0-7).
    """

    bids: np.ndarray
    asks: np.ndarray
    trades: np.ndarray
    iceberg_meta: np.ndarray
    n_active: int
    ts_counter: int
    book_full: bool
    truncated: bool

    def _side_array(self, side: int) -> np.ndarray:
        return np.asarray(self.bids if int(side) == BUY else self.asks)

    def resting(self, side: int) -> Tuple[Tuple[int, int, int, int, int, int], ...]:
        """``(price, qty, oid, agent, sec, nsec)`` of rows with ``OID > 0`` and ``Q > 0``."""
        arr = self._side_array(side)
        if arr.size == 0:
            return ()
        live = arr[(arr[:, _OID] > 0) & (arr[:, _Q] > 0)]
        return tuple(
            (int(r[_P]), int(r[_Q]), int(r[_OID]), int(r[_TID]), int(r[_SEC]), int(r[_NSEC]))
            for r in live
        )

    def best_bid(self) -> Optional[int]:
        prices = [r[0] for r in self.resting(BUY)]
        return max(prices) if prices else None

    def best_ask(self) -> Optional[int]:
        prices = [r[0] for r in self.resting(SELL)]
        return min(prices) if prices else None

    def trade_rows_used(self) -> int:
        """Rows whose SEC column is not -1 (the column the engine's writer tests)."""
        t = np.asarray(self.trades)
        if t.size == 0:
            return 0
        return int(np.sum(t[:, _TR_SEC] != -1))

    def visible_qty(self, oid: int, side: int) -> int:
        return int(sum(r[1] for r in self.resting(side) if r[2] == int(oid)))

    def hidden_qty(self, oid: int) -> int:
        meta = np.asarray(self.iceberg_meta)
        if meta.size == 0:
            return 0
        rows = meta[meta[:, COL_ORDER_ID] == int(oid)]
        return int(rows[:, COL_REMAINING_HIDDEN].sum()) if len(rows) else 0


@dataclass(frozen=True)
class Cancellation:
    """One §5.1 fate of an order's quantity (V4).

    ``event_index``: for ``"cancellation"`` the index into
    ``RunRecord.cancel_events``; for ``"ioc_remainder"`` / ``"fok_kill"`` the
    order's own submission; for ``"gate_expiry"`` ``n_submissions``.
    """

    oid: int
    cause: str
    qty_int: int
    event_index: int


@dataclass(frozen=True)
class EngineReport:
    """The five ``SubmitResult`` fields of one submission, as Python scalars."""

    submission: int
    accepted: bool
    traded: bool
    trade_qty_mw: float
    trade_price_eur: float
    settlement_value_eur: float


@dataclass(frozen=True, eq=False)
class RunRecord:
    """Everything the battery reads about one run on one contract (V3)."""

    tape: np.ndarray
    n_submissions: int
    accepted: Mapping[int, AcceptedOrder]
    rejections: Tuple[Rejection, ...]
    fills: Tuple[Fill, ...]
    pre_snapshots: Tuple[BookSnapshot, ...]
    post_snapshots: Tuple[BookSnapshot, ...]
    final_snapshot: BookSnapshot
    filled: Mapping[int, int]
    still_resting: Mapping[int, int]
    hidden_remaining: Mapping[int, int]
    cancelled_or_expired: Mapping[int, Tuple[Cancellation, ...]]
    evicted: Mapping[int, int]
    cancel_events: Tuple[Tuple[int, int], ...]
    expired_at_gct: bool
    engine_reports: Tuple[EngineReport, ...]
    engine_settlements: Mapping[int, float]
    participants: frozenset
    flags: Mapping[str, bool]
    observed_capacity: Mapping[str, int]
    capacity_bounds: Mapping[str, int]
    unobserved: Tuple[str, ...]
    jit: bool
    product_config: OMIEProductConfig
    market_config: OMIEMarketConfig


# ---------------------------------------------------------------------------
# Pure readers shared by the builder and RB
# ---------------------------------------------------------------------------
def _snapshot(state) -> BookSnapshot:
    n = int(state.n_active)
    return BookSnapshot(
        bids=np.array(state.lob.bids, copy=True),
        asks=np.array(state.lob.asks, copy=True),
        trades=np.array(state.lob.trades, copy=True),
        iceberg_meta=np.array(np.asarray(state.iceberg_meta)[:n], copy=True),
        n_active=n,
        ts_counter=int(state.ts_counter),
        book_full=bool(state.book_full),
        truncated=bool(state.truncated),
    )


def _diff_rows(pre: BookSnapshot, post: BookSnapshot) -> List[int]:
    """Trade-row indices that differ between two snapshots (ascending)."""
    a, b = np.asarray(pre.trades), np.asarray(post.trades)
    if a.shape != b.shape:
        return []
    return [int(i) for i in np.nonzero(np.any(a != b, axis=1))[0]]


def _fill_from_row(row, row_index: int, submission: int) -> Fill:
    q = int(row[_TR_Q])
    pas_oid, agr_oid = int(row[_TR_PASS_OID]), int(row[_TR_AGRS_OID])
    pas_tid, agr_tid = int(row[_TR_PASS_TID]), int(row[_TR_AGRS_TID])
    if q < 0:   # the aggressor bought (JaxOrderBookArrays.py:206-215)
        buyer_oid, seller_oid, buyer_agent, seller_agent = agr_oid, pas_oid, agr_tid, pas_tid
    else:       # the passive side bought
        buyer_oid, seller_oid, buyer_agent, seller_agent = pas_oid, agr_oid, pas_tid, agr_tid
    return Fill(
        row_index=int(row_index), submission=int(submission),
        price_int=int(row[_TR_P]), qty_int=abs(q), qty_signed=q,
        passive_oid=pas_oid, aggressor_oid=agr_oid,
        passive_agent=pas_tid, aggressor_agent=agr_tid,
        buyer_oid=buyer_oid, seller_oid=seller_oid,
        buyer_agent=buyer_agent, seller_agent=seller_agent,
        sec=int(row[_TR_SEC]), nsec=int(row[_TR_NSEC]),
    )


def _fills_of_submission(pre: BookSnapshot, post: BookSnapshot, s: int) -> List[Fill]:
    after = np.asarray(post.trades)
    return [_fill_from_row(after[r], r, s) for r in _diff_rows(pre, post)
            if int(after[r, _TR_SEC]) != -1]


def _live_icebergs_on_side(snap: BookSnapshot, side: int) -> int:
    meta = np.asarray(snap.iceberg_meta)
    if meta.size == 0:
        return 0
    return int(np.sum(meta[:, COL_SIDE] == int(side)))


def _evictions(pre_snapshots, post_snapshots, fills, cancel_events,
               book_depth: int) -> Dict[int, int]:
    """V4's six-condition level-wipe signature, read from the book diff."""
    out: Dict[int, int] = {}
    n = min(len(pre_snapshots), len(post_snapshots))
    for s in range(n):
        pre, post = pre_snapshots[s], post_snapshots[s]
        named = {o for f in fills if f.submission == s for o in (f.passive_oid, f.aggressor_oid)}
        cancelled_here = {oid for (b, oid) in cancel_events if b == s}
        for side in (BUY, SELL):
            rows = pre.resting(side)
            if not rows:
                continue
            post_oids = {r[2] for r in post.resting(side)}
            prices = [r[0] for r in rows]
            worst = min(prices) if side == BUY else max(prices)
            full_enough = len(rows) + 1 + _live_icebergs_on_side(pre, side) >= int(book_depth)
            for (price, qty, oid, _a, _sec, _nsec) in rows:
                if (oid not in post_oids and oid not in named and oid not in cancelled_here
                        and price == worst and full_enough):
                    out[oid] = out.get(oid, 0) + int(qty)
    return out


def _all_snapshots(rec) -> List[BookSnapshot]:
    snaps = list(rec.pre_snapshots) + list(rec.post_snapshots)
    if rec.final_snapshot is not None:
        snaps.append(rec.final_snapshot)
    return snaps


def _observe_capacity(snaps: Sequence[BookSnapshot],
                      final_snapshot: Optional[BookSnapshot]) -> Dict[str, int]:
    """Peak resting rows per side and live icebergs over ``snaps``; trade rows used at the end."""
    return {
        "resting_per_side": max((max(len(s.resting(BUY)), len(s.resting(SELL))) for s in snaps),
                                default=0),
        "trades": final_snapshot.trade_rows_used() if final_snapshot is not None else 0,
        "concurrent_icebergs": max((int(s.n_active) for s in snaps), default=0),
    }


def _snapshot_flags(pre_snapshots: Sequence[BookSnapshot], post_snapshots: Sequence[BookSnapshot],
                    final_snapshot: Optional[BookSnapshot]) -> Dict[str, bool]:
    """The three capacity flags, recomputed from the snapshots (V11/V14, v5.187).

    ``book_full`` / ``truncated``: the field is true on any pre-, post- or final
    snapshot; ``trade_buffer_saturated``: some post-snapshot has every trade row used.
    Shared by the builder, RB and I8.
    """
    snaps = list(pre_snapshots) + list(post_snapshots)
    if final_snapshot is not None:
        snaps.append(final_snapshot)
    return {
        "book_full": any(bool(sn.book_full) for sn in snaps),
        "truncated": any(bool(sn.truncated) for sn in snaps),
        "trade_buffer_saturated": any(
            sn.trade_rows_used() >= np.asarray(sn.trades).shape[0] for sn in post_snapshots),
    }


def _declaration(row, submission: int) -> AcceptedOrder:
    """An accepted order's declaration, decoded from its tape row."""
    return AcceptedOrder(
        oid=int(row[_T_OID]), submission=int(submission), side=int(row[_T_SIDE]),
        price_int=int(row[_T_PRICE]), qty_int=int(row[_T_QTY]), agent=int(row[_T_AGENT]),
        tape_ts=int(row[_T_TS]), exec_restriction=int(row[_T_EXEC]),
        peak_int=int(row[_T_PEAK]), ppd_int=int(row[_T_PPD]))


def _real_rows(tape) -> List[np.ndarray]:
    t = np.asarray(tape)
    return [t[i] for i in range(t.shape[0]) if int(t[i, _T_OID]) >= 1]


def _trade_agents(fills: Sequence[Fill]) -> set:
    return {f.buyer_agent for f in fills} | {f.seller_agent for f in fills}


def _settlement_tape(fills: Sequence[Fill], contract_id: int) -> np.ndarray:
    """The session's fills flattened to the ``(M, 6)`` int32 settlement tape."""
    t = np.array(
        [[f.sec, f.price_int, f.qty_int, f.buyer_agent, f.seller_agent, contract_id]
         for f in fills],
        dtype=np.int32,
    ).reshape(-1, 6)
    # OMIE-AGENT-PNL-SHAPE-GUARD-1: compute_agent_pnl must see (M, 6).
    assert t.ndim == 2 and t.shape == (len(fills), 6), t.shape
    return t


def _engine_settlements(fills: Sequence[Fill], contract_id: int) -> Dict[int, float]:
    if not fills:
        return {}
    t = jnp.asarray(_settlement_tape(fills, contract_id))
    return {int(a): float(compute_agent_pnl(t, jnp.int32(a))) for a in sorted(_trade_agents(fills))}


# ---------------------------------------------------------------------------
# The record builder (V2)
# ---------------------------------------------------------------------------
def _validate_tape(tape) -> np.ndarray:
    if not isinstance(tape, np.ndarray):
        raise RecordBuildError(f"tape must be a numpy array, got {type(tape).__name__}")
    if tape.ndim != 2 or tape.shape[1] != _N_COLS:
        raise RecordBuildError(f"tape must be (N, {_N_COLS}), got shape {tape.shape}")
    if tape.dtype != np.int32:
        raise RecordBuildError(f"tape must be int32, got {tape.dtype}")
    return tape


def build_record(
    tape,
    *,
    product_config: OMIEProductConfig = OMIEProductConfig(),
    market_config: OMIEMarketConfig = OMIEMarketConfig(),
    contract_status: ContractStatus = ContractStatus.OPEN,
    cancels_before: Mapping[int, Sequence[int]] = {},
    expire_at_gct: bool = False,
    capacity_bounds: Optional[Mapping[str, int]] = None,
    jit: bool = True,
) -> RunRecord:
    """Drive one run of ``tape`` on one contract and record what the engine did.

    Args:
        tape: ``(N, 13)`` int32 tape. Real rows (``order_id >= 1``) are submitted
            in order; pad rows are skipped wherever they sit.
        product_config / market_config: engine configuration.
        contract_status: the contract's status; non-OPEN rejects every row
            (``reason="contract_not_open"``).
        cancels_before: ``{order_id of row i: (oid, ...)}``: those accepted oids
            are cancelled (``ContractBook.cancel_order``) before row ``i``.
        expire_at_gct: apply ``ContractBook.expire_at_gct`` after the last row.
        capacity_bounds: analytic capacity bounds; default
            ``tape_capacity_bounds(tape)``.
        jit: ``False`` runs the whole build under ``jax.disable_jit()`` (eager).

    Raises:
        RecordBuildError: only for an unusable script (wrong shape / dtype, mixed
            contract ids among real rows, duplicate order ids, a cancel of an oid
            that was never accepted, or a cancel scheduled before a row that is
            not on the tape).
    """
    with contextlib.nullcontext() if jit else jax.disable_jit():
        return _build_record(tape, product_config=product_config, market_config=market_config,
                             contract_status=contract_status, cancels_before=cancels_before,
                             expire_at_gct=expire_at_gct, capacity_bounds=capacity_bounds, jit=jit)


def _build_record(tape, *, product_config, market_config, contract_status,
                  cancels_before, expire_at_gct, capacity_bounds, jit) -> RunRecord:
    tape = _validate_tape(tape)
    real_positions = [i for i in range(tape.shape[0]) if int(tape[i, _T_OID]) >= 1]
    real = [tape[i] for i in real_positions]
    cids = {int(r[_T_CONTRACT]) for r in real}
    if len(cids) > 1:
        raise RecordBuildError(f"mixed contract_id among real rows: {sorted(cids)}")
    oids = [int(r[_T_OID]) for r in real]
    if len(set(oids)) != len(oids):
        raise RecordBuildError("duplicate order_id among real rows")
    missing = {int(k) for k in cancels_before} - set(oids)
    if missing:
        raise RecordBuildError(f"cancels scheduled before rows not on the tape: {sorted(missing)}")
    contract_id = int(real[0][_T_CONTRACT]) if real else 0

    contract = Contract(contract_id=contract_id, delivery_start_minutes=_DELIVERY_START,
                        delivery_end_minutes=_DELIVERY_END, gct_minutes=_GCT,
                        status=contract_status)
    book = ContractBook(contract, product_config, market_config)
    state = book.init()
    is_open = contract_status == ContractStatus.OPEN

    accepted: Dict[int, AcceptedOrder] = {}
    rejections: List[Rejection] = []
    fills: List[Fill] = []
    pre_snaps: List[BookSnapshot] = []
    post_snaps: List[BookSnapshot] = []
    reports: List[EngineReport] = []
    cancel_events: List[Tuple[int, int]] = []
    cancellations: Dict[int, List[Cancellation]] = {}

    def cancel(c_oid, cause, qty, event_index):
        cancellations.setdefault(c_oid, []).append(
            Cancellation(oid=c_oid, cause=cause, qty_int=int(qty), event_index=event_index))

    for s, row in enumerate(real):
        oid = int(row[_T_OID])
        for c_oid in cancels_before.get(oid, ()):
            c_oid = int(c_oid)
            if c_oid not in accepted:
                raise RecordBuildError(f"cancel of oid {c_oid}, which was never accepted")
            side = accepted[c_oid].side
            before = _snapshot(state)   # state immediately before cancel_order
            qty = before.visible_qty(c_oid, side) + before.hidden_qty(c_oid)
            cancel(c_oid, "cancellation", qty, len(cancel_events))
            cancel_events.append((s, c_oid))
            state = book.cancel_order(state, c_oid, side)

        pre = _snapshot(state)
        state, res = book.submit_order(state, _row_to_order_traced(jnp.asarray(row), product_config))
        post = _snapshot(state)
        pre_snaps.append(pre)
        post_snaps.append(post)
        ok = bool(res.accepted)
        reports.append(EngineReport(
            submission=s, accepted=ok, traded=bool(res.traded),
            trade_qty_mw=float(res.trade_qty), trade_price_eur=float(res.trade_price),
            settlement_value_eur=float(res.settlement_value)))
        if not ok:
            rejections.append(Rejection(
                submission=s, oid=oid, row=tuple(int(x) for x in row),
                reason="validator" if is_open else "contract_not_open"))
            fills.extend(_fills_of_submission(pre, post, s))   # expected none; RB judges
            continue
        a = _declaration(row, s)
        accepted[oid] = a
        sub_fills = _fills_of_submission(pre, post, s)
        fills.extend(sub_fills)
        own = sum(f.qty_int for f in sub_fills if f.aggressor_oid == oid)
        if a.exec_restriction == IOC and a.qty_int - own > 0:
            cancel(oid, "ioc_remainder", a.qty_int - own, s)
        elif a.exec_restriction == FOK and not sub_fills:
            cancel(oid, "fok_kill", a.qty_int, s)

    n = len(real)
    if expire_at_gct:
        before = _snapshot(state)   # state immediately before expire_at_gct
        for oid, a in accepted.items():
            qty = before.visible_qty(oid, a.side) + before.hidden_qty(oid)
            if qty > 0:
                cancel(oid, "gate_expiry", qty, n)
        state = book.expire_at_gct(state)
    final = _snapshot(state)

    fill_qty = _qty_by_oid(fills)
    filled = {oid: fill_qty.get(oid, 0) for oid in accepted}
    still_resting = {oid: final.visible_qty(oid, a.side) for oid, a in accepted.items()}
    hidden = {oid: final.hidden_qty(oid) for oid in accepted}
    evicted = {oid: 0 for oid in accepted}
    evicted.update(_evictions(pre_snaps, post_snaps, fills, cancel_events,
                              market_config.book_depth))
    all_snaps = pre_snaps + post_snaps + [final]
    flags = _snapshot_flags(pre_snaps, post_snaps, final)
    bounds = dict(capacity_bounds) if capacity_bounds is not None else tape_capacity_bounds(tape)

    return RunRecord(
        tape=np.array(tape, copy=True),
        n_submissions=n,
        accepted=accepted,
        rejections=tuple(rejections),
        fills=tuple(fills),
        pre_snapshots=tuple(pre_snaps),
        post_snapshots=tuple(post_snaps),
        final_snapshot=final,
        filled=filled,
        still_resting=still_resting,
        hidden_remaining=hidden,
        cancelled_or_expired={oid: tuple(cancellations.get(oid, ())) for oid in
                              list(accepted) + [o for o in cancellations if o not in accepted]},
        evicted=evicted,
        cancel_events=tuple(cancel_events),
        expired_at_gct=bool(expire_at_gct),
        engine_reports=tuple(reports),
        engine_settlements=_engine_settlements(fills, contract_id),
        participants=frozenset(a.agent for a in accepted.values()),
        flags=flags,
        observed_capacity=_observe_capacity(all_snaps, final),
        capacity_bounds={k: int(v) for k, v in bounds.items()},
        unobserved=("rejection_reason",),
        jit=bool(jit),
        product_config=product_config,
        market_config=market_config,
    )


def record_from_draw(draw: Draw, **kw) -> RunRecord:
    """:func:`build_record` on ``draw.tape`` with the draw's capacity bounds and,
    when ``draw.meta`` has ``"cancel_before_row_of_oid"``, its scheduled cancel."""
    kw.setdefault("capacity_bounds", dict(draw.capacity_bounds))
    if "cancel_before_row_of_oid" in draw.meta and "cancels_before" not in kw:
        kw["cancels_before"] = {int(draw.meta["cancel_before_row_of_oid"]):
                                (int(draw.params["cancel_oid"]),)}
    return build_record(draw.tape, **kw)


def _snap_key(s: Optional[BookSnapshot]):
    if s is None:
        return None
    arrays = [np.asarray(getattr(s, k)) for k in ("bids", "asks", "trades", "iceberg_meta")]
    return (*(x for arr in arrays for x in (arr.tobytes(), arr.shape)),
            int(s.n_active), int(s.ts_counter), bool(s.book_full), bool(s.truncated))


def _report_key(er: EngineReport, qty_tick: float):
    return (int(er.submission), bool(er.accepted), bool(er.traded),
            int(round(float(er.trade_qty_mw) / qty_tick)))


def compare_records(a: RunRecord, b: RunRecord) -> Tuple[str, ...]:
    """Names of the **integer** ``RunRecord`` fields that differ.

    Excluded: the euro floats (``engine_settlements``; ``engine_reports``'
    ``trade_price_eur`` and ``settlement_value_eur``, whose eager and jitted
    values legitimately differ, V10) and ``jit`` itself. ``engine_reports`` is
    compared on submission, accepted, traded and the reported quantity in ticks.
    """
    qt = float(a.product_config.qty_tick)
    out = []
    for f in fields(RunRecord):
        name = f.name
        if name in ("engine_settlements", "jit"):
            continue
        x, y = getattr(a, name), getattr(b, name)
        if name == "tape":
            same = np.array_equal(np.asarray(x), np.asarray(y)) and np.asarray(x).dtype == np.asarray(y).dtype
        elif name in ("pre_snapshots", "post_snapshots"):
            same = [_snap_key(s) for s in x] == [_snap_key(s) for s in y]
        elif name == "final_snapshot":
            same = _snap_key(x) == _snap_key(y)
        elif name == "engine_reports":
            same = [_report_key(r, qt) for r in x] == [_report_key(r, qt) for r in y]
        elif isinstance(x, Mapping) or isinstance(y, Mapping):
            same = dict(x) == dict(y)
        elif isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
            same = tuple(x) == tuple(y)
        else:
            same = x == y
        if not same:
            out.append(name)
    return tuple(out)


# ---------------------------------------------------------------------------
# Check API (V6)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Violation:
    check_id: str
    field: str
    message: str
    submission: Optional[int] = None
    oid: Optional[int] = None
    fill_row: Optional[int] = None
    expected: Any = None
    observed: Any = None


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: str            # "pass" | "fail" | "invalid" (I8 only) | "not_applicable"
    violations: Tuple[Violation, ...]
    unobserved: Tuple[str, ...]
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class BatteryReport:
    """RB and I1-I9 for one record, in run order."""

    results: Tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(r.status in ("pass", "not_applicable") for r in self.results)

    @property
    def valid(self) -> bool:
        return all(not (r.check_id == "I8" and r.status == "invalid") for r in self.results)

    def statuses(self) -> Dict[str, str]:
        return {r.check_id: r.status for r in self.results}


def _result(check_id, violations, unobserved=(), evidence=None, *, bad_status="fail"):
    v = tuple(violations)
    return CheckResult(check_id=check_id, status=bad_status if v else "pass", violations=v,
                       unobserved=tuple(unobserved), evidence=dict(evidence or {}))


def _not_applicable(check_id, reason, unobserved=(), evidence=None):
    ev = dict(evidence or {})
    ev["reason"] = reason
    return CheckResult(check_id=check_id, status="not_applicable", violations=(),
                       unobserved=tuple(unobserved), evidence=ev)


def _never_raises(check_id: str, unobserved: Tuple[str, ...] = ()):
    """V6: a check never raises on content. Argument binding errors (a missing
    required keyword) still raise ``TypeError``; any error while reading the
    record becomes a ``"fail"`` with ``field="check_error"``."""

    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            sig.bind(*args, **kwargs)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - a notebook renders every check
                return CheckResult(
                    check_id=check_id, status="fail",
                    violations=(Violation(check_id, "check_error",
                                          f"{type(exc).__name__}: {exc}"),),
                    unobserved=tuple(unobserved), evidence={"exception": repr(exc)})
        return wrapper
    return deco


def assert_check_fails(result: CheckResult, check_id: str, *, field: Optional[str] = None,
                       status: str = "fail") -> CheckResult:
    """Assert ``result`` is ``check_id``'s result with ``status`` (and, if given, a
    violation on ``field``). Returns ``result``."""
    assert isinstance(result, CheckResult), f"not a CheckResult: {type(result).__name__}"
    assert result.check_id == check_id, f"check_id {result.check_id!r} != {check_id!r}"
    assert result.status == status, (
        f"{check_id}: status {result.status!r} != {status!r}; violations={result.violations}")
    if field is not None:
        got = {v.field for v in result.violations}
        assert field in got, f"{check_id}: no violation on field {field!r}; fields={sorted(got)}"
    return result


# Small helpers ---------------------------------------------------------------
def _sub_oid(rec: RunRecord) -> Dict[int, int]:
    """submission index -> submitting order id (accepted and rejected)."""
    m = {a.submission: a.oid for a in rec.accepted.values()}
    for r in rec.rejections:
        m.setdefault(r.submission, r.oid)
    return m


def _qty_by_oid(fills: Sequence[Fill]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for f in fills:
        for o in {f.passive_oid, f.aggressor_oid}:
            out[o] = out.get(o, 0) + int(f.qty_int)
    return out


def _expected_cancellations(rec: RunRecord, bad) -> Dict[int, List[Tuple[str, int, int]]]:
    """V14 (v5.187): the ``(cause, qty_int, event_index)`` entries each accepted oid
    must carry, recomputed from the record's snapshots and declarations. ``bad`` is
    RB's violation sink (an unreadable cancel event is itself a violation)."""
    out: Dict[int, List[Tuple[str, int, int]]] = {}
    n = int(rec.n_submissions)
    named: set = set()
    for j, ev in enumerate(rec.cancel_events):
        s, oid = int(ev[0]), int(ev[1])
        a = rec.accepted.get(oid)
        if a is None:
            bad("cancelled_or_expired", f"cancel event {j} names oid {oid}, which is not "
                f"accepted", oid=oid, submission=s)
            continue
        if not 1 <= s <= len(rec.post_snapshots):
            bad("cancelled_or_expired", f"cancel event {j} before submission {s} has no "
                f"pre-cancel snapshot", oid=oid, submission=s)
            continue
        if (s, oid) in named:
            q = 0
        else:
            before = rec.post_snapshots[s - 1]   # the state immediately before the cancel
            q = before.visible_qty(oid, a.side) + before.hidden_qty(oid)
        named.add((s, oid))
        out.setdefault(oid, []).append(("cancellation", int(q), j))
    for oid, a in rec.accepted.items():
        s = int(a.submission)
        if a.exec_restriction == IOC:
            own = sum(int(f.qty_int) for f in rec.fills
                      if int(f.submission) == s and int(f.aggressor_oid) == oid)
            if int(a.qty_int) - own > 0:
                out.setdefault(oid, []).append(("ioc_remainder", int(a.qty_int) - own, s))
        elif a.exec_restriction == FOK and not any(int(f.submission) == s for f in rec.fills):
            out.setdefault(oid, []).append(("fok_kill", int(a.qty_int), s))
    if rec.expired_at_gct:
        if n >= 1 and len(rec.post_snapshots) >= n:
            before = rec.post_snapshots[n - 1]   # the state immediately before expire_at_gct
            for oid, a in rec.accepted.items():
                q = before.visible_qty(oid, a.side) + before.hidden_qty(oid)
                if q > 0:
                    out.setdefault(oid, []).append(("gate_expiry", int(q), n))
        elif n >= 1:
            bad("cancelled_or_expired", "expired run has no pre-expiry snapshot")
    return out


# ---------------------------------------------------------------------------
# RB — record consistency (V14)
# ---------------------------------------------------------------------------
@_never_raises("RB", ("rejection_reason",))
def check_record(rec: RunRecord) -> CheckResult:
    """RB: the record is internally consistent with the engine state it holds."""
    V: List[Violation] = []

    def bad(field_, msg, **kw):
        V.append(Violation("RB", field_, msg, **kw))

    n = int(rec.n_submissions)
    if len(rec.pre_snapshots) != n:
        bad("pre_snapshots", f"{len(rec.pre_snapshots)} pre-snapshots for {n} submissions")
    if len(rec.post_snapshots) != n:
        bad("post_snapshots", f"{len(rec.post_snapshots)} post-snapshots for {n} submissions")
    n_pairs = min(len(rec.pre_snapshots), len(rec.post_snapshots), n)

    # fills: in range, ascending, inside the diff, equal to the row, complete.
    prev = None
    for f in rec.fills:
        s = int(f.submission)
        if not 0 <= s < n:
            bad("fills", f"fill row {f.row_index} tagged with submission {s} outside [0, {n})",
                submission=s, fill_row=f.row_index)
            continue
        key = (s, int(f.row_index))
        if prev is not None and key <= prev:
            bad("fills", f"fill (submission, row) {key} not ascending after {prev}",
                submission=s, fill_row=f.row_index)
        prev = key
        if s >= n_pairs:
            continue
        diff = _diff_rows(rec.pre_snapshots[s], rec.post_snapshots[s])
        if f.row_index not in diff:
            bad("fills", f"row {f.row_index} is not in submission {s}'s trade diff",
                submission=s, fill_row=f.row_index)
            continue
        want = _fill_from_row(np.asarray(rec.post_snapshots[s].trades)[f.row_index],
                              f.row_index, s)
        if want != f:
            bad("fills", f"fill disagrees with its trade row: {f} != {want}",
                submission=s, fill_row=f.row_index, expected=want, observed=f)
    present = {(int(g.submission), int(g.row_index)) for g in rec.fills}
    for s in range(n_pairs):
        for f in _fills_of_submission(rec.pre_snapshots[s], rec.post_snapshots[s], s):
            if (s, f.row_index) not in present:
                bad("fills", f"trade row {f.row_index} of submission {s} has no fill",
                    submission=s, fill_row=f.row_index)

    # filled == the sum over fills.
    want_filled = _qty_by_oid(rec.fills)
    for oid in set(rec.filled) | set(rec.accepted):
        if int(rec.filled.get(oid, 0)) != int(want_filled.get(oid, 0)):
            bad("filled", f"filled[{oid}]={rec.filled.get(oid, 0)} but fills sum to "
                f"{want_filled.get(oid, 0)}", oid=oid,
                expected=want_filled.get(oid, 0), observed=rec.filled.get(oid, 0))

    # still_resting / hidden_remaining == final_snapshot.
    fin = rec.final_snapshot
    for oid in set(rec.still_resting) | set(rec.accepted):
        a = rec.accepted.get(oid)
        want = fin.visible_qty(oid, a.side) if a is not None else 0
        if int(rec.still_resting.get(oid, 0)) != want:
            bad("still_resting", f"still_resting[{oid}]={rec.still_resting.get(oid, 0)} but the "
                f"final book holds {want}", oid=oid, expected=want,
                observed=rec.still_resting.get(oid, 0))
    for oid in set(rec.hidden_remaining) | set(rec.accepted):
        want = fin.hidden_qty(oid)
        if int(rec.hidden_remaining.get(oid, 0)) != want:
            bad("hidden_remaining", f"hidden_remaining[{oid}]={rec.hidden_remaining.get(oid, 0)}"
                f" but iceberg_meta holds {want}", oid=oid, expected=want,
                observed=rec.hidden_remaining.get(oid, 0))

    # rejections: disjoint from accepted, book unchanged, reported not accepted.
    acc_subs = {a.submission for a in rec.accepted.values()}
    for r in rec.rejections:
        if r.oid in rec.accepted:
            bad("rejections", f"oid {r.oid} is both rejected and accepted", oid=r.oid,
                submission=r.submission)
        if r.submission in acc_subs:
            bad("rejections", f"submission {r.submission} is both rejected and accepted",
                submission=r.submission)
        if 0 <= r.submission < n_pairs and _snap_key(rec.pre_snapshots[r.submission]) != \
                _snap_key(rec.post_snapshots[r.submission]):
            bad("rejections", f"rejected submission {r.submission} changed the book",
                submission=r.submission, oid=r.oid)
        if 0 <= r.submission < len(rec.engine_reports) and \
                rec.engine_reports[r.submission].accepted:
            bad("rejections", f"rejected submission {r.submission} is reported accepted",
                submission=r.submission, oid=r.oid)

    # engine_reports: one per submission, accepted agrees.
    if len(rec.engine_reports) != n:
        bad("engine_reports", f"{len(rec.engine_reports)} engine reports for {n} submissions")
    for i, er in enumerate(rec.engine_reports):
        if er.submission != i:
            bad("engine_reports", f"report {i} carries submission {er.submission}", submission=i)
        if bool(er.accepted) != (i in acc_subs):
            bad("engine_reports", f"report {i} accepted={er.accepted} but the record says "
                f"{'accepted' if i in acc_subs else 'rejected'}", submission=i)

    # evicted: each entry meets V4's signature.
    want_ev = _evictions(rec.pre_snapshots, rec.post_snapshots, rec.fills, rec.cancel_events,
                         rec.market_config.book_depth)
    for oid in set(rec.evicted) | set(want_ev):
        if int(rec.evicted.get(oid, 0)) != int(want_ev.get(oid, 0)):
            bad("evicted", f"evicted[{oid}]={rec.evicted.get(oid, 0)} but the level-wipe "
                f"signature gives {want_ev.get(oid, 0)}", oid=oid,
                expected=want_ev.get(oid, 0), observed=rec.evicted.get(oid, 0))

    # observed_capacity == recomputed from the snapshots.
    want_oc = _observe_capacity([*rec.pre_snapshots, *rec.post_snapshots, rec.final_snapshot],
                                rec.final_snapshot)
    for k, v in want_oc.items():
        if int(rec.observed_capacity.get(k, -1)) != v:
            bad("observed_capacity", f"observed_capacity[{k!r}]="
                f"{rec.observed_capacity.get(k)} but the snapshots give {v}",
                expected=v, observed=rec.observed_capacity.get(k))

    # engine_settlements cover every trade-row agent.
    for a in sorted(_trade_agents(rec.fills) - set(rec.engine_settlements)):
        bad("engine_settlements", f"trade-row agent {a} has no settlement entry")

    # accepted (v5.187): the accepted real rows, each equal to its tape declaration.
    real = _real_rows(rec.tape)
    want_acc: Dict[int, AcceptedOrder] = {}
    for s, row in enumerate(real):
        if 0 <= s < len(rec.engine_reports) and bool(rec.engine_reports[s].accepted):
            want_acc[int(row[_T_OID])] = _declaration(row, s)
    for oid in sorted(set(rec.accepted) | set(want_acc)):
        got, want = rec.accepted.get(oid), want_acc.get(oid)
        if got is None:
            bad("accepted", f"oid {oid} is accepted by the engine but missing from accepted",
                oid=oid, submission=want.submission, expected=want)
        elif want is None:
            bad("accepted", f"oid {oid} is in accepted but is not an accepted real tape row",
                oid=oid, observed=got)
        elif got != want:
            bad("accepted", f"accepted[{oid}]={got} disagrees with its tape declaration {want}",
                oid=oid, submission=want.submission, expected=want, observed=got)

    # participants (v5.187) == the accepted agents.
    want_part = frozenset(int(a.agent) for a in rec.accepted.values())
    if frozenset(rec.participants) != want_part:
        bad("participants", f"participants {sorted(rec.participants)} != accepted agents "
            f"{sorted(want_part)}", expected=want_part, observed=frozenset(rec.participants))

    # flags (v5.187) == recomputed from the snapshots.
    want_flags = _snapshot_flags(rec.pre_snapshots, rec.post_snapshots, rec.final_snapshot)
    for k, v in want_flags.items():
        if bool(rec.flags.get(k, False)) != v:
            bad("flags", f"flags[{k!r}]={rec.flags.get(k)} but the snapshots give {v}",
                expected=v, observed=rec.flags.get(k))
    for k in set(rec.flags) - set(want_flags):
        bad("flags", f"unknown flag {k!r}", observed=rec.flags.get(k))

    # cancelled_or_expired (v5.187) == recomputed from engine state and declarations.
    want_ce = _expected_cancellations(rec, bad)
    for oid in sorted(set(rec.cancelled_or_expired) | set(rec.accepted)):
        got_ms = sorted((str(c.cause), int(c.qty_int), int(c.event_index))
                        for c in rec.cancelled_or_expired.get(oid, ()))
        want_ms = sorted(want_ce.get(oid, ())) if oid in rec.accepted else []
        if got_ms != want_ms:
            bad("cancelled_or_expired", f"cancelled_or_expired[{oid}]={got_ms} but the engine "
                f"state and declarations give {want_ms}", oid=oid, expected=want_ms,
                observed=got_ms)

    return _result("RB", V, ("rejection_reason",), {"n_submissions": n, "n_fills": len(rec.fills)})


# ---------------------------------------------------------------------------
# I1 — volume balance (V7, amended v5.186)
# ---------------------------------------------------------------------------
@_never_raises("I1")
def check_i1_volume_balance(rec: RunRecord) -> CheckResult:
    """I1: matched buy volume == matched sell volume, each leg by its declared side."""
    if not rec.fills:
        return _not_applicable("I1", "no fills in this run")
    V: List[Violation] = []
    acc = rec.accepted
    by_side: Dict[int, Dict[int, int]] = {BUY: {}, SELL: {}}   # side -> submission -> qty
    for f in rec.fills:
        for leg, oid, want_side in (("buyer", f.buyer_oid, BUY), ("seller", f.seller_oid, SELL)):
            a = acc.get(oid)
            if a is None:
                V.append(Violation("I1", "fills", f"fill row {f.row_index}: {leg} oid {oid} "
                                   f"is not an accepted order", submission=f.submission,
                                   oid=oid, fill_row=f.row_index))
            elif int(a.side) != want_side:
                V.append(Violation("I1", "fill_sides", f"fill row {f.row_index}: {leg} oid "
                                   f"{oid} is declared {'BUY' if a.side == BUY else 'SELL'}",
                                   submission=f.submission, oid=oid, fill_row=f.row_index,
                                   expected=want_side, observed=int(a.side)))
            else:
                leg_qty = by_side[want_side]
                leg_qty[f.submission] = leg_qty.get(f.submission, 0) + int(f.qty_int)
    buy_by_sub, sell_by_sub = by_side[BUY], by_side[SELL]
    for s in sorted(set(buy_by_sub) | set(sell_by_sub)):
        bq, sq = buy_by_sub.get(s, 0), sell_by_sub.get(s, 0)
        if bq != sq:
            V.append(Violation("I1", "fills", f"submission {s}: buy {bq} != sell {sq}",
                               submission=s, expected=bq, observed=sq))
    bt, st = sum(buy_by_sub.values()), sum(sell_by_sub.values())
    if bt != st:
        V.append(Violation("I1", "fills", f"cumulative buy {bt} != sell {st}",
                           expected=bt, observed=st))
    return _result("I1", V, (), {"buy_total": bt, "sell_total": st, "n_fills": len(rec.fills)})


# ---------------------------------------------------------------------------
# I2 — quantity conservation (V4)
# ---------------------------------------------------------------------------
@_never_raises("I2")
def check_i2_quantity_conservation(rec: RunRecord, *, allow_eviction: bool = False) -> CheckResult:
    """I2: filled + still_resting + cancelled_or_expired + evicted + hidden_remaining
    == accepted, per accepted order, exact in ticks."""
    if not rec.accepted:
        return _not_applicable("I2", "no accepted orders")
    V: List[Violation] = []
    for oid, cs in rec.cancelled_or_expired.items():
        a = rec.accepted.get(oid)
        for c in cs:
            def bad(msg):
                V.append(Violation("I2", "cancelled_or_expired", msg, oid=oid,
                                   observed=(c.cause, c.qty_int, c.event_index)))
            if a is None:
                bad(f"oid {oid} has a {c.cause!r} entry but was never accepted")
                continue
            if c.oid != oid:
                bad(f"entry under oid {oid} names oid {c.oid}")
            if c.cause not in CANCEL_CAUSES:
                bad(f"oid {oid}: cause {c.cause!r} is not a §5.1 cause")
            elif c.cause == "ioc_remainder" and a.exec_restriction != IOC:
                bad(f"oid {oid}: an IOC remainder on a non-IOC order")
            elif c.cause == "fok_kill" and a.exec_restriction != FOK:
                bad(f"oid {oid}: a FOK kill on a non-FOK order")
            elif c.cause == "cancellation":
                ev = c.event_index
                if not (0 <= ev < len(rec.cancel_events)) or int(rec.cancel_events[ev][1]) != oid:
                    bad(f"oid {oid}: a cancellation no cancel event names")
            elif c.cause == "gate_expiry" and not rec.expired_at_gct:
                bad(f"oid {oid}: a gate expiry on a run that was not expired")
            if int(c.qty_int) < 0:
                bad(f"oid {oid}: negative cancelled quantity {c.qty_int}")
    for oid, a in rec.accepted.items():
        ev = int(rec.evicted.get(oid, 0))
        if ev != 0 and not allow_eviction:
            V.append(Violation("I2", "evicted", f"oid {oid}: evicted {ev} outside C14",
                               oid=oid, observed=ev, expected=0))
        filled = int(rec.filled.get(oid, 0))
        resting = int(rec.still_resting.get(oid, 0))
        hidden = int(rec.hidden_remaining.get(oid, 0))
        cancelled = sum(int(c.qty_int) for c in rec.cancelled_or_expired.get(oid, ()))
        lhs = filled + resting + cancelled + ev + hidden
        if lhs != int(a.qty_int):
            V.append(Violation(
                "I2", "identity",
                f"oid {oid}: filled {filled} + still_resting {resting} + cancelled_or_expired "
                f"{cancelled} + evicted {ev} + hidden_remaining {hidden} = {lhs} != accepted "
                f"{a.qty_int}", oid=oid, submission=a.submission, expected=int(a.qty_int),
                observed=lhs))
    return _result("I2", V, (), {"n_orders": len(rec.accepted)})


# ---------------------------------------------------------------------------
# I3 — uncrossed on every return (V7)
# ---------------------------------------------------------------------------
@_never_raises("I3")
def check_i3_uncrossed_on_return(rec: RunRecord) -> CheckResult:
    """I3: best_bid < best_ask (when both exist) on every post-snapshot and the final one."""
    V: List[Violation] = []
    snaps = [("post_snapshots", i, s) for i, s in enumerate(rec.post_snapshots)]
    if rec.final_snapshot is not None:
        snaps.append(("final_snapshot", None, rec.final_snapshot))
    for name, i, s in snaps:
        bb, ba = s.best_bid(), s.best_ask()
        if bb is not None and ba is not None and not bb < ba:
            V.append(Violation("I3", name, f"{name}[{i}]: best bid {bb} >= best ask {ba}",
                               submission=i, observed=(bb, ba)))
    return _result("I3", V, (), {"n_snapshots": len(snaps)})


# ---------------------------------------------------------------------------
# I4 — priority (V8)
# ---------------------------------------------------------------------------
def _is_hop(f: Fill, o: Optional[int]) -> bool:
    """A §4.2.1 Rule-3 hop fill of a submission whose submitting oid is ``o``."""
    return o is not None and int(f.aggressor_oid) != int(o) and int(f.passive_oid) == int(o)


def _priority_violations(rec: RunRecord, check_id: str = "I4",
                         hop_counter: Optional[List[int]] = None
                         ) -> List[Tuple[Violation, str, int, int]]:
    """``(violation, kind, maker_oid, competitor_oid)``; kind is ``"price"``,
    ``"time"`` or ``"hop_priority"`` (V8, amended v5.187)."""
    out = []
    sub_oid = _sub_oid(rec)
    depth: Dict[int, Dict[Tuple[int, int], int]] = {}
    entries: Dict[int, Dict[int, List[Tuple[int, int, int, int, int]]]] = {}
    n = len(rec.pre_snapshots)
    for f in rec.fills:
        s = int(f.submission)
        if not 0 <= s < n:
            out.append((Violation(check_id, "fills", f"fill row {f.row_index} claims submission "
                                  f"{s} with no pre-snapshot", submission=s,
                                  fill_row=f.row_index), "record", f.passive_oid, -1))
            continue
        if s not in depth:
            pre = rec.pre_snapshots[s]
            depth[s] = {}
            entries[s] = {}
            for side in (BUY, SELL):
                rows = [(r[0], r[1], r[2], r[4], r[5]) for r in pre.resting(side)]
                entries[s][side] = rows
                for (p, q, o, _sec, _nsec) in rows:
                    depth[s][(side, o)] = depth[s].get((side, o), 0) + int(q)
        if f.qty_signed < 0:
            side = SELL          # the aggressor bought: the maker sold
        elif f.qty_signed > 0:
            side = BUY
        else:
            a = rec.accepted.get(f.passive_oid)
            side = a.side if a is not None else SELL
        m = int(f.passive_oid)
        left = depth[s]
        m_row = next((e for e in entries[s][side] if e[2] == m), None)
        is_new = m_row is None or left.get((side, m), 0) <= 0
        m_price = int(f.price_int) if is_new else int(m_row[0])
        m_ts = None if is_new else (int(m_row[3]), int(m_row[4]))
        for (p, _q, o, sec, nsec) in entries[s][side]:
            if o == m or left.get((side, o), 0) <= 0:
                continue
            better = p < m_price if side == SELL else p > m_price
            earlier = p == m_price and (is_new or (sec, nsec) < m_ts)
            if better or earlier:
                kind = "price" if better else "time"
                out.append((Violation(
                    check_id, "fills",
                    f"submission {s}, fill row {f.row_index}: maker {m} @ {m_price} filled while "
                    f"oid {o} @ {p} (ts {(sec, nsec)}) had depth "
                    f"{left.get((side, o), 0)} and "
                    f"{'a better price' if better else 'an equal price and earlier time'}"
                    f"{' (maker is a new entry)' if is_new else ''}",
                    submission=s, oid=m, fill_row=f.row_index, observed=o), kind, m, o))
        o = sub_oid.get(s)
        if _is_hop(f, o):
            # V8 (v5.187), D3-01: the iceberg is also a new entry on its own side
            # (opposite o's declared side); nothing there may still cross the fill.
            if hop_counter is not None:
                hop_counter[0] += 1
            a_o = rec.accepted.get(int(o))
            o_side = int(a_o.side) if a_o is not None else side
            ice_side = SELL if o_side == BUY else BUY
            ice = int(f.aggressor_oid)
            price = int(f.price_int)
            for (p, _q, c, sec, nsec) in entries[s][ice_side]:
                if c == ice or left.get((ice_side, c), 0) <= 0:
                    continue
                crosses = p <= price if ice_side == SELL else p >= price
                if crosses:
                    out.append((Violation(
                        check_id, "hop_priority",
                        f"submission {s}, fill row {f.row_index}: iceberg {ice} hopped against "
                        f"oid {o} @ {price} while oid {c} @ {p} (ts {(sec, nsec)}) on the "
                        f"iceberg's side still had depth {left.get((ice_side, c), 0)}",
                        submission=s, oid=c, fill_row=f.row_index, observed=ice),
                        "hop_priority", ice, c))
        if (side, m) in left:
            left[(side, m)] -= int(f.qty_int)
    return out


@_never_raises("I4", ("queue_timestamp_correctness",))
def check_i4_priority(rec: RunRecord) -> CheckResult:
    """I4: no fill against maker m while a better-priced, or equal-priced and earlier
    (engine timestamps), resting order still had depth in the pre-submission book;
    at a Rule-3 hop, no entry on the iceberg's side that still crosses the fill
    price had depth (``field="hop_priority"``, v5.187)."""
    unobs = ("queue_timestamp_correctness",)
    if not rec.fills:
        return _not_applicable("I4", "no fills in this run", unobs)
    hops = [0]
    vs = [v for (v, _k, _m, _o) in _priority_violations(rec, hop_counter=hops)]
    return _result("I4", vs, unobs, {"n_fills": len(rec.fills), "hop_fills": hops[0]})


# ---------------------------------------------------------------------------
# I5 — price-setter (V9)
# ---------------------------------------------------------------------------
def _ladder(a: AcceptedOrder) -> Tuple[int, ...]:
    if a.exec_restriction != ICE or int(a.ppd_int) == 0:
        return (int(a.price_int),)
    peak = int(a.peak_int)
    n_slices = max(1, -(-int(a.qty_int) // peak)) if peak > 0 else 1
    return tuple(int(a.price_int) + k * int(a.ppd_int) for k in range(n_slices))


@_never_raises("I5")
def check_i5_price_setter(rec: RunRecord) -> CheckResult:
    """I5: every trade prints at its price-setter's declared limit (ICE: its declared
    ladder base + k*ppd, 0 <= k < ceil(qty/peak)); at the Rule-3 hop (an accepted ICE
    aggressor opposite the submitting order), the iceberg; and every trade crosses
    both legs' declared limits (``field="limit_cross"``, v5.187)."""
    if not rec.fills:
        return _not_applicable("I5", "no fills in this run", (), {"rule3_hop_fills": 0})
    V: List[Violation] = []
    sub_oid = _sub_oid(rec)
    hops = 0
    for f in rec.fills:
        o = sub_oid.get(int(f.submission))
        if o is None:
            V.append(Violation("I5", "fills", f"fill row {f.row_index}: submission "
                               f"{f.submission} has no submitting order", submission=f.submission,
                               fill_row=f.row_index))
            continue
        # D3-02 (v5.187): matchability, independent of the price-setter rule.
        buyer, seller = rec.accepted.get(f.buyer_oid), rec.accepted.get(f.seller_oid)
        if buyer is not None and seller is not None:
            b_lim, s_lim = max(_ladder(buyer)), min(_ladder(seller))
            for leg, a_leg, ok, lim in (("buyer", buyer, int(f.price_int) <= b_lim, b_lim),
                                        ("seller", seller, int(f.price_int) >= s_lim, s_lim)):
                if not ok:
                    V.append(Violation("I5", "limit_cross",
                                       f"fill row {f.row_index} printed {f.price_int}, "
                                       f"{'above' if leg == 'buyer' else 'below'} the {leg} oid "
                                       f"{a_leg.oid}'s limit {lim}", submission=f.submission,
                                       oid=int(a_leg.oid), fill_row=f.row_index,
                                       expected=lim, observed=int(f.price_int)))
        field_ = "fills"
        a_o = rec.accepted.get(o)
        a_ag = rec.accepted.get(int(f.aggressor_oid))
        if f.aggressor_oid == o:
            setter = f.passive_oid
        elif (f.passive_oid == o and a_ag is not None and a_ag.exec_restriction == ICE
              and a_o is not None and int(a_ag.side) != int(a_o.side)):
            hops += 1
            field_ = "rule3_hop"
            setter = f.aggressor_oid if RULE3_HOP_PRICE_SETTER == "iceberg" else f.passive_oid
        else:
            V.append(Violation("I5", "unruled_hop",
                               f"fill row {f.row_index} (submission {f.submission}): an entry "
                               f"other than the submitting order aggressed (oid "
                               f"{f.aggressor_oid}) against oid {f.passive_oid}: on sign-valid "
                               f"input a correct engine cannot produce this; probable cause: "
                               f"matchable hidden quantity left unmatched at an earlier "
                               f"submission (§3.3; §4.2.1 Rules 2–3) or a deferred slice "
                               f"activation (§4.2.1 Rule 4); on a sign-invalid ICEwPI input "
                               f"(§4.2, §13.6) it is a recorded observation",
                               submission=f.submission, oid=f.passive_oid, fill_row=f.row_index))
            continue
        a = rec.accepted.get(setter)
        if a is None:
            V.append(Violation("I5", field_, f"fill row {f.row_index}: price-setter {setter} is "
                               f"not an accepted order", submission=f.submission, oid=setter,
                               fill_row=f.row_index))
            continue
        allowed = _ladder(a)
        if int(f.price_int) not in allowed:
            V.append(Violation("I5", field_, f"fill row {f.row_index} printed {f.price_int}; "
                               f"setter {setter} declared {allowed if len(allowed) > 1 else allowed[0]}",
                               submission=f.submission, oid=setter, fill_row=f.row_index,
                               expected=allowed, observed=int(f.price_int)))
    return _result("I5", V, (), {"rule3_hop_fills": hops, "n_fills": len(rec.fills),
                                 "rule3_hop_price_setter": RULE3_HOP_PRICE_SETTER})


# ---------------------------------------------------------------------------
# I6 — settlement (V10)
# ---------------------------------------------------------------------------
def _i6_constants(pc: OMIEProductConfig) -> Tuple[float, int]:
    k = float(pc.qty_tick) * float(pc.price_tick) * float(pc.delivery_duration_hours)
    offset = int(round(-pc.price_min / pc.price_tick))
    return k, offset


def i6_tolerance(rows: Sequence[Tuple[int, int]], product_config: OMIEProductConfig = OMIEProductConfig(),
                 *, chunks: Optional[int] = None) -> Tuple[float, float]:
    """``(tau_s, V_s)`` for one submission's fill rows ``[(price_int, qty_int), ...]``.

    $\\tau_s = k\\,u\\,(c_s W_s + O\\,Q_s + 5\\lvert D_s\\rvert)(1 + (c_s + 8)u)$,
    $V_s = k D_s$, with $W_s=\\sum p_i q_i$, $Q_s=\\sum q_i$, $D_s = W_s - O Q_s$
    (V10, amended v5.187). ``chunks`` is $c_s$, the number of settlement chunks
    (distinct engine timestamps among the rows); ``None`` means $c_s = m_s$, one
    chunk per row (the chunk-agnostic worst case).
    """
    k, offset = _i6_constants(product_config)
    W = sum(int(p) * int(q) for p, q in rows)
    Q = sum(int(q) for _p, q in rows)
    D = W - offset * Q
    c = len(rows) if chunks is None else int(chunks)
    tau = k * _U * (c * W + offset * Q + 5 * abs(D)) * (1 + (c + 8) * _U)
    return tau, k * D


@_never_raises("I6", ("independent_valuation",))
def check_i6_settlement(rec: RunRecord) -> CheckResult:
    """I6: (b) every accepted submission's report agrees with its trade rows; (a) the
    engine's per-participant settlements sum to zero within the summed tolerance."""
    unobs = ("independent_valuation",)
    if not rec.accepted:
        return _not_applicable("I6", "no accepted submissions", unobs)
    V: List[Violation] = []
    pc = rec.product_config
    by_sub: Dict[int, List[Fill]] = {}
    for f in rec.fills:
        by_sub.setdefault(int(f.submission), []).append(f)
    acc_subs = {a.submission: a for a in rec.accepted.values()}
    for s in by_sub:
        if s not in acc_subs:
            V.append(Violation("I6", "fills", f"submission {s} wrote trade rows but is not an "
                               f"accepted submission", submission=s))
    tau_sum = 0.0
    max_ratio = 0.0
    chunks_by_sub: Dict[int, int] = {}
    for s in sorted(acc_subs):
        sub_fills = by_sub.get(s, [])
        rows = [(f.price_int, f.qty_int) for f in sub_fills]
        m = len(rows)
        Q = sum(q for _p, q in rows)
        c = len({(int(f.sec), int(f.nsec)) for f in sub_fills})   # c_s, the settlement chunks
        chunks_by_sub[s] = c
        tau, value = i6_tolerance(rows, pc, chunks=c)
        tau_sum += tau
        if Q >= 2 ** 24:
            V.append(Violation("I6", "engine_reports", f"submission {s}: Q_s={Q} >= 2^24 breaks "
                               f"the float32 precondition", submission=s, observed=Q))
        if not 0 <= s < len(rec.engine_reports):
            V.append(Violation("I6", "engine_reports", f"submission {s} has no engine report",
                               submission=s))
            continue
        er = rec.engine_reports[s]
        if bool(er.traded) != (m > 0):
            V.append(Violation("I6", "engine_reports", f"submission {s}: traded={er.traded} but "
                               f"{m} trade rows", submission=s, expected=m > 0,
                               observed=bool(er.traded)))
        q_rep = int(round(float(er.trade_qty_mw) / float(pc.qty_tick)))
        if q_rep != Q:
            V.append(Violation("I6", "engine_reports", f"submission {s}: reported quantity "
                               f"{er.trade_qty_mw} MW = {q_rep} ticks != {Q} ticks in its rows",
                               submission=s, expected=Q, observed=q_rep))
        err = abs(float(er.settlement_value_eur) - value)
        if tau > 0:
            max_ratio = max(max_ratio, err / tau)
        if not err <= tau:
            V.append(Violation("I6", "engine_reports", f"submission {s}: reported settlement "
                               f"{er.settlement_value_eur!r} vs rows {value!r} (|err|={err:.3e} > "
                               f"tau={tau:.3e})", submission=s, expected=value,
                               observed=float(er.settlement_value_eur)))
    # (a)
    for f in rec.fills:
        for leg, oid, agent in (("buyer", f.buyer_oid, f.buyer_agent),
                                ("seller", f.seller_oid, f.seller_agent)):
            if agent not in rec.participants:
                V.append(Violation("I6", "participants", f"fill row {f.row_index}: {leg} agent "
                                   f"{agent} is not an enumerated participant",
                                   submission=f.submission, oid=oid, fill_row=f.row_index,
                                   observed=agent))
            a = rec.accepted.get(oid)
            if a is None or int(a.agent) != int(agent):
                V.append(Violation("I6", "participants", f"fill row {f.row_index}: {leg} agent "
                                   f"{agent} is not the declared agent of oid {oid} "
                                   f"({None if a is None else a.agent})",
                                   submission=f.submission, oid=oid, fill_row=f.row_index,
                                   expected=None if a is None else a.agent, observed=agent))
    for a in sorted(_trade_agents(rec.fills) - set(rec.engine_settlements)):
        V.append(Violation("I6", "engine_settlements", f"trade-row agent {a} has no settlement"))
    total = float(sum(float(v) for v in rec.engine_settlements.values()))
    if not abs(total) <= tau_sum:
        V.append(Violation("I6", "engine_settlements", f"settlements sum to {total!r}, beyond "
                           f"the summed tolerance {tau_sum:.3e}", expected=0.0, observed=total))
    return _result("I6", V, unobs, {"settlement_sum": total, "tau_sum": tau_sum,
                                    "max_err_over_tau": max_ratio, "chunks": chunks_by_sub})


# ---------------------------------------------------------------------------
# I7 — no manufactured liquidity (V7)
# ---------------------------------------------------------------------------
@_never_raises("I7")
def check_i7_no_manufactured_liquidity(rec: RunRecord) -> CheckResult:
    """I7: total traded <= min(total buy accepted, total sell accepted)."""
    traded = sum(int(f.qty_int) for f in rec.fills)
    buy = sum(int(a.qty_int) for a in rec.accepted.values() if int(a.side) == BUY)
    sell = sum(int(a.qty_int) for a in rec.accepted.values() if int(a.side) == SELL)
    cap = min(buy, sell)
    V = []
    if traded > cap:
        V.append(Violation("I7", "fills", f"traded {traded} > min(buy {buy}, sell {sell})",
                           expected=cap, observed=traded))
    return _result("I7", V, (), {"traded": traded, "buy_accepted": buy, "sell_accepted": sell})


# ---------------------------------------------------------------------------
# I8 — capacity headroom (V11)
# ---------------------------------------------------------------------------
@_never_raises("I8", ("matching_iterations",))
def check_i8_capacity_headroom(rec: RunRecord, *, envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
                               allow_breach: bool = False) -> CheckResult:
    """I8: the run stayed inside the capacity envelope; a breach is INVALID."""
    unobs = ("matching_iterations",)
    observed = _observe_capacity(_all_snapshots(rec), rec.final_snapshot)
    bounds = dict(rec.capacity_bounds) if rec.capacity_bounds else tape_capacity_bounds(rec.tape)
    if "hops" not in bounds:
        bounds = dict(bounds, hops=tape_capacity_bounds(rec.tape)["hops"])
    observed["hops"] = int(bounds["hops"])
    margins = {
        "resting_per_side": int(envelope.max_resting_per_side),
        "trades": int(envelope.max_trades),
        "hops": int(envelope.max_hops),
        "concurrent_icebergs": int(envelope.max_concurrent_icebergs),
    }
    V = []
    for axis, v in observed.items():
        if v > margins[axis]:
            V.append(Violation("I8", "capacity_bounds" if axis == "hops" else "observed_capacity",
                               f"{axis}={v} > margin {margins[axis]}", expected=margins[axis],
                               observed=v))
    # V11 (v5.187): the OR of rec.flags and the predicates recomputed from the snapshots.
    snap_flags = _snapshot_flags(rec.pre_snapshots, rec.post_snapshots, rec.final_snapshot)
    flags = {k: bool(rec.flags.get(k, False)) or bool(snap_flags[k]) for k in
             ("book_full", "truncated", "trade_buffer_saturated")}
    for k, v in flags.items():
        if v:
            V.append(Violation("I8", "flags", f"{k} raised", observed=True))
    evidence = {"observed": observed, "margins": margins, "flags": flags,
                "breaches": [v.message for v in V]}
    if V and allow_breach:
        return CheckResult("I8", "not_applicable", tuple(V), unobs,
                           dict(evidence, reason="capacity breach permitted (C14 mode)"))
    return _result("I8", V, unobs, evidence, bad_status="invalid")


# ---------------------------------------------------------------------------
# I9 — non-vacuity (V7)
# ---------------------------------------------------------------------------
@_never_raises("I9")
def check_i9_non_vacuity(rec: RunRecord, *, expects_trade: bool) -> CheckResult:
    """I9: at least one accepted order and, when ``expects_trade``, at least one fill."""
    V = []
    if not rec.accepted:
        V.append(Violation("I9", "accepted", "no order was accepted"))
    if expects_trade and not rec.fills:
        V.append(Violation("I9", "fills", "the construction implies a trade; none occurred"))
    return _result("I9", V, (), {"n_accepted": len(rec.accepted), "n_fills": len(rec.fills),
                                 "expects_trade": bool(expects_trade)})


# ---------------------------------------------------------------------------
# M1-M6 — metamorphic relations (V12)
# ---------------------------------------------------------------------------
def _replace_violation(v: Violation, **kw) -> Violation:
    d = {f.name: getattr(v, f.name) for f in fields(Violation)}
    d.update(kw)
    return Violation(**d)


def _total(fills) -> int:
    return sum(int(f.qty_int) for f in fills)


@_never_raises("M1")
def check_m1_tie_permutation(base: RunRecord, permuted: RunRecord, *,
                             tie_oids: Sequence[int]) -> CheckResult:
    """M1: equal aggregate matched volume; no I4 price-priority violation involving a
    ``tie_oids`` maker in either record."""
    V = []
    a, b = _total(base.fills), _total(permuted.fills)
    if a != b:
        V.append(Violation("M1", "fills", f"aggregate volume {a} != {b}", expected=a, observed=b))
    ties = {int(o) for o in tie_oids}
    for name, rec in (("base", base), ("permuted", permuted)):
        for v, kind, maker, comp in _priority_violations(rec, "M1"):
            if kind == "price" and (maker in ties or comp in ties):
                V.append(_replace_violation(v, message=f"{name}: {v.message}"))
    return _result("M1", V, (), {"base_volume": a, "permuted_volume": b})


@_never_raises("M2")
def check_m2_monotonicity(base: RunRecord, variant: RunRecord, *, aggressor_oid: int,
                          relation: str) -> CheckResult:
    """M2: a larger aggressor never matches less; a tighter limit never matches more."""
    oid = int(aggressor_oid)
    a = _qty_by_oid(base.fills).get(oid, 0)
    b = _qty_by_oid(variant.fills).get(oid, 0)
    V = []
    if relation == "size_increased":
        if b < a:
            V.append(Violation("M2", "fills", f"size increased but matched {b} < {a}",
                               oid=oid, expected=a, observed=b))
    elif relation == "limit_tightened":
        if b > a:
            V.append(Violation("M2", "fills", f"limit tightened but matched {b} > {a}",
                               oid=oid, expected=a, observed=b))
    else:
        V.append(Violation("M2", "relation", f"unknown relation {relation!r}"))
    return _result("M2", V, (), {"base_matched": a, "variant_matched": b, "relation": relation})


def _m3_key(f: Fill):
    return (f.price_int, f.qty_int, f.passive_oid, f.aggressor_oid, f.buyer_agent, f.seller_agent)


@_never_raises("M3")
def check_m3_ioc_non_equivalence(ioc: RunRecord, non: RunRecord, *, oid: int) -> CheckResult:
    """M3: identical trade lists; the IOC's own remainder never rests."""
    V = []
    ka, kb = [_m3_key(f) for f in ioc.fills], [_m3_key(f) for f in non.fills]
    if ka != kb:
        V.append(Violation("M3", "fills", f"IOC trades {ka} != NON trades {kb}",
                           expected=kb, observed=ka))
    oid = int(oid)
    if int(ioc.still_resting.get(oid, 0)) != 0:
        V.append(Violation("M3", "still_resting", f"IOC oid {oid} still rests "
                           f"{ioc.still_resting.get(oid)}", oid=oid))
    a = ioc.accepted.get(oid)
    if a is None:
        V.append(Violation("M3", "accepted", f"IOC oid {oid} was not accepted", oid=oid))
    else:
        for i in range(a.submission, len(ioc.post_snapshots)):
            snap = ioc.post_snapshots[i]
            if any(r[2] == oid for side in (BUY, SELL) for r in snap.resting(side)):
                V.append(Violation("M3", "post_snapshots", f"IOC oid {oid} rests in "
                                   f"post_snapshots[{i}]", oid=oid, submission=i))
    return _result("M3", V, (), {"n_fills": len(ioc.fills)})


def _byte_identical(a: BookSnapshot, b: BookSnapshot) -> bool:
    for name in ("bids", "asks", "trades", "iceberg_meta"):
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
            return False
    return int(a.n_active) == int(b.n_active)


@_never_raises("M4")
def check_m4_fok_dichotomy(rec: RunRecord, *, fok_oid: int) -> CheckResult:
    """M4: the FOK fills exactly its accepted quantity, or zero fills and a byte-identical book."""
    oid = int(fok_oid)
    a = rec.accepted.get(oid)
    if a is None:
        return _result("M4", [Violation("M4", "accepted", f"FOK oid {oid} was not accepted",
                                        oid=oid)])
    s = a.submission
    own = [f for f in rec.fills if f.submission == s and oid in (f.passive_oid, f.aggressor_oid)]
    got = _total(own)
    V = []
    if got == int(a.qty_int):
        pass
    elif got == 0 and not any(f.submission == s for f in rec.fills):
        if not _byte_identical(rec.pre_snapshots[s], rec.post_snapshots[s]):
            V.append(Violation("M4", "post_snapshots", f"killed FOK {oid} changed the book",
                               oid=oid, submission=s))
    else:
        V.append(Violation("M4", "fills", f"FOK {oid} filled {got} of {a.qty_int}",
                           oid=oid, submission=s, expected=int(a.qty_int), observed=got))
    return _result("M4", V, (), {"filled": got, "accepted_qty": int(a.qty_int)})


def _m5_map(rec: RunRecord, aggressors) -> Dict[Tuple[int, int], int]:
    """V12 (v5.187): ``(counterparty, price_int) -> qty`` over every fill of the
    aggressors' submissions; the counterparty is ``passive_oid`` on a plain fill and
    the iceberg ``aggressor_oid`` at a hop fill."""
    aggressors = {int(o) for o in aggressors}
    subs = {int(rec.accepted[o].submission) for o in aggressors if o in rec.accepted}
    subs |= {int(f.submission) for f in rec.fills if int(f.aggressor_oid) in aggressors}
    sub_oid = _sub_oid(rec)
    out: Dict[Tuple[int, int], int] = {}
    for f in rec.fills:
        s = int(f.submission)
        if s not in subs:
            continue
        cp = int(f.aggressor_oid) if _is_hop(f, sub_oid.get(s)) else int(f.passive_oid)
        k = (cp, int(f.price_int))
        out[k] = out.get(k, 0) + int(f.qty_int)
    return out


@_never_raises("M5")
def check_m5_split_aggressor(whole: RunRecord, split: RunRecord, *, whole_oid: int,
                             split_oids: Sequence[int]) -> CheckResult:
    """M5: equal maps (counterparty, price_int) -> qty over the aggressors' submissions' fills."""
    a = _m5_map(whole, {int(whole_oid)})
    b = _m5_map(split, {int(o) for o in split_oids})
    V = []
    if a != b:
        V.append(Violation("M5", "fills", f"whole {a} != split {b}", expected=a, observed=b))
    return _result("M5", V, (), {"whole": a, "split": b})


@_never_raises("M6")
def check_m6_ownership_blindness(base: RunRecord, relabelled: RunRecord, *,
                                 agent_map: Mapping[int, int]) -> CheckResult:
    """M6: identical trade lists under an agent relabelling; agents map through ``agent_map``."""
    V = []
    ka, kb = ([(f.price_int, f.qty_int, f.passive_oid, f.aggressor_oid, f.sec, f.nsec)
               for f in r.fills] for r in (base, relabelled))
    if ka != kb:
        V.append(Violation("M6", "fills", f"trade list changed: {ka} != {kb}",
                           expected=ka, observed=kb))
    for fa, fb in zip(base.fills, relabelled.fills):
        for leg in ("buyer_agent", "seller_agent"):
            want = agent_map.get(getattr(fa, leg), getattr(fa, leg))
            if int(getattr(fb, leg)) != int(want):
                V.append(Violation("M6", "fills", f"fill row {fb.row_index}: {leg} "
                                   f"{getattr(fb, leg)} != mapped {want}",
                                   fill_row=fb.row_index, expected=want,
                                   observed=getattr(fb, leg)))
    return _result("M6", V, (), {"n_fills": len(base.fills)})


# ---------------------------------------------------------------------------
# Registry, statements, driver
# ---------------------------------------------------------------------------
ALL_CHECKS: Tuple[str, ...] = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9",
                               "M1", "M2", "M3", "M4", "M5", "M6")

CHECKS: Mapping[str, Callable[..., CheckResult]] = {
    "RB": check_record,
    "I1": check_i1_volume_balance,
    "I2": check_i2_quantity_conservation,
    "I3": check_i3_uncrossed_on_return,
    "I4": check_i4_priority,
    "I5": check_i5_price_setter,
    "I6": check_i6_settlement,
    "I7": check_i7_no_manufactured_liquidity,
    "I8": check_i8_capacity_headroom,
    "I9": check_i9_non_vacuity,
    "M1": check_m1_tie_permutation,
    "M2": check_m2_monotonicity,
    "M3": check_m3_ioc_non_equivalence,
    "M4": check_m4_fok_dichotomy,
    "M5": check_m5_split_aggressor,
    "M6": check_m6_ownership_blindness,
}

#: Each statement verbatim from the story card's Description (markdown emphasis removed).
STATEMENTS: Mapping[str, str] = {
    "RB": (
        "the record builder's output is internally consistent: snapshot sequences, "
        "fills, buckets, rejections, reports, evictions, observed capacity and "
        "settlement coverage agree with the engine state the record holds."
    ),
    'I1': (
        'matched buy volume == matched sell volume, per contract, per step and cumulatively, each side read independently from the trade rows.'
    ),
    'I2': (
        "per accepted order, $\\mathit{filled} + \\mathit{still\\_resting} + \\mathit{cancelled\\_or\\_expired} + \\mathit{evicted} + \\mathit{hidden\\_remaining} = \\mathit{accepted}$, exact in integer ticks; cancelled_or_expired comes from order semantics — the §5.1 causes only (IOC remainder, FOK kill, cancellation, gate expiry) — never as the residual accepted − filled − resting; an iceberg's hidden quantity is its own bucket, read from the engine; evicted (an implementation-limit fate, not a §5.1 cause) is its own bucket, read from the book diff, and must be 0 outside C14 (I8); rejected orders are recorded separately and are not in the identity."
    ),
    'I3': (
        "the book is never left crossed once submit_order returns (§4.2.1 Rule 4's post-condition), checked after every return, including the last submission."
    ),
    'I4': (
        "no trade against a resting order while a better-priced, or equal-priced and earlier, resting order is unfilled; each fill carries the index of the submission that produced it, depth is decremented as a sweep walks the queue, checked against the book before each submission. Cannot see: the queue key is the engine's own stored timestamp, so whether the engine assigned those timestamps per §5.2 (partial execution keeps its timestamp; a new iceberg slice gets a new one) is not observable through this check (C05's interloper certifies it)."
    ),
    'I5': (
        "every trade price equals the limit price of that trade's price-setter — the resting order under §6.2 item 3, or the iceberg under §6.2 item 7 and §5.2; at the §4.2.1 Rule-3 hop, the iceberg (OQ-31 RESOLVED 2026-10-02, Reading B; market-spec.md v2.13 §4.2.1 Rule 3 and §6.2 item 7: the hop trades at the regenerated slice's own limit, never at the aggressor's limit) — with one narrow exemption: a price-incremented iceberg's own declared ladder $\\mathit{base} + k\\,\\mathit{ppd}$, $k$ bounded by its declared slice count. Every trade is also matchable (§6.2 item 2): its price is no higher than the buyer's declared limit and no lower than the seller's, an ICE leg taking the most permissive rung of its declared ladder. I5 owns mispricing."
    ),
    'I6': (
        "(a) the engine's per-participant settlements, summed over every participant id that appears in a trade row, equal zero within the tolerance derived for (b), and every trade row's buyer and seller are enumerated participants (the integer per-row identity holds by construction and is not a check; its broken-input fixture breaks the settlement output, e.g. drops one participant's leg); (b) the settlement the engine reports for each submission agrees with the trade rows that submission wrote. Cannot see: a trade at the wrong price (settlement is computed from the same rows, so both sides move together); that is I5's and C02's job. I6 is the cash leg only (§8.4's energy value is exogenous)."
    ),
    'I7': (
        'total traded volume ≤ min(total buy accepted, total sell accepted).'
    ),
    'I8': (
        "capacity headroom per the plan's capacity-validity table. Cannot see: the matching-iteration count (reported unobserved); evictions by regenerated slices (they do not raise book_full)."
    ),
    'I9': (
        'non-vacuity: every draw produced at least one accepted order and, where the construction implies one, at least one trade.'
    ),
    'M1': (
        'permuting orders that share a timestamp does not change aggregate matched volume; where they compete on price, the tie-break follows §6.1.'
    ),
    'M2': (
        "increasing an aggressor's size never decreases matched volume; tightening its limit never increases it."
    ),
    'M3': (
        "IOC and NON with identical price and quantity produce identical trade lists; the IOC's own remainder (by its own id) never rests; the resting maker legitimately survives."
    ),
    'M4': (
        'FOK either fills exactly the requested quantity, or produces zero fills and a byte-identical book.'
    ),
    'M5': (
        'one aggressor of size $Q$ against a static book == $k$ consecutive aggressors of size $Q/k$.'
    ),
    'M6': (
        'permuting agent identifiers leaves the trade list unchanged, including when the permutation makes an order match its own owner (§6.6, no self-trade prevention).'
    ),
}


def run_battery(rec: RunRecord, *, expects_trade: bool, envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
                allow_capacity_breach: bool = False) -> BatteryReport:
    """RB and I1-I9 on one record, in that order, with no short-circuit."""
    return BatteryReport(results=(
        check_record(rec),
        check_i1_volume_balance(rec),
        check_i2_quantity_conservation(rec),
        check_i3_uncrossed_on_return(rec),
        check_i4_priority(rec),
        check_i5_price_setter(rec),
        check_i6_settlement(rec),
        check_i7_no_manufactured_liquidity(rec),
        check_i8_capacity_headroom(rec, envelope=envelope, allow_breach=allow_capacity_breach),
        check_i9_non_vacuity(rec, expects_trade=expects_trade),
    ))
