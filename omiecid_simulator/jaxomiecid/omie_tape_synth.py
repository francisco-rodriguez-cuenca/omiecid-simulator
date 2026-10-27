"""Synthetic OMIE order-tape generator — the data source for the certification ladder.

Specification: ``docs/omiecid/scenario-ladder.md``. An earlier version of this
generator (the *archive*) is the starting point: constructions marked
"archive-identical" keep its tape bytes, and the corrected constructions
(rule-3 and incoming icebergs, ICEwPI, near-bound validation, FOK boundary,
gating by contract status) are new.

**Parametric goldens.** The generator never draws an arbitrary tape and then asks
what happens to it. It draws the parameters of a *construction* whose outcome is
a closed form in those parameters, and emits the tape for that construction. Each
``make_*`` function is one construction and returns a :class:`Draw` carrying the
tape, the drawn ``params`` (domain units *and* their tick integers) and the
generator author's construction-side ``expected`` values (integer ticks only).
``Draw.expected`` is **not** the certification oracle: a scenario's K2 derives its
own expectation from ``params`` and the spec (independence rule 1).

**Layer and import boundary.** This module lives in ``jaxomiecid/`` as a
sibling producer of the ``(N, 13)`` int32 tape that ``Session`` and
``ContractBook`` consume. It imports only numpy, the standard library, the omie
value layer (``encoding``, ``omie_config``, ``order_types``, ``contract``) and the
loader's public :data:`PAD_ORDER_ROW`. It must never import the engine
(``contract_book``, ``iceberg_manager``, ``order_validator``, ``settlement``,
``trade_extraction``) or ``omie_session``; it uses no JAX op.

**Determinism.** Every draw comes from one ``numpy.random.Generator(PCG64)``
stream seeded by the draw's integer seed. Reproducibility for a fixed code version
is exact (same seed, same bytes, across processes and backends), but it is not
forward-compatible: inserting a draw into a construction shifts every later value.
Construction digests are therefore stored per code version in
``tests/omiecid/fixtures/cert/synth_digests.json``, written only by::

    python -m omiecid_simulator.jaxomiecid.omie_tape_synth --write-digests <path>

**Tick arithmetic.** Every ``expected`` value is computed from encoded
integers, never as ``encode(float expression)``. The constructions kept
byte-identical to the archive (C02, C03, C05 aggressor, C06, C07, C08 cascade)
still draw floats and encode them through ``encoding.py`` so their tapes are
unchanged; their expectations are integer arithmetic over the encoded fields.

**Capacity envelope (invariant I8).** A draw must not reach the
simulator's fixed-array ceilings (``book_depth``, ``nTrades``,
``max_match_iterations``, ``max_icebergs``). :class:`CapacityEnvelope` carries the
margins and :func:`check_envelope` refuses a draw whose analytic upper bounds
(``Draw.capacity_bounds``) exceed them. The bounds are sound upper bounds, never
estimates: a *unit* is 1 per non-iceberg order and ``ceil(total / peak)`` per
iceberg; every trade exhausts at least one unit. C14 alone may exceed the
envelope, through the :data:`C14_CEILING_OPT_OUT` sentinel; every mode in this
module refuses it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import types
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from omiecid_simulator.jaxomiecid.omie_data_loader import PAD_ORDER_ROW
from omiecid_simulator.omiecid.contract import ContractStatus
from omiecid_simulator.omiecid.encoding import encode_ppd, encode_price, encode_qty
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, Status, Validity

# Column indices of the 13-wide int32 tape row, in the loader's order:
# [0] contract_id [1] side [2] price [3] quantity [4] order_id [5] agent_id
# [6] timestamp [7] exec_restriction [8] validity [9] expiry_time [10] status
# [11] peak_qty [12] ppd
COL_CONTRACT, COL_SIDE, COL_PRICE, COL_QTY = 0, 1, 2, 3
COL_OID, COL_AGENT, COL_TS = 4, 5, 6
COL_EXEC, COL_VALIDITY, COL_EXPIRY = 7, 8, 9
COL_STATUS, COL_PEAK, COL_PPD = 10, 11, 12
N_COLS = 13

DIGEST_SEEDS: Tuple[int, ...] = tuple(range(8))
DIGEST_SCHEMA_VERSION = 1
DIGEST_ALGORITHM = "blake2b-128"

_BUY, _SELL = int(Side.BUY), int(Side.SELL)
_NON, _IOC, _FOK, _ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                          int(ExecRestriction.FOK), int(ExecRestriction.ICE))

AXES: Tuple[str, ...] = ("resting_per_side", "trades", "hops", "concurrent_icebergs")


# ---------------------------------------------------------------------------
# Capacity envelope (invariant I8)
# ---------------------------------------------------------------------------
_ENVELOPE_CEILING_FIELD = {
    "max_resting_per_side": "book_depth",
    "max_trades": "nTrades",
    "max_hops": "max_match_iterations",
    "max_concurrent_icebergs": "max_icebergs",
}
_AXIS_TO_FIELD = {
    "resting_per_side": "max_resting_per_side",
    "trades": "max_trades",
    "hops": "max_hops",
    "concurrent_icebergs": "max_concurrent_icebergs",
}


@dataclass(frozen=True)
class CapacityEnvelope:
    """Generation-side margins keeping a draw clear of the engine's hard ceilings.

    Four independent axes (there is no combined size knob). Each must be a
    positive ``int`` strictly below the matching ``OMIEMarketConfig()`` ceiling:
    ``max_resting_per_side`` < ``book_depth``, ``max_trades`` < ``nTrades`` (per
    contract), ``max_hops`` < ``max_match_iterations`` (per submission) and
    ``max_concurrent_icebergs`` < ``max_icebergs``.
    """

    max_resting_per_side: int = 80
    max_trades: int = 80
    max_hops: int = 160
    max_concurrent_icebergs: int = 800

    def __post_init__(self) -> None:
        market = OMIEMarketConfig()
        for f in fields(self):
            value = getattr(self, f.name)
            ceiling = int(getattr(market, _ENVELOPE_CEILING_FIELD[f.name]))
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{f.name} must be an int, got {value!r}")
            if not 0 < value < ceiling:
                raise ValueError(
                    f"{f.name}={value} must satisfy 0 < value < {ceiling} "
                    f"(OMIEMarketConfig().{_ENVELOPE_CEILING_FIELD[f.name]})"
                )


DEFAULT_ENVELOPE = CapacityEnvelope()


class _C14CeilingOptOut:
    """Type of the :data:`C14_CEILING_OPT_OUT` sentinel (singleton)."""

    _instance: Optional["_C14CeilingOptOut"] = None

    def __new__(cls) -> "_C14CeilingOptOut":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "C14_CEILING_OPT_OUT"


#: The single explicit opt-out from the I8 envelope. Accepted by
#: :func:`check_envelope`; refused (``ValueError``) by every mode of this module, so
#: only C14's future ceiling-reaching modes can use it. ``None`` is never an opt-out.
C14_CEILING_OPT_OUT = _C14CeilingOptOut()


class CapacityBreach(ValueError):
    """A draw's analytic bound exceeds its envelope on ``axis``.

    Attributes:
        axis: one of ``resting_per_side``, ``trades``, ``hops``,
            ``concurrent_icebergs``.
        value: the draw's bound on that axis.
        limit: the envelope's (inclusive) limit on that axis.
    """

    def __init__(self, axis: str, value: int, limit: int, detail: str = "") -> None:
        self.axis = axis
        self.value = value
        self.limit = limit
        msg = f"capacity bound {axis}={value} exceeds the envelope limit {limit}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


def check_envelope(
    envelope: Any,
    *,
    resting_per_side: int,
    trades: int,
    hops: int,
    concurrent_icebergs: int,
) -> None:
    """Refuse a draw whose analytic bounds exceed ``envelope`` (limits inclusive).

    Axes are checked in the order resting_per_side, trades, hops,
    concurrent_icebergs, and the first breach raises :class:`CapacityBreach`.
    ``C14_CEILING_OPT_OUT`` returns without checking; ``None`` (or anything that
    is not a :class:`CapacityEnvelope`) raises ``TypeError``.
    """
    if envelope is C14_CEILING_OPT_OUT:
        return
    if not isinstance(envelope, CapacityEnvelope):
        raise TypeError(
            f"envelope must be a CapacityEnvelope or C14_CEILING_OPT_OUT, got {envelope!r}"
        )
    values = {
        "resting_per_side": resting_per_side,
        "trades": trades,
        "hops": hops,
        "concurrent_icebergs": concurrent_icebergs,
    }
    for axis in AXES:
        limit = int(getattr(envelope, _AXIS_TO_FIELD[axis]))
        value = int(values[axis])
        if value > limit:
            raise CapacityBreach(axis, value, limit)


def _require_envelope(envelope: Any) -> CapacityEnvelope:
    """Mode-side guard: only a real envelope is accepted by the construction modes."""
    if envelope is C14_CEILING_OPT_OUT:
        raise ValueError(
            "C14_CEILING_OPT_OUT is reserved for C14's ceiling-reaching modes; "
            "this construction must stay inside the I8 envelope"
        )
    if not isinstance(envelope, CapacityEnvelope):
        raise TypeError(f"envelope must be a CapacityEnvelope, got {envelope!r}")
    return envelope


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------
def _canon(x: Any) -> Any:
    """Canonical JSON form: numpy scalars -> Python, tuples -> lists, maps -> dicts."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, Mapping):
        return {k: _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    return x


def _tape_bytes(tape: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(tape).astype("<i4")).tobytes()


def tape_digest(tape: np.ndarray) -> str:
    """blake2b-128 hex digest of a tape's little-endian int32 C-order bytes."""
    return hashlib.blake2b(_tape_bytes(tape), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# The Draw
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Draw:
    """One generated construction.

    Attributes:
        construction: the registry key (e.g. ``"C02.crossing_pair"``).
        seed: the integer seed, or ``None`` for seedless constructions.
        tape: ``(N, 13)`` C-contiguous int32 array; real rows first, then pad rows.
        params: drawn parameters, in domain units and their tick integers
            (JSON-serialisable). Euro / MW figures live here.
        expected: the construction-side expectation, integer ticks only (ints,
            bools and lists / str-keyed maps of these). Not the certification oracle.
        aux_tapes: named counterfactual tapes (e.g. ``"counterfactual"``).
        hop_bound: analytic upper bound on matching iterations of any submission.
        capacity_bounds: analytic upper bounds with exactly the keys
            ``resting_per_side``, ``trades``, ``hops`` (== ``hop_bound``) and
            ``concurrent_icebergs``.
        oracle_kind: ``exact-literal`` | ``exact-parametric`` | ``invariant`` |
            ``observation``.
        pending: expectations deliberately withheld (e.g. awaiting an OQ ruling).
        meta: free-form provenance.
    """

    construction: str
    seed: Optional[int]
    tape: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)
    aux_tapes: Mapping[str, np.ndarray] = field(default_factory=dict)
    hop_bound: int = 0
    capacity_bounds: Mapping[str, int] = field(default_factory=dict)
    oracle_kind: str = "exact-parametric"
    pending: Tuple[str, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)

    def n_real_rows(self) -> int:
        """Rows that are real orders (``order_id >= 1``), not the pad sentinel."""
        return len(_real_rows(self.tape))

    def digest(self) -> str:
        """Construction digest: blake2b-128 over tape, aux tapes (sorted) and canonical JSON."""
        h = hashlib.blake2b(digest_size=16)
        h.update(_tape_bytes(self.tape))
        for name in sorted(self.aux_tapes):
            h.update(name.encode("utf-8"))
            h.update(_tape_bytes(self.aux_tapes[name]))
        blob = json.dumps(
            {"params": _canon(self.params), "expected": _canon(self.expected)},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        h.update(blob.encode("utf-8"))
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Row construction (integer ticks)
# ---------------------------------------------------------------------------
def _one_of(name: str, float_value: Any, int_value: Any, encode: Callable[[float], int],
            *, required: bool) -> int:
    """Resolve a float/int argument pair to its tick integer (0 when both are absent)."""
    if float_value is not None and int_value is not None:
        raise ValueError(f"{name}: pass the float or the integer form, not both")
    if required and float_value is None and int_value is None:
        raise ValueError(f"{name}: one of the float or the integer form is required")
    if float_value is not None:
        return int(encode(float(float_value)))
    return 0 if int_value is None else int(int_value)


class TapeBuilder:
    """Accumulates rows in the 13-column int32 layout.

    Timestamps default to a counter starting at 1 with step 1; order ids start at
    1 and increase. For each of price, qty, peak and ppd, :meth:`add` takes either
    the float (``price_eur``, ``qty_mw``, ``peak_qty_mw``, ``ppd_eur``; encoded
    through ``encoding.py``) or the integer (``price_int``, ``qty_int``,
    ``peak_int``, ``ppd_int``; written verbatim), and exactly one of the two for
    price and qty.
    """

    def __init__(
        self,
        config: OMIEProductConfig = OMIEProductConfig(),
        *,
        contract_id: int = 0,
        first_timestamp: int = 1,
        timestamp_step: int = 1,
    ) -> None:
        self.config = config
        self.contract_id = int(contract_id)
        self._rows: List[List[int]] = []
        self._ts = int(first_timestamp)
        self._ts_step = int(timestamp_step)
        self._next_oid = 1

    def add(
        self,
        *,
        side: int,
        price_eur: Optional[float] = None,
        qty_mw: Optional[float] = None,
        price_int: Optional[int] = None,
        qty_int: Optional[int] = None,
        agent_id: int = 1,
        exec_restriction: int = ExecRestriction.NON,
        validity: int = Validity.GFS,
        expiry_time: int = 0,
        peak_qty_mw: Optional[float] = None,
        peak_int: Optional[int] = None,
        ppd_eur: Optional[float] = None,
        ppd_int: Optional[int] = None,
        timestamp: Optional[int] = None,
        order_id: Optional[int] = None,
        contract_id: Optional[int] = None,
    ) -> int:
        """Append one order row; returns its ``order_id``."""
        cfg = self.config
        p = _one_of("price", price_eur, price_int, lambda x: encode_price(x, cfg), required=True)
        q = _one_of("qty", qty_mw, qty_int, lambda x: encode_qty(x, cfg), required=True)
        pk = _one_of("peak", peak_qty_mw, peak_int, lambda x: encode_qty(x, cfg), required=False)
        pp = _one_of("ppd", ppd_eur, ppd_int, lambda x: encode_ppd(x, cfg), required=False)

        oid = self._next_oid if order_id is None else int(order_id)
        if order_id is None:
            self._next_oid += 1
        ts = self._ts if timestamp is None else int(timestamp)
        if timestamp is None:
            self._ts += self._ts_step

        row = [0] * N_COLS
        row[COL_CONTRACT] = self.contract_id if contract_id is None else int(contract_id)
        row[COL_SIDE] = int(side)
        row[COL_PRICE] = p
        row[COL_QTY] = q
        row[COL_OID] = oid
        row[COL_AGENT] = int(agent_id)
        row[COL_TS] = ts
        row[COL_EXEC] = int(exec_restriction)
        row[COL_VALIDITY] = int(validity)
        row[COL_EXPIRY] = int(expiry_time)
        row[COL_STATUS] = int(Status.ACTIVE)
        row[COL_PEAK] = pk
        row[COL_PPD] = pp
        self._rows.append(row)
        return int(oid)

    def row(self, oid: int) -> List[int]:
        """The row (a copy) of a previously added order id."""
        for r in self._rows:
            if r[COL_OID] == int(oid):
                return list(r)
        raise KeyError(oid)

    @property
    def last_timestamp(self) -> int:
        return int(self._rows[-1][COL_TS]) if self._rows else 0

    @property
    def n_rows(self) -> int:
        return len(self._rows)

    def build(self, pad_extra: int = 8) -> np.ndarray:
        """Materialise the tape, right-padded with ``pad_extra`` pad rows."""
        return pad_to(self._rows, len(self._rows) + pad_extra)


def pad_to(tape: np.ndarray, n_padded: int) -> np.ndarray:
    """Right-pad a tape to ``n_padded`` rows with the loader's :data:`PAD_ORDER_ROW`."""
    tape = np.asarray(tape, dtype=np.int32).reshape(-1, N_COLS)
    n_pad = int(n_padded) - tape.shape[0]
    if n_pad > 0:
        pad = np.tile(np.asarray(PAD_ORDER_ROW, dtype=np.int32), (n_pad, 1))
        tape = np.concatenate([tape, pad], axis=0)
    return np.ascontiguousarray(tape, dtype=np.int32)


def stack_tapes(tapes: Sequence[np.ndarray], pad_extra: int = 8) -> np.ndarray:
    """Stack variable-length tapes into ``(n_tapes, N_padded, 13)`` int32."""
    if not tapes:
        raise ValueError("stack_tapes requires at least one tape")
    n = max(int(np.asarray(t).reshape(-1, N_COLS).shape[0]) for t in tapes) + pad_extra
    return np.ascontiguousarray(np.stack([pad_to(t, n) for t in tapes], axis=0),
                                dtype=np.int32)


def _variant(kind: str, table: Mapping[str, Callable[..., "Draw"]], variant: str):
    """Look up a construction variant's builder; unknown names raise ``ValueError``."""
    if variant not in table:
        raise ValueError(f"unknown {kind} variant {variant!r}")
    return table[variant]


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed)))


def _snap_price(price_eur: float, config: OMIEProductConfig) -> float:
    """Snap to the price tick grid (archive-identical float path)."""
    return round(price_eur / config.price_tick) * config.price_tick


def _snap_qty(qty_mw: float, config: OMIEProductConfig) -> float:
    return round(qty_mw / config.qty_tick) * config.qty_tick


def _other(side: int) -> int:
    return _SELL if int(side) == _BUY else _BUY


def _toward(side_taking: int, ticks: int) -> int:
    """Signed tick offset that moves a limit *through* the book for ``side_taking``."""
    return int(ticks) if int(side_taking) == _BUY else -int(ticks)


def _deeper(rest_side: int, ticks: int) -> int:
    """Signed tick offset that makes a resting price worse for a taker."""
    return int(ticks) if int(rest_side) == _SELL else -int(ticks)


def _price_eur(price_int: int, config: OMIEProductConfig) -> float:
    return float(price_int * config.price_tick + config.price_min)


def _qty_mw(qty_int: int, config: OMIEProductConfig) -> float:
    return float(qty_int * config.qty_tick)


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


# ---------------------------------------------------------------------------
# Analytic capacity bounds
# ---------------------------------------------------------------------------
def _real_rows(tape: np.ndarray) -> np.ndarray:
    tape = np.asarray(tape).reshape(-1, N_COLS)
    return tape[tape[:, COL_OID] >= 1]


def _units(row) -> int:
    """1 per non-iceberg order; ``ceil(total / peak)`` per iceberg."""
    if int(row[COL_EXEC]) == _ICE:
        return _ceil_div(int(row[COL_QTY]), max(1, int(row[COL_PEAK])))
    return 1


def _reach_price(row) -> int:
    """The most aggressive price any slice of the order can carry."""
    p = int(row[COL_PRICE])
    if int(row[COL_EXEC]) == _ICE and int(row[COL_PPD]) != 0:
        last = p + (_units(row) - 1) * int(row[COL_PPD])
        return max(p, last) if int(row[COL_SIDE]) == _BUY else min(p, last)
    return p


def _crossable(a, b) -> bool:
    """Opposite-side orders ``a`` and ``b`` can trade by price."""
    if int(a[COL_SIDE]) == int(b[COL_SIDE]):
        return False
    buy, sell = (a, b) if int(a[COL_SIDE]) == _BUY else (b, a)
    return _reach_price(buy) >= _reach_price(sell)


def _row_is_rejectable(row, config: OMIEProductConfig = OMIEProductConfig()) -> bool:
    """Spec-derived: the market rejects ``row`` at submission.

    Implements the ``market-spec.md`` §5.4 conditions the generator can produce:
    side not BUY/SELL; ``price_int`` outside ``[encode(price_min), encode(price_max)]``;
    ``qty_int`` outside ``[encode(qty_min), encode(qty_max)]``; and, for ICE rows
    only (the validator gates them the same way), peak <= 0, peak > qty, and the
    §13.6 PPD sign rule (BUY needs PPD <= 0, SELL needs PPD >= 0). Off-tick is N/A
    (the tape is already in integer ticks). Deliberately does **not** import the
    engine's ``order_validator`` (generator independence). Anything not listed
    here returns False, so an uncertain row is counted and the bounds stay upper
    bounds.
    """
    side = int(row[COL_SIDE])
    if side not in (_BUY, _SELL):
        return True
    price, qty = int(row[COL_PRICE]), int(row[COL_QTY])
    if not encode_price(config.price_min, config) <= price <= encode_price(config.price_max, config):
        return True
    if not encode_qty(config.qty_min, config) <= qty <= encode_qty(config.qty_max, config):
        return True
    if int(row[COL_EXEC]) == _ICE:
        peak, ppd = int(row[COL_PEAK]), int(row[COL_PPD])
        if peak <= 0 or peak > qty:
            return True
        if (side == _BUY and ppd > 0) or (side == _SELL and ppd < 0):
            return True
    return False


def tape_capacity_bounds(tape: np.ndarray) -> Dict[str, int]:
    """Sound upper bounds for replaying ``tape``'s real rows on one contract.

    Rows the market rejects at submission (``_row_is_rejectable`` under the
    default ``OMIEProductConfig()``) are excluded from all four bounds.

    * ``resting_per_side``: NON or ICE orders submitted on a side (never decremented).
    * ``trades``: units of the orders that can trade with some opposite order by
      price; every trade exhausts at least one unit.
    * ``hops``: max over submissions of (units of earlier opposite orders it can
      cross + its own units, + 2 only when the order is crossable with at least
      one other counted row, earlier or later).
    * ``concurrent_icebergs``: ICE orders submitted.
    """
    real = _real_rows(tape)
    real = real[np.array([not _row_is_rejectable(r) for r in real], dtype=bool)]
    n = len(real)
    rests = [int(r[COL_EXEC]) in (_NON, _ICE) for r in real]
    per_side = {_BUY: 0, _SELL: 0}
    for r, rest in zip(real, rests):
        if rest:
            per_side[int(r[COL_SIDE])] += 1
    units = [_units(r) for r in real]
    trades = 0
    hops = 0
    for i in range(n):
        has_counterparty = any(_crossable(real[i], real[j]) for j in range(n) if j != i)
        if has_counterparty:
            trades += units[i]
        opp = sum(units[j] for j in range(i) if _crossable(real[i], real[j]))
        hops = max(hops, opp + units[i] + (2 if has_counterparty else 0))
    icebergs = int(sum(1 for r in real if int(r[COL_EXEC]) == _ICE))
    return {
        "resting_per_side": int(max(per_side.values())),
        "trades": int(trades),
        "hops": int(hops),
        "concurrent_icebergs": icebergs,
    }


def _oracle_kind(construction: str) -> str:
    if construction.startswith("C00."):
        return "exact-literal"
    if construction == "C09.icewpi_wrong_sign":
        return "observation"
    if construction.startswith("C15."):
        return "invariant"
    return "exact-parametric"


def _finish(
    construction: str,
    seed: Optional[int],
    tape: np.ndarray,
    params: Dict[str, Any],
    expected: Dict[str, Any],
    envelope: Any,
    *,
    aux_tapes: Optional[Mapping[str, np.ndarray]] = None,
    bounds: Optional[Mapping[str, int]] = None,
    pending: Tuple[str, ...] = (),
    meta: Optional[Dict[str, Any]] = None,
) -> Draw:
    """Compute (or take) the bounds, check them against ``envelope``, build the Draw.

    ``envelope=_NO_ENVELOPE`` is used only by the bookless constructions
    (``C00.empty``, ``C00.pad_only``, ``C01.validation``).
    """
    cb = dict(tape_capacity_bounds(tape) if bounds is None else bounds)
    if envelope is not _NO_ENVELOPE:
        check_envelope(envelope, **cb)
    tape = np.ascontiguousarray(np.asarray(tape, dtype=np.int32))
    aux = {k: np.ascontiguousarray(np.asarray(v, dtype=np.int32))
           for k, v in (aux_tapes or {}).items()}
    return Draw(
        construction=construction,
        seed=None if seed is None else int(seed),
        tape=tape,
        params=params,
        expected=expected,
        aux_tapes=types.MappingProxyType(aux),
        hop_bound=int(cb["hops"]),
        capacity_bounds=types.MappingProxyType({a: int(cb[a]) for a in AXES}),
        oracle_kind=_oracle_kind(construction),
        pending=tuple(pending),
        meta=dict(meta or {}),
    )


_NO_ENVELOPE = object()


# ---------------------------------------------------------------------------
# C00 — inertness and the status gate
# ---------------------------------------------------------------------------
def make_empty() -> Draw:
    """A truly empty ``(0, 13)`` tape: zero rows, zero real orders."""
    return _finish(
        "C00.empty", None, np.zeros((0, N_COLS), dtype=np.int32),
        params={}, expected={"n_trades": 0, "n_resting": 0}, envelope=_NO_ENVELOPE,
    )


def make_pad_only(n_pad: int = 64) -> Draw:
    """A tape made entirely of ``n_pad`` pad rows (:data:`PAD_ORDER_ROW`)."""
    tape = pad_to(np.zeros((0, N_COLS), dtype=np.int32), int(n_pad))
    return _finish(
        "C00.pad_only", None, tape,
        params={"n_pad": int(n_pad)}, expected={"n_trades": 0, "n_resting": 0},
        envelope=_NO_ENVELOPE,
    )


def make_status_gate(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
) -> Draw:
    """The ``C02.crossing_pair`` tape for ``seed``, to be submitted to a contract in
    each of PENDING, SUSPENDED and CLOSED. Expected: 0 trades, 0 accepted.

    The bounds are the crossing pair's (sound for an OPEN replay too).
    """
    base = make_crossing_pair(seed, config, envelope=envelope)
    params = dict(base.params)
    params["contract_statuses"] = [int(ContractStatus.PENDING),
                                   int(ContractStatus.SUSPENDED),
                                   int(ContractStatus.CLOSED)]
    return _finish(
        "C00.status_gate", seed, base.tape, params,
        {"n_trades": 0, "n_accepted": 0}, envelope,
        bounds=base.capacity_bounds,
    )


# ---------------------------------------------------------------------------
# C01 — validation
# ---------------------------------------------------------------------------
def make_validation_cases(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
) -> Draw:
    """One order per validation class, with its accept/reject verdict.

    Classes: in-bounds on-tick; price and qty each below and above bound;
    sub-half-tick off-grid (accepted, snapped); over-half-tick off-grid (commits to
    the next tick, in bounds here); iceberg peak > qty and peak == 0 (rejected);
    and the near-bound pair at ``price_max``: a sub-half-tick offset
    (drawn in [0.05, 0.45] ticks; accepted, snapped to ``price_max``) and a
    >= half-tick offset (drawn in [0.55, 0.95] ticks; commits to ``price_max`` + 1
    tick, rejected). The exact half tick is excluded: it is a float tie that
    cannot be represented, and is recorded as not certified.

    There is no book, so the tape is empty and the cases travel in
    ``params["cases"]``. ``expected["verdicts"]`` is computed from the encoded
    integers (the validator snaps to the tick grid, then checks bounds).
    """
    rng = _rng(seed)
    tick = config.price_tick
    p_ok = _snap_price(float(rng.uniform(20.0, 120.0)), config)
    q_ok = _snap_qty(float(rng.uniform(1.0, 50.0)), config)
    sub_off = float(rng.uniform(0.1, 0.4))
    over_off = float(rng.uniform(0.6, 0.9))
    near_sub = float(rng.uniform(0.05, 0.45))
    near_over = float(rng.uniform(0.55, 0.95))
    ice = int(ExecRestriction.ICE)

    cases: List[Dict[str, Any]] = [
        {"name": "in_bounds_on_tick", "price_eur": p_ok, "qty_mw": q_ok},
        {"name": "price_below_min", "price_eur": config.price_min - 1.0, "qty_mw": q_ok},
        {"name": "price_above_max", "price_eur": config.price_max + 1.0, "qty_mw": q_ok},
        {"name": "qty_below_min", "price_eur": p_ok, "qty_mw": config.qty_min * 0.3},
        {"name": "qty_above_max", "price_eur": p_ok, "qty_mw": config.qty_max + 1.0},
        {"name": "sub_half_tick_off_grid", "price_eur": p_ok + sub_off * tick,
         "qty_mw": q_ok, "offset_ticks": sub_off},
        {"name": "over_half_tick_off_grid", "price_eur": p_ok + over_off * tick,
         "qty_mw": q_ok, "offset_ticks": over_off},
        {"name": "iceberg_peak_exceeds_qty", "price_eur": p_ok, "qty_mw": q_ok,
         "peak_qty_mw": q_ok + 1.0, "exec_restriction": ice},
        {"name": "iceberg_peak_zero", "price_eur": p_ok, "qty_mw": q_ok,
         "peak_qty_mw": 0.0, "exec_restriction": ice},
        {"name": "near_max_sub_half_tick",
         "price_eur": config.price_max + near_sub * tick, "qty_mw": q_ok,
         "offset_ticks": near_sub},
        {"name": "near_max_over_half_tick",
         "price_eur": config.price_max + near_over * tick, "qty_mw": q_ok,
         "offset_ticks": near_over},
    ]

    price_max_int = encode_price(config.price_max, config)
    qty_min_int = encode_qty(config.qty_min, config)
    qty_max_int = encode_qty(config.qty_max, config)
    verdicts: Dict[str, bool] = {}
    snapped: Dict[str, int] = {}
    for c in cases:
        price_int = encode_price(float(c["price_eur"]), config)
        qty_int = encode_qty(float(c["qty_mw"]), config)
        c["price_int"] = int(price_int)
        c["qty_int"] = int(qty_int)
        ok = 0 <= price_int <= price_max_int and qty_min_int <= qty_int <= qty_max_int
        if int(c.get("exec_restriction", _NON)) == _ICE:
            peak_int = encode_qty(float(c.get("peak_qty_mw", 0.0)), config)
            c["peak_int"] = int(peak_int)
            ok = ok and 0 < peak_int <= qty_int
        verdicts[c["name"]] = bool(ok)
        if ok:
            snapped[c["name"]] = int(price_int)

    return _finish(
        "C01.validation", seed, np.zeros((0, N_COLS), dtype=np.int32),
        params={"cases": cases, "p_ok_eur": p_ok, "q_ok_mw": q_ok,
                "not_certified": ["exact half-tick offset (float tie)"]},
        expected={"verdicts": verdicts, "snapped_price_int": snapped},
        envelope=_NO_ENVELOPE,
        meta={"not_covered": [
            "§5.4 agent guarantee / operating limit (unimplemented)",
            "§5.4 TSO unavailability check (unimplemented)",
        ]},
    )


# ---------------------------------------------------------------------------
# C02 — single crossing pair (archive-identical tape)
# ---------------------------------------------------------------------------
def _crossing_pair_parts(seed: int, config: OMIEProductConfig):
    rng = _rng(seed)
    rest_side = Side.SELL if rng.random() < 0.5 else Side.BUY
    agg_side = Side.BUY if rest_side == Side.SELL else Side.SELL
    p_rest = _snap_price(float(rng.uniform(20.0, 120.0)), config)
    gap_ticks = int(rng.integers(1, 51))
    gap = gap_ticks * config.price_tick
    p_agg = _snap_price(p_rest + gap if agg_side == Side.BUY else p_rest - gap, config)
    q_rest = _snap_qty(float(rng.uniform(1.0, 60.0)), config)
    q_agg = _snap_qty(float(rng.uniform(1.0, 60.0)), config)

    b = TapeBuilder(config)
    rest_oid = b.add(side=rest_side, price_eur=p_rest, qty_mw=q_rest, agent_id=1)
    agg_oid = b.add(side=agg_side, price_eur=p_agg, qty_mw=q_agg, agent_id=2)
    rr, ar = b.row(rest_oid), b.row(agg_oid)
    q_rest_int, q_agg_int = rr[COL_QTY], ar[COL_QTY]
    params = {
        "rest_side": int(rest_side), "agg_side": int(agg_side),
        "rest_oid": rest_oid, "agg_oid": agg_oid,
        "p_rest_int": rr[COL_PRICE], "p_agg_int": ar[COL_PRICE],
        "gap_ticks": gap_ticks, "q_rest_int": q_rest_int, "q_agg_int": q_agg_int,
        "p_rest_eur": p_rest, "p_agg_eur": p_agg, "q_rest_mw": q_rest, "q_agg_mw": q_agg,
    }
    expected = {
        "n_trades": 1,
        "trade_price_int": rr[COL_PRICE],
        "trade_qty_int": min(q_rest_int, q_agg_int),
        "residual_qty_int": abs(q_agg_int - q_rest_int),
    }
    return b.build(), params, expected


def make_crossing_pair(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
) -> Draw:
    """One resting NON order, one NON aggressor priced through it by ``gap_ticks``.

    Expected: one trade of ``min(q_rest, q_agg)`` at the resting price (§6.2).
    """
    _require_envelope(envelope)
    tape, params, expected = _crossing_pair_parts(seed, config)
    return _finish("C02.crossing_pair", seed, tape, params, expected, envelope)


# ---------------------------------------------------------------------------
# C03 — price-time priority (archive-identical ladder) and same owner
# ---------------------------------------------------------------------------
def make_priority_ladder(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
    variant: str = "ladder",
) -> Draw:
    """C03 price-time priority constructions.

    * ``variant="ladder"`` (``C03.priority_ladder``, archive-identical): ``k``
      same-price resting orders with increasing timestamps, plus worse decoys
      submitted *after* them. The aggressor is sized to consume exactly ``j < k``
      of them. Expected: the ``j`` earliest order ids fill, the rest and the
      decoys are untouched.
    * ``variant="price_over_time"`` (``C03.price_over_time``): the worse-priced
      decoys are submitted *before* the touch orders and are reachable by the
      aggressor's limit, so a time-only ranker disagrees on every draw.

    An unknown ``variant`` raises ``ValueError``.
    """
    table = {"ladder": _priority_ladder, "price_over_time": _price_over_time}
    return _variant("priority-ladder", table, variant)(seed, config, envelope)


def _priority_ladder(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    _require_envelope(envelope)
    rng = _rng(seed)
    k = int(rng.integers(2, 7))
    j = int(rng.integers(1, k))
    n_decoys = int(rng.integers(1, 4))
    rest_side = Side.SELL if rng.random() < 0.5 else Side.BUY
    agg_side = Side.BUY if rest_side == Side.SELL else Side.SELL
    p_touch = _snap_price(float(rng.uniform(30.0, 90.0)), config)
    q_each = _snap_qty(float(rng.uniform(2.0, 20.0)), config)

    b = TapeBuilder(config)
    at_touch = [b.add(side=rest_side, price_eur=p_touch, qty_mw=q_each, agent_id=1)
                for _ in range(k)]
    decoys = []
    for d in range(1, n_decoys + 1):
        p_decoy = _snap_price(
            p_touch + d * 0.50 if rest_side == Side.SELL else p_touch - d * 0.50, config)
        decoys.append(b.add(side=rest_side, price_eur=p_decoy, qty_mw=q_each, agent_id=1))
    q_agg = _snap_qty(q_each * j, config)
    p_agg = _snap_price(p_touch + 1.0 if agg_side == Side.BUY else p_touch - 1.0, config)
    agg_oid = b.add(side=agg_side, price_eur=p_agg, qty_mw=q_agg, agent_id=2)

    touch_row, agg_row = b.row(at_touch[0]), b.row(agg_oid)
    p_touch_int, q_each_int = touch_row[COL_PRICE], touch_row[COL_QTY]
    params = {
        "k": k, "j": j, "n_decoys": n_decoys,
        "rest_side": int(rest_side), "agg_side": int(agg_side),
        "p_touch_int": p_touch_int, "q_each_int": q_each_int,
        "at_touch_oids": at_touch, "decoy_oids": decoys, "agg_oid": agg_oid,
        "q_agg_int": agg_row[COL_QTY], "p_agg_int": agg_row[COL_PRICE],
        "p_touch_eur": p_touch, "q_each_mw": q_each, "q_agg_mw": q_agg,
    }
    expected = {
        "filled_oids": at_touch[:j],
        "untouched_oids": at_touch[j:] + decoys,
        "n_trades": j,
        "trade_price_int": p_touch_int,
        "total_qty_int": j * q_each_int,
    }
    return _finish("C03.priority_ladder", seed, b.build(), params, expected, envelope)


def _price_over_time(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    """Price over time: ``n_early`` worse-priced decoys first, then ``k`` touch orders, then an
    aggressor whose limit reaches the deepest decoy. Price priority must fill the
    ``j`` earliest touch orders; a time-only ranker would take decoys first.
    """
    _require_envelope(envelope)
    rng = _rng(seed)
    k = int(rng.integers(2, 7))
    j = int(rng.integers(1, k))
    n_early = int(rng.integers(1, 4))
    rest_side = Side.SELL if rng.random() < 0.5 else Side.BUY
    agg_side = Side.BUY if rest_side == Side.SELL else Side.SELL
    p_touch = _snap_price(float(rng.uniform(30.0, 90.0)), config)
    q_each = _snap_qty(float(rng.uniform(2.0, 20.0)), config)
    s = int(rng.integers(1, 51))

    p_touch_int = int(encode_price(p_touch, config))
    q_each_int = int(encode_qty(q_each, config))
    decoy_price_ints = [p_touch_int + _deeper(rest_side, d * s)
                        for d in range(1, n_early + 1)]

    b = TapeBuilder(config)
    early = [b.add(side=rest_side, price_int=p, qty_int=q_each_int, agent_id=1)
             for p in decoy_price_ints]
    at_touch = [b.add(side=rest_side, price_int=p_touch_int, qty_int=q_each_int, agent_id=1)
                for _ in range(k)]
    q_agg_int = j * q_each_int
    p_agg_int = decoy_price_ints[-1]
    agg_oid = b.add(side=agg_side, price_int=p_agg_int, qty_int=q_agg_int, agent_id=2)

    params = {
        "k": k, "j": j, "n_early": n_early, "step_ticks": s,
        "rest_side": int(rest_side), "agg_side": int(agg_side),
        "p_touch_int": p_touch_int, "q_each_int": q_each_int,
        "early_decoy_oids": early, "decoy_price_ints": decoy_price_ints,
        "at_touch_oids": at_touch, "agg_oid": agg_oid,
        "q_agg_int": q_agg_int, "p_agg_int": p_agg_int,
        "p_touch_eur": p_touch, "q_each_mw": q_each,
    }
    expected = {
        "filled_oids": at_touch[:j],
        "untouched_oids": at_touch[j:] + early,
        "n_trades": j,
        "trade_price_int": p_touch_int,
        "total_qty_int": j * q_each_int,
    }
    return _finish("C03.price_over_time", seed, b.build(), params, expected, envelope)


def make_same_owner_pair(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
) -> Draw:
    """A crossing pair whose two sides carry the same agent id (§6.6: no
    self-trade prevention, so it must trade). ``aux_tapes["counterfactual"]`` is
    the distinct-owner ``C02.crossing_pair`` tape (relation M6).
    """
    base = make_crossing_pair(seed, config, envelope=envelope)
    owner = 7
    same = base.tape.copy()
    same[same[:, COL_OID] >= 1, COL_AGENT] = owner
    return _finish(
        "C03.same_owner_pair", seed, same,
        {**base.params, "owner_agent_id": owner},
        {**base.expected, "must_trade": True}, envelope,
        aux_tapes={"counterfactual": base.tape},
        meta={"relation": "M6"},
    )


# ---------------------------------------------------------------------------
# C04 — cancellation
# ---------------------------------------------------------------------------
def _cancel_ladder(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    rng = _rng(seed)
    k = int(rng.integers(2, 6))
    j = int(rng.integers(1, k))  # 1 <= j <= k - 1 survivors are taken
    n_decoys = int(rng.integers(1, 4))
    rest_side = _SELL if rng.random() < 0.5 else _BUY
    agg_side = _other(rest_side)
    p_touch = encode_price(_snap_price(float(rng.uniform(30.0, 90.0)), config), config)
    q_each = int(rng.integers(20, 201))
    victim_idx = int(rng.integers(0, k))
    through = int(rng.integers(1, 51))
    decoy_step = int(rng.integers(1, 51))

    b = TapeBuilder(config)
    resting = [b.add(side=rest_side, price_int=p_touch, qty_int=q_each, agent_id=1)
               for _ in range(k)]
    p_agg = p_touch + _toward(agg_side, through)
    decoys = []
    for d in range(1, n_decoys + 1):
        # Strictly beyond the aggressor's limit: unreachable by price.
        p_decoy = p_agg + _deeper(rest_side, d * decoy_step)
        decoys.append(b.add(side=rest_side, price_int=p_decoy, qty_int=q_each, agent_id=1))
    agg_oid = b.add(side=agg_side, price_int=p_agg, qty_int=j * q_each, agent_id=2)
    cancel_oid = resting[victim_idx]
    survivors = [o for o in resting if o != cancel_oid]
    params = {
        "k": k, "j": j, "n_decoys": n_decoys,
        "rest_side": rest_side, "agg_side": agg_side,
        "p_touch_int": p_touch, "q_each_int": q_each,
        "resting_oids": resting, "decoy_oids": decoys,
        "cancel_oid": cancel_oid, "agg_oid": agg_oid,
        "p_agg_int": p_agg, "q_agg_int": j * q_each,
        "p_touch_eur": _price_eur(p_touch, config), "q_each_mw": _qty_mw(q_each, config),
    }
    expected = {
        "survivor_oids_in_queue_order": survivors,
        "filled_oids": survivors[:j],
        "n_trades": j,
        "trades_from_cancel": 0,
        "trade_price_int": p_touch,
    }
    return _finish("C04.cancellation", seed, b.build(), params, expected, envelope,
                   meta={"cancel_before_row_of_oid": agg_oid})


def _cancel_iceberg_fok(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    rng = _rng(seed)
    rest_side = _SELL if rng.random() < 0.5 else _BUY
    fok_side = _other(rest_side)
    p_ice = encode_price(_snap_price(float(rng.uniform(30.0, 90.0)), config), config)
    peak = int(rng.integers(10, 81))
    n_slices = int(rng.integers(2, 21))
    rem = int(rng.integers(1, peak + 1))
    total = (n_slices - 1) * peak + rem
    n_vis = int(rng.integers(0, 4))
    vis_qtys = [int(rng.integers(10, 201)) for _ in range(n_vis)]
    v_vis = sum(vis_qtys)
    req = int(rng.integers(v_vis + 1, v_vis + total + 1))
    through = int(rng.integers(1, 51))

    b = TapeBuilder(config)
    ice_oid = b.add(side=rest_side, price_int=p_ice, qty_int=total, agent_id=1,
                    exec_restriction=ExecRestriction.ICE, peak_int=peak)
    visible = [b.add(side=rest_side, price_int=p_ice, qty_int=q, agent_id=3)
               for q in vis_qtys]
    p_fok = p_ice + _toward(fok_side, through)
    fok_oid = b.add(side=fok_side, price_int=p_fok, qty_int=req, agent_id=2,
                    exec_restriction=ExecRestriction.FOK)
    params = {
        "rest_side": rest_side, "ice_oid": ice_oid, "cancel_oid": ice_oid,
        "p_ice_int": p_ice, "total_int": total, "peak_int": peak,
        "n_slices": _ceil_div(total, peak), "n_vis": n_vis, "visible_oids": visible,
        "v_vis_int": v_vis, "fok_oid": fok_oid, "req_int": req, "p_fok_int": p_fok,
        "p_ice_eur": _price_eur(p_ice, config), "req_mw": _qty_mw(req, config),
    }
    expected = {"accepted": False, "filled_qty_int": 0, "book_unchanged_by_fok": True}
    return _finish("C04.iceberg_cancel_fok", seed, b.build(), params, expected, envelope,
                   meta={"cancel_before_row_of_oid": fok_oid})


def make_cancellation(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
    variant: str = "ladder",
) -> Draw:
    """Cancellation before an aggressor (the cancel is an engine call, not a row;
    it applies before the last real row).

    * ``"ladder"`` (``C04.cancellation``): ``k`` same-price resting orders plus
      ``n_decoys`` worse-priced decoys beyond the aggressor's limit; ``cancel_oid``
      is cancelled; the aggressor takes exactly ``j`` survivors. Expected: the
      survivors keep their queue order, the fill set is ``survivors[:j]``, the
      cancellation prints no trade.
    * ``"iceberg_fok"`` (``C04.iceberg_cancel_fok``): a resting iceberg plus
      ``n_vis`` visible NON orders at its price; the iceberg is cancelled; a FOK
      priced through asks ``req`` in ``[V_vis + 1, V_vis + total]``. Expected: not
      accepted, nothing filled, the book unchanged by the FOK.
    """
    _require_envelope(envelope)
    table = {"ladder": _cancel_ladder, "iceberg_fok": _cancel_iceberg_fok}
    return _variant("cancellation", table, variant)(seed, config, envelope)


# ---------------------------------------------------------------------------
# C05 — partial fill and the residual's time priority
# ---------------------------------------------------------------------------
def _residual_aggressor(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    # Archive-identical tape.
    rng = _rng(seed)
    rest_side = Side.SELL if rng.random() < 0.5 else Side.BUY
    agg_side = Side.BUY if rest_side == Side.SELL else Side.SELL
    p_rest = _snap_price(float(rng.uniform(30.0, 90.0)), config)
    q_rest = _snap_qty(float(rng.uniform(2.0, 20.0)), config)
    q_agg = _snap_qty(q_rest + float(rng.uniform(2.0, 30.0)), config)
    p_agg = _snap_price(p_rest + 0.50 if agg_side == Side.BUY else p_rest - 0.50, config)
    q_interloper = _snap_qty(float(rng.uniform(2.0, 20.0)), config)

    b = TapeBuilder(config)
    rest_oid = b.add(side=rest_side, price_eur=p_rest, qty_mw=q_rest, agent_id=1)
    agg_oid = b.add(side=agg_side, price_eur=p_agg, qty_mw=q_agg, agent_id=2)
    interloper_oid = b.add(side=agg_side, price_eur=p_agg, qty_mw=q_interloper, agent_id=3)
    residual_mw = _snap_qty(q_agg - q_rest, config)
    p_take = _snap_price(p_agg - 0.50 if agg_side == Side.BUY else p_agg + 0.50, config)
    take_oid = b.add(side=rest_side, price_eur=p_take, qty_mw=residual_mw, agent_id=4)

    rr, ar = b.row(rest_oid), b.row(agg_oid)
    q_rest_int, q_agg_int = rr[COL_QTY], ar[COL_QTY]
    params = {
        "rest_side": int(rest_side), "agg_side": int(agg_side),
        "rest_oid": rest_oid, "agg_oid": agg_oid, "interloper_oid": interloper_oid,
        "take_oid": take_oid, "residual_oid": agg_oid,
        "residual_original_timestamp": ar[COL_TS],
        "p_rest_int": rr[COL_PRICE], "p_agg_int": ar[COL_PRICE],
        "q_rest_int": q_rest_int, "q_agg_int": q_agg_int,
        "p_rest_eur": p_rest, "p_agg_eur": p_agg, "q_rest_mw": q_rest,
        "q_agg_mw": q_agg, "q_interloper_mw": q_interloper,
    }
    expected = {
        "first_fill_qty_int": q_rest_int,
        "residual_qty_int": q_agg_int - q_rest_int,
        "residual_price_int": ar[COL_PRICE],
        "residual_timestamp": ar[COL_TS],
        "second_fill_oid": agg_oid,
        "interloper_must_remain_unfilled": True,
    }
    return _finish("C05.aggressor_residual", seed, b.build(), params, expected, envelope,
                   meta={"ruling": "2026-09-23"})


def _residual_resting(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    # Plan variant (ii): the RESTING order is partially filled.
    rng = _rng(seed)
    rest_side = _SELL if rng.random() < 0.5 else _BUY
    agg_side = _other(rest_side)
    p_rest = encode_price(_snap_price(float(rng.uniform(30.0, 90.0)), config), config)
    q_rest = int(rng.integers(30, 301))
    q_agg = int(rng.integers(1, q_rest))  # strictly smaller: the resting order keeps a residual
    through = int(rng.integers(1, 51))
    q_interloper = int(rng.integers(10, 201))
    take_through = int(rng.integers(0, 51))

    b = TapeBuilder(config)
    rest_oid = b.add(side=rest_side, price_int=p_rest, qty_int=q_rest, agent_id=1)
    p_agg = p_rest + _toward(agg_side, through)
    agg_oid = b.add(side=agg_side, price_int=p_agg, qty_int=q_agg, agent_id=2)
    interloper_oid = b.add(side=rest_side, price_int=p_rest, qty_int=q_interloper,
                           agent_id=3)
    residual = q_rest - q_agg
    p_take = p_rest + _toward(agg_side, take_through)
    take_oid = b.add(side=agg_side, price_int=p_take, qty_int=residual, agent_id=4)

    rest_ts = b.row(rest_oid)[COL_TS]
    params = {
        "rest_side": rest_side, "agg_side": agg_side,
        "rest_oid": rest_oid, "agg_oid": agg_oid, "interloper_oid": interloper_oid,
        "take_oid": take_oid, "residual_oid": rest_oid,
        "residual_original_timestamp": rest_ts,
        "p_rest_int": p_rest, "p_agg_int": p_agg,
        "q_rest_int": q_rest, "q_agg_int": q_agg,
        "p_take_int": p_take, "q_interloper_int": q_interloper,
        "p_rest_eur": _price_eur(p_rest, config), "q_rest_mw": _qty_mw(q_rest, config),
        "q_agg_mw": _qty_mw(q_agg, config),
    }
    expected = {
        "first_fill_qty_int": q_agg,
        "residual_qty_int": residual,
        "residual_price_int": p_rest,
        "residual_timestamp": rest_ts,
        "second_fill_oid": rest_oid,
        "interloper_must_remain_unfilled": True,
    }
    return _finish("C05.resting_residual", seed, b.build(), params, expected, envelope,
                   meta={"ruling": "2026-09-23", "plan_variant": "ii"})


def make_partial_fill_residual(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
    variant: str = "aggressor",
) -> Draw:
    """A partial fill, an interloper at the residual's side and price, then a second
    aggressor sized exactly to the residual.

    * ``"aggressor"``: the aggressor exceeds the resting depth; its residual rests
      at its own limit carrying its original timestamp (archive-identical tape).
    * ``"resting"``: a smaller aggressor partially fills the resting order; the
      resting residual keeps its original timestamp.

    Expected (both): the second aggressor fills the residual (by its order id),
    the interloper stays unfilled.
    """
    _require_envelope(envelope)
    table = {"aggressor": _residual_aggressor, "resting": _residual_resting}
    return _variant("partial-fill", table, variant)(seed, config, envelope)


# ---------------------------------------------------------------------------
# C06 — multi-level walk (archive-identical tape for stress=False)
# ---------------------------------------------------------------------------
def make_multi_level_walk(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
    stress: bool = False,
) -> Draw:
    """A ladder of ``L`` levels swept by one aggressor across ``m`` whole levels plus
    a partial of level ``m``.

    Expected: the per-level fill vector, and the volume-weighted price as the
    integer numerator ``sum(price_int * qty_int)`` (no division). ``stress=True``
    places the draw past the int32 product boundary (~1.9e10) inside the product
    bounds.
    """
    _require_envelope(envelope)
    construction = "C06.multi_level_walk_stress" if stress else "C06.multi_level_walk"
    rng = _rng(seed)
    L = int(rng.integers(3, 13))
    m = int(rng.integers(1, L))
    rest_side = Side.SELL if rng.random() < 0.5 else Side.BUY
    agg_side = Side.BUY if rest_side == Side.SELL else Side.SELL
    p0 = _snap_price(float(rng.uniform(30.0, 90.0)), config)
    step = _snap_price(float(rng.uniform(0.05, 0.50)), config)
    if stress:
        L, m = 2, 1
        p0 = _snap_price(9000.0, config)
        qtys = [_snap_qty(config.qty_max, config) for _ in range(L)]
    else:
        qtys = [_snap_qty(float(rng.uniform(1.0, 30.0)), config) for _ in range(L)]

    b = TapeBuilder(config)
    prices, oids = [], []
    for i in range(L):
        p = _snap_price(p0 + i * step if rest_side == Side.SELL else p0 - i * step, config)
        prices.append(p)
        oids.append(b.add(side=rest_side, price_eur=p, qty_mw=qtys[i], agent_id=1))
    partial = _snap_qty(qtys[m] / 2.0, config) if m < L else 0.0
    q_agg = _snap_qty(sum(qtys[:m]) + partial, config)
    if q_agg > config.qty_max:
        # An aggressor is bound by qty_max (§2.3): trim the sweep to a legal order.
        q_agg = _snap_qty(config.qty_max, config)
        consumed, fills_cap = 0.0, 0
        for i, q in enumerate(qtys):
            if consumed + q > q_agg:
                break
            consumed += q
            fills_cap = i + 1
        m = max(1, fills_cap)
        partial = _snap_qty(q_agg - consumed, config) if q_agg > consumed else 0.0
    p_agg = prices[m] if m < L else prices[-1]
    agg_oid = b.add(side=agg_side, price_eur=p_agg, qty_mw=q_agg, agent_id=2)

    rows = [b.row(o) for o in oids]
    price_ints = [r[COL_PRICE] for r in rows]
    qty_ints = [r[COL_QTY] for r in rows]
    partial_int = encode_qty(partial, config) if partial > 0 else 0
    fills = [[price_ints[i], qty_ints[i]] for i in range(m)]
    if partial_int > 0:
        fills.append([price_ints[m], partial_int])
    agg_row = b.row(agg_oid)
    params = {
        "L": L, "m": m, "stress": bool(stress),
        "rest_side": int(rest_side), "agg_side": int(agg_side),
        "level_price_ints": price_ints, "level_qty_ints": qty_ints,
        "resting_oids": oids, "agg_oid": agg_oid,
        "q_agg_int": agg_row[COL_QTY], "p_agg_int": agg_row[COL_PRICE],
        "partial_qty_int": partial_int,
        "step_eur": step, "level_prices_eur": prices, "level_qtys_mw": qtys,
        "partial_mw": partial, "q_agg_mw": q_agg,
    }
    expected = {
        "per_level_fills": fills,
        "num_fills": len(fills),
        "total_qty_int": sum(q for _, q in fills),
        "weighted_price_int": sum(p * q for p, q in fills),
    }
    return _finish(construction, seed, b.build(), params, expected, envelope)


# ---------------------------------------------------------------------------
# C07 — IOC regimes (archive-identical tape)
# ---------------------------------------------------------------------------
_IOC_REGIMES = ("none", "partial", "full")


def make_ioc_regimes(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
    regime: Optional[str] = None,
) -> Draw:
    """An IOC aggressor in one of three liquidity regimes (no cross, partial, full).

    ``regime=None`` draws it from the stream (archive-identical); an explicit
    regime still consumes that draw so the rest of the stream is unchanged.
    ``aux_tapes["counterfactual"]`` is the same tape with the aggressor NON
    (relation M3: identical trade lists).
    """
    _require_envelope(envelope)
    if regime is not None and regime not in _IOC_REGIMES:
        raise ValueError(f"unknown IOC regime {regime!r}")
    rng = _rng(seed)
    drawn = _IOC_REGIMES[int(rng.integers(0, 3))]
    regime = drawn if regime is None else regime
    rest_side = Side.SELL if rng.random() < 0.5 else Side.BUY
    agg_side = Side.BUY if rest_side == Side.SELL else Side.SELL
    p_rest = _snap_price(float(rng.uniform(30.0, 90.0)), config)
    q_rest = _snap_qty(float(rng.uniform(5.0, 30.0)), config)
    if regime == "none":
        p_agg = _snap_price(p_rest - 1.0 if agg_side == Side.BUY else p_rest + 1.0, config)
        q_agg = _snap_qty(float(rng.uniform(1.0, 10.0)), config)
    else:
        p_agg = _snap_price(p_rest + 1.0 if agg_side == Side.BUY else p_rest - 1.0, config)
        if regime == "partial":
            q_agg = _snap_qty(q_rest + float(rng.uniform(2.0, 20.0)), config)
        else:
            q_agg = _snap_qty(max(config.qty_tick, q_rest / 2.0), config)

    def _build(restriction: ExecRestriction):
        b = TapeBuilder(config)
        r = b.add(side=rest_side, price_eur=p_rest, qty_mw=q_rest, agent_id=1)
        a = b.add(side=agg_side, price_eur=p_agg, qty_mw=q_agg, agent_id=2,
                  exec_restriction=restriction)
        return b, r, a

    b, rest_oid, agg_oid = _build(ExecRestriction.IOC)
    cf, _, _ = _build(ExecRestriction.NON)
    rr, ar = b.row(rest_oid), b.row(agg_oid)
    q_rest_int, q_agg_int = rr[COL_QTY], ar[COL_QTY]
    if regime == "none":
        traded, n_trades = 0, 0
    elif regime == "partial":
        traded, n_trades = q_rest_int, 1
    else:
        traded, n_trades = q_agg_int, 1
    params = {
        "regime": regime, "rest_side": int(rest_side), "agg_side": int(agg_side),
        "rest_oid": rest_oid, "agg_oid": agg_oid,
        "p_rest_int": rr[COL_PRICE], "p_agg_int": ar[COL_PRICE],
        "q_rest_int": q_rest_int, "q_agg_int": q_agg_int,
        "p_rest_eur": p_rest, "p_agg_eur": p_agg, "q_rest_mw": q_rest, "q_agg_mw": q_agg,
    }
    expected = {"traded_qty_int": traded, "n_trades": n_trades,
                "ioc_residual_rests": False}
    return _finish("C07.ioc_regimes", seed, b.build(), params, expected, envelope,
                   aux_tapes={"counterfactual": cf.build()}, meta={"relation": "M3"})


# ---------------------------------------------------------------------------
# C08 — iceberg cascade, Rule 3, incoming iceberg
# ---------------------------------------------------------------------------
def _iceberg_resting(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    # Archive-identical tape (the archive's rule3=False branch).
    rng = _rng(seed)
    peak = _snap_qty(float(rng.uniform(1.0, 10.0)), config)
    n_slices = int(rng.integers(2, 40))
    remainder = _snap_qty(float(rng.uniform(0.1, max(0.2, peak))), config)
    total = _snap_qty(peak * (n_slices - 1) + remainder, config)
    ice_side = Side.SELL if rng.random() < 0.5 else Side.BUY
    agg_side = Side.BUY if ice_side == Side.SELL else Side.SELL
    p_ice = _snap_price(float(rng.uniform(30.0, 90.0)), config)
    p_agg = _snap_price(p_ice + 2.0 if agg_side == Side.BUY else p_ice - 2.0, config)

    b = TapeBuilder(config)
    ice_oid = b.add(side=ice_side, price_eur=p_ice, qty_mw=total, agent_id=1,
                    exec_restriction=ExecRestriction.ICE, peak_qty_mw=peak)
    agg_oid = b.add(side=agg_side, price_eur=p_agg, qty_mw=total, agent_id=2)

    ir, ar = b.row(ice_oid), b.row(agg_oid)
    total_int, peak_int = ir[COL_QTY], ir[COL_PEAK]
    n = _ceil_div(total_int, peak_int)
    remainder_int = total_int - (n - 1) * peak_int
    params = {
        "ice_side": int(ice_side), "agg_side": int(agg_side),
        "ice_oid": ice_oid, "agg_oid": agg_oid,
        "p_ice_int": ir[COL_PRICE], "p_agg_int": ar[COL_PRICE],
        "total_int": total_int, "peak_int": peak_int, "n_slices": n,
        "remainder_int": remainder_int, "q_agg_int": ar[COL_QTY],
        "peak_mw": peak, "total_mw": total, "remainder_mw": remainder,
        "p_ice_eur": p_ice, "p_agg_eur": p_agg,
    }
    expected = {
        "slice_sizes_int": [peak_int] * (n - 1) + [remainder_int],
        "n_slices_consumed": n,
        "trade_price_int": ir[COL_PRICE],
        "truncated": False,
        "book_uncrossed_on_return": True,
    }
    return _finish("C08.iceberg_cascade", seed, b.build(), params, expected, envelope)


def _iceberg_rule3(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    # Rule 3: n >= 3, peak < q_agg <= total - peak, limit >= 1 tick through.
    rng = _rng(seed)
    peak = int(rng.integers(10, 101))
    n_slices = int(rng.integers(3, 40))
    rem = int(rng.integers(1, peak + 1))
    total = (n_slices - 1) * peak + rem
    q_agg = int(rng.integers(peak + 1, total - peak + 1))
    ice_side = _SELL if rng.random() < 0.5 else _BUY
    agg_side = _other(ice_side)
    p_ice = encode_price(_snap_price(float(rng.uniform(30.0, 90.0)), config), config)
    through = int(rng.integers(1, 201))
    p_agg = p_ice + _toward(agg_side, through)

    b = TapeBuilder(config)
    ice_oid = b.add(side=ice_side, price_int=p_ice, qty_int=total, agent_id=1,
                    exec_restriction=ExecRestriction.ICE, peak_int=peak)
    agg_oid = b.add(side=agg_side, price_int=p_agg, qty_int=q_agg, agent_id=2)
    full, part = divmod(q_agg, peak)
    params = {
        "ice_side": ice_side, "agg_side": agg_side,
        "ice_oid": ice_oid, "agg_oid": agg_oid,
        "p_ice_int": p_ice, "p_agg_int": p_agg,
        "total_int": total, "peak_int": peak, "n_slices": _ceil_div(total, peak),
        "remainder_int": total - (n_slices - 1) * peak, "q_agg_int": q_agg,
        "through_ticks": through,
        "p_ice_eur": _price_eur(p_ice, config), "total_mw": _qty_mw(total, config),
        "peak_mw": _qty_mw(peak, config), "q_agg_mw": _qty_mw(q_agg, config),
    }
    expected = {
        "slice_sizes_int": [peak] * full + ([part] if part else []),
        "filled_qty_int": q_agg,
        "first_trade_price_int": p_ice,
        "aggressor_rests": False,
        "iceberg_remaining_int": total - q_agg,
    }
    return _finish("C08.rule3", seed, b.build(), params, expected, envelope,
                   pending=("rule3_hop_trade_prices:OQ-31",),
                   meta={"open_question": "OQ-31"})


def _iceberg_incoming(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    # Incoming iceberg: walks L resting NON levels, slice by slice.
    rng = _rng(seed)
    L = int(rng.integers(2, 9))
    rest_side = _SELL if rng.random() < 0.5 else _BUY
    ice_side = _other(rest_side)
    p0 = encode_price(_snap_price(float(rng.uniform(30.0, 90.0)), config), config)
    step = int(rng.integers(1, 51))
    peak = int(rng.integers(10, 61))
    n_slices = int(rng.integers(2, 31))
    rem = int(rng.integers(1, peak + 1))
    total = (n_slices - 1) * peak + rem
    extra = int(rng.integers(1, 101))
    depth = total + extra  # >= total + 1 tick
    cuts = sorted(int(c) for c in rng.choice(np.arange(1, depth), size=L - 1, replace=False))
    bounds_ = [0] + cuts + [depth]
    level_qtys = [bounds_[i + 1] - bounds_[i] for i in range(L)]
    level_prices = [p0 + _deeper(rest_side, i * step) for i in range(L)]

    # Two-pointer merge of the slice partition with the level partition.
    slices = [peak] * (n_slices - 1) + [rem]
    lev = list(level_qtys)
    trades: List[List[int]] = []
    i = j = deepest = 0
    while i < len(slices) and j < len(lev):
        t = min(slices[i], lev[j])
        trades.append([level_prices[j], t])
        slices[i] -= t
        lev[j] -= t
        deepest = j
        if slices[i] == 0:
            i += 1
        if lev[j] == 0:
            j += 1
    p_ice = level_prices[deepest]

    b = TapeBuilder(config)
    oids = [b.add(side=rest_side, price_int=p, qty_int=q, agent_id=1)
            for p, q in zip(level_prices, level_qtys)]
    ice_oid = b.add(side=ice_side, price_int=p_ice, qty_int=total, agent_id=2,
                    exec_restriction=ExecRestriction.ICE, peak_int=peak)
    params = {
        "rest_side": rest_side, "ice_side": ice_side, "L": L,
        "level_price_ints": level_prices, "level_qty_ints": level_qtys,
        "resting_oids": oids, "ice_oid": ice_oid, "p_ice_int": p_ice,
        "total_int": total, "peak_int": peak, "n_slices": n_slices,
        "step_ticks": step, "deepest_level": deepest,
        "total_mw": _qty_mw(total, config), "peak_mw": _qty_mw(peak, config),
    }
    expected = {"trades_int": trades, "reported_qty_int": total, "iceberg_rests": False}
    tape = b.build()
    cb = tape_capacity_bounds(tape)
    cb["trades"] = n_slices + L - 1
    cb["hops"] = n_slices + L + 2
    return _finish("C08.incoming_iceberg", seed, tape, params, expected, envelope,
                   bounds=cb)


def _iceberg_incoming_remainder(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    # Incoming remainder: an incoming standard iceberg (PPD 0) whose total
    # crossing depth M is strictly below its total and NOT a multiple of its
    # peak, so the sweep ends inside instance k_e = M // P (u = M - k_e P > 0)
    # and a remainder R = total - M > 0 rests (market-spec §4.2.1 Rule 2,
    # *Incoming iceberg*, item (4), amended 2026-10-20). Own function and own
    # stream: the draws of every other mode are unchanged.
    rng = _rng(seed)
    L = int(rng.integers(2, 9))
    rest_side = _SELL if rng.random() < 0.5 else _BUY
    ice_side = _other(rest_side)
    p0 = encode_price(_snap_price(float(rng.uniform(30.0, 90.0)), config), config)
    step = int(rng.integers(1, 51))
    peak = int(rng.integers(10, 61))
    k_e = int(rng.integers(0, 30))
    u = int(rng.integers(max(1, L - k_e * peak), peak))  # 1 <= u <= P - 1; M >= L
    depth = k_e * peak + u
    rem = int(rng.integers(max(1, peak - depth), 3 * peak + 1))  # total >= P
    total = depth + rem
    cuts = sorted(int(c) for c in rng.choice(np.arange(1, depth), size=L - 1, replace=False))
    bounds_ = [0] + cuts + [depth]
    level_qtys = [bounds_[i + 1] - bounds_[i] for i in range(L)]
    level_prices = [p0 + _deeper(rest_side, i * step) for i in range(L)]
    p_ice = level_prices[-1]  # every level crosses; the sweep takes the whole depth

    # Two-pointer merge of the swept instance partition with the level partition.
    inst = [peak] * k_e + [u]
    lev = list(level_qtys)
    trades: List[List[int]] = []
    i = j = 0
    while i < len(inst) and j < len(lev):
        t = min(inst[i], lev[j])
        trades.append([level_prices[j], t])
        inst[i] -= t
        lev[j] -= t
        if inst[i] == 0:
            i += 1
        if lev[j] == 0:
            j += 1
    visible = min(peak - u, rem)

    b = TapeBuilder(config)
    oids = [b.add(side=rest_side, price_int=p, qty_int=q, agent_id=1)
            for p, q in zip(level_prices, level_qtys)]
    ice_oid = b.add(side=ice_side, price_int=p_ice, qty_int=total, agent_id=2,
                    exec_restriction=ExecRestriction.ICE, peak_int=peak)
    params = {
        "rest_side": rest_side, "ice_side": ice_side, "L": L,
        "level_price_ints": level_prices, "level_qty_ints": level_qtys,
        "resting_oids": oids, "ice_oid": ice_oid, "p_ice_int": p_ice,
        "total_int": total, "peak_int": peak, "ppd_int": 0,
        "n_slices": _ceil_div(total, peak), "step_ticks": step,
        "total_mw": _qty_mw(total, config), "peak_mw": _qty_mw(peak, config),
    }
    expected = {
        "trades_int": trades,
        "swept_qty_int": depth,
        "rest_instance": k_e,
        "used_in_rest_instance_int": u,
        "remainder_int": rem,
        "visible_int": visible,
        "hidden_int": rem - visible,
        "rest_price_int": p_ice,
        "iceberg_rests": True,
    }
    tape = b.build()
    cb = tape_capacity_bounds(tape)
    # Rows split at resting entries and at instance boundaries; instances
    # 0..k_e are touched, so at most L + k_e rows. Hops: one pass per level or
    # instance boundary, with the +2 slack of C08.incoming_iceberg.
    cb["trades"] = L + k_e
    cb["hops"] = L + k_e + 3
    return _finish("C08.incoming_remainder", seed, tape, params, expected, envelope,
                   bounds=cb)


def make_iceberg_cascade(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
    variant: str = "resting",
) -> Draw:
    """Iceberg constructions (§4.2.1).

    * ``"resting"`` (``C08.iceberg_cascade``, archive-identical tape): a resting
      iceberg consumed whole by one aggressor in ``ceil(total/peak)`` slices.
    * ``"rule3"`` (``C08.rule3``): ``n_slices >= 3``,
      ``peak < q_agg <= total - peak``, limit at least one tick through, so the
      regenerated slice must re-cross. The hop trade prices are withheld pending
      OQ-31 (``pending``).
    * ``"incoming"`` (``C08.incoming_iceberg``): ``L`` resting NON levels
      with depth >= total + 1 tick; the iceberg's limit is the deepest level it
      needs. Expected: the two-pointer merge of the slice and level partitions,
      each trade at its level's price; nothing rests.
    * ``"incoming_remainder"`` (``C08.incoming_remainder``;
      not in :data:`CONSTRUCTIONS`, whose key set is pinned by the test suite):
      ``L`` in 2..8 resting NON levels on a random side at off-round prices, and
      an incoming standard iceberg (PPD 0, peak ``P`` in 10..60) whose limit is
      the deepest level, so every level crosses. The total depth ``M`` is
      strictly below the iceberg's total and ``M mod P != 0``:
      ``k_e = M // P`` in 0..29, ``u = M - k_e P`` in ``1..P-1``, remainder
      ``R = total - M`` in ``1..3P`` (and ``total >= P``). This exercises
      §4.2.1 Rule 2 *Incoming iceberg* item (4) (amended 2026-10-20): the
      remainder rests as instance ``k_e`` at ``b`` and shows ``min(P - u, R)``.
      Params: as ``"incoming"`` minus ``deepest_level`` (always ``L - 1``) plus
      ``ppd_int`` (0); ``params`` carries no swept or remainder figure.
      Construction-side ``expected``: ``trades_int`` (the merge of the instance
      partition ``[P]*k_e + [u]`` with the level partition), ``swept_qty_int``,
      ``rest_instance``, ``used_in_rest_instance_int``, ``remainder_int``,
      ``visible_int``, ``hidden_int``, ``rest_price_int``, ``iceberg_rests``.
    """
    _require_envelope(envelope)
    table = {"resting": _iceberg_resting, "rule3": _iceberg_rule3,
             "incoming": _iceberg_incoming,
             "incoming_remainder": _iceberg_incoming_remainder}
    return _variant("iceberg-cascade", table, variant)(seed, config, envelope)


# ---------------------------------------------------------------------------
# C09 — price-incremented iceberg (ICEwPI) drift
# ---------------------------------------------------------------------------
def make_icewpi_drift(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
    wrong_sign: bool = False,
) -> Draw:
    """A price-incremented iceberg crossed for exactly ``m`` slices.

    ``n_slices >= 3``, ``total = n_slices * peak``, ``m`` in ``[1, n_slices - 2]``;
    the aggressor asks ``m * peak + e`` with ``e`` in ``[1, peak - 1]`` ticks and
    its limit is exactly ``base + (m - 1) * ppd``. Expected: trade prices
    ``base + i * ppd`` for ``i < m``, stop at slice ``m``, the aggressor's residual
    ``e`` rests at its limit, hidden remaining ``(n - m - 1) * peak``.

    The ppd sign rule (§13.6: sell > 0, buy < 0) is respected; its magnitude range
    here is a test envelope, never a bound. ``wrong_sign=True`` (
    ``C09.icewpi_wrong_sign``) emits the same draw with the ppd sign flipped;
    ``oracle_kind="observation"`` and ``expected`` is empty.
    """
    _require_envelope(envelope)
    rng = _rng(seed)
    ice_side = _SELL if rng.random() < 0.5 else _BUY
    agg_side = _other(ice_side)
    ppd_ticks = int(rng.integers(1, 51))
    ppd = ppd_ticks if ice_side == _SELL else -ppd_ticks  # §13.6 sign rule
    base = encode_price(_snap_price(float(rng.uniform(30.0, 90.0)), config), config)
    peak = int(rng.integers(10, 81))
    n_slices = int(rng.integers(3, 20))
    total = n_slices * peak
    m = int(rng.integers(1, n_slices - 1))  # [1, n - 2]
    e = int(rng.integers(1, peak))  # [1, peak - 1]
    q_agg = m * peak + e
    p_agg = base + (m - 1) * ppd

    emitted_ppd = -ppd if wrong_sign else ppd
    b = TapeBuilder(config)
    ice_oid = b.add(side=ice_side, price_int=base, qty_int=total, agent_id=1,
                    exec_restriction=ExecRestriction.ICE, peak_int=peak,
                    ppd_int=emitted_ppd)
    agg_oid = b.add(side=agg_side, price_int=p_agg, qty_int=q_agg, agent_id=2)
    params = {
        "ice_side": ice_side, "agg_side": agg_side, "ice_oid": ice_oid,
        "agg_oid": agg_oid, "base_int": base, "ppd_int": emitted_ppd,
        "ppd_ticks": ppd_ticks, "peak_int": peak, "total_int": total,
        "n_slices": n_slices, "m": m, "e_int": e, "q_agg_int": q_agg, "p_agg_int": p_agg,
        "base_eur": _price_eur(base, config), "ppd_eur": emitted_ppd * config.price_tick,
        "peak_mw": _qty_mw(peak, config), "total_mw": _qty_mw(total, config),
    }
    if wrong_sign:
        construction, expected = "C09.icewpi_wrong_sign", {}
    else:
        construction = "C09.icewpi_drift"
        expected = {
            "trade_price_sequence_int": [base + i * ppd for i in range(m)],
            "stop_index": m,
            "n_trades": m,
            "iceberg_rests_with_hidden": True,
            "residual_qty_int": e,
            "residual_price_int": p_agg,
            "hidden_remaining_int": (n_slices - m - 1) * peak,
        }
    return _finish(construction, seed, b.build(), params, expected, envelope,
                   meta={"sign_rule": "§13.6"})


# ---------------------------------------------------------------------------
# C10 — fill-or-kill boundary
# ---------------------------------------------------------------------------
def _fok_finish(construction: str, seed: int, tape: np.ndarray, params: Dict[str, Any],
                req: int, matchable: int, envelope) -> Draw:
    """All-or-nothing verdict: accepted iff ``req <= matchable`` (relation M4)."""
    accepted = req <= matchable
    expected = {"accepted": accepted, "filled_qty_int": req if accepted else 0,
                "matchable_qty_int": matchable}
    return _finish(construction, seed, tape, params, expected, envelope,
                   meta={"relation": "M4"})


def _fok_visible(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    rng = _rng(seed)
    rest_side = _SELL if rng.random() < 0.5 else _BUY
    fok_side = _other(rest_side)
    n_levels = int(rng.integers(1, 6))
    p0 = encode_price(_snap_price(float(rng.uniform(30.0, 90.0)), config), config)
    step = int(rng.integers(1, 21))
    qtys = [int(rng.integers(10, 151)) for _ in range(n_levels)]
    prices = [p0 + _deeper(rest_side, i * step) for i in range(n_levels)]
    delta = int(rng.integers(-1, 2))
    v = sum(qtys)
    req = v + delta
    b = TapeBuilder(config)
    for p, q in zip(prices, qtys):
        b.add(side=rest_side, price_int=p, qty_int=q, agent_id=1)
    p_fok = prices[-1]
    fok_oid = b.add(side=fok_side, price_int=p_fok, qty_int=req, agent_id=2,
                    exec_restriction=ExecRestriction.FOK)
    params = {
        "rest_side": rest_side, "n_levels": n_levels, "level_price_ints": prices,
        "level_qty_ints": qtys, "v_int": v, "delta_ticks": delta, "req_int": req,
        "fok_oid": fok_oid, "p_fok_int": p_fok, "v_mw": _qty_mw(v, config),
    }
    return _fok_finish("C10.fok_visible", seed, b.build(), params, req, v, envelope)


def _fok_hidden(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    rng = _rng(seed)
    rest_side = _SELL if rng.random() < 0.5 else _BUY
    fok_side = _other(rest_side)
    peak = int(rng.integers(10, 51))
    n_slices = int(rng.integers(2, 15))
    rem = int(rng.integers(1, peak + 1))
    total = (n_slices - 1) * peak + rem
    p_ice = encode_price(_snap_price(float(rng.uniform(30.0, 90.0)), config), config)
    through = int(rng.integers(1, 101))
    over = bool(rng.integers(0, 2))
    req = total + 1 if over else int(rng.integers(peak + 1, total + 1))
    b = TapeBuilder(config)
    ice_oid = b.add(side=rest_side, price_int=p_ice, qty_int=total, agent_id=1,
                    exec_restriction=ExecRestriction.ICE, peak_int=peak)
    p_fok = p_ice + _toward(fok_side, through)
    fok_oid = b.add(side=fok_side, price_int=p_fok, qty_int=req, agent_id=2,
                    exec_restriction=ExecRestriction.FOK)
    params = {
        "rest_side": rest_side, "ice_oid": ice_oid, "p_ice_int": p_ice,
        "total_int": total, "peak_int": peak, "n_slices": _ceil_div(total, peak),
        "req_int": req, "fok_oid": fok_oid, "p_fok_int": p_fok,
        "total_mw": _qty_mw(total, config), "peak_mw": _qty_mw(peak, config),
    }
    return _fok_finish("C10.fok_hidden", seed, b.build(), params, req, total, envelope)


def _fok_icewpi(seed: int, config: OMIEProductConfig, envelope) -> Draw:
    rng = _rng(seed)
    rest_side = _SELL if rng.random() < 0.5 else _BUY
    fok_side = _other(rest_side)
    peak = int(rng.integers(10, 51))
    n_slices = int(rng.integers(3, 15))
    total = n_slices * peak
    ppd_ticks = int(rng.integers(1, 30))
    ppd = ppd_ticks if rest_side == _SELL else -ppd_ticks  # §13.6 sign rule
    base = encode_price(_snap_price(float(rng.uniform(30.0, 90.0)), config), config)
    reachable = int(rng.integers(1, n_slices))
    over = bool(rng.integers(0, 2))
    p_fok = base + (reachable - 1) * ppd
    matchable = min(total, peak * reachable)
    req = matchable + 1 if over else matchable
    b = TapeBuilder(config)
    ice_oid = b.add(side=rest_side, price_int=base, qty_int=total, agent_id=1,
                    exec_restriction=ExecRestriction.ICE, peak_int=peak, ppd_int=ppd)
    fok_oid = b.add(side=fok_side, price_int=p_fok, qty_int=req, agent_id=2,
                    exec_restriction=ExecRestriction.FOK)
    params = {
        "rest_side": rest_side, "ice_oid": ice_oid, "base_int": base, "ppd_int": ppd,
        "peak_int": peak, "total_int": total, "n_slices": n_slices,
        "reachable_slices": reachable, "req_int": req, "fok_oid": fok_oid,
        "p_fok_int": p_fok, "base_eur": _price_eur(base, config),
        "ppd_eur": ppd * config.price_tick,
    }
    return _fok_finish("C10.fok_icewpi", seed, b.build(), params, req, matchable,
                       envelope)


def make_fok_boundary(
    seed: int,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
    variant: str = "visible",
) -> Draw:
    """A fill-or-kill order at the boundary of what is matchable.

    * ``"visible"``: a NON ladder of depth ``V``; the request is ``V + delta`` with
      ``delta`` in ``{-1, 0, +1}`` qty ticks; accepted iff ``delta <= 0``.
    * ``"hidden"``: a resting iceberg; ``req > peak``; accepted iff ``req <= total``
      (the pre-check sees the hidden volume, §4.2.1 Rule 1).
    * ``"icewpi"``: a price-incremented iceberg; matchable is
      ``min(total, peak * #{k : base + k*ppd within the limit})``.
    """
    _require_envelope(envelope)
    table = {"visible": _fok_visible, "hidden": _fok_hidden, "icewpi": _fok_icewpi}
    return _variant("FOK", table, variant)(seed, config, envelope)


# ---------------------------------------------------------------------------
# C15 — randomised fuzz sessions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FuzzRegime:
    """One cell of the fuzz sweep grid (swept systematically, not sampled)."""

    name: str
    order_types: Tuple[ExecRestriction, ...]
    cross_prob: float
    tie_density: float
    n_orders: int
    icewpi: bool = False
    same_owner_prob: float = 0.05


FUZZ_TYPE_MIXES: Mapping[str, Tuple[ExecRestriction, ...]] = types.MappingProxyType({
    "non": (ExecRestriction.NON,),
    "non_ioc": (ExecRestriction.NON, ExecRestriction.IOC),
    "non_ioc_fok": (ExecRestriction.NON, ExecRestriction.IOC, ExecRestriction.FOK),
    "non_ioc_fok_ice": (ExecRestriction.NON, ExecRestriction.IOC, ExecRestriction.FOK,
                        ExecRestriction.ICE),
    "all_icewpi": (ExecRestriction.NON, ExecRestriction.IOC, ExecRestriction.FOK,
                   ExecRestriction.ICE),
})


def fuzz_grid(
    *,
    mixes: Sequence[str] = tuple(FUZZ_TYPE_MIXES),
    cross_probs: Sequence[float] = (0.05, 0.2, 0.4, 0.6),
    tie_densities: Sequence[float] = (0.0, 0.4),
    lengths: Sequence[int] = (32, 96, 160),
) -> List[FuzzRegime]:
    """The full sweep grid (5 mixes x 4 cross probs x 2 tie densities x 3 lengths).

    The length axis stops at 160: a tape carries no cancellations, so every
    non-crossing order rests for the whole session, and 160 orders is the most
    two sides of ``max_resting_per_side = 80`` can hold. Longer sessions are C14's.
    """
    out: List[FuzzRegime] = []
    for mix in mixes:
        for cp in cross_probs:
            for td in tie_densities:
                for n in lengths:
                    out.append(FuzzRegime(
                        name=f"{mix}|cross={cp}|tie={td}|n={n}",
                        order_types=tuple(FUZZ_TYPE_MIXES[mix]),
                        cross_prob=float(cp), tie_density=float(td), n_orders=int(n),
                        icewpi=(mix == "all_icewpi"),
                    ))
    return out


def _session_attempt(rng, regime: FuzzRegime, config: OMIEProductConfig,
                     envelope: CapacityEnvelope):
    """One attempt at a session; returns (builder, safe_from_index, nonsafe_units) or None."""
    b = TapeBuilder(config)
    cap = envelope.max_resting_per_side
    budget = envelope.max_trades
    price_max_int = encode_price(config.price_max, config)
    qty_max_int = encode_qty(config.qty_max, config)
    mid = encode_price(_snap_price(float(rng.uniform(40.0, 80.0)), config), config)
    resting = {_BUY: 0, _SELL: 0}
    nonsafe_units = 0
    safe_from: Optional[int] = None
    min_ask: Optional[int] = None
    max_bid: Optional[int] = None
    owners = (1, 2, 3, 4)
    ts = 0

    for i in range(regime.n_orders):
        side = _BUY if rng.random() < 0.5 else _SELL
        crosses = rng.random() < regime.cross_prob
        restriction = int(regime.order_types[int(rng.integers(0, len(regime.order_types)))])
        dist = 1 + int(math.floor(float(rng.exponential(30.0))))
        qty = int(min(max(round(float(rng.lognormal(1.6, 0.7)) * 10), 1), 600))
        qty = min(qty, qty_max_int)
        peak = ppd = 0
        if restriction == _ICE:
            peak = max(1, qty // int(rng.integers(2, 6)))
            if regime.icewpi and rng.random() < 0.5:
                ppd = int(rng.integers(1, 20))
        agent = owners[int(rng.integers(0, len(owners)))]
        if rng.random() < regime.same_owner_prob:
            agent = owners[0]
        tie = rng.random() < regime.tie_density
        gap = 1 + int(math.floor(float(rng.exponential(1.0))))

        rests = restriction in (_NON, _ICE)
        if rests and resting[side] >= cap:
            side = _other(side)  # a side at its resting cap flips the side
            if resting[side] >= cap:
                return None
        if restriction == _ICE and ppd and side == _BUY:
            ppd = -ppd  # §13.6 sign rule
        if crosses:
            price = mid + dist if side == _BUY else mid - dist
        else:
            price = mid - dist if side == _BUY else mid + dist
        price = int(min(max(price, 1), price_max_int - 1))

        units = _ceil_div(qty, peak) if restriction == _ICE else 1
        if safe_from is None and nonsafe_units + units > budget:
            safe_from = i
        if safe_from is not None:
            # Safe: priced so it can never trade with any order ever submitted.
            if side == _BUY and min_ask is not None:
                price = min(price, min_ask - 1)
            elif side == _SELL and max_bid is not None:
                price = max(price, max_bid + 1)
        else:
            nonsafe_units += units
        if not 0 <= price <= price_max_int:
            return None

        if i == 0 or not tie:
            ts = ts + gap if i else 1
        b.add(side=side, price_int=price, qty_int=qty, agent_id=agent,
              exec_restriction=restriction, peak_int=peak, ppd_int=ppd, timestamp=ts)
        if rests:
            resting[side] += 1
        if side == _BUY:
            max_bid = price if max_bid is None else max(max_bid, price)
        else:
            min_ask = price if min_ask is None else min(min_ask, price)
    return b, safe_from, nonsafe_units


def make_random_session(
    seed: int,
    regime: FuzzRegime,
    config: OMIEProductConfig = OMIEProductConfig(),
    *,
    envelope: CapacityEnvelope = DEFAULT_ENVELOPE,
    max_retries: int = 24,
) -> Draw:
    """A whole random session for the fuzz scenario (oracle: the invariant battery).

    The session length is always exactly ``regime.n_orders``. Inter-arrival gaps
    are ``1 + floor(Exp(1))`` ticks; a tie (probability ``tie_density``) reuses the
    previous timestamp. The draw adapts to ``envelope`` instead of being rejected:

    * a NON or ICE order whose side is at ``max_resting_per_side`` flips side;
    * *safe mode*: once one more order would push the unit sum of non-safe
      orders past ``max_trades``, that order and every later one is priced so it
      can never trade (a buy strictly below every ask ever submitted, a sell
      strictly above every bid ever submitted). ``params["safe_from_index"]``
      records the switch (``None`` if it never engaged).

    The attempt is discarded and redrawn from a fresh stream
    (``seed + attempt * 1_000_003``) if both sides reach the resting cap, or if
    the finished tape's capacity bounds breach ``envelope`` on **any** axis.
    ``params["retries"]`` is the index of the accepted attempt,
    i.e. the number of discarded attempts. After ``max_retries`` failed attempts
    the last attempt's breach is raised: the :class:`CapacityBreach` from the
    envelope check if the last attempt reached it, otherwise a breach on
    ``resting_per_side``.
    """
    envelope = _require_envelope(envelope)
    last_breach: Optional[CapacityBreach] = None
    for attempt in range(int(max_retries)):
        rng = _rng(int(seed) + attempt * 1_000_003)
        out = _session_attempt(rng, regime, config, envelope)
        if out is None:
            last_breach = None
            continue
        b, safe_from, nonsafe_units = out
        params = {
            "regime": regime.name, "n_orders": int(regime.n_orders),
            "order_types": [int(t) for t in regime.order_types],
            "cross_prob": float(regime.cross_prob),
            "tie_density": float(regime.tie_density),
            "icewpi": bool(regime.icewpi),
            "safe_from_index": safe_from, "nonsafe_units": int(nonsafe_units),
            "retries": attempt,
        }
        try:
            return _finish("C15.random_session", seed, b.build(), params, {}, envelope)
        except CapacityBreach as exc:
            last_breach = exc
    if last_breach is not None:
        raise last_breach
    raise CapacityBreach(
        "resting_per_side", regime.n_orders, envelope.max_resting_per_side,
        f"regime {regime.name!r} cannot be drawn with both sides at most "
        f"{envelope.max_resting_per_side} resting orders in {max_retries} attempts",
    )


def _registry_random_session(seed: int) -> Draw:
    grid = fuzz_grid()
    return make_random_session(seed, grid[int(seed) % len(grid)])


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
CONSTRUCTIONS: Mapping[str, Callable[[int], Draw]] = types.MappingProxyType({
    "C00.empty": lambda seed: make_empty(),
    "C00.pad_only": lambda seed: make_pad_only(),
    "C00.status_gate": lambda seed: make_status_gate(seed),
    "C01.validation": lambda seed: make_validation_cases(seed),
    "C02.crossing_pair": lambda seed: make_crossing_pair(seed),
    "C03.priority_ladder": lambda seed: make_priority_ladder(seed),
    "C03.price_over_time": lambda seed: make_priority_ladder(seed, variant="price_over_time"),
    "C03.same_owner_pair": lambda seed: make_same_owner_pair(seed),
    "C04.cancellation": lambda seed: make_cancellation(seed, variant="ladder"),
    "C04.iceberg_cancel_fok": lambda seed: make_cancellation(seed, variant="iceberg_fok"),
    "C05.aggressor_residual": lambda seed: make_partial_fill_residual(
        seed, variant="aggressor"),
    "C05.resting_residual": lambda seed: make_partial_fill_residual(
        seed, variant="resting"),
    "C06.multi_level_walk": lambda seed: make_multi_level_walk(seed, stress=False),
    "C06.multi_level_walk_stress": lambda seed: make_multi_level_walk(seed, stress=True),
    "C07.ioc_regimes": lambda seed: make_ioc_regimes(seed),
    "C08.iceberg_cascade": lambda seed: make_iceberg_cascade(seed, variant="resting"),
    "C08.rule3": lambda seed: make_iceberg_cascade(seed, variant="rule3"),
    "C08.incoming_iceberg": lambda seed: make_iceberg_cascade(seed, variant="incoming"),
    "C09.icewpi_drift": lambda seed: make_icewpi_drift(seed, wrong_sign=False),
    "C09.icewpi_wrong_sign": lambda seed: make_icewpi_drift(seed, wrong_sign=True),
    "C10.fok_visible": lambda seed: make_fok_boundary(seed, variant="visible"),
    "C10.fok_hidden": lambda seed: make_fok_boundary(seed, variant="hidden"),
    "C10.fok_icewpi": lambda seed: make_fok_boundary(seed, variant="icewpi"),
    "C15.random_session": _registry_random_session,
})


def construction_digest(key: str, seeds: Sequence[int] = DIGEST_SEEDS) -> str:
    """blake2b-128 over the concatenated hex ``Draw.digest()`` of ``seeds``."""
    hexes = "".join(CONSTRUCTIONS[key](int(s)).digest() for s in seeds)
    return hashlib.blake2b(hexes.encode("ascii"), digest_size=16).hexdigest()


def compute_digest_table() -> Dict[str, str]:
    """``{registry key: construction digest}`` over seeds 0-7, sorted by key."""
    return {key: construction_digest(key) for key in sorted(CONSTRUCTIONS)}


def write_digests(path: str) -> Dict[str, Any]:
    """Write the construction-digest fixture to ``path`` and return its content."""
    payload = {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "algorithm": DIGEST_ALGORITHM,
        "seeds": list(DIGEST_SEEDS),
        "constructions": compute_digest_table(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m omiecid_simulator.jaxomiecid.omie_tape_synth",
        description="Synthetic OMIE tape generator utilities.",
    )
    parser.add_argument("--write-digests", metavar="PATH", required=True,
                        help="write the construction-digest fixture to PATH")
    args = parser.parse_args(argv)
    payload = write_digests(args.write_digests)
    print(f"wrote {len(payload['constructions'])} construction digests to "
          f"{args.write_digests}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
