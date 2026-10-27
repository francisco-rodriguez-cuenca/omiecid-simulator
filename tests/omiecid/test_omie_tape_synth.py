"""Locked behavioural contract for the synthetic tape generator (OMIE-CERT-SYNTH-1).

Story card: ``docs/omiecid/agile-plan.md`` -> "Story OMIE-CERT-SYNTH-1", binding Step-2
decisions S1-S14 (v5.183). This file is the S13 structural / digest / envelope /
spot-check suite. The engine-confirmed capacity test (S7) lives in
``tests/omiecid/test_omie_tape_synth_capacity.py``.

Module under test: ``omiecid_simulator/jaxomiecid/omie_tape_synth.py`` (imported as
``synth``) plus the S9 edit to ``omiecid_simulator/jaxomiecid/omie_data_loader.py``.

The import of the generator is guarded so this file *collects* before the module
exists; every test that needs it fails with a clear message instead.

----------------------------------------------------------------------------------
Contract pinned by THIS file beyond the card's literal text (the card leaves the
per-construction ``params`` / ``expected`` key names open; the spot checks need
them, so they are fixed here and the developer implements them verbatim). All
``*_int`` values are encoded integers (price ticks via the offset encoding of
``encoding.encode_price``; qty ticks via ``encode_qty``; ppd ticks via
``encode_ppd``). ``*_oid`` values are the ``order_id`` column of a real tape row.
``side`` values are ``int(Side.*)``. Euro/MW floats may sit in ``params`` beside
these; they are never read by the checks.

C00.empty           params: -          expected: n_trades=0, n_resting=0
C00.pad_only        params: n_pad      expected: n_trades=0, n_resting=0
C00.status_gate     params: contract_statuses=[0, 2, 3] (PENDING, SUSPENDED, CLOSED)
                    expected: n_trades=0, n_accepted=0
C01.validation      params: cases = list of dicts, each with name, price_eur, qty_mw,
                      optional peak_qty_mw and exec_restriction (int); the near-bound
                      pair also carries offset_ticks (float). Required case names:
                      in_bounds_on_tick, price_below_min, price_above_max,
                      qty_below_min, qty_above_max, sub_half_tick_off_grid,
                      over_half_tick_off_grid, iceberg_peak_exceeds_qty,
                      iceberg_peak_zero, near_max_sub_half_tick,
                      near_max_over_half_tick
                    expected: verdicts {name: bool},
                      snapped_price_int {name: int} for every accepted case
C02.crossing_pair   params: rest_side, agg_side, rest_oid, agg_oid, p_rest_int,
                      p_agg_int, gap_ticks, q_rest_int, q_agg_int
                    expected: n_trades, trade_price_int, trade_qty_int, residual_qty_int
C03.priority_ladder params: k, j, n_decoys, rest_side, agg_side, p_touch_int,
                      q_each_int, at_touch_oids, decoy_oids, agg_oid, q_agg_int,
                      p_agg_int
                    expected: filled_oids, untouched_oids, n_trades, trade_price_int,
                      total_qty_int
C03.same_owner_pair params: the C02 params + owner_agent_id
                    aux_tapes["counterfactual"]: the distinct-owner C02 tape
                    expected: the C02 expected + must_trade=True
C04.cancellation    params: k, j, n_decoys, rest_side, agg_side, p_touch_int,
                      q_each_int, resting_oids, decoy_oids, cancel_oid, agg_oid
                    expected: survivor_oids_in_queue_order, filled_oids, n_trades,
                      trades_from_cancel=0, trade_price_int
C04.iceberg_cancel_fok params: rest_side, ice_oid, cancel_oid, p_ice_int, total_int,
                      peak_int, n_slices, n_vis, visible_oids, v_vis_int, fok_oid,
                      req_int, p_fok_int
                    expected: accepted=False, filled_qty_int=0, book_unchanged_by_fok=True
C05.aggressor_residual / C05.resting_residual
                    params: rest_side, agg_side, rest_oid, agg_oid, interloper_oid,
                      take_oid, residual_oid, residual_original_timestamp, p_rest_int,
                      p_agg_int, q_rest_int, q_agg_int
                    expected: first_fill_qty_int, residual_qty_int, residual_price_int,
                      residual_timestamp, second_fill_oid,
                      interloper_must_remain_unfilled=True
C06.multi_level_walk / C06.multi_level_walk_stress
                    params: L, m, stress, rest_side, agg_side, level_price_ints,
                      level_qty_ints, resting_oids, agg_oid, q_agg_int, p_agg_int,
                      partial_qty_int
                    expected: per_level_fills [[price_int, qty_int], ...], num_fills,
                      total_qty_int, weighted_price_int
C07.ioc_regimes     params: regime, rest_side, agg_side, rest_oid, agg_oid, p_rest_int,
                      p_agg_int, q_rest_int, q_agg_int
                    aux_tapes["counterfactual"]: the same tape with the aggressor NON
                    expected: traded_qty_int, n_trades, ioc_residual_rests=False
C08.iceberg_cascade params: ice_side, agg_side, ice_oid, agg_oid, p_ice_int, p_agg_int,
                      total_int, peak_int, n_slices, remainder_int, q_agg_int
                    expected: slice_sizes_int, n_slices_consumed, trade_price_int,
                      truncated=False, book_uncrossed_on_return=True
C08.rule3           params: as C08.iceberg_cascade (remainder_int optional)
                    expected: slice_sizes_int (partition of q_agg_int), filled_qty_int,
                      first_trade_price_int, aggressor_rests=False,
                      iceberg_remaining_int
                    pending == ("rule3_hop_trade_prices:OQ-31",)
C08.incoming_iceberg params: rest_side, ice_side, L, level_price_ints, level_qty_ints,
                      resting_oids, ice_oid, p_ice_int, total_int, peak_int, n_slices
                    expected: trades_int [[price_int, qty_int], ...], reported_qty_int,
                      iceberg_rests=False
C09.icewpi_drift    params: ice_side, agg_side, ice_oid, agg_oid, base_int, ppd_int,
                      ppd_ticks, peak_int, total_int, n_slices, m, e_int, q_agg_int,
                      p_agg_int
                    expected: trade_price_sequence_int, stop_index, n_trades,
                      iceberg_rests_with_hidden=True, residual_qty_int,
                      residual_price_int, hidden_remaining_int
C09.icewpi_wrong_sign params: as C09.icewpi_drift, ppd_int = the emitted (wrong) sign
                    expected: {} (oracle_kind "observation")
C10.fok_visible     params: rest_side, n_levels, level_price_ints, level_qty_ints,
                      v_int, delta_ticks, req_int, fok_oid, p_fok_int
                    expected: accepted, filled_qty_int, matchable_qty_int
C10.fok_hidden      params: rest_side, ice_oid, p_ice_int, total_int, peak_int, n_slices,
                      req_int, fok_oid, p_fok_int
                    expected: accepted, filled_qty_int, matchable_qty_int
C10.fok_icewpi      params: rest_side, ice_oid, base_int, ppd_int, peak_int, total_int,
                      n_slices, reachable_slices, req_int, fok_oid, p_fok_int
                    expected: accepted, filled_qty_int, matchable_qty_int
C15.random_session  params: n_orders, safe_from_index (int in [0, n_orders]; n_orders
                      or None when safe mode never engaged)

Other pins: aux tape name "counterfactual"; ``FuzzRegime`` fields ``order_types``,
``cross_prob``, ``tie_density``, ``n_orders``; ``TapeBuilder.add`` float keywords are
the archive's ``price_eur`` / ``qty_mw`` / ``peak_qty_mw`` / ``ppd_eur`` beside the
card's ``price_int`` / ``qty_int`` / ``peak_int`` / ``ppd_int``; ``n_real_rows`` may be
a method or a property; the fixture at ``tests/omiecid/fixtures/cert/synth_digests.json``
is written only by ``python -m omiecid_simulator.jaxomiecid.omie_tape_synth
--write-digests <path>`` (S8), so the stored-digest tests are RED until the developer
runs it.

Plan rule 1 (each check seen failing on a deliberately broken input): every
spot check ``_check_<key>`` is asserted ``== []`` on the real draw and ``!= []`` on a
mutated draw; the shared structural checkers (layout, integer-only, S8 digest,
import boundary) each have a self-test on a hand-built broken input that runs
without the generator.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid import omie_data_loader as loader
from omiecid_simulator.omiecid.contract import ContractStatus
from omiecid_simulator.omiecid.encoding import encode_price, encode_qty
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, Status, Validity

try:  # guarded: the file must collect before the module exists
    from omiecid_simulator.jaxomiecid import omie_tape_synth as synth
    _IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - exercised only pre-implementation
    synth = None
    _IMPORT_ERROR = repr(_exc)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cert" / "synth_digests.json"
PRODUCT = OMIEProductConfig()
MARKET = OMIEMarketConfig()

# Column indices, written out independently of the module (the loader's order).
C_CONTRACT, C_SIDE, C_PRICE, C_QTY, C_OID, C_AGENT, C_TS = 0, 1, 2, 3, 4, 5, 6
C_EXEC, C_VALIDITY, C_EXPIRY, C_STATUS, C_PEAK, C_PPD = 7, 8, 9, 10, 11, 12

BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, IOC, FOK, ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                      int(ExecRestriction.FOK), int(ExecRestriction.ICE))

PAD_VALUES = [0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0]

PRICE_MAX_INT = encode_price(PRODUCT.price_max, PRODUCT)
QTY_MIN_INT = encode_qty(PRODUCT.qty_min, PRODUCT)
QTY_MAX_INT = encode_qty(PRODUCT.qty_max, PRODUCT)

S1_KEYS = (
    "C00.empty", "C00.pad_only", "C00.status_gate", "C01.validation",
    "C02.crossing_pair", "C03.priority_ladder", "C03.same_owner_pair",
    "C04.cancellation", "C04.iceberg_cancel_fok", "C05.aggressor_residual",
    "C05.resting_residual", "C06.multi_level_walk", "C06.multi_level_walk_stress",
    "C07.ioc_regimes", "C08.iceberg_cascade", "C08.rule3", "C08.incoming_iceberg",
    "C09.icewpi_drift", "C09.icewpi_wrong_sign", "C10.fok_visible",
    "C10.fok_hidden", "C10.fok_icewpi", "C15.random_session",
)
SEEDLESS_KEYS = ("C00.empty", "C00.pad_only")
NO_ENVELOPE_KEYS = ("C00.empty", "C00.pad_only", "C01.validation")
SEEDS = tuple(range(8))

ARCHIVE_TAPE_DIGESTS = {
    "C02.crossing_pair": "092b654f11300b162909d7ca1d7a22bf",
    "C03.priority_ladder": "2482d3fd8f8e4c40778874e6ef65354b",
    "C05.aggressor_residual": "293b9abd103336b1a5547aa479558552",
    "C06.multi_level_walk": "377c9a4746f9d4dcbb95b6af95dd5e76",
    "C07.ioc_regimes": "68485ff30dd14df3c520b82c44228238",
    "C08.iceberg_cascade": "01817ba01834e49639dfba2477937d3d",
}

AXES = ("resting_per_side", "trades", "hops", "concurrent_icebergs")
AXIS_FIELD = {
    "resting_per_side": "max_resting_per_side",
    "trades": "max_trades",
    "hops": "max_hops",
    "concurrent_icebergs": "max_concurrent_icebergs",
}
AXIS_CEILING = {
    "max_resting_per_side": MARKET.book_depth,
    "max_trades": MARKET.nTrades,
    "max_hops": MARKET.max_match_iterations,
    "max_concurrent_icebergs": MARKET.max_icebergs,
}
DEFAULT_LIMITS = {
    "max_resting_per_side": 80, "max_trades": 80, "max_hops": 160,
    "max_concurrent_icebergs": 800,
}


# ---------------------------------------------------------------------------
# Access helpers
# ---------------------------------------------------------------------------
def S():
    """The generator module, or a clear failure if it does not exist yet."""
    if synth is None:
        pytest.fail(
            "omiecid_simulator.jaxomiecid.omie_tape_synth is not importable "
            f"({_IMPORT_ERROR}); OMIE-CERT-SYNTH-1 is not implemented yet."
        )
    return synth


def _registry_keys():
    """Registry keys if the module imports (so later modes are picked up), else S1."""
    if synth is not None and hasattr(synth, "CONSTRUCTIONS"):
        return tuple(sorted(synth.CONSTRUCTIONS))
    return tuple(sorted(S1_KEYS))


MODE_CALLS = {
    "C00.empty": lambda seed, **kw: S().make_empty(),
    "C00.pad_only": lambda seed, **kw: S().make_pad_only(),
    "C00.status_gate": lambda seed, **kw: S().make_status_gate(seed, **kw),
    "C01.validation": lambda seed, **kw: S().make_validation_cases(seed),
    "C02.crossing_pair": lambda seed, **kw: S().make_crossing_pair(seed, **kw),
    "C03.priority_ladder": lambda seed, **kw: S().make_priority_ladder(seed, **kw),
    "C03.same_owner_pair": lambda seed, **kw: S().make_same_owner_pair(seed, **kw),
    "C04.cancellation": lambda seed, **kw: S().make_cancellation(
        seed, variant="ladder", **kw),
    "C04.iceberg_cancel_fok": lambda seed, **kw: S().make_cancellation(
        seed, variant="iceberg_fok", **kw),
    "C05.aggressor_residual": lambda seed, **kw: S().make_partial_fill_residual(
        seed, variant="aggressor", **kw),
    "C05.resting_residual": lambda seed, **kw: S().make_partial_fill_residual(
        seed, variant="resting", **kw),
    "C06.multi_level_walk": lambda seed, **kw: S().make_multi_level_walk(
        seed, stress=False, **kw),
    "C06.multi_level_walk_stress": lambda seed, **kw: S().make_multi_level_walk(
        seed, stress=True, **kw),
    "C07.ioc_regimes": lambda seed, **kw: S().make_ioc_regimes(seed, **kw),
    "C08.iceberg_cascade": lambda seed, **kw: S().make_iceberg_cascade(
        seed, variant="resting", **kw),
    "C08.rule3": lambda seed, **kw: S().make_iceberg_cascade(
        seed, variant="rule3", **kw),
    "C08.incoming_iceberg": lambda seed, **kw: S().make_iceberg_cascade(
        seed, variant="incoming", **kw),
    "C09.icewpi_drift": lambda seed, **kw: S().make_icewpi_drift(
        seed, wrong_sign=False, **kw),
    "C09.icewpi_wrong_sign": lambda seed, **kw: S().make_icewpi_drift(
        seed, wrong_sign=True, **kw),
    "C10.fok_visible": lambda seed, **kw: S().make_fok_boundary(
        seed, variant="visible", **kw),
    "C10.fok_hidden": lambda seed, **kw: S().make_fok_boundary(
        seed, variant="hidden", **kw),
    "C10.fok_icewpi": lambda seed, **kw: S().make_fok_boundary(
        seed, variant="icewpi", **kw),
}
ENVELOPE_KEYS = tuple(k for k in MODE_CALLS if k not in NO_ENVELOPE_KEYS)


@functools.lru_cache(maxsize=None)
def _draw(key: str, seed: int):
    """Registry draw, cached (draws are frozen; tests never mutate them)."""
    s = S()
    if key not in s.CONSTRUCTIONS:
        pytest.fail(f"{key!r} is not registered in CONSTRUCTIONS")
    return s.CONSTRUCTIONS[key](seed)


def _n_real(draw) -> int:
    v = draw.n_real_rows
    return int(v() if callable(v) else v)


def _fuzz_cells(n_orders=None, cross_prob=None, non_only=None):
    out = []
    for r in S().fuzz_grid():
        if n_orders is not None and r.n_orders != n_orders:
            continue
        if cross_prob is not None and not math.isclose(r.cross_prob, cross_prob):
            continue
        if non_only is not None:
            is_non = tuple(int(t) for t in r.order_types) == (NON,)
            if is_non != non_only:
                continue
        out.append(r)
    return out


def _canon(x):
    """S8 canonical form: numpy scalars -> Python, tuples -> lists, mappings -> dicts."""
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


def _thaw(x):
    """Deep, mutable copy of a params/expected structure."""
    return _canon(x)


def _s8_draw_digest(draw) -> str:
    """S8 ``Draw.digest()``, recomputed independently from the card's text."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(np.asarray(draw.tape).astype("<i4")).tobytes())
    for name in sorted(draw.aux_tapes):
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(
            np.asarray(draw.aux_tapes[name]).astype("<i4")).tobytes())
    blob = json.dumps(
        {"params": _canon(draw.params), "expected": _canon(draw.expected)},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    h.update(blob.encode("utf-8"))
    return h.hexdigest()


def _construction_digest(hexes) -> str:
    return hashlib.blake2b("".join(hexes).encode("ascii"), digest_size=16).hexdigest()


def _tape_only_digest(tapes) -> str:
    blob = b"".join(np.ascontiguousarray(np.asarray(t).astype("<i4")).tobytes()
                    for t in tapes)
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _load_fixture() -> dict:
    if not FIXTURE.exists():
        pytest.fail(
            f"{FIXTURE} does not exist. Per S8 it is written only by "
            "`python -m omiecid_simulator.jaxomiecid.omie_tape_synth --write-digests "
            "tests/omiecid/fixtures/cert/synth_digests.json` once the code is final."
        )
    return json.loads(FIXTURE.read_text())


# ---------------------------------------------------------------------------
# Shared structural checkers (each has a rule-1 self-test below)
# ---------------------------------------------------------------------------
def _layout_findings(tape) -> list:
    """S10 layout: (N, 13) C-contiguous int32; real rows first, then pad rows."""
    f = []
    if not isinstance(tape, np.ndarray):
        return [f"tape is {type(tape).__name__}, not ndarray"]
    if tape.dtype != np.int32:
        f.append(f"dtype {tape.dtype} != int32")
    if tape.ndim != 2 or tape.shape[1] != 13:
        return f + [f"shape {tape.shape} is not (N, 13)"]
    if not tape.flags["C_CONTIGUOUS"]:
        f.append("tape is not C-contiguous")
    pad = np.asarray(PAD_VALUES, dtype=np.int64)
    is_pad = np.all(tape.astype(np.int64) == pad[None, :], axis=1)
    is_real = tape[:, C_OID] >= 1
    bad = ~(is_pad | is_real)
    if bad.any():
        f.append(f"rows {np.flatnonzero(bad).tolist()[:5]} are neither real "
                 "(order_id >= 1) nor the loader pad row")
    n_real = int(is_real.sum())
    if not is_real[:n_real].all():
        f.append("a pad row precedes a real row (real rows must come first)")
    real = tape[:n_real]
    if n_real:
        oids = real[:, C_OID].tolist()
        if len(set(oids)) != len(oids):
            f.append("duplicate order_id among real rows")
        ts = real[:, C_TS]
        if (ts < 1).any():
            f.append("a real row has timestamp < 1")
        if (np.diff(ts) < 0).any():
            f.append("real-row timestamps decrease")
        if (real[:, C_STATUS] != int(Status.ACTIVE)).any():
            f.append("a real row is not Status.ACTIVE")
        if not np.isin(real[:, C_SIDE], [BUY, SELL]).all():
            f.append("a real row has an invalid side")
        if not np.isin(real[:, C_EXEC], [NON, IOC, FOK, ICE]).all():
            f.append("a real row has an invalid exec_restriction")
        if not np.isin(real[:, C_VALIDITY], [int(v) for v in Validity]).all():
            f.append("a real row has an invalid validity")
        if ((real[:, C_PRICE] < 0) | (real[:, C_PRICE] > PRICE_MAX_INT)).any():
            f.append("a real row's price is outside the product bounds")
        if ((real[:, C_QTY] < QTY_MIN_INT) | (real[:, C_QTY] > QTY_MAX_INT)).any():
            f.append("a real row's qty is outside the product bounds")
        ice = real[:, C_EXEC] == ICE
        if ice.any():
            peaks, qtys = real[ice, C_PEAK], real[ice, C_QTY]
            if ((peaks <= 0) | (peaks > qtys)).any():
                f.append("an ICE row violates 0 < peak <= qty")
        if (~ice).any():
            if (real[~ice, C_PEAK] != 0).any() or (real[~ice, C_PPD] != 0).any():
                f.append("a non-ICE row carries a peak or ppd")
    return f


def _integer_only_findings(x, path="expected") -> list:
    """S4: ``expected`` holds ints, bools, and lists (or str-keyed maps) of these."""
    if isinstance(x, (bool, np.bool_, int, np.integer)):
        return []
    if isinstance(x, Mapping):
        out = []
        for k, v in x.items():
            if not isinstance(k, str):
                out.append(f"{path}: non-str key {k!r}")
            out += _integer_only_findings(v, f"{path}[{k!r}]")
        return out
    if isinstance(x, (list, tuple)):
        out = []
        for i, v in enumerate(x):
            out += _integer_only_findings(v, f"{path}[{i}]")
        return out
    return [f"{path} is {type(x).__name__}, not integer-only"]


FORBIDDEN_MODULE_PARTS = {
    "contract_book", "iceberg_manager", "order_validator", "settlement",
    "trade_extraction", "omie_session",
}
ALLOWED_PROJECT_MODULES = {
    "omiecid_simulator.omiecid.encoding",
    "omiecid_simulator.omiecid.omie_config",
    "omiecid_simulator.omiecid.order_types",
    "omiecid_simulator.omiecid.contract",
    "omiecid_simulator.jaxomiecid.omie_data_loader",
}


def _import_boundary_findings(source: str) -> list:
    """S12: only numpy, stdlib and the allowed omie value-layer modules."""
    f = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [(a.name, None) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                f.append(f"relative import at line {node.lineno}")
                continue
            mods = [(node.module or "", [a.name for a in node.names])]
        for mod, names in mods:
            parts = set(mod.split("."))
            if names:
                parts |= set(names)  # catches ``from omiecid_simulator.omiecid import X``
            if parts & FORBIDDEN_MODULE_PARTS:
                f.append(f"forbidden import {mod!r} (line {node.lineno})")
            if mod == "jax" or mod.startswith("jax.") or mod.startswith("jaxlib"):
                f.append(f"JAX import {mod!r} (line {node.lineno}); S12 forbids it")
            if mod.startswith("omiecid_simulator"):
                if mod not in ALLOWED_PROJECT_MODULES:
                    f.append(f"project import {mod!r} outside the S12 allow-list")
                if mod == "omiecid_simulator.jaxomiecid.omie_data_loader" and (
                        names is None or set(names) != {"PAD_ORDER_ROW"}):
                    f.append("the loader may be imported only for PAD_ORDER_ROW")
    # No pad literal of its own: no 13-element list/tuple display.
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 13:
            f.append(f"a 13-element literal at line {node.lineno} (own pad row?)")
    return f


# ---------------------------------------------------------------------------
# Spot-check helpers
# ---------------------------------------------------------------------------
def _findings_on_error(fn):
    """A missing key / malformed field is a finding, not a crash."""
    @functools.wraps(fn)
    def wrapper(draw):
        try:
            return fn(draw)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            return [f"{fn.__name__}: malformed draw ({type(exc).__name__}: {exc})"]
    return wrapper


def _real(tape):
    tape = np.asarray(tape)
    return tape[tape[:, C_OID] >= 1]


def _rows(tape) -> dict:
    return {int(r[C_OID]): r for r in _real(tape)}


def _row_index(tape, oid) -> int:
    idx = np.flatnonzero(np.asarray(tape)[:, C_OID] == int(oid))
    return int(idx[0])


def _through(agg_side, p_agg, p_rest) -> bool:
    """The aggressor's limit reaches the resting price (inclusive)."""
    return p_agg >= p_rest if agg_side == BUY else p_agg <= p_rest


def _worse_for_taker(rest_side, p, ref) -> bool:
    """``p`` is strictly worse than ``ref`` for someone taking from ``rest_side``."""
    return p > ref if rest_side == SELL else p < ref


class _F:
    """Findings accumulator with a typed equality check."""

    def __init__(self, name):
        self.name = name
        self.out = []

    def eq(self, label, got, want):
        if isinstance(want, bool):
            if not isinstance(got, (bool, np.bool_)) or bool(got) != want:
                self.out.append(f"{self.name}: {label} = {got!r}, expected {want!r}")
            return
        if _canon(got) != _canon(want):
            self.out.append(f"{self.name}: {label} = {_canon(got)!r}, "
                            f"expected {_canon(want)!r}")

    def true(self, label, cond):
        if not cond:
            self.out.append(f"{self.name}: {label}")


def _row_is(F, rows, oid, label, *, side=None, price=None, qty=None, exec_=None,
            peak=None, ppd=None):
    oid = int(oid)
    if oid not in rows:
        F.out.append(f"{F.name}: {label} oid {oid} not in tape")
        return None
    r = rows[oid]
    for col, want, nm in ((C_SIDE, side, "side"), (C_PRICE, price, "price"),
                          (C_QTY, qty, "qty"), (C_EXEC, exec_, "exec"),
                          (C_PEAK, peak, "peak"), (C_PPD, ppd, "ppd")):
        if want is not None:
            F.eq(f"{label} row {nm}", int(r[col]), int(want))
    return r


def _ceil_div(a, b):
    return -(-a // b)


def _partition(total, peak):
    full, rem = divmod(total, peak)
    return [peak] * full + ([rem] if rem else [])


# ---------------------------------------------------------------------------
# The 23 spot checks: recompute the expectation from params by its formula
# ---------------------------------------------------------------------------
@_findings_on_error
def _check_C00_empty(draw):
    F = _F("C00.empty")
    F.eq("tape.shape", list(np.asarray(draw.tape).shape), [0, 13])
    F.eq("n_real_rows", _n_real(draw), 0)
    F.eq("expected.n_trades", draw.expected["n_trades"], 0)
    F.eq("expected.n_resting", draw.expected["n_resting"], 0)
    return F.out


@_findings_on_error
def _check_C00_pad_only(draw):
    F = _F("C00.pad_only")
    n = int(draw.params["n_pad"])
    F.eq("tape.shape", list(np.asarray(draw.tape).shape), [n, 13])
    F.true("every row is the pad row",
           all(list(map(int, r)) == PAD_VALUES for r in np.asarray(draw.tape)))
    F.eq("expected.n_trades", draw.expected["n_trades"], 0)
    F.eq("expected.n_resting", draw.expected["n_resting"], 0)
    return F.out


@_findings_on_error
def _check_C00_status_gate(draw):
    F = _F("C00.status_gate")
    F.eq("params.contract_statuses", draw.params["contract_statuses"],
         [int(ContractStatus.PENDING), int(ContractStatus.SUSPENDED),
          int(ContractStatus.CLOSED)])
    F.true("the tape carries real orders", len(_real(draw.tape)) >= 1)
    F.eq("expected.n_trades", draw.expected["n_trades"], 0)
    F.eq("expected.n_accepted", draw.expected["n_accepted"], 0)
    return F.out


C01_FIXED_VERDICTS = {
    "in_bounds_on_tick": True, "price_below_min": False, "price_above_max": False,
    "qty_below_min": False, "qty_above_max": False, "sub_half_tick_off_grid": True,
    "iceberg_peak_exceeds_qty": False, "iceberg_peak_zero": False,
    "near_max_sub_half_tick": True, "near_max_over_half_tick": False,
}
C01_REQUIRED = set(C01_FIXED_VERDICTS) | {"over_half_tick_off_grid"}


@_findings_on_error
def _check_C01_validation(draw):
    F = _F("C01.validation")
    F.eq("tape rows", int(np.asarray(draw.tape).shape[0]), 0)
    cases = {c["name"]: c for c in draw.params["cases"]}
    F.true(f"missing case classes {sorted(C01_REQUIRED - set(cases))}",
           C01_REQUIRED <= set(cases))
    verdicts = draw.expected["verdicts"]
    snapped = draw.expected["snapped_price_int"]
    F.eq("verdict names", sorted(verdicts), sorted(cases))
    pc = PRODUCT
    for name, c in cases.items():
        price_int = round((float(c["price_eur"]) - pc.price_min) / pc.price_tick)
        qty_int = round(float(c["qty_mw"]) / pc.qty_tick)
        ok = 0 <= price_int <= PRICE_MAX_INT and QTY_MIN_INT <= qty_int <= QTY_MAX_INT
        if int(c.get("exec_restriction", NON)) == ICE:
            peak_int = round(float(c.get("peak_qty_mw", 0.0)) / pc.qty_tick)
            ok = ok and 0 < peak_int <= qty_int
        F.eq(f"verdicts[{name!r}] (recomputed)", verdicts.get(name), ok)
        if name in C01_FIXED_VERDICTS:
            F.eq(f"verdicts[{name!r}] (class)", verdicts.get(name),
                 C01_FIXED_VERDICTS[name])
        if ok:
            F.eq(f"snapped_price_int[{name!r}]", snapped.get(name), price_int)
    F.eq("snapped_price_int names", sorted(snapped),
         sorted(n for n, v in verdicts.items() if v))
    for name, lo, hi in (("near_max_sub_half_tick", 0.05, 0.45),
                         ("near_max_over_half_tick", 0.55, 0.95)):
        c = cases[name]
        off = float(c["offset_ticks"])
        F.true(f"{name} offset {off} outside [{lo}, {hi}]", lo <= off <= hi)
        F.true(f"{name} price is not price_max + offset ticks",
               abs(float(c["price_eur"]) - (pc.price_max + off * pc.price_tick)) < 1e-6)
    F.eq("snapped_price_int[near_max_sub_half_tick] == price_max tick",
         snapped.get("near_max_sub_half_tick"), PRICE_MAX_INT)
    return F.out


def _crossing_pair_findings(F, draw):
    p, e, rows = draw.params, draw.expected, _rows(draw.tape)
    rs, ag = int(p["rest_side"]), int(p["agg_side"])
    F.true("rest_side is a side", rs in (BUY, SELL))
    F.eq("agg_side", ag, 1 - rs)
    F.eq("real rows", len(rows), 2)
    _row_is(F, rows, p["rest_oid"], "rest", side=rs, price=p["p_rest_int"],
            qty=p["q_rest_int"], exec_=NON)
    _row_is(F, rows, p["agg_oid"], "agg", side=ag, price=p["p_agg_int"],
            qty=p["q_agg_int"], exec_=NON)
    F.true("resting order precedes the aggressor",
           _row_index(draw.tape, p["rest_oid"]) < _row_index(draw.tape, p["agg_oid"]))
    gap = int(p["gap_ticks"])
    F.true("gap_ticks >= 1", gap >= 1)
    F.eq("p_agg_int = p_rest_int +/- gap_ticks", p["p_agg_int"],
         p["p_rest_int"] + gap if ag == BUY else p["p_rest_int"] - gap)
    q_r, q_a = int(p["q_rest_int"]), int(p["q_agg_int"])
    F.eq("expected.n_trades", e["n_trades"], 1)
    F.eq("expected.trade_price_int (resting price)", e["trade_price_int"],
         p["p_rest_int"])
    F.eq("expected.trade_qty_int", e["trade_qty_int"], min(q_r, q_a))
    F.eq("expected.residual_qty_int", e["residual_qty_int"], abs(q_a - q_r))


@_findings_on_error
def _check_C02_crossing_pair(draw):
    F = _F("C02.crossing_pair")
    _crossing_pair_findings(F, draw)
    return F.out


@_findings_on_error
def _check_C03_priority_ladder(draw):
    F = _F("C03.priority_ladder")
    p, e, rows = draw.params, draw.expected, _rows(draw.tape)
    k, j, nd = int(p["k"]), int(p["j"]), int(p["n_decoys"])
    rs, ag = int(p["rest_side"]), int(p["agg_side"])
    F.true("2 <= k <= 6", 2 <= k <= 6)
    F.true("1 <= j < k", 1 <= j < k)
    F.true("1 <= n_decoys <= 3", 1 <= nd <= 3)
    F.eq("agg_side", ag, 1 - rs)
    touch, decoys = list(p["at_touch_oids"]), list(p["decoy_oids"])
    F.eq("len(at_touch_oids)", len(touch), k)
    F.eq("len(decoy_oids)", len(decoys), nd)
    ts = []
    for o in touch:
        r = _row_is(F, rows, o, "at-touch", side=rs, price=p["p_touch_int"],
                    qty=p["q_each_int"], exec_=NON)
        if r is not None:
            ts.append(int(r[C_TS]))
    F.true("at-touch timestamps strictly increase", all(a < b for a, b in zip(ts, ts[1:])))
    for o in decoys:
        r = _row_is(F, rows, o, "decoy", side=rs, exec_=NON)
        if r is not None:
            F.true("decoy is strictly worse-priced than the touch",
                   _worse_for_taker(rs, int(r[C_PRICE]), int(p["p_touch_int"])))
    a = _row_is(F, rows, p["agg_oid"], "agg", side=ag, price=p["p_agg_int"],
                qty=p["q_agg_int"])
    F.eq("q_agg_int = j * q_each_int", p["q_agg_int"], j * int(p["q_each_int"]))
    if a is not None:
        F.true("aggressor reaches the touch",
               _through(ag, int(p["p_agg_int"]), int(p["p_touch_int"])))
    F.eq("expected.filled_oids", e["filled_oids"], touch[:j])
    F.eq("expected.untouched_oids", e["untouched_oids"], touch[j:] + decoys)
    F.eq("expected.n_trades", e["n_trades"], j)
    F.eq("expected.trade_price_int", e["trade_price_int"], p["p_touch_int"])
    F.eq("expected.total_qty_int", e["total_qty_int"], j * int(p["q_each_int"]))
    return F.out


@_findings_on_error
def _check_C03_same_owner_pair(draw):
    F = _F("C03.same_owner_pair")
    _crossing_pair_findings(F, draw)
    owner = int(draw.params["owner_agent_id"])
    real = _real(draw.tape)
    F.true("every real row carries the one owner",
           bool((real[:, C_AGENT] == owner).all()))
    F.eq("expected.must_trade", draw.expected["must_trade"], True)
    cf = np.asarray(draw.aux_tapes["counterfactual"])
    tape = np.asarray(draw.tape)
    F.eq("counterfactual shape", list(cf.shape), list(tape.shape))
    if cf.shape == tape.shape:
        diff = np.argwhere(cf != tape)
        F.true("counterfactual differs only in agent_id",
               all(int(c) == C_AGENT for _, c in diff))
        cf_real = _real(cf)
        F.true("counterfactual owners are distinct",
               len(set(cf_real[:, C_AGENT].tolist())) == len(cf_real))
    return F.out


@_findings_on_error
def _check_C04_cancellation(draw):
    F = _F("C04.cancellation")
    p, e, rows = draw.params, draw.expected, _rows(draw.tape)
    k, j, nd = int(p["k"]), int(p["j"]), int(p["n_decoys"])
    rs, ag = int(p["rest_side"]), int(p["agg_side"])
    F.true("2 <= k <= 5", 2 <= k <= 5)
    F.true("1 <= n_decoys <= 3", 1 <= nd <= 3)
    F.true("1 <= j <= k - 1", 1 <= j <= k - 1)
    F.eq("agg_side", ag, 1 - rs)
    resting, decoys = list(p["resting_oids"]), list(p["decoy_oids"])
    F.eq("len(resting_oids)", len(resting), k)
    F.eq("len(decoy_oids)", len(decoys), nd)
    F.true("cancel_oid is one of the at-touch orders", p["cancel_oid"] in resting)
    ts = []
    for o in resting:
        r = _row_is(F, rows, o, "resting", side=rs, price=p["p_touch_int"],
                    qty=p["q_each_int"], exec_=NON)
        if r is not None:
            ts.append(int(r[C_TS]))
    F.true("at-touch timestamps strictly increase", all(a < b for a, b in zip(ts, ts[1:])))
    for o in decoys:
        r = _row_is(F, rows, o, "decoy", side=rs, exec_=NON)
        if r is not None:
            F.true("decoy strictly worse-priced",
                   _worse_for_taker(rs, int(r[C_PRICE]), int(p["p_touch_int"])))
    a = _row_is(F, rows, p["agg_oid"], "agg", side=ag, qty=j * int(p["q_each_int"]))
    if a is not None:
        F.true("aggressor reaches the touch",
               _through(ag, int(a[C_PRICE]), int(p["p_touch_int"])))
        F.true("aggressor is the last real row",
               _row_index(draw.tape, p["agg_oid"]) == len(rows) - 1)
    survivors = [o for o in resting if o != p["cancel_oid"]]
    F.eq("expected.survivor_oids_in_queue_order",
         e["survivor_oids_in_queue_order"], survivors)
    F.eq("expected.filled_oids", e["filled_oids"], survivors[:j])
    F.eq("expected.n_trades", e["n_trades"], j)
    F.eq("expected.trades_from_cancel", e["trades_from_cancel"], 0)
    F.eq("expected.trade_price_int", e["trade_price_int"], p["p_touch_int"])
    return F.out


@_findings_on_error
def _check_C04_iceberg_cancel_fok(draw):
    F = _F("C04.iceberg_cancel_fok")
    p, e, rows = draw.params, draw.expected, _rows(draw.tape)
    rs = int(p["rest_side"])
    total, peak = int(p["total_int"]), int(p["peak_int"])
    n = int(p["n_slices"])
    F.eq("cancel_oid is the iceberg", p["cancel_oid"], p["ice_oid"])
    F.eq("n_slices = ceil(total/peak)", n, _ceil_div(total, peak))
    F.true("n_slices >= 2", n >= 2)
    _row_is(F, rows, p["ice_oid"], "iceberg", side=rs, price=p["p_ice_int"],
            qty=total, exec_=ICE, peak=peak, ppd=0)
    n_vis = int(p["n_vis"])
    vis = list(p["visible_oids"])
    F.true("0 <= n_vis <= 3", 0 <= n_vis <= 3)
    F.eq("len(visible_oids)", len(vis), n_vis)
    v = 0
    for o in vis:
        r = _row_is(F, rows, o, "visible", side=rs, price=p["p_ice_int"], exec_=NON)
        if r is not None:
            v += int(r[C_QTY])
    F.eq("v_vis_int = sum of visible qty", p["v_vis_int"], v)
    req = int(p["req_int"])
    F.true("v_vis + 1 <= req <= v_vis + total", v + 1 <= req <= v + total)
    f = _row_is(F, rows, p["fok_oid"], "fok", side=1 - rs, price=p["p_fok_int"],
                qty=req, exec_=FOK)
    if f is not None:
        F.true("FOK is priced through the iceberg",
               _through(1 - rs, int(p["p_fok_int"]), int(p["p_ice_int"])))
        F.true("FOK is the last real row",
               _row_index(draw.tape, p["fok_oid"]) == len(rows) - 1)
    F.eq("expected.accepted", e["accepted"], False)
    F.eq("expected.filled_qty_int", e["filled_qty_int"], 0)
    F.eq("expected.book_unchanged_by_fok", e["book_unchanged_by_fok"], True)
    return F.out


def _residual_findings(F, draw, variant):
    p, e, rows, tape = draw.params, draw.expected, _rows(draw.tape), draw.tape
    rs, ag = int(p["rest_side"]), int(p["agg_side"])
    F.eq("agg_side", ag, 1 - rs)
    q_r, q_a = int(p["q_rest_int"]), int(p["q_agg_int"])
    p_r, p_a = int(p["p_rest_int"]), int(p["p_agg_int"])
    _row_is(F, rows, p["rest_oid"], "rest", side=rs, price=p_r, qty=q_r, exec_=NON)
    _row_is(F, rows, p["agg_oid"], "agg", side=ag, price=p_a, qty=q_a, exec_=NON)
    F.true("aggressor crosses the resting order", _through(ag, p_a, p_r))
    if variant == "aggressor":
        F.eq("residual_oid", p["residual_oid"], p["agg_oid"])
        F.true("q_agg > q_rest", q_a > q_r)
        filled, residual, res_side, res_price = q_r, q_a - q_r, ag, p_a
    else:
        F.eq("residual_oid", p["residual_oid"], p["rest_oid"])
        F.true("q_agg < q_rest", q_a < q_r)
        filled, residual, res_side, res_price = q_a, q_r - q_a, rs, p_r
    il = _row_is(F, rows, p["interloper_oid"], "interloper", side=res_side,
                 price=res_price, exec_=NON)
    tk = _row_is(F, rows, p["take_oid"], "take", side=1 - res_side, qty=residual,
                 exec_=NON)
    if tk is not None:
        F.true("second aggressor reaches the residual price",
               _through(1 - res_side, int(tk[C_PRICE]), res_price))
    order = [_row_index(tape, p[k]) for k in
             ("rest_oid", "agg_oid", "interloper_oid", "take_oid")]
    F.true("row order is rest, agg, interloper, take", order == sorted(order))
    res_row = rows.get(int(p["residual_oid"]))
    if res_row is not None:
        F.eq("residual_original_timestamp = residual row ts",
             p["residual_original_timestamp"], int(res_row[C_TS]))
    if il is not None:
        F.true("interloper arrives strictly after the residual's original ts",
               int(il[C_TS]) > int(p["residual_original_timestamp"]))
    F.eq("expected.first_fill_qty_int", e["first_fill_qty_int"], filled)
    F.eq("expected.residual_qty_int", e["residual_qty_int"], residual)
    F.eq("expected.residual_price_int", e["residual_price_int"], res_price)
    F.eq("expected.residual_timestamp", e["residual_timestamp"],
         p["residual_original_timestamp"])
    F.eq("expected.second_fill_oid", e["second_fill_oid"], p["residual_oid"])
    F.eq("expected.interloper_must_remain_unfilled",
         e["interloper_must_remain_unfilled"], True)


@_findings_on_error
def _check_C05_aggressor_residual(draw):
    F = _F("C05.aggressor_residual")
    _residual_findings(F, draw, "aggressor")
    return F.out


@_findings_on_error
def _check_C05_resting_residual(draw):
    F = _F("C05.resting_residual")
    _residual_findings(F, draw, "resting")
    return F.out


def _walk_findings(F, draw, stress):
    p, e, rows = draw.params, draw.expected, _rows(draw.tape)
    L, m = int(p["L"]), int(p["m"])
    rs, ag = int(p["rest_side"]), int(p["agg_side"])
    F.eq("params.stress", p["stress"], stress)
    F.eq("agg_side", ag, 1 - rs)
    if stress:
        F.true("L >= 2", L >= 2)
    else:
        F.true("3 <= L <= 12", 3 <= L <= 12)
    F.true("1 <= m < L", 1 <= m < L)
    prices, qtys = list(p["level_price_ints"]), list(p["level_qty_ints"])
    oids = list(p["resting_oids"])
    F.eq("len(level_price_ints)", len(prices), L)
    F.eq("len(level_qty_ints)", len(qtys), L)
    F.eq("len(resting_oids)", len(oids), L)
    for i, o in enumerate(oids):
        _row_is(F, rows, o, f"level {i}", side=rs, price=prices[i], qty=qtys[i],
                exec_=NON)
    F.true("level prices strictly worsen with depth",
           all(_worse_for_taker(rs, b, a) for a, b in zip(prices, prices[1:])))
    partial = int(p["partial_qty_int"])
    F.true("0 <= partial < level m qty", 0 <= partial < int(qtys[m]))
    fills = [[int(prices[i]), int(qtys[i])] for i in range(m)]
    if partial > 0:
        fills.append([int(prices[m]), partial])
    total = sum(q for _, q in fills)
    weighted = sum(pp * q for pp, q in fills)
    _row_is(F, rows, p["agg_oid"], "agg", side=ag, price=p["p_agg_int"],
            qty=p["q_agg_int"], exec_=NON)
    F.eq("q_agg_int = total of fills", p["q_agg_int"], total)
    F.true("aggressor reaches its last filled level",
           _through(ag, int(p["p_agg_int"]), fills[-1][0]))
    F.eq("expected.per_level_fills", e["per_level_fills"], fills)
    F.eq("expected.num_fills", e["num_fills"], len(fills))
    F.eq("expected.total_qty_int", e["total_qty_int"], total)
    F.eq("expected.weighted_price_int", e["weighted_price_int"], weighted)
    if stress:
        F.true("stress draw sits past the int32 product boundary",
               weighted > 2**31 - 1)


@_findings_on_error
def _check_C06_multi_level_walk(draw):
    F = _F("C06.multi_level_walk")
    _walk_findings(F, draw, stress=False)
    return F.out


@_findings_on_error
def _check_C06_multi_level_walk_stress(draw):
    F = _F("C06.multi_level_walk_stress")
    _walk_findings(F, draw, stress=True)
    return F.out


@_findings_on_error
def _check_C07_ioc_regimes(draw):
    F = _F("C07.ioc_regimes")
    p, e, rows = draw.params, draw.expected, _rows(draw.tape)
    regime = p["regime"]
    F.true(f"regime {regime!r} is none/partial/full", regime in ("none", "partial", "full"))
    rs, ag = int(p["rest_side"]), int(p["agg_side"])
    F.eq("agg_side", ag, 1 - rs)
    q_r, q_a = int(p["q_rest_int"]), int(p["q_agg_int"])
    p_r, p_a = int(p["p_rest_int"]), int(p["p_agg_int"])
    F.eq("real rows", len(rows), 2)
    _row_is(F, rows, p["rest_oid"], "rest", side=rs, price=p_r, qty=q_r, exec_=NON)
    _row_is(F, rows, p["agg_oid"], "agg", side=ag, price=p_a, qty=q_a, exec_=IOC)
    crosses = _through(ag, p_a, p_r)
    if regime == "none":
        F.true("'none' regime does not cross", not crosses)
        traded, n_tr = 0, 0
    elif regime == "partial":
        F.true("'partial' regime crosses", crosses)
        F.true("'partial' has q_agg > q_rest", q_a > q_r)
        traded, n_tr = q_r, 1
    else:
        F.true("'full' regime crosses", crosses)
        F.true("'full' has q_agg <= q_rest", q_a <= q_r)
        traded, n_tr = q_a, 1
    F.eq("expected.traded_qty_int", e["traded_qty_int"], traded)
    F.eq("expected.n_trades", e["n_trades"], n_tr)
    F.eq("expected.ioc_residual_rests", e["ioc_residual_rests"], False)
    cf = np.asarray(draw.aux_tapes["counterfactual"])
    tape = np.asarray(draw.tape)
    if cf.shape != tape.shape:
        F.out.append("C07.ioc_regimes: counterfactual shape differs")
    else:
        ai = _row_index(tape, p["agg_oid"])
        want = tape.copy()
        want[ai, C_EXEC] = NON
        F.true("counterfactual = tape with the aggressor NON", np.array_equal(cf, want))
    return F.out


def _iceberg_rows(F, draw, *, require_remainder):
    p, rows = draw.params, _rows(draw.tape)
    isd, ag = int(p["ice_side"]), int(p["agg_side"])
    F.eq("agg_side", ag, 1 - isd)
    total, peak, n = int(p["total_int"]), int(p["peak_int"]), int(p["n_slices"])
    F.eq("n_slices = ceil(total/peak)", n, _ceil_div(total, peak))
    _row_is(F, rows, p["ice_oid"], "iceberg", side=isd, price=p["p_ice_int"],
            qty=total, exec_=ICE, peak=peak, ppd=0)
    _row_is(F, rows, p["agg_oid"], "agg", side=ag, price=p["p_agg_int"],
            qty=p["q_agg_int"], exec_=NON)
    if require_remainder:
        F.eq("remainder_int = total - (n-1)*peak", p["remainder_int"],
             total - (n - 1) * peak)
    return isd, ag, total, peak, n


@_findings_on_error
def _check_C08_iceberg_cascade(draw):
    F = _F("C08.iceberg_cascade")
    p, e = draw.params, draw.expected
    isd, ag, total, peak, n = _iceberg_rows(F, draw, require_remainder=True)
    F.true("n_slices >= 2", n >= 2)
    F.eq("q_agg_int = total (one aggressor consumes the lot)", p["q_agg_int"], total)
    F.true("aggressor reaches the iceberg",
           _through(ag, int(p["p_agg_int"]), int(p["p_ice_int"])))
    F.eq("expected.slice_sizes_int", e["slice_sizes_int"],
         [peak] * (n - 1) + [total - (n - 1) * peak])
    F.eq("expected.n_slices_consumed", e["n_slices_consumed"], n)
    F.eq("expected.trade_price_int", e["trade_price_int"], p["p_ice_int"])
    F.eq("expected.truncated", e["truncated"], False)
    F.eq("expected.book_uncrossed_on_return", e["book_uncrossed_on_return"], True)
    return F.out


@_findings_on_error
def _check_C08_rule3(draw):
    F = _F("C08.rule3")
    p, e = draw.params, draw.expected
    isd, ag, total, peak, n = _iceberg_rows(F, draw, require_remainder=False)
    q = int(p["q_agg_int"])
    F.true("n_slices >= 3", n >= 3)
    F.true("peak < q_agg <= total - peak", peak < q <= total - peak)
    pa, pi = int(p["p_agg_int"]), int(p["p_ice_int"])
    F.true("limit at least 1 tick through the iceberg price",
           pa >= pi + 1 if ag == BUY else pa <= pi - 1)
    F.eq("expected.slice_sizes_int", e["slice_sizes_int"], _partition(q, peak))
    F.eq("expected.filled_qty_int", e["filled_qty_int"], q)
    F.eq("expected.first_trade_price_int", e["first_trade_price_int"], pi)
    F.eq("expected.aggressor_rests", e["aggressor_rests"], False)
    F.eq("expected.iceberg_remaining_int", e["iceberg_remaining_int"], total - q)
    F.eq("pending", list(draw.pending), ["rule3_hop_trade_prices:OQ-31"])
    return F.out


@_findings_on_error
def _check_C08_incoming_iceberg(draw):
    F = _F("C08.incoming_iceberg")
    p, e, rows = draw.params, draw.expected, _rows(draw.tape)
    rs, isd = int(p["rest_side"]), int(p["ice_side"])
    F.eq("ice_side", isd, 1 - rs)
    L = int(p["L"])
    F.true("2 <= L <= 8", 2 <= L <= 8)
    prices, qtys = list(p["level_price_ints"]), list(p["level_qty_ints"])
    oids = list(p["resting_oids"])
    F.eq("len(level_price_ints)", len(prices), L)
    F.eq("len(level_qty_ints)", len(qtys), L)
    F.eq("len(resting_oids)", len(oids), L)
    for i, o in enumerate(oids):
        _row_is(F, rows, o, f"level {i}", side=rs, price=prices[i], qty=qtys[i],
                exec_=NON)
    F.true("level prices strictly worsen with depth",
           all(_worse_for_taker(rs, b, a) for a, b in zip(prices, prices[1:])))
    total, peak, n = int(p["total_int"]), int(p["peak_int"]), int(p["n_slices"])
    F.eq("n_slices = ceil(total/peak)", n, _ceil_div(total, peak))
    F.true("n_slices >= 2", n >= 2)
    F.true("depth >= total + 1 tick", sum(int(q) for q in qtys) >= total + 1)
    _row_is(F, rows, p["ice_oid"], "iceberg", side=isd, price=p["p_ice_int"],
            qty=total, exec_=ICE, peak=peak, ppd=0)
    if int(p["ice_oid"]) in rows:
        F.true("iceberg is the last real row",
               _row_index(draw.tape, p["ice_oid"]) == len(rows) - 1)
    # Two-pointer merge of the slice partition with the level partition.
    slices = [peak] * (n - 1) + [total - (n - 1) * peak]
    lev = [int(x) for x in qtys]
    trades, i, j, deepest = [], 0, 0, 0
    while i < len(slices) and j < len(lev):
        t = min(slices[i], lev[j])
        trades.append([int(prices[j]), t])
        slices[i] -= t
        lev[j] -= t
        deepest = j
        if slices[i] == 0:
            i += 1
        if lev[j] == 0:
            j += 1
    F.true("the merge exhausts the iceberg", i == len(slices))
    F.eq("iceberg limit = deepest level it needs", p["p_ice_int"], prices[deepest])
    F.eq("expected.trades_int", e["trades_int"], trades)
    F.eq("expected.reported_qty_int", e["reported_qty_int"], total)
    F.eq("expected.iceberg_rests", e["iceberg_rests"], False)
    F.eq("capacity_bounds.trades = n_slices + L - 1",
         draw.capacity_bounds["trades"], n + L - 1)
    F.eq("hop_bound = n_slices + L + 2", draw.hop_bound, n + L + 2)
    return F.out


def _icewpi_common(F, draw, *, wrong_sign):
    p, rows = draw.params, _rows(draw.tape)
    isd, ag = int(p["ice_side"]), int(p["agg_side"])
    F.eq("agg_side", ag, 1 - isd)
    ppd, ticks = int(p["ppd_int"]), int(p["ppd_ticks"])
    F.true("ppd_ticks >= 1", ticks >= 1)
    F.eq("|ppd_int| = ppd_ticks", abs(ppd), ticks)
    valid_sign = ppd > 0 if isd == SELL else ppd < 0
    if wrong_sign:
        F.true("ppd sign is the WRONG one (sell < 0, buy > 0)", not valid_sign)
    else:
        F.true("ppd sign is valid (sell > 0, buy < 0; market-spec 13.6)", valid_sign)
    total, peak, n = int(p["total_int"]), int(p["peak_int"]), int(p["n_slices"])
    m, e_int = int(p["m"]), int(p["e_int"])
    F.true("n_slices >= 3", n >= 3)
    F.eq("total = n_slices * peak", total, n * peak)
    F.true("1 <= m <= n_slices - 2", 1 <= m <= n - 2)
    F.true("peak >= 2 ticks", peak >= 2)
    F.true("1 <= e <= peak - 1", 1 <= e_int <= peak - 1)
    _row_is(F, rows, p["ice_oid"], "iceberg", side=isd, price=p["base_int"],
            qty=total, exec_=ICE, peak=peak, ppd=ppd)
    _row_is(F, rows, p["agg_oid"], "agg", side=ag, price=p["p_agg_int"],
            qty=p["q_agg_int"], exec_=NON)
    F.eq("q_agg_int = m*peak + e", p["q_agg_int"], m * peak + e_int)
    signed = -ppd if wrong_sign else ppd  # the construction's sign-valid increment
    F.eq("p_agg_int = base + (m-1)*ppd", p["p_agg_int"],
         int(p["base_int"]) + (m - 1) * signed)
    return isd, ag, ppd, total, peak, n, m, e_int


@_findings_on_error
def _check_C09_icewpi_drift(draw):
    F = _F("C09.icewpi_drift")
    p, e = draw.params, draw.expected
    isd, ag, ppd, total, peak, n, m, e_int = _icewpi_common(F, draw, wrong_sign=False)
    base = int(p["base_int"])
    F.eq("expected.trade_price_sequence_int", e["trade_price_sequence_int"],
         [base + i * ppd for i in range(m)])
    F.eq("expected.stop_index", e["stop_index"], m)
    F.eq("expected.n_trades", e["n_trades"], m)
    F.eq("expected.iceberg_rests_with_hidden", e["iceberg_rests_with_hidden"], True)
    F.eq("expected.residual_qty_int", e["residual_qty_int"], e_int)
    F.eq("expected.residual_price_int", e["residual_price_int"], p["p_agg_int"])
    F.eq("expected.hidden_remaining_int", e["hidden_remaining_int"],
         (n - m - 1) * peak)
    return F.out


@_findings_on_error
def _check_C09_icewpi_wrong_sign(draw):
    F = _F("C09.icewpi_wrong_sign")
    _icewpi_common(F, draw, wrong_sign=True)
    F.eq("oracle_kind", draw.oracle_kind, "observation")
    F.eq("expected (an observation asserts nothing)", dict(draw.expected), {})
    return F.out


def _fok_row(F, draw, rest_side, reach_price):
    p, rows = draw.params, _rows(draw.tape)
    f = _row_is(F, rows, p["fok_oid"], "fok", side=1 - rest_side, price=p["p_fok_int"],
                qty=p["req_int"], exec_=FOK)
    if f is not None:
        F.true("FOK is the last real row",
               _row_index(draw.tape, p["fok_oid"]) == len(rows) - 1)
        if reach_price is not None:
            F.true("FOK limit reaches the liquidity",
                   _through(1 - rest_side, int(p["p_fok_int"]), reach_price))


def _fok_verdict(F, draw, matchable):
    e, req = draw.expected, int(draw.params["req_int"])
    acc = req <= matchable
    F.true("req >= 1 tick", req >= 1)
    F.eq("expected.accepted", e["accepted"], acc)
    F.eq("expected.filled_qty_int", e["filled_qty_int"], req if acc else 0)
    F.eq("expected.matchable_qty_int", e["matchable_qty_int"], matchable)


@_findings_on_error
def _check_C10_fok_visible(draw):
    F = _F("C10.fok_visible")
    p = draw.params
    rs = int(p["rest_side"])
    nl = int(p["n_levels"])
    prices, qtys = list(p["level_price_ints"]), list(p["level_qty_ints"])
    F.true("1 <= n_levels <= 5", 1 <= nl <= 5)
    F.eq("len(level_price_ints)", len(prices), nl)
    F.eq("len(level_qty_ints)", len(qtys), nl)
    lv = [r for r in _real(draw.tape) if int(r[C_OID]) != int(p["fok_oid"])]
    F.eq("level rows", len(lv), nl)
    for r, pp, q in zip(lv, prices, qtys):
        F.eq("level row side", int(r[C_SIDE]), rs)
        F.eq("level row price", int(r[C_PRICE]), int(pp))
        F.eq("level row qty", int(r[C_QTY]), int(q))
        F.eq("level row exec", int(r[C_EXEC]), NON)
    v = sum(int(q) for q in qtys)
    F.eq("v_int = sum(level qty)", p["v_int"], v)
    d = int(p["delta_ticks"])
    F.true("delta_ticks in {-1, 0, +1}", d in (-1, 0, 1))
    F.eq("req_int = V + delta", p["req_int"], v + d)
    worst = max(prices) if rs == SELL else min(prices)
    _fok_row(F, draw, rs, int(worst))
    _fok_verdict(F, draw, v)
    F.eq("accepted iff delta <= 0", draw.expected["accepted"], d <= 0)
    return F.out


@_findings_on_error
def _check_C10_fok_hidden(draw):
    F = _F("C10.fok_hidden")
    p, rows = draw.params, _rows(draw.tape)
    rs = int(p["rest_side"])
    total, peak, n = int(p["total_int"]), int(p["peak_int"]), int(p["n_slices"])
    F.eq("n_slices = ceil(total/peak)", n, _ceil_div(total, peak))
    F.true("n_slices >= 2", n >= 2)
    _row_is(F, rows, p["ice_oid"], "iceberg", side=rs, price=p["p_ice_int"],
            qty=total, exec_=ICE, peak=peak, ppd=0)
    F.eq("real rows", len(rows), 2)
    F.true("req > peak (only hidden volume can satisfy it)",
           int(p["req_int"]) > peak)
    _fok_row(F, draw, rs, int(p["p_ice_int"]))
    _fok_verdict(F, draw, total)
    return F.out


@_findings_on_error
def _check_C10_fok_icewpi(draw):
    F = _F("C10.fok_icewpi")
    p, rows = draw.params, _rows(draw.tape)
    rs = int(p["rest_side"])
    ppd = int(p["ppd_int"])
    F.true("ppd sign valid and non-zero", ppd > 0 if rs == SELL else ppd < 0)
    total, peak, n = int(p["total_int"]), int(p["peak_int"]), int(p["n_slices"])
    F.eq("n_slices = ceil(total/peak)", n, _ceil_div(total, peak))
    base = int(p["base_int"])
    _row_is(F, rows, p["ice_oid"], "iceberg", side=rs, price=base, qty=total,
            exec_=ICE, peak=peak, ppd=ppd)
    F.eq("real rows", len(rows), 2)
    limit = int(p["p_fok_int"])
    count = sum(1 for k in range(n) if _through(1 - rs, limit, base + k * ppd))
    F.true("at least one slice is reachable", count >= 1)
    F.eq("reachable_slices = #{k : base + k*ppd within limit}",
         p["reachable_slices"], count)
    _fok_row(F, draw, rs, None)
    _fok_verdict(F, draw, min(total, peak * count))
    return F.out


@_findings_on_error
def _check_C15_random_session(draw):
    F = _F("C15.random_session")
    p = draw.params
    n = int(p["n_orders"])
    real = _real(draw.tape)
    F.true("n_orders in {32, 96, 160}", n in (32, 96, 160))
    F.eq("real rows = n_orders", len(real), n)
    s = p["safe_from_index"]
    s = n if s is None else int(s)
    F.true("0 <= safe_from_index <= n_orders", 0 <= s <= n)
    for i in range(s, len(real)):
        row, prev = real[i], real[:i]
        if int(row[C_SIDE]) == BUY:
            asks = prev[prev[:, C_SIDE] == SELL][:, C_PRICE]
            if asks.size and not int(row[C_PRICE]) < int(asks.min()):
                F.out.append(f"C15: safe buy at index {i} is not below every ask")
        else:
            bids = prev[prev[:, C_SIDE] == BUY][:, C_PRICE]
            if bids.size and not int(row[C_PRICE]) > int(bids.max()):
                F.out.append(f"C15: safe sell at index {i} is not above every bid")
    return F.out


CHECKS = {key: globals()["_check_" + key.replace(".", "_")] for key in S1_KEYS}


# Rule-1 mutators: each corrupts a load-bearing value the check recomputes.
def _with_expected(draw, **updates):
    exp = _thaw(draw.expected)
    exp.update(updates)
    return dataclasses.replace(draw, expected=exp)


def _bump(draw, key, by=1):
    return _with_expected(draw, **{key: _canon(draw.expected[key]) + by})


def _flip(draw, key):
    return _with_expected(draw, **{key: not bool(draw.expected[key])})


def _mut_c01(d):
    exp = _thaw(d.expected)
    exp["verdicts"]["near_max_over_half_tick"] = True
    return dataclasses.replace(d, expected=exp)


def _mut_c03_ladder(d):
    j = int(d.params["j"])
    return _with_expected(d, filled_oids=list(d.params["at_touch_oids"])[1:j + 1])


def _mut_c08_incoming(d):
    tr = _thaw(d.expected["trades_int"])
    tr[-1][1] += 1
    return _with_expected(d, trades_int=tr)


def _mut_c15(d):
    par = _thaw(d.params)
    par["n_orders"] = int(par["n_orders"]) + 1
    return dataclasses.replace(d, params=par)


MUTATORS = {
    "C00.empty": lambda d: _with_expected(d, n_trades=1),
    "C00.pad_only": lambda d: _with_expected(d, n_resting=1),
    "C00.status_gate": lambda d: _with_expected(d, n_accepted=1),
    "C01.validation": _mut_c01,
    "C02.crossing_pair": lambda d: _bump(d, "trade_price_int"),
    "C03.priority_ladder": _mut_c03_ladder,
    "C03.same_owner_pair": lambda d: _bump(d, "trade_qty_int"),
    "C04.cancellation": lambda d: _with_expected(d, filled_oids=[d.params["cancel_oid"]]),
    "C04.iceberg_cancel_fok": lambda d: _flip(d, "accepted"),
    "C05.aggressor_residual": lambda d: _with_expected(
        d, second_fill_oid=d.params["interloper_oid"]),
    "C05.resting_residual": lambda d: _with_expected(
        d, second_fill_oid=d.params["interloper_oid"]),
    "C06.multi_level_walk": lambda d: _bump(d, "weighted_price_int"),
    "C06.multi_level_walk_stress": lambda d: _bump(d, "weighted_price_int"),
    "C07.ioc_regimes": lambda d: _bump(d, "traded_qty_int"),
    "C08.iceberg_cascade": lambda d: _bump(d, "n_slices_consumed"),
    "C08.rule3": lambda d: _bump(d, "iceberg_remaining_int"),
    "C08.incoming_iceberg": _mut_c08_incoming,
    "C09.icewpi_drift": lambda d: _bump(d, "hidden_remaining_int"),
    "C09.icewpi_wrong_sign": lambda d: _with_expected(d, n_trades=0),
    "C10.fok_visible": lambda d: _flip(d, "accepted"),
    "C10.fok_hidden": lambda d: _flip(d, "accepted"),
    "C10.fok_icewpi": lambda d: _bump(d, "matchable_qty_int"),
    "C15.random_session": _mut_c15,
}


# ===========================================================================
# Rule-1 self-tests of the shared checkers (run without the generator)
# ===========================================================================
def _hand_tape(rows):
    real = np.asarray(rows, dtype=np.int32).reshape(-1, 13)
    pad = np.tile(np.asarray(PAD_VALUES, dtype=np.int32), (3, 1))
    return np.ascontiguousarray(np.concatenate([real, pad], axis=0))


def _hand_row(oid, ts, side=SELL, price=1_005_000, qty=50, exec_=NON, peak=0, ppd=0):
    return [0, side, price, qty, oid, 1, ts, exec_, 0, 0, int(Status.ACTIVE), peak, ppd]


class TestCheckerSelfTests:
    """Plan rule 1: every shared checker is seen failing on a broken input."""

    def test_layout_checker_accepts_a_well_formed_hand_tape(self):
        tape = _hand_tape([_hand_row(1, 1), _hand_row(2, 2, side=BUY)])
        assert _layout_findings(tape) == []

    @pytest.mark.parametrize("breakage", [
        "float_dtype", "pad_before_real", "duplicate_oid", "ts_decreasing",
        "status_not_active", "price_out_of_bounds", "ice_peak_gt_qty",
        "non_ice_with_peak", "twelve_columns", "zero_timestamp",
    ])
    def test_layout_checker_flags_broken_tapes(self, breakage):
        tape = _hand_tape([_hand_row(1, 1), _hand_row(2, 2, side=BUY)])
        if breakage == "float_dtype":
            tape = tape.astype(np.float32)
        elif breakage == "pad_before_real":
            tape = np.ascontiguousarray(np.roll(tape, 1, axis=0))
        elif breakage == "duplicate_oid":
            tape[1, C_OID] = 1
        elif breakage == "ts_decreasing":
            tape[0, C_TS] = 5
        elif breakage == "status_not_active":
            tape[0, C_STATUS] = int(Status.FILLED)
        elif breakage == "price_out_of_bounds":
            tape[0, C_PRICE] = PRICE_MAX_INT + 1
        elif breakage == "ice_peak_gt_qty":
            tape[0, C_EXEC], tape[0, C_PEAK] = ICE, 51
        elif breakage == "non_ice_with_peak":
            tape[0, C_PEAK] = 10
        elif breakage == "twelve_columns":
            tape = np.ascontiguousarray(tape[:, :12])
        elif breakage == "zero_timestamp":
            tape[0, C_TS] = 0
        assert _layout_findings(tape) != []

    def test_integer_only_checker(self):
        assert _integer_only_findings({"a": 1, "b": [True, [2, 3]], "c": {"x": 4}}) == []
        assert _integer_only_findings({"a": 1.0}) != []
        assert _integer_only_findings({"a": [1, np.float64(2.0)]}) != []
        assert _integer_only_findings({"a": "text"}) != []
        assert _integer_only_findings({"a": np.zeros((2, 13), np.int32)}) != []

    def test_import_boundary_checker(self):
        ok = ("import hashlib\nimport numpy as np\n"
              "from omiecid_simulator.omiecid.encoding import encode_price\n"
              "from omiecid_simulator.jaxomiecid.omie_data_loader import PAD_ORDER_ROW\n")
        assert _import_boundary_findings(ok) == []
        for bad in (
            "from omiecid_simulator.omiecid.contract_book import ContractBook\n",
            "import omiecid_simulator.omiecid.omie_session\n",
            "from omiecid_simulator.omiecid import order_validator\n",
            "import jax.random\n",
            "from omiecid_simulator.jaxomiecid.omie_data_loader import row_to_order\n",
            "PAD = [0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0]\n",
        ):
            assert _import_boundary_findings(bad) != [], bad

    def test_s8_digest_recomputation_is_sensitive(self):
        base = types.SimpleNamespace(
            tape=_hand_tape([_hand_row(1, 1)]), aux_tapes={},
            params={"a": 1, "f": 0.5}, expected={"n": 1})
        d0 = _s8_draw_digest(base)
        assert len(d0) == 32
        t = base.tape.copy()
        t[0, C_QTY] += 1
        assert _s8_draw_digest(types.SimpleNamespace(**{**vars(base), "tape": t})) != d0
        assert _s8_draw_digest(types.SimpleNamespace(
            **{**vars(base), "aux_tapes": {"counterfactual": base.tape}})) != d0
        assert _s8_draw_digest(types.SimpleNamespace(
            **{**vars(base), "params": {"a": 2, "f": 0.5}})) != d0
        assert _s8_draw_digest(types.SimpleNamespace(
            **{**vars(base), "expected": {"n": 2}})) != d0


# ===========================================================================
# S10 — layout
# ===========================================================================
def test_col_indices_follow_the_loader_order():
    s = S()
    assert s.N_COLS == 13
    assert (s.COL_CONTRACT, s.COL_SIDE, s.COL_PRICE, s.COL_QTY, s.COL_OID,
            s.COL_AGENT, s.COL_TS, s.COL_EXEC, s.COL_VALIDITY, s.COL_EXPIRY,
            s.COL_STATUS, s.COL_PEAK, s.COL_PPD) == tuple(range(13))


@pytest.mark.parametrize("key", S1_KEYS)
def test_every_tape_has_the_engine_layout(key):
    for seed in SEEDS:
        draw = _draw(key, seed)
        assert _layout_findings(draw.tape) == [], (key, seed)
        for name, aux in draw.aux_tapes.items():
            assert _layout_findings(np.asarray(aux)) == [], (key, seed, name)


def test_make_empty_is_truly_empty():
    d = S().make_empty()
    assert isinstance(d.tape, np.ndarray) and d.tape.dtype == np.int32
    assert d.tape.shape == (0, 13)
    assert _n_real(d) == 0


@pytest.mark.parametrize("n_pad", [64, 5])
def test_make_pad_only_is_n_pad_pad_rows(n_pad):
    d = S().make_pad_only() if n_pad == 64 else S().make_pad_only(n_pad)
    assert d.tape.shape == (n_pad, 13)
    assert d.tape.dtype == np.int32
    assert all(r.tolist() == PAD_VALUES for r in d.tape)
    assert _n_real(d) == 0
    assert d.params["n_pad"] == n_pad


def test_validation_cases_carry_an_empty_tape():
    d = S().make_validation_cases(3)
    assert d.tape.shape == (0, 13)
    assert len(d.params["cases"]) >= len(C01_REQUIRED)


def test_fuzz_grid_has_120_cells_at_lengths_32_96_160():
    grid = S().fuzz_grid()
    assert len(grid) == 120
    assert {r.n_orders for r in grid} == {32, 96, 160}
    assert all(isinstance(r, S().FuzzRegime) for r in grid)
    crosses = {round(r.cross_prob, 6) for r in grid}
    assert min(crosses) == pytest.approx(0.05) and max(crosses) == pytest.approx(0.6)


def test_every_fuzz_cell_draws_its_full_length_inside_the_default_envelope():
    for i, regime in enumerate(S().fuzz_grid()):
        d = S().make_random_session(i, regime)
        assert _n_real(d) == regime.n_orders, (i, regime)
        assert _layout_findings(d.tape) == [], (i, regime)
        for axis in AXES:
            assert d.capacity_bounds[axis] <= DEFAULT_LIMITS[AXIS_FIELD[axis]], (i, axis)


def test_c15_timestamps_obey_the_tie_rule():
    """Ties only where tie_density allows them; a tie reuses the previous ts."""
    for regime in _fuzz_cells(n_orders=160):
        ts = _real(S().make_random_session(0, regime).tape)[:, C_TS]
        gaps = np.diff(ts)
        assert (gaps >= 0).all()
        if regime.tie_density == 0.0:
            assert (gaps >= 1).all(), "tie_density 0 must give strictly increasing ts"
        else:
            assert (gaps == 0).any(), "tie_density 0.4 over 160 orders shows no tie"


def test_c15_interarrival_gaps_are_poisson_not_a_unit_counter():
    """S7: gaps are 1 + floor(Exp(1)) ticks; the archive's unit counter is gone."""
    regime = next(r for r in _fuzz_cells(n_orders=160) if r.tie_density == 0.0)
    gaps = np.diff(_real(S().make_random_session(0, regime).tape)[:, C_TS])
    assert (gaps > 1).any(), "every gap is 1: still a unit counter"
    assert (gaps == 1).any()


# ===========================================================================
# S9 — the single pad sentinel
# ===========================================================================
def test_loader_alias_is_the_same_object():
    assert loader._PAD_ORDER_ROW is loader.PAD_ORDER_ROW


def test_loader_pad_row_type_and_values_are_unchanged():
    assert isinstance(loader.PAD_ORDER_ROW, list)
    assert loader.PAD_ORDER_ROW == PAD_VALUES


def test_generator_pad_row_is_the_loader_object():
    assert S().PAD_ORDER_ROW is loader.PAD_ORDER_ROW


def test_pad_to_and_stack_tapes_use_the_loader_sentinel():
    s = S()
    t = _hand_tape([_hand_row(1, 1), _hand_row(2, 2, side=BUY)])[:2]
    padded = s.pad_to(t, 6)
    assert padded.shape == (6, 13) and padded.dtype == np.int32
    assert np.array_equal(padded[:2], t)
    assert all(r.tolist() == PAD_VALUES for r in padded[2:])
    stacked = s.stack_tapes([t, t[:1]])
    assert stacked.ndim == 3 and stacked.shape[0] == 2 and stacked.shape[2] == 13
    assert stacked.dtype == np.int32
    assert all(r.tolist() == PAD_VALUES for r in stacked[1, 1:])


# ===========================================================================
# S4 — the Draw
# ===========================================================================
DRAW_FIELDS = {"construction", "seed", "tape", "params", "expected", "aux_tapes",
               "hop_bound", "capacity_bounds", "oracle_kind", "pending", "meta"}


def _oracle_kind_for(key):
    if key.startswith("C00."):
        return "exact-literal"
    if key == "C09.icewpi_wrong_sign":
        return "observation"
    if key.startswith("C15."):
        return "invariant"
    return "exact-parametric"


def test_draw_is_a_frozen_dataclass_with_the_s4_fields():
    s = S()
    assert dataclasses.is_dataclass(s.Draw)
    assert DRAW_FIELDS <= {f.name for f in dataclasses.fields(s.Draw)}
    d = _draw("C02.crossing_pair", 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.hop_bound = 0
    assert isinstance(d.digest(), str) and len(d.digest()) == 32


@pytest.mark.parametrize("key", S1_KEYS)
def test_draw_metadata_follows_s4(key):
    for seed in SEEDS:
        d = _draw(key, seed)
        assert d.construction == key
        assert d.oracle_kind == _oracle_kind_for(key)
        assert isinstance(d.pending, tuple) and all(isinstance(x, str) for x in d.pending)
        cb = d.capacity_bounds
        assert set(cb) == set(AXES), (key, sorted(cb))
        for axis in AXES:
            v = cb[axis]
            assert isinstance(v, (int, np.integer)) and not isinstance(v, bool), (key, axis)
            assert 0 <= v <= DEFAULT_LIMITS[AXIS_FIELD[axis]], (key, seed, axis, v)
        assert cb["hops"] == d.hop_bound
        assert isinstance(d.hop_bound, (int, np.integer))
        if key in SEEDLESS_KEYS:
            assert d.seed is None
        else:
            assert d.seed == seed
        json.dumps(_canon(d.params), allow_nan=False)  # JSON-serialisable
        assert isinstance(d.aux_tapes, Mapping)
        if key == "C09.icewpi_wrong_sign":
            assert dict(d.expected) == {}


@pytest.mark.parametrize("key", S1_KEYS)
def test_expected_is_integer_ticks_only(key):
    for seed in SEEDS:
        assert _integer_only_findings(_draw(key, seed).expected) == [], (key, seed)


@pytest.mark.parametrize("key", ["C03.same_owner_pair", "C07.ioc_regimes"])
def test_counterfactual_tapes_travel_in_aux_tapes(key):
    for seed in SEEDS:
        d = _draw(key, seed)
        assert "counterfactual" in d.aux_tapes
        cf = d.aux_tapes["counterfactual"]
        assert isinstance(cf, np.ndarray) and cf.dtype == np.int32 and cf.shape[1] == 13
        assert not any(isinstance(v, np.ndarray) for v in d.expected.values())


def test_rule3_withholds_the_hop_prices_pending_oq31():
    for seed in SEEDS:
        assert _draw("C08.rule3", seed).pending == ("rule3_hop_trade_prices:OQ-31",)


@pytest.mark.parametrize("key", S1_KEYS)
def test_resting_bound_counts_every_non_and_ice_order(key):
    """S7: resting bound per side = NON or ICE orders submitted on it, never decremented."""
    for seed in SEEDS:
        d = _draw(key, seed)
        real = _real(d.tape)
        rests = np.isin(real[:, C_EXEC], [NON, ICE])
        per_side = [int(((real[:, C_SIDE] == s) & rests).sum()) for s in (BUY, SELL)]
        assert d.capacity_bounds["resting_per_side"] == max(per_side), (key, seed)


# ===========================================================================
# S3 — registry and signatures
# ===========================================================================
def test_registry_is_a_read_only_mapping():
    reg = S().CONSTRUCTIONS
    assert isinstance(reg, types.MappingProxyType)
    with pytest.raises(TypeError):
        reg["C99.new"] = lambda seed: None  # type: ignore[index]


def test_registry_contains_every_s1_key():
    missing = set(S1_KEYS) - set(S().CONSTRUCTIONS)
    assert not missing, sorted(missing)


@pytest.mark.parametrize("key", [k for k in S1_KEYS if k != "C15.random_session"])
def test_registry_entry_is_its_mode_with_defaults(key):
    for seed in (0, 1):
        reg = _draw(key, seed)
        direct = MODE_CALLS[key](seed)
        assert direct.construction == key
        assert reg.digest() == direct.digest(), (key, seed)


def test_c15_registry_entry_draws_a_fuzz_session():
    for seed in (0, 1):
        d = _draw("C15.random_session", seed)
        assert _n_real(d) == int(d.params["n_orders"])


@pytest.mark.parametrize("call", [
    lambda s: s.make_cancellation(0, variant="bogus"),
    lambda s: s.make_partial_fill_residual(0, variant="bogus"),
    lambda s: s.make_ioc_regimes(0, regime="bogus"),
    lambda s: s.make_iceberg_cascade(0, variant="bogus"),
    lambda s: s.make_fok_boundary(0, variant="bogus"),
], ids=["cancellation", "partial_fill_residual", "ioc_regimes", "iceberg_cascade",
        "fok_boundary"])
def test_unknown_variant_raises_value_error(call):
    with pytest.raises(ValueError):
        call(S())


def test_iceberg_cascade_no_longer_takes_the_archive_rule3_bool():
    with pytest.raises(TypeError):
        S().make_iceberg_cascade(0, rule3=True)


@pytest.mark.parametrize("regime", ["none", "partial", "full"])
def test_ioc_regime_keyword_selects_the_regime(regime):
    for seed in (0, 1, 2):
        d = S().make_ioc_regimes(seed, regime=regime)
        assert d.params["regime"] == regime
        assert _check_C07_ioc_regimes(d) == []


def test_ioc_regime_none_draws_every_regime_from_the_stream():
    seen = {_draw("C07.ioc_regimes", s).params["regime"] for s in SEEDS}
    seen |= {S().make_ioc_regimes(s).params["regime"] for s in range(8, 24)}
    assert seen == {"none", "partial", "full"}


def test_status_gate_tape_is_the_crossing_pair_tape():
    for seed in SEEDS:
        assert np.array_equal(_draw("C00.status_gate", seed).tape,
                              _draw("C02.crossing_pair", seed).tape)


def test_wrong_sign_is_the_drift_draw_with_the_ppd_sign_flipped():
    for seed in SEEDS:
        good, bad = _draw("C09.icewpi_drift", seed), _draw("C09.icewpi_wrong_sign", seed)
        want = good.tape.copy()
        i = _row_index(want, good.params["ice_oid"])
        want[i, C_PPD] = -want[i, C_PPD]
        assert np.array_equal(bad.tape, want), seed
        assert bad.params["ppd_int"] == -good.params["ppd_int"]


@pytest.mark.parametrize("key", ["C10.fok_visible", "C10.fok_hidden", "C10.fok_icewpi"])
def test_fok_draws_land_on_both_sides_of_the_boundary(key):
    verdicts = {bool(MODE_CALLS[key](s).expected["accepted"]) for s in range(32)}
    assert verdicts == {True, False}, key


def test_fok_visible_draws_every_delta():
    deltas = {int(MODE_CALLS["C10.fok_visible"](s).params["delta_ticks"]) for s in range(32)}
    assert deltas == {-1, 0, 1}


# ===========================================================================
# S5 — TapeBuilder tick arithmetic
# ===========================================================================
def test_tapebuilder_writes_integer_fields_verbatim():
    b = S().TapeBuilder(PRODUCT)
    o1 = b.add(side=Side.SELL, price_int=1_012_345, qty_int=57)
    o2 = b.add(side=Side.BUY, price_int=1_012_400, qty_int=120,
               exec_restriction=ExecRestriction.ICE, peak_int=30, ppd_int=-7)
    tape = b.build()
    rows = _rows(tape)
    assert set(rows) == {int(o1), int(o2)}
    r1, r2 = rows[int(o1)], rows[int(o2)]
    assert (int(r1[C_PRICE]), int(r1[C_QTY]), int(r1[C_SIDE])) == (1_012_345, 57, SELL)
    assert (int(r2[C_PRICE]), int(r2[C_QTY]), int(r2[C_PEAK]), int(r2[C_PPD])) == (
        1_012_400, 120, 30, -7)
    assert int(r2[C_EXEC]) == ICE
    assert _layout_findings(tape) == []


def test_tapebuilder_encodes_float_fields_through_encoding():
    b = S().TapeBuilder(PRODUCT)
    oid = b.add(side=Side.BUY, price_eur=47.13, qty_mw=12.3,
                exec_restriction=ExecRestriction.ICE, peak_qty_mw=4.1, ppd_eur=-0.05)
    r = _rows(b.build())[int(oid)]
    assert int(r[C_PRICE]) == encode_price(47.13, PRODUCT)
    assert int(r[C_QTY]) == encode_qty(12.3, PRODUCT)
    assert int(r[C_PEAK]) == encode_qty(4.1, PRODUCT)
    assert int(r[C_PPD]) == -5


@pytest.mark.parametrize("both", [
    {"price_eur": 50.0, "price_int": 1_004_900},
    {"qty_mw": 2.0, "qty_int": 20},
    {"peak_qty_mw": 1.0, "peak_int": 10},
    {"ppd_eur": 0.05, "ppd_int": 5},
], ids=["price", "qty", "peak", "ppd"])
def test_tapebuilder_refuses_float_and_int_for_the_same_field(both):
    kw = {"price_int": 1_004_900, "qty_int": 20}
    kw.update(both)
    if "peak_qty_mw" in both or "ppd_eur" in both:
        kw["exec_restriction"] = ExecRestriction.ICE
        kw.setdefault("peak_int", 10)
    b = S().TapeBuilder(PRODUCT)
    with pytest.raises((ValueError, TypeError)):
        b.add(side=Side.SELL, **kw)


@pytest.mark.parametrize("missing", ["price", "qty"])
def test_tapebuilder_refuses_a_missing_price_or_qty(missing):
    kw = {"price_int": 1_004_900, "qty_int": 20}
    kw.pop(f"{missing}_int")
    with pytest.raises((ValueError, TypeError)):
        S().TapeBuilder(PRODUCT).add(side=Side.SELL, **kw)


def test_tape_digest_is_a_stable_hex_string():
    t = _hand_tape([_hand_row(1, 1)])
    a, b = S().tape_digest(t), S().tape_digest(t.copy())
    assert a == b and isinstance(a, str) and len(a) == 32
    t2 = t.copy()
    t2[0, C_QTY] += 1
    assert S().tape_digest(t2) != a


# ===========================================================================
# Determinism
# ===========================================================================
DETERMINISM_KEYS = ("C02.crossing_pair", "C08.iceberg_cascade", "C15.random_session")


@pytest.mark.parametrize("key", DETERMINISM_KEYS)
def test_same_seed_same_draw_in_process(key):
    reg = S().CONSTRUCTIONS[key]
    for seed in SEEDS:
        a = reg(seed)
        np.random.seed(12345)  # ambient numpy state must not leak into a draw
        _ = np.random.random(17)
        b = reg(seed)
        assert np.array_equal(a.tape, b.tape) and a.digest() == b.digest(), (key, seed)


@pytest.mark.parametrize("key", [k for k in S1_KEYS if k not in SEEDLESS_KEYS])
def test_different_seeds_give_different_draws(key):
    digests = {_draw(key, s).digest() for s in SEEDS}
    assert len(digests) > 1, key


@pytest.mark.slow
def test_same_seed_same_draw_in_a_subprocess():
    """slow: a fresh interpreter (CPU-only env) pays the package import cost once."""
    script = (
        "import json\n"
        "from omiecid_simulator.jaxomiecid import omie_tape_synth as s\n"
        f"keys = {list(DETERMINISM_KEYS)!r}\n"
        "print(json.dumps({k: [s.CONSTRUCTIONS[k](i).digest() for i in range(8)] "
        "for k in keys}))\n"
    )
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO_ROOT),
           "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "WANDB_MODE": "disabled"}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env=env, cwd=str(REPO_ROOT), timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    want = {k: [S().CONSTRUCTIONS[k](i).digest() for i in SEEDS] for k in DETERMINISM_KEYS}
    assert got == want


# ===========================================================================
# S8 — digests
# ===========================================================================
@pytest.mark.parametrize("key", S1_KEYS)
def test_draw_digest_follows_the_s8_algorithm(key):
    for seed in (0, 1):
        d = _draw(key, seed)
        assert d.digest() == _s8_draw_digest(d), (key, seed)


def test_digest_is_sensitive_to_one_tape_byte():
    d = _draw("C02.crossing_pair", 0)
    t = d.tape.copy()
    t[0, C_QTY] += 1
    assert dataclasses.replace(d, tape=t).digest() != d.digest()


def test_digest_is_sensitive_to_one_aux_byte():
    d = _draw("C03.same_owner_pair", 0)
    aux = {k: np.array(v, copy=True) for k, v in d.aux_tapes.items()}
    aux["counterfactual"][0, C_AGENT] += 1
    assert dataclasses.replace(d, aux_tapes=aux).digest() != d.digest()


def test_digest_is_sensitive_to_one_params_value():
    d = _draw("C02.crossing_pair", 0)
    par = _thaw(d.params)
    par["gap_ticks"] = int(par["gap_ticks"]) + 1
    assert dataclasses.replace(d, params=par).digest() != d.digest()


def test_digest_is_sensitive_to_one_expected_value():
    d = _draw("C02.crossing_pair", 0)
    assert _bump(d, "trade_qty_int").digest() != d.digest()


def test_digest_fixture_has_the_s8_schema():
    fx = _load_fixture()
    assert fx["schema_version"] == 1
    assert fx["algorithm"] == "blake2b-128"
    assert fx["seeds"] == list(range(8))
    assert all(isinstance(v, str) and len(v) == 32 for v in fx["constructions"].values())


def test_registry_and_fixture_keys_are_equal():
    """A mode registered without a digest (or a stale digest entry) fails here."""
    assert set(_load_fixture()["constructions"]) == set(S().CONSTRUCTIONS)


@pytest.mark.parametrize("key", _registry_keys())
def test_recomputed_construction_digest_equals_the_stored_one(key):
    stored = _load_fixture()["constructions"].get(key)
    assert stored is not None, f"{key} has no stored digest"
    got = _construction_digest(S().CONSTRUCTIONS[key](s).digest() for s in SEEDS)
    assert got == stored, (
        f"{key}'s draws changed. Digests are per code version (AC 7): update "
        "synth_digests.json in the same commit and say why."
    )


def test_compute_digest_table_matches_the_fixture_and_the_recomputation():
    table = dict(S().compute_digest_table())
    assert table == _load_fixture()["constructions"]
    for key in S1_KEYS:
        assert table[key] == _construction_digest(_draw(key, s).digest() for s in SEEDS)


@pytest.mark.slow
def test_write_digests_cli_writes_the_s8_schema(tmp_path):
    """slow: runs the module CLI in a fresh interpreter."""
    target = tmp_path / "synth_digests.json"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO_ROOT),
           "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "WANDB_MODE": "disabled"}
    out = subprocess.run(
        [sys.executable, "-m", "omiecid_simulator.jaxomiecid.omie_tape_synth",
         "--write-digests", str(target)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    fx = json.loads(target.read_text())
    assert fx["schema_version"] == 1 and fx["algorithm"] == "blake2b-128"
    assert fx["seeds"] == list(range(8))
    assert fx["constructions"] == dict(S().compute_digest_table())


@pytest.mark.parametrize("key", sorted(ARCHIVE_TAPE_DIGESTS))
def test_archive_continuity_of_the_unchanged_constructions(key):
    """AC 1 / AC 12: tape-only bytes of seeds 0-7 equal the archive literals."""
    got = _tape_only_digest(_draw(key, s).tape for s in SEEDS)
    assert got == ARCHIVE_TAPE_DIGESTS[key]


# ===========================================================================
# S6 — the I8 refusal API
# ===========================================================================
def test_envelope_defaults_are_the_i8_margins():
    env = S().CapacityEnvelope()
    for f, v in DEFAULT_LIMITS.items():
        assert getattr(env, f) == v
    assert S().DEFAULT_ENVELOPE == env


def test_envelope_axes_are_four_independent_fields():
    names = {f.name for f in dataclasses.fields(S().CapacityEnvelope)}
    assert names == set(DEFAULT_LIMITS)
    env = S().CapacityEnvelope(max_resting_per_side=12, max_concurrent_icebergs=900)
    assert (env.max_resting_per_side, env.max_trades, env.max_hops,
            env.max_concurrent_icebergs) == (12, 80, 160, 900)


def test_envelope_is_frozen():
    env = S().CapacityEnvelope()
    with pytest.raises(dataclasses.FrozenInstanceError):
        env.max_trades = 1


@pytest.mark.parametrize("field", sorted(DEFAULT_LIMITS))
@pytest.mark.parametrize("bad", ["zero", "negative", "at_ceiling", "above_ceiling",
                                 "float", "str"])
def test_envelope_post_init_refuses_values_outside_the_margins(field, bad):
    ceiling = AXIS_CEILING[field]
    value = {"zero": 0, "negative": -3, "at_ceiling": ceiling,
             "above_ceiling": ceiling + 7, "float": 10.5, "str": "10"}[bad]
    with pytest.raises(ValueError):
        S().CapacityEnvelope(**{field: value})


@pytest.mark.parametrize("field", sorted(DEFAULT_LIMITS))
def test_envelope_accepts_one_and_one_below_the_ceiling(field):
    for v in (1, AXIS_CEILING[field] - 1):
        assert getattr(S().CapacityEnvelope(**{field: v}), field) == v


CUSTOM = {"max_resting_per_side": 11, "max_trades": 23, "max_hops": 37,
          "max_concurrent_icebergs": 41}


def _custom_env():
    return S().CapacityEnvelope(**CUSTOM)


@pytest.mark.parametrize("axis", AXES)
def test_check_envelope_limit_is_inclusive_and_limit_plus_one_breaches(axis):
    s, env = S(), _custom_env()
    zeros = {a: 0 for a in AXES}
    limit = CUSTOM[AXIS_FIELD[axis]]
    assert s.check_envelope(env, **{**zeros, axis: limit}) is None
    with pytest.raises(s.CapacityBreach) as ei:
        s.check_envelope(env, **{**zeros, axis: limit + 1})
    assert ei.value.axis == axis
    assert ei.value.value == limit + 1
    assert ei.value.limit == limit


def test_capacity_breach_is_a_value_error():
    assert issubclass(S().CapacityBreach, ValueError)


def test_check_envelope_axes_are_keyword_required():
    s, env = S(), _custom_env()
    with pytest.raises(TypeError):
        s.check_envelope(env, 0, 0, 0, 0)
    for axis in AXES:
        kw = {a: 0 for a in AXES if a != axis}
        with pytest.raises(TypeError):
            s.check_envelope(env, **kw)


def test_check_envelope_reports_the_first_breach_in_axis_order():
    s, env = S(), _custom_env()
    over = {a: CUSTOM[AXIS_FIELD[a]] + 1 for a in AXES}
    for i, axis in enumerate(AXES):
        kw = {a: (over[a] if j >= i else 0) for j, a in enumerate(AXES)}
        with pytest.raises(s.CapacityBreach) as ei:
            s.check_envelope(env, **kw)
        assert ei.value.axis == axis


def test_c14_opt_out_is_accepted_by_check_envelope():
    s = S()
    assert s.check_envelope(s.C14_CEILING_OPT_OUT, resting_per_side=10**6,
                            trades=10**6, hops=10**6, concurrent_icebergs=10**6) is None


def test_none_envelope_is_not_an_implicit_opt_out():
    with pytest.raises(TypeError):
        S().check_envelope(None, resting_per_side=0, trades=0, hops=0,
                           concurrent_icebergs=0)


@pytest.mark.parametrize("key", ENVELOPE_KEYS)
def test_every_s1_mode_refuses_the_opt_out_and_none(key):
    with pytest.raises(ValueError):
        MODE_CALLS[key](0, envelope=S().C14_CEILING_OPT_OUT)
    with pytest.raises(TypeError):
        MODE_CALLS[key](0, envelope=None)


def test_random_session_refuses_the_opt_out_and_none():
    regime = S().fuzz_grid()[0]
    with pytest.raises(ValueError):
        S().make_random_session(0, regime, envelope=S().C14_CEILING_OPT_OUT)
    with pytest.raises(TypeError):
        S().make_random_session(0, regime, envelope=None)


@pytest.mark.parametrize("key", ENVELOPE_KEYS)
def test_bounds_are_wired_through_check_envelope(key):
    """For each axis with bound b >= 2: envelope b passes (same draw), b - 1 raises.

    Axes with b == 1 cannot be tightened to 0 (``__post_init__`` refuses it), and
    axes with b == 0 are excluded by the card ("each axis with a non-zero bound").

    Re-baselined by OMIE-CERT-HARNESS-2 (licensed, card re-baseline list (ii)):
    ``C09.icewpi_wrong_sign`` is exempt from the ``exercised > 0`` guard. Its
    wrong-sign BUY ICE is rejectable (market-spec §13.6), so the generator no longer
    counts it, and the lone SELL NON left has no crossable counterparty (no ``+ 2``
    hop term). Its bounds are (1, 0, 1, 0): no axis is >= 2, so instead every one
    of its four bounds must be <= 1.
    """
    s = S()
    exercised = 0
    for seed in SEEDS:
        d = MODE_CALLS[key](seed)
        if key == "C09.icewpi_wrong_sign":
            assert all(int(d.capacity_bounds[a]) <= 1 for a in AXES), (
                key, seed, dict(d.capacity_bounds))
        for axis in AXES:
            b = int(d.capacity_bounds[axis])
            if b < 2:
                continue
            field = AXIS_FIELD[axis]
            same = MODE_CALLS[key](seed, envelope=s.CapacityEnvelope(**{field: b}))
            assert same.digest() == d.digest(), (key, seed, axis)
            with pytest.raises(s.CapacityBreach) as ei:
                MODE_CALLS[key](seed, envelope=s.CapacityEnvelope(**{field: b - 1}))
            assert ei.value.axis == axis, (key, seed, axis, ei.value.axis)
            exercised += 1
    if key == "C09.icewpi_wrong_sign":
        return
    assert exercised > 0, f"{key}: no axis had a bound >= 2 (hop_bound must be >= 2)"


def test_c15_adapts_to_a_tight_trade_envelope_instead_of_raising():
    s = S()
    env = s.CapacityEnvelope(max_trades=20)
    for regime in _fuzz_cells(n_orders=160, cross_prob=0.6):
        for seed in SEEDS:
            d = s.make_random_session(seed, regime, envelope=env)
            assert _n_real(d) == regime.n_orders
            for axis in AXES:
                assert d.capacity_bounds[axis] <= getattr(env, AXIS_FIELD[axis]), (
                    regime, seed, axis)
            assert _check_C15_random_session(d) == []


def test_c15_non_only_160_cannot_fit_40_resting_per_side():
    s = S()
    regime = _fuzz_cells(n_orders=160, non_only=True)[0]
    with pytest.raises(s.CapacityBreach) as ei:
        s.make_random_session(0, regime, envelope=s.CapacityEnvelope(max_resting_per_side=40))
    assert ei.value.axis == "resting_per_side"


def test_c15_safe_mode_engages_and_the_check_catches_an_unsafe_order():
    s = S()
    regime = next(r for r in _fuzz_cells(n_orders=160, cross_prob=0.6, non_only=True)
                  if r.tie_density == 0.0)
    d = s.make_random_session(0, regime, envelope=s.CapacityEnvelope(max_trades=20))
    sf = d.params["safe_from_index"]
    assert sf is not None and int(sf) < regime.n_orders, "safe mode never engaged"
    assert _check_C15_random_session(d) == []
    tape = d.tape.copy()
    real = _real(tape)
    for i in range(int(sf), len(real)):
        prev = real[:i]
        opp = prev[prev[:, C_SIDE] == (SELL if int(real[i, C_SIDE]) == BUY else BUY)]
        if opp.size:
            tape[i, C_PRICE] = (opp[:, C_PRICE].min() if int(real[i, C_SIDE]) == BUY
                                else opp[:, C_PRICE].max())
            break
    else:
        pytest.fail("no safe order with an earlier opposite order to mutate")
    assert _check_C15_random_session(dataclasses.replace(d, tape=tape)) != []


# ===========================================================================
# S12 — import boundary
# ===========================================================================
def test_generator_respects_the_import_boundary():
    src = Path(S().__file__).read_text()
    assert _import_boundary_findings(src) == []


# ===========================================================================
# Spot checks: one hand-derivable check per S1 key (rule 1 on each)
# ===========================================================================
def test_every_s1_key_has_a_spot_check_and_a_mutator():
    assert set(CHECKS) == set(S1_KEYS) == set(MUTATORS)


@pytest.mark.parametrize("key", S1_KEYS)
def test_spot_check_holds_on_the_real_draw(key):
    for seed in SEEDS:
        assert CHECKS[key](_draw(key, seed)) == [], (key, seed)


@pytest.mark.parametrize("key", S1_KEYS)
def test_spot_check_fails_on_a_mutated_draw(key):
    for seed in (0, 1):
        mutated = MUTATORS[key](_draw(key, seed))
        assert CHECKS[key](mutated) != [], (key, seed)
