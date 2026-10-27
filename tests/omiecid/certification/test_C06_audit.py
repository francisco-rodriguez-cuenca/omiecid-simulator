"""K4 audit twin for OMIE-CERT-C06-1 — multi-level walk and volume-weighted price.

Written in fresh context, adversarially, from the specification only:

* market-spec §6.1 — price-time priority: the resting SELL side ranks lowest price first, the
  resting BUY side highest price first, ties by earliest timestamp. This fixes the order in which
  levels are consumed.
* market-spec §6.2 items 1-3 and 6 — the incoming order matches the best counterpart while
  ``sell_price <= buy_price``; each trade executes at the **resting** order's limit price; when a
  level is exhausted matching continues at the next best level.
* market-spec §8.1 — one trade per level, value ``Q x p``; the certification compares the integer
  numerator ``sum(price_int * qty_int)`` and denominator ``sum(qty_int)`` **before any division**.
* §2.3 — order quantity bounded by ``qty_max`` (9990 ticks) and price by +-9999 EUR, so the
  largest single per-trade product is ``1_999_800 * 9990 ~= 2.0e10``: far beyond int32. The int64
  accumulator is an ENGINEERING INVARIANT (OQ-18 context), not a spec rule.

The expectation is re-derived here from the generated tape **rows only** (never from the
generator's own ``Draw.expected``, never from the notebook), using Python's unbounded ints after
an explicit ``int()`` of every numpy int32 cell, and asserted against the **observed** and the
**expected** values the emitted ``C06.json`` records — whatever its status.

Independence: this file does not open or import ``C06_multi_level_walk_vwap.ipynb``,
``contract_book.py``, ``iceberg_manager.py``, ``order_validator.py`` or ``settlement.py``.
"""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_tape_synth import (
    DEFAULT_ENVELOPE,
    construction_digest,
    make_multi_level_walk,
)
from notebooks.certification.cert_support import derive_failure_kinds, validate_results

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO / "notebooks" / "certification" / "results"

WALK = "C06.multi_level_walk"
STRESS = "C06.multi_level_walk_stress"
INT32_MAX = 2**31 - 1
QTY_MAX_INT = 9990            # §2.3: 999.0 MW at 0.1 MW per tick
PRICE_INT_MAX = 1_999_800     # §2.3: +9999 EUR offset-encoded at 0.01 EUR per tick
PRODUCT_CEILING = PRICE_INT_MAX * QTY_MAX_INT   # ~2.0e10: the largest single price x qty
BATTERY = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")

# tape column layout (13-column OMIE order row)
_SIDE, _PRICE, _QTY, _OID, _TS = 1, 2, 3, 4, 6
BUY, SELL = 0, 1

_NAME = re.compile(
    r"^(?P<cons>C06\.multi_level_walk(?:_stress)?) seed (?P<seed>\d+)(?P<eager> \(eager\))?: "
    r"\((?P<letter>[A-Z])\)"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _results_dir() -> Path:
    env = os.environ.get("OMIE_CERT_RESULTS_DIR")
    return Path(env) if env else DEFAULT_RESULTS_DIR


def _load(results_dir: Path) -> Dict[str, Any]:
    return json.loads((results_dir / "C06.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    path = _results_dir() / "C06.json"
    if not path.exists():
        pytest.skip(f"C06 results JSON not emitted yet: {path} is absent (K3/K5 has not run)")
    return _load(_results_dir())


# ---------------------------------------------------------------------------
# Independent derivation (from the tape rows only, exact Python integers)
# ---------------------------------------------------------------------------
def _real_rows(tape: np.ndarray) -> List[Tuple[int, ...]]:
    # int() on every cell: from here on, every product is an unbounded Python int.
    return [tuple(int(x) for x in r) for r in np.asarray(tape) if int(r[_OID]) >= 1]


def derive(tape: np.ndarray) -> Dict[str, Any]:
    """The spec answer for one tape: L resting rows on one side, then one aggressor."""
    rows = _real_rows(tape)
    assert len(rows) >= 2, "a walk needs a ladder and an aggressor"
    agg, ladder = rows[-1], rows[:-1]
    rest_side = ladder[0][_SIDE]
    assert all(r[_SIDE] == rest_side for r in ladder), "ladder rows must all rest on one side"
    assert agg[_SIDE] != rest_side, "the last row must be the opposite-side aggressor"
    # §6.1 ranking of the resting side.
    if rest_side == SELL:
        ranked = sorted(ladder, key=lambda r: (r[_PRICE], r[_TS]))
    else:
        ranked = sorted(ladder, key=lambda r: (-r[_PRICE], r[_TS]))
    remaining = agg[_QTY]
    fills: List[int] = []
    trades: List[List[int]] = []
    for lvl in ranked:
        crosses = lvl[_PRICE] <= agg[_PRICE] if agg[_SIDE] == BUY else lvl[_PRICE] >= agg[_PRICE]
        q = min(remaining, lvl[_QTY]) if (crosses and remaining > 0) else 0
        fills.append(q)
        if q:
            # §6.2 item 3: the trade prints at the resting order's own limit.
            trades.append([lvl[_OID], agg[_OID], lvl[_PRICE], q, len(rows) - 1])
            remaining -= q
    book = {"BUY": [], "SELL": []}
    for lvl, q in zip(ranked, fills):
        left = lvl[_QTY] - q
        if left > 0:
            book["SELL" if rest_side == SELL else "BUY"].append([lvl[_PRICE], left, lvl[_OID]])
    if remaining > 0:  # NON residual (§4.5) would rest; never expected in C06
        book["BUY" if agg[_SIDE] == BUY else "SELL"].append([agg[_PRICE], remaining, agg[_OID]])
    total = sum(t[3] for t in trades)
    numer = sum(t[2] * t[3] for t in trades)
    whole = sum(1 for lvl, q in zip(ranked, fills) if q and q == lvl[_QTY])
    partial = next((q for lvl, q in zip(ranked, fills) if q and q < lvl[_QTY]), 0)
    prices = [t[2] for t in trades]
    return {
        "L": len(ladder),
        "m": whole,
        "r": partial,
        "rest_side": rest_side,
        "accepted": [r[_OID] for r in rows],
        "V": fills,
        "Q": trades,
        "K": [total, numer, len(trades)],
        "B": book,
        "agg_residual": remaining,
        "bracket": [min(prices) - 1, max(prices) + 1] if prices else None,
        "max_product": max((t[2] * t[3] for t in trades), default=0),
        "submission": len(rows) - 1,
    }


def _make(cons: str, seed: int):
    return make_multi_level_walk(seed, stress=(cons == STRESS))


_DERIVED: Dict[Tuple[str, int], Dict[str, Any]] = {}


def derived_for(cons: str, seed: int) -> Dict[str, Any]:
    key = (cons, seed)
    if key not in _DERIVED:
        _DERIVED[key] = derive(_make(cons, seed).tape)
    return _DERIVED[key]


# ---------------------------------------------------------------------------
# Indexing the emitted assertions
# ---------------------------------------------------------------------------
def index(p: Mapping[str, Any]) -> Tuple[Dict[Tuple[str, int, str, bool], dict], List[str]]:
    idx: Dict[Tuple[str, int, str, bool], dict] = {}
    errs: List[str] = []
    for a in p.get("assertions", []):
        m = _NAME.match(a.get("name", ""))
        if not m:
            continue
        key = (m["cons"], int(m["seed"]), m["letter"], bool(m["eager"]))
        if key in idx:
            errs.append(f"duplicate per-draw assertion {key}")
        if a.get("seed") != int(m["seed"]):
            errs.append(f"{a['name']!r}: seed field {a.get('seed')!r} != name seed {m['seed']}")
        idx[key] = a
    return idx, errs


def draws_in(p) -> List[Tuple[str, int]]:
    idx, _ = index(p)
    return sorted({(c, s) for (c, s, _l, _e) in idx})


# ---------------------------------------------------------------------------
# Checks — each returns a list of tagged error strings (empty = clean)
# ---------------------------------------------------------------------------
def check_presence(p) -> List[str]:
    idx, errs = index(p)
    draws = draws_in(p)
    walk = sorted(s for c, s in draws if c == WALK)
    stress = sorted(s for c, s in draws if c == STRESS)
    if len(walk) < 500:
        errs.append(f"[presence] only {len(walk)} walk draws (default is 500)")
    if not stress:
        errs.append("[presence] no stress draw at the ~2e10 product boundary")
    for c, s in draws:
        for letter in "AQVKBE":
            if (c, s, letter, False) not in idx:
                errs.append(f"[presence] {c} seed {s}: ({letter}) missing")
    n = len(draws)
    if p["seeds"]["count"] != n or p["draws"]["valid"] != n:
        errs.append(f"[presence] seeds.count={p['seeds']['count']} draws.valid={p['draws']['valid']} "
                    f"!= {n} draws carrying per-draw evidence")
    if p["draws"]["requested"] != p["draws"]["valid"] + p["draws"]["discarded"]:
        errs.append("[presence] requested != valid + discarded")
    return errs


def _check_letter(p, letter: str, want_key: str, tag: str) -> List[str]:
    idx, errs = index(p)
    for (c, s, l, eager), a in sorted(idx.items(), key=lambda kv: kv[0]):
        if l != letter or eager:
            continue
        want = derived_for(c, s)[want_key]
        for side in ("observed", "expected"):
            if a.get(side) != want:
                errs.append(f"[{tag}] {c} seed {s}: {side} {a.get(side)!r} != derived {want!r}")
        if a.get("status") != "pass":
            errs.append(f"[{tag}] {c} seed {s}: status {a.get('status')!r} although derivation matches"
                        if a.get("observed") == want else f"[{tag}] {c} seed {s}: status fail")
    return errs


def check_fill_vector(p) -> List[str]:
    return _check_letter(p, "V", "V", "V")


def check_triple(p) -> List[str]:
    errs = _check_letter(p, "K", "K", "K")
    idx, _ = index(p)
    for (c, s, l, eager), a in idx.items():
        if l == "K" and not eager:
            obs = a.get("observed")
            # Each component must be a true integer (not a float that rounded the numerator).
            if not (isinstance(obs, list) and len(obs) == 3 and all(type(x) is int for x in obs)):
                errs.append(f"[K] {c} seed {s}: observed triple is not three exact ints: {obs!r}")
    return errs


def check_trades(p) -> List[str]:
    return _check_letter(p, "Q", "Q", "Q")


def check_book(p) -> List[str]:
    """(B) is serialised in ascending-price order on both sides (a presentation convention of the
    results JSON, not a ranking claim — §6.1 ranking is carried by (Q)/(V) and by I4). Content is
    compared exactly, in that canonical order, so a reordered-but-wrong row still fails."""
    for (c, s) in draws_in(p):
        d = derived_for(c, s)
        d.setdefault("B_canon", {k: sorted(v, key=lambda r: (r[0], r[2])) for k, v in d["B"].items()})
    return _check_letter(p, "B", "B_canon", "B")


def check_accepted(p) -> List[str]:
    errs: List[str] = []
    idx, e0 = index(p)
    errs += e0
    for (c, s, l, eager), a in idx.items():
        if l == "A" and not eager:
            want = [derived_for(c, s)["accepted"], []]
            if a.get("observed") != want or a.get("expected") != want:
                errs.append(f"[A] {c} seed {s}: {a.get('observed')!r} != {want!r}")
    return errs


def check_engine_report(p) -> List[str]:
    """ENGINEERING INVARIANT (int64 accumulator): reported qty == total; reported price is the
    VWAP numerator / denominator within one tick, and inside the fill-price bracket +-1."""
    errs: List[str] = []
    idx, _ = index(p)
    for (c, s, l, eager), a in idx.items():
        if l != "E":
            continue
        d = derived_for(c, s)
        total, numer, _n = d["K"]
        obs, exp = a.get("observed") or {}, a.get("expected") or {}
        tag = f"[E] {c} seed {s}{' (eager)' if eager else ''}"
        if exp.get("qty_int") != total or exp.get("price_int_in") != d["bracket"]:
            errs.append(f"{tag}: expected {exp!r} != derived qty {total}, bracket {d['bracket']}")
        if obs.get("traded") is not True or obs.get("qty_int") != total:
            errs.append(f"{tag}: observed {obs!r}, derived qty {total}")
            continue
        if round(float(obs.get("trade_qty_mw", -1)) * 10) != total:
            errs.append(f"{tag}: trade_qty_mw {obs.get('trade_qty_mw')!r} does not encode {total}")
        pi = obs.get("price_int")
        lo, hi = d["bracket"]
        if not isinstance(pi, int) or not (lo <= pi <= hi):
            errs.append(f"{tag}: reported price_int {pi!r} outside bracket [{lo}, {hi}]")
        elif abs(pi * total - numer) > total:   # |p - W/Q| <= 1 tick, in exact integers
            errs.append(f"{tag}: reported price_int {pi} is not the VWAP {numer}/{total} within 1 tick")
        if a.get("status") != "pass":
            errs.append(f"{tag}: status {a.get('status')!r}")
    return errs


def check_stress(p) -> List[str]:
    errs: List[str] = []
    idx, _ = index(p)
    stress = [s for c, s in draws_in(p) if c == STRESS]
    for s in stress:
        d = derived_for(STRESS, s)
        if not (d["max_product"] > INT32_MAX and d["max_product"] >= PRODUCT_CEILING // 2):
            errs.append(f"[stress] seed {s}: largest product {d['max_product']} is not at the "
                        f"~2e10 boundary (ceiling {PRODUCT_CEILING})")
        if d["max_product"] > PRODUCT_CEILING:
            errs.append(f"[stress] seed {s}: product {d['max_product']} beyond the §2.3 ceiling")
        k = idx.get((STRESS, s, "K", False))
        if k is None or k.get("observed", [None, None])[1] != d["K"][1]:
            errs.append(f"[stress] seed {s}: numerator missing or wrong (derived {d['K'][1]})")
    return errs


def check_eager_jit(p) -> List[str]:
    errs: List[str] = []
    idx, _ = index(p)
    if p.get("eager_jit_identical") is not True:
        errs.append(f"[eager_jit] eager_jit_identical = {p.get('eager_jit_identical')!r}")
    ej = [a for a in p.get("assertions", []) if a.get("kind") == "eager_jit"]
    if not ej:
        errs.append("[eager_jit] no eager-vs-jitted evidence in the results (absence = failure)")
    for a in ej:
        if a.get("status") != "pass" or a.get("observed") is not True:
            errs.append(f"[eager_jit] {a.get('name')!r}: {a.get('status')!r}")
    seeds = {a.get("seed") for a in ej}
    stress = {s for c, s in draws_in(p) if c == STRESS}
    if not seeds & stress:
        errs.append("[eager_jit] no stress draw was run both eager and jitted")
    big = {s for c, s in draws_in(p) if c == WALK and derived_for(c, s)["K"][1] > INT32_MAX}
    if big and not seeds & big:
        errs.append("[eager_jit] no int32-overflowing walk draw was run both eager and jitted")
    for s in seeds:
        cons = STRESS if s in stress else WALK
        if (cons, s, "E", True) not in idx:
            errs.append(f"[eager_jit] seed {s}: no eager-run engine report recorded")
    return errs


def check_battery(p) -> List[str]:
    errs: List[str] = []
    rows = {r["check_id"]: r for r in p.get("invariants", [])}
    n = p["draws"]["valid"]
    for cid in BATTERY:
        r = rows.get(cid)
        if r is None:
            errs.append(f"[battery] {cid} missing")
            continue
        if r["n_pass"] + r["n_fail"] + r["n_not_applicable"] != n:
            errs.append(f"[battery] {cid}: counts do not cover all {n} draws")
        if r["status"] != "pass" or r["n_fail"] or r["n_pass"] != n:
            errs.append(f"[battery] {cid}: {r['status']} pass={r['n_pass']} fail={r['n_fail']}")
    return errs


def check_quotes(p) -> List[str]:
    errs: List[str] = []
    quotes = p["spec"]["quotes"]
    secs = {(q["source"], q["section"]) for q in quotes}
    for need in (("market-spec.md", "§6.1"), ("market-spec.md", "§8.1"),
                 ("market-spec.md", "§6.2 item 6")):
        if need not in secs:
            errs.append(f"[quotes] required quote {need} absent")
    for q in quotes:
        if q.get("ok") is not True:
            errs.append(f"[quotes] {q['source']} {q['section']}: {q.get('reason')!r}")
    return errs


def check_capacity_and_coverage(p) -> List[str]:
    errs: List[str] = []
    env = p["capacity"]["envelope"]
    want_env = {"resting_per_side": DEFAULT_ENVELOPE.max_resting_per_side, "trades": DEFAULT_ENVELOPE.max_trades,
                "hops": DEFAULT_ENVELOPE.max_hops, "concurrent_icebergs": DEFAULT_ENVELOPE.max_concurrent_icebergs}
    if env != want_env or env["resting_per_side"] > 80 or env["trades"] > 80:
        errs.append(f"[capacity] envelope {env!r} != I8 {want_env!r}")
    ds = [derived_for(c, s) for c, s in draws_in(p)]
    peaks = p["capacity"]["peaks"]
    if peaks["resting_per_side"] != max(d["L"] for d in ds):
        errs.append(f"[capacity] resting peak {peaks['resting_per_side']} != max L {max(d['L'] for d in ds)}")
    if peaks["trades"] != max(d["K"][2] for d in ds):
        errs.append(f"[capacity] trade peak {peaks['trades']} != max num_fills {max(d['K'][2] for d in ds)}")
    walk = [derived_for(c, s) for c, s in draws_in(p) if c == WALK]
    for d in ds:
        if d["agg_residual"] != 0 or d["K"][2] < 1:
            errs.append("[dataset] a draw leaves an aggressor residual or prints no trade")
    for d in walk:
        if not (3 <= d["L"] <= 12 and 1 <= d["m"] < d["L"]):
            errs.append(f"[dataset] walk draw with L={d['L']} m={d['m']} outside L in [3,12], 1<=m<L")
    cov = next((a for a in p["assertions"] if a["name"].startswith(f"dataset: {WALK} covers")), None)
    if cov is None:
        errs.append("[dataset] walk coverage assertion missing")
    else:
        want = {
            "resting BUY": sum(d["rest_side"] == BUY for d in walk),
            "resting SELL": sum(d["rest_side"] == SELL for d in walk),
            "L = 3": sum(d["L"] == 3 for d in walk),
            "L = 12": sum(d["L"] == 12 for d in walk),
            "m = 1": sum(d["m"] == 1 for d in walk),
            "m = L - 1": sum(d["m"] == d["L"] - 1 for d in walk),
            "W > int32 max": sum(d["K"][1] > INT32_MAX for d in walk),
        }
        if cov["observed"] != want:
            errs.append(f"[dataset] coverage {cov['observed']!r} != derived {want!r}")
        if any(v < 1 for v in want.values()):
            errs.append(f"[dataset] a coverage cell is empty: {want!r}")
    return errs


def check_status(p) -> List[str]:
    errs = validate_results(p)
    if p["failure_kinds"] != derive_failure_kinds(p):
        errs.append("[status] failure_kinds not derived from content")
    return [f"[schema] {e}" for e in errs]


ALL_CHECKS = (check_status, check_presence, check_accepted, check_trades, check_fill_vector,
              check_triple, check_book, check_engine_report, check_stress, check_eager_jit,
              check_battery, check_quotes, check_capacity_and_coverage)


def audit_errors(p) -> List[str]:
    out: List[str] = []
    for chk in ALL_CHECKS:
        out += chk(p)
    return out


# ---------------------------------------------------------------------------
# Tests against the emitted results
# ---------------------------------------------------------------------------
class TestEmittedResults:
    def test_results_validate_against_schema_and_status_is_derived(self, payload):
        assert payload["cert_id"] == "C06"
        assert payload["oracle_kind"] == "exact-parametric"
        assert check_status(payload) == []

    def test_generator_digests_are_the_current_generator(self, payload):
        """Ties the JSON to the tapes regenerated here (else the derivation audits other draws)."""
        digests = payload["generator"]["digests"]
        assert set(digests) == {WALK, STRESS}
        for key, dig in digests.items():
            assert construction_digest(key) == dig, key

    def test_every_draw_carries_full_per_draw_evidence(self, payload):
        assert check_presence(payload) == []

    def test_accepted_and_rejected_ids(self, payload):
        assert check_accepted(payload) == []

    def test_counterparty_sequence_walks_levels_best_first_at_resting_limits(self, payload):
        assert check_trades(payload) == []

    def test_per_level_fill_vector_exact(self, payload):
        assert check_fill_vector(payload) == []

    def test_total_qty_weighted_numerator_num_fills_exact_integers(self, payload):
        assert check_triple(payload) == []

    def test_book_after_walk_keeps_level_m_remainder_and_deeper_levels_whole(self, payload):
        assert check_book(payload) == []

    def test_engine_report_is_the_exact_vwap_engineering_invariant(self, payload):
        assert check_engine_report(payload) == []

    def test_stress_draws_sit_at_the_2e10_product_boundary(self, payload):
        assert check_stress(payload) == []

    def test_eager_vs_jitted_evidence_present_and_identical(self, payload):
        assert check_eager_jit(payload) == []

    def test_full_battery_ran_and_holds_on_every_draw(self, payload):
        assert check_battery(payload) == []

    def test_quotes_present_and_unchanged(self, payload):
        assert check_quotes(payload) == []

    def test_capacity_peaks_and_dataset_coverage_match_derivation(self, payload):
        assert check_capacity_and_coverage(payload) == []

    def test_status_pass_iff_audit_clean(self, payload):
        errs = audit_errors(payload)
        assert (payload["status"] == "PASS") == (errs == []), errs[:10]


# ---------------------------------------------------------------------------
# The derivation itself on hand-checked numbers (guards against a wrong oracle)
# ---------------------------------------------------------------------------
class TestDerivationSanity:
    def test_hand_ladder_sell_side(self):
        # three SELL levels 100/101/102 ticks x 5/7/9, BUY aggressor qty 15 at limit 102
        rows = [[0, SELL, 101, 7, 2, 1, 2, 0, 0, 0, 0, 0, 0],
                [0, SELL, 100, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, SELL, 102, 9, 3, 1, 3, 0, 0, 0, 0, 0, 0],
                [0, BUY, 102, 15, 4, 2, 4, 0, 0, 0, 0, 0, 0]]
        d = derive(np.asarray(rows, dtype=np.int32))
        assert d["V"] == [5, 7, 3]
        assert d["K"] == [15, 100 * 5 + 101 * 7 + 102 * 3, 3]
        assert d["B"] == {"BUY": [], "SELL": [[102, 6, 3]]}

    def test_hand_ladder_buy_side_stops_at_limit(self):
        rows = [[0, BUY, 200, 4, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, BUY, 199, 4, 2, 1, 2, 0, 0, 0, 0, 0, 0],
                [0, BUY, 198, 4, 3, 1, 3, 0, 0, 0, 0, 0, 0],
                [0, SELL, 199, 6, 4, 2, 4, 0, 0, 0, 0, 0, 0]]
        d = derive(np.asarray(rows, dtype=np.int32))
        assert d["V"] == [4, 2, 0]
        assert d["K"] == [6, 200 * 4 + 199 * 2, 2]

    def test_derivation_survives_int32_cells_at_the_ceiling(self):
        rows = [[0, SELL, PRICE_INT_MAX, QTY_MAX_INT, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, BUY, PRICE_INT_MAX, QTY_MAX_INT, 2, 2, 2, 0, 0, 0, 0, 0, 0]]
        tape = np.asarray(rows, dtype=np.int32)
        d = derive(tape)
        assert d["K"][1] == PRODUCT_CEILING == 19_978_002_000
        # the trap this audit avoids: a numpy int32 product wraps silently
        with np.errstate(over="ignore"):
            wrapped = int(tape[0, _PRICE] * tape[0, _QTY])
        assert wrapped != PRODUCT_CEILING


# ---------------------------------------------------------------------------
# Deliberately broken copies must be rejected
# ---------------------------------------------------------------------------
def _write_and_reload(p, tmp_path: Path) -> Dict[str, Any]:
    (tmp_path / "C06.json").write_text(json.dumps(p), encoding="utf-8")
    return _load(tmp_path)


def _pick(p, pred) -> Tuple[str, int]:
    for c, s in draws_in(p):
        if pred(c, s, derived_for(c, s)):
            return c, s
    pytest.skip("no draw of the required shape in the emitted results")


def _set_both(p, cons, seed, letter, fn):
    for a in p["assertions"]:
        m = _NAME.match(a["name"])
        if m and m["cons"] == cons and int(m["seed"]) == seed and m["letter"] == letter:
            a["observed"] = fn(copy.deepcopy(a["observed"]))
            if letter != "E":
                a["expected"] = fn(copy.deepcopy(a["expected"]))


class TestBrokenCopiesRejected:
    def test_skipped_last_level_is_rejected(self, payload, tmp_path):
        c, s = _pick(payload, lambda c, s, d: d["r"] > 0 and d["K"][2] >= 2)
        d = derived_for(c, s)
        last_q = d["Q"][-1]
        broken = copy.deepcopy(payload)
        _set_both(broken, c, s, "Q", lambda q: q[:-1])
        _set_both(broken, c, s, "V", lambda v: [x if i != d["m"] else 0 for i, x in enumerate(v)])
        _set_both(broken, c, s, "K", lambda k: [k[0] - last_q[3], k[1] - last_q[2] * last_q[3], k[2] - 1])
        broken = _write_and_reload(broken, tmp_path)
        errs = audit_errors(broken)
        for tag in ("[Q]", "[V]", "[K]"):
            assert any(e.startswith(tag) and f"seed {s}:" in e for e in errs), (tag, errs[:5])

    def test_double_counted_level_is_rejected(self, payload, tmp_path):
        c, s = _pick(payload, lambda c, s, d: d["K"][2] >= 2)
        d = derived_for(c, s)
        first = d["Q"][0]
        broken = copy.deepcopy(payload)
        _set_both(broken, c, s, "Q", lambda q: [q[0]] + q)
        _set_both(broken, c, s, "V", lambda v: [2 * v[0]] + v[1:])
        _set_both(broken, c, s, "K", lambda k: [k[0] + first[3], k[1] + first[2] * first[3], k[2] + 1])
        broken = _write_and_reload(broken, tmp_path)
        errs = audit_errors(broken)
        for tag in ("[Q]", "[V]", "[K]"):
            assert any(e.startswith(tag) and f"seed {s}:" in e for e in errs), (tag, errs[:5])

    def test_int32_wrapped_numerator_is_rejected(self, payload, tmp_path):
        c, s = _pick(payload, lambda c, s, d: c == STRESS)
        wrap = lambda x: (x + 2**31) % 2**32 - 2**31  # noqa: E731
        broken = copy.deepcopy(payload)
        _set_both(broken, c, s, "K", lambda k: [k[0], wrap(k[1]), k[2]])
        broken = _write_and_reload(broken, tmp_path)
        errs = audit_errors(broken)
        assert any(e.startswith("[K]") and f"seed {s}:" in e for e in errs)
        assert any(e.startswith("[stress]") and f"seed {s}:" in e for e in errs)

    def test_int32_wrapped_walk_numerator_is_rejected(self, payload, tmp_path):
        c, s = _pick(payload, lambda c, s, d: c == WALK and d["K"][1] > INT32_MAX)
        wrap = lambda x: (x + 2**31) % 2**32 - 2**31  # noqa: E731
        broken = copy.deepcopy(payload)
        _set_both(broken, c, s, "K", lambda k: [k[0], wrap(k[1]), k[2]])
        errs = audit_errors(_write_and_reload(broken, tmp_path))
        assert any(e.startswith("[K]") and f"seed {s}:" in e for e in errs)

    def test_float_rounded_numerator_is_rejected(self, payload, tmp_path):
        c, s = _pick(payload, lambda c, s, d: c == STRESS)
        broken = copy.deepcopy(payload)
        _set_both(broken, c, s, "K", lambda k: [k[0], float(np.float32(k[1])), k[2]])
        errs = audit_errors(_write_and_reload(broken, tmp_path))
        assert any(e.startswith("[K]") and f"seed {s}:" in e for e in errs)

    def test_missing_eager_jit_evidence_is_rejected(self, payload, tmp_path):
        broken = copy.deepcopy(payload)
        broken["assertions"] = [a for a in broken["assertions"] if a["kind"] != "eager_jit"]
        errs = check_eager_jit(_write_and_reload(broken, tmp_path))
        assert any("absence = failure" in e for e in errs)

    def test_eager_jit_without_stress_is_rejected(self, payload, tmp_path):
        broken = copy.deepcopy(payload)
        broken["assertions"] = [a for a in broken["assertions"]
                                if not (a["kind"] == "eager_jit" and a["seed"] >= 1_000_000)]
        errs = check_eager_jit(_write_and_reload(broken, tmp_path))
        assert any("no stress draw was run both eager and jitted" in e for e in errs)

    def test_missing_stress_draw_is_rejected(self, payload, tmp_path):
        broken = copy.deepcopy(payload)
        broken["assertions"] = [a for a in broken["assertions"] if not a["name"].startswith(STRESS)]
        errs = check_presence(_write_and_reload(broken, tmp_path))
        assert any("no stress draw" in e for e in errs)

    def test_missing_battery_row_is_rejected(self, payload, tmp_path):
        broken = copy.deepcopy(payload)
        broken["invariants"] = [r for r in broken["invariants"] if r["check_id"] != "I5"]
        errs = check_battery(_write_and_reload(broken, tmp_path))
        assert any("I5 missing" in e for e in errs)

    def test_vwap_off_by_two_ticks_report_is_rejected(self, payload, tmp_path):
        c, s = _pick(payload, lambda c, s, d: d["bracket"][1] - d["bracket"][0] > 6)
        broken = copy.deepcopy(payload)

        def nudge(o):
            o["price_int"] = o["price_int"] + 2
            return o
        _set_both(broken, c, s, "E", nudge)
        errs = check_engine_report(_write_and_reload(broken, tmp_path))
        assert any(f"seed {s}" in e and "VWAP" in e for e in errs)


# ---------------------------------------------------------------------------
# Optional independent re-run on the engine (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("cons,seed", [(WALK, 0), (WALK, 7), (WALK, 93), (STRESS, 1_000_000)])
def test_rerun_engine_matches_derivation(cons, seed):
    """slow: jitted engine replay of one draw via the battery's recorder (compile-dominated)."""
    from tests.omiecid.reference.omie_invariants import compare_records, record_from_draw

    draw = _make(cons, seed)
    d = derive(draw.tape)
    rec = record_from_draw(draw)
    trades = [[int(f.passive_oid), int(f.aggressor_oid), int(f.price_int), int(f.qty_int),
               int(f.submission)] for f in rec.fills]
    assert trades == d["Q"]
    total = sum(t[3] for t in trades)
    numer = sum(t[2] * t[3] for t in trades)
    assert [total, numer, len(trades)] == d["K"]
    if cons == STRESS:
        eager = record_from_draw(draw, jit=False)
        assert compare_records(rec, eager) == ()
