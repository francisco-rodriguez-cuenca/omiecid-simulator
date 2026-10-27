"""K4 audit twin for OMIE-CERT-C15-1 (randomised fuzz across the parameter space) -- r4.

Re-authored at K4 of the ``OMIE-CERT-RERUN-3`` re-run (r4), after the S2 engine fix
``OMIE-ICE-INCOMING-LADDER-1`` (E-A per-instance sweep limits, E-B remainder
continuation and display, E-C row split at incoming instance boundaries) and the K2
r4 update of the notebook's reference walk W. Fresh-context, adversarial audit of the
**emitted** ``C15.json``, whatever its status. Every expectation was worked out again
from the cited documents only:

* the card (``agile-plan.md`` §OMIE-CERT-C15-1), the RERUN-3 r4 table row for C15
  (W's incoming branch follows the amended Rule 2 for every d; F18 retired; F5
  restated; OQ-33 "Ruling..." and OQ-34 "The simulator's convention..." quotes
  retired), ``scenario-ladder.md`` §C15 (C15a smoke: NON and IOC only, 200 draws;
  C15b gate: the full grid, 2 000 draws; lengths 32/96/160; book depth and
  iceberg-slot pressure as independent axes; systematic sweep; discard rate
  reported; every draw eager **and** jitted; battery I1-I9 and M1-M6; one violation
  fails the certification; the shrinker; the capacity-validity precondition);
* ``market-spec.md`` **v2.14**: §4.2 (a new slice gets a new timestamp); §4.2.1
  Rule 1 (FOK sees every hidden slice whose own drifted price crosses, OQ-11);
  Rule 2 and its *Re-ranking* bullet; Rule 2 *Incoming iceberg* items (1)-(6) as
  **amended 2026-10-20** -- an incoming ICE / ICEwPI (price b, peak P, increment d)
  is a chain of instances of size P; instance k trades only at prices its limit
  ``L_k = b + k d`` crosses; the sweep stops at the first instance that cannot trade;
  every trade prices at the resting entry; a row ends at a resting entry's end or
  an incoming instance's end; the remainder ``R`` rests as instance
  ``k_e = floor(M / P)`` at ``b + k_e d`` showing ``v_1 = min(P - u, R)``,
  ``u = M - k_e P``, hidden ``R - v_1``, with no new timestamp; later slices continue
  the ladder at ``b + (k_e + 1) d``, ...; Rule 3 *ICE aggressor* (the iceberg already
  resting sets every price); Rule 4 *Slice activation timing* (eager); §4.5; §5.2;
  §5.4 (v2.14 iceberg rejections); §6.1; §6.2 items 2, 3, 7;
* ``open-questions.md`` OQ-32 (same-second processing order: modelling choice), the
  2026-10-20 blocks of OQ-33 (rows split at incoming instance boundaries) and OQ-34
  (instance limits, continuation, display C3), OQ-35 (drifted prices beyond the
  limits: question only; the generator's drifts stay far inside), OQ-20 correction
  note (eager activation), and ``OMIE-SLICE-STAMP-TAPE-SEC-1`` E2 (a slice takes its
  queue place at its activation event; the aggressor's own remainder keeps its entry
  stamp).

Independence rules 3/4: this file never opens or executes the C15 notebook's code
cells, nor reads the engine modules' bodies or the S2 diff. It uses the generator's
public API (``fuzz_grid`` / ``make_random_session`` / ``construction_digest``), the
loader's row converter and ``ContractBook``'s public API (``init`` /
``submit_order``). Its price-time walk (:func:`reference_walk`) is written here from
the rules above; the superseded 2026-10-07 reading is kept only as the planted
defect ``incoming=OLD``.

**History.** r2 failed W on 618 draws (stale W + the engine deviation "D4b" on
100000254, 100001177, 100001190, 100001489: a slice regenerated in an earlier
process stored ahead of the tape clock). r3 was PASS after E1/E2. r4 changes the
incoming-iceberg rule.

**r4 judgement.** A PASS file carries no engine trades, so this twin (a) checks the
twin walk against the emitted per-draw (trade rows, volume) on **all** 2 200 draws
(the row count carries the item (2) split), with the superseded rule as a non-null
twin; and (b) re-judges the engine itself: it replays through ``ContractBook`` every
draw holding an incoming iceberg that crosses on entry, every D4b-pattern draw and
the first draw of every sweep cell, and checks after **every** submission the trade
rows, both queues in priority order with prices and visible quantities (F6; item
(4) display), unique stored stamps, and every stored SEC equal to the second of the
row that set it (arrival, or activation for a slice). Accounting checks derive their
counts from the payload (no literal count of any run is pinned).

Results JSON: ``$OMIE_CERT_RESULTS_DIR/C15.json`` else
``notebooks/certification/results/C15.json``; absent -> skip. Shrunk fixtures are
looked up in the results directory, at their recorded path,
``tests/omiecid/fixtures/cert/`` and ``notebooks/certification/fixtures/``; absent ->
skip.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from notebooks.certification.cert_support import derive_failure_kinds, validate_results

REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_ID = "C15"
SPEC_VERSION = "v2.14"

# Battery ids in the order the per-draw status strings carry them.
I_IDS = ("RB",) + tuple(f"I{i}" for i in range(1, 10))
M_IDS = tuple(f"M{i}" for i in range(1, 7))
ALL_IDS = I_IDS + M_IDS

MIXES = ("non", "non_ioc", "non_ioc_fok", "non_ioc_fok_ice", "all_icewpi")
SMOKE_MIXES = ("non", "non_ioc")
CROSS = (0.05, 0.2, 0.4, 0.6)
TIES = (0.0, 0.4)
LENGTHS = (32, 96, 160)
DEPTHS = (60, 100)
SLOTS = (64, 1024)
# Card: C15a smoke 200 draws, C15b gate 2 000 draws. Seed blocks as emitted.
VARIATIONS = {"C15a": (0, 200), "C15b": (100_000_000, 2000)}
# The capacity-validity precondition: documented ceilings and margins.
CEILINGS = [100, 100, 200, 1024, 64]
ENVELOPE = {"resting_per_side": 80, "trades": 80, "hops": 160, "concurrent_icebergs": 800}
MAX_SLICES_MARGIN = 64

BUY, SELL = 0, 1
NON, IOC, FOK, ICE = 0, 1, 2, 3
# Tape columns (13-wide int32, loader order).
C_SIDE, C_PRICE, C_QTY, C_OID, C_AGENT, C_TS, C_EXEC, C_PEAK, C_PPD = 1, 2, 3, 4, 5, 6, 7, 11, 12

_CELL_RE = re.compile(r"^(C15[ab]) cell (\d+): every draw clean$")
_DRAW_RE = re.compile(r"^(C15[ab]) seed (\d+): every check on the draw")
_SHRINK_RE = re.compile(r"^shrinker: (C15[ab]) seed (\d+) -> (\d+) of (\d+) rows")
_EAGER_RE = re.compile(r"^eager == jitted \(seed (\d+)\)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _results_dir() -> Path:
    d = os.environ.get("OMIE_CERT_RESULTS_DIR")
    return Path(d) if d else REPO_ROOT / "notebooks" / "certification" / "results"


def _load(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    p = _results_dir() / f"{CERT_ID}.json"
    if not p.is_file():
        pytest.skip(f"C15 results JSON absent: {p} (run C15_randomised_fuzz.ipynb section 8 first)")
    return _load(p)


def _roundtrip(tmp_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    p = tmp_path / f"{CERT_ID}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return _load(p)


def _mutable(p: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    """Shallow copy of ``p`` with ``keys`` deep-copied (cheap broken copies)."""
    q = dict(p)
    for k in keys:
        q[k] = copy.deepcopy(p[k])
    return q


# ---------------------------------------------------------------------------
# Indexing the payload
# ---------------------------------------------------------------------------
def parse_draw(s: str) -> Dict[str, Any]:
    """``seed|digest12|I-status(10)|M-status(6)|W|n_trades,volume|marks``."""
    parts = s.split("|")
    if len(parts) != 7:
        raise ValueError(f"draw string has {len(parts)} fields: {s!r}")
    seed, dig, ist, mst, w, nv, marks = parts
    n_tr, vol = (int(x) for x in nv.split(","))
    return {"seed": int(seed), "digest12": dig, "status": dict(zip(ALL_IDS, ist + mst)),
            "i_len": len(ist), "m_len": len(mst), "W": w, "n_trades": n_tr, "volume": vol,
            "marks": marks}


def index(p: Dict[str, Any]) -> Dict[str, Any]:
    cells: Dict[Tuple[str, int], Dict[str, Any]] = {}
    draws: Dict[int, Dict[str, Any]] = {}
    per_draw_fail: Dict[int, Dict[str, Any]] = {}
    shrinks, eager, summary = [], [], {}
    for a in p.get("assertions", []):
        name = a.get("name", "")
        m = _CELL_RE.match(name)
        if m:
            key = (m.group(1), int(m.group(2)))
            obs = a["observed"]
            cells[key] = {"assertion": a, "cell": obs["cell"], "regime": obs["regime"],
                          "draws": [parse_draw(s) for s in obs["draws"]]}
            for dd in cells[key]["draws"]:
                dd["variation"], dd["cell_index"] = key
                dd["cell"] = obs["cell"]
                dd["regime"] = obs["regime"]
                if dd["seed"] in draws:
                    dd["duplicate"] = True
                draws.setdefault(dd["seed"], dd)
            continue
        m = _DRAW_RE.match(name)
        if m:
            per_draw_fail[int(m.group(2))] = a
            continue
        m = _SHRINK_RE.match(name)
        if m:
            shrinks.append((a, m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))))
            continue
        if a.get("kind") == "eager_jit":
            eager.append(a)
            continue
        summary[name] = a
    return {"cells": cells, "draws": draws, "fail": per_draw_fail, "shrinks": shrinks,
            "eager": eager, "summary": summary,
            "inv": {r["check_id"]: r for r in p.get("invariants", [])}}


@pytest.fixture(scope="module")
def ix(payload) -> Dict[str, Any]:
    return index(payload)


def _summary(ix_, prefix: str) -> Optional[Dict[str, Any]]:
    for name, a in ix_["summary"].items():
        if name.startswith(prefix):
            return a
    return None


def _pressures(ix_) -> Dict[str, Dict[str, int]]:
    a = _summary(ix_, "pressure configurations")
    return {} if a is None else a["observed"]


# ---------------------------------------------------------------------------
# (a) schema, status, failure kinds, draw accounting
# ---------------------------------------------------------------------------
def kinds_from_content(p: Dict[str, Any]) -> List[str]:
    """Failure kinds re-derived here from the payload's content (independent of cert_support)."""
    kinds = set()
    for a in p["assertions"]:
        if a["status"] == "fail":
            kinds.add({"oracle": "oracle_mismatch", "eager_jit": "eager_jit_disagreement"}[a["kind"]])
    for r in p["invariants"]:
        if r["n_fail"] > 0 or r["status"] == "fail":
            kinds.add("vacuous_run" if r["check_id"] == "I9" else "invariant_violation")
    if any(not q["ok"] for q in p["spec"]["quotes"]):
        kinds.add("quote_drift")
    if p["draws"]["valid"] < p["draws"]["requested"]:
        kinds.add("capacity_flag")
    return sorted(kinds)


def status_errors(p: Dict[str, Any]) -> List[str]:
    errs = list(validate_results(p))
    if errs:
        return errs
    ix_ = index(p)
    if p["cert_id"] != CERT_ID:
        errs.append(f"cert_id {p['cert_id']!r}")
    if p["failure_kinds"] != derive_failure_kinds(p) or p["failure_kinds"] != kinds_from_content(p):
        errs.append(f"failure_kinds {p['failure_kinds']} vs derived {kinds_from_content(p)}")
    if p["oracle_kind"] != "invariant / distributional":
        errs.append(f"oracle_kind {p['oracle_kind']!r}")
    any_fail = bool(p["seeds"]["failing"]) or any(
        dd["W"] == "f" or "f" in dd["status"].values() for dd in ix_["draws"].values())
    n_failed = sum(a["status"] == "fail" for a in p["assertions"])
    # Pass criterion: zero exceptions across every draw.
    if any_fail and p["status"] != "NOT CERTIFIED":
        errs.append("a failing draw exists but status is not NOT CERTIFIED")
    if p["status"] == "PASS":
        if any_fail or n_failed or p["failure_kinds"]:
            errs.append(f"PASS with {n_failed} failed assertion(s) / kinds {p['failure_kinds']}")
        if p["failure_explanation"] is not None:
            errs.append("PASS carries a failure_explanation")
    elif not any_fail and n_failed == 0 and not p["failure_kinds"]:
        errs.append(f"status {p['status']!r} but no draw, assertion or failure kind failed")
    if p["status"] == "NOT CERTIFIED":
        fe = p["failure_explanation"]
        if not isinstance(fe, dict):
            errs.append("NOT CERTIFIED without a failure_explanation")
            return errs
        # The count is derived from the payload's own failed assertions, never a literal.
        m = re.match(r"^(\d+) assertion\(s\) failed", fe.get("what_failed") or "")
        if not m or int(m.group(1)) != n_failed:
            errs.append(f"what_failed count != {n_failed} failed assertions")
        sc = fe.get("smallest_case")
        if sc is None or sc["seed"] not in set(p["seeds"]["failing"]):
            errs.append("smallest_case missing or its seed is not a failing seed")
        elif not sc["rows"] or sc["rows"] < 1:
            errs.append("smallest_case rows must be >= 1")
    return errs


def accounting_errors(p: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    ix_ = index(p)
    d = p["draws"]
    total = sum(n for _, n in VARIATIONS.values())
    if d["requested"] != total:
        errs.append(f"requested {d['requested']} != {total} (C15a 200 + C15b 2000)")
    if d["valid"] != d["requested"]:
        errs.append(f"valid {d['valid']} != requested {d['requested']}")
    denom = d["valid"] + d["discarded"]
    if not math.isclose(d["discard_rate"], d["discarded"] / denom if denom else 0.0, abs_tol=1e-12):
        errs.append("discard_rate != discarded / (valid + discarded)")
    if any(dd.get("duplicate") for dd in ix_["draws"].values()):
        errs.append("a seed appears in two cells")
    seeds = sorted(ix_["draws"])
    if p["seeds"]["count"] != len(seeds) or len(seeds) != d["valid"]:
        errs.append(f"seeds.count {p['seeds']['count']} / distinct seeds {len(seeds)} / valid {d['valid']}")
    if seeds and (p["seeds"]["first"] != seeds[0] or p["seeds"]["last"] != seeds[-1]):
        errs.append("seeds.first/last do not bracket the drawn seeds")
    # Per-variation seed blocks, systematic sweep, per-cell detail.
    discarded = 0
    for var, (base, n) in VARIATIONS.items():
        vs = sorted(s for s, dd in ix_["draws"].items() if dd["variation"] == var)
        if vs != list(range(base, base + n)):
            errs.append(f"{var}: seeds are not the contiguous block {base}..{base + n - 1}")
        vc = [k for k in ix_["cells"] if k[0] == var]
        n_cells = len(vc)
        for key in vc:
            c = ix_["cells"][key]
            m = re.match(r"^(\d+) valid, (\d+) discarded$", c["assertion"]["detail"])
            if not m or int(m.group(1)) != len(c["draws"]):
                errs.append(f"{var} cell {key[1]}: detail does not match its draw count")
            else:
                discarded += int(m.group(2))
            for dd in c["draws"]:
                if (dd["seed"] - base) % n_cells != key[1]:
                    errs.append(f"{var} seed {dd['seed']} is not in cell (seed-base) mod {n_cells}")
            failing_cell = any(dd["W"] == "f" or "f" in dd["status"].values() for dd in c["draws"])
            if (c["assertion"]["status"] == "fail") != failing_cell:
                errs.append(f"{var} cell {key[1]}: status {c['assertion']['status']} vs its draws")
        a = _summary(ix_, f"{var}: valid draws == requested")
        if a is None or a["observed"] != n or a["expected"] != n or a["status"] != "pass":
            errs.append(f"{var}: missing or wrong 'valid draws == requested' assertion")
    if discarded != d["discarded"]:
        errs.append(f"per-cell discards {discarded} != draws.discarded {d['discarded']}")
    # Failing seeds: the list, the per-draw failure records, the status strings.
    failing = p["seeds"]["failing"]
    if failing != sorted(set(failing)):
        errs.append("seeds.failing is not sorted and unique")
    by_string = sorted(s for s, dd in ix_["draws"].items()
                       if dd["W"] == "f" or "f" in dd["status"].values())
    if sorted(failing) != by_string:
        errs.append("seeds.failing != draws whose status string holds an 'f'")
    if sorted(ix_["fail"]) != sorted(failing):
        errs.append("seeds.failing != seeds of the per-draw failure records")
    for s, a in ix_["fail"].items():
        dd = ix_["draws"].get(s)
        if dd is None or a["status"] != "fail" or a["seed"] != s:
            errs.append(f"per-draw failure record {s} is inconsistent")
            continue
        fc = set(a["observed"]["failing_checks"])
        want = {k for k, v in dd["status"].items() if v == "f"}
        if fc != want:
            errs.append(f"seed {s}: failing_checks {sorted(fc)} != status string {sorted(want)}")
        if (dd["W"] == "f") != bool(a["observed"]["reference_walk"]):
            errs.append(f"seed {s}: reference-walk evidence present iff W failed")
    # Invariant rows against the per-draw status strings.
    if sorted(ix_["inv"]) != sorted(ALL_IDS):
        errs.append(f"invariant rows {sorted(ix_['inv'])} != {sorted(ALL_IDS)}")
    for cid, r in ix_["inv"].items():
        col = Counter(dd["status"].get(cid) for dd in ix_["draws"].values())
        if r["n_pass"] + r["n_fail"] + r["n_not_applicable"] != d["valid"]:
            errs.append(f"{cid}: counts do not sum to the valid draws")
        if (col["p"], col["f"], col["n"]) != (r["n_pass"], r["n_fail"], r["n_not_applicable"]):
            errs.append(f"{cid}: counts {r['n_pass']},{r['n_fail']},{r['n_not_applicable']} != strings "
                        f"{col['p']},{col['f']},{col['n']}")
        fs = sorted(s for s, dd in ix_["draws"].items() if dd["status"].get(cid) == "f")
        if sorted(r["failing_seeds"]) != fs:
            errs.append(f"{cid}: failing_seeds != draws marked 'f'")
        if not set(r["failing_seeds"]) <= set(failing):
            errs.append(f"{cid}: failing_seeds not within seeds.failing")
    return errs


# ---------------------------------------------------------------------------
# (b) coverage
# ---------------------------------------------------------------------------
def _regime_name(mix: str, cross: float, tie: float, n: int) -> str:
    return f"{mix}|cross={cross}|tie={tie}|n={n}"


def coverage_errors(p: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    ix_ = index(p)
    press = _pressures(ix_)
    want_press = {f"D={dd},M={m}" for dd in DEPTHS for m in SLOTS}
    if set(press) != want_press:
        errs.append(f"pressure configurations {sorted(press)} != {sorted(want_press)}")
        return errs
    for k, v in press.items():
        if not (0 < v["resting_margin"] < v["book_depth"] and 0 < v["slot_margin"] < v["max_icebergs"]):
            errs.append(f"{k}: a margin is not strictly below its ceiling")
        if v["resting_margin"] > ENVELOPE["resting_per_side"] or v["slot_margin"] > ENVELOPE["concurrent_icebergs"]:
            errs.append(f"{k}: a margin exceeds the documented envelope")
    # Independent axes: both depths with both slot counts.
    for var, mixes in (("C15a", SMOKE_MIXES), ("C15b", MIXES)):
        got = Counter()
        for (v, _), c in ix_["cells"].items():
            if v != var:
                continue
            cc = c["cell"]
            if c["regime"] != _regime_name(cc["mix"], cc["cross"], cc["tie"], cc["n"]):
                errs.append(f"{var}: cell regime label {c['regime']!r} disagrees with its cell")
            got[(cc["mix"], cc["cross"], cc["tie"], cc["n"], cc["depth"], cc["slots"])] += 1
            if not c["draws"]:
                errs.append(f"{var}: a cell has no valid draw")
        lengths = (32,) if var == "C15a" else LENGTHS
        want = set()
        for mix in mixes:
            for cp in CROSS:
                for td in TIES:
                    for n in lengths:
                        for dep in DEPTHS:
                            for sl in SLOTS:
                                margin = press[f"D={dep},M={sl}"]["resting_margin"]
                                if n <= 2 * margin:  # a tape has no cancellations
                                    want.add((mix, cp, td, n, dep, sl))
        if set(got) != want:
            missing = sorted(want - set(got))[:4]
            extra = sorted(set(got) - want)[:4]
            errs.append(f"{var}: cells != the systematic sweep (missing {missing}, extra {extra})")
        if any(v > 1 for v in got.values()):
            errs.append(f"{var}: a sweep cell is repeated")
        if var == "C15b" and not {k[3] for k in got} >= set(LENGTHS):
            errs.append("C15b: lengths 32/96/160 are not all swept")
        if var == "C15b" and not {k[4] for k in got if k[3] == 160} >= {100}:
            errs.append("C15b: n=160 not swept at full depth")
        a = _summary(ix_, f"{var}: every relation applied at least")
        if a is None:
            errs.append(f"{var}: relation-coverage assertion missing")
        else:
            obs = a["observed"]
            need = [m for m in M_IDS if not (var == "C15a" and m == "M4")]  # no FOK in the smoke mixes
            for m in need:
                if obs.get(m, 0) < 5:
                    errs.append(f"{var}: {m} applied {obs.get(m, 0)} < 5 times")
        a = _summary(ix_, f"{var}: M6 exercised a self-match")
        if a is None or not isinstance(a["observed"], int) or a["observed"] < 1:
            errs.append(f"{var}: no M6 draw holding a self-match")
    # Relation counts across variations == applicable rows of the M checks.
    tot = Counter()
    for var in VARIATIONS:
        a = _summary(ix_, f"{var}: every relation applied at least")
        if a is not None:
            tot.update(a["observed"])
    for m in M_IDS:
        r = ix_["inv"].get(m)
        if r is not None and r["n_pass"] + r["n_fail"] != tot[m]:
            errs.append(f"{m}: applied {tot[m]} != n_pass+n_fail {r['n_pass'] + r['n_fail']}")
    # Capacity: the five ceilings, the envelope, the observed peaks.
    a = _summary(ix_, "the five configured ceilings")
    if a is None or a["observed"] != CEILINGS or a["expected"] != CEILINGS or a["status"] != "pass":
        errs.append("the five ceilings assertion is missing or wrong")
    if p["capacity"]["envelope"] != ENVELOPE:
        errs.append(f"capacity.envelope {p['capacity']['envelope']} != {ENVELOPE}")
    peaks = p["capacity"]["peaks"]
    for k, lim in ENVELOPE.items():
        if peaks[k] is not None and peaks[k] > lim:
            errs.append(f"capacity peak {k}={peaks[k]} > {lim}")
    if peaks["hops"] is None and "matching_iterations" not in ix_["inv"].get("I8", {}).get("unobserved", []):
        errs.append("hop peak unreported but I8 does not report matching_iterations unobserved")
    band_peaks = defaultdict(int)
    largest = 0
    for var in VARIATIONS:
        b = _summary(ix_, f"REPORTED, not asserted: {var} aggregate")
        if b is None:
            errs.append(f"{var}: aggregate band report missing")
            continue
        largest = max(largest, int(b["observed"]["largest_slice_count"]))
        for pk in b["observed"]["peaks_per_pressure"].values():
            for k, v in pk.items():
                band_peaks[k] = max(band_peaks[k], v)
    for k in ("resting_per_side", "trades", "concurrent_icebergs"):
        if peaks[k] != band_peaks[k]:
            errs.append(f"capacity peak {k}={peaks[k]} != max per pressure {band_peaks[k]}")
    if largest > MAX_SLICES_MARGIN:
        errs.append(f"largest slice count {largest} > {MAX_SLICES_MARGIN}")
    if ix_["inv"].get("I8", {}).get("status") != "pass":
        errs.append("I8 capacity headroom did not pass on every draw")
    return errs


# ---------------------------------------------------------------------------
# (c) eager vs jitted
# ---------------------------------------------------------------------------
def eager_errors(p: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    ix_ = index(p)
    if p["eager_jit_identical"] is not True:
        errs.append(f"eager_jit_identical is {p['eager_jit_identical']!r}; evidence absent or disagreeing")
    seeded = []
    for a in ix_["eager"]:
        if a["status"] != "pass":
            errs.append(f"eager_jit assertion failed: {a['name']}")
        m = _EAGER_RE.match(a["name"])
        if m:
            s = int(m.group(1))
            if a["seed"] != s or a["observed"] is not True or a["expected"] is not True:
                errs.append(f"{a['name']}: not a passing seeded comparison")
            seeded.append(s)
            dd = ix_["draws"].get(s)
            mm = re.match(r"^mix (\w+): identical$", a["detail"])
            if dd is None:
                errs.append(f"eager seed {s} is not a valid draw")
            elif not mm or mm.group(1) != dd["cell"]["mix"]:
                errs.append(f"eager seed {s}: detail mix does not match its cell")
    mixes = Counter(ix_["draws"][s]["cell"]["mix"] for s in seeded if s in ix_["draws"])
    for mix in MIXES:
        if mixes[mix] < 1:
            errs.append(f"no eager==jitted comparison for mix {mix}")
    # Per-mix presence records (when emitted) must count the seeded comparisons.
    for a in ix_["eager"]:
        mm = re.match(r"^eager comparison present for mix (\w+)$", a["name"])
        if mm and a["observed"] != mixes[mm.group(1)]:
            errs.append(f"{a['name']}: observed {a['observed']} != {mixes[mm.group(1)]} seeded comparisons")
    for var in VARIATIONS:
        vs = sorted(s for s in seeded if s in ix_["draws"] and ix_["draws"][s]["variation"] == var)
        if not vs:
            errs.append(f"{var}: no eager==jitted comparison")
        b = _summary(ix_, f"REPORTED, not asserted: {var} aggregate")
        if b is not None and sorted(int(e[1]) for e in b["observed"]["eager_draws"]) != vs:
            errs.append(f"{var}: band eager_draws != eager_jit assertion seeds")
    marked = sorted(s for s, dd in ix_["draws"].items() if "E" in dd["marks"])
    if marked != sorted(seeded):
        errs.append("draws marked eager in their status string != eager_jit assertion seeds")
    return errs




def failed_assertion_errors(p: Dict[str, Any]) -> List[str]:
    """Every failed assertion is a per-draw record or the cell holding it; nothing else fails."""
    errs: List[str] = []
    ix_ = index(p)
    failed = [a for a in p["assertions"] if a["status"] == "fail"]
    n_cells = sum(c["assertion"]["status"] == "fail" for c in ix_["cells"].values())
    other = [a["name"] for a in failed
             if not (_CELL_RE.match(a["name"]) or _DRAW_RE.match(a["name"]))]
    if other:
        errs.append(f"unexpected failed assertions: {other[:3]}")
    if len(failed) != len(p["seeds"]["failing"]) + n_cells:
        errs.append(f"{len(failed)} failed != {len(p['seeds']['failing'])} draws + {n_cells} cells")
    return errs


# ---------------------------------------------------------------------------
# The independent price-time reference walk (written from v2.14, Rule 2 amended 2026-10-20)
# ---------------------------------------------------------------------------
class _Entry:
    __slots__ = ("oid", "side", "base", "price", "vis", "hidden", "peak", "ppd", "k", "seq", "act",
                 "regen", "incoming", "ke")

    def __init__(self, **kw):
        self.regen, self.incoming, self.ke = False, False, 0
        for k, v in kw.items():
            setattr(self, k, v)


def rejected_v214(r) -> bool:
    """§5.4 (v2.14) iceberg rejections the generator can produce: peak <= 0 or
    peak > total; ICEwPI sign rule (BUY needs PPD <= 0, SELL needs PPD >= 0)."""
    if int(r[C_EXEC]) != ICE:
        return False
    q, pk, ppd, side = int(r[C_QTY]), int(r[C_PEAK]), int(r[C_PPD]), int(r[C_SIDE])
    return pk <= 0 or pk > q or (side == BUY and ppd > 0) or (side == SELL and ppd < 0)


# Incoming-iceberg rule variants. ``R4`` is the binding rule; ``OLD`` is the
# superseded 2026-10-07 reading (base-limit sweep, one row per resting entry,
# remainder at slice 0 showing min(P, R)) -- used ONLY by planted-defect twins.
R4, OLD = "r4", "old"


def reference_walk(rows: Sequence[Sequence[int]], *, trace: Optional[list] = None,
                   book_out: Optional[dict] = None,
                   books_out: Optional[list] = None,
                   ice_out: Optional[list] = None,
                   acts_out: Optional[list] = None,
                   incoming: str = R4) -> List[Optional[List[List[int]]]]:
    """Expected trade rows per submission, ``[buyer, seller, price, qty, submitter_oid]``
    (``None`` for a §5.4-rejected row).

    * One global sequential queue in **processing (row) order** (§5.2; OQ-32
      modelling choice for equal tape seconds). Partial execution keeps the stamp.
    * The aggressor always trades the best-ranked opposite entry next: best price,
      then earliest stamp (§6.1); a trade needs ``sell <= buy`` (§6.2 item 2) and
      prints at the resting entry's price (§6.2 item 3, item 7, Rule 3 / OQ-31).
    * Rule 2 + Rule 4 (v2.14): when a resting iceberg's slice is exhausted, its next
      slice ``base + k*ppd`` is activated **at once** with a fresh stamp (eager,
      exhaustion order) and re-ranked as a new entry; matching continues.
    * Incoming iceberg (Rule 2 items (1)-(6), amended 2026-10-20; Rule 3 *ICE
      aggressor*): price ``b``, peak ``P``, increment ``d``. The total quantity is a
      chain of instances of size ``P``; with ``F`` units already swept, the current
      instance is ``k = F // P`` with limit ``L_k = b + k d``. It trades the best
      opposite entry only if ``L_k`` crosses it, else the sweep **stops** (even if
      ``b`` would still cross). A row ends at the resting entry's end or at the
      instance's end ``P (k + 1)``. With ``M`` swept, ``k_e = M // P``,
      ``u = M - k_e P``, a remainder ``R > 0`` rests as instance ``k_e`` at
      ``b + k_e d`` showing ``v_1 = min(P - u, R)``, hidden ``R - v_1``, with its
      **arrival** stamp; its later slices continue at ``b + (k_e + 1) d``, ...
    * FOK (Rule 1, OQ-11): accepted iff the volume whose own (drifted) price crosses
      covers it; otherwise no trade, no book change. IOC remainder is cancelled;
      NON remainder rests (§4.5).

    ``trace`` (optional) receives one dict per row: the counterparty, whether its
    visible part is a regenerated slice and the row index at which that entry's
    current stamp was set. ``book_out`` receives the final book as
    ``{oid: (price, visible, hidden)}``. ``books_out`` receives, after every row, each
    side's queue in priority order as ``[(oid, price, visible), ...]`` (F6 "slice
    shown"). ``ice_out`` receives one dict per accepted incoming iceberg (sweep facts).
    ``acts_out`` receives, after every row, ``{oid: row index where its current stamp
    was set}`` (its arrival row; for a regenerated slice, its activation row).
    """
    book: Dict[int, List[_Entry]] = {BUY: [], SELL: []}
    counter = [0]

    def nseq() -> int:
        counter[0] += 1
        return counter[0]

    def key(e: _Entry):
        return (-e.price, e.seq) if e.side == BUY else (e.price, e.seq)

    def crosses(limit: int, px: int, is_buy: bool) -> bool:
        return px <= limit if is_buy else px >= limit

    def snap() -> None:
        if books_out is not None:
            books_out.append({s: [(e.oid, e.price, e.vis) for e in sorted(book[s], key=key)]
                              for s in (BUY, SELL)})
        if acts_out is not None:
            acts_out.append({e.oid: e.act for s in (BUY, SELL) for e in book[s]})

    out: List[Optional[List[List[int]]]] = []
    for i, r in enumerate(rows):
        side, price, qty, oid = int(r[C_SIDE]), int(r[C_PRICE]), int(r[C_QTY]), int(r[C_OID])
        ex, peak, ppd = int(r[C_EXEC]), int(r[C_PEAK]), int(r[C_PPD])
        if rejected_v214(r):
            out.append(None)
            snap()
            continue
        opp, is_buy = (SELL if side == BUY else BUY), side == BUY
        chain = ex == ICE and incoming == R4
        fills: List[List[int]] = []
        if ex == FOK:
            tot = 0
            for e in book[opp]:
                if crosses(price, e.price, is_buy):
                    tot += e.vis
                k, hid = e.k, e.hidden
                while hid > 0:
                    k += 1
                    if not crosses(price, e.base + k * e.ppd, is_buy):
                        break
                    v = min(e.peak, hid)
                    tot += v
                    hid -= v
            if tot < qty:
                out.append(fills)
                snap()
                continue
        my_seq = nseq()
        rem, swept = qty, 0
        facts = {"row": i, "oid": oid, "d": ppd, "P": peak, "b": price, "total": qty,
                 "stopped_by_instance": False, "boundary_cuts": 0, "rows": 0} if ex == ICE else None
        while rem > 0 and book[opp]:
            b = min(book[opp], key=key)
            lim = price + (swept // peak) * ppd if chain else price
            if not crosses(lim, b.price, is_buy):
                if facts is not None and crosses(price, b.price, is_buy):
                    facts["stopped_by_instance"] = True   # b would still cross: item (6)
                break
            q = min(rem, b.vis)
            if chain:
                room = peak * (swept // peak + 1) - swept
                if room < q:
                    facts["boundary_cuts"] += 1         # item (2): the incoming instance ends
                q = min(q, room)
            rem -= q
            swept += q
            b.vis -= q
            fills.append([oid, b.oid, b.price, q, oid] if is_buy else [b.oid, oid, b.price, q, oid])
            if facts is not None:
                facts["rows"] += 1
            if trace is not None:
                # Same-price entries ranked behind ``b`` whose stamp was set at a later row.
                behind = sorted(e.oid for e in book[opp]
                                if e is not b and e.price == b.price and e.act > b.act)
                trace.append({"cp": b.oid, "slice": b.regen, "stamped_at_row": b.act, "row": i,
                              "behind": behind,
                              # Continuation (item (4)): a trade against an incoming
                              # iceberg's rested remainder at an instance index >= k_e.
                              "incoming_k": b.k if b.incoming else None,
                              "incoming_ke": b.ke if b.incoming else None,
                              # The D4b pattern: a slice activated in an *earlier* process is
                              # served before a same-price order that arrived after it.
                              "d4b": b.regen and b.act < i and bool(behind)})
            if b.vis == 0:
                if b.hidden > 0:
                    b.k += 1
                    b.price = b.base + b.k * b.ppd
                    b.vis = min(b.peak, b.hidden)
                    b.hidden -= b.vis
                    b.seq, b.act, b.regen = nseq(), i, True
                else:
                    book[opp].remove(b)
        if rem > 0 and ex == NON:
            book[side].append(_Entry(oid=oid, side=side, base=price, price=price, vis=rem, hidden=0,
                                     peak=0, ppd=0, k=0, seq=my_seq, act=i))
        elif rem > 0 and ex == ICE:
            if chain:
                ke = swept // peak
                u = swept - ke * peak
                v1 = min(peak - u, rem)
            else:
                ke, u, v1 = 0, 0, min(peak, rem)
            book[side].append(_Entry(oid=oid, side=side, base=price, price=price + ke * ppd, vis=v1,
                                     hidden=rem - v1, peak=peak, ppd=ppd, k=ke, seq=my_seq, act=i,
                                     incoming=True, ke=ke))
        if facts is not None:
            ke = swept // peak
            facts.update(M=swept, R=rem, ke=ke, u=swept - ke * peak)
            if ice_out is not None:
                ice_out.append(facts)
        out.append(fills)
        snap()
    if book_out is not None:
        for s in (BUY, SELL):
            book_out[s] = {e.oid: (e.price, e.vis, e.hidden) for e in sorted(book[s], key=key)}
    return out


def _flat(per_sub) -> List[List[int]]:
    return [f for sub in per_sub if sub for f in sub]


# Classes of the first divergence between the r4 walk and the engine.
AGREE = "engine_agrees_with_r4_walk"                                 # W stale: judging artefact
REGEN_STAMP_AHEAD = "regenerated_slice_stamped_after_a_later_arrival"  # engine deviation (§5.2, Rule 4)
OTHER = "unclassified"


def classify(rows, engine: List[List[int]]) -> Tuple[str, Dict[str, Any]]:
    """Classify ``engine`` (flat ``[buyer, seller, price, qty, submitter]``) against the walk."""
    trace: list = []
    mine = _flat(reference_walk(rows, trace=trace))
    if engine == mine:
        return AGREE, {}
    i = 0
    while i < min(len(mine), len(engine)) and mine[i] == engine[i]:
        i += 1
    if i >= min(len(mine), len(engine)):
        return OTHER, {"at": i}
    m, e = mine[i], engine[i]
    sub = m[4]
    info: Dict[str, Any] = {"at": i, "expected": m, "observed": e, "submitter": sub}
    if e[4] != sub or e[2] != m[2]:
        return OTHER, info
    buy = int(next(r for r in rows if int(r[C_OID]) == sub)[C_SIDE]) == BUY
    cp_m, cp_e = (m[1], e[1]) if buy else (m[0], e[0])
    row_of = {int(r[C_OID]): j for j, r in enumerate(rows)}
    t = trace[i]
    info.update(cp_expected=cp_m, cp_engine=cp_e, expected_slice_stamped_at_row=t["stamped_at_row"],
                engine_cp_row=row_of[cp_e])
    # Same price, other counterparty. The walk's counterparty is a regenerated slice
    # stamped (eager activation) in an earlier process than the engine's counterparty
    # arrived: §5.2's single sequential queue puts the slice first.
    if t["slice"] and t["stamped_at_row"] < row_of[cp_e] < row_of[sub]:
        return REGEN_STAMP_AHEAD, info
    return OTHER, info


def _price_volume_by_submission(trades: List[List[int]]) -> Dict[int, Counter]:
    out: Dict[int, Counter] = defaultdict(Counter)
    for f in trades:
        out[f[4]][f[2]] += f[3]
    return dict(out)


# ---------------------------------------------------------------------------
# Regenerating draws by seed (public generator API)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _grid():
    from omiecid_simulator.jaxomiecid.omie_tape_synth import fuzz_grid
    return {r.name: r for r in fuzz_grid()}


def regenerate(ix_, seed: int):
    from omiecid_simulator.jaxomiecid.omie_tape_synth import CapacityEnvelope, make_random_session
    dd = ix_["draws"][seed]
    c = dd["cell"]
    pr = _pressures(ix_)[f"D={c['depth']},M={c['slots']}"]
    env = CapacityEnvelope(max_resting_per_side=pr["resting_margin"],
                           max_concurrent_icebergs=pr["slot_margin"])
    return make_random_session(seed, _grid()[dd["regime"]], envelope=env)


def _real(tape) -> np.ndarray:
    tape = np.asarray(tape)
    return tape[tape[:, C_OID] >= 1]


def _engine_trades(ix_, seed: int) -> List[List[int]]:
    return ix_["fail"][seed]["observed"]["reference_walk"]["main"]["trade_sequence"]["observed_trades"]


def _recorded_w(ix_, seed: int) -> List[List[int]]:
    return ix_["fail"][seed]["observed"]["reference_walk"]["main"]["trade_sequence"]["expected_eager_trades"]


# ---------------------------------------------------------------------------
# Independent engine probes via ContractBook's public API (jitted submit)
# ---------------------------------------------------------------------------
def _cb_run(rows: Sequence[Sequence[int]], *, book_depth: int = 100, max_icebergs: int = 1024,
            n_trades: int = 100):
    """Submit 13-column rows in order; returns (final state, per-submit states)."""
    import jax.numpy as jnp
    from omiecid_simulator.jaxomiecid.omie_data_loader import _row_to_order_traced
    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.contract_book import ContractBook
    from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
    contract = Contract(contract_id=0, delivery_start_minutes=555, delivery_end_minutes=570,
                        gct_minutes=495, status=ContractStatus.OPEN)
    pc = OMIEProductConfig()
    book = ContractBook(contract, pc, OMIEMarketConfig(book_depth=book_depth, max_icebergs=max_icebergs,
                                                       nTrades=n_trades))
    state = book.init()
    states = []
    for r in rows:
        state, res = book.submit_order(state, _row_to_order_traced(jnp.asarray(np.asarray(r, dtype=np.int32)), pc))
        assert bool(res.accepted), list(r)
        states.append(state)
    return state, states


def _cb_trades(state) -> List[Tuple[int, int, frozenset]]:
    """(price, qty, {passive oid, aggressor oid}) in ring order."""
    from omiecid_simulator.jaxob.jaxob_constants import TradesFeat as T
    t = np.asarray(state.lob.trades)
    t = t[t[:, T.P.value] > 0]
    return [(int(r[T.P.value]), abs(int(r[T.Q.value])),
             frozenset((int(r[T.PASS_OID.value]), int(r[T.AGRS_OID.value])))) for r in t]


def _cb_side(state, side: int) -> Dict[int, Tuple[int, int, int, int]]:
    """oid -> (price, visible qty, SEC, NSEC) of the live rows of one side."""
    from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat as F
    a = np.asarray(state.lob.bids if side == BUY else state.lob.asks)
    a = a[(a[:, F.OID.value] > 0) & (a[:, F.Q.value] > 0)]
    return {int(r[F.OID.value]): (int(r[F.P.value]), int(r[F.Q.value]), int(r[F.SEC.value]),
                                  int(r[F.NSEC.value])) for r in a}


def _as_cb(trades: List[List[int]]) -> List[Tuple[int, int, frozenset]]:
    return [(f[2], f[3], frozenset((f[0], f[1]))) for f in trades]


def _cb_queue(state, side: int) -> List[Tuple[int, int, int]]:
    """One side's live rows in the engine's stored priority order (best price, then
    lowest stored ``[SEC, NSEC]``) as ``[(oid, price, visible), ...]``."""
    rows = _cb_side(state, side)
    sgn = -1 if side == BUY else 1
    order = sorted(rows, key=lambda o: (sgn * rows[o][0], rows[o][2], rows[o][3]))
    return [(o, rows[o][0], rows[o][1]) for o in order]


N_TRADES_RING = 100   # nTrades: the trade ring (the E-C split is all-or-nothing on it)


def replay_errors(rows: Sequence[Sequence[int]], *, book_depth: int = 100, max_icebergs: int = 1024,
                  recorded: Optional[Tuple[int, int]] = None, tamper=None,
                  incoming: str = R4) -> List[str]:
    """Replays a tape-only session through ``ContractBook`` and judges it against the
    r4 walk **after every submission**: the trade rows so far (prices, quantities,
    both parties, in matching order: item (2) row split included), each side's queue in
    priority order with prices and visible quantities (F6 "slice shown"; §6.1; E2;
    item (4): the rested remainder at ``b + k_e d`` showing ``v_1``), stored
    ``[SEC, NSEC]`` unique across both sides (§5.2 single queue), every stored SEC
    equal to the tape second of the row that set that entry's stamp (arrival; for a
    slice its activation event, E2; item (4): no new timestamp on partial execution),
    hence none ahead of the tape second just processed. ``recorded`` = the emitted
    ``(n_trades, volume)`` of the draw. ``tamper(books)`` and ``incoming=OLD``
    (planted-defect tests only) corrupt the expectation."""
    errs: List[str] = []
    ts = [int(r[C_TS]) for r in rows]
    if any(b < a for a, b in zip(ts, ts[1:])):
        return ["precondition: tape seconds are not non-decreasing"]
    books: list = []
    acts: list = []
    per_sub = reference_walk(rows, books_out=books, acts_out=acts, incoming=incoming)
    flat = _flat(per_sub)
    if len(flat) > N_TRADES_RING:
        return [f"precondition: {len(flat)} walk rows exceed the {N_TRADES_RING}-row ring (OQ-15 gap)"]
    if tamper is not None:
        tamper(books)
    _, sts = _cb_run(rows, book_depth=book_depth, max_icebergs=max_icebergs)
    want: List[List[int]] = []
    for i, st in enumerate(sts):
        want += per_sub[i] or []
        if _cb_trades(st) != _as_cb(want):
            got = _cb_trades(st)
            j = next((j for j, (x, y) in enumerate(zip(got, _as_cb(want))) if x != y), min(len(got), len(want)))
            errs.append(f"row {i}: trades differ from the r4 walk at trade {j}: engine "
                        f"{got[j:j + 3]} vs walk {_as_cb(want)[j:j + 3]}")
            break
        for s in (BUY, SELL):
            if _cb_queue(st, s) != books[i][s]:
                errs.append(f"row {i}: side {s} queue {_cb_queue(st, s)} != walk {books[i][s]}")
        live = {o: v for s in (BUY, SELL) for o, v in _cb_side(st, s).items()}
        stamps = [v[2:] for v in live.values()]
        if len(set(stamps)) != len(stamps):
            errs.append(f"row {i}: duplicate stored [SEC, NSEC] in the contract")
        if stamps and max(sec for sec, _ in stamps) > ts[i]:
            errs.append(f"row {i}: a stored SEC {max(sec for sec, _ in stamps)} is ahead of the tape second {ts[i]}")
        wrong = {o: (v[2], ts[acts[i][o]]) for o, v in live.items() if o in acts[i] and v[2] != ts[acts[i][o]]}
        if wrong:
            errs.append(f"row {i}: stored SEC != the second of the row that set the stamp {wrong}")
        if errs:
            break
    if recorded is not None and (len(flat), sum(f[3] for f in flat)) != tuple(recorded):
        errs.append(f"walk counts {(len(flat), sum(f[3] for f in flat))} != recorded {tuple(recorded)}")
    return errs


def d4b_sites(rows: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    """Walk fills where a slice activated in an earlier process is served before a
    same-price order that arrived after the activation (the r2 "D4b" pattern)."""
    tr: list = []
    reference_walk(rows, trace=tr)
    return [dict(t, at=j) for j, t in enumerate(tr) if t["d4b"]]


def chain_sites(rows: Sequence[Sequence[int]]) -> Dict[str, int]:
    """Counts of the amended-rule features the walk exercises on one session: incoming
    icebergs that cross on entry (``crossing``; ``crossing_d`` with d != 0), sweeps
    stopped by an instance limit while ``b`` still crosses (item (6)), rows cut at an
    incoming instance boundary (item (2)), remainders resting with ``u > 0`` (item (4)
    display) and trades against a rested remainder at instance >= k_e > 0
    (``continued``, item (4) continuation)."""
    ice: list = []
    tr: list = []
    reference_walk(rows, ice_out=ice, trace=tr)
    c: Counter = Counter()
    for x in ice:
        if x["M"] == 0:
            continue
        c["crossing"] += 1
        c["crossing_d"] += bool(x["d"])
        c["stopped_by_instance"] += x["stopped_by_instance"]
        c["boundary_cut_d0" if x["d"] == 0 else "boundary_cut_d"] += bool(x["boundary_cuts"])
        c["rest_part_used"] += x["R"] > 0 and x["u"] > 0
        c["rest_ke_pos_d"] += x["R"] > 0 and x["ke"] > 0 and bool(x["d"])
    c["continued"] = sum(1 for t in tr if t["incoming_k"] is not None and t["incoming_ke"] > 0)
    return dict(c)


def _row(oid, side, price, qty, ex, peak=0, ppd=0, ts=None):
    r = [0] * 13
    r[C_SIDE], r[C_PRICE], r[C_QTY], r[C_OID] = side, price, qty, oid
    r[C_AGENT], r[C_TS], r[C_EXEC], r[C_PEAK], r[C_PPD] = 1, oid if ts is None else ts, ex, peak, ppd
    return r


def _px(eur: float) -> int:
    return int(round((eur + 9999.0) * 100))


# ===========================================================================
# Fast tests -- the judge itself on the spec's own examples
# ===========================================================================
class TestReferenceWalkOnTheSpecExamples:
    """The r4 walk, checked on market-spec's own worked examples and the amended Rule 2 items (rule 1)."""

    def test_rule1_icewpi_fok_rejected_then_accepted(self):
        ice = _row(1, SELL, _px(50.0), 100, ICE, peak=20, ppd=50)
        assert reference_walk([ice, _row(2, BUY, _px(51.0), 80, FOK)])[1] == []
        got = reference_walk([ice, _row(2, BUY, _px(51.0), 60, FOK)])[1]
        assert [(f[2], f[3]) for f in got] == [(_px(50.0), 20), (_px(50.5), 20), (_px(51.0), 20)]

    def test_rule1_standard_iceberg_fok_sees_hidden(self):
        ice = _row(1, SELL, _px(50.0), 70, ICE, peak=20)
        got = reference_walk([ice, _row(2, BUY, _px(50.0), 30, FOK)])[1]
        assert [(f[2], f[3]) for f in got] == [(_px(50.0), 20), (_px(50.0), 10)]

    def test_rule2_one_order_consumes_four_slices(self):
        ice = _row(1, SELL, _px(50.0), 100, ICE, peak=30)
        got = reference_walk([ice, _row(2, BUY, _px(55.0), 100, NON)])[1]
        assert [(f[2], f[3]) for f in got] == [(_px(50.0), q) for q in (30, 30, 30, 10)]

    def test_rule2_reranking_older_same_price_order_goes_before_the_new_slice(self):
        rows = [_row(1, SELL, _px(50.0), 40, ICE, peak=10), _row(2, SELL, _px(50.0), 15, NON),
                _row(3, BUY, _px(50.0), 30, NON)]
        got = reference_walk(rows)[2]
        assert [(f[1], f[3]) for f in got] == [(1, 10), (2, 15), (1, 5)]

    def test_rule2_reranking_never_a_worse_level_while_a_better_slice_can_trade(self):
        rows = [_row(1, BUY, 1000103, 90, ICE, peak=22), _row(2, BUY, 1000101, 64, ICE, peak=32),
                _row(3, SELL, 1000090, 60, IOC)]
        assert _flat(reference_walk(rows)) == [[1, 3, 1000103, 22, 3], [1, 3, 1000103, 22, 3],
                                               [1, 3, 1000103, 16, 3]]

    def test_rule4_eager_activation_stamps_in_exhaustion_order(self):
        # OMIE-REGEN-EXHAUSTION-ORDER-1 reproducer: B (oid 1) then A (oid 2), both 10/1.
        rows = [_row(1, SELL, _px(50.0), 10, ICE, peak=1), _row(2, SELL, _px(50.0), 10, ICE, peak=1),
                _row(3, BUY, _px(50.0), 1, NON), _row(4, BUY, _px(50.0), 2, NON),
                _row(5, BUY, _px(50.0), 1, NON)]
        w = reference_walk(rows)
        assert [f[1] for f in w[3]] == [2, 1] and [f[1] for f in w[4]] == [2]
        big = rows[:3] + [_row(4, BUY, _px(50.0), 4, NON)]
        assert [f[1] for f in reference_walk(big)[3]] == [2, 1, 2, 1]  # A1, B2, A2, B3

    # --- Rule 2 *Incoming iceberg* (amended 2026-10-20): the instance chain ----------
    def test_incoming_iceberg_rows_split_at_instance_boundaries_d0(self):
        """Item (2), d = 0: one row per (resting entry, incoming instance) -- 20, 20, 10,
        not one 50 row. M = 50, k_e = 2, u = 10, R = 10: v_1 = min(P - u, R) = 10."""
        rows = [_row(1, SELL, _px(50.0), 50, NON), _row(2, BUY, _px(50.0), 60, ICE, peak=20)]
        book: dict = {}
        w = reference_walk(rows, book_out=book)
        assert w[1] == [[2, 1, _px(50.0), q, 2] for q in (20, 20, 10)]
        assert book[BUY] == {2: (_px(50.0), 10, 0)} and book[SELL] == {}

    def test_incoming_iceberg_part_used_instance_display_d0(self):
        """Items (2), (4), d = 0 (re-authored from r3 :925-931). M = 45: rows 30, 15;
        k_e = 1, u = 15, R = 55 rests showing v_1 = min(30 - 15, 55) = 15, hidden 40 (not
        min(P, R) = 30). A later SELL 40 takes 15, then the next slice min(30, 40) = 30
        is activated and gives 25, leaving 5 shown and 10 hidden."""
        rows = [_row(1, SELL, _px(50.0), 45, NON), _row(2, BUY, _px(50.0), 100, ICE, peak=30)]
        book: dict = {}
        w = reference_walk(rows, book_out=book)
        assert [f[3] for f in w[1]] == [30, 15]
        assert book[BUY] == {2: (_px(50.0), 15, 40)}
        rows.append(_row(3, SELL, _px(50.0), 40, NON))
        w = reference_walk(rows, book_out=book)
        assert [(f[2], f[3]) for f in w[2]] == [(_px(50.0), 15), (_px(50.0), 25)]
        assert book[BUY] == {2: (_px(50.0), 5, 10)} and book[SELL] == {}

    def test_incoming_iceberg_vs_resting_iceberg_priced_by_the_resting_one(self):
        """Shape of seed 100000244 (items (2), (3); Rule 3 *ICE aggressor*): every trade at
        the resting iceberg's limit; rows end at each resting slice (6) and at the
        incoming instance boundary 31: 6 x 5 = 30, then the last slice 4 splits 1 + 3.
        M = 34, k_e = 1, u = 3, R = 28: v_1 = min(31 - 3, 28) = 28."""
        rows = [_row(3, SELL, 1005194, 34, ICE, peak=6, ts=8), _row(5, BUY, 1005245, 62, ICE, peak=31, ts=11)]
        book: dict = {}
        w = reference_walk(rows, book_out=book)
        assert [(f[2], f[3]) for f in w[1]] == [(1005194, q) for q in (6, 6, 6, 6, 6, 1, 3)]
        assert book[BUY] == {5: (1005245, 28, 0)} and book[SELL] == {}

    def test_incoming_icewpi_stops_at_first_instance_that_cannot_trade(self):
        """Item (6) (re-authored from r3 :942-948; the OQ-34 base-limit choice and finding
        F18 are retired): instance 1's limit 49.00 does not cross the 50.00 ask, so the
        sweep stops although the base 50.00 still crosses. M = 10, k_e = 1, u = 0: the
        remainder 40 rests as instance 1 at 49.00 showing 10, hidden 30."""
        rows = [_row(1, SELL, _px(50.0), 30, NON), _row(2, BUY, _px(50.0), 50, ICE, peak=10, ppd=-100)]
        book: dict = {}
        w = reference_walk(rows, book_out=book)
        assert w[1] == [[2, 1, _px(50.0), 10, 2]]
        assert book[BUY] == {2: (_px(49.0), 10, 30)} and book[SELL] == {1: (_px(50.0), 20, 0)}

    def test_incoming_icewpi_stop_mirrored_sell_side_off_round(self):
        """Item (6), SELL side, off-round prices, 3- and 5-digit OIDs: instance 0 (limit b)
        takes 7 of the 12 best-bid units, instance 1 (b + 13 = 1004984) still crosses
        1004990 and takes the other 5; instance 1 does not cross 1004975 -> stop with
        base-crossing liquidity left. M = 12, k_e = 1, u = 5: R = 28 rests at b + 13
        showing min(7 - 5, 28) = 2, hidden 26."""
        b = 1004971
        rows = [_row(40404, BUY, 1004990, 12, NON, ts=1), _row(777, BUY, 1004975, 9, NON, ts=2),
                _row(31337, SELL, b, 40, ICE, peak=7, ppd=13, ts=3)]
        book: dict = {}
        w = reference_walk(rows, book_out=book)
        assert w[2] == [[40404, 31337, 1004990, 7, 31337], [40404, 31337, 1004990, 5, 31337]]
        assert book[SELL] == {31337: (b + 13, 2, 26)}
        assert book[BUY] == {777: (1004975, 9, 0)}

    def test_gestvd2_replica_continuation_at_k_e(self):
        """OQ-34 known positive (orders_20250610.1:125794), tape units (0.1 MW, 0.01 EUR):
        SELL ICE 110.00, 100 MW, P = 5, d = +0.75 vs bids 1.0 @ 112.36, 20.5 @ 112.34,
        11.9 @ 112.00, 10.0 @ 110.00 -> 1.0 @ 112.36 then 4.0, 5.0, 5.0, 5.0 @ 112.34;
        instance 4 (113.00) stops. M = 20 MW, k_e = 4, u = 0: 5.0 @ 113.00 shown. Item (4)
        continuation: a later BUY 7.0 @ 114.00 trades 5.0 @ 113.00 and 2.0 @ 113.75."""
        rows = [_row(11, BUY, _px(112.36), 10, NON), _row(12, BUY, _px(112.34), 205, NON),
                _row(13, BUY, _px(112.00), 119, NON), _row(14, BUY, _px(110.00), 100, NON),
                _row(15, SELL, _px(110.00), 1000, ICE, peak=50, ppd=75)]
        book: dict = {}
        w = reference_walk(rows, book_out=book)
        assert [(f[0], f[2], f[3]) for f in w[4]] == [(11, _px(112.36), 10)] + [
            (12, _px(112.34), q) for q in (40, 50, 50, 50)]
        assert book[SELL] == {15: (_px(113.00), 50, 750)}
        assert book[BUY] == {12: (_px(112.34), 15, 0), 13: (_px(112.00), 119, 0),
                             14: (_px(110.00), 100, 0)}
        w = reference_walk(rows + [_row(16, BUY, _px(114.00), 70, NON)], book_out=book)
        assert [(f[2], f[3]) for f in w[5]] == [(_px(113.00), 50), (_px(113.75), 20)]
        assert book[SELL] == {15: (_px(113.75), 30, 700)}

    def test_part_used_icewpi_instance_and_its_continuation(self):
        """Items (2), (4), (6), BUY d = -0.50 (tape units): BUY ICE 51.00, 10.0, P = 2.0 vs
        SELL 3.0 @ 50.00 -> 2.0 and 1.0 @ 50.00 (instance 1, limit 50.50, still crosses).
        k_e = 1, u = 1.0: the remainder 7.0 shows 1.0 @ 50.50. A later SELL 7.0 @ 49.00
        prints 1.0 @ 50.50, 2.0 @ 50.00, 2.0 @ 49.50, 2.0 @ 49.00 (continued, not restarted)."""
        rows = [_row(1, SELL, _px(50.0), 30, NON), _row(2, BUY, _px(51.0), 100, ICE, peak=20, ppd=-50)]
        book: dict = {}
        w = reference_walk(rows, book_out=book)
        assert [(f[2], f[3]) for f in w[1]] == [(_px(50.0), 20), (_px(50.0), 10)]
        assert book[BUY] == {2: (_px(50.5), 10, 60)}
        w = reference_walk(rows + [_row(3, SELL, _px(49.0), 70, NON)], book_out=book)
        assert [(f[2], f[3]) for f in w[2]] == [(_px(50.5), 10), (_px(50.0), 20), (_px(49.5), 20),
                                                (_px(49.0), 20)]
        assert book[BUY] == {} and book[SELL] == {}

    def test_standard_incoming_iceberg_split_over_two_levels(self):
        """Item (2), d = 0, two price levels: BUY ICE 51.00, 20.0, P = 4.0 vs SELL 6.0 @
        50.00, 6.0 @ 50.50 -> rows 4.0, 2.0 @ 50.00 and 2.0, 4.0 @ 50.50; M = 12.0,
        k_e = 3, u = 0: the remainder shows 4.0 @ 51.00, hidden 4.0."""
        rows = [_row(1, SELL, _px(50.0), 60, NON), _row(2, SELL, _px(50.5), 60, NON),
                _row(3, BUY, _px(51.0), 200, ICE, peak=40)]
        book: dict = {}
        w = reference_walk(rows, book_out=book)
        assert [(f[1], f[2], f[3]) for f in w[2]] == [(1, _px(50.0), 40), (1, _px(50.0), 20),
                                                      (2, _px(50.5), 20), (2, _px(50.5), 40)]
        assert book[BUY] == {3: (_px(51.0), 40, 40)}

    def test_incoming_iceberg_remainder_keeps_arrival_stamp_and_rank(self):
        """Item (4) / §5.2: no new timestamp on partial execution. The rested remainder
        (v_1 = min(4 - 1, 25) = 3) ranks ahead of a later same-price NON."""
        rows = [_row(1, SELL, _px(50.0), 5, NON), _row(2, BUY, _px(50.0), 30, ICE, peak=4),
                _row(3, BUY, _px(50.0), 9, NON), _row(4, SELL, _px(50.0), 3, NON)]
        w = reference_walk(rows)
        assert [f[3] for f in w[1]] == [4, 1]
        assert w[3] == [[2, 4, _px(50.0), 3, 4]]

    def test_fok_after_entry_counts_the_remainder_from_k_e(self):
        """Rule 1 with item (4): the FOK pre-check projects the rested remainder's slices
        from instance k_e on, not from slice 0. SELL ICEwPI 50.00, P = 2, d = +0.50 sweeps
        3 against a bid at 51.00 (both instances cross; priced 51.00 by the resting bid);
        k_e = 1, u = 1: it rests 1 @ 50.50, next slices at 51.00, 51.50, ... A BUY FOK 5 @
        51.00 sees 1 + 2 = 3 < 5 -> killed; a FOK 3 fills 1 @ 50.50 and 2 @ 51.00."""
        base = [_row(1, BUY, _px(51.0), 3, NON), _row(2, SELL, _px(50.0), 10, ICE, peak=2, ppd=50)]
        assert [(f[2], f[3]) for f in reference_walk(base)[1]] == [(_px(51.0), 2), (_px(51.0), 1)]
        assert reference_walk(base + [_row(3, BUY, _px(51.0), 5, FOK)])[2] == []
        got = reference_walk(base + [_row(3, BUY, _px(51.0), 3, FOK)])[2]
        assert [(f[2], f[3]) for f in got] == [(_px(50.5), 1), (_px(51.0), 2)]

    def test_superseded_rule_variant_differs_on_the_amended_items(self):
        """Non-null twin of the r4 walk: the superseded reading (``OLD``, planted-defect
        use only) gives the r3 answers on the cases above."""
        rows = [_row(1, SELL, _px(50.0), 30, NON), _row(2, BUY, _px(50.0), 50, ICE, peak=10, ppd=-100)]
        book: dict = {}
        assert reference_walk(rows, book_out=book, incoming=OLD)[1] == [[2, 1, _px(50.0), 30, 2]]
        assert book[BUY] == {2: (_px(50.0), 10, 10)}
        rows = [_row(1, SELL, _px(50.0), 50, NON), _row(2, BUY, _px(50.0), 60, ICE, peak=20)]
        assert reference_walk(rows, incoming=OLD)[1] == [[2, 1, _px(50.0), 50, 2]]
        rows = [_row(1, SELL, _px(50.0), 45, NON), _row(2, BUY, _px(50.0), 100, ICE, peak=30)]
        reference_walk(rows, book_out=book, incoming=OLD)
        assert book[BUY] == {2: (_px(50.0), 30, 25)}

    def test_v214_iceberg_rejections(self):
        assert reference_walk([_row(1, BUY, _px(50.0), 10, ICE, peak=5, ppd=10)]) == [None]
        assert reference_walk([_row(1, SELL, _px(50.0), 10, ICE, peak=5, ppd=-10)]) == [None]
        assert reference_walk([_row(1, SELL, _px(50.0), 10, ICE, peak=11)]) == [None]
        assert reference_walk([_row(1, SELL, _px(50.0), 10, ICE, peak=10)]) == [[]]

    def test_ioc_remainder_cancelled_fok_kill_leaves_book(self):
        book: dict = {}
        w = reference_walk([_row(1, SELL, _px(50.0), 5, NON), _row(2, BUY, _px(50.0), 9, IOC),
                            _row(3, SELL, _px(49.0), 4, NON), _row(4, BUY, _px(49.0), 5, FOK)],
                           book_out=book)
        assert w[1] == [[2, 1, _px(50.0), 5, 2]] and w[3] == []
        assert book[BUY] == {} and book[SELL] == {3: (_px(49.0), 4, 0)}

    def test_classifier_on_synthetic_divergences(self):
        rows = [_row(1, SELL, 1005000, 40, ICE, peak=2, ts=1), _row(2, BUY, 1005000, 21, NON, ts=2),
                _row(3, SELL, 1005000, 15, NON, ts=3), _row(4, BUY, 1005000, 1, NON, ts=4)]
        mine = _flat(reference_walk(rows))
        assert classify(rows, mine)[0] == AGREE
        assert mine[-1] == [4, 1, 1005000, 1, 4]
        late = mine[:-1] + [[4, 3, 1005000, 1, 4]]
        c, info = classify(rows, late)
        assert c == REGEN_STAMP_AHEAD and info["cp_expected"] == 1 and info["cp_engine"] == 3
        repriced = mine[:-1] + [[4, 1, 1005001, 1, 4]]
        assert classify(rows, repriced)[0] == OTHER


# ===========================================================================
# Fast tests -- the emitted JSON only
# ===========================================================================
class TestSchemaStatusAccounting:
    def test_schema_and_semantic_validator_accept(self, payload):
        assert validate_results(payload) == []

    def test_status_failure_kinds_and_explanation_consistent(self, payload):
        """Status, kinds and explanation agree with counts derived from the payload itself
        (failed assertions, failing seeds, draw strings) -- no run's literal count."""
        assert status_errors(payload) == []
        n_failed = sum(a["status"] == "fail" for a in payload["assertions"])
        n_fail_cells = sum(a["status"] == "fail" for a in payload["assertions"] if _CELL_RE.match(a["name"]))
        assert n_failed == len(payload["seeds"]["failing"]) + n_fail_cells
        assert (payload["status"] == "PASS") == (n_failed == 0 and not payload["failure_kinds"])

    def test_failure_kinds_rederived_independently(self, payload):
        assert payload["failure_kinds"] == kinds_from_content(payload)

    def test_draw_and_seed_accounting(self, payload):
        assert accounting_errors(payload) == []

    def test_failed_assertions_are_draw_records_and_their_cells(self, payload):
        assert failed_assertion_errors(payload) == []

    def test_only_the_reference_walk_fails(self, payload, ix):
        # Battery and relations clean on every draw; every failing draw fails W alone.
        for cid in ALL_IDS:
            assert ix["inv"][cid]["n_fail"] == 0 and ix["inv"][cid]["status"] == "pass", cid
        for s in payload["seeds"]["failing"]:
            assert ix["draws"][s]["W"] == "f", s
            assert ix["fail"][s]["observed"]["failing_checks"] == {}, s

    def test_r4_every_draw_clean_and_status_pass(self, payload, ix):
        """r4 (after S2 E-A/E-B/E-C and the K2 r4 W): zero exceptions on all 2 200 draws.
        Backed independently by the engine replays in TestEngineReplay (every draw with
        a crossing incoming iceberg) and TestRegenStampDeviation."""
        assert payload["status"] == "PASS" and payload["failure_kinds"] == []
        assert payload["failure_explanation"] is None
        assert payload["seeds"]["failing"] == [] and ix["fail"] == {}
        assert len(ix["draws"]) == payload["draws"]["valid"] == payload["draws"]["requested"]
        bad = [s for s, dd in ix["draws"].items() if dd["W"] != "p" or "f" in dd["status"].values()]
        assert bad == []
        assert all(a["status"] == "pass" for a in payload["assertions"])

    def test_known_result_states_the_amended_incoming_rule(self, payload):
        kr = payload["known_result"]
        for frag in ("chain of instances", "b + k d", "first instance that cannot trade",
                     "k_e = floor(M/P)", "min(P - u, R)", "OQ-34"):
            assert frag in kr, frag
        assert "base limit" not in kr and "modelling choice" not in kr

    def test_no_failing_check_is_hidden_behind_not_applicable(self, ix):
        for cid in ALL_IDS:
            r = ix["inv"][cid]
            assert r["n_pass"] + r["n_fail"] > 0, f"{cid} never applied"

    def test_unobserved_vocabulary_is_named(self, ix):
        vocab = {"rejection_reason", "queue_timestamp_correctness", "independent_valuation",
                 "matching_iterations"}
        for r in ix["inv"].values():
            assert set(r["unobserved"]) <= vocab, r

    def test_market_spec_quotes_stamped_v2_14_and_ok(self, payload):
        ms = [q for q in payload["spec"]["quotes"] if q["source"] == "market-spec.md"]
        assert ms and all(q["version"] == SPEC_VERSION and q["ok"] for q in ms)
        assert all(q["ok"] for q in payload["spec"]["quotes"])
        head = (REPO_ROOT / "docs" / "omiecid" / "market-spec.md").read_text(encoding="utf-8")[:2000]
        assert f"**Status:** {SPEC_VERSION}" in head

    def test_superseded_oq33_oq34_blocks_retired_and_new_blocks_quoted(self, payload):
        """r4 table: the OQ-33 "Ruling..." and OQ-34 "The simulator's convention..." quotes
        are retired; the 2026-10-20 blocks of OQ-33 and OQ-34, and OQ-35, are quoted."""
        oq = [(q["section"], q["block"] or "") for q in payload["spec"]["quotes"]
              if q["source"] == "open-questions.md"]
        assert not any(sec == "OQ-33" and blk.startswith("**Ruling") for sec, blk in oq), oq
        assert not any(sec == "OQ-34" and "simulator's convention" in blk for sec, blk in oq), oq
        assert ("OQ-33", "**Revised by user ruling (2026-10-20") in oq
        assert ("OQ-34", "**Resolved by user ruling (2026-10-20") in oq
        assert any(sec == "OQ-35" for sec, _ in oq)
        text = (REPO_ROOT / "docs" / "omiecid" / "open-questions.md").read_text(encoding="utf-8")
        assert "**Revised by user ruling (2026-10-20" in text and "**Resolved by user ruling (2026-10-20" in text


class TestCoverage:
    def test_sweep_cells_regimes_lengths_and_pressures(self, payload):
        assert coverage_errors(payload) == []

    def test_every_grid_regime_is_in_the_gate(self, ix):
        regimes = {c["regime"] for (v, _), c in ix["cells"].items() if v == "C15b"}
        assert regimes == set(_grid()), sorted(set(_grid()) ^ regimes)[:5]

    def test_generator_registry_digest_matches(self, payload):
        from omiecid_simulator.jaxomiecid.omie_tape_synth import construction_digest
        got = payload["generator"]["digests"].get("C15.random_session")
        assert payload["generator"]["module"] == "omiecid_simulator.jaxomiecid.omie_tape_synth"
        assert got == construction_digest("C15.random_session")


class TestEagerJit:
    def test_eager_evidence_present_and_identical(self, payload):
        assert eager_errors(payload) == []


class TestShrinker:
    def test_shrinker_records_point_at_failing_seeds(self, payload, ix):
        failing = set(payload["seeds"]["failing"])
        if not failing:
            assert ix["shrinks"] == [] or all(a["status"] == "pass" for a, *_ in ix["shrinks"])
            return
        assert ix["shrinks"], "failing draws but no shrinker record"
        for a, var, seed, r, n in ix["shrinks"]:
            assert seed in failing
            assert 1 <= r < n == ix["draws"][seed]["cell"]["n"]
            assert a["expected"] in a["observed"], a["name"]
            m = re.search(r"rows (\[\[.*\]\])$", a["detail"])
            assert m, a["detail"]
            rows = json.loads(m.group(1))
            assert len(rows) == r and all(len(x) == 13 for x in rows)


class TestSeed100000244UnderTheR4Walk:
    """The r2/r3 exemplar of the incoming-iceberg row split (BUY ICE 62/31 against a
    resting SELL ICE 34/6). Under r4 the last resting slice (4) is split 1 + 3 at the
    incoming instance boundary 31 (item (2)); the emitted counts must carry the split."""

    def test_seed_100000244_clean_with_r4_counts(self, ix):
        """~1 s: one generator call + the walk (no engine)."""
        dd = ix["draws"][100000244]
        assert dd["W"] == "p" and "f" not in dd["status"].values()
        rows = _real(regenerate(ix, 100000244).tape)
        w = reference_walk(rows)
        sub5 = [f for sub in w if sub for f in sub if f[4] == 5]
        assert [(f[2], f[3]) for f in sub5] == [(1005194, q) for q in (6, 6, 6, 6, 6, 1, 3)]
        flat = _flat(w)
        assert (len(flat), sum(f[3] for f in flat)) == (dd["n_trades"], dd["volume"])
        old = _flat(reference_walk(rows, incoming=OLD))
        assert len(old) != dd["n_trades"]           # non-null: the superseded rule is caught


class TestBrokenCopiesAreRejected:
    """Deliberately broken copies, written to tmp_path and reloaded (rule 1)."""

    def test_status_flipped(self, payload, tmp_path):
        """PASS -> NOT CERTIFIED (with a plausible explanation) or NOT CERTIFIED -> PASS."""
        q = _mutable(payload)
        if payload["status"] == "PASS":
            q["status"] = "NOT CERTIFIED"
            q["failure_kinds"] = ["oracle_mismatch"]
            s = sorted(ix_s for ix_s in index(payload)["draws"])[0]
            q["failure_explanation"] = {"what_failed": "0 assertion(s) failed",
                                        "smallest_case": {"seed": s, "rows": 1}}
        else:
            q["status"] = "PASS"
        assert status_errors(_roundtrip(tmp_path, q))

    def test_failure_kinds_tampered(self, payload, tmp_path):
        q = _mutable(payload)
        q["failure_kinds"] = (payload["failure_kinds"][1:] if payload["failure_kinds"]
                              else ["oracle_mismatch"])
        assert status_errors(_roundtrip(tmp_path, q))

    def test_failing_seeds_tampered(self, payload, tmp_path):
        q = _mutable(payload, "seeds")
        if q["seeds"]["failing"]:
            q["seeds"]["failing"] = q["seeds"]["failing"][1:]
        else:  # plant: a clean draw listed as failing
            q["seeds"]["failing"] = [sorted(index(payload)["draws"])[7]]
        assert accounting_errors(_roundtrip(tmp_path, q))

    @staticmethod
    def _plant_in_draw(payload, field: int, ch: str):
        """Copy with one draw string's status field ``field`` (2 = I/RB, 4 = W) set to ``ch``
        at position 0; returns (copy, seed)."""
        q = _mutable(payload, "assertions")
        for a in q["assertions"]:
            if _CELL_RE.match(a["name"]) and a["observed"]["draws"]:
                parts = a["observed"]["draws"][0].split("|")
                parts[field] = ch + parts[field][1:]
                a["observed"]["draws"][0] = "|".join(parts)
                return q, int(parts[0])
        raise AssertionError("no cell with a draw")

    def test_planted_w_failure_under_unchanged_status(self, payload, tmp_path):
        """A draw whose W failed, with status/seeds/records left as emitted, must be caught."""
        q, seed = self._plant_in_draw(payload, 4, "f")
        q = _roundtrip(tmp_path, q)
        assert index(q)["draws"][seed]["W"] == "f"
        if payload["status"] == "PASS":
            assert any("failing draw exists" in e or e.startswith("PASS with") for e in status_errors(q))
        assert any("seeds.failing" in e for e in accounting_errors(q))

    def test_planted_battery_failure_under_unchanged_status(self, payload, tmp_path):
        """An RB 'f' in one draw string disagrees with the invariant row counts."""
        q, seed = self._plant_in_draw(payload, 2, "f")
        q = _roundtrip(tmp_path, q)
        assert index(q)["draws"][seed]["status"]["RB"] == "f"
        assert any(e.startswith("RB:") for e in accounting_errors(q))
        if payload["status"] == "PASS":
            assert status_errors(q)

    def test_invariant_count_edited(self, payload, tmp_path):
        q = _mutable(payload, "invariants")
        row = next(r for r in q["invariants"] if r["check_id"] == "I4")
        row["n_pass"] -= 1
        row["n_not_applicable"] += 1
        assert accounting_errors(_roundtrip(tmp_path, q))

    def test_discarded_draws_hidden(self, payload, tmp_path):
        q = _mutable(payload, "draws")
        q["draws"]["discarded"] = 7
        assert accounting_errors(_roundtrip(tmp_path, q))

    def test_length_160_cells_removed(self, payload, tmp_path):
        q = _mutable(payload)
        q["assertions"] = [a for a in payload["assertions"]
                           if not (_CELL_RE.match(a["name"]) and a["observed"]["cell"]["n"] == 160)]
        assert coverage_errors(_roundtrip(tmp_path, q))

    def test_eager_evidence_absent(self, payload, tmp_path):
        q = _mutable(payload)
        q["eager_jit_identical"] = None
        q["assertions"] = [a for a in payload["assertions"] if a["kind"] != "eager_jit"]
        assert eager_errors(_roundtrip(tmp_path, q))

    def test_eager_evidence_missing_for_one_mix(self, payload, tmp_path):
        q = _mutable(payload)
        q["assertions"] = [a for a in payload["assertions"]
                           if not (a["kind"] == "eager_jit" and "all_icewpi" in a["detail"])]
        assert eager_errors(_roundtrip(tmp_path, q))

    def test_ceiling_misreported(self, payload, tmp_path):
        q = _mutable(payload, "capacity")
        q["capacity"]["envelope"]["trades"] = 100
        assert coverage_errors(_roundtrip(tmp_path, q))

    def test_cell_status_tampered(self, payload, tmp_path):
        """A failing cell masked as pass, or (clean payload) a clean cell marked failed."""
        q = _mutable(payload, "assertions")
        a = next((a for a in q["assertions"] if _CELL_RE.match(a["name"]) and a["status"] == "fail"),
                 None)
        if a is not None:
            a["status"] = "pass"
        else:
            a = next(a for a in q["assertions"] if _CELL_RE.match(a["name"]))
            a["status"] = "fail"
        assert any("cell" in e and "status" in e for e in accounting_errors(_roundtrip(tmp_path, q)))

    def test_eager_mix_presence_count_tampered(self, payload, tmp_path):
        q = _mutable(payload, "assertions")
        a = next((a for a in q["assertions"] if a["name"].startswith("eager comparison present for mix")),
                 None)
        if a is None:
            pytest.skip("no per-mix eager presence record in this results file")
        a["observed"] += 1
        assert eager_errors(_roundtrip(tmp_path, q))

    def test_extra_failed_assertion(self, payload, tmp_path):
        q = _mutable(payload, "assertions")
        a = next(a for a in q["assertions"] if a["name"].startswith("C15a: valid draws"))
        a["status"] = "fail"
        assert failed_assertion_errors(_roundtrip(tmp_path, q))


# ===========================================================================
# Slow tests -- regeneration by seed, re-judging
# ===========================================================================
_SCAN_CACHE: Dict[int, Dict[int, Dict[str, Any]]] = {}


def _scan(ix_) -> Dict[int, Dict[str, Any]]:
    """Walks every emitted draw once per process (~30 s, generator + walk, no engine):
    the r4 walk's (rows, volume), the superseded rule's (rows, volume), the
    amended-rule features (:func:`chain_sites`) and whether the D4b pattern occurs."""
    if id(ix_) not in _SCAN_CACHE:
        out = {}
        for s in ix_["draws"]:
            rows = _real(regenerate(ix_, s).tape)
            tr: list = []
            f = _flat(reference_walk(rows, trace=tr))
            fo = _flat(reference_walk(rows, incoming=OLD))
            out[s] = {"r4": (len(f), sum(x[3] for x in f)), "old": (len(fo), sum(x[3] for x in fo)),
                      "chain": chain_sites(rows), "d4b": any(t["d4b"] for t in tr)}
        _SCAN_CACHE[id(ix_)] = out
    return _SCAN_CACHE[id(ix_)]


def _chain_draws(ix_) -> List[int]:
    """Every emitted draw holding an incoming iceberg that crosses on entry."""
    return sorted(s for s, v in _scan(ix_).items() if v["chain"].get("crossing"))


@pytest.mark.slow
class TestRegeneration:
    def test_every_draw_regenerates_by_seed(self, payload, ix):
        """slow: regenerates all 2 200 sessions (~15 s, generator only, no engine)."""
        largest = 0
        for seed, dd in ix["draws"].items():
            dr = regenerate(ix, seed)
            assert dr.digest()[:12] == dd["digest12"], seed
            assert dr.n_real_rows() == dd["cell"]["n"], seed
            assert dr.params["regime"] == dd["regime"]
            real = _real(dr.tape)
            assert len(set(real[:, C_OID].tolist())) == len(real)
            assert not any(rejected_v214(r) for r in real), seed  # RB: no §5.4 rejection expected
            ice = real[real[:, C_EXEC] == ICE]
            if len(ice):
                largest = max(largest, int(max(-(-int(q) // int(pk)) for q, pk in
                                               zip(ice[:, C_QTY], ice[:, C_PEAK]))))
                # OQ-35 (open, question only): no instance of any draw's ladder, incoming or
                # resting, can reach a price beyond the ±9,999 EUR/MWh limits.
                for r in ice:
                    deepest = int(r[C_PRICE]) + (-(-int(r[C_QTY]) // int(r[C_PEAK])) - 1) * int(r[C_PPD])
                    assert 0 <= deepest <= _px(9999.0), (seed, list(r))
            if dd["cell"]["mix"] in SMOKE_MIXES:
                assert set(real[:, C_EXEC].tolist()) <= {NON, IOC}
        assert largest <= MAX_SLICES_MARGIN
        reported = max(_summary(ix, f"REPORTED, not asserted: {v} aggregate")["observed"]
                       ["largest_slice_count"] for v in VARIATIONS)
        assert largest == reported

    def test_r4_walk_agrees_with_recorded_counts_on_every_draw(self, payload, ix):
        """slow (the walk scan, ~30 s): on every emitted draw the twin's own r4 walk
        gives the emitted (trade rows, volume) -- the row count carries the item (2)
        split, so W and this walk agree on it on all 2 200 draws."""
        sc = _scan(ix)
        bad = {s: (v["r4"], (ix["draws"][s]["n_trades"], ix["draws"][s]["volume"]))
               for s, v in sc.items() if v["r4"] != (ix["draws"][s]["n_trades"], ix["draws"][s]["volume"])}
        assert bad == {}, sorted(bad.items())[:5]
        assert len(sc) == len(ix["draws"]) > 0
        peak = max(v["r4"][0] for v in sc.values())
        assert peak == payload["capacity"]["peaks"]["trades"]     # the split rows are what is counted
        assert peak <= ENVELOPE["trades"] < N_TRADES_RING          # E-C split always had room

    def test_superseded_rule_disagrees_with_the_emitted_counts(self, ix):
        """Non-null twin: the superseded incoming-iceberg reading (base-limit sweep, no
        instance split, slice-0 restart, min(P, R) display) misses the emitted counts on
        some draws, all of them draws with a crossing incoming iceberg."""
        sc = _scan(ix)
        rec = {s: (dd["n_trades"], dd["volume"]) for s, dd in ix["draws"].items()}
        miss = sorted(s for s, v in sc.items() if v["old"] != rec[s])
        assert miss, "the emitted counts do not discriminate the amended rule"
        assert set(miss) <= set(_chain_draws(ix))

    def test_amended_rule_features_are_exercised(self, ix):
        """Coverage of the amended items on the emitted draws (walk-derived): crossing
        incoming icebergs with d = 0 and d != 0, item (6) stops while b still crosses,
        item (2) instance-boundary cuts for d = 0 and d != 0, item (4) part-used
        remainders (u > 0), k_e > 0 ICEwPI remainders and trades on a continued ladder."""
        tot: Counter = Counter()
        for v in _scan(ix).values():
            tot.update(v["chain"])
        for k in ("crossing", "crossing_d", "stopped_by_instance", "boundary_cut_d0", "boundary_cut_d",
                  "rest_part_used", "rest_ke_pos_d", "continued"):
            assert tot[k] >= 1, (k, dict(tot))
        assert tot["crossing"] > tot["crossing_d"]                # both d = 0 and d != 0


@pytest.mark.slow
class TestRejudgeFailingDraws:
    """Every W-failing draw, regenerated by seed and re-judged by the r4 walk."""

    @pytest.fixture(scope="class")
    def judged(self, payload, ix):
        out = {}
        for seed in payload["seeds"]["failing"]:
            rows = _real(regenerate(ix, seed).tape)
            out[seed] = (rows,) + classify(rows, _engine_trades(ix, seed))
        return out

    def test_every_failing_draw_is_classified(self, judged):
        """slow: walks every W-failing session (r2: 618; r3, r4: none)."""
        bad = {s: info for s, (_, c, info) in judged.items() if c == OTHER}
        assert bad == {}

    def test_recorded_w_contradicts_r4_walk_on_every_failing_draw(self, ix, judged):
        """W is stale: on every W-failing draw the r4 derivation differs from W."""
        same = [s for s, (rows, _, _) in judged.items() if _flat(reference_walk(rows)) == _recorded_w(ix, s)]
        assert same == []

    def test_engine_agrees_with_r4_walk_on_every_w_failure(self, judged):
        """After E2 no engine deviation is tolerated: on any W failure the engine's
        emitted trades must be the r4 derivation (W would be the artefact)."""
        agree = [s for s, (_, c, _) in judged.items() if c == AGREE]
        assert len(agree) == len(judged), sorted(set(judged) - set(agree))[:10]

    def test_no_residual_regen_slice_stamped_ahead(self, judged):
        """r2 found D4b on 100000254, 100001177, 100001190, 100001489; E2 fixed it. No
        W-failing draw may still show the pattern (r3, r4: none fails at all; the engine
        itself is re-judged on every D4b-pattern draw by TestRegenStampDeviation)."""
        dev = sorted(s for s, (_, c, _) in judged.items() if c == REGEN_STAMP_AHEAD)
        assert dev == []

    def test_regen_stamp_deviation_has_no_price_or_volume_effect(self, ix, judged):
        for s, (rows, c, info) in judged.items():
            if c != REGEN_STAMP_AHEAD:
                continue
            mine = _flat(reference_walk(rows))
            eng = _engine_trades(ix, s)
            assert _price_volume_by_submission(mine) == _price_volume_by_submission(eng), s
            assert info["expected"][2] == info["observed"][2]


# ---------------------------------------------------------------------------
# Shrunk fixtures (flipped by S3/S4: they now conform) and engine probes
# ---------------------------------------------------------------------------
def _fixture_candidates(payload) -> Dict[str, Path]:
    recorded: List[str] = []
    fe = payload.get("failure_explanation") or {}
    if (fe.get("smallest_case") or {}).get("fixture"):
        recorded.append(fe["smallest_case"]["fixture"])
    for a in payload["assertions"]:
        m = re.search(r"frozen to (\S+\.npz)", a.get("detail", ""))
        if m and a["name"].startswith("shrinker:"):
            recorded.append(m.group(1))
    for name in ("C15_shrunk_seed100000244.npz", "C15_shrunk_seed100000246.npz",
                 "C15_shrunk_seed100000249.npz"):
        recorded.append(name)
    found: Dict[str, Path] = {}
    for rp in recorded:
        name = Path(rp).name
        for cand in (_results_dir() / name, REPO_ROOT / rp,
                     REPO_ROOT / "tests" / "omiecid" / "fixtures" / "cert" / name,
                     REPO_ROOT / "notebooks" / "certification" / "fixtures" / name):
            if cand.is_file():
                found[name] = cand
                break
    return found


def _load_fixture(payload, seed: int):
    name = f"C15_shrunk_seed{seed}.npz"
    path = _fixture_candidates(payload).get(name)
    if path is None:
        pytest.skip(f"C15 fixture {name} absent")
    z = np.load(path, allow_pickle=False)
    return np.asarray(z["tape"]), json.loads(str(z["params_json"]))


def _fixture_seeds(payload) -> List[int]:
    return sorted(int(re.search(r"seed(\d+)", n).group(1)) for n in _fixture_candidates(payload)
                  if n.startswith("C15_shrunk_"))


@pytest.mark.slow
@pytest.mark.jit
class TestShrunkFixtures:
    """Each frozen r1 counterexample now conforms to v2.14 (S3/S4 flipped them)."""

    def test_every_fixture_now_conforms_to_the_rules(self, payload):
        """slow: one _submit_order compile + a few submits per fixture (~10 s)."""
        seeds = _fixture_seeds(payload)
        if not seeds:
            pytest.skip("no C15 shrunk fixture found")
        for seed in seeds:
            tape, params = _load_fixture(payload, seed)
            assert tape.dtype == np.int32 and tape.shape[1] == 13
            assert params.get("shrunk") is True and params.get("target") == "W"
            assert 1 <= params["rows_after"] < params["rows_before"]
            real = _real(tape)
            st, _ = _cb_run(real, book_depth=int(params["cell"]["depth"]),
                            max_icebergs=int(params["cell"]["slots"]))
            assert _cb_trades(st) == _as_cb(_flat(reference_walk(real))), seed

    def test_incoming_iceberg_sweep_priced_by_the_resting_iceberg(self, payload):
        """slow: 100000244 -- sell ICE 34/6 @1005194, buy ICE 62/31 @1005245 (Rule 2 items
        (2)-(4), Rule 3 *ICE aggressor*; re-authored for r4). Every row prices at the
        resting iceberg's limit; rows end at each resting slice and at the incoming
        instance boundary 31 (the last slice 4 prints 1 + 3). M = 34, k_e = 1, u = 3,
        R = 28: shown v_1 = min(31 - 3, 28) = 28 at b + k_e d = b (d = 0), arrival stamp."""
        tape, _ = _load_fixture(payload, 100000244)
        real = _real(tape)
        assert [int(real[1][c]) for c in (C_SIDE, C_PRICE, C_QTY, C_EXEC, C_PEAK, C_PPD)] == \
            [BUY, 1005245, 62, ICE, 31, 0]
        st, _ = _cb_run(real)
        assert _cb_trades(st) == [(1005194, q, frozenset((3, 5))) for q in (6, 6, 6, 6, 6, 1, 3)]
        bids = _cb_side(st, BUY)
        assert {o: v[:2] for o, v in bids.items()} == {5: (1005245, 28)}
        assert bids[5][2] == int(real[1][C_TS])                             # arrival stamp kept
        assert _cb_side(st, SELL) == {}
        assert replay_errors(real) == []

    def test_ioc_takes_the_better_iceberg_slices_first(self, payload):
        """slow: 100000246 -- two buy icebergs, sell IOC 168 sweeps (§6.1, Rule 2 re-ranking)."""
        tape, _ = _load_fixture(payload, 100000246)
        real = _real(tape)
        st, _ = _cb_run(real, book_depth=60)
        tr = _cb_trades(st)
        assert [p for p, _, _ in tr] == sorted((p for p, _, _ in tr), reverse=True)
        assert sum(q for _, q, _ in tr) == 90 + 64
        assert _cb_side(st, BUY) == {} and _cb_side(st, SELL) == {}     # IOC remainder 14 cancelled

    def test_fok_fills_from_the_iceberg_slices_before_the_worse_level(self, payload):
        """slow: 100000249 -- FOK 45 vs sell ICE 87/21 @1004403 and NON 48 @1004456 (Rule 1/2)."""
        tape, _ = _load_fixture(payload, 100000249)
        real = _real(tape)
        st, _ = _cb_run(real, max_icebergs=64)
        assert _cb_trades(st) == [(1004403, q, frozenset((2, 17))) for q in (21, 21, 3)]
        asks = _cb_side(st, SELL)
        assert asks[7][:2] == (1004456, 48)
        assert asks[2][:2] == (1004403, 18)

    def test_fixture_replay_identical_eager_and_jitted(self, payload):
        """slow: 100000244 under jax.disable_jit vs jitted."""
        import jax
        tape, _ = _load_fixture(payload, 100000244)
        real = _real(tape)
        st_j, _ = _cb_run(real)
        with jax.disable_jit():
            st_e, _ = _cb_run(real)
        assert _cb_trades(st_e) == _cb_trades(st_j)
        assert _cb_side(st_e, BUY) == _cb_side(st_j, BUY)


# ---------------------------------------------------------------------------
# Engine re-judgement on the r4 draws (the PASS file carries no engine trades)
# ---------------------------------------------------------------------------
_FORMER_D4B = (100000254, 100001177, 100001190, 100001489)   # r2's genuine deviations
# Under r4, 100001177's incoming ICEwPI sweeps stop at an instance limit (item (6)), so
# the session takes another path and its D4b site is gone (the superseded reading still
# has it). It stays replayed as an incoming-iceberg draw. The other three keep the site.
_D4B_GONE_UNDER_R4 = (100001177,)
_FORMER_D4B_R4 = tuple(s for s in _FORMER_D4B if s not in _D4B_GONE_UNDER_R4)
_PRESSURES = [(d, m) for d in DEPTHS for m in SLOTS]
_N_SHARDS = 3


def _replay_draw(ix_, seed: int, **kw) -> List[str]:
    dd = ix_["draws"][seed]
    c = dd["cell"]
    rows = _real(regenerate(ix_, seed).tape)
    return replay_errors(rows, book_depth=int(c["depth"]), max_icebergs=int(c["slots"]),
                         recorded=(dd["n_trades"], dd["volume"]), **kw)


def _d4b_draws(ix_) -> List[int]:
    """Every emitted draw whose r4 walk holds the D4b pattern (from the walk scan)."""
    return sorted(s for s, v in _scan(ix_).items() if v["d4b"])


def _at(ix_, seeds, depth, slots) -> List[int]:
    return [s for s in seeds if (ix_["draws"][s]["cell"]["depth"], ix_["draws"][s]["cell"]["slots"]) == (depth, slots)]


@pytest.mark.slow
@pytest.mark.jit
class TestEngineReplay:
    """The engine itself, judged by the twin's r4 walk after every submission."""

    @pytest.mark.parametrize("depth,slots", _PRESSURES)
    def test_first_draw_of_every_cell_replays_clean(self, ix, depth, slots):
        """slow: one compile per pressure config + ~120 sessions (~40 s per config)."""
        seeds = sorted(c["draws"][0]["seed"] for c in ix["cells"].values()
                       if c["draws"] and (c["cell"]["depth"], c["cell"]["slots"]) == (depth, slots))
        assert seeds, (depth, slots)
        bad = {s: e[:2] for s in seeds for e in [_replay_draw(ix, s)] if e}
        assert bad == {}

    @pytest.mark.parametrize("shard", range(_N_SHARDS))
    @pytest.mark.parametrize("depth,slots", _PRESSURES)
    def test_every_incoming_iceberg_draw_replays_clean(self, ix, depth, slots, shard):
        """slow: the walk scan (~30 s), then every draw with an incoming iceberg that
        crosses on entry, at this pressure config (one shard of three), replayed
        through ContractBook and judged after every submission against the r4 walk
        (instance limits and stop, row split, continuation at k_e, display v_1,
        arrival stamp)."""
        seeds = _at(ix, _chain_draws(ix), depth, slots)[shard::_N_SHARDS]
        assert seeds, (depth, slots, shard)
        bad = {s: e[:2] for s in seeds for e in [_replay_draw(ix, s)] if e}
        assert bad == {}

    def test_incoming_iceberg_replays_cover_every_amended_feature(self, ix):
        """slow: the walk scan. Non-null: the draws replayed above hold every amended-rule
        feature, at every pressure config."""
        sc = _scan(ix)
        for depth, slots in _PRESSURES:
            tot: Counter = Counter()
            for s in _at(ix, _chain_draws(ix), depth, slots):
                tot.update(sc[s]["chain"])
            for k in ("crossing", "crossing_d", "stopped_by_instance", "boundary_cut_d0",
                      "rest_part_used", "continued"):
                assert tot[k] >= 1, (depth, slots, k, dict(tot))

    @pytest.mark.parametrize("depth,slots", _PRESSURES)
    def test_every_d4b_pattern_draw_replays_clean(self, ix, depth, slots):
        """slow: the walk scan (~30 s), then the D4b-pattern draws at this pressure
        config through ContractBook (~16 s compile)."""
        seeds = _at(ix, _d4b_draws(ix), depth, slots)
        bad = {s: e[:2] for s in seeds for e in [_replay_draw(ix, s)] if e}
        assert bad == {}
        for s in seeds:
            assert ix["draws"][s]["W"] == "p" and "f" not in ix["draws"][s]["status"].values(), s

    def test_d4b_pattern_draws_cover_the_r2_deviations(self, ix):
        """slow: the walk scan. Non-null: the class the replays above judge is not
        empty and contains every r2 deviation."""
        found = _d4b_draws(ix)
        assert set(_FORMER_D4B_R4) <= set(found), sorted(set(_FORMER_D4B_R4) - set(found))
        sc = _scan(ix)
        for s in _D4B_GONE_UNDER_R4:
            rows = _real(regenerate(ix, s).tape)
            tr: list = []
            reference_walk(rows, trace=tr, incoming=OLD)
            assert s not in found and any(t["d4b"] for t in tr), s
            assert sc[s]["chain"].get("stopped_by_instance", 0) >= 1, s
            assert s in _chain_draws(ix)                     # still replayed above

    def test_superseded_rule_is_caught_by_the_engine_replay(self, ix):
        """slow: planted-defect twin of the incoming-iceberg replays. Judged by the
        superseded reading (``OLD``), the engine is rejected on the first emitted draws
        where the two readings differ, for d = 0 and for d != 0."""
        sc = _scan(ix)
        rec = {s: (dd["n_trades"], dd["volume"]) for s, dd in ix["draws"].items()}
        diff = [s for s in _chain_draws(ix) if sc[s]["old"] != rec[s]]
        pick = [next(s for s in diff if sc[s]["chain"].get("crossing_d", 0) == 0),
                next(s for s in diff if sc[s]["chain"].get("stopped_by_instance", 0) >= 1)]
        for s in pick:
            errs = _replay_draw(ix, s, incoming=OLD)
            assert errs and any("trades differ" in e or "queue" in e for e in errs), (s, errs)


# Engine probes on the amended rule, written here from the spec (tape units).
_GESTVD2 = [_row(11, BUY, _px(112.36), 10, NON, ts=1), _row(12, BUY, _px(112.34), 205, NON, ts=1),
            _row(13, BUY, _px(112.00), 119, NON, ts=2), _row(14, BUY, _px(110.00), 100, NON, ts=3),
            _row(15, SELL, _px(110.00), 1000, ICE, peak=50, ppd=75, ts=5),
            _row(16, BUY, _px(114.00), 70, NON, ts=9)]


@pytest.mark.slow
@pytest.mark.jit
class TestEngineIncomingChainProbes:
    """Rule 2 *Incoming iceberg* items (2), (4), (6) through ContractBook (one compile
    + a few submits each), judged after every submission and pinned explicitly."""

    def test_gestvd2_replica(self):
        """slow: item (6) stop at instance 4, item (4) continuation at 113.00, 113.75."""
        st, sts = _cb_run(_GESTVD2)
        tr = _cb_trades(sts[4])
        assert tr == [(_px(112.36), 10, frozenset((11, 15)))] + [
            (_px(112.34), q, frozenset((12, 15))) for q in (40, 50, 50, 50)]
        asks = _cb_side(sts[4], SELL)
        assert asks == {15: (_px(113.00), 50, 5, asks[15][3])}            # arrival stamp, second 5
        assert {o: v[:2] for o, v in _cb_side(sts[4], BUY).items()} == {
            12: (_px(112.34), 15), 13: (_px(112.00), 119), 14: (_px(110.00), 100)}
        assert _cb_trades(st)[-2:] == [(_px(113.00), 50, frozenset((15, 16))),
                                       (_px(113.75), 20, frozenset((15, 16)))]
        assert _cb_side(st, SELL)[15][:2] == (_px(113.75), 30)
        assert replay_errors(_GESTVD2) == []

    def test_part_used_icewpi_instance_buy_side(self):
        """slow: BUY ICE 51.00, 10.0, P = 2.0, d = -0.50 vs SELL 3.0 @ 50.00; the remainder
        shows 1.0 @ 50.50; SELL 7.0 @ 49.00 prints 1.0 @ 50.50, 2.0 @ 50.00, 49.50, 49.00."""
        rows = [_row(3141, SELL, _px(50.0), 30, NON, ts=4), _row(26, BUY, _px(51.0), 100, ICE, peak=20, ppd=-50, ts=4),
                _row(535897, SELL, _px(49.0), 70, NON, ts=6)]
        st, sts = _cb_run(rows)
        assert _cb_trades(sts[1]) == [(_px(50.0), q, frozenset((3141, 26))) for q in (20, 10)]
        assert _cb_side(sts[1], BUY)[26][:3] == (_px(50.5), 10, 4)
        assert [(p, q) for p, q, _ in _cb_trades(st)[2:]] == [(_px(50.5), 10), (_px(50.0), 20),
                                                               (_px(49.5), 20), (_px(49.0), 20)]
        assert _cb_side(st, BUY) == {} and _cb_side(st, SELL) == {}
        assert replay_errors(rows) == []

    def test_stop_with_base_crossing_liquidity_left_sell_side(self):
        """slow: the mirrored stop, off-round prices, d = +13 ticks."""
        b = 1004971
        rows = [_row(40404, BUY, 1004990, 12, NON, ts=1), _row(777, BUY, 1004975, 9, NON, ts=2),
                _row(31337, SELL, b, 40, ICE, peak=7, ppd=13, ts=3)]
        st, _ = _cb_run(rows)
        assert _cb_trades(st) == [(1004990, 7, frozenset((40404, 31337))), (1004990, 5, frozenset((40404, 31337)))]
        assert _cb_side(st, SELL)[31337][:3] == (b + 13, 2, 3)
        assert {o: v[:2] for o, v in _cb_side(st, BUY).items()} == {777: (1004975, 9)}
        assert replay_errors(rows) == []

    def test_standard_iceberg_split_and_part_used_display(self):
        """slow: d = 0. BUY ICE 51.00, 20.0, P = 4.0 vs SELL 6.0 @ 50.00 and 6.0 @ 50.50
        -> 4.0, 2.0 | 2.0, 4.0; then BUY ICE 50.70, 10.0, P = 4.0 vs SELL 3.0 @ 50.60 ->
        3.0; the remainder 7.0 shows v_1 = 1.0, and a SELL 2.0 @ 50.70 prints 1.0 + 1.0
        (the remainder's next slice is activated)."""
        rows = [_row(1, SELL, _px(50.0), 60, NON, ts=1), _row(2, SELL, _px(50.5), 60, NON, ts=1),
                _row(3, BUY, _px(51.0), 200, ICE, peak=40, ts=2)]
        st, _ = _cb_run(rows)
        assert [(p, q) for p, q, _ in _cb_trades(st)] == [(_px(50.0), 40), (_px(50.0), 20),
                                                          (_px(50.5), 20), (_px(50.5), 40)]
        assert _cb_side(st, BUY)[3][:3] == (_px(51.0), 40, 2)
        assert replay_errors(rows) == []
        probe = [_row(7, SELL, _px(50.6), 30, NON, ts=1), _row(8, BUY, _px(50.7), 100, ICE, peak=40, ts=2),
                 _row(9, SELL, _px(50.7), 20, NON, ts=3)]
        st, sts = _cb_run(probe)
        assert [(p, q) for p, q, _ in _cb_trades(sts[1])] == [(_px(50.6), 30)]
        assert _cb_side(sts[1], BUY)[8][:3] == (_px(50.7), 10, 2)        # v_1 = min(40 - 30, 70)
        assert [(p, q) for p, q, _ in _cb_trades(st)[1:]] == [(_px(50.7), 10), (_px(50.7), 10)]
        assert _cb_side(st, BUY)[8][:2] == (_px(50.7), 30)
        assert replay_errors(probe) == []


# A minimal reproducer of r2's D4b deviation (four rows, written here).
_STAMP_ROWS = [_row(11, SELL, 1005000, 40, ICE, peak=2, ts=1), _row(22, BUY, 1005000, 21, NON, ts=2),
               _row(33, SELL, 1005000, 15, NON, ts=3), _row(44, BUY, 1005000, 1, NON, ts=4)]
# The same pattern mirrored (iceberg on the BUY side), 1- to 6-digit OIDs, an off-round
# price, and the later arrival in the *same* tape second as the activation (NSEC decides).
_STAMP_ROWS_MIRROR = [_row(123456, BUY, 1004987, 40, ICE, peak=2, ts=7),
                      _row(7, SELL, 1004987, 21, NON, ts=9),
                      _row(98765, BUY, 1004987, 15, NON, ts=9),
                      _row(4321, SELL, 1004980, 1, NON, ts=12),
                      _row(55555, SELL, 1004983, 1, NON, ts=12)]


def _key(v) -> Tuple[int, int]:
    return v[2], v[3]


@pytest.mark.slow
@pytest.mark.jit
class TestRegenStampDeviation:
    """r2's D4b (a regenerated slice stored ahead of the tape clock, so a later
    same-price arrival was served first), re-judged on the fixed engine (E2)."""

    def test_full_draw_100001190_engine_matches_r4_walk_and_record(self, ix):
        """slow: 32-row r2 deviation draw through ContractBook (~16 s compile)."""
        dd, c = ix["draws"][100001190], ix["draws"][100001190]["cell"]
        assert dd["W"] == "p" and "f" not in dd["status"].values()
        rows = _real(regenerate(ix, 100001190).tape)
        assert replay_errors(rows, book_depth=int(c["depth"]), max_icebergs=int(c["slots"]),
                             recorded=(dd["n_trades"], dd["volume"])) == []
        sites = d4b_sites(rows)
        assert sites, "100001190 no longer holds the D4b pattern"
        _, sts = _cb_run(rows, book_depth=int(c["depth"]), max_icebergs=int(c["slots"]))
        row_of = {int(r[C_OID]): j for j, r in enumerate(rows)}
        n_cmp = 0
        for t in sites:
            side = int(rows[row_of[t["cp"]]][C_SIDE])
            pre = _cb_side(sts[t["row"] - 1], side)
            early, a = pre[t["cp"]], t["stamped_at_row"]
            # E2: stored at its activation event (the activating order's tape second) ...
            assert early[2] == int(rows[a][C_TS])
            for late in t["behind"]:
                if not (a < row_of[late] < t["row"] and late in pre):
                    continue
                assert pre[late][0] == early[0]                                   # same price
                if int(rows[row_of[late]][C_EXEC]) == NON:
                    assert pre[late][2] == int(rows[row_of[late]][C_TS])          # its tape second
                # ... so it ranks ahead of every later same-price arrival.
                assert _key(early) < _key(pre[late])
                n_cmp += 1
        assert n_cmp >= 1

    def test_former_deviation_draws_replay_clean(self, ix):
        """slow: the four r2 D4b draws, each replayed and judged after every submission
        (the three that keep the D4b site under r4, and 100001177, see _D4B_GONE_UNDER_R4)."""
        for s in _FORMER_D4B:
            dd = ix["draws"][s]
            assert dd["W"] == "p" and "f" not in dd["status"].values(), s
            assert bool(d4b_sites(_real(regenerate(ix, s).tape))) == (s in _FORMER_D4B_R4), s
            assert _replay_draw(ix, s) == [], s

    def test_classifier_flags_a_planted_d4b_on_each_former_deviation_draw(self, ix):
        """Non-null twin of the clean replays: serving the later arrival at the site
        (the r2 engine's choice) is classified REGEN_STAMP_AHEAD, never AGREE."""
        for s in _FORMER_D4B_R4:
            rows = _real(regenerate(ix, s).tape)
            row_of = {int(r[C_OID]): j for j, r in enumerate(rows)}
            t = d4b_sites(rows)[0]
            late = next(o for o in t["behind"] if t["stamped_at_row"] < row_of[o] < t["row"])
            mine = _flat(reference_walk(rows))
            f = list(mine[t["at"]])
            pos = 1 if f[0] == f[4] else 0          # the counterparty column
            f[pos] = late
            cls, info = classify(rows, mine[:t["at"]] + [f] + mine[t["at"] + 1:])
            assert cls == REGEN_STAMP_AHEAD and info["cp_engine"] == late, (s, cls, info)

    def test_probe_slice_stored_at_its_activation_event_not_ahead_of_the_tape(self):
        """slow: one compile + 4 submits. Rewritten from the r2 pin of the defect (the slice
        stored with a counter SEC above the later arrival's 3): E2 stores the slice at
        its activation event, tape second 2, ahead of SELL 33 (second 3)."""
        _, sts = _cb_run(_STAMP_ROWS)
        asks1 = _cb_side(sts[1], SELL)
        assert asks1[11][:3] == (1005000, 1, 2)          # activated by BUY 22 at second 2
        asks = _cb_side(sts[2], SELL)
        assert asks[11][:3] == (1005000, 1, 2) and asks[33][:3] == (1005000, 15, 3)
        assert _key(asks[11]) < _key(asks[33])
        assert [o for o, _, _ in _cb_queue(sts[2], SELL)] == [11, 33]
        for st, sec in zip(sts, (1, 2, 3, 4)):
            stamps = [_key(v) for s in (BUY, SELL) for v in _cb_side(st, s).values()]
            assert len(set(stamps)) == len(stamps) and all(x <= sec for x, _ in stamps)

    def test_probe_v2_14_slice_activated_earlier_trades_first(self):
        """slow: one compile + 5 submits. Was a strict xfail on r2; passes after E2. The
        slice activated at second 2 trades before SELL 33 (second 3); the next slice,
        activated at second 4 by BUY 44, queues behind SELL 33."""
        st, _ = _cb_run(_STAMP_ROWS)
        want = _as_cb(_flat(reference_walk(_STAMP_ROWS)))
        assert want[-1] == (1005000, 1, frozenset((11, 44)))
        assert _cb_trades(st) == want
        rows = _STAMP_ROWS + [_row(55, BUY, 1005000, 1, NON, ts=5)]
        st, sts = _cb_run(rows)
        assert _cb_trades(st)[-2:] == [(1005000, 1, frozenset((11, 44))), (1005000, 1, frozenset((33, 55)))]
        asks = _cb_side(sts[3], SELL)
        assert asks[11][:3] == (1005000, 2, 4) and _key(asks[33]) < _key(asks[11])
        assert replay_errors(rows) == []

    def test_probe_mirrored_same_second_arrival(self):
        """slow: one compile + 5 submits. BUY-side iceberg, large/small OIDs, off-round
        price; BUY 98765 arrives in the activation's own second (9): the slice, activated
        first in processing order, still ranks ahead (NSEC)."""
        rows = _STAMP_ROWS_MIRROR
        st, sts = _cb_run(rows)
        bids = _cb_side(sts[2], BUY)
        assert bids[123456][:3] == (1004987, 1, 9) and bids[98765][:3] == (1004987, 15, 9)
        assert bids[123456][3] < bids[98765][3]
        assert _cb_trades(st)[-2:] == [(1004987, 1, frozenset((123456, 4321))),
                                       (1004987, 1, frozenset((98765, 55555)))]
        assert replay_errors(rows) == []

    @pytest.mark.parametrize("ex", [NON, ICE])
    def test_probe_own_remainder_precedes_the_slice_its_process_activated(self, ex):
        """slow: one compile + 2 submits. A resting SELL ICEwPI (drift +7 ticks) loses its
        first slice to an aggressor at second 2 whose remainder rests (the next slice
        drifts above its limit). Both rest from one process: the remainder keeps its entry
        stamp and precedes the slice on the per-contract key (§5.2; Rule 2 (4); E2). For
        the ICE aggressor (P = 3, d = 0, r4 item (4)): M = 2, k_e = 0, u = 2, so the
        remainder 7 shows v_1 = min(3 - 2, 7) = 1, not min(P, R) = 3."""
        rows = [_row(31, SELL, 1005000, 40, ICE, peak=2, ppd=7, ts=1),
                _row(2046, BUY, 1005003, 9, ex, peak=3 if ex == ICE else 0, ts=2)]
        st, _ = _cb_run(rows)
        assert _cb_trades(st) == [(1005000, 2, frozenset((31, 2046)))]
        bid, ask = _cb_side(st, BUY)[2046], _cb_side(st, SELL)[31]
        assert bid[:3] == (1005003, 7 if ex == NON else 1, 2) and ask[:3] == (1005007, 2, 2)
        assert _key(bid) < _key(ask)
        assert replay_errors(rows) == []

    def test_d4b_fixture_seed100000254(self, payload):
        """slow: the 4-row smallest case of OMIE-SLICE-STAMP-TAPE-SEC-1 E2: SELL ICE 24
        (P = 5, d = 0) trades 2 @ oid 5 (its slice, activated by FOK 8 at second 8), then
        21 @ oid 13, split at the incoming instance boundaries 5, 10, 15, 20 (r4 item (2)):
        3, 5, 5, 5, 3. oid 5's next slice (3), activated at second 20, ranks behind 13."""
        name = "C15_r2_slice_stamp_seed100000254.npz"
        path = next((p for p in (_results_dir() / name, REPO_ROOT / "notebooks" / "certification" / "fixtures" / name,
                                 REPO_ROOT / "tests" / "omiecid" / "fixtures" / "cert" / name) if p.is_file()), None)
        if path is None:
            pytest.skip(f"{name} absent")
        z = np.load(path, allow_pickle=False)
        real = _real(np.asarray(z["tape"]))
        st, _ = _cb_run(real)
        assert _cb_trades(st) == ([(1006992, q, frozenset((5, 8))) for q in (16, 16, 16, 14)]
                                  + [(1006992, 2, frozenset((5, 24)))]
                                  + [(1006992, q, frozenset((13, 24))) for q in (3, 5, 5, 5, 3)])
        assert replay_errors(real) == []

        def swap_site(books):   # plant: the r2 queue order (13 ahead of 5's slice) at row 2
            q = books[2][BUY]
            i, j = [o for o, _, _ in q].index(5), [o for o, _, _ in q].index(13)
            q[i], q[j] = q[j], q[i]

        assert any("queue" in e for e in replay_errors(real, tamper=swap_site))
