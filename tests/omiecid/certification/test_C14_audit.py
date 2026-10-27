"""K4 audit twin for OMIE-CERT-C14-1 (capacity edges, one part per ceiling) -- r2.

Re-authored for the ``OMIE-CERT-RERUN-1`` re-run (r2). Fresh-context, adversarial
audit. Every expectation below was worked out again from the cited documents:

* the card (``agile-plan.md`` §OMIE-CERT-C14-1), the RERUN-1 C14 correction text
  ("rebuild a′ (full only at regeneration); part c targets the 100-pass bound;
  correct finding F4 (the engine reports the true burst quantity); record the full
  outcome; the last-row lookup probes become passing controls (S2)") and the
  entry-15 verdict rationale;
* ``scenario-ladder.md`` v2.4 §C14 and §The capacity-validity precondition (the
  pass ceiling is ``_TRIP_BOUND`` = 100 passes; ``max_match_iterations`` = 200 is
  inert; the slice ceiling: ``truncated`` at registration iff ``ceil(T/p) >= 66``);
* ``book-depth-cap-finding.md`` correction 1 and **correction 1a** (the wipe runs
  after matching on the incoming order's own side and fires on any dispatch
  against a full own side, even one that executes completely; NON/IOC passes
  and an incoming iceberg's sweep are flag sites; regenerations are not);
* ``OMIE-JAXOB-LOOKUP-MISS-1`` (S2: a lookup miss is a miss -- a side whose last
  row is occupied loses nothing it should not, its icebergs regenerate);
* ``OMIE-ICE-PRICE-PRIORITY-1`` (S3) and ``market-spec.md`` v2.14 §6.1, §4.2,
  §4.2.1 Rule 2 and its v2.14 re-ranking clarification (the aggressor never trades
  a worse level while a better iceberg's next slice can still trade), and the
  v2.14 incoming-iceberg clarification (an incoming ICE aggresses with its total);
* OQ-15 (a saturated ring writes every later trade to its last row), the
  ``OMIE-ICE-INCOMING-FILL-REPORT-1`` close-out record (reports add up pass by
  pass; an overflowed iceberg table reports the peak only and sets ``truncated``),
  ``jax_constants.py`` §B1 (slot overflow: meta unchanged, sticky ``truncated``)
  and §B2 (``MAX_SLICES_PER_REGEN = 64``).

It asserts against the **emitted** per-draw records of ``C14.json``, whatever its
status. Independence rules 3/4: it never imports or executes the notebook's
expectation code (sections 1-5 and 7); the notebook's section-4 dataset table
(markdown) was read for the construction shapes only. Independent engine probes
build their own tapes and drive the engine through ``ContractBook``'s public API
(``init`` / ``submit_order`` / ``cancel_order``) or the reference
``build_record``; frozen r1 counterexample fixtures are replayed as passing
controls (S2/S3).

Results JSON: ``$OMIE_CERT_RESULTS_DIR/C14.json`` else
``notebooks/certification/results/C14.json``; absent -> skip.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from notebooks.certification.cert_support import derive_failure_kinds, validate_results

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS = REPO_ROOT / "docs" / "omiecid"
FIXTURES = REPO_ROOT / "notebooks" / "certification" / "fixtures"
CERT_ID = "C14"
SPEC_VERSION = "v2.14"
LADDER_VERSION = "v2.5"

PARTS = ("C14a", "C14a′", "C14b", "C14c", "C14d", "C14e-65", "C14e-66",
         "C14c-below", "C14-LR-NON", "C14-LR-ICE")
SEED_BLOCK = {"C14a": 0, "C14a′": 100000, "C14b": 200000, "C14c": 300000, "C14d": 400000,
              "C14e-65": 500000, "C14e-66": 600000, "C14c-below": 700000,
              "C14-LR-NON": 800000, "C14-LR-ICE": 900000}
DRAWS_PER_PART = 3
REQUIRED_KINDS = {
    "C14a": {"outcome", "flags", "detail", "battery", "I8"},
    "C14a′": {"outcome", "flags", "detail", "battery", "I8"},
    "C14b": {"outcome", "flags", "detail", "battery", "I1", "I2", "I6", "I8"},
    "C14c": {"outcome", "flags", "walk", "full", "battery", "I3", "I8"},
    "C14d": {"outcome", "flags", "detail", "battery", "I2", "I8"},
    "C14e-65": {"outcome", "flags", "battery", "I8"},
    "C14e-66": {"outcome", "flags", "battery", "I8"},
    "C14c-below": {"outcome", "flags", "battery", "I8"},
    "C14-LR-NON": {"outcome", "flags", "detail", "battery", "I8"},
    "C14-LR-ICE": {"outcome", "flags", "detail", "battery", "I8"},
}
# The full outcome every draw records (RERUN-1: final_book, filled,
# hidden_remaining, reported_qty, still_resting). C14c keeps accepted / rejected /
# evicted / cancelled in `outcome` and the rest in its `full outcome` record.
OUTCOME_KEYS = ("accepted", "cancelled", "evicted", "filled", "fills", "final_book",
                "hidden_remaining", "rejected", "reported_qty", "still_resting")
C14C_FULL_KEYS = ("filled", "fills", "final_book", "hidden_remaining", "reported_qty", "still_resting")
BATTERY = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I9")
# Documented exclusions per part (re-derived; see TestExclusions for the two places
# this departs from the card's literal list).
EXCLUDED = {p: {"I8"} for p in PARTS}
EXCLUDED["C14b"] |= {"I1", "I2", "I6"}
EXCLUDED["C14c"] |= {"I3"}
EXCLUDED["C14d"] |= {"I2"}
# I8 breach set each construction must produce (axis names as the run records them).
I8_BREACH = {
    "C14a": ({"book_full", "resting_per_side"}, {"book_full", "resting_per_side"}),
    "C14a′": ({"resting_per_side"}, {"resting_per_side"}),
    "C14b": ({"trade_buffer_saturated", "trades"}, {"trade_buffer_saturated", "trades"}),
    "C14c": ({"truncated"}, {"truncated", "trades"}),
    "C14c-below": (set(), {"trades"}),
    "C14d": ({"truncated"}, {"truncated"}),
    "C14e-65": ({"slices"}, {"slices"}),
    "C14e-66": ({"slices", "truncated"}, {"slices", "truncated"}),
    "C14-LR-NON": ({"book_full", "resting_per_side"}, {"book_full", "resting_per_side"}),
    "C14-LR-ICE": ({"resting_per_side"}, {"resting_per_side"}),
}
BOOK_DEPTH, N_TRADES, MAX_ITER, MAX_ICE, MAX_SLICES = 100, 100, 200, 1024, 64
PASS_BOUND = 100           # _TRIP_BOUND (scenario-ladder v2.3+/v2.4, OMIE-CAPACITY-DOCS-1)
HOP_MARGIN = 160

_PER_DRAW_RE = re.compile(
    r"^(C14(?:a′|a|b|c-below|c|d|e-65|e-66)|C14-LR-NON|C14-LR-ICE) seed (\d+)"
    r"(?: \(recorded, not certified\))?: (.+)$")


# ---------------------------------------------------------------------------
# Loading and indexing
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
        pytest.skip(f"C14 results JSON absent: {p} (run the C14 notebook, section 8, first)")
    return _load(p)


def _roundtrip(tmp_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    p = tmp_path / f"{CERT_ID}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return _load(p)


def _kind(tail: str) -> Optional[str]:
    if tail == "engine outcome":
        return "outcome"
    for prefix, k in (("flags", "flags"), ("part detail", "detail"), ("walk", "walk"),
                      ("full outcome", "full"), ("battery", "battery")):
        if tail.startswith(prefix):
            return k
    m = re.match(r"(I\d) excluded", tail)
    return m.group(1) if m else None


def _obs(a: Dict[str, Any]) -> Any:
    v = a["observed"]
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def index(payload) -> Tuple[Dict[int, Dict[str, Dict[str, Any]]], Dict[int, str]]:
    idx: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    part_of: Dict[int, str] = {}
    for a in payload.get("assertions", []):
        m = _PER_DRAW_RE.match(a.get("name", ""))
        if not m:
            continue
        part, seed, tail = m.group(1), int(m.group(2)), m.group(3)
        k = _kind(tail)
        if k is not None:
            idx[seed][k] = a
            part_of[seed] = part
    return idx, part_of


def seeds_of(part_of: Dict[int, str], part: str) -> List[int]:
    return sorted(s for s, p in part_of.items() if p == part)


def _o(idx, seed, kind):
    a = idx.get(seed, {}).get(kind)
    return None if a is None else _obs(a)


def _ceil(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _ints(d: Dict[str, Any]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def _resting(o) -> Dict[int, int]:
    return {k: v for k, v in _ints(o["still_resting"]).items() if v > 0}


def _book(o) -> Dict[int, Tuple[str, int, int]]:
    """oid -> (side, price, qty) from the recorded final book."""
    return {int(e[2]): (side, int(e[0]), int(e[1]))
            for side in ("BUY", "SELL") for e in o["final_book"].get(side, [])}


def _fills_by_sub(o) -> Dict[int, List[List[int]]]:
    out: Dict[int, List[List[int]]] = defaultdict(list)
    for f in o["fills"]:                     # [submission, ring row, price, qty, buyer, seller]
        out[int(f[0])].append([int(x) for x in f])
    return out


def _book_matches_resting(o) -> bool:
    return {k: v[2] for k, v in _book(o).items()} == _resting(o)


# ---------------------------------------------------------------------------
# Judges: each returns a list of problems ([] == the documented behaviour holds)
# ---------------------------------------------------------------------------
def judge_envelope(payload) -> List[str]:
    out = [f"schema: {e}" for e in validate_results(payload)]
    if payload.get("cert_id") != CERT_ID:
        out.append(f"cert_id {payload.get('cert_id')!r}")
    if payload.get("failure_kinds") != derive_failure_kinds(payload):
        out.append(f"failure_kinds {payload.get('failure_kinds')} != derived {derive_failure_kinds(payload)}")
    asserts = payload.get("assertions", [])
    any_fail = any(a.get("status") == "fail" for a in asserts)
    if any_fail != (payload.get("status") == "NOT CERTIFIED"):
        out.append(f"status {payload.get('status')!r} with failing assertions = {any_fail}")
    failing = sorted({a["seed"] for a in asserts if a.get("status") == "fail" and a.get("seed") is not None})
    if sorted(payload.get("seeds", {}).get("failing", [])) != failing:
        out.append(f"seeds.failing {payload.get('seeds', {}).get('failing')} != {failing}")
    d = payload.get("draws", {})
    n = len(PARTS) * DRAWS_PER_PART
    if (d.get("requested"), d.get("valid"), d.get("discarded")) != (n, n, 0):
        out.append(f"draws {d} != {n} requested / valid, 0 discarded")
    return out


def judge_quotes(payload) -> List[str]:
    from notebooks.certification.cert_support import SpecQuote, check_quote, quote_digest

    out = []
    m = re.search(r"^\*\*Status:\*\*\s*(v\d+(?:\.\d+)*)", (DOCS / "market-spec.md").read_text(encoding="utf-8"), re.M)
    ms = m.group(1) if m else None
    if ms != SPEC_VERSION:
        out.append(f"market-spec.md is {ms}; this r2 twin is stamped {SPEC_VERSION}")
    quotes = payload.get("spec", {}).get("quotes", [])
    if not quotes:
        out.append("no spec quotes recorded")
    for q in quotes:
        if q.get("ok") is not True or q.get("reason") != "ok":
            out.append(f"quote {q.get('source')} {q.get('section')}: {q.get('reason')!r}")
        if q.get("source") == "market-spec.md" and q.get("version") != SPEC_VERSION:
            out.append(f"quote {q.get('section')}: version {q.get('version')!r} != {SPEC_VERSION}")
    digests = {q.get("digest") for q in quotes}
    # The rules the independent probes below rest on must be among the quoted ones.
    for section, text in (("§6.1", "**Buy side:** ranked highest price first; ties broken by earliest timestamp."),
                          ("§4.2", "When a slice is exhausted, a new slice of peak size is activated and "
                                   "receives a new timestamp.")):
        sq = SpecQuote("market-spec.md", section, text)
        chk = check_quote(sq)
        if not chk.ok:
            out.append(f"required quote {section} does not match market-spec.md: {chk.reason}")
        if quote_digest(sq) not in digests:
            out.append(f"required quote {section} {text[:40]!r} absent from the emitted quotes")
    return out


def judge_eager_jit(payload) -> List[str]:
    ej = [a for a in payload.get("assertions", []) if a.get("kind") == "eager_jit"]
    if not ej:
        return ["no eager-vs-jitted evidence"]
    out = []
    if payload.get("eager_jit_identical") is not True:
        out.append(f"eager_jit_identical = {payload.get('eager_jit_identical')!r}")
    for a in ej:
        if a.get("status") != "pass" or a.get("observed") is not True:
            out.append(f"{a.get('name')}: eager != jitted")
    _, part_of = index(payload)
    covered = {part_of.get(a.get("seed")) for a in ej}
    for p in PARTS:
        if p not in covered:
            out.append(f"{p}: no eager-vs-jitted record")
    return out


def judge_per_draw_full(payload) -> List[str]:
    """Every draw of every part carries its values in full (parseable, all keys)."""
    idx, part_of = index(payload)
    out = []
    extra = set(part_of.values()) - set(PARTS)
    if extra:
        out.append(f"unexpected parts {sorted(extra)}")
    for p in PARTS:
        seeds = seeds_of(part_of, p)
        want = [SEED_BLOCK[p] + i for i in range(DRAWS_PER_PART)]
        if seeds != want:
            out.append(f"{p}: seeds {seeds} != {want}")
        for s in seeds:
            missing = REQUIRED_KINDS[p] - set(idx[s])
            if missing:
                out.append(f"{p} seed {s}: missing {sorted(missing)}")
            for k, a in idx[s].items():
                if not isinstance(_obs(a), dict):
                    out.append(f"{p} seed {s} {k}: observed not a full JSON object")
            o = _o(idx, s, "outcome")
            if not isinstance(o, dict):
                continue
            if p == "C14c":
                lack = [k for k in ("accepted", "cancelled", "evicted", "rejected") if k not in o]
                full = _o(idx, s, "full")
                lack += [f"full.{k}" for k in C14C_FULL_KEYS if not isinstance(full, dict) or k not in full]
            else:
                lack = [k for k in OUTCOME_KEYS if k not in o]
            if lack:
                out.append(f"{p} seed {s}: outcome lacks {lack}")
    return out


def judge_ceilings(payload) -> List[str]:
    from omiecid_simulator.omiecid.jax_constants import MAX_SLICES_PER_REGEN
    from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig

    m = OMIEMarketConfig()
    mine = [m.book_depth, m.nTrades, m.max_match_iterations, m.max_icebergs, MAX_SLICES_PER_REGEN]
    out = []
    if mine != [BOOK_DEPTH, N_TRADES, MAX_ITER, MAX_ICE, MAX_SLICES]:
        out.append(f"configured ceilings {mine} != documented")
    by_name = {a.get("name", ""): a for a in payload.get("assertions", []) if a.get("seed") is None}
    rec = [a for n, a in by_name.items() if n.startswith("the five configured ceilings")]
    if len(rec) != 1 or _obs(rec[0]) != mine or rec[0].get("status") != "pass":
        out.append(f"recorded ceilings {[_obs(a) for a in rec]} != {mine}")
    pb = [a for n, a in by_name.items() if n.startswith("the binding pass ceiling")]
    if len(pb) != 1 or _obs(pb[0]) != {"PASS_BOUND": PASS_BOUND, "max_match_iterations": MAX_ITER}:
        out.append(f"pass-bound record {[_obs(a) for a in pb]} != PASS_BOUND {PASS_BOUND} / 200")
    ov = _override(payload)
    if not (isinstance(ov.get("C14C_N_TRADES"), int) and ov["C14C_N_TRADES"] > N_TRADES):
        out.append(f"C14c ring override {ov.get('C14C_N_TRADES')!r} not above {N_TRADES}")
    cap = ov.get("C14D_MAX_ICEBERGS")
    if not (isinstance(cap, int) and 2 <= cap < MAX_ICE):
        out.append(f"C14d slot cap override {cap!r} not in [2, 1024)")
    return out


def _override(payload) -> Dict[str, Any]:
    for a in payload.get("assertions", []):
        if a.get("name", "").startswith("C14c/C14d configuration overrides"):
            o = _obs(a)
            return o if isinstance(o, dict) else {}
    return {}


# --- per part --------------------------------------------------------------
def judge_c14a(payload) -> List[str]:
    """The 101st resting add (correction 1): book_full at that submission and only
    there, the worst level wiped into the evicted bucket, no trade, no report, the
    newcomer rests (side 100 -> 100 - k + 1), evicted oids out of the book, I2 closes."""
    idx, part_of = index(payload)
    out = []
    for s in seeds_of(part_of, "C14a"):
        o, f, det, bat = (_o(idx, s, k) for k in ("outcome", "flags", "detail", "battery"))
        last = len(o["reported_qty"]) - 1
        if f.get("book_full_posts") != [last]:
            out.append(f"seed {s}: book_full_posts {f.get('book_full_posts')} != [{last}]")
        if f.get("truncated_posts") != [] or f.get("trade_buffer_saturated") is not False:
            out.append(f"seed {s}: unexpected truncated/saturation {f}")
        if o["fills"] or any(int(v) for v in o["filled"].values()) or any(o["reported_qty"]):
            out.append(f"seed {s}: a trade was manufactured {o['fills']} / reports {o['reported_qty']}")
        ev = _ints(o["evicted"])
        if not 1 <= len(ev) or any(v <= 0 for v in ev.values()):
            out.append(f"seed {s}: evicted bucket empty or non-positive {ev}")
        if set(ev) & set(_book(o)):
            out.append(f"seed {s}: evicted oid still in the book {set(ev) & set(_book(o))}")
        if not _book_matches_resting(o):
            out.append(f"seed {s}: still_resting != final book")
        if {int(a) for a in o["accepted"]} != set(_resting(o)) | set(ev):
            out.append(f"seed {s}: an accepted order is neither resting nor evicted (silent loss)")
        rows = det.get("full_side_rows_posts", [])
        if len(rows) < 2 or rows[-2] != BOOK_DEPTH or rows[-1] != BOOK_DEPTH - len(ev) + 1 \
                or max(rows) != BOOK_DEPTH:
            out.append(f"seed {s}: full-side rows {rows[-2:]} != [100, {BOOK_DEPTH - len(ev) + 1}]")
        if bat.get("I2") != "pass":
            out.append(f"seed {s}: I2 {bat.get('I2')} (evicted quantity not accounted)")
    return out


def judge_c14a_prime(payload) -> List[str]:
    """r2 a′ -- side S full only at regeneration (correction 1a: regenerations are not
    flag sites; every flag-site dispatch meets 99). Derived: posts 0-98 -> 1..99,
    post 99 (B on O) 99, post 100 100 (x trades p against B, its regenerated slice
    rests), post 101 100 (A consumes that slice; x regenerates into the row it
    freed). Two trades of p at b, no flag, nothing evicted, nothing else lost."""
    idx, part_of = index(payload)
    out = []
    for s in seeds_of(part_of, "C14a′"):
        o, f, det, bat = (_o(idx, s, k) for k in ("outcome", "flags", "detail", "battery"))
        if f.get("book_full_posts") or f.get("truncated_posts") or f.get("trade_buffer_saturated"):
            out.append(f"seed {s}: a flag was raised {f}")
        if o["evicted"] or o["cancelled"] or o["rejected"]:
            out.append(f"seed {s}: evicted {o['evicted']} cancelled {o['cancelled']} rejected {o['rejected']}")
        n = len(o["reported_qty"])
        fills = [[int(x) for x in r] for r in o["fills"]]
        if n != 102 or [r[:2] for r in fills] != [[100, 0], [101, 1]]:
            out.append(f"seed {s}: fills {fills} (expected one trade at post 100, row 0 and one at 101, row 1)")
            continue
        (_, _, b1, q1, by1, se1), (_, _, b2, q2, by2, se2) = fills
        x = ({by1, se1} & {by2, se2})
        if b1 != b2 or q1 != q2 or len(x) != 1:
            out.append(f"seed {s}: the two trades are not x's peak then x's regenerated slice at one price: {fills}")
            continue
        x = x.pop()
        p = q1
        counter = ({by1, se1} | {by2, se2}) - {x}
        rest = _resting(o)
        if counter & set(rest) or {int(a) for a in o["accepted"]} != set(rest) | counter:
            out.append(f"seed {s}: B/A {counter} still rest, or another order vanished")
        book = _book(o)
        hid = int(o["hidden_remaining"].get(str(x), 0))
        if x not in book or book[x][1] != b1 or not 0 < rest.get(x, 0) <= p or (hid and rest.get(x) != p):
            out.append(f"seed {s}: x={x} slice {book.get(x)} / hidden {hid} not min(p, T - 2p) at b")
        if set(_ints(o["hidden_remaining"])) - {x}:
            out.append(f"seed {s}: hidden quantity on a non-iceberg {o['hidden_remaining']}")
        sides = Counter(v[0] for v in book.values())
        if sorted(sides.values()) != [BOOK_DEPTH] or book[x][0] not in sides:
            out.append(f"seed {s}: final book sides {dict(sides)} (expected S alone at 100, x on S)")
        want_rows = list(range(1, 100)) + [99, 100, 100]
        if det.get("full_side_rows_posts") != want_rows:
            out.append(f"seed {s}: full-side rows {det.get('full_side_rows_posts')[-4:]} != [..., 99, 99, 100, 100]")
        if o["reported_qty"] != [0] * 100 + [p, p]:
            out.append(f"seed {s}: reports {o['reported_qty'][-3:]} != [0, {p}, {p}]")
        if bat.get("I2") != "pass":
            out.append(f"seed {s}: quantity lost (I2 {bat.get('I2')})")
    return out


def judge_c14b(payload) -> List[str]:
    """OQ-15 + corrected F4: the ring saturates (flag), a burst after saturation keeps
    only its final trade in the last row; the engine still reports the TRUE burst
    quantity T_b, so the collapse is detectable (T_b > r_b). Every pre-burst
    submission's report equals its recorded rows exactly."""
    idx, part_of = index(payload)
    out = []
    for s in seeds_of(part_of, "C14b"):
        o, f, det = (_o(idx, s, k) for k in ("outcome", "flags", "detail"))
        if f.get("trade_buffer_saturated") is not True:
            out.append(f"seed {s}: ring not flagged saturated")
        if f.get("book_full_posts") or f.get("truncated_posts"):
            out.append(f"seed {s}: unrelated flag {f}")
        by_sub = _fills_by_sub(o)
        last = len(o["reported_qty"]) - 1
        burst = by_sub.get(last, [])
        if len(burst) != 1 or burst[0][1] != N_TRADES - 1:
            out.append(f"seed {s}: burst submission {last} kept {burst}, documented one last-row trade")
            continue
        r_b, t_b = burst[0][3], int(o["reported_qty"][last])
        if not t_b > r_b:
            out.append(f"seed {s}: report {t_b} not above the kept row {r_b} (F4 corrected: true quantity)")
        for sub, rows in by_sub.items():
            if sub != last and sum(r[3] for r in rows) != int(o["reported_qty"][sub]):
                out.append(f"seed {s}: pre-burst submission {sub} report != its rows")
        ring = det.get("ring", [])
        if len(ring) != N_TRADES or ring[-1] != burst[0][2:]:
            out.append(f"seed {s}: ring size {len(ring)} / last row {ring[-1:]} != burst {burst[0][2:]}")
        else:
            for r in o["fills"]:
                if int(r[1]) < N_TRADES - 1 and ring[int(r[1])] != [int(x) for x in r[2:]]:
                    out.append(f"seed {s}: ring row {r[1]} != recorded fill {r}")
                    break
        used = det.get("rows_used_posts", [])
        if not used or used[-1] != N_TRADES or any(a > b for a, b in zip(used, used[1:])) or max(used) > N_TRADES \
                or used[-2] != N_TRADES:
            out.append(f"seed {s}: rows used {used} (must saturate before the burst and stay at 100)")
        i2, i6 = _o(idx, s, "I2"), _o(idx, s, "I6")
        parties = set(burst[0][4:6])
        v2 = i2.get("violations", [])
        if i2.get("status") != "fail" or {int(v[0]) for v in v2} != parties or \
                any(v[1] != "identity" or int(v[2]) - int(v[3]) != t_b - r_b for v in v2):
            out.append(f"seed {s}: I2 violations {v2} != the burst parties short by T_b - r_b = {t_b - r_b}")
        v6 = i6.get("violations", [])
        if i6.get("status") != "fail" or [last, "engine_reports", r_b, t_b] not in v6 or \
                any(int(v[0]) != last for v in v6):
            out.append(f"seed {s}: I6 violations {v6} (must be exactly on the burst, report {t_b} vs rows {r_b})")
    return out


def _c14c_icebergs(o_full, agg: int, agg_side: str):
    """Per resting iceberg: price, peak (largest fill), total T = filled + resting + hidden."""
    fills = [[int(x) for x in r] for r in o_full["fills"]]
    info: Dict[int, Dict[str, int]] = {}
    for r in fills:
        ice = r[4] if r[5] == agg else r[5]
        d = info.setdefault(ice, {"price": r[2], "peak": 0})
        d["peak"] = max(d["peak"], r[3])
        if d["price"] != r[2]:
            d["price_conflict"] = 1
    for ice, d in info.items():
        d["T"] = (int(o_full["filled"].get(str(ice), 0)) + int(o_full["still_resting"].get(str(ice), 0))
                  + int(o_full["hidden_remaining"].get(str(ice), 0)))
    best_first = sorted(info, key=lambda i: -info[i]["price"] if agg_side == "SELL" else info[i]["price"])
    return info, best_first


def _walk(info, order) -> List[Tuple[int, int, int]]:
    """§6.1 + §4.2 + Rule 2 (v2.14 re-ranking): drain each iceberg, slice by slice,
    before the next price level. (price, qty, iceberg)."""
    out = []
    for ice in order:
        T, p = info[ice]["T"], info[ice]["peak"]
        out += [(info[ice]["price"], p, ice)] * (T // p) + ([(info[ice]["price"], T % p, ice)] if T % p else [])
    return out


def judge_c14c(payload) -> List[str]:
    """100-pass bound (scenario-ladder v2.4): truncated only at the aggressor's post;
    the fills are the price-priority walk up to the cut, then at most one visible
    slice per later iceberg in price order (the post-loop re-submit); 100 <= #fills
    <= 100 + (n-1) + n; 0 < filled(agg) < Q; full outcome consistent; the Rule-4
    departure (I3 in substance) present as a consumed slot or a visible cross."""
    idx, part_of = index(payload)
    out = []
    for s in seeds_of(part_of, "C14c"):
        o, f, w, full, i3, i8 = (_o(idx, s, k) for k in ("outcome", "flags", "walk", "full", "I3", "I8"))
        n_sub = len(full["reported_qty"])
        agg_sub = n_sub - 1
        if f.get("truncated_posts") != [agg_sub] or f.get("book_full_posts") or f.get("trade_buffer_saturated"):
            out.append(f"seed {s}: flags {f} (documented: truncated only at the aggressor's post {agg_sub})")
        vals = i8.get("values", {})
        if not PASS_BOUND < int(vals.get("hops", 0)) <= HOP_MARGIN or int(vals.get("slices", 99)) > MAX_SLICES:
            out.append(f"seed {s}: construction outside the r2 target (hops {vals.get('hops')}, slices {vals.get('slices')})")
        fills = [[int(x) for x in r] for r in full["fills"]]
        if any(r[0] != agg_sub for r in fills) or [r[1] for r in fills] != list(range(len(fills))):
            out.append(f"seed {s}: fills not all on the aggressor's submission in ring order")
            continue
        common = set(fills[0][4:6]) if fills else set()
        for r in fills:
            common &= set(r[4:6])
        if len(common) != 1:
            out.append(f"seed {s}: no single aggressor across the fills")
            continue
        agg = common.pop()
        book = _book(full)
        if agg not in book:
            out.append(f"seed {s}: aggressor {agg} has no resting remainder (it should: 0 < filled < Q)")
            continue
        agg_side, agg_price, agg_rest = book[agg]
        info, order = _c14c_icebergs(full, agg, agg_side)
        if any("price_conflict" in d for d in info.values()) or len(info) != len(o["accepted"]) - 1:
            out.append(f"seed {s}: icebergs {sorted(info)} vs accepted {o['accepted']}")
            continue
        Q = sum(d["T"] for d in info.values())
        fa = int(full["filled"].get(str(agg), 0))
        if not 0 < fa < Q or fa + agg_rest != Q or str(agg) in full["hidden_remaining"]:
            out.append(f"seed {s}: aggressor filled {fa}, rests {agg_rest}, Q {Q}")
        if agg_price != info[order[-1]]["price"]:
            out.append(f"seed {s}: aggressor rests at {agg_price}, not at P4 {info[order[-1]]['price']}")
        E = _walk(info, order)
        got = [(r[2], r[3], r[4] if r[5] == agg else r[5]) for r in fills]
        cut = 0
        while cut < min(len(E), len(got)) and got[cut] == E[cut]:
            cut += 1
        tail = got[cut:]
        rank = {ice: i for i, ice in enumerate(order)}
        start = rank[E[cut][2]] if cut < len(E) else len(order)
        tr = [rank[t[2]] for t in tail]
        if tr != sorted(set(tr)) or (tr and tr[0] < start) or any(t[1] > info[t[2]]["peak"] for t in tail):
            out.append(f"seed {s}: fills leave the documented walk at {cut}: tail {tail[:4]}")
        nice = len(order)
        if not PASS_BOUND <= len(got) <= PASS_BOUND + (nice - 1) + nice:
            out.append(f"seed {s}: {len(got)} fills is not what a 100-pass cut allows ([100, {2 * nice + 99}])")
        if w.get("cut") is not None and not (cut >= int(w["cut"])):
            out.append(f"seed {s}: recorded cut {w.get('cut')} beyond the audit's prefix {cut}")
        per = Counter()
        for r in fills:
            per[r[4]] += r[3]
            per[r[5]] += r[3]
        if _ints(full["filled"]) != dict(per):
            out.append(f"seed {s}: filled != per-order sum of fills")
        if full["reported_qty"] != [0] * agg_sub + [fa]:
            out.append(f"seed {s}: reported {full['reported_qty']} != [0,...,{fa}]")
        if not _book_matches_resting(full):
            out.append(f"seed {s}: final book != still_resting")
        if any(_resting(full).get(i, 0) > info[i]["peak"] for i in info):
            out.append(f"seed {s}: an iceberg rests more than its peak")
        if o["evicted"] or o["cancelled"] or o["rejected"]:
            out.append(f"seed {s}: evicted/cancelled/rejected {o['evicted']} {o['cancelled']} {o['rejected']}")
        # Rule-4 departure (the excluded I3, asserted in substance): matchable hidden
        # quantity opposite the resting aggressor -- a consumed slot (hidden, no
        # visible slice) or a visible cross.
        crossing = [i for i in info if (info[i]["price"] >= agg_price if agg_side == "SELL"
                                        else info[i]["price"] <= agg_price)]
        consumed = [i for i in crossing if int(full["hidden_remaining"].get(str(i), 0)) > 0
                    and _resting(full).get(i, 0) == 0]
        visible = [i for i in crossing if _resting(full).get(i, 0) > 0]
        if not (consumed or visible):
            out.append(f"seed {s}: no Rule-4 departure in the final book")
        if (i3.get("i3_status") == "fail") != bool(visible):
            out.append(f"seed {s}: I3 {i3.get('i3_status')} but visibly crossed = {bool(visible)}")
    return out


def judge_c14c_below(payload) -> List[str]:
    """Below the 100-pass bound: no truncation; the aggressor takes the whole
    price-priority walk on its own submission in ring rows 0..; everything exhausts."""
    idx, part_of = index(payload)
    out = []
    for s in seeds_of(part_of, "C14c-below"):
        o, f, bat, i8 = (_o(idx, s, k) for k in ("outcome", "flags", "battery", "I8"))
        if f.get("truncated_posts") or f.get("book_full_posts") or f.get("trade_buffer_saturated"):
            out.append(f"seed {s}: flags {f}")
        if not int(i8.get("values", {}).get("hops", 999)) < PASS_BOUND:
            out.append(f"seed {s}: hop bound {i8.get('values', {}).get('hops')} not below {PASS_BOUND}")
        fills = [[int(x) for x in r] for r in o["fills"]]
        sub = len(o["reported_qty"]) - 1
        if not fills or any(r[0] != sub for r in fills) or [r[1] for r in fills] != list(range(len(fills))):
            out.append(f"seed {s}: fills not on the aggressor's submission in ring rows 0..")
            continue
        common = set(fills[0][4:6])
        for r in fills:
            common &= set(r[4:6])
        if len(common) != 1:
            out.append(f"seed {s}: no single aggressor")
            continue
        agg = common.pop()
        agg_side = "BUY" if fills[0][4] == agg else "SELL"
        info, order = _c14c_icebergs(o, agg, agg_side)
        got = [(r[2], r[3], r[4] if r[5] == agg else r[5]) for r in fills]
        if got != _walk(info, order):
            out.append(f"seed {s}: fills depart from the price-priority walk")
        Q = sum(d["T"] for d in info.values())
        if int(o["filled"].get(str(agg), 0)) != Q or o["reported_qty"] != [0] * sub + [Q]:
            out.append(f"seed {s}: aggressor not fully filled / reported ({o['filled'].get(str(agg))} vs {Q})")
        if _resting(o) or o["hidden_remaining"] or o["evicted"]:
            out.append(f"seed {s}: something left: {_resting(o)} {o['hidden_remaining']} {o['evicted']}")
        if bat.get("I3") != "pass" or bat.get("I2") != "pass":
            out.append(f"seed {s}: I2/I3 {bat.get('I2')}/{bat.get('I3')}")
    return out


def judge_c14d(payload) -> List[str]:
    """Slot overflow (§B1): truncated at the overflowing registration (sticky), meta
    unchanged, n_active plateaus at the cap; the newcomer rests its peak only with
    its hidden part untracked (I2 identity fails on it alone); survivors intact --
    a later aggressor still takes a survivor's regenerated slices (Rule 2)."""
    idx, part_of = index(payload)
    cap = _override(payload).get("C14D_MAX_ICEBERGS")
    if not isinstance(cap, int):
        return [f"C14d cap {cap!r}"]
    out = []
    for s in seeds_of(part_of, "C14d"):
        o, f, det, i2 = (_o(idx, s, k) for k in ("outcome", "flags", "detail", "I2"))
        posts = f.get("truncated_posts", [])
        n = len(o["reported_qty"])
        if posts != list(range(cap, n)):
            out.append(f"seed {s}: truncated posts {posts} != sticky from registration {cap}")
        if det.get("n_active_posts_through_overflow") != list(range(1, cap + 1)) + [cap]:
            out.append(f"seed {s}: n_active {det.get('n_active_posts_through_overflow')}")
        if det.get("meta_unchanged_at_overflow") is not True:
            out.append(f"seed {s}: side channel changed at overflow")
        viol = i2.get("violations", [])
        if i2.get("status") != "fail" or len(viol) != 1 or viol[0][1] != "identity":
            out.append(f"seed {s}: I2 {viol} (documented: exactly the overflowed iceberg)")
            continue
        oid, _, acc, got = int(viol[0][0]), viol[0][1], int(viol[0][2]), int(viol[0][3])
        rest = _resting(o).get(oid, 0)
        if str(oid) in o["hidden_remaining"] or rest != got or not 0 < rest < acc or oid in _ints(o["filled"]):
            out.append(f"seed {s}: overflowed {oid}: resting {rest}, accounted {got}, accepted {acc}")
        if len(o["hidden_remaining"]) > cap:
            out.append(f"seed {s}: {len(o['hidden_remaining'])} tracked icebergs > cap {cap}")
        by_sub = _fills_by_sub(o)
        later = by_sub.get(n - 1, [])
        parties = {x for r in later for x in r[4:6]}
        if len(later) < 2 or len(parties) != 2 or oid in parties or set(by_sub) != {n - 1}:
            out.append(f"seed {s}: the post-overflow aggressor did not take >= 2 slices of one survivor: {later}")
        if o["evicted"] or f.get("book_full_posts"):
            out.append(f"seed {s}: eviction/book_full at a slot overflow")
    return out


def judge_c14e(payload) -> List[str]:
    """65 slices: truncated False; 66: truncated at registration (post 0); never a
    trade; the book holds exactly the peak; ceil(total/peak) = 65 / 66; I2 closes."""
    idx, part_of = index(payload)
    out = []
    for part, n_sl, posts in (("C14e-65", 65, []), ("C14e-66", 66, [0])):
        for s in seeds_of(part_of, part):
            o, f, bat, i8 = (_o(idx, s, k) for k in ("outcome", "flags", "battery", "I8"))
            if f.get("truncated_posts") != posts or f.get("book_full_posts") or f.get("trade_buffer_saturated"):
                out.append(f"{part} seed {s}: flags {f}")
            if o["fills"] or len(o["accepted"]) != 1 or any(o["reported_qty"]):
                out.append(f"{part} seed {s}: trades {o['fills']} / accepted {o['accepted']}")
                continue
            oid = str(o["accepted"][0])
            book = [e for side in ("BUY", "SELL") for e in o["final_book"].get(side, [])]
            peak = int(o["still_resting"].get(oid, 0))
            hidden = int(o["hidden_remaining"].get(oid, -1))
            if len(book) != 1 or int(book[0][1]) != peak or peak <= 0:
                out.append(f"{part} seed {s}: book {book} is not exactly the visible peak {peak}")
            elif _ceil(peak + hidden, peak) != n_sl:
                out.append(f"{part} seed {s}: ceil(total/peak) = {_ceil(peak + hidden, peak)} != {n_sl}")
            if bat.get("I2") != "pass":
                out.append(f"{part} seed {s}: I2 {bat.get('I2')}")
            if int(i8.get("values", {}).get("slices", -1)) != n_sl:
                out.append(f"{part} seed {s}: slices value {i8.get('values')}")
    return out


def judge_lr_non(payload) -> List[str]:
    """S2 control, crossing NON: both last rows occupied; X on S executes completely
    against E. Correction 1a: the dispatch meets a full own side, so the worst level
    of S is wiped and book_full is raised at that post; nothing else is lost -- the
    last-row occupants of both sides still rest (the lookup miss is a miss)."""
    idx, part_of = index(payload)
    out = []
    for s in seeds_of(part_of, "C14-LR-NON"):
        o, f, det, bat = (_o(idx, s, k) for k in ("outcome", "flags", "detail", "battery"))
        n = len(o["reported_qty"])
        last = n - 1
        if f.get("book_full_posts") != [last] or f.get("truncated_posts") or f.get("trade_buffer_saturated"):
            out.append(f"seed {s}: flags {f} (documented: book_full at post {last} only)")
        fills = [[int(x) for x in r] for r in o["fills"]]
        if len(fills) != 1 or fills[0][:2] != [last, 0]:
            out.append(f"seed {s}: fills {fills} (one trade, X against E)")
            continue
        q = fills[0][3]
        parties = set(fills[0][4:6])
        if o["reported_qty"] != [0] * last + [q] or any(_ints(o["filled"]).get(p) != q for p in parties) \
                or parties & set(_resting(o)):
            out.append(f"seed {s}: X/E not exactly filled {o['filled']} / reported {o['reported_qty'][-1]}")
        ev = _ints(o["evicted"])
        if not 1 <= len(ev) <= 4 or set(ev) & set(_book(o)) or set(ev) & parties:
            out.append(f"seed {s}: evicted {ev} (the k in [1,4] worst-level orders, none in the book)")
        if set(int(a) for a in o["accepted"]) != set(_resting(o)) | set(ev) | parties:
            out.append(f"seed {s}: an accepted order vanished without a bucket (silent loss)")
        rows, other = det.get("full_side_rows_posts", []), det.get("other_side_rows_posts", [])
        if rows[:100] != list(range(1, 101)) or set(rows[100:last]) != {BOOK_DEPTH} or rows[last] != BOOK_DEPTH - len(ev):
            out.append(f"seed {s}: S rows {rows[98:]} (100 from post 99, then 100 - k)")
        if other[:100] != [0] * 100 or other[100:last] != list(range(1, 101)) or other[last] != BOOK_DEPTH - 1:
            out.append(f"seed {s}: O rows {other[98:]} (0 until post 99, 1..100, then 99 after X)")
        lr = det.get("last_rows_resting", {})
        book = _book(o)
        if set(lr) != {"S", "O"} or not all(v[1] is True and int(v[0]) in book for v in lr.values()):
            out.append(f"seed {s}: last-row occupants {lr} not both resting")
        if not _book_matches_resting(o) or bat.get("I2") != "pass":
            out.append(f"seed {s}: book/resting mismatch or I2 {bat.get('I2')}")
    return out


def judge_lr_ice(payload) -> List[str]:
    """S2 control, iceberg regeneration with the last row occupied: Z (first row) is
    hit by A1 for p then by A2 for its second slice min(p, T - p); Z regenerates into
    the row it freed each time (the lookup does not mistake the last row for its
    consumed slice). No flag, nothing evicted, the last-row occupant still rests."""
    idx, part_of = index(payload)
    out = []
    for s in seeds_of(part_of, "C14-LR-ICE"):
        o, f, det, bat = (_o(idx, s, k) for k in ("outcome", "flags", "detail", "battery"))
        if f.get("book_full_posts") or f.get("truncated_posts") or f.get("trade_buffer_saturated"):
            out.append(f"seed {s}: a flag was raised {f}")
        if o["evicted"] or o["cancelled"] or o["rejected"]:
            out.append(f"seed {s}: evicted/cancelled/rejected {o['evicted']} {o['cancelled']} {o['rejected']}")
        fills = [[int(x) for x in r] for r in o["fills"]]
        if len(o["reported_qty"]) != 102 or [r[:2] for r in fills] != [[100, 0], [101, 1]]:
            out.append(f"seed {s}: fills {fills}")
            continue
        z = set(fills[0][4:6]) & set(fills[1][4:6])
        if len(z) != 1 or fills[0][2] != fills[1][2]:
            out.append(f"seed {s}: the two trades are not both against Z at b: {fills}")
            continue
        z = z.pop()
        a1, a2 = (set(fills[0][4:6]) - {z}).pop(), (set(fills[1][4:6]) - {z}).pop()
        p, v2 = fills[0][3], fills[1][3]
        T = (int(o["filled"].get(str(z), 0)) + int(o["still_resting"].get(str(z), 0))
             + int(o["hidden_remaining"].get(str(z), 0)))
        if v2 != min(p, T - p):
            out.append(f"seed {s}: second slice {v2} != min(p, T - p) = {min(p, T - p)}")
        zr = _resting(o).get(z, 0)
        if zr != min(p, T - 2 * p) or (zr and _book(o).get(z, ("", 0))[1] != fills[0][2]):
            out.append(f"seed {s}: Z rests {zr}, documented min(p, T - 2p) = {min(p, T - 2 * p)} at b")
        if o["reported_qty"] != [0] * 100 + [p, v2] or {a1, a2} & set(_resting(o)):
            out.append(f"seed {s}: reports {o['reported_qty'][-2:]} / A1, A2 resting")
        if set(int(a) for a in o["accepted"]) != set(_resting(o)) | {a1, a2} | ({z} if zr == 0 else set()):
            out.append(f"seed {s}: an accepted order vanished")
        rows = det.get("full_side_rows_posts", [])
        want = list(range(1, 101)) + [BOOK_DEPTH, BOOK_DEPTH if zr else BOOK_DEPTH - 1]
        if rows != want:
            out.append(f"seed {s}: S rows {rows[-4:]} != {want[-4:]}")
        lr = det.get("last_rows_resting", {})
        if set(lr) != {"S"} or lr["S"][1] is not True or int(lr["S"][0]) not in _book(o) or int(lr["S"][0]) == z:
            out.append(f"seed {s}: last-row occupant {lr} not resting")
        if not _book_matches_resting(o) or bat.get("I2") != "pass":
            out.append(f"seed {s}: book/resting mismatch or I2 {bat.get('I2')}")
    return out


def _statuses(idx, s) -> Dict[str, str]:
    st = dict(_o(idx, s, "battery") or {})
    for ck in ("I1", "I2", "I6"):
        o = _o(idx, s, ck)
        if isinstance(o, dict) and "status" in o:
            st[ck] = o["status"]
    o3 = _o(idx, s, "I3")
    if isinstance(o3, dict) and "i3_status" in o3:
        st["I3"] = o3["i3_status"]
    return st


def judge_exclusions(payload) -> List[str]:
    """Each documented exclusion asserted; every other battery check pass / n/a."""
    idx, part_of = index(payload)
    out = []
    for s, part in sorted(part_of.items()):
        st = _statuses(idx, s)
        for ck in BATTERY:
            got = st.get(ck)
            if ck in EXCLUDED[part]:
                continue
            if got not in ("pass", "not_applicable"):
                out.append(f"{part} seed {s}: {ck} not excluded but {got}")
        for ck in ("I2", "I6") if part == "C14b" else ("I2",) if part == "C14d" else ():
            if st.get(ck) != "fail":
                out.append(f"{part} seed {s}: {ck} excluded (expected-violated) but {st.get(ck)}")
        i8 = _o(idx, s, "I8") or {}
        must, may = I8_BREACH[part]
        br = set(i8.get("breach", []))
        if not must <= br <= may or (not br and part != "C14c-below"):
            out.append(f"{part} seed {s}: I8 breach {sorted(br)} not within {sorted(must)}..{sorted(may)}")
        pin = {"C14a": ("resting_per_side", 100), "C14a′": ("resting_per_side", 100),
               "C14-LR-NON": ("resting_per_side", 100), "C14-LR-ICE": ("resting_per_side", 100),
               "C14b": ("trades", 100)}.get(part)
        if pin and i8.get("values", {}).get(pin[0]) != pin[1]:
            out.append(f"{part} seed {s}: I8 {pin[0]} = {i8.get('values', {}).get(pin[0])} != {pin[1]}")
    return out


# --- citation fidelity (each fragment must be in the named source) -----------
def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == heading)
    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


_SRC_RE = re.compile(r"^([\w./-]+\.(?:md|py))(?::(\d+)(?:-(\d+))?)?")


def judge_citations(payload) -> List[str]:
    out = []
    cits = payload.get("citations", [])
    if not cits:
        return ["no citations"]
    for c in cits:
        src = c.get("source", "")
        m = _SRC_RE.match(src)
        if not m:
            out.append(f"{src}: no source path")
            continue
        path = REPO_ROOT / m.group(1)
        if not path.is_file():
            out.append(f"{src}: source missing")
            continue
        raw = path.read_text(encoding="utf-8")
        if path.name == "sprint-status.md":
            raw = _section(raw, "## OMIE-ICE-INCOMING-FILL-REPORT-1")
        vm = re.match(r"^[\w./-]+\.md,?\s+(v\d+\.\d+)\b", src)   # a version stamped on the document itself
        if vm and path.suffix == ".md":
            hdr = re.search(r"^\*\*Status:\*\*\s*(v\d+(?:\.\d+)*)", raw, re.M)
            if hdr and hdr.group(1) != vm.group(1):
                out.append(f"{src}: cites {vm.group(1)}, source is {hdr.group(1)}")
        hay = _norm(raw)
        frags = [x.strip() for x in c.get("text", "").split("[…]") if x.strip()]
        if not frags:
            out.append(f"{src}: empty citation")
        for fr in frags:
            if _norm(fr) not in hay:
                out.append(f"{src}: fragment not in source: {fr[:70]!r}")
        if m.group(2) and frags:
            lo, hi = int(m.group(2)), int(m.group(3) or m.group(2))
            pat = r"\s+".join(re.escape(t) for t in frags[0].split()[:6])
            hit = re.search(pat, raw)
            line = raw.count("\n", 0, hit.start()) + 1 if hit else None
            if line is None or not lo - 1 <= line <= hi + 1:
                out.append(f"{src}: first fragment at line {line}, cited {lo}-{hi}")
    return out


# ---------------------------------------------------------------------------
# 1. Envelope, ceilings, quotes, citations
# ---------------------------------------------------------------------------
class TestEnvelope:
    def test_schema_status_failure_kinds_and_seed_bookkeeping(self, payload):
        assert judge_envelope(payload) == []

    def test_configured_ceilings_and_the_100_pass_bound(self, payload):
        assert judge_ceilings(payload) == []

    def test_spec_stamp_v214_and_quotes_match(self, payload):
        assert judge_quotes(payload) == []

    def test_every_citation_fragment_is_in_its_named_source(self, payload):
        """r3: the ``omie_config.py:36-38, :48`` citation now re-quotes the current
        comment ``max_match_iterations: int = 200  # truncation cap; inert while >
        `_TRIP_BOUND` (100), which is the effective outer pass bound`` (the r2 finding --
        a stale quote of the pre-OMIE-CAPACITY-DOCS-1 "safety bound for multi-slice
        matching loops" comment -- is resolved). Every fragment must be in its source."""
        assert judge_citations(payload) == []

    def test_ladder_doc_version(self):
        hdr = re.search(r"^\*\*Status:\*\*\s*(v\d+(?:\.\d+)*)",
                        (DOCS / "scenario-ladder.md").read_text(encoding="utf-8"), re.M)
        assert hdr and hdr.group(1) == LADDER_VERSION


# ---------------------------------------------------------------------------
# 2. Eager vs jitted (absence = failure) and per-draw completeness
# ---------------------------------------------------------------------------
class TestEvidence:
    def test_eager_jit_present_for_every_part_and_identical(self, payload):
        assert judge_eager_jit(payload) == []

    def test_every_draw_reports_its_full_outcome(self, payload):
        """RERUN-1: every draw (C14c included) records final_book, filled,
        hidden_remaining, reported_qty and still_resting."""
        assert judge_per_draw_full(payload) == []


# ---------------------------------------------------------------------------
# 3. Per part, documented behaviour against the emitted per-draw records
# ---------------------------------------------------------------------------
class TestParts:
    def test_c14a_book_depth_flagged_worst_level_wipe(self, payload):
        assert judge_c14a(payload) == []

    def test_c14a_prime_full_only_at_regeneration_unflagged_nothing_lost(self, payload):
        """Recorded, not certified -- but any departure fails it."""
        assert judge_c14a_prime(payload) == []

    def test_c14b_ring_collapse_detectable_report_is_true_burst_quantity(self, payload):
        assert judge_c14b(payload) == []

    def test_c14c_100_pass_bound_truncates_walk_in_price_order(self, payload):
        assert judge_c14c(payload) == []

    def test_c14c_below_bound_exact_walk_no_truncation(self, payload):
        assert judge_c14c_below(payload) == []

    def test_c14d_slot_overflow_bounded_without_side_channel_corruption(self, payload):
        assert judge_c14d(payload) == []

    def test_c14e_slice_ceiling_boundary_pair(self, payload):
        assert judge_c14e(payload) == []

    def test_lr_non_control_crossing_on_full_side_loses_only_the_wiped_level(self, payload):
        assert judge_lr_non(payload) == []

    def test_lr_ice_control_iceberg_regenerates_with_last_row_occupied(self, payload):
        assert judge_lr_ice(payload) == []


# ---------------------------------------------------------------------------
# 4. Exclusions
# ---------------------------------------------------------------------------
class TestExclusions:
    def test_excluded_violated_as_documented_and_all_others_hold(self, payload):
        assert judge_exclusions(payload) == []

    def test_c14b_i1_cannot_fail_both_legs_from_one_row(self, payload):
        """Card text lists I1 as expected-violated on C14b, but OQ-15's collapse keeps
        BOTH legs of the kept trade in one row, so I1 (buy leg == sell leg) cannot
        fail. Asserted as derived: I1 pass, and the kept row carries both parties.
        (Card-wording finding, not an engine defect.)"""
        idx, part_of = index(payload)
        for s in seeds_of(part_of, "C14b"):
            assert _o(idx, s, "I1")["status"] == "pass", s
            o = _o(idx, s, "outcome")
            last = len(o["reported_qty"]) - 1
            kept = _fills_by_sub(o)[last]
            assert len(kept) == 1 and kept[0][4] != kept[0][5], (s, kept)

    def test_c14d_i2_failure_is_the_documented_untracked_hidden_part(self, payload):
        """§B1 / FILL-REPORT-1 close-out: an overflowed iceberg rests its peak only and
        its hidden part is not tracked, so I2 identity fails on that order alone and by
        exactly accepted - peak. The card's exclusion list omits C14d's I2 (finding)."""
        idx, part_of = index(payload)
        for s in seeds_of(part_of, "C14d"):
            (oid, field, acc, got), = _o(idx, s, "I2")["violations"]
            o = _o(idx, s, "outcome")
            assert field == "identity" and got == _resting(o)[int(oid)] and acc > got > 0, s


# ---------------------------------------------------------------------------
# 5. Deliberately broken copies (rule 1: every judge seen failing)
# ---------------------------------------------------------------------------
def _find(data, seed, kind):
    idx, _ = index(data)
    return idx[seed][kind]


def _set_obs(a, obj):
    a["observed"] = json.dumps(obj) if isinstance(a["observed"], str) else obj


def _mutate(d, seed, kind, fn):
    a = _find(d, seed, kind)
    o = _obs(a)
    fn(o)
    _set_obs(a, o)


class TestBrokenCopies:
    def test_missing_eager_jit_evidence(self, payload, tmp_path):
        d = copy.deepcopy(payload)
        d["assertions"] = [a for a in d["assertions"] if a.get("kind") != "eager_jit"]
        assert judge_eager_jit(_roundtrip(tmp_path, d)) == ["no eager-vs-jitted evidence"]

    def test_one_part_without_eager_jit(self, payload, tmp_path):
        d = copy.deepcopy(payload)
        d["assertions"] = [a for a in d["assertions"] if not (a.get("kind") == "eager_jit" and a.get("seed") == 900000)]
        assert any("C14-LR-ICE" in p for p in judge_eager_jit(_roundtrip(tmp_path, d)))

    def test_missing_full_outcome_key(self, payload, tmp_path):
        d = copy.deepcopy(payload)
        _mutate(d, 300001, "full", lambda o: o.pop("hidden_remaining"))
        _mutate(d, 800002, "outcome", lambda o: o.pop("reported_qty"))
        assert len(judge_per_draw_full(_roundtrip(tmp_path, d))) >= 2

    def test_envelope_tampering(self, payload, tmp_path):
        d = copy.deepcopy(payload)
        d["status"] = "NOT CERTIFIED"
        d["draws"]["discarded"] = 1
        assert len(judge_envelope(_roundtrip(tmp_path, d))) >= 2

    def test_quote_version_tampering(self, payload, tmp_path):
        d = copy.deepcopy(payload)
        q = next(q for q in d["spec"]["quotes"] if q["source"] == "market-spec.md")
        q["version"] = "v2.13"
        assert judge_quotes(_roundtrip(tmp_path, d)) != []

    def test_citation_tampering(self, payload, tmp_path):
        """K4 (correction (b), 2026-10-27): the tampered fragment is derived from
        citation 1's own quoted text, never a hard-coded phrase, so a re-pointed
        citation can not turn the tamper into a silent no-op (the K2 defect: the
        hard-coded "worst level" vanished from citations[1] and the copy passed)."""
        assert judge_citations(payload) == []                     # untampered baseline is clean
        d = copy.deepcopy(payload)
        cit = d["citations"][1]
        before = cit["text"]
        frags = [x.strip() for x in before.split("[…]") if x.strip()]
        assert frags, "citation 1 has no quoted fragment to tamper with"
        words = re.findall(r"[A-Za-z]{5,}", frags[0])
        assert words, "citation 1's first fragment has no word to tamper with"
        word = max(words, key=len)
        altered = word[::-1] if word[::-1] != word else word + "q"
        cit["text"] = before.replace(word, altered, 1)
        # precondition: the tamper really changed citation 1's quoted text
        assert cit["text"] != before
        assert altered in cit["text"]
        problems = judge_citations(_roundtrip(tmp_path, d))
        assert problems != []
        assert any(p.startswith(cit["source"]) and "fragment not in source" in p for p in problems)

    def test_c14a_manufactured_trade_and_ghost(self, payload, tmp_path):
        for mut in ("trade", "flag", "ghost"):
            d = copy.deepcopy(payload)
            if mut == "flag":
                _mutate(d, 1, "flags", lambda o: o.update(book_full_posts=[]))
            elif mut == "trade":
                _mutate(d, 1, "outcome", lambda o: o.update(fills=[[101, 0, 1, 1, 2, 3]]))
            else:
                def ghost(o):
                    o["evicted"].pop(next(iter(o["evicted"])))
                _mutate(d, 1, "outcome", ghost)
            assert judge_c14a(_roundtrip(tmp_path, d)) != [], mut

    def test_c14a_prime_the_r1_signature_is_rejected(self, payload, tmp_path):
        """The r1 signature (book_full raised, worst level evicted, no regenerated
        slice, side ends below 100) must fail the r2 judge."""
        d = copy.deepcopy(payload)
        _mutate(d, 100000, "flags", lambda o: o.update(book_full_posts=[100]))

        def r1(o):
            x = next(iter(o["hidden_remaining"]))
            o["evicted"] = {"999999": 4}
            o["still_resting"][x] = 0
            for side in ("BUY", "SELL"):
                o["final_book"][side] = [e for e in o["final_book"][side] if str(e[2]) != x]
        _mutate(d, 100000, "outcome", r1)
        _mutate(d, 100000, "detail", lambda o: o["full_side_rows_posts"].__setitem__(-1, 97))
        assert len(judge_c14a_prime(_roundtrip(tmp_path, d))) >= 3

    def test_c14b_unsaturated_uncollapsed_and_row_report(self, payload, tmp_path):
        d = copy.deepcopy(payload)
        _mutate(d, 200000, "flags", lambda o: o.update(trade_buffer_saturated=False))

        def extra(o):
            last = len(o["reported_qty"]) - 1
            o["fills"].append([last, 98, 1, 1, 2, 3])
        _mutate(d, 200001, "outcome", extra)

        def f4_old(o):   # r1's F4: the report equals the surviving row
            last = len(o["reported_qty"]) - 1
            o["reported_qty"][last] = _fills_by_sub(o)[last][0][3]
        _mutate(d, 200002, "outcome", f4_old)
        p = judge_c14b(_roundtrip(tmp_path, d))
        assert any("200000" in x for x in p) and any("200001" in x for x in p) and any("200002" in x for x in p)

    def test_c14c_round_robin_and_no_truncation_rejected(self, payload, tmp_path):
        d = copy.deepcopy(payload)
        _mutate(d, 300000, "flags", lambda o: o.update(truncated_posts=[]))

        def rr(o):
            f = o["fills"]
            f[1][2], f[1][3], f[1][4], f[1][5] = f[-1][2], f[-1][3], f[-1][4], f[-1][5]
        _mutate(d, 300001, "full", rr)
        p = judge_c14c(_roundtrip(tmp_path, d))
        assert any("300000" in x for x in p) and any("300001" in x for x in p)

    def test_c14c_below_truncated_rejected(self, payload, tmp_path):
        d = copy.deepcopy(payload)
        _mutate(d, 700000, "flags", lambda o: o.update(truncated_posts=[4]))
        assert judge_c14c_below(_roundtrip(tmp_path, d)) != []

    def test_c14d_corrupted_side_channel(self, payload, tmp_path):
        d = copy.deepcopy(payload)

        def bad(o):
            o["meta_unchanged_at_overflow"] = False
            o["n_active_posts_through_overflow"][-1] = 5
        _mutate(d, 400002, "detail", bad)
        assert len(judge_c14d(_roundtrip(tmp_path, d))) >= 2

    def test_c14e_boundary_swapped(self, payload, tmp_path):
        d = copy.deepcopy(payload)
        _mutate(d, 500000, "flags", lambda o: o.update(truncated_posts=[0]))

        def shrink(o):
            oid = str(o["accepted"][0])
            o["hidden_remaining"][oid] -= 2 * o["still_resting"][oid]
        _mutate(d, 600001, "outcome", shrink)
        assert len(judge_c14e(_roundtrip(tmp_path, d))) >= 2

    def test_lr_non_r1_silent_last_row_deletion_rejected(self, payload, tmp_path):
        """The r1 defect (the last-row occupant silently deleted) must fail."""
        d = copy.deepcopy(payload)

        def lose(o):
            s_oid = str(_obs(_find(d, 800000, "detail"))["last_rows_resting"]["S"][0])
            o["still_resting"].pop(s_oid)
            for side in ("BUY", "SELL"):
                o["final_book"][side] = [e for e in o["final_book"][side] if str(e[2]) != s_oid]
        _mutate(d, 800000, "outcome", lose)
        _mutate(d, 800000, "detail", lambda o: o["last_rows_resting"]["S"].__setitem__(1, False))
        assert len(judge_lr_non(_roundtrip(tmp_path, d))) >= 2

    def test_lr_ice_r1_stranded_iceberg_rejected(self, payload, tmp_path):
        """The r1 defect (Z stranded: no second trade) must fail."""
        d = copy.deepcopy(payload)

        def strand(o):
            o["fills"] = o["fills"][:1]
            o["reported_qty"][-1] = 0
        _mutate(d, 900000, "outcome", strand)
        assert judge_lr_ice(_roundtrip(tmp_path, d)) != []

    def test_exclusion_not_violated_or_unrelated_failure(self, payload, tmp_path):
        d = copy.deepcopy(payload)
        _mutate(d, 200002, "I6", lambda o: o.update(status="pass"))
        _mutate(d, 0, "battery", lambda o: o.update(I5="fail"))
        p = judge_exclusions(_roundtrip(tmp_path, d))
        assert any("C14b seed 200002: I6" in x for x in p) and any("C14a seed 0: I5" in x for x in p)


# ---------------------------------------------------------------------------
# 6. Independent engine probes via ContractBook's public API (jitted submit)
# ---------------------------------------------------------------------------
def _cb_book(market_config=None):
    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.contract_book import ContractBook
    from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
    contract = Contract(contract_id=0, delivery_start_minutes=555, delivery_end_minutes=570,
                        gct_minutes=495, status=ContractStatus.OPEN)
    pc = OMIEProductConfig()
    return ContractBook(contract, pc, market_config or OMIEMarketConfig()), pc


def _cb_run(rows: Sequence[Sequence[int]], market_config=None):
    """Submit 13-column rows in order; return (book, final state, per-submit states)."""
    import jax.numpy as jnp
    import numpy as np
    from omiecid_simulator.jaxomiecid.omie_data_loader import _row_to_order_traced
    book, pc = _cb_book(market_config)
    state = book.init()
    states = []
    for r in rows:
        state, res = book.submit_order(state, _row_to_order_traced(jnp.asarray(np.asarray(r, dtype=np.int32)), pc))
        assert bool(res.accepted), list(r)
        states.append(state)
    return book, state, states


def _cb_trades(state) -> List[Tuple[int, int, int, int]]:
    """(price, qty, passive oid, aggressor oid) in ring order."""
    import numpy as np
    from omiecid_simulator.jaxob.jaxob_constants import TradesFeat as T
    t = np.asarray(state.lob.trades)
    t = t[t[:, T.P.value] > 0]
    return [(int(r[T.P.value]), abs(int(r[T.Q.value])), int(r[T.PASS_OID.value]), int(r[T.AGRS_OID.value]))
            for r in t]


def _cb_side(state, side: int) -> Dict[int, Tuple[int, int]]:
    """oid -> (price, qty) of the live rows of one side."""
    import numpy as np
    from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat as F
    a = np.asarray(state.lob.bids if side == 0 else state.lob.asks)
    a = a[(a[:, F.OID.value] > 0) & (a[:, F.Q.value] > 0)]
    return {int(r[F.OID.value]): (int(r[F.P.value]), int(r[F.Q.value])) for r in a}


def _rows(spec) -> List[List[int]]:
    """spec: (side, price_int, qty_int, oid, peak_int or None)."""
    from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder
    from omiecid_simulator.omiecid.order_types import ExecRestriction
    tb = TapeBuilder()
    for side, p, q, oid, pk in spec:
        kw = dict(exec_restriction=int(ExecRestriction.ICE), peak_int=pk) if pk else {}
        tb.add(side=side, price_int=p, qty_int=q, order_id=oid, **kw)
    return tb._rows


def _spec_walk(levels, agg_qty) -> List[Tuple[int, int, int]]:
    """§6.1 + §4.2 + Rule 2 (v2.14 re-ranking), levels best-first: (price, qty, oid, peak)."""
    left, out = agg_qty, []
    for price, qty, oid, peak in levels:
        avail = qty
        while left and avail:
            q = min(left, peak or qty, avail)
            out.append((price, q, oid))
            left, avail = left - q, avail - q
    return out


# Off-round int prices, mixed oid magnitudes.
_PRIORITY_CASES = {
    "two BUY icebergs, SELL 25": (
        [(0, 1004517, 30, 11, 10), (0, 1004503, 30, 22987, 10)], (1, 1004400, 25, 33)),
    "two SELL icebergs, BUY 17": (
        [(1, 1004503, 21, 7, 6), (1, 1004517, 21, 40411, 6)], (0, 1004600, 17, 905)),
    "BUY iceberg over a NON level, SELL 19": (
        [(0, 1004517, 30, 61, 8), (0, 1004503, 30, 22, None)], (1, 1004400, 19, 3)),
}


@pytest.mark.slow
@pytest.mark.jit
@pytest.mark.parametrize("case", sorted(_PRIORITY_CASES))
def test_cb_probe_price_priority_over_better_iceberg_next_slice(case):
    """slow: one _submit_order compile + 3 submits (~12 s). S3 control (r1 finding
    (i) flipped): an aggressor larger than the better iceberg's peak trades that
    iceberg's next slices at the better price before any worse level."""
    rest, (aside, ap, aq, aoid) = _PRIORITY_CASES[case]
    _, st, _ = _cb_run(_rows(rest + [(aside, ap, aq, aoid, None)]))
    want = [(p, q, o, aoid) for p, q, o in _spec_walk([(p, q, o, pk) for _, p, q, o, pk in rest], aq)]
    assert _cb_trades(st) == want, case


@pytest.mark.slow
@pytest.mark.jit
@pytest.mark.parametrize("n,truncates", [(60, True), (45, False)])
def test_cb_probe_100_pass_bound_sets_truncated(n, truncates):
    """slow: one _submit_order compile (ring 256) + 3 submits (~15-25 s). Two BUY
    icebergs of n one-tick slices (n <= 64, so registration never truncates) and a
    SELL for 2n. 2n = 120 slices need > 100 passes: truncated is raised at the
    aggressor only, the fills are the price-priority walk prefix (P1 drained first),
    100 <= #fills <= 103, and the residual rests opposite a consumed slot. 2n = 90:
    no truncation, the whole walk, nothing left."""
    from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig
    rest = [(0, 1004517, n, 11, 1), (0, 1004503, n, 22, 1)]
    _, st, sts = _cb_run(_rows(rest + [(1, 1004400, 2 * n, 33, None)]), OMIEMarketConfig(nTrades=256))
    assert [bool(s.truncated) for s in sts] == [False, False, truncates]
    assert not any(bool(s.book_full) for s in sts)
    tr = _cb_trades(st)
    walk = [(p, q, o, 33) for p, q, o in _spec_walk([(p, q, o, pk) for _, p, q, o, pk in rest], 2 * n)]
    if truncates:
        assert PASS_BOUND <= len(tr) <= PASS_BOUND + 1 + 2, len(tr)
        assert tr == walk[:len(tr)]
        resid = _cb_side(st, 1)
        assert resid == {33: (1004400, 2 * n - len(tr))}
        assert 22 not in _cb_side(st, 0)            # consumed slot: hidden left, no visible slice
    else:
        assert tr == walk and not _cb_side(st, 0) and not _cb_side(st, 1)


@pytest.mark.slow
@pytest.mark.jit
def test_cb_probe_last_row_cancel_by_id_miss_deletes_nothing():
    """slow: 100 jitted submits + 2 cancels (~25 s). S2 control: with the BUY side
    full (its last physical row occupied), ContractBook.cancel_order for an unknown
    oid changes nothing -- every one of the 100 orders, the last-row one included,
    still rests with its quantity; a real cancel of the last-row order removes it
    alone."""
    import numpy as np
    spec = [(0, 1004100 + 3 * i, 7 + i % 5, 500 + 7 * i, None) for i in range(BOOK_DEPTH)]
    book, st, _ = _cb_run(_rows(spec))
    before = _cb_side(st, 0)
    assert len(before) == BOOK_DEPTH and not bool(st.book_full)
    lob_before = np.asarray(st.lob.bids).copy()
    miss = book.cancel_order(st, 777777, 0)
    assert _cb_side(miss, 0) == before
    assert (np.asarray(miss.lob.bids) == lob_before).all()
    miss_other = book.cancel_order(st, 777777, 1)
    assert _cb_side(miss_other, 0) == before and not _cb_side(miss_other, 1)
    last_oid = 500 + 7 * (BOOK_DEPTH - 1)
    hit = book.cancel_order(st, last_oid, 0)
    assert _cb_side(hit, 0) == {k: v for k, v in before.items() if k != last_oid}


# ---------------------------------------------------------------------------
# 7. Independent probes via the reference record builder (own tapes)
# ---------------------------------------------------------------------------
def _engine():
    import numpy as np
    from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
    from tests.omiecid.reference import omie_invariants as inv
    return np, TapeBuilder, int(Side.BUY), int(Side.SELL), int(ExecRestriction.ICE), inv


def _full_side_tape(n_buy: int, newcomer, follow_up=None):
    """SELL 0.1 @ 60.00 resting first, then n_buy BUY 1.0 (3 at the worst price
    40.00, the rest 41.00-41.49; oid 199 in the side's last physical row), then the
    newcomer (and an optional SELL follow-up)."""
    np, TapeBuilder, BUY, SELL, ICE, _ = _engine()
    tb = TapeBuilder()
    tb.add(side=SELL, price_eur=60.00, qty_mw=0.1, order_id=5000)
    for i in range(n_buy):
        tb.add(side=BUY, price_eur=40.00 if i < 3 else 41.00 + 0.01 * (i % 50), qty_mw=1.0, order_id=100 + i)
    kind, price, qty = newcomer
    kw = dict(exec_restriction=ICE, peak_qty_mw=0.1) if kind == "ICE" else {}
    tb.add(side=BUY, price_eur=price, qty_mw=qty, order_id=9000, **kw)
    if follow_up:
        tb.add(side=SELL, price_eur=follow_up[0], qty_mw=follow_up[1], order_id=9100)
    return np.asarray(tb.build())


WORST = {100, 101, 102}


def _gone(rec, s, side):
    return {r[2] for r in rec.pre_snapshots[s].resting(side)} - {r[2] for r in rec.post_snapshots[s].resting(side)}


@pytest.mark.slow
@pytest.mark.jit
def test_probe_c14a_resting_newcomer_wipes_only_the_worst_level():
    """slow: 102-order build (~15 s). Correction 1: a non-crossing 101st BUY raises
    book_full, wipes exactly the worst level into the evicted bucket, rests, trades
    nothing; I2 holds."""
    _, _, BUY, _, _, inv = _engine()
    rec = inv.build_record(_full_side_tape(100, ("NON", 45.00, 1.0)))
    s = rec.n_submissions - 1
    assert [p.book_full for p in rec.post_snapshots][s - 1:] == [False, True]
    assert {k for k, v in rec.evicted.items() if v} == WORST
    assert rec.still_resting[9000] == 10 and not rec.fills
    assert len(rec.post_snapshots[s].resting(BUY)) == 100 - 3 + 1
    assert inv.check_i2_quantity_conservation(rec, allow_eviction=True).status == "pass"


@pytest.mark.slow
@pytest.mark.jit
def test_probe_crossing_newcomer_on_full_side_flagged_wipe_last_row_kept():
    """slow: 102-order build (~15 s). Correction 1a + S2 (r1 finding (ii) flipped): a
    crossing BUY that executes completely on a full own side wipes the worst level
    with book_full; the last-row order (oid 199) is NOT lost; I2 holds."""
    _, _, BUY, _, _, inv = _engine()
    rec = inv.build_record(_full_side_tape(100, ("NON", 60.00, 0.1)))
    s = rec.n_submissions - 1
    assert rec.post_snapshots[s].book_full and not rec.post_snapshots[s - 1].book_full
    assert _gone(rec, s, BUY) == WORST
    assert rec.still_resting[199] == 10
    assert [(f.passive_oid, f.aggressor_oid, f.qty_int) for f in rec.fills] == [(5000, 9000, 1)]
    assert inv.check_i2_quantity_conservation(rec, allow_eviction=True).status == "pass"


@pytest.mark.slow
@pytest.mark.jit
def test_probe_off_ceiling_incoming_iceberg_regenerates():
    """slow: 101-order build (~15 s). Control: with 99 bids the incoming iceberg
    sweeps 0.1, rests a 0.1 slice (side 100) with no flag and no eviction; a later
    SELL 0.5 takes five slices (Rule 2)."""
    _, _, _, _, _, inv = _engine()
    rec = inv.build_record(_full_side_tape(99, ("ICE", 60.00, 1.0), follow_up=(60.00, 0.5)))
    assert not any(rec.evicted.values()) and not rec.flags["book_full"]
    assert [(f.passive_oid, f.aggressor_oid) for f in rec.fills] == [(5000, 9000)] + [(9000, 9100)] * 5
    assert (rec.still_resting[9000], rec.hidden_remaining[9000]) == (1, 3)


@pytest.mark.slow
@pytest.mark.jit
def test_probe_incoming_iceberg_on_full_side_is_a_flag_site_and_regenerates():
    """slow: 103-order build (~15 s). Correction 1a (the incoming iceberg's sweep is a
    flag site; it fires even when the pass executes) + S2 + v2.14 (an incoming ICE
    aggresses with its total; its remainder enters as an iceberg, visible
    min(peak, R)). Expected: book_full at the iceberg's post, the worst level wiped,
    visible 0.1 + hidden 0.8 tracked; a later SELL 0.5 reaches five slices."""
    _, _, BUY, _, _, inv = _engine()
    rec = inv.build_record(_full_side_tape(100, ("ICE", 60.00, 1.0), follow_up=(60.00, 0.5)))
    s = rec.n_submissions - 2
    assert rec.post_snapshots[s].book_full and not rec.post_snapshots[s - 1].book_full
    assert _gone(rec, s, BUY) == WORST
    assert rec.filled[9100] == 5
    assert (rec.still_resting[9000], rec.hidden_remaining[9000]) == (1, 3)
    assert rec.still_resting[199] == 10
    assert inv.check_i2_quantity_conservation(rec, allow_eviction=True).status == "pass"


def _fixture(name: str):
    import numpy as np
    p = FIXTURES / f"{name}.npz"
    if not p.is_file():
        pytest.skip(f"fixture absent: {p}")
    z = np.load(p, allow_pickle=True)
    return np.asarray(z["tape"]), json.loads(str(z["params_json"]))


@pytest.mark.slow
@pytest.mark.jit
def test_fixture_last_row_crossing_non_now_loses_nothing():
    """slow: 103-row replay (~15 s). Frozen r1 counterexample, now an S2 control:
    the crossing NON fully executes against oid 5000 on a side of 99 (no flag site
    on a full side), and oid 199 in the last physical row still rests 1.0 MW."""
    from tests.omiecid.reference import omie_invariants as inv
    tape, params = _fixture("C14_last_row_lookup_crossing_non")
    rec = inv.build_record(tape)
    assert rec.still_resting[199] == 10
    assert not rec.flags["book_full"] and not any(rec.evicted.values())
    assert inv.check_i2_quantity_conservation(rec, allow_eviction=True).status == "pass"


@pytest.mark.slow
@pytest.mark.jit
def test_fixture_last_row_iceberg_no_longer_stranded():
    """slow: 104-row replay (~15 s). Frozen r1 counterexample, now an S2 control: the
    iceberg regenerates, so the later SELL 0.5 @ its price fills five slices."""
    from tests.omiecid.reference import omie_invariants as inv
    tape, _ = _fixture("C14_last_row_lookup_iceberg_stranded")
    rec = inv.build_record(tape)
    assert rec.filled[9100] == 5
    assert (rec.still_resting[9000], rec.hidden_remaining[9000]) == (1, 3)
    assert inv.check_i2_quantity_conservation(rec, allow_eviction=True).status == "pass"


@pytest.mark.slow
@pytest.mark.jit
def test_fixture_price_priority_follows_spec_eager_equals_jitted():
    """slow: eager + jitted 3-row replay (~15 s). Frozen r1 counterexample, now an S3
    control: fills == the §6.1/§4.2 walk recorded in the fixture; eager == jitted."""
    from tests.omiecid.reference import omie_invariants as inv
    tape, params = _fixture("C14_price_priority_two_buy_icebergs")
    rj, re_ = inv.build_record(tape), inv.build_record(tape, jit=False)
    assert inv.compare_records(rj, re_) == ()
    got = [[f.price_int, f.qty_int, f.passive_oid, f.aggressor_oid] for f in rj.fills]
    assert got == params["expected_fills_spec"]
    assert inv.run_battery(rj, expects_trade=True).passed


@pytest.mark.slow
@pytest.mark.jit
def test_fixture_r1_c14c_now_hits_the_100_pass_bound_in_price_order():
    """slow: replays the r1 C14c tape (four icebergs, 61/61/61/60 slices, 243 in all)
    with a 256-row ring (~20 s). r1 saw no truncation and a round-robin walk; r2
    derivation: 243 slices need > 100 passes, so truncated is raised at the
    aggressor, the fills are a prefix of the price-priority walk followed by at most
    one slice per later iceberg, and 100 <= #fills <= 107."""
    from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig
    from tests.omiecid.reference import omie_invariants as inv
    tape, params = _fixture("C14_seed300000")
    rec = inv.build_record(tape, market_config=OMIEMarketConfig(nTrades=int(params["n_trades_ring"])))
    ices = sorted(params["ices"], key=lambda i: i["rank"])
    assert rec.flags["truncated"] is True
    assert [p.truncated for p in rec.post_snapshots][:-1] == [False] * (rec.n_submissions - 1)
    walk = _spec_walk([(i["price"], i["T"], i["oid"], i["p"]) for i in ices], params["Q"])
    got = [(f.price_int, f.qty_int, f.passive_oid) for f in rec.fills]
    cut = 0
    while cut < len(got) and got[cut] == walk[cut]:
        cut += 1
    rank = {i["oid"]: i["rank"] for i in ices}
    tail = [rank[g[2]] for g in got[cut:]]
    assert tail == sorted(set(tail)) and (not tail or tail[0] >= rank[walk[cut][2]]), got[cut:]
    assert PASS_BOUND <= len(got) <= PASS_BOUND + 3 + 4, len(got)
    assert 0 < rec.filled[params["agg_oid"]] < params["Q"]
    assert inv.check_i2_quantity_conservation(rec).status == "pass"
