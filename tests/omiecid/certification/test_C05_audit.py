"""K4 audit twin for ``OMIE-CERT-C05-1`` — partial fill and the resting residual.

Fresh-context audit (certification cycle K4; C05 is a high-risk scenario under
independence rule 4). Written from the specification only:

* ``docs/omiecid/agile-plan.md`` (the C05 card),
* ``docs/omiecid/market-spec.md`` §4.5, §5.1, §5.2, §6.1, §6.2,
* ``docs/omiecid/scenario-ladder.md`` / ``scenario-ladder-plan.md`` §C5 and the
  capacity-validity precondition,
* ``docs/omiecid/scenario-ladder-review.md`` §8 (the 2026-09-23 residual-priority ruling),
* the regulatory corpus (Continuous Trading Matching Algorithm §4.3, §5.1.3),

plus the public generator API (``omie_tape_synth.make_partial_fill_residual``)
and the battery's public record builder (``omie_invariants.build_record``).
The C05 notebook, the engine (``contract_book.py``, ``iceberg_manager.py``,
``order_validator.py``, ``settlement.py``) and the K2 expectation are NOT read.

What is audited is the EMITTED ``results/C05.json``, whatever its status:

1. Every per-run value recorded in the JSON is re-derived here, independently,
   from the drawn parameters and the rules, with a tiny price-time-priority
   reference matcher (§6.1 ranking, §6.2 items 1-4 maker pricing and NON
   residual, §5.2 single global queue: an order's place in the queue is its
   arrival, and a partial execution does not move it).
2. Each recorded assertion's status must be exactly what the observation
   implies under that derivation, and the scenario status / failure_kinds must
   follow from the payload (``validate_results`` + ``derive_failure_kinds``).
3. Eager-vs-jitted evidence must be present for every construction and run
   kind, and identical (absence is a failure).
4. Battery, capacity (I8), non-vacuity (I9), quotes and citations are checked.
5. Deliberately broken copies of the JSON (written to ``tmp_path``) — a
   re-stamped residual, a residual at the wrong price, an interloper filled
   first, a forged PASS, missing eager/jit evidence — must be rejected.

Re-run r2 (``OMIE-CERT-RERUN-1``), after ``OMIE-ICE-PRICE-PRIORITY-1`` and
``OMIE-STAMP-UNIQUE-1`` merged. Judged against market-spec **v2.14** (pinned). r1
recorded a same-second stamp collision (residual and interloper both stored
``[sec, 0]``) contrary to §5.2 "unique, sequential"; the r1 characterisation of that
collision and the strict xfail on unique stamps are now plain rule assertions:
no run may violate the rule, and the stored stamps of a same-second residual and
interloper are distinct and in arrival order. Independent engine probes drive
``ContractBook``'s public API directly (same-second residual vs interloper at one
price, including a freed-slot reuse after a cancel) and cross-check stored stamps.

Results location: ``$OMIE_CERT_RESULTS_DIR/C05.json`` when the variable is set,
else ``notebooks/certification/results/C05.json``; absent -> skip (named reason).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_tape_synth import (
    COL_EXEC,
    COL_OID,
    COL_PRICE,
    COL_QTY,
    COL_SIDE,
    COL_TS,
    construction_digest,
    make_partial_fill_residual,
)
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from notebooks.certification.cert_support import (
    SpecQuote,
    check_quote,
    derive_failure_kinds,
    normalise,
    validate_results,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "notebooks" / "certification" / "results"

BUY, SELL = int(Side.BUY), int(Side.SELL)
NON = int(ExecRestriction.NON)
SIDE_NAME = {BUY: "BUY", SELL: "SELL"}

AGG = "C05.aggressor_residual"
RST = "C05.resting_residual"
VARIANT = {AGG: "aggressor", RST: "resting"}
DRAWS_PER_CONSTRUCTION = 500          # card: "Random scenarios default to 500 draws"
RESTING_SEED_OFFSET = 1_000_000       # the resting construction's seed block, as emitted
BATTERY_IDS = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")
SPEC_VERSION = "v2.14"                # pinned: the r2 re-run is judged against market-spec v2.14
ENVELOPE = {"resting_per_side": 80, "trades": 80, "hops": 160, "concurrent_icebergs": 800}

# Run kinds: the as-drawn tape, the interloper sharing the partial-fill
# aggressor's tape second, and (aggressor construction only) a no-cross control.
NORMAL, SAME, CTRL = "normal", "same_second", "no_cross_control"
RUN_KINDS = {AGG: (NORMAL, SAME, CTRL), RST: (NORMAL, SAME)}

_NAME_RE = re.compile(
    r"^(C05\.(?:aggressor|resting)_residual) seed (\d+)(?: (no_cross_control|same_second))?: (.*)$")
_EJ_DETAIL_RE = re.compile(
    r"^(C05\.(?:aggressor|resting)_residual) seed (\d+)(?: (no_cross_control|same_second))?: identical$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _results_path() -> Path:
    d = os.environ.get("OMIE_CERT_RESULTS_DIR")
    return (Path(d) if d else DEFAULT_RESULTS_DIR) / "C05.json"


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    path = _results_path()
    if not path.is_file():
        pytest.skip(f"C05 results JSON not emitted yet: {path} is absent "
                    "(run the C05 notebook, or set OMIE_CERT_RESULTS_DIR)")
    return _load(path)


# ---------------------------------------------------------------------------
# Independent reference: price-time priority from the rules
# ---------------------------------------------------------------------------
@dataclass
class _Row:
    oid: int
    side: int
    price: int
    qty: int
    ts: int
    exec_restriction: int


@dataclass
class _Entry:
    price: int
    qty: int
    oid: int
    ts: int
    seq: int          # arrival in the single global queue (§5.2) — never renewed by a fill


def _priority_key(side: int):
    # §6.1: buy side highest price first, sell side lowest price first; ties by
    # the earliest timestamp, i.e. the earliest arrival in the global queue.
    if side == BUY:
        return lambda e: (-e.price, e.seq)
    return lambda e: (e.price, e.seq)


def _book_view(book: Mapping[int, List[_Entry]]) -> Dict[str, List[List[int]]]:
    return {SIDE_NAME[s]: [[e.price, e.qty, e.oid] for e in sorted(book[s], key=_priority_key(s))]
            for s in (BUY, SELL)}


def _canon_book(x):
    if not isinstance(x, dict):
        return x
    return {k: sorted(map(tuple, v)) if isinstance(v, list) else v for k, v in x.items()}


def _copy_book(book):
    return {s: [_Entry(e.price, e.qty, e.oid, e.ts, e.seq) for e in book[s]] for s in (BUY, SELL)}


def reference_run(rows: Sequence[_Row]) -> Dict[str, Any]:
    """Submit ``rows`` in order through a rules-only matcher (NON orders only).

    §6.2: the incoming order matches the best opposite order while prices cross
    (items 1-2), each trade at the resting order's limit (item 3), level after
    level (item 6); an unfilled NON remainder enters the book with its original
    entry timestamp (item 4, §5.2). A partially executed resting order keeps its
    place (§5.2). Returns trades ``[passive, aggressor, price, qty, submission]``
    and the book before / after every submission.
    """
    book: Dict[int, List[_Entry]] = {BUY: [], SELL: []}
    trades: List[List[int]] = []
    pre, post = [], []
    for s, r in enumerate(rows):
        assert r.exec_restriction == NON, "C05 draws are NON-only (§4.5)"
        pre.append(_copy_book(book))
        opp_side = SELL if r.side == BUY else BUY
        rem = r.qty
        while rem > 0:
            crossing = [e for e in book[opp_side]
                        if (e.price <= r.price if r.side == BUY else e.price >= r.price)]
            if not crossing:
                break
            best = min(crossing, key=_priority_key(opp_side))
            q = min(rem, best.qty)
            trades.append([best.oid, r.oid, best.price, q, s])
            best.qty -= q
            rem -= q
            if best.qty == 0:
                book[opp_side].remove(best)
        if rem > 0:
            book[r.side].append(_Entry(r.price, rem, r.oid, r.ts, s))
        post.append(_copy_book(book))
    return {"trades": trades, "pre": pre, "post": post}


def _rows_of(tape: np.ndarray) -> List[_Row]:
    out = []
    for r in np.asarray(tape):
        if int(r[COL_OID]) >= 1:
            out.append(_Row(int(r[COL_OID]), int(r[COL_SIDE]), int(r[COL_PRICE]), int(r[COL_QTY]),
                            int(r[COL_TS]), int(r[COL_EXEC])))
    return out


def run_rows(construction: str, params: Mapping[str, Any], base: Sequence[_Row], kind: str) -> List[_Row]:
    """The row script of one run kind, derived from the drawn tape."""
    rows = [_Row(**vars(r)) for r in base]
    by_oid = {r.oid: r for r in rows}
    if kind == SAME:
        # The interloper shares the tape second of the partial-fill aggressor (submission 1).
        by_oid[params["interloper_oid"]].ts = by_oid[params["agg_oid"]].ts
    elif kind == CTRL:
        # The resting order is moved one tick behind the aggressor's limit: nothing crosses
        # at submission 1, so the aggressor rests whole and its entry stamp is observable.
        rest = by_oid[params["rest_oid"]]
        p_agg = by_oid[params["agg_oid"]].price
        rest.price = p_agg - 1 if rest.side == BUY else p_agg + 1
    return rows


@dataclass(frozen=True)
class RunDerivation:
    construction: str
    seed: int
    kind: str
    params: Mapping[str, Any]
    rows: Tuple[_Row, ...]
    ref: Mapping[str, Any]

    def ts_of(self, oid: int) -> int:
        return next(r.ts for r in self.rows if r.oid == oid)

    def qty_of(self, oid: int) -> int:
        return next(r.qty for r in self.rows if r.oid == oid)


@lru_cache(maxsize=None)
def _draw(construction: str, seed: int):
    return make_partial_fill_residual(seed, variant=VARIANT[construction])


@lru_cache(maxsize=None)
def derivation(construction: str, seed: int, kind: str) -> RunDerivation:
    d = _draw(construction, seed)
    rows = run_rows(construction, d.params, _rows_of(d.tape), kind)
    return RunDerivation(construction, seed, kind, d.params, tuple(rows), reference_run(rows))


def expected_seeds(construction: str) -> List[int]:
    off = 0 if construction == AGG else RESTING_SEED_OFFSET
    return list(range(off, off + DRAWS_PER_CONSTRUCTION))


# ---------------------------------------------------------------------------
# The per-run expectation, re-derived
# ---------------------------------------------------------------------------
def _prefix(construction: str, seed: int, kind: str) -> str:
    return f"{construction} seed {seed}" + ("" if kind == NORMAL else f" {kind}") + ": "


def expected_assertions(rd: RunDerivation) -> Dict[str, Tuple[str, Any]]:
    """``{assertion suffix: (check kind, derived value)}`` for one run."""
    p = rd.params
    ref = rd.ref
    trades = ref["trades"]
    accepted = [[r.oid for r in rd.rows], []]          # every C05 row is in §2.3 bounds
    if rd.kind == CTRL:
        return {
            "battery (RB, I1-I9) passes": ("battery", None),
            "(A) accepted order ids / rejected order ids": ("eq", accepted),
            "(Q) counterparty sequence": ("eq", trades),
            "(B1) book after submission 1 (a rests whole)": ("book", _book_view(ref["post"][1])),
            "(B) book at the end": ("book", _book_view(ref["post"][-1])),
            "the aggressor holds exactly one book entry after submission 1":
                ("stamp1", {"sec": rd.ts_of(p["agg_oid"])}),
        }
    owner = int(p["residual_oid"])
    interloper = int(p["interloper_oid"])
    post1 = ref["post"][1]
    owner_entries = [e for s in (BUY, SELL) for e in post1[s] if e.oid == owner]
    first_rest = next(k for k, b in enumerate(ref["post"])
                      if any(e.oid == owner for s in (BUY, SELL) for e in b[s]))
    labels = [f"post[{k}]" for k in range(first_rest, 3)] + ["pre[3]"]
    take_passives: List[int] = []
    for t in trades:
        if t[4] == 3 and t[0] not in take_passives:
            take_passives.append(t[0])
    interloper_filled = sum(t[3] for t in trades if interloper in (t[0], t[1]))
    filled: Dict[int, int] = {}
    for t in trades:
        filled[t[0]] = filled.get(t[0], 0) + t[3]
        filled[t[1]] = filled.get(t[1], 0) + t[3]
    final = ref["post"][-1]
    resting_end = {e.oid: e.qty for s in (BUY, SELL) for e in final[s]}
    conservation = {str(r.oid): filled.get(r.oid, 0) + resting_end.get(r.oid, 0) for r in rd.rows}
    declared = {str(r.oid): r.qty for r in rd.rows}
    assert conservation == declared, "reference matcher must conserve quantity (I2)"
    out: Dict[str, Tuple[str, Any]] = {
        "(A) accepted order ids / rejected order ids": ("eq", accepted),
        "(Q) counterparty sequence [passive, aggressor, price_int, qty_int, submission]": ("eq", trades),
        "(F) quantity traded at the partial fill (submission 1)":
            ("eq", sum(t[3] for t in trades if t[4] == 1)),
        f"(R) residual quantity of oid {owner} after the partial fill":
            ("eq", sum(e.qty for e in owner_entries)),
        f"(R) residual resting price(s) of oid {owner} after the partial fill (its own limit)":
            ("eq", [e.price for e in owner_entries]),
        "(B1) book after the partial fill [price_int, qty_int, oid]": ("book", _book_view(post1)),
        f"(T1) stored [sec, nsec] of oid {owner} unchanged from its first rest to the take":
            ("t1", {"labels": labels, "sec": rd.ts_of(owner)}),
        "(T3) just before the take, residual's [sec, nsec] < interloper's":
            ("t3", {"interloper_sec": rd.ts_of(interloper)}),
        "(O) take's passive oids / interloper's filled qty_int": ("eq", [take_passives, interloper_filled]),
        "(B) book at the end [price_int, qty_int, oid]": ("book", _book_view(final)),
        "(I2) filled + resting per oid == declared qty_int": ("eq", declared),
    }
    if rd.construction == AGG:
        out["(T2) residual's stored [sec, nsec] after the partial fill == the aggressor's entry "
            "stamp in the no_cross_control"] = ("t2", {"sec": rd.ts_of(p["agg_oid"])})
    if rd.kind == NORMAL:
        out["(I2) battery I2 status"] = ("i2b", None)
    else:
        out["battery (RB, I1-I9) passes"] = ("battery", None)
    return out


# ---------------------------------------------------------------------------
# The audit: problems + rule-implied violations
# ---------------------------------------------------------------------------
def _is_stamp(x) -> bool:
    return isinstance(x, list) and len(x) == 2 and all(isinstance(v, int) for v in x)


def _one_stamp(lst) -> Optional[List[int]]:
    if isinstance(lst, list) and len(lst) == 1 and _is_stamp(lst[0]):
        return lst[0]
    return None


def group_runs(payload) -> Tuple[Dict[Tuple[str, int, str], Dict[str, dict]], List[str]]:
    runs: Dict[Tuple[str, int, str], Dict[str, dict]] = {}
    problems: List[str] = []
    for a in payload["assertions"]:
        if a["kind"] != "oracle" or a["seed"] is None:
            continue
        m = _NAME_RE.match(a["name"])
        if not m:
            problems.append(f"unparseable per-run assertion name {a['name']!r}")
            continue
        key = (m.group(1), int(m.group(2)), m.group(3) or NORMAL)
        if key[1] != a["seed"]:
            problems.append(f"{a['name']!r}: seed field {a['seed']} != seed in name")
        slot = runs.setdefault(key, {})
        if m.group(4) in slot:
            problems.append(f"duplicate assertion {a['name']!r}")
        slot[m.group(4)] = a
    return runs, problems


def audit_per_run(payload) -> Tuple[List[str], List[Tuple[str, int, str, str]]]:
    """Return (problems, rule_violations).

    ``problems`` are evidence defects (missing / extra assertions, a recorded
    expectation that differs from the derivation, a status the observation does
    not imply, inconsistent cross-references). ``rule_violations`` are runs whose
    OBSERVED engine behaviour contradicts the rule-derived value; each must be
    recorded as a failed assertion (and only those may be).
    """
    runs, problems = group_runs(payload)
    violations: List[Tuple[str, int, str, str]] = []
    want_keys = {(c, s, k) for c in (AGG, RST) for s in expected_seeds(c) for k in RUN_KINDS[c]}
    for missing in sorted(want_keys - set(runs)):
        problems.append(f"run {missing} has no recorded assertions")
    for extra in sorted(set(runs) - want_keys):
        problems.append(f"unexpected run {extra} in the results")

    ctrl_stamp: Dict[int, Optional[List[int]]] = {}
    for (c, s, k) in sorted(want_keys & set(runs), key=lambda x: (x[0], x[1], x[2] != CTRL)):
        rd = derivation(c, s, k)
        want = expected_assertions(rd)
        got = runs[(c, s, k)]
        for name in sorted(set(want) - set(got)):
            problems.append(f"{_prefix(c, s, k)}{name}: assertion missing")
        for name in sorted(set(got) - set(want)):
            problems.append(f"{_prefix(c, s, k)}{name}: unexpected assertion")
        t1_stamp: Optional[List[int]] = None
        for name in [n for n in want if n in got]:
            ck, val = want[name]
            a = got[name]
            full = _prefix(c, s, k) + name
            exp, obs = a["expected"], a["observed"]
            if ck == "book":
                # Books are compared as per-side multisets of [price, qty, oid]: the emitted
                # lists are not in priority order (priority is asserted by Q and O, not here).
                if _canon_book(exp) != _canon_book(val):
                    problems.append(f"{full}: recorded expected {exp!r} != independent derivation {val!r}")
                ok = _canon_book(obs) == _canon_book(val)
            elif ck == "eq":
                if exp != val:
                    problems.append(f"{full}: recorded expected {exp!r} != independent derivation {val!r}")
                ok = obs == val
            elif ck == "battery":
                if exp != "pass / not_applicable":
                    problems.append(f"{full}: recorded expected {exp!r}")
                ok = (isinstance(obs, dict) and tuple(obs) == BATTERY_IDS
                      and all(v in ("pass", "not_applicable") for v in obs.values()))
            elif ck == "i2b":
                if exp != "pass":
                    problems.append(f"{full}: recorded expected {exp!r}")
                ok = obs == "pass"
            elif ck == "stamp1":
                st = _one_stamp(obs)
                ok = st is not None and st[0] == val["sec"]
                ctrl_stamp[s] = st
            elif ck == "t1":
                ok = (isinstance(obs, list) and [e[0] for e in obs if isinstance(e, list)] == val["labels"]
                      and all(isinstance(e, list) and len(e) == 2 and _one_stamp(e[1]) is not None
                              for e in obs))
                if ok:
                    stamps = [e[1][0] for e in obs]
                    ok = all(x == stamps[0] for x in stamps) and stamps[0][0] == val["sec"]
                    t1_stamp = stamps[0]
            elif ck == "t2":
                ok = _is_stamp(obs) and exp == obs and obs[0] == val["sec"]
                if _is_stamp(exp) and ctrl_stamp.get(s) is not None and exp != ctrl_stamp[s]:
                    problems.append(f"{full}: expected {exp} is not the control's entry stamp "
                                    f"{ctrl_stamp[s]}")
                if ok and t1_stamp is not None and obs != t1_stamp:
                    problems.append(f"{full}: observed {obs} disagrees with the T1 stamp {t1_stamp}")
            elif ck == "t3":
                if exp != "residual strictly earlier (lexicographic), one entry each":
                    problems.append(f"{full}: recorded expected {exp!r}")
                res = _one_stamp(obs.get("residual")) if isinstance(obs, dict) else None
                itl = _one_stamp(obs.get("interloper")) if isinstance(obs, dict) else None
                # §5.2: unique, sequential stamps from one global queue; the residual
                # entered the queue before the interloper, so it is strictly earlier.
                ok = res is not None and itl is not None and tuple(res) < tuple(itl)
                if res is not None and t1_stamp is not None and res != t1_stamp:
                    problems.append(f"{full}: residual stamp {res} disagrees with T1 {t1_stamp}")
                if itl is not None and itl[0] != val["interloper_sec"]:
                    problems.append(f"{full}: interloper stored second {itl[0]} != its entry "
                                    f"second {val['interloper_sec']}")
            else:  # pragma: no cover
                raise AssertionError(ck)
            want_status = "pass" if ok else "fail"
            if a["status"] != want_status:
                problems.append(f"{full}: recorded status {a['status']!r} but the observation "
                                f"{obs!r} implies {want_status!r}")
            if a["unit"] not in ("price_int", "qty_int", "count", "bool", "other"):
                problems.append(f"{full}: unit {a['unit']!r}")
            if not ok:
                violations.append((c, s, k, name))
    return problems, violations


def audit_eager_jit(payload) -> List[str]:
    problems: List[str] = []
    ej = [a for a in payload["assertions"] if a["kind"] == "eager_jit"]
    if not ej:
        return ["no eager-vs-jitted evidence recorded (absence is a failure)"]
    covered = set()
    for a in ej:
        m = _EJ_DETAIL_RE.match(a["detail"])
        if not m:
            problems.append(f"eager_jit detail not 'identical': {a['detail']!r}")
            continue
        if int(m.group(2)) != a["seed"]:
            problems.append(f"eager_jit seed mismatch: {a['seed']} vs {a['detail']!r}")
        if int(m.group(2)) not in expected_seeds(m.group(1)):
            problems.append(f"eager_jit on a seed outside the dataset: {a['detail']!r}")
        if not (a["status"] == "pass" and a["expected"] is True and a["observed"] is True):
            problems.append(f"eager/jitted disagreement recorded: {a['name']} {a['detail']}")
        covered.add((m.group(1), m.group(3) or NORMAL))
    for c in (AGG, RST):
        for k in RUN_KINDS[c]:
            if (c, k) not in covered:
                problems.append(f"no eager-vs-jitted evidence for {c} {k}")
    all_same = all(a["status"] == "pass" for a in ej)
    if payload["eager_jit_identical"] is not True or not all_same:
        problems.append(f"eager_jit_identical={payload['eager_jit_identical']!r} with "
                        f"all-pass={all_same}")
    return problems


def _dataset_counts(construction: str) -> Dict[str, int]:
    out = {"resting BUY": 0, "resting SELL": 0, "q_i > residual": 0, "q_i <= residual": 0}
    for s in expected_seeds(construction):
        rd = derivation(construction, s, NORMAL)
        p = rd.params
        out["resting BUY" if p["rest_side"] == BUY else "resting SELL"] += 1
        residual = abs(int(p["q_agg_int"]) - int(p["q_rest_int"]))
        q_i = rd.qty_of(p["interloper_oid"])
        out["q_i > residual" if q_i > residual else "q_i <= residual"] += 1
    return out


GLOBAL_NAMES = {
    "section 5: closed form agrees with the walk (W) on every run of the first seeds",
    "section 6 wired (run_tape defined)",
    f"dataset: every {AGG} draw is as section 4 describes (precondition)",
    f"dataset: every {RST} draw is as section 4 describes (precondition)",
    "section 5: closed form agrees with the walk (W) on every run",
    "dataset: every derived tape is within §2.3's bounds and the capacity margin",
    "dataset separates the rule from the re-stamp model on every draw and same_second run",
    f"dataset: {AGG} covers both resting sides and an interloper larger / not larger than the residual",
    f"dataset: {RST} covers both resting sides and an interloper larger / not larger than the residual",
}


def audit_global(payload) -> List[str]:
    problems: List[str] = []
    glob = [a for a in payload["assertions"] if a["kind"] == "oracle" and a["seed"] is None]
    names = [a["name"] for a in glob]
    if sorted(names) != sorted(GLOBAL_NAMES):
        problems.append(f"global assertions {sorted(names)} != {sorted(GLOBAL_NAMES)}")
    for a in glob:
        if a["status"] != "pass":
            problems.append(f"global assertion failed: {a['name']}")
        for c in (AGG, RST):
            if a["name"] == f"dataset: {c} covers both resting sides and an interloper larger / not " \
                            "larger than the residual":
                if a["observed"] != _dataset_counts(c):
                    problems.append(f"{a['name']}: observed {a['observed']} != generator counts "
                                    f"{_dataset_counts(c)}")
                if any(v < 1 for v in a["observed"].values()):
                    problems.append(f"{a['name']}: a coverage cell is empty")
    return problems


def audit_status(payload, violations, *, schema: bool = True) -> List[str]:
    """Status / failure_kinds / failing seeds follow from the payload and the rules.

    ``schema=False`` skips the (slow, ~0.8 s) JSON-Schema pass; the semantic checks
    below and ``derive_failure_kinds`` still run."""
    problems = list(validate_results(payload)) if schema else []
    kinds = derive_failure_kinds(payload)
    if payload["failure_kinds"] != kinds:
        problems.append(f"failure_kinds {payload['failure_kinds']} != derived {kinds}")
    failed = [a for a in payload["assertions"] if a["status"] == "fail"]
    failed_seeds = sorted({a["seed"] for a in failed if a["seed"] is not None})
    if payload["seeds"]["failing"] != failed_seeds:
        problems.append(f"seeds.failing has {len(payload['seeds']['failing'])} entries, "
                        f"failed assertions name {len(failed_seeds)} seeds")
    rule_says_fail = bool(violations)
    if rule_says_fail != (payload["status"] == "NOT CERTIFIED"):
        problems.append(f"status {payload['status']!r} but the rule-derived violations are "
                        f"{len(violations)}")
    if rule_says_fail != ("oracle_mismatch" in payload["failure_kinds"]):
        problems.append("oracle_mismatch in failure_kinds disagrees with the rule-derived violations")
    fe = payload["failure_explanation"]
    if payload["status"] == "NOT CERTIFIED":
        if fe is None:
            problems.append("NOT CERTIFIED with no failure_explanation")
        else:
            sc = fe.get("smallest_case") or {}
            if failed_seeds and sc.get("seed") != failed_seeds[0]:
                problems.append(f"smallest_case.seed {sc.get('seed')} != smallest failing seed "
                                f"{failed_seeds[0]}")
            if sc.get("seed") is not None:
                c = AGG if sc["seed"] < RESTING_SEED_OFFSET else RST
                n_rows = len(derivation(c, sc["seed"], NORMAL).rows)
                if sc.get("rows") != n_rows:
                    problems.append(f"smallest_case.rows {sc.get('rows')} != {n_rows}")
            first = failed[0]["name"] if failed else None
            if first and first not in fe["what_failed"]:
                problems.append("what_failed does not name the first failing assertion")
    elif fe is not None:
        problems.append("PASS with a failure_explanation")
    return problems


def audit_battery_and_capacity(payload) -> List[str]:
    problems: List[str] = []
    inv = payload["invariants"]
    if [r["check_id"] for r in inv] != list(BATTERY_IDS):
        problems.append(f"battery rows {[r['check_id'] for r in inv]} != {list(BATTERY_IDS)}")
    valid = payload["draws"]["valid"]
    for r in inv:
        if r["status"] != "pass" or r["n_fail"] or r["failing_seeds"]:
            problems.append(f"battery {r['check_id']} not clean: {r}")
        if r["n_pass"] + r["n_fail"] + r["n_not_applicable"] != valid:
            problems.append(f"battery {r['check_id']} counts do not cover the {valid} valid draws")
    by_id = {r["check_id"]: r for r in inv}
    for must_run in ("I2", "I4", "I8", "I9"):
        if by_id.get(must_run, {}).get("n_pass") != valid:
            problems.append(f"{must_run} did not pass on every valid draw")
    cap = payload["capacity"]
    if cap["envelope"] != ENVELOPE:
        problems.append(f"capacity envelope {cap['envelope']} != {ENVELOPE}")
    # Exact peaks from the rules-only reference over every run of every draw.
    peak_rest, peak_trades = 0, 0
    for c in (AGG, RST):
        for s in expected_seeds(c):
            for k in RUN_KINDS[c]:
                ref = derivation(c, s, k).ref
                peak_trades = max(peak_trades, len(ref["trades"]))
                for b in ref["post"]:
                    peak_rest = max(peak_rest, len(b[BUY]), len(b[SELL]))
    peaks = cap["peaks"]
    if peaks["resting_per_side"] != peak_rest:
        problems.append(f"peak resting_per_side {peaks['resting_per_side']} != derived {peak_rest}")
    if peaks["trades"] != peak_trades:
        problems.append(f"peak trades {peaks['trades']} != derived {peak_trades}")
    if peaks["concurrent_icebergs"] != 0:
        problems.append(f"peak concurrent_icebergs {peaks['concurrent_icebergs']} != 0 (NON-only)")
    if peaks["hops"] is None:
        if "matching_iterations" not in by_id.get("I8", {}).get("unobserved", []):
            problems.append("peaks.hops is null but I8 does not report matching_iterations unobserved")
    elif peaks["hops"] > ENVELOPE["hops"]:
        problems.append(f"peak hops {peaks['hops']} > {ENVELOPE['hops']}")
    for axis, lim in ENVELOPE.items():
        v = peaks[axis]
        if v is not None and v > lim:
            problems.append(f"capacity breach on {axis}: {v} > {lim}")
    return problems


def audit_all(payload, *, schema: bool = True) -> List[str]:
    problems, violations = audit_per_run(payload)
    problems += audit_eager_jit(payload)
    problems += audit_global(payload)
    problems += audit_status(payload, violations, schema=schema)
    problems += audit_battery_and_capacity(payload)
    return problems


# ---------------------------------------------------------------------------
# Tests on the emitted JSON
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def per_run(payload):
    return audit_per_run(payload)


class TestIdentityAndSchema:
    def test_schema_and_semantic_validator_accept_the_payload(self, payload):
        assert validate_results(payload) == []

    def test_failure_kinds_are_derived_from_the_payload(self, payload):
        assert payload["failure_kinds"] == derive_failure_kinds(payload)
        assert (payload["status"] == "PASS") == (payload["failure_kinds"] == [])

    def test_identity_fields(self, payload):
        assert payload["cert_id"] == "C05"
        assert payload["notebook"] == "C05_partial_fill_residual.ipynb"
        assert payload["oracle_kind"] == "exact-parametric"
        assert payload["generator"]["module"] == "omiecid_simulator.jaxomiecid.omie_tape_synth"

    def test_generator_digests_are_the_current_constructions(self, payload):
        digests = payload["generator"]["digests"]
        assert set(digests) == {AGG, RST}
        for key in (AGG, RST):
            assert digests[key] == construction_digest(key), key

    def test_draws_and_seeds_cover_500_per_construction_with_no_discard(self, payload):
        d = payload["draws"]
        assert d == {"requested": 2 * DRAWS_PER_CONSTRUCTION, "valid": 2 * DRAWS_PER_CONSTRUCTION,
                     "discarded": 0, "discard_rate": 0.0}
        seeds = sorted({a["seed"] for a in payload["assertions"]
                        if a["kind"] == "oracle" and a["seed"] is not None})
        assert seeds == expected_seeds(AGG) + expected_seeds(RST)
        assert payload["seeds"]["count"] == len(seeds)
        assert payload["seeds"]["first"] == seeds[0]
        assert payload["seeds"]["last"] == seeds[-1]

    def test_every_draw_meets_the_capacity_margin_by_construction(self, payload):
        for c in (AGG, RST):
            for s in expected_seeds(c):
                cb = _draw(c, s).capacity_bounds
                assert cb["hops"] <= ENVELOPE["hops"] and cb["trades"] <= ENVELOPE["trades"]
                assert cb["resting_per_side"] <= ENVELOPE["resting_per_side"]


class TestDatasetIsTheCardsDataset:
    """The two §5.2 cases the card requires, each with an interloper at the residual's price."""

    def test_aggressor_draws_exceed_crossing_depth_and_cross(self, payload):
        for s in expected_seeds(AGG):
            rd = derivation(AGG, s, NORMAL)
            p = rd.params
            assert p["q_agg_int"] > p["q_rest_int"] > 0          # q_agg > total crossing depth
            assert p["residual_oid"] == p["agg_oid"]
            itl = next(r for r in rd.rows if r.oid == p["interloper_oid"])
            assert itl.side == p["agg_side"] and itl.price == p["p_agg_int"]

    def test_resting_draws_are_smaller_aggressors(self, payload):
        for s in expected_seeds(RST):
            rd = derivation(RST, s, NORMAL)
            p = rd.params
            assert 0 < p["q_agg_int"] < p["q_rest_int"]
            assert p["residual_oid"] == p["rest_oid"]
            itl = next(r for r in rd.rows if r.oid == p["interloper_oid"])
            assert itl.side == p["rest_side"] and itl.price == p["p_rest_int"]

    def test_global_dataset_assertions(self, payload):
        assert audit_global(payload) == []


class TestPerRunValues:
    def test_every_run_carries_its_full_assertion_set_and_values_match_the_derivation(self, per_run):
        problems, _ = per_run
        assert problems == [], "\n".join(problems[:20])

    def test_known_result_closed_form_on_every_normal_run(self, payload):
        """(i) filled = crossing depth; residual = q_agg - filled at the aggressor's limit.
        (ii) the resting order keeps q_rest - q_agg at its own limit. Exact integers."""
        runs, _ = group_runs(payload)
        for c in (AGG, RST):
            for s in expected_seeds(c):
                p = derivation(c, s, NORMAL).params
                got = runs[(c, s, NORMAL)]
                owner = p["residual_oid"]
                if c == AGG:
                    fill, resid, price = p["q_rest_int"], p["q_agg_int"] - p["q_rest_int"], p["p_agg_int"]
                else:
                    fill, resid, price = p["q_agg_int"], p["q_rest_int"] - p["q_agg_int"], p["p_rest_int"]
                assert got["(F) quantity traded at the partial fill (submission 1)"]["observed"] == fill
                assert got[f"(R) residual quantity of oid {owner} after the partial fill"]["observed"] == resid
                assert got[f"(R) residual resting price(s) of oid {owner} after the partial fill "
                           "(its own limit)"]["observed"] == [price]
                # the take fills the residual only; the interloper stays whole
                assert got["(O) take's passive oids / interloper's filled qty_int"]["observed"] == [[owner], 0]

    def test_partial_fill_trades_at_the_resting_price_and_take_at_the_residual_price(self, payload):
        runs, _ = group_runs(payload)
        for c in (AGG, RST):
            for s in expected_seeds(c)[:50]:
                p = derivation(c, s, NORMAL).params
                q = runs[(c, s, NORMAL)][
                    "(Q) counterparty sequence [passive, aggressor, price_int, qty_int, submission]"]["observed"]
                assert q[0][0] == p["rest_oid"] and q[0][2] == p["p_rest_int"]          # maker price
                assert q[1][0] == p["residual_oid"] and q[1][1] == p["take_oid"]
                assert q[1][2] == (p["p_agg_int"] if c == AGG else p["p_rest_int"])


class TestTimestamps:
    """§5.2: no new timestamp on partial execution; one global queue, unique and sequential."""

    def test_T1_residual_stamp_never_changes_and_is_the_original_entry_second(self, payload):
        runs, _ = group_runs(payload)
        for c in (AGG, RST):
            owner_name = None
            for s in expected_seeds(c):
                p = derivation(c, s, NORMAL).params
                owner_name = (f"(T1) stored [sec, nsec] of oid {p['residual_oid']} unchanged from its "
                              "first rest to the take")
                for k in (NORMAL, SAME):
                    a = runs[(c, s, k)][owner_name]
                    stamps = [e[1][0] for e in a["observed"]]
                    assert all(len(e[1]) == 1 for e in a["observed"]), a
                    assert len(set(map(tuple, stamps))) == 1, a
                    assert stamps[0][0] == p["residual_original_timestamp"], a
                    assert a["status"] == "pass"

    def test_T2_aggressor_residual_stamp_equals_its_unmatched_entry_stamp(self, payload):
        runs, _ = group_runs(payload)
        t2 = ("(T2) residual's stored [sec, nsec] after the partial fill == the aggressor's entry stamp "
              "in the no_cross_control")
        for s in expected_seeds(AGG):
            ctrl = runs[(AGG, s, CTRL)]["the aggressor holds exactly one book entry after submission 1"]
            assert len(ctrl["observed"]) == 1
            for k in (NORMAL, SAME):
                a = runs[(AGG, s, k)][t2]
                assert a["observed"] == a["expected"] == ctrl["observed"][0]

    def test_T3_status_is_exactly_the_rule_on_the_observation(self, per_run):
        problems, violations = per_run
        t3 = [p for p in problems if "(T3)" in p]
        assert t3 == []

    def test_rule_violations_are_recorded_failures_and_nothing_else_fails(self, payload, per_run):
        _, violations = per_run
        recorded = set()
        for a in payload["assertions"]:
            if a["status"] == "fail":
                m = _NAME_RE.match(a["name"])
                assert m, f"failed non-per-run assertion {a['name']}"
                recorded.add((m.group(1), int(m.group(2)), m.group(3) or NORMAL, m.group(4)))
        assert recorded == set(violations)

    def test_no_run_violates_the_rule(self, per_run):
        """r2: the fixes have merged, so the rule-derived violation set must be empty
        (r1 recorded 500 (T3) same_second stamp collisions)."""
        _, violations = per_run
        assert violations == [], sorted(violations)[:5]

    def test_T3_same_second_stamps_are_distinct_and_in_arrival_order(self, payload):
        """§5.2 'unique, sequential' on the stored field: on every same_second run the
        residual and the interloper hold DISTINCT stamps, the residual's strictly earlier
        (it entered the single global queue first)."""
        runs, _ = group_runs(payload)
        t3 = "(T3) just before the take, residual's [sec, nsec] < interloper's"
        for c in (AGG, RST):
            for s in expected_seeds(c):
                rd = derivation(c, s, SAME)
                obs = runs[(c, s, SAME)][t3]["observed"]
                res, itl = _one_stamp(obs["residual"]), _one_stamp(obs["interloper"])
                assert res is not None and itl is not None, (c, s, obs)
                assert res != itl, (c, s, obs)
                assert tuple(res) < tuple(itl), (c, s, obs)
                assert itl[0] == rd.ts_of(rd.params["interloper_oid"]), (c, s, obs)
                if c == AGG:
                    # the interloper shares the residual's own entry second: the collision case
                    assert res[0] == itl[0], (c, s, obs)

    def test_no_recorded_failure_anywhere(self, payload):
        assert [a["name"] for a in payload["assertions"] if a["status"] != "pass"] == []


class TestStatusConsistency:
    def test_status_failure_kinds_and_explanation_follow_from_payload_and_rules(self, payload, per_run):
        _, violations = per_run
        problems = audit_status(payload, violations)
        assert problems == [], problems

    def test_rule_implied_verdict(self, payload, per_run):
        _, violations = per_run
        expected_status = "NOT CERTIFIED" if violations else "PASS"
        assert payload["status"] == expected_status


class TestEagerJit:
    def test_eager_vs_jitted_evidence_present_for_every_construction_and_run_kind_and_identical(
            self, payload):
        assert audit_eager_jit(payload) == []


class TestBatteryAndCapacity:
    def test_full_battery_runs_clean_and_capacity_peaks_are_exact(self, payload):
        assert audit_battery_and_capacity(payload) == []

    def test_per_run_battery_records_cover_every_check(self, payload):
        runs, _ = group_runs(payload)
        for (c, s, k), got in runs.items():
            if k == NORMAL:
                assert got["(I2) battery I2 status"]["observed"] == "pass"
            else:
                assert tuple(got["battery (RB, I1-I9) passes"]["observed"]) == BATTERY_IDS


class TestQuotesAndCitations:
    REQUIRED_SECTIONS = {("market-spec.md", "§5.1"), ("market-spec.md", "§5.2"),
                         ("market-spec.md", "§6.2 item 4"), ("scenario-ladder-review.md", "§8")}

    def test_emitted_quotes_all_ok_and_cover_the_card(self, payload):
        quotes = payload["spec"]["quotes"]
        assert quotes and all(q["ok"] and q["reason"] == "ok" for q in quotes)
        have = {(q["source"], q["section"]) for q in quotes}
        assert self.REQUIRED_SECTIONS <= have, self.REQUIRED_SECTIONS - have

    @pytest.mark.parametrize("source,section,text", [
        ("market-spec.md", "§5.2",
         "Partial execution is NOT a modification and does NOT assign a new timestamp."),
        ("market-spec.md", "§5.2",
         "The unexecuted residual of a NON incoming order enters the book with its original entry "
         "timestamp (same order, not modified). Likewise a resting order that is partially executed "
         "keeps its timestamp."),
        ("market-spec.md", "§5.2",
         "All timestamps across orders and explicit capacity requests are assigned from a single "
         "global queue (unique, sequential)."),
        ("market-spec.md", "§6.1", "ranked lowest price first; ties broken by earliest timestamp."),
        ("market-spec.md", "§6.2 item 3", "Trade executes at the resting order's limit price (maker pricing)."),
        ("market-spec.md", "§6.2 item 4",
         "If incoming order is partially filled and has NON restriction: residual enters the book with "
         "its original entry timestamp (see §5.2)."),
        ("scenario-ladder-review.md", "§8",
         "Resolved — user ruling, 2026-09-23: the residual keeps its time priority."),
    ])
    def test_the_rules_this_audit_relies_on_are_current(self, source, section, text):
        res = check_quote(SpecQuote(source=source, section=section, text=text))
        assert res.ok, (source, section, res.reason)

    def test_spec_version_pinned_v2_14(self, payload):
        """The r2 re-run is judged against market-spec v2.14: the document and every
        emitted market-spec quote carry that version."""
        text = (REPO_ROOT / "docs" / "omiecid" / "market-spec.md").read_text(encoding="utf-8")
        assert re.search(r"^\*\*Status:\*\*\s*v2\.14\b", text, re.M), "market-spec.md is not v2.14"
        ms = [q for q in payload["spec"]["quotes"] if q["source"] == "market-spec.md"]
        assert ms and {q.get("version") for q in ms} == {SPEC_VERSION}


class TestCounterexampleFixture:
    def test_fixture_is_the_smallest_failing_draw(self, payload):
        fe = payload["failure_explanation"]
        if fe is None or not fe.get("smallest_case") or not fe["smallest_case"].get("fixture"):
            pytest.skip("no counterexample fixture recorded (status PASS or none frozen)")
        path = Path(fe["smallest_case"]["fixture"])
        if not path.is_file():
            pytest.skip(f"counterexample fixture {path} not present on this machine")
        z = np.load(path, allow_pickle=False)
        seed = fe["smallest_case"]["seed"]
        c = AGG if seed < RESTING_SEED_OFFSET else RST
        draw = _draw(c, seed)
        real = [r for r in z["tape"] if int(r[COL_OID]) >= 1]
        assert len(real) == fe["smallest_case"]["rows"]
        assert np.array_equal(np.asarray(real), np.asarray([r for r in draw.tape if int(r[COL_OID]) >= 1]))
        params = json.loads(str(z["params_json"]))
        for k, v in draw.params.items():
            if isinstance(v, int):
                assert params[k] == v, k


# ---------------------------------------------------------------------------
# Deliberately broken copies (tmp_path) — the audit must reject each
# ---------------------------------------------------------------------------
def _write_broken(tmp_path: Path, payload, mutate) -> Dict[str, Any]:
    p = dict(payload)
    p["assertions"] = list(payload["assertions"])
    mutate(p)
    path = tmp_path / "C05.json"
    path.write_text(json.dumps(p), encoding="utf-8")
    return _load(path)


def _replace(p, name: str, **changes):
    for i, a in enumerate(p["assertions"]):
        if a["name"] == name:
            b = dict(a)
            b.update(changes)
            p["assertions"][i] = b
            return
    raise KeyError(name)


def _get(payload, name: str) -> dict:
    return next(a for a in payload["assertions"] if a["name"] == name)


class TestBrokenCopiesAreRejected:
    SEED = 5

    def _names(self):
        p = derivation(AGG, self.SEED, NORMAL).params
        pre = _prefix(AGG, self.SEED, NORMAL)
        o = p["residual_oid"]
        return p, pre, o

    def test_restamped_residual(self, payload, tmp_path):
        """The residual re-submitted with a fresh stamp (the card's 'Failure implicates')."""
        p, pre, o = self._names()
        t1 = pre + f"(T1) stored [sec, nsec] of oid {o} unchanged from its first rest to the take"
        t3 = pre + "(T3) just before the take, residual's [sec, nsec] < interloper's"
        orig = _get(payload, t1)["observed"]
        fresh = [orig[0][1][0][0] + 2, 0]          # a later counter second, after the interloper
        restamped = [orig[0]] + [[lbl, [fresh]] for lbl, _ in orig[1:]]

        def mutate(q):
            _replace(q, t1, observed=restamped)                     # status left 'pass'
            obs = dict(_get(payload, t3)["observed"])
            obs["residual"] = [fresh]
            _replace(q, t3, observed=obs)
        broken = _write_broken(tmp_path, payload, mutate)
        problems = audit_all(broken, schema=False)
        assert any("(T1)" in x and "implies 'fail'" in x for x in problems), problems[:5]
        assert any("(T3)" in x for x in problems), problems[:5]

    def test_residual_at_the_wrong_price(self, payload, tmp_path):
        """The residual rests at the price it traded at (p_rest), not its own limit."""
        p, pre, o = self._names()
        rp = pre + f"(R) residual resting price(s) of oid {o} after the partial fill (its own limit)"
        b1 = pre + "(B1) book after the partial fill [price_int, qty_int, oid]"
        wrong = p["p_rest_int"]

        def mutate(q):
            _replace(q, rp, expected=[wrong], observed=[wrong])     # forger edits both
            book = json.loads(json.dumps(_get(payload, b1)["observed"]))
            for side in book.values():
                for e in side:
                    if e[2] == o:
                        e[0] = wrong
            _replace(q, b1, expected=book, observed=book)
        problems = audit_all(_write_broken(tmp_path, payload, mutate), schema=False)
        assert any("(R) residual resting price" in x and "independent derivation" in x for x in problems)
        assert any("(B1)" in x and "independent derivation" in x for x in problems)

    def test_interloper_filled_first(self, payload, tmp_path):
        p, pre, o = self._names()
        itl = p["interloper_oid"]
        on = pre + "(O) take's passive oids / interloper's filled qty_int"
        qn = pre + "(Q) counterparty sequence [passive, aggressor, price_int, qty_int, submission]"
        resid = p["q_agg_int"] - p["q_rest_int"]

        def mutate(q):
            _replace(q, on, expected=[[itl], resid], observed=[[itl], resid])
            seq = json.loads(json.dumps(_get(payload, qn)["observed"]))
            seq[1][0] = itl
            _replace(q, qn, expected=seq, observed=seq)
        problems = audit_all(_write_broken(tmp_path, payload, mutate), schema=False)
        assert any("(O)" in x and "independent derivation" in x for x in problems)
        assert any("(Q)" in x and "independent derivation" in x for x in problems)

    def test_interloper_filled_first_with_honest_status(self, payload, tmp_path):
        """Same defect, but the forger leaves expected alone: observed != derivation and status 'pass'."""
        p, pre, o = self._names()
        on = pre + "(O) take's passive oids / interloper's filled qty_int"

        def mutate(q):
            _replace(q, on, observed=[[p["interloper_oid"]], 1])
        problems = audit_all(_write_broken(tmp_path, payload, mutate), schema=False)
        assert any("(O)" in x and "implies 'fail'" in x for x in problems)

    def test_forged_pass(self, payload, tmp_path):
        """Every failure flipped to pass with a consistent PASS envelope: the schema validator
        accepts it, the rule on the observation does not."""
        def mutate(q):
            for i, a in enumerate(q["assertions"]):
                if a["status"] == "fail":
                    b = dict(a)
                    b["status"] = "pass"
                    q["assertions"][i] = b
            q["status"] = "PASS"
            q["failure_kinds"] = []
            q["failure_explanation"] = None
            q["seeds"] = dict(q["seeds"], failing=[])
        broken = _write_broken(tmp_path, payload, mutate)
        if not any(a["status"] == "fail" for a in payload["assertions"]):
            pytest.skip("emitted payload has no failure to forge away")
        problems = audit_all(broken, schema=False)
        assert any("implies 'fail'" in x for x in problems)
        assert any(x.startswith("status 'PASS'") for x in problems)

    def test_missing_eager_jit_evidence(self, payload, tmp_path):
        def mutate(q):
            q["assertions"] = [a for a in q["assertions"] if a["kind"] != "eager_jit"]
        problems = audit_all(_write_broken(tmp_path, payload, mutate), schema=False)
        assert any("absence is a failure" in x for x in problems)

    def test_eager_jit_evidence_missing_for_one_run_kind(self, payload, tmp_path):
        def mutate(q):
            q["assertions"] = [a for a in q["assertions"]
                               if not (a["kind"] == "eager_jit" and "same_second" in a["detail"]
                                       and a["detail"].startswith(RST))]
        problems = audit_all(_write_broken(tmp_path, payload, mutate), schema=False)
        assert any(f"no eager-vs-jitted evidence for {RST} same_second" in x for x in problems)

    def test_hidden_eager_jit_disagreement(self, payload, tmp_path):
        def mutate(q):
            for i, a in enumerate(q["assertions"]):
                if a["kind"] == "eager_jit":
                    q["assertions"][i] = dict(a, observed=False)
                    return
        problems = audit_all(_write_broken(tmp_path, payload, mutate), schema=False)
        assert any("eager/jitted disagreement" in x for x in problems)

    def test_dropped_per_run_assertion(self, payload, tmp_path):
        p, pre, o = self._names()
        t3 = pre + "(T3) just before the take, residual's [sec, nsec] < interloper's"

        def mutate(q):
            q["assertions"] = [a for a in q["assertions"] if a["name"] != t3]
        problems = audit_all(_write_broken(tmp_path, payload, mutate), schema=False)
        assert any(x.endswith("(T3) just before the take, residual's [sec, nsec] < interloper's: "
                              "assertion missing") for x in problems)

    def test_partial_fill_quantity_off_by_one_tick(self, payload, tmp_path):
        p, pre, o = self._names()
        fn = pre + "(F) quantity traded at the partial fill (submission 1)"

        def mutate(q):
            v = _get(payload, fn)["observed"] + 1
            _replace(q, fn, expected=v, observed=v)
        problems = audit_all(_write_broken(tmp_path, payload, mutate), schema=False)
        assert any("(F)" in x and "independent derivation" in x for x in problems)


    def test_r1_stamp_collision_with_pass_status(self, payload, tmp_path):
        """The r1 defect re-introduced: residual and interloper store the SAME stamp on a
        same_second run, status left 'pass'. §5.2 'unique, sequential' must reject it."""
        t3 = _prefix(AGG, self.SEED, SAME) + "(T3) just before the take, residual's [sec, nsec] < interloper's"

        def mutate(q):
            obs = json.loads(json.dumps(_get(payload, t3)["observed"]))
            obs["interloper"] = [list(obs["residual"][0])]
            _replace(q, t3, observed=obs)
        problems = audit_all(_write_broken(tmp_path, payload, mutate), schema=False)
        assert any("(T3)" in x and "implies 'fail'" in x for x in problems), problems[:5]
        assert any(x.startswith("status 'PASS'") for x in problems), problems[:5]

    def test_forged_failure_on_a_passing_run(self, payload, tmp_path):
        """A pass flipped to fail is equally a mis-recorded status."""
        name = (_prefix(RST, RESTING_SEED_OFFSET + self.SEED, NORMAL)
                + "(F) quantity traded at the partial fill (submission 1)")

        def mutate(q):
            _replace(q, name, status="fail")
        problems = audit_all(_write_broken(tmp_path, payload, mutate), schema=False)
        assert any("(F)" in x and "implies 'pass'" in x for x in problems), problems[:5]


# ---------------------------------------------------------------------------
# Slow: re-run a few seeds through the public record builder
# ---------------------------------------------------------------------------
def _tape_for(construction: str, seed: int, kind: str) -> np.ndarray:
    draw = _draw(construction, seed)
    rows = {r.oid: r for r in derivation(construction, seed, kind).rows}
    tape = np.array(draw.tape, copy=True)
    for i in range(tape.shape[0]):
        oid = int(tape[i, COL_OID])
        if oid >= 1:
            tape[i, COL_PRICE] = rows[oid].price
            tape[i, COL_TS] = rows[oid].ts
    return tape


def _live_book(snap):
    from tests.omiecid.reference.omie_invariants import BUY as IB, SELL as IS
    return {"BUY": sorted([list(r[:3]) for r in snap.resting(IB)]),
            "SELL": sorted([list(r[:3]) for r in snap.resting(IS)])}


def _stamps(snap, oid):
    from tests.omiecid.reference.omie_invariants import BUY as IB, SELL as IS
    return [[r[4], r[5]] for side in (IB, IS) for r in snap.resting(side) if r[2] == oid]


@pytest.mark.slow
@pytest.mark.jit
def test_rerun_reproduces_the_recorded_observations(payload):
    """slow: ~10 engine runs (eager + jitted) through build_record; confirms the JSON is
    reproducible evidence, including the recorded (distinct) T3 stamps."""
    from tests.omiecid.reference.omie_invariants import build_record, compare_records

    runs, _ = group_runs(payload)
    for c, s in ((AGG, 0), (AGG, 7), (RST, RESTING_SEED_OFFSET), (RST, RESTING_SEED_OFFSET + 7)):
        draw = _draw(c, s)
        p = draw.params
        for k in RUN_KINDS[c]:
            tape = _tape_for(c, s, k)
            rec = build_record(tape, capacity_bounds=dict(draw.capacity_bounds))
            if k != CTRL:
                eager = build_record(tape, capacity_bounds=dict(draw.capacity_bounds), jit=False)
                assert compare_records(rec, eager) == (), (c, s, k)
            got = runs[(c, s, k)]
            q = [[f.passive_oid, f.aggressor_oid, f.price_int, f.qty_int, f.submission] for f in rec.fills]
            qname = ("(Q) counterparty sequence" if k == CTRL else
                     "(Q) counterparty sequence [passive, aggressor, price_int, qty_int, submission]")
            assert q == got[qname]["observed"], (c, s, k)
            assert q == derivation(c, s, k).ref["trades"]
            endname = "(B) book at the end" if k == CTRL else "(B) book at the end [price_int, qty_int, oid]"
            want_end = {sd: sorted(v) for sd, v in got[endname]["observed"].items()}
            assert _live_book(rec.final_snapshot) == want_end
            if k == CTRL:
                continue
            owner, itl = p["residual_oid"], p["interloper_oid"]
            t3 = got["(T3) just before the take, residual's [sec, nsec] < interloper's"]["observed"]
            assert _stamps(rec.pre_snapshots[3], owner) == t3["residual"]
            assert _stamps(rec.pre_snapshots[3], itl) == t3["interloper"]
            assert all(f.passive_oid == owner for f in rec.fills if f.submission == 3)


@pytest.mark.slow
@pytest.mark.jit
def test_resting_residual_fills_before_an_interloper_in_its_own_second():
    """slow: engine runs. Coverage the emitted dataset lacks: in the resting case the
    same_second interloper shares the AGGRESSOR's second, never the residual's own.
    Here resting order, aggressor and interloper all share one tape second; the
    partially executed resting order must still fill ahead of the interloper (§5.2, §6.1)."""
    from tests.omiecid.reference.omie_invariants import build_record

    for s in (RESTING_SEED_OFFSET, RESTING_SEED_OFFSET + 3):
        draw = _draw(RST, s)
        p = draw.params
        tape = np.array(draw.tape, copy=True)
        sec = int(tape[0, COL_TS])
        for i in range(3):
            tape[i, COL_TS] = sec
        rec = build_record(tape, capacity_bounds=dict(draw.capacity_bounds))
        take = [f for f in rec.fills if f.submission == 3]
        assert [f.passive_oid for f in take] == [p["residual_oid"]]
        assert sum(f.qty_int for f in take) == p["q_rest_int"] - p["q_agg_int"]


@pytest.mark.slow
@pytest.mark.jit
def test_stored_stamps_are_unique_when_residual_and_interloper_share_a_second():
    """slow: engine run. §5.2 'unique, sequential': the residual's stored stamp is
    distinct from, and strictly earlier than, a same-second interloper's, in the
    resting-residual case as well (r1 strict xfail, now a plain assertion)."""
    from tests.omiecid.reference.omie_invariants import build_record

    draw = _draw(RST, RESTING_SEED_OFFSET)
    p = draw.params
    tape = np.array(draw.tape, copy=True)
    for i in range(3):
        tape[i, COL_TS] = int(tape[0, COL_TS])
    rec = build_record(tape, capacity_bounds=dict(draw.capacity_bounds))
    res = _stamps(rec.pre_snapshots[3], p["residual_oid"])
    itl = _stamps(rec.pre_snapshots[3], p["interloper_oid"])
    assert len(res) == len(itl) == 1
    assert res[0] != itl[0]
    assert tuple(res[0]) < tuple(itl[0])


# ---------------------------------------------------------------------------
# Independent engine probes through ContractBook's public API (jitted submit path).
# Hand-built rows, expectation from §5.2 / §6.1 / §6.2 only.
# ---------------------------------------------------------------------------
_PROBE_CID = 47


def _probe_row(side, price_int, qty_int, oid, ts):
    r = np.zeros(13, dtype=np.int32)
    r[0], r[COL_SIDE], r[COL_PRICE], r[COL_QTY] = _PROBE_CID, side, price_int, qty_int
    r[COL_OID], r[5], r[COL_TS], r[COL_EXEC] = oid, oid % 97 + 1, ts, NON
    return r


class _Probe:
    """One ContractBook driven through submits and cancels (public API only)."""

    def __init__(self):
        from omiecid_simulator.omiecid.contract import Contract, ContractStatus
        from omiecid_simulator.omiecid.contract_book import ContractBook
        from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
        contract = Contract(contract_id=_PROBE_CID, delivery_start_minutes=555,
                            delivery_end_minutes=570, gct_minutes=495, status=ContractStatus.OPEN)
        self.pc = OMIEProductConfig()
        self.book = ContractBook(contract, self.pc, OMIEMarketConfig())
        self.state = self.book.init()

    def submit(self, row):
        import jax.numpy as jnp
        from omiecid_simulator.jaxomiecid.omie_data_loader import _row_to_order_traced
        self.state, res = self.book.submit_order(self.state, _row_to_order_traced(jnp.asarray(row), self.pc))
        return res

    def cancel(self, oid, side):
        self.state = self.book.cancel_order(self.state, oid, side)

    def resting(self, side) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """oid -> [(price, qty, sec, nsec), ...] (one entry per stored row)."""
        from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat as F
        a = np.asarray(self.state.lob.bids if side == BUY else self.state.lob.asks)
        a = a[(a[:, F.OID.value] > 0) & (a[:, F.Q.value] > 0)]
        out: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for r in a:
            out.setdefault(int(r[F.OID.value]), []).append(
                (int(r[F.P.value]), int(r[F.Q.value]), int(r[F.SEC.value]), int(r[F.NSEC.value])))
        return out

    def trades(self) -> List[Tuple[int, int, int, int]]:
        """[(passive, aggressor, price, qty)] in print order."""
        from omiecid_simulator.jaxob.jaxob_constants import TradesFeat as T
        t = np.asarray(self.state.lob.trades)
        t = t[t[:, T.P.value] > 0]
        return [(int(r[T.PASS_OID.value]), int(r[T.AGRS_OID.value]), int(r[T.P.value]),
                 abs(int(r[T.Q.value]))) for r in t]


def _stamp(entry) -> Tuple[int, int]:
    return (entry[2], entry[3])


@pytest.mark.jit
@pytest.mark.parametrize("rest_side", [BUY, SELL])
@pytest.mark.parametrize("decoy_cancel", [False, True])
def test_probe_aggressor_residual_beats_same_second_interloper(rest_side, decoy_cancel):
    """§6.2 item 4 + §5.2: the aggressor's NON residual rests at its own limit with its entry
    second; a later interloper entered in the SAME second at the SAME price must store a
    distinct, later stamp and fill after it. With ``decoy_cancel`` an earlier opposite-to-r
    order is cancelled first, freeing a book slot the interloper can land in (slot order
    must not leak into time priority)."""
    agg_side = SELL if rest_side == BUY else BUY
    p_rest = 1004519 if rest_side == BUY else 1004469
    p_agg = 1004469 if rest_side == BUY else 1004519
    p_decoy = 1005000 if agg_side == SELL else 1004000      # on the aggressor's side, never crosses
    q_rest, q_agg, q_itl = 27, 52, 166
    pr = _Probe()
    if decoy_cancel:
        pr.submit(_probe_row(agg_side, p_decoy, 10, 5, 1))
    pr.submit(_probe_row(rest_side, p_rest, q_rest, 1, 2))
    pr.submit(_probe_row(agg_side, p_agg, q_agg, 2, 3))
    # partial fill at the maker's price for the crossing depth; residual at the aggressor's limit
    assert pr.trades() == [(1, 2, p_rest, q_rest)]
    resid = pr.resting(agg_side)[2]
    assert [(e[0], e[1]) for e in resid] == [(p_agg, q_agg - q_rest)]
    assert resid[0][2] == 3                                  # original entry second, not re-stamped
    if decoy_cancel:
        pr.cancel(5, agg_side)
        assert 5 not in pr.resting(agg_side)
    pr.submit(_probe_row(agg_side, p_agg, q_itl, 3, 3))      # interloper, same second, same price
    book = pr.resting(agg_side)
    assert len(book[2]) == 1 and len(book[3]) == 1
    s_res, s_itl = _stamp(book[2][0]), _stamp(book[3][0])
    assert s_res[0] == s_itl[0] == 3
    assert s_res != s_itl                                    # §5.2 unique
    assert s_res < s_itl                                     # §5.2 sequential, residual first
    pr.submit(_probe_row(rest_side, p_agg, q_agg - q_rest, 4, 4))   # take exactly the residual
    assert pr.trades()[1:] == [(2, 4, p_agg, q_agg - q_rest)]
    assert 2 not in pr.resting(agg_side)
    assert pr.resting(agg_side)[3] == [(p_agg, q_itl, s_itl[0], s_itl[1])]   # interloper whole, unmoved


@pytest.mark.jit
@pytest.mark.parametrize("rest_side", [BUY, SELL])
def test_probe_resting_residual_keeps_stamp_and_beats_same_second_interloper(rest_side):
    """§5.2: a partially executed resting order keeps its timestamp (stored stamp
    byte-identical before and after the fill) and fills ahead of a later interloper
    at its price, even when all four orders share one tape second."""
    agg_side = SELL if rest_side == BUY else BUY
    p_rest = 1004561
    p_agg = 1004517 if rest_side == BUY else 1004600         # crosses; trades at p_rest
    pr = _Probe()
    pr.submit(_probe_row(rest_side, p_rest, 60, 1, 7))
    before = pr.resting(rest_side)[1]
    pr.submit(_probe_row(agg_side, p_agg, 20, 2, 7))
    assert pr.trades() == [(1, 2, p_rest, 20)]
    after = pr.resting(rest_side)[1]
    assert len(after) == 1 and after[0][:2] == (p_rest, 40)
    assert _stamp(after[0]) == _stamp(before[0])             # no new timestamp on partial execution
    assert 2 not in pr.resting(agg_side)
    pr.submit(_probe_row(rest_side, p_rest, 50, 3, 7))       # interloper
    book = pr.resting(rest_side)
    assert _stamp(book[1][0]) != _stamp(book[3][0])
    assert _stamp(book[1][0]) < _stamp(book[3][0])
    pr.submit(_probe_row(agg_side, p_rest, 40, 4, 7))
    assert pr.trades()[1:] == [(1, 4, p_rest, 40)]
    assert 1 not in pr.resting(rest_side)
    assert [e[1] for e in pr.resting(rest_side)[3]] == [50]


@pytest.mark.jit
def test_probe_take_larger_than_residual_spills_to_interloper_in_order():
    """§6.1 / §6.2 item 6: a take larger than the residual fills the residual first, then
    the same-second interloper for the remainder, at the same price."""
    pr = _Probe()
    pr.submit(_probe_row(BUY, 1004519, 27, 1, 2))
    pr.submit(_probe_row(SELL, 1004469, 52, 2, 3))           # residual 25 @ 1004469
    pr.submit(_probe_row(SELL, 1004469, 166, 3, 3))          # interloper
    pr.submit(_probe_row(BUY, 1004469, 40, 4, 4))
    assert pr.trades()[1:] == [(2, 4, 1004469, 25), (3, 4, 1004469, 15)]
    assert [e[1] for e in pr.resting(SELL)[3]] == [151]
