"""K4 audit twin for OMIE-CERT-C16-1 (multi-contract isolation), re-run r2 under OMIE-CERT-RERUN-1;
re-derived at r5 (user K7-U REPEAT on log entry 52) for the 2026-10-20 amendment of §4.2.1 Rule 2
*Incoming iceberg* (instance chain: rows split at incoming instance boundaries; the remainder
rests as instance ``k_e = M // P`` at ``b + k_e d`` showing ``P - u``) -- see ``HAND_TRADES``
lane C, ``HAND_C_AFTER_STEP2``, ``HAND_KE_*`` and ``price_setter_errors``.

Fresh-context, adversarial audit of the **emitted** ``C16.json``, whatever its
status. Everything asserted here was re-derived from the cited rules only
(``market-spec.md`` **v2.14**):

* §2.1 (the quarter-hourly product "will be composed of 96 contracts"), §2.2 ("Each
  contract has its own independent order book"), §6.4 ("Orders on different
  contracts are never matched against each other"), §8.1 (``V = Q x p x d`` with
  ``d = 0.25 h`` for every 15-minute contract, so ``d`` belongs to the product and
  never to the contract id), §3.1 (per-contract gate closure, hence staggered gates);
* §6.2 items 3 and 7 and §4.2.1 Rules 2-4 (v2.14 clarifications "Re-ranking of a
  regenerated slice" and "Incoming iceberg (ICE aggressor)", OQ-31 Reading B, OQ-33,
  OQ-34): the **I5 price-setter relation** -- on every trade the order that was
  already resting is the passive side and sets the price (a resting iceberg at its
  slice's own limit ``base + k * ppd``, ``k`` the slice index), and the incoming order
  of that submission is the aggressor; this includes the Rule-3 hop and an incoming
  iceberg's total-quantity sweep;
* the card (``agile-plan.md`` §OMIE-CERT-C16-1) and the RERUN-1 correction text:
  C16 asserts **S = p** -- each stacked lane's outcome (trades, final book, flags)
  equals the outcome of the same contract run in isolation -- and checks I5's
  price-setter relation. ``n_tapes >= 3`` distinct contract ids, overlapping prices,
  staggered gates, every iceberg within ``ceil(total/peak) <= 64``; exact-parametric
  oracle; eager == jitted; full battery; I8 envelope (80 / 80 / 160 / 800); I9;
  byte-exact quotes; 500 draws.

What the per-draw records carry, and the **reporting gap**. Each lane token is
``cid:rows:ntrades:digest10:SA`` -- the contract id, its tape rows, the isolated
run's trade count and trade digest, and two one-letter verdicts (``S`` stacked ==
isolated, ``A`` environment-alone == isolated). The records therefore carry the
*verdict* S = p for every lane and the isolated trade list (as a digest), but NOT
the stacked lane's own trade digest, NOT the final resting book per side, NOT the
iceberg side-channel and NOT the ``book_full``/``truncated`` flags of either arm. The
I5 relation is carried only as the battery's per-draw letter (``R...`` string) and
the I5 table row; no per-fill (price, passive id, aggressor id, slice index) data is
emitted. Those parts of S = p and of I5 are covered here by **independent engine
probes**: (a) the frozen seed-112 draw replayed through ``Session`` (stacked tapes,
the production slicing path) vs ``build_record`` isolated runs, and (b) hand-built
multi-contract sessions on ``ContractBook``'s public API (``init`` /
``submit_order`` / ``expire_at_gct``): the same lanes ``vmap``-stacked in two lane
orders and interleaved round-robin across per-contract books, each compared lane by
lane (whole ``OMIEBookState`` pytree: trades incl. stamps, both sides, iceberg
side-channel, ``n_active``, ``ts_counter``, ``book_full``, ``truncated``) with the
contract run alone, and every trade checked against spec-derived expected rows and
the price-setter relation.

The per-lane trade digest was reverse-identified (not read from the notebook):
``blake2b(repr(tuple(rows)), digest_size=5)`` over the engine's trade rows
``(P, Q, PASS_OID, AGRS_OID, PASS_TID, AGRS_TID)`` in row order; the empty list
hashes to ``9f444bf3c5``. The slow class re-derives the seed-112 digests.

Independence rules 3/4: this file never opens or imports
``C16_multi_contract_isolation.ipynb``. It uses the generator's public helpers
(``TapeBuilder``, ``stack_tapes``, ``construction_digest``), the recorder
``omie_invariants.build_record`` (it records engine state; the price-setter relation
is re-derived here, not taken from ``check_i5_price_setter``), ``Session``'s public
``reset``/``step_env`` (REL-09: the Session API -- ``reset(key) -> state``,
``step_env(key, state, AgentOrder) -> (state, FillReport, done)``; the participant's
"do nothing" is ``AgentOrder.none()``, broadcast over the vmapped lanes) and
``ContractBook``'s public methods.

Note (engine fact, not a C16 defect): ``jax.vmap`` of ``ContractBook.submit_order``
only lowers under ``jax.experimental.enable_x64()`` (``extract_fills`` accumulates in
int64), which is what ``Session`` does; the stacked probes do the same.

Results JSON: ``$OMIE_CERT_RESULTS_DIR/C16.json`` else
``notebooks/certification/results/C16.json``; absent -> skip.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pytest

from notebooks.certification.cert_support import derive_failure_kinds, validate_results

REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_ID = "C16"
FIXTURE = REPO_ROOT / "notebooks" / "certification" / "fixtures" / "C16_seed112.npz"
FIXTURE_SEED = 112
# The r1 shrunk counterexample: contract 10 of seed 112 alone (I5 rule3_hop on the pre-fix engine).
SHRUNK_FIXTURE = REPO_ROOT / "notebooks" / "certification" / "fixtures" / "C16_seed112_contract10.npz"

BATTERY = ("RB",) + tuple(f"I{i}" for i in range(1, 10))
N_CONTRACTS = 96
REQUESTED_DRAWS = 500  # card: random scenarios default to 500 draws
SPEC_VERSION = "v2.14"
ENVELOPE = {"resting_per_side": 80, "trades": 80, "hops": 160, "concurrent_icebergs": 800}
CEILINGS = [100, 100, 200, 1024, 64]
SLICE_MARGIN = 64
PAD_EXTRA = 8  # the tape-set builder's trailing pad rows

# Tape columns (13-wide int32 loader layout).
C_CID, C_SIDE, C_PRICE, C_QTY, C_OID, C_AGENT, C_TS, C_EXEC, C_PEAK, C_PPD = 0, 1, 2, 3, 4, 5, 6, 7, 11, 12
BUY, SELL = 0, 1
# Trade-row columns kept by the strong form (timestamps SEC/NSEC dropped).
TRADE_COLS = (0, 1, 2, 3, 6, 7)
T_SEC = 4

REQUIRED_QUOTES = {
    "§2.2": "Each contract has its own independent order book",
    "§6.4": "Orders on different contracts are never matched against each other",
    "§2.1": "will be composed of 96 contracts",
}

_POOLED_RE = re.compile(r"^pooled stack (\d+): every contract of draws (\d+)-(\d+) equals its isolated run$")
_OWN_RE = re.compile(r"^seed (\d+): the draw's own stack \((\d+) distinct ids\) equals the isolated runs$")
_LANE_RE = re.compile(r"^(\d+):(\d+):(\d+):([0-9a-f]{10}):(.)(.)$")
_HEX12 = re.compile(r"^[0-9a-f]{12}$")


def trade_digest(rows: Sequence[Sequence[int]]) -> str:
    """The per-lane trade digest (reverse-identified, see module docstring)."""
    return hashlib.blake2b(repr(tuple(tuple(int(x) for x in r) for r in rows)).encode(),
                           digest_size=5).hexdigest()


EMPTY_DIGEST = trade_digest(())


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
        pytest.skip(f"C16 results JSON absent: {p} (run C16_multi_contract_isolation.ipynb section 8 first)")
    return _load(p)


def _roundtrip(tmp_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    p = tmp_path / f"{CERT_ID}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return _load(p)


def _broken(p: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(p)


# ---------------------------------------------------------------------------
# Indexing the payload
# ---------------------------------------------------------------------------
def parse_draw(s: str) -> Dict[str, Any]:
    """``seed|digest12|ops<k>|cid:rows:ntrades:digest10:SA;...|R<10 statuses>|E<+|.>``."""
    parts = s.split("|")
    if len(parts) != 6:
        raise ValueError(f"draw string has {len(parts)} fields: {s!r}")
    seed, dig, ops, lanes_s, bat, env = parts
    if not ops.startswith("ops") or not bat.startswith("R") or not env.startswith("E"):
        raise ValueError(f"malformed draw string: {s!r}")
    lanes = []
    for tok in lanes_s.split(";"):
        m = _LANE_RE.match(tok)
        if not m:
            raise ValueError(f"malformed lane token {tok!r} in {s!r}")
        lanes.append({"cid": int(m.group(1)), "rows": int(m.group(2)), "ntrades": int(m.group(3)),
                      "digest": m.group(4), "S": m.group(5), "A": m.group(6)})
    return {"seed": int(seed), "digest12": dig, "ops": int(ops[3:]), "lanes": lanes,
            "battery_raw": bat[1:], "battery": dict(zip(BATTERY, bat[1:])), "env_alone": env[1:]}


def index(p: Dict[str, Any]) -> Dict[str, Any]:
    pooled, own, eager, summary = [], {}, [], {}
    draws: Dict[int, Dict[str, Any]] = {}
    dup: List[int] = []
    for a in p.get("assertions", []):
        name = a.get("name", "")
        m = _POOLED_RE.match(name)
        if m:
            rows = [parse_draw(s) for s in a["observed"]]
            pooled.append({"assertion": a, "stack": int(m.group(1)), "lo": int(m.group(2)),
                           "hi": int(m.group(3)), "draws": rows})
            for d in rows:
                d["stack"] = int(m.group(1))
                if d["seed"] in draws:
                    dup.append(d["seed"])
                draws.setdefault(d["seed"], d)
            continue
        m = _OWN_RE.match(name)
        if m:
            own[int(m.group(1))] = {"assertion": a, "n_ids": int(m.group(2))}
            continue
        if a.get("kind") == "eager_jit":
            eager.append(a)
            continue
        summary[name] = a
    return {"pooled": pooled, "own": own, "eager": eager, "summary": summary, "draws": draws,
            "dup": dup, "inv": {r["check_id"]: r for r in p.get("invariants", [])}}


@pytest.fixture(scope="module")
def ix(payload) -> Dict[str, Any]:
    return index(payload)


def _summary(ix_, prefix: str) -> Optional[Dict[str, Any]]:
    for name, a in ix_["summary"].items():
        if name.startswith(prefix):
            return a
    return None


# ---------------------------------------------------------------------------
# Checks (each returns a list of human-readable errors; [] = clean)
# ---------------------------------------------------------------------------
def kinds_from_content(p: Dict[str, Any]) -> List[str]:
    """Failure kinds re-derived here, independently of cert_support."""
    kinds = set()
    for a in p["assertions"]:
        if a["status"] == "fail":
            kinds.add({"oracle": "oracle_mismatch", "eager_jit": "eager_jit_disagreement"}.get(a["kind"], "?"))
    if any(not q["ok"] for q in p["spec"]["quotes"]):
        kinds.add("quote_drift")
    if p["draws"]["valid"] < p["draws"]["requested"]:
        kinds.add("capacity_flag")
    for r in p["invariants"]:
        if r["status"] == "fail":
            kinds.add("vacuous_run" if r["check_id"] == "I9" else "invariant_violation")
    return sorted(kinds)


def status_errors(p: Dict[str, Any]) -> List[str]:
    errs = [f"validator: {e}" for e in validate_results(p)]
    if p.get("cert_id") != CERT_ID:
        errs.append(f"cert_id {p.get('cert_id')!r}")
    if p.get("oracle_kind", "").split(" ")[0] != "exact-parametric":
        errs.append(f"oracle_kind {p.get('oracle_kind')!r} is not exact-parametric")
    kinds = kinds_from_content(p)
    if p["failure_kinds"] != kinds:
        errs.append(f"failure_kinds {p['failure_kinds']} != content-derived {kinds}")
    if p["failure_kinds"] != derive_failure_kinds(p):
        errs.append("failure_kinds disagree with cert_support.derive_failure_kinds")
    if (p["status"] == "PASS") != (not kinds):
        errs.append(f"status {p['status']!r} but content implies kinds {kinds}")
    fe = p["failure_explanation"]
    if p["status"] != "PASS":
        if not fe:
            errs.append("NOT CERTIFIED without a failure_explanation")
        else:
            failing = set(p["seeds"]["failing"])
            sc = fe.get("smallest_case") or {}
            if sc.get("seed") is not None and sc["seed"] not in failing:
                errs.append(f"smallest_case seed {sc['seed']} is not a failing seed")
            for r in p["invariants"]:
                if r["status"] == "fail" and r["check_id"] not in fe.get("what_failed", ""):
                    errs.append(f"what_failed does not name {r['check_id']}")
    return errs


def accounting_errors(p: Dict[str, Any], ix_: Dict[str, Any]) -> List[str]:
    errs = []
    d, s = p["draws"], p["seeds"]
    if d["requested"] != REQUESTED_DRAWS:
        errs.append(f"requested {d['requested']} != {REQUESTED_DRAWS}")
    if d["valid"] != d["requested"]:
        errs.append(f"valid {d['valid']} != requested {d['requested']} (INVALID draws are re-drawn)")
    denom = d["valid"] + d["discarded"]
    if not math.isclose(d["discard_rate"], d["discarded"] / denom if denom else 0.0, abs_tol=1e-12):
        errs.append("discard_rate inconsistent")
    if s["count"] != d["valid"]:
        errs.append(f"seeds.count {s['count']} != valid draws {d['valid']}")
    seeds = sorted(ix_["draws"])
    if ix_["dup"]:
        errs.append(f"draws recorded twice: {ix_['dup'][:10]}")
    if len(seeds) != d["valid"]:
        errs.append(f"{len(seeds)} draws carry per-draw evidence, {d['valid']} valid draws")
    if seeds and (seeds[0] != s["first"] or seeds[-1] != s["last"]):
        errs.append(f"per-draw seeds span {seeds[0]}-{seeds[-1]}, seeds block {s['first']}-{s['last']}")
    for st in ix_["pooled"]:
        got = sorted(dd["seed"] for dd in st["draws"])
        if got != list(range(st["lo"], st["hi"] + 1)):
            errs.append(f"pooled stack {st['stack']} does not carry exactly draws {st['lo']}-{st['hi']}")
    if not ix_["pooled"]:
        errs.append("no pooled stacked run recorded")
    cov = _summary(ix_, "every valid draw ran in a pooled stacked run")
    if cov is None or cov["observed"] != len(seeds) or cov["observed"] != d["valid"]:
        errs.append("coverage assertion absent or inconsistent with the per-draw evidence")
    return errs


def isolation_errors(ix_: Dict[str, Any]) -> List[str]:
    """The strong form, per lane, plus the consistency of the assertions that report it."""
    errs = []
    env_alone_lanes = 0
    for st in ix_["pooled"]:
        bad_here = []
        for dd in st["draws"]:
            for ln in dd["lanes"]:
                if ln["S"] != "p":
                    bad_here.append((dd["seed"], ln["cid"]))
                    errs.append(f"seed {dd['seed']} cid {ln['cid']}: stacked lane differs from its isolated run "
                                f"(S={ln['S']!r})")
                if ln["A"] not in ("p", "-"):
                    errs.append(f"seed {dd['seed']} cid {ln['cid']}: environment-alone run differs (A={ln['A']!r})")
                if (ln["A"] == "p") != (dd["env_alone"] == "+"):
                    errs.append(f"seed {dd['seed']} cid {ln['cid']}: A={ln['A']!r} with E{dd['env_alone']}")
                env_alone_lanes += ln["A"] == "p"
            if dd["env_alone"] == "+" and len(dd["lanes"]) < 3:
                errs.append(f"seed {dd['seed']}: environment-alone comparison on < 3 contracts")
        a = st["assertion"]
        if a["kind"] != "oracle":
            errs.append(f"pooled stack {st['stack']} is not an oracle assertion")
        if (a["status"] == "pass") != (not bad_here):
            errs.append(f"pooled stack {st['stack']} status {a['status']!r} masks lanes {bad_here[:5]}")
    for seed, o in ix_["own"].items():
        dd = ix_["draws"].get(seed)
        if dd is None:
            errs.append(f"own-stack assertion for unrecorded seed {seed}")
            continue
        flags = "".join(ln["S"] for ln in dd["lanes"])
        a = o["assertion"]
        if o["n_ids"] != len({ln["cid"] for ln in dd["lanes"]}):
            errs.append(f"seed {seed}: own stack claims {o['n_ids']} ids, draw has {len(dd['lanes'])}")
        if a["observed"] != flags or a["expected"] != "p" * len(dd["lanes"]):
            errs.append(f"seed {seed}: own-stack observed {a['observed']!r} vs per-lane flags {flags!r}")
        if (a["status"] == "pass") != (a["observed"] == a["expected"]):
            errs.append(f"seed {seed}: own-stack status inconsistent")
    if not ix_["own"]:
        errs.append("no draw ran in its own stack")
    alone = _summary(ix_, "environment-alone comparisons made")
    if alone is None or alone["observed"] != env_alone_lanes or env_alone_lanes < 3:
        errs.append(f"environment-alone count {None if alone is None else alone['observed']} "
                    f"vs {env_alone_lanes} lanes flagged A=p (need >= 3)")
    return errs


def dataset_errors(ix_: Dict[str, Any]) -> List[str]:
    errs = []
    ids = set()
    pre = _summary(ix_, "environment slicing precondition")
    ops_per_stack = [st["draws"][0]["ops"] for st in sorted(ix_["pooled"], key=lambda s: s["stack"])]
    for st in ix_["pooled"]:
        if len({dd["ops"] for dd in st["draws"]}) != 1:
            errs.append(f"pooled stack {st['stack']} mixes orders_per_step")
    if pre is None or pre["observed"] != ops_per_stack:
        errs.append(f"slicing precondition {None if pre is None else pre['observed']} != per-stack ops {ops_per_stack}")
    for seed, dd in ix_["draws"].items():
        cids = [ln["cid"] for ln in dd["lanes"]]
        if len(cids) < 3:
            errs.append(f"seed {seed}: n_tapes {len(cids)} < 3")
        if len(set(cids)) != len(cids):
            errs.append(f"seed {seed}: contract ids not distinct {cids}")
        if any(not 0 <= c < N_CONTRACTS for c in cids):
            errs.append(f"seed {seed}: contract id outside 0-95 {cids}")
        if any(ln["rows"] < 1 for ln in dd["lanes"]):
            errs.append(f"seed {seed}: an empty lane")
        if not 1 <= dd["ops"] <= PAD_EXTRA:
            errs.append(f"seed {seed}: orders_per_step {dd['ops']} outside [1, {PAD_EXTRA}]")
        if not _HEX12.match(dd["digest12"]):
            errs.append(f"seed {seed}: draw digest {dd['digest12']!r}")
        ids.update(cids)
    if not {0, N_CONTRACTS - 1} <= ids:
        errs.append(f"contract ids 0 and 95 not both exercised ({min(ids)}-{max(ids)})")
    ends = _summary(ix_, "§2.1: contract ids 0 and 95")
    if ends is None or ends["observed"] != {"distinct_ids": len(ids), "min": min(ids), "max": max(ids)}:
        errs.append("§2.1 coverage assertion disagrees with the per-draw contract ids")
    n96 = _summary(ix_, "§2.1: the market configuration models 96 contracts")
    if n96 is None or n96["observed"] != N_CONTRACTS or n96["status"] != "pass":
        errs.append("§2.1: 96-contract configuration not asserted")
    pc = _summary(ix_, "dataset precondition: failed draws")
    if pc is None or pc["observed"] != 0:
        errs.append("dataset precondition (overlap, stagger, slice margin) not clean")
    return errs


def digest_errors(ix_: Dict[str, Any]) -> List[str]:
    """Per-lane evidence self-consistency: empty trade list <=> empty digest; no two
    trading lanes of one draw carry the same trade list (a copied book is a leak)."""
    errs = []
    for seed, dd in ix_["draws"].items():
        seen: Dict[str, int] = {}
        for ln in dd["lanes"]:
            if (ln["ntrades"] == 0) != (ln["digest"] == EMPTY_DIGEST):
                errs.append(f"seed {seed} cid {ln['cid']}: {ln['ntrades']} trades with digest {ln['digest']}")
            if ln["ntrades"] > 0:
                if ln["digest"] in seen:
                    errs.append(f"seed {seed}: cids {seen[ln['digest']]} and {ln['cid']} carry the same trade list")
                seen[ln["digest"]] = ln["cid"]
    return errs


def battery_errors(p: Dict[str, Any], ix_: Dict[str, Any]) -> List[str]:
    errs = []
    inv = ix_["inv"]
    if sorted(inv) != sorted(BATTERY):
        errs.append(f"battery rows {sorted(inv)} != {sorted(BATTERY)}")
    tally = {c: {"p": 0, "f": 0, "n": 0, "seeds": []} for c in BATTERY}
    for seed, dd in sorted(ix_["draws"].items()):
        if len(dd["battery_raw"]) != len(BATTERY):
            errs.append(f"seed {seed}: battery string {dd['battery_raw']!r} is not the full battery")
            continue
        trades = sum(ln["ntrades"] for ln in dd["lanes"])
        for c, v in dd["battery"].items():
            if v == "p":
                tally[c]["p"] += 1
            elif v == "f":
                tally[c]["f"] += 1
                tally[c]["seeds"].append(seed)
            else:
                tally[c]["n"] += 1
                if trades > 0:
                    errs.append(f"seed {seed}: {c} not_applicable ({v!r}) on a draw with {trades} fills")
        if dd["battery"].get("I9") == "p" and trades == 0:
            errs.append(f"seed {seed}: I9 non-vacuity passes on a draw with no trade")
    for c in BATTERY:
        r = inv.get(c)
        if r is None:
            continue
        t = tally[c]
        if (r["n_pass"], r["n_fail"], r["n_not_applicable"]) != (t["p"], t["f"], t["n"]):
            errs.append(f"{c}: table {r['n_pass']}/{r['n_fail']}/{r['n_not_applicable']} vs per-draw "
                        f"{t['p']}/{t['f']}/{t['n']}")
        if sorted(r["failing_seeds"]) != t["seeds"]:
            errs.append(f"{c}: failing_seeds differ from the per-draw battery")
        if (r["status"] == "fail") != (t["f"] > 0):
            errs.append(f"{c}: status {r['status']!r} with {t['f']} failing draws")
        if r["status"] == "not_applicable" and t["p"] + t["f"] > 0:
            errs.append(f"{c}: not_applicable overall although it ran")
    want_failing = sorted({s for c in BATTERY for s in tally[c]["seeds"]}
                          | {a["seed"] for a in p["assertions"] if a["status"] == "fail" and a["seed"] is not None})
    if sorted(p["seeds"]["failing"]) != want_failing:
        errs.append("seeds.failing is not the union of the per-draw failures")
    return errs


def capacity_errors(p: Dict[str, Any], ix_: Dict[str, Any]) -> List[str]:
    errs = []
    cap = p["capacity"]
    if cap["envelope"] != ENVELOPE:
        errs.append(f"envelope {cap['envelope']} != {ENVELOPE}")
    ceil = _summary(ix_, "the five configured ceilings")
    if ceil is None or ceil["observed"] != CEILINGS or ceil["expected"] != CEILINGS:
        errs.append("the five ceilings are not recorded as documented")
    pk = cap["peaks"]
    max_tr = max((ln["ntrades"] for dd in ix_["draws"].values() for ln in dd["lanes"]), default=0)
    if pk.get("trades") != max_tr:
        errs.append(f"peak trades {pk.get('trades')} != per-lane maximum {max_tr}")
    for k, lim in ENVELOPE.items():
        v = pk.get(k)
        if v is None:
            if not (k == "hops" and "matching_iterations" in ix_["inv"].get("I8", {}).get("unobserved", [])):
                errs.append(f"peak {k} absent and not reported unobserved")
        elif v > lim:
            errs.append(f"peak {k} = {v} > envelope {lim} (draw should have been INVALID)")
    return errs


def eager_errors(p: Dict[str, Any], ix_: Dict[str, Any]) -> List[str]:
    """Eager-vs-jitted evidence must be PRESENT for both arms and identical; absence fails."""
    errs = []
    rec = [a for a in ix_["eager"] if "environment" not in a.get("detail", "")]
    env = [a for a in ix_["eager"] if "environment" in a.get("detail", "")]
    if not rec:
        errs.append("no eager == jitted comparison of the isolated (record) arm")
    if not env:
        errs.append("no eager == jitted comparison of the stacked environment arm")
    for a in ix_["eager"]:
        if a["status"] != "pass" or a["observed"] is not True or a["expected"] is not True:
            errs.append(f"{a['name']}: {a['status']} observed={a['observed']!r}")
        if a["seed"] not in ix_["draws"]:
            errs.append(f"{a['name']}: seed {a['seed']} is not a recorded valid draw")
    for prefix, n in (("record eager comparisons made", len(rec)), ("environment eager comparisons made", len(env))):
        s = _summary(ix_, prefix)
        if s is None or s["observed"] != n or s["status"] != "pass":
            errs.append(f"{prefix!r}: {None if s is None else s['observed']} vs {n} recorded")
    all_ok = bool(ix_["eager"]) and all(a["status"] == "pass" for a in ix_["eager"])
    if p["eager_jit_identical"] is not True or p["eager_jit_identical"] != all_ok:
        errs.append(f"eager_jit_identical={p['eager_jit_identical']!r} vs per-comparison evidence")
    return errs


def wc_errors(ix_: Dict[str, Any]) -> List[str]:
    """§8.1: W_c identical in every arm; a non-empty mismatch list is a failure, and a
    'pass' over a non-empty list is a masked failure."""
    a = _summary(ix_, "§8.1: per-contract integer trade value W_c")
    if a is None:
        return ["§8.1 W_c assertion absent"]
    errs = []
    if a["observed"] not in ([], None) or a["expected"] != []:
        errs.append(f"W_c mismatches recorded: {a['observed']!r}")
    if (a["status"] == "pass") != (a["observed"] == a["expected"]):
        errs.append("W_c assertion status masks its observation")
    d = _summary(ix_, "§8.1: d = 0.25 h")
    if d is None or d["observed"] != {"delivery_duration_hours": 0.25, "mtu_minutes": 15}:
        errs.append("§8.1 d = 0.25 h not asserted on the configuration")
    return errs


def quote_errors(p: Dict[str, Any]) -> List[str]:
    errs = []
    spec = (REPO_ROOT / "docs" / "omiecid" / "market-spec.md").read_text(encoding="utf-8")
    secs = {(q["source"], q["section"]): q for q in p["spec"]["quotes"]}
    for sec, text in REQUIRED_QUOTES.items():
        q = secs.get(("market-spec.md", sec))
        if q is None or not q["ok"]:
            errs.append(f"quote market-spec.md {sec} absent or not ok")
        if text not in spec:
            errs.append(f"market-spec.md no longer contains {text!r}")
    for q in p["spec"]["quotes"]:
        if not q["ok"]:
            errs.append(f"quote {q['source']} {q['section']}: {q['reason']}")
    return errs


def attribute_failures(p: Dict[str, Any], ix_: Dict[str, Any]) -> Dict[int, str]:
    """``seed -> 'isolation_leak' | 'common_matching_defect'`` for every failing draw."""
    out = {}
    for seed in p["seeds"]["failing"]:
        dd = ix_["draws"].get(seed)
        if dd is None:
            out[seed] = "unrecorded"
        elif any(ln["S"] != "p" for ln in dd["lanes"]):
            out[seed] = "isolation_leak"
        elif any(v == "f" for v in dd["battery"].values()):
            out[seed] = "common_matching_defect"
        else:
            out[seed] = "unexplained"
    return out


def i5_record_errors(p: Dict[str, Any], ix_: Dict[str, Any]) -> List[str]:
    """What the records carry of I5 (the price-setter relation): a per-draw verdict
    letter and the table row. On a PASS run every draw with a fill must say 'p'."""
    errs = []
    row = ix_["inv"].get("I5")
    if row is None:
        return ["I5 row absent"]
    letters = {s: dd["battery"].get("I5") for s, dd in ix_["draws"].items()}
    traded = {s for s, dd in ix_["draws"].items() if sum(ln["ntrades"] for ln in dd["lanes"]) > 0}
    if p["status"] == "PASS":
        bad = sorted(s for s in traded if letters[s] != "p")
        if bad:
            errs.append(f"PASS but I5 not 'p' on draws with fills: {bad[:10]}")
        if row["status"] != "pass" or row["n_fail"] != 0:
            errs.append(f"PASS but I5 row {row['status']} n_fail={row['n_fail']}")
    if row["unobserved"]:
        errs.append(f"I5 reports unobserved outputs {row['unobserved']}")
    if not traded:
        errs.append("no draw with a fill: I5 would be vacuous")
    return errs


def spec_stamp_errors(p: Dict[str, Any]) -> List[str]:
    errs = []
    head = (REPO_ROOT / "docs" / "omiecid" / "market-spec.md").read_text(encoding="utf-8")[:400]
    if f"**Status:** {SPEC_VERSION}" not in head:
        errs.append(f"market-spec.md is not {SPEC_VERSION}")
    for q in p["spec"]["quotes"]:
        if q["source"] == "market-spec.md" and q["version"] != SPEC_VERSION:
            errs.append(f"quote {q['section']} stamped {q['version']!r}, not {SPEC_VERSION}")
    return errs


def all_fast_errors(p: Dict[str, Any]) -> List[str]:
    i = index(p)
    return (status_errors(p) + accounting_errors(p, i) + isolation_errors(i) + dataset_errors(i)
            + digest_errors(i) + battery_errors(p, i) + capacity_errors(p, i) + eager_errors(p, i)
            + wc_errors(i) + quote_errors(p) + i5_record_errors(p, i) + spec_stamp_errors(p))


# ===========================================================================
# Fast tests -- the emitted JSON only
# ===========================================================================
class TestSchemaAndStatus:
    def test_schema_and_semantic_validator_accept(self, payload):
        assert validate_results(payload) == []

    def test_status_failure_kinds_and_explanation_consistent(self, payload):
        assert status_errors(payload) == []

    def test_draw_and_seed_accounting(self, payload, ix):
        assert accounting_errors(payload, ix) == []


class TestStrongFormIsolation:
    """§2.2 / §6.4: every lane of every stacked run equals its isolated run."""

    def test_every_lane_equals_its_isolated_run(self, ix):
        assert isolation_errors(ix) == []

    def test_per_lane_evidence_present_in_full(self, payload, ix):
        """Every valid draw carries a per-lane token for every one of its contracts."""
        assert len(ix["draws"]) == payload["draws"]["valid"]
        assert all(dd["lanes"] and all(ln["S"] in "pf" for ln in dd["lanes"]) for dd in ix["draws"].values())

    def test_trade_digests_self_consistent_and_no_copied_book(self, ix):
        assert EMPTY_DIGEST == "9f444bf3c5"
        assert digest_errors(ix) == []

    def test_dataset_shape(self, ix):
        assert dataset_errors(ix) == []


class TestBatteryCapacityEagerQuotes:
    def test_full_battery_tallies_match_per_draw(self, payload, ix):
        assert battery_errors(payload, ix) == []

    def test_capacity_envelope(self, payload, ix):
        assert capacity_errors(payload, ix) == []

    def test_eager_jit_evidence_present_and_identical(self, payload, ix):
        assert eager_errors(payload, ix) == []

    def test_w_c_identical_in_every_arm(self, ix):
        assert wc_errors(ix) == []

    def test_quotes_present_and_byte_exact(self, payload):
        assert quote_errors(payload) == []

    def test_spec_stamp_is_v2_14(self, payload):
        assert spec_stamp_errors(payload) == []


class TestWhatTheRecordsCarry:
    """S = p and I5 as far as the emitted per-draw records carry them."""

    def test_s_equals_p_on_every_lane_of_every_draw(self, payload, ix):
        lanes = [ln for dd in ix["draws"].values() for ln in dd["lanes"]]
        assert len(ix["draws"]) == payload["draws"]["valid"] == REQUESTED_DRAWS
        assert lanes and all(ln["S"] == "p" for ln in lanes), "a stacked lane differs from its isolated run"

    def test_i5_price_setter_verdict_on_every_draw(self, payload, ix):
        assert i5_record_errors(payload, ix) == []

    def test_reporting_gap_lane_token_carries_trades_only(self, payload):
        """Reporting gap (stated, not a failure): each lane token has exactly six
        fields -- cid, rows, isolated ntrades, isolated trade digest, S, A. The final
        book per side, the iceberg side-channel, the book_full/truncated flags and the
        stacked arm's own digest are NOT emitted, and I5 is emitted only as a verdict
        letter (no per-fill price / passive / aggressor / slice data). Those parts of
        S = p and I5 rest on the notebook's verdict plus this file's engine probes
        (TestIndependentProbe, TestContractBookStackedProbe). If the token grows, this
        test fails so the twin is upgraded to assert the new fields."""
        n = 0
        for a in payload["assertions"]:
            if not _POOLED_RE.match(a["name"]):
                continue
            for s in a["observed"]:
                fields = s.split("|")
                assert len(fields) == 6, s
                for tok in fields[3].split(";"):
                    parts = tok.split(":")
                    assert len(parts) == 5 and len(parts[4]) == 2, tok
                    n += 1
        assert n > 0


class TestAttribution:
    """NOT CERTIFIED: is any failure an isolation leak?"""

    def test_no_failure_is_an_isolation_leak(self, payload, ix):
        attr = attribute_failures(payload, ix)
        assert not [s for s, k in attr.items() if k != "common_matching_defect"], attr

    def test_not_certified_is_owned_by_the_battery_on_isolated_runs(self, payload, ix):
        if payload["status"] == "PASS":
            pytest.skip("C16 PASSED; nothing to attribute")
        failing_checks = sorted(r["check_id"] for r in payload["invariants"] if r["status"] == "fail")
        failing_oracles = [a["name"] for a in payload["assertions"] if a["status"] == "fail"]
        assert failing_oracles == [], f"an oracle (isolation) assertion failed: {failing_oracles}"
        assert payload["failure_kinds"] == ["invariant_violation"]
        assert failing_checks, "NOT CERTIFIED without a failing check"


class TestFixtureSeed112:
    """The frozen smallest case: its lanes, the dataset properties the card requires."""

    @pytest.fixture(scope="class")
    def fx(self):
        if not FIXTURE.is_file():
            pytest.skip(f"C16 fixture absent: {FIXTURE}")
        z = np.load(FIXTURE, allow_pickle=False)
        tape = np.asarray(z["tape"])
        return tape, json.loads(str(z["params_json"]))

    def test_fixture_is_a_recorded_draw_with_the_same_lanes(self, payload, ix, fx):
        """The frozen seed-112 draw (the r1 smallest case) is a recorded valid draw of
        the r2 run, with the same contract ids and per-lane row counts; on the fixed
        engine it is no longer failing (r2 is PASS)."""
        tape, params = fx
        real = tape[tape[:, C_OID] >= 1]
        dd = ix["draws"][FIXTURE_SEED]
        assert [ln["cid"] for ln in dd["lanes"]] == params["contract_ids"]
        for ln in dd["lanes"]:
            assert int(np.sum(real[:, C_CID] == ln["cid"])) == ln["rows"]
        assert sum(ln["rows"] for ln in dd["lanes"]) == len(real)
        assert (FIXTURE_SEED in payload["seeds"]["failing"]) == (payload["status"] != "PASS"
                                                                 and any(v == "f" for v in dd["battery"].values()))

    def test_overlapping_prices_staggered_gates_slice_margin(self, fx):
        tape, params = fx
        real = tape[tape[:, C_OID] >= 1]
        lanes = {c: real[real[:, C_CID] == c] for c in params["contract_ids"]}
        assert len(lanes) >= 3
        rng = {c: (int(l[:, C_PRICE].min()), int(l[:, C_PRICE].max())) for c, l in lanes.items()}
        cs = list(rng)
        assert all(max(rng[a][0], rng[b][0]) <= min(rng[a][1], rng[b][1]) for a in cs for b in cs), rng
        leak = sum(1 for a in cs for b in cs if a != b
                   for bp in lanes[a][lanes[a][:, C_SIDE] == BUY][:, C_PRICE]
                   for sp in lanes[b][lanes[b][:, C_SIDE] == SELL][:, C_PRICE] if bp >= sp)
        assert leak > 0, "no cross-contract crossable pair: a leak would not match"
        gates = [c["gate_s"] for c in params["contracts"]]
        assert len(set(gates)) == len(gates), gates
        ice = real[real[:, C_PEAK] > 0]
        assert all(-(-int(r[C_QTY]) // int(r[C_PEAK])) <= SLICE_MARGIN for r in ice)


# ---------------------------------------------------------------------------
# Broken copies -- each must be rejected
# ---------------------------------------------------------------------------
def _first_lane(p, pred=lambda dd, ln: True):
    for k, a in enumerate(p["assertions"]):
        if _POOLED_RE.match(a["name"]):
            for j, s in enumerate(a["observed"]):
                dd = parse_draw(s)
                for li, ln in enumerate(dd["lanes"]):
                    if pred(dd, ln):
                        return k, j, li
    raise AssertionError("no lane matches")


def _edit_lane(p, k, j, li, **kw):
    parts = p["assertions"][k]["observed"][j].split("|")
    toks = parts[3].split(";")
    cid, rows, ntr, dig, sa = toks[li].split(":")
    f = {"cid": cid, "rows": rows, "ntrades": ntr, "digest": dig, "S": sa[0], "A": sa[1]}
    f.update({x: str(y) for x, y in kw.items()})
    toks[li] = f"{f['cid']}:{f['rows']}:{f['ntrades']}:{f['digest']}:{f['S']}{f['A']}"
    parts[3] = ";".join(toks)
    p["assertions"][k]["observed"][j] = "|".join(parts)


class TestBrokenCopiesAreRejected:
    def test_lane_trades_differ_from_isolated_run(self, payload, tmp_path):
        b = _broken(payload)
        _edit_lane(b, *_first_lane(b, lambda dd, ln: ln["ntrades"] > 0), S="f")
        b = _roundtrip(tmp_path, b)
        assert isolation_errors(index(b)), "a lane differing from its isolated run was accepted"
        assert all_fast_errors(b)

    def test_lane_copied_from_another_lane(self, payload, tmp_path):
        """A foreign book's trades in a lane (same trade list as a sibling lane)."""
        b = _broken(payload)
        k, j, _ = _first_lane(b, lambda dd, ln: sum(x["ntrades"] > 0 for x in dd["lanes"]) >= 2)
        dd = parse_draw(b["assertions"][k]["observed"][j])
        trading = [i for i, ln in enumerate(dd["lanes"]) if ln["ntrades"] > 0]
        src = dd["lanes"][trading[0]]
        _edit_lane(b, k, j, trading[1], ntrades=src["ntrades"], digest=src["digest"])
        b = _roundtrip(tmp_path, b)
        assert digest_errors(index(b))

    def test_trading_lane_with_empty_digest(self, payload, tmp_path):
        b = _broken(payload)
        _edit_lane(b, *_first_lane(b, lambda dd, ln: ln["ntrades"] > 0), digest=EMPTY_DIGEST)
        assert digest_errors(index(_roundtrip(tmp_path, b)))

    def test_contract_dependent_w_c(self, payload, tmp_path):
        b = _broken(payload)
        for a in b["assertions"]:
            if a["name"].startswith("§8.1: per-contract integer trade value W_c"):
                a["observed"] = [{"seed": FIXTURE_SEED, "cid": 10, "isolated": 1004,
                                  "stacked": 1004, "relabelled_cid50": 2008}]
        b = _roundtrip(tmp_path, b)
        assert wc_errors(index(b))

    def test_env_alone_flag_disagrees(self, payload, tmp_path):
        b = _broken(payload)
        _edit_lane(b, *_first_lane(b, lambda dd, ln: ln["A"] == "p"), A="f")
        assert isolation_errors(index(_roundtrip(tmp_path, b)))

    def test_eager_evidence_absent(self, payload, tmp_path):
        b = _broken(payload)
        b["assertions"] = [a for a in b["assertions"] if a["kind"] != "eager_jit"]
        b = _roundtrip(tmp_path, b)
        assert eager_errors(b, index(b))

    def test_environment_eager_evidence_absent(self, payload, tmp_path):
        b = _broken(payload)
        b["assertions"] = [a for a in b["assertions"]
                           if not (a["kind"] == "eager_jit" and "environment" in a["detail"])]
        b = _roundtrip(tmp_path, b)
        assert eager_errors(b, index(b))

    def test_not_applicable_with_fills(self, payload, tmp_path):
        b = _broken(payload)
        k, j, _ = _first_lane(b, lambda dd, ln: ln["ntrades"] > 0)
        parts = b["assertions"][k]["observed"][j].split("|")
        parts[4] = "R" + parts[4][1:6] + "n" + parts[4][7:]
        b["assertions"][k]["observed"][j] = "|".join(parts)
        b = _roundtrip(tmp_path, b)
        assert battery_errors(b, index(b))

    def test_battery_tally_edited(self, payload, tmp_path):
        """Move one I5 draw between pass and fail in the table only."""
        b = _broken(payload)
        for r in b["invariants"]:
            if r["check_id"] == "I5":
                if r["n_pass"] > 0:
                    r["n_pass"], r["n_fail"] = r["n_pass"] - 1, r["n_fail"] + 1
                else:
                    r["n_pass"], r["n_fail"] = r["n_pass"] + 1, r["n_fail"] - 1
        b = _roundtrip(tmp_path, b)
        assert battery_errors(b, index(b))

    def test_i5_failure_in_a_draw_hidden_by_the_table(self, payload, tmp_path):
        """A per-draw I5 'f' (a price-setter violation) under an all-pass I5 row."""
        b = _broken(payload)
        k, j, _ = _first_lane(b, lambda dd, ln: ln["ntrades"] > 0 and dd["battery"]["I5"] == "p")
        parts = b["assertions"][k]["observed"][j].split("|")
        i5 = BATTERY.index("I5")
        parts[4] = "R" + parts[4][1:1 + i5] + "f" + parts[4][2 + i5:]
        b["assertions"][k]["observed"][j] = "|".join(parts)
        b = _roundtrip(tmp_path, b)
        assert battery_errors(b, index(b))
        assert i5_record_errors(b, index(b))

    def test_status_flipped(self, payload, tmp_path):
        """The emitted status inverted without touching the content is rejected."""
        b = _broken(payload)
        if b["status"] == "PASS":
            b["status"] = "NOT CERTIFIED"
        else:
            b["status"], b["failure_kinds"], b["failure_explanation"] = "PASS", [], None
        assert status_errors(_roundtrip(tmp_path, b))

    def test_pass_with_a_lane_flagged_different(self, payload, tmp_path):
        """S != p on one lane while every assertion still says pass."""
        b = _broken(payload)
        _edit_lane(b, *_first_lane(b, lambda dd, ln: ln["ntrades"] == 0), S="f")
        b = _roundtrip(tmp_path, b)
        assert isolation_errors(index(b))

    def test_draw_dropped(self, payload, tmp_path):
        b = _broken(payload)
        k, _, _ = _first_lane(b)
        b["assertions"][k]["observed"].pop(3)
        b = _roundtrip(tmp_path, b)
        assert accounting_errors(b, index(b))

    def test_peak_trades_misreported(self, payload, tmp_path):
        b = _broken(payload)
        b["capacity"]["peaks"]["trades"] = 12
        b = _roundtrip(tmp_path, b)
        assert capacity_errors(b, index(b))

    def test_quote_drift_hidden(self, payload, tmp_path):
        b = _broken(payload)
        b["spec"]["quotes"][1]["ok"], b["spec"]["quotes"][1]["reason"] = False, "absent"
        assert quote_errors(_roundtrip(tmp_path, b))

    def test_too_few_contracts_in_a_draw(self, payload, tmp_path):
        b = _broken(payload)
        k, j, _ = _first_lane(b, lambda dd, ln: len(dd["lanes"]) == 3)
        parts = b["assertions"][k]["observed"][j].split("|")
        parts[3] = ";".join(parts[3].split(";")[:2])
        b["assertions"][k]["observed"][j] = "|".join(parts)
        assert dataset_errors(index(_roundtrip(tmp_path, b)))



def test_generator_registry_digest_matches(payload):
    """The C15 random-session registry the C16 dataset is built from is unchanged (~0.15 s)."""
    from omiecid_simulator.jaxomiecid.omie_tape_synth import construction_digest
    for key, dig in payload["generator"]["digests"].items():
        assert construction_digest(key) == dig, key


# ===========================================================================
# The I5 price-setter relation, re-derived (pure function, no engine)
# ===========================================================================
ICE_ER = 3  # ExecRestriction.ICE (tape / AcceptedOrder encoding)


class Decl(NamedTuple):
    """An accepted order's declaration in integer ticks."""

    oid: int
    submission: int
    price: int
    er: int
    peak: int
    ppd: int


class TradeFill(NamedTuple):
    submission: int
    price: int
    qty: int
    passive: int
    aggressor: int


def price_setter_errors(decls: Mapping[int, Decl], sub_oid: Mapping[int, int],
                        fills: Sequence[TradeFill]) -> List[str]:
    """§6.2 items 3/7, §4.2.1 Rules 2-4 (v2.14), OQ-31 Reading B, OQ-33, OQ-34.

    For every fill, in row order:
    * the aggressor is the order of the submission that wrote the row (the incoming
      order, also on a Rule-3 hop and on an incoming iceberg's sweep);
    * the passive side is a different order that was accepted at an earlier
      submission (it was already resting);
    * the price is the passive order's limit; for a passive iceberg it is its slice's
      own limit ``base + k * ppd`` with ``k = (M + qty already filled against it as
      the passive side) // peak``, ``M`` the quantity the iceberg swept as the incoming
      order of its own submission (0 if it rested untraded) -- instances are consumed
      in order, each ``peak`` except the last. r5 re-derivation (market-spec v2.14 as
      amended 2026-10-20, §4.2.1 Rule 2 *Incoming iceberg* item (4), Rule 3 *ICE
      aggressor*; OQ-34 2026-10-20): a rested incoming-iceberg remainder keeps its
      place in the ladder as instance ``k_e = M // peak`` at ``base + k_e * ppd``,
      showing ``peak - u`` (``u = M - k_e * peak``), so the passive position simply
      continues from ``M``. (The r2 helper restarted it at slice 0: the superseded
      OQ-34 reading.) The price never comes from the incoming order.
    """
    errs = []
    passive_filled: Dict[int, int] = defaultdict(int)
    swept: Dict[int, int] = defaultdict(int)  # M: qty an ICE traded as its own submission's aggressor
    for i, f in enumerate(fills):
        inc = sub_oid.get(f.submission)
        if inc is None:
            errs.append(f"row {i}: written by unknown submission {f.submission}")
            continue
        if f.aggressor != inc:
            errs.append(f"row {i}: aggressor {f.aggressor} is not the incoming order {inc}")
        if f.passive == inc:
            errs.append(f"row {i}: the incoming order {inc} is the passive side")
        d = decls.get(f.passive)
        if d is None:
            errs.append(f"row {i}: passive {f.passive} is not an accepted order of this contract")
            continue
        if d.submission >= f.submission:
            errs.append(f"row {i}: passive {f.passive} was not resting before submission {f.submission}")
        if d.er == ICE_ER and d.peak > 0:
            k = (swept[d.oid] + passive_filled[d.oid]) // d.peak
            want = d.price + k * d.ppd
        else:
            k, want = None, d.price
        if f.price != want:
            errs.append(f"row {i}: price {f.price} != passive {f.passive} limit {want} (slice {k})")
        passive_filled[d.oid] += f.qty
        if f.aggressor == inc:
            swept[inc] += f.qty
    return errs


def _record_inputs(rec) -> Tuple[Dict[int, Decl], Dict[int, int], List[TradeFill]]:
    decls = {a.oid: Decl(a.oid, a.submission, a.price_int, a.exec_restriction, a.peak_int, a.ppd_int)
             for a in rec.accepted.values()}
    sub_oid = {a.submission: a.oid for a in rec.accepted.values()}
    fills = [TradeFill(f.submission, f.price_int, f.qty_int, f.passive_oid, f.aggressor_oid) for f in rec.fills]
    return decls, sub_oid, fills


class TestPriceSetterRelationIsDiscriminating:
    """The re-derived relation itself rejects each way of breaking I5 (fast, no engine)."""

    DECLS = {1: Decl(1, 0, 1000, ICE_ER, 2, 5), 2: Decl(2, 1, 1020, 0, 0, 0), 3: Decl(3, 2, 990, 0, 0, 0)}
    SUBS = {0: 1, 1: 2, 2: 3}
    GOOD = [TradeFill(1, 1000, 2, 1, 2), TradeFill(1, 1005, 2, 1, 2), TradeFill(2, 1010, 1, 1, 3)]

    def test_good_sequence_accepted(self):
        assert price_setter_errors(self.DECLS, self.SUBS, self.GOOD) == []

    def test_hop_printed_at_the_aggressor_limit_rejected(self):
        bad = [self.GOOD[0], self.GOOD[1]._replace(price=1020), self.GOOD[2]]
        assert price_setter_errors(self.DECLS, self.SUBS, bad)

    def test_slice_drift_not_applied_rejected(self):
        bad = [self.GOOD[0], self.GOOD[1]._replace(price=1000), self.GOOD[2]]
        assert price_setter_errors(self.DECLS, self.SUBS, bad)

    def test_roles_swapped_rejected(self):
        bad = [self.GOOD[0]._replace(passive=2, aggressor=1, price=1020)] + self.GOOD[1:]
        assert price_setter_errors(self.DECLS, self.SUBS, bad)

    def test_passive_not_resting_before_rejected(self):
        bad = self.GOOD + [TradeFill(1, 990, 1, 3, 2)]
        assert price_setter_errors(self.DECLS, self.SUBS, bad)

    # r5 (amended Rule 2 item (4), 2026-10-20): an incoming BUY ICEwPI (oid 4, base 1100,
    # peak 2, ppd -5) sweeps M = 3 against NON 3; k_e = 3 // 2 = 1, u = 1, so its remainder
    # rests as instance 1 at 1095 showing 2 - 1 = 1. A later SELL (oid 5) takes that 1 at
    # 1095, then instance 2 (size 2) at 1090.
    KE_DECLS = {3: Decl(3, 0, 990, 0, 0, 0), 4: Decl(4, 1, 1100, ICE_ER, 2, -5), 5: Decl(5, 2, 900, 0, 0, 0)}
    KE_SUBS = {0: 3, 1: 4, 2: 5}
    KE_GOOD = [TradeFill(1, 990, 3, 3, 4), TradeFill(2, 1095, 1, 4, 5), TradeFill(2, 1090, 2, 4, 5)]

    def test_rested_incoming_remainder_continues_at_instance_k_e(self):
        assert price_setter_errors(self.KE_DECLS, self.KE_SUBS, self.KE_GOOD) == []

    def test_rested_incoming_remainder_restarting_at_slice_0_rejected(self):
        """The superseded OQ-34 reading (remainder restarts at base, slice 0)."""
        bad = [self.KE_GOOD[0], self.KE_GOOD[1]._replace(price=1100), self.KE_GOOD[2]._replace(price=1095)]
        assert price_setter_errors(self.KE_DECLS, self.KE_SUBS, bad)


# ===========================================================================
# Slow: the frozen seed-112 draw, isolated (build_record) vs stacked (Session)
# ===========================================================================
def _trade_rows(trades: np.ndarray) -> List[Tuple[int, ...]]:
    t = np.asarray(trades)
    t = t[t[:, T_SEC] != -1]
    return [tuple(int(x) for x in r[list(TRADE_COLS)]) for r in t]


def _w(rows) -> int:
    return sum(abs(r[1]) * r[0] for r in rows)


def _resting(side_arr: np.ndarray) -> List[Tuple[int, int, int]]:
    a = np.asarray(side_arr)
    live = a[(a[:, 2] > 0) & (a[:, 1] > 0)]  # OrderSideFeat P=0, Q=1, OID=2
    return sorted((int(r[0]), int(r[1]), int(r[2])) for r in live)


def _bait_lanes():
    """Two leak baits built with TapeBuilder: contract 0 is BUY-only well above the
    fixture's prices, contract 95 is SELL-only well below. Alone they cannot trade;
    any leak into one another or into the fixture lanes would trade at once."""
    from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder
    out = []
    for cid, side, px, base in ((0, BUY, 1004400, 70_000_001), (95, SELL, 1003500, 80_000_001)):
        tb = TapeBuilder(contract_id=cid, first_timestamp=40_000)
        for k in range(8):
            tb.add(side=side, price_int=px + (k if side == BUY else -k), qty_int=40 + k, order_id=base + k)
        tape = tb.build(pad_extra=0)
        out.append(tape[tape[:, C_OID] >= 1])
    return out


@pytest.fixture(scope="module")
def probe():
    """Isolated records, a stacked env run in two width-4 batches, and an env-alone run."""
    if not FIXTURE.is_file():
        pytest.skip(f"C16 fixture absent: {FIXTURE}")
    import jax
    import jax.experimental
    import jax.numpy as jnp

    from omiecid_simulator.jaxomiecid.omie_tape_synth import stack_tapes
    from omiecid_simulator.omiecid.omie_session import AgentOrder, Session
    from tests.omiecid.reference.omie_invariants import build_record

    z = np.load(FIXTURE, allow_pickle=False)
    tape, params = np.asarray(z["tape"]), json.loads(str(z["params_json"]))
    real = tape[tape[:, C_OID] >= 1]
    cids = list(params["contract_ids"])
    lanes = [real[real[:, C_CID] == c] for c in cids]
    relabel = lanes[0].copy()
    relabel[:, C_CID] = 50  # same tape, another contract id (same order ids, deliberately)
    lanes = lanes + _bait_lanes() + [relabel]
    names = [str(c) for c in cids] + ["bait0", "bait95", f"{cids[0]}->50"]

    recs = [build_record(l) for l in lanes]
    iso = [_trade_rows(r.final_snapshot.trades) for r in recs]

    def run(env, idx):
        w = len(idx)
        st = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), w))
        st = st.replace(tape_idx=jnp.asarray(idx, dtype=jnp.int32))
        none = AgentOrder.none()  # the participant submits nothing; broadcast to every lane
        step = jax.vmap(env.step_env, in_axes=(0, 0, None))
        keys = jax.random.split(jax.random.PRNGKey(1), w)
        with jax.experimental.enable_x64():
            for _ in range(env.steps_per_episode):
                st, rep, done = step(keys, st, none)
                # full replay: no lane reaches ``done`` (the span below keeps every clock < GCT)
                assert not bool(np.asarray(done).any())
                # an empty order submits nothing and is credited nothing on any lane
                assert not bool(np.asarray(rep.submitted).any())
                assert not np.asarray(rep.cash_delta_eur).any()
                assert not np.asarray(rep.inventory_delta).any()
        bk = st.book
        return [{"trades": _trade_rows(np.asarray(bk.lob.trades)[k]),
                 "bids": _resting(np.asarray(bk.lob.bids)[k]), "asks": _resting(np.asarray(bk.lob.asks)[k]),
                 "meta": np.asarray(bk.iceberg_meta)[k][:int(np.asarray(bk.n_active)[k])],
                 "n_active": int(np.asarray(bk.n_active)[k]),
                 "flags": (bool(np.asarray(bk.book_full)[k]), bool(np.asarray(bk.truncated)[k]))}
                for k in range(w)]

    stack = stack_tapes(lanes, pad_extra=PAD_EXTRA)
    # The release Session ignores every call after ``done`` (the lane is frozen), and
    # its default span-scaled clock maps [0, S] seconds onto [0, GCT] minutes as
    # t_k = max(t_{k-1}, round(m_k * GCT / S)), with m_k the slice's largest timestamp
    # and done = (t >= GCT). With S the tape set's own largest timestamp, ``done`` fires
    # at the first slice whose m_k is within half a scaled minute of S -- possibly several
    # slices before the last real one -- and raising GCT does not help (the slice holding
    # S always reaches it). S = p is about contract isolation in the stacked slicing path,
    # not the clock (gate closure is C11's), so every row must be replayed: no slice may
    # reach GCT, i.e. round(M * GCT / S') < GCT, exactly S' > M * GCT / (GCT - 1/2), with
    # M the largest real timestamp over the probe's whole tape set (every lane). S' = 2M
    # is the margin (every clock <= round(GCT / 2)); the default GCT is kept, and every
    # run below asserts that no lane reaches ``done`` over ``steps_per_episode`` steps.
    span = 2.0 * float(max(max(int(l[:, C_TS].max()) for l in lanes), 1))
    horizon = {"orders_per_step": 4, "tape_seconds_span": span}
    env = Session(dict(horizon, tapes=stack))
    batches = ((0, 1, 2, 3), (4, 5, 0, 2))  # lanes 0 and 2 in both, at other positions
    stacked: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for idx in batches:
        for pos, out in zip(idx, run(env, list(idx))):
            stacked[pos].append(out)
    alone_env = Session(dict(horizon, tapes=stack_tapes([lanes[0]], pad_extra=PAD_EXTRA)))
    alone = run(alone_env, [0])[0]
    shrunk = None
    if SHRUNK_FIXTURE.is_file():
        zs = np.load(SHRUNK_FIXTURE, allow_pickle=False)
        shrunk = build_record(np.asarray(zs["tape"]))
    return {"lanes": lanes, "names": names, "cids": cids, "recs": recs, "iso": iso,
            "stacked": stacked, "alone": alone, "shrunk": shrunk}


def lane_evidence_errors(p: Dict[str, Any], iso_by_cid: Mapping[int, List[Tuple[int, ...]]],
                         tape_by_cid: Mapping[int, np.ndarray]) -> List[str]:
    """Recorded seed-112 lane tokens vs this file's own isolated runs: same trade count,
    same digest, and every traded order id is the lane's own."""
    errs = []
    dd = index(p)["draws"].get(FIXTURE_SEED)
    if dd is None:
        return [f"seed {FIXTURE_SEED} not recorded"]
    for ln in dd["lanes"]:
        rows = iso_by_cid.get(ln["cid"])
        if rows is None:
            errs.append(f"cid {ln['cid']} has no isolated run")
            continue
        if ln["ntrades"] != len(rows) or ln["digest"] != trade_digest(rows):
            errs.append(f"cid {ln['cid']}: recorded {ln['ntrades']}/{ln['digest']} vs isolated "
                        f"{len(rows)}/{trade_digest(rows)}")
        own = set(int(x) for x in tape_by_cid[ln["cid"]][:, C_OID])
        foreign = {o for r in rows for o in (r[2], r[3]) if o not in own}
        if foreign:
            errs.append(f"cid {ln['cid']}: foreign order ids {sorted(foreign)}")
    return errs


@pytest.mark.slow
@pytest.mark.jit
class TestIndependentProbe:
    """slow: jit-compiles the record builder and two Session instances (~60 s on CPU)."""

    def test_isolated_runs_reproduce_recorded_lane_digests(self, payload, probe):
        n = len(probe["cids"])
        iso = {c: probe["iso"][k] for k, c in enumerate(probe["cids"])}
        tapes = {c: probe["lanes"][k] for k, c in enumerate(probe["cids"])}
        assert len(iso) == n
        assert lane_evidence_errors(payload, iso, tapes) == []

    def test_stacked_lanes_equal_isolated_runs_strong_form(self, probe):
        """S = p: trades, resting book per side, iceberg side-channel, n_active, flags."""
        for k, outs in probe["stacked"].items():
            fs = probe["recs"][k].final_snapshot
            for out in outs:
                name = probe["names"][k]
                assert out["trades"] == probe["iso"][k], name
                assert out["bids"] == _resting(fs.bids), name
                assert out["asks"] == _resting(fs.asks), name
                assert out["n_active"] == fs.n_active, name
                assert np.array_equal(out["meta"], np.asarray(fs.iceberg_meta)), name
                assert out["flags"] == (bool(fs.book_full), bool(fs.truncated)), name

    def test_lane_outcome_independent_of_batch_company_and_position(self, probe):
        for k in (0, 2):
            a, b = probe["stacked"][k]
            assert a["trades"] == b["trades"] and a["bids"] == b["bids"] and a["asks"] == b["asks"]
            assert np.array_equal(a["meta"], b["meta"]) and a["flags"] == b["flags"]

    def test_no_lane_trades_a_foreign_order_id(self, probe):
        for k, outs in probe["stacked"].items():
            own = set(int(x) for x in probe["lanes"][k][:, C_OID])
            for out in outs:
                assert {o for r in out["trades"] for o in (r[2], r[3])} <= own, probe["names"][k]

    def test_leak_baits_never_trade(self, probe):
        for k in (3, 4):
            assert probe["iso"][k] == [], "bait trades alone: bad bait"
            assert all(o["trades"] == [] for o in probe["stacked"][k]), probe["names"][k]

    def test_contract_alone_in_the_environment_agrees(self, probe):
        assert probe["alone"]["trades"] == probe["iso"][0] == probe["stacked"][0][0]["trades"]
        assert probe["alone"]["bids"] == probe["stacked"][0][0]["bids"]
        assert probe["alone"]["asks"] == probe["stacked"][0][0]["asks"]
        assert probe["alone"]["flags"] == probe["stacked"][0][0]["flags"]

    def test_w_c_identical_in_every_arm_and_independent_of_contract_id(self, probe):
        """§8.1: d = 0.25 h for every contract, so W_c depends on the trades alone."""
        w0 = _w(probe["iso"][0])
        assert w0 > 0
        assert _w(probe["stacked"][0][0]["trades"]) == _w(probe["alone"]["trades"]) == w0
        assert probe["iso"][5] == probe["iso"][0]  # relabelled cid: identical trade list
        assert _w(probe["iso"][5]) == _w(probe["stacked"][5][0]["trades"]) == w0

    def test_i5_price_setter_relation_on_every_seed112_fill(self, probe):
        """Every fill of every seed-112 lane (isolated run; the stacked lanes are
        byte-identical, asserted above): the resting order sets the price (an
        iceberg at its slice's own limit) and the incoming order is the aggressor.
        Non-vacuous: icebergs are traded as the passive side."""
        n_ice = 0
        for k, _c in enumerate(probe["cids"]):
            rec = probe["recs"][k]
            decls, sub_oid, fills = _record_inputs(rec)
            assert price_setter_errors(decls, sub_oid, fills) == [], probe["names"][k]
            n_ice += sum(1 for f in fills if decls[f.passive].er == ICE_ER)
        assert n_ice > 0, "no iceberg traded as the passive side: the relation was vacuous"

    def test_i5_price_setter_on_the_r1_counterexample_lane(self, probe):
        """The r1 shrunk counterexample (contract 10 of seed 112 alone; three Rule-3 hop
        rows printed at the resting remainder's price on the pre-fix engine) now
        satisfies the relation on every fill."""
        rec = probe["shrunk"]
        if rec is None:
            pytest.skip(f"shrunk fixture absent: {SHRUNK_FIXTURE}")
        decls, sub_oid, fills = _record_inputs(rec)
        assert fills, "the counterexample lane no longer trades"
        assert price_setter_errors(decls, sub_oid, fills) == []
        assert any(decls[f.passive].er == ICE_ER and decls[f.passive].ppd != 0 for f in fills)

    def test_broken_foreign_order_id_copy_is_rejected(self, payload, probe, tmp_path):
        """A JSON whose seed-112 lane digest encodes a trade against another lane's order."""
        iso = {c: probe["iso"][k] for k, c in enumerate(probe["cids"])}
        tapes = {c: probe["lanes"][k] for k, c in enumerate(probe["cids"])}
        c0, c1 = probe["cids"][0], probe["cids"][1]
        alien = int(tapes[c1][0, C_OID])
        bad_rows = [tuple(r) for r in iso[c0]]
        bad_rows[0] = bad_rows[0][:2] + (alien,) + bad_rows[0][3:]
        b = _broken(payload)
        k, j, li = _first_lane(b, lambda dd, ln: dd["seed"] == FIXTURE_SEED and ln["cid"] == c0)
        _edit_lane(b, k, j, li, digest=trade_digest(bad_rows))
        b = _roundtrip(tmp_path, b)
        assert lane_evidence_errors(b, iso, tapes)
        bad_iso = dict(iso)
        bad_iso[c0] = bad_rows
        assert any("foreign" in e for e in lane_evidence_errors(b, bad_iso, tapes))


# ===========================================================================
# Slow: hand-built multi-contract sessions on ContractBook's public API
# ===========================================================================
# An order: (side, price in cents, qty in tenths of MW, oid, exec, peak tenths, ppd cents).
NON_, IOC_, FOK_, ICE_ = 0, 1, 2, 3
CB_BOOK_DEPTH = 5  # small, so lane D can fill one side (book_full) in a few steps

HAND_LANES: Dict[str, Tuple[int, List[Tuple[int, int, int, int, int, int, int]]]] = {
    # Rule 2 multi-slice (PPD = 0), IOC across iceberg + NON, FOK onto a NON.
    "A": (0, [(SELL, 5000, 60, 1001, ICE_, 20, 0), (BUY, 5100, 50, 1002, NON_, 0, 0),
              (SELL, 5040, 15, 1003, NON_, 0, 0), (BUY, 5070, 40, 1004, IOC_, 0, 0),
              (BUY, 4900, 20, 1005, NON_, 0, 0), (SELL, 4850, 20, 1006, FOK_, 0, 0),
              (BUY, 4700, 10, 1007, NON_, 0, 0)]),
    # ICEwPI SELL drifting up; v2.14 re-ranking (a NON between slices 1 and 2);
    # slice that no longer crosses stops the sweep; FOK takes the last slice.
    "B": (37, [(SELL, 5000, 100, 2001, ICE_, 20, 30), (SELL, 5045, 10, 2002, NON_, 0, 0),
               (BUY, 5070, 70, 2003, NON_, 0, 0), (BUY, 5100, 30, 2004, NON_, 0, 0),
               (SELL, 5100, 10, 2005, NON_, 0, 0), (BUY, 5120, 20, 2006, FOK_, 0, 0),
               (SELL, 5500, 10, 2007, NON_, 0, 0)]),
    # Incoming icebergs as aggressors (total-quantity sweep, remainder rests as ICE),
    # and a BUY ICEwPI drifting down taken by one NON across three slices.
    "C": (95, [(SELL, 5020, 30, 3001, NON_, 0, 0), (SELL, 5030, 40, 3002, ICE_, 20, 20),
               (BUY, 5100, 90, 3003, ICE_, 20, 0), (SELL, 5080, 50, 3004, ICE_, 10, 0),
               (BUY, 4950, 30, 3005, ICE_, 10, -10), (SELL, 4930, 30, 3006, NON_, 0, 0),
               (BUY, 4800, 10, 3007, NON_, 0, 0)]),
    # Leak bait and flag lane: six BUYs above every other lane's asks (a leak would
    # trade), one more than CB_BOOK_DEPTH (book_full), then a 70-slice iceberg
    # (ceil(70/1) > 64: truncated). Alone it never trades.
    "D": (61, [(BUY, 5095 + 5 * i, 10, 4001 + i, NON_, 0, 0) for i in range(6)]
          + [(SELL, 6000, 700, 4007, ICE_, 10, 0)]),
}

# Spec-derived expected trades: lane -> step -> [(price cents, qty tenths, passive oid)];
# the aggressor is always the step's own order.
HAND_TRADES: Dict[str, Dict[int, List[Tuple[int, int, int]]]] = {
    "A": {1: [(5000, 20, 1001), (5000, 20, 1001), (5000, 10, 1001)],
          3: [(5000, 10, 1001), (5040, 15, 1003)],
          5: [(4900, 20, 1005)]},
    "B": {2: [(5000, 20, 2001), (5030, 20, 2001), (5045, 10, 2002), (5060, 20, 2001)],
          3: [(5090, 20, 2001)],
          4: [(5100, 10, 2004)],
          5: [(5120, 20, 2001)]},
    # Lane C, re-derived at r5 from market-spec v2.14 as amended 2026-10-20 (§4.2.1 Rule 2
    # *Incoming iceberg* items (1)-(6), Rule 3 *ICE aggressor*). The r2 table encoded the
    # superseded "one row per counter bid" reading and the display min(P, R).
    #  s0 SELL NON 3001 @5020 x30 rests. s1 SELL ICEwPI 3002 @5030 x40 P20 d+20: no bid,
    #     M = 0, rests as instance 0 at 5030 showing 20, hidden 20.
    #  s2 BUY ICE 3003 @5100 x90 P20 d0: instances of 20 (0-20, 20-40, 40-60, 60-80, 80-90),
    #     every limit 5100 (item (6), d = 0). Rows split at resting entries AND at incoming
    #     instance boundaries (item (2)):
    #       inst0 x20 vs 3001 @5020 | inst1 x10 vs 3001 @5020 (3001 done, M = 30)
    #       inst1 x10 vs 3002 slice0 @5030 | inst2 x10 vs 3002 slice0 @5030 (slice0 done)
    #       3002 slice1 @5030 + 20 = 5050 x20 (only ask; re-ranked):
    #       inst2 x10 @5050 | inst3 x10 @5050 (3002 done, M = 70)
    #     r2: [(5020, 30, 3001), (5030, 20, 3002), (5050, 20, 3002)].
    #     Remainder (item (4)): R = 20, k_e = 70 // 20 = 3, u = 70 - 60 = 10, rests at
    #     5100 + 3 * 0 = 5100 showing v1 = min(20 - 10, 20) = 10, hidden 10
    #     (r2: showed min(P, R) = 20, hidden 0).
    #  s3 SELL ICE 3004 @5080 x50 P10 d0: instances of 10. Best bid 3003 @5100 visible 10:
    #       inst0 x10 vs 3003 (rested instance 3) @5100; 3003 regenerates its last 10 at
    #       5100 (new timestamp); inst1 x10 vs it @5100 (3003 done, M = 20). Two resting
    #       instances -> two rows (and the incoming instance boundary falls at 10 too).
    #     r2: [(5100, 20, 3003)] (one counter bid of 20 under the retired display).
    #     Remainder: R = 30, k_e = 20 // 10 = 2, u = 0, rests at 5080 showing
    #     min(10 - 0, 30) = 10, hidden 20 (same as r2).
    #  s4 BUY ICEwPI 3005 @4950 x30 P10 d-10: best ask 5080 does not cross; M = 0, rests
    #     as instance 0 @4950 showing 10, hidden 20.
    #  s5 SELL NON 3006 @4930 x30: 3005 slices @4950, @4940, @4930 (each crosses 4930), x10
    #     each, resting instances, one row each (unchanged).
    #  s6 BUY NON 3007 @4800 x10 rests (no ask <= 4800).
    "C": {2: [(5020, 20, 3001), (5020, 10, 3001), (5030, 10, 3002), (5030, 10, 3002),
              (5050, 10, 3002), (5050, 10, 3002)],
          3: [(5100, 10, 3003), (5100, 10, 3003)],
          5: [(4950, 10, 3005), (4940, 10, 3005), (4930, 10, 3005)]},
    "D": {},
}

# Spec-derived final resting book: lane -> (bids, asks) as {(price cents, qty tenths, oid)},
# plus hidden quantity per live iceberg.
HAND_FINAL = {
    "A": ({(4700, 10, 1007)}, set(), {}),
    "B": (set(), {(5500, 10, 2007)}, {}),
    # Lane C (r5 re-derivation above): 3001, 3002, 3003, 3005, 3006 exhausted; bids 3007
    # @4800 x10; asks 3004's remainder as instance k_e = 2 @5080 + 2 * 0, showing
    # v1 = min(10 - 0, 30) = 10, hidden 30 - 10 = 20. Unchanged from r2.
    "C": ({(4800, 10, 3007)}, {(5080, 10, 3004)}, {3004: 20}),
}

# r5: lane C's book right after step 2 (amended Rule 2 item (4)): 3003's remainder R = 20
# rests as instance k_e = 3 at 5100 showing v1 = P - u = 10, hidden 10; no ask left.
# The final book cannot see this (3003 is consumed at step 3), so it is pinned here.
HAND_C_AFTER_STEP2 = ({(5100, 10, 3003)}, set(), {3003: 10})


# r5: an incoming BUY ICEwPI whose sweep stops at an instance limit and whose remainder
# rests mid-ladder (amended Rule 2 items (2), (4), (6); Rule 3 *ICE aggressor*). Run alone
# on its own contract (50), outside the stacked/interleaved sessions.
#  s0 SELL NON 5001 @5000 x30 rests; s1 SELL NON 5003 @5095 x10 rests.
#  s2 BUY ICE 5002 @5100 x100 P20 d-10: instance k limit 5100 - 10k.
#     inst0 (lim 5100) x20 vs 5001 @5000 | inst1 (lim 5090) x10 vs 5001 @5000 (M = 30).
#     Next ask 5095: inst1's limit 5090 does not cross -> the sweep stops (item (6)),
#     although b = 5100 would cross. R = 70, k_e = 30 // 20 = 1, u = 10: rests at
#     5100 + 1 * (-10) = 5090 showing v1 = min(20 - 10, 70) = 10, hidden 60.
#     (Superseded reading: rest at 5100 showing min(P, R) = 20, which would cross 5095.)
#  s3 SELL NON 5004 @5000 x40: 5002's instance 1 x10 @5090, instance 2 x20 @5080,
#     instance 3 x10 of 20 @5070 (each crosses 5000; price = resting instance's limit).
#     5002 keeps 30: instance 3 @5070 showing 10, hidden 20.
HAND_KE_CID = 50
HAND_KE_ROWS = [(SELL, 5000, 30, 5001, NON_, 0, 0), (SELL, 5095, 10, 5003, NON_, 0, 0),
                (BUY, 5100, 100, 5002, ICE_, 20, -10), (SELL, 5000, 40, 5004, NON_, 0, 0)]
HAND_KE_TRADES = {2: [(5000, 20, 5001), (5000, 10, 5001)],
                  3: [(5090, 10, 5002), (5080, 20, 5002), (5070, 10, 5002)]}
HAND_KE_BOOK = {2: ({(5090, 10, 5002)}, {(5095, 10, 5003)}, {5002: 60}),
                3: ({(5070, 10, 5002)}, {(5095, 10, 5003)}, {5002: 20})}


def _cb_lane_ts(name: str, step: int) -> int:
    return {"A": 1000, "B": 2000, "C": 3000, "D": 4000, "KE": 5000}[name] + 7 * step + 1


def _cb_orders(name: str):
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
    cid, rows = (HAND_KE_CID, HAND_KE_ROWS) if name == "KE" else HAND_LANES[name]
    out = []
    for step, (side, cents, tenths, oid, er, peak, ppd) in enumerate(rows):
        out.append(make_order(contract_id=cid, side=Side(side), price=cents / 100, quantity=tenths / 10,
                              order_id=oid, agent_id=oid % 7, timestamp=_cb_lane_ts(name, step),
                              exec_restriction=ExecRestriction(er), peak_qty=peak / 10, ppd=ppd / 100))
    return out


def _cb_decls(name: str, off: int) -> Tuple[Dict[int, Decl], Dict[int, int]]:
    _cid, rows = (HAND_KE_CID, HAND_KE_ROWS) if name == "KE" else HAND_LANES[name]
    decls = {oid: Decl(oid, s, cents + off, er, peak, ppd) for s, (_sd, cents, _q, oid, er, peak, ppd) in enumerate(rows)}
    return decls, {s: r[3] for s, r in enumerate(rows)}


def _leaves(state) -> List[np.ndarray]:
    import jax
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(state)]


def _lane(state, k: int):
    import jax
    return jax.tree_util.tree_map(lambda x: x[k], state)


def _new_rows(before, after) -> np.ndarray:
    b, a = np.asarray(before), np.asarray(after)
    return a[np.any(b != a, axis=1)]


@pytest.fixture(scope="module")
def cb_probe():
    """Each hand lane alone, the four lanes vmap-stacked in two lane orders, and an
    interleaved session over four per-contract books with a staggered gate (lane C
    expires at its GCT while the others keep trading)."""
    import jax
    import jax.experimental
    import jax.numpy as jnp

    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.contract_book import ContractBook
    from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig

    pc, mc = OMIEProductConfig(), OMIEMarketConfig(book_depth=CB_BOOK_DEPTH)
    off = int(round(-pc.price_min / pc.price_tick))

    def contract(cid):
        start = cid * 15
        return Contract(contract_id=cid, delivery_start_minutes=start, delivery_end_minutes=start + 15,
                        gct_minutes=start - 60, status=ContractStatus.OPEN)

    names = list(HAND_LANES)
    books = {n: ContractBook(contract(HAND_LANES[n][0]), pc, mc) for n in names}
    orders = {n: _cb_orders(n) for n in names}

    books["KE"] = ContractBook(contract(HAND_KE_CID), pc, mc)
    orders["KE"] = _cb_orders("KE")
    alone, per_step, accepted, snaps = {}, {}, {}, {}
    for n in names + ["KE"]:
        st, steps, acc, sn = books[n].init(), {}, [], {}
        for s, o in enumerate(orders[n]):
            before = st.lob.trades
            st, r = books[n].submit_order(st, o)
            acc.append(bool(r.accepted))
            steps[s] = [tuple(int(x) for x in row) for row in _new_rows(before, st.lob.trades)]
            sn[s] = st
        alone[n], per_step[n], accepted[n], snaps[n] = st, steps, acc, sn

    def as_batch(os_):
        return jax.tree_util.tree_map(
            lambda *x: jnp.asarray(np.array(x, dtype=np.float32 if isinstance(x[0], float) else np.int32)), *os_)

    stacked = {}
    shared = books["A"]  # one book object: the per-lane contract lives in each order
    vsubmit = jax.vmap(shared.submit_order)
    for order_ in (("A", "B", "C", "D"), ("D", "C", "A", "B")):
        st = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[shared.init()] * len(order_))
        for s in range(len(orders["A"])):
            with jax.experimental.enable_x64():
                st, _r = vsubmit(st, as_batch([orders[n][s] for n in order_]))
        stacked[order_] = {n: _lane(st, k) for k, n in enumerate(order_)}

    # Interleaved: C runs first (steps interleaved with A's), then expires at its gate;
    # B and D join afterwards; one Python session, four per-contract books.
    schedule = []
    for s in range(7):
        schedule += [("C", s), ("A", s)]
    schedule += [("GCT", "C")]
    for s in range(7):
        schedule += [("B", s), ("D", s)]
    inter = {n: books[n].init() for n in names}
    for n, s in schedule:
        if n == "GCT":
            inter[s] = books[s].expire_at_gct(inter[s])
        else:
            inter[n], _r = books[n].submit_order(inter[n], orders[n][s])
    c_expired_alone = books["C"].expire_at_gct(alone["C"])
    return {"names": names, "alone": alone, "per_step": per_step, "accepted": accepted, "stacked": stacked,
            "snaps": snaps,
            "inter": inter, "c_expired_alone": c_expired_alone, "off": off, "qty_scale": int(round(0.1 / pc.qty_tick))}


@pytest.mark.slow
@pytest.mark.jit
class TestContractBookStackedProbe:
    """slow: jit-compiles ContractBook.submit_order unbatched and vmapped (~30-60 s on CPU)."""

    def test_every_hand_order_accepted_alone(self, cb_probe):
        for n in ("A", "B", "C"):
            assert all(cb_probe["accepted"][n]), (n, cb_probe["accepted"][n])

    def test_alone_trades_equal_spec_derived_rows(self, cb_probe):
        off, qs = cb_probe["off"], cb_probe["qty_scale"]
        for n in cb_probe["names"]:
            oids = [r[3] for r in HAND_LANES[n][1]]
            for s, rows in cb_probe["per_step"][n].items():
                got = [(r[0] - off, abs(r[1]) // qs, r[2], r[3]) for r in rows]
                want = [(p, q, po, oids[s]) for p, q, po in HAND_TRADES[n].get(s, [])]
                assert got == want, (n, s)

    def test_alone_final_book_equals_spec_derived_book(self, cb_probe):
        from omiecid_simulator.omiecid.iceberg_manager import COL_ORDER_ID, COL_REMAINING_HIDDEN
        off, qs = cb_probe["off"], cb_probe["qty_scale"]
        for n, (bids, asks, hidden) in HAND_FINAL.items():
            st = cb_probe["alone"][n]
            got_b = {(p - off, q // qs, o) for p, q, o in _resting(st.lob.bids)}
            got_a = {(p - off, q // qs, o) for p, q, o in _resting(st.lob.asks)}
            assert (got_b, got_a) == (bids, asks), n
            meta = np.asarray(st.iceberg_meta)[:int(st.n_active)]
            got_h = {int(r[COL_ORDER_ID]): int(r[COL_REMAINING_HIDDEN]) // qs for r in meta
                     if int(r[COL_REMAINING_HIDDEN]) > 0}
            assert got_h == hidden, n

    @staticmethod
    def _book(cb_probe, st):
        from omiecid_simulator.omiecid.iceberg_manager import COL_ORDER_ID, COL_REMAINING_HIDDEN
        off, qs = cb_probe["off"], cb_probe["qty_scale"]
        got_b = {(p - off, q // qs, o) for p, q, o in _resting(st.lob.bids)}
        got_a = {(p - off, q // qs, o) for p, q, o in _resting(st.lob.asks)}
        meta = np.asarray(st.iceberg_meta)[:int(st.n_active)]
        got_h = {int(r[COL_ORDER_ID]): int(r[COL_REMAINING_HIDDEN]) // qs for r in meta
                 if int(r[COL_REMAINING_HIDDEN]) > 0}
        return got_b, got_a, got_h

    def test_lane_c_incoming_remainder_rests_as_instance_k_e(self, cb_probe):
        """r5, amended Rule 2 item (4): after step 2, 3003's remainder shows P - u = 10
        (not min(P, R) = 20) at 5100 and hides 10."""
        assert self._book(cb_probe, cb_probe["snaps"]["C"][2]) == HAND_C_AFTER_STEP2

    def test_incoming_icewpi_stops_at_instance_limit_and_rests_at_k_e(self, cb_probe):
        """r5, amended Rule 2 items (2), (4), (6): the sweep stops at the first instance
        that cannot trade, the remainder rests at b + k_e d showing P - u, and later
        trades continue the instance ladder from k_e (lane KE, alone)."""
        off, qs = cb_probe["off"], cb_probe["qty_scale"]
        assert all(cb_probe["accepted"]["KE"]), cb_probe["accepted"]["KE"]
        oids = [r[3] for r in HAND_KE_ROWS]
        for s, rows in cb_probe["per_step"]["KE"].items():
            got = [(r[0] - off, abs(r[1]) // qs, r[2], r[3]) for r in rows]
            assert got == [(p, q, po, oids[s]) for p, q, po in HAND_KE_TRADES.get(s, [])], s
        for s, want in HAND_KE_BOOK.items():
            assert self._book(cb_probe, cb_probe["snaps"]["KE"][s]) == want, s
        decls, sub_oid = _cb_decls("KE", off)
        fills = [TradeFill(s, r[0], abs(r[1]), r[2], r[3])
                 for s, rows in sorted(cb_probe["per_step"]["KE"].items()) for r in rows]
        assert price_setter_errors(decls, sub_oid, fills) == []

    def test_i5_price_setter_relation_on_every_hand_fill(self, cb_probe):
        """Every fill: the resting order sets the price (a resting iceberg at base +
        k * ppd), the incoming order (NON, IOC, FOK or an incoming ICE) is the aggressor."""
        off, n_drift, n_ice_aggr = cb_probe["off"], 0, 0
        for n in cb_probe["names"]:
            decls, sub_oid = _cb_decls(n, off)
            fills = [TradeFill(s, r[0], abs(r[1]), r[2], r[3])
                     for s, rows in sorted(cb_probe["per_step"][n].items()) for r in rows]
            assert price_setter_errors(decls, sub_oid, fills) == [], n
            n_drift += sum(1 for f in fills if decls[f.passive].ppd != 0 and f.price != decls[f.passive].price)
            n_ice_aggr += sum(1 for f in fills if decls[f.aggressor].er == ICE_ER)
        assert n_drift >= 4 and n_ice_aggr >= 4, (n_drift, n_ice_aggr)

    def test_flags_belong_to_their_own_lane_alone(self, cb_probe):
        for n in cb_probe["names"]:
            st = cb_probe["alone"][n]
            want = (n == "D", n == "D")
            assert (bool(st.book_full), bool(st.truncated)) == want, n
        assert cb_probe["per_step"]["D"] and not any(cb_probe["per_step"]["D"].values()), "bait lane traded alone"

    @pytest.mark.parametrize("order_", [("A", "B", "C", "D"), ("D", "C", "A", "B")])
    def test_stacked_lane_state_equals_alone_state(self, cb_probe, order_):
        """S = p, strongest form: the whole OMIEBookState pytree of each stacked lane
        (trades incl. stamps, bids, asks, iceberg side-channel, n_active, ts_counter,
        book_full, truncated) equals the contract run alone, in either lane order."""
        trades = [np.asarray(cb_probe["alone"][n].lob.trades).tobytes() for n in order_]
        assert len(set(trades)) == len(order_), "lanes indistinguishable: a lane swap would go unseen"
        for n in order_:
            a, b = _leaves(cb_probe["alone"][n]), _leaves(cb_probe["stacked"][order_][n])
            assert len(a) == len(b)
            for i, (x, y) in enumerate(zip(a, b)):
                assert np.array_equal(x, y), (n, i)

    def test_interleaved_per_contract_books_equal_alone_with_staggered_gate(self, cb_probe):
        """One session over four books, C expiring at its gate mid-session: each lane
        equals its alone run (C: alone then expired); C's gate disturbs no other lane."""
        for n in cb_probe["names"]:
            ref = cb_probe["c_expired_alone"] if n == "C" else cb_probe["alone"][n]
            for i, (x, y) in enumerate(zip(_leaves(ref), _leaves(cb_probe["inter"][n]))):
                assert np.array_equal(x, y), (n, i)
        st = cb_probe["inter"]["C"]
        assert _resting(st.lob.bids) == [] and _resting(st.lob.asks) == []
