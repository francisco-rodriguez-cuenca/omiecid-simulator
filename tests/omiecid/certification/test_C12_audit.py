"""K4 audit twin for OMIE-CERT-C12-1 (settlement and the cash leg).

Fresh-context, adversarial audit (high-risk scenario, independence rule 4). The
expectation is re-derived from the rules alone and judged against the EMITTED
results JSON, whatever its status. It never opens or executes
``notebooks/certification/C12_settlement_cash_leg.ipynb`` and never reads
``contract_book.py``, ``iceberg_manager.py``, ``order_validator.py`` or
``settlement.py`` (independence rules 3/4); the settlement entry point is used
through its public signature ``compute_agent_pnl(trades, agent_id)`` only.

Rules
-----
* §8.1 / §2.3 / §11.1: ``V (EUR) = Q (MW) x p (EUR/MWh) x d (h)``, ``d = 0.25`` for a
  15-min contract. On the tick grid (price tick 0.01 EUR/MWh, offset-encoded with
  ``O = 999 900`` so that ``price_int = 0`` is -9 999.00; qty tick 0.1 MW) a trade
  row's exact value is the integer ``D_i = q (p_int - O)`` ticks and
  ``V_i = k D_i`` with ``k = 0.1 * 0.01 * 0.25 = 2.5e-4`` EUR per tick-tick.
* §8.4: the seller receives ``+V``, the buyer pays ``-V`` (cash leg; the energy value
  is exogenous). A participant's cash leg is ``D_a = sum_sells D_i - sum_buys D_i``; a
  self-match row (§6.6, legal) contributes ``+D_i - D_i = 0``.
* INVARIANTS-1 V10 / I6(b): per submission ``s`` with rows ``i``: ``W_s = sum p_i q_i``,
  ``Q_s = sum q_i``, ``D_s = W_s - O Q_s``; ``traded == (m_s > 0)``, reported quantity
  equals ``Q_s`` exactly, ``|settlement - k D_s| <= tau_s`` with
  ``tau_s = k u (c_s W_s + O Q_s + 5|D_s|)(1 + (c_s + 8) u)``, ``u = 2^-24``. The per-draw
  JSON carries no trade-row timestamps, so the audit uses ``c_s = m_s`` (the
  chunk-agnostic worst case, an upper bound of V10's ``c_s``); the slow probe below
  uses the exact ``c_s`` from the engine's own ``(sec, nsec)``.
* I6(a): the engine's per-participant settlements, over exactly the agents on trade
  rows, sum to zero within ``sum_s tau_s``.
* Per-participant reconciliation (scenario-ladder §C12 hazard paragraph, review V3-03):
  every participant's settlement equals the independent per-row recomputation
  ``k D_a`` within ``tau_a``. Derivation of ``tau_a`` from the documented settlement
  arithmetic (OQ-18: int64 two-term accumulators inside ``compute_agent_pnl``; review
  V3-04: the int64 accumulators are internal, the output is a float32 EUR value): the
  accumulators are exact, so the only roundings are the float32 casts of the
  accumulated terms (either net, or one per side), the offset product, the
  subtraction(s) and the products by ``fl(0.1)``, ``fl(0.01)`` (0.25 exact). Over the
  participant's legs ``L_a`` (a self-match row counted on both sides):
  ``tau_a = k u (W_a + O Q_a + 8 G_a)(1 + 10 u)`` with ``W_a = sum p q``,
  ``Q_a = sum q``, ``G_a = sum |p - O| q`` (gross over legs). It bounds every reading
  of the "two-term" accumulator (net or per side). It is deliberately tighter than a
  per-leg float32 accumulation, which the documentation excludes.
* OQ-18: ``q p_int`` reaches ~2e10 and ``|D_a|`` exceeds 2^31 at the envelope; an int32
  wrap anywhere in the cash leg is a defect. Every integer in the JSON is checked
  in full width.
* Euro figures are checked within the derived tolerance only, never exactly.

Per-draw record layout as emitted (``observed`` / ``expected`` JSON strings):
``trades`` rows ``[buyer_oid, seller_oid, buyer_agent, seller_agent, price_int,
qty_int, submission]`` (submission 0-based, its oid ``s + 1``); ``ledger``
``{agent: [bought_qty, sold_qty, D_a]}``; ``submissions``
``{s: [traded, qty_int, D_s]}``; ``trade_value_ticks``; ``gross_value_ticks``;
``settlement_tape_shape``; ``settled_agents``; ``filled``. The tolerance assertion
carries ``observed = {"agents": {a: [settlement, ...]}, "submissions": {s: report}}``.

Results JSON: ``$OMIE_CERT_RESULTS_DIR/C12.json`` else
``notebooks/certification/results/C12.json``; absent -> skip.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from notebooks.certification.cert_support import (
    SpecQuote,
    check_quote,
    derive_failure_kinds,
    quote_digest,
    validate_results,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "notebooks" / "certification" / "results"
CERT_ID = "C12"

# --- the specification's constants (§2.3, §8.1), not the engine's -------------
PRICE_TICK = 0.01            # EUR/MWh
QTY_TICK = 0.1               # MW
D_HOURS = 0.25               # h, 15-min contract
PRICE_MIN_EUR = -9999.0
PRICE_MAX_EUR = 9999.0
O = 999_900                  # round(-price_min / price_tick)
P_INT_MAX = 1_999_800        # +9 999.00
Q_INT_MIN, Q_INT_MAX = 1, 9_990
K = QTY_TICK * PRICE_TICK * D_HOURS    # EUR per tick*tick (2.5e-4)
U = 2.0 ** -24               # float32 unit roundoff
TWO31 = 2 ** 31

DRAWS = 500
ENVELOPE = {"resting_per_side": 80, "trades": 80, "hops": 160, "concurrent_icebergs": 800}
RECORD_KEYS = frozenset({
    "accepted", "cancelled", "evicted", "filled", "gross_value_ticks", "hidden_remaining",
    "ledger", "rejected", "settled_agents", "settlement_tape_shape", "still_resting",
    "submissions", "trade_value_ticks", "trades",
})
SEED_RE = re.compile(r"^seed (\d+) \[(.+)\]$")
TOL_RE = re.compile(r"^seed (\d+)( \(eager\))?: settlements and reports within the derived tolerance$")
EJ_RE = re.compile(r"^eager == jitted \(seed (\d+)\)$")
REGIMES = ("ordinary", "negative", "zero", "large_qty", "extreme_high", "extreme_low")
SELF_PAIR = "a self-match pair (a participant whose cash leg is exactly zero)"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _results_path() -> Path:
    d = os.environ.get("OMIE_CERT_RESULTS_DIR")
    return (Path(d) if d else DEFAULT_RESULTS_DIR) / f"{CERT_ID}.json"


@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    p = _results_path()
    if not p.is_file():
        pytest.skip(f"C12 results JSON absent at {p} (run notebook section 8 first)")
    return json.loads(p.read_text(encoding="utf-8"))


def _reload(obj: Dict[str, Any], tmp_path: Path, name: str) -> Dict[str, Any]:
    f = tmp_path / name
    f.write_text(json.dumps(obj), encoding="utf-8")
    return json.loads(f.read_text(encoding="utf-8"))


def _load(x):
    return json.loads(x) if isinstance(x, str) else x


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _wrap32(x: int) -> int:
    return (int(x) + TWO31) % (2 ** 32) - TWO31


# ---------------------------------------------------------------------------
# The rules, as arithmetic
# ---------------------------------------------------------------------------
def trade_value(p: int, q: int) -> int:
    """Exact §8.1 value of one row in tick*tick units (int, full width)."""
    return int(q) * (int(p) - O)


def tau_sub(rows: List[Tuple[int, int]], chunks: Optional[int] = None) -> float:
    """V10 tau_s (own implementation); ``chunks=None`` -> c_s = m_s."""
    W = sum(int(p) * int(q) for p, q in rows)
    Q = sum(int(q) for _, q in rows)
    D = W - O * Q
    c = len(rows) if chunks is None else int(chunks)
    return K * U * (c * W + O * Q + 5 * abs(D)) * (1 + (c + 8) * U)


def tau_agent(legs: List[Tuple[int, int, int]]) -> float:
    """Per-participant tolerance over its legs ``(sign, p, q)`` (see module docstring)."""
    W = sum(int(p) * int(q) for _, p, q in legs)
    Q = sum(int(q) for _, _, q in legs)
    G = sum(abs(int(p) - O) * int(q) for _, p, q in legs)
    return K * U * (W + O * Q + 8 * G) * (1 + 10 * U)


def legs_by_agent(trades) -> Dict[int, List[Tuple[int, int, int]]]:
    legs: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for _bo, _so, ba, sa, p, q, _s in trades:
        legs[int(sa)].append((+1, int(p), int(q)))
        legs[int(ba)].append((-1, int(p), int(q)))
    return dict(legs)


def cash_leg(legs) -> int:
    return sum(sg * trade_value(p, q) for sg, p, q in legs)


def rows_by_submission(trades) -> Dict[int, List[Tuple[int, int]]]:
    out: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for t in trades:
        out[int(t[6])].append((int(t[4]), int(t[5])))
    return dict(out)


# ---------------------------------------------------------------------------
# Index of the per-seed assertions
# ---------------------------------------------------------------------------
def seed_index(payload) -> Dict[int, Dict[str, Any]]:
    """seed -> {"rec": (i, a, m), "tol": (i, a), "tol_eager": (i, a), "ej": (i, a)}."""
    idx: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for i, a in enumerate(payload["assertions"]):
        n = a["name"]
        m = SEED_RE.match(n)
        if m and not n.startswith("seed " + m.group(1) + ":"):
            idx[a["seed"]].setdefault("recs", []).append((i, a, m))
            continue
        t = TOL_RE.match(n)
        if t:
            idx[a["seed"]].setdefault("tol_eager" if t.group(2) else "tol", []).append((i, a, t))
            continue
        e = EJ_RE.match(n)
        if e:
            idx[a["seed"]].setdefault("ej", []).append((i, a, e))
    return dict(idx)


# ---------------------------------------------------------------------------
# Independent judgement of one per-draw record
# ---------------------------------------------------------------------------
def judge_record(rec: Dict[str, Any], tag: str = "") -> List[str]:
    errs: List[str] = []
    e = errs.append
    if not isinstance(rec, dict):
        return [f"{tag}per-draw record not a mapping"]
    missing = RECORD_KEYS - set(rec)
    if missing:
        return [f"{tag}per-draw record not in full: missing {sorted(missing)}"]

    accepted = rec["accepted"]
    subs = rec["submissions"]
    n_sub = len(subs)
    if sorted(int(k) for k in subs) != list(range(n_sub)):
        e(f"{tag}submissions keys {sorted(subs)} are not 0..{n_sub - 1}")
    if not all(_is_int(x) for x in accepted) or sorted(accepted) != accepted:
        e(f"{tag}accepted {accepted} malformed")
    if rec["rejected"]:
        e(f"{tag}rejected {rec['rejected']}: the drawn orders are all valid (§2.3 bounds)")
    if set(accepted) != set(range(1, n_sub + 1)):
        e(f"{tag}accepted {accepted} is not every submission's oid 1..{n_sub}")

    trades = rec["trades"]
    if not trades:
        e(f"{tag}no trade rows: the cash-leg check would run on nothing (I9)")
    agent_of: Dict[int, int] = {}
    for t in trades:
        if not (isinstance(t, list) and len(t) == 7 and all(_is_int(x) for x in t)):
            e(f"{tag}trade row {t!r} is not 7 integers")
            continue
        bo, so, ba, sa, p, q, s = t
        if not (0 <= p <= P_INT_MAX):
            e(f"{tag}trade {t}: price_int {p} outside [-9999, +9999] EUR/MWh (§2.3)")
        if not (Q_INT_MIN <= q <= Q_INT_MAX):
            e(f"{tag}trade {t}: qty_int {q} outside [0.1, 999] MW (§2.3)")
        if bo == so:
            e(f"{tag}trade {t}: one order on both sides (a trade is between two orders, §5.3)")
        if not (0 <= s < n_sub):
            e(f"{tag}trade {t}: writing submission {s} out of range")
        elif s + 1 not in (bo, so):
            e(f"{tag}trade {t}: submission {s}'s own order {s + 1} is not a party")
        else:
            maker = so if bo == s + 1 else bo
            if not (1 <= maker < s + 1):
                e(f"{tag}trade {t}: maker {maker} was not submitted before the aggressor {s + 1}")
        for oid, ag in ((bo, ba), (so, sa)):
            if oid not in accepted:
                e(f"{tag}trade {t}: party {oid} not accepted")
            if agent_of.setdefault(oid, ag) != ag:
                e(f"{tag}order {oid} appears with two agents {agent_of[oid]} and {ag}")
    if errs:
        return errs

    # --- per-row exact values (§8.1), full width (OQ-18) ----------------------------
    tv = rec["trade_value_ticks"]
    want_tv = [trade_value(t[4], t[5]) for t in trades]
    if not isinstance(tv, list) or len(tv) != len(trades):
        e(f"{tag}trade_value_ticks has {len(tv) if isinstance(tv, list) else tv!r} entries, "
          f"{len(trades)} rows written")
    else:
        for j, (got, want) in enumerate(zip(tv, want_tv)):
            if not _is_int(got) or got != want:
                e(f"{tag}row {j} {trades[j]}: value {got!r} ticks != q (p_int - O) = {want} (§8.1)")
    gross = sum(abs(v) for v in want_tv)
    if not _is_int(rec["gross_value_ticks"]) or rec["gross_value_ticks"] != gross:
        e(f"{tag}gross_value_ticks {rec['gross_value_ticks']!r} != sum |D_i| = {gross}")
    if gross <= 0:
        e(f"{tag}gross cash leg sum |V| = 0: vacuous (I9, V3-03)")

    # --- the settlement tape and the participant set (I6(a), F8) ---------------------
    if rec["settlement_tape_shape"] != [len(trades), 6]:
        e(f"{tag}settlement tape shape {rec['settlement_tape_shape']} != (M, 6) = "
          f"({len(trades)}, 6) (OMIE-AGENT-PNL-SHAPE-GUARD-1)")
    legs = legs_by_agent(trades)
    agents = sorted(legs)
    if rec["settled_agents"] != agents:
        e(f"{tag}settled agents {rec['settled_agents']} != the agents on trade rows {agents} "
          f"(a participant leg dropped or invented)")

    # --- the ledger, participant by participant (§8.4 sign convention) ---------------
    ledger = rec["ledger"]
    if sorted(int(a) for a in ledger) != agents:
        e(f"{tag}ledger participants {sorted(ledger)} != the agents on trade rows {agents}")
    for a in agents:
        want = [sum(q for sg, _, q in legs[a] if sg < 0), sum(q for sg, _, q in legs[a] if sg > 0),
                cash_leg(legs[a])]
        got = ledger.get(str(a))
        if not (isinstance(got, list) and len(got) == 3 and all(_is_int(x) for x in got)) \
                or got != want:
            e(f"{tag}agent {a}: ledger {got!r} != [bought, sold, D_a = sells - buys] {want} (§8.4)")

    # --- per-order filled ---------------------------------------------------------------
    filled: Dict[int, int] = defaultdict(int)
    for t in trades:
        filled[t[0]] += t[5]
        filled[t[1]] += t[5]
    got_f = {int(k): int(v) for k, v in rec["filled"].items() if int(v) != 0}
    if got_f != dict(filled):
        e(f"{tag}filled {got_f} != the rows' per-order quantities {dict(filled)}")

    # --- per-submission reports (I6(b), integer half) --------------------------------
    rows = rows_by_submission(trades)
    for k, v in subs.items():
        s = int(k)
        r = rows.get(s, [])
        want = [bool(r), sum(q for _, q in r), sum(trade_value(p, q) for p, q in r)]
        if not (isinstance(v, list) and len(v) == 3 and isinstance(v[0], bool)
                and _is_int(v[1]) and _is_int(v[2])) or v != want:
            e(f"{tag}submission {s}: report {v!r} != [traded, Q_s, D_s] {want} from its rows (I6(b))")
    return errs


def judge_tolerance(tol: Dict[str, Any], rec: Dict[str, Any], tag: str = "") -> List[str]:
    """Euro figures within the derived tolerance (I6(a), I6(b), per participant)."""
    errs: List[str] = []
    e = errs.append
    if not isinstance(tol, dict) or set(tol) != {"agents", "submissions"}:
        return [f"{tag}tolerance evidence not in full: {type(tol).__name__}"]
    trades = rec["trades"]
    legs = legs_by_agent(trades)
    rows = rows_by_submission(trades)
    agents = tol["agents"]
    if sorted(int(a) for a in agents) != sorted(legs):
        e(f"{tag}settlements for agents {sorted(agents)} != the agents on trade rows {sorted(legs)} "
          f"(I6(a): a participant leg dropped or invented)")
    width = None
    for a, vals in agents.items():
        if int(a) not in legs:
            continue
        if not (isinstance(vals, list) and vals and all(isinstance(x, (int, float)) and
                                                       not isinstance(x, bool) and math.isfinite(x)
                                                       for x in vals)):
            e(f"{tag}agent {a}: settlement evidence {vals!r} malformed")
            continue
        width = len(vals) if width is None else width
        if len(vals) != width:
            e(f"{tag}agent {a}: {len(vals)} settlement figures, others have {width}")
        D = cash_leg(legs[int(a)])
        tau = tau_agent(legs[int(a)])
        for x in vals:
            if abs(x - K * D) > tau:
                e(f"{tag}agent {a}: settlement {x!r} EUR != k D_a = {K * D!r} within tau_a = {tau:.4g} "
                  f"(err {abs(x - K * D):.4g}; §8.1 x d, §8.4 sign)")
    sub_tau = 0.0
    traded = sorted(s for s in rows)
    sm = tol["submissions"]
    if sorted(int(s) for s in sm) != traded:
        e(f"{tag}euro reports for submissions {sorted(sm)} != the trading submissions {traded}")
    for s, r in rows.items():
        sub_tau += tau_sub(r)
        x = sm.get(str(s))
        if x is None:
            continue
        D = sum(trade_value(p, q) for p, q in r)
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) \
                or abs(x - K * D) > tau_sub(r):
            e(f"{tag}submission {s}: reported settlement {x!r} != k D_s = {K * D!r} within "
              f"tau_s = {tau_sub(r):.4g} (I6(b))")
    # I6(a): the cash leg sums to zero within the summed tolerance, for each figure set.
    if width and not errs:
        for j in range(width):
            tot = sum(v[j] for v in agents.values())
            if abs(tot) > sub_tau:
                e(f"{tag}I6(a): settlements (figure {j}) sum to {tot!r} EUR, |.| > sum tau_s = "
                  f"{sub_tau:.4g}")
    return errs


def judge_expected_tolerance(exp: Dict[str, Any], rec: Dict[str, Any], tag: str = "") -> List[str]:
    """The K2 expectation's euro targets and tau_s against the rules (not looser than V10)."""
    errs = []
    legs = legs_by_agent(rec["trades"])
    rows = rows_by_submission(rec["trades"])
    if not isinstance(exp, dict) or set(exp) != {"agents", "submissions"}:
        return [f"{tag}K2 tolerance expectation not in full"]
    if sorted(int(a) for a in exp["agents"]) != sorted(legs):
        errs.append(f"{tag}K2 expected agents {sorted(exp['agents'])} != trade-row agents")
    for a, (target, _tau_nb) in exp["agents"].items():
        if int(a) in legs:
            D = cash_leg(legs[int(a)])
            if abs(target - K * D) > tau_agent(legs[int(a)]):
                errs.append(f"{tag}K2 expected agent {a} target {target} != k D_a = {K * D}")
    if sorted(int(s) for s in exp["submissions"]) != sorted(rows):
        errs.append(f"{tag}K2 expected submissions != trading submissions")
    for s, (target, tau_nb) in exp["submissions"].items():
        r = rows.get(int(s))
        if r is None:
            continue
        D = sum(trade_value(p, q) for p, q in r)
        if abs(target - K * D) > tau_sub(r):
            errs.append(f"{tag}K2 expected submission {s} target {target} != k D_s = {K * D}")
        if tau_nb > tau_sub(r) * (1 + 1e-9):
            errs.append(f"{tag}K2 tau_s {tau_nb} for submission {s} looser than V10's chunk-agnostic "
                        f"bound {tau_sub(r)}")
    return errs


def judge_label(rec: Dict[str, Any], m: re.Match, tag: str = "") -> List[str]:
    errs = []
    labels = m.group(2).split(", ")
    regimes = [x[len("regime: "):] for x in labels if x.startswith("regime: ")]
    if len(regimes) != 1 or regimes[0] not in REGIMES:
        return [f"{tag}label {labels} does not name exactly one regime of {REGIMES}"]
    ps = [t[4] - O for t in rec["trades"]]
    qs = [t[5] for t in rec["trades"]]
    reg = regimes[0]
    if reg == "zero" and 0 not in ps:
        errs.append(f"{tag}regime zero without a trade at exactly 0.00 EUR/MWh")
    if reg == "negative" and not any(p < 0 for p in ps):
        errs.append(f"{tag}regime negative without a negative-price trade")
    if reg == "extreme_low" and not any(p <= -500_000 for p in ps):
        errs.append(f"{tag}regime extreme_low without a trade at <= -5 000 EUR/MWh")
    if reg == "extreme_high" and not any(p >= 500_000 for p in ps):
        errs.append(f"{tag}regime extreme_high without a trade at >= +5 000 EUR/MWh")
    if reg == "large_qty" and not any(q >= 1_000 for q in qs):
        errs.append(f"{tag}regime large_qty without a trade of >= 100 MW")
    if SELF_PAIR in labels:
        legs = legs_by_agent(rec["trades"])
        ok = any(cash_leg(legs[t[2]]) == 0 for t in rec["trades"] if t[2] == t[3])
        if not ok:
            errs.append(f"{tag}label claims a zero-cash self-match participant; none in the rows")
    unknown = [x for x in labels if not x.startswith("regime: ") and x != SELF_PAIR]
    if unknown:
        errs.append(f"{tag}unknown label classes {unknown}")
    return errs


def judge_draws(payload, *, seeds_only: Optional[set] = None) -> List[str]:
    errs: List[str] = []
    idx = seed_index(payload)
    if seeds_only is None:
        valid = payload["draws"]["valid"]
        n_recs = sum(1 for v in idx.values() if v.get("recs"))
        if n_recs != valid:
            errs.append(f"per-seed records {n_recs} != valid draws {valid}")
    for seed, ent in sorted(idx.items(), key=lambda kv: (kv[0] is None, kv[0])):
        if seeds_only is not None and seed not in seeds_only:
            continue
        tag = f"seed {seed}: "
        recs, tols = ent.get("recs", []), ent.get("tol", [])
        if len(recs) != 1 or len(tols) != 1:
            errs.append(f"{tag}expected one record and one tolerance assertion, got "
                        f"{len(recs)} and {len(tols)}")
            continue
        _, a, m = recs[0]
        if int(m.group(1)) != seed:
            errs.append(f"{tag}label names another seed")
        try:
            obs, exp = _load(a["observed"]), _load(a["expected"])
        except (TypeError, ValueError) as exc:
            errs.append(f"{tag}per-draw evidence not JSON: {exc!r}")
            continue
        bad = judge_record(obs, tag)
        errs.extend(bad)
        errs.extend(judge_record(exp, tag + "K2 expected: "))
        if not bad:
            errs.extend(judge_label(obs, m, tag))
            if exp != obs:
                errs.append(f"{tag}observed record differs from the K2 expectation "
                            f"(a row not pinned by the walk) yet status is {a['status']!r}"
                            if a["status"] == "pass" else f"{tag}observed != expected")
        _, ta, _ = tols[0]
        tbad: List[str] = []
        if not bad:
            tbad = judge_tolerance(_load(ta["observed"]), obs, tag)
            errs.extend(tbad)
            errs.extend(judge_expected_tolerance(_load(ta["expected"]), obs, tag))
            for _, tea, _ in ent.get("tol_eager", []):
                eb = judge_tolerance(_load(tea["observed"]), obs, tag + "(eager) ")
                errs.extend(eb)
                if eb and tea["status"] != "fail":
                    errs.append(f"{tag}(eager) recorded status {tea['status']!r} on a "
                                f"rules-violating settlement")
        if bad and a["status"] != "fail":
            errs.append(f"{tag}recorded status {a['status']!r} on a rules-violating record")
        if not bad and a["status"] != "pass":
            errs.append(f"{tag}recorded status {a['status']!r} on a rules-conforming record")
        if tbad and ta["status"] != "fail":
            errs.append(f"{tag}recorded status {ta['status']!r} on a rules-violating settlement")
        if not bad and not tbad and ta["status"] != "pass":
            errs.append(f"{tag}recorded status {ta['status']!r} on a rules-conforming settlement")
    return errs


def judge_eager_jit(payload) -> List[str]:
    errs = []
    if payload.get("eager_jit_identical") is not True:
        errs.append(f"eager_jit_identical = {payload.get('eager_jit_identical')!r}")
    idx = seed_index(payload)
    ej = [(s, x) for s, v in idx.items() for x in v.get("ej", [])]
    if not ej:
        return errs + ["no eager-vs-jitted evidence"]
    cover = defaultdict(bool)
    for seed, (_, a, m) in ej:
        if a["kind"] != "eager_jit" or int(m.group(1)) != seed:
            errs.append(f"eager-jit assertion {a['name']!r} mislabelled")
        if a["status"] != "pass" or a["observed"] is not True or a["expected"] is not True:
            errs.append(f"eager != jitted on seed {seed}")
        ent = idx[seed]
        if not ent.get("recs") or not ent.get("tol_eager"):
            errs.append(f"eager-jit seed {seed} lacks its record or its eager settlement evidence")
            continue
        rec = _load(ent["recs"][0][1]["observed"])
        tea = ent["tol_eager"][0][1]
        if tea["status"] != "pass":
            errs.append(f"seed {seed}: eager settlement assertion status {tea['status']!r}")
        T = rec["trades"]
        led = legs_by_agent(T)
        cover["negative price"] |= any(t[4] < O for t in T)
        cover["zero price"] |= any(t[4] == O for t in T)
        cover["|D_a| >= 2^31"] |= any(abs(cash_leg(v)) >= TWO31 for v in led.values())
        cover["q p_int >= 2^31"] |= any(t[4] * t[5] >= TWO31 for t in T)
        cover["self-match row"] |= any(t[2] == t[3] for t in T)
    for k in ("negative price", "zero price", "|D_a| >= 2^31", "q p_int >= 2^31", "self-match row"):
        if not cover[k]:
            errs.append(f"eager-jit evidence covers no draw with a {k}")
    return errs


def judge_battery(payload) -> List[str]:
    errs = []
    rows = payload["invariants"]
    ids = [r["check_id"] for r in rows]
    if ids != ["RB"] + [f"I{i}" for i in range(1, 10)]:
        return [f"battery incomplete or out of order: {ids}"]
    n = payload["draws"]["valid"]
    for r in rows:
        cid = r["check_id"]
        if r["n_pass"] + r["n_fail"] + r["n_not_applicable"] != n:
            errs.append(f"{cid}: did not run on every valid draw")
        if r["n_fail"] != 0 or r["failing_seeds"] or r["status"] != "pass":
            errs.append(f"{cid}: status {r['status']}, {r['n_fail']} failing draws {r['failing_seeds']}")
    by = {r["check_id"]: r for r in rows}
    # every draw trades, so I6 and I9 must have judged every draw (no not_applicable)
    for cid in ("I6", "I9", "I1", "I5"):
        if by[cid]["n_pass"] != n:
            errs.append(f"{cid}: passed on {by[cid]['n_pass']} of {n} draws (every draw trades)")
    if by["I6"]["unobserved"] != ["independent_valuation"]:
        errs.append(f"I6 unobserved {by['I6']['unobserved']} != ['independent_valuation'] (V10)")
    return errs


def judge_status(payload) -> List[str]:
    errs = list(validate_results(payload))
    if payload["failure_kinds"] != derive_failure_kinds(payload):
        errs.append("failure_kinds not derived from payload")
    any_fail = any(a["status"] == "fail" for a in payload["assertions"])
    if any_fail != (payload["status"] != "PASS"):
        errs.append(f"status {payload['status']!r} vs any failing assertion = {any_fail}")
    return errs


def judge_all(payload) -> List[str]:
    return judge_status(payload) + judge_draws(payload) + judge_eager_jit(payload) + \
        judge_battery(payload)


def records(payload):
    idx = seed_index(payload)
    return [(s, _load(v["recs"][0][1]["observed"])) for s, v in sorted(idx.items()) if v.get("recs")]


# ---------------------------------------------------------------------------
# Tests on the emitted payload
# ---------------------------------------------------------------------------
class TestPayloadShape:
    def test_schema_and_semantic_validation(self, payload):
        assert validate_results(payload) == []

    def test_identity_fields(self, payload):
        assert payload["cert_id"] == "C12"
        assert payload["notebook"] == "C12_settlement_cash_leg.ipynb"
        assert payload["oracle_kind"].startswith("exact-parametric")
        assert payload["environment"]["backend"] == "cpu"

    def test_status_and_failure_kinds_consistent(self, payload):
        assert judge_status(payload) == []

    def test_draw_counts_and_seeds(self, payload):
        d, s = payload["draws"], payload["seeds"]
        assert d["requested"] == DRAWS and d["valid"] == DRAWS
        assert s["count"] == d["valid"] + d["discarded"]
        recs = records(payload)
        assert len(recs) == d["valid"]
        seeds = [x for x, _ in recs]
        assert len(set(seeds)) == len(seeds)
        assert s["first"] <= seeds[0] and seeds[-1] <= s["last"]
        assert sorted(s["failing"]) == sorted(
            a["seed"] for a in payload["assertions"] if a["seed"] is not None and a["status"] == "fail")

    def test_spec_constants_match_the_engine_config(self, payload):
        """The rules' constants (§2.3, §8.1) are the configuration's (k = 2.5e-4, O = 999 900)."""
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert (pc.price_tick, pc.qty_tick, pc.delivery_duration_hours) == (PRICE_TICK, QTY_TICK, D_HOURS)
        assert (pc.price_min, pc.price_max) == (PRICE_MIN_EUR, PRICE_MAX_EUR)
        assert round(-pc.price_min / pc.price_tick) == O
        assert round(pc.price_max / pc.price_tick) + O == P_INT_MAX
        assert round(pc.qty_max / pc.qty_tick) == Q_INT_MAX
        assert K == pytest.approx(2.5e-4, rel=1e-15)
        f3 = [a for a in payload["assertions"] if a["name"].startswith("OMIEProductConfig()")]
        assert len(f3) == 1 and f3[0]["status"] == "pass"
        assert f3[0]["observed"] == [0.01, 0.1, -9999.0, 0.25]

    def test_v10_pinned_tolerance_example(self, payload):
        """V10: one row p = 1 004 598, q = 20 -> tau = 6.043878e-4 EUR (7 s.f.)."""
        assert f"{tau_sub([(1_004_598, 20)]):.6e}" == "6.043878e-04"
        from tests.omiecid.reference.omie_invariants import i6_tolerance
        assert i6_tolerance([(1_004_598, 20)])[0] == pytest.approx(tau_sub([(1_004_598, 20)]), rel=1e-12)
        pin = [a for a in payload["assertions"] if a["name"].startswith("tau_s reproduces V10")]
        assert len(pin) == 1 and pin[0]["status"] == "pass" and pin[0]["observed"] == "6.043878e-04"


class TestCashLegAgainstRules:
    def test_every_draw_conforms(self, payload):
        """Per-row exact values, ledger, participants, reports, and the euro tolerances."""
        errs = judge_draws(payload)
        assert errs == [], "\n".join(errs[:30])

    def test_every_trade_value_recomputed_exactly_in_ticks(self, payload):
        n_rows = 0
        for seed, rec in records(payload):
            for t, v in zip(rec["trades"], rec["trade_value_ticks"]):
                assert _is_int(v) and v == t[5] * (t[4] - O), (seed, t, v)
                n_rows += 1
        assert n_rows >= DRAWS * 2

    def test_participant_settlements_match_independent_recomputation(self, payload):
        """Each settlement == k * (sells - buys) within tau_a: zero / sign-flip / d-wrong all fail."""
        worst = 0.0
        n = 0
        for seed, ent in seed_index(payload).items():
            if not ent.get("recs"):
                continue
            rec = _load(ent["recs"][0][1]["observed"])
            legs = legs_by_agent(rec["trades"])
            for _, ta, _ in ent["tol"] + ent.get("tol_eager", []):
                for a, vals in _load(ta["observed"])["agents"].items():
                    D = cash_leg(legs[int(a)])
                    tau = tau_agent(legs[int(a)])
                    for x in vals:
                        n += 1
                        if tau > 0:
                            worst = max(worst, abs(x - K * D) / tau)
                        else:
                            assert x == 0.0, (seed, a, x)
                        assert abs(x - K * D) <= tau, (seed, a, x, K * D, tau)
                        if abs(K * D) > 3 * tau:   # the wrong readings are distinguishable here
                            assert abs(-x - K * D) > tau and abs(4 * x - K * D) > tau and x != 0.0
        assert n >= DRAWS and worst <= 1.0

    def test_cash_leg_sums_to_zero_within_derived_tolerance(self, payload):
        """I6(a): |sum_a settlement_a| <= sum_s tau_s, over exactly the trade-row agents."""
        for seed, ent in seed_index(payload).items():
            if not ent.get("recs"):
                continue
            rec = _load(ent["recs"][0][1]["observed"])
            bound = sum(tau_sub(r) for r in rows_by_submission(rec["trades"]).values())
            for _, ta, _ in ent["tol"] + ent.get("tol_eager", []):
                ag = _load(ta["observed"])["agents"]
                assert sorted(int(a) for a in ag) == sorted(legs_by_agent(rec["trades"]))
                for j in range(len(next(iter(ag.values())))):
                    assert abs(sum(v[j] for v in ag.values())) <= bound, seed

    def test_gross_cash_leg_positive_on_every_draw(self, payload):
        for seed, rec in records(payload):
            assert rec["gross_value_ticks"] == sum(abs(t[5] * (t[4] - O)) for t in rec["trades"]) > 0, seed

    def test_integers_are_full_width_not_int32(self, payload):
        """OQ-18: values beyond 2^31 are present and recorded unwrapped."""
        big_rows = big_agents = 0
        for _, rec in records(payload):
            for v in rec["trade_value_ticks"]:
                big_rows += abs(v) >= TWO31
            for led in rec["ledger"].values():
                big_agents += abs(led[2]) >= TWO31
                assert -2 ** 63 < led[2] < 2 ** 63
        assert big_rows >= 1 and big_agents >= 1

    def test_regimes_cover_negative_zero_and_extreme_prices(self, payload):
        """Re-derived from the rows, not from the notebook's counts."""
        c = defaultdict(int)
        for _, rec in records(payload):
            T = rec["trades"]
            legs = legs_by_agent(T)
            c["negative"] += any(t[4] < O for t in T)
            c["zero"] += any(t[4] == O for t in T)
            c["<= -5000"] += any(t[4] - O <= -500_000 for t in T)
            c[">= +5000"] += any(t[4] - O >= 500_000 for t in T)
            c[">= 100 MW"] += any(t[5] >= 1_000 for t in T)
            c["self row"] += any(t[2] == t[3] for t in T)
            c["zero-cash participant"] += any(cash_leg(v) == 0 for v in legs.values())
            c["bought and sold"] += any(any(s > 0 for s, _, _ in v) and any(s < 0 for s, _, _ in v)
                                        for v in legs.values())
            c["q p_int >= 2^31"] += any(t[4] * t[5] >= TWO31 for t in T)
            c["|D_i| >= 2^31"] += any(abs(t[5] * (t[4] - O)) >= TWO31 for t in T)
            c["|D_a| >= 2^31"] += any(abs(cash_leg(v)) >= TWO31 for v in legs.values())
        for k in ("negative", "zero", "<= -5000", ">= +5000", ">= 100 MW", "self row",
                  "zero-cash participant", "bought and sold", "q p_int >= 2^31", "|D_i| >= 2^31",
                  "|D_a| >= 2^31"):
            assert c[k] >= 1, k

    def test_notebook_coverage_counts_agree_with_the_rows(self, payload):
        cov = [a for a in payload["assertions"] if a["name"].startswith("dataset: the valid draws cover")]
        assert len(cov) == 1 and cov[0]["status"] == "pass"
        obs = cov[0]["observed"]
        recs = records(payload)

        def count(pred):
            return sum(1 for _, r in recs if pred(r))

        def wrap_changes(r, reading):
            T = r["trades"]
            D, Dw = defaultdict(int), defaultdict(int)
            for _bo, _so, ba, sa, p, q, _s in T:
                v = q * (p - O)
                w = {"row": _wrap32(q * p) - q * O, "two": _wrap32(q * p) - _wrap32(q * O)}.get(reading, v)
                D[sa] += v; D[ba] -= v; Dw[sa] += w; Dw[ba] -= w
            if reading == "acc":
                return any(_wrap32(D[a]) != D[a] for a in D)
            return any(Dw[a] != D[a] for a in D)

        want = {
            "a trade at a negative price (V < 0)": count(lambda r: any(t[4] < O for t in r["trades"])),
            "a trade at exactly 0.00 EUR/MWh (V = 0)": count(lambda r: any(t[4] == O for t in r["trades"])),
            "a self-match trade row (buyer agent == seller agent)":
                count(lambda r: any(t[2] == t[3] for t in r["trades"])),
            "a participant who both bought and sold":
                count(lambda r: any(v[0] > 0 and v[1] > 0 for v in r["ledger"].values())),
            "a submission writing >= 2 trade rows":
                count(lambda r: any(len(x) >= 2 for x in rows_by_submission(r["trades"]).values())),
            "OQ-18: a trade row with q * p_int >= 2^31":
                count(lambda r: any(t[4] * t[5] >= TWO31 for t in r["trades"])),
            "OQ-18: a trade row with |q * (p_int - O)| >= 2^31":
                count(lambda r: any(abs(t[5] * (t[4] - O)) >= TWO31 for t in r["trades"])),
            "OQ-18: a participant with |D_a| >= 2^31":
                count(lambda r: any(abs(v[2]) >= TWO31 for v in r["ledger"].values())),
            "discriminated: int32 wrap of each row's q * p_int": count(lambda r: wrap_changes(r, "row")),
            "discriminated: int32 wrap of each row's two products q * p_int and q * O":
                count(lambda r: wrap_changes(r, "two")),
            "discriminated: int32 wrap of the participant's accumulated D_a":
                count(lambda r: wrap_changes(r, "acc")),
        }
        for k, v in want.items():
            assert obs.get(k) == v and v >= 1, (k, obs.get(k), v)
        # regime counts partition the valid draws and agree with the labels
        assert sum(obs[f"regime: {g}"] for g in REGIMES) == payload["draws"]["valid"]
        idx = seed_index(payload)
        for g in REGIMES:
            assert obs[f"regime: {g}"] == sum(
                1 for v in idx.values() if v.get("recs") and f"regime: {g}" in v["recs"][0][1]["name"])
        n_pair = sum(1 for v in idx.values() if v.get("recs") and SELF_PAIR in v["recs"][0][1]["name"])
        assert obs[SELF_PAIR] == n_pair >= 1

    def test_reports_of_non_trading_submissions_are_zero(self, payload):
        """I6(b): traded == (m_s > 0); a non-trading accepted submission reports Q = 0, D = 0."""
        n_quiet = 0
        for _, rec in records(payload):
            rows = rows_by_submission(rec["trades"])
            for k, v in rec["submissions"].items():
                if int(k) not in rows:
                    n_quiet += 1
                    assert v == [False, 0, 0]
        assert n_quiet >= 1


class TestEvidence:
    def test_eager_vs_jitted_present_and_identical(self, payload):
        assert judge_eager_jit(payload) == []

    def test_eager_and_jitted_euro_reports_each_within_tolerance(self, payload):
        """Float reports may differ between modes (FMA); each must sit within tau_s."""
        idx = seed_index(payload)
        n = 0
        for seed, ent in idx.items():
            if not ent.get("tol_eager"):
                continue
            rec = _load(ent["recs"][0][1]["observed"])
            rows = rows_by_submission(rec["trades"])
            j = _load(ent["tol"][0][1]["observed"])["submissions"]
            e = _load(ent["tol_eager"][0][1]["observed"])["submissions"]
            assert set(j) == set(e) == {str(s) for s in rows}
            for s, r in rows.items():
                D = sum(trade_value(p, q) for p, q in r)
                assert abs(j[str(s)] - K * D) <= tau_sub(r) and abs(e[str(s)] - K * D) <= tau_sub(r)
                n += 1
        assert n >= 1

    def test_full_battery_on_every_valid_draw(self, payload):
        assert judge_battery(payload) == []

    def test_shape_guard_and_capacity(self, payload):
        f8 = [a for a in payload["assertions"] if a["name"].startswith("settle passed an (M, 6)")]
        assert len(f8) == 1 and f8[0]["status"] == "pass"
        cap = payload["capacity"]
        assert cap["envelope"] == ENVELOPE
        mx = max(len(r["trades"]) for _, r in records(payload))
        assert cap["peaks"]["trades"] >= mx and mx <= ENVELOPE["trades"]
        hb = [a for a in payload["assertions"] if a["name"].startswith("hop bound <= 160")]
        assert len(hb) == 1 and hb[0]["status"] == "pass" and max(hb[0]["observed"]) <= ENVELOPE["hops"]
        fl = [a for a in payload["assertions"] if a["name"].startswith("no capacity flag")]
        assert len(fl) == 1 and fl[0]["status"] == "pass"

    def test_quotes_present_ok_and_load_bearing_passages_byte_exact(self, payload):
        qs = payload["spec"]["quotes"]
        assert qs and all(q["ok"] for q in qs)
        digests = {q["digest"] for q in qs}
        need = [
            ("§8.1", "V (EUR) = Q (MW) × p (EUR/MWh) × d (h)"),
            ("§8.1", "`d`: delivery duration of the contract in hours — **for 15-min contracts: d = 0.25**"),
            ("§8.1", "`p`: trade price in EUR/MWh (resting order's limit price in regular matching; "
                     "arithmetic mean of last pair in batch matching)"),
            ("§8.4", "**Buy agent PnL per trade:** `−V` (cash outflow) + market value of acquired energy "
                     "(exogenous to the simulator; relevant for imbalance resolution)."),
            ("§8.4", "**Sell agent PnL per trade:** `+V` (cash inflow) − cost of delivered energy "
                     "(exogenous)."),
            ("§8.4", "**Net realized PnL:** `sum(sell_trades: V) − sum(buy_trades: V)`, plus "
                     "mark-to-market of open positions if applicable."),
            ("§2.3", "**Trade value formula:** `V (EUR) = Q (MW) × p (EUR/MWh) × d (h)` where d = 0.25 h "
                     "for 15-min contracts. Source: Continuous Trading Matching Algorithm §5.3."),
            ("§2.3", "**Price tick (minimum increment):** 0.01 EUR/MWh (2 decimal places maximum)."),
            ("§2.3", "**Quantity tick (minimum increment):** 0.1 MW (1 decimal place maximum)."),
            ("§2.3", "**Price bounds:** −9,999 EUR/MWh (minimum) to +9,999 EUR/MWh (maximum)."),
        ]
        for sec, text in need:
            q = SpecQuote(source="market-spec.md", section=sec, text=text)
            assert check_quote(q).ok, (sec, text)      # still in the source today
            assert quote_digest(q) in digests, (sec, text)   # and quoted by the notebook
        secs = {(q["source"], q["section"]) for q in qs}
        for want in [("market-spec.md", "§6.6"), ("market-spec.md", "§11.1"),
                     ("open-questions.md", "OQ-18"), ("scenario-ladder-review.md", "V3-03"),
                     ("scenario-ladder-review.md", "V3-04")]:
            assert want in secs, want
        assert payload["citations"] and all(c["checked"] is False for c in payload["citations"])
        assert any("V (EUR) = Q (MW) * p (EUR/MWh) * d (h)" in c["text"] for c in payload["citations"])

    def test_assertion_units_never_euros(self, payload):
        for a in payload["assertions"]:
            assert a["unit"] in ("bool", "count", "other", "ticks", "qty_ticks", "price_ticks"), a["name"]

    def test_whole_payload_judgement_is_clean(self, payload):
        errs = judge_all(payload)
        assert errs == [], "\n".join(errs[:30])


# ---------------------------------------------------------------------------
# Deliberately broken copies (each must be rejected)
# ---------------------------------------------------------------------------
def _pick(payload, pred):
    for seed, ent in sorted(seed_index(payload).items()):
        if ent.get("recs") and pred(_load(ent["recs"][0][1]["observed"])):
            return seed, ent
    raise AssertionError("no draw satisfies the predicate")


def _big_agent(rec):
    """An agent whose |k D_a| dwarfs its tolerance (so wrong readings are visible)."""
    legs = legs_by_agent(rec["trades"])
    for a, v in sorted(legs.items()):
        if abs(K * cash_leg(v)) > 10 * tau_agent(v) and abs(K * cash_leg(v)) > 1.0:
            return a
    return None


def _mut_tol(p, ent, fn, key="tol"):
    i = ent[key][0][0]
    obs = _load(p["assertions"][i]["observed"])
    fn(obs)
    p["assertions"][i]["observed"] = json.dumps(obs)


def _mut_rec(p, ent, fn, both=False):
    i = ent["recs"][0][0]
    obs = _load(p["assertions"][i]["observed"])
    fn(obs)
    p["assertions"][i]["observed"] = json.dumps(obs)
    if both:
        p["assertions"][i]["expected"] = json.dumps(obs)


def _errs_for(p, seed):
    return judge_draws(p, seeds_only={seed})


class TestBrokenCopiesRejected:
    def test_sign_flipped_settlement_is_rejected(self, payload, tmp_path):
        """Buyer receives / seller pays: every participant's euro settlement negated."""
        p = copy.deepcopy(payload)
        seed, ent = _pick(p, lambda r: _big_agent(r) is not None)

        def flip(o):
            o["agents"] = {a: [-x for x in v] for a, v in o["agents"].items()}
        _mut_tol(p, ent, flip)
        p = _reload(p, tmp_path, "C12_sign_flip.json")
        errs = _errs_for(p, seed)
        assert any("settlement" in e and "tau_a" in e for e in errs), errs
        assert any("recorded status 'pass' on a rules-violating settlement" in e for e in errs), errs
        assert judge_all(p) != []

    def test_sign_flipped_ledger_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        seed, ent = _pick(p, lambda r: _big_agent(r) is not None)

        def flip(o):
            o["ledger"] = {a: [v[0], v[1], -v[2]] for a, v in o["ledger"].items()}
        _mut_rec(p, ent, flip, both=True)
        p = _reload(p, tmp_path, "C12_ledger_flip.json")
        errs = _errs_for(p, seed)
        assert any("D_a = sells - buys" in e for e in errs), errs

    def test_one_hour_duration_is_rejected(self, payload, tmp_path):
        """d = 1 h instead of 0.25 h: every euro figure x4."""
        p = copy.deepcopy(payload)
        seed, ent = _pick(p, lambda r: _big_agent(r) is not None)

        def times4(o):
            o["agents"] = {a: [4 * x for x in v] for a, v in o["agents"].items()}
            o["submissions"] = {s: 4 * x for s, x in o["submissions"].items()}
        _mut_tol(p, ent, times4)
        p = _reload(p, tmp_path, "C12_d1h.json")
        errs = _errs_for(p, seed)
        assert any("tau_a" in e for e in errs), errs
        assert any("tau_s" in e and "I6(b)" in e for e in errs), errs

    def test_dropped_participant_leg_is_rejected(self, payload, tmp_path):
        """One participant missing from the settlement (I6(a) would sum over the rest)."""
        p = copy.deepcopy(payload)
        seed, ent = _pick(p, lambda r: _big_agent(r) is not None)
        rec = _load(ent["recs"][0][1]["observed"])
        a = str(_big_agent(rec))

        def drop_tol(o):
            del o["agents"][a]
        _mut_tol(p, ent, drop_tol)
        p1 = _reload(p, tmp_path, "C12_drop_tol.json")
        errs = _errs_for(p1, seed)
        assert any("I6(a): a participant leg dropped" in e for e in errs), errs

        p = copy.deepcopy(payload)

        def drop_rec(o):
            o["settled_agents"] = [x for x in o["settled_agents"] if str(x) != a]
            del o["ledger"][a]
        _mut_rec(p, ent, drop_rec, both=True)
        p2 = _reload(p, tmp_path, "C12_drop_rec.json")
        errs = _errs_for(p2, seed)
        assert any("participant leg dropped or invented" in e for e in errs), errs
        assert any("ledger participants" in e for e in errs), errs

    def test_one_dropped_trade_leg_inside_a_settlement_is_rejected(self, payload, tmp_path):
        """A participant's settlement missing one row's +-V (e.g. a mis-shaped tape)."""
        p = copy.deepcopy(payload)
        seed, ent = _pick(p, lambda r: any(abs(K * v) > 1.0 for v in r["trade_value_ticks"]))
        rec = _load(ent["recs"][0][1]["observed"])
        j = next(j for j, v in enumerate(rec["trade_value_ticks"]) if abs(K * v) > 1.0)
        seller = str(rec["trades"][j][3])
        leg = K * rec["trade_value_ticks"][j]

        def drop(o):
            o["agents"][seller] = [x - leg for x in o["agents"][seller]]
        _mut_tol(p, ent, drop)
        p = _reload(p, tmp_path, "C12_drop_leg.json")
        errs = _errs_for(p, seed)
        assert any(f"agent {seller}:" in e for e in errs), errs

    def test_int32_wrapped_trade_value_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        seed, ent = _pick(p, lambda r: any(abs(v) >= TWO31 for v in r["trade_value_ticks"]))

        def wrap(o):
            o["trade_value_ticks"] = [_wrap32(v) for v in o["trade_value_ticks"]]
        _mut_rec(p, ent, wrap, both=True)
        p = _reload(p, tmp_path, "C12_wrap_row.json")
        errs = _errs_for(p, seed)
        assert any("value" in e and "q (p_int - O)" in e for e in errs), errs
        assert any("recorded status 'pass' on a rules-violating record" in e for e in errs), errs

    def test_int32_wrapped_participant_cash_leg_is_rejected(self, payload, tmp_path):
        """Accumulated D_a wrapped, and the euro settlement computed from the wrapped integer."""
        p = copy.deepcopy(payload)
        seed, ent = _pick(p, lambda r: any(abs(v[2]) >= TWO31 for v in r["ledger"].values()))
        rec = _load(ent["recs"][0][1]["observed"])
        a = next(k for k, v in rec["ledger"].items() if abs(v[2]) >= TWO31)

        def wrap_ledger(o):
            o["ledger"][a][2] = _wrap32(o["ledger"][a][2])
        _mut_rec(p, ent, wrap_ledger, both=True)

        def wrap_eur(o):
            o["agents"][a] = [K * _wrap32(rec["ledger"][a][2]) for _ in o["agents"][a]]
        _mut_tol(p, ent, wrap_eur)
        p = _reload(p, tmp_path, "C12_wrap_acc.json")
        errs = _errs_for(p, seed)
        assert any(f"agent {a}: ledger" in e for e in errs), errs
        # the wrapped euro figure alone (record intact) is also rejected
        p = copy.deepcopy(payload)
        _mut_tol(p, ent, wrap_eur)
        p = _reload(p, tmp_path, "C12_wrap_eur.json")
        errs = _errs_for(p, seed)
        assert any(f"agent {a}: settlement" in e for e in errs), errs

    def test_vacuous_zero_settlement_is_rejected(self, payload, tmp_path):
        """The (steps, 2, 6) hazard: every participant settles to 0.0, the zero-sum holds."""
        p = copy.deepcopy(payload)
        seed, ent = _pick(p, lambda r: _big_agent(r) is not None)

        def zero(o):
            o["agents"] = {a: [0.0 for _ in v] for a, v in o["agents"].items()}
        _mut_tol(p, ent, zero)
        p = _reload(p, tmp_path, "C12_zero.json")
        assert any("tau_a" in e for e in _errs_for(p, seed))

    def test_undecoded_price_settlement_is_rejected(self, payload, tmp_path):
        """k * sum(+-q p_int) without the offset O."""
        p = copy.deepcopy(payload)
        seed, ent = _pick(p, lambda r: _big_agent(r) is not None)
        rec = _load(ent["recs"][0][1]["observed"])
        legs = legs_by_agent(rec["trades"])

        def undecoded(o):
            o["agents"] = {a: [K * sum(sg * p_ * q for sg, p_, q in legs[int(a)]) for _ in v]
                           for a, v in o["agents"].items()}
        _mut_tol(p, ent, undecoded)
        p = _reload(p, tmp_path, "C12_undecoded.json")
        assert any("tau_a" in e for e in _errs_for(p, seed))

    def test_report_quantity_off_by_one_tick_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        seed, ent = _pick(p, lambda r: any(v[0] for v in r["submissions"].values()))
        rec = _load(ent["recs"][0][1]["observed"])
        s = next(k for k, v in rec["submissions"].items() if v[0])

        def bump(o):
            o["submissions"][s][1] += 1
        _mut_rec(p, ent, bump, both=True)
        p = _reload(p, tmp_path, "C12_qty.json")
        assert any(f"submission {s}: report" in e for e in _errs_for(p, seed))

    def test_missing_or_disagreeing_eager_jit_evidence_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        p["assertions"] = [a for a in p["assertions"] if a["kind"] != "eager_jit"]
        p = _reload(p, tmp_path, "C12_no_ej.json")
        assert "no eager-vs-jitted evidence" in judge_eager_jit(p)

        p = copy.deepcopy(payload)
        i = next(i for i, a in enumerate(p["assertions"]) if a["kind"] == "eager_jit")
        p["assertions"][i]["observed"] = False
        p = _reload(p, tmp_path, "C12_ej_false.json")
        assert any(e.startswith("eager != jitted") for e in judge_eager_jit(p))

    def test_status_pass_on_a_failing_battery_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        row = next(r for r in p["invariants"] if r["check_id"] == "I6")
        row["n_pass"] -= 1
        row["n_fail"] += 1
        p = _reload(p, tmp_path, "C12_i6_fail.json")
        assert judge_battery(p) != [] and judge_status(p) != []


# ---------------------------------------------------------------------------
# Slow probe: an independent construction, incl. the exact price bounds
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_own_construction_settles_at_bounds_negative_zero_and_self_match():
    """slow: jit compile of build_record (~10 s) + eager run of a 12-order tape.

    Own book (TapeBuilder): trades at exactly -9 999.00 and +9 999.00 EUR/MWh with
    999 MW (|D_i| ~ 1e10 >> 2^31), at 0.00, at -123.45, an iceberg walk, two legal
    self-matches (§6.6). Expected trades derived from §6 (resting price, price-time).
    """
    import numpy as np
    from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
    from omiecid_simulator.omiecid.settlement import compute_agent_pnl
    from tests.omiecid.reference.omie_invariants import build_record, compare_records, run_battery

    tb = TapeBuilder()
    NEG = O - 12_345                                   # -123.45 EUR/MWh
    tb.add(side=Side.SELL, price_int=0, qty_int=9990, agent_id=1)                          # s0
    tb.add(side=Side.BUY, price_int=P_INT_MAX, qty_int=9990, agent_id=2,
           exec_restriction=ExecRestriction.IOC)                                           # s1
    tb.add(side=Side.SELL, price_int=O, qty_int=50, agent_id=3)                            # s2
    tb.add(side=Side.SELL, price_int=P_INT_MAX, qty_int=9990, agent_id=2)                  # s3
    tb.add(side=Side.BUY, price_int=P_INT_MAX, qty_int=9990, agent_id=3)                   # s4
    tb.add(side=Side.BUY, price_int=P_INT_MAX, qty_int=50, agent_id=1)                     # s5
    tb.add(side=Side.SELL, price_int=NEG, qty_int=777, agent_id=4)                         # s6
    tb.add(side=Side.BUY, price_int=O, qty_int=300, agent_id=4)                            # s7
    tb.add(side=Side.BUY, price_int=O, qty_int=477, agent_id=5,
           exec_restriction=ExecRestriction.IOC)                                           # s8
    tb.add(side=Side.SELL, price_int=O + 100, qty_int=900, peak_int=100, agent_id=6,
           exec_restriction=ExecRestriction.ICE)                                           # s9
    tb.add(side=Side.BUY, price_int=O + 200, qty_int=950, agent_id=7)                      # s10
    tb.add(side=Side.SELL, price_int=O + 200, qty_int=50, agent_id=6)                      # s11
    tape = tb.build()

    # (submission, price_int, buyer_agent, seller_agent) -> qty, from §6 (resting price)
    expected = {
        (1, 0, 2, 1): 9990,
        (4, O, 3, 3): 50, (4, P_INT_MAX, 3, 2): 9940,
        (5, P_INT_MAX, 1, 2): 50,
        (7, NEG, 4, 4): 300,
        (8, NEG, 5, 4): 477,
        (10, O + 100, 7, 6): 900,
        (11, O + 200, 7, 6): 50,
    }

    rec_j = build_record(tape, jit=True)
    rec_e = build_record(tape, jit=False)
    assert compare_records(rec_j, rec_e) == ()                    # eager == jitted (integers)
    assert not rec_j.rejections and len(rec_j.accepted) == 12

    for rec in (rec_j, rec_e):
        got = defaultdict(int)
        for f in rec.fills:
            got[(f.submission, f.price_int, f.buyer_agent, f.seller_agent)] += f.qty_int
        assert dict(got) == expected

        trades = [[f.buyer_oid, f.seller_oid, f.buyer_agent, f.seller_agent, f.price_int, f.qty_int,
                   f.submission] for f in rec.fills]
        vals = [trade_value(t[4], t[5]) for t in trades]
        assert max(abs(v) for v in vals) >= TWO31                      # OQ-18 regime reached
        assert vals[0] == 9990 * (0 - O) == -9_989_001_000              # -9999.00 x 999 MW
        legs = legs_by_agent(trades)
        assert any(abs(cash_leg(v)) >= TWO31 for v in legs.values())

        # Battery, incl. I6 with the exact V10 chunk count
        rep = run_battery(rec, expects_trade=True)
        assert all(st == "pass" for st in rep.statuses().values()), rep.statuses()

        # I6(a)/per participant: the engine's own settlements and an independent call on (M, 6)
        T = np.array([[f.sec, f.price_int, f.qty_int, f.buyer_agent, f.seller_agent, 0]
                      for f in rec.fills], dtype=np.int32)
        assert T.shape == (len(rec.fills), 6)
        assert set(rec.engine_settlements) == set(legs)
        bound = 0.0
        for s in {f.submission for f in rec.fills}:
            fs = [f for f in rec.fills if f.submission == s]
            c_s = len({(f.sec, f.nsec) for f in fs})
            r = [(f.price_int, f.qty_int) for f in fs]
            tau = tau_sub(r, chunks=c_s)
            bound += tau
            er = rec.engine_reports[s]
            D_s = sum(trade_value(p, q) for p, q in r)
            assert er.accepted and er.traded
            assert round(er.trade_qty_mw / QTY_TICK) == sum(q for _, q in r)
            assert abs(er.settlement_value_eur - K * D_s) <= tau, (s, er, K * D_s, tau)
        for er in rec.engine_reports:
            if er.submission not in {f.submission for f in rec.fills}:
                assert not er.traded and er.settlement_value_eur == 0.0

        mine = {}
        for a, lg in legs.items():
            D = cash_leg(lg)
            tau = tau_agent(lg)
            x_eng = float(rec.engine_settlements[a])
            x_own = float(compute_agent_pnl(T, a))
            mine[a] = x_own
            for x in (x_eng, x_own):
                assert abs(x - K * D) <= tau, (a, x, K * D, tau)
                if D != 0:
                    assert abs(-x - K * D) > tau and abs(4 * x - K * D) > tau
        # §8.4 signs at the bounds: agent 1 SOLD 999 MW at -9999.00 -> pays; agent 2 bought -> receives
        assert mine[1] < 0 and cash_leg(legs[1]) == -9_989_001_000 - 50 * (P_INT_MAX - O)
        # agent 3: self-match at 0.00 (zero) + bought 994 MW at +9999.00 -> pays
        assert cash_leg(legs[3]) == -9940 * (P_INT_MAX - O) and mine[3] < 0
        # agent 4: self-match at -123.45 nets to 0; sold 47.7 MW at -123.45 -> pays
        assert cash_leg(legs[4]) == 477 * (NEG - O) and mine[4] < 0
        assert abs(sum(rec.engine_settlements.values())) <= bound            # I6(a)
        assert abs(sum(mine.values())) <= bound
        # agent 9 never traded: the settlement of a non-participant is exactly 0
        assert float(compute_agent_pnl(T, 9)) == 0.0
