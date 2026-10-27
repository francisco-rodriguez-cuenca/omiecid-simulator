"""K4 audit twin for OMIE-CERT-C07-1 — immediate-or-cancel (IOC), carried by relation M3.

Written in fresh context, adversarially, from the specification only:

* market-spec §4.5, IOC row — partial fill **Yes**, book entry if unmatched **No (cancelled)**;
  NON row — partial fill Yes, book entry Yes.
* market-spec §6.1 / §6.2 items 1-3 — a match needs ``sell_price <= buy_price``; the trade prints at
  the **resting** order's limit; matching does not look at the execution restriction.
* market-spec §6.2 items 4-5 — the unfilled residual of an incoming NON rests (original timestamp,
  §5.2); the residual of an incoming IOC is cancelled.
* market-spec §5.1 — ``CANCELLED (IOC/FOK unmatched; GTD expired; GCT reached; halt/hibernation)``.
* scenario-ladder(-plan) §C7 / M3 — the same price and quantity submitted as IOC and as NON against
  the same book trade **identically**; the IOC's own id never rests; the maker legitimately survives.
  Three regimes (no liquidity, partial, full) must all be exercised.

Derivation (exact Python ints, from the two generated tapes' rows only — never from
``Draw.expected`` and never from the notebook). One resting maker ``r`` and one opposite-side
aggressor ``a`` (submission 1):

    crosses = p_buy >= p_sell;   f = min(q_r, q_a) if crosses else 0
    regime  = none (f == 0) | full (f == q_a) | partial (0 < f == q_r < q_a)
    trades  (IOC == NON) = [[o_r, o_a, p_r, f, 1]] if f > 0 else []
    IOC fate of o_a     = [filled f, resting 0, hidden 0, evicted 0, ioc_remainder q_a - f]
    final book IOC      = maker keeps q_r - f at p_r; nothing of o_a
    final book NON      = the IOC book plus o_a's residual q_a - f at p_a (if > 0)

Independence: this file does not open or import ``C07_ioc.ipynb``, ``contract_book.py``,
``iceberg_manager.py``, ``order_validator.py`` or ``settlement.py``.
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
    make_ioc_regimes,
)
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from notebooks.certification.cert_support import (
    SpecQuote,
    check_quote,
    derive_failure_kinds,
    quote_digest,
    validate_results,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO / "notebooks" / "certification" / "results"

CONS = "C07.ioc_regimes"
REGIMES = ("none", "partial", "full")
BATTERY = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")
TRADE_DEPENDENT = ("I1", "I4", "I5")   # vacuous (not applicable) on a draw that prints no trade
BUY, SELL = int(Side.BUY), int(Side.SELL)
IOC, NON = int(ExecRestriction.IOC), int(ExecRestriction.NON)

# 13-column OMIE order row
_SIDE, _PRICE, _QTY, _OID, _AGENT, _TS, _EXEC = 1, 2, 3, 4, 5, 6, 7

# Fill fields as recorded in the (M3) row, in order.
FILL_FIELDS = ["row_index", "submission", "price_int", "qty_int", "qty_signed", "passive_oid",
               "aggressor_oid", "passive_agent", "aggressor_agent", "buyer_oid", "seller_oid",
               "buyer_agent", "seller_agent", "sec", "nsec"]
FI = {k: i for i, k in enumerate(FILL_FIELDS)}

# The three certified passages, byte-copied from market-spec.md (v2.13).
CERTIFIED_QUOTES = (
    ("§4.5", "| IOC | Yes | No (cancelled) | Regular |"),
    ("§5.1", "├─ CANCELLED (IOC/FOK unmatched; GTD expired; GCT reached; halt/hibernation)"),
    ("§6.2 item 5", "If incoming order is partially filled and has IOC restriction: residual is "
                    "cancelled."),
)

R_STATUS = {"none": "CANCELLED (unmatched)",
            "partial": "PARTIALLY_EXECUTED, remainder CANCELLED",
            "full": "EXECUTED"}

_NAME = re.compile(r"^seed (?P<seed>\d+) \((?P<reg>[a-z]+)\): (?P<tag>.*)$")
_TAGS = (("(A) ", "A"), ("(Q) ", "Q"), ("(M3) ", "M3"), ("(R) ", "R"), ("(B) ", "B"),
         ("(M3-B) ", "M3B"), ("(E) ", "E"), ("battery M3", "BM3"),
         ("NON counterfactual battery", "NONB"))
LETTERS = tuple(k for _t, k in _TAGS)
_EJ = re.compile(r"^eager == jitted \(seed (?P<seed>\d+), (?P<tape>IOC|NON) tape\)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _results_dir() -> Path:
    env = os.environ.get("OMIE_CERT_RESULTS_DIR")
    return Path(env) if env else DEFAULT_RESULTS_DIR


def _load(results_dir: Path) -> Dict[str, Any]:
    return json.loads((results_dir / "C07.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    path = _results_dir() / "C07.json"
    if not path.exists():
        pytest.skip(f"C07 results JSON not emitted yet: {path} is absent (K3/K5 has not run)")
    return _load(_results_dir())


# ---------------------------------------------------------------------------
# Independent derivation
# ---------------------------------------------------------------------------
def _rows(tape) -> List[Tuple[int, ...]]:
    return [tuple(int(x) for x in r) for r in np.asarray(tape) if int(r[_OID]) >= 1]


def derive_pair(ioc_tape, non_tape) -> Dict[str, Any]:
    """The spec answer for one (IOC tape, NON counterfactual) pair."""
    ri, rn = _rows(ioc_tape), _rows(non_tape)
    assert len(ri) == 2 and len(rn) == 2, "C07 is one maker plus one aggressor"
    maker, agg = ri
    # M3 precondition: identical orders except the aggressor's restriction.
    assert rn[0] == maker, "the NON counterfactual changes the maker row"
    assert agg[_EXEC] == IOC and rn[1][_EXEC] == NON, "aggressor restriction is not IOC / NON"
    assert rn[1][:_EXEC] + rn[1][_EXEC + 1:] == agg[:_EXEC] + agg[_EXEC + 1:], \
        "the NON aggressor differs from the IOC aggressor in more than the restriction"
    assert maker[_SIDE] != agg[_SIDE], "maker and aggressor on the same side"
    assert maker[_EXEC] == NON, "the maker must be a plain resting order"
    buy, sell = (agg, maker) if agg[_SIDE] == BUY else (maker, agg)
    crosses = buy[_PRICE] >= sell[_PRICE]                      # §6.2 item 2
    q_r, q_a, p_r, p_a = maker[_QTY], agg[_QTY], maker[_PRICE], agg[_PRICE]
    f = min(q_r, q_a) if crosses else 0
    regime = "none" if f == 0 else ("full" if f == q_a else "partial")
    trades = [[maker[_OID], agg[_OID], p_r, f, 1]] if f else []  # §6.2 item 3: maker's limit
    book_ioc: Dict[str, list] = {"BUY": [], "SELL": []}
    side_name = {BUY: "BUY", SELL: "SELL"}
    if q_r - f > 0:
        book_ioc[side_name[maker[_SIDE]]].append([p_r, q_r - f, maker[_OID]])
    book_non = copy.deepcopy(book_ioc)
    if q_a - f > 0:                                            # §6.2 item 4 (NON residual rests)
        book_non[side_name[agg[_SIDE]]].append([p_a, q_a - f, agg[_OID]])
    return {
        "regime": regime, "f": f, "q_r": q_r, "q_a": q_a, "p_r": p_r, "p_a": p_a,
        "maker": maker, "agg": agg, "rest_side": maker[_SIDE],
        "accepted": [[maker[_OID], agg[_OID]], []],
        "Q": trades,
        "fate": [f, 0, 0, 0, q_a - f],                         # §4.5 / §6.2 item 5 / §5.1
        "B": {"IOC": book_ioc, "NON": book_non},
        "E": [[True, False, 0], [True, f > 0, f]],
        "bracket": [p_r - 1, p_r + 1] if f else None,
        "buyer": buy, "seller": sell,
    }


_DERIVED: Dict[int, Dict[str, Any]] = {}


def derived_for(seed: int) -> Dict[str, Any]:
    if seed not in _DERIVED:
        d = make_ioc_regimes(seed)
        out = derive_pair(d.tape, d.aux_tapes["counterfactual"])
        out["generator_regime"] = d.params.get("regime")
        _DERIVED[seed] = out
    return _DERIVED[seed]


# ---------------------------------------------------------------------------
# Indexing the emitted assertions
# ---------------------------------------------------------------------------
def _letter(tag: str):
    for prefix, key in _TAGS:
        if tag.startswith(prefix):
            return key
    return None


def index(p: Mapping[str, Any]) -> Tuple[Dict[Tuple[int, str], dict], Dict[int, str], List[str]]:
    idx: Dict[Tuple[int, str], dict] = {}
    labels: Dict[int, str] = {}
    errs: List[str] = []
    for a in p.get("assertions", []):
        if a.get("kind") != "oracle":
            continue
        m = _NAME.match(a.get("name", ""))
        if not m:
            continue
        seed, key = int(m["seed"]), _letter(m["tag"])
        if key is None:
            errs.append(f"[presence] unrecognised per-draw assertion {a['name']!r}")
            continue
        if (seed, key) in idx:
            errs.append(f"[presence] duplicate per-draw assertion seed {seed} {key}")
        if a.get("seed") != seed:
            errs.append(f"[presence] {a['name']!r}: seed field {a.get('seed')!r} != {seed}")
        if labels.setdefault(seed, m["reg"]) != m["reg"]:
            errs.append(f"[regime] seed {seed}: inconsistent regime labels")
        idx[(seed, key)] = a
    return idx, labels, errs


def seeds_in(p) -> List[int]:
    return sorted(index(p)[1])


def _both(a, want, tag, seed) -> List[str]:
    errs = []
    for side in ("observed", "expected"):
        if a.get(side) != want:
            errs.append(f"[{tag}] seed {seed}: {side} {a.get(side)!r} != derived {want!r}")
    if a.get("status") != "pass":
        errs.append(f"[{tag}] seed {seed}: status {a.get('status')!r}")
    return errs


# ---------------------------------------------------------------------------
# Checks — each returns a list of tagged error strings (empty = clean)
# ---------------------------------------------------------------------------
def check_presence(p) -> List[str]:
    idx, labels, errs = index(p)
    seeds = sorted(labels)
    if len(seeds) < 500:
        errs.append(f"[presence] only {len(seeds)} draws carry per-draw evidence (default is 500)")
    for s in seeds:
        for key in LETTERS:
            if (s, key) not in idx:
                errs.append(f"[presence] seed {s}: ({key}) missing")
    sd, dr = p["seeds"], p["draws"]
    if sd["count"] != len(seeds) or dr["valid"] != len(seeds):
        errs.append(f"[presence] seeds.count={sd['count']} draws.valid={dr['valid']} != "
                    f"{len(seeds)} draws with evidence")
    if seeds and (sd["first"] != seeds[0] or sd["last"] != seeds[-1]):
        errs.append("[presence] seeds.first/last disagree with the per-draw evidence")
    if dr["requested"] != dr["valid"] + dr["discarded"]:
        errs.append("[presence] requested != valid + discarded")
    if dr["requested"] and abs(dr["discard_rate"] - dr["discarded"] / dr["requested"]) > 1e-12:
        errs.append("[presence] discard_rate != discarded / requested")
    return errs


def check_regimes(p) -> List[str]:
    """All three regimes present; every label equals the derived regime (and the generator's)."""
    _idx, labels, errs = index(p)
    for s, lab in labels.items():
        d = derived_for(s)
        if lab != d["regime"]:
            errs.append(f"[regime] seed {s}: labelled {lab!r}, derived {d['regime']!r}")
        if d["generator_regime"] != d["regime"]:
            errs.append(f"[regime] seed {s}: generator says {d['generator_regime']!r}, "
                        f"tape derives {d['regime']!r}")
    for r in REGIMES:
        if not any(derived_for(s)["regime"] == r for s in labels):
            errs.append(f"[regime] regime {r} missing from the emitted draws")
    return errs


def check_accepted(p) -> List[str]:
    idx, labels, errs = index(p)
    for s in labels:
        a = idx.get((s, "A"))
        if a is not None:
            want = {"IOC": derived_for(s)["accepted"], "NON": derived_for(s)["accepted"]}
            errs += _both(a, want, "A", s)    # an unmatched IOC is cancelled, never rejected
    return errs


def check_trades(p) -> List[str]:
    idx, labels, errs = index(p)
    for s in labels:
        a = idx.get((s, "Q"))
        if a is not None:
            errs += _both(a, derived_for(s)["Q"], "Q", s)
    return errs


def check_m3(p) -> List[str]:
    """M3: IOC trade rows == NON trade rows, field for field; each row's spec fields derived."""
    idx, labels, errs = index(p)
    signs: Dict[int, set] = {BUY: set(), SELL: set()}
    for s in labels:
        a = idx.get((s, "M3"))
        if a is None:
            continue
        d = derived_for(s)
        ioc, non = a.get("observed"), a.get("expected")
        if ioc != non:
            errs.append(f"[M3] seed {s}: IOC trade rows {ioc!r} != NON trade rows {non!r}")
        if a.get("status") != "pass":
            errs.append(f"[M3] seed {s}: status {a.get('status')!r}")
        for which, rows in (("IOC", ioc), ("NON", non)):
            if not isinstance(rows, list) or len(rows) != len(d["Q"]):
                errs.append(f"[M3] seed {s}: {which} has {rows!r}, derived {len(d['Q'])} trade(s)")
                continue
            for row in rows:
                if len(row) != len(FILL_FIELDS):
                    errs.append(f"[M3] seed {s}: {which} row is not a full Fill: {row!r}")
                    continue
                m, g = d["maker"], d["agg"]
                want = {"row_index": 0, "submission": 1, "price_int": d["p_r"], "qty_int": d["f"],
                        "passive_oid": m[_OID], "aggressor_oid": g[_OID],
                        "passive_agent": m[_AGENT], "aggressor_agent": g[_AGENT],
                        "buyer_oid": d["buyer"][_OID], "seller_oid": d["seller"][_OID],
                        "buyer_agent": d["buyer"][_AGENT], "seller_agent": d["seller"][_AGENT]}
                for k, v in want.items():
                    if row[FI[k]] != v:
                        errs.append(f"[M3] seed {s}: {which} {k} = {row[FI[k]]!r}, derived {v!r}")
                if abs(row[FI["qty_signed"]]) != d["f"]:
                    errs.append(f"[M3] seed {s}: {which} |qty_signed| != {d['f']}")
                signs[g[_SIDE]].add(int(np.sign(row[FI["qty_signed"]])))
    for side, sg in signs.items():
        if len(sg) > 1:
            errs.append(f"[M3] qty_signed sign is not a function of the aggressor side ({side}: {sg})")
    if signs[BUY] and signs[SELL] and signs[BUY] == signs[SELL]:
        errs.append("[M3] qty_signed carries the same sign for BUY and SELL aggressors")
    return errs


def check_zero_residual(p) -> List[str]:
    """§4.5 / §6.2 item 5 / §5.1: the IOC's own remainder is cancelled, never rests, never hides."""
    idx, labels, errs = index(p)
    for s in labels:
        a = idx.get((s, "R"))
        if a is None:
            continue
        d = derived_for(s)
        want = {"fate": d["fate"], "other_causes": [], "post_snapshots_holding": []}
        errs += _both(a, want, "R", s)
        if not a["name"].endswith(f"({R_STATUS[d['regime']]})"):
            errs.append(f"[R] seed {s}: status label in {a['name']!r} is not "
                        f"{R_STATUS[d['regime']]!r}")
        fate = a.get("observed", {}).get("fate") or [None] * 5
        if fate[1] != 0 or fate[2] != 0:
            errs.append(f"[R] seed {s}: IOC remainder rests (resting {fate[1]}, hidden {fate[2]})")
        if a.get("observed", {}).get("post_snapshots_holding"):
            errs.append(f"[R] seed {s}: a post-submission book holds the IOC id")
    return errs


def check_books(p) -> List[str]:
    idx, labels, errs = index(p)
    maker_ts = set()
    for s in labels:
        d = derived_for(s)
        a = idx.get((s, "B"))
        if a is not None:
            errs += _both(a, d["B"], "B", s)
            ioc_book = (a.get("observed") or {}).get("IOC") or {}
            if any(r[2] == d["agg"][_OID] for side in ioc_book.values() for r in side):
                errs.append(f"[B] seed {s}: the IOC id {d['agg'][_OID]} rests in the final book")
        b = idx.get((s, "M3B"))
        if b is None:
            continue
        if b.get("observed") != b.get("expected"):
            errs.append(f"[M3B] seed {s}: IOC book {b.get('observed')!r} != NON book less the "
                        f"aggressor {b.get('expected')!r}")
        if b.get("status") != "pass":
            errs.append(f"[M3B] seed {s}: status {b.get('status')!r}")
        for which in ("observed", "expected"):
            got = b.get(which) or {}
            short = {k: [r[:3] for r in v] for k, v in got.items()}
            if short != d["B"]["IOC"]:
                errs.append(f"[M3B] seed {s}: {which} {short!r} != derived {d['B']['IOC']!r}")
            for v in got.values():
                for r in v:
                    if r[2] == d["maker"][_OID]:
                        if r[3] != d["maker"][_AGENT]:
                            errs.append(f"[M3B] seed {s}: maker agent {r[3]} != {d['maker'][_AGENT]}")
                        maker_ts.add((r[4], r[5]))
    if len(maker_ts) > 1:   # §5.2: a partially executed resting order keeps its timestamp
        errs.append(f"[M3B] the surviving maker's (sec, nsec) varies across draws: {maker_ts}")
    return errs


def check_engine_report(p) -> List[str]:
    """ENGINEERING INVARIANT: per-submission [accepted, traded, qty_int]; price near the maker."""
    idx, labels, errs = index(p)
    for s in labels:
        a = idx.get((s, "E"))
        if a is None:
            continue
        d = derived_for(s)
        obs, exp = a.get("observed") or {}, a.get("expected") or {}
        for run in ("IOC", "NON"):
            if obs.get(run) != d["E"] or exp.get(run) != d["E"]:
                errs.append(f"[E] seed {s}: {run} {obs.get(run)!r} / {exp.get(run)!r} != {d['E']!r}")
        if exp.get("agg_price_int_in") != d["bracket"]:
            errs.append(f"[E] seed {s}: bracket {exp.get('agg_price_int_in')!r} != {d['bracket']!r}")
        if d["f"]:
            prices = obs.get("agg_price_int") or {}
            lo, hi = d["bracket"]
            for run in ("IOC", "NON"):
                pr = prices.get(run)
                if not (isinstance(pr, list) and len(pr) == 1 and lo <= pr[0] <= hi):
                    errs.append(f"[E] seed {s}: {run} reported price {pr!r} outside [{lo}, {hi}]")
            if prices.get("IOC") != prices.get("NON"):
                errs.append(f"[E] seed {s}: IOC and NON report different prices {prices!r}")
        if a.get("status") != "pass":
            errs.append(f"[E] seed {s}: status {a.get('status')!r}")
    return errs


def check_per_draw_battery(p) -> List[str]:
    idx, labels, errs = index(p)
    for s in labels:
        a = idx.get((s, "BM3"))
        if a is not None and not (a.get("observed") == a.get("expected") == "pass"
                                  and a.get("status") == "pass"):
            errs.append(f"[BM3] seed {s}: battery M3 {a.get('observed')!r} ({a.get('status')!r})")
        b = idx.get((s, "NONB"))
        if b is not None and not (b.get("observed") == {} and b.get("expected") == {}
                                  and b.get("status") == "pass"):
            errs.append(f"[NONB] seed {s}: NON counterfactual battery failures {b.get('observed')!r}")
    return errs


def check_eager_jit(p) -> List[str]:
    errs: List[str] = []
    if p.get("eager_jit_identical") is not True:
        errs.append(f"[eager_jit] eager_jit_identical = {p.get('eager_jit_identical')!r}")
    ej = [a for a in p.get("assertions", []) if a.get("kind") == "eager_jit"]
    if not ej:
        errs.append("[eager_jit] no eager-vs-jitted evidence in the results (absence = failure)")
        return errs
    tapes: Dict[int, set] = {}
    for a in ej:
        if a.get("status") != "pass" or a.get("observed") is not True or a.get("expected") is not True:
            errs.append(f"[eager_jit] {a.get('name')!r}: {a.get('status')!r} {a.get('observed')!r}")
        m = _EJ.match(a.get("name", ""))
        if not m or a.get("seed") != int(m["seed"]):
            errs.append(f"[eager_jit] unrecognised evidence {a.get('name')!r}")
            continue
        tapes.setdefault(int(m["seed"]), set()).add(m["tape"])
    for s, t in tapes.items():
        if t != {"IOC", "NON"}:
            errs.append(f"[eager_jit] seed {s}: only the {sorted(t)} tape(s) ran eager and jitted")
    covered = {derived_for(s)["regime"] for s in tapes}
    for r in REGIMES:
        if r not in covered:
            errs.append(f"[eager_jit] no {r}-regime draw was run both eager and jitted")
    return errs


def check_battery(p) -> List[str]:
    errs: List[str] = []
    rows = {r["check_id"]: r for r in p.get("invariants", [])}
    seeds = seeds_in(p)
    n = p["draws"]["valid"]
    n_none = sum(derived_for(s)["regime"] == "none" for s in seeds)
    for cid in BATTERY:
        r = rows.get(cid)
        if r is None:
            errs.append(f"[battery] {cid} missing")
            continue
        if r["n_pass"] + r["n_fail"] + r["n_not_applicable"] != n:
            errs.append(f"[battery] {cid}: counts do not cover all {n} draws")
        if r["status"] != "pass" or r["n_fail"] or r["failing_seeds"]:
            errs.append(f"[battery] {cid}: {r['status']} fail={r['n_fail']}")
        want_na = n_none if cid in TRADE_DEPENDENT else 0
        if r["n_not_applicable"] != want_na:
            errs.append(f"[battery] {cid}: {r['n_not_applicable']} not-applicable draws, derived "
                        f"{want_na} (no-trade draws only)")
    return errs


def check_quotes(p) -> List[str]:
    errs: List[str] = []
    quotes = p["spec"]["quotes"]
    have = {(q["source"], q["section"], q["digest"]) for q in quotes}
    for section, text in CERTIFIED_QUOTES:
        sq = SpecQuote("market-spec.md", section, text)
        if ("market-spec.md", section, quote_digest(sq)) not in have:
            errs.append(f"[quotes] certified passage {section} {text!r} is not quoted verbatim")
        chk = check_quote(sq)
        if not chk.ok:
            errs.append(f"[quotes] {section} no longer matches market-spec.md: {chk.reason}")
    for q in quotes:
        if q.get("ok") is not True:
            errs.append(f"[quotes] {q['source']} {q['section']}: {q.get('reason')!r}")
    return errs


def check_capacity_and_coverage(p) -> List[str]:
    errs: List[str] = []
    env = p["capacity"]["envelope"]
    want_env = {"resting_per_side": DEFAULT_ENVELOPE.max_resting_per_side,
                "trades": DEFAULT_ENVELOPE.max_trades, "hops": DEFAULT_ENVELOPE.max_hops,
                "concurrent_icebergs": DEFAULT_ENVELOPE.max_concurrent_icebergs}
    if env != want_env or env["resting_per_side"] > 80 or env["trades"] > 80 or env["hops"] > 160:
        errs.append(f"[capacity] envelope {env!r} != I8 {want_env!r}")
    seeds = seeds_in(p)
    peaks = p["capacity"]["peaks"]
    # one maker on one side, at most one NON residual on the other: never more than 1 per side
    if peaks["resting_per_side"] != 1:
        errs.append(f"[capacity] resting peak {peaks['resting_per_side']} != 1")
    want_trades = max((len(derived_for(s)["Q"]) for s in seeds), default=0)
    if peaks["trades"] != want_trades:
        errs.append(f"[capacity] trade peak {peaks['trades']} != {want_trades}")
    if peaks["concurrent_icebergs"] != 0:
        errs.append(f"[capacity] {peaks['concurrent_icebergs']} icebergs in a no-iceberg scenario")
    cov = next((a for a in p["assertions"] if a["name"].startswith("dataset: the valid draws cover")),
               None)
    want = {f"{r}, resting {sn}": sum(1 for s in seeds if derived_for(s)["regime"] == r
                                      and derived_for(s)["rest_side"] == sv)
            for r in REGIMES for sn, sv in (("BUY", BUY), ("SELL", SELL))}
    if cov is None:
        errs.append("[coverage] regime x side coverage assertion missing")
    elif cov.get("observed") != want or cov.get("status") != "pass":
        errs.append(f"[coverage] {cov.get('observed')!r} != derived {want!r}")
    for k, v in want.items():
        if v < 1:
            errs.append(f"[coverage] cell {k!r} is empty")
    return errs


def check_status(p) -> List[str]:
    errs = validate_results(p)
    if p.get("failure_kinds") != derive_failure_kinds(p):
        errs.append("failure_kinds not derived from content")
    return [f"[schema] {e}" for e in errs]


ALL_CHECKS = (check_status, check_presence, check_regimes, check_accepted, check_trades, check_m3,
              check_zero_residual, check_books, check_engine_report, check_per_draw_battery,
              check_eager_jit, check_battery, check_quotes, check_capacity_and_coverage)


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
        assert payload["cert_id"] == "C07"
        assert payload["oracle_kind"] == "exact-parametric"
        assert check_status(payload) == []

    def test_generator_digest_is_the_current_generator(self, payload):
        digests = payload["generator"]["digests"]
        assert set(digests) == {CONS}
        assert construction_digest(CONS) == digests[CONS]

    def test_every_draw_carries_full_per_draw_evidence(self, payload):
        assert check_presence(payload) == []

    def test_all_three_regimes_present_and_correctly_labelled(self, payload):
        assert check_regimes(payload) == []

    def test_every_order_accepted_in_both_runs(self, payload):
        assert check_accepted(payload) == []

    def test_counterparty_sequence_exact(self, payload):
        assert check_trades(payload) == []

    def test_m3_ioc_trade_rows_equal_non_trade_rows(self, payload):
        assert check_m3(payload) == []

    def test_ioc_remainder_cancelled_never_rests(self, payload):
        assert check_zero_residual(payload) == []

    def test_final_books_ioc_and_non(self, payload):
        assert check_books(payload) == []

    def test_engine_reports(self, payload):
        assert check_engine_report(payload) == []

    def test_per_draw_m3_and_non_counterfactual_battery(self, payload):
        assert check_per_draw_battery(payload) == []

    def test_eager_vs_jitted_evidence_present_and_identical(self, payload):
        assert check_eager_jit(payload) == []

    def test_full_battery_ran_and_holds(self, payload):
        assert check_battery(payload) == []

    def test_certified_quotes_present_and_byte_exact(self, payload):
        assert check_quotes(payload) == []

    def test_capacity_and_coverage(self, payload):
        assert check_capacity_and_coverage(payload) == []

    def test_status_pass_iff_audit_clean(self, payload):
        errs = audit_errors(payload)
        assert (payload["status"] == "PASS") == (errs == []), errs[:10]


# ---------------------------------------------------------------------------
# The derivation itself on hand-checked numbers
# ---------------------------------------------------------------------------
def _row(side, price, qty, oid, agent, ts, exec_):
    return [0, side, price, qty, oid, agent, ts, exec_, 0, 0, 0, 0, 0]


def _pair(maker, agg_side, agg_price, agg_qty):
    m = _row(*maker)
    ioc = np.asarray([m, _row(agg_side, agg_price, agg_qty, 2, 2, 2, IOC)], dtype=np.int32)
    non = np.asarray([m, _row(agg_side, agg_price, agg_qty, 2, 2, 2, NON)], dtype=np.int32)
    return ioc, non


class TestDerivationSanity:
    def test_partial_buy_aggressor(self):
        d = derive_pair(*_pair((SELL, 100, 5, 1, 1, 1, NON), BUY, 101, 8))
        assert d["regime"] == "partial" and d["Q"] == [[1, 2, 100, 5, 1]]
        assert d["fate"] == [5, 0, 0, 0, 3]
        assert d["B"] == {"IOC": {"BUY": [], "SELL": []},
                          "NON": {"BUY": [[101, 3, 2]], "SELL": []}}

    def test_full_sell_aggressor(self):
        d = derive_pair(*_pair((BUY, 200, 10, 1, 1, 1, NON), SELL, 199, 4))
        assert d["regime"] == "full" and d["Q"] == [[1, 2, 200, 4, 1]]
        assert d["fate"] == [4, 0, 0, 0, 0]
        assert d["B"]["IOC"] == d["B"]["NON"] == {"BUY": [[200, 6, 1]], "SELL": []}

    def test_none_unmatched_ioc_is_cancelled_whole(self):
        d = derive_pair(*_pair((SELL, 100, 5, 1, 1, 1, NON), BUY, 99, 7))
        assert d["regime"] == "none" and d["Q"] == [] and d["fate"] == [0, 0, 0, 0, 7]
        assert d["B"]["NON"]["BUY"] == [[99, 7, 2]] and d["B"]["IOC"]["BUY"] == []

    def test_equal_quantities_is_full_not_partial(self):
        d = derive_pair(*_pair((SELL, 100, 5, 1, 1, 1, NON), BUY, 100, 5))
        assert d["regime"] == "full" and d["fate"] == [5, 0, 0, 0, 0]

    def test_counterfactual_differing_beyond_restriction_is_refused(self):
        ioc, non = _pair((SELL, 100, 5, 1, 1, 1, NON), BUY, 101, 8)
        non[1, _QTY] = 9
        with pytest.raises(AssertionError):
            derive_pair(ioc, non)


# ---------------------------------------------------------------------------
# Deliberately broken copies must be rejected
# ---------------------------------------------------------------------------
def _write_and_reload(p, tmp_path: Path) -> Dict[str, Any]:
    (tmp_path / "C07.json").write_text(json.dumps(p), encoding="utf-8")
    return _load(tmp_path)


def _pick(p, regime: str) -> int:
    for s in seeds_in(p):
        if derived_for(s)["regime"] == regime:
            return s
    pytest.skip(f"no {regime}-regime draw in the emitted results")


def _edit(p, seed: int, key: str, fn) -> None:
    for a in p["assertions"]:
        m = _NAME.match(a["name"])
        if m and a["kind"] == "oracle" and int(m["seed"]) == seed and _letter(m["tag"]) == key:
            fn(a)


def _has(errs, tag, seed):
    return any(e.startswith(tag) and f"seed {seed}:" in e for e in errs)


class TestBrokenCopiesRejected:
    def test_ioc_remainder_that_rests_is_rejected(self, payload, tmp_path):
        s = _pick(payload, "partial")
        d = derived_for(s)
        left = d["q_a"] - d["f"]
        side = "BUY" if d["agg"][_SIDE] == BUY else "SELL"
        broken = copy.deepcopy(payload)

        def rests(a):
            a["observed"]["fate"] = [d["f"], left, 0, 0, 0]
            a["observed"]["post_snapshots_holding"] = [1]
            a["expected"] = copy.deepcopy(a["observed"])

        def book(a):
            for k in ("observed", "expected"):
                a[k]["IOC"][side].append([d["p_a"], left, d["agg"][_OID]])

        _edit(broken, s, "R", rests)
        _edit(broken, s, "B", book)
        errs = audit_errors(_write_and_reload(broken, tmp_path))
        assert _has(errs, "[R]", s) and _has(errs, "[B]", s)
        assert any("IOC remainder rests" in e for e in errs)

    def test_ioc_trade_list_differing_from_non_is_rejected(self, payload, tmp_path):
        s = _pick(payload, "partial")
        broken = copy.deepcopy(payload)

        def shrink(a):
            a["observed"] = [r[:3] + [r[3] - 1] + r[4:] for r in a["observed"]]

        def shrink_q(a):
            a["observed"] = [r[:3] + [r[3] - 1] + r[4:] for r in a["observed"]]

        _edit(broken, s, "M3", shrink)
        _edit(broken, s, "Q", shrink_q)
        errs = audit_errors(_write_and_reload(broken, tmp_path))
        assert any(e.startswith("[M3]") and f"seed {s}: IOC trade rows" in e for e in errs)
        assert _has(errs, "[Q]", s)

    def test_consistently_wrong_trade_lists_are_rejected(self, payload, tmp_path):
        """IOC == NON but both print at the aggressor's limit: M3 holds, the derivation does not."""
        s = _pick(payload, "full")
        p_a = derived_for(s)["p_a"]
        broken = copy.deepcopy(payload)

        def reprice(a):
            for k in ("observed", "expected"):
                a[k] = [r[:2] + [p_a] + r[3:] for r in a[k]]

        _edit(broken, s, "M3", reprice)
        errs = audit_errors(_write_and_reload(broken, tmp_path))
        assert any(e.startswith("[M3]") and f"seed {s}:" in e and "price_int" in e for e in errs)

    def test_missing_regime_is_rejected(self, payload, tmp_path):
        broken = copy.deepcopy(payload)
        broken["assertions"] = [a for a in broken["assertions"]
                                if not (a["kind"] == "oracle" and a["name"].startswith("seed ")
                                        and " (partial): " in a["name"])]
        errs = audit_errors(_write_and_reload(broken, tmp_path))
        assert "[regime] regime partial missing from the emitted draws" in errs
        assert any(e.startswith("[coverage]") for e in errs)

    def test_mislabelled_regime_is_rejected(self, payload, tmp_path):
        s = _pick(payload, "none")
        broken = copy.deepcopy(payload)
        for a in broken["assertions"]:
            if a.get("seed") == s and a["kind"] == "oracle":
                a["name"] = a["name"].replace(f"seed {s} (none):", f"seed {s} (partial):")
        errs = check_regimes(_write_and_reload(broken, tmp_path))
        assert any(f"seed {s}: labelled 'partial'" in e for e in errs)

    def test_missing_eager_jit_evidence_is_rejected(self, payload, tmp_path):
        broken = copy.deepcopy(payload)
        broken["assertions"] = [a for a in broken["assertions"] if a["kind"] != "eager_jit"]
        errs = check_eager_jit(_write_and_reload(broken, tmp_path))
        assert any("absence = failure" in e for e in errs)

    def test_eager_jit_on_ioc_tape_only_is_rejected(self, payload, tmp_path):
        broken = copy.deepcopy(payload)
        broken["assertions"] = [a for a in broken["assertions"]
                                if not (a["kind"] == "eager_jit" and "NON tape" in a["name"])]
        errs = check_eager_jit(_write_and_reload(broken, tmp_path))
        assert any("only the ['IOC'] tape(s)" in e for e in errs)

    def test_missing_battery_row_is_rejected(self, payload, tmp_path):
        broken = copy.deepcopy(payload)
        broken["invariants"] = [r for r in broken["invariants"] if r["check_id"] != "I2"]
        errs = check_battery(_write_and_reload(broken, tmp_path))
        assert "[battery] I2 missing" in errs

    def test_failing_per_draw_m3_battery_is_rejected(self, payload, tmp_path):
        s = _pick(payload, "full")
        broken = copy.deepcopy(payload)

        def fail(a):
            a["observed"] = "fail"
        _edit(broken, s, "BM3", fail)
        errs = check_per_draw_battery(_write_and_reload(broken, tmp_path))
        assert _has(errs, "[BM3]", s)

    def test_status_pass_with_broken_content_is_inconsistent(self, payload, tmp_path):
        s = _pick(payload, "none")
        broken = copy.deepcopy(payload)

        def resting(a):
            a["observed"]["fate"] = [0, derived_for(s)["q_a"], 0, 0, 0]
        _edit(broken, s, "R", resting)
        broken = _write_and_reload(broken, tmp_path)
        errs = audit_errors(broken)
        assert errs and broken["status"] == "PASS"   # a PASS here would be a false certificate


# ---------------------------------------------------------------------------
# Optional independent re-run on the engine (slow)
# ---------------------------------------------------------------------------
def _seed_of(regime: str) -> int:
    return next(s for s in range(64) if derived_for(s)["regime"] == regime)


@pytest.mark.slow
@pytest.mark.parametrize("regime", REGIMES)
def test_rerun_engine_matches_derivation(regime):
    """slow: jitted + eager engine replay of both tapes via the battery recorder."""
    from tests.omiecid.reference.omie_invariants import (
        build_record, check_m3_ioc_non_equivalence, compare_records, record_from_draw,
    )

    seed = _seed_of(regime)
    draw = make_ioc_regimes(seed)
    d = derive_pair(draw.tape, draw.aux_tapes["counterfactual"])
    ioc = record_from_draw(draw)
    non = build_record(draw.aux_tapes["counterfactual"], capacity_bounds=dict(draw.capacity_bounds))
    for rec in (ioc, non):
        got = [[f.passive_oid, f.aggressor_oid, f.price_int, f.qty_int, f.submission]
               for f in rec.fills]
        assert got == d["Q"]
    agg = d["agg"][_OID]
    assert int(ioc.still_resting.get(agg, 0)) == 0
    assert int(non.still_resting.get(agg, 0)) == d["q_a"] - d["f"]
    assert check_m3_ioc_non_equivalence(ioc, non, oid=agg).status == "pass"
    assert compare_records(ioc, record_from_draw(draw, jit=False)) == ()
