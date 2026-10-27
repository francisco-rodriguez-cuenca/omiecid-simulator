"""K4 audit twin for OMIE-CERT-C10-1, re-run r2 (fill-or-kill, in four parts).

Fresh-context, adversarial audit (C10 is high-risk, independence rule 4). Every
expectation is re-derived here from the rules alone. It never reads or executes
the C10 notebook's sections 1-5 / 7, the generator's ``Draw.expected`` /
``Draw.params``, or the engine modules (independence rules 3/4). Spec stamp:
``market-spec.md`` **v2.14**.

Rules used, all in integer ticks (price tick 0.01 EUR/MWh, qty tick 0.1 MW):

* §6.2 item 2: a maker crosses a FOK of limit ``L`` iff ``sell <= buy``.
* §4.2.1 Rule 1 / OQ-11 (Interpretation B): the FOK pre-check volume ``M`` is the
  sum of every visible maker crossing ``L`` plus, for an iceberg, every slice
  (hidden ones included) whose own price crosses ``L``. Slice ``k`` of an
  iceberg ``(base, ppd, peak, total)`` holds ``min(peak, total - k*peak)`` at
  ``base + k*ppd`` (§4.2; standard icebergs have ``ppd = 0``).
* §4.5 FOK row / §5.1: executed iff ``req <= M``; then exactly ``req`` is filled;
  otherwise killed: zero trades, the whole ``req`` cancelled as ``fok_kill``, the
  book after == the book before (M4).
* Fill order (§6.1, §6.2 items 1/6, §4.2.1 Rule 2 incl. the v2.14 *Re-ranking*
  clarification): the FOK always trades the best-ranked crossing counter entry
  next. Each regenerated slice is a NEW entry at its own drifted price with a new
  (later) timestamp, so a slice that ties a resting NON's price ranks after it,
  and a slice priced worse than the NON is traded only after the NON. One trade
  per counter entry (each slice is its own counter bid), at that entry's price
  (§6.2 items 3 and 7; never capped at ``L``).
* After a slice is consumed exactly while hidden quantity remains, the next
  slice is shown at ``base + (k+1)*ppd`` with ``min(peak, remaining)`` (v2.14
  eager activation) or, as the OQ-20 between-aggressor allowance, absent.

C10d (``C10.fok_ice_worse_level``, RERUN-1): one iceberg (standard or sign-valid
ICEwPI) and a strictly worse NON level on the same side, then a FOK. My own
derivation: ``T_c = min(total, peak * r)`` with ``r = #{k : base + k*ppd crosses
L}``, ``M = T_c + q_v``; runs ``T_c, T_c+1, M, M+1``. C10d draws are notebook-
local (no ``omie_tape_synth`` mode, no digest) and the FOK limit is not emitted,
so the draw's book parameters are read from the record's killed run (the input
book) and ``r`` from the record's request list; the audit then checks that a
limit realising that ``r`` exists (visible crossing, exactly ``r`` slices
crossing) and derives every run from it. The C10d pre-check *projection* itself
is certified independently by the ContractBook probes at the end of this file
(own constructions, own limits).

Section-7 aggregates (RERUN-1): M4 / I3 / I5 fail only on runs with ``status ==
"fail"``; ``not_applicable`` (I5 on a killed, zero-trade FOK) is vacuously
satisfied and must be reported as a count, never counted as a failure.

Results JSON: ``$OMIE_CERT_RESULTS_DIR/C10.json`` else
``notebooks/certification/results/C10.json``; absent -> skip.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import functools
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_tape_synth import construction_digest, make_fok_boundary
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from notebooks.certification.cert_support import (
    SpecQuote,
    check_quote,
    derive_failure_kinds,
    quote_digest,
    validate_results,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "notebooks" / "certification" / "results"
SPEC_VERSION = "v2.14"

BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, FOK, ICE = int(ExecRestriction.NON), int(ExecRestriction.FOK), int(ExecRestriction.ICE)
SIDE_NAME = {BUY: "BUY", SELL: "SELL"}

# Tape columns (13-wide int32 loader layout).
C_SIDE, C_PRICE, C_QTY, C_OID, C_TS, C_EXEC, C_PEAK, C_PPD = 1, 2, 3, 4, 6, 7, 11, 12

SYNTH_PARTS = {
    "C10a": ("visible", "C10.fok_visible"),
    "C10b": ("hidden", "C10.fok_hidden"),
    "C10c": ("icewpi", "C10.fok_icewpi"),
}
ALL_PARTS = ("C10a", "C10b", "C10c", "C10d")
DRAWS_PER_PART = 500          # "Random scenarios default to 500 draws"
SLICE_MARGIN = 64             # ceil(total/peak) <= 64
ENVELOPE = {"resting_per_side": 80, "trades": 80, "hops": 160, "concurrent_icebergs": 800}

SEED_RE = re.compile(r"^seed (\d+) \[(C10[abcd]): (.*?)\], requests \[([0-9, ]*)\]$")
EJ_RE = re.compile(r"^eager == jitted \((C10[abcd]) seed (\d+), every run\)$")
I5_ENTRY_RE = re.compile(r"^(C10[abcd]) (\d+):(\d+)$")
AGG_DETAIL_RE = re.compile(
    r"failing runs: (\d+); not_applicable \(vacuously satisfied\) on (\d+) runs: (\{.*\})")
AGG_PREFIX = {"M4": "M4: no run of any valid draw returns status 'fail'",
              "I3": "I3: no run of any valid draw returns status 'fail'",
              "I5": "I5: no run of any valid draw returns status 'fail'"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _results_path() -> Path:
    d = os.environ.get("OMIE_CERT_RESULTS_DIR")
    return (Path(d) if d else DEFAULT_RESULTS_DIR) / "C10.json"


@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    p = _results_path()
    if not p.is_file():
        pytest.skip(f"C10 results JSON absent at {p} (run notebook section 8 first)")
    return json.loads(p.read_text(encoding="utf-8"))


def _reload(obj: Dict[str, Any], tmp_path: Path, name: str) -> Dict[str, Any]:
    """Write a (broken) copy to tmp_path and read it back, as the auditor would."""
    f = tmp_path / name
    f.write_text(json.dumps(obj), encoding="utf-8")
    return json.loads(f.read_text(encoding="utf-8"))


def _js(x):
    return json.loads(x) if isinstance(x, str) else x


# ---------------------------------------------------------------------------
# The independent derivation
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _draw(part: str, seed: int):
    return make_fok_boundary(int(seed), variant=SYNTH_PARTS[part][0])


def _real_rows(tape: np.ndarray) -> List[np.ndarray]:
    return [r for r in np.asarray(tape) if int(r[C_OID]) >= 1]


def _crosses(fok_side: int, limit: int, maker_price: int) -> bool:
    """§6.2 item 2: sell price <= buy price."""
    return maker_price <= limit if fok_side == BUY else maker_price >= limit


def _better(rest_side: int, a: int, b: int) -> bool:
    """§6.1: price ``a`` ranks strictly before price ``b`` on ``rest_side``."""
    return a > b if rest_side == BUY else a < b


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _slices(row) -> List[Tuple[int, int]]:
    """(price, qty) of every slice of an iceberg row, hidden ones included (§4.2)."""
    total, peak, ppd, base = int(row[C_QTY]), int(row[C_PEAK]), int(row[C_PPD]), int(row[C_PRICE])
    n = _ceil_div(total, peak)
    return [(base + k * ppd, min(peak, total - k * peak)) for k in range(n)]


def _units(row) -> List[Tuple[int, int]]:
    return _slices(row) if int(row[C_EXEC]) == ICE else [(int(row[C_PRICE]), int(row[C_QTY]))]


def _mk_row(side, price, qty, oid, ts, exec_, peak=0, ppd=0) -> Tuple[int, ...]:
    r = [0] * 13
    r[C_SIDE], r[C_PRICE], r[C_QTY], r[C_OID], r[C_TS], r[C_EXEC] = side, price, qty, oid, ts, exec_
    r[C_PEAK], r[C_PPD] = peak, ppd
    return tuple(int(x) for x in r)


def _precheck(makers, fok_side: int, limit: int) -> int:
    """§4.2.1 Rule 1 / OQ-11 B: every crossing visible maker and crossing slice."""
    return sum(q for r in makers for p, q in _units(r) if _crosses(fok_side, limit, p))


@dataclasses.dataclass(frozen=True)
class Scene:
    part: str
    seed: int
    makers: Tuple[Tuple[int, ...], ...]   # rest rows as int tuples (tape layout)
    fok_side: int
    rest_side: int
    limit: int
    drawn_req: int
    fok_oid: int
    fok_sub: int
    M: int                                 # pre-check volume (rules)
    M_card: Optional[int] = None           # C10c card closed form
    reach: Optional[int] = None            # C10c / C10d: #crossing slices r
    T_c: Optional[int] = None              # C10d: iceberg crossing volume
    max_slices: int = 0
    label: str = ""
    problems: Tuple[str, ...] = ()         # C10d: construction inconsistencies


def scene(part: str, seed: int) -> Scene:
    """C10a-c: the scene read row by row from the generator tape."""
    d = _draw(part, seed)
    rows = _real_rows(d.tape)
    fok = rows[-1]
    makers = rows[:-1]
    assert int(fok[C_EXEC]) == FOK, f"{part} seed {seed}: last real row is not FOK"
    fok_side, limit = int(fok[C_SIDE]), int(fok[C_PRICE])
    rest_sides = {int(r[C_SIDE]) for r in makers}
    assert len(rest_sides) == 1 and fok_side not in rest_sides, f"{part} seed {seed}: sides"
    rest_side = rest_sides.pop()
    M = _precheck(makers, fok_side, limit)
    max_slices = max([len(_slices(r)) for r in makers if int(r[C_EXEC]) == ICE] or [0])
    reach = M_card = None
    ices = [r for r in makers if int(r[C_EXEC]) == ICE]
    for r in makers:
        assert int(r[C_EXEC]) in (NON, ICE)
    if part == "C10a":
        assert not ices
        label = f"ladder {SIDE_NAME[rest_side]}, {'1 level' if len(makers) == 1 else '>= 2 levels'}"
    else:
        assert len(makers) == 1 and len(ices) == 1, f"{part} seed {seed}: one resting iceberg"
        ice = ices[0]
        if part == "C10b":
            assert int(ice[C_PPD]) == 0
            req = int(fok[C_QTY])
            label = (f"iceberg {SIDE_NAME[rest_side]}, "
                     f"{'drawn req = total + 1' if req == int(ice[C_QTY]) + 1 else 'drawn req <= total'}")
        else:
            total, peak, ppd, base = (int(ice[C_QTY]), int(ice[C_PEAK]), int(ice[C_PPD]),
                                      int(ice[C_PRICE]))
            # §13.6 sign rule: the C10c draws must be sign-valid ICEwPI.
            assert (ppd < 0) if rest_side == BUY else (ppd > 0), f"seed {seed}: ppd sign"
            # The card's set #{k >= 0 : base + k*ppd crosses L}: monotone in k for a
            # sign-valid ppd, so the count is the first non-crossing k.
            k = 0
            while _crosses(fok_side, limit, base + k * ppd):
                k += 1
                assert k <= 10_000
            reach = k
            M_card = min(total, peak * k)
            label = f"iceberg {SIDE_NAME[rest_side]}, {'r = 1' if k == 1 else 'r >= 2'}"
    return Scene(part=part, seed=int(seed), makers=tuple(tuple(int(x) for x in r) for r in makers),
                 fok_side=fok_side, rest_side=rest_side, limit=limit, drawn_req=int(fok[C_QTY]),
                 fok_oid=int(fok[C_OID]), fok_sub=len(makers), M=M, M_card=M_card, reach=reach,
                 max_slices=max_slices, label=label)


def scene_d(seed: int, a: Dict[str, Any]) -> Scene:
    """C10d: the input book from the record's killed run; ``r`` from the request list.

    The killed run is the one at the largest request (M + 1). Its K2-declared book
    is the book the FOK met (the input); its ``remaining`` gives each maker's full
    quantity. The iceberg is the better-priced maker (the NON is strictly worse).
    ``ppd`` is read from the iceberg's slice prices appearing anywhere in the
    K2 record (sign-checked, arithmetic-progression-checked).
    """
    m = SEED_RE.match(a["name"])
    reqs = [int(x) for x in m.group(4).split(",")]
    exp = _js(a["expected"])
    probs: List[str] = []
    kreq = max(reqs)
    kill = exp[str(kreq)]
    book0 = kill["final_book_one_of"][0]
    sides = [s for s in ("BUY", "SELL") if book0.get(s)]
    assert len(sides) == 1, f"C10d seed {seed}: makers on {sides}"
    rest_side = BUY if sides[0] == "BUY" else SELL
    fok_side = SELL if rest_side == BUY else BUY
    entries = sorted(book0[sides[0]], key=lambda e: (-e[0] if rest_side == BUY else e[0]))
    assert len(entries) == 2, f"C10d seed {seed}: two makers expected, got {entries}"
    (base, vis0, ice_oid), (p_v, q_v, vis_oid) = entries
    fok_oid = [int(o) for o in kill["accepted"] if int(o) not in (ice_oid, vis_oid)]
    assert len(fok_oid) == 1
    fok_oid = fok_oid[0]
    total = int(kill["remaining"][str(ice_oid)])
    if int(kill["remaining"][str(vis_oid)]) != q_v:
        probs.append("visible NON is not shown in full in the input book")
    if not _better(rest_side, base, p_v):
        probs.append(f"the NON level {p_v} is not strictly worse than the iceberg base {base}")
    peak = vis0  # min(peak, total); == total for a one-slice iceberg
    # ppd: iceberg prices seen in the K2 record (trades + books), other than base.
    ice_prices = set()
    for run in exp.values():
        for t in run["trades"]:
            if ice_oid in (t[0], t[1]):
                ice_prices.add(int(t[2]))
        for b in run["final_book_one_of"]:
            for e in b[sides[0]]:
                if e[2] == ice_oid:
                    ice_prices.add(int(e[0]))
    others = sorted((p for p in ice_prices if p != base), key=lambda p: abs(p - base))
    ppd = (others[0] - base) if others else 0
    n = _ceil_div(total, peak)
    if others and ((ppd < 0) if rest_side == SELL else (ppd > 0)):
        probs.append(f"ppd {ppd} violates the §13.6 sign rule for a {SIDE_NAME[rest_side]} iceberg")
    for p in ice_prices:
        if ppd == 0:
            if p != base:
                probs.append(f"iceberg price {p} off the standard base {base}")
        elif (p - base) % ppd or not 0 <= (p - base) // ppd < n:
            probs.append(f"iceberg price {p} is not a slice price base + k*ppd")
    ice_row = _mk_row(rest_side, base, total, ice_oid, 1, ICE, peak, ppd)
    vis_row = _mk_row(rest_side, p_v, q_v, vis_oid, 2, NON)
    slices = _slices(ice_row)
    # r from the requests [T_c, T_c+1, M, M+1].
    if len(reqs) != 4:
        probs.append(f"requests {reqs} are not four runs")
    T_c = reqs[0]
    M = reqs[-1] - 1
    if M - T_c != q_v:
        probs.append(f"M - T_c = {M - T_c} != q_v {q_v}")
    if T_c == total:
        r = n
    elif 0 < T_c < total and T_c % peak == 0:
        r = T_c // peak
    else:
        r = None
        probs.append(f"T_c {T_c} is not min(total, peak*r) for any r >= 1")
    limit = None
    if r is not None:
        # a limit realising exactly r crossing slices AND a crossing visible level
        need = [slices[r - 1][0], p_v]
        if fok_side == BUY:
            lo = max(need)
            hi = slices[r][0] - 1 if r < n else lo
        else:
            lo = min(need)
            hi = slices[r][0] + 1 if r < n else lo
        ok = lo <= hi if fok_side == BUY else lo >= hi
        if not ok:
            probs.append(f"no limit crosses the NON {p_v} and exactly {r} slices {slices[:r + 1]}")
        limit = lo
    if limit is None:
        limit = base
    want = [T_c, T_c + 1, M, M + 1]
    if reqs != want:
        probs.append(f"requests {reqs} != [T_c, T_c+1, M, M+1] = {want}")
    M_rules = _precheck([ice_row, vis_row], fok_side, limit)
    if M_rules != M:
        probs.append(f"pre-check at a realising limit {M_rules} != M {M}")
    if ppd == 0:
        kind = "standard"
    else:
        interleaved = any(not _better(rest_side, p, p_v) for p, _ in slices[:r or 0])
        kind = f"ICEwPI, visible {'interleaved' if interleaved else 'behind'}"
    label = f"iceberg {SIDE_NAME[rest_side]}, {kind}"
    return Scene(part="C10d", seed=int(seed), makers=(ice_row, vis_row), fok_side=fok_side,
                 rest_side=rest_side, limit=limit, drawn_req=M, fok_oid=fok_oid, fok_sub=2, M=M,
                 reach=r, T_c=T_c, max_slices=n, label=label, problems=tuple(probs))


def scene_of(part: str, seed: int, a: Dict[str, Any]) -> Scene:
    return scene_d(seed, a) if part == "C10d" else scene(part, seed)


def requests_for(sc: Scene) -> List[int]:
    if sc.part == "C10a":
        return [sc.M - 1, sc.M, sc.M + 1]
    if sc.part == "C10b":
        total = sc.makers[0][C_QTY]
        return sorted({r for r in (sc.drawn_req, total, total + 1)})
    if sc.part == "C10c":
        return [sc.M, sc.M + 1]
    return [sc.T_c, sc.T_c + 1, sc.M, sc.M + 1]


def _initial_book(sc: Scene) -> Dict[str, List[List[int]]]:
    book = {"BUY": [], "SELL": []}
    for r in sc.makers:
        q = min(r[C_PEAK], r[C_QTY]) if r[C_EXEC] == ICE else r[C_QTY]
        book[SIDE_NAME[r[C_SIDE]]].append([r[C_PRICE], q, r[C_OID]])
    return book


def _canon_book(b: Dict[str, Any]) -> Tuple:
    return tuple((s, tuple(sorted(tuple(int(x) for x in e) for e in b.get(s, [])))) for s in ("BUY", "SELL"))


def walk(makers, fok_side: int, rest_side: int, limit: int, req: int):
    """Rank-ordered fill of ``req`` (§6.1, §6.2, §4.2.1 Rule 2 + v2.14 Re-ranking).

    Returns (trades as (maker_oid, price, qty), live entries, freshly activated entries).
    """
    sgn = -1 if rest_side == BUY else 1
    ent = []
    for r in makers:
        u = _units(r)
        ent.append({"oid": r[C_OID], "units": u, "k": 0, "price": u[0][0], "qty": u[0][1],
                    "ts": r[C_TS], "fresh": False})
    next_ts = 10 ** 9
    left, trades = req, []
    while left > 0:
        live = [e for e in ent if e["qty"] > 0 and _crosses(fok_side, limit, e["price"])]
        if not live:
            break
        e = min(live, key=lambda e: (sgn * e["price"], e["ts"]))
        take = min(left, e["qty"])
        e["qty"] -= take
        e["fresh"] = False
        left -= take
        trades.append((e["oid"], e["price"], take))
        if e["qty"] == 0 and e["k"] + 1 < len(e["units"]):
            e["k"] += 1
            e["price"], e["qty"] = e["units"][e["k"]]
            e["ts"], e["fresh"] = next_ts, True     # new entry, new timestamp (§4.2)
            next_ts += 1
    return trades, left, ent


def derive_run(sc: Scene, req: int) -> Dict[str, Any]:
    """The rules' outcome of one FOK run with request ``req``."""
    oids = sorted([r[C_OID] for r in sc.makers] + [sc.fok_oid])
    qty0 = {r[C_OID]: r[C_QTY] for r in sc.makers}
    filled = {o: 0 for o in oids}
    if req > sc.M:
        rem = dict(qty0)
        rem[sc.fok_oid] = 0
        return {"accepted": oids, "cancelled": {sc.fok_oid: [["fok_kill", req]]}, "evicted": {},
                "filled": filled, "final_book_one_of": [_initial_book(sc)], "rejected": [],
                "remaining": rem, "trades": [], "verdict": "killed", "fok_book_unchanged": True}
    tr, left, ent = walk(sc.makers, sc.fok_side, sc.rest_side, sc.limit, req)
    assert left == 0, (sc.part, sc.seed, req)
    trades = []
    for oid, p, q in tr:
        filled[oid] += q
        buyer, seller = (oid, sc.fok_oid) if sc.rest_side == BUY else (sc.fok_oid, oid)
        trades.append([buyer, seller, p, q, sc.fok_sub])
    filled[sc.fok_oid] = req
    side = SIDE_NAME[sc.rest_side]
    eager = {"BUY": [], "SELL": []}
    lazy = {"BUY": [], "SELL": []}
    for e in ent:
        if e["qty"] > 0:
            eager[side].append([e["price"], e["qty"], e["oid"]])
            if not e["fresh"]:
                lazy[side].append([e["price"], e["qty"], e["oid"]])
    alts = [eager] if _canon_book(eager) == _canon_book(lazy) else [eager, lazy]
    rem = {o: qty0[o] - filled[o] for o in qty0}
    rem[sc.fok_oid] = 0
    return {"accepted": oids, "cancelled": {}, "evicted": {}, "filled": filled,
            "final_book_one_of": alts, "rejected": [], "remaining": rem, "trades": trades,
            "verdict": "executed"}


def _maker(sc: Scene, t) -> int:
    """The resting (maker) oid of a trade row [buyer, seller, price, qty, sub]."""
    return t[0] if sc.rest_side == BUY else t[1]


def _intkeys(m: Dict[str, Any]) -> Dict[int, Any]:
    return {int(k): v for k, v in m.items()}


def judge_run(obs: Dict[str, Any], exp: Dict[str, Any]) -> List[str]:
    """Differences between an emitted run and the rules' derivation (empty = conforms)."""
    errs = []
    need = {"accepted", "cancelled", "evicted", "filled", "final_book", "rejected", "remaining",
            "trades", "verdict"}
    missing = need - set(obs)
    if missing:
        return [f"run evidence incomplete, missing {sorted(missing)}"]
    if obs["verdict"] != exp["verdict"]:
        errs.append(f"verdict {obs['verdict']!r} != {exp['verdict']!r}")
    if sorted(obs["accepted"]) != exp["accepted"]:
        errs.append(f"accepted {obs['accepted']} != {exp['accepted']}")
    if obs["rejected"] != []:
        errs.append(f"rejected {obs['rejected']}")
    if any(v for v in obs["evicted"].values()):
        errs.append(f"evicted {obs['evicted']}")
    cancelled = {k: [list(c) for c in v] for k, v in _intkeys(obs["cancelled"]).items()}
    if cancelled != exp["cancelled"]:
        errs.append(f"cancelled {obs['cancelled']} != {exp['cancelled']}")
    if _intkeys(obs["filled"]) != exp["filled"]:
        errs.append(f"filled {obs['filled']} != {exp['filled']}")
    if _intkeys(obs["remaining"]) != exp["remaining"]:
        errs.append(f"remaining {obs['remaining']} != {exp['remaining']}")
    if [list(t) for t in obs["trades"]] != exp["trades"]:
        errs.append(f"trades {obs['trades']} != {exp['trades']}")
    if _canon_book(obs["final_book"]) not in {_canon_book(b) for b in exp["final_book_one_of"]}:
        errs.append(f"final_book {obs['final_book']} not in {exp['final_book_one_of']}")
    if exp["verdict"] == "killed" and obs.get("fok_book_unchanged") is not True:
        errs.append(f"M4: fok_book_unchanged={obs.get('fok_book_unchanged')!r} on a kill")
    return errs


# ---------------------------------------------------------------------------
# Payload judges (each returns a list of error strings; empty = conforms)
# ---------------------------------------------------------------------------
def seed_assertions(payload) -> List[Tuple[Dict[str, Any], re.Match]]:
    out = []
    for a in payload["assertions"]:
        m = SEED_RE.match(a["name"])
        if m:
            out.append((a, m))
    return out


def all_scenes(payload) -> Dict[Tuple[str, int], Tuple[Scene, Dict[str, Any]]]:
    return {(m.group(2), int(m.group(1))): (scene_of(m.group(2), int(m.group(1)), a), a)
            for a, m in seed_assertions(payload)}


def judge_runs(payload) -> List[str]:
    """Every per-seed record: runs present in full, equal to the derivation, status consistent."""
    errs: List[str] = []
    per_part: Dict[str, set] = {p: set() for p in ALL_PARTS}
    for a, m in seed_assertions(payload):
        seed, part, label = int(m.group(1)), m.group(2), m.group(3)
        reqs = [int(x) for x in m.group(4).split(",")]
        tag = f"{part} seed {seed}"
        if a["seed"] != seed:
            errs.append(f"{tag}: seed field {a['seed']}")
        if seed in per_part[part]:
            errs.append(f"{tag}: duplicate per-seed record")
        per_part[part].add(seed)
        sc = scene_of(part, seed, a)
        errs.extend(f"{tag}: construction: {p}" for p in sc.problems)
        if label != sc.label:
            errs.append(f"{tag}: class {label!r} != {sc.label!r}")
        want = requests_for(sc)
        if reqs != want:
            errs.append(f"{tag}: requests {reqs} != {want}")
        obs = _js(a["observed"])
        if sorted(int(k) for k in obs) != sorted(set(want)):
            errs.append(f"{tag}: observed runs {sorted(obs)} != {want} (evidence not in full)")
        run_errs = []
        for r in sorted(set(want)):
            if str(r) not in obs:
                continue
            for e in judge_run(obs[str(r)], derive_run(sc, r)):
                run_errs.append(f"{tag} run {r}: {e}")
        errs.extend(run_errs)
        conforms = (not run_errs and not sc.problems
                    and sorted(int(k) for k in obs) == sorted(set(want)))
        if (a["status"] == "pass") != conforms:
            errs.append(f"{tag}: recorded status {a['status']!r} but the rules say "
                        f"{'conforms' if conforms else 'deviates'}")
    for part, seeds in per_part.items():
        if len(seeds) < DRAWS_PER_PART:
            errs.append(f"{part}: {len(seeds)} per-seed records < {DRAWS_PER_PART}")
    return errs


def judge_eager_jit(payload) -> List[str]:
    errs = []
    seen: Dict[str, List[int]] = {p: [] for p in ALL_PARTS}
    for a in payload["assertions"]:
        if a["kind"] != "eager_jit":
            continue
        m = EJ_RE.match(a["name"])
        if not m:
            errs.append(f"eager_jit assertion with unparsable name {a['name']!r}")
            continue
        part, seed = m.group(1), int(m.group(2))
        if a["status"] != "pass" or a["observed"] is not True or a["expected"] is not True:
            errs.append(f"{part} seed {seed}: eager != jitted ({a['status']}, {a['detail']!r})")
        seen[part].append(seed)
    seeded = {(m.group(2), int(m.group(1))): a for a, m in seed_assertions(payload)}
    for part, seeds in seen.items():
        if not seeds:
            errs.append(f"{part}: no eager-vs-jitted evidence")
            continue
        for s in seeds:
            if (part, s) not in seeded:
                errs.append(f"{part} seed {s}: eager-jit seed is not a valid draw of the part")
        sides = {scene_of(part, s, seeded[(part, s)]).rest_side for s in seeds if (part, s) in seeded}
        if sides != {BUY, SELL}:
            errs.append(f"{part}: eager-jit evidence covers rest sides {sorted(sides)} only")
    if payload.get("eager_jit_identical") is not True:
        errs.append(f"eager_jit_identical = {payload.get('eager_jit_identical')!r}")
    return errs


def judge_status(payload) -> List[str]:
    errs = list(validate_results(payload))
    if payload["failure_kinds"] != derive_failure_kinds(payload):
        errs.append("failure_kinds not derived from payload")
    any_fail = any(a["status"] == "fail" for a in payload["assertions"])
    if any_fail != (payload["status"] != "PASS"):
        errs.append(f"status {payload['status']!r} vs any failing assertion = {any_fail}")
    if payload["status"] != "PASS":
        fe = payload["failure_explanation"]
        first = next(a for a in payload["assertions"] if a["status"] == "fail") if any_fail else None
        if first is not None and first["name"] not in fe["what_failed"]:
            errs.append("failure_explanation.what_failed does not name the first failing assertion")
    return errs


def killed_runs(payload) -> Dict[str, List[str]]:
    """Per part, every run the RULES say is killed, as 'C10x seed:req'."""
    out: Dict[str, List[str]] = {p: [] for p in ALL_PARTS}
    for (part, seed), (sc, _) in all_scenes(payload).items():
        for r in sorted(set(requests_for(sc))):
            if r > sc.M:
                out[part].append(f"{part} {seed}:{r}")
    return out


def judge_aggregates(payload, killed: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """RERUN-1: M4 / I3 / I5 aggregates count ``status == "fail"`` only and report n/a."""
    errs = []
    killed = killed_runs(payload) if killed is None else killed
    n_killed = sum(len(v) for v in killed.values())
    for check, prefix in AGG_PREFIX.items():
        rows = [a for a in payload["assertions"] if a["name"].startswith(prefix)]
        if len(rows) != 1:
            errs.append(f"{check}: {len(rows)} fail-only aggregates (want exactly 1)")
            continue
        a = rows[0]
        if "not_applicable counts as satisfied" not in a["name"]:
            errs.append(f"{check}: aggregate name does not state n/a = satisfied")
        m = AGG_DETAIL_RE.search(a["detail"] or "")
        if not m:
            errs.append(f"{check}: aggregate detail does not report the n/a counts: {a['detail']!r}")
            continue
        n_fail, n_na, by = int(m.group(1)), int(m.group(2)), json.loads(m.group(3))
        obs = a["observed"] if isinstance(a["observed"], list) else _js(a["observed"])
        if len(obs) != n_fail:
            errs.append(f"{check}: observed lists {len(obs)} runs but detail says {n_fail} failing")
        if (a["status"] == "fail") != (n_fail > 0):
            errs.append(f"{check}: status {a['status']!r} with {n_fail} failing runs")
        if sum(by.values()) != n_na:
            errs.append(f"{check}: n/a breakdown {by} does not sum to {n_na}")
        if check == "I5":
            # I5 has nothing to judge on a killed FOK: n/a exactly there, never a failure.
            want = {f"{p} killed": len(v) for p, v in killed.items() if v}
            if by != want:
                errs.append(f"I5: n/a breakdown {by} != the rules' killed runs {want}")
            if n_na != n_killed:
                errs.append(f"I5: {n_na} n/a runs != {n_killed} killed runs")
            flagged_kills = set(map(str, obs)) & {e for v in killed.values() for e in v}
            if flagged_kills:
                errs.append(f"I5: {len(flagged_kills)} killed (n/a) runs counted as failures, "
                            f"e.g. {sorted(flagged_kills)[:3]}")
    return errs


def classify_failures(payload) -> Tuple[List[str], List[str]]:
    """Split failing assertions into (judging artefacts, engine deviations).

    The one recognised artefact: an I5 aggregate whose entries are exactly runs the
    rules say are killed with zero trades (I5 ``not_applicable``: vacuous truth).
    """
    artefacts, deviations = [], []
    sc_map = all_scenes(payload)
    for a in payload["assertions"]:
        if a["status"] != "fail":
            continue
        if a["name"].startswith("I5"):
            bad = []
            for e in a["observed"]:
                m = I5_ENTRY_RE.match(str(e))
                if not m:
                    bad.append(f"unparsable I5 entry {e!r}")
                    continue
                part, seed, req = m.group(1), int(m.group(2)), int(m.group(3))
                if (part, seed) not in sc_map:
                    bad.append(f"I5 flagged {e}: unknown draw")
                    continue
                sc, rec = sc_map[(part, seed)]
                exp = derive_run(sc, req)
                o = _js(rec["observed"]).get(str(req))
                if exp["verdict"] != "killed" or o is None or o["trades"] != [] \
                        or o["verdict"] != "killed":
                    bad.append(f"I5 flagged {e}: a run with trades (rules verdict "
                               f"{exp['verdict']}) - possible engine deviation")
            if bad:
                deviations.extend(bad)
            else:
                artefacts.append(f"I5 aggregate: {len(a['observed'])} entries, every one a "
                                 f"killed zero-trade run (I5 not applicable, vacuous)")
        else:
            deviations.append(f"unclassified failing assertion {a['name']!r}")
    return artefacts, deviations


# ---------------------------------------------------------------------------
# Tests on the emitted payload
# ---------------------------------------------------------------------------
class TestPayloadShape:
    def test_schema_and_semantic_validation(self, payload):
        assert validate_results(payload) == []

    def test_identity_fields(self, payload):
        assert payload["cert_id"] == "C10"
        assert payload["notebook"] == "C10_fok_and_hidden_volume.ipynb"
        assert payload["oracle_kind"] == "exact-parametric, plus M4"
        assert payload["environment"]["backend"] == "cpu"

    def test_status_and_failure_kinds_consistent_with_payload(self, payload):
        assert judge_status(payload) == []

    def test_draw_counts_and_seeds(self, payload):
        d, s = payload["draws"], payload["seeds"]
        assert d["discarded"] == 0 and d["discard_rate"] == 0.0
        assert d["valid"] == d["requested"] == s["count"] == DRAWS_PER_PART * len(ALL_PARTS)
        recs = seed_assertions(payload)
        assert len(recs) == d["valid"]
        assert Counter(m.group(2) for _, m in recs) == {p: DRAWS_PER_PART for p in ALL_PARTS}
        failing = sorted(a["seed"] for a, _ in recs if a["status"] == "fail")
        assert sorted(s["failing"]) == failing
        seeds = sorted(a["seed"] for a, _ in recs)
        assert s["first"] == seeds[0] and s["last"] == seeds[-1]
        assert len(set(seeds)) == len(seeds), "seed ranges of the parts overlap"

    def test_generator_digests_match_regenerated_constructions(self, payload):
        g = payload["generator"]
        assert g["module"] == "omiecid_simulator.jaxomiecid.omie_tape_synth"
        for _, key in SYNTH_PARTS.values():
            assert g["digests"][key] == construction_digest(key), key

    def test_spec_stamp_is_v2_14(self, payload):
        head = (REPO_ROOT / "docs" / "omiecid" / "market-spec.md").read_text(encoding="utf-8")[:400]
        assert f"**Status:** {SPEC_VERSION}" in head
        ms = [q for q in payload["spec"]["quotes"] if q["source"] == "market-spec.md"]
        assert ms and all(q["version"] == SPEC_VERSION for q in ms), \
            sorted({q["version"] for q in ms})


class TestDerivationSanity:
    """The derivation reproduces the spec's own worked examples and the v2.14 re-ranking."""

    def _sc(self, makers, fok_side, limit, part="C10c"):
        return Scene(part=part, seed=-1, makers=tuple(makers), fok_side=fok_side,
                     rest_side=1 - fok_side, limit=limit, drawn_req=0, fok_oid=99,
                     fok_sub=len(makers), M=_precheck(makers, fok_side, limit))

    def test_icewpi_worked_example_8mw_rejected_6mw_fills_at_drifted_prices(self):
        # §4.2.1 Rule 1: ICEwPI SELL base=50.00, ppd=+0.50, peak=2, total=10; FOK BUY 51.00.
        base, ppd = 1_005_000, 50
        ice = _mk_row(SELL, base, 100, 1, 1, ICE, peak=20, ppd=ppd)
        sc = self._sc([ice], BUY, base + 100)
        assert sc.M == 60
        assert derive_run(sc, 80)["verdict"] == "killed"
        ex = derive_run(sc, 60)
        assert [t[2] for t in ex["trades"]] == [base, base + 50, base + 100]
        assert [t[3] for t in ex["trades"]] == [20, 20, 20]

    def test_standard_iceberg_example_hidden_counts(self):
        ice = _mk_row(SELL, 1_005_000, 70, 1, 1, ICE, peak=20)
        sc = self._sc([ice], BUY, 1_005_000, part="C10b")
        assert sc.M == 70
        ex = derive_run(sc, 30)
        assert ex["verdict"] == "executed" and [t[3] for t in ex["trades"]] == [20, 10]

    def test_reranking_slice_worse_than_visible_trades_after_it(self):
        # SELL ICEwPI base 100, ppd +10, peak 5, total 20; NON SELL at 115 x 7; FOK BUY L 120.
        ice = _mk_row(SELL, 1_000_100, 20, 1, 1, ICE, peak=5, ppd=10)
        vis = _mk_row(SELL, 1_000_115, 7, 2, 2, NON)
        sc = self._sc([ice, vis], BUY, 1_000_120, part="C10d")
        assert sc.M == 5 + 5 + 5 + 7      # slices 100, 110, 120 cross; 130 does not
        ex = derive_run(sc, sc.M)
        assert [(_maker(sc, t), t[2], t[3]) for t in ex["trades"]] == [(1, 1_000_100, 5), (1, 1_000_110, 5), (2, 1_000_115, 7),
                                           (1, 1_000_120, 5)]
        assert derive_run(sc, sc.M + 1)["verdict"] == "killed"

    def test_reranking_tie_regenerated_slice_ranks_after_older_nonslice(self):
        # BUY ICEwPI base 200, ppd -10, peak 4, total 12; NON BUY at 190 x 3; FOK SELL L 180.
        ice = _mk_row(BUY, 1_000_200, 12, 1, 1, ICE, peak=4, ppd=-10)
        vis = _mk_row(BUY, 1_000_190, 3, 2, 2, NON)
        sc = self._sc([ice, vis], SELL, 1_000_180, part="C10d")
        ex = derive_run(sc, sc.M)
        assert [(t[0], t[2], t[3]) for t in ex["trades"]] == [
            (1, 1_000_200, 4), (2, 1_000_190, 3), (1, 1_000_190, 4), (1, 1_000_180, 4)]


class TestRunsAgainstRules:
    def test_every_run_of_every_draw_equals_the_rules_derivation(self, payload):
        errs = judge_runs(payload)
        assert errs == [], "\n".join(errs[:30])

    def test_c10c_closed_form_equals_slice_sum_on_every_draw(self, payload):
        for a, m in seed_assertions(payload):
            if m.group(2) == "C10c":
                sc = scene("C10c", int(m.group(1)))
                assert sc.M == sc.M_card, (sc.seed, sc.M, sc.M_card)
                assert 1 <= sc.reach < _ceil_div(sc.makers[0][C_QTY], sc.makers[0][C_PEAK])

    def test_c10d_construction_realisable_on_every_draw(self, payload):
        """A limit exists that crosses the NON level and exactly r slices; M = min(T, peak*r) + q_v."""
        bad = []
        for a, m in seed_assertions(payload):
            if m.group(2) == "C10d":
                sc = scene_d(int(m.group(1)), a)
                ice, vis = sc.makers
                if sc.problems:
                    bad.append((sc.seed, sc.problems))
                    continue
                assert sc.T_c == min(ice[C_QTY], ice[C_PEAK] * sc.reach)
                assert sc.M == sc.T_c + vis[C_QTY]
                assert _better(sc.rest_side, ice[C_PRICE], vis[C_PRICE])
        assert bad == [], bad[:5]

    def test_c10d_fill_crosses_iceberg_then_worse_visible_level_every_draw(self, payload):
        """Non-vacuity: the M run trades the iceberg AND the strictly worse NON level; on
        'interleaved' draws a slice is traded after the NON (re-ranking exercised)."""
        n_inter = 0
        for a, m in seed_assertions(payload):
            if m.group(2) != "C10d":
                continue
            sc = scene_d(int(m.group(1)), a)
            ice, vis = sc.makers
            run = _js(a["observed"])[str(sc.M)]
            maker = [t[0] if sc.rest_side == BUY else t[1] for t in run["trades"]]
            assert ice[C_OID] in maker and vis[C_OID] in maker, (sc.seed, run["trades"])
            # T_c + 1 must reach beyond the iceberg's crossing volume
            r1 = _js(a["observed"])[str(sc.T_c + 1)]
            assert sum(t[3] for t in r1["trades"]) == sc.T_c + 1
            if "interleaved" in sc.label:
                n_inter += 1
                iv = maker.index(vis[C_OID])
                assert ice[C_OID] in maker[iv + 1:], (sc.seed, run["trades"])
        assert n_inter > 0

    def test_both_verdicts_on_every_draw_of_every_part(self, payload):
        for a, m in seed_assertions(payload):
            part, seed = m.group(2), int(m.group(1))
            verdicts = {v["verdict"] for v in _js(a["observed"]).values()}
            assert verdicts == {"executed", "killed"}, (part, seed, verdicts)

    def test_c10b_hidden_volume_is_load_bearing(self, payload):
        """Every C10b draw has an executed run whose request exceeds the visible peak."""
        for a, m in seed_assertions(payload):
            if m.group(2) != "C10b":
                continue
            sc = scene("C10b", int(m.group(1)))
            peak = sc.makers[0][C_PEAK]
            obs = _js(a["observed"])
            assert any(int(r) > peak and v["verdict"] == "executed" for r, v in obs.items())

    def test_m4_on_every_run(self, payload):
        """Executed: filled == req exactly. Killed: zero fills, book unchanged, fok_kill req."""
        for (part, seed), (sc, a) in all_scenes(payload).items():
            for r, v in _js(a["observed"]).items():
                f = int(v["filled"][str(sc.fok_oid)])
                if v["verdict"] == "executed":
                    assert f == int(r) and sum(t[3] for t in v["trades"]) == int(r)
                else:
                    assert f == 0 and v["trades"] == [] and v["fok_book_unchanged"] is True
                    assert _canon_book(v["final_book"]) == _canon_book(_initial_book(sc))
                    assert v["cancelled"] == {str(sc.fok_oid): [["fok_kill", int(r)]]}

    def test_fill_prices_are_makers_own_uncapped(self, payload):
        """No trade prints anywhere but at a maker's / slice's own price, which crosses L."""
        for (part, seed), (sc, a) in all_scenes(payload).items():
            prices = {p for r in sc.makers for p, _ in _units(r)}
            for v in _js(a["observed"]).values():
                for t in v["trades"]:
                    assert t[2] in prices, (part, seed, t)
                    assert _crosses(sc.fok_side, sc.limit, t[2]), (part, seed, t)

    def test_k2_expectation_is_a_subset_of_what_the_rules_allow(self, payload):
        """The recorded K2 expectation never allows a book or outcome the rules forbid."""
        for (part, seed), (sc, a) in all_scenes(payload).items():
            for r, e in _js(a["expected"]).items():
                mine = derive_run(sc, int(r))
                allowed = {_canon_book(b) for b in mine["final_book_one_of"]}
                assert {_canon_book(b) for b in e["final_book_one_of"]} <= allowed, (part, seed, r)
                fake = dict(e, final_book=e["final_book_one_of"][0])
                assert judge_run(fake, mine) == [], (part, seed, r)


class TestAggregatesFailOnly:
    """RERUN-1 correction: count status == 'fail' only; report not_applicable counts."""

    def test_m4_i3_i5_aggregates_fail_only_and_report_na(self, payload):
        assert judge_aggregates(payload) == []

    def test_i5_not_applicable_on_killed_foks_is_not_counted_as_failure(self, payload):
        killed = killed_runs(payload)
        assert {p: len(v) for p, v in killed.items()} == {p: DRAWS_PER_PART for p in ALL_PARTS}
        a = next(a for a in payload["assertions"] if a["name"].startswith(AGG_PREFIX["I5"]))
        assert a["status"] == "pass" and a["observed"] == []
        n_na = int(AGG_DETAIL_RE.search(a["detail"]).group(2))
        assert n_na == sum(len(v) for v in killed.values()) > 0

    def test_no_aggregate_uses_not_equal_pass_semantics(self, payload):
        """r1's 'I5 passes on every run' (any non-pass = fail) must be gone."""
        assert not [a["name"] for a in payload["assertions"] if "passes on every run" in a["name"]]

    def test_i5_per_draw_battery_row_records_no_violation(self, payload):
        row = next(r for r in payload["invariants"] if r["check_id"] == "I5")
        assert row["n_fail"] == 0 and row["status"] != "fail"


class TestEvidence:
    def test_eager_vs_jitted_present_and_identical_per_part(self, payload):
        assert judge_eager_jit(payload) == []

    def test_full_battery_ran_on_every_valid_draw(self, payload):
        rows = payload["invariants"]
        assert [r["check_id"] for r in rows] == ["RB"] + [f"I{i}" for i in range(1, 10)]
        n = payload["draws"]["valid"]
        for r in rows:
            assert r["n_pass"] + r["n_fail"] + r["n_not_applicable"] == n, r["check_id"]
            assert r["status"] in ("pass", "fail", "not_applicable")
            assert r["n_fail"] == 0 and r["failing_seeds"] == [], r["check_id"]

    def test_class_coverage_matches_independent_labels(self, payload):
        cov = next(a for a in payload["assertions"] if a["name"].startswith("dataset: the valid draws cover"))
        mine = Counter(f"{part}: {sc.label}" for (part, _), (sc, _) in all_scenes(payload).items())
        observed = cov["observed"] if isinstance(cov["observed"], dict) else ast.literal_eval(cov["observed"])
        assert dict(mine) == observed
        assert len(mine) == 18 and min(mine.values()) >= 1

    def test_capacity_margins_and_peaks_match_derivation(self, payload):
        cap = payload["capacity"]
        assert cap["envelope"] == ENVELOPE
        max_slices = max_trades = max_rest = max_ice = 0
        for (part, seed), (sc, _) in all_scenes(payload).items():
            max_slices = max(max_slices, sc.max_slices)
            max_rest = max(max_rest, len(sc.makers))
            max_ice = max(max_ice, sum(1 for r in sc.makers if r[C_EXEC] == ICE))
            for r in requests_for(sc):
                max_trades = max(max_trades, len(derive_run(sc, r)["trades"]))
        assert max_slices <= SLICE_MARGIN
        pk = cap["peaks"]
        assert pk["trades"] == max_trades <= ENVELOPE["trades"]
        assert pk["resting_per_side"] == max_rest <= ENVELOPE["resting_per_side"]
        assert pk["concurrent_icebergs"] == max_ice <= ENVELOPE["concurrent_icebergs"]
        sl = [a for a in payload["assertions"] if a["name"].startswith("largest ceil(qty/peak)")]
        assert len(sl) == 1 and int(sl[0]["observed"]) == max_slices and sl[0]["status"] == "pass"
        hb = [a for a in payload["assertions"] if a["name"].startswith("hop bound <= 160")]
        assert len(hb) == 1 and hb[0]["status"] == "pass" and max(_js(hb[0]["observed"])) <= 160
        flags = [a for a in payload["assertions"] if a["name"].startswith("no capacity flag")]
        assert len(flags) == 1 and flags[0]["status"] == "pass"

    def test_quotes_present_ok_and_required_passages_still_in_source(self, payload):
        qs = payload["spec"]["quotes"]
        assert qs and all(q["ok"] for q in qs)
        secs = {(q["source"], q["section"]) for q in qs}
        for need in [("market-spec.md", "§4.2.1"), ("market-spec.md", "§4.5"),
                     ("market-spec.md", "§6.2 item 7"), ("open-questions.md", "OQ-11")]:
            assert need in secs, need
        rule1 = SpecQuote(source="market-spec.md", section="§4.2.1",
                          text="FOK orders see the iceberg's full remaining quantity for matching purposes")
        assert check_quote(rule1).ok
        fok_row = SpecQuote(source="market-spec.md", section="§4.5",
                            text="| FOK | No (full or cancel) | No (cancelled) | Regular, Linked basket |")
        assert check_quote(fok_row).ok
        assert quote_digest(fok_row) in {q["digest"] for q in qs}
        rerank = SpecQuote(source="market-spec.md", section="§4.2.1",
                           text="The aggressor always trades the best-ranked counter bid in the book next.")
        assert check_quote(rerank).ok
        for c in payload["citations"]:
            assert c["checked"] is False

    def test_assertion_units_are_never_euros(self, payload):
        for a in payload["assertions"]:
            assert a["unit"] in ("bool", "count", "other", "ticks", "qty_ticks", "price_ticks"), a["name"]


class TestFailureDetermination:
    def test_every_failing_assertion_is_explained(self, payload):
        artefacts, deviations = classify_failures(payload)
        assert deviations == [], deviations
        if payload["status"] != "PASS":
            assert artefacts, "NOT CERTIFIED with no failing assertion classified"

    def test_status_pass_has_no_failing_assertion(self, payload):
        if payload["status"] == "PASS":
            assert [a["name"] for a in payload["assertions"] if a["status"] == "fail"] == []
            assert payload["failure_kinds"] == [] and payload["failure_explanation"] is None


# ---------------------------------------------------------------------------
# Deliberately broken copies (each must be rejected)
# ---------------------------------------------------------------------------
def _find(payload, part, pred):
    for i, a in enumerate(payload["assertions"]):
        m = SEED_RE.match(a["name"])
        if m and m.group(2) == part:
            sc = scene_of(part, int(m.group(1)), a)
            if pred(sc):
                return i, sc
    raise AssertionError(f"no {part} draw satisfies the predicate")


def _set_run(p, i, req, run):
    obs = _js(p["assertions"][i]["observed"])
    obs[str(req)] = run
    p["assertions"][i]["observed"] = json.dumps(obs)


def _get_run(p, i, req):
    return _js(p["assertions"][i]["observed"])[str(req)]


def _only(p, i):
    """Judge only the mutated record (keeps the test fast)."""
    q = dict(p)
    q["assertions"] = [p["assertions"][i]]
    return [e for e in judge_runs(q) if "per-seed records <" not in e]


class TestBrokenCopiesRejected:
    def test_fok_accepted_on_visible_volume_only_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10b", lambda s: s.makers[0][C_QTY] > s.makers[0][C_PEAK])
        total = sc.makers[0][C_QTY]
        killed = derive_run(sc, total + 1)
        run = {k: v for k, v in killed.items() if k != "final_book_one_of"}
        run.update(final_book=_initial_book(sc), cancelled={str(sc.fok_oid): [["fok_kill", total]]})
        run["filled"] = {str(k): v for k, v in run["filled"].items()}
        run["remaining"] = {str(k): v for k, v in run["remaining"].items()}
        _set_run(p, i, total, run)
        p = _reload(p, tmp_path, "C10_hidden_ignored.json")
        errs = _only(p, i)
        assert any(f"run {total}: verdict 'killed' != 'executed'" in e for e in errs), errs

    def test_fok_partially_filled_on_visible_peak_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10b", lambda s: s.makers[0][C_QTY] > s.makers[0][C_PEAK])
        total, peak = sc.makers[0][C_QTY], sc.makers[0][C_PEAK]
        run = _get_run(p, i, total)
        run["trades"] = run["trades"][:1]
        run["filled"] = {k: (peak if v else 0) for k, v in run["filled"].items()}
        _set_run(p, i, total, run)
        p = _reload(p, tmp_path, "C10_visible_partial.json")
        errs = _only(p, i)
        assert any("trades" in e for e in errs) and any("filled" in e for e in errs), errs

    @pytest.mark.parametrize("part", ["C10a", "C10b", "C10c", "C10d"])
    def test_capped_fill_price_is_rejected(self, payload, tmp_path, part):
        p = copy.deepcopy(payload)
        pred = {"C10a": lambda s: len(s.makers) >= 2, "C10c": lambda s: s.reach >= 2,
                "C10b": lambda s: True, "C10d": lambda s: s.limit != s.makers[1][C_PRICE]}[part]
        i, sc = _find(p, part, pred)
        req = sc.M
        run = _get_run(p, i, req)
        assert run["verdict"] == "executed"
        run["trades"] = [[t[0], t[1], sc.limit, t[3], t[4]] for t in run["trades"]]
        _set_run(p, i, req, run)
        p = _reload(p, tmp_path, f"C10_capped_{part}.json")
        errs = _only(p, i)
        assert any(f"run {req}: trades" in e for e in errs), errs

    def test_dirty_rollback_book_changed_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10a", lambda s: True)
        req = sc.M + 1
        run = _get_run(p, i, req)
        run["final_book"][SIDE_NAME[sc.rest_side]][0][1] -= 1
        _set_run(p, i, req, run)
        p = _reload(p, tmp_path, "C10_dirty_book.json")
        assert any(f"run {req}: final_book" in e for e in _only(p, i))

    def test_dirty_rollback_flag_false_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10d", lambda s: True)
        req = sc.M + 1
        run = _get_run(p, i, req)
        run["fok_book_unchanged"] = False
        _set_run(p, i, req, run)
        p = _reload(p, tmp_path, "C10_dirty_flag.json")
        assert any("M4: fok_book_unchanged=False" in e for e in _only(p, i))

    def test_kill_leaving_a_partial_trade_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10b", lambda s: True)
        req = sc.makers[0][C_QTY] + 1
        run = _get_run(p, i, req)
        run["trades"] = _get_run(p, i, sc.makers[0][C_QTY])["trades"][:1]
        _set_run(p, i, req, run)
        p = _reload(p, tmp_path, "C10_partial_trade_on_kill.json")
        assert any(f"run {req}: trades" in e for e in _only(p, i))

    def test_icewpi_projection_one_slice_too_far_is_rejected(self, payload, tmp_path):
        """Engine counts reach+1 slices: M+1 executes, the extra tick at slice ``reach``."""
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10c", lambda s: True)
        ice = sc.makers[0]
        req = sc.M + 1
        good = _get_run(p, i, sc.M)
        run = copy.deepcopy(good)
        buyer, seller = (ice[C_OID], sc.fok_oid) if sc.rest_side == BUY else (sc.fok_oid, ice[C_OID])
        run["trades"] = good["trades"] + [[buyer, seller, ice[C_PRICE] + sc.reach * ice[C_PPD], 1,
                                           sc.fok_sub]]
        run["filled"] = {str(ice[C_OID]): sc.M + 1, str(sc.fok_oid): req}
        run["remaining"] = {str(ice[C_OID]): ice[C_QTY] - sc.M - 1, str(sc.fok_oid): 0}
        _set_run(p, i, req, run)
        p = _reload(p, tmp_path, "C10_projection_plus_one.json")
        assert any(f"run {req}: verdict 'executed' != 'killed'" in e for e in _only(p, i))

    def test_icewpi_projection_one_slice_short_is_rejected(self, payload, tmp_path):
        """Engine counts reach-1 slices: the run at M is killed."""
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10c", lambda s: True)
        run = _get_run(p, i, sc.M + 1)
        run["cancelled"] = {str(sc.fok_oid): [["fok_kill", sc.M]]}
        _set_run(p, i, sc.M, run)
        p = _reload(p, tmp_path, "C10_projection_minus_one.json")
        assert any(f"run {sc.M}: verdict 'killed' != 'executed'" in e for e in _only(p, i))

    def test_c10d_iceberg_drained_before_better_visible_is_rejected(self, payload, tmp_path):
        """No re-ranking: the FOK walks every crossing slice before the better NON level."""
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10d", lambda s: "interleaved" in s.label)
        ice_oid = sc.makers[0][C_OID]
        run = _get_run(p, i, sc.M)
        maker = (lambda t: t[0]) if sc.rest_side == BUY else (lambda t: t[1])
        run["trades"] = ([t for t in run["trades"] if maker(t) == ice_oid]
                         + [t for t in run["trades"] if maker(t) != ice_oid])
        _set_run(p, i, sc.M, run)
        p = _reload(p, tmp_path, "C10d_no_rerank.json")
        assert any(f"run {sc.M}: trades" in e for e in _only(p, i))

    def test_c10d_precheck_ignoring_the_worse_visible_level_is_rejected(self, payload, tmp_path):
        """Pre-check counts the iceberg only: the run at M (> T_c) is killed."""
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10d", lambda s: True)
        run = _get_run(p, i, sc.M + 1)
        run["cancelled"] = {str(sc.fok_oid): [["fok_kill", sc.M]]}
        _set_run(p, i, sc.M, run)
        p = _reload(p, tmp_path, "C10d_visible_ignored.json")
        assert any(f"run {sc.M}: verdict 'killed' != 'executed'" in e for e in _only(p, i))

    def test_c10d_slice_split_merged_into_one_trade_is_rejected(self, payload, tmp_path):
        """Two consecutive trades of the same maker merged (one trade per counter entry)."""
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10d", lambda s: "standard" in s.label and s.T_c > s.makers[0][C_PEAK])
        run = _get_run(p, i, sc.M)
        t0, t1 = run["trades"][0], run["trades"][1]
        run["trades"] = [[t0[0], t0[1], t0[2], t0[3] + t1[3], t0[4]]] + run["trades"][2:]
        _set_run(p, i, sc.M, run)
        p = _reload(p, tmp_path, "C10d_merged.json")
        assert any(f"run {sc.M}: trades" in e for e in _only(p, i))

    def test_c10d_wrong_class_label_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10d", lambda s: "behind" in s.label)
        p["assertions"][i]["name"] = p["assertions"][i]["name"].replace("behind", "interleaved")
        p = _reload(p, tmp_path, "C10d_label.json")
        assert any("class 'iceberg" in e for e in _only(p, i))

    def test_missing_run_evidence_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10d", lambda s: True)
        obs = _js(p["assertions"][i]["observed"])
        del obs[str(sc.T_c + 1)]
        p["assertions"][i]["observed"] = json.dumps(obs)
        p = _reload(p, tmp_path, "C10_missing_run.json")
        assert any("evidence not in full" in e for e in _only(p, i))

    def test_status_lie_on_a_conforming_record_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10a", lambda s: True)
        p["assertions"][i]["status"] = "fail"
        p = _reload(p, tmp_path, "C10_status_lie.json")
        assert any("recorded status 'fail'" in e for e in _only(p, i))

    def test_absent_eager_jit_evidence_for_c10d_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        p["assertions"] = [a for a in p["assertions"]
                           if not (a["kind"] == "eager_jit" and "C10d" in a["name"])]
        p = _reload(p, tmp_path, "C10_no_ej_d.json")
        assert "C10d: no eager-vs-jitted evidence" in judge_eager_jit(p)

    def test_eager_jit_disagreement_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        a = next(a for a in p["assertions"] if a["kind"] == "eager_jit")
        a["observed"] = False
        p = _reload(p, tmp_path, "C10_ej_false.json")
        assert any("eager != jitted" in e for e in judge_eager_jit(p))

    def test_status_pass_with_a_failure_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        p["assertions"][0]["status"] = "fail"
        p["status"], p["failure_kinds"], p["failure_explanation"] = "PASS", [], None
        p = _reload(p, tmp_path, "C10_status_pass.json")
        assert judge_status(p) != []

    def test_aggregate_counting_not_applicable_as_failure_is_rejected(self, payload, tmp_path):
        """The r1 defect: I5 n/a on every killed FOK listed as a failure."""
        killed = killed_runs(payload)
        p = copy.deepcopy(payload)
        a = next(a for a in p["assertions"] if a["name"].startswith(AGG_PREFIX["I5"]))
        flat = [e for v in killed.values() for e in v]
        a["observed"], a["status"] = flat, "fail"
        a["detail"] = re.sub(r"failing runs: \d+", f"failing runs: {len(flat)}", a["detail"])
        p = _reload(p, tmp_path, "C10_i5_na_as_fail.json")
        errs = judge_aggregates(p, killed)
        assert any("counted as failures" in e for e in errs), errs
        # ... and the classifier recognises it as a judging artefact, not an engine deviation.
        art, dev = classify_failures(p)
        assert art and dev == []

    def test_aggregate_hiding_a_failing_run_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        a = next(a for a in p["assertions"] if a["name"].startswith(AGG_PREFIX["M4"]))
        a["detail"] = re.sub(r"failing runs: \d+", "failing runs: 1", a["detail"])
        p = _reload(p, tmp_path, "C10_m4_hidden.json")
        errs = judge_aggregates(p, killed_runs(payload))
        assert any(e.startswith("M4:") for e in errs), errs

    def test_aggregate_without_na_counts_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        a = next(a for a in p["assertions"] if a["name"].startswith(AGG_PREFIX["I5"]))
        a["detail"] = "valid draws: 2000; failing runs: 0"
        p = _reload(p, tmp_path, "C10_no_na.json")
        assert any("does not report the n/a counts" in e for e in judge_aggregates(p, killed_runs(payload)))

    def test_i5_flag_on_an_executed_run_is_classified_as_deviation(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, sc = _find(p, "C10d", lambda s: True)
        entry = f"C10d {sc.seed}:{sc.M}"            # an executed run
        a = next(a for a in p["assertions"] if a["name"].startswith(AGG_PREFIX["I5"]))
        a["observed"], a["status"] = [entry], "fail"
        p = _reload(p, tmp_path, "C10_i5_exec.json")
        _, deviations = classify_failures(p)
        assert any(entry in d for d in deviations), deviations


# ---------------------------------------------------------------------------
# Optional engine re-run of the generator-backed parts (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.jit
def test_rerun_two_seeds_per_synth_part_eager_and_jitted(payload):
    """slow: compiles submit_order and runs 6 draws x 2-3 requests, eager and jitted."""
    from tests.omiecid.reference.omie_invariants import (
        check_m4_fok_dichotomy, compare_records, record_from_draw, run_battery)

    recs = {(m.group(2), int(m.group(1))): _js(a["observed"]) for a, m in seed_assertions(payload)}
    for part in SYNTH_PARTS:
        picked = {}
        for (pp, seed) in sorted(recs):
            if pp == part:
                picked.setdefault(scene(part, seed).rest_side, seed)
        for seed in picked.values():
            sc = scene(part, seed)
            d = _draw(part, seed)
            fok_row = [i for i, r in enumerate(d.tape) if int(r[C_OID]) == sc.fok_oid][0]
            for req in requests_for(sc):
                tape = np.array(d.tape, copy=True)
                tape[fok_row, C_QTY] = req
                dr = dataclasses.replace(d, tape=tape)
                eager = record_from_draw(dr, jit=False)
                jitted = record_from_draw(dr, jit=True)
                assert compare_records(eager, jitted) == (), (part, seed, req)
                exp = derive_run(sc, req)
                trades = [[f.buyer_oid, f.seller_oid, f.price_int, f.qty_int, f.submission]
                          for f in jitted.fills]
                assert trades == exp["trades"], (part, seed, req)
                assert trades == [list(t) for t in recs[(part, seed)][str(req)]["trades"]]
                fin = jitted.final_snapshot
                book = {SIDE_NAME[s]: [[e[0], e[1], e[2]] for e in fin.resting(s)] for s in (BUY, SELL)}
                assert _canon_book(book) in {_canon_book(b) for b in exp["final_book_one_of"]}
                assert check_m4_fok_dichotomy(jitted, fok_oid=sc.fok_oid).status == "pass"
                rep = run_battery(jitted, expects_trade=exp["verdict"] == "executed")
                st = rep.statuses()
                assert rep.passed, st
                # I5 is vacuous (not_applicable) on a kill, pass on an execution - never fail.
                assert st["I5"] == ("not_applicable" if exp["verdict"] == "killed" else "pass"), st


# ---------------------------------------------------------------------------
# Independent engine probes (ContractBook public API) for C10d: iceberg + a
# strictly worse NON level, FOK at exactly the matchable quantity / one more.
# Hand-built, off-round constructions; expected fills hand-pinned AND walked.
# ---------------------------------------------------------------------------
_PROBE_CID = 87
_TP, _TQ, _TPASS, _TAGRS = 0, 1, 2, 3     # jaxob trade rows: (P, Q, PASS_OID, AGRS_OID, ...)

# name: (ice_side, base, ppd, peak, total, vis_price, vis_qty, L, M, hand-pinned fills
#        as (maker 'I'/'V', price, qty) in rank order)
_D_CASES = {
    # SELL ICEwPI, NON between slice 1 and slice 2: s0, s1, NON, s2.
    "sell_icewpi_interleaved": (SELL, 1_004_311, 9, 13, 61, 1_004_325, 17, 1_004_329, 56,
                                [("I", 1_004_311, 13), ("I", 1_004_320, 13), ("V", 1_004_325, 17),
                                 ("I", 1_004_329, 13)]),
    # BUY ICEwPI, slice 2 ties the NON price: the older NON first, then the regenerated slice.
    "buy_icewpi_tie_at_visible": (BUY, 1_006_207, -7, 11, 50, 1_006_193, 19, 1_006_186, 63,
                                  [("I", 1_006_207, 11), ("I", 1_006_200, 11), ("V", 1_006_193, 19),
                                   ("I", 1_006_193, 11), ("I", 1_006_186, 11)]),
    # SELL standard iceberg, every slice at base, then the worse NON.
    "sell_standard": (SELL, 1_002_517, 0, 7, 30, 1_002_521, 23, 1_002_530, 53,
                      [("I", 1_002_517, 7), ("I", 1_002_517, 7), ("I", 1_002_517, 7),
                       ("I", 1_002_517, 7), ("I", 1_002_517, 2), ("V", 1_002_521, 23)]),
    # BUY ICEwPI, NON behind every crossing slice (slice 4 at 1_003_378 does not cross).
    "buy_icewpi_behind": (BUY, 1_003_402, -6, 9, 40, 1_003_380, 14, 1_003_380, 50,
                          [("I", 1_003_402, 9), ("I", 1_003_396, 9), ("I", 1_003_390, 9),
                           ("I", 1_003_384, 9), ("V", 1_003_380, 14)]),
}
_ICE_OID, _VIS_OID, _FOK_OID = 6_101, 6_102, 6_109


@pytest.fixture(scope="module")
def probe_book():
    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.contract_book import ContractBook
    from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig

    c = Contract(contract_id=_PROBE_CID, delivery_start_minutes=18 * 60 + 45,
                 delivery_end_minutes=19 * 60, gct_minutes=17 * 60 + 45,
                 status=ContractStatus.OPEN)
    pc = OMIEProductConfig()
    return ContractBook(c, pc, OMIEMarketConfig()), pc


def _probe_order(pc, side, p_tick, q_tick, oid, ts, exec_r, peak_tick=0, ppd_tick=0):
    from omiecid_simulator.omiecid.order_types import make_order

    off = int(round(-pc.price_min / pc.price_tick))
    return make_order(
        contract_id=_PROBE_CID, side=Side(side), price=round((p_tick - off) * pc.price_tick, 2),
        quantity=round(q_tick * pc.qty_tick, 1), order_id=oid, agent_id=oid % 97 + 1, timestamp=ts,
        exec_restriction=ExecRestriction(exec_r),
        peak_qty=round(peak_tick * pc.qty_tick, 1) if peak_tick else 0.0,
        ppd=round(ppd_tick * pc.price_tick, 2))


def _resting(state, side) -> List[List[int]]:
    arr = np.asarray(state.lob.bids if side == BUY else state.lob.asks)
    return sorted([int(r[0]), int(r[1]), int(r[2])] for r in arr if int(r[2]) > 0 and int(r[1]) > 0)


def _book_arrays(state):
    return [np.asarray(x) for x in (state.lob.bids, state.lob.asks, state.lob.trades,
                                    state.iceberg_meta, state.n_active)]


def _case_scene(case) -> Scene:
    ice_side, base, ppd, peak, total, pv, qv, L, M, _ = _D_CASES[case]
    fok_side = SELL if ice_side == BUY else BUY
    makers = (_mk_row(ice_side, base, total, _ICE_OID, 1, ICE, peak, ppd),
              _mk_row(ice_side, pv, qv, _VIS_OID, 2, NON))
    return Scene(part="C10d", seed=-1, makers=makers, fok_side=fok_side, rest_side=ice_side,
                 limit=L, drawn_req=M, fok_oid=_FOK_OID, fok_sub=2, M=_precheck(makers, fok_side, L))


def _setup(book, pc, case):
    ice_side, base, ppd, peak, total, pv, qv, *_ = _D_CASES[case]
    s0 = book.init()
    s1, r1 = book.submit_order(s0, _probe_order(pc, ice_side, base, total, _ICE_OID, 1, ICE, peak, ppd))
    s2, r2 = book.submit_order(s1, _probe_order(pc, ice_side, pv, qv, _VIS_OID, 2, NON))
    assert bool(r1.accepted) and bool(r2.accepted) and not bool(r1.traded) and not bool(r2.traded)
    assert _resting(s2, ice_side) == sorted([[base, min(peak, total), _ICE_OID], [pv, qv, _VIS_OID]])
    return s2


def _fok(book, pc, state, case, req, limit):
    fok_side = SELL if _D_CASES[case][0] == BUY else BUY
    before = np.asarray(state.lob.trades)
    s, r = book.submit_order(state, _probe_order(pc, fok_side, limit, req, _FOK_OID, 3, FOK))
    after = np.asarray(s.lob.trades)
    rows = after[np.any(before != after, axis=1)]
    got = [(int(t[_TPASS]), int(t[_TP]), abs(int(t[_TQ]))) for t in rows]
    return s, r, got


class TestCaseDerivation:
    """The hand pins agree with the walk and the pre-check (no engine)."""

    @pytest.mark.parametrize("case", sorted(_D_CASES))
    def test_hand_pins_equal_walk(self, case):
        sc = _case_scene(case)
        *_, M, pins = _D_CASES[case]
        assert sc.M == M
        oid = {"I": _ICE_OID, "V": _VIS_OID}
        ex = derive_run(sc, M)
        maker = (lambda t: t[0]) if sc.rest_side == BUY else (lambda t: t[1])
        assert [(maker(t), t[2], t[3]) for t in ex["trades"]] == [(oid[k], p, q) for k, p, q in pins]
        assert derive_run(sc, M + 1)["verdict"] == "killed"


@pytest.mark.jit
class TestEngineProbesC10d:
    """jit: ContractBook.submit_order drives the module-level jitted submit (compile per worker)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("case", sorted(_D_CASES))
    def test_fok_at_exact_matchable_quantity_fills_in_rank_order(self, probe_book, case):
        """slow: jitted submit. req = M executes in full, each piece at its maker's price, in rank order."""
        book, pc = probe_book
        sc = _case_scene(case)
        *_, L, M, pins = _D_CASES[case]
        s, r, got = _fok(book, pc, _setup(book, pc, case), case, M, L)
        oid = {"I": _ICE_OID, "V": _VIS_OID}
        assert got == [(oid[k], p, q) for k, p, q in pins], f"engine fills {got}"
        assert bool(r.accepted) and bool(r.traded)
        exp = derive_run(sc, M)
        assert got == [(_maker(sc, t), t[2], t[3]) for t in exp["trades"]]
        fin = {"BUY": _resting(s, BUY), "SELL": _resting(s, SELL)}
        assert _canon_book(fin) in {_canon_book(b) for b in exp["final_book_one_of"]}, fin
        assert all(e[2] != _FOK_OID for side in fin.values() for e in side), "a FOK never rests"
        assert not bool(s.truncated) and not bool(s.book_full)

    @pytest.mark.slow
    @pytest.mark.parametrize("case", sorted(_D_CASES))
    def test_fok_one_lot_more_is_killed_book_unchanged(self, probe_book, case):
        """slow: jitted submit. req = M + 1: no trade, book / trades / iceberg state byte-identical."""
        book, pc = probe_book
        *_, L, M, _ = _D_CASES[case]
        s0 = _setup(book, pc, case)
        s, r, got = _fok(book, pc, s0, case, M + 1, L)
        assert got == [] and not bool(r.traded)
        for x, y in zip(_book_arrays(s0), _book_arrays(s)):
            assert x.dtype == y.dtype and np.array_equal(x, y)

    @pytest.mark.slow
    @pytest.mark.parametrize("case", ["sell_icewpi_interleaved", "buy_icewpi_tie_at_visible",
                                      "buy_icewpi_behind"])
    def test_fok_limit_one_tick_less_aggressive_is_killed(self, probe_book, case):
        """slow: jitted submit. ICEwPI projection: L one tick short of the last crossing slice drops
        that slice from the pre-check, so req = M is killed and the book is unchanged."""
        book, pc = probe_book
        ice_side, *_, L, M, _ = _D_CASES[case]
        L2 = L - 1 if ice_side == SELL else L + 1
        sc = _case_scene(case)
        assert _precheck(sc.makers, sc.fok_side, L2) < M, "case precondition"
        s0 = _setup(book, pc, case)
        s, r, got = _fok(book, pc, s0, case, M, L2)
        assert got == [] and not bool(r.traded)
        for x, y in zip(_book_arrays(s0), _book_arrays(s)):
            assert np.array_equal(x, y)

    @pytest.mark.slow
    def test_fok_limit_reaching_next_slice_executes_m_plus_one(self, probe_book):
        """slow: jitted submit. L moved to the next slice price adds that slice: M + 1 executes,
        and its last tick prints at that slice's drifted price (uncapped, slice is the price-setter)."""
        book, pc = probe_book
        case = "sell_icewpi_interleaved"
        _, base, ppd, *_, L, M, pins = _D_CASES[case]
        L2 = base + 3 * ppd          # L sits on slice 2; slice 3 is the next one
        sc = dataclasses.replace(_case_scene(case), limit=L2)
        sc = dataclasses.replace(sc, M=_precheck(sc.makers, sc.fok_side, L2))
        assert sc.M > M
        s, r, got = _fok(book, pc, _setup(book, pc, case), case, M + 1, L2)
        assert bool(r.traded)
        assert got == [(_maker(sc, t), t[2], t[3]) for t in derive_run(sc, M + 1)["trades"]]
        assert got[-1] == (_ICE_OID, L2, 1)
