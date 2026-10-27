"""K4 audit twin for OMIE-CERT-C09-1 (price-incremented icebergs, ICEwPI) -- re-run r2.

Fresh-context, adversarial audit (C09 is high-risk, independence rule 4). The
expectation is re-derived from the rules alone (``docs/omiecid/market-spec.md``
**v2.14**) and from the drawn tapes of ``omie_tape_synth`` (``C09.icewpi_drift``,
``C09.icewpi_wrong_sign``). It never reads or executes the C09 notebook's
expectation code, the generator's ``Draw.expected``/``Draw.params``, or the engine
modules' internals (independence rules 3/4); the engine is touched only through
``ContractBook``'s public API in the probes at the end.

r2 (OMIE-CERT-RERUN-1): the scope is one iceberg and one NON aggressor, no
competing level, on every draw. The wrong-sign draws are no longer an un-asserted
observation: §5.4 (v2.14) rejects an ICEwPI BUY with PPD > 0 or SELL with PPD < 0,
so every wrong-sign draw is an ASSERTED rejection -- the iceberg rejected at entry,
zero trades, the book unchanged by its submission, and the aggressor resting whole
at its limit.

Rules used, all in integer ticks:

* §4.2 ICEwPI: slice ``k`` rests at ``base + k*ppd``; sign rule (§13.6): BUY
  ``ppd < 0``, SELL ``ppd > 0``. The main draws must be sign-valid.
* §4.2.1 Rule 2: one aggressive order consumes slice after slice inside one
  ``submit_order`` and stops at the first regenerated slice whose drifted price
  no longer crosses its limit (§6.2 item 2: ``sell <= buy``). That slice index is
  the stop index ``m``; the price sequence is ``base, base+ppd, ..., base+(m-1)ppd``.
* §6.2 item 7 / §5.1.2: the iceberg sets every trade price.
* §6.2 item 4: the aggressor's NON residual rests at its own limit ``L``.
* Iceberg left resting with positive hidden quantity. Its slice ``m`` is either
  shown at exactly ``base + m*ppd`` or not yet activated (OQ-20 ruling: a permitted
  between-aggressor state); nothing else.
* §5.4 (v2.14) + §13.6 item 1 + OMIE Rules Annex §4.5: a wrong-sign ICEwPI is
  REJECTED (§5.1). Each wrong-sign draw (iceberg first, then a NON aggressor that
  crosses its base, so an engine ignoring the sign rule WOULD trade) must show:
  the iceberg in ``rejected`` (submission 0), only the aggressor accepted, zero
  trades, zero fills, no hidden quantity, the book unchanged by submission 0, and
  the aggressor resting whole at ``L`` (§6.2 item 4) as the only resting entry.

Results JSON: ``$OMIE_CERT_RESULTS_DIR/C09.json`` else
``notebooks/certification/results/C09.json``; absent -> skip.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid import omie_tape_synth as synth
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from notebooks.certification.cert_support import (
    SpecQuote,
    check_quote,
    derive_failure_kinds,
    normalise,
    quote_digest,
    validate_results,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_ID = "C09"
MAIN_KEY = "C09.icewpi_drift"
WRONG_KEY = "C09.icewpi_wrong_sign"

BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, ICE = int(ExecRestriction.NON), int(ExecRestriction.ICE)
SIDE_NAME = {BUY: "BUY", SELL: "SELL"}

# Tape columns (loader layout, documented in omie_tape_synth).
C_SIDE, C_PRICE, C_QTY, C_OID, C_EXEC, C_PEAK, C_PPD = 1, 2, 3, 4, 7, 11, 12

ALL_CLASSES = ("iceberg BUY, m = 1", "iceberg BUY, m >= 2",
               "iceberg SELL, m = 1", "iceberg SELL, m >= 2")
WS_CLASSES = ("wrong-sign BUY (ppd > 0)", "wrong-sign SELL (ppd < 0)")
BATTERY = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")
ENVELOPE = {"resting_per_side": 80, "trades": 80, "hops": 160, "concurrent_icebergs": 800}
SLICE_MARGIN = 64
SPEC_VERSION = "v2.14"   # pinned: the r2 re-run is judged against market-spec v2.14 (§5.4 sign rejection)
N_MAIN, N_WRONG = 500, 20  # the card's draw counts (random-scenario default; 20 wrong-sign draws)

_PER_DRAW_RE = re.compile(r"^seed (\d+) \[(iceberg (?:BUY|SELL), m (?:= 1|>= 2))\]: ")
_WS_RE = re.compile(r"^seed (\d+) \[(wrong-sign (?:BUY \(ppd > 0\)|SELL \(ppd < 0\)))\]: ")
_EJ_RE = re.compile(r"^eager == jitted \(seed (\d+)\)$")
_REQUIRED_OBS_KEYS = ("accepted", "filled", "remaining", "trades", "final_book", "price_sequence",
                      "stop_index", "hidden_positive", "rejected", "lost")
_REQUIRED_WS_KEYS = ("accepted", "rejected", "filled", "remaining", "trades", "final_book",
                     "ice_hidden", "lost", "book_unchanged_by_submission_0")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _results_path() -> Path:
    d = os.environ.get("OMIE_CERT_RESULTS_DIR")
    base = Path(d) if d else REPO_ROOT / "notebooks" / "certification" / "results"
    return base / f"{CERT_ID}.json"


def _load(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    p = _results_path()
    if not p.is_file():
        pytest.skip(f"C09 results JSON absent: {p} (run the C09 notebook, section 8, first)")
    return _load(p)


def _roundtrip(tmp_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    """Write a (broken) copy to tmp_path and read it back, as the real file is read."""
    p = tmp_path / f"{CERT_ID}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return _load(p)


# ---------------------------------------------------------------------------
# Independent derivation (rules + drawn tape only)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _draw(key: str, seed: int):
    return synth.CONSTRUCTIONS[key](int(seed))


def _real(tape: np.ndarray) -> np.ndarray:
    t = np.asarray(tape).reshape(-1, 13)
    return t[t[:, C_OID] >= 1]


def _ceil(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _crosses(taker_side: int, taker_limit: int, maker_price: int) -> bool:
    """§6.2 item 2: match iff sell_price <= buy_price."""
    return taker_limit >= maker_price if taker_side == BUY else taker_limit <= maker_price


def _other(side: int) -> int:
    return SELL if side == BUY else BUY


def _sign_valid(side: int, ppd: int) -> bool:
    """§13.6 hard sign rule: buy => PPD < 0, sell => PPD > 0."""
    return ppd < 0 if side == BUY else ppd > 0


def _book_key(book: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    return {s: sorted([list(map(int, e)) for e in book.get(s, [])]) for s in ("BUY", "SELL")}


def _walk(ice: np.ndarray, agg: np.ndarray, ppd: int) -> Dict[str, Any]:
    """§4.2 + §4.2.1 Rule 2 walk of one NON aggressor through an ICEwPI ladder.

    Slice k (size peak, last one the remainder) rests at base + k*ppd; the
    aggressor consumes slices while the slice crosses its limit; the iceberg sets
    every price (§6.2 item 7). ``ppd`` is passed explicitly so the literal and the
    sign-flipped ladders can be derived with the same code.
    """
    ice_oid, agg_oid = int(ice[C_OID]), int(agg[C_OID])
    base, total, peak = int(ice[C_PRICE]), int(ice[C_QTY]), int(ice[C_PEAK])
    L, q, agg_side = int(agg[C_PRICE]), int(agg[C_QTY]), int(agg[C_SIDE])
    n = _ceil(total, peak)
    sizes = [peak] * (n - 1) + [total - (n - 1) * peak]
    buyer, seller = (agg_oid, ice_oid) if agg_side == BUY else (ice_oid, agg_oid)
    k, q_rem, trades = 0, q, []
    while True:
        if k == n:
            reason = "iceberg_exhausted"
            break
        p = base + k * ppd
        if not _crosses(agg_side, L, p):
            reason = "non_crossing"
            break
        t = min(q_rem, sizes[k])
        trades.append([buyer, seller, p, t, 1])
        q_rem -= t
        sizes[k] -= t
        if q_rem == 0:
            reason = "aggressor_filled"
            break
        k += 1
    return {"trades": trades, "stop": k, "reason": reason, "q_rem": q_rem, "sizes": sizes,
            "n": n, "ice_oid": ice_oid, "agg_oid": agg_oid, "base": base, "ppd": ppd, "peak": peak,
            "total": total, "L": L, "q": q, "ice_side": int(ice[C_SIDE]), "agg_side": agg_side}


@functools.lru_cache(maxsize=None)
def _derive(seed: int) -> Dict[str, Any]:
    """The expected outcome of one main (sign-valid) draw."""
    real = _real(_draw(MAIN_KEY, seed).tape)
    assert len(real) == 2, "dataset: one iceberg and one aggressor"
    ice, agg = real[0], real[1]
    ppd = int(ice[C_PPD])
    assert int(ice[C_EXEC]) == ICE and ppd != 0, "dataset: row 0 is an ICEwPI iceberg (ppd != 0)"
    assert _sign_valid(int(ice[C_SIDE]), ppd), "dataset: main draws are sign-valid (§13.6)"
    assert int(agg[C_EXEC]) == NON and int(agg[C_SIDE]) != int(ice[C_SIDE]), "dataset: row 1 NON, opposite"
    assert 0 < int(ice[C_PEAK]) <= int(ice[C_QTY])
    w = _walk(ice, agg, ppd)
    m = w["stop"]
    # Dataset precondition: the aggressor crosses exactly m >= 1 slices, has quantity
    # left, and slice m exists (so the iceberg is still resting).
    assert w["reason"] == "non_crossing" and m >= 1 and w["q_rem"] > 0 and m < w["n"], \
        f"seed {seed}: dataset precondition (crosses exactly m slices, quantity exceeds them)"
    filled = w["q"] - w["q_rem"]
    ice_rem = w["total"] - filled
    shown = w["sizes"][m]  # slice m, untouched: min(peak, remaining)
    hidden_if_shown = ice_rem - shown
    ice_name, agg_name = SIDE_NAME[w["ice_side"]], SIDE_NAME[w["agg_side"]]
    agg_rest = [[w["L"], w["q_rem"], w["agg_oid"]]]
    books = [
        _book_key({ice_name: [[w["base"] + m * ppd, shown, w["ice_oid"]]], agg_name: agg_rest}),
        _book_key({ice_name: [], agg_name: agg_rest}),  # slice m not yet activated (OQ-20)
    ]
    return {
        **w,
        "m": m,
        "price_sequence": [w["base"] + k * ppd for k in range(m)],
        "accepted": [w["ice_oid"], w["agg_oid"]],
        "rejected": [],
        "lost": {},
        "filled": {str(w["ice_oid"]): filled, str(w["agg_oid"]): filled},
        "remaining": {str(w["ice_oid"]): ice_rem, str(w["agg_oid"]): w["q_rem"]},
        "final_books": books,
        "slice_m_price": w["base"] + m * ppd,
        "slice_m_size": shown,
        "hidden_if_shown": hidden_if_shown,
        "hidden_if_absent": ice_rem,
        "label": f"iceberg {ice_name}, m {'= 1' if m == 1 else '>= 2'}",
    }


@functools.lru_cache(maxsize=None)
def _derive_wrong(seed: int) -> Dict[str, Any]:
    """The expected outcome of one wrong-sign draw: §5.4 (v2.14) rejection at entry.

    Submission 0 is the iceberg; it is rejected, so nothing of it reaches the SOB
    (§5.1 REJECTED, §5.2 no timestamp). Submission 1, the NON aggressor, then meets
    an empty opposite side: no trade, and it rests whole at its own limit (§6.2 item 4).
    """
    real = _real(_draw(WRONG_KEY, seed).tape)
    assert len(real) == 2, "dataset: one iceberg and one aggressor, no competing level"
    ice, agg = real[0], real[1]
    ppd, ice_side = int(ice[C_PPD]), int(ice[C_SIDE])
    assert int(ice[C_EXEC]) == ICE and ppd != 0, "dataset: row 0 is an ICEwPI iceberg (ppd != 0)"
    assert not _sign_valid(ice_side, ppd), "dataset: wrong-sign draws violate §13.6 / §5.4"
    assert int(agg[C_EXEC]) == NON and int(agg[C_SIDE]) != ice_side, "dataset: row 1 NON, opposite"
    assert 0 < int(ice[C_PEAK]) <= int(ice[C_QTY]), "dataset: peak rule satisfied (only the sign is wrong)"
    # Non-vacuity: an engine that ignored the sign rule would trade on this draw.
    lit = _walk(ice, agg, ppd)
    assert lit["trades"], f"seed {seed}: aggressor does not cross the iceberg's base (vacuous)"
    ice_oid, agg_oid = int(ice[C_OID]), int(agg[C_OID])
    L, q = int(agg[C_PRICE]), int(agg[C_QTY])
    ice_name, agg_name = SIDE_NAME[ice_side], SIDE_NAME[int(agg[C_SIDE])]
    return {
        "ice_oid": ice_oid, "agg_oid": agg_oid, "ice_side": ice_side, "agg_side": int(agg[C_SIDE]),
        "base": int(ice[C_PRICE]), "ppd": ppd, "L": L, "q": q,
        "literal_trades": lit["trades"],
        "label": f"wrong-sign {ice_name} (ppd {'> 0' if ppd > 0 else '< 0'})",
        "accepted": [agg_oid],
        "rejected_oid": ice_oid,
        "trades": [],
        "filled": {str(ice_oid): 0, str(agg_oid): 0},
        "remaining": {str(ice_oid): 0, str(agg_oid): q},
        "final_book": _book_key({ice_name: [], agg_name: [[L, q, agg_oid]]}),
        "ice_hidden": 0,
        "lost": {},
    }


def _safe_derive_wrong(seed: int) -> Optional[Dict[str, Any]]:
    try:
        return _derive_wrong(seed)
    except AssertionError:
        return None


@functools.lru_cache(maxsize=None)
def _derive_flipped(seed: int) -> List[List[int]]:
    """The trades a sign-error engine (slices drifting the wrong way) would print."""
    real = _real(_draw(MAIN_KEY, seed).tape)
    return _walk(real[0], real[1], -int(real[0][C_PPD]))["trades"]


# ---------------------------------------------------------------------------
# Checks on an emitted payload; each returns a list of problems ([] = clean)
# ---------------------------------------------------------------------------
def _parse(v):
    return json.loads(v) if isinstance(v, str) else v


def _per_draw(payload) -> List[Tuple[int, str, Dict[str, Any]]]:
    out = []
    for a in payload.get("assertions", []):
        m = _PER_DRAW_RE.match(a.get("name", ""))
        if m and a.get("kind") == "oracle":
            out.append((int(m.group(1)), m.group(2), a))
    return out


def _ws_draws(payload) -> List[Tuple[int, str, Dict[str, Any]]]:
    out = []
    for a in payload.get("assertions", []):
        m = _WS_RE.match(a.get("name", ""))
        if m and a.get("kind") == "oracle":
            out.append((int(m.group(1)), m.group(2), a))
    return out


def _crossed(book: Dict[str, List[List[int]]]) -> bool:
    bids = [int(e[0]) for e in book.get("BUY", []) if int(e[1]) > 0]
    asks = [int(e[0]) for e in book.get("SELL", []) if int(e[1]) > 0]
    return bool(bids and asks and max(bids) >= min(asks))


def draw_problems(seed: int, label: str, a: Dict[str, Any]) -> List[str]:
    """Compare one emitted per-draw assertion against the independent derivation."""
    tag = f"seed {seed}"
    out: List[str] = []
    try:
        d = _derive(seed)
    except AssertionError as exc:
        return [f"{tag}: draw does not meet the C09 dataset precondition ({exc})"]
    if label != d["label"]:
        out.append(f"{tag}: class label {label!r} != derived {d['label']!r}")
    if a.get("status") != "pass":
        out.append(f"{tag}: assertion status {a.get('status')!r}")
    try:
        obs, exp = _parse(a.get("observed")), _parse(a.get("expected"))
    except (TypeError, ValueError) as exc:
        return out + [f"{tag}: observed/expected not parseable JSON ({exc})"]
    if not isinstance(obs, dict):
        return out + [f"{tag}: observed missing"]
    missing = [k for k in _REQUIRED_OBS_KEYS if k not in obs]
    if missing:
        return out + [f"{tag}: observed lacks {missing} (per-draw values not present in full)"]

    trades = [list(map(int, t)) for t in obs["trades"]]
    seq = [int(p) for p in obs["price_sequence"]]
    # Pass criterion 1: exact integer-tick price sequence (§4.2 ICEwPI, sign §13.6).
    if seq != d["price_sequence"]:
        out.append(f"{tag}: price sequence {seq} != derived {d['price_sequence']} (§4.2 base + k*ppd)")
    # Pass criterion 2: exact stop index (§4.2.1 Rule 2).
    if obs["stop_index"] != d["m"] or isinstance(obs["stop_index"], bool):
        out.append(f"{tag}: stop_index {obs['stop_index']!r} != derived m = {d['m']} (§4.2.1 Rule 2)")
    # The reported sequence must be what the trade rows actually say.
    if seq != [t[2] for t in trades]:
        out.append(f"{tag}: price_sequence {seq} disagrees with the trade rows' prices")
    if obs["stop_index"] != len(trades):
        out.append(f"{tag}: stop_index {obs['stop_index']} != number of slices traded {len(trades)}")
    if trades != d["trades"]:
        out.append(f"{tag}: trades {trades} != derived {d['trades']}")
    # §6.2 item 7: every price is the iceberg's own slice price, never anything else.
    for k, t in enumerate(trades):
        if t[2] != d["base"] + k * d["ppd"]:
            out.append(f"{tag}: trade {k} at {t[2]} is not the iceberg's slice price "
                       f"{d['base'] + k * d['ppd']} (§6.2 item 7)")
            break
    if any(t[4] != 1 for t in trades):
        out.append(f"{tag}: a row not written by the aggressor's own submission (Rule 2, one submit)")
    for key in ("accepted", "rejected", "lost", "filled", "remaining"):
        if obs[key] != d[key]:
            out.append(f"{tag}: {key} {obs[key]} != derived {d[key]}")
    fb = _book_key(obs["final_book"])
    if fb not in d["final_books"]:
        out.append(f"{tag}: final book {fb} not in derived {d['final_books']} "
                   f"(slice m at base + m*ppd = {d['slice_m_price']}, residual at L)")
    if _crossed(obs["final_book"]):
        out.append(f"{tag}: final book crossed (I3)")
    # Iceberg left resting with POSITIVE hidden quantity.
    ice_vis = sum(int(e[1]) for e in fb[SIDE_NAME[d["ice_side"]]] if int(e[2]) == d["ice_oid"])
    ice_rem = obs["remaining"].get(str(d["ice_oid"])) if isinstance(obs["remaining"], dict) else None
    if not isinstance(ice_rem, int) or ice_rem - ice_vis <= 0:
        out.append(f"{tag}: iceberg hidden quantity {None if ice_rem is None else ice_rem - ice_vis} "
                   f"is not positive (iceberg must be left resting with hidden quantity)")
    if obs["hidden_positive"] is not True:
        out.append(f"{tag}: hidden_positive is {obs['hidden_positive']!r}")

    # Oracle drift: K2's own expectation must agree with this derivation, and admit
    # nothing the rules do not.
    if isinstance(exp, dict):
        if [int(p) for p in exp.get("price_sequence", [])] != d["price_sequence"]:
            out.append(f"{tag}: K2 expected price sequence disagrees with the audit derivation")
        if exp.get("stop_index") != d["m"]:
            out.append(f"{tag}: K2 expected stop index disagrees with the audit derivation")
        if [list(map(int, t)) for t in exp.get("trades", [])] != d["trades"]:
            out.append(f"{tag}: K2 expected trades disagree with the audit derivation")
        k2 = [_book_key(b) for b in exp.get("final_book_one_of", [])]
        if not k2 or any(b not in d["final_books"] for b in k2):
            out.append(f"{tag}: K2 admits a final book the rules do not: {k2}")
        if exp.get("hidden_positive") is not True:
            out.append(f"{tag}: K2 expectation does not require positive hidden quantity")
    else:
        out.append(f"{tag}: K2 expectation missing")
    return out


def all_draw_problems(payload) -> List[str]:
    out = []
    for seed, label, a in _per_draw(payload):
        out += draw_problems(seed, label, a)
    return out


def coverage_problems(payload) -> List[str]:
    out = []
    rows = _per_draw(payload)
    seeds = [s for s, _, _ in rows]
    draws, sd = payload.get("draws", {}), payload.get("seeds", {})
    if not rows:
        return ["no per-draw assertions"]
    ws_seeds = [s for s, _, _ in _ws_draws(payload)]
    all_seeds = seeds + ws_seeds
    if len(all_seeds) != len(set(all_seeds)):
        out.append("duplicate per-draw seeds (within or across the two constructions)")
    if len(seeds) < N_MAIN:
        out.append(f"{len(seeds)} main per-draw assertions < {N_MAIN} (random scenario default)")
    # draws.* and seeds.* cover the main draws plus the wrong-sign draws, nothing else.
    if len(all_seeds) != draws.get("valid"):
        out.append(f"per-draw assertions {len(seeds)} main + {len(ws_seeds)} wrong-sign "
                   f"vs draws.valid {draws.get('valid')}")
    if draws.get("requested") != draws.get("valid", 0) + draws.get("discarded", 0):
        out.append("draws: requested != valid + discarded")
    if sd.get("count") != len(all_seeds):
        out.append(f"seeds.count {sd.get('count')} != main + wrong-sign draws {len(all_seeds)}")
    if sd.get("first") != min(all_seeds) or sd.get("last") != max(all_seeds):
        out.append("seeds.first/last disagree with the emitted draws")
    if sd.get("failing"):
        out.append(f"failing seeds {sd.get('failing')}")
    labels = {_derive(s)["label"] for s in seeds if _safe_derive(s)}
    missing = [c for c in ALL_CLASSES if c not in labels]
    if missing:
        out.append(f"classes never drawn: {missing}")
    # Non-vacuity of the drift itself: per class, at least one draw shows slice m at
    # its drifted price (otherwise the m = 1 class never observes a drifted price).
    shown = {c: 0 for c in ALL_CLASSES}
    for s, label, a in rows:
        d = _safe_derive(s)
        obs = _parse(a.get("observed"))
        if d and isinstance(obs, dict) and "final_book" in obs:
            if _book_key(obs["final_book"]) == d["final_books"][0]:
                shown[d["label"]] = shown.get(d["label"], 0) + 1
    if any(v == 0 for v in shown.values()):
        out.append(f"a class never shows slice m at base + m*ppd (drift unobserved): {shown}")
    return out


def _safe_derive(seed: int) -> Optional[Dict[str, Any]]:
    try:
        return _derive(seed)
    except AssertionError:
        return None


def eager_jit_problems(payload) -> List[str]:
    out = []
    ej = [a for a in payload.get("assertions", []) if a.get("kind") == "eager_jit"]
    if payload.get("eager_jit_identical") is not True:
        out.append(f"eager_jit_identical is {payload.get('eager_jit_identical')!r}")
    if not ej:
        return out + ["no eager-vs-jitted evidence (absence = failure)"]
    valid = {s: label for s, label, _ in _per_draw(payload) + _ws_draws(payload)}
    covered = set()
    for a in ej:
        if a.get("status") != "pass" or a.get("observed") is not True or a.get("expected") is not True:
            out.append(f"eager/jit disagreement: {a.get('name')}")
        m = _EJ_RE.match(a.get("name", ""))
        if not m:
            out.append(f"eager/jit row without a seed: {a.get('name')}")
            continue
        s = int(m.group(1))
        if s not in valid:
            out.append(f"eager/jit seed {s} is not a valid draw of the run")
        else:
            covered.add(valid[s])
    missing = [c for c in ALL_CLASSES + WS_CLASSES if c not in covered]
    if missing:
        out.append(f"no eager-vs-jitted evidence for class(es) {missing}")
    return out


def battery_problems(payload) -> List[str]:
    out = []
    rows = {r.get("check_id"): r for r in payload.get("invariants", [])}
    valid = payload.get("draws", {}).get("valid")
    n_main, n_ws = len(_per_draw(payload)), len(_ws_draws(payload))
    for cid in BATTERY:
        r = rows.get(cid)
        if r is None:
            out.append(f"battery row {cid} missing")
            continue
        if r.get("status") != "pass" or r.get("n_fail") != 0 or r.get("failing_seeds"):
            out.append(f"{cid}: status {r.get('status')}, n_fail {r.get('n_fail')}")
        n_pass, n_na = r.get("n_pass", 0), r.get("n_not_applicable", 0)
        # Every valid draw of both constructions is accounted for, exactly once.
        if n_pass + n_na != valid:
            out.append(f"{cid}: n_pass {n_pass} + n_not_applicable {n_na} != valid draws {valid}")
        # Every MAIN draw is actually checked (trading draws are never vacuous); only the
        # rejected wrong-sign draws (no trade) may be not-applicable.
        if n_pass < n_main or n_na > n_ws:
            out.append(f"{cid}: n_pass {n_pass} < main draws {n_main} or n/a {n_na} > wrong-sign draws {n_ws}")
    for prefix in ("I3 holds", "I5 holds"):
        hit = [a for a in payload.get("assertions", []) if a.get("name", "").startswith(prefix)]
        if not hit or any(a.get("status") != "pass" for a in hit):
            out.append(f"'{prefix} on every valid draw' assertion missing or failed")
    return out


def capacity_problems(payload) -> List[str]:
    out = []
    cap = payload.get("capacity", {})
    if cap.get("envelope") != ENVELOPE:
        out.append(f"capacity envelope {cap.get('envelope')} != {ENVELOPE}")
    for axis, lim in ENVELOPE.items():
        v = cap.get("peaks", {}).get(axis)
        if v is not None and v > lim:
            out.append(f"capacity peak {axis}={v} > {lim}")
    rows = _per_draw(payload)
    if not rows:
        return out + ["no per-draw rows to bound"]
    drawn = [_draw(MAIN_KEY, s) for s, _, _ in rows] + [_draw(WRONG_KEY, s) for s, _, _ in _ws_draws(payload)]
    max_slices = max(_ceil(int(r[C_QTY]), int(r[C_PEAK]))
                     for d in drawn for r in _real(d.tape) if int(r[C_EXEC]) == ICE)
    hop = max(int(d.hop_bound) for d in drawn)
    max_trades = max(_derive(s)["m"] for s, _, _ in rows if _safe_derive(s))
    if max_slices > SLICE_MARGIN:
        out.append(f"an iceberg has ceil(total/peak) = {max_slices} > {SLICE_MARGIN}")
    if hop > ENVELOPE["hops"]:
        out.append(f"analytic hop bound {hop} > 160")
    if max_trades > ENVELOPE["trades"]:
        out.append(f"{max_trades} trades > 80")
    by_name = {a.get("name", ""): a for a in payload.get("assertions", [])}
    sl = [a for n, a in by_name.items() if n.startswith("largest ceil(qty/peak)")]
    if not sl or sl[0].get("status") != "pass" or sl[0].get("observed") != max_slices:
        out.append(f"slice-margin assertion absent/failed or reports {sl[0].get('observed') if sl else None}"
                   f" != re-derived {max_slices}")
    hb = [a for n, a in by_name.items() if n.startswith("hop bound <=")]
    if not hb or hb[0].get("status") != "pass" or not isinstance(hb[0].get("observed"), list) \
            or max(hb[0]["observed"]) > ENVELOPE["hops"] or hb[0]["observed"][0] != hop:
        out.append(f"hop-bound assertion absent/failed or disagrees with re-derived {hop}")
    cf = [a for n, a in by_name.items() if n.startswith("no capacity flag")]
    if not cf or cf[0].get("status") != "pass":
        out.append("no-capacity-flag (truncated/book_full/ring) assertion absent or failed")
    if cap.get("peaks", {}).get("trades") is not None and cap["peaks"]["trades"] != max_trades:
        out.append(f"capacity.peaks.trades {cap['peaks']['trades']} != re-derived {max_trades}")
    return out


# --- the wrong-sign draws: asserted rejections (§5.4 v2.14) ---------------------------
def ws_draw_problems(seed: int, label: str, a: Dict[str, Any]) -> List[str]:
    """One wrong-sign draw against the independent §5.4 derivation (observed AND K2 expected)."""
    tag = f"wrong-sign seed {seed}"
    try:
        d = _derive_wrong(seed)
    except AssertionError as exc:
        return [f"{tag}: draw does not meet the wrong-sign dataset precondition ({exc})"]
    out: List[str] = []
    if label != d["label"]:
        out.append(f"{tag}: class label {label!r} != derived {d['label']!r}")
    name = a.get("name", "")
    if "REJECTED" not in name or "OBSERVATION" in name or "NOT asserted" in name:
        out.append(f"{tag}: not labelled as an asserted rejection: {name!r}")
    if a.get("status") != "pass":
        out.append(f"{tag}: assertion status {a.get('status')!r}")
    if "§5.4" not in str(a.get("rule", "")):
        out.append(f"{tag}: rule does not cite §5.4 (the v2.14 sign rejection)")
    try:
        obs, exp = _parse(a.get("observed")), _parse(a.get("expected"))
    except (TypeError, ValueError) as exc:
        return out + [f"{tag}: observed/expected not parseable JSON ({exc})"]
    for which, rec in (("observed", obs), ("K2 expected", exp)):
        if not isinstance(rec, dict):
            out.append(f"{tag}: {which} missing or not a record (an un-asserted observation?)")
            continue
        missing = [k for k in _REQUIRED_WS_KEYS if k not in rec]
        if missing:
            out.append(f"{tag}: {which} lacks {missing} (not present in full)")
            continue
        out += [f"{tag}: {which} {p}" for p in _ws_record_problems(rec, d)]
    return out


def _ws_record_problems(rec: Dict[str, Any], d: Dict[str, Any]) -> List[str]:
    out = []
    # Rejected at entry: the iceberg (submission 0) is rejected, only the aggressor accepted.
    rej = rec["rejected"]
    if not (isinstance(rej, list) and len(rej) == 1 and isinstance(rej[0], list) and len(rej[0]) >= 2
            and rej[0][0] == 0 and rej[0][1] == d["rejected_oid"]):
        out.append(f"rejected {rej!r}: the iceberg (submission 0, oid {d['rejected_oid']}) is not "
                   f"the one rejected order (§5.4 v2.14)")
    elif len(rej[0]) > 2 and not (isinstance(rej[0][2], str) and rej[0][2]):
        out.append(f"rejected {rej!r}: rejection reason empty")
    if rec["accepted"] != d["accepted"]:
        out.append(f"accepted {rec['accepted']} != {d['accepted']} (only the aggressor enters the SOB)")
    # Zero trades, zero fills, nothing lost.
    if rec["trades"] != []:
        out.append(f"trades {rec['trades']} != [] (a rejected iceberg cannot trade)")
    for key in ("filled", "remaining", "lost"):
        if rec[key] != d[key]:
            out.append(f"{key} {rec[key]} != derived {d[key]}")
    if rec["ice_hidden"] != 0 or isinstance(rec["ice_hidden"], bool):
        out.append(f"ice_hidden {rec['ice_hidden']!r} != 0 (a rejected iceberg holds nothing)")
    # Book unchanged by the rejected submission.
    if rec["book_unchanged_by_submission_0"] != []:
        out.append(f"book changed by submission 0: {rec['book_unchanged_by_submission_0']!r}")
    # Aggressor rests whole at L; the iceberg's side is empty.
    fb = rec["final_book"]
    if not isinstance(fb, dict) or _book_key(fb) != d["final_book"]:
        out.append(f"final book {fb} != derived {d['final_book']} (aggressor whole at L, iceberg absent)")
    elif _crossed(fb):
        out.append("final book crossed (I3)")
    return out


def wrong_sign_problems(payload) -> List[str]:
    """Every wrong-sign draw is present, labelled and ASSERTED as a §5.4 rejection."""
    rows = _ws_draws(payload)
    if not rows:
        return ["wrong-sign draws absent"]
    out = []
    names = [a.get("name", "") for a in payload.get("assertions", [])]
    leftover = [n for n in names if "OBSERVATION" in n or "NOT asserted" in n or n.startswith("wrong-sign draw")]
    if leftover:
        out.append(f"an un-asserted wrong-sign observation is still emitted: {leftover}")
    for a in payload.get("assertions", []):
        if "none is asserted" in str(a.get("expected", "")):
            out.append(f"an assertion pins no outcome ('none is asserted'): {a.get('name')}")
    if len(rows) != N_WRONG:
        out.append(f"{len(rows)} wrong-sign draws != {N_WRONG} (card)")
    main_seeds = {s for s, _, _ in _per_draw(payload)}
    if main_seeds & {s for s, _, _ in rows}:
        out.append("a wrong-sign seed also appears as a main draw")
    seen = {_derive_wrong(s)["label"] for s, _, _ in rows if _safe_derive_wrong(s)}
    missing = [c for c in WS_CLASSES if c not in seen]
    if missing:
        out.append(f"wrong-sign side(s) never drawn: {missing}")
    for seed, label, a in rows:
        out += ws_draw_problems(seed, label, a)
    reached = [a for a in payload.get("assertions", [])
               if a.get("name", "").startswith(f"{WRONG_KEY}: ") and "valid draws reached" in a.get("name", "")]
    if not reached or reached[0].get("status") != "pass" or reached[0].get("observed") != len(rows):
        out.append(f"'{WRONG_KEY}: N valid draws reached' assertion absent, failed, or != {len(rows)}")
    return out


# --- quotes and citations -------------------------------------------------------------
REQUIRED_QUOTES = (
    ("§13.6", "Only a sign constraint is authoritative."),
    ("§13.6", "ICEwPI buy orders must have `PPD < 0`; ICEwPI sell orders must have `PPD > 0`."),
    ("§13.6", "`PPD = -3.00 EUR/MWh`"),  # ASCII hyphen, byte-copied
    ("§6.2 item 7", "the iceberg is always the price-setter"),
    ("§4.2.1", "the regenerated slice's price no longer crosses the aggressive order's limit"),
    # r2: the v2.14 rejection the wrong-sign draws certify
    ("§5.4", "An iceberg with a non-zero price increment (PPD) of the wrong sign is rejected"),
)
_STATUS_RE = re.compile(r"^\*\*Status:\*\*\s*(v\d+(?:\.\d+)*)", re.M)


def _quote_candidates(source: str, anchor: str) -> List[str]:
    text = (REPO_ROOT / "docs" / "omiecid" / source).read_text(encoding="utf-8")
    out = []
    for line in text.split("\n"):
        if anchor not in line:
            continue
        body = re.sub(r"^(\s*- |\s*\d+\. |> ?)", "", line).strip()
        sents = re.split(r"(?<=[.:;)\"*])\s+", body)  # '*': split after a '*(Added vX)*' tag
        out.append(body)
        for a in range(len(sents)):
            for b in range(a + 1, len(sents) + 1):
                c = " ".join(sents[a:b])
                if anchor in c:
                    out.append(c)
    return out


def quote_problems(payload) -> List[str]:
    out = []
    quotes = payload.get("spec", {}).get("quotes", [])
    if not quotes:
        return ["no quotes"]
    bad = [q for q in quotes if not q.get("ok")]
    if bad:
        out.append(f"{len(bad)} quote(s) drifted: {[(q['source'], q['section']) for q in bad]}")
    ms_text = (REPO_ROOT / "docs" / "omiecid" / "market-spec.md").read_text(encoding="utf-8")
    ms_version = _STATUS_RE.search(ms_text).group(1)
    if ms_version != SPEC_VERSION:
        out.append(f"market-spec.md is {ms_version}; this r2 twin is pinned to {SPEC_VERSION}")
    stale = [q["section"] for q in quotes if q.get("source") == "market-spec.md" and q.get("version") != SPEC_VERSION]
    if stale:
        out.append(f"market-spec quote(s) not at version {SPEC_VERSION}: {stale}")
    digests = {q.get("digest") for q in quotes}
    for section, anchor in REQUIRED_QUOTES:
        hits = [c for c in _quote_candidates("market-spec.md", anchor)
                if quote_digest(SpecQuote("market-spec.md", section, c)) in digests]
        if not hits:
            out.append(f"required passage not quoted byte-exact: {section} '{anchor}'")
            continue
        chk = check_quote(SpecQuote("market-spec.md", section, hits[0]))
        if not chk.ok:
            out.append(f"quoted passage {section} '{anchor}' no longer matches its source: {chk.reason}")
    return out


# REL-11: not redistributable, removed from docs/omiecid/corpus/; citations of these are not checked.
REMOVED_CORPUS_DOCS = ("Continuous_Trading_Matching_Algorithm_Public_Description.md",
                       "Single_Intraday_Coupling_XBID_Information_Package.md")


def _corpus_text(source: str) -> Optional[str]:
    m = re.search(r"(docs/omiecid/corpus/\S+?\.md)", source)
    if not m:
        return None
    p = REPO_ROOT / m.group(1)
    return p.read_text(encoding="utf-8") if p.is_file() else None


def scope_problems(payload) -> List[str]:
    out = []
    for a in payload.get("assertions", []):
        if "iceberg_meta" in a.get("name", "") or "iceberg_meta" in str(a.get("expected", "")):
            out.append(f"asserts on iceberg_meta (must-not-assert, OQ-20/OQ-23): {a.get('name')}")
        if a.get("status") == "fail":
            out.append(f"failed assertion: {a.get('name')}")
    # r2 scope restatement: one iceberg, no competing level; wrong-sign draws REJECTED.
    kr = str(payload.get("known_result", ""))
    if "One iceberg, no competing level" not in kr:
        out.append("known_result does not restate the scope (one iceberg, no competing level)")
    if "REJECTED" not in kr or "§5.4" not in kr:
        out.append("known_result does not state the §5.4 rejection of the wrong-sign draws")
    if re.search(r"observation|not asserted", kr, re.I):
        out.append("known_result still describes the wrong-sign draw as an un-asserted observation")
    return out


def audit_problems(payload) -> List[str]:
    return (all_draw_problems(payload) + coverage_problems(payload) + eager_jit_problems(payload)
            + battery_problems(payload) + capacity_problems(payload) + wrong_sign_problems(payload)
            + scope_problems(payload))


# ---------------------------------------------------------------------------
# Tests on the emitted JSON
# ---------------------------------------------------------------------------
class TestEmittedResults:
    def test_schema_and_semantic_validation_clean(self, payload):
        assert validate_results(payload) == []
        assert payload["failure_kinds"] == derive_failure_kinds(payload)

    def test_identity_and_oracle_kind(self, payload):
        assert payload["cert_id"] == "C09"
        assert payload["notebook"] == "C09_icewpi_price_drift.ipynb"
        assert payload["oracle_kind"] == "exact-parametric"

    def test_status_pass_and_no_failure_kinds(self, payload):
        assert payload["status"] == "PASS", payload.get("failure_explanation")
        assert payload["failure_kinds"] == []
        assert payload["failure_explanation"] is None

    def test_generator_digests_match_this_code_version(self, payload):
        gen = payload["generator"]
        assert gen["module"] == "omiecid_simulator.jaxomiecid.omie_tape_synth"
        for key in (MAIN_KEY, WRONG_KEY):
            assert gen["digests"].get(key) == synth.construction_digest(key), key

    def test_every_valid_draw_has_a_per_draw_exact_assertion_and_every_class_is_covered(self, payload):
        assert coverage_problems(payload) == []

    def test_no_draw_discarded_inside_the_margin(self, payload):
        d = payload["draws"]
        assert d["discarded"] == 0 and d["discard_rate"] == 0.0 and d["valid"] == d["requested"]

    def test_every_main_draw_is_sign_valid_and_meets_the_dataset_precondition(self, payload):
        for seed, _, _ in _per_draw(payload):
            real = _real(_draw(MAIN_KEY, seed).tape)
            assert _sign_valid(int(real[0][C_SIDE]), int(real[0][C_PPD])), seed
            assert _safe_derive(seed) is not None, seed

    def test_per_draw_price_sequence_and_stop_index_equal_independent_derivation(self, payload):
        problems = all_draw_problems(payload)
        assert problems == [], "\n".join(problems[:20])

    def test_price_sequence_drifts_by_exactly_ppd_in_the_sign_rule_direction(self, payload):
        """BUY icebergs walk down, SELL icebergs walk up, by exactly |ppd| per slice."""
        for seed, _, a in _per_draw(payload):
            d = _derive(seed)
            seq = _parse(a["observed"])["price_sequence"]
            assert seq[0] == d["base"], seed
            steps = {b - a_ for a_, b in zip(seq, seq[1:])}
            assert steps <= {d["ppd"]}, (seed, steps)
            if d["ice_side"] == BUY:
                assert all(b < a_ for a_, b in zip(seq, seq[1:])), seed
            else:
                assert all(b > a_ for a_, b in zip(seq, seq[1:])), seed

    def test_stop_slice_is_the_first_non_crossing_drifted_slice(self, payload):
        """Last traded slice crosses L (here: at L exactly), slice m does not."""
        for seed, _, a in _per_draw(payload):
            d = _derive(seed)
            m = _parse(a["observed"])["stop_index"]
            assert _crosses(d["agg_side"], d["L"], d["base"] + (m - 1) * d["ppd"]), seed
            assert not _crosses(d["agg_side"], d["L"], d["base"] + m * d["ppd"]), seed

    def test_iceberg_left_resting_with_positive_hidden_quantity(self, payload):
        for seed, _, a in _per_draw(payload):
            obs = _parse(a["observed"])
            d = _derive(seed)
            vis = sum(e[1] for e in obs["final_book"].get(SIDE_NAME[d["ice_side"]], []) if e[2] == d["ice_oid"])
            assert obs["remaining"][str(d["ice_oid"])] - vis > 0, seed
            assert obs["hidden_positive"] is True, seed

    def test_sign_error_is_discriminated_on_every_draw(self, payload):
        """Non-vacuity: a wrong-way drift prints a different trade list on every draw."""
        for seed, _, _ in _per_draw(payload):
            assert _derive_flipped(seed) != _derive(seed)["trades"], seed

    def test_final_book_never_crossed_on_any_draw(self, payload):
        for seed, _, a in _per_draw(payload):
            assert not _crossed(_parse(a["observed"])["final_book"]), seed

    def test_eager_and_jitted_evidence_present_and_identical(self, payload):
        assert eager_jit_problems(payload) == []

    def test_full_battery_ran_and_passed_on_every_main_draw(self, payload):
        assert battery_problems(payload) == []

    def test_capacity_margins_and_no_capacity_flag(self, payload):
        assert capacity_problems(payload) == []

    def test_every_wrong_sign_draw_is_an_asserted_rejection(self, payload):
        """§5.4 v2.14: rejected at entry, zero trades, book unchanged, aggressor rests whole."""
        probs = wrong_sign_problems(payload)
        assert probs == [], "\n".join(probs[:20])

    def test_wrong_sign_draws_are_sign_invalid_non_vacuous_and_cover_both_sides(self, payload):
        """Each wrong-sign draw violates §13.6 and WOULD trade if the sign rule were ignored."""
        rows = _ws_draws(payload)
        assert len(rows) == N_WRONG
        sides = set()
        for seed, _, _ in rows:
            d = _derive_wrong(seed)
            assert not _sign_valid(d["ice_side"], d["ppd"]), seed
            assert d["literal_trades"] and _crosses(d["agg_side"], d["L"], d["base"]), seed
            sides.add(d["ice_side"])
        assert sides == {BUY, SELL}

    def test_wrong_sign_aggressor_rests_whole_at_its_limit(self, payload):
        for seed, _, a in _ws_draws(payload):
            d, obs = _derive_wrong(seed), _parse(a["observed"])
            agg = obs["final_book"][SIDE_NAME[d["agg_side"]]]
            assert agg == [[d["L"], d["q"], d["agg_oid"]]], seed
            assert obs["final_book"][SIDE_NAME[d["ice_side"]]] == [], seed
            assert obs["trades"] == [] and obs["filled"][str(d["agg_oid"])] == 0, seed

    def test_scope_one_iceberg_no_competing_level_on_every_draw(self, payload):
        """r2 scope: every draw of both constructions is one iceberg + one NON aggressor."""
        for key, rows in ((MAIN_KEY, _per_draw(payload)), (WRONG_KEY, _ws_draws(payload))):
            for seed, _, _ in rows:
                real = _real(_draw(key, seed).tape)
                assert len(real) == 2 and sum(int(r[C_EXEC]) == ICE for r in real) == 1, (key, seed)
        assert scope_problems(payload) == []

    def test_required_quotes_present_and_byte_exact(self, payload):
        assert quote_problems(payload) == []

    def test_regulatory_sign_rule_citations_are_verbatim_in_the_corpus(self, payload):
        cits = payload.get("citations", [])
        assert cits, "no regulatory citation of the ICEwPI sign rule"
        for c in cits:
            if any(doc in c["source"] for doc in REMOVED_CORPUS_DOCS):
                continue
            text = _corpus_text(c["source"])
            assert text is not None, c["source"]
            assert normalise(c["text"]) in normalise(text), c["source"]
        assert any("negative price increments" in c["text"] for c in cits)

    def test_no_iceberg_meta_claim_and_no_failed_assertion(self, payload):
        assert scope_problems(payload) == []


# ---------------------------------------------------------------------------
# Deliberately broken copies must be rejected
# ---------------------------------------------------------------------------
def _first(payload, pred=lambda d: True) -> int:
    for i, a in enumerate(payload["assertions"]):
        m = _PER_DRAW_RE.match(a.get("name", ""))
        if m and a.get("kind") == "oracle" and pred(_derive(int(m.group(1)))):
            return i
    pytest.skip("no draw matching the predicate in this run")


def _seed_of(a) -> int:
    return int(_PER_DRAW_RE.match(a["name"]).group(1))


def _edit(a: Dict[str, Any], fn) -> None:
    """Apply fn to both observed and expected, so the K2 expectation cannot save it."""
    for key in ("observed", "expected"):
        if isinstance(a.get(key), str):
            obj = json.loads(a[key])
            fn(obj)
            a[key] = json.dumps(obj)


def _books_of(o) -> List[Dict[str, List[List[int]]]]:
    return [o["final_book"]] if "final_book" in o else o.get("final_book_one_of", [])


class TestBrokenCopiesRejected:
    def test_sanity_real_payload_is_clean(self, payload):
        assert audit_problems(payload) == []

    def test_sign_flipped_price_ladder_rejected(self, payload, tmp_path):
        """Consistent volumes, every price drifted the wrong way (incl. the resting slice)."""
        bad = copy.deepcopy(payload)
        i = _first(bad, lambda d: d["m"] >= 2)
        d = _derive(_seed_of(bad["assertions"][i]))

        def flip(o):
            for k, t in enumerate(o["trades"]):
                t[2] = d["base"] - k * d["ppd"]
            o["price_sequence"] = [d["base"] - k * d["ppd"] for k in range(len(o["price_sequence"]))]
            for b in _books_of(o):
                for e in b.get(SIDE_NAME[d["ice_side"]], []):
                    e[0] = d["base"] - d["m"] * d["ppd"]
        _edit(bad["assertions"][i], flip)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("price sequence" in p for p in probs), probs
        assert any("§6.2 item 7" in p for p in probs), probs

    def test_sign_flipped_resting_slice_on_m1_draw_rejected(self, payload, tmp_path):
        """m = 1: the only drifted price is the resting slice; flipping it must be caught."""
        bad = copy.deepcopy(payload)
        i = _first(bad, lambda d: d["m"] == 1)
        d = _derive(_seed_of(bad["assertions"][i]))

        def flip_rest(o):
            for b in _books_of(o):
                for e in b.get(SIDE_NAME[d["ice_side"]], []):
                    e[0] = d["base"] - d["ppd"]
        _edit(bad["assertions"][i], flip_rest)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("final book" in p for p in probs), probs

    def test_off_by_one_stop_index_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        i = _first(bad)

        def plus_one(o):
            o["stop_index"] = o["stop_index"] + 1
        _edit(bad["assertions"][i], plus_one)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("stop_index" in p for p in probs), probs

    def test_off_by_one_overmatch_fully_consistent_rejected(self, payload, tmp_path):
        """An engine that also trades the first NON-crossing slice m, reported consistently."""
        bad = copy.deepcopy(payload)
        i = _first(bad)
        d = _derive(_seed_of(bad["assertions"][i]))
        extra = min(d["slice_m_size"], d["q_rem"])

        def overmatch(o):
            p = d["base"] + d["m"] * d["ppd"]
            o["trades"].append([o["trades"][0][0], o["trades"][0][1], p, extra, 1])
            o["price_sequence"].append(p)
            o["stop_index"] = d["m"] + 1
            for oid in o["filled"]:
                o["filled"][oid] += extra
            for oid in o["remaining"]:
                o["remaining"][oid] -= extra
            for b in _books_of(o):
                ice_side = b.get(SIDE_NAME[d["ice_side"]], [])
                for e in ice_side:
                    e[0] = d["base"] + (d["m"] + 1) * d["ppd"]
                for e in b.get(SIDE_NAME[d["agg_side"]], []):
                    e[1] -= extra
        _edit(bad["assertions"][i], overmatch)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("stop_index" in p and "Rule 2" in p for p in probs), probs
        assert any("price sequence" in p for p in probs), probs

    def test_zero_hidden_iceberg_rejected(self, payload, tmp_path):
        """Iceberg reported resting with only its visible slice, no hidden quantity."""
        bad = copy.deepcopy(payload)
        i = _first(bad)
        d = _derive(_seed_of(bad["assertions"][i]))

        def no_hidden(o):
            o["remaining"][str(d["ice_oid"])] = d["slice_m_size"]
            o["hidden_positive"] = False
        _edit(bad["assertions"][i], no_hidden)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("hidden quantity" in p and "not positive" in p for p in probs), probs

    def test_iceberg_exhausted_at_stop_rejected(self, payload, tmp_path):
        """Iceberg gone from the book with nothing remaining: 'exhausted liquidity' misread."""
        bad = copy.deepcopy(payload)
        i = _first(bad)
        d = _derive(_seed_of(bad["assertions"][i]))
        a = bad["assertions"][i]
        obs = json.loads(a["observed"])
        obs["remaining"][str(d["ice_oid"])] = 0
        obs["final_book"][SIDE_NAME[d["ice_side"]]] = []
        a["observed"] = json.dumps(obs)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("not positive" in p for p in probs), probs

    def test_trade_priced_at_aggressor_limit_rejected(self, payload, tmp_path):
        """§6.2 item 7: printing slice 0 at L instead of base (m >= 2, so L != base)."""
        bad = copy.deepcopy(payload)
        i = _first(bad, lambda d: d["m"] >= 2)
        d = _derive(_seed_of(bad["assertions"][i]))

        def at_L(o):
            o["trades"][0][2] = d["L"]
            o["price_sequence"][0] = d["L"]
        _edit(bad["assertions"][i], at_L)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("§6.2 item 7" in p for p in probs), probs

    def test_truncated_per_draw_observation_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        i = _first(bad)
        obs = json.loads(bad["assertions"][i]["observed"])
        del obs["price_sequence"]
        bad["assertions"][i]["observed"] = json.dumps(obs)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("not present in full" in p for p in probs), probs

    def test_dropped_draw_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        del bad["assertions"][_first(bad)]
        assert coverage_problems(_roundtrip(tmp_path, bad)) != []

    def test_missing_eager_jit_evidence_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        bad["assertions"] = [a for a in bad["assertions"] if a.get("kind") != "eager_jit"]
        probs = eager_jit_problems(_roundtrip(tmp_path, bad))
        assert any("absence" in p for p in probs), probs

    def test_eager_jit_missing_for_one_class_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        ej = [a for a in bad["assertions"] if a.get("kind") == "eager_jit"]
        valid = {s: lab for s, lab, _ in _per_draw(bad)}
        target = valid[int(_EJ_RE.match(ej[0]["name"]).group(1))]
        bad["assertions"] = [a for a in bad["assertions"] if not (
            a.get("kind") == "eager_jit" and valid.get(int(_EJ_RE.match(a["name"]).group(1))) == target)]
        probs = eager_jit_problems(_roundtrip(tmp_path, bad))
        assert any(target in p for p in probs), probs

    def test_eager_jit_disagreement_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        for a in bad["assertions"]:
            if a.get("kind") == "eager_jit":
                a["observed"] = False
                break
        probs = eager_jit_problems(_roundtrip(tmp_path, bad))
        assert any("disagreement" in p for p in probs), probs

    def test_missing_wrong_sign_draws_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        bad["assertions"] = [a for a in bad["assertions"] if not _WS_RE.match(a.get("name", ""))]
        probs = wrong_sign_problems(_roundtrip(tmp_path, bad))
        assert any("absent" in p for p in probs), probs

    def test_one_wrong_sign_draw_dropped_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        i = next(i for i, a in enumerate(bad["assertions"]) if _WS_RE.match(a.get("name", "")))
        del bad["assertions"][i]
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any(f"!= {N_WRONG}" in p for p in probs), probs
        assert any("draws.valid" in p or "seeds.count" in p for p in probs), probs

    def test_wrong_sign_accepted_and_traded_on_the_literal_ladder_rejected(self, payload, tmp_path):
        """The pre-S1 engine: wrong-sign iceberg accepted and walked base + k*ppd, reported
        consistently in BOTH observed and K2 expected, status pass. Must fail the audit."""
        bad = copy.deepcopy(payload)
        seed, _, a = _ws_draws(bad)[0]
        d = _derive_wrong(seed)
        lit = d["literal_trades"]
        filled = sum(t[3] for t in lit)

        def accept(o):
            o["accepted"] = [d["ice_oid"], d["agg_oid"]]
            o["rejected"] = []
            o["trades"] = lit
            o["filled"] = {str(d["ice_oid"]): filled, str(d["agg_oid"]): filled}
            o["remaining"][str(d["agg_oid"])] = d["q"] - filled
        for key in ("observed", "expected"):
            obj = json.loads(a[key])
            accept(obj)
            a[key] = json.dumps(obj)
        probs = wrong_sign_problems(_roundtrip(tmp_path, bad))
        assert any("not the one rejected order" in p for p in probs), probs
        assert any("a rejected iceberg cannot trade" in p for p in probs), probs
        assert any("K2 expected" in p for p in probs), probs

    def test_wrong_sign_rejected_but_book_mutated_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        seed, _, a = _ws_draws(bad)[0]
        obs = json.loads(a["observed"])
        obs["book_unchanged_by_submission_0"] = ["lob.bids[0]"]
        a["observed"] = json.dumps(obs)
        probs = wrong_sign_problems(_roundtrip(tmp_path, bad))
        assert any("book changed by submission 0" in p for p in probs), probs

    def test_wrong_sign_aggressor_not_resting_whole_rejected(self, payload, tmp_path):
        """Aggressor resting short of its full quantity (or off its limit) must be caught."""
        bad = copy.deepcopy(payload)
        seed, _, a = _ws_draws(bad)[0]
        d = _derive_wrong(seed)
        obs = json.loads(a["observed"])
        obs["final_book"][SIDE_NAME[d["agg_side"]]] = [[d["L"], d["q"] - 1, d["agg_oid"]]]
        a["observed"] = json.dumps(obs)
        probs = wrong_sign_problems(_roundtrip(tmp_path, bad))
        assert any("aggressor whole at L" in p for p in probs), probs

    def test_wrong_sign_iceberg_left_resting_rejected(self, payload, tmp_path):
        """A 'rejected' iceberg whose base slice nevertheless sits in the book."""
        bad = copy.deepcopy(payload)
        seed, _, a = _ws_draws(bad)[0]
        d = _derive_wrong(seed)
        obs = json.loads(a["observed"])
        obs["final_book"][SIDE_NAME[d["ice_side"]]] = [[d["base"], 1, d["ice_oid"]]]
        a["observed"] = json.dumps(obs)
        probs = wrong_sign_problems(_roundtrip(tmp_path, bad))
        assert any("iceberg absent" in p for p in probs), probs

    def test_wrong_sign_reverted_to_unasserted_observation_rejected(self, payload, tmp_path):
        """The r1 shape (labelled OBSERVATION, 'none is asserted') is no longer acceptable."""
        bad = copy.deepcopy(payload)
        seed, label, a = _ws_draws(bad)[0]
        a["name"] = f"seed {seed} [{label}]: OBSERVATION, NOT asserted"
        a["expected"] = "an outcome is logged; none is asserted"
        probs = wrong_sign_problems(_roundtrip(tmp_path, bad))
        assert any("not labelled as an asserted rejection" in p for p in probs), probs
        assert any("none is asserted" in p for p in probs), probs

    def test_wrong_sign_failing_status_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        _, _, a = _ws_draws(bad)[0]
        a["status"] = "fail"
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("assertion status 'fail'" in p for p in probs), probs

    def test_wrong_sign_side_mislabelled_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        seed, label, a = _ws_draws(bad)[0]
        other = [c for c in WS_CLASSES if c != label][0]
        a["name"] = a["name"].replace(label, other)
        probs = wrong_sign_problems(_roundtrip(tmp_path, bad))
        assert any("class label" in p for p in probs), probs

    def test_battery_vacuous_on_a_main_draw_rejected(self, payload, tmp_path):
        """A battery row that marks a TRADING main draw not-applicable must be caught."""
        bad = copy.deepcopy(payload)
        for r in bad["invariants"]:
            if r["check_id"] == "I5":
                r["n_pass"] -= 1
                r["n_not_applicable"] += 1
        probs = battery_problems(_roundtrip(tmp_path, bad))
        assert any(p.startswith("I5") and "main draws" in p for p in probs), probs

    def test_battery_draw_double_counted_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        for r in bad["invariants"]:
            r["n_pass"] += 1
        assert battery_problems(_roundtrip(tmp_path, bad)) != []

    def test_stale_market_spec_quote_version_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        for q in bad["spec"]["quotes"]:
            if q.get("source") == "market-spec.md" and q.get("section") == "§5.4":
                q["version"] = "v2.13"
        probs = quote_problems(_roundtrip(tmp_path, bad))
        assert any("not at version v2.14" in p for p in probs), probs

    def test_scope_statement_dropped_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        bad["known_result"] = bad["known_result"].replace("One iceberg, no competing level. ", "")
        probs = scope_problems(_roundtrip(tmp_path, bad))
        assert any("restate the scope" in p for p in probs), probs

    def test_I5_battery_failure_on_a_main_draw_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        for r in bad["invariants"]:
            if r["check_id"] == "I5":
                r.update(status="fail", n_pass=r["n_pass"] - 1, n_fail=1, failing_seeds=[0])
        assert any(p.startswith("I5") for p in battery_problems(_roundtrip(tmp_path, bad)))

    def test_K2_expectation_admitting_a_wrong_book_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        i = _first(bad)
        d = _derive(_seed_of(bad["assertions"][i]))
        a = bad["assertions"][i]
        exp = json.loads(a["expected"])
        exp["final_book_one_of"].append({SIDE_NAME[d["ice_side"]]: [[d["base"], d["peak"], d["ice_oid"]]],
                                         SIDE_NAME[d["agg_side"]]: []})
        a["expected"] = json.dumps(exp)
        probs = all_draw_problems(_roundtrip(tmp_path, bad))
        assert any("K2 admits" in p for p in probs), probs

    def test_minus_sign_counter_example_quote_rejected(self, payload, tmp_path):
        """§13.6's counter-example is written with an ASCII hyphen; a U+2212 copy drifts."""
        bad = copy.deepcopy(payload)
        ascii_c = [c for c in _quote_candidates("market-spec.md", "`PPD = -3.00 EUR/MWh`")
                   if quote_digest(SpecQuote("market-spec.md", "§13.6", c))
                   in {q["digest"] for q in bad["spec"]["quotes"]}]
        assert ascii_c
        good = quote_digest(SpecQuote("market-spec.md", "§13.6", ascii_c[0]))
        minus = quote_digest(SpecQuote("market-spec.md", "§13.6", ascii_c[0].replace("-3.00", "−3.00")))
        for q in bad["spec"]["quotes"]:
            if q["digest"] == good:
                q["digest"] = minus
        assert any("-3.00" in p for p in quote_problems(_roundtrip(tmp_path, bad)))


# ---------------------------------------------------------------------------
# Slow: re-run a few seeds through the engine, eager and jitted
# ---------------------------------------------------------------------------
SLOW_SEEDS = (0, 1, 2, 3)


@pytest.mark.jit
@pytest.mark.slow
@pytest.mark.parametrize("seed", SLOW_SEEDS)
def test_engine_rerun_matches_derivation_and_emitted_json(seed, payload):
    """slow: compiles and runs the engine eager + jitted (tens of seconds per shape)."""
    from tests.omiecid.reference import omie_invariants as inv

    draw = _draw(MAIN_KEY, seed)
    jit_rec = inv.record_from_draw(draw, jit=True)
    eager_rec = inv.record_from_draw(draw, jit=False)
    assert inv.compare_records(jit_rec, eager_rec) == ()

    trades = [[f.buyer_oid, f.seller_oid, f.price_int, f.qty_int, f.submission] for f in jit_rec.fills]
    book = {"BUY": [[p, q, o] for p, q, o, *_ in jit_rec.final_snapshot.resting(BUY)],
            "SELL": [[p, q, o] for p, q, o, *_ in jit_rec.final_snapshot.resting(SELL)]}
    d = _derive(seed)
    assert trades == d["trades"]
    assert [t[2] for t in trades] == d["price_sequence"] and len(trades) == d["m"]
    assert _book_key(book) in d["final_books"]
    for snap in jit_rec.post_snapshots:
        bb, ba = snap.best_bid(), snap.best_ask()
        assert bb is None or ba is None or bb < ba
        assert not snap.truncated and not snap.book_full
    report = inv.run_battery(jit_rec, expects_trade=True)
    assert report.passed and report.valid, report.statuses()

    rows = {s: a for s, _, a in _per_draw(payload)}
    assert seed in rows, f"seed {seed} absent from the emitted JSON"
    emitted = _parse(rows[seed]["observed"])
    assert emitted["trades"] == trades
    assert _book_key(emitted["final_book"]) == _book_key(book)


def _ws_slow_seeds() -> Tuple[int, ...]:
    """One wrong-sign BUY and one wrong-sign SELL seed of the emitted run (both sides)."""
    p = _results_path()
    if not p.is_file():
        return ()
    rows = _ws_draws(_load(p))
    pick = {}
    for seed, label, _ in rows:
        pick.setdefault(label, seed)
    return tuple(sorted(pick.values()))


@pytest.mark.jit
@pytest.mark.slow
@pytest.mark.parametrize("seed", _ws_slow_seeds())
def test_engine_rerun_wrong_sign_draw_is_rejected(seed, payload):
    """slow: compiles and runs the engine eager + jitted on a wrong-sign tape (§5.4 v2.14)."""
    from tests.omiecid.reference import omie_invariants as inv

    d = _derive_wrong(seed)
    draw = _draw(WRONG_KEY, seed)
    jit_rec = inv.record_from_draw(draw, jit=True)
    eager_rec = inv.record_from_draw(draw, jit=False)
    assert inv.compare_records(jit_rec, eager_rec) == ()
    assert [(r.submission, r.oid) for r in jit_rec.rejections] == [(0, d["ice_oid"])]
    assert jit_rec.fills == ()
    assert sorted(jit_rec.accepted) == [d["agg_oid"]]
    book = {"BUY": [[p, q, o] for p, q, o, *_ in jit_rec.final_snapshot.resting(BUY)],
            "SELL": [[p, q, o] for p, q, o, *_ in jit_rec.final_snapshot.resting(SELL)]}
    assert _book_key(book) == d["final_book"]
    emitted = _parse({s: a for s, _, a in _ws_draws(payload)}[seed]["observed"])
    assert emitted["trades"] == [] and _book_key(emitted["final_book"]) == _book_key(book)


# ---------------------------------------------------------------------------
# Independent engine probes (ContractBook public API): §5.4 v2.14 sign rule
# and the §4.2 drift of a right-sign ICEwPI, on hand-built off-round cases.
# ---------------------------------------------------------------------------
_PROBE_CID = 83
# jaxob trade rows: (P, Q, PASS_OID, AGRS_OID, ...); book rows: (P, Q, OID, ...).
_TP, _TQ, _TPASS, _TAGRS = 0, 1, 2, 3


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


def _probe_order(pc, side, p_tick, q_tick, oid, ts, peak_tick=0, ppd_tick=0):
    from omiecid_simulator.omiecid.order_types import make_order

    off = int(round(-pc.price_min / pc.price_tick))
    return make_order(
        contract_id=_PROBE_CID, side=Side(side), price=round((p_tick - off) * pc.price_tick, 2),
        quantity=round(q_tick * pc.qty_tick, 1), order_id=oid, agent_id=oid % 97 + 1, timestamp=ts,
        exec_restriction=ExecRestriction.ICE if peak_tick else ExecRestriction.NON,
        peak_qty=round(peak_tick * pc.qty_tick, 1) if peak_tick else 0.0,
        ppd=round(ppd_tick * pc.price_tick, 2))


def _row(side, price, qty, oid, exec_r, peak=0, ppd=0) -> np.ndarray:
    r = np.zeros(13, dtype=np.int64)
    r[C_SIDE], r[C_PRICE], r[C_QTY], r[C_OID], r[C_EXEC], r[C_PEAK], r[C_PPD] = \
        side, price, qty, oid, exec_r, peak, ppd
    return r


def _leaves_equal(a, b) -> bool:
    import jax
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).dtype == np.asarray(y).dtype and np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(la, lb))


def _resting(state, side) -> List[List[int]]:
    arr = np.asarray(state.lob.bids if side == BUY else state.lob.asks)
    return sorted([int(r[0]), int(r[1]), int(r[2])] for r in arr if int(r[2]) > 0 and int(r[1]) > 0)


# (iceberg side, base tick, total, peak, |ppd|, aggressor limit offset in slices, aggressor qty)
_DRIFT_CASES = {
    "buy_iceberg_walks_down": (BUY, 1_005_037, 47, 11, 7, 2, 39),
    "sell_iceberg_walks_up": (SELL, 1_003_911, 29, 8, 5, 1, 21),
    "sell_iceberg_large_ppd_300_ticks": (SELL, 1_002_413, 33, 9, 300, 2, 31),  # -3.00-sized |PPD| (§13.6)
}


@pytest.mark.jit
class TestEngineProbesICEwPISign:
    """jit: ContractBook.submit_order drives the module-level jitted submit (compile per worker)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("ice_side", [BUY, SELL], ids=["buy_ppd_positive", "sell_ppd_negative"])
    def test_wrong_sign_iceberg_rejected_on_empty_book(self, probe_book, ice_side):
        """slow: first submit compiles the jitted body. §5.4 v2.14: rejected, state untouched."""
        book, pc = probe_book
        s0 = book.init()
        wrong = 13 if ice_side == BUY else -13
        s1, r = book.submit_order(s0, _probe_order(pc, ice_side, 1_004_217, 41, 7_311, 1, 9, wrong))
        assert bool(r.accepted) is False and bool(r.traded) is False
        assert _leaves_equal(s1, s0), "a rejected wrong-sign iceberg mutated the book"

    @pytest.mark.slow
    @pytest.mark.parametrize("ice_side", [BUY, SELL], ids=["buy_ppd_positive", "sell_ppd_negative"])
    def test_wrong_sign_then_crossing_aggressor_rests_whole(self, probe_book, ice_side):
        """slow: jitted submit. The C09 wrong-sign scenario through the public API: iceberg
        rejected at entry, then a NON aggressor that crosses its base trades nothing and
        rests whole at its limit; the iceberg's side stays empty."""
        book, pc = probe_book
        base, ppd = 1_004_617, (6 if ice_side == BUY else -6)
        agg_side = _other(ice_side)
        L = base - 4 if agg_side == SELL else base + 4  # crosses the base (non-vacuous)
        s0 = book.init()
        s1, r1 = book.submit_order(s0, _probe_order(pc, ice_side, base, 50, 8_101, 1, 12, ppd))
        assert bool(r1.accepted) is False and _leaves_equal(s1, s0)
        assert _walk(_row(ice_side, base, 50, 8_101, ICE, 12, ppd),
                     _row(agg_side, L, 23, 8_102, NON), ppd)["trades"], "probe must be non-vacuous"
        before = np.asarray(s1.lob.trades)
        s2, r2 = book.submit_order(s1, _probe_order(pc, agg_side, L, 23, 8_102, 2))
        assert bool(r2.accepted) is True and bool(r2.traded) is False
        assert np.array_equal(np.asarray(s2.lob.trades), before), "zero trades"
        assert _resting(s2, agg_side) == [[L, 23, 8_102]]
        assert _resting(s2, ice_side) == []
        assert not bool(s2.truncated) and not bool(s2.book_full)

    @pytest.mark.slow
    @pytest.mark.parametrize("ice_side", [BUY, SELL], ids=["buy_ppd_negative", "sell_ppd_positive"])
    def test_right_sign_iceberg_accepted_and_rests_at_base(self, probe_book, ice_side):
        """slow: jitted submit. The sign-valid counterpart is accepted: rests showing its peak at base."""
        book, pc = probe_book
        ppd = -13 if ice_side == BUY else 13
        s1, r = book.submit_order(book.init(), _probe_order(pc, ice_side, 1_004_217, 41, 7_313, 1, 9, ppd))
        assert bool(r.accepted) is True and bool(r.traded) is False
        assert _resting(s1, ice_side) == [[1_004_217, 9, 7_313]]

    @pytest.mark.slow
    @pytest.mark.parametrize("case", sorted(_DRIFT_CASES))
    def test_right_sign_iceberg_drifts_exact_ladder_and_stops(self, probe_book, case):
        """slow: jitted submit. §4.2 + §4.2.1 Rule 2: one aggressor walks base + k*ppd and stops
        at the first non-crossing slice; iceberg sets every price; residual rests at L."""
        book, pc = probe_book
        ice_side, base, total, peak, mag, m, q = _DRIFT_CASES[case]
        ppd = -mag if ice_side == BUY else mag
        agg_side = _other(ice_side)
        # L sits strictly between slice m-1 and slice m, so exactly m slices cross.
        L = base + (m - 1) * ppd + (ppd // 2 if abs(ppd) > 1 else 0)
        ice_oid, agg_oid = 9_001 + mag, 9_501 + mag
        ice_row = _row(ice_side, base, total, ice_oid, ICE, peak, ppd)
        agg_row = _row(agg_side, L, q, agg_oid, NON)
        w = _walk(ice_row, agg_row, ppd)
        assert w["stop"] == m and w["reason"] == "non_crossing" and w["q_rem"] > 0, "case precondition"

        s1, r1 = book.submit_order(book.init(), _probe_order(pc, ice_side, base, total, ice_oid, 1, peak, ppd))
        assert bool(r1.accepted) is True and bool(r1.traded) is False
        before = np.asarray(s1.lob.trades)
        s2, r2 = book.submit_order(s1, _probe_order(pc, agg_side, L, q, agg_oid, 2))
        after = np.asarray(s2.lob.trades)
        rows = after[np.any(before != after, axis=1)]
        got = [(int(t[_TP]), abs(int(t[_TQ])), int(t[_TPASS]), int(t[_TAGRS])) for t in rows]
        exp = [(t[2], t[3], ice_oid, agg_oid) for t in w["trades"]]
        assert got == exp, f"engine rows {got} != derived ladder {exp}"
        assert [g[0] for g in got] == [base + k * ppd for k in range(m)]
        assert bool(r2.accepted) is True and bool(r2.traded) is True
        assert _resting(s2, agg_side) == [[L, w["q_rem"], agg_oid]]
        ice_rest = _resting(s2, ice_side)
        assert ice_rest in ([[base + m * ppd, w["sizes"][m], ice_oid]], []), \
            f"iceberg slice m must show at base + m*ppd = {base + m * ppd} or be not yet shown (OQ-20)"
        assert not bool(s2.truncated) and not bool(s2.book_full)
