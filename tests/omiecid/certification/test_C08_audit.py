"""K4 audit twin for OMIE-CERT-C08-1 (iceberg regeneration), re-run r4 (OMIE-CERT-RERUN-3).

Fresh-context, adversarial audit. The expectation below is re-derived from the
rules alone, ``market-spec.md`` v2.14 §4.2, §4.2.1 Rules 2/3/4 (Rule 2
*Incoming iceberg* items (1)-(6) and Rule 3 *ICE aggressor*, both amended
2026-10-20; Rule 3 priced per OQ-31 Reading B), §5.2 (iceberg-slice bullet),
§6.1, §6.2 items 3/7, OMIE Rules Annex §4.4-§4.5, ``open-questions.md`` OQ-33
("Revised by user ruling (2026-10-20 ...)") and OQ-34 ("Resolved by user
ruling (2026-10-20 ...)"), and from the drawn tapes of ``omie_tape_synth``
(``C08.iceberg_cascade``, ``C08.rule3``, ``C08.incoming_iceberg`` and
``make_iceberg_cascade(seed, variant="incoming_remainder")``). It never reads the
C08 notebook's code, the generator's ``Draw.expected``, or the engine modules
(independence rules 3/4).

What is derived, per draw, in integer ticks:

* resting iceberg (cascade and Rule-3 draws): one trade per slice, sizes
  ``peak, ..., peak, r`` with ``n = ceil(total/peak)`` slices for the cascade and
  ``peak x floor(q/peak)`` then ``q mod peak`` for Rule 3; every trade at the
  iceberg's own limit ``p_ice`` and never at the aggressor's limit ``L``; every
  row written by the aggressor's own submission (Rule 4); the visible slice left
  resting is a later-timestamped one.
* incoming iceberg (r4 correction, Rule 2 *Incoming iceberg* (1)-(6) amended
  2026-10-20; OQ-33 revised; OQ-34 resolved): it aggresses with its **total**
  quantity as a chain of instances of size ``P`` (``d = 0`` here, so every
  instance has limit ``b``). Each trade is between one resting counter-bid entry
  and one incoming instance: a row ends where a resting entry ends **or** where
  one incoming instance ends and the next begins; two resting instances are
  never merged. Each trade prints at the resting entry's limit (§6.2 item 3).
  A remainder ``R > 0`` rests as instance ``k_e = floor(M/P)`` at ``b + k_e d``
  (= ``b``) showing ``v1 = min(P - u, R)`` (``u = M - k_e P``) with ``R - v1``
  hidden. The OQ-33 "Ruling: one trade per counter bid" reading is retired.
* every submission: reported quantity == the trade rows it wrote (I6(b),
  engineering invariant) and the final book uncrossed (I3).
* capacity: the analytic hop bound stays under the **100-pass bound**.

Independent engine probes (ContractBook public API, jitted ``submit_order``)
send an incoming iceberg into a ladder of resting icebergs and check the rows
split at resting entries and at incoming instance boundaries, the remainder
display ``v1``, and (by consuming it with a follow-up NON) the hidden ``R - v1``.

Results JSON: ``$OMIE_CERT_RESULTS_DIR/C08.json`` else
``notebooks/certification/results/C08.json``; absent -> skip.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid import omie_tape_synth as synth
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from notebooks.certification.cert_support import (
    SpecQuote,
    check_quote,
    derive_failure_kinds,
    quote_digest,
    validate_results,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_ID = "C08"

BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, ICE = int(ExecRestriction.NON), int(ExecRestriction.ICE)
SIDE_NAME = {BUY: "BUY", SELL: "SELL"}

# Tape columns (loader layout, documented in omie_tape_synth).
C_SIDE, C_PRICE, C_QTY, C_OID, C_EXEC, C_PEAK, C_PPD = 1, 2, 3, 4, 7, 11, 12

# kind -> (construction key, make_iceberg_cascade variant)
CONSTRUCTIONS = {
    "cascade": ("C08.iceberg_cascade", "resting"),
    "rule3": ("C08.rule3", "rule3"),
    "incoming": ("C08.incoming_iceberg", "incoming"),
    "remainder": ("C08.incoming_remainder", "incoming_remainder"),
}
REGISTRY_KINDS = ("cascade", "rule3", "incoming")  # in synth.CONSTRUCTIONS (digested)
INCOMING_KINDS = ("incoming", "remainder")
ALL_CLASSES = (
    "cascade, iceberg BUY, r < peak", "cascade, iceberg BUY, r = peak",
    "cascade, iceberg SELL, r < peak", "cascade, iceberg SELL, r = peak",
    "rule3, iceberg BUY, q_agg mod peak > 0", "rule3, iceberg BUY, q_agg mod peak = 0",
    "rule3, iceberg SELL, q_agg mod peak > 0", "rule3, iceberg SELL, q_agg mod peak = 0",
    "incoming, iceberg BUY, one level", "incoming, iceberg BUY, several levels",
    "incoming, iceberg SELL, one level", "incoming, iceberg SELL, several levels",
    "remainder, iceberg BUY", "remainder, iceberg SELL",
)
BATTERY = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")
ENVELOPE = {"resting_per_side": 80, "trades": 80, "hops": 160, "concurrent_icebergs": 800}
PASS_CEILING = 100  # outer matching passes (_TRIP_BOUND); scenario-ladder v2.4
SLICE_MARGIN = 64
SPEC_VERSION = "v2.14"

_PER_DRAW_RE = re.compile(r"^seed (\d+) \[((cascade|rule3|incoming|remainder), [^\]]+)\]: ")
_REQUIRED_OBS_KEYS = ("accepted", "filled", "remaining", "trades", "final_book", "reports",
                      "rows_qty", "rejected", "lost")
_INCOMING_OBS_KEYS = ("passive", "incoming_rest", "instance_split")


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
        pytest.skip(f"C08 results JSON absent: {p} (run the C08 notebook, section 8, first)")
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
def _draw(kind: str, seed: int):
    _, variant = CONSTRUCTIONS[kind]
    return synth.make_iceberg_cascade(int(seed), variant=variant)


def _real(tape: np.ndarray) -> np.ndarray:
    t = np.asarray(tape).reshape(-1, 13)
    return t[t[:, C_OID] >= 1]


def _ceil(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _other(side: int) -> int:
    return SELL if side == BUY else BUY


def _crosses(taker_side: int, taker_limit: int, maker_price: int) -> bool:
    return taker_limit >= maker_price if taker_side == BUY else taker_limit <= maker_price


def _buyer_seller(a_side: int, a_oid: int, b_oid: int) -> Tuple[int, int]:
    return (a_oid, b_oid) if a_side == BUY else (b_oid, a_oid)


def _book_key(book: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    return {s: sorted([list(map(int, e)) for e in book.get(s, [])]) for s in ("BUY", "SELL")}


def _derive_resting(real: np.ndarray) -> Dict[str, Any]:
    """Resting iceberg (row 0) hit by one NON aggressor (row 1): cascade or Rule-3."""
    ice, agg = real[0], real[1]
    assert int(ice[C_EXEC]) == ICE and int(ice[C_PPD]) == 0, "dataset: row 0 must be a PPD=0 iceberg"
    assert int(agg[C_EXEC]) == NON and int(agg[C_SIDE]) != int(ice[C_SIDE]), "dataset: row 1 NON, opposite side"
    ice_oid, agg_oid = int(ice[C_OID]), int(agg[C_OID])
    total, peak, p_ice = int(ice[C_QTY]), int(ice[C_PEAK]), int(ice[C_PRICE])
    q, L, agg_side = int(agg[C_QTY]), int(agg[C_PRICE]), int(agg[C_SIDE])
    assert 0 < peak <= total
    assert L != p_ice and _crosses(agg_side, L, p_ice), "dataset: aggressor limit strictly through p_ice"
    buyer, seller = _buyer_seller(agg_side, agg_oid, ice_oid)

    # Rule 2 / Rule 4: the aggressor works through slices inside its own process;
    # each exhausted slice is replaced by a new slice min(peak, hidden).
    hidden_total, visible, q_rem = total, min(peak, total), q
    slice_index, trades, slices_consumed = 0, [], []
    exhausted_at_end = False
    while q_rem > 0 and hidden_total > 0:
        t = min(q_rem, visible)
        trades.append([buyer, seller, p_ice, t, 1])  # price set by the iceberg (§5.1.2, OQ-31 B)
        q_rem -= t
        visible -= t
        hidden_total -= t
        slices_consumed.append(t)
        exhausted_at_end = visible == 0
        if visible == 0 and hidden_total > 0 and q_rem > 0:
            visible = min(peak, hidden_total)
            slice_index += 1
    filled = q - q_rem
    ice_side_name = SIDE_NAME[int(ice[C_SIDE])]
    agg_side_name = SIDE_NAME[agg_side]

    books: List[Dict[str, List[List[int]]]] = []
    agg_rest = [[L, q_rem, agg_oid]] if q_rem > 0 else []
    if hidden_total == 0:
        books.append({ice_side_name: [], agg_side_name: agg_rest})
        new_ts = None
    elif not exhausted_at_end:
        books.append({ice_side_name: [[p_ice, visible, ice_oid]], agg_side_name: agg_rest})
        new_ts = {"later"} if slice_index >= 1 else {"same"}
    else:
        # Peak fully consumed exactly as the aggressor finished: the next slice is
        # activated now, or registered for later activation (Rule 4 scope
        # clarification permits the latter; OQ-20/OQ-23). Either way at p_ice, size
        # min(peak, hidden).
        books.append({ice_side_name: [[p_ice, min(peak, hidden_total), ice_oid]],
                      agg_side_name: agg_rest})
        books.append({ice_side_name: [], agg_side_name: agg_rest})
        new_ts = {"later", "absent"}
    return {
        "accepted": [ice_oid, agg_oid],
        "rejected": [],
        "lost": {},
        "filled": {str(ice_oid): filled, str(agg_oid): filled},
        "remaining": {str(ice_oid): hidden_total, str(agg_oid): q_rem},
        "trades": trades,
        "reports": [[True, False, 0], [True, filled > 0, filled]],
        "rows_qty": [0, filled],
        "final_books": [_book_key(b) for b in books],
        "new_slice_ts": new_ts,
        "slices_consumed": slices_consumed,
        "total": total, "peak": peak, "q": q, "p_ice": p_ice, "L": L,
        "ice_side": int(ice[C_SIDE]), "agg_submission": 1,
    }


def _sweep(entries: List[Dict[str, int]], taker_side: int, limit: int, qty: int,
           taker_oid: int, sub: int, peak: int) -> Tuple[List[List[int]], List[Dict[str, int]], int]:
    """Incoming iceberg sweeping its TOTAL quantity as a chain of instances
    (§4.2.1 Rule 2 *Incoming iceberg* (1), (2), (3), (6), amended 2026-10-20).

    ``entries``: resting counter bids ``{p, vis, hidden, peak, oid, ts}``. The
    aggressor always trades the best-ranked entry (§6.1 price-time; Re-ranking
    bullet). When a resting iceberg instance is exhausted its next instance is a
    NEW entry with a NEW (later) timestamp (§4.2, §5.2, Annex §4.4 "its own
    creation time"). A row ends where the resting entry ends OR where the
    incoming instance (size ``peak``) ends. ``d = 0``: every incoming instance
    has limit ``limit``. Each trade prints at the resting entry's limit.
    Returns (trades, book, remainder).
    """
    book = [dict(e) for e in entries]
    ts_next = max([e["ts"] for e in book], default=0) + 1
    trades: List[List[int]] = []
    rem, swept = int(qty), 0
    while rem > 0:
        live = [e for e in book if e["vis"] > 0]
        if not live:
            break
        best = min(live, key=lambda e: (e["p"] if taker_side == BUY else -e["p"], e["ts"]))
        if not _crosses(taker_side, limit, best["p"]):
            break
        left_in_instance = peak - (swept % peak)
        t = min(rem, best["vis"], left_in_instance)
        b, s_ = _buyer_seller(taker_side, taker_oid, best["oid"])
        trades.append([b, s_, best["p"], t, sub])
        best["vis"] -= t
        rem -= t
        swept += t
        if best["vis"] == 0 and best["hidden"] > 0:
            nxt = min(best["peak"], best["hidden"])
            best["hidden"] -= nxt
            best["vis"] = nxt
            best["ts"] = ts_next
            ts_next += 1
    return trades, book, rem


def _rest_display(total: int, peak: int, rem: int) -> Tuple[int, int, int, int]:
    """Rule 2 *Incoming iceberg* (4): (k_e, u, v1, hidden) of a remainder R."""
    M = total - rem
    k_e = M // peak
    u = M - k_e * peak
    v1 = min(peak - u, rem)
    return k_e, u, v1, rem - v1


def _derive_incoming(real: np.ndarray) -> Dict[str, Any]:
    """Resting NON ladder (rows 0..L-1), then an incoming iceberg (row L)."""
    ice = real[-1]
    levels = real[:-1]
    assert int(ice[C_EXEC]) == ICE and int(ice[C_PPD]) == 0, "dataset: incoming iceberg PPD 0"
    assert all(int(r[C_EXEC]) == NON and int(r[C_SIDE]) != int(ice[C_SIDE]) for r in levels)
    sub = len(levels)  # the incoming iceberg's submission index
    ice_oid, ice_side = int(ice[C_OID]), int(ice[C_SIDE])
    total, peak, p_lim = int(ice[C_QTY]), int(ice[C_PEAK]), int(ice[C_PRICE])
    assert 0 < peak <= total
    entries = [{"p": int(r[C_PRICE]), "vis": int(r[C_QTY]), "hidden": 0, "peak": int(r[C_QTY]),
                "oid": int(r[C_OID]), "ts": i + 1} for i, r in enumerate(levels)]
    trades, book, rem = _sweep(entries, ice_side, p_lim, total, ice_oid, sub, peak)
    passive = [t[1] if ice_side == BUY else t[0] for t in trades]
    swept = total - rem
    k_e, u, v1, hidden = _rest_display(total, peak, rem)
    rest_price = p_lim  # b + k_e d with d = 0
    rest_name, ice_name = SIDE_NAME[_other(ice_side)], SIDE_NAME[ice_side]
    final = {rest_name: [[e["p"], e["vis"], e["oid"]] for e in book if e["vis"] > 0],
             ice_name: [[rest_price, v1, ice_oid]] if rem > 0 else []}
    left = {e["oid"]: e["vis"] for e in book}
    filled = {str(int(r[C_OID])): int(r[C_QTY]) - left[int(r[C_OID])] for r in levels}
    filled[str(ice_oid)] = swept
    remaining = {str(o): q for o, q in left.items()}
    remaining[str(ice_oid)] = rem
    return {
        "accepted": [int(r[C_OID]) for r in real],
        "rejected": [],
        "lost": {},
        "filled": filled,
        "remaining": remaining,
        "trades": trades,
        "passive": passive,
        "reports": [[True, False, 0]] * sub + [[True, swept > 0, swept]],
        "rows_qty": [0] * sub + [swept],
        "final_books": [_book_key(final)],
        "incoming_rest": [rest_price, v1, hidden] if rem > 0 else [],
        "total": total, "peak": peak, "p_lim": p_lim, "ice_side": ice_side, "ice_oid": ice_oid,
        "swept": swept, "R": rem, "k_e": k_e, "u": u, "v1": v1, "hidden": hidden,
        "n_levels_touched": len(set(passive)), "sub": sub,
        "level_price": {int(r[C_OID]): int(r[C_PRICE]) for r in levels},
    }


@functools.lru_cache(maxsize=None)
def _derive(kind: str, seed: int) -> Dict[str, Any]:
    real = _real(_draw(kind, seed).tape)
    return _derive_incoming(real) if kind in INCOMING_KINDS else _derive_resting(real)


def _class_label(kind: str, d: Dict[str, Any]) -> str:
    side = SIDE_NAME[d["ice_side"]]
    if kind == "cascade":
        r = d["total"] - (_ceil(d["total"], d["peak"]) - 1) * d["peak"]
        return f"cascade, iceberg {side}, r {'<' if r < d['peak'] else '='} peak"
    if kind == "rule3":
        return f"rule3, iceberg {side}, q_agg mod peak {'>' if d['q'] % d['peak'] else '='} 0"
    if kind == "remainder":
        return f"remainder, iceberg {side}"
    return f"incoming, iceberg {side}, {'one level' if d['n_levels_touched'] == 1 else 'several levels'}"


# ---------------------------------------------------------------------------
# Checks on an emitted payload; each returns a list of problems ([] = clean)
# ---------------------------------------------------------------------------
def _per_draw(payload) -> List[Tuple[str, int, str, Dict[str, Any]]]:
    out = []
    for a in payload.get("assertions", []):
        m = _PER_DRAW_RE.match(a.get("name", ""))
        if m and a.get("kind") == "oracle":
            out.append((m.group(3), int(m.group(1)), m.group(2), a))
    return out


def _crossed(book: Dict[str, List[List[int]]]) -> bool:
    bids = [int(e[0]) for e in book.get("BUY", []) if int(e[1]) > 0]
    asks = [int(e[0]) for e in book.get("SELL", []) if int(e[1]) > 0]
    return bool(bids and asks and max(bids) >= min(asks))


def _i6b_problems(obs: Dict[str, Any], tag: str) -> List[str]:
    """Report == tape: per submission, reported qty == rows it wrote == its trade rows."""
    out = []
    reports, rows_qty, trades = obs["reports"], obs["rows_qty"], obs["trades"]
    if len(reports) != len(rows_qty):
        return [f"{tag}: reports/rows_qty length mismatch"]
    for k, (rep, rq) in enumerate(zip(reports, rows_qty)):
        from_rows = sum(int(t[3]) for t in trades if int(t[4]) == k)
        if int(rep[2]) != int(rq) or int(rq) != from_rows:
            out.append(f"{tag}: submission {k} reports {rep[2]}, rows_qty {rq}, trade rows {from_rows} (I6(b))")
        if bool(rep[1]) != (from_rows > 0):
            out.append(f"{tag}: submission {k} traded flag {rep[1]} vs {from_rows} rows")
    return out


def _instance_split_problems(trades: List[List[int]], passive: List[int], d: Dict[str, Any],
                             tag: str) -> List[str]:
    """§4.2.1 Rule 2 *Incoming iceberg* (2), amended 2026-10-20; OQ-33 revised:
    a row is one (resting entry, incoming instance) pair. So: every row <= P;
    every multiple jP < M is a cumulative row boundary; a row's passive side is
    exactly one resting entry; consecutive rows against the same resting order
    are separated by an instance boundary (otherwise they would be one row);
    every row written by the incoming iceberg's own submission."""
    out = []
    P, M, ice_oid = d["peak"], d["swept"], d["ice_oid"]
    if len(trades) != len(passive):
        out.append(f"{tag}: {len(trades)} trade rows but {len(passive)} passive oids")
    for t, p in zip(trades, passive):
        row_pass = t[1] if t[0] == ice_oid else t[0]
        if ice_oid not in (t[0], t[1]) or row_pass != p:
            out.append(f"{tag}: row {t} is not (incoming iceberg, resting {p})")
    over = [t[3] for t in trades if t[3] > P]
    if over:
        out.append(f"{tag}: rows {over} exceed the incoming peak {P} (instance boundary crossed "
                   f"inside a row; OQ-33 revised 2026-10-20)")
    cum, acc = set(), 0
    for t in trades:
        acc += t[3]
        cum.add(acc)
    missing = [j * P for j in range(1, _ceil(M, P)) if j * P < M and j * P not in cum]
    if missing:
        out.append(f"{tag}: instance boundaries {missing} are not row boundaries (OQ-33 revised)")
    acc = 0
    for i in range(1, len(trades)):
        acc += trades[i - 1][3]
        if passive[i] == passive[i - 1] and acc % P != 0:
            out.append(f"{tag}: rows {i - 1},{i} against resting {passive[i]} split without an "
                       f"instance boundary (resting NON is one counter bid)")
    if any(t[4] != d["sub"] for t in trades):
        out.append(f"{tag}: an incoming-iceberg row not written by its own submission")
    return out


def _remainder_problems(obs: Dict[str, Any], d: Dict[str, Any], tag: str) -> List[str]:
    """§4.2.1 Rule 2 *Incoming iceberg* (4), amended 2026-10-20; OQ-34 resolved."""
    out = []
    ice_name = SIDE_NAME[d["ice_side"]]
    mine = [list(map(int, e)) for e in obs["final_book"].get(ice_name, []) if int(e[2]) == d["ice_oid"]]
    if d["R"] > 0:
        want = [d["p_lim"], d["v1"], d["ice_oid"]]
        if mine != [want]:
            out.append(f"{tag}: remainder rests as {mine}, not [[b + k_e d, v1 = min(P - u, R), oid]] "
                       f"= [{want}] (k_e={d['k_e']}, u={d['u']}, R={d['R']}; Rule 2 Incoming (4))")
        if [int(x) for x in obs["incoming_rest"]] != d["incoming_rest"]:
            out.append(f"{tag}: incoming_rest {obs['incoming_rest']} != derived [price, v1, hidden] "
                       f"{d['incoming_rest']}")
        if int(obs["remaining"][str(d["ice_oid"])]) != d["R"]:
            out.append(f"{tag}: remaining {obs['remaining'][str(d['ice_oid'])]} != R {d['R']}")
    else:
        if mine or obs["incoming_rest"]:
            out.append(f"{tag}: incoming iceberg rests {mine or obs['incoming_rest']} with R = 0")
    return out


def draw_problems(kind: str, seed: int, label: str, a: Dict[str, Any]) -> List[str]:
    """Compare one emitted per-draw assertion against the independent derivation."""
    tag = f"{kind} seed {seed}"
    out: List[str] = []
    d = _derive(kind, seed)
    if label != _class_label(kind, d):
        out.append(f"{tag}: class label {label!r} != derived {_class_label(kind, d)!r}")
    if a.get("status") != "pass":
        out.append(f"{tag}: assertion status {a.get('status')!r}")
    try:
        obs = json.loads(a["observed"]) if isinstance(a.get("observed"), str) else a.get("observed")
        exp = json.loads(a["expected"]) if isinstance(a.get("expected"), str) else a.get("expected")
    except (TypeError, ValueError) as exc:
        return out + [f"{tag}: observed/expected not parseable JSON ({exc})"]
    if not isinstance(obs, dict):
        return out + [f"{tag}: observed missing"]
    missing = [k for k in _REQUIRED_OBS_KEYS if k not in obs]
    if kind in INCOMING_KINDS:
        missing += [k for k in _INCOMING_OBS_KEYS if k not in obs]
    if kind == "rule3" and "new_slice_ts" not in obs:
        missing.append("new_slice_ts")
    if missing:
        return out + [f"{tag}: observed lacks {missing} (per-draw values not present in full)"]

    trades = [list(map(int, t)) for t in obs["trades"]]
    # Exact trade sequence (buyer, seller, price, qty, originating submission).
    if trades != d["trades"]:
        out.append(f"{tag}: trades {trades} != derived {d['trades']}")
    for key in ("accepted", "rejected", "lost", "filled", "remaining", "rows_qty"):
        if obs[key] != d[key]:
            out.append(f"{tag}: {key} {obs[key]} != derived {d[key]}")
    if [[bool(r[0]), bool(r[1]), int(r[2])] for r in obs["reports"]] != d["reports"]:
        out.append(f"{tag}: reports {obs['reports']} != derived {d['reports']}")
    fb = _book_key(obs["final_book"])
    if fb not in d["final_books"]:
        out.append(f"{tag}: final book {fb} not in derived {d['final_books']}")
    if _crossed(obs["final_book"]):
        out.append(f"{tag}: final book crossed (I3, §4.2.1 Rule 4)")
    out += _i6b_problems(obs, tag)

    if kind in ("cascade", "rule3"):
        prices = {t[2] for t in trades}
        if prices - {d["p_ice"]}:
            out.append(f"{tag}: trade price(s) {sorted(prices)} != p_ice {d['p_ice']} (§6.2 item 7, OQ-31 B)")
        if d["L"] in prices:
            out.append(f"{tag}: a trade printed at the aggressor's limit L={d['L']} (OQ-31 forbids)")
        if any(t[4] != d["agg_submission"] for t in trades):
            out.append(f"{tag}: a row not written by the originating matching process (Rule 4)")
        sizes = [t[3] for t in trades]
        if kind == "cascade":
            n = _ceil(d["total"], d["peak"])
            want = [d["peak"]] * (n - 1) + [d["total"] - (n - 1) * d["peak"]]
            if sizes != want or len(trades) != n:
                out.append(f"{tag}: slice sizes {sizes} != {want} (n = ceil(total/peak) = {n})")
        else:
            full, part = divmod(d["q"], d["peak"])
            want = [d["peak"]] * full + ([part] if part else [])
            if sizes != want:
                out.append(f"{tag}: Rule-3 slice sizes {sizes} != {want}")
            if obs["new_slice_ts"] not in d["new_slice_ts"]:
                out.append(f"{tag}: new_slice_ts {obs['new_slice_ts']!r} not in {sorted(d['new_slice_ts'])}")
            book_present = bool(fb[SIDE_NAME[d['ice_side']]])
            if (obs["new_slice_ts"] == "absent") == book_present:
                out.append(f"{tag}: new_slice_ts {obs['new_slice_ts']!r} inconsistent with final book")
    else:
        passive = [int(p) for p in obs["passive"]]
        if passive != d["passive"]:
            out.append(f"{tag}: passive {obs['passive']} != derived {d['passive']}")
        if obs["reports"][-1][2] != d["swept"] or sum(t[3] for t in trades) != d["swept"]:
            out.append(f"{tag}: incoming iceberg reported {obs['reports'][-1][2]} != swept M {d['swept']}")
        if kind == "incoming" and d["swept"] != d["total"]:
            out.append(f"{tag}: incoming draw does not sweep its total ({d['swept']} of {d['total']})")
        # Every trade at its price-setter's (the resting entry's) limit, §6.2 item 3.
        off = [t for t, p in zip(trades, passive) if t[2] != d["level_price"].get(p)]
        if off:
            out.append(f"{tag}: rows {off} not at their resting entry's limit (Rule 2 Incoming (3))")
        out += _instance_split_problems(trades, passive, d, tag)
        out += _remainder_problems(obs, d, tag)
        if obs["instance_split"] != {"missing_boundaries": [], "rows_over_peak": []}:
            out.append(f"{tag}: emitted instance_split {obs['instance_split']} is not clean")

    # Cross-check K2's own expectation against this derivation (oracle drift).
    if isinstance(exp, dict):
        if [list(map(int, t)) for t in exp.get("trades", [])] != d["trades"]:
            out.append(f"{tag}: K2 expected trades disagree with the audit derivation")
        if "final_book_one_of" in exp:
            k2 = [_book_key(b) for b in exp["final_book_one_of"]]
            if any(b not in d["final_books"] for b in k2):
                out.append(f"{tag}: K2 admits a final book the rules do not: {k2}")
        if kind in INCOMING_KINDS and "incoming_rest" in exp \
                and [int(x) for x in exp["incoming_rest"]] != d["incoming_rest"]:
            out.append(f"{tag}: K2 expected incoming_rest {exp['incoming_rest']} != derived {d['incoming_rest']}")
    return out


def all_draw_problems(payload) -> List[str]:
    out = []
    for kind, seed, label, a in _per_draw(payload):
        out += draw_problems(kind, seed, label, a)
    return out


def coverage_problems(payload) -> List[str]:
    out = []
    rows = _per_draw(payload)
    seeds = [s for _, s, _, _ in rows]
    draws, sd = payload.get("draws", {}), payload.get("seeds", {})
    if len(seeds) != len(set(seeds)):
        out.append("duplicate per-draw seeds")
    if not (len(seeds) == draws.get("valid") == sd.get("count")):
        out.append(f"per-draw assertions {len(seeds)} vs draws.valid {draws.get('valid')} "
                   f"vs seeds.count {sd.get('count')}")
    if draws.get("requested") != draws.get("valid", 0) + draws.get("discarded", 0):
        out.append("draws: requested != valid + discarded")
    if seeds and (sd.get("first") != min(seeds) or sd.get("last") != max(seeds)):
        out.append("seeds.first/last disagree with the per-draw seeds")
    if sd.get("failing"):
        out.append(f"failing seeds {sd.get('failing')}")
    per_kind = {k: sum(1 for kk, *_ in rows if kk == k) for k in CONSTRUCTIONS}
    if any(v == 0 for v in per_kind.values()):
        out.append(f"a construction has no draws: {per_kind}")
    labels = {_class_label(k, _derive(k, s)) for k, s, _, _ in rows}
    missing = [c for c in ALL_CLASSES if c not in labels]
    if missing:
        out.append(f"classes never drawn: {missing}")
    return out


def eager_jit_problems(payload) -> List[str]:
    out = []
    ej = [a for a in payload.get("assertions", []) if a.get("kind") == "eager_jit"]
    if payload.get("eager_jit_identical") is not True:
        out.append(f"eager_jit_identical is {payload.get('eager_jit_identical')!r}")
    if not ej:
        return out + ["no eager-vs-jitted evidence (absence = failure)"]
    for a in ej:
        if a.get("status") != "pass" or a.get("observed") is not True:
            out.append(f"eager/jit disagreement: {a.get('name')}")
    valid = {(k, s) for k, s, _, _ in _per_draw(payload)}
    for kind in CONSTRUCTIONS:
        hits = [a for a in ej if re.search(rf"\(seed (\d+), {kind}\)", a.get("name", ""))]
        if not hits:
            out.append(f"no eager-vs-jitted evidence for construction {kind}")
        for a in hits:
            s = int(re.search(r"seed (\d+)", a["name"]).group(1))
            if (kind, s) not in valid:
                out.append(f"eager/jit seed {s} ({kind}) is not a valid draw of the run")
    return out


def battery_problems(payload) -> List[str]:
    out = []
    rows = {r.get("check_id"): r for r in payload.get("invariants", [])}
    valid = payload.get("draws", {}).get("valid")
    for cid in BATTERY:
        r = rows.get(cid)
        if r is None:
            out.append(f"battery row {cid} missing")
            continue
        if r.get("status") != "pass" or r.get("n_fail") != 0 or r.get("failing_seeds"):
            out.append(f"{cid}: status {r.get('status')}, n_fail {r.get('n_fail')}")
        if r.get("n_pass") != valid:
            out.append(f"{cid}: n_pass {r.get('n_pass')} != valid draws {valid} (not run on every draw)")
    i3 = [a for a in payload.get("assertions", []) if a.get("name", "").startswith(("I3 passes", "I3 holds"))]
    if not i3 or any(a.get("status") != "pass" for a in i3):
        out.append("I3-on-every-draw assertion missing or failed")
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
    # Slice margin, re-derived from the tapes.
    max_slices = max(_ceil(int(r[C_QTY]), int(r[C_PEAK]))
                     for k, s, _, _ in rows for r in _real(_draw(k, s).tape) if int(r[C_EXEC]) == ICE)
    hop = max(int(_draw(k, s).hop_bound) for k, s, _, _ in rows)
    max_trades = max(len(_derive(k, s)["trades"]) for k, s, _, _ in rows)
    if max_slices > SLICE_MARGIN:
        out.append(f"an iceberg has ceil(total/peak) = {max_slices} > {SLICE_MARGIN}")
    if hop > PASS_CEILING:
        out.append(f"analytic hop bound {hop} > {PASS_CEILING} outer matching passes (_TRIP_BOUND)")
    if max_trades > ENVELOPE["trades"]:
        out.append(f"{max_trades} trades > 80")
    by_name = {a.get("name", ""): a for a in payload.get("assertions", [])}
    sl = [a for n, a in by_name.items() if n.startswith("largest ceil(qty/peak)")]
    if not sl or sl[0].get("status") != "pass" or sl[0].get("observed") != max_slices:
        out.append(f"slice-margin assertion absent/failed or reports {sl[0].get('observed') if sl else None}"
                   f" != re-derived {max_slices}")
    hb = [a for n, a in by_name.items() if n.startswith(f"hop bound <= {PASS_CEILING} ")]
    if not hb or hb[0].get("status") != "pass" or not isinstance(hb[0].get("observed"), list) \
            or max(hb[0]["observed"]) > PASS_CEILING or hb[0]["observed"][0] != hop:
        out.append(f"hop-bound assertion (<= {PASS_CEILING} passes) absent/failed or disagrees "
                   f"with re-derived {hop}")
    cf = [a for n, a in by_name.items() if n.startswith("no capacity flag")]
    if not cf or cf[0].get("status") != "pass":
        out.append("no-capacity-flag (truncated/book_full/ring) assertion absent or failed")
    if cap.get("peaks", {}).get("trades") is not None and cap["peaks"]["trades"] != max_trades:
        out.append(f"capacity.peaks.trades {cap['peaks']['trades']} != re-derived {max_trades}")
    if "capacity_flag" in payload.get("failure_kinds", []):
        out.append("failure_kinds carries capacity_flag")
    return out


# Required verbatim passages: (source, section, block, anchor substring of the source line).
REQUIRED_QUOTES = (
    ("market-spec.md", "§4.2", None, "receives a new timestamp"),
    ("market-spec.md", "§4.2.1", None, "a crossed order book cannot persist"),
    ("market-spec.md", "§4.2.1", None, "the entire cascade MUST resolve to a fixed point"),
    ("market-spec.md", "§4.2.1", None, "never left in a crossed state"),
    ("market-spec.md", "§4.2.1", None, "It never trades at the aggressor's limit"),
    ("market-spec.md", "§5.2", None,
     "its timestamp does not supersede the incoming order for price determination purposes"),
    ("market-spec.md", "§6.2 item 7", None, "the iceberg is always the price-setter"),
    # r4: the amended (2026-10-20) Incoming iceberg items the incoming expectation rests on.
    ("market-spec.md", "§4.2.1", None, "Total quantity."),
    ("market-spec.md", "§4.2.1", None, "A trade row ends where a resting entry ends"),
    ("market-spec.md", "§4.2.1", None, "Every trade prices at the resting entry"),
    ("market-spec.md", "§4.2.1", None, "It shows $v_1 = \\min(P - u, R)$"),
    ("open-questions.md", "OQ-33", "Revised by user ruling (2026-10-20",
     "A row ends where a resting entry ends"),
)
# OQ blocks that must be quoted (by block lead) and the one that must NOT be (retired at r4).
REQUIRED_OQ_BLOCKS = (("OQ-31", "Resolution (2026-10-02"), ("OQ-33", "Revised by user ruling (2026-10-20"),
                      ("OQ-34", "Resolved by user ruling (2026-10-20"))
RETIRED_OQ_BLOCKS = (("OQ-33", "Ruling: one trade per counter bid"),
                     ("OQ-34", "The simulator's convention"))


def _quote_candidates(source: str, anchor: str) -> List[str]:
    text = (REPO_ROOT / "docs" / "omiecid" / source).read_text(encoding="utf-8")
    out = []
    for line in text.split("\n"):
        if anchor not in line:
            continue
        body = re.sub(r"^(\s*- |\s*\d+\. )", "", line).strip()
        sents = re.split(r"(?<=[.:;)\"])\s+", body)
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
    digests = {q.get("digest") for q in quotes}
    for source, section, block, anchor in REQUIRED_QUOTES:
        hits = [c for c in _quote_candidates(source, anchor)
                if quote_digest(SpecQuote(source, section, c, block)) in digests]
        if not hits:
            out.append(f"required passage not quoted: {source} {section} '{anchor}'")
            continue
        chk = check_quote(SpecQuote(source, section, hits[0], block))
        if not chk.ok:
            out.append(f"quoted passage {section} '{anchor}' no longer matches its source: {chk.reason}")
    oq = [q for q in quotes if q.get("source") == "open-questions.md"]
    for sec, lead in REQUIRED_OQ_BLOCKS:
        if not any(q.get("section") == sec and str(q.get("block", "")).startswith(lead) and q.get("ok")
                   for q in oq):
            out.append(f"{sec} '{lead}' ruling not quoted")
    for sec, lead in RETIRED_OQ_BLOCKS:
        if any(q.get("section") == sec and str(q.get("block", "")).startswith(lead) for q in oq):
            out.append(f"{sec} '{lead}' is quoted, but it is retired for incoming icebergs (2026-10-20)")
    stale = sorted({q.get("version") for q in quotes
                    if q.get("source") == "market-spec.md" and q.get("version") != SPEC_VERSION})
    if stale:
        out.append(f"market-spec quotes stamped {stale}, not {SPEC_VERSION}")
    return out


def scope_problems(payload) -> List[str]:
    out = []
    for a in payload.get("assertions", []):
        if "iceberg_meta" in a.get("name", "") or "iceberg_meta" in str(a.get("expected", "")):
            out.append(f"asserts on iceberg_meta (must-not-assert, OQ-20/OQ-23): {a.get('name')}")
    for a in payload.get("assertions", []):
        if a.get("status") == "fail" and a.get("kind") == "oracle":
            out.append(f"failed oracle assertion: {a.get('name')}")
    return out


def audit_problems(payload) -> List[str]:
    return (all_draw_problems(payload) + coverage_problems(payload) + eager_jit_problems(payload)
            + battery_problems(payload) + capacity_problems(payload) + scope_problems(payload))


def _rows(payload, kind: str):
    return [r for r in _per_draw(payload) if r[0] == kind]


# ---------------------------------------------------------------------------
# Tests on the emitted JSON
# ---------------------------------------------------------------------------
class TestEmittedResults:
    def test_schema_and_semantic_validation_clean(self, payload):
        assert validate_results(payload) == []
        assert payload["failure_kinds"] == derive_failure_kinds(payload)

    def test_identity_and_oracle_kind(self, payload):
        assert payload["cert_id"] == "C08"
        assert payload["notebook"] == "C08_iceberg_regeneration.ipynb"
        assert payload["oracle_kind"] == "exact-parametric"

    def test_status_pass_and_no_failure_kinds(self, payload):
        assert payload["status"] == "PASS", payload.get("failure_explanation")
        assert payload["failure_kinds"] == []
        assert payload["failure_explanation"] is None

    def test_generator_digests_match_this_code_version(self, payload):
        gen = payload["generator"]
        assert gen["module"] == "omiecid_simulator.jaxomiecid.omie_tape_synth"
        for kind in REGISTRY_KINDS:
            key = CONSTRUCTIONS[kind][0]
            assert gen["digests"].get(key) == synth.construction_digest(key), key

    def test_remainder_draws_reproduce_from_this_generator(self, payload):
        """C08.incoming_remainder has no registry digest; its provenance is checked by
        re-drawing every emitted seed and matching its exact trade sequence instead."""
        rows = _rows(payload, "remainder")
        assert len(rows) >= 1
        bad = [s for _, s, _, a in rows
               if [list(map(int, t)) for t in json.loads(a["observed"])["trades"]] != _derive("remainder", s)["trades"]]
        assert bad == []

    def test_every_valid_draw_has_a_per_draw_exact_assertion(self, payload):
        assert coverage_problems(payload) == []

    def test_no_draw_discarded_inside_the_margin(self, payload):
        d = payload["draws"]
        assert d["discarded"] == 0 and d["discard_rate"] == 0.0 and d["valid"] == d["requested"]

    def test_per_draw_values_equal_independent_derivation(self, payload):
        problems = all_draw_problems(payload)
        assert problems == [], "\n".join(problems[:20])

    def test_cascade_slice_count_and_sizes(self, payload):
        """Exactly ceil(total/peak) slices, sizes peak,...,peak,r, all at p_ice."""
        rows = _rows(payload, "cascade")
        assert len(rows) >= 1
        for _, seed, _, a in rows:
            obs = json.loads(a["observed"])
            d = _derive("cascade", seed)
            n = _ceil(d["total"], d["peak"])
            r = d["total"] - (n - 1) * d["peak"]
            assert 1 <= r <= d["peak"]
            assert [t[3] for t in obs["trades"]] == [d["peak"]] * (n - 1) + [r], seed
            assert {t[2] for t in obs["trades"]} == {d["p_ice"]}, seed
            assert not any(obs["final_book"].values()), seed

    def test_rule3_prices_at_iceberg_limit_never_at_L(self, payload):
        rows = _rows(payload, "rule3")
        assert len(rows) >= 1
        for _, seed, _, a in rows:
            obs = json.loads(a["observed"])
            d = _derive("rule3", seed)
            assert d["L"] != d["p_ice"]
            assert all(t[2] == d["p_ice"] for t in obs["trades"]), seed
            assert obs["remaining"][str(1)] == d["total"] - d["q"], seed

    def test_rule3_resting_slice_carries_a_later_timestamp(self, payload):
        """§4.2: a new slice receives a new timestamp; with q mod peak > 0 the resting
        slice is a regenerated one and must be present and later."""
        for _, seed, _, a in _rows(payload, "rule3"):
            obs = json.loads(a["observed"])
            d = _derive("rule3", seed)
            assert obs["new_slice_ts"] in d["new_slice_ts"], seed
            if d["q"] % d["peak"]:
                assert obs["new_slice_ts"] == "later", seed

    def test_incoming_report_equals_total_and_tape(self, payload):
        """Rule 2 Incoming (1): the incoming draw sweeps its total; I6(b)."""
        rows = _rows(payload, "incoming")
        assert len(rows) >= 1
        for _, seed, _, a in rows:
            obs = json.loads(a["observed"])
            d = _derive("incoming", seed)
            assert d["R"] == 0, seed
            assert obs["reports"][-1][2] == d["total"] == sum(t[3] for t in obs["trades"]), seed
            assert _i6b_problems(obs, str(seed)) == []

    @pytest.mark.parametrize("kind", INCOMING_KINDS)
    def test_incoming_rows_split_at_resting_entries_and_instance_boundaries(self, payload, kind):
        """Rule 2 Incoming (2), amended 2026-10-20; OQ-33 revised: each row is one
        (resting entry, incoming instance) pair; every row <= P; every jP < M is a
        cumulative row boundary; each row at its resting entry's limit (item (3))."""
        rows = _rows(payload, kind)
        assert len(rows) >= 1
        split_at_instance = 0
        for _, seed, _, a in rows:
            obs = json.loads(a["observed"])
            d = _derive(kind, seed)
            trades = [list(map(int, t)) for t in obs["trades"]]
            passive = [int(x) for x in obs["passive"]]
            assert _instance_split_problems(trades, passive, d, str(seed)) == []
            level_price = {int(e[C_OID]): int(e[C_PRICE]) for e in _real(_draw(kind, seed).tape)}
            assert all(t[2] == level_price[p] for t, p in zip(trades, passive)), seed
            assert sum(t[3] for t in trades) == d["swept"], seed
            # One resting NON level's fill cut into two rows at an incoming instance boundary.
            split_at_instance += any(passive[i] == passive[i - 1] for i in range(1, len(passive)))
        # Non-vacuity: the r3 (one-row-per-counter-bid) shape would differ on these draws.
        assert split_at_instance > 0

    def test_remainder_rests_as_instance_k_e_showing_v1(self, payload):
        """Rule 2 Incoming (4), amended 2026-10-20; OQ-34 resolved: R > 0 rests at
        b + k_e d (d = 0: b) showing v1 = min(P - u, R), R - v1 hidden."""
        rows = _rows(payload, "remainder")
        assert len(rows) >= 1
        discriminating = hidden = 0
        for _, seed, _, a in rows:
            obs = json.loads(a["observed"])
            d = _derive("remainder", seed)
            assert d["R"] > 0 and d["u"] > 0, f"dataset: seed {seed} lacks R > 0, u > 0"
            assert _remainder_problems(obs, d, str(seed)) == []
            assert obs["reports"][-1][2] == d["total"] - d["R"], seed
            discriminating += d["v1"] != min(d["peak"], d["R"])  # C3 vs the rejected C1
            hidden += d["hidden"] > 0
        assert discriminating > 0 and hidden > 0

    def test_remainder_dataset_matches_its_construction_contract(self, payload):
        """Generator contract (make_iceberg_cascade docstring): 2..8 levels, P in 10..60,
        k_e in 0..29, u in 1..P-1, R in 1..3P, total >= P; every level crosses."""
        for _, seed, _, _a in _rows(payload, "remainder"):
            d = _derive("remainder", seed)
            real = _real(_draw("remainder", seed).tape)
            assert 2 <= len(real) - 1 <= 8, seed
            assert 10 <= d["peak"] <= 60 and 0 <= d["k_e"] <= 29, seed
            assert 1 <= d["u"] <= d["peak"] - 1 and 1 <= d["R"] <= 3 * d["peak"], seed
            assert d["total"] >= d["peak"], seed
            assert d["n_levels_touched"] == len(real) - 1, seed

    def test_market_spec_quotes_stamped_v2_14(self, payload):
        versions = {q.get("version") for q in payload["spec"]["quotes"] if q.get("source") == "market-spec.md"}
        assert versions == {SPEC_VERSION}
        head = (REPO_ROOT / "docs" / "omiecid" / "market-spec.md").read_text(encoding="utf-8")[:400]
        assert f"**Status:** {SPEC_VERSION}" in head

    def test_hop_bound_under_the_100_pass_ceiling(self, payload):
        rows = _per_draw(payload)
        hop = max(int(_draw(k, s).hop_bound) for k, s, _, _ in rows)
        assert hop <= PASS_CEILING
        hb = [a for a in payload["assertions"] if a["name"].startswith(f"hop bound <= {PASS_CEILING} ")]
        assert len(hb) == 1 and hb[0]["status"] == "pass" and hb[0]["observed"][0] == hop

    def test_final_book_never_crossed_on_any_draw(self, payload):
        for kind, seed, _, a in _per_draw(payload):
            assert not _crossed(json.loads(a["observed"])["final_book"]), (kind, seed)

    def test_eager_and_jitted_evidence_present_and_identical(self, payload):
        assert eager_jit_problems(payload) == []

    def test_full_battery_ran_and_passed_on_every_draw(self, payload):
        assert battery_problems(payload) == []

    def test_capacity_margins_and_no_capacity_flag(self, payload):
        assert capacity_problems(payload) == []

    def test_required_quotes_present_and_byte_exact(self, payload):
        assert quote_problems(payload) == []

    def test_no_iceberg_meta_claim_and_no_failed_oracle(self, payload):
        assert scope_problems(payload) == []


# ---------------------------------------------------------------------------
# Deliberately broken copies must be rejected
# ---------------------------------------------------------------------------
def _first(payload, kind: str, pred=lambda d: True) -> int:
    for i, a in enumerate(payload["assertions"]):
        m = _PER_DRAW_RE.match(a.get("name", ""))
        if m and m.group(3) == kind and pred(_derive(kind, int(m.group(1)))):
            return i
    pytest.skip(f"no {kind} draw matching the predicate in this run")


def _seed_of(payload, i: int) -> int:
    return int(_PER_DRAW_RE.match(payload["assertions"][i]["name"]).group(1))


def _edit(a: Dict[str, Any], fn) -> None:
    """Apply fn to both observed and expected, so the K2 expectation cannot save it."""
    for key in ("observed", "expected"):
        if isinstance(a.get(key), str):
            obj = json.loads(a[key])
            fn(obj)
            a[key] = json.dumps(obj)


def _merge_same_passive(trades: List[List[int]], ice_oid: int) -> List[List[int]]:
    """The retired r3 reading: one row per resting counter bid (no instance split)."""
    out: List[List[int]] = []
    for t in trades:
        p = t[1] if t[0] == ice_oid else t[0]
        if out and (out[-1][1] if out[-1][0] == ice_oid else out[-1][0]) == p:
            out[-1] = out[-1][:3] + [out[-1][3] + t[3], out[-1][4]]
        else:
            out.append(list(t))
    return out


class TestBrokenCopiesRejected:
    def test_sanity_real_payload_is_clean(self, payload):
        assert audit_problems(payload) == []
        assert quote_problems(payload) == []

    def test_rule3_trade_printed_at_L_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        i = _first(bad, "rule3")
        L = _derive("rule3", _seed_of(bad, i))["L"]

        def at_L(o):
            o["trades"][-1][2] = L
        _edit(bad["assertions"][i], at_L)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("aggressor's limit" in p for p in probs), probs

    def test_cascade_missing_final_slice_r_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        i = _first(bad, "cascade", lambda d: d["total"] % d["peak"] != 0)
        a = bad["assertions"][i]

        def drop_r(o):  # a self-consistent run that never activated the last slice
            r = o["trades"].pop()[3]
            for oid in o["filled"]:
                o["filled"][oid] -= r
            for k in o["remaining"]:
                o["remaining"][k] += r
            o["rows_qty"][1] -= r
            o["reports"][1][2] -= r
        _edit(a, drop_r)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("slice sizes" in p for p in probs), probs

    def test_incoming_iceberg_under_reported_fill_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        i = _first(bad, "incoming", lambda d: d["total"] > d["peak"])
        peak = _derive("incoming", _seed_of(bad, i))["peak"]

        def first_slice_only(o):  # the OMIE-ICE-INCOMING-FILL-REPORT-1 defect shape
            o["reports"][-1][2] = peak
        _edit(bad["assertions"][i], first_slice_only)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("I6(b)" in p for p in probs), probs

    def test_crossed_final_book_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        i = _first(bad, "rule3", lambda d: d["q"] % d["peak"] != 0)
        d = _derive("rule3", _seed_of(bad, i))
        agg_name = SIDE_NAME[_other(d["ice_side"])]
        a = bad["assertions"][i]
        obs = json.loads(a["observed"])
        obs["final_book"][agg_name] = [[d["L"], 1, 2]]  # aggressor remainder left through the slice
        a["observed"] = json.dumps(obs)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("crossed" in p for p in probs), probs

    def test_missing_eager_jit_evidence_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        bad["assertions"] = [a for a in bad["assertions"] if a.get("kind") != "eager_jit"]
        probs = eager_jit_problems(_roundtrip(tmp_path, bad))
        assert any("absence" in p for p in probs), probs

    @pytest.mark.parametrize("kind", INCOMING_KINDS)
    def test_eager_jit_missing_for_one_construction_rejected(self, payload, tmp_path, kind):
        bad = copy.deepcopy(payload)
        bad["assertions"] = [a for a in bad["assertions"]
                             if not (a.get("kind") == "eager_jit" and f"{kind})" in a.get("name", ""))]
        probs = eager_jit_problems(_roundtrip(tmp_path, bad))
        assert any(f"construction {kind}" in p for p in probs), probs

    @pytest.mark.parametrize("key", _INCOMING_OBS_KEYS)
    def test_truncated_per_draw_observation_rejected(self, payload, tmp_path, key):
        bad = copy.deepcopy(payload)
        i = _first(bad, "remainder")
        obs = json.loads(bad["assertions"][i]["observed"])
        del obs[key]
        bad["assertions"][i]["observed"] = json.dumps(obs)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("not present in full" in p for p in probs), probs

    @pytest.mark.parametrize("kind", ("cascade", "remainder"))
    def test_dropped_draw_rejected(self, payload, tmp_path, kind):
        bad = copy.deepcopy(payload)
        i = _first(bad, kind)
        del bad["assertions"][i]
        assert coverage_problems(_roundtrip(tmp_path, bad)) != []

    def test_all_remainder_draws_dropped_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        bad["assertions"] = [a for a in bad["assertions"] if "[remainder," not in a.get("name", "")]
        n = len(_per_draw(bad))
        bad["draws"].update(requested=n, valid=n)
        bad["seeds"]["count"] = n
        probs = coverage_problems(_roundtrip(tmp_path, bad))
        assert any("no draws" in p or "never drawn" in p for p in probs), probs

    def test_I3_battery_failure_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        for r in bad["invariants"]:
            if r["check_id"] == "I3":
                r.update(status="fail", n_pass=r["n_pass"] - 1, n_fail=1, failing_seeds=[0])
        assert any(p.startswith("I3") for p in battery_problems(_roundtrip(tmp_path, bad)))

    def test_slice_margin_breach_reported_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        for a in bad["assertions"]:
            if a["name"].startswith("largest ceil(qty/peak)"):
                a["observed"] = 66
        assert capacity_problems(_roundtrip(tmp_path, bad)) != []

    @pytest.mark.parametrize("kind", INCOMING_KINDS)
    def test_r3_one_row_per_counter_bid_shape_rejected(self, payload, tmp_path, kind):
        """The retired r3 expectation (rows merged across incoming instance boundaries,
        one row per resting counter bid) must not pass under the 2026-10-20 rule."""
        bad = copy.deepcopy(payload)
        i = _first(bad, kind, lambda d: len(set(d["passive"])) != len(d["passive"]))
        d = _derive(kind, _seed_of(bad, i))
        ice_oid = d["ice_oid"]

        def merge(o):  # same fills, same reports; rows not split at instance boundaries
            o["trades"] = _merge_same_passive(o["trades"], ice_oid)
            o["passive"] = [t[1] if t[0] == ice_oid else t[0] for t in o["trades"]]
        _edit(bad["assertions"][i], merge)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("OQ-33 revised" in p for p in probs), probs

    def test_remainder_shown_as_full_peak_C1_rejected(self, payload, tmp_path):
        """The rejected display C1 (min(P, R)) after a part-used instance must not pass."""
        bad = copy.deepcopy(payload)
        i = _first(bad, "remainder", lambda d: d["v1"] != min(d["peak"], d["R"]))
        d = _derive("remainder", _seed_of(bad, i))
        c1 = min(d["peak"], d["R"])
        ice_name = SIDE_NAME[d["ice_side"]]

        def show_c1(o):
            o["final_book"][ice_name] = [[d["p_lim"], c1, d["ice_oid"]]]
            o["incoming_rest"] = [d["p_lim"], c1, d["R"] - c1]
        a = bad["assertions"][i]
        obs = json.loads(a["observed"])
        show_c1(obs)
        a["observed"] = json.dumps(obs)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("Rule 2 Incoming (4)" in p for p in probs), probs

    def test_remainder_wrong_hidden_split_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        i = _first(bad, "remainder", lambda d: d["hidden"] > 0)
        d = _derive("remainder", _seed_of(bad, i))

        def drop_hidden(o):  # a display claiming the hidden part vanished
            o["incoming_rest"] = [d["p_lim"], d["v1"], 0]
        _edit(bad["assertions"][i], drop_hidden)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("incoming_rest" in p for p in probs), probs

    def test_remainder_cancelled_like_IOC_rejected(self, payload, tmp_path):
        """Rule 2 Incoming (5): an iceberg is never IOC; its remainder is not cancelled."""
        bad = copy.deepcopy(payload)
        i = _first(bad, "remainder")
        d = _derive("remainder", _seed_of(bad, i))
        ice_name = SIDE_NAME[d["ice_side"]]

        def cancel(o):
            o["final_book"][ice_name] = []
            o["incoming_rest"] = []
        a = bad["assertions"][i]
        obs = json.loads(a["observed"])
        cancel(obs)
        a["observed"] = json.dumps(obs)
        probs = audit_problems(_roundtrip(tmp_path, bad))
        assert any("remainder rests as []" in p for p in probs), probs

    def test_hop_bound_over_100_passes_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        for a in bad["assertions"]:
            if a["name"].startswith(f"hop bound <= {PASS_CEILING} "):
                a["observed"] = [a["observed"][0], 120]
        probs = capacity_problems(_roundtrip(tmp_path, bad))
        assert any("hop-bound" in p for p in probs), probs

    def test_stale_spec_stamp_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        for q in bad["spec"]["quotes"]:
            if q["source"] == "market-spec.md":
                q["version"] = "v2.13"
                break
        probs = quote_problems(_roundtrip(tmp_path, bad))
        assert any("v2.13" in p for p in probs), probs

    def test_retired_OQ33_ruling_quote_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        q = next(q for q in bad["spec"]["quotes"] if q.get("section") == "OQ-33")
        bad["spec"]["quotes"].append(dict(q, block="Ruling: one trade per counter bid, where each "
                                                   "instance of a resting iceberg is its own counter bid."))
        probs = quote_problems(_roundtrip(tmp_path, bad))
        assert any("retired" in p for p in probs), probs

    def test_missing_OQ34_resolution_quote_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        bad["spec"]["quotes"] = [q for q in bad["spec"]["quotes"] if q.get("section") != "OQ-34"]
        probs = quote_problems(_roundtrip(tmp_path, bad))
        assert any("OQ-34" in p for p in probs), probs

    def test_K2_expectation_admitting_a_wrong_book_rejected(self, payload, tmp_path):
        bad = copy.deepcopy(payload)
        i = _first(bad, "rule3", lambda d: d["q"] % d["peak"] != 0)
        a = bad["assertions"][i]
        exp = json.loads(a["expected"])
        exp["final_book_one_of"].append({"BUY": [], "SELL": []})
        a["expected"] = json.dumps(exp)
        probs = all_draw_problems(_roundtrip(tmp_path, bad))
        assert any("K2 admits" in p for p in probs), probs

    def test_K2_expectation_with_r3_trades_rejected(self, payload, tmp_path):
        """K2's own expectation drifting back to the r3 split is caught even if the
        observation is right."""
        bad = copy.deepcopy(payload)
        i = _first(bad, "remainder", lambda d: len(set(d["passive"])) != len(d["passive"]))
        a = bad["assertions"][i]
        d = _derive("remainder", _seed_of(bad, i))
        exp = json.loads(a["expected"])
        exp["trades"] = _merge_same_passive(exp["trades"], d["ice_oid"])
        a["expected"] = json.dumps(exp)
        probs = all_draw_problems(_roundtrip(tmp_path, bad))
        assert any("K2 expected trades" in p for p in probs), probs


# ---------------------------------------------------------------------------
# Independent engine probes: incoming iceberg vs a ladder of resting icebergs
# (ContractBook public API; Rule 2 Incoming iceberg amended 2026-10-20)
# ---------------------------------------------------------------------------
_PROBE_CID = 81
# jaxob trade rows: (P, Q, PASS_OID, AGRS_OID, ...); book rows: (P, Q, OID, ...).
_TP, _TQ, _TPASS, _TAGRS = 0, 1, 2, 3


@pytest.fixture(scope="module")
def probe_book():
    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.contract_book import ContractBook
    from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig

    c = Contract(contract_id=_PROBE_CID, delivery_start_minutes=17 * 60 + 15,
                 delivery_end_minutes=17 * 60 + 30, gct_minutes=16 * 60 + 15,
                 status=ContractStatus.OPEN)
    pc = OMIEProductConfig()
    return ContractBook(c, pc, OMIEMarketConfig()), pc


def _probe_order(pc, side, p_tick, q_tick, oid, ts, peak_tick=0):
    from omiecid_simulator.omiecid.order_types import make_order

    off = int(round(-pc.price_min / pc.price_tick))
    return make_order(
        contract_id=_PROBE_CID, side=side, price=round((p_tick - off) * pc.price_tick, 2),
        quantity=round(q_tick * pc.qty_tick, 1), order_id=oid, agent_id=oid, timestamp=ts,
        exec_restriction=ExecRestriction.ICE if peak_tick else ExecRestriction.NON,
        peak_qty=round(peak_tick * pc.qty_tick, 1) if peak_tick else 0.0, ppd=0.0)


def _probe_entries(rest) -> List[Dict[str, int]]:
    out = []
    for ts, (price, total, peak, oid) in enumerate(rest, start=1):
        vis = min(peak, total) if peak else total
        out.append({"p": price, "vis": vis, "hidden": total - vis, "peak": peak or total,
                    "oid": oid, "ts": ts})
    return out


def _probe_cases():
    """Deterministic, varied ladders of resting PPD=0 icebergs (plus one NON), both
    sides, off-round prices/OIDs, a shared price level to exercise re-ranking, and
    two hand-built cases with a remainder after a part-used instance (u > 0)."""
    rng = np.random.default_rng(20261021)
    cases = []
    for k in range(6):
        taker = BUY if k % 2 == 0 else SELL
        base = 1_000_000 + 4_000 + int(rng.integers(0, 3_000))
        n_rest = int(rng.integers(2, 5))
        steps = sorted(int(x) for x in rng.choice(np.arange(1, 40), size=n_rest, replace=False))
        if k in (2, 3):
            steps[1] = steps[0]  # two resting entries share a price level (time priority)
        rest = []
        for j, st in enumerate(steps):
            peak = int(rng.integers(2, 25))
            total = peak * int(rng.integers(1, 5)) + int(rng.integers(0, peak))
            if j == n_rest - 1 and k == 4:
                peak = 0  # one resting NON counter bid in the mix
            price = base + st if taker == BUY else base - st
            rest.append((price, total, peak, 100 + 37 * k + j))
        cap = sum(t for _, t, _, _ in rest)
        lim_idx = n_rest - 1 if k % 3 else n_rest - 2  # sometimes stop short of the deepest level
        limit = rest[lim_idx][0]
        i_peak = int(rng.integers(2, 30))
        i_total = cap - int(rng.integers(0, 4)) if k % 3 else cap + int(rng.integers(5, 40))
        cases.append((taker, limit, i_total, i_peak, 70_000 + 911 * k, rest))
    # Hand-built: resting SELL iceberg 17 (peak 5) + NON 9; incoming BUY P = 7, total 40.
    # M = 26 -> k_e = 3, u = 5, R = 14, v1 = min(2, 14) = 2 (C1 would show 7).
    b = 1_004_321
    cases.append((BUY, b + 5, 40, 7, 77_007, [(b + 3, 17, 5, 401), (b + 5, 9, 0, 402)]))
    # Mirror on the SELL side with a shallower limit stop: M = 23, P = 6 -> k_e = 3, u = 5,
    # R = 31, v1 = 1.
    b = 1_006_543
    cases.append((SELL, b - 4, 54, 6, 77_013, [(b - 2, 11, 4, 503), (b - 4, 12, 0, 504),
                                               (b - 9, 30, 8, 505)]))
    return cases


PROBE_CASES = _probe_cases()


def _engine_rows(before: np.ndarray, after: np.ndarray) -> List[Tuple[int, int, int, int]]:
    rows = after[np.any(before != after, axis=1)]
    return [(int(t[_TP]), abs(int(t[_TQ])), int(t[_TPASS]), int(t[_TAGRS])) for t in rows]


@pytest.mark.jit
class TestEngineIncomingIcebergProbes:
    """jit: ContractBook.submit_order drives the module-level jitted submit."""

    @pytest.mark.slow
    @pytest.mark.parametrize("case", range(len(PROBE_CASES)))
    def test_rows_split_at_resting_entries_and_instance_boundaries(self, probe_book, case):
        """slow: jitted submit_order compile per xdist worker (~2 s first call)."""
        book, pc = probe_book
        taker, limit, i_total, i_peak, i_oid, rest = PROBE_CASES[case]
        maker = _other(taker)
        state = book.init()
        for ts, (price, total, peak, oid) in enumerate(rest, start=1):
            state, r = book.submit_order(state, _probe_order(pc, maker, price, total, oid, ts, peak))
            assert bool(r.accepted) and not bool(r.traded)
        want, _, rem = _sweep(_probe_entries(rest), taker, limit, i_total, i_oid, 0, i_peak)
        assert len(want) < np.asarray(state.lob.trades).shape[0], "probe would saturate the trade ring"

        before = np.asarray(state.lob.trades)
        state, r = book.submit_order(
            state, _probe_order(pc, taker, limit, i_total, i_oid, len(rest) + 1, i_peak))
        got = _engine_rows(before, np.asarray(state.lob.trades))
        exp = [(t[2], t[3], t[1] if taker == BUY else t[0], i_oid) for t in want]

        # One row per (resting entry, incoming instance) pair, price-time order, at the
        # resting entry's limit (Rule 2 Incoming (2), (3); OQ-33 revised 2026-10-20).
        assert got == exp, f"engine rows {got} != derived {exp}"
        filled = i_total - rem
        assert bool(r.accepted) and bool(r.traded) == (filled > 0)
        assert float(r.trade_qty) == pytest.approx(filled * pc.qty_tick, abs=1e-4)
        assert not bool(state.truncated) and not bool(state.book_full)
        # Remainder rests as instance k_e at b showing v1 = min(P - u, R) (item (4)).
        own = np.asarray(state.lob.bids if taker == BUY else state.lob.asks)
        mine = own[(own[:, 2] == i_oid) & (own[:, 1] > 0)]
        _, _, v1, hidden = _rest_display(i_total, i_peak, rem)
        if rem > 0:
            assert mine.shape[0] == 1
            assert (int(mine[0, 0]), int(mine[0, 1])) == (limit, v1)
        else:
            assert mine.shape[0] == 0
        # I3: uncrossed on return.
        bids, asks = np.asarray(state.lob.bids), np.asarray(state.lob.asks)
        bb = bids[(bids[:, 2] > 0) & (bids[:, 1] > 0)][:, 0]
        ba = asks[(asks[:, 2] > 0) & (asks[:, 1] > 0)][:, 0]
        assert bb.size == 0 or ba.size == 0 or bb.max() < ba.min()

        if rem > 0:
            # The hidden R - v1 exists: a NON for R + 3 at b consumes v1, then slices of
            # min(P, hidden) (Rule 2 for a resting iceberg), all at b, then rests 3.
            f_oid = i_oid + 1
            before = np.asarray(state.lob.trades)
            state, r2 = book.submit_order(
                state, _probe_order(pc, maker, limit, rem + 3, f_oid, len(rest) + 2))
            got2 = _engine_rows(before, np.asarray(state.lob.trades))
            sizes = [v1] + [i_peak] * (hidden // i_peak) + ([hidden % i_peak] if hidden % i_peak else [])
            assert got2 == [(limit, q, i_oid, f_oid) for q in sizes], (got2, sizes)
            assert float(r2.trade_qty) == pytest.approx(rem * pc.qty_tick, abs=1e-4)
            own = np.asarray(state.lob.bids if taker == BUY else state.lob.asks)
            assert own[(own[:, 2] == i_oid) & (own[:, 1] > 0)].shape[0] == 0

    def test_probe_cases_discriminate_the_r4_correction(self):
        """Some probe splits a resting entry at an incoming instance boundary (the r3
        one-row-per-counter-bid shape would differ), some resting iceberg yields several
        rows (merged resting instances would differ), and some probe leaves R > 0 with
        u > 0 and v1 != min(P, R) (the C1 display would differ), with hidden > 0."""
        inst_split = multi_inst = c3 = hidden = no_rem = False
        for taker, limit, i_total, i_peak, i_oid, rest in PROBE_CASES:
            want, _, rem = _sweep(_probe_entries(rest), taker, limit, i_total, i_oid, 0, i_peak)
            pas = [t[1] if taker == BUY else t[0] for t in want]
            acc = 0
            entry_peak = {oid: (pk or tot) for _, tot, pk, oid in rest}
            for i in range(1, len(want)):
                acc += want[i - 1][3]
                if pas[i] == pas[i - 1]:
                    if acc % i_peak == 0:
                        inst_split = True
                    if entry_peak[pas[i]] < sum(t[3] for t, p in zip(want, pas) if p == pas[i]):
                        multi_inst = True
            _, u, v1, h = _rest_display(i_total, i_peak, rem)
            c3 |= rem > 0 and u > 0 and v1 != min(i_peak, rem)
            hidden |= rem > 0 and h > 0
            no_rem |= rem == 0
        assert inst_split and multi_inst and c3 and hidden and no_rem


# ---------------------------------------------------------------------------
# Slow: re-run a few seeds per construction through the engine, eager and jitted
# ---------------------------------------------------------------------------
SLOW_SEEDS = (("cascade", 0), ("cascade", 206), ("rule3", 1000000), ("rule3", 1000010),
              ("incoming", 2000000), ("incoming", 2000022),
              ("remainder", 3000000), ("remainder", 3000001), ("remainder", 3000002))


@pytest.mark.slow
@pytest.mark.jit
@pytest.mark.parametrize("kind,seed", SLOW_SEEDS)
def test_engine_rerun_matches_derivation_and_emitted_json(kind, seed, payload):
    """slow: compiles and runs the engine eager + jitted (tens of seconds per shape)."""
    from tests.omiecid.reference import omie_invariants as inv

    draw = _draw(kind, seed)
    jit_rec = inv.record_from_draw(draw, jit=True)
    eager_rec = inv.record_from_draw(draw, jit=False)
    assert inv.compare_records(jit_rec, eager_rec) == ()

    qt = float(jit_rec.product_config.qty_tick)
    obs = {
        "trades": [[f.buyer_oid, f.seller_oid, f.price_int, f.qty_int, f.submission] for f in jit_rec.fills],
        "final_book": {"BUY": [[p, q, o] for p, q, o, *_ in jit_rec.final_snapshot.resting(BUY)],
                       "SELL": [[p, q, o] for p, q, o, *_ in jit_rec.final_snapshot.resting(SELL)]},
        "reports": [[bool(r.accepted), bool(r.traded), int(round(float(r.trade_qty_mw) / qt))]
                    for r in jit_rec.engine_reports],
        "rows_qty": [sum(f.qty_int for f in jit_rec.fills if f.submission == k)
                     for k in range(jit_rec.n_submissions)],
    }
    d = _derive(kind, seed)
    assert obs["trades"] == d["trades"]
    assert _book_key(obs["final_book"]) in d["final_books"]
    assert obs["reports"] == d["reports"] and obs["rows_qty"] == d["rows_qty"]
    assert _i6b_problems(obs, f"{kind} {seed}") == []
    if kind in INCOMING_KINDS:
        passive = [t[1] if d["ice_side"] == BUY else t[0] for t in obs["trades"]]
        assert _instance_split_problems(obs["trades"], passive, d, f"{kind} {seed}") == []
    if kind == "remainder":
        mine = [r for r in jit_rec.final_snapshot.resting(d["ice_side"]) if r[2] == d["ice_oid"]]
        assert [(r[0], r[1]) for r in mine] == [(d["p_lim"], d["v1"])]
    # I3 at every return, read independently from the engine's post-submission books.
    for snap in jit_rec.post_snapshots:
        bb, ba = snap.best_bid(), snap.best_ask()
        assert bb is None or ba is None or bb < ba
        assert not snap.truncated and not snap.book_full
    if kind == "rule3" and d["q"] % d["peak"]:
        ice_side = d["ice_side"]
        first = [r for r in jit_rec.post_snapshots[0].resting(ice_side) if r[2] == 1]
        last = [r for r in jit_rec.final_snapshot.resting(ice_side) if r[2] == 1]
        assert first and last and (last[0][4], last[0][5]) > (first[0][4], first[0][5])
    report = inv.run_battery(jit_rec, expects_trade=True)
    assert report.passed and report.valid, report.statuses()

    # The emitted JSON row for this seed reflects this engine run.
    rows = {(k, s): a for k, s, _, a in _per_draw(payload)}
    a = rows.get((kind, seed))
    assert a is not None, f"seed {seed} absent from the emitted JSON"
    emitted = json.loads(a["observed"])
    assert emitted["trades"] == obs["trades"]
    assert _book_key(emitted["final_book"]) == _book_key(obs["final_book"])
    assert emitted["rows_qty"] == obs["rows_qty"]
