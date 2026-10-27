"""Audit tests for OMIE-CERT-SYNTH-1 — the synthetic tape generator.

Mode: audit (post-implementation, fresh context). Written independently from the
story card (``docs/omiecid/agile-plan.md`` §"Story OMIE-CERT-SYNTH-1", decisions
S1–S14), ``docs/omiecid/scenario-ladder.md`` and ``docs/omiecid/scenario-ladder-plan.md``.
The original locked test files were not read.

Revision (S15, v5.184): adds ``C03.price_over_time`` to the audit key set and
CHECKS, and retargets the C03 price-vs-time discrimination test to that variant.

Every expectation here is re-derived from the emitted tape and ``params`` with the
audit's own arithmetic — including a small reference price-time matcher for the
iceberg-free constructions — never by calling generator helpers. Each re-derivation
check is also seen failing on a deliberately mutated draw (plan rule 1).

Fast suite: pure numpy. Slow suite (``@pytest.mark.slow``): engine replays on CPU
through ``ContractBook`` (JIT warm-up) and fresh-interpreter subprocesses.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import omiecid_simulator.jaxomiecid.omie_data_loader as loader
import omiecid_simulator.jaxomiecid.omie_tape_synth as synth
from omiecid_simulator.omiecid.contract import ContractStatus
from omiecid_simulator.omiecid.encoding import encode_price, encode_qty
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, Status, Validity

REPO = Path(__file__).resolve().parents[2]
SYNTH_SRC = REPO / "omiecid_simulator" / "jaxomiecid" / "omie_tape_synth.py"
FIXTURE = REPO / "tests" / "omiecid" / "fixtures" / "cert" / "synth_digests.json"

CFG = OMIEProductConfig()
MKT = OMIEMarketConfig()
PRICE_MAX_INT = encode_price(CFG.price_max, CFG)
PRICE_MIN_INT = encode_price(CFG.price_min, CFG)
QTY_MIN_INT = encode_qty(CFG.qty_min, CFG)
QTY_MAX_INT = encode_qty(CFG.qty_max, CFG)

BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, IOC, FOK, ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                      int(ExecRestriction.FOK), int(ExecRestriction.ICE))

# The loader's column order, written out by the audit (S10).
C_CONTRACT, C_SIDE, C_PRICE, C_QTY, C_OID, C_AGENT, C_TS = 0, 1, 2, 3, 4, 5, 6
C_EXEC, C_VALIDITY, C_EXPIRY, C_STATUS, C_PEAK, C_PPD = 7, 8, 9, 10, 11, 12

S1_KEYS = {
    "C00.empty", "C00.pad_only", "C00.status_gate", "C01.validation",
    "C02.crossing_pair", "C03.priority_ladder", "C03.same_owner_pair",
    "C04.cancellation", "C04.iceberg_cancel_fok", "C05.aggressor_residual",
    "C05.resting_residual", "C06.multi_level_walk", "C06.multi_level_walk_stress",
    "C07.ioc_regimes", "C08.iceberg_cascade", "C08.rule3", "C08.incoming_iceberg",
    "C09.icewpi_drift", "C09.icewpi_wrong_sign", "C10.fok_visible", "C10.fok_hidden",
    "C10.fok_icewpi", "C15.random_session",
    "C03.price_over_time",  # S15 (v5.184)
}
AXES = ("resting_per_side", "trades", "hops", "concurrent_icebergs")
ENV_FIELD = {"resting_per_side": "max_resting_per_side", "trades": "max_trades",
             "hops": "max_hops", "concurrent_icebergs": "max_concurrent_icebergs"}
CEILING = {"max_resting_per_side": MKT.book_depth, "max_trades": MKT.nTrades,
           "max_hops": MKT.max_match_iterations,
           "max_concurrent_icebergs": MKT.max_icebergs}

EXPECTED_PAD = [0, 0, 0, 0, -1, 0, -1, NON, int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0]

def _fuzz_cell(name_prefix: str, n: int, cross: float):
    for r in synth.fuzz_grid():
        if r.name.startswith(name_prefix + "|") and r.n_orders == n and r.cross_prob == cross:
            return r
    raise LookupError((name_prefix, n, cross))


# Every mode that takes an envelope, as (seed, envelope) -> Draw (S3 signatures).
ENVELOPED_MODES = {
    "C00.status_gate": lambda s, e: synth.make_status_gate(s, envelope=e),
    "C02.crossing_pair": lambda s, e: synth.make_crossing_pair(s, envelope=e),
    "C03.priority_ladder": lambda s, e: synth.make_priority_ladder(s, envelope=e),
    "C03.price_over_time": lambda s, e: synth.make_priority_ladder(
        s, envelope=e, variant="price_over_time"),
    "C03.same_owner_pair": lambda s, e: synth.make_same_owner_pair(s, envelope=e),
    "C04.cancellation": lambda s, e: synth.make_cancellation(s, envelope=e, variant="ladder"),
    "C04.iceberg_cancel_fok": lambda s, e: synth.make_cancellation(
        s, envelope=e, variant="iceberg_fok"),
    "C05.aggressor_residual": lambda s, e: synth.make_partial_fill_residual(
        s, envelope=e, variant="aggressor"),
    "C05.resting_residual": lambda s, e: synth.make_partial_fill_residual(
        s, envelope=e, variant="resting"),
    "C06.multi_level_walk": lambda s, e: synth.make_multi_level_walk(s, envelope=e),
    "C06.multi_level_walk_stress": lambda s, e: synth.make_multi_level_walk(
        s, envelope=e, stress=True),
    "C07.ioc_regimes": lambda s, e: synth.make_ioc_regimes(s, envelope=e),
    "C08.iceberg_cascade": lambda s, e: synth.make_iceberg_cascade(s, envelope=e,
                                                                   variant="resting"),
    "C08.rule3": lambda s, e: synth.make_iceberg_cascade(s, envelope=e, variant="rule3"),
    "C08.incoming_iceberg": lambda s, e: synth.make_iceberg_cascade(s, envelope=e,
                                                                    variant="incoming"),
    "C09.icewpi_drift": lambda s, e: synth.make_icewpi_drift(s, envelope=e),
    "C09.icewpi_wrong_sign": lambda s, e: synth.make_icewpi_drift(s, envelope=e,
                                                                  wrong_sign=True),
    "C10.fok_visible": lambda s, e: synth.make_fok_boundary(s, envelope=e, variant="visible"),
    "C10.fok_hidden": lambda s, e: synth.make_fok_boundary(s, envelope=e, variant="hidden"),
    "C10.fok_icewpi": lambda s, e: synth.make_fok_boundary(s, envelope=e, variant="icewpi"),
    "C15.random_session": lambda s, e: synth.make_random_session(
        s, synth.fuzz_grid()[0], envelope=e),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def real_rows(tape):
    t = np.asarray(tape).reshape(-1, 13)
    return t[t[:, C_OID] >= 1]


def row_of(tape, oid):
    r = real_rows(tape)
    hit = r[r[:, C_OID] == oid]
    assert len(hit) == 1, f"oid {oid} appears {len(hit)} times"
    return [int(x) for x in hit[0]]


def crosses(buy_price, sell_price):
    return buy_price >= sell_price


def units(r):
    if int(r[C_EXEC]) == ICE:
        return -(-int(r[C_QTY]) // int(r[C_PEAK]))
    return 1


def leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from leaves(v)
    else:
        yield x


def mutate_first_int(expected):
    """Return a deep copy of ``expected`` with its first int leaf bumped (or first bool flipped)."""
    out = copy.deepcopy(expected)

    def walk(node):
        items = node.items() if isinstance(node, dict) else enumerate(node)
        for k, v in list(items):
            if isinstance(v, bool):
                continue
            if isinstance(v, int):
                node[k] = v + 1
                return True
            if isinstance(v, (dict, list)) and walk(v):
                return True
        for k, v in list(node.items() if isinstance(node, dict) else enumerate(node)):
            if isinstance(v, bool):
                node[k] = not v
                return True
        return False

    assert walk(out), "nothing to mutate"
    return out


def reference_match(tape, *, cancel_oid=None, cancel_before_oid=None):
    """Audit's own price-time matcher for iceberg-free tapes (market-spec §6.1/§6.2).

    Trades print at the resting price; a NON residual rests with its original arrival
    rank; IOC residuals are discarded; FOK is all-or-nothing against crossable depth.
    Returns (trades, book, verdicts) with trades as (agg_oid, rest_oid, price, qty).
    """
    book = {BUY: [], SELL: []}
    trades, verdicts = [], {}
    for seq, r in enumerate(real_rows(tape)):
        side, p, q, oid, ex = (int(r[C_SIDE]), int(r[C_PRICE]), int(r[C_QTY]),
                               int(r[C_OID]), int(r[C_EXEC]))
        if cancel_before_oid is not None and oid == cancel_before_oid:
            for s in book:
                book[s] = [o for o in book[s] if o["oid"] != cancel_oid]
        assert ex != ICE, "reference matcher is iceberg-free"
        opp = book[SELL if side == BUY else BUY]
        if side == BUY:
            cands = sorted((o for o in opp if p >= o["price"]), key=lambda o: (o["price"], o["seq"]))
        else:
            cands = sorted((o for o in opp if p <= o["price"]), key=lambda o: (-o["price"], o["seq"]))
        if ex == FOK and sum(o["qty"] for o in cands) < q:
            verdicts[oid] = {"accepted": False, "filled": 0}
            continue
        rem = q
        for o in cands:
            if rem == 0:
                break
            t = min(rem, o["qty"])
            trades.append((oid, o["oid"], o["price"], t))
            o["qty"] -= t
            rem -= t
        book[SELL if side == BUY else BUY] = [o for o in opp if o["qty"] > 0]
        verdicts[oid] = {"accepted": True, "filled": q - rem}
        if rem > 0 and ex == NON:
            book[side].append({"price": p, "seq": seq, "oid": oid, "qty": rem})
    return trades, book, verdicts


# ---------------------------------------------------------------------------
# independent closed-form re-derivations, one per registry key
# ---------------------------------------------------------------------------
def chk_empty(d):
    e = []
    if d.tape.shape != (0, 13):
        e.append(f"shape {d.tape.shape}")
    if d.expected != {"n_trades": 0, "n_resting": 0}:
        e.append(f"expected {d.expected}")
    return e


def chk_pad_only(d):
    e = []
    if d.tape.shape != (64, 13):
        e.append(f"shape {d.tape.shape}")
    if not (d.tape == np.asarray(EXPECTED_PAD)).all():
        e.append("non-pad row")
    if d.expected.get("n_trades") != 0 or d.expected.get("n_resting") != 0:
        e.append(f"expected {d.expected}")
    return e


def chk_status_gate(d):
    e = []
    base = synth.make_crossing_pair(d.seed)
    if not np.array_equal(d.tape, base.tape):
        e.append("tape != C02 tape for the seed")
    want = [int(ContractStatus.PENDING), int(ContractStatus.SUSPENDED), int(ContractStatus.CLOSED)]
    if d.params.get("contract_statuses") != want:
        e.append(f"statuses {d.params.get('contract_statuses')}")
    if d.expected.get("n_trades") != 0 or d.expected.get("n_accepted") != 0:
        e.append(f"expected {d.expected}")
    return e


def chk_validation(d):
    e = []
    if d.tape.shape[0] != 0:
        e.append("C01 tape must be empty")
    cases = {c["name"]: c for c in d.params["cases"]}
    tick = Fraction(CFG.price_tick)
    p_ok_int = encode_price(float(d.params["p_ok_eur"]), CFG)
    want = {
        "in_bounds_on_tick": (True, p_ok_int), "price_below_min": (False, None),
        "price_above_max": (False, None), "qty_below_min": (False, None),
        "qty_above_max": (False, None), "sub_half_tick_off_grid": (True, p_ok_int),
        "over_half_tick_off_grid": (True, p_ok_int + 1),
        "iceberg_peak_exceeds_qty": (False, None), "iceberg_peak_zero": (False, None),
        "near_max_sub_half_tick": (True, PRICE_MAX_INT),
        "near_max_over_half_tick": (False, None),
    }
    if set(cases) != set(want):
        e.append(f"case names {sorted(cases)}")
        return e
    ranges = {"sub_half_tick_off_grid": (0.0, 0.5), "over_half_tick_off_grid": (0.5, 1.0),
              "near_max_sub_half_tick": (0.05, 0.45), "near_max_over_half_tick": (0.55, 0.95)}
    for name, (lo, hi) in ranges.items():
        off = float(cases[name]["offset_ticks"])
        if not (lo <= off <= hi) or off == 0.5:
            e.append(f"{name} offset {off} outside [{lo},{hi}]")
    # Offsets measured exactly from the floats actually emitted.
    for name, base_eur in (("near_max_sub_half_tick", CFG.price_max),
                           ("near_max_over_half_tick", CFG.price_max)):
        exact = (Fraction(cases[name]["price_eur"]) - Fraction(base_eur)) / tick
        if name.endswith("sub_half_tick") and not exact < Fraction(1, 2):
            e.append(f"{name} exact offset {float(exact)} not < 1/2 tick")
        if name.endswith("over_half_tick") and not exact > Fraction(1, 2):
            e.append(f"{name} exact offset {float(exact)} not > 1/2 tick")
    for name, (ok, snapped) in want.items():
        if d.expected["verdicts"].get(name) is not ok:
            e.append(f"verdict {name}={d.expected['verdicts'].get(name)} want {ok}")
        if ok and d.expected["snapped_price_int"].get(name) != snapped:
            e.append(f"snapped {name}={d.expected['snapped_price_int'].get(name)} want {snapped}")
    if set(d.expected["snapped_price_int"]) != {n for n, (ok, _) in want.items() if ok}:
        e.append("snapped_price_int keys != accepted cases")
    if cases["near_max_over_half_tick"]["price_int"] != PRICE_MAX_INT + 1:
        e.append("near-max over-half case must commit to price_max + 1 tick")
    return e


def chk_crossing_pair(d):
    e = []
    rows = real_rows(d.tape)
    if len(rows) != 2:
        return [f"{len(rows)} real rows"]
    rest, agg = [int(x) for x in rows[0]], [int(x) for x in rows[1]]
    if rest[C_SIDE] == agg[C_SIDE]:
        e.append("same side")
    buy, sell = (agg, rest) if agg[C_SIDE] == BUY else (rest, agg)
    gap = abs(agg[C_PRICE] - rest[C_PRICE])
    if not crosses(buy[C_PRICE], sell[C_PRICE]) or gap < 1:
        e.append("aggressor not >= 1 tick through")
    if d.params["gap_ticks"] != gap:
        e.append("gap_ticks != tape gap")
    trades, _, _ = reference_match(d.tape)
    x = d.expected
    if x.get("n_trades") != len(trades) or len(trades) != 1:
        e.append(f"n_trades {x.get('n_trades')} vs {len(trades)}")
    if trades and (x["trade_price_int"] != rest[C_PRICE] or trades[0][2] != rest[C_PRICE]):
        e.append("trade price must be the resting price")
    if x.get("trade_qty_int") != min(rest[C_QTY], agg[C_QTY]):
        e.append("trade qty")
    if x.get("residual_qty_int") != abs(rest[C_QTY] - agg[C_QTY]):
        e.append("residual qty")
    return e


def chk_priority_ladder(d):
    e = []
    P, x = d.params, d.expected
    k, j = P["k"], P["j"]
    if not (2 <= k <= 6 and 1 <= j < k and 1 <= P["n_decoys"] <= 3):
        e.append(f"k={k} j={j} n_decoys={P['n_decoys']}")
    touch = [row_of(d.tape, o) for o in P["at_touch_oids"]]
    if len({r[C_PRICE] for r in touch}) != 1 or len({r[C_SIDE] for r in touch}) != 1:
        e.append("touch orders not at one price/side")
    ts = [r[C_TS] for r in touch]
    if ts != sorted(ts) or len(set(ts)) != len(ts):
        e.append("touch timestamps not strictly increasing")
    p_touch = touch[0][C_PRICE]
    for o in P["decoy_oids"]:
        r = row_of(d.tape, o)
        worse = r[C_PRICE] > p_touch if r[C_SIDE] == SELL else r[C_PRICE] < p_touch
        if not worse:
            e.append(f"decoy {o} not worse-priced")
    trades, _, _ = reference_match(d.tape)
    filled = [t[1] for t in trades]
    if x.get("filled_oids") != filled:
        e.append(f"filled {x.get('filled_oids')} vs reference {filled}")
    if x.get("filled_oids") != P["at_touch_oids"][:j]:
        e.append("filled set is not the j earliest")
    if sorted(x.get("untouched_oids", [])) != sorted(set(P["at_touch_oids"][j:]) | set(P["decoy_oids"])):
        e.append("untouched set")
    if x.get("n_trades") != len(trades) or x.get("total_qty_int") != sum(t[3] for t in trades):
        e.append("n_trades / total qty")
    if any(t[2] != x.get("trade_price_int") for t in trades):
        e.append("trade price")
    return e


def chk_price_over_time(d):
    """S15: worse-priced decoys arrive *before* the touch and are reachable by the aggressor."""
    e = []
    P, x = d.params, d.expected
    k, j, ne, s = P["k"], P["j"], P["n_early"], P["step_ticks"]
    if not (2 <= k <= 6 and 1 <= j < k and 1 <= ne <= 3 and 1 <= s <= 50):
        e.append(f"k={k} j={j} n_early={ne} step_ticks={s}")
    rows = [[int(v) for v in r] for r in real_rows(d.tape)]
    order = P["early_decoy_oids"] + P["at_touch_oids"] + [P["agg_oid"]]
    if [r[C_OID] for r in rows] != order:
        return e + [f"submission order {[r[C_OID] for r in rows]} != {order}"]
    touch = [row_of(d.tape, o) for o in P["at_touch_oids"]]
    early = [row_of(d.tape, o) for o in P["early_decoy_oids"]]
    agg = row_of(d.tape, P["agg_oid"])
    p_touch = touch[0][C_PRICE]
    if {r[C_PRICE] for r in touch} != {p_touch} or len({r[C_SIDE] for r in touch + early}) != 1:
        e.append("touch orders not at one price / rest orders not on one side")
    rest_side = touch[0][C_SIDE]
    sgn = 1 if rest_side == SELL else -1
    want = [p_touch + sgn * dd * s for dd in range(1, ne + 1)]
    if [r[C_PRICE] for r in early] != want or P["decoy_price_ints"] != want:
        e.append(f"decoy prices {[r[C_PRICE] for r in early]} want {want}")
    if agg[C_SIDE] == rest_side or agg[C_PRICE] != want[-1]:
        e.append("aggressor must take from the rest side at the deepest decoy price")
    if agg[C_QTY] != j * touch[0][C_QTY] or any(r[C_QTY] != touch[0][C_QTY] for r in early):
        e.append("quantities")
    if max(r[C_TS] for r in early) >= min(r[C_TS] for r in touch):
        e.append("an early decoy is not earlier than every touch order")
    trades, _, _ = reference_match(d.tape)
    filled = [t[1] for t in trades]
    if x.get("filled_oids") != filled or filled != P["at_touch_oids"][:j]:
        e.append(f"filled {x.get('filled_oids')} vs reference {filled}")
    if x.get("untouched_oids") != P["at_touch_oids"][j:] + P["early_decoy_oids"]:
        e.append("untouched set")
    if x.get("n_trades") != len(trades) or x.get("total_qty_int") != sum(t[3] for t in trades):
        e.append("n_trades / total qty")
    if x.get("trade_price_int") != p_touch or any(t[2] != p_touch for t in trades):
        e.append("trade price")
    return e


def chk_same_owner(d):
    e = []
    rows = real_rows(d.tape)
    if len({int(a) for a in rows[:, C_AGENT]}) != 1:
        e.append("owners differ")
    cf = d.aux_tapes.get("counterfactual")
    if cf is None:
        return e + ["no counterfactual"]
    if not np.array_equal(cf, synth.make_crossing_pair(d.seed).tape):
        e.append("counterfactual != C02 tape")
    diff = np.argwhere(cf != d.tape)
    if len(diff) and set(diff[:, 1].tolist()) != {C_AGENT}:
        e.append("tape differs from counterfactual outside the agent column")
    if len(set(real_rows(cf)[:, C_AGENT].tolist())) < 2:
        e.append("counterfactual owners not distinct")
    t1 = [(a, r, p, q) for a, r, p, q in reference_match(d.tape)[0]]
    t2 = [(a, r, p, q) for a, r, p, q in reference_match(cf)[0]]
    if t1 != t2 or not t1:
        e.append("M6: trade lists differ or empty")
    if d.expected.get("must_trade") is not True:
        e.append("must_trade")
    e += [m for m in chk_crossing_pair(dataclasses.replace(d, tape=cf)) if "gap" not in m]
    return e


def chk_cancellation(d):
    e = []
    P, x = d.params, d.expected
    k, j = P["k"], P["j"]
    if not (2 <= k <= 5 and 1 <= j <= k - 1 and 1 <= P["n_decoys"] <= 3):
        e.append(f"k={k} j={j} n_decoys={P['n_decoys']}")
    if P["cancel_oid"] not in P["resting_oids"]:
        e.append("cancel_oid not a resting order")
    agg = row_of(d.tape, P["agg_oid"])
    if real_rows(d.tape)[-1][C_OID] != P["agg_oid"]:
        e.append("aggressor is not the last row")
    for o in P["decoy_oids"]:
        r = row_of(d.tape, o)
        worse = r[C_PRICE] > P["p_touch_int"] if r[C_SIDE] == SELL else r[C_PRICE] < P["p_touch_int"]
        if not worse:
            e.append("decoy not worse-priced")
    trades, _, _ = reference_match(d.tape, cancel_oid=P["cancel_oid"],
                                   cancel_before_oid=P["agg_oid"])
    filled = [t[1] for t in trades]
    survivors = [o for o in P["resting_oids"] if o != P["cancel_oid"]]
    if x.get("survivor_oids_in_queue_order") != survivors:
        e.append("survivors")
    if x.get("filled_oids") != filled or filled != survivors[:j]:
        e.append(f"filled {x.get('filled_oids')} vs reference {filled}")
    if P["cancel_oid"] in filled:
        e.append("cancelled order filled")
    if x.get("n_trades") != len(trades) or x.get("trades_from_cancel") != 0:
        e.append("n_trades / trades_from_cancel")
    if any(t[2] != x.get("trade_price_int") for t in trades):
        e.append("trade price")
    if agg[C_QTY] != j * P["q_each_int"]:
        e.append("aggressor not sized for j")
    return e


def chk_iceberg_cancel_fok(d):
    e = []
    P, x = d.params, d.expected
    ice = row_of(d.tape, P["cancel_oid"])
    if ice[C_EXEC] != ICE or P["cancel_oid"] != P["ice_oid"]:
        e.append("cancel_oid is not the iceberg")
    n = -(-ice[C_QTY] // ice[C_PEAK])
    if n < 2:
        e.append("iceberg needs >= 2 slices")
    vis = [row_of(d.tape, o) for o in P["visible_oids"]]
    if not 0 <= len(vis) <= 3:
        e.append("n_vis")
    if any(r[C_PRICE] != ice[C_PRICE] or r[C_SIDE] != ice[C_SIDE] or r[C_EXEC] != NON for r in vis):
        e.append("visible orders not NON at the iceberg's price/side")
    v_vis = sum(r[C_QTY] for r in vis)
    fok = row_of(d.tape, P["fok_oid"])
    if fok[C_EXEC] != FOK:
        e.append("not a FOK")
    if not (v_vis + 1 <= fok[C_QTY] <= v_vis + ice[C_QTY]):
        e.append(f"req {fok[C_QTY]} outside [V_vis+1, V_vis+total]=[{v_vis + 1},{v_vis + ice[C_QTY]}]")
    buy, sell = (fok, ice) if fok[C_SIDE] == BUY else (ice, fok)
    if not buy[C_PRICE] > sell[C_PRICE]:
        e.append("FOK not priced through")
    # After the cancel only the visible NON orders remain; the audit matcher decides.
    no_ice = d.tape[d.tape[:, C_OID] != P["ice_oid"]]
    _, _, verdicts = reference_match(no_ice)
    v = verdicts[P["fok_oid"]]
    if x.get("accepted") is not v["accepted"] or x.get("filled_qty_int") != v["filled"]:
        e.append(f"expected {x} vs reference {v}")
    if x.get("book_unchanged_by_fok") is not True:
        e.append("book_unchanged_by_fok")
    return e


def _chk_residual(d, variant):
    e = []
    P, x = d.params, d.expected
    trades, book, _ = reference_match(d.tape)
    res = P["residual_oid"]
    take_trades = [t for t in trades if t[0] == P["take_oid"]]
    if [t[1] for t in take_trades] != [res]:
        e.append(f"take fills {[t[1] for t in take_trades]}, want only the residual {res}")
    if any(t[1] == P["interloper_oid"] or t[0] == P["interloper_oid"] for t in trades):
        e.append("interloper filled")
    resid_row = row_of(d.tape, res)
    inter = row_of(d.tape, P["interloper_oid"])
    if inter[C_SIDE] != resid_row[C_SIDE] or inter[C_TS] <= resid_row[C_TS]:
        e.append("interloper must be later, same side as the residual")
    first = [t for t in trades if res in (t[0], t[1])][0]
    if x.get("first_fill_qty_int") != first[3]:
        e.append("first fill qty")
    second_qty = sum(t[3] for t in take_trades)
    if x.get("residual_qty_int") != second_qty or row_of(d.tape, P["take_oid"])[C_QTY] != second_qty:
        e.append("residual qty / take size")
    if x.get("residual_timestamp") != resid_row[C_TS]:
        e.append("residual timestamp must be the original")
    if x.get("second_fill_oid") != res or x.get("interloper_must_remain_unfilled") is not True:
        e.append("second_fill_oid / interloper flag")
    if variant == "aggressor":
        if inter[C_PRICE] != resid_row[C_PRICE] or x.get("residual_price_int") != resid_row[C_PRICE]:
            e.append("residual price is the aggressor's limit")
        if not resid_row[C_QTY] > row_of(d.tape, P["rest_oid"])[C_QTY]:
            e.append("aggressor must exceed the crossing depth")
    else:
        if inter[C_PRICE] != resid_row[C_PRICE] or x.get("residual_price_int") != resid_row[C_PRICE]:
            e.append("interloper/residual price")
        if not row_of(d.tape, P["agg_oid"])[C_QTY] < resid_row[C_QTY]:
            e.append("first aggressor must be smaller than the resting order")
    return e


def chk_multi_level_walk(d, stress=False):
    e = []
    P, x = d.params, d.expected
    L, m = P["L"], P["m"]
    if stress:
        if x["weighted_price_int"] <= 2**31 - 1:
            e.append("stress draw below the int32 product boundary")
    elif not (3 <= L <= 12 and 1 <= m < L):
        e.append(f"L={L} m={m}")
    rest = [row_of(d.tape, o) for o in P["resting_oids"]]
    prices = [r[C_PRICE] for r in rest]
    side = rest[0][C_SIDE]
    mono = all((b > a) if side == SELL else (b < a) for a, b in zip(prices, prices[1:]))
    if not mono:
        e.append("levels not strictly deeper")
    trades, _, _ = reference_match(d.tape)
    fills = [[t[2], t[3]] for t in trades if t[0] == P["agg_oid"]]
    if x.get("per_level_fills") != fills:
        e.append(f"fills {x.get('per_level_fills')} vs reference {fills}")
    if x.get("num_fills") != len(fills) or x.get("total_qty_int") != sum(q for _, q in fills):
        e.append("num_fills / total")
    if x.get("weighted_price_int") != sum(p * q for p, q in fills):
        e.append("weighted numerator")
    whole = sum(1 for (p, q), r in zip(fills, rest) if q == r[C_QTY])
    if not stress and whole != m:
        e.append(f"{whole} whole levels swept, m={m}")
    return e


def chk_ioc(d):
    e = []
    P, x = d.params, d.expected
    if P["regime"] not in ("none", "partial", "full"):
        e.append("regime")
    agg = row_of(d.tape, P["agg_oid"])
    if agg[C_EXEC] != IOC:
        e.append("aggressor not IOC")
    trades, book, _ = reference_match(d.tape)
    traded = sum(t[3] for t in trades)
    want = {"none": (0, 0), "partial": (P["q_rest_int"], 1), "full": (agg[C_QTY], 1)}[P["regime"]]
    if (traded, len(trades)) != want:
        e.append(f"regime {P['regime']} reference ({traded},{len(trades)}) != formula {want}")
    if x.get("traded_qty_int") != traded or x.get("n_trades") != len(trades):
        e.append("expected traded/n_trades")
    if any(o["oid"] == P["agg_oid"] for s in book for o in book[s]):
        e.append("IOC residual rests")
    if x.get("ioc_residual_rests") is not False:
        e.append("ioc_residual_rests")
    cf = d.aux_tapes["counterfactual"]
    diff = np.argwhere(cf != d.tape)
    if set(map(tuple, diff.tolist())) != {(1, C_EXEC)} or row_of(cf, P["agg_oid"])[C_EXEC] != NON:
        e.append("counterfactual must differ only in the aggressor's restriction (NON)")
    if reference_match(cf)[0] != trades:
        e.append("M3: trade lists differ")
    return e


def _partition(q, peak):
    full, part = divmod(q, peak)
    return [peak] * full + ([part] if part else [])


def chk_iceberg_cascade(d):
    e = []
    P, x = d.params, d.expected
    ice, agg = row_of(d.tape, P["ice_oid"]), row_of(d.tape, P["agg_oid"])
    n = -(-ice[C_QTY] // ice[C_PEAK])
    if not n >= 2:
        e.append("n_slices < 2")
    if agg[C_QTY] != ice[C_QTY]:
        e.append("aggressor does not consume the lot")
    buy, sell = (agg, ice) if agg[C_SIDE] == BUY else (ice, agg)
    if not buy[C_PRICE] > sell[C_PRICE]:
        e.append("aggressor not through")
    if x.get("slice_sizes_int") != _partition(ice[C_QTY], ice[C_PEAK]):
        e.append("slice sizes")
    if x.get("n_slices_consumed") != n or x.get("trade_price_int") != ice[C_PRICE]:
        e.append("n_slices / price")
    return e


def chk_rule3(d):
    e = []
    P, x = d.params, d.expected
    ice, agg = row_of(d.tape, P["ice_oid"]), row_of(d.tape, P["agg_oid"])
    total, peak, q = ice[C_QTY], ice[C_PEAK], agg[C_QTY]
    n = -(-total // peak)
    if not n >= 3:
        e.append(f"n_slices={n} < 3")
    if not (peak < q <= total - peak):
        e.append(f"q_agg={q} outside (peak, total-peak]")
    through = (agg[C_PRICE] - ice[C_PRICE]) if agg[C_SIDE] == BUY else (ice[C_PRICE] - agg[C_PRICE])
    if through < 1:
        e.append("limit not >= 1 tick through")
    if x.get("slice_sizes_int") != _partition(q, peak):
        e.append("slice partition of q_agg")
    if x.get("filled_qty_int") != q or x.get("first_trade_price_int") != ice[C_PRICE]:
        e.append("filled / first price")
    if x.get("aggressor_rests") is not False or x.get("iceberg_remaining_int") != total - q:
        e.append("aggressor_rests / remainder")
    if "rule3_hop_trade_prices:OQ-31" not in d.pending:
        e.append("hop prices not withheld")
    if any(k for k in x if "hop" in k and "price" in k):
        e.append("hop prices asserted despite OQ-31")
    return e


def chk_incoming(d):
    e = []
    P, x = d.params, d.expected
    rows = real_rows(d.tape)
    ice = [int(v) for v in rows[-1]]
    levels = [[int(v) for v in r] for r in rows[:-1]]
    L = len(levels)
    if ice[C_EXEC] != ICE or not 2 <= L <= 8 or any(r[C_EXEC] != NON for r in levels):
        e.append("shape of the draw")
    rest_side = levels[0][C_SIDE]
    prices = [r[C_PRICE] for r in levels]
    if not all((b > a) if rest_side == SELL else (b < a) for a, b in zip(prices, prices[1:])):
        e.append("levels not strictly deeper")
    total, peak = ice[C_QTY], ice[C_PEAK]
    depth = sum(r[C_QTY] for r in levels)
    if depth < total + 1:
        e.append("depth < total + 1 tick")
    cum, need = 0, None
    for i, r in enumerate(levels):
        cum += r[C_QTY]
        if cum >= total:
            need = i
            break
    if ice[C_PRICE] != prices[need]:
        e.append(f"iceberg limit {ice[C_PRICE]} != deepest needed level {prices[need]}")
    # audit's own merge
    sl = _partition(total, peak)
    lv = [r[C_QTY] for r in levels]
    out, a, b = [], 0, 0
    while a < len(sl) and b < len(lv):
        t = min(sl[a], lv[b])
        out.append([prices[b], t])
        sl[a] -= t
        lv[b] -= t
        a += sl[a] == 0
        b += lv[b] == 0
    if x.get("trades_int") != out:
        e.append("trade list != slice/level merge")
    if x.get("reported_qty_int") != total or x.get("iceberg_rests") is not False:
        e.append("reported qty / rests")
    n = -(-total // peak)
    if d.capacity_bounds["trades"] != n + L - 1 or d.hop_bound != n + L + 2:
        e.append("S11(d) bound formulas")
    return e


def chk_icewpi(d):
    e = []
    P, x = d.params, d.expected
    ice, agg = row_of(d.tape, P["ice_oid"]), row_of(d.tape, P["agg_oid"])
    base, ppd, peak, total = ice[C_PRICE], ice[C_PPD], ice[C_PEAK], ice[C_QTY]
    n = total // peak
    if total % peak or n < 3:
        e.append("total must be n*peak with n >= 3")
    if ppd == 0 or (ice[C_SIDE] == SELL) != (ppd > 0):
        e.append(f"ppd sign rule violated (side {ice[C_SIDE]}, ppd {ppd})")
    m = P["m"]
    if not 1 <= m <= n - 2:
        e.append(f"m={m} outside [1, n-2]")
    e_ = agg[C_QTY] - m * peak
    if not 1 <= e_ <= peak - 1:
        e.append(f"e={e_} outside [1, peak-1]")
    if agg[C_PRICE] != base + (m - 1) * ppd:
        e.append("limit != base + (m-1) ppd")
    reach = [k for k in range(n)
             if (crosses(agg[C_PRICE], base + k * ppd) if agg[C_SIDE] == BUY
                 else crosses(base + k * ppd, agg[C_PRICE]))]
    if reach != list(range(m)):
        e.append(f"crossing slices {reach} != first m={m}")
    want = {
        "trade_price_sequence_int": [base + i * ppd for i in range(m)], "stop_index": m,
        "n_trades": m, "iceberg_rests_with_hidden": True, "residual_qty_int": e_,
        "residual_price_int": agg[C_PRICE], "hidden_remaining_int": (n - m - 1) * peak,
    }
    for k, v in want.items():
        if x.get(k) != v:
            e.append(f"{k}={x.get(k)} want {v}")
    if want["hidden_remaining_int"] <= 0:
        e.append("hidden remaining not positive")
    return e


def chk_wrong_sign(d):
    e = []
    drift = synth.make_icewpi_drift(d.seed)
    diff = np.argwhere(drift.tape != d.tape)
    if set(diff[:, 1].tolist()) != {C_PPD}:
        e.append("differs from the drift draw outside the ppd column")
    if not np.array_equal(d.tape[:, C_PPD], -drift.tape[:, C_PPD]):
        e.append("ppd not sign-flipped")
    ice = row_of(d.tape, d.params["ice_oid"])
    if (ice[C_SIDE] == SELL) == (ice[C_PPD] > 0):
        e.append("wrong-sign draw respects the sign rule")
    if d.expected != {} or d.oracle_kind != "observation":
        e.append("observation draw must carry no expectation")
    return e


def chk_fok_visible(d):
    e = []
    P, x = d.params, d.expected
    rows = real_rows(d.tape)
    fok = [int(v) for v in rows[-1]]
    lv = [[int(v) for v in r] for r in rows[:-1]]
    V = sum(r[C_QTY] for r in lv)
    delta = fok[C_QTY] - V
    if delta not in (-1, 0, 1) or fok[C_EXEC] != FOK:
        e.append(f"delta {delta}")
    _, _, verdicts = reference_match(d.tape)
    v = verdicts[fok[C_OID]]
    if v["accepted"] != (delta <= 0):
        e.append("reference: FOK reaches the whole ladder?")
    if x.get("accepted") is not (delta <= 0) or x.get("filled_qty_int") != (fok[C_QTY] if delta <= 0 else 0):
        e.append(f"expected {x} vs delta {delta}")
    if x.get("matchable_qty_int") != V:
        e.append("matchable")
    return e


def chk_fok_hidden(d):
    e = []
    P, x = d.params, d.expected
    ice, fok = row_of(d.tape, P["ice_oid"]), row_of(d.tape, P["fok_oid"])
    total, peak, req = ice[C_QTY], ice[C_PEAK], fok[C_QTY]
    if not req > peak:
        e.append("req must exceed the visible peak")
    if not (peak + 1 <= req <= total + 1):
        e.append("req range")
    buy, sell = (fok, ice) if fok[C_SIDE] == BUY else (ice, fok)
    if not buy[C_PRICE] > sell[C_PRICE]:
        e.append("FOK not through")
    ok = req <= total
    if x.get("accepted") is not ok or x.get("filled_qty_int") != (req if ok else 0) \
            or x.get("matchable_qty_int") != total:
        e.append(f"expected {x}")
    return e


def chk_fok_icewpi(d):
    e = []
    P, x = d.params, d.expected
    ice, fok = row_of(d.tape, P["ice_oid"]), row_of(d.tape, P["fok_oid"])
    base, ppd, peak, total = ice[C_PRICE], ice[C_PPD], ice[C_PEAK], ice[C_QTY]
    n = -(-total // peak)
    if ppd == 0 or (ice[C_SIDE] == SELL) != (ppd > 0):
        e.append("ppd sign rule")
    cnt = sum(1 for k in range(n)
              if (crosses(fok[C_PRICE], base + k * ppd) if fok[C_SIDE] == BUY
                  else crosses(base + k * ppd, fok[C_PRICE])))
    matchable = min(total, peak * cnt)
    if not 1 <= cnt <= n - 1:
        e.append(f"reachable slices {cnt} not in [1, n-1]")
    if fok[C_QTY] not in (matchable, matchable + 1):
        e.append("req not at the boundary")
    ok = fok[C_QTY] <= matchable
    if x.get("accepted") is not ok or x.get("matchable_qty_int") != matchable \
            or x.get("filled_qty_int") != (fok[C_QTY] if ok else 0):
        e.append(f"expected {x} vs matchable {matchable}")
    return e


def chk_random_session(d, envelope=None):
    env = envelope or synth.DEFAULT_ENVELOPE
    e = []
    rows = real_rows(d.tape)
    if len(rows) != d.params["n_orders"]:
        e.append("session length")
    g = np.diff(rows[:, C_TS])
    if (g < 0).any() or rows[0][C_TS] < 1:
        e.append("timestamps")
    if d.params["tie_density"] == 0.0 and (g < 1).any():
        e.append("a tie with tie_density 0")
    for r in rows:
        if int(r[C_EXEC]) == ICE:
            if not 0 < r[C_PEAK] <= r[C_QTY]:
                e.append("invalid iceberg peak")
            if r[C_PPD] and ((r[C_SIDE] == SELL) != (r[C_PPD] > 0)):
                e.append("C15 ppd sign rule")
            if r[C_PPD] and not d.params["icewpi"]:
                e.append("ppd outside the ICEwPI mix")
        elif r[C_PEAK] or r[C_PPD]:
            e.append("peak/ppd on a non-iceberg")
    sf = d.params["safe_from_index"]
    u = [units(r) for r in rows]
    if sf is None:
        if sum(u) > env.max_trades:
            e.append("unit sum exceeds max_trades without safe mode")
    else:
        if not sum(u[:sf]) <= env.max_trades < sum(u[:sf + 1]):
            e.append("safe-mode switch point")
        asks = rows[rows[:, C_SIDE] == SELL][:, C_PRICE]
        bids = rows[rows[:, C_SIDE] == BUY][:, C_PRICE]
        for r in rows[sf:]:
            if r[C_SIDE] == BUY and len(asks) and r[C_PRICE] >= asks.min():
                e.append("safe buy can trade")
                break
            if r[C_SIDE] == SELL and len(bids) and r[C_PRICE] <= bids.max():
                e.append("safe sell can trade")
                break
    rests = [int(r[C_SIDE]) for r in rows if int(r[C_EXEC]) in (NON, ICE)]
    if max(rests.count(BUY), rests.count(SELL)) > env.max_resting_per_side:
        e.append("resting cap")
    return e


CHECKS = {
    "C00.empty": chk_empty, "C00.pad_only": chk_pad_only, "C00.status_gate": chk_status_gate,
    "C01.validation": chk_validation, "C02.crossing_pair": chk_crossing_pair,
    "C03.priority_ladder": chk_priority_ladder, "C03.same_owner_pair": chk_same_owner,
    "C03.price_over_time": chk_price_over_time,
    "C04.cancellation": chk_cancellation, "C04.iceberg_cancel_fok": chk_iceberg_cancel_fok,
    "C05.aggressor_residual": lambda d: _chk_residual(d, "aggressor"),
    "C05.resting_residual": lambda d: _chk_residual(d, "resting"),
    "C06.multi_level_walk": chk_multi_level_walk,
    "C06.multi_level_walk_stress": lambda d: chk_multi_level_walk(d, stress=True),
    "C07.ioc_regimes": chk_ioc, "C08.iceberg_cascade": chk_iceberg_cascade,
    "C08.rule3": chk_rule3, "C08.incoming_iceberg": chk_incoming,
    "C09.icewpi_drift": chk_icewpi, "C09.icewpi_wrong_sign": chk_wrong_sign,
    "C10.fok_visible": chk_fok_visible, "C10.fok_hidden": chk_fok_hidden,
    "C10.fok_icewpi": chk_fok_icewpi, "C15.random_session": chk_random_session,
}


# ===========================================================================
# 1. Registry and public surface (S1, S3)
# ===========================================================================
class TestRegistry:
    def test_registry_keys_are_exactly_the_s1_set(self):
        assert set(synth.CONSTRUCTIONS) == S1_KEYS
        assert set(CHECKS) == S1_KEYS

    def test_registry_is_read_only(self):
        with pytest.raises(TypeError):
            synth.CONSTRUCTIONS["C99.new"] = lambda s: None  # type: ignore[index]

    @pytest.mark.parametrize("key", sorted(S1_KEYS))
    def test_draw_construction_names_its_registry_key(self, key):
        assert synth.CONSTRUCTIONS[key](3).construction == key

    @pytest.mark.parametrize("key", ["C00.empty", "C00.pad_only"])
    def test_seedless_modes_ignore_the_seed(self, key):
        a, b = synth.CONSTRUCTIONS[key](0), synth.CONSTRUCTIONS[key](12345)
        assert a.digest() == b.digest() and a.seed is None

    @pytest.mark.parametrize("key", sorted(S1_KEYS - {"C00.empty", "C00.pad_only"}))
    def test_seeded_modes_vary_with_the_seed(self, key):
        ds = {synth.CONSTRUCTIONS[key](s).digest() for s in range(6)}
        assert len(ds) == 6

    def test_column_constants_follow_the_loader_order(self):
        assert synth.N_COLS == 13
        assert (synth.COL_CONTRACT, synth.COL_SIDE, synth.COL_PRICE, synth.COL_QTY,
                synth.COL_OID, synth.COL_AGENT, synth.COL_TS, synth.COL_EXEC,
                synth.COL_VALIDITY, synth.COL_EXPIRY, synth.COL_STATUS, synth.COL_PEAK,
                synth.COL_PPD) == tuple(range(13))

    @pytest.mark.parametrize("call", [
        lambda: synth.make_cancellation(0, variant="bogus"),
        lambda: synth.make_priority_ladder(0, variant="bogus"),
        lambda: synth.make_partial_fill_residual(0, variant="bogus"),
        lambda: synth.make_iceberg_cascade(0, variant="bogus"),
        lambda: synth.make_fok_boundary(0, variant="bogus"),
        lambda: synth.make_ioc_regimes(0, regime="bogus"),
    ])
    def test_unknown_variant_raises_value_error(self, call):
        with pytest.raises(ValueError):
            call()

    @pytest.mark.parametrize("seed", range(6))
    def test_explicit_ioc_regime_matching_the_drawn_one_is_byte_identical(self, seed):
        d = synth.make_ioc_regimes(seed)
        assert synth.make_ioc_regimes(seed, regime=d.params["regime"]).digest() == d.digest()

    @pytest.mark.parametrize("regime", ["none", "partial", "full"])
    def test_explicit_ioc_regime_keeps_the_rest_of_the_stream(self, regime):
        d0 = synth.make_ioc_regimes(5)
        d = synth.make_ioc_regimes(5, regime=regime)
        assert d.params["regime"] == regime
        assert (d.params["rest_side"], d.params["p_rest_int"], d.params["q_rest_int"]) == \
            (d0.params["rest_side"], d0.params["p_rest_int"], d0.params["q_rest_int"])
        assert chk_ioc(d) == []


# ===========================================================================
# 2. Layout (S10) and the pad sentinel (S9)
# ===========================================================================
class TestLayoutAndPad:
    def test_pad_row_is_the_loaders_object_and_value(self):
        assert loader._PAD_ORDER_ROW is loader.PAD_ORDER_ROW
        assert synth.PAD_ORDER_ROW is loader.PAD_ORDER_ROW
        assert list(loader.PAD_ORDER_ROW) == EXPECTED_PAD
        assert isinstance(loader.PAD_ORDER_ROW, list) and all(
            type(v) is int for v in loader.PAD_ORDER_ROW)

    def test_pad_rows_are_built_from_the_live_loader_object(self):
        """A changed loader sentinel must flow into every generated pad row (no copy)."""
        saved = loader.PAD_ORDER_ROW[C_AGENT]
        loader.PAD_ORDER_ROW[C_AGENT] = 4242
        try:
            t = synth.make_pad_only(3).tape
            t2 = synth.make_crossing_pair(0).tape
        finally:
            loader.PAD_ORDER_ROW[C_AGENT] = saved
        assert (t[:, C_AGENT] == 4242).all()
        assert (t2[real_rows(t2).shape[0]:, C_AGENT] == 4242).all()
        assert list(loader.PAD_ORDER_ROW) == EXPECTED_PAD

    @pytest.mark.parametrize("key", sorted(S1_KEYS))
    def test_layout_invariants_over_many_seeds(self, key):
        for seed in range(12):
            d = synth.CONSTRUCTIONS[key](seed)
            for name, t in [("tape", d.tape), *d.aux_tapes.items()]:
                assert t.dtype == np.int32 and t.ndim == 2 and t.shape[1] == 13, name
                assert t.flags["C_CONTIGUOUS"], name
                n = real_rows(t).shape[0]
                assert (t[:n, C_OID] >= 1).all(), f"{name}: real rows must come first"
                assert (t[n:] == np.asarray(EXPECTED_PAD, dtype=np.int32)).all(), \
                    f"{name}: every non-real row must be the pad sentinel"
                r = t[:n]
                if n == 0:
                    continue
                assert len(set(r[:, C_OID].tolist())) == n, "unique order ids"
                assert (r[:, C_TS] >= 1).all() and (np.diff(r[:, C_TS]) >= 0).all()
                assert (r[:, C_STATUS] == int(Status.ACTIVE)).all()
                assert ((r[:, C_PRICE] >= PRICE_MIN_INT) & (r[:, C_PRICE] <= PRICE_MAX_INT)).all()
                assert ((r[:, C_QTY] >= QTY_MIN_INT) & (r[:, C_QTY] <= QTY_MAX_INT)).all()
                assert set(r[:, C_SIDE].tolist()) <= {BUY, SELL}
                assert set(r[:, C_EXEC].tolist()) <= {NON, IOC, FOK, ICE}
                ice = r[r[:, C_EXEC] == ICE]
                assert ((ice[:, C_PEAK] > 0) & (ice[:, C_PEAK] <= ice[:, C_QTY])).all(), \
                    "every iceberg must be valid (0 < peak <= qty) or the construction is vacuous"
                non_ice = r[r[:, C_EXEC] != ICE]
                assert (non_ice[:, C_PEAK] == 0).all() and (non_ice[:, C_PPD] == 0).all()
                if key != "C09.icewpi_wrong_sign":
                    assert not ((ice[:, C_SIDE] == SELL) & (ice[:, C_PPD] < 0)).any()
                    assert not ((ice[:, C_SIDE] == BUY) & (ice[:, C_PPD] > 0)).any()

    def test_strict_timestamps_outside_c15(self):
        for key in S1_KEYS - {"C15.random_session"}:
            for seed in range(6):
                r = real_rows(synth.CONSTRUCTIONS[key](seed).tape)
                assert (np.diff(r[:, C_TS]) >= 1).all(), key

    def test_tapes_and_aux_tapes_do_not_share_memory(self):
        for key in ("C03.same_owner_pair", "C07.ioc_regimes"):
            d = synth.CONSTRUCTIONS[key](1)
            for t in d.aux_tapes.values():
                assert not np.shares_memory(d.tape, t)

    def test_empty_and_pad_only_shapes(self):
        assert synth.make_empty().tape.shape == (0, 13)
        assert synth.make_empty().n_real_rows() == 0
        assert synth.make_pad_only().tape.shape == (64, 13)
        assert synth.make_pad_only(5).tape.shape == (5, 13)
        assert synth.make_pad_only().n_real_rows() == 0

    def test_pad_to_and_stack_tapes(self):
        a = synth.make_crossing_pair(0).tape[:2]
        b = synth.make_priority_ladder(0).tape
        p = synth.pad_to(a, 7)
        assert p.shape == (7, 13) and p.dtype == np.int32
        assert (p[2:] == np.asarray(EXPECTED_PAD)).all()
        s = synth.stack_tapes([a, b], pad_extra=3)
        n = max(len(a), len(b)) + 3
        assert s.shape == (2, n, 13) and s.dtype == np.int32 and s.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError):
            synth.stack_tapes([])


class TestTapeBuilder:
    def test_float_or_int_but_not_both(self):
        b = synth.TapeBuilder()
        with pytest.raises(ValueError):
            b.add(side=BUY, price_eur=50.0, price_int=1, qty_int=10)
        with pytest.raises(ValueError):
            b.add(side=BUY, price_int=1, qty_mw=1.0, qty_int=10)
        with pytest.raises(ValueError):
            b.add(side=BUY, qty_int=10)
        with pytest.raises(ValueError):
            b.add(side=BUY, price_int=10)
        with pytest.raises(ValueError):
            b.add(side=BUY, price_int=10, qty_int=10, peak_int=2, peak_qty_mw=0.2)
        with pytest.raises(ValueError):
            b.add(side=BUY, price_int=10, qty_int=10, ppd_int=2, ppd_eur=0.02)

    def test_float_is_encoded_and_int_is_verbatim(self):
        b = synth.TapeBuilder()
        o1 = b.add(side=SELL, price_eur=42.37, qty_mw=3.4)
        o2 = b.add(side=BUY, price_int=1_004_237, qty_int=34, peak_int=7, ppd_int=-3,
                   exec_restriction=ICE)
        t = b.build(pad_extra=2)
        assert t.shape == (4, 13)
        r1, r2 = row_of(t, o1), row_of(t, o2)
        assert r1[C_PRICE] == encode_price(42.37, CFG) and r1[C_QTY] == encode_qty(3.4, CFG)
        assert (r2[C_PRICE], r2[C_QTY], r2[C_PEAK], r2[C_PPD]) == (1_004_237, 34, 7, -3)
        assert (o1, o2) == (1, 2) and r1[C_TS] < r2[C_TS]


# ===========================================================================
# 3. Closed-form expectations, re-derived (S4, S5, S11, AC 8)
# ===========================================================================
class TestClosedForms:
    @pytest.mark.parametrize("key", sorted(S1_KEYS - {"C15.random_session"}))
    def test_rederived_expectation_holds(self, key):
        for seed in range(40):
            d = synth.CONSTRUCTIONS[key](seed)
            assert CHECKS[key](d) == [], (key, seed, CHECKS[key](d))

    def test_c15_rederived_invariants_hold_on_every_grid_cell(self):
        for reg in synth.fuzz_grid():
            d = synth.make_random_session(0, reg)
            assert chk_random_session(d) == [], (reg.name, chk_random_session(d))

    @pytest.mark.parametrize("key", sorted(S1_KEYS - {
        "C15.random_session", "C09.icewpi_wrong_sign"}))
    def test_rederivation_is_seen_failing_on_a_mutated_expectation(self, key):
        d = synth.CONSTRUCTIONS[key](2)
        bad = dataclasses.replace(d, expected=mutate_first_int(d.expected))
        assert CHECKS[key](bad) != []

    def test_wrong_sign_check_is_seen_failing_on_the_sign_valid_tape(self):
        d = synth.make_icewpi_drift(2, wrong_sign=True)
        good_tape = synth.make_icewpi_drift(2).tape
        assert chk_wrong_sign(dataclasses.replace(d, tape=good_tape)) != []

    def test_c15_check_is_seen_failing_on_a_crossing_safe_order(self):
        reg = _fuzz_cell("non", 160, 0.6)
        d = synth.make_random_session(0, reg)
        sf = d.params["safe_from_index"]
        assert sf is not None, "this cell should engage safe mode"
        t = d.tape.copy()
        i = sf + 1
        t[i, C_PRICE] = PRICE_MAX_INT - 1 if t[i, C_SIDE] == BUY else 1
        assert chk_random_session(dataclasses.replace(d, tape=t)) != []

    def test_c01_validation_agrees_with_the_tick_rule_for_many_seeds(self):
        for seed in range(200):
            assert chk_validation(synth.make_validation_cases(seed)) == []

    def test_c06_stress_is_past_int32_and_in_bounds(self):
        d = synth.make_multi_level_walk(0, stress=True)
        assert d.expected["weighted_price_int"] > 2**31 - 1
        r = real_rows(d.tape)
        assert (r[:, C_QTY] <= QTY_MAX_INT).all() and (r[:, C_PRICE] <= PRICE_MAX_INT).all()

    def test_c03_decoys_discriminate_price_ranking_from_time_ranking(self):
        """Plan C03: decoys exist "so price ranking and time ranking are tested together".

        S15 (v5.184): retargeted to ``variant="price_over_time"``, whose worse-priced
        decoys are submitted before the touch orders and are reachable by the
        aggressor. A ranker that ignores price (time only, among price-crossable
        orders) must give a different fill set on every draw.
        """
        agreeing = []
        for seed in range(200):
            d = synth.make_priority_ladder(seed, variant="price_over_time")
            agg = row_of(d.tape, d.params["agg_oid"])
            resting = [[int(v) for v in r] for r in real_rows(d.tape)
                       if int(r[C_OID]) != agg[C_OID]]
            if agg[C_SIDE] == BUY:
                pool = [r for r in resting if agg[C_PRICE] >= r[C_PRICE]]
            else:
                pool = [r for r in resting if agg[C_PRICE] <= r[C_PRICE]]
            pool.sort(key=lambda r: r[C_TS])  # time-only ranking
            rem, fifo = agg[C_QTY], []
            for r in pool:
                if rem == 0:
                    break
                fifo.append(r[C_OID])
                rem -= min(rem, r[C_QTY])
            if fifo == d.expected["filled_oids"]:
                agreeing.append(seed)
        assert agreeing == [], (
            f"time-only priority reproduces the price-time fill set on seeds {agreeing}")

    @pytest.mark.parametrize("key,field,values", [
        ("C07.ioc_regimes", "regime", {"none", "partial", "full"}),
        ("C10.fok_visible", "delta_ticks", {-1, 0, 1}),
        ("C04.iceberg_cancel_fok", "n_vis", {0, 1, 2, 3}),
        ("C04.cancellation", "n_decoys", {1, 2, 3}),
        ("C03.priority_ladder", "k", {2, 3, 4, 5, 6}),
        ("C03.price_over_time", "n_early", {1, 2, 3}),
        ("C03.price_over_time", "k", {2, 3, 4, 5, 6}),
        ("C08.incoming_iceberg", "L", set(range(2, 9))),
    ])
    def test_draws_cover_their_declared_ranges(self, key, field, values):
        seen = {synth.CONSTRUCTIONS[key](s).params[field] for s in range(200)}
        assert seen == values

    @pytest.mark.parametrize("key", ["C10.fok_hidden", "C10.fok_icewpi"])
    def test_fok_boundaries_draw_both_verdicts(self, key):
        seen = {synth.CONSTRUCTIONS[key](s).expected["accepted"] for s in range(60)}
        assert seen == {True, False}

    @pytest.mark.parametrize("key", ["C02.crossing_pair", "C03.priority_ladder",
                                     "C03.price_over_time", "C08.rule3", "C09.icewpi_drift", "C10.fok_icewpi"])
    def test_both_sides_are_drawn(self, key):
        seen = set()
        for s in range(40):
            d = synth.CONSTRUCTIONS[key](s)
            seen.add(int(real_rows(d.tape)[0][C_SIDE]))
        assert seen == {BUY, SELL}


# ===========================================================================
# 4. Draw contract (S4) and digests (S8)
# ===========================================================================
class TestDrawContract:
    @pytest.mark.parametrize("key", sorted(S1_KEYS))
    def test_draw_fields(self, key):
        d = synth.CONSTRUCTIONS[key](1)
        assert set(d.capacity_bounds) == set(AXES)
        assert d.capacity_bounds["hops"] == d.hop_bound
        assert all(type(v) is int and v >= 0 for v in d.capacity_bounds.values())
        want_kind = ("exact-literal" if key.startswith("C00.") else
                     "observation" if key == "C09.icewpi_wrong_sign" else
                     "invariant" if key.startswith("C15.") else "exact-parametric")
        assert d.oracle_kind == want_kind
        assert isinstance(d.pending, tuple)
        if key != "C08.rule3":
            assert not any("OQ-31" in p for p in d.pending)
        for leaf in leaves(d.expected):
            assert type(leaf) in (int, bool), f"{key}: non-integer expected leaf {leaf!r}"
        json.dumps(d.params, allow_nan=False)
        json.dumps(d.expected, allow_nan=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.hop_bound = 0  # type: ignore[misc]

    def test_independent_draw_digest(self):
        for key in ("C03.same_owner_pair", "C07.ioc_regimes", "C08.rule3", "C15.random_session"):
            d = synth.CONSTRUCTIONS[key](4)
            h = hashlib.blake2b(digest_size=16)
            h.update(np.ascontiguousarray(d.tape.astype("<i4")).tobytes())
            for name in sorted(d.aux_tapes):
                h.update(name.encode("utf-8"))
                h.update(np.ascontiguousarray(d.aux_tapes[name].astype("<i4")).tobytes())
            h.update(json.dumps({"params": d.params, "expected": d.expected}, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8"))
            assert d.digest() == h.hexdigest(), key

    def test_digest_sensitivity(self):
        d = synth.make_same_owner_pair(3)
        base = d.digest()
        t = d.tape.copy()
        t[0, C_QTY] += 1
        assert dataclasses.replace(d, tape=t).digest() != base
        aux = {"counterfactual2": d.aux_tapes["counterfactual"]}
        assert dataclasses.replace(d, aux_tapes=aux).digest() != base
        p = copy.deepcopy(d.params)
        p["p_rest_eur"] = p["p_rest_eur"] + 0.01
        assert dataclasses.replace(d, params=p).digest() != base
        x = dict(d.expected, must_trade=1)  # bool -> int must not alias
        assert dataclasses.replace(d, expected=x).digest() != base

    def test_fixture_schema_and_full_recompute(self):
        fx = json.loads(FIXTURE.read_text())
        assert fx["schema_version"] == 1
        assert fx["algorithm"] == "blake2b-128"
        assert fx["seeds"] == list(range(8))
        assert set(fx["constructions"]) == S1_KEYS
        for key, stored in fx["constructions"].items():
            hexes = "".join(synth.CONSTRUCTIONS[key](s).digest() for s in range(8))
            assert hashlib.blake2b(hexes.encode("ascii"), digest_size=16).hexdigest() == stored, key

    @pytest.mark.parametrize("key,literal", [
        ("C02.crossing_pair", "092b654f11300b162909d7ca1d7a22bf"),
        ("C03.priority_ladder", "2482d3fd8f8e4c40778874e6ef65354b"),
        ("C05.aggressor_residual", "293b9abd103336b1a5547aa479558552"),
        ("C06.multi_level_walk", "377c9a4746f9d4dcbb95b6af95dd5e76"),
        ("C07.ioc_regimes", "68485ff30dd14df3c520b82c44228238"),
        ("C08.iceberg_cascade", "01817ba01834e49639dfba2477937d3d"),
    ])
    def test_archive_continuity_tape_only(self, key, literal):
        blob = b"".join(np.ascontiguousarray(synth.CONSTRUCTIONS[key](s).tape.astype("<i4")).tobytes()
                        for s in range(8))
        assert hashlib.blake2b(blob, digest_size=16).hexdigest() == literal


# ===========================================================================
# 5. I8 envelope (S6) — per-axis refusal, opt-out confinement
# ===========================================================================
class TestEnvelope:
    def test_defaults(self):
        e = synth.DEFAULT_ENVELOPE
        assert (e.max_resting_per_side, e.max_trades, e.max_hops, e.max_concurrent_icebergs) == \
            (80, 80, 160, 800)
        assert e == synth.CapacityEnvelope()

    @pytest.mark.parametrize("field", sorted(CEILING))
    def test_each_field_is_bounded_independently_below_its_ceiling(self, field):
        ceil = CEILING[field]
        assert getattr(synth.CapacityEnvelope(**{field: ceil - 1}), field) == ceil - 1
        assert getattr(synth.CapacityEnvelope(**{field: 1}), field) == 1
        for bad in (ceil, ceil + 1, 0, -1, True, 5.0, "5"):
            with pytest.raises(ValueError):
                synth.CapacityEnvelope(**{field: bad})

    def test_book_depth_and_max_icebergs_are_independent(self):
        e = synth.CapacityEnvelope(max_resting_per_side=5, max_concurrent_icebergs=1000)
        assert (e.max_resting_per_side, e.max_concurrent_icebergs) == (5, 1000)
        e = synth.CapacityEnvelope(max_resting_per_side=99, max_concurrent_icebergs=1)
        assert (e.max_resting_per_side, e.max_concurrent_icebergs) == (99, 1)

    @pytest.mark.parametrize("axis", AXES)
    def test_check_envelope_inclusive_limit_per_axis(self, axis):
        env = synth.CapacityEnvelope(max_resting_per_side=10, max_trades=11, max_hops=12,
                                     max_concurrent_icebergs=13)
        limit = getattr(env, ENV_FIELD[axis])
        vals = {a: 0 for a in AXES}
        vals[axis] = limit
        assert synth.check_envelope(env, **vals) is None
        vals[axis] = limit + 1
        with pytest.raises(synth.CapacityBreach) as ei:
            synth.check_envelope(env, **vals)
        assert (ei.value.axis, ei.value.value, ei.value.limit) == (axis, limit + 1, limit)
        assert isinstance(ei.value, ValueError)

    def test_first_breach_in_axis_order_is_reported(self):
        env = synth.CapacityEnvelope(max_resting_per_side=1, max_trades=1, max_hops=1,
                                     max_concurrent_icebergs=1)
        big = 10_000
        for k, axis in enumerate(AXES):
            vals = {a: (big if i >= k else 0) for i, a in enumerate(AXES)}
            with pytest.raises(synth.CapacityBreach) as ei:
                synth.check_envelope(env, **vals)
            assert ei.value.axis == axis

    def test_check_envelope_is_keyword_only_and_complete(self):
        env = synth.DEFAULT_ENVELOPE
        with pytest.raises(TypeError):
            synth.check_envelope(env, 0, 0, 0, 0)  # type: ignore[misc]
        with pytest.raises(TypeError):
            synth.check_envelope(env, resting_per_side=0, trades=0, hops=0)  # type: ignore[call-arg]

    def test_none_is_not_an_opt_out_and_the_sentinel_is(self):
        kw = dict(resting_per_side=10**6, trades=10**6, hops=10**6, concurrent_icebergs=10**6)
        with pytest.raises(TypeError):
            synth.check_envelope(None, **kw)
        with pytest.raises(TypeError):
            synth.check_envelope(object(), **kw)
        assert synth.check_envelope(synth.C14_CEILING_OPT_OUT, **kw) is None

    @pytest.mark.parametrize("key", sorted(ENVELOPED_MODES))
    def test_every_s1_mode_refuses_the_opt_out_and_none(self, key):
        mode = ENVELOPED_MODES[key]
        with pytest.raises(ValueError) as ei:
            mode(0, synth.C14_CEILING_OPT_OUT)
        assert not isinstance(ei.value, synth.CapacityBreach)
        with pytest.raises(TypeError):
            mode(0, None)

    def test_bookless_modes_take_no_envelope(self):
        for fn in (synth.make_empty, synth.make_pad_only, synth.make_validation_cases):
            with pytest.raises(TypeError):
                fn(0, envelope=synth.DEFAULT_ENVELOPE) if fn is not synth.make_empty \
                    else fn(envelope=synth.DEFAULT_ENVELOPE)

    @pytest.mark.parametrize("key", sorted(set(ENVELOPED_MODES) - {"C15.random_session"}))
    def test_envelope_equal_to_the_bounds_is_accepted(self, key):
        """Inclusive limits at the mode level: an envelope set exactly at the draw's bounds passes."""
        for seed in range(4):
            d = ENVELOPED_MODES[key](seed, synth.DEFAULT_ENVELOPE)
            tight = synth.CapacityEnvelope(**{ENV_FIELD[a]: max(1, d.capacity_bounds[a])
                                              for a in AXES})
            assert ENVELOPED_MODES[key](seed, tight).digest() == d.digest()

    def test_c15_reported_bounds_respect_a_tight_envelope(self):
        env = synth.CapacityEnvelope(max_resting_per_side=50, max_trades=20)
        for name in ("non", "non_ioc_fok_ice", "all_icewpi"):
            reg = _fuzz_cell(name, 96, 0.6)
            for seed in range(3):
                d = synth.make_random_session(seed, reg, envelope=env)
                for a in AXES:
                    assert d.capacity_bounds[a] <= getattr(env, ENV_FIELD[a])
                assert chk_random_session(d, env) == []

    def test_c15_impossible_resting_cell_raises_on_the_resting_axis(self):
        reg = _fuzz_cell("non", 160, 0.05)
        with pytest.raises(synth.CapacityBreach) as ei:
            synth.make_random_session(0, reg,
                                      envelope=synth.CapacityEnvelope(max_resting_per_side=40))
        assert ei.value.axis == "resting_per_side"

    def test_c15_redraws_instead_of_refusing_on_the_iceberg_axis(self):
        """Plan §C15 step 3: the draw is rejection-sampled against all four I8 margins,
        re-seeded within a retry budget; card S13: C15 "adapts instead of raising".

        Iceberg counts per attempt on this cell are ~35-45, so a limit of 45 is met
        by most attempts; a retrying generator should never refuse.
        """
        reg = _fuzz_cell("non_ioc_fok_ice", 160, 0.05)
        env = synth.CapacityEnvelope(max_concurrent_icebergs=45)
        refused = []
        for seed in range(16):
            try:
                synth.make_random_session(seed, reg, envelope=env)
            except synth.CapacityBreach as exc:
                refused.append((seed, exc.axis, exc.value))
        assert refused == [], f"refused on attempt 0 without redrawing: {refused}"


# ===========================================================================
# 6. Fuzz grid (S10, AC 5)
# ===========================================================================
class TestFuzzGrid:
    def test_grid_shape(self):
        g = synth.fuzz_grid()
        assert len(g) == 120 and len({r.name for r in g}) == 120
        assert {r.n_orders for r in g} == {32, 96, 160}
        assert min(r.cross_prob for r in g) == 0.05 and max(r.cross_prob for r in g) == 0.6
        assert any(r.tie_density == 0 for r in g) and any(r.tie_density > 0 for r in g)
        assert {r.icewpi for r in g} == {True, False}
        assert all(ExecRestriction.ICE in r.order_types for r in g if r.icewpi)

    def test_tie_density_produces_ties(self):
        tie_reg = [r for r in synth.fuzz_grid() if r.tie_density > 0 and r.n_orders == 160][0]
        r = real_rows(synth.make_random_session(0, tie_reg).tape)
        assert (np.diff(r[:, C_TS]) == 0).any()

    def test_icewpi_cells_emit_price_increments(self):
        reg = _fuzz_cell("all_icewpi", 160, 0.2)
        r = real_rows(synth.make_random_session(0, reg).tape)
        assert (r[r[:, C_EXEC] == ICE][:, C_PPD] != 0).any()


# ===========================================================================
# 7. Import boundary (S12)
# ===========================================================================
class TestImportBoundary:
    ALLOWED_PKG = {
        "omiecid_simulator.omiecid.encoding", "omiecid_simulator.omiecid.omie_config",
        "omiecid_simulator.omiecid.order_types", "omiecid_simulator.omiecid.contract",
        "omiecid_simulator.jaxomiecid.omie_data_loader",
    }

    def test_ast_imports(self):
        tree = ast.parse(SYNTH_SRC.read_text())
        stdlib = set(sys.stdlib_module_names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    top = a.name.split(".")[0]
                    assert top == "numpy" or top in stdlib, a.name
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "relative import"
                mod = node.module or ""
                top = mod.split(".")[0]
                if top == "omiecid_simulator":
                    assert mod in self.ALLOWED_PKG, mod
                    if mod == "omiecid_simulator.jaxomiecid.omie_data_loader":
                        assert [a.name for a in node.names] == ["PAD_ORDER_ROW"]
                    if mod == "omiecid_simulator.omiecid.contract":
                        assert {a.name for a in node.names} <= {"ContractStatus"}
                else:
                    assert top == "numpy" or top in stdlib, mod
            elif isinstance(node, ast.Name):
                assert node.id not in ("__import__",), "dynamic import"
            elif isinstance(node, ast.Attribute):
                assert node.attr != "import_module", "dynamic import"
        src = SYNTH_SRC.read_text()
        assert "import jax" not in src and "jax.random" not in src

    @pytest.mark.slow
    def test_transitive_import_pulls_no_engine_module(self):
        """slow: fresh interpreter importing jax via the loader (~2 s)."""
        code = ("import sys, omiecid_simulator.jaxomiecid.omie_tape_synth\n"
                "bad=[m for m in sys.modules if m.split('.')[-1] in ('contract_book',"
                "'iceberg_manager','order_validator','settlement','trade_extraction','omie_session')]\n"
                "print(bad); sys.exit(1 if bad else 0)\n")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", PYTHONPATH=str(REPO))
        p = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stdout + p.stderr[-2000:]

    @pytest.mark.slow
    def test_digest_table_is_stable_across_processes_and_hash_seeds(self):
        """slow: two fresh interpreters recompute the whole digest table (~5 s)."""
        code = ("import json, omiecid_simulator.jaxomiecid.omie_tape_synth as s\n"
                "print(json.dumps(s.compute_digest_table(), sort_keys=True))\n")
        stored = json.loads(FIXTURE.read_text())["constructions"]
        for hs in ("0", "12345"):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                       PYTHONPATH=str(REPO), PYTHONHASHSEED=hs)
            p = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                               capture_output=True, text=True, timeout=300)
            assert p.returncode == 0, p.stderr[-2000:]
            assert json.loads(p.stdout.strip().splitlines()[-1]) == stored


# ===========================================================================
# 8. Sound capacity bounds confirmed against the engine (S7) — beyond seeds 0-1
# ===========================================================================
@pytest.fixture(scope="module")
def engine():
    jax = pytest.importorskip("jax")
    from omiecid_simulator.omiecid.contract import Contract
    from omiecid_simulator.omiecid.contract_book import ContractBook
    from omiecid_simulator.omiecid.encoding import decode_ppd, decode_price, decode_qty
    from omiecid_simulator.omiecid.order_types import OMIEOrder

    book = ContractBook(Contract(0, 600, 615, 540, ContractStatus.OPEN), CFG)

    def to_order(r):
        r = [int(v) for v in r]
        return OMIEOrder(r[0], r[1], float(decode_price(r[2], CFG)), float(decode_qty(r[3], CFG)),
                         r[4], r[5], r[6], r[7], r[8], r[9], r[10],
                         float(decode_qty(r[11], CFG)), float(decode_ppd(r[12], CFG)))

    def replay(tape):
        with jax.default_device(jax.devices("cpu")[0]):
            st = book.init()
            ring = []
            for r in real_rows(tape):
                st, _ = book.submit_order(st, to_order(r))
                ring.append(int((np.asarray(st.lob.trades)[:, 0] != -1).sum()))
            asks = int((np.asarray(st.lob.asks)[:, 0] >= 0).sum())
            bids = int((np.asarray(st.lob.bids)[:, 0] >= 0).sum())
            return {"ring": ring, "asks": asks, "bids": bids,
                    "truncated": bool(st.truncated), "book_full": bool(st.book_full)}

    return replay


def _assert_bounds_hold(d, out):
    cb = d.capacity_bounds
    assert (out["ring"][-1] if out["ring"] else 0) <= cb["trades"]
    assert max(out["asks"], out["bids"]) <= cb["resting_per_side"]
    assert not out["truncated"] and not out["book_full"]


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(S1_KEYS - {"C00.empty", "C00.pad_only",
                                                  "C01.validation", "C15.random_session"}))
def test_engine_confirms_bounds_on_unlocked_seeds(engine, key):
    """slow: ContractBook JIT warm-up + row-by-row replay, seeds 2-3."""
    for seed in (2, 3):
        d = synth.CONSTRUCTIONS[key](seed)
        _assert_bounds_hold(d, engine(d.tape))


@pytest.mark.slow
@pytest.mark.parametrize("mix", ["non_ioc_fok_ice", "all_icewpi", "non"])
@pytest.mark.parametrize("cross", [0.2, 0.6])
def test_engine_confirms_c15_bounds_and_safe_mode_never_trades(engine, mix, cross):
    """slow: 160-order sessions replayed through ContractBook."""
    d = synth.make_random_session(1, _fuzz_cell(mix, 160, cross))
    out = engine(d.tape)
    _assert_bounds_hold(d, out)
    sf = d.params["safe_from_index"]
    if sf is not None and sf > 0:
        assert out["ring"][-1] == out["ring"][sf - 1], "a safe-mode order traded"


@pytest.mark.slow
def test_engine_confirms_c15_under_a_tight_envelope(engine):
    """slow: 160-order ICEwPI session replayed under max_trades=20."""
    env = synth.CapacityEnvelope(max_resting_per_side=80, max_trades=20)
    d = synth.make_random_session(2, _fuzz_cell("all_icewpi", 160, 0.6), envelope=env)
    out = engine(d.tape)
    assert out["ring"][-1] <= 20
    _assert_bounds_hold(d, out)
