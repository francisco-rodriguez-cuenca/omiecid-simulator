"""S2 contract for the new C03 construction ``C03.price_over_time`` (OMIE-CERT-SYNTH-1, S15).

Story card: ``docs/omiecid/agile-plan.md`` -> "Story OMIE-CERT-SYNTH-1", row
"Step-5 ruling decisions (v5.184, binding)", decision S15. Plan reference:
``docs/omiecid/scenario-ladder-plan.md`` §C3 ("price ranking and time ranking are
tested together").

Contract pinned here (verbatim from S15):

* ``make_priority_ladder(seed, config=OMIEProductConfig(), *, envelope=DEFAULT_ENVELOPE,
  variant="ladder")``; ``variant`` is ``"ladder"`` or ``"price_over_time"``; anything
  else raises ``ValueError``. ``variant="ladder"`` is byte-identical to the default.
* ``variant="price_over_time"`` draws from ``PCG64(seed)`` in this order:
  k = integers(2, 7); j = integers(1, k); n_early = integers(1, 4);
  rest side SELL if random() < 0.5 else BUY; p_touch = snap(uniform(30, 90));
  q_each = snap(uniform(2, 20)); s = integers(1, 51) price ticks.
* Submission order: n_early early decoys (ascending d = 1..n_early, rest side,
  qty q_each_int, price p_touch_int + d*s if resting SELL else p_touch_int - d*s),
  then k touch orders at p_touch_int, then one aggressor (other side,
  qty j*q_each_int, price = deepest decoy price). Rest agent 1, aggressor agent 2.
* ``params``: k, j, n_early, step_ticks, rest_side, agg_side, p_touch_int,
  q_each_int, early_decoy_oids, decoy_price_ints, at_touch_oids, agg_oid,
  q_agg_int, p_agg_int, p_touch_eur, q_each_mw.
* ``expected``: filled_oids = at_touch_oids[:j]; untouched_oids =
  at_touch_oids[j:] + early_decoy_oids; n_trades = j; trade_price_int =
  p_touch_int; total_qty_int = j * q_each_int.
* ``oracle_kind == "exact-parametric"``; ``capacity_bounds`` are the generic
  ``tape_capacity_bounds`` of the tape.
* Registry key ``"C03.price_over_time"`` == ``make_priority_ladder(seed,
  variant="price_over_time")``.

Fast suite only: pure numpy, no engine.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

import omiecid_simulator.jaxomiecid.omie_tape_synth as synth
from omiecid_simulator.omiecid.encoding import encode_price, encode_qty
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side

KEY = "C03.price_over_time"
CFG = OMIEProductConfig()
BUY, SELL = int(Side.BUY), int(Side.SELL)
NON = int(ExecRestriction.NON)

# Loader column order (S10).
C_SIDE, C_PRICE, C_QTY, C_OID, C_AGENT, C_TS, C_EXEC = 1, 2, 3, 4, 5, 6, 7

PARAM_KEYS = {
    "k", "j", "n_early", "step_ticks", "rest_side", "agg_side", "p_touch_int",
    "q_each_int", "early_decoy_oids", "decoy_price_ints", "at_touch_oids", "agg_oid",
    "q_agg_int", "p_agg_int", "p_touch_eur", "q_each_mw",
}
EXPECTED_KEYS = {"filled_oids", "untouched_oids", "n_trades", "trade_price_int",
                 "total_qty_int"}
SEEDS = range(40)


def _draw(seed):
    return synth.make_priority_ladder(seed, variant="price_over_time")


def _real_rows(tape):
    t = np.asarray(tape).reshape(-1, 13)
    return [[int(v) for v in r] for r in t if int(r[C_OID]) >= 1]


def _crossable(agg_side, agg_price, rest_price):
    return agg_price >= rest_price if agg_side == BUY else agg_price <= rest_price


def _price_time_fills(rows):
    """Reference price-time matcher: the last row is the aggressor; returns (rest_oid, price, qty)."""
    *resting, agg = rows
    pool = [r for r in resting if r[C_SIDE] != agg[C_SIDE]
            and _crossable(agg[C_SIDE], agg[C_PRICE], r[C_PRICE])]
    sign = 1 if agg[C_SIDE] == BUY else -1
    pool.sort(key=lambda r: (sign * r[C_PRICE], r[C_TS]))
    rem, out = agg[C_QTY], []
    for r in pool:
        if rem == 0:
            break
        q = min(rem, r[C_QTY])
        out.append((r[C_OID], r[C_PRICE], q))
        rem -= q
    return out


def _time_only_fills(rows):
    """A defective ranker: FIFO among price-crossable resting orders, ignoring price rank."""
    *resting, agg = rows
    pool = [r for r in resting if r[C_SIDE] != agg[C_SIDE]
            and _crossable(agg[C_SIDE], agg[C_PRICE], r[C_PRICE])]
    pool.sort(key=lambda r: r[C_TS])
    rem, out = agg[C_QTY], []
    for r in pool:
        if rem == 0:
            break
        out.append(r[C_OID])
        rem -= min(rem, r[C_QTY])
    return out


def _check_C03_price_over_time(draw) -> list[str]:
    """Recompute ``expected`` (and the tape layout) from ``params`` by the S15 formula."""
    e: list[str] = []
    try:
        if draw.construction != KEY:
            e.append(f"construction {draw.construction!r}")
        P, x = draw.params, draw.expected
        missing = PARAM_KEYS - set(P)
        if missing:
            return e + [f"missing params {sorted(missing)}"]
        if set(x) != EXPECTED_KEYS:
            e.append(f"expected keys {sorted(x)}")
        k, j, ne, s = int(P["k"]), int(P["j"]), int(P["n_early"]), int(P["step_ticks"])
        rs, ag = int(P["rest_side"]), int(P["agg_side"])
        pt, qe = int(P["p_touch_int"]), int(P["q_each_int"])
        if not 2 <= k <= 6:
            e.append(f"k={k} outside [2,6]")
        if not 1 <= j < k:
            e.append(f"j={j} outside [1,k)")
        if not 1 <= ne <= 3:
            e.append(f"n_early={ne} outside [1,3]")
        if not 1 <= s <= 50:
            e.append(f"step_ticks={s} outside [1,50]")
        if rs not in (BUY, SELL) or ag != (BUY if rs == SELL else SELL):
            e.append(f"rest_side={rs} agg_side={ag}")
        if pt != encode_price(float(P["p_touch_eur"]), CFG):
            e.append("p_touch_int != encode_price(p_touch_eur)")
        if qe != encode_qty(float(P["q_each_mw"]), CFG):
            e.append("q_each_int != encode_qty(q_each_mw)")
        early, touch = list(P["early_decoy_oids"]), list(P["at_touch_oids"])
        sgn = 1 if rs == SELL else -1
        want_decoy_prices = [pt + sgn * d * s for d in range(1, ne + 1)]
        if len(early) != ne:
            e.append(f"len(early_decoy_oids)={len(early)} != n_early={ne}")
        if len(touch) != k:
            e.append(f"len(at_touch_oids)={len(touch)} != k={k}")
        if list(P["decoy_price_ints"]) != want_decoy_prices:
            e.append(f"decoy_price_ints {P['decoy_price_ints']} != {want_decoy_prices}")
        p_agg_want = pt + sgn * ne * s
        if int(P["p_agg_int"]) != p_agg_want:
            e.append(f"p_agg_int {P['p_agg_int']} != deepest decoy price {p_agg_want}")
        if int(P["q_agg_int"]) != j * qe:
            e.append("q_agg_int != j * q_each_int")

        # Tape: early decoys (ascending d), then touch orders, then the aggressor.
        rows = _real_rows(draw.tape)
        want_oids = early + touch + [int(P["agg_oid"])]
        if [r[C_OID] for r in rows] != want_oids:
            e.append(f"tape order {[r[C_OID] for r in rows]} != {want_oids}")
            return e
        want_rows = ([(rs, p, qe, 1) for p in want_decoy_prices]
                     + [(rs, pt, qe, 1)] * k + [(ag, p_agg_want, j * qe, 2)])
        for r, (side, price, qty, agent) in zip(rows, want_rows):
            got = (r[C_SIDE], r[C_PRICE], r[C_QTY], r[C_AGENT])
            if got != (side, price, qty, agent):
                e.append(f"oid {r[C_OID]}: (side,price,qty,agent)={got} "
                         f"want {(side, price, qty, agent)}")
            if r[C_EXEC] != NON:
                e.append(f"oid {r[C_OID]}: exec {r[C_EXEC]} != NON")
        ts = [r[C_TS] for r in rows]
        if any(a >= b for a, b in zip(ts, ts[1:])):
            e.append("timestamps not strictly increasing")
        if not all(_crossable(ag, p_agg_want, p) for p in want_decoy_prices + [pt]):
            e.append("an early decoy is not price-crossable by the aggressor")

        # expected, by the S15 formula
        if x.get("filled_oids") != touch[:j]:
            e.append(f"filled_oids {x.get('filled_oids')} != {touch[:j]}")
        if x.get("untouched_oids") != touch[j:] + early:
            e.append(f"untouched_oids {x.get('untouched_oids')} != {touch[j:] + early}")
        if x.get("n_trades") != j:
            e.append(f"n_trades {x.get('n_trades')} != {j}")
        if x.get("trade_price_int") != pt:
            e.append(f"trade_price_int {x.get('trade_price_int')} != {pt}")
        if x.get("total_qty_int") != j * qe:
            e.append(f"total_qty_int {x.get('total_qty_int')} != {j * qe}")

        # Cross-check against an independent price-time matcher.
        fills = _price_time_fills(rows)
        if [f[0] for f in fills] != x.get("filled_oids"):
            e.append(f"reference matcher fills {[f[0] for f in fills]}")
        if any(f[1] != pt for f in fills) or sum(f[2] for f in fills) != j * qe:
            e.append("reference matcher price/qty disagree")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        e.append(f"check raised {type(exc).__name__}: {exc}")
    return e


# ---------------------------------------------------------------------------
# Registry, signature, variant dispatch
# ---------------------------------------------------------------------------
class TestRegistryAndVariant:
    def test_registry_has_price_over_time_key(self):
        assert KEY in synth.CONSTRUCTIONS

    @pytest.mark.parametrize("seed", [0, 3, 17])
    def test_registry_entry_equals_variant_call(self, seed):
        a = synth.CONSTRUCTIONS[KEY](seed)
        assert a.construction == KEY
        assert a.digest() == _draw(seed).digest()

    def test_unknown_variant_raises_value_error(self):
        with pytest.raises(ValueError):
            synth.make_priority_ladder(0, variant="bogus")

    @pytest.mark.parametrize("seed", range(8))
    def test_ladder_variant_is_byte_identical_to_default(self, seed):
        a = synth.make_priority_ladder(seed)
        b = synth.make_priority_ladder(seed, variant="ladder")
        assert a.construction == b.construction == "C03.priority_ladder"
        assert a.digest() == b.digest()

    def test_variant_is_keyword_only(self):
        with pytest.raises(TypeError):
            synth.make_priority_ladder(0, OMIEProductConfig(), synth.DEFAULT_ENVELOPE,  # type: ignore[misc]
                                       "price_over_time")

    def test_price_over_time_differs_from_ladder(self):
        assert _draw(0).digest() != synth.make_priority_ladder(0).digest()

    def test_seeded_draws_vary(self):
        assert len({_draw(s).digest() for s in range(8)}) == 8

    def test_same_seed_is_deterministic(self):
        assert _draw(11).digest() == _draw(11).digest()


# ---------------------------------------------------------------------------
# S2 spot check: passes on real draws, fails on mutated draws
# ---------------------------------------------------------------------------
class TestSpotCheck:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_check_passes_on_real_draw(self, seed):
        d = _draw(seed)
        assert _check_C03_price_over_time(d) == []

    def _mutations(self, d):
        x, P = d.expected, d.params
        out = {}
        bad = copy.deepcopy(x)
        bad["filled_oids"] = list(P["early_decoy_oids"][:1]) + list(x["filled_oids"][1:])
        out["filled_oids_swapped_for_decoy"] = dataclasses.replace(d, expected=bad)
        bad = copy.deepcopy(x)
        bad["n_trades"] = x["n_trades"] + 1
        out["n_trades_bumped"] = dataclasses.replace(d, expected=bad)
        bad = copy.deepcopy(x)
        bad["trade_price_int"] = P["decoy_price_ints"][0]
        out["trade_price_at_decoy"] = dataclasses.replace(d, expected=bad)
        bad = copy.deepcopy(x)
        bad["untouched_oids"] = list(x["untouched_oids"])[:-1]
        out["untouched_drops_a_decoy"] = dataclasses.replace(d, expected=bad)
        bad = copy.deepcopy(x)
        bad["total_qty_int"] = x["total_qty_int"] - 1
        out["total_qty_off_by_one"] = dataclasses.replace(d, expected=bad)
        # Tape mutation: the first early decoy moved to the touch price.
        t = d.tape.copy()
        rows = np.where(t[:, C_OID] == P["early_decoy_oids"][0])[0]
        t[rows[0], C_PRICE] = P["p_touch_int"]
        out["decoy_at_touch_price"] = dataclasses.replace(d, tape=t)
        # Tape mutation: aggressor limit stops short of the deepest decoy.
        t = d.tape.copy()
        r = np.where(t[:, C_OID] == P["agg_oid"])[0][0]
        t[r, C_PRICE] = P["p_touch_int"]
        out["aggressor_does_not_reach_decoys"] = dataclasses.replace(d, tape=t)
        # Params mutation: step_ticks inconsistent with the tape.
        p = copy.deepcopy(P)
        p["step_ticks"] = P["step_ticks"] + 1
        out["step_ticks_bumped"] = dataclasses.replace(d, params=p)
        # Construction-name mutation.
        out["wrong_construction"] = dataclasses.replace(d, construction="C03.priority_ladder")
        return out

    @pytest.mark.parametrize("name", [
        "filled_oids_swapped_for_decoy", "n_trades_bumped", "trade_price_at_decoy",
        "untouched_drops_a_decoy", "total_qty_off_by_one", "decoy_at_touch_price",
        "aggressor_does_not_reach_decoys", "step_ticks_bumped", "wrong_construction",
    ])
    def test_check_fails_on_mutated_draw(self, name):
        d = _draw(2)
        assert _check_C03_price_over_time(d) == []
        assert _check_C03_price_over_time(self._mutations(d)[name]) != []

    def test_check_fails_on_the_ladder_draw(self):
        """The ladder construction is not the price-over-time construction."""
        assert _check_C03_price_over_time(synth.make_priority_ladder(2)) != []


# ---------------------------------------------------------------------------
# The point of the construction: price ranking vs time ranking
# ---------------------------------------------------------------------------
class TestDiscrimination:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_time_only_ranker_disagrees_on_every_seed(self, seed):
        d = _draw(seed)
        fifo = _time_only_fills(_real_rows(d.tape))
        assert fifo != d.expected["filled_oids"], (seed, fifo)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_time_only_ranker_fills_early_decoys_first(self, seed):
        d = _draw(seed)
        P = d.params
        n = min(int(P["j"]), int(P["n_early"]))
        fifo = _time_only_fills(_real_rows(d.tape))
        assert fifo[:n] == list(P["early_decoy_oids"][:n])

    @pytest.mark.parametrize("seed", SEEDS)
    def test_price_time_reference_agrees_on_every_seed(self, seed):
        d = _draw(seed)
        assert [f[0] for f in _price_time_fills(_real_rows(d.tape))] == d.expected["filled_oids"]


# ---------------------------------------------------------------------------
# Stream order, ranges, Draw fields
# ---------------------------------------------------------------------------
class TestStreamAndDrawFields:
    @pytest.mark.parametrize("seed", range(12))
    def test_params_follow_the_s15_stream_order(self, seed):
        rng = np.random.Generator(np.random.PCG64(seed))
        k = int(rng.integers(2, 7))
        j = int(rng.integers(1, k))
        n_early = int(rng.integers(1, 4))
        rest_side = SELL if rng.random() < 0.5 else BUY
        p_touch = round(float(rng.uniform(30.0, 90.0)) / CFG.price_tick) * CFG.price_tick
        q_each = round(float(rng.uniform(2.0, 20.0)) / CFG.qty_tick) * CFG.qty_tick
        s = int(rng.integers(1, 51))
        P = _draw(seed).params
        assert (P["k"], P["j"], P["n_early"], P["rest_side"], P["step_ticks"]) == \
            (k, j, n_early, rest_side, s)
        assert P["p_touch_int"] == encode_price(p_touch, CFG)
        assert P["q_each_int"] == encode_qty(q_each, CFG)

    @pytest.mark.parametrize("field,values", [
        ("k", {2, 3, 4, 5, 6}), ("n_early", {1, 2, 3}),
    ])
    def test_draws_cover_declared_ranges(self, field, values):
        assert {_draw(s).params[field] for s in range(200)} == values

    def test_both_rest_sides_are_drawn(self):
        assert {_draw(s).params["rest_side"] for s in SEEDS} == {BUY, SELL}

    def test_j_exceeding_n_early_and_not_are_both_drawn(self):
        seen = {_draw(s).params["j"] > _draw(s).params["n_early"] for s in range(200)}
        assert seen == {True, False}

    @pytest.mark.parametrize("seed", [0, 1, 5])
    def test_draw_fields(self, seed):
        d = _draw(seed)
        assert d.oracle_kind == "exact-parametric"
        assert d.seed == seed
        assert set(PARAM_KEYS) <= set(d.params)
        assert set(d.expected) == EXPECTED_KEYS
        assert dict(d.capacity_bounds) == dict(synth.tape_capacity_bounds(d.tape))
        assert d.capacity_bounds["hops"] == d.hop_bound
        t = d.tape
        assert t.dtype == np.int32 and t.ndim == 2 and t.shape[1] == 13
        assert t.flags["C_CONTIGUOUS"]

    def test_expected_leaves_are_ints(self):
        x = _draw(4).expected
        for v in x.values():
            for leaf in (v if isinstance(v, list) else [v]):
                assert type(leaf) is int

    def test_respects_the_envelope_parameter(self):
        d = _draw(0)
        cb = d.capacity_bounds
        fields = {"max_resting_per_side": max(1, cb["resting_per_side"]),
                  "max_trades": max(1, cb["trades"]), "max_hops": max(1, cb["hops"]),
                  "max_concurrent_icebergs": max(1, cb["concurrent_icebergs"])}
        tight = synth.CapacityEnvelope(**fields)
        assert synth.make_priority_ladder(0, envelope=tight,
                                          variant="price_over_time").digest() == d.digest()
        assert cb["trades"] > 1, "a price-over-time draw always has >= 2 tradeable orders"
        too_tight = synth.CapacityEnvelope(**{**fields, "max_trades": cb["trades"] - 1})
        with pytest.raises(synth.CapacityBreach):
            synth.make_priority_ladder(0, envelope=too_tight, variant="price_over_time")
