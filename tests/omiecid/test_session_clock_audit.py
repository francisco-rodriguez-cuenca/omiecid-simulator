"""Audit (Step 5) for OMIE-SESSION-CLOCK-1 — the opt-in real-tape session clock.

Independent adversarial black-box contracts derived from the story card
(agile-plan v5.163, "#### Story OMIE-SESSION-CLOCK-1") and market-spec v2.11
§3.1 / §11.1 (hour-aligned GCT: GCT(k) = GOT + 60·(460 + 60·⌊k/4⌋) s in the L1
frame) and §13.4 (every row of product (D, k) is sent in [GOT, GCT(k))).

Contract under test (``session_clock="got_relative"`` only):
  * time = floor((max_ts_in_slice − got_seconds) / 60), int32, pad rows masked;
  * the first ALL-PAD slice sets time = gct_minutes_from_got;
  * time is monotone (never regresses); done = time ≥ gct and is sticky;
  * constructor guards (ValueError naming the key): unknown session_clock (any
    mode); a real row before got_seconds; a real row at/after GCT; fewer than
    orders_per_step trailing pad rows on the longest tape;
  * the default ``span_scaled`` clock follows the pre-story span-scaled
    formula and never terminates on all-pad.

RELEASE NOTE (REL-07): the participant's order is the public ``AgentOrder``;
``ttg`` below is the session clock ``gct - state.time``. Removed with the
research code they exercised: the RL obs / dense-GCT reward checks, the jaxpr
identity against the research env at 5d7b965, and the eval scoring seam
(``omie_eval`` / ``omie_heldout_eval`` / the training config files). The
shipped-config real-data class inlines that config's session keys and dates.

Runtime: every distinct env config costs one XLA compile of ``step_env``
(~1-2 min on CPU), so each env's rollouts are computed ONCE in a module-scoped
fixture (host numpy results, JAX caches cleared before each build) and the
assertions read those results. All env-stepping tests are ``@pytest.mark.slow``
(compile cost); constructor-guard / whitelist / identity tests are fast.
Real-data checks read ``data/rawOMIE`` into a tmp cache and must RUN when the data is present.
"""

from __future__ import annotations

import os
import pathlib
from datetime import date, timedelta

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.omie_session import AgentOrder, Session
from omiecid_simulator.omiecid.order_types import ExecRestriction

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "rawOMIE"
# Session keys and train dates of the research config
# omie_settle_control_op4_session_clock_train.json (not shipped), inlined.
TRAIN_ENV_CFG = {"orders_per_step": 4, "gct_minutes_from_got": 460,
                 "session_clock": "got_relative", "got_seconds": 55_200}
TRAIN_CONTRACT_IDS = [0]
TRAIN_DATES = [f"2025-06-{d:02d}" for d in range(1, 25)]   # 2025-06-01..24

G = 55_200            # GOT = 15:20 D-1 in the L1 frame
PAD_ROW = [0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0]
_PC = OMIEProductConfig()
# Research Discrete(13) action 5 (BUY, CROSS bucket, qty bucket 0): a resting NON
# BUY at the post-background best ask, quantity qty_min. On T0 step 0 the slice
# rests BUY 40 / SELL 60 rows, so the best ask is 60.0.
BUY_CROSS_MIN_QTY = AgentOrder.create(
    side=0, price=60.0, quantity=float(_PC.qty_min),
    exec_restriction=int(ExecRestriction.NON),
)

HELD_OUT_DATES = ["2025-06-25", "2025-06-26", "2025-06-27",
                  "2025-06-28", "2025-06-29", "2025-06-30"]


def m(k, s=0, got=G):
    """Absolute L1 second of minute ``k`` (+``s`` s) after GOT."""
    return got + 60 * k + s


def gct_hour_aligned(k):
    """market-spec v2.11 §3.1: GCT(k) in minutes from GOT (contract 0 -> 460)."""
    return 460 + 60 * (k // 4)


def _row(i, ts):
    side = i % 2                                  # BUY 40 / SELL 60, never crossing
    price = 40.0 if side == 0 else 60.0
    return [0, side, int(round((price - _PC.price_min) / _PC.price_tick)),
            int(round(1.0 / _PC.qty_tick)), 1000 + i, 777, int(ts), 0, 0, 0, 0, 0, 0]


def build_tapes(tape_specs, n_padded=None):
    """``tape_specs``: list of tapes, each a flat list of timestamps (``None`` = pad
    row). Right-padded with pad rows to ``n_padded`` (default: longest tape)."""
    n = n_padded or max(len(t) for t in tape_specs)
    out = []
    for t in tape_specs:
        rows = [list(PAD_ROW) if ts is None else _row(i, ts) for i, ts in enumerate(t)]
        rows += [list(PAD_ROW) for _ in range(n - len(rows))]
        out.append(rows)
    return np.asarray(out, dtype=np.int32)


def ref_got_relative(ts_row, ops, got, gct, n_steps):
    """Independent card reference for the got_relative clock on ONE tape."""
    n = len(ts_row)
    t, times = 0, []
    for s in range(n_steps):
        start = min(s * ops, n - ops)
        sl = ts_row[start:start + ops]
        real = sl[sl > 0]
        cand = max(0, (int(real.max()) - got) // 60) if real.size else gct
        t = max(t, cand)
        times.append(t)
    return np.asarray(times)


def ref_span_scaled(ts_row, ops, gct, span, n_steps):
    """Pre-story span-scaled clock (float32 scale, round-half-even)."""
    n = len(ts_row)
    scale = np.float32(gct) / np.float32(max(span, 1.0))
    t, times = 0, []
    for s in range(n_steps):
        start = min(s * ops, n - ops)
        sl = ts_row[start:start + ops]
        real = sl[sl > 0]
        if real.size:
            t = max(t, int(np.round(np.float32(real.max()) * scale)))
        times.append(t)
    return np.asarray(times)


# ---------------------------------------------------------------------------
# rollout helpers (host-materialized results)
# ---------------------------------------------------------------------------
def _rollout_single(env, tape_idx, n_steps, orders=None, stop_extra=None):
    """Serial rollout of one tape (the tape is selected by overriding
    ``state.tape_idx`` so one compile serves every tape of the set). With
    ``stop_extra`` the loop stops ``stop_extra`` steps after the first done.
    ``orders``: optional per-step ``AgentOrder`` list (default: no order)."""
    st = env.reset(jax.random.PRNGKey(0))
    st = st.replace(tape_idx=jnp.int32(tape_idx))
    rec = {"time": [], "done": [], "ttg": [], "inv": [], "dtype": None}
    key = jax.random.PRNGKey(1)
    first_done = None
    for s in range(n_steps):
        if stop_extra is not None and first_done is not None and s > first_done + stop_extra:
            break
        k, key = jax.random.split(key)
        order = AgentOrder.none() if orders is None else orders[s]
        st, _report, done = env.step_env(k, st, order)
        rec["dtype"] = st.time.dtype
        rec["time"].append(int(st.time))
        rec["done"].append(bool(done))
        rec["ttg"].append(float(env.gct_minutes_from_got - int(st.time)))
        rec["inv"].append(float(st.agent_inventory[0]))
        if first_done is None and rec["done"][-1]:
            first_done = s
    return {k_: (np.asarray(v) if isinstance(v, list) else v) for k_, v in rec.items()}


def _lanes(order, active):
    """Batch one ``AgentOrder`` over the lanes; inactive lanes get ``none()``."""
    m = jnp.asarray(active, jnp.bool_)
    return jax.tree_util.tree_map(lambda o, n: jnp.where(m, o, n),
                                  order, AgentOrder.none())


def _rollout_vmapped(env, n_steps, orders_tb=None):
    """Vmapped rollout. ``vmap(step_env)`` only lowered under an outer
    ``enable_x64`` scope in the research env (WF-2 / the
    ``test_agent_trades_perfill_tape`` vmap precedent); running it there also
    checks the clock stays int32 with x64 on."""
    with jax.experimental.enable_x64():
        return _rollout_vmapped_body(env, n_steps, orders_tb)


@pytest.mark.jit  # lint scope for the jax.jit below (no-op on a helper)
def _rollout_vmapped_body(env, n_steps, orders_tb=None):
    """``orders_tb``: optional per-step batched ``AgentOrder`` (see ``_lanes``)."""
    b = env.n_tapes
    st = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), b))
    st = st.replace(tape_idx=jnp.arange(b, dtype=jnp.int32))
    step = jax.jit(jax.vmap(env.step_env))
    rec = {"time": [], "done": [], "ttg": [], "inv": []}
    key = jax.random.PRNGKey(1)
    dtype = None
    no_order = _lanes(AgentOrder.none(), np.zeros((b,), bool))
    for s in range(n_steps):
        key, sub = jax.random.split(key)
        orders = no_order if orders_tb is None else orders_tb[s]
        st, _report, done = step(jax.random.split(sub, b), st, orders)
        dtype = st.time.dtype
        rec["time"].append(np.asarray(st.time))
        rec["done"].append(np.asarray(done))
        rec["ttg"].append(np.float32(env.gct_minutes_from_got)
                          - np.asarray(st.time).astype(np.float32))
        rec["inv"].append(np.asarray(st.agent_inventory[:, 0]))
    out = {k_: np.stack(v) for k_, v in rec.items()}          # (T, B)
    out["dtype"] = dtype
    return out


def _first_true(x):
    idx = np.flatnonzero(x)
    return int(idx[0]) if idx.size else None


# ===========================================================================
# E1 — ops=4, contract 0 (gct 460), GOT 55,200; five adversarial tapes, vmapped
# ===========================================================================
E1_SLICES = {
    # T0: longest tape, 12 real slices (48 rows) + EXACTLY ops=4 trailing pads;
    # out-of-order ts inside slices; last real ts 82,799 (time 459, max legal).
    "T0": [
        [G, G, G, G],
        [m(1, 59), m(0, 10), m(1), m(0, 59)],
        [m(100), m(137), m(120), m(136, 59)],
        [m(137, 1), m(137, 59), m(137, 2), m(137, 3)],
        [m(200), m(229, 59), m(229), m(210)],
        [m(230), m(229, 1), m(229, 2), m(229, 3)],
        [m(230, 30)] * 4,
        [m(299, 59), m(300), m(250), m(260)],
        [m(400)] * 4,
        [m(458, 59), m(458), m(457), m(401)],
        [m(459), m(458), m(458, 1), m(459, 59)],
        [m(459, 1)] * 4,
    ],
    # T1: TC2 geometry — a real slice with max ts >= 80,920 before the last real
    # slice (last order 81,600 = 22:40 D-1).
    "T1": [
        [m(5), m(10), m(7), m(8)],
        [m(300)] * 4,
        [m(428, 40), m(420), m(425), m(428)],
        [m(430), m(435), m(440), m(439)],
    ],
    # T2: pad rows interleaved inside real slices.
    "T2": [
        [None, m(5), None, m(7, 10)],
        [m(20), None, None, None],
        [None, None, None, m(50)],
    ],
    # T3: an entirely empty (all-pad) tape.
    "T3": [],
    # T4: non-monotone slice maxima across slices (time must hold).
    "T4": [
        [m(90), m(95), m(100), m(99)],
        [m(50)] * 4,
        [m(120), m(60), m(70), m(80)],
    ],
}
# T5: an INTERIOR all-pad slice. Moved out of the shared E1 set by
# OMIE-SESSION-CLOCK-INTERIOR-PAD-GUARD-1 (A1): under got_relative it is now
# rejected at construction (see TestConstructorGuards).
T5_SLICES = [
    [m(10)] * 4,
    [None] * 4,
    [m(20)] * 4,
]
E1_ORDER = ["T0", "T1", "T2", "T3", "T4"]
E1_EXPECTED_TIMES = {
    "T0": [0, 1, 137, 137, 229, 230, 230, 300, 400, 458, 459, 459, 460],
    "T1": [10, 300, 428, 440, 460],
    "T2": [7, 20, 50, 460],
    "T3": [460],
    "T4": [100, 100, 120, 460],
}
E1_N = 48 + 4
E1_EXTRA = 3


def _flat(slices):
    return [ts for sl in slices for ts in sl]


@pytest.fixture(scope="module")
def e1_tapes():
    return build_tapes([_flat(E1_SLICES[n]) for n in E1_ORDER], n_padded=E1_N)


@pytest.fixture(scope="module")
def e1(e1_tapes):
    """slow: one vmapped step_env compile over the five adversarial tapes."""
    jax.clear_caches()
    env = Session({
        "tapes": e1_tapes, "orders_per_step": 4, "gct_minutes_from_got": 460,
        "session_clock": "got_relative", "got_seconds": G,
    })
    n_steps = env.steps_per_episode + E1_EXTRA
    # T0 only, step 0 only: take a small long position.
    t0_lane = np.arange(env.n_tapes) == 0
    no_lane = np.zeros((env.n_tapes,), bool)
    orders = [_lanes(BUY_CROSS_MIN_QTY, t0_lane if s == 0 else no_lane)
              for s in range(n_steps)]
    res = _rollout_vmapped(env, n_steps, orders)
    res["env"] = env
    res["n_steps"] = n_steps
    return res


@pytest.mark.slow
@pytest.mark.jit
class TestE1GotRelativeOps4:
    def test_steps_per_episode_is_minimal_pad_geometry(self, e1):
        """slow: shares the E1 compile. 48 real rows + 4 pad -> 13 steps; the done
        step of the longest tape is the LAST step of the episode."""
        assert e1["env"].steps_per_episode == 13
        assert _first_true(e1["done"][:, 0]) == 12

    @pytest.mark.parametrize("name", E1_ORDER)
    def test_time_trajectory_matches_card_formula(self, e1, name):
        """slow: shares the E1 compile."""
        b = E1_ORDER.index(name)
        exp = E1_EXPECTED_TIMES[name]
        got = e1["time"][:, b]
        np.testing.assert_array_equal(got[: len(exp)], exp)
        # after the done step, time stays at GCT
        assert (got[len(exp) - 1:] == 460).all()

    @pytest.mark.parametrize("name", E1_ORDER)
    def test_time_matches_independent_reference(self, e1, e1_tapes, name):
        """slow: shares the E1 compile."""
        b = E1_ORDER.index(name)
        ref = ref_got_relative(e1_tapes[b, :, 6], 4, G, 460, e1["n_steps"])
        np.testing.assert_array_equal(e1["time"][:, b], ref)

    @pytest.mark.parametrize("name", E1_ORDER)
    def test_done_fires_exactly_at_first_all_pad_slice_and_is_sticky(self, e1, name):
        """slow: shares the E1 compile."""
        b = E1_ORDER.index(name)
        done = e1["done"][:, b]
        first = len(E1_EXPECTED_TIMES[name]) - 1
        assert _first_true(done) == first
        assert done[first:].all(), "done must be sticky after the first all-pad slice"
        assert not done[:first].any()

    def test_time_is_int32_and_monotone(self, e1):
        """slow: shares the E1 compile."""
        assert e1["dtype"] == jnp.int32
        assert (np.diff(e1["time"], axis=0) >= 0).all()

    def test_tc1_time_137_ttg_323(self, e1):
        """slow: shares the E1 compile. Max ts 55,200 + 60·137 in a slice."""
        assert e1["time"][2, 0] == 137
        assert e1["ttg"][2, 0] == 323.0

    def test_tc2_short_tape_not_done_before_exhaustion(self, e1):
        """slow: shares the E1 compile. Last order 81,600: the slice with max ts
        80,920 (span-scaled would fire there) does NOT end the episode."""
        b = E1_ORDER.index("T1")
        assert not e1["done"][:4, b].any()
        assert e1["done"][4, b] and e1["time"][4, b] == 460

    def test_last_legal_second_does_not_end_episode(self, e1):
        """slow: shares the E1 compile. ts 82,799 -> time 459, ttg 1, not done."""
        assert e1["time"][10, 0] == 459 and e1["ttg"][10, 0] == 1.0
        assert not e1["done"][:12, 0].any()

    def test_empty_tape_done_on_first_step(self, e1):
        """slow: shares the E1 compile."""
        b = E1_ORDER.index("T3")
        assert e1["done"][0, b] and e1["time"][0, b] == 460

    def test_vmap_different_exhaustion_steps(self, e1):
        """slow: shares the E1 compile. Each batch lane terminates independently."""
        firsts = [_first_true(e1["done"][:, b]) for b in range(len(E1_ORDER))]
        assert firsts == [len(E1_EXPECTED_TIMES[n]) - 1 for n in E1_ORDER]
        assert len(set(firsts)) >= 4

    def test_agent_actually_holds_inventory_for_ramp_checks(self, e1):
        """slow: shares the E1 compile. Precondition for the dense-GCT ramp tests."""
        assert e1["inv"][0, 0] > 0.0
        assert (e1["inv"][:, 0] == e1["inv"][0, 0]).all()


# ===========================================================================
# E2..E5 — orders_per_step 1, 2, 3, 5 with varied got_seconds / GCT
# (incl. contracts with k mod 4 != 0 under 460 + 60·⌊k/4⌋)
# ===========================================================================
OPS_CASES = {
    # ops=1, non-default GOT 3,600 s, tiny GCT 30 min; one row per step.
    "ops1_got3600_gct30": dict(
        ops=1, got=3600, gct=30,
        tapes=[[3600, 3659, 3660, 4000, 3700, 5399, None],
               [3601, 4200]],
        expected={0: [0, 0, 1, 6, 6, 29, 30], 1: [0, 10, 30]},
    ),
    # ops=2, contract 16 (k mod 4 == 0): GCT 700 min.
    "ops2_k16_gct700": dict(
        ops=2, got=G, gct=gct_hour_aligned(16),
        tapes=[[82_800, 90_000, 97_199, 97_000, None, None],
               [60_000]],
        expected={0: [580, 699, 700], 1: [80, 700]},
    ),
    # ops=3, contract 81 (k mod 4 == 1): GCT 1,660 min; 7 real rows (not a
    # multiple of ops) + exactly 3 trailing pads -> the clamped slice is all-pad.
    "ops3_k81_gct1660": dict(
        ops=3, got=G, gct=gct_hour_aligned(81),
        tapes=[[55_200, 100_000, 120_000, 130_000, 140_000, 150_000, 154_799,
                None, None, None],
               [154_000, 154_100, 154_200]],
        expected={0: [1080, 1580, 1659, 1660], 1: [1650, 1660]},
    ),
    # ops=5, contract 6 (k mod 4 == 2): GCT 520 min; exact multiple + 5 pads.
    "ops5_k6_gct520": dict(
        ops=5, got=G, gct=gct_hour_aligned(6),
        tapes=[[m(10), m(200), m(246, 40), m(100), m(1),
                m(519, 59), m(300), m(519), m(400), m(518),
                None, None, None, None, None],
               [m(3), m(2), m(1)]],
        expected={0: [246, 519, 520], 1: [3, 520]},
    ),
}
OPS_EXTRA = 3


@pytest.fixture(scope="module")
def ops_results():
    """slow: one step_env compile per orders_per_step case (4 compiles)."""
    out = {}
    for name, c in OPS_CASES.items():
        jax.clear_caches()
        tapes = build_tapes(c["tapes"])
        cfg = {"tapes": tapes, "orders_per_step": c["ops"],
               "gct_minutes_from_got": c["gct"], "session_clock": "got_relative",
               "got_seconds": c["got"]}
        env = Session(cfg)
        n_steps = env.steps_per_episode + OPS_EXTRA
        per_tape = {b: _rollout_single(env, b, n_steps) for b in range(env.n_tapes)}
        # One compile per config: a second, equal-config instance reuses the entry.
        size_before = Session.step_env._cache_size()
        env2 = Session(dict(cfg))
        st2 = env2.reset(jax.random.PRNGKey(0))
        env2.step_env(jax.random.PRNGKey(0), st2, AgentOrder.none())
        size_after = Session.step_env._cache_size()
        out[name] = dict(env=env, tapes=tapes, per_tape=per_tape, n_steps=n_steps,
                         cache_growth=size_after - size_before, c=c)
    return out


@pytest.mark.slow
class TestOrdersPerStepVariants:
    @pytest.mark.parametrize("name", list(OPS_CASES))
    def test_time_trajectory_and_done_step(self, ops_results, name):
        """slow: shares the per-ops compile."""
        r = ops_results[name]
        c = r["c"]
        for b, exp in c["expected"].items():
            pt = r["per_tape"][b]
            np.testing.assert_array_equal(pt["time"][: len(exp)], exp)
            assert (pt["time"][len(exp) - 1:] == c["gct"]).all()
            assert _first_true(pt["done"]) == len(exp) - 1
            assert pt["done"][len(exp) - 1:].all()
            assert pt["dtype"] == jnp.int32

    @pytest.mark.parametrize("name", list(OPS_CASES))
    def test_matches_independent_reference(self, ops_results, name):
        """slow: shares the per-ops compile."""
        r = ops_results[name]
        c = r["c"]
        for b in range(r["env"].n_tapes):
            ref = ref_got_relative(r["tapes"][b, :, 6], c["ops"], c["got"], c["gct"],
                                   r["n_steps"])
            np.testing.assert_array_equal(r["per_tape"][b]["time"], ref)
            np.testing.assert_array_equal(r["per_tape"][b]["done"], ref >= c["gct"])

    @pytest.mark.parametrize("name", list(OPS_CASES))
    def test_longest_tape_done_is_last_episode_step(self, ops_results, name):
        """slow: shares the per-ops compile. Minimal (== ops) trailing pad."""
        r = ops_results[name]
        assert _first_true(r["per_tape"][0]["done"]) == r["env"].steps_per_episode - 1

    @pytest.mark.parametrize("name", list(OPS_CASES))
    def test_one_compile_per_config(self, ops_results, name):
        """slow: shares the per-ops compile."""
        assert ops_results[name]["cache_growth"] == 0


# ===========================================================================
# Constructor guards (fast; no compile)
# ===========================================================================
GUARD_FRAMES = [(G, 460), (G, 520), (G, 1660), (3600, 30), (1, 2)]


class TestConstructorGuards:
    @pytest.mark.parametrize("got,gct", GUARD_FRAMES)
    def test_row_exactly_at_got_accepted(self, got, gct):
        env = Session({"tapes": build_tapes([[got, got + 1, None, None, None, None]]),
                       "session_clock": "got_relative", "got_seconds": got,
                       "gct_minutes_from_got": gct, "orders_per_step": 4})
        assert env.session_clock == "got_relative"

    @pytest.mark.parametrize("got,gct", [f for f in GUARD_FRAMES if f[0] > 1])
    def test_row_one_second_before_got_rejected_naming_got_seconds(self, got, gct):
        with pytest.raises(ValueError, match="got_seconds"):
            Session({"tapes": build_tapes([[got - 1, got, None, None, None, None]]),
                     "session_clock": "got_relative", "got_seconds": got,
                     "gct_minutes_from_got": gct, "orders_per_step": 4})

    @pytest.mark.parametrize("got,gct", GUARD_FRAMES)
    def test_last_second_before_gct_accepted(self, got, gct):
        last = got + 60 * gct - 1
        Session({"tapes": build_tapes([[got, last, None, None, None, None]]),
                 "session_clock": "got_relative", "got_seconds": got,
                 "gct_minutes_from_got": gct, "orders_per_step": 4})

    @pytest.mark.parametrize("got,gct", GUARD_FRAMES)
    def test_row_at_gct_rejected_naming_gct_key(self, got, gct):
        at_gct = got + 60 * gct
        with pytest.raises(ValueError, match="gct_minutes_from_got"):
            Session({"tapes": build_tapes([[got, at_gct, None, None, None, None]]),
                     "session_clock": "got_relative", "got_seconds": got,
                     "gct_minutes_from_got": gct, "orders_per_step": 4})

    def test_quarter_hour_gct_for_k_mod_4_nonzero_would_accept_but_hour_aligned_is_tight(self):
        """k = 6: hour-aligned GCT 520 rejects a row at 86,400; a row at 86,399
        (last second before 00:00 D) is the last legal one."""
        assert gct_hour_aligned(6) == 520 and gct_hour_aligned(7) == 520
        Session({"tapes": build_tapes([[G, 86_399, None, None, None, None]]),
                 "session_clock": "got_relative", "gct_minutes_from_got": 520,
                 "orders_per_step": 4})
        with pytest.raises(ValueError, match="gct_minutes_from_got"):
            Session({"tapes": build_tapes([[G, 86_400, None, None, None, None]]),
                     "session_clock": "got_relative", "gct_minutes_from_got": 520,
                     "orders_per_step": 4})

    def test_out_of_frame_row_on_a_second_tape_rejected(self):
        tapes = build_tapes([[G, G + 60, None, None, None, None],
                             [G, 60, None, None, None, None]])
        with pytest.raises(ValueError, match="got_seconds"):
            Session({"tapes": tapes, "session_clock": "got_relative",
                     "gct_minutes_from_got": 460, "orders_per_step": 4})

    @pytest.mark.parametrize("ops", [1, 2, 3, 4, 5])
    def test_trailing_pad_equal_to_ops_accepted(self, ops):
        Session({"tapes": build_tapes([[G + i for i in range(7)] + [None] * ops]),
                 "session_clock": "got_relative", "gct_minutes_from_got": 460,
                 "orders_per_step": ops})

    @pytest.mark.parametrize("ops", [1, 2, 3, 4, 5])
    def test_trailing_pad_below_ops_rejected_naming_orders_per_step(self, ops):
        with pytest.raises(ValueError, match="orders_per_step"):
            Session({"tapes": build_tapes([[G + i for i in range(7)] + [None] * (ops - 1)]),
                     "session_clock": "got_relative", "gct_minutes_from_got": 460,
                     "orders_per_step": ops})

    def test_trailing_pad_checked_on_longest_tape_of_a_set(self):
        tapes = build_tapes([[G] * 3, [G + i for i in range(9)] + [None] * 3])
        with pytest.raises(ValueError, match="orders_per_step"):
            Session({"tapes": tapes, "session_clock": "got_relative",
                     "gct_minutes_from_got": 460, "orders_per_step": 4})

    def test_interior_pad_rows_do_not_count_as_trailing_pad(self):
        tapes = build_tapes([[G, None, None, None, None, G + 1, G + 2, None]])
        with pytest.raises(ValueError, match="orders_per_step"):
            Session({"tapes": tapes, "session_clock": "got_relative",
                     "gct_minutes_from_got": 460, "orders_per_step": 4})

    def test_interior_all_pad_slice_tape_t5_rejected_under_got_relative(self):
        """OMIE-SESSION-CLOCK-INTERIOR-PAD-GUARD-1 (A4): T5 (an interior all-pad
        slice) as tape 1 of {T0, T5} is rejected at construction under
        got_relative, and the same set constructs under the default clock."""
        tapes = build_tapes([_flat(E1_SLICES["T0"]), _flat(T5_SLICES)], n_padded=E1_N)
        with pytest.raises(ValueError, match=r"session_clock.*tape 1.*all-pad slice"):
            Session({"tapes": tapes, "orders_per_step": 4, "gct_minutes_from_got": 460,
                     "session_clock": "got_relative", "got_seconds": G})
        env = Session({"tapes": tapes, "orders_per_step": 4, "gct_minutes_from_got": 460})
        assert env.session_clock == "span_scaled"

    def test_all_pad_tape_set_constructs(self):
        Session({"tapes": build_tapes([[None] * 8]), "session_clock": "got_relative",
                 "gct_minutes_from_got": 460, "orders_per_step": 4})

    @pytest.mark.parametrize("bad", ["GOT_RELATIVE", "got-relative", "", "span-scaled",
                                     "Span_Scaled", "none", None])
    @pytest.mark.parametrize("with_tapes", [False, True])
    def test_unknown_session_clock_rejected_in_every_mode(self, bad, with_tapes):
        cfg = {"session_clock": bad}
        if with_tapes:
            cfg.update(tapes=build_tapes([[G, G + 1] + [None] * 4]),
                       gct_minutes_from_got=460)
        with pytest.raises(ValueError, match="session_clock"):
            Session(cfg)

    @pytest.mark.parametrize("tapes_spec", [
        [[60, 90, 120, None]],                    # pre-L1 / synthetic frame
        [[G, 82_800, 90_000, None, None, None]],  # rows at/after contract-0 GCT
        [[G, G + 1, G + 2, G + 3, G + 4]],        # zero trailing pad
    ])
    def test_same_tapes_construct_under_span_scaled(self, tapes_spec):
        env = Session({"tapes": build_tapes(tapes_spec), "gct_minutes_from_got": 460,
                       "orders_per_step": 4})
        assert env.session_clock == "span_scaled"
        Session({"tapes": build_tapes(tapes_spec), "gct_minutes_from_got": 460,
                 "orders_per_step": 4, "session_clock": "span_scaled"})

    def test_got_relative_rejects_default_synthetic_tape_set(self):
        """The zero-config synthetic tapes are session-relative (ts from 60 s)."""
        with pytest.raises(ValueError, match="got_seconds"):
            Session({"session_clock": "got_relative"})


# ===========================================================================
# JIT identity (fast; no compile)
# ===========================================================================
class TestJitIdentity:
    def _tapes(self):
        return build_tapes([[G, m(10), m(20), m(30), None, None, None, None]])

    def test_absent_keys_equal_explicit_defaults(self):
        a = Session({"tapes": self._tapes(), "gct_minutes_from_got": 460})
        b = Session({"tapes": self._tapes(), "gct_minutes_from_got": 460,
                     "session_clock": "span_scaled", "got_seconds": 55_200})
        assert a == b and hash(a) == hash(b)
        assert a.session_clock == "span_scaled" and a.got_seconds == 55_200

    def test_clock_modes_never_share_identity(self):
        a = Session({"tapes": self._tapes(), "gct_minutes_from_got": 460})
        b = Session({"tapes": self._tapes(), "gct_minutes_from_got": 460,
                     "session_clock": "got_relative"})
        assert a != b

    def test_got_seconds_distinguishes_identity(self):
        a = Session({"tapes": self._tapes(), "gct_minutes_from_got": 460,
                     "session_clock": "got_relative", "got_seconds": G})
        b = Session({"tapes": self._tapes(), "gct_minutes_from_got": 461,
                     "session_clock": "got_relative", "got_seconds": G - 60})
        c = Session({"tapes": self._tapes(), "gct_minutes_from_got": 461,
                     "session_clock": "got_relative", "got_seconds": G})
        assert a != b and b != c

    def test_equal_got_relative_configs_share_identity(self):
        cfg = {"tapes": self._tapes(), "gct_minutes_from_got": 460,
               "session_clock": "got_relative", "got_seconds": G}
        assert Session(dict(cfg)) == Session(dict(cfg))
        assert hash(Session(dict(cfg))) == hash(Session(dict(cfg)))

    def test_got_seconds_coerced_to_int(self):
        env = Session({"tapes": self._tapes(), "gct_minutes_from_got": 460,
                       "session_clock": "got_relative", "got_seconds": np.int64(G)})
        assert type(env.got_seconds) is int and env.got_seconds == G


# ===========================================================================
# E0 — default span_scaled clock: byte-identical, no all-pad termination
# ===========================================================================
E0_TAPES_SPEC = [
    [m(100), m(137), m(120), m(130), 69_000, 70_000, 69_500, 68_000,
     80_000, 82_799, 81_000, 82_000, None, None, None, None],
    [60_000] * 4 + [81_600, 81_000, 80_000, 81_500],
]


@pytest.fixture(scope="module")
def e0():
    """slow: one step_env compile of the default-clock env."""
    jax.clear_caches()
    tapes = build_tapes(E0_TAPES_SPEC)
    env = Session({"tapes": tapes, "orders_per_step": 4})   # keys absent, gct 60
    n_steps = env.steps_per_episode + 3
    per_tape = {b: _rollout_single(env, b, n_steps) for b in range(env.n_tapes)}
    return dict(env=env, tapes=tapes, per_tape=per_tape, n_steps=n_steps)


@pytest.mark.slow
class TestDefaultSpanScaledUnchanged:
    def test_span_is_set_max_ts(self, e0):
        """slow: shares the E0 compile."""
        assert e0["env"].tape_seconds_span == 82_799.0

    @pytest.mark.parametrize("b", [0, 1])
    def test_time_matches_pre_story_span_scaled_formula(self, e0, b):
        """slow: shares the E0 compile."""
        ref = ref_span_scaled(e0["tapes"][b, :, 6], 4, 60, 82_799.0, e0["n_steps"])
        np.testing.assert_array_equal(e0["per_tape"][b]["time"], ref)
        np.testing.assert_array_equal(e0["per_tape"][b]["done"], ref >= 60)

    def test_tc1_main_value_46(self, e0):
        """slow: shares the E0 compile. Span-scaled with span 82,799: 63,420 -> 46."""
        assert e0["per_tape"][0]["time"][0] == 46

    def test_all_pad_slice_never_terminates_under_default(self, e0):
        """slow: shares the E0 compile. Tape 1 ends at 81,600 -> time 59; the
        subsequent all-pad slices keep 59 and done never fires."""
        pt = e0["per_tape"][1]
        assert not pt["done"].any()
        assert (pt["time"][1:] == 59).all()

    def test_real_row_at_span_max_terminates_under_default(self, e0):
        """slow: shares the E0 compile."""
        pt = e0["per_tape"][0]
        assert _first_true(pt["done"]) == 2 and pt["done"][2:].all()


# ===========================================================================
# Real data (data/rawOMIE) — must RUN when the data directory is present
# ===========================================================================
@pytest.fixture(scope="module")
def real_cache(tmp_path_factory):
    if not DATA_PATH.is_dir():
        pytest.skip(f"real OMIE data not present ({DATA_PATH} missing)")
    return str(tmp_path_factory.mktemp("sc_audit_npz")) + "/"


def _first_missing_product_file(data_dir, dates):
    """First missing session file of the products of ``dates``, else ``None``.

    A product (D, k) reads ``orders_{D-1}.1``, ``orders_{D}.1``,
    ``trades_{D-1}.1`` and ``trades_{D}.1``; all four must be present for every
    delivery date, otherwise the loader refuses the product as incomplete.
    """
    for d_iso in dates:
        d = date.fromisoformat(d_iso)
        for kind in ("orders", "trades"):
            for day in (d - timedelta(days=1), d):
                name = f"{kind}_{day:%Y%m%d}.1"
                if not os.path.isfile(os.path.join(str(data_dir), name)):
                    return name
    return None


def _skip_unless_products_present(data_dir, dates):
    missing = _first_missing_product_file(data_dir, dates)
    if missing is not None:
        pytest.skip(f"real OMIE data incomplete: {missing} missing from {data_dir} "
                    "(partial data/rawOMIE, e.g. after tutorial T03)")


def _load_tapes(cache, contract_ids, dates):
    _skip_unless_products_present(DATA_PATH, dates)
    from omiecid_simulator.jaxomiecid.omie_data_loader import build_fixed_tape_set
    return np.asarray(build_fixed_tape_set(
        data_path=str(DATA_PATH), contract_ids=contract_ids, delivery_dates=dates,
        cache_path=cache, pad_extra=8,
    ))


@pytest.fixture(scope="module")
def heldout(real_cache):
    """slow: loads 6 held-out contract-0 products + one step_env compile; each
    tape rolled serially until 2 steps past its done (the vmapped matching
    engine is too slow on real books for the audit budget)."""
    tapes = _load_tapes(real_cache, [0], HELD_OUT_DATES)
    jax.clear_caches()
    env = Session({"tapes": tapes, "orders_per_step": 4, "gct_minutes_from_got": 460,
                   "session_clock": "got_relative", "got_seconds": G})
    n_steps = env.steps_per_episode + 2
    per_tape = [_rollout_single(env, b, n_steps, stop_extra=2) for b in range(env.n_tapes)]
    return dict(env=env, tapes=tapes, per_tape=per_tape)


@pytest.mark.slow
class TestRealHeldOut:
    def test_real_tapes_have_no_interior_pad_and_are_in_frame(self, heldout):
        """slow: real-data load. The interior-all-pad semantics are moot on real
        tapes: pads are strictly trailing; all rows in [GOT, GCT)."""
        for tape in heldout["tapes"]:
            ts = tape[:, 6]
            real = ts > 0
            last = np.flatnonzero(real)[-1]
            assert real[: last + 1].all()
            assert ts[real].min() >= G and ts[real].max() < G + 60 * 460

    def test_pre_registered_done_steps(self, heldout):
        """slow: v5.163 pre-registration: 216/210/175/310/409/263, mean 263.83,
        3/6 >= 256."""
        firsts = [_first_true(pt["done"]) for pt in heldout["per_tape"]]
        assert firsts == [216, 210, 175, 310, 409, 263]
        assert np.mean(firsts) == pytest.approx(263.83, abs=0.005)
        assert sum(f >= 256 for f in firsts) == 3

    def test_done_step_is_first_all_pad_slice(self, heldout):
        """slow: shares the held-out compile."""
        for pt, tape in zip(heldout["per_tape"], heldout["tapes"]):
            n_real = int((tape[:, 6] > 0).sum())
            f = _first_true(pt["done"])
            assert f == -(-n_real // 4)
            assert pt["done"][f:].all() and len(pt["done"]) == f + 3

    def test_ttg_last_real_step_and_zero_at_done(self, heldout):
        """slow: pre-registered time_to_gct 1,1,3,1,5,1 on the last real step."""
        last = []
        for pt in heldout["per_tape"]:
            f = _first_true(pt["done"])
            last.append(float(pt["ttg"][f - 1]))
            assert pt["ttg"][f] == 0.0 and pt["time"][f] == 460
            assert (pt["ttg"][f:] == 0.0).all() and (pt["time"][f:] == 460).all()
        assert last == [1.0, 1.0, 3.0, 1.0, 5.0, 1.0]

    def test_full_trajectory_matches_reference(self, heldout):
        """slow: shares the held-out compile."""
        for pt, tape in zip(heldout["per_tape"], heldout["tapes"]):
            ref = ref_got_relative(tape[:, 6], 4, G, 460, len(pt["time"]))
            np.testing.assert_array_equal(pt["time"], ref)
            assert pt["dtype"] == jnp.int32
            assert (np.diff(pt["time"]) >= 0).all()

    def test_session_starts_near_zero(self, heldout):
        """slow: shares the held-out compile. The first slice is minutes after
        GOT, not ~40 (span-scaled) or ~58 (pre-L1)."""
        assert all(pt["time"][0] <= 2 for pt in heldout["per_tape"])


@pytest.fixture(scope="module")
def train(real_cache):
    """slow: loads the 24 train products of the (inlined) shipped config + one
    compile; serial per-tape rollouts to 2 steps past done."""
    tapes = _load_tapes(real_cache, TRAIN_CONTRACT_IDS, TRAIN_DATES)
    jax.clear_caches()
    env_cfg = dict(TRAIN_ENV_CFG)
    env_cfg["tapes"] = tapes
    env = Session(env_cfg)
    n_steps = env.steps_per_episode
    per_tape = [_rollout_single(env, b, n_steps, stop_extra=2) for b in range(env.n_tapes)]
    return dict(env=env, tapes=tapes, per_tape=per_tape)


@pytest.mark.slow
class TestRealTrainConfig:
    def test_shipped_config_env_is_got_relative(self, train):
        """slow: real-data load."""
        env = train["env"]
        assert env.session_clock == "got_relative" and env.got_seconds == G
        assert env.gct_minutes_from_got == 460 and env.orders_per_step == 4
        assert env.n_tapes == 24 and env.steps_per_episode == 643

    def test_pre_registered_train_envelope(self, train):
        """slow: v5.163 probe: op4 done step [99, 641], mean 301.83, 14/24 >= 256."""
        firsts = [_first_true(pt["done"]) for pt in train["per_tape"]]
        assert None not in firsts, "every train episode must terminate within 643 steps"
        assert min(firsts) == 99 and max(firsts) == 641
        assert np.mean(firsts) == pytest.approx(301.83, abs=0.005)
        assert sum(f >= 256 for f in firsts) == 14

    def test_every_tape_sticky_after_done_and_matches_reference(self, train):
        """slow: shares the train compile."""
        for pt, tape in zip(train["per_tape"], train["tapes"]):
            f = _first_true(pt["done"])
            assert pt["done"][f:].all() and (pt["time"][f:] == 460).all()
            ref = ref_got_relative(tape[:, 6], 4, G, 460, len(pt["time"]))
            np.testing.assert_array_equal(pt["time"], ref)


@pytest.mark.parametrize("k,date", [(3, "2025-06-01"), (5, "2025-06-01"),
                                    (83, "2025-06-01"), (0, "2025-06-25")])
@pytest.mark.slow
def test_real_contract_hour_aligned_gct_is_tight(real_cache, k, date):
    """slow: real-data load (~3 s). For k mod 4 != 0 the hour-aligned GCT
    460 + 60·⌊k/4⌋ is accepted and GCT − 1 min is rejected (the product's last
    row lies in the final minute before the hourly gate)."""
    tapes = _load_tapes(real_cache, [k], [date])
    gct = gct_hour_aligned(k)
    env = Session({"tapes": tapes, "orders_per_step": 4, "gct_minutes_from_got": gct,
                   "session_clock": "got_relative"})
    assert env.gct_minutes_from_got == gct
    with pytest.raises(ValueError, match="gct_minutes_from_got"):
        Session({"tapes": tapes, "orders_per_step": 4, "gct_minutes_from_got": gct - 1,
                 "session_clock": "got_relative"})


@pytest.fixture(scope="module")
def real_k5(real_cache):
    """slow: one real k=5 (k mod 4 != 0) product, compile + rollout."""
    tapes = _load_tapes(real_cache, [5], ["2025-06-01"])
    jax.clear_caches()
    gct = gct_hour_aligned(5)
    env = Session({"tapes": tapes, "orders_per_step": 4, "gct_minutes_from_got": gct,
                   "session_clock": "got_relative"})
    n_steps = env.steps_per_episode + 2
    return dict(res=_rollout_single(env, 0, n_steps, stop_extra=2), tapes=tapes, gct=gct,
                n_steps=n_steps)


@pytest.mark.slow
def test_real_k5_product_replayed_to_exhaustion(real_k5):
    """slow: real k=5 rollout. GCT 520 (00:00 D); done on the first all-pad slice,
    time_to_gct >= 1 on the last real step, trajectory matches the reference."""
    r = real_k5
    ts = r["tapes"][0, :, 6]
    ref = ref_got_relative(ts, 4, G, r["gct"], len(r["res"]["time"]))
    np.testing.assert_array_equal(r["res"]["time"], ref)
    n_real = int((ts > 0).sum())
    f = _first_true(r["res"]["done"])
    assert f == -(-n_real // 4)
    assert r["res"]["done"][f:].all()
    assert r["res"]["ttg"][f - 1] >= 1.0 and r["res"]["ttg"][f] == 0.0
