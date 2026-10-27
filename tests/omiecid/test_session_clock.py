"""Locked behavioural contract for ``OMIE-SESSION-CLOCK-1`` (agile-plan v5.163).

Opt-in real-tape session clock. Two new config keys:

  * ``session_clock`` in {"span_scaled" (DEFAULT, byte-identical to today),
    "got_relative"}; any other value raises ``ValueError`` in every mode.
  * ``got_seconds`` (int, default 55,200 = 15:20 D-1 in the L1 tape frame,
    market-spec §3.1).

Under ``got_relative`` ONLY:

  * ``time = max(0, (max_ts_in_slice - got_seconds) // 60)`` (integer floor
    minutes since GOT; the monotone ``max`` with the prior time is kept);
  * the first ALL-PAD slice (no row with ts > 0: the tape is exhausted) sets
    ``time = gct_minutes_from_got``;
  * ``done = time >= gct_minutes_from_got``, which covers both clauses and is
    sticky (the STOP-AT-DONE M1 mask relies on it);
  * constructor guards (``ValueError`` naming the key): a real row with
    ts < ``got_seconds``; a real row with ts >= ``got_seconds`` +
    60 * ``gct_minutes_from_got``; fewer than ``orders_per_step`` trailing pad
    rows on the longest tape.

Contract 0 GCT is 460 min after GOT, i.e. second 82,800 in the L1 frame
(market-spec §3.1 / §13.4). Every real row of a product is sent in
[GOT, GCT), so the tape-exhaustion clause is the ONLY terminator on real data.

RELEASE NOTE (REL-07): the research eval-whitelist / held-out-reader / training
config deliverables (``omie_eval``, ``omie_heldout_eval``,
``config/env_configs/omie_settle_control_op4*_train.json``) and the RL
observation / dense-GCT reward channels are not part of the release; their
tests (TC3, TC4, TC8, the eval-whitelist class) were removed. The participant's
order is the public ``AgentOrder``; ``time_to_gct`` (``ttg`` below) is the
session clock ``gct_minutes_from_got - state.time`` (the research obs[4]
channel carried exactly this value).

Test-first map (card numbering) and the expected result on the pre-change code:

  TC1  time/time_to_gct after a slice at GOT + 137 min            RED
  TC2  done on the first all-pad slice, time == 460               RED
  TC3  dense-GCT weight at GOT + 230 min == 0.5 * lambda          RED
  TC4  per-tape eval env clock == training env clock              RED
  TC5  default (keys absent) unchanged; all-pad does NOT end       GREEN
  TC6  one compile per config                                      RED (identity)
  TC7  constructor guards raise under got_relative only            RED (raises)
  TC8  held-out reader + per-tape eval env keep the clock          RED
  TC9  done / time sticky after the first all-pad slice            RED
  Real data (held-out contract-0 L1 tapes, op4): done step per date
  216/210/175/310/409/263, time_to_gct 1/1/3/1/5/1 on the last real step,
  0 at done                                                        RED

Every expected value below was re-derived by the test author from the rule
above (and, for real data, from the loaded tapes' real-row counts), not read off
an implementation.

STATE SOURCE / RUNTIME: synthetic in-source tapes (ts in 55,200..82,799) for
TC1-TC9; the real ``data/rawOMIE`` held-out tapes for the integration class.
Static / construction-only tests are fast. Tests that compile and roll
``step_env`` exceed the 1 s budget and carry ``slow`` + ``jit``.
"""

import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from omiecid_simulator.omiecid.omie_session import AgentOrder, Session
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction, Validity, Status, Side,
)


# ---------------------------------------------------------------------------
# Constants (contract 0, L1 frame).
# ---------------------------------------------------------------------------
GOT_SECONDS = 55_200          # 15:20 D-1
GCT_MIN = 460                 # contract 0: GCT - GOT in minutes
GCT_SECOND = GOT_SECONDS + 60 * GCT_MIN   # 82,800 = 23:00 D-1
OPS = 4

BUY = int(Side.BUY)
SELL = int(Side.SELL)

NOOP = AgentOrder.none()
# Research action 5 (BUY, CROSS bucket, qty bucket 0) priced a resting NON BUY
# at the post-background best ask with quantity qty_min. It is only ever
# submitted on step 0, where every tape below has best ask 51.0 after the
# first slice (resting SELLs at 51.0 / 51.5, or 51 / 52 on the dense tape).
BUY_CROSS_SMALL = AgentOrder.create(
    side=BUY, price=51.0, quantity=float(OMIEProductConfig().qty_min),
    exec_restriction=int(ExecRestriction.NON),
)

CLOCK_KW = {"session_clock": "got_relative", "got_seconds": GOT_SECONDS}


# ---------------------------------------------------------------------------
# Tape builders (13-column int32 loader layout).
# ---------------------------------------------------------------------------
def _row(side, price, qty, oid, ts):
    pc = OMIEProductConfig()
    return [
        0, int(side),
        int(round((price - float(pc.price_min)) / float(pc.price_tick))),
        int(round(qty / float(pc.qty_tick))),
        int(oid), 777, int(ts),
        int(ExecRestriction.NON), int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0,
    ]


def _pad():
    return [0, 0, 0, 0, -1, 0, -1, int(ExecRestriction.NON),
            int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0]


def _resting_rows(timestamps, oid0):
    """Non-crossing resting limit orders, alternating BUY (<= 49) / SELL (>= 51)."""
    rows = []
    for i, ts in enumerate(timestamps):
        if i % 2 == 0:
            rows.append(_row(BUY, 49.0 - 0.5 * (i // 2), 10.0, oid0 + i, ts))
        else:
            rows.append(_row(SELL, 51.0 + 0.5 * (i // 2), 10.0, oid0 + i, ts))
    return rows


def _tape(timestamps, n_pad, oid0=100):
    return _resting_rows(timestamps, oid0) + [_pad() for _ in range(n_pad)]


def _stack(tapes):
    """Right-pad every tape with pad rows to a common length and stack."""
    n = max(len(t) for t in tapes)
    return np.asarray(
        [list(t) + [_pad() for _ in range(n - len(t))] for t in tapes],
        dtype=np.int32,
    )


# The synthetic tape SET (one env, one compile, tape chosen by ``tape_idx``).
# Each list is the real-row timestamps; slices are OPS = 4 rows each.
T_SPAN = [55_260, 57_000, 60_000, 63_420,          # slice 0: max GOT + 137 min
          69_000, 75_000, 80_000, 82_799]          # slice 1: max 82,799 (set max)
T_FLOOR = [55_200, 55_210, 55_230, 55_259,         # slice 0: max GOT + 59 s -> 0
           57_000, 60_000, 63_000, 63_479,         # slice 1: GOT + 137 min + 59 s
           64_000, 66_000, 68_000, 69_000]         # slice 2: GOT + 230 min
T_EXHAUST = [55_300, 56_000, 58_000, 60_000,       # slice 0: max 60,000 -> 80
             70_000, 76_000, 80_000, 81_550,       # slice 1: max 81,550 -> 439
             81_560, 81_570, 81_580, 81_600]       # slice 2: last order 22:40 -> 440
T_SHORT = [55_400, 58_000, 62_000, 66_000,         # slice 0: max 66,000 -> 180
           67_000, 68_000, 69_000, 70_000]         # slice 1: max 70,000 -> 246

IDX_SPAN, IDX_FLOOR, IDX_EXHAUST, IDX_DENSE, IDX_SHORT = 0, 1, 2, 3, 4


def _dense_tape():
    """TC3 tape: slice 0 holds two resting asks (51, 52) and ends at GOT + 230
    min (ts 69,000); slice 1 holds only non-crossing bids and ends at 75,000
    (GOT + 330 min); then pads."""
    rows = [
        _row(BUY, 49.0, 10.0, 400, 55_300),
        _row(SELL, 51.0, 10.0, 401, 60_000),
        _row(BUY, 48.0, 10.0, 402, 65_000),
        _row(SELL, 52.0, 10.0, 403, 69_000),
        _row(BUY, 47.0, 10.0, 404, 70_000),
        _row(BUY, 46.0, 10.0, 405, 72_000),
        _row(BUY, 45.0, 10.0, 406, 74_000),
        _row(BUY, 44.0, 10.0, 407, 75_000),
    ]
    return rows


def _tape_set():
    return _stack([
        _tape(T_SPAN, 8, oid0=100),
        _tape(T_FLOOR, 8, oid0=200),
        _tape(T_EXHAUST, 8, oid0=300),
        _dense_tape() + [_pad() for _ in range(8)],
        _tape(T_SHORT, 8, oid0=500),
    ])


def _base_cfg(tapes, **extra):
    cfg = {
        "tapes": np.asarray(tapes, dtype=np.int32),
        "orders_per_step": OPS,
        "gct_minutes_from_got": GCT_MIN,
    }
    cfg.update(extra)
    return cfg


def _got_minutes(ts):
    """The card's rule, written independently: floor minutes since GOT."""
    return max(0, (int(ts) - GOT_SECONDS) // 60)


def _rollout(env, tape_idx, n_steps, actions=None, seed=0):
    """Eager walk of the public ``reset`` / ``step_env`` on a fixed tape.

    ``actions`` is a list of ``AgentOrder`` (the last one repeats). Returns a
    list of per-step dicts (time, done, time_to_gct = gct - time, inventory)."""
    key = jax.random.PRNGKey(seed)
    state = env.reset(key)
    state = state.replace(tape_idx=jnp.int32(tape_idx))
    out = []
    for i in range(n_steps):
        k, key = jax.random.split(key)
        order = NOOP if actions is None else actions[min(i, len(actions) - 1)]
        state, _report, done = env.step_env(k, state, order)
        t = int(np.asarray(state.time))
        out.append({
            "time": t,
            "done": bool(np.asarray(done)),
            "ttg": float(env.gct_minutes_from_got - t),
            "inv": float(np.asarray(state.agent_inventory)[0]),
        })
    return out


# ---------------------------------------------------------------------------
# Module-scoped envs (one compile per config).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tapes():
    return _tape_set()


@pytest.fixture(scope="module")
def clock_env(tapes):
    return Session(_base_cfg(tapes, **CLOCK_KW))


@pytest.fixture(scope="module")
def default_env(tapes):
    return Session(_base_cfg(tapes))


# ===========================================================================
# Fixture sanity (fast): the synthetic set satisfies the card's constraints.
# ===========================================================================
class TestFixtureSanity:

    def test_set_max_timestamp_is_the_last_second_before_gct(self, tapes):
        assert int(tapes[:, :, 6].max()) == GCT_SECOND - 1 == 82_799

    def test_every_real_row_lies_in_got_gct(self, tapes):
        ts = tapes[:, :, 6]
        real = ts[ts > 0]
        assert real.min() >= GOT_SECONDS and real.max() < GCT_SECOND

    def test_short_and_exhaust_tapes_end_below_the_set_max(self, tapes):
        # TC4 is only RED on the old code if the chosen tape's max < set max.
        assert int(tapes[IDX_SHORT, :, 6].max()) == 70_000 < 82_799
        # TC2: a real slice with max ts >= 80,920 precedes the last real slice.
        assert max(T_EXHAUST[4:8]) >= 80_920 and max(T_EXHAUST) == 81_600

    def test_longest_tape_has_at_least_orders_per_step_trailing_pads(self, tapes):
        n_real = (tapes[:, :, 6] > 0).sum(axis=1)
        assert tapes.shape[1] - int(n_real.max()) >= OPS


# ===========================================================================
# TC1 — time since GOT after a slice at GOT + 137 min (RED on old code).
# ===========================================================================
@pytest.mark.slow
@pytest.mark.jit
class TestTC1GotRelativeTime:
    """slow: compiles and rolls step_env on the synthetic set."""

    def test_time_is_137_after_slice_at_got_plus_137_min(self, clock_env):
        assert max(T_SPAN[:4]) == GOT_SECONDS + 60 * 137
        steps = _rollout(clock_env, IDX_SPAN, 1)
        assert steps[0]["time"] == 137

    def test_time_to_gct_obs_is_323_after_slice_at_got_plus_137_min(self, clock_env):
        steps = _rollout(clock_env, IDX_SPAN, 1)
        assert steps[0]["ttg"] == pytest.approx(float(GCT_MIN - 137))  # 323
        assert not steps[0]["done"]

    def test_slice_ending_at_82799_maps_to_459_and_is_not_done(self, clock_env):
        # (82,799 - 55,200) // 60 = 459: the last real second never reaches GCT.
        steps = _rollout(clock_env, IDX_SPAN, 2)
        assert steps[1]["time"] == _got_minutes(82_799) == 459
        assert steps[1]["ttg"] == pytest.approx(1.0)
        assert not steps[1]["done"]

    def test_time_uses_integer_floor_not_rounding(self, clock_env):
        steps = _rollout(clock_env, IDX_FLOOR, 3)
        # slice 0 max = GOT + 59 s -> 0 (rounding would give 1)
        assert steps[0]["time"] == 0
        # slice 1 max = GOT + 137 min + 59 s -> 137 (rounding would give 138)
        assert steps[1]["time"] == 137
        # slice 2 max = GOT + 230 min exactly -> 230
        assert steps[2]["time"] == 230

    def test_got_seconds_defaults_to_55200(self, tapes):
        env = Session(_base_cfg(tapes, session_clock="got_relative"))
        assert env == Session(_base_cfg(tapes, **CLOCK_KW))
        steps = _rollout(env, IDX_SPAN, 1)
        assert steps[0]["time"] == 137


# ===========================================================================
# TC2 + TC9 — tape-exhaustion terminator and stickiness (RED on old code).
# ===========================================================================
@pytest.mark.slow
@pytest.mark.jit
class TestTC2TC9TapeExhaustion:
    """slow: compiles and rolls step_env on the synthetic set / a single tape."""

    EXPECTED_TIMES = [80, 439, 440]   # (60,000 / 81,550 / 81,600 - 55,200) // 60

    def _check_exhaust_trajectory(self, steps):
        assert [s["time"] for s in steps[:3]] == self.EXPECTED_TIMES
        assert [s["done"] for s in steps[:3]] == [False, False, False], (
            "done must NOT fire before the tape is exhausted (every real row is "
            "before GCT)"
        )
        assert steps[2]["ttg"] == pytest.approx(20.0)
        # Step 3 reads the first all-pad slice: done, time set to GCT.
        assert steps[3]["done"] is True
        assert steps[3]["time"] == GCT_MIN
        assert steps[3]["ttg"] == pytest.approx(0.0)

    def test_done_fires_on_first_all_pad_slice_in_training_set(self, clock_env):
        steps = _rollout(clock_env, IDX_EXHAUST, 4)
        self._check_exhaust_trajectory(steps)

    def test_done_fires_on_first_all_pad_slice_single_tape_env(self, tapes):
        # The per-tape env (the eval construction). The old per-tape span would
        # make 81,550 map to round(81,550 * 460 / 81,600) = 460 and fire on
        # step 1, before exhaustion.
        single = Session(_base_cfg(tapes[IDX_EXHAUST][None], **CLOCK_KW))
        steps = _rollout(single, 0, 4)
        self._check_exhaust_trajectory(steps)

    def test_done_and_time_stay_sticky_after_first_all_pad_slice(self, clock_env):
        # TC9: well past the end of the tape (the clamped slice keeps reading
        # pad rows): done stays True and time stays at GCT on every step.
        n = clock_env.steps_per_episode + 3
        steps = _rollout(clock_env, IDX_EXHAUST, n)
        tail = steps[3:]
        assert len(tail) >= 3
        assert all(s["done"] for s in tail)
        assert all(s["time"] == GCT_MIN for s in tail)
        assert all(s["ttg"] == pytest.approx(0.0) for s in tail)

    def test_short_tape_ends_on_its_own_first_all_pad_slice(self, clock_env):
        steps = _rollout(clock_env, IDX_SHORT, 4)
        assert [s["time"] for s in steps] == [
            _got_minutes(66_000), _got_minutes(70_000), GCT_MIN, GCT_MIN,
        ] == [180, 246, 460, 460]
        assert [s["done"] for s in steps] == [False, False, True, True]


# ===========================================================================
# TC5 — default clock unchanged (GREEN on both).
# ===========================================================================
@pytest.mark.slow
@pytest.mark.jit
class TestTC5DefaultUnchanged:
    """slow: compiles and rolls step_env under the default clock."""

    def test_default_all_pad_slice_does_not_terminate(self, default_env):
        # span_scaled, set span 82,799: 70,000 -> round(70,000*460/82,799) = 389.
        assert round(70_000 * GCT_MIN / 82_799) == 389
        n = default_env.steps_per_episode + 2
        steps = _rollout(default_env, IDX_SHORT, n)
        assert steps[1]["time"] == 389
        assert all(s["time"] == 389 for s in steps[1:])
        assert not any(s["done"] for s in steps)

    def test_default_span_scaled_maps_the_set_max_to_gct(self, default_env):
        steps = _rollout(default_env, IDX_SPAN, 2)
        assert steps[0]["time"] == round(63_420 * GCT_MIN / 82_799)  # 352
        assert steps[1]["time"] == GCT_MIN and steps[1]["done"]

    def test_explicit_span_scaled_trajectory_equals_keys_absent(self, default_env, tapes):
        # got_seconds is inert under span_scaled (and no guard applies).
        explicit = Session(_base_cfg(tapes, session_clock="span_scaled", got_seconds=1))
        n = default_env.steps_per_episode + 1
        for idx in (IDX_SPAN, IDX_EXHAUST, IDX_DENSE, IDX_SHORT):
            acts = [BUY_CROSS_SMALL, NOOP]
            assert _rollout(explicit, idx, n, acts) == _rollout(default_env, idx, n, acts)


# ===========================================================================
# TC6 — identity key and one compile per config.
# ===========================================================================
class TestTC6IdentityAndCompile:

    def test_default_env_exposes_span_scaled_clock_attributes(self, tapes):
        env = Session(_base_cfg(tapes))
        assert env.session_clock == "span_scaled"
        assert env.got_seconds == GOT_SECONDS

    def test_explicit_default_keys_equal_keys_absent(self, tapes):
        a = Session(_base_cfg(tapes))
        b = Session(_base_cfg(tapes, session_clock="span_scaled", got_seconds=GOT_SECONDS))
        assert a == b and hash(a) == hash(b)

    def test_session_clock_is_part_of_the_identity(self, tapes):
        a = Session(_base_cfg(tapes))
        b = Session(_base_cfg(tapes, **CLOCK_KW))
        assert a != b
        assert hash(a) != hash(b)

    def test_got_seconds_is_part_of_the_identity(self, tapes):
        # 55,140 keeps every real ts >= got_seconds; the GCT second becomes
        # 55,140 + 27,600 = 82,740, so drop the 82,799 tape from this set.
        sub = _stack([_tape(T_EXHAUST, 8), _tape(T_SHORT, 8)])
        a = Session(_base_cfg(sub, **CLOCK_KW))
        b = Session(_base_cfg(sub, session_clock="got_relative", got_seconds=55_140))
        assert a != b
        assert hash(a) != hash(b)

    def test_byte_identical_got_relative_envs_are_equal(self, tapes):
        a = Session(_base_cfg(tapes, **CLOCK_KW))
        b = Session(_base_cfg(tapes, **CLOCK_KW))
        assert a == b and hash(a) == hash(b)

    @pytest.mark.slow
    @pytest.mark.jit
    def test_one_step_env_compile_per_got_relative_config(self, tapes):
        """slow: compiles step_env for a fresh config."""
        # A config no other test uses => fresh key. ``tape_seconds_span`` is
        # part of the jit identity and inert under got_relative (the research
        # test varied the removed ``dense_gct_lambda`` reward key instead).
        cfg = dict(_base_cfg(tapes, **CLOCK_KW), tape_seconds_span=12_345.0)
        before = Session.step_env._cache_size()
        env = Session(cfg)
        for idx in (IDX_SPAN, IDX_EXHAUST, IDX_SHORT):
            _rollout(env, idx, env.steps_per_episode + 1, [BUY_CROSS_SMALL, NOOP])
        mid = Session.step_env._cache_size()
        assert mid - before == 1
        again = Session(dict(cfg))
        _rollout(again, IDX_DENSE, 3)
        assert Session.step_env._cache_size() == mid


# ===========================================================================
# TC7 — constructor guards (fast: construction only).
# ===========================================================================
class TestTC7ConstructorGuards:

    @pytest.mark.parametrize("bad", ["wall_clock", "GOT_RELATIVE", ""])
    def test_invalid_session_clock_rejected_default_tapes(self, bad):
        with pytest.raises(ValueError, match="session_clock"):
            Session({"session_clock": bad})

    def test_invalid_session_clock_rejected_l1_tapes(self, tapes):
        with pytest.raises(ValueError, match="session_clock"):
            Session(_base_cfg(tapes, session_clock="minutes"))

    # -- ts < got_seconds (wrong frame) -------------------------------------
    def test_row_before_got_raises_naming_got_seconds(self):
        sub = _stack([_tape([55_199, 60_000, 61_000, 62_000], 8)])
        with pytest.raises(ValueError, match="got_seconds"):
            Session(_base_cfg(sub, **CLOCK_KW))

    def test_pre_l1_synthetic_default_tapes_raise_under_got_relative(self):
        # The built-in synthetic set uses ts from 60 s (a different frame).
        with pytest.raises(ValueError, match="got_seconds"):
            Session({"session_clock": "got_relative"})

    def test_row_exactly_at_got_constructs(self):
        sub = _stack([_tape([55_200, 60_000, 61_000, 62_000], 8)])
        Session(_base_cfg(sub, **CLOCK_KW))

    # -- ts >= GCT second (GCT does not match the loaded contract) ----------
    def test_row_at_gct_second_raises_naming_gct(self):
        sub = _stack([_tape([55_300, 60_000, 70_000, GCT_SECOND], 8)])
        with pytest.raises(ValueError, match="gct_minutes_from_got"):
            Session(_base_cfg(sub, **CLOCK_KW))

    def test_row_one_second_before_gct_constructs(self):
        sub = _stack([_tape([55_300, 60_000, 70_000, GCT_SECOND - 1], 8)])
        Session(_base_cfg(sub, **CLOCK_KW))

    def test_default_gct_60_with_contract0_tape_raises(self, tapes):
        # 55,200 + 60*60 = 58,800 <= many real rows: a forgotten GCT override.
        cfg = _base_cfg(tapes, **CLOCK_KW)
        cfg["gct_minutes_from_got"] = 60
        with pytest.raises(ValueError, match="gct_minutes_from_got"):
            Session(cfg)

    # -- trailing pad rows ---------------------------------------------------
    def test_fewer_than_orders_per_step_trailing_pads_raises(self):
        sub = _stack([_tape([55_300, 60_000, 70_000, 80_000], OPS - 1)])
        with pytest.raises(ValueError, match=r"orders_per_step|pad"):
            Session(_base_cfg(sub, **CLOCK_KW))

    def test_pad_guard_checks_the_longest_tape(self):
        # The short tape has plenty of pads; the longest has only 3.
        long_t = _tape([55_300, 60_000, 70_000, 72_000, 74_000, 76_000, 78_000,
                        80_000], OPS - 1)
        sub = _stack([long_t, _tape([55_300, 60_000], 0)])
        assert (sub[1, :, 6] <= 0).sum() >= OPS
        with pytest.raises(ValueError, match=r"orders_per_step|pad"):
            Session(_base_cfg(sub, **CLOCK_KW))

    def test_exactly_orders_per_step_trailing_pads_constructs(self):
        sub = _stack([_tape([55_300, 60_000, 70_000, 80_000], OPS)])
        Session(_base_cfg(sub, **CLOCK_KW))

    def test_orders_per_step_one_needs_one_trailing_pad(self):
        sub0 = _stack([_tape([55_300, 60_000], 0)])
        cfg0 = _base_cfg(sub0, **CLOCK_KW)
        cfg0["orders_per_step"] = 1
        with pytest.raises(ValueError, match=r"orders_per_step|pad"):
            Session(cfg0)
        sub1 = _stack([_tape([55_300, 60_000], 1)])
        cfg1 = _base_cfg(sub1, **CLOCK_KW)
        cfg1["orders_per_step"] = 1
        Session(cfg1)

    # -- the same tapes under span_scaled construct fine --------------------
    @pytest.mark.parametrize("timestamps, n_pad, gct", [
        ([55_199, 60_000, 61_000, 62_000], 8, GCT_MIN),     # before GOT
        ([55_300, 60_000, 70_000, GCT_SECOND], 8, GCT_MIN),  # at GCT
        ([55_300, 60_000, 70_000, 80_000], OPS - 1, GCT_MIN),  # too few pads
        ([55_300, 60_000, 70_000, 80_000], 8, 60),          # GCT mismatch
    ])
    @pytest.mark.parametrize("clock_kw", [{}, {"session_clock": "span_scaled"}])
    def test_guards_inactive_under_span_scaled(self, timestamps, n_pad, gct, clock_kw):
        sub = _stack([_tape(timestamps, n_pad)])
        cfg = _base_cfg(sub, got_seconds=GOT_SECONDS, **clock_kw)
        cfg["gct_minutes_from_got"] = gct
        Session(cfg)

    def test_default_synthetic_env_still_constructs(self):
        Session({})
        Session({"session_clock": "span_scaled"})


# ===========================================================================
# Real data (integration): held-out contract-0 L1 tapes under got_relative.
# ===========================================================================
# Re-derived by the test author from the loaded tapes (real-row counts n):
# op4 done step = ceil(n / 4) (0-indexed first all-pad slice); time_to_gct on
# the last real step = 460 - (max_ts - 55,200) // 60.
HELD_OUT_EXPECTED = {
    "2025-06-25": (216, 1),
    "2025-06-26": (210, 1),
    "2025-06-27": (175, 3),
    "2025-06-28": (310, 1),
    "2025-06-29": (409, 5),
    "2025-06-30": (263, 1),
}


@pytest.fixture(scope="module")
def held_out(tmp_path_factory):
    if not os.path.isdir("data/rawOMIE"):
        pytest.skip("real OMIE data not present (data/rawOMIE missing)")
    from omiecid_simulator.jaxomiecid.omie_data_loader import build_fixed_tape_set
    # The research eval split (``load_eval_split``) is not shipped; its six
    # held-out dates are the keys of HELD_OUT_EXPECTED.
    dates = list(HELD_OUT_EXPECTED)
    t = build_fixed_tape_set(
        data_path="data/rawOMIE", contract_ids=[0], delivery_dates=dates,
        cache_path=str(tmp_path_factory.mktemp("sc_cache")),
    )
    return dates, np.asarray(t)


def _done_walk(env, tape_idx, max_steps):
    """NOOP walk; returns (done_step, ttg on the step before done, time at done,
    ttg at done, done/time sticky on the next step); ttg = gct - time."""
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    state = state.replace(tape_idx=jnp.int32(tape_idx))
    prev_ttg = None
    for i in range(max_steps):
        k, key = jax.random.split(key)
        state, _report, done = env.step_env(k, state, NOOP)
        ttg = float(env.gct_minutes_from_got - int(np.asarray(state.time)))
        if bool(np.asarray(done)):
            t_done = int(np.asarray(state.time))
            k, key = jax.random.split(key)
            state2, _report2, done2 = env.step_env(k, state, NOOP)
            sticky = bool(np.asarray(done2)) and int(np.asarray(state2.time)) == t_done
            return i, prev_ttg, t_done, ttg, sticky
        prev_ttg = ttg
    return None, prev_ttg, None, None, None


@pytest.mark.slow
@pytest.mark.jit
class TestRealHeldOutSessionClock:
    """slow: loads the real held-out L1 tapes and rolls ~1,600 op4 steps."""

    def test_real_rows_lie_in_got_gct(self, held_out):
        _dates, t = held_out
        ts = t[:, :, 6]
        assert ts[ts > 0].min() >= GOT_SECONDS
        assert ts.max() < GCT_SECOND

    def test_done_step_per_held_out_date(self, held_out):
        dates, t = held_out
        env = Session({
            "tapes": t, "orders_per_step": OPS, "gct_minutes_from_got": GCT_MIN,
            **CLOCK_KW,
        })
        got = {}
        for i, d in enumerate(dates):
            done_step, ttg_last, t_done, ttg_done, sticky = _done_walk(
                env, i, env.steps_per_episode + 2)
            got[d] = (done_step, ttg_last)
            assert t_done == GCT_MIN, d
            assert ttg_done == pytest.approx(0.0), d
            assert sticky, d
        assert got == {d: (s, float(g)) for d, (s, g) in HELD_OUT_EXPECTED.items()}
