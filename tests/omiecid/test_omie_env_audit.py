"""Independent audit tests for P3.2a — Session (reset/step + inline interleave).

Audit mode (post-implementation verification). These tests are authored
*independently* from the original P3.2a locked suite and probe **domain
correctness** beyond happy-path shape checks, derived from the same sprint
requirements:

  * ``docs/omiecid/agile-plan.md`` §Story P3.2a
  * ``docs/omiecid/migration-plan.md`` §P3.2 "Per-step tape-slice contract",
    §Reward, §"Trade extraction detector invariant"
  * ``docs/omiecid/market-spec.md`` (GOT/GCT, anonymity §7/§9.2)

Release translation (REL-07): the participant's orders are the public
``AgentOrder`` (``AgentOrder.none()`` is the former NOOP action index 0; a real
order carries the economic fields the research action decode produced for the
book in each test — side, price at the touch it targeted, quantity, NON / GFS).
The realized cash of a step is ``FillReport.cash_delta_eur`` (unscaled EUR) and
the visible book is read with ``Session.top_of_book``. The RL tests
(reward scaling, the JaxMARL dones dict, the observation schema / width, the
Discrete(13) action space) were removed with the RL layer.

Focus areas (each a domain property a subtly-wrong implementation could break):
  1. Cash sign correctness: +1·settlement for SELL, −1·settlement for BUY.
  2. Fill-isolation: the whole-row diff wraps the AGENT ``_submit_order`` — a
     step where the BACKGROUND tape trades but the agent submits no order must
     yield a cash delta of exactly 0.0 (no background-fill leakage).
  3. Interleave ordering (background-then-agent): the agent acts against the
     POST-background book and can fill against liquidity the background placed
     within the same step.
  4. tape_cursor / episode-length arithmetic: cursor advances by exactly
     ``orders_per_step``; ``steps_per_episode = ceil(N_orders_padded /
     orders_per_step)``; the final short step reads clamped pad rows without OOB.
  5. done semantics at the GCT boundary: monotone, fires exactly when
     ``time >= GCT``.
  6. Anonymity (HARD RULE): the visible book reader exposes only anonymous
     aggregated VISIBLE volume; a foreign iceberg's hidden qty never moves
     ``q_bid``/``q_ask``.
  7. No-order invariance: the empty order leaves the agent's own book
     contribution unchanged (structural, stronger than cash==0 alone).
  8. vmap one-compile: stepping the session across the empty order and the 12
     order variants of the former action space compiles the jitted
     ``step_env`` exactly once; the vmapped step runs under an OUTER
     ``enable_x64`` context (the interleave reaches a scoped-x64 int64 leg in
     ``extract_fills`` that does not lower under ``jax.vmap`` without it —
     a known finding, exercised here, NOT worked around by weakening the test).

Environment / tiering:
  * Run: ``CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python3 -m pytest \\
    tests/omiecid/test_omie_env_audit.py -v -n0`` (GPU 0 pinned; harness
    CPU-default via ``tests/omiecid/conftest.py``).
  * ``jax.jit`` / ``vmap`` compilation tests carry ``@pytest.mark.jit``
    (pre-push tier), NOT ``@pytest.mark.slow``. Structural / small-fixture
    contract checks stay in the fast tier (< 1s each).
  * Numeric cash assertions use ``atol``/``rtol`` (float32); the no-order
    cash exact-0.0 assertion is a true structural zero (empty agent whole-row
    diff).

Import-guarded skip: if a public interface this audit needs is not exposed, the
module skips cleanly at collection with a clear reason (the implementation is
finished and on disk, so the tests are expected to run for real).
"""

from __future__ import annotations

import contextlib
import functools
import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import jax.experimental

# ---------------------------------------------------------------------------
# Import guard — skip the whole module cleanly if the P3.2a surface is absent.
# ---------------------------------------------------------------------------
_IMPORT_ERR = None
try:
    from omiecid_simulator.omiecid.omie_session import AgentOrder, Session
    from omiecid_simulator.omiecid.omie_states import OMIEContractState
    from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig
    from omiecid_simulator.omiecid.order_types import (
        Side,
        ExecRestriction,
        Validity,
        Status,
    )
    from omiecid_simulator.jaxomiecid.omie_data_loader import _replay_orders
except Exception as exc:  # pragma: no cover - collection-time guard
    _IMPORT_ERR = exc

pytestmark = pytest.mark.skipif(
    _IMPORT_ERR is not None,
    reason=f"P3.2a Session surface unavailable: {_IMPORT_ERR}",
)


# ---------------------------------------------------------------------------
# Synthetic-tape helpers (contract/unit tier — inputs visible in-source,
# microsecond runtime). The 13-column int32 loader row layout is:
#   [contract_id, side, price_int, qty_int, order_id, agent_id, timestamp,
#    exec_restriction, validity, expiry_time, status, peak_qty_int, ppd_int]
# ---------------------------------------------------------------------------
_PC = OMIEProductConfig()
_BG_AGENT_ID = 777  # anonymous background trader id (never the agent's)


def _enc_price(p: float) -> int:
    return int(round((p - _PC.price_min) / _PC.price_tick))


def _enc_qty(q: float) -> int:
    return int(round(q / _PC.qty_tick))


def _row(side, price, qty, oid, ts, exec_r=int(ExecRestriction.NON), peak=0.0):
    return [
        0,
        int(side),
        _enc_price(price),
        _enc_qty(qty),
        int(oid),
        _BG_AGENT_ID,
        int(ts),
        int(exec_r),
        int(Validity.GFS),
        0,
        int(Status.ACTIVE),
        _enc_qty(peak),
        0,
    ]


# Inert pad sentinel: order_id <= 0 and timestamp <= 0 => no fill, no time advance.
_PAD = [0, 0, 0, 0, -1, 0, -1, int(ExecRestriction.NON), int(Validity.GFS), 0,
        int(Status.ACTIVE), 0, 0]

_BUY = int(Side.BUY)
_SELL = int(Side.SELL)


def _pad_to(rows, n):
    rows = list(rows)
    while len(rows) < n:
        rows.append(list(_PAD))
    return rows


def _tapes(rows, n_padded=None):
    """Wrap a single row-list into a (1, N, 13) int32 tape set."""
    if n_padded is None:
        n_padded = len(rows)
    return np.asarray([_pad_to(rows, n_padded)], dtype=np.int32)


def _mk_env(rows, orders_per_step=2, n_padded=None, **cfg):
    """Build a Session over a single synthetic tape.

    ``tape_seconds_span`` is fixed so the timestamp->minutes mapping is
    deterministic and independent of the last row's ts, letting each test place
    GCT-crossing timestamps precisely.
    """
    base = {
        "tapes": _tapes(rows, n_padded),
        "orders_per_step": orders_per_step,
        "gct_minutes_from_got": 60,
        "tape_seconds_span": 3600.0,
    }
    base.update(cfg)
    return Session(base)


# The participant's orders. Each is the economic content the research
# Discrete(13) decode produced in these tests (touch-relative geometry,
# P3.2b-FOLLOWUP-1; every order NON / GFS):
#   CROSS bucket = opposite touch: a BUY priced at the resting best_ask, a SELL
#   priced at the resting best_bid of the post-background book.
#   qty buckets: the tests that pinned ``qty_buckets=(2.0, 3.0)`` used the small
#   bucket 2.0 MW; the default large bucket is ``qty_max / 2`` = 499.5 MW.
_QTY_LARGE_DEFAULT = _PC.qty_max / 2.0  # 499.5 MW


def _noop():
    """The empty order (the former NOOP action index 0)."""
    return AgentOrder.none()


def _buy(price, qty):
    return AgentOrder.create(side=_BUY, price=price, quantity=qty)


def _sell(price, qty):
    return AgentOrder.create(side=_SELL, price=price, quantity=qty)


def _order_variants():
    """The empty order + the 12 economic variants of the former Discrete(13)
    space on the default synthetic book (touches ~49 / ~51, mid ~50):
    {BUY, SELL} x {own-side touch, mid, opposite touch} x {qty_min, qty_max/2},
    all NON."""
    orders = [_noop()]
    for side, prices in ((_BUY, (49.0, 50.0, 51.0)), (_SELL, (51.0, 50.0, 49.0))):
        for price in prices:
            for qty in (_PC.qty_min, _QTY_LARGE_DEFAULT):
                orders.append(AgentOrder.create(side=side, price=price, quantity=qty))
    return orders


@contextlib.contextmanager
def _x64():
    """Outer scoped-x64 context — required so the interleave's int64
    extract_fills leg lowers (known P3.2a finding; the P3.3 harness supplies it).

    xdist hygiene (P3.2a-audit-flake fix): ``jax.experimental.enable_x64()`` flips
    the *process-global* ``jax_enable_x64`` config for the duration of the block.
    Under the canonical CPU xdist harness (``CUDA_VISIBLE_DEVICES=""``, ``-n 24``)
    a co-scheduled float32 test on the same worker must never observe x64=True,
    so this wrapper adds a belt-and-braces save/restore around the inner context:
    the pre-block value is captured and unconditionally reasserted in a
    ``finally`` (even if the body raises or the inner context's own restore is
    bypassed by a JAX/XLA edge case). Callers MUST keep ALL JAX work the test
    triggers — including trailing ``float(...)`` / ``jnp.asarray(...)`` /
    ``jax.block_until_ready(...)`` conversions — INSIDE the ``with _x64():`` block
    so no trace/compile escapes the x64 scope onto the shared worker.
    """
    _prev = jax.config.read("jax_enable_x64")
    try:
        with jax.experimental.enable_x64():
            yield
    finally:
        # Deterministically restore the process-global flag regardless of how the
        # inner context exited, so nothing leaks to a co-scheduled float32 test.
        if jax.config.read("jax_enable_x64") != _prev:
            jax.config.update("jax_enable_x64", _prev)


# ---------------------------------------------------------------------------
# Per-test in-memory-cache reset (P3.2a-audit-flake fix).
#
# The load-bearing hygiene fix for the CPU xdist flake. Each ``step_env`` /
# ``reset`` / vmap call in this file compiles under BOTH the float32 (reset,
# schema) and the scoped-x64 (interleave) regimes, so every test adds several
# distinct XLA executables to the worker process's IN-MEMORY compilation cache.
# Under ``-n 24`` a worker also runs many OTHER OMIE files whose lift compiles
# (``_replay_orders`` / ``_submit_order`` / ``step_env``) pile onto the SAME
# process; once the accumulated executable pool crosses XLA's single-process
# threshold the worker SEGFAULTS ("node down: Not properly terminated"),
# non-deterministically killing whatever test is mid-flight — including the
# co-scheduled float32 tests that were the observed "victims". (This is the same
# single-process XLA fragility that segfaults the file under ``-n0``.)
#
# ``jax.clear_caches()`` clears ONLY this process's in-memory tracing / lowering
# / compilation caches — it does NOT evict the shared class-level jit cache the
# way ``step_fn.clear_cache()`` did, and it does NOT touch the on-disk
# ``.jax_cache`` persistent cache (so warm-cache reuse across the suite is
# preserved). Resetting it after every test in this module keeps the per-worker
# in-memory executable pool BOUNDED, so the audit file no longer pushes any
# co-scheduled worker over the segfault threshold. It is a per-process,
# per-worker operation — inherently xdist-safe, mutating no cross-worker or
# cross-test shared state.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _bound_inmemory_jit_cache():
    """Reset THIS process's in-memory JAX compile caches after each test so the
    audit file's x64/vmap compiles cannot accumulate past XLA's single-process
    segfault threshold on a shared ``-n 24`` worker. Process-local; persistent
    ``.jax_cache`` untouched (warm reuse preserved)."""
    yield
    jax.clear_caches()




# ===========================================================================
# 1. Cash sign correctness (the load-bearing property).
# ===========================================================================
class TestRewardSignCorrectness:
    """The participant books POSITIVE realized cash for a SELL that fills and
    NEGATIVE realized cash for a BUY that fills (+1 SELL / −1 BUY convention,
    settlement.py:141-143).
    """

    def _seed_two_sided(self):
        # Resting SELL@50 (10MW) and BUY@49 (10MW): mid ~49.5 after step0.
        rows = [
            _row(_SELL, 50.0, 10.0, 101, 10),
            _row(_BUY, 49.0, 10.0, 102, 20),
        ]
        env = _mk_env(rows, orders_per_step=2)
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        # step0 (no order) folds the two resting orders into the book.
        with _x64():
            s0, _, _ = env.step_env(key, st, _noop())
        return env, key, s0

    def test_buy_that_fills_yields_negative_reward(self):
        env, key, s0 = self._seed_two_sided()
        with _x64():
            # 2 MW BUY at the best ask 50.0 (former action 5, BUY CROSS small).
            sb, rb, _ = env.step_env(key, s0, _buy(50.0, 2.0))
            # Convert INSIDE the x64 block so no trace/compile escapes the scope.
            r = float(rb.cash_delta_eur)
            inv = float(sb.agent_inventory[0])
        assert r < 0.0, f"BUY that fills must yield NEGATIVE cash flow, got {r}"
        # inventory moves +qty on a BUY.
        assert inv > 0.0, "BUY must increase signed inventory"

    def test_sell_that_fills_yields_positive_reward(self):
        env, key, s0 = self._seed_two_sided()
        with _x64():
            # 2 MW SELL at the best bid 49.0 (former action 11, SELL CROSS small).
            ss, rs, _ = env.step_env(key, s0, _sell(49.0, 2.0))
            r = float(rs.cash_delta_eur)
            inv = float(ss.agent_inventory[0])
        assert r > 0.0, f"SELL that fills must yield POSITIVE cash flow, got {r}"
        # inventory moves -qty on a SELL.
        assert inv < 0.0, "SELL must decrease signed inventory"

    def test_buy_and_sell_reward_signs_are_opposite(self):
        env, key, s0 = self._seed_two_sided()
        with _x64():
            _, rb, _ = env.step_env(key, s0, _buy(50.0, 2.0))
            _, rs, _ = env.step_env(key, s0, _sell(49.0, 2.0))
            rb_f, rs_f = float(rb.cash_delta_eur), float(rs.cash_delta_eur)
        assert rb_f < 0.0 < rs_f, (
            "BUY cash must be negative and SELL cash positive (opposite signs)"
        )

    def test_buy_reward_magnitude_equals_neg_Q_p_d(self):
        """Exact magnitude: cash = -Q·p·d for a 2MW BUY filling SELL@50.

        V = 2 MW · 50 EUR/MWh · 0.25 h = 25.0; BUY sign −1 => cash = −25.0.
        Catches both a flipped sign AND a wrong settlement scaling.
        """
        env, key, s0 = self._seed_two_sided()
        with _x64():
            _, rb, _ = env.step_env(key, s0, _buy(50.0, 2.0))
            rb_f = float(rb.cash_delta_eur)
        expected = -(2.0 * 50.0 * _PC.delivery_duration_hours)  # -25.0
        np.testing.assert_allclose(rb_f, expected, atol=1e-2, rtol=1e-4)

    def test_sell_reward_magnitude_equals_pos_Q_p_d(self):
        """Exact magnitude: cash = +Q·p·d for a 2MW SELL filling BUY@49.

        V = 2 MW · 49 EUR/MWh · 0.25 h = 24.5; SELL sign +1 => cash = +24.5.
        """
        env, key, s0 = self._seed_two_sided()
        with _x64():
            _, rs, _ = env.step_env(key, s0, _sell(49.0, 2.0))
            rs_f = float(rs.cash_delta_eur)
        expected = +(2.0 * 49.0 * _PC.delivery_duration_hours)  # +24.5
        np.testing.assert_allclose(rs_f, expected, atol=1e-2, rtol=1e-4)


# ===========================================================================
# 2. Fill-isolation — the whole-row diff wraps the AGENT call, not background.
# ===========================================================================
class TestFillIsolation:
    """When the BACKGROUND tape produces fills but the agent submits no order,
    the agent's cash delta must be exactly 0.0 — background fills MUST NOT leak
    into the agent's cash (the whole-row diff wraps the agent's _submit_order).
    """

    def test_background_trade_does_not_leak_into_noop_reward(self):
        # step0: resting SELL@50, then an aggressive IOC BUY@52 that crosses it
        # => a BACKGROUND trade occurs. The agent submits no order.
        rows = [
            _row(_SELL, 50.0, 10.0, 101, 10),
            _row(_BUY, 52.0, 10.0, 102, 20, exec_r=int(ExecRestriction.IOC)),
        ]
        env = _mk_env(rows, orders_per_step=2)
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        with _x64():
            s, r, _ = env.step_env(key, st, _noop())
            r_f = float(r.cash_delta_eur)
            inv_f = float(s.agent_inventory[0])
        # Background trade happened (some qty crossed) but the agent's cash is 0.
        assert r_f == 0.0, (
            "no-order cash delta must be EXACTLY 0.0 even when the background "
            f"tape trades (background fill leaked into agent cash: got {r_f})"
        )
        assert inv_f == 0.0, (
            "NOOP must leave agent inventory at 0 despite background fills"
        )

    def test_agent_reward_reflects_only_agent_qty_not_background_qty(self):
        """In a step where the background trades a LARGE qty and the agent fills
        a SMALL qty, the agent's cash magnitude must reflect ONLY the agent's
        own fill, not the sum of background + agent fills.
        """
        # Seed a two-sided book across two steps; in the agent step, the
        # background re-adds a large resting SELL and the agent takes a tiny bite.
        rows = [
            _row(_SELL, 50.0, 30.0, 101, 10),   # step0: large resting SELL
            _row(_BUY, 49.0, 30.0, 102, 20),    # step0: large resting BUY (mid ~49.5)
            _row(_SELL, 50.0, 20.0, 103, 30),   # step1 background: more resting SELL
            _row(_BUY, 48.0, 20.0, 104, 40, exec_r=int(ExecRestriction.IOC)),  # step1 bg trade
        ]
        env = _mk_env(rows, orders_per_step=2)
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        with _x64():
            s0, _, _ = env.step_env(key, st, _noop())                # build book
            s1, r1, _ = env.step_env(key, s0, _buy(50.0, 2.0))       # 2MW agent BUY
            r1_mag = abs(float(r1.cash_delta_eur))
        # Agent fill is 2 MW @ 50 => |cash| = 25.0. Background traded ~20 MW in
        # the same step; if that leaked, |cash| would be an order of magnitude
        # larger. Assert the magnitude matches ONLY the agent's 2 MW fill.
        expected_mag = 2.0 * 50.0 * _PC.delivery_duration_hours  # 25.0
        np.testing.assert_allclose(r1_mag, expected_mag, atol=1e-2, rtol=1e-3)


# ===========================================================================
# 3. Interleave ordering — background-then-agent.
# ===========================================================================
class TestInterleaveOrdering:
    """The agent acts against the POST-background book: it can fill against
    liquidity the background slice placed WITHIN the same step.
    """

    def test_agent_fills_against_liquidity_the_background_added_this_step(self):
        # step0: seed a two-sided book (mid ~50).
        # step1: the background places a FRESH resting SELL@49.5 (below mid), then
        # the agent's BUY at that best ask crosses it. If ordering were
        # agent-then-background, the SELL would not yet exist and the agent
        # could not fill.
        rows = [
            _row(_SELL, 51.0, 10.0, 101, 10),   # step0 resting
            _row(_BUY, 49.0, 10.0, 102, 20),    # step0 resting => mid ~50
            _row(_SELL, 49.5, 7.0, 103, 30),    # step1 background: fresh crossable SELL
        ]
        env = _mk_env(rows, orders_per_step=2, n_padded=4)
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        with _x64():
            s0, _, _ = env.step_env(key, st, _noop())                  # build book
            # BUY at the post-background best ask 49.5, large default qty bucket
            # (former action 6, BUY CROSS large) — crosses the fresh SELL.
            s1, r1, _ = env.step_env(key, s0, _buy(49.5, _QTY_LARGE_DEFAULT))
            inv1 = float(s1.agent_inventory[0])
            r1_f = float(r1.cash_delta_eur)
        # The agent bought the 7 MW the background placed THIS step.
        assert inv1 > 0.0, (
            "agent BUY must fill against the SELL the background added this step "
            "(proves background-then-agent ordering)"
        )
        np.testing.assert_allclose(inv1, 7.0, atol=1e-2, rtol=1e-3)
        assert r1_f < 0.0, "BUY fill against fresh background liquidity => negative cash"

    def test_agent_cannot_fill_against_liquidity_only_present_next_step(self):
        """Control: a crossable SELL that appears only in the NEXT step's slice
        must NOT be visible in the book this step (bounds the interleave to the
        current slice, not the whole tape).
        """
        rows = [
            _row(_SELL, 51.0, 10.0, 101, 10),   # step0 resting
            _row(_BUY, 49.0, 10.0, 102, 20),    # step0 resting => mid ~50
            _row(_SELL, 49.5, 7.0, 103, 30),    # step1 slice — NOT visible in step0
            _row(_BUY, 49.0, 5.0, 104, 40),
        ]
        env = _mk_env(rows, orders_per_step=2)
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        # The step1 SELL@49.5 is NOT in step0's slice, so it cannot be resting yet.
        with _x64():
            s0, _, _ = env.step_env(key, st, _noop())  # book empty at reset
            # First real step is step0's no-order step building the book; assert
            # the step1 SELL is not yet resting (visible q_ask reflects only
            # step0's SELL@51). top_of_book may trace/compile, so keep it inside
            # the x64 scope.
            q_ask0 = float(env.top_of_book(s0).q_ask)
        # q_ask should equal the step0 resting SELL volume (10 MW), not include the
        # step1 SELL (7 MW) — proving the slice window is bounded to this step.
        np.testing.assert_allclose(q_ask0, 10.0, atol=1e-2, rtol=1e-3)


# ===========================================================================
# 4. tape_cursor / episode-length arithmetic + dynamic_slice clamping.
# ===========================================================================
class TestCursorAndEpisodeArithmetic:

    def test_reset_initialises_cursor_and_time_to_origin(self):
        env = Session.from_test_config()
        key = jax.random.PRNGKey(3)
        st = env.reset(key)
        assert int(st.tape_cursor) == 0, "reset must set tape_cursor = 0"
        assert int(st.time) == 0, "reset must set time = 0 (GOT origin)"
        assert int(st.step_counter) == 0

    def test_cursor_advances_by_exactly_orders_per_step(self):
        env = Session.from_test_config()
        key = jax.random.PRNGKey(0)
        s = env.reset(key)
        prev = int(s.tape_cursor)
        for _ in range(3):
            with _x64():
                s, _, _ = env.step_env(key, s, _noop())
                cur = int(s.tape_cursor)
            assert cur - prev == env.orders_per_step, (
                f"tape_cursor must advance by exactly orders_per_step "
                f"({env.orders_per_step}); saw delta {cur - prev}"
            )
            prev = cur

    def test_steps_per_episode_is_ceil_divide(self):
        env = Session.from_test_config()
        expected = math.ceil(env.n_orders_padded / env.orders_per_step)
        assert env.steps_per_episode == expected, (
            "steps_per_episode must be ceil(N_orders_padded / orders_per_step)"
        )

    def test_steps_per_episode_non_divisible_padding(self):
        """A non-divisible (N, orders_per_step) pair must still ceil correctly
        (the final short step reads pad rows).
        """
        env = Session.from_test_config()
        N = env.n_orders_padded
        ops = 5 if (N % 5) != 0 else 3
        env2 = Session({"tapes": np.asarray(env.tapes), "orders_per_step": ops})
        assert env2.steps_per_episode == math.ceil(N / ops)


    def test_final_short_step_reads_clamped_pad_without_oob(self):
        """Stepping to AND past the derived episode length must not raise (the
        dynamic_slice clamps the start index; the final short step reads inert
        pad rows). Cursor keeps advancing by orders_per_step regardless.

        The Session ignores calls made after ``done`` (release plan §4.4), so
        the default synthetic tape set is replayed on a clock whose GCT is never
        reached (``tape_seconds_span`` far beyond the tapes' last timestamp):
        every one of the ``steps_per_episode + 2`` steps then runs the
        background slice, and the clamp regime is exercised exactly as before.
        """
        env = Session({"tape_seconds_span": 1.0e6})
        key = jax.random.PRNGKey(1)
        s = env.reset(key)
        n = env.steps_per_episode + 2  # deliberately step PAST the end
        for _ in range(n):
            with _x64():
                s, r, d = env.step_env(key, s, _noop())
                # no crash; cash finite; keep the isfinite trace inside x64.
                r_finite = bool(jnp.isfinite(jnp.asarray(r.cash_delta_eur)))
                done = bool(d)
            assert r_finite, "cash must stay finite past end"
            assert not done, "test precondition: GCT is never reached"
        # cursor advanced n times by orders_per_step (arithmetic, not clamped)
        assert int(s.tape_cursor) == n * env.orders_per_step


# ===========================================================================
# 5. done semantics at the GCT boundary.
# ===========================================================================
class TestDoneSemantics:

    def _episode_with_gct_at_last_step(self):
        # Timestamps: step0 ts small, step1 ts small, step2 ts == span => time==GCT.
        rows = [
            _row(_SELL, 51.0, 10.0, 101, 10),   # step0
            _row(_BUY, 49.0, 10.0, 102, 20),    # step0
            _row(_SELL, 51.0, 10.0, 103, 30),   # step1
            _row(_BUY, 49.0, 10.0, 104, 40),    # step1
            _row(_SELL, 51.0, 10.0, 105, 3600), # step2 ts == span => time crosses GCT
            _row(_BUY, 49.0, 10.0, 106, 3600),  # step2
        ]
        env = _mk_env(rows, orders_per_step=2)
        return env

    def test_done_fires_exactly_when_time_reaches_gct(self):
        env = self._episode_with_gct_at_last_step()
        key = jax.random.PRNGKey(0)
        s = env.reset(key)
        dones = []
        for step in range(env.steps_per_episode):
            with _x64():
                s, _, d = env.step_env(key, s, _noop())
                dones.append(bool(d))
        # Not done before the last step; done exactly at the step that crosses GCT.
        assert dones[:-1] == [False] * (len(dones) - 1), (
            f"done must be False before GCT is reached; saw {dones}"
        )
        assert dones[-1] is True, "done must be True on the step whose time >= GCT"
        assert int(s.time) >= env.gct_minutes_from_got

    def test_done_is_monotone_never_flips_back(self):
        env = self._episode_with_gct_at_last_step()
        key = jax.random.PRNGKey(0)
        s = env.reset(key)
        seen_true = False
        for _ in range(env.steps_per_episode + 3):  # keep stepping past done
            with _x64():
                s, _, d = env.step_env(key, s, _noop())
                cur = bool(d)
            if seen_true:
                assert cur is True, "done must be monotone — never flip back to False"
            seen_true = seen_true or cur
        assert seen_true, "done must fire within the episode"


# ===========================================================================
# 6. Anonymity (HARD RULE, market-spec §7/§9.2).
# ===========================================================================
class TestAnonymity:

    def test_q_ask_reflects_visible_peak_not_foreign_iceberg_hidden_qty(self):
        """A foreign iceberg's HIDDEN total qty must not move ``q_ask`` — the
        book reader reads visible LOB volume only (value-anonymity, P3.1 R2
        contract). A background iceberg SELL with peak=5, total=50 must show
        q_ask == 5.
        """
        rows = [
            _row(_SELL, 51.0, 50.0, 201, 10, exec_r=int(ExecRestriction.ICE),
                 peak=5.0),                      # iceberg: 5 MW visible, 50 MW hidden
            _row(_BUY, 49.0, 3.0, 202, 20),
        ]
        env = _mk_env(rows, orders_per_step=2)
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        with _x64():
            s, _, _ = env.step_env(key, st, _noop())
            q_ask = float(env.top_of_book(s).q_ask)
        assert q_ask == pytest.approx(5.0, abs=1e-2), (
            f"q_ask must reflect the VISIBLE iceberg peak (5 MW), not the hidden "
            f"total (50 MW); got {q_ask} — hidden qty leaked into the book reader"
        )
        # And it must NOT be the hidden total.
        assert q_ask < 50.0 - 1.0, "hidden iceberg total must never appear in q_ask"

    def test_q_bid_reflects_visible_resting_volume(self):
        rows = [
            _row(_SELL, 51.0, 4.0, 201, 10),
            _row(_BUY, 49.0, 6.0, 202, 20),   # 6 MW visible bid
        ]
        env = _mk_env(rows, orders_per_step=2)
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        with _x64():
            s, _, _ = env.step_env(key, st, _noop())
            tob = env.top_of_book(s)
            q_bid = float(tob.q_bid)
            q_ask = float(tob.q_ask)
        np.testing.assert_allclose(q_bid, 6.0, atol=1e-2, rtol=1e-3)
        np.testing.assert_allclose(q_ask, 4.0, atol=1e-2, rtol=1e-3)


# ===========================================================================
# 7. No-order structural invariance (stronger than cash == 0 alone).
# ===========================================================================
class TestNoopStructuralInvariance:

    def test_noop_book_equals_background_only_replay(self):
        """The empty order leaves the agent's OWN contribution to the book
        empty: the resulting book must be byte-identical to a background-only
        replay of the same slice (the agent's _submit_order is masked out).
        """
        rows = [_row(_SELL, 51.0, 10.0, 101, 10), _row(_BUY, 49.0, 10.0, 102, 20)]
        env = _mk_env(rows, orders_per_step=2)
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        with _x64():
            s_noop, _, _ = env.step_env(key, st, _noop())
            # Background-only replay of the same first slice, and the whole-book
            # comparison, both kept INSIDE the x64 scope so no trace escapes it.
            step_slice = jnp.asarray(env.tapes[st.tape_idx])[0:env.orders_per_step]
            book_bg, _ = _replay_orders(
                st.book, step_slice, env.product_config, env.market_config,
            )
            asks_equal = bool(jnp.array_equal(s_noop.book.lob.asks, book_bg.lob.asks))
            bids_equal = bool(jnp.array_equal(s_noop.book.lob.bids, book_bg.lob.bids))
        assert asks_equal, (
            "NOOP book asks must equal background-only replay (no agent order)"
        )
        assert bids_equal, (
            "NOOP book bids must equal background-only replay (no agent order)"
        )

    def test_noop_leaves_inventory_and_cash_unchanged(self):
        rows = [_row(_SELL, 50.0, 10.0, 101, 10),
                _row(_BUY, 52.0, 10.0, 102, 20, exec_r=int(ExecRestriction.IOC))]
        env = _mk_env(rows, orders_per_step=2)
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        with _x64():
            s, r, _ = env.step_env(key, st, _noop())
            inv_s, inv_st = float(s.agent_inventory[0]), float(st.agent_inventory[0])
            cash_s, cash_st = float(s.agent_cash[0]), float(st.agent_cash[0])
            r_f = float(r.cash_delta_eur)
        assert inv_s == inv_st == 0.0
        assert cash_s == cash_st == 0.0
        assert r_f == 0.0

    def test_noop_still_advances_time_cursor_and_step(self):
        """The empty order must still advance time / cursor / step_counter
        (only the agent's book contribution is suppressed).
        """
        env = self._env_with_time_advance()
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        with _x64():
            s, _, _ = env.step_env(key, st, _noop())
            cursor = int(s.tape_cursor)
            step_counter = int(s.step_counter)
            time_s, time_st = int(s.time), int(st.time)
        assert cursor == env.orders_per_step
        assert step_counter == 1
        assert time_s >= time_st  # monotone; advances if slice has ts>0

    @staticmethod
    def _env_with_time_advance():
        rows = [_row(_SELL, 51.0, 10.0, 101, 100), _row(_BUY, 49.0, 10.0, 102, 200)]
        return _mk_env(rows, orders_per_step=2)


# ===========================================================================
# 8. vmap one-compile invariant (independent check).
# ===========================================================================
class TestVmapOneCompile:
    """The jitted ``step_env`` compiles ONCE across the empty order and the 12
    order variants of the former action space (the order is traced, not
    static). Under ``vmap`` the step runs correctly only inside an OUTER
    ``enable_x64`` context (the interleave reaches a scoped-x64 int64 leg in
    ``extract_fills`` that does not lower under ``jax.vmap`` otherwise — a
    known P3.2a finding).
    """

    # Class-body ``pytestmark`` assignment (NOT a ``@pytest.mark.jit`` class
    # decorator) so ``test_jit_marker_lint.py`` recognizes the jit marker is in
    # scope for the explicit ``jax.jit(...)`` call in the one-compile test below
    # (the lint resolves module/class ``pytestmark = ...`` assignments and
    # per-function decorators, not class decorators).
    pytestmark = pytest.mark.jit

    def test_step_env_compiles_once_across_all_13_actions(self):
        env = Session.from_test_config()
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        # step_env is a jitted PjitFunction with a _cache_size probe.
        step_fn = type(env).step_env
        assert hasattr(step_fn, "_cache_size"), (
            "step_env must be a jitted function exposing _cache_size"
        )
        # xdist hygiene (P3.2a-audit-flake fix): DO NOT clear the shared,
        # process-global class-level ``step_env`` jit cache — under ``-n 24`` that
        # cache is reused by every co-scheduled test on the same worker, and
        # ``clear_cache()`` would evict their warm compiles mid-schedule (a
        # process-global mutation that churns the shared XLA process and crashed
        # co-scheduled float32 tests). Instead build a FRESH private ``jax.jit``
        # over the same underlying ``step_env`` body (``__wrapped__``, with ``env``
        # bound); its cache starts empty and is owned solely by this test, so the
        # one-compile assertion is measured without touching global state.
        private_step = jax.jit(functools.partial(step_fn.__wrapped__, env))
        assert private_step._cache_size() == 0, "fresh private jit must start empty"
        orders = _order_variants()
        assert len(orders) == 13
        for order in orders:
            with _x64():
                _, r, _ = private_step(key, st, order)
                jax.block_until_ready(r)
        assert private_step._cache_size() == 1, (
            "step_env must compile EXACTLY ONCE across all 13 orders "
            f"(the order is traced, not static); saw {private_step._cache_size()} "
            "entries"
        )

    def test_vmap_step_runs_across_all_actions_under_x64(self):
        """Stepping the vmapped session over NUM_ENVS across all 13 orders
        succeeds inside the outer enable_x64 context and yields finite cash.
        """
        env = Session.from_test_config()
        NUM_ENVS = 4
        keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
        reset_v = jax.vmap(env.reset)
        states = reset_v(keys)
        step_v = jax.vmap(env.step_env, in_axes=(0, 0, 0))
        for a, order in enumerate(_order_variants()):
            orders = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (NUM_ENVS,) + x.shape), order,
            )
            with _x64():
                _, r, _ = step_v(keys, states, orders)
                jax.block_until_ready(r)
                all_finite = bool(jnp.all(jnp.isfinite(jnp.asarray(r.cash_delta_eur))))
            assert all_finite, (
                f"vmapped step cash must be finite for order variant {a}"
            )

    def test_vmap_reset_produces_batched_leading_axis(self):
        env = Session.from_test_config()
        NUM_ENVS = 3
        keys = jax.random.split(jax.random.PRNGKey(7), NUM_ENVS)
        states = jax.vmap(env.reset)(keys)
        assert states.tape_cursor.shape == (NUM_ENVS,), (
            "vmap reset must batch tape_cursor over NUM_ENVS"
        )
        assert states.time.shape == (NUM_ENVS,)
        # cursor initialised to 0 in every env
        assert bool(jnp.all(states.tape_cursor == 0))


# ===========================================================================
# 9. Cross-cutting shape / spec conformance (kept minimal — not the focus).
# ===========================================================================
class TestSpecConformance:

    def test_reset_returns_state_with_tape_cursor_leaf(self):
        env = Session.from_test_config()
        key = jax.random.PRNGKey(0)
        st = env.reset(key)
        assert isinstance(st, OMIEContractState)
        assert hasattr(st, "tape_cursor"), "state must carry the additive tape_cursor leaf"
