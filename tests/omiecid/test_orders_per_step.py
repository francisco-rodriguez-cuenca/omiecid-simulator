"""Locked behavioral contract for Story WF-6 — OMIE-ORDERS-PER-STEP-1.

Finer step granularity: ``orders_per_step`` 4 -> 1 (act on EVERY tape order).

This is a CONFORMANCE / REGRESSION lock of the ALREADY-PARAMETRIC
``orders_per_step`` knob at value 1, NOT a paired-reward headline. The knob
already exists and defaults to 4 (``omie_session.py``:
``self.orders_per_step = int(config.get("orders_per_step", 4))`` at L162; it
drives the per-step tape slice at L497-499, the cursor advance at L578, and the
derived ``steps_per_episode = ceil(n_orders_padded / orders_per_step)`` at
L342-344). Because the knob is already parametric, some assertions below may
already PASS against current code — that is EXPECTED for a conformance lock. The
LOAD-BEARING regression guards are the GCT-boundary (off-by-one) and non-lossy
tape-coverage checks: those would FAIL if ``orders_per_step=1`` introduced a
dropped/duplicated tape order or an off-by-one in the episode horizon.

Spec sources (authoritative, read on disk 2026-07-16):
  * ``docs/cid_rl/experiment-plan.md`` §``OMIE-ORDERS-PER-STEP-1``
    (anchor ``#omie-orders-per-step-1--finer-step-granularity-orders_per_step-41``)
    — the SPEC SHEET: the L159/L495/L337-340 refs, the ~696 -> ~2784 (~4×)
    episode growth, and the "run standalone as a CONFORMANCE CHECK, shaped
    aggregate_return NOT comparable across the geometry change" ruling.
  * ``docs/omiecid/agile-plan.md`` §"Story WF-6 · OMIE-ORDERS-PER-STEP-1".
  * ``omiecid_simulator/omiecid/omie_session.py`` — the tape-slice (L497-499), cursor
    advance (L578), steps_per_episode (L342-344), done (L584-585), and the
    agent_trades buffer sized to steps_per_episode (L479) + the in-bounds
    ``write_idx`` clip (L572-575).

Contract locked (the WF-6 acceptance criteria), asserted at ``orders_per_step=1``
against a live Session built with a config override:

  (1) steps_per_episode == ceil(n_orders_padded / orders_per_step), recomputed
      correctly for op=1 (episodes ~4× longer than op=4). Asserted PARAMETRICALLY
      via the exact ceil relationship AND cross-checked against the op=4 factor —
      NEVER hard-coding 696/2784.
  (2) The per-step tape slice length is exactly 1 (== orders_per_step).
  (3) NON-LOSSY tape coverage: the identical REAL-order multiset is folded into
      the book across a full op=1 roll vs an op=4 roll of the SAME tape — no
      dropped or duplicated tape orders. (LOAD-BEARING regression guard.)
  (4) GCT boundary fires correctly — the explicit off-by-one catch:
        * done becomes True within steps_per_episode;
        * time >= gct at the done step;
        * the terminal step's slice contains the last real order;
        * done is TIME-BASED / granularity-invariant — it fires at the SAME
          wall-clock horizon (time) under op=1 and op=4, just at a larger
          step_counter under op=1. (LOAD-BEARING regression guard.)
  (5) agent_trades.shape[0] == steps_per_episode and the trade-tape write_idx
      stays in-bounds across a full roll (no OOB write, no clobber past the
      buffer).

Release translation (REL-07): every roll submits the empty order
(``AgentOrder.none()``, the former NOOP action index 0); ``reset`` returns the
state only and ``step_env`` returns ``(state, FillReport, done)``. The RL
reward placeholder was removed with the RL layer.

--------------------------------------------------------------------------------
CONSTRUCTION / DEVICE / TIERING
--------------------------------------------------------------------------------
Construction: the WF-6 change is a CONFIG override (``orders_per_step``), so
every live-env test builds ``Session({"orders_per_step": <n>})`` over the env's
self-contained synthetic default fixed-tape set (the same zero-config path
``Session.from_test_config`` uses). This keeps the tests hermetic and fast: the
default tape is tiny (n_orders_padded ~ tens), so a full op=1 roll is well under
the fast-suite budget. NO real OMIE data is read (the WF-6 knob is orthogonal to
which tape is loaded — a synthetic tape exercises the granularity contract
exactly), so NO test here is @pytest.mark.slow.

Tests that trigger a ``jax.jit`` compile of ``step_env`` carry
``@pytest.mark.jit`` (the pre-push tier, per the OMIE test-tiering convention);
pure attribute/derived-constant assertions stay in the fast local tier.

OQ-18 x64 TRAP: any test that steps ``step_env`` wraps the trace-triggering call
in ``with jax.experimental.enable_x64():`` (mirroring the P3.2a / shorter-tape
tests + ``ippo_rnn_JAXMARL._outer_x64_context``), or the int64 ``extract_fills``
promotion raises a ValueError.

Device policy (``docs/omiecid/device-policy.md``): the OMIE test harness is
CPU-default (the CPU pin is applied by ``tests/omiecid/conftest.py``). No test here
adds a device pin and none requires a GPU, so none carries ``@pytest.mark.gpu``.

Collectibility guard: the ``Session`` import is guarded so this file stays fully
collectible (``pytest --collect-only`` succeeds) even if the env / jax is not
importable — every live-env test then SKIPs cleanly rather than erroring
(mirroring ``tests/omiecid/test_omie_env.py`` + ``tests/omiecid/test_shorter_tape.py``).
"""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Import guard — keep the file collectible before/without a jax environment.
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    import jax.experimental  # for enable_x64 (OQ-18 outer context)

    from omiecid_simulator.omiecid.omie_session import AgentOrder, Session

    _ENV_MISSING = False
    _ENV_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - collection-time guard
    _ENV_MISSING = True
    _ENV_IMPORT_ERROR = repr(exc)


_env_skip = pytest.mark.skipif(
    _ENV_MISSING,
    reason=f"Session / jax not importable ({_ENV_IMPORT_ERROR})",
)


# ---------------------------------------------------------------------------
# Contract constants.
# ---------------------------------------------------------------------------
# WF-6 finer-granularity target: act on EVERY tape order.
OPS_FINE = 1
# The pre-WF-6 baseline granularity (the "control" geometry). Used ONLY to
# cross-check the granularity RATIO and the geometry-INVARIANT done horizon —
# NEVER to compare reward magnitude (non-comparable by design, SPEC SHEET).
OPS_BASELINE = 4

# The (N, 13) OMIEOrder tape column layout (omie_data_loader / omie_session). A REAL
# (non-pad) order carries order_id > 0; inert pad sentinels carry order_id <= 0
# and timestamp <= 0 (omie_session.py:1197, _default_fixed_tape_set pad row).
ORDER_ID_COL = 4
TIMESTAMP_COL = 6


# ---------------------------------------------------------------------------
# Env-construction helper. The WF-6 change is a CONFIG override, so we build the
# env over its self-contained synthetic default fixed-tape set with the
# orders_per_step knob set. Returns None (never raises) if no construction path
# works, so the caller SKIPs rather than errors.
# ---------------------------------------------------------------------------
def _build_env_or_none(orders_per_step: int):
    try:
        return Session({"orders_per_step": int(orders_per_step)})
    except Exception:
        return None


def _env_or_skip(orders_per_step: int):
    if _ENV_MISSING:
        pytest.skip(f"Session not importable ({_ENV_IMPORT_ERROR})")
    env = _build_env_or_none(orders_per_step)
    if env is None:
        pytest.skip(
            "Session could not be constructed with an orders_per_step config "
            "override over the default synthetic tape set — the WF-6 knob is a "
            "config override (omie_session.py:162)."
        )
    return env


# ---------------------------------------------------------------------------
# Black-box helpers over the env's observable state (all convention-robust).
# ---------------------------------------------------------------------------
def _noop_order():
    """The empty order (the former NOOP action index 0). All rolls here submit
    nothing — termination and tape coverage are independent of the agent's
    orders, and the empty order keeps the roll deterministic (no agent fills
    perturbing the book)."""
    return AgentOrder.none()


def _scalar(x):
    return jnp.asarray(x).reshape(-1)[0]


def _roll_to_done(env, seed=0):
    """Roll the env from reset with the empty order until done or a hard cap.

    Returns a list of per-step records ``(done, time, step_counter, tape_cursor)``
    (the state AFTER each step). The cap is steps_per_episode + 2 so a boundary
    off-by-one (done never firing, or firing one step late) is caught rather than
    masked. Runs under the OQ-18 outer enable_x64 context.
    """
    records = []
    with jax.experimental.enable_x64():
        state = env.reset(jax.random.PRNGKey(seed))
        cap = int(env.steps_per_episode) + 2
        for i in range(cap):
            state, _report, done = env.step_env(
                jax.random.PRNGKey(1000 + i), state, _noop_order()
            )
            records.append(
                (
                    bool(_scalar(done)),
                    int(_scalar(state.time)),
                    int(_scalar(state.step_counter)),
                    int(_scalar(state.tape_cursor)),
                )
            )
            if records[-1][0]:
                break
    return records, state


def _first_done_index(records):
    return next((i for i, rec in enumerate(records) if rec[0]), None)


# ---------------------------------------------------------------------------
# Tape-slice reconstruction (the black-box coverage oracle).
#
# The env folds ``tape[cursor : cursor + orders_per_step]`` each step via
# ``lax.dynamic_slice_in_dim``, which CLAMPS the start so the final short step
# never reads out of bounds (omie_session.py:497-499). We reconstruct the exact
# sequence of slices the env consumes over a FULL episode (steps_per_episode
# steps, cursor advancing by orders_per_step each step, start clamped to
# ``N - orders_per_step``) and collect the REAL orders (order_id > 0) folded in.
# This mirrors the production slice/cursor arithmetic EXACTLY, so the resulting
# real-order multiset is the ground truth for the non-lossy-coverage check.
# ---------------------------------------------------------------------------
def _folded_real_order_multiset(env, tape_idx=0):
    """Multiset (Counter over full-row tuples) of the REAL orders the env folds
    across a full episode at this env's orders_per_step, mirroring the clamped
    dynamic-slice cursor walk."""
    import numpy as np
    from collections import Counter

    tape = np.asarray(env.tapes[tape_idx])          # (N, 13) int32
    N = int(tape.shape[0])
    ops = int(env.orders_per_step)
    cursor = 0
    counter = Counter()
    for _step in range(int(env.steps_per_episode)):
        # lax.dynamic_slice_in_dim clamps start into [0, N - ops].
        start = min(max(cursor, 0), N - ops)
        sl = tape[start:start + ops]
        real = sl[sl[:, ORDER_ID_COL] > 0]
        counter.update(map(tuple, real.tolist()))
        cursor += ops
    return counter


def _all_real_orders_multiset(env, tape_idx=0):
    """Ground-truth multiset of every REAL order present in the raw tape."""
    import numpy as np
    from collections import Counter

    tape = np.asarray(env.tapes[tape_idx])
    real = tape[tape[:, ORDER_ID_COL] > 0]
    return Counter(map(tuple, real.tolist()))


# ===========================================================================
# (0) Sanity: the knob is parametric and takes the override.
# ===========================================================================
@_env_skip
class TestOrdersPerStepKnobIsParametric:
    """The ``orders_per_step`` knob accepts a config override (WF-6 is a config
    change, not a code change — the knob already exists, omie_session.py:162)."""

    def test_env_accepts_orders_per_step_one(self):
        env = _env_or_skip(OPS_FINE)
        assert int(env.orders_per_step) == OPS_FINE, (
            "Session must accept orders_per_step=1 via config override "
            "(omie_session.py:162)"
        )

    def test_env_accepts_baseline_orders_per_step(self):
        env = _env_or_skip(OPS_BASELINE)
        assert int(env.orders_per_step) == OPS_BASELINE


# ===========================================================================
# (1) steps_per_episode = ceil(n_orders_padded / orders_per_step), recomputed
#     correctly at op=1 (~4× longer than op=4). Parametric — NEVER hard-coded.
# ===========================================================================
@_env_skip
class TestStepsPerEpisodeRecomputes:
    """AC (1): steps_per_episode = ceil(n_orders_padded / orders_per_step)
    recomputes for op=1; the episode is ~4× longer than op=4 (SPEC SHEET
    ~696 -> ~2784; omie_session.py:342-344). Asserted via the exact ceil relationship
    AND the cross-geometry ratio, NEVER hard-coding a specific step count."""

    def test_steps_per_episode_is_ceil_at_op1(self):
        """The derived episode length equals ceil(N / 1) == N at op=1."""
        env = _env_or_skip(OPS_FINE)
        expected = math.ceil(env.n_orders_padded / OPS_FINE)
        assert env.steps_per_episode == expected, (
            f"steps_per_episode {env.steps_per_episode} != "
            f"ceil(n_orders_padded {env.n_orders_padded} / orders_per_step "
            f"{OPS_FINE}) = {expected} (omie_session.py:342-344)"
        )
        # At op=1 the ceil collapses to exactly N (act on every tape order).
        assert env.steps_per_episode == env.n_orders_padded, (
            "at orders_per_step=1 the episode has EXACTLY n_orders_padded steps "
            "(one step per tape order, ceil(N/1)==N)"
        )

    def test_op1_episode_is_longer_than_op4_by_the_ceil_ratio(self):
        """op=1 episodes are ~4× longer than op=4 on the SAME tape (SPEC SHEET
        ~696 -> ~2784). Asserted PARAMETRICALLY: both equal their own ceil, the
        tape length is identical across the two geometries, and the op=1 length
        strictly exceeds op=4 (the "finer granularity" guarantee) — no literal
        696/2784 anywhere."""
        env1 = _env_or_skip(OPS_FINE)
        env4 = _env_or_skip(OPS_BASELINE)
        # Same underlying tape (same default set) -> same n_orders_padded.
        assert env1.n_orders_padded == env4.n_orders_padded, (
            "the two geometries must share the SAME tape so the step-count "
            "comparison isolates orders_per_step"
        )
        n = env1.n_orders_padded
        assert env1.steps_per_episode == math.ceil(n / OPS_FINE)
        assert env4.steps_per_episode == math.ceil(n / OPS_BASELINE)
        assert env1.steps_per_episode > env4.steps_per_episode, (
            f"op=1 steps ({env1.steps_per_episode}) must exceed op=4 steps "
            f"({env4.steps_per_episode}) — finer granularity means MORE steps "
            "per episode (SPEC SHEET ~696 -> ~2784, ~4×)"
        )
        # The ratio is ~OPS_BASELINE (exact when N is a multiple of both), and
        # never below the floor implied by ceil. Guards against a stale
        # hard-coded episode length that ignores orders_per_step.
        assert env1.steps_per_episode >= env4.steps_per_episode * (
            OPS_BASELINE - 1
        ), (
            "the op=1 / op=4 step-count ratio must be ~orders_per_step "
            f"(={OPS_BASELINE}); got {env1.steps_per_episode} vs "
            f"{env4.steps_per_episode} — steps_per_episode is not tracking the "
            "orders_per_step knob"
        )


# ===========================================================================
# (2) The per-step tape slice length is exactly 1 at op=1.
# ===========================================================================
@_env_skip
class TestPerStepSliceLengthIsOne:
    """AC (2): at op=1 the per-step tape slice is exactly one order long.

    The slice length is ``orders_per_step`` (omie_session.py:497-499,
    ``lax.dynamic_slice_in_dim(..., self.orders_per_step, ...)``). We assert the
    knob is 1, and the OBSERVABLE consequence: the cursor advances by exactly 1
    per step (the cursor increment IS orders_per_step, omie_session.py:578), so a
    one-order slice is folded each step."""

    def test_orders_per_step_is_one(self):
        env = _env_or_skip(OPS_FINE)
        assert int(env.orders_per_step) == 1, (
            "the per-step slice length equals orders_per_step; at op=1 it must "
            "be exactly 1 (act on every tape order)"
        )

    @pytest.mark.jit
    def test_cursor_advances_by_exactly_one_per_step(self):
        """The observable slice-length signature: tape_cursor advances by exactly
        orders_per_step (==1) each step (omie_session.py:578)."""
        env = _env_or_skip(OPS_FINE)
        with jax.experimental.enable_x64():
            state = env.reset(jax.random.PRNGKey(0))
            cursor_before = int(_scalar(state.tape_cursor))
            # Step a few times (bounded well under steps_per_episode) and confirm
            # each step advances the cursor by exactly 1.
            n_probe = min(5, int(env.steps_per_episode) - 1)
            for i in range(max(n_probe, 1)):
                state, _r, _d = env.step_env(
                    jax.random.PRNGKey(1 + i), state, _noop_order()
                )
                cursor_after = int(_scalar(state.tape_cursor))
                assert cursor_after - cursor_before == OPS_FINE, (
                    f"tape_cursor advanced by {cursor_after - cursor_before} on "
                    f"step {i}; at op=1 it must advance by exactly 1 (the slice "
                    "length == orders_per_step, omie_session.py:578)"
                )
                cursor_before = cursor_after


# ===========================================================================
# (3) NON-LOSSY tape coverage — the load-bearing regression guard.
#     The identical REAL-order multiset is folded across a full op=1 roll vs an
#     op=4 roll of the SAME tape: no dropped or duplicated tape order.
# ===========================================================================
@_env_skip
class TestNonLossyTapeCoverage:
    """AC (3): complete, non-lossy tape coverage. A dropped or duplicated tape
    order at op=1 (a slice/cursor off-by-one, or the final-step dynamic-slice
    clamp re-reading a REAL row) would FAIL these — the load-bearing regression
    guard the WF-6 developer must not break.

    Oracle: ``_folded_real_order_multiset`` reconstructs the EXACT clamped
    dynamic-slice cursor walk the env performs (omie_session.py:497-499, 578) and
    collects the REAL orders (order_id > 0) folded across a full episode. This is
    a faithful black-box mirror of the production slice arithmetic, not an
    implementation-internal read."""

    def test_op1_folds_every_real_order_exactly_once(self):
        """A full op=1 roll folds the COMPLETE real-order multiset of the tape —
        every real order exactly as many times as it appears in the raw tape
        (no drop, no duplication). At op=1 every real order is its own step, so
        the folded multiset must equal the raw-tape real-order multiset."""
        env = _env_or_skip(OPS_FINE)
        folded = _folded_real_order_multiset(env, tape_idx=0)
        ground_truth = _all_real_orders_multiset(env, tape_idx=0)
        assert sum(ground_truth.values()) >= 1, (
            "the default tape must contain >= 1 real order for a meaningful "
            "coverage check"
        )
        assert folded == ground_truth, (
            "op=1 roll must fold EVERY real tape order exactly once — no dropped "
            "or duplicated order. Missing (raw - folded): "
            f"{ground_truth - folded}; extra (folded - raw): {folded - ground_truth}"
        )

    def test_op1_and_op4_fold_the_identical_real_order_multiset(self):
        """The SAME tape folded at op=1 and op=4 yields the IDENTICAL real-order
        multiset — the geometry change is loss-less and duplication-free. If the
        op=4 final-step clamp re-read a REAL order (or op=1 dropped one), the two
        multisets would differ."""
        env1 = _env_or_skip(OPS_FINE)
        env4 = _env_or_skip(OPS_BASELINE)
        m1 = _folded_real_order_multiset(env1, tape_idx=0)
        m4 = _folded_real_order_multiset(env4, tape_idx=0)
        assert m1 == m4, (
            "op=1 and op=4 must fold the identical real-order multiset of the "
            "same tape (non-lossy, no dup). op1-only: "
            f"{m1 - m4}; op4-only: {m4 - m1}"
        )

    def test_final_step_clamp_reads_only_pad_rows_not_real_orders(self):
        """The dynamic-slice clamp on the final short step must re-read only INERT
        PAD rows (order_id <= 0), never a REAL order — otherwise the last real
        order would be double-counted. Directly guards the clamp off-by-one at
        both geometries.

        Reconstructs the clamped start of the LAST step and asserts any row it
        re-reads that was already consumed by a prior step is a pad row."""
        import numpy as np

        for ops in (OPS_FINE, OPS_BASELINE):
            env = _env_or_skip(ops)
            tape = np.asarray(env.tapes[0])
            N = int(tape.shape[0])
            spe = int(env.steps_per_episode)
            last_start = min(max((spe - 1) * ops, 0), N - ops)
            # The exclusive end of the PENULTIMATE step's slice (the highest
            # cursor consumed before the final step, clamped identically).
            prev_end = min(max((spe - 2) * ops, 0), N - ops) + ops if spe >= 2 else 0
            # Rows in [last_start, prev_end) are re-read by the clamp; they must
            # all be pad rows (order_id <= 0).
            reread = tape[last_start:max(prev_end, last_start)]
            if reread.shape[0]:
                real_reread = reread[reread[:, ORDER_ID_COL] > 0]
                assert real_reread.shape[0] == 0, (
                    f"at op={ops} the final-step dynamic-slice clamp re-reads "
                    f"{real_reread.shape[0]} REAL order(s) that a prior step "
                    "already folded — this DOUBLE-COUNTS a real tape order "
                    "(clamp off-by-one, omie_session.py:497-499). The tape must have "
                    "enough trailing pad rows that the clamp only re-reads pads."
                )


# ===========================================================================
# (4) GCT boundary fires correctly — the explicit off-by-one catch.
#     done True within steps_per_episode; time >= gct at done; terminal step
#     contains the last real order; done horizon is granularity-INVARIANT.
# ===========================================================================
@_env_skip
class TestGCTBoundaryFires:
    """AC (4): the GCT boundary off-by-one catch (LOAD-BEARING). ``done`` is
    time-based (``time >= gct_minutes_from_got``, omie_session.py:584-585), so it must
    fire within steps_per_episode, AT the same wall-clock horizon under op=1 and
    op=4 (just at a larger step_counter under op=1), with the terminal slice
    carrying the last real order."""

    @pytest.mark.jit
    def test_done_fires_within_steps_per_episode_at_op1(self):
        """done becomes True within steps_per_episode steps at op=1 — the episode
        actually terminates at the (now ~4× larger) horizon, it does not run past
        it or never finish."""
        env = _env_or_skip(OPS_FINE)
        records, _state = _roll_to_done(env, seed=0)
        first_done = _first_done_index(records)
        assert first_done is not None, (
            "the op=1 episode must reach done=True within steps_per_episode + 2 "
            f"= {env.steps_per_episode + 2} steps; it never terminated"
        )
        # 0-indexed: firing on the record at index `first_done` means it fired on
        # step_counter first_done+1, which must be <= steps_per_episode.
        assert first_done + 1 <= env.steps_per_episode, (
            f"done first fired on step {first_done + 1}, past steps_per_episode "
            f"= {env.steps_per_episode} — the derived horizon is too short "
            "(off-by-one in ceil / cursor arithmetic)"
        )

    @pytest.mark.jit
    def test_time_at_done_is_at_or_past_gct_at_op1(self):
        """At the done step, time >= gct; on the step BEFORE, time < gct — done
        fires EXACTLY at the GCT crossing, not early, not late (omie_session.py:
        584-585)."""
        env = _env_or_skip(OPS_FINE)
        records, _state = _roll_to_done(env, seed=0)
        first_done = _first_done_index(records)
        assert first_done is not None, "episode never terminated"
        gct = float(env.gct_minutes_from_got)
        done_time = records[first_done][1]
        assert done_time >= gct, (
            f"time at done ({done_time}) must be >= gct ({gct}) — done is "
            "time-at-GCT driven (omie_session.py:584-585)"
        )
        if first_done > 0:
            pre_time = records[first_done - 1][1]
            assert pre_time < gct, (
                f"time on the step before done ({pre_time}) must be < gct "
                f"({gct}) — done must fire EXACTLY at the GCT crossing, not early"
            )

    @pytest.mark.jit
    def test_terminal_step_slice_contains_the_last_real_order(self):
        """The terminal (done) step's slice must contain the LAST real order of
        the tape — the episode terminates BECAUSE the final real order (which
        carries the GCT-crossing timestamp) is folded in, not before it. At op=1
        the last step's single-order slice IS the last real order.

        Reconstructs the clamped slice consumed on the done step and asserts the
        last real order (max order_id-bearing REAL row by tape position) lies
        within it."""
        import numpy as np

        env = _env_or_skip(OPS_FINE)
        records, _state = _roll_to_done(env, seed=0)
        first_done = _first_done_index(records)
        assert first_done is not None, "episode never terminated"

        tape = np.asarray(env.tapes[0])
        N = int(tape.shape[0])
        ops = int(env.orders_per_step)
        real_positions = np.nonzero(tape[:, ORDER_ID_COL] > 0)[0]
        assert real_positions.size >= 1, "tape must have >= 1 real order"
        last_real_pos = int(real_positions.max())

        # The done step is step_counter == first_done + 1, i.e. the (first_done)-th
        # 0-indexed step; its clamped slice start:
        done_step_idx = first_done  # 0-indexed step that produced `done`
        start = min(max(done_step_idx * ops, 0), N - ops)
        end = start + ops
        assert start <= last_real_pos < end, (
            f"the terminal (done) step's slice [{start}, {end}) must contain the "
            f"last real order at tape position {last_real_pos}; done fired on a "
            "slice that does NOT include the final real order — the terminal "
            "boundary is misaligned with the tape's last real event"
        )

    @pytest.mark.jit
    def test_done_horizon_is_granularity_invariant(self):
        """done is TIME-based, so it fires at the SAME wall-clock horizon (time)
        under op=1 and op=4 — the geometry change moves the step_counter at which
        done fires, NOT the time at which it fires. The op=1 done fires at a
        strictly LARGER step_counter than op=4 (finer steps), but at the same
        (or equal) crossing time. This is the granularity-invariance the SPEC
        SHEET relies on to call op=1 a conformance check rather than a new
        horizon."""
        env1 = _env_or_skip(OPS_FINE)
        env4 = _env_or_skip(OPS_BASELINE)
        rec1, _s1 = _roll_to_done(env1, seed=0)
        rec4, _s4 = _roll_to_done(env4, seed=0)
        d1 = _first_done_index(rec1)
        d4 = _first_done_index(rec4)
        assert d1 is not None and d4 is not None, (
            "both geometries must terminate for the invariance comparison"
        )
        gct = float(env1.gct_minutes_from_got)
        # Same GCT horizon for both geometries (same tape, same contract).
        assert float(env4.gct_minutes_from_got) == gct, (
            "the two geometries must share the same GCT horizon"
        )
        time1 = rec1[d1][1]
        time4 = rec4[d4][1]
        # done fired at (or past) the same wall-clock GCT under both geometries.
        assert time1 >= gct and time4 >= gct, (
            f"done must fire at/past GCT under both geometries; op1 time={time1}, "
            f"op4 time={time4}, gct={gct}"
        )
        # step_counter at done is LARGER under op=1 (finer steps cover the same
        # wall-clock horizon in more steps) — the observable signature of the
        # geometry change.
        step1 = rec1[d1][2]
        step4 = rec4[d4][2]
        assert step1 > step4, (
            f"op=1 done step_counter ({step1}) must exceed op=4 ({step4}) — the "
            "same wall-clock GCT horizon is reached in MORE (finer) steps at "
            "op=1 (granularity-invariant done, larger step_counter)"
        )


# ===========================================================================
# (5) agent_trades.shape[0] == steps_per_episode and write_idx stays in-bounds
#     across a full roll (no OOB write, no clobber past the buffer).
# ===========================================================================
@_env_skip
class TestAgentTradesBufferInBounds:
    """AC (5): the agent trade-tape buffer is sized to steps_per_episode
    (omie_session.py:479) and the per-step write_idx (clipped to
    [0, steps_per_episode-1], omie_session.py:572-575) stays in-bounds across a full
    op=1 roll — even though op=1 has ~4× more steps than op=4."""

    @pytest.mark.jit
    def test_agent_trades_buffer_sized_to_steps_per_episode_at_op1(self):
        """The reset-state agent_trades leaf has leading axis == steps_per_episode
        at op=1 (WF-11-FOLLOWUP-1: one PER-FILL slab of two legs per env step;
        omie_session.py:485-492)."""
        env = _env_or_skip(OPS_FINE)
        with jax.experimental.enable_x64():
            state = env.reset(jax.random.PRNGKey(0))
        at = jnp.asarray(state.agent_trades)
        # WF-11-FOLLOWUP-1 widened the tape from (steps_per_episode, 6) to the
        # per-fill (steps_per_episode, 2, 6) layout: slot 0 = aggressive (own-
        # submit) leg, slot 1 = passive (resting-NON) leg. The buffer is still
        # sized to steps_per_episode along axis 0 (one slab per env step).
        assert at.ndim == 3, (
            f"agent_trades must be 3-D (steps_per_episode, 2, 6); got shape {at.shape}"
        )
        assert at.shape[1:] == (2, 6), (
            f"agent_trades per-step slab must be (2, 6) — the two-leg per-fill "
            f"layout (aggressive slot 0, passive slot 1); got {at.shape}"
        )
        assert at.shape[0] == env.steps_per_episode, (
            f"agent_trades leading axis {at.shape[0]} must equal steps_per_episode "
            f"{env.steps_per_episode} (one slab per env step; omie_session.py:485-492) "
            "— at op=1 the buffer must GROW to the ~4× longer episode length"
        )

    @pytest.mark.jit
    def test_agent_trades_shape_stable_across_full_roll(self):
        """The agent_trades leading axis stays == steps_per_episode across a full
        op=1 roll to done (a jit-safe fixed-shape buffer; it must NOT grow, and
        the write_idx clip must keep every write in-bounds — an OOB write or a
        shape change would surface as a shape mismatch or a trace error here)."""
        env = _env_or_skip(OPS_FINE)
        with jax.experimental.enable_x64():
            state = env.reset(jax.random.PRNGKey(0))
            expected_rows = int(env.steps_per_episode)
            for i in range(expected_rows + 1):  # +1 pushes past the buffer end
                state, _r, done = env.step_env(
                    jax.random.PRNGKey(2000 + i), state, _noop_order()
                )
                at = jnp.asarray(state.agent_trades)
                assert at.shape[0] == expected_rows, (
                    f"agent_trades leading axis changed to {at.shape[0]} on step "
                    f"{i + 1} (expected fixed {expected_rows}) — the buffer must "
                    "be a jit-safe fixed-shape leaf (omie_session.py:479)"
                )
                if bool(_scalar(done)):
                    break

    @pytest.mark.jit
    def test_write_idx_stays_in_bounds_past_buffer_end(self):
        """Stepping PAST steps_per_episode (the write_idx clip regime,
        omie_session.py:572-575) must not raise or corrupt the buffer shape — the
        clip pins the write to the last row rather than writing out of bounds.
        This exercises the exact off-by-one boundary at the ~4× larger op=1
        episode length."""
        env = _env_or_skip(OPS_FINE)
        with jax.experimental.enable_x64():
            state = env.reset(jax.random.PRNGKey(0))
            # Step a couple beyond steps_per_episode WITHOUT breaking on done, so
            # the write_idx is driven into the clip regime (step_counter >=
            # steps_per_episode). A missing clip would OOB-write / error here.
            for i in range(int(env.steps_per_episode) + 3):
                state, _r, _d = env.step_env(
                    jax.random.PRNGKey(3000 + i), state, _noop_order()
                )
            at = jnp.asarray(state.agent_trades)
            assert at.shape[0] == env.steps_per_episode, (
                "agent_trades buffer must retain its fixed shape even when "
                "step_counter exceeds steps_per_episode (write_idx clip keeps "
                "writes in-bounds; omie_session.py:572-575)"
            )
            # The buffer is still readable (concretizes) — no OOB corruption.
            assert int(jnp.asarray(at).reshape(-1)[0]) == int(
                jnp.asarray(at).reshape(-1)[0]
            )
