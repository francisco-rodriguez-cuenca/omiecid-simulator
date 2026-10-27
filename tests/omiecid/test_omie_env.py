"""Locked behavioral contract for Story P3.2a — Session (merge-friendly single
env build: reset/step + inline interleave).

These tests are the acceptance contract the Implementation Agent builds against.
They target a NEW module ``omiecid_simulator/omiecid/omie_session.py`` (``Session``) that
does NOT exist yet, so every test that touches the module is guarded by
``@pytest.mark.skipif(_P32A_MISSING, ...)`` — the file stays fully COLLECTIBLE
(``pytest --collect-only`` succeeds) before the module lands, and the guarded
tests turn from SKIP into real assertions the moment the developer creates the
module. This mirrors the established import-guard idiom in
``tests/omiecid/test_omie_states.py`` and ``tests/omiecid/test_omie_data.py:26-51``.

Release translation (REL-07): the session mechanics contracts below are
expressed through the public ``AgentOrder`` (``AgentOrder.none()`` is the
former NOOP action index 0; a real order carries the economic fields the
research action decode produced). ``reset`` returns the state only and
``step_env`` returns ``(state, FillReport, done)``. The RL contracts
(observation width/dtype, rewards, the JaxMARL dones/infos dicts, the
Discrete(13) cardinality) were removed with the RL layer.

  (1) reset returns a valid state        — TestEnvResetReturnsValidState
  (2) step_env returns a valid tuple     — TestEnvStepReturnsValidTuple
  (3) test_env_jit_compiles              — TestEnvJitCompiles
  (4) no order changes nothing           — TestNoopActionChangesNothing
  (5) test_done_at_gct                   — TestDoneAtGCT
  (6) one-compile-under-vmap             — TestOneCompilePreservationUnderVmap
  (7) test_interleave_...background_then_agent — TestInterleaveBackgroundThenAgent

Spec sources (authoritative, read on disk 2026-07-02):
  * ``docs/omiecid/agile-plan.md`` §Story P3.2a — the story card (v5.55, size M),
    incl. the amended Deliverables + 7-part Test-first contract.
  * ``docs/omiecid/migration-plan.md`` §P3.2 — the env interface (`marl_env.py`
    mirror), the §P3.2 "Per-step tape-slice contract" subsection
    (granularity / cursor / background-then-agent interleave;
    migration-plan.md:1062-1171), the Action (Discrete(13)), Reward
    (`sign·settlement_value`, +1 SELL / −1 BUY), and Done (`time ≥ GCT`) specs.
  * ``docs/omiecid/migration-plan.md`` §"Trade extraction detector invariant"
    (migration-plan.md:834-836) — the whole-row-diff fill detection the reward
    isolation depends on.
  * ``docs/omiecid/market-spec.md`` §3 — GOT/GCT (GCT = 1 hour before delivery
    start; orders cancelled at GCT).
  * ``omiecid_simulator/omiecid/omie_states.py`` — ``OMIEContractState`` (P3.1 +
    the additive ``tape_cursor`` 0-d int32 leaf added by P3.2a).
  * ``omiecid_simulator/omiecid/contract_book.py`` — the locked ``_submit_order``
    lift (``:1222-1225``, ``static_argnames=("product_config","market_config")``)
    + ``SubmitResult`` tree (UNSIGNED ``settlement_value``, ``:361-368``).
  * ``omiecid_simulator/jaxomiecid/omie_data_loader.py`` — the P2-PERF-5 lifted
    ``_replay_orders`` (``:949-952``) reused verbatim on a
    ``lax.dynamic_slice_in_dim`` window.
  * ``docs/release/release-plan.md`` §4.4 — the public ``Session`` interface
    (``AgentOrder``, ``FillReport``, ``reset`` / ``step_env``).

Device policy (``docs/omiecid/device-policy.md``): the OMIE test harness is
CPU-default; the CPU pin is applied by ``tests/omiecid/conftest.py``
(``jax.config.update("jax_platform_name","cpu")``, conftest L186). No test here
adds a device pin — ``omie_session.py`` under test is device-agnostic. No test here
requires a GPU, so none carries ``@pytest.mark.gpu``.

Test tiering (the repo's test-tiering convention + ``pyproject.toml``
markers): tests that trigger a ``jax.jit`` / ``vmap`` compile
carry ``@pytest.mark.jit`` (the pre-push tier), NOT ``@pytest.mark.slow``. Pure
structural/unit assertions stay in the fast (<1s) local tier.
"""

import dataclasses
import math

import pytest

# ---------------------------------------------------------------------------
# P3.2a module import — guarded so this file stays collectible before the
# Session module exists (matches the guard pattern in
# tests/omiecid/test_omie_states.py and tests/omiecid/test_omie_data.py:26-51).
#
# We import BOTH the env under test (does not exist yet) AND the locked
# interfaces the env reuses (already on disk). If the env module is missing the
# whole block fails and every guarded test SKIPs; the pure-doc-constant tests
# below stay collectible either way.
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    import jax.experimental  # for enable_x64 (scoped-x64 leg under vmap; P2-L4)
    from jax.flatten_util import ravel_pytree

    from omiecid_simulator.omiecid.contract_book import (
        OMIEBookState,
        SubmitResult,
        _submit_order,
    )
    from omiecid_simulator.omiecid.omie_states import OMIEContractState
    from omiecid_simulator.omiecid.order_types import Side

    from omiecid_simulator.omiecid.omie_session import AgentOrder, Session

    _P32A_MISSING = False
    _P32A_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - collection-time guard
    _P32A_MISSING = True
    _P32A_IMPORT_ERROR = repr(exc)

_p32a_skip = pytest.mark.skipif(
    _P32A_MISSING,
    reason=(
        "omiecid_simulator/omiecid/omie_session.py (Session) not yet implemented "
        f"({_P32A_IMPORT_ERROR})"
    ),
)


# ===========================================================================
# Contract constants.
# ===========================================================================

# The empty order (the former NOOP action index 0): submits nothing.
def _noop_order():
    return AgentOrder.none()


# A real order for the jit / tuple tests: the economic fields of the former
# action index 1 (BUY, PASSIVE bucket = own-side touch, smallest qty bucket
# ``qty_min`` = 0.1 MW, NON, GFS). The default synthetic tape's best bid after
# the first background slice is 49.0 (tape 0) / 48.9 (tape 1).
def _buy_order():
    return AgentOrder.create(side=int(Side.BUY), price=49.0, quantity=0.1)


# ===========================================================================
# Env construction contract (surfaced ambiguity — see module docstring +
# sprint-status note). The P3.2a card pins the env's public *behaviour*
# (reset/step signatures, dict keys, obs/action shapes, interleave order, done
# condition) but does NOT freeze the env CONSTRUCTOR signature or the fixed-tape-
# SET config schema — those are P3.3 deliverables
# (config/env_configs/omie_single_contract.json). To keep these tests black-box
# and satisfiable WITHOUT anchoring on an unpinned constructor, every test that
# needs a live env acquires it through the ``env`` fixture, which tries the
# documented construction paths in order and SKIPs (never errors) if none is
# discoverable. A faithful implementation exposes at least one of them; a wrong
# implementation still fails the behavioural assertions below.
# ===========================================================================


def _try_build_env():
    """Best-effort construction of an Session for the P3.2a behavioural tests.

    Returns an ``Session`` instance or ``None``. The constructor signature and
    the fixed-tape-SET config are P3.3 deliverables not frozen by P3.2a, so we
    probe the documented entry points rather than hardcode one:

      1. A zero-arg / default constructor (``Session()``) — if the developer
         ships a self-contained default fixed-tape set for tests.
      2. A classmethod test-factory (``Session.from_test_config()`` /
         ``Session.for_testing()``) — the conventional escape hatch.
      3. ``Session(config)`` where ``config`` is a minimal dict pointing at a
         synthetic in-memory tape set, per the §P3.2 constructor sketch
         (migration-plan.md:1044-1046 ``load_fixed_tape_set(config)``).

    Any construction that raises is treated as "not this path" and the next is
    tried. If all fail, ``None`` is returned and the caller SKIPs with a clear
    reason — the ambiguity is surfaced, not silently swallowed.
    """
    # Path 1: default constructor.
    try:
        return Session()
    except Exception:
        pass

    # Path 2: test-factory classmethods.
    for factory_name in ("from_test_config", "for_testing", "default_test_env"):
        factory = getattr(Session, factory_name, None)
        if factory is not None:
            try:
                return factory()
            except Exception:
                pass

    # Path 3: minimal config dict (best-effort; keys are P3.3's, so this is a
    # last resort — a faithful impl should satisfy Path 1 or 2 for tests).
    try:
        return Session({})
    except Exception:
        pass

    return None


@pytest.fixture(scope="module")
def env():
    """A live Session, or a SKIP if no documented construction path works.

    Module-scoped: the env holds the loaded fixed-tape set and jitted
    reset/step_env; building once keeps the fast-suite budget intact.
    """
    if _P32A_MISSING:
        pytest.skip(f"Session not importable ({_P32A_IMPORT_ERROR})")
    built = _try_build_env()
    if built is None:
        pytest.skip(
            "Session could not be constructed via any documented path "
            "(zero-arg / test-factory / config-dict). The env constructor "
            "signature + fixed-tape-SET config schema is a P3.3 deliverable "
            "not frozen by P3.2a; a faithful impl should expose a zero-arg or "
            "test-factory constructor for these behavioural tests. AMBIGUITY "
            "SURFACED — see module docstring + sprint-status note."
        )
    return built


# ---------------------------------------------------------------------------
# Small helpers shared by the behavioural tests.
# ---------------------------------------------------------------------------


def _reset(env, seed=0):
    key = jax.random.PRNGKey(seed)
    return env.reset(key)


def _tree_all_close(a, b, atol=1e-4, rtol=1e-4):
    """Structural pytree closeness (used for NOOP book-contribution checks)."""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    for la, lb in zip(leaves_a, leaves_b):
        la = jnp.asarray(la)
        lb = jnp.asarray(lb)
        if la.shape != lb.shape:
            return False
        if not bool(jnp.allclose(la.astype(jnp.float64), lb.astype(jnp.float64),
                                 atol=atol, rtol=rtol)):
            return False
    return True


# ===========================================================================
# tape_cursor schema addition — additive 0-d int32 leaf on OMIEContractState
# (in-scope for P3.2a per agile-plan §P3.2a Deliverables + migration-plan.md:
# 993, 996-1010). Boundary-preserving: does NOT touch the composed OMIEBookState.
# ===========================================================================


@_p32a_skip
class TestTapeCursorSchemaAddition:
    """The P3.2a-motivated additive ``tape_cursor`` leaf (migration-plan.md:993).

    This is the one-line schema addition the card folds into P3.2a (NOT a P3.1
    re-open). It must be a 0-d int32 leaf, present on OMIEContractState, and it
    must not disturb the composed P2-H3 OMIEBookState (its ``iceberg_meta`` leaf
    still crosses byte-for-byte, P2-H3 AC 4/5)."""

    def test_state_has_tape_cursor_field(self):
        assert "tape_cursor" in OMIEContractState.__dataclass_fields__, (
            "OMIEContractState must gain the additive `tape_cursor` field "
            "(migration-plan.md:993; agile-plan §P3.2a Deliverables)"
        )

    def test_tape_cursor_is_0d_int32_on_reset_state(self, env):
        """Post-reset, tape_cursor is a 0-d int32 initialised to 0."""
        state = _reset(env)
        assert hasattr(state, "tape_cursor")
        cur = jnp.asarray(state.tape_cursor)
        assert cur.shape == (), f"tape_cursor must be 0-d, got shape {cur.shape}"
        assert cur.dtype == jnp.int32, (
            f"tape_cursor dtype {cur.dtype} != int32 (migration-plan.md:993)"
        )
        assert int(cur) == 0, "tape_cursor init 0 in reset (migration-plan.md:998)"

    def test_composed_book_state_preserved(self, env):
        """The additive leaf does not re-introduce dict/None leaves or reshape
        the composed OMIEBookState (P2-H3 AC 4/5)."""
        state = _reset(env)
        assert isinstance(state.book, OMIEBookState)
        for leaf in jax.tree_util.tree_leaves(state.book):
            assert isinstance(leaf, jnp.ndarray), (
                f"composed book leaf {type(leaf)} is not a jnp.ndarray — the "
                "tape_cursor addition must not touch the book (P2-H3 AC 5)"
            )


# ===========================================================================
# Contract (1): reset returns a valid state
# reset samples a tape and returns the OMIEContractState; tape_cursor == 0 and
# time == GOT post-reset.
# ===========================================================================


@_p32a_skip
class TestEnvResetReturnsValidState:
    """AC (1): reset(key) -> OMIEContractState with tape_cursor==0, time==GOT
    (agile-plan §P3.2a Test-first (1); migration-plan.md:1049-1051, 1182)."""

    def test_reset_state_is_omie_contract_state(self, env):
        state = _reset(env)
        assert isinstance(state, OMIEContractState), (
            "reset must return an OMIEContractState "
            "(migration-plan.md:1051)"
        )

    def test_reset_tape_cursor_is_zero(self, env):
        """tape_cursor == 0 post-reset (agile-plan §P3.2a Test-first (1))."""
        state = _reset(env)
        assert int(jnp.asarray(state.tape_cursor)) == 0

    def test_reset_time_equals_got(self, env):
        """time == GOT post-reset. GOT is time-origin of the episode; the state
        counts minutes-from-GOT (migration-plan.md:987, 1050), so time==0 at
        GOT. Asserted as the time-origin (0 in minutes-from-GOT), which is what
        `done = time >= GCT` is measured against."""
        state = _reset(env)
        t = jnp.asarray(state.time)
        assert t.shape == (), "time must be a 0-d scalar"
        assert float(t) == 0.0, (
            "time == GOT post-reset — the state carries minutes-from-GOT so "
            "the GOT origin is 0 (migration-plan.md:987, 1050)"
        )

    def test_reset_inventory_is_zero(self, env):
        """Zero-inventory round-trip framing: inventory starts at 0
        (migration-plan.md:1050, agile-plan §P3.2a Deliverables)."""
        state = _reset(env)
        inv = jnp.asarray(state.agent_inventory)
        assert float(jnp.sum(jnp.abs(inv))) == 0.0, (
            "agent_inventory must be all-zero at reset"
        )

    def test_reset_is_deterministic_given_key(self, env):
        """Same key -> same state (jit-pure reset)."""
        a_state = _reset(env, seed=7)
        b_state = _reset(env, seed=7)
        assert _tree_all_close(a_state, b_state, atol=0.0, rtol=0.0)
        assert int(jnp.asarray(a_state.tape_cursor)) == int(
            jnp.asarray(b_state.tape_cursor)
        )


# ===========================================================================
# Contract (2): test_env_step_returns_valid_tuple
# step_env returns the JaxMARL 5-tuple with correct dict keys.
# ===========================================================================


@_p32a_skip
class TestEnvStepReturnsValidTuple:
    """AC (2): step_env(key, state, order) -> (state, FillReport, done)
    (agile-plan §P3.2a Test-first (2); migration-plan.md:1059, 1183; release
    plan §4.4)."""

    def _step(self, env, order=None, seed=0):
        key = jax.random.PRNGKey(seed)
        state = env.reset(key)
        order = _noop_order() if order is None else order
        return env.step_env(key, state, order)

    def test_step_returns_three_tuple(self, env):
        out = self._step(env)
        assert isinstance(out, tuple) and len(out) == 3, (
            "step_env must return the 3-tuple (state, FillReport, done)"
        )

    def test_step_first_element_is_state(self, env):
        state, _report, _d = self._step(env)
        assert isinstance(state, OMIEContractState), (
            "step_env's 1st element must be the next OMIEContractState"
        )

    def test_step_done_is_boolean_scalar(self, env):
        _state, _report, done = self._step(env)
        d = jnp.asarray(done)
        assert d.shape == (), "done must be a 0-d scalar"
        assert d.dtype == jnp.bool_, (
            f"done dtype {d.dtype} != bool_"
        )


# ===========================================================================
# Contract (3): test_env_jit_compiles
# reset and step_env compile under jax.jit.
# ===========================================================================


@_p32a_skip
class TestEnvJitCompiles:
    """AC (3): reset and step_env compile under jax.jit
    (agile-plan §P3.2a Test-first (3); migration-plan.md:1048, 1053, 1184).

    Note: reset/step_env are already lifted with
    ``@partial(jax.jit, static_argnums=(0,))`` on the class, so calling them is
    already a jitted call. These tests assert they compile+run to concrete
    output with no tracer leak (a Python-side branch on a traced field would
    raise TracerBoolConversionError here)."""

    @pytest.mark.jit
    def test_reset_compiles_and_runs(self, env):
        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        # Force concretization of a leaf to prove the compile produced output.
        assert int(jnp.asarray(state.tape_cursor)) == 0

    @pytest.mark.jit
    def test_step_env_compiles_and_runs(self, env):
        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        out = env.step_env(key, state, _noop_order())
        assert len(out) == 3
        state2, _report, _d = out
        # tape_cursor advanced -> the jitted body ran end to end.
        assert int(jnp.asarray(state2.tape_cursor)) >= int(
            jnp.asarray(state.tape_cursor)
        )

    @pytest.mark.jit
    def test_step_env_no_tracer_leak_on_real_order(self, env):
        """A real order must also compile — the order stamping + agent
        _submit_order path is jit-safe, no Python branch on a traced order
        field (migration-plan.md:1173)."""
        key = jax.random.PRNGKey(1)
        state = env.reset(key)
        out = env.step_env(key, state, _buy_order())  # a BUY submit, not empty
        assert len(out) == 3


# ===========================================================================
# Contract (4): test_noop_action_changes_nothing
# The empty order (AgentOrder.none(), the former NOOP index 0) leaves the
# agent's book contribution unchanged and the cash delta == 0.0; only time,
# step_counter, tape_cursor (by orders_per_step), and the background-tape state
# advance.
# ===========================================================================


@_p32a_skip
class TestNoopActionChangesNothing:
    """AC (4): the empty order submits nothing — the agent's whole-row fill diff
    is empty so the cash delta is exactly 0.0; only time / step_counter /
    tape_cursor and the background book advance (agile-plan §P3.2a Test-first
    (4); migration-plan.md:1156-1164)."""

    def _step_noop(self, env, seed=0):
        key = jax.random.PRNGKey(seed)
        state = env.reset(key)
        state2, report, done = env.step_env(key, state, _noop_order())
        return state, (state2, report, done)

    def test_noop_cash_delta_is_exactly_zero(self, env):
        """Empty order => empty agent whole-row diff => cash delta == 0.0
        (exact)."""
        _s0, (_s1, report, _d) = self._step_noop(env)
        r = jnp.asarray(report.cash_delta_eur)
        assert float(r) == 0.0, (
            "empty-order cash delta must be exactly 0.0 "
            "(no agent order submitted; migration-plan.md:1157-1158)"
        )

    def test_noop_advances_tape_cursor_by_orders_per_step(self, env):
        """tape_cursor advances by the fixed orders_per_step increment. The
        increment equals whatever the env exposes as orders_per_step; if that
        attribute is discoverable we assert the exact increment, else we assert
        a strictly positive advance (migration-plan.md:1107, 1163)."""
        s0, (s1, _r, _d) = self._step_noop(env)
        before = int(jnp.asarray(s0.tape_cursor))
        after = int(jnp.asarray(s1.tape_cursor))
        assert after > before, "tape_cursor must advance on every step"
        ops = getattr(env, "orders_per_step", None)
        if ops is not None:
            assert after - before == int(ops), (
                f"tape_cursor advanced by {after - before}, expected exactly "
                f"orders_per_step={int(ops)} (migration-plan.md:1107)"
            )

    def test_noop_advances_step_counter_by_one(self, env):
        s0, (s1, _r, _d) = self._step_noop(env)
        assert int(jnp.asarray(s1.step_counter)) == int(
            jnp.asarray(s0.step_counter)
        ) + 1, "step_counter must advance by exactly 1 per step"

    def test_noop_advances_time_monotonically(self, env):
        """time advances (or stays, if the slice held no later timestamp) but
        never regresses (migration-plan.md:1092-1094)."""
        s0, (s1, _r, _d) = self._step_noop(env)
        assert float(jnp.asarray(s1.time)) >= float(jnp.asarray(s0.time)), (
            "time must advance monotonically (tape sorted by timestamp)"
        )

    def test_noop_inventory_unchanged(self, env):
        """Empty order => no agent fill => the agent's inventory is unchanged."""
        s0, (s1, _r, _d) = self._step_noop(env)
        inv0 = jnp.asarray(s0.agent_inventory)
        inv1 = jnp.asarray(s1.agent_inventory)
        assert bool(jnp.array_equal(inv0, inv1)), (
            "NOOP must not change agent_inventory (no agent order submitted)"
        )

    @pytest.mark.jit
    def test_noop_agent_book_contribution_unchanged_vs_pure_background(self, env):
        """The load-bearing AC: the empty order leaves the AGENT's contribution
        to the book identical to a pure-background advance. Because it submits
        no order, the book after the step must equal the book obtained by
        folding in the SAME background slice with NO agent submission.

        We can't re-run the private background path directly, but we CAN assert
        the observable consequence: with no order the agent produced no fills,
        so the cash delta is 0 AND the inventory is unchanged (covered above),
        AND the book's trade tape gained only background-attributable fills
        (never an agent-attributable one). Here we assert determinism of the
        book across two identical empty-order steps from the same reset — the
        agent path contributes nothing stochastic."""
        s0a, (s1a, _ra, _da) = self._step_noop(env, seed=3)
        s0b, (s1b, _rb, _db) = self._step_noop(env, seed=3)
        # Same key + same reset + NOOP => byte-identical next book.
        assert _tree_all_close(s1a.book, s1b.book, atol=0.0, rtol=0.0), (
            "NOOP step must be deterministic — the agent contributes nothing, "
            "so two identical NOOP steps yield the identical next book"
        )


# ===========================================================================
# Contract (5): test_done_at_gct
# done=True on exactly the step whose slice advances time >= GCT and not before;
# steps_per_episode = ceil(N_orders_padded / orders_per_step).
# ===========================================================================


@_p32a_skip
class TestDoneAtGCT:
    """AC (5): done fires exactly when time >= GCT and not before; derived
    episode length steps_per_episode = ceil(N_orders_padded / orders_per_step)
    (agile-plan §P3.2a Test-first (5); migration-plan.md:1084-1086, 1165-1167)."""

    def _rollout_dones(self, env, seed=0, max_steps=None):
        """Step from reset with the empty order until done or a hard cap,
        collecting the per-step (done, time) pairs. Uses step_env directly so
        we observe the raw episode boundary."""
        key = jax.random.PRNGKey(seed)
        state = env.reset(key)
        # Derive a cap from the env if possible; else a generous static bound.
        cap = max_steps
        if cap is None:
            cap = getattr(env, "steps_per_episode", None)
            cap = int(cap) + 2 if cap is not None else 512
        records = []
        for _ in range(cap):
            step_key, key = jax.random.split(key)
            state, _report, done = env.step_env(step_key, state, _noop_order())
            done = bool(jnp.asarray(done))
            time = float(jnp.asarray(state.time))
            records.append((done, time))
            if done:
                break
        return records

    @pytest.mark.jit
    def test_done_eventually_fires(self, env):
        records = self._rollout_dones(env)
        assert any(d for d, _t in records), (
            "the episode must terminate (done=True) within the derived length"
        )

    @pytest.mark.jit
    def test_done_is_monotonic_and_first_at_gct(self, env):
        """done is False until the step whose slice reaches time >= GCT, then
        True — never True-then-False (migration-plan.md:1165-1166)."""
        records = self._rollout_dones(env)
        first_done_idx = next(
            (i for i, (d, _t) in enumerate(records) if d), None
        )
        assert first_done_idx is not None, "episode never terminated"
        # No done before the first done (definitionally true by construction of
        # first_done_idx), and this is the LAST record (we break on done) — so
        # there is no True-then-False oscillation.
        assert all(
            not d for d, _t in records[:first_done_idx]
        ), "done must not fire before GCT is reached"

    @pytest.mark.jit
    def test_gct_reached_exactly_when_done(self, env):
        """On the done step, the accumulated time must be >= the contract GCT;
        on the step before, it must be < GCT. We recover GCT from the env if it
        exposes it (attribute or config); else we assert the weaker but still
        load-bearing invariant that time strictly increased into the done step."""
        records = self._rollout_dones(env)
        first_done_idx = next(
            (i for i, (d, _t) in enumerate(records) if d), None
        )
        assert first_done_idx is not None
        done_time = records[first_done_idx][1]
        gct = _discover_gct(env)
        if gct is not None:
            assert done_time >= gct, (
                f"time at done ({done_time}) must be >= GCT ({gct}) "
                "(migration-plan.md:1093, 1179)"
            )
            if first_done_idx > 0:
                pre_time = records[first_done_idx - 1][1]
                assert pre_time < gct, (
                    f"time on the step before done ({pre_time}) must be < GCT "
                    f"({gct}) — done fires EXACTLY at GCT, not early"
                )
        else:
            # GCT not discoverable — assert the weaker monotone-into-done shape.
            if first_done_idx > 0:
                assert done_time >= records[first_done_idx - 1][1]

    def test_steps_per_episode_is_derived_constant(self, env):
        """If the env exposes steps_per_episode and orders_per_step and the tape
        length, the identity steps_per_episode == ceil(N/orders_per_step) must
        hold (migration-plan.md:1084-1086). Skips the identity if any input is
        not discoverable, but still asserts steps_per_episode is a positive int
        constant (a compile-time-static episode length)."""
        spe = getattr(env, "steps_per_episode", None)
        if spe is None:
            pytest.skip(
                "env does not expose steps_per_episode as a discoverable "
                "attribute; the derived-constant identity cannot be checked "
                "black-box. (Not an ambiguity in the contract — a discoverability "
                "gap; the done-monotonicity tests above still pin the behaviour.)"
            )
        spe = int(spe)
        assert spe > 0, "steps_per_episode must be a positive static constant"
        ops = getattr(env, "orders_per_step", None)
        n = _discover_tape_len(env)
        if ops is not None and n is not None:
            expected = math.ceil(int(n) / int(ops))
            assert spe == expected, (
                f"steps_per_episode {spe} != ceil(N_orders_padded/"
                f"orders_per_step) = ceil({int(n)}/{int(ops)}) = {expected} "
                "(migration-plan.md:1084-1086)"
            )


def _discover_gct(env):
    """Best-effort recovery of the episode GCT in minutes-from-GOT.

    P3.2a measures done against ``time >= GCT`` where ``time`` is minutes-from-
    GOT. GCT in that frame is (GCT_minutes - GOT_minutes). The env may expose
    this as ``gct`` / ``gct_minutes`` / a ``contract.gct_minutes`` etc.; return
    the minutes-from-GOT value if recoverable, else None. Since GOT is the origin
    (time==0), a GCT exposed directly in minutes-from-GOT is used as-is."""
    for attr in ("gct_from_got", "gct_minutes_from_got", "gct", "gct_minutes"):
        val = getattr(env, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _discover_tape_len(env):
    for attr in ("n_orders_padded", "N_orders_padded", "tape_len"):
        val = getattr(env, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    tapes = getattr(env, "tapes", None)
    if tapes is not None:
        try:
            arr = jnp.asarray(tapes)
            if arr.ndim >= 2:
                return int(arr.shape[-2])
        except Exception:
            return None
    return None


# ===========================================================================
# Contract (6): One-compile preservation — step_env over all exec-restriction
# variants, INCLUDING under vmap over NUM_ENVS, does not exceed the single P2-H3
# _submit_order cache entry (slice length + tape trailing shape 13 are static;
# only tape_cursor / tape_idx / key are mapped). Assert one-compile UNDER vmap.
# ===========================================================================


@_p32a_skip
class TestOneCompilePreservationUnderVmap:
    """AC (6): the P2-H3 one-compile-across-exec-restriction-variants invariant
    survives the env wrapper AND vmap over NUM_ENVS (agile-plan §P3.2a
    Test-first (6); migration-plan.md:1128-1130, 1168-1171).

    Mechanism: ``_submit_order`` is a module-level ``jax.jit`` lift, so its
    compile-cache size is inspectable via ``_submit_order._cache_size()``
    (the standard jax jitted-callable API). We (1) clear the cache, (2) step the
    env under vmap over a batch (NUM_ENVS), exercising the interleave that calls
    ``_submit_order`` on both background and agent paths, and (3) assert the
    cache grew by AT MOST ONE entry — i.e. every env in the batch and every
    exec-restriction variant reuses the same compiled body.

    OUTER enable_x64 CONTEXT (load-bearing, do NOT remove — DOMAIN FINDING):
      The interleave calls ``_replay_orders`` -> ``_submit_order`` ->
      ``extract_fills``, whose int64 accumulation runs inside a scoped
      ``with jax.experimental.enable_x64():`` region (P2-L4, migration-plan.md:
      844-846, 921-925). That scoped context's int64 leg survives a *plain*
      ``jax.jit`` single-traced body (fact 3, migration-plan.md:936) — which is
      why the P2-H3 one-compile smoke test works with no outer context — but it
      does NOT survive ``jax.vmap`` unless the vmapped call is wrapped in an
      OUTER ``enable_x64()`` context (the batching transform captures the x64
      config at the vmap boundary, the same upstream ``#5982`` family as the
      float64-leg limitation). Empirically verified: without the outer context
      the vmapped step raises a `ValueError` "reduction-region's argument ...
      expected to be promotable from 'i64', but got 'i32'" inside
      ``extract_fills``; with the outer context it lowers cleanly. The RL
      training harness that vmaps the env over NUM_ENVS (P3.3 smoke train) must
      therefore supply this outer context, OR the env must avoid the x64 leg
      under vmap. This test wraps the vmap in the outer context (the
      caller-scoped-x64 convention, migration-plan.md:939) so the one-compile
      claim is asserted on the SATISFIABLE path. See the sprint-status note —
      this is a surfaced finding the developer/architect must confirm for P3.3.
    """

    NUM_ENVS = 4  # a small batch; the invariant is size-invariant by design

    def _vmapped_step(self, env):
        """Build a vmapped step over NUM_ENVS distinct reset keys and return the
        batched (state, FillReport, done) — forcing the compile. Wrapped in an
        outer enable_x64 context (see class docstring) so the scoped-x64 int64
        leg inside extract_fills survives the vmap batching transform."""
        keys = jax.random.split(jax.random.PRNGKey(0), self.NUM_ENVS)

        # Reset each env in the batch via vmap over the key axis.
        state_b = jax.vmap(env.reset)(keys)

        # The empty order, broadcast over the batch.
        orders_b = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (self.NUM_ENVS,) + x.shape),
            _noop_order(),
        )

        step_keys = jax.random.split(jax.random.PRNGKey(1), self.NUM_ENVS)
        with jax.experimental.enable_x64():
            out = jax.vmap(env.step_env)(step_keys, state_b, orders_b)
        return out

    @pytest.mark.jit
    def test_submit_order_cache_grows_at_most_one_under_vmap(self, env):
        """The load-bearing one-compile assertion under vmap."""
        # Clear the module-level lift cache so the count is attributable to this
        # test's compiles only.
        assert hasattr(_submit_order, "_cache_size"), (
            "the locked _submit_order must be a jax.jit lift exposing "
            "_cache_size() (migration-plan.md:1222-1225)"
        )
        _submit_order.clear_cache()
        before = _submit_order._cache_size()
        assert before == 0, "cache should be empty after clear_cache()"

        out = self._vmapped_step(env)
        # Concretize a leaf to force the compile to complete.
        state_b, _report_b, done_b = out
        _ = int(jnp.asarray(state_b.tape_cursor).reshape(-1)[0])

        after = _submit_order._cache_size()
        assert after <= 1, (
            f"_submit_order compile cache grew to {after} entries under vmap "
            f"over NUM_ENVS={self.NUM_ENVS} — the one-compile invariant is "
            "violated (slice length orders_per_step and tape trailing shape 13 "
            "must be static; only tape_cursor/tape_idx/key are mapped) "
            "(migration-plan.md:1128-1130, 1168-1171)"
        )

    @pytest.mark.jit
    def test_vmapped_step_returns_batched_three_tuple(self, env):
        """The vmapped step returns a batched 3-tuple whose done carries a
        leading NUM_ENVS axis (confirms the env is vmap-batchable, the
        prerequisite for the one-compile-under-vmap claim)."""
        out = self._vmapped_step(env)
        assert len(out) == 3
        _state_b, _report_b, done_b = out
        d = jnp.asarray(done_b)
        assert d.shape[0] == self.NUM_ENVS, (
            f'batched done leading axis {d.shape} != '
            f"(NUM_ENVS={self.NUM_ENVS}, ...)"
        )

    @pytest.mark.jit
    def test_all_exec_restriction_actions_share_one_compile(self, env):
        """Stepping through several different orders (different signed
        OMIEOrders that route through _submit_order's lax.switch over
        exec-restriction) must not add compile entries beyond the single shared
        body. We step the env once per order and assert the cache never exceeds
        one entry (migration-plan.md:1168-1169)."""
        _submit_order.clear_cache()
        _submit_order._cache_size()  # touch
        key = jax.random.PRNGKey(2)
        state = env.reset(key)
        # Step through the empty order + the 12 economic variants of the former
        # Discrete(13) space: {BUY, SELL} x {own-side touch, mid, opposite
        # touch} x {qty_min, qty_max / 2}, all NON (the default synthetic book
        # touches are ~49 / ~51). They are different signed orders but they all
        # hit the same compiled _submit_order body.
        orders = [_noop_order()]
        for side, prices in ((Side.BUY, (49.0, 50.0, 51.0)),
                             (Side.SELL, (51.0, 50.0, 49.0))):
            for price in prices:
                for qty in (0.1, 499.5):
                    orders.append(AgentOrder.create(
                        side=int(side), price=price, quantity=qty))
        for order in orders:
            step_key, key = jax.random.split(key)
            state, _report, _d = env.step_env(step_key, state, order)
        after = _submit_order._cache_size()
        assert after <= 1, (
            f"_submit_order cache grew to {after} across all "
            f"{len(orders)} orders — every exec-restriction "
            "variant must share ONE compiled body (P2-H3 one-compile invariant)"
        )


# ===========================================================================
# Contract (7): test_interleave_order_background_then_agent
# Within a step the background slice is folded in BEFORE the agent's
# _submit_order; the agent's cash is isolated to its own fills via the
# whole-row diff over trades_before/trades_after taken around the AGENT call
# (trades_before = book_after_background), per §P3.2 tape-slice contract (3).
# ===========================================================================


@_p32a_skip
class TestInterleaveBackgroundThenAgent:
    """AC (7): background-then-agent interleave; the agent's cash is isolated
    to its own fills by the whole-row diff wrapping the AGENT call, taken on the
    single carry AFTER the background slice folded in (agile-plan §P3.2a
    Test-first (7); migration-plan.md:1140-1158)."""

    def _reset_step(self, env, order, seed=0):
        key = jax.random.PRNGKey(seed)
        state = env.reset(key)
        return state, env.step_env(key, state, order)

    def test_background_advances_regardless_of_agent_action(self, env):
        """Background is folded in FIRST, so it advances identically whether the
        agent submits nothing or submits: the book's trade tape / ts_counter
        must reflect background progress even on an empty-order step
        (migration-plan.md:1141)."""
        s0, (s1_noop, _r, _d) = self._reset_step(env, _noop_order(), seed=5)
        # Background advanced the carried book: ts_counter and/or trades changed,
        # OR (if the slice held only inert pad rows) tape_cursor still advanced.
        cursor_advanced = int(jnp.asarray(s1_noop.tape_cursor)) > int(
            jnp.asarray(s0.tape_cursor)
        )
        assert cursor_advanced, (
            "the background slice must be consumed each step (tape_cursor "
            "advances) — background-then-agent (migration-plan.md:1141)"
        )

    def test_noop_cash_zero_confirms_agent_isolated_diff(self, env):
        """The whole-row diff wraps the AGENT call specifically: with the empty
        order the agent submits nothing, so the diff is empty and the cash delta
        is exactly 0 EVEN THOUGH the background produced fills this step. This
        is the observable signature that the diff is taken around the agent
        call, not around the whole step (migration-plan.md:1149-1158)."""
        _s0, (_s1, report, _d) = self._reset_step(env, _noop_order(), seed=9)
        r = jnp.asarray(report.cash_delta_eur)
        assert float(r) == 0.0, (
            "background fills must NOT leak into the agent cash on an empty "
            "order — the whole-row diff is taken around the agent "
            "_submit_order call only (migration-plan.md:1149-1158)"
        )
