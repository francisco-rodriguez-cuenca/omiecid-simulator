"""Locked behavioral contract for Story P3.4-FOLLOWUP-2 — harden
``Session._jit_identity()`` / ``__hash__`` / ``__eq__``.

These tests are the acceptance contract the Implementation Agent builds against.
They pin the TARGET behavior of the hardening described in
``docs/omiecid/agile-plan.md`` §P3.4-FOLLOWUP-2 (lines ~208-217):

  (1) test_env_hash_distinguishes_product_config
        two OMIEEnvs differing ONLY in a ``product_config`` field (``price_tick``)
        are NOT ``==`` and are hash-unequal — guards the currently-open silent
        wrong-executable collision (Deliverable 1).
  (2) test_env_hash_distinguishes_market_config
        same for a ``market_config`` field (``book_depth``).
  (3) test_env_hash_stable_and_dedups_byte_identical
        two byte-identical default-config envs remain ``==`` AND hash-equal —
        the P3.4 segfault-fix dedup must be PRESERVED.
  (4) test_env_hash_is_o1_per_call
        ``__hash__`` does not re-materialize the tape per call. Pinned
        STRUCTURALLY (per the card: "no ``tobytes`` on the hot path / a
        timing-independent structural assertion on the identity tuple's tape
        element") via a spy on ``ndarray.tobytes`` PLUS a structural check that
        ``_jit_identity()`` carries the cached ``_tape_digest`` + shape + dtype.
        NO wall-clock timing assertion (flaky).
  (5) test_default_env_reset_step_smoke
        behavior-neutrality: constructing a default Session and calling
        ``reset``/``step_env`` once still works (import + one-step smoke), so the
        hardening is provably behavior-neutral at THIS file's level.

TEST-FIRST NOTE (read carefully): the production change (adding
``product_config``/``market_config`` to ``_jit_identity()`` and caching
``_tape_digest`` at construction) does NOT exist yet. Tests (1), (2), (4) express
the TARGET contract and are EXPECTED to FAIL against the current unmodified code
— that is CORRECT: they define the contract the developer implements against.
Tests (3) and (5) pin PRESERVED behavior and should PASS against current code.
The tests are NOT weakened to pass against today's code.

Spec sources (authoritative, read on disk 2026-07-09):
  * ``docs/omiecid/agile-plan.md`` §P3.4-FOLLOWUP-2 — the story card (Description,
    Deliverables, 5-part Test-first contract).
  * ``omiecid_simulator/omiecid/omie_session.py`` — the current ``_jit_identity`` /
    ``__hash__`` / ``__eq__`` (~L273-294) + constructor (``product_config`` /
    ``market_config`` / ``tapes`` load, ~L140-201).
  * ``omiecid_simulator/omiecid/omie_config.py`` — ``OMIEProductConfig`` /
    ``OMIEMarketConfig`` (both ``@dataclass(frozen=True)`` ⇒ value-hashable).
  * ``docs/omiecid/migration-plan.md`` §P3.2 "One-compile + vmap invariant".

Device policy (``docs/omiecid/device-policy.md``): the OMIE test harness is
CPU-default; the CPU pin is applied by ``tests/omiecid/conftest.py`` (L186). No test
here adds a device pin, and none requires a GPU (no ``@pytest.mark.gpu``).

xdist-safety: tests (1)/(2) build genuinely-distinct-config OMIEEnvs (differing
``product_config``/``market_config``) which produce distinct compiled
executables; test (5) triggers a real ``reset``/``step_env`` compile. To keep the
file crash-free under ``-n 24`` shared workers, an autouse function-scoped
``jax.clear_caches()`` teardown fixture (mirroring
``test_omie_env_audit.py:243-250`` / the P3.3 ``_bound_inmemory_jit_cache``
precedent) bounds the per-worker in-memory compiled-executable pool. It clears
ONLY this process's in-memory tracing/lowering/compilation caches (does NOT
touch the on-disk ``.jax_cache`` — warm reuse preserved) and is inherently
xdist-safe.
"""

import pytest

# ---------------------------------------------------------------------------
# Import guard — keeps the file fully collectible (``pytest --collect-only``
# succeeds) even if the env module or its interfaces cannot be imported. Mirrors
# the guard idiom in ``tests/omiecid/test_omie_env.py``.
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp

    from omiecid_simulator.omiecid.omie_session import AgentOrder, Session
    from omiecid_simulator.omiecid.omie_config import (
        OMIEProductConfig,
        OMIEMarketConfig,
    )

    _MISSING = False
    _IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - collection-time guard
    _MISSING = True
    _IMPORT_ERROR = repr(exc)

_skip = pytest.mark.skipif(
    _MISSING,
    reason=(
        "omiecid_simulator/omiecid/omie_session.py (Session) / omie_config.py not "
        f"importable ({_IMPORT_ERROR})"
    ),
)


# ---------------------------------------------------------------------------
# Per-test in-memory JAX compile-cache bound (P3.3 ``_bound_inmemory_jit_cache``
# precedent; mirror of ``test_omie_env_audit.py:252-259``).
#
# Tests here build genuinely-distinct-config OMIEEnvs and run a real
# reset/step_env compile, each adding distinct XLA executables to this worker's
# in-memory compilation cache. Resetting after every test keeps the per-worker
# pool BOUNDED so the file cannot push a co-scheduled ``-n 24`` worker over the
# single-process XLA segfault threshold. Process-local / per-worker ⇒ inherently
# xdist-safe; the on-disk ``.jax_cache`` is untouched (warm reuse preserved).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _bound_inmemory_jit_cache():
    yield
    if not _MISSING:
        jax.clear_caches()


# ---------------------------------------------------------------------------
# Construction helpers.
# ---------------------------------------------------------------------------

def _default_env():
    """A default-config Session (self-contained synthetic tape set)."""
    return Session({})


def _shared_tapes():
    """A small, fixed, valid (n_tapes, N_padded, 13) int32 tape set to pass
    EXPLICITLY to both envs in the distinguish-config tests.

    CRITICAL for test faithfulness: the zero-arg constructor derives its default
    tapes from ``product_config`` (``_default_fixed_tape_set`` tick-encodes
    prices/qtys using ``price_tick``/``qty_tick``), so two envs built with
    different ``product_config`` would ALSO get different DEFAULT tapes — the
    current (tape-including) ``_jit_identity`` would then distinguish them for
    the WRONG reason (differing tape bytes, not the config field). Passing the
    SAME explicit tapes to both envs pins the tape digest identical, so the ONLY
    differing identity input is the ``product_config`` / ``market_config`` field
    under test — isolating exactly the collision the card targets.

    Rows use the loader's 13-column int32 layout; a couple of resting orders on
    each side plus inert pad rows (order_id <= 0, timestamp <= 0). The exact
    values are immaterial here — only that the array is a valid, shared,
    byte-identical tape set across the two envs.
    """
    tape = jnp.zeros((1, 8, 13), dtype=jnp.int32)
    # Two inert-but-well-formed resting orders (order_id > 0, timestamp > 0) so
    # the array is a plausible tape; remaining rows stay inert pad (all-zero).
    #   cols: [contract_id, side, price_int, qty_int, order_id, agent_id,
    #          timestamp, exec_restriction, validity, expiry, status,
    #          peak_qty_int, ppd_int]
    tape = tape.at[0, 0].set(
        jnp.array([0, 0, 500000, 100, 1, 777, 10, 0, 0, 0, 1, 0, 0],
                  dtype=jnp.int32)
    )
    tape = tape.at[0, 1].set(
        jnp.array([0, 1, 500100, 100, 2, 777, 20, 0, 0, 0, 1, 0, 0],
                  dtype=jnp.int32)
    )
    return tape


def _env_with_configs(product_config=None, market_config=None, tapes=None):
    """An Session whose product/market config is overridden while ALL other
    identity fields (tapes, orders_per_step, buckets, gct, ...) stay fixed — so
    any hash/eq difference is attributable ONLY to the config field under test.

    The caller passes an EXPLICIT shared ``tapes`` so the tape digest is held
    byte-identical across the pair (see ``_shared_tapes`` for why this matters).
    """
    config = {}
    if product_config is not None:
        config["product_config"] = product_config
    if market_config is not None:
        config["market_config"] = market_config
    if tapes is not None:
        config["tapes"] = tapes
    return Session(config)


# ===========================================================================
# Contract (1): test_env_hash_distinguishes_product_config
# ===========================================================================

@_skip
class TestEnvHashDistinguishesProductConfig:
    """Deliverable 1 (correctness robustness): two envs differing ONLY in a
    ``product_config`` field must NOT be ``==`` and must be hash-unequal —
    otherwise the value-based static jit key silently collides two functionally
    different envs onto ONE compiled executable (a wrong-executable bug that
    fires the moment a non-default product config is passed).

    EXPECTED-RED against current code: ``_jit_identity()`` omits
    ``product_config`` today, so the two envs are wrongly ``==`` / hash-equal.
    """

    def test_env_hash_distinguishes_product_config(self):
        default_prod = OMIEProductConfig()
        # Differ in EXACTLY one product_config field (price_tick). All other
        # fields keep their defaults so the ONLY difference between the two envs
        # is a product_config value.
        other_prod = OMIEProductConfig(price_tick=default_prod.price_tick * 2.0)
        assert other_prod != default_prod, (
            "test precondition: the two product configs must differ"
        )

        # Shared explicit tapes ⇒ the tape digest is byte-identical across the
        # pair, so the ONLY differing identity input is product_config (see
        # _shared_tapes docstring — otherwise the default tapes would differ and
        # the test would pass for the wrong reason).
        tapes = _shared_tapes()
        env_a = _env_with_configs(product_config=default_prod, tapes=tapes)
        env_b = _env_with_configs(product_config=other_prod, tapes=tapes)

        assert env_a != env_b, (
            "OMIEEnvs differing only in product_config.price_tick must NOT "
            "compare equal — the compiled reset/step_env jaxpr closes over "
            "product tick/bound math, so treating them as equal collides two "
            "functionally different envs onto one executable (agile-plan "
            "§P3.4-FOLLOWUP-2 Deliverable 1)."
        )
        assert hash(env_a) != hash(env_b), (
            "product_config difference must propagate into __hash__ (the value "
            "must appear in _jit_identity())."
        )

    def test_env_hash_identity_tuple_contains_product_config(self):
        """Structural pin: the identity tuple must actually CARRY the
        product_config value (frozen dataclass ⇒ value-hashable, O(1)), so the
        distinction above is not accidental."""
        env = _default_env()
        identity = env._jit_identity()
        assert env.product_config in identity, (
            "_jit_identity() must include self.product_config (a frozen, "
            "value-hashable dataclass) so functionally-different configs never "
            "collide (agile-plan §P3.4-FOLLOWUP-2 Deliverable 1)."
        )


# ===========================================================================
# Contract (2): test_env_hash_distinguishes_market_config
# ===========================================================================

@_skip
class TestEnvHashDistinguishesMarketConfig:
    """Deliverable 1 (correctness robustness), market side: the compiled jaxpr
    closes over ``market_config`` via ``self._jaxob_cfg = _make_jaxob_cfg(...)``
    (best-bid/ask/volume reads baked into ``_get_obs``). Two envs differing only
    in a ``market_config`` field must therefore NOT be ``==`` / hash-equal.

    EXPECTED-RED against current code.
    """

    def test_env_hash_distinguishes_market_config(self):
        default_mkt = OMIEMarketConfig()
        # Differ in EXACTLY one market_config field (book_depth).
        other_mkt = OMIEMarketConfig(book_depth=default_mkt.book_depth + 1)
        assert other_mkt != default_mkt, (
            "test precondition: the two market configs must differ"
        )

        # Shared explicit tapes ⇒ only market_config differs across the pair.
        tapes = _shared_tapes()
        env_a = _env_with_configs(market_config=default_mkt, tapes=tapes)
        env_b = _env_with_configs(market_config=other_mkt, tapes=tapes)

        assert env_a != env_b, (
            "OMIEEnvs differing only in market_config.book_depth must NOT "
            "compare equal — the compiled jaxpr closes over market_config via "
            "_make_jaxob_cfg (agile-plan §P3.4-FOLLOWUP-2 Deliverable 1)."
        )
        assert hash(env_a) != hash(env_b), (
            "market_config difference must propagate into __hash__."
        )

    def test_env_hash_identity_tuple_contains_market_config(self):
        """Structural pin: the identity tuple must carry the market_config."""
        env = _default_env()
        identity = env._jit_identity()
        assert env.market_config in identity, (
            "_jit_identity() must include self.market_config (a frozen, "
            "value-hashable dataclass) (agile-plan §P3.4-FOLLOWUP-2 "
            "Deliverable 1)."
        )


# ===========================================================================
# Contract (3): test_env_hash_stable_and_dedups_byte_identical
# ===========================================================================

@_skip
class TestEnvHashStableAndDedupsByteIdentical:
    """The P3.4 segfault-fix dedup must be PRESERVED: two byte-identical
    default-config envs are functionally interchangeable and MUST share one
    static jit key (``==`` AND hash-equal) so they compile once, not twice.

    EXPECTED-GREEN against current code (this pins preserved behavior).
    """

    def test_two_default_envs_are_equal_and_hash_equal(self):
        env_a = _default_env()
        env_b = _default_env()
        assert env_a == env_b, (
            "two byte-identical default-config OMIEEnvs must compare equal so "
            "they dedup onto one compiled executable (the P3.4 cross-call "
            "segfault fix — agile-plan §P3.4-FOLLOWUP-2 Deliverable 2 / test 3)."
        )
        assert hash(env_a) == hash(env_b), (
            "equal envs must hash equal (Python __hash__/__eq__ contract) so "
            "they collapse to one static jit key."
        )

    def test_hash_is_stable_across_repeated_calls(self):
        """__hash__ is deterministic for a fixed env (no per-call drift from a
        re-materialized/re-ordered tape digest)."""
        env = _default_env()
        h0 = hash(env)
        for _ in range(3):
            assert hash(env) == h0, (
                "hash(env) must be stable across repeated calls on the same "
                "instance"
            )

    def test_equal_env_usable_as_dict_key_dedups(self):
        """The dedup is observable at the dict/set level: two equal envs collapse
        to ONE key (this is precisely how the static-argnums jit cache keys)."""
        env_a = _default_env()
        env_b = _default_env()
        assert len({env_a, env_b}) == 1, (
            "two equal default-config envs must collapse to a single set/dict "
            "key — the mechanism that bounds distinct compiled executables by "
            "distinct CONFIG rather than by instance count."
        )


# ===========================================================================
# Contract (4): test_env_hash_is_o1_per_call
# ===========================================================================

@_skip
class TestEnvHashIsO1PerCall:
    """Deliverable 2 (perf robustness): the tape content digest is computed ONCE
    at construction (``self._tape_digest``) and REFERENCED in ``_jit_identity()``
    alongside ``self.tapes.shape`` + ``str(self.tapes.dtype)`` — NOT a
    per-dispatch ``self.tapes.tobytes()``. Pinned STRUCTURALLY / via a spy on
    ``ndarray.tobytes`` (no wall-clock timing assertion — those are flaky).

    EXPECTED-RED against current code: ``_jit_identity()`` calls
    ``self.tapes.tobytes()`` on every invocation today.
    """

    def test_construction_caches_tape_digest(self):
        """The construction-time digest attribute must exist after __init__."""
        env = _default_env()
        assert hasattr(env, "_tape_digest"), (
            "__init__ must cache the tape content digest as self._tape_digest "
            "(agile-plan §P3.4-FOLLOWUP-2 Deliverable 2) so __hash__ is O(1)."
        )

    def test_identity_tuple_carries_cached_digest_shape_dtype(self):
        """Structural pin: the identity tuple must reference the CACHED digest
        plus the (cheap, O(1)) tape shape + dtype string — not the raw tape
        bytes recomputed per call."""
        env = _default_env()
        identity = env._jit_identity()
        assert env._tape_digest in identity, (
            "_jit_identity() must reference the cached self._tape_digest, not a "
            "per-dispatch self.tapes.tobytes()."
        )
        assert tuple(env.tapes.shape) in identity, (
            "_jit_identity() must carry the tape shape (O(1)) so envs with "
            "differently-shaped tape sets never collide."
        )
        assert str(env.tapes.dtype) in identity, (
            "_jit_identity() must carry str(tapes.dtype) (O(1)) to disambiguate "
            "same-content/different-dtype tapes."
        )

    def test_hash_does_not_call_tobytes_on_hot_path(self):
        """Timing-independent structural guard: hashing an already-constructed
        env must NOT re-materialize the tape via ``tapes.tobytes()``.

        The JAX/NumPy array types are immutable C types (their ``tobytes``
        cannot be monkeypatched), so we swap the env instance's ``self.tapes``
        for a thin PROXY that delegates every attribute/operation to the real
        array but COUNTS ``tobytes`` calls. We then assert ``tobytes`` is NOT
        called during ``hash(env)`` / ``env._jit_identity()``. This is the
        structural (not wall-clock) pin the card asks for."""
        env = _default_env()
        real_tapes = env.tapes
        # Force any lazily-deferred construction-time work to settle first, so
        # the counter below only captures the HOT path (hash/identity), not
        # __init__ (the target impl digests the tape ONCE here).
        _ = repr(real_tapes.shape)

        calls = {"n": 0}

        class _TobytesSpyingTape:
            """Delegates everything to the wrapped array; counts tobytes()."""

            def __init__(self, wrapped):
                object.__setattr__(self, "_wrapped", wrapped)

            def tobytes(self, *args, **kwargs):
                calls["n"] += 1
                return self._wrapped.tobytes(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(object.__getattribute__(self, "_wrapped"), name)

            def __getitem__(self, item):
                return object.__getattribute__(self, "_wrapped")[item]

        env.tapes = _TobytesSpyingTape(real_tapes)
        try:
            # The hot path: repeated hashing (the ~1x-per-jit-dispatch path).
            for _ in range(5):
                hash(env)
                env._jit_identity()
        finally:
            env.tapes = real_tapes

        assert calls["n"] == 0, (
            "hash(env) / _jit_identity() must NOT call ndarray.tobytes() on the "
            "hot path — the tape digest is computed ONCE at construction and "
            "cached as self._tape_digest, turning per-dispatch O(tape) into "
            f"O(1) (agile-plan §P3.4-FOLLOWUP-2 Deliverable 2). Saw "
            f"{calls['n']} tobytes call(s)."
        )


# ===========================================================================
# Contract (5): behavior-neutrality smoke — reset + one step still work.
# ===========================================================================

@_skip
@pytest.mark.jit
class TestDefaultEnvResetStepSmoke:
    """Behavior-neutrality guard at this file's level: the hardening is additive
    (identity-key completeness + a cached digest) and must not perturb any
    observable env behavior. Constructing a default Session and running
    reset + one step_env must still succeed and return the Session tuples.

    EXPECTED-GREEN against current code (pins preserved behavior). Marked
    ``jit`` (triggers a real reset/step_env compile) per the OMIE test-tiering
    convention.
    """

    def test_default_env_reset_step_smoke(self):
        env = _default_env()
        key = jax.random.PRNGKey(0)

        state = env.reset(key)
        assert state is not None, "reset must return a state"

        out = env.step_env(key, state, AgentOrder.none())  # no order
        assert isinstance(out, tuple) and len(out) == 3, (
            "step_env must return the 3-tuple (next_state, FillReport, done)"
        )
        _next_state, _report, done = out
        assert jnp.asarray(done).dtype == jnp.bool_, (
            "step_env's done must be a boolean scalar"
        )
