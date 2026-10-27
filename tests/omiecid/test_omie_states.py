"""Locked behavioral contract for Story P3.1 — OMIEContractState.

These tests are the acceptance contract the Implementation Agent builds against.
They target a NEW module ``omiecid_simulator/omiecid/omie_states.py`` (mirrors the
``jaxen/StatesandParams.py`` pattern) that does NOT exist yet, so every test that
touches the module is guarded by ``@pytest.mark.skipif(_P31_MISSING, ...)`` — the
file stays fully COLLECTIBLE (`pytest --collect-only` succeeds) before the module
lands, and the guarded tests turn from SKIP into real assertions the moment the
developer creates the module.

Spec sources (authoritative):
  * ``docs/omiecid/agile-plan.md`` §P3.1 — the story card (6-field, size S).
  * ``docs/omiecid/migration-plan.md`` §P3.1 — the field list, dtypes, and the
    reserved ``day_ahead_anchor=0`` slot (migration-plan.md:977-1018).
  * ``omiecid_simulator/omiecid/contract_book.py:99-151`` — the locked P2-H3
    ``OMIEBookState`` (``iceberg_meta`` is a ``jnp.ndarray`` per P2-H3 AC 4;
    no dict/None leaves per AC 5).

Device policy: the OMIE state/obs harness is CPU-default; the CPU pin is applied
by ``tests/omiecid/conftest.py`` (``jax.config.update("jax_platform_name","cpu")``).
No test here requires a GPU, so none carries ``@pytest.mark.gpu``.

Runtime: all tests are pure state unit tests targeting the < 1s fast-suite
budget. ``test_state_is_jax_compatible`` compiles a trivial identity under
``jax.jit`` and carries ``@pytest.mark.jit`` per the repo's test-tiering
convention (``pyproject.toml`` markers).

Release translation (REL-07): the ``OMIEAgentObs`` observation contracts
(shape, anonymity schema) were removed with the RL layer; the
``OMIEContractState`` contracts are unchanged.
"""

import pytest

# ---------------------------------------------------------------------------
# P3.1 module import — guarded so this file stays collectible before the module
# exists (matches the M7.2 guard pattern in tests/omiecid/test_omie_data.py:26-51).
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp

    from omiecid_simulator.omiecid.contract_book import OMIEBookState
    from omiecid_simulator.omiecid.jax_constants import MAX_ICEBERGS

    from omiecid_simulator.omiecid.omie_states import OMIEContractState

    _P31_MISSING = False
    _P31_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - collection-time guard
    _P31_MISSING = True
    _P31_IMPORT_ERROR = repr(exc)

_p31_skip = pytest.mark.skipif(
    _P31_MISSING,
    reason=(
        "omiecid_simulator/omiecid/omie_states.py not yet implemented "
        f"({_P31_IMPORT_ERROR})"
    ),
)


# ===========================================================================
# Helpers — build a real OMIEBookState.
# ===========================================================================

def _make_contract_state(empty_state, **overrides):
    """Construct an OMIEContractState composing a real OMIEBookState.

    ``empty_state`` is the conftest ``book.init()`` fixture — a real P2-H3
    OMIEBookState with a genuine ``iceberg_meta`` jnp.ndarray leaf. The env-level
    fields (time / contract_status / step_counter / agent_inventory /
    agent_cash / tape_idx) are supplied as JAX-compatible scalars/arrays.
    """
    num_agents = 1
    fields = dict(
        book=empty_state,
        time=jnp.int32(0),
        contract_status=jnp.int32(1),  # OPEN=1 per migration-plan.md:986
        step_counter=jnp.int32(0),
        agent_inventory=jnp.zeros((num_agents,), dtype=jnp.float32),
        agent_cash=jnp.zeros((num_agents,), dtype=jnp.float32),
        tape_idx=jnp.int32(0),
    )
    fields.update(overrides)
    return OMIEContractState(**fields)


# ===========================================================================
# Contract (1): test_state_is_jax_compatible
# OMIEContractState (with a real OMIEBookState inside) passes through jax.jit.
# ===========================================================================

@_p31_skip
class TestStateIsJaxCompatible:
    """AC (1): OMIEContractState is a registered pytree that survives jax.jit."""

    def test_state_composes_omie_book_state(self, empty_state):
        """The env state COMPOSES the locked P2-H3 OMIEBookState (a `book` field),
        it does not replace or re-shape it."""
        state = _make_contract_state(empty_state)
        assert hasattr(state, "book"), "OMIEContractState must have a `book` field"
        assert isinstance(
            state.book, OMIEBookState
        ), "state.book must be the composed P2-H3 OMIEBookState instance"

    def test_state_has_all_env_level_fields(self, empty_state):
        """All six env-level fields from migration-plan.md:983-991 are present."""
        state = _make_contract_state(empty_state)
        for name in (
            "book",
            "time",
            "contract_status",
            "step_counter",
            "agent_inventory",
            "agent_cash",
            "tape_idx",
        ):
            assert hasattr(state, name), f"OMIEContractState missing field `{name}`"

    def test_state_is_registered_pytree(self, empty_state):
        """OMIEContractState must be a JAX pytree (flattens to leaves + treedef).

        A @struct.dataclass registers automatically; a plain class or NamedTuple
        with a non-array leaf would fail this. The composed OMIEBookState has no
        dict/None leaves (P2-H3 AC 5), so every leaf here is a jnp array."""
        state = _make_contract_state(empty_state)
        leaves, treedef = jax.tree_util.tree_flatten(state)
        assert len(leaves) > 0, "OMIEContractState must flatten to >=1 leaf"
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, OMIEContractState)

    @pytest.mark.jit
    def test_state_passes_through_jax_jit(self, empty_state):
        """The load-bearing AC: OMIEContractState round-trips cleanly through a
        jitted identity — no adapter, no re-shaping of the composed book."""
        state = _make_contract_state(empty_state)

        @jax.jit
        def _identity(s):
            # touch a leaf inside the composed book to force it into the trace
            return s._replace(
                step_counter=s.step_counter + jnp.int32(1)
            ) if hasattr(s, "_replace") else s.replace(
                step_counter=s.step_counter + jnp.int32(1)
            )

        out = _identity(state)
        assert isinstance(out, OMIEContractState)
        # the composed book survived the trace unchanged in structure
        assert isinstance(out.book, OMIEBookState)
        assert int(out.step_counter) == int(state.step_counter) + 1

    @pytest.mark.jit
    def test_state_is_vmappable(self, empty_state):
        """The env batches the state over NUM_ENVS via vmap; a leaf with a
        non-array (e.g. Python dict/None) would break this."""
        state = _make_contract_state(empty_state)
        batched = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (4,) + jnp.asarray(x).shape), state
        )

        @jax.jit
        @jax.vmap
        def _touch(s):
            return s.time + jnp.int32(1)

        out = _touch(batched)
        assert out.shape == (4,)


# ===========================================================================
# Contract (4): book.iceberg_meta carried as jnp.ndarray unchanged from P2-H3.
# No re-shaping, no dict re-introduction.
# ===========================================================================

@_p31_skip
class TestIcebergMetaCarriedUnchanged:
    """AC (4): OMIEContractState carries book.iceberg_meta as a jnp.ndarray
    unchanged from the locked P2-H3 contract (contract_book.py:142) — no
    re-shaping, no dict/None re-introduction (P2-H3 AC 4/5)."""

    def test_iceberg_meta_is_jnp_ndarray(self, empty_state):
        """iceberg_meta remains a JAX array (not a dict/list/None)."""
        state = _make_contract_state(empty_state)
        meta = state.book.iceberg_meta
        assert isinstance(meta, jnp.ndarray), (
            "book.iceberg_meta must remain jnp.ndarray (P2-H3 AC 4)"
        )

    def test_iceberg_meta_shape_unchanged(self, book, empty_state):
        """Shape is (max_icebergs, 8) — asserted PARAMETRICALLY against the
        book's configured ``max_icebergs``, not a hardcoded magic number.

        P3-PERF-REGEN-CAP-1: the first dim is now config-derived
        (``book._market_config.max_icebergs``) rather than the
        ``MAX_ICEBERGS`` ceiling constant; ``MAX_ICEBERGS`` remains the hard
        structural ceiling. P3.1 must still not re-shape the P2-H3 leaf."""
        cap = book._market_config.max_icebergs
        state = _make_contract_state(empty_state)
        assert state.book.iceberg_meta.shape == (cap, 8), (
            f"iceberg_meta shape {state.book.iceberg_meta.shape} != "
            f"(max_icebergs={cap}, 8) — P3.1 must not re-shape the "
            "P2-H3 leaf"
        )

    def test_iceberg_meta_dtype_unchanged(self, empty_state):
        """dtype is int32 per P2-H3 AC 4 (contract_book.py:142)."""
        state = _make_contract_state(empty_state)
        assert state.book.iceberg_meta.dtype == jnp.int32, (
            f"iceberg_meta dtype {state.book.iceberg_meta.dtype} != int32 "
            "(P2-H3 AC 4)"
        )

    def test_iceberg_meta_identical_to_source_book(self, empty_state):
        """Composing does not copy/re-shape the leaf — the array carried inside
        the state is byte-equal to the source OMIEBookState's iceberg_meta."""
        state = _make_contract_state(empty_state)
        assert jnp.array_equal(
            state.book.iceberg_meta, empty_state.iceberg_meta
        ), "composition must carry iceberg_meta unchanged (no re-shape/copy diff)"

    def test_composed_book_has_no_dict_or_none_leaves(self, empty_state):
        """P2-H3 AC 5: the composed OMIEBookState pytree has only array leaves —
        no dict/list/None re-introduced by the env-state wrapper. Every leaf of
        the composed book must be a JAX array."""
        state = _make_contract_state(empty_state)
        leaves = jax.tree_util.tree_leaves(state.book)
        assert len(leaves) > 0
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray), (
                f"composed book leaf {type(leaf)} is not a jnp.ndarray "
                "(dict/None re-introduction violates P2-H3 AC 5)"
            )
