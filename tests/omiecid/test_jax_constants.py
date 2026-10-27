"""Locked behavioral tests for Story P2-F1 — Static JAX constants.

Contract source:
  - docs/omiecid/agile-plan.md §Phase 2 → P2-F1 story card
  - docs/omiecid/migration-plan.md §Phase 2 Blockers table (B1, B2, B5)
  - docs/omiecid/sprint-status.md §"P2-F1 — Architecture Review" (2026-04-22)
  - docs/omiecid/phase1-5-summary.md §4 (agent_id = pair), §8.1

After P2-F1 merges, ``omiecid_simulator/omiecid/jax_constants.py`` must expose
three module-level Python ``int`` constants:

  * ``MAX_ICEBERGS`` (B1) — static upper bound on the number of iceberg
    orders tracked simultaneously by a single ``ContractBook``. Used as
    the row-count of ``OMIEBookState.iceberg_meta: jnp.ndarray`` in P2-H1.
  * ``MAX_SLICES_PER_REGEN`` (B2) — static upper bound on slice generations
    inside one ``regenerate_slices`` / ``matchable_hidden_volume`` call.
    Used as the max-trip-count of the ICEwPI ``jax.lax.while_loop`` in P2-H1.
  * ``MAX_AGENTS`` (B5) — static upper bound on the distinct
    ``(Agente, Unidad)`` pair universe of a session. Used as the per-agent
    pad dimension for settlement (P2-L2) and the optional B4-scoped
    per-agent query arrays (P2-H2).

Behavioral invariants locked by this file (derived from the P2-F1 Test-first
contract in the story card):

  1. Public import surface:
       ``from omiecid_simulator.omiecid.jax_constants import (
            MAX_ICEBERGS, MAX_SLICES_PER_REGEN, MAX_AGENTS)`` succeeds.

  2. Each constant is a Python ``int`` (not ``bool``, not ``jnp.int32``,
     not ``jnp.ndarray``). This is what lets JIT treat them as compile-time
     static-argument values — tracer leakage is avoided by keeping the
     type concrete.

  3. Each constant is ``> 0``.

  4. Each constant can be passed as a ``static_argnums`` value to a
     ``jax.jit``-wrapped function without tracer leakage; the compiled
     function runs and returns the expected value.

  5. ``MAX_ICEBERGS`` coupling: ``MAX_ICEBERGS >= OMIEMarketConfig().book_depth``
     (the jaxob-level ceiling). An iceberg order occupies a resting book
     slot for its visible peak; any overflow of the iceberg pool implies
     an overflow of the underlying book, which already has its own overflow
     path. See architect-review §3.

  6. Rationale docstring exists: ``jax_constants.__doc__`` is non-empty
     AND each of the three names has an attached rationale (either the
     module docstring or an adjacent ``__all__`` / docstring convention).
     The story card demands "module-level docstring paragraph stating the
     chosen value, how it was derived, and the cost of revising it."

  7. ``MAX_AGENTS`` empirical sufficiency (slow — real-data fixture):
     For each of three representative sampled sessions (summer
     ``2025-06-01``, autumn ``2025-10-15``, winter ``2026-01-12``) loaded
     via ``load_omie_agent_metadata(config)``, the observed distinct
     ``(Agente, Unidad)`` pair count is ``<= MAX_AGENTS``. Per the
     architect review §2, the sampling window is a deliberate 3-session
     seasonal spread and a single canonical contract (``contract_id=81``)
     is used per session to keep the fixture load under the CI budget.

  8. ``MAX_SLICES_PER_REGEN`` is at least as large as an iceberg peeled
     at its minimum meaningful peak_qty for a reasonable session: must
     exceed 1 (single-slice case would defeat iceberg semantics). The
     exact floor is an architect/developer decision (recommended ``64``
     per architect review §3); this test only pins the existence of a
     non-trivial bound. The truncation hazard is called out in the
     rationale (covered by test 6).

Collectibility: every real-data-dependent test is guarded by
``pytest.mark.skipif(not file.exists(), ...)`` so ``pytest --collect-only``
always succeeds, even on machines lacking ``data/rawOMIE/``.

Expected failure mode against current code: ``omiecid_simulator/omiecid/
jax_constants.py`` does not exist on the pre-P2-F1 branch, so every test
that imports from that module raises ``ModuleNotFoundError`` at collection
time *would* break collection — to avoid that we import lazily inside
each test via ``importlib`` / ``pytest.importorskip``-style guarding.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig


# ---------------------------------------------------------------------------
# Fixtures & constants
# ---------------------------------------------------------------------------

REAL_DATA_PATH = "data/rawOMIE"

# Three representative sessions per architect review §2 (seasonal spread).
# A single canonical contract_id=81 per session keeps fixture load cost
# inside the fast-suite budget when combined with file-level memoisation.
SAMPLED_SESSIONS = [
    ("2025-06-01", 81),  # summer
    ("2025-10-15", 81),  # autumn shoulder
    ("2026-01-12", 81),  # winter (also the canonical P2-F0 regression date)
]


def _orders_file_for(delivery_date: str) -> Path:
    return Path(REAL_DATA_PATH) / f"orders_{delivery_date.replace('-', '')}.1"


def _all_sampled_files_present() -> bool:
    return all(_orders_file_for(d).exists() for d, _ in SAMPLED_SESSIONS)


JAX_CONSTANTS_MODULE = "omiecid_simulator.omiecid.jax_constants"


def _import_jax_constants():
    """Import the jax_constants module, raising a clear pytest failure if absent.

    Using importlib (not a top-level import) ensures pytest collection
    succeeds even when the module does not yet exist on disk — the tests
    then fail informatively at run time rather than aborting collection
    for every test in the file.
    """
    return importlib.import_module(JAX_CONSTANTS_MODULE)


# ---------------------------------------------------------------------------
# 1. Public API — import surface
# ---------------------------------------------------------------------------


class TestPublicImportSurface:
    """The three named constants must be publicly importable."""

    def test_module_imports(self):
        """``omiecid_simulator.omiecid.jax_constants`` is importable."""
        mod = _import_jax_constants()
        assert mod is not None

    def test_max_icebergs_is_attribute(self):
        mod = _import_jax_constants()
        assert hasattr(mod, "MAX_ICEBERGS"), (
            "jax_constants module must expose MAX_ICEBERGS (B1)."
        )

    def test_max_slices_per_regen_is_attribute(self):
        mod = _import_jax_constants()
        assert hasattr(mod, "MAX_SLICES_PER_REGEN"), (
            "jax_constants module must expose MAX_SLICES_PER_REGEN (B2)."
        )

    def test_max_agents_is_attribute(self):
        mod = _import_jax_constants()
        assert hasattr(mod, "MAX_AGENTS"), (
            "jax_constants module must expose MAX_AGENTS (B5)."
        )

    def test_from_import_syntax_works(self):
        """``from ... import MAX_ICEBERGS, MAX_SLICES_PER_REGEN, MAX_AGENTS`` succeeds."""
        mod = _import_jax_constants()
        # simulate the public `from` import syntax by reading attributes.
        max_icebergs = getattr(mod, "MAX_ICEBERGS")
        max_slices = getattr(mod, "MAX_SLICES_PER_REGEN")
        max_agents = getattr(mod, "MAX_AGENTS")
        assert max_icebergs is not None
        assert max_slices is not None
        assert max_agents is not None


# ---------------------------------------------------------------------------
# 2. Type contract — Python int, not bool, not jnp.int32, not jnp.ndarray
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["MAX_ICEBERGS", "MAX_SLICES_PER_REGEN", "MAX_AGENTS"],
)
class TestTypeContract:
    """Each constant must be a concrete Python int.

    Python int (and NOT bool, even though bool is a subclass of int) is
    the type JIT treats as a compile-time static argument. ``jnp.int32``
    / ``jnp.ndarray`` would become a traced value and break downstream
    static_argnums usage.
    """

    def test_constant_is_int(self, name):
        mod = _import_jax_constants()
        value = getattr(mod, name)
        assert isinstance(value, int), (
            f"{name} must be a Python int, got {type(value).__name__}"
        )

    def test_constant_is_not_bool(self, name):
        """bool is a subclass of int in Python; explicitly reject it."""
        mod = _import_jax_constants()
        value = getattr(mod, name)
        assert not isinstance(value, bool), (
            f"{name} must not be a bool (bool is a subclass of int but "
            f"semantically wrong here); got {value!r}"
        )

    def test_constant_is_not_jax_array(self, name):
        """A jnp.ndarray / jnp.int32 scalar would become a tracer in JIT."""
        mod = _import_jax_constants()
        value = getattr(mod, name)
        assert not isinstance(value, jnp.ndarray), (
            f"{name} must not be a jnp.ndarray (would become a tracer); "
            f"got {type(value).__name__}"
        )
        # jnp.int32 scalars are also not acceptable.
        assert "jax" not in type(value).__module__, (
            f"{name} must not be a jax scalar type; got "
            f"{type(value).__module__}.{type(value).__name__}"
        )

    def test_constant_is_positive(self, name):
        mod = _import_jax_constants()
        value = getattr(mod, name)
        assert value > 0, f"{name} must be > 0, got {value}"


# ---------------------------------------------------------------------------
# 3. JIT staticness — passing each constant via static_argnums compiles
#    without tracer leakage.
# ---------------------------------------------------------------------------


class TestJITStaticness:
    """Each constant must be usable as a ``static_argnums`` value."""

    @pytest.mark.jit
    def test_max_icebergs_static_argnum(self):
        mod = _import_jax_constants()
        max_icebergs = getattr(mod, "MAX_ICEBERGS")

        def make_zero_iceberg_meta(n: int):
            # n is a Python int; we need it static because shape depends on it.
            return jnp.zeros((n, 8), dtype=jnp.int32)

        # jit-ing a shape-dependent function with a Python-int constant must
        # compile cleanly. If MAX_ICEBERGS were a tracer / jnp.ndarray, the
        # shape argument would raise a TracerIntegerConversionError.
        static = jax.jit(make_zero_iceberg_meta, static_argnums=(0,))
        result = static(max_icebergs)
        assert result.shape == (max_icebergs, 8)
        assert result.dtype == jnp.int32

    @pytest.mark.jit
    def test_max_slices_per_regen_static_argnum(self):
        mod = _import_jax_constants()
        max_slices = getattr(mod, "MAX_SLICES_PER_REGEN")

        def make_slice_buffer(n: int):
            return jnp.zeros((n,), dtype=jnp.int32)

        static = jax.jit(make_slice_buffer, static_argnums=(0,))
        result = static(max_slices)
        assert result.shape == (max_slices,)
        assert result.dtype == jnp.int32

    @pytest.mark.jit
    def test_max_agents_static_argnum(self):
        mod = _import_jax_constants()
        max_agents = getattr(mod, "MAX_AGENTS")

        def make_per_agent_array(n: int):
            return jnp.zeros((n,), dtype=jnp.int32)

        static = jax.jit(make_per_agent_array, static_argnums=(0,))
        result = static(max_agents)
        assert result.shape == (max_agents,)
        assert result.dtype == jnp.int32

    @pytest.mark.jit
    def test_all_three_combined_in_one_jit_shape(self):
        """Combined use: a single JITted function keys all three as static."""
        mod = _import_jax_constants()
        max_ic = getattr(mod, "MAX_ICEBERGS")
        max_sl = getattr(mod, "MAX_SLICES_PER_REGEN")
        max_ag = getattr(mod, "MAX_AGENTS")

        def _alloc(a: int, b: int, c: int):
            return (
                jnp.zeros((a, 8), dtype=jnp.int32),
                jnp.zeros((b,), dtype=jnp.int32),
                jnp.zeros((c,), dtype=jnp.int32),
            )

        static = jax.jit(_alloc, static_argnums=(0, 1, 2))
        ic, sl, ag = static(max_ic, max_sl, max_ag)
        assert ic.shape == (max_ic, 8)
        assert sl.shape == (max_sl,)
        assert ag.shape == (max_ag,)


# ---------------------------------------------------------------------------
# 4. MAX_ICEBERGS coupling with OMIEMarketConfig.book_depth
# ---------------------------------------------------------------------------


class TestMaxIcebergsBookDepthCoupling:
    """MAX_ICEBERGS must be >= book_depth.

    Rationale (architect-review §3): a single contract's iceberg population
    is bounded by the underlying jaxob book's resting-order capacity. An
    iceberg occupies a book slot for its visible peak (see
    ``iceberg_manager.py:201-211``). Choosing ``MAX_ICEBERGS >= book_depth``
    means an iceberg-slot overflow can only occur via a book overflow,
    which already has its own handling path.
    """

    def test_max_icebergs_geq_default_book_depth(self):
        mod = _import_jax_constants()
        max_icebergs = getattr(mod, "MAX_ICEBERGS")
        default_book_depth = OMIEMarketConfig().book_depth
        assert max_icebergs >= default_book_depth, (
            f"MAX_ICEBERGS ({max_icebergs}) < OMIEMarketConfig().book_depth "
            f"({default_book_depth}) — iceberg pool would overflow before "
            f"the book does; see architect-review §3."
        )


# ---------------------------------------------------------------------------
# 5. Non-trivial bounds for MAX_SLICES_PER_REGEN
# ---------------------------------------------------------------------------


class TestMaxSlicesPerRegenNonTrivial:
    """MAX_SLICES_PER_REGEN must admit a realistic iceberg slice sequence.

    A single-slice bound would defeat the purpose of iceberg orders (the
    whole point is that total_qty is peeled across > 1 slice). The
    architect review §3 recommends ``64`` as a safe, analytically-derived
    default for typical peak_qty values. We pin a soft floor of ``2``
    here because the exact number is a developer-owned decision and
    further tightening is the architecture-review's job, not this file's.
    """

    def test_max_slices_per_regen_at_least_two(self):
        mod = _import_jax_constants()
        value = getattr(mod, "MAX_SLICES_PER_REGEN")
        assert value >= 2, (
            f"MAX_SLICES_PER_REGEN must admit > 1 slice (single-slice "
            f"defeats iceberg semantics); got {value}"
        )


# ---------------------------------------------------------------------------
# 6. Rationale docstring
# ---------------------------------------------------------------------------


class TestModuleDocstring:
    """The story card demands rationale documentation in docstrings.

    Quoting P2-F1 description: "Each constant carries a module-level
    docstring paragraph stating the chosen value, how it was derived
    (data snapshot + method for B5; analytical upper bound for B1/B2),
    and the cost of revising it (which arrays change shape, which caches
    invalidate)."

    These are coarse-grained structural checks, not prose-quality checks.
    """

    def test_module_has_docstring(self):
        mod = _import_jax_constants()
        assert mod.__doc__ is not None, (
            "jax_constants must have a module-level docstring."
        )
        assert len(mod.__doc__.strip()) > 0, (
            "jax_constants module docstring must not be empty."
        )

    @pytest.mark.parametrize(
        "name",
        ["MAX_ICEBERGS", "MAX_SLICES_PER_REGEN", "MAX_AGENTS"],
    )
    def test_constant_mentioned_in_module_doc(self, name):
        """The three constants must be named in the module-level docstring
        (where the P2-F1 card demands the rationale paragraph lives)."""
        mod = _import_jax_constants()
        doc = (mod.__doc__ or "")
        assert name in doc, (
            f"{name} must appear in jax_constants module docstring "
            f"(rationale paragraph required per P2-F1 story card)."
        )


# ---------------------------------------------------------------------------
# 7. MAX_AGENTS empirical sufficiency — SLOW, real-data fixture
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMaxAgentsEmpiricalSufficiency:
    """MAX_AGENTS must exceed observed pair counts in the sampled sessions.

    The story card Test-first contract states: "``MAX_AGENTS`` is >= max
    distinct ``(Agente, Unidad)`` pairs observed in the sampled sessions
    (property asserted in a unit test loading the sampled fixtures)."

    Per architect review §2, the sampled window is three representative
    daily sessions across summer / autumn / winter. We query each session
    via ``load_omie_agent_metadata(config)`` on a single canonical
    contract (``contract_id=81``) to keep the fixture load under the CI
    budget. A 2× safety margin rolled into MAX_AGENTS's chosen value (per
    the architect recommendation) means this test should pass comfortably
    against any realistic derivation.

    Slow marker justification: each sampled session loads a 30-44 MB CSV
    through ``load_omie_agent_metadata`` (with parser overhead) — the
    multi-session spread can exceed the < 1 s fast-suite budget on cold
    cache.
    """

    @pytest.mark.skipif(
        not _all_sampled_files_present(),
        reason=(
            "Not all sampled session files present under data/rawOMIE/. "
            "Required: "
            + ", ".join(str(_orders_file_for(d)) for d, _ in SAMPLED_SESSIONS)
        ),
    )
    def test_max_agents_geq_observed_pair_count_per_sampled_session(self):
        """For every sampled (delivery_date, contract_id), observed pair
        count is <= MAX_AGENTS."""
        # Import lazily so the module not existing yet produces a clear
        # ModuleNotFoundError per-test rather than aborting collection.
        mod = _import_jax_constants()
        max_agents = getattr(mod, "MAX_AGENTS")

        # Also import the loader accessor lazily (already available under
        # jaxomie — this import should always succeed post Phase 1.5).
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )

        observed_counts: dict[tuple[str, int], int] = {}
        for delivery_date, contract_id in SAMPLED_SESSIONS:
            cfg = OMIEDataConfig(
                data_path=REAL_DATA_PATH,
                contract_id=contract_id,
                delivery_date=delivery_date,
            )
            id_to_pair = load_omie_agent_metadata(cfg)
            observed_counts[(delivery_date, contract_id)] = len(id_to_pair)

        max_observed = max(observed_counts.values()) if observed_counts else 0

        assert max_agents >= max_observed, (
            f"MAX_AGENTS={max_agents} is smaller than the observed "
            f"distinct-pair count across sampled sessions "
            f"(max_observed={max_observed}; per-session counts="
            f"{observed_counts}). See architect-review §2 and P2-F1 "
            f"Test-first contract."
        )

    @pytest.mark.skipif(
        not _all_sampled_files_present(),
        reason="Not all sampled session files present under data/rawOMIE/.",
    )
    def test_observed_pair_counts_are_positive(self):
        """Sanity check: the sampled sessions are not silently empty.

        If a future data-path change accidentally drops the raw CSV
        fixtures, this guards against MAX_AGENTS's empirical property
        becoming vacuously true (``max_agents >= 0`` is trivially
        satisfied). Requires at least one sampled (date, contract) to
        carry a non-empty pair universe.
        """
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )

        counts = []
        for delivery_date, contract_id in SAMPLED_SESSIONS:
            cfg = OMIEDataConfig(
                data_path=REAL_DATA_PATH,
                contract_id=contract_id,
                delivery_date=delivery_date,
            )
            counts.append(len(load_omie_agent_metadata(cfg)))

        assert max(counts) > 0, (
            "All sampled sessions returned empty pair universes; the "
            "MAX_AGENTS empirical-sufficiency property is vacuous. "
            "Confirm data/rawOMIE/ fixtures are intact."
        )


# ---------------------------------------------------------------------------
# 8. No I/O boundary leakage — import path is under omie/, not jaxomie/
# ---------------------------------------------------------------------------


class TestModulePlacement:
    """Architect review §4 locks placement: ``omiecid_simulator/omiecid/``.

    Placing the constants under ``jaxomie/`` would invert the existing
    dependency direction (``jaxomie → omie`` only, per P2-F0 audit).
    This test pins the correct location.
    """

    def test_module_is_under_omie_namespace(self):
        mod = _import_jax_constants()
        assert mod.__name__ == JAX_CONSTANTS_MODULE, (
            f"jax_constants must live at {JAX_CONSTANTS_MODULE}; "
            f"got {mod.__name__}"
        )
        # Module file path must be under omiecid_simulator/omiecid/, not
        # omiecid_simulator/jaxomiecid/. (This is a belt-and-suspenders check
        # on top of the import-path assertion above.)
        file_path = getattr(mod, "__file__", "") or ""
        assert "omiecid_simulator/omiecid/" in file_path.replace("\\", "/"), (
            f"jax_constants module file must live under "
            f"omiecid_simulator/omiecid/, got: {file_path}"
        )
        assert "omiecid_simulator/jaxomiecid/" not in file_path.replace("\\", "/"), (
            f"jax_constants must NOT live under omiecid_simulator/jaxomiecid/ "
            f"(would invert dependency direction); got: {file_path}"
        )
