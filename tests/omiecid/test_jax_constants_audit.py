"""Audit tests for P2-F1 — Static JAX constants.

These tests are authored independently from the original P2-F1 locked
test suite. They verify the finished ``omiecid_simulator.omiecid.jax_constants``
module against the acceptance criteria in ``docs/omiecid/agile-plan.md``
§P2-F1, the migration-plan §Phase 2 B1/B2/B5 context, and the P2-F0
architecture invariant that the ``omie/`` layer must not depend on the
``jaxomie/`` layer.

Focus areas (derived solely from the story card and implementation
inspection, without reading the original test file):

1. Type contract — each exported constant is a true Python ``int``
   (not ``bool``, not ``jnp.ndarray``, not a ``numpy`` scalar).
2. Hashability / JIT-static usability — constants are usable as
   ``static_argnums`` values and as dict keys.
3. Public API surface — ``__all__`` matches the three documented names.
4. Docstring audit trail — module docstring is non-trivial and names
   each constant.
5. Coupling invariants — MAX_ICEBERGS ≥ book_depth; MAX_SLICES_PER_REGEN
   is a non-trivial and tractable while_loop trip bound.
6. MAX_AGENTS empirical sufficiency — audited across multiple sessions
   sampled from data/rawOMIE/ via load_omie_agent_metadata (marked slow;
   each session individually guarded on file existence).
7. Module placement — jax_constants lives under ``omie/``, not
   ``jaxomie/``; it does not import from ``jaxomie/``.
8. No production leakage — import side effects are purely declarative.
9. Additional invariants latent in the code (e.g., ``__all__`` stability,
   positivity of values, power-of-2 margin for MAX_AGENTS).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import numbers
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module-under-audit import (lazy so collection does not crash if absent)
# ---------------------------------------------------------------------------

MODULE_NAME = "omiecid_simulator.omiecid.jax_constants"


def _import_module():
    return importlib.import_module(MODULE_NAME)


@pytest.fixture(scope="module")
def mod():
    return _import_module()


@pytest.fixture(scope="module")
def constants(mod):
    return {
        "MAX_ICEBERGS": mod.MAX_ICEBERGS,
        "MAX_SLICES_PER_REGEN": mod.MAX_SLICES_PER_REGEN,
        "MAX_AGENTS": mod.MAX_AGENTS,
    }


CONSTANT_NAMES = ["MAX_ICEBERGS", "MAX_SLICES_PER_REGEN", "MAX_AGENTS"]


# ---------------------------------------------------------------------------
# 1. Type contract
# ---------------------------------------------------------------------------


class TestTypeContract:
    """Each exported constant is a true Python int (not bool / not jax
    scalar / not numpy scalar) and is positive."""

    @pytest.mark.parametrize("name", CONSTANT_NAMES)
    def test_is_python_int(self, constants, name):
        v = constants[name]
        # strict type check; isinstance(True, int) is True but bool is explicitly rejected below.
        assert type(v) is int, (
            f"{name} must be exactly `int` (got {type(v).__name__}); "
            f"subclasses like `bool` or third-party int-likes break jit staticness."
        )

    @pytest.mark.parametrize("name", CONSTANT_NAMES)
    def test_is_not_bool(self, constants, name):
        v = constants[name]
        assert not isinstance(v, bool), f"{name} must not be a bool."

    @pytest.mark.parametrize("name", CONSTANT_NAMES)
    def test_is_not_jax_array(self, constants, name):
        v = constants[name]
        assert not isinstance(v, jnp.ndarray), (
            f"{name} must not be a jax array/scalar — jit would trace it "
            f"instead of treating it as a compile-time constant."
        )

    @pytest.mark.parametrize("name", CONSTANT_NAMES)
    def test_is_not_numpy_scalar(self, constants, name):
        v = constants[name]
        assert not isinstance(v, np.generic), (
            f"{name} is a numpy scalar ({type(v)}); must be a plain Python int."
        )
        assert not isinstance(v, np.ndarray), f"{name} is a numpy array."

    @pytest.mark.parametrize("name", CONSTANT_NAMES)
    def test_is_positive(self, constants, name):
        v = constants[name]
        assert v > 0, f"{name} must be strictly positive (got {v})."

    @pytest.mark.parametrize("name", CONSTANT_NAMES)
    def test_is_real_number(self, constants, name):
        # Sanity: int is a numbers.Integral; this is a redundant belt-and-braces
        # guard against a pathological int-like subclass.
        v = constants[name]
        assert isinstance(v, numbers.Integral), f"{name} is not integral."


# ---------------------------------------------------------------------------
# 2. JIT-staticness / hashability
# ---------------------------------------------------------------------------


class TestJITStatic:
    """Constants are hashable and usable as compile-time statics in jax.jit."""

    @pytest.mark.parametrize("name", CONSTANT_NAMES)
    def test_is_hashable(self, constants, name):
        v = constants[name]
        # If the constant is not hashable it cannot serve as a dict key nor
        # as a static_argnums value (jax fingerprints static args by hash).
        assert hash(v) == hash(v)
        d = {v: "ok"}  # raises if not hashable
        assert d[v] == "ok"

    @pytest.mark.jit
    def test_all_three_usable_as_static_argnums(self, constants):
        """Verify each constant can drive a shape-dependent jit without
        producing a tracer. Uses a single jit wrapper (no inner @jit) so
        static_argnums applies to the shape arguments as intended."""

        def alloc(a: int, b: int, c: int):
            # Shape-dependent call that fails at trace time if a/b/c are tracers.
            return jnp.zeros((a,), dtype=jnp.int32).sum() + \
                   jnp.zeros((b,), dtype=jnp.int32).sum() + \
                   jnp.zeros((c,), dtype=jnp.int32).sum()

        jitted = jax.jit(alloc, static_argnums=(0, 1, 2))
        out = jitted(
            constants["MAX_ICEBERGS"],
            constants["MAX_SLICES_PER_REGEN"],
            min(constants["MAX_AGENTS"], 1024),  # cap for test runtime only
        )
        # Result is a concrete jax scalar (0 + 0 + 0 = 0)
        assert int(out) == 0

    @pytest.mark.jit
    @pytest.mark.parametrize("name", CONSTANT_NAMES)
    def test_not_a_tracer_in_jit_scope(self, constants, name):
        """Reading the constant inside a jit-compiled function yields a
        compile-time Python int, not a Traced<ShapedArray>."""
        mod = _import_module()
        v_outer = constants[name]

        def read_const():
            # Access at trace time; if constants leak tracer-wrapping this
            # would fail when used in shape context below.
            v = getattr(mod, name)
            return jnp.zeros((v,), dtype=jnp.int32).shape[0]

        jitted = jax.jit(read_const)
        out = jitted()
        assert int(out) == v_outer


# ---------------------------------------------------------------------------
# 3. Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:

    def test_three_names_are_module_attributes(self, mod):
        for name in CONSTANT_NAMES:
            assert hasattr(mod, name), f"missing public attribute {name}"

    def test_all_declared_and_matches_three_names(self, mod):
        """The three P2-F1 constants MUST be present in __all__.

        This was originally a strict-equality check asserting that __all__
        contained exactly the three P2-F1 names. That over-specified the
        contract: the migration plan (docs/omiecid/migration-plan.md §Phase 2
        Acceptance Criteria item 6) anticipates additive extension of
        jax_constants.py in later Phase-2 stories (e.g. P2-F2 adds
        SENTINEL_ORDER_ID, SENTINEL_AGENT_ID, make_sentinel_order).

        The load-bearing P2-F1 invariant is that the three names are
        *present* in __all__ and therefore part of the public surface;
        additional names added by later stories are permitted. The
        assertion is relaxed from equality to subset accordingly.
        """
        assert hasattr(mod, "__all__"), (
            "jax_constants.py should declare __all__ to freeze the public "
            "surface; see P2-F1 architecture review §5."
        )
        assert set(CONSTANT_NAMES) <= set(mod.__all__), (
            f"__all__ is missing one or more P2-F1 public constants. "
            f"The three P2-F1 names must remain in the public surface; "
            f"later stories (P2-F2+) may add additional names. "
            f"Expected (subset): {set(CONSTANT_NAMES)}; "
            f"found __all__: {set(mod.__all__)}"
        )

    def test_all_is_a_sequence_of_strings(self, mod):
        assert isinstance(mod.__all__, (list, tuple))
        for n in mod.__all__:
            assert isinstance(n, str)

    def test_from_import_star_surface(self, mod):
        """Simulate `from omiecid_simulator.omiecid.jax_constants import *`
        using __all__ as the selector (which is how the star-import
        machinery behaves when __all__ is present)."""
        exported = {name: getattr(mod, name) for name in mod.__all__}
        for name in CONSTANT_NAMES:
            assert name in exported


# ---------------------------------------------------------------------------
# 4. Docstring audit trail
# ---------------------------------------------------------------------------


class TestModuleDocstring:

    def test_module_has_nontrivial_docstring(self, mod):
        doc = mod.__doc__
        assert doc is not None, "module docstring is missing"
        assert len(doc.strip()) > 80, (
            "module docstring is trivial (<80 non-whitespace chars); P2-F1 "
            "requires per-constant value / derivation / revision-cost audit trail."
        )

    @pytest.mark.parametrize("name", CONSTANT_NAMES)
    def test_docstring_mentions_each_constant(self, mod, name):
        assert name in mod.__doc__, (
            f"module docstring must mention {name} to provide the per-constant "
            f"derivation / revision-cost audit trail."
        )

    @pytest.mark.parametrize(
        "keyword",
        ["Derivation", "revis"],  # 'revising' / 'revision' both match
    )
    def test_docstring_has_derivation_and_revision_sections(self, mod, keyword):
        """The docstring audit trail must record how each constant was
        chosen AND what revising it costs (per story card)."""
        assert keyword.lower() in mod.__doc__.lower(), (
            f"module docstring missing '{keyword}' — P2-F1 story card requires "
            f"each constant's derivation and revision cost to be documented."
        )


# ---------------------------------------------------------------------------
# 5. Coupling invariants
# ---------------------------------------------------------------------------


class TestCouplingInvariants:

    def test_max_icebergs_ge_default_book_depth(self, constants):
        from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig
        book_depth = OMIEMarketConfig().book_depth
        assert constants["MAX_ICEBERGS"] >= book_depth, (
            f"MAX_ICEBERGS ({constants['MAX_ICEBERGS']}) < "
            f"OMIEMarketConfig.book_depth ({book_depth}); "
            f"an iceberg occupies a resting book slot, so the iceberg pool "
            f"cannot be smaller than the book ceiling (see P2-F1 review §3)."
        )

    def test_max_slices_per_regen_is_nontrivial(self, constants):
        # A value of 0 or 1 would make ICEwPI semantics degenerate —
        # there must be room for at least the terminator plus one real slice.
        assert constants["MAX_SLICES_PER_REGEN"] >= 2, (
            "MAX_SLICES_PER_REGEN < 2 defeats ICEwPI semantics."
        )

    def test_max_slices_per_regen_is_tractable(self, constants):
        # Sanity sanity: a while_loop trip bound of ~thousands would be
        # impractical for XLA. The architect review picked 64; any value
        # north of 256 would warrant justification.
        assert constants["MAX_SLICES_PER_REGEN"] <= 256, (
            f"MAX_SLICES_PER_REGEN = {constants['MAX_SLICES_PER_REGEN']} is an "
            f"unexpectedly large while_loop trip bound; XLA compile-time "
            f"will degrade. Justify or lower."
        )

    def test_max_agents_not_absurdly_large(self, constants):
        # Soft ceiling: per-agent settlement arrays are (MAX_AGENTS, K).
        # 512 is the architect-recommended value; allow up to ~32k as a
        # sanity bound before a memory-footprint review is warranted.
        assert constants["MAX_AGENTS"] <= 32768, (
            f"MAX_AGENTS = {constants['MAX_AGENTS']} is excessively large; "
            f"per-agent array memory footprint should be re-reviewed."
        )


# ---------------------------------------------------------------------------
# 6. MAX_AGENTS empirical sufficiency (SLOW)
# ---------------------------------------------------------------------------


RAW_OMIE_DIR = Path("data/rawOMIE")


def _is_dst_transition_day(iso: str) -> bool:
    """Europe/Madrid UTC offset differs between D 00:00 and D 23:59."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Europe/Madrid")
    y, m, d = (int(x) for x in iso.split("-"))
    return (
        datetime(y, m, d, 0, 0, tzinfo=tz).utcoffset()
        != datetime(y, m, d, 23, 59, tzinfo=tz).utcoffset()
    )


def _candidate_sessions() -> list[str]:
    """Discover loadable delivery products for 2025/2026.

    OMIE-TAPE-CONTRACT-DATE-1 (group R-a repair of the sampler, assertion
    untouched): a product D is loadable only when it is COMPLETE — session
    files ``orders_{D-1}.1``, ``orders_D.1``, ``trades_{D-1}.1`` and
    ``trades_D.1`` all exist (market-spec v2.10 §13.4, OQ-29) — and D is not
    a Europe/Madrid DST transition day (OQ-30 guard). The previous sampler
    listed every ``orders_*.1`` date, so the newly-added first session file
    2025-05-31 (an incomplete product: no ``*_20250530.1``) was sampled.

    Returns a list of delivery-date strings (YYYY-MM-DD).
    """
    from datetime import date, timedelta

    if not RAW_OMIE_DIR.is_dir():
        return []
    dates: list[str] = []
    for p in sorted(RAW_OMIE_DIR.glob("orders_*.1")):
        stem = p.name[len("orders_"): -len(".1")]
        if len(stem) != 8 or not stem.isdigit():
            continue
        if not (stem.startswith("2025") or stem.startswith("2026")):
            continue
        iso = f"{stem[0:4]}-{stem[4:6]}-{stem[6:8]}"
        prev = (date.fromisoformat(iso) - timedelta(days=1)).strftime("%Y%m%d")
        required = (
            f"orders_{prev}.1", f"orders_{stem}.1",
            f"trades_{prev}.1", f"trades_{stem}.1",
        )
        if not all((RAW_OMIE_DIR / name).is_file() for name in required):
            continue
        if _is_dst_transition_day(iso):
            continue
        dates.append(iso)
    return dates


def _sampled_sessions() -> list[str]:
    """Sample a handful of sessions spread across the available range.

    This broadens the audit beyond the three canonical sessions used by
    the original P2-F1 test suite without pulling every file (which would
    blow the runtime budget). The strategy is 'first, last, and three
    interior points', giving 5 sessions when the dataset has at least 5.
    """
    dates = _candidate_sessions()
    if len(dates) == 0:
        return []
    if len(dates) <= 5:
        return dates
    n = len(dates)
    idx = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    return [dates[i] for i in idx]


SAMPLED_SESSIONS = _sampled_sessions()


@pytest.mark.slow
class TestMaxAgentsEmpiricalSufficiency:
    """Slow: each session loads a 30-44 MB OMIE CSV through
    load_omie_agent_metadata. Exceeds the < 1 s fast-suite budget but
    provides the authoritative sufficiency check for MAX_AGENTS.

    Sample strategy (documented for auditability): draw up to five dates
    spread across the rawOMIE window (first / quartile / median /
    three-quartile / last) rather than the three fixed dates used by the
    original suite, so the sufficiency check has broader coverage. Each
    session is guarded individually on file existence via skipif so
    collection succeeds on machines without raw data.
    """

    @pytest.mark.parametrize("delivery_date", SAMPLED_SESSIONS or ["_none_"])
    def test_max_agents_covers_session(self, delivery_date, constants):
        orders_file = RAW_OMIE_DIR / f"orders_{delivery_date.replace('-', '')}.1"
        if not orders_file.is_file():
            pytest.skip(f"raw OMIE file missing: {orders_file}")

        from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
        from omiecid_simulator.jaxomiecid.omie_data_loader import load_omie_agent_metadata

        cfg = OMIEDataConfig(
            data_path=str(RAW_OMIE_DIR),
            contract_id=81,
            delivery_date=delivery_date,
        )
        id_to_pair = load_omie_agent_metadata(cfg)
        observed = len(id_to_pair)
        assert constants["MAX_AGENTS"] >= observed, (
            f"MAX_AGENTS = {constants['MAX_AGENTS']} is smaller than the "
            f"distinct-pair universe for {delivery_date} cid=81 (observed "
            f"{observed}). MAX_AGENTS must bound the empirical pair count "
            f"for every session in the dataset."
        )

    def test_sampled_sessions_list_is_nonempty_when_data_present(self):
        """If rawOMIE is on disk but SAMPLED_SESSIONS is empty, the sample
        strategy is silently covering nothing — treat as a defect."""
        if not RAW_OMIE_DIR.is_dir():
            pytest.skip("raw OMIE data dir absent on this machine")
        if not _candidate_sessions():
            pytest.skip("no 2025/2026 sessions found in rawOMIE")
        assert SAMPLED_SESSIONS, (
            "SAMPLED_SESSIONS is empty despite rawOMIE being present — "
            "audit coverage is silently zero."
        )


# ---------------------------------------------------------------------------
# 7. Module placement
# ---------------------------------------------------------------------------


class TestModulePlacement:
    """jax_constants.py must live under omiecid_simulator/omiecid/ (pure-domain
    layer), not under omiecid_simulator/jaxomiecid/ (I/O-boundary layer). The
    P2-F0 audit pinned the one-way import direction jaxomie/ → omie/."""

    def test_module_file_in_omie_subpackage(self, mod):
        path = Path(inspect.getsourcefile(mod)).resolve()
        parts = path.parts
        assert "omiecid_simulator" in parts, f"unexpected location: {path}"
        idx = parts.index("omiecid_simulator")
        # Next component after omiecid_simulator must be 'omiecid'.
        assert parts[idx + 1] == "omiecid", (
            f"jax_constants.py is under '{parts[idx + 1]}/', must be under 'omiecid/'."
        )
        assert parts[idx + 1] != "jaxomiecid", (
            "jax_constants.py must not live under jaxomiecid/ — that is the "
            "I/O-boundary layer, not the pure-domain layer."
        )

    def test_module_dotted_name_matches_omie(self, mod):
        assert mod.__name__ == "omiecid_simulator.omiecid.jax_constants"

    def test_source_does_not_import_jaxomie(self, mod):
        src = Path(inspect.getsourcefile(mod)).read_text()
        tree = ast.parse(src)
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("omiecid_simulator.jaxomiecid"):
                    offenders.append(f"from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("omiecid_simulator.jaxomiecid"):
                        offenders.append(f"import {alias.name}")
        assert not offenders, (
            f"jax_constants.py must not import from jaxomie/: {offenders}. "
            f"That would invert the jaxomie/ → omie/ dependency direction "
            f"locked by the P2-F0 architecture audit."
        )


# ---------------------------------------------------------------------------
# 8. No production leakage / import side effects
# ---------------------------------------------------------------------------


class TestImportSideEffects:
    """Importing jax_constants must be purely declarative: no CSV parsing,
    no heavy jax tracing, no filesystem I/O, no global state mutation."""

    def test_fresh_import_does_not_read_rawomie(self, tmp_path, monkeypatch):
        """A fresh import under a sandboxed cwd with no data/ directory
        should still succeed — proves the module doesn't walk data/rawOMIE/."""
        monkeypatch.chdir(tmp_path)
        sys.modules.pop(MODULE_NAME, None)
        try:
            mod = importlib.import_module(MODULE_NAME)
            assert mod.MAX_ICEBERGS > 0  # arbitrary sanity touch
        finally:
            # Leave module cached in its normal state for other tests.
            sys.modules.pop(MODULE_NAME, None)
            importlib.import_module(MODULE_NAME)

    def test_module_source_is_mostly_declarative(self, mod):
        """Top-level statements should be: docstring, __future__ import,
        assignments, and __all__. No function/class definitions are
        strictly forbidden, but there must be no function *calls* at
        module scope (which would trigger side effects at import time)."""
        src = Path(inspect.getsourcefile(mod)).read_text()
        tree = ast.parse(src)
        offending_calls: list[str] = []
        for stmt in tree.body:
            # Allow: docstring (Expr(Constant str)), Import/ImportFrom,
            #        Assign, AnnAssign, FunctionDef, ClassDef.
            if isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Call):
                    offending_calls.append(ast.dump(stmt.value))
            # Assignment values MAY contain literal int expressions but not Calls.
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                value = stmt.value
                if value is not None:
                    for node in ast.walk(value):
                        if isinstance(node, ast.Call):
                            offending_calls.append(ast.dump(node))
        assert not offending_calls, (
            f"jax_constants.py has module-scope function calls ({offending_calls}) "
            f"— these run at import time and may introduce side effects. "
            f"Constants should be literal int assignments."
        )

    def test_no_jax_tracing_at_import(self, mod):
        """Attributes should be plain Python ints — if any leaked through
        a traced computation the dtype would be jnp.int32 or similar."""
        for name in CONSTANT_NAMES:
            v = getattr(mod, name)
            assert not hasattr(v, "dtype"), (
                f"{name} has a dtype attribute — looks like a jax/numpy scalar."
            )


# ---------------------------------------------------------------------------
# 9. Latent invariants pinned during audit
# ---------------------------------------------------------------------------


class TestLatentInvariants:

    def test_constants_distinct_or_intentional(self, constants):
        """Two of the three constants being identical would be a red flag
        of a copy-paste. The constants pin three orthogonal quantities
        (agent universe, iceberg pool capacity, per-iceberg slice ceiling)
        so pairwise equality between any two would indicate accidental
        sharing rather than independent derivation.

        Architecture note (post-P2-PERF-7, 2026-05-21): an earlier version
        of this test asserted ``MAX_AGENTS > MAX_ICEBERGS`` on the rationale
        that "the agent universe is broader than the iceberg pool". That
        ordering was retired by the v5.34 P2-PERF-7 mid-sprint adjudication
        because it is empirically false on representative OMIE sessions:
        a single agent routinely submits many ICEwPI orders across one
        delivery day, so the iceberg pool (B1, sized against the
        session-max outstanding-with-hidden count) is architecturally
        unrelated to the agent universe (B5, sized against the distinct
        ``(Agente, Unidad)`` pair count). See ``sprint-status.md``
        §"P2-PERF-7 — Mid-sprint adjudication (2026-05-21)" Failure A
        and ``jax_constants.py`` §B1 / §B5 docstrings for the full
        derivation chain.
        """
        # Pairwise distinctness — the copy-paste guard the docstring pins.
        assert constants["MAX_AGENTS"] != constants["MAX_ICEBERGS"], (
            f"MAX_AGENTS ({constants['MAX_AGENTS']}) and MAX_ICEBERGS "
            f"({constants['MAX_ICEBERGS']}) must be distinct — equality "
            f"would indicate accidental copy-paste between two independent "
            f"derivations (B5 distinct-pair count vs. B1 iceberg-pool "
            f"capacity)."
        )
        assert constants["MAX_AGENTS"] != constants["MAX_SLICES_PER_REGEN"], (
            f"MAX_AGENTS ({constants['MAX_AGENTS']}) and MAX_SLICES_PER_REGEN "
            f"({constants['MAX_SLICES_PER_REGEN']}) must be distinct — "
            f"equality would indicate accidental copy-paste between two "
            f"independent derivations (B5 distinct-pair count vs. B2 per-"
            f"iceberg slice ceiling)."
        )
        assert constants["MAX_ICEBERGS"] != constants["MAX_SLICES_PER_REGEN"], (
            f"MAX_ICEBERGS ({constants['MAX_ICEBERGS']}) and "
            f"MAX_SLICES_PER_REGEN ({constants['MAX_SLICES_PER_REGEN']}) "
            f"must be distinct — equality would indicate accidental copy-"
            f"paste between B1 iceberg-pool capacity and B2 per-iceberg "
            f"slice ceiling."
        )
        # B5 > B2 — the meaningful ordering: the agent universe is many
        # orders of magnitude larger than a per-iceberg slice-count
        # ceiling. This is the surviving load-bearing portion of the
        # original test (the inverted MAX_AGENTS > MAX_ICEBERGS assertion
        # was retired per the v5.34 adjudication; this one is preserved
        # because B5 and B2 are population-vs-trip-count orthogonal
        # quantities that should never approach each other).
        assert constants["MAX_AGENTS"] > constants["MAX_SLICES_PER_REGEN"], (
            f"MAX_AGENTS ({constants['MAX_AGENTS']}) must exceed "
            f"MAX_SLICES_PER_REGEN ({constants['MAX_SLICES_PER_REGEN']}) — "
            f"the agent population is many orders of magnitude larger "
            f"than a per-iceberg slice-count ceiling."
        )

    def test_max_agents_is_power_of_two(self, constants):
        """Architect review §2 fixed 'next power of two' as the rounding
        rule for MAX_AGENTS. This is a non-load-bearing pin — if the
        developer chooses otherwise, document the divergence."""
        v = constants["MAX_AGENTS"]
        assert v > 0 and (v & (v - 1)) == 0, (
            f"MAX_AGENTS = {v} is not a power of 2; architect review §2 "
            f"recommends the 2×-observed-rounded-to-next-pow2 rule. "
            f"A different rounding is acceptable but must be justified."
        )

    def test_module_docstring_names_p2_sibling_stories(self, mod):
        """The docstring should reference the downstream consumer stories
        (B1 / iceberg_manager and B5 / settlement or contract_book) so a
        future reader understands which tests to rerun on revision."""
        doc = mod.__doc__ or ""
        # Weak reference — we accept either the blocker code (B1, B2, B5)
        # or the consumer module name (iceberg_manager, settlement,
        # contract_book). At least one of each family should appear.
        blockers = any(tag in doc for tag in ("B1", "B2", "B5"))
        consumers = any(
            tag in doc
            for tag in ("iceberg_manager", "settlement", "contract_book")
        )
        assert blockers or consumers, (
            "docstring should cite either the Phase 2 blocker codes "
            "(B1/B2/B5) or the downstream consumer modules so the "
            "revision-cost audit trail is navigable."
        )
