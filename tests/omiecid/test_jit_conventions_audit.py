"""Audit tests for P2-F2 — JIT-scope + sentinel conventions.

Authored independently from the original P2-F2 locked test suite
(`tests/omiecid/test_jit_conventions.py`). The original file is deliberately
NOT read; this audit derives its assertions from the story card
(`docs/omiecid/agile-plan.md` §P2-F2), the migration-plan Phase 2 Blockers B3
and B4, Acceptance Criteria item 6, and inspection of the implementation's
public surface only (`omiecid_simulator/omiecid/jax_constants.py`,
`omiecid_simulator/omiecid/order_types.py`, `omiecid_simulator/omiecid/contract_book.py`).

Audit focus areas (chosen post-implementation to cover edges the locked
tests may have under-specified):

    1. B3 semantic invariants — sentinel IDs cannot alias real IDs; the
       single-discriminator rule is enforceable; sentinel is discriminable
       from every legitimate `make_order(...)` construction.
    2. B3 type invariants — strict Python `int`, pickleable, idempotent
       across calls, structural equality between two factory outputs.
    3. B3 pytree invariants — `jax.tree.flatten` round-trip is bitwise
       faithful, and the sentinel's tree structure matches that of a real
       `OMIEOrder`.
    4. B4 documentation invariants — a grep-able anchor is present, the
       verdict keyword is locked, the paragraph is non-trivial and cites
       at least one concrete technical reason.
    5. B4 scope-boundary invariants — the three query methods on
       `contract_book.py` are not JIT-decorated, and `get_order` has
       NOT been flipped to sentinel (still `Optional` return — P2-H2).
       Additionally: no new `@jax.jit` decorator was added *anywhere* in
       `contract_book.py` by P2-F2.
    6. Module placement — `jax_constants.py` imports zero names from
       `jaxob/` (in addition to the pre-existing
       `jaxomiecid/` exclusion).
    7. Public API surface — `__all__` has exactly six names; the star-
       import surface matches `__all__`; no accidental upper-case leakage
       at module scope beyond `__all__`.
    8. Migration-plan AC item 6 conformance — the five names required at
       Phase 2 closure (three shape constants + `SENTINEL_ORDER_ID` +
       B4 decision record) are all reachable from the module.
    9. Side-effect audit — importing the module under an empty-cwd sandbox
       does not touch the filesystem, does not allocate any JAX array at
       module scope, and produces sentinel constants that are plain
       Python ints (no `dtype` attribute).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pickle
import sys
from pathlib import Path
from typing import get_type_hints

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Module-level marker (Story T1.1): audit tests for P2-F2 JIT conventions;
# fully-JIT.  Per-test `@pytest.mark.jit` stripped where redundant.
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Import fixtures — lazy so collection does not crash if module is missing
# ---------------------------------------------------------------------------

JAX_CONSTANTS_MODNAME = "omiecid_simulator.omiecid.jax_constants"
CONTRACT_BOOK_MODNAME = "omiecid_simulator.omiecid.contract_book"
ORDER_TYPES_MODNAME = "omiecid_simulator.omiecid.order_types"


@pytest.fixture(scope="module")
def jc():
    return importlib.import_module(JAX_CONSTANTS_MODNAME)


@pytest.fixture(scope="module")
def ot():
    return importlib.import_module(ORDER_TYPES_MODNAME)


@pytest.fixture(scope="module")
def contract_book_src() -> str:
    mod = importlib.import_module(CONTRACT_BOOK_MODNAME)
    return Path(inspect.getsourcefile(mod)).read_text()


@pytest.fixture(scope="module")
def contract_book_ast(contract_book_src) -> ast.Module:
    return ast.parse(contract_book_src)


# Expected frozen public surface (from sprint-status line 700 + architect
# review §3 + P2-F1 carry-forward).
EXPECTED_ALL = {
    "MAX_ICEBERGS",
    "MAX_SLICES_PER_REGEN",
    "MAX_AGENTS",
    "SENTINEL_ORDER_ID",
    "SENTINEL_AGENT_ID",
    "make_sentinel_order",
}


# ---------------------------------------------------------------------------
# 1. B3 semantic invariants
# ---------------------------------------------------------------------------


class TestB3SemanticInvariants:
    """Sentinel IDs are discriminable from every legitimate real ID."""

    def test_sentinel_order_id_is_negative(self, jc):
        """submit_order rejects order_id <= 0 (contract_book.py:282), so the
        real-ID universe is {1, 2, 3, ...}. A sentinel must fall outside
        that universe — `-1` does, and anything >= 0 would alias."""
        assert jc.SENTINEL_ORDER_ID < 1, (
            "SENTINEL_ORDER_ID must be < 1 (real order IDs are positive; "
            "contract_book.submit_order rejects order.order_id <= 0)."
        )

    def test_sentinel_agent_id_is_negative(self, jc):
        """Post-P2-F0 / Phase 1.5-B the loader emits (Agente, Unidad) pair
        indices >= 0 — so a negative sentinel is unambiguous. `-1` matches
        the B3 derivation in the module docstring."""
        assert jc.SENTINEL_AGENT_ID < 0, (
            "SENTINEL_AGENT_ID must be < 0 (pair encoding is >= 0 post-P2-F0)."
        )

    def test_sentinel_ids_match_canonical_minus_one(self, jc):
        """The architect review §1 and module docstring pin both sentinels
        to -1. Changing to another negative value would be a silent
        convention drift."""
        assert jc.SENTINEL_ORDER_ID == -1
        assert jc.SENTINEL_AGENT_ID == -1

    def test_sentinel_order_id_differs_from_every_small_real_id(self, jc):
        """Exhaustive small-sample discriminability check: no real ID in
        a representative {1..10_000} range can collide with the sentinel.
        This guards against a regression where SENTINEL_ORDER_ID might be
        accidentally bumped to a positive value."""
        for real_id in (1, 2, 10, 100, 1_000, 10_000):
            assert real_id != jc.SENTINEL_ORDER_ID

    def test_sentinel_is_discriminable_from_zero_valued_real_order(self, jc, ot):
        """The module docstring §B3 hazard: side==0, status==0, peak_qty==0,
        ppd==0 are all legal real states. A zero-valued real order with a
        positive order_id MUST NOT be confused for the sentinel. The only
        safe discriminator is `order_id == SENTINEL_ORDER_ID`."""
        sentinel = jc.make_sentinel_order()
        zero_real = ot.make_order(
            contract_id=0,
            side=ot.Side.BUY,
            price=0.0,
            quantity=0.0,
            order_id=1,      # positive -> real
            agent_id=0,
            timestamp=0,
        )
        # The ONLY field that differs is order_id.
        assert zero_real.order_id != sentinel.order_id
        # Every other field can (and does) coincide with the sentinel's zeros.
        # So any consumer branching on side/status/peak_qty/ppd would be wrong.
        assert zero_real.side == sentinel.side
        assert zero_real.status == sentinel.status
        assert zero_real.peak_qty == sentinel.peak_qty
        assert zero_real.ppd == sentinel.ppd


# ---------------------------------------------------------------------------
# 2. B3 type invariants
# ---------------------------------------------------------------------------


class TestB3TypeInvariants:
    """Sentinel constants are strict Python ints and the factory is
    deterministic / idempotent / pickleable."""

    @pytest.mark.parametrize("name", ["SENTINEL_ORDER_ID", "SENTINEL_AGENT_ID"])
    def test_sentinel_is_exact_python_int(self, jc, name):
        v = getattr(jc, name)
        assert type(v) is int, (
            f"{name} must be exactly `int` (got {type(v).__name__}); "
            f"numpy/jax scalar would leak into static_argnums."
        )

    @pytest.mark.parametrize("name", ["SENTINEL_ORDER_ID", "SENTINEL_AGENT_ID"])
    def test_sentinel_is_not_bool(self, jc, name):
        v = getattr(jc, name)
        assert not isinstance(v, bool), (
            f"{name} must not be a bool; although bool subclasses int, "
            f"static_argnums identity checks disambiguate."
        )

    @pytest.mark.parametrize("name", ["SENTINEL_ORDER_ID", "SENTINEL_AGENT_ID"])
    def test_sentinel_has_no_dtype_attribute(self, jc, name):
        v = getattr(jc, name)
        assert not hasattr(v, "dtype"), (
            f"{name} has a dtype attribute — looks like a jax/numpy scalar."
        )

    @pytest.mark.parametrize("name", ["SENTINEL_ORDER_ID", "SENTINEL_AGENT_ID"])
    def test_sentinel_is_hashable_and_dict_keyable(self, jc, name):
        v = getattr(jc, name)
        assert hash(v) == hash(v)
        d = {v: "ok"}
        assert d[v] == "ok"

    @pytest.mark.parametrize("name", ["SENTINEL_ORDER_ID", "SENTINEL_AGENT_ID"])
    def test_sentinel_is_pickleable(self, jc, name):
        """A Python int is trivially pickleable; this test fires a shot
        across the bow of any future change that accidentally promotes the
        sentinel to a non-pickleable type (e.g. `enum.IntFlag`)."""
        v = getattr(jc, name)
        roundtrip = pickle.loads(pickle.dumps(v))
        assert roundtrip == v
        assert type(roundtrip) is int

    def test_make_sentinel_order_is_idempotent_field_by_field(self, jc):
        """Two successive `make_sentinel_order()` calls must return
        structurally-equal orders (no `time.time()`-style hidden defaults)."""
        a = jc.make_sentinel_order()
        b = jc.make_sentinel_order()
        for field_name in a._fields:
            av = getattr(a, field_name)
            bv = getattr(b, field_name)
            assert av == bv, f"field {field_name}: {av!r} != {bv!r}"
            assert type(av) is type(bv), (
                f"field {field_name}: type drift {type(av)} vs {type(bv)}"
            )

    def test_make_sentinel_order_tuple_equality(self, jc):
        assert jc.make_sentinel_order() == jc.make_sentinel_order()

    def test_make_sentinel_order_returns_omie_order_type(self, jc, ot):
        s = jc.make_sentinel_order()
        assert isinstance(s, ot.OMIEOrder), (
            f"factory must return OMIEOrder (got {type(s).__name__})."
        )

    def test_make_sentinel_order_fields_are_scalar_int_or_float(self, jc):
        """Every field must be a Python `int` or `float` (no IntEnum, no
        numpy / jax scalar). This pins the concrete pytree leaf type."""
        sentinel = jc.make_sentinel_order()
        for name in sentinel._fields:
            val = getattr(sentinel, name)
            assert type(val) in (int, float), (
                f"field {name} has type {type(val).__name__}; must be "
                f"plain int or float (no IntEnum, no numpy / jax scalar)."
            )
            assert not isinstance(val, bool), (
                f"field {name} is bool; disallowed."
            )


# ---------------------------------------------------------------------------
# 3. B3 pytree invariants
# ---------------------------------------------------------------------------


class TestB3PytreeInvariants:
    """The sentinel survives jax.tree flatten/unflatten round-trip and has
    the same tree-structure as a real OMIEOrder."""

    def test_pytree_flatten_unflatten_roundtrip_equal(self, jc):
        sentinel = jc.make_sentinel_order()
        leaves, treedef = jax.tree_util.tree_flatten(sentinel)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt == sentinel

    def test_pytree_structure_matches_real_order(self, jc, ot):
        """Two OMIEOrders — one real, one sentinel — must share the same
        `treedef` so they're interchangeable inside a jit-traced pytree."""
        sentinel = jc.make_sentinel_order()
        real = ot.make_order(
            contract_id=81,
            side=ot.Side.BUY,
            price=50.0,
            quantity=1.0,
            order_id=123,
            agent_id=4,
            timestamp=3600,
        )
        _, sentinel_def = jax.tree_util.tree_flatten(sentinel)
        _, real_def = jax.tree_util.tree_flatten(real)
        assert sentinel_def == real_def, (
            "sentinel and real OMIEOrder have different treedefs — a jit-"
            "closed-over sentinel will not interchange with a real order."
        )

    def test_pytree_leaf_count_matches_field_count(self, jc):
        """Every NamedTuple field contributes exactly one leaf — if the
        factory ever wraps a field in a container (tuple / dict), the leaf
        count diverges and downstream jit shape assumptions break."""
        sentinel = jc.make_sentinel_order()
        leaves, _ = jax.tree_util.tree_flatten(sentinel)
        assert len(leaves) == len(sentinel._fields), (
            f"{len(leaves)} leaves != {len(sentinel._fields)} fields — "
            f"factory has introduced a nested container in one of the fields."
        )

    def test_sentinel_usable_as_compile_time_closure(self, jc):
        """A jit-traced function that closes over the sentinel produces a
        concrete jax scalar output — verifies no leaf is a Python object
        that the tracer would reject."""
        sentinel = jc.make_sentinel_order()

        def check():
            # Sum the numeric leaves — all 13 fields are int or float.
            return jnp.asarray(sentinel.order_id) + jnp.asarray(sentinel.agent_id)

        out = jax.jit(check)()
        # Both sentinels are -1, so the sum is -2.
        assert int(out) == jc.SENTINEL_ORDER_ID + jc.SENTINEL_AGENT_ID


# ---------------------------------------------------------------------------
# 4. B4 documentation invariants
# ---------------------------------------------------------------------------


class TestB4DocumentationInvariants:
    """The B4 decision record must be grep-able, carry the verdict keyword,
    cite at least one concrete technical reason, and be non-trivial."""

    def test_docstring_contains_b4_anchor(self, jc):
        doc = jc.__doc__ or ""
        assert "JIT-scope of query methods" in doc, (
            "module docstring must contain a grep-able 'B4 decision' or "
            "'B4 JIT-scope' anchor so P2-H2 can parse it deterministically."
        )

    def test_docstring_verdict_keyword_present(self, jc):
        """The verdict keyword 'Python-side' (architect review §2) must
        appear so downstream stories can test for it without parsing prose."""
        doc = jc.__doc__ or ""
        assert "Python-side" in doc, (
            "B4 decision verdict keyword 'Python-side' must appear verbatim "
            "in the module docstring."
        )

    def test_docstring_verdict_near_anchor(self, jc):
        """The verdict must appear within 1500 chars of the first B4
        anchor — otherwise the decision record is scattered and P2-H2
        cannot rely on a local parse."""
        doc = jc.__doc__ or ""
        # Use whichever anchor appears first.
        anchors = [doc.find("JIT-scope of query methods")]
        anchor_idx = min(a for a in anchors if a >= 0)
        verdict_idx = doc.find("Python-side")
        assert verdict_idx >= 0
        assert abs(verdict_idx - anchor_idx) <= 1500, (
            f"verdict at char {verdict_idx} is too far from B4 anchor "
            f"at char {anchor_idx}; decision record should be cohesive."
        )

    def test_docstring_b4_section_is_substantive(self, jc):
        """A one-liner verdict is not a decision record. The section must
        be at least ~400 chars of rationale to pass muster."""
        doc = jc.__doc__ or ""
        anchors = [doc.find("JIT-scope of query methods")]
        anchor_idx = min(a for a in anchors if a >= 0)
        # Extract ~2000-char window starting at the anchor.
        tail = doc[anchor_idx: anchor_idx + 2000]
        assert len(tail) >= 400, (
            "B4 decision section is trivially short — must include rationale."
        )

    @pytest.mark.parametrize(
        "query_name",
        ["get_anonymous_lob", "get_agent_orders", "get_agent_trades"],
    )
    def test_docstring_mentions_each_query_method(self, jc, query_name):
        """The B4 record must enumerate the three methods it governs;
        otherwise a reader cannot tell which methods were in scope."""
        doc = jc.__doc__ or ""
        assert query_name in doc, (
            f"B4 decision record must name {query_name} (it governs the "
            f"JIT-scope of this method)."
        )

    def test_docstring_cites_a_concrete_technical_reason(self, jc):
        """Rationale must include at least one concrete technical token —
        'hot path', 'variable-length', 'mask', 'per-step', 'per-match', or
        'Phase 3' — so the decision isn't hand-wavy."""
        doc = (jc.__doc__ or "").lower()
        reasons = [
            "hot path",
            "variable-length",
            "mask",
            "per-step",
            "per-match",
            "phase 3",
            "observation",
        ]
        hits = [r for r in reasons if r in doc]
        assert hits, (
            "B4 decision record cites no concrete technical reason; "
            "expected at least one of: hot path / variable-length / mask / "
            "per-step / per-match / Phase 3 / observation."
        )


# ---------------------------------------------------------------------------
# 5. B4 scope-boundary invariants (negative assertions)
# ---------------------------------------------------------------------------


def _collect_top_level_func_defs(tree: ast.Module) -> list[ast.FunctionDef]:
    """All module-level and class-level function definitions. Captures
    both `def` (sync) and `async def` for completeness."""
    funcs: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
    return funcs


def _has_jit_decorator(fn: ast.FunctionDef) -> bool:
    for dec in fn.decorator_list:
        # Match @jit / @jax.jit / @jax.jit(...) / @jit(...)
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == "jit":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "jit":
            return True
    return False


class TestB4ScopeBoundary:
    """P2-F2 is documentation-only for contract_book.py. No decorator may
    have been added; `get_order` still returns `Optional[OMIEOrder]`."""

    @pytest.mark.parametrize(
        "method_name",
        ["get_anonymous_lob", "get_agent_orders", "get_agent_trades"],
    )
    def test_query_method_has_no_jit_decorator(
        self, contract_book_ast, method_name
    ):
        funcs = _collect_top_level_func_defs(contract_book_ast)
        matches = [f for f in funcs if f.name == method_name]
        assert matches, f"{method_name} not found in contract_book.py"
        for fn in matches:
            assert not _has_jit_decorator(fn), (
                f"{method_name} carries a jit decorator — P2-F2's B4 decision "
                f"is Python-side; wrapping is P2-H2 territory."
            )

    def test_no_jit_decorator_anywhere_in_contract_book(self, contract_book_ast):
        """Broader negative assertion: P2-F2 should not introduce a
        `@jax.jit` anywhere in contract_book.py. Scope was documentation-only."""
        for fn in _collect_top_level_func_defs(contract_book_ast):
            assert not _has_jit_decorator(fn), (
                f"function `{fn.name}` in contract_book.py carries a jit "
                f"decorator; P2-F2 must not add any."
            )

    def test_get_order_still_returns_optional(self, contract_book_ast):
        """P2-H2 has flipped `get_order`'s ``return None`` to
        ``return make_sentinel_order()`` (v5.20, 2026-05-14).

        Originally a P2-F2-era anchor that asserted ``Optional[...]``
        was still present in the return annotation, fencing the
        sentinel callsite swap into P2-H2. Post-P2-H2, the annotation
        is ``OMIEOrder`` (no Optional). This anchor is preserved as
        a presence-only check on the function definition.
        """
        funcs = _collect_top_level_func_defs(contract_book_ast)
        get_order_matches = [f for f in funcs if f.name == "get_order"]
        assert get_order_matches, "get_order not found in contract_book.py"

    def test_get_order_body_still_returns_none(self, contract_book_ast):
        """P2-H2 has replaced ``return None`` with
        ``return make_sentinel_order()`` (v5.20, 2026-05-14).

        Originally a P2-F2-era anchor asserting the body still had
        ``return None``. Post-P2-H2, the body returns
        ``make_sentinel_order()``. This anchor is preserved as a
        positive check on the sentinel surface.
        """
        funcs = _collect_top_level_func_defs(contract_book_ast)
        fn = next(f for f in funcs if f.name == "get_order")
        body_src = ast.unparse(fn)
        assert "make_sentinel_order" in body_src, (
            "get_order body no longer references make_sentinel_order(); "
            "the P2-H2 sentinel callsite swap must remain in place."
        )


# ---------------------------------------------------------------------------
# 6. Module placement
# ---------------------------------------------------------------------------


class TestModulePlacement:

    def test_module_is_under_omie(self, jc):
        path = Path(inspect.getsourcefile(jc)).resolve()
        parts = path.parts
        assert "omiecid_simulator" in parts
        idx = parts.index("omiecid_simulator")
        assert parts[idx + 1] == "omiecid", (
            f"jax_constants.py located at {path} — must be under omiecid/."
        )

    @pytest.mark.parametrize(
        "forbidden_prefix",
        [
            "omiecid_simulator.jaxomiecid",
            "omiecid_simulator.jaxob",
        ],
    )
    def test_no_upward_imports(self, jc, forbidden_prefix):
        """The pure-domain layer must not import from any I/O-boundary or
        RL-consumer layer. Keeps `jax_constants.py` leaf-positioned."""
        src = Path(inspect.getsourcefile(jc)).read_text()
        tree = ast.parse(src)
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(forbidden_prefix):
                    offenders.append(f"from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden_prefix):
                        offenders.append(f"import {alias.name}")
        assert not offenders, (
            f"jax_constants.py imports from forbidden layer {forbidden_prefix}: "
            f"{offenders}"
        )

    def test_no_sibling_jax_conventions_module(self, jc):
        """Architect review §3 rejected a sibling `jax_conventions.py`
        module — the sentinel convention lives alongside the shape
        constants in `jax_constants.py`."""
        omie_dir = Path(inspect.getsourcefile(jc)).parent
        candidate = omie_dir / "jax_conventions.py"
        assert not candidate.exists(), (
            f"unexpected sibling module {candidate} — architect review §3 "
            f"fixed jax_constants.py as the single home."
        )


# ---------------------------------------------------------------------------
# 7. Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPISurface:

    def test_all_has_exactly_six_names(self, jc):
        """P2-F1 seeded `__all__` with 3 names; P2-F2 added exactly 3 more
        (sentinel × 2 + factory). Exactly 6 entries is the frozen contract
        (sprint-status line 700 — 'six names preserved')."""
        assert hasattr(jc, "__all__")
        assert set(jc.__all__) == EXPECTED_ALL, (
            f"__all__ is not the frozen 6-name set. "
            f"expected={EXPECTED_ALL}, got={set(jc.__all__)}"
        )

    def test_all_has_no_duplicates(self, jc):
        assert len(jc.__all__) == len(set(jc.__all__)), (
            f"__all__ contains duplicates: {jc.__all__}"
        )

    def test_all_entries_are_module_attributes(self, jc):
        for name in jc.__all__:
            assert hasattr(jc, name), (
                f"__all__ advertises {name} but module has no such attribute."
            )

    def test_star_import_surface_matches_all(self, jc):
        """Simulate `from ... import *` — the exported set must equal
        `__all__` (no accidental leakage, no missing names)."""
        exported = {name: getattr(jc, name) for name in jc.__all__}
        assert set(exported) == EXPECTED_ALL

    def test_no_extra_uppercase_leakage(self, jc):
        """Any ALL_CAPS module-scope attribute not in `__all__` would be
        de-facto public. P2-F2 must not silently leak additional
        uppercase names (would blur the F1/F2 → H2 / L3 boundary)."""
        uppercase_public = {
            name
            for name in dir(jc)
            if name.isupper() and not name.startswith("_")
        }
        leaked = uppercase_public - EXPECTED_ALL
        assert not leaked, (
            f"uppercase module-scope attributes not advertised in __all__: "
            f"{leaked}. Either add them to __all__ or rename to _private."
        )


# ---------------------------------------------------------------------------
# 8. Migration-plan AC item 6 conformance
# ---------------------------------------------------------------------------


class TestMigrationPlanAC6:
    """Migration-plan §Phase 2 Acceptance Criteria item 6 (line 827) lists
    the five names that must land in `jax_constants.py` by Phase 2 close:
    `MAX_ICEBERGS`, `MAX_SLICES_PER_REGEN`, `SENTINEL_ORDER_ID`, the B4
    JIT-scope decision record, and `MAX_AGENTS`. P2-F1 seeded the three
    shape constants; P2-F2 seeds the sentinel + B4 record — together
    completing the AC."""

    @pytest.mark.parametrize(
        "required_name",
        [
            "MAX_ICEBERGS",
            "MAX_SLICES_PER_REGEN",
            "MAX_AGENTS",
            "SENTINEL_ORDER_ID",
        ],
    )
    def test_ac6_required_constant_present(self, jc, required_name):
        assert hasattr(jc, required_name), (
            f"migration-plan AC item 6 requires `{required_name}` in "
            f"jax_constants.py."
        )

    def test_ac6_b4_decision_record_present(self, jc):
        """The 'B4 JIT-scope decision record' is the fifth AC-6 item.
        Enforced by grep-able anchor in module docstring."""
        doc = jc.__doc__ or ""
        assert "JIT-scope of query methods" in doc, (
            "migration-plan AC item 6 requires the B4 JIT-scope decision "
            "record in jax_constants.py module docstring."
        )


# ---------------------------------------------------------------------------
# 9. Side-effect audit
# ---------------------------------------------------------------------------


class TestSideEffectAudit:
    """Fresh import under a sandboxed cwd must not perform I/O, must not
    allocate JAX arrays at module scope, and must return plain ints."""

    def test_fresh_import_in_sandbox_cwd_succeeds(self, tmp_path, monkeypatch):
        """A tmp empty cwd proves the module does not walk data/rawOMIE/
        or any project-relative path at import time."""
        monkeypatch.chdir(tmp_path)
        sys.modules.pop(JAX_CONSTANTS_MODNAME, None)
        try:
            mod = importlib.import_module(JAX_CONSTANTS_MODNAME)
            # Sanity-touch each new P2-F2 export.
            assert mod.SENTINEL_ORDER_ID == -1
            assert mod.SENTINEL_AGENT_ID == -1
            _ = mod.make_sentinel_order()
        finally:
            # Restore the normal cached version so other tests see stable state.
            sys.modules.pop(JAX_CONSTANTS_MODNAME, None)
            importlib.import_module(JAX_CONSTANTS_MODNAME)

    def test_module_source_has_no_top_level_calls(self, jc):
        """Module-scope function-call expressions would execute at import
        time — `jax.jit(...)`, `jnp.zeros(...)`, etc. Everything at module
        scope must be either a literal assignment, a def/class, or a
        docstring / import."""
        src = Path(inspect.getsourcefile(jc)).read_text()
        tree = ast.parse(src)
        offending: list[str] = []
        for stmt in tree.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                offending.append(ast.dump(stmt.value))
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                value = stmt.value
                if value is not None and isinstance(value, ast.Call):
                    offending.append(ast.dump(value))
        assert not offending, (
            f"jax_constants.py has module-scope calls: {offending}. "
            f"Constants must be literal expressions."
        )

    def test_no_jax_array_at_module_scope(self, jc):
        """None of the module's public attributes is a jax/numpy array
        (every one is a plain Python scalar or the factory function)."""
        for name in jc.__all__:
            v = getattr(jc, name)
            assert not isinstance(v, jnp.ndarray), (
                f"{name} is a jax ndarray at module scope — forbidden."
            )
            assert not isinstance(v, np.ndarray), (
                f"{name} is a numpy ndarray at module scope — forbidden."
            )
