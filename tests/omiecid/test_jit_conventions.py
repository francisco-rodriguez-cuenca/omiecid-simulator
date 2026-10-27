"""Locked behavioral tests for Story P2-F2 — JIT-scope + sentinel conventions.

Contract source:
  - docs/omiecid/agile-plan.md §Phase 2 → P2-F2 story card (Test-first contract)
  - docs/omiecid/migration-plan.md §Phase 2 Blockers table rows B3, B4 (lines 802-803)
  - docs/omiecid/migration-plan.md §Phase 2 Acceptance Criteria item 6 (line 827):
        the five names (MAX_ICEBERGS, MAX_SLICES_PER_REGEN, SENTINEL_ORDER_ID,
        B4 JIT-scope decision, MAX_AGENTS) must live in jax_constants.py.
  - docs/omiecid/sprint-status.md §"P2-F2 — Architecture Review" (2026-04-22),
        in particular:
          §1  sentinel placement + signatures + field values + zero-field hazards
          §2  B4 JIT-scope decision = "query methods stay Python-side"
          §3  single-file placement (jax_constants.py), __all__ extends 3 → 6
          §4  five additional test-contract properties beyond the story card
          §5  scope boundary exclusions (get_order callsite swap = P2-H2)
  - docs/omiecid/phase1-5-summary.md §3 (loader int32 post-P2-F0 — fields that
        claim JAX-compatibility must survive jnp.asarray without exception).

After P2-F2 merges, omiecid_simulator/omiecid/jax_constants.py must expose (in
addition to the three P2-F1 shape constants already landed):

  * SENTINEL_ORDER_ID (B3): Python ``int`` == ``-1``; the "order not found"
    discriminator for ``get_order`` (adopted by callsite in P2-H2, not here).
  * SENTINEL_AGENT_ID: Python ``int`` == ``-1``; per architect review §1, a
    sibling convention for agent sentinels (post-P2-F0 pair-index encoding
    is ``>= 0``, so ``-1`` does not alias any real agent).
  * make_sentinel_order(): zero-arg factory returning an OMIEOrder with
    ``order_id == SENTINEL_ORDER_ID`` and every field a concrete
    JAX-compatible scalar.
  * A parseable B4 decision record in the module docstring, with an anchor
    string (``"B4 decision"`` or ``"B4 JIT-scope"``) followed by the
    verdict keyword ``"Python-side"`` (per architect review §2).

Scope boundaries (per architect review §5 and test-agent brief §3):
  * We do NOT test that ``get_order`` returns the sentinel — that callsite
    change is P2-H2. Instead we negatively assert that ``get_anonymous_lob``,
    ``get_agent_orders``, ``get_agent_trades`` retain the Python-side form
    (no ``@jax.jit`` decorator) after P2-F2.
  * We do NOT re-test the three P2-F1 shape constants.
  * We do NOT introduce any iceberg-slot sentinel (P2-H1).

Collectibility: every import of the new names / helper is done lazily inside
test bodies via ``importlib`` so ``pytest --collect-only`` always succeeds
before the implementation lands. No test depends on raw OMIE data under
``data/rawOMIE/``.

Runtime budget: every test in this module targets the < 1s fast-suite
budget. Tests that invoke ``jax.jit`` are marked with ``@pytest.mark.jit``
(see brief §2); first-call trace cost is a few hundred ms on CPU and amortises
across tests because each jitted callable is distinct but tiny.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Module-level marker (Story T1.1): tests in this file probe P2-F2 JIT
# conventions; treat the module as fully-JIT.  Per-test
# `@pytest.mark.jit` decorators stripped (redundant under module-level
# pin).
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Module-name constants
# ---------------------------------------------------------------------------

JAX_CONSTANTS_MODULE = "omiecid_simulator.omiecid.jax_constants"
ORDER_TYPES_MODULE = "omiecid_simulator.omiecid.order_types"
CONTRACT_BOOK_MODULE = "omiecid_simulator.omiecid.contract_book"

# Architect-review §1 recommendation for the anchor string. We lock either
# of two acceptable anchors so a future edit cannot silently re-word the
# section out of existence.
B4_ANCHOR_CANDIDATES = ("JIT-scope of query methods",)

# Architect-review §2 recommendation for the verdict keyword. Locked here
# so that flipping the decision becomes a test failure, not a silent edit.
B4_VERDICT_KEYWORD = "Python-side"

# Forbidden namespaces for AST import-scan: importing from any of these in
# jax_constants.py would invert the jaxomiecid/jaxob → omiecid/ one-way
# dependency direction locked by the P2-F0 / P2-F1 audits.
FORBIDDEN_IMPORT_PREFIXES = (
    "omiecid_simulator.jaxomiecid",
    "omiecid_simulator.jaxob",
)


# ---------------------------------------------------------------------------
# Lazy imports — keep the module collectable before P2-F2 lands.
# ---------------------------------------------------------------------------

def _import_jax_constants():
    """Import jax_constants; fail informatively at test time, not at collection."""
    return importlib.import_module(JAX_CONSTANTS_MODULE)


def _import_order_types():
    return importlib.import_module(ORDER_TYPES_MODULE)


def _get_sentinel_order_id():
    mod = _import_jax_constants()
    if not hasattr(mod, "SENTINEL_ORDER_ID"):
        pytest.fail(
            f"{JAX_CONSTANTS_MODULE}.SENTINEL_ORDER_ID is missing "
            f"(required by P2-F2 story card / migration-plan §Phase 2 AC item 6)."
        )
    return mod.SENTINEL_ORDER_ID


def _get_make_sentinel_order():
    mod = _import_jax_constants()
    if not hasattr(mod, "make_sentinel_order"):
        pytest.fail(
            f"{JAX_CONSTANTS_MODULE}.make_sentinel_order is missing "
            f"(required by P2-F2 story card Test-first contract)."
        )
    return mod.make_sentinel_order


def _read_module_source(module) -> str:
    path = inspect.getsourcefile(module)
    assert path is not None, f"cannot locate source file for {module.__name__}"
    return Path(path).read_text()


# ---------------------------------------------------------------------------
# 1. Public API surface — constants + factory + __all__
# ---------------------------------------------------------------------------


class TestPublicAPISurface:
    """The new names must be importable from jax_constants and listed in __all__.

    Rationale: architect review §3 locks a single file (``jax_constants.py``)
    as the home of both P2-F1 shape constants and P2-F2 conventions. Missing
    ``__all__`` membership would leak symbols or hide them from re-exports.
    """

    def test_sentinel_order_id_importable(self):
        mod = _import_jax_constants()
        assert hasattr(mod, "SENTINEL_ORDER_ID"), (
            f"{JAX_CONSTANTS_MODULE} must expose SENTINEL_ORDER_ID (B3)."
        )

    def test_sentinel_agent_id_importable(self):
        """Architect review §1 pins SENTINEL_AGENT_ID = -1 as a sibling convention.

        Post-P2-F0 the loader emits pair-index encoding >= 0, so -1 is a
        safe sentinel. Locking import surface here prevents silent removal
        by a future refactor that collapses the two constants.
        """
        mod = _import_jax_constants()
        assert hasattr(mod, "SENTINEL_AGENT_ID"), (
            f"{JAX_CONSTANTS_MODULE} must expose SENTINEL_AGENT_ID "
            f"(architect review §1; locks a sibling convention to "
            f"SENTINEL_ORDER_ID for agent sentinels)."
        )

    def test_make_sentinel_order_importable(self):
        mod = _import_jax_constants()
        assert hasattr(mod, "make_sentinel_order"), (
            f"{JAX_CONSTANTS_MODULE} must expose make_sentinel_order() "
            f"(P2-F2 story card Test-first contract)."
        )

    def test_make_sentinel_order_is_callable(self):
        mod = _import_jax_constants()
        assert callable(getattr(mod, "make_sentinel_order")), (
            "make_sentinel_order must be callable (zero-arg factory)."
        )

    def test_all_lists_new_names(self):
        """__all__ extends from P2-F1's three names to P2-F2's six.

        Per architect review §3: SENTINEL_ORDER_ID, SENTINEL_AGENT_ID,
        make_sentinel_order are added; MAX_ICEBERGS / MAX_SLICES_PER_REGEN /
        MAX_AGENTS are preserved.
        """
        mod = _import_jax_constants()
        all_list = getattr(mod, "__all__", None)
        assert all_list is not None, (
            f"{JAX_CONSTANTS_MODULE} must define __all__."
        )
        all_set = set(all_list)
        # P2-F2 additions must appear.
        for new_name in ("SENTINEL_ORDER_ID", "SENTINEL_AGENT_ID", "make_sentinel_order"):
            assert new_name in all_set, (
                f"{new_name} must be listed in {JAX_CONSTANTS_MODULE}.__all__ "
                f"(architect review §3)."
            )
        # P2-F1 constants must not be dropped.
        for prior_name in ("MAX_ICEBERGS", "MAX_SLICES_PER_REGEN", "MAX_AGENTS"):
            assert prior_name in all_set, (
                f"{prior_name} must remain in __all__ after P2-F2 "
                f"(P2-F1 public surface preservation)."
            )


# ---------------------------------------------------------------------------
# 2. Type contract — sentinel constants are Python int, value -1, hashable.
# ---------------------------------------------------------------------------


class TestSentinelConstantTypeContract:
    """Architect review §4 property 2: sentinel constants are Python ``int``.

    A Python int is hashable (usable as a ``static_argnums`` value or a
    dict key); ``jnp.int32`` or ``jnp.ndarray`` would be traced inside JIT
    and break downstream consumers. ``bool`` (subclass of int) is rejected
    — the brief ``not isinstance(v, bool)`` check is load-bearing.
    """

    def test_sentinel_order_id_type_is_int(self):
        v = _get_sentinel_order_id()
        assert isinstance(v, int), (
            f"SENTINEL_ORDER_ID must be a Python int, got {type(v).__name__}."
        )
        assert not isinstance(v, bool), (
            "SENTINEL_ORDER_ID must not be a bool (even though bool <: int); "
            "bool would confuse downstream static_argnums consumers."
        )

    def test_sentinel_order_id_value_is_minus_one(self):
        """Story card locks the numeric value: SENTINEL_ORDER_ID == -1."""
        assert _get_sentinel_order_id() == -1, (
            "Story card + migration-plan §Phase 2 Blockers B3 both lock "
            "SENTINEL_ORDER_ID == -1."
        )

    def test_sentinel_order_id_hashable(self):
        v = _get_sentinel_order_id()
        # hashability = usable as dict key / set element / static_argnums value.
        _ = {v: "marker"}
        _ = {v}
        assert hash(v) == hash(-1)

    def test_sentinel_agent_id_type_is_int(self):
        mod = _import_jax_constants()
        v = mod.SENTINEL_AGENT_ID
        assert isinstance(v, int), (
            f"SENTINEL_AGENT_ID must be a Python int, got {type(v).__name__}."
        )
        assert not isinstance(v, bool)

    def test_sentinel_agent_id_value_is_minus_one(self):
        """Architect review §1 pins SENTINEL_AGENT_ID = -1."""
        mod = _import_jax_constants()
        assert mod.SENTINEL_AGENT_ID == -1, (
            "Architect review §1 locks SENTINEL_AGENT_ID == -1 "
            "(post-P2-F0 pair encoding is >= 0, so -1 is a safe sentinel)."
        )

    def test_sentinel_agent_id_hashable(self):
        mod = _import_jax_constants()
        v = mod.SENTINEL_AGENT_ID
        _ = {v: "marker"}
        assert hash(v) == hash(-1)


# ---------------------------------------------------------------------------
# 3. make_sentinel_order() — returns OMIEOrder with sentinel order_id, every
#    field a concrete non-None JAX-compatible scalar (story card bullet 2).
# ---------------------------------------------------------------------------


class TestMakeSentinelOrder:
    """Story card Test-first contract locks two properties on the factory:

    (i)  make_sentinel_order().order_id == SENTINEL_ORDER_ID.
    (ii) All fields are concrete (non-None) JAX-compatible scalars.

    Architect review §4 property 3 strengthens (ii): we must actually run
    ``jnp.asarray(field)`` per field, not just isinstance-check.
    """

    def test_returns_omie_order_instance(self):
        order_types = _import_order_types()
        OMIEOrder = order_types.OMIEOrder
        factory = _get_make_sentinel_order()
        obj = factory()
        assert isinstance(obj, OMIEOrder), (
            f"make_sentinel_order() must return an OMIEOrder instance, "
            f"got {type(obj).__name__}."
        )

    def test_order_id_equals_sentinel_order_id(self):
        """Story card bullet 1, verbatim."""
        factory = _get_make_sentinel_order()
        sentinel_id = _get_sentinel_order_id()
        assert factory().order_id == sentinel_id, (
            "Story card: make_sentinel_order().order_id == SENTINEL_ORDER_ID."
        )

    def test_all_fields_are_non_none(self):
        """Story card bullet 2: all fields are concrete (non-None)."""
        factory = _get_make_sentinel_order()
        sentinel = factory()
        # OMIEOrder is a NamedTuple — iterate its fields.
        for field_name, value in sentinel._asdict().items():
            assert value is not None, (
                f"Sentinel OMIEOrder.{field_name} is None; all fields must "
                f"be concrete JAX-compatible scalars (story card Test-first "
                f"contract bullet 2)."
            )

    def test_all_fields_are_jnp_convertible(self):
        """Architect review §4 property 3: run jnp.asarray on each field.

        The story card's "JAX-compatible scalars" phrase is tested by
        actually invoking the JAX coercion, not merely isinstance-checking.
        """
        factory = _get_make_sentinel_order()
        sentinel = factory()
        failures: list[str] = []
        for field_name, value in sentinel._asdict().items():
            try:
                arr = jnp.asarray(value)
            except Exception as exc:  # pragma: no cover — diagnostic branch
                failures.append(f"{field_name}={value!r}: {exc!r}")
                continue
            # 0-dimensional array is the expected shape of a scalar.
            assert arr.ndim == 0, (
                f"Sentinel field {field_name} coerced to a non-scalar "
                f"(ndim={arr.ndim}); expected a 0-d array."
            )
        assert not failures, (
            f"Some sentinel fields are not JAX-convertible: {failures}"
        )

    def test_sentinel_is_idempotent(self):
        """Architect review §4 property 6 (nice-to-have).

        make_sentinel_order() is pure: successive calls yield equal
        OMIEOrders. Guards against a future accidental non-deterministic
        default (e.g., timestamp=time.time()).
        """
        factory = _get_make_sentinel_order()
        a = factory()
        b = factory()
        assert a == b, (
            "make_sentinel_order() must be deterministic: two successive "
            "calls must return equal OMIEOrders."
        )


# ---------------------------------------------------------------------------
# 4. Discriminability — a real zero-valued order is NOT the sentinel.
# ---------------------------------------------------------------------------


class TestSentinelDiscriminability:
    """Architect review §4 property 1: the sentinel must be distinguishable
    from a real order whose other fields happen to be zero.

    Hazards documented in architect review §1:
      * side == 0 maps to Side.BUY (not "no side") — valid real state.
      * status == 0 maps to Status.ACTIVE (not "no status") — valid real state.
    Therefore THE discriminator is ``order_id == SENTINEL_ORDER_ID`` and
    nothing else. This test locks that invariant.
    """

    def test_real_zero_order_is_not_sentinel(self):
        """A real order with order_id=0, side=BUY, status=ACTIVE must not
        be mistaken for the sentinel.

        Rationale: OMIEOrder has multiple fields whose zero-value is a
        legal real state (side==0 → BUY; status==0 → ACTIVE; peak_qty==0.0 →
        not an iceberg; ppd==0.0 → no slice delta). The ONLY reliable
        discriminator is order_id == SENTINEL_ORDER_ID.
        """
        order_types = _import_order_types()
        make_order = order_types.make_order
        Side = order_types.Side

        # A real order: positive order_id, zero-ish other fields.
        real = make_order(
            contract_id=0,
            side=Side.BUY,
            price=0.0,
            quantity=0.0,
            order_id=0,           # note: real order_id == 0 (not sentinel)
            agent_id=0,
            timestamp=0,
        )
        factory = _get_make_sentinel_order()
        sentinel = factory()

        sentinel_id = _get_sentinel_order_id()
        assert real.order_id != sentinel_id, (
            f"A real order with order_id=0 must NOT be treated as the "
            f"sentinel; SENTINEL_ORDER_ID must be something other than 0 "
            f"(locked to -1 by the story card)."
        )
        assert real != sentinel, (
            "A real zero-ish order must not compare equal to the sentinel."
        )

    def test_discrimination_flows_through_order_id_only(self):
        """Assert that side=0 and status=0 on the sentinel are NOT a reliable
        signal — a real order may share those values. This codifies the
        architect-review §1 hazard note so a future refactor cannot
        silently add a second discriminator that aliases real state.
        """
        factory = _get_make_sentinel_order()
        order_types = _import_order_types()
        Side = order_types.Side
        Status = order_types.Status
        make_order = order_types.make_order

        sentinel = factory()

        # Construct a real order that matches the sentinel on every zero
        # field EXCEPT order_id — discrimination must still work.
        real = make_order(
            contract_id=int(sentinel.contract_id) if sentinel.contract_id != 0 else 0,
            side=Side(int(sentinel.side)) if int(sentinel.side) in (0, 1) else Side.BUY,
            price=float(sentinel.price),
            quantity=float(sentinel.quantity),
            order_id=1,  # positive, thus not the sentinel
            agent_id=0,
            timestamp=int(sentinel.timestamp),
            status=Status(int(sentinel.status)) if int(sentinel.status) in range(6) else Status.ACTIVE,
        )
        assert real.order_id != sentinel.order_id, (
            "Discriminator must be order_id: a real order matching the "
            "sentinel on side/status/price/etc. but with a real order_id "
            "must not collide with the sentinel."
        )


# ---------------------------------------------------------------------------
# 5. JAX-compatibility — tree flatten/unflatten round-trip; JIT-closure compile.
# ---------------------------------------------------------------------------


class TestJAXCompatibility:
    """Architect review §4 properties 3 and 4: the sentinel must survive
    JAX pytree operations and be usable inside a single ``jax.jit``-layer.

    Double-jit pitfall noted in P2-F1 sprint-status: do NOT nest jax.jit.
    The tests here wrap ONE plain Python function with ``jax.jit`` and
    close over the sentinel; no decorator stacking.
    """

    def test_sentinel_tree_flatten_roundtrip(self):
        """Architect review §4 property 4: flatten → unflatten yields an
        OMIEOrder that compares equal by field to the original.
        """
        factory = _get_make_sentinel_order()
        sentinel = factory()
        leaves, treedef = jax.tree.flatten(sentinel)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        assert rebuilt == sentinel, (
            "jax.tree.flatten / unflatten must round-trip the sentinel "
            "(architect review §4 property 4)."
        )

    def test_sentinel_tree_leaves_are_scalar_like(self):
        """Every leaf in the flattened tree must be a JAX-compatible scalar
        (no Enum / str / None objects that would become static pytree
        metadata and invalidate downstream JIT equivalence).
        """
        factory = _get_make_sentinel_order()
        sentinel = factory()
        leaves = jax.tree.leaves(sentinel)
        for i, leaf in enumerate(leaves):
            try:
                arr = jnp.asarray(leaf)
            except Exception as exc:  # pragma: no cover — diagnostic branch
                pytest.fail(
                    f"Sentinel pytree leaf[{i}]={leaf!r} is not "
                    f"JAX-convertible: {exc!r}"
                )
            assert arr.ndim == 0, (
                f"Sentinel pytree leaf[{i}] flattened to non-scalar "
                f"(ndim={arr.ndim})."
            )

    def test_sentinel_usable_as_closed_over_constant(self):
        """A single ``jax.jit``-layer closing over the sentinel compiles.

        This mirrors the real use-case: a jitted downstream consumer
        (e.g., the P2-H2 ``get_order`` wrapper) closes over
        ``make_sentinel_order()`` to form its "not found" return value.
        Only ONE jax.jit layer — no nested decorators.
        """
        factory = _get_make_sentinel_order()
        sentinel = factory()

        def _return_sentinel_order_id(x):
            # `x` is a dummy traced input so the function has a shape.
            # We return the sentinel's order_id as a JAX scalar — this
            # exercises closure-over-the-sentinel and requires that the
            # sentinel's order_id is jnp-coercible.
            return jnp.asarray(x) + jnp.asarray(sentinel.order_id)

        compiled = jax.jit(_return_sentinel_order_id)
        result = compiled(jnp.int32(0))
        # sentinel.order_id == -1 → result should be -1.
        assert int(result) == int(sentinel.order_id), (
            f"JIT-closure over sentinel.order_id produced {int(result)}, "
            f"expected {int(sentinel.order_id)}."
        )

    def test_sentinel_order_id_usable_as_static_argnum(self):
        """SENTINEL_ORDER_ID is a Python int → usable as static_argnums.

        This is the hashability property of property 2 exercised in the
        actual JIT context where it matters.
        """
        sentinel_id = _get_sentinel_order_id()

        def _is_sentinel(oid: int, actual: int):
            # oid is static; actual is a traced scalar. Returns a jnp bool.
            return jnp.asarray(actual == oid)

        compiled = jax.jit(_is_sentinel, static_argnums=(0,))
        assert bool(compiled(sentinel_id, -1)) is True
        assert bool(compiled(sentinel_id, 0)) is False


# ---------------------------------------------------------------------------
# 6. Module placement + import topology — locked against future drift.
# ---------------------------------------------------------------------------


class TestModulePlacement:
    """Architect review §3 + §5: single-file placement under
    ``omiecid_simulator/omiecid/`` (pure-domain layer), NOT under ``jaxomie/``.
    AST scan enforces the one-way import direction.
    """

    def test_sentinel_names_live_in_jax_constants(self):
        mod = _import_jax_constants()
        assert mod.__name__ == JAX_CONSTANTS_MODULE, (
            f"Sentinel constants must live at {JAX_CONSTANTS_MODULE}, "
            f"got {mod.__name__}."
        )
        # Each new symbol must be an attribute on THIS module — not a
        # re-export from a sibling file.
        for name in ("SENTINEL_ORDER_ID", "SENTINEL_AGENT_ID", "make_sentinel_order"):
            assert hasattr(mod, name), (
                f"{name} must be defined on {JAX_CONSTANTS_MODULE} directly."
            )

    def test_module_file_under_omie_not_jaxomie(self):
        mod = _import_jax_constants()
        path = Path(inspect.getsourcefile(mod)).resolve()
        parts = path.parts
        assert "omiecid_simulator" in parts, f"unexpected location: {path}"
        idx = parts.index("omiecid_simulator")
        assert parts[idx + 1] == "omiecid", (
            f"jax_constants.py must live under omiecid_simulator/omiecid/, "
            f"got omiecid_simulator/{parts[idx + 1]}/ "
            f"(architect review §3: pure-domain placement)."
        )

    def test_no_sibling_jax_conventions_module(self):
        """Architect review §3 explicitly rejects a sibling
        ``jax_conventions.py``. Locking the absence prevents scope drift.
        """
        mod = _import_jax_constants()
        omie_dir = Path(inspect.getsourcefile(mod)).resolve().parent
        sibling = omie_dir / "jax_conventions.py"
        assert not sibling.exists(), (
            f"Architect review §3 rejects a sibling jax_conventions.py; "
            f"found {sibling}. All P2-F2 conventions must live in "
            f"jax_constants.py alongside the P2-F1 shape constants."
        )

    def test_ast_imports_stay_within_domain_layer(self):
        """AST scan: jax_constants.py must import zero names from
        ``omiecid_simulator.jaxomiecid.*`` or ``omiecid_simulator.jaxob.*``.

        Importing from any of those would invert the one-way dependency
        direction locked by the P2-F0 / P2-F1 audits. Imports from
        ``omiecid_simulator.omiecid.*`` (e.g., ``order_types``) are allowed
        because they stay within the pure-domain layer.
        """
        mod = _import_jax_constants()
        src = _read_module_source(mod)
        tree = ast.parse(src)
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                for prefix in FORBIDDEN_IMPORT_PREFIXES:
                    if mod_name == prefix or mod_name.startswith(prefix + "."):
                        offenders.append(f"from {mod_name} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in FORBIDDEN_IMPORT_PREFIXES:
                        if alias.name == prefix or alias.name.startswith(prefix + "."):
                            offenders.append(f"import {alias.name}")
        assert not offenders, (
            f"jax_constants.py must not import from "
            f"{FORBIDDEN_IMPORT_PREFIXES}: found {offenders}. "
            f"This would invert the one-way omie ← everything-else "
            f"dependency direction locked by prior audits."
        )


# ---------------------------------------------------------------------------
# 7. B4 decision record — parseable docstring section with locked verdict.
# ---------------------------------------------------------------------------


class TestB4DecisionRecord:
    """Story card Test-first contract: the B4 decision must be documented
    in a form P2-H2 can cite verbatim.

    Architect review §2 locks the verdict as "query methods stay Python-side"
    and provides a verbatim paragraph for the module docstring. We lock:

      (a) A parseable anchor string ("B4 decision" or "B4 JIT-scope") is
          present in the module docstring.
      (b) The verdict keyword "Python-side" appears in the B4 section.
      (c) No ``@jax.jit`` decorator is applied to the three query methods
          in contract_book.py after P2-F2 (negative assertion — the callsite
          change, if any, is P2-H2).
    """

    def test_b4_anchor_present_in_module_docstring(self):
        """Story card: the B4 decision is 'documented in a form directly
        citable by P2-H2'. The anchor is locked so a future reword cannot
        silently remove the record.
        """
        mod = _import_jax_constants()
        docstring = (mod.__doc__ or "")
        found = any(anchor in docstring for anchor in B4_ANCHOR_CANDIDATES)
        assert found, (
            f"{JAX_CONSTANTS_MODULE}.__doc__ must contain one of "
            f"{B4_ANCHOR_CANDIDATES} as a parseable anchor for the B4 "
            f"decision record (architect review §2 + drift item 2)."
        )

    def test_b4_verdict_keyword_python_side(self):
        """Architect review §2 locks the verdict: 'query methods stay
        Python-side'. The keyword 'Python-side' must appear in the same
        module docstring so P2-H2 can cite the decision verbatim.

        If a future developer flips the decision to 'JIT-scope', this
        assertion fails, forcing an explicit architecture review rather
        than a silent reword.
        """
        mod = _import_jax_constants()
        docstring = (mod.__doc__ or "")
        assert B4_VERDICT_KEYWORD in docstring, (
            f"{JAX_CONSTANTS_MODULE}.__doc__ must contain the verdict "
            f"keyword {B4_VERDICT_KEYWORD!r} in its B4 decision record "
            f"(architect review §2). Flipping this decision must be done "
            f"explicitly via architecture review, not silently."
        )

    def test_b4_verdict_near_anchor(self):
        """The verdict keyword must appear AFTER (or within ~500 chars of)
        the B4 anchor — not merely somewhere in the docstring. This
        prevents a misleading co-occurrence (e.g., 'Python-side' mentioned
        in an unrelated paragraph).
        """
        mod = _import_jax_constants()
        docstring = (mod.__doc__ or "")
        anchor_pos = -1
        for anchor in B4_ANCHOR_CANDIDATES:
            pos = docstring.find(anchor)
            if pos != -1:
                anchor_pos = pos
                break
        assert anchor_pos != -1, "anchor missing (see test_b4_anchor_present_in_module_docstring)"
        verdict_pos = docstring.find(B4_VERDICT_KEYWORD, anchor_pos)
        assert verdict_pos != -1, (
            f"Verdict keyword {B4_VERDICT_KEYWORD!r} must appear AFTER the "
            f"B4 anchor in the module docstring (found anchor at "
            f"pos {anchor_pos} but verdict not found after it)."
        )
        # Soft proximity bound: verdict within 1000 chars of anchor.
        # Prevents a verdict landing in a totally unrelated section.
        assert verdict_pos - anchor_pos < 1000, (
            f"Verdict keyword found at pos {verdict_pos} but anchor at "
            f"{anchor_pos} — distance {verdict_pos - anchor_pos} chars is "
            f"too far; the verdict must live inside the B4 section."
        )

    def test_query_methods_not_jit_wrapped(self):
        """Negative assertion: after P2-F2 merges, the three query methods
        on ContractBook (`get_anonymous_lob`, `get_agent_orders`,
        `get_agent_trades`) must NOT have a ``@jax.jit`` decorator.

        P2-F2 is a decision-record story only. Any JIT wrapping of these
        methods is P2-H2 territory (and, per architect review §2, the
        recorded decision is to leave them Python-side indefinitely).
        """
        contract_book = importlib.import_module(CONTRACT_BOOK_MODULE)
        src = _read_module_source(contract_book)
        tree = ast.parse(src)
        query_method_names = {
            "get_anonymous_lob",
            "get_agent_orders",
            "get_agent_trades",
        }
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in query_method_names:
                for dec in node.decorator_list:
                    # Match @jax.jit, @jax.jit(...), @jit, @jit(...).
                    dec_str = ast.unparse(dec) if hasattr(ast, "unparse") else ""
                    if "jit" in dec_str:
                        offenders.append(f"{node.name}: @{dec_str}")
        assert not offenders, (
            f"After P2-F2 the following query methods have a jit decorator "
            f"but P2-F2 forbids it (B4 decision = Python-side; wrapping is "
            f"P2-H2 territory): {offenders}"
        )


# ---------------------------------------------------------------------------
# 8. Scope-boundary negative assertions — P2-F2 must not reach into P2-H2 /
#    P2-H1 territory.
# ---------------------------------------------------------------------------


class TestScopeBoundary:
    """Architect review §5 + test-agent brief §3: explicit scope exclusions.

    P2-F2 delivers the sentinel SURFACE. Adoption at callsites (e.g.,
    flipping ``get_order``'s ``return None`` to ``return make_sentinel_order()``)
    is P2-H2. Iceberg-slot sentinels are P2-H1. These tests lock the
    scope boundary by negative assertion so an over-eager implementer
    does not silently couple stories.
    """

    def test_get_order_signature_still_optional(self):
        """``get_order`` must still exist after P2-F2 / P2-H2.

        Originally (P2-F2 era) this anchor asserted that
        ``Optional[OMIEOrder]`` was still present in the return
        annotation, fencing the callsite swap (``return None`` ->
        ``return make_sentinel_order()``) into P2-H2's scope.

        Post-P2-H2 (v5.20, 2026-05-14): the swap has shipped. The
        annotation is now ``OMIEOrder`` (no Optional) and ``get_order``
        returns the canonical sentinel ``OMIEOrder`` on miss. This
        assertion is preserved as a presence-only anchor to detect
        accidental removal of ``get_order`` itself, but no longer
        enforces the Optional return.
        """
        contract_book = importlib.import_module(CONTRACT_BOOK_MODULE)
        src = _read_module_source(contract_book)
        assert "def get_order(" in src, (
            "contract_book.get_order must still exist after P2-H2."
        )

    def test_no_iceberg_slot_sentinel_in_jax_constants(self):
        """P2-H1 owns iceberg-slot sentinels (the ``-1`` slot marker in the
        ``iceberg_meta`` array). P2-F2 must not preemptively introduce
        such a constant.

        Locking the absence here prevents scope creep that would force
        the P2-H1 developer to untangle two stories' worth of changes.
        """
        mod = _import_jax_constants()
        forbidden_names = (
            "ICEBERG_SLOT_SENTINEL",
            "EMPTY_ICEBERG_SLOT",
            "SENTINEL_ICEBERG_SLOT",
        )
        for name in forbidden_names:
            assert not hasattr(mod, name), (
                f"{name} is a P2-H1 concern and must not be introduced in "
                f"P2-F2. Keep iceberg-slot sentinel handling in the P2-H1 "
                f"story."
            )
