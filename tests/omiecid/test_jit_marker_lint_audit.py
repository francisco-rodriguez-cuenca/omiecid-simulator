"""Audit-grade independent probes for Story T1.1.

This file is the audit-mode counterpart to ``test_jit_marker_lint.py``.
It probes contract corners that are adjacent to (not duplicative of) the
original locked-test surface, derived independently from the T1.1
story-card Deliverables and Architecture-audit checks (a)-(g).

Probe categories:
  1. Cache config invariant (AST-parsed conftest)
  2. ``.gitignore`` line presence
  3. No ``import jax`` in lint module (AST)
  4. Module-level ``pytestmark`` coverage on the 11 fully-jit files
  5. No surviving redundant per-test ``@pytest.mark.jit`` decorators in
     the 11 fully-jit files (architecture-audit check (c) negative side)
  6. Per-test marker presence in mixed files at function/class/module scope
  7. Synthetic regression: lint flags ``jax.pmap`` decorator nested
     inside a class
  8. ``# noqa: jit-marker`` is line-scoped, not module-scoped
  9. Cache directory side effect: ``jax.config`` reads back the three knobs

Probes 1-8 are AST-only (sub-second). Probe 9 imports JAX and is marked
``@pytest.mark.jit`` so it falls outside the ``not jit`` fast-path.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest

# Public lint entry point (probe 7 only). Importing this module does NOT
# import jax (the module is pure-stdlib by design — see probe 3).
from tests.omiecid.test_jit_marker_lint import lint_source


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TESTS_OMIE = _REPO_ROOT / "tests" / "omiecid"
_CONFTEST = _TESTS_OMIE / "conftest.py"
_LINT_MODULE = _TESTS_OMIE / "test_jit_marker_lint.py"
_GITIGNORE = _REPO_ROOT / ".gitignore"

# Story-card Deliverable (3): 11 fully-jit files that must carry
# module-level ``pytestmark = pytest.mark.jit`` and have no surviving
# per-test ``@pytest.mark.jit`` decorators.
_FULLY_JIT_FILES = [
    "test_contract_status_jit_decision.py",
    "test_contract_status_jit_decision_audit.py",
    "test_settlement_jit.py",
    "test_settlement_jit_audit.py",
    "test_order_validator_jit.py",
    "test_iceberg_manager_jit.py",
    "test_iceberg_manager_jit_audit.py",
    "test_jit_conventions.py",
    "test_jit_conventions_audit.py",
    "test_trade_extraction_jit.py",
    "test_trade_extraction_jit_audit.py",
]

# Story-card Deliverable (4): mixed files with per-test markers.
# Each test function (or its enclosing class / module) that calls
# ``jax.jit`` / ``jax.pmap`` must have ``pytest.mark.jit`` in scope.
_MIXED_FILES = [
    "test_jax_constants.py",
    "test_jax_constants_audit.py",
    "test_encoding_snap_semantics.py",
    "test_order_validator_audit.py",
    "test_order_validator_tick_precision.py",
    "test_order_validator_tick_precision_audit.py",
    "test_trade_extraction_ring_buffer_audit.py",
]


# ---------------------------------------------------------------------------
# AST helpers (independent re-implementation; do not call into the lint
# module's private helpers).
# ---------------------------------------------------------------------------


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _is_pytest_mark_jit(node: ast.expr) -> bool:
    """``pytest.mark.jit`` (Attribute) or ``pytest.mark.jit()`` (Call)."""
    if isinstance(node, ast.Call):
        return _is_pytest_mark_jit(node.func)
    if isinstance(node, ast.Attribute) and node.attr == "jit":
        v = node.value
        if isinstance(v, ast.Attribute) and v.attr == "mark":
            base = v.value
            if isinstance(base, ast.Name) and base.id == "pytest":
                return True
    return False


def _value_contains_jit_marker(value: ast.expr) -> bool:
    if _is_pytest_mark_jit(value):
        return True
    if isinstance(value, (ast.List, ast.Tuple)):
        return any(_is_pytest_mark_jit(e) for e in value.elts)
    return False


def _body_assigns_pytestmark_with_jit(body: list[ast.stmt]) -> bool:
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "pytestmark":
                    if _value_contains_jit_marker(stmt.value):
                        return True
        elif isinstance(stmt, ast.AnnAssign):
            if (
                isinstance(stmt.target, ast.Name)
                and stmt.target.id == "pytestmark"
                and stmt.value is not None
                and _value_contains_jit_marker(stmt.value)
            ):
                return True
    return False


def _attr_chain_root_jax(node: ast.expr) -> tuple[bool, str | None]:
    if not isinstance(node, ast.Attribute):
        return (False, None)
    attr = node.attr
    cur: ast.expr = node.value
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id == "jax":
        return (True, attr)
    return (False, None)


# ---------------------------------------------------------------------------
# Probe 1 — Cache config invariant (AST-parsed conftest)
# ---------------------------------------------------------------------------


class TestCacheConfigInvariant:
    """``conftest.py::pytest_configure`` sets all three knobs to the
    aggressive (-1, 0.0, ``.jax_cache``) values per story-card
    Deliverable (1) and Architecture-audit check (a)."""

    def _collect_jax_config_updates(self) -> dict[str, ast.expr]:
        """Return ``{key_str: value_expr}`` for every
        ``jax.config.update("KEY", VALUE)`` call inside ``pytest_configure``.
        """
        tree = _parse(_CONFTEST)
        # Find the function def named ``pytest_configure``.
        fn = None
        for stmt in tree.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "pytest_configure":
                fn = stmt
                break
        assert fn is not None, "conftest.py must define pytest_configure"
        out: dict[str, ast.expr] = {}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            # Match jax.config.update(...)
            if not (
                isinstance(f, ast.Attribute)
                and f.attr == "update"
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "config"
                and isinstance(f.value.value, ast.Name)
                and f.value.value.id == "jax"
            ):
                continue
            if len(node.args) < 2:
                continue
            key_node = node.args[0]
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                out[key_node.value] = node.args[1]
        return out

    def test_cache_dir_set_to_jax_cache(self):
        updates = self._collect_jax_config_updates()
        assert "jax_compilation_cache_dir" in updates, (
            "conftest must set jax_compilation_cache_dir"
        )
        v = updates["jax_compilation_cache_dir"]
        assert isinstance(v, ast.Constant) and v.value == ".jax_cache", (
            f"expected .jax_cache, got {ast.dump(v)}"
        )

    def test_min_entry_size_bytes_is_negative_one(self):
        updates = self._collect_jax_config_updates()
        assert "jax_persistent_cache_min_entry_size_bytes" in updates
        v = updates["jax_persistent_cache_min_entry_size_bytes"]
        # Value is -1: an UnaryOp(USub, Constant(1)) on Python-source level.
        if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub):
            inner = v.operand
            assert isinstance(inner, ast.Constant) and inner.value == 1
        else:
            # Fallback: literal Constant(-1) (rare but legal).
            assert isinstance(v, ast.Constant) and v.value == -1

    def test_min_compile_time_secs_is_zero_float(self):
        updates = self._collect_jax_config_updates()
        assert "jax_persistent_cache_min_compile_time_secs" in updates
        v = updates["jax_persistent_cache_min_compile_time_secs"]
        assert isinstance(v, ast.Constant), f"expected literal, got {ast.dump(v)}"
        # Story-card pin: exactly 0.0 (float). Accept 0 or 0.0 numerically;
        # the architecture-audit check (a) cites "= 0.0" so we lock the
        # numeric value, not the literal type.
        assert v.value == 0 or v.value == 0.0
        assert isinstance(v.value, (int, float))


# ---------------------------------------------------------------------------
# Probe 2 — .gitignore line present
# ---------------------------------------------------------------------------


def test_gitignore_contains_jax_cache_on_noncomment_line():
    """Architecture-audit check (b): ``.gitignore`` gains exactly one new
    line ``.jax_cache/``. We tolerate either ``.jax_cache/`` or
    ``.jax_cache`` (both are valid gitignore syntax) but require it to
    appear on a non-comment, non-empty line.
    """
    content = _GITIGNORE.read_text(encoding="utf-8")
    found = False
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in (".jax_cache/", ".jax_cache"):
            found = True
            break
    assert found, ".gitignore must contain .jax_cache/ on a non-comment line"


# ---------------------------------------------------------------------------
# Probe 3 — No `import jax` in lint module (AST)
# ---------------------------------------------------------------------------


def test_lint_module_has_no_jax_import_via_ast():
    """T1.1 AC #8 / arch-audit check (e): the lint module must remain
    JAX-free so the default ``pytest -m "not jit"`` fast-path stays cheap.
    Independent re-walk of the AST (the original suite has its own
    version; this is the audit's redundant-but-stricter check).
    """
    tree = _parse(_LINT_MODULE)
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "jax" or alias.name.startswith("jax."):
                    offending.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "jax" or mod.startswith("jax."):
                offending.append(f"from {mod} import ...")
    assert not offending, (
        f"test_jit_marker_lint.py must not import jax; found: {offending}"
    )


# ---------------------------------------------------------------------------
# Probe 4 — Module-pytestmark coverage on the 11 fully-jit files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", _FULLY_JIT_FILES)
def test_fully_jit_file_has_module_pytestmark(filename):
    """Architecture-audit check (c) positive side: each of the 11
    fully-jit files carries ``pytestmark = pytest.mark.jit`` (or a
    list/tuple containing it) at module scope.
    """
    path = _TESTS_OMIE / filename
    assert path.exists(), f"fully-jit file missing on disk: {filename}"
    tree = _parse(path)
    assert _body_assigns_pytestmark_with_jit(tree.body), (
        f"{filename} is missing module-level `pytestmark = pytest.mark.jit`"
    )


# ---------------------------------------------------------------------------
# Probe 5 — No surviving redundant per-test @pytest.mark.jit decorators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", _FULLY_JIT_FILES)
def test_fully_jit_file_has_no_redundant_per_test_jit_decorators(filename):
    """Architecture-audit check (c) negative side: with the module-level
    ``pytestmark`` in place, any per-test ``@pytest.mark.jit`` decorator
    is redundant and should have been stripped during P2-T1.1
    implementation. AST-walk every function definition in the module and
    confirm none carry the marker as a decorator.
    """
    path = _TESTS_OMIE / filename
    tree = _parse(path)
    redundant: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if _is_pytest_mark_jit(dec):
                    redundant.append(f"{node.name}@line{dec.lineno}")
    assert not redundant, (
        f"{filename} carries redundant @pytest.mark.jit decorators "
        f"(module-level pytestmark already covers them): {redundant}"
    )


# ---------------------------------------------------------------------------
# Probe 6 — Per-test marker presence in mixed files
#
# This is stronger than AC #5 (collected-count). For each mixed file, we
# AST-walk every test function. If the function (in its lexical scope)
# either calls ``jax.jit`` / ``jax.pmap`` or uses one as a decorator, we
# require a ``pytest.mark.jit`` marker reachable from one of:
#   - decorator on the function
#   - class-level ``pytestmark`` on an enclosing class
#   - module-level ``pytestmark``
# ---------------------------------------------------------------------------


def _function_invokes_jit_or_pmap(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function body or its decorators reference
    ``jax.jit`` / ``jax.pmap`` (call or decorator form).
    """
    # Decorator form on the function itself.
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        is_jax, name = _attr_chain_root_jax(target)
        if is_jax and name in {"jit", "pmap"}:
            return True
    # Body: any Call whose func is ``jax.jit`` / ``jax.pmap``.
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            is_jax, name = _attr_chain_root_jax(node.func)
            if is_jax and name in {"jit", "pmap"}:
                return True
        # Nested decorator on inner def.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                is_jax, name = _attr_chain_root_jax(target)
                if is_jax and name in {"jit", "pmap"}:
                    return True
    return False


@pytest.mark.parametrize("filename", _MIXED_FILES)
def test_mixed_file_per_test_marker_presence(filename):
    """Every test function in a mixed file that invokes jit/pmap has a
    ``pytest.mark.jit`` marker resolvable in lexical scope."""
    path = _TESTS_OMIE / filename
    if not path.exists():
        pytest.skip(f"mixed file not present (story-card delta): {filename}")
    tree = _parse(path)
    module_in_scope = _body_assigns_pytestmark_with_jit(tree.body)

    unmarked: list[str] = []

    def _visit(stmts: list[ast.stmt], class_in_scope: bool, class_name: str | None):
        for stmt in stmts:
            if isinstance(stmt, ast.ClassDef):
                cls_scope = class_in_scope or _body_assigns_pytestmark_with_jit(stmt.body)
                _visit(stmt.body, cls_scope, stmt.name)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not stmt.name.startswith("test_"):
                    continue
                if not _function_invokes_jit_or_pmap(stmt):
                    continue
                fn_marked = any(_is_pytest_mark_jit(d) for d in stmt.decorator_list)
                if module_in_scope or class_in_scope or fn_marked:
                    continue
                where = f"{class_name}::{stmt.name}" if class_name else stmt.name
                unmarked.append(f"{where}@line{stmt.lineno}")

    _visit(tree.body, class_in_scope=False, class_name=None)
    assert not unmarked, (
        f"{filename} has jit/pmap-invoking test(s) without "
        f"`pytest.mark.jit` in scope: {unmarked}"
    )


# ---------------------------------------------------------------------------
# Probe 7 — Synthetic regression flagged
#
# Independent shape (different from the original suite's self-tests):
# a ``jax.pmap`` *decorator* on a method nested inside a class, with
# no marker in scope at any level. Must be flagged.
# ---------------------------------------------------------------------------


def test_lint_flags_pmap_decorator_on_class_method():
    src = textwrap.dedent(
        """
        import jax
        import pytest

        class TestNested:

            @jax.pmap
            def test_method(self):
                return 1
        """
    )
    violations = lint_source(src)
    assert len(violations) == 1, f"expected 1 violation, got {violations}"
    v = violations[0]
    assert v.name == "pmap"
    assert v.kind == "decorator"


def test_lint_clears_pmap_decorator_when_class_pytestmark_present():
    """Counterpart probe: same shape, but with class pytestmark — must
    NOT be flagged (proves the class-scope resolution path is wired)."""
    src = textwrap.dedent(
        """
        import jax
        import pytest

        class TestNested:
            pytestmark = pytest.mark.jit

            @jax.pmap
            def test_method(self):
                return 1
        """
    )
    assert lint_source(src) == []


# ---------------------------------------------------------------------------
# Probe 8 — `# noqa: jit-marker` is line-scoped
# ---------------------------------------------------------------------------


def test_noqa_on_wrong_line_does_not_silence_violation():
    """A ``# noqa: jit-marker`` comment placed on a *different* line than
    the offending call must NOT silence the violation. The escape hatch
    is per-site, not module-wide."""
    src = textwrap.dedent(
        """
        # noqa: jit-marker
        import jax

        def test_thing():
            # noqa: jit-marker
            f = jax.jit(lambda x: x + 1)
            f(1)
        """
    )
    violations = lint_source(src)
    # The standalone-comment-line noqa above the call MUST NOT clear the
    # violation; only a same-line noqa on the offending statement does.
    assert len(violations) == 1, (
        f"noqa on a different line should not silence the violation; "
        f"got {violations}"
    )
    assert violations[0].name == "jit"
    assert violations[0].kind == "call"


def test_noqa_on_decorator_line_silences_only_that_decorator():
    """Same-line noqa on a decorator clears that decorator's violation
    but not a separate jit call later in the same function."""
    src = textwrap.dedent(
        """
        import jax

        def test_thing():
            @jax.jit  # noqa: jit-marker
            def inner(x):
                return x + 1
            f = jax.jit(lambda y: y * 2)
            inner(1)
            f(2)
        """
    )
    violations = lint_source(src)
    # Decorator noqa-cleared; the bare ``jax.jit(...)`` call remains.
    assert len(violations) == 1
    assert violations[0].kind == "call"
    assert violations[0].name == "jit"


# ---------------------------------------------------------------------------
# Probe 9 — JAX cache config side effect (jit-marked, JAX import)
# ---------------------------------------------------------------------------


@pytest.mark.jit
def test_jax_config_reads_back_three_knobs():
    """After conftest's ``pytest_configure`` runs (session-scoped), the
    three JAX-config knobs must read back as the aggressive (-1, 0.0,
    ``.jax_cache``) values. This is the runtime counterpart to probe 1
    (which only inspects the source).

    Marked ``@pytest.mark.jit`` because it imports JAX and the lint
    requires the marker.
    """
    import jax

    def _read(flag_name, attr_name):
        # Some JAX builds register flags as contextmanagers, which makes
        # ``jax.config.read(name)`` raise ``AttributeError``. Fall back to
        # the attribute form on the config object in that case.
        try:
            return jax.config.read(flag_name)
        except (AttributeError, LookupError):
            return getattr(jax.config, attr_name)

    cache_dir = _read("jax_compilation_cache_dir", "jax_compilation_cache_dir")
    min_bytes = _read(
        "jax_persistent_cache_min_entry_size_bytes",
        "jax_persistent_cache_min_entry_size_bytes",
    )
    min_secs = _read(
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_compile_time_secs",
    )

    # ``jax.config.update`` stores the value verbatim; we asserted the
    # source pin in probe 1, here we lock the runtime read-back.
    assert cache_dir == ".jax_cache", f"cache dir read-back: {cache_dir!r}"
    assert min_bytes == -1, f"min_entry_size_bytes read-back: {min_bytes!r}"
    assert min_secs == 0.0, f"min_compile_time_secs read-back: {min_secs!r}"
