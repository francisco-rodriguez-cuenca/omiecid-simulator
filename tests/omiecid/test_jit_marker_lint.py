"""AST-lint test for ``@pytest.mark.jit`` discipline (Story T1.1).

This module is deliberately self-contained: it imports only the standard
library (``ast``, ``pathlib``, ``tokenize``, ``io``, ``textwrap``) plus
``pytest``.  In particular it MUST NOT ``import jax`` — the lint runs in
the default suite (sub-second target) and must not pull in JAX's
top-level import cost on every collection.

The lint walks every ``tests/omiecid/test_*.py`` file and flags any
``jax.jit`` / ``jax.pmap`` *call* or *decorator* whose enclosing test
function does not have a ``pytest.mark.jit`` marker in scope.  Markers
in scope are (in order of resolution):

  - decorator on the function (``@pytest.mark.jit``)
  - class-level ``pytestmark = pytest.mark.jit`` on the enclosing class
    (or a list/tuple containing it)
  - module-level ``pytestmark = pytest.mark.jit`` (or list/tuple)

A ``# noqa: jit-marker`` comment on the offending source line acts as
an escape hatch; the lint records but does not flag it.

Out of scope (intentionally not flagged):
  - ``jax.vmap`` — vmap traces but does not invoke the persistent cache,
    so it falls outside the cache-discipline contract.
  - ``lax.scan`` / ``lax.while_loop`` *inside* a jitted function — too
    noisy without a full call-graph walker.

The module also ships its own synthetic-fixture self-tests (the
``Test*`` classes below) covering the five flag/clear/noqa/scope cases
required by the T1.1 story card.
"""

from __future__ import annotations

import ast
import io
import pathlib
import textwrap
import tokenize
from dataclasses import dataclass
from typing import Iterable

import pytest


pytestmark = pytest.mark.jit  # this lint module's own tests trigger no
                              # JAX compiles, but marking it keeps the
                              # T1.1 story-card discipline self-consistent
                              # (the self-tests below construct synthetic
                              #  source strings, never actually call jax).
# noqa: jit-marker  (this docstring/comment block is module-level, not a
#                    real jit call site — the linter does not parse it.)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Names that, when called or used as decorators on a top-level ``jax``
# attribute access, require ``pytest.mark.jit`` in scope.  Restricted to
# the JIT-cache-relevant primitives (``jit`` and ``pmap``); ``vmap`` is
# explicitly not flagged because it does not invoke the persistent cache.
_FLAGGED_NAMES = frozenset({"jit", "pmap"})

_NOQA_TAG = "noqa: jit-marker"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintViolation:
    """One offending jit/pmap site with no ``pytest.mark.jit`` in scope."""

    file: str
    lineno: int
    name: str               # "jit" or "pmap"
    kind: str               # "call" or "decorator"
    enclosing: str          # function name (or "<module>" if at module level)


# ---------------------------------------------------------------------------
# Core lint logic
# ---------------------------------------------------------------------------


def _attr_chain_endswith_jax(node: ast.expr) -> tuple[bool, str | None]:
    """Return (True, attr_name) if ``node`` is ``jax.<NAME>`` (any depth).

    Accepts:
      - ``jax.jit``                  -> (True, "jit")
      - ``jax.pmap``                 -> (True, "pmap")
      - ``jax.vmap``                 -> (True, "vmap")  (caller filters)
      - ``some_module.jax.jit``      -> (True, "jit")   (rare; tolerated)

    Rejects:
      - ``Name``                     -> (False, None)
      - ``foo.bar`` where neither node references ``jax``
      - ``jit`` (bare name from ``from jax import jit``) — by spec we only
        flag attribute-access chains rooted at ``jax``.

    The "any depth" allowance is deliberate: it tolerates re-export
    patterns like ``mymod.jax.jit`` without false-negatives, while the
    bare-name ``jit`` is intentionally not flagged because it would
    trigger on unrelated identifiers and we prefer false-negatives over
    false-positives in a lint that's meant to act as a soft gate.
    """
    if not isinstance(node, ast.Attribute):
        return (False, None)

    attr_name = node.attr
    # Walk up the chain looking for a Name("jax") base.
    current: ast.expr = node.value
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name) and current.id == "jax":
        return (True, attr_name)
    return (False, None)


def _is_jit_marker(node: ast.expr) -> bool:
    """Return True if ``node`` is ``pytest.mark.jit``.

    Tolerates either the attribute form (``pytest.mark.jit``) or the call
    form (``pytest.mark.jit()`` — pytest does not require the parens but
    some authors add them).
    """
    if isinstance(node, ast.Call):
        return _is_jit_marker(node.func)
    if isinstance(node, ast.Attribute) and node.attr == "jit":
        if isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
            base = node.value.value
            if isinstance(base, ast.Name) and base.id == "pytest":
                return True
    return False


def _pytestmark_contains_jit(value: ast.expr) -> bool:
    """Return True if a ``pytestmark = ...`` RHS includes ``pytest.mark.jit``.

    Handles three RHS forms:
      - ``pytest.mark.jit``                      (single marker)
      - ``[pytest.mark.jit, pytest.mark.slow]``  (list of markers)
      - ``(pytest.mark.jit,)``                   (tuple of markers)
    """
    if _is_jit_marker(value):
        return True
    if isinstance(value, (ast.List, ast.Tuple)):
        return any(_is_jit_marker(elt) for elt in value.elts)
    return False


def _module_has_jit_pytestmark(tree: ast.Module) -> bool:
    """Return True if the module assigns ``pytestmark`` containing jit."""
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "pytestmark":
                    if _pytestmark_contains_jit(stmt.value):
                        return True
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == "pytestmark":
                if stmt.value is not None and _pytestmark_contains_jit(stmt.value):
                    return True
    return False


def _class_has_jit_pytestmark(cls: ast.ClassDef) -> bool:
    """Return True if a class body assigns ``pytestmark`` containing jit."""
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "pytestmark":
                    if _pytestmark_contains_jit(stmt.value):
                        return True
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == "pytestmark":
                if stmt.value is not None and _pytestmark_contains_jit(stmt.value):
                    return True
    return False


def _function_has_jit_decorator(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if any decorator on the function is ``pytest.mark.jit``."""
    return any(_is_jit_marker(d) for d in fn.decorator_list)


def _noqa_lines(source: str) -> set[int]:
    """Return the set of 1-based line numbers carrying ``# noqa: jit-marker``.

    Implementation note: we tokenise with the ``tokenize`` module so that
    a ``# noqa: jit-marker`` substring inside a triple-quoted docstring
    or string literal is *not* treated as a real escape hatch.
    """
    out: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT and _NOQA_TAG in tok.string:
                out.add(tok.start[0])
    except tokenize.TokenizeError:
        # Best-effort: if the file fails to tokenise (it shouldn't, since
        # we already AST-parsed it) fall through with an empty noqa set.
        pass
    return out


def _walk_node(
    node: ast.AST,
    *,
    in_jit_scope: bool,
    enclosing: str,
    noqa: set[int],
    out: list[LintViolation],
    file_label: str,
) -> None:
    """Recursive AST walk emitting violations for unmarked jit/pmap sites.

    ``in_jit_scope`` carries the marker-resolution state inherited from
    the module / class / function context.
    """
    # Function definitions: descend into the body with a possibly-elevated
    # in_jit_scope flag (decorator on this function counts as a marker).
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # First, scan the decorators themselves for jit/pmap-as-decorator
        # *uses* (not for the marker — which is handled below).
        for dec in node.decorator_list:
            _check_decorator(dec, in_jit_scope, node.name, noqa, out, file_label)

        fn_in_scope = in_jit_scope or _function_has_jit_decorator(node)
        for child in node.body:
            _walk_node(
                child,
                in_jit_scope=fn_in_scope,
                enclosing=node.name,
                noqa=noqa,
                out=out,
                file_label=file_label,
            )
        return

    # Class definitions: descend with a possibly-elevated scope (class
    # pytestmark counts).  The class itself is not a "test" but methods
    # inside it might be.
    if isinstance(node, ast.ClassDef):
        cls_in_scope = in_jit_scope or _class_has_jit_pytestmark(node)
        for child in node.body:
            _walk_node(
                child,
                in_jit_scope=cls_in_scope,
                enclosing=node.name,
                noqa=noqa,
                out=out,
                file_label=file_label,
            )
        return

    # Calls: check whether ``node.func`` is jax.jit / jax.pmap.
    if isinstance(node, ast.Call):
        _check_call(node, in_jit_scope, enclosing, noqa, out, file_label)
        # Fall through and recurse into children (call args may contain
        # nested calls or decorators we still need to scan).

    # Generic recursion.
    for child in ast.iter_child_nodes(node):
        _walk_node(
            child,
            in_jit_scope=in_jit_scope,
            enclosing=enclosing,
            noqa=noqa,
            out=out,
            file_label=file_label,
        )


def _check_call(
    node: ast.Call,
    in_jit_scope: bool,
    enclosing: str,
    noqa: set[int],
    out: list[LintViolation],
    file_label: str,
) -> None:
    is_jax, name = _attr_chain_endswith_jax(node.func)
    if not is_jax or name not in _FLAGGED_NAMES:
        return
    if in_jit_scope:
        return
    if node.lineno in noqa:
        return
    out.append(
        LintViolation(
            file=file_label,
            lineno=node.lineno,
            name=name,
            kind="call",
            enclosing=enclosing,
        )
    )


def _check_decorator(
    dec: ast.expr,
    in_jit_scope: bool,
    enclosing: str,
    noqa: set[int],
    out: list[LintViolation],
    file_label: str,
) -> None:
    """Flag ``@jax.jit`` / ``@jax.pmap`` used as a decorator without marker."""
    # Decorator can be ``@jax.jit`` (Attribute) or ``@jax.jit(...)`` (Call).
    target: ast.expr = dec.func if isinstance(dec, ast.Call) else dec
    is_jax, name = _attr_chain_endswith_jax(target)
    if not is_jax or name not in _FLAGGED_NAMES:
        return
    if in_jit_scope:
        return
    if dec.lineno in noqa:
        return
    out.append(
        LintViolation(
            file=file_label,
            lineno=dec.lineno,
            name=name,
            kind="decorator",
            enclosing=enclosing,
        )
    )


def lint_source(source: str, *, file_label: str = "<inline>") -> list[LintViolation]:
    """Lint a single source string and return all violations.

    Public entry point: takes raw Python source, returns a list of
    :class:`LintViolation` records (empty if the source is clean).

    This is the function exercised by the synthetic-fixture self-tests.
    """
    tree = ast.parse(source)
    module_in_scope = _module_has_jit_pytestmark(tree)
    noqa = _noqa_lines(source)
    out: list[LintViolation] = []
    for stmt in tree.body:
        _walk_node(
            stmt,
            in_jit_scope=module_in_scope,
            enclosing="<module>",
            noqa=noqa,
            out=out,
            file_label=file_label,
        )
    return out


def lint_file(path: pathlib.Path) -> list[LintViolation]:
    """Lint a single file on disk and return its violations."""
    source = path.read_text(encoding="utf-8")
    return lint_source(source, file_label=str(path))


def _omie_test_files() -> Iterable[pathlib.Path]:
    """Yield every ``tests/omiecid/test_*.py`` file (excluding this lint module)."""
    here = pathlib.Path(__file__).resolve().parent
    self_path = pathlib.Path(__file__).resolve()
    for p in sorted(here.glob("test_*.py")):
        if p.resolve() == self_path:
            continue
        yield p


# ---------------------------------------------------------------------------
# Real-file lint test (the gate)
# ---------------------------------------------------------------------------


def test_omie_suite_jit_marker_discipline():
    """Every jit/pmap call/decorator in tests/omiecid/ has a marker in scope.

    This is the production test gate.  If it fails, the failure message
    points at the offending file, line, and enclosing function so the
    developer can either add ``pytest.mark.jit`` (preferred) or, in the
    rare case the call genuinely should not be marked, append
    ``# noqa: jit-marker`` to the line.
    """
    all_violations: list[LintViolation] = []
    for path in _omie_test_files():
        all_violations.extend(lint_file(path))

    if all_violations:
        msg_lines = ["jit/pmap calls without `pytest.mark.jit` in scope:"]
        for v in all_violations:
            msg_lines.append(
                f"  {v.file}:{v.lineno}  jax.{v.name} ({v.kind}) "
                f"in {v.enclosing}"
            )
        msg_lines.append(
            "Fix: add `pytestmark = pytest.mark.jit` at module scope, "
            "or `@pytest.mark.jit` on the test, or `# noqa: jit-marker` "
            "on the offending line."
        )
        pytest.fail("\n".join(msg_lines), pytrace=False)


def test_lint_module_does_not_import_jax():
    """T1.1 AC #8 — this lint module must not import jax.

    Rationale: the lint runs in every default-suite invocation; pulling
    in JAX would defeat the ``pytest -m "not jit"`` fast-path.  We check
    by parsing this file's own AST and verifying no ``import jax`` /
    ``from jax import ...`` statements exist.
    """
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                assert not alias.name.startswith("jax"), (
                    f"test_jit_marker_lint.py must not import jax "
                    f"(found `import {alias.name}`)"
                )
        elif isinstance(stmt, ast.ImportFrom):
            assert not (stmt.module or "").startswith("jax"), (
                f"test_jit_marker_lint.py must not import from jax "
                f"(found `from {stmt.module} import ...`)"
            )


# ---------------------------------------------------------------------------
# Synthetic-fixture self-tests
#
# These exercise the lint engine on tiny inline source strings to prove
# the flag/clear/noqa/scope cases all behave as the T1.1 story card
# requires.  Five required fixtures (story-card §6):
#   (a) jit call without marker -> flagged
#   (b) jit call with module pytestmark -> passes
#   (c) jit call with `noqa: jit-marker` -> passes
#   (d) jax.vmap call without marker -> not flagged (out of scope)
#   (e) jit decorator without marker -> flagged
# Plus a few additional self-tests that lock the scope-resolution logic
# (function-decorator marker, class pytestmark, list-form pytestmark).
# ---------------------------------------------------------------------------


class TestSelfFixtures:
    """Synthetic-fixture self-tests for the lint engine."""

    def test_a_jit_call_without_marker_is_flagged(self):
        """Fixture (a): unmarked jit call is flagged."""
        src = textwrap.dedent(
            """
            import jax

            def test_thing():
                f = jax.jit(lambda x: x + 1)
                f(1)
            """
        )
        violations = lint_source(src)
        assert len(violations) == 1
        assert violations[0].name == "jit"
        assert violations[0].kind == "call"
        assert violations[0].enclosing == "test_thing"

    def test_b_jit_call_with_module_pytestmark_passes(self):
        """Fixture (b): module-level pytestmark clears the violation."""
        src = textwrap.dedent(
            """
            import jax
            import pytest

            pytestmark = pytest.mark.jit

            def test_thing():
                f = jax.jit(lambda x: x + 1)
                f(1)
            """
        )
        assert lint_source(src) == []

    def test_c_jit_call_with_noqa_passes(self):
        """Fixture (c): a `# noqa: jit-marker` comment clears the violation."""
        src = textwrap.dedent(
            """
            import jax

            def test_thing():
                f = jax.jit(lambda x: x + 1)  # noqa: jit-marker
                f(1)
            """
        )
        assert lint_source(src) == []

    def test_d_jax_vmap_without_marker_is_not_flagged(self):
        """Fixture (d): jax.vmap is intentionally out of scope."""
        src = textwrap.dedent(
            """
            import jax

            def test_thing():
                f = jax.vmap(lambda x: x + 1)
                f([1, 2, 3])
            """
        )
        assert lint_source(src) == []

    def test_e_jit_decorator_without_marker_is_flagged(self):
        """Fixture (e): @jax.jit on a function with no marker is flagged."""
        src = textwrap.dedent(
            """
            import jax

            def test_thing():
                @jax.jit
                def inner(x):
                    return x + 1
                inner(1)
            """
        )
        violations = lint_source(src)
        assert len(violations) == 1
        assert violations[0].name == "jit"
        assert violations[0].kind == "decorator"

    # ---- Additional scope-resolution locks ----

    def test_function_decorator_marker_clears(self):
        """`@pytest.mark.jit` on the function clears the violation."""
        src = textwrap.dedent(
            """
            import jax
            import pytest

            @pytest.mark.jit
            def test_thing():
                f = jax.jit(lambda x: x + 1)
                f(1)
            """
        )
        assert lint_source(src) == []

    def test_class_pytestmark_clears(self):
        """Class-level pytestmark clears violations in methods."""
        src = textwrap.dedent(
            """
            import jax
            import pytest

            class TestThing:
                pytestmark = pytest.mark.jit

                def test_method(self):
                    f = jax.jit(lambda x: x + 1)
                    f(1)
            """
        )
        assert lint_source(src) == []

    def test_pytestmark_list_form_clears(self):
        """`pytestmark = [pytest.mark.jit, pytest.mark.slow]` clears."""
        src = textwrap.dedent(
            """
            import jax
            import pytest

            pytestmark = [pytest.mark.jit, pytest.mark.slow]

            def test_thing():
                f = jax.jit(lambda x: x + 1)
                f(1)
            """
        )
        assert lint_source(src) == []

    def test_pmap_call_is_flagged(self):
        """`jax.pmap` is also in scope (per story card)."""
        src = textwrap.dedent(
            """
            import jax

            def test_thing():
                f = jax.pmap(lambda x: x + 1)
                f([1])
            """
        )
        violations = lint_source(src)
        assert len(violations) == 1
        assert violations[0].name == "pmap"

    def test_noqa_only_clears_its_own_line(self):
        """A `# noqa: jit-marker` on line N does not silence line M (M != N)."""
        src = textwrap.dedent(
            """
            import jax

            def test_thing():
                f = jax.jit(lambda x: x + 1)  # noqa: jit-marker
                g = jax.jit(lambda y: y * 2)
                f(1)
                g(2)
            """
        )
        violations = lint_source(src)
        assert len(violations) == 1
        # The unmarked one is on the second jit call.
        assert violations[0].kind == "call"

    def test_noqa_inside_string_is_not_an_escape_hatch(self):
        """`noqa: jit-marker` inside a string literal is not honoured."""
        src = textwrap.dedent(
            '''
            import jax

            def test_thing():
                msg = "noqa: jit-marker"
                f = jax.jit(lambda x: x + 1)
                f(1)
                _ = msg
            '''
        )
        violations = lint_source(src)
        # The string-embedded noqa must NOT silence the violation.
        assert len(violations) == 1
