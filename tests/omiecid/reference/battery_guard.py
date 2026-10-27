"""The body-inspecting guard over the battery's broken-fixture tests (V14, AC 2).

``uncovered(source)`` parses a test module with :mod:`ast` and returns the
:data:`REQUIRED_BROKEN_FIXTURES` entries the module does **not** cover.

A test body covers ``(ID, field)`` only when **all** of these hold:

* it is a ``test_*`` function at module level or inside a ``Test*`` class, and it
  is not decorated with ``skip``, ``skipif`` or ``xfail``;
* it contains ``assert_check_fails(<arg0>, "<ID>", field="<field>")`` as a
  **top-level statement** of its body: an ``ast.Expr`` whose value is that call,
  sitting directly in the function's ``body`` list (V14 as amended at v5.186).
  A call inside any compound statement (``if``, ``try``, ``with``, ``for``,
  ``while``, ``match``, an inner ``def`` / ``lambda`` / ``class``) never counts,
  nor does a call after a top-level ``return``, ``raise`` or
  ``pytest.skip`` / ``pytest.xfail`` / ``pytest.fail`` call;
* ``<arg0>`` is a call to ``CHECKS[ID]``'s function, or a name bound to such a
  call by an earlier top-level ``ast.Assign`` of the same body;
* the call's ``status`` is ``"invalid"`` for I8 and ``"fail"`` (the default) for
  every other check.

A call with ``field="f"`` covers both ``(ID, "f")`` and ``(ID, None)``. Test names
are never read.
"""

from __future__ import annotations

import ast
from typing import Dict, Optional, Set, Tuple

from tests.omiecid.reference.omie_invariants import CHECKS

_RB_FIELDS = ("pre_snapshots", "post_snapshots", "fills", "filled", "still_resting",
              "hidden_remaining", "rejections", "engine_reports", "evicted",
              "observed_capacity", "engine_settlements", "cancelled_or_expired", "flags",
              "accepted", "participants")

REQUIRED_BROKEN_FIXTURES: Tuple[Tuple[str, Optional[str]], ...] = (
    tuple((cid, None) for cid in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9",
                                  "M1", "M2", "M3", "M4", "M5", "M6"))
    + (
        ("I2", "evicted"),
        ("I2", "cancelled_or_expired"),
        ("I6", "engine_settlements"),
        ("I6", "engine_reports"),
        ("I5", "unruled_hop"),
        ("I5", "rule3_hop"),
        ("I1", "fill_sides"),
        ("I4", "hop_priority"),
        ("I5", "limit_cross"),
    )
    + tuple(("RB", f) for f in _RB_FIELDS)
)

#: The status each check's broken fixture must assert.
_REQUIRED_STATUS = {cid: ("invalid" if cid == "I8" else "fail") for cid in CHECKS}

#: function name -> check id
_FUNC_TO_ID: Dict[str, str] = {fn.__name__: cid for cid, fn in CHECKS.items()}

_SKIP_DECORATORS = {"skip", "skipif", "xfail"}
_TERMINATING_CALLS = {"skip", "xfail", "fail"}


def _callee_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_skipped(fn: ast.AST) -> bool:
    for dec in getattr(fn, "decorator_list", ()):
        node = dec.func if isinstance(dec, ast.Call) else dec
        if _callee_name(node) in _SKIP_DECORATORS:
            return True
    return False


def _check_id_of_call(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Call):
        return _FUNC_TO_ID.get(_callee_name(node.func) or "")
    return None


def _terminates(stmt: ast.stmt) -> bool:
    if isinstance(stmt, (ast.Return, ast.Raise)):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        func = stmt.value.func
        if (isinstance(func, ast.Attribute) and func.attr in _TERMINATING_CALLS
                and isinstance(func.value, ast.Name) and func.value.id == "pytest"):
            return True
    return False


def _const_str(node: Optional[ast.AST]) -> Tuple[bool, Optional[str]]:
    """(ok, value) for a string constant or ``None`` literal."""
    if node is None:
        return True, None
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
        return True, node.value
    return False, None


def _body_coverage(fn: ast.AST) -> Set[Tuple[str, Optional[str]]]:
    covered: Set[Tuple[str, Optional[str]]] = set()
    bound: Dict[str, str] = {}
    for stmt in fn.body:
        if _terminates(stmt):
            break
        if isinstance(stmt, ast.Assign):
            cid = _check_id_of_call(stmt.value)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    if cid is not None and len(stmt.targets) == 1:
                        bound[target.id] = cid
                    else:
                        bound.pop(target.id, None)
            continue
        if isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(stmt.target, ast.Name):
                bound.pop(stmt.target.id, None)
            continue
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            continue
        call = stmt.value
        if _callee_name(call.func) != "assert_check_fails" or len(call.args) < 2:
            continue
        arg0, arg1 = call.args[0], call.args[1]
        if not (isinstance(arg1, ast.Constant) and isinstance(arg1.value, str)):
            continue
        cid = arg1.value
        got = bound.get(arg0.id) if isinstance(arg0, ast.Name) else _check_id_of_call(arg0)
        if got != cid or len(call.args) > 2 or any(k.arg is None for k in call.keywords):
            continue
        kw = {k.arg: k.value for k in call.keywords}
        ok_f, field = _const_str(kw.get("field"))
        ok_s, status = _const_str(kw.get("status"))
        if not (ok_f and ok_s):
            continue
        status = "fail" if status is None else status
        if status != _REQUIRED_STATUS.get(cid, "fail"):
            continue
        covered.add((cid, None))
        if field is not None:
            covered.add((cid, field))
    return covered


def _test_functions(body):
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") and not _is_skipped(node):
                yield node
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test") and not _is_skipped(node):
                yield from _test_functions(node.body)


def covered(source: str) -> Set[Tuple[str, Optional[str]]]:
    """Every ``(ID, field)`` some non-skipped test body of ``source`` covers."""
    tree = ast.parse(source)
    out: Set[Tuple[str, Optional[str]]] = set()
    for fn in _test_functions(tree.body):
        out |= _body_coverage(fn)
    return out


def uncovered(source: str) -> Set[Tuple[str, Optional[str]]]:
    """The :data:`REQUIRED_BROKEN_FIXTURES` entries ``source`` leaves uncovered."""
    return set(REQUIRED_BROKEN_FIXTURES) - covered(source)
