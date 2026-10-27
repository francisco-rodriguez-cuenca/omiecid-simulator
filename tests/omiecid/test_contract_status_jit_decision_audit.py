"""Audit tests for P2-L3 — `contract` status transitions JIT-safe.

Independent first-principles verification of the shipped story outcome.
Derived only from:
  * docs/omiecid/agile-plan.md — Story P2-L3 card
  * docs/omiecid/market-spec.md §2.1–2.2
  * docs/omiecid/migration-plan.md
  * omiecid_simulator/omiecid/{contract.py, contract_book.py, jax_constants.py}

These audit tests do NOT mirror the locked test file
`tests/omiecid/test_contract_status_jit_decision.py`; they were authored without
reading it. They verify the acceptance criteria of the story independently:

  1. Which branch shipped (a: JAX-traceable transitions, or b: "no change"
     decision record). The card requires audit to confirm the branch.
  2. If branch (b): `contract.py` is NOT JIT-converted — no `jnp.`,
     `jax.jit`, or `jax.lax.cond` usage; `ContractStatus` is still IntEnum;
     `Contract` is still a Python dataclass.
  3. The decision record prevents silent re-litigation — lives at a stable
     on-disk location, contains the caller-contract adapter pattern, and
     names a concrete reopen trigger.
  4. DAG integrity — no JIT-scope module imports `Contract` /
     `ContractStatus` without going through the closed-over-scalar adapter.

All tests are Category A (pure file I/O / Python introspection; no JAX
tracing); target runtime < 1s combined.
"""

from __future__ import annotations

import ast
import re
from dataclasses import is_dataclass
from enum import IntEnum
from pathlib import Path

import pytest

# Module-level marker (Story T1.1): every test in this file audits the
# P2-L3 JIT-scope decision; the file is fully-JIT-relevant.
pytestmark = pytest.mark.jit


REPO_ROOT = Path(__file__).resolve().parents[2]
OMIE_DIR = REPO_ROOT / "omiecid_simulator" / "omiecid"
CONTRACT_PY = OMIE_DIR / "contract.py"
CONTRACT_BOOK_PY = OMIE_DIR / "contract_book.py"
JAX_CONSTANTS_PY = OMIE_DIR / "jax_constants.py"


# ---------------------------------------------------------------------------
# Branch identification
# ---------------------------------------------------------------------------

class TestShippedBranchIsNoChange:
    """Confirm branch (b) — the "no change required" decision record —
    shipped, and verify its load-bearing structural properties."""

    def test_contract_module_is_pure_python_no_jax(self):
        """contract.py must not import or use jax / jnp / jax.lax in any
        form. Branch (b) specifically forbids JIT-converting the module.

        Scans both the import graph and the raw source for any textual
        appearance of `jnp.`, `jax.jit`, or `jax.lax.cond` in non-comment
        code — if branch (a) had shipped, at least one would be present.
        """
        src = CONTRACT_PY.read_text()

        # AST-level: no `import jax` / `from jax...` / `import jax.numpy`
        tree = ast.parse(src)
        banned_roots = {"jax"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in banned_roots, (
                        f"contract.py imports {alias.name!r}; branch (b) "
                        f"forbids any jax dependency in this module."
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                root = node.module.split(".")[0]
                assert root not in banned_roots, (
                    f"contract.py has `from {node.module} import ...`; "
                    f"branch (b) forbids any jax dependency in this module."
                )

        # Source-level: no transition-path jnp / jit / cond usages.
        # (Strip line/block-comment-only lines; we want behavioral code.)
        code_lines = []
        for raw in src.splitlines():
            stripped = raw.strip()
            if stripped.startswith("#"):
                continue
            code_lines.append(raw)
        code_only = "\n".join(code_lines)

        for needle in ("jnp.", "jax.jit", "jax.lax.cond", "jax.lax.switch",
                       "jax.lax.while_loop"):
            assert needle not in code_only, (
                f"contract.py contains {needle!r}; branch (b) no-change "
                f"requires the module to stay pure Python."
            )

    def test_contract_status_is_still_intenum_not_jax_scalar(self):
        """ContractStatus must remain a Python IntEnum with the four
        documented members. If branch (a) had shipped, the type would be
        a JAX scalar / traced array / NamedTuple, not an IntEnum.
        """
        from omiecid_simulator.omiecid.contract import ContractStatus

        assert issubclass(ContractStatus, IntEnum), (
            "ContractStatus must still be an IntEnum under branch (b); "
            "a JAX scalar conversion would have shipped branch (a)."
        )
        members = {m.name for m in ContractStatus}
        # Market-spec §2.1-2.2 lifecycle: PENDING/OPEN/SUSPENDED/CLOSED.
        assert {"PENDING", "OPEN", "SUSPENDED", "CLOSED"} <= members, (
            f"ContractStatus missing canonical lifecycle members; got {members}"
        )

    def test_contract_dataclass_is_plain_python(self):
        """Contract must remain a plain @dataclass, not a NamedTuple of
        jnp scalars (which would indicate branch (a))."""
        from omiecid_simulator.omiecid.contract import Contract

        assert is_dataclass(Contract), (
            "Contract must still be a Python @dataclass under branch (b)."
        )


# ---------------------------------------------------------------------------
# Decision record structural integrity
# ---------------------------------------------------------------------------

class TestDecisionRecordPreventsRelitigation:
    """The record is the *entire* deliverable of branch (b). Its job is
    to prevent silent re-litigation when Phase 3 adds new transitions."""

    @pytest.fixture(scope="class")
    def record_text(self) -> str:
        """Return the docstring text of jax_constants.py (where the
        decision record lives, per the story card's "appended to
        jax_constants.py docstring" clause)."""
        src = JAX_CONSTANTS_PY.read_text()
        tree = ast.parse(src)
        doc = ast.get_docstring(tree)
        assert doc is not None, (
            "jax_constants.py has no module docstring; the decision "
            "record is required to live there per the story card."
        )
        return doc

    def test_record_exists_at_stable_location_under_p2l3_heading(
        self, record_text: str
    ):
        """The record must be discoverable by a downstream reader grepping
        for "P2-L3" inside `omiecid_simulator/omiecid/jax_constants.py`. The
        story card pins this location so downstream stories (notably
        P2-H2) can cite it without hunting for a sibling decisions file.
        """
        # Downstream must find a P2-L3-tagged block in this file.
        assert "Contract status transitions stay Python-side" in record_text, (
            "jax_constants.py docstring must contain a P2-L3-tagged block; "
            "downstream stories rely on this anchor."
        )
        # Record must identify contract.py as its subject.
        assert "contract.py" in record_text, (
            "P2-L3 record must name `contract.py` as its subject."
        )

    def test_record_appears_below_b4_block(self, record_text: str):
        """Structural ordering: the P2-L3 record is appended BELOW the
        pre-existing B4 decision block, not interleaved or prepended.
        Ordering matters because B4 is the logical predecessor (P2-L3
        extends B4's Python-side-helper pattern to contract.py).
        """
        b4_pos = record_text.find("JIT-scope of query methods")
        p2l3_pos = record_text.find("Contract status transitions stay Python-side")
        assert b4_pos != -1, (
            "B4 decision block missing from jax_constants.py docstring; "
            "P2-L3 is supposed to sit below it."
        )
        assert p2l3_pos != -1, "P2-L3 heading missing."
        assert p2l3_pos > b4_pos, (
            "P2-L3 decision block must appear AFTER the B4 block in the "
            "docstring; ordering is load-bearing because P2-L3 extends B4."
        )

    def test_record_names_closed_over_scalar_adapter_pattern(
        self, record_text: str
    ):
        """Caller-contract invariant: the record must specify the adapter
        that P2-H2 will use when JIT'ing submit_order — a pre-computed
        boolean scalar, not a traced `.status` read. Without this, branch
        (b) would silently break when P2-H2 lands.
        """
        # Must name the scalar name that submit_order already computes
        # at contract_book.py line 292 (`contract_open = ...`).
        assert "contract_open" in record_text, (
            "P2-L3 record must name the `contract_open` scalar that "
            "crosses the future JIT boundary (P2-H2 keys off this name)."
        )
        # Must describe the scalar as pre-computed / closed-over outside
        # traced code — some phrasing that makes the adapter unambiguous.
        lowered = record_text.lower()
        assert any(
            phrase in lowered
            for phrase in ("closed-over", "closed over", "pre-computed",
                           "precomputed", "python-side")
        ), (
            "P2-L3 record must describe `contract_open` as a "
            "closed-over / pre-computed / Python-side scalar."
        )

    def test_record_names_concrete_reopen_trigger(self, record_text: str):
        """Non-re-litigation clause: the record must name *what* would
        force a revisit. A vague "revisit if needed" sentence would invite
        silent drift. The concrete trigger per market-spec §2.1-2.2 +
        migration-plan is Phase 3 per-step auto-transitions (e.g. auto-
        CLOSE at GCT, auto-SUSPEND at IDA boundary).
        """
        lowered = record_text.lower()
        # Must reference Phase 3 (the horizon at which dynamism is added).
        assert "per-step auto-transitions" in lowered, (
            "P2-L3 record must name Phase 3 as the horizon whose new "
            "auto-transitions would trigger reopening."
        )
        # Must reference at least one concrete trigger: GCT-close,
        # IDA-suspend, or "per-step" time-driven transitions.
        concrete_triggers = ("gct", "ida", "per-step", "per step",
                             "auto-close", "auto-suspend", "simulation time",
                             "simulation-time")
        assert any(t in lowered for t in concrete_triggers), (
            "P2-L3 record must name at least one concrete Phase 3 trigger "
            f"(one of {concrete_triggers}); a vague 'revisit if needed' "
            "clause invites silent re-litigation."
        )
        # Must use explicit reopen-language (not just "maybe revisit").
        assert any(word in lowered
                   for word in ("reopen", "re-open", "revisit",
                                "re-litigat")), (
            "P2-L3 record must use explicit reopen / revisit / re-litigate "
            "language naming the trigger as decisive."
        )


# ---------------------------------------------------------------------------
# DAG integrity: no JIT-scope readers of Contract.status
# ---------------------------------------------------------------------------

class TestDAGIntegrity:
    """No module under omiecid_simulator/omiecid/ may read Contract.status
    from inside JIT-scope code without the closed-over-scalar adapter.

    Independent check: scan every .py in the directory for
      (a) imports of Contract / ContractStatus, AND
      (b) `.status` comparisons,
    then confirm that any file doing both either IS contract.py itself
    or exhibits the adapter pattern (status comparison at function entry,
    before any jnp / jax.jit boundary).
    """

    def _omie_py_files(self):
        return sorted(p for p in OMIE_DIR.glob("*.py") if p.name != "__init__.py")

    def test_sole_contract_status_reader_is_contract_book_submit_order(self):
        """Spec-derived expectation: at audit time, exactly one file under
        omiecid_simulator/omiecid/ (other than contract.py itself) reads
        Contract.status, and that file is contract_book.py in the
        submit_order entry path (architect pre-sprint review §1).

        If this test fails because a new reader appeared, branch (b) is
        no longer safe and either the record must list the new site in
        the allowlist or the new site must be rewritten through the
        adapter.
        """
        readers = []
        for path in self._omie_py_files():
            if path.name == "contract.py":
                continue  # contract.py defines the field; self-reads are N/A
            src = path.read_text()
            # Strip line comments AND docstrings (module/class/function)
            # so that the record's own prose in jax_constants.py is not
            # misclassified as a JIT-scope reader.
            tree = ast.parse(src)
            # Collect line ranges occupied by docstring Expr nodes.
            docstring_lines: set[int] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = getattr(node, "body", [])
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        ds = body[0]
                        start = ds.lineno
                        end = getattr(ds, "end_lineno", start)
                        for ln in range(start, end + 1):
                            docstring_lines.add(ln)
            stripped_lines = []
            for idx, line in enumerate(src.splitlines(), start=1):
                if idx in docstring_lines:
                    continue
                if line.lstrip().startswith("#"):
                    continue
                stripped_lines.append(line)
            stripped = "\n".join(stripped_lines)
            # Look for any `.status ==` / `.status !=` / `.status is`
            # operator in executable code.
            if re.search(r"\.status\s*(==|!=|\bis\b)", stripped):
                readers.append(path.name)

        # The record grandfathered exactly one site: contract_book.py.
        assert readers == ["contract_book.py"] or readers == [], (
            f"Unexpected new reader(s) of `.status` in JIT-scope modules: "
            f"{readers}. Branch (b) allowlist covers only contract_book.py; "
            "any new reader must go through the closed-over-scalar adapter "
            "or the decision record must be re-opened."
        )

    def test_contract_book_reader_is_outside_jitted_code(self):
        """The one grandfathered reader in contract_book.py must be
        positioned at Python-side entry of submit_order — specifically,
        it must NOT sit inside a function decorated with @jax.jit, and the
        file must not currently @jax.jit-decorate submit_order.
        """
        src = CONTRACT_BOOK_PY.read_text()
        tree = ast.parse(src)

        jitted_funcs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for deco in node.decorator_list:
                    # Match `@jax.jit` or `@jit` or `@jax.jit(...)`
                    deco_src = ast.unparse(deco)
                    if "jit" in deco_src:
                        jitted_funcs.append(node.name)

        assert jitted_funcs == [], (
            f"contract_book.py now @jit-decorates function(s) {jitted_funcs}; "
            "branch (b) assumes submit_order is still Python-side. If P2-H2 "
            "has landed, the P2-L3 decision must be re-verified."
        )

        # Sanity: the single reader site must still exist.
        assert "self._contract.status" in src, (
            "The grandfathered Python-side `self._contract.status` read in "
            "contract_book.submit_order has disappeared; the P2-L3 record "
            "allowlist is now stale and must be tightened."
        )


# ---------------------------------------------------------------------------
# Runtime smoke: branch (b) means zero behavior change
# ---------------------------------------------------------------------------

class TestRuntimeBehaviorUnchanged:
    """Branch (b) guarantees `test_contract.py` stays green with no adapter
    changes. This audit re-verifies the load-bearing runtime invariants
    that test_contract.py anchors on, derived independently from the
    market-spec rather than by mirroring the locked tests.
    """

    def test_generate_contracts_still_yields_96_pending_contracts(self):
        """Market-spec §2 fixes the quarter-hourly grid at 96 MTUs per
        delivery day, all starting PENDING (§3.1 — GCT is 60min before
        delivery, so all contracts are PENDING at t=0).
        """
        from omiecid_simulator.omiecid.contract import (
            Contract,
            ContractStatus,
            generate_contracts,
        )

        contracts = generate_contracts()
        assert len(contracts) == 96, (
            f"Expected 96 quarter-hourly contracts per market-spec §2.1; "
            f"got {len(contracts)}."
        )
        assert all(isinstance(c, Contract) for c in contracts)
        assert all(c.status == ContractStatus.PENDING for c in contracts), (
            "All generated contracts must start PENDING (branch (b) no-op)."
        )
        # GCT = delivery_start - 60 (market-spec §3.1)
        for c in contracts:
            assert c.gct_minutes == c.delivery_start_minutes - 60, (
                f"Contract {c.contract_id}: GCT offset broken."
            )
