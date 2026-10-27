"""Locked behavioral tests for Story P2-L3 — `contract` status transitions JIT-safe.

Contract source:
  - docs/omiecid/agile-plan.md §"Story P2-L3 — `contract` status transitions JIT-safe"
  - docs/omiecid/market-spec.md §2.1–2.2 (four-state contract lifecycle)
  - docs/omiecid/migration-plan.md §Phase 2 Blockers (B4)
  - docs/omiecid/sprint-status.md §"P2-L3 — Architecture pre-sprint review (2026-04-24)"
  - omiecid_simulator/omiecid/jax_constants.py lines 147–170 (B4 decision record)

Deliverable branch (per story card and architect review): branch (b),
"no change required" decision record. The `contract.py` ContractStatus
IntEnum, Contract dataclass, and generate_contracts() factory stay
Python-side. Status transitions are not JIT-traced.

These two tests lock that decision on disk so that:

  1. Downstream stories cannot silently re-litigate branch (b) by deleting
     or watering down the decision record (Test #1 — documentation invariant).
  2. Downstream stories cannot silently add new JIT-scope readers of
     `Contract.status` without first introducing the closed-over-scalar
     adapter pattern specified in the decision record (Test #2 — boundary).

Both tests are Category A (pure file I/O; no JAX tracing, no fixture
loading, no model construction) and must complete well under 1 second
combined.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# Module-level marker (Story T1.1): every test in this file exercises the
# P2-L3 contract-status JIT-scope decision invariants.  Per the T1.1 lint
# discipline, fully-JIT files carry `pytestmark` at module scope rather
# than per-test decorators.
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Repository roots — resolved once, at import time, from this file's location.
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_OMIE_DIR = _REPO_ROOT / "omiecid_simulator" / "omiecid"
_JAX_CONSTANTS = _OMIE_DIR / "jax_constants.py"


# ---------------------------------------------------------------------------
# Test #1 — Documentation-invariant test.
# ---------------------------------------------------------------------------
#
# Asserts that `omiecid_simulator/omiecid/jax_constants.py` carries a P2-L3
# decision record appended *below* the existing B4 block (lines 147–170 in
# the pre-P2-L3 file). The record must contain substring anchors for each
# of the five required elements enumerated in the story card's Step 3 and
# the architect pre-sprint review §3:
#
#   (a) P2-L3 section heading + explicit "Python-side" / "no JIT" scope
#       statement covering `contract.py` status transitions.
#   (b) Caller-contract invariant naming `contract_open` as a
#       closed-over / pre-computed scalar argument.
#   (c) Cross-references to jax_constants.py §B4 AND market-spec.md §2.1–2.2.
#   (d) Non-re-litigation clause naming Phase 3 auto-transitions as the
#       trigger that could re-open the decision.
#   (e) DAG annotation forbidding JIT-scope imports of
#       `Contract` / `ContractStatus` without the adapter pattern.
#
# Each sub-assertion has an error message citing the missing element so a
# future author editing the record knows exactly which clause they broke.

class TestP2L3DecisionRecord:
    """Lock the P2-L3 'no change required' decision on disk.

    The B4 block is the anchor: the P2-L3 record must sit *below* it in the
    same file. Separating them (e.g., moving B4 or P2-L3 into a sibling
    decisions file) must fail this test — downstream stories that do that
    are re-litigating the architect's decision without an explicit review.
    """

    @pytest.fixture(scope="class")
    def text(self) -> str:
        assert _JAX_CONSTANTS.exists(), (
            f"jax_constants.py not found at {_JAX_CONSTANTS} — "
            "P2-F1/P2-F2 must land before P2-L3."
        )
        return _JAX_CONSTANTS.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def p2l3_section(self, text: str) -> str:
        """Return the file text starting at the P2-L3 heading.

        Ordering requirement: the P2-L3 record is appended *below* the
        existing B4 block, so its heading must appear after the B4 heading
        marker 'B4 decision — JIT-scope of query methods'.
        """
        b4_marker = "JIT-scope of query methods"
        b4_idx = text.find(b4_marker)
        assert b4_idx != -1, (
            f"B4 decision block not found in {_JAX_CONSTANTS}. "
            "P2-L3 requires the pre-existing B4 block as its anchor; "
            "if B4 has been moved or renamed, the P2-L3 record must be "
            "re-architected, not silently patched."
        )
        # Find the P2-L3 heading strictly after the B4 marker.
        tail = text[b4_idx:]
        match = re.search(r"Contract status transitions stay Python-side", tail)
        assert match is not None, (
            "P2-L3 decision heading not found below the B4 block in "
            f"{_JAX_CONSTANTS}. The 'no change required' record must be "
            "appended below B4 — see agile-plan.md §P2-L3 Step 3 and "
            "sprint-status.md §'P2-L3 pre-sprint review' §3."
        )
        return tail[match.start():]

    def test_scope_statement_python_side(self, p2l3_section: str) -> None:
        """Scope: contract.py stays Python-side; transitions are not JIT-traced."""
        lowered = p2l3_section.lower()
        assert "contract.py" in lowered or "`contract.py`" in p2l3_section, (
            "P2-L3 record must name `contract.py` explicitly in its scope "
            "statement (architect review §3 bullet 1)."
        )
        assert "python-side" in lowered, (
            "P2-L3 record must use the phrase 'Python-side' in its scope "
            "statement — this is the load-bearing verdict ('contract.py "
            "remains Python-side'). See sprint-status.md §P2-L3 §3 bullet 1."
        )
        # "not traced" / "no JIT" / "not JIT-traced" — any of these phrasings
        # counts, but at least one must appear to make the JIT exclusion
        # explicit rather than implied.
        jit_exclusion_phrases = [
            "not traced",
            "not jit-traced",
            "no jit",
            "not jit ",
            "no jit conversion",
        ]
        assert any(p in lowered for p in jit_exclusion_phrases), (
            "P2-L3 record must explicitly state that status transitions are "
            "NOT JIT-traced (any of: 'not traced', 'not JIT-traced', "
            "'no JIT conversion'). Implicit exclusion invites drift."
        )

    def test_caller_contract_invariant(self, p2l3_section: str) -> None:
        """Invariant: JIT consumers receive `contract_open: bool` as closed-over scalar."""
        assert "contract_open" in p2l3_section, (
            "P2-L3 record must name the `contract_open` boolean scalar "
            "explicitly — this is the caller-contract invariant P2-H2 will "
            "key off. See sprint-status.md §P2-L3 §3 bullet 2."
        )
        lowered = p2l3_section.lower()
        # At least one of 'closed-over' or 'pre-computed' must appear to
        # pin down HOW contract_open crosses the JIT boundary.
        assert ("closed-over" in lowered) or ("pre-computed" in lowered) or (
            "precomputed" in lowered
        ), (
            "P2-L3 record must describe `contract_open` as a closed-over or "
            "pre-computed scalar argument — NOT a live read of "
            "`self._contract.status` inside traced code. This is the "
            "load-bearing constraint that keeps branch (b) coherent with P2-H2."
        )
        # Must explicitly forbid the alternative (reading .status inside traced code).
        assert "self._contract.status" in p2l3_section or ".status" in p2l3_section, (
            "P2-L3 record must name the forbidden pattern "
            "(reading `.status` / `self._contract.status` inside traced code) "
            "so the invariant is actionable, not abstract."
        )

    def test_cross_references(self, p2l3_section: str) -> None:
        """Record must cite jax_constants.py §B4 and market-spec.md §2.1–2.2."""
        assert "JIT-scope of query methods" in p2l3_section, (
            "P2-L3 record must cross-reference jax_constants.py §B4 "
            "decision block. See sprint-status.md §P2-L3 §3 bullet 3."
        )
        # market-spec cross-ref: accept a few spelling variants for the
        # section range (2.1–2.2 with em dash, 2.1-2.2 with hyphen, or the
        # two section numbers mentioned near each other).
        lowered = p2l3_section.lower()
        assert "market-spec" in lowered or "market_spec" in lowered, (
            "P2-L3 record must cross-reference docs/omiecid/market-spec.md "
            "(the four-state lifecycle source). See sprint-status.md §P2-L3 §3."
        )
        has_section_range = bool(
            re.search(r"2\.1\s*[–\-]\s*2\.2", p2l3_section)
            or ("2.1" in p2l3_section and "2.2" in p2l3_section)
        )
        assert has_section_range, (
            "P2-L3 record must cite market-spec.md §2.1–2.2 specifically "
            "(four-state lifecycle). A generic pointer at the whole spec "
            "is not actionable."
        )

    def test_non_relitigation_clause(self, p2l3_section: str) -> None:
        """Must name Phase 3 auto-transitions as the re-open trigger."""
        lowered = p2l3_section.lower()
        assert "per-step auto-transitions" in lowered, (
            "P2-L3 record must name Phase 3 as the context in which this "
            "decision may be reopened. See sprint-status.md §P2-L3 §3 bullet 4."
        )
        # The clause must identify auto-transitions (IDA SUSPEND / GCT CLOSE
        # per the architect review's worked example) as the trigger.
        auto_transition_terms = [
            "auto-suspend",
            "auto-close",
            "auto-transition",
            "auto transitions",
            "per-step status",
            "simulation time",
        ]
        assert any(t in lowered for t in auto_transition_terms), (
            "P2-L3 record must name Phase 3 *auto-transitions* "
            "(e.g., 'auto-SUSPEND at IDA boundary', 'auto-CLOSE at GCT', "
            "or 'per-step status transitions driven by simulation time') "
            "as the trigger that would force reopening this decision. "
            "Without this clause the record invites silent drift the "
            "first time a new transition is added."
        )
        # And must use language that signals 'reopen / revisit' of the decision.
        reopen_terms = ["reopen", "re-open", "revisit", "re-litigate", "revised"]
        assert any(t in lowered for t in reopen_terms), (
            "P2-L3 non-re-litigation clause must use explicit reopen-language "
            "(e.g., 'must be reopened', 'revisit', 're-litigated'), not "
            "hedged wording."
        )

    def test_dag_annotation(self, p2l3_section: str) -> None:
        """Must forbid JIT-scope imports of Contract/ContractStatus without adapter."""
        # Both class names must appear so the forbidden import surface is concrete.
        assert "Contract" in p2l3_section, (
            "P2-L3 DAG annotation must name `Contract` (the dataclass) "
            "in its import-surface prohibition."
        )
        assert "ContractStatus" in p2l3_section, (
            "P2-L3 DAG annotation must name `ContractStatus` (the IntEnum) "
            "in its import-surface prohibition. See sprint-status.md "
            "§P2-L3 §3 bullet 5."
        )
        lowered = p2l3_section.lower()
        # 'import' must appear — this is the DAG-level hook the annotation is about.
        assert "import" in lowered, (
            "P2-L3 DAG annotation must reference *imports* from "
            "JIT-boundary modules — it is an import-graph constraint, not "
            "just a call-site constraint."
        )
        # Must reference 'adapter' or the closed-over-scalar pattern as the
        # sanctioned workaround.
        assert ("adapter" in lowered) or ("closed-over" in lowered) or (
            "pre-computed scalar" in lowered or "precomputed scalar" in lowered
        ), (
            "P2-L3 DAG annotation must name the sanctioned workaround "
            "(the adapter / closed-over-scalar pattern). A bare prohibition "
            "with no escape hatch leaves Phase 3 authors with no guidance."
        )


# ---------------------------------------------------------------------------
# Test #2 — Boundary test: no new `.status` comparisons in JIT-scope code.
# ---------------------------------------------------------------------------
#
# Grep-style scan of `omiecid_simulator/omiecid/*.py` asserting that no
# `.status ==` or `.status !=` comparison appears inside JIT-scope code.
#
# "JIT scope" for Phase 2, per the P2-F2 B4 decision and the migration plan,
# means: files/functions that are (or will be, per P2-H2) wrapped by
# `jax.jit` / `jax.vmap`. The two allowlisted sites are:
#
#   - `contract.py`                 — Python-side by decision (B4 + P2-L3);
#                                     the `ContractStatus` IntEnum and
#                                     `Contract` dataclass live here and
#                                     may compare their own `.status` freely.
#   - `contract_book.py:291`        — the single in-tree reader
#                                     (`submit_order` → `contract_open`);
#                                     P2-H2 will adapter-ise this line into
#                                     a closed-over scalar before JIT-ing
#                                     `submit_order`. Until then it is the
#                                     only sanctioned site.
#
# Any NEW `.status ==` / `.status !=` comparison in `omiecid_simulator/omiecid/`
# outside that allowlist fails the test — it signals a downstream story
# has silently introduced a JIT-scope status reader without the adapter
# pattern required by the P2-L3 decision record.

_STATUS_CMP_REGEX = re.compile(r"\.status\s*(==|!=)")


class TestP2L3StatusComparisonBoundary:
    """Guard the JIT boundary: no new `.status`-comparison readers in omie/.

    Allowlist rationale:
      * `contract.py` — canonical Python-side home of `ContractStatus` and
        `Contract`; comparing `.status` here is expected and explicitly
        permitted by the P2-L3 decision record (branch (b)).
      * `contract_book.py` line 291 (`self._contract.status == ContractStatus.OPEN`
        inside `submit_order`) — documented in the P2-L3 architect review
        §1 as the one existing reader; P2-H2 will convert it to a closed-over
        scalar as part of JIT-ing `submit_order`. This single line is
        grandfathered so P2-L3 can ship without touching production code.

    Any file:line outside this allowlist that contains a `.status ==` or
    `.status !=` comparison constitutes a new JIT-scope reader and must
    route through the adapter pattern specified in the P2-L3 decision
    record — not silently appear.
    """

    # (filename relative to omiecid_simulator/omiecid/, exact line substring or None).
    # None means "any line in this file is allowed" (whole-file allowlist).
    ALLOWLIST: tuple[tuple[str, str | None], ...] = (
        ("contract.py", None),  # whole-file: Python-side by decision.
        (
            "contract_book.py",
            "self._contract.status == ContractStatus.OPEN",
        ),  # single grandfathered reader.
    )

    def _is_allowlisted(self, rel_path: str, line_text: str) -> bool:
        for allowed_file, allowed_substr in self.ALLOWLIST:
            if rel_path != allowed_file:
                continue
            if allowed_substr is None:
                return True
            if allowed_substr in line_text:
                return True
        return False

    def test_no_unallowed_status_comparisons_in_omie(self) -> None:
        assert _OMIE_DIR.exists(), (
            f"omiecid_simulator/omiecid/ not found at {_OMIE_DIR}."
        )
        violations: list[str] = []
        for py_file in sorted(_OMIE_DIR.glob("*.py")):
            rel = py_file.name
            try:
                lines = py_file.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError as exc:  # pragma: no cover - defensive
                pytest.fail(f"Could not read {py_file}: {exc}")
            for lineno, line in enumerate(lines, start=1):
                # Skip comments to avoid false positives on documentation
                # that discusses the forbidden pattern (e.g., the P2-L3
                # decision record itself, which mentions `.status`).
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if not _STATUS_CMP_REGEX.search(line):
                    continue
                if self._is_allowlisted(rel, line):
                    continue
                violations.append(f"  {rel}:{lineno}: {line.rstrip()}")

        assert not violations, (
            "P2-L3 boundary test: new `.status ==` / `.status !=` "
            "comparison(s) found in omiecid_simulator/omiecid/ outside the "
            "sanctioned allowlist.\n"
            "Allowlist (see test docstring for rationale):\n"
            "  - contract.py (entire file, Python-side by decision)\n"
            "  - contract_book.py: the single `self._contract.status "
            "== ContractStatus.OPEN` line in `submit_order` "
            "(grandfathered; P2-H2 will adapter-ise).\n"
            "Any other JIT-scope reader of `.status` must route through "
            "the closed-over-scalar / adapter pattern specified in the "
            "P2-L3 decision record in "
            "`omiecid_simulator/omiecid/jax_constants.py`.\n"
            "Offending site(s):\n" + "\n".join(violations)
        )

    def test_allowlist_sites_still_exist(self) -> None:
        """Sanity: the grandfathered reader still exists exactly where documented.

        If P2-H2 (or any other story) moves or removes the
        `self._contract.status == ContractStatus.OPEN` line in
        `contract_book.py`, the allowlist entry becomes stale. That is
        a GOOD thing — but the allowlist must be updated in the same
        commit. This test fails loudly so the allowlist does not silently
        rot into a broader carve-out than intended.

        `contract.py` is whole-file allowlisted, so we only assert it
        exists; we do not require any particular line inside it.
        """
        contract_py = _OMIE_DIR / "contract.py"
        assert contract_py.exists(), (
            f"contract.py missing at {contract_py} — "
            "P2-L3 allowlist references a file that does not exist."
        )
        contract_book_py = _OMIE_DIR / "contract_book.py"
        assert contract_book_py.exists(), (
            f"contract_book.py missing at {contract_book_py} — "
            "P2-L3 allowlist references a file that does not exist."
        )
        text = contract_book_py.read_text(encoding="utf-8")
        assert "self._contract.status == ContractStatus.OPEN" in text, (
            "Grandfathered allowlist site "
            "`self._contract.status == ContractStatus.OPEN` no longer "
            "exists in contract_book.py. If P2-H2 has adapter-ised this "
            "line (replacing it with a closed-over `contract_open` "
            "scalar), the P2-L3 allowlist entry for contract_book.py "
            "must be removed in the same commit. This failure is "
            "expected at that transition — update the allowlist, do "
            "not silently widen it."
        )
