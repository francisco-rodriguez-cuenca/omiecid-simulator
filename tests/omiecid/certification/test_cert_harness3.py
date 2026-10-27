"""OMIE-CERT-HARNESS-3 -- section 9 and the log's "Why it failed" come from the audit
twin when the notebook PASSes; the crash text appears only when no results JSON exists.

Locked behavioural contract (agile-plan, Story OMIE-CERT-HARNESS-3, Test-first contract):

(1) A PASS notebook whose audit twin fails:
    * ``append_failure_section(nb, results, ..., explanation=<twin text>)`` writes the
      section-9 cell from the twin text (headings 9 and 9.1-9.5 kept, in order; 9.1 opens
      with "the notebook passed; its audit twin failed") and returns the results payload
      **unchanged** (byte-equal; still a valid PASS payload with a null
      ``failure_explanation``).
    * ``close_scenario(..., twin_why=<twin text>)`` on a non-K6 route writes the twin text
      as "Why it failed", never the crash text. Precedence on a non-K6 route: the results
      paragraph, else ``twin_why``, else (results present) the "no explanation" line,
      else (no results) the crash text.
(2) Results present without an explanation and no ``twin_why``: "Why it failed" reads
    "explanation missing — the results carry no failure explanation"; never "crashed".
(3) Guards (green before and after): the crash path without results keeps the crash text;
    a FAIL payload's paragraph is still used; ``validate_results``' PASS <=> null rule and
    the existing ``append_failure_section`` behaviour are unchanged.

Only the public ``notebooks.certification.cert_support`` API is used. All writes go to
scratch copies under ``tmp_path``. Scenario ids and twin texts are varied across tests so
that no assertion passes by coincidence of a fixed constant.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import nbformat
import pytest

from tests.omiecid.reference.omie_invariants import BatteryReport, CheckResult

try:  # collection must succeed even if the harness import breaks; the tests then fail
    from notebooks.certification import cert_support as cs
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    cs = None
    _IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_LOG = REPO_ROOT / "docs" / "omiecid" / "certification-log.md"
REAL_CERT = REPO_ROOT / "docs" / "omiecid" / "certificate.md"
TEMPLATE = REPO_ROOT / "notebooks" / "certification" / "_template.ipynb"

SECTION9 = ("## 9. Why this certification failed", "### 9.1 What failed", "### 9.2 The smallest failing case",
            "### 9.3 Which rule it concerns", "### 9.4 Where the fault most plausibly lies",
            "### 9.5 What it affects downstream")
PASS_TWIN_LINE = "the notebook passed; its audit twin failed"
NO_EXPLANATION = "explanation missing — the results carry no failure explanation"
CRASH_PREFIX = "explanation missing — notebook crashed"

AWAIT_OPEN = "**Awaiting the human (K7-U):** "
AWAIT_CLOSE = " — every"
TOC_PLACEHOLDER = ("| — | — | *(no entries yet — the ladder runs in a later session, after the three "
                   "`OMIE-CERT-*` prerequisite stories)* | — | — | — | — |")

# Varied twin texts (distinct ids, files, lines) so no assertion passes on a shared constant.
TWIN_TEXTS = {
    "C14": ("test_C14_audit.py::test_source_citation_is_current failed 1/49: the notebook quotes the "
            "pre-S7 omie_config.py:38 comment; the engine checks all pass"),
    "C07": ("test_C07_audit.py::test_fok_quote_digest failed 2/31: stale market-spec v2.12 wording of "
            "§4.4 item 2 (kill-on-partial)"),
    "C11": ("test_C11_audit.py::test_gct_boundary_citation failed 1/18: cites open-questions.md OQ-9 "
            "'Resolution' block that moved to OQ-19"),
}


@pytest.fixture(autouse=True)
def _require_harness():
    if cs is None:
        pytest.fail(f"notebooks.certification.cert_support is not importable: {_IMPORT_ERROR!r}")


# ---------------------------------------------------------------------------
# Builders (public API only)
# ---------------------------------------------------------------------------
def _report():
    return BatteryReport(results=tuple(
        CheckResult(check_id=c, status="pass", violations=(), unobserved=(), evidence={})
        for c in ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")))


def _results(cert_id: str, *, ok: bool, scenario: str = "Probe scenario") -> dict:
    cert = cs.Certification(
        cert_id=cert_id, notebook=f"{cert_id}_probe.ipynb", scenario=scenario,
        known_result="p_trade = p_rest", oracle_kind="exact-parametric",
        quotes=(cs.SpecQuote("market-spec.md", "§6.2 item 3",
                             "Trade executes at the **resting order's limit price** (maker pricing)."),),
        requested_draws=1)
    cert.check("trade price equals resting price", ok, seed=0, expected=100, observed=100 if ok else 101,
               unit="price_int")
    cert.set_eager_jit(0, True)
    cert.add_battery(_report(), seed=0)
    payload = cert.payload()
    assert cs.validate_results(payload) == []
    assert (payload["status"] == "PASS") is ok
    assert (payload["failure_explanation"] is None) is ok
    return payload


def _skeleton_log_text() -> str:
    text = REAL_LOG.read_text(encoding="utf-8")
    head, sep, _ = text.partition("\n## Entries\n")
    assert sep, "the real log must carry a '## Entries' heading"
    text = head + "\n## Entries\n\n*(none yet)*\n"
    i = text.index(AWAIT_OPEN) + len(AWAIT_OPEN)
    j = text.index(AWAIT_CLOSE, i)
    text = text[:i] + "*(none yet)*" + text[j:]
    lines = text.split("\n")
    k = next(n for n, ln in enumerate(lines) if ln.startswith("| # | Date | Scenario"))
    end = k + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    lines[k + 2:end] = [TOC_PLACEHOLDER]
    return "\n".join(lines)


def _skeleton_cert_text() -> str:
    text = REAL_CERT.read_text(encoding="utf-8")
    lines = text.split("\n")
    for n, ln in enumerate(lines):
        if re.match(r"^\| C\d+ ", ln):
            cells = ln.split("|")
            assert len(cells) == 7, ln
            cells[3:6] = [" — ", " — ", " — "]
            lines[n] = "|".join(cells)
    text = "\n".join(lines)
    a = text.index("\n## Scenario sections\n")
    first = re.search(r"^### C\d+ — ", text[a:], re.M)
    if first:
        start = a + first.start()
        stop = text.index("\n---\n", start)
        text = text[:start].rstrip("\n") + "\n" + text[stop:]
    return text


@pytest.fixture
def scratch(tmp_path):
    log = tmp_path / "certification-log.md"
    cert = tmp_path / "certificate.md"
    log.write_text(_skeleton_log_text(), encoding="utf-8")
    cert.write_text(_skeleton_cert_text(), encoding="utf-8")
    return log, cert


def _close(scratch, route, results, cid, **kw):
    log, cert = scratch
    return cs.close_scenario(
        route=route, results=results, log_path=log, certificate_path=cert,
        card=f"OMIE-CERT-{cid}-1", branch=f"omie/cert-{cid.lower()}-probe", commit="f00dcafe",
        scenario=kw.pop("scenario", f"Probe {cid}"), date=kw.pop("date", "2026-10-08"), **kw)


def _entry_block(text: str, n: int) -> str:
    sec = text.split("\n## Entries\n", 1)[1]
    m2 = re.search(r"^## ", sec, re.M)
    sec = sec[:m2.start()] if m2 else sec
    m = re.search(rf"^### {n} · .*$", sec, re.M)
    assert m, f"entry {n} not found"
    nxt = re.search(r"^### \d+ · ", sec[m.end():], re.M)
    return sec[m.start(): m.end() + (nxt.start() if nxt else len(sec))]


def _why(log: Path, n: int) -> str:
    """The entry's 'Why it failed' field (continuation lines joined, whitespace collapsed)."""
    lines = _entry_block(log.read_text(encoding="utf-8"), n).splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("- **Why it failed"):
            parts = [ln]
            for nxt in lines[i + 1:]:
                if not nxt.strip() or nxt.startswith("- ") or nxt.startswith("#"):
                    break
                parts.append(nxt)
            return re.sub(r"[ \t]+", " ", " ".join(p.strip() for p in parts)).strip()
    raise AssertionError(f"entry {n} has no 'Why it failed' line")


def _has_why(log: Path, n: int) -> bool:
    return any(ln.startswith("- **Why it failed")
               for ln in _entry_block(log.read_text(encoding="utf-8"), n).splitlines())


def _notebook(tmp_path: Path, cid: str) -> Path:
    path = tmp_path / f"{cid}_probe.ipynb"
    path.write_bytes(TEMPLATE.read_bytes())
    return path


def _tags(cell):
    return list(cell.get("metadata", {}).get("tags", []))


def _section9_cells(nb):
    return [c for c in nb.cells if "section-9" in _tags(c)]


def _canon(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _flat(text: str) -> str:
    return " ".join(text.split())


# ===========================================================================
# (1a) Section 9 from the twin text on a PASS payload -- FAILS on main
# ===========================================================================
class TestSection9FromTwin:
    @pytest.mark.parametrize("cid", ["C14", "C07"])
    def test_pass_payload_with_twin_explanation_writes_section9_last(self, tmp_path, cid):
        path = _notebook(tmp_path, cid)
        results = _results(cid, ok=True)
        cs.append_failure_section(path, results, hypothesis=f"stale citation in {cid}",
                                  explanation=TWIN_TEXTS[cid])
        nb = nbformat.read(path, as_version=4)
        assert len(_section9_cells(nb)) == 1, "exactly one section-9 cell"
        last = nb.cells[-1]
        assert "section-9" in _tags(last) and last.cell_type == "markdown"
        pos = [last.source.find(h) for h in SECTION9]
        assert all(p >= 0 for p in pos), "headings 9 and 9.1-9.5 must all be present"
        assert pos == sorted(pos), "headings 9 and 9.1-9.5 must be in order"
        assert cs.notebook_structure_errors(nb) == []

    @pytest.mark.parametrize("cid", ["C14", "C11"])
    def test_twin_text_is_written_into_section9(self, tmp_path, cid):
        path = _notebook(tmp_path, cid)
        cs.append_failure_section(path, _results(cid, ok=True), hypothesis="h",
                                  explanation=TWIN_TEXTS[cid])
        src = _flat(nbformat.read(path, as_version=4).cells[-1].source)
        assert TWIN_TEXTS[cid] in src, "the twin's explanation must appear in section 9"
        assert not any(t in src for k, t in TWIN_TEXTS.items() if k != cid)

    @pytest.mark.parametrize("cid", ["C07", "C11"])
    def test_9_1_opens_with_the_pass_twin_line_and_keeps_its_heading(self, tmp_path, cid):
        path = _notebook(tmp_path, cid)
        cs.append_failure_section(path, _results(cid, ok=True), hypothesis="h",
                                  explanation=TWIN_TEXTS[cid])
        src = nbformat.read(path, as_version=4).cells[-1].source
        assert src.count(SECTION9[1]) == 1, "the 9.1 heading is kept, exactly once"
        body_91 = src[src.index(SECTION9[1]) + len(SECTION9[1]):src.index(SECTION9[2])].strip()
        assert body_91.casefold().startswith(PASS_TWIN_LINE.casefold()), (
            f"9.1 must open with {PASS_TWIN_LINE!r}; got {body_91[:120]!r}")

    @pytest.mark.parametrize("cid", ["C14", "C07", "C11"])
    def test_pass_payload_is_returned_byte_equal(self, tmp_path, cid):
        path = _notebook(tmp_path, cid)
        results = _results(cid, ok=True)
        before = copy.deepcopy(results)
        out = cs.append_failure_section(path, results, hypothesis=f"hyp {cid}",
                                        downstream=f"down {cid}", explanation=TWIN_TEXTS[cid])
        assert _canon(out) == _canon(before), "the payload must be returned unchanged"
        assert cs.results_digest(out) == cs.results_digest(before)
        assert out["status"] == "PASS" and out["failure_explanation"] is None
        assert cs.validate_results(out) == []
        assert _canon(results) == _canon(before), "the caller's payload must not be mutated"

    def test_sections_1_to_8_untouched(self, tmp_path):
        path = _notebook(tmp_path, "C07")
        before = nbformat.read(path, as_version=4)
        cs.append_failure_section(path, _results("C07", ok=True), hypothesis="h",
                                  explanation=TWIN_TEXTS["C07"])
        after = nbformat.read(path, as_version=4)
        assert [c.source for c in after.cells[:-1]] == [c.source for c in before.cells]

    def test_rewrite_replaces_the_earlier_twin_section(self, tmp_path):
        path = _notebook(tmp_path, "C11")
        res = _results("C11", ok=True)
        cs.append_failure_section(path, res, hypothesis="h", explanation=TWIN_TEXTS["C07"])
        cs.append_failure_section(path, res, hypothesis="h", explanation=TWIN_TEXTS["C11"])
        nb = nbformat.read(path, as_version=4)
        s9 = _section9_cells(nb)
        assert len(s9) == 1
        flat = _flat(s9[0].source)
        assert TWIN_TEXTS["C11"] in flat and TWIN_TEXTS["C07"] not in flat
        assert cs.notebook_structure_errors(nb) == []


# ===========================================================================
# (1b) The log's "Why it failed" takes the twin text -- FAILS on main
# ===========================================================================
class TestLogWhyFromTwin:
    @pytest.mark.parametrize("cid", ["C14", "C07"])
    def test_merge_evidence_pass_results_uses_twin_text(self, scratch, cid):
        log, _ = scratch
        out = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results(cid, ok=True), cid,
                     twin_why=TWIN_TEXTS[cid])
        why = _why(log, out.entry_n)
        assert TWIN_TEXTS[cid] in why
        assert "crashed" not in why and "traceback" not in why
        assert "explanation missing" not in why
        assert cs.parse_log(log)[-1].result == "NOT CERTIFIED"

    def test_twin_text_wins_over_a_stray_traceback_when_results_exist(self, scratch):
        log, _ = scratch
        tb = "Traceback (most recent call last):\nKeyError: 'unrelated_c11_marker'"
        out = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C11", ok=True), "C11",
                     twin_why=TWIN_TEXTS["C11"], traceback=tb)
        why = _why(log, out.entry_n)
        assert TWIN_TEXTS["C11"] in why
        assert "unrelated_c11_marker" not in why and "crashed" not in why

    def test_record_only_pass_results_uses_twin_text(self, scratch):
        log, _ = scratch
        out = _close(scratch, cs.Route.K6F_RECORD_ONLY, _results("C07", ok=True), "C07",
                     twin_why=TWIN_TEXTS["C07"], tip_sha="beadfeed")
        why = _why(log, out.entry_n)
        assert TWIN_TEXTS["C07"] in why and "crashed" not in why

    def test_results_paragraph_takes_precedence_over_twin_text(self, scratch):
        log, _ = scratch
        res = _results("C14", ok=False)
        paragraph = _flat(res["failure_explanation"]["paragraph"])
        out = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, res, "C14", twin_why=TWIN_TEXTS["C14"])
        why = _why(log, out.entry_n)
        assert paragraph in why
        assert TWIN_TEXTS["C14"] not in why

    def test_k6_route_writes_no_why_line_even_with_twin_text(self, scratch):
        log, _ = scratch
        out = _close(scratch, cs.Route.K6, _results("C11", ok=True), "C11", twin_why=TWIN_TEXTS["C11"])
        assert cs.parse_log(log)[-1].result == "CERTIFIED"
        assert not _has_why(log, out.entry_n)
        assert TWIN_TEXTS["C11"] not in log.read_text(encoding="utf-8")


# ===========================================================================
# (2) Results present, no explanation, no twin text -- FAILS on main
# ===========================================================================
class TestResultsWithoutExplanation:
    @pytest.mark.parametrize("cid", ["C14", "C11"])
    def test_merge_evidence_pass_results_no_twin_says_results_carry_none(self, scratch, cid):
        log, _ = scratch
        out = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results(cid, ok=True), cid)
        why = _why(log, out.entry_n)
        assert NO_EXPLANATION in why
        assert "crashed" not in why and "traceback" not in why

    def test_traceback_is_ignored_when_results_exist(self, scratch):
        log, _ = scratch
        tb = "Traceback (most recent call last):\nRuntimeError: c07-should-not-appear"
        out = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C07", ok=True), "C07", traceback=tb)
        why = _why(log, out.entry_n)
        assert NO_EXPLANATION in why
        assert "c07-should-not-appear" not in why and "crashed" not in why

    def test_record_only_pass_results_no_twin_says_results_carry_none(self, scratch):
        log, _ = scratch
        out = _close(scratch, cs.Route.K6F_RECORD_ONLY, _results("C11", ok=True), "C11", tip_sha="0ddba11")
        why = _why(log, out.entry_n)
        assert NO_EXPLANATION in why and "crashed" not in why

    def test_explicit_none_twin_matches_omitted_twin(self, scratch):
        log, _ = scratch
        out = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C14", ok=True), "C14", twin_why=None)
        why = _why(log, out.entry_n)
        assert NO_EXPLANATION in why and "crashed" not in why


# ===========================================================================
# (3) Guards -- GREEN on main and after the fix
# ===========================================================================
class TestGuards:
    @pytest.mark.parametrize("cid,err", [("C07", "ValueError: boom-c07"), ("C14", "KeyError: 'c14-key'")])
    def test_crash_without_results_keeps_crash_text(self, scratch, cid, err):
        log, cert = scratch
        tb = f'Traceback (most recent call last):\n  File "<cell>", line 3, in <module>\n{err}'
        out = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, None, cid, traceback=tb)
        why = _why(log, out.entry_n)
        assert f"{CRASH_PREFIX}; traceback: Traceback (most recent call last): … {err}" in why
        assert NO_EXPLANATION not in why
        assert "no results JSON (the notebook crashed)" in cert.read_text(encoding="utf-8")

    def test_crash_without_results_or_traceback_says_unavailable(self, scratch):
        log, _ = scratch
        out = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, None, "C11")
        assert f"{CRASH_PREFIX}; traceback: unavailable" in _why(log, out.entry_n)

    @pytest.mark.parametrize("cid", ["C07", "C11"])
    def test_fail_payload_paragraph_still_used(self, scratch, cid):
        log, _ = scratch
        res = _results(cid, ok=False)
        paragraph = _flat(res["failure_explanation"]["paragraph"])
        out = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, res, cid)
        why = _why(log, out.entry_n)
        assert paragraph in why and "crashed" not in why

    def test_validate_results_still_rejects_explanation_on_pass(self):
        bad = copy.deepcopy(_results("C14", ok=True))
        bad["failure_explanation"] = copy.deepcopy(_results("C14", ok=False)["failure_explanation"])
        assert any("failure_explanation must be null exactly when status is PASS" in e
                   for e in cs.validate_results(bad))

    def test_append_failure_section_without_explanation_unchanged_on_fail_payload(self, tmp_path):
        path = _notebook(tmp_path, "C11")
        res = _results("C11", ok=False)
        out = cs.append_failure_section(path, res, hypothesis="hyp-c11-guard", downstream="down-c11-guard")
        assert out["failure_explanation"]["hypothesis"] == "hyp-c11-guard"
        assert out["failure_explanation"]["downstream"] == "down-c11-guard"
        assert "hyp-c11-guard" in out["failure_explanation"]["paragraph"]
        assert cs.validate_results(out) == []
        nb = nbformat.read(path, as_version=4)
        assert cs.notebook_structure_errors(nb) == []
        src = nb.cells[-1].source
        assert "hyp-c11-guard" in src
        assert PASS_TWIN_LINE.casefold() not in src.casefold()
