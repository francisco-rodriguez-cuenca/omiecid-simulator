"""OMIE-CERT-HARNESS-1 -- reading and writing the certification log and the
certificate (decisions H10, H11; AC (5), (6)).

Locked behavioural contract:

* ``parse_log`` / ``gated_notebooks`` / ``next_scenario`` (H10): latest entry is
  the highest N; a scenario is gated iff its latest Result starts with
  ``CERTIFIED`` and its Verdict is not a human verdict; the fenced entry
  template is never parsed; malformed logs raise ``LogParseError``; a gated id
  with no notebook raises ``GateSelectionError``.
* ``close_scenario`` (H11) round-trips through ``parse_log``, writes the entry in
  the field order of the log's own ``## Entry template`` block, maintains the
  table of contents and the "Awaiting the human" list, and writes / signs the
  certificate section and coverage-matrix row by route.

All writes go to scratch copies under ``tmp_path``. The scratch log is rebuilt
from the real log's preamble and entry template, with its three placeholders
reset, and the scratch certificate from the real certificate with every
coverage-matrix Status/Date/Entry cell reset and every scenario section removed,
so these tests stay valid once the real ladder has written entries and signed
sections.
"""

from __future__ import annotations

import builtins
import re
from pathlib import Path

import nbformat
import pytest

from tests.omiecid.reference.omie_invariants import BatteryReport, CheckResult, Violation

try:  # collection must succeed before the harness exists; the tests then fail
    from notebooks.certification import cert_support as cs
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    cs = None
    _IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_LOG = REPO_ROOT / "docs" / "omiecid" / "certification-log.md"
REAL_CERT = REPO_ROOT / "docs" / "omiecid" / "certificate.md"
ORDER = ("C90", "C91", "C92")

TOC_PLACEHOLDER = ("| — | — | *(no entries yet — the ladder runs in a later session, after the three "
                   "`OMIE-CERT-*` prerequisite stories)* | — | — | — | — |")
AWAIT_OPEN = "**Awaiting the human (K7-U):** "
AWAIT_CLOSE = " — every"


@pytest.fixture(autouse=True)
def _require_harness():
    if cs is None:
        pytest.fail(f"notebooks.certification.cert_support is not importable: {_IMPORT_ERROR!r}")


# ---------------------------------------------------------------------------
# Scratch log / certificate
# ---------------------------------------------------------------------------
def _skeleton_log_text() -> str:
    text = REAL_LOG.read_text(encoding="utf-8")
    # entries section -> placeholder
    head, sep, _ = text.partition("\n## Entries\n")
    assert sep, "the real log must carry a '## Entries' heading"
    text = head + "\n## Entries\n\n*(none yet)*\n"
    # awaiting list -> placeholder
    i = text.index(AWAIT_OPEN) + len(AWAIT_OPEN)
    j = text.index(AWAIT_CLOSE, i)
    text = text[:i] + "*(none yet)*" + text[j:]
    # table of contents rows -> placeholder row
    lines = text.split("\n")
    k = next(n for n, ln in enumerate(lines) if ln.startswith("| # | Date | Scenario"))
    end = k + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    lines[k + 2:end] = [TOC_PLACEHOLDER]
    return "\n".join(lines)


def _skeleton_cert_text() -> str:
    """The real certificate with every coverage-matrix Status/Date/Entry cell reset to '—' and
    every scenario section removed (the intro paragraph of '## Scenario sections' is kept) --
    independent of how many scenarios the live ladder has signed (copied from
    test_cert_harness_audit.py)."""
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
    nbdir = tmp_path / "nbs"
    nbdir.mkdir()
    return log, cert, nbdir


def _template_block(text: str) -> str:
    start = text.index("## Entry template")
    a = text.index("```markdown", start)
    b = text.index("```", a + len("```markdown"))
    return text[a:b]


def _template_labels(text: str):
    return [m.group(1).rstrip(":") for m in re.finditer(r"^- \*\*([^*]+?)\*\*", _template_block(text), re.M)]


def _entries_section(text: str) -> str:
    rest = text.split("\n## Entries\n", 1)[1]
    m = re.search(r"^## ", rest, re.M)
    return rest[:m.start()] if m else rest


def _entry_block(text: str, n: int) -> str:
    sec = _entries_section(text)
    m = re.search(rf"^### {n} · .*$", sec, re.M)
    assert m, f"entry {n} not found"
    nxt = re.search(r"^### \d+ · ", sec[m.end():], re.M)
    return sec[m.start(): m.end() + (nxt.start() if nxt else len(sec))]


def _labels_of(block: str):
    return [m.group(1).rstrip(":") for m in re.finditer(r"^- \*\*([^*]+?)\*\*", block, re.M)]


def _line(block: str, label: str) -> str:
    """The field starting ``- **<label>``, continuation lines joined, whitespace collapsed."""
    lines = block.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(f"- **{label}"):
            parts = [ln]
            for nxt in lines[i + 1:]:
                if not nxt.strip() or nxt.startswith("- ") or nxt.startswith("#"):
                    break
                parts.append(nxt)
            return re.sub(r"[ \t]+", " ", " ".join(p.strip() for p in parts)).strip()
    raise AssertionError(f"no '{label}' line in entry")


def _awaiting(text: str) -> str:
    i = text.index(AWAIT_OPEN) + len(AWAIT_OPEN)
    return text[i:text.index(AWAIT_CLOSE, i)]


def _toc(text: str) -> str:
    return text[text.index("## Table of contents"):text.index(AWAIT_OPEN)]


def _section(text: str, heading_prefix: str) -> str:
    i = text.index(heading_prefix)
    m = re.search(r"^#{2,3} ", text[i + len(heading_prefix):], re.M)
    return text[i: i + len(heading_prefix) + (m.start() if m else len(text))]


def _matrix(text: str) -> str:
    return _section(text, "## Coverage matrix")


def _matrix_row(text: str, prefix: str):
    for ln in _matrix(text).splitlines():
        if ln.startswith("| ") and ln[2:].startswith(prefix):
            return [c.strip() for c in ln.strip().strip("|").split("|")]
    raise AssertionError(f"no matrix row {prefix!r}")


# ---------------------------------------------------------------------------
# Hand-written entries for the reader tests
# ---------------------------------------------------------------------------
def _entry(n, cid, result, verdict="______", rationale="______", *, date="2026-10-03", scenario="Probe",
           drop=()):
    fields = [
        ("Card", f"- **Card:** `OMIE-CERT-{cid}-1` · **Branch:** `omie/cert-x` · **Commit:** `abc1234`"),
        ("Notebook", f"- **Notebook:** `notebooks/certification/{cid}_x.ipynb` · **Results:** `results/{cid}.json`"),
        ("Certifies", "- **Certifies:** §6.2 item 3 (market-spec v2.13)"),
        ("Dataset", "- **Dataset:** C02.crossing_pair, 3 draws, seeds 0–2"),
        ("Known result", "- **Known result:** p_trade = p_rest"),
        ("Outcome", "- **Outcome:** 3/3 assertions passed"),
        ("Capacity peaks", "- **Capacity peaks:** 1/1/1/0"),
        ("Eager vs jitted", "- **Eager vs jitted:** identical"),
        ("Wall clock", "- **Wall clock:** 12 (forecast 20)"),
        ("Anything", "- **Anything the run revealed that the card did not anticipate:** nothing"),
        ("Result", f"- **Result:** {result}"),
        ("Merged", "- **Merged:** branch merged"),
        ("Verdict", f"- **Verdict:** {verdict} · **rationale:** {rationale}"),
    ]
    body = "\n".join(line for key, line in fields if key not in drop)
    return f"### {n} · {date} · `{cid}` — {scenario}\n\n{body}\n"


def _write_log(path: Path, *entries: str, extra_after: str = "") -> Path:
    text = _skeleton_log_text().replace("## Entries\n\n*(none yet)*\n", "## Entries\n\n" + "\n".join(entries))
    path.write_text(text + extra_after, encoding="utf-8")
    return path


def _nb(path: Path) -> Path:
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, path)
    return path


# ---------------------------------------------------------------------------
# H10 -- reading
# ---------------------------------------------------------------------------
class TestParseLog:
    def test_real_log_parses(self):
        cs.parse_log(REAL_LOG)

    def test_skeleton_has_no_entries(self, scratch):
        log, _, _ = scratch
        assert cs.parse_log(log) == ()

    def test_entry_fields(self, scratch):
        log, _, _ = scratch
        _write_log(log, _entry(1, "C03", "CERTIFIED", "CERTIFIED", "K5 pass", scenario="Price-time priority"))
        (e,) = cs.parse_log(log)
        assert (e.n, e.date, e.cert_id, e.scenario, e.result, e.verdict) == (
            1, "2026-10-03", "C03", "Price-time priority", "CERTIFIED", "CERTIFIED")

    def test_blank_verdict_parses_empty(self, scratch):
        log, _, _ = scratch
        _write_log(log, _entry(1, "C03", "NOT CERTIFIED"))
        assert cs.parse_log(log)[0].verdict == ""

    def test_human_verdict_parsed(self, scratch):
        log, _, _ = scratch
        _write_log(log, _entry(1, "C03", "NOT CERTIFIED", "DEFECT-FOUND", "engine bug"))
        assert cs.parse_log(log)[0].verdict == "DEFECT-FOUND"

    def test_provisional_result(self, scratch):
        log, _, _ = scratch
        _write_log(log, _entry(1, "C03", "CERTIFIED — provisional on C02, C05", "CERTIFIED", "K5 pass"))
        assert cs.parse_log(log)[0].result == "CERTIFIED — provisional on C02, C05"

    def test_entry_template_block_never_parsed(self, scratch):
        log, _, _ = scratch
        fenced = "```markdown\n" + _entry(7, "C05", "CERTIFIED", "CERTIFIED", "K5 pass") + "```\n"
        _write_log(log, _entry(1, "C03", "CERTIFIED", "CERTIFIED", "K5 pass"), fenced)
        assert [e.n for e in cs.parse_log(log)] == [1]

    def test_text_after_the_next_level2_heading_ignored(self, scratch):
        log, _, _ = scratch
        _write_log(log, _entry(1, "C03", "CERTIFIED", "CERTIFIED", "K5 pass"),
                   extra_after="\n## Appendix\n\n" + _entry(2, "C04", "CERTIFIED", "CERTIFIED", "K5 pass"))
        assert [e.n for e in cs.parse_log(log)] == [1]

    @pytest.mark.parametrize("result", [
        "MAYBE", "CERTIFIED - provisional on C02", "CERTIFIED — provisional on 02",
        "CERTIFIED — provisional on", "not certified", "",
    ])
    def test_ill_formed_result_raises(self, scratch, result):
        log, _, _ = scratch
        _write_log(log, _entry(1, "C03", result))
        with pytest.raises(cs.LogParseError):
            cs.parse_log(log)

    def test_missing_result_raises(self, scratch):
        log, _, _ = scratch
        _write_log(log, _entry(1, "C03", "CERTIFIED", drop=("Result",)))
        with pytest.raises(cs.LogParseError):
            cs.parse_log(log)

    def test_missing_verdict_line_raises(self, scratch):
        log, _, _ = scratch
        _write_log(log, _entry(1, "C03", "CERTIFIED", drop=("Verdict",)))
        with pytest.raises(cs.LogParseError):
            cs.parse_log(log)

    @pytest.mark.parametrize("ns", [(2, 1), (1, 1), (1, 3, 2)])
    def test_non_increasing_numbers_raise(self, scratch, ns):
        log, _, _ = scratch
        _write_log(log, *[_entry(n, f"C0{i}", "CERTIFIED", "CERTIFIED", "K5 pass") for i, n in enumerate(ns)])
        with pytest.raises(cs.LogParseError):
            cs.parse_log(log)

    def test_errors_are_failures_not_skips(self):
        from _pytest.outcomes import Skipped
        for err in (cs.LogParseError, cs.GateSelectionError):
            assert issubclass(err, Exception)
            assert not issubclass(err, Skipped)


class TestGatedNotebooks:
    def _gated(self, scratch, *entries, ids=("C00", "C01", "C02", "C03")):
        log, _, nbdir = scratch
        for cid in ids:
            _nb(nbdir / f"{cid}_probe.ipynb")
        _write_log(log, *entries)
        return [p.name for p in cs.gated_notebooks(log, nbdir)]

    def test_latest_certified_gated(self, scratch):
        assert self._gated(scratch, _entry(1, "C00", "CERTIFIED", "CERTIFIED", "K5 pass")) == ["C00_probe.ipynb"]

    def test_provisional_gated(self, scratch):
        assert self._gated(scratch, _entry(1, "C03", "CERTIFIED — provisional on C02", "CERTIFIED",
                                           "K5 pass")) == ["C03_probe.ipynb"]

    def test_not_certified_skipped(self, scratch):
        assert self._gated(scratch, _entry(1, "C01", "NOT CERTIFIED")) == []

    def test_certified_then_not_certified_skipped(self, scratch):
        assert self._gated(scratch, _entry(1, "C01", "CERTIFIED", "CERTIFIED", "K5 pass"),
                           _entry(2, "C01", "NOT CERTIFIED")) == []

    def test_not_certified_then_certified_gated(self, scratch):
        assert self._gated(scratch, _entry(1, "C01", "NOT CERTIFIED", "REPEAT", "flaky expectation"),
                           _entry(2, "C01", "CERTIFIED", "CERTIFIED", "K5 pass")) == ["C01_probe.ipynb"]

    @pytest.mark.parametrize("verdict", ["REPEAT", "DEFECT-FOUND", "SCOPE-CHANGE", "INVALID"])
    def test_human_verdict_on_certified_ungates(self, scratch, verdict):
        assert self._gated(scratch, _entry(1, "C02", "CERTIFIED", "CERTIFIED", "K5 pass"),
                           _entry(2, "C02", "CERTIFIED", verdict, "retroactive correction")) == []

    def test_sorted_and_mixed(self, scratch):
        got = self._gated(scratch,
                          _entry(1, "C03", "CERTIFIED", "CERTIFIED", "K5 pass"),
                          _entry(2, "C01", "NOT CERTIFIED"),
                          _entry(3, "C00", "CERTIFIED", "CERTIFIED", "K5 pass"),
                          _entry(4, "C02", "CERTIFIED — provisional on C01", "CERTIFIED", "K5 pass"))
        assert got == ["C00_probe.ipynb", "C02_probe.ipynb", "C03_probe.ipynb"]

    def test_returns_paths_in_the_notebook_dir(self, scratch):
        log, _, nbdir = scratch
        _nb(nbdir / "C00_probe.ipynb")
        _write_log(log, _entry(1, "C00", "CERTIFIED", "CERTIFIED", "K5 pass"))
        assert cs.gated_notebooks(log, nbdir) == [nbdir / "C00_probe.ipynb"]

    def test_gated_id_without_notebook_raises(self, scratch):
        log, _, nbdir = scratch
        _write_log(log, _entry(1, "C05", "CERTIFIED", "CERTIFIED", "K5 pass"))
        with pytest.raises(cs.GateSelectionError):
            cs.gated_notebooks(log, nbdir)

    def test_gated_id_with_two_notebooks_raises(self, scratch):
        log, _, nbdir = scratch
        _nb(nbdir / "C05_a.ipynb")
        _nb(nbdir / "C05_b.ipynb")
        _write_log(log, _entry(1, "C05", "CERTIFIED", "CERTIFIED", "K5 pass"))
        with pytest.raises(cs.GateSelectionError):
            cs.gated_notebooks(log, nbdir)

    def test_skipped_id_needs_no_notebook(self, scratch):
        log, _, nbdir = scratch
        _write_log(log, _entry(1, "C05", "NOT CERTIFIED"))
        assert cs.gated_notebooks(log, nbdir) == []

    def test_malformed_log_raises_through_gate_selection(self, scratch):
        log, _, nbdir = scratch
        _write_log(log, _entry(1, "C05", "MAYBE"))
        with pytest.raises(cs.LogParseError):
            cs.gated_notebooks(log, nbdir)


class TestNextScenario:
    def test_empty_log_starts_at_the_first(self, scratch):
        log, _, _ = scratch
        assert cs.next_scenario(log) == "C00"
        assert cs.next_scenario(log, order=ORDER) == "C90"

    def test_blank_verdict_not_certified_does_not_hold_the_ladder(self, scratch):
        log, _, _ = scratch
        _write_log(log, _entry(1, "C00", "CERTIFIED", "CERTIFIED", "K5 pass"), _entry(2, "C01", "NOT CERTIFIED"))
        assert cs.next_scenario(log) == "C02"

    def test_first_id_without_entry_in_order(self, scratch):
        log, _, _ = scratch
        _write_log(log, _entry(1, "C91", "CERTIFIED", "CERTIFIED", "K5 pass"))
        assert cs.next_scenario(log, order=ORDER) == "C90"

    def test_exhausted_order_is_none(self, scratch):
        log, _, _ = scratch
        _write_log(log, *[_entry(i + 1, c, "NOT CERTIFIED") for i, c in enumerate(ORDER)])
        assert cs.next_scenario(log, order=ORDER) is None


# ---------------------------------------------------------------------------
# H11 -- writing
# ---------------------------------------------------------------------------
def _report(**status):
    ids = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")
    return BatteryReport(results=tuple(
        CheckResult(check_id=c, status=status.get(c, "pass"),
                    violations=(Violation(c, "probe", "broken"),) if status.get(c) == "fail" else (),
                    unobserved=(), evidence={})
        for c in ids))


def _results(cert_id: str, *, ok: bool) -> dict:
    cert = cs.Certification(
        cert_id=cert_id, notebook=f"{cert_id}_probe.ipynb", scenario="Probe scenario",
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
    return payload


def _close(scratch, route, results, cid, *, date="2026-10-02", scenario=None, **kw):
    log, cert, _ = scratch
    order = kw.pop("order", ORDER)
    return cs.close_scenario(
        route=route, results=results, log_path=log, certificate_path=cert,
        card=f"OMIE-CERT-{cid}-1", branch=f"omie/cert-{cid.lower()}-probe", commit="abc1234",
        scenario=scenario or f"Probe {cid}", date=date, order=order, **kw)


@pytest.fixture
def no_prompt(monkeypatch):
    def _refuse(*a, **k):
        raise AssertionError("close_scenario must never ask")
    monkeypatch.setattr(builtins, "input", _refuse)


class TestCloseK6:
    def test_outcome(self, scratch, no_prompt):
        out = _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90")
        assert out.entry_n == 1
        assert out.route is cs.Route.K6
        assert out.merged is True and out.signed is True
        assert out.next_scenario == "C91"

    def test_entry_round_trips(self, scratch, no_prompt):
        log, _, _ = scratch
        _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90", scenario="Dry pass", date="2026-10-04")
        (e,) = cs.parse_log(log)
        assert (e.n, e.date, e.cert_id, e.scenario, e.result, e.verdict) == (
            1, "2026-10-04", "C90", "Dry pass", "CERTIFIED", "CERTIFIED")

    def test_verdict_line_on_k6(self, scratch, no_prompt):
        log, _, _ = scratch
        _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90")
        block = _entry_block(log.read_text(encoding="utf-8"), 1)
        assert _line(block, "Verdict") == "- **Verdict:** CERTIFIED · **rationale:** K5 pass"
        merged = _line(block, "Merged")
        assert "merged" in merged and "NOT merged" not in merged and "evidence merged" not in merged

    def test_field_order_follows_template(self, scratch, no_prompt):
        log, _, _ = scratch
        tmpl = _template_labels(log.read_text(encoding="utf-8"))
        _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90")
        got = _labels_of(_entry_block(log.read_text(encoding="utf-8"), 1))
        assert got == [lb for lb in tmpl if lb in got]
        assert set(tmpl) - set(got) <= {"Why it failed"}

    def test_field_order_is_read_from_the_template_block(self, scratch, no_prompt):
        log, _, _ = scratch
        text = log.read_text(encoding="utf-8")
        tb = _template_block(text)
        ds = re.search(r"^- \*\*Dataset:\*\*.*$", tb, re.M).group(0)
        kr = re.search(r"^- \*\*Known result:\*\*.*$", tb, re.M).group(0)
        swapped = tb.replace(ds, "\x00").replace(kr, ds).replace("\x00", kr)
        log.write_text(text.replace(tb, swapped), encoding="utf-8")
        _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90")
        got = _labels_of(_entry_block(log.read_text(encoding="utf-8"), 1))
        assert got.index("Known result") < got.index("Dataset")

    def test_placeholders_replaced_and_toc_row(self, scratch, no_prompt):
        log, _, _ = scratch
        _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90")
        text = log.read_text(encoding="utf-8")
        assert "*(no entries yet" not in text
        assert "*(none yet)*" not in _entries_section(text)
        assert sum("C90" in ln for ln in _toc(text).splitlines() if ln.startswith("|")) == 1
        assert "*(none yet)*" in _awaiting(text)  # nothing awaits the human after a pass

    def test_template_block_and_preamble_untouched(self, scratch, no_prompt):
        log, _, _ = scratch
        before = log.read_text(encoding="utf-8")
        _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90")
        after = log.read_text(encoding="utf-8")
        assert _template_block(after) == _template_block(before)
        cut = before.index("## Table of contents")
        assert after[:cut] == before[:cut]

    def test_certificate_section_signed(self, scratch, no_prompt):
        _, cert, _ = scratch
        _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90", scenario="Dry pass", date="2026-10-04")
        text = cert.read_text(encoding="utf-8")
        assert text.index("## Scenario sections") < text.index("### C90 — Dry pass") < text.index("## Appendix A")
        sec = _section(text, "### C90 — Dry pass")
        labels = _labels_of(sec)
        assert labels[:4] == ["Status", "Signed", "Certifies", "Evidence"]
        assert re.search(r"^- \*\*Signed:\*\* 2026-10-04 · log entry 1 · results blake2b-128 [0-9a-f]{32}\s*$",
                         sec, re.M)
        assert "CERTIFIED" in _line(sec, "Status")

    def test_harness_id_leaves_matrix_untouched(self, scratch, no_prompt):
        _, cert, _ = scratch
        before = _matrix(cert.read_text(encoding="utf-8"))
        _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90")
        assert _matrix(cert.read_text(encoding="utf-8")) == before

    def test_ladder_id_updates_its_matrix_row_only(self, scratch, no_prompt):
        _, cert, _ = scratch
        before = cert.read_text(encoding="utf-8")
        c12 = _matrix_row(before, "C12 ")
        _close(scratch, cs.Route.K6, _results("C02", ok=True), "C02", date="2099-01-02",
               order=cs.LADDER)
        after = cert.read_text(encoding="utf-8")
        row = _matrix_row(after, "C2 ")
        assert row[0].startswith("C2 The single crossing pair")
        assert "CERTIFIED" in row[2] and "NOT" not in row[2]
        assert row[3] == "2099-01-02"
        assert re.search(r"\b1\b", row[4])
        assert _matrix_row(after, "C12 ") == c12

    def test_provisional_result_word(self, scratch, no_prompt):
        log, _, nbdir = scratch
        _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90", provisional_on=("C02", "C05"))
        (e,) = cs.parse_log(log)
        assert e.result == "CERTIFIED — provisional on C02, C05"
        _nb(nbdir / "C90_probe.ipynb")
        assert cs.gated_notebooks(log, nbdir) == [nbdir / "C90_probe.ipynb"]


class TestCloseK6F:
    @pytest.fixture
    def two(self, scratch, no_prompt):
        fail = _results("C91", ok=False)
        o1 = _close(scratch, cs.Route.K6, _results("C90", ok=True), "C90")
        o2 = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, fail, "C91", scenario="Dry fail")
        return scratch, o1, o2, fail

    def test_outcome(self, two):
        _, _, o2, _ = two
        assert o2.entry_n == 2
        assert o2.route is cs.Route.K6F_MERGE_EVIDENCE
        assert o2.merged is True and o2.signed is False
        assert o2.next_scenario == "C92"

    def test_entry_not_certified_with_blank_verdict(self, two):
        (log, _, _), _, _, fail = two
        e = cs.parse_log(log)[-1]
        assert (e.n, e.cert_id, e.result, e.verdict) == (2, "C91", "NOT CERTIFIED", "")
        block = _entry_block(log.read_text(encoding="utf-8"), 2)
        assert _line(block, "Verdict") == "- **Verdict:** ______ · **rationale:** ______"
        assert fail["failure_explanation"]["paragraph"] in _line(block, "Why it failed")
        assert "evidence merged, certificate section unsigned" in _line(block, "Merged")

    def test_failed_entry_field_order(self, two):
        (log, _, _), _, _, _ = two
        text = log.read_text(encoding="utf-8")
        assert _labels_of(_entry_block(text, 2)) == _template_labels(text)

    def test_awaiting_list_names_the_failure(self, two):
        (log, _, _), _, _, _ = two
        text = log.read_text(encoding="utf-8")
        aw = _awaiting(text)
        assert "`C91` (entry 2)" in aw
        assert "C90" not in aw
        assert "*(none yet)*" not in aw
        assert sum(ln.startswith("|") and ("C90" in ln or "C91" in ln) for ln in _toc(text).splitlines()) == 2

    def test_certificate_section_unsigned(self, two):
        (_, cert, _), _, _, _ = two
        sec = _section(cert.read_text(encoding="utf-8"), "### C91 — Dry fail")
        assert _line(sec, "Signed") == \
            "- **Signed:** *unsigned — NOT CERTIFIED; awaiting the human (K7-U)*"

    def test_gate_skips_the_failed_scenario(self, two):
        (log, _, nbdir), _, _, _ = two
        _nb(nbdir / "C90_dry_pass.ipynb")
        _nb(nbdir / "C91_dry_fail.ipynb")
        assert cs.gated_notebooks(log, nbdir) == [nbdir / "C90_dry_pass.ipynb"]


class TestCloseRecordOnlyAndCrash:
    def test_record_only(self, scratch, no_prompt):
        log, cert, _ = scratch
        cert_before = cert.read_bytes()
        out = _close(scratch, cs.Route.K6F_RECORD_ONLY, _results("C91", ok=False), "C91", tip_sha="feedbee")
        assert out.merged is False and out.signed is False
        assert cert.read_bytes() == cert_before, "record-only writes no certificate section"
        e = cs.parse_log(log)[-1]
        assert e.result == "NOT CERTIFIED" and e.verdict == ""
        merged = _line(_entry_block(log.read_text(encoding="utf-8"), out.entry_n), "Merged")
        assert "branch NOT merged" in merged
        assert "omie/cert-c91-probe" in merged and "feedbee" in merged
        assert "`C91`" in _awaiting(log.read_text(encoding="utf-8"))

    def test_crash_without_results(self, scratch, no_prompt):
        log, cert, _ = scratch
        tb = 'Traceback (most recent call last):\n  File "<cell>", line 1, in <module>\nValueError: boom'
        out = _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, None, "C91", scenario="Crash", traceback=tb)
        assert out.signed is False
        e = cs.parse_log(log)[-1]
        assert e.result == "NOT CERTIFIED" and e.verdict == ""
        block = _entry_block(log.read_text(encoding="utf-8"), out.entry_n)
        assert ("explanation missing — notebook crashed; traceback: Traceback (most recent call last): "
                "… ValueError: boom") in _line(block, "Why it failed")
        assert _line(block, "Known result").rstrip().endswith("—")
        assert _line(block, "Outcome").rstrip().endswith("—")
        sec = _section(cert.read_text(encoding="utf-8"), "### C91 — Crash")
        assert "unsigned" in _line(sec, "Signed")


# ---------------------------------------------------------------------------
# Step-5 re-close ruling (agile-plan v5.189, R1 / R2; invariant INV-C):
# the log is the history (append-only); the certificate is the current state
# (at most one section per id, signed iff the log gates the id).
# ---------------------------------------------------------------------------
_BLANK_VERDICT = "- **Verdict:** ______ · **rationale:** ______"
_UNSIGNED_K7U = "- **Signed:** *unsigned — NOT CERTIFIED; awaiting the human (K7-U)*"
_HUMAN_WORDS = ("REPEAT", "DEFECT-FOUND", "SCOPE-CHANGE", "INVALID")


def _region(text: str) -> str:
    """The ``## Scenario sections`` region of the certificate."""
    i = text.index("## Scenario sections")
    return text[i:text.index("## Appendix A", i)]


def _headings(text: str, cid: str):
    return re.findall(rf"^### {cid} — .*$", _region(text), re.M)


def _sec(text: str, cid: str) -> str:
    hs = _headings(text, cid)
    assert len(hs) == 1, f"expected exactly one section for {cid}, got {hs!r}"
    return _section(text, hs[0])


def _raw_field(block: str, label: str) -> str:
    """The raw (unnormalised) text of the field ``- **<label>``, with its continuation lines."""
    lines = block.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(f"- **{label}"):
            out = [ln]
            for nxt in lines[i + 1:]:
                if not nxt.strip() or nxt.startswith("- ") or nxt.startswith("#"):
                    break
                out.append(nxt)
            return "\n".join(out)
    raise AssertionError(f"no '{label}' field")


def _signed_ids(text: str) -> set:
    out = set()
    for m in re.finditer(r"^### (C\d{2}) — .*$", _region(text), re.M):
        if not _line(_section(text, m.group(0)), "Signed").startswith("- **Signed:** *unsigned"):
            out.add(m.group(1))
    return out


def _fill_verdict(log: Path, n: int, verdict_and_rationale: str) -> None:
    """The human types a Verdict into entry ``n`` (e.g. ``"REPEAT · **rationale:** x"``)."""
    text = log.read_text(encoding="utf-8")
    block = _entry_block(text, n)
    assert _BLANK_VERDICT in block
    log.write_text(text.replace(block, block.replace(_BLANK_VERDICT, f"- **Verdict:** {verdict_and_rationale}"),
                                1), encoding="utf-8")
    assert cs.parse_log(log)[n - 1].verdict == verdict_and_rationale.split(" ")[0]


def _append_correction(log: Path, n: int, cid: str, verdict: str, *, result: str = "CERTIFIED",
                       date: str = "2026-10-05") -> None:
    """The human records a correction entry (e.g. a retroactive REPEAT) at the end of the log."""
    text = log.read_text(encoding="utf-8").rstrip("\n")
    log.write_text(text + "\n\n" + _entry(n, cid, result, verdict, "retroactive correction", date=date),
                   encoding="utf-8")
    last = cs.parse_log(log)[-1]
    assert (last.n, last.cert_id, last.verdict) == (n, cid, verdict)


def _unsign(scratch, cid: str):
    log, cert, _ = scratch
    return cs.unsign_scenario(cert_id=cid, log_path=log, certificate_path=cert)


class TestReclose:
    """R1 -- a re-close replaces the id's section in place; never a second section."""

    def test_t1_k6_reclose_replaces_in_place(self, scratch, no_prompt):
        _, cert, _ = scratch
        _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C05", ok=False), "C05")
        _close(scratch, cs.Route.K6, _results("C06", ok=True), "C06")
        before = cert.read_text(encoding="utf-8")
        c06_before = _sec(before, "C06")
        _close(scratch, cs.Route.K6, _results("C05", ok=True), "C05")
        after = cert.read_text(encoding="utf-8")
        assert len(_headings(after, "C05")) == 1
        sec = _sec(after, "C05")
        assert re.search(r"^- \*\*Signed:\*\* 2026-10-02 · log entry 3 · results blake2b-128 [0-9a-f]{32}\s*$",
                         sec, re.M)
        assert "log entry 3" in _line(sec, "Evidence")
        status = _line(sec, "Status")
        assert "CERTIFIED" in status and "NOT" not in status
        assert after.index(_headings(after, "C05")[0]) < after.index(_headings(after, "C06")[0])
        assert _sec(after, "C06") == c06_before
        row = _matrix_row(after, "C5 ")
        assert row[2] == "CERTIFIED"
        assert row[4] == "3"

    def test_t2_k6f_merge_evidence_reclose_unsigns_in_place(self, scratch, no_prompt):
        _, cert, _ = scratch
        _close(scratch, cs.Route.K6, _results("C05", ok=True), "C05")
        _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C05", ok=False), "C05")
        text = cert.read_text(encoding="utf-8")
        sec = _sec(text, "C05")
        assert _line(sec, "Signed") == _UNSIGNED_K7U
        assert "log entry 2" in _line(sec, "Evidence")
        assert "log entry 1" not in sec
        row = _matrix_row(text, "C5 ")
        assert row[2] == "NOT CERTIFIED (section unsigned)"
        assert row[4] == "2"

    def test_t3_record_only_reclose_unsigns_existing_section(self, scratch, no_prompt):
        _, cert, _ = scratch
        _close(scratch, cs.Route.K6, _results("C05", ok=True), "C05")
        out = _close(scratch, cs.Route.K6F_RECORD_ONLY, _results("C05", ok=False), "C05", tip_sha="deadbee")
        assert out.signed is False
        assert out.merged is False
        text = cert.read_text(encoding="utf-8")
        sec = _sec(text, "C05")
        assert _line(sec, "Status") == "- **Status:** NOT CERTIFIED — section unsigned"
        assert _line(sec, "Signed") == _UNSIGNED_K7U
        ev = _line(sec, "Evidence")
        assert "NOT merged" in ev and "deadbee" in ev and "log entry 2" in ev
        row = _matrix_row(text, "C5 ")
        assert row[2] == "NOT CERTIFIED (section unsigned)"
        assert row[4] == "2"

    def test_t4_reclose_updates_the_heading(self, scratch, no_prompt):
        _, cert, _ = scratch
        _close(scratch, cs.Route.K6, _results("C05", ok=True), "C05", scenario="Old name")
        _close(scratch, cs.Route.K6, _results("C05", ok=True), "C05", scenario="New name")
        text = cert.read_text(encoding="utf-8")
        assert _headings(text, "C05") == ["### C05 — New name"]
        assert "### C05 — Old name" not in text

    def test_t5_prior_log_entries_kept_and_toc_one_row_per_entry(self, scratch, no_prompt):
        log, _, _ = scratch
        _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C05", ok=False), "C05")
        _close(scratch, cs.Route.K6, _results("C06", ok=True), "C06")
        before = log.read_text(encoding="utf-8")
        e1, e2 = _entry_block(before, 1), _entry_block(before, 2)
        _close(scratch, cs.Route.K6, _results("C05", ok=True), "C05")
        after = log.read_text(encoding="utf-8")
        assert _entry_block(after, 1).rstrip() == e1.rstrip()
        assert _entry_block(after, 2).rstrip() == e2.rstrip()
        assert [e.n for e in cs.parse_log(log)] == [1, 2, 3]
        rows = [ln for ln in _toc(after).splitlines() if ln.startswith("|")]
        assert sum("`C05`" in ln for ln in rows) == 2
        assert sum("`C06`" in ln for ln in rows) == 1

    @pytest.mark.parametrize("route", ["K6", "K6F_MERGE_EVIDENCE", "K6F_RECORD_ONLY"])
    def test_t6_duplicate_sections_raise_and_write_nothing(self, scratch, no_prompt, route):
        log, cert, _ = scratch
        text = cert.read_text(encoding="utf-8")

        def dup(k):
            return (f"### C03 — Dup {k}\n\n- **Status:** CERTIFIED\n- **Signed:** 2026-10-01 · log entry {k} · "
                    f"results blake2b-128 {'0' * 32}\n- **Certifies:** x\n- **Evidence:** y\n")
        i = text.index("## Scenario sections")
        j = text.index("\n---\n", i)
        cert.write_text(text[:j] + "\n\n" + dup(1) + "\n" + dup(2) + text[j:], encoding="utf-8")
        assert len(_headings(cert.read_text(encoding="utf-8"), "C03")) == 2
        log_before, cert_before = log.read_bytes(), cert.read_bytes()
        r = getattr(cs.Route, route)
        with pytest.raises(ValueError):
            _close(scratch, r, _results("C03", ok=r is cs.Route.K6), "C03", tip_sha="deadbee")
        assert log.read_bytes() == log_before
        assert cert.read_bytes() == cert_before

    def test_t7_awaiting_list_recomputed_after_verdict(self, scratch, no_prompt):
        log, _, _ = scratch
        _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C01", ok=False), "C01")
        _fill_verdict(log, 1, "REPEAT · **rationale:** x")
        _close(scratch, cs.Route.K6, _results("C01", ok=True), "C01")
        assert _awaiting(log.read_text(encoding="utf-8")) == "*(none yet)*"

    def test_t7_awaiting_list_lists_only_blank_verdict_failures(self, scratch, no_prompt):
        log, _, _ = scratch
        _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C01", ok=False), "C01")
        _fill_verdict(log, 1, "REPEAT · **rationale:** x")
        _close(scratch, cs.Route.K6, _results("C01", ok=True), "C01")
        _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C02", ok=False), "C02")
        assert _awaiting(log.read_text(encoding="utf-8")) == "`C02` (entry 3)"

    def test_t8_inv_c_signed_set_equals_gated_set(self, scratch, no_prompt):
        log, cert, nbdir = scratch
        for cid in ("C00", "C01", "C02"):
            _nb(nbdir / f"{cid}_x.ipynb")
        _close(scratch, cs.Route.K6, _results("C00", ok=True), "C00")                      # 1
        _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C01", ok=False), "C01")     # 2
        _close(scratch, cs.Route.K6, _results("C01", ok=True), "C01")                      # 3
        _close(scratch, cs.Route.K6, _results("C02", ok=True), "C02")                      # 4
        _close(scratch, cs.Route.K6F_RECORD_ONLY, _results("C02", ok=False), "C02",
               tip_sha="deadbee")                                                           # 5
        _append_correction(log, 6, "C00", "REPEAT")                                         # 6
        assert _unsign(scratch, "C00") is True                                              # 7
        text = cert.read_text(encoding="utf-8")
        for cid in ("C00", "C01", "C02"):
            assert len(_headings(text, cid)) <= 1
        gated = {p.name[:3] for p in cs.gated_notebooks(log, nbdir)}
        assert gated == {"C01"}
        assert _signed_ids(text) == gated


class TestUnsign:
    """R2 -- ``unsign_scenario(*, cert_id, log_path, certificate_path) -> bool``."""

    @pytest.fixture
    def revoked(self, scratch, no_prompt):
        """C02 signed at K6 (entry 1), then the human's correction entry 2 (CERTIFIED, REPEAT)."""
        log, cert, _ = scratch
        _close(scratch, cs.Route.K6, _results("C02", ok=True), "C02")
        before = cert.read_text(encoding="utf-8")
        _append_correction(log, 2, "C02", "REPEAT", date="2026-10-05")
        return scratch, before

    def test_u1_revokes_the_section(self, revoked):
        scratch, before = revoked
        log, cert, _ = scratch
        log_before = log.read_bytes()
        sec_before = _sec(before, "C02")
        assert _unsign(scratch, "C02") is True
        after = cert.read_text(encoding="utf-8")
        sec = _sec(after, "C02")
        assert _line(sec, "Status") == "- **Status:** REVOKED — section unsigned pending re-run"
        signed = _line(sec, "Signed")
        assert "revoked by the human (log entry 2)" in signed
        assert "blake2b" not in signed
        assert signed == "- **Signed:** *unsigned — revoked by the human (log entry 2); awaiting re-run*"
        for label in ("Certifies", "Evidence"):
            assert _raw_field(sec, label) == _raw_field(sec_before, label)
        row = _matrix_row(after, "C2 ")
        assert row[2] == "REVOKED (section unsigned)"
        assert row[3] == "2026-10-05"
        assert row[4] == "2"
        assert log.read_bytes() == log_before

    def test_u2_refuses_while_gated(self, scratch, no_prompt):
        log, cert, _ = scratch
        _close(scratch, cs.Route.K6, _results("C02", ok=True), "C02")
        log_before, cert_before = log.read_bytes(), cert.read_bytes()
        with pytest.raises(ValueError):
            _unsign(scratch, "C02")
        assert cert.read_bytes() == cert_before
        assert log.read_bytes() == log_before

    def test_u3_second_call_is_a_no_op(self, revoked):
        scratch, _ = revoked
        log, cert, _ = scratch
        assert _unsign(scratch, "C02") is True
        log_before, cert_before = log.read_bytes(), cert.read_bytes()
        assert _unsign(scratch, "C02") is False
        assert cert.read_bytes() == cert_before
        assert log.read_bytes() == log_before

    def test_u3_already_unsigned_k6f_section_is_a_no_op(self, scratch, no_prompt):
        log, cert, _ = scratch
        _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C02", ok=False), "C02")
        _fill_verdict(log, 1, "INVALID · **rationale:** x")
        log_before, cert_before = log.read_bytes(), cert.read_bytes()
        assert _unsign(scratch, "C02") is False
        assert cert.read_bytes() == cert_before
        assert log.read_bytes() == log_before

    def test_u4_no_section_is_a_no_op(self, scratch, no_prompt):
        log, cert, _ = scratch
        _close(scratch, cs.Route.K6F_RECORD_ONLY, _results("C02", ok=False), "C02", tip_sha="deadbee")
        _fill_verdict(log, 1, "DEFECT-FOUND · **rationale:** x")
        assert _headings(cert.read_text(encoding="utf-8"), "C02") == []
        log_before, cert_before = log.read_bytes(), cert.read_bytes()
        assert _unsign(scratch, "C02") is False
        assert cert.read_bytes() == cert_before
        assert log.read_bytes() == log_before

    @pytest.mark.parametrize("cid", ["C07", "C2"], ids=["no-log-entry", "malformed-id"])
    def test_u5_unknown_or_malformed_id_raises(self, scratch, no_prompt, cid):
        log, cert, _ = scratch
        _close(scratch, cs.Route.K6F_MERGE_EVIDENCE, _results("C02", ok=False), "C02")
        log_before, cert_before = log.read_bytes(), cert.read_bytes()
        with pytest.raises(ValueError):
            _unsign(scratch, cid)
        assert cert.read_bytes() == cert_before
        assert log.read_bytes() == log_before

    def test_u6_never_writes_a_human_verdict_word(self, revoked):
        scratch, _ = revoked
        _, cert, _ = scratch
        before = cert.read_text(encoding="utf-8")
        assert _unsign(scratch, "C02") is True
        after = cert.read_text(encoding="utf-8")
        for word in _HUMAN_WORDS:
            assert after.count(word) == before.count(word), word

    def test_u7_k6_reclose_after_unsign_signs_with_the_new_entry(self, revoked):
        scratch, _ = revoked
        _, cert, _ = scratch
        assert _unsign(scratch, "C02") is True
        out = _close(scratch, cs.Route.K6, _results("C02", ok=True), "C02")
        assert out.entry_n == 3
        text = cert.read_text(encoding="utf-8")
        sec = _sec(text, "C02")
        assert re.search(r"^- \*\*Signed:\*\* 2026-10-02 · log entry 3 · results blake2b-128 [0-9a-f]{32}\s*$",
                         sec, re.M)
        assert "REVOKED" not in sec

    def test_u8_never_prompts(self, revoked, no_prompt):
        scratch, _ = revoked
        assert _unsign(scratch, "C02") is True

    def test_u9_works_on_a_provisional_entry(self, scratch, no_prompt):
        log, cert, _ = scratch
        _close(scratch, cs.Route.K6, _results("C03", ok=True), "C03", provisional_on=("C01",))
        _append_correction(log, 2, "C03", "REPEAT", result="CERTIFIED — provisional on C01")
        assert _unsign(scratch, "C03") is True
        sec = _sec(cert.read_text(encoding="utf-8"), "C03")
        assert _line(sec, "Status") == "- **Status:** REVOKED — section unsigned pending re-run"
