"""OMIE-CERT-HARNESS-1 -- the two K0-K7 cycle dry runs, end to end on the
template (decision H12; AC (5), (6)).

Module-scoped fixtures build scratch ``certification-log.md`` and
``certificate.md`` from the **real** files' preamble, entry template and
certificate structure, with entries, TOC rows, the awaiting list, coverage-matrix
status cells and scenario sections reset to the empty state (orchestrator ruling,
a recorded deviation from H12's "copy the real files exactly": the dry runs test
the K0-K7 cycle mechanics, not the live ladder state, which grows through C17).
``_template.ipynb`` is copied as ``C90_dry_pass.ipynb`` and
``C91_dry_fail.ipynb``. Each copy runs through
``execute_notebook``; its route comes from ``route_k5`` (audit twin passed, no
suite regression, independence ok); it is closed with
``close_scenario(order=("C90", "C91", "C92"))``.

* **Pass** must auto-certify (``CERTIFIED`` entry, certificate section signed) and
  advance without asking (``next_scenario == "C91"``).
* **Fail** (``INJECT_FAILURE=True``: the expected trade price moved by +1 tick, the
  engine untouched) must execute cleanly yet fail the gate, write the frozen
  counterexample fixture and section 9, log ``NOT CERTIFIED`` with the Verdict
  blank, start the next scenario (``"C92"``), and be skipped by
  ``gated_notebooks``.
* Also: a crash (no results; explanation missing) and record-only (suite
  regressed: no certificate section, branch not merged).
* The real log, the real certificate and the real ``results/`` are byte-identical
  before and after the module.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pytest

pytestmark = pytest.mark.slow  # slow: each dry run starts a kernel and jit-compiles the record builder (~20-30 s)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_LOG = REPO_ROOT / "docs" / "omiecid" / "certification-log.md"
REAL_CERT = REPO_ROOT / "docs" / "omiecid" / "certificate.md"
NB_DIR = REPO_ROOT / "notebooks" / "certification"
TEMPLATE = NB_DIR / "_template.ipynb"
REAL_RESULTS = NB_DIR / "results"
ORDER = ("C90", "C91", "C92")
DATE = "2026-10-02"
SECTION9 = ("## 9. Why this certification failed", "### 9.1 What failed", "### 9.2 The smallest failing case",
            "### 9.3 Which rule it concerns", "### 9.4 Where the fault most plausibly lies",
            "### 9.5 What it affects downstream")
FAIL_TEXT = "Why this certification failed"


def _cs():
    try:
        from notebooks.certification import cert_support
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"notebooks.certification.cert_support is not importable: {exc!r}")
    return cert_support


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _hash_real() -> dict:
    h = {"log": _sha(REAL_LOG), "certificate": _sha(REAL_CERT)}
    if REAL_RESULTS.exists():
        for f in sorted(REAL_RESULTS.rglob("*")):
            if f.is_file():
                h[str(f.relative_to(NB_DIR))] = _sha(f)
    return h


def _tags(cell):
    return list(cell.get("metadata", {}).get("tags", []))


def _output_text(cell) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if "text" in out:
            parts.append(out["text"] if isinstance(out["text"], str) else "".join(out["text"]))
        for v in out.get("data", {}).values():
            parts.append(v if isinstance(v, str) else json.dumps(v))
    return "\n".join(parts)


def _render_cell(nb):
    cells = [c for c in nb.cells if "section-9-render" in _tags(c)]
    assert len(cells) == 1
    return cells[0]


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


def _field(block: str, label: str) -> str:
    lines = block.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(f"- **{label}"):
            parts = [ln]
            for nxt in lines[i + 1:]:
                if not nxt.strip() or nxt.startswith("- ") or nxt.startswith("#"):
                    break
                parts.append(nxt)
            return re.sub(r"[ \t]+", " ", " ".join(p.strip() for p in parts)).strip()
    raise AssertionError(f"no '{label}' field")


def _cert_section(text: str, heading: str) -> str:
    i = text.index(heading)
    m = re.search(r"^#{2,3} ", text[i + len(heading):], re.M)
    return text[i: i + len(heading) + (m.start() if m else len(text))]


def _awaiting(text: str) -> str:
    a = "**Awaiting the human (K7-U):** "
    i = text.index(a) + len(a)
    return text[i:text.index(" — every", i)]


def _latest(entries, cert_id):
    mine = [e for e in entries if e.cert_id == cert_id]
    assert mine, f"no entry for {cert_id}"
    return max(mine, key=lambda e: e.n)


def _route(cs, gate, *, suite_regressed=False):
    return cs.route_k5(cs.K5Outcome(
        notebook_executed=gate.executed,
        results_status=(gate.results or {}).get("status"),
        audit_twin_passed=True, suite_regressed=suite_regressed, independence_ok=True))


TOC_PLACEHOLDER = ("| — | — | *(no entries yet — the ladder runs in a later session, after the three "
                   "`OMIE-CERT-*` prerequisite stories)* | — | — | — | — |")
AWAIT_OPEN = "**Awaiting the human (K7-U):** "
AWAIT_CLOSE = " — every"


def _skeleton_log_text(real_log: Path = None) -> str:
    """The real log's preamble and entry template with entries, TOC rows and the awaiting
    list reset to the empty state -- independent of how many real entries exist."""
    text = Path(real_log or REAL_LOG).read_text(encoding="utf-8")
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


def _skeleton_cert_text(real_cert: Path = None) -> str:
    """The real certificate with every coverage-matrix Status/Date/Entry cell reset to '—' and
    every scenario section removed (the intro paragraph of '## Scenario sections' is kept)."""
    text = Path(real_cert or REAL_CERT).read_text(encoding="utf-8")
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


def _scratch(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    log, cert, nbdir = root / "certification-log.md", root / "certificate.md", root / "notebooks"
    log.write_text(_skeleton_log_text(), encoding="utf-8")
    cert.write_text(_skeleton_cert_text(), encoding="utf-8")
    nbdir.mkdir(exist_ok=True)
    return log, cert, nbdir


def _dirs(root: Path, tag: str):
    r, f = root / f"results_{tag}", root / f"fixtures_{tag}"
    r.mkdir()
    f.mkdir()
    return r, f


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_hashes():
    return _hash_real()


@pytest.fixture(scope="module")
def cycle(tmp_path_factory, real_hashes):
    cs = _cs()
    root = tmp_path_factory.mktemp("dry_cycle")
    log, cert, nbdir = _scratch(root)
    ns = SimpleNamespace(cs=cs, root=root, log=log, cert=cert, nbdir=nbdir)
    ns.pass_nb, ns.fail_nb = nbdir / "C90_dry_pass.ipynb", nbdir / "C91_dry_fail.ipynb"
    shutil.copyfile(TEMPLATE, ns.pass_nb)
    shutil.copyfile(TEMPLATE, ns.fail_nb)
    ns.pass_res, ns.pass_fix = _dirs(root, "pass")
    ns.fail_res, ns.fail_fix = _dirs(root, "fail")

    # --- pass run --------------------------------------------------------
    ns.pass_gate = cs.execute_notebook(ns.pass_nb, parameters={"CERT_ID": "C90"},
                                       results_dir=ns.pass_res, fixture_dir=ns.pass_fix)
    ns.pass_route = _route(cs, ns.pass_gate)
    ns.pass_close = cs.close_scenario(
        route=ns.pass_route, results=ns.pass_gate.results, log_path=log, certificate_path=cert,
        card="OMIE-CERT-C90-1", branch="omie/cert-c90-dry-pass", commit="0000000", scenario="Dry pass",
        date=DATE, traceback=ns.pass_gate.error, order=ORDER)
    ns.log_after_pass = log.read_text(encoding="utf-8")

    # --- deliberately failed run ------------------------------------------
    ns.fail_gate = cs.execute_notebook(ns.fail_nb, parameters={"CERT_ID": "C91", "INJECT_FAILURE": True},
                                       results_dir=ns.fail_res, fixture_dir=ns.fail_fix)
    ns.fail_route = _route(cs, ns.fail_gate)
    ns.fail_updated = cs.append_failure_section(ns.fail_nb, ns.fail_gate.results, hypothesis="dry-run hypothesis")
    (ns.fail_res / "C91.json").write_text(json.dumps(ns.fail_updated), encoding="utf-8")
    ns.fail_close = cs.close_scenario(
        route=ns.fail_route, results=ns.fail_updated, log_path=log, certificate_path=cert,
        card="OMIE-CERT-C91-1", branch="omie/cert-c91-dry-fail", commit="1111111", scenario="Dry fail",
        date=DATE, traceback=ns.fail_gate.error, order=ORDER)
    ns.gated = cs.gated_notebooks(log, nbdir)
    return ns


@pytest.fixture(scope="module")
def crash(tmp_path_factory, real_hashes):
    cs = _cs()
    root = tmp_path_factory.mktemp("dry_crash")
    log, cert, nbdir = _scratch(root)
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [nbformat.v4.new_code_cell("x = 1"),
                nbformat.v4.new_code_cell("raise RuntimeError('dry-run crash before section 9')")]
    path = nbdir / "C91_dry_crash.ipynb"
    nbformat.write(nb, path)
    res, fix = _dirs(root, "crash")
    gate = cs.execute_notebook(path, results_dir=res, fixture_dir=fix)
    route = _route(cs, gate)
    close = cs.close_scenario(
        route=route, results=None, log_path=log, certificate_path=cert, card="OMIE-CERT-C91-1",
        branch="omie/cert-c91-dry-crash", commit="2222222", scenario="Dry crash", date=DATE,
        traceback=gate.error, order=ORDER)
    return SimpleNamespace(cs=cs, log=log, cert=cert, gate=gate, route=route, close=close)


@pytest.fixture(scope="module")
def record_only(cycle, tmp_path_factory):
    cs = cycle.cs
    root = tmp_path_factory.mktemp("dry_record_only")
    log, cert, _ = _scratch(root)
    cert_before = cert.read_bytes()
    route = _route(cs, cycle.fail_gate, suite_regressed=True)
    close = cs.close_scenario(
        route=route, results=cycle.fail_updated, log_path=log, certificate_path=cert, card="OMIE-CERT-C91-1",
        branch="omie/cert-c91-dry-fail", commit="1111111", scenario="Dry fail", date=DATE,
        tip_sha="3333333", order=ORDER)
    return SimpleNamespace(cs=cs, log=log, cert=cert, cert_before=cert_before, route=route, close=close)


# ---------------------------------------------------------------------------
# The passing dry run
# ---------------------------------------------------------------------------
class TestPassingDryRun:
    def test_gate_passes(self, cycle):
        g = cycle.pass_gate
        assert g.executed is True, g.error
        assert g.results_errors == []
        assert g.results["status"] == "PASS"
        assert g.results["failure_kinds"] == []
        assert g.results["cert_id"] == "C90"
        assert g.passed is True
        assert (cycle.pass_res / "C90.json").is_file()
        assert cycle.cs.validate_results(json.loads((cycle.pass_res / "C90.json").read_text())) == []

    def test_route_is_k6(self, cycle):
        assert cycle.pass_route is cycle.cs.Route.K6

    def test_log_entry_certified(self, cycle):
        e = _latest(cycle.cs.parse_log(cycle.log), "C90")
        assert e.result == "CERTIFIED"
        assert e.verdict == "CERTIFIED"

    def test_certificate_section_signed(self, cycle):
        sec = _cert_section(cycle.cert.read_text(encoding="utf-8"), "### C90 — Dry pass")
        signed = _field(sec, "Signed")
        assert re.search(rf"{DATE} · log entry {cycle.pass_close.entry_n} · results blake2b-128 [0-9a-f]{{32}}",
                         signed)
        assert "unsigned" not in signed

    def test_advances_without_asking(self, cycle):
        assert cycle.pass_close.next_scenario == "C91"
        assert cycle.pass_close.merged is True and cycle.pass_close.signed is True

    def test_render_cell_shows_no_failure_section(self, cycle):
        assert FAIL_TEXT not in _output_text(_render_cell(cycle.pass_gate.notebook))

    def test_pass_notebook_file_never_written(self, cycle):
        assert cycle.pass_nb.read_bytes() == TEMPLATE.read_bytes()


# ---------------------------------------------------------------------------
# The deliberately failed dry run
# ---------------------------------------------------------------------------
class TestFailedDryRun:
    def test_executes_but_fails_the_gate(self, cycle):
        g = cycle.fail_gate
        assert g.executed is True, g.error
        assert g.passed is False
        assert g.results_errors == []
        assert g.results["status"] == "NOT CERTIFIED"
        assert g.results["failure_kinds"] == ["oracle_mismatch"]

    def test_counterexample_fixture_frozen(self, cycle):
        files = sorted(cycle.fail_fix.glob("C91_seed*.npz"))
        assert files, "no counterexample fixture written"
        with np.load(files[0], allow_pickle=False) as z:
            assert z["tape"].dtype == np.int32
            v = z["params_json"]
            s = v.item() if v.shape == () else bytes(v).decode("utf-8")
            json.loads(s.decode("utf-8") if isinstance(s, bytes) else s)
        sc = cycle.fail_gate.results["failure_explanation"]["smallest_case"]
        assert sc is not None and Path(sc["fixture"]).name in {f.name for f in files}

    def test_render_cell_shows_the_failure_section(self, cycle):
        assert FAIL_TEXT in _output_text(_render_cell(cycle.fail_gate.notebook))

    def test_route_is_merge_evidence(self, cycle):
        assert cycle.fail_route is cycle.cs.Route.K6F_MERGE_EVIDENCE

    def test_section_9_persisted_last(self, cycle):
        nb = nbformat.read(cycle.fail_nb, as_version=4)
        last = nb.cells[-1]
        assert last.cell_type == "markdown" and "section-9" in _tags(last)
        pos = [last.source.index(h) for h in SECTION9]
        assert pos == sorted(pos)
        s94 = last.source[last.source.index(SECTION9[4]):last.source.index(SECTION9[5])]
        assert "Hypothesis:" in s94 and "dry-run hypothesis" in s94
        assert cycle.cs.notebook_structure_errors(nb) == []

    def test_updated_results_carry_the_hypothesis(self, cycle):
        assert cycle.fail_updated["failure_explanation"]["hypothesis"] == "dry-run hypothesis"
        assert cycle.cs.validate_results(cycle.fail_updated) == []

    def test_log_entry_not_certified_verdict_blank(self, cycle):
        e = _latest(cycle.cs.parse_log(cycle.log), "C91")
        assert e.result == "NOT CERTIFIED"
        assert e.verdict == ""
        block = _entry_block(cycle.log.read_text(encoding="utf-8"), e.n)
        para = cycle.fail_updated["failure_explanation"]["paragraph"]
        assert para and para in _field(block, "Why it failed")

    def test_awaiting_list_names_c91(self, cycle):
        assert "`C91`" in _awaiting(cycle.log.read_text(encoding="utf-8"))

    def test_certificate_section_unsigned(self, cycle):
        sec = _cert_section(cycle.cert.read_text(encoding="utf-8"), "### C91 — Dry fail")
        assert _field(sec, "Signed") == "- **Signed:** *unsigned — NOT CERTIFIED; awaiting the human (K7-U)*"

    def test_starts_the_next_scenario(self, cycle):
        assert cycle.fail_close.next_scenario == "C92"
        assert cycle.fail_close.merged is True and cycle.fail_close.signed is False

    def test_merge_gate_skips_the_failed_notebook(self, cycle):
        assert cycle.gated == [cycle.pass_nb]


# ---------------------------------------------------------------------------
# Crash and record-only
# ---------------------------------------------------------------------------
class TestCrashAndRecordOnly:
    def test_crash_is_not_executed(self, crash):
        assert crash.gate.executed is False
        assert crash.gate.passed is False
        assert "dry-run crash before section 9" in crash.gate.error

    def test_crash_routes_merge_evidence(self, crash):
        assert crash.route is crash.cs.Route.K6F_MERGE_EVIDENCE

    def test_crash_entry_explanation_missing(self, crash):
        e = _latest(crash.cs.parse_log(crash.log), "C91")
        assert e.result == "NOT CERTIFIED" and e.verdict == ""
        block = _entry_block(crash.log.read_text(encoding="utf-8"), e.n)
        assert "explanation missing — notebook crashed; traceback:" in _field(block, "Why it failed")

    def test_record_only_route(self, record_only):
        assert record_only.route is record_only.cs.Route.K6F_RECORD_ONLY
        assert record_only.close.merged is False and record_only.close.signed is False

    def test_record_only_writes_no_certificate_section(self, record_only):
        assert record_only.cert.read_bytes() == record_only.cert_before
        assert "### C91 —" not in record_only.cert.read_text(encoding="utf-8")

    def test_record_only_merged_line(self, record_only):
        e = _latest(record_only.cs.parse_log(record_only.log), "C91")
        assert e.result == "NOT CERTIFIED" and e.verdict == ""
        merged = _field(_entry_block(record_only.log.read_text(encoding="utf-8"), e.n), "Merged")
        assert "branch NOT merged" in merged
        assert "omie/cert-c91-dry-fail" in merged and "3333333" in merged


# ---------------------------------------------------------------------------
# The real documents are never touched
# ---------------------------------------------------------------------------
def test_zz_real_docs_and_results_unchanged(cycle, crash, record_only, real_hashes):
    assert _hash_real() == real_hashes
