"""OMIE-CERT-HARNESS-1 -- harness layout, notebook structure and the import
boundary (decisions H1, H2, H8, H15).

Locked behavioural contract:

* the H1 files exist (README, results schema, template, ``.gitattributes``,
  ``.gitignore``, ``.gitkeep``s, the two empty package ``__init__``s), and the
  H1 doc edit is made (``certificate.md`` ``## Scenario sections``);
* ``notebook_structure_errors(_template) == []`` and the same for every
  ``C??_*.ipynb`` present; each structural rule of H8 is seen failing on a
  deliberately broken copy of the template;
* ``append_failure_section`` writes the section-9 markdown cell last, with
  9.1-9.5 in order and 9.4 prefixed "Hypothesis:";
* the H2 boundary: ``cert_support.py`` imports at module level only the standard
  library, ``numpy``, ``jsonschema``, ``nbformat`` and ``nbclient``; nothing under
  ``omiecid_simulator/`` or ``tests/omiecid/reference/`` imports the harness or the
  notebook tooling. The AST scanners are themselves shown flagging a broken input.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
NB_DIR = REPO_ROOT / "notebooks" / "certification"
TEMPLATE = NB_DIR / "_template.ipynb"
CERT_SUPPORT = NB_DIR / "cert_support.py"
HEADINGS = ("## 1. Purpose", "## 2. The rule, quoted", "## 3. Configuration", "## 4. Dataset",
            "## 5. Known result", "## 6. Run", "## 7. Certification", "## 8. Verdict")
SECTION9 = ("## 9. Why this certification failed", "### 9.1 What failed", "### 9.2 The smallest failing case",
            "### 9.3 Which rule it concerns", "### 9.4 Where the fault most plausibly lies",
            "### 9.5 What it affects downstream")


@pytest.fixture
def cs():
    try:
        from notebooks.certification import cert_support
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"notebooks.certification.cert_support is not importable: {exc!r}")
    return cert_support


@pytest.fixture
def template():
    return nbformat.read(TEMPLATE, as_version=4)


def _tags(cell):
    return list(cell.get("metadata", {}).get("tags", []))


def _heading_index(nb, heading):
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and re.search(rf"^{re.escape(heading)}\s*$", c.source, re.M):
            return i
    raise AssertionError(f"heading {heading!r} not found")


def _section_cells(nb, k):
    return [c for c in nb.cells if f"section-{k}" in _tags(c)]


def _section_source(nb, k):
    return "\n".join(c.source for c in _section_cells(nb, k))


# ---------------------------------------------------------------------------
# H1 files and doc edits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rel", [
    "notebooks/__init__.py", "notebooks/certification/__init__.py", "notebooks/certification/cert_support.py",
    "notebooks/certification/results.schema.json", "notebooks/certification/_template.ipynb",
    "notebooks/certification/README.md", "notebooks/certification/.gitattributes",
    "notebooks/certification/.gitignore", "notebooks/certification/results/.gitkeep",
    "notebooks/certification/fixtures/.gitkeep", "tests/omiecid/certification/__init__.py",
])
def test_harness_file_exists(rel):
    assert (REPO_ROOT / rel).is_file(), rel


@pytest.mark.parametrize("rel", ["notebooks/__init__.py", "notebooks/certification/__init__.py"])
def test_package_inits_are_empty(rel):
    assert (REPO_ROOT / rel).read_text(encoding="utf-8").strip() == ""


def test_gitignore_entries():
    lines = {ln.strip() for ln in (NB_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert {".ipynb_checkpoints/", "__pycache__/", "results/C9?.json", "fixtures/C9?_*"} <= lines


def test_results_dir_is_not_ignored_for_ladder_ids():
    lines = [ln.strip() for ln in (NB_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "results/" not in lines and "results/*" not in lines and "results/*.json" not in lines


def test_readme_records_the_git_filter_commands():
    text = (NB_DIR / "README.md").read_text(encoding="utf-8")
    for key in ("filter.nbstripout.clean", "filter.nbstripout.smudge", "filter.nbstripout.required",
                "diff.ipynb.textconv"):
        assert key in text, key


def test_certificate_has_scenario_sections_heading():
    text = (REPO_ROOT / "docs" / "omiecid" / "certificate.md").read_text(encoding="utf-8")
    assert re.search(r"^## Scenario sections\s*$", text, re.M)
    assert text.index("## Scenario sections") < text.index("## Appendix A")
    sec = text[text.index("## Scenario sections"):text.index("## Appendix A")]
    assert "close_scenario" in sec


# ---------------------------------------------------------------------------
# H8 -- template structure and content
# ---------------------------------------------------------------------------
def test_template_structure_is_clean(cs, template):
    assert cs.notebook_structure_errors(template) == []


_LADDER_NOTEBOOKS = sorted(NB_DIR.glob("C[0-9][0-9]_*.ipynb"))


@pytest.mark.parametrize("path", _LADDER_NOTEBOOKS, ids=[p.stem for p in _LADDER_NOTEBOOKS])
def test_ladder_notebook_structure_is_clean(cs, path):
    assert cs.notebook_structure_errors(nbformat.read(path, as_version=4)) == []


def test_template_headings_in_order(template):
    idx = [_heading_index(template, h) for h in HEADINGS]
    assert idx == sorted(idx)


def test_every_template_cell_tagged_with_its_section(template):
    current = 0
    for c in template.cells:
        sec = [t for t in _tags(c) if re.fullmatch(r"section-\d", t)]
        if c is template.cells[-1]:
            assert "section-9-render" in _tags(c)
            continue
        assert len(sec) == 1, f"cell without exactly one section tag: {c.source[:60]!r}"
        k = int(sec[0][-1])
        assert k >= current, "section tags out of order"
        current = k


def test_template_parameter_cell(template):
    params = [c for c in template.cells if "parameters" in _tags(c)]
    assert len(params) == 1
    p = params[0]
    assert p.cell_type == "code" and "section-3" in _tags(p)
    for line in ('CERT_ID = "C99"', "DRAWS = 3", "SEED0 = 0", "INJECT_FAILURE = False"):
        assert re.search(rf"^{re.escape(line)}\s*(#.*)?$", p.source, re.M), line


def test_template_last_cell_renders_section_9(template):
    last = template.cells[-1]
    assert last.cell_type == "code"
    assert "section-9-render" in _tags(last)
    assert "failure_section_markdown" in last.source
    assert not any("## 9. Why this certification failed" in c.source
                   for c in template.cells if c.cell_type == "markdown")


def test_template_content(template):
    allsrc = "\n".join(c.source for c in template.cells)
    assert "C02.crossing_pair" in allsrc
    assert "notebooks.certification.cert_support" in allsrc
    assert "sys.path" not in allsrc, "the harness is imported by its package path, never via sys.path"
    s2 = _section_source(template, 2)
    assert "\u00a76.2 item 3" in s2 and "resting order's limit price" in s2 and "market-spec.md" in s2
    s5 = _section_source(template, 5)
    assert "p_rest_int" in s5 and "q_rest_int" in s5 and "q_agg_int" in s5 and "INJECT_FAILURE" in s5
    assert ".expected[" not in s5 and ".expected.get" not in s5
    assert any(c.cell_type == "markdown" for c in _section_cells(template, 5))
    s6 = _section_source(template, 6)
    assert "jit=False" in s6 and "jit=True" in s6 and "compare_records" in s6
    s7 = _section_source(template, 7)
    assert "run_battery" in s7 and "expects_trade=True" in s7
    assert "matplotlib" in allsrc


def test_template_committed_without_outputs(template):
    for c in template.cells:
        if c.cell_type == "code":
            assert c.outputs == [] and c.execution_count is None


def test_template_kernelspec(template):
    assert template.metadata["kernelspec"]["name"] == "python3"


# ---------------------------------------------------------------------------
# H8 -- each structural rule seen failing
# ---------------------------------------------------------------------------
def _broken(template, how):
    nb = copy.deepcopy(template)
    if how == "headings-swapped":
        i, j = _heading_index(nb, HEADINGS[0]), _heading_index(nb, HEADINGS[1])
        nb.cells[i].source, nb.cells[j].source = nb.cells[j].source, nb.cells[i].source
    elif how == "heading-renamed":
        i = _heading_index(nb, "## 5. Known result")
        nb.cells[i].source = nb.cells[i].source.replace("## 5. Known result", "## 5. Expected result")
    elif how == "untagged-cell":
        c = _section_cells(nb, 4)[-1]
        c.metadata["tags"] = [t for t in _tags(c) if t != "section-4"]
    elif how == "tag-out-of-order":
        c = _section_cells(nb, 6)[-1]
        c.metadata["tags"] = [("section-2" if t == "section-6" else t) for t in _tags(c)]
    elif how == "no-parameters-cell":
        for c in nb.cells:
            c.metadata["tags"] = [t for t in _tags(c) if t != "parameters"]
    elif how == "two-parameters-cells":
        i = next(k for k, c in enumerate(nb.cells) if "parameters" in _tags(c))
        nb.cells.insert(i + 1, copy.deepcopy(nb.cells[i]))
    elif how == "parameters-outside-section-3":
        c = next(c for c in nb.cells if "parameters" in _tags(c))
        c.metadata["tags"] = [("section-4" if t == "section-3" else t) for t in _tags(c)]
    elif how == "cell-after-render":
        nb.cells.append(nbformat.v4.new_code_cell("x = 1", metadata={"tags": ["section-8"]}))
    elif how == "render-cell-missing":
        nb.cells = [c for c in nb.cells if "section-9-render" not in _tags(c)]
    elif how == "section-9-not-last":
        nb.cells.insert(len(nb.cells) - 1, nbformat.v4.new_markdown_cell(
            "\n\n".join(SECTION9), metadata={"tags": ["section-9"]}))
    elif how == "heading-missing":
        i = _heading_index(nb, "## 7. Certification")
        nb.cells[i].source = nb.cells[i].source.replace("## 7. Certification", "Certification")
    else:  # pragma: no cover
        raise ValueError(how)
    return nb


@pytest.mark.parametrize("how", [
    "headings-swapped", "heading-renamed", "heading-missing", "untagged-cell", "tag-out-of-order",
    "no-parameters-cell", "two-parameters-cells", "parameters-outside-section-3", "cell-after-render",
    "render-cell-missing", "section-9-not-last",
])
def test_structure_errors_detect_broken_template(cs, template, how):
    assert cs.notebook_structure_errors(_broken(template, how)) != []


# ---------------------------------------------------------------------------
# H8 -- append_failure_section
# ---------------------------------------------------------------------------
def _not_certified_results(cs, cert_id="C91"):
    from tests.omiecid.reference.omie_invariants import BatteryReport, CheckResult
    cert = cs.Certification(
        cert_id=cert_id, notebook=f"{cert_id}_probe.ipynb", scenario="Probe",
        known_result="p_trade = p_rest", oracle_kind="exact-parametric",
        quotes=(cs.SpecQuote("market-spec.md", "\u00a76.2 item 3",
                             "Trade executes at the **resting order's limit price** (maker pricing)."),),
        requested_draws=1)
    cert.check("trade price equals resting price", False, seed=0, expected=100, observed=101, unit="price_int")
    cert.add_battery(BatteryReport(results=tuple(
        CheckResult(check_id=c, status="pass", violations=(), unobserved=(), evidence={})
        for c in ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9"))), seed=0)
    return cert.payload()


def test_append_failure_section(cs, tmp_path):
    path = tmp_path / "C91_probe.ipynb"
    path.write_bytes(TEMPLATE.read_bytes())
    results = _not_certified_results(cs)
    updated = cs.append_failure_section(path, results, hypothesis="the probe hypothesis",
                                        downstream="the probe downstream")
    nb = nbformat.read(path, as_version=4)
    last = nb.cells[-1]
    assert last.cell_type == "markdown" and "section-9" in _tags(last)
    pos = [last.source.index(h) for h in SECTION9]
    assert pos == sorted(pos)
    s94 = last.source[last.source.index(SECTION9[4]):last.source.index(SECTION9[5])]
    assert "Hypothesis:" in s94 and "the probe hypothesis" in s94
    assert "the probe downstream" in last.source[last.source.index(SECTION9[5]):]
    assert sum(1 for c in nb.cells if "section-9" in _tags(c)) == 1
    assert cs.notebook_structure_errors(nb) == []
    assert updated["failure_explanation"]["hypothesis"] == "the probe hypothesis"
    assert cs.validate_results(updated) == []


def test_append_failure_section_keeps_sections_1_to_8(cs, tmp_path):
    path = tmp_path / "C91_probe.ipynb"
    path.write_bytes(TEMPLATE.read_bytes())
    before = nbformat.read(path, as_version=4)
    cs.append_failure_section(path, _not_certified_results(cs), hypothesis="h")
    after = nbformat.read(path, as_version=4)
    assert [c.source for c in after.cells[:-1]] == [c.source for c in before.cells]


# ---------------------------------------------------------------------------
# H2 -- the import boundary
# ---------------------------------------------------------------------------
ALLOWED_THIRD_PARTY = {"numpy", "jsonschema", "nbformat", "nbclient"}
FORBIDDEN_IN_ENGINE = ("notebooks", "nbformat", "nbclient", "nbstripout")


def _module_level_imports(tree: ast.AST):
    """Imports executed at import time: everything except function and lambda bodies."""
    out = []

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Import):
                out.extend(a.name for a in child.names)
            elif isinstance(child, ast.ImportFrom):
                out.append("." * child.level + (child.module or ""))
            visit(child)

    visit(tree)
    return out


def _boundary_violations(source: str):
    bad = []
    for name in _module_level_imports(ast.parse(source)):
        top = name.split(".")[0]
        if name.startswith(".") or (top not in sys.stdlib_module_names and top not in ALLOWED_THIRD_PARTY):
            bad.append(name)
    return bad


def _all_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _regex_imports(source: str):
    """Fallback for files this interpreter cannot parse (e.g. Python 3.12 f-string syntax)."""
    for m in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import\b|import\s+([\w., ]+))", source, re.M):
        if m.group(1):
            yield m.group(1)
        else:
            yield from (part.strip().split(" ")[0] for part in m.group(2).split(",") if part.strip())


def _engine_violations(source: str):
    try:
        names = list(_all_imports(ast.parse(source)))
    except SyntaxError:
        names = list(_regex_imports(source))
    return [n for n in names if n.split(".")[0] in FORBIDDEN_IN_ENGINE or "cert_support" in n]


@pytest.mark.parametrize("src,flagged", [
    ("import jax\n", ["jax"]),
    ("from omiecid_simulator.jaxomiecid import omie_tape_synth\n", ["omiecid_simulator.jaxomiecid"]),
    ("class A:\n    import jax\n", ["jax"]),
    ("try:\n    import omiecid_simulator\nexcept ImportError:\n    pass\n", ["omiecid_simulator"]),
    ("from . import other\n", ["."]),
    ("def f():\n    import jax\n    from tests.omiecid.reference import omie_invariants\n", []),
    ("import json, re, hashlib\nimport numpy as np\nimport jsonschema, nbformat\nfrom nbclient import NotebookClient\n",
     []),
])
def test_boundary_scanner_self_check(src, flagged):
    assert _boundary_violations(src) == flagged


def test_engine_scanner_self_check():
    assert _engine_violations("def f():\n    import nbclient\n") == ["nbclient"]
    assert _engine_violations("from notebooks.certification.cert_support import route_k5\n") == [
        "notebooks.certification.cert_support"]
    assert _engine_violations("import numpy\n") == []
    unparsable = 'x = f"{d["k"]}"\nimport nbformat\n'
    assert _engine_violations(unparsable) == ["nbformat"]


def test_cert_support_module_level_imports():
    assert _boundary_violations(CERT_SUPPORT.read_text(encoding="utf-8")) == []


def test_engine_and_reference_never_import_the_harness():
    bad = {}
    for root in (REPO_ROOT / "omiecid_simulator", REPO_ROOT / "tests" / "omiecid" / "reference"):
        for py in root.rglob("*.py"):
            v = _engine_violations(py.read_text(encoding="utf-8"))
            if v:
                bad[str(py.relative_to(REPO_ROOT))] = v
    assert bad == {}


def test_harness_import_and_light_calls_load_no_jax():
    """Importing the harness and using the quote, routing and log surfaces never imports JAX."""
    code = (
        "import sys\n"
        "import notebooks.certification.cert_support as cs\n"
        "cs.check_quote(cs.SpecQuote('market-spec.md', '\u00a76.2 item 3', "
        "\"Trade executes at the resting order's limit price\"))\n"
        "cs.route_k5(cs.K5Outcome(True, 'PASS', True, False, True))\n"
        f"cs.parse_log({str(REPO_ROOT / 'docs' / 'omiecid' / 'certification-log.md')!r})\n"
        "bad = sorted(k for k in sys.modules if k == 'jax' or k.startswith(('jax.', 'jaxlib', 'omiecid_simulator',"
        " 'tests.omiecid.reference')))\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_schema_file_is_json():
    json.loads((NB_DIR / "results.schema.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Step-5 re-close ruling (agile-plan v5.189, R2) -- the unsign writer is documented
# ---------------------------------------------------------------------------
def test_s1_certificate_scenario_sections_names_unsign_scenario():
    text = (REPO_ROOT / "docs" / "omiecid" / "certificate.md").read_text(encoding="utf-8")
    sec = text[text.index("## Scenario sections"):text.index("## Appendix A")]
    assert "unsign_scenario" in sec
