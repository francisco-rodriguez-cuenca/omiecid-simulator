"""OMIE-CERT-HARNESS-1 -- the headless nbclient gate (decision H7; AC (1)).

Locked behavioural contract for ``execute_notebook(path, *, parameters=None,
results_dir=None, fixture_dir=None, timeout=1800, startup_timeout=120) ->
GateResult``:

* the **control test**: a notebook whose last cell raises fails the gate
  (``pytest --nbmake`` once reported a pass for exactly this, which
  disqualified it);
* a notebook that executes cleanly but emits ``NOT CERTIFIED``, no JSON, an
  invalid JSON or two JSONs fails the gate; only a clean run with one valid
  ``PASS`` JSON passes;
* the kernel runs the caller's interpreter and sees ``OMIE_CERT_RESULTS_DIR``,
  ``OMIE_CERT_FIXTURE_DIR``, ``JAX_PLATFORMS=cpu``, ``CUDA_VISIBLE_DEVICES=""``
  and an absolute ``PYTHONPATH`` headed by the repository root;
* parameter injection overrides the single ``parameters`` cell, rejects
  non-literal values and rejects zero or two ``parameters`` cells;
* the notebook file on disk is never written.

Every test that starts a kernel is marked ``slow`` (kernel start-up alone is
1-3 s, over the 1 s fast-suite budget).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import nbformat
import pytest

from tests.omiecid.reference.omie_invariants import BatteryReport, CheckResult

try:  # collection must succeed before the harness exists; the tests then fail
    from notebooks.certification import cert_support as cs
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    cs = None
    _IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_RESULTS = REPO_ROOT / "notebooks" / "certification" / "results"
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}

slow_kernel = pytest.mark.slow  # slow: starts a Jupyter kernel (1-3 s start-up)


@pytest.fixture(autouse=True)
def _require_harness():
    if cs is None:
        pytest.fail(f"notebooks.certification.cert_support is not importable: {_IMPORT_ERROR!r}")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _code(src: str, *tags: str):
    cell = nbformat.v4.new_code_cell(src)
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def _write_nb(path: Path, cells) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = dict(KERNELSPEC)
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = list(cells)
    nbformat.write(nb, path)
    return path


def _report() -> BatteryReport:
    return BatteryReport(results=tuple(
        CheckResult(check_id=c, status="pass", violations=(), unobserved=(), evidence={})
        for c in ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")))


def _payload(*, ok: bool, cert_id: str = "C98") -> dict:
    cert = cs.Certification(
        cert_id=cert_id, notebook=f"{cert_id}_probe.ipynb", scenario="Gate probe",
        known_result="p_trade = p_rest", oracle_kind="exact-parametric",
        quotes=(cs.SpecQuote("market-spec.md", "§6.2 item 3",
                             "Trade executes at the **resting order's limit price** (maker pricing)."),),
        requested_draws=1)
    cert.check("probe", ok, seed=0, expected=1, observed=1 if ok else 2, unit="price_int")
    cert.add_battery(_report(), seed=0)
    p = cert.payload()
    assert cs.validate_results(p) == []
    return p


def _emit(payload: dict, name: str = "C98.json"):
    return _code(
        "import os, pathlib\n"
        f"pathlib.Path(os.environ['OMIE_CERT_RESULTS_DIR'], {name!r}).write_text({json.dumps(payload)!r})\n")


def _probe(out: Path):
    return _code(
        "import json, os, sys, pathlib\n"
        "import notebooks.certification.cert_support as _cs\n"
        "keys = ('OMIE_CERT_RESULTS_DIR', 'OMIE_CERT_FIXTURE_DIR', 'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES',\n"
        "        'PYTHONPATH')\n"
        f"pathlib.Path({str(out)!r}).write_text(json.dumps({{\n"
        "    'executable': sys.executable, 'cwd': os.getcwd(), 'cs_file': _cs.__file__,\n"
        "    'env': {k: os.environ.get(k, '<<unset>>') for k in keys}}))\n")


def _dirs(tmp_path: Path) -> dict:
    r, f = tmp_path / "r", tmp_path / "f"
    r.mkdir(exist_ok=True)
    f.mkdir(exist_ok=True)
    return {"results_dir": r, "fixture_dir": f}


def _digest(path: Path):
    st = path.stat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), st.st_mtime_ns


# ---------------------------------------------------------------------------
# The control test and the pass/fail semantics
# ---------------------------------------------------------------------------
@slow_kernel
def test_control_raising_last_cell_fails_the_gate(tmp_path):
    nb = _write_nb(tmp_path / "broken.ipynb", [
        _code("print('this cell is fine')"),
        _code("raise AssertionError('deliberate failure in the last cell')"),
    ])
    before = _digest(nb)
    g = cs.execute_notebook(nb, **_dirs(tmp_path))
    assert g.executed is False
    assert g.passed is False
    assert "deliberate failure in the last cell" in (g.error or "")
    assert _digest(nb) == before


@slow_kernel
def test_raise_after_a_valid_pass_json_still_fails(tmp_path):
    nb = _write_nb(tmp_path / "late_raise.ipynb", [
        _emit(_payload(ok=True)),
        _code("raise RuntimeError('raised after the results were written')"),
    ])
    g = cs.execute_notebook(nb, **_dirs(tmp_path))
    assert g.executed is False
    assert g.passed is False


@slow_kernel
def test_clean_run_with_pass_json_passes(tmp_path):
    nb = _write_nb(tmp_path / "good.ipynb", [_emit(_payload(ok=True))])
    g = cs.execute_notebook(nb, **_dirs(tmp_path))
    assert g.executed is True and g.error is None
    assert g.results is not None and g.results["status"] == "PASS"
    assert g.results_errors == []
    assert g.passed is True
    assert Path(g.path) == nb
    assert isinstance(g.notebook, nbformat.NotebookNode)


@slow_kernel
def test_not_certified_executes_but_fails_the_gate(tmp_path):
    nb = _write_nb(tmp_path / "nc.ipynb", [_emit(_payload(ok=False))])
    g = cs.execute_notebook(nb, **_dirs(tmp_path))
    assert g.executed is True
    assert g.results["status"] == "NOT CERTIFIED"
    assert g.results_errors == []
    assert g.passed is False


@slow_kernel
def test_no_results_json_fails_the_gate(tmp_path):
    nb = _write_nb(tmp_path / "silent.ipynb", [_code("x = 1 + 1")])
    g = cs.execute_notebook(nb, **_dirs(tmp_path))
    assert g.executed is True
    assert g.results is None
    assert g.passed is False


@slow_kernel
def test_invalid_results_json_fails_the_gate(tmp_path):
    bad = _payload(ok=True)
    bad["schema_version"] = 2
    nb = _write_nb(tmp_path / "invalid.ipynb", [_emit(bad)])
    g = cs.execute_notebook(nb, **_dirs(tmp_path))
    assert g.executed is True
    assert g.results_errors != []
    assert g.passed is False


@slow_kernel
def test_two_results_jsons_fail_the_gate(tmp_path):
    nb = _write_nb(tmp_path / "two.ipynb", [
        _emit(_payload(ok=True), "C98.json"),
        _emit(_payload(ok=True, cert_id="C97"), "C97.json"),
    ])
    g = cs.execute_notebook(nb, **_dirs(tmp_path))
    assert g.results_errors != []
    assert g.passed is False


@slow_kernel
def test_dead_kernel_fails_the_gate(tmp_path):
    nb = _write_nb(tmp_path / "dead.ipynb", [_code("import os\nos._exit(3)")])
    g = cs.execute_notebook(nb, **_dirs(tmp_path), timeout=60)
    assert g.executed is False
    assert g.passed is False
    assert g.error


@slow_kernel
def test_timeout_fails_the_gate(tmp_path):
    nb = _write_nb(tmp_path / "slow.ipynb", [_code("import time\ntime.sleep(60)")])
    g = cs.execute_notebook(nb, **_dirs(tmp_path), timeout=3)
    assert g.executed is False
    assert g.passed is False


# ---------------------------------------------------------------------------
# Kernel interpreter and environment
# ---------------------------------------------------------------------------
@slow_kernel
def test_kernel_interpreter_and_environment(tmp_path):
    rdir, fdir = tmp_path / "r", tmp_path / "f"
    rdir.mkdir()
    fdir.mkdir()
    out = tmp_path / "probe.json"
    nbdir = tmp_path / "elsewhere"
    nbdir.mkdir()
    nb = _write_nb(nbdir / "probe.ipynb", [_probe(out)])
    g = cs.execute_notebook(nb, results_dir=rdir, fixture_dir=fdir)
    assert g.executed is True, g.error
    probe = json.loads(out.read_text(encoding="utf-8"))
    assert probe["executable"] == sys.executable
    assert Path(probe["cwd"]).resolve() == nbdir.resolve()
    env = probe["env"]
    assert Path(env["OMIE_CERT_RESULTS_DIR"]).resolve() == rdir.resolve()
    assert Path(env["OMIE_CERT_FIXTURE_DIR"]).resolve() == fdir.resolve()
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    first = env["PYTHONPATH"].split(os.pathsep)[0]
    assert Path(first).is_absolute() and Path(first).resolve() == REPO_ROOT.resolve()
    assert Path(probe["cs_file"]).resolve() == (REPO_ROOT / "notebooks" / "certification" / "cert_support.py").resolve()


@slow_kernel
def test_default_dirs_are_fresh_temporaries_not_the_committed_results(tmp_path):
    out = tmp_path / "probe.json"
    committed_before = sorted(p.name for p in COMMITTED_RESULTS.glob("*")) if COMMITTED_RESULTS.exists() else []
    nb = _write_nb(tmp_path / "defaults.ipynb", [_probe(out), _emit(_payload(ok=True), "C98.json")])
    g = cs.execute_notebook(nb)
    assert g.executed is True, g.error
    env = json.loads(out.read_text(encoding="utf-8"))["env"]
    for key in ("OMIE_CERT_RESULTS_DIR", "OMIE_CERT_FIXTURE_DIR"):
        d = Path(env[key])
        assert d.is_absolute()
        assert REPO_ROOT.resolve() not in d.resolve().parents, f"{key} must not point into the repository"
    assert g.results is not None and g.results["cert_id"] == "C98"
    after = sorted(p.name for p in COMMITTED_RESULTS.glob("*")) if COMMITTED_RESULTS.exists() else []
    assert after == committed_before


# ---------------------------------------------------------------------------
# Parameter injection
# ---------------------------------------------------------------------------
def _param_nb(path: Path, out: Path, n_param_cells: int = 1) -> Path:
    cells = [_code("import json, pathlib")]
    cells += [_code("A = 1\nB = 1.0\nC = 'x'\nD = False\nE = 0", "section-3", "parameters")
              for _ in range(n_param_cells)]
    cells.append(_code(f"pathlib.Path({str(out)!r}).write_text(json.dumps([repr(v) for v in (A, B, C, D, E)]))"))
    return _write_nb(path, cells)


@slow_kernel
def test_parameter_injection_overrides(tmp_path):
    out = tmp_path / "params.json"
    nb = _param_nb(tmp_path / "params.ipynb", out)
    before = _digest(nb)
    g = cs.execute_notebook(nb, parameters={"A": 7, "B": 2.5, "C": "hi", "D": True, "E": None},
                            **_dirs(tmp_path))
    assert g.executed is True, g.error
    assert json.loads(out.read_text(encoding="utf-8")) == ["7", "2.5", "'hi'", "True", "None"]
    cells = g.notebook.cells
    i = next(k for k, c in enumerate(cells) if "parameters" in c.metadata.get("tags", []))
    injected = cells[i + 1]
    assert injected.cell_type == "code"
    assert "injected-parameters" in injected.metadata.get("tags", [])
    assert "A = 7" in injected.source
    assert _digest(nb) == before, "the notebook file on disk must never be written"


@slow_kernel
def test_without_parameters_the_committed_values_run(tmp_path):
    out = tmp_path / "params.json"
    nb = _param_nb(tmp_path / "params.ipynb", out)
    g = cs.execute_notebook(nb, **_dirs(tmp_path))
    assert g.executed is True, g.error
    assert json.loads(out.read_text(encoding="utf-8")) == ["1", "1.0", "'x'", "False", "0"]


@pytest.mark.parametrize("value", [[1], (1,), {"a": 1}, object(), {1}, b"bytes"],
                         ids=["list", "tuple", "dict", "object", "set", "bytes"])
def test_non_literal_parameter_rejected(tmp_path, value):
    nb = _param_nb(tmp_path / "params.ipynb", tmp_path / "params.json")
    with pytest.raises(ValueError):
        cs.execute_notebook(nb, parameters={"A": value}, **_dirs(tmp_path))


def test_zero_parameters_cells_rejected(tmp_path):
    nb = _param_nb(tmp_path / "params.ipynb", tmp_path / "params.json", n_param_cells=0)
    with pytest.raises(ValueError):
        cs.execute_notebook(nb, parameters={"A": 2}, **_dirs(tmp_path))


def test_two_parameters_cells_rejected(tmp_path):
    nb = _param_nb(tmp_path / "params.ipynb", tmp_path / "params.json", n_param_cells=2)
    with pytest.raises(ValueError):
        cs.execute_notebook(nb, parameters={"A": 2}, **_dirs(tmp_path))


# ---------------------------------------------------------------------------
# Step-5 re-close ruling (agile-plan v5.189, R3a) -- the cert_id cross-check.
# Expected id: parameters["CERT_ID"] if given, else ^(C\d{2})_ of the file name,
# else no check. A mismatch appends a results_errors entry naming cert_id.
# ---------------------------------------------------------------------------
def _cert_id_param_nb(path: Path, emitted: str) -> Path:
    """One ``parameters`` cell (CERT_ID = 'C98'); always emits a valid PASS payload for
    ``emitted``, whatever CERT_ID is."""
    return _write_nb(path, [
        _code("CERT_ID = 'C98'", "section-3", "parameters"),
        _emit(_payload(ok=True, cert_id=emitted), f"{emitted}.json"),
    ])


@slow_kernel
def test_g1_file_name_id_mismatch_fails_the_gate(tmp_path):
    nb = _write_nb(tmp_path / "C93_x.ipynb", [_emit(_payload(ok=True, cert_id="C94"), "C94.json")])
    g = cs.execute_notebook(nb, **_dirs(tmp_path))
    assert g.executed is True, g.error
    assert g.results is not None and g.results["cert_id"] == "C94"
    assert g.passed is False
    assert any("cert_id" in e for e in g.results_errors), g.results_errors


@slow_kernel
def test_g2_parameter_id_mismatch_fails_the_gate(tmp_path):
    nb = _cert_id_param_nb(tmp_path / "probe.ipynb", "C98")
    g = cs.execute_notebook(nb, parameters={"CERT_ID": "C95"}, **_dirs(tmp_path))
    assert g.executed is True, g.error
    assert g.passed is False
    assert any("cert_id" in e for e in g.results_errors), g.results_errors


@slow_kernel
def test_g3_parameter_id_match_passes_the_gate(tmp_path):
    nb = _cert_id_param_nb(tmp_path / "probe.ipynb", "C98")
    g = cs.execute_notebook(nb, parameters={"CERT_ID": "C98"}, **_dirs(tmp_path))
    assert g.executed is True, g.error
    assert g.results_errors == []
    assert g.passed is True
