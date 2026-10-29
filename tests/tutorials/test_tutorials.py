"""The tutorials under ``tutorials/``: each one executes headless, and each committed file is clean.

Executing a tutorial takes minutes, so those tests carry the marker ``tutorials`` and are
deselected unless the ``-m`` expression names it (see ``conftest.py``). Run them one at a time:

    pytest -m tutorials -n 0                 # all tutorials
    pytest -m tutorials -n 0 -k T01          # one tutorial

The other tests here only read files and stay in the default suite.

Every tutorial is a file ``tutorials/Tnn_<name>.ipynb`` (two digits). Executing one never
rewrites the committed file: the notebook is executed in memory and then discarded.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS_DIR = REPO_ROOT / "tutorials"
NOTEBOOKS = sorted(p for p in TUTORIALS_DIR.glob("T*_*.ipynb") if re.fullmatch(r"T\d\d_[^/]+\.ipynb", p.name))
IDS = [p.stem for p in NOTEBOOKS]

KERNEL_NAME = "python3"
CELL_TIMEOUT_S = 600  # per cell; a first-time data download can be slow


def _digest(path):
    return hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()


# ------------------------------------------------------------------ execution (marker: tutorials)
@pytest.mark.tutorials
@pytest.mark.parametrize("path", NOTEBOOKS, ids=IDS)
def test_tutorial_executes(path, monkeypatch):
    """The tutorial runs top to bottom without an error, with the repository root importable,
    and the committed file is left byte-identical."""
    import nbclient
    import nbformat

    before = _digest(path)
    # The kernel is a child process and inherits this environment: make the repository root
    # importable and keep the run on the CPU unless the caller chose otherwise.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(REPO_ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    monkeypatch.setenv("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=CELL_TIMEOUT_S, kernel_name=KERNEL_NAME,
                                     resources={"metadata": {"path": str(TUTORIALS_DIR)}})
    client.execute()  # raises CellExecutionError on the first failing cell

    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors, errors
    assert _digest(path) == before, f"{path.name} was rewritten by its own test"


# ------------------------------------------------------------------ the committed files (default suite)
def _cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _outputs(path):
    return [o for c in _cells(path) if c.get("cell_type") == "code" for o in c.get("outputs", [])]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=IDS)
def test_tutorial_is_committed_with_outputs_and_a_figure(path):
    """Tutorials ship executed (nbstripout must not apply to tutorials/), with at least one PNG figure."""
    outs = _outputs(path)
    assert outs, f"{path.name} has no outputs"
    assert any("image/png" in o.get("data", {}) for o in outs), f"{path.name} has no PNG figure"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=IDS)
def test_tutorial_has_no_execution_timestamps(path):
    stamped = [i for i, c in enumerate(_cells(path)) if "execution" in c.get("metadata", {})]
    assert not stamped, f"{path.name}: cells {stamped} carry execution timestamps"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=IDS)
def test_tutorial_carries_no_machine_strings(path):
    """No repository root, virtual environment, home folder, user name or host name in the file.
    The values are read from this machine at run time, never written here."""
    text = path.read_text(encoding="utf-8")
    for label, value in [("repository root", str(REPO_ROOT)), ("virtual environment", sys.prefix),
                         ("home folder", str(Path.home()))]:
        assert value not in text, f"{path.name} contains the {label}"
    for label, value in [("user name", getpass.getuser()), ("host name", socket.gethostname())]:
        if value:
            assert not re.search(rf"(?i)(?<![\w.-]){re.escape(value)}(?![\w-])", text), \
                f"{path.name} contains the {label}"


def test_no_root_gitattributes_ipynb_rule():
    """A root attribute file with an ``*.ipynb`` rule would strip the tutorials' outputs on commit."""
    ga = REPO_ROOT / ".gitattributes"
    if ga.exists():
        rules = [ln.split()[0] for ln in ga.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        assert not any(r.endswith(".ipynb") for r in rules), rules


def test_tutorials_marker_is_registered():
    """The marker line lives in pyproject.toml, so ``-m tutorials`` raises no unknown-marker warning."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = cfg["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.split(":")[0].strip() == "tutorials" for m in markers), markers


def _collect(*args):
    """Collect this file in a fresh pytest process (no xdist) and return its report."""
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-n", "0", "-p", "no:cacheprovider",
                        *args, str(Path(__file__).relative_to(REPO_ROOT))],
                       cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.slow
def test_plain_run_deselects_tutorials_and_marker_run_selects_them():
    """slow: two fresh pytest collection processes. Plain pytest collects but deselects the
    execution tests; ``-m tutorials`` selects them and only them."""
    code, out = _collect()
    assert code == 0, out
    assert "test_tutorial_executes" not in out and re.search(r"\b\d+ deselected\b", out), out
    assert "test_no_root_gitattributes_ipynb_rule" in out, out

    code, out = _collect("-m", "tutorials")
    assert code == 0, out
    assert "test_tutorial_executes" in out, out
    assert "test_no_root_gitattributes_ipynb_rule" not in out, out

    code, out = _collect("-m", "slow or tutorials")
    assert code == 0 and "test_tutorial_executes" in out, out
