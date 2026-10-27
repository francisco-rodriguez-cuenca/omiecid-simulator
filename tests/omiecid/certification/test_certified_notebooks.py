"""OMIE-CERT-HARNESS-1 -- the merge gate re-executes certified notebooks
(decision H13; AC (6)).

Picked up by the existing per-file merge gate. It executes, through the nbclient
gate, every notebook whose **latest** log entry is ``CERTIFIED`` (including
``CERTIFIED -- provisional on ...``) and whose Verdict is not a human verdict, with
its committed parameters (no draw override: the certified run is what is
re-run). A ``NOT CERTIFIED`` notebook is merged as evidence and is **not** gated.

Until C00 certifies the parametrised test collects zero cases, which pytest
reports as one skip; that is expected. The template is not in the log and is not
gated here; ``test_cert_dry_runs.py`` exercises it on every gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_LOG = REPO_ROOT / "docs" / "omiecid" / "certification-log.md"
NB_DIR = REPO_ROOT / "notebooks" / "certification"


def _collect():
    """Gated notebooks at collection time; a selection error becomes a failing case, never a silent zero."""
    try:
        from notebooks.certification.cert_support import gated_notebooks
        paths = gated_notebooks(REAL_LOG, NB_DIR)
    except Exception as exc:  # noqa: BLE001
        return [pytest.param(None, repr(exc), id="gate-selection-error")]
    return [pytest.param(p, None, id=p.stem) for p in paths]


def test_real_log_parses():
    from notebooks.certification.cert_support import gated_notebooks, parse_log
    parse_log(REAL_LOG)
    gated_notebooks(REAL_LOG, NB_DIR)


def test_template_is_not_gated():
    from notebooks.certification.cert_support import gated_notebooks
    assert all(not p.name.startswith("_") for p in gated_notebooks(REAL_LOG, NB_DIR))


@pytest.mark.slow  # slow: re-executes a certified notebook at its committed draw count
@pytest.mark.parametrize("notebook,selection_error", _collect())
def test_certified_notebook_still_passes(notebook, selection_error):
    assert selection_error is None, f"gate selection failed: {selection_error}"
    from notebooks.certification import cert_support as cs
    # OMIE-CERT-HARNESS-2 (J7): per-scenario per-cell timeout (default 1,800 s;
    # C16 runs ~100 min, so it gets its own entry in GATE_CELL_TIMEOUT_S).
    timeout = int(cs.GATE_CELL_TIMEOUT_S.get(notebook.name.split("_")[0], 1800))
    g = cs.execute_notebook(notebook, timeout=timeout)
    assert g.passed, (
        f"{notebook.name}: executed={g.executed} status="
        f"{(g.results or {}).get('status')!r} results_errors={g.results_errors!r}\n{g.error or ''}")
