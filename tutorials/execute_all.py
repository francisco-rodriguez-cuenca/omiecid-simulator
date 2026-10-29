#!/usr/bin/env python3
"""Re-run the tutorials in place and sanitize their outputs.

    python tutorials/execute_all.py              # every tutorials/Tnn_*.ipynb
    python tutorials/execute_all.py T01 T03      # the ones named (by Tnn prefix or file name)

Each tutorial is executed headless from the ``tutorials/`` folder with the kernel ``python3``,
the repository root on ``PYTHONPATH`` and JAX on the CPU unless ``JAX_PLATFORMS`` says otherwise.
Then its outputs are sanitized before the file is written back:

- per-cell execution timestamps are removed;
- in every output, the virtual-environment path becomes ``<venv>`` and the repository root ``<repo>``;
- progress-bar widget state is cleared;
- the file is checked for the home folder, the user name and the host name of this machine. These
  values are read here at run time, never written in this file. If one remains, the tutorial is
  NOT written back and the script fails.

A tutorial whose execution raises is not written back either. Exit status: 0 if every tutorial
ran and was written, 1 otherwise.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KERNEL_NAME = "python3"
CELL_TIMEOUT_S = 600  # per cell; a first-time data download can be slow
WIDGET_MIME = "application/vnd.jupyter.widget-view+json"
KERNELSPEC = {"name": KERNEL_NAME, "display_name": "Python 3 (ipykernel)", "language": "python"}


def tutorials(names):
    """The tutorial files to run: all ``Tnn_*.ipynb``, or those matching ``names`` (``T01``, ``T01_x.ipynb``)."""
    every = sorted(p for p in HERE.glob("T*_*.ipynb") if re.fullmatch(r"T\d\d_[^/]+\.ipynb", p.name))
    if not names:
        return every
    chosen = []
    for n in names:
        stem = Path(n).name.removesuffix(".ipynb")
        hits = [p for p in every if p.stem == stem or p.stem.split("_")[0] == stem]
        if not hits:
            sys.exit(f"execute_all: no tutorial matches {n!r}")
        chosen += [p for p in hits if p not in chosen]
    return chosen


def _paths(path):
    """A path as written and resolved, longest first, with no duplicates."""
    return sorted({str(path), str(Path(path).resolve())}, key=len, reverse=True)


def _replace(value, pairs):
    if isinstance(value, str):
        for old, new in pairs:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace(v, pairs) for v in value]
    if isinstance(value, dict):
        return {k: _replace(v, pairs) for k, v in value.items()}
    return value


def sanitize(nb):
    """Sanitize an executed notebook in place (timestamps, machine paths, widget state)."""
    # The virtual environment usually sits inside the repository: replace it first.
    pairs = [(p, "<venv>") for p in _paths(sys.prefix)] + [(p, "<repo>") for p in _paths(ROOT)]
    nb.metadata.pop("widgets", None)
    nb.metadata["kernelspec"] = dict(KERNELSPEC)
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
        if cell.cell_type != "code":
            continue
        outputs = []
        for o in cell.get("outputs", []):
            data = o.get("data")
            if data is not None and WIDGET_MIME in data:
                data.pop(WIDGET_MIME)
                if not data:
                    continue  # a bare progress-bar widget: nothing left to show
            for key in ("text", "data", "traceback", "evalue"):
                if key in o:
                    o[key] = _replace(o[key], pairs)
            outputs.append(o)
        cell.outputs = outputs
    return nb


def machine_strings(text):
    """Labels of the machine-specific values of this computer that ``text`` still contains."""
    found = [label for label, v in [("home folder", str(Path.home()))] if v and v in text]
    for label, v in [("user name", getpass.getuser()), ("host name", socket.gethostname())]:
        if v and re.search(rf"(?i)(?<![\w.-]){re.escape(v)}(?![\w-])", text):
            found.append(label)
    return found


def run(path):
    """Execute, sanitize and write back one tutorial. Return an error message or None."""
    import nbclient
    import nbformat

    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=CELL_TIMEOUT_S, kernel_name=KERNEL_NAME, record_timing=False,
                                     resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    except Exception as exc:  # CellExecutionError, timeout, dead kernel
        return f"execution failed: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    sanitize(nb)
    text = nbformat.writes(nb) + "\n"
    if (left := machine_strings(text)):
        return f"machine-specific strings remain after sanitizing: {', '.join(left)}"
    path.write_text(text, encoding="utf-8")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="tutorials to run (Tnn or file name); default: all")
    args = ap.parse_args(argv)

    # The kernel is a child process and inherits this environment.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.pop("MPLBACKEND", None)  # let the kernel use its inline backend, so figures are kept as PNG

    paths = tutorials(args.names)
    if not paths:
        print("execute_all: no tutorials found")
        return 0
    failed = 0
    for path in paths:
        t0 = time.time()
        err = run(path)
        status = "FAIL" if err else "ok"
        print(f"{path.name}: {status} ({time.time() - t0:.0f} s)" + (f" - {err}" if err else ""))
        failed += bool(err)
    print(f"execute_all: {len(paths) - failed}/{len(paths)} written")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
