# Testing

The test suite is in `tests/omiecid/`. It covers the engine module by module,
the data loader, the synthetic tape generator, the `Session` contract, and the
certification harness. Most test files have an independently written audit twin
(`*_audit.py`) that checks the same contract from the specification, without
reading the implementation.

## Running the suite

Install the validation extra, which brings `pytest` and `pytest-xdist`, then run
from the repository root:

```bash
pip install ".[validation]"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest tests/omiecid/
```

The suite defaults to the CPU. `CUDA_VISIBLE_DEVICES=""` keeps JAX off any GPU;
tests marked `gpu` skip when no CUDA device is visible.

### Parallelism: `-n 24` by default

`pyproject.toml` sets `addopts = "-n 24 --max-worker-restart=3"`: pytest-xdist
starts 24 worker processes. Each worker compiles its own JAX programs, so the
default suits a machine with at least 24 cores and memory to match (memory use
grows with the number of workers). On a smaller machine, override it on the command line:

```bash
python -m pytest tests/omiecid/ -n auto     # one worker per CPU core
python -m pytest tests/omiecid/ -n 4        # four workers
python -m pytest tests/omiecid/ -n0         # no workers: one process, lowest memory
```

If workers are killed for lack of memory (pytest reports a worker crash or
restart), reduce `-n`.

### Lighter selections

Markers select parts of the suite; combine them with `and` / `not`:

| Marker | Selects |
|---|---|
| `slow` | Tests that compile a full program (a `Session` step, a full replay). `-m "not slow"` gives a quick check. |
| `jit` | Tests that exercise a `jax.jit` compile. |
| `audit` | The independent audit twins (`*_audit.py`). `-m "not audit"` skips them for routine checks; two core audit files always run. |
| `perf` | Wall-clock budget checks rather than correctness. Machine-dependent; `-m "not perf"` skips them. |
| `gpu` | Tests that need a CUDA GPU; skipped automatically without one. |

For example:

```bash
python -m pytest tests/omiecid/ -n 4 -m "not slow"           # quick check
python -m pytest tests/omiecid/ -n auto -m "not audit and not perf"
python -m pytest tests/omiecid/test_session.py -n0           # one file
```

### Compilation cache

The test configuration enables JAX's persistent compilation cache in
`.jax_cache/` at the repository root. The first run compiles everything; later
runs are much faster. Deleting `.jax_cache/` is always safe.

## Certification results

The certification audit tests (`tests/omiecid/certification/test_C<nn>_audit.py`
and some harness tests) read the results JSON written by the certification
notebooks. These files are not committed: they will be published as an asset of
the GitHub release (about 75 MB, with blake2b digests). Until they are present,
the tests that need them **skip** with the message "results JSON absent".

To run them, once the asset is published, download it, check the digests, and place
`C00.json` … `C16.json` in:

```text
notebooks/certification/results/
```

or set `OMIE_CERT_RESULTS_DIR` to the directory that holds them. You can also
produce your own results by running the notebooks (see
[Validation](validation.md#re-running-a-notebook)).

The merge-gate test `tests/omiecid/certification/test_certified_notebooks.py`
re-executes every certified notebook headless and does not need the asset; it is
the slowest part of the suite.
