# Testing

The test suite is in `tests/omiecid/`. It covers the engine module by module,
the data loader, the synthetic tape generator, the `Session` contract, and the
certification harness. Most test files have an independently written audit twin
(`*_audit.py`) that checks the same contract from the specification, without
reading the implementation.

## Running the suite

Install the package as described in the
[installation section of the README](../README.md#installation), which also
gives a quick check. To run the full suite, from the repository root:

```bash
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
notebooks. These files are not committed. The results of the release run are
published as the pre-release
[`certification-results-2026-10`](https://github.com/francisco-rodriguez-cuenca/omiecid-simulator/releases/tag/certification-results-2026-10):
`C00.json` … `C16.json` (80.8 MB in total) and their 256-bit BLAKE2b digests in
`DIGESTS.b2`. Until the files are present, the tests that need them **skip**
with the message "results JSON absent".

To run them, download the pre-release and check the digests, as the
[installation section of the README](../README.md#3-before-running-a-certification-notebook)
shows. Then either place `C00.json` … `C16.json` in:

```text
notebooks/certification/results/
```

or set `OMIE_CERT_RESULTS_DIR` to the folder that holds them.

`OMIE_CERT_RESULTS_DIR` is read and written:

- the audit tests **read** the results from the folder it names if it is set,
  otherwise from `notebooks/certification/results/`;
- a notebook run **writes** its `Cnn.json` to the same folder, and replaces the
  file of the same name.

So never point it at the downloaded files when you run a notebook; use a
scratch folder. You can also produce your own results by running the notebooks
(see [Running a notebook](../notebooks/certification/README.md#running-a-notebook)).

The test `tests/omiecid/certification/test_certified_notebooks.py` re-executes
every certified notebook headless and does not need the published results; it
is the slowest part of the suite.
