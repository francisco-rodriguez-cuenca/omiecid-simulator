# Certification notebooks

One notebook per scenario of the certification ladder, `C00_<slug>.ipynb` …
`C16_<slug>.ipynb`, so the folder listing is the ladder in order. Each notebook
is self-contained: it quotes the rule it certifies, builds its own data with the
synthetic tape generator (`omiecid_simulator/jaxomiecid/omie_tape_synth.py`),
runs the engine eagerly and compiled with `jax.jit`, states the known result,
checks the invariant battery (`tests/omiecid/reference/omie_invariants.py`) and
emits its verdict.

| Notebook | Certifies |
|---|---|
| [`C00_inertness_and_status`](C00_inertness_and_status.ipynb) | A fresh state is reproducible, padding rows are inert, and orders outside the trading window are refused. |
| [`C01_order_validation`](C01_order_validation.ipynb) | Price and quantity bounds, tick grids and the iceberg peak rule, including the snap of an off-grid price to the nearest tick. |
| [`C02_single_crossing_pair`](C02_single_crossing_pair.ipynb) | Two crossing orders trade once, at the resting order's price, for the smaller quantity. |
| [`C03_priority_ties_and_ownership`](C03_priority_ties_and_ownership.ipynb) | Price-time priority, ties at one price, and matching that ignores who owns an order. |
| [`C04_cancellation`](C04_cancellation.ipynb) | Cancelling removes exactly that order, creates no trade and leaves the queue order of the other orders unchanged. |
| [`C05_partial_fill_residual`](C05_partial_fill_residual.ipynb) | The remainder of a partly filled order: its quantity, the price it rests at (its own limit) and its timestamp. |
| [`C06_multi_level_walk_vwap`](C06_multi_level_walk_vwap.ipynb) | An order that walks several price levels trades at each level and reports the volume-weighted price. |
| [`C07_ioc`](C07_ioc.ipynb) | An immediate-or-cancel order trades what it can and cancels the rest. |
| [`C08_iceberg_regeneration`](C08_iceberg_regeneration.ipynb) | Iceberg slices regenerate inside the same matching process, including when a new slice crosses the aggressor's resting remainder. |
| [`C09_icewpi_price_drift`](C09_icewpi_price_drift.ipynb) | Each new slice of a price-incremented iceberg moves its price by the increment, in the right direction; matching stops at the first slice that no longer crosses. |
| [`C10_fok_and_hidden_volume`](C10_fok_and_hidden_volume.ipynb) | The fill-or-kill pre-check counts visible, hidden and drifted iceberg volume correctly, and a killed order leaves the book byte-identical. |
| [`C11_gct_expiry`](C11_gct_expiry.ipynb) | At gate closure every resting order (GFS and GTD) is removed, without creating trades. |
| [`C12_settlement_cash_leg`](C12_settlement_cash_leg.ipynb) | Each trade settles at $V = p \cdot q \cdot d$, and cash sums to zero across participants. |
| [`C13_anonymity`](C13_anonymity.ipynb) | The book is shown anonymously, and a participant sees only its own trades. |
| [`C14_capacity_edges`](C14_capacity_edges.ipynb) | At each fixed capacity of the engine, the documented behaviour (flag raised, quantity conserved) is what happens. |
| [`C15_randomised_fuzz`](C15_randomised_fuzz.ipynb) | Random sessions across the parameter space keep every invariant, eagerly and compiled. |
| [`C16_multi_contract_isolation`](C16_multi_contract_isolation.ipynb) | Contracts never interact: each contract run alone gives the same trades as run alongside the others. |

- What each scenario means: [`scenario-ladder.md`](../../docs/omiecid/scenario-ladder.md).
- The evidence: [`certificate.md`](../../docs/omiecid/certificate.md) and
  [`certification-log.md`](../../docs/omiecid/certification-log.md).
- How the evidence fits together: [Validation](../../docs/validation.md).

## Layout

| Path | What it is |
|---|---|
| `cert_support.py` | The harness (imported as `notebooks.certification.cert_support`): quote checker, the `Certification` accumulator, results validator, the headless runner `execute_notebook`, and the readers of the certification log. |
| `results.schema.json` | JSON Schema (draft 2020-12) of `results/C<nn>.json`. |
| `_template.ipynb` | The nine-section template. Not a certification; the test suite executes it as a dry run. |
| `results/` | Results of a run, one `C<nn>.json` per scenario. Empty in the repository; see below. |
| `fixtures/C<nn>_seed<k>.npz` | Frozen counterexamples kept as regression data (keys `tape`, `params_json`). |
| `../../tests/omiecid/certification/test_C<nn>_audit.py` | The scenario's independent audit test. |

## The nine sections

Every notebook has the same shape, and every cell carries the tag of its section
(`section-<k>`).

| # | Heading | What it holds |
|---|---|---|
| 1 | `## 1. Purpose` | The scenario's purpose, verbatim from the ladder specification. |
| 2 | `## 2. The rule, quoted` | Quotes checked at run time, byte for byte, against the document each names (`market-spec.md`, `open-questions.md`, `scenario-ladder-review.md`). Quotes of the regulatory sources are shown as citations, not checked. |
| 3 | `## 3. Configuration` | One code cell tagged `parameters` (`CERT_ID`, `DRAWS`, `SEED0`, …), then derived settings. |
| 4 | `## 4. Dataset` | The tapes in readable units beside the raw integers, and a plot of the book. |
| 5 | `## 5. Known result` | The formula, beside the code that computes it from the drawn parameters. |
| 6 | `## 6. Run` | The engine run eagerly and compiled; the two records must be identical. |
| 7 | `## 7. Certification` | Checks against the known result, the invariant battery, capacity peaks and the discard rate. |
| 8 | `## 8. Verdict` | Writes `results/C<nn>.json` and prints `PASS` or `NOT CERTIFIED`. |
| 9 | (last cell) | On `NOT CERTIFIED`, explains why the certification failed. |

`tests/omiecid/certification/test_cert_structure.py` checks this structure on
the template and on every notebook.

## Running a notebook

Install the package and set up the notebook prerequisites as described in the
[installation section of the README](../../README.md#installation). Then, from
the repository root:

```bash
# interactively, results to a scratch folder (see the README, section 3)
mkdir -p /tmp/omiecid-results
OMIE_CERT_RESULTS_DIR=/tmp/omiecid-results PYTHONPATH="$PWD" jupyter lab notebooks/certification/C02_single_crossing_pair.ipynb

# headless, as the test suite runs it (results go to a fresh temporary directory)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -c "
from notebooks.certification.cert_support import execute_notebook
g = execute_notebook('notebooks/certification/C02_single_crossing_pair.ipynb')
print(g.passed, g.results['status'])"
```

`execute_notebook` runs an in-memory copy (the file is never written) through
`nbclient`, on CPU, with the environment variables `OMIE_CERT_RESULTS_DIR` and
`OMIE_CERT_FIXTURE_DIR` pointing at fresh temporary directories unless
`results_dir=` / `fixture_dir=` are given. A notebook passes only if it executes
cleanly **and** leaves exactly one valid results JSON whose status is `PASS`.
`tests/omiecid/certification/test_certified_notebooks.py` re-executes every
certified notebook this way. Some notebooks take several minutes on CPU.

Run interactively without `OMIE_CERT_RESULTS_DIR`, a notebook writes to
`results/` and replaces any file of the same name there, including downloaded
results; always set it to a scratch folder.

## Results of the release run

The results of the release run are not committed. They are published as the
pre-release
[`certification-results-2026-10`](https://github.com/francisco-rodriguez-cuenca/omiecid-simulator/releases/tag/certification-results-2026-10).
[Testing](../../docs/testing.md#certification-results) describes them, how to
use them with the audit tests, and how `OMIE_CERT_RESULTS_DIR` is read and
written.

## Output stripping

Notebooks are committed without outputs. `.gitattributes` in this folder names
an `nbstripout` filter for `*.ipynb` files in this folder only. If you plan to
commit notebook changes, define the filter once in your clone (with the
validation extra installed):

```bash
git config filter.nbstripout.clean "python -m nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
git config diff.ipynb.textconv "python -m nbstripout -t"
```

Do not run `nbstripout --install`: it applies the filter to the whole
repository.
