# Demos

Three scripts that run the simulator on public OMIE data. They run on the CPU by
default.

## Before you start

Install the package with the `examples` extra, and download the OMIE files the
demos read (no OMIE data ships with this package). A contract delivered on day
$D$ needs the order **and** trade files of $D-1$ and $D$. From the repository
root:

```bash
pip install ".[examples]"

# for the defaults of the benchmark and the real-data walkthrough (2026-01-12)
python -m scripts.download_omie_data --months 202601 --data-type orders
python -m scripts.download_omie_data --months 202601 --data-type trades

# for the default of the comparison (2025-06-02)
python -m scripts.download_omie_data --months 202506 --data-type orders
python -m scripts.download_omie_data --months 202506 --data-type trades
```

The files go to `data/rawOMIE/`, which is where the demos look. Run the demos
from the repository root. See [Data](../docs/data.md) for the file layout.

## `benchmark_single_contract.py`: replay speed

Replays every order of one contract through the matching engine (the compiled
`lax.scan` replay) and prints the load time, the cold (compiling) and warm
replay times, the number of fills and the traded volume. Writes no files.

```bash
python demos/benchmark_single_contract.py
python demos/benchmark_single_contract.py --contract-id 58 --delivery-date 2026-01-12
```

Options: `--contract-id` (default 81), `--delivery-date` (default 2026-01-12),
`--backend cpu|gpu` (default `cpu`).

## `compare_sim_vs_real.py`: simulated versus real trades

Replays one contract's real orders and prints the simulated trades beside the
trades OMIE actually printed: trade count, traded volume, volume-weighted price
and settlement value, with their differences, plus match metrics and a
diagnostic of how full the book got.

```bash
python demos/compare_sim_vs_real.py
python demos/compare_sim_vs_real.py --contract-id 58 --delivery-date 2025-06-10
```

Options: `--contract-id` (default 81), `--delivery-date` (default 2025-06-02),
`--data-path` (default `data/rawOMIE`), `--backend cpu|gpu` (default `cpu`).
The script stops with an error if the trade file of the date is missing.

The two are **not** expected to match. Public OMIE files contain no
cancellations, so orders withdrawn in reality stay in the replayed book
([OQ-10](../docs/omiecid/open-questions.md#oq-10--cancelled-orders-not-identifiable-in-omie-public-data));
cross-border trades have one side outside the Iberian files; and the replay does
not model the intraday-auction windows. The report measures the distance; it is
a diagnostic, not a test. See
[Modelling assumptions](../docs/modelling-assumptions.md#replaying-public-omie-data).

## `generate_real_data_demo.py`: an annotated walkthrough

Replays a short window of one contract's real orders (35 orders by default) and
writes an interactive HTML page that shows each order arriving, the book before
and after it, and every trade it produced, with its settlement value. The page
needs no server; open it in a browser.

```bash
python demos/generate_real_data_demo.py
python demos/generate_real_data_demo.py --contract-id 58 --slice-start 699   # FOK-dense window
python demos/generate_real_data_demo.py --contract-id 88 --slice-start 48    # ICEwPI-dense window
```

Options: `--delivery-date` (default 2026-01-12), `--contract-id` (default 81),
`--slice-start` (first order of the window, default 987), `--slice-k` (window
length, default 35). With the defaults the page is written to
`demos/20260112_omie_real/demo.html`; otherwise to
`demos/<YYYYMMDD>_omie_real_cid<k>_s<start>/demo.html` (for example
`demos/20260112_omie_real_cid58_s0699/demo.html`).

The two alternative windows above were chosen on 2026-01-12 as the windows with
the most fill-or-kill orders (contract 58) and the most price-incremented
icebergs (contract 88).
