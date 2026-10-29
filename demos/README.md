# Benchmark demo

One script that measures how fast the simulator replays public OMIE data. It
runs on the CPU by default.

To learn how to download, load and replay real OMIE data step by step, see tutorial
[T03 — Real OMIE data](../tutorials/T03_real_data.ipynb).

## Before you start

Install the package as described in the
[installation section of the README](../README.md#installation). Then download
the OMIE files the benchmark reads (no OMIE data ships with this package). A
contract delivered on day $D$ needs the order **and** trade files of $D-1$ and $D$. The default date
is 2026-01-12, so January 2026 is enough. From the repository root:

```bash
python -m scripts.download_omie_data --months 202601 --data-type orders
python -m scripts.download_omie_data --months 202601 --data-type trades
```

The files go to `data/rawOMIE/`, which is where the benchmark looks. Run it from
the repository root. See [Data](../docs/data.md) for the file layout.

## `benchmark_single_contract.py`: replay speed

Replays every order of one contract through the matching engine (the compiled
`lax.scan` replay) and prints the load time, the cold (compiling) and warm
replay times, the number of fills and the traded volume. Writes no files.

```bash
python demos/benchmark_single_contract.py
python demos/benchmark_single_contract.py --contract-id 58 --delivery-date 2026-01-12
```

Options: `--contract-id` (default 81), `--delivery-date` (default 2026-01-12),
`--backend cpu|gpu` (default `cpu`). Any other date needs the files of that day
and the day before.
