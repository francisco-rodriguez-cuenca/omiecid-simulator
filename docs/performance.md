# Performance

The simulator is written in JAX. Its cost has two parts: **compiling** a program
once per configuration, and **running** it. On a CPU, compiling dominates short
runs.

## Indicative numbers

Measured on CPU (AMD Ryzen Threadripper PRO 7975WX, 32 cores; JAX 0.6.2), for
one real OMIE contract (2025-06-25, slot 51: 3,153 orders), with
`orders_per_step=8`, `session_clock="got_relative"` and `nTrades=4096`, which
gives 395 steps:

| Run | First call (compiles) | Later calls |
|---|---|---|
| Python loop over `session.step` | 75 s | 1.8 s (about 220 steps/s) |
| One `jax.jit` of `reset` plus `jax.lax.scan` over `step_env` | 24 s | 0.57 s (about 690 steps/s) |
| Loading the contract from the raw files (`load_omie_orders`) | 2.3 s | 2.3 s |

These numbers vary with the machine, the JAX version and the configuration
(above all `nTrades`, `book_depth`, `max_icebergs` and `orders_per_step`). To
measure the bare engine on your machine, run
[the benchmark demo](../demos/README.md), which replays one real contract.

## What triggers a compile

`reset`, `step_env` and `top_of_book` are compiled with the `Session` as a
static argument. Two sessions built from equal configurations share one compiled
program; anything that changes the configuration compiles a new one:

- the tape set (its shape **and** its content);
- `orders_per_step`, `gct_minutes_from_got`, `session_clock`, `got_seconds`,
  `tape_seconds_span`;
- `product_config` and `market_config`.

Real and empty orders share one program, and so do all tapes of one set. To
replay many contracts, put them in **one tape set** and choose the tape with
`state.replace(tape_idx=i)` (see [Sessions](sessions.md#choosing-a-tape)) instead
of building one session per contract.

## Persistent compilation cache

JAX can store compiled programs on disk and reuse them across processes:

```python
import jax
jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
```

Set this before the first compile. The test suite and the benchmark demo
already do it. Deleting the cache directory is always safe.

## Running faster

- **Avoid the Python loop for long runs.** Calling `step` from Python costs a
  dispatch per step. Wrapping a whole session in `jax.lax.scan` over
  `step_env`, inside one `jax.jit`, runs it as one program (about three times
  faster in the measurement above, and it compiles faster too). The participant's
  decisions must then be written in JAX.
- **On a CPU, loop over a compiled session; do not batch with `jax.vmap`.**
  Many independent sessions (different tapes or different participant
  behaviour) can run as one batch under `jax.vmap`, inside an outer
  `jax.experimental.enable_x64()` context (see [Sessions](sessions.md#batching)
  and [Limitations](limitations.md)). The results are the same as running the
  sessions one at a time, but on a CPU the batch is much slower: under `vmap`,
  both branches of every `jax.lax.cond` in the engine run for every session
  (see the [JAX documentation of `lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html)),
  and the cost of a batch grows with its size. In tutorial
  [T09](../tutorials/T09_scaling.ipynb), on a CPU, a batch of four sessions took
  about 13 s, about 3.3 s per session, while the same sessions compiled without
  `vmap` took about 0.02 s each.
  On a CPU, compile one session (a `jax.lax.scan` over `step_env` inside
  `jax.jit`) and call it in a loop over seeds and tapes.
- **Size the arrays to the problem.** `nTrades`, `book_depth` and `max_icebergs`
  set fixed array sizes; larger values cost memory and compile time on every
  step. Keep `nTrades` large enough not to saturate, but no larger.
- **GPU.** Install the `cuda12` extra to run on an NVIDIA GPU. The engine's
  matching loop is sequential, so a single session is not necessarily faster on
  a GPU. Whether a `jax.vmap` batch is faster than a loop on a GPU has not been
  measured; tutorial [T09](../tutorials/T09_scaling.ipynb) has a cell that
  measures both on your GPU. Measure your own workload. The tests and the
  benchmark demo default to the CPU.

## Memory

A session's state holds the book (`book_depth` rows per side), the trade buffer
(`nTrades` rows), the iceberg table (`max_icebergs` rows) and the participant's
trade tape (two rows per step). The tape set is held on the device for the whole
session. Under `vmap`, each batch element carries its own state.
