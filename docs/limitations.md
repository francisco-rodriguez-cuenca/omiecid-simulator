# Limitations

This page lists the known limitations of the `Session` and of the release as a
whole. The limitations of the market model itself (what the engine does not
model, and the rulings it relies on) are in
[Modelling assumptions](modelling-assumptions.md); the certificate records the
same items from the certification side ([certificate](omiecid/certificate.md)).

## Session

1. **One participant per session.** A `Session` carries exactly one
   participant. Several participants are possible directly at book level, with
   `ContractBook.submit_order` and `ContractBook.cancel_order`, each order
   carrying its own `agent_id` (see [Sessions](sessions.md#several-participants)).

2. **No cancellation inside the compiled step.** `step_env` submits; it never
   cancels. Cancel between steps, eagerly, with `ContractBook.cancel_order` and
   the `order_id` from the step's `FillReport`
   (see [Sessions](sessions.md#cancelling-an-order)).

3. **Trade-buffer saturation.** The book stores up to `nTrades` trades (default
   100) over the whole session, background and participant together. Beyond
   that, new trades **overwrite the last row** of the buffer: the book's trade
   record is then incomplete, and the passive-fill attribution, which reads that
   record, can miss the participant's fills. Treat a session whose buffer
   saturated as unreliable. `FillReport.trades_saturated` flags the full
   buffer. Set `nTrades` at least to the expected number of trades in the
   session (see [Sessions](sessions.md#sizing-ntrades)). A
   structural fix (a write pointer that never overwrites) is planned for v0.2.

4. **The Session is verified, not certified separately.** The certification
   ladder certifies the matching engine. The `Session` is the research
   environment of the project with its reinforcement-learning layers removed; it
   is verified by a byte-identical parity check against that research
   environment, step by step, eagerly, under `jit` and under `vmap`; the only
   allowed difference is the attribution of the passive side of a self-match,
   which the comparator checks separately (see
   [Validation](validation.md#the-session)). It is not certified by its own
   ladder scenarios; C00, C13 and C16 run through it.

## Early gate closure under the span-scaled clock

The default `"span_scaled"` clock rounds to the nearest minute and assumes the
tape's timestamps are seconds from GOT spread over $[0, S]$, with $S$ =
`tape_seconds_span` (by default the largest timestamp over the whole tape set).
So `done` can fire at the first slice whose largest timestamp is within half
a scaled minute of $S$, possibly several slices before the last real one, and
at the very first step for tapes whose timestamps are offset from GOT or
clustered near $S$. Since $S$ is shared by the whole set, a shorter tape in a
stack may never reach `done`. Calls after `done` are ignored, so the tape rows
after the slice that set `done` are **not replayed** (see
[Sessions](sessions.md#configuration), *Clocks*).

**Full-tape replay recipe.** Set `tape_seconds_span` to at least twice the
largest real timestamp over the whole tape set (every batch lane), keep
`gct_minutes_from_got`, and run exactly `session.steps_per_episode` steps
($\lceil N_\text{padded}/\text{orders\_per\_step}\rceil$). Every slice clock
then stays at or below half of GCT, so `done` never fires and every row is
replayed. Only `state.time`, `done` and the time column of the participant's
trade rows differ. Raising `gct_minutes_from_got` does **not** help: a slice
that holds the largest timestamp always reaches GCT. For real OMIE tapes, use
the `"got_relative"` clock with the contract's gate times (see
[Sessions](sessions.md#configuration)).

## Settlement over the participant's trade tape

`state.agent_cash` and `state.agent_inventory` are exact: they sum every fill,
aggressive and passive, row by row (up to the float32 rounding of a running
sum). The book's trade tape is exact too. `compute_agent_pnl` over the
participant's own trade tape, `state.agent_trades`, is **exact only if no step
had passive fills in both directions**. Each step has a single passive row that
combines all of the step's passive fills: the net direction, the total quantity
and the quantity-weighted price. One row cannot represent a passive buy and a
passive sell in the same step, so on such steps the tape's settlement differs
from `agent_cash`. Even in one direction, the row rounds the quantity-weighted
price to a tick, so when a step's passive fills happen at several prices the
tape's settlement can differ slightly from `agent_cash`. Use `agent_cash` as the
settled result. This is a known
limitation; separate passive slots per direction are planned for v0.2. See
[Sessions](sessions.md#settlement).

## Numerics and JAX

- **`vmap` needs an outer `enable_x64` context.** The engine's fill extraction
  accumulates in int64 inside a scoped 64-bit context. Under `jax.vmap` that
  inner context cannot promote dtypes, and the lowering fails ("Cannot lower
  jaxpr with verifier errors ... expected to be promotable from 'i64'"). Wrap
  the batched call in `with jax.experimental.enable_x64():`
  (see [Sessions](sessions.md#batching)). Plain eager and `jit` calls need
  nothing.
- **`compute_agent_pnl` is not supported under `jax.jit`.** It accumulates in int64.
  It works eagerly and under `jax.vmap` inside an outer `enable_x64` context; it
  is not supported under `jax.jit`
  ([OQ-18](omiecid/open-questions.md#oq-18--phase-3-reopen-of-jaxjitcompute_agent_pnl-pending-jax-jax-mljax5982)).
  Inside the step, the session tracks cash in float32 (`agent_cash`).
- **Fixed capacities.** Resting orders per side (`book_depth`), icebergs
  (`max_icebergs`) and trades (`nTrades`) are fixed array sizes chosen at
  construction. Scenario C14 tests their behaviour at each limit; it is
  CERTIFIED in this release (see the [certificate](omiecid/certificate.md)).
  Larger sizes cost memory and compile time.

## Data

- **Replay is not reality.** Public OMIE files contain no cancellations, so an
  order that did not trade in reality rests until gate closure in the replay,
  and the simulated session can trade more than the real one
  ([OQ-10](omiecid/open-questions.md#oq-10--cancelled-orders-not-identifiable-in-omie-public-data)).
  The certification covers the engine, not the replay of public data. See
  [Modelling assumptions](modelling-assumptions.md#replaying-public-omie-data).
- **No raw data ships.** Real sessions need the public OMIE files, downloaded
  with the provided script (see [Data](data.md)).

## Scope

- One contract per session; no cross-contract or cross-border flow.
- No reinforcement-learning layer: actions, rewards and observations are left to
  the user.
