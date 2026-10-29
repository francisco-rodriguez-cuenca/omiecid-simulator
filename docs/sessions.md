# Sessions

A `Session` replays an order tape for **one** quarter-hour contract on the market
clock, from gate opening (GOT) to gate closure (GCT), and lets one participant
submit its own orders between the background arrivals. The engine underneath is
the certified matching engine described in the [market model](market-model.md).

```python
from omiecid_simulator.omiecid.omie_session import (
    Session, AgentOrder, FillReport, TopOfBook, AGENT_TRADER_ID,
)

session = Session(config)                                  # config: dict, optional
state = session.reset(key)                                 # OMIEContractState
state, report, done = session.step(key, state, order)      # order: AgentOrder | None
state, report, done = session.step_env(key, state, order)  # compiled core, vmappable
tob = session.top_of_book(state)                           # TopOfBook
```

The [README quickstart](../README.md#quickstart) is a complete, runnable example.
Tutorial [T05](../tutorials/T05_session.ipynb) walks through a session step by
step, and the [tutorials](../tutorials/README.md) cover the rest of the package.

## One step

Each call to `step` does, in order:

1. **Background.** The next `orders_per_step` rows of the tape, read at
   `state.tape_cursor`, are replayed into the book. If this flow hits a resting
   order of the participant, the fill is attributed to the participant as a
   passive fill.
2. **Participant.** The `AgentOrder` (if active) is stamped and submitted. The
   session sets the order id, the participant id (`AGENT_TRADER_ID`), the contract
   and the time-priority timestamp. The timestamp is the book's arrival counter
   after the background replay, so the order queues behind every earlier arrival
   at its price. The user controls only the economic fields.
3. **Clock.** `state.time` advances to the latest timestamp of the slice, in
   minutes from GOT; `done = state.time >= gct_minutes_from_got`. How timestamps
   map to minutes depends on the clock (see [Clocks](#configuration)).

`reset`, `step_env` and `top_of_book` are compiled with `jax.jit`; `step` is a
thin Python wrapper that accepts `None` and casts the order's fields before
calling `step_env`. Real and empty orders share one compiled program. The first
call compiles; see [Performance](performance.md).

## `AgentOrder`

The participant's order: economic fields only. Build it with
`AgentOrder.create(...)`, which casts every field to its dtype, or use
`AgentOrder.none()` (or pass `None` to `step`) for "no order".

| Field | dtype | Default | Meaning |
|---|---|---|---|
| `active` | bool | `True` from `create`, `False` from `none` | `False` means no order this step. |
| `side` | int32 | required | `Side.BUY` (0) or `Side.SELL` (1). |
| `price` | float32 | required | Limit price, EUR/MWh. |
| `quantity` | float32 | required | Quantity, MW. |
| `exec_restriction` | int32 | `ExecRestriction.NON` (0) | `NON` (limit), `IOC` (1), `FOK` (2), `ICE` (3, iceberg). |
| `validity` | int32 | `Validity.GFS` (0) | `GFS` (until gate closure) or `GTD` (1). |
| `expiry_time` | int32 | 0 | Carried with the order; 0 means the contract's gate closure. There is no expiry sweep before gate closure (see [Modelling assumptions](modelling-assumptions.md#simplifications)). |
| `peak_qty` | float32 | 0.0 | Iceberg peak (visible slice), MW. |
| `ppd` | float32 | 0.0 | ICEwPI price increment per slice, EUR/MWh (buy negative, sell positive). |

The enums are in `omiecid_simulator.omiecid.order_types`. The order is validated
by the engine exactly like a tape order; a rejected order shows
`report.submit.accepted == False`.

## `FillReport`

A fixed-shape record of the step, safe under `jit` and `vmap`.

| Field | Meaning |
|---|---|
| `submitted` | The step carried an active order. |
| `order_id` | The stamped id of that order (0 when none). Use it to cancel the order later. |
| `submit` | The book's `SubmitResult` for the order: `accepted`, `traded`, `trade_qty` (MW), `trade_price` (EUR/MWh, quantity-weighted over walked levels), `settlement_value` (EUR, unsigned). All zero when no order. |
| `passive_qty_signed` | MW of the participant's resting orders filled during this step (by the background flow, or by the participant's own order in a [self-match](#self-trades)); positive on a buy. |
| `passive_value_eur` | Realized EUR of those passive fills; **positive on a sell**, negative on a buy. |
| `inventory_delta` | Change of the participant's net position, MW, positive on a buy (aggressive plus passive). |
| `cash_delta_eur` | Realized cash of the step, EUR, **unscaled**: $+V$ on a sell, $-V$ on a buy, with $V = p \cdot q \cdot d$ (aggressive plus passive). |
| `trade_rows` | `(2, 6)` int32: the rows written to the participant's trade tape this step (row 0 the aggressive leg, row 1 the passive leg; all zero when no fill). |
| `trades_saturated` | The book's trade buffer is full after the step: every one of its `nTrades` rows holds a trade, so later trades overwrite the last row. See [sizing `nTrades`](#sizing-ntrades). |

After `done`, further calls are ignored: they return the **same state and an
empty report** (every field zero), so the trade tape cannot be overwritten.
Tape rows after the slice that set `done` are therefore not replayed; under the
default clock `done` can come before the last real slice (see
[Clocks](#configuration)).

## The state

`OMIEContractState` is a JAX pytree (a `flax.struct` dataclass; change fields
with `state.replace(...)`):

| Field | Meaning |
|---|---|
| `book` | The contract's order book (`OMIEBookState`). |
| `time` | Minutes from GOT. |
| `step_counter` | Steps taken. |
| `agent_inventory` | `(1,)` float32: net position, MW, positive long. |
| `agent_cash` | `(1,)` float32: realized cash, EUR (aggressive plus passive, unscaled). |
| `agent_trades` | `(steps_per_episode, 2, 6)` int32: the participant's trade tape. |
| `tape_idx` | Which tape of the set is replayed. |
| `tape_cursor` | Next tape row to replay. |
| `contract_status` | Contract status code. |

## Settlement

The participant's settled result is **`state.agent_cash`**: the realized cash of
every fill, aggressive and passive, summed fill by fill. `state.agent_inventory`
is the net position, summed the same way. Both are exact, up to the float32
rounding of a running sum. The book's own trade tape (`state.book.lob.trades`)
is exact too.

`compute_agent_pnl` over the participant's trade tape, `state.agent_trades`, is
**exact only if no step had passive fills in both directions**:

```python
from omiecid_simulator.omiecid.settlement import compute_agent_pnl
pnl = compute_agent_pnl(state.agent_trades.reshape(-1, 6), AGENT_TRADER_ID)
```

The reason is that each step has one passive row (slot 1). That row combines all
of the step's passive fills into a single trade: the net direction, the total
quantity and the quantity-weighted price, rounded to a tick. When a step has a
passive buy and a passive sell, one row cannot represent both, so
`compute_agent_pnl(state.agent_trades)` differs from `agent_cash`. This is a
known limitation. Separate passive slots per direction are planned for v0.2
(see [Limitations](limitations.md#settlement-over-the-participants-trade-tape)).
Even in one direction, if a step's passive fills happen at several prices, the
row carries their quantity-weighted price rounded to a tick. Use `agent_cash`
for results.

Call `compute_agent_pnl` eagerly; it is not supported under `jax.jit` (see
[Limitations](limitations.md#numerics-and-jax)).

## `TopOfBook`

`session.top_of_book(state)` returns the anonymous best bid and ask (`p_bid`,
`p_ask`, EUR/MWh), their presence (`has_bid`, `has_ask`), the `mid` (0.0 unless
both sides are present) and the **visible** quantity at each touch (`q_bid`,
`q_ask`, MW). Hidden iceberg quantity is never shown. An empty side reads price
and quantity 0.0.

## Configuration

`Session(config)` reads these keys; every key is optional. **Unknown keys are
ignored**, so check spelling.

| Key | Default | Meaning |
|---|---|---|
| `tapes` | a small built-in synthetic set (two tapes) | `(n_tapes, N_padded, 13)` int32 array of background orders; see [Data](data.md). |
| `orders_per_step` | 4 | Tape rows replayed per step. The session lasts `ceil(N_padded / orders_per_step)` steps at most. |
| `product_config` | `OMIEProductConfig()` | Price and quantity ticks and bounds, delivery duration. |
| `market_config` | `OMIEMarketConfig()` | Array capacities: `nTrades` (100), `book_depth` (100 resting orders per side), `max_icebergs` (1024), `max_match_iterations` (200). |
| `session_clock` | `"span_scaled"` | How tape timestamps map to minutes from GOT; `"span_scaled"` or `"got_relative"`. Any other value raises `ValueError`. |
| `gct_minutes_from_got` | 60 | Gate closure, in minutes from GOT. |
| `got_seconds` | 55200 | GOT in the tape's timestamp frame (15:20 on the day before delivery, in seconds since that day's midnight). Used by `"got_relative"`. |
| `tape_seconds_span` | the largest timestamp over the whole tape set `tapes` | Used by `"span_scaled"` only; see [Clocks](#configuration) below. |

`product_config` and `market_config` are frozen dataclasses; build variants with
`OMIEMarketConfig(nTrades=1000)` or `dataclasses.replace(...)`.

**Clocks.**

- `"span_scaled"` (default, for synthetic tapes): the clock of slice $k$ is
  $t_k = \max\left(t_{k-1},\ \mathrm{round}(m_k \cdot \mathrm{gct}/S)\right)$,
  where $m_k$ is the largest real timestamp in the slice, $\mathrm{gct}$ is
  `gct_minutes_from_got`, $S$ is `tape_seconds_span` (by default the largest
  timestamp over the **whole tape set**, not over the tape being played), and
  $\mathrm{round}$ rounds to the nearest minute. The clock assumes the tape's
  timestamps are seconds from GOT spread over $[0, S]$, and maps that interval
  onto $[0, \mathrm{gct}]$ minutes. Since $\mathrm{done} = (t \ge \mathrm{gct})$,
  and $t$ is rounded to the nearest minute, `done` fires at the first slice with
  $m_k \ge S(\mathrm{gct} - \tfrac{1}{2})/\mathrm{gct}$, that is, at the first
  slice whose largest timestamp is within half a scaled minute of $S$,
  **possibly several slices before the last real slice**; raising
  `gct_minutes_from_got` does not prevent this. Tapes whose timestamps are
  offset from GOT or clustered near $S$ (for example absolute times of day) can
  reach `done` at the very first step. Because $S$ is shared by the whole set,
  a shorter tape in a stack may never reach `done`. Calls after `done` are
  ignored (see [above](#fillreport)), so the tape rows after the slice that set
  `done` are **not replayed**. For a full-tape replay, see
  [Limitations](limitations.md#early-gate-closure-under-the-span-scaled-clock).
- `"got_relative"` (for real OMIE tapes): $t = \max\left(0, \lfloor (ts - \text{got\_seconds}) / 60 \rfloor\right)$,
  and the first slice made only of padding (tape exhausted) sets
  $t = \text{gct}$. The constructor checks the tapes and raises `ValueError` if a
  row lies before GOT or at or after GCT, if the longest tape has fewer than
  `orders_per_step` trailing padding rows, or if a slice made only of padding
  comes before the last real row. For contract $k$ (0-indexed from 00:00 of the
  delivery day) GCT falls at the hour before its delivery hour, so
  `gct_minutes_from_got = 460 + 60 * (k // 4)`
  ([market specification §3.1](omiecid/market-spec.md#31-continuous-market-gate-times)).
  All tapes of one session must belong to the same delivery hour.

## Choosing a tape

`reset(key)` draws one tape of the set at random. To replay a given tape:

```python
import jax.numpy as jnp
state = session.reset(key).replace(tape_idx=jnp.int32(i))
```

## Cancelling an order

The compiled step does not cancel. Cancel between steps, eagerly, with the
book-level API and `report.order_id`:

```python
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.order_types import Side

contract = Contract(contract_id=0, delivery_start_minutes=120, delivery_end_minutes=135,
                    gct_minutes=60, status=ContractStatus.OPEN)
book = ContractBook(contract, session.product_config, session.market_config)

state, report, done = session.step(key, state, AgentOrder.create(side=Side.BUY, price=49.5, quantity=1.0))
# ... later, while the order still rests:
state = state.replace(book=book.cancel_order(state.book, int(report.order_id), Side.BUY))
```

Pass the order's side. Cancelling an id that no longer rests leaves the book
unchanged. Each submitted order gets a new id (`AGENT_ORDER_ID_BASE +
step_counter`), so keep the ids of the orders you may want to cancel.

## Self-trades

OMIE's matching has no self-trade prevention (provisional, medium-high
confidence: [OQ-12](omiecid/open-questions.md#oq-12--anti-self-match--self-trade-prevention-scope)),
so the participant's order can trade against its own resting order, and the
trade is printed in the book like any other. European market-abuse rules (REMIT)
still forbid wash trades; that is checked by surveillance after the fact, not by
the matching. The session attributes the passive
side of such a match as well as the aggressive side. After a self-match, the
participant's inventory and cash therefore net to zero **within the float32
rounding of their running sums** (for example about $10^{-7}$ MW), not
necessarily to an exact zero. `compute_agent_pnl` over the two matched legs is
exactly zero.

## Sizing `nTrades`

The book stores up to `market_config.nTrades` trades (default 100), counting the
whole session: background trades and the participant's. Beyond that, new trades
overwrite the last row of the buffer, and `report.trades_saturated` turns true.
Set `nTrades` at least to the expected number of trades in the session; for a
real OMIE contract, the number of rows in its trade file is a good guide, with a
margin, because the replay can produce more trades than the market did
([OQ-10](omiecid/open-questions.md#oq-10--cancelled-orders-not-identifiable-in-omie-public-data)).
Check `trades_saturated` at the end of the session. A larger buffer costs memory
and compile time. A structural fix is planned for v0.2 (see
[Limitations](limitations.md)).

## Batching

`step_env` can be batched with `jax.vmap`, inside an outer
`jax.experimental.enable_x64()` context:

```python
import jax
keys = jax.random.split(jax.random.PRNGKey(0), 8)
with jax.experimental.enable_x64():
    states = jax.vmap(session.reset)(keys)
    orders = jax.vmap(lambda _: AgentOrder.none())(keys)
    states, reports, dones = jax.vmap(session.step_env)(keys, states, orders)
```

The 64-bit context is needed for the reason given in [Limitations](limitations.md).

A batch gives the same results as running the sessions one at a time, but it
is not a speed-up on a CPU: under `jax.vmap` every branch of the engine runs for
every session, and the cost of a batch grows with its size. On a CPU, compile one
session with `jax.jit` of a `jax.lax.scan` over `step_env` and call it in a loop
over seeds and tapes. See [Performance](performance.md#running-faster) and
tutorial [T09](../tutorials/T09_scaling.ipynb).

## Several participants

A session has one participant. Several participants can trade on one contract
directly at book level, with `ContractBook.submit_order` and
`ContractBook.cancel_order`, each with its own `agent_id`. Tutorial
[T07](../tutorials/T07_several_participants.ipynb) shows it with three traders,
including the zero-sum check of their cash and positions.
