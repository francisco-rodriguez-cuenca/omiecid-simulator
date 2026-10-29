# Tutorials

Executed notebooks that teach the OMIE continuous intraday (CID) market and the
`omiecid_simulator` package, one question per tutorial. Each tutorial explains every market
concept the first time it is used, in a short "Concept" box, so you can start where you need to.
Every notebook is committed with its outputs: you can read it on the web without running it.

## Index

| Tutorial | Question it answers | Run time (CPU) | Network |
|---|---|---|---|
| [T01 — Your first contract](T01_first_contract.ipynb) | What happens when two orders meet? | under 1 min | no |
| [T02 — Order types and matching](T02_order_types.ipynb) | How do the order types of OMIE behave? | under 1 min | no |
| [T03 — Real OMIE data](T03_real_data.ipynb) | How do I replay a real trading day? | about 2 min (plus a first-time download of 107.8 MB, about 10 s on a fast line) | yes, first run only; skipped when offline or when the four files exist |
| [T04 — Synthetic order flow](T04_synthetic_flow.ipynb) | How do I build my own order flow, without real data? | about 2 min | no |
| [T05 — Trading in a session](T05_session.ipynb) | How does my own order enter a market that keeps moving? | about 4 min (three session configurations, about 70 s of compilation each) | no |
| [T06 — Writing an agent](T06_writing_an_agent.ipynb) | How do I plug my own strategy into the market? | under 2 min (one compilation) | no |
| [T07 — Several participants at book level](T07_several_participants.ipynb) | How do I simulate several traders against each other? | under 1 min | no |
| [T08 — The overseer's view](T08_overseers_view.ipynb) | What does the market operator see that participants do not? | about 2 min | no (an optional cell reuses T03's files if present; it never downloads) |
| [T09 — Running many sessions fast](T09_scaling.ipynb) | How do I run thousands of sessions fast? | about 3 min the first time; about 30 s when the compiled programs are in `.jax_cache/tutorials/` | no |

The run times were measured on one machine. Most of each is compiling, and they vary with the machine.

## Learning paths

Dependencies between tutorials are light: each notebook reads on its own, and earlier tutorials
are linked rather than required.

| Reader | Knows | New to | Suggested path |
|---|---|---|---|
| **Energy-market researcher** | Electricity markets, OMIE, intraday trading | JAX, order-book mechanics, integer tick encoding | [T01](T01_first_contract.ipynb) → [T02](T02_order_types.ipynb) → [T05](T05_session.ipynb) → [T06](T06_writing_an_agent.ipynb) → [T03](T03_real_data.ipynb) → [T08](T08_overseers_view.ipynb) → [T09](T09_scaling.ipynb) |
| **ML / quant developer** | Python, JAX, limit-order books | OMIE CID: contracts, delivery periods, gate closure, iceberg orders with price increment (ICEwPI) | [T01](T01_first_contract.ipynb) → [T02](T02_order_types.ipynb) → [T04](T04_synthetic_flow.ipynb) → [T05](T05_session.ipynb) → [T06](T06_writing_an_agent.ipynb) → [T09](T09_scaling.ipynb) → [T07](T07_several_participants.ipynb) |
| **SoftwareX reader** | The article's claims | Seeing the package work | [T01](T01_first_contract.ipynb) → [T03](T03_real_data.ipynb) → [T05](T05_session.ipynb) → [T06](T06_writing_an_agent.ipynb) |

```
T01 first contract ──> T02 order types ──> T07 several participants ──> T08 overseer's view
   │
   ├──> T03 real data ─────────────────────────────────────────────> (T08 real-day cell)
   └──> T04 synthetic flow ──> T05 Session ──> T06 writing an agent ──> T09 scaling
```

## Running the tutorials

Install the package as described in the
[installation section of the main README](../README.md#installation). Then, from the
repository root:

```bash
jupyter lab tutorials/
```

Each notebook's setup cell finds the repository root by itself and runs on the CPU by default.

## Glossary

Every term the tutorials introduce, in alphabetical order, with the tutorial that introduces it.

- **64-bit mode** — `jax.experimental.enable_x64()`; a batch of sessions must be built and called inside it (T09).
- **Agent** — a function that looks at the market (top of the book, its own state) and returns an order, or no order (T06).
- **Aggressor / passive order** — the newly arrived order that trades at once / the resting order it trades against (T01).
- **Anonymous book** — the public view of the book: in this simulator, the price levels and their visible quantities, without who sent them (T01).
- **Best bid, best ask, spread** — the highest bid, the lowest ask, and the gap between them (T01).
- **Bid / ask** — a resting buy order / a resting sell order (T01).
- **Bidding unit** — the unit in whose name a company sends orders to OMIE and is settled; one company can own several. OMIE's files name the bidding unit of every order, and the loader gives each one a number (T07, T08).
- **Cash (`agent_cash`)** — what a trader received for its sales minus what it paid for its purchases, in EUR; in a session, the settled result (T06, T07).
- **Compiling** — JAX turns a Python function into a program for the device on its first call with given input shapes; later calls with the same shapes reuse it (T09).
- **Continuous intraday market (CID)** — the OMIE market where electricity for the same or the next day is traded continuously, order by order, apart from short pauses during the intraday auctions (T01).
- **Contract** — one product of the CID: electricity delivered during one 15-minute period; a delivery day has 96 (92 or 100 on the two daylight-saving days) (T01).
- **Crossing** — an order crosses the book when its limit reaches the best price on the other side, so it trades at once (T01).
- **`cuda12` extra** — the install option that adds NVIDIA GPU support (T09).
- **Delivery day $D$ and slot $k$** — a contract is the quarter-hour $k$ (0–95, $k = 0$ is 00:00–00:15) of delivery day $D$ (T03).
- **`done`** — the flag that the session clock has reached gate closure; later calls are ignored (T05).
- **Execution restriction** — the rule that decides what happens to the part of an order that cannot trade at once (T02).
- **Fill-or-kill (FOK)** — trades only if it can be filled completely at once; otherwise nothing trades (T02).
- **`FillReport`** — what happened to the participant during one step: its new order's result, its resting orders that traded, and the change of position and cash (T05).
- **Gate closure time (GCT)** — the time a contract stops trading, one hour before the start of its delivery hour; unmatched orders are then cancelled (T01).
- **Gate opening time (GOT)** — the time when all contracts of a delivery day open for trading, about 15:20 on the day before; the session clock counts minutes from it (T01, T05).
- **`got_relative` clock** — the clock for real OMIE tapes: minutes since gate opening, read from clock-time timestamps; the session ends when the tape runs out (T05).
- **Hidden remainder** — the part of an iceberg not shown in the book; only the operator and the owner see it (T08).
- **Iceberg order (ICE)** — an order that shows only a part of its quantity, the peak, and hides the rest (T02).
- **Iceberg with price increment (ICEwPI)** — an iceberg whose price moves by a fixed increment at each new peak, $p_k = b + k\,d$ (T02).
- **Immediate-or-cancel (IOC)** — trades what it can at once; the rest is cancelled (T02).
- **Integer tick encoding** — inside the engine, every price and quantity is stored as a whole number of ticks (T01).
- **`jax.jit`** — compiles a function into one program (T09).
- **`jax.lax.scan`** — repeats a step function a fixed number of times, carrying the state; with `jax.jit`, a whole session becomes one program (T09).
- **`jax.vmap`** — turns a function of one session into a function of a batch of sessions, each input with a leading batch axis (T09).
- **`JAX_PLATFORMS`** — the environment variable that chooses the device (`cpu` or `cuda`), set before JAX starts (T09).
- **Limit order (NON)** — an order with no restriction: its unfilled part rests in the book (T02).
- **Market operator** — the operator of the order book, which sees all of it: every order's owner, every iceberg's hidden part, both sides of every trade; in the real market the book is the European shared order book, OMIE knows the identities of its own (Iberian) participants' orders, and the simulator models one Iberian book (T08).
- **Modification** — changing an order's price, quantity or execution restriction; it gets a new timestamp and loses its time priority (T02).
- **MW, EUR/MWh** — quantities are power in megawatts; prices are in euros per megawatt-hour (T01).
- **OMIE** — *Operador del Mercado Ibérico de Energía*, which runs the day-ahead and intraday electricity markets of Spain and Portugal (T01).
- **Operator view / participant view** — the full internal state of the book / what one participant sees: the anonymous price levels, its own orders and its own trades, and, in OMIE, the anonymous price and volume of the area's trades (not offered by this version) (T08).
- **Order book** — the list of orders of one contract that wait to trade (T01).
- **Order file / trade file** — OMIE's daily public files `orders_YYYYMMDD.1` (one row per order submission) and `trades_YYYYMMDD.1` (one row per trade) (T03).
- **Padding row** — a tape row with order id and time $\le 0$; the engine skips it, and it gives tapes of different lengths one shape (T04).
- **Participant number (`agent_id`)** — the number that identifies the sender of an order; at book level the caller chooses it, and a `Session` uses one fixed number for its participant (T07).
- **Participation** — your own orders, added on top of the replayed tape; the tape does not react to them (T05).
- **Passive fill** — a trade against one of your resting orders, caused by a later order that crossed it (T05).
- **Peak** — the visible part of an iceberg; a used-up peak is replaced at once by a new one, which joins the back of the queue (T02).
- **Persistent compilation cache** — a folder where JAX stores compiled programs, so a later run loads them instead of compiling (`.jax_cache/`) (T09).
- **Position (inventory)** — what a trader bought minus what it sold, in MW: positive (long) after buying more, negative (short) after selling more (T06, T07).
- **Price increment (PPD)** — the step $d$ of an ICEwPI: positive for a sell, negative for a buy; not the delivery duration $d = 0.25$ h of the trade value (T02).
- **Price-time priority** — the best price trades first; at equal prices, the earlier order trades first (T01).
- **Regime (`FuzzRegime`)** — the recipe of a random tape: order types, share of crossing orders, share of same-second ties, number of orders (T04).
- **Rejected order** — an order that fails a validation rule; it changes nothing in the book (T02).
- **REMIT, wash trade** — the European rules against market abuse in wholesale energy markets (REMIT) forbid wash trades: trades that change no beneficial ownership. They are checked by surveillance after the fact, not by the matching (T07).
- **Replay** — sending a tape's orders, in time order, to an empty order book of the contract (T03).
- **Replay rule (rest until gate closure)** — a limit or iceberg order that does not trade stays in the replay's book until gate closure, because the public files record no cancellation (T03).
- **Resting order** — an order that waits in the book because nothing on the other side crosses it (T01).
- **Self-trade** — a trade whose buyer and seller are the same participant; OMIE's matching has no self-trade prevention ([OQ-12](../docs/omiecid/open-questions.md#oq-12--anti-self-match--self-trade-prevention-scope), provisional), so it can happen, and it moves neither money nor energy for that participant; REMIT forbids wash trades (T07).
- **Session** — a simulation of one contract from gate opening to gate closure: it replays a tape into the order book and lets one participant add its own orders between the slices (T05).
- **Session day** — the window covered by one OMIE file: session day $D$ runs from 22:20 on $D-1$ to 22:00 on $D$; a contract of day $D$ spans the files of $D-1$ and $D$ (T03).
- **SIDC** — Single Intraday Coupling, the European continuous intraday market with one shared order book; OMIE passes its own participants' orders to it, anonymised, and the simulator models one Iberian book (T08).
- **`span_scaled` clock** — the default clock for synthetic tapes: it stretches timestamps in $[0, S]$ seconds onto $[0, \mathrm{gct}]$ minutes, rounded to whole minutes (T05).
- **Step** — one call to `Session.step`: replay the next `orders_per_step` tape orders, then stamp and submit your order behind every earlier arrival, then advance the clock (T05).
- **Taker and maker** — a taker's order trades on arrival at the resting order's price; a maker's order rests and waits to be traded against (T06).
- **Tape** — the time-ordered array of order rows (13 integer columns) that drives a replay (T03).
- **Tick** — the smallest step of a price (0.01 EUR/MWh) or a quantity (0.1 MW) (T01).
- **Time-priority stamp** — the timestamp that ranks orders at the same price, the earliest first; the simulator stores it as a pair (second, sequence number within the second) (T08).
- **Top of the book (touch)** — the best bid and the best ask, with their visible quantities (T05).
- **Traceable agent** — an agent that decides with `jnp` operations only (no Python `if` on market values), so it can run inside a compiled program; "trade or not" is the order's `active` flag (T06, T09).
- **Trade buffer (`nTrades`)** — the fixed number of trade rows a book stores; when full, new trades overwrite the last row and the record is incomplete (T05).
- **Trade price** — always the passive order's price (T01).
- **Trade record** — the book's own row for one trade: price and quantity in ticks, the passive and the aggressor order ids and participants, and the time; its quantity is negative when the aggressor bought (T07).
- **Trade value (settlement value)** — the money of a trade, $V = p \cdot q \cdot d$ EUR, for $q$ MW at $p$ EUR/MWh over a delivery period of $d = 0.25$ h; the seller receives $V$ and the buyer pays it (T01, T06).
- **Validity: good-for-session (GFS) / good-till-date (GTD)** — valid until gate closure / until a chosen time no later than gate closure; in this simulator both rest until they trade, are cancelled, or the contract closes (T02).
- **Volume-weighted average price (VWAP)** — the average trade price with each trade counted in proportion to its quantity, $\bar p = \sum_i p_i q_i / \sum_i q_i$ (T03).
- **Zero-sum check** — summed over every trader of the book, cash adds up to exactly zero and positions add up to exactly zero, because every trade has a buyer and a seller and both are counted; the cash is the value of the trades only, without fees (T07).
