# Market model

OmieCID-Simulator reproduces order matching in the Iberian continuous intraday
market (CID): the MIBEL (Iberian Electricity Market) segment of the European Single Intraday Coupling (SIDC,
formerly XBID), operated by OMIE as the Iberian NEMO (Nominated Electricity Market Operator). This guide summarises what
the simulator models. Each point links to the section of the
[market specification](omiecid/market-spec.md) it comes from. That specification
cites the regulatory sources (OMIE market rules, the SIDC matching algorithm
description, OMIE Instruction 1/2024; see [the corpus](omiecid/corpus/README.md)).

Where the simulator interprets or simplifies the regulation, the ruling is in
[Modelling assumptions](modelling-assumptions.md). The identifiers used below
(OQ-nn, Cnn) are explained in the [evidence directory guide](omiecid/README.md).
How the model is certified is in [Validation](validation.md); how a participant
trades in it is in [Sessions](sessions.md); what the simulator does not do is in
[Limitations](limitations.md).

## Contracts

- **Product.** The active CID product for Spain and Portugal is the quarter-hour:
  96 contracts per delivery day, one per 15-minute delivery period
  ([§2.1](omiecid/market-spec.md#21-products-supported-in-sidc),
  [§2.2](omiecid/market-spec.md#22-contract-generation)). Daylight-saving days
  have 92 or 100 contracts ([OQ-30](omiecid/open-questions.md#oq-30--daylight-saving-transitions-contract-strings-slot-count-and-wall-clock-timestamps)).
- **One book per contract.** Each contract has its own order book. Orders on
  different contracts never match
  ([§6.4](omiecid/market-spec.md#64-no-cross-contract-matching)).
- **Contract identity.** A contract is the pair (delivery date $D$, slot $k$),
  with $k = 4h + m/15$ for a delivery start at $h{:}m$. The same slot on two
  delivery dates is two contracts
  ([§13.4](omiecid/market-spec.md#134-notes-for-data-loading-m7);
  [OQ-27](omiecid/open-questions.md#oq-27--what-is-a-tape-product-identity-delivery-date--slot-session-day-file-partition-and-which-files-a-product-spans)).
- **In the simulator.** A `Session` replays the order flow of one contract. At
  book level, `ContractBook` holds one contract's book; scenario C16 certifies
  that books of different contracts stay isolated.

## Trading window and gate closure

- **Gate opening (GOT).** All 96 contracts of delivery day $D$ open together at
  about 15:20 on $D-1$, after the first intraday auction
  ([§3.1](omiecid/market-spec.md#31-continuous-market-gate-times)).
- **Gate closure (GCT).** Closure is aligned to the delivery hour. The four
  quarter-hours of one hour close together, one hour before that hour starts:
  $\mathrm{GCT}(k) = 60\lfloor k/4 \rfloor - 60$ minutes from $D$ 00:00
  ([§3.1](omiecid/market-spec.md#31-continuous-market-gate-times),
  [§11.1](omiecid/market-spec.md#111-hard-rules-from-authoritative-documents)).
  The trading window of contract $k$ lasts $460 + 60\lfloor k/4 \rfloor$ minutes.
- **At gate closure** unmatched orders are cancelled. Matched trades stay firm,
  and the market does not close out open positions
  ([§8.5](omiecid/market-spec.md#85-closing-out-open-positions)).
- **In the simulator.** The session clock counts minutes from gate opening, and
  `done` becomes true when it reaches gate closure (`gct_minutes_from_got` in the
  session configuration). Calls made after `done` return the same state and an
  empty report. Expiry at the gate is certified by scenario C11. The intraday
  auction windows that interrupt continuous trading are not modelled
  ([§3.3](omiecid/market-spec.md#33-continuous-market-rounds-and-ida-suspension-model);
  see [Modelling assumptions](modelling-assumptions.md)).

## Order types

| Restriction | Partial fill | Unfilled part | Spec |
|---|---|---|---|
| NON (limit) | Yes | Rests in the book | [§4.1](omiecid/market-spec.md#41-regular-limit-orders) |
| IOC (immediate or cancel) | Yes | Cancelled | [§4.5](omiecid/market-spec.md#45-execution-restrictions-summary) |
| FOK (fill or kill) | No | Whole order cancelled unless fully fillable | [§4.5](omiecid/market-spec.md#45-execution-restrictions-summary) |
| ICE (iceberg) | Yes, slice by slice | Rests; only the peak is shown | [§4.2](omiecid/market-spec.md#42-iceberg-orders) |
| ICEwPI (iceberg with price increment) | Yes, slice by slice | Rests; each new slice moves the price by the increment | [§4.2](omiecid/market-spec.md#42-iceberg-orders) |

- **Validity.** Good-for-session (GFS, the default: valid until gate closure) or
  good-till-date (GTD) ([§4.6](omiecid/market-spec.md#46-validity-conditions)).
- **Not modelled:** user-defined block orders, basket orders and all-or-nothing
  (AON) ([§4.3](omiecid/market-spec.md#43-user-defined-block-orders),
  [§4.4](omiecid/market-spec.md#44-basket-orders)).
- **In the simulator.** A participant's order carries the economic fields only:
  side, price, quantity, execution restriction, validity, and for icebergs the
  peak quantity and the price increment (PPD).

### Icebergs

- **Slices.** An iceberg shows a peak of size $P$ and hides the rest. When the
  shown quantity is fully traded, a new slice is activated at once, inside the
  same matching process, with a new timestamp
  ([§4.2](omiecid/market-spec.md#42-iceberg-orders),
  [§5.2](omiecid/market-spec.md#52-timestamp-rules),
  [§4.2.1 Rule 4](omiecid/market-spec.md#421-iceberg-interaction-semantics-simulator-design-decisions)).
- **Price increment.** Slice $k$ of an ICEwPI with base price $b$ and increment
  $d$ is priced $b + k\,d$. A buy must have $d < 0$ and a sell $d > 0$; the wrong
  sign is rejected. No rule bounds the size of $d$
  ([§13.6](omiecid/market-spec.md#136-domain-finding-da-ppd-2026-04-authoritative-ppd-envelope-for-icewpi);
  [OQ-13](omiecid/open-questions.md#oq-13--numeric-magnitude-bound-on-icewpi-ppd)).
- **Resting iceberg.** One aggressive order can trade several slices in one
  matching process. Each regenerated slice is ranked as a new entry at its own
  price, and the iceberg always sets the trade price
  ([§4.2.1 Rules 2 and 3](omiecid/market-spec.md#421-iceberg-interaction-semantics-simulator-design-decisions);
  [OQ-31](omiecid/open-questions.md#oq-31--price-setter-in-the-421-rule-3-hop)).
- **Incoming iceberg.** An iceberg that crosses on entry aggresses with its total
  quantity, as a chain of instances of size $P$. Instance $k$ trades only where
  its own limit $b + k\,d$ crosses, and the sweep stops at the first instance
  that cannot trade. With $M$ the swept quantity, the remainder $R$ rests as
  instance $k_e = \lfloor M/P \rfloor$ and shows $v_1 = \min(P - u, R)$, where
  $u = M - k_e P$
  ([§4.2.1 Rule 2](omiecid/market-spec.md#421-iceberg-interaction-semantics-simulator-design-decisions);
  [OQ-33](omiecid/open-questions.md#oq-33--trade-split-for-an-incoming-iceberg-one-trade-per-counter-bid-or-one-per-incoming-slice),
  [OQ-34](omiecid/open-questions.md#oq-34--does-an-incoming-icewpi-sweep-at-its-base-price-or-at-drifted-prices)).
- **Fill-or-kill against icebergs.** The FOK check counts the hidden quantity,
  but for an ICEwPI only the slices whose drifted price crosses the FOK limit
  ([§4.2.1 Rule 1](omiecid/market-spec.md#421-iceberg-interaction-semantics-simulator-design-decisions);
  [OQ-11](omiecid/open-questions.md#oq-11--fok-interaction-with-icewpi-price-drift-pre-check-quantity-computation)).

## Validation

An order is rejected when its price is outside $[-9{,}999,\ +9{,}999]$ EUR/MWh,
its quantity is outside $[0.1,\ 999]$ MW, the contract is not open, an ICEwPI has
an increment of the wrong sign, or an iceberg's peak is zero or less or exceeds its total quantity
([§5.4](omiecid/market-spec.md#54-order-rejection-conditions)). The guarantee
(operating limit) and transmission system operator (TSO) availability checks of §5.4 are not modelled.

## Price-time priority and matching

- **Priority.** Bids rank highest price first, asks lowest price first; at equal
  price the earlier timestamp wins
  ([§6.1](omiecid/market-spec.md#61-price-time-priority)).
- **Matching.** An incoming order matches the best opposite order when
  $p_{\text{sell}} \le p_{\text{buy}}$. The trade prints at the limit price of the
  order that was already resting, and matching walks to the next price level when
  one is exhausted ([§6.2](omiecid/market-spec.md#62-regular-matching-triggered-by-new-timestamp-order)).
- **Timestamps.** Every order gets a unique, sequential timestamp on entry.
  Partial execution keeps the timestamp; modification resets it
  ([§5.2](omiecid/market-spec.md#52-timestamp-rules),
  [§5.3](omiecid/market-spec.md#53-modification-rules)).
- **No crossed book.** Every crossing created during a matching process,
  including iceberg regeneration cascades, is resolved inside that process
  ([§4.2.1 Rule 4](omiecid/market-spec.md#421-iceberg-interaction-semantics-simulator-design-decisions)).
- **No self-trade prevention** (provisional). Matching ignores ownership: a participant's buy
  can trade against its own sell
  ([§6.6](omiecid/market-spec.md#66-self-trade-prevention-stp-none-at-xbid-level);
  [OQ-12](omiecid/open-questions.md#oq-12--anti-self-match--self-trade-prevention-scope)).
  For the participant's accounting of a self-match, see
  [Sessions](sessions.md#self-trades).
- **Anonymity.** The book shows prices and quantities only; a participant sees
  its own trades, not the counterparty
  ([§7](omiecid/market-spec.md#7-anonymity-and-information-visibility)).
  In OMIE a participant also sees the price and volume of the trades made in its
  area, without names; this version offers no participant call for them.
- **In the simulator.** The session stamps the participant's order with the
  book's arrival counter after the step's background flow, so the order queues
  behind every earlier arrival at its price.

## Partial fills

A NON order that is partly filled rests with its remaining quantity and its
original timestamp. An IOC remainder is cancelled. A FOK either fills completely
or is cancelled without trading
([§4.5](omiecid/market-spec.md#45-execution-restrictions-summary),
[§5.2](omiecid/market-spec.md#52-timestamp-rules),
[§6.2](omiecid/market-spec.md#62-regular-matching-triggered-by-new-timestamp-order)).

## Units and ticks

| Quantity | Unit | Tick | Bounds | Spec |
|---|---|---|---|---|
| Price | EUR/MWh | 0.01 | $-9{,}999$ to $+9{,}999$ | [§2.3](omiecid/market-spec.md#23-price-and-quantity-constraints) |
| Quantity | MW | 0.1 | 0.1 to 999 | [§2.3](omiecid/market-spec.md#23-price-and-quantity-constraints) |
| Delivery duration | h | — | $d = 0.25$ for a quarter-hour | [§2.3](omiecid/market-spec.md#23-price-and-quantity-constraints) |

These are the defaults of `OMIEProductConfig`. Inside the engine, prices and
quantities are integer tick indices; an input between two ticks is rounded to
the nearest tick before the bound check.

## Settlement and sign convention

The value of a trade is $V = p \cdot q \cdot d$, with $p$ in EUR/MWh, $q$ in MW
and $d$ in hours ([§8.1](omiecid/market-spec.md#81-trade-settlement-formula)).
The sell leg books $+V$ and the buy leg $-V$, so a participant's realized result
is

$$\mathrm{PnL} = \sum_{\text{sells}} V - \sum_{\text{buys}} V$$

([§8.4](omiecid/market-spec.md#84-pnl-for-simulator-agents)). The value of the
energy bought or delivered, and the cost of any open position at gate closure,
are outside the market and are not computed
([§8.5](omiecid/market-spec.md#85-closing-out-open-positions)). A self-match
nets to zero.

In a session the settled result is `state.agent_cash` (see
[Sessions](sessions.md#settlement)).
During a session, `FillReport.inventory_delta` is positive for a buy and
`cash_delta_eur` positive for a sale. Scenario C12 certifies that the cash leg of
every trade sums to zero.
