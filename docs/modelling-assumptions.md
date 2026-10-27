# Modelling assumptions

The [market model](market-model.md) follows the
[market specification](omiecid/market-spec.md). Where the regulation is silent,
ambiguous or out of scope, the simulator relies on a recorded ruling or a stated
simplification. This page lists them. Each ruling is recorded as a numbered open
question (OQ-nn) in [open-questions.md](omiecid/open-questions.md); the
identifiers are explained in the [evidence directory guide](omiecid/README.md).
The [certificate](omiecid/certificate.md) lists the same items from the
certification side, as conformance gaps, engineering limits and residual risk;
[Validation](validation.md) explains that evidence, and
[Limitations](limitations.md) lists the limits of the software itself.

## Rulings that interpret the regulation

These rulings are resolved. Each states which source text it rests on and how
confident it is.

- **Price and quantity limits.** Prices are bounded at $\pm 9{,}999$ EUR/MWh with
  a 0.01 tick, quantities at 0.1 to 999 MW with a 0.1 tick. The price limits are
  the harmonised SIDC values and can be revised by SIDC
  ([OQ-01](omiecid/open-questions.md#oq-01--exact-price-and-quantity-bounds-per-product),
  [OQ-06](omiecid/open-questions.md#oq-06--tick-size-minimum-price-increment)).
  OMIE's notification thresholds are warnings, not rejections, and are not
  modelled ([§2.4](omiecid/market-spec.md#24-price-notification-thresholds)). No
  rule bounds the price of executed trades more tightly
  ([OQ-19](omiecid/open-questions.md#oq-19--operational-price-range-of-sidc-continuous-trades-distinct-from-the-9999-hard-envelope)).
- **Gate times.** All contracts of a delivery day open together; closure is
  aligned to the delivery hour
  ([OQ-02](omiecid/open-questions.md#oq-02--gate-opening-time-for-continuous-contracts-exact-rule-vs-nominal),
  [OQ-27](omiecid/open-questions.md#oq-27--what-is-a-tape-product-identity-delivery-date--slot-session-day-file-partition-and-which-files-a-product-spans)).
- **Contract identity.** A contract is (delivery date, slot). OMIE's public files
  are split by session day, so one contract spans two files
  ([OQ-27](omiecid/open-questions.md#oq-27--what-is-a-tape-product-identity-delivery-date--slot-session-day-file-partition-and-which-files-a-product-spans)).
- **Anonymity.** Participant identity is kept inside the engine for accounting;
  the book a participant sees is anonymous
  ([OQ-07](omiecid/open-questions.md#oq-07--agent-identity-and-anonymity-in-order-book)).
- **No self-trade prevention.** No source describes self-trade prevention in the
  SIDC matching engine, so matching ignores ownership (provisional, medium-high
  confidence) ([OQ-12](omiecid/open-questions.md#oq-12--anti-self-match--self-trade-prevention-scope)).
- **Increment of an ICEwPI.** Only the sign is a rule (buy negative, sell
  positive). No magnitude bound is applied
  ([OQ-13](omiecid/open-questions.md#oq-13--numeric-magnitude-bound-on-icewpi-ppd)).
- **Fill-or-kill against an ICEwPI.** The FOK check counts only the slices whose
  drifted price crosses the FOK limit, and each slice trades at its own price,
  without capping
  ([OQ-11](omiecid/open-questions.md#oq-11--fok-interaction-with-icewpi-price-drift-pre-check-quantity-computation)).
- **Slice activation is eager.** A new iceberg slice appears in the same matching
  process that exhausts the previous one, so between matching processes an
  iceberg with quantity left always shows a slice (correction note under
  [OQ-20](omiecid/open-questions.md#oq-20--simulator-internal-iceberg-meta-side-channel-cleanliness-vs-rule-4)).
- **Price of the regeneration hop.** When a regenerated slice crosses the
  aggressor's resting remainder, the trade prints at the slice's own price, not
  at the aggressor's limit; this also holds when the aggressor is an iceberg
  ([OQ-31](omiecid/open-questions.md#oq-31--price-setter-in-the-421-rule-3-hop)).
- **Incoming iceberg.** An iceberg that crosses on entry trades as a chain of
  instances of the peak size. A trade row ends where a resting entry ends or
  where one incoming instance ends
  ([OQ-33](omiecid/open-questions.md#oq-33--trade-split-for-an-incoming-iceberg-one-trade-per-counter-bid-or-one-per-incoming-slice)).
  Each instance trades only at its own limit, and the remainder continues the
  price ladder
  ([OQ-34](omiecid/open-questions.md#oq-34--does-an-incoming-icewpi-sweep-at-its-base-price-or-at-drifted-prices)).
  No source text states the incoming-side split; it rests on a binding ruling
  and on OMIE data. The display of a partly used instance,
  $v_1 = \min(P - u, R)$, is derived from OMIE data (99.93 % of decisive cases),
  not stated by the sources.

The iceberg rules of
[§4.2.1](omiecid/market-spec.md#421-iceberg-interaction-semantics-simulator-design-decisions)
are this project's reading of the SIDC description. The certificate certifies
conformance to these rulings.

## Open questions with a recorded choice

- **Same-second arrivals.** OMIE's files give send times to the second and are
  not in arrival order. Rows sharing a second are processed in file order, a
  proxy for arrival. Trade prices and level volumes do not depend on it; which
  resting order fills first at equal price can
  ([OQ-32](omiecid/open-questions.md#oq-32--same-second-arrival-order-in-replay)).
- **Drifted price beyond the limits.** A deep ICEwPI slice can be priced beyond
  $\pm 9{,}999$ EUR/MWh. The simulator does not clamp it, so such a slice rests
  where nothing can cross it. The market's handling is not stated by the sources
  ([OQ-35](omiecid/open-questions.md#oq-35--a-drifted-instance-or-slice-price-beyond-9999-eurmwh)).
- **Daylight-saving days.** Contract strings and slot counts change on the two
  transition days. Loading a contract whose window contains a transition is
  refused unless the caller opts in
  ([OQ-30](omiecid/open-questions.md#oq-30--daylight-saving-transitions-contract-strings-slot-count-and-wall-clock-timestamps)).
- **Orders before gate opening.** How a submission before gate opening is
  handled is not stated; none is observed in the data
  ([OQ-14](omiecid/open-questions.md#oq-14--pre-got-order-submission-behavior)).

## Simplifications

- **Single local book, no cross-border flow.** Each contract is one Iberian book.
  Cross-border capacity, remote orders and batch matching are not modelled
  ([§6.3](omiecid/market-spec.md#63-batch-matching-triggered-by-cross-border-capacity-increase),
  [§6.5](omiecid/market-spec.md#65-cross-border-visibility)). The public files
  hold only Iberian participants' orders; a trade with a non-Iberian
  counterparty shows that side blank, and its order cannot be replayed
  ([§3.1](omiecid/market-spec.md#31-continuous-market-gate-times),
  [§13.2](omiecid/market-spec.md#132-trades-file-columns)).
- **No intraday-auction suspension.** Rounds and the two suspension phases
  around each intraday auction are not driven. Resting orders are carried across
  the auction window and later orders are applied one by one; the batch match at
  reopen is not run
  ([§3.3](omiecid/market-spec.md#33-continuous-market-rounds-and-ida-suspension-model);
  [OQ-28](omiecid/open-questions.md#oq-28--do-continuous-orders-survive-the-ida-clearing-window-and-how-is-the-book-re-matched-at-reopen)).
- **No halts or hibernation**
  ([§3.4](omiecid/market-spec.md#34-halts-and-hibernation)).
- **No block, basket or all-or-nothing orders**
  ([OQ-08](omiecid/open-questions.md#oq-08--block-order-constraints-minimummaximum-block-length),
  [OQ-09](omiecid/open-questions.md#oq-09--basket-order-handling-independence-of-sub-orders-from-capacity-constraints)).
- **GTD lapses at the gate only.** Validity is carried on the order, but there is
  no expiry sweep before gate closure
  ([§4.6](omiecid/market-spec.md#46-validity-conditions)).
- **Partial validation.** The guarantee (operating limit) and TSO availability
  checks are not modelled
  ([§5.4](omiecid/market-spec.md#54-order-rejection-conditions)). Prices and
  quantities between two ticks are rounded to the nearest tick, not rejected.
- **Modification** exists at book level but is not certified
  ([§5.3](omiecid/market-spec.md#53-modification-rules)).
- **Settlement per participant, not per bidding unit**
  ([§8.3](omiecid/market-spec.md#83-settlement-granularity-per-unit-hard-rule)).
  The list of a participant's trades carries price, quantity and time, not side
  ([§9.2](omiecid/market-spec.md#92-observable-market-state-variables)).
- **No firmness or imbalance.** Delivery obligations, imbalance costs and any
  close-out price are outside the market and not computed
  ([§8.2](omiecid/market-spec.md#82-trade-firmness),
  [§8.5](omiecid/market-spec.md#85-closing-out-open-positions);
  [OQ-24](omiecid/open-questions.md#oq-24--terminal-forced-close-out-reference-price-for-an-open-agent-position-at-episode-end-inventory-aware-reward)).
- **Self-matches.** As OMIE has no self-trade prevention, a participant's order
  can trade against its own resting order. The session attributes the passive
  side of such a match, so inventory and cash net to zero within float32
  rounding ([Sessions](sessions.md#self-trades)).
- **One participant per session.** Several participants can trade directly at
  book level (`ContractBook`) ([Limitations](limitations.md#session)).
- **No cancellation inside the compiled step.** Cancel between steps, eagerly,
  with `report.order_id` ([Sessions](sessions.md#cancelling-an-order)).

## Replaying public OMIE data

- **No cancellations in the data.** OMIE's order files are a submission log with
  no order IDs and no cancellations. A replayed order that does not trade rests
  until gate closure, which can leave liquidity in the book that was withdrawn in
  reality ([OQ-10](omiecid/open-questions.md#oq-10--cancelled-orders-not-identifiable-in-omie-public-data)).
  The certification covers the engine, not this replay heuristic. After an
  intraday auction, re-sent orders can be counted twice (extension of OQ-10,
  open).
- **Incomplete contracts are excluded.** A contract needs both session files,
  for orders and for trades; a contract with a missing file is not loaded
  ([OQ-29](omiecid/open-questions.md#oq-29--incomplete-delivery-products-a-required-session-file-missing)).

## Engineering limits

None of these is a market rule.

- **Trade buffer.** The book keeps up to `nTrades` trades (default 100); beyond
  that, new trades overwrite the last row, and `FillReport.trades_saturated`
  flags it. Set `nTrades` at least to the expected number of trades in the
  session ([Sessions](sessions.md#sizing-ntrades),
  [Limitations](limitations.md#session)). On a full buffer an incoming iceberg's rows are left unsplit
  ([OQ-15](omiecid/open-questions.md#oq-15--ring-buffer-occupancy-invariant-for-omiebookstatelobtrades)).
- **Fixed capacities.** The resting orders per side, the icebergs and the slices
  per iceberg are bounded by fixed array sizes. Their behaviour at each limit is
  recorded in the [certificate](omiecid/certificate.md) (scenario C14).
