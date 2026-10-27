# OMIE Continuous Intraday Market — Open Questions

**Status:** OQ-01–OQ-11 resolved; OQ-12 provisionally resolved; OQ-13 provisionally resolved (no rule-level bound); OQ-14 open (pre-GOT buffering); OQ-15 open (ring-buffer occupancy — engineering invariant, tracked here per P2-HOTFIX-ioc-fill-detection-FOLLOWUP directive); OQ-16 open (Method A XLA-safety envelope — engineering invariant, tracked per P2-L1-HOTFIX-tick-precision Option C adjudication); OQ-17 open (full `OMIEOrder` int32 migration — engineering debt, deferred to P2-H2 or Phase 3); OQ-18 open (Phase 3 reopen of `jax.jit(compute_agent_pnl)`); OQ-19 open (operational SIDC price range distinct from ±9,999 hard envelope); OQ-20 open (simulator-internal iceberg-meta cleanup invariant — engineering invariant, tracked per P2-PERF-12 scope ruling 2026-05-28); OQ-21 open (no corpus rule for per-contract liquidity / traded-volume ranking or near-GCT concentration — filed per P3-PERF-TAPE-1 tape-selection ruling 2026-07-03; spec-grounded proxy recorded in market-spec §14); OQ-22 resolved (P3.2b-FOLLOWUP-1 touch-crossing agent order is faithful to OMIE CID semantics; OQ-10 phantom-liquidity interaction acceptable-for-v1 with a STAGE01 run-report banner — filed + ruled 2026-07-08); OQ-23 resolved/closed (latent `regenerate_slices` quiescence gap: CONSTRUCTIBLE but NOT a Rule 4 violation — resting book stays uncrossed; the OQ-20-class ENGINEERING invariant, no market-spec change — filed + ruled 2026-07-09; the scoped ENGINEERING follow-up landed via P3-PERF-REGEN-QUIESCENCE-1 [merged `3d604e3`], NOT P3-PERF-REGEN-MASK-1 [ABANDONED, shipped nothing]: the outer fixed-point now re-arms on cross-slot peak consumption, the `iceberg_manager.py:352-364` post-condition genuinely holds on the gap shape, and the `no_fill_path` reliance is sound — see §OQ-23 CORRECTION + CLOSURE NOTE and the locked constructor `tests/omie/test_regen_quiescence_rearm.py`; closed 2026-07-09); OQ-24 resolved (domain-semantics half of `P3-REWARD-INVENTORY-1` inventory-aware reward: a terminal forced-close-out reference price is NOT a first-class OMIE quantity — the continuous market has no automatic close-out [§8.5], the counter-leg energy value is explicitly exogenous [§8.4], and imbalance resolution is outside OMIE scope and classified a *modeling choice* [§8.5/§10/§11.2]; NO reference price is spec-grounded — `day_ahead_anchor` is not in the spec and cross-market [not domain-defensible without an explicit basis caveat], contract VWAP is the *less*-unfaithful modeling assumption, penalty-only [candidate (b)] invents no exogenous price and is most domain-honest; firm-delivery → the consequence is an exogenous TSO-set imbalance cost, which supports a PENALTY framing over a MARK-TO-MARKET framing; NO market-spec change; the (a)/(b)/(c) RL-mechanism choice is routed to the rl-research-engineer, out of scope here — filed + ruled 2026-07-10); OQ-25 resolved-diagnosis (agent-order fill-starvation in the non-reactive H1 replay = OQ-10 phantom residual out-prioritising the agent's later-timestamped maker order on price-time priority; reactive-tape build WELL-POSED, fix locus is the replay heuristic/tape side only per OQ-22, never the agent action semantics; NO spec change — filed 2026-07-22, `WF-25-FOLLOWUP-1`); OQ-26 filed (day-ahead obs anchor is episode-constant for a single-contract episode; intra-episode variation domain-grounded only for multi-contract; obs-only, orthogonal to OQ-24 — filed 2026-07-22, WF-13). **2026-09-30, `OMIE-TAPE-CONTRACT-DATE-1` Step 0:**
- OQ-27 resolved: a tape is the delivery product (D, k); the file date and `Fecha` are the session day; GCT is hour-aligned; a product spans files D−1 and D only; the "D+1 when k ≥ 93" rule is refuted.
- OQ-28 filed: orders survive the IDA and are batch-matched at reopen, medium-high confidence; sequential replay is a recorded simplification.
- OQ-29 resolved: incomplete products are excluded. (On-disk snapshot refreshed on 2026-09-30 by `OMIE-TAPE-CONTRACT-DATE-1-DOC-1`: 2025-06-01 is now complete and 2025-05-31 is incomplete. No rule change.)
- OQ-30 open: DST contract strings, slot count, and wall-clock timestamps; guard only.
- OQ-10 extended with the post-IDA re-quote double-count (OPEN, measurement pending).
- OQ-14 annotated: its negative-timestamp trigger was the file-date error, not pre-GOT sends.
- OQ-25 annotated (2026-10-01, `OMIE-AGENT-TIME-PRIORITY-1-DOC-1`): the "agent queues at the back" premise was false before `OMIE-AGENT-TIME-PRIORITY-1`. The proximate-mechanism attribution is flagged for re-measurement. The status, root attribution and fix-locus ruling are unchanged.

**2026-10-02, scenario-ladder domain review v3.0 (V3-02):**
- OQ-31 filed: who sets the price in the §4.2.1 Rule-3 hop (the resting remainder, or the iceberg slice).
- **OQ-31 RESOLVED the same day (2026-10-02), Reading B:** the iceberg slice sets the price, and the hop trades at the slice's (drifted) limit, not the aggressor's limit $L$. Evidence: OMIE Rules Iceberg §4.4 (a counter bid larger than the visible peak trades each iceberg instance, and the iceberg stays in the book), Matching Algorithm §5.1.2 (price follows the iceberg even when its timestamp is renewed) and §5.1.3 (the aggressor's remainder enters the book only once it can no longer execute). Applied as a clarification in market-spec v2.13 §4.2.1 Rule 3. This unblocks C8's K2 and I5's Rule-3 branch.

**2026-10-07, `OMIE-SPEC-CERT-AMEND-1` (certification fix pass, market-spec v2.14; additive only):**
- OQ-20 annotated with a correction note: slice activation is **eager** (Matching Algorithm §3.3). The original ruling text is kept, and its engineering conclusion is not reopened.
- OQ-31 annotated with an addendum: when the aggressor is an incoming iceberg, the resting iceberg still sets the hop price. The architect countersigns.
- OQ-32 filed, OPEN: same-second arrival order in replay. OMIE files are not arrival-ordered within a second. The simulator's "processing order" is a recorded modelling choice.
- OQ-33 filed and RESOLVED: an incoming iceberg records **one trade per counter bid**, each resting instance being its own counter bid (Annex §4.4). Confirmed at `OMIE-ICE-INCOMING-HOP-1` Step 2. **Revised by the user's ruling of 2026-10-20:** rows also split at incoming instance boundaries (see below).
- OQ-34 filed, OPEN: an incoming ICEwPI sweeps at its base price or at drifted prices. The simulator's base-limit sweep is a recorded modelling choice. **Resolved by the user's ruling of 2026-10-20:** instance chain, continuation, C3 display (see below).

**2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1` (certification fix pass v3, S1; market-spec v2.14, dated non-additive amendment of §4.2.1 Rule 2 *Incoming iceberg*):**
- OQ-33 REVISED by user ruling: an incoming iceberg is a chain of instances of size $P$; each trade is between one resting counter-bid entry and one incoming instance, so rows split at resting entries **and** at incoming instance boundaries. The 2026-10-07 ruling is superseded for incoming icebergs.
- OQ-34 RESOLVED by user ruling: instance $k$ trades only where its limit $b + k\,d$ crosses; the sweep stops at the first instance that cannot trade; the remainder continues the ladder as instance $k_e = \lfloor M/P \rfloor$ and shows $v_1 = \min(P - u, R)$ (display C3, derived from OMIE data over 317 session days). C3 display confirmed by the user 2026-10-22.
- OQ-35 filed, OPEN (question only): a drifted instance or slice price beyond ±9,999 EUR/MWh (§13.6 item 2). No engine change.

**Date:** 2026-10-07 (originally 2026-03-20)
**Owner:** Domain Analyst (`/domain-analyst`). This document must only be modified by the Domain Analyst role.
**Related spec:** `docs/omie/market-spec.md`

These are unresolved ambiguities or missing specifics identified during domain analysis. Architecture and implementation work must not silently resolve these — each open question must be explicitly decided before the affected component is designed.

---

## OQ-01 — Exact price and quantity bounds per product

**What is unknown:** The OMIE Market Rules ("Reglas de funcionamiento de los mercados diario e intradiario de electricidad", approved by CNMC Resolution DCOOR/DE/009/24, effective 13 June 2024) contain the specific minimum/maximum price and minimum/maximum quantity constraints per product (hourly, 15-min, block). These values are referenced in the XBID matching algorithm spec (§3.1) but not reproduced in the available source documents.

**Decision dependency:** Order validation logic, LOB price axis bounds, and RL action space bounds.

**Where to look:** OMIE Market Rules document (Spanish), Annex 1 (timing schedule) and the product specification articles. Document path in knowledge base: referenced in CNMC_Resolution_DCOOR_DE_009_24.md but the rules document itself is the OMIE Rulebook (not fully reproduced in corpus).

**Suggested resolution:** Retrieve current numeric bounds from OMIE's published Electricity Market Activity Rules or OMIE's API documentation.

**Resolution:** **Resolved** (with caveats noted below).

Price bounds: OMIE Market Rules Annex 3 (OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md, pp. 208–209) establishes that the maximum and minimum price limits for the continuous intraday market are the **harmonized SIDC clearing price limits**, set per the ACER-approved Single Intraday Coupling methodology. Current values confirmed via web research: **−9,999 EUR/MWh (minimum) to +9,999 EUR/MWh (maximum)**. These are the hard bid-rejection limits. Additionally, Annex 3 defines *notification thresholds* (softer warnings, not rejections): approximately −150 EUR/MWh (minimum) and +1,500 EUR/MWh (maximum) for the continuous intraday market in the MIBEL zone. These thresholds may be modified by OMIE directive.

Quantity bounds: Annex 2 of the OMIE Rules and the European Continuous Trading Platform configuration govern quantity limits. Per the XBID matching algorithm spec (§3.1, Continuous_Trading_Matching_Algorithm_Public_Description.md): "Products also have a minimum and maximum contract quantity." OMIE Rules §45.2.1 state the maximum power amount is "pursuant to the provisions of the European Continuous Trading Platform." Web research indicates the SIDC-level quantity range is **0.1 MW (minimum) to 999 MW (maximum)**, with resolution of 0.1 MW. The OMIE rules (§38.2, §45.2.1) confirm quantities are expressed in MW to one decimal place (0.1 MW resolution).

Caveat: Annex 3 explicitly states that current price limits are dynamic and "can be consulted on the agent website." The ±9,999 EUR/MWh values are current as of this writing but may be revised under the harmonized SIDC methodology. For simulator implementation, use ±9,999 EUR/MWh as the hard action-space bounds.

**Sources:** OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (Annex 3, §45.2.1); Continuous_Trading_Matching_Algorithm_Public_Description.md (§3.1); web research confirming SIDC limits.

---

## OQ-02 — Gate Opening Time for continuous contracts: exact rule vs. nominal

**What is unknown:** The XBID Information Package (§8) states that gate times for SIDC market trading are product- and border-specific, with a table not rendered in the local document. The nominal GOT of ~15:20 CET (after IDA1) from OMIE Instruction 1/2024 applies to the Iberian zone for delivery day D+1. It is unclear whether:
(a) all 24 hourly contracts of the delivery day open simultaneously at IDA1 completion, or
(b) contracts open on a rolling basis as the delivery time approaches.

**Decision dependency:** Contract lifecycle management; when to create and open order books in the simulator.

**Suggested resolution:** Check OMIE Market Rules Annex 1 (timing schedule) which is referenced in OMIE Instruction 1/2024 as containing "the most important timings of the intraday markets (IDAs and continuous)."

**Resolution:** **Resolved.**

The OMIE Market Rules Annex 2 schedule table (OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md, pp. 200–202) and OMIE Instruction 1/2024 together answer this definitively: **all 96 quarter-hourly contracts (or 24 hourly contracts in the pre-MTU15 model) for delivery day D+1 open simultaneously at 15:20 CET on day D**, immediately after IDA1 matching completes (IDA1 GCT 15:00, results confirmed 15:20). This is "Round 18" in the OMIE continuous market schedule.

Specifically, from the Annex 2 schedule table: at D period 16 (15:20–16:00), continuous round 18, all contracts 1–96 for D+1 open for trading simultaneously. OMIE Instruction 1/2024 (§7) confirms: "continuous trading for delivery day 14/06/2024 will open at 15:20h (13/06/2024) starting with Round 18th, after successful completion of IDA1."

Contracts close on a **rolling basis**: the schedule table shows contracts closing one round at a time (each "round" covers a 1-hour wall-clock window, closing 4 quarter-hourly contracts per round). The last contracts for a delivery day close approximately 1 hour before physical delivery of that period, consistent with the stated GCT of "one hour before delivery" in the OMIE details document.

**Sources:** OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (Annex 2 schedule table, pp. 200–202); OMIE_Instruction_1_2024_IDA_Go-Live_Operations.md (§7).

---

## OQ-03 — Whether 15-minute MTU products are active for Spain in SIDC

**What is unknown:** ACER Decision 03/2025 amends SIDC products to introduce 15-min MTU products and IDA mandatory/optional products. The Decision was dated 27 March 2025. The XBID Information Package (older) lists 15-min as supported in SIDC but notes product availability varies per market area. It is unclear whether Spain/Iberia has activated 15-min products in SIDC as of 2026-03-20.

**Decision dependency:** Whether the v1 simulator must model 15-min contracts (24×4 = 96 contracts per day) or only hourly (24 contracts).

**Suggested resolution:** Check OMIE's published product configuration or the current OMIE market rules for the active product set per delivery area.

**Resolution:** **Resolved.**

**15-minute MTU products are active in Spain for SIDC as of 18 March 2025.** OMIE launched 15-min quarter-hourly (MTU15) products simultaneously for the continuous intraday market (MIC) and the intraday auctions (IDAs) on 18 March 2025, with first delivery on 19 March 2025. This is confirmed by the OMIE press release (published on omip.pt, 2025-03-18). The OMIE Market Rules (Annex 2, §9841–9857 in the corpus) already reflect the quarter-hourly product as the primary product: "The quarter-hourly product for the continuous intraday market will be composed of 96 contracts… each contract corresponds to each of the quarter-hourly periods of the day."

Therefore, the current active SIDC product set for Spain/MIBEL as of 2026-03-20 is:
- **96 quarter-hourly (15-min) contracts per delivery day** for the continuous intraday market.
- The same for IDAs (IDA1: periods 1–96 of D+1; IDA2: periods 1–96 of D+1; IDA3: periods 49–96 of D+1).

The v1 simulator should model 96 quarter-hourly contracts per delivery day, not 24 hourly contracts. The schedule table in Annex 2 of the OMIE Rules already uses this 96-contract structure.

Note: ACER Decision 03/2025 (27 March 2025) governs the updated SIDC product framework at EU level; Spain's activation pre-dates this decision, which was primarily a framework harmonization rather than a trigger for Spain's activation.

**Sources:** OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (Annex 2, §9841–9857); ACER_03_2025_SIDC_PRODUCTS.md; web research confirming OMIE MTU15 go-live on 18 March 2025.

---

## OQ-04 — Exact IDA session timing under normal operations

**What is unknown:** OMIE Instruction 1/2024 gives nominal times for IDA1 (GOT 14:00, GCT 15:00, results 15:20), IDA2 (GOT 21:00, GCT 22:00, results 22:20), IDA3 (GOT 09:00, GCT 10:00, results 10:20). However:
(a) IDA3 covers only Hours 13–24 of the delivery day. It is unclear whether additional IDA sessions (IDA4, IDA5, etc.) exist covering the remaining hours closer to delivery, or whether the continuous market handles intraday adjustments for hours 1–12 exclusively via continuous trading after IDA3.
(b) Whether there is an IDA for Hours 1–12 on the delivery day itself (e.g. a morning session).

**Decision dependency:** Simulator time structure; when to pause capacity allocation; how many IDA sessions to model.

**Suggested resolution:** Consult OMIE Market Rules Annex 1 (the timing table) and the current IDA schedule published by OMIE/SIDC.

**Resolution:** **Resolved.**

There are exactly **3 IDA sessions** per delivery day in the current SIDC/OMIE configuration. The OMIE Annex 2 schedule table (corpus pp. 200–202) and OMIE Instruction 1/2024 together show the complete picture:

| Session | GOT (day D) | GCT | Results | Coverage (quarter-hourly contracts) |
|---------|-------------|-----|---------|--------------------------------------|
| IDA1    | 14:00       | 15:00 | 15:20 | Periods 1–96 of D+1 (full day) |
| IDA2    | 21:00       | 22:00 | 22:20 | Periods 1–96 of D+1 (full day) |
| IDA3    | 09:00 (D+1) | 10:00 | 10:20 | Periods 49–96 of D+1 (hours 13–24) |

IDA3 intentionally covers only the second half of the delivery day (periods 49–96, i.e. hours 13–24) because periods 1–48 (hours 1–12) have already passed or are imminent by the time IDA3 runs at 09:00–10:20 on the delivery day itself. For those periods, only the continuous market serves. There is no IDA4, IDA5, or morning session for hours 1–12 on the delivery day under the current SIDC framework for Spain.

Each IDA also halts continuous capacity allocation approximately 20 minutes before GCT (i.e., ~20 min pre-GCT the capacity halt occurs). OMIE Instruction 1/2024 confirms: for IDA2, "Continuous capacity allocation during Round 24th will be halted at 21:40h."

**Sources:** OMIE_Instruction_1_2024_IDA_Go-Live_Operations.md (§6, §8, §9); OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (Annex 2 schedule table); web search confirming 3 IDAs per day.

---

## OQ-05 — Capacity halt during IDA: does order matching also halt?

**What is unknown:** OMIE Instruction 1/2024 states "Continuous capacity allocation during Round Nth will be halted at ~[GCT − 20 min]." The Continuous Trading Matching Algorithm spec (§5.5) states batch matching suspends regular matching and explicit allocation. It is ambiguous whether:
(a) Regular matching (new order entries triggering immediate match) is also suspended during the ~20-minute pre-IDA window, or only cross-border (batch) matching is suspended.
(b) Order submission itself is halted during this window.

**Decision dependency:** Whether agents can still submit orders and trigger local (intra-zone) matches during the IDA pre-window.

**Suggested resolution:** The Continuous Trading Matching Algorithm spec §5.5 implies only batch matching suspends regular matching; regular matching continues. Capacity-constrained visibility updates are suspended. Confirm with OMIE operational documentation.

**Resolution:** **Resolved** (with distinction between the pre-IDA capacity halt and the IDA matching window clarified).

Two separate suspensions apply in the IDA pre-window and during IDA clearing:

1. **Pre-IDA capacity halt (GCT − 20 min, e.g. 21:40 for IDA2):** Cross-border capacity allocation is halted. The XBID CMM stops accepting new implicit capacity allocations from cross-border matches. However, **local (intra-zone) order submission and matching are not suspended** during this pre-IDA window; agents continue to submit orders and local matches can still execute.

2. **During IDA clearing (GCT to results, e.g. 22:00–22:20 for IDA2):** Per Continuous_Trading_Matching_Algorithm_Public_Description.md §5.5: "As soon as a batch matching round starts, both regular matching and explicit capacity allocation are suspended." This means that **during the ~20-minute IDA clearing window, both regular matching and explicit allocation are suspended**. Order submission may or may not be technically possible during this window depending on local implementation, but matched execution is frozen. OMIE Instruction 1/2024 §8 confirms: "Continuous allocation (and order submission) will continue with Round 1st (14/06/2024) at 22:20h (13/06/2024) after the successful completion of IDA2" — implying both allocation and order submission resume only after IDA completion.

For the simulator: model the GCT−20min point as the start of the cross-border capacity freeze. Model the GCT-to-results window (20 minutes) as a full suspension of both matching and effective order submission for the affected delivery-day contracts.

**Sources:** Continuous_Trading_Matching_Algorithm_Public_Description.md (§5.5); OMIE_Instruction_1_2024_IDA_Go-Live_Operations.md (§8); web research on MDPI study confirming bidding suspension during IDA clearing.

---

## OQ-06 — Tick size (minimum price increment)

**What is unknown:** The minimum price increment (tick size) for OMIE CID contracts is not stated in the available source documents (it is in the OMIE Market Rules numeric parameters, same source as OQ-01).

**Decision dependency:** LOB price axis resolution; RL action space discretization.

**Suggested resolution:** Retrieve from OMIE Market Rules or OMIE API specifications.

**Resolution:** **Resolved.**

**Price tick size: 0.01 EUR/MWh.** The OMIE Market Rules (§38.2 and §45.2.1, OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md) state that bid prices are expressed "in euros per MWh, with a maximum of two decimal places." This two-decimal-place constraint in EUR/MWh corresponds to a minimum price increment of 0.01 EUR/MWh. This is confirmed by web research on the SIDC/XBID system specification.

**Quantity tick size: 0.1 MW.** Both the OMIE Rules (§38.2: "power amount in MW… rounded to a maximum of one decimal place") and web research confirm a minimum quantity increment of 0.1 MW.

**Sources:** OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (§38.2, §45.2.1); web research on SIDC quantity/price parameters.

---

## OQ-07 — Agent identity and anonymity in order book

**What is unknown:** The XBID spec states orders are "anonymized" when sent from LTS to SOB. It is unclear whether:
(a) In the simulator, agent identity attached to orders should be tracked internally (for PnL attribution) while the LOB itself is anonymous.
(b) The Trade Book published to participants includes counterparty identity.

**Decision dependency:** Observation space design for RL agents.

**Suggested resolution:** OMIE Details of Intraday Market §3.2 states results are communicated to the agent — implying the agent knows their own match but not the counterparty. Assume internal tracking + anonymous LOB. Mark as a modeling choice if counterparty information is irrelevant for the simulator.

**Resolution:** **Resolved** (confirmed by corpus, classified as Modeling Choice for the simulator implementation).

The OMIE Rulebook §45.2.12 (OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md, p. 112) explicitly states: "information available on the Market Operator's Market Platform for market agents regarding products, contracts and data on the orders contained on the list of local orders is to be displayed on an **anonymous basis**, identifying solely those operations performed by the agent itself." The XBID Information Package (§2) confirms orders are anonymized when sent from LTS to SOB.

This means: (a) The order book (local view) is anonymous — agents see prices and quantities but not counterparty identity. (b) Each agent sees only their own matched trades in the trade notification ("the result is to be notified to the agent in question" — §46.1). Counterparty identity is not disclosed.

**Modeling choice for simulator:** Maintain agent identity internally (for PnL accounting and observation). Expose the LOB to agents as anonymous. Each agent's observation includes only their own pending orders and matched trades. This matches the real market rule exactly.

**Sources:** OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (§45.2.12, §46.1); Single_Intraday_Coupling_XBID_Information_Package.md (§2).

---

## OQ-08 — Block order constraints: minimum/maximum block length

**What is unknown:** User-defined block orders span "multiple consecutive delivery periods." The minimum and maximum block length (in number of MTUs) is not specified in the available documents.

**Decision dependency:** Whether to model block orders in v1, and the valid block order parameter space.

**Suggested resolution:** Check OMIE Market Rules for block order constraints. For v1, consider deferring block orders.

**Resolution:** **Modeling Choice** (Deferred for v1; minimum structural parameters confirmed from corpus).

From the Continuous_Trading_Matching_Algorithm_Public_Description.md (§3.4): "a block order is an order with a total delivery period **greater than** the smallest delivery period of the product for which the order is being entered." This sets the minimum block length as ≥ 2 MTUs (i.e. ≥ 30 minutes for 15-min products, ≥ 2 hours for hourly products). No explicit maximum block length is stated in the algorithm spec or OMIE Rules corpus; in practice, the maximum is limited to the number of contracts remaining open in the delivery day (up to 96 for 15-min products or 24 for hourly).

For IDA auction block orders ("Simple Block Orders"), OMIE Rules §9.629 states a maximum of 5 Simple Blocks per bid, but this is an IDA auction constraint, not a CID constraint.

For CID specifically, the OMIE Rules do not enumerate a numeric minimum or maximum block span beyond the definitional minimum of > 1 MTU. The specific parameter constraints (e.g., maximum span) are delegated to "the provisions of the European Continuous Trading Platform" (§45.2.1) and not hard-coded in the Rules text available in corpus.

**Recommendation:** Defer block orders from v1 of the simulator. When implemented: use minimum block length = 2 MTUs, maximum = remaining contracts in the open trading window (up to 96 for 15-min products). Mark as modeling choice pending confirmation from OMIE API documentation.

**Sources:** Continuous_Trading_Matching_Algorithm_Public_Description.md (§3.4); OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (§45.2.1, §9.629).

---

## OQ-09 — Basket order handling: independence of sub-orders from capacity constraints

**What is unknown:** When a basket order spans multiple contracts and market areas, and one sub-order fails due to capacity constraints, it is unclear how the basket fails/succeeds in the AON and independent cases. The XBID spec (§3.6, §3.7) does not detail this scenario explicitly.

**Decision dependency:** Basket order implementation logic.

**Suggested resolution:** For a single-zone v1 simulator, basket orders across zones are not needed. Defer multi-zone basket behavior. Within-zone: AON basket fails if any sub-order cannot be fully matched; independent basket: each sub-order treated independently.

**Resolution:** **Modeling Choice** (Deferred for multi-zone; within-zone behavior confirmed from corpus).

The Continuous_Trading_Matching_Algorithm_Public_Description.md §3.6 states: "All orders in a linked basket must be executed immediately and with their full quantity (i.e., they always have a FOK execution restriction). If this is not possible, the linked orders will all be deleted." The XBID Information Package (§4) confirms the same for Linked (AON) baskets. For independent baskets, sub-orders are "processed as if they would have been submitted separately" (XBID Information Package, §4 footnote 1).

The OMIE Rules §45.2.10 confirm basket support in the continuous intraday market, with conditions for execution/validity specified in Annex 2. The basket is rejected entirely if any included contract is not in TRADE status (§45.2.10).

For the v1 single-zone simulator: implement only independent basket behavior (each sub-order independent, FOK/linked behavior deferred). Cross-zone basket capacity-constraint behavior is irrelevant for single-zone scope and deferred.

**Sources:** Continuous_Trading_Matching_Algorithm_Public_Description.md (§3.6); Single_Intraday_Coupling_XBID_Information_Package.md (§4); OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (§45.2.10).

---

## OQ-10 — Cancelled orders not identifiable in OMIE public data

**What is unknown:** Whether orders that were submitted and later cancelled (by the agent, by IOC/FOK rules, or by GCT expiry) appear in the publicly downloadable OMIE order files, and if so, how they are represented.

**Decision dependency:** M7 data loading; LOB reconstruction fidelity; replay-based simulator validation.

**Resolution:** **Resolved.**

**Cancelled orders are not explicitly identifiable in the OMIE public order data.** The orders file published at `https://www.omie.es/es/file-download` is a **submission log**, not a lifecycle log. Key findings:

1. **No status field.** The orders file has 13 columns (see `market-spec.md` §13.1). There is no `status`, `cancelled`, or `active` column. Every submitted order appears exactly once, at the time of submission, regardless of its eventual outcome.

2. **No order ID.** Order IDs are internal to the XBID/SOB system and not published. Without IDs, even if cancellation records were published separately, they could not be linked back to specific submissions.

3. **IOC/FOK orders appear in the file.** The ~1% IOC and <0.01% FOK orders observed in the data (see `market-spec.md` §13.3) confirm the file records submissions — these orders by definition never rest in the book (IOC: remainder cancelled; FOK: entirely cancelled if not fully fillable).

4. **Regulatory basis.** OMIE Rulebook §15.b states "the publication of the bids sent to the European Continuous [Trading] Platform, after 90 days have passed." The operative term is "bids sent" — submissions, not lifecycle events. The 90-day delay is a REMIT transparency provision.

**Implications for LOB reconstruction:** The order book state at any point in time **cannot** be reconstructed from the orders file alone. Using both orders and trades files together, one can compute the **unmatched residual** for each submission (submitted qty minus traded qty), but cannot distinguish between three outcomes for that residual:
- Cancelled by the agent during the trading session
- Expired at GCT (unmatched at contract close)
- Still resting in the book when the trading session ended

All three are observationally equivalent in the public data. Any LOB reconstruction or replay-based validation will require heuristic assumptions about the fate of unmatched order volume.

**Sources:** `docs/omie/market-spec.md` §13.1, §13.4; OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (§15.b, lines 722–726).

**Decision impact:**
- `migration-plan.md` §M7.2: replay heuristic H1 adopted; `replay_orders_through_book()` parameterized with heuristic selection (default: `rest_until_gct`)
- `migration-plan.md` §M7 tests: `test_replay_produces_matching_trades()` reframed as superset-matching with divergence tolerance
- `migration-plan.md` §2.4 tests: `test_replay_without_agent_matches_history()` reframed as approximate matching with tolerance
- `migration-plan.md` Risks table: new risk R-OQ10 (replay divergence) added
- Pre-M7 spike recommended: measure empirical divergence on real data before committing to tolerance thresholds
- Book-overflow eviction (added 2026-06-11, `P3-PREP-REPLAY-1-FOLLOWUP-1`): the full-side worst-price level-wipe at the `book_depth=100` cap is an **accepted documented limitation** — a corrective fix (single-victim price-time eviction, wrapper-only) was designed and deferred, with no fix stories registered; the busy-contract replay-fidelity gap is reclassified under this OQ (the OQ-10 over-matching mechanism, not the cap, dominates the gap). See `book-depth-cap-finding.md` §Resolution (status now "Closed — documented accepted limitation (2026-06-11)").
- Agent-order fill-starvation (added 2026-07-22, `WF-25-FOLLOWUP-1`): the H1 phantom residual out-prioritises the agent's later-timestamped resting maker order on price-time priority, starving it of background-aggressor flow (PASSIVE own-touch 0.63%, MID 12%, CROSS 99.2%). This is the symmetric face of the OQ-10 over-match. Diagnosis + reactive-tape well-posedness ruling: see `reactive-tape-fill-diagnosis.md` and OQ-25.
  - CORRECTION (2026-10-01, OMIE-AGENT-TIME-PRIORITY-1), pointer only: the "later-timestamped" premise is withdrawn. Before that fix the agent was stamped with the minutes clock and sat AHEAD of same-price non-agent orders. See the CORRECTION block under OQ-25. This OQ-10 ruling is unchanged.

### OQ-10 EXTENSION — post-IDA re-quote double-counting (added 2026-09-30, `OMIE-TAPE-CONTRACT-DATE-1` Q3; the root ruling above is unchanged)

**Status: OPEN (measurement pending).** The measurement is card Deliverable (4), report-only. There is no de-duplication in that card.

**Context.** Since market-spec v2.10 (§13.4, OQ-27), a tape is the full delivery product (D, k), built from session files D−1 and D. The merged tape therefore crosses the IDA2 gap at 22:00–22:20 on D−1. Orders sent before the gap survive it and are batch-matched at reopen (market-spec §3.3, OQ-28).

**The ambiguity.**
- Public data has no order IDs (finding 2 above) and no cancellations (finding 1).
- Right after 22:20, participants send **re-submission bursts**. For example, DREUA sends an identical V 115.24 × 9.70 at 15:22:24 and again at 22:20:48 (as measured in the card), and there are 1,733 rows in 22:20–22:24.
- The public data cannot distinguish two cases:
  - (a) the participant cancelled the pre-IDA order (or it was consumed) and re-sent it;
  - (b) the participant added a second, independent order at the same terms.
- Under H1 (`rest_until_gct`) the pre-IDA copy is never cancelled. The replay then books **both** copies, which double-counts the re-quoted liquidity. This worsens the OQ-10 over-matching exactly at the reopen boundary.

**What is needed to resolve it.** Card Deliverable (4) measures, per held-out product, the rows sent in 22:20–22:30 whose (Agente, Unidad, side, price, quantity) equals a row sent earlier in the same product's D−1 part: count, MW, and share of the post-IDA burst.

**Resolution routes.** Any fate-of-residual refinement (for example, treating an identical post-reopen re-quote as replacing its pre-IDA twin) is a **replay-heuristic modeling choice** under this OQ's domain. It is not a market rule, because the corpus gives no way to link submissions. It must be recorded as such when adopted.

**Re-open / close when:** the Deliverable (4) figures are recorded (close as "accepted limitation" if the share is small), or a de-duplication heuristic card is filed.

---

## OQ-11 — FOK interaction with ICEwPI price drift: pre-check quantity computation

**What is unknown:** When a FOK order is checked for matchable volume against an ICEwPI iceberg (iceberg with non-zero peak price delta), should the pre-check count the iceberg's **total** remaining quantity as matchable (using only the base price for the crossing test), or should it compute how many individual slices would actually cross at their natural drifted prices?

**Decision dependency:** FOK acceptance/rejection logic; whether price capping is needed during FOK multi-slice fill against ICEwPI icebergs.

**Two interpretations considered:**

*Interpretation A — Total quantity, price capping:* The pre-check counts all remaining ICEwPI quantity as matchable if the base price crosses. During filling, slices whose natural drifted price exceeds the FOK limit are capped at the FOK limit price. This allows the FOK to fill its full requested quantity but later slices trade at the FOK limit instead of the iceberg's natural price, violating the price-determination rule.

*Interpretation B — Slice-by-slice crossing, natural prices:* The pre-check computes the number of slices whose natural drifted price actually crosses the FOK limit. Only that quantity is counted as matchable. If the FOK requests more than the crossing quantity, it is rejected. If accepted, all slices trade at their natural (drifted) iceberg prices. No price capping occurs.

**Resolution:** **Resolved. Interpretation B is correct.**

The source documentation supports Interpretation B through three converging lines of evidence:

1. **ICEwPI slices have distinct limit prices.** Continuous Trading Matching Algorithm §3.3: "Iceberg orders can be entered with a peak price delta. Each new slice will then be entered with a new limit price." Each slice is effectively a new order at a new price. The phrase "may come with a range of limit prices" in §3.3's opening sentence acknowledges this explicitly.

2. **Execution stops when prices no longer cross.** §5.2.1 (Iceberg orders in batch matching): "Iceberg orders with a non-zero peak price delta participate with each slice individually. After the first slice is executed completely the next slice with a new price limit and timestamp will participate until either all slices are fully executed or **no further execution is possible anymore**." The stopping condition "no further execution is possible" means execution halts when a slice's drifted price does not cross the counterparty's limit. While §5.2.1 describes batch matching specifically, it reveals the intended semantic for ICEwPI price drift: each slice is evaluated individually against crossing conditions.

3. **Trade price is determined by the iceberg's limit price.** §5.1.2 (Iceberg orders in regular matching): "the price is always determined by the iceberg order already in the order book." This means each slice trades at the iceberg's (drifted) price, not the incoming order's price. Capping a slice at the FOK's limit would violate this price-determination rule, because the iceberg (passive side) should set the trade price.

4. **FOK requires full-quantity matchability.** §3.5.3: "executed immediately and with its full quantity or, if the order cannot be matched with its entire quantity, deleted." A FOK requesting more volume than the number of slices that actually cross must be rejected — the remaining slices cannot be "matched" because the crossing condition fails.

**Concrete example:** ICEwPI SELL: base=50.00, ppd=+0.50, peak=2 MW, total=10 MW. FOK BUY at limit 51.00 for 8 MW.

Slices at natural prices: 2@50.00, 2@50.50, 2@51.00 = 6 MW crossing. Next slice at 51.50 does NOT cross (51.50 > 51.00). Matchable volume = 6 MW. FOK requests 8 MW > 6 MW matchable. **Result: FOK rejected.**

If the FOK requested 6 MW: accepted, fills 2@50.00 + 2@50.50 + 2@51.00 at natural iceberg prices.

**Impact on current implementation:** The current implementation uses Interpretation A (total quantity with price capping). This must be corrected:
- `_matchable_volume()` must compute slice-by-slice crossing for ICEwPI icebergs, summing only slices whose projected price crosses the FOK limit.
- `_fok_multi_slice_fill()` no longer needs price capping logic for ICEwPI slices, because all slices that execute will naturally cross.
- Standard icebergs (PPD=0) are unaffected — all slices have the same price, so total quantity remains the correct matchable volume.

**Sources:** Continuous_Trading_Matching_Algorithm_Public_Description.md (§3.3, §3.5.3, §5.1.2, §5.2.1); OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (§4.5, lines 10274–10281).

---

## OQ-12 — Anti-self-match / self-trade prevention scope

**What was unknown:** Whether the XBID/OMIE matching engine prevents matches between orders submitted by the same agent, the same bidding unit, both, or neither.

**Decision dependency:** Whether the simulator must implement self-trade prevention (and at what granularity: agent-level, unit-level, or none). This affects order matching logic and potentially RL agent design if agents control multiple units.

**Resolution:** **Provisionally resolved — verdict (A): no STP exists in the XBID/SIDC matching engine.** Confidence: **medium-high.**

The XBID/M7 Shared Order Book matching engine does not implement self-trade prevention (STP). Matching is pure price-time priority with no ownership-based exclusion. The evidence:

1. **Matching algorithm specification is exhaustive and silent on STP.** The Continuous Trading Matching Algorithm Public Description (§5.1 regular matching, §5.2 batch matching) defines matching purely on price-time priority. The batch matching exclusion criteria (§5.2.3) enumerate exactly four disqualifying conditions — already-executed, capacity-constrained, price-incompatible, and price-span-outside — none of which reference order ownership. Source: Continuous_Trading_Matching_Algorithm_Public_Description.md §5.1, §5.2.3.

2. **OMIE bid validation checks are exhaustive and contain no STP.** OMIE Rulebook §45.2.2 lists acceptance checks: contract status, agent authorization, agent registration, bidding unit validity, and collateral. No check prevents an agent's order from entering the book when the same agent already has a resting order on the opposite side. Source: OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md §45.2.2 (lines 5072–5098).

3. **Nord Pool explicitly excludes XBID products from its local STP feature.** Nord Pool's self-trading prevention documentation states: "Self Trade Prevention functionality does not apply for Xbid products" — it only operates on "Local Products offered by Nord Pool." Same-portfolio self-matching is blocked by default for local products, and cross-portfolio restrictions are opt-in. But none of this applies to XBID-matched orders. Source: [Nord Pool Support — Self-trading Prevention](https://support.nordpoolgroup.com/support/solutions/articles/8000061466-self-trading-prevention).

4. **Regulatory framework treats wash trades as a post-trade surveillance concern, not a matching-engine constraint.** REMIT (Regulation 1227/2011) prohibits wash trades (fictitious trades with no change in beneficial interest) as market manipulation. However, ACER's guidance places the compliance obligation on market participants and PPAETs (Persons Professionally Arranging or Executing Transactions), not on the matching engine. ACER notes that "unintended misleading price signals are also potentially breaches of Article 5 of REMIT" but does not mandate exchange-level STP. The EPEX SPOT glossary defines "self-trade" as "a Transaction where an Exchange Member is both on the buy and the sell side for the same product" — this is a classification for index exclusion and regulatory reporting, not a matching rejection trigger. Source: ACER REMIT Guidance v6.0 (2017); [EPEX SPOT Glossary](https://www.epexspot.com/en/glossary); EPEX SPOT Market Indices Description (self-trades excluded from index calculation but not from execution).

**What would change this verdict:** Discovery of an M7 participant manual, OMIE technical bulletin, or EPEX SPOT Exchange Rules provision that documents a configurable STP feature at the XBID/SOB level. The current evidence is that STP, where it exists in European energy exchanges, is a NEMO-local feature that explicitly does not extend to XBID-matched orders.

### Provisional resolution for MVP

The simulator must **not** implement self-trade prevention. An agent's buy order can match its own sell order; two units of the same agent can match against each other. This is faithful to the XBID matching specification.

**Diagnostic metric (recommended):** After each match, compute `self_match_agent_count` (matches where `buy_agent_id == sell_agent_id`) and `self_match_unit_count` (matches where `buy_unidad_id == sell_unidad_id`). These are observability metrics for RL analysis, not rejection criteria. If self-match rates are unexpectedly high during training, this signals a reward-shaping issue in the RL agent, not a matching-engine defect.

**Upgrade path:** If a future OMIE technical bulletin or M7 participant manual reveals an STP mechanism at the XBID level, add a pre-match ownership check in `contract_book.submit_order()` that skips passive orders sharing the relevant identity (agent or unit). The diagnostic metrics already provide the data needed to assess impact.

**Sources consulted:**
- Continuous_Trading_Matching_Algorithm_Public_Description.md (§5.1, §5.2, §5.2.3 exclusion criteria)
- OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (§45.2.2 bid validation, §46.1 matching process)
- Single_Intraday_Coupling_XBID_Information_Package.md
- Commission_Regulation_EU_2015_1222_CACM.md
- Third_Amendment_to_the_Intraday_Operations_Agreement_IDOA.md
- ACER_Decision_11_2024_Annex_I_Algorithm_Methodology.md
- Nord Pool Support: Self-trading Prevention (https://support.nordpoolgroup.com/support/solutions/articles/8000061466-self-trading-prevention)
- ACER REMIT Guidance v6.0 (2017) — wash trade guidance
- EPEX SPOT Glossary — self-trade definition
- EPEX SPOT Market Indices Description — self-trade exclusion from indices

---

## OQ-13 — Numeric magnitude bound on ICEwPI PPD

**What is unknown:** Whether any OMIE/SIDC rule, technical circular, or M7/XBID operational directive imposes a numeric maximum magnitude on the peak price delta (`PPD`) of ICEwPI iceberg orders, beyond the sign rule (buy: negative; sell: positive) and the implicit constraint that every slice price must remain within the SIDC harmonized price range (-9,999 to +9,999 EUR/MWh).

**Decision dependency:** Validation envelopes for ICEwPI orders in the simulator; test-envelope derivation (must tests rely on an empirical dataset-specific range, or on a rule-level constant?).

**Where to look:**
- OMIE operational directives not currently in the corpus (e.g. post-2024 OMIE technical instructions beyond Instruction 1/2024).
- M7/XBID participant manuals (not publicly mirrored in the corpus).
- EPEX SPOT / OMIE LTS configuration bulletins.

**Current state (2026-04-22):** Exhaustive search of the `docs/omiecid/corpus/` corpus (OMIE Rulebook 2025, OMIE Details of the Intraday Market's Operation, XBID Information Package, Continuous Trading Matching Algorithm Public Description, CACM Regulation, ACER Decisions 11/2024, 13/2024, 03/2025, 11/2025, CNMC Resolution DCOOR/DE/009/24, OMIE Instruction 1/2024, OMIE Directive 1/2022, IDOA, ANIDOA) for "price increment", "peak price delta", "incremental de precio", and magnitude-bearing phrases returns only:
- the sign rule (buy negative, sell positive);
- the definitional statement that each new slice price equals previous slice price plus PPD;
- no numeric magnitude bound.

Empirical OMIE data contradicts the previously documented "approximately -1.55 to +1.45 EUR/MWh" range: delivery date 2026-01-12 contract 81 shows `PPD = -3.00 EUR/MWh` on ICEwPI BUY rows (agent `ISMSL`, unit `ISMSC01`). This confirms that the prior range was a sample statistic, not a rule.

**Provisional resolution:** No rule-level magnitude bound exists in the available corpus. The hard rules are (a) sign, and (b) derived: every slice price must stay within SIDC price limits. Any numeric magnitude envelope used downstream (for validation, tests, or action-space design) must be treated as an empirical choice tied to a specific dataset, not as a market rule. See §13.6 of `market-spec.md` for the authoritative statement.

**What would change this verdict:** Discovery of an OMIE technical bulletin, M7 participant manual section, or EPEX SPOT exchange rule provision specifying a numeric `PPD` magnitude limit.

**Sources consulted:** entire `docs/omiecid/corpus/` corpus (negative search); empirical counter-example from `data/rawOMIE/` 2026-01-12 contract 81.

---

## OQ-14 — Pre-GOT order submission behavior

**What is unknown:** OMIE's public regulatory documentation is silent on whether market participants may submit orders *before* the Gate Opening Time (GOT) for a contract and, if so, how such submissions are handled. `docs/omie/market-spec.md` §5.4 addresses post-GCT handling symmetrically, but the pre-GOT case is not covered. Specifically, it is unspecified whether pre-GOT submissions are:
(a) rejected outright by the matching platform;
(b) buffered by the local trading system (LTS) and queued for insertion at GOT;
(c) accepted into a pre-opening state with special priority treatment; or
(d) handled some other way.

**Decision dependency:**
- Loader semantics: signed session-relative timestamps (negative values for pre-GOT submissions) are observed in real OMIE fixtures, implying submissions with `submit_time < GOT` exist in the published data.
- Priority tiebreak: any matching rule that uses a pre-GOT timestamp as a tiebreaker (price-time priority) is under-specified until this is resolved.
- The `% 86400` wrap currently in the loader operates correctly on the 2026-01-12 contract 81 oracle fixture (see P2-HOTFIX-ioc-fill-detection-FOLLOWUP trigger), but its behavior on genuinely negative timestamps is a latent risk.

**Where to look:**
- OMIE Market Rules sections on order submission and validation (beyond §45.2.2 checks, which cover post-GOT bid validation).
- XBID / M7 participant manuals (not publicly mirrored in the corpus) regarding LTS buffering.
- OMIE technical circulars post-Instruction 1/2024.

**Status:** **Open, tracked.** Does not block any Phase 2 story. The primary P2-HOTFIX-ioc-fill-detection story landed with this question unresolved and the `% 86400` timestamp wrap in place. Revisit if (a) a contract fixture exhibits divergent behavior attributable to negative-timestamp handling, or (b) documentation surfaces clarifying pre-GOT semantics.

**Sources consulted:** OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (§45.2.1–§45.2.2, §46.1); OMIE_Instruction_1_2024_IDA_Go-Live_Operations.md; Single_Intraday_Coupling_XBID_Information_Package.md; Continuous_Trading_Matching_Algorithm_Public_Description.md. Negative search for pre-GOT / pre-opening / buffered submission semantics.

**Annotation (2026-09-30, `OMIE-TAPE-CONTRACT-DATE-1`; OQ-27).**
- The "negative session-relative timestamps" listed under *Decision dependency* are **not pre-GOT submissions**. They are product-D rows sent on D−1 after the 15:20 GOT, which the old loader measured from D 00:00. The cause is the file-date error corrected in market-spec v2.10 §13 / §13.4: files are split by session day.
- Data check: `orders_20250624.1` has **0** rows of product 06-25 sent before 15:20 on 06-24 (and 5,623 in 15:20–15:29). No pre-GOT submission is observed.
- So this OQ's data trigger is dissolved. The regulatory question (how a genuine pre-GOT submission is handled) stays open but does not bind the loader.
- The `% 86400` wrap named here is removed by `OMIE-TAPE-CONTRACT-DATE-1`.

---

## OQ-15 — Ring-buffer occupancy invariant for `OMIEBookState.lob.trades`

**Note on category:** This is primarily an **engineering / architectural invariant** rather than a regulatory or market-domain ambiguity. It is recorded here as explicitly directed by the P2-HOTFIX-ioc-fill-detection-FOLLOWUP story card so that future readers encounter it alongside related market-semantics items. Domain analysts reviewing this file should treat this entry as a tracking pointer to Phase 3 engineering work, not as an unresolved market rule.

**What is unknown (engineering invariant):** Once the trades ring buffer saturates (`NTRADE_CAP = 100` in `gymnax_exchange/jaxob/jaxob_constants.py`), the `OMIEBookState.lob.trades` array stays permanently full for the remainder of the replay, losing its intended rolling-window semantics. The root cause lives in `gymnax_exchange/jaxob/JaxOrderBookArrays.py` (`add_trade`), which uses `jnp.where(trades == -1, size=1, fill_value=-1)` to locate the next free slot. On a saturated buffer no slot equals `-1`, so every subsequent `add_trade` call writes to index `-1` (Python-style last-slot), silently clobbering earlier fills of the same `submit_order` invocation.

**Consequence for Phase 2 (observed):**
- The whole-row diff introduced by `P2-HOTFIX-ioc-fill-detection` tolerates saturation for single-step fill detection (sufficient for IOC fill attribution at the primary call site).
- Under multi-level crossings on a saturated buffer, the diff-based fill count collapses to `num_fills == 1`. The locked behavioral test `test_multi_level_new_fills_detected_on_saturated_buffer` asserts this limit.
- Any observer that reads `lob.trades` directly as a FIFO "recent trades" view will see a non-FIFO, permanently-full buffer whose contents no longer represent a true rolling window.

**Decision dependency:**
- Phase 3 hardening: a proper write-pointer (the deferred `LobState.n_trades` field referenced under "Rejected alternative" in the P2-HOTFIX-ioc-fill-detection story Description) would restore rolling-window semantics and lift the single-step multi-level cap.
- Observer / telemetry design: any downstream component (RL observation builder, logging, replay-divergence metrics) that wants true recent-trades semantics must not rely on `lob.trades` under saturation until the write-pointer refactor lands.

**Status:** **Open, tracked as a Phase 3 hardening item.** Cross-reference the deferred `LobState.n_trades` write-pointer work mentioned in the P2-HOTFIX-ioc-fill-detection story card's Description ("Rejected alternative"). Not a market-rule ambiguity; recorded here per explicit FOLLOWUP directive.

**Sources consulted:** `gymnax_exchange/jaxob/jaxob_constants.py` (`NTRADE_CAP`); `gymnax_exchange/jaxob/JaxOrderBookArrays.py` (`add_trade`); `docs/omie/agile-plan.md` Story P2-HOTFIX-ioc-fill-detection and its FOLLOWUP card; locked test `test_multi_level_new_fills_detected_on_saturated_buffer`.

---

## OQ-16 — Method A XLA-safety invariant for tick-encoding in the traced validator

**Note on category:** This is primarily an **engineering / architectural invariant** rather than a regulatory or market-domain ambiguity. It is recorded here as directed by the P2-L1-HOTFIX-tick-precision architect adjudication (Option C, §3.12) so that future readers encounter the XLA-safety envelope alongside the related `OQ-06` tick-size rule.

**What is known (engineering invariant):** Under Option C Method A, the traced validator encodes prices and quantities from `float` to `int32` tick indices via:

```python
jnp.round(jnp.float32(p) * jnp.float32(round(1.0 / tick))).astype(jnp.int32)
```

This produces the exact tick index for on-tick inputs at OMIE defaults (`price_tick = 0.01 EUR/MWh`, `qty_tick = 0.1 MW`).

**Verified:** the full 1,999,801-case price grid (−9,999.00 to +9,999.00 in 0.01 steps) and the 9,990-case qty grid (0.1 to 999.0 in 0.1 steps) were exercised under `jit+vmap`; **0 mismatches on both CPU and GPU (CUDA) backends**. The GPU verification (performed during developer dispatch at commit `b8e8753`) closed the architect's highest-risk residual concern — that CUDA's XLA lowering might take an approximate-reciprocal path that CPU does not.

**Invariant (required for correctness):**
- `jnp.float32(round(1.0 / tick))` must be exactly representable as a float32. Holds for `tick = 10^-k` with `k ≤ ~7` (float32 has a 24-bit mantissa; `10^7 = 10_000_000 < 2^24 = 16_777_216`).
- `p * scale` must stay within float32's integer-exact range `±2^24 = ±16_777_216`. OMIE defaults give max `|p * scale| = 9999.00 * 100 = 999,900`, a margin of **16.8×**.

**When this envelope would need re-verification:** a future OMIE product config with `tick = 1e-4` or a price range widened to `±2 × 10^5` would push `|p * scale|` closer to the float32 integer-exact boundary. Such a change would require:
1. Re-measuring the full on-tick price grid under `jit+vmap` on both CPU and GPU backends.
2. Possibly adding a startup guard in `gymnax_exchange/omie/omie_config.py` that asserts the `|p_max| * (1 / tick) < 2^24` invariant at config-load time.

Neither is implemented in the P2-L1-HOTFIX-tick-precision hotfix; both are deferred until a product-config change triggers the need.

**Status:** **Non-blocking; architectural note for future product configs.** Tracked here so that a future developer altering OMIE tick-size or price-range bounds does not silently cross the envelope.

**Sources consulted:**
- `docs/omie/sprint-status.md` §`P2-L1-HOTFIX-tick-precision — Second fix-mechanism re-adjudication (Option C)` §3 (architect Method A specification and grid-verification record).
- Commit `b8e8753` — production Method A apply in `gymnax_exchange/omie/order_validator.py`, including the GPU re-verification noted in the commit body.
- IEEE 754 binary32 specification (mantissa width = 24 bits including implicit leading 1).

---

## OQ-17 — Full `OMIEOrder` int32 migration (deferred to P2-H2 or Phase 3 hardening)

**Note on category:** Engineering / architectural debt item recorded at the direction of the P2-L1-HOTFIX-tick-precision architect adjudication (Option C, §3.12). Not a market-rule ambiguity.

**What is deferred:** The P2-L1-HOTFIX-tick-precision hotfix takes the *int32-internal-to-validator* slice only. The dataclass `OMIEOrder` in `gymnax_exchange/omie/omie_types.py` retains `price: float`, `quantity: float`, `peak_qty: float` annotations; the validator encodes to `int32` on entry via Method A (OQ-16), and the rest of the traced path continues to consume float fields. This keeps the hotfix surface limited to `gymnax_exchange/omie/order_validator.py` as architect-scoped.

**Canonical end state (not implemented in this hotfix):** full `int32` field typing for tick-quantised quantities on `OMIEOrder`:

```python
@dataclass
class OMIEOrder:
    price: jnp.int32      # encoded tick index (was: float EUR/MWh)
    quantity: jnp.int32   # encoded tick index (was: float MW)
    peak_qty: jnp.int32   # encoded tick index (was: float MW)
    ppd: jnp.int32        # encoded tick index (was: float EUR/MWh)
    ...
```

Moving to this end state removes all `float → int` cast boundaries from the traced path, eliminates any residual XLA-lowering concern about `float * reciprocal` idioms, and collapses the OQ-16 invariant window to a compile-time check at config load.

**Downstream cascade (scope of the deferred migration):**
- `gymnax_exchange/omie/omie_data_loader.py::row_to_order` — currently calls `decode_price` / `decode_qty` to produce float fields; would instead produce int32 tick indices directly from the pre-encoded arrays.
- `gymnax_exchange/omie/contract_book.py::submit_order` — many call sites to `_omie_price_to_jaxob(order.price, cfg)`; cast boundary moves from entry to jaxob-interop edge only.
- `gymnax_exchange/omie/iceberg_manager.py::register` — consumes `order.peak_qty`, `order.price`, `order.ppd`; int32 propagation is mechanical but cascades through tested arithmetic.
- **~40 test files** that construct `OMIEOrder` with float literals (e.g. `OMIEOrder(price=129.68, quantity=2.5, ...)`) — either replaced with encoded int32 constants, or an eager-shim helper (`_make_order_from_floats`) added to the test utilities.

**Scope owner:** **P2-H2** (`contract_book.submit_order` JIT-safe) is the natural gate for this migration — it is the largest call-site cluster and it is already scoped to harden the order-submission traced path. Alternatively the migration can land as a dedicated Phase 3 hardening story if P2-H2's scope becomes unwieldy.

**Status:** **Non-blocking; known debt.** Tracked here so that a future P2-H2 architect or Phase 3 hardening planner finds the cascade enumerated without having to re-derive it from the P2-L1-HOTFIX-tick-precision adjudication.

**Sources consulted:**
- `docs/omie/sprint-status.md` §`P2-L1-HOTFIX-tick-precision — Second fix-mechanism re-adjudication (Option C)` §3.12 (architect guidance to file OQ-17 at this scope).
- `gymnax_exchange/omie/omie_types.py` (current float field typing).
- `gymnax_exchange/omie/omie_data_loader.py`, `gymnax_exchange/omie/contract_book.py`, `gymnax_exchange/omie/iceberg_manager.py` (consumer-site inventory).
- `docs/omie/agile-plan.md` Story P2-H2 (`contract_book.submit_order` JIT-safe) — proposed scope owner.

---

## OQ-18 — Phase 3 reopen of `jax.jit(compute_agent_pnl)` pending JAX `jax-ml/jax#5982`

**Filed:** 2026-04-24 (P2-L2 — `settlement` JIT-safe, Option 3a adjudication).
**Scope:** `gymnax_exchange/omie/settlement.py::compute_agent_pnl`.
**Status:** **Non-blocking; deferred.** Tracked for Phase 3.

**Context.** The P2-L2 story card originally required `compute_agent_pnl`
to be `jax.jit`-compilable. During implementation the in-body
`with jax.experimental.enable_x64():` wrapper was found to be a no-op under
`jit` tracing — the x64 flag is read at trace time, so `int64` casts inside
the traced body silently collapse to `int32`, losing the two-term
accumulator's overflow guarantees at the OMIE envelope (per-row
`Q_tick × p_tick_offset ≈ 2×10¹⁰`, outside int32 range ~2.15×10⁹).

The known upstream JAX issue is **`jax-ml/jax#5982`** — the `enable_x64`
context manager does not propagate its dtype-promotion into `jit`-traced
functions. Until that issue is addressed upstream, there is no clean path
to a jit-safe `compute_agent_pnl` without either:

  1. flipping the **global** `jax.config.update("jax_enable_x64", True)`
     flag (architect-rejected — would break the P2-L1 Method A XLA-safety
     invariant that the validator jaxpr is strictly int32), or
  2. requiring every call site to be wrapped in `with enable_x64():` and
     still accepting that `jit` won't lift the promotion through the trace.

**Option 3a adjudication (migration-plan §Phase 2 — Settlement API pin
§jit-contract adjudication, v5.6; agile-plan §P2-L2 v5.8 item (g)).** The
public `jax.jit` contract on `compute_agent_pnl` is **dropped** for Phase 2.
The function remains JAX-traceable and supports `jax.vmap` when the caller
supplies an outer `with jax.experimental.enable_x64():` context. The body
opens its own scoped `enable_x64` for eager callers (a no-op under tracing,
active on the eager path).

**Reopen criteria (Phase 3).** Reconsider `jax.jit(compute_agent_pnl)` when
**any** of the following holds:

  - JAX issue `jax-ml/jax#5982` is resolved upstream such that
    `with enable_x64():` propagates into `jit` traces, or an equivalent
    per-call dtype-promotion hook lands.
  - The OMIE tick grid is rearchitected so that `Q_tick × p_tick_offset`
    fits inside int32 (e.g. coarser ticks or a bounded envelope subset) —
    making the int64 accumulator unnecessary.
  - The PPO training loop's throughput profile shows the Python-eager
    settlement call to be a measured bottleneck warranting a dedicated
    jit-specialized settlement path (benchmarked, not assumed).

**Sources consulted:**
- `docs/omie/migration-plan.md` §Phase 2 — Settlement API pin §jit-contract
  adjudication (v5.6).
- `docs/omie/agile-plan.md` §P2-L2 v5.8 item (g) (Option 3a adjudication).
- `docs/omie/sprint-status.md` §`P2-L2 — jit vs int64 adjudication`.
- JAX upstream issue `jax-ml/jax#5982`.

---

## OQ-19 — Operational price range of SIDC continuous trades (distinct from the ±9,999 hard envelope)

**Filed:** 2026-05-27 (raised during architecture planning for the int32 price-encoding change; the int32 budget admits a price excursion of only ±2,002 EUR/MWh from the pivot, which does not contain the ±9,999 hard envelope).

**What is unknown:** Whether the market-spec or the OMIE source corpus defines or implies any *operational* price bound for SIDC continuous **executed/traded** prices that is narrower than the ±9,999 EUR/MWh hard bid-rejection envelope (§2.3, OQ-01) — e.g. a price collar, a circuit-breaker band, a market-maker quoting obligation band, or a documented typical-traded-range.

**Findings (negative on all three counts):**

1. **No operational executed-price band exists at the rule level.** The only sub-envelope construct in the corpus is the Annex 3 **price *notification* threshold** (OMIE Rulebook §45.2.1, §45.2.2; Annex 3 §4, lines 10420–10438). This governs an **informative warning at bid insertion time only**: §45.2.1 (lines 5052–5060) states the threshold check "shall be of an informative nature for market participants" and is issued "without prejudice to any other validation." It does **not** reject orders and does **not** bound executed trade prices — a participant may knowingly submit and trade past the threshold. It is therefore not a collar, not a circuit breaker, and not a bound on the traded-price distribution.

2. **The "±150 EUR/MWh notification band attributed to §2.4" is partly a mis-citation.** market-spec §2.4 records the MIBEL notification thresholds as **approximately −150 / +1,500 EUR/MWh** (asymmetric; the upper bound is +1,500, not +150). Even these numbers are **not present in the corpus Annex 3 text** — Annex 3 §4 defers all numeric values to "the agent website" (lines 10433–10435). The −150 / +1,500 figures in §2.4 and OQ-01 are web-research approximations flagged as dynamic ("may be modified by OMIE directive"). They govern bid-insertion alerting, not executed prices, and cannot be relied on as a hard operational trade-price bound.

3. **No basis to assert executed SIDC prices stay within ±2,002 EUR/MWh of any pivot.** Scarcity pricing, negative prices, and price spikes are genuine SIDC regimes; the harmonized SIDC methodology sets the ±9,999 limits precisely to admit such excursions, and Annex 3 confirms the limits are dynamic. The corpus contains no statistical characterization of the executed-price distribution. The only empirical price reference is §13.3 (June 2025 sample) and §13 fixtures, neither of which the spec characterizes by price range. An empirical operational bound can only be pinned per-dataset (cf. the PPD precedent in §13.6 / OQ-13), never as a market rule.

**Decision dependency:** Whether a dynamic-range assert guarding the *operational* SIDC price range (rather than the full ±9,999 envelope) is domain-defensible, and at what bound. **Domain ruling: not defensible as a market-rule constant.** Any narrower runtime bound must be (a) declared an engineering/empirical envelope tied to the specific replay corpus under test, (b) measured from that corpus, and (c) accompanied by a fallback for out-of-band prices — never cited as an OMIE rule, and never set to the notification-threshold numbers as if they bounded trades.

**What would change this verdict:** Discovery of an OMIE technical circular, M7/XBID participant manual, or EPEX SPOT exchange-rule provision that documents a price collar or executed-price band for continuous trading; or a spec-blessed empirical characterization of the replay corpus price distribution.

**Sources consulted:** OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (§45.2.1 lines 5044–5071, §45.2.2 lines 5050–5060, Annex 3 §3–§4 lines 10404–10438 — all numeric values deferred to agent website); market-spec.md §2.3, §2.4, §13.3; OQ-01; OQ-13 (PPD empirical-vs-rule precedent).

---

## OQ-20 — Simulator-internal iceberg-meta side-channel cleanliness vs. Rule 4

**Filed:** 2026-05-28 (raised during P2-PERF-12 sprint — orchestrator-routed adjudication request after the rewritten cid 81 quiescence test detected nine slots in `peak_consumed=True` at submit_order #336 against both `main` and the in-flight WIP optimization).

**Note on category:** This is primarily an **engineering / architectural invariant** rather than a regulatory or market-domain ambiguity. It is recorded here because the P2-PERF-12 sprint requested a domain ruling on whether the spec mandates this invariant, and the ruling is "no — but the test is detecting a real simulator-internal inconsistency worth tracking." Domain analysts reviewing this file should treat this entry as a tracking pointer to engineering work, not as an unresolved market rule. See `market-spec.md` §4.2.1 Rule 4 *Scope clarification* sub-bullet for the authoritative scope statement.

**What was asked.** Does market-spec §4.2.1 Rule 4 require that "after every `submit_order` call, no active iceberg slot is in a `peak_consumed=True` / absent-visible-slice state" as an invariant on full-session real-data replays, or only that the intra-call cascade chase converges within the call?

**Ruling.** **(B) Intra-call only on the resting order book.** Rule 4 is a within-call (within-process) fixed-point requirement on the resting **`OrderBook`** (the LOB). It does not impose a post-call invariant on simulator-internal side-channel state (`iceberg_meta`). An active iceberg slot whose `order_id` is absent from the LOB at rest between aggressors — i.e. the visible peak was fully consumed by a prior aggressor's matching process and the next slice has not yet been activated — is a **permitted between-aggressor state** under the matching algorithm. The spec contemplates exactly this: Continuous Trading Matching Algorithm §3.3 (lines 265–275) describes slice activation as occurring "when the quantity of the visible peak is reduced to zero" — i.e. the next slice is activated when subsequent matching activity engages the iceberg, not eagerly. There is no spec text requiring continuous visible presence of an iceberg's peak between aggressor arrivals.

**Status of the observed signature on real cid 81 (full 2937-order session, submit_order #336, 9 slots with `visible_qty == -1`).**

1. **Not a Rule 4 violation.** The slots' OIDs are absent from the resting LOB (not crossed; the LOB is internally consistent). Trade sequence and trade timestamps emitted during each preceding matching process were emitted *during that process*, not deferred. No "across submit cycle" cascade-drain is occurring. The resting book between aggressors is uncrossed, which is the binding observable Rule 4 protects.

2. **Distinct from the P2-PERF-8 order-20→21 case.** The P2-PERF-8 anchor exposed a *crossed* book left between aggressors — a regenerated visible slice resting in the book at a price that crossed an opposite-side resting order, with the cascade trade emitted under the *next* aggressor's timestamp. That was a Rule 4 violation. The P2-PERF-12 #336 signature is different: the slot's OID is absent from the book entirely (sentinel `-1` from `get_order`), not resting in a crossed state. The iceberg is at rest with no visible peak; the simulator's `iceberg_meta` row points to an OID that no longer indexes a live LOB entry.

3. **What it likely represents (engineering, not spec).** A simulator-side-channel state where: (i) a prior aggressor's matching process consumed the full visible peak of iceberg `J`, (ii) the regen helper either did not register the next slice (e.g. because `remaining_hidden == 0` and the iceberg should be cleaned up), or registered it but in a way that left the meta row pointing at the now-defunct previous OID. The meta row is **stale** in the sense that the OID it stores does not reference a live LOB order; the iceberg is in a between-peaks rest state. This is a simulator implementation detail of `IcebergManager`, not a market-rule concern.

**Why the test is still load-bearing as an engineering invariant.** A stale `iceberg_meta` row that points to an absent OID is a real implementation hazard for the current simulator code:

- `IcebergManager.matchable_hidden_volume` and similar consumers walk `iceberg_meta[:n_active]` and treat each slot as a live iceberg. A slot whose OID is absent from the LOB but whose `remaining_hidden > 0` would either over-report matchable hidden volume (if the meta row's quantity fields are still valid and the next slice has yet to activate) or under-report (if the slot represents a fully-consumed iceberg that was not cleaned up). Both are correctness-affecting silent bugs at the FOK pre-check path (cf. the P2-HOTFIX-iceberg-cap-overflow precedent for FOK pre-check correctness regressions).
- The `_match_type_label` / demo-rendering paths consult these slots; stale slots produce mis-labeled trade rows.
- The cascade-quiescence outer fixed-point in `regenerate_slices` re-arms on `slice_re_consumed = need_regen & (post_row[2] <= 0)`. If a stale slot persists across aggressors with `need_regen=True` (because `peak_consumed` evaluates True on the absent OID via the sentinel), the next regen call would attempt to re-dispatch a slice for an iceberg that has either no remaining hidden quantity (cleanup miss) or whose next-slice was already activated by a different code path (double-activation).

The simulator design choice that motivates a "no stale meta row" engineering invariant is therefore a defensible one — but it is **not** mandated by Rule 4 and should not be advertised as a market-spec requirement.

**Decision dependency:**
- The P2-PERF-12 quiescence-after-every-submit test on real cid 81 is currently over-strict if interpreted as a Rule 4 conformance probe. It is correctly framed only as an engineering invariant probe on `iceberg_manager`'s side-channel cleanliness.
- The implementer should choose whether the simulator carries the engineering invariant "no active slot whose OID is absent from the LOB persists between aggressors" as an internal contract of `IcebergManager`. If yes: the test stays, but is relabeled as an engineering invariant (not Rule 4), and the failure on `main` is filed as a hotfix story (the meta row should be cleaned up at consume-time or at register-of-next-slice-time). If no: the test must be relaxed to assert Rule 4's binding observable — no crossed resting book between aggressors, and intra-call cascade convergence — and the stale-meta state is documented as a known benign simulator condition with cleanup deferred to lazy activation on the next touching aggressor.

**Recommendation for the P2-PERF-12 orchestrator.** Adopt (B) interpretation. The test contract should be revised to one of:

(i) **Relabel as engineering invariant**, keep the assertion, and file a hotfix story to make `IcebergManager` enforce "no active slot whose OID is absent from the LOB persists across `submit_order` boundaries" by cleaning up the meta row when the visible peak is consumed and no new slice is registered. This is the implementer's call, not the domain analyst's — it depends on whether `matchable_hidden_volume` / `_match_type_label` correctness is acceptable under the laxer invariant.

(ii) **Relax the assertion** to "no crossed resting book between aggressors" plus "intra-call cascade convergence on the truncated cid 81 slice and on the deep-cascade synthetic" (both already covered by the P2-PERF-8 anchor's `TestP2Perf8CascadeQuiescence` deep-cascade test and its order-20→21 truncated-real-data test). The full-session real-data quiescence-after-every-submit test would be removed or rewritten to check the laxer invariant.

Either disposition is consistent with the spec. The orchestrator should not file a "Rule 4 hotfix story" against `main` on the basis of the absent-OID signature — there is no Rule 4 violation, and `main` is spec-correct on the binding observable (intra-call resolution + uncrossed resting book between aggressors).

**Distinguishing the two failure modes at test-time:**
- A **Rule 4 violation** (P2-PERF-8 shape) shows up as `get_order(...)[2] == 0` on a slot whose OID *is* present in the LOB but with zero visible quantity (peak fully consumed and not yet refreshed — but the OID still indexes a live row), AND the LOB contains a crossed pair between the regenerated slice and resting opposite-side liquidity. The cross is the binding observable.
- The **OQ-20 absent-OID shape** (P2-PERF-12 #336) shows up as `get_order(...)[2] == -1` (the `NEGATIVE_RETURN_ID = -99` sentinel propagating through column 2 — actually `-99` per `iceberg_manager.py:367-369`; the test prints `-1` after a `<=0` collapse but the underlying value is `-99`), meaning the OID is *not present anywhere* in the resting side array. The LOB is internally uncrossed. This is the simulator-internal stale-meta state, not a Rule 4 cross.

A test attempting to lock Rule 4 specifically should assert "the LOB is uncrossed after every `submit_order`" (`best_bid_price < best_ask_price` or a no-cross direct probe on the resting book), not the side-channel `peak_consumed` derivation.

**Status:** **Open, tracked.** Not a market-rule ambiguity. The simulator-internal cleanup strategy is an engineering decision owned by the implementer / architect of `IcebergManager`; the P2-PERF-12 test contract revision and any follow-up hotfix is owned by the orchestrator. Mark this entry as resolved once the engineering decision (i) vs. (ii) is committed.

**Sources consulted:**
- `docs/omie/market-spec.md` §4.2.1 Rule 4 (updated 2026-05-28 with *Scope clarification* sub-bullet under the same dispatch as this OQ).
- `docs/omie/sprint-status.md` §"P2-PERF-8 — Domain Ruling: iceberg regen-cascade timing (2026-05-26)" and §"P2-PERF-8 — Cascade Adjudication (2026-05-26)" §3 (prior ruling scope).
- `docs/omie/sprint-status.md` §"P2-PERF-12 — Tests Rewritten (test-agent, slow-tier runtime fix, 2026-05-28)" §"Verification" (the cid 81 #336 failure trace that triggered this OQ).
- `docs/omiecid/corpus/Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/Continuous_Trading_Matching_Algorithm_Public_Description.md` §3.3 (lines 265–275: slice activation when previous slice quantity reaches zero — not a continuous-presence requirement); §4.1 (lines 354–360: order-book updates conclude resulting trades "after each order book update" — within-process, on the resting book); §5.1.2 (lines 474–478); §5.1.3 (matching runs to fixed point while incoming remains executable); §5.2.1 (batch-matching slice semantics, citing same intra-process resolution).

**CORRECTION NOTE (2026-10-07, `OMIE-SPEC-CERT-AMEND-1`; additive, the entry above is kept as written).** The *Ruling* paragraph says that "the next slice is activated when subsequent matching activity engages the iceberg, not eagerly". That premise is **corrected: activation is eager.**
- **(a) Source.** Continuous Trading Matching Algorithm §3.3 (`Continuous_Trading_Matching_Algorithm_Public_Description.md:265-274`): "If the shown quantity before the trade was less than or equal to the subtracted amount, a new slice of order quantity is made available in the market sized at the peak size of the iceberg order or at the remaining quantity, if this is smaller than the peak size." The new slice is made available by the trade that exhausts the shown quantity, inside that same matching process. It does not wait for a later aggressor. OMIE Rules Annex §4.4 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10253-10255`) agrees: "When the viewable reduced part of the bid is fully matched, a new bid will be automatically created in the Bid Book". Matching Algorithm §4.1 lists the "activation of a new iceberg slice" as an order-book update.
- **(b) Consequence for the market model.** Between matching processes, an iceberg with remaining quantity always shows a slice in the real order book. An iceberg with hidden quantity left but no visible slice is a simulator state, not a market state.
- **(c) What is not reopened.** The ruling's conclusion stands: a stale side-channel slot whose OID is absent from the LOB is an engineering matter, not a §4.2.1 Rule 4 violation. The (i)/(ii) dispositions and OQ-23 are unaffected.
- **(d) Open for the architect, not a domain question.** Does the engine's lazy state ever cause an observable difference against eager activation? Two candidates are a resting-book snapshot that lacks the slice, and a later slice timestamp that changes time priority at the slice's price. This note does not settle that, and the answer must not be taken from engine behaviour.
- **Spec:** market-spec v2.14 §4.2.1 Rule 4, *Slice activation timing* bullet. This resolves certification finding F6 (C15) in favour of the eager reading.

**ANSWER TO CORRECTION-NOTE ITEM (d) (2026-10-08, `OMIE-SPEC-STAMP-AMEND-1`; additive, the entries above are kept as written; architect countersign: given 2026-10-08). D4b was not caused by lazy activation.**
- **(1) What D4b was.** In the three C15 r2 class-B draws (certification-log entry 37; fixture `notebooks/certification/fixtures/C15_r2_slice_stamp_seed100000254.npz`) the slice was activated eagerly, inside the exhausting process, but stamped from the book counter, which runs ahead of the tape clock. That changed time priority at its price: a later arrival at the same price was served first. The rules broken are market-spec v2.14 §5.2 (new timestamp on "activation of a new iceberg slice"; "single global queue (unique, sequential)") and §6.1 (ties broken by earliest timestamp), read with Rule 4's eager activation. `OMIE-SLICE-STAMP-TAPE-SEC-1` fixes it. The queue place the slice must take is text (3b), queued in the OQ-32 scope correction.
- **(2) Lazy activation.** No observable difference caused by lazy activation was found in r2.
- **(3) What remains to check.** The "slice shown" outcome is checked by F6 at C15's next K2 (certification-log entry 37 verdict: "tighten F6 to 'slice shown'"). "No visible row" between matching processes is not a market outcome: v2.14 §4.2.1 Rule 4 (`market-spec.md:285`), "between matching processes, an iceberg with remaining quantity always shows a slice"; Continuous Trading Matching Algorithm §3.3 (`:265-274`); OMIE Rules Annex §4.4 (`:10253-10255`).
- **Status of item (d):** answered, subject to the architect's countersignature. The ruling's engineering conclusions, the (i)/(ii) dispositions and OQ-23 are not reopened.
- **Architect countersign (2026-10-08):** Concur. The engine creates the new slice inside the submit whose matching exhausted the previous one (`regenerate_slices` runs within that submit), so activation is already eager; the observable defect is only the stamp, a counter value ahead of the tape clock, which is exactly E2 of `OMIE-SLICE-STAMP-TAPE-SEC-1` and moves no price or volume. This answer makes no engine claim beyond that card, and the "slice shown" check stays with F6 at C15's next K2.

---

## OQ-21 — No corpus rule for per-contract liquidity / traded-volume ranking or near-GCT concentration

**Filed:** 2026-07-03 (raised during the `P3-PERF-TAPE-1` domain ruling — the shorter-liquid-hour-tape story asked the domain analyst to rule which contract/window is a defensible "liquid peak-hour" tape for the v1 same-set-overfit RL experiment).

**What is unknown / missing.** The domain-analyst corpus (`docs/omiecid/corpus/`) contains **no authoritative rule** that ranks OMIE continuous-intraday contracts by expected order flow or traded volume, and **no rule** that trading activity (liquidity) concentrates near gate closure (GCT) or in any particular band of a contract's GOT→GCT window. An exhaustive search for "liquid*", "volume", "peak hour / peak-hour", "order flow", "near clos*", "approaching delivery", and related terms returns only *liquidación* (settlement) matches and generic descriptions of matching/gate mechanics — never a volume-by-hour or intensity-by-time-to-delivery ranking. The OMIE Rulebook, XBID Continuous Trading Matching Algorithm description, XBID Information Package, ACER Decisions 03/2025 & 11/2024, OMIE Instruction 1/2024, and the CID educational summary are all silent on this.

**Decision dependency.** Any experiment or config decision that needs to pick a "liquid" or "peak-hour" contract/window — e.g. `P3-PERF-TAPE-1`'s shorter-tape selection, and future Stage-2+ held-out-day selection that wants representative-liquidity contracts — cannot appeal to a market-spec liquidity definition. The informal phrase "liquid peak-hour tape" in `outputs/cid_rl/runs/stage0_20260703_585f775/report.md` §"3 candidate optimizations" #3 has no spec backing.

**Spec-grounded proxy used in lieu of a rule (recorded in market-spec §14, DA-TAPE-2026-07).** The one deterministic, spec-derivable quantity that *bounds* per-contract cumulative order flow is the **GOT→GCT trading-window length**: all 96 contracts of a delivery day open at a common GOT (~15:20 CET D−1) and each closes at GCT = delivery-start − 1h, so window length is a strictly increasing function of the day-D delivery time, i.e. of `contract_id` (`contract_id = start_hour*4 + start_minute//15`). Shortest-window = earliest-delivery = smallest `contract_id`. This is a bound on *capacity to accumulate flow*, not a measured liquidity ranking, and it does not by itself guarantee non-degeneracy — hence the independent non-degeneracy bar (≥ 1 historical trade in `trades_YYYYMMDD.1` for the contract + both order sides present + non-zero scripted-rollout fills) in market-spec §14.4.

**Suggested resolution.** Resolve empirically, NOT by inventing a rule: derive a per-contract traded-volume / order-count distribution directly from `data/rawOMIE/` (orders + trades files, filtered by `contract_id`) across the available delivery days, and record the observed distribution as a **data-driven engineering characterization** (analogous to §13.3's observed statistics), explicitly labelled as a sample statistic and not a market rule. If OMIE or ACER later publish an authoritative liquidity/volume profile, cite it here and promote the relevant facts into the spec. Until then, tape/contract selection that needs a "liquidity" notion must use the §14.2 window-length proxy plus the §14.4 non-degeneracy bar, both labelled as proxies/thresholds rather than rules.

**Status:** **Open, tracked.** Not a market-rule ambiguity that blocks the simulator core — it is a *missing domain concept* that downstream tape/contract-selection decisions must route around with the documented proxy. Mark resolved if/when an authoritative liquidity ranking is sourced, or when a data-driven characterization is adopted as the standing reference.

**Sources consulted:**
- Negative evidence: exhaustive corpus search under `docs/omiecid/corpus/` for liquidity/volume/peak-hour/near-GCT-concentration terms returned no authoritative ranking (only *liquidación*/settlement hits and generic mechanics).
- `docs/omie/market-spec.md` §14 (DA-TAPE-2026-07 ruling: window-length proxy §14.2, tape-selection axes §14.3, non-degeneracy bar §14.4), §2.2/§3.1/§11.1 (gate-timing hard rules the proxy rests on), §13.1/§13.2/§13.4 (orders/trades columns + `contract_id` mapping), §13.3 (precedent for a data-driven observed-statistics characterization).
- `docs/omie/agile-plan.md` §"Story P3-PERF-TAPE-1".
- `outputs/cid_rl/runs/stage0_20260703_585f775/report.md` §2 Setup, §"3 candidate optimizations" #3 (the 5459/1365 baseline the shorter tape is measured against; the informal "liquid peak-hour" phrasing this OQ pins down).

---

## OQ-22 — Touch-crossing agent orders (P3.2b-FOLLOWUP-1) — CID faithfulness + OQ-10 phantom-liquidity interaction

**Filed:** 2026-07-08 (raised by story `P3.2b-FOLLOWUP-1`, "Touch-relative price reach in `_decode_action`"; the RL-advisor action-geometry advisory named this the domain-analyst touchpoint — `docs/cid_rl/phase3-reviews/P3-PERF-TAPE-1-action-geometry-advisory.md` §3, final bullet).

**Note on scope.** This is a **market-semantics sign-off**, not an implementation or RL-design decision. The pricing formula, the action-space cardinality, and the reward plumbing are owned by the architect / developer / RL owner; this OQ rules only on (A) whether a marketable agent order crossing the historical resting touch is faithful to OMIE CID continuous-trading semantics, and (B) whether the resulting fills' interaction with the OQ-10 H1 (`rest_until_gct`) over-matching tolerance requires a constraint on the crossing behavior or is acceptable for the v1 collapsed Stage-0/1 same-set-overfit learnability experiment (`P3-STAGE01-LEARN`).

**Background.** Today the RL agent in `_decode_action` prices every non-NOOP order at `mid ± price_offsets[bucket]` with `price_offsets = (-1, 0, +1)` EUR — a half-spread reach of at most 1 EUR. On the real windows in scope the minimum top-of-book spread is ≈5.5 EUR (half-spread ≈2.75 EUR), so `1 < 2.75` ⇒ no bucket ever reaches the opposite touch, and (because agent orders are IOC and the step interleave is background→agent) there is no passive-fill path either. The fix makes the marketable ("cross") bucket price at or through the opposite top-of-book touch (`best_ask` for a BUY, `best_bid` for a SELL), so the agent can actually aggress against the resting book and fill.

---

### Part (A) — Is a touch-crossing agent order faithful to OMIE CID semantics? **VERDICT: FAITHFUL.**

A marketable agent order that crosses the historical resting touch — pricing at/through `best_ask` (BUY) or `best_bid` (SELL) and matching against resting liquidity under price-time priority — is **normal, spec-conformant continuous-trading behavior**. It is exactly the aggressive/marketable-order case the matching algorithm is written to handle. It violates **no** OMIE CID rule.

**Market-spec citations (verified on-disk 2026-07-08):**

1. **§6.1 Price-Time Priority** (market-spec.md lines 324–329): "A match occurs when best bid price ≥ best ask price." This is precisely the crossing condition an aggressive touch-reaching order satisfies. Hard rule, mandated by ACER Decision 11/2024 Art. 5 and CACM Regulation Art. 51.

2. **§6.2 Regular matching** (lines 331–339): "Incoming order attempts to match against best counterpart in the consolidated order book. Match is possible if: orders are opposite sides, same contract, and `sell_price ≤ buy_price`. Trade executes at the **resting order's limit price** (maker pricing)." A BUY priced at `≥ best_ask` gives `sell_price ≤ buy_price` at the ask, i.e. a match; a SELL priced at `≤ best_bid` gives the symmetric match at the bid. The incoming (agent) order is the aggressor/taker; the historical tape order is the passive maker and **sets the trade price** — the agent does not dictate execution price by crossing.

3. **§2.3 / §8.1 value formula** `V (EUR) = Q (MW) × p (EUR/MWh) × d (h)`, d = 0.25 h for 15-min contracts (market-spec.md line 93 and lines 398–406). This is the settlement value of the resulting fill and is unchanged by *who* aggresses.

4. **§6.6 Self-trade prevention: none at XBID level** (lines 359–367; OQ-12): there is no ownership-based exclusion. An agent order crossing resting liquidity — including liquidity that historically originated from any participant — is matched purely on price-time priority. No STP rule blocks it.

> **Citation-anchor correction (for the architect/test-agent locking this).** The advisory (`P3-PERF-TAPE-1-action-geometry-advisory.md` §3) and the `P3.2b-FOLLOWUP-1` card both cite "market-spec §5.3" for the `V = Q·p·d` value formula and for "the §5 price-time-priority matching." **That market-spec section number is a mis-cite.** In `market-spec.md`, §5.3 is *"Modification rules"* (timestamp reset on modify), NOT the value formula. The correct market-spec anchors are: **§2.3 (and §8.1)** for `V = Q·p·d`, and **§6.1 / §6.2** for crossing and price-time-priority matching. The "§5.3" origin is the *source document* (Continuous Trading Matching Algorithm §5.3), where the value formula does live and which market-spec §2.3 cites — the confusion is a source-vs-spec section-number collision. Any test docstring or card text asserting CID-faithfulness of the crossing fill should anchor on **market-spec §6.1/§6.2 (crossing + maker pricing) and §2.3/§8.1 (value)**, not §5.3.

**Consequence for the fix.** Pricing the cross bucket at the opposite touch is not merely permissible — it is the *only* way to express the ordinary "take liquidity now" action that the CID continuous session supports. The prior `mid ± 1 EUR` geometry was, if anything, *less* faithful: it structurally forbade an action (aggressing across a >2 EUR half-spread) that a real CID participant can always take. **No CID rule constrains how deep an aggressive order may price** beyond the ±9,999 EUR/MWh hard envelope (§2.3, OQ-01) and the tick grid — a taker may price arbitrarily far through the book; it simply trades against successive resting levels at their maker prices until its quantity is exhausted or no further level crosses (§6.2 point 6: "If matching exhausts one price level, continues to next best price level").

---

### Part (B) — Interaction with OQ-10 H1 (`rest_until_gct`) over-matching tolerance. **VERDICT: ACCEPTABLE-FOR-V1, WITH A STAGE01 RUN-REPORT BANNER. No constraint required on the crossing formula.**

**(i) Is the phantom-liquidity concern real for this fix?** **Yes, the mechanism is real** — and the P3.2b-FOLLOWUP-1 fix is precisely what makes it *reachable* by the agent for the first time.

OQ-10 established that OMIE public data is a submission log with no lifecycle events, so a replay cannot distinguish (a) agent-cancelled, (b) GCT-expired, and (c) still-resting residual volume. Heuristic H1 (`rest_until_gct`, the default in `replay_orders_through_book()`) resolves this by leaving unmatched residual resting until GCT. On busy contracts this can leave **residual liquidity resting in the simulated book that was in fact cancelled historically and was not available at that moment** — "phantom" liquidity (see OQ-10 "Decision impact" and `book-depth-cap-finding.md` §Resolution, which reclassifies the busy-contract replay-fidelity gap under the OQ-10 over-matching mechanism). Under the *old* ±1 EUR geometry the agent could never reach the touch, so it could never aggress against phantom residual — the interaction was latent. Under the P3.2b-FOLLOWUP-1 fix a touch-crossing agent BUY/SELL can now fill against phantom residual liquidity, i.e. execute a trade that could not have happened historically. This is a genuine (not hypothetical) replay-fidelity gap for the agent's own fills.

**(ii) Does it require a constraint on the crossing behavior?** **No.** A bounded-crossing-depth constraint (e.g. "cross at most K ticks past the touch," or "cross at most the top level") is **not domain-defensible** and would itself introduce a *new* infidelity:

- There is **no CID rule** bounding how far an aggressive order may price through the book (Part A; §2.3 hard envelope + tick grid are the only price constraints; §6.2 point 6 explicitly walks successive levels). Imposing a crossing-depth cap would fabricate a market rule that does not exist — the same error OQ-13 (PPD magnitude) and OQ-19 (operational price band) warn against: an empirical/engineering envelope must never be dressed as a market rule.
- The phantom-liquidity problem is **a property of the H1 replay heuristic, not of the agent's order**. The correct locus for any fidelity fix is the replay heuristic (OQ-10's domain — e.g. a better fate-of-residual heuristic, or the deferred single-victim eviction noted in OQ-10), never a distortion of the agent's action semantics. Constraining the agent to "work around" an H1 artifact would corrupt the very thing this experiment is trying to measure (whether a learnable signal exists in the *faithful* action space) while leaving the underlying replay gap unaddressed.

Therefore the pricing formula and its locked tests **must NOT** encode a crossing-depth bound on domain grounds. (This resolves the card's Size-M contingency — "would rise to M only if the domain ruling forces a bounded-crossing-depth model": **it does not.** The story stays Size S on the domain axis.)

**(iii) Is it acceptable for the v1 collapsed Stage-0/1 learnability experiment?** **Yes — acceptable-for-v1, recorded as a `P3-STAGE01-LEARN` run-report banner (not an OQ-10 heuristic change, not a test constraint).**

The reasoning is scoped to what `P3-STAGE01-LEARN` is: a **deliberate same-set overfit** whose stated purpose is to **detect whether a learnable signal exists** (agent beats the random + always-NOOP floors), *not* to achieve perfect fill fidelity or an optimal agent (agile-plan §P3-STAGE01-LEARN Description: "Goal = prove a learnable signal and lock the obs/action/reward contracts, NOT find an optimal agent"; Eval-split disposition: "N/A-by-design — the v1 experiment is a deliberate same-set overfit"). For that question:

- The phantom-residual fills are a **fidelity** limitation (some agent fills would not have occurred against the true historical book), **not a learnability confound**: they do not create a spurious *learnable* signal that the floors cannot also access. The random and always-NOOP floors are paired on the **same tapes and seeds** and run against the **same H1 book state** (agile-plan §P3-STAGE01-LEARN Baseline: "Paired on the same tapes and same seeds ... no cherry-picked best-seed comparison"). Any phantom liquidity present is equally available to the random floor's marketable orders. The promotion gate is a *relative* comparison (trained vs. random vs. NOOP on identical conditions), so a fill-fidelity offset that is common to all three arms does not manufacture a false *positive* "signal" — it cannot make the trained agent beat a floor that faces the identical book.
- Conversely, admitting these fills is what **un-degenerates** the baseline: the whole reason for the fix (advisory §2) is that without a reachable fill *all three* arms collapse to `0 ≈ 0 ≈ 0`, a false negative. Accepting some over-optimistic fills (fidelity cost) to obtain a non-degenerate, faithfully-priced action space (the experiment's actual precondition) is the correct v1 trade-off.
- Perfect fill fidelity is explicitly **out of scope for v1** and is already tracked: OQ-10 carries the busy-contract over-matching gap as an **accepted documented limitation** (OQ-10 "Decision impact", 2026-06-11: "the busy-contract replay-fidelity gap is reclassified under this OQ ... the OQ-10 over-matching mechanism ... dominates the gap"). The P3.2b-FOLLOWUP-1 fix does not worsen the *replay*; it merely lets the agent *reach* the pre-existing, already-accepted gap.

**What to record, and where.**

- **`P3-STAGE01-LEARN` run report — banner (required).** The experiment's run report (canonical path `outputs/cid_rl/runs/stage01_<date>_<git-hash>/report.md`, `type: experiment`, RL-owner-owned) must carry a one-line **fidelity caveat banner** stating: *"Agent fills may include matches against OQ-10 H1 (`rest_until_gct`) phantom residual liquidity — liquidity that historically may have been cancelled before this moment. This is an accepted v1 fill-fidelity limitation (OQ-10, OQ-22); the same-set-overfit learnability claim is a relative trained-vs-floor comparison on identical H1 book state and is not invalidated by it."* This sits naturally alongside the existing Eval-split "N/A-by-design with a run-report banner" pattern the card already uses (agile-plan §P3-STAGE01-LEARN Eval-split disposition). The banner is the RL owner's to place; this OQ authorizes it as the correct disposition.
- **OQ-10 — no change required.** The H1 over-matching tolerance and its known-limitation framing are unchanged by this ruling; OQ-22 records the *newly-reachable* interaction and cross-references it. No OQ-10 heuristic change is mandated by P3.2b-FOLLOWUP-1. (Any future decision to tighten H1 — e.g. the deferred single-victim eviction — remains OQ-10's domain and would reduce this fidelity gap for free; it is not a prerequisite for STAGE01.)
- **Locked tests — no crossing-depth constraint.** Per (ii), the P3.2b-FOLLOWUP-1 tests must lock only that the cross bucket *reaches* the touch (`≥ best_ask` for BUY / `≤ best_bid` for SELL) and that `fill_rate > 0` on a real ≈5.5 EUR-spread tape; they must **not** encode any upper bound on crossing depth as a "domain rule." The card's Test-first contract item "(If the domain ruling constrains the crossing depth, the test-agent adds a lock for that constraint)" is **discharged with: no such lock — the ruling imposes no crossing-depth constraint.**

---

**Summary verdicts.**
- **(A) FAITHFUL.** A touch-crossing marketable agent order is normal CID continuous-trading behavior under price-time priority — market-spec **§6.1** (crossing condition), **§6.2** (regular matching, maker pricing), **§2.3/§8.1** (value `V = Q·p·d`), **§6.6/OQ-12** (no STP). Violates no OMIE CID rule. (The card/advisory "§5.3" citation is a market-spec mis-cite; anchor on §6.1/§6.2 + §2.3/§8.1.)
- **(B) ACCEPTABLE-FOR-V1, banner required, no formula/test constraint.** The phantom-liquidity interaction is real and newly-reachable, but (i) it is an OQ-10 H1 replay-heuristic property, not an agent-order property; (ii) a crossing-depth constraint is NOT domain-defensible (no such CID rule; would fabricate a market rule and corrupt the measured action space); (iii) it is acceptable for the same-set-overfit learnability experiment because the promotion gate is a relative comparison on identical H1 book state — record a fidelity caveat banner in the `P3-STAGE01-LEARN` run report; leave OQ-10 unchanged.

**Spec impact:** **None.** No `market-spec.md` clarification is required — the ruling rests entirely on existing §2.3, §6.1, §6.2, §6.6, §8.1 text. (The only spec-hygiene note is the §5.3-vs-§6.1/§2.3 citation-anchor correction above, which is a downstream-citation fix, not a spec change.)

**Sources consulted:**
- `docs/omie/market-spec.md` §2.3 (value formula), §6.1 (price-time priority), §6.2 (regular matching / maker pricing), §6.6 (no STP), §8.1 (settlement value), §11.1 (hard-rules table).
- `docs/omie/open-questions.md` OQ-10 (H1 `rest_until_gct` over-matching tolerance + accepted-limitation framing), OQ-12 (no STP), OQ-13 / OQ-19 (empirical-envelope-vs-market-rule precedent).
- `docs/cid_rl/phase3-reviews/P3-PERF-TAPE-1-action-geometry-advisory.md` §1 (confirmed no-cross + no-passive-fill mechanism), §2 (`0≈0≈0` degeneration), §3 (the domain-analyst touchpoint, options (i)/(ii), and the "§5.3" citation this ruling corrects).
- `docs/omie/agile-plan.md` §"Story P3.2b-FOLLOWUP-1" (the card: deliverables, test-first contract, Size-M contingency on a bounded-crossing-depth model) and §"Story P3-STAGE01-LEARN" (same-set-overfit learnability experiment: Description "prove a learnable signal, NOT an optimal agent"; paired-floor Baseline; Eval-split "N/A-by-design with a run-report banner").

---

## OQ-23 — Latent quiescence gap in `regenerate_slices`: a `dirty`-invisible cross-slot peak consumption

**Filed & dispositioned:** 2026-07-09 (consulting dispatch for story `P3-PERF-REGEN-MASK-1`; advisory / non-blocking — the current story replicates the behavior byte-identically regardless of this ruling).

**Category:** Mixed. The **market-rule** half (Q2) is a domain question about market-spec §4.2.1 Rule 4. The **simulator-internal invariant** half (the `iceberg_manager.py:352-354` post-condition and the `test_perf_a1_fused_regen_gate.py::TestP2Perf8CascadeQuiescence` assertion) is an **engineering invariant** — this OQ is the direct successor of OQ-20 (same class: a `peak_consumed=True` side-channel slot whose OID is absent from the LOB), reached by a *different mechanism*.

---

### The question

In `regenerate_slices` (`gymnax_exchange/omie/iceberg_manager.py:296-561`), the inner slot scan visits active slots in ascending index order. For a slot *j* the body:
1. dispatches slot *j*'s regen slice via `process_order_array` (`:440-445`, a match-then-rest aggressor),
2. re-checks **only slot *j*'s own OID** (`:456-457`: `post_row = get_order(new_lob, lookup_side, order_id)`; `order_id` is *j*'s OID) to set `slice_re_consumed`,
3. OR-accumulates `dirty = dirty | slice_re_consumed` (`:520`).

The outer fixed point (`:541-560`) re-runs the whole scan only while `dirty` is True.

**The gap:** if slot *j*'s regen slice **partially rests** — it consumes the visible peak of an *already-visited, lower-index, opposite-side* slot *k* (a trade), then rests its remainder visibly — then for slot *j*, `post_row[2] > 0` (its own remainder is resting), so `slice_re_consumed=False` and `dirty` is NOT set by *j*. Slot *k*'s peak, however, has now been consumed; slot *k* was already visited earlier in this pass and will not be revisited. Nothing in the body re-checks *k*. Unless some **other** slot independently sets `dirty`, the outer loop exits with slot *k* in `peak_consumed=True` (its OID absent from the LOB → `get_order(...)[2] == -99`). This violates the documented post-condition at `iceberg_manager.py:352-354` ("after `regenerate_slices` returns, no active slot is in `peak_consumed=True`") and, if it occurs on the final regen call of a `submit_order`, is exactly the state the LOCKED `test_perf_a1_fused_regen_gate.py::TestP2Perf8CascadeQuiescence` `_quiescence_violations` probe flags (`order_row[2] <= 0` on an active slot).

---

### Q1 — Constructibility. **VERDICT: CONSTRUCTIBLE on valid OMIE inputs (plausibly-scriptable order flow); it is NOT structurally unreachable.**

The mechanism requires two opposite-side icebergs, *k* (lower slot index, visited first) and *j* (higher slot index), such that *j*'s **regenerated** slice, when dispatched, crosses *k*'s **currently-visible** peak and only partially fills against it, then rests. Every link in this chain is producible by valid OMIE order flow:

- **Two opposite-side icebergs both active with visible peaks.** Standard: any BUY iceberg and any SELL iceberg both resting. Valid, common.
- **Slot ordering *k* < *j*.** `register` appends to the packed prefix in submission order, so *k*'s iceberg was registered before *j*'s. Controllable by submission order.
- **A regen of *j* that crosses *k*'s peak.** *j*'s regenerated slice is priced at *j*'s (possibly ICEwPI-drifted) slice price. For a SELL iceberg *j* regenerating a slice at a price ≤ BUY iceberg *k*'s visible bid, the SELL slice crosses *k*. This is a §4.2.1 Rule 3 / Rule 4 event (a regenerated slice matching resting opposite-side liquidity) — explicitly a spec-contemplated interaction, not a pathological one.
- **The regen slice of *j* must PARTIALLY rest (not be fully re-consumed).** This is the discriminating condition that keeps `dirty` off for *j*. It holds whenever *j*'s regen slice quantity (`min(peak_j, remaining_hidden_j)`) exceeds *k*'s visible peak quantity: the slice fills *k*'s peak (a trade), and the surplus rests. Controllable by sizing `peak_j > peak_k` (both are free order parameters within the 0.1 MW … 999 MW quantity envelope, OQ-01).
- **No OTHER slot sets `dirty` in the same pass.** With only *j* and *k* as the interacting icebergs (plus optionally some inert non-crossing icebergs), the only `slice_re_consumed`-True candidate is *j* itself, which is False by the partial-rest condition. So `dirty` is False at pass end and the outer loop exits with *k* un-regenerated.

The scenario is therefore **constructible** — a deliberately-scriptable synthetic order sequence realizes it (probe below). Its *frequency* on real replay tapes is a separate, unmeasured question (the existing `test_perf_a1_fused_regen_gate` fixtures do not happen to construct it, which is why the gap is currently latent); constructibility does not require it to be common. I mark this **CONSTRUCTIBLE (plausibly-scriptable; not yet observed on a real fixture)** rather than "observed", because I did not run a real-tape sweep to find a natural instance — but there is no structural barrier, and the synthetic probe is a valid OMIE order sequence.

---

### Q2 — Rule-4 verdict. **VERDICT: NOT a Rule 4 violation (within tolerance). It is the SAME class of simulator-internal side-channel inconsistency ruled on in OQ-20 — an engineering invariant, not an OMIE market-rule breach.**

This is the load-bearing ruling, and it turns entirely on **what Rule 4 binds**, already settled by the market-spec §4.2.1 *Scope clarification* sub-bullet (P2-PERF-12 ruling, 2026-05-28) and OQ-20:

> Rule 4 is a **within-call fixed-point requirement on the resting `OrderBook` (the LOB)**. It requires (a) the resting book is **uncrossed** when the matching process returns, and (b) every regenerated slice that crossed during the process was **fully consumed or fully rested at a non-crossing price** by return, with all resultant trades emitted **under the originating process's timestamp** (never deferred across submit cycles). It is **NOT** a post-call cleanliness invariant on `iceberg_meta`. "An active iceberg slot whose `order_id` is absent from the LOB … is a permitted between-aggressor state." (market-spec §4.2.1, line 229.)

Applying that to the OQ-23 end-state:

1. **The resting book is UNCROSSED.** Slot *j*'s regen slice consumed *k*'s peak (a completed trade) and rested its own surplus. After the pass: *k*'s consumed peak OID is gone from *k*'s side; *j*'s remainder rests on *j*'s side at *j*'s slice price. For the pair to have crossed, *j*'s resting remainder would have to cross *k*'s remaining visible liquidity — but *k*'s crossing peak is exactly what *j*'s slice just consumed. There is no crossed resting pair left between *j* and *k*. (If *k* still exposed a *second* crossing peak, *j*'s slice would have kept matching it, not rested — contradicting the partial-rest premise.) The binding Rule 4 observable (uncrossed resting book) **holds**.

2. **All trades were emitted intra-call.** The trade that consumed *k*'s peak was concluded during *this* `submit_order`'s matching process, carrying this process's timestamp. Nothing is deferred to the next aggressor. The observable-distinguishability concerns Rule 4 protects against (§4.2.1: crossed-LOB snapshot between aggressors, wrong triggering timestamp, reordered trade sequence) **do not arise** — no cascade trade is left pending, and no crossed slice is left resting.

3. **What survives is a stale side-channel slot, not a crossed book.** Slot *k*'s `iceberg_meta` row now points to an OID absent from the LOB, with `remaining_hidden_k > 0` and its next slice not yet activated. This is byte-for-byte the OQ-20 signature (`get_order(...)[2] == -99` on an active slot; internally-uncrossed LOB), reached by a new route (cross-slot consumption during *j*'s regen, rather than an `_ice_path` across-call drain). Per OQ-20 and market-spec line 229, a between-aggressor iceberg at rest with its visible peak consumed and the next slice not yet activated is **spec-permitted** — the matching algorithm (Continuous Trading Matching Algorithm §3.3, lines 265–275) activates the next slice when subsequent matching engages the iceberg, not eagerly.

**Therefore Q2 = NOT a Rule 4 violation; within tolerance.** The market-spec requires no change. The `iceberg_manager.py:352-354` post-condition ("no active slot in `peak_consumed=True`") and the `TestP2Perf8CascadeQuiescence` assertion are, per OQ-20, a **simulator implementation invariant**, not an OMIE market rule — and they are *over-strict as a Rule 4 conformance probe* (OQ-20 §"Decision dependency"), exactly as they were for the OQ-20 signature.

---

### Is a must-fix follow-up warranted? **NO must-fix on Rule 4 / market-spec grounds. YES — a scoped ENGINEERING correctness review is warranted, on the OQ-20 axis, and it is more pressing than the OQ-20 signature was.** (Owner: architect / `IcebergManager` implementer — NOT the domain analyst.)

The domain verdict does **not** spawn a Rule-4 must-fix story. But this OQ is NOT merely a re-run of OQ-20's "benign stale slot" — the between-aggressor consumers OQ-20 flagged as hazards are **live and correctness-affecting** in the OQ-23 shape, for a specific reason worth handing to the architect:

- **`no_fill_path` relies on the post-condition (`contract_book.py:721-733`).** P2-PERF-8 gated regen out of no-fill iterations *on the guarantee* that no active slot is left `peak_consumed=True` between aggressors (the code comment at `:723-732` states this explicitly). If the OQ-23 gap fires on the **last regen call of a `submit_order`**, a slot with a consumed peak survives into the next aggressor's arrival while `no_fill_path` skips regen — so that consumed hidden liquidity is **not** re-exposed as a fresh slice when it should be. This is an *observable-liquidity* consequence on the simulator's own book evolution (the hidden quantity that should have re-surfaced as slot *k*'s next visible peak stays hidden one aggressor too long), distinct from OQ-20's "benign, will lazily activate" framing — because the P2-PERF-8 no-fill gate is specifically designed to NOT lazily activate it.
- **`matchable_hidden_volume` / FOK pre-check (OQ-20 §"Why the test is still load-bearing").** A stale slot *k* with `remaining_hidden_k > 0` but a consumed visible peak can mis-report matchable hidden volume at the FOK pre-check path — the same correctness hazard OQ-20 enumerated, now reachable via OQ-23's route.

**Recommendation to the orchestrator:** After `P3-PERF-REGEN-MASK-1` merges (it replicates this behavior byte-identically and must not be re-scoped), file a **FOLLOW-UP engineering story** (not a market-spec change) to close the OQ-23 gap in `regenerate_slices`, owned by the architect/`IcebergManager` implementer. The fix is a bounded-cost extension of the existing outer fixed-point: **the `dirty` re-arm must reflect cross-slot peak consumption, not only self-OID re-consumption.** Candidate mechanisms (implementer's call, cited illustratively — the domain analyst does not prescribe module structure): (i) after each pass, re-derive `any_active_peak_consumed` across `meta[:n_active]` (the same `get_order(...)[2] <= 0` derivation the LOCKED quiescence test already uses) and OR it into the outer-loop `dirty` predicate, so a pass that consumed a lower-index peak forces another pass; or (ii) have the body OR into `dirty` whenever its dispatched slice produced a fill that was NOT its own remainder resting (i.e. it aggressed another slot). Option (i) is the minimal, test-aligned choice. Either way the truncation backstop (`_TRIP_BOUND` / `max_match_iterations`) already bounds the extra passes.

This follow-up is **engineering** (correct the simulator-internal invariant the `no_fill_path` optimization depends on), NOT a Rule 4 conformance fix — `main` and `P3-PERF-REGEN-MASK-1` remain **spec-correct on Rule 4's binding observable** (uncrossed resting book + intra-call trade emission). The follow-up should relabel any test docstrings that call this "spec-violating per market-spec §4.2.1 Rule 4" to "engineering-invariant violation (OQ-23; not a Rule 4 breach — the resting book is uncrossed)", consistent with the OQ-20 relabel guidance and the §4.2.1 Scope clarification. The domain analyst also flags that `test_perf_a1_fused_regen_gate.py`'s `TestP2Perf8CascadeQuiescence` docstrings currently assert the `peak_consumed` derivation IS "spec-violating per market-spec §4.2.1 Rule 4" (`:1444-1446`, `:1505-1506`); that framing pre-dates the 2026-05-28 Scope clarification and is imprecise (it is an engineering invariant, not a Rule 4 breach) — a test-agent docstring correction, not a domain-spec change, and out of scope for `P3-PERF-REGEN-MASK-1`.

---

### Constructibility probe (a synthetic OMIE order sequence that WOULD construct the scenario)

Precise enough for a test-agent to script as a fixture; prices/quantities are on-tick at OMIE defaults (`price_tick=0.01 EUR/MWh`, `qty_tick=0.1 MW`; contract 37 mid-day OPEN window as in the existing `_cascade_synthetic_orders` fixture). The construction forces slot ordering `k < j`, an opposite-side crossing regen of *j* against *k*'s peak, a **partial rest** of *j*'s slice, and **no other slot** re-arming `dirty`.

Standard (PPD=0) icebergs are sufficient — ICEwPI is not required. The idea: register the BUY iceberg *k* first (lower slot), then the SELL iceberg *j* whose peak is smaller than *k*'s peak so *j*'s **first** peak rests without fully draining *k*; then send an aggressor that consumes *j*'s visible peak, triggering *j*'s **regeneration** — and size *j*'s regen slice larger than *k*'s remaining visible peak so it consumes *k*'s peak (a trade) and rests its surplus (partial rest → `slice_re_consumed=False` for *j*).

Pseudo-order list (chronological; all NON unless noted; `[side, price EUR/MWh, total qty MW, peak MW, exec]`):

1. **ICE BUY  @ 60.00, total 1.0, peak 0.2** — iceberg *k*. Rests; registers at slot 0. Visible bid 0.2 @ 60.00.
2. **ICE SELL @ 60.10, total 1.0, peak 0.6** — iceberg *j*. Peak (0.6) @ 60.10 does NOT cross the 60.00 bid (60.10 > 60.00), so *j*'s first peak rests on the ask; registers at slot 1. Book uncrossed: bid 0.2@60.00, ask 0.6@60.10.
   - *(Slot ordering achieved: k=slot 0 < j=slot 1. j's peak (0.6) > k's peak (0.2) — the partial-rest sizing.)*
3. **NON BUY  @ 60.10, qty 0.6** — aggressor. Crosses and fully consumes *j*'s visible ask peak (0.6 @ 60.10). `submit_order` matches it; *j*'s peak OID is now consumed → post-match `regenerate_slices` runs.
   - Inside `regenerate_slices`, the pass visits slot 0 (*k*) first: *k*'s bid peak is still resting (0.2 @ 60.00, untouched) → `peak_consumed_k = False`, no action, advance. Then it visits slot 1 (*j*): *j*'s peak was consumed → `need_regen_j = True`. It dispatches *j*'s **next SELL slice**. To make that slice CROSS *k*'s resting bid, *j*'s next slice must price ≤ 60.00. **Use an ICEwPI SELL for *j* is NOT needed** — instead price *j*'s base so its slice rests at/below the bid: set *j*'s limit at **60.00** (peak 0.6) from the start (adjust step 2 to `ICE SELL @ 60.00, total 1.0, peak 0.6`), and make step-2 not cross by ensuring *k*'s bid is entered at a **non-crossing** price at that instant — see the tightened variant below.

**Tightened, self-consistent variant** (removes the “peak must not cross at rest but must cross at regen” tension by using ICEwPI drift on *j*, which is the natural OMIE mechanism for a slice to re-price into a crossing region):

1. **ICE BUY @ 60.00, total 1.0, peak 0.2** — iceberg *k*, slot 0. Visible bid 0.2 @ 60.00.
2. **ICEwPI SELL @ 60.20, ppd = −0.20, total 1.2, peak 0.6** — iceberg *j*, slot 1. First peak @ 60.20 does not cross the 60.00 bid → rests on the ask. (Sell ICEwPI normally drifts *up*; here we want the slice to drift *down into* the bid. If the sign convention forbids a negative sell PPD, use the standard-iceberg variant below instead — the PPD is only a device to move *j*'s regen slice price to ≤ 60.00.)
3. **NON BUY @ 60.20, qty 0.6** — consumes *j*'s visible ask peak (0.6 @ 60.20). Post-match `regenerate_slices`: slot 0 (*k*) still resting (no-op); slot 1 (*j*) regenerates its **slice 2** at the drifted price 60.20 + (−0.20)·slice_count = **60.00**, quantity `min(peak=0.6, remaining=0.6)=0.6`. Dispatched as SELL 0.6 @ 60.00 → **crosses *k*'s bid peak (0.2 @ 60.00)**: 0.2 MW trades (consuming *k*'s visible peak), and the surplus **0.4 MW rests** on the ask @ 60.00.
   - For slot *j*: `post_row = get_order(SELL side, j_oid)` → 0.4 MW resting → `peak_consumed=False` → `slice_re_consumed=False` → **`dirty` not set by *j***.
   - Slot *k*: its visible bid peak (0.2 @ 60.00) is now **consumed** — `get_order(BUY side, k_oid)[2] == -99` — but *k* was visited earlier in the pass and is not revisited. No other slot sets `dirty`.
   - Outer fixed point sees `dirty == False` → **exits with slot *k* (slot 0) in `peak_consumed=True`.** ← the constructed gap.

**Standard-iceberg (PPD=0) variant, if the ICEwPI sign convention blocks a downward-drifting SELL** (buy: negative PPD; sell: positive PPD — market-spec §4.2, so a SELL cannot drift down): swap the roles so the *regenerating* iceberg is a **BUY** whose slice drifts **down** is also wrong-signed; instead use **flat (PPD=0)** slices and make the crossing happen because the counterparty peak is at the *same* price. Concretely:

1. **ICE SELL @ 60.00, total 1.0, peak 0.2** — iceberg *k*, slot 0. Visible ask 0.2 @ 60.00.
2. **ICE BUY  @ 60.00, total 1.2, peak 0.6** — iceberg *j*, slot 1. Its peak (0.6 @ 60.00) **crosses *k*'s ask immediately on registration** — so this fires on `_ice_path`'s own dispatch, not on a later regen. That consumes *k*'s 0.2 ask and rests *j*'s 0.4 surplus as a bid; the post-`_ice_path` `regenerate_slices` then sees *k* (slot 0) consumed. Whether *k* is regenerated in that same call depends on visit order (slot 0 visited before *j*'s own state settles) — this is a **valid alternate constructor** but it exercises the gap on the *registration* dispatch rather than on a *regen* dispatch. The cleanest faithful reproduction of the exact `dirty`-invisibility described in the question is the **ICEwPI variant above**; the standard-iceberg path reaches the same end-state (`peak_consumed=True` on slot 0 at pass end) if `peak_j > peak_k` so *j*'s crossing slice partially rests.

**Fixture guidance for the test-agent (characterization pin, NOT a locked contract — per the standing disposition):** script the ICEwPI variant (steps 1–3) as the primary constructor; assert `_quiescence_violations(state)` is NON-empty after order 3's `submit_order` (i.e. slot 0 is left `peak_consumed=True`) to *pin the current behavior* of `P3-PERF-REGEN-MASK-1`, and label the test docstring "characterization pin for OQ-23 (latent quiescence gap; NOT a Rule 4 violation — the resting book is uncrossed)". If the ICEwPI sign convention is validated to forbid a downward-drifting SELL slice, fall back to the standard-iceberg registration-dispatch constructor, verifying the `peak_j > peak_k` partial-rest sizing still yields a resting surplus. Either constructor must first be validated against `validate_order` (positive OIDs, in-window contract, on-tick prices/quantities within the ±9,999 EUR/MWh and 0.1–999 MW envelopes, OQ-01) — all values above are inside those envelopes.

---

### CORRECTION + CLOSURE NOTE (additive; appended 2026-07-09, `P3-PERF-REGEN-QUIESCENCE-1-FOLLOWUP-1`)

**Context.** This note is filed by the `P3-PERF-REGEN-QUIESCENCE-1` architecture audit as an *additive* correction to the probe section above (the historical text is retained unchanged; the corrections below supersede it where they conflict). The engineering follow-up this OQ recommended has now landed. Two of the constructors scripted in the probe section are now known to be inaccurate, and the canonical gap constructor is the locked one in `tests/omie/test_regen_quiescence_rearm.py`. The probe results were established during the story's test-authoring (recorded in `sprint-status.md` §"P3-PERF-REGEN-QUIESCENCE-1 — Tests Written (Step 3, 2026-07-09)").

**(a) The PRIMARY ICEwPI-SELL probe is NOT valid OMIE order flow — do not script it.** The "Tightened, self-consistent variant" above (step 2: `ICEwPI SELL @ 60.20, ppd = −0.20`) uses a **negative PPD on a SELL**, which market-spec **§13.6** (DA-PPD-2026-04) rules HARD-forbidden: "ICEwPI buy orders must have `PPD < 0`; ICEwPI sell orders must have `PPD > 0`" (sourced from OMIE Rulebook §4.5, XBID Continuous Trading Matching Algorithm §3.3). The probe section's own hedge — "If the sign convention forbids a negative sell PPD, use the standard-iceberg variant below instead" — is now **VALIDATED TRUE**: the sign convention does forbid it. That primary probe *does* mechanically construct the gap (the simulator's `validate_order` does not currently inspect PPD sign, so it is accepted at the code boundary), but it is **not valid OMIE order flow** and must **not** be scripted by future readers as an OQ-23 constructor. A locked contract must never pin behavior on domain-forbidden inputs.

**(b) The literal standard-iceberg (PPD=0) fallback SELF-HEALS — it does NOT construct the gap.** The "Standard-iceberg (PPD=0) variant" above (step 1 `ICE SELL @ 60.00, peak 0.2`; step 2 `ICE BUY @ 60.00, peak 0.6`) was probe-verified at authoring time and does **not** leave a quiescence violation: the registration-consumed slot (slot 0) is visited FIRST by the regen pass, and every subsequent cross is a FULL consumption that re-arms `dirty` via the self-OID channel, so the outer fixed point drains and self-heals. (Mechanically, with `PPD = 0` a slot's regen slice prices at its own base, so it cannot cross liquidity its previous resting peak did not — the "crossed-at-regen but not at rest" tension this OQ itself flagged.) It is therefore **not** a valid OQ-23 gap constructor.

**(c) The CANONICAL gap constructor is the locked valid-sign refinement in `tests/omie/test_regen_quiescence_rearm.py`.** The working, spec-valid constructor is a refinement of the registration-dispatch family that uses only §13.6-valid signs and sizes both regen dispatches to partially rest (the `dirty`-invisibility discriminant):

1. **ICEwPI BUY *k* @ 60.20, `ppd = −0.20`** (§13.6-valid: buy PPD < 0), total 1.0 MW, peak 0.3 MW → slot 0; bid 0.3 @ 60.20.
2. **ICE SELL *j* @ 60.00, `ppd = 0`**, total 1.0 MW, peak 0.4 MW → `_ice_path` dispatches the peak, which crosses *k*'s bid (trade 0.3 @ 60.20) and rests 0.1 ask @ 60.00; *j* registers at slot 1. The post-registration `regenerate_slices` pass then produces **two partial-rest dispatches** — slot 0's slice-2 drifts to `60.20 + (−0.20)·1 = 60.00`, consumes *j*'s 0.1 ask and rests 0.2 bid (partial rest → `slice_re_consumed` False); slot 1's slice-2 @ 60.00 consumes *k*'s fresh 0.2 bid and rests 0.2 ask (partial rest → `slice_re_consumed` False). `dirty` is never set; the outer loop exits with **slot 0 (*k*) left `peak_consumed=True`, 0.4 MW hidden stranded** — the exact `dirty`-invisibility mechanism (a regen-pass dispatch consuming an already-visited slot's peak). Both orders pass `validate_order` and honor the §13.6 sign rule.

This constructor is locked (and parametrized over both small AND realistic OIDs, per the `GAP_OID_PAIRS` amendment — the small-OID fixtures had accidentally passed a wrong-column gate) in `tests/omie/test_regen_quiescence_rearm.py`; see its module docstring for the full mechanism trace. Future readers should treat that file as the authoritative OQ-23 constructor, not the probe section above.

**(d) STATUS: OQ-23 CLOSED / RESOLVED.** The engineering follow-up this OQ recommended has landed via **`P3-PERF-REGEN-QUIESCENCE-1`** (merged `3d604e3`, 2026-07-09) — **NOT** via `P3-PERF-REGEN-MASK-1`, which was **ABANDONED** (shipped nothing to `main`; the probe section's "after `P3-PERF-REGEN-MASK-1` merges" recommendation is superseded — the fix landed independently). The fix re-arms the `regenerate_slices` outer fixed-point on **cross-slot peak consumption** (the minimal option-(i) shape: re-derive `any_active_peak_consumed` across `meta[:n_active]` each pass and OR it into the `dirty` predicate, with a trade-predicate gate forced by the ≤10% throughput bound). Post-fix:

- The `iceberg_manager.py:352-364` post-condition ("no active slot in `peak_consumed=True` after `regenerate_slices` returns") now **genuinely HOLDS** on the OQ-23 gap shape (synthetic constructor RED→GREEN; plus a NATURAL real-tape gap event on cid 90 — 121 stranded slots on `main` at one submit, healed 121→0 to exact convergence by the fix).
- The `contract_book.py` `no_fill_path` P2-PERF-8 regen-skip reliance on that post-condition is therefore **SOUND** — the hazards this OQ flagged (mis-exposed hidden liquidity across aggressors; FOK pre-check mis-report) are closed.

**Framing preserved (do NOT import the developer-log imprecision):** the closure does **not** reclassify Q2. This was and remains an **ENGINEERING invariant** (the OQ-20-class side-channel-slot inconsistency the `no_fill_path` optimization depends on), **NOT** a Rule-4 breach — the resting book was always uncrossed and all trades were always emitted intra-call (Q2 above stands unchanged). The story's developer log loosely labeled the cid-90 violations a "Rule-4 quiescence violation"; that is imprecise and this domain record does **not** adopt it. The market-spec required **no** change (confirmed at merge — no `market-spec.md` edit was owed or made).

**Cross-links:** locked constructor `tests/omie/test_regen_quiescence_rearm.py` (canonical); story `P3-PERF-REGEN-QUIESCENCE-1` (`docs/omie/agile-plan.md`; merged `3d604e3`); probe record `docs/omie/sprint-status.md` §"P3-PERF-REGEN-QUIESCENCE-1 — Tests Written (Step 3, 2026-07-09)" and §"… — Architecture Audit (2026-07-09)"; sign rule `docs/omie/market-spec.md` §13.6.

---

### Summary verdicts

- **Q1 (constructibility): CONSTRUCTIBLE** on valid OMIE inputs (plausibly-scriptable; probe above). Not structurally unreachable. Not observed on an existing real fixture (unmeasured frequency).
- **Q2 (Rule 4): NOT a Rule 4 violation — within tolerance.** The resting order book is uncrossed and all trades are emitted intra-call; the surviving state is a stale `iceberg_meta` side-channel slot (OID absent from LOB), which market-spec §4.2.1 *Scope clarification* + OQ-20 rule is a permitted between-aggressor state, NOT an OMIE market-rule breach. No market-spec change.
- **Follow-up: NO Rule-4 must-fix. YES a scoped ENGINEERING correctness follow-up is warranted** (owned by architect/`IcebergManager` implementer, filed AFTER `P3-PERF-REGEN-MASK-1` merges) — because the P2-PERF-8 `no_fill_path` regen-skip optimization *depends* on the post-condition this gap breaks, so the OQ-20 hazards (mis-exposed hidden liquidity across aggressors; FOK-pre-check mis-report) are live in this shape. The minimal fix re-arms the outer fixed point on cross-slot peak consumption (re-derive `any_active_peak_consumed` across `meta[:n_active]` each pass and OR into the `dirty` predicate). `P3-PERF-REGEN-MASK-1` is NOT re-scoped by this ruling — it replicates the behavior byte-identically and authors adversarial coverage as characterization pins citing OQ-23, not locked Rule-4 contracts.

**Sources consulted:**
- `docs/omie/market-spec.md` §4.2.1 Rule 4 + *Scope clarification* sub-bullet (lines 219–230), §4.2 (iceberg / ICEwPI sign rule), §2.3 / OQ-01 (price & quantity envelopes).
- `docs/omie/open-questions.md` OQ-20 (the direct precursor ruling: `peak_consumed=True`/absent-OID slot is an engineering invariant, not a Rule 4 breach; the enumerated `matchable_hidden_volume` / FOK-pre-check hazards), OQ-10 (adjacent over-matching context).
- `gymnax_exchange/omie/iceberg_manager.py:296-561` (`regenerate_slices`: self-OID re-check `:456-457`, `dirty` OR `:520`, outer fixed point `:541-560`, post-condition docstring `:352-364`), `:414` (`peak_consumed = order_row[2] <= 0`).
- `gymnax_exchange/omie/contract_book.py:721-733` (`no_fill_path` reliance on the post-condition — the P2-PERF-8 regen-skip gate).
- `gymnax_exchange/jaxob/JaxOrderBookArrays.py:1049-1069` + `jaxob_constants.py:10` (`get_order_by_id` returns the `NEGATIVE_RETURN_ID=-99` row for an absent OID → `[2] <= 0`).
- `tests/omie/test_perf_a1_fused_regen_gate.py` (LOCKED: `TestP2Perf8RegenGateByteIdentity` byte-identity 9-tuple; `TestP2Perf8CascadeQuiescence` `_quiescence_violations` probe `:1247-1289`, `:1437-1447`, `:1497-1508` — the assertion this scenario would latently trip, and whose "spec-violating per §4.2.1 Rule 4" docstring framing this OQ corrects to "engineering invariant").
- `docs/omie/sprint-status.md` §"Open domain question — latent quiescence gap in `regenerate_slices` (2026-07-09)" (the dispatch question) and §P3-PERF-REGEN-MASK-1 card (v5.71).

---

## OQ-24 — Terminal forced-close-out reference price for an open agent position at episode end (inventory-aware reward)

**Filed & ruled:** 2026-07-10 (domain-semantics dispatch for story `P3-REWARD-INVENTORY-1`, "Inventory-aware reward — close the free-short reward gap"). This is a **market-semantics ruling on the settlement/reference-price half only.** It does **NOT** select the RL mechanism (candidate (a) terminal mark-to-market close-out / (b) running per-step inventory penalty / (c) both) — that arbitration is the `rl-research-engineer` (advisor mode) call downstream of this ruling. Reward-function, action-space, and observation-design decisions are explicitly out of the domain analyst's scope.

**Category:** Market domain semantics (settlement / delivery-obligation), with a spec cross-reference. Not an engineering invariant.

---

### The question

The v1 OMIE dense reward pays `sign · settlement_value` per fill and ends the episode on wall-clock alone (`done = new_time >= gct_minutes_from_got`), with the agent's signed `agent_inventory` tracked but never charged — so an unbounded, unsettled short is reward-positive (a reward-specification exploit; the finding record is `sprint-status.md` §"P3-REWARD-INVENTORY-1 — reward-exploit finding (2026-07-10)"). The `P3-REWARD-INVENTORY-1` card proposes closing this with a terminal close-out (candidate (a)/(c)), which requires settling the open position against a *reference price*. This OQ rules the three domain-semantics sub-questions the card routes:

1. Is a terminal forced-close-out reference price a defensible, first-class OMIE quantity, or is it outside the intraday-simulator scope?
2. If a reference price is used, which is domain-defensible — the reserved `day_ahead_anchor` obs slot, the contract's realized VWAP, or none?
3. What does the real OMIE/XBID firm-delivery obligation imply for an agent that ends a 15-min contract episode NOT flat — does the economically-correct consequence support a *penalty* framing over a *mark-to-market* framing?

---

### Q1 — Is a terminal forced-close-out reference price a first-class OMIE quantity? **VERDICT: NO. There is no automatic position close-out in the OMIE continuous intraday market, and no spec-grounded intraday reference/settlement price for an unclosed position. A terminal mark-to-market close-out is OUTSIDE the intraday-simulator scope as a market fact.**

The card's assertion — that the market-spec places delivery/imbalance resolution outside OMIE scope — is **VERIFIED TRUE against the on-disk spec text** (all five card-cited line references confirmed exact at 2026-07-10):

1. **No automatic close-out exists.** market-spec **§8.5** (line 435): *"The continuous market has no automatic position close-out at GCT. Open orders are cancelled at GCT but already-matched trades are firm."* The market itself does **not** liquidate an open (already-matched, net-nonzero) position at gate closure. A forced close-out is therefore not something the OMIE continuous market *does* — it is a modeling construct the simulator would *impose*.

2. **The counter-leg energy value is explicitly exogenous.** market-spec **§8.4** (lines 429–430): the buy-agent per-trade PnL is *"`−V` (cash outflow) + market value of acquired energy (**exogenous to the simulator**; relevant for imbalance resolution)"* and the sell-agent PnL is *"`+V` (cash inflow) − cost of delivered energy (**exogenous**)."* The value of the energy the agent is obligated to deliver/receive is, by the spec's own words, **not a quantity the intraday simulator produces**. Line 431 admits *"plus mark-to-market of open positions if applicable"* — but "if applicable" is a conditional the spec never discharges with a spec-grounded price; it is a placeholder, not a defined quantity.

3. **Imbalance resolution is outside OMIE scope — stated three times.** market-spec **§8.5** (line 436): *"Imbalance resolution (positions not physically delivered) is handled by TSOs in balancing markets — outside the scope of the intraday simulator."* market-spec **§10** (line 477, HFT/LOBSTER differences table): *"Trades are firm delivery obligations; imbalance resolved externally by TSOs … imbalance resolution is outside OMIE scope."* market-spec **§11.2** (line 521, Modeling choices, NOT hard rules): *"How to handle imbalance costs (outside OMIE market rules scope)."*

The decisive point is §11.2: the spec **already classifies "how to handle imbalance costs" as a MODELING CHOICE, not a hard rule.** So the economically-correct consequence of an unclosed position (an imbalance cost) is real, but its *magnitude and pricing are not fixed by any OMIE market rule.* There is no OMIE-published intraday settlement/clearing price for the continuous market against which an open position could be marked (the continuous market has no clearing price — it is bilateral, price-time-priority matching at maker prices, §6.2). The only OMIE-defined settlement quantity is the **realized trade value** `V = Q·p·d` (§8.1, §8.3), which by construction only exists for *matched* volume, not for an *open residual*.

**Conclusion for Q1:** A terminal forced-close-out reference price is **NOT a first-class OMIE-spec quantity.** Any close-out price the simulator uses is a **documented modeling assumption**, not a spec-grounded value. This ruling is directly analogous to the OQ-13 (PPD magnitude) and OQ-19 (operational price band) precedents: a number that downstream code needs but that the corpus does not define as a rule must be declared an engineering/modeling choice, never dressed as an OMIE rule.

---

### Q2 — If a reference price IS used, which is domain-defensible? **VERDICT: NONE is spec-grounded. Both candidates are modeling assumptions; of the two, the contract's realized VWAP is the *less* unfaithful, and the reserved `day_ahead_anchor` slot is NOT domain-defensible as currently specified. A penalty-style term (candidate (b)) needs NO reference price and invents no exogenous quantity — it is the option that commits the simulator to the fewest unsupported assumptions.**

Evaluating the two reference-price candidates the card names, plus the no-reference-price option:

**(i) The reserved `day_ahead_anchor` obs slot (`omie_env.py:526`, currently `0.0`) — NOT domain-defensible as a close-out price.**

- The term **"day_ahead_anchor" appears NOWHERE in `market-spec.md`** (exhaustive grep 2026-07-10). It is a *simulator-side observation slot*, not a spec quantity. The domain analyst must not infer a domain rule from a reserved code field (that would violate the "do not infer domain rules from code behavior" constraint).
- Semantically, the **day-ahead market is a distinct market from the continuous intraday market.** The spec references the day-ahead only as a *separate* session (the glossary PDBF/PDVP row, line 56, is TSO day-ahead availability declarations; line 72 notes the hourly product is "still used in Day-Ahead"). There is **no spec statement** that the day-ahead clearing price is the correct settlement/close-out reference for an *intraday 15-min contract* position. Using the day-ahead price to close out an intraday position is an economic modeling assumption about the day-ahead↔intraday basis being zero (or a known constant) — which the corpus neither states nor supports. The real economic close-out reference for an undelivered intraday position is the **TSO imbalance/balancing price** (§8.5, line 436), which is *neither* the day-ahead price *nor* an OMIE quantity — it is exogenous and TSO-set.
- **If (a)/(c) is nonetheless chosen with `day_ahead_anchor`:** it is admissible **only as a documented modeling assumption**, and that assumption commits the simulator to: *"the economic value of an unsettled intraday position is marked at an exogenous day-ahead anchor price, standing in for the (unmodeled, TSO-set) imbalance price; the day-ahead↔intraday-15min basis and the real imbalance-price spread are assumed away."* That is a strong, unsupported commitment and must be labelled as such in the config + merge note, not presented as OMIE-faithful.

**(ii) The contract's realized VWAP — a modeling assumption, but the *less* unfaithful of the two reference-price candidates.**

- VWAP is **derivable from spec-grounded quantities** — it is the quantity-weighted mean of the contract's realized trade values, each of which is `V = Q·p·d` (§8.1), the one settlement quantity the spec *does* define. So VWAP does not invent an *exogenous* price the way the day-ahead anchor does; it is an *endogenous summary of the same contract's own executed trades.*
- **But it is still not a spec-defined settlement reference.** The spec defines no "contract reference price," no volume-weighted index, and (§6.2) the continuous market has no single clearing price. Marking an open position at the contract's own VWAP assumes the agent could have closed its residual at the average traded price — an idealization (it ignores that closing a residual near/after GCT means crossing the *remaining* book, or that liquidity may be gone). It is a *defensible* idealization for a v1 close-out because it uses only in-contract, spec-grounded trade values and requires no exogenous market — but it remains a **modeling assumption**, and must be documented as one (analogous to §14's window-length *proxy* and §13.3's *observed-statistics* framing: derivable from data/spec quantities, labelled a proxy, never a rule).
- **If (a)/(c) is chosen with VWAP:** the commitment is milder than the day-ahead case: *"an unsettled residual is marked at the contract's own realized volume-weighted trade price, idealizing that the residual could be liquidated at the average executed price; the near-GCT liquidity/market-impact reality of actually closing the residual is assumed away."*

**(iii) No reference price (candidate (b), running per-step inventory penalty) — the option that invents no exogenous quantity.**

- A `−λ·|inventory|` or `−λ·inventory²` term requires **no reference price at all.** It does not claim to be a *settlement* — it is an explicit shaping/holding-cost term that makes carrying inventory costly without asserting any market-price for the close-out. This is the option most honest about the domain fact that *there is no spec-grounded close-out price* (Q1). It has an in-repo precedent (`mm_env` `inv_penalty`, `mm_env.py:2516-2519`), and it is the only candidate that does **not** commit the simulator to an unsupported exogenous-price assumption.
- Domain-wise, (b) is best read as a **proxy for the (exogenous, TSO-set) imbalance cost** of an undelivered position (see Q3) — a penalty for holding an obligation the agent did not settle, rather than a claim about what price it settled at. That framing is faithful to §8.5/§11.2 (imbalance cost is real but exogenous and rule-unspecified) *without* fabricating a price.

**Conclusion for Q2:** **No reference price is domain-grounded.** If the RL advisor selects a reference-price close-out (a)/(c), it must be recorded as a **documented modeling assumption**; of the two named candidates the **contract VWAP is preferable to `day_ahead_anchor`** (endogenous & spec-derivable vs. exogenous & cross-market), and `day_ahead_anchor` should not be used without the explicit basis-assumption caveat above. The **penalty-only** path (b) is the sole candidate that commits the simulator to **no** unsupported exogenous price and is the most domain-honest expression of "an open position costs something" given that OMIE defines no intraday close-out price. The domain ruling does **not** force any of the three — it constrains the *pricing semantics* so the RL advisor chooses on solid ground.

---

### Q3 — Firm-delivery-obligation semantics for a non-flat episode end: does the real obligation support a *penalty* framing over a *mark-to-market* framing? **VERDICT: YES. The economically-correct consequence of ending a contract non-flat is an exogenous, TSO-set balancing/imbalance cost — not an intraday market close-out. This supports a PENALTY (holding-cost) framing over a mark-to-market framing, because a mark-to-market presupposes an intraday close-out price the market does not provide.**

What the real OMIE/XBID firm-delivery obligation implies:

1. **Every matched trade is a firm physical obligation.** market-spec **§8.2** (line 411, verified exact): *"All trades in XBID/SIDC are firm delivery obligations. A buy trade implies an obligation to purchase; a sell trade implies an obligation to deliver."* (OMIE Rules §46.2; §11.1 hard-rules table, line 505.) An agent that ends a 15-min contract net-short has a firm obligation to *deliver* energy it did not procure; net-long, a firm obligation to *take/pay for* energy it did not on-sell.

2. **The obligation is settled physically, and any shortfall is resolved in the balancing market by the TSO — not by the intraday exchange.** market-spec **§8.5** (line 436) and **§10** (line 477): imbalance resolution for positions not physically delivered is *"handled by TSOs in balancing markets — outside the scope of the intraday simulator."* So the *consequence* of a non-flat end is real and costly (an imbalance settlement), but its *mechanism* is **exogenous** to OMIE and its *price* is **TSO-set** (the imbalance price), not an OMIE quantity.

3. **Why this supports a penalty framing over a mark-to-market framing:**
   - A **mark-to-market close-out** framing asserts *"the position was liquidated at price `p_ref`"* — but the continuous market performs **no such liquidation** (§8.5: "no automatic position close-out at GCT"), and OMIE publishes **no intraday reference price** to liquidate against (Q1/Q2). A mark-to-market thus necessarily fabricates a price the market does not provide. It is the framing that most strongly presupposes a non-existent spec quantity.
   - A **penalty (holding-cost) framing** asserts *"carrying an unsettled obligation into delivery incurs a cost"* — which is **exactly what the real market does** via the exogenous TSO imbalance settlement (§8.5, §11.2). It models the *existence and disincentive* of the imbalance cost without falsely claiming an intraday close-out occurred at a specific market price. The penalty coefficient `λ` is then honestly a **modeling stand-in for the (rule-unspecified, exogenous) imbalance-cost intensity** (§11.2 classifies "how to handle imbalance costs" as a modeling choice) — not a claim about a settlement price.
   - Therefore, on **domain-faithfulness grounds**, a penalty framing is *more* consistent with the firm-delivery/exogenous-imbalance semantics than a mark-to-market framing. A mark-to-market is admissible only as a documented idealization (Q2 (ii)) and is strictly *more* assumption-laden than a penalty.

   (Note the scope boundary: this is a *domain-faithfulness* observation, not an RL-mechanism selection. RL considerations the domain analyst does **not** rule on — dense-reward density, Markov integrity, credit-assignment horizon — may still lead the RL advisor to prefer (a) or (c). Q3 establishes only that the *market semantics* lean toward a penalty framing and that any mark-to-market term must carry the Q2 modeling-assumption caveat.)

---

### What this commits the simulator to (summary of assumption-costs)

| Candidate | Reference price needed | Spec-grounded? | If chosen, the documented assumption is… |
|-----------|------------------------|----------------|-------------------------------------------|
| (a)/(c) with `day_ahead_anchor` | Yes (exogenous, cross-market) | **No** — not in spec at all; day-ahead is a distinct market | "Unsettled intraday position marked at an exogenous day-ahead anchor standing in for the unmodeled TSO imbalance price; DA↔ID-15min basis and imbalance spread assumed away." **Strong, discouraged.** |
| (a)/(c) with contract VWAP | Yes (endogenous) | **No**, but derivable from §8.1 `V=Q·p·d` trade values | "Residual marked at the contract's own realized volume-weighted trade price; near-GCT liquidity/market-impact of actually closing the residual assumed away." **Milder; the less-unfaithful reference option.** |
| (b) running penalty | **No** | N/A (invents no price) | "Carrying an unsettled obligation incurs a holding cost `λ·f(inventory)` as a proxy for the exogenous, TSO-set imbalance cost (§11.2 modeling choice); no close-out price is claimed." **Fewest assumptions; most domain-honest.** |

---

### Summary verdicts

- **Q1 (is a terminal close-out reference price a first-class OMIE quantity?): NO.** The continuous market has no automatic close-out (§8.5, line 435), the counter-leg energy value is explicitly exogenous (§8.4, lines 429–430), and imbalance resolution is outside OMIE scope and classified a *modeling choice* (§8.5 line 436; §10 line 477; §11.2 line 521). All five card-cited line references verified exact on-disk 2026-07-10. Any close-out price is a **documented modeling assumption**, not a spec quantity (OQ-13 / OQ-19 precedent).
- **Q2 (which reference price?): NONE is spec-grounded.** `day_ahead_anchor` is not in the spec and is cross-market → **not domain-defensible** as a close-out price without an explicit basis-assumption caveat; contract VWAP is a **modeling assumption** but the *less* unfaithful one (endogenous, derivable from §8.1 trade values); **penalty-only (b) needs no reference price and invents no exogenous quantity** — the most domain-honest option. The ruling constrains the pricing semantics; it does **not** select the RL mechanism.
- **Q3 (firm-delivery → penalty vs mark-to-market?): PENALTY framing is more domain-faithful.** The real consequence of a non-flat end is an **exogenous, TSO-set balancing/imbalance cost** (§8.2 firm obligation, line 411; §8.5 TSO balancing, line 436), not an intraday close-out. A penalty models the *existence/disincentive* of that cost without fabricating a non-existent intraday close-out price; a mark-to-market presupposes exactly such a price and is admissible only as a documented idealization. (This is a domain-faithfulness observation; the (a)/(b)/(c) selection remains the RL advisor's, weighing RL factors the domain analyst does not rule on.)

**Spec impact: NONE.** No `market-spec.md` change is required — the ruling rests entirely on existing §8.2 / §8.4 / §8.5 / §10 / §11.2 text, which already (a) states there is no automatic close-out, (b) marks the counter-leg value exogenous, and (c) classifies imbalance-cost handling as a modeling choice. The spec is already correct and sufficient for this ruling; adding a "close-out price" would fabricate a quantity §8.5 explicitly says the market does not provide.

**Status:** **Resolved (domain-semantics half).** The RL-mechanism selection (a)/(b)/(c) is routed to the `rl-research-engineer` (advisor mode) per the `P3-REWARD-INVENTORY-1` card and is **out of this OQ's scope.** Re-open only if OMIE/ACER later publish an intraday reference/settlement price for open continuous positions, or a spec-blessed imbalance-price model is sourced into the corpus.

**Sources consulted:**
- `docs/omie/market-spec.md` §8.1 (line 400, `V=Q·p·d` — the only spec-defined settlement quantity), §8.2 (line 411, firm delivery obligation), §8.4 (lines 429–431, exogenous counter-leg value + conditional "mark-to-market … if applicable"), §8.5 (lines 435–436, no automatic close-out + imbalance handled by TSOs outside scope), §10 (line 477, firm-delivery / imbalance-outside-scope difference row), §11.1 (line 505, firm-delivery hard rule), §11.2 (line 521, "how to handle imbalance costs" = modeling choice), §6.2 (maker-price matching, no single clearing price); negative grep — "day_ahead_anchor" appears nowhere in `market-spec.md`.
- `docs/omie/open-questions.md` OQ-13 (PPD empirical-vs-rule precedent), OQ-19 (operational-price-band empirical-vs-rule precedent) — the "a needed number the corpus does not define as a rule must be a documented engineering/modeling choice" pattern this ruling follows.
- `docs/omie/agile-plan.md` §"Story P3-REWARD-INVENTORY-1" (v5.84: Description, OPEN QUESTION routing, Input artifacts — the exact framing and the three candidate designs); the band preamble (v5.84 finding, verified against `HEAD`).
- `docs/omie/sprint-status.md` §"P3-REWARD-INVENTORY-1 — reward-exploit finding (2026-07-10)" (the finding record; consulted, not modified).

---

## OQ-25 — Fill-starvation of agent resting maker orders in the non-reactive H1 replay; well-posedness of a reactive tape

**Filed & resolved (diagnosis):** 2026-07-22 (`WF-25-FOLLOWUP-1` / `EXP-NON-ONLY-1-FOLLOWUP-1`). Canonical diagnosis note: `docs/omie/reactive-tape-fill-diagnosis.md`.

**Question.** Why does the non-reactive historical replay not cross the agent's resting NON/maker orders (WF-25 telemetry: own-touch PASSIVE 0.63%, price-improve MID 12%, cross-on-submit CROSS 99.2%), and is an agent-order-aware ("reactive") tape a well-posed fix?

**Verdict.** Attributed to a MIX of OQ-10 H1 phantom residual liquidity (ROOT, cause 1) and price-time priority queue position (PROXIMATE mechanism, cause 2); genuine microstructure (cause 3) is minor. The agent's resting orders ARE in the background-matching path (`omie_env.py::step_env` carries the book; `_extract_passive_fill` credits hits) — they are reachable but buried: the agent's order carries the current session timestamp and the matching core (`JaxOrderBookArrays.py::_get_top_{bid,ask}_order_idx`) is price-then-earliest-arrival-time priority, so at equal price the agent queues behind the entire never-cancelled phantom wall (H1 `rest_until_gct`). This resolves the apparent tension (phantom over-rest vs. agent starvation): they are one mechanism — phantom liquidity out-prioritises the agent at the same price and absorbs the aggressor flow first.

**Ruling.** The reactive-tape build is WELL-POSED. The faithful fix locus is the tape/replay side ONLY — (1) a better fate-of-residual heuristic in `replay_orders_through_book` (OQ-10's domain, thins the phantom queue), and/or (2) a reactive/heterogeneous flow model that responds to the agent's resting quotes. NEVER the agent's action semantics (no crossing-depth cap, no maker-priority boost, no action-space surgery, no reward reshaping — same grounds as OQ-22 Part B(ii)). Spec impact: NONE (rests on existing §6.1/§6.2 price-time priority; OQ-10; OQ-22).

**Related:** OQ-10 (phantom liquidity, root substrate), OQ-22 (touch-crossing agent orders + fidelity-fix-locus ruling), `book-depth-cap-finding.md` (~2× over-fill, `match_rate 0.13`).

**CORRECTION (2026-10-01, OMIE-AGENT-TIME-PRIORITY-1).** This block corrects the Verdict's premise that "the agent's order carries the current session timestamp … so at equal price the agent queues behind the entire never-cancelled phantom wall". The Verdict and the Ruling are kept as written. The OQ-25 status does not change. The full block is in `reactive-tape-fill-diagnosis.md` §2 (CORRECTION of the same date).
- **(a) Pre-fix fact.** The stamp was `state.time`, the minutes-from-GOT clock (≤ 60). That is far below every tape second, so the agent sat AHEAD of every same-price non-agent order, not behind it.
  - Measured: the non-agent quantity ahead of a no-fill agent rest, at a level holding Q > 0, was 0 on 100% of such rests.
  - Pre-L1 (3): 334/334 PASSIVE BUY, 327/327 PASSIVE SELL, 22/22 MID BUY. Post-L1 (3b): 1,400/1,400, 1,446/1,446, 12/12.
  - The WF-25 telemetry (PASSIVE 0.63%, MID 12%, CROSS 99.2%) was produced with front-of-queue priority.
- **(b) Post-fix behaviour.** The agent is stamped `book.ts_counter` (`omie_env.py:1143`). It rests behind every earlier arrival (market-spec §5.2). Measured on (3b): ahead == Q on 100% of rests (1,410 / 1,397 / 11).
  - **Residual R1**, (3b) fix leg: later non-agent arrivals that rank ahead of the resting agent at its own price.
  - PASSIVE BUY: mean 0.033, max 2, 47 of 1,449 rests. PASSIVE SELL: 0.001, max 1, 2 of 1,447. MID BUY and MID SELL: 0.
- **(c) Passive-fill steps, base → fix, six held-out days at op4.**
  - Filing-time expectation, (3) @ `de1882d` (post-S5, pre-S1/S2; the literals equal the card's pre-S1/S2/S5 scratch): PASSIVE BUY 37 → 23, SELL 51 → 26, MID BUY 56 → 51, MID SELL 69 → 69.
  - Current base, (3b) @ `ddc847a` (post-S1/S2/S5, default clock, window about 4× longer): PASSIVE BUY 44 → 32, SELL 53 → 26, MID BUY 68 → 69, MID SELL 211 → 211.
- **(d) Flagged for re-measurement.** The Verdict's PROXIMATE-mechanism attribution ("price-time priority queue position … queues behind the entire … phantom wall") is NOT supported by the WF-25 measurement, because the agent was at the front when it was measured. Re-measurement is the user's decision.
- **(e) Not reopened.** The ROOT attribution (OQ-10 H1 phantom liquidity), the fix-locus ruling (tape/replay side only, never action semantics; OQ-22 Part B(ii)) and the WELL-POSED reactive-tape ruling all stand.
- **(f) Further WF-25 confounds. These are pointers, not rulings.** `OMIE-TAPE-CONTRACT-DATE-1` (S1: two products merged in one tape), `OMIE-SESSION-CLOCK-1` (S2: clock stuck at 58–60 minutes), `OMIE-EVAL-STOP-AT-DONE-1` (S5: held-out eval ignored `done`).
- Sources: `docs/cid_rl/phase3-reviews/OMIE-AGENT-TIME-PRIORITY-1-measurement.md` (3), `…-measurement-3b.md` (3b).

## OQ-26 — Day-ahead anchor is episode-constant for a single-contract episode (intra-episode variation is not domain-grounded until multi-contract)

**Filed:** 2026-07-22 (`WF-13 · DOM-DAYAHEAD-SOURCE-1`; canonical note: `docs/omie/dayahead-sourcing-note.md`).

**Category:** Market domain semantics (day-ahead↔CID alignment), obs-only. **Orthogonal to OQ-24** — the day-ahead anchor here is an observation input, NOT a reward/settlement reference.

**Question.** The `day_ahead_anchor` obs channel must be a *per-step, state-aligned* value (WF-13 Markov note: it must add information the single-step obs lacks, without being a future value or a whole-episode constant that lets the policy memorize a tape). The domain-grounded day-ahead price for a 15-min CID contract is the `marginalpdbc` marginal price of that contract's own delivery hour (`anchor(c,D) = marginalpdbc(D, hour (c//4)+1)`). For a **single-contract episode** (the frozen split fixes `contract_id 0`) this is **one scalar for the whole episode** — episode-constant by domain construction. Does that satisfy the Markov/per-step requirement, and how should within-episode variation be treated?

**Verdict.** The step-function alignment (§3 of the sourcing note) is the **correct domain quantity** and satisfies "held across session steps" and no-lookahead (§4). It is state-aligned (the fair-value level of exactly the energy this contract delivers) and known before the session opens. **But it is episode-constant for a single-contract episode** because OMIE publishes exactly one day-ahead price per delivery hour and a single-contract episode trades one contract in one hour. **Genuine intra-episode variation of the anchor is domain-grounded ONLY in a multi-contract episode**, where the active contract (hence its delivery-hour day-ahead price) changes across steps. This is out of scope for the current single-contract stage (multi-contract is Stage-4+).

**Ruling / constraints for WF-14 + rl-re.**
- Feed the domain-correct step-function anchor as specified; do NOT fabricate spurious within-episode variation to make the channel "look" per-step-varying.
- Any normalization/standardization of the channel MUST be a **fixed, tape-independent** transform (shared with the price channels). It must NOT be a per-episode/per-tape statistic — that would re-encode tape identity and re-introduce the memorization risk the Markov note warns against.
- Whether an episode-constant-but-informative anchor is *useful* to the policy (vs. redundant with tape identity) is an **RL-design** question for rl-re/WF-14, out of the domain analyst's scope. The domain contribution is fixed: exactly one day-ahead price per contract-hour; genuine variation needs multi-contract episodes.

**Spec impact:** NONE (rests on existing §3.1 GOT/GCT timing, §13.4 contract mapping, §14.2 window; day-ahead is a distinct market already noted in the glossary). **Related:** OQ-02 (GOT timing), OQ-04 (MTU15 go-live), OQ-24 (day-ahead NOT a settlement reference — orthogonal).

**Re-open if:** the stage moves to multi-contract episodes (then specify the per-step contract→hour→price selection over the traded contract sequence), or the day-ahead auction's 15-min MTU data (post-2025-10-01) is brought in-scope for a later split (then the hour→quarter-hour broadcast in §3 step 2 is replaced by a direct 15-min lookup).

---

## OQ-27 — What is a "tape"? Product identity (delivery date + slot), session-day file partition, and which files a product spans

**Filed & resolved:** 2026-09-30 (`OMIE-TAPE-CONTRACT-DATE-1` Step 0; formalises card Q1 and settles the "k ≥ 93" derivation). **Spec:** market-spec v2.10 §3.1, §13, §13.1, §13.2, §13.4.

**Question.** Does a real-data tape "(date D, contract k)" mean the slot-k rows of file D, or the delivery product (D, k)? What does the file date mean? Which session files does product (D, k) span? In particular, does it need file D+1 when k ≥ 93, as the card proposes from 22:15–22:45 quarter-hour GCTs?

**Ruling.**
1. **Q1: a tape is the delivery product (D, k). YES (high confidence).**
   - D is the `Contrato` **start date** and k = 4·hour + minute/15 of the `Contrato` start.
   - The `eval_split.json` `delivery_date` is the product's delivery date.
   - Rows with the same k and a different `Contrato` date are a different product with its own book (market-spec §6.4).
2. **The file date and `Fecha` are the SESSION day (hard fact of the data format).**
   - File D holds the sends (orders) or matches (trades) in [22:20 D−1, 22:00 D), for products D and D+1.
   - `Fecha` equals the file date on every row (`orders_20250626.1`: 0 exceptions among its rows of both products).
   - Consistent with OMIE Instruction 1/2024 §8 (continuous restarts at 22:20 after IDA2).
3. **GCT is hour-aligned (hard rule).**
   - Each hourly round closes four consecutive quarter-hour contracts (OMIE Rules Annex 2 §2 + Figure 4). So GCT(k) = 60·⌊k/4⌋ − 60 min from D 00:00.
   - Data: k = 3 has no sends after 22:59:59 on D−1; k = 4 is traded until 23:59; k = 51 until 10:59 (2025-06-25, 2026-01-10).
4. **Product (D, k) spans session files D−1 and D ONLY, for every k. The "D+1 when k ≥ 93" rule is REFUTED.**
   - k = 92–95 close together at 22:00 on D, when the IDA2 for D+1 starts. Figure 4 lists only "1-96 (D+1)" as open from 22:00 on D.
   - Data:
     - `orders_20250626.1` and `trades_20250626.1` contain **0** rows of any 06-25 product.
     - All 10,383 rows of (06-25, k = 93–95) are in `orders_20250625.1`, the last sent at 21:59:29.
     - `orders_20260110.1` contains 0 rows of any 01-09 product.
     - No order or trade file on disk has a row in 22:00:00–22:19:59.
5. **The same union applies to orders, trades, and the (Agente, Unidad) universe.**

**Downstream consequences (for the architect/developer; not domain decisions).**
- The file-D+1 read proposed in the card's L1 (i) is unnecessary and must never contribute rows.
- The timestamp upper bound 168,300 s (card L1 (ii)) was derived from the refuted rule. The domain window of a product is [15:20 D−1, GCT(k)) ⊆ [55,200, 165,600) s from D−1 00:00.
- `OMIE-SESSION-CLOCK-1` sets `gct_minutes_from_got = 460 + 15·k`, which assumes quarter-hour GCTs. Under the hour-aligned rule the value is **460 + 60·⌊k/4⌋**. The two agree for contract 0, the frozen-split contract, and diverge for k mod 4 ≠ 0. The architect should reconcile this on that card.
- Open fact (non-binding): whether non-Iberian SIDC participants keep trading quarter-hours 2–4 of an hour cross-border after the Iberian round closes cannot be seen in the Iberian public files. It is irrelevant to the Iberian replay.

**Sources:** OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md (Annex 2 §1 Figures 1–3, §2 and Figure 4, around lines 9741–10030); OMIE_Instruction_1_2024_IDA_Go-Live_Operations.md §8; `data/rawOMIE/` (grep evidence as above, 2026-09-30).

---

## OQ-28 — Do continuous orders survive the IDA clearing window, and how is the book re-matched at reopen?

**Filed:** 2026-09-30 (`OMIE-TAPE-CONTRACT-DATE-1` Step 0; formalises card Q2). **Spec:** market-spec v2.10 §3.3.

**Ruling: orders survive the IDA2 (and IDA3) clearing window and are batch-matched at reopen (medium-high confidence).**
- **Regulatory basis.**
  - The IDA suspends continuous submission and matching ("both regular matching and explicit capacity allocation are suspended", Continuous Trading Matching Algorithm §5.5). No corpus rule cancels resting continuous orders of contracts that stay open.
  - Continuous trading "will continue with Round 1st … at 22:20h" (Instruction 1/2024 §8).
  - Batch matching is the procedure for re-matching a book that may be crossed (market-spec §6.3).
- **Data basis.**
  - 14 trades in `trades_20250626.1` are stamped exactly 25/06/2025 22:20:00, and no order in `orders_20250626.1` was sent at 22:20:00. These trades can only pair orders resting from before 22:00. Several have a blank (non-Iberian) side.
  - No file has sends or matches in 22:00:00–22:19:59.
  - The card cites 6 such trades on the date it measured.
  - The IDA3 gap (10:00–10:20 on D) likewise has 0 sends in `orders_20250625.1`.

**Recorded simplification (modeling choice, not a rule).** The replay carries the book across the gap and applies post-reopen orders sequentially in timestamp order. There is no explicit reopen batch-matching event, so the 22:20:00 trades of the historical record are approximated by regular matching triggered by the first post-reopen orders. Explicit batch matching at reopen is out of scope (`OMIE-TAPE-CONTRACT-DATE-1` "NOT in scope"; `OMIE-SESSION-CLOCK-1` NOT in scope).

**Interaction:** post-IDA re-quote bursts combined with no order IDs cause double-counting under H1 → OQ-10 EXTENSION.

**Re-open if:** a corpus source is found showing that OMIE cancels continuous orders at an IDA gate closure (then the replay must purge the book at 22:00), or a card needs the reopen trades reproduced exactly.

---

## OQ-29 — Incomplete delivery products (a required session file missing)

**Filed & resolved:** 2026-09-30 (`OMIE-TAPE-CONTRACT-DATE-1` Step 0; formalises card Q4). **Spec:** market-spec v2.10 §13.4 *Completeness*.

**Ruling: exclude incomplete products; never load a partial product (high confidence).**
- Product (D, k) requires `orders_{D−1}.1`, `orders_D.1`, `trades_{D−1}.1` and `trades_D.1` (OQ-27 item 4). If any is absent, the product is incomplete.
- A partial product is a different market, with no D−1 order flow and a book that starts at 22:20 already mid-session. It must not be used as a tape or as a §14.4 non-degeneracy witness.
- Raising an error on a missing file, as the card proposes, implements this ruling. The exclusion list is a data fact, not a rule.
- **On-disk state on 2026-09-30 (`data/rawOMIE/`), a snapshot, not an assumption:**
  - ~~**2025-06-01** is incomplete: neither `orders_20250531.1` nor `trades_20250531.1` exists. This matches the user decision of 2026-09-29 to drop it from `train_dates`.~~ **CORRECTION (2026-09-30, OMIE-TAPE-CONTRACT-DATE-1):** superseded. The user added `orders_20250531.1` and `trades_20250531.1` on 2026-09-30, so product 2025-06-01 is now **complete** (contract 0 has 2,546 order rows: 1,731 in `orders_20250531.1` + 815 in `orders_20250601.1`). The 2026-09-29 decision to drop it from `train_dates` was based on the missing file and no longer applies on data grounds. Whether it goes back into `train_dates` is a decision for the story/user, not a domain rule. The earliest session file on disk is now `*_20250531.1`, so **2025-05-31** is incomplete (no `orders_20250530.1` / `trades_20250530.1`).
  - Missing `orders_20260228.1` makes products **2026-02-28** and **2026-03-01** incomplete.
  - Missing `orders_20260406.1` / `trades_20260406.1` make **2026-04-06** and **2026-04-07** incomplete.
  - The card's preliminary list also named the first day of the 2025-10 / 2025-12 / 2026-01 order bands. `orders_20250930.1`, `orders_20251130.1` and `orders_20251231.1` and their trades files are now on disk (for 2026-01-01, `trades_20251231.1` is present), so those band-start products are complete today. Re-check against the disk at load time.
  - **Current incomplete list (CORRECTION 2026-09-30, verified on disk after the 2025-05-31 files were added):** **2025-05-31**, **2026-02-28**, **2026-03-01**, **2026-04-06**, **2026-04-07**, and any delivery date from 2026-04-15 onward (the last session file is `*_20260414.1`).
    - Session files are otherwise continuous from 2025-05-31 through 2026-04-14: 318 `trades_*.1` and 317 `orders_*.1` files, and the only gaps are `orders_20260228.1` and `orders_/trades_20260406.1`.
    - Cross-reference: market-spec v2.11 §13.4 *Completeness* and the §14.5 *No-assume-data* dated snapshot. `docs/data/omie-layout.md`'s coverage matrix (2026-07-14) is stale.

**Re-open if:** a use case needs partial products, for example an "evening-session-only" experiment. That would be a new, explicitly labelled tape type, not a relaxation of this rule.

---

## OQ-30 — Daylight-saving transitions: contract strings, slot count, and wall-clock timestamps

**Filed:** 2026-09-30 (`OMIE-TAPE-CONTRACT-DATE-1` Step 0). **Status: OPEN.** The card ships only a guard: raise when a product's window contains a Europe/Madrid transition, unless the caller opts in.

**What is known.**
- **Regulatory.** A spring (last Sunday of March) delivery day has 23 hours and **92** contracts. An autumn (last Sunday of October) delivery day has 25 hours and **100** contracts. The IDA scopes change accordingly (1–92 / 1–100 of D+1; IDA3 45–92 / 53–100 of D). OMIE Rules Annex 2 §1 Figures 2–3 and §2: "96 contracts (92 or 100 contracts on the corresponding official time change days)", with "23 or 25 rounds". Operational changes may be published by Instruction.
- **Data, spring (`orders_20260329.1`).** There are no `02:xx` contract strings. The quarter-hour before the jump is written `20260329 01:45-20260329 03:00`, and the next is `03:00-03:15`.
- **Data, autumn (`orders_20251026.1`).** The repeated hour appears as `…01:45-20251026 02:00A`, then `02:00A…02:45A` and `02:00B…02:45B` slots (10,588 rows with A/B-suffixed 02:xx strings), and ends `20251026 02:45B-20251026 03:00`.
- **Timestamps.** `Momento de envío` / `Momento casación` are local wall-clock values with no offset or A/B marker. Sends during the repeated hour on the autumn night are ambiguous, and the spring night has a one-hour gap. Wall-clock seconds from D−1 00:00 are then neither monotone-safe nor duration-true.

**What is unknown / to decide (implementation-facing).**
- (a) The slot mapping on DST days. `4·hour + minute/15` maps both A and B slots to the same k and leaves k = 8–11 empty in spring. It may also fail on the `A`/`B` suffix, so the parser may not handle these strings at all.
- (b) The timestamp representation for the transition night. Options include UTC, an explicit offset, or resolving the A/B hour from `Contrato` context.
- (c) Whether the per-day contract count may be 92 or 100 anywhere downstream, which currently assumes 96.
- **Affected products:** every product of the **transition delivery day** D, for example 2025-10-26 and 2026-03-29 (both on disk).
  - Each product (D, k) has a window [15:20 D−1, GCT(k) ≤ 22:00 D], which contains the 02:00–03:00 transition on D.
  - Products of D−1 end by 22:00 on D−1, and products of D+1 start at 15:20 on D, so neither is affected.

**Resolution routes.** Keep the guard, excluding the transition day by default (acceptable: 2 days/year). Or implement (a) + (b) as an explicit DST card. The choice is an engineering/scope decision. The domain facts above are fixed.

---

## OQ-31 — Price-setter in the §4.2.1 Rule-3 hop

**Filed:** 2026-10-02 (scenario-ladder domain review v3.0, finding V3-02, `docs/omie/scenario-ladder-review.md` §13). **Status: RESOLVED 2026-10-02 — Reading B (the iceberg slice sets the price).** See *Resolution* at the end of this entry. **Spec:** market-spec v2.12 §4.2.1 Rule 3, §5.2, §6.2 items 3 and 7; clarified in market-spec **v2.13** §4.2.1 Rule 3.

**The question.** In the §4.2.1 Rule-3 hop, an aggressive NON order consumes an iceberg's visible peak and its unfilled remainder rests in the book. The iceberg's **regenerated** slice, which is the newer entry with a new timestamp, then crosses that resting remainder in the same submit cycle and trades again. Which order's limit price is the trade price of that hop: the resting remainder's, or the iceberg slice's?

**Reading A — the resting remainder sets the price.**
- §4.2.1 Rule 3: *"The resting remainder acts as passive liquidity for the new slice"*, matched *"consistent with how any new order (including a regenerated slice) triggers regular matching against the current book state"* (`market-spec.md:237`). Its Classification: *"The regenerated slice is a new-timestamp entry and legitimately matches against any resting order."* (`:240`).
- §6.2 item 3 and Continuous Trading Matching Algorithm §5.1.1: *"Two orders are matched at the limit price of the order that was already in the order book."* (`Continuous_Trading_Matching_Algorithm_Public_Description.md:469`).
- Under this reading the slice is the incoming order, so the trade prints at the **remainder's** limit, which is the aggressor's own limit $L$.

**Reading B — the iceberg sets the price.**
- §6.2 item 7: *"when a single incoming order matches multiple slices of the same iceberg, the iceberg is always the price-setter"* (`market-spec.md:363`).
- §5.2, iceberg-slice bullet: *"the iceberg is treated as the passive (resting) side throughout and its timestamp does not supersede the incoming order for price determination purposes"* (`market-spec.md:326`).
- Continuous Trading Matching Algorithm §5.1.2: *"the price is always determined by the iceberg order already in the order book and never by the order with the new timestamp, even if the timestamp of the iceberg order is renewed during the matching process"* (`Continuous_Trading_Matching_Algorithm_Public_Description.md:475-479`). §4.2.1 Rule 4 quotes the same clause as confirming intra-process resolution (`market-spec.md:249`).
- §4.2.1 Rule 1 states the same principle for FOK multi-slice fills: *"Each slice trades at the iceberg's (drifted) limit price per the price-determination rule (§5.1.2)."* (`market-spec.md:223`).
- Under this reading the trade prints at the **iceberg slice's** limit (the base price, or the drifted price for ICEwPI).

**When the readings differ.** Whenever the aggressor's limit $L$ differs from the slice price. For a buy aggressor against a sell iceberg, $L \ge$ slice price, and the two readings disagree whenever $L$ is strictly above it. The sell-aggressor case is symmetric. For an ICEwPI slice the gap includes the drift.

**Provisional reading (not a ruling): Reading B, medium-high confidence.**
- In the real algorithm the Rule-3 configuration does not arise as a separate event. The incoming order keeps matching successive slices inside one matching process (§4.2.1 Rule 2; Matching Algorithm §5.1.2 and §5.1.3, `:475-479`, `:484`), and §5.1.2 fixes the price to the iceberg "even if the timestamp of the iceberg order is renewed during the matching process". The aggressor's remainder resting and then being crossed by the renewed slice is the simulator's decomposition of that one process.
- Rule 3's "passive liquidity" phrase describes **matchability** (the slice may trade against the resting remainder). It does not state a price rule, and it should not be read as overriding §6.2 item 7 and §5.2, which do.
- Residual doubt: Rule 3 predates the Rule-2 multi-slice upgrade (M6.4) and is worded as ordinary regular matching. The literal reading of its last sentence supports A.

**What it blocks.**
- **C8's K2** in the scenario ladder (`docs/omie/scenario-ladder.md`, C8 Known result: "Rule-3 trade price per OQ-31"). Under either reading, a record builder that assumes the other one will hard-fail a correct engine or pass a wrong one.
- **I5 in `OMIE-CERT-INVARIANTS-1`.** I5 ("every trade price equals the limit price of that trade's price-setter … including the §4.2.1 Rule-3 hop, per OQ-31") owns mispricing and cannot be left silent at this hop.

**Implementation-side fact to establish (not a domain question; for the architect).** Whether the Rule-3 configuration is still reachable in the engine after the Rule-2 multi-slice upgrade, and if so which price the engine prints today. The engine's current behaviour is evidence of what was built, not of what the rule is.

**Who resolves it.** Either a **user ruling**, or the **domain analyst with source evidence** (for example, a corpus passage that addresses a regenerated slice crossing the same order's resting remainder). It must not be resolved by reading the engine's behaviour.

**If Reading B is adopted.** The §4.2.1 Rule 3 wording becomes a spec clarification ("matchable as passive liquidity; the trade price is the iceberg slice's limit per §6.2 item 7 and §5.2"). That is a market-spec edit for a later spec card. It is recorded here and not applied. *(Superseded 2026-10-02: Reading B was adopted with source evidence, and the clarification is applied in market-spec v2.13 §4.2.1 Rule 3. See Resolution.)*

**Resolution (2026-10-02, domain analyst, source evidence; not derived from engine behaviour). RESOLVED — Reading B.** The trade price of the Rule-3 hop is the **iceberg slice's own limit price**: the base price for a standard iceberg (PPD = 0), or the slice's drifted price $\mathit{base} + k\,\mathit{ppd}$ for an ICEwPI slice $k$. It is **not** the aggressor's limit $L$ carried by its resting remainder.

The corpus contains a passage that addresses this configuration directly, which the provisional reading did not cite. Quotes below are byte-exact, with source line breaks rendered as single spaces.

1. **OMIE Rules, Iceberg bids (Annex, §4.4): an incoming counter bid larger than the visible peak trades against each iceberg instance in turn, and the iceberg is what remains in the book.**
   - `docs/omiecid/corpus/Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10262-10267`: *"- If, while there is an “Iceberg” bid in the Bid Book, a counter bid with a competitive price at a quantity higher than the viewable quantity of the “Iceberg” bid is entered, different transactions will be performed for each instance of the “Iceberg” bid, each with its own creation time. The viewable amount of the “Iceberg” bid in the Bid Book will be the amount that has remained unmatched after the last instance."*
   - Same text in `docs/omiecid/corpus/Authoritative_Document_Map_Spanish_Electricity_Market_Interrelationships_DA_ID_SIDC_REE_Balancing/OMIE_RULEBOOK_2025.md:10262-10267`. Spanish official text, `docs/omiecid/corpus/Authoritative_Document_Map_Spanish_Electricity_Market_Interrelationships_DA_ID_SIDC_REE_Balancing/CNMC_DCOOR_DE_009_24.md:14819-14824`: *"- Si, existiendo una oferta “Iceberg” en el Libro de Ofertas, llegara una oferta contraria con un precio competitivo y una cantidad superior a la cantidad visible de la oferta “Iceberg”, se llevarán a cabo transacciones diferentes por cada instanciación de l a oferta “Iceberg”, cada una con su tiempo de creación. La cantidad visible de la oferta “Iceberg” en el Libro de Ofertas, será la cantidad que haya quedado sin casar tras la última instanciación."*
   - This is exactly the Rule-3 trigger (an aggressor whose quantity exceeds the visible peak). The rule frames every instance, including the one created mid-process "with its own creation time", as a transaction of the **incoming counter bid against the iceberg in the Bid Book**. The iceberg is the side that remains in the book afterwards. Nothing in it lets the counter bid's remainder become the resting side for the iceberg's next instance.
2. **Matching Algorithm §5.1.2: the price follows the iceberg across renewed slice timestamps.** `docs/omiecid/corpus/Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/Continuous_Trading_Matching_Algorithm_Public_Description.md:475-479`: *"In a matching process where a  single order with a new timestamp  is matched against more than one slice of an iceberg order already in the order book, the price is always determined by the iceberg order already in the order book and never by the order  with the new timestamp, even if the timestamp of the iceberg order is renewed during the matching  process."*
3. **Matching Algorithm §5.1.3: the incoming order's remainder enters the book only after it is no longer executable.** Same file, `:490-493`: *"This process continuous as long as the incoming order remains executable and has a positive order quantity. Subsequently , the order is deleted if the order quantity has reached zero or if it has the IOC execution restriction. In all other cases, the order is entered into the order book with its remaining quantity."* While a crossing iceberg slice is still available, the aggressor is still executable. Its remainder is therefore not yet "an order already present in the order book" (§5.1.1, `:466-467`) relative to that slice.
4. **Matching Algorithm §3.3: the iceberg's full quantity is matchable.** Same file, `:257-258`: *"Iceberg orders are limit orders which are only visible with part of their total quantity in the market, while their full quantity is exposed to the market for matching"*.

**Reasoning.**
- In the real algorithm the Rule-3 configuration (aggressor remainder resting, then crossed by the iceberg's renewed slice) does not occur as a separate event. Items 1, 3 and 4 make it one matching process in which the incoming order trades successive iceberg instances.
- Item 2 fixes that process's price to the iceberg, explicitly including slices whose timestamp was renewed mid-process. The §5.1.1 "already in the order book" rule (`:469`) cannot select the aggressor: under item 3, the aggressor's remainder is not in the book while it can still trade against the iceberg.
- Reading A rests only on the simulator's decomposition (remainder rested first, slice regenerated second). Per OQ-31's own terms, that decomposition cannot set the rule.

**Consequences.**
- **Spec:** market-spec v2.13 §4.2.1 Rule 3 now says that the resting remainder is matchable by the regenerated slice, that the trade price is the iceberg slice's limit price (§5.1.2, §6.2 item 7), and that the iceberg is the passive side for that hop (§5.2). There is no market-rule change: §5.2 and §6.2 item 7 already stated this.
- **I5 / C8's K2:** at the Rule-3 hop, the price-setter is **the iceberg**. The hop trade price equals the regenerated slice's limit price. For ICEwPI that is $\mathit{base} + k\,\mathit{ppd}$ with $k \ge 1$, which lies within I5's declared-ladder exemption. A trade at $L \ne$ slice price is a mispricing. When $L$ equals the slice price the readings coincide, so a record builder should draw $L$ strictly through the slice price for the test to discriminate.
- **Engine-facing (for the architect; not a domain question):** two things are not settled here. (a) Whether the engine prints the slice price at this hop today. (b) Whether any passive/aggressor attribution downstream, such as passive-fill or maker attribution, labels the hop's sides consistently with the iceberg being passive. This ruling settles neither and should not be taken from the engine.

**Re-open if:** a corpus or OMIE/SIDC source is found that prices a renewed iceberg instance at the incoming order's limit, or that has the incoming order's remainder enter the book before a crossing iceberg instance is activated.

**ADDENDUM (2026-10-07, `OMIE-SPEC-CERT-AMEND-1`; cross-reference, additive; architect countersignature pending). The ruling also covers an incoming-iceberg aggressor.** When the aggressor is an incoming iceberg (ICE), the iceberg that was **already resting** is the price-setter on every trade of the incoming iceberg's sweep. That includes slices regenerated during the sweep, and it includes any Rule-3-shaped hop in which a regenerated slice of the resting iceberg crosses the incoming iceberg's rested remainder. The hop trades at the resting slice's own limit: the base price, or $\mathit{base} + k\,\mathit{ppd}$. It never trades at the incoming iceberg's limit.
- **Source.** OMIE Rules Annex §4.4 (`:10256-10261`): an incoming iceberg's total quantity is considered, with "a transaction for each counter bid that it matches with". The same Annex §4.4 at `:10262-10267` and Matching Algorithm §5.1.2 (`:475-479`) apply unchanged, because the incoming iceberg is "a single order with a new timestamp" matched against the slices of the iceberg "already in the order book". §5.1.3 (`:490-493`): the incoming iceberg's remainder enters the book only once it is no longer executable.
- **Incoming ICEwPI.** The incoming iceberg's own price path is a separate question, filed as OQ-34. It does not affect who sets the price.
- **Spec:** market-spec v2.14 §4.2.1 Rule 3, *ICE aggressor* bullet, and Rule 2, *Incoming iceberg* bullet.

---

## OQ-32 — Same-second arrival order in replay

**Filed:** 2026-10-07 (`OMIE-SPEC-CERT-AMEND-1`; spec gap routed from `OMIE-STAMP-UNIQUE-1`). **Status: OPEN. The simulator's convention is recorded as a modelling choice.** **Spec:** market-spec §5.2 ("single global queue (unique, sequential)"), §6.1, §13.1 (`Momento de envío`).

**The question.** OMIE's public order files give send times to the second (`Momento de envío`, DD/MM/YYYY HH:MM:SS). In the real SOB, orders sent in the same second still arrive in a strict order, and each gets a unique, sequential timestamp (Matching Algorithm §4.3; market-spec §5.2). That order sets time priority at equal price (§6.1). In what order should the replay process rows that share a second?

**What is known.**
- **The files are not in arrival order.** `data/rawOMIE/orders_20250610.1` rows 4–25 are grouped by agent (all `ACCGR` rows, then `ACENH`). Inside a group the send times are out of order: row 4 is 09/06/2025 22:54:42, row 5 is 22:51:23, rows 6–9 are 10/06/2025 10:39:10, row 10 is 10:39:48, row 11 is 10:39:10 again. Several rows share one second (six `ACCGR` rows at 10:39:10). Neither file order nor the timestamp gives the true arrival order within a second.
- **No source settles it.** The corpus has no rule on row order in the public files. The second-resolution send time is all the data carries.
- **The simulator's convention (modelling choice, not a rule).** The loader sorts rows stably by send second: rows sharing a second keep file order, with session file D−1 before file D. The engine then gives same-price, same-second resting entries strictly increasing stamps in **processing order** (`OMIE-STAMP-UNIQUE-1`: NSEC rank on each resting add). Time priority among same-second rows is therefore file order, a proxy for arrival.

**Why it matters.** At equal price, the replayed queue among same-second orders may differ from the real queue. This can change which resting order a later aggressor fills. Neither the trade price nor the total traded quantity at a level changes. Uniqueness of the stamps (§5.2) is a hard rule and is met. Matching the real arrival order is not achievable from the public data.

**Re-open / resolve if:** OMIE publishes sub-second send times or an arrival sequence number, or a source states the row order of the public files. Until then any same-second ordering is a modelling choice, and tests that pin one must say so.

**SCOPE CORRECTION (2026-10-08, `OMIE-SPEC-STAMP-AMEND-1`; additive, the entry above is kept as written). The convention covers same-second entries per contract, across prices and sides, not per price.**
- **(1) Scope.** The stamp convention above applies to every same-second entry **of one contract**, the observable scope of the system-wide queue: at any price and on either side, not only among same-price entries. Source: Continuous Trading Matching Algorithm §4.3 (`Continuous_Trading_Matching_Algorithm_Public_Description.md:417-422`), "All timestamps are assigned sequentially", and §5.5 (`:612-615`), "all orders and all explicit capacity requests have unique timestamps, so that it is always possible to sort them by time stamp unequivocally"; market-spec v2.14 §5.2, "All timestamps across orders and explicit capacity requests are assigned from a single global queue (unique, sequential)". The order among same-second rows (file order as a proxy for arrival) stays a modelling choice, as above.
- **(2) Correction.** The sentence "Uniqueness of the stamps (§5.2) is a hard rule and is met" under *Why it matters* was wrong at v2.14. Uniqueness is a hard rule, but it was met only per price, not per contract (certification-log entry 25, C03 r2: same-second entries at different prices stored the same stamp). It is met once `OMIE-SLICE-STAMP-TAPE-SEC-1` merges.
- **(3) Queued for the next market-spec revision (not applied now; market-spec stays v2.14).** These are clarifications. v2.14 already states the rules the 2026-10-08 verdicts rest on: §5.2 "All timestamps across orders … single global queue (unique, sequential)"; §5.2 "activation of a new iceberg slice" gets a new timestamp; §4.2.1 Rule 4 eager activation and "always shows a slice". Texts, verbatim:
  - **(3a) §5.2:** *"Uniqueness binds the stored entry stamp: within one contract (the observable scope of the system-wide queue), no two orders, at any price or on either side, share a timestamp."* Source: Continuous Trading Matching Algorithm §4.3 (`:417-422`) and §5.5 (`:612-615`); market-spec v2.14 §5.2.
  - **(3b) §5.2:** *"A slice activated within a matching process takes its queue place at that moment: after the order whose matching process activated it, behind every order then resting at its price, and ahead of every order that arrives later (§5.2; Matching Algorithm §4.3, §5.5; Annex §4.4, 'each with its own creation time')."* Source: Continuous Trading Matching Algorithm §4.3 (`:420-422`, "Iceberg orders also receive a new time stamp whenever a new slice is activated") and §5.5 (`:611-615`); OMIE Rules Annex §4.4 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10265`, "each with its own creation time").
  - **(3c) §4.2.1 Rule 4, *Slice activation timing*:** delete the lazy alternative "or when subsequent matching draws against the remaining hidden quantity" from the Scope clarification (`market-spec.md:275`), and add: *"the new slice is activated by the trade that exhausts the shown quantity, inside that matching process (OQ-20 correction note (a))."* Source: Continuous Trading Matching Algorithm §3.3 (`:265-274`); OMIE Rules Annex §4.4 (`:10253-10255`); OQ-20 correction note (a).
- **Covers:** certification-log entry 25 (C03, E1 scope).

---

## OQ-33 — Trade split for an incoming iceberg: one trade per counter bid, or one per incoming slice

**Filed and resolved:** 2026-10-07 (`OMIE-SPEC-CERT-AMEND-1`). The answer was confirmed by the domain analyst at `OMIE-ICE-INCOMING-HOP-1` Step 2, recorded in `docs/omie/sprint-status.md`. **Status: RESOLVED — one trade per counter bid.** **Spec:** market-spec v2.14 §4.2.1 Rule 2, *Incoming iceberg* bullet, item (2).

**The question.** An incoming iceberg with a competitive price matches resting liquidity. Should the trades be split once per **incoming** slice, so that the incoming peak bounds each trade? Or once per **resting** counter-bid entry?

**Ruling: one trade per counter bid, where each instance of a resting iceberg is its own counter bid.**
- OMIE Rules Annex §4.4 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10256-10258`): "When entering an “Iceberg” bid with a competitive price, the quantity to be considered will be the total quantity of the “Iceberg” bid, recording a transaction for each counter bid that it matches with."
- When the counter bid is a resting iceberg, Annex §4.4 (`:10262-10267`) applies: "different transactions will be performed for each instance of the “Iceberg” bid, each with its own creation time".
- Matching Algorithm §5.1.3 (`:486-488`): an incoming order "may sequentially generate multiple transactions at different prices against multiple different orders that already existed in the order book". §5.3 (`:576-577`): "a trade is always between exactly two orders".
- So there is one trade row per resting entry matched. Two resting instances are never merged into one row, and an incoming peak boundary never splits a row.
- The alternative, one trade per incoming slice (peak-first), has no source support once the incoming order sweeps its total quantity.
- **Worked check** (from the S4 card): incoming BUY ICE with peak 2.0 against a resting SELL iceberg with peak 1.0 at 50.00 gives four rows of 1.0 @ 50.00, not two rows of 2.0.

**Re-open if:** a source is found that splits an incoming iceberg's transactions by its own peak.

**Revised by user ruling (2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1`; additive, the blocks above are kept as written). Rows also split at incoming instance boundaries.**
- **The ruling (user, 2026-10-20, binding):** *"The real market treats it as a chain of instances, each the size of the peak. Instance may only trade at its own limit, where is the increment. The sweep stops at the first instance that cannot trade. The leftover keeps its place in the ladder, and trade rows split where one instance ends and the next begins."*
- **The rule now.** The incoming iceberg is a chain of instances of size $P$ (the last may be smaller). Each trade is between one resting counter-bid entry and one incoming instance. A row ends where a resting entry ends **or** where one incoming instance ends and the next begins. Each resting NON order is one counter bid, each instance of a resting iceberg is its own counter bid, and two resting instances are never merged into one row. This holds for standard icebergs ($d = 0$) and for ICEwPI.
- **What is superseded.** The 2026-10-07 ruling above is superseded for incoming icebergs where it says that an incoming peak boundary never splits a row, and where it calls one-trade-per-incoming-instance unsupported. Its other half stands: no row spans two resting entries. The *Ruling* block above stays byte-identical as history; the notebooks that quote it are retired by hand at their next K2.
- **Sources.** The rule is compatible with the texts the 2026-10-07 ruling cited, read with the incoming iceberg as a chain of instances: Annex §4.4 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10253-10255`, an instance is "a new bid" of the reduced part; `:10256-10258`, "a transaction for each counter bid that it matches with") and §4.5 (`:10276-10278`, "each new instance will have a new price"); Matching Algorithm §5.3 (`:576-577`), "a trade is always between exactly two orders". The deciding input is the user's ruling plus the data below; no corpus text states the incoming-side split explicitly.
- **Evidence (`scripts/omie/incoming_ice_entry_evidence.py`, read-only, deterministic; run of 2026-10-20 over 317 session days, `orders_20250531.1` to `orders_20260414.1`, every orders file with its trades file).** Multi-instance entries ($M > P$) with a trade row larger than $P$: ICEwPI ($d \ne 0$) 0 of 11,104 (0 of 10,422 under the one-live-row filter); standard ($d = 0$) 10 of 129,398 (1 of 118,973 filtered). Row-split alignment (every multiple $jP < M$ is a cumulative row boundary, best-first order): $d \ne 0$ 93.97 % of all, 94.45 % filtered, 99.26 % filtered and domestic; $d = 0$ 99.93 %, 99.99 %, 99.99 % (cases where the subset search hit its budget are counted apart, not as aligned). Known positive `orders_20250610.1:125791` (GESTC01 SELL ICE $b = 0.00$, 8.5 MW, $P = 5$, $d = 1.00$): EGVD440's 5.0 MW prints as 4.80 + 0.20 at the incoming instance boundary; PASS on every stated attribute. The "one live order row" filter is defined in the script's docstring (*One-live-row filter*): the orders files carry no order IDs or cancellations, so an entry is kept only if no other orders row of the same unit, contract and side, sent earlier, could account for a print the table uses.
- **Simulator limit (not a market rule).** On a trade ring without room for the split rows, the rows are left unsplit (all or nothing; volumes and prices exact). Recorded under the existing saturated-ring certificate gap (OQ-15).
- **Engine.** No engine change in this story; `OMIE-ICE-INCOMING-LADDER-1` (E-C) implements the split.
- **Spec:** market-spec v2.14 §4.2.1 Rule 2, *Incoming iceberg*, item (2) (amended 2026-10-20).

---

## OQ-34 — Does an incoming ICEwPI sweep at its base price or at drifted prices?

**Filed:** 2026-10-07 (`OMIE-SPEC-CERT-AMEND-1`; open point raised at `OMIE-ICE-INCOMING-HOP-1` Step 2, nuance (v); user directive 2026-10-07). **Status: OPEN. The simulator's convention is recorded as a modelling choice.** **Spec:** market-spec v2.14 §4.2.1 Rule 2, *Incoming iceberg* bullet, item (6).

**The question.** An incoming iceberg with a non-zero price increment (ICEwPI) and a competitive price aggresses with its total quantity (OQ-33; Annex §4.4). Two readings are possible:
- **Reading 1:** the whole total sweeps at the **base** limit. The remainder then rests with its first slice at the base price.
- **Reading 2:** only the first peak may trade at the base limit, and later quantity is capped at the **drifted** limits $\mathit{base} + k\,\mathit{ppd}$, $k \ge 1$. A BUY drifts down and a SELL drifts up (§4.2; §13.6).

Under Reading 2 an incoming ICEwPI BUY could stop sweeping at a resting price it would still cross at its base limit.

**Evidence.**
- **For Reading 1 (Annex-literal).** Annex §4.4 (`:10244-10246`) defines the iceberg as total quantity, a reduced part, and "the requested price" (one price). Its competitive-price clause (`:10256-10258`) considers "the total quantity" with no price ladder. The increment rule, Annex §4.5 (`:10276-10278`: "each new instance will have a new price, which will be calculated as the price of the previous instance plus the price increment"), speaks of *instances*. Instances are created in the Bid Book (§4.4 `:10253-10255`), which an incoming order has not yet entered.
- **For Reading 2.** Matching Algorithm §3.3 (`:257-259`): iceberg orders' "full quantity is exposed to the market for matching, and which may come with a range of limit prices". This could be read as giving the post-peak quantity of an incoming ICEwPI its drifted limits during the sweep too. Batch matching §5.2.1 (`:515-519`) makes non-zero-PPD icebergs participate "with each slice individually", but only for orders already in the book.
- **Neither source addresses an incoming ICEwPI directly.**

**The simulator's convention (modelling choice, not a rule).** Reading 1. The sweep runs at the base limit. A remainder $R > 0$ rests as an iceberg showing $\min(\mathit{peak}, R)$ at the base price, and its drift restarts at slice 0 (`OMIE-ICE-INCOMING-HOP-1`, finding A4). Any test that pins incoming-ICEwPI behaviour pins this choice and must say so in its docstring.

**Re-open / resolve if:** an OMIE/SIDC source or an OMIE clarification addresses the price path of an incoming ICEwPI. Real data could also discriminate: an `ICE` row with PPD ≠ 0 whose same-second trades in the trades file print beyond its first-peak quantity at prices that only its base limit crosses.

**Resolved by user ruling (2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1`; additive, the blocks above are kept as written). Per-instance limits, continuation of the ladder, display C3 derived from data.**
- **The ruling (user, 2026-10-20, binding):** *"The real market treats it as a chain of instances, each the size of the peak. Instance may only trade at its own limit, where is the increment. The sweep stops at the first instance that cannot trade. The leftover keeps its place in the ladder, and trade rows split where one instance ends and the next begins."*
- **Resolution.** Every incoming iceberg, standard and ICEwPI, is a chain of instances. With $b$ the price, $P$ the peak, $d$ the increment ($d = 0$ for a standard iceberg) and $M$ the swept quantity:
  - **Per-instance limits.** Instance $k$ trades only at prices its own limit $b + k\,d$ crosses. The sweep stops at the first instance that cannot trade, even if $b$ would still cross. Trades stay priced by the resting entry (OQ-31). This is Reading 2 above, completed by the stop rule.
  - **Continuation.** A remainder $R > 0$ keeps its place in the ladder: it rests as instance $k_e = \lfloor M/P \rfloor$ at $b + k_e\,d$, and later instances continue from there. The drift does not restart at slice 0.
  - **Display C3 (derived from data).** With $u = M - k_e P$, the rested instance shows $v_1 = \min(P - u, R)$, its unused part, and $R - v_1$ is hidden. This is "at most the reduced part" (Annex §4.4); the sources do not state the display of a part-used instance, so this choice is derived from the data below. The alternative C1 (a full $\min(P, R)$) is rejected by the data.
  - The simulator's convention above (Reading 1: base-limit sweep, drift restarted at slice 0) is superseded. The block stays byte-identical as history; the notebooks that quote it are retired by hand at their next K2.
- **Sources.** Annex §4.5 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10276-10278`): "each new instance will have a new price, which will be calculated as the price of the previous instance plus the price increment". Annex §4.4 (`:10256-10261`): the total quantity is considered, and the remainder shows "at most the reduced part … even if the matched quantity was not a multiple of the reduced part". Matching Algorithm §3.3 (`:257-259`): the full quantity "may come with a range of limit prices". §5.1.3 (`:486-493`): matching continues while the incoming order "remains executable", then it enters the book "with its remaining quantity".
- **Evidence (`scripts/omie/incoming_ice_entry_evidence.py`, read-only, deterministic; run of 2026-10-20 over 317 session days, `orders_20250531.1` to `orders_20260414.1`).** Population: 16,806 ICEwPI entries crossed on entry (11,104 multi-instance, $M > P$) and 162,589 standard ones. Every exclusion is reported by $\lvert d \rvert$ bucket in the script's section 7; 4,421,281 column-truncated or unparsable orders lines in 10 files and 330 trades lines are skipped and counted. "Filtered" below means the one-live-row filter, whose definition is in the script's docstring (*One-live-row filter*: an entry is kept only if no other earlier orders row of the same unit, contract and side could account for a print the table uses, since the files carry no order IDs or cancellations).
  - **Per-instance limits (sweep-limit table).** Entries with a print worse than its instance limit $b + k\,d$: 971 of 16,806 (5.78 %) unfiltered; 787 of 15,434 (5.10 %) filtered; 70 of 9,548 (0.73 %) filtered and domestic (no blank cross-border counterparty). The residual is recorded, not explained away; it concentrates where an unobserved counter-source (cross-border, another live row) is possible.
  - **Continuation (step $j$ of the first later passive print, $k_e > 0$).** Filtered: $j = k_e$ in 5,507 of 5,509 (99.96 %), $j \ge k_e$ in 99.98 %; $j = 0$ (the old restart) in 1 case. Unfiltered: 6,773 of 6,809 (99.47 %). The architects' pilot-comparable figure: $j \ge k_e$ in 99.52 % of 6,808.
  - **Display C3 (decisive cases, $u > 0$, $j = k_e$).** Filtered: C3 2,697 of 2,699 (99.93 %), C1 0. Unfiltered: 3,152 of 3,159 (99.78 %), C1 3. Pilot-comparable: 2,562 of 2,567 (99.81 %). Every $\lvert d \rvert$ bucket is at or above 99.90 %. Standard icebergs ($d = 0$), filtered: C3 only 3,246 against C1 only 3 of 3,249 decisive (99.91 %). The corrected C3 share is above the card's 90 % bar, so C3 stands.
  - **Known positive `orders_20250610.1:125794`** (GESTVD2 SELL ICE $b = 110.00$, 100 MW, $P = 5$, $d = +0.75$): $M = 20$, $k_e = 4$, $u = 0$, no row above $P$, no step-limit violation, first later passive print on step $j = 4$ (113.00). PASS on every stated attribute; `orders_20250610.1:125791` likewise PASS (OQ-33 block of 2026-10-20).
  - **Depth.** ICEwPI instances used on entry: median 2, max 50. Ladder depth $\lceil Q/P \rceil > 64$: 89 of 16,806 (max 124).
- **Engine.** No engine change in this story; `OMIE-ICE-INCOMING-LADDER-1` (E-A sweep limits, E-B remainder) implements it. Until then the engine follows the superseded convention.
- **Spec:** market-spec v2.14 §4.2.1 Rule 2, *Incoming iceberg*, items (4) and (6) (amended 2026-10-20); Rule 3, *ICE aggressor*. Related: OQ-35.

**Display C3 confirmed by the user (2026-10-22; additive, the blocks above are kept as written).**
- **The confirmation (user, 2026-10-22):** *"2. confirm"*, in reply to item 2 of the cert fix v3 run report: "Confirm the display choice C3, $v_1 = \min(P - u, R)$ … derived from data (99.93 % of decisive cases) and not stated by the regulatory sources".
- **What is confirmed.** The remainder of an incoming iceberg rests as instance $k_e = \lfloor M/P \rfloor$ and shows $v_1 = \min(P - u, R)$, with $u = M - k_e P$: a part-used instance shows only its unused part, and $R - v_1$ is hidden.
- **Status of the rule.** The display stays **derived from OMIE data** (the 2026-10-20 evidence above), not stated by the regulatory sources; the user's confirmation accepts it as the simulator's rule without turning it into a source-stated one. market-spec v2.14 §4.2.1 Rule 2, *Incoming iceberg*, item (4) is unchanged.

---

## OQ-35 — A drifted instance or slice price beyond ±9,999 EUR/MWh

**Filed:** 2026-10-20 (`OMIE-SPEC-ICE-INCOMING-AMEND-1`; the analyst's open edge from the 2026-10-19 ICEwPI review, `sprint-status.md` "Final check (user, 2026-10-19)"). **Status: OPEN, question only. No engine change and no modelling choice is ruled here.** **Spec:** market-spec §2.3 (price bounds), §13.6 item 2.

**The question.** An ICEwPI's instance $k$ is priced $b + k\,d$ (Annex §4.5). For a SELL with $d > 0$ or a BUY with $d < 0$, a deep enough instance can fall beyond the SIDC limits of ±9,999 EUR/MWh. What does the market do with such an instance: clamp its price to the limit, end the iceberg (cancel or expire the rest), refuse the order at entry, or something else?

**What is known.**
- §13.6 item 2 states, as a derived (not explicit) constraint, that every slice price must remain within the harmonized range. No corpus text says how the SOB enforces it during regeneration. §2.3 gives the limits only as bid-rejection limits at submission.
- **The simulator today (observed, not a ruling):** drifted slice prices are not clamped. The analyst's probe of 2026-10-19 shows a slice resting at 10,000.00, where no valid counter bid can cross it, so it cannot trade. This conflicts with §13.6 item 2.
- The 2026-10-20 instance chain (OQ-34) makes the edge slightly more reachable: an incoming iceberg's remainder now starts at instance $k_e$, not at slice 0.

**Why it matters.** Only extreme prices or very deep ladders are affected; the replay data have not been searched for such a case. It is recorded as a certificate gap so that the seal states it.

**Re-open / resolve if:** an OMIE/SIDC source or an OMIE clarification states how an out-of-range iceberg instance is handled, or the data show an ICEwPI whose ladder reaches the limits.

---

*Last updated: 2026-10-22 (OQ-34: display C3 confirmed by the user; previously 2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1`: OQ-33 revised, OQ-34 resolved, OQ-35 filed; previously 2026-10-07, `OMIE-SPEC-CERT-AMEND-1`: OQ-20 correction note, OQ-31 addendum, OQ-32–OQ-34 filed). Update as ambiguities are resolved or new questions arise.*
