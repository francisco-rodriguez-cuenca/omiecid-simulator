# OMIE Continuous Intraday Market — Implementation-Ready Market Specification

**Status:** v2.14
**Date:** 2026-10-07
**Owner:** Domain Analyst (`/domain-analyst`). This document must only be modified by the Domain Analyst role.
**Source corpus:** `docs/omiecid/corpus/` (all subdirectories)

### Version history

| Version | Date | Changes |
|---------|------|---------|
| v2.14 | 2026-10-07 | `OMIE-SPEC-CERT-AMEND-1` (certification fix pass, S9). **Additive only:** every amendment is a new dated bullet or note, and no byte of an existing passage changes, so every passage quoted by a certification notebook stays byte-identical. **§5.4:** two new rejection conditions for icebergs. (a) The ICEwPI sign rule: a BUY with PPD > 0 or a SELL with PPD < 0 is rejected (Annex §4.5, `:10280-10281`; §13.6 item 1). (b) A peak larger than the total quantity is rejected (Annex §4.4, `:10268-10270`). **§4.2.1 Rule 2:** new *Incoming iceberg* bullet. An incoming iceberg aggresses with its **total** quantity. It records one transaction per resting counter-bid entry, and each instance (slice) of a resting iceberg is its own counter bid. Any remainder rests as an iceberg showing $\min(\mathit{peak}, R)$ (Annex §4.4, `:10256-10261`; Matching Algorithm §3.3, §5.1.3). The incoming-ICEwPI price is filed as OQ-34; the simulator's base-limit sweep is a modelling choice. Also a new *Re-ranking* bullet: subject to §6.1, each regenerated slice is ranked as a new entry at its own (drifted) price. **§4.2.1 Rule 3:** new *ICE aggressor* bullet. The iceberg that was resting sets the hop price, also when the aggressor is an iceberg. This is the OQ-31 addendum, which the architect countersigns. **§4.2.1 Rule 4:** new *Trade timestamps* bullet, per Matching Algorithm §5.3. Each trade's timestamp is the moment its pair is found executable. The trades of one matching process need not share a single timestamp value. Also a new *Slice activation timing* bullet: activation is eager (OQ-20 correction note). **§9.2:** a note adds **side** to the trade-book fields (spec only, no engine change filed). New OQs: OQ-32 (same-second arrival order in replay, open, modelling choice recorded), OQ-33 (incoming-ICE trade split, resolved: one trade per counter bid), OQ-34 (incoming-ICEwPI sweep price, open, modelling choice recorded). OQ-20 gets a correction note; OQ-31 gets an addendum. **Amendment (2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1`, certification fix pass v3, S1; user ruling 2026-10-20). This amendment is NOT additive.** §4.2.1 Rule 2 *Incoming iceberg* items (2), (4) and (6) and its *Classification* were rewritten, and the superseded sentences were deleted, under the kept v2.14 token. So this row's "Additive only … no byte of an existing passage changes" no longer holds for §4.2.1 Rule 2 *Incoming iceberg*. Also: the bullet's lead-in gets a dated clause, a *Sources of the 2026-10-20 amendment* sub-bullet is added, and the first sub-bullet of Rule 3's *ICE aggressor* bullet is rewritten to name the instance chain and the rest price $b + k_e\,d$. Items (1), (3) and (5) are byte-identical. The new rule: every incoming iceberg, standard ($d = 0$) and ICEwPI, is a chain of instances of size $P$ (the last may be smaller). Instance $k$ has limit $b + k\,d$, and the sweep stops at the first instance that cannot trade. The remainder keeps its place in the ladder as instance $k_e = \lfloor M/P \rfloor$ ($M$ = swept quantity) and shows $v_1 = \min(P - u, R)$ with $u = M - k_e P$. Trade rows split at resting entries and at incoming instance boundaries. Open questions: OQ-33 revised, OQ-34 resolved, OQ-35 filed (question only). The superseded texts are recorded in the OQ-33 and OQ-34 blocks of 2026-10-20 and in `sprint-status.md`. |
| v2.13 | 2026-10-02 | **OQ-31 RESOLVED (Reading B): spec clarification, no market-rule change.** **§4.2.1 Rule 3:** new *Trade price of the hop* bullet. The aggressor's resting remainder can be matched by the regenerated iceberg slice, but it does not set the price. The hop trades at the **iceberg slice's limit** (base price, or $\mathit{base} + k\,\mathit{ppd}$ for ICEwPI), never at the aggressor's limit $L$. The iceberg is the passive side for price determination. Evidence: OMIE Rules Iceberg §4.4 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10262-10267`; Spanish `CNMC_DCOOR_DE_009_24.md:14819-14824`), Continuous Trading Matching Algorithm §5.1.2 (`:475-479`) and §5.1.3 (`:490-493`). The *Real market rule* and *Classification* bullets are aligned. **§6.2 item 7:** now cross-references the Rule-3 hop. §5.2 and §6.2 item 7 already implied this ruling; Rule 3's "passive liquidity" wording was the only text that read otherwise. Unblocks C8's K2 and I5's Rule-3 branch (`OMIE-CERT-INVARIANTS-1`). |
| v2.12 | 2026-10-01 | `OMIE-TRADES-MATCH-TS-HEADER-1` domain ruling (data-format reference; **no market-rule change**). New source: OMIE *Modelo de Ficheros para la distribución pública de Información del mercado de electricidad* v1.38 (22/09/2026), §3.2, §5, §5.1.1.1, §5.3.2.7, §5.3.3.1. **§13 general format:** new *Number format* block. Orders and trades use Spanish numbers (`.` thousands, `,` decimal, e.g. `-9.999,00`). `marginalpdbc` is DOT-decimal (`36.07`; 0 cells with `,`), the explicit exception allowed by §3.2. Also: header names are accented in Latin-1. CORRECTION: row 2 is blank and the column headers are on row 3, not row 2. **§13.1 / §13.2:** `Precio` / `Cantidad` notes now name the Spanish format. `Momento casación` is the accented header spelling, and the full trades header is recorded. **New §13.2.1:** a date-only `Momento casación` cell `DD/MM/YYYY` (332 cells in 77 trades files, each equal to the file date; 0 explicit `00:00:00` in trades, whereas orders write `00:00:00` and are never date-only) **means 00:00:00 of that date**, for trades only. The trades match time is CONFIRMED to lie in the product frame [55,200, 165,600) s from D−1 00:00, with $t < \mathrm{GCT}(k)$. A date-only cell maps to 86,400 s, which is inside every product carrying one ($k \ge 8$). No new OQ. |
| v2.11 | 2026-09-30 | `OMIE-TAPE-CONTRACT-DATE-1-DOC-1` (doc hygiene; **no rule change**). Dated `CORRECTION (2026-09-30, OMIE-TAPE-CONTRACT-DATE-1)` notes refresh stale data examples and snapshots after the user added `orders_20250531.1` / `trades_20250531.1` on 2026-09-30. **§13.4 *Completeness*:** the 2025-06-01 example is withdrawn, since that product is now complete (2,546 order rows on contract 0: 1,731 in `orders_20250531.1` + 815 in `orders_20250601.1`). It is replaced by 2025-05-31 (no `*_20250530.1`). **§14.5 *No-assume-data*:** the v2.8 "trades ONLY for 2025-06 / ~105 order-days in four bands / only June 2025 clears §14.4" snapshot is superseded by a dated snapshot verified on disk: session files are continuous from 2025-05-31 through 2026-04-14 (318 trades files, 317 orders files; gaps `orders_20260228.1` and `orders_/trades_20260406.1`). Also records that `docs/data/omie-layout.md` (coverage matrix, file-date wording) is stale. `open-questions.md` OQ-29 snapshot refreshed to match. |
| v2.10 | 2026-09-30 | `OMIE-TAPE-CONTRACT-DATE-1` Step 0 (spec error correction; no market-rule change). **§13 / §13.1 / §13.2:** the file date and the `Fecha` column are the **SESSION day**, not the delivery day. File `{orders,trades}_D.1` holds every row sent (orders) or matched (trades) in the local session [22:20 D−1, 22:00 D), for products of delivery day D **and** D+1. **§13.4:** a delivery product is **(delivery date, slot)**, taken from the `Contrato` start, and product (D, k) is split across session files **D−1 and D ONLY**. The architect-derived "plus file D+1 when k ≥ 93" rule is **REFUTED** by the Rulebook schedule and by the raw data: contracts 93–96 (1-indexed) of D close at 22:00 on D and are not traded after it (OQ-27). **§3.1 correction:** the Iberian continuous GCT is aligned to the delivery hour. Each hourly round closes four consecutive quarter-hour contracts at (delivery hour start − 1 h), so GCT(k) = 60·⌊k/4⌋ − 60 min from D 00:00 (0-indexed k). It is not 60 min before each quarter-hour start. **§3.3:** added the note that orders survive the IDA2 (and IDA3) clearing window and are batch-matched at reopen, with the replay simplification (OQ-28). **§11.1** GCT row, **§14.2** and **§14.4** aligned. New OQs: OQ-27 (product = date + slot, file partition), OQ-28 (IDA survival and reopen batch matching), OQ-29 (incomplete products are excluded), OQ-30 (DST, open). OQ-10 extended (post-IDA re-quote double-count). OQ-14 annotated. |
| v2.9 | 2026-09-29 | Spec clarification (not a new OQ; motivates `OMIE-NON-REMAINDER-TS-1`): §5.2 now states explicitly that partial execution does NOT assign a new timestamp — the NON incoming residual enters the book with its original entry timestamp, and a partially executed resting order keeps its timestamp (Continuous Trading Matching Algorithm §4.3, §5.1.3). §6.2 item 4 cross-references §5.2. No market-rule change. |
| v2.8 | 2026-07-14 | `P3-DATA-OMIE-EXPAND-1` data-reality correction: §14.5 *No-assume-data* bullet's stale "~30 days across 2025-06 / 2025-10 / 2025-12" on-disk figure replaced with the VERIFIED (2026-07-14) coverage — ~**105 order-days** across FOUR bands (2025-06 30d, 2025-10 31d, 2025-12 31d, PARTIAL 2026-01 13d) but **trades ONLY for 2025-06 (30 trade-days)**. Made the orders-vs-trades asymmetry explicit and recorded that trades are the **binding constraint** gating the replay `match_rate` diagnostic and the §14.4 non-degeneracy check (only June 2025 clears §14.4 today). Cross-references the canonical coverage matrix in `docs/data/omie-layout.md` and `docs/cid_rl/experiment-plan.md` §"Data reality". Additive correction only — no §14 restructuring, no market-rule change. |
| v2.7 | 2026-07-09 | OQ-23 disposition (`P3-PERF-REGEN-MASK-1` consulting dispatch): added a cross-reference in §4.2.1 Rule 4 *Scope clarification* recording that the latent `regenerate_slices` `dirty`-invisible cross-slot peak-consumption gap is CONSTRUCTIBLE on valid inputs but is NOT a Rule 4 violation (resting book stays uncrossed; intra-call trade emission preserved) — it is the OQ-20-class engineering invariant, warranting a scoped engineering follow-up (no market-spec rule change). No semantic change to Rule 4. |
| v2.6 | 2026-07-03 | DA-TAPE-2026-07 (`P3-PERF-TAPE-1` ruling): added §14 (contract order-flow magnitude and tape-selection semantics). Negative finding — the corpus has no per-contract liquidity/volume ranking and no "liquidity concentrates near GCT" rule (§14.1 → filed OQ-21). Ruled the spec-grounded proxy for a *shorter* tape: GOT→GCT window length is a strictly increasing function of the day-D delivery time, so shortest-window = earliest-delivery = smallest `contract_id` (§14.2); the defensible tape-selection axes are contract-id selection (primary, available today) and a head (first-K-by-submission-time) per-tape order cap (secondary, new loader arg), with a tail "final-N-minutes-before-GCT" truncation explicitly NOT domain-defensible (§14.3); the non-degenerate-fills bar for locked test (2) is ≥1 historical trade + both sides present in the tape + non-zero scripted-rollout fills (§14.4); additivity + no-assume-data developer constraints (§14.5). |
| v2.5 | 2026-05-28 | P2-PERF-12 scope ruling: added Rule 4 *Scope clarification* sub-bullet in §4.2.1 — Rule 4 governs the resting order book (intra-call fixed point + no crossed book left between aggressors), not simulator-internal side-channel cleanliness; an active iceberg slot whose `order_id` is absent from the LOB (peak fully consumed, next slice not yet activated) is a permitted between-aggressor state under the matching algorithm. Filed OQ-20 to track the simulator-internal cleanup invariant as an engineering concern. |
| v2.4 | 2026-04-22 | DA-PPD-2026-04: Added authoritative §13.6 establishing that ICEwPI PPD has no rule-level numeric magnitude bound (only sign rule is hard). Revised §13.3 PPD bullet to mark empirical range as non-authoritative and reference the 2026-01-12 counter-example (`PPD = -3.00`). Opened OQ-13. |
| v2.3 | 2026-04-15 | SD-1: Revised §13.4 zone-filtering instruction to reflect Phase 1.5-Z removal of zone filtering. SD-2: Added §8.3 (per-unit settlement granularity as hard rule, citing OMIE Rulebook §50.1); renumbered subsequent subsections. |
| v2.2 | 2026-03-26 | Phase 1.5 Unidad promotion: DA-UNIDAD finding in §13.5; unit field requirements in §13.4. |

---

## 1. Glossary

| Term | Definition |
|------|------------|
| **SIDC** | Single Intraday Coupling — the pan-European continuous intraday market framework (formerly XBID project). Governed by CACM Regulation (EU) 2015/1222 and ACER Decision 11/2024. |
| **XBID** | Cross-Border Intraday — legacy name for the SIDC technical platform. Still used in operational documents. |
| **SOB** | Shared Order Book — central matching module within XBID/SIDC that holds all orders across all bidding zones. |
| **CMM** | Capacity Management Module — XBID component managing available transmission capacity between bidding zones. |
| **LTS** | Local Trading Solution — each NEMO's (e.g. OMIE) front-end interface through which market participants submit orders to the SOB. |
| **NEMO** | Nominated Electricity Market Operator — a power exchange designated to run SIDC in a given bidding zone. OMIE is the Iberian NEMO. |
| **MIC** | Mercado Intradiario Continuo — OMIE's name for the continuous intraday market (same as SIDC continuous session). |
| **MIx (MI1–MI6)** | MIBEL regional intraday auction sessions (auction-based, not continuous). MI6 was the last CRIDA session, replaced by pan-European IDAs from 13 June 2024. |
| **IDA1/IDA2/IDA3** | Pan-European Intraday Auctions — periodic implicit auctions within SIDC that allocate cross-zonal capacity by price; separate from the continuous trading session. Go-live: 13 June 2024. |
| **CRIDA** | Complementary Regional Intraday Auction — the legacy Iberian regional auction sessions (MI1–MI6). Superseded by IDAs. |
| **Contract** | The tradable object: a specific delivery period (e.g. "Q37 of 2024-06-14", i.e. 09:00–09:15 CET) on a given product. Each contract has its own order book. |
| **Product** | A template specifying delivery duration (15-min, 30-min, 60-min, user-defined block). Source: XBID Info Package §4; ACER Decision 03/2025. |
| **Delivery Period** | The physical time window during which energy must be delivered for a given contract. |
| **MTU** | Market Time Unit — the delivery granularity. For Spain/Iberia: 15-min MTU products active since 18 March 2025 (OMIE go-live). 96 quarter-hourly contracts per delivery day. Source: OMIE Market Rules Annex 2, §9841–9857; ACER Decision 03/2025. |
| **GOT** | Gate Opening Time — when order submission opens for a given session or contract. |
| **GCT** | Gate Closure Time — when order submission closes. For continuous contracts: 1 hour before start of delivery period. |
| **ATC** | Available Transfer Capacity — maximum tradable cross-border capacity on a given interconnector in a given direction and period. |
| **NON** | No execution restriction — resting limit order. |
| **IOC** | Immediate or Cancel — fill what is available immediately, cancel remainder. |
| **FOK** | Fill or Kill — fill entirely or cancel entirely. |
| **ICE** | Iceberg execution restriction — used in the OMIE/XBID data feed to identify iceberg orders. Behaves like NON (resting, partial fills) but tracks hidden total quantity and regenerates visible peak slices. |
| **AON** | All or Nothing — fill entirely against one counterorder or enter the book; only applicable to user-defined block orders. |
| **GFS** | Good For Session — order valid until contract's GCT. Default validity. |
| **GTD** | Good Till Date — order valid until a participant-specified time, no later than GCT. |
| **ICEwPI** | Iceberg with Price Increment — iceberg variant where each new slice price is offset by a fixed delta. |
| **Block order** | An order spanning multiple consecutive quarter-hourly contracts. Always AON. |
| **Basket order** | A set of simultaneous orders on possibly different contracts. Can be independent (each treated separately), all-or-nothing, or linked (all FOK). |
| **Round** | OMIE operational concept: the segment of continuous trading between two consecutive contract closures. Rounds are numbered sequentially within a delivery day. |
| **PHFC** | Final Continuous Program Schedule — the programme published by OMIE to TSOs at end of each round, updating their net position and cross-border capacity values. |
| **PIBCI** | Basic Intraday Program of Incremental Matching — incremental results from each auction session sent to TSOs. |
| **PHF** | Final Schedule Program — consolidated schedule from TSOs, incorporating PIBCI and PHFC. |
| **PDBF/PDVP** | Day-Ahead and technical unit availability declarations published by TSOs. Required before IDA1 GOT. |
| **Batch matching** | Matching process triggered by a cross-border capacity increase (capacity update event), not by a new order timestamp. |
| **Regular matching** | Matching process triggered by an order with a new timestamp. |
| **Price-Time Priority** | The matching rule: best price first; ties broken by earliest timestamp. |

---

## 2. Tradable Products and Contract Granularity

**Source:** XBID Information Package §4; ACER Decision 03/2025 Annex I; Continuous Trading Matching Algorithm Public Description §3.1; OMIE Market Rules Annex 2 §9841–9857.

### 2.1 Products supported in SIDC

| Product | Delivery Duration | MTU | Status for Iberia (Spain) |
|---------|------------------|-----|--------------------------|
| 15-minute (Quarter-hourly) | 15 min | 15 min | **Active since 18 March 2025** (OMIE MTU15 go-live, first delivery 19 March 2025). Primary product. |
| 60-minute (Hourly) | 1 hour | 60 min | Superseded by 15-min for CID; still used in Day-Ahead. |
| 30-minute | 30 min | 30 min | Not active for Spain/Iberia CID. |
| User-Defined Block | Multiple consecutive 15-min periods | 15 min | Active; minimum = 2 MTUs (30 min). Defer for v1. |

OMIE Market Rules Annex 2 §9841–9857 state: "The quarter-hourly product for the continuous intraday market will be composed of 96 contracts… each contract corresponds to each of the quarter-hourly periods of the day." ACER Decision 03/2025 governs the updated SIDC product framework at EU level; Spain's activation pre-dates this decision (framework harmonization, not trigger).

> **Implementation note:** The v1 simulator must model **96 quarter-hourly (15-min) contracts per delivery day**. Hourly contracts are no longer the active CID product for Spain.

### 2.2 Contract generation

- **96 contracts per delivery day**: one per 15-minute period (Q1–Q96, covering 00:00–24:00 CET).
- All 96 contracts for delivery day D+1 open **simultaneously at 15:20 CET on day D** after IDA1 completion (Round 18). OMIE Instruction 1/2024 §7: "continuous trading for delivery day 14/06/2024 will open at 15:20h (13/06/2024) starting with Round 18th, after successful completion of IDA1." OMIE Market Rules Annex 2 schedule table confirms: at D period 16 (15:20–16:00), continuous round 18, all contracts 1–96 for D+1 open simultaneously.
- Contracts close on a **rolling basis**: approximately 4 contracts close per hour-long "round", each closing ~1 hour before its delivery period. The schedule table shows contracts closing one round at a time (each round covers a 1-hour wall-clock window, closing 4 quarter-hourly contracts per round).
- Contracts are generated automatically by the SOB according to a predefined trading schedule. Source: Continuous Trading Matching Algorithm §1.2.1.
- Each contract has its own independent order book.
- Contracts from different products are not cross-matched.
- Block order contracts span consecutive delivery periods of the same product (minimum 2 MTUs = 2 × 15 min).

### 2.3 Price and quantity constraints

- **Trading unit:** MW (megawatt).
- **Currency:** EUR.
- **Trade value formula:** `V (EUR) = Q (MW) × p (EUR/MWh) × d (h)` where d = 0.25 h for 15-min contracts. Source: Continuous Trading Matching Algorithm §5.3.
- **Price bounds:** −9,999 EUR/MWh (minimum) to +9,999 EUR/MWh (maximum). These are the hard SIDC bid-rejection limits established by OMIE Market Rules Annex 3 (pp. 208–209) as the harmonized SIDC clearing price limits set per the ACER-approved Single Intraday Coupling methodology.
- **Price tick (minimum increment):** 0.01 EUR/MWh (2 decimal places maximum). OMIE Rules §38.2: prices expressed "in euros per MWh, with a maximum of two decimal places."
- **Quantity bounds:** 0.1 MW (minimum) to 999 MW (maximum). OMIE Rules §45.2.1: maximum power amount "pursuant to the provisions of the European Continuous Trading Platform." XBID matching algorithm §3.1 confirms products have a minimum and maximum contract quantity.
- **Quantity tick (minimum increment):** 0.1 MW (1 decimal place maximum). OMIE Rules §38.2: "power amount in MW… rounded to a maximum of one decimal place."
- Order quantity and price must pass OMIE system validations at submission time. The economic value of the order must not exceed the agent's operating limit (guarantee) deposited at OMIE.

### 2.4 Price notification thresholds

In addition to the hard bid-rejection limits, OMIE Market Rules Annex 3 defines **notification thresholds** (softer warnings, not rejections) for the MIBEL zone:
- Minimum notification threshold: ~−150 EUR/MWh
- Maximum notification threshold: ~+1,500 EUR/MWh

These thresholds may be modified by OMIE directive. They do not prevent order acceptance but may trigger warnings. For the simulator, only the hard ±9,999 EUR/MWh limits are functionally relevant.

> **Caveat on price bounds:** The ±9,999 EUR/MWh limits are SIDC harmonized values as of 2026-03-23. Annex 3 explicitly states current price limits are dynamic and "can be consulted on the agent website." These values may be revised under the harmonized SIDC methodology.

---

## 3. Session Timing and Gate Opening/Closure

**Source:** OMIE Details of Intraday Market Operation §3; OMIE Instruction 1/2024; XBID Information Package §5, §8; OMIE Market Rules Annex 2 schedule table (pp. 200–202).

### 3.1 Continuous market gate times

- **24/7 operation:** The continuous market runs 24 hours per day, 7 days per week. There is no daily market open/close analogous to an equity exchange session.
- **Per-contract gate closure (GCT):** 1 hour before the start of the delivery period. Orders on a contract are automatically cancelled at GCT if unmatched.
  - **CORRECTION (v2.10, `OMIE-TAPE-CONTRACT-DATE-1`; hard rule, confirmed in the data).** In the Iberian continuous market, GCT is aligned to the **delivery hour**, not to each quarter-hour. The continuous market runs in 24 hourly rounds, and "each closing round of the continuous intraday market will in turn entail the closing of trading for four consecutive contracts" (OMIE Rules Annex 2 §2, "Intraday continuous market products and schedules", and the Figure 4 schedule table: in each hourly round the next four contracts leave the open set together). So the four quarter-hours of delivery hour h (0-indexed k = 4h … 4h+3) all close at **(h − 1):00**. With 0-indexed k counted from D 00:00, **GCT(k) = 60·⌊k/4⌋ − 60 min**. The rule is exact at 60 min before delivery only for the first quarter-hour of each hour. The later quarter-hours close 75, 90 and 105 min before their delivery start.
    - **Data evidence (`data/rawOMIE/`).** On 2025-06-25, contract 00:45 (k = 3): 0 orders and 0 trades after 23:00:00 on 06-24, and sends up to 22:59:10. Contract 01:00 (k = 4): 118 sends in 23:50–23:59 on 06-24 and 0 after 00:00. Contract 12:45 (k = 51): 86 sends in 10:50–10:59 on 06-25 and 0 in 11:xx. The same shape holds on 2026-01-10 (k = 51: 79 sends in 10:50–10:59, 0 in 11:xx). Contracts 23:15/23:30/23:45 of 06-25: last sends at 21:59:29 on 06-25, and 0 orders or trades after 22:00.
    - **Consequence.** The last four contracts of D (0-indexed k = 92–95) close at **22:00 on D**, which is the IDA2 gate closure for D+1. No product of day D is traded after 22:00 on D.
    - **Scope note.** The public files cover Iberian participants' orders and trades, including cross-border trades with one side blank (§13.2). Whether non-Iberian SIDC participants may still trade these quarter-hours cross-border after the local round closes is not observable in this data and does not affect the Iberian replay (OQ-27).
- **Per-contract gate opening (GOT):** All 96 contracts for delivery day D+1 open simultaneously after IDA1 completion at approximately 15:20 CET on day D. Specific GOTs are defined per border and product in the SIDC configuration; the 15:20 CET value applies to the Iberian zone. (Data check, v2.10: file `orders_20250624.1` has 0 rows of product 06-25 sent before 15:20 on 06-24 and 5,623 sent in 15:20–15:29.)

### 3.2 IDA sessions (pan-European, from 13 June 2024)

There are exactly **3 IDA sessions per delivery day** under the current SIDC/OMIE configuration. No IDA4+ sessions exist. Source: OMIE Instruction 1/2024 §6, §8, §9; OMIE Market Rules Annex 2.

| Session | GOT (CET) | GCT (CET) | Results | Scope (quarter-hourly periods) |
|---------|-----------|-----------|---------|-------------------------------|
| IDA1 | ~14:00 (D−1) | ~15:00 (D−1) | ~15:20 (D−1) | Periods 1–96 of D+1 (full day) |
| IDA2 | ~21:00 (D−1) | ~22:00 (D−1) | ~22:20 (D−1) | Periods 1–96 of D+1 (full day) |
| IDA3 | ~09:00 (D) | ~10:00 (D) | ~10:20 (D) | Periods 49–96 of D+1 (hours 13–24 only) |

IDA3 covers only the second half of the delivery day (periods 49–96) because periods 1–48 (hours 1–12) have already passed or are imminent by 09:00 on the delivery day. For those periods, only continuous trading serves. There is no IDA4, IDA5, or morning session for hours 1–12 on the delivery day.

> **Note:** These are nominal timings from OMIE Instruction 1/2024. Actual GOTs depend on successful completion of the Day-Ahead session and TSO capacity publications. IDA sessions are auction-based (not continuous); they use batch-matching price determination (arithmetic mean of last matched pair). IDA capacity allocation updates feed back into continuous market ATC values.

### 3.3 Continuous market rounds and IDA suspension model

OMIE internally segments continuous trading into "rounds":
- A round opens when the previous contract's trading closes, and closes when the next contract's trading closes.
- Rounds are numbered sequentially. Round 18 is the first round of a delivery day under the IDA1 regime (following IDA1 completion at ~15:20 CET D−1).
- Round numbering is an OMIE-operational concept. The underlying SOB is continuous; rounds do not introduce gaps or halts in normal operations.

**Two-phase IDA suspension model:**

Each IDA session causes two distinct suspension phases affecting continuous trading:

**Phase 1 — Pre-IDA capacity halt (GCT − 20 min):**
- Cross-border capacity allocation is halted. The XBID CMM stops accepting new implicit capacity allocations from cross-border matches.
- **Local (intra-zone) order submission and matching are NOT suspended.** Agents continue to submit orders and local matches can still execute.
- Example: IDA2 capacity halt begins at ~21:40 CET. Source: OMIE Instruction 1/2024 §8: "Continuous capacity allocation during Round 24th will be halted at 21:40h."

**Phase 2 — IDA clearing window (GCT to results, ~20 min):**
- **Both regular matching and explicit allocation are fully suspended.** Source: Continuous Trading Matching Algorithm §5.5: "As soon as a batch matching round starts, both regular matching and explicit capacity allocation are suspended."
- **Order submission is also suspended** for affected contracts. Source: OMIE Instruction 1/2024 §8: "Continuous allocation (and order submission) will continue with Round 1st (14/06/2024) at 22:20h (13/06/2024) after the successful completion of IDA2."
- Example: IDA2 clearing runs from ~22:00 to ~22:20 CET.

For the simulator: model the GCT−20min point as the start of the cross-border capacity freeze. Model the GCT-to-results window (20 minutes) as a full suspension of both matching and order submission for the affected delivery-day contracts.

**Orders survive the IDA clearing window and are batch-matched at reopen (v2.10, `OMIE-TAPE-CONTRACT-DATE-1` Q2; OQ-28).**
- **Rule.** An IDA suspends continuous submission and matching. It does not cancel orders resting in the continuous book for contracts that stay open, so those orders survive the gap. When continuous trading resumes, the book is re-matched by the batch-matching procedure (Continuous Trading Matching Algorithm §5.5 and §6.3). OMIE Instruction 1/2024 §8 says continuous allocation and order submission "will continue with Round 1st … at 22:20h" after IDA2.
- **Evidence (medium-high confidence).**
  - In `trades_20250626.1`, 14 trades carry a match time of exactly 25/06/2025 22:20:00.
  - `orders_20250626.1` has no order sent at 22:20:00. The first sends come in the following seconds (334 in 22:20:00–22:20:09, none of them at :00).
  - No order file on disk has a send in 22:00:00–22:19:59, and no trades file has a match in that window.
  - So the 22:20:00 trades pair orders that rested through the gap. Several of them have one side blank, meaning a non-Iberian counterparty. At least one Iberian side predates 22:00, for example GRTEN/GRTEC01 buy orders on contract 00:30 of 06-26 sent from 16:05 on 06-25.
- **Gaps inside one product's window.**
  - **IDA2 (22:00–22:20 on D−1):** in every product (D, k).
  - **IDA3 (10:00–10:20 on D):** also in products (D, k ≥ 48). In `orders_20250625.1` there are 0 sends in 10:00–10:19 for any contract, against 42 sends in 09:50–09:59 for contracts ≥ 12:00.
  - For k ≤ 47 there is also no submission in 10:00–10:20, because those contracts have already closed by 10:00 (GCT rule, §3.1).
- **Replay simplification (recorded modeling choice, not a rule).** The replay carries the book across the gap unchanged and applies the post-reopen orders one by one, in timestamp order. It does not run a separate reopen batch-matching step. Any crossed state left by the reopen is resolved by the first incoming order under regular matching (§6.2). Modeling the reopen batch match is out of scope for `OMIE-TAPE-CONTRACT-DATE-1`.

### 3.4 Halts and hibernation

- When OMIE performs local maintenance, continuous trading is halted. Orders in XBID are hibernated.
- Hibernated orders remain inactive after maintenance ends; the market participant must manually reactivate them.

---

## 4. Order Types

**Source:** XBID Information Package §7 (order type table); OMIE Details of Intraday Market §3.1; Continuous Trading Matching Algorithm §3.

### 4.1 Regular (limit) orders

- Buy or sell a quantity of a specific contract at a limit price.
- Default execution restriction: NON.
- Validity: GFS or GTD.
- Submitted per delivery area.
- Eligible for predefined products only (not user-defined blocks).

### 4.2 Iceberg orders

- A regular limit order with hidden total quantity and visible peak quantity.
- Only the peak size is shown in the order book.
- When a slice is exhausted, a new slice of peak size is activated and receives a new timestamp.
- Execution restriction: **ICE** in the OMIE/XBID data feed. Icebergs behave like NON orders (resting, partial fills allowed) but carry hidden total quantity. The restriction value `ICE` replaces `NON` for iceberg orders — icebergs cannot combine with IOC or FOK behavior.
- Validity: GFS or GTD.
- **ICEwPI (Iceberg with Price Increment):** Each new slice is priced at the previous slice price ± a fixed delta. Buy icebergs: negative delta (price decreases per slice). Sell icebergs: positive delta (price increases per slice).

> **Data feed encoding:** In the OMIE public data (`orders_YYYYMMDD.1`), iceberg orders are identified by the execution restriction field = `ICE` and a non-zero "Cantidad reducida" (peak quantity) value. The PPD field carries the per-slice price increment (0 for standard icebergs, non-zero for ICEwPI).

### 4.2.1 Iceberg interaction semantics (simulator design decisions)

The following three rules govern how icebergs interact with aggressive orders in the simulator. They **deviate from the real SIDC/XBID matching algorithm** and are documented as explicit simulator design decisions. The authoritative algorithm (Continuous Trading Matching Algorithm §3.3, §5.1.2) specifies that iceberg full quantity is "exposed to the market for matching" and that a single incoming order may match "more than one slice of an iceberg order already in the order book." The simulator simplifies this behavior as described below.

**Rule 1 — FOK visibility: liquidity check with ICEwPI slice-by-slice price awareness (M6.3, updated OQ-11).**
A Fill-or-Kill order's pre-match liquidity check considers the total iceberg quantity for opposite-side icebergs at crossing prices. For **standard icebergs (PPD=0)**, all remaining quantity is matchable if the base price crosses. For **ICEwPI icebergs (PPD != 0)**, the pre-check must compute how many slices would actually cross at their natural drifted prices and count only that volume as matchable.

**Standard iceberg example:** FOK BUY 3 MW is accepted against an iceberg with peak=2 and total=7 (PPD=0), because all 7 MW are at the same crossing price.

**ICEwPI example:** ICEwPI SELL base=50.00, ppd=+0.50, peak=2, total=10. FOK BUY limit=51.00 for 8 MW. Slices at natural prices: 2@50.00, 2@50.50, 2@51.00 = 6 MW crossing. Slice at 51.50 does NOT cross. Matchable = 6 MW < 8 MW requested. **FOK rejected.** If the FOK requested 6 MW: accepted, fills at natural iceberg prices (50.00, 50.50, 51.00). No price capping occurs.

When accepted, the FOK uses multi-slice matching to fill the full FOK quantity across iceberg slice regenerations within a single `submit_order` call. Each slice trades at the iceberg's (drifted) limit price per the price-determination rule (§5.1.2).

- *Real market rule:* FOK orders see the iceberg's full remaining quantity for matching purposes (Continuous Trading Matching Algorithm §3.3: "full quantity is exposed to the market for matching, and which may come with a range of limit prices"). For ICEwPI, §5.2.1 states slices "participate with each slice individually... until either all slices are fully executed or no further execution is possible anymore." The price-determination rule (§5.1.2) requires trade price to be set by the iceberg's limit price, not the incoming order's. **See OQ-11 for full analysis.**
- *Implementation:* `_matchable_volume()` must compute slice-by-slice projected prices for ICEwPI icebergs, summing only slices whose projected price crosses the FOK limit. Standard icebergs (PPD=0) sum total remaining quantity as before. `_fok_multi_slice_fill()` iterates: dispatch limit order, regenerate iceberg slices at natural drifted prices, re-dispatch remaining quantity, until the FOK is fully filled. No price capping is needed.
- *Classification:* **Consistent with market rules** (updated from simulator design decision in M6.3; ICEwPI slice-awareness added per OQ-11).

**Rule 2 — Multi-slice matching per aggressive order (NON, IOC, FOK; M6.4 update).**
A single aggressive order (NON, IOC, or FOK) can fill multiple iceberg slices within one match cycle. After the visible peak is consumed, the iceberg regenerates a new slice (at its natural drifted price for ICEwPI) and matching continues against the same aggressive order until either the aggressive order is fully filled, the iceberg is fully exhausted, or the regenerated slice's price no longer crosses the aggressive order's limit. All slice interactions occur within a single `submit_order` call.

- *Consequence:* To consume a 10 MW standard iceberg (PPD=0) with peak = 3 MW, a single NON, IOC, or FOK order for 10 MW fills all four slices (3 + 3 + 3 + 1 MW) within one match cycle. For ICEwPI icebergs, matching stops at the slice whose drifted price no longer crosses the aggressive order's limit.
- *Real market rule:* A single incoming order may match multiple slices of the same iceberg in one matching process (Continuous Trading Matching Algorithm §5.1.2: "a single order with a new timestamp is matched against more than one slice of an iceberg order already in the order book").
- *Classification:* **Consistent with market rules** (NON/IOC upgraded from single-slice in M6.4; FOK multi-slice introduced in M6.3).
- *Re-ranking of a regenerated slice (clarification, v2.14, 2026-10-07, `OMIE-SPEC-CERT-AMEND-1`):* "matching continues against the same aggressive order" above does not mean the aggressor trades the iceberg's next slice before anything else. Subject to §6.1, each regenerated slice is ranked as a new entry at its own (drifted) price. The aggressor always trades the best-ranked counter bid in the book next. That may be a different resting order when the drifted slice price is worse than that order's price, or when the slice price equals it and the other order has the earlier timestamp. The aggressor never trades at a worse price level while a better-priced counter bid can still trade, and that includes the next slice of a better-priced iceberg. Stopping conditions are as in Rule 2.
  - *Sources:* Continuous Trading Matching Algorithm §5.1 (`:462-464`: "a new slice of an iceberg order" is an order with a new time stamp) and §5.1.3 (`:485-490`: an incoming order "may sequentially generate multiple transactions at different prices against multiple different orders", moving to "the next best price level"). §4.1: the activation of a new iceberg slice is an order-book update. OMIE Rules Annex §4.4 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10262-10267`): each instance of the iceberg has "its own creation time". §5.2 (new timestamp on slice activation) and §6.1 (price-time priority, a hard rule).
- *Incoming iceberg (ICE aggressor) (clarification, v2.14, 2026-10-07, `OMIE-SPEC-CERT-AMEND-1`; user directive 2026-10-07; domain confirmation at `OMIE-ICE-INCOMING-HOP-1` Step 2):* Rule 2 above names NON, IOC and FOK aggressors. An incoming iceberg (ICE) with a competitive price is also an aggressor, and these rules apply to it: *(amended 2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1`, user ruling 2026-10-20: items (2), (4) and (6) and the classification state the instance chain; history in the v2.14 row, OQ-33 and OQ-34).* Below, $b$ is the incoming iceberg's price, $P$ its peak (reduced part), $d$ its price increment (PPD; $d = 0$ for a standard iceberg), $M$ the quantity it sweeps on entry and $R$ its remainder.
  - **(1) Total quantity.** It aggresses with its **total** quantity, not only its visible peak. It keeps matching while it remains executable and has quantity left.
  - **(2) One transaction per (counter bid, instance) pair.** The incoming iceberg is a chain of instances, each of size $P$; the last instance may be smaller. Each trade is between one resting counter-bid entry and one incoming instance. Each resting NON order is one counter bid, and each instance of a resting iceberg is its own counter bid, with its own creation time. A trade row ends where a resting entry ends **or** where one incoming instance ends and the next begins. Two resting instances are never merged into one row. (OQ-33, revised 2026-10-20.)
  - **(3) Price.** Every trade prices at the resting entry's limit (§6.2 item 3). For a resting ICEwPI that is the matched slice's drifted price $\mathit{base} + k\,\mathit{ppd}$ (§5.1.2; OQ-31).
  - **(4) Remainder.** If a remainder $R > 0$ is left once the order is no longer executable, it enters the book **as an iceberg** with total $R$. With $k_e = \lfloor M/P \rfloor$ and $u = M - k_e P$ (the part of instance $k_e$ already traded), the remainder keeps its place in the ladder: it continues as instance $k_e$, at limit $b + k_e\,d$. It shows $v_1 = \min(P - u, R)$, the unused part of instance $k_e$, and keeps $R - v_1$ hidden. This is "at most the reduced part … even if the matched quantity was not a multiple of the reduced part" (Annex §4.4). As for a NON residual, partial execution assigns no new timestamp (§5.2). The display after a part-used instance ($u > 0$) is derived from OMIE data, not stated by the sources (OQ-34 resolution).
  - **(5) Never IOC.** An iceberg is never an IOC order (§4.2). The remainder is not cancelled, and no output may label it IOC. Whatever internal device the simulator uses for the sweep is an implementation detail.
  - **(6) Instance limits.** Instance $k$ ($k = 0, 1, 2, \dots$) trades only at prices its own limit $b + k\,d$ crosses; for $d = 0$ every instance has limit $b$. The sweep stops at the first instance that cannot trade, even if $b$ would still cross. Each trade is still priced by the resting entry (item (3)). (Annex §4.5: each new instance has the price of the previous instance plus the increment; OQ-34.)
  - *Sources:* OMIE Rules Annex §4.4 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10256-10261`): "When entering an “Iceberg” bid with a competitive price, the quantity to be considered will be the total quantity of the “Iceberg” bid, recording a transaction for each counter bid that it matches with. Additionally, if the total quantity does not match, the viewable quantity that will appear in the Bid Book will be at most the reduced part specified when creating the bid, even if the matched quantity was not a multiple of the reduced part." The same section at `:10262-10267` says a resting iceberg trades "different transactions … for each instance". Continuous Trading Matching Algorithm §3.3 (`:257-259`): "their full quantity is exposed to the market for matching". §3.3 (`:265`): the "total quantity is reduced by the trade quantity". §5.1.3 (`:486-493`): one incoming order may produce "multiple transactions at different prices against multiple different orders", and the order enters the book "with its remaining quantity" only once it is no longer executable.
  - *Sources of the 2026-10-20 amendment (items (2), (4), (6)):* the user's ruling of 2026-10-20: "The real market treats it as a chain of instances, each the size of the peak. Instance may only trade at its own limit, where is the increment. The sweep stops at the first instance that cannot trade. The leftover keeps its place in the ladder, and trade rows split where one instance ends and the next begins." OMIE Rules Annex §4.5 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10276-10278`): "each new instance will have a new price, which will be calculated as the price of the previous instance plus the price increment". Annex §4.4 (`:10256-10261`, quoted above): "at most the reduced part". Continuous Trading Matching Algorithm §3.3 (`:257-259`): the full quantity is exposed for matching and "may come with a range of limit prices". §5.1.3 (`:486-493`): matching continues "as long as the incoming order remains executable", then the order enters the book "with its remaining quantity". OMIE data, 317 session days: OQ-33 and OQ-34 (blocks of 2026-10-20).
  - *Classification:* **Consistent with market rules** for (1)–(6) (user ruling 2026-10-20; OMIE data; Annex §4.4–§4.5). *Simulator limit:* on a trade ring without room for the split rows of item (2), the rows are left unsplit (volumes and prices unchanged; a certificate gap, not a market rule).

**Rule 3 — Post-match slice regeneration crossing.**
When a NON aggressive order partially fills an iceberg's visible peak and the aggressive order's unfilled remainder rests in the book, the regenerated iceberg slice (submitted during post-match regeneration within the same submit cycle) may cross that resting remainder. The resting remainder acts as passive liquidity for the new slice, producing an additional trade. This is correct simulator behavior consistent with how any new order (including a regenerated slice) triggers regular matching against the current book state.

- *Trade price of the hop (clarification, v2.13, OQ-31 RESOLVED 2026-10-02):* "passive liquidity" above describes **matchability only**. The resting remainder can be matched by the regenerated slice. It does **not** set the trade price. The hop trades at the **iceberg slice's own limit price**: the base price for PPD = 0, or the slice's drifted price $\mathit{base} + k\,\mathit{ppd}$ for ICEwPI. It never trades at the aggressor's limit $L$. For price determination the iceberg is the passive (resting) side of the hop, as in §5.2 and §6.2 item 7. If the regenerated slice's price does not cross $L$, there is no hop trade (§6.2 item 2; Rule 2).
  - *Source:* OMIE Rules, Iceberg bids §4.4 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10262-10267`): a counter bid "at a quantity higher than the viewable quantity of the “Iceberg” bid" produces "different transactions … for each instance of the “Iceberg” bid, each with its own creation time", and the iceberg is what remains in the Bid Book.
  - *Source:* Continuous Trading Matching Algorithm §5.1.2 (`:475-479`): the price is "always determined by the iceberg order already in the order book … even if the timestamp of the iceberg order is renewed during the matching process".
  - *Source:* Continuous Trading Matching Algorithm §5.1.3 (`:490-493`): the incoming order is entered into the book with its remaining quantity only after it no longer "remains executable".
  - In the real algorithm this hop is not a separate event. It is one more slice inside the aggressor's single matching process. The remainder-rests-then-slice-crosses sequence is the simulator's decomposition of that process.
- *Real market rule:* Consistent with the algorithm's treatment of new iceberg slices as new-timestamp entries that trigger regular matching (Continuous Trading Matching Algorithm §5.1, line: "a new slice of an iceberg order" triggers regular matching). The §5.1.1 "order already in the order book" price rule does not select the aggressor's remainder here: per §5.1.3 the remainder is not yet in the book while it can still execute against the iceberg. Pricing is governed by §5.1.2.
- *Classification:* **Consistent with market rules.** The regenerated slice is a new-timestamp entry and legitimately matches against any resting order. At this hop the trade price is the iceberg slice's limit (OQ-31).
- *ICE aggressor (clarification, v2.14, 2026-10-07, `OMIE-SPEC-CERT-AMEND-1`; OQ-31 addendum):* Rule 3 above is worded for a NON aggressor. The same pricing holds when the aggressor is an **incoming iceberg** (Rule 2, *Incoming iceberg*):
  - The incoming iceberg aggresses with its total quantity as a chain of instances: instance $k$ trades only where its own limit $b + k\,d$ crosses (Rule 2, *Incoming iceberg*, items (2) and (6)). Only its non-executable remainder rests, and only after the sweep ends, at the rest price $b + k_e\,d$ with $k_e = \lfloor M/P \rfloor$ (item (4)) (Annex §4.4, `:10256-10261`, and §4.5, `:10276-10278`; Matching Algorithm §5.1.3, `:490-493`; user ruling 2026-10-20).
  - On every trade of the sweep, the iceberg that was **already resting** is the price-setter. That includes slices regenerated during the sweep. The trade prints at that slice's own limit: the base price, or $\mathit{base} + k\,\mathit{ppd}$ for ICEwPI. It never prints at the incoming iceberg's limit.
  - The simulator may split this into steps: the incoming iceberg's remainder rests, then a regenerated slice of the resting iceberg crosses it. That hop is priced exactly as the NON hop above. The previously resting iceberg sets the price, and the incoming iceberg's rested remainder is the passive side for matchability only (Matching Algorithm §5.1.2, `:475-479`; OQ-31, Reading B).
  - *Classification:* **Consistent with market rules** (OQ-31 addendum, 2026-10-07).

**Rule 4 — Regen-immediate-rematch cascade resolves within the originating matching process (M; P2-PERF-8 ruling, 2026-05-26).**
When a regenerated iceberg slice (Rule 2/Rule 3) is **itself** crossed by still-resting opposite-side liquidity present in the book — and when the resulting trade exhausts that new peak and triggers a *further* regeneration that again crosses, and so on — **the entire cascade MUST resolve to a fixed point within the originating matching process** (all resultant trades carry that matching process's timestamp; the order book is never left in a crossed state between aggressor arrivals). It is **spec-violating** to leave the regenerated slice resting visibly in a crossed book until the next incoming aggressive order triggers matching ("draining the cascade across submit cycles").

- *Operative regulatory text (Continuous Trading Matching Algorithm):*
  - §4.1, "Order book": the order book "is updated whenever orders are added to it, deleted from it, modified (volume, price, execution restriction, **activation of a new iceberg slice**)... **After each order book update... the algorithm validates if new matching opportunities have arisen and, if so, the resulting trades are concluded.**" — Slice activation is itself a book-update event, and any crossing it creates is concluded at once, not deferred.
  - §5.1: "Regular matching is triggered by the entry of an order with a new time stamp. An order with a new time stamp may be a newly entered order, a modified order, a (re)activated order that was inactive before, **or a new slice of an iceberg order**." — Each regenerated slice is an independent regular-matching trigger.
  - §5.1.3: an order with a new timestamp "continues to be matched against orders entered at this price level. This process continues **as long as the incoming order remains executable and has a positive order quantity.**" — Matching runs to a fixed point within the process; a partially-crossing remainder does not pause.
  - §5.1.2: the iceberg's price governs the trade "**even if the timestamp of the iceberg order is renewed during the matching process**" — explicitly contemplates renewed slice timestamps *within a single matching process*, confirming intra-process resolution.
- *Why across-cycle draining is observably distinguishable (i.e. why this is M, not merely E):* In **regular** (continuous) trading a crossed order book cannot persist. Only **batch matching** (§5.2), triggered solely by a cross-border capacity update, allows a previously-crossed book to wait for a separate matching process; there is no analogous deferral in regular matching. Leaving a regenerated slice resting in a crossed book until the next aggressor would (a) expose a crossed LOB snapshot to any observer reading the book between aggressor arrivals, (b) emit the cascade trades under a later (wrong) triggering timestamp, and (c) reorder the trade sequence relative to any orders that arrive in between. All three are observable.
- *Scope note vs. Rule 3:* Rule 3 covers the single-hop case where the new slice crosses the *resting remainder of the same aggressor*. Rule 4 generalizes to the multi-hop cascade where the new slice crosses *any* resting opposite-side liquidity and may itself trigger further regenerations — the fixed-point requirement applies to the whole chain.
- *Scope clarification — what Rule 4 governs vs. what it does NOT (P2-PERF-12 ruling, 2026-05-28).* Rule 4 is a **within-call (intra-process) fixed-point requirement** on the resting **order book** (`OrderBook` / `lob`). It is **not** a post-call cleanliness invariant on simulator-internal side-channel state (e.g. `iceberg_meta`). The post-condition the spec requires after a `submit_order` (or any other matching-triggering event) returns is exactly this: the resting order book must not be crossed, and no regenerated visible peak that crossed during the call may have been left visibly resting in a crossed book ("draining across submit cycles" — explicitly forbidden, see above). The spec does NOT require that every slot in a simulator-side bookkeeping array describing icebergs hold an `order_id` currently present in the resting book; an iceberg whose visible peak was fully consumed by a prior aggressor's matching process and whose next slice has been registered for later regeneration is a permitted between-aggressor state under the matching algorithm (a new slice is activated when the previous one's quantity reaches zero or when subsequent matching draws against the remaining hidden quantity — see Continuous Trading Matching Algorithm §3.3, lines 265–275). The "no `peak_consumed=True` resting slot" assertion is a **simulator implementation invariant** of the current `iceberg_manager` design, **not** an OMIE market rule. The spec requirement is: every regenerated slice that crossed during the matching process must have been fully consumed (zero remaining visible) or fully rested (positive remaining visible at a non-crossing price) by the time the matching process returns. A side-channel slot whose `order_id` is absent from the LOB is consistent with the spec when the slot represents an iceberg whose previous peak was fully consumed and whose next peak has not yet been activated.
- *Classification:* **Consistent with market rules — within-process (within-call) resolution is mandated on the resting order book.** The scope clarification above is reaffirmed (not amended) market semantics: post-call invariants on simulator-internal bookkeeping are an engineering concern, governed by Rule 4 only insofar as they cause an observable difference in the resting order book or the trade sequence. See `docs/omie/sprint-status.md` (P2-PERF-8 domain ruling 2026-05-26 and P2-PERF-12 scope ruling 2026-05-28) for the implementation history; see also OQ-20 for the engineering-invariant tracking entry. **OQ-23 (2026-07-09)** dispositions a second, distinct route to the same OQ-20-class state — a `dirty`-invisible cross-slot peak consumption inside `regenerate_slices` that leaves a lower-index iceberg slot with a consumed visible peak at pass end; it is **CONSTRUCTIBLE on valid inputs but is NOT a Rule 4 violation** (the resting book remains uncrossed and all trades are emitted intra-call — the surviving state is a permitted between-aggressor stale side-channel slot per this Scope clarification), and it warrants a scoped **engineering** follow-up (not a market-spec change) because the P2-PERF-8 `no_fill_path` regen-skip optimization depends on the broken post-condition. See OQ-23 for the full ruling and a constructibility probe.
- *Trade timestamps (clarification, v2.14, 2026-10-07, `OMIE-SPEC-CERT-AMEND-1`):* the parenthetical "all resultant trades carry that matching process's timestamp" above overstates the source. Read it as follows:
  - Every trade of the cascade is **concluded within, and attributed to, the originating matching process**. No cascade trade may be stamped with the arrival of a later aggressor; that is the across-cycle defect forbidden above.
  - The cascade's trades need **not** share one identical timestamp value. Per Continuous Trading Matching Algorithm §5.3 (`:589-590`), "The timestamp of a trade reflects the moment that one of the orders that will enter into the trade is found to be executable (against the other order) by the algorithm". The trade's attributes are "derived from the characteristics of the order matching event" (`:578-579`).
  - Within one process each regenerated slice is a new-timestamp entry (§5.1, `:462-464`; §5.2 above). So per-slice trade timestamps that increase within the process satisfy this rule.
  - What Rule 4 requires is unchanged: resolution inside the process and an uncrossed resting book between aggressors.
- *Slice activation timing (correction note, v2.14, 2026-10-07; OQ-20 correction note):* the Scope clarification above says a new slice is activated "when the previous one's quantity reaches zero or when subsequent matching draws against the remaining hidden quantity". Only the first alternative has source support. Activation is **eager**.
  - Continuous Trading Matching Algorithm §3.3 (`:265-274`): when a trade reduces the shown quantity to zero, "a new slice of order quantity is made available in the market sized at the peak size of the iceberg order or at the remaining quantity, if this is smaller than the peak size".
  - OMIE Rules Annex §4.4 (`:10253-10255`): "When the viewable reduced part of the bid is fully matched, a new bid will be automatically created in the Bid Book".
  - So in the market, between matching processes, an iceberg with remaining quantity always shows a slice.
  - The engineering conclusions of the Scope clarification and of OQ-20/OQ-23 are not reopened: a stale side-channel slot is not a Rule 4 violation. Whether a lazily activated slice makes any observable difference is an engineering question for the architect. Examples: a resting book snapshot that lacks the slice, or a later slice timestamp. See the OQ-20 correction note.

### 4.3 User-defined block orders

- A single order spanning multiple consecutive delivery periods of the same product.
- **Minimum block length: 2 MTUs** (i.e. ≥ 30 minutes for 15-min products). Source: Continuous Trading Matching Algorithm §3.4: "a block order is an order with a total delivery period greater than the smallest delivery period of the product for which the order is being entered."
- **Maximum block length:** up to the remaining open contracts in the delivery window (up to 96 for 15-min products). No explicit regulatory upper bound is stated in the OMIE Rules corpus; the specific parameter constraints are delegated to "the provisions of the European Continuous Trading Platform" (OMIE Rules §45.2.1).
- Always AON: must be executed entirely or entered into the book.
- Cannot be an iceberg.
- Validity: GFS or GTD.
- Eligible for user-defined blocks only.
- **For routing/capacity:** the most restrictive ATC across any sub-period is used.
- **v1 scope:** Defer block order implementation to a later version.

### 4.4 Basket orders

| Basket type | Behaviour | Execution restriction |
|-------------|-----------|----------------------|
| Independent | Each order processed independently; may enter book separately. Sub-orders are "processed as if they would have been submitted separately" (XBID Information Package §4 footnote 1). | Per-order |
| All-or-nothing | All orders accepted or all rejected | All must match |
| Linked | All orders FOK; all must execute immediately and fully or all deleted. Source: Continuous Trading Matching Algorithm §3.6: "All orders in a linked basket must be executed immediately and with their full quantity (i.e., they always have a FOK execution restriction). If this is not possible, the linked orders will all be deleted." | FOK (all) |

- Basket orders may span different contracts, products, delivery areas, and market areas.
- The basket is rejected entirely if any included contract is not in TRADE status (OMIE Rules §45.2.10).
- **v1 scope:** Implement independent basket only. Linked (FOK-all) and AON baskets deferred. Cross-zone basket behavior deferred entirely.

### 4.5 Execution restrictions summary

| Restriction | Partial fill | Book entry if unmatched | Applicable to |
|-------------|-------------|-------------------------|---------------|
| NON | Yes | Yes | Regular, Block |
| IOC | Yes | No (cancelled) | Regular |
| FOK | No (full or cancel) | No (cancelled) | Regular, Linked basket |
| AON | No (full or enter book) | Yes | Block orders only |
| ICE | Yes (peak slices) | Yes (hidden qty rests) | Iceberg only |

> **Note on ICE encoding:** In the OMIE/XBID data feed, iceberg orders use `ICE` as their execution restriction value rather than `NON`. ICE orders exhibit NON-like resting behavior (partial fills allowed, residual stays in book) but additionally track hidden total quantity and regenerate visible peak slices. The statement "NON restriction only" in §4.2 means icebergs cannot use IOC or FOK behavior — the `ICE` encoding replaces `NON` for icebergs. See §13 for the full data format reference.

### 4.6 Validity conditions

| Condition | Meaning | Default |
|-----------|---------|---------|
| GFS | Valid until contract's GCT | Yes (default) |
| GTD | Valid until agent-specified time ≤ GCT | No |

---

## 5. Order Lifecycle Transitions

**Source:** Continuous Trading Matching Algorithm §3, §4, §5; OMIE Details of Intraday Market §3.1.

### 5.1 State model

```
SUBMITTED → [validation]
    ├─ REJECTED (fails OMIE or XBID validation)
    └─ ACCEPTED
         ├─ EXECUTED (fully matched immediately)
         ├─ PARTIALLY_EXECUTED + resting in BOOK
         ├─ CANCELLED (IOC/FOK unmatched; GTD expired; GCT reached; halt/hibernation)
         └─ IN_BOOK
              ├─ MODIFIED (price, quantity, or execution restriction change → new timestamp)
              ├─ PARTIALLY_EXECUTED (residual remains in book or cancelled per IOC)
              ├─ FULLY_EXECUTED
              ├─ CANCELLED (by agent; GTD expiry; GCT; market halt)
              └─ HIBERNATED (market halt; requires manual reactivation)
```

### 5.2 Timestamp rules

- Each order receives a timestamp when entering the SOB.
- A **new timestamp** is assigned on: modification (price, quantity, execution restriction), reactivation from inactive, or activation of a new iceberg slice.
- **Partial execution is NOT a modification and does NOT assign a new timestamp.** The unexecuted residual of a NON incoming order enters the book with its **original entry timestamp** (same order, not modified). Likewise a resting order that is partially executed keeps its timestamp. Source: Continuous Trading Matching Algorithm §4.3 (the new-timestamp triggers are volume/price/execution-restriction *modification*, inactive→active, new iceberg slice) and §5.1.3 ("the order is entered into the order book with its remaining quantity" — the same order, only its quantity reduced by the trade per §5).
- Iceberg slices receive a new timestamp per slice, **except**: when a single incoming order is matched against multiple iceberg slices, the iceberg is treated as the passive (resting) side throughout and its timestamp does not supersede the incoming order for price determination purposes.
- All timestamps across orders and explicit capacity requests are assigned from a single global queue (unique, sequential). Source: Continuous Trading Matching Algorithm §4.3.

### 5.3 Modification rules

- Modifying price, quantity, or execution restriction of an order resets its timestamp (loses time priority).
- This is a hard rule from the XBID system.

### 5.4 Order rejection conditions

- Price outside product min/max (below −9,999 or above +9,999 EUR/MWh).
- Quantity outside product min/max (below 0.1 or above 999 MW).
- Agent's operating limit (guarantee) would be exceeded.
- Order submitted after GCT for the target contract.
- Order fails TSO unavailability/limitation check (sum of offered energy + prior program + existing book + new order must stay within declared capacity bounds).
- *(Added v2.14, 2026-10-07, `OMIE-SPEC-CERT-AMEND-1`.)* **ICEwPI price-increment sign.** An iceberg with a non-zero price increment (PPD) of the wrong sign is rejected: a BUY with PPD > 0, or a SELL with PPD < 0. PPD = 0 is a standard iceberg and is always valid on this check. The magnitude of PPD has no rule-level bound (§13.6 item 2; OQ-13).
  - *Sources:* OMIE Rules Annex §4.5 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10280-10281`): "“Iceberg” purchase orders can only be entered with negative price increments, while sell orders can only be entered with positive price increments". OMIE Details of the Intraday Market's Operation (`OMIE_Details_of_the_Intraday_Market_s_Operation.md:322-323`). §13.6 item 1 (hard rule).
- *(Added v2.14, 2026-10-07, `OMIE-SPEC-CERT-AMEND-1`.)* **Iceberg peak larger than total.** An iceberg whose visible peak ("Cantidad reducida") is greater than its total quantity is rejected. A peak equal to the total is valid.
  - *Source:* OMIE Rules Annex §4.4 (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:10268-10270`): "In addition to the validations included in the “Bid Validation” Rule for the continuous intraday market, it will be verified that the reduced part of the bid to be shown is less or equal to the total quantity of the product bided."

---

## 6. Matching Rules and Priority

**Source:** Continuous Trading Matching Algorithm §4.4, §5; ACER Decision 11/2024 Art. 5; CACM Regulation Art. 51; CID Summary.

### 6.1 Price-Time Priority

- **Buy side:** ranked highest price first; ties broken by earliest timestamp.
- **Sell side:** ranked lowest price first; ties broken by earliest timestamp.
- A match occurs when best bid price ≥ best ask price.
- This is a hard rule mandated by ACER Decision 11/2024 Art. 5 and CACM Regulation Art. 51.

### 6.2 Regular matching (triggered by new-timestamp order)

1. Incoming order attempts to match against best counterpart in the consolidated order book.
2. Match is possible if: orders are opposite sides, same contract, and `sell_price ≤ buy_price`.
3. Trade executes at the **resting order's limit price** (maker pricing). Source: Continuous Trading Matching Algorithm §5.1.1.
4. If incoming order is partially filled and has NON restriction: residual enters the book with its original entry timestamp (see §5.2).
5. If incoming order is partially filled and has IOC restriction: residual is cancelled.
6. If matching exhausts one price level, continues to next best price level.
7. Iceberg: when a single incoming order matches multiple slices of the same iceberg, the iceberg is always the price-setter. Source: Continuous Trading Matching Algorithm §5.1.2. This includes the §4.2.1 Rule-3 hop, where a regenerated slice crosses the same aggressor's resting remainder (OQ-31, resolved 2026-10-02).

### 6.3 Batch matching (triggered by cross-border capacity increase)

- Only relevant in multi-zone simulation. Not relevant for single-zone (e.g. Spain-only) simulation.
- Process: collect all crossed pairs → remove non-matchable pairs → retain pairs with largest price difference → select pair with oldest order → match and reserve capacity → repeat.
- All pairs matched in the same batch round receive the same price: arithmetic mean of the last matched pair's limit prices. Source: Continuous Trading Matching Algorithm §5.2.3.
- Contract (order book) sequence in batch matching: longest delivery period first; ties by earliest start time; ties by largest minimum tradable unit; ties by longest delivery unit duration. Source: Continuous Trading Matching Algorithm §5.2.2.
- Batch matching suspends regular matching and explicit allocation for its duration. Source: Continuous Trading Matching Algorithm §5.5.

### 6.4 No cross-contract matching

- Orders on different contracts are never matched against each other. Source: Continuous Trading Matching Algorithm §5.

### 6.5 Cross-border visibility

- In a multi-zone simulator: orders from a remote zone are shown in the local order book only if ATC to that zone is positive.
- AON orders are shown either with full quantity or not at all (never partial display).
- Non-AON orders may be shown with a fraction of their quantity equal to available ATC.

### 6.6 Self-trade prevention (STP): none at XBID level

The XBID/M7 matching engine does **not** implement self-trade prevention. Matching is pure price-time priority; an agent's buy order can match its own sell order, and orders from two different units of the same agent can match against each other. No ownership-based exclusion or STP mechanism is documented in the matching algorithm specification, the CACM regulation, or the OMIE Rulebook bid validation rules (§45.2.2).

Where STP exists in European energy NEMOs, it is a local feature that explicitly does not extend to XBID-matched orders. Nord Pool's self-trading prevention documentation confirms: "Self Trade Prevention functionality does not apply for Xbid products." REMIT prohibits wash trades (fictitious trades with no change in beneficial interest) as market manipulation, but this is a post-trade surveillance and compliance obligation on participants, not a matching-engine constraint.

**Simulator implication:** The matching engine must not reject or skip matches based on order ownership. Diagnostic metrics (`self_match_agent_count`, `self_match_unit_count`) should be computed for observability but must not affect matching decisions.

**Source:** Continuous Trading Matching Algorithm §5.1, §5.2.3 (exhaustive exclusion criteria, none ownership-based); OMIE Rulebook §45.2.2 (bid validation checks, no STP); Nord Pool Self-trading Prevention support article; OQ-12 resolution in `docs/omie/open-questions.md`.

---

## 7. Anonymity and Information Visibility

**Source:** OMIE Market Rules §45.2.12, §46.1; XBID Information Package §2.

### 7.1 Anonymous order book

OMIE Rules §45.2.12 explicitly states: "information available on the Market Operator's Market Platform for market agents regarding products, contracts and data on the orders contained on the list of local orders is to be displayed on an **anonymous basis**, identifying solely those operations performed by the agent itself." Orders are anonymized when transmitted from LTS to SOB (XBID Information Package §2).

### 7.2 Trade notification

Each agent sees only their own matched trades. OMIE Rules §46.1: "the result is to be notified to the agent in question." Counterparty identity is not disclosed.

### 7.3 Simulator modeling

- Maintain agent identity internally (for PnL accounting and observation).
- Expose the LOB to agents as anonymous (prices and quantities only, no counterparty identity).
- Each agent's observation includes only their own pending orders and matched trades.
- This matches the real market rule exactly.

---

## 8. Settlement and PnL

**Source:** Continuous Trading Matching Algorithm §5.3; OMIE Details of Intraday Market §3.2.

### 8.1 Trade settlement formula

```
V (EUR) = Q (MW) × p (EUR/MWh) × d (h)
```

Where:
- `V`: financial value in EUR
- `Q`: traded quantity in MW
- `p`: trade price in EUR/MWh (resting order's limit price in regular matching; arithmetic mean of last pair in batch matching)
- `d`: delivery duration of the contract in hours — **for 15-min contracts: d = 0.25**

### 8.2 Trade firmness

- All trades in XBID/SIDC are firm delivery obligations. A buy trade implies an obligation to purchase; a sell trade implies an obligation to deliver. Source: OMIE Rules §46.2; OMIE Details of Intraday Market §3.2 (citing CACM Regulation).

### 8.3 Settlement granularity: per-unit (hard rule)

**Source:** OMIE Rulebook §50.1 (OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md, pp. 118-119).

The primary settlement granularity is the **bidding unit (Unidad)**, not the agent. The OMIE settlement formula DCMIC(d,c,t,u,r) is parameterized by `u` (bidding unit or portfolio). Settlement entries are recorded "in the register kept for such purposes by each bidding unit or portfolio" (§50.1). This is a hard rule, not a modeling choice.

Agent-level PnL is derived by summation over units:

```
agent_pnl(a) = sum(unit_pnl(u) for each unit u owned by agent a)
```

This sum-over-units invariant must hold exactly. See also §13.5 (DA-UNIDAD finding) for the full regulatory basis.

### 8.4 PnL for simulator agents

- **Buy agent PnL per trade:** `−V` (cash outflow) + market value of acquired energy (exogenous to the simulator; relevant for imbalance resolution).
- **Sell agent PnL per trade:** `+V` (cash inflow) − cost of delivered energy (exogenous).
- **Net realized PnL:** `sum(sell_trades: V) − sum(buy_trades: V)`, plus mark-to-market of open positions if applicable.

### 8.5 Closing-out open positions

- The continuous market has no automatic position close-out at GCT. Open orders are cancelled at GCT but already-matched trades are firm.
- Imbalance resolution (positions not physically delivered) is handled by TSOs in balancing markets — outside the scope of the intraday simulator.

---

## 9. Participant Actions and Market-State Variables

### 9.1 Participant actions

| Action | Parameters | Notes |
|--------|------------|-------|
| Submit order | contract_id, side (BUY/SELL), quantity (MW), price (EUR/MWh), execution_restriction (NON/IOC/FOK), validity (GFS/GTD + expiry_time), order_type (regular/iceberg/block) | For iceberg: also peak_size [, price_delta] |
| Modify order | order_id, new_price and/or new_quantity and/or new_execution_restriction | Resets timestamp |
| Cancel order | order_id | Only possible if order is active in book |
| Submit basket | list of orders + basket_instruction (independent/AON/linked) | v1: independent only |
| Reactivate hibernated order | order_id | Only after maintenance halt |

### 9.2 Observable market-state variables

| Variable | Description |
|----------|-------------|
| LOB per contract | Bid/ask price levels with aggregated visible quantities and timestamps. **Anonymous** — counterparty identity not shown (OMIE Rules §45.2.12). |
| Best bid / best ask | Top-of-book per contract |
| Own orders | Agent's own active orders (showing full iceberg quantity) |
| Trade book | Executed trades: price, quantity, timestamp, contract. Own trades only (OMIE Rules §46.1). |
| Current program | Agent's net scheduled position per delivery period (PHFC) |
| ATC per border/period | Cross-border available capacity (relevant only for multi-zone) |
| Contract status | OPEN / CLOSED (past GCT) |
| Session timing | Current time, next IDA GOT/GCT, next delivery period GCT |
| IDA suspension phase | NORMAL / CAPACITY_HALT (pre-IDA) / FULL_SUSPENSION (IDA clearing) |

*Trade book: side (addendum, v2.14, 2026-10-07, `OMIE-SPEC-CERT-AMEND-1`; spec only, no engine change filed, by user decision).* Besides price, quantity, timestamp and contract, each own trade in the trade book also carries its **side**: BUY or SELL, from the agent's point of view, which is the side of the agent's own order in that trade.
- *Sources:* Continuous Trading Matching Algorithm §5.3 (`:572-577`): a trade is an obligation "to transfer a certain amount of energy from seller to buyer" and "is always between exactly two orders". The OMIE public trades file keeps buyer and seller in separate columns (§13.2, columns 3–8). Settlement needs the side for its sign: the buy leg books −V and the sell leg +V (§8.4).

---

## 10. Differences from HFT/LOBSTER Assumptions

This section documents which assumptions embedded in the existing JaxMARL-HFT simulator (designed for LOBSTER equity LOB data) are **invalid** in the OMIE CID context.

| HFT/LOBSTER Assumption | OMIE CID Reality | Impact |
|------------------------|-----------------|--------|
| Single tradable instrument (one stock) | **96 quarter-hourly contracts per delivery day** (15-min MTU active since 2025-03-18), each with its own LOB | Must represent 96 simultaneous LOBs |
| Uniform time resolution (microseconds/milliseconds) | Energy delivery in 15-min intervals; trading timestamps are sequential integers, not wall-clock microseconds | Time granularity and timestamp model must change |
| No expiry — orders live until cancelled or matched | Orders expire at GCT (1h before delivery); contracts have a finite trading window | Must implement per-contract expiry |
| Financial settlement only (no physical delivery) | Trades are firm delivery obligations; imbalance resolved externally by TSOs | Simulator must represent trade firmness; imbalance resolution is outside OMIE scope |
| No capacity constraints | Cross-border ATC constrains visibility and matchability of remote orders | Needed only for multi-zone; can simplify to single-zone for v1 |
| Trade value = Q × p | Trade value = Q × p × d (delivery duration factor; d = 0.25 h for 15-min) | Financial formula differs |
| All orders same validity type | GFS and GTD validity conditions; iceberg, block, basket orders | More order types required |
| Price-time priority (same) | Price-time priority (same) | No change needed |
| No session structure | Rounds concept; two-phase IDA suspension; maintenance hibernation | Session/round state machine required |
| No participation constraints | TSO-declared unit availability limits; guarantee (operating limit) per agent | Order validation more complex |
| LOB data is historical replay | OMIE CID is forward-looking; contracts are for future delivery hours | Data model is different: synthetic or real historical MIC order flow needed |
| Single market area | Multi-zone with implicit cross-border allocation | For single-zone v1: ATC and CMM can be omitted |
| Maker price = resting order price (same) | Maker price = resting order price (same) | No change |
| No auction sessions | IDA sessions cause two-phase suspension: capacity halt then full matching freeze | Need to model suspension phases |

---

## 11. Hard Rules vs. Modeling Choices

### 11.1 Hard rules (from authoritative documents)

| Rule | Source |
|------|--------|
| Price-time priority for matching | ACER Decision 11/2024 Art. 5; CACM Regulation Art. 51 |
| Execution price = resting order's limit price in regular matching | Continuous Trading Matching Algorithm §5.1.1 |
| GCT = 1 hour before delivery start for all continuous contracts. **Iberian precision (v2.10):** GCT is hour-aligned. Each hourly round closes its four quarter-hour contracts together at (delivery hour − 1 h), so GCT(k) = 60·⌊k/4⌋ − 60 min from D 00:00 (§3.1). | OMIE Market Rules; OMIE Details of Intraday Market §3; OMIE Rules Annex 2 §2 + Figure 4 |
| Delivery hours are 24/7 with no market closure | Continuous Trading Matching Algorithm §1.1 |
| Block orders are always AON; minimum block length = 2 MTUs | Continuous Trading Matching Algorithm §3.4 |
| Iceberg orders use ICE restriction (NON-like behavior; cannot combine with IOC/FOK) | XBID Information Package §7; OMIE public data feed |
| Order modification resets timestamp | Continuous Trading Matching Algorithm §4.3 |
| Trade value = Q × p × d; d = 0.25 h for 15-min contracts | Continuous Trading Matching Algorithm §5.3 |
| All SIDC trades are firm delivery obligations | OMIE Rules §46.2 |
| Price bounds: −9,999 to +9,999 EUR/MWh; tick = 0.01 EUR/MWh | OMIE Rules §38.2, Annex 3 |
| Quantity bounds: 0.1 to 999 MW; tick = 0.1 MW | OMIE Rules §38.2, §45.2.1 |
| Active CID product for Spain: 15-min MTU (96 contracts/day) | OMIE Rules Annex 2 §9841–9857 (active since 2025-03-18) |
| All 96 contracts open simultaneously at 15:20 CET (D−1) after IDA1; close rolling ~1 h before delivery | OMIE Instruction 1/2024 §7; OMIE Rules Annex 2 schedule table |
| LOB is anonymous; agents see only their own matched trades | OMIE Rules §45.2.12, §46.1 |
| Exactly 3 IDA sessions (IDA1, IDA2, IDA3); no IDA4+ | OMIE Instruction 1/2024 §6, §8, §9; OMIE Rules Annex 2 |
| Pre-IDA capacity halt (GCT−20 min): cross-border capacity only; local matching continues | OMIE Instruction 1/2024 §8 |
| IDA clearing window (GCT to results): full matching and submission suspension | Continuous Trading Matching Algorithm §5.5; OMIE Instruction 1/2024 §8 |
| Notification thresholds (~−150 / +1,500 EUR/MWh for MIBEL): warnings only, not rejection limits | OMIE Rules Annex 3 |

### 11.2 Modeling choices (not mandated by regulation)

- Whether to model a single zone (Spain only) or multi-zone (Iberian + European).
- Whether to model only 15-min (required for Spain CID) or also block/basket orders in v1.
- Whether to model IDA sessions as distinct events or abstract them away.
- How to handle imbalance costs (outside OMIE market rules scope).
- Whether to model agent guarantee/operating-limit validation.
- Whether to replay historical MIC order flow data or use synthetic generation.
- Whether to model TSO unit availability constraints.

---

## 12. Sources Index

All local paths are relative to `docs/omiecid/corpus/`.

| Document | Local Path | Key sections for this spec |
|----------|-----------|---------------------------|
| OMIE Day-Ahead and Intraday Electricity Market Rules (English) | `Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md` | §38.2 (price/qty format), §45.2.1 (CID order params), §45.2.10 (baskets), §45.2.12 (anonymity), §46.1–46.2 (trade firmness), Annex 2 (schedule table, §9841–9857 MTU15), Annex 3 (price limits) |
| OMIE Details of Intraday Market's Operation | `Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/OMIE_Details_of_the_Intraday_Market_s_Operation.md` | §3.1 order types, §3.2 matching, §3.3 rounds |
| Continuous Trading Matching Algorithm Public Description | `Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/Continuous_Trading_Matching_Algorithm_Public_Description.md` | §3 order types, §3.4 block orders, §3.6 linked baskets, §4.3 timestamps, §4.4 price-time priority, §5 matching, §5.3 trade value, §5.5 priority/suspension |
| XBID/SIDC Information Package | `Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/Single_Intraday_Coupling_XBID_Information_Package.md` | §2 anonymization, §4 products, §7 order type table, §8 gate times |
| ACER Decision 11/2024 — Algorithm Methodology | `Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/ACER_Decision_11_2024_Annex_I_Algorithm_Methodology.md` | Art. 5 (continuous trading matching algorithm requirements) |
| ACER Decision 03/2025 — SIDC Products | `Authoritative_Document_Map_Spanish_Electricity_Market_Interrelationships_DA_ID_SIDC_REE_Balancing/ACER_03_2025_SIDC_PRODUCTS.md` | 15-min MTU products, IDA mandatory/optional products |
| OMIE Instruction 1/2024 — IDA Go-Live Operations | `Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/OMIE_Instruction_1_2024_IDA_Go-Live_Operations.md` | §6–§9: IDA session timing, rounds, capacity halts, maintenance |
| CNMC Resolution DCOOR/DE/009/24 | `Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/CNMC_Resolution_DCOOR_DE_009_24.md` | Approval of OMIE Market Rules (effective 13 June 2024) |
| CACM Regulation (EU) 2015/1222 | `Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE/Commission_Regulation_EU_2015_1222_CACM.md` | Art. 51 (matching algorithm requirements) |
| CID Summary (educational) | `education-materials/CID_summary.md` | Matching logic overview |
| OMIE — Modelo de Ficheros para la distribución pública de Información del mercado de electricidad, v1.38 (22/09/2026) | External (not in corpus): `https://www.omie.es/sites/default/files/2026-09/formato_ficheros_inf_pub_138_1.pdf` (linked from `https://www.omie.es/es/publicaciones/64`) | §3.2 (file structure, `.` thousands / `,` decimal), §5 (times in CET), §5.1.1.1 (`marginalpdbc`, dot decimal), §5.3.2.7 (trades fields incl. match time), §5.3.3.1 (orders fields) — cited in §13 (v2.12) |

---

## 13. Real Data Format Reference

**Source:** OMIE public data downloads at `https://www.omie.es/es/file-download`.

OMIE publishes daily order and trade files for the continuous intraday market. These are the authoritative data source for the M7 data loading milestone and for validating simulator behavior against real market outcomes.

**Data availability:** Monthly zip archives, one file per **session day** (see the file-date correction below).
- Orders: `https://www.omie.es/es/file-download?parents=orders&filename=orders_YYYYMM.zip`
- Trades: `https://www.omie.es/es/file-download?parents=trades&filename=trades_YYYYMM.zip`

**File naming convention:** `orders_YYYYMMDD.1` (orders), `trades_YYYYMMDD.1` (trades), where the date is the **SESSION day**.

> **CORRECTION (v2.10, `OMIE-TAPE-CONTRACT-DATE-1`).** Up to v2.9 this line said "the date is the delivery day". That was wrong.
>
> - File `orders_D.1` contains every order **sent** in the local session window **[22:20 D−1, 22:00 D)**. File `trades_D.1` contains every trade **matched** in the same window. The window runs from the IDA2 reopen on D−1 to the IDA2 gate closure on D.
> - Each file therefore holds rows of **two delivery days**:
>   - product D, for the part of its window after 22:20 on D−1;
>   - product D+1, for the part of its window from its GOT at 15:20 on D until 22:00 on D.
> - **Data evidence.**
>   - `orders_20250626.1` has 0 sends before 22:20 on 06-25, 0 sends in 22:00–23:59 on 06-26, 20,549 sends at 22:xx on 06-25 (all at or after 22:20), and 45,446 rows of product 06-27.
>   - `orders_20250625.1` has 0 sends at 22:xx on 06-25 and 22,514 rows sent on 06-24.
>   - No order or trade file on disk has a row stamped 22:00:00–22:19:59.
>   - The file header line names the session date: `orders_20250626.1` row 1 reads `…;26/06/2025;Órdenes enviadas a XBID;…`.
>
> Source: OMIE Instruction 1/2024 §8 (IDA2 22:00–22:20, continuous restarts at 22:20); OMIE Rules Annex 2 Figure 4 (round 1 starts at 22:00 with only D+1 contracts open). See §13.4 for how to build a delivery product from these files.

**General format properties:**
- Delimiter: semicolon (`;`)
- Number format: see *Number format* below. Orders and trades use Spanish numbers (`.` thousands, `,` decimal). `marginalpdbc` uses a dot decimal.
- Encoding: Latin-1 (ISO-8859-1). Header names carry accented letters (`Momento casación`, `Momento de envío`, `Condiciones a la ejecución`), and so does the row-1 metadata line (`Fecha Emisión`, `Órdenes enviadas a XBID`). None of the 635 real orders and trades files decodes as UTF-8 (architect whole-disk check, agile-plan v5.169 H-1).
- Row 1: report header line (OMIE boilerplate: issue timestamp, session date, report title)
- Row 2: blank (corrected v2.12; see below)
- Row 3: column headers
- Rows 4+: data rows
- Last row: a row of `;` characters only (end of report)
- Trailing semicolon on each row

> **CORRECTION (v2.12, `OMIE-TRADES-MATCH-TS-HEADER-1`).** Up to v2.11 this list put the column headers on row 2. On the real files row 2 is **blank** and the column headers are on **row 3**. Examples: `trades_20250610.1` and `orders_20250610.1`, rows 1–3. This matches the OMIE file-format document, which says the report header "finaliza con una línea en blanco" (*Modelo de Ficheros…*, v1.38, §3.2). Loaders should find the header row by its content (it starts with `Fecha;Contrato;`), not by its row number.

**Number format (v2.12; hard format rule; source: OMIE, *Modelo de Ficheros para la distribución pública de Información del mercado de electricidad*, v1.38, 22/09/2026, §3.2 and §5.1.1.1).**
- **General rule (§3.2):** "El separador de miles será, prácticamente en todos los ficheros, el carácter punto (".") y el de decimales será el carácter coma (","). En los que no sea así se indicará explícitamente en el formato del fichero." Also: "Los números negativos llevarán un signo menos ("-") sin separador."
- **Orders and trades files (`Precio`, `Cantidad`, `Cantidad reducida`, `PPD`): Spanish format.** `.` is the thousands separator, `,` is the decimal separator, and `-` is a leading minus sign.
  - Examples: `18,01` = 18.01; `-9.999,00` = −9,999.00; `-1.249,90` = −1,249.90; `0` = 0 (`PPD` may be written as a bare integer).
  - On disk, a thousands separator appears only in `Precio` values with |price| ≥ 1,000. Measured on all of `data/rawOMIE/` at the `OMIE-PRICE-THOUSANDS-SEP-1` Step-2 review (agile-plan v5.166): 478,932 order `Precio` cells and 10 trade `Precio` cells, all inside [−9,999, +9,999] (§2.3), and all correctly grouped (regex `[+-]?\d{1,3}(\.\d{3})+,\d+`). Verified again for this version, for example `orders_20250610.1` row 202234: `-9.999,00`. Quantities never reach 1,000 MW in the sample, so they never carry a `.`. The rule still applies to them.
  - **Consequence for a parser:** in these files a `.` is never a decimal point. Replacing `,` with `.` without first removing the thousands `.` gives a wrong or unparseable value.
- **`marginalpdbc` (day-ahead marginal price) files: DOT decimal, no thousands separator.** This is the explicit exception that §3.2 allows. The format table gives the price fields as `F8.2 (0.0 a 99999.99)`, and the example rows are written `2025;02;05;1;59.65;59.65;` (§5.1.1.1).
  - Raw data: `marginalpdbc_20250601.1` rows 2–4 read `2025;06;01;1;60;60;`, `…;2;36.07;36.07;`, `…;3;35.01;35.01;`. No cell in any `marginalpdbc_*` file on disk contains `,`.
  - **Consequence:** a `.` in `marginalpdbc` IS the decimal point. A parser that removes every `.` unconditionally reads 36.07 as 3,607. The Spanish-format rule must therefore apply only to the orders and trades families (or, equivalently, only when the cell contains `,`).
- How the loader enforces this (strictness, errors) is an implementation decision and is not prescribed here.

### 13.1 Orders file columns

| # | Column (Spanish) | Column (English) | Type | Example | Notes |
|---|---|---|---|---|---|
| 1 | Fecha | Date | date | `01/06/2025` | **Session day** (DD/MM/YYYY). It always equals the file date, also on rows of the D+1 product. **It is NOT the delivery day** (corrected v2.10; `orders_20250626.1`: 0 rows with a different `Fecha`, including its 45,446 rows of product 06-27). |
| 2 | Contrato | Contract | string | `20250602 04:00-20250602 04:15` | Delivery window (start–end). **The only field that carries the delivery date**: the product is (start date, start slot), §13.4. DST days use special strings (OQ-30). |
| 3 | Zona | Bidding zone | string | `10YES-REE------0` | EIC code; Spain=`10YES-REE------0`, Portugal=`10YPT-REN------W` |
| 4 | Agente | Agent | string | `ACCGR` | OMIE market participant code |
| 5 | Unidad | Unit | string | `ACCGC02` | OMIE bidding unit or portfolio code (sub-entity of agent); required for settlement and dispatch (see §13.5) |
| 6 | Precio | Price | float | `18,01`, `-9.999,00` | EUR/MWh; Spanish number format (`.` thousands, `,` decimal; *Number format* above) |
| 7 | Cantidad | Quantity | float | `7,50` | MW; Spanish number format |
| 8 | Tipo oferta | Side | char | `C` or `V` | `C`=Compra (buy), `V`=Venta (sell) |
| 9 | Condiciones a la ejecución | Exec restriction | string | empty, `ICE`, `IOC`, `FOK` | Empty=regular (NON); `ICE`=iceberg |
| 10 | Condiciones a la validez | Validity | string | `GTD` | `GTD` when explicit; GFS orders may show empty or `GTD` |
| 11 | Cantidad reducida | Peak quantity | float | `5,80` | Iceberg visible slice size; empty for non-icebergs |
| 12 | PPD | Price per delta | float | `0,50`, `-1,00` | ICEwPI per-slice price increment; `0` for standard icebergs; empty for non-icebergs |
| 13 | Momento de envío | Submission time | datetime | `01/06/2025 17:41:12` | DD/MM/YYYY HH:MM:SS format. Header spelled with an accent (`envío`, Latin-1). Always carries a time part, including at exact midnight (`… 00:00:00` occurs; no date-only cell exists in any orders file on disk; contrast §13.2.1). |

### 13.2 Trades file columns

| # | Column (Spanish) | Column (English) | Type | Example | Notes |
|---|---|---|---|---|---|
| 1 | Fecha | Date | date | `01/06/2025` | **Session day** (equals the file date; NOT the delivery day; corrected v2.10) |
| 2 | Contrato | Contract | string | `20250602 04:00-20250602 04:15` | Delivery window; the only field carrying the delivery date (§13.4) |
| 3 | Agente compra | Buy agent | string | `SECOB` | Buyer participant code |
| 4 | Unidad compra | Buy unit | string | `SOECC01` | Buyer bidding unit or portfolio; required for settlement (see §13.5) |
| 5 | Zona compra | Buy zone | string | `10YES-REE------0` | Buyer's zone |
| 6 | Agente venta | Sell agent | string | `FEN` | Seller participant code |
| 7 | Unidad venta | Sell unit | string | `FENC01` | Seller bidding unit or portfolio; required for settlement (see §13.5) |
| 8 | Zona venta | Sell zone | string | `10YES-REE------0` | Seller's zone |
| 9 | Precio | Price | float | `20,48` | EUR/MWh; trade execution price; Spanish number format (`.` thousands, `,` decimal) |
| 10 | Cantidad | Quantity | float | `1,00` | MW; traded quantity; Spanish number format |
| 11 | Momento casación | Match time | datetime | `31/12/2024 22:21:41`; exceptionally `10/06/2025` | DD/MM/YYYY HH:MM:SS. **Header spelled with an accent: `Momento casación`** (Latin-1 byte 0xF3). Exactly-midnight matches are written **date-only**. See §13.2.1. |

The header row is the same in every real trades file (318 files; architect audit, agile-plan v5.169 H-2):
`Fecha;Contrato;Agente compra;Unidad compra;Zona compra;Agente venta;Unidad venta;Zona venta;Precio;Cantidad;Momento casación;`
It also matches the example in the OMIE file-format document (*Modelo de Ficheros…*, v1.38, §5.3.2.7). Only the 11th name carries a non-ASCII letter. A loader that matches header names must treat `Momento casación` and the ASCII spelling `Momento casacion` as the same column.

### 13.2.1 Trades match time (`Momento casación`): format, the date-only midnight convention, and the frame (v2.12, `OMIE-TRADES-MATCH-TS-HEADER-1`)

**Documented format.** OMIE file-format document v1.38 §5.3.2.7 ("Transacciones realizadas en el mercado intradiario continuo") defines the match time as six fields: `DíaCas`, `MesCas`, `AñoCas` (day, month, year of the match) and `HoraCas`, `MinutoCas`, `SegundCas` (hour, minute, second). They are published as the single column `Momento casación`, as in the example `14/08/2018 17:41:48`. The document defines no date-only form. It also states (§5) that the hours in the downloads are Central European time, i.e. the Iberian local wall clock used throughout this spec (DST: OQ-30).

**Observed deviation: date-only cells (data finding; whole-disk scan of `data/rawOMIE/`, repeated for this version).**
- **332** `Momento casación` cells in **77** trades files contain only `DD/MM/YYYY`, with no time part.
- Every one of them equals the date of its own file, i.e. the session date `Fecha`.
- **0** trades cells are written `… 00:00:00`.
- The neighbouring seconds are well populated: 809 cells at `00:00:01`, 3,182 cells in `00:00:01`–`00:00:09`, and 381 at `23:59:59`. So 332 is about the count expected for the missing second `00:00:00`.
- Date-only rows sit between rows stamped `23:59:5x` of D−1 and rows stamped `00:00:xx` of D (spot-checked here; whole-disk per the architect, agile-plan v5.169 H-4). Example, `trades_20250610.1` lines 3349–3352: `09/06/2025 23:59:59` → `10/06/2025` (k18) → `10/06/2025` (k19) → `10/06/2025 00:00:10`.
- Orders files are different: they write exact midnight explicitly (`… 00:00:00` occurs, e.g. 179 cells in `orders_202506{10,11,14,18,19}.1`), and **0** orders cells are date-only.

**Ruling (data-format convention; strong inference, not stated by OMIE).** In a trades file, a `Momento casación` cell that is exactly `DD/MM/YYYY` means **00:00:00 local of that date**. The trades exporter drops a time part that is exactly midnight. The cell is a valid match time, not a missing value, and it must not be dropped or defaulted.
- **Scope:** trades only. This convention is not evidenced for orders `Momento de envío` (no such cell exists there).
- **Evidence covers only cells whose date equals the file date.** A date-only cell with any other date has never been observed. Midnight of any other date falls outside the file's session window [22:20 D−1, 22:00 D) (§13 correction), so such a cell would be an anomaly with no evidenced reading.

**Frame: CONFIRMED.** A trade is a match in the product's book, so its match time lies in the product's trading window, like a submission time (§13.4).
- Product (D, k) is traded in [15:20 D−1, GCT(k)). Its trades therefore satisfy $t_{\text{match}} \in [55{,}200,\ 165{,}600)$ s, measured from D−1 00:00 local.
- This half-open interval contains every product's window, as for orders. The tighter per-product bound is $t_{\text{match}} < \mathrm{GCT}(k)$.
- A date-only cell in file D maps to $t = 86{,}400$ s (D 00:00).
  - All 332 date-only rows belong to products of delivery day D with delivery start ≥ 02:00, i.e. $k \ge 8$. No date-only row has a `Contrato` start at 00:xx or 01:xx.
  - For $k \ge 8$, $\mathrm{GCT}(k) = 60\lfloor k/4\rfloor - 60 \ge 60$ min after D 00:00. So 86,400 s lies strictly inside the window of every product that carries such a row.
  - A date-only row could not belong to a product (D+1, ·): those open at 15:20 on D.
- Architect measurement (agile-plan v5.169 H-5; 314 complete non-DST delivery days, 15,343,605 product trade rows): every match time is in [55,202, 165,599] s, 0 rows are outside the frame, and every match is ≥ 3,601 s before its delivery start. This agrees with the ruling.

**Not prescribed here:** how the loader handles unparseable or empty match-time cells (raise vs default), and whether it validates the window. Both are implementation decisions (see the card).

### 13.3 Observed statistics (June 2025)

Based on analysis of `orders_20250601.1` (~387,000 orders):
- **Product mix:** 100% 15-minute contracts (hourly-to-15min transition complete for Spain CID)
- **Exec restriction breakdown:** ~87% regular (NON), ~12% ICE (iceberg), ~1% IOC, <0.01% FOK
- **Bidding zones:** Spain (`10YES-REE------0`) and Portugal (`10YPT-REN------W`)
- **PPD range for ICEwPI (empirical only, non-authoritative):** observed in `orders_20250601.1` within approximately −1.55 to +1.45 EUR/MWh on that date. This is a sample statistic, not a rule-level bound. Later OMIE dates have exhibited `PPD = −3.00 EUR/MWh` on ICEwPI rows (e.g. `data/rawOMIE/` for delivery date 2026-01-12, contract 81, agent `ISMSL/ISMSC01`, BUY). See §13.6 for the authoritative PPD envelope.
- **Agent/Unit relationship:** Each agent may have multiple trading units (Unidad)

### 13.4 Notes for data loading (M7)

- **No order_id in public data:** Order IDs are internal to XBID/SOB. The data loader must assign sequential order_id values during parsing.
- **No order status in public data:** The orders file is a submission log, not a lifecycle log. Status is a simulator-internal concept.
- **Unit field (Unidad):** Required on both orders and trades. The Unidad (bidding unit or portfolio code) is the granularity at which OMIE performs settlement (OMIE Rulebook §50.1: settlement entries are recorded "by each bidding unit or portfolio"), the granularity at which orders are submitted (§45.1: "Several sale and/or purchase orders may be submitted for the same contract being traded and for the same trading unit"), and the granularity at which dispatch programs are communicated to System Operators (§46.2, §47). The data loader must parse and preserve Unidad on orders and both Unidad compra / Unidad venta on trades, and the simulator order representation must carry the unit code. See domain finding DA-UNIDAD-2026-04 below for full rationale.
- **Contract-to-ID mapping:** The delivery window string (e.g. `20250602 04:00-20250602 04:15`) maps deterministically to an integer contract_id (0–95) via: `contract_id = (start_hour * 4) + (start_minute / 15)`. **This slot index alone does NOT identify a product (v2.10).** See the next bullet.
- **A delivery product is (delivery date, slot) (v2.10, `OMIE-TAPE-CONTRACT-DATE-1`; hard rule; OQ-27).**
  - The tradable product (§2) is the pair **(D, k)**. D is the **start date** in `Contrato` and k is the slot above. Two rows with the same k but different `Contrato` start dates are **different products**, with separate books (§6.4, no cross-contract matching).
  - A loader that keys on k alone from one file mixes product D with product D+1. This was the pre-v2.10 behaviour, and it produced two half-products per tape.
  - **"Tape (D, k)" and the `eval_split.json` `delivery_date` mean the delivery product (D, k)** (Q1 ruling).
- **Which files a product spans (hard rule, confirmed in the data).**
  - Product (D, k) is traded from its GOT (15:20 on D−1, §3.1) to its GCT (60·⌊k/4⌋ − 60 min from D 00:00, §3.1), with the IDA2 gap 22:00–22:20 on D−1 and, for k ≥ 48, the IDA3 gap 10:00–10:20 on D (§3.3).
  - With the session partition above, its rows sit in **exactly two session files**:
    - **file D−1**: sends or matches in [15:20, 22:00) on D−1;
    - **file D**: sends or matches in [22:20 D−1, GCT(k)).
  - This applies to orders AND trades, and so to the set of (Agente, Unidad) pairs in the product.
  - **File D+1 NEVER contains product D, for any k.** GCT(k) ≤ 22:00 on D for every k, because k = 92–95 close together at 22:00 on D (§3.1), and file D+1 starts at 22:20 on D.
- **The architect-derived "plus file D+1 when k ≥ 93" rule is REFUTED.** The derivation assumed quarter-hour GCTs of 22:15/22:30/22:45 on D.
  - **Regulatory basis:** Rules Annex 2 Figure 4 lists only "1-96 (D+1)" as open from 22:00 on D.
  - **Data basis:**
    - `orders_20250626.1` and `trades_20250626.1` each have 0 rows of any product dated 2025-06-25.
    - The 10,383 rows of product (06-25, k = 93–95) are all in `orders_20250625.1`, the last sent at 21:59:29.
    - `orders_20260110.1` has 0 rows of any 2026-01-09 product.
    - `orders_20250625.1` has 0 rows of any 2025-06-24 product.
- **Submission-time window of a product (wall clock, local).** All rows of product (D, k) fall in [15:20 D−1, GCT(k)) ⊆ [15:20 D−1, 22:00 D). Measured in seconds from D−1 00:00, that is [55,200, 165,600).
  - For downstream consumers: the upper bound 168,300 quoted on the `OMIE-TAPE-CONTRACT-DATE-1` card assumed the refuted k ≥ 93 rule and quarter-hour GCTs. Under the domain window above, a product's own rows never reach it.
  - The choice of timestamp frame is an implementation decision and is not prescribed here.
- **Completeness.** A product is complete only if **both** session files D−1 and D exist, for orders AND for trades. A product with a missing file is incomplete and must be **excluded, not loaded partially** (Q4 ruling; OQ-29). Example: 2025-05-31 (no `orders_20250530.1` / `trades_20250530.1` on disk).
  - **CORRECTION (2026-09-30, OMIE-TAPE-CONTRACT-DATE-1):** the v2.10 example was 2025-06-01. It is withdrawn because the user added `orders_20250531.1` and `trades_20250531.1` on 2026-09-30, so product 2025-06-01 is now complete (contract 0 has 2,546 order rows: 1,731 in `orders_20250531.1` + 815 in `orders_20250601.1`). The rule itself has not changed. Which products are incomplete is a fact about the disk, not a rule. See the OQ-29 snapshot and check the disk at load time.
- **DST days.** On the two Europe/Madrid transition days the `Contrato` strings change form and a day has 92 or 100 contracts (OMIE Rules Annex 2 Figures 2–3). The slot formula above does not apply as written. See OQ-30 (open).
- **Zone filtering (removed):** Phase 1.5-Z retired the single-zone simplification. The data loader no longer filters by zone; all Iberian orders and trades (Spain and Portugal) are loaded. `OMIEDataConfig` does not contain a `zona` field. The zona column is parsed but not used for filtering.

### 13.5 Domain finding DA-UNIDAD-2026-04: Unidad on orders and trades

**Verdict: Required on both orders and trades.**

The Unidad (bidding unit / portfolio code) is semantically load-bearing in the OMIE Continuous Intraday Market for the following reasons:

1. **Settlement granularity.** OMIE Rulebook §50.1 (OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md, pp. 118-119): "The Market Operator shall specify the clearing of energy for each agent participating on the continuous intraday market for the sale and purchase orders that have been matched for each contract with one or more of its bidding units or portfolios." The settlement formula DCMIC(d,c,t,u,r) is parameterized by `u` (bidding unit or portfolio). Settlement entries are recorded "in the register kept for such purposes by each bidding unit or portfolio." PnL attribution in the simulator must therefore be at unit level, not agent level, to faithfully represent settlement.

2. **Order submission identity.** OMIE Rulebook §45.1 (p. 103): "Several sale and/or purchase orders may be submitted for the same contract being traded and for the same trading unit that shall be processed separately." Orders are submitted per bidding unit or portfolio (§45.2.2, p. 104: "The bidding unit or portfolio for which the order is submitted is valid, in force, and the agent is responsible for submitting orders for that unit"). The Unidad is a mandatory order field.

3. **Dispatch program communication.** At the end of each round, "the Market Operator shall notify the corresponding system operators... the total volume of energy in the bidding unit corresponding to the trades arranged" (§46.2, p. 113). Portfolio units must be disaggregated to bidding units (§47, pp. 113-115). Each bidding unit maps 1:1 to a scheduling unit (unidad de programacion) registered with the System Operator (§11, p. 28: "A market agent cannot have a bidding unit without a corresponding programming unit, nor can there be a programming unit without an associated bidding unit").

4. **OMIE public data publications.** The orders file includes Unidad as column 5. The trades file includes Unidad compra (column 4) and Unidad venta (column 7). OMIE Rulebook §15.b mandates publication of "the bids sent to the European Continuous Trading Platform" with 90-day delay; these publications include the unit field.

5. **Regulatory reporting (CNMC/REMIT).** CNMC Resolution DCOOR/DE/009/24 (p. 15, lines 760-762) references REMIT reporting obligations at the "unidad de oferta" (bidding unit) level. Non-compliance with REMIT reporting at this level can lead to suspension of specific bidding units (OMIE Rulebook §10, pp. 26-27).

**Anti-self-match / cross-prevention.** No rule preventing matches between orders from the same agent or the same unit was found in the source corpus. The XBID Continuous Trading Matching Algorithm (Continuous_Trading_Matching_Algorithm_Public_Description.md, §5.1) describes matching purely on price-time priority with no ownership-based exclusion. The OMIE Rulebook §46.1 defers matching rules to the European Continuous Intraday Trading Platform (CACM). The source documentation is silent on self-trade prevention at either the agent or unit level. This silence is noted as OQ-12 in open-questions.md.

**Cardinality.** Unidad is a sub-entity of Agente (N:1). Each bidding unit is uniquely coded and owned by exactly one agent (OMIE Rulebook §11, p. 28). An agent may own multiple bidding units. There is no evidence in the corpus of a unit belonging to multiple agents simultaneously, though shared-ownership bidding units exist where multiple agents share the economic exposure of a single unit (§45.2.2, p. 105: "If the bidding unit is under shared ownership, the owner agents are to have a sufficient operating limit to cover the percentage of the order under their ownership"). A unit's parent agent does not change within a trading session.

**Previous MVP deferral revised.** The prior decision at line 567 of this document ("Deferred to future work") is hereby superseded. Unidad is required on the simulator's order representation and trade output because settlement, dispatch, and regulatory reporting all depend on unit-level identity. Agent-level identity alone is insufficient.

### 13.6 Domain finding DA-PPD-2026-04: Authoritative PPD envelope for ICEwPI

**Verdict: No numeric magnitude bound exists at the rule level. Only a sign constraint is authoritative.**

Hard rules (authoritative, from source corpus):

1. **Sign rule (hard).** ICEwPI buy orders must have `PPD < 0`; ICEwPI sell orders must have `PPD > 0`. Sources: OMIE Rulebook §4.5 ("Iceberg purchase orders can only be entered with negative price increments, while sell orders can only be entered with positive price increments", OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md lines 10280-10281); OMIE Details of the Intraday Market's Operation (OMIE_Details_of_the_Intraday_Market_s_Operation.md lines 322-323); XBID Continuous Trading Matching Algorithm §3.3 ("reduced by the peak price delta for buy orders and increased by the peak price delta for sell orders", Continuous_Trading_Matching_Algorithm_Public_Description.md lines 276-278).

2. **Implicit magnitude ceiling from SIDC price limits (derived, not explicit).** Because each slice price equals `previous_slice_price + PPD` and every slice price must remain within the SIDC harmonized price range (-9,999 to +9,999 EUR/MWh; see OQ-01), `|PPD|` cannot exceed 19,998 EUR/MWh and is effectively constrained by the distance from the base limit price to the far SIDC bound. No document in the corpus states a tighter numeric bound on the magnitude of `PPD` itself.

Non-rules (explicitly absent from the source corpus):

- No OMIE Rulebook article, CNMC resolution, ACER decision, XBID Info Package section, or matching-algorithm clause reviewed specifies a minimum or maximum numeric magnitude for `PPD`. Searches across the full domain-analyst corpus for "price increment", "peak price delta", "incremental de precio", and related terms return only the sign rule and the definitional statements above.

- No tick-size rule governs `PPD` independently. The general SIDC tick size (see OQ-06) applies to prices, from which slice prices inherit their tick-alignment; this does not by itself bound `|PPD|`.

Empirical observations (for reference only; not rule bounds):

- `orders_20250601.1` sample: `|PPD|` up to approximately 1.55 EUR/MWh.
- `data/rawOMIE/` delivery date 2026-01-12, contract id 81: `PPD = -3.00 EUR/MWh` observed on ICEwPI BUY rows (agent `ISMSL`, unit `ISMSC01`, contracts 81-83). This invalidates the prior approximate-range statement in §13.3.
- Empirical `|PPD|` is therefore unbounded from the perspective of the market-spec; test envelopes must be derived empirically from the specific dataset under test, not from a fixed spec constant.

Authoritative statement for downstream implementers:

- Treat `PPD = 0` as "standard iceberg" and `PPD != 0` as "ICEwPI".
- Validate the sign rule (buy => negative, sell => positive) as a hard constraint.
- Do not encode a fixed numeric magnitude envelope for `PPD` in simulator validation or tests. Any numeric envelope used for robustness/sanity checking must be justified as an empirical choice tied to a specific dataset, not cited as a market rule.

**Sources:**
- OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md §4.5 (lines 10274-10281)
- OMIE_Details_of_the_Intraday_Market_s_Operation.md lines 319-323
- Continuous_Trading_Matching_Algorithm_Public_Description.md §3.3 (lines 276-278), §5 (lines 515-517)
- OMIE Rulebook 2025 §4.5 (OMIE_RULEBOOK_2025.md lines 10274-10281)
- CNMC_DCOOR_DE_009_24.md line 14832 ("Ofertas Iceberg con Incremental de Precio", Spanish equivalent, same sign-only rule)
- Negative evidence: exhaustive corpus search for numeric PPD bounds returned no hits (see OQ-13).
- Empirical counter-example: `data/rawOMIE/` delivery 2026-01-12 contract 81, PPD = -3.00 EUR/MWh.

---

## 14. Contract order-flow magnitude and tape-selection semantics (DA-TAPE-2026-07)

**Status:** Domain ruling for story `P3-PERF-TAPE-1` (shorter liquid-hour tape). Filed 2026-07-03.
**Scope:** what the domain does and does not say about "liquid" / "peak-hour" contract selection, and the defensible spec-grounded proxy for choosing a *shorter* tape that still exercises real fills. This section rules on domain criteria only; the config/loader wiring (`config/env_configs/omie_single_contract.json` `data_source` block + `build_fixed_tape_set` args) is a developer deliverable, not a domain artifact.

### 14.1 There is no liquidity / peak-hour rule in the corpus (negative finding → OQ-21)

An exhaustive search of the domain-analyst corpus (`docs/omiecid/corpus/`) for a per-contract, per-hour, or per-delivery-period **liquidity or traded-volume ranking**, and for any "liquidity concentrates near gate closure" statement, returns **no authoritative rule**. Every corpus hit for "liquid*" resolves to *liquidación* (settlement), not market liquidity. The OMIE Rulebook, the XBID Continuous Trading Matching Algorithm description, the XBID Information Package, ACER Decisions 03/2025 and 11/2024, OMIE Instruction 1/2024, and the CID educational summary describe *how* matching, gate timing, and settlement work — none rank contracts by expected order flow or trade volume, and none assert that activity intensifies as delivery approaches.

**Consequence:** the term "liquid peak-hour tape" used informally in `outputs/cid_rl/runs/stage0_20260703_585f775/report.md` §"3 candidate optimizations" #3 has **no market-spec definition to appeal to.** Selecting a tape by "liquidity" is therefore not a spec-derivable operation. This gap is tracked as **OQ-21** (`docs/omie/open-questions.md`). The rest of this section gives the best *spec-grounded proxy* — do not read it as a discovered liquidity rule.

### 14.2 Spec-grounded proxy for per-contract order-flow magnitude: GOT→GCT trading-window length

The one deterministic, spec-derivable quantity that bounds how much order flow a single contract's book can accumulate is the **length of its continuous trading window (GOT→GCT)**:

- **Common GOT.** All 96 quarter-hourly contracts of delivery day D open **simultaneously at ~15:20 CET on D−1** (§2.2, §3.1; OQ-02 resolution; OMIE Instruction 1/2024 §7; OMIE Market Rules Annex 2 schedule table). Every contract of a given delivery day shares the same gate-open instant.
- **Per-contract GCT.** Each contract closes at **GCT = delivery-period start − 1 hour** (§3.1; Hard Rules §11.1; OMIE Details of Intraday Market §3). Contracts close on a rolling basis, ~4 per hour-long round.
- **Deterministic contract_id → delivery time.** `contract_id = start_hour*4 + start_minute//15` (§13.4), so a contract's delivery time (and therefore its GCT, and therefore its window length) is a **known function of its `contract_id`** with no data access required.

> **Precision (v2.10).** GCT is hour-aligned (§3.1): the four contracts of one delivery hour share a GCT. So "strictly increasing" below should read **non-decreasing, stepping by 60 min every four contracts**. Window length = 460 + 60·⌊k/4⌋ min from the 15:20 GOT (contracts 0–3: 7 h 40 min). The ordering conclusion ("smallest `contract_id` = shortest window") is unchanged.

Because GOT is common and GCT is a strictly increasing function of the delivery time within day D, **the GOT→GCT window length is a strictly increasing function of the contract's day-D delivery time.** The earliest-delivering day-D contracts (Q1–Q4 = `contract_id` 0–3, delivering 00:00–01:00 CET on D, GCT 23:00–00:00 on D−1) have the **shortest** trading windows (~7.7–9.7 h); late-delivery day-D contracts (e.g. Q80–Q96, delivering ~19:45–23:45) have windows exceeding a full day. Since orders can only be submitted while the gate is open, a shorter window is a **domain-defensible upper bound on cumulative order count**, hence on the loader's `n_orders_padded` and the derived `steps_per_episode`.

> **This proxy is a bound on *capacity to accumulate flow*, not a measured liquidity ranking.** It is domain-defensible (it follows directly from the gate-timing hard rules) and requires no invented liquidity concept, but it does **not** guarantee a given short-window contract is non-degenerate — a thin overnight-delivery contract could clear the window bar and still have too few fills. §14.4 supplies the independent non-degeneracy bar that must also be met.

**Note on the current baseline (context, not a rule).** The reproducibility-anchor contract is `contract_id 16` (delivery 04:00–04:15 CET on D; GCT 03:00 on D; window from 15:20 D−1 ≈ 11.7 h), which is an early-morning off-peak delivery hour that nonetheless carried 5459 loaded orders in the June-2025 window (report §2). Naively "picking a peak-demand hour" would select a **longer-window, higher-flow** contract — the opposite of the throughput goal. The window-length proxy correctly points the shorter-tape search toward **early day-D delivery hours** (small `contract_id`), not toward peak-demand evening hours.

### 14.3 Defensible tape-selection axes (expressible without any `OMIEEnv` / matching-code change)

The shorter window MUST be expressed purely as a tape-selection choice through the existing `data_source` block and `build_fixed_tape_set` (P3.3 loader-glue), never as an env-code change. Two domain-defensible axes exist:

1. **Contract-id selection (PRIMARY; available today).** Choose a `contract_id` whose GOT→GCT window is short per §14.2 — i.e. an **early day-D delivery hour, small `contract_id`** — subject to the §14.4 non-degeneracy bar. This is directly expressible as the existing `data_source.contract_ids` list and requires no loader change. The delivery date(s) remain a free axis (`data_source.delivery_dates`); the ruling does not constrain which in-window dates are used beyond the additive/data-availability constraints in §14.5.

2. **Per-tape order cap (SECONDARY; requires a NEW loader arg).** A hard cap on the number of real orders loaded per tape (e.g. keep the first `K` chronologically-sorted orders, `load_omie_orders` already sorts by submission time — §"Step 3" of `load_omie_orders`) directly bounds `n_orders_padded` and hence `steps_per_episode` **independently of which contract is chosen.** This is domain-defensible **only** as a truncation of the *earliest* portion of the GOT→GCT flow (submission-time order preserved), i.e. it models "the market up to submission-time cutoff T" — a valid partial-session view. `build_fixed_tape_set` / `OMIEDataConfig` have **no such argument today**; adding one is the developer's job. A cap is the most reliable mechanical lever because it bounds tape length regardless of contract choice, but it changes the *market state at episode end* (the book never sees late flow) and so is the secondary, not primary, lever.

   > **Do NOT truncate to "the final N minutes before GCT."** The card floats "the last N minutes before GCT" as a candidate. The corpus contains **no rule** that liquidity concentrates near GCT (§14.1), so a *tail* truncation is not domain-defensible as a "most-liquid slice." A *head* truncation (first-K by submission time) IS defensible because it is simply an earlier gate-closure — a shorter but complete-from-open session — which the gate-timing rules do support. If a future measurement establishes tail concentration empirically, that becomes a data-driven engineering choice, still not a market rule (record under OQ-21).

### 14.4 Non-degenerate-fills criterion (domain intent behind locked test `test_shorter_tape_still_exercises_real_fills`)

A shortened tape must still test the Stage 0/1 *learnability* question, which requires **agent-reachable fills** — a real crossing book, not a degenerate near-empty one. The domain-grounded, verifiable bar is:

**Qualitative bar (the ruling).** Over the chosen contract's GOT→GCT window (or the head-truncated prefix, if a cap is applied), the historical order flow must have **actually crossed and produced executed trades on BOTH sides** — i.e. the tape must contain resting bid liquidity and resting ask liquidity that meet, so that an agent posting a marketable order into that book can realistically be filled. A tape whose historical flow produced (near-)zero trades, or trades on one side only, is degenerate and fails the experiment's premise regardless of how fast it runs.

**Directly verifiable from real data (no simulation needed).** OMIE publishes a per-delivery-day **trades file** (`trades_YYYYMMDD.1`, §13.2) alongside the orders file. Filtering it by the chosen `contract_id` (via the same `_parse_contract_id` mapping, §13.4) yields the **historical executed trades** for that contract. The non-degeneracy bar is met iff that filtered trade set is non-trivial. This is a real-data fact about the chosen (contract, date), independent of simulator behavior — the historical market itself either produced fills on that contract or it did not.

> **Correction (v2.10).** Trades files are split by session day, the same way as the orders files (§13, §13.4). "The trades of product (D, k)" means the rows of `trades_{D−1}.1` **and** `trades_D.1` whose `Contrato` start is (D, k). It does not mean every slot-k row of `trades_D.1`, which mixes products D and D+1.

**Defensible minimum (a floor, not a market rule).** The corpus sets no minimum trade count (that would be an invented rule — see §14.1/OQ-21). As a defensible non-degeneracy floor for the locked test, require that the chosen (contract, date) tape clears **all** of:
- **≥ 1 executed trade** in the historical `trades_YYYYMMDD.1` for that `contract_id` (the book demonstrably crossed at least once); and
- **both sides present in the order flow** — the loaded tape contains at least one BUY (`C`) and at least one SELL (`V`) order (§13.1 column 8), so a crossing is structurally possible; and
- **a scripted-agent rollout over the shortened tape produces a non-zero simulated fill count** (the operational form the test-agent will assert — this is the "agent-reachable" half of the bar, on top of the historical-crossing half).

The "≥ 1 historical trade + both sides present" pair is the *domain* floor (a fact about the real tape). The "scripted rollout produces non-zero fills" is the *behavioral* confirmation that the shortened tape is agent-reachable in the simulator. The test-agent may set a stricter numeric threshold if the throughput/learnability tradeoff warrants it, but ≥ 1 historical trade with both sides present is the minimum below which the tape is provably degenerate. **These numeric floors are non-degeneracy engineering thresholds, not market rules; cite them as such.**

### 14.5 Data-availability and additivity constraints (for the developer)

- **Additive, not mutating.** The current D-D window (`contract_id 16` / `2025-06-01..03`) is the throughput-baseline reproducibility anchor and MUST be preserved (orchestrator decision). The shorter window is a NEW tape pointer added alongside it, wired as an additive/parametrised `data_source` selection — not a replacement of the D-D block.
- **Real OMIE data only.** `data/rawOMIE/` (OMIE-native XBID CSVs), NEVER LOBSTER, per `data_source.loader = "omie"`.
- **No-assume-data.** This ruling does **not** assert that any specific short-window `contract_id`/date is present on disk or clears §14.4. On-disk availability (gitignored, local; VERIFIED 2026-07-14 by `P3-DATA-OMIE-EXPAND-1`) is **asymmetric between orders and trades**, and the trades side is the binding constraint:
  - **Orders:** ~**105 order-days** across FOUR bands — 2025-06 (30d), 2025-10 (31d), 2025-12 (31d), and a PARTIAL 2026-01 (13d).
  - **Trades:** **ONLY 2025-06 (30 trade-days)**. No trades exist on disk for 2025-10, 2025-12, or 2026-01.
  - **Trades are the binding constraint.** Trades gate the replay `match_rate` diagnostic (`ReplayResult.match_rate` in `omie_data_loader.py`) and the §14.4 non-degeneracy check (≥1 historical trade + both sides present). Orders alone suffice for *training* (the scripted/RL agent submits its own orders), but any date whose §14.4 bar must be cleared needs a trades file too — so **only June 2025 clears §14.4 today**. Any candidate outside June 2025 requires fetching that month's trades first (see `docs/data/omie-layout.md` "How to fetch additional months"; canonical coverage matrix in `docs/data/omie-layout.md` and `docs/cid_rl/experiment-plan.md` §"Data reality").
  - **CORRECTION (2026-09-30, OMIE-TAPE-CONTRACT-DATE-1): the three bullets above are a superseded 2026-07-14 snapshot.** They are kept for history only. They do not describe the disk today and they were never a rule. **On-disk snapshot, verified 2026-09-30 (`data/rawOMIE/`):**
    - Session files are continuous from **2025-05-31 through 2026-04-14** (319 session days). There are **318 `trades_*.1`** files (the only gap is `trades_20260406.1`) and **317 `orders_*.1`** files (the gaps are `orders_20260228.1` and `orders_20260406.1`).
    - Trades are therefore no longer confined to June 2025. The v2.8 statement "only June 2025 clears §14.4 today" is stale. Any complete product (OQ-29) now has the trades files needed for the replay `match_rate` diagnostic and the §14.4 non-degeneracy check. Whether a given (contract, date) actually clears §14.4 must still be checked (step (iii) below).
    - Incomplete products (OQ-29) given this snapshot: **2025-05-31**, **2026-02-28**, **2026-03-01**, **2026-04-06**, **2026-04-07**, and any delivery date from 2026-04-15 onward (no `*_20260415.1`).
    - File dates are **session** days (§13, v2.10), so product D needs files D−1 and D. The "months" above are session-file months, not delivery months.
    - `docs/data/omie-layout.md` is **stale** on this point. Its "Orders-vs-trades coverage matrix (VERIFIED on disk 2026-07-14)" still shows trades for June 2025 only, and its file-naming note still says `YYYYMMDD` "is the delivery date" (refuted by §13 v2.10 / OQ-27). That file is not owned by the domain analyst, so the refresh is left to its owner. Until then, check the disk rather than the matrix.
    - This is a snapshot, not an assumption. Re-check the disk at load time.

  The developer MUST, for the candidate (contract, date): (i) confirm the `orders_YYYYMMDD.1` **and** `trades_YYYYMMDD.1` files exist under `data/rawOMIE/` (per the asymmetry above, the trades file is the one most likely absent outside June 2025 — *CORRECTION 2026-09-30: stale; see the 2026-09-30 snapshot. Under v2.10 §13.4, this means files D−1 **and** D for both orders and trades*); (ii) confirm the loaded tape's `n_orders_padded` and derived `steps_per_episode = ceil(n_orders/orders_per_step)` are materially smaller than the 5459/1365 baseline (locked test `test_shorter_tape_reduces_steps_per_episode`); and (iii) confirm the §14.4 non-degeneracy bar (locked test `test_shorter_tape_still_exercises_real_fills`). If the first candidate short-window contract fails (ii) or (iii), escalate the contract_id toward a slightly longer window (or apply a §14.3(2) head cap on a known-good longer-window contract) until both hold — the domain constraint is "shortest window that still clears the non-degeneracy bar," not a single hard-coded contract.
- **Recommended search order (domain guidance, not a fixed value).** Start from the shortest-window day-D-early-delivery contracts (small `contract_id`, e.g. the 00:00–03:00 delivery band, `contract_id` ~0–11) present on disk, take the first that clears §14.4, and prefer it over the baseline `contract_id 16`. If none of the early-delivery contracts on the available dates clear the non-degeneracy bar (plausible for deep-overnight hours), fall back to a §14.3(2) head-cap of the baseline-style contract rather than selecting a long-window peak-demand contract (which would defeat the throughput goal).

**Sources:** §2.2, §3.1 (gate timing), §11.1 (hard rules: common GOT, GCT = delivery−1h), §13.1/§13.2 (orders/trades file columns), §13.4 (`contract_id` mapping, chronological sort); OQ-02 resolution (simultaneous GOT / rolling GCT); OMIE Instruction 1/2024 §7; OMIE Market Rules Annex 2 schedule table; OMIE Details of Intraday Market §3. Negative finding: exhaustive corpus search for a liquidity/volume ranking or near-GCT concentration rule returned no hits → **OQ-21**.
