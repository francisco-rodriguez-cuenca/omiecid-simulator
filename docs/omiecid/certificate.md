# OmieCID-Simulator — Certificate

**Status:** OmieCID-Simulator 0.1.0 — certification of scenarios C00–C16 on the released package
**Date:** 2026-10-26, amended 2026-10-27 (C14 and C16 re-runs) and 2026-10-29 (editorial; see [Appendix A](#appendix-a--amendments))
**Scope:** the OMIE Continuous Intraday **matching engine** (`omiecid_simulator/omiecid/`)
**Specification:** [`scenario-ladder.md`](scenario-ladder.md) · **Rules:** [`market-spec.md`](market-spec.md) (v2.14, amended 2026-10-20)
**Verdict log:** [`certification-log.md`](certification-log.md)

This certificate records the certification of OmieCID-Simulator 0.1.0: the seventeen scenarios
C00–C16 of the ladder, each re-run on the released package and judged by its own notebook and its
independent audit test.

The certificate holds one section per scenario. A section is **signed** when the scenario is
`CERTIFIED` and **unsigned** when it is `NOT CERTIFIED`. It states what is **not** certified as
carefully as what is: see the [conformance gaps](#conformance-gaps), the
[engineering limits](#engineering-limits-the-ladder-works-around) and the
[residual risk](#residual-risk).

**Evidence.** Each signed section names the scenario's entry in the certification log and the
blake2b-128 digest of its results payload's canonical JSON
(`notebooks.certification.cert_support.results_digest`: sorted keys, separators `,` and `:`); the
results files are `notebooks/certification/results/C<nn>.json`. The results files are published with
the release, not committed; the published files' own blake2b-256 digests are listed in `DIGESTS.b2`. In the section's
**Evidence** line, the `branch` field carries the package version and the `commit` field the
results digest.

---

## What "certified" means here, and what it does not

The ladder generates datasets whose trade outcome is **known in advance**, runs them through the
engine, and checks the outcome. The knowledge comes from three layers, each covering the others'
blind spot:

- **Goldens** — a hand-computed answer for one specific input. Exact, and the only evidence that
  distinguishes "self-consistent" from "right", but it does not scale past a handful of orders.
- **Parametric goldens** — a closed-form expectation over a *randomly drawn* construction's own
  parameters. This is how a random dataset still has a known answer, and it is the ladder's workhorse.
- **Conservation and metamorphic invariants** — properties that must hold on every run, and required
  relationships between two related runs. They scale to arbitrary random input and catch the defects
  the other two are blind to.

**Every check was seen failing before it was trusted.** Each invariant, and the record builder that
feeds them, was shown to catch a deliberately broken input. A check that cannot see something (the
matching-iteration count, hidden order identities) reports that axis as *unobserved* rather than
passing it.

**What a pass does not rule out** is stated in [Residual risk](#residual-risk), below the gap and
limit lists: chiefly, there is no independent second engine, so a shared misreading of
`market-spec.md` would survive all three layers.

---

## Coverage matrix

*(One row per scenario; Status, Date and Entry are those of the scenario's latest, governing log entry,
written by `close_scenario`. Status vocabulary: `CERTIFIED` · `CERTIFIED — provisional on C<nn>` ·
`NOT CERTIFIED` (section unsigned).)*

| Scenario | Certifies (market-spec) | Status | Date | Entry |
|---|---|---|---|---|
| C0 Inertness and the contract-status gate | §3.1, §5.4, §9.2 | CERTIFIED | 2026-10-26 | 1 |
| C1 Order validation and rejection | §2.3, §4.2, §5.4 | CERTIFIED | 2026-10-26 | 2 |
| C2 The single crossing pair | §6.1, §6.2 item 3, §11.1 | CERTIFIED | 2026-10-26 | 3 |
| C3 Price-time priority, ties, ownership blindness | §6.1, §5.2, §6.6, §11.1 | CERTIFIED | 2026-10-26 | 4 |
| C4 Cancellation | §5.1, §6.1 | CERTIFIED | 2026-10-26 | 5 |
| C5 Partial fill and the resting residual | §5.1, §6.2 item 4, §5.2 | CERTIFIED | 2026-10-26 | 6 |
| C6 Multi-level walk and volume-weighted price | §6.2 item 6 | CERTIFIED | 2026-10-26 | 7 |
| C7 Immediate-or-cancel | §4.5, §6.2 item 5, §5.1 | CERTIFIED | 2026-10-26 | 8 |
| C8 Iceberg regeneration | §4.2.1 R2/R3/R4 (R2 *Incoming iceberg* items (2), (4) as amended 2026-10-20), §4.2, §5.2, §6.2 item 7 | CERTIFIED | 2026-10-26 | 9 |
| C9 Price-incremented icebergs | §4.2, §4.2.1 R2, §6.2 item 7, §13.6 magnitude and sign (wrong-sign ICEwPI asserted rejected) | CERTIFIED | 2026-10-26 | 10 |
| C10 Fill-or-kill, in three parts | §4.5, §4.2.1 R1/R2, §6.2 item 7 | CERTIFIED | 2026-10-26 | 11 |
| C11 Expiry at gate closure | §3.1, §5.1, §8.5 | CERTIFIED | 2026-10-26 | 12 |
| C12 Settlement and the cash leg | §8.1, §8.4, §11.1 | CERTIFIED | 2026-10-26 | 13 |
| C13 Anonymity and information visibility | §7.1, §7.2, §7.3, §9.2; §4.2.1 R2 *Incoming iceberg* items (1), (4) as amended 2026-10-20 | CERTIFIED | 2026-10-26 | 14 |
| C14 Capacity edges | *(no spec section — implementation limits)* | CERTIFIED | 2026-10-27 | 20 |
| C15 Randomised fuzz | §4.5, §6.1, §6.2, §6.6; §4.2.1 R2 *Incoming iceberg* items (1), (2), (4), (6) as amended 2026-10-20 | CERTIFIED | 2026-10-26 | 16 |
| C16 Multi-contract isolation | §6.4, §2.1, §2.2, §8.1 | CERTIFIED | 2026-10-27 | 19 |

### Incoming-iceberg instance chain, by rule component

Every incoming iceberg, standard ($d = 0$) and ICEwPI, is a chain of instances of size $P$
(`market-spec.md` v2.14 §4.2.1 Rule 2 *Incoming iceberg*, items (2), (4) and (6) as amended
2026-10-20; OQ-33 revised, OQ-34 resolved). Here $b$ is the incoming price, $d$ the increment, $M$ the
swept quantity, $R$ the remainder, $k_e = \lfloor M/P \rfloor$ and $u = M - k_e P$. Only C08, C13,
C15 and C16 exercise a crossing incoming iceberg, so only they can certify a component (C16 was certified on its re-run,
log entry 19). "Audit test"
below means the scenario's independent audit test, which is evidence for the notebook's check, not a
separate certification.

| Rule component | Certified by | How it is exercised | Not exercised, or not certified |
|---|---|---|---|
| **Per-instance limits** — instance $k$ trades only where $b + k\,d$ crosses (item (6)) | **C15** | The independent reference walk W applies the limit for every $d$; 241 draws have a trading incoming ICEwPI. The audit test's own walk equals the results on all 2,200 draws (the superseded base-limit rule misses 613) | C08 draws $d = 0$ only, where every limit is $b$ (finding F15); C13 has no ICEwPI (F21); C09 and C10 drive NON and FOK aggressors, not incoming ICEwPI. So for $d \neq 0$ the only certifying evidence is C15's reference walk, with no hand golden |
| **Stop** — the sweep ends at the first instance that cannot trade, even where $b$ still crosses (item (6)) | **C15** | 160 draws with an instance-limit stop, checked against W | The C16 audit test adds an engine probe of an ICEwPI stop resting at $k_e$; C08 and C13 do not reach a stop |
| **Continuation** — the remainder keeps its place as instance $k_e$ at $b + k_e\,d$, and later instances continue from there (item (4)) | **C15**, **C08** | C15: 378 draws with a $u > 0$ remainder, 146 with a print on the continued ladder. C08 `incoming_remainder` (250 draws): the remainder rests as instance $k_e$ at the iceberg's price ($d = 0$, so the price continuation is trivial there) | C13: every sweep stays in instance 0 ($k_e = 0$). The C16 audit test's price-setter helper continues at $k_e$ |
| **Display $v_1 = \min(P - u, R)$**, hidden $R - v_1$ (item (4); display C3) | **C08**, **C15**, **C13** | C08 `incoming_remainder`: $u > 0$ on all 250 draws, $v_1 < R$ on 217, so the display is tested non-vacuously. C15: W shows $\min(P - u, R)$. C13: asserted on the 4 draws where its premise fails (all $u > 0$); elsewhere $k_e = 0$ and $u = f$, so the rule-implied difference is zero by construction | **Derived from OMIE data, not stated by the sources** (99.93 % of 2,699 decisive cases; OQ-34), and adopted as the simulator's rule. See [Residual risk](#residual-risk) |
| **Row split** — a trade row ends at a resting entry's end or at an incoming instance boundary (item (2)) | **C08**, **C15**, **C16** (certified on its re-run, log entry 19) | C08: every row $\le P$ and every multiple of $P$ below $M$ a cumulative row boundary. C15: W emits one row per (resting entry, incoming instance) for every $d$. C16: each lane's rows equal its isolated run's; the audit test's hand lane C is derived under item (2) | The fallback on a saturated trade ring (rows left unsplit, all or nothing) is not certified by any scenario: C14 was not authored against the row split. See the gap list |

Two further consequences follow from items (1) and (3), which the amendment left byte-identical: the
incoming iceberg aggresses with its total quantity, and every row prices at the resting entry's limit.
C08 and C15 assert both on every incoming draw.

---

## Conformance gaps

Behaviours the specification requires that this simulator does not implement, or does not implement as
the rule says, and the simulator limits that cut into a certified rule. They are listed because an
unimplemented rule missing from a certificate reads as if it were certified. Status vocabulary:
**Closed** (fixed and asserted) · **Not implemented** · **Not represented** (outside the simulator's
scope) · **Implemented, not certified** · **Open question** (the market's behaviour is not stated by
the sources) · **Accepted limit** (a simulator limit, recorded rather than fixed). Every row except
the closed one is **not certified**.

| Rule | Status | Note |
|---|---|---|
| §13.6 PPD sign rule | **Closed** | The validator rejects a wrong-sign ICEwPI (market-spec v2.14 §5.4; OQ-23); C9 asserts the rejection. |
| §4.6 mid-session GTD lapse | **Not implemented** | Validity and expiry are carried on the order, but there is no mid-session expiry sweep, and the session hardcodes good-for-session. C11 certifies the at-gate sweep only. |
| §9.2 trade side | **Not implemented** | Spec only (v2.14 §9.2 addendum; no engine change). `get_agent_trades` (`omiecid_simulator/omiecid/contract_book.py`) reports price, quantity and time, not side. Not certified (C13 F16). |
| §8.3 per-unit (Unidad) settlement | **Not implemented** | `OMIEOrder` has no unit field and settlement is agent-level; the loader folds `(Agente, Unidad)` into one identifier. A real gap, since §13.4/§13.5 require the loader to preserve Unidad. |
| §5.3 order modification and its timestamp reset | **Implemented, not certified** | Excluded by scope decision, not by absence. `modify_order` exists; §5.3 calls the priority reset *"a hard rule from the XBID system"*. One hard rule therefore goes uncertified. |
| §3.3 rounds and the two-phase IDA suspension | **Not implemented** | The SUSPENDED status exists, but nothing drives the transitions. Its `!= OPEN` rejection is conformant for submission (C0). |
| §3.3 (v2.10) survival of resting orders through the IDA window, and the reopen batch match | **Not implemented** | No engine path (OQ-28); the replay carries the book across the gap as a recorded simplification. |
| §3.4 halts, hibernation, reactivation | **Not implemented** | — |
| §4.3 / §4.4 block and basket orders, §4.5 AON | **Not implemented** | Independent baskets are, by definition, sequential submissions; what is absent is any basket construct. |
| §5.4 guarantee / operating-limit and TSO-unavailability checks | **Not implemented** | C1 certifies the implemented subset and names these as non-coverage. |
| §8.2 trade firmness, §8.5 imbalance, §2.4 notification thresholds | **Not represented** | Outside the simulator's scope. |
| §13.6 item 2 drifted instance or slice price within ±9,999 EUR/MWh | **Open question** (OQ-35; not implemented) | Drifted ICEwPI prices are not clamped: a probe shows a slice resting at 10,000.00, where no valid counter bid can cross it. The market's handling (clamp, end the iceberg, refuse at entry) is not stated by the sources. The instance chain (a remainder starts at instance $k_e$) makes the edge slightly more reachable. Not certified. |
| §4.2.1 Rule 2 item (2) row split, on a saturated trade ring | **Accepted limit** (OQ-15) | When the 100-row trade ring has no room for the split rows, an incoming iceberg's rows stay unsplit, all or nothing; volumes and prices are unchanged. On a saturated ring `SubmitResult.trade_qty` can also under-report the quantity taken off an incoming iceberg. The split makes a saturated ring more frequent (18 calls of ten high-instance real contracts need 100 rows or more, against 1 without it); no evaluation tape saturates. Not certified: C14 was not authored against the split. Detail under [Engineering limits](#engineering-limits-the-ladder-works-around). |
| Regeneration at the `book_depth` cap: pass-count-dependent regeneration order and heal, observed only at `book_depth` so far | **Accepted limit** | While a side is at `book_depth`, each matching pass with fills regenerates slices and can debit the hidden quantity of a resting iceberg whose visible slice was evicted, so the outcome depends on the pass count. Trades and `SubmitResult` are unaffected at the call where it happens; the slot order of the side table and the within-level order of regenerated slices can differ. A dead side-table row (hidden 0, no visible slice) is observed at one real-tape call. Every scenario except C14 stays below the cap by construction, so the class is not certified. Detail under [Engineering limits](#engineering-limits-the-ladder-works-around). |

**Known inconsistencies in `market-spec.md` itself** (named, not fixed by the ladder):

- **§3.2's IDA table** mixes two day frames: it times IDA1/IDA2 "(D−1)" and IDA3 "(D)" but scopes all
  three "of D+1". All three serve delivery day D. Its quarter-hour periods are correct under the
  current OMIE Rules (domain review v3.0, ruling R-2).
- **§3.3's phase-2 citation** (Matching Algorithm §5.5) concerns capacity-update batch matching; the
  IDA-window suspension rests on OMIE Rules §58.2 (e)–(f) and Instruction 1/2024 §8. The rule is
  right, the source is wrong (ruling R-1). C0 quotes the phase description, not the citation.

---

## Engineering limits the ladder works around

None of these is a market rule. Every scenario except C14 is generated to stay clear of them: at most
80 resting orders per side, 80 trades, a hop bound of 160 and 800 icebergs. C14 is authored to certify what
happens at each limit, and it is CERTIFIED in this release (log entry 20); the unsplit rows on a saturated ring and the depth-cap heal class
are recorded, not certified, and appear in the gap list above.

- A regenerated iceberg slice that rests on a full side does **not** raise `book_full` (C14a′).
  **Pass-count-dependent regeneration order and heal (observed only at `book_depth` so far).** Each
  matching pass with fills calls `regenerate_slices`, and while a side is at `book_depth` that call
  debits the hidden quantity of a resting iceberg whose visible slice was evicted. So the outcome
  depends on how many passes a call makes. This is a risk at the cap, not an observed loss: at one
  real-tape call the engine keeps the affected BUY icebergs at `slice_count` 1, hidden 50, visible 50
  (correct; no trade touches them). At another real-tape call the engine keeps a dead side-table row
  (hidden 0, no visible slice; `n_active` 133 against 132). The within-level order of regenerated
  slices can also differ; the side table is compared as a set keyed by order ID, since its slot order
  is not observable. Trades and `SubmitResult` are unaffected at the call where this happens.
- The matching-iteration count cannot be read after a run; that axis is reported as unobserved.
- A **saturated trade ring** collapses every further trade into its last row (C14b). Also, an incoming
  iceberg's rows stay unsplit when the ring has no room for the split rows (all or nothing; volumes and
  prices unchanged; `market-spec.md` v2.14 §4.2.1 Rule 2 *Incoming iceberg*, classification). On a
  saturated ring, `SubmitResult.trade_qty` can under-report the quantity taken off an incoming
  iceberg, and the counter side can lose more than the incoming order; the stepped sweep's extra
  passes make this more frequent. Splitting rows at instance boundaries also fills the 100-row ring
  sooner: on ten high-instance real contracts, 18 calls need 100 rows or more, against 1 without the
  split.
- An iceberg needing more slices than the per-iceberg slice cap raises the shared `truncated` flag at
  registration, even if it never trades, and a raised flag does not say which ceiling was reached
  (C14e: 65 slices leave the flag false, 66 set it). The edge is reachable on real tapes: OQ-34's scan
  counts 89 of 16,806 crossing ICEwPI entries whose ladder depth $\lceil Q/P \rceil$ exceeds 64
  (max 124).
- In the truncated regime (C14c only), the NON re-submit after the loop does not regenerate icebergs.
- The order-book tape's empty-slot detector reads the right value for the wrong reason (it indexes a
  column that happens to hold −1 on empty rows).

---

## Out of scope, because not applicable

- §6.3 batch matching and §6.5 cross-border capacity — single-zone simulation; batch matching is
  triggered only by a cross-border capacity update.
- §13.x loader and data-feed semantics — a data-layer concern, not a matching-engine one. One
  exception is pulled in: §13.6's PPD sign rule, drawn sign-valid in C9, whose wrong-sign draws
  assert its rejection (the closed row above).

---

## What this certificate cannot say about historical replay

The ladder certifies the **engine**. It cannot certify how faithfully the *replay heuristic* reproduces
history, and the distinction matters.

OMIE's public data is a submission log with no cancellations and no lifecycle events. A replay
therefore cannot know whether an unmatched order was withdrawn or rested until the gate: the possible
fates look identical in the file (OQ-10). The loader assumes the order rested, which leaves "phantom"
liquidity in the simulated book.

OQ-25 attributed a low passive-fill rate for an added participant to the same mechanism: phantom
residuals queueing ahead of it at equal price under §6.1. **That attribution is in doubt.** The figure
was measured while added orders were stamped at the *front* of the queue; this was corrected (see the
dated correction to OQ-25), and the figure has not been re-measured.

The engine behaviour underneath is **in scope**: C3 certifies exactly the priority rule involved. A
synthetic tape sidesteps the phantom problem by construction, because it has no unobserved
cancellations. What the ladder disclaims is the **tape**.

---

## Residual risk

A passing certificate says that the engine produced the known outcome on every check the ladder ran.
It does not rule out the following.

**1. A shared misreading, since there is no second engine.** There is no independent second
implementation of the matching rules. An expectation and the engine written from the same misreading
of `market-spec.md` would agree, and the disagreement would survive goldens, parametric goldens and
invariants alike. The defence is **independence, not redundancy**, and it reduces the risk without
eliminating it:

- **separate authors** — each scenario's expectation is written from the specification only, never
  from the engine, and engine changes are made against tests their author may not edit;
- **audit tests** — each scenario has an independent audit test that re-derives the expectation by
  its own route and checks the notebook's results against it. The defence works in both directions:
  an audit test has disagreed with the engine before, and the review found the audit test's
  hand table, not the engine, out of date;
- **placement** — C2, C9 and C10 pin price attribution, where a shared error would most plausibly hide;
- **every check seen failing** before it was trusted, and **byte-exact quotes**, each checked against
  its source by digest, so a notebook cannot drift from the text it claims to certify.

A fully independent reference engine remains possible as a later upgrade; it is not part of this
certificate. One rule component rests on a single independent derivation: for $d \neq 0$, the
per-instance limits and the stop of an incoming ICEwPI are certified only through C15's reference walk
(see the [instance-chain table](#incoming-iceberg-instance-chain-by-rule-component)).

**2. Conformance to this repository's rulings, not to XBID.** `market-spec.md` §4.2.1's iceberg
rules are recorded there as *this repository's rulings*. The §4.2.1 preamble describes them as
deviations from the literal SIDC/XBID description, while each rule's classification records it as
consistent with the market rules (domain review v3.0, V3-21). The incoming-iceberg instance chain rests
on a modelling ruling and on OMIE data: no corpus text states the incoming-side row split explicitly
(OQ-33). The iceberg and fill-or-kill scenarios therefore certify conformance **to those rulings**.

**3. Display C3 is derived from data, not stated by the sources.** The remainder of an incoming
iceberg shows $v_1 = \min(P - u, R)$, the unused part of instance $k_e$. The sources say only "at most
the reduced part" (Annex §4.4). The choice fits 2,697 of 2,699 decisive real cases under the one-live-row
filter (99.93 %; 99.78 % unfiltered), and it is adopted as the simulator's rule (OQ-34). C08, C13 and
C15 certify that the engine follows that rule, not that OMIE's book does in every case. The same
evidence leaves a recorded, unexplained residual for the per-instance limits: 5.10 % of filtered
ICEwPI entries print worse than their instance limit (0.73 % of domestic ones), concentrated where an
unobserved counter-source is possible.

**4. The replay-fidelity gap.** The certificate covers the engine, not the replay heuristic. OMIE's
public data carry no cancellations, so a replayed book holds phantom liquidity (OQ-10). The real
cancellation rate cannot be observed, so the distortion on a real tape is unknown, and the OQ-25
passive-fill attribution has not been re-measured since the queue-stamp correction.

**5. What the cash-leg check means.** C12 certifies that the cash leg sums to zero, exactly. It does
not claim the trades were economically neutral: §8.4 adds an explicitly exogenous energy-value term to
PnL. And it cannot catch a trade at the wrong price, since settlement is computed from the same trade
rows. Mispricing is certified by C2 and by invariant I5 on every scenario.

**6. Finite draws inside a margin.** Every scenario except C14 draws a finite random sample inside the
80/80/160/800 envelope; behaviour at and beyond the ceilings rests on C14, which is CERTIFIED in
this release (log entry 20); the unsplit rows on a saturated ring and the depth-cap heal class are recorded, not certified. Four axes are unobserved on every scenario (independent
valuation, matching iterations, queue-timestamp correctness, rejection reason) and are reported as such,
never passed.

**7. Scenarios that do not exercise the amended rule.** All seventeen scenarios were re-run on the
released engine. Those other than C08, C13, C15 and C16 were not authored against the 2026-10-20
incoming-iceberg amendment, which none of them exercises.

**8. The `Session` is verified, not certified.** The ladder certifies the engine. C00, C13 and C16 run
through `Session`, which is otherwise verified by a byte-identical parity check against the research
environment. The one allowed difference is the self-trade attribution: OMIE has no self-trade
prevention (OQ-12), so in a self-match the session attributes the passive side to the participant as
well as the aggressive side. No scenario certifies the documented `Session` limitations: the
participant's trade tape has one passive row per step, so `compute_agent_pnl` over it is exact only when
a step's passive fills go one way (use `agent_cash`); batched calls under `jax.vmap` need an outer
`jax.experimental.enable_x64()`; and under the default span-scaled clock, `done` can fire at the first
slice whose largest timestamp is within half a scaled minute of $S$, possibly several slices before a
tape's last real slice, after which the remaining rows are not replayed (the full-tape replay recipe
sets `tape_seconds_span` to at least twice the largest real timestamp over the tape set). See
[Limitations](../limitations.md).

---

## Scenario sections

One section per scenario, replaced in place when the scenario is re-closed (the log keeps every run); written by `notebooks.certification.cert_support.close_scenario`, unsigned by `notebooks.certification.cert_support.unsign_scenario`, never by hand.

### C00 — Inertness and the contract-status gate

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 1 · results blake2b-128 28a3833bcea0b9c8a87a2a25c318a809
- **Certifies:** §3.1 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §3.3 (market-spec.md v2.14); §3.3 (market-spec.md v2.14); §3.3 (market-spec.md v2.14); §3.3 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); R-1 (scenario-ladder-review.md v3.3); §3.1 (market-spec.md v2.14); §3.3 (market-spec.md v2.14)
- **Evidence:** `notebooks/certification/results/C00.json` · log entry 1 · branch `omiecid-simulator 0.1.0` · commit `28a3833bcea0b9c8a87a2a25c318a809`

### C01 — Order validation and rejection

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 2 · results blake2b-128 6c958a8d4735a67e422effb13d1d8d25
- **Certifies:** §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §10 (scenario-ladder-review.md v3.3); §5.4 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §2 (scenario-ladder-review.md v3.3); OQ-16 (open-questions.md OQ-16 · What is known); §2.3 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §2 (scenario-ladder-review.md v3.3); §5.4 (market-spec.md v2.14)
- **Evidence:** `notebooks/certification/results/C01.json` · log entry 2 · branch `omiecid-simulator 0.1.0` · commit `6c958a8d4735a67e422effb13d1d8d25`

### C02 — The single crossing pair

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 3 · results blake2b-128 f2200c780852bee04849c6dca540790b
- **Certifies:** §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §11.1 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14)
- **Evidence:** `notebooks/certification/results/C02.json` · log entry 3 · branch `omiecid-simulator 0.1.0` · commit `f2200c780852bee04849c6dca540790b`

### C03 — Price-time priority, ties, and ownership blindness

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 4 · results blake2b-128 d53a623cc7f6d0e6ca39f11d92d94ff3
- **Certifies:** §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §11.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §13.1 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14)
- **Evidence:** `notebooks/certification/results/C03.json` · log entry 4 · branch `omiecid-simulator 0.1.0` · commit `d53a623cc7f6d0e6ca39f11d92d94ff3`

### C04 — Cancellation

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 5 · results blake2b-128 7b9305f936e350f0d24f218700b9306e
- **Certifies:** §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §9.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5.3 (market-spec.md v2.14); §5.3 (market-spec.md v2.14); OQ-20 (open-questions.md OQ-20 · Why the test is still load-bearing); §2 (scenario-ladder-review.md v3.3)
- **Evidence:** `notebooks/certification/results/C04.json` · log entry 5 · branch `omiecid-simulator 0.1.0` · commit `7b9305f936e350f0d24f218700b9306e`

### C05 — Partial fill and the resting residual

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 6 · results blake2b-128 d779bb22bd3e51b112b77481b2769dc3
- **Certifies:** §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §8 (scenario-ladder-review.md v3.3); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14)
- **Evidence:** `notebooks/certification/results/C05.json` · log entry 6 · branch `omiecid-simulator 0.1.0` · commit `d779bb22bd3e51b112b77481b2769dc3`

### C06 — Multi-level walk and volume-weighted price

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 7 · results blake2b-128 0ade9998fd2a7353e59fa87086b5b65d
- **Certifies:** §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); OQ-18 (open-questions.md OQ-18 · Context.); §5 item 2 (scenario-ladder-review.md v3.3)
- **Evidence:** `notebooks/certification/results/C06.json` · log entry 7 · branch `omiecid-simulator 0.1.0` · commit `0ade9998fd2a7353e59fa87086b5b65d`

### C07 — Immediate-or-cancel

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 8 · results blake2b-128 a8547bc3774f338e17d1425be5ba4f06
- **Certifies:** §1 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §6.2 item 5 (market-spec.md v2.14); §16.1 (scenario-ladder-review.md v3.3)
- **Evidence:** `notebooks/certification/results/C07.json` · log entry 8 · branch `omiecid-simulator 0.1.0` · commit `a8547bc3774f338e17d1425be5ba4f06`

### C08 — Iceberg regeneration

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 9 · results blake2b-128 3c0b0891729194a0c0c276df58280a46
- **Certifies:** §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §6.2 item 7 (market-spec.md v2.14); OQ-31 (open-questions.md OQ-31 · Resolution (2026-10-02); OQ-31 (open-questions.md OQ-31 · Consequences.); OQ-33 (open-questions.md OQ-33 · Revised by user ruling (2026-10-20); OQ-33 (open-questions.md OQ-33 · Revised by user ruling (2026-10-20); OQ-33 (open-questions.md OQ-33 · Revised by user ruling (2026-10-20); OQ-33 (open-questions.md OQ-33 · Revised by user ruling (2026-10-20); OQ-34 (open-questions.md OQ-34 · Resolved by user ruling (2026-10-20); OQ-34 (open-questions.md OQ-34 · Resolved by user ruling (2026-10-20); OQ-20 (open-questions.md OQ-20 · Ruling.); OQ-23 (open-questions.md OQ-23 · Q2 — Rule-4 verdict); §17.3 (scenario-ladder-review.md v3.3); D3-08 (scenario-ladder-review.md v3.3)
- **Evidence:** `notebooks/certification/results/C08.json` · log entry 9 · branch `omiecid-simulator 0.1.0` · commit `3c0b0891729194a0c0c276df58280a46`

### C09 — Price-incremented icebergs

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 10 · results blake2b-128 7bc9a7dddcb4b2e79a365547aed6627d
- **Certifies:** §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §6.2 item 7 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); OQ-23 (open-questions.md OQ-23 · **(a) The PRIMARY ICEwPI-SELL probe); OQ-20 (open-questions.md OQ-20 · Ruling.); §16.1 (scenario-ladder-review.md v3.3); D3-08 (scenario-ladder-review.md v3.3)
- **Evidence:** `notebooks/certification/results/C09.json` · log entry 10 · branch `omiecid-simulator 0.1.0` · commit `7bc9a7dddcb4b2e79a365547aed6627d`

### C10 — Fill-or-kill, in three parts

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 11 · results blake2b-128 69257a4285a32b797492d02137fd1a64
- **Certifies:** §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §6.2 item 7 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); OQ-11 (open-questions.md OQ-11 · Two interpretations considered:); OQ-11 (open-questions.md OQ-11 · Two interpretations considered:); OQ-11 (open-questions.md OQ-11 · Resolution:); OQ-11 (open-questions.md OQ-11 · Resolution:); OQ-20 (open-questions.md OQ-20 · Ruling.); D3-08 (scenario-ladder-review.md v3.3)
- **Evidence:** `notebooks/certification/results/C10.json` · log entry 11 · branch `omiecid-simulator 0.1.0` · commit `69257a4285a32b797492d02137fd1a64`

### C11 — Expiry at gate closure

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 12 · results blake2b-128 c4fdca867994b353aca37683c51ac003
- **Certifies:** §3.1 (market-spec.md v2.14); §4.6 (market-spec.md v2.14); §4.6 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.1 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); V3-16 (scenario-ladder-review.md v3.3); V3-16 (scenario-ladder-review.md v3.3); V3-16 (scenario-ladder-review.md v3.3); OQ-20 (open-questions.md OQ-20 · Ruling.); §4.2.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); V3-01 (scenario-ladder-review.md v3.3); §8.5 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); D3-08 (scenario-ladder-review.md v3.3); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14)
- **Evidence:** `notebooks/certification/results/C11.json` · log entry 12 · branch `omiecid-simulator 0.1.0` · commit `c4fdca867994b353aca37683c51ac003`

### C12 — Settlement and the cash leg

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 13 · results blake2b-128 47e2891739352d8b532762edabdd7bb2
- **Certifies:** §8.1 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §10 (market-spec.md v2.14); §11.1 (market-spec.md v2.14); §8.4 (market-spec.md v2.14); §8.4 (market-spec.md v2.14); §8.4 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); V3-03 (scenario-ladder-review.md v3.3); V3-03 (scenario-ladder-review.md v3.3); V3-04 (scenario-ladder-review.md v3.3); OQ-18 (open-questions.md OQ-18 · Context.); OQ-18 (open-questions.md OQ-18 · Option 3a adjudication); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §6.2 item 5 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); D3-08 (scenario-ladder-review.md v3.3); §8.3 (market-spec.md v2.14); §8.3 (market-spec.md v2.14)
- **Evidence:** `notebooks/certification/results/C12.json` · log entry 13 · branch `omiecid-simulator 0.1.0` · commit `47e2891739352d8b532762edabdd7bb2`

### C13 — Anonymity and information visibility

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 14 · results blake2b-128 66145d154f9c55179da7df53b2115299
- **Certifies:** §7.1 (market-spec.md v2.14); §7.2 (market-spec.md v2.14); §7.2 (market-spec.md v2.14); §7.2 (market-spec.md v2.14); §7.3 (market-spec.md v2.14); §7.3 (market-spec.md v2.14); §7.3 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); OQ-34 (open-questions.md OQ-34 · **Resolved by user ruling (2026-10-20); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §9.1 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.6 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §6.2 item 5 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); OQ-07 (open-questions.md OQ-07 · **Modeling choice for simulator:**); §2 (scenario-ladder-review.md v3.3); D3-08 (scenario-ladder-review.md v3.3)
- **Evidence:** `notebooks/certification/results/C13.json` · log entry 14 · branch `omiecid-simulator 0.1.0` · commit `66145d154f9c55179da7df53b2115299`

### C14 — Capacity edges, one part per ceiling

- **Status:** CERTIFIED
- **Signed:** 2026-10-27 · log entry 20 · results blake2b-128 8bac2ec9648f2cddf8c71c2b6dde1dbd
- **Certifies:** OQ-10 (open-questions.md OQ-10 · **Decision impact:**); §5 (scenario-ladder-review.md v3.3); §5 item 3 (scenario-ladder-review.md v3.3); V3-14 (scenario-ladder-review.md v3.3); OQ-15 (open-questions.md OQ-15 · **What is unknown (engineering invariant):**); OQ-15 (open-questions.md OQ-15 · **Consequence for Phase 2 (observed):**); §11.1 (scenario-ladder-review.md v3.3); §5 item 2 (scenario-ladder-review.md v3.3); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5 item 1 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); V3-15 (scenario-ladder-review.md v3.3); §4.2 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); OQ-20 (open-questions.md OQ-20 · **Ruling.**); §4.2.1 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); V3-15 (scenario-ladder-review.md v3.3); V3-15 (scenario-ladder-review.md v3.3)
- **Evidence:** `notebooks/certification/results/C14.json` · log entry 20 · branch `omiecid-simulator 0.1.0` · commit `8bac2ec9648f2cddf8c71c2b6dde1dbd`

### C15 — Randomised fuzz across the parameter space

- **Status:** CERTIFIED
- **Signed:** 2026-10-26 · log entry 16 · results blake2b-128 f3016bac4479ed878ca56353d842afad
- **Certifies:** §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §6.2 item 5 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §6.2 item 7 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §13.6 item 1 (market-spec.md v2.14); OQ-11 (open-questions.md OQ-11 · **Resolution:**); OQ-20 (open-questions.md OQ-20 · **CORRECTION NOTE (2026-10-07); OQ-20 (open-questions.md OQ-20 · **ANSWER TO CORRECTION-NOTE ITEM (d)); OQ-32 (open-questions.md OQ-32 · **SCOPE CORRECTION (2026-10-08); OQ-33 (open-questions.md OQ-33 · **Revised by user ruling (2026-10-20); OQ-33 (open-questions.md OQ-33 · **Revised by user ruling (2026-10-20); OQ-34 (open-questions.md OQ-34 · **Resolved by user ruling (2026-10-20); OQ-34 (open-questions.md OQ-34 · **Resolved by user ruling (2026-10-20); OQ-34 (open-questions.md OQ-34 · **Resolved by user ruling (2026-10-20); OQ-35 (open-questions.md OQ-35 · **The question.); §2 (scenario-ladder-review.md v3.3); §5 item 2 (scenario-ladder-review.md v3.3); §5 item 3 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); D3-08 (scenario-ladder-review.md v3.3); D3-08 (scenario-ladder-review.md v3.3)
- **Evidence:** `notebooks/certification/results/C15.json` · log entry 16 · branch `omiecid-simulator 0.1.0` · commit `f3016bac4479ed878ca56353d842afad`

### C16 — Multi-contract isolation

- **Status:** CERTIFIED
- **Signed:** 2026-10-27 · log entry 19 · results blake2b-128 59b8d6a93b5b60387514fbeba4410694
- **Certifies:** §2.2 (market-spec.md v2.14); §6.4 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §2.1 (market-spec.md v2.14); §2.1 (market-spec.md v2.14); §2.2 (market-spec.md v2.14); §2.2 (market-spec.md v2.14); §3.1 (market-spec.md v2.14); §3.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §13.6 item 1 (market-spec.md v2.14); §3 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3)
- **Evidence:** `notebooks/certification/results/C16.json` · log entry 19 · branch `omiecid-simulator 0.1.0` · commit `59b8d6a93b5b60387514fbeba4410694`

---

## Appendix A — amendments

Amendment (2026-10-29): editorial; the description of a scenario outside the release was removed.
No scenario section, verdict, signature or results digest changed.
