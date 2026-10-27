# Scenario Ladder — Domain-Analyst Completeness Review

**Status:** v3.3 — D3 return review of the v5.187 battery fixes (`OMIE-CERT-INVARIANTS-1`), signed
**Date:** 2026-10-02 (v3.2, v3.1 and v3.0 signed 2026-10-02; v2.0 signed 2026-09-23)
**Reviewer:** Domain Analyst (`/domain-analyst`)
**Subject:** `docs/omie/scenario-ladder-plan.md` — Track A (the certification ladder), reviewed for
completeness against [`market-spec.md`](market-spec.md): v2.8 (v1.0 pass), v2.9 (v2.0 pass), **v2.12**
(v3.0 pass), and against [`open-questions.md`](open-questions.md) as of 2026-10-01
**Disposition:** v1.0 and v2.0 findings were incorporated into the plan before the first attempt was
approved; sections 1–10 are that signed record, **unchanged**. v3.0 findings (sections 11–13) are
handed to the orchestrator, which incorporates each one or answers it in writing. This document is
the durable record of what was found and how each item was resolved (plan deliverable **D3**).

**Restoration note (v3.0).** Sections 1–10 were restored from the first attempt's rollback copy,
`/tmp/omie-cert-rollback-111213/docs/omie/scenario-ladder-review.md`. It was compared with
`presentations/260924_reunion_directores/fuentes/scenario-ladder-review.md`. The two copies are
identical: 242 lines each, same text line for line. Neither is more complete, so the rollback copy
was used. Apart from the header block above and the v3.0 row of the version history, nothing in the
v2.0 text was edited.

### Version history

| Version | Date | Changes |
|---------|------|---------|
| v3.3 | 2026-10-02 | **D3 return review of the v5.187 fixes (section 17).** Required by §16.3's re-review trigger and V16 as amended. D3-01 (I4 `hop_priority`), D3-02 (I5 `limit_cross`) and D3-03 (ICE-only hop branch) are CLOSED: each v3.2 wrong engine now fails, and the real-engine record passes. The V10 $\tau_s$ amendment (per-dispatch $c_s$) and the V12 M5 counterparty key CONFORM. One new Low residual, D3-08: I4's visible-depth map cannot order two icebergs' regenerated slices. It is recorded, not a precondition. No new OQ. C08 K2 and C12 K2 may proceed unconditionally as far as D3 is concerned (§17.3). |
| v3.2 | 2026-10-02 | **D3 re-review of the merged-candidate battery (section 16).** Required by `OMIE-CERT-INVARIANTS-1` V16 before C08's and C12's K2. Covers V4 (I2 eviction), V8 (I4 queue key), V9 (I5 price-setter, Rule-3 hop, `unruled_hop`) and V10 (I6 tolerance and cash leg), against market-spec v2.13 and OQ-31. Verdicts: I2, I5 and I6 CONFORM; I4's key conforms. 7 items: 2 Medium, 4 Low, 1 informational (D3-01…D3-07). D3-01: I4 checks Rule-3 hop priority on the wrong side, so a wrong engine passes. D3-02: no invariant covers §6.2 item 2 matchability. D3-05 corrects this review's own V3-18 premise. No new OQ. C12 K2 may proceed; C08 K2 may proceed under two conditions (§16.3). |
| v3.1 | 2026-10-02 | **Incorporation check (section 15).** Verified V3-01…V3-21 against `scenario-ladder.md` v2.0 (D2) and the plan's change-table row 32, the HARNESS-1 acceptance criteria and L5. 20 of 21 fully incorporated, 1 partial (V3-04). None of V3-01's five v2.0 regressions remains in D2. Six new items, 2 Medium and 4 Low (N-1…N-6). N-5 corrects this review's own V3-02 wording. OQ-31 filed in `open-questions.md`. |
| v3.0 | 2026-10-02 | **Third pass — delta review of plan revision 2 (2026-09-28, updated 2026-10-02) against market-spec v2.12 and open-questions.md.** Covers every Track A change in the revision (red table rows 1–31, decisions, battery, capacity precondition, C0–C17, out-of-scope list, notebook architecture). 21 findings: 3 High, 13 Medium, 5 Low. Headline: the rewrite lost five signed v2.0 fixes (V3-01). Two rulings recorded for the notebooks to quote, without amending market-spec: the §3.3 phase-2 citation (R-1) and the §3.2 IDA table (R-2, confirmed in part, refuted in part). One new ambiguity needs an OQ before C8 is authored (V3-02). D4 confirmed: the §5.2 rule is in v2.12. |
| v2.0 | 2026-09-23 | **Second pass — reviewed `scenario-ladder.md` v1.0 against market-spec v2.9.** All seven first-pass incorporations verified correct. Found one new citation error (C10 attributed the FOK-visibility sentence to §4.2; it is in §4.2.1 Rule 1), two attribution stretches (C2's ACER/CACM provenance belongs to the price-time-priority row, not the execution-price row; C11 leaned §8.5 further than it says), a ruling-scope mismatch (§5.2's ruling is worded for the *resting* case, C5 drew the *aggressor* case), a systematic quotability hazard (markdown emphasis in the source defeats verbatim substring matching), and two residual coverage gaps (§3.2/GOT schedule; four §9.2 observable rows). All fixed. |
| v1.0 | 2026-09-23 | First pass. Reviewed the ladder as it stood at C0–C13, against market-spec v2.8 and the implementation. Four coverage gaps, three overclaims, one systematic citation error, one ordering inversion, one open point. All incorporated; ladder renumbered C0–C17. |

---

## 1. Headline

The ladder was strong on the **matching core** (§6.1–§6.2, §4.2.1) and weak on everything around it:
order-lifecycle actions (§5.2/§5.3), validation and rejection (§2.3/§5.4), the contract-status
machine (§3.x), anonymity (§7), settlement granularity (§8.3), and the absence of self-trade
prevention (§6.6).

Three findings were **overclaims** — the draft asserted behaviour the simulator does not implement.
Those matter most, because a certification that asserts an unimplemented behaviour produces either a
misleading pass or a confusing failure, and in both cases the certificate becomes untrustworthy.

---

## 2. Coverage gaps

Each was checked against the implementation before classification.

| Behaviour | Spec | Implemented? | Verdict | Resolution |
|---|---|---|---|---|
| **Cancellation** | §5.1 `CANCELLED (by agent)` | **Yes** — `contract_book.py:1528`, with `iceberg_manager.cleanup` | Genuinely missing | **Added as C4.** The non-obvious claim is that survivors keep their relative queue order (array compaction can silently reorder), plus iceberg slot release: a cancelled iceberg's hidden remainder must be invisible to a later FOK pre-check, which reads `iceberg_meta` directly. |
| **Modification resets timestamp** | §5.3 (hard rule, §11.1); §5.2 | **Yes** — `contract_book.py:1575`, cancel-then-resubmit | Genuinely missing; rated **the highest-value gap** | **Subsequently excluded by user scope decision** (third round of corrections). Recorded in the plan as *out of scope by decision, though implemented* — a distinct category from "unimplemented" — so §5.3 stands as a named, uncertified hard rule rather than a silent omission. The test is specified should it return. |
| **Price/qty bounds, tick validation, rejection** | §2.3, §5.4, §4.2 | **Yes** — `order_validator.py:39-98` | Genuinely missing — no scenario exercised the validator at all | **Added as C1**, placed second because every later scenario's random draw assumes its orders are *accepted*; a silently-rejecting validator would let a downstream scenario pass vacuously. The tick-snap boundary (sub-half-tick offsets are accepted and snapped; ≥half-tick commits to a different tick) is documented behaviour that nobody had certified. §5.4's guarantee/operating-limit and TSO-unavailability conditions are unimplemented and named as non-coverage. |
| **Anonymity** | §7.1, §7.2, §7.3, §9.2 | Obs is L1 aggregate | Genuinely missing — no scenario asserted an observation property | **Added as C13**, with the metamorphic formulation: the agent's observation must be bit-identical to one taken from a counterfactual book with the foreign hidden remainder removed and identifiers permuted. Side-conditioned on ownership, since §7.3/§9.2 permit an agent to see *its own* full iceberg quantity. |
| **Contract statuses** | §3.1, §3.3, §9.2 | Partially — all four defined (`contract.py:15-25`), gate tests `== OPEN` only (`contract_book.py:1517`) | Thin | **C0 extended** to all three non-OPEN statuses. The `!= OPEN` collapse is certified as *observed behaviour*, with the §3.3 divergence recorded (SUSPENDED has a distinct meaning the collapse does not express). |
| **Per-unit settlement** | §8.3 (hard rule, OMIE Rulebook §50.1), §13.5 | **No** — `OMIEOrder` has no unit field; `settlement.py:93` is agent-level | Out of scope, unimplemented | **Overclaim removed from C12.** The loader folds `(Agente, Unidad)` into one identifier, so a simulator "agent" already *is* a unit and the sum-over-units identity is untestable. Recorded as a **real conformance gap**, since §13.4/§13.5 require the loader to preserve Unidad. |
| **No self-trade prevention** | §6.6, OQ-12 | Yes — matching is ownership-blind | Was **misfiled**: listed out of scope while the prose claimed it was asserted | **Moved into scope** as metamorphic relation **M6** (agent-id permutation leaves the trade list unchanged, including a self-match), asserted in C3 and swept at scale in C15. §6.6 is a positive, testable requirement, and it costs one cell. |
| **GTD vs GFS** | §4.6, §5.1 | Carried, not enforced — no mid-session expiry sweep; `omie_env.py` hardcodes GFS | Split | **Overclaim removed from C11**, which asserted "a lapsed GTD is unmatchable before the gate". C11 now certifies only the at-gate sweep, which is implemented; **§4.6 is recorded as not fully satisfied**. |
| **Global unique sequential timestamps** | §5.2 | Yes | Missing — C3's whole tie-break claim rests on it | **Folded into C3** as a monotonicity-and-uniqueness assertion. |
| **Iceberg passive-side timestamp exemption** | §5.2 bullet 3 | Yes | Citation missing | **Added to C8**, which previously cited only the §4.2.1 Rule 4 half. |
| **Data-feed semantics** | §13.1–§13.4 | Loader-side | Out of scope for a matching-engine certificate | One exception pulled in: **§13.6's PPD sign rule**, now asserted in C9, with the explicit instruction that the draw range is an empirical envelope and must **never** be cited as a magnitude bound (§13.6 establishes there is none, citing the 2026-01-12 `PPD = −3.00` counter-example). |

---

## 3. Misattributed citations

The plan requires notebooks to quote spec text verbatim, so wrong citations would have propagated
into eighteen notebooks. Two systematic errors and several specific ones:

- **§6.2 is a numbered list**, so "§6.2.4" is not a valid market-spec reference; the correct form is
  "§6.2 item 4". Fixed across C2, C5, C6, C7.
- **Bare "§5.1.2", "§3.3" and similar** mixed *source-document* sections (Continuous Trading Matching
  Algorithm) with market-spec sections. Qualified or replaced throughout.
- **§11.2 is the modeling-choices list**, not capacity limits. The four array ceilings have **no
  market-spec section at all** — corrected in C14, which now grounds itself in
  `book-depth-cap-finding.md`, OQ-10's 2026-06-11 accepted-limitation ruling, OQ-15 and OQ-20, and
  states that it certifies *implementation limits* that are **departures from** §6.1/§6.2.
- **§8.3 was wrongly cited for per-contract settlement** in the multi-contract scenario. Corrected to
  §2.2 (*"Each contract has its own independent order book"*) and §8.1.
- **Zero-sum** is not stated in §8.3 or §8.4. §8.4 defines PnL with an explicitly **exogenous**
  energy-value term, so `Σ PnL = 0` holds of the **cash leg** only. Invariant I6 and C12 narrowed
  accordingly.

The plan's mitigation against recurrence is structural: each notebook renders its quoted passages
from `market-spec.md` at execution time and **fails if the quoted text no longer matches source**.

---

## 4. Ordering

The dependency chain was mostly sound. One genuine inversion and two placements:

1. **Fill-or-kill ran before the iceberg scenarios.** Its hidden-volume and price-incremented parts
   depend on exactly what the iceberg scenarios certify — the ICEwPI part's oracle is literally the
   price-drift sequence summed. A failure would have been unattributable between pre-check counting
   and drift computation, which is the separation that scenario exists to provide. **Reordered:**
   iceberg regeneration (C8) and price drift (C9) now precede fill-or-kill (C10).
2. **Cancellation placed immediately after the priority scenario** it depends on, not appended at the
   end.
3. **The validator placed before everything that draws random orders**, for the vacuity reason above.

---

## 5. Capacity-limit interaction

The plan's principle — every scenario except the capacity one must stay strictly inside the ceilings,
or its result does not count — is domain-correct, and the framing of the caps as *documented
degradations* rather than defects matches OQ-10's accepted-limitation ruling. Three places needed
sharpening:

1. **`max_match_iterations = 200` versus §4.2.1 Rule 4.** Rule 4 requires the cascade to reach a fixed
   point *within the originating matching process*, so the iteration budget is the resource
   conformance is paid out of. Bounding the nominal slice count is insufficient — Rule-3 re-crossing
   adds hops. The generator must emit a per-draw **worst-case hop bound** and the notebook must assert
   it is under the ceiling **and** that truncation did not fire; otherwise a truncated draw is
   indistinguishable from a Rule-4 violation.
2. **`nTrades = 100` is the likeliest accidental collision**, and nothing flagged it. A multi-level
   walk, an iceberg cascade, a settlement draw or a fuzz session can each exceed 100 trades, and a
   wrap silently corrupts I1/I2/I6 — *the very invariants used as the oracle*. Must be a
   **draw-validity precondition checked before the oracle comparison**.
3. **`book_depth = 100` interacts with the priority and walk scenarios specifically**, because the
   documented overflow is a full-side *worst-price level wipe* — which preferentially deletes the
   worse-priced decoys the priority scenario relies on. Such a draw would delete its own control group
   and still pass. Also: `book_depth` and `max_icebergs` are independent axes
   (per `P3-PERF-REGEN-CAP-1`) and the fuzz grid must bound them independently.

Added requirement: the precondition is checked **per draw and the discard rate reported**. A ladder
that silently discards a large fraction of its draws is measuring a biased subspace.

All incorporated as invariant **I8** plus the plan's capacity-validity precondition section.

> **Correction note (2026-10-07, software architect, `OMIE-CAPACITY-DOCS-1`; additive, the review text
> above is unchanged).** Item 1's `max_match_iterations = 200` is not the engine's outer bound. The
> outer match loop runs a fixed `_TRIP_BOUND` = 100 passes and does not read `max_match_iterations`;
> `truncated` fires on either bound, so 200 can never bind first. The iteration budget item 1 speaks
> of is therefore **100 passes**, and its requirement (a per-draw hop bound under the ceiling, plus
> `truncated == False`) reads against 100. A binding bound is flagged (`truncated`), never silent.
> The plan's capacity-validity precondition and C14c are corrected the same way (`scenario-ladder.md`
> v2.3).

---

## 6. The cancellation-fidelity scenario (C17)

**Well-posed as a measurement, provided its claim is scoped tightly.** OQ-10 establishes that
cancelled, GCT-expired and still-resting are **observationally equivalent** in OMIE's public file, so
no real-data experiment can recover ground truth. A synthetic tape with *stipulated* cancellations is
the only construction in which ground truth exists — which is what makes C17 well-posed, and equally
what caps what it can conclude.

**Should measure**, as a sensitivity curve over an explicit cancellation rate: the over-match in
volume and trade count; the phantom resting depth; **the passive fill-rate delta for a reference
resting order** (the OQ-25 starvation face — measuring only over-match measures half the object); and
the price-path distortion at the touch, since the phantom wall mis-sets the touch that Track B's
touch-relative action geometry is defined against.

**Cannot claim:** the real over-match (the true cancellation process is unobservable — that is OQ-10);
anything about engine conformance (it varies the tape, holding the engine fixed); a mandate to fix
(OQ-22 rules the gap acceptable-for-v1; OQ-25 rules the fix locus is tape/replay side only); or
validation of the heuristic against history.

**One design constraint:** the synthetic cancellation process must be **exogenous** — drawn
lifetimes, never book-state-dependent — otherwise it measures a reactive-flow model rather than the
heuristic, and OQ-25 rules the reactive tape a separate, distinct build.

Disposition: incorporated in full; C17 is placed in the certificate as an **appendix on replay
fidelity, explicitly outside the certified surface**.

---

## 7. Out-of-scope list

**Confirmed correct:** §6.3 batch matching (the spec itself says it is not relevant to single-zone);
§6.5 cross-border/ATC; §4.3 block orders (v1 defers); §4.4 baskets — with the refinement that
*independent* baskets are definitionally sequential submissions, so what is absent is any basket
*construct*.

**Additions required** (all unimplemented, all previously readable as "certified by omission"): §3.3
rounds and the two-phase IDA suspension plus the SUSPENDED path; §3.4 halts, hibernation and
reactivation; §4.5 AON; §5.4's guarantee and TSO validations; §8.3 per-unit settlement; §8.2 trade
firmness; §8.5 imbalance; §2.4 notification thresholds; §4.6 mid-session GTD lapse.

**Framing correction on OQ-10/OQ-25:** both are properties of the replay heuristic on *real* data, and
a synthetic tape sidesteps them by construction — correct as stated. But OQ-25's mechanism is §6.1
price-time priority **operating correctly on a wrong book**, so the engine behaviour underneath it
*is* in scope and the priority scenario certifies exactly that rule. The ladder certifies the
mechanism and disclaims only the tape.

**Residual-risk addition:** §4.2.1's four iceberg rules are recorded in `market-spec.md` as **this
repository's rulings**, adopted after noting they deviate from the literal SIDC/XBID description. The
iceberg and fill-or-kill scenarios therefore certify conformance *to those rulings*, not to XBID.
Legitimate and documented, but it must be stated rather than blurred — incorporated next to the
shared-misreading paragraph.

---

## 8. Open point, and its resolution

The specification does not state directly whether a NON residual resting after a partial fill **keeps
or renews** its time priority. §5.2 assigns a new timestamp on *modification*, and a partial fill is
not a modification. The consequence is material: if it renews, a large order that partially fills
loses its queue position, which is a meaningfully different market.

Flagged as requiring a ruling rather than an assertion.

**Resolved — user ruling, 2026-09-23: the residual keeps its time priority.** To be transcribed into
`market-spec.md` §5.2 by the Domain Analyst (plan phase P3). C5 asserts it both on the stored field
and behaviourally, via an interloper that must fill *behind* the residual.

---

## 9. Disposition summary

| # | Finding | Status |
|---|---|---|
| 1 | Per-unit settlement overclaim | Removed; recorded as conformance gap |
| 2 | Mid-session GTD overclaim | Removed; §4.6 recorded as not fully satisfied |
| 3 | Zero-sum overclaim | Narrowed to the cash leg |
| 4 | Citation errors (§11.2, §8.3, §6.2 numbering, source-doc sections) | Corrected; quote-checker prevents recurrence |
| 5 | §6.6 self-match misfiled as out of scope | Moved into scope as relation M6 |
| 6 | Fill-or-kill ordering inversion | Reordered behind the iceberg scenarios |
| 7 | Four coverage gaps (validation, cancellation, anonymity, statuses) | Added as C1, C4, C13, and a C0 extension |
| 8 | No non-vacuity check | Added as invariant I9 |
| 9 | Residual-priority open point | Ruled by the user, 2026-09-23 |
| 10 | Capacity-interaction sharpening (hop bound, trade ring, decoy wipe, discard rate) | Incorporated as invariant I8 |
| 11 | C17 scoping and the missing passive-fill-delta measurement | Incorporated; placed as a certificate appendix |

**Re-review trigger.** This pass reviewed the plan. Per the plan's own D3 verification criterion, the
canonical spec document ([`scenario-ladder.md`](scenario-ladder.md)) is re-reviewed once written, and
any scenario whose expectation changes materially at K2 returns here before it is authored.

---

## 10. Second pass — `scenario-ladder.md` v1.0 against market-spec v2.9

**Verdict: sound on every first-pass incorporation.** All seven were verified in the spec text —
C12 drops §8.3 and narrows zero-sum to the cash leg; C11 drops mid-session GTD lapse; C14 no longer
cites §11.2; C16 cites §2.2; §6.6 is in scope as relation M6; C10 follows C08/C09; and §6.2
references use the "item N" form with no bare source-document section presented as a market-spec one.
The v2.9 §5.2 ruling is correctly reflected in C5 and its provenance is not overstated.

Seven further items were found and fixed:

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | **C10 misattributed the FOK-visibility quote.** *"FOK orders see the iceberg's full remaining quantity for matching purposes"* is in **§4.2.1 Rule 1**, not §4.2 — and §4.2 in fact says icebergs "cannot combine with IOC or FOK behavior", so the citation was not merely imprecise but pointed at text that contradicts the claim. | **High** — would hard-fail the K5 quote checker | Re-cited to §4.2.1 Rule 1, with the §4.2 trap named in the row so it is not reintroduced. |
| 2 | **Systematic quotability hazard.** `market-spec.md` bolds the lead-in of most rule bullets (`- **Buy side:** ranked highest price first`), so a scenario quoting the sentence as a human would write it fails a raw substring test on the asterisks alone. Verified empirically: the §3.1 and §6.1 quotes both blocked. | **High** — would have blocked most scenarios at K5 on correct citations | Fixed in the **checker**, not the spec: `_normalise` now strips markdown emphasis and collapses whitespace, and an ellipsis splits a quote into fragments that must each appear. The matching contract is documented on `SpecQuote` and locked by `tests/omie/certification/test_spec_quote_checker.py` (11 tests, both directions). |
| 3 | **C2 attributed the wrong provenance.** §11.1's *"Execution price = resting order's limit price"* row is sourced to Continuous Trading Matching Algorithm §5.1.1; ACER Decision 11/2024 Art. 5 / CACM Art. 51 source the **price-time-priority** row, which C3 cites. | Medium | Corrected in C2; the ACER/CACM provenance stays with C3 where it belongs. |
| 4 | **C11 leaned §8.5 further than it says.** §8.5 establishes no automatic close-out, cancellation at GCT and trade firmness — it does **not** speak to whether the expiry sweep may print trades, which was the claim resting on it. | Medium | Re-grounded on §3.1 (the literal sweep rule) plus §6.2 (matching is triggered by a new-timestamp order, and an expiry sweep is not one). §8.5 retained for firmness only, quoted as its two actual sentences. |
| 5 | **Ruling-scope mismatch.** §5.2's ruling is worded for a *resting* NON order partially filled; C5's dataset drew the *aggressor* case, where the residual enters the book under §6.2 item 4. For an incoming order the two timestamps coincide, so nothing was wrong — but the scenario did not exercise the case the ruling is about. | Medium | C5 now draws **both**: an aggressor residual and a resting residual, each with an interloper. The spec states why rather than letting one stand for the other. |
| 6 | **C16 cited the wrong §2.2 clause.** §2.2's no-cross-matching sentence is about different *products*; C16 concerns 96 contracts of the same product, whose governing rule is §6.4. | Low | §2.2 now quoted only for "Each contract has its own independent order book"; no-cross-matching taken from §6.4. |
| 7 | **Coverage omissions.** §3.2's IDA timetable, §3.1's GOT rule and §2.2's simultaneous-opening schedule — all §11.1 hard rules — were named nowhere, nor were four §9.2 observable rows. C0/C11 certify GCT only; GOT is never exercised. | Low | Added to the out-of-scope list under the section's own standard that an unimplemented behaviour absent from it reads as certified. |

Minor granularity corrections also applied: §2.3 is a bullet list rather than a table; the iceberg
`0 < peak ≤ quantity` constraint is **derived** from §4.2 rather than stated by it, and is labelled a
validator invariant; C4's and C7's state-diagram quotes now take the whole `CANCELLED (…)` branch
rather than a partial parenthetical, which would not have matched verbatim; and C0 records that
**PENDING has no market-spec existence** — it is a simulator status, like the pad row.

**Confirmed sound and left alone:** the §5.3 "out of scope by decision, though implemented"
framing (the category is correct, the consequence is stated, and the replacement test is
pre-specified), and the three-way categorisation of the out-of-scope list.

---

## 11. Third pass (v3.0) — plan revision 2 against market-spec v2.12

**Scope.** Every Track A change in revision 2 of the plan (2026-09-28, updated 2026-10-02): the red
"What changed" rows 1–31 as they touch Track A, the Decisions table, the three-kinds-of-evidence and
invariant-battery sections, the capacity-validity precondition, C0–C17, "Out of scope, and
conformance gaps", and Notebook architecture. Each quote was checked against `market-spec.md` v2.12
under the archived checker's matching contract. That contract strips `**`, `__`, `*`, `_` and
backticks, collapses whitespace, and splits on an ellipsis
(`/tmp/omie-cert-rollback-111213/notebooks/certification/cert_support.py:87`, `:95-113`). It is
deliberately **not** Unicode- or punctuation-insensitive (`:98-99`). Claims were checked against the
source corpus where the spec cites it, and against code only where a claim is about the code.

**Verdict.** Most of the revision's Track A changes are domain-correct:

- C5 now depends on a merged fix and correctly expects a pass.
- C8's report-equals-tape claim is labelled an engineering invariant, which is right.
- C12 now hands mispricing to C2 and I5.
- The OQ-25 premise is corrected consistently with the OQ-25 CORRECTION block.
- The new engineering-limits list is accurate.

Three things need fixing before the cards run:

1. **Revision 2 is a rewrite, and it lost five fixes that v2.0 signed** (V3-01). One of them is the
   High-severity quote attribution v2.0 found first.
2. **Two of the revision's new or restated expectations are wrong by the spec.** C12's per-participant
   non-zero settlement fails on a legal self-match (V3-03). I5's "maker" is undefined at exactly the
   Rule-3 hop, where §4.2.1 and §6.2 give different readings (V3-02).
3. **Of the two spec inconsistencies the plan names, only one holds.** The §3.2 day labels are
   inconsistent, but the quarter-hour periods are correct (R-2). The §3.3 citation is quoted
   correctly but proves something else (R-1).

### 11.1 Confirmed sound in this revision (no action)

- **D4 / Decision 5.** `market-spec.md` v2.12 §5.2 carries the rule: *"Partial execution is NOT a
  modification and does NOT assign a new timestamp."* (`market-spec.md:325`). It covers both the
  incoming residual and the partially executed resting order, and it is sourced to Continuous Trading
  Matching Algorithm §4.3 and §5.1.3. D4 passes as a check, with no edit needed.
- **C3 / C17 / out-of-scope "Bounded" paragraph — OQ-25 premise.** This matches
  `open-questions.md:993-1006` (CORRECTION 2026-10-01):
  - (a) Before the fix, the agent sat **ahead** of the queue.
  - (d) The PROXIMATE (queue-position) attribution is "NOT supported" and flagged for re-measurement
    at the user's discretion.
  - (e) The ROOT attribution (OQ-10 H1) and the fix-locus ruling stand.

  Row 24's figures (passive buy 44 → 32, sell 53 → 26) are OQ-25 (c)'s (3b) row. The plan neither
  overstates nor understates this.
- **C9 conformance gap, factual half.** The validator checks only `peak_qty > 0` and
  `peak_qty <= qty` (`gymnax_exchange/omie/order_validator.py:96-97`), with no PPD sign test. §13.6
  makes the sign a hard rule and says *"Validate the sign rule (buy => negative, sell => positive) as a
  hard constraint."* (`market-spec.md:774`). The gap is real.
- **C14b.** "Every further trade lands in the last row" is OQ-15's documented mechanism
  (`open-questions.md:422`: on a saturated buffer every `add_trade` "writes to index `-1`").
- **C12, mispricing ownership.** Settlement is computed from the engine's own trade rows
  (`settlement.py:128-147`), so I6 cannot see a wrong price, and handing it to C2 and I5 is correct.
  The shape-hazard description matches `compute_agent_pnl`, which indexes `trades[:, 3]` and
  `trades[:, 4]` with no shape check (`settlement.py:120`, `:128-129`).
- **Capacity precondition.** The four margins and the "flag is not the guarantee" rule are sound. The
  hazards are grounded in OQ-10 (2026-06-11 level wipe), OQ-15 and §4.2.1 Rule 4.
- **Decision 6.** It correctly separates result-wrong defects (both fixed) from limits the ladder
  works around.

---

## 12. Rulings for the notebooks to quote

These rulings are recorded **here only**. Per the plan, `market-spec.md` is not amended by this
deliverable. A later Domain Analyst spec card may transcribe R-1 and R-2. Until then a notebook that
cites them quotes this document by ID and version (see V3-07 for the checker consequence).

### R-1 — the §3.3 phase-2 citation (requested re-check)

**Question.** Is market-spec §3.3's phase-2 source, *Continuous Trading Matching Algorithm §5.5: "As
soon as a batch matching round starts, both regular matching and explicit capacity allocation are
suspended."* (`market-spec.md:163`), the right ground for the IDA clearing-window suspension? v2.10
also cites §5.5 together with §6.3 for the batch match at reopen (`market-spec.md:170`).

**Ruling (2026-10-02, confidence high).**

1. **The quote is verbatim and correctly located, but it proves something else.**
   - The sentence is in Matching Algorithm §5.5, "Priority setting between implicit and explicit
     allocation". It immediately follows *"Batch matching is triggered by a capacity update."*
     (`Continuous_Trading_Matching_Algorithm_Public_Description.md:616-617`).
   - The same document calls these capacity-update rounds *"intra-day auctions"*: *"5.2. Batch
     matching (or intra-day auction) — Batch matching is triggered by an increase of the cross-border
     capacity"* (`:498-501`).
   - That is the XBID algorithm's own term for capacity-increase batch rounds. It is a false friend
     of the pan-European IDAs (IDA1–IDA3, from 13 June 2024).
   - So the citation **does** refer to capacity-increase batch matching, as the plan suspected. It
     does not establish the phase-2 suspension.
2. **What does ground phase 2:**
   - OMIE Rules §58.2, continuous-market sequence, item (e): *"bid submission on the continuous market
     shall be stopped from the closing of the bid submission period for each intraday session and
     until the end of the intraday auction"*.
   - Item (f) of the same sequence: *"the continuous market contracts that were suspended from trading
     because of the intraday auction session shall be re-opened for trading"*
     (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:8712-8719`).
   - OMIE Instruction 1/2024 §8, which spec §3.3 already cites.
   - Matching is frozen as a consequence. Regular matching is triggered only by an order with a new
     timestamp (Matching Algorithm §5.1), and none can enter while submission is stopped. So the
     spec's bullet *"Both regular matching and explicit allocation are fully suspended."* is
     **correct as a rule**, but its cited source is wrong.
3. **The v2.10 reopen citation (§5.5 with §6.3) is a defensible inference, not a direct statement.**
   - While submission is suspended, no local order can cross the book. A crossed book at reopen can
     therefore arise only when the IDA results come back as a cross-zonal capacity update, and
     capacity-update batch matching (§5.2) is exactly the procedure for that case.
   - The 22:20:00 trades with a blank, non-Iberian side in OQ-28 fit this reading.
   - The citation is apt for the reopen. It should be read at OQ-28's stated confidence
     (medium-high), and spec §6.3's own line *"Not relevant for single-zone"* (`market-spec.md:367`)
     should be read with that exception in mind.
4. **What C0 should quote.** C0 should quote the phase **description** from §3.3, byte-copied:
   - `Phase 2 — IDA clearing window (GCT to results, ~20 min):`
   - `Both regular matching and explicit allocation are fully suspended.`
   - `Order submission is also suspended for affected contracts.`

   It should **not** quote the Matching Algorithm sentence as authority for the suspension. If the
   notebook shows the citation at all, it should add: *"Domain review v3.0 R-1: this source sentence
   concerns capacity-update batch matching; the IDA-window suspension rests on OMIE Rules §58.2 (e)–(f)
   and Instruction 1/2024 §8."*

### R-2 — the §3.2 IDA table (requested confirm/refute)

**Ruling (2026-10-02, confidence high): confirmed in part, refuted in part.**

1. **Day labels — CONFIRMED inconsistent.**
   - §3.2 times IDA1 and IDA2 at "(D−1)" but scopes them "of D+1", two days apart. It times IDA3 at
     "(D)" and also scopes it "of D+1" (`market-spec.md:138-140`).
   - The prose below the table says IDA3 serves the delivery day itself (`:142`), which contradicts
     the "D+1" label.
   - In a single frame where D is the delivery day: IDA1 and IDA2 run on D−1, IDA3 runs on D, and all
     three cover D (IDA3 the second half only).
   - Instruction 1/2024 agrees: all three sessions are "for delivery day 14/06/2024", with IDA1 and
     IDA2 operated on 13/06 and IDA3 on 14/06 (`OMIE_Instruction_1_2024_IDA_Go-Live_Operations.md:115-169`).
   - Likely origin: OQ-04's table uses the session-day frame consistently ("GOT (day D)", IDA3 "09:00
     (D+1)", scope "of D+1"; `open-questions.md:102-106`), and so does the Rules' Annex 2 Figure 1. §3.2
     relabelled the run times to the delivery-day frame and kept the session-day scope labels.
2. **Quarter-hour periods — REFUTED as an error.**
   - The current Rules, Annex 2 §1 "Intraday auction market products and schedules", Figure 1, give
     *"TRADING TIMEFRAME (Quarter-hour periods)"*: *"24 hours (1-96 D+1)"* for the first and second
     sessions and *"12 hours (49-96) D"* for the third. The prose reads *"distributed in 96
     consecutive quarter-hourly trading periods for the first two sessions and 48 consecutive
     quarter-hourly trading periods for the third session"*
     (`OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md:9747-9771`).
   - Figure 2 gives 1–92 and 45–92 for the 23-hour day (`:9795-9807`).
   - Instruction 1/2024's hourly periods ("periods 1 to 24", "13 to 24") describe the June-2024
     go-live, before the 15-minute MTU (spec §2.1: active since 18 March 2025).
   - §3.2's source line cites both documents. For the current regime the quarter-hour periods are
     right.
3. **Consequence.** The plan's out-of-scope bullet (plan line 1362-1364) states the opposite of what
   the Rules say on the periods. The replacement text is in V3-11. Track B's L5 uses the IDA timetable
   and should be checked against R-2 by its owner. That falls outside this Track A review.

---

## 13. Findings (v3.0)

Severity scale: **High** means a notebook would hard-fail on a correct engine, or pass on a wrong
one, or the certificate would assert something the spec does not say. **Medium** means a wrong or
ungrounded citation, label or claim that misleads a reader or an auditor without flipping a verdict
by itself. **Low** means wording, dating or a coverage note.

### V3-01 — High — C2, C5, C10, C11, C16, C1/C7 — the rewrite lost five signed v2.0 fixes

**Finding.** v2.0 recorded these as fixed (section 10, items 1, 3, 4, 5, 6 and the minor corrections).
Revision 2 of the plan contains the **pre-fix** text again, and so does the current
`docs/omie/scenario-ladder.md`.

**Evidence and replacement text (plan line → required text):**

1. **C10 Quotes** (plan:1020; spec doc `scenario-ladder.md:694`). The plan has
   §4.2 *"FOK orders see the iceberg's full remaining quantity for matching purposes"*. That sentence
   is under §4.2.1 Rule 1 (`market-spec.md:225`). §4.2 says the opposite: icebergs *"cannot combine
   with IOC or FOK behavior"* (`:206`). With section-label validation (HARNESS-1) this hard-fails K5.
   → Replace with: *"§4.2.1 Rule 1 (Real market rule) "FOK orders see the iceberg's full remaining
   quantity for matching purposes"; §4.2.1 Rule 1 in full with its ICEwPI worked example. (Not §4.2:
   §4.2 says icebergs "cannot combine with IOC or FOK behavior".)"*
2. **C2 Quotes** (plan:731; `scenario-ladder.md:405`). The plan has *"§11.1's hard-rule row, which
   carries the ACER/CACM provenance the certificate wants"*. The execution-price row is sourced to
   *"Continuous Trading Matching Algorithm §5.1.1"* (`market-spec.md:521`). ACER/CACM source the
   price-time row (`:520`).
   → Replace with: *"§11.1's row "Execution price = resting order's limit price in regular matching",
   sourced to Continuous Trading Matching Algorithm §5.1.1 (the ACER/CACM provenance belongs to the
   price-time-priority row, which C3 cites)"*.
3. **C11 Purpose and Quotes** (plan:1036-1037, :1049). The plan has *"§8.5 is the precise ground for
   the second claim"* and *"§8.5 in full, as the ground for 'no new trades at expiry'"*. §8.5 says
   nothing on whether a sweep may print trades (`market-spec.md:458-459`).
   → Replace the Purpose sentence with: *"The ground for the second claim is §3.1 ("Orders on a
   contract are automatically cancelled at GCT if unmatched") together with §6.2 (regular matching is
   triggered by a new-timestamp order, and an expiry sweep is not one); §8.5 is quoted for trade
   firmness only."* Replace the Quotes cell with: *"§3.1's gate rule; §6.2's heading "Regular matching
   (triggered by new-timestamp order)"; §8.5's two sentences, for firmness."*
4. **C5 Dataset** (plan:857). The plan has *"Random $q_{\text{agg}} >$ total crossing depth, plus an
   interloper…"*, which is the aggressor case only. v2.9 §5.2 covers both cases explicitly
   (`market-spec.md:325`: *"Likewise a resting order that is partially executed keeps its
   timestamp."*).
   → Replace with: *"Two draws, each with an interloper at the residual's price: (i) aggressor
   residual — random $q_{\text{agg}} >$ total crossing depth, the residual rests under §6.2 item 4;
   (ii) resting residual — a resting NON order partially filled by a smaller aggressor. Both are
   needed because §5.2 states the rule for each separately."* Add to Known result: *"in (ii), the
   partially executed resting order keeps its timestamp and fills ahead of the interloper"*.
5. **C16 Quotes** (plan:1263). The plan has *"§2.2's independent-order-book sentence and its
   no-cross-matching clause"*. §2.2's clause is *"Contracts from different products are not
   cross-matched"* (`market-spec.md:92`), which is about products. C16's rule is §6.4 (`:375`).
   → Replace with: *"§2.2 "Each contract has its own independent order book"; §6.4 "Orders on
   different contracts are never matched against each other"; §2.1's 96-contract sentence"*.
6. **Minor regressions.**
   - C1 Certifies (plan:706) should read *"§4.2 iceberg peak constraint `0 < peak ≤ quantity` — derived
     from §4.2, not stated by it; a validator invariant"*.
   - C7 Certifies (plan:905) and C11 Certifies (plan:1048) name partial parentheticals,
     `CANCELLED (IOC/FOK unmatched)` and `CANCELLED (… GCT reached)`. Neither matches source, because
     the source branch is *"CANCELLED (IOC/FOK unmatched; GTD expired; GCT reached; halt/hibernation)"*
     (`market-spec.md:312`) and `GCT reached)` is not a substring. Quote the whole branch.
   - C0 should restore *"PENDING has no market-spec existence — it is a simulator status"* (see also
     V3-12).

**Disposition.** Plan change, with the texts above. Then grep `scenario-ladder.md` (D2) for the same
five strings and correct it in the same edit. **Process note:** any future rewrite of the plan should
re-run the v1.0/v2.0 disposition tables as a checklist.

### V3-02 — High — I5 and C8 Rule-3 draw — "maker" is undefined at the Rule-3 hop, and the spec reads two ways there

**Finding.** I5 requires *"Every trade price equals **that maker's own** limit price"* (plan:450). In
the Rule-3 hop, the aggressor's remainder rests first, and the iceberg's **regenerated** slice (the
newer, new-timestamp entry) crosses it. The spec gives two readings of who sets the price:

- **Reading A.** §4.2.1 Rule 3: *"The resting remainder acts as passive liquidity for the new slice"*,
  matched *"consistent with how any new order (including a regenerated slice) triggers regular
  matching"* (`market-spec.md:237`). Under §6.2 item 3 / Matching Algorithm §5.1.1 the price would be
  the **remainder's** limit (the aggressor's limit $L$).
- **Reading B.** §6.2 item 7: *"the iceberg is always the price-setter"* (`:363`). §5.2: *"the iceberg
  is treated as the passive (resting) side throughout and its timestamp does not supersede the
  incoming order for price determination purposes"* (`:326`). Matching Algorithm §5.1.2: the iceberg
  sets the price *"even if the timestamp of the iceberg order is renewed during the matching
  process"* (`Continuous_Trading_Matching_Algorithm_Public_Description.md:475-479`). All three give
  the **iceberg slice's** price.

The two differ whenever $L$ exceeds the slice price (buy aggressor). Whichever reading the record
builder assumes, a C8 Rule-3 draw either hard-fails a correct engine or passes a wrong one. I5 is the
check that "owns mispricing", so it cannot be silent here.

**Provisional reading (not a ruling; to be confirmed through an OQ). Reading B, medium-high
confidence.**

- In the real algorithm the Rule-3 configuration does not arise as a separate event. The incoming
  order keeps matching successive slices inside one matching process (Rule 2; Matching Algorithm
  §5.1.2), and §5.1.2 fixes the price to the iceberg.
- Rule 3's "passive liquidity" phrase speaks to matchability, not price.

**Disposition.**

1. The orchestrator routes this to the `domain-analyst` as a new OQ, proposed as OQ-31 "Price-setter
   in the §4.2.1 Rule-3 hop". It must be resolved before C8's K2.
2. Plan change to the I5 row: replace *"Every trade price equals **that maker's own** limit price"*
   with *"Every trade price equals the limit price of **that trade's price-setter** — the resting order
   under §6.2 item 3, or the iceberg under §6.2 item 7 and §5.2 (including the §4.2.1 Rule-3 hop, per
   OQ-31)"*. Keep the ICEwPI ladder exemption.
3. Add to C8's Rule-3 draw Known result: *"Rule-3 trade price per OQ-31"*.
4. If OQ-31 adopts Reading B, the §4.2.1 Rule 3 wording is a fourth spec clarification. It is
   recorded only and would be applied by a spec card.

### V3-03 — High — C12 — the new per-participant non-zero settlement assertion is wrong by the spec

**Finding.** C12 now asserts *"every participant that traded has a **non-zero** settlement (I9's
non-vacuity, applied per participant)"* (plan:1090-1091). This contradicts the rules in two legal
cases:

- **Self-match.** §6.6 makes self-matching legal (`market-spec.md:385`). The settlement function nets
  a row where buyer == seller == agent to zero: *"A row where buyer == seller == agent_id collapses to
  zero (legal self-match)"* (`settlement.py:139-143`). So does §8.4's cash leg.
- **Round-trip at equal price.** A participant who buys and sells the same quantity at the same price
  has exactly zero cash, by §8.4's net-PnL formula (`market-spec.md:454`).

A random multi-participant tape can produce either case, so a correct engine would be logged
`NOT CERTIFIED`. The hazard the assertion targets is real: `(steps, 2, 6)` input is silently wrong.
But this guard is the wrong one.

**Disposition.** Plan change. Replace the last sentence of C12's "known hazard" paragraph with:
*"C12 therefore always flattens the trades to (M, 6) before settling, as the production callers do,
asserts the shape it passes, and guards against a vacuous zero-sum in two ways: the gross cash leg
$\sum |V|$ over all trades must be positive (I9), and each participant's reported settlement must
equal an independent per-row recomputation of $\sum (\pm Q\cdot p\cdot d)$ for that participant — which
a mis-shaped call fails even when its output is 0."*

### V3-04 — Medium — I6(a) and C12 — "exact integer zero-sum" is either tautological or not observable

**Finding.** I6(a) and C12 assert the cash leg sums to zero *"exact in integer ticks"* (plan:451,
:1066-1067, :1100). The engine's settlement output is a float EUR value (`settlement.py:123-124`:
*"0-d `jnp.float32` PnL in EUR"*). The int64 accumulators are internal (`:145-147`), so no integer
settlement can be read from the engine.

If the integer sum is computed from trade rows, it is zero by algebra: each row is $+V$ to its seller
and $-V$ to its buyer. That is the *"cash check comparing a ledger against a copy of itself"* that
rule 1 (plan:396-397) forbids. The non-trivial content is narrower:

1. every trade row's buyer and seller ids belong to the enumerated participant set, or a sum over
   participants drops legs;
2. the engine's reported per-participant settlements sum to zero within the I6(b) decode tolerance.

**Disposition.** Plan change to the I6 row:
*"**(a)** The engine's per-participant settlements, summed over every participant id that appears in
a trade row, equal zero within the tolerance derived for (b); every trade row's buyer and seller are
enumerated participants. (The integer per-row identity holds by construction and is not a check.)"*
In C12's Pass row, replace *"Exact integer zero-sum on the cash leg"* with *"(a) and (b) as above;
per-trade $Q\cdot p\cdot d$ recomputed exactly in ticks against each row"*. The broken-input fixture for
I6(a) must break the **settlement output** (for example, drop one participant's leg). Breaking a row
does not test it.

### V3-05 — Medium — C9 — the wrong-sign draw pins behaviour on a domain-forbidden input, and "Certifies §13.6" overclaims

**Finding.**

1. C9's Known result lists *"the wrong-sign draw is **accepted** (recorded gap)"* (plan:985) as an
   expectation. The OQ the plan cites forbids exactly this. OQ-23's CORRECTION + CLOSURE NOTE (a):
   *"A locked contract must never pin behavior on domain-forbidden inputs."*
   (`open-questions.md:838`). If the validator is ever fixed, C9 flips to `NOT CERTIFIED` on a
   **correct** engine.
2. C9's Certifies row lists *"§13.6"* (plan:982). C9 does not certify §13.6. It records its violation,
   since §13.6 requires sign validation *"as a hard constraint"* (`market-spec.md:774`). The
   out-of-scope list also still says *"§13.6's PPD sign rule, asserted in C9"* (plan:1411-1412),
   which contradicts the conformance-gap bullet (plan:1379-1380).
3. The citation *"OQ-23 (a)"* is ambiguous. OQ-23's main body has no item (a). The text is item (a)
   of the CORRECTION + CLOSURE NOTE appended on 2026-07-09 (`open-questions.md:834-838`).

**Disposition.** Plan change.

- C9 Known result: replace the last clause with *"the labelled wrong-sign draw is an **observation**,
  reported and not asserted: its accept/reject outcome is logged as evidence for the §13.6 gap and is
  excluded from the pass criterion"*.
- C9 Certifies: replace *"§13.6"* with *"§13.6's magnitude statement (no rule-level bound) as the
  ground for the draw envelope; §13.6's sign-validation requirement is **recorded as a conformance
  gap**, not certified"*.
- Out-of-scope §13.x bullet: replace *"asserted in C9"* with *"drawn sign-valid in C9, whose
  wrong-sign observation is the evidence for the conformance gap above"*.
- Cite as *"OQ-23, CORRECTION + CLOSURE NOTE (a)"* in C9 and in the gap list.

### V3-06 — Medium — C9, C12, C0 and every notebook — Unicode quotability hazard

**Finding.** The checker does not normalise Unicode (`cert_support.py:98-101`), so a quote must match
the source's code points exactly. The spec mixes look-alike characters within single passages:

- §13.6 writes the counter-example with an **ASCII hyphen**: *"`PPD = -3.00 EUR/MWh` observed on
  ICEwPI BUY rows"* (`market-spec.md:768`).
- §13.3 writes it with U+2212: *"`PPD = −3.00 EUR/MWh`"* (`:692`).
- The plan writes "−3.00" (U+2212) at plan:970.

A C9 notebook quoting §13.6 the way the plan words it fails K5 on a correct citation. Other passages
the plan directs notebooks to quote carry U+2212, U+00D7, U+00B7, U+230A/U+230B and U+2014:
§8.4 `−V`, §11.1 "Q × p × d", §3.1 "GCT(k) = 60·⌊k/4⌋ − 60 min", §3.3 "Phase 2 — …".

**Disposition.**

- Plan change to `OMIE-CERT-HARNESS-1` acceptance criteria. Add: *"the quote checker's tests include
  quotes containing U+2212, U+00D7, U+00B7 and U+2014, passing when byte-copied and failing on an
  ASCII look-alike; notebooks byte-copy quoted text from source"*.
- Do **not** add Unicode folding. The checker is punctuation-strict by design, and folding would let
  a sign error pass.
- C9 should quote §13.6's sentence with its ASCII `-3.00`.

### V3-07 — Medium — C8, C9, C14, C17, C0 (R-1) — quotes from documents the checker cannot read

**Finding.** The plan directs several notebooks to quote verbatim from documents other than
`market-spec.md`:

- C17 quotes OQ-10 and OQ-25 (plan:1341).
- C14 is grounded in OQ-10, OQ-15 and OQ-20 (plan:1169).
- C8 and C9 rely on OQ-20 and OQ-23.
- C0 would cite R-1, which itself quotes OMIE Rules §58.2.

The archived checker reads one file (`cert_support.py:31`:
`SPEC_PATH = REPO_ROOT / "docs" / "omie" / "market-spec.md"`) and takes the version from its
`**Status:**` line (`:116-120`). `open-questions.md` has no version stamp (`open-questions.md:1`). As
specified, these quotes are either unverifiable or fail as "not found". Notebook architecture item 2
promises *"the spec version quoted from"* for every quote.

**Disposition.** Plan change to `OMIE-CERT-HARNESS-1` acceptance criteria and Notebook architecture
item 2. Add: *"a `SpecQuote` names its source document — `market-spec.md` (versioned by its Status
line), `open-questions.md` (versioned by the OQ id plus the dated block quoted, e.g. "OQ-25
CORRECTION 2026-10-01"), or `scenario-ladder-review.md` (versioned by its Status line) — and is
checked against that file; regulatory-corpus text is shown as a citation and is not machine-checked"*.

### V3-08 — Medium — C8 — "§5.2 bullet 3" now points at the partial-execution bullet

**Finding.** v2.9 inserted the partial-execution bullet into §5.2. The iceberg passive-side exemption
is now **bullet 4** (`market-spec.md:326`), and bullet 3 is the partial-execution rule (`:325`). C8's
Certifies and Quotes rows say *"§5.2 bullet 3"* (plan:944-945; `scenario-ladder.md:618-619`), with the
exemption text.

The text match still resolves. Row 25's *"No section was renumbered, so every quote still resolves"*
is true of section numbers and text, but not of bullet positions. A human reader of C8, or a checker
that validates bullet positions, is pointed at the wrong rule.

**Disposition.** Plan change. In C8's Certifies and Quotes, replace *"§5.2 bullet 3"* with *"§5.2, the
iceberg-slice bullet ("Iceberg slices receive a new timestamp per slice, except …")"*. Add to
Notebook architecture item 2: *"labels are section-level (§5.2, §6.2 item N for numbered lists);
bullet positions are never used as labels"*. Row 25 should read *"No section was renumbered; §5.2
gained a bullet in v2.9, so bullet positions shifted (C8 corrected)"*.

### V3-09 — Medium — C8 — the incoming-iceberg draw is undefined, and its single-timestamp claim is ungrounded

**Finding.**

1. C8's Known result ends *"reported quantity = Σ slice fills = $\mathit{total}$ (incoming-iceberg
   draw)"* (plan:947). The Dataset row defines only a resting iceberg consumed by one aggressor, plus
   a Rule-3 draw (plan:946). The incoming-iceberg draw, which is the case the fill-report fix
   targets, is never specified.
2. The same Known result claims *"all trades carry the **originating** matching process's
   timestamp"*. That claim is grounded for a **resting** iceberg consumed by one incoming order:
   §4.2.1 Rule 4 (`market-spec.md:243`) and the §5.2 iceberg exemption (`:326`).
3. For an **incoming** iceberg the spec says each new slice is activated and *"receives a new
   timestamp"* (§4.2, `:205`; §5.2, `:324`). Each slice is then a new regular-matching trigger
   (Matching Algorithm §5.1). The exemption covers only the iceberg as the passive side. No rule says
   an incoming iceberg's later-slice trades carry the first slice's timestamp, and Matching Algorithm
   §5.3 ties a trade's timestamp to the moment an order is found executable (`:589-590`).

**Disposition.** Plan change.

- C8 Dataset: append *"Separate incoming-iceberg draw: an iceberg `(total, peak)` submitted against a
  resting ladder deep enough to fill $\lceil \mathit{total}/\mathit{peak}\rceil$ of its own slices."*
- C8 Known result: restrict the timestamp clause to *"(resting-iceberg and Rule-3 draws)"*.
- For the incoming-iceberg draw, assert only quantity, I5 price, report-equals-tape (I6(b)) and I3,
  plus *"no single-timestamp claim: each of its slices is a new-timestamp entry (§4.2, §5.2)"*.

### V3-10 — Medium — C0 — the §3.3 phase-2 citation (ruling R-1)

**Finding.** The plan's caution (plan:673) is justified. The ruling is in section 12, R-1: the
Matching Algorithm §5.5 sentence concerns capacity-update batch matching, and phase 2 rests on OMIE
Rules §58.2 (e)–(f) and Instruction 1/2024 §8.

**Disposition.** Plan change to the C0 Quotes row. Replace the sentence beginning *"**Caution:**
§3.3's phase-2 citation …"* through *"… the re-check is still owed."* with:
*"§3.3's phase-2 description — "Both regular matching and explicit allocation are fully suspended." and
"Order submission is also suspended for affected contracts." — quoted without its Matching Algorithm
§5.5 source sentence, which concerns capacity-update batch matching (domain review v3.0 R-1); the
suspension rests on OMIE Rules §58.2 (e)–(f) and Instruction 1/2024 §8. The v2.10 reopen citation
(§5.5 with §6.3) is apt for the reopen batch match (R-1 point 3)."* In the out-of-scope sub-bullet
(plan:1365), replace *"may point at the wrong Matching Algorithm clause (see C0)"* with *"cites a
Matching Algorithm clause about capacity-update batch matching for the IDA-window suspension (domain
review v3.0 R-1; correct rule, wrong source)"*. Add under "Out of scope because unimplemented": *"§3.3
(v2.10) survival of resting orders through the IDA clearing window and the reopen batch match (OQ-28)
— no engine path; the replay carries the book across the gap as a recorded simplification"*.

### V3-11 — Medium — out-of-scope list, L5 — the §3.2 IDA table (ruling R-2)

**Finding.** The plan's bullet (plan:1362-1364) says §3.2's periods are wrong because Instruction
1/2024 uses hourly periods. R-2 refutes that half: the current Rules, Annex 2 Figure 1, give
quarter-hour periods 1–96 and 49–96. Only the day labels are inconsistent. As written, the
certificate would record a spec error that does not exist and omit the one that does.

**Disposition.** Plan change. Replace the bullet with:
*"**§3.2's IDA table** times IDA1/IDA2 "(D−1)" and IDA3 "(D)" but scopes all three "of D+1", mixing
two day frames: IDA1 and IDA2 run on D−1 and IDA3 on D, all for delivery day D (OMIE Instruction 1/2024
§6, §8, §9; OMIE Rules Annex 2 §1 Figure 1, whose "D+1" is relative to the session day). The
quarter-hour periods (1–96; 49–96 for IDA3) are correct under the current Rules (Annex 2 §1 Figure 1);
Instruction 1/2024's hourly periods (1–24, 13–24) predate the 15-minute MTU (domain review v3.0 R-2)."*
Red-table row 18 should read *"the §3.2 IDA table (inconsistent day labels; its quarter-hour periods
are correct, per R-2)"*. Track B's L5 IDA arm should be checked against R-2 by its owner.

### V3-12 — Medium — C0 and the conformance-gap list — the SUSPENDED collapse is miscategorised

**Finding.**

1. For **submission**, the `!= OPEN` rejection of a SUSPENDED contract **is** what §3.3 phase 2
   requires: *"Order submission is also suspended for affected contracts."* (`market-spec.md:164`).
   The simulator documents SUSPENDED the same way: *"SUSPENDED: IDA clearing window — no submission or
   matching (§3.3)."* (`contract.py:20`).
2. What the collapse cannot express is what v2.10 added to §3.3. Resting orders **survive** the window
   and are batch-matched at reopen (`market-spec.md:169-170`), whereas a CLOSED contract's orders are
   cancelled (§3.1). That is an **unimplemented path**, already listed (plan:1358-1360).
3. Listing the collapse under *"Implemented, but not as the rule says"* (plan:1381) therefore
   overstates the gap.
4. §9.2's "Contract status" row is **OPEN / CLOSED (past GCT)** only (`market-spec.md:485`).
   Suspension is a separate observable, *"IDA suspension phase | NORMAL / CAPACITY_HALT /
   FULL_SUSPENSION"* (`:487`). So SUSPENDED, like PENDING, has no market-spec existence as a
   **contract status**. C0's sentence *"§3.3 gives SUSPENDED a distinct meaning"* (plan:663-664) puts
   the simulator's enum into the spec.

**Disposition.** Plan change.

- C0 second paragraph: replace *"but §3.3 gives SUSPENDED a *distinct* meaning (submission **and**
  matching frozen during the IDA clearing window) that the collapse does not express"* with *"which
  is conformant for submission — §3.3 suspends order submission in the IDA clearing window — but cannot
  express the rest of §3.3's phase 2: resting orders survive the window and are batch-matched at
  reopen (v2.10, OQ-28), whereas at CLOSED they are cancelled. PENDING and SUSPENDED are simulator
  statuses; §9.2's contract status is OPEN / CLOSED, and suspension is its separate "IDA suspension
  phase" variable"*.
- Remove the bullet *"§3.3 **SUSPENDED** collapsed into a single `!= OPEN` rejection (C0)"* from
  "Implemented, but not as the rule says". Its content is covered by the unimplemented-path bullet
  and by V3-10's OQ-28 addition.

### V3-13 — Medium — C0, C11 — the v2.10 hour-aligned GCT is quoted but exercised by no scenario

**Finding.** C0 now quotes §3.1's v2.10 correction (plan:673), and C0 and C11 both list §3.1 GCT under
**Certifies** (plan:672, :1048). But the engine takes GCT as a per-contract **input**: the environment
builds `gct_minutes=gct_minutes_from_got`, with delivery start `gct_minutes_from_got + 60`
(`omie_env.py:1850-1861`).

C0's dataset is status-driven, and C11 sweeps at a supplied gate. Neither derives a contract's GCT from
its slot. The §3.1 schedule *"GCT(k) = 60·⌊k/4⌋ − 60 min"* (`market-spec.md:126`) is therefore not
certified by Track A. OQ-27 also records an unverified reconciliation owed for `OMIE-SESSION-CLOCK-1`
(*"460 + 60·⌊k/4⌋"*, `open-questions.md:1060`). This review did not check whether it landed.

**Disposition.** Plan change.

- C0 Certifies: replace *"§3.1 gate times"* with *"§3.1's gate rule for a contract's **given** GCT
  (the status gate); the v2.10 hour-aligned GCT(k) schedule is quoted as context, not certified"*.
- C11: same qualifier on *"§3.1 GCT"*.
- Out-of-scope, unimplemented-or-elsewhere: add *"§3.1 (v2.10) hour-aligned GCT(k) schedule — a
  loader/environment property (`gct_minutes_from_got`, OQ-27), not an engine rule; no Track A
  scenario derives GCT from the slot"*.
- Suggested exact quote for C0: `In the Iberian continuous market, GCT is aligned to the delivery
  hour, not to each quarter-hour.`

### V3-14 — Medium — C14a′ — certifying an unflagged eviction contradicts C14's own standard

**Finding.** C14's Purpose says *"overflowing it silently and losing orders would be a defect"*
(plan:1148-1149), and its Failure-implicates row says *"A silent capacity overflow — strictly worse
than the documented degradation"* (plan:1171). C14a′ is a silent capacity overflow by its own
description: *"a regenerated slice that rests **does not raise** the full flag, so this eviction is
unflagged"* (plan:1164). Under C14's standard it would be both "certified" and a failure. The closing
claim *"its behaviour at the envelope's edge is known and flagged"* (plan:1154-1155) is false for this
part.

C14a′'s limit, C14b's last-row collapse beyond OQ-15, and C14c's unreadable iteration count are also
grounded only as *"recorded when the iceberg fill-report defect was fixed"* (plan:1384). That record is
not cited in C14's Certifies row.

**Disposition.** Plan change.

- C14a′ Known result: append *"**Recorded, not certified**: the part observes and reports the
  unflagged eviction (quantity read from the book diff, rule 2) as a known engineering gap outside the
  certified surface; it passes if the eviction is exactly the documented one and fails if any other
  quantity is lost"*.
- C14 Purpose last paragraph: change *"known and flagged"* to *"known, and flagged except where C14a′
  records an unflagged case"*.
- C14 Certifies row: add *"the `OMIE-ICE-INCOMING-FILL-REPORT-1` close-out record (2026-09-28,
  `1f4755a`) for C14a′, C14b and C14c's observability limits"*, with its exact `sprint-status.md`
  heading filled in by the orchestrator.

### V3-15 — Medium — C14 — battery exclusions are incomplete

**Finding.** C14 runs *"the battery (excluding I8, which this scenario deliberately violates)"*
(plan:1170). Some other invariants also break by construction at C14's ceilings:

- **C14b.** Under OQ-15 a saturated ring clobbers rows (`open-questions.md:422`). That breaks I1 and
  I6, which are read from trade rows, and any I2 term derived from rows. The capacity precondition
  says exactly this (plan:558).
- **C14c.** A truncated cascade can leave a regenerated slice resting in a crossed book. That breaks
  I3 and is a §4.2.1 Rule 4 departure by construction (`market-spec.md:243`). The plan asserts only
  that *"I2 still closes"*.

Running the full battery would log `NOT CERTIFIED` on documented behaviour.

**Disposition.** Plan change. Replace the Oracle-kind cell with: *"exact-parametric on documented
behaviour, plus the battery except: I8 (all parts); I1, I6 and row-derived I2 terms on C14b (OQ-15);
I3 on C14c (truncation is a Rule-4 departure by construction). Each excluded invariant is reported as
**expected-violated**, and its violation is itself asserted, so the exclusion cannot hide an
unrelated failure."*

### V3-16 — Medium — C11 — no iceberg draw, so GCT expiry of hidden quantity is uncertified

**Finding.** C11's dataset is *"a mix of validities, all resting at the gate"* (plan:1050), with no
iceberg. GCT cancellation applies to the whole order, *"CANCELLED (… GCT reached …)"* (§5.1,
`market-spec.md:312`) and *"Orders on a contract are automatically cancelled at GCT if unmatched"*
(§3.1, `:125`). That includes an iceberg's hidden remainder. I2 has a separate hidden bucket read from
the side channel (plan:447). After the sweep, a stale side-channel slot would book the remainder as
"hidden" rather than "expired". I2 would still close, but with the wrong semantics.

**Disposition.** Plan change.

- C11 Dataset: append *"including resting icebergs (standard and ICEwPI) with positive hidden
  remainder"*.
- C11 Known result: append *"each iceberg's hidden remainder is classified expired by order semantics
  (I2), and no trade, book entry or FOK-visible volume remains for the closed contract"*.
- Keep the C8 caution: assert no `iceberg_meta` slot state (OQ-20; `market-spec.md:252`).

### V3-17 — Low — I2 — "eviction" is listed as a cancellation cause

**Finding.** I2 derives `cancelled` from *"order semantics (IOC remainder, FOK kill, cancellation,
gate expiry, eviction)"* (plan:447). Eviction is not a §5.1 cancellation cause (`market-spec.md:312`,
`:317`). It is an implementation-limit fate, and C14a already accounts evicted quantity *"in its own
I2 bucket"* (plan:1161).

**Disposition.** Plan change. I2 becomes `filled + still_resting + cancelled_or_expired + evicted +
hidden_remaining == accepted`, with `cancelled_or_expired` limited to the §5.1 causes, and `evicted`
required to be 0 outside C14 (I8).

### V3-18 — Low — I4 Cannot-see and rule 3 — the hidden order identities are attributed to anonymity

**Finding.** Rule 3 says the simulator *"hides order identities by design (the anonymity rule C13
certifies)"* (plan:406), and I4's Cannot-see cell starts *"Order identities are hidden"* (plan:449).
But §7 governs what **participants** observe, and §7.3 says *"Maintain agent identity internally"*
(`market-spec.md:409`). The harness reads engine state, and C3, C4 and C5 identify filled orders by
id. The real limit is the trade-row schema, which carries no maker order id:
`[timestamp, price, quantity_tick, buyer_agent_id, seller_agent_id, contract_id]`
(`settlement.py:46-48`). So two same-price makers produce identical trade rows.

**Disposition.** Plan change. In rule 3, replace *"it hides order identities by design (the anonymity
rule C13 certifies)"* with *"its trade rows carry no maker order id"*. In I4, replace *"Order identities
are hidden, so"* with *"Trade rows carry no maker order id, so at equal price"*.

### V3-19 — Low — C5 — stale purpose text and a non-spec status name

**Finding.**

- *"The specification does not state directly whether such a residual keeps or renews its time
  priority"* and *"The `domain-analyst` records this in `market-spec.md` §5.2 before the expectation is
  authored"* (plan:822-826) were true before v2.9. They are now stale. Since v2.9 the rule is also
  grounded in source text, not only in the user ruling (Matching Algorithm §4.3, §5.1.3;
  `market-spec.md:325`), and the notebook can say so.
- Known result says *"status `PARTIALLY_FILLED`"* (plan:858). That is the simulator's enum
  (`order_types.py:47`). The §5.1 diagram says `PARTIALLY_EXECUTED`.

**Disposition.** Plan change. Replace those two sentences with: *"The user ruled on 2026-09-23 that the
residual keeps its time priority; `market-spec.md` §5.2 states it since v2.9, sourced to Continuous
Trading Matching Algorithm §4.3 and §5.1.3: "Partial execution is NOT a modification and does NOT
assign a new timestamp.""* In Known result, write *"simulator status `PARTIALLY_FILLED` (§5.1's
`PARTIALLY_EXECUTED + resting in BOOK`)"*.

### V3-20 — Low — C17 — wrong correction date, and two confounds not named

**Finding.**

- C17 Quotes says OQ-25's *"2026-09-29 correction"* (plan:1341). The block is dated **2026-10-01**:
  *"CORRECTION (2026-10-01, OMIE-AGENT-TIME-PRIORITY-1)"* (`open-questions.md:993`).
- C17's synthetic sessions have no IDA gap, so they cannot measure the OQ-10 EXTENSION (post-IDA
  re-quote double-counting, `open-questions.md:259-277`). That item is currently OPEN and is part of
  the same phantom-liquidity object.
- If the reference agent is driven through the environment, OQ-25 CORRECTION (b) records a small
  post-fix priority residual, *"Residual R1 … later non-agent arrivals that rank ahead of the resting
  agent at its own price"* (`open-questions.md:999-1000`). It contaminates the passive fill-rate delta
  slightly.

**Disposition.** Plan change. Correct the date to 2026-10-01. Add to "What it cannot claim": *"the
OQ-10 EXTENSION (post-IDA re-quote double-counting) — the synthetic sessions contain no IDA gap"*.
Add to the passive fill-rate bullet: *"if driven through the environment, report OQ-25 CORRECTION (b)
residual R1 alongside the delta"*.

### V3-21 — Low — residual-risk paragraph — a third spec inconsistency, recorded only

**Finding.** The plan keeps v2.0's residual-risk wording: §4.2.1's iceberg rules were *"adopted after
noting they deviate from the literal SIDC/XBID description"* (plan:376-379). `market-spec.md` is
internally inconsistent on this point:

- §4.2.1's preamble says *"The following three rules … deviate from the real SIDC/XBID matching
  algorithm"* (`market-spec.md:214`). There are **four** rules.
- Each rule's own Classification reads *"Consistent with market rules"* (`:227`, `:234`, `:240`,
  `:253`).

The plan's conclusion still holds, but its premise rests on the stale preamble.

**Disposition.** Recorded only; `market-spec.md` is not edited here. Plan change to the sentence:
*"…recorded there as **this repository's rulings** — the §4.2.1 preamble describes them as deviations
from the literal SIDC/XBID description, while each rule's classification records it as consistent
with the market rules (an inconsistency noted in domain review v3.0, V3-21) — so C8, C9 and C10 certify
conformance to those rulings, not to XBID itself."*

---

## 14. Disposition table (v3.0)

| ID | Severity | Scenario / area | Finding | Recommended disposition | Plan change? |
|---|---|---|---|---|---|
| V3-01 | High | C2, C5, C10, C11, C16, C1/C7/C0 | The rewrite lost five signed v2.0 fixes (C10 §4.2 attribution, C2 ACER/CACM, C11 §8.5, C5 resting-residual draw, C16 §2.2 clause) plus minor ones | Restore with the texts in V3-01; correct D2 in the same edit | Yes |
| V3-02 | High | I5, C8 | "Maker" is undefined at the Rule-3 hop; §4.2.1 Rule 3 and §6.2 item 7/§5.2 read differently | File OQ-31 before C8 K2; I5 says "price-setter" | Yes, plus an OQ |
| V3-03 | High | C12 | The per-participant non-zero settlement assertion fails on a legal self-match or round-trip | Replace with gross $\sum\lvert V\rvert > 0$ plus per-participant recomputation | Yes |
| V3-04 | Medium | I6(a), C12 | Exact integer zero-sum is tautological from rows, or not observable from the engine | Restate I6(a) on engine output within the I6(b) tolerance | Yes |
| V3-05 | Medium | C9 | The wrong-sign draw pins behaviour on a forbidden input (OQ-23 note (a)); "Certifies §13.6" overclaims | Make it an observation; record §13.6 as a gap; fix the §13.x bullet | Yes |
| V3-06 | Medium | C9, C12, C0, all | Unicode look-alikes (§13.6 uses ASCII `-3.00`; ×, ·, −, —) defeat verbatim matching | HARNESS-1 Unicode tests; byte-copy quotes | Yes (HARNESS-1) |
| V3-07 | Medium | C8, C9, C14, C17, C0 | Quotes from open-questions and the review cannot be checked; the checker reads market-spec only | `SpecQuote` names its source document | Yes (HARNESS-1) |
| V3-08 | Medium | C8 | "§5.2 bullet 3" now points at the partial-execution bullet; the exemption is bullet 4 | Label by lead words; no bullet positions | Yes |
| V3-09 | Medium | C8 | Incoming-iceberg draw undefined; its single-timestamp claim is ungrounded | Define the draw; scope the timestamp claim | Yes |
| V3-10 | Medium | C0, out-of-scope | §3.3 phase-2 citation: correct rule, wrong source (R-1); OQ-28 reopen not listed | Quote the phase description; add the OQ-28 bullet | Yes |
| V3-11 | Medium | Out-of-scope, L5 | §3.2: day labels inconsistent (confirmed); quarter-hour periods correct (refuted) (R-2) | Replace the bullet and row 18 | Yes |
| V3-12 | Medium | C0, gap list | SUSPENDED collapse is conformant for submission; SUSPENDED is not a §9.2 contract status | Reword C0; remove the gap bullet | Yes |
| V3-13 | Medium | C0, C11 | Hour-aligned GCT(k) is quoted but certified nowhere (GCT is an input) | Qualify Certifies; add out-of-scope bullet | Yes |
| V3-14 | Medium | C14a′ | Certifying an unflagged eviction contradicts C14's own defect standard | "Recorded, not certified"; cite the fill-report close-out | Yes |
| V3-15 | Medium | C14 | Battery exclusions incomplete (I1/I6/I2 on C14b; I3 on C14c) | Per-part exclusions, asserted expected-violated | Yes |
| V3-16 | Medium | C11 | No iceberg draw; GCT expiry of hidden quantity is uncertified | Add iceberg draws; classify hidden as expired | Yes |
| V3-17 | Low | I2 | "Eviction" listed as a §5.1 cancellation cause | Separate `evicted` bucket, 0 outside C14 | Yes |
| V3-18 | Low | I4, rule 3 | Hidden maker identity attributed to anonymity, not to the trade-row schema | Reword | Yes |
| V3-19 | Low | C5 | Stale "spec does not state" text; `PARTIALLY_FILLED` is not a spec status | Reword; cite the v2.9 source | Yes |
| V3-20 | Low | C17 | OQ-25 correction misdated (09-29 → 10-01); OQ-10 EXTENSION and R1 not named | Correct the date; add two caveats | Yes |
| V3-21 | Low | Residual-risk paragraph | §4.2.1 preamble ("three rules … deviate") versus four rules each "consistent" | Record only; reword the plan sentence | Yes (wording) |

**Rulings recorded (section 12):** R-1 (§3.3 phase-2 citation: correct rule, wrong source; the reopen
citation is apt) and R-2 (§3.2: day labels inconsistent, quarter-hour periods correct). **Open item
raised:** OQ-31 (V3-02), proposed for `open-questions.md` by a later domain-analyst dispatch. This
review did not file it.

**Re-review trigger.** When the orchestrator incorporates or answers these findings, any change to
I5, I6 or C8's expectations returns here before C8's or C12's K2. So does OQ-31's resolution. D2
(`scenario-ladder.md`) must be re-checked against V3-01 in the same pass.

*Signed: Domain Analyst, v3.0, 2026-10-02.*

---

## 15. Incorporation check (v3.1)

**Scope.** Each v3.0 finding was checked against `scenario-ladder.md` v2.0 (D2, the plan's Track A
regenerated verbatim). The plan was checked for its change table (row 32, and rows 18 and 25), the
`OMIE-CERT-HARNESS-1` acceptance criteria (plan:620) and L5 (plan:2165-2204). Line references below
are to D2 unless marked "plan:". Quotes were re-checked against `market-spec.md` v2.12 under the
archived checker's contract (section 11).

**V3-01 regressions.** D2 contains **none** of the five v2.0 regressions, and neither does the plan.
A grep of both files for the pre-fix strings ("§8.5 is the precise ground", "the ACER/CACM provenance
the certificate wants", "no-cross-matching clause", the §4.2 FOK attribution, the single-draw C5
dataset, `CANCELLED (IOC/FOK unmatched)` and `(… GCT reached)` partials) returns no match. The
restored texts are at C10 Quotes (:699), C2 Quotes (:409), C11 Purpose and Quotes (:715-718, :731),
C5 Dataset and Known result (:535-536) and C16 Quotes (:953). The minor ones are at C1 (:384), C7
(:583), C11 (:730) and C0 (:342-344).

| ID | Incorporated? | Note |
|---|---|---|
| V3-01 | yes | All five restored, plus the minor ones. See the paragraph above |
| V3-02 | yes | I5 (:181) and C8 Known result (:625). OQ-31 is now filed (`open-questions.md:1136`). See N-5 on the I5 wording |
| V3-03 | yes | C12 hazard paragraph (:774-780) |
| V3-04 | **partial** | I6 row (:182), C12 Purpose (:748-752) and C12 Pass (:789) are correct. The "Conservation invariants" primer still says the cash *"must sum to exactly zero across participants (I6)"* (:55-56), which contradicts I6(a)'s tolerance. Replace with *"must sum to zero across participants, within the price-decode tolerance (I6)"* |
| V3-05 | yes | C9 Known result (:664), Certifies (:661), §13.x bullet (:1114-1116), OQ-23 citation form (:653, :1083). See N-1 on a separate overclaim in the same Certifies row |
| V3-06 | yes | HARNESS-1 Unicode tests (plan:620), notebook item 2 (:1148-1150), C9's ASCII note (:648). The plan no longer has the U+2212 "−3.00" |
| V3-07 | yes | HARNESS-1 (plan:620) and notebook item 2 (:1150-1153) |
| V3-08 | yes | C8 Certifies and Quotes (:622-623), notebook item 2 (:1154-1155), row 25 (plan:38) |
| V3-09 | yes | C8 Dataset and Known result (:624-625) |
| V3-10 | yes | C0 Quotes (:351), §3.3 sub-bullet (:1067-1068), OQ-28 bullet (:1054-1055). See N-4 on where the sub-bullets were placed |
| V3-11 | yes | Bullet (:1062-1066) and row 18 (plan:31). L5 (plan:2197-2204) does not quote §3.2 and takes periods from the current Rules, consistent with R-2. Its remaining hedge, whether quarter-hour IDA products are in force for a tape's dates, is the Track B owner's to close. Spec §2.1 dates the 15-minute MTU from 18 March 2025, before every tape on disk |
| V3-12 | yes | C0 second paragraph (:338-345). The gap-list bullet is removed; the gap list now holds §13.6 only (:1082-1084) |
| V3-13 | yes | C0 Certifies and Quotes (:350-351), C11 Certifies (:730), out-of-scope bullet (:1056-1057). The exact v2.10 quote matches source after emphasis stripping |
| V3-14 | yes | C14a′ (:854), Purpose (:843-844), Certifies (:859). The close-out record's `sprint-status.md` heading is the story id itself (`## OMIE-ICE-INCOMING-FILL-REPORT-1`, `sprint-status.md:13864`), so naming the story identifies it |
| V3-15 | yes | C14 Oracle kind (:860) |
| V3-16 | yes | C11 Purpose, Dataset and Known result (:718-719, :732-733) |
| V3-17 | yes | I2 (:178) |
| V3-18 | yes | Rule 3 (:135-138) and I4 (:180) |
| V3-19 | yes | C5 Purpose (:500-504) and Known result (:536) |
| V3-20 | yes | C17 Quotes (:1034), cannot-claim (:1024-1025), R1 (:1011-1013) |
| V3-21 | yes | Residual-risk paragraph (:105-110) |

### 15.1 New items

| ID | Severity | Where | Finding | Recommended change |
|---|---|---|---|---|
| N-1 | Medium | C9 Certifies (:661) | It still lists *"§4.2's ICEwPI sign rule"* as certified. The engine does not enforce that rule (the §13.6 gap the same row records), and C9's main draws are sign-valid by construction. What C9 can certify is the drift **given** a sign-valid increment. This predates revision 2; v3.0 missed it | Replace with *"§4.2's ICEwPI slice pricing (each new slice at the previous slice price ± the fixed delta), on sign-valid draws; the sign rule itself is the §13.6 conformance gap"* |
| N-4 | Medium | Out-of-scope list (:1059-1078; plan:1386-1404) | The "Two known inconsistencies in `market-spec.md`" paragraph and its two bullets were inserted **inside** the "Out of scope because unimplemented" list. The bullets that follow (§3.4 halts, §4.3/§4.4/§4.5, §4.6, §5.4, §8.3, §8.2/§8.5/§2.4) now read as market-spec inconsistencies. This breaks the categories the section says it keeps distinct | Move the paragraph and its two bullets to after the last unimplemented bullet (§8.2/§8.5/§2.4), or give them their own bold heading. Do not change the text |
| N-2 | Low | C11 Certifies (:730) | The italic *"no automatic position close-out at GCT; open orders are cancelled at GCT but already-matched trades are firm"* looks like a quote but is a paraphrase. Source: *"The continuous market has no automatic position close-out at GCT. Open orders are cancelled at GCT but already-matched trades are firm."* (`market-spec.md:458`). The Quotes cell is correct | Drop the italics, or byte-copy the two sentences |
| N-3 | Low | C4 Certifies (:480) | `CANCELLED (by agent)` is the same partial-parenthetical hazard as V3-01's minor C7/C11 item. The source branch is `CANCELLED (by agent; GTD expiry; GCT; market halt)` (`market-spec.md:317`) | Quote the whole branch, as C7 and C11 now do |
| N-5 | Low | I5 (:181; plan:453) | The parenthesis "(including the §4.2.1 Rule-3 hop, per OQ-31 …)" is attached to the **iceberg** clause, so it presumes Reading B before OQ-31 is ruled. This is this review's own V3-02 wording | Move it out of the iceberg clause: *"… or the iceberg under §6.2 item 7 and §5.2; at the §4.2.1 Rule-3 hop, the price-setter OQ-31 rules (to be resolved before C8's K2)"* |
| N-6 | Low | Plan row 32 (plan:45) | *"Rulings R-1 and R-2 correct the two `market-spec.md` inconsistencies of row 18"* overstates them. The rulings do not edit `market-spec.md`. R-1 finds the rule correct and its source wrong. R-2 confirms the day labels and refutes the periods | *"Rulings R-1 and R-2 settle the two `market-spec.md` points of row 18 (R-1: correct rule, wrong source; R-2: day labels inconsistent, periods correct); `market-spec.md` is not edited"* |

None of N-1…N-6 changes a scenario's expected result. N-1 and N-3 matter for the certificate's
wording and the K5 quote check. N-4 matters for how a reader classifies a gap. N-5 matters only
until OQ-31 is ruled.

**Re-review trigger (unchanged).** OQ-31's resolution, and any change to I5, I6 or C8's expectations,
returns here before C8's or C12's K2.

*Signed: Domain Analyst, v3.1, 2026-10-02.*

---

## 16. D3 re-review of the merged-candidate invariant battery (v3.2)

**Scope.** This is the re-review that `OMIE-CERT-INVARIANTS-1` decision V16 makes a K2 precondition of
C08 and C12. It covers the battery as pinned by the card's Step-2 decisions (`agile-plan.md:2140`,
v5.185, amended v5.186) and its Description (`agile-plan.md:2141`): I2's eviction signature (V4), I4's
queue key (V8), I5's price-setter, including the Rule-3 hop and `unruled_hop` (V9), and I6's tolerance
and cash-leg scope (V10). The rules are `market-spec.md` v2.13 (§4.2, §4.2.1, §5.1, §5.2, §6.1, §6.2,
§6.6, §8.1, §8.4), `open-questions.md` §OQ-31 (RESOLVED 2026-10-02, Reading B) and the Continuous
Trading Matching Algorithm. I read `tests/omie/reference/omie_invariants.py` only to confirm what each
pinned statement does. Code quality is not judged here. The question asked of each statement: would it
fail a correct engine, or pass a wrong one?

### 16.1 Verdicts

| Statement | Decision | Verdict | Note |
|---|---|---|---|
| **I2**, eviction signature and the §5.1 buckets | V4 | **CONFORMS** | See below. D3-07 is informational |
| **I4**, queue key on engine timestamps; regenerated slices are new entries | V8 | **CONFORMS** on the key. **FINDING** at the Rule-3 hop | D3-01 (Medium), D3-05 (Low) |
| **I5**, price-setter, Rule-3 hop pinned to the iceberg | V9 | **CONFORMS** | Matches v2.13 §4.2.1 Rule 3 and OQ-31. D3-03 (Low) |
| **I5**, `unruled_hop` fails | V9 | **CONFORMS** (failing is the right verdict) | The message gives the wrong reason: D3-04 (Low) |
| **I5**, matchability | — | **FINDING** (a gap: no check covers it) | D3-02 (Medium) |
| **I6**, tolerance inputs and cash-leg scope | V10 | **CONFORMS** | See below |

**I2 (V4), CONFORMS.** Evictions are their own bucket, not a §5.1 cause, as V3-17 required. Each §5.1
bucket comes from the order's own fate:

- `ioc_remainder` subtracts only the IOC's own aggressor fills, never resting quantity. An engine that
  wrongly rests an IOC remainder therefore counts it twice and fails I2.
- `fok_kill` is non-zero only when the FOK has zero fills. A partial FOK fails.
- `cancellation` and `gate_expiry` read visible + hidden quantity just before the call. A cancel or
  expiry that leaves visible or hidden quantity behind counts it twice and fails. Counting hidden
  quantity as expired at GCT is what V3-16 requires (§5.1 *"CANCELLED (by agent; GTD expiry; GCT;
  market halt)"*, `market-spec.md:323`).

On a correct engine, an order that disappears within one submission with no fill and no cancel can only
be an implementation-limit fate. No market rule removes it, so the six-condition signature cannot
misclassify a rule-governed fate. Outside C14 any `evicted > 0` fails, which is the safe direction.

**I4 (V8), queue key CONFORMS.** The key is the engine's stored `(SEC, NSEC)`, and a slice whose depth
is used up is treated as a new entry behind every same-price order that still has depth. This is the rule:

- §5.2: *"A **new timestamp** is assigned on: … activation of a new iceberg slice"* (`market-spec.md:330`).
- §6.1: *"ties broken by earliest timestamp"* (`:356-357`).
- Matching Algorithm: *"Orders with the same price limit are prioritized by their time stamp oldest
  first."* (`Continuous_Trading_Matching_Algorithm_Public_Description.md:457`).

Tape timestamps could not express the lost priority, so V8 rightly rejects them. The decomposition is
also correct for the Rule-2 multi-slice path. There, an aggressor that trades the iceberg's renewed
slice while an earlier same-price order still has depth is flagged.

**I5 (V9), CONFORMS.**

- When the submitting order is the aggressor, the price-setter is the resting order (§6.2 item 3,
  `:365`; Matching Algorithm §5.1.1, `:469`).
- When a passive iceberg's slices are traded, the price-setter is the iceberg (§6.2 item 7, `:369`;
  §5.1.2, `:475-479`).
- At the Rule-3 hop (`passive_oid` is the submitting order's own remainder), the price-setter is the
  iceberg. This is the v2.13 text: *"The hop trades at the **iceberg slice's own limit price** … It
  never trades at the aggressor's limit $L$."* (`market-spec.md:240`).

`RULE3_HOP_PRICE_SETTER = "iceberg"` matches OQ-31's Resolution. The `rule3_hop` broken fixture prices a
hop at the aggressor's limit, which is the discriminating case OQ-31 asks for (*"draw $L$ strictly
through the slice price"*).

The ladder exemption accepts any rung $\mathit{base} + k\,\mathit{ppd}$, $0 \le k < \lceil
\mathit{qty}/\mathit{peak} \rceil$, rather than the slice actually live. This is the narrow exemption
the battery states. It is a known residual, not a finding: a slice priced on the wrong rung passes
I5. C09 owns the drift order. A tighter bound derived from cumulative fills is ambiguous for an iceberg
that first traded as an incoming aggressor, so I do not recommend one.

**I6 (V10), CONFORMS.** The valuation is §8.1, *"V (EUR) = Q (MW) × p (EUR/MWh) × d (h)"* with *"d =
0.25"* (`market-spec.md:429`, `:436`). In ticks it is $V_s = k\,D_s$, with $k = 0.1 \times 0.01 \times
0.25 = 2.5\times10^{-4}$ EUR and $O = 9999/0.01 = 999\,900$. Both match the §5.4 price bounds (*"below
−9,999 or above +9,999 EUR/MWh"*, `:342`).

I recomputed the pinned example. For $p = 1\,004\,598$ (46.98 EUR/MWh) and $q = 20$ (2 MW):

- $V = 23.49$ EUR, which is $2 \times 46.98 \times 0.25$.
- $\tau = 6.043878\times10^{-4}$ EUR.

The other domain inputs:

- **Cash-leg scope.** I6 covers the cash leg only, as §8.4 requires: *"`−V` (cash outflow) + market
  value of acquired energy (exogenous …)"* (`:458`).
- **Self-matches.** A self-match nets to zero, consistent with §6.6, *"an agent's buy order can match
  its own sell order"* (`:391`), and with V3-03.
- **Participant enumeration.** Every buyer and seller agent is enumerated and equals the declared agent
  of its oid. This is the V3-04 content.
- **The I6(a) bound.** `compute_agent_pnl` accumulates in int64 and float64 and makes one float32 cast
  per agent (`settlement.py:145-160`). The per-agent error is then at most
  $u\,\lvert\mathit{pnl}_a\rvert$. Summed over agents this is at most $2ku\sum_i\lvert p_i - O\rvert
  q_i$, which lies inside $\sum_s \tau_s$. V10's own argument (`agile-plan.md:2140`) therefore holds.

The float32 derivation of $\tau_s$ is numerical analysis, outside domain authority. I find nothing in
it that contradicts a rule. Settlement granularity at the unit level (§8.3) does not affect the check,
because a zero sum holds over any partition of the participants.

### 16.2 Findings (v3.2)

#### D3-01 — Medium — I4 checks priority on the wrong side at the Rule-3 hop, so a wrong engine passes

**Rule.**

- v2.13 §4.2.1 Rule 3: *"For price determination the iceberg is the passive (resting) side of the hop"*
  (`market-spec.md:240`).
- The same section: *"In the real algorithm this hop is not a separate event. It is one more slice
  inside the aggressor's single matching process."* (`:244`).
- Matching Algorithm §5: *"the best (i.e. highest-price) buy order is always matched with the best (i.e.
  lowest-price) sell order first. Orders with the same price limit are prioritized by their time stamp
  oldest first."* (`:455-457`).
- §5.1.3: matching continues *"as long as the incoming order remains executable"* before the remainder
  is entered into the book (`:490-493`).

**Finding.**

- **What I4 does.** V8 treats the hop's maker as *"the submitting order's own remainder at a Rule-3 hop"*.
  It looks for competitors on the **remainder's** side (`omie_invariants.py:1021-1048`, `side` taken
  from the sign of `Q`, maker = `passive_oid`).
- **Why that is vacuous.** On sign-valid input no earlier order on that side can be priced at or better
  than the hop price. The pre-submission book is uncrossed, and the slice price is at or beyond the
  iceberg's earlier visible price.
- **What the rule requires.** The priority question at the hop is on the **iceberg's** side. By the
  rules, the hop is the aggressor $o$ trading the iceberg's renewed slice, a new-timestamp entry. So no
  order on the iceberg's side, priced better or equal, may still have depth.
- **A wrong engine that passes.** $o$ = BUY NON 10 @ 55. Resting: sell iceberg $j$ (peak 2 @ 50, PPD 0),
  and sell $k$ 5 @ 50, later than $j$'s first slice. The engine stops $o$ after $j$'s peak and rests
  8 @ 55, though $k$ still crosses. $j$'s regenerated slices then hop-consume the 8.
  - The rules require $j$ 2, then $k$ 5, then $j$ 3.
  - The engine prints $j$ 2, then $j$ 8 through hops; $k$ never trades.
  - I3 passes, because the final book is uncrossed. I5 passes (all at $j$'s limit), as do I2 and I7.
    I4 sees maker $o$ with no competitors on the BUY side and also passes.

C08 is the scenario that draws this configuration, so this check is weakest exactly where C08 relies
on it.

**Correct engines are not failed by the fix.** For a hop fill (`aggressor_oid ≠ o`, `passive_oid ==
o`), also evaluate priority with the iceberg `aggressor_oid` as the maker, as a **new entry**, on the
iceberg's side. Every pre-snapshot entry on that side with remaining depth and a better or equal price
is then a violation. A correct engine never trips this: $o$ rests its remainder only after every
crossing entry on the iceberg's side has been exhausted (§5.1.3), so all such entries have depth 0 in
the map. Keep the existing check.

**Disposition.** Battery change to I4 (V8), with a broken fixture of the configuration above. I4 is
not I5 or I6, so V16's return trigger does not apply. It is in this review's V8 scope, though, so the
change should come back here. It does **not** block C08's K2. It must be closed before C08's K5
verdict, unless C08's Rule-3 Known result pins the exact counterparty sequence (see §16.3).

#### D3-02 — Medium — no check in the battery covers matchability (§6.2 item 2), so a non-crossing trade passes

**Rule.**

- §6.2 item 2: *"Match is possible if: orders are opposite sides, same contract, and `sell_price ≤
  buy_price`."* (`market-spec.md:364`).
- v2.13 Rule 3: *"If the regenerated slice's price does not cross $L$, there is no hop trade (§6.2 item
  2; Rule 2)."* (`:240`).

**Finding.**

- **What I5 checks.** I5 checks that the price equals the **setter's** limit. Nothing checks that the
  **other** leg's limit reaches that price.
- **A wrong engine that passes.** It matches a BUY @ 40 against a resting SELL @ 50 at 50. I5 passes
  (the price is the setter's), as do I1, I2, I3 and I7. I4 also passes when the SELL is the best ask.
- **At the hop.** A slice printed on its ladder at a rung beyond $L$ also passes I5.
- **Coverage elsewhere.** Exact-oracle scenarios (C1, C2, C8) would catch it on their own draws. The
  battery's role on random draws (V15(b), C16) is the generic net, and that net misses this case.
- **History.** This is the counterpart of the rejected formulation the card records (*"a fill taking a
  60.00 maker at 50.00"*). That example ran in the price direction; this one runs in the matchability
  direction.

**Disposition.** Add a sub-claim to I5 (proposed `field="limit_cross"`) or a new invariant.

- For each fill: $\mathit{price} \le$ the buyer's declared limit and $\mathit{price} \ge$ the seller's
  declared limit.
- For an ICE leg, use the most permissive rung of its declared ladder: the maximum rung for a buyer and
  the minimum rung for a seller. On sign-valid input both are $\mathit{base}$, since buy increments are
  negative and sell increments positive (§4.2, `:209`).
- This direction never fails a correct engine.

The change touches I5, so it returns here (V16). It is not a K2 precondition: C08's Known result is
exact-parametric and C12 is settlement-only.

#### D3-03 — Low — I5's hop branch takes any aggressor as the price-setter

**Rule.** Matching Algorithm §5.1: *"An order with a new time stamp may be a newly entered order, a
modified order, a (re)activated order that was inactive before, or a new slice of an iceberg order."*
(`:462-464`).

**Finding.**

- **What can legitimately aggress.** The battery's driving loop does no modification or reactivation
  (V2). So inside submission *s* the only entry that can legitimately aggress against $o$'s remainder
  is a **new iceberg slice**.
- **What the hop branch does.** It accepts any `aggressor_oid` as the setter
  (`omie_invariants.py:1095-1098`). It prices a non-ICE aggressor at that order's own declared limit.
- **A wrong engine that passes.** It re-submits a resting NON order mid-submission, and I5 passes.

**Disposition.** In the hop branch, require `aggressor_oid` to be an accepted **ICE** order on the side
opposite $o$. Otherwise report `unruled_hop`. This is a small change, but it is a change to I5, so it
returns here.

#### D3-04 — Low — `unruled_hop` is right to fail, but its stated reason is wrong; the sign-invalid case must be excluded

**Rule.**

- Matching Algorithm §3.3: *"their full quantity is exposed to the market for matching"* (`:257-258`).
- §4.2.1 Rule 4: *"It is **spec-violating** to leave the regenerated slice resting visibly in a crossed
  book until the next incoming aggressive order triggers matching"* (`market-spec.md:249`).
- §13.6 / §4.2: the ICEwPI sign rule (`:209`).

**Finding.** The message says *"no ruling covers it"* (`omie_invariants.py:1103`).

- **Sign-valid input.** A slice activated in submission *s* that trades against an order other than
  $o$ needs that order to rest at a price crossing the iceberg's next slice. That means one of two
  earlier defects:
  - matchable hidden quantity went unmatched at an earlier submission (§3.3; Rule 2/3), or
  - an activation was deferred across cycles (Rule 4).

  So a correct engine cannot produce this configuration. Failing it is the **rule-correct** verdict, but
  the configuration is ruled against, not unruled.
- **Sign-invalid ICEwPI** (sell increment < 0, buy increment > 0). A drifted slice can cross
  pre-existing orders. The input is domain-forbidden (§13.6 conformance gap), so here "no ruling" is
  literally true. C09's wrong-sign draw is an observation (V3-05).

**Disposition.**

1. Reword the violation message to name §3.3 / Rule 4 as the probable cause. Keep the `unruled_hop`
   field name; the guard pins it.
2. C09's wrong-sign observation draw must list I5 `unruled_hop` (and any I4 consequence) as an
   **expected, recorded** outcome, never a `NOT CERTIFIED` cause. A note to C09's K2.

No change to the verdict and no new OQ.

#### D3-05 — Low — I4's *Cannot see* text contradicts V5 and V8, and corrects this review's V3-18

**Rule.** The battery reads jaxob trade rows, where `PASS_OID` names the resting order on every row
(V5: *"**The maker oid is observable:** `PASS_OID` names the resting order on every jaxob trade row.
What lacks it is the `(M, 6)` settlement tape"*). It also keys the queue on engine timestamps (V8).

**Finding.** Three places still say *"trade rows carry no maker order id, so at equal price the queue
is a reconstruction using tape timestamps"*:

- the card's Description (`agile-plan.md:2141`);
- `STATEMENTS["I4"]` (`omie_invariants.py:1488`);
- D2's I4 row (`scenario-ladder.md:180`).

The battery-preamble clause *"its trade rows carry no maker order id"* is the same claim. Both premises
are false for the battery. The conclusion still holds, for a different reason: the partial-fill
priority ruling (§5.2, `:331`) cannot be falsified *through I4*, because I4 trusts the engine's own
timestamps (`unobserved = ("queue_timestamp_correctness",)`). This review's V3-18 wording caused the
error. V3-18 was right about the `(M, 6)` settlement schema, but the battery does not read that schema.

**Disposition.** Restate the *Cannot see* text: *"The queue key is the engine's own stored timestamp, so
whether the engine assigned those timestamps per §5.2 (partial execution keeps its timestamp; a new
iceberg slice gets a new one) is not observable through this check (C05's interloper certifies it)."*
Apply the same correction to the battery preamble and to plan rule 3, where that rule speaks of the
battery's trade rows. `STATEMENTS` is verbatim from the Description, so all three must change in one
edit, together with any test that pins the text. Wording only. It changes no verdict and does not block
K2. The notebooks will print this text, though, so fix it before the first one is certified.

#### D3-06 — Low — D2's I5 row is stale after OQ-31

**Finding.** `scenario-ladder.md:181` still reads *"at the §4.2.1 Rule-3 hop, whichever price-setter
OQ-31 rules (it must be resolved before C8's K2)"*. The card and `STATEMENTS["I5"]` carry the resolved
wording (*"at the §4.2.1 Rule-3 hop, the iceberg (OQ-31 RESOLVED 2026-10-02, Reading B …)"*). C08's K2
author is given the specification paths, and D2 is among them.

**Disposition.** Replace the D2 I5 row with the card's wording before C08's K2 is dispatched. This is a
document edit with no rule content.

#### D3-07 — Informational — "the §5.1 causes only" is a subset of §5.1

**Rule.** §5.1: *"CANCELLED (IOC/FOK unmatched; GTD expired; GCT reached; halt/hibernation)"*
(`market-spec.md:318`).

**Finding.** `CANCEL_CAUSES` (`omie_invariants.py:98-99`) and the I2 statement list four causes. They
omit GTD expiry and halt/hibernation.

- **Today this is correct for the engine.** The engine does not enforce GTD expiry (§4.6; no GTD
  handling in `contract_book.py`), and halts are out of scope (§3.4).
- **The consequence.** A GTD order rests past its expiry today. That is an unimplemented-rule gap,
  already on the out-of-scope list, and the battery neither catches nor certifies it.
- **The hazard.** If GTD expiry is ever implemented, I2 would fail a **correct** engine until a
  `gtd_expiry` cause is added.

**Disposition.** Record only. When convenient, read *"the §5.1 causes the simulator implements"*.

**Outside this review's scope (note for the architect).** The engine labels a hop's iceberg as
`AGRS_OID`, while the rules treat it as the passive side. These are the same attribution question:
- OQ-31's engine-facing item (b).
- D3-01 above.
- M5's map, keyed on `aggressor_oid ∈ aggressors` (`omie_invariants.py:1407`). It leaves out the hop
  trades of the aggressor's own matching process. A split-versus-whole pair in which only one side hops
  would make M5 differ on a correct engine.

V16's probe found 0 hops, so this is latent.

### 16.3 Disposition and gate

| ID | Severity | Statement | Wrong engine passes / correct engine fails | Change | Returns to D3? | Blocks K2? |
|---|---|---|---|---|---|---|
| D3-01 | Medium | I4 (V8) at the Rule-3 hop | A wrong engine passes | Also check priority on the iceberg's side, iceberg as a new entry | Yes (V8 scope) | No. Close it before C08's K5, or pin it in C08's oracle |
| D3-02 | Medium | Matchability (I5 or new) | A wrong engine passes | `limit_cross` sub-claim | Yes (touches I5) | No |
| D3-03 | Low | I5 hop branch | A wrong engine passes (exotic) | Require an ICE aggressor at the hop | Yes (touches I5) | No |
| D3-04 | Low | I5 `unruled_hop` | Neither; reason misstated; C09 sign-invalid draw | Reword the message; C09 lists it as expected | No | No |
| D3-05 | Low | I4 *Cannot see*, preamble, D2 row | Neither (text) | Restate (§16.2) | No | No |
| D3-06 | Low | D2 I5 row | Neither (stale text) | Copy the card's wording | No | No (do it before C08's K2 dispatch) |
| D3-07 | Info | I2 cause list | Latent: a correct engine with GTD expiry would fail | Record only | No | No |

**New OQ needed?** **No.** Every finding applies a rule already stated in `market-spec.md` v2.13 or the
Matching Algorithm. None depends on an ambiguous text. `market-spec.md` is not edited.

**Gate.**

- **C12 K2 may proceed.** I6 conforms, and no finding touches it.
- **C08 K2 may proceed**, provided that:
  1. D2's I5 row is updated first (D3-06).
  2. The Rule-3 draw's Known result pins the **exact** trade sequence: price, quantity and counterparty
     order, with the remainder resting only after no crossing entry on the iceberg's side has depth.
     It also draws $L$ strictly through the slice price (OQ-31's Consequences). This covers D3-01 and
     D3-02 at the oracle until the battery fixes merge.

**Re-review trigger.** The fixes for D3-01, D3-02 and D3-03 return here. So does any other change to I5
or I6. D3-04 to D3-07 do not.

*Signed: Domain Analyst, v3.2, 2026-10-02.*

---

## 17. D3 return review of the v5.187 fixes (v3.3)

**Scope.** This is the return pass that §16.3's re-review trigger and the card's V16 (as amended at
v5.187) require. It covers the implemented fixes for D3-01 (V8 `hop_priority`), D3-02 (V9
`limit_cross`) and D3-03 (V9 ICE-only hop branch), and the V10 tolerance amendment (per-dispatch
$c_s$), which touches I6. I also read V12's M5 counterparty key, because §16.2 raised it. The rules are
`market-spec.md` v2.13 (§4.2, §4.2.1, §5.2, §6.1, §6.2, §8.1) and `open-questions.md` §OQ-31 (RESOLVED
2026-10-02, Reading B). The implementation read is `tests/omie/reference/omie_invariants.py`:
`_priority_violations` / `check_i4_priority`, `_ladder` / `check_i5_price_setter`, `i6_tolerance` /
`check_i6_settlement`, `_m5_map`, and `STATEMENTS["I4"]` / `STATEMENTS["I5"]`. I also read the broken
fixtures in `tests/omie/test_omie_invariants_checks.py` (`TestI4Broken`, `TestI5Broken`). As before, the
question is rule conformance: would the check fail a correct engine, or pass a wrong one? Code style is
not judged.

### 17.1 Verdicts

| Item | Decision | Verdict |
|---|---|---|
| D3-01, I4 priority on the iceberg's side at the Rule-3 hop | V8 (v5.187) | **CLOSED** |
| D3-02, matchability (`limit_cross`) | V9 (v5.187) | **CLOSED** |
| D3-03, hop branch requires an accepted ICE aggressor on the opposite side | V9 (v5.187) | **CLOSED** |
| V10, $\tau_s$ with $c_s$ and $(1 + (c_s + 8)u)$ | V10 (v5.187) | **CONFORMS** (no rule content changed) |
| V12, M5 keyed by counterparty | V12 (v5.187) | **CONFORMS** |
| D3-04 message, D3-05 *Cannot see*, D3-06 D2 row (no return required) | V9, Description | Confirmed applied |

**D3-01, CLOSED.**

- **The rule.** v2.13 §4.2.1 Rule 3: *"For price determination the iceberg is the passive (resting)
  side of the hop"* and *"In the real algorithm this hop is not a separate event. It is one more slice
  inside the aggressor's single matching process."* (`market-spec.md:240`, `:244`). Matching
  Algorithm §5: *"Orders with the same price limit are prioritized by their time stamp oldest first."*
  (`:457`). A new iceberg slice gets a new timestamp (§5.2, `market-spec.md:330`).
- **What the code does.** For a fill with `aggressor_oid != o` and `passive_oid == o`, the iceberg's
  side is taken opposite $o$'s **declared** side. Every pre-snapshot entry on that side, other than the
  iceberg's own rows, that still has depth in the map and is priced at or better than the fill price
  (≤ for SELL, ≥ for BUY) is a `hop_priority` violation. The hop does not decrement the map (in the
  code, $o$ is not a pre-snapshot key). The original walk is kept.
- **The v3.2 wrong engine now fails.** $o$ = BUY 10 @ 55; sell iceberg $j$ peak 2 @ 50; sell $k$ 5 @ 50,
  later. The engine prints $j$ 2 and then $j$ hops for the rest. $k$ still has depth 5 at 50 ≤ 50, so
  I4 fails with `oid = k`. This is `test_i4_fails_on_a_hop_while_another_entry_on_the_icebergs_side_has_depth`.
- **A correct engine passes.** The real-engine record of that configuration prints $j$ 2, $k$ 5, $j$ 2,
  $j$ 1 (`test_d3_01_real_record_has_the_probed_fills_and_passes_i4_and_i5`). That is the rule
  sequence, and it passes I4 and I5. A correct hop fill can only follow the exhaustion of every
  crossing entry on the iceberg's side (§5.1.3, matching continues *"as long as the incoming order
  remains executable"*), so all such entries have depth 0 in the map. The control
  (`test_i4_accepts_a_hop_once_the_icebergs_side_is_exhausted`) passes with `hop_fills == 2`.
- **The comparison price is the right one.** A competitor that crosses $o$'s limit but is priced worse
  than the hop price is correctly ignored: the slice at the better price legitimately trades first
  (§6.2 item 1, best price first). A competitor at a price equal to the hop's is ahead of the slice,
  because the slice is a new-timestamp entry. So the test "≤ fill price" (SELL side) is exactly the
  set of entries the rules put ahead of the slice.
- **A residual, not a reopening (D3-08 below).** The map holds pre-snapshot **visible** depth. A second
  iceberg whose visible slice is used up in the map, but which still has a crossing regenerated slice,
  is not counted as a competitor. That limitation is shared with V8's original walk, and it is not
  specific to the hop.

**D3-02, CLOSED.**

- **The rule.** §6.2 item 2: *"Match is possible if: orders are opposite sides, same contract, and
  `sell_price ≤ buy_price`."* (`market-spec.md:364`). v2.13 Rule 3: *"If the regenerated slice's price
  does not cross $L$, there is no hop trade"* (`:240`).
- **What the code does.** For every fill whose two legs are accepted, I5 requires `price_int ≤`
  max(buyer's ladder) and `price_int ≥` min(seller's ladder), with `field="limit_cross"`. `_ladder`
  gives the declared price for a non-ICE leg (and for ICE with PPD 0), and $\mathit{base} +
  k\,\mathit{ppd}$, $0 \le k < \lceil \mathit{qty}/\mathit{peak} \rceil$, otherwise. This is the most
  permissive rung on each leg, as §16.2 asked. Legs that are not accepted are left to I1 (`fill_sides` /
  `fills`). The check runs before, and independently of, the price-setter branch.
- **The v3.2 wrong engine now fails.** For a BUY @ 40 matched with a SELL @ 50 at 50, the price is
  above the buyer's limit and the check fails on the buyer leg
  (`test_i5_fails_a_fill_above_the_aggressing_buyers_limit` is the same configuration, buyer one tick
  below). At the hop, a slice printed at a rung beyond the aggressor's $L$ fails on $o$'s leg, because
  $o$ is non-ICE in the Rule-3 configuration, or ICE with the base as its permissive rung.
- **A correct engine passes.** Every legitimate trade satisfies §6.2 item 2 at the price-setter's
  limit. The other leg's limit crosses that price, or the match would not have happened. For an ICE
  leg the most permissive rung only widens the bound. On sign-valid input it is the base
  (§4.2, `:209`): `test_i5_ice_buyer_with_negative_ppd_crosses_at_its_base` and its one-tick-above
  twin pin both directions.
- **The statement.** `STATEMENTS["I5"]` carries the Description's new sentence verbatim: *"Every trade
  is also matchable (§6.2 item 2): its price is no higher than the buyer's declared limit and no lower
  than the seller's, an ICE leg taking the most permissive rung of its declared ladder."*

**D3-03, CLOSED.**

- **The rule.** Matching Algorithm §5.1: *"An order with a new time stamp may be a newly entered order,
  a modified order, a (re)activated order that was inactive before, or a new slice of an iceberg
  order."* (`:462-464`). The builder does no modification or reactivation (V2), so inside submission
  *s* only a new iceberg slice may aggress against $o$'s remainder.
- **What the code does.** The hop branch is taken only when `passive_oid == o`, `aggressor_oid` is an
  accepted order with `exec_restriction == ICE`, and its declared side is opposite $o$'s. Every other
  `aggressor_oid != o` fill fails with `unruled_hop`, a passive other than $o$ included.
- **The v3.2 wrong engine now fails.** A NON order that aggresses against $o$ mid-submission is
  rejected: `test_i5_fails_a_hop_whose_aggressor_is_not_an_ice_order` (aggressor = $k$, a NON sell).
- **A correct engine passes.** An incoming ICE $o$ whose own slices aggress has `aggressor_oid == o`,
  so it takes the ordinary branch. A resting opposite iceberg that hops against $o$'s rested slice takes
  the hop branch. Both are legitimate and both pass.

**D3-04 to D3-06 (no return required; checked in passing).** The `unruled_hop` message now names §3.3,
Rules 2–3, Rule 4 and the sign-invalid ICEwPI case, as §16.2 asked. `STATEMENTS["I4"]` and the
Description carry the restated *Cannot see* text ("the queue key is the engine's own stored timestamp
…"). D2's I5 row (`scenario-ladder.md:182`) carries the resolved OQ-31 wording. C09's K2 note from
D3-04 (the wrong-sign draw lists `unruled_hop` as an expected, recorded outcome) belongs to C09's card,
and I did not check it here.

**V10, CONFORMS.**

- **Domain inputs unchanged.** The valuation constants are the ones §16.1 verified: $k = 0.1 \times
  0.01 \times 0.25 = 2.5\times10^{-4}$ EUR from §8.1, *"V (EUR) = Q (MW) × p (EUR/MWh) × d (h)"* with
  *"d = 0.25"* (`market-spec.md:436`), and $O = 999\,900$ from the §5.4 bounds. $V_s = k\,D_s$ is the
  same. Only the rounding envelope changed.
- **Direction of the change.** $c_s \le m_s$, so the leading term $c_s W_s$ is never looser than the
  old $m_s W_s$. The factor $(1 + (c_s + 8)u)$ differs from $(1 + 8u)$ only at second order. So the
  amendment tightens I6(b) whenever one dispatch writes several rows. It loosens nothing materially.
- **The failure mode is the safe one.** $c_s$ is read from distinct trade-row timestamps. If a future
  engine let two dispatches share a timestamp, $c_s$ would under-count, $\tau_s$ would shrink, and I6
  would fail a correct report loudly. It cannot pass a wrong one. That the chunks are the
  `extract_fills` dispatches, and that their message times strictly increase, is an engine fact
  (`contract_book.py`, `JaxOrderBookArrays.py`), outside domain authority. No rule says otherwise:
  §5.2's timestamp rules govern **order** priority, not the time stamped on a trade row.
- **The pinned example.** I recomputed it with the amended formula. For $p = 1\,004\,598$ and $q = 20$:
  $W = 20\,091\,960$, $O\,Q = 19\,998\,000$, $D = 93\,960$, and the bracket is $40\,559\,760$. Then
  $\tau = k\,u \times 40\,559\,760 \times (1 + 9u) = 6.043878\times10^{-4}$ EUR. The $(1+9u)$ factor does
  not move the seventh significant figure, so the value still matches the card.
- **The I6(a) bound still holds.** $W_s \ge 0$ (offset-encoded prices), so $c_s W_s \ge W_s$ for $c_s \ge
  1$. Then $W + O\,Q + 5\lvert D\rvert = 2\,O\,Q + D + 5\lvert D\rvert \ge 2\,O\,Q \ge 2\sum_i \lvert p_i -
  O\rvert q_i$, using $\lvert p_i - O\rvert \le O$ (the §5.4 bounds). This is the card's own argument
  (`agile-plan.md:2140`), and it is sound. A submission with no fills has $c_s = 0$, so $\tau_s = 0$ and
  $V_s = 0$. The report must then be exactly zero, which is the rule (no trade, no cash).
- **Economic size.** I checked one large case. At 50 EUR/MWh, 1 000 MW and $c_s = 10$, $\tau_s$ is about
  1.6 EUR against $V_s$ = 12 500 EUR, a relative $1.3\times10^{-4}$. That is the float32 settlement chain
  the engine actually runs, and I6 does not claim an independent valuation (`unobserved =
  ("independent_valuation",)`). Mispricing stays I5's and C02's job, as the I6 *Cannot see* says. No
  domain objection.

**V12 (M5), CONFORMS.** The map now runs over every fill of the aggressors' submissions. Its key is
`(counterparty, price_int)`: `passive_oid` on a plain fill, and the iceberg `aggressor_oid` at a hop.
This follows §4.2.1 Rule 3's *"one more slice inside the aggressor's single matching process"*
(`market-spec.md:244`). A split-versus-whole pair in which only one side is decomposed as a hop no
longer differs on a correct engine. Comparing the iceberg as counterparty at its slice price is also
the price OQ-31 Reading B fixes.

### 17.2 New item (v3.3)

#### D3-08 — Low — regenerated slices of a second iceberg are invisible to I4's depth map, on the plain walk and at the hop alike

**Rule.** §5.2: *"A **new timestamp** is assigned on: … activation of a new iceberg slice"*
(`market-spec.md:330`). Matching Algorithm §5: *"Orders with the same price limit are prioritized by
their time stamp oldest first."* (`:457`).

**Finding.**

- **The limitation.** I4's map holds each entry's pre-snapshot **visible** depth. When an iceberg's
  visible slice is used up in the map, the iceberg drops out as a competitor, even though its next
  slice, a new-timestamp entry, may still cross.
- **A wrong engine that passes.** Two sell icebergs, $j_1$ earlier than $j_2$, both peak 2 @ 50 with
  hidden quantity, and $o$ = BUY 10 @ 55.
  - The rules require $j_1$ 2, $j_2$ 2, $j_1$ 2, $j_2$ 2, $j_1$ 2. Each new slice queues behind the other
    iceberg's older one.
  - The engine prints $j_1$ 2, $j_2$ 2, $j_1$ 2, $j_1$ 2, $j_1$ 2. This could be plain fills, or a rest
    of $o$ followed by $j_1$ hops.
  - After $j_2$'s first slice, $j_2$ has depth 0 in the map, so neither the original walk nor
    `hop_priority` counts it. I3 passes if $o$ ends fully filled. I5 and `limit_cross` pass, since every
    price is 50.
- **Why this does not reopen D3-01.** D3-01 was a check on the wrong side, which made it vacuous for
  every hop. This is a narrower limit that the hop shares with the plain walk V8 already pinned: the
  map cannot order two icebergs' regenerated slices among themselves. That is close to the declared
  `unobserved = ("queue_timestamp_correctness",)`. The battery trusts the engine's queue for entries it
  does not hold in the pre-snapshot.

**Disposition.** Record as a residual. No battery change is required for C08 or C12. The fix belongs
at a scenario oracle, not in I4. Any scenario that draws two same-price icebergs against one
aggressor (C05's interloper family, C08's multi-slice draw, C09) should pin the exact counterparty
sequence in its Known result. The other option is to tighten I4 by tracking each iceberg's slices
through `iceberg_meta` and the per-dispatch trade timestamps. That is an architect's choice, and I do
not require it. If I4 is ever tightened that way, the change returns here (it is in V8's scope). No
new OQ: the rule is unambiguous.

### 17.3 Disposition and gate (v3.3)

| ID | Status after v3.3 |
|---|---|
| D3-01 | **CLOSED** (V8 `hop_priority`) |
| D3-02 | **CLOSED** (V9 `limit_cross`) |
| D3-03 | **CLOSED** (V9 ICE-only hop branch) |
| D3-04, D3-05, D3-06 | Applied. D3-04's C09 K2 note is still open on C09's card |
| D3-07 | Recorded (unchanged) |
| D3-08 | New, Low, recorded residual. Not a K2 or K5 precondition |
| V10 amendment | **CONFORMS** |
| V12 M5 amendment | **CONFORMS** |

**New OQ needed?** **No.** `market-spec.md` and `open-questions.md` are not edited.

**Gate.**

- **C12 K2 may proceed unconditionally.** I6 as amended conforms. No item open against I6.
- **C08 K2 may proceed unconditionally**, as far as D3 is concerned. The two v3.2 conditions are both
  discharged:
  1. D3-06 is done. D2's I5 row (`scenario-ladder.md:182`) has the resolved OQ-31 wording.
  2. The Rule-3 Known-result pin was required only to cover D3-01 and D3-02 at the oracle until the
     battery fixes merged. Both are now closed in the battery.

  C08's K2 must still follow V16 and OQ-31's Consequences on their own terms. Every `C08.rule3` trade
  prints at the slice price, and $L$ is drawn strictly through the slice price. That requirement comes
  from the OQ-31 ruling and is not a D3 condition. Pinning the exact counterparty sequence remains
  good practice for D3-08, but it is not required.
- **C08 K5.** The v3.2 requirement to close D3-01 before C08's K5 verdict is met.

**Re-review trigger.** Any further change to I4's queue model (D3-08), to I5 or to I6 returns here.

*Signed: Domain Analyst, v3.3, 2026-10-02.*
