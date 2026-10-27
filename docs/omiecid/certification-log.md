# OmieCID-Simulator — Certification Log

> **The certification run of OmieCID-Simulator 0.1.0.** This log records the certification of
> scenarios C00–C16 of the ladder ([`scenario-ladder.md`](scenario-ladder.md)) on the released
> package: one entry per scenario, written as the scenario completes. The ladder specification also
> describes a scenario C17 (replay fidelity of the public data); it is not part of this release.
>
> **How a scenario is judged.** Each scenario's notebook (`notebooks/certification/C<nn>_<slug>.ipynb`)
> is executed headless at its full draw count and writes `results/C<nn>.json`; the scenario's
> independent audit test (`tests/omiecid/certification/test_C<nn>_audit.py`) then checks that
> results file. The entry's result is `CERTIFIED` only when the notebook executed cleanly, its results
> status is `PASS` and the audit test passed. Otherwise the result is `NOT CERTIFIED`, the entry
> carries the one-paragraph explanation from the notebook's last section ("Why this certification
> failed"), and the Verdict line stays blank until a maintainer reviews it with one of `REPEAT` ·
> `DEFECT-FOUND` · `SCOPE-CHANGE` · `INVALID`. Nothing is re-authored to force a pass.
>
> **Fields of this run.** Entries are written by `notebooks.certification.cert_support.close_scenario`.
> In this release run the **Card** field carries the scenario id, the **Branch** field the package
> version (`omiecid-simulator 0.1.0`) and the **Commit** field the blake2b-128 digest of the
> canonical JSON of the scenario's results payload (`cert_support.results_digest`), the same digest
> the certificate signs. The results files themselves will be published with the release, with
> their blake2b-256 file digests (`DIGESTS.b2`), not committed.
>
> **Append-only.** A past entry is never edited or deleted; a correction is a new entry. The one
> exception is a maintainer filling in a blank Verdict line.
>
> - Specification: [`scenario-ladder.md`](scenario-ladder.md)
> - Certificate: [`certificate.md`](certificate.md)

---

## Table of contents (the five-minute read)

| # | Date | Scenario | Draws | Result | Verdict | One-line note |
|---|---|---|---|---|---|---|
| 1 | 2026-10-26 | `C00` — Inertness and the contract-status gate | 34 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 2 | 2026-10-26 | `C01` — Order validation and rejection | 500 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 3 | 2026-10-26 | `C02` — The single crossing pair | 500 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 4 | 2026-10-26 | `C03` — Price-time priority, ties, and ownership blindness | 1100 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 5 | 2026-10-26 | `C04` — Cancellation | 1000 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 6 | 2026-10-26 | `C05` — Partial fill and the resting residual | 1000 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 7 | 2026-10-26 | `C06` — Multi-level walk and volume-weighted price | 504 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 8 | 2026-10-26 | `C07` — Immediate-or-cancel | 500 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 9 | 2026-10-26 | `C08` — Iceberg regeneration | 1250 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 10 | 2026-10-26 | `C09` — Price-incremented icebergs | 520 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 11 | 2026-10-26 | `C10` — Fill-or-kill, in three parts | 2000 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 12 | 2026-10-26 | `C11` — Expiry at gate closure | 500 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 13 | 2026-10-26 | `C12` — Settlement and the cash leg | 500 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 14 | 2026-10-26 | `C13` — Anonymity and information visibility | 500 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 15 | 2026-10-26 | `C14` — Capacity edges, one part per ceiling | 30 | NOT CERTIFIED | REPEAT | K5 failure; evidence merged; re-run at K2 |
| 16 | 2026-10-26 | `C15` — Randomised fuzz across the parameter space | 2200 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 17 | 2026-10-26 | `C16` — Multi-contract isolation | 500 | NOT CERTIFIED | REPEAT | oracle_mismatch; evidence merged; re-run at K2 |
| 18 | 2026-10-27 | `C14` — Capacity edges, one part per ceiling | 30 | NOT CERTIFIED | REPEAT | K5 failure; evidence merged |
| 19 | 2026-10-27 | `C16` — Multi-contract isolation | 500 | CERTIFIED | CERTIFIED | auto-certified at K6 |
| 20 | 2026-10-27 | `C14` — Capacity edges, one part per ceiling | 30 | CERTIFIED | CERTIFIED | auto-certified at K6 |

**Awaiting the human (K7-U):** *(none yet)* — every `NOT CERTIFIED` entry whose Verdict line is blank
is listed here.

---

## Entry template (copy verbatim; fixed order)

```markdown
### <N> · <YYYY-MM-DD> · `C<nn>` — <scenario name>

- **Card:** `C<nn>` · **Branch:** `<package version>` · **Commit:** `<results blake2b-128 digest>`
- **Notebook:** `notebooks/certification/C<nn>_<slug>.ipynb` · **Results:** `results/C<nn>.json`
- **Certifies:** <market-spec sections, as the notebook quotes them, with the spec version>
- **Dataset:** <recipe + draw count + seed range + generator digest>
- **Known result:** <the expectation, in one line — the formula, not the value>
- **Outcome:** <assertions passed/failed; invariants passed/failed/unobserved; discard rate>
- **Capacity peaks:** <resting/side, trades, hops, icebergs — against the 80/80/160/800 envelope>
- **Eager vs jitted:** <identical / diverged, with the divergence if any>
- **Wall clock:** <s> (forecast <s>)
- **Anything the run revealed that the card did not anticipate:** <one line, or "nothing">
- **Result:** CERTIFIED | CERTIFIED — provisional on C<nn> | NOT CERTIFIED
- **Why it failed** (NOT CERTIFIED only): <one paragraph, copied from notebook section 9 — what failed,
  on which seed, expected vs observed; the smallest failing case (fixture path); the rule it concerns;
  where the fault most plausibly lies, written as a hypothesis with its evidence; what it affects
  downstream. If the notebook crashed before section 9: "explanation missing — notebook crashed;
  traceback: <first and last lines>">
- **Merged:** <branch merged | evidence merged, certificate section unsigned>
- **Verdict:** ______ · **rationale:** ______
```

**Filling the Verdict line.**

- **On `CERTIFIED`:** the verdict is `CERTIFIED` too. Everything above it is evidence, and the line
  records that the evidence held.
- **On `NOT CERTIFIED`:** the line stays blank until a maintainer writes one of the four verdicts,
  with a one-line rationale.

---

## Entries

### 1 · 2026-10-26 · `C00` — Inertness and the contract-status gate

- **Card:** `C00` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `28a3833bcea0b9c8a87a2a25c318a809`
- **Notebook:** `notebooks/certification/C00_inertness_and_status.ipynb` · **Results:** `results/C00.json`
- **Certifies:** §3.1 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §3.3 (market-spec.md v2.14); §3.3 (market-spec.md v2.14); §3.3 (market-spec.md v2.14); §3.3 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); R-1 (scenario-ladder-review.md v3.3); §3.1 (market-spec.md v2.14); §3.3 (market-spec.md v2.14)
- **Dataset:** C00.empty, C00.pad_only, C00.status_gate; 34 valid of 34 requested draws; seeds 0–7; generator digest C00.empty 14fa8d8dad852966, C00.pad_only 8711027ae7671c3d, C00.status_gate 5b6c29288a91368f
- **Known result:** zero trades; state == fresh init leaf for leaf; every submission rejected; n_active = 0; no flags raised
- **Outcome:** 380/380 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 1, trades 1, hops unobserved, concurrent_icebergs 0 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 54.051 (forecast 40.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 2 · 2026-10-26 · `C01` — Order validation and rejection

- **Card:** `C01` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `6c958a8d4735a67e422effb13d1d8d25`
- **Notebook:** `notebooks/certification/C01_order_validation.ipynb` · **Results:** `results/C01.json`
- **Certifies:** §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §10 (scenario-ladder-review.md v3.3); §5.4 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §2 (scenario-ladder-review.md v3.3); OQ-16 (open-questions.md OQ-16 · What is known); §2.3 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §2 (scenario-ladder-review.md v3.3); §5.4 (market-spec.md v2.14)
- **Dataset:** C01.validation; 500 valid of 500 requested draws; seeds 0–499; generator digest C01.validation 2b7bd6c5239d06f0
- **Known result:** accept iff 0 <= n_p <= 1,999,800 and 1 <= n_q <= 9,990 and (ICE => 0 < n_k <= n_q and PPD sign valid: BUY n_d <= 0, SELL n_d >= 0), n = nint after the tick snap; on accept the order rests at n_p with visible n_q (NON) or n_k (ICE); on reject a byte-identical book
- **Outcome:** 182/182 oracle assertions passed; invariants 7 pass / 0 fail / 3 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 13, trades 0, hops unobserved, concurrent_icebergs 3 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 67.524 (forecast 180.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 3 · 2026-10-26 · `C02` — The single crossing pair

- **Card:** `C02` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `f2200c780852bee04849c6dca540790b`
- **Notebook:** `notebooks/certification/C02_single_crossing_pair.ipynb` · **Results:** `results/C02.json`
- **Certifies:** §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §11.1 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14)
- **Dataset:** C02.crossing_pair; 500 valid of 500 requested draws; seeds 0–499; generator digest C02.crossing_pair b602b93502dc6ed1
- **Known result:** exactly one trade; price_int = p_rest (the resting order's limit, never the aggressor's); qty_int = min(q_rest, q_agg); passive leg = the resting order
- **Outcome:** 2150/2150 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 1, trades 1, hops unobserved, concurrent_icebergs 0 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 21.051 (forecast 60.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 4 · 2026-10-26 · `C03` — Price-time priority, ties, and ownership blindness

- **Card:** `C03` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `d53a623cc7f6d0e6ca39f11d92d94ff3`
- **Notebook:** `notebooks/certification/C03_priority_ties_and_ownership.ipynb` · **Results:** `results/C03.json`
- **Certifies:** §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §11.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §13.1 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14)
- **Dataset:** C03.priority_ladder, C03.price_over_time, C03.same_owner_pair; 1100 valid of 1100 requested draws; seeds 0–499; generator digest C03.priority_ladder 321b9c63c73962b2, C03.price_over_time 5ab6a9465f8a1a95, C03.same_owner_pair 80a4e209455814fd
- **Known result:** the filled set is exactly the j earliest touch order ids, filled in entry order at the touch price (exact counterparty sequence); decoys and the other k - j touch orders untouched; the stored timestamps of the resting orders unique and strictly increasing in entry order, also among orders that enter in the same second, at one price and across prices (r2); the same-owner pair trades, with a trade list identical to the distinct-owner counterfactual (M6); an entry-order reversal moves the filled set with it (M1)
- **Outcome:** 14851/14851 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 9, trades 5, hops unobserved, concurrent_icebergs 0 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 64.091 (forecast 180.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 5 · 2026-10-26 · `C04` — Cancellation

- **Card:** `C04` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `7b9305f936e350f0d24f218700b9306e`
- **Notebook:** `notebooks/certification/C04_cancellation.ipynb` · **Results:** `results/C04.json`
- **Certifies:** §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §9.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5.3 (market-spec.md v2.14); §5.3 (market-spec.md v2.14); OQ-20 (open-questions.md OQ-20 · Why the test is still load-bearing); §2 (scenario-ladder-review.md v3.3)
- **Dataset:** C04.cancellation, C04.iceberg_cancel_fok; 1000 valid of 1000 requested draws; seeds 0–499; generator digest C04.cancellation 2872106e0b3b7ffe, C04.iceberg_cancel_fok 6865e069b261cf12
- **Known result:** the cancelled order is gone (no visible or hidden quantity, one 'cancellation' fate of its full size) and nothing else in the book changes; the cancellation prints no trade; the survivors keep their stamps and queue, so the fill set is the j earliest remaining originals, filled in entry order at the touch price; after cancelling an iceberg its hidden volume is invisible to the FOK pre-check, so a FOK only the iceberg could have satisfied fills nothing and leaves the book unchanged (the no-cancel control fills R exactly)
- **Outcome:** 18468/18468 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 8, trades 4, hops unobserved, concurrent_icebergs 1 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 61.493 (forecast 120.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 6 · 2026-10-26 · `C05` — Partial fill and the resting residual

- **Card:** `C05` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `d779bb22bd3e51b112b77481b2769dc3`
- **Notebook:** `notebooks/certification/C05_partial_fill_residual.ipynb` · **Results:** `results/C05.json`
- **Certifies:** §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §8 (scenario-ladder-review.md v3.3); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14)
- **Dataset:** C05.aggressor_residual, C05.resting_residual; 1000 valid of 1000 requested draws; seeds 0–1000499; generator digest C05.aggressor_residual e190d2197ffc6572, C05.resting_residual e0f5b332cd1588e8
- **Known result:** (i) filled = crossing depth q_r; the residual q_a - q_r rests at the aggressor's own limit p_a with its original entry timestamp (equal to its stamp when it rests unmatched, unchanged until the take, earlier than the interloper's); (ii) the resting order filled q_a keeps q_r - q_a at p_r with its stamp unchanged; in both, the take fills the residual's owner for the whole residual at its limit and the interloper stays whole, also when the interloper shares the tape second; I2 closes
- **Outcome:** 28009/28009 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 2, trades 2, hops unobserved, concurrent_icebergs 0 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 38.74 (forecast 60.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 7 · 2026-10-26 · `C06` — Multi-level walk and volume-weighted price

- **Card:** `C06` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `0ade9998fd2a7353e59fa87086b5b65d`
- **Notebook:** `notebooks/certification/C06_multi_level_walk_vwap.ipynb` · **Results:** `results/C06.json`
- **Certifies:** §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); OQ-18 (open-questions.md OQ-18 · Context.); §5 item 2 (scenario-ladder-review.md v3.3)
- **Dataset:** C06.multi_level_walk, C06.multi_level_walk_stress; 504 valid of 504 requested draws; seeds 0–1000003; generator digest C06.multi_level_walk c08f8ddfa4b98ce3, C06.multi_level_walk_stress 4d925ad8cccf7fe5
- **Known result:** the aggressor walks levels 0..m in rank order, one trade per level at that level's own limit (levels 0..m-1 whole, level m for r), never beyond; exact per-level fill vector (q_0..q_{m-1}, r, 0..0); [total_qty_int, weighted_price_int, num_fills] = [sum q_i + r, sum p_i q_i + p_m r, m + 1[r>0]] in exact integers before any division; level m keeps q_m - r, deeper levels whole, no aggressor residual; ENGINEERING INVARIANT: the engine's report quantity equals Q and its price lies in the fill-price bracket +-1 tick, including at the ~1.9e10 single-product stress draw
- **Outcome:** 3042/3042 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 12, trades 12, hops unobserved, concurrent_icebergs 0 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 28.022 (forecast 120.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 8 · 2026-10-26 · `C07` — Immediate-or-cancel

- **Card:** `C07` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `a8547bc3774f338e17d1425be5ba4f06`
- **Notebook:** `notebooks/certification/C07_ioc.ipynb` · **Results:** `results/C07.json`
- **Certifies:** §1 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §6.2 item 5 (market-spec.md v2.14); §16.1 (scenario-ladder-review.md v3.3)
- **Dataset:** C07.ioc_regimes; 500 valid of 500 requested draws; seeds 0–499; generator digest C07.ioc_regimes 25a7a757e402403a
- **Known result:** IOC and NON with identical price and quantity produce identical trade lists, field for field (M3): [[o_r, o_a, p_r, f, 1]] if f > 0, with f = 0 (none), q_r (partial), q_a (full); the IOC's own id never rests and its fate is [filled, resting, hidden, evicted, cancelled] = [f, 0, 0, 0, q_a - f] (status CANCELLED when unmatched, remainder CANCELLED when partial, EXECUTED when full; status itself unobserved); the NON's leftover q_a - f rests at p_a; the maker keeps q_r - f in both runs; every order accepted
- **Outcome:** 4505/4505 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 1, trades 1, hops unobserved, concurrent_icebergs 0 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 20.616 (forecast 120.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 9 · 2026-10-26 · `C08` — Iceberg regeneration

- **Card:** `C08` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `3c0b0891729194a0c0c276df58280a46`
- **Notebook:** `notebooks/certification/C08_iceberg_regeneration.ipynb` · **Results:** `results/C08.json`
- **Certifies:** §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §6.2 item 7 (market-spec.md v2.14); OQ-31 (open-questions.md OQ-31 · Resolution (2026-10-02); OQ-31 (open-questions.md OQ-31 · Consequences.); OQ-33 (open-questions.md OQ-33 · Revised by user ruling (2026-10-20); OQ-33 (open-questions.md OQ-33 · Revised by user ruling (2026-10-20); OQ-33 (open-questions.md OQ-33 · Revised by user ruling (2026-10-20); OQ-33 (open-questions.md OQ-33 · Revised by user ruling (2026-10-20); OQ-34 (open-questions.md OQ-34 · Resolved by user ruling (2026-10-20); OQ-34 (open-questions.md OQ-34 · Resolved by user ruling (2026-10-20); OQ-20 (open-questions.md OQ-20 · Ruling.); OQ-23 (open-questions.md OQ-23 · Q2 — Rule-4 verdict); §17.3 (scenario-ladder-review.md v3.3); D3-08 (scenario-ladder-review.md v3.3)
- **Dataset:** C08.iceberg_cascade, C08.rule3, C08.incoming_iceberg; 1250 valid of 1250 requested draws; seeds 0–3000249; generator digest C08.iceberg_cascade 5ffb5c540eb151c3, C08.rule3 710ca239fd81a876, C08.incoming_iceberg 0f3ed6a65eacb0ec
- **Known result:** cascade: one aggressor of size total consumes all n = ceil(total/peak) slices of a resting iceberg in its own submission, trades [buyer, seller, p_ice, x_k, 1] with sizes peak,...,peak,r; Rule-3: an aggressor peak < q_agg <= total - peak, limit L strictly through p_ice, trades peak x floor(q_agg/peak) then q_agg mod peak, every one at p_ice (OQ-31 Reading B), never at L; the iceberg keeps total - q_agg and its resting slice carries a later timestamp; incoming: an iceberg walking a resting ladder aggresses with its total as a chain of size-peak instances, one trade per (counter bid, instance) pair: rows split at resting entries and at its own instance boundaries (every row <= peak, every multiple of peak below M a cumulative row boundary), each at the level's limit, filled total; incoming remainder: the same iceberg against levels of total depth M < total, M mod peak != 0, sweeps every level and rests R = total - M as instance k_e = floor(M/peak) at p_ice showing v_1 = min(peak - u, R), u = M - k_e*peak, hidden R - v_1 (v2.14 Rule 2 Incoming iceberg items (2), (4), amended 2026-10-20; OQ-33 revised, OQ-34 resolved, display C3); every report equals the rows its submission wrote (engineering invariant, I6(b)); book uncrossed at every return (I3); hop bound <= 100 outer matching passes; no capacity flag (truncated = False) inside the margin; no iceberg_meta claim
- **Outcome:** 1262/1262 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 8, trades 39, hops unobserved, concurrent_icebergs 1 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 41.659 (forecast 480.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 10 · 2026-10-26 · `C09` — Price-incremented icebergs

- **Card:** `C09` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `7bc9a7dddcb4b2e79a365547aed6627d`
- **Notebook:** `notebooks/certification/C09_icewpi_price_drift.ipynb` · **Results:** `results/C09.json`
- **Certifies:** §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §6.2 item 7 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); OQ-23 (open-questions.md OQ-23 · **(a) The PRIMARY ICEwPI-SELL probe); OQ-20 (open-questions.md OQ-20 · Ruling.); §16.1 (scenario-ladder-review.md v3.3); D3-08 (scenario-ladder-review.md v3.3)
- **Dataset:** C09.icewpi_drift, C09.icewpi_wrong_sign; 520 valid of 520 requested draws; seeds 0–1000019; generator digest C09.icewpi_drift d4052f451c6d1001, C09.icewpi_wrong_sign 27cbbdedf88e437c
- **Known result:** a resting ICEwPI iceberg (sign-valid ppd != 0: BUY < 0, SELL > 0) and one NON aggressor whose limit L crosses exactly the first m slices and whose quantity exceeds them: trades [buyer, seller, base + k*ppd, peak, 1] for k = 0..m-1, so the exact price sequence base, base+ppd, ..., base+(m-1)ppd and the stop index m; the aggressor's residual rests at L; the iceberg keeps total - m*peak with positive hidden quantity, its slice m shown at exactly base + m*ppd or not yet shown (OQ-20); the iceberg sets every price (§6.2 item 7). One iceberg, no competing level. Every wrong-sign draw (BUY ppd > 0, SELL ppd < 0) is REJECTED (§5.4 v2.14; §13.6 sign rule): no trade, the book unchanged by its submission, and the aggressor resting whole at L
- **Outcome:** 537/537 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 1, trades 17, hops unobserved, concurrent_icebergs 1 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 20.806 (forecast 300.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 11 · 2026-10-26 · `C10` — Fill-or-kill, in three parts

- **Card:** `C10` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `69257a4285a32b797492d02137fd1a64`
- **Notebook:** `notebooks/certification/C10_fok_and_hidden_volume.ipynb` · **Results:** `results/C10.json`
- **Certifies:** §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §6.2 item 7 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14); OQ-11 (open-questions.md OQ-11 · Two interpretations considered:); OQ-11 (open-questions.md OQ-11 · Two interpretations considered:); OQ-11 (open-questions.md OQ-11 · Resolution:); OQ-11 (open-questions.md OQ-11 · Resolution:); OQ-20 (open-questions.md OQ-20 · Ruling.); D3-08 (scenario-ladder-review.md v3.3)
- **Dataset:** C10.fok_visible, C10.fok_hidden, C10.fok_icewpi; 2000 valid of 2000 requested draws; seeds 0–3000499; generator digest C10.fok_visible df6113a825a62449, C10.fok_hidden 17be65aa0f3889d4, C10.fok_icewpi 4e46267d87d5b907
- **Known result:** FOK pre-check volume M from the book on the other side (visible orders crossing the limit; every iceberg slice whose drifted price crosses it: §4.2.1 Rule 1, OQ-11 B); executed iff req <= M, filling exactly req at each maker's own (slice) price, uncapped; else killed: no trade, the book before == the book after, the whole req cancelled as fok_kill (M4). C10a: M = V, runs V-1, V, V+1. C10b: M = total (hidden quantity counts), runs req, total, total+1. C10c: M = min(total, peak * #{k >= 0: base + k*ppd crosses the limit}), runs M, M+1. C10d (iceberg + a worse visible level): M = min(total, peak * r) + q_v, runs T_c, T_c+1, M, M+1; fills in rank order (better price first; a slice tying the visible price after it), each piece at its maker's price
- **Outcome:** 2016/2016 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 5, trades 14, hops unobserved, concurrent_icebergs 1 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 65.571 (forecast 560.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 12 · 2026-10-26 · `C11` — Expiry at gate closure

- **Card:** `C11` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `c4fdca867994b353aca37683c51ac003`
- **Notebook:** `notebooks/certification/C11_gct_expiry.ipynb` · **Results:** `results/C11.json`
- **Certifies:** §3.1 (market-spec.md v2.14); §4.6 (market-spec.md v2.14); §4.6 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.1 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); V3-16 (scenario-ladder-review.md v3.3); V3-16 (scenario-ladder-review.md v3.3); V3-16 (scenario-ladder-review.md v3.3); OQ-20 (open-questions.md OQ-20 · Ruling.); §4.2.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); V3-01 (scenario-ladder-review.md v3.3); §8.5 (market-spec.md v2.14); §5.4 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); D3-08 (scenario-ladder-review.md v3.3); §13.6 (market-spec.md v2.14); §13.6 (market-spec.md v2.14)
- **Dataset:** no registered construction; 500 valid of 500 requested draws; seeds 0–521; generator digest no construction digest
- **Known result:** At the gate every resting order is removed, GFS and GTD alike, icebergs with their hidden remainder: the final book is empty on both sides and no order keeps visible or hidden quantity; each order's whole remaining quantity at the gate (declared - filled = visible + hidden) is cancelled as gate_expiry and nothing else is cancelled or evicted; the sweep writes no trade (the trade ring is byte-identical across the expiry call, with the same row count, so the pre-gate trades stay firm, §8.5); I2 exact: filled + gate_expiry = declared, per order
- **Outcome:** 518/518 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.042
- **Capacity peaks:** resting_per_side 8, trades 2, hops unobserved, concurrent_icebergs 13 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 27.999 (forecast 180.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 13 · 2026-10-26 · `C12` — Settlement and the cash leg

- **Card:** `C12` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `47e2891739352d8b532762edabdd7bb2`
- **Notebook:** `notebooks/certification/C12_settlement_cash_leg.ipynb` · **Results:** `results/C12.json`
- **Certifies:** §8.1 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §10 (market-spec.md v2.14); §11.1 (market-spec.md v2.14); §8.4 (market-spec.md v2.14); §8.4 (market-spec.md v2.14); §8.4 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); V3-03 (scenario-ladder-review.md v3.3); V3-03 (scenario-ladder-review.md v3.3); V3-04 (scenario-ladder-review.md v3.3); OQ-18 (open-questions.md OQ-18 · Context.); OQ-18 (open-questions.md OQ-18 · Option 3a adjudication); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §6.2 item 5 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §5.1 (market-spec.md v2.14); D3-08 (scenario-ladder-review.md v3.3); §8.3 (market-spec.md v2.14); §8.3 (market-spec.md v2.14)
- **Dataset:** no registered construction; 500 valid of 500 requested draws; seeds 0–499; generator digest no construction digest
- **Known result:** Every trade row pinned by the walk (counterparties included); each trade's value D_i = q (p_int - O) recomputed exactly in ticks from its own row (V = Q p d, d = 0.25 h, k = 2.5e-4 EUR per tick-tick); each participant's engine settlement within tau_a of k D_a, D_a = sells - buys (§8.4 sign convention), from an (M, 6) settlement tape, for exactly the participants on trade rows; each trading submission's reported settlement within tau_s of k D_s and its reported quantity exact; gross sum |D_i| > 0; I6(a) and (b) pass; the integer zero-sum is algebra and not asserted
- **Outcome:** 1026/1026 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 17, trades 22, hops unobserved, concurrent_icebergs 0 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 43.637 (forecast 180.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 14 · 2026-10-26 · `C13` — Anonymity and information visibility

- **Card:** `C13` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `66145d154f9c55179da7df53b2115299`
- **Notebook:** `notebooks/certification/C13_anonymity.ipynb` · **Results:** `results/C13.json`
- **Certifies:** §7.1 (market-spec.md v2.14); §7.2 (market-spec.md v2.14); §7.2 (market-spec.md v2.14); §7.2 (market-spec.md v2.14); §7.3 (market-spec.md v2.14); §7.3 (market-spec.md v2.14); §7.3 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §9.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); OQ-34 (open-questions.md OQ-34 · **Resolved by user ruling (2026-10-20); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §9.1 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.6 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §6.2 item 5 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); OQ-07 (open-questions.md OQ-07 · **Modeling choice for simulator:**); §2 (scenario-ladder-review.md v3.3); D3-08 (scenario-ladder-review.md v3.3)
- **Dataset:** no registered construction; 500 valid of 500 requested draws; seeds 0–499; generator digest no construction digest
- **Known result:** The Session participant view (top_of_book and get_anonymous_lob after reset and every step, the FillReport and done flag of every step, own trades, own orders, cash and inventory at the end) is bit-identical between a world with foreign resting icebergs and its twin with every foreign hidden remainder removed and every identifier permuted (noop and random AgentOrder policy) where no tape iceberg fills a resting participant order on entry, and where one does, the rule-implied display of section 4.2.1 Rule 2 (1), (4) as amended 2026-10-20 (remainder as instance k_e = floor(M/P) showing v1 = min(P - u, R); here k_e = 0, u = f, so A shows P - f like B, hidden h, and the twin stays bit-identical on every step), while a one-tick change of the visible touch is seen by top_of_book; the participant's own-orders reader shows its own icebergs at full quantity, the anonymous book only their visible slice; get_agent_trades(a) returns exactly a's legs [price_int, qty_int], byte-identical when every identifier but a's is permuted, and the union over agents is every trade once per side; each iceberg rests under its owner with its full remaining quantity held; both engine worlds' trades pinned by the walk
- **Outcome:** 5035/5035 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 11, trades 7, hops unobserved, concurrent_icebergs 7 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 675.832 (forecast 180.0)
- **Anything the run revealed that the card did not anticipate:** the audit test's reader did not accept the field names the notebook records for the participant's end-of-episode own trades, cash and inventory (`own_report`, `cash`, `inventory`), so 25 of its 97 checks failed on a missing key; the reader was widened to accept those names, with no expectation changed, and the audit test then passed 97/97
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 15 · 2026-10-26 · `C14` — Capacity edges, one part per ceiling

- **Card:** `C14` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `916c3775d976f6521ba4a989c225623b`
- **Notebook:** `notebooks/certification/C14_capacity_edges.ipynb` · **Results:** `results/C14.json`
- **Certifies:** OQ-10 (open-questions.md OQ-10 · **Decision impact:**); §5 (scenario-ladder-review.md v3.3); §5 item 3 (scenario-ladder-review.md v3.3); V3-14 (scenario-ladder-review.md v3.3); OQ-15 (open-questions.md OQ-15 · **What is unknown (engineering invariant):**); OQ-15 (open-questions.md OQ-15 · **Consequence for Phase 2 (observed):**); §11.1 (scenario-ladder-review.md v3.3); §5 item 2 (scenario-ladder-review.md v3.3); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5 item 1 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); V3-15 (scenario-ladder-review.md v3.3); §4.2 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); OQ-20 (open-questions.md OQ-20 · **Ruling.**); §4.2.1 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); V3-15 (scenario-ladder-review.md v3.3); V3-15 (scenario-ladder-review.md v3.3)
- **Dataset:** no registered construction; 30 valid of 30 requested draws; seeds 0–900002; generator digest no construction digest
- **Known result:** C14a: the 101st resting add raises book_full and wipes every worst-price order into the evicted bucket, no trade; C14a′ (recorded, not certified; rebuilt r2, full only at regeneration): no flag site meets a full side, the regenerations replace their own consumed rows, no flag and nothing evicted or lost; C14b: after 100 trades the ring keeps only a burst's final trade, in its last row, detectably (saturation flag), while the engine reports the true burst quantity (I6 and I2 violated exactly on the burst); C14c: an aggressor needing > 100 passes (hop bound inside the 160 margin) raises truncated, the loop terminates, I2 closes, the full outcome consistent with its fills; C14c-below: hop bound < 100, no truncation, the exact walk; C14d: an iceberg beyond the slot cap raises truncated at registration, leaves the side channel unchanged, its hidden remainder untracked, the survivors intact; C14e: 65 slices leave truncated false, 66 set it at registration with no trade; S2 controls: with the last rows occupied, a crossing NON on a full own side loses only the documented worst-level wipe, and an iceberg regenerates and is reached; excluded invariants violated exactly as stated
- **Outcome:** 165/165 oracle assertions passed; invariants 9 pass / 0 fail / 1 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 100, trades 104, hops unobserved, concurrent_icebergs 4 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 85.279 (forecast 900.0)
- **Anything the run revealed that the card did not anticipate:** the audit test's citation check depends on research documents and a source comment that the release does not ship unchanged
- **Result:** NOT CERTIFIED
- **Why it failed:** The notebook passed; its audit test failed: test_C14_audit.py::TestEnvelope::test_every_citation_fragment_is_in_its_named_source failed (48 of 49 checks passed): 8 of the notebook's 13 display-only citations (not quote-checked) do not resolve in the released repository: 7 name research documents that are not part of the release, and 1 quotes a comment in omiecid_simulator/omiecid/omie_config.py whose wording changed in the release (the cited 'nTrades' comment). Every engine check, the quote check and the invariant battery passed, and the results match the earlier sealed certification run on status, assertions, invariants, generator digests and quote digests.
- **Merged:** evidence merged, certificate section unsigned
- **Verdict:** REPEAT · **rationale:** the maintainer, 2026-10-27: re-enter at K2 — re-point or drop the display-only citations that name documents not part of the release, and align the quoted `nTrades` comment with the release wording; then re-run.

### 16 · 2026-10-26 · `C15` — Randomised fuzz across the parameter space

- **Card:** `C15` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `f3016bac4479ed878ca56353d842afad`
- **Notebook:** `notebooks/certification/C15_randomised_fuzz.ipynb` · **Results:** `results/C15.json`
- **Certifies:** §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 1 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); §6.2 item 4 (market-spec.md v2.14); §6.2 item 5 (market-spec.md v2.14); §6.2 item 6 (market-spec.md v2.14); §6.2 item 7 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §6.6 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §13.6 item 1 (market-spec.md v2.14); OQ-11 (open-questions.md OQ-11 · **Resolution:**); OQ-20 (open-questions.md OQ-20 · **CORRECTION NOTE (2026-10-07); OQ-20 (open-questions.md OQ-20 · **ANSWER TO CORRECTION-NOTE ITEM (d)); OQ-32 (open-questions.md OQ-32 · **SCOPE CORRECTION (2026-10-08); OQ-33 (open-questions.md OQ-33 · **Revised by user ruling (2026-10-20); OQ-33 (open-questions.md OQ-33 · **Revised by user ruling (2026-10-20); OQ-34 (open-questions.md OQ-34 · **Resolved by user ruling (2026-10-20); OQ-34 (open-questions.md OQ-34 · **Resolved by user ruling (2026-10-20); OQ-34 (open-questions.md OQ-34 · **Resolved by user ruling (2026-10-20); OQ-35 (open-questions.md OQ-35 · **The question.); §2 (scenario-ladder-review.md v3.3); §5 item 2 (scenario-ladder-review.md v3.3); §5 item 3 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); D3-08 (scenario-ladder-review.md v3.3); D3-08 (scenario-ladder-review.md v3.3)
- **Dataset:** C15.random_session; 2200 valid of 2200 requested draws; seeds 0–100001999; generator digest C15.random_session df4d104b115333a5
- **Known result:** zero exceptions across every draw: RB and I1-I9 never fail (not_applicable is allowed; I8 invalid discards); the engine's trade sequence, cancellations and remaining quantities equal the independent price-time reference walk W (strict: a slice is activated eagerly, queued at its activation event and shown in the book after every submission, F6; an incoming iceberg sweeps its total quantity as a chain of instances of size P, instance k only where its limit b + k d crosses, stopping at the first instance that cannot trade, one row per (resting entry, incoming instance), the remainder resting as instance k_e = floor(M/P) at b + k_e d showing min(P - u, R), its ladder continued; v2.14 §4.2.1 Rule 2 amended 2026-10-20, OQ-33, OQ-34); M1-M6 hold on their variants, which also pass the battery and W; eager == jitted on the eager draws; plus a pre-registered closed-form band on aggregate matched volume, reported and never a pass criterion
- **Outcome:** 478/478 oracle assertions passed; invariants 16 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 80, trades 72, hops unobserved, concurrent_icebergs 54 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 578.683 (forecast 840.0)
- **Anything the run revealed that the card did not anticipate:** nothing
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 17 · 2026-10-26 · `C16` — Multi-contract isolation

- **Card:** `C16` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `aa15e109d1b9620124c6af5d1bcc08d2`
- **Notebook:** `notebooks/certification/C16_multi_contract_isolation.ipynb` · **Results:** `results/C16.json`
- **Certifies:** §2.2 (market-spec.md v2.14); §6.4 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §2.1 (market-spec.md v2.14); §2.1 (market-spec.md v2.14); §2.2 (market-spec.md v2.14); §2.2 (market-spec.md v2.14); §3.1 (market-spec.md v2.14); §3.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §13.6 item 1 (market-spec.md v2.14); §3 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3)
- **Dataset:** C15.random_session; 500 valid of 500 requested draws; seeds 0–499; generator digest C15.random_session df4d104b115333a5
- **Known result:** for every contract of every draw, its lane in a stacked multi-contract environment run ends with exactly the outcome of its isolated engine run (trade rows except timestamps, resting book per side, hidden remainder per iceberg, book_full/truncated); no lane trades a foreign order id; the integer trade value W_c is identical (d = 0.25 h); the contract alone in the environment agrees with both; the isolated runs pass the battery
- **Outcome:** 16/19 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 28, trades 61, hops unobserved, concurrent_icebergs 18 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 7441.072 (forecast 240.0)
- **Anything the run revealed that the card did not anticipate:** the stacked run's explicit horizon (gct_minutes_from_got) does not keep the lane holding the stack's largest timestamp from ending one slice early under the span_scaled clock
- **Result:** NOT CERTIFIED
- **Why it failed:** C16 NOT CERTIFIED (oracle_mismatch). What failed: 3 assertion(s) failed, first 'seed 235: contract 95 in a pooled stack == its isolated engine run' on seed 235: expected {'trade_digest': 'd39cd7d6cd', 'n_trades': 5}, observed {'trade_digest': '88deba6710', 'n_trades': 4, 'diff': {'trades': {'n_expected': 5, 'n_observed': 4, 'rows': [[4, [1005282, 39, 12394006, 12394032, 1, 1], None]]}, 'bids': {'missing': [[1005282, 111, 12394006, 1]], 'extra': [[1005282, 150, 12394006, 1]]}}, 'foreign_oids': {}, 'where': {'pooled_stack': 1, 'lane': 583, 'n_lanes': 584, 'orders_per_step': 3, 'stack_shape': [584, 64, 13]}, 'trade_columns': ['price_int', 'qty_signed', 'passive_oid', 'aggressor_oid', 'passive_tid', 'aggressor_tid']} (other). Smallest case: seed 235 (152 rows), frozen as C16_seed235.npz. Rule: §2.2 (own independent order book), §6.4 (never matched against each other): the strong form. Fault (hypothesis): The notebook's stacked-run configuration, not matching or isolation. The failing lane (contract 95 of seed 235, the last lane of pooled stack 1) holds the stack's largest timestamp, 110395 s, and its second-to-last slice of three orders (rows 27-29) already carries 110395. Under the session's default span_scaled clock a slice's time is round(m_k * gct / S), with S the stack's largest timestamp, so that slice reaches the gate-closure time and ends the session; the session ignores every later call, and the last slice (rows 30-31, the 39-tick sell that trades against the resting bid of 185) is never replayed: exactly the missing fifth trade and the bid resting at 150 instead of 111. Setting gct_minutes_from_got to 1,000,000 does not prevent this, because the span_scaled clock scales with gct; a tape_seconds_span beyond every lane's largest timestamp would. The same draw passed in the earlier sealed run, whose stacked environment did not stop replaying at done.. Downstream: any multi-contract Session run under the span_scaled clock whose last slices share the stack's largest timestamp; C16 results for the other 499 draws are unaffected.
- **Merged:** evidence merged, certificate section unsigned
- **Verdict:** REPEAT · **rationale:** the maintainer, 2026-10-27: re-enter at K2 — build every stacked and environment-alone Session with the corrected full-replay recipe (`tape_seconds_span` at least twice the largest real timestamp of the tape set) and restate findings F8/F10 accordingly; then re-run.

### 18 · 2026-10-27 · `C14` — Capacity edges, one part per ceiling

- **Card:** `C14` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `b2aeeb47ae0bf4be7abe6f60760e84b4`
- **Notebook:** `notebooks/certification/C14_capacity_edges.ipynb` · **Results:** `results/C14.json`
- **Certifies:** OQ-10 (open-questions.md OQ-10 · **Decision impact:**); §5 (scenario-ladder-review.md v3.3); §5 item 3 (scenario-ladder-review.md v3.3); V3-14 (scenario-ladder-review.md v3.3); OQ-15 (open-questions.md OQ-15 · **What is unknown (engineering invariant):**); OQ-15 (open-questions.md OQ-15 · **Consequence for Phase 2 (observed):**); §11.1 (scenario-ladder-review.md v3.3); §5 item 2 (scenario-ladder-review.md v3.3); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5 item 1 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); V3-15 (scenario-ladder-review.md v3.3); §4.2 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); OQ-20 (open-questions.md OQ-20 · **Ruling.**); §4.2.1 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); V3-15 (scenario-ladder-review.md v3.3); V3-15 (scenario-ladder-review.md v3.3)
- **Dataset:** no registered construction; 30 valid of 30 requested draws; seeds 0–900002; generator digest no construction digest
- **Known result:** C14a: the 101st resting add raises book_full and wipes every worst-price order into the evicted bucket, no trade; C14a′ (recorded, not certified; rebuilt r2, full only at regeneration): no flag site meets a full side, the regenerations replace their own consumed rows, no flag and nothing evicted or lost; C14b: after 100 trades the ring keeps only a burst's final trade, in its last row, detectably (saturation flag), while the engine reports the true burst quantity (I6 and I2 violated exactly on the burst); C14c: an aggressor needing > 100 passes (hop bound inside the 160 margin) raises truncated, the loop terminates, I2 closes, the full outcome consistent with its fills; C14c-below: hop bound < 100, no truncation, the exact walk; C14d: an iceberg beyond the slot cap raises truncated at registration, leaves the side channel unchanged, its hidden remainder untracked, the survivors intact; C14e: 65 slices leave truncated false, 66 set it at registration with no trade; S2 controls: with the last rows occupied, a crossing NON on a full own side loses only the documented worst-level wipe, and an iceberg regenerates and is reached; excluded invariants violated exactly as stated
- **Outcome:** 165/165 oracle assertions passed; invariants 9 pass / 0 fail / 1 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 100, trades 104, hops unobserved, concurrent_icebergs 4 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 84.595 (forecast 900.0)
- **Anything the run revealed that the card did not anticipate:** REPEAT re-run of log entry 15 (verdict REPEAT at K2, 2026-10-27); the audit test's citation-tamper check is tied to the second citation's wording, which the K2 re-pointing changed
- **Result:** NOT CERTIFIED
- **Why it failed:** The notebook passed; its audit test failed: test_C14_audit.py::TestBrokenCopies::test_citation_tampering failed (48 of 49 checks passed): the check tampers the results' second citation by replacing 'worst level' with 'last row' and expects the citation judge to reject the copy, but after the K2 re-pointing the second citation (scenario-ladder.md v2.5, the capacity-validity precondition) no longer contains 'worst level', so the tamper changes nothing and the copy is still accepted. Every engine check, the quote check, the invariant battery and the citation check itself (all 12 citations resolve in the released repository) passed.
- **Merged:** evidence merged, certificate section unsigned
- **Verdict:** REPEAT · **rationale:** the maintainer, 2026-10-27: re-enter at K2 with the citations unchanged; at K4 the audit test's broken-copy citation check tampers with a fragment that citation 1 actually contains; then re-run.

### 19 · 2026-10-27 · `C16` — Multi-contract isolation

- **Card:** `C16` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `59b8d6a93b5b60387514fbeba4410694`
- **Notebook:** `notebooks/certification/C16_multi_contract_isolation.ipynb` · **Results:** `results/C16.json`
- **Certifies:** §2.2 (market-spec.md v2.14); §6.4 (market-spec.md v2.14); §6.2 item 2 (market-spec.md v2.14); §2.1 (market-spec.md v2.14); §2.1 (market-spec.md v2.14); §2.2 (market-spec.md v2.14); §2.2 (market-spec.md v2.14); §3.1 (market-spec.md v2.14); §3.1 (market-spec.md v2.14); §5.2 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §8.1 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); §13.6 item 1 (market-spec.md v2.14); §3 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3)
- **Dataset:** C15.random_session; 500 valid of 500 requested draws; seeds 0–499; generator digest C15.random_session df4d104b115333a5
- **Known result:** for every contract of every draw, its lane in a stacked multi-contract environment run ends with exactly the outcome of its isolated engine run (trade rows except timestamps, resting book per side, hidden remainder per iceberg, book_full/truncated); no lane trades a foreign order id; the integer trade value W_c is identical (d = 0.25 h); the contract alone in the environment agrees with both; the isolated runs pass the battery
- **Outcome:** 18/18 oracle assertions passed; invariants 10 pass / 0 fail / 0 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 28, trades 61, hops unobserved, concurrent_icebergs 18 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 7452.457 (forecast 240.0)
- **Anything the run revealed that the card did not anticipate:** REPEAT re-run of log entry 17 (verdict REPEAT at K2, 2026-10-27), every Session built with the full-replay span; nothing else
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass

### 20 · 2026-10-27 · `C14` — Capacity edges, one part per ceiling

- **Card:** `C14` · **Branch:** `omiecid-simulator 0.1.0` · **Commit:** `8bac2ec9648f2cddf8c71c2b6dde1dbd`
- **Notebook:** `notebooks/certification/C14_capacity_edges.ipynb` · **Results:** `results/C14.json`
- **Certifies:** OQ-10 (open-questions.md OQ-10 · **Decision impact:**); §5 (scenario-ladder-review.md v3.3); §5 item 3 (scenario-ladder-review.md v3.3); V3-14 (scenario-ladder-review.md v3.3); OQ-15 (open-questions.md OQ-15 · **What is unknown (engineering invariant):**); OQ-15 (open-questions.md OQ-15 · **Consequence for Phase 2 (observed):**); §11.1 (scenario-ladder-review.md v3.3); §5 item 2 (scenario-ladder-review.md v3.3); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §4.2.1 (market-spec.md v2.14); §5 item 1 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); §5 (scenario-ladder-review.md v3.3); V3-15 (scenario-ladder-review.md v3.3); §4.2 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.1 (market-spec.md v2.14); §6.2 item 3 (market-spec.md v2.14); OQ-20 (open-questions.md OQ-20 · **Ruling.**); §4.2.1 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §4.5 (market-spec.md v2.14); §2.3 (market-spec.md v2.14); V3-15 (scenario-ladder-review.md v3.3); V3-15 (scenario-ladder-review.md v3.3)
- **Dataset:** no registered construction; 30 valid of 30 requested draws; seeds 0–900002; generator digest no construction digest
- **Known result:** C14a: the 101st resting add raises book_full and wipes every worst-price order into the evicted bucket, no trade; C14a′ (recorded, not certified; rebuilt r2, full only at regeneration): no flag site meets a full side, the regenerations replace their own consumed rows, no flag and nothing evicted or lost; C14b: after 100 trades the ring keeps only a burst's final trade, in its last row, detectably (saturation flag), while the engine reports the true burst quantity (I6 and I2 violated exactly on the burst); C14c: an aggressor needing > 100 passes (hop bound inside the 160 margin) raises truncated, the loop terminates, I2 closes, the full outcome consistent with its fills; C14c-below: hop bound < 100, no truncation, the exact walk; C14d: an iceberg beyond the slot cap raises truncated at registration, leaves the side channel unchanged, its hidden remainder untracked, the survivors intact; C14e: 65 slices leave truncated false, 66 set it at registration with no trade; S2 controls: with the last rows occupied, a crossing NON on a full own side loses only the documented worst-level wipe, and an iceberg regenerates and is reached; excluded invariants violated exactly as stated
- **Outcome:** 165/165 oracle assertions passed; invariants 9 pass / 0 fail / 1 not applicable; unobserved: independent_valuation, matching_iterations, queue_timestamp_correctness, rejection_reason; discard rate 0.000
- **Capacity peaks:** resting_per_side 100, trades 104, hops unobserved, concurrent_icebergs 4 — against the 80/80/160/800 envelope
- **Eager vs jitted:** identical
- **Wall clock:** 84.91 (forecast 900.0)
- **Anything the run revealed that the card did not anticipate:** REPEAT re-run of log entry 18 (verdict REPEAT at K2, correction (b): the audit test's broken-copy citation check tampers with a fragment citation 1 contains); nothing else changed
- **Result:** CERTIFIED
- **Merged:** branch merged
- **Verdict:** CERTIFIED · **rationale:** K5 pass
