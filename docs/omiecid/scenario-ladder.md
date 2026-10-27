# Scenario ladder — Track A: the certification scenarios

**Status:** v2.5 · 2026-10-08, amended 2026-10-16, amended 2026-10-20, amended 2026-10-22 (amendment of 2026-10-22, version token kept as for 2026-10-16 and 2026-10-20 (a bump would turn the certified C14 twin's `LADDER_VERSION` pin red): §C13's Known result states the remainder generally as $R = T - M$, $M$ the swept quantity on entry including the agent's fill, in place of $R = T - f$, which holds only when nothing but the agent's order is swept; no change of substance, finding of the r4 C13 K2, certification-log entry 50, `OMIE-CERT-RERUN-3`) (amendment of 2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1`, version token kept as for 2026-10-16: §C8's Dataset and Known result and §C13's Known result and Pass restated for the incoming-iceberg instance chain of the user's ruling of 2026-10-20 (`market-spec.md` v2.14 §4.2.1 Rule 2 *Incoming iceberg* items (2), (4), (6), amended 2026-10-20; OQ-33 revised, OQ-34 resolved): rows split at resting entries and at incoming instance boundaries, and the remainder shows $v_1 = \min(P - u, R)$; re-entry at `OMIE-CERT-RERUN-3`) (amendment, version token kept so the certified citations of `scenario-ladder.md` v2.5 stay valid: §C13's Known result and Pass rows restated after the r2 SCOPE-CHANGE verdict (certification-log entries 35, 45): bit-identity where no tape iceberg fills a resting agent order on entry, the §4.2.1 Rule 2 (1), (4) rule-implied difference asserted where one does; v2.5: §C9 and the out-of-scope list restated after `OMIE-ICEWPI-SIGN-REJECT-1` — the wrong-sign draws are an asserted rejection, not an observation, and the §13.6 sign-rule conformance gap is closed; §C1's peak constraint credits `peak ≤ quantity` to §5.4 v2.14 (`OMIE-SPEC-CERT-AMEND-1`), `0 < peak` staying derived from §4.2; the §9.2 trade side (spec only, C13 F16) added to the unimplemented list, by `OMIE-CERT-LADDER-DOCS-1`; v2.4: the slices-per-iceberg row's hop-bound example (a lone 100-slice iceberg bounds to 100, not 102) and the "sound but loose" paragraph corrected after `OMIE-CERT-HARNESS-2`; v2.3: the iteration ceiling corrected to the 100-pass bound (`_TRIP_BOUND`; `max_match_iterations` = 200 never binds) in the capacity-validity precondition, C8's dataset and C14c, by `OMIE-CAPACITY-DOCS-1`; v2.2: the 64-slices-per-iceberg ceiling added to the capacity-validity precondition, the iceberg recipes and C14e; v2.1: battery rule 3, I4–I5 and I6(b) tolerance text aligned with `OMIE-CERT-INVARIANTS-1`) · supersedes v1.0 of 2026-09-23 (archived in
[`docs/archive/scenario-ladder-attempt-1/`](../archive/scenario-ladder-attempt-1/README.md))
**Source:** revision 2 of the [scenario-ladder plan](scenario-ladder-plan.md), accepted 2026-10-02 — this
document is its Track A, unchanged in substance
**Companions:**

- [the domain review](scenario-ladder-review.md)
- the certification cycle K0–K7 in the research repository's orchestration workflow (not published)
- [the certification log](certification-log.md) and [the certificate](certificate.md)
- the stories and cards in [`agile-plan.md`](agile-plan.md) (`OMIE-CERT-*`)

This is the canonical description of every certification scenario, C0 to C17: its purpose, the dataset
it generates, the result known in advance, the pass criterion, and the expected runtime. Each scenario
runs as one notebook in `notebooks/certification/`, under the K0–K7 cycle.

**The RL scenarios (Track B) are not repeated here.** They are in the separate proposal
[`docs/cid_rl/scenario-ladder-track-b.md`](../cid_rl/scenario-ladder-track-b.md).

---

## The three kinds of evidence, explained


The whole plan rests on three testing concepts. They are worth stating plainly, because which one a
scenario uses determines what that scenario can and cannot prove.

### Goldens

A **golden** is a recorded correct answer. You work out, once and by a means you trust more than the
code, what the program *should* produce for one specific input; you freeze that answer; and from then
on the test re-runs the program and compares against the frozen value. The repo already does this in
two places: `tests/omie/test_scenario_single_contract.py` derives expected trades by hand from
`market-spec.md` and asserts them literally, and `tests/omie/omie_eval_fast_golden.json` freezes a
scored output that only a named story may ever regenerate.

*Strength:* a golden pins the **exact** answer, including things no general property would notice —
the precise price a trade printed at, which specific order filled first. It is the only evidence that
distinguishes "the engine did something self-consistent" from "the engine did the right thing".
*Weakness:* a golden only covers the one input somebody bothered to write down, and it does not scale
— nobody hand-derives the expected trade list of a thousand-order random session. It also has to be
re-derived by hand whenever behaviour legitimately changes, which is why regenerating one is a
governed act in this repo rather than a routine one.

### Conservation invariants

A **conservation invariant** is a property that must hold after *every* run, whatever the input,
because it follows from an accounting identity: nothing may be created or destroyed. The key feature
is that **you do not need to know the right answer to check it**. You do not need to know what a shop
sold today to check that the till balances.

Concretely here: every MW submitted to the book must end the session either filled, still resting, or
cancelled/expired — never lost, never duplicated (invariant I2 below). Total volume bought must equal
total volume sold (I1). The cash that changes hands must sum to zero across participants, within the
settlement-arithmetic tolerance (I6). Each of these is checkable on a completely random session where nobody could say in advance
what the trades should be.

*Strength:* runs on anything, costs almost nothing per scenario once written, and catches the entire
class of "quantity leaked somewhere in the matching loop" — historically the most damaging and most
silent defect class. *Weakness:* an engine can be **consistently wrong yet perfectly conserving**. If
it executed every trade at the aggressor's price instead of the resting order's, volume would still
conserve, cash would still sum to zero, and every conservation invariant would pass while every euro
figure in the thesis was wrong. That specific case is caught by invariant I5 and scenario C2; the
general lesson is that conservation alone is never sufficient.

### Metamorphic invariants

A **metamorphic invariant** is a required relationship between the outputs of **two related runs**.
It is the standard answer to what testing literature calls the *oracle problem*: you cannot compute
what run A should produce, but you know exactly how run A's output must relate to run B's.

Four of the ones used here, in plain terms:

- *Immediate-or-cancel versus resting:* submit the same price and quantity as an IOC order and as a
  NON order against the same book. Nobody needs to know what either one trades — but they must trade
  **identically**, because the two order types differ only in what happens to the leftover quantity
  afterwards. Any difference in the trade list itself is a defect (relation M3).
- *Splitting an order:* against a book that nothing else touches, one order for 90 MW must produce
  the same total volume and the same volume-weighted price as three consecutive orders for 30 MW
  (M5).
- *Fill-or-kill:* an FOK order either fills completely, or it does nothing at all — and in the second
  case the book afterwards must be **byte-identical** to the book before. Historically it is the
  "does nothing at all" half that breaks, because a partial match gets applied and then rolled back
  imperfectly (M4).
- *Ownership blindness:* relabelling which participant owns which order must not change the trade
  list at all. OMIE has **no self-trade prevention**, so an agent's buy matching its own sell is
  correct behaviour; this relation turns that rule into a testable claim (M6).

*Strength:* metamorphic relations catch behavioural defects on arbitrary random input, which goldens
cannot reach, and they catch *relational* errors that conservation misses entirely. *Weakness:* they
constrain the answer without determining it — an engine that is wrong in the same way on both sides
of the relation passes.

### Why all three, and what the combination does not cover

Goldens pin exact answers on a few inputs; conservation invariants check accounting on every input;
metamorphic invariants check behaviour on every input. Each covers the others' blind spot, and the
plan therefore runs the full invariant battery on **every** scenario, not just the fuzz scenario.

Two residual risks belong in the certificate. First, without an independent second implementation of
the matching rules, a **shared misreading of `market-spec.md`** between whoever writes the
expectation and whoever wrote the simulator survives all three layers; the mitigations are the
audit-twin convention, the agent-independence rules below, and the deliberate placement of C2, C9 and
C10, which pin the price-attribution rules where such an error would most plausibly hide. Second, and
subtler: `market-spec.md` §4.2.1's four iceberg rules are recorded there as **this repository's
rulings**. The §4.2.1 preamble describes them as deviations from the literal SIDC/XBID description,
while each rule's own classification records it as consistent with the market rules (an inconsistency
noted in domain review v3.0, V3-21). Scenarios C8, C9 and C10 therefore certify conformance *to those
rulings*, not to XBID itself. That is a legitimate
and documented position, but it must be stated rather than blurred. A fully independent reference
matching engine remains available as a later upgrade if the certificate is ever challenged; per
decision 1 it is **not** in this plan.

---

## Rules the first attempt taught

The first attempt showed that the expectations were right and the **checks** were the weak point. A
check is code too, nothing checks it, and a wrong check looks exactly like a right one: it passes. Four
rules follow, and every later section applies them.

**1. Every check is seen failing before it is trusted.** Each invariant, and each piece of code that
feeds one, must be demonstrated catching a deliberately broken input. The first attempt produced three
checks that *could not fail* — a volume balance that added the same number to both sides, a
conservation identity whose "cancelled" term was computed as the residual (so it reduced to
`submitted = submitted`), and a cash check comparing a ledger against a copy of itself. All three
looked reasonable. The broken-input rule caught two of them within minutes.

**2. The record builder reads the engine; it never re-derives from the input.** The quantities the
battery grades — what filled, what rests, what was cancelled, what hidden quantity remains — are read
from engine state after the run. Anything re-computed from the input tape inherits the input's
assumptions and hides exactly the defects the check exists to catch.

**3. Every check says what it cannot see.** Some information the simulator simply does not expose: it
has no second valuation of a trade, it does not report its matching-iteration count, and nothing it
exposes shows whether the queue timestamps it stores were assigned correctly (the engine's trade rows
do name the resting order, `PASS_OID`; only the `(M, 6)` settlement tape lacks a maker order id). A check that needs one of these is
either narrowed to what it can actually see, or reported as *unobserved*. An inert check dressed as a
live one is worse than an absent one.

**4. Verification is most of the work.** Three independent review rounds cost roughly as much as the
build, each found defects the previous one missed, and two defects were introduced *inside* repairs a
review had demanded. The stories below budget for that from the start.

---

## The load-bearing idea: parametric goldens


Choosing goldens and invariants over a second engine would normally mean randomly generated datasets
get only property checks, never an expected answer — which would not satisfy the request, since the
request asks for a *randomly generated dataset* **and** a *known trade result*.

That is avoidable. The generator does not draw an arbitrary tape and then ask what happens to it. It
**draws the parameters of a construction whose outcome is a closed form in those parameters**, and
then emits the tape. The dataset is genuinely random; the answer is still known, because it is a
formula over the draw rather than a value someone looked up.

Two examples. A C3 draw is "seed $k \in [2,6]$ resting sell orders at one price with increasing
timestamps, plus decoys at worse prices, then one aggressor sized to consume $j < k$ of them" — the
expected fill set is the $j$ earliest order IDs, for every draw, by construction. A C10c draw picks
`(base, ppd, peak, total, limit)` at random, and the volume the fill-or-kill pre-check must see is
$\min\left(\mathit{total},\ \mathit{peak}\times\#\{k\ge 0 : \mathit{base}+k\,\mathit{ppd}\le \mathit{limit}\}\right)$ — again a formula, evaluated per draw.

So every scenario declares its **oracle kind**: **exact-literal** (a hand-computed trade list for one
fixed input — the classic golden), **exact-parametric** (a formula over the draw's own parameters —
the workhorse), or **invariant/distributional** (properties that must hold on every draw, plus a
pre-registered closed-form band).

### The invariant battery

Written once in `tests/omie/reference/omie_invariants.py`, called by every scenario notebook. The
**Cannot see** column is rule 3 made concrete.

| ID | Statement | Cannot see |
|---|---|---|
| **I1** | Matched buy volume == matched sell volume, per contract, per step and cumulatively, each side read **independently** from the trade rows | — |
| **I2** | Per **accepted** order: `filled + still_resting + cancelled_or_expired + evicted + hidden_remaining == accepted`, exact in integer ticks. `cancelled_or_expired` comes from **order semantics** — the §5.1 causes only (IOC remainder, FOK kill, cancellation, gate expiry) — never as the residual `accepted − filled − resting`; an iceberg's hidden quantity is its own bucket, read from the engine; `evicted` (an implementation-limit fate, not a §5.1 cause) is its own bucket, read from the book diff, and must be 0 outside C14 (I8). Rejected orders are recorded separately and are not in the identity | — |
| **I3** | The book is never left crossed once `submit_order` returns (§4.2.1 Rule 4's own post-condition) — checked on the book **after every return**, including the last submission | — |
| **I4** | No trade occurs against a resting order while a better-priced — or equal-priced and earlier — resting order is still unfilled. Each fill carries the index of the **submission** that produced it (one submission can produce many fills), and depth is decremented as a sweep walks the queue; checked against the book **before each submission** | The queue key is the engine's own stored timestamp, so whether the engine assigned those timestamps per §5.2 (partial execution keeps its timestamp; a new iceberg slice gets a new one) is not observable through this check; C5's interloper test is what certifies it |
| **I5** | Every trade price equals the limit price of **that trade's price-setter** — the resting order under §6.2 item 3, or the iceberg under §6.2 item 7 and §5.2; at the §4.2.1 Rule-3 hop, the iceberg (OQ-31 RESOLVED 2026-10-02, Reading B; `market-spec.md` v2.13 §4.2.1 Rule 3 and §6.2 item 7: the hop trades at the regenerated slice's own limit, never at the aggressor's limit) — with one narrow exemption: a price-incremented iceberg's own declared ladder $\mathit{base} + k\,\mathit{ppd}$, $k$ bounded by its declared slice count. Every trade is also matchable (§6.2 item 2): its price is no higher than the buyer's declared limit and no lower than the seller's, an ICE leg taking the most permissive rung of its declared ladder. **This is the check that owns mispricing** | — |
| **I6** | **(a)** The engine's per-participant settlements, summed over every participant id that appears in a trade row, equal zero within the tolerance derived for (b), and every trade row's buyer and seller are enumerated participants (the integer per-row identity holds by construction and is not a check; its broken-input fixture breaks the **settlement output**, e.g. drops one participant's leg); **(b)** the settlement the engine reports for each submission agrees with the trade rows that submission wrote | A trade at the wrong price: settlement is computed from the same rows, so both sides move together. Mispricing is I5's and C2's job |
| **I7** | Total traded volume ≤ min(total buy accepted, total sell accepted) — no liquidity is manufactured | — |
| **I8** | **Capacity headroom** — see [§ The capacity-validity precondition](#the-capacity-validity-precondition) | The matching-iteration count (reported *unobserved*); evictions by regenerated slices (they do not raise `book_full`); which ceiling raised `truncated`, since the iteration and per-iceberg slice ceilings share it, so the generator guarantees both |
| **I9** | **Non-vacuity** — every draw produced at least one accepted order and, where the construction implies one, at least one trade | — |
| **M1** | Permuting orders that share a timestamp does not change aggregate matched volume; where they compete on price, the tie-break must follow §6.1 | — |
| **M2** | Increasing an aggressor's size never decreases matched volume; tightening its limit never increases it | — |
| **M3** | IOC and NON with identical price and quantity produce identical trade lists; the IOC's **own** remainder (by its own id) never rests. The resting maker legitimately survives | — |
| **M4** | FOK either fills exactly the requested quantity, or produces zero fills **and a byte-identical book** | — |
| **M5** | One aggressor of size $Q$ against a static book == $k$ consecutive aggressors of size `Q/k` | — |
| **M6** | Permuting agent identifiers leaves the trade list unchanged — including when the permutation makes an order match its own owner (§6.6, no self-trade prevention) | — |

Two formulations were tried and **rejected**, and are recorded so they are not tried again. I5's
exemption once accepted "any price resting on that side of the book"; a fill that took a 60.00 maker
at 50.00 then passed the entire battery. And I6 was once asked to be an independent check on prices;
there is no independent valuation in the system, so it cannot be.

**What I6 does and does not say.** §8.4 defines a participant's PnL as the cash flow **plus the
market value of the energy acquired or delivered, which it declares explicitly exogenous to the
simulator**. So the quantity that sums to zero is the *cash leg* only: what one side pays, the other
receives. It is not a claim that the trade was economically neutral, and the certificate must say so.

### The record builder

Between the engine and the battery sits the **record builder**: the code that turns one simulator run
into the record every invariant reads. It is where the first attempt's defects concentrated, so it is
specified here rather than left to implementation.

- It records **two snapshot sequences**, because two invariants ask different questions at different
  instants: the book **before** each submission (what I4's priority question is about), and the book
  **after** each return (what I3's no-crossed-book question is about). One shared sequence missed the
  state after the final submission — usually the aggressor, the only order that could leave a crossed
  book.
- Every fill is tagged with the **submission** that produced it.
- Every quantity in I2 is **read from engine state**: resting quantity from the book, hidden quantity
  from the iceberg side channel, cancellations from the order's own fate.
- Rejections are recorded in their own list, with the validator's reason.
- It is **tested on its own**, with broken fixtures, before any invariant is built on it (rule 1).

---

## What "eager", "jitted" and "assertion" mean here


These three terms appear in every scenario row, so they are defined once, here, and reproduced in
each notebook's header.

**Eager execution.** JAX can run a function in two ways. *Eager* means each operation executes
immediately as Python reaches it, exactly like ordinary NumPy: readable, step-through-able in a
debugger, slow. This is the mode in which a human can inspect what the engine did at every
intermediate step, so it is the mode the notebook uses when it displays the book after each
submission.

**Jitted execution.** `jax.jit` traces the function once, compiles the whole thing to a single
optimised program, and thereafter runs the compiled version. It is one to three orders of magnitude
faster, and it is the mode all real training runs use. But compilation changes *how* the code
executes — control flow is flattened into `lax.cond` and `lax.while_loop`, intermediate values are
fused away, and floating-point operations may be re-associated or contracted into fused
multiply-adds. That last point is not hypothetical here: it is exactly the documented
`OMIE-PRICE-DECODE-FMA-1` behaviour, where the price decode gives different last-bit results
depending on backend.

**Why every scenario runs both, and asserts they agree.** A defect that appears only under
compilation — a loop bound that traces differently, a conditional that was fine eagerly but wrong
once flattened — would otherwise be invisible in a notebook and would corrupt every training run.
Asserting that the eager and jitted runs produce the *same* result makes the compiled path, which is
the one that matters in production, carry the same certificate as the readable path.
`tests/omie/test_omie_env_jit_identity.py` and `test_match_loop_fori_parity.py` are existing
precedents.

**Assertion.** A check that records a failure if it is not true. The notebook does **not** stop at
the first failed assertion: every assertion is evaluated and recorded, the run completes, and a
failure is explained in the notebook's last section (see the certification cycle). Three strengths
are used, and each scenario row says which:

- **exact integer equality** — the two values are identical integers. Used for every price, quantity
  and identity claim, because prices and quantities live on integer tick grids (0.01 EUR/MWh, 0.1 MW)
  and the integer is the ground truth; the euro figure is a decoded view of it.
- **structural equality** — two state trees equal leaf for leaf. Used for "nothing changed" claims,
  such as a fill-or-kill rejection leaving the book untouched.
- **bounded tolerance** — equality within a stated margin. Used **only** for reported euro figures,
  never for a pass/fail criterion, precisely because of the FMA behaviour above. A scenario that can
  only state its claim in euros is under-specified and must be restated in ticks.


**The one check that must compare euros.** I6(b) compares the settlement the engine *reports*, a
float, against the trade rows. The input decode `price_int × 0.01 − 9999` is not the arithmetic that
matters: the engine computes the reported value in float32 from the integer weighted price, per
settlement chunk ($(\mathrm{fl}(W) - \mathrm{fl}(O\cdot Q))\cdot\mathrm{fl}(0.1)\cdot\mathrm{fl}(0.01)\cdot 0.25$,
`contract_book.py`). The tolerance for that comparison is therefore **derived from that settlement
chain**, per submission (the formula and its derivation are pinned in the `OMIE-CERT-INVARIANTS-1`
card, V10), not guessed; the `0.02` EUR "practical" figure from the fill-report fix would have failed
correct runs (largest measured error 0.31 EUR). Guessing took the first attempt three tries: one bound was ~91× too
tight and one had the exact and inexact halves backwards.

---

## The capacity-validity precondition

The simulator has five hard ceilings, all consequences of fixed-size arrays and fixed loop bounds: 100
resting orders per side, 100 trades per contract in the trade tape, 100 outer matching passes, 1024
concurrent icebergs, and 64 slices per iceberg (`MAX_SLICES_PER_REGEN`, `jax_constants.py` §B2).
C14 certifies what happens **at** those ceilings. Every other scenario must certify what happens
**away from** them, because behaviour at a ceiling is by design *not the market rule*.

The rule is simple: **the generator guarantees the margin, and the flags only confirm it.** The flags
cannot be the guarantee, because not every ceiling has a working flag:

| Ceiling | Margin the generator enforces | What the run confirms | Why the flag is not enough |
|---|---|---|---|
| Book depth, 100 per side | ≤ 80 resting per side | `book_full == False` | A regenerated iceberg slice that rests on a full side evicts **without** raising the flag |
| Trade tape, 100 per contract | ≤ 80 trades | `n_trades ≤ 80` checked **before** the oracle comparison | When saturated, every further trade lands in the last row; I1, I2 and I6 — the oracle itself — would be silently corrupted |
| Outer matching passes, 100 (`_TRIP_BOUND`; *corrected 2026-10-07, was "Matching iterations, 200"*) | Analytic worst-case **hop bound**, emitted per draw. The generator's envelope still admits up to 160 (`CapacityEnvelope.max_hops`, checked against the inert 200), so for a hop bound between 100 and 160 only the run's `truncated` check enforces this ceiling | `truncated == False` | The pass count cannot be read after a run; the axis is reported *unobserved*. A binding pass bound always sets `truncated`, so it is flagged, never silent |
| Iceberg slots, 1024 | ≤ 800 concurrent | peak slot count | — |
| Slices per iceberg, 64 | Every iceberg drawn with $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$, in integer ticks — by construction in the mode, and re-checked from the tape before the run | `truncated == False`, plus the largest $\lceil \mathit{total}/\mathit{peak} \rceil$ in the draw, reported | The `truncated` flag is **shared** with the iteration ceiling, so `truncated == True` alone does not say which ceiling was reached. It is raised at **registration**, even for an iceberg that never trades, and the hop bound does not cover it: one 100-slice iceberg alone has a hop bound of 100 (102 once it has a crossable counterparty; `OMIE-CERT-HARNESS-2`) |

**The pass ceiling** *(corrected 2026-10-07, `OMIE-CAPACITY-DOCS-1`; documentation only, no engine change)*. This section first named the ceiling "200 matching iterations", after `max_match_iterations` in `omie_config.py`. That value does not bound the outer match loop: the loop's continue test does not read it, and the loop runs a fixed `_TRIP_BOUND` = 100 passes (`contract_book.py`). The `truncated` flag fires when either bound is reached with quantity left, so only the 100-pass bound can ever bind, and when it binds the breach is flagged, never silent (a NON aggressor then re-submits its residual once, outside the loop, which is ceiling behaviour that C14 records and does not certify). The C14c construction (`C14_seed300000`) now reaches the 100-pass bound and is flagged `truncated`. The generator's hop margin (`CapacityEnvelope.max_hops` = 160) is still validated against the inert 200; that is a generator-code matter outside this documentation correction. The only first-run log entry that reports a hop figure (C17) reports 46, and the iceberg modes draw at most 40 slices, so no current mode is known to come near 100. The book-depth row's wipe also fires on a dispatch that executes completely against a full own side; see `book-depth-cap-finding.md` correction 1a.

Three hazards make this more than a general precaution. **The trade tape is the most likely
accidental collision**: a multi-level walk, an iceberg cascade or a fuzz session can pass 100 trades
unnoticed, which is why that check runs before the oracle, so a wrap aborts the draw instead of
manufacturing a false failure. **Book overflow destroys exactly the controls the priority scenarios
depend on**: the documented overflow wipes the worst-price level, which is where C3's decoys and C6's
deep ladder live, so an overflowed C3 draw would delete its own control group and still pass. **For
icebergs the bound must be analytic, not nominal**: §4.2.1 Rule 4 pays for its regeneration cascade out
of the 100-pass budget, and Rule-3 re-crossing adds hops beyond the nominal slice count, so a
truncated draw is indistinguishable from a Rule-4 violation unless the hop bound was guaranteed first.

**The slice ceiling** *(added 2026-10-03, found by the C01 K6-F investigation)*. When an iceberg
enters the book, the engine adds the slices already issued to the slices its hidden remainder $h$
still needs, $\lceil h/\mathit{peak} \rceil$, and sets the sticky `truncated` flag if the sum exceeds
64. At registration no slice has been issued and $h = \mathit{total} - \mathit{peak}$, so the flag
is set exactly when $\lceil (\mathit{total}-\mathit{peak})/\mathit{peak} \rceil > 64$, that is when
$\lceil \mathit{total}/\mathit{peak} \rceil \ge 66$. The sum does not change as the iceberg regenerates,
so the outcome is fixed the moment the iceberg is registered, whether or not it ever trades. Before
this row existed, I8 marked every draw holding a larger iceberg `INVALID`: each such draw was
discarded and re-drawn, and the ladder silently kept only the small icebergs, which is the biased
subspace the discard-rate report exists to expose.

The generator's margin is one slice tighter than the engine's boundary: $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$. This
is the generator's own unit count for an iceberg, and inside it the fill-or-kill pre-check also
sees every hidden slice, since it projects at most 64 of them. A mode guarantees it **by
construction**: draw the slice count $n \in [2, 64]$ first, then set
$\mathit{total} = (n-1)\,\mathit{peak} + r$ with $1 \le r \le \mathit{peak}$, in integer ticks. A mode
that draws `total` and `peak` independently must redraw any iceberg with more than 64 slices before
the tape is emitted, never after the run. Every existing mode already meets the margin: the C04, C08,
C09 and C10 modes draw at most 40 slices, and the fuzz peak law ($\mathit{peak} = \lfloor q/k \rfloor$,
$k \in [2,5]$) gives at most 9. Each notebook re-checks the margin from the tape before the run,
taking $\lceil \mathit{qty}/\mathit{peak} \rceil$ over its iceberg rows, and reports the maximum
beside the other capacity peaks. This needs no new story and no new envelope axis. A scenario that
needs a new iceberg mode adds it under its own card's K3 deliverable (b), within the `OMIE-CERT-SYNTH-1`
contract. Only C14e may exceed the margin.

The hop bound is sound but loose. Since `OMIE-CERT-HARNESS-2`, `tape_capacity_bounds` skips rows the
market rejects at submission and adds its margin of 2 only to a row that can cross another counted row.
It still counts earlier opposite orders that may already have traded away, and rows whose only
counterparty arrives later, so a draw can be refused that would have run well under 100
passes. This costs discards, not correctness, and those discards appear in the discard rate.

`book_depth` and `max_icebergs` are bounded **independently**, since they are independent axes. A run
that breaches a margin is **INVALID, not failed** — it tested nothing and is re-drawn — and each
notebook reports its **discard rate**, because a ladder that silently threw away 30 % of its draws is
measuring a biased subspace.

---

## Track A — the certification scenarios

Every scenario: runs eager and jitted and asserts the two agree; asserts in **integer tick space**
and only *reports* euro figures; runs the full invariant battery including the capacity precondition
I8 and the non-vacuity check I9; and **quotes the governing market-spec text verbatim in the
notebook**, not merely by section number — the quoted sentence is the specification the expectation
is derived from, and having it on screen next to the code is what lets a reviewer check the
derivation without leaving the notebook. Random scenarios default to 500 draws.

**Execution-time forecasts.** Every row carries an estimated runtime, derived from the repo's own
measured warm-pass throughput on the dev workstation (per `docs/omie/test-perf-gate.md` §6): **0.032
ms per order on the NON path** and **~0.95 ms per order on the all-iceberg path**, plus 10–30 s of
compilation per distinct traced shape on a cold cache. Forecasts assume a warm cache; a cold first
run adds the compile term once. They are budgeting estimates, and each card records the measured
figure when it runs.

---

### C0 — Inertness and the contract-status gate

**Purpose.** Before any claim about what the engine *does*, establish what it does when it should do
nothing. Three things are pinned. First, that an empty book is genuinely empty and a fresh state is
reproducible leaf-for-leaf. Second — and this is the one that matters downstream — that the **pad row
sentinel is truly inert**: every tape is right-padded to a common length, and if a pad row ever
submitted an order or advanced the clock, every episode length and every reward in Track B would be
quietly wrong with no visible symptom. Third, that the contract-status gate refuses orders outside
the trading window.

The third claim carries a wrinkle worth certifying honestly. The code gates on a single
`status == OPEN` test, collapsing PENDING, SUSPENDED and CLOSED into one rejection branch. That is
conformant for submission — §3.3 suspends order submission in the IDA clearing window — but it cannot
express the rest of §3.3's phase 2: resting orders survive the window and are batch-matched at reopen
(v2.10, OQ-28), whereas at CLOSED they are cancelled. PENDING and SUSPENDED are **simulator statuses**
with no market-spec existence as contract statuses: §9.2's contract status is OPEN / CLOSED, and
suspension is its separate "IDA suspension phase" variable. The scenario therefore exercises all three
non-OPEN statuses and certifies the collapse as **observed behaviour**. Note also that pad-row inertness has **no spec section at all** — it is a
simulator artefact, and saying so is more honest than laundering it under a lifecycle citation.

| | |
|---|---|
| **Certifies** | §3.1's gate rule for a contract's **given** GCT (the status gate) — the v2.10 hour-aligned GCT(k) schedule is quoted as context, not certified; §5.4 bullet 4 (*"Order submitted after GCT for the target contract"*); §9.2 row "Contract status". Pad-row inertness: **no spec section — simulator artefact** |
| **Quotes** | §3.1 *"Per-contract gate closure (GCT): 1 hour before the start of the delivery period"*; §5.4's rejection bullet; §3.3's phase-2 description — "Both regular matching and explicit allocation are fully suspended." and "Order submission is also suspended for affected contracts." — quoted without its Matching Algorithm §5.5 source sentence, which concerns capacity-update batch matching (domain review v3.0 R-1); the suspension rests on OMIE Rules §58.2 (e)–(f) and Instruction 1/2024 §8. The v2.10 reopen citation (§5.5 with §6.3) is apt for the reopen batch match (R-1 point 3). **And:** §3.1's v2.10 correction, quoted as context: "In the Iberian continuous market, GCT is aligned to the delivery hour, not to each quarter-hour." |
| **Dataset** | Deterministic: empty tape; 64 pure pad rows; a valid random tape submitted to each of PENDING, SUSPENDED and CLOSED |
| **Known result** | Zero trades; state equals `book.init()` leaf-for-leaf; every submission rejected; `n_active = 0`, no flags raised |
| **Oracle kind** | exact-literal |
| **Pass** | Structural equality on every leaf, for all three non-OPEN statuses |
| **Failure implicates** | Pad-row semantics or the status gate |
| **Runtime** | ~40 s (compile-dominated) |
| **Notebook** | `C00_inertness_and_status.ipynb` |

---

### C1 — Order validation and rejection

**Purpose.** Before certifying what the book does with an order, certify which orders are allowed in.
The validator enforces §2.3's price and quantity bounds and tick grids and §4.2's iceberg peak
constraint, in fused integer arithmetic **after a tick-snap** — which produces a subtle documented
behaviour nobody has certified: a price off the grid by *less* than half a tick is snapped to the
nearest tick and **accepted**, while one off by half a tick or more commits to a different tick and
may then fall out of bounds. That boundary is exactly where a validator quietly diverges from its
specification.

There is a second, structural justification that raises this scenario's priority: **every later
scenario's random draw assumes its generated orders are accepted**. If the validator silently
rejected, say, an off-tick draw, a downstream scenario could pass *vacuously* with zero trades.
Certifying acceptance first is what makes every subsequent draw trustworthy — and it is why invariant
I9 exists.

Scope note: two of §5.4's five rejection conditions — the agent guarantee/operating limit and the TSO
unavailability check — are **not implemented**. The notebook names them as known non-coverage so the
certificate cannot read as though §5.4 were fully certified.

| | |
|---|---|
| **Certifies** | §2.3 bounds and ticks; §4.2 iceberg peak constraint `0 < peak ≤ quantity` — `peak ≤ quantity` stated by §5.4 since v2.14 (Rules Annex §4.4), `0 < peak` derived from §4.2, not stated by it; a validator invariant; §5.4's implemented subset |
| **Quotes** | §2.3's bounds table verbatim; §5.4's rejection list, with the two unimplemented conditions named out of scope |
| **Dataset** | Stratified random draws over seven classes: in-bounds on-tick (accept); price and quantity each below and above bound (reject); sub-half-tick off-grid (**accept, snapped**); ≥half-tick off-grid (commits to a different tick — accept or reject per bounds); iceberg `peak > quantity` or `peak ≤ 0` (reject) |
| **Known result** | Accept/reject per class by construction; on accept, the exact snapped integer the order enters with; on reject, a byte-identical book |
| **Oracle kind** | exact-parametric, reusing M4's no-mutation relation |
| **Pass** | Exact accept/reject per draw; exact snapped integer; structural equality on reject |
| **Failure implicates** | Bound arithmetic or tick-snap semantics — would let malformed orders into every later scenario |
| **Runtime** | ~1 min |
| **Notebook** | `C01_order_validation.ipynb` |

---

### C2 — The single crossing pair

**Purpose.** This certifies the single most consequential rule in the market specification: **a trade
executes at the resting order's limit price, never at the aggressor's**. If this is wrong, the engine
still trades, still conserves volume, still balances cash — every conservation invariant passes — and
every euro figure the thesis reports is wrong by the spread. There is no cheaper place to catch it
and no scenario that matters more. The draw is randomised over price, gap, both sizes and side so the
rule is pinned across the grid rather than at one lucky point, and the assertion is in integer ticks
so it cannot be blurred by floating-point decode.

| | |
|---|---|
| **Certifies** | §6.1 ranking and match condition; §6.2 item 3 (price determination); §11.1 row *"Execution price = resting order's limit price in regular matching"* |
| **Quotes** | §6.1 *"A match occurs when best bid price ≥ best ask price"*; §11.1's row "Execution price = resting order's limit price in regular matching", sourced to Continuous Trading Matching Algorithm §5.1.1 (the ACER/CACM provenance belongs to the price-time-priority row, which C3 cites) |
| **Dataset** | Random $(p_{\text{rest}},\ \Delta \ge 1\ \text{tick},\ q_{\text{rest}},\ q_{\text{agg}},\ \text{side})$ on the tick grid |
| **Known result** | Exactly one trade, $\text{qty} = \min(q_{\text{rest}}, q_{\text{agg}})$, **$\text{price} = p_{\text{rest}}$** |
| **Oracle kind** | exact-parametric |
| **Pass** | Per draw, exact integer equality on price and quantity |
| **Failure implicates** | Price attribution — invalidates every PnL number downstream |
| **Runtime** | ~1 min |
| **Notebook** | `C02_single_crossing_pair.ipynb` |

---

### C3 — Price-time priority, ties, and ownership blindness

**Purpose.** Fairness of the queue is what makes a limit order book a market rather than a lottery,
and it is the rule an agent's passive orders live or die by. (OQ-25 once traced the agent's 0.63 %
passive fill rate to its queue position behind never-cancelled phantom liquidity. That attribution is
**no longer supported**: until `OMIE-AGENT-TIME-PRIORITY-1`, the environment stamped agent orders from
its minutes clock, which put them at the *front* of every same-price queue. The figure was therefore
measured with front-of-queue priority, and it is flagged for re-measurement — see
[the note in Track B](../cid_rl/scenario-ladder-track-b.md#before-track-b-the-rl-model-is-simplified).) The scenario's important
design choice is **what it asserts**: not traded volume, but *which specific orders filled*, by order
ID. A priority bug leaves total volume unchanged, so a volume assertion would pass over it silently.
Decoys at worse prices are included in every draw so price ranking and time ranking are tested
together.

Two further claims ride along. §5.2 requires that all timestamps come from **a single global, unique,
sequential queue** — the property this scenario's entire tie-break claim rests on — so a monotonicity
and uniqueness assertion on the timestamp counter is folded in here. And §6.6's absence of self-trade
prevention becomes a positive assertion: a crossing pair carrying the **same** agent identifier must
print a trade, identical in price and volume to the distinct-agent counterfactual. That is relation
M6, and it costs one cell.

| | |
|---|---|
| **Certifies** | §6.1 price-time priority; §5.2 global unique sequential timestamps; §6.6 no self-trade prevention; §11.1 row "Price-time priority" |
| **Quotes** | §6.1 *"Buy side: ranked highest price first; ties broken by earliest timestamp"* and its sell counterpart; §5.2 *"All timestamps … are assigned from a single global queue (unique, sequential)"*; §6.6 *"The matching engine must not reject or skip matches based on order ownership"* |
| **Dataset** | Random: $k \in [2,6]$ resting orders at one price with increasing timestamps, plus worse-priced decoys; aggressor sized for $j < k$. Separate M6 draw with a same-owner crossing pair |
| **Known result** | The filled set is exactly the $j$ earliest `order_id`s; decoys untouched; timestamps strictly increasing and unique; the same-owner pair trades, identically to the permuted-owner counterfactual |
| **Oracle kind** | exact-parametric, plus M1 and M6 |
| **Pass** | Per draw, set equality on filled order IDs; M6 exact trade-list equality |
| **Failure implicates** | The ranking key, the tie-break, or an ownership-based exclusion that OMIE does not have |
| **Runtime** | ~2 min |
| **Notebook** | `C03_priority_ties_and_ownership.ipynb` |

---

### C4 — Cancellation

**Purpose.** Removing a resting order must remove exactly that order, produce no trade, and — the
non-obvious part — leave **the queue order of the survivors unchanged**. That last clause is the
reason this scenario is not trivial: the removal compacts a fixed-size array, and a compaction that
swaps the vacated slot with the last element (a normal, efficient implementation choice) silently
reorders the survivors. Every quantity would still conserve, no trade would be wrong, and C3's
priority guarantee would quietly stop holding for every order behind a cancellation. It is asserted
by identity, not by count.

The second claim concerns icebergs, and it reaches into a different subsystem. Cancelling an iceberg
must release its side-channel slot and make its hidden remainder genuinely disappear — because a
later fill-or-kill pre-check reads that side-channel directly, and a stale slot would report phantom
hidden volume. That is not a hypothetical failure mode: it is the shape OQ-20 and OQ-23 already
document. So the scenario cancels an iceberg and then submits a fill-or-kill sized to succeed *only
if* the cancelled hidden volume were still visible to the pre-check, and requires it to be rejected.

**Why modification is excluded, recorded so the gap is visible.** §5.3 makes modification reset an
order's timestamp — *"a hard rule from the XBID system"* — and `modify_order` is implemented (as
cancel-then-resubmit). Excluding it is a **scope decision, not an absence**, and it leaves one hard
rule uncertified. The certificate must say so in those terms rather than letting §5.3 read as
covered.

| | |
|---|---|
| **Certifies** | §5.1's branch `CANCELLED (by agent; GTD expiry; GCT; market halt)`, quoted whole; §6.1 priority preservation across a removal; iceberg side-channel release |
| **Quotes** | §5.1's cancellation branch, quoted whole; §6.1's ranking rule, as the property the survivors must retain |
| **Dataset** | Random: $k \in [2,5]$ resting orders at one price with increasing timestamps, plus worse-priced decoys; cancel a randomly chosen one; then an aggressor sized for $j$ fills. Separate iceberg draw: cancel a resting iceberg, then submit a FOK that only its hidden volume could have satisfied |
| **Known result** | The named order is gone; the survivors' relative order is preserved, so the fill set is the $j$ earliest of the remaining originals; no trade is printed by the cancellation itself; the cancelled iceberg's hidden volume is invisible to the subsequent pre-check, which rejects |
| **Oracle kind** | exact-parametric |
| **Pass** | Exact fill-identity set after the cancellation; exact book composition; zero trades from the cancel; FOK rejected on the iceberg draw |
| **Failure implicates** | Array-compaction reordering (silently voids C3's guarantee downstream of any cancellation) or a stale iceberg slot |
| **Does not certify** | §5.3 order modification — excluded by scope decision, though implemented |
| **Runtime** | ~2 min |
| **Notebook** | `C04_cancellation.ipynb` |

---

### C5 — Partial fill and the resting residual

**Purpose.** Most orders in a real session do not fill completely, so the handling of what is left
over is the ordinary case, not an edge case. Three things are asserted separately: the residual
**quantity**, the **price** it rests at (its own limit, not the price it just traded at), and its
**timestamp**. This is also the first scenario where conservation invariant I2 does real work.

**The user ruled on 2026-09-23 that the residual keeps its time priority**; `market-spec.md` §5.2
states it since v2.9, sourced to Continuous Trading Matching Algorithm §4.3 and §5.1.3: "Partial
execution is NOT a modification and does NOT assign a new timestamp." §5.2 states the rule separately
for the incoming residual and for a partially executed resting order, so the notebook quotes a settled
rule for both cases rather than an inference.

The ruling is asserted twice over. First on the stored field: the residual's timestamp is
**unchanged from the original submission**. Then — and this is the assertion that carries the weight
— behaviourally: after the partial fill, an interloper arrives at the residual's price, a second
aggressor is sent, and the residual must fill **ahead of** it. Timestamp equality alone could pass on
a field that the priority comparison never actually reads, which is exactly the kind of defect a
stored-value check invites.

The risk this guards against is concrete: the natural way to implement a resting residual is to
re-submit it, and a re-submission ordinarily takes a fresh timestamp. An implementation that took
that route would lose the residual's queue position on every partial fill, conserve every quantity,
print no wrong trade, and pass every other invariant in the battery. The interloper is what makes it
visible.

**The simulator does exactly this today.** On 2026-09-29 it was confirmed that when an incoming NON
order partially fills, its residual rests with a **new** timestamp (at least one tick later than its
original), because the matching loop cancels and re-dispatches it; a competing order sent in the same
second then jumps the queue. The fix, **`OMIE-NON-REMAINDER-TS-1`**, merged on 2026-09-29
(`231a5c9`), and `market-spec.md` v2.9 states the rule explicitly. The C5 card depends on it, and
the dependency is met: C5 is now expected to **pass**, and it is the certification that shows the fix
holds on randomised draws, not just on the story's own fixtures.

The ruling also has a bearing on Track B worth recording: an agent whose resting order is partially
filled does not go to the back of the queue, so partial fills cannot be a source of passive-fill
starvation.

| | |
|---|---|
| **Certifies** | §5.1's `PARTIALLY_EXECUTED + resting in BOOK` branch; §6.2 item 4 (NON residual rests); §5.2 as amended by the 2026-09-23 ruling |
| **Quotes** | §5.1's state-diagram branch; §5.2's new-timestamp list **and the ruling recorded beside it**, so the reader sees that a partial fill is not among the events that renew priority |
| **Dataset** | Two draws, each with an interloper at the residual's price: (i) aggressor residual — random $q_{\text{agg}} >$ total crossing depth, the residual rests under §6.2 item 4; (ii) resting residual — a resting NON order partially filled by a smaller aggressor. Both are needed because §5.2 states the rule for each separately |
| **Known result** | $\text{filled} = \sum \text{crossing depth}$; residual $= q_{\text{agg}} - \text{filled}$ rests at the aggressor's own limit, **carrying its original timestamp**; simulator status `PARTIALLY_FILLED` (§5.1's `PARTIALLY_EXECUTED + resting in BOOK`); the residual fills **before** the later-arriving interloper; in (ii), the partially executed resting order keeps its timestamp and fills ahead of the interloper; I2 closes |
| **Oracle kind** | exact-parametric |
| **Pass** | Exact residual quantity, price and original timestamp; exact fill order against the interloper; I2 in integer ticks |
| **Failure implicates** | A residual re-submitted with a fresh timestamp, which would silently cost every partially-filled order its queue position |
| **Runtime** | ~1 min |
| **Notebook** | `C05_partial_fill_residual.ipynb` |

---

### C6 — Multi-level walk and volume-weighted price

**Purpose.** A large order consumes several price levels and prints several trades at different
prices; the resulting average is what the agent actually pays, so this is where execution cost
becomes real. Two failure modes are targeted. The first is level iteration — stopping one level early
or double-counting one. The second is quieter and is why the scenario carries a deliberate stress
draw: the accumulator summing $\text{price}\times\text{quantity}$ runs in integer space and can overflow.
`trade_extraction.py` documents an int32-to-int64 promotion for exactly this reason, so one
parameterisation sits at the ~2 × 10¹⁰ product boundary. The comparison is made on the integer
numerator and denominator **before any division**, so an overflow cannot be masked by rounding. The
trade-ring precondition binds here in particular: a twelve-level sweep repeated across draws is one
of the easiest ways to wrap the 100-trade buffer.

| | |
|---|---|
| **Certifies** | §6.2 item 6 (level continuation). The int64 accumulator is an **engineering invariant, not a spec rule** — labelled as such |
| **Quotes** | §6.1 ranking, which fixes the order levels are consumed in; §8.1's per-trade value formula, since the walk prints one trade per level |
| **Dataset** | Random ladder of $L \in [3,12]$ levels; aggressor sweeping $m < L$ levels plus a partial. Depth ≤ 80 per side and trades ≤ 80 per I8. One stress draw at the promotion boundary |
| **Known result** | Exact per-level fill vector; `(total_qty_int, weighted_price_int, num_fills)` |
| **Oracle kind** | exact-parametric |
| **Pass** | Exact integer equality on all three, before any float division |
| **Failure implicates** | Level iteration, or silent integer overflow |
| **Runtime** | ~2 min |
| **Notebook** | `C06_multi_level_walk_vwap.ipynb` |

---

### C7 — Immediate-or-cancel

**Purpose.** The first scenario carried primarily by a **metamorphic** relation rather than a computed
answer, included partly to demonstrate the technique on something simple. IOC and NON are the *same
order* as far as matching is concerned; they diverge only afterwards, in what happens to the unfilled
remainder. So the trade lists must be identical (M3), and nobody needs to know what either contains
to check that. Three regimes — no liquidity, partial, full — are drawn every time, because the
interesting IOC bug only appears when there is a leftover to dispose of.

| | |
|---|---|
| **Certifies** | §4.5 table, IOC row; §6.2 item 5; §5.1's branch `CANCELLED (IOC/FOK unmatched; GTD expired; GCT reached; halt/hibernation)`, quoted whole |
| **Quotes** | §4.5's IOC row (partial fill permitted, book entry not); §5.1's cancellation branch, quoted whole |
| **Dataset** | Random, three regimes by construction |
| **Known result** | Trade list identical to the NON counterfactual; residual never rests; status `CANCELLED` when partial |
| **Oracle kind** | exact-parametric, carried by M3 |
| **Pass** | M3 equality plus a zero-residual assertion |
| **Failure implicates** | The IOC post-match branch or path dispatch |
| **Runtime** | ~2 min (each draw runs twice) |
| **Notebook** | `C07_ioc.ipynb` |

---

### C8 — Iceberg regeneration

**Purpose.** An iceberg shows a small peak and hides the rest; when the peak is consumed a new slice
appears and may immediately be consumed too. §4.2.1 Rule 4 makes a strong demand about this cascade:
it must reach a fixed point **inside the originating matching process**. Draining it across later
submissions would be a specification violation and would show up as a book left crossed between
arrivals — which is exactly what invariant I3 asserts. Rule 3 gets its own draw because it is the
genuinely awkward case: the aggressor only partially consumes the peak and rests, and the
*regenerated* slice must then cross that resting remainder and trade again.

Two precision points. The hop bound, not the nominal slice count, is what must stay under the
iteration ceiling — Rule-3 re-crossing adds hops, and a truncated draw cannot be distinguished from a
Rule-4 violation by looking at the trade list. Every iceberg also stays within the slice margin,
$\lceil \mathit{total}/\mathit{peak} \rceil \le 64$, because the same flag is raised at registration by any larger iceberg
([§ The capacity-validity precondition](#the-capacity-validity-precondition)). And the scenario must **not** assert anything about
internal `iceberg_meta` slot state: OQ-20 and OQ-23 rule that a stale slot is an engineering
invariant, explicitly **not** a Rule-4 breach, so asserting it would fail the scenario on a non-rule.

**What the caller is told.** The first attempt at this ladder found a real defect here: an
*incoming* iceberg that filled across several of its own slices wrote every slice's trade to the book
but reported only the first slice back to the caller — 30 MW traded, 10 MW and 125.00 EUR reported
instead of 375.00. It was fixed by `OMIE-ICE-INCOMING-FILL-REPORT-1` (merged 2026-09-28, `1f4755a`).
C8 therefore also certifies that the quantity and settlement a submission **reports** equal the trade
rows that submission **wrote** (invariant I6(b)), for incoming and resting icebergs alike. The claim
holds under four stated preconditions that I8 and the generator already guarantee: no truncation, no
book-full eviction, a sign-valid price increment, and a trade ring that is not saturated.

| | |
|---|---|
| **Certifies** | §4.2.1 Rules 2/3/4; §4.2 (*"a new slice … receives a new timestamp"*); §5.2, the iceberg-slice bullet ("Iceberg slices receive a new timestamp per slice, except …"; the passive-side exemption for price determination); §6.2 item 7. Report-equals-tape is an **engineering invariant**, labelled as such |
| **Quotes** | §4.2.1 Rule 4 in full, including *"a crossed order book cannot persist"*; §5.2, the iceberg-slice bullet, *"the iceberg is treated as the passive (resting) side throughout and its timestamp does not supersede the incoming order for price determination purposes"* |
| **Dataset** | Random `(total, peak)` with $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$ and an analytic worst-case hop bound under the 100-pass bound (the generator envelope alone admits ≤ 160; see the capacity-validity precondition, corrected 2026-10-07); one aggressor consuming the lot. Separate Rule-3 draw. Separate incoming-iceberg draw: an iceberg `(total, peak)` submitted against a resting ladder deep enough to fill $\lceil \mathit{total}/\mathit{peak}\rceil$ of its own slices; it is a chain of instances of size $\mathit{peak}$, and its trade rows split at resting entries **and** at its own instance boundaries (`market-spec.md` v2.14 §4.2.1 Rule 2 *Incoming iceberg* item (2); *amended 2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1`: was split at resting entries only*). Every iceberg in every draw, incoming ones included, satisfies the slice margin |
| **Known result** | Exactly $\lceil \mathit{total}/\mathit{peak} \rceil$ slices; sizes `peak,…,peak,r`; all trades carry the **originating** matching process's timestamp (resting-iceberg and Rule-3 draws); Rule-3 trade price per OQ-31; book uncrossed on return; `truncated = False`. Incoming-iceberg draw: reported quantity = Σ slice fills = $\mathit{total}$, every trade at its price-setter's limit (I5), report equals tape (I6(b)), I3; its rows split at resting entries and at incoming instance boundaries (every row at most $\mathit{peak}$; every multiple of $\mathit{peak}$ below the swept quantity $M$ is a cumulative row boundary), and a remainder $R > 0$, if any, rests as instance $k_e = \lfloor M/\mathit{peak} \rfloor$ showing $v_1 = \min(\mathit{peak} - u, R)$ with $u = M - k_e\,\mathit{peak}$ (§4.2.1 Rule 2 *Incoming iceberg* items (2), (4); *amended 2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1`*) — and **no single-timestamp claim**: each of its slices is a new-timestamp entry (§4.2, §5.2) |
| **Oracle kind** | exact-parametric, plus I3 and I6(b) |
| **Pass** | Exact slice-size sequence and count; I3 at every return; hop bound asserted under the ceiling; reported fill equals the tape rows written, exact in ticks |
| **Must not assert** | `iceberg_meta` slot cleanliness — OQ-20/OQ-23 rule it a non-rule |
| **Failure implicates** | Cross-cycle draining (a specification violation), or the regeneration-skip optimisation flagged in OQ-23 |
| **Runtime** | ~8 min |
| **Notebook** | `C08_iceberg_regeneration.ipynb` |

---

### C9 — Price-incremented icebergs

**Purpose.** When an iceberg carries a price increment, each regenerated slice rests at a price offset
from the last, so one resting order walks the price ladder as it is consumed — and the iceberg, not
the aggressor, sets every one of those prices. The scenario earns its own place because of a specific
nasty failure: a **sign error on the increment**. Flip the sign and the slices drift the wrong way,
the engine still trades, volume still conserves, cash still balances, and every invariant passes.
Only an assertion on the exact price *sequence* catches it. It also pins where matching stops — at
the first slice whose drifted price no longer crosses — leaving the iceberg resting with hidden
quantity, which an agent would otherwise misread as exhausted liquidity.

One constraint that is easy to get backwards. §13.6 makes the **sign** rule hard (buy increments
negative, sell positive) and explicitly establishes that there is **no rule-level magnitude bound**,
citing a real `PPD = -3.00` counter-example (§13.6 writes it with an ASCII hyphen; the notebook byte-copies it). So draws must respect the sign rule, and the draw
*range* must be justified as an empirical test envelope — never cited as though it were a bound.

**The sign rule is enforced, and C9 asserts it (since r2).** Until `OMIE-ICEWPI-SIGN-REJECT-1` the
validator checked only `0 < peak ≤ quantity` and accepted a wrong-sign increment (`open-questions.md`,
OQ-23, CORRECTION + CLOSURE NOTE (a)), so the first run logged one wrong-sign draw as an unasserted
observation. The validator (`order_validator.py`) now also rejects a BUY with PPD > 0 and a SELL with
PPD < 0, the rule `market-spec.md` v2.14 §5.4 states (Rules Annex §4.5; §13.6 item 1). So the main
draws stay sign-valid by construction, and a separate, clearly labelled set of wrong-sign draws, from a
disjoint seed range, is an **asserted rejection**: each must be rejected, with no trade, the book
unchanged by its submission and the aggressor resting whole at its limit. These draws are part of the
pass criterion, and the §13.6 conformance gap is closed (C9 r2, log entry 31).

| | |
|---|---|
| **Certifies** | ICEwPI slice pricing on sign-valid draws; §4.2.1 Rule 2 (stop at the first non-crossing drifted slice); §6.2 item 7 (iceberg is always the price-setter); §13.6's magnitude statement (no rule-level bound) as the ground for the draw envelope; §13.6's sign rule, as §5.4 (v2.14) states it: a wrong-sign increment is **rejected** (asserted since r2, after `OMIE-ICEWPI-SIGN-REJECT-1`) |
| **Quotes** | §13.6 on the sign rule being hard and the magnitude explicitly unbounded, with its 2026-01-12 counter-example; §5.4's ICEwPI sign-rejection bullet (v2.14); §6.2 item 7 |
| **Dataset** | Random $\mathit{base}$, sign-valid $\mathit{ppd} \ne 0$, aggressor limit crossing exactly $m$ slices; every iceberg within the slice margin $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$. Draw range documented as an empirical envelope. A labelled set of wrong-sign draws (BUY $\mathit{ppd} > 0$, SELL $\mathit{ppd} < 0$) from a disjoint seed range |
| **Known result** | Exact price sequence $\mathit{base},\ \mathit{base}+\mathit{ppd},\ \mathit{base}+2\,\mathit{ppd},\ \ldots$; stop index $m$; iceberg left resting with positive hidden quantity. Every labelled wrong-sign draw is **rejected**: no trade, the book unchanged by its submission, the aggressor resting whole at its limit |
| **Oracle kind** | exact-parametric |
| **Pass** | Exact integer-tick equality on the full price sequence and the stop index; every wrong-sign draw rejected as stated |
| **Failure implicates** | Price-setter attribution or PPD sign handling — invisible to every volume invariant |
| **Runtime** | ~5 min |
| **Notebook** | `C09_icewpi_price_drift.ipynb` |

---

### C10 — Fill-or-kill, in three parts

**Purpose.** Fill-or-kill has the most dangerous failure mode in the engine, because it requires the
simulator to *look before it leaps*: count available volume first, then decide whether to match at
all. Two things can go wrong. The count can be wrong — and OMIE's rules make it genuinely subtle,
because the pre-check may see volume **hidden from the public book**, and when the iceberg carries a
price increment the pre-check must project the price ladder forward slice by slice to know how much
is actually reachable. Or the rollback can be wrong: a partial match applied and imperfectly undone,
leaving the book corrupted with no trade to show for it. The three parts separate these concerns, and
M4 pins the rollback by requiring the rejected case to leave the book byte-identical.
`tests/omie/test_fok_icewpi_precheck.py` is the locked predecessor this generalises.

This scenario deliberately **follows** the iceberg scenarios. Parts (b) and (c) depend on
hidden-volume accounting and the drifted price ladder that C8 and C9 certify; C10c's oracle is
literally C9's price sequence summed, so running it first would make a failure unattributable between
pre-check counting and drift computation — the very separation this scenario exists to provide.
Parts (b) and (c) keep every iceberg within the slice margin of the capacity-validity
precondition, $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$. Past it the registration raises `truncated`, so the draw
would be `INVALID` before the pre-check is ever exercised.

| | Part | Dataset | Known result |
|---|---|---|---|
| **C10a** | Boundary | Random visible ladder; FOK sized at $V \pm 1$ tick around available volume $V$ | The M4 dichotomy, both sides every draw |
| **C10b** | Hidden volume | Random resting iceberg `(peak, total)` with $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$, request $\mathit{req}$ | Accepted iff $\mathit{req} \le \mathit{total}$ — hidden quantity counts |
| **C10c** | Price-incremented | Random `(base, ppd, peak, total, limit)` with $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$ | $\min\left(\mathit{total},\ \mathit{peak}\times\#\{k\ge 0 : \mathit{base}+k\,\mathit{ppd}\le \mathit{limit}\}\right)$; on accept, fills at the drifted prices, uncapped |

| | |
|---|---|
| **Certifies** | §4.5 FOK row; §4.2.1 Rule 1 (the load-bearing one) and Rule 2 (multi-slice per aggressive order); §6.2 item 7 (no price capping); OQ-11 |
| **Quotes** | §4.2.1 Rule 1 (Real market rule) *"FOK orders see the iceberg's full remaining quantity for matching purposes"*; §4.2.1 Rule 1 in full with its ICEwPI worked example. (Not §4.2: §4.2 says icebergs *"cannot combine with IOC or FOK behavior"*.) |
| **Oracle kind** | exact-parametric, plus M4 |
| **Pass** | Exact accept/reject and fill quantity; on reject, structural equality of the pre- and post-submit book |
| **Failure implicates** | Pre-check counting (a/b), price projection (c), or rollback (M4) |
| **Runtime** | ~6 min |
| **Notebook** | `C10_fok_and_hidden_volume.ipynb` |

---

### C11 — Expiry at gate closure

**Purpose.** Gate closure is the hard boundary that makes this an electricity market rather than a
continuous exchange: at one moment everything still resting is cancelled and the contract is over. It
is also the exact moment Track B's headline reward is paid, so an error here corrupts every euro
figure there rather than merely a test. Two rules are asserted: everything resting is removed,
good-for-session and good-till-date alike; and expiry removes orders **without creating trades** — a
sweep that accidentally matched on its way out would manufacture volume that never existed. The
ground for the second claim is §3.1 ("Orders on a contract are automatically cancelled at GCT if
unmatched") together with §6.2 (regular matching is triggered by a new-timestamp order, and an expiry
sweep is not one); §8.5 is quoted for trade firmness only. Icebergs are included, because GCT
cancels the whole order, hidden remainder and all.

**What this scenario does not claim.** Mid-session lapse of a good-till-date order is §4.6 behaviour
and appears **not to be implemented**: validity and expiry are carried on the order, but there is no
mid-session expiry sweep, and the environment hardcodes every synthesised order to good-for-session.
Rather than assert a behaviour its own author cannot ground in code — which would produce either a
misleading pass or a confusing failure — the claim is omitted, and the certificate records that
**§4.6 is therefore not fully satisfied**, as a named conformance gap rather than a silent omission.

| | |
|---|---|
| **Certifies** | §3.1's gate rule for a contract's **given** GCT (the hour-aligned GCT(k) schedule is not certified); §5.1's branch `CANCELLED (IOC/FOK unmatched; GTD expired; GCT reached; halt/hibernation)`, quoted whole; §8.5 trade firmness (paraphrased here; quoted in full under Quotes) |
| **Quotes** | §3.1's gate rule; §6.2's heading "Regular matching (triggered by new-timestamp order)"; §8.5's two sentences, for firmness |
| **Dataset** | Random tape with a mix of validities, all resting at the gate, including resting icebergs (standard and ICEwPI) with positive hidden remainder, each within the slice margin $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$ |
| **Known result** | At the gate every resting order is removed, both validities alike; **no new trades**; I2 closes with every MW filled-or-expired; each iceberg's hidden remainder is classified expired by order semantics (I2), and no trade, book entry or FOK-visible volume remains for the closed contract. As in C8, no `iceberg_meta` slot state is asserted (OQ-20) |
| **Oracle kind** | exact-parametric |
| **Pass** | Empty book; trade count unchanged across the expiry call; I2 exact |
| **Does not certify** | Mid-session GTD lapse (§4.6) — unimplemented, recorded as a conformance gap |
| **Runtime** | ~3 min |
| **Notebook** | `C11_gct_expiry.ipynb` |

---

### C12 — Settlement and the cash leg

**Purpose.** Everything above certifies that the right trades happen; this certifies that the right
**money** changes hands. OMIE values a trade as quantity × price × duration, with duration fixed at a
quarter hour — the $\times d$ factor is a genuine departure from equity-market convention and one of the
assumption violations §10 tabulates. The strongest available check is that the cash leg sums to zero
across participants. Two halves make it non-trivial: each trade's $Q\cdot p\cdot d$ is recomputed
exactly in ticks against its row, and the engine's own per-participant settlements must sum to zero
within the tolerance derived from the settlement arithmetic. (Summing the rows in integers would be zero by
algebra — a ledger checked against itself, which rule 1 forbids; and the engine reports its settlement
as a float, so no integer settlement can be read from it.) The euro figure is
computed and reported but never asserted — the price decode is FMA-contracted and backend-unstable,
so a euro assertion would be flaky between CPU and GPU while proving nothing extra.

**Two claims deliberately narrowed.** §8.3's per-unit identity `agent_pnl = Σ unit_pnl` is
**untestable today**: `OMIEOrder` has no unit field and the settlement function is agent-level,
because the loader folds `(Agente, Unidad)` into a single identifier — so a simulator "agent" already
*is* a unit. And "zero-sum" holds of the **cash leg** only: §8.4 defines PnL as cash *plus* an
explicitly exogenous energy-value term, so economic zero-sum is not a claim this simulator can make.

**What this scenario cannot catch: a trade at the wrong price.** The engine computes settlement *from
its own trade rows*; there is no second, independent valuation anywhere in the system. If a trade were
booked at the wrong price, the trade row and the settlement would both carry that wrong price and
agree perfectly. So C12 certifies the **arithmetic** of settlement — $Q\cdot p\cdot d$, the sign convention, the
zero-sum cash leg, and that the reported settlement agrees with the rows written — and says plainly
that **mispricing is caught elsewhere**: by C2 (the resting-price rule, as an exact formula) and by
invariant I5 (every trade at its price-setter's own limit) on every scenario.

**A known hazard C12 must not fall into.** `compute_agent_pnl` expects a flat list of trades, shaped
(M, 6), but does not check it. Given the environment's per-step trade tape, shaped (steps, 2, 6), it
silently returns a wrong value — observed 0.0 (filed 2026-10-01, unscheduled, as
`OMIE-AGENT-PNL-SHAPE-GUARD-1`). A zero for every participant sums to zero, so the cash-leg check
would pass on nothing. C12 therefore always flattens the trades to (M, 6) before settling, as the
production callers do, asserts the shape it passes, and guards against a vacuous zero-sum in two
ways: the gross cash leg $\sum \lvert V\rvert$ over all trades must be positive (I9), and each
participant's reported settlement must equal an independent per-row recomputation of
$\sum (\pm Q\cdot p\cdot d)$ for that participant — which a mis-shaped call fails even when its output
is 0. (A per-participant "non-zero" check would be wrong: a legal self-match or an equal-price round
trip settles to exactly zero — domain review v3.0, V3-03.)

| | |
|---|---|
| **Certifies** | §8.1 $V = Q\cdot p\cdot d$ with $d = 0.25\ \text{h}$; §8.4's sign convention; §11.1 row *"Trade value = Q × p × d"*; OQ-18 for the x64 scoping |
| **Quotes** | §8.1's formula with the $d$ definition; §8.4's buy/sell convention **including its exogenous-energy-value sentence**, which is what bounds the zero-sum claim |
| **Dataset** | Random multi-participant tape, both sides trading, trades ≤ 80 per I8 |
| **Known result** | The engine's per-participant settlements sum to zero within the derived tolerance; $\sum \lvert V\rvert > 0$; each trade's value re-computed as $Q\cdot p\cdot d$ from its row and reconciled participant by participant; reported settlement equals the rows written |
| **Oracle kind** | exact-parametric, carried by I6 |
| **Pass** | I6(a) and (b) as restated in the battery; per-trade $Q\cdot p\cdot d$ recomputed exactly in ticks against each row; euro figures otherwise reported, not asserted |
| **Does not certify** | Mispricing (C2 and I5 own it); §8.3 per-unit granularity — unimplemented; §8.2 trade firmness — not represented |
| **Runtime** | ~3 min |
| **Notebook** | `C12_settlement_cash_leg.ipynb` |

---

### C13 — Anonymity and information visibility

**Purpose.** OMIE Rules §45.2.12 require the order book to be displayed on an anonymous basis, and
§46.1 restricts trade notification to the agent's own trades. This is not decoration: it bounds what
an RL agent is *allowed* to observe, and a leak here would be the most insidious defect in the whole
system, because it would make the agent look skilful while it was in fact reading information no real
participant can see.

The sharpest available formulation is metamorphic rather than structural: construct a book containing
a foreign iceberg's hidden remainder and a counterparty identifier, and assert the agent's
observation is **bit-identical** to the observation from a counterfactual book in which that hidden
remainder is absent and the counterparty identifiers are permuted. If the observation cannot tell
those two worlds apart, nothing in it can be leaking. Crucially the assertion must be
**side-conditioned on ownership**, not blanket: §7.3 and §9.2 explicitly permit an agent to see *its
own* full iceberg quantity, so a blanket hidden-quantity-invisibility claim would be wrong.

This scenario is also the engine-level twin of Track B's L0 leak control, and it is what makes L0
falsifiable: C13 proves the observation surface *cannot* carry the information, L0 proves no policy
profits from anything else.

| | |
|---|---|
| **Certifies** | §7.1 anonymous order book; §7.2 trade notification; §7.3 simulator modelling rules; §9.2 rows "LOB per contract", "Own orders", "Trade book" |
| **Quotes** | §7.1 *"…displayed on an anonymous basis, identifying solely those operations performed by the agent itself"*; §7.2 *"the result is to be notified to the agent in question"*; §9.2's "Own orders … showing full iceberg quantity", which is the ownership side-condition |
| **Dataset** | Random multi-participant tape with foreign resting icebergs, each within the slice margin $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$, plus the counterfactual twin (hidden remainder removed, identifiers permuted) |
| **Known result** | The observation is bit-identical across the two worlds wherever no tape iceberg fills a resting order of the RL agent on entry (the premise). Where one does, the two worlds differ exactly as §4.2.1 Rule 2 (1), (4) imply, and that difference is asserted instead: for an iceberg of total $T$ and peak $P$, with $f$ the agent's fill on its entry, $M$ the iceberg's swept quantity on entry (the agent's fill $f$ included), $R = T - M$ and $h = T - P$ (*amended 2026-10-22, `OMIE-CERT-RERUN-3` r4 K2 finding, certification-log entry 50: was "$R = T - f$", which holds only when nothing but the agent's order is swept, $M = f$*), the iceberg world A's visible quantity at that level exceeds the twin B's by $v_1 - (P - f)$ and A's hidden remainder is $R - v_1$, with $v_1 = \min(P - u, R)$, $u = M - \lfloor M/P \rfloor P$ and $M$ the iceberg's swept quantity on entry, the agent's fill $f$ included (§4.2.1 Rule 2 *Incoming iceberg* item (4); *amended 2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1`, user ruling 2026-10-20: was "by $\min(P, R) - (P - f) = \min(f, h)$ and A's hidden remainder is $R - \min(P, R) = \max(0, h - f)$"*); the observation and reward stay bit-identical before the entry step, and the done flags and the agent's own trade report throughout (*amended 2026-10-16 after the r2 SCOPE-CHANGE verdict, certification-log entries 35 and 45: was "The observation is bit-identical across the two worlds"*); own-iceberg full quantity **is** visible to its owner; `get_agent_trades(a)` returns exactly $a$'s fills with no counterparty identifier; the union over agents reconstructs the trade tape without duplication |
| **Oracle kind** | exact-parametric, carried by a metamorphic twin |
| **Pass** | Bit-identical observation where the premise holds, and the rule-implied difference exactly where it fails (*amended 2026-10-16: was "Bit-identical observation"*), computed with A's display $v_1 = \min(P - u, R)$, $u$ taken from the iceberg's swept quantity on entry including the agent's fill (*amended 2026-10-20, `OMIE-SPEC-ICE-INCOMING-AMEND-1`: was computed with $\min(P, R)$*); exact per-agent trade partition; ownership side-condition respected |
| **Failure implicates** | An observation leak — would invalidate every RL result by allowing skill that is really information |
| **Runtime** | ~3 min |
| **Notebook** | `C13_anonymity.ipynb` |

---

### C14 — Capacity edges, one part per ceiling

**Purpose.** Every array and every loop bound in a JAX simulator is fixed-size, so there are five hard ceilings. These
scenarios **certify that the documented behaviour at each ceiling is what actually happens** — they
are not fix requests, and that framing must survive into the certificate. The point is the
distinction between a *known, flagged, bounded* degradation and a *silent* one. Overflowing a book
and raising the full flag is an accepted limitation recorded in `book-depth-cap-finding.md`;
overflowing it silently and losing orders would be a defect. Hitting the iteration bound must set the
truncation flag and **still** conserve quantity: the flag may be the only observable consequence,
never a wrong trade list.

This is the **only** scenario permitted to reach a ceiling; every other asserts the opposite through
I8. Together the two halves let the certificate say both "the engine is correct in its operating
envelope" and "its behaviour at the envelope's edge is known, and flagged except where C14a′ records an
unflagged case". Note that these caps have
**no market-spec section at all** — §11.2 is the modeling-choices list and has nothing to do with
array capacities — which is precisely the honest framing: C14 certifies *implementation limits*, and
each is a **departure from** §6.1/§6.2, not an instance of them.

| | Part | Dataset | Known result |
|---|---|---|---|
| **C14a** | Book depth | >100 resting on one side | Full flag raised; worst-price level wiped; no trade manufactured; evicted quantity accounted in its own I2 bucket |
| **C14b** | Trade ring | >100 trades | Documented behaviour: once the ring is saturated, every new trade lands in its **last row**, so a burst keeps only its final trade. The part certifies that this is what happens and that it is detectable, never that it is correct |
| **C14c** | Outer matching passes | Pass count > 100 (`_TRIP_BOUND`; *corrected 2026-10-07, was "Hop count >200": `max_match_iterations` = 200 never binds*) | Loop terminates, truncation flag set, and that flag is the **only** observable — I2 still closes. The iteration count itself cannot be read after the fact, so the part is driven by constructions whose hop count is known analytically. Those constructions keep every iceberg within the slice margin $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$ and assert `truncated == False` after the last registration, before the triggering aggressor, so the flag is attributable to the iteration ceiling and not to C14e's |
| **C14a′** | Regen-slice rests on a full side | Iceberg regeneration into a side already at 100 | Documented limit: a regenerated slice that rests **does not raise** the full flag, so this eviction is unflagged. **Recorded, not certified**: the part observes and reports the unflagged eviction (quantity read from the book diff, rule 2) as a known engineering gap outside the certified surface; it passes if the eviction is exactly the documented one and fails if any other quantity is lost. This is why I8 never relies on the flag alone |
| **C14d** | Iceberg slots | >1024 concurrent icebergs | Bounded per documented behaviour; no side-channel corruption |
| **C14e** | Slices per iceberg | A boundary pair of resting icebergs that never trade, each in a fresh book: $\lceil \mathit{total}/\mathit{peak} \rceil = 65$ and $= 66$ | Documented behaviour (`jax_constants.py` §B2; the capacity-validity precondition): the 65-slice iceberg leaves `truncated == False`, and the 66-slice iceberg sets it **at registration**, with no aggressor and no trade. In both, the book holds exactly the visible peak and I2 closes, with the hidden remainder in its own bucket. What an over-ceiling iceberg does **afterwards** is **recorded, not certified**: its regeneration past the 64th slice, and the fill-or-kill pre-check, which projects only 64 hidden slices. §B2 names this hazard but documents no exact behaviour for it |

| | |
|---|---|
| **Certifies** | No market-spec section. Grounded in `book-depth-cap-finding.md`, OQ-10's 2026-06-11 accepted-limitation ruling, OQ-15 (ring occupancy), OQ-20, and the `OMIE-ICE-INCOMING-FILL-REPORT-1` close-out record in `sprint-status.md` (2026-09-28, `1f4755a`) for C14a′, C14b and C14c's observability limits; `jax_constants.py` §B2 (`MAX_SLICES_PER_REGEN`) for C14e |
| **Oracle kind** | exact-parametric on documented behaviour, plus the battery except: I8 (all parts); I1, I6 and row-derived I2 terms on C14b (OQ-15); I3 on C14c (truncation is a Rule-4 departure by construction). Each excluded invariant is reported as **expected-violated**, and its violation is itself asserted, so the exclusion cannot hide an unrelated failure |
| **Pass** | Per part, an exact match against the documented behaviour in the table above; each excluded invariant violated exactly as expected; every other invariant holds |
| **Failure implicates** | A silent capacity overflow — strictly worse than the documented degradation |
| **Runtime** | ~15 min |
| **Notebook** | `C14_capacity_edges.ipynb` |

---

### C15 — Randomised fuzz across the parameter space

**Purpose.** This is the scenario that converts "the simulator is correct" from a list of anecdotes
into a defensible claim, and it is the direct answer to the request for randomly generated datasets
at scale. Every scenario above fixes a construction and randomises within it; C15 randomises the
construction itself, precisely to reach the interaction bugs that only appear when an iceberg
regenerates into a fill-or-kill pre-check during a tie.

**How the dataset is generated.** One fuzz *draw* is a whole trading session, built in four steps
from a single integer seed.

1. **Pick a regime.** The seed selects one cell of a grid: order-type mix (NON only; NON+IOC; +FOK;
   +ICE; +ICEwPI), crossing probability (how aggressively orders are priced, 0.05 to 0.6), tie
   density (whether many orders share a timestamp), and session length (32, 96 or 160 orders). The length stops at 160 for a structural reason: a tape
   carries no cancellations, so an order that does not cross rests all session, and at 100 resting
   orders per side a longer session cannot exist without triggering the documented eviction — longer
   sessions are C14's subject, not C15's. The
   grid is swept systematically rather than sampled, so every cell gets its share of seeds and no
   regime is accidentally untested. Book depth and iceberg-slot pressure are swept as **independent
   axes**, not via one "size" knob, because they are independent in the implementation.
2. **Draw the orders.** Within the regime, each order's side, price distance from the touch, size,
   type, and — for icebergs — peak and price increment are drawn from the regime's laws (the peak law
   keeps every iceberg within the slice margin $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$). Arrival
   times come from a Poisson process and are sorted. Some draws deliberately give two orders the same
   owner, so ownership blindness (M6) is exercised at scale.
3. **Enforce the capacity bounds.** The draw is rejection-sampled against the I8 margins — all five,
   independently — so a fuzz session never reaches a ceiling. A draw that cannot satisfy them within
   a retry budget is discarded and re-seeded, and **the discard rate is reported**: a high rate would
   mean the regime is only reachable in the degraded envelope, which is itself a finding.
4. **Emit.** The tape is written in the 13-column integer format the engine consumes, with the seed,
   regime and a content digest recorded alongside it.

**How it is certified.** No per-draw answer can be computed here — that is the entire point of
fuzzing, and it is why the oracle is the invariant battery rather than a formula. Each session runs
through the engine twice (eager and jitted, asserted identical), and then every invariant I1–I9 and
every metamorphic relation M1–M6 is evaluated on it. The pass criterion is **zero exceptions across
every draw**: one violation is a certification failure, not a statistic. Alongside, aggregate matched
volume is compared against a pre-registered closed-form band derived from the regime's crossing
probability and size law — a much weaker check, reported as a sanity signal and never used alone as a
pass criterion, because a band that fails invites widening.

**Why a shrinker is not optional.** A violation on a thousand-order tape tells you almost nothing:
the defect could be any of the thousand orders or any interaction between them. The shrinker takes
the failing seed and bisects — repeatedly truncating the tape and re-running — until it finds the
shortest prefix that still violates the same invariant. What comes out is typically three to ten
orders a human can read in a minute, and it is emitted as a frozen regression fixture so the defect
can never silently return. Without this step a fuzz suite produces failures nobody can act on, and
experience says such a suite gets disabled.

Three variations, run in this order:

| Variation | Scope | Draws | Purpose |
|---|---|---|---|
| **C15a** — smoke | NON and IOC only, short sessions | 200 | Proves the harness, the shrinker and the invariant wiring work before spending time on hard regimes |
| **C15b** — gate | Full grid, all order types | 2 000 | The figure the merge gate runs; the standing regression |
| **C15c** — deep | Full grid, weighted toward iceberg and FOK interaction | 50 000 | Thesis-grade evidence; runs nightly, not in the gate |

| | |
|---|---|
| **Certifies** | §4.5, §6.1, §6.2 jointly, under interaction; §6.6 at scale via M6 |
| **Known result** | Zero exceptions across I1–I9 and M1–M6, plus the pre-registered aggregate band |
| **Oracle kind** | invariant / distributional |
| **On failure** | Shrink to the minimal violating prefix; emit a frozen regression tape under `tests/omie/fixtures/cert/` |
| **Runtime** | C15a ~2 min; C15b ~12 min; C15c ~5 h nightly |
| **Notebook** | `C15_randomised_fuzz.ipynb` |

This scenario also discharges a standing debt. The `P3-PERF-REGEN-MASK-1` failure record requires a
persisted real-tape golden — a 709→795 fill divergence on contract 81 once passed every gate for want
of one — and that file does not exist, since `tests/perf_sandbox/` holds only stale bytecode. C15's
frozen fixtures plus one real-contract golden restore it.

---

### C16 — Multi-contract isolation

**Purpose.** OMIE runs 96 quarter-hourly contracts a day and each has **its own** order book; there is
no matching across them. Since the environment carries tapes as one stacked array and slices per
contract, a leak between them is an entirely plausible indexing error — and a damaging one, because
it would manufacture trades that could not exist. The scenario takes the strong form: rather than
merely checking that no cross-contract trade appears, it runs each contract **in isolation** and
requires per-contract trade lists to be identical. That subsumes the weak check and also catches
state bleeding that produces no visible cross trade. Prices are deliberately overlapped so a leak
*would* match, and gates are staggered so one contract's closure cannot disturb another's.

| | |
|---|---|
| **Certifies** | §6.4 no cross-contract matching; §2.1 the 96-contract structure; §2.2 *"Each contract has its own independent order book"*; §8.1, since $d$ is a contract property |
| **Quotes** | §2.2 "Each contract has its own independent order book"; §6.4 "Orders on different contracts are never matched against each other"; §2.1 *"the quarter-hourly product … will be composed of 96 contracts"* |
| **Dataset** | `n_tapes ≥ 3` with distinct contract ids, overlapping price ranges, staggered gates; any iceberg on any tape within the slice margin $\lceil \mathit{total}/\mathit{peak} \rceil \le 64$ |
| **Known result** | Per-contract trade lists identical to running each contract alone |
| **Oracle kind** | exact-parametric (equality against the isolated runs) |
| **Pass** | Per-contract exact equality |
| **Failure implicates** | Book keying or tape-set slicing |
| **Runtime** | ~4 min |
| **Notebook** | `C16_multi_contract_isolation.ipynb` |

---

### C17 — The cancellation-fidelity gap

**Purpose.** This scenario certifies nothing about the matching engine. It **measures a known
limitation of the replay construction**, and it is arguably the most decision-relevant number the
whole ladder produces. It belongs in the certificate as an **appendix on replay fidelity, explicitly
outside the certified surface**.

The problem is OQ-10. OMIE's public data is a *submission log*: it records orders as sent, with no
cancellations, no modifications, no lifecycle events. So when the loader replays a historical session
it cannot know whether an order that never traded was withdrawn two minutes later or sat there until
the gate — the three possible fates are **observationally equivalent in the public file**. The
loader's heuristic assumes the order rested until gate closure. On a busy contract this leaves
liquidity in the simulated book that was, in reality, long withdrawn. OQ-10 calls it phantom
liquidity. OQ-25 attributed two consequences to it as *one mechanism*: the replay over-matches
relative to the real trade file, and an agent's own resting orders are starved because phantom
residuals out-prioritise them at equal price under §6.1. The first is an accepted limitation. The
second is **now in doubt**: the passive-fill figure behind it was measured while the environment
stamped agent orders at the *front* of the queue (corrected by `OMIE-AGENT-TIME-PRIORITY-1`, with a
dated correction to OQ-25 in `OMIE-AGENT-TIME-PRIORITY-1-DOC-1`; both merged 2026-10-01). Neither has ever been **measured
under known ground truth**, because on real data the ground truth does not exist — which makes this
scenario more useful, not less.

Synthetic data closes that circle. The generator knows which orders it cancelled, because it
cancelled them. So the same session runs twice — once with cancellations applied (the truth), once
with them stripped and the heuristic applied (what the loader does to historical data) — and the
difference is the distortion, isolated and controllable.

**The construction.** For each draw the generator emits a session with explicit cancellations at a
controlled rate $c$. Two tapes are derived: the **truth tape**, and the **heuristic tape** with every
cancellation row deleted, reproducing exactly the information loss the published data imposes. Both
run through the identical engine, paired, across $c \in \{0, 0.1, 0.25, 0.5, 0.75\}$, with $c = 0$ as a
built-in null where the two tapes are identical and every difference must be exactly zero. One design
constraint: the cancellation process must be **exogenous** — driven by drawn lifetimes, never by book
state. A state-dependent cancellation process would make this a reactive-flow model, which OQ-25
rules a separate and distinct build.

**What it measures,** per cancellation rate:

- **Over-match** — matched volume and trade count under the heuristic minus ground truth. OQ-10's
  headline gap.
- **Phantom depth** — time-averaged resting volume that ground truth says should not exist.
- **The passive fill-rate delta** — with a reference agent posting resting orders identically in both
  runs, the difference in its fill rate. This measures the starvation claim directly, under ground
  truth, and measuring only the over-match would measure half the object. The reference agent's
  orders must be stamped from the book's arrival counter, so that they queue behind earlier arrivals,
  as `OMIE-AGENT-TIME-PRIORITY-1` fixes in the environment. If the reference agent is driven through
  the environment, **C17 depends on that story**, a dependency met on 2026-10-01. Otherwise the reference agent would jump the queue,
  and the delta would measure the old defect instead of the phantom wall. If driven through the
  environment, it also reports OQ-25 CORRECTION (b)'s residual R1 (later non-agent arrivals ranking
  ahead of the resting agent at its own price) alongside the delta.
- **Price-path distortion** — touch and VWAP divergence, because the phantom wall also mis-sets the
  touch that Track B's touch-relative action geometry is defined against.

**What it cannot claim — and these bounds must survive into the certificate.** It cannot calibrate
the *real* over-match, because the true cancellation process is unobservable by construction; the
output is a **sensitivity curve over an assumed $c$**, never "the June-2025 over-match is X %". It
cannot certify the engine: it varies the tape while holding the engine fixed, so it says nothing
about §6.1/§6.2 conformance. It cannot mandate a fix: OQ-22 rules the gap acceptable-for-v1 with a
banner, and OQ-25 rules the fix locus is the tape and replay side only — never action semantics,
crossing caps, priority boosts or reward reshaping. It cannot validate the heuristic by agreement
with history, since it establishes ground truth on synthetic data only. And it cannot measure the OQ-10
EXTENSION (post-IDA re-quote double-counting): the synthetic sessions contain no IDA gap.

**Why it earns its place.** It tells you how much the rest of the programme's historical results
should be trusted, and it is the natural input to any future decision about the fate-of-residual
heuristic that the R-ladder's R3 assumes.

| | |
|---|---|
| **Measures (does not certify)** | The replay heuristic's distortion; OQ-10, OQ-22, OQ-25 |
| **Quotes** | OQ-10 on the three fates being observationally equivalent; OQ-25's verdict attributing starvation to phantom residual out-prioritising the agent under price-time priority, **together with its CORRECTION of 2026-10-01**, landed by `OMIE-AGENT-TIME-PRIORITY-1-DOC-1` |
| **Dataset** | Paired synthetic sessions at $c \in \{0, 0.1, 0.25, 0.5, 0.75\}$, 200 draws each, exogenous cancellation lifetimes |
| **Known result** | At $c = 0$ the two runs are **byte-identical** — the built-in null. For $c > 0$ the over-match ratio is bounded below by 1 and increases monotonically in $c$; every reported difference's **sign** is known in advance, its magnitude is the measurement |
| **Oracle kind** | invariant (the null and the monotonicity) plus reported measurement |
| **Pass** | The $c = 0$ null holds exactly; monotonicity holds; the curve is produced with confidence intervals |
| **Runtime** | ~10 min |
| **Notebook** | `C17_cancellation_fidelity_gap.ipynb` |

---

### Out of scope, and conformance gaps

An unimplemented behaviour absent from this list reads as though it were certified, so the list is
exhaustive and the categories are kept distinct.

**Out of scope because unimplemented:**

- §3.3 rounds and the two-phase IDA suspension (the capacity halt and the full freeze), and the
  SUSPENDED status *path* — the status exists but nothing drives transitions; C0 certifies the
  `!= OPEN` collapse as observed behaviour (conformant for submission).
- §3.3 (v2.10) survival of resting orders through the IDA clearing window and the reopen batch match
  (OQ-28) — no engine path; the replay carries the book across the gap as a recorded simplification.
- §3.1 (v2.10) hour-aligned GCT(k) schedule — a loader/environment property (`gct_minutes_from_got`,
  OQ-27), not an engine rule; no Track A scenario derives GCT from the slot.

- §3.4 market halts, hibernation and manual reactivation.
- §4.3 block orders and §4.4 baskets, and §4.5 AON. Note that *independent* baskets are definitionally
  just sequential submissions, so their semantics are trivially satisfied; what is absent is any
  basket construct.
- §4.6 mid-session GTD lapse (see C11) — **a real conformance gap**.
- §5.4's guarantee/operating-limit and TSO-unavailability rejection conditions.
- §8.3 per-unit settlement — no unit field exists. **A real conformance gap**, since §13.4/§13.5
  require the loader to preserve Unidad.
- §8.2 trade firmness (not represented), §8.5 imbalance resolution (explicitly outside the
  simulator), §2.4 notification thresholds (warnings only).
- §9.2 trade side (v2.14 addendum) — spec only, by user decision: `get_agent_trades` reports price,
  quantity and time, not side. Not certified (C13 F16).

**Known inconsistencies in `market-spec.md` itself** — both in the IDA area above, both outside this
ladder's scope, and both ruled on in the domain review v3.0 (section 12):

- **§3.2's IDA table** times IDA1/IDA2 "(D−1)" and IDA3 "(D)" but scopes all three "of D+1", mixing
  two day frames: IDA1 and IDA2 run on D−1 and IDA3 on D, all for delivery day D (OMIE Instruction
  1/2024 §6, §8, §9; OMIE Rules Annex 2 §1 Figure 1, whose "D+1" is relative to the session day). The
  quarter-hour periods (1–96; 49–96 for IDA3) are correct under the current Rules (Annex 2 §1
  Figure 1); Instruction 1/2024's hourly periods (1–24, 13–24) predate the 15-minute MTU (R-2).
- **§3.3's phase-2 citation** cites a Matching Algorithm clause about capacity-update batch matching
  for the IDA-window suspension (R-1; correct rule, wrong source).

**Implemented, but not as the rule says — conformance gaps the ladder records:**

- None open. §13.6 **PPD sign rule** — closed 2026-10-07: the validator did not enforce it (OQ-23,
  CORRECTION + CLOSURE NOTE (a)); `OMIE-ICEWPI-SIGN-REJECT-1` makes it reject a wrong-sign ICEwPI
  (`market-spec.md` v2.14 §5.4), and C9 asserts the rejection.

**Engineering limits the ladder works around rather than certifies** (all pre-existing, all recorded
when the iceberg fill-report defect was fixed, none a market rule):

- A regenerated iceberg slice that rests on a full side does **not** raise `book_full` (C14a′). I8
  therefore relies on the generator's margins, not on the flag.
- The matching-iteration count is **not readable** after a run. I8 reports that axis as unobserved,
  and the generator's analytic hop bound is the guarantee.
- A **saturated trade ring** collapses every further trade into its last row (C14b). The ≤80-trade
  margin keeps every other scenario well clear of it.
- An iceberg needing more than 64 slices raises the shared `truncated` flag **at registration**, even if
  it never trades (C14e). I8 therefore relies on the generator's slice margin, and a raised flag
  does not say which ceiling was reached.
- In the truncated regime (C14c only), the post-loop re-submit of a NON order does not regenerate
  icebergs, leaving a consumed slot. It is reachable only when truncation is already flagged.
- The order-book tape's empty-slot detector reads the right value for the wrong reason (it indexes
  a different column that happens to hold `-1` on empty rows). It is harmless today and noted so that a
  future layout change does not break it silently.

**Out of scope by decision, though implemented:**

- §5.3 **order modification** and its timestamp reset. `modify_order` exists (as cancel-then-resubmit)
  and §5.3 calls the priority reset *"a hard rule from the XBID system"*, but modification is
  excluded from this plan. So one hard rule goes uncertified, and the certificate says that in those
  words rather than by omission. If it is ever brought back, it is a single scenario slotting
  immediately after C4, and the test is already designed: modify an order sitting ahead of a
  same-price peer, aggress for one fill, and require the **peer** to fill.

**Out of scope because not applicable:**

- §6.3 batch matching and §6.5 cross-border capacity — single-zone simulation; batch matching is
  triggered only by a cross-border capacity update.
- §13.x loader and data-feed semantics — a data-layer concern, not a matching-engine one. One
  exception is pulled *into* scope: §13.6's PPD sign rule — drawn sign-valid in C9, whose wrong-sign
  draws assert its rejection (§5.4, v2.14).

**Bounded, not excluded:** OQ-10 phantom liquidity and the starvation OQ-25 attributed to it are
properties of the replay heuristic on *real* data. The mechanism OQ-25 proposed is §6.1 price-time
priority **operating correctly on a wrong book** — so the engine behaviour beneath it *is* in scope,
and C3 certifies exactly that rule — though whether that mechanism explains the measured starvation
is now open (see C17). The
ladder certifies the mechanism and disclaims only the tape — and C17 quantifies the disclaimer
instead of merely stating it.

**Total estimated Track A runtime:** ~1 h 10 min for the full ladder excluding C15c, which runs
nightly. The whole ladder fits inside one working session, which is what makes re-running it after
any engine change practical.

---

## Notebook architecture

Each scenario is **fully contained in one notebook**, all in `notebooks/certification/`, named
`C<nn>_<slug>.ipynb` so the folder listing is the ladder in order. A notebook is self-contained and
runnable on its own at any time.

Fixed nine-section skeleton, identical across all eighteen so that reading one teaches you the rest:

1. **Purpose** — the rationale paragraph from the spec document, verbatim.
2. **The rule, quoted** — the governing passages **quoted verbatim** in a blockquote, each with its
   section number and the spec version quoted from. This makes the notebook self-contained as an
   argument: read the rule, then the expectation derived from it, without leaving the page. A helper
   renders these from the spec file at execution time and **fails the notebook if the quoted text no
   longer matches the source, or if the quote is not under the section label it is cited as**, so a
   spec revision cannot silently orphan a certification. Matching must be robust to the source's
   markdown emphasis, an elided quote's fragments must appear in order within a bounded span, and an
   empty quote is refused — three ways the first attempt's checker was defeated. Matching is
   **punctuation- and Unicode-strict**: quotes are byte-copied from source (a minus sign U+2212 is not
   an ASCII hyphen). Each quote names its **source document** — `market-spec.md` (versioned by its
   Status line), `open-questions.md` (versioned by the OQ id plus the dated block quoted, e.g. "OQ-25
   CORRECTION 2026-10-01") or `scenario-ladder-review.md` (versioned by its Status line) — and is
   checked against that file; regulatory-corpus text is shown as a citation and is not machine-checked.
   Labels are section-level (§5.2, or §6.2 item N for numbered lists); bullet positions are never used
   as labels.
3. **Configuration** — one visible parameter cell (draws, seed, tick grids, capacity margins).
4. **Dataset** — builds the tape, then *displays* it: order rows in human-readable units beside the
   raw integer encoding, and a book-shape plot. This is where "the dataset is known" becomes visible
   rather than asserted.
5. **Known result** — the expectation stated in markdown as a formula, then computed in code, side by
   side so a reviewer can check one against the other.
6. **Run** — submits through the engine, eager and jitted, asserting the two agree.
7. **Certification** — the oracle comparison, then the invariant battery, as a pass/fail table with
   one row per invariant and per assertion, plus the capacity-headroom report and the draw-discard
   rate. This table is the certificate.
8. **Verdict** — emits `results/C<nn>.json` (draws, seeds, generator digest, per-assertion outcomes,
   capacity peaks, discard rate, measured wall-clock) and prints one line: `PASS` or `NOT CERTIFIED`.
9. **Why this certification failed** — written only when the result is `NOT CERTIFIED`, and always
   the last thing in the notebook. It explains the failure in plain words for a reader who did not see
   the run, in a fixed order:
   1. **What failed** — which assertion or invariant, on which draw (seed), expected versus observed,
      in integer ticks and in readable units.
   2. **The smallest failing case** — the shrunk counterexample where one exists, as a short table of
      orders a human can read in a minute, also saved as a frozen fixture.
   3. **Which rule it concerns** — the quoted market-spec passage the expectation came from.
   4. **Where the fault most plausibly lies** — the simulator, the expectation or the reading of the
      rule, the judging code, or the run itself (configuration, capacity, crash). Written as a
      **hypothesis with its evidence, never as a verdict**; it only helps the human decide quickly.
   5. **What it affects downstream** — the later scenarios that rely on the same rule or component, so
      their results can be read with it in mind.

   The same explanation is copied, as one paragraph, into the certification-log entry. A notebook that
   crashes before it can write this section is still recorded: the orchestrator writes the log entry
   from the traceback and marks the explanation *missing*.

**How this coexists with the locked-test convention.** The notebook is the primary artifact and where
the logic lives; it is **not** duplicated into a parallel pytest file, which would guarantee drift.
Instead: the merge gate executes notebooks headless by driving **`nbclient`** directly, and carries a
control test proving it can fail. (`pytest --nbmake` was tried and reported a **pass** for a notebook
whose last cell raised; since the ladder asks for a human only on failure, a gate that can silently
pass is disqualifying.) The **audit twin stays a `.py` file** (`tests/omie/certification/test_C<nn>_audit.py`) that
re-derives the expectation independently from `market-spec.md` and asserts against the emitted JSON;
and notebook outputs are stripped before commit, so the repository stores code and prose while the
durable record of a run is the emitted JSON plus the certificate.

**New dependencies** (none currently in `.venv`): `jupyter`, `nbformat`, `nbclient`, `nbstripout`.
The `nbstripout` git filter must actually be **installed**, not merely named in `.gitattributes` (an
entry naming an uninstalled filter is inert and strips nothing), and the install is scoped to this
folder rather than applied repository-wide, which is what the bare command does. Authoring and test
only; they do not enter the training path.
