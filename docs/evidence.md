# Validation and evidence

The matching engine (`omiecid_simulator/omiecid/`) is **certified** against a ladder of 17 scenarios,
C00–C16, and all 17 are CERTIFIED. The [certificate](omiecid/certificate.md) is dated 2026-10-26,
amended 2026-10-27 and 2026-10-29. The evidence documents and the notebooks are linked here on
GitHub, at the release tag; they are not reproduced on this site. [Validation](validation.md)
explains the method in more detail.

## What was certified, and how

Each scenario isolates one behaviour of the market, quotes the rule it certifies from the
[market specification](omiecid/market-spec.md) or a recorded ruling in
[open-questions.md](omiecid/open-questions.md), builds its own data with the synthetic tape
generator, and compares the engine against a result known in advance. A scenario combines three
kinds of evidence:

- **goldens**: expected results computed independently of the engine, either by hand for one fixed
  input or as a closed form in the parameters of a randomly drawn construction;
- **conservation invariants**: accounting identities that must hold after every run, whatever the
  input;
- **metamorphic invariants**: required relations between two related runs.

Every scenario runs the engine twice, eagerly and compiled with `jax.jit`, and requires the two
records to be identical. Each scenario also has an independently written audit test. The
[scenario ladder](omiecid/scenario-ladder.md) specifies the scenarios and the method.

## What was not certified

The certificate states what is not certified as carefully as what is: the
[conformance gaps](omiecid/certificate.md#conformance-gaps) (behaviours the specification
requires that the simulator does not implement, or does not implement as the rule says), the
[engineering limits](omiecid/certificate.md#engineering-limits-the-ladder-works-around) and the
[residual risk](omiecid/certificate.md#residual-risk). Chiefly, there is no independent second
engine, so a shared misreading of the market specification would survive all three kinds of
evidence.

The `Session` built on the engine is **verified, not certified**: a parity check compares it byte
for byte with the research environment it was extracted from, apart from one intended, exactly
checked difference (the self-match fix). See
[Validation](validation.md#the-session).

## The evidence documents

- [certificate.md](omiecid/certificate.md): one signed section per scenario, the conformance gaps,
  the engineering limits and the residual risk;
- [certification-log.md](omiecid/certification-log.md): every certification run, in order;
- [market-spec.md](omiecid/market-spec.md): the market rules the engine implements;
- [open-questions.md](omiecid/open-questions.md): the recorded rulings where the regulation is
  silent or ambiguous;
- [scenario-ladder.md](omiecid/scenario-ladder.md) and
  [scenario-ladder-review.md](omiecid/scenario-ladder-review.md): the ladder specification and its
  domain review.

The [evidence directory guide](omiecid/README.md) explains the identifiers used in these documents.

## The certification notebooks

One notebook per scenario, 17 certification notebooks in all, stored without outputs. The
[notebooks guide](../notebooks/certification/README.md) explains their structure and how to re-run
them. The results of the release run are published as the pre-release
[`certification-results-2026-10`](https://github.com/francisco-rodriguez-cuenca/omiecid-simulator/releases/tag/certification-results-2026-10).

| ID | Scenario |
|---|---|
| C00 | [Inertness and the contract-status gate](../notebooks/certification/C00_inertness_and_status.ipynb) |
| C01 | [Order validation and rejection](../notebooks/certification/C01_order_validation.ipynb) |
| C02 | [The single crossing pair](../notebooks/certification/C02_single_crossing_pair.ipynb) |
| C03 | [Price-time priority, ties, and ownership blindness](../notebooks/certification/C03_priority_ties_and_ownership.ipynb) |
| C04 | [Cancellation](../notebooks/certification/C04_cancellation.ipynb) |
| C05 | [Partial fill and the resting residual](../notebooks/certification/C05_partial_fill_residual.ipynb) |
| C06 | [Multi-level walk and volume-weighted price](../notebooks/certification/C06_multi_level_walk_vwap.ipynb) |
| C07 | [Immediate-or-cancel](../notebooks/certification/C07_ioc.ipynb) |
| C08 | [Iceberg regeneration](../notebooks/certification/C08_iceberg_regeneration.ipynb) |
| C09 | [Price-incremented icebergs](../notebooks/certification/C09_icewpi_price_drift.ipynb) |
| C10 | [Fill-or-kill, in three parts](../notebooks/certification/C10_fok_and_hidden_volume.ipynb) |
| C11 | [Expiry at gate closure](../notebooks/certification/C11_gct_expiry.ipynb) |
| C12 | [Settlement and the cash leg](../notebooks/certification/C12_settlement_cash_leg.ipynb) |
| C13 | [Anonymity and information visibility](../notebooks/certification/C13_anonymity.ipynb) |
| C14 | [Capacity edges, one part per ceiling](../notebooks/certification/C14_capacity_edges.ipynb) |
| C15 | [Randomised fuzz across the parameter space](../notebooks/certification/C15_randomised_fuzz.ipynb) |
| C16 | [Multi-contract isolation](../notebooks/certification/C16_multi_contract_isolation.ipynb) |
