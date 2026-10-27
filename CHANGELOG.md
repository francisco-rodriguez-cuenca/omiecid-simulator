# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] — Unreleased

Initial release of OmieCID-Simulator.

- JAX limit-order-book engine for the Iberian continuous intraday (CID) market
  operated by OMIE: per-contract order books, order validation, limit, iceberg,
  IOC and FOK orders, cancellation, GCT expiry, trade extraction and settlement.
- `Session` interface: reset/step over a market tape, with agent order
  injection and agent P&L.
- OMIE tape loading and caching, and a synthetic tape generator.
- Certification notebooks C00–C16 with recorded results, and the test suite.
- Packaging (`pyproject.toml`, `constraints/verified.txt`) and legal metadata
  (`NOTICE`, `AUTHORS`, `CITATION.cff`) under the inherited Apache-2.0 licence.
