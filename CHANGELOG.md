# Changelog

All notable changes to this project are documented in this file.

## [0.1.2] — 2026-11-06

The first version published on PyPI. The package code is the same as in 0.1.1.

### Added

- PyPI: the package installs with `pip install omiecid-simulator`. The README
  shows this first, with an optional constraints file for the verified
  versions, and keeps the clone path for the tutorials, the certification
  notebooks and the tests.
- Documentation site at <https://omiecid-simulator.readthedocs.io/>, built from
  the guides, the tutorials and an API reference.

### Changed

- Packaging metadata: project links, a bounded `requires-python`, and
  classifiers for the Python versions and the operating system it was
  verified on (Linux). On PyPI, the README's relative links point to the files
  on GitHub at the release tag.
- Source distribution: it no longer includes the nested README files from
  other directories.
- Tutorials: small touch-ups to the code, the text and the setup cell; the
  printed results are unchanged apart from measured times and cache counts.

## [0.1.1] — 2026-11-04

### Added

- Tutorials: nine executed notebooks in `tutorials/` (T01–T09), from a first
  contract to running many sessions fast, with an index, learning paths for
  three kinds of reader, and a glossary in `tutorials/README.md`. Every
  notebook is committed with its outputs and runs on the CPU.

### Removed

- The two real-data demo scripts. Tutorial T03 now covers downloading, loading
  and replaying real OMIE data. The benchmark demo stays.

### Changed

- Documentation tidy-up: one installation path in the README, from a clone to
  a passing check, which every other page links to; the published certification
  results linked; abbreviations spelled out; each topic and caveat described on
  one page; links to the tutorials from the README and the guides.
- Performance and Sessions: on a CPU, a `jax.vmap` batch of sessions gives the
  same results as running them one at a time but is much slower; the guides
  now recommend compiling one session and calling it in a loop. Whether a batch
  is faster on a GPU has not been measured.
- Certificate: an editorial amendment (Appendix A) removes the description of a
  scenario outside the release. No scenario section, verdict, signature or
  results digest changed.

### Fixed

- The quick check (`pytest -m "not slow"`) no longer fails depending on the
  order in which the tests run: the two tests that count compiled programs of
  the order-submission code now clear its cache first.
- `generate_contracts()` sets `gct_minutes` to the hour-aligned gate closure.

### Known issues

- The `--keep-zips` option of the download script has no effect: the downloaded
  archives are always kept, in a `.zips` folder inside the output folder. The
  Data page says so.

## [0.1.0] — 2026-10-28

Initial release of OmieCID-Simulator.

- JAX limit-order-book engine for the Iberian continuous intraday (CID) market
  operated by OMIE: per-contract order books, order validation, limit, iceberg,
  IOC and FOK orders, cancellation, expiry at gate closure, trade extraction
  and settlement.
- `Session` interface: reset/step over a market tape, with agent order
  injection and agent P&L.
- OMIE tape loading and caching, and a synthetic tape generator.
- Certification notebooks C00–C16 with recorded results, published as the
  pre-release `certification-results-2026-10`, and the test suite.
- Packaging (`pyproject.toml`, `constraints/verified.txt`) and legal metadata
  (`NOTICE`, `AUTHORS`, `CITATION.cff`) under the inherited Apache-2.0 licence.

[0.1.2]: https://github.com/francisco-rodriguez-cuenca/omiecid-simulator/releases/tag/v0.1.2
[0.1.1]: https://github.com/francisco-rodriguez-cuenca/omiecid-simulator/releases/tag/v0.1.1
[0.1.0]: https://github.com/francisco-rodriguez-cuenca/omiecid-simulator/releases/tag/v0.1.0
