# OmieCID-Simulator

A JAX-based Python package that reproduces and analyzes order matching in the Iberian continuous
intraday (CID) market operated by OMIE.

Install it from PyPI (the other ways to install are under [Installation](#installation) below):

```bash
pip install omiecid-simulator
```

- The [project overview](../README.md) lists what the package contains and runs a first session.
- [Validation and evidence](evidence.md) states what was certified and how, with links to the
  certificate and the certification notebooks.
- [Citing](citing.md) gives the citation.
- The source is on [GitHub](https://github.com/francisco-rodriguez-cuenca/omiecid-simulator).

## Installation

```{include} ../README.md
:start-after: "## Installation"
:end-before: "## Quickstart"
:relative-docs: docs/
```

## Contents

```{toctree}
:caption: Overview
:maxdepth: 1

Project overview </README>
```

```{toctree}
:caption: User guide
:maxdepth: 1

market-model
modelling-assumptions
sessions
data
performance
limitations
testing
```

```{toctree}
:caption: Tutorials
:maxdepth: 1

/tutorials/README
/tutorials/T01_first_contract
/tutorials/T02_order_types
/tutorials/T03_real_data
/tutorials/T04_synthetic_flow
/tutorials/T05_session
/tutorials/T06_writing_an_agent
/tutorials/T07_several_participants
/tutorials/T08_overseers_view
/tutorials/T09_scaling
```

```{toctree}
:caption: Validation and evidence
:maxdepth: 1

evidence
validation
```

```{toctree}
:caption: Reference
:maxdepth: 1

api
Changelog </CHANGELOG>
citing
```
