# API reference

The public modules of the package, with their classes, functions and docstrings as written in the
source. The reference is generated from the source files without importing them. The inherited
order-book core and the internal helper modules are not listed.

## Session

The main entry point: a session runs one contract's order book on a tape of orders and lets an
agent submit its own orders (`Session`, `AgentOrder`, `AGENT_TRADER_ID`, `FillReport`, `TopOfBook`).

```{toctree}
:caption: Session
:maxdepth: 1

/autoapi/omiecid_simulator/omiecid/omie_session/index
```

## Market

The contract book, the contracts, the order types, the market configuration, settlement and the
encoding of orders and trades.

```{toctree}
:caption: Market
:maxdepth: 1

/autoapi/omiecid_simulator/omiecid/contract_book/index
/autoapi/omiecid_simulator/omiecid/contract/index
/autoapi/omiecid_simulator/omiecid/order_types/index
/autoapi/omiecid_simulator/omiecid/omie_config/index
/autoapi/omiecid_simulator/omiecid/settlement/index
/autoapi/omiecid_simulator/omiecid/encoding/index
```

## Data

Loading public OMIE data, its configuration, and building order tapes (`TapeBuilder`,
`stack_tapes`).

```{toctree}
:caption: Data
:maxdepth: 1

/autoapi/omiecid_simulator/jaxomiecid/omie_data_loader/index
/autoapi/omiecid_simulator/jaxomiecid/omie_data_config/index
/autoapi/omiecid_simulator/jaxomiecid/omie_tape_synth/index
```

## Packages

```{toctree}
:caption: Packages
:maxdepth: 1

/autoapi/omiecid_simulator/index
/autoapi/omiecid_simulator/omiecid/index
/autoapi/omiecid_simulator/jaxomiecid/index
```
