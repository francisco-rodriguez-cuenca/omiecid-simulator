# OmieCID-Simulator

**OmieCID-Simulator** is a JAX-based Python package that reproduces and analyzes order matching in the Iberian CID market, the MIBEL segment of the European Single Intraday Coupling operated by OMIE, with out-of-the-box compatibility with public OMIE data.

It contains:

- the OMIE matching engine: one limit order book per quarter-hour contract, with
  price-time priority, limit (NON), IOC, FOK, iceberg and ICEwPI orders,
  cancellation, gate closure and settlement;
- the data pipeline: a loader for the public OMIE order and trade files, and a
  generator of synthetic order tapes;
- a **Session**: replay of an order tape on the market clock, in which one
  participant submits its own orders until gate closure;
- the book-level order-entry API (`ContractBook`);
- the certification evidence: 17 scenario notebooks (C00–C16), the certificate
  and its log.

Everything is written in JAX: the session step is compiled once and can be
`jit`-ed and `vmap`-ed.

## Installation

Python 3.10 or later. From a clone of this repository:

```bash
pip install .                         # core: jax, flax, chex, numpy, pandas
pip install ".[validation]"           # + notebooks and the test suite
pip install ".[examples]"             # + the demo scripts
pip install ".[cuda12]"               # + the NVIDIA GPU backend
pip install -c constraints/verified.txt ".[validation]"   # exact verified versions
```

The distribution is named `omiecid-simulator` (`pip install omiecid-simulator`
once it is published on PyPI); the import package is `omiecid_simulator`.
Everything runs on CPU; the GPU is optional.

## Quickstart

A synthetic tape, a session, one participant, and the settled result:

```python
import jax
from omiecid_simulator.omiecid.omie_session import Session, AgentOrder, AGENT_TRADER_ID
from omiecid_simulator.omiecid.order_types import Side
from omiecid_simulator.omiecid.settlement import compute_agent_pnl
from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder, stack_tapes

# 1. A synthetic background tape: a two-sided book around 50 EUR/MWh,
#    then six 1 MW sellers that cross the bids, then a few far asks.
tb = TapeBuilder(timestamp_step=30)                  # one order every 30 s
for i in range(5):
    tb.add(side=Side.BUY, price_eur=49.0 - i, qty_mw=2.0)
    tb.add(side=Side.SELL, price_eur=51.0 + i, qty_mw=2.0)
for _ in range(6):
    tb.add(side=Side.SELL, price_eur=47.0, qty_mw=1.0)
for i in range(4):
    tb.add(side=Side.SELL, price_eur=56.0 + i, qty_mw=1.0)
tapes = stack_tapes([tb.build()])                    # (n_tapes, N, 13) int32

# 2. A session replays 4 tape orders per step; the participant acts in between.
session = Session({"tapes": tapes, "orders_per_step": 4})
state = session.reset(jax.random.PRNGKey(0))

plan = {
    1: AgentOrder.create(side=Side.BUY, price=48.5, quantity=1.0),   # rests, filled passively
    4: AgentOrder.create(side=Side.SELL, price=45.0, quantity=1.0),  # crosses the best bid
}
key, done, step = jax.random.PRNGKey(1), False, 0
while not done:
    key, sub = jax.random.split(key)
    state, report, done = session.step(sub, state, plan.get(step))
    tob = session.top_of_book(state)
    print(f"step {step}: t={int(state.time):2d} min  "
          f"bid/ask {float(tob.p_bid):.2f}/{float(tob.p_ask):.2f}  "
          f"passive {float(report.passive_qty_signed):+.1f} MW  "
          f"own {float(report.submit.trade_qty):.1f} MW  "
          f"cash {float(report.cash_delta_eur):+8.3f} EUR")
    step += 1

# 3. Settlement: agent_cash is the participant's realized cash, exact.
#    compute_agent_pnl over the participant's trade tape is a cross-check; it is
#    exact here because no step had passive fills in both directions.
pnl = float(state.agent_cash[0])
check = float(compute_agent_pnl(state.agent_trades.reshape(-1, 6), AGENT_TRADER_ID))
print(f"inventory {float(state.agent_inventory[0]):+.1f} MW, "
      f"settled P&L {pnl:+.3f} EUR (trade-tape check {check:+.3f} EUR)")
```

Output (the first call compiles the step, which takes about a minute on CPU):

```text
step 0: t=10 min  bid/ask 49.00/51.00  passive +0.0 MW  own 0.0 MW  cash   +0.000 EUR
step 1: t=22 min  bid/ask 49.00/51.00  passive +0.0 MW  own 0.0 MW  cash   +0.000 EUR
step 2: t=35 min  bid/ask 48.50/51.00  passive +0.0 MW  own 0.0 MW  cash   +0.000 EUR
step 3: t=47 min  bid/ask 47.00/51.00  passive +1.0 MW  own 0.0 MW  cash  -12.125 EUR
step 4: t=60 min  bid/ask 46.00/51.00  passive +0.0 MW  own 1.0 MW  cash  +11.750 EUR
inventory +0.0 MW, settled P&L -0.375 EUR (trade-tape check -0.375 EUR)
```

The bid at 48.50 EUR/MWh rests behind the background book and is hit at step 3;
the sell at step 4 crosses the best bid (47.00). A trade's value is
$V = Q \cdot p \cdot d$ with $d = 0.25$ h, so the round trip settles at
$(47.00 - 48.50) \cdot 1 \cdot 0.25 = -0.375$ EUR. `state.agent_cash` is the
settled result; `compute_agent_pnl` over `state.agent_trades` agrees only when no
step had passive fills in both directions (see [Sessions](docs/sessions.md#settlement)
and [Limitations](docs/limitations.md)).

## Real OMIE data

No OMIE data ships with the package. Download the public files from OMIE
(both order and trade files are needed), then load one contract:

```bash
python -m scripts.download_omie_data --months 202506 --data-type orders
python -m scripts.download_omie_data --months 202506 --data-type trades
```

```python
from omiecid_simulator.omiecid.omie_session import Session
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig
from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import load_omie_orders
from omiecid_simulator.jaxomiecid.omie_tape_synth import stack_tapes

k = 51                                   # delivery 12:45-13:00 on 2025-06-25
orders = load_omie_orders(OMIEDataConfig(data_path="data/rawOMIE",
                                         contract_id=k, delivery_date="2025-06-25"))
session = Session({
    "tapes": stack_tapes([orders]),
    "orders_per_step": 8,
    "session_clock": "got_relative",               # the real market clock
    "gct_minutes_from_got": 460 + 60 * (k // 4),   # gate closure of contract k
    "market_config": OMIEMarketConfig(nTrades=4096),
})
```

Use `session_clock="got_relative"` for real data. See [Data](docs/data.md).

## Documentation

- [Market model](docs/market-model.md): what the engine implements, rule by rule.
- [Modelling assumptions](docs/modelling-assumptions.md): rulings and simplifications.
- [Sessions](docs/sessions.md): the `Session` API.
- [Data](docs/data.md): OMIE files, the tape layout, synthetic tapes.
- [Validation](docs/validation.md): the certification ladder and its evidence.
- [Testing](docs/testing.md) and [Performance](docs/performance.md).
- [Limitations](docs/limitations.md).
- [Certification evidence](docs/omiecid/README.md), [notebooks](notebooks/certification/README.md)
  and [demos](demos/README.md).

## Citing

If you use OmieCID-Simulator, please cite it as described in
[`CITATION.cff`](CITATION.cff).

The limit-order-book core derives from JAX-LOB, through
[JaxMARL-HFT](https://github.com/vmohl/JaxMARL-HFT). Please also cite JAX-LOB:

```bibtex
@misc{frey2023jaxlob,
  title={JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading},
  author={Sascha Frey and Kang Li and Peer Nagy and Silvia Sapora and Chris Lu and Stefan Zohren and Jakob Foerster and Anisoara Calinescu},
  year={2023},
  eprint={2308.13289},
  archivePrefix={arXiv},
  primaryClass={q-fin.TR}
}
```

## License

Apache License 2.0, see [`LICENSE`](LICENSE). Copyright and attribution
notices are in [`NOTICE`](NOTICE); the authors are listed in [`AUTHORS`](AUTHORS).
The regulatory documents in [`docs/omiecid/corpus/`](docs/omiecid/corpus/README.md)
are OMIE documents, not covered by this licence.

This software is for research and education. It is not intended for live trading.
