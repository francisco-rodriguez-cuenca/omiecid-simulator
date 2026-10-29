# Data

A session replays an **order tape**: the background orders of one contract, in
time order. Tapes come from the public OMIE files or from the synthetic tape
generator. No OMIE data ships with this package.

## The tape layout

A tape is an `(N, 13)` int32 array, one order per row:

| Column | Field | Encoding |
|---|---|---|
| 0 | `contract_id` | Quarter-hour slot $k$ of the delivery day, 0–95. |
| 1 | `side` | 0 buy, 1 sell. |
| 2 | `price` | Price tick offset from the lower bound: $(p - p_{\min}) / 0.01$. |
| 3 | `quantity` | Quantity in ticks of 0.1 MW. |
| 4 | `order_id` | Positive; padding rows carry $\le 0$. |
| 5 | `agent_id` | Anonymous participant number (in a loaded OMIE tape, one per bidding unit). |
| 6 | `timestamp` | Seconds (see below); padding rows carry $\le 0$. |
| 7 | `exec_restriction` | 0 NON, 1 IOC, 2 FOK, 3 ICE. |
| 8 | `validity` | 0 GFS, 1 GTD. |
| 9 | `expiry_time` | 0 = gate closure. |
| 10 | `status` | Order status code. |
| 11 | `peak_qty` | Iceberg peak, in quantity ticks. |
| 12 | `ppd` | ICEwPI price increment, signed price ticks. |

A session takes a **tape set**: an `(n_tapes, N_padded, 13)` int32 array in
which shorter tapes are right-padded with inert padding rows (the loader's
`PAD_ORDER_ROW`). `stack_tapes` and `pad_to` in
`omiecid_simulator.jaxomiecid.omie_tape_synth` build it. The encoding functions
are in `omiecid_simulator.omiecid.encoding`; the ticks and bounds are in
`OMIEProductConfig` (see [Market model](market-model.md#units-and-ticks)).

## Public OMIE data

### Downloading

OMIE publishes the continuous intraday orders and trades as monthly archives of
daily flat files at <https://www.omie.es>. The download script, in the
repository (not in the installed package), fetches and extracts them:

```bash
python -m scripts.download_omie_data --months 202506 --data-type orders
python -m scripts.download_omie_data --months 202506 --data-type trades
python -m scripts.download_omie_data --date 2025-06-15 --data-type orders  # month of a date
python -m scripts.download_omie_data --months 202506 --dry-run             # no network
```

Options: `--data-type` (`orders`, `trades` or `marginalpdbc`, the day-ahead
marginal prices) and `--output-dir` (default `data/rawOMIE`). The downloaded
archives are always kept, in a `.zips` folder inside the output folder
(`data/rawOMIE/.zips` by default); the `--keep-zips` option has no effect. A
month already on disk is skipped. The extracted files land as:

```text
data/rawOMIE/
  orders_YYYYMMDD.1
  trades_YYYYMMDD.1
  marginalpdbc_YYYYMMDD.1
```

The data is OMIE's; see OMIE's terms of use on its website.

### Loading one contract

A contract is identified by its delivery date $D$ and slot $k$. OMIE's files are
split by **session day**, and a contract is traded from 15:20 on $D-1$ until its
gate closure on $D$, so its rows come from the files of $D-1$ and $D$
([market specification §13.4](omiecid/market-spec.md#134-notes-for-data-loading-m7)).
The loader needs all four files, `orders_` and `trades_` for both days, and
raises `IncompleteProductError` if one is missing
([OQ-29](omiecid/open-questions.md#oq-29--incomplete-delivery-products-a-required-session-file-missing)).

```python
from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import load_omie_orders, load_omie_trades

cfg = OMIEDataConfig(data_path="data/rawOMIE", contract_id=51, delivery_date="2025-06-25")
orders = load_omie_orders(cfg)    # (N, 13) int32 tape, sorted by send time
trades = load_omie_trades(cfg)    # (M, 6) int32: the trades OMIE actually printed
```

- **Timestamps** are seconds since 00:00 of $D-1$, local wall clock, with no
  wrap-around; real rows lie in $[55200, 165600)$ (15:20 on $D-1$ to 22:00 on
  $D$). Rows sent in the same second keep file order
  ([OQ-32](omiecid/open-questions.md#oq-32--same-second-arrival-order-in-replay)).
- **Participants** are numbered from the (`Agente`, `Unidad`) pair; the same
  numbering is shared by orders and trades.
- **Both Iberian bidding zones** (Spain and Portugal) are included; the loader
  does not filter by zone. OMIE's files hold no order from outside Iberia, and a
  cross-border trade appears in the trade file with the non-Iberian side blank
  ([Modelling assumptions › Simplifications](modelling-assumptions.md#simplifications);
  tutorial [T03](../tutorials/T03_real_data.ipynb), section 6).
- **Daylight-saving days** are refused unless `allow_dst_transition=True`
  ([OQ-30](omiecid/open-questions.md#oq-30--daylight-saving-transitions-contract-strings-slot-count-and-wall-clock-timestamps)).

The loader parses the raw files on every call (a few seconds per contract); save
the returned array with `numpy.save` if you replay it often.

### Replaying it in a session

```python
from omiecid_simulator.omiecid.omie_session import Session
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig
from omiecid_simulator.jaxomiecid.omie_tape_synth import stack_tapes

k = 51
session = Session({
    "tapes": stack_tapes([orders], pad_extra=8),
    "orders_per_step": 8,
    "session_clock": "got_relative",
    "gct_minutes_from_got": 460 + 60 * (k // 4),
    "market_config": OMIEMarketConfig(nTrades=4096),
})
```

- Use `session_clock="got_relative"`: the session clock then follows the tape's
  real send times from gate opening, and the session ends when the tape is
  exhausted.
- `gct_minutes_from_got` must match the contract: gate closure falls at the hour
  before the delivery hour, $460 + 60 \lfloor k/4 \rfloor$ minutes after gate
  opening.
- Keep `pad_extra` at least `orders_per_step`.
- Size `nTrades` for the whole session (see
  [Sessions](sessions.md#sizing-ntrades)).
- Several contracts or days in one tape set must share the delivery hour.
  `build_fixed_tape_set(data_path=..., contract_ids=[...], delivery_dates=[...])`
  in `omie_data_loader` loads and stacks several at once.

Public files contain no cancellations, so the replay can trade more than the
real market did; `load_omie_trades` gives the real trades to compare with (see
[Modelling assumptions](modelling-assumptions.md#replaying-public-omie-data) and
the [real-data tutorial](../tutorials/T03_real_data.ipynb)).

## Synthetic tapes

`TapeBuilder` writes tapes by hand, in readable units:

```python
from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder, stack_tapes
from omiecid_simulator.omiecid.order_types import Side, ExecRestriction

tb = TapeBuilder(timestamp_step=30)        # timestamps 1, 31, 61, ... seconds
tb.add(side=Side.BUY, price_eur=49.0, qty_mw=2.0)
tb.add(side=Side.SELL, price_eur=52.0, qty_mw=5.0, peak_qty_mw=1.0,
       exec_restriction=ExecRestriction.ICE)                     # iceberg
tb.add(side=Side.SELL, price_eur=48.0, qty_mw=1.0,
       exec_restriction=ExecRestriction.IOC)
tape = tb.build()                          # (n_rows + 8, 13), padded
tapes = stack_tapes([tape])                # (1, N_padded, 13) for a Session
```

`add` takes either the float (`price_eur`, `qty_mw`, `peak_qty_mw`, `ppd_eur`)
or the integer tick (`price_int`, `qty_int`, `peak_int`, `ppd_int`) for each
field, and optional `timestamp`, `order_id`, `agent_id`, `validity` and
`expiry_time`. Order ids and timestamps are assigned automatically otherwise.
Synthetic tapes use the default `"span_scaled"` clock, which maps the tape's
time span onto the session's `gct_minutes_from_got`.

The same module holds the constructions used by the certification notebooks
(`make_crossing_pair`, `make_iceberg_cascade`, `make_random_session`, and
others). Each draws the parameters of a construction from a seed and returns a
`Draw` with the tape, the parameters and the construction's expected outcome.

## Day-ahead prices

`parse_marginalpdbc_file` and `day_ahead_anchor` in `omie_data_loader` read the
OMIE day-ahead marginal price files (`marginalpdbc_YYYYMMDD.1`), for users who
want a day-ahead reference price for a contract.
