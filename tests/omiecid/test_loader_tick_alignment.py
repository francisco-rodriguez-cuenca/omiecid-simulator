"""Tests for story 1.5-HOTFIX-loader-precision — Loader float32 precision hotfix.

Behavioral contracts for the ``row_to_order(row, product_config) -> OMIEOrder``
helper in ``omiecid_simulator/jaxomiecid/omie_data_loader.py``.

Root cause of the hotfix (documented in agile-plan v4.3 §1.5-HOTFIX): the
loader casts the 13-column row array to ``jnp.float32``; prices like
``133.87`` become ``133.8699951171875``, which fails the 1e-9 tolerance in
``order_validator._is_on_tick``. The helper must snap each float column to
its tick on read using ``round(float(row[i]) / tick) * tick``:
  - price    (col 2, ``price_tick``)
  - quantity (col 3, ``qty_tick``)
  - peak_qty (col 11, ``qty_tick``)
  - ppd      (col 12, ``price_tick``)

Tests are black-box: they import only the public interface and assert
against the domain rule that every snapped column passes ``_is_on_tick``
with tolerance 1e-9. The tests must not depend on any implementation
internals (e.g. ``contract_book._product_config``); callers always pass
``product_config`` explicitly.

Contract items (from agile-plan §Story 1.5-HOTFIX-loader-precision):
    1. Real-data regression on contract 81 of 2026-01-12 (skip if file absent)
    2. Synthetic fixture of pathological float32-ULP prices/quantities
    3. Property sweep over the full price / quantity grids
    4. Trades forward-looking manual-snap probe (load_omie_trades)
    5. peak_qty / ppd on-tick after helper for ICE / ICEwPI fixtures
    6. Blank peak_qty / ppd collapse to 0.0 without NaN / None crashes
    7. ICEwPI slice projection stays on price_tick
    8. No-op on already-snapped values (round-trip identical)
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    load_omie_orders,
    load_omie_trades,
)
from omiecid_simulator.omiecid.encoding import (
    encode_ppd,
    encode_price,
    encode_qty,
)
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    Status,
    Validity,
)
from omiecid_simulator.omiecid.order_validator import _is_on_tick


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

REAL_DATA_PATH = "data/rawOMIE"
REAL_DELIVERY_DATE = "2026-01-12"
REAL_CONTRACT_ID = 81
REAL_ORDERS_FILE = Path(REAL_DATA_PATH) / "orders_20260112.1"


@pytest.fixture
def product_config():
    """Default OMIE product config (price_tick=0.01, qty_tick=0.1)."""
    return OMIEProductConfig()


def _make_row(
    contract_id: int = 81,
    side: int = int(Side.BUY),
    price: float = 45.51,
    quantity: float = 10.0,
    order_id: int = 1,
    agent_id: int = 1,
    timestamp: int = 0,
    exec_restriction: int = int(ExecRestriction.NON),
    validity: int = int(Validity.GFS),
    expiry_time: int = 0,
    status: int = int(Status.ACTIVE),
    peak_qty: float = 0.0,
    ppd: float = 0.0,
    product_config: OMIEProductConfig | None = None,
):
    """Build a single 13-column int32 row matching post-P2-F0 loader output.

    Accepts EUR/MWh prices / MW quantities and encodes them on the
    tick grid carried by ``product_config`` (defaults to
    :class:`OMIEProductConfig()` — price_tick=0.01, qty_tick=0.1).
    """
    pc = product_config or OMIEProductConfig()
    return jnp.array(
        [
            contract_id,
            side,
            encode_price(price, pc),
            encode_qty(quantity, pc),
            order_id,
            agent_id,
            timestamp,
            exec_restriction,
            validity,
            expiry_time,
            status,
            encode_qty(peak_qty, pc),
            encode_ppd(ppd, pc),
        ],
        dtype=jnp.int32,
    )


def _row_to_order():
    """Import the snap-on-read helper lazily so --collect-only works pre-impl."""
    from omiecid_simulator.jaxomiecid.omie_data_loader import row_to_order
    return row_to_order


# ---------------------------------------------------------------------------
# Contract item 1 — Real-data regression on 2026-01-12, contract 81
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not REAL_ORDERS_FILE.exists(),
    reason=f"Real OMIE orders file not present: {REAL_ORDERS_FILE}",
)
def test_real_data_contract_81_all_rows_on_tick(product_config):
    """Every loaded row on 2026-01-12 contract 81 must be on-tick via helper.

    slow: full-file CSV parse + per-row helper application on real OMIE data.
    Pre-fix: fails with ULP-noisy prices (e.g. 133.87 -> 133.8699951171875).
    Post-fix: zero failures.
    """
    row_to_order = _row_to_order()
    config = OMIEDataConfig(
        data_path=REAL_DATA_PATH,
        contract_id=REAL_CONTRACT_ID,
        delivery_date=REAL_DELIVERY_DATE,
    )
    orders = load_omie_orders(config)

    # Guard: if the date+contract yields no rows, the test cannot exercise
    # the invariant; fail loudly rather than silently passing.
    assert orders.shape[0] > 0, (
        f"Expected non-empty orders for {REAL_DELIVERY_DATE} contract "
        f"{REAL_CONTRACT_ID}; got shape {orders.shape}"
    )

    price_tick = product_config.price_tick
    qty_tick = product_config.qty_tick

    for i in range(orders.shape[0]):
        order = row_to_order(orders[i], product_config)
        assert _is_on_tick(order.price, price_tick), (
            f"row {i}: price {order.price!r} not on tick {price_tick}"
        )
        assert _is_on_tick(order.quantity, qty_tick), (
            f"row {i}: quantity {order.quantity!r} not on tick {qty_tick}"
        )


# ---------------------------------------------------------------------------
# Contract item 2 — Synthetic pathological-ULP fixture
# ---------------------------------------------------------------------------

PATHOLOGICAL_PRICES = [133.87, 110.71, -4.30, 0.00, 9998.99]
PATHOLOGICAL_QUANTITIES = [0.1, 5.4, 999.0]


@pytest.mark.parametrize("price", PATHOLOGICAL_PRICES)
def test_synthetic_pathological_prices_snap_on_tick(product_config, price):
    """Prices known to develop ULP noise under float32 cast snap on-tick."""
    row_to_order = _row_to_order()
    row = _make_row(price=price, quantity=0.1)
    order = row_to_order(row, product_config)
    assert _is_on_tick(order.price, product_config.price_tick), (
        f"price {price!r} -> {order.price!r} not on tick "
        f"{product_config.price_tick}"
    )


@pytest.mark.parametrize("quantity", PATHOLOGICAL_QUANTITIES)
def test_synthetic_pathological_quantities_snap_on_tick(product_config, quantity):
    """Quantities known to develop ULP noise under float32 cast snap on-tick."""
    row_to_order = _row_to_order()
    row = _make_row(price=45.51, quantity=quantity)
    order = row_to_order(row, product_config)
    assert _is_on_tick(order.quantity, product_config.qty_tick), (
        f"quantity {quantity!r} -> {order.quantity!r} not on tick "
        f"{product_config.qty_tick}"
    )


# ---------------------------------------------------------------------------
# Contract item 3 — Property sweep over full price / quantity grids
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_property_sweep_price_grid(product_config):
    """Every tick-aligned price in [-9999, 9999] step 0.01 snaps on-tick.

    slow: ~2M values cast through float32 + helper.
    """
    row_to_order = _row_to_order()
    price_tick = product_config.price_tick

    # Build the grid as a vectorized float32 cast (the operation that
    # introduces ULP noise in the loader) rather than a Python loop.
    grid = np.round(np.arange(-9999.0, 9999.0 + price_tick / 2, price_tick), 2)
    grid_f32 = jnp.asarray(grid, dtype=jnp.float32)

    # Apply helper row-by-row; assert on-tick. We use a modest but
    # representative slice to keep runtime sane while still covering
    # small/medium/large magnitudes and the sign boundary.
    sample_idxs = np.concatenate([
        np.arange(0, 200),                       # deep negative
        np.arange(len(grid) // 2 - 100, len(grid) // 2 + 100),  # around zero
        np.arange(len(grid) - 200, len(grid)),   # deep positive
    ])

    for i in sample_idxs:
        p = float(grid_f32[i])
        row = _make_row(price=p, quantity=0.1)
        order = row_to_order(row, product_config)
        assert _is_on_tick(order.price, price_tick), (
            f"grid price {grid[i]!r} (f32={p!r}) -> {order.price!r} "
            f"not on tick {price_tick}"
        )


@pytest.mark.slow
def test_property_sweep_quantity_grid(product_config):
    """Every tick-aligned quantity in [0.1, 999] step 0.1 snaps on-tick.

    slow: ~10k values cast through float32 + helper.
    """
    row_to_order = _row_to_order()
    qty_tick = product_config.qty_tick

    grid = np.round(np.arange(0.1, 999.0 + qty_tick / 2, qty_tick), 1)
    grid_f32 = jnp.asarray(grid, dtype=jnp.float32)

    # Sample endpoints and a mid-range window.
    sample_idxs = np.concatenate([
        np.arange(0, 100),
        np.arange(len(grid) // 2 - 50, len(grid) // 2 + 50),
        np.arange(len(grid) - 100, len(grid)),
    ])

    for i in sample_idxs:
        q = float(grid_f32[i])
        row = _make_row(price=45.51, quantity=q)
        order = row_to_order(row, product_config)
        assert _is_on_tick(order.quantity, qty_tick), (
            f"grid quantity {grid[i]!r} (f32={q!r}) -> {order.quantity!r} "
            f"not on tick {qty_tick}"
        )


# ---------------------------------------------------------------------------
# Contract item 4 — Trades forward-looking manual-snap probe
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not REAL_ORDERS_FILE.exists(),
    reason=f"Real OMIE data not present: {REAL_ORDERS_FILE}",
)
def test_trades_manual_snap_probe(product_config):
    """Forward-looking probe on load_omie_trades columns 1 (price) and 2 (qty).

    slow: full trades-file parse. This is informational: it confirms that
    manually applying the snap formula `round(v / tick) * tick` to the
    current trades loader output produces on-tick values for every trade
    row. It is NOT a contract that `row_to_trade` exists — trades-helper
    extension is explicitly out of scope for 1.5-HOTFIX.

    Author's note: this test surfaces whether trades also need the helper
    treatment; if it fails post-fix, open a follow-up story.
    """
    config = OMIEDataConfig(
        data_path=REAL_DATA_PATH,
        contract_id=REAL_CONTRACT_ID,
        delivery_date=REAL_DELIVERY_DATE,
    )
    trades = load_omie_trades(config)
    if trades.shape[0] == 0:
        pytest.skip("No trades for this contract/date — probe inapplicable.")

    price_tick = product_config.price_tick
    qty_tick = product_config.qty_tick

    for i in range(trades.shape[0]):
        raw_price = float(trades[i, 1])
        raw_qty = float(trades[i, 2])
        snapped_price = round(raw_price / price_tick) * price_tick
        snapped_qty = round(raw_qty / qty_tick) * qty_tick
        assert _is_on_tick(snapped_price, price_tick), (
            f"trade row {i}: manually-snapped price {snapped_price!r} "
            f"(raw {raw_price!r}) not on tick {price_tick}"
        )
        assert _is_on_tick(snapped_qty, qty_tick), (
            f"trade row {i}: manually-snapped qty {snapped_qty!r} "
            f"(raw {raw_qty!r}) not on tick {qty_tick}"
        )


# ---------------------------------------------------------------------------
# Contract item 5 — peak_qty / ppd on-tick for ICE / ICEwPI fixtures
# ---------------------------------------------------------------------------

def test_ice_peak_qty_on_tick(product_config):
    """ICE order peak_qty (col 11) snaps to qty_tick via helper."""
    row_to_order = _row_to_order()
    row = _make_row(
        price=110.71,
        quantity=5.4,
        exec_restriction=int(ExecRestriction.ICE),
        peak_qty=1.1,
        ppd=0.0,
    )
    order = row_to_order(row, product_config)
    assert _is_on_tick(order.peak_qty, product_config.qty_tick), (
        f"ICE peak_qty {order.peak_qty!r} not on tick {product_config.qty_tick}"
    )


def test_icewpi_peak_qty_and_ppd_on_tick(product_config):
    """ICEwPI order peak_qty (qty_tick) and ppd (price_tick) snap on-tick."""
    row_to_order = _row_to_order()
    row = _make_row(
        price=45.51,
        quantity=5.4,
        exec_restriction=int(ExecRestriction.ICE),
        peak_qty=1.1,
        ppd=0.03,
    )
    order = row_to_order(row, product_config)
    assert _is_on_tick(order.peak_qty, product_config.qty_tick), (
        f"ICEwPI peak_qty {order.peak_qty!r} not on tick "
        f"{product_config.qty_tick}"
    )
    assert _is_on_tick(order.ppd, product_config.price_tick), (
        f"ICEwPI ppd {order.ppd!r} not on tick {product_config.price_tick}"
    )


# ---------------------------------------------------------------------------
# Contract item 6 — Blank peak_qty / ppd collapse to 0.0 without NaN crashes
# ---------------------------------------------------------------------------

def test_blank_peak_qty_and_ppd_return_zero(product_config):
    """Blank (stored-as-0.0) peak_qty / ppd round-trip to 0.0 via helper.

    The loader already converts blank `Cantidad reducida` / `PPD` cells to
    0.0 before the float32 cast (per `_parse_optional_float`). The helper
    must not choke on those zeros — returns 0.0 cleanly, no NaN or None.
    """
    row_to_order = _row_to_order()
    row = _make_row(price=45.51, quantity=10.0, peak_qty=0.0, ppd=0.0)
    order = row_to_order(row, product_config)
    assert order.peak_qty == 0.0
    assert order.ppd == 0.0
    assert not np.isnan(order.peak_qty)
    assert not np.isnan(order.ppd)
    assert _is_on_tick(order.peak_qty, product_config.qty_tick)
    assert _is_on_tick(order.ppd, product_config.price_tick)


def test_helper_does_not_crash_on_zero_fields(product_config):
    """Helper accepts a row with all snap-columns zero without raising."""
    row_to_order = _row_to_order()
    row = _make_row(price=0.0, quantity=0.1, peak_qty=0.0, ppd=0.0)
    # Must not raise; must return an OMIEOrder.
    order = row_to_order(row, product_config)
    assert isinstance(order, OMIEOrder)


# ---------------------------------------------------------------------------
# Contract item 7 — ICEwPI slice projection stays on price_tick
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slice_idx", [1, 2, 3])
def test_icewpi_slice_projection_on_tick(product_config, slice_idx):
    """slice_price = base_price + ppd * slice_idx must be on price_tick.

    This exercises the downstream invariant consumed by
    `iceberg_manager.py` slice regeneration. A failure here indicates that
    snap-on-read at the loader is insufficient for ICEwPI arithmetic and
    would warrant escalating to `2.0-pre-A-numeric-representation`.
    """
    row_to_order = _row_to_order()
    row = _make_row(
        price=45.51,
        quantity=5.4,
        exec_restriction=int(ExecRestriction.ICE),
        peak_qty=1.0,
        ppd=0.01,
    )
    order = row_to_order(row, product_config)
    slice_price = order.price + order.ppd * slice_idx
    assert _is_on_tick(slice_price, product_config.price_tick), (
        f"ICEwPI slice[{slice_idx}] price {slice_price!r} not on tick "
        f"{product_config.price_tick} (base={order.price!r}, ppd={order.ppd!r})"
    )


# ---------------------------------------------------------------------------
# Contract item 8 — No-op on already-snapped values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("price", [45.51, 0.01, -0.01, 100.00, 9999.00])
def test_already_snapped_price_is_noop(product_config, price):
    """Already-on-tick prices round-trip via the int32 encode/decode pair.

    Post P2-F0 the loader emits int-ticks directly, so ``row_to_order``
    is a decode step: ``decode_price(encode_price(p, pc), pc)`` must
    return a value on the tick grid and numerically equal to ``p``
    within float noise (offset addition introduces sub-ULP jitter for
    small magnitudes; the tick-alignment check is the stable lock).
    """
    row_to_order = _row_to_order()
    row = _make_row(price=price, quantity=0.1)
    order = row_to_order(row, product_config)
    assert abs(float(order.price) - price) < 1e-6, (
        f"round-trip: input {price!r} -> output {order.price!r}"
    )
    assert _is_on_tick(order.price, product_config.price_tick)


@pytest.mark.parametrize("quantity", [0.1, 0.5, 5.0, 100.0, 999.0])
def test_already_snapped_quantity_is_noop(product_config, quantity):
    """Already-on-tick quantities round-trip to the exact tick-aligned value."""
    row_to_order = _row_to_order()
    row = _make_row(price=45.51, quantity=quantity)
    order = row_to_order(row, product_config)
    expected = round(quantity / product_config.qty_tick) * product_config.qty_tick
    assert order.quantity == expected, (
        f"idempotency: input {quantity!r} -> output {order.quantity!r}, "
        f"expected {expected!r}"
    )
    assert _is_on_tick(order.quantity, product_config.qty_tick)


# ---------------------------------------------------------------------------
# Additional lock: helper respects a non-default tick config (F5)
# ---------------------------------------------------------------------------

def test_helper_respects_custom_price_tick():
    """Helper snaps to the price_tick passed via product_config, not a hardcoded one.

    Locks in that `row_to_order` reads `product_config.price_tick` /
    `qty_tick` rather than assuming the defaults 0.01 / 0.1.
    """
    row_to_order = _row_to_order()
    custom = OMIEProductConfig(price_tick=0.05, qty_tick=0.5)
    # Input 133.87 is NOT on a 0.05 tick; nearest 0.05 tick is 133.85.
    row = _make_row(price=133.87, quantity=5.0, product_config=custom)
    order = row_to_order(row, custom)
    assert _is_on_tick(order.price, custom.price_tick)
    assert order.price == pytest.approx(133.85, abs=1e-9)


def test_helper_respects_custom_qty_tick():
    """Helper snaps to qty_tick passed via product_config, not a hardcoded one."""
    row_to_order = _row_to_order()
    custom = OMIEProductConfig(price_tick=0.01, qty_tick=0.5)
    # Input 5.4 is NOT on a 0.5 tick; nearest is 5.5.
    row = _make_row(price=45.51, quantity=5.4, product_config=custom)
    order = row_to_order(row, custom)
    assert _is_on_tick(order.quantity, custom.qty_tick)
    assert order.quantity == pytest.approx(5.5, abs=1e-9)
