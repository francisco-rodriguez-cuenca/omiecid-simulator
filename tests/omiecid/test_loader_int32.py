"""Locked behavioral tests for Story P2-F0 — Loader numeric representation → int32.

Contract source:
  - docs/omiecid/agile-plan.md §Phase 2 → P2-F0 story card
  - docs/omiecid/migration-plan.md §Phase 2 "Loader dtype contract" (lines ~806-815)
  - docs/omiecid/sprint-status.md §"P2-F0 — Architecture Review" (2026-04-22)

After P2-F0 merges the loader must emit:

  * ``load_omie_orders(config)`` → ``jnp.ndarray`` of shape ``(N, 13)`` with
    ``dtype == jnp.int32``.
  * ``load_omie_trades(config)`` → ``jnp.ndarray`` of shape ``(M, 6)`` with
    ``dtype == jnp.int32``.
  * ``timestamp`` column is true int32 in ``[0, 86400)``.
  * ``price`` and ``peak_qty``: offset-encoded via ``encoding.encode_price`` /
    ``encoding.encode_qty`` on their respective tick grids (unsigned).
  * ``ppd``: signed int-ticks, NOT offset-encoded.
    Formula: ``encode_ppd(ppd_eur) = jnp.int32(jnp.round(ppd_eur / price_tick))``.
  * ``peak_qty_int <= quantity_int`` row-wise.
  * ``peak_qty_int == 0`` for every row whose ``exec_restriction`` is not ICE
    (ICE and ICEwPI are the only order_types that carry peak_qty).
  * Every encoded column fits ``int32`` and lies within its declared
    ``[lo, hi]`` tick-band.
  * Cache schema bumps to v4; ``load → write v4 → reload`` yields a bitwise
    identical ``jnp.ndarray`` (compared via ``jnp.array_equal`` on raw buffers).

These tests are black-box contracts. They are expected to FAIL against the
current (float32) loader and to PASS once P2-F0 is implemented.

Runtime budget: all unit / property tests are fast (<1s). Only the real-data
regression test is marked ``@pytest.mark.slow``.
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
    decode_price,
    decode_qty,
    encode_price,
    encode_qty,
)
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction


# ---------------------------------------------------------------------------
# Fixtures & constants
# ---------------------------------------------------------------------------

REAL_DATA_PATH = "data/rawOMIE"
REAL_DELIVERY_DATE = "2026-01-12"
REAL_CONTRACT_ID = 81
REAL_ORDERS_FILE = Path(REAL_DATA_PATH) / "orders_20260112.1"
REAL_TRADES_FILE = Path(REAL_DATA_PATH) / "trades_20260112.1"


# Column indices for the 13-column orders array (from loader docstring).
COL_CONTRACT_ID = 0
COL_SIDE = 1
COL_PRICE = 2
COL_QUANTITY = 3
COL_ORDER_ID = 4
COL_AGENT_ID = 5
COL_TIMESTAMP = 6
COL_EXEC_RESTRICTION = 7
COL_VALIDITY = 8
COL_EXPIRY_TIME = 9
COL_STATUS = 10
COL_PEAK_QTY = 11
COL_PPD = 12

# Column indices for the 6-column trades array.
TCOL_CONTRACT_ID = 0
TCOL_PRICE = 1
TCOL_QUANTITY = 2
TCOL_BUY_AGENT_ID = 3
TCOL_SELL_AGENT_ID = 4
TCOL_MATCH_TIMESTAMP = 5


@pytest.fixture
def product_config():
    """Default OMIE product config (price_tick=0.01, qty_tick=0.1)."""
    return OMIEProductConfig()


@pytest.fixture
def real_config():
    """Real-data OMIE config pinned to 2026-01-12 / contract 81."""
    return OMIEDataConfig(
        data_path=REAL_DATA_PATH,
        contract_id=REAL_CONTRACT_ID,
        delivery_date=REAL_DELIVERY_DATE,
    )


def _have_real_data() -> bool:
    return REAL_ORDERS_FILE.exists()


# ---------------------------------------------------------------------------
# 1. dtype + shape contract — orders
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_load_omie_orders_dtype_is_int32(real_config):
    """load_omie_orders must return jnp.int32 (currently float32 — expected fail)."""
    orders = load_omie_orders(real_config)
    assert orders.dtype == jnp.int32, (
        f"expected jnp.int32, got {orders.dtype}"
    )


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_load_omie_orders_shape_is_N_by_13(real_config):
    """load_omie_orders must return shape (N, 13)."""
    orders = load_omie_orders(real_config)
    assert orders.ndim == 2 and orders.shape[1] == 13, (
        f"expected shape (N, 13), got {orders.shape}"
    )


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_load_omie_orders_empty_filter_still_int32(real_config):
    """Filtering to a non-existent contract yields (0,13) int32, not (0,13) float32.

    Locks the empty-path dtype: the loader's "no matching rows" branch
    (currently returns ``jnp.zeros((0, 13), dtype=jnp.float32)`` — see
    ``omie_data_loader.py:637, 646``) must switch to int32 post P2-F0.
    """
    cfg = OMIEDataConfig(
        data_path=real_config.data_path,
        contract_id=-1,  # guaranteed to match no rows
        delivery_date=real_config.delivery_date,
    )
    orders = load_omie_orders(cfg)
    assert orders.shape == (0, 13)
    assert orders.dtype == jnp.int32


# ---------------------------------------------------------------------------
# 2. dtype + shape contract — trades
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_load_omie_trades_dtype_is_int32(real_config):
    """load_omie_trades must return jnp.int32."""
    trades = load_omie_trades(real_config)
    assert trades.dtype == jnp.int32, (
        f"expected jnp.int32, got {trades.dtype}"
    )


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_load_omie_trades_shape_is_M_by_6(real_config):
    """load_omie_trades must return shape (M, 6)."""
    trades = load_omie_trades(real_config)
    assert trades.ndim == 2 and trades.shape[1] == 6, (
        f"expected shape (M, 6), got {trades.shape}"
    )


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_load_omie_trades_empty_filter_still_int32(real_config):
    """Filtering to a non-existent contract yields (0,6) int32 trades.

    Locks the empty-path dtype: ``omie_data_loader.py:730, 733, 741``
    currently returns float32 on the empty branch.
    """
    cfg = OMIEDataConfig(
        data_path=real_config.data_path,
        contract_id=-1,
        delivery_date=real_config.delivery_date,
    )
    trades = load_omie_trades(cfg)
    assert trades.shape == (0, 6)
    assert trades.dtype == jnp.int32


# ---------------------------------------------------------------------------
# 3. Timestamp column is true int32 in [0, 86400)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_orders_timestamp_column_is_int32_and_in_day_window(real_config):
    """Timestamp column (col 6) must be int32 values in [55200, 165600).

    OMIE-TAPE-CONTRACT-DATE-1 (group L re-baseline): the column is seconds
    since D-1 00:00 with no modulo; a real product's sends lie in
    [15:20 D-1, 22:00 D) = [55200, 165600) (market-spec v2.10 §13.4).
    Was [0, 86400) (wrapped time-of-day) before L1.
    """
    orders = load_omie_orders(real_config)
    if orders.shape[0] == 0:
        pytest.skip("No orders for this contract/date; timestamp invariant vacuous.")
    ts = orders[:, COL_TIMESTAMP]
    assert ts.dtype == jnp.int32, f"timestamp column dtype {ts.dtype} != int32"
    ts_np = np.asarray(ts)
    assert (ts_np >= 55200).all(), (
        f"timestamp column has values < 55200: min={int(ts_np.min())}"
    )
    assert (ts_np < 165600).all(), (
        f"timestamp column has values >= 165600: max={int(ts_np.max())}"
    )


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_trades_timestamp_column_is_int32_and_in_day_window(real_config):
    """Match timestamp column (col 5) must be int32 values in [55200, 165600).

    OMIE-TRADES-MATCH-TS-HEADER-1 (licensed re-baseline R-1): the column is
    seconds since D-1 00:00 with no modulo (OMIE-TAPE-CONTRACT-DATE-1 L1); a
    real product's matches lie in [15:20 D-1, 22:00 D) = [55200, 165600)
    (market-spec v2.12 §13.2.1). Was [0, 86400), which passed only because
    the accented ``Momento casación`` header was dropped and every value
    was 0.
    """
    trades = load_omie_trades(real_config)
    if trades.shape[0] == 0:
        pytest.skip("No trades for this contract/date; timestamp invariant vacuous.")
    ts = trades[:, TCOL_MATCH_TIMESTAMP]
    assert ts.dtype == jnp.int32
    ts_np = np.asarray(ts)
    assert (ts_np >= 55200).all(), (
        f"match timestamp column has values < 55200: min={int(ts_np.min())}"
    )
    assert (ts_np < 165600).all(), (
        f"match timestamp column has values >= 165600: max={int(ts_np.max())}"
    )


# ---------------------------------------------------------------------------
# 4. Round-trip property — decode_price(encode_price(p)) == p; decode_qty / encode_qty
# ---------------------------------------------------------------------------

# Sample a stratified set of on-tick prices from the configured bounds:
# endpoints, around zero (sign boundary), and representative middle values.
_ROUND_TRIP_PRICES = [
    -9999.0, -9998.99, -1000.00, -0.02, -0.01, 0.00,
    0.01, 0.02, 1.00, 45.51, 110.71, 133.87, 9998.99, 9999.00,
]
_ROUND_TRIP_QUANTITIES = [0.1, 0.2, 0.5, 1.0, 5.4, 10.0, 100.0, 500.5, 999.0]


@pytest.mark.parametrize("price", _ROUND_TRIP_PRICES)
def test_encode_decode_price_round_trip(product_config, price):
    """decode_price(encode_price(p)) == p for on-tick prices in configured bounds."""
    encoded = encode_price(price, product_config)
    # Encoded price must fit int32 and be non-negative (offset encoding).
    assert int(np.int32(encoded)) == int(encoded), (
        f"encoded price {encoded} does not fit int32"
    )
    assert encoded >= 0, f"offset-encoded price must be >= 0, got {encoded}"
    decoded = decode_price(encoded, product_config)
    assert decoded == pytest.approx(price, abs=1e-9), (
        f"round-trip failed: {price} -> {encoded} -> {decoded}"
    )


@pytest.mark.parametrize("qty", _ROUND_TRIP_QUANTITIES)
def test_encode_decode_qty_round_trip(product_config, qty):
    """decode_qty(encode_qty(q)) == q for on-tick quantities in configured bounds."""
    encoded = encode_qty(qty, product_config)
    assert int(np.int32(encoded)) == int(encoded), (
        f"encoded qty {encoded} does not fit int32"
    )
    assert encoded >= 0, f"qty encoding must be >= 0, got {encoded}"
    decoded = decode_qty(encoded, product_config)
    assert decoded == pytest.approx(qty, abs=1e-9), (
        f"round-trip failed: {qty} -> {encoded} -> {decoded}"
    )


# ---------------------------------------------------------------------------
# 5. ppd is signed int-ticks (sign preserving, NOT offset-encoded)
# ---------------------------------------------------------------------------

# The architect-pinned formula:
#   encode_ppd(ppd_eur) = jnp.int32(jnp.round(ppd_eur / price_tick))
# decoder is the exact inverse:
#   decode_ppd(ppd_int) = ppd_int * price_tick


def _encode_ppd_reference(ppd_eur: float, config: OMIEProductConfig) -> int:
    """Reference implementation of the ppd encoding formula."""
    return int(jnp.int32(jnp.round(jnp.asarray(ppd_eur) / config.price_tick)))


def _decode_ppd_reference(ppd_int: int, config: OMIEProductConfig) -> float:
    return ppd_int * config.price_tick


_PPD_VALUES = [-1.55, -1.00, -0.50, -0.03, -0.01, 0.00, 0.01, 0.03, 0.50, 1.00, 1.45]


@pytest.mark.parametrize("ppd", _PPD_VALUES)
def test_ppd_round_trip_preserves_sign(product_config, ppd):
    """Encoding a negative ppd decodes back to the same negative value."""
    encoded = _encode_ppd_reference(ppd, product_config)
    decoded = _decode_ppd_reference(encoded, product_config)
    assert decoded == pytest.approx(ppd, abs=1e-9)
    # Sign preserved.
    if ppd < 0:
        assert encoded < 0, f"negative ppd {ppd} encoded to non-negative {encoded}"
    elif ppd > 0:
        assert encoded > 0
    else:
        assert encoded == 0


def test_ppd_is_not_offset_encoded(product_config):
    """Encoding 0.0 ppd yields 0 — confirms no offset is applied.

    If the loader mistakenly offset-encoded ppd via encode_price (which maps
    0.0 EUR to +999_900 ticks), the encoded value for 0.0 ppd would be
    999_900, not 0. The architect-pinned formula is sign-preserving.
    """
    assert _encode_ppd_reference(0.0, product_config) == 0


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_loader_ppd_column_is_sign_preserving(real_config, product_config):
    """Loader column 12 (ppd) must follow the signed int-tick encoding.

    Per market-spec v2.4 §13.6 (finding DA-PPD-2026-04) the authoritative
    PPD rule is the **sign rule**: for ICEwPI (ICE exec_restriction with
    non-zero PPD), BUY rows must have ``PPD < 0`` and SELL rows must have
    ``PPD > 0``. All non-ICEwPI rows must have ``PPD == 0``. No rule-level
    magnitude envelope exists — see §13.6 and open-questions OQ-13.

    This test asserts the sign rule directly on the loader output:

      * ppd column dtype is int32 (shared with the whole array);
      * encoding is a pure int-tick scaling (round-trip through the
        reference decode/encode is exact);
      * ICEwPI BUY rows satisfy ``decoded_ppd < 0``;
      * ICEwPI SELL rows satisfy ``decoded_ppd > 0``;
      * non-ICEwPI rows (exec_restriction != ICE, or ICE with ppd == 0)
        satisfy ``decoded_ppd == 0``.
    """
    from omiecid_simulator.omiecid.order_types import Side

    orders = load_omie_orders(real_config)
    if orders.shape[0] == 0:
        pytest.skip("No orders for this contract/date; ppd invariant vacuous.")
    assert orders.dtype == jnp.int32
    ppd_col = np.asarray(orders[:, COL_PPD])
    side_col = np.asarray(orders[:, COL_SIDE])
    exec_col = np.asarray(orders[:, COL_EXEC_RESTRICTION])

    # Round-trip: decode then encode must be an exact identity on the
    # encoded int-tick column (locks the "signed int-tick, not offset" rule).
    decoded = ppd_col * product_config.price_tick
    reencoded = np.round(decoded / product_config.price_tick).astype(np.int32)
    assert np.array_equal(reencoded, ppd_col.astype(np.int32)), (
        "ppd encoding is not a pure signed int-tick scaling"
    )

    # Partition by ICEwPI membership.
    icewpi_mask = (exec_col == int(ExecRestriction.ICE)) & (ppd_col != 0)
    non_icewpi_mask = ~icewpi_mask

    # Sign rule on ICEwPI rows.
    if icewpi_mask.sum() > 0:
        buy_icewpi = icewpi_mask & (side_col == int(Side.BUY))
        sell_icewpi = icewpi_mask & (side_col == int(Side.SELL))
        if buy_icewpi.sum() > 0:
            assert (decoded[buy_icewpi] < 0).all(), (
                f"{int((decoded[buy_icewpi] >= 0).sum())} ICEwPI BUY rows "
                f"have decoded_ppd >= 0 (spec v2.4 §13.6 sign rule)"
            )
        if sell_icewpi.sum() > 0:
            assert (decoded[sell_icewpi] > 0).all(), (
                f"{int((decoded[sell_icewpi] <= 0).sum())} ICEwPI SELL rows "
                f"have decoded_ppd <= 0 (spec v2.4 §13.6 sign rule)"
            )

    # Non-ICEwPI rows must carry ppd == 0.
    if non_icewpi_mask.sum() > 0:
        assert (ppd_col[non_icewpi_mask] == 0).all(), (
            f"{int((ppd_col[non_icewpi_mask] != 0).sum())} non-ICEwPI rows "
            f"carry non-zero ppd (spec v2.4 §13.6)"
        )


# ---------------------------------------------------------------------------
# 6. Real-data regression: 2026-01-12 contract 81 replay — zero price_tick
#    rejections (same gate as 1.5-HOTFIX).
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not _have_real_data(),
    reason=f"Real OMIE orders file not present: {REAL_ORDERS_FILE}",
)
def test_real_data_2026_01_12_cid81_zero_price_tick_rejections(
    real_config, product_config
):
    """Loading contract 81 on 2026-01-12 produces zero price_tick rejections.

    slow: full-file CSV parse on real OMIE data.

    Post P2-F0 the loader output is already int-tick by construction, so
    every encoded price must decode back on-tick exactly. Any failure here
    means the loader produced a price that is not an exact multiple of
    ``price_tick`` — the same failure mode the 1.5-HOTFIX story fixed, now
    locked at the loader boundary instead of the row_to_order boundary.
    """
    orders = load_omie_orders(real_config)
    assert orders.shape[0] > 0, (
        f"Expected non-empty orders for {REAL_DELIVERY_DATE} contract "
        f"{REAL_CONTRACT_ID}; got shape {orders.shape}"
    )
    assert orders.dtype == jnp.int32

    price_col = np.asarray(orders[:, COL_PRICE])
    qty_col = np.asarray(orders[:, COL_QUANTITY])

    # Decoded EUR/MWh prices must be exact multiples of price_tick — i.e.
    # ``(decoded / price_tick)`` must round-trip to the same integer.
    # Because the loader now stores int-ticks directly, the check is trivial
    # on the decoded side but the assertion enumerates the rejection gate.
    price_tick = product_config.price_tick
    qty_tick = product_config.qty_tick

    decoded_prices = price_col * price_tick + product_config.price_min
    decoded_qtys = qty_col * qty_tick

    # Zero rejections: every decoded price lies within [price_min, price_max]
    # and every decoded quantity lies within [0, qty_max], exactly on-tick.
    assert (decoded_prices >= product_config.price_min - 1e-9).all()
    assert (decoded_prices <= product_config.price_max + 1e-9).all()
    assert (decoded_qtys >= 0.0).all()
    assert (decoded_qtys <= product_config.qty_max + 1e-9).all()

    # Tick alignment check: `round(decoded / tick) * tick == decoded` within
    # float noise. This is the exact "price_tick rejection" gate.
    price_rejects = np.abs(
        np.round(decoded_prices / price_tick) * price_tick - decoded_prices
    ) > 1e-6
    qty_rejects = np.abs(
        np.round(decoded_qtys / qty_tick) * qty_tick - decoded_qtys
    ) > 1e-6
    assert price_rejects.sum() == 0, (
        f"{price_rejects.sum()} rows failed price_tick alignment"
    )
    assert qty_rejects.sum() == 0, (
        f"{qty_rejects.sum()} rows failed qty_tick alignment"
    )


# ---------------------------------------------------------------------------
# 7. Cross-column invariant: peak_qty_int <= quantity_int row-wise
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_peak_qty_le_quantity_row_wise(real_config):
    """peak_qty_int must be <= quantity_int on every loaded row."""
    orders = load_omie_orders(real_config)
    if orders.shape[0] == 0:
        pytest.skip("No orders for this contract/date.")
    peak = np.asarray(orders[:, COL_PEAK_QTY])
    qty = np.asarray(orders[:, COL_QUANTITY])
    violations = peak > qty
    assert violations.sum() == 0, (
        f"{violations.sum()} rows violate peak_qty <= quantity "
        f"(first offending row={int(np.argmax(violations))})"
    )


# ---------------------------------------------------------------------------
# 8. Cross-column invariant: peak_qty_int == 0 for non-ICE rows
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_peak_qty_zero_for_non_ice_rows(real_config):
    """peak_qty must be exactly 0 for every row whose exec_restriction != ICE.

    ICE and ICEwPI are the only order_types that carry a non-zero peak_qty
    in OMIE data (ICEwPI is ICE with non-zero ppd; both share
    ExecRestriction.ICE). NON / IOC / FOK rows must have peak_qty_int == 0.
    """
    orders = load_omie_orders(real_config)
    if orders.shape[0] == 0:
        pytest.skip("No orders for this contract/date.")
    exec_col = np.asarray(orders[:, COL_EXEC_RESTRICTION])
    peak = np.asarray(orders[:, COL_PEAK_QTY])
    non_ice_mask = exec_col != int(ExecRestriction.ICE)
    if non_ice_mask.sum() == 0:
        pytest.skip("All rows are ICE; invariant vacuous.")
    violations = (peak[non_ice_mask] != 0)
    assert violations.sum() == 0, (
        f"{int(violations.sum())} non-ICE rows have non-zero peak_qty "
        f"(expected 0 for NON/IOC/FOK)"
    )


# ---------------------------------------------------------------------------
# 9. Encoded-range bounds: every encoded column fits int32 and sits within
#    its declared [lo, hi] tick-band.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_encoded_columns_fit_int32(real_config):
    """Every column value must fit signed int32."""
    orders = load_omie_orders(real_config)
    if orders.shape[0] == 0:
        pytest.skip("No orders for this contract/date.")
    arr = np.asarray(orders)
    assert arr.dtype == np.int32
    i32_min = np.iinfo(np.int32).min
    i32_max = np.iinfo(np.int32).max
    assert (arr >= i32_min).all() and (arr <= i32_max).all()


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_encoded_price_column_within_declared_tick_band(
    real_config, product_config
):
    """Price column (offset-encoded) sits within [0, (price_max - price_min) / price_tick]."""
    orders = load_omie_orders(real_config)
    if orders.shape[0] == 0:
        pytest.skip("No orders.")
    price_col = np.asarray(orders[:, COL_PRICE])
    lo = 0
    hi = int(round(
        (product_config.price_max - product_config.price_min)
        / product_config.price_tick
    ))
    assert (price_col >= lo).all(), (
        f"price col min {int(price_col.min())} < {lo}"
    )
    assert (price_col <= hi).all(), (
        f"price col max {int(price_col.max())} > {hi}"
    )


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_encoded_quantity_column_within_declared_tick_band(
    real_config, product_config
):
    """Quantity column sits within [0, qty_max / qty_tick]."""
    orders = load_omie_orders(real_config)
    if orders.shape[0] == 0:
        pytest.skip("No orders.")
    qty_col = np.asarray(orders[:, COL_QUANTITY])
    lo = 0
    hi = int(round(product_config.qty_max / product_config.qty_tick))
    assert (qty_col >= lo).all()
    assert (qty_col <= hi).all()


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_encoded_peak_qty_column_within_declared_tick_band(
    real_config, product_config
):
    """peak_qty column sits within [0, qty_max / qty_tick]."""
    orders = load_omie_orders(real_config)
    if orders.shape[0] == 0:
        pytest.skip("No orders.")
    peak_col = np.asarray(orders[:, COL_PEAK_QTY])
    hi = int(round(product_config.qty_max / product_config.qty_tick))
    assert (peak_col >= 0).all()
    assert (peak_col <= hi).all()


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_encoded_ppd_column_within_signed_tick_band(real_config, product_config):
    """ppd column is signed int-ticks; bounded by the SIDC-derived envelope.

    Market-spec v2.4 §13.6 (DA-PPD-2026-04) states there is no rule-level
    numeric magnitude bound on ICEwPI PPD — the spec only fixes the sign
    rule. A derived ceiling arises from the SIDC harmonized price range
    (-9,999 to +9,999 EUR/MWh, OQ-01): since every slice price equals
    ``base_price + k * PPD`` and must remain in that band,
    ``|PPD_EUR| <= 19998`` holds unconditionally.

    This test asserts that envelope in tick units, computed from
    ``price_tick`` (not hardcoded). It must pass on real data including
    the observed ``PPD = -3.00 EUR/MWh`` (-300 ticks) on 2026-01-12
    contract 81 ICEwPI BUY rows.
    """
    orders = load_omie_orders(real_config)
    if orders.shape[0] == 0:
        pytest.skip("No orders.")
    ppd_col = np.asarray(orders[:, COL_PPD])
    # SIDC-derived signed int-tick envelope: 19998 EUR/MWh expressed in
    # ticks via the inverse of price_tick.
    sidc_bound_eur = 19998.0
    envelope_ticks = int(round(sidc_bound_eur / product_config.price_tick))
    assert (ppd_col >= -envelope_ticks).all(), (
        f"ppd col has value {int(ppd_col.min())} < -{envelope_ticks} ticks "
        f"(exceeds SIDC-derived envelope)"
    )
    assert (ppd_col <= envelope_ticks).all(), (
        f"ppd col has value {int(ppd_col.max())} > +{envelope_ticks} ticks "
        f"(exceeds SIDC-derived envelope)"
    )


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_encoded_timestamp_column_within_day_band(real_config):
    """timestamp column (orders) is within [55200, 165600).

    OMIE-TAPE-CONTRACT-DATE-1 (group L re-baseline; was [0, 86400)):
    seconds since D-1 00:00, no modulo (market-spec v2.10 §13.4).
    """
    orders = load_omie_orders(real_config)
    if orders.shape[0] == 0:
        pytest.skip("No orders.")
    ts = np.asarray(orders[:, COL_TIMESTAMP])
    assert (ts >= 55200).all() and (ts < 165600).all()


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_encoded_trades_columns_fit_int32_and_within_bands(
    real_config, product_config
):
    """Every trades column fits int32; price / qty / timestamp within declared bands."""
    trades = load_omie_trades(real_config)
    if trades.shape[0] == 0:
        pytest.skip("No trades.")
    arr = np.asarray(trades)
    assert arr.dtype == np.int32

    price_col = arr[:, TCOL_PRICE]
    qty_col = arr[:, TCOL_QUANTITY]
    ts_col = arr[:, TCOL_MATCH_TIMESTAMP]

    p_hi = int(round(
        (product_config.price_max - product_config.price_min)
        / product_config.price_tick
    ))
    q_hi = int(round(product_config.qty_max / product_config.qty_tick))

    assert (price_col >= 0).all() and (price_col <= p_hi).all()
    assert (qty_col >= 0).all() and (qty_col <= q_hi).all()
    assert (ts_col >= 55200).all() and (ts_col < 165600).all()


# ---------------------------------------------------------------------------
# 10. Cache round-trip (schema v4): load → write v4 → reload is bitwise-identical
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_cache_schema_version_bumped_to_v4():
    """CACHE_SCHEMA_VERSION must be 4 post P2-F0."""
    from omiecid_simulator.jaxomiecid.omie_cache import CACHE_SCHEMA_VERSION
    assert CACHE_SCHEMA_VERSION == 4, (
        f"expected CACHE_SCHEMA_VERSION == 4 post P2-F0, got "
        f"{CACHE_SCHEMA_VERSION}"
    )


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_cache_round_trip_orders_bitwise_identical(tmp_path, real_config):
    """load → write v4 → reload yields a bitwise-identical orders ndarray."""
    from omiecid_simulator.jaxomiecid.omie_cache import (
        load_omie_cache,
        save_omie_cache,
    )

    orders = load_omie_orders(real_config)
    assert orders.dtype == jnp.int32

    # Redirect cache to tmp_path to avoid colliding with on-disk caches.
    write_cfg = OMIEDataConfig(
        data_path=real_config.data_path,
        contract_id=real_config.contract_id,
        delivery_date=real_config.delivery_date,
        cache_path=str(tmp_path),
    )
    save_omie_cache(write_cfg, orders=np.asarray(orders))
    payload = load_omie_cache(write_cfg)
    assert payload is not None, "cache reload returned None (schema mismatch?)"
    reloaded = jnp.asarray(payload["orders"])

    assert reloaded.dtype == orders.dtype, (
        f"reloaded dtype {reloaded.dtype} differs from original {orders.dtype}"
    )
    assert reloaded.shape == orders.shape
    assert bool(jnp.array_equal(reloaded, orders)), (
        "cache round-trip altered orders contents"
    )


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_cache_round_trip_trades_bitwise_identical(tmp_path, real_config):
    """load → write v4 → reload yields a bitwise-identical trades ndarray."""
    from omiecid_simulator.jaxomiecid.omie_cache import (
        load_omie_cache,
        save_omie_cache,
    )

    trades = load_omie_trades(real_config)
    assert trades.dtype == jnp.int32

    write_cfg = OMIEDataConfig(
        data_path=real_config.data_path,
        contract_id=real_config.contract_id,
        delivery_date=real_config.delivery_date,
        cache_path=str(tmp_path),
    )
    save_omie_cache(write_cfg, trades=np.asarray(trades))
    payload = load_omie_cache(write_cfg)
    assert payload is not None
    reloaded = jnp.asarray(payload["trades"])

    assert reloaded.dtype == trades.dtype
    assert reloaded.shape == trades.shape
    assert bool(jnp.array_equal(reloaded, trades)), (
        "cache round-trip altered trades contents"
    )


@pytest.mark.skipif(not _have_real_data(), reason="Real OMIE data not present")
def test_cache_rejects_v3_payload_as_stale(tmp_path, real_config):
    """A cache file written under schema v3 must be rejected after the v4 bump.

    Locks the schema-version guard: a stale v3 cache on disk (e.g. from a
    developer who ran the loader pre-P2-F0) must be treated as missing,
    triggering a fresh parse. Without this guard the reader could silently
    return float32 arrays, violating the dtype contract.
    """
    from omiecid_simulator.jaxomiecid.omie_cache import (
        _cache_filename,
        load_omie_cache,
    )

    cfg = OMIEDataConfig(
        data_path=real_config.data_path,
        contract_id=real_config.contract_id,
        delivery_date=real_config.delivery_date,
        cache_path=str(tmp_path),
    )
    # Manually write a v3 payload (old schema).
    import os
    os.makedirs(cfg.cache_path, exist_ok=True)
    np.savez_compressed(
        _cache_filename(cfg),
        schema_version=np.array(3, dtype=np.int32),
        orders=np.zeros((1, 13), dtype=np.float32),
    )
    payload = load_omie_cache(cfg)
    assert payload is None, (
        "v3 cache must be rejected post P2-F0, got a non-None payload"
    )
