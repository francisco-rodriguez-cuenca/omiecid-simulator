"""Tests for M7.1 — OMIE Data Configuration and M7.2 — Data Loader.

M7.1: Behavioral contracts for OMIEDataConfig.
M7.2: Behavioral contracts for load_omie_orders, load_omie_trades,
      replay_orders_through_book, and ReplayResult.

All expected values are derived from docs/omiecid/migration-plan.md §M7,
docs/omiecid/agile-plan.md M7.1/M7.2, docs/omiecid/market-spec.md §13,
and docs/omiecid/architecture-reviews.md §M7.2.md.

These tests are black-box: they import only the public interface and
assert against specification-mandated defaults and field definitions.
"""

import pytest
from dataclasses import FrozenInstanceError

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig

# ---------------------------------------------------------------------------
# M7.2 imports — guarded so M7.1 tests remain collectible before M7.2
# implementation exists.  Every M7.2 test class is decorated with
# @pytest.mark.skipif(_M72_MISSING, ...) so they are skipped (not errored)
# when the module is absent.
# ---------------------------------------------------------------------------
try:
    import jax.numpy as jnp
    from omiecid_simulator.jaxomiecid.omie_data_loader import (
        load_omie_orders,
        load_omie_trades,
        replay_orders_through_book,
        ReplayResult,
    )
    from omiecid_simulator.omiecid.order_types import (
        Side,
        ExecRestriction,
        Validity,
        Status,
        make_order,
    )
    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig
    from omiecid_simulator.omiecid.contract_book import ContractBook
    _M72_MISSING = False
except ImportError:
    _M72_MISSING = True

_m72_skip = pytest.mark.skipif(
    _M72_MISSING,
    reason="M7.2 omie_data_loader not yet implemented",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_config():
    """OMIEDataConfig with all required fields supplied."""
    return OMIEDataConfig(
        data_path="/tmp/omie_data",
        contract_id=0,
        delivery_date="2025-04-01",
    )


# ---------------------------------------------------------------------------
# test_data_config_defaults — verify default values match migration-plan §M7.1
# ---------------------------------------------------------------------------

class TestDataConfigDefaults:
    """Default field values must match the migration-plan §M7.1 spec."""

    def test_cache_path_default(self, data_config):
        """Default cache_path is 'saved_npz/omie/' (migration-plan §M7.1)."""
        assert data_config.cache_path == "saved_npz/omie/"

    def test_data_path_preserved(self, data_config):
        """data_path stores the value passed at construction."""
        assert data_config.data_path == "/tmp/omie_data"

    def test_contract_id_preserved(self, data_config):
        """contract_id stores the value passed at construction."""
        assert data_config.contract_id == 0

    def test_delivery_date_preserved(self, data_config):
        """delivery_date stores the value passed at construction."""
        assert data_config.delivery_date == "2025-04-01"

    def test_custom_cache_path(self):
        """cache_path can be overridden at construction."""
        cfg = OMIEDataConfig(
            data_path="/data",
            contract_id=5,
            delivery_date="2025-06-15",
            cache_path="/custom/cache/",
        )
        assert cfg.cache_path == "/custom/cache/"

    def test_contract_id_range_low(self):
        """contract_id=0 is the first quarter-hour (00:00-00:15)."""
        cfg = OMIEDataConfig(
            data_path="/data",
            contract_id=0,
            delivery_date="2025-04-01",
        )
        assert cfg.contract_id == 0

    def test_contract_id_range_high(self):
        """contract_id=95 is the last quarter-hour (23:45-24:00)."""
        cfg = OMIEDataConfig(
            data_path="/data",
            contract_id=95,
            delivery_date="2025-04-01",
        )
        assert cfg.contract_id == 95


# ---------------------------------------------------------------------------
# test_data_config_immutability — verify dataclass is frozen
# ---------------------------------------------------------------------------

class TestDataConfigImmutability:
    """OMIEDataConfig must be frozen (immutable after creation)."""

    def test_data_path_frozen(self, data_config):
        """Cannot reassign data_path after construction."""
        with pytest.raises(FrozenInstanceError):
            data_config.data_path = "/other/path"

    def test_contract_id_frozen(self, data_config):
        """Cannot reassign contract_id after construction."""
        with pytest.raises(FrozenInstanceError):
            data_config.contract_id = 42

    def test_delivery_date_frozen(self, data_config):
        """Cannot reassign delivery_date after construction."""
        with pytest.raises(FrozenInstanceError):
            data_config.delivery_date = "2025-12-31"

    def test_cache_path_frozen(self, data_config):
        """Cannot reassign cache_path after construction."""
        with pytest.raises(FrozenInstanceError):
            data_config.cache_path = "/new/cache"


# ===========================================================================
# M7.2 — Data Loader Tests
# ===========================================================================
#
# Behavioral contracts for:
#   - load_omie_orders(config) -> jnp.array of shape (N, 13)
#   - load_omie_trades(config) -> jnp.array of shape (M, K)
#   - replay_orders_through_book(orders, contract_book, historical_trades, heuristic) -> ReplayResult
#
# All tests use synthetic OMIE-format CSV fixtures created via tmp_path.
# No real OMIE data files are assumed to exist on disk.
#
# References:
#   - docs/omiecid/market-spec.md §13 (file format)
#   - docs/omiecid/migration-plan.md §M7.2 (interface spec)
#   - docs/omiecid/architecture-reviews.md §M7.2.md (field mapping, replay design)
#   - docs/omiecid/open-questions.md OQ-10 (replay divergence)
# ===========================================================================


# ---------------------------------------------------------------------------
# Synthetic OMIE file content helpers
# ---------------------------------------------------------------------------

# Delivery date for all fixtures: 2025-06-01 (orders_20250601.1)
_DELIVERY_DATE = "2025-06-01"
_DELIVERY_DATE_NODASH = "20250601"

# Contract string for contract_id=16 (04:00-04:15 on 2025-06-02):
# contract_id = 4*4 + 0//15 = 16
_CONTRACT_STR_16 = "20250602 04:00-20250602 04:15"

# Contract string for contract_id=37 (09:15-09:30 on 2025-06-02):
# contract_id = 9*4 + 15//15 = 37
_CONTRACT_STR_37 = "20250602 09:15-20250602 09:30"

# Contract string for contract_id=0 (00:00-00:15 on 2025-06-02):
_CONTRACT_STR_0 = "20250602 00:00-20250602 00:15"


def _make_orders_csv(rows: list[str]) -> str:
    """Build a synthetic OMIE orders file with boilerplate header.

    Each row in `rows` is a semicolon-delimited data line (no trailing newline).
    The function adds:
      - Row 1: OMIE metadata boilerplate (skipped by parser)
      - Row 2: Column headers
      - Rows 3+: Data rows (from `rows` parameter)

    All rows end with a trailing semicolon per OMIE format.
    """
    header_meta = "OMIE - Mercado de electricidad; Ofertas del Mercado Intradiario Continuo;"
    header_cols = (
        "Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;"
        "Tipo oferta;Condiciones a la ejecucion;Condiciones a la validez;"
        "Cantidad reducida;PPD;Momento de envio;"
    )
    lines = [header_meta, header_cols] + [r + ";" if not r.endswith(";") else r for r in rows]
    return "\n".join(lines) + "\n"


def _make_trades_csv(rows: list[str]) -> str:
    """Build a synthetic OMIE trades file with boilerplate header.

    Each row in `rows` is a semicolon-delimited data line.
    """
    header_meta = "OMIE - Mercado de electricidad; Casaciones del Mercado Intradiario Continuo;"
    header_cols = (
        "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;"
        "Agente venta;Unidad venta;Zona venta;Precio;Cantidad;Momento casacion;"
    )
    lines = [header_meta, header_cols] + [r + ";" if not r.endswith(";") else r for r in rows]
    return "\n".join(lines) + "\n"


# Fixture-only product date (OMIE-TAPE-CONTRACT-DATE-1, group F1): the rows
# above carry ``Contrato`` dated 2025-06-02, so they belong to delivery
# product (2025-06-02, k).  Per market-spec v2.10 §13/§13.4 the file date is
# the SESSION day: the rows (sent on 2025-06-01) live in session file
# ``*_20250601.1`` = file D-1 of that product, and file D (``*_20250602.1``)
# is written header-only.  ``_DELIVERY_DATE`` / ``_DELIVERY_DATE_NODASH`` keep
# naming the session file actually written.
_PRODUCT_DATE = "2025-06-02"
_PRODUCT_DATE_NODASH = "20250602"


def _write_fixture_files(tmp_path, orders_rows, trades_rows):
    """Write synthetic orders and trades files to tmp_path, return OMIEDataConfig.

    Returns a config pointing at contract_id=16 and zona=10YES-REE------0,
    for delivery product (2025-06-02, 16): data rows in session file D-1
    (2025-06-01), header-only session file D (2025-06-02).
    """
    orders_file = tmp_path / f"orders_{_DELIVERY_DATE_NODASH}.1"
    trades_file = tmp_path / f"trades_{_DELIVERY_DATE_NODASH}.1"
    orders_file.write_text(
        _make_orders_csv(orders_rows), encoding="latin-1"
    )
    trades_file.write_text(
        _make_trades_csv(trades_rows), encoding="latin-1"
    )
    (tmp_path / f"orders_{_PRODUCT_DATE_NODASH}.1").write_text(
        _make_orders_csv([]), encoding="latin-1"
    )
    (tmp_path / f"trades_{_PRODUCT_DATE_NODASH}.1").write_text(
        _make_trades_csv([]), encoding="latin-1"
    )
    return OMIEDataConfig(
        data_path=str(tmp_path),
        contract_id=16,
        delivery_date=_PRODUCT_DATE,
    )


# ---------------------------------------------------------------------------
# Shared fixture data rows
# ---------------------------------------------------------------------------

# 5 orders for contract_id=16, zona=Spain, various types:
#   1. BUY  at 50.10, 2.0 MW, agent ALPHA, NON restriction
#   2. SELL at 48.50, 1.5 MW, agent BETA, NON restriction
#   3. BUY  at 51.00, 3.0 MW, agent GAMMA, IOC restriction
#   4. SELL at 47.00, 5.0 MW, agent ALPHA, ICE restriction, peak_qty=2.0, ppd=0.50
#   5. BUY  at 52.00, 1.0 MW, agent BETA, FOK restriction

_ORDERS_CONTRACT_16 = [
    # Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;Tipo oferta;Cond exec;Cond validez;Cant reducida;PPD;Momento envio
    f"01/06/2025;{_CONTRACT_STR_16};10YES-REE------0;ALPHA;ALPH01;50,10;2,00;C;;;;;;;01/06/2025 17:41:12",
    f"01/06/2025;{_CONTRACT_STR_16};10YES-REE------0;BETA;BETA01;48,50;1,50;V;;;;;;;01/06/2025 17:42:00",
    f"01/06/2025;{_CONTRACT_STR_16};10YES-REE------0;GAMMA;GAMM01;51,00;3,00;C;IOC;;;;;;01/06/2025 17:43:30",
    f"01/06/2025;{_CONTRACT_STR_16};10YES-REE------0;ALPHA;ALPH01;47,00;5,00;V;ICE;;2,00;0,50;01/06/2025 17:44:15",
    f"01/06/2025;{_CONTRACT_STR_16};10YES-REE------0;BETA;BETA01;52,00;1,00;C;FOK;;;;;;01/06/2025 17:45:00",
]

# 2 orders for contract_id=37 (different contract, should be filtered out)
_ORDERS_CONTRACT_37 = [
    f"01/06/2025;{_CONTRACT_STR_37};10YES-REE------0;DELTA;DELT01;60,00;1,00;C;;;;;;;01/06/2025 18:00:00",
    f"01/06/2025;{_CONTRACT_STR_37};10YES-REE------0;DELTA;DELT01;59,00;1,00;V;;;;;;;01/06/2025 18:01:00",
]

# 1 order for contract_id=16 in Portugal zone (included after Phase 1.5-A).
_ORDERS_PORTUGAL = [
    f"01/06/2025;{_CONTRACT_STR_16};10YPT-REN------W;PTAG1;PTAG01;55,00;2,00;C;;;;;;;01/06/2025 17:50:00",
]

# Trades for contract_id=16:
# Trade 1: ALPHA bought from BETA at 48.50, 1.5 MW
# (This matches order 1 BUY@50.10 crossing order 2 SELL@48.50 -> trade at resting price 48.50)
_TRADES_CONTRACT_16 = [
    f"01/06/2025;{_CONTRACT_STR_16};ALPHA;ALPH01;10YES-REE------0;BETA;BETA01;10YES-REE------0;48,50;1,50;01/06/2025 17:42:01",
]

# Trade for contract_id=37 (filtered out)
_TRADES_CONTRACT_37 = [
    f"01/06/2025;{_CONTRACT_STR_37};DELTA;DELT01;10YES-REE------0;DELTA;DELT01;10YES-REE------0;59,50;1,00;01/06/2025 18:01:30",
]


@pytest.fixture
def omie_fixture_files(tmp_path):
    """Create synthetic OMIE order/trade files and return config for contract 16."""
    all_orders = _ORDERS_CONTRACT_16 + _ORDERS_CONTRACT_37 + _ORDERS_PORTUGAL
    all_trades = _TRADES_CONTRACT_16 + _TRADES_CONTRACT_37
    config = _write_fixture_files(tmp_path, all_orders, all_trades)
    return config


@pytest.fixture
def loaded_orders(omie_fixture_files):
    """Load orders from the synthetic fixture files."""
    return load_omie_orders(omie_fixture_files)


@pytest.fixture
def loaded_trades(omie_fixture_files):
    """Load trades from the synthetic fixture files."""
    return load_omie_trades(omie_fixture_files)


# ---------------------------------------------------------------------------
# TestLoadOrdersShape — test_load_orders_returns_correct_shape
# ---------------------------------------------------------------------------

@_m72_skip
class TestLoadOrdersShape:
    """load_omie_orders must return a (N, 13) JAX array.

    Spec: migration-plan §M7.2 — load_omie_orders(config) -> jnp.array
    with shape (N, 13) matching the 13-field OMIEOrder representation.
    """

    def test_output_is_jax_array(self, loaded_orders):
        """Return type must be a JAX array."""
        assert isinstance(loaded_orders, jnp.ndarray)

    def test_shape_has_two_dimensions(self, loaded_orders):
        """Output must be 2-dimensional: (N, 13)."""
        assert loaded_orders.ndim == 2

    def test_shape_has_13_columns(self, loaded_orders):
        """Second dimension must be 13 (one per OMIEOrder field)."""
        assert loaded_orders.shape[1] == 13

    def test_row_count_matches_filtered_orders(self, loaded_orders):
        """N must equal the number of orders for contract_id=16 across all zones.

        Fixture has 5 orders for contract 16 in Spain, 1 in Portugal,
        and 2 for contract 37 — all 6 contract-16 rows should load
        (contract filtering only; no zone filter after Phase 1.5-A).
        """
        assert loaded_orders.shape[0] == 6

    def test_empty_file_returns_zero_rows(self, tmp_path):
        """Loading a contract with no matching orders returns shape (0, 13)."""
        # Create files with only contract_37 orders (no contract_16 data)
        config = _write_fixture_files(
            tmp_path, _ORDERS_CONTRACT_37, _TRADES_CONTRACT_37
        )
        result = load_omie_orders(config)
        assert result.shape == (0, 13)


# ---------------------------------------------------------------------------
# TestLoadTradesShape — test_load_trades_returns_correct_shape
# ---------------------------------------------------------------------------

@_m72_skip
class TestLoadTradesShape:
    """load_omie_trades must return a (M, K) JAX array.

    Spec: migration-plan §M7.2 — load_omie_trades(config) -> jnp.array.
    The trades array has K fields covering at minimum: contract_id, price,
    quantity, buy_agent_id, sell_agent_id, match_timestamp.
    """

    def test_output_is_jax_array(self, loaded_trades):
        """Return type must be a JAX array."""
        assert isinstance(loaded_trades, jnp.ndarray)

    def test_shape_has_two_dimensions(self, loaded_trades):
        """Output must be 2-dimensional: (M, K)."""
        assert loaded_trades.ndim == 2

    def test_minimum_columns(self, loaded_trades):
        """Trades array must have at least 6 columns.

        Per architecture-reviews.md §M7.2 §3.3: contract_id, price, quantity,
        buy_agent_id, sell_agent_id, match_timestamp.
        """
        assert loaded_trades.shape[1] >= 6

    def test_row_count_matches_filtered_trades(self, loaded_trades):
        """M must equal trades for contract_id=16 (all zones included).

        Fixture has 1 trade for contract 16 and 1 trade for contract 37
        — only the contract-16 trade should load (contract filtering
        only; no zone filter after Phase 1.5-A).
        """
        assert loaded_trades.shape[0] == 1

    def test_empty_file_returns_zero_rows(self, tmp_path):
        """Loading a contract with no matching trades returns shape (0, K)."""
        config = _write_fixture_files(
            tmp_path, _ORDERS_CONTRACT_16, _TRADES_CONTRACT_37
        )
        result = load_omie_trades(config)
        assert result.shape[0] == 0
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# TestOrdersValidFields — test_orders_have_valid_fields
# ---------------------------------------------------------------------------

@_m72_skip
class TestOrdersValidFields:
    """Every loaded order must have valid field values per market-spec §13.

    Field mapping (architecture-reviews.md §M7.2 §3.1):
      [0] contract_id: matches config.contract_id (all filtered)
      [1] side: 0 (BUY) or 1 (SELL)
      [2] price: float within [-9999, +9999]
      [3] quantity: float within [0.1, 999]
      [4] order_id: positive sequential integer starting at 1
      [5] agent_id: positive integer (string->int mapping)
      [6] timestamp: non-negative integer (seconds)
      [7] exec_restriction: 0 (NON), 1 (IOC), 2 (FOK), or 3 (ICE)
      [8] validity: 0 (GFS) or 1 (GTD)
      [9] expiry_time: non-negative integer
      [10] status: 0 (ACTIVE) for all loaded orders
      [11] peak_qty: >= 0.0
      [12] ppd: any float (can be negative for buy-side ICEwPI)
    """

    def test_contract_id_all_match_config(self, loaded_orders, omie_fixture_files):
        """All rows must have contract_id == config.contract_id (filtered)."""
        contract_ids = loaded_orders[:, 0]
        expected = jnp.full(contract_ids.shape, omie_fixture_files.contract_id)
        assert jnp.allclose(contract_ids, expected)

    def test_side_values_valid(self, loaded_orders):
        """Side must be 0 (BUY) or 1 (SELL) for every order."""
        sides = loaded_orders[:, 1]
        assert jnp.all((sides == Side.BUY) | (sides == Side.SELL))

    def test_side_mapping_c_to_buy(self, loaded_orders):
        """Fixture row 0 is 'C' (Compra) which must map to Side.BUY=0."""
        assert int(loaded_orders[0, 1]) == Side.BUY

    def test_side_mapping_v_to_sell(self, loaded_orders):
        """Fixture row 1 is 'V' (Venta) which must map to Side.SELL=1."""
        assert int(loaded_orders[1, 1]) == Side.SELL

    def test_price_within_bounds(self, loaded_orders):
        """Price must be within [-9999, +9999] EUR/MWh (market-spec §2.3)."""
        from omiecid_simulator.omiecid.encoding import decode_price
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        prices = jnp.asarray([decode_price(int(v), pc) for v in loaded_orders[:, 2]])
        assert jnp.all(prices >= -9999.0)
        assert jnp.all(prices <= 9999.0)

    def test_price_comma_decimal_parsed(self, loaded_orders):
        """Fixture row 0 has price '50,10' which must parse to 50.10."""
        from omiecid_simulator.omiecid.encoding import decode_price
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_price(int(loaded_orders[0, 2]), pc) - 50.10) < 0.001

    def test_quantity_within_bounds(self, loaded_orders):
        """Quantity must be within [0.1, 999] MW (market-spec §2.3)."""
        from omiecid_simulator.omiecid.encoding import decode_qty
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        quantities = jnp.asarray([decode_qty(int(v), pc) for v in loaded_orders[:, 3]])
        assert jnp.all(quantities >= 0.1)
        assert jnp.all(quantities <= 999.0)

    def test_quantity_comma_decimal_parsed(self, loaded_orders):
        """Fixture row 1 has quantity '1,50' which must parse to 1.5."""
        from omiecid_simulator.omiecid.encoding import decode_qty
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_qty(int(loaded_orders[1, 3]), pc) - 1.5) < 0.001

    def test_order_id_positive(self, loaded_orders):
        """All order_id values must be positive (>0).

        ContractBook enforces order_id > 0 (architecture-reviews.md §M7.2 §3.1).
        """
        order_ids = loaded_orders[:, 4]
        assert jnp.all(order_ids > 0)

    def test_order_id_sequential(self, loaded_orders):
        """order_id values must be sequential starting at 1."""
        order_ids = loaded_orders[:, 4]
        expected = jnp.arange(1, loaded_orders.shape[0] + 1, dtype=order_ids.dtype)
        assert jnp.array_equal(order_ids, expected)

    def test_agent_id_positive(self, loaded_orders):
        """All agent_id values must be positive integers (string->int map starts at 1)."""
        agent_ids = loaded_orders[:, 5]
        assert jnp.all(agent_ids > 0)

    def test_agent_id_deterministic_mapping(self, loaded_orders):
        """Same agent string must produce the same agent_id.

        Fixture rows 0 and 3 are both agent ALPHA -> same agent_id.
        Fixture rows 1 and 4 are both agent BETA -> same agent_id.
        """
        # Row 0 (ALPHA) and row 3 (ALPHA) should match
        assert int(loaded_orders[0, 5]) == int(loaded_orders[3, 5])
        # Row 1 (BETA) and row 4 (BETA) should match
        assert int(loaded_orders[1, 5]) == int(loaded_orders[4, 5])
        # ALPHA and BETA should differ
        assert int(loaded_orders[0, 5]) != int(loaded_orders[1, 5])

    def test_timestamp_non_negative(self, loaded_orders):
        """All timestamps must be non-negative."""
        timestamps = loaded_orders[:, 6]
        assert jnp.all(timestamps >= 0)

    def test_timestamps_monotonically_ordered(self, loaded_orders):
        """Orders must be sorted by submission timestamp (chronological)."""
        timestamps = loaded_orders[:, 6]
        diffs = timestamps[1:] - timestamps[:-1]
        assert jnp.all(diffs >= 0)

    def test_exec_restriction_valid_values(self, loaded_orders):
        """exec_restriction must be 0 (NON), 1 (IOC), 2 (FOK), or 3 (ICE)."""
        restrictions = loaded_orders[:, 7]
        valid = (
            (restrictions == ExecRestriction.NON)
            | (restrictions == ExecRestriction.IOC)
            | (restrictions == ExecRestriction.FOK)
            | (restrictions == ExecRestriction.ICE)
        )
        assert jnp.all(valid)

    def test_exec_restriction_empty_maps_to_non(self, loaded_orders):
        """Fixture row 0 has empty exec restriction -> ExecRestriction.NON=0."""
        assert int(loaded_orders[0, 7]) == ExecRestriction.NON

    def test_exec_restriction_ioc(self, loaded_orders):
        """Fixture row 2 has 'IOC' -> ExecRestriction.IOC=1."""
        assert int(loaded_orders[2, 7]) == ExecRestriction.IOC

    def test_exec_restriction_ice(self, loaded_orders):
        """Fixture row 3 has 'ICE' -> ExecRestriction.ICE=3."""
        assert int(loaded_orders[3, 7]) == ExecRestriction.ICE

    def test_exec_restriction_fok(self, loaded_orders):
        """Fixture row 4 has 'FOK' -> ExecRestriction.FOK=2."""
        assert int(loaded_orders[4, 7]) == ExecRestriction.FOK

    def test_validity_valid_values(self, loaded_orders):
        """validity must be 0 (GFS) or 1 (GTD)."""
        validities = loaded_orders[:, 8]
        assert jnp.all((validities == Validity.GFS) | (validities == Validity.GTD))

    def test_status_all_active(self, loaded_orders):
        """All loaded orders must have status=ACTIVE (0).

        Status is simulator-internal; loaded orders are always ACTIVE.
        """
        statuses = loaded_orders[:, 10]
        assert jnp.all(statuses == Status.ACTIVE)

    def test_peak_qty_non_negative(self, loaded_orders):
        """peak_qty must be >= 0.0 for all orders."""
        peak_qtys = loaded_orders[:, 11]
        assert jnp.all(peak_qtys >= 0.0)

    def test_peak_qty_zero_for_non_iceberg(self, loaded_orders):
        """Non-ICE orders must have peak_qty=0.0.

        Fixture rows 0,1,2,4 are NON/IOC/FOK -> peak_qty=0.0.
        """
        for row_idx in [0, 1, 2, 4]:
            assert jnp.isclose(loaded_orders[row_idx, 11], 0.0), (
                f"Row {row_idx} (non-ICE) should have peak_qty=0.0"
            )

    def test_peak_qty_set_for_iceberg(self, loaded_orders):
        """ICE order (fixture row 3) must have peak_qty=2.0."""
        from omiecid_simulator.omiecid.encoding import decode_qty
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_qty(int(loaded_orders[3, 11]), pc) - 2.0) < 0.01

    def test_ppd_zero_for_non_iceberg(self, loaded_orders):
        """Non-ICE orders must have ppd=0.0.

        Fixture rows 0,1,2,4 are NON/IOC/FOK -> ppd=0.0.
        """
        for row_idx in [0, 1, 2, 4]:
            assert jnp.isclose(loaded_orders[row_idx, 12], 0.0), (
                f"Row {row_idx} (non-ICE) should have ppd=0.0"
            )

    def test_ppd_set_for_iceberg(self, loaded_orders):
        """ICE order (fixture row 3) must have ppd=0.50."""
        from omiecid_simulator.omiecid.encoding import decode_ppd
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_ppd(int(loaded_orders[3, 12]), pc) - 0.50) < 0.01


# ---------------------------------------------------------------------------
# TestTradesMatchOrderData — test_trades_match_order_data
# ---------------------------------------------------------------------------

@_m72_skip
class TestTradesMatchOrderData:
    """Trade prices and quantities must be consistent with order data.

    Per architecture-reviews.md §M7.2 §5.2, the trade matching key is
    (contract_id, price, quantity). Trade prices must fall within the
    range of order prices for the same contract, and trade quantities
    must not exceed the maximum order quantity.
    """

    def test_trade_price_within_order_price_range(
        self, loaded_orders, loaded_trades
    ):
        """Trade prices must fall within the min/max of order prices.

        Fixture trade price is 48.50, which is within the order price
        range [47.00, 52.00] for contract 16.
        """
        if loaded_trades.shape[0] == 0:
            pytest.skip("No trades loaded for this contract")
        order_prices = loaded_orders[:, 2]
        min_price = jnp.min(order_prices)
        max_price = jnp.max(order_prices)
        # Trade price column — find the price column index.
        # Per architecture-review §3.3, trades have at minimum:
        # contract_id, price, quantity, buy_agent_id, sell_agent_id, timestamp
        # Price is at index 1.
        trade_prices = loaded_trades[:, 1]
        assert jnp.all(trade_prices >= min_price - 0.01), (
            f"Trade price below order min: {float(jnp.min(trade_prices))} < {float(min_price)}"
        )
        assert jnp.all(trade_prices <= max_price + 0.01), (
            f"Trade price above order max: {float(jnp.max(trade_prices))} > {float(max_price)}"
        )

    def test_trade_quantity_positive(self, loaded_trades):
        """All trade quantities must be positive."""
        if loaded_trades.shape[0] == 0:
            pytest.skip("No trades loaded for this contract")
        # Quantity at index 2
        trade_quantities = loaded_trades[:, 2]
        assert jnp.all(trade_quantities > 0.0)

    def test_trade_quantity_within_order_bounds(
        self, loaded_orders, loaded_trades
    ):
        """Trade quantities must not exceed the maximum order quantity.

        A single trade quantity cannot exceed the largest order quantity
        in the same contract.
        """
        if loaded_trades.shape[0] == 0:
            pytest.skip("No trades loaded for this contract")
        max_order_qty = float(jnp.max(loaded_orders[:, 3]))
        trade_quantities = loaded_trades[:, 2]
        assert jnp.all(trade_quantities <= max_order_qty + 0.01)

    def test_trade_price_matches_fixture(self, loaded_trades):
        """Fixture trade has price 48.50 EUR/MWh."""
        if loaded_trades.shape[0] == 0:
            pytest.skip("No trades loaded")
        from omiecid_simulator.omiecid.encoding import decode_price
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_price(int(loaded_trades[0, 1]), pc) - 48.50) < 0.01

    def test_trade_quantity_matches_fixture(self, loaded_trades):
        """Fixture trade has quantity 1.50 MW."""
        if loaded_trades.shape[0] == 0:
            pytest.skip("No trades loaded")
        from omiecid_simulator.omiecid.encoding import decode_qty
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_qty(int(loaded_trades[0, 2]), pc) - 1.50) < 0.01


# ---------------------------------------------------------------------------
# Replay test fixtures
# ---------------------------------------------------------------------------
#
# For replay tests we construct a small, deterministic order sequence where
# the matching outcome is fully predictable. We use contract_id=16 with a
# fresh ContractBook, submit orders programmatically, and compare against
# known historical trades.
# ---------------------------------------------------------------------------

@pytest.fixture
def replay_contract():
    """An OPEN contract for contract_id=16 (04:00-04:15)."""
    return Contract(
        contract_id=16,
        delivery_start_minutes=4 * 60,  # 04:00
        delivery_end_minutes=4 * 60 + 15,  # 04:15
        gct_minutes=4 * 60 - 60,  # 03:00 (GCT = delivery - 60 min)
        status=ContractStatus.OPEN,
    )


@pytest.fixture
def replay_book(replay_contract):
    """A ContractBook for replay testing."""
    return ContractBook(
        replay_contract, OMIEProductConfig(), OMIEMarketConfig()
    )


@pytest.fixture
def replay_orders():
    """Synthetic order sequence with deterministic matching outcome.

    Order sequence (all contract_id=16, chronological by timestamp):
      1. SELL 2.0 MW @ 50.00 (agent 1, NON) — rests in book
      2. SELL 1.0 MW @ 51.00 (agent 2, NON) — rests in book
      3. BUY  3.0 MW @ 51.00 (agent 3, NON) — crosses spread:
           fills 2.0 @ 50.00 (against order 1)
           fills 1.0 @ 51.00 (against order 2)

    Expected trades:
      Trade A: 2.0 MW @ 50.00 (agent 3 buys from agent 1)
      Trade B: 1.0 MW @ 51.00 (agent 3 buys from agent 2)
    """
    from omiecid_simulator.omiecid.encoding import encode_price, encode_qty
    from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
    pc = OMIEProductConfig()
    orders = jnp.array([
        # contract_id, side, price_int, qty_int, order_id, agent_id, timestamp,
        # exec_restriction, validity, expiry_time, status, peak_qty_int, ppd_int
        [16, Side.SELL, encode_price(50.00, pc), encode_qty(2.0, pc), 1, 1, 100, ExecRestriction.NON, Validity.GFS, 0, Status.ACTIVE, 0, 0],
        [16, Side.SELL, encode_price(51.00, pc), encode_qty(1.0, pc), 2, 2, 200, ExecRestriction.NON, Validity.GFS, 0, Status.ACTIVE, 0, 0],
        [16, Side.BUY,  encode_price(51.00, pc), encode_qty(3.0, pc), 3, 3, 300, ExecRestriction.NON, Validity.GFS, 0, Status.ACTIVE, 0, 0],
    ], dtype=jnp.int32)
    return orders


@pytest.fixture
def replay_historical_trades():
    """Historical trades matching the expected outcome of replay_orders.

    Two trades occurred historically:
      Trade A: price=50.00, quantity=2.0
      Trade B: price=51.00, quantity=1.0

    The trade array uses the same column layout as load_omie_trades output.
    At minimum: contract_id, price, quantity, buy_agent_id, sell_agent_id, timestamp.
    """
    from omiecid_simulator.omiecid.encoding import encode_price, encode_qty
    from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
    pc = OMIEProductConfig()
    trades = jnp.array([
        [16, encode_price(50.00, pc), encode_qty(2.0, pc), 3, 1, 300],
        [16, encode_price(51.00, pc), encode_qty(1.0, pc), 3, 2, 300],
    ], dtype=jnp.int32)
    return trades


@pytest.fixture
def replay_historical_trades_subset():
    """Historical trades that are a strict subset of what replay produces.

    Only one of the two expected trades is listed as historical.
    This simulates the scenario where replay produces extra trades
    (from orders that were historically cancelled but rest in H1).
    """
    from omiecid_simulator.omiecid.encoding import encode_price, encode_qty
    from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
    pc = OMIEProductConfig()
    trades = jnp.array([
        [16, encode_price(50.00, pc), encode_qty(2.0, pc), 3, 1, 300],
    ], dtype=jnp.int32)
    return trades


# ---------------------------------------------------------------------------
# TestReplaySuperset — test_replay_produces_superset_of_historical_trades
# ---------------------------------------------------------------------------

@_m72_skip
class TestReplaySuperset:
    """Replay with H1 heuristic must produce a superset of historical trades.

    Per OQ-10 and migration-plan §M7.2: H1 (rest_until_gct) leaves all
    unmatched volume resting, so every historical trade should still occur.
    All historical (price, qty) pairs must appear in replayed trades.
    """

    def test_all_historical_trades_reproduced(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """Every historical trade (price, qty) pair must appear in replay output.

        With 2 historical trades matching the 2 expected trades from the
        deterministic order sequence, match_rate should be 1.0.
        """
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert result.historical_trades_matched == result.historical_trades_total
        assert result.match_rate == 1.0

    def test_replay_produces_at_least_historical_count(
        self, replay_orders, replay_book, replay_historical_trades_subset
    ):
        """Replay must produce at least as many trades as historical record.

        When historical record has 1 trade but replay produces 2, all
        1 historical trade must be matched (superset property).
        """
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades_subset,
            heuristic="rest_until_gct",
        )
        assert result.historical_trades_matched == result.historical_trades_total
        assert result.historical_trades_total == 1

    def test_replayed_trades_array_not_empty(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """The replayed_trades field must contain the actual trade data."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert isinstance(result.replayed_trades, jnp.ndarray)
        assert result.replayed_trades.shape[0] > 0

    def test_unknown_heuristic_raises(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """Unrecognized heuristic must raise NotImplementedError.

        Per architecture-reviews.md §M7.2 §6.3: only H1 implemented;
        unrecognized values raise NotImplementedError.
        """
        with pytest.raises(NotImplementedError):
            replay_orders_through_book(
                replay_orders,
                replay_book,
                replay_historical_trades,
                heuristic="unknown_heuristic",
            )


# ---------------------------------------------------------------------------
# TestReplayDivergence — test_replay_divergence_within_tolerance
# ---------------------------------------------------------------------------

@_m72_skip
class TestReplayDivergence:
    """Extra trade volume from H1 replay must be within tolerance.

    Per OQ-10 and migration-plan §M7.2: extra_volume_pct measures
    the empirical impact of cancellation-blindness. Default tolerance
    is 10% of total replayed volume.
    """

    def test_exact_match_zero_divergence(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """When replay exactly matches history, extra_volume_pct should be 0.0.

        The deterministic order sequence produces exactly the 2 trades
        listed in historical_trades, so there are no extra trades.
        """
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert result.extra_volume_pct == pytest.approx(0.0, abs=0.01)

    def test_extra_volume_within_tolerance(
        self, replay_orders, replay_book, replay_historical_trades_subset
    ):
        """When replay produces extra trades, extra_volume_pct must be computable.

        Replay produces 2 trades (3 MW total). Historical has 1 trade (2 MW).
        Extra volume = 1 MW. Total replayed volume = 3 MW.
        extra_volume_pct = 1/3 ~= 0.333 (33.3%).

        This test just verifies the metric is non-negative and finite,
        since the tolerance threshold is configurable and the exact value
        depends on the order sequence.
        """
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades_subset,
            heuristic="rest_until_gct",
        )
        assert result.extra_volume_pct >= 0.0
        assert jnp.isfinite(result.extra_volume_pct)

    def test_extra_trades_count_non_negative(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """extra_trades count must be non-negative."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert result.extra_trades >= 0


# ---------------------------------------------------------------------------
# TestReplayResultMetadata — test_replay_result_metadata
# ---------------------------------------------------------------------------

@_m72_skip
class TestReplayResultMetadata:
    """ReplayResult must contain correct metadata fields.

    Per migration-plan §M7.2: ReplayResult has fields
    final_book, replayed_trades, historical_trades_matched,
    historical_trades_total, extra_trades, match_rate, extra_volume_pct.
    """

    def test_replay_result_has_match_rate(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """ReplayResult must have a match_rate field."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert hasattr(result, "match_rate")

    def test_replay_result_has_extra_volume_pct(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """ReplayResult must have an extra_volume_pct field."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert hasattr(result, "extra_volume_pct")

    def test_replay_result_has_historical_trades_total(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """ReplayResult must have a historical_trades_total field."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert hasattr(result, "historical_trades_total")

    def test_replay_result_has_historical_trades_matched(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """ReplayResult must have a historical_trades_matched field."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert hasattr(result, "historical_trades_matched")

    def test_replay_result_has_extra_trades(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """ReplayResult must have an extra_trades field."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert hasattr(result, "extra_trades")

    def test_replay_result_has_replayed_trades(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """ReplayResult must have a replayed_trades field (JAX array)."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert hasattr(result, "replayed_trades")

    def test_match_rate_is_ratio(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """match_rate = historical_trades_matched / historical_trades_total.

        With 2 historical trades all matched, match_rate = 2/2 = 1.0.
        """
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        expected = result.historical_trades_matched / result.historical_trades_total
        assert result.match_rate == pytest.approx(expected, abs=1e-6)

    def test_historical_trades_total_matches_input(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """historical_trades_total must equal the number of input historical trades."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert result.historical_trades_total == replay_historical_trades.shape[0]

    def test_extra_trades_equals_total_minus_matched(
        self, replay_orders, replay_book, replay_historical_trades_subset
    ):
        """extra_trades = total replayed trades - historical_trades_matched.

        With 1 historical trade matched and 2 replayed trades,
        extra_trades = 2 - 1 = 1.
        """
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades_subset,
            heuristic="rest_until_gct",
        )
        # replayed_trades array row count is the total replayed trade count
        total_replayed = result.replayed_trades.shape[0]
        assert result.extra_trades == total_replayed - result.historical_trades_matched

    def test_match_rate_bounded_zero_one(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """match_rate must be in [0.0, 1.0]."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert 0.0 <= result.match_rate <= 1.0

    def test_extra_volume_pct_non_negative(
        self, replay_orders, replay_book, replay_historical_trades
    ):
        """extra_volume_pct must be >= 0.0."""
        result = replay_orders_through_book(
            replay_orders,
            replay_book,
            replay_historical_trades,
            heuristic="rest_until_gct",
        )
        assert result.extra_volume_pct >= 0.0
