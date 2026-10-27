"""Audit tests for M7.2 — OMIE Data Loader.

Independent black-box audit tests derived from:
  - docs/omiecid/agile-plan.md (M7.2 story definition)
  - docs/omiecid/market-spec.md §13 (data format)
  - docs/omiecid/architecture-reviews.md §M7.2 (field mapping, replay design)
  - docs/omiecid/open-questions.md OQ-10 (replay divergence)
  - omiecid_simulator/omiecid/order_types.py (13-field order, enums)

Focus: edge cases, boundary conditions, acceptance criteria gaps.
"""

import os
import textwrap
from pathlib import Path

import jax.numpy as jnp
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    ReplayResult,
    _map_exec_restriction,
    _map_validity,
    _parse_contract_id,
    _parse_timestamp,
    load_omie_orders,
    load_omie_trades,
    replay_orders_through_book,
)
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    Status,
    Validity,
)


# ============================================================================
# Fixture helpers — synthetic OMIE-format CSV generation
# ============================================================================

# Standard metadata + header for orders file (rows 1-2)
ORDERS_HEADER = (
    "OMIE - Ofertas del Mercado Intradiario Continuo;\n"
    "Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;Tipo oferta;"
    "Condiciones a la ejecucion;Condiciones a la validez;"
    "Cantidad reducida;PPD;Momento de envio;\n"
)

# Standard metadata + header for trades file (rows 1-2)
# Row 1 = OMIE boilerplate metadata (skipped by parser)
# Row 2 = column headers (used as pandas header)
TRADES_HEADER = (
    "OMIE - Casaciones del Mercado Intradiario Continuo;\n"
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;"
    "Agente venta;Unidad venta;Zona venta;"
    "Precio;Cantidad;Momento casacion;\n"
)


def _write_orders_file(tmp_path, date_nodash, rows_text):
    """Write a synthetic orders file with proper header."""
    path = tmp_path / f"orders_{date_nodash}.1"
    content = ORDERS_HEADER + rows_text
    path.write_text(content, encoding="latin-1")
    return str(tmp_path)


def _write_trades_file(tmp_path, date_nodash, rows_text):
    """Write a synthetic trades file with proper header."""
    path = tmp_path / f"trades_{date_nodash}.1"
    content = TRADES_HEADER + rows_text
    path.write_text(content, encoding="latin-1")
    return str(tmp_path)


# Fixture-only product date (OMIE-TAPE-CONTRACT-DATE-1, group F1).  Every
# fixture row carries a ``Contrato`` dated 2025-06-02, i.e. delivery product
# (2025-06-02, k).  Per market-spec v2.10 §13/§13.4 the written session file
# ``*_20250601.1`` is file D-1 of that product; the product also spans session
# file D (``*_20250602.1``).  ``DATE_NODASH`` / ``DELIVERY_DATE`` keep naming
# the session file the tests write.
_PRODUCT_DATE = "2025-06-02"
_PRODUCT_SESSION_FILES_NODASH = ("20250601", "20250602")  # (D-1, D)


def _complete_product_files(data_path):
    """Add a header-only file for every product session file not written.

    Uses the same header writers as the data files (unaccented
    ``Momento casacion``).  Files a test wrote are never touched.
    """
    base = Path(data_path)
    for date_nodash in _PRODUCT_SESSION_FILES_NODASH:
        if not (base / f"orders_{date_nodash}.1").exists():
            _write_orders_file(base, date_nodash, "")
        if not (base / f"trades_{date_nodash}.1").exists():
            _write_trades_file(base, date_nodash, "")


def _make_config(data_path, contract_id=16, delivery_date=_PRODUCT_DATE):
    """Build an OMIEDataConfig (zone filtering removed in Phase 1.5-A).

    Completes the product's session files (D-1, D) with header-only files
    first, so the config names a complete product.
    """
    _complete_product_files(data_path)
    return OMIEDataConfig(
        data_path=data_path,
        contract_id=contract_id,
        delivery_date=delivery_date,
    )


# Contract 16 = 04:00-04:15
CONTRACT_16_STR = "20250602 04:00-20250602 04:15"
# Contract 0 = 00:00-00:15
CONTRACT_0_STR = "20250602 00:00-20250602 00:15"
# Contract 95 = 23:45-24:00
CONTRACT_95_STR = "20250602 23:45-20250603 00:00"
# Contract 37 = 09:15-09:30
CONTRACT_37_STR = "20250602 09:15-20250602 09:30"

DATE_NODASH = "20250601"
DELIVERY_DATE = "2025-06-01"
SPAIN_ZONA = "10YES-REE------0"
PORTUGAL_ZONA = "10YPT-REN------W"


# ============================================================================
# TestParseContractId — boundary cases for contract_id mapping
# ============================================================================

class TestParseContractId:
    """Contract ID mapping from Contrato strings (0-95 range)."""

    def test_contract_0_midnight_start(self):
        """Contract 0 = 00:00-00:15 => contract_id 0."""
        assert _parse_contract_id("20250602 00:00-20250602 00:15") == 0

    def test_contract_95_last_quarter(self):
        """Contract 95 = 23:45-24:00 => contract_id 95."""
        assert _parse_contract_id("20250602 23:45-20250603 00:00") == 95

    def test_contract_1_first_nontrivial(self):
        """Contract 1 = 00:15-00:30 => contract_id 1."""
        assert _parse_contract_id("20250602 00:15-20250602 00:30") == 1

    def test_contract_4_first_whole_hour(self):
        """Contract 4 = 01:00-01:15 => contract_id 4."""
        assert _parse_contract_id("20250602 01:00-20250602 01:15") == 4

    def test_contract_16_four_am(self):
        """Contract 16 = 04:00-04:15 => contract_id 16."""
        assert _parse_contract_id(CONTRACT_16_STR) == 16

    def test_contract_48_noon(self):
        """Contract 48 = 12:00-12:15 => contract_id 48."""
        assert _parse_contract_id("20250602 12:00-20250602 12:15") == 48

    def test_contract_with_leading_trailing_spaces(self):
        """Parser should handle whitespace around the Contrato string."""
        assert _parse_contract_id("  20250602 04:00-20250602 04:15  ") == 16


# ============================================================================
# TestExecRestrictionMapping — completeness of all enum values
# ============================================================================

class TestExecRestrictionMapping:
    """ExecRestriction mapping: empty->NON, IOC->1, FOK->2, ICE->3."""

    def test_empty_string_maps_to_non(self):
        assert _map_exec_restriction("") == int(ExecRestriction.NON)

    def test_none_maps_to_non(self):
        assert _map_exec_restriction(None) == int(ExecRestriction.NON)

    def test_nan_maps_to_non(self):
        import math
        assert _map_exec_restriction(float("nan")) == int(ExecRestriction.NON)

    def test_ioc_maps_correctly(self):
        assert _map_exec_restriction("IOC") == int(ExecRestriction.IOC)

    def test_fok_maps_correctly(self):
        assert _map_exec_restriction("FOK") == int(ExecRestriction.FOK)

    def test_ice_maps_correctly(self):
        assert _map_exec_restriction("ICE") == int(ExecRestriction.ICE)

    def test_lowercase_ioc(self):
        """Case insensitivity: 'ioc' should map to IOC."""
        assert _map_exec_restriction("ioc") == int(ExecRestriction.IOC)

    def test_mixed_case_fok(self):
        """Case insensitivity: 'Fok' should map to FOK."""
        assert _map_exec_restriction("Fok") == int(ExecRestriction.FOK)

    def test_whitespace_padded_ice(self):
        """Whitespace around value should be handled."""
        assert _map_exec_restriction("  ICE  ") == int(ExecRestriction.ICE)

    def test_unknown_string_maps_to_non(self):
        """Unknown restriction values default to NON."""
        assert _map_exec_restriction("UNKNOWN") == int(ExecRestriction.NON)


# ============================================================================
# TestValidityMapping
# ============================================================================

class TestValidityMapping:
    """Validity mapping: empty->GFS, GTD->1."""

    def test_empty_maps_to_gfs(self):
        assert _map_validity("") == int(Validity.GFS)

    def test_none_maps_to_gfs(self):
        assert _map_validity(None) == int(Validity.GFS)

    def test_gtd_maps_correctly(self):
        assert _map_validity("GTD") == int(Validity.GTD)

    def test_lowercase_gtd(self):
        assert _map_validity("gtd") == int(Validity.GTD)

    def test_unknown_maps_to_gfs(self):
        assert _map_validity("RANDOM") == int(Validity.GFS)


# ============================================================================
# TestAgentMapping — DELETED in 1.5-B.
# ----------------------------------------------------------------------------
# The 5 Agente-only mapping tests (``test_single_agent``,
# ``test_alphabetical_ordering``, ``test_determinism_independent_of_input_order``,
# ``test_ids_start_at_one``, ``test_no_duplicate_ids``) asserted the legacy
# ``_build_agent_mapping(agent_series) -> dict[str, int]`` contract. Under the
# 1.5-B pair-based signature ``_build_agent_mapping(agente_series,
# unidad_series) -> tuple[dict, dict]`` those calls would raise TypeError or
# unpack-errors and their hard-coded integer expectations (e.g. ``ALPHA=1,
# BETA=2, ZETA=3``) are unreachable. The class is replaced wholesale by pair-
# oriented coverage in ``tests/omiecid/test_agente_unidad_concatenation.py``
# (see TestBuildAgentMappingSignature, TestPairDeterminism,
# TestPairDistinctness, TestRoundTripLosslessness, TestIndexing).
# ============================================================================


# ============================================================================
# TestTimestampParsing — DELETED in 1.5-C.
# ----------------------------------------------------------------------------
# The 3 single-argument timestamp tests (``test_known_timestamp_parses``,
# ``test_ordering_preserved``, ``test_midnight_crossing``) called
# ``_parse_timestamp("DD/MM/YYYY HH:MM:SS")`` with a single argument and
# asserted Unix-epoch-scale behaviour (``ts > 0``, cross-midnight ordering).
# After Story 1.5-C, ``_parse_timestamp`` requires a second ``epoch_base: int``
# argument; single-argument calls raise TypeError.  Additionally, the
# Unix-epoch-scale assertions are wrong after the session-relative conversion
# (values are now in [0, 86400), not ~1.75×10⁹).
# These tests are replaced wholesale by ``TestTimestampSessionRelative`` and
# related classes in ``tests/omiecid/test_timestamp_session_relative.py``
# (see TestParseTimestampKnownValues, TestParseTimestampSignature,
# TestSessionRelativeRange, TestMonotonicOrderingPreserved, TestFloat32Roundtrip).
# ============================================================================


# ============================================================================
# TestLoadOrdersEdgeCases — CSV parsing edge cases
# ============================================================================

class TestLoadOrdersEdgeCases:
    """Edge cases in CSV parsing: empty files, trailing semicolons, etc."""

    def test_empty_file_returns_zero_rows(self, tmp_path):
        """Orders file with only headers returns shape (0, 13)."""
        data_path = _write_orders_file(tmp_path, DATE_NODASH, "")
        config = _make_config(data_path)
        result = load_omie_orders(config)
        assert result.shape == (0, 13)

    def test_no_matching_contract_returns_empty(self, tmp_path):
        """Rows exist for a different contract => empty result."""
        row = (
            f"01/06/2025;{CONTRACT_37_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape == (0, 13)

    def test_trailing_semicolons_handled(self, tmp_path):
        """Rows with trailing semicolons parse correctly."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;;;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape[0] == 1
        assert result.shape[1] == 13

    def test_comma_decimal_price_parsed(self, tmp_path):
        """Price '18,01' should be parsed as 18.01."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "18,01;7,50;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape[0] == 1
        # Column 2 = price (int32 tick-encoded post P2-F0)
        from omiecid_simulator.omiecid.encoding import decode_price
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_price(int(result[0, 2]), pc) - 18.01) < 0.01

    def test_comma_decimal_quantity_parsed(self, tmp_path):
        """Quantity '7,50' should be parsed as 7.5."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "18,01;7,50;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape[0] == 1
        # Column 3 = quantity (int32 tick-encoded post P2-F0)
        from omiecid_simulator.omiecid.encoding import decode_qty
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_qty(int(result[0, 3]), pc) - 7.5) < 0.01

    def test_single_order_returns_one_row(self, tmp_path):
        """Single valid order row produces shape (1, 13)."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape == (1, 13)

    def test_output_is_float32(self, tmp_path):
        """Output array dtype is int32 (Category B: P2-F0)."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.dtype == jnp.int32


# ============================================================================
# TestLoadOrdersFieldMapping — correctness of all 13 fields
# ============================================================================

class TestLoadOrdersFieldMapping:
    """Verify each of the 13 order fields is mapped correctly."""

    @pytest.fixture
    def multi_row_orders(self, tmp_path):
        """Fixture with multiple rows, different sides, agents, restrictions.

        After Phase 1.5-A all zones are loaded, so the Portuguese row is
        included. Timestamp-sorted row order:
          0: BETA  (Spain,    17:00:00) BUY  NON
          1: ALPHA (Spain,    17:00:05) SELL IOC
          2: GAMMA (Spain,    17:00:10) BUY  ICE  peak=5.0 ppd=0.5
          3: DELTA (Portugal, 17:00:15) SELL NON
          4: ALPHA (Spain,    17:00:20) SELL FOK
        """
        rows = (
            # Buy order, NON restriction
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};BETA;UNIT1;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
            # Sell order, IOC restriction
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};ALPHA;UNIT2;"
            "48,50;5,5;V;IOC;;;;01/06/2025 17:00:05;\n"
            # Buy order, ICE with peak_qty and ppd
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};GAMMA;UNIT3;"
            "55,00;20,0;C;ICE;;5,0;0,50;01/06/2025 17:00:10;\n"
            # Portuguese row — included after Phase 1.5-A (zone filter removed)
            f"01/06/2025;{CONTRACT_16_STR};{PORTUGAL_ZONA};DELTA;UNIT4;"
            "60,00;3,0;V;;;;;01/06/2025 17:00:15;\n"
            # FOK order
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};ALPHA;UNIT2;"
            "49,00;8,0;V;FOK;;;;01/06/2025 17:00:20;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, rows)
        config = _make_config(data_path, contract_id=16)
        return load_omie_orders(config)

    def test_all_zone_rows_included(self, multi_row_orders):
        """All 5 rows (4 Spain + 1 Portugal) must be loaded."""
        assert multi_row_orders.shape[0] == 5

    def test_contract_id_column(self, multi_row_orders):
        """Column 0 (contract_id) should be 16 for all rows."""
        assert jnp.all(multi_row_orders[:, 0] == 16)

    def test_buy_side_mapping(self, multi_row_orders):
        """'C' (Compra) maps to Side.BUY (0). First row is a buy."""
        assert int(multi_row_orders[0, 1]) == int(Side.BUY)

    def test_sell_side_mapping(self, multi_row_orders):
        """'V' (Venta) maps to Side.SELL (1). Second row is a sell."""
        assert int(multi_row_orders[1, 1]) == int(Side.SELL)

    def test_sequential_order_ids(self, multi_row_orders):
        """order_id (column 4) should be sequential starting at 1."""
        order_ids = multi_row_orders[:, 4]
        for i in range(multi_row_orders.shape[0]):
            assert int(order_ids[i]) == i + 1

    def test_agent_id_mapping_properties(self, multi_row_orders):
        """Agent-id mapping is deterministic, positive, and distinct for
        distinct agents.

        Row order by timestamp: BETA, ALPHA, GAMMA, DELTA, ALPHA.
        Properties asserted (not specific integer values, since story
        1.5-B will renumber agents):
          - all ids are positive
          - the two ALPHA rows (indexes 1 and 4) share the same id
          - BETA, ALPHA, GAMMA, DELTA each get a distinct id
        """
        agent_ids = multi_row_orders[:, 5]
        assert jnp.all(agent_ids > 0)
        # same agent -> same id
        assert int(agent_ids[1]) == int(agent_ids[4])  # ALPHA, ALPHA
        # distinct agents -> distinct ids
        distinct = {int(agent_ids[0]), int(agent_ids[1]),
                    int(agent_ids[2]), int(agent_ids[3])}
        assert len(distinct) == 4  # BETA, ALPHA, GAMMA, DELTA

    def test_timestamps_monotonically_increasing(self, multi_row_orders):
        """Orders are sorted by timestamp (column 6)."""
        timestamps = multi_row_orders[:, 6]
        for i in range(1, multi_row_orders.shape[0]):
            assert float(timestamps[i]) >= float(timestamps[i - 1])

    def test_exec_restriction_non(self, multi_row_orders):
        """First row has empty exec restriction -> NON (0)."""
        assert int(multi_row_orders[0, 7]) == int(ExecRestriction.NON)

    def test_exec_restriction_ioc(self, multi_row_orders):
        """Second row has IOC exec restriction."""
        assert int(multi_row_orders[1, 7]) == int(ExecRestriction.IOC)

    def test_exec_restriction_ice(self, multi_row_orders):
        """Third row has ICE exec restriction."""
        assert int(multi_row_orders[2, 7]) == int(ExecRestriction.ICE)

    def test_exec_restriction_portuguese_non(self, multi_row_orders):
        """Fourth row (Portugal) has empty exec restriction -> NON."""
        assert int(multi_row_orders[3, 7]) == int(ExecRestriction.NON)

    def test_exec_restriction_fok(self, multi_row_orders):
        """Fifth row has FOK exec restriction."""
        assert int(multi_row_orders[4, 7]) == int(ExecRestriction.FOK)

    def test_status_always_active(self, multi_row_orders):
        """Column 10 (status) should be ACTIVE (0) for all rows."""
        assert jnp.all(multi_row_orders[:, 10] == int(Status.ACTIVE))

    def test_peak_qty_for_iceberg(self, multi_row_orders):
        """ICE order (row 2) should have peak_qty = 5.0."""
        from omiecid_simulator.omiecid.encoding import decode_qty
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_qty(int(multi_row_orders[2, 11]), pc) - 5.0) < 0.01

    def test_ppd_for_iceberg(self, multi_row_orders):
        """ICE order (row 2) should have ppd = 0.5."""
        from omiecid_simulator.omiecid.encoding import decode_ppd
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_ppd(int(multi_row_orders[2, 12]), pc) - 0.5) < 0.01

    def test_peak_qty_zero_for_non_iceberg(self, multi_row_orders):
        """Non-iceberg orders should have peak_qty = 0.0."""
        assert float(multi_row_orders[0, 11]) == 0.0
        assert float(multi_row_orders[1, 11]) == 0.0

    def test_ppd_zero_for_non_iceberg(self, multi_row_orders):
        """Non-iceberg orders should have ppd = 0.0."""
        assert float(multi_row_orders[0, 12]) == 0.0
        assert float(multi_row_orders[1, 12]) == 0.0

    def test_expiry_time_always_zero(self, multi_row_orders):
        """Column 9 (expiry_time) should be 0 for all orders."""
        assert jnp.all(multi_row_orders[:, 9] == 0)


# ============================================================================
# TestLoadOrdersBoundaryContracts — contract 0 and 95
# ============================================================================

class TestLoadOrdersBoundaryContracts:
    """Loading orders filtered to contract 0 and contract 95."""

    def test_load_contract_0(self, tmp_path):
        """Contract 0 (00:00-00:15) can be loaded and filtered."""
        row = (
            f"02/06/2025;{CONTRACT_0_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "30,00;5,0;C;;;;;01/06/2025 20:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=0)
        result = load_omie_orders(config)
        assert result.shape[0] == 1
        assert int(result[0, 0]) == 0

    def test_load_contract_95(self, tmp_path):
        """Contract 95 (23:45-24:00) can be loaded and filtered."""
        row = (
            f"02/06/2025;{CONTRACT_95_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "30,00;5,0;V;;;;;01/06/2025 20:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=95)
        result = load_omie_orders(config)
        assert result.shape[0] == 1
        assert int(result[0, 0]) == 95


# ============================================================================
# TestLoadTradesEdgeCases
# ============================================================================

class TestLoadTradesEdgeCases:
    """Edge cases for trades file loading."""

    def test_empty_trades_file_no_data_rows(self, tmp_path):
        """Trades file with metadata + header but no data rows returns (0, 6)."""
        # A valid OMIE trades file always has at least metadata + header rows.
        # Writing just the header (no data rows) should return empty.
        data_path = _write_trades_file(tmp_path, DATE_NODASH, "")
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape == (0, 6)

    def test_trades_filtered_by_contract(self, tmp_path):
        """Only trades for the matching contract_id are included."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};BUYER1;UNIT1;{SPAIN_ZONA};"
            f"SELLER1;UNIT2;{SPAIN_ZONA};50,00;10,0;01/06/2025 17:30:00\n"
            f"01/06/2025;{CONTRACT_37_STR};BUYER1;UNIT1;{SPAIN_ZONA};"
            f"SELLER1;UNIT2;{SPAIN_ZONA};55,00;5,0;01/06/2025 17:31:00\n"
        )
        data_path = _write_trades_file(tmp_path, DATE_NODASH, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape[0] == 1
        # Contract ID in column 0
        assert int(result[0, 0]) == 16

    def test_trades_output_shape_is_6_columns(self, tmp_path):
        """Trades array has exactly 6 columns."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};BUYER1;UNIT1;{SPAIN_ZONA};"
            f"SELLER1;UNIT2;{SPAIN_ZONA};50,00;10,0;01/06/2025 17:30:00\n"
        )
        data_path = _write_trades_file(tmp_path, DATE_NODASH, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape[1] == 6

    def test_trades_output_is_float32(self, tmp_path):
        """Trades output dtype is float32."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};BUYER1;UNIT1;{SPAIN_ZONA};"
            f"SELLER1;UNIT2;{SPAIN_ZONA};50,00;10,0;01/06/2025 17:30:00\n"
        )
        data_path = _write_trades_file(tmp_path, DATE_NODASH, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.dtype == jnp.int32

    def test_trades_comma_decimal_price(self, tmp_path):
        """Trade price '20,48' should be parsed as 20.48."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};BUYER1;UNIT1;{SPAIN_ZONA};"
            f"SELLER1;UNIT2;{SPAIN_ZONA};20,48;1,00;01/06/2025 17:30:00\n"
        )
        data_path = _write_trades_file(tmp_path, DATE_NODASH, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        from omiecid_simulator.omiecid.encoding import decode_price
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_price(int(result[0, 1]), pc) - 20.48) < 0.01


# ============================================================================
# TestReplayResult — dataclass field consistency
# ============================================================================

class TestReplayResultFields:
    """ReplayResult dataclass has all expected fields and correct semantics."""

    def test_replay_result_has_all_fields(self):
        """ReplayResult must have all 7 documented fields."""
        r = ReplayResult(
            final_book=None,
            replayed_trades=jnp.zeros((0, 3)),
            historical_trades_matched=0,
            historical_trades_total=0,
            extra_trades=0,
            match_rate=1.0,
            extra_volume_pct=0.0,
        )
        assert hasattr(r, "final_book")
        assert hasattr(r, "replayed_trades")
        assert hasattr(r, "historical_trades_matched")
        assert hasattr(r, "historical_trades_total")
        assert hasattr(r, "extra_trades")
        assert hasattr(r, "match_rate")
        assert hasattr(r, "extra_volume_pct")

    def test_match_rate_is_ratio(self):
        """match_rate = historical_trades_matched / historical_trades_total."""
        r = ReplayResult(
            final_book=None,
            replayed_trades=jnp.zeros((0, 3)),
            historical_trades_matched=3,
            historical_trades_total=5,
            extra_trades=2,
            match_rate=0.6,
            extra_volume_pct=0.1,
        )
        expected = r.historical_trades_matched / r.historical_trades_total
        assert abs(r.match_rate - expected) < 1e-6

    def test_extra_trades_non_negative(self):
        """extra_trades should never be negative."""
        r = ReplayResult(
            final_book=None,
            replayed_trades=jnp.zeros((0, 3)),
            historical_trades_matched=0,
            historical_trades_total=0,
            extra_trades=0,
            match_rate=1.0,
            extra_volume_pct=0.0,
        )
        assert r.extra_trades >= 0

    def test_extra_volume_pct_non_negative(self):
        """extra_volume_pct should be >= 0."""
        r = ReplayResult(
            final_book=None,
            replayed_trades=jnp.zeros((0, 3)),
            historical_trades_matched=0,
            historical_trades_total=0,
            extra_trades=0,
            match_rate=1.0,
            extra_volume_pct=0.0,
        )
        assert r.extra_volume_pct >= 0.0

    def test_replayed_trades_is_jax_array(self):
        """replayed_trades should be a JAX array."""
        r = ReplayResult(
            final_book=None,
            replayed_trades=jnp.zeros((0, 3)),
            historical_trades_matched=0,
            historical_trades_total=0,
            extra_trades=0,
            match_rate=1.0,
            extra_volume_pct=0.0,
        )
        assert isinstance(r.replayed_trades, jnp.ndarray)


# ============================================================================
# TestReplayHeuristic — H1 and error on unknown
# ============================================================================

class TestReplayHeuristic:
    """Replay heuristic parameter validation."""

    def test_unknown_heuristic_raises_not_implemented(self):
        """Passing an unknown heuristic should raise NotImplementedError."""
        orders = jnp.zeros((0, 13), dtype=jnp.float32)
        trades = jnp.zeros((0, 6), dtype=jnp.float32)
        with pytest.raises(NotImplementedError):
            replay_orders_through_book(
                orders, None, trades, heuristic="ioc_all"
            )

    def test_unknown_heuristic_mentions_name_in_error(self):
        """Error message should mention the unsupported heuristic name."""
        orders = jnp.zeros((0, 13), dtype=jnp.float32)
        trades = jnp.zeros((0, 6), dtype=jnp.float32)
        with pytest.raises(NotImplementedError, match="ioc_all"):
            replay_orders_through_book(
                orders, None, trades, heuristic="ioc_all"
            )


# ============================================================================
# TestReplayWithBook — integration with ContractBook
# ============================================================================

class TestReplayWithBook:
    """Replay through ContractBook produces correct results.

    Uses the shared conftest fixtures (book, empty_state) which provide
    a ContractBook for contract 37.
    """

    def _make_order_row(self, side, price, qty, order_id, timestamp,
                        contract_id=37, agent_id=1,
                        exec_restriction=0, validity=0):
        """Create a single (1, 13) int32 order row for replay (P2-F0)."""
        from omiecid_simulator.omiecid.encoding import (
            encode_price, encode_qty,
        )
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        return jnp.array([[
            contract_id, side, encode_price(price, pc), encode_qty(qty, pc),
            order_id, agent_id,
            timestamp, exec_restriction, validity, 0,
            int(Status.ACTIVE), 0, 0,
        ]], dtype=jnp.int32)

    def test_no_orders_produces_empty_replay(self, book):
        """Replaying zero orders produces zero trades and match_rate=1.0."""
        orders = jnp.zeros((0, 13), dtype=jnp.float32)
        hist_trades = jnp.zeros((0, 6), dtype=jnp.float32)
        result = replay_orders_through_book(
            orders, book, hist_trades, heuristic="rest_until_gct"
        )
        assert result.replayed_trades.shape[0] == 0
        assert result.match_rate == 1.0
        assert result.extra_trades == 0

    def test_non_crossing_orders_produce_no_trades(self, book):
        """Buy below sell produces no trades."""
        buy = self._make_order_row(int(Side.BUY), 40.0, 5.0, 1, 1000)
        sell = self._make_order_row(int(Side.SELL), 60.0, 5.0, 2, 1001)
        orders = jnp.concatenate([buy, sell], axis=0)
        hist_trades = jnp.zeros((0, 6), dtype=jnp.float32)
        result = replay_orders_through_book(
            orders, book, hist_trades, heuristic="rest_until_gct"
        )
        assert result.replayed_trades.shape[0] == 0

    def test_crossing_orders_produce_trade(self, book):
        """Buy at 50 meets sell at 45 => trade occurs."""
        # Sell rests first, then buy crosses
        sell = self._make_order_row(int(Side.SELL), 45.0, 5.0, 1, 1000)
        buy = self._make_order_row(int(Side.BUY), 50.0, 5.0, 2, 1001)
        orders = jnp.concatenate([sell, buy], axis=0)
        hist_trades = jnp.zeros((0, 6), dtype=jnp.float32)
        result = replay_orders_through_book(
            orders, book, hist_trades, heuristic="rest_until_gct"
        )
        assert result.replayed_trades.shape[0] >= 1

    def test_replay_superset_property(self, book):
        """Replayed trades should be a superset of historical trades.

        Create orders that must cross, and a historical trade matching
        the expected outcome. match_rate should be 1.0.
        """
        sell = self._make_order_row(int(Side.SELL), 45.0, 5.0, 1, 1000)
        buy = self._make_order_row(int(Side.BUY), 50.0, 5.0, 2, 1001)
        orders = jnp.concatenate([sell, buy], axis=0)
        # Historical trade: price=45.0 (resting price), qty=5.0
        hist_trades = jnp.array([[37, 45.0, 5.0, 1, 2, 1001]], dtype=jnp.float32)
        result = replay_orders_through_book(
            orders, book, hist_trades, heuristic="rest_until_gct"
        )
        assert result.match_rate == 1.0
        assert result.historical_trades_matched == 1
        assert result.historical_trades_total == 1

    def test_extra_trades_counted(self, book):
        """If replay produces trades not in history, extra_trades > 0."""
        sell = self._make_order_row(int(Side.SELL), 45.0, 5.0, 1, 1000)
        buy = self._make_order_row(int(Side.BUY), 50.0, 5.0, 2, 1001)
        orders = jnp.concatenate([sell, buy], axis=0)
        # No historical trades => all replayed trades are "extra"
        hist_trades = jnp.zeros((0, 6), dtype=jnp.float32)
        result = replay_orders_through_book(
            orders, book, hist_trades, heuristic="rest_until_gct"
        )
        if result.replayed_trades.shape[0] > 0:
            assert result.extra_trades > 0

    def test_match_rate_between_zero_and_one(self, book):
        """match_rate should always be in [0.0, 1.0]."""
        sell = self._make_order_row(int(Side.SELL), 45.0, 5.0, 1, 1000)
        buy = self._make_order_row(int(Side.BUY), 50.0, 5.0, 2, 1001)
        orders = jnp.concatenate([sell, buy], axis=0)
        hist_trades = jnp.array([[37, 45.0, 5.0, 1, 2, 1001]], dtype=jnp.float32)
        result = replay_orders_through_book(
            orders, book, hist_trades, heuristic="rest_until_gct"
        )
        assert 0.0 <= result.match_rate <= 1.0

    def test_historical_trades_total_matches_input(self, book):
        """historical_trades_total should equal the number of input trades."""
        orders = jnp.zeros((0, 13), dtype=jnp.float32)
        hist_trades = jnp.array([
            [37, 45.0, 5.0, 1, 2, 1001],
            [37, 46.0, 3.0, 1, 2, 1002],
        ], dtype=jnp.float32)
        result = replay_orders_through_book(
            orders, book, hist_trades, heuristic="rest_until_gct"
        )
        assert result.historical_trades_total == 2

    def test_extra_volume_pct_consistent(self, book):
        """extra_volume_pct should be >= 0 and <= 1.0."""
        sell = self._make_order_row(int(Side.SELL), 45.0, 5.0, 1, 1000)
        buy = self._make_order_row(int(Side.BUY), 50.0, 5.0, 2, 1001)
        orders = jnp.concatenate([sell, buy], axis=0)
        hist_trades = jnp.array([[37, 45.0, 5.0, 1, 2, 1001]], dtype=jnp.float32)
        result = replay_orders_through_book(
            orders, book, hist_trades, heuristic="rest_until_gct"
        )
        assert 0.0 <= result.extra_volume_pct <= 1.0


# ============================================================================
# TestNegativePrices — OMIE allows negative prices
# ============================================================================

class TestNegativePrices:
    """OMIE allows prices in [-9999, +9999]. Negative prices must parse."""

    def test_negative_price_parsed(self, tmp_path):
        """Negative price '-5,50' should be parsed as -5.5."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "-5,50;10,0;V;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape[0] == 1
        from omiecid_simulator.omiecid.encoding import decode_price
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_price(int(result[0, 2]), pc) - (-5.5)) < 0.01

    def test_zero_price_parsed(self, tmp_path):
        """Zero price '0,00' should be parsed as 0.0."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "0,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape[0] == 1
        from omiecid_simulator.omiecid.encoding import decode_price
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_price(int(result[0, 2]), pc)) < 0.01


# ============================================================================
# TestNegativePPD — ICEwPI can have negative PPD
# ============================================================================

class TestNegativePPD:
    """ICEwPI orders can have negative PPD (per-slice price delta)."""

    def test_negative_ppd_parsed(self, tmp_path):
        """PPD '-1,00' should be parsed as -1.0."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "50,00;20,0;C;ICE;;5,0;-1,00;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, DATE_NODASH, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape[0] == 1
        from omiecid_simulator.omiecid.encoding import decode_ppd
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_ppd(int(result[0, 12]), pc) - (-1.0)) < 0.01
