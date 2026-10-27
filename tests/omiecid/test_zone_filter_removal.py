"""Locked behavioral contract tests for Story 1.5-A — Zone Filter Removal.

Source acceptance criteria (docs/omiecid/agile-plan.md, story 1.5-A):
  - OMIEDataConfig has no ``zona`` attribute.
  - ``load_omie_orders`` / ``load_omie_trades`` return data from all zones.
  - No ``zona`` filtering in production code.
  - Array shapes unchanged: orders ``(N, 13)``, trades ``(M, 6)``.
  - All existing tests pass after updates.

Domain references:
  - docs/omiecid/market-spec.md §13.1 (orders file columns, Zona is column 3)
  - docs/omiecid/market-spec.md §13.2 (trades file columns, Zona compra / Zona venta)
  - docs/omiecid/market-spec.md §13.4 (zone filtering removed; zona column is parsed
    but not used for filtering)
  - docs/omiecid/market-spec.md §13.5 (Unidad on orders and trades)

These tests are black-box: they import only the public interface
(``OMIEDataConfig``, ``load_omie_orders``, ``load_omie_trades``) and
assert against the acceptance criteria above.

No assumption is made about the integer values of ``agent_id`` — only
the mapping properties (deterministic, positive, distinct for distinct
inputs). Story 1.5-B will redefine agent-id numbering.
"""

import dataclasses
import inspect
from pathlib import Path

import jax.numpy as jnp
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    load_omie_orders,
    load_omie_trades,
)


# ============================================================================
# Synthetic OMIE CSV fixture helpers
# ============================================================================

SPAIN_ZONA = "10YES-REE------0"
PORTUGAL_ZONA = "10YPT-REN------W"

DATE_NODASH = "20250601"
DELIVERY_DATE = "2025-06-01"

# contract_id 16 = 04:00-04:15 (4*4 + 0 = 16)
CONTRACT_16_STR = "20250602 04:00-20250602 04:15"
# contract_id 37 = 09:15-09:30 (9*4 + 1 = 37)
CONTRACT_37_STR = "20250602 09:15-20250602 09:30"


ORDERS_HEADER = (
    "OMIE - Ofertas del Mercado Intradiario Continuo;\n"
    "Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;Tipo oferta;"
    "Condiciones a la ejecucion;Condiciones a la validez;"
    "Cantidad reducida;PPD;Momento de envio;\n"
)

TRADES_HEADER = (
    "OMIE - Casaciones del Mercado Intradiario Continuo;\n"
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;"
    "Agente venta;Unidad venta;Zona venta;"
    "Precio;Cantidad;Momento casacion;\n"
)


def _write_orders_file(tmp_path: Path, rows_text: str) -> str:
    path = tmp_path / f"orders_{DATE_NODASH}.1"
    path.write_text(ORDERS_HEADER + rows_text, encoding="latin-1")
    return str(tmp_path)


def _write_trades_file(tmp_path: Path, rows_text: str) -> str:
    path = tmp_path / f"trades_{DATE_NODASH}.1"
    path.write_text(TRADES_HEADER + rows_text, encoding="latin-1")
    return str(tmp_path)


# Fixture-only product date (OMIE-TAPE-CONTRACT-DATE-1, group F1).  Every
# fixture row carries a ``Contrato`` dated 2025-06-02, i.e. delivery product
# (2025-06-02, k).  Per market-spec v2.10 §13/§13.4 the written session file
# ``*_20250601.1`` is file D-1 of that product; the product also spans session
# file D (``*_20250602.1``).  ``DATE_NODASH`` / ``DELIVERY_DATE`` keep naming
# the session file the tests write.
_PRODUCT_DATE = "2025-06-02"
_PRODUCT_SESSION_FILES_NODASH = ("20250601", "20250602")  # (D-1, D)


def _complete_product_files(data_path: str) -> None:
    """Add a header-only file for every product session file not written.

    Reuses this module's headers (unaccented ``Momento casacion``).  Files a
    test wrote are never touched.
    """
    base = Path(data_path)
    for date_nodash in _PRODUCT_SESSION_FILES_NODASH:
        orders = base / f"orders_{date_nodash}.1"
        trades = base / f"trades_{date_nodash}.1"
        if not orders.exists():
            orders.write_text(ORDERS_HEADER, encoding="latin-1")
        if not trades.exists():
            trades.write_text(TRADES_HEADER, encoding="latin-1")


def _make_config(data_path: str, contract_id: int = 16) -> OMIEDataConfig:
    """Build an OMIEDataConfig without a ``zona`` keyword.

    After 1.5-A, ``OMIEDataConfig`` has no ``zona`` attribute, so this
    helper never passes one.  The config names delivery product
    (2025-06-02, contract_id); its session files are completed first.
    """
    _complete_product_files(data_path)
    return OMIEDataConfig(
        data_path=data_path,
        contract_id=contract_id,
        delivery_date=_PRODUCT_DATE,
    )


# ============================================================================
# Config surface — zona must be removed from OMIEDataConfig
# ============================================================================

class TestConfigSurface:
    """OMIEDataConfig must no longer expose a ``zona`` attribute."""

    def test_config_has_no_zona_field(self):
        """``zona`` must not appear in the dataclass fields."""
        field_names = {f.name for f in dataclasses.fields(OMIEDataConfig)}
        assert "zona" not in field_names

    def test_config_instance_has_no_zona_attribute(self):
        """An instance must not carry a ``zona`` attribute."""
        cfg = OMIEDataConfig(
            data_path="/tmp/omie_data",
            contract_id=0,
            delivery_date=DELIVERY_DATE,
        )
        assert not hasattr(cfg, "zona")

    def test_config_constructor_rejects_zona_kwarg(self):
        """Passing a ``zona=`` keyword must raise ``TypeError``.

        After 1.5-A, ``zona`` is not a constructor parameter. Attempting
        to pass it should be a TypeError (unexpected keyword argument).
        """
        with pytest.raises(TypeError):
            OMIEDataConfig(
                data_path="/tmp/omie_data",
                contract_id=0,
                delivery_date=DELIVERY_DATE,
                zona=SPAIN_ZONA,  # type: ignore[call-arg]
            )

    def test_config_required_fields_preserved(self):
        """The non-zona fields (data_path, contract_id, delivery_date,
        cache_path) remain on the dataclass."""
        field_names = {f.name for f in dataclasses.fields(OMIEDataConfig)}
        assert {"data_path", "contract_id", "delivery_date", "cache_path"} <= field_names


# ============================================================================
# Orders loader — no zona filtering; Spain and Portugal both included
# ============================================================================

class TestLoadOrdersNoZonaFilter:
    """``load_omie_orders`` must return rows from every zone in the input file."""

    def test_portugal_only_file_returns_rows(self, tmp_path):
        """A file containing only Portuguese rows must return those rows.

        Under Phase 1 this would have returned shape (0, 13). After 1.5-A
        Portuguese rows are no longer filtered out.
        """
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{PORTUGAL_ZONA};PTAG1;PTAG01;"
            "55,00;2,0;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape == (1, 13)

    def test_spain_only_file_returns_rows(self, tmp_path):
        """A file containing only Spanish rows must return those rows."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENT1;UNIT1;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape == (1, 13)

    def test_mixed_zones_returns_all_rows(self, tmp_path):
        """A mixed Spain+Portugal file for one contract returns every row."""
        rows = (
            # Spanish row
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENTS;UNITS;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
            # Portuguese row
            f"01/06/2025;{CONTRACT_16_STR};{PORTUGAL_ZONA};AGENTP;UNITP;"
            "55,00;2,0;V;;;;;01/06/2025 17:00:01;\n"
        )
        data_path = _write_orders_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape == (2, 13)

    def test_mixed_zones_produces_distinct_agent_ids(self, tmp_path):
        """Spain and Portugal agents in the same file must receive distinct,
        positive agent_id values.

        Per the architect's review (F4), this test asserts mapping
        *properties* — deterministic positivity and distinctness — not
        specific integer values, because story 1.5-B will renumber.
        """
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENTS;UNITS;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
            f"01/06/2025;{CONTRACT_16_STR};{PORTUGAL_ZONA};AGENTP;UNITP;"
            "55,00;2,0;V;;;;;01/06/2025 17:00:01;\n"
        )
        data_path = _write_orders_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        agent_ids = result[:, 5]
        # positivity
        assert jnp.all(agent_ids > 0)
        # distinctness (AGENTS ≠ AGENTP)
        assert int(agent_ids[0]) != int(agent_ids[1])

    def test_mixed_zones_other_contract_still_filtered(self, tmp_path):
        """Contract-id filtering is orthogonal to zone removal: rows for
        other contracts must still be excluded regardless of zone."""
        rows = (
            # contract 16, Spain — kept
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENTS;UNITS;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
            # contract 37, Spain — dropped (wrong contract)
            f"01/06/2025;{CONTRACT_37_STR};{SPAIN_ZONA};AGENTS;UNITS;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:01;\n"
            # contract 37, Portugal — dropped (wrong contract)
            f"01/06/2025;{CONTRACT_37_STR};{PORTUGAL_ZONA};AGENTP;UNITP;"
            "55,00;2,0;V;;;;;01/06/2025 17:00:02;\n"
        )
        data_path = _write_orders_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape == (1, 13)
        assert int(result[0, 0]) == 16

    def test_unknown_zona_value_not_filtered(self, tmp_path):
        """A zone value other than Spain or Portugal must also be loaded
        (no zone filtering at all)."""
        unknown_zona = "10YXX-UNKNOWN---0"
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{unknown_zona};AGENTX;UNITX;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape == (1, 13)


# ============================================================================
# Trades loader — no zona filtering; Spain-Spain, Spain-Portugal,
# Portugal-Portugal, and unknown-zone trades all included
# ============================================================================

class TestLoadTradesNoZonaFilter:
    """``load_omie_trades`` must return trades regardless of buy/sell zones.

    Behavioral inversion from Phase 1:
      - Phase 1: Portugal–Portugal trades were dropped (union filter against
        Spain).
      - After 1.5-A: every trade in the file is returned, subject only to
        contract_id filtering.
    """

    def test_spain_spain_trade_included(self, tmp_path):
        """A trade with both legs in Spain must be returned."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};BUYER1;UNIT1;{SPAIN_ZONA};"
            f"SELLER1;UNIT2;{SPAIN_ZONA};50,00;10,0;01/06/2025 17:30:00\n"
        )
        data_path = _write_trades_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape == (1, 6)

    def test_portugal_portugal_trade_included(self, tmp_path):
        """A trade with both legs in Portugal must be returned.

        This is the explicit behavioral inversion from Phase 1.
        """
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};BUYER1;UNIT1;{PORTUGAL_ZONA};"
            f"SELLER1;UNIT2;{PORTUGAL_ZONA};50,00;10,0;01/06/2025 17:30:00\n"
        )
        data_path = _write_trades_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape == (1, 6)

    def test_cross_zone_spain_to_portugal_included(self, tmp_path):
        """A cross-zone trade (Spain buy, Portugal sell) must be returned."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};BUYER1;UNIT1;{SPAIN_ZONA};"
            f"SELLER1;UNIT2;{PORTUGAL_ZONA};50,00;10,0;01/06/2025 17:30:00\n"
        )
        data_path = _write_trades_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape == (1, 6)

    def test_cross_zone_portugal_to_spain_included(self, tmp_path):
        """A cross-zone trade (Portugal buy, Spain sell) must be returned."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};BUYER1;UNIT1;{PORTUGAL_ZONA};"
            f"SELLER1;UNIT2;{SPAIN_ZONA};50,00;10,0;01/06/2025 17:30:00\n"
        )
        data_path = _write_trades_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape == (1, 6)

    def test_all_zone_combinations_included(self, tmp_path):
        """All four zone combinations in one file must all be returned."""
        rows = (
            # ES-ES
            f"01/06/2025;{CONTRACT_16_STR};B1;U1;{SPAIN_ZONA};"
            f"S1;U2;{SPAIN_ZONA};50,00;1,0;01/06/2025 17:30:00\n"
            # ES-PT
            f"01/06/2025;{CONTRACT_16_STR};B2;U3;{SPAIN_ZONA};"
            f"S2;U4;{PORTUGAL_ZONA};51,00;1,0;01/06/2025 17:30:01\n"
            # PT-ES
            f"01/06/2025;{CONTRACT_16_STR};B3;U5;{PORTUGAL_ZONA};"
            f"S3;U6;{SPAIN_ZONA};52,00;1,0;01/06/2025 17:30:02\n"
            # PT-PT
            f"01/06/2025;{CONTRACT_16_STR};B4;U7;{PORTUGAL_ZONA};"
            f"S4;U8;{PORTUGAL_ZONA};53,00;1,0;01/06/2025 17:30:03\n"
        )
        data_path = _write_trades_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape == (4, 6)

    def test_other_contract_still_filtered(self, tmp_path):
        """Trades for a different contract must still be excluded."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};B1;U1;{PORTUGAL_ZONA};"
            f"S1;U2;{PORTUGAL_ZONA};50,00;1,0;01/06/2025 17:30:00\n"
            f"01/06/2025;{CONTRACT_37_STR};B2;U3;{PORTUGAL_ZONA};"
            f"S2;U4;{PORTUGAL_ZONA};51,00;1,0;01/06/2025 17:30:01\n"
        )
        data_path = _write_trades_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape == (1, 6)
        assert int(result[0, 0]) == 16


# ============================================================================
# Array shape invariance — orders (N, 13), trades (M, 6)
# ============================================================================

class TestShapeInvariance:
    """Published array contract must be preserved after zone removal:
    orders are (N, 13), trades are (M, 6). No extra zona column is appended.
    """

    def test_orders_shape_13_columns_mixed_zones(self, tmp_path):
        """With mixed-zone input, the orders array must still have 13 columns."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};{SPAIN_ZONA};AGENTS;UNITS;"
            "50,00;10,0;C;;;;;01/06/2025 17:00:00;\n"
            f"01/06/2025;{CONTRACT_16_STR};{PORTUGAL_ZONA};AGENTP;UNITP;"
            "55,00;2,0;V;;;;;01/06/2025 17:00:01;\n"
        )
        data_path = _write_orders_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.ndim == 2
        assert result.shape[1] == 13

    def test_orders_shape_13_columns_single_portuguese_row(self, tmp_path):
        """A single Portuguese order row must yield shape (1, 13)."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{PORTUGAL_ZONA};PTAG1;PTAG01;"
            "55,00;2,0;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape == (1, 13)

    def test_trades_shape_6_columns_mixed_zones(self, tmp_path):
        """With mixed-zone input, the trades array must still have 6 columns."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};B1;U1;{SPAIN_ZONA};"
            f"S1;U2;{PORTUGAL_ZONA};50,00;1,0;01/06/2025 17:30:00\n"
            f"01/06/2025;{CONTRACT_16_STR};B2;U3;{PORTUGAL_ZONA};"
            f"S2;U4;{PORTUGAL_ZONA};51,00;1,0;01/06/2025 17:30:01\n"
        )
        data_path = _write_trades_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.ndim == 2
        assert result.shape[1] == 6

    def test_trades_shape_6_columns_portugal_only(self, tmp_path):
        """A single Portugal-Portugal trade must yield shape (1, 6)."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};B1;U1;{PORTUGAL_ZONA};"
            f"S1;U2;{PORTUGAL_ZONA};50,00;1,0;01/06/2025 17:30:00\n"
        )
        data_path = _write_trades_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape == (1, 6)

    def test_orders_dtype_float32(self, tmp_path):
        """Orders array dtype must remain float32."""
        row = (
            f"01/06/2025;{CONTRACT_16_STR};{PORTUGAL_ZONA};PTAG1;PTAG01;"
            "55,00;2,0;C;;;;;01/06/2025 17:00:00;\n"
        )
        data_path = _write_orders_file(tmp_path, row)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.dtype == jnp.int32

    def test_trades_dtype_float32(self, tmp_path):
        """Trades array dtype must remain float32."""
        rows = (
            f"01/06/2025;{CONTRACT_16_STR};B1;U1;{PORTUGAL_ZONA};"
            f"S1;U2;{PORTUGAL_ZONA};50,00;1,0;01/06/2025 17:30:00\n"
        )
        data_path = _write_trades_file(tmp_path, rows)
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.dtype == jnp.int32


# ============================================================================
# Production code must not reference config.zona
# ============================================================================

class TestNoZonaInProductionCode:
    """The production code under ``omiecid_simulator/jaxomiecid/`` must not
    reference ``config.zona`` or ``.zona`` on any config object.

    This test formalizes the acceptance line "No zona filtering in
    production code."  It scans the data-loader and data-config source
    text directly (no dynamic import introspection), so it fails fast
    and independent of test fixtures.
    """

    def _production_source(self) -> str:
        """Concatenated source of the two Phase-1.5 data modules."""
        from omiecid_simulator.jaxomiecid import omie_data_config, omie_data_loader
        return "\n".join(
            inspect.getsource(mod) for mod in (omie_data_config, omie_data_loader)
        )

    def test_no_config_dot_zona_reference(self):
        """The string ``config.zona`` must not appear in production source."""
        source = self._production_source()
        assert "config.zona" not in source

    def test_no_zona_attribute_reference(self):
        """No ``.zona`` attribute access on any object must remain."""
        source = self._production_source()
        # The string ".zona" would appear as either ``config.zona`` or
        # ``self.zona``; neither is permissible in post-1.5-A code.
        assert ".zona" not in source

    def test_no_zona_dataclass_field_declaration(self):
        """The data-config module must not declare a ``zona`` field.

        This catches lines like ``zona: str = "10YES-REE------0"``.
        """
        from omiecid_simulator.jaxomiecid import omie_data_config
        source = inspect.getsource(omie_data_config)
        # Either ``zona:`` (typed field) or ``zona =`` (untyped default)
        # would be a re-introduction of the removed attribute.
        assert "zona:" not in source
        assert "zona =" not in source


# ============================================================================
# Empty / negative cases
# ============================================================================

class TestEmptyFiles:
    """Empty input files must still return the canonical empty shapes,
    independent of zone handling."""

    def test_orders_empty_file_returns_zero_by_13(self, tmp_path):
        """An orders file with only headers returns shape (0, 13)."""
        data_path = _write_orders_file(tmp_path, "")
        config = _make_config(data_path, contract_id=16)
        result = load_omie_orders(config)
        assert result.shape == (0, 13)

    def test_trades_empty_file_returns_zero_by_6(self, tmp_path):
        """A trades file with only headers returns shape (0, 6)."""
        data_path = _write_trades_file(tmp_path, "")
        config = _make_config(data_path, contract_id=16)
        result = load_omie_trades(config)
        assert result.shape == (0, 6)
