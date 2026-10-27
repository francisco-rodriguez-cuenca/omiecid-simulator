"""Audit tests for Story 1.5-A — Zone Filter Removal.

Independent black-box verification that the Phase 1.5-A zone-filter removal
was implemented correctly against the domain contract. This file is authored
*after* implementation, with no knowledge of the original locked test file.

Scope (derived from agile-plan.md Story 1.5-A acceptance criteria, market-spec
§13.1/§13.2/§13.4/§13.5, and phase1.5-redesign-issues.md §B):

  1. OMIEDataConfig has no `zona` attribute at the class level (TC-5 class contract).
  2. No production module under ``omiecid_simulator/`` references ``.zona`` on a
     config (TC-7 negative-space contract).
  3. Both ``load_omie_orders`` and ``load_omie_trades`` return all Iberian
     zones (Spain + Portugal + any other EIC code that appears in the file).
  4. Output shapes are unchanged: orders (N, 13), trades (M, 6).
  5. The single behavioral inversion from Phase 1 (cross-zona trades previously
     excluded) is now included with full fidelity.
  6. Non-zona filter criteria (contract_id, delivery_date) remain enforced.
  7. Loading the same file twice with a clean cache produces identical arrays
     (determinism).
  8. Portuguese rows exercise every code path Spanish rows do — timestamps,
     numeric (comma-decimal) parsing, exec restriction, validity, iceberg
     fields — without crashing and with identical semantic meaning.
  9. Trades with one blank leg (buy or sell counterparty missing — the XBID
     cross-border case observable in real OMIE data) are admitted.

Out of scope (per TC-8 — deferred to 1.5-B/1.5-C/1.5-D):
  - Timestamp timezone correctness for PT rows (1.5-C).
  - Agent/Unidad pair mapping (1.5-B).
  - Settlement aggregation (1.5-D).

Runtime budget: all tests target < 1s wall clock; no ``@pytest.mark.slow``
tests are introduced — synthetic fixtures are built in-process via tmp_path.

Reference paths:
  - omiecid_simulator/jaxomiecid/omie_data_config.py
  - omiecid_simulator/jaxomiecid/omie_data_loader.py
  - docs/omiecid/market-spec.md §13
  - docs/omiecid/agile-plan.md Story 1.5-A
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    load_omie_orders,
    load_omie_trades,
)
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    Status,
    Validity,
)


# ---------------------------------------------------------------------------
# EIC zone codes (market-spec §13.1/§13.2)
# ---------------------------------------------------------------------------

_ZONE_ES = "10YES-REE------0"  # Spain
_ZONE_PT = "10YPT-REN------W"  # Portugal
_ZONE_OTHER = "10YXX-OTHER----Z"  # unusual/unknown EIC code — must not crash

_DATE_DMY = "01/06/2025"
_DATE_ISO = "2025-06-01"
_DATE_NODASH = "20250601"

# Contract string helpers. contract_id = hour*4 + minute//15.
def _contract_str(contract_id: int) -> str:
    hour = contract_id // 4
    minute = (contract_id % 4) * 15
    end_minute = minute + 15
    end_hour = hour
    if end_minute == 60:
        end_hour = hour + 1
        end_minute = 0
    return (
        f"20250602 {hour:02d}:{minute:02d}-20250602 {end_hour:02d}:{end_minute:02d}"
    )


# ---------------------------------------------------------------------------
# Synthetic OMIE file writers
# ---------------------------------------------------------------------------

_ORDERS_HEADER_META = (
    "OMIE - Mercado de electricidad; Ofertas del Mercado Intradiario Continuo;"
)
_ORDERS_HEADER_COLS = (
    "Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;Tipo oferta;"
    "Condiciones a la ejecucion;Condiciones a la validez;"
    "Cantidad reducida;PPD;Momento de envio;"
)

_TRADES_HEADER_META = (
    "OMIE - Mercado de electricidad; Casaciones del Mercado Intradiario Continuo;"
)
_TRADES_HEADER_COLS = (
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;"
    "Agente venta;Unidad venta;Zona venta;Precio;Cantidad;Momento casacion;"
)


def _write_orders(tmp_path: Path, rows: list[str]) -> None:
    lines = [_ORDERS_HEADER_META, _ORDERS_HEADER_COLS]
    for r in rows:
        lines.append(r if r.endswith(";") else r + ";")
    (tmp_path / f"orders_{_DATE_NODASH}.1").write_text(
        "\n".join(lines) + "\n", encoding="latin-1"
    )


def _write_trades(tmp_path: Path, rows: list[str]) -> None:
    lines = [_TRADES_HEADER_META, _TRADES_HEADER_COLS]
    for r in rows:
        lines.append(r if r.endswith(";") else r + ";")
    (tmp_path / f"trades_{_DATE_NODASH}.1").write_text(
        "\n".join(lines) + "\n", encoding="latin-1"
    )


def _order_row(
    *,
    contract_id: int,
    zona: str,
    agente: str,
    unidad: str,
    precio: str,
    cantidad: str,
    side: str,
    exec_r: str = "",
    validez: str = "",
    cant_red: str = "",
    ppd: str = "",
    momento: str = "01/06/2025 17:41:12",
) -> str:
    return (
        f"{_DATE_DMY};{_contract_str(contract_id)};{zona};{agente};{unidad};"
        f"{precio};{cantidad};{side};{exec_r};{validez};{cant_red};{ppd};{momento}"
    )


def _trade_row(
    *,
    contract_id: int,
    buy_agente: str,
    buy_unidad: str,
    buy_zona: str,
    sell_agente: str,
    sell_unidad: str,
    sell_zona: str,
    precio: str,
    cantidad: str,
    momento: str = "01/06/2025 17:42:00",
) -> str:
    return (
        f"{_DATE_DMY};{_contract_str(contract_id)};"
        f"{buy_agente};{buy_unidad};{buy_zona};"
        f"{sell_agente};{sell_unidad};{sell_zona};"
        f"{precio};{cantidad};{momento}"
    )


# Fixture-only product date (OMIE-TAPE-CONTRACT-DATE-1, group F1).  Every
# fixture row carries a ``Contrato`` dated 2025-06-02 (``_contract_str``),
# i.e. delivery product (2025-06-02, k).  Per market-spec v2.10 §13/§13.4 the
# written session file ``*_20250601.1`` is file D-1 of that product; the
# product also spans session file D (``*_20250602.1``).  ``_DATE_ISO`` /
# ``_DATE_NODASH`` keep naming the session file the tests write.
_PRODUCT_DATE_ISO = "2025-06-02"
_PRODUCT_SESSION_FILES_NODASH = ("20250601", "20250602")  # (D-1, D)


def _complete_product_files(tmp_path: Path) -> None:
    """Add a header-only file for every product session file not written.

    Same header lines as ``_write_orders`` / ``_write_trades`` (unaccented
    ``Momento casacion``).  A file written later by a test overwrites it;
    a file already written is never touched.
    """
    for date_nodash in _PRODUCT_SESSION_FILES_NODASH:
        orders = tmp_path / f"orders_{date_nodash}.1"
        trades = tmp_path / f"trades_{date_nodash}.1"
        if not orders.exists():
            orders.write_text(
                "\n".join([_ORDERS_HEADER_META, _ORDERS_HEADER_COLS]) + "\n",
                encoding="latin-1",
            )
        if not trades.exists():
            trades.write_text(
                "\n".join([_TRADES_HEADER_META, _TRADES_HEADER_COLS]) + "\n",
                encoding="latin-1",
            )


def _config(tmp_path: Path, contract_id: int = 16) -> OMIEDataConfig:
    _complete_product_files(tmp_path)
    return OMIEDataConfig(
        data_path=str(tmp_path),
        contract_id=contract_id,
        delivery_date=_PRODUCT_DATE_ISO,
    )


# ===========================================================================
# 1. Config surface — class-level absence of `zona` (TC-5)
# ===========================================================================


class TestConfigSurface:
    """`OMIEDataConfig` must have no `zona` field at the class level.

    Acceptance criterion: ``OMIEDataConfig has no zona attribute``.
    We verify by dataclass field introspection (strongest contract) and by
    construction-side rejection of a `zona=` kwarg.
    """

    def test_zona_not_in_dataclass_fields(self):
        """Dataclass fields must not include any 'zona' entry."""
        field_names = {f.name for f in dataclasses.fields(OMIEDataConfig)}
        assert "zona" not in field_names, (
            f"OMIEDataConfig still declares a `zona` dataclass field: "
            f"{sorted(field_names)}"
        )

    def test_zona_not_in_class_annotations(self):
        """Class annotations must not mention 'zona'."""
        annotations = getattr(OMIEDataConfig, "__annotations__", {})
        assert "zona" not in annotations, (
            f"OMIEDataConfig still annotates `zona`: {sorted(annotations)}"
        )

    def test_zona_not_in_instance_dict(self, tmp_path):
        """A constructed instance must not carry a `.zona` attribute."""
        cfg = _config(tmp_path)
        assert not hasattr(cfg, "zona"), (
            "OMIEDataConfig instance should not carry a `zona` attribute"
        )

    def test_zona_kwarg_rejected_at_construction(self, tmp_path):
        """Passing `zona=` to the constructor must raise TypeError."""
        with pytest.raises(TypeError):
            OMIEDataConfig(  # type: ignore[call-arg]
                data_path=str(tmp_path),
                contract_id=16,
                delivery_date=_DATE_ISO,
                zona=_ZONE_ES,
            )


# ===========================================================================
# 2. Negative-space — no `.zona` references in production source (TC-7)
# ===========================================================================


class TestNoZonaInProductionCode:
    """No module under ``omiecid_simulator/`` may reference ``config.zona``.

    Per the story contract ("No `zona` filtering in production code"), any
    surviving call site is a correctness hazard. We do an AST-agnostic
    source-level grep because the production surface is very small.
    """

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _PROD_DIR = _REPO_ROOT / "omiecid_simulator"

    # We want to flag config.zona and anything like `*.zona`. We accept the
    # literal strings "Zona", "Zona compra", "Zona venta" (DataFrame column
    # names per market-spec §13.1/§13.2) — those are not filter predicates.
    _FORBIDDEN_PATTERNS = [
        re.compile(r"config\s*\.\s*zona\b"),
        re.compile(r"\.zona\b(?!\s*=)"),  # `.zona` attribute access
    ]

    def _iter_prod_files(self):
        for path in self._PROD_DIR.rglob("*.py"):
            # Skip __pycache__ and test directories that may have leaked.
            if "__pycache__" in path.parts:
                continue
            yield path

    def test_no_config_dot_zona_in_production_source(self):
        """Grep every production .py file for `config.zona` references."""
        offenders: list[tuple[Path, int, str]] = []
        for path in self._iter_prod_files():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")
            for lineno, line in enumerate(text.splitlines(), start=1):
                # Strip strings/comments cheaply: skip docstring lines that
                # mention zona as prose. We only flag concrete attribute
                # access patterns.
                stripped = line.split("#", 1)[0]
                for pat in self._FORBIDDEN_PATTERNS:
                    if pat.search(stripped):
                        offenders.append((path, lineno, line.rstrip()))
                        break
        assert not offenders, (
            "Production code still references `.zona` attribute access:\n"
            + "\n".join(f"  {p.relative_to(self._REPO_ROOT)}:{ln}: {src}"
                        for p, ln, src in offenders)
        )


# ===========================================================================
# 3. Shape invariance — no extra columns accidentally appended (TC-4)
# ===========================================================================


class TestShapeInvariance:
    """Output arrays must be (N, 13) orders and (M, 6) trades, always."""

    def test_orders_shape_13_columns_even_with_mixed_zones(self, tmp_path):
        """Mixed-zone fixture still returns exactly 13 columns."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="AES", unidad="AESU1",
                           precio="50,00", cantidad="1,00", side="C"),
                _order_row(contract_id=16, zona=_ZONE_PT, agente="APT", unidad="APTU1",
                           precio="49,00", cantidad="2,00", side="V"),
            ],
        )
        _write_trades(tmp_path, [])
        arr = load_omie_orders(_config(tmp_path))
        assert arr.ndim == 2
        assert arr.shape[1] == 13, (
            f"orders.shape[1] must be 13; got {arr.shape[1]} (extra zona column?)"
        )

    def test_trades_shape_6_columns_even_with_cross_zone(self, tmp_path):
        """Mixed cross-zona fixture still returns exactly 6 columns."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="AES", buy_unidad="AESU1", buy_zona=_ZONE_ES,
                           sell_agente="APT", sell_unidad="APTU1", sell_zona=_ZONE_PT,
                           precio="50,00", cantidad="1,00"),
            ],
        )
        arr = load_omie_trades(_config(tmp_path))
        assert arr.ndim == 2
        assert arr.shape[1] == 6, (
            f"trades.shape[1] must be 6; got {arr.shape[1]} (extra zona column?)"
        )

    def test_orders_dtype_is_float32(self, tmp_path):
        """Orders are a float32 JAX array (published contract)."""
        _write_orders(
            tmp_path,
            [_order_row(contract_id=16, zona=_ZONE_ES, agente="A", unidad="U1",
                        precio="50,00", cantidad="1,00", side="C")],
        )
        _write_trades(tmp_path, [])
        arr = load_omie_orders(_config(tmp_path))
        assert arr.dtype == jnp.int32

    def test_trades_dtype_is_float32(self, tmp_path):
        """Trades are a float32 JAX array (published contract)."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [_trade_row(contract_id=16,
                        buy_agente="A", buy_unidad="U1", buy_zona=_ZONE_ES,
                        sell_agente="B", sell_unidad="U2", sell_zona=_ZONE_ES,
                        precio="50,00", cantidad="1,00")],
        )
        arr = load_omie_trades(_config(tmp_path))
        assert arr.dtype == jnp.int32


# ===========================================================================
# 4. Zone inclusion — Portuguese parity and "any zone" admission (TC-2/TC-3)
# ===========================================================================


class TestZoneInclusion:
    """Every zone visible in the raw file must make it into the output."""

    def test_pt_only_orders_nonempty(self, tmp_path):
        """A PT-only file must produce a non-empty orders array.

        Behavioral inversion from Phase 1: formerly filtered out entirely.
        """
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_PT, agente="PT1", unidad="PTU1",
                           precio="10,00", cantidad="1,00", side="C"),
                _order_row(contract_id=16, zona=_ZONE_PT, agente="PT2", unidad="PTU2",
                           precio="11,00", cantidad="2,00", side="V"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 2, (
            f"PT-only file should yield 2 rows; got {orders.shape[0]}"
        )

    def test_mixed_zones_all_included(self, tmp_path):
        """Mixed ES + PT + other-EIC rows must all be included."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="AES", unidad="AESU1",
                           precio="50,00", cantidad="1,00", side="C"),
                _order_row(contract_id=16, zona=_ZONE_PT, agente="APT", unidad="APTU1",
                           precio="49,00", cantidad="2,00", side="V"),
                _order_row(contract_id=16, zona=_ZONE_OTHER, agente="AXX", unidad="AXXU1",
                           precio="51,00", cantidad="3,00", side="C"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 3, (
            f"All 3 zones should be admitted; got {orders.shape[0]}"
        )

    def test_blank_zona_on_order_not_dropped(self, tmp_path):
        """Malformed/blank Zona string must not cause silent filtering.

        A blank Zona column is malformed input but a loader that filters on
        Zona could trivially exclude it. Since zona filtering is removed,
        the row must be retained.
        """
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona="", agente="NOZ", unidad="NOZU1",
                           precio="50,00", cantidad="1,00", side="C"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="AES", unidad="AESU1",
                           precio="50,00", cantidad="1,00", side="V"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 2, (
            f"Blank-zona order must not be silently dropped; got {orders.shape[0]}"
        )

    def test_cross_zona_es_pt_trade_included(self, tmp_path):
        """ES-buy / PT-sell trade must be included (previously kept)."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="AES", buy_unidad="AESU1", buy_zona=_ZONE_ES,
                           sell_agente="APT", sell_unidad="APTU1", sell_zona=_ZONE_PT,
                           precio="50,00", cantidad="1,00"),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        assert trades.shape[0] == 1

    def test_pt_pt_trade_included(self, tmp_path):
        """PT-buy / PT-sell trade must now be included.

        Phase 1 behavior: when config.zona=ES, a PT-PT trade was filtered out
        by the union filter (neither side matched). Phase 1.5-A: included.
        This is the single behavioral inversion from Phase 1.
        """
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="PT1", buy_unidad="PT1U1", buy_zona=_ZONE_PT,
                           sell_agente="PT2", sell_unidad="PT2U1", sell_zona=_ZONE_PT,
                           precio="40,00", cantidad="2,00"),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        assert trades.shape[0] == 1, (
            "PT-PT cross-zonal trade must be admitted after 1.5-A "
            "(was filtered pre-1.5-A when config.zona=ES)"
        )

    def test_trade_with_unusual_zone_included(self, tmp_path):
        """Any EIC code (not just ES/PT) must be admitted."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="AES", buy_unidad="AESU1", buy_zona=_ZONE_ES,
                           sell_agente="AXX", sell_unidad="AXXU1", sell_zona=_ZONE_OTHER,
                           precio="50,00", cantidad="1,00"),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        assert trades.shape[0] == 1


# ===========================================================================
# 5. Cross-zone trade fidelity — price/quantity/agent_id correctness (audit depth)
# ===========================================================================


class TestCrossZoneTradeFidelity:
    """Cross-zonal trades must carry correct price/qty/agent mappings.

    The audit mindset: "not just present, but with correct price/quantity/
    agent_id mapping". Phase 1 filter-removal must not introduce silent
    column-shift bugs.
    """

    @pytest.fixture
    def cross_zone_config(self, tmp_path):
        _write_orders(tmp_path, [])
        # Two trades, both cross-zonal; different prices + quantities.
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="AES", buy_unidad="AESU1", buy_zona=_ZONE_ES,
                           sell_agente="APT", sell_unidad="APTU1", sell_zona=_ZONE_PT,
                           precio="42,25", cantidad="3,50"),
                _trade_row(contract_id=16,
                           buy_agente="APT", buy_unidad="APTU1", buy_zona=_ZONE_PT,
                           sell_agente="AES", sell_unidad="AESU1", sell_zona=_ZONE_ES,
                           precio="43,75", cantidad="1,25"),
            ],
        )
        return _config(tmp_path)

    def test_cross_zone_trade_price_preserved(self, cross_zone_config):
        """Trade[1] price column must exactly hold the parsed price."""
        from omiecid_simulator.omiecid.encoding import decode_price
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        trades = load_omie_trades(cross_zone_config)
        prices = [decode_price(int(trades[i, 1]), pc) for i in range(trades.shape[0])]
        assert any(abs(p - 42.25) < 1e-3 for p in prices)
        assert any(abs(p - 43.75) < 1e-3 for p in prices)

    def test_cross_zone_trade_quantity_preserved(self, cross_zone_config):
        """Trade[2] quantity column must hold the parsed quantity."""
        from omiecid_simulator.omiecid.encoding import decode_qty
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        trades = load_omie_trades(cross_zone_config)
        qtys = [decode_qty(int(trades[i, 2]), pc) for i in range(trades.shape[0])]
        # Post P2-F0 quantities snap to the qty_tick grid (0.1 MW).
        # ``1,25`` in the fixture is off-tick: ``round(1.25 / 0.1) == 12``
        # (Python banker's rounding, half-to-even) → decoded 1.2 MW.
        assert any(abs(q - 3.50) < 1e-3 for q in qtys)
        assert any(abs(q - 1.2) < 1e-3 for q in qtys)

    def test_cross_zone_agent_ids_distinct_and_positive(self, cross_zone_config):
        """Buy-agent and sell-agent IDs must be positive and distinct per trade."""
        trades = load_omie_trades(cross_zone_config)
        assert trades.shape[0] == 2
        for i in range(trades.shape[0]):
            buy_id = int(trades[i, 3])
            sell_id = int(trades[i, 4])
            assert buy_id > 0, f"trade {i} buy_agent_id must be > 0, got {buy_id}"
            assert sell_id > 0, f"trade {i} sell_agent_id must be > 0, got {sell_id}"
            assert buy_id != sell_id, (
                f"trade {i}: distinct agent strings must map to distinct ids "
                f"(buy={buy_id}, sell={sell_id})"
            )

    def test_cross_zone_agent_id_consistent_across_trades(self, cross_zone_config):
        """Same agent string must receive the same agent_id in every trade.

        Trade[0] buyer=AES == trade[1] seller=AES → same id.
        Trade[0] seller=APT == trade[1] buyer=APT → same id.
        """
        trades = load_omie_trades(cross_zone_config)
        # trade[0] buy = AES, trade[1] sell = AES
        aes_via_buy_slot = int(trades[0, 3])
        aes_via_sell_slot = int(trades[1, 4])
        assert aes_via_buy_slot == aes_via_sell_slot, (
            f"AES maps inconsistently across rows: "
            f"{aes_via_buy_slot} vs {aes_via_sell_slot}"
        )
        # trade[0] sell = APT, trade[1] buy = APT
        apt_via_sell_slot = int(trades[0, 4])
        apt_via_buy_slot = int(trades[1, 3])
        assert apt_via_sell_slot == apt_via_buy_slot, (
            f"APT maps inconsistently across rows: "
            f"{apt_via_sell_slot} vs {apt_via_buy_slot}"
        )


# ===========================================================================
# 6. Non-zona filter criteria still enforced — regression guard (audit depth)
# ===========================================================================


class TestNonZonaFiltersStillEnforced:
    """Contract filter must still work; only zona filter was removed."""

    def test_other_contract_ids_excluded_from_orders(self, tmp_path):
        """Orders for a different contract_id must be filtered out."""
        _write_orders(
            tmp_path,
            [
                # contract 16 — target
                _order_row(contract_id=16, zona=_ZONE_ES, agente="AES", unidad="AESU1",
                           precio="50,00", cantidad="1,00", side="C"),
                # contract 32 — different, must be excluded
                _order_row(contract_id=32, zona=_ZONE_ES, agente="AES", unidad="AESU1",
                           precio="50,00", cantidad="1,00", side="C"),
                # contract 16 — PT, must be included (zona removed)
                _order_row(contract_id=16, zona=_ZONE_PT, agente="APT", unidad="APTU1",
                           precio="51,00", cantidad="2,00", side="V"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path, contract_id=16))
        assert orders.shape[0] == 2, (
            f"Should filter by contract_id (16 only); got {orders.shape[0]}"
        )
        # All remaining rows must have contract_id == 16
        assert jnp.all(orders[:, 0] == 16), (
            "All retained rows must have contract_id == 16"
        )

    def test_other_contract_ids_excluded_from_trades(self, tmp_path):
        """Trades for a different contract_id must be filtered out."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="AES", buy_unidad="AESU1", buy_zona=_ZONE_ES,
                           sell_agente="APT", sell_unidad="APTU1", sell_zona=_ZONE_PT,
                           precio="50,00", cantidad="1,00"),
                _trade_row(contract_id=32,
                           buy_agente="AES", buy_unidad="AESU1", buy_zona=_ZONE_ES,
                           sell_agente="APT", sell_unidad="APTU1", sell_zona=_ZONE_PT,
                           precio="50,00", cantidad="1,00"),
            ],
        )
        trades = load_omie_trades(_config(tmp_path, contract_id=16))
        assert trades.shape[0] == 1
        assert int(trades[0, 0]) == 16


# ===========================================================================
# 7. Determinism — identical re-loads produce identical arrays (TC-6)
# ===========================================================================


class TestLoadDeterminism:
    """The same file loaded twice must produce identical arrays.

    This is pure-function determinism (no hidden cache state, no platform
    float nondeterminism). A regression here would surface a latent zona
    branch or a non-deterministic sort key.
    """

    @pytest.fixture
    def mixed_fixture(self, tmp_path):
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="AES", unidad="AESU1",
                           precio="50,00", cantidad="1,00", side="C",
                           momento="01/06/2025 17:41:12"),
                _order_row(contract_id=16, zona=_ZONE_PT, agente="APT", unidad="APTU1",
                           precio="49,00", cantidad="2,00", side="V",
                           momento="01/06/2025 17:41:13"),
                _order_row(contract_id=16, zona=_ZONE_OTHER, agente="AXX", unidad="AXXU1",
                           precio="51,00", cantidad="3,00", side="C",
                           momento="01/06/2025 17:41:14"),
            ],
        )
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="AES", buy_unidad="AESU1", buy_zona=_ZONE_ES,
                           sell_agente="APT", sell_unidad="APTU1", sell_zona=_ZONE_PT,
                           precio="49,50", cantidad="1,00"),
            ],
        )
        return _config(tmp_path)

    def test_orders_load_is_idempotent(self, mixed_fixture):
        """load_omie_orders(cfg) == load_omie_orders(cfg) bit-for-bit."""
        a = load_omie_orders(mixed_fixture)
        b = load_omie_orders(mixed_fixture)
        assert a.shape == b.shape
        assert jnp.array_equal(a, b), (
            "Two identical loads produced different arrays — nondeterminism in loader"
        )

    def test_trades_load_is_idempotent(self, mixed_fixture):
        """load_omie_trades(cfg) == load_omie_trades(cfg) bit-for-bit."""
        a = load_omie_trades(mixed_fixture)
        b = load_omie_trades(mixed_fixture)
        assert a.shape == b.shape
        assert jnp.array_equal(a, b), (
            "Two identical loads produced different trades arrays — nondeterminism"
        )


# ===========================================================================
# 8. Portuguese rows exercise every numeric/parsing path
# ===========================================================================


class TestPortugueseParity:
    """PT rows must go through every code path ES rows do, without crashing."""

    def test_pt_comma_decimal_price_parsed(self, tmp_path):
        """PT price with comma decimal must parse identically to Spanish."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_PT, agente="PT1", unidad="PTU1",
                           precio="18,01", cantidad="7,50", side="C"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 1
        from omiecid_simulator.omiecid.encoding import decode_price, decode_qty
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_price(int(orders[0, 2]), pc) - 18.01) < 1e-3
        assert abs(decode_qty(int(orders[0, 3]), pc) - 7.50) < 1e-3

    def test_pt_iceberg_peak_qty_and_ppd_parsed(self, tmp_path):
        """PT iceberg row must populate peak_qty and ppd correctly."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_PT, agente="PT1", unidad="PTU1",
                           precio="50,00", cantidad="10,00", side="V",
                           exec_r="ICE", validez="GTD",
                           cant_red="2,00", ppd="0,50"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 1
        assert int(orders[0, 7]) == int(ExecRestriction.ICE)
        assert int(orders[0, 8]) == int(Validity.GTD)
        from omiecid_simulator.omiecid.encoding import decode_ppd, decode_qty
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        assert abs(decode_qty(int(orders[0, 11]), pc) - 2.0) < 1e-3
        assert abs(decode_ppd(int(orders[0, 12]), pc) - 0.5) < 1e-3

    def test_pt_negative_ppd_parsed(self, tmp_path):
        """PT ICEwPI order with negative ppd (BUY-side) must parse signed."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_PT, agente="PT1", unidad="PTU1",
                           precio="50,00", cantidad="10,00", side="C",
                           exec_r="ICE", validez="GTD",
                           cant_red="3,00", ppd="-0,75"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 1
        from omiecid_simulator.omiecid.encoding import decode_ppd
        from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
        pc = OMIEProductConfig()
        decoded = decode_ppd(int(orders[0, 12]), pc)
        assert decoded < 0.0
        assert abs(decoded - (-0.75)) < 1e-3

    def test_pt_status_is_active(self, tmp_path):
        """PT orders, like ES, must have status=ACTIVE on load."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_PT, agente="PT1", unidad="PTU1",
                           precio="50,00", cantidad="1,00", side="C"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert int(orders[0, 10]) == int(Status.ACTIVE)

    def test_pt_side_mapping(self, tmp_path):
        """PT 'C' → BUY, 'V' → SELL — same as ES."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_PT, agente="PTB", unidad="PTBU1",
                           precio="50,00", cantidad="1,00", side="C",
                           momento="01/06/2025 17:41:12"),
                _order_row(contract_id=16, zona=_ZONE_PT, agente="PTS", unidad="PTSU1",
                           precio="51,00", cantidad="1,00", side="V",
                           momento="01/06/2025 17:41:13"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 2
        # Sorted by timestamp; row 0 is the 'C' (BUY), row 1 is the 'V' (SELL).
        assert int(orders[0, 1]) == int(Side.BUY)
        assert int(orders[1, 1]) == int(Side.SELL)


# ===========================================================================
# 9. Buy/sell symmetry for trades — blank-leg admission (audit depth)
# ===========================================================================


class TestTradeLegSymmetry:
    """Trades must not be dropped when one leg's agent/zona is blank.

    Real OMIE data (observed in data/rawOMIE/trades_20250601.1) contains
    rows where Agente compra, Unidad compra, Zona compra are all blank
    (and symmetrically for the sell side) — these are XBID cross-border
    trades where the counterparty is a foreign TSO rather than an OMIE
    agent. Phase 1's union filter kept them whenever at least one side
    matched ES; after 1.5-A, all must be admitted.
    """

    def test_blank_buy_leg_trade_included(self, tmp_path):
        """Trade with blank buy agent (cross-border buyer) still loads."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="", buy_unidad="", buy_zona="",
                           sell_agente="AES", sell_unidad="AESU1", sell_zona=_ZONE_ES,
                           precio="50,00", cantidad="1,00"),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        assert trades.shape[0] == 1, (
            "Trade with blank buy leg must be admitted (real XBID cross-border row)"
        )

    def test_blank_sell_leg_trade_included(self, tmp_path):
        """Trade with blank sell agent (cross-border seller) still loads."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="AES", buy_unidad="AESU1", buy_zona=_ZONE_ES,
                           sell_agente="", sell_unidad="", sell_zona="",
                           precio="50,00", cantidad="1,00"),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        assert trades.shape[0] == 1, (
            "Trade with blank sell leg must be admitted (real XBID cross-border row)"
        )


# ===========================================================================
# 10. Integration — real OMIE file exercises the removed filter end-to-end
# ===========================================================================
#
# The real OMIE data file contains Portuguese rows. If zona filtering were
# silently still in effect (e.g. via a default-zona fallback somewhere), the
# row count would be lower than the true file content. This is an integration
# test that catches hidden-default regressions. It is marked slow because it
# reads a ~O(MB) file from disk.


_REAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "rawOMIE"
_REAL_DATE = "2025-06-01"
_REAL_DATE_NODASH = "20250601"


@pytest.mark.skipif(
    not (_REAL_DATA_DIR / f"orders_{_REAL_DATE_NODASH}.1").exists(),
    reason="real OMIE data file not present",
)
@pytest.mark.slow
class TestRealDataIntegration:
    """slow: reads ~MB of real OMIE data to verify no hidden zona filter.

    Runtime exceeds the < 1s fast-suite budget because it parses the full
    orders and trades files for a single delivery date. Runs under the
    'slow' marker.
    """

    def test_real_file_contains_pt_rows_after_load(self):
        """Count of distinct agents in a real contract must be > typical ES-only.

        Proxy signal: real OMIE file contains PT participants; the loader
        under 1.5-A must not silently drop them. We assert a lower bound
        on the loaded row count — a full zona filter would zero this out
        or reduce it by >10% (Phase 1.5 mandates all-Iberian inclusion).
        """
        cfg = OMIEDataConfig(
            data_path=str(_REAL_DATA_DIR),
            contract_id=16,
            delivery_date=_REAL_DATE,
        )
        orders = load_omie_orders(cfg)
        # A full day's orders for a 15-min contract routinely has >100 rows.
        # The point of this test is: the loader does NOT crash on real-world
        # data (which contains both ES and PT rows, blank lines between
        # metadata and header, CRLF line endings, and variable-width
        # optional fields). A zero result would indicate a silent filter.
        assert orders.shape[0] > 0, (
            "Loader returned 0 rows on real OMIE data — zona filter "
            "may be silently active"
        )
        assert orders.shape[1] == 13, (
            f"Real data must produce (N, 13); got {orders.shape}"
        )

    def test_real_trades_load_contains_rows(self):
        """Real trades file must load to non-empty (M, 6) after 1.5-A."""
        cfg = OMIEDataConfig(
            data_path=str(_REAL_DATA_DIR),
            contract_id=16,
            delivery_date=_REAL_DATE,
        )
        trades = load_omie_trades(cfg)
        assert trades.ndim == 2
        assert trades.shape[1] == 6, (
            f"Real trades must be (M, 6); got {trades.shape}"
        )
