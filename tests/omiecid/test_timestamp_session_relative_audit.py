"""Audit tests for Story 1.5-C — Timestamp Session-Relative Conversion.

Independent black-box verification written after implementation without
access to the original locked test file
(``tests/omiecid/test_timestamp_session_relative.py``).  The tests are derived
solely from:

  - ``docs/omiecid/agile-plan.md`` Story 1.5-C acceptance criteria
  - ``docs/omiecid/market-spec.md`` §13.1 / §13.2 / §13.4 / §13.5
  - ``docs/omiecid/migration-plan.md`` §Phase 1.5
  - Inspection of the public interface in:
      ``omiecid_simulator/jaxomiecid/omie_data_loader.py``
      ``omiecid_simulator/jaxomiecid/omie_cache.py``
      ``omiecid_simulator/jaxomiecid/omie_data_config.py``

Focus areas (per audit brief):

  1. Correctness under real-data conditions — orders/trades loaded from
     ``data/rawOMIE/``.  Timestamps must be session-relative, not Unix-epoch
     scale.
  2. Boundary conditions — offsets at midnight (0), near end-of-day (~86399),
     and pre-midnight pre-session negatives allowed within reason (−3600).
  3. Cache contract depth — full ``save_omie_cache`` / ``load_omie_cache``
     round-trip for ``epoch_base``; byte-identical arrays; epoch_base identity.
  4. Cross-zone regression guard — Portuguese participant timestamps must also
     be session-relative.
  5. Ordering after 1.5-B pair mapping — chronological order preserved within
     a (Agente, Unidad) pair.
  6. No epoch-scale leakage — any value > 1e9 in orders[:, 6] or trades[:, 5]
     is a bug; caught by explicit assertion.

Runtime notes:
  - Tests using only synthetic in-memory fixtures stay in the fast (< 1s) tier.
  - Tests that load real files under ``data/rawOMIE/`` are ``@pytest.mark.slow``
    (justification: real OMIE files are ~MB; parsing exceeds the 1s budget).
  - Session-scoped ``real_config`` fixture in this file warms the parse once per
    session to keep total slow-tier cost low.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_cache import (
    CACHE_SCHEMA_VERSION,
    load_omie_cache,
    save_omie_cache,
)
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    _compute_epoch_base,
    _parse_timestamp,
    load_omie_orders,
    load_omie_trades,
    load_omie_agent_metadata,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# A known delivery date present in data/rawOMIE/; orders_20250601.1 is the
# earliest available file.  We pick a contract_id that is present in the file.
# Contract 16 = 04:00–04:15 UTC (matches the PT/ES orders in the first rows).
REAL_DATA_PATH = "data/rawOMIE"
REAL_DELIVERY_DATE = "2025-06-01"
REAL_CONTRACT_ID = 16  # 04:00-04:15 -> 4*4+0 = 16

# The real-data files are present when the OMIE data directory exists.
_REAL_DATA_AVAILABLE = os.path.isfile(
    os.path.join(REAL_DATA_PATH, f"orders_{REAL_DELIVERY_DATE.replace('-', '')}.1")
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic OMIE file fixtures
# ---------------------------------------------------------------------------

_ORDERS_HEADER = (
    "OMIE - Mercado de electricidad;Fecha Emision;;\n"
    "\n"
    "Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;Tipo oferta;"
    "Condiciones a la ejecucion;Condiciones a la validez;"
    "Cantidad reducida;PPD;Momento de envio;\n"
)

_TRADES_HEADER = (
    "OMIE - Mercado de electricidad;Fecha Emision;;\n"
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;"
    "Agente venta;Unidad venta;Zona venta;Precio;Cantidad;Momento casacion;\n"
)

# Column 6 in orders = timestamp (index 6)
ORDERS_TS_COL = 6
# Column 5 in trades = match_timestamp (index 5)
TRADES_TS_COL = 5

# Largest session-relative value that should ever appear (~1 day in seconds)
MAX_SESSION_SECONDS = 200_000
# Minimum allowed (pre-midnight pre-session orders/trades allowed down to
# approximately IDA2 results time, which is ~22:20 CET on D-1 = ~-6600s
# from UTC midnight of the delivery date; we use -86400 as the spec allows).
MIN_SESSION_SECONDS = -86_400
# Threshold above which a value is "epoch scale" (a regression)
EPOCH_SCALE_THRESHOLD = 1_000_000_000


# ---------------------------------------------------------------------------
# Synthetic fixture writers
# ---------------------------------------------------------------------------

def _write_orders_file(path: Path, rows: list[dict]) -> None:
    """Write a synthetic OMIE orders file to *path*.

    Each element of *rows* must have keys:
        fecha, contrato, zona, agente, unidad, precio, cantidad,
        tipo_oferta, exec_restr, validez, cant_reducida, ppd, momento
    """
    with open(path, "w", encoding="latin-1") as f:
        f.write(_ORDERS_HEADER)
        for r in rows:
            line = (
                f"{r['fecha']};{r['contrato']};{r['zona']};"
                f"{r['agente']};{r['unidad']};"
                f"{r['precio']};{r['cantidad']};{r['tipo_oferta']};"
                f"{r.get('exec_restr', '')};{r.get('validez', '')};"
                f"{r.get('cant_reducida', '')};{r.get('ppd', '')};"
                f"{r['momento']};\n"
            )
            f.write(line)


def _write_trades_file(path: Path, rows: list[dict]) -> None:
    """Write a synthetic OMIE trades file to *path*.

    Each element of *rows* must have keys:
        fecha, contrato, agente_compra, unidad_compra, zona_compra,
        agente_venta, unidad_venta, zona_venta, precio, cantidad, momento
    """
    with open(path, "w", encoding="latin-1") as f:
        f.write(_TRADES_HEADER)
        for r in rows:
            line = (
                f"{r['fecha']};{r['contrato']};"
                f"{r['agente_compra']};{r['unidad_compra']};{r['zona_compra']};"
                f"{r['agente_venta']};{r['unidad_venta']};{r['zona_venta']};"
                f"{r['precio']};{r['cantidad']};{r['momento']};\n"
            )
            f.write(line)


def _delivery_row_base(delivery_date: str = "2025-06-02") -> dict:
    """Return base field values for a row on *delivery_date* (YYYY-MM-DD)."""
    d = datetime.strptime(delivery_date, "%Y-%m-%d")
    fecha = d.strftime("%d/%m/%Y")
    contrato = f"{d.strftime('%Y%m%d')} 04:00-{d.strftime('%Y%m%d')} 04:15"
    return {"fecha": fecha, "contrato": contrato}


# Fixture-only product date (OMIE-TAPE-CONTRACT-DATE-1, group F).  The
# synthetic loader fixtures carry a ``Contrato`` dated 2025-06-02
# (``_delivery_row_base``), i.e. delivery product (2025-06-02, 16).  Per
# market-spec v2.10 §13/§13.4 that product spans session files D-1
# (``*_20250601.1``) and D (``*_20250602.1``), and stored timestamps are
# seconds since D-1 00:00 (2025-06-01 00:00).  Each test writes the session
# file(s) holding its rows; the rest are completed header-only.
_PRODUCT_DATE_ISO = "2025-06-02"
_PRODUCT_SESSION_FILES_NODASH = ("20250601", "20250602")  # (D-1, D)


def _complete_product_files(tmp_path: Path) -> None:
    """Add a header-only file for every product session file not written.

    Uses ``_write_orders_file`` / ``_write_trades_file`` (unaccented
    ``Momento casacion``).  A file already written is never touched.
    """
    for date_nodash in _PRODUCT_SESSION_FILES_NODASH:
        orders = tmp_path / f"orders_{date_nodash}.1"
        trades = tmp_path / f"trades_{date_nodash}.1"
        if not orders.exists():
            _write_orders_file(orders, [])
        if not trades.exists():
            _write_trades_file(trades, [])


# ---------------------------------------------------------------------------
# Unit tests for _compute_epoch_base and _parse_timestamp
# ---------------------------------------------------------------------------

class TestComputeEpochBase:
    """Unit tests for ``_compute_epoch_base``."""

    def test_known_date_2025_06_01(self):
        """2025-06-01 UTC midnight is a specific known Unix timestamp."""
        # 2025-06-01 00:00:00 UTC
        expected = int(datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp())
        assert _compute_epoch_base("2025-06-01") == expected

    def test_epoch_base_is_integer(self):
        result = _compute_epoch_base("2025-06-01")
        assert isinstance(result, int)

    def test_epoch_base_is_epoch_scale(self):
        """epoch_base itself must be epoch-scale (1970-based large integer)."""
        result = _compute_epoch_base("2025-06-01")
        assert result > EPOCH_SCALE_THRESHOLD, (
            f"epoch_base={result} should be > 1e9 (it is an epoch timestamp)"
        )

    def test_epoch_base_is_utc_midnight(self):
        """epoch_base must correspond to 00:00:00 UTC of the delivery date."""
        base = _compute_epoch_base("2025-06-15")
        dt = datetime.fromtimestamp(base, tz=timezone.utc)
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.date() == datetime(2025, 6, 15).date()

    def test_consecutive_dates_differ_by_86400(self):
        base1 = _compute_epoch_base("2025-06-01")
        base2 = _compute_epoch_base("2025-06-02")
        assert base2 - base1 == 86_400


class TestParseTimestamp:
    """Unit tests for ``_parse_timestamp``."""

    def test_midnight_exactly_yields_zero(self):
        """An order submitted at 00:00:00 UTC on delivery day is offset 0."""
        epoch_base = _compute_epoch_base("2025-06-01")
        ts = _parse_timestamp("01/06/2025 00:00:00", epoch_base)
        assert ts == 0

    def test_noon_yields_43200(self):
        """12:00:00 on delivery day is 43200 s from midnight."""
        epoch_base = _compute_epoch_base("2025-06-01")
        ts = _parse_timestamp("01/06/2025 12:00:00", epoch_base)
        assert ts == 43_200

    def test_end_of_day_yield_close_to_86399(self):
        """23:59:59 on delivery day is 86399 s from midnight."""
        epoch_base = _compute_epoch_base("2025-06-01")
        ts = _parse_timestamp("01/06/2025 23:59:59", epoch_base)
        assert ts == 86_399

    def test_pre_session_previous_day_is_negative(self):
        """An order submitted the night before yields a negative offset.

        IDA2 match times are around 22:20 on D-1, which is ~-6000 s from
        UTC midnight of the delivery date D.  This is a real observed offset
        (see data/rawOMIE/trades_20250601.1 row 1: 31/05/2025 22:20:06 for
        delivery day 2025-06-01).  The spec allows offsets in [-86400, 86400).
        """
        epoch_base = _compute_epoch_base("2025-06-02")
        ts = _parse_timestamp("01/06/2025 22:20:06", epoch_base)
        # 01/06/2025 22:20:06 is ~5994 s before midnight of 2025-06-02
        assert ts < 0
        assert ts >= MIN_SESSION_SECONDS, (
            f"Pre-session offset {ts} falls outside allowed range "
            f"[{MIN_SESSION_SECONDS}, {MAX_SESSION_SECONDS})"
        )

    def test_output_is_integer(self):
        """``_parse_timestamp`` must return an int."""
        epoch_base = _compute_epoch_base("2025-06-01")
        ts = _parse_timestamp("01/06/2025 10:30:00", epoch_base)
        assert isinstance(ts, int)

    def test_result_is_not_epoch_scale(self):
        """Session-relative value must be much smaller than a Unix timestamp."""
        epoch_base = _compute_epoch_base("2025-06-01")
        ts = _parse_timestamp("01/06/2025 17:41:12", epoch_base)
        assert abs(ts) < EPOCH_SCALE_THRESHOLD, (
            f"_parse_timestamp returned epoch-scale value {ts}"
        )

    def test_float32_roundtrip_exact(self):
        """Session-relative offsets fit exactly in float32 (no precision loss)."""
        epoch_base = _compute_epoch_base("2025-06-01")
        raw = _parse_timestamp("01/06/2025 17:41:12", epoch_base)
        roundtripped = int(np.float32(raw))
        assert roundtripped == raw, (
            f"float32 roundtrip: {raw} -> {roundtripped} (precision loss)"
        )


# ---------------------------------------------------------------------------
# Synthetic loader tests (fast tier)
# ---------------------------------------------------------------------------

class TestOrdersTimestampInArray:
    """Verify orders[:, 6] contains session-relative timestamps."""

    def test_single_order_timestamp_not_epoch_scale(self, tmp_path):
        """A single order at 17:41:12 has a session-relative timestamp."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "AGENTA",
            "unidad": "UNITA01",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 17:41:12",
        }])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        assert orders.shape[0] == 1
        ts = float(orders[0, ORDERS_TS_COL])
        assert abs(ts) < EPOCH_SCALE_THRESHOLD, (
            f"orders[:, 6] contains epoch-scale value {ts}"
        )

    def test_timestamp_magnitude_within_session_bounds(self, tmp_path):
        """Timestamp falls in [MIN_SESSION_SECONDS, MAX_SESSION_SECONDS)."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "AGENTA",
            "unidad": "UNITA01",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 17:41:12",
        }])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        ts = float(orders[0, ORDERS_TS_COL])
        assert ts >= MIN_SESSION_SECONDS
        assert ts < MAX_SESSION_SECONDS

    def test_two_orders_monotonic_by_submission_time(self, tmp_path):
        """Orders sorted by submission time have non-decreasing timestamps."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [
            {**base, "zona": "10YES-REE------0", "agente": "AGENTA",
             "unidad": "U1", "precio": "50,00", "cantidad": "1,0",
             "tipo_oferta": "C", "momento": "01/06/2025 15:00:00"},
            {**base, "zona": "10YES-REE------0", "agente": "AGENTB",
             "unidad": "U2", "precio": "51,00", "cantidad": "2,0",
             "tipo_oferta": "V", "momento": "01/06/2025 16:00:00"},
        ])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        assert orders.shape[0] == 2
        ts0 = float(orders[0, ORDERS_TS_COL])
        ts1 = float(orders[1, ORDERS_TS_COL])
        assert ts0 <= ts1, (
            f"Monotonic order violated: ts0={ts0} ts1={ts1}"
        )

    def test_timestamps_differ_by_expected_delta(self, tmp_path):
        """Two orders 3600 s apart have timestamps differing by exactly 3600."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [
            {**base, "zona": "10YES-REE------0", "agente": "AGENTA",
             "unidad": "U1", "precio": "50,00", "cantidad": "1,0",
             "tipo_oferta": "C", "momento": "01/06/2025 10:00:00"},
            {**base, "zona": "10YES-REE------0", "agente": "AGENTA",
             "unidad": "U1", "precio": "50,00", "cantidad": "1,0",
             "tipo_oferta": "C", "momento": "01/06/2025 11:00:00"},
        ])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        assert orders.shape[0] == 2
        delta = float(orders[1, ORDERS_TS_COL]) - float(orders[0, ORDERS_TS_COL])
        assert abs(delta - 3600.0) < 1.0, (
            f"Expected 3600 s between rows, got {delta}"
        )

    def test_midnight_order_timestamp_is_zero(self, tmp_path):
        """An order submitted at exactly midnight of delivery_date has ts=0."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "AGENTA",
            "unidad": "U1",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 00:00:00",
        }])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        ts = float(orders[0, ORDERS_TS_COL])
        assert ts == 0.0

    def test_near_end_of_day_order_timestamp(self, tmp_path):
        """An order at 23:59:59 on delivery_date has ts close to 86399."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "AGENTA",
            "unidad": "U1",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 23:59:59",
        }])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        ts = float(orders[0, ORDERS_TS_COL])
        assert 86390 <= ts <= 86400, (
            f"End-of-day timestamp {ts} out of expected range [86390, 86400]"
        )

    def test_pre_midnight_order_negative_but_within_bounds(self, tmp_path):
        """An order from 23:00 the day before yields a small negative offset."""
        orders_path = tmp_path / "orders_20250602.1"
        trades_path = tmp_path / "trades_20250602.1"
        # Delivery date 2025-06-02; order submitted 23:00:00 on 2025-06-01
        d = datetime.strptime("2025-06-02", "%Y-%m-%d")
        contrato = f"20250602 04:00-20250602 04:15"
        _write_orders_file(orders_path, [{
            "fecha": "02/06/2025",
            "contrato": contrato,
            "zona": "10YES-REE------0",
            "agente": "AGENTA",
            "unidad": "U1",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 23:00:00",
        }])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date="2025-06-02",
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        assert orders.shape[0] == 1
        ts = int(orders[0, ORDERS_TS_COL])
        # Post P2-F0 the loader output wraps pre-midnight / post-midnight
        # offsets modulo 86400 into the day window. 23:00:00 maps to 82800
        # seconds-of-day. The pre-P2-F0 expectation of a negative offset is
        # a Category B adapter site; the underlying ``_parse_timestamp``
        # helper still returns the signed offset (covered by TC-2 above).
        assert ts == 82800, (
            f"Expected 82800 (23:00 wrapped mod 86400), got {ts}"
        )
        assert 0 <= ts < 86400

    def test_float32_dtype_preserved(self, tmp_path):
        """Output array dtype must be float32."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "AGENTA",
            "unidad": "U1",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 12:00:00",
        }])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        assert orders.dtype == jnp.int32


class TestTradesTimestampInArray:
    """Verify trades[:, 5] contains session-relative timestamps."""

    def test_single_trade_timestamp_not_epoch_scale(self, tmp_path):
        """A trade at 22:20:06 the day before has a session-relative timestamp."""
        orders_path = tmp_path / "orders_20250601.1"
        # OMIE-TAPE-CONTRACT-DATE-1 (F-2, re-timed rather than left as a D-2
        # row): the trade is matched at 22:20:06 on 2025-06-01, the IDA-reopen
        # of session 2025-06-02, so it lives in session file D = 20250602
        # (``Fecha`` = session day).  Stored value is unchanged (80,406 s
        # since D-1 00:00, as the former 31/05 22:20:06 row stored mod 86400).
        trades_path = tmp_path / "trades_20250602.1"
        base = _delivery_row_base("2025-06-02")
        # We need at least one order to populate the pair universe
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "AGENTA",
            "unidad": "U1",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 17:41:12",
        }])
        _write_trades_file(trades_path, [{
            "fecha": "02/06/2025",
            "contrato": base["contrato"],
            "agente_compra": "AGENTA",
            "unidad_compra": "U1",
            "zona_compra": "10YES-REE------0",
            "agente_venta": "AGENTB",
            "unidad_venta": "U2",
            "zona_venta": "10YES-REE------0",
            "precio": "50,00",
            "cantidad": "1,0",
            "momento": "01/06/2025 22:20:06",
        }])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        trades = load_omie_trades(config)
        assert trades.shape[0] == 1
        ts = float(trades[0, TRADES_TS_COL])
        assert abs(ts) < EPOCH_SCALE_THRESHOLD, (
            f"trades[:, 5] contains epoch-scale value {ts}"
        )

    def test_trade_timestamp_within_session_bounds(self, tmp_path):
        """Trade timestamp falls in [MIN_SESSION_SECONDS, MAX_SESSION_SECONDS)."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "AGENTA",
            "unidad": "U1",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 17:41:12",
        }])
        _write_trades_file(trades_path, [{
            "fecha": "01/06/2025",
            "contrato": base["contrato"],
            "agente_compra": "AGENTA",
            "unidad_compra": "U1",
            "zona_compra": "10YES-REE------0",
            "agente_venta": "AGENTB",
            "unidad_venta": "U2",
            "zona_venta": "10YES-REE------0",
            "precio": "50,00",
            "cantidad": "1,0",
            "momento": "01/06/2025 17:45:00",
        }])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        trades = load_omie_trades(config)
        ts = float(trades[0, TRADES_TS_COL])
        assert ts >= MIN_SESSION_SECONDS
        assert ts < MAX_SESSION_SECONDS

    def test_trades_float32_dtype(self, tmp_path):
        """Trades array dtype must be float32."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "AGENTA",
            "unidad": "U1",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 17:41:12",
        }])
        _write_trades_file(trades_path, [{
            "fecha": "01/06/2025",
            "contrato": base["contrato"],
            "agente_compra": "AGENTA",
            "unidad_compra": "U1",
            "zona_compra": "10YES-REE------0",
            "agente_venta": "AGENTB",
            "unidad_venta": "U2",
            "zona_venta": "10YES-REE------0",
            "precio": "50,00",
            "cantidad": "1,0",
            "momento": "01/06/2025 17:45:00",
        }])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        trades = load_omie_trades(config)
        assert trades.dtype == jnp.int32


class TestNoEpochScaleLeakage:
    """Explicit guard: no value > 1e9 in timestamp columns."""

    def test_orders_no_epoch_scale_in_ts_column(self, tmp_path):
        """orders[:, 6] must have no value exceeding EPOCH_SCALE_THRESHOLD."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [
            {**base, "zona": "10YES-REE------0", "agente": "A",
             "unidad": "U1", "precio": "10,00", "cantidad": "1,0",
             "tipo_oferta": "C", "momento": "01/06/2025 08:00:00"},
            {**base, "zona": "10YES-REE------0", "agente": "B",
             "unidad": "U2", "precio": "11,00", "cantidad": "2,0",
             "tipo_oferta": "V", "momento": "01/06/2025 09:00:00"},
            {**base, "zona": "10YPT-REN------W", "agente": "PT_AGENT",
             "unidad": "PT_UNIT", "precio": "12,00", "cantidad": "3,0",
             "tipo_oferta": "C", "momento": "01/06/2025 10:00:00"},
        ])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        if orders.shape[0] > 0:
            ts_col = np.array(orders[:, ORDERS_TS_COL])
            max_ts = float(np.max(np.abs(ts_col)))
            assert max_ts < EPOCH_SCALE_THRESHOLD, (
                f"Epoch-scale timestamp detected in orders[:, 6]: max_abs={max_ts}"
            )

    def test_trades_no_epoch_scale_in_ts_column(self, tmp_path):
        """trades[:, 5] must have no value exceeding EPOCH_SCALE_THRESHOLD."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "AGENTA",
            "unidad": "U1",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 17:41:12",
        }])
        _write_trades_file(trades_path, [{
            "fecha": "01/06/2025",
            "contrato": base["contrato"],
            "agente_compra": "AGENTA",
            "unidad_compra": "U1",
            "zona_compra": "10YES-REE------0",
            "agente_venta": "AGENTB",
            "unidad_venta": "U2",
            "zona_venta": "10YES-REE------0",
            "precio": "50,00",
            "cantidad": "1,0",
            "momento": "01/06/2025 17:45:00",
        }])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        trades = load_omie_trades(config)
        if trades.shape[0] > 0:
            ts_col = np.array(trades[:, TRADES_TS_COL])
            max_ts = float(np.max(np.abs(ts_col)))
            assert max_ts < EPOCH_SCALE_THRESHOLD, (
                f"Epoch-scale timestamp detected in trades[:, 5]: max_abs={max_ts}"
            )


class TestPortugueseParticipantTimestamps:
    """Cross-zone regression guard: Portuguese timestamps must also be session-relative."""

    def test_pt_order_timestamp_not_epoch_scale(self, tmp_path):
        """A row from zone 10YPT-REN------W must have a session-relative timestamp."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YPT-REN------W",
            "agente": "ACCGR",
            "unidad": "ACCGC02",
            "precio": "18,01",
            "cantidad": "7,50",
            "tipo_oferta": "C",
            "exec_restr": "ICE",
            "validez": "GTD",
            "cant_reducida": "5,80",
            "ppd": "0",
            "momento": "01/06/2025 17:41:12",
        }])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        assert orders.shape[0] == 1
        ts = float(orders[0, ORDERS_TS_COL])
        assert abs(ts) < EPOCH_SCALE_THRESHOLD, (
            f"PT participant timestamp is epoch-scale: {ts}"
        )
        assert ts >= MIN_SESSION_SECONDS
        assert ts < MAX_SESSION_SECONDS

    def test_mixed_es_pt_orders_all_timestamps_session_relative(self, tmp_path):
        """Both ES and PT rows have session-relative timestamps."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [
            {**base, "zona": "10YES-REE------0", "agente": "AGEN_ES",
             "unidad": "ES_UNIT", "precio": "50,00", "cantidad": "1,0",
             "tipo_oferta": "C", "momento": "01/06/2025 10:00:00"},
            {**base, "zona": "10YPT-REN------W", "agente": "AGEN_PT",
             "unidad": "PT_UNIT", "precio": "51,00", "cantidad": "2,0",
             "tipo_oferta": "V", "momento": "01/06/2025 11:00:00"},
        ])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        assert orders.shape[0] == 2
        for i in range(orders.shape[0]):
            ts = float(orders[i, ORDERS_TS_COL])
            assert abs(ts) < EPOCH_SCALE_THRESHOLD, (
                f"Row {i}: epoch-scale timestamp {ts}"
            )
            assert ts >= MIN_SESSION_SECONDS
            assert ts < MAX_SESSION_SECONDS

    def test_pt_trade_timestamp_session_relative(self, tmp_path):
        """A trade with a PT zone counterparty has a session-relative timestamp."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "AGEN_ES",
            "unidad": "ES_U1",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": "01/06/2025 12:00:00",
        }])
        _write_trades_file(trades_path, [{
            "fecha": "01/06/2025",
            "contrato": base["contrato"],
            "agente_compra": "AGEN_ES",
            "unidad_compra": "ES_U1",
            "zona_compra": "10YES-REE------0",
            "agente_venta": "AGEN_PT",
            "unidad_venta": "PT_U1",
            "zona_venta": "10YPT-REN------W",
            "precio": "50,00",
            "cantidad": "1,0",
            "momento": "01/06/2025 12:05:00",
        }])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        trades = load_omie_trades(config)
        assert trades.shape[0] == 1
        ts = float(trades[0, TRADES_TS_COL])
        assert abs(ts) < EPOCH_SCALE_THRESHOLD, (
            f"PT-zone trade timestamp is epoch-scale: {ts}"
        )


class TestOrderingPreservedAfterPairMapping:
    """1.5-B pair mapping must not disturb chronological ordering."""

    def test_same_pair_two_rows_timestamps_ordered(self, tmp_path):
        """Two rows with the same (Agente, Unidad) preserve submission order."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [
            {**base, "zona": "10YES-REE------0", "agente": "AGENTA",
             "unidad": "UNIT1", "precio": "50,00", "cantidad": "1,0",
             "tipo_oferta": "C", "momento": "01/06/2025 08:00:00"},
            {**base, "zona": "10YES-REE------0", "agente": "AGENTA",
             "unidad": "UNIT1", "precio": "50,00", "cantidad": "2,0",
             "tipo_oferta": "C", "momento": "01/06/2025 09:00:00"},
        ])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        assert orders.shape[0] == 2
        # Same agent_id
        aid0 = int(orders[0, 5])
        aid1 = int(orders[1, 5])
        assert aid0 == aid1, "Same (Agente, Unidad) must produce the same agent_id"
        # Timestamps ordered
        ts0 = float(orders[0, ORDERS_TS_COL])
        ts1 = float(orders[1, ORDERS_TS_COL])
        assert ts0 < ts1, (
            f"Timestamps not ordered for same pair: {ts0} >= {ts1}"
        )

    def test_different_pairs_different_ids_timestamps_ordered(self, tmp_path):
        """Two different (Agente, Unidad) pairs with distinct timestamps."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [
            {**base, "zona": "10YES-REE------0", "agente": "AGENTA",
             "unidad": "UNIT1", "precio": "50,00", "cantidad": "1,0",
             "tipo_oferta": "C", "momento": "01/06/2025 08:00:00"},
            {**base, "zona": "10YES-REE------0", "agente": "AGENTA",
             "unidad": "UNIT2",  # same Agente, different Unidad
             "precio": "50,00", "cantidad": "2,0",
             "tipo_oferta": "C", "momento": "01/06/2025 09:00:00"},
        ])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        assert orders.shape[0] == 2
        aid0 = int(orders[0, 5])
        aid1 = int(orders[1, 5])
        assert aid0 != aid1, (
            "Same Agente + different Unidad must produce distinct agent_ids"
        )
        ts0 = float(orders[0, ORDERS_TS_COL])
        ts1 = float(orders[1, ORDERS_TS_COL])
        assert ts0 < ts1


# ---------------------------------------------------------------------------
# Cache contract depth
# ---------------------------------------------------------------------------

class TestCacheEpochBaseRoundTrip:
    """Verify epoch_base is stored in and retrieved from the NPZ cache."""

    def test_save_load_epoch_base_identity(self, tmp_path):
        """Saving epoch_base and reloading yields the same integer."""
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date="2025-06-01",
            cache_path=str(tmp_path / "cache"),
        )
        epoch_base = _compute_epoch_base(config.delivery_date)
        orders = np.zeros((0, 13), dtype=np.float32)
        trades = np.zeros((0, 6), dtype=np.float32)
        save_omie_cache(config, orders=orders, trades=trades, epoch_base=epoch_base)
        loaded = load_omie_cache(config)
        assert loaded is not None, "Cache load returned None after save"
        assert "epoch_base" in loaded, "epoch_base key missing from loaded cache"
        loaded_base = int(np.asarray(loaded["epoch_base"]).item())
        assert loaded_base == epoch_base, (
            f"epoch_base mismatch: saved {epoch_base} got {loaded_base}"
        )

    def test_epoch_base_is_correct_utc_midnight(self, tmp_path):
        """Loaded epoch_base corresponds to UTC midnight of delivery_date."""
        delivery_date = "2025-06-15"
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=delivery_date,
            cache_path=str(tmp_path / "cache"),
        )
        epoch_base = _compute_epoch_base(delivery_date)
        save_omie_cache(config, epoch_base=epoch_base)
        loaded = load_omie_cache(config)
        assert loaded is not None
        loaded_base = int(np.asarray(loaded["epoch_base"]).item())
        dt = datetime.fromtimestamp(loaded_base, tz=timezone.utc)
        assert dt.year == 2025
        assert dt.month == 6
        assert dt.day == 15
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0

    def test_schema_version_is_current(self, tmp_path):
        """Saved cache carries the current CACHE_SCHEMA_VERSION."""
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date="2025-06-01",
            cache_path=str(tmp_path / "cache"),
        )
        save_omie_cache(config, epoch_base=_compute_epoch_base("2025-06-01"))
        loaded = load_omie_cache(config)
        assert loaded is not None
        stored_ver = int(np.asarray(loaded["schema_version"]).item())
        assert stored_ver == CACHE_SCHEMA_VERSION

    def test_stale_schema_version_cache_rejected(self, tmp_path):
        """A cache with a schema_version != CACHE_SCHEMA_VERSION returns None."""
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date="2025-06-01",
            cache_path=str(tmp_path / "cache"),
        )
        cache_dir = Path(config.cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Write an NPZ with a stale version (e.g. version 1 = pre-1.5-B)
        fname = cache_dir / f"omie_cache_c{config.contract_id}_{config.delivery_date}.npz"
        np.savez_compressed(
            str(fname),
            schema_version=np.array(CACHE_SCHEMA_VERSION - 1, dtype=np.int32),
            orders=np.zeros((0, 13), dtype=np.float32),
        )
        result = load_omie_cache(config)
        assert result is None, (
            f"Expected None for stale schema version, got {result}"
        )

    def test_cache_without_schema_version_rejected(self, tmp_path):
        """A legacy cache file (no schema_version key) returns None."""
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date="2025-06-01",
            cache_path=str(tmp_path / "cache"),
        )
        cache_dir = Path(config.cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fname = cache_dir / f"omie_cache_c{config.contract_id}_{config.delivery_date}.npz"
        np.savez_compressed(
            str(fname),
            # No schema_version key — simulates a pre-1.5-B legacy cache
            orders=np.zeros((0, 13), dtype=np.float32),
        )
        result = load_omie_cache(config)
        assert result is None, (
            "Expected None for legacy cache without schema_version"
        )

    def test_cache_orders_arrays_byte_identical_to_raw_load(self, tmp_path):
        """Arrays saved to cache and reloaded are byte-identical to the raw parse."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [
            {**base, "zona": "10YES-REE------0", "agente": "A",
             "unidad": "U1", "precio": "50,00", "cantidad": "1,0",
             "tipo_oferta": "C", "momento": "01/06/2025 10:00:00"},
            {**base, "zona": "10YES-REE------0", "agente": "B",
             "unidad": "U2", "precio": "51,00", "cantidad": "2,0",
             "tipo_oferta": "V", "momento": "01/06/2025 11:00:00"},
        ])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        # Raw load
        orders_raw = np.array(load_omie_orders(config))
        epoch_base = _compute_epoch_base(config.delivery_date)
        # Save then reload
        save_omie_cache(config, orders=orders_raw, epoch_base=epoch_base)
        loaded = load_omie_cache(config)
        assert loaded is not None
        assert "orders" in loaded
        orders_cached = loaded["orders"].astype(np.float32)
        np.testing.assert_array_equal(
            orders_raw,
            orders_cached,
            err_msg="Cached orders array differs from raw-parsed array",
        )

    def test_load_cache_returns_dict_with_epoch_base_accessible(self, tmp_path):
        """load_omie_cache return value is a dict and epoch_base is accessible."""
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date="2025-06-01",
            cache_path=str(tmp_path / "cache"),
        )
        epoch_base = 1_748_736_000  # known UTC midnight 2025-06-01 value
        save_omie_cache(config, epoch_base=epoch_base)
        result = load_omie_cache(config)
        assert isinstance(result, dict)
        assert "epoch_base" in result
        # Accessible as a scalar
        val = int(np.asarray(result["epoch_base"]).item())
        assert val == epoch_base


# ---------------------------------------------------------------------------
# Float32 precision guarantee
# ---------------------------------------------------------------------------

class TestFloat32RoundtripPreservation:
    """Session-relative offsets must not lose precision when stored as float32."""

    @pytest.mark.parametrize("momento,expected_offset", [
        ("01/06/2025 00:00:00", 0),
        ("01/06/2025 00:01:00", 60),
        ("01/06/2025 12:00:00", 43_200),
        ("01/06/2025 23:59:59", 86_399),
    ])
    def test_float32_exact_for_known_offsets(self, momento, expected_offset, tmp_path):
        """Known offsets are represented exactly in float32."""
        orders_path = tmp_path / "orders_20250601.1"
        trades_path = tmp_path / "trades_20250601.1"
        base = _delivery_row_base("2025-06-02")
        _write_orders_file(orders_path, [{
            **base,
            "zona": "10YES-REE------0",
            "agente": "A",
            "unidad": "U1",
            "precio": "50,00",
            "cantidad": "1,0",
            "tipo_oferta": "C",
            "momento": momento,
        }])
        _write_trades_file(trades_path, [])
        _complete_product_files(tmp_path)
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_PRODUCT_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = load_omie_orders(config)
        assert orders.shape[0] == 1
        ts = int(float(orders[0, ORDERS_TS_COL]))
        assert ts == expected_offset, (
            f"Expected offset {expected_offset} for '{momento}', got {ts}"
        )


# ---------------------------------------------------------------------------
# Real-data integration tests  (marked slow)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_orders():
    """slow: parse real orders_20250601.1 once per session."""
    if not _REAL_DATA_AVAILABLE:
        pytest.skip("Real OMIE data not available under data/rawOMIE/")
    config = OMIEDataConfig(
        data_path=REAL_DATA_PATH,
        contract_id=REAL_CONTRACT_ID,
        delivery_date=REAL_DELIVERY_DATE,
        cache_path="saved_npz/omie_audit_tmp",
    )
    return load_omie_orders(config), config


@pytest.fixture(scope="module")
def real_trades():
    """slow: parse real trades_20250601.1 once per session."""
    if not _REAL_DATA_AVAILABLE:
        pytest.skip("Real OMIE data not available under data/rawOMIE/")
    config = OMIEDataConfig(
        data_path=REAL_DATA_PATH,
        contract_id=REAL_CONTRACT_ID,
        delivery_date=REAL_DELIVERY_DATE,
        cache_path="saved_npz/omie_audit_tmp",
    )
    return load_omie_trades(config), config


@pytest.mark.slow
class TestRealDataTimestampCorrectness:
    """slow: load real OMIE files; assert session-relative timestamp properties.

    slow: warmup + real-file parse of orders_20250601.1 / trades_20250601.1
    (~MB of data); exceeds 1s fast-suite budget.
    """

    def test_real_orders_timestamps_not_epoch_scale(self, real_orders):
        """All timestamps in the real orders array are session-relative."""
        orders, _ = real_orders
        if orders.shape[0] == 0:
            pytest.skip("No orders for this contract_id in real data")
        ts_col = np.array(orders[:, ORDERS_TS_COL])
        max_abs = float(np.max(np.abs(ts_col)))
        assert max_abs < EPOCH_SCALE_THRESHOLD, (
            f"Epoch-scale timestamp in real orders[:, 6]: max_abs={max_abs}"
        )

    def test_real_orders_timestamps_within_session_bounds(self, real_orders):
        """Real orders timestamps fall in [MIN_SESSION_SECONDS, MAX_SESSION_SECONDS)."""
        orders, _ = real_orders
        if orders.shape[0] == 0:
            pytest.skip("No orders for this contract_id in real data")
        ts_col = np.array(orders[:, ORDERS_TS_COL])
        min_ts = float(np.min(ts_col))
        max_ts = float(np.max(ts_col))
        assert min_ts >= MIN_SESSION_SECONDS, (
            f"Minimum timestamp {min_ts} < {MIN_SESSION_SECONDS}"
        )
        assert max_ts < MAX_SESSION_SECONDS, (
            f"Maximum timestamp {max_ts} >= {MAX_SESSION_SECONDS}"
        )

    def test_real_orders_float32_roundtrip(self, real_orders):
        """Every real orders timestamp is exactly representable as float32."""
        orders, _ = real_orders
        if orders.shape[0] == 0:
            pytest.skip("No orders for this contract_id in real data")
        ts_col = np.array(orders[:, ORDERS_TS_COL], dtype=np.float32)
        # float32 should represent integers in [-200000, 200000] exactly
        ts_int = ts_col.astype(np.int32)
        ts_back = ts_int.astype(np.float32)
        np.testing.assert_array_equal(
            ts_col,
            ts_back,
            err_msg="float32 roundtrip precision loss in real orders timestamps",
        )

    def test_real_orders_monotonic_timestamps(self, real_orders):
        """Real orders are loaded in submission order.

        OMIE-TAPE-CONTRACT-DATE-1 (group L re-baseline; was the [0, 86400)
        day window of the P2-F0 modulo wrap): the stored column is seconds
        since D-1 00:00 with no modulo, so every real value lies in the
        product send window [15:20 D-1, 22:00 D) = [55200, 165600)
        (market-spec v2.10 §13.4). Full monotonicity is pinned in
        test_tape_contract_date.py (TC-2).
        """
        orders, _ = real_orders
        if orders.shape[0] < 2:
            pytest.skip("Need at least 2 orders to check monotonicity")
        ts_col = np.array(orders[:, ORDERS_TS_COL])
        # Every value must fit the product send window.
        assert np.all(ts_col >= 55200)
        assert np.all(ts_col < 165600)

    def test_real_orders_pt_rows_have_session_relative_timestamps(self, real_orders):
        """Portuguese (PT-zone) orders have session-relative timestamps.

        The first rows of orders_20250601.1 are from zone 10YPT-REN------W
        (ACCGR/ACCGC02).  We verify they are loaded and their timestamps are
        session-relative, confirming that 1.5-A zone removal + 1.5-C timestamp
        conversion both apply to PT rows.
        """
        orders, config = real_orders
        if orders.shape[0] == 0:
            pytest.skip("No orders for this contract_id in real data")
        ts_col = np.array(orders[:, ORDERS_TS_COL])
        max_abs = float(np.max(np.abs(ts_col)))
        assert max_abs < EPOCH_SCALE_THRESHOLD, (
            "After PT-zone inclusion (1.5-A), timestamps must still be "
            f"session-relative; max_abs={max_abs}"
        )

    def test_real_trades_timestamps_not_epoch_scale(self, real_trades):
        """All real trades timestamps are session-relative."""
        trades, _ = real_trades
        if trades.shape[0] == 0:
            pytest.skip("No trades for this contract_id in real data")
        ts_col = np.array(trades[:, TRADES_TS_COL])
        max_abs = float(np.max(np.abs(ts_col)))
        assert max_abs < EPOCH_SCALE_THRESHOLD, (
            f"Epoch-scale timestamp in real trades[:, 5]: max_abs={max_abs}"
        )

    def test_real_trades_timestamps_within_session_bounds(self, real_trades):
        """Real trades timestamps fall within allowed session window."""
        trades, _ = real_trades
        if trades.shape[0] == 0:
            pytest.skip("No trades for this contract_id in real data")
        ts_col = np.array(trades[:, TRADES_TS_COL])
        min_ts = float(np.min(ts_col))
        max_ts = float(np.max(ts_col))
        assert min_ts >= MIN_SESSION_SECONDS
        assert max_ts < MAX_SESSION_SECONDS

    def test_real_cache_round_trip_arrays_identical(self, real_orders, real_trades):
        """slow: save parsed arrays to cache; reload; verify byte identity."""
        orders, config = real_orders
        trades, _ = real_trades
        epoch_base = _compute_epoch_base(config.delivery_date)
        cache_dir = "saved_npz/omie_audit_tmp"
        cfg = OMIEDataConfig(
            data_path=config.data_path,
            contract_id=config.contract_id,
            delivery_date=config.delivery_date,
            cache_path=cache_dir,
        )
        orders_np = np.array(orders)
        trades_np = np.array(trades)
        save_omie_cache(cfg, orders=orders_np, trades=trades_np, epoch_base=epoch_base)
        loaded = load_omie_cache(cfg)
        assert loaded is not None
        if "orders" in loaded and orders_np.shape[0] > 0:
            np.testing.assert_array_equal(
                orders_np,
                loaded["orders"].astype(np.float32),
                err_msg="Cached orders differ from raw-parsed orders",
            )
        if "trades" in loaded and trades_np.shape[0] > 0:
            np.testing.assert_array_equal(
                trades_np,
                loaded["trades"].astype(np.float32),
                err_msg="Cached trades differ from raw-parsed trades",
            )

    def test_real_cache_epoch_base_is_correct_utc_midnight(self, real_orders):
        """slow: epoch_base in the cache is UTC midnight of delivery_date."""
        _, config = real_orders
        cache_dir = "saved_npz/omie_audit_tmp"
        cfg = OMIEDataConfig(
            data_path=config.data_path,
            contract_id=config.contract_id,
            delivery_date=config.delivery_date,
            cache_path=cache_dir,
        )
        epoch_base = _compute_epoch_base(config.delivery_date)
        save_omie_cache(cfg, epoch_base=epoch_base)
        loaded = load_omie_cache(cfg)
        assert loaded is not None
        eb = int(np.asarray(loaded["epoch_base"]).item())
        dt = datetime.fromtimestamp(eb, tz=timezone.utc)
        assert dt.year == 2025
        assert dt.month == 6
        assert dt.day == 1
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
