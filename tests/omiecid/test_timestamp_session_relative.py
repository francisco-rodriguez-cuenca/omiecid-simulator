"""Locked behavioral contract tests for Story 1.5-C — Timestamp Session-Relative Conversion.

Scope (derived from docs/omiecid/agile-plan.md Story 1.5-C, docs/omiecid/migration-plan.md
v4.0 §Phase 1.5, docs/omiecid/market-spec.md §13.1/§13.2/§13.4/§13.5, and the 1.5-C
pre-sprint architecture review in docs/omiecid/sprint-status.md §1.5-C — Architecture
Review):

  Timestamps produced by ``load_omie_orders`` and ``load_omie_trades`` must be
  session-relative (seconds from UTC midnight of ``delivery_date``), not absolute
  Unix epoch values (~1.75e9).  The motivating precision failure: Unix epoch
  timestamps stored as float32 have a ULP of 128 s at ~1.75×10⁹, collapsing a
  60-second separation to zero.  Session-relative values for a single day fit in
  [0, 86400) — every integer in this range is exactly representable as float32.

  Architectural decisions locked by the architecture review (F3, F4, F6):
    - All timestamps are treated as UTC-naive, with
      ``epoch_base = datetime(y, m, d, tzinfo=timezone.utc).timestamp()``.
    - ``_parse_timestamp(momento_str, epoch_base)`` now takes two arguments.
    - ``_compute_epoch_base(delivery_date: str) -> int`` is a new module-level
      helper.
    - Array dtype stays float32 — no int32 promotion (F6).
    - ``CACHE_SCHEMA_VERSION`` must equal 3 after this story (F5).
    - NPZ payload must include an ``epoch_base`` key (F5).

  Invariants enforced here (TC-1 through TC-9):
    TC-1  Session-relative range: orders[:, 6] and trades[:, 5] are < 200000 and >= 0.
          Negative assertion: values are NOT ~1.75×10⁹ (Unix epoch scale).
    TC-2  ``_parse_timestamp`` with epoch_base — known values at UTC midnight,
          +1 h, and +23:59:59 produce exact session-relative integers 0, 3600, 86399.
    TC-3  Monotonic ordering across session-relative conversion — two timestamps
          60 seconds apart produce session-relative values that differ by exactly 60,
          and this difference is preserved after float32 cast (the precision failure
          motivator).
    TC-4  float32 roundtrip: for session-relative values in [0, 86400], the float32
          roundtrip is exact.  Contrast with the broken Unix-epoch case.
    TC-5  ``_parse_timestamp`` two-argument signature — calling with one argument
          raises TypeError; calling with two works.
    TC-6  ``epoch_base`` from ``_compute_epoch_base(delivery_date)`` is in the
          plausible range for years 2020-2030 (~1577836800 to ~1893456000).
    TC-7  ``CACHE_SCHEMA_VERSION == 3`` — explicit version-3 invariant (not ">= 2"
          as in the 1.5-B audit; this is the tighter post-1.5-C assertion).
    TC-8  ``epoch_base`` stored in NPZ — after ``save_omie_cache(..., epoch_base=N)``
          + ``load_omie_cache(config)``, the returned dict contains ``"epoch_base"``
          with the correct integer value.
    TC-9  Stale v2 cache rejected — an NPZ hand-crafted with ``schema_version=2``
          is treated as a cache miss (``load_omie_cache`` returns ``None``).

  Cross-session independence (from TC-1 / TC-6 logic):
    Two configs with different ``delivery_date`` values produce independently
    correct session-relative timestamps anchored to UTC midnight of *their own*
    delivery date.  Verified via ``_compute_epoch_base`` producing distinct bases.

Out of scope (per architect F10 — do NOT test or implement here):
  - 1.5-D retired — settlement ``compute_agent_pnl`` unchanged at pair granularity.
  - Phase 2 JAX dtype conversions (int32 arrays, JIT-traceability).
  - ``OMIEDataConfig.timezone`` field (not a configuration surface).
  - Array shape changes (shapes stay (N, 13) / (M, 6)).

Runtime budget: all tests target < 1s wall-clock.  Fixtures are built in-process
via ``tmp_path`` or as pure Python computations.  No ``@pytest.mark.slow`` tests.

Reference paths:
  - omiecid_simulator/jaxomiecid/omie_data_loader.py
  - omiecid_simulator/jaxomiecid/omie_data_config.py
  - omiecid_simulator/jaxomiecid/omie_cache.py
  - docs/omiecid/market-spec.md §13.1/§13.2/§13.4/§13.5
  - docs/omiecid/migration-plan.md v4.0 §Phase 1.5
  - docs/omiecid/agile-plan.md v4.1 Story 1.5-C
"""

from __future__ import annotations

import inspect
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    _parse_timestamp,
    load_omie_orders,
    load_omie_trades,
)
from omiecid_simulator.jaxomiecid.omie_cache import (
    CACHE_SCHEMA_VERSION,
    load_omie_cache,
    save_omie_cache,
)


# ---------------------------------------------------------------------------
# Synthetic OMIE CSV fixture helpers
# (follow the same conventions as test_agente_unidad_concatenation.py)
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

_ZONE_ES = "10YES-REE------0"

# Primary test delivery date.  "2025-06-01" is the canonical date used across
# the Phase 1.5 test suite.  Its UTC midnight Unix timestamp is 1748736000.
_DATE_ISO = "2025-06-01"
_DATE_DMY = "01/06/2025"
_DATE_NODASH = "20250601"

# A second delivery date for cross-session independence tests.
_DATE2_ISO = "2025-07-15"
_DATE2_DMY = "15/07/2025"
_DATE2_NODASH = "20250715"

# Expected UTC epoch base for 2025-06-01 (seconds since 1970-01-01 UTC).
# Computed: datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()
_EPOCH_BASE_20250601: int = int(
    datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()
)


def _contract_str(contract_id: int) -> str:
    """Build OMIE contract string from a contract_id (0–95)."""
    hour = contract_id // 4
    minute = (contract_id % 4) * 15
    end_minute = minute + 15
    end_hour = hour
    if end_minute == 60:
        end_hour = hour + 1
        end_minute = 0
    return (
        f"20250602 {hour:02d}:{minute:02d}-20250602 "
        f"{end_hour:02d}:{end_minute:02d}"
    )


def _write_orders(tmp_path: Path, rows: list[str], date_nodash: str = _DATE_NODASH) -> None:
    """Write a synthetic orders file (two-line header + data rows)."""
    lines = [_ORDERS_HEADER_META, _ORDERS_HEADER_COLS]
    for r in rows:
        lines.append(r if r.endswith(";") else r + ";")
    (tmp_path / f"orders_{date_nodash}.1").write_text(
        "\n".join(lines) + "\n", encoding="latin-1"
    )


def _write_trades(tmp_path: Path, rows: list[str], date_nodash: str = _DATE_NODASH) -> None:
    """Write a synthetic trades file (two-line header + data rows)."""
    lines = [_TRADES_HEADER_META, _TRADES_HEADER_COLS]
    for r in rows:
        lines.append(r if r.endswith(";") else r + ";")
    (tmp_path / f"trades_{date_nodash}.1").write_text(
        "\n".join(lines) + "\n", encoding="latin-1"
    )


def _order_row(
    *,
    contract_id: int = 16,
    agente: str = "AGENT1",
    unidad: str = "UNIT1",
    zona: str = _ZONE_ES,
    precio: str = "50,00",
    cantidad: str = "1,00",
    side: str = "C",
    exec_r: str = "",
    validez: str = "",
    cant_red: str = "",
    ppd: str = "",
    momento: str = "01/06/2025 04:00:00",
    date_dmy: str = _DATE_DMY,
) -> str:
    return (
        f"{date_dmy};{_contract_str(contract_id)};{zona};{agente};{unidad};"
        f"{precio};{cantidad};{side};{exec_r};{validez};{cant_red};{ppd};"
        f"{momento}"
    )


def _trade_row(
    *,
    contract_id: int = 16,
    buy_agente: str = "AGENT1",
    buy_unidad: str = "UNIT1",
    sell_agente: str = "AGENT2",
    sell_unidad: str = "UNIT2",
    buy_zona: str = _ZONE_ES,
    sell_zona: str = _ZONE_ES,
    precio: str = "50,00",
    cantidad: str = "1,00",
    momento: str = "01/06/2025 17:42:00",
    date_dmy: str = _DATE_DMY,
) -> str:
    return (
        f"{date_dmy};{_contract_str(contract_id)};"
        f"{buy_agente};{buy_unidad};{buy_zona};"
        f"{sell_agente};{sell_unidad};{sell_zona};"
        f"{precio};{cantidad};{momento}"
    )


# Fixture-only product date (OMIE-TAPE-CONTRACT-DATE-1, group F1).  Every
# fixture row carries a ``Contrato`` dated 2025-06-02 (``_contract_str``),
# i.e. delivery product (2025-06-02, k).  Per market-spec v2.10 §13/§13.4 the
# written session file ``*_20250601.1`` is file D-1 of that product, and the
# stored timestamps are seconds since D-1 00:00 (= 2025-06-01 00:00).  The
# product also spans session file D (``*_20250602.1``), written header-only.
# ``_DATE_ISO`` / ``_DATE_NODASH`` / ``_EPOCH_BASE_20250601`` are unchanged:
# they name the session day, and the pure ``_parse_timestamp`` /
# ``_compute_epoch_base`` tests read them.
_PRODUCT_DATE_ISO = "2025-06-02"
_PRODUCT_SESSION_FILES_NODASH = (_DATE_NODASH, "20250602")  # (D-1, D)


def _complete_product_files(tmp_path: Path) -> None:
    """Add a header-only file for every product session file not written.

    Same header lines as ``_write_orders`` / ``_write_trades`` (unaccented
    ``Momento casacion``).  A file already written is never touched.
    """
    for date_nodash in _PRODUCT_SESSION_FILES_NODASH:
        if not (tmp_path / f"orders_{date_nodash}.1").exists():
            _write_orders(tmp_path, [], date_nodash=date_nodash)
        if not (tmp_path / f"trades_{date_nodash}.1").exists():
            _write_trades(tmp_path, [], date_nodash=date_nodash)


def _config(
    tmp_path: Path,
    contract_id: int = 16,
    delivery_date: str = _PRODUCT_DATE_ISO,
) -> OMIEDataConfig:
    _complete_product_files(tmp_path)
    return OMIEDataConfig(
        data_path=str(tmp_path),
        contract_id=contract_id,
        delivery_date=delivery_date,
    )


# ===========================================================================
# TC-1 / TC-1b  Session-relative range — orders and trades
# ===========================================================================


class TestSessionRelativeRange:
    """TC-1: Timestamps loaded by the data loader are session-relative (< 200000),
    not Unix-epoch scale (~1.75×10⁹).

    The test sets up a minimal synthetic fixture with a known submission
    time and asserts the magnitude constraint from the story acceptance
    criteria.  A negative assertion confirms the value is not in the
    Unix-epoch range.
    """

    def test_order_timestamps_are_session_relative(self, tmp_path):
        """Orders[:, 6] timestamp column values are in [0, 200000)."""
        rows = [
            _order_row(momento="01/06/2025 08:00:00"),
            _order_row(momento="01/06/2025 16:30:00"),
        ]
        _write_orders(tmp_path, rows)
        config = _config(tmp_path)

        orders = load_omie_orders(config)
        assert orders.shape[0] == 2
        timestamps = orders[:, 6]

        # Session-relative values must be small non-negative integers.
        assert float(timestamps[0]) >= 0
        assert float(timestamps[1]) >= 0
        assert float(timestamps[0]) < 200_000
        assert float(timestamps[1]) < 200_000

    def test_order_timestamps_are_not_unix_epoch_scale(self, tmp_path):
        """Orders[:, 6] are NOT in Unix epoch range (~1.75×10⁹).

        This is the explicit negative assertion: if timestamps were
        returned as absolute Unix epoch values they would be around 1.748
        billion seconds, far above the 200 000 threshold.
        """
        rows = [_order_row(momento="01/06/2025 12:00:00")]
        _write_orders(tmp_path, rows)
        config = _config(tmp_path)

        orders = load_omie_orders(config)
        ts = float(orders[0, 6])
        assert ts < 1_000_000_000, (
            f"Timestamp {ts} looks like a Unix epoch value (~1.75e9), "
            "not a session-relative value (expected < 200000)"
        )

    def test_trade_timestamps_are_session_relative(self, tmp_path):
        """Trades[:, 5] match-timestamp column values are in [0, 200000).

        TC-8 from the architecture review: the fix must be applied
        symmetrically to both orders *and* trades.
        """
        rows = [_trade_row(momento="01/06/2025 17:42:00")]
        _write_orders(tmp_path, [])
        _write_trades(tmp_path, rows)
        config = _config(tmp_path)

        trades = load_omie_trades(config)
        assert trades.shape[0] == 1
        ts = float(trades[0, 5])

        assert ts >= 0
        assert ts < 200_000

    def test_trade_timestamps_are_not_unix_epoch_scale(self, tmp_path):
        """Trades[:, 5] are NOT in Unix epoch range (~1.75×10⁹)."""
        rows = [_trade_row(momento="01/06/2025 17:42:00")]
        _write_orders(tmp_path, [])
        _write_trades(tmp_path, rows)
        config = _config(tmp_path)

        trades = load_omie_trades(config)
        ts = float(trades[0, 5])
        assert ts < 1_000_000_000, (
            f"Trade timestamp {ts} looks like a Unix epoch value, "
            "not a session-relative value"
        )


# ===========================================================================
# TC-2  _parse_timestamp with epoch_base — known-value contract
# ===========================================================================


class TestParseTimestampKnownValues:
    """TC-2: Unit-level black-box tests for ``_parse_timestamp(momento_str, epoch_base)``.

    Verifies the new two-argument signature against UTC-anchored known
    values.  epoch_base = 1748736000 is UTC midnight of 2025-06-01.
    """

    def test_utc_midnight_yields_zero(self):
        """Submission time at UTC midnight on delivery date → 0."""
        ts = _parse_timestamp("01/06/2025 00:00:00", _EPOCH_BASE_20250601)
        assert ts == 0

    def test_one_hour_past_midnight_yields_3600(self):
        """Submission time at 01:00:00 UTC → 3600 seconds."""
        ts = _parse_timestamp("01/06/2025 01:00:00", _EPOCH_BASE_20250601)
        assert ts == 3600

    def test_end_of_day_yields_86399(self):
        """Submission time at 23:59:59 UTC → 86399 seconds."""
        ts = _parse_timestamp("01/06/2025 23:59:59", _EPOCH_BASE_20250601)
        assert ts == 86399

    def test_midnight_crossing_lower_bound(self):
        """TC-9 (from arch review): 23:59:00 on delivery date → near 86340.

        Confirms the UTC-midnight anchor: a timestamp very close to midnight
        at end-of-day produces a value near 86400, positive and < 86400.
        """
        ts = _parse_timestamp("01/06/2025 23:59:00", _EPOCH_BASE_20250601)
        assert ts == 86340
        assert 0 < ts < 86400

    def test_result_is_numeric(self):
        """``_parse_timestamp`` returns a numeric (int or float) value."""
        ts = _parse_timestamp("01/06/2025 12:00:00", _EPOCH_BASE_20250601)
        assert isinstance(ts, (int, float))


# ===========================================================================
# TC-3  Monotonic ordering preserved across session-relative conversion
# ===========================================================================


class TestMonotonicOrderingPreserved:
    """TC-3: Two orders submitted at different times preserve t1 < t2 in the
    loaded array, and the 60-second difference is represented exactly in
    the session-relative domain — both as Python integers/floats and after
    float32 cast.

    This is the core motivating test: Unix-epoch-scale float32 would
    collapse two timestamps 60 seconds apart to the same value (ULP ≈ 128 s);
    session-relative float32 preserves it exactly.
    """

    def test_earlier_submission_gets_smaller_timestamp(self, tmp_path):
        """t1 < t2 in submission time → loaded array column 6 has t1 < t2."""
        rows = [
            _order_row(agente="AGENT1", unidad="UNIT1", momento="01/06/2025 10:00:00"),
            _order_row(agente="AGENT2", unidad="UNIT2", momento="01/06/2025 10:01:00"),
        ]
        _write_orders(tmp_path, rows)
        config = _config(tmp_path)

        orders = load_omie_orders(config)
        assert orders.shape[0] == 2
        t1 = float(orders[0, 6])
        t2 = float(orders[1, 6])
        assert t1 < t2, f"Expected t1={t1} < t2={t2}"

    def test_60_second_gap_preserved_in_session_relative_values(self, tmp_path):
        """Two timestamps 60 s apart differ by exactly 60 in session-relative domain."""
        rows = [
            _order_row(agente="AGENT1", unidad="UNIT1", momento="01/06/2025 10:00:00"),
            _order_row(agente="AGENT2", unidad="UNIT2", momento="01/06/2025 10:01:00"),
        ]
        _write_orders(tmp_path, rows)
        config = _config(tmp_path)

        orders = load_omie_orders(config)
        t1 = float(orders[0, 6])
        t2 = float(orders[1, 6])
        assert t2 - t1 == 60, f"Expected difference 60, got {t2 - t1}"


# ===========================================================================
# TC-4  float32 roundtrip exactness
# ===========================================================================


class TestFloat32Roundtrip:
    """TC-4: Session-relative integers up to 86400 are exactly representable
    as float32; a 60-second gap is not lost in the float32 cast.

    This directly verifies the acceptance criterion:
    "float32 roundtrip preserves exact value (session-relative integers
    up to 86400 are exactly representable)".

    Contrast: at Unix epoch scale (~1.748e9), float32 ULP ≈ 128 s, so
    ``np.float32(1748736060) - np.float32(1748736000) == 0.0``.
    """

    def test_session_relative_60_second_gap_preserved_in_float32(self):
        """float32(t + 60) - float32(t) == 60 for session-relative t in [0, 86400]."""
        # Choose a representative session-relative base value.
        session_base = 36000  # 10:00:00 UTC = 36000 s past midnight

        f32_t = np.float32(session_base)
        f32_t60 = np.float32(session_base + 60)
        diff = float(f32_t60) - float(f32_t)
        assert diff == 60.0, (
            f"float32 gap at session_base={session_base}: expected 60, got {diff}"
        )

    def test_unix_epoch_60_second_gap_is_lost_in_float32(self):
        """Confirm the motivating failure: Unix epoch scale loses 60 s in float32.

        This test documents the problem the story is designed to fix.
        If this assertion ever fails it means float32 precision improved
        at epoch scale, which would make the fix still-correct but the
        original motivation moot.
        """
        # epoch = datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()
        epoch_base = float(_EPOCH_BASE_20250601)

        f32_t = np.float32(epoch_base)
        f32_t60 = np.float32(epoch_base + 60)
        diff = float(f32_t60) - float(f32_t)
        # At epoch scale the 60-second gap collapses to 0 (ULP ≈ 128 s).
        assert diff == 0.0, (
            "Expected float32 precision loss at Unix epoch scale "
            f"(ULP ≈ 128 s at ~1.75e9), but got diff={diff}. "
            "The motivating failure may be machine-dependent."
        )

    def test_all_intraday_integers_roundtrip_exactly_in_float32(self):
        """For all second offsets in [0, 86400], float32 roundtrip is exact."""
        # Sample a dense grid: every 60 s over [0, 86400].
        offsets = np.arange(0, 86401, 60, dtype=np.int32)
        roundtrip = np.array(offsets, dtype=np.float32).astype(np.int32)
        assert np.all(roundtrip == offsets), (
            "Some session-relative offsets in [0, 86400] are not exactly "
            "representable as float32 integers."
        )


# ===========================================================================
# TC-5  _parse_timestamp two-argument signature contract
# ===========================================================================


class TestParseTimestampSignature:
    """TC-5: ``_parse_timestamp`` must accept exactly ``(momento_str, epoch_base)``
    and raise TypeError when called with only one argument (the old signature).

    This is the hard contract break from the architecture review (F1):
    the old ``_parse_timestamp("DD/MM/YYYY HH:MM:SS")`` single-argument
    call is illegal after 1.5-C.
    """

    def test_two_argument_call_succeeds(self):
        """``_parse_timestamp(momento_str, epoch_base)`` does not raise."""
        ts = _parse_timestamp("01/06/2025 12:00:00", _EPOCH_BASE_20250601)
        # Value not under test here; just assert it returned without error.
        assert ts is not None

    def test_one_argument_call_raises_type_error(self):
        """Calling with a single argument must raise TypeError.

        This assertion locks the new two-argument contract: any call site
        left over from Phase 1 / 1.5-A that passes only the datetime string
        must be updated to also pass ``epoch_base``.
        """
        with pytest.raises(TypeError):
            _parse_timestamp("01/06/2025 12:00:00")  # type: ignore[call-arg]

    def test_signature_has_two_parameters(self):
        """Inspect the live signature to confirm exactly two positional parameters."""
        sig = inspect.signature(_parse_timestamp)
        params = list(sig.parameters)
        assert len(params) == 2, (
            f"Expected 2 parameters, got {len(params)}: {params}"
        )


# ===========================================================================
# TC-6  epoch_base from _compute_epoch_base is plausible
# ===========================================================================


class TestComputeEpochBase:
    """TC-6: ``_compute_epoch_base(delivery_date)`` returns a Unix timestamp
    in the plausible range for years 2020–2030.

    Import is attempted lazily so that collection does not fail if the
    function does not yet exist (pre-implementation); the test itself
    will fail with ImportError if missing.
    """

    def test_epoch_base_is_in_year_range_2020_to_2030(self):
        """_compute_epoch_base('2025-06-01') is in [2020-01-01, 2030-01-01] epoch range."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import _compute_epoch_base  # noqa

        base = _compute_epoch_base("2025-06-01")

        low = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
        high = int(datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp())
        assert low <= base <= high, (
            f"epoch_base={base} is outside the 2020–2030 range [{low}, {high}]"
        )

    def test_epoch_base_matches_utc_midnight(self):
        """_compute_epoch_base('2025-06-01') equals the UTC midnight of that date."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import _compute_epoch_base  # noqa

        base = _compute_epoch_base("2025-06-01")
        expected = int(datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp())
        assert base == expected, (
            f"epoch_base={base} does not equal UTC midnight 2025-06-01 ({expected})"
        )

    def test_different_delivery_dates_produce_different_bases(self):
        """Two different delivery dates produce distinct epoch_base values."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import _compute_epoch_base  # noqa

        base1 = _compute_epoch_base("2025-06-01")
        base2 = _compute_epoch_base("2025-07-15")
        assert base1 != base2, (
            "Different delivery dates must produce distinct epoch_base values"
        )

    def test_epoch_base_advances_by_one_day(self):
        """epoch_base for consecutive days differs by exactly 86400 seconds."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import _compute_epoch_base  # noqa

        base1 = _compute_epoch_base("2025-06-01")
        base2 = _compute_epoch_base("2025-06-02")
        assert base2 - base1 == 86400, (
            f"Consecutive delivery dates: expected base difference 86400, "
            f"got {base2 - base1}"
        )


# ===========================================================================
# TC-7  CACHE_SCHEMA_VERSION == 3
# ===========================================================================


class TestCacheSchemaVersion:
    """TC-7: After Story 1.5-C, ``CACHE_SCHEMA_VERSION`` must equal 3.

    This is a concrete post-1.5-C invariant (the 1.5-B audit test asserts
    >= 2, which is intentionally future-compatible; this locked test is
    specific to version 3).

    Refs: sprint-status.md §1.5-C F5 and TC-6 (architecture review numbering).
    """

    def test_cache_schema_version_equals_3(self):
        """CACHE_SCHEMA_VERSION bumped to 4 by Story P2-F0 (Category B)."""
        assert CACHE_SCHEMA_VERSION == 4, (
            f"Expected CACHE_SCHEMA_VERSION == 4 after Story P2-F0, "
            f"got {CACHE_SCHEMA_VERSION}"
        )


# ===========================================================================
# TC-8  epoch_base stored in NPZ cache
# ===========================================================================


class TestEpochBaseInCache:
    """TC-8: ``save_omie_cache(..., epoch_base=N)`` stores the value in the NPZ,
    and ``load_omie_cache`` returns it as key ``'epoch_base'``.

    Ref: sprint-status.md §1.5-C F5 — epoch_base must be stored so a future
    reader can recover absolute timestamps and so the cache is self-describing.
    """

    def test_epoch_base_round_trips_through_cache(self, tmp_path):
        """epoch_base saved to cache is retrieved unchanged by load_omie_cache."""
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = np.zeros((2, 13), dtype=np.float32)
        trades = np.zeros((1, 6), dtype=np.float32)
        epoch_base = _EPOCH_BASE_20250601

        save_omie_cache(config, orders=orders, trades=trades, epoch_base=epoch_base)
        result = load_omie_cache(config)

        assert result is not None, "Cache should be loadable after save"
        assert "epoch_base" in result, (
            "Cache payload must contain 'epoch_base' key after Story 1.5-C"
        )
        assert int(result["epoch_base"]) == epoch_base, (
            f"epoch_base mismatch: expected {epoch_base}, "
            f"got {int(result['epoch_base'])}"
        )

    def test_cache_orders_and_trades_still_present_after_epoch_base_added(self, tmp_path):
        """Adding epoch_base to cache does not displace orders or trades keys."""
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        orders = np.array([[1.0] * 13], dtype=np.float32)
        trades = np.array([[2.0] * 6], dtype=np.float32)
        epoch_base = _EPOCH_BASE_20250601

        save_omie_cache(config, orders=orders, trades=trades, epoch_base=epoch_base)
        result = load_omie_cache(config)

        assert result is not None
        assert "orders" in result
        assert "trades" in result
        assert result["orders"].shape == (1, 13)
        assert result["trades"].shape == (1, 6)


# ===========================================================================
# TC-9  Stale v2 cache rejected
# ===========================================================================


class TestStaleCacheRejected:
    """TC-9: A cache file written with ``schema_version=2`` (the 1.5-B schema)
    is treated as a cache miss (``load_omie_cache`` returns ``None``).

    This enforces the version bump: after 1.5-C changes timestamp semantics,
    any existing v2 NPZ on disk would silently return epoch-scale timestamps
    if the version check were absent.

    The test hand-crafts an NPZ with schema_version=2 (mimicking what the
    1.5-B cache writer would have produced) and asserts rejection.
    """

    def _write_v2_cache(self, tmp_path: Path, config: OMIEDataConfig) -> None:
        """Write a hand-crafted NPZ with schema_version=2 at the expected path."""
        from omiecid_simulator.jaxomiecid.omie_cache import _cache_filename  # noqa

        path = _cache_filename(config)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.array(2, dtype=np.int32),
            orders=np.zeros((1, 13), dtype=np.float32),
            trades=np.zeros((1, 6), dtype=np.float32),
        )

    def test_v2_cache_returns_none(self, tmp_path):
        """load_omie_cache returns None for a v2 cache after CACHE_SCHEMA_VERSION=3."""
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        self._write_v2_cache(tmp_path, config)

        result = load_omie_cache(config)
        assert result is None, (
            "Expected load_omie_cache to return None for a v2 cache "
            f"(CACHE_SCHEMA_VERSION={CACHE_SCHEMA_VERSION}), but got a result."
        )

    def test_v2_cache_rejected_not_raising(self, tmp_path):
        """Version mismatch must be handled silently (None), not with an exception."""
        config = OMIEDataConfig(
            data_path=str(tmp_path),
            contract_id=16,
            delivery_date=_DATE_ISO,
            cache_path=str(tmp_path / "cache"),
        )
        self._write_v2_cache(tmp_path, config)

        # Must not raise any exception — just return None.
        try:
            result = load_omie_cache(config)
        except Exception as exc:
            pytest.fail(
                f"load_omie_cache raised {type(exc).__name__} on a v2 cache; "
                "expected None (silent miss), not an exception."
            )
        assert result is None


# ===========================================================================
# Cross-session independence
# ===========================================================================


class TestCrossSessionIndependence:
    """Two configs with different delivery_date values produce independently
    correct session-relative timestamps anchored to UTC midnight of *their
    own* session date.

    Derived from TC-1 / TC-6 combination in the architecture review.
    """

    def test_same_wall_clock_time_different_sessions_produce_same_offset(
        self, tmp_path
    ):
        """The same wall-clock time on two different days yields the same
        session-relative offset (both are measured from their respective
        UTC midnights), assuming the time part is identical.

        Concretely: '01/06/2025 08:00:00' in session 2025-06-01 and
        '15/07/2025 08:00:00' in session 2025-07-15 should both produce
        offset 28800 (= 8 * 3600).
        """
        from omiecid_simulator.jaxomiecid.omie_data_loader import _compute_epoch_base  # noqa

        # Session A: 2025-06-01
        epoch_a = _compute_epoch_base("2025-06-01")
        ts_a = _parse_timestamp("01/06/2025 08:00:00", epoch_a)

        # Session B: 2025-07-15
        epoch_b = _compute_epoch_base("2025-07-15")
        ts_b = _parse_timestamp("15/07/2025 08:00:00", epoch_b)

        assert ts_a == ts_b == 8 * 3600, (
            f"Expected both offsets to be {8*3600} (08:00 UTC), "
            f"got ts_a={ts_a}, ts_b={ts_b}"
        )

    def test_epoch_bases_for_two_sessions_differ_by_full_days(self):
        """epoch_base for 2025-06-01 and 2025-07-15 differ by exactly 44 days."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import _compute_epoch_base  # noqa

        base_a = _compute_epoch_base("2025-06-01")
        base_b = _compute_epoch_base("2025-07-15")
        # 2025-06-01 to 2025-07-15 = 44 days
        assert base_b - base_a == 44 * 86400, (
            f"Expected {44 * 86400} s difference between 2025-06-01 and "
            f"2025-07-15 epoch bases, got {base_b - base_a}"
        )
