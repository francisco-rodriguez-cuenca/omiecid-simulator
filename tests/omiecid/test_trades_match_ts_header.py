"""Locked behavioural contract tests for OMIE-TRADES-MATCH-TS-HEADER-1.

Real OMIE trades files spell the match-time header ``Momento casación``
(accented, Latin-1 byte 0xF3). The trades loader must read that column, so
trades column [5] (match timestamp) carries the real match time, in seconds
since D-1 00:00 local wall clock with no modulo (the L1 frame of
OMIE-TAPE-CONTRACT-DATE-1).

Sources:
  * docs/omiecid/agile-plan.md, card "Story OMIE-TRADES-MATCH-TS-HEADER-1"
    (v5.170): Step-2 decisions H-1..H-8 + H-4a, Test-first contract
    T-1..T-8 with the fresh-eyes amendments F-a..F-d.
  * docs/omiecid/market-spec.md §13.2 / §13.2.1 (v2.12): header spelling,
    date-only midnight convention, frame [55,200, 165,600).

Public contract pinned here (the Implementation Agent codes against this):

  * Header names of trades files are accent-normalised, so ``Momento
    casación`` and ``Momento casacion`` are the same column (H-1). Cell
    VALUES are never normalised (T-8).
  * ``load_omie_trades`` column [5] = seconds since D-1 00:00 of the
    ``Momento casacion`` cell (``DD/MM/YYYY HH:MM:SS``), rows in file order
    (file D-1 rows first, then file D rows; no sort). Columns [0..4] are
    unchanged by this story (H-6).
  * A date-only cell ``DD/MM/YYYY`` whose date equals the delivery date D
    means D 00:00, i.e. exactly 86,400 (H-4). A date-only cell for any other
    date raises (H-4a).
  * ``OMIETimestampParseError`` is a NEW module-level ``ValueError``
    subclass in ``omiecid_simulator.jaxomiecid.omie_data_loader`` (H-3). It is
    distinct from ``OMIENumberParseError``. ``load_omie_trades`` raises it
    (exact type, not wrapped) for an empty match-time cell, an unparseable
    one, a trades file without the match-time column, and a non-D date-only
    cell. The message names the column ``'Momento casacion'`` and the raw
    value. There is no silent 0 default.
  * Scope (H-3, T-5(e)): the raise lives in ``load_omie_trades`` only;
    ``load_omie_orders``, ``load_omie_agent_metadata`` and
    ``build_fixed_tape_set`` never parse the match time and keep loading.

``OMIETimestampParseError`` is looked up with ``getattr`` inside each test
(F-a), so this module collects on the unfixed loader.

Expected status on the CURRENT (unfixed) loader:
  TestErrorClass::*                                                   RED
  TestAccentedHeader::test_accented_header_exact_seconds (T-2)        RED
  TestAccentedHeader::test_mixed_header_spellings_across_files        RED
  TestAccentedHeader::test_accented_and_unaccented_arrays_identical  RED (T-3 equality)
  TestAccentedHeader::test_unaccented_header_exact_seconds            GREEN (guard, T-3)
  TestDateOnlyMidnight::test_date_only_cell_equal_to_d_is_86400 (T-4) RED
  TestDateOnlyMidnight::test_date_only_cell_not_equal_to_d_raises[*]  RED (T-4(b), 3 cases)
  TestMatchTimeRaise::test_empty_cell_raises (T-5(a))                 RED
  TestMatchTimeRaise::test_garbage_cell_raises[*] (T-5(b))            RED (2 cases)
  TestMatchTimeRaise::test_missing_column_raises (T-5(c))             RED
  TestMatchTimeRaise::test_column_missing_in_file_d_only_raises (T-5(d)) RED
  TestScopeGuard::test_orders_metadata_and_builder_still_load (T-5(e)) GREEN (guard)
  TestCellValuesUntouched::* (T-8)                                    GREEN (guard)
  TestRealProduct20250625::test_match_times_in_frame_and_non_zero (T-1)   RED
  TestRealProduct20250625::test_match_times_equal_independent_reference   RED
  TestRealProduct20250625::test_columns_0_to_4_unchanged (T-6)        GREEN (guard)
  TestRealProduct20250625::test_orders_array_unchanged (T-7)          GREEN (guard)
  TestRealProduct20250610K18::test_loads_in_frame_with_one_midnight (T-4 real) RED
  TestRealProduct20250610K18::test_match_times_equal_independent_reference RED

Runtime: synthetic tests write small tmp_path CSVs (< 1 s each). The
real-product tests are ``@pytest.mark.slow`` (each product parses two real
30-45 MB orders session files plus two trades files) and share one
module-scoped load per product. They skip only when ``data/rawOMIE`` is
absent.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import omiecid_simulator.jaxomiecid.omie_data_loader as odl
from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    build_fixed_tape_set,
    load_omie_agent_metadata,
    load_omie_orders,
    load_omie_trades,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRD_BUY_AGENT, TRD_SELL_AGENT, TRD_MATCH_TS = 3, 4, 5

FRAME_LO, FRAME_HI = 55_200, 165_600  # [15:20 D-1, 22:00 D), seconds since D-1 00:00
MIDNIGHT_D = 86_400

COLUMN = "Momento casacion"  # canonical (ASCII) column name in the message

_REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA = _REPO_ROOT / "data" / "rawOMIE"

# Synthetic product (D, K): delivery 2025-06-02, slot 40 (10:00-10:15).
# Session files: D-1 = 2025-06-01, D = 2025-06-02 (the product's two files).
D = "2025-06-02"
D_PREV = "2025-06-01"
K = 40
CONTRATO = "20250602 10:00-20250602 10:15"
OTHER_CONTRATO = "20250602 10:15-20250602 10:30"  # slot 41: not the product
_ZONE = "10YES-REE------0"

_TRADES_META = (
    "OMIE - Mercado de electricidad;Fecha Emisión :02/06/2025 - 00:45;;"
    "{fecha};Transacciones realizadas en XBID;;;;"
)
_TRADES_COLS_ACCENTED = (
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;"
    "Agente venta;Unidad venta;Zona venta;Precio;Cantidad;Momento casación;"
)
_TRADES_COLS_ASCII = (
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;"
    "Agente venta;Unidad venta;Zona venta;Precio;Cantidad;Momento casacion;"
)
_TRADES_COLS_NO_TS = (
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;"
    "Agente venta;Unidad venta;Zona venta;Precio;Cantidad;"
)
_ORDERS_META = (
    "OMIE - Mercado de electricidad;Fecha Emisión :02/06/2025 - 00:45;;"
    "{fecha};Órdenes enviadas a XBID;;;;"
)
_ORDERS_COLS = (
    "Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;Tipo oferta;"
    "Condiciones a la ejecución;Condiciones a la validez;"
    "Cantidad reducida;PPD;Momento de envío;"
)

# Three well-formed product rows (T-2): one in file D-1, two in file D.
# Expected column [5] = seconds since 2025-06-01 00:00.
#   file D-1: 01/06/2025 16:30:15 -> 16*3600 + 30*60 + 15          =  59,415
#   file D  : 01/06/2025 22:30:15 -> 22*3600 + 30*60 + 15          =  81,015
#   file D  : 02/06/2025 08:05:07 -> 86,400 + 8*3600 + 5*60 + 7    = 115,507
_ROWS_PREV = ["01/06/2025 16:30:15"]
_ROWS_D = ["01/06/2025 22:30:15", "02/06/2025 08:05:07"]
_EXPECTED_SECONDS = [59_415, 81_015, 115_507]


# ---------------------------------------------------------------------------
# Accessor for the name the fix introduces (F-a)
# ---------------------------------------------------------------------------

def _err_cls():
    cls = getattr(odl, "OMIETimestampParseError", None)
    if cls is None:
        pytest.fail(
            "omie_data_loader.OMIETimestampParseError is missing "
            "(card H-3: a named, module-level ValueError subclass)"
        )
    assert isinstance(cls, type) and issubclass(cls, ValueError), (
        "OMIETimestampParseError must subclass ValueError"
    )
    return cls


def _cfg(data_path, contract_id=K, delivery_date=D):
    return OMIEDataConfig(
        data_path=str(data_path),
        contract_id=int(contract_id),
        delivery_date=delivery_date,
        cache_path=str(data_path),
    )


# ---------------------------------------------------------------------------
# Synthetic session-file writers. Files are written in Latin-1 with CRLF line
# ends, row 1 metadata, row 2 blank, row 3 header (market-spec v2.12 §13),
# exactly like the real files. Cells are written VERBATIM.
# ---------------------------------------------------------------------------

def _trade_line(momento, *, contrato=CONTRATO, buy=("BYR", "BU1"),
                sell=("SLR", "SU1"), precio="50,00", cantidad="1,00",
                fecha="02/06/2025", with_ts=True):
    cells = [fecha, contrato, buy[0], buy[1], _ZONE, sell[0], sell[1], _ZONE,
             precio, cantidad]
    if with_ts:
        cells.append(momento)
    return ";".join(cells) + ";"


def _order_line(fecha="02/06/2025", send="02/06/2025 08:00:00"):
    return (
        f"{fecha};{CONTRATO};{_ZONE};ABC;U1;50,00;1,0;V;;;;;{send};"
    )


def _write_file(path, meta, header, lines):
    body = [meta, "", header] + list(lines) + [";;;;;;;;;;;"]
    Path(path).write_bytes("\r\n".join(body).encode("latin-1") + b"\r\n")


def _write(dirpath, *, trades_prev, trades_d, header_prev=_TRADES_COLS_ACCENTED,
           header_d=_TRADES_COLS_ACCENTED):
    """Write a complete product (orders + trades files D-1 and D).

    ``trades_prev`` / ``trades_d`` are lists of raw trade lines for file D-1
    and file D. Each orders file carries one well-formed product order, so
    the orders tape is non-empty.
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    for iso, fecha, t_lines, t_header, send in (
        (D_PREV, "01/06/2025", trades_prev, header_prev, "01/06/2025 16:00:00"),
        (D, "02/06/2025", trades_d, header_d, "02/06/2025 08:00:00"),
    ):
        nd = iso.replace("-", "")
        _write_file(dirpath / f"orders_{nd}.1", _ORDERS_META.format(fecha=fecha),
                    _ORDERS_COLS, [_order_line(fecha=fecha, send=send)])
        _write_file(dirpath / f"trades_{nd}.1", _TRADES_META.format(fecha=fecha),
                    t_header, t_lines)
    return dirpath


def _write_standard(dirpath, header_prev=_TRADES_COLS_ACCENTED,
                    header_d=_TRADES_COLS_ACCENTED):
    """The T-2 product: 3 product rows + one non-product row per file."""
    prev = [_trade_line(m, fecha="01/06/2025") for m in _ROWS_PREV]
    prev.append(_trade_line("01/06/2025 17:00:00", contrato=OTHER_CONTRATO,
                            fecha="01/06/2025"))
    d = [_trade_line(m) for m in _ROWS_D]
    d.insert(1, _trade_line("02/06/2025 07:00:00", contrato=OTHER_CONTRATO))
    return _write(dirpath, trades_prev=prev, trades_d=d,
                  header_prev=header_prev, header_d=header_d)


def _write_one_bad_cell(dirpath, momento):
    """A product whose file-D product row carries match-time cell ``momento``.

    File D-1 carries one well-formed product row, so the bad row is not the
    only row of the product.
    """
    return _write(
        dirpath,
        trades_prev=[_trade_line(_ROWS_PREV[0], fecha="01/06/2025")],
        trades_d=[_trade_line(momento)],
    )


def _ts(dirpath):
    return np.asarray(load_omie_trades(_cfg(dirpath)))[:, TRD_MATCH_TS].tolist()


def _assert_raises_timestamp_error(dirpath, *, raw=None):
    """``load_omie_trades`` raises exactly ``OMIETimestampParseError``.

    The message must contain the column name, and ``raw`` when given.
    """
    # Load first, so on the unfixed loader the failure is the defect itself
    # ("DID NOT RAISE": the silent 0), not the missing class name.
    with pytest.raises(ValueError, match=re.escape(COLUMN)) as info:
        load_omie_trades(_cfg(dirpath))
    err = _err_cls()
    assert type(info.value) is err, (
        f"expected exactly OMIETimestampParseError, got "
        f"{type(info.value).__name__}: {info.value}"
    )
    if raw is not None:
        assert raw in str(info.value), (
            f"message must contain the raw value {raw!r}: {info.value}"
        )
    return info.value


# ---------------------------------------------------------------------------
# H-3: the error class
# ---------------------------------------------------------------------------

class TestErrorClass:
    def test_omie_timestamp_parse_error_is_a_value_error(self):
        cls = _err_cls()
        assert issubclass(cls, ValueError)

    def test_omie_timestamp_parse_error_is_not_the_number_error(self):
        """H-3: a timestamp is not a numeric cell; the classes are distinct."""
        cls = _err_cls()
        num = getattr(odl, "OMIENumberParseError", None)
        assert num is not None
        assert cls is not num
        assert not issubclass(cls, num)
        assert not issubclass(num, cls)


# ---------------------------------------------------------------------------
# T-2 / T-3: accented header (Latin-1) vs unaccented header
# ---------------------------------------------------------------------------

class TestAccentedHeader:
    def test_accented_header_exact_seconds(self, tmp_path):
        """T-2: Latin-1 ``Momento casación`` header -> exact seconds per row,
        file D-1 row first, then the file-D rows in file order."""
        _write_standard(tmp_path)
        ts = _ts(tmp_path)
        assert ts == _EXPECTED_SECONDS
        assert all(FRAME_LO <= t < FRAME_HI for t in ts)

    def test_mixed_header_spellings_across_files(self, tmp_path):
        """H-1: normalisation is per file; file D-1 accented + file D ASCII
        (and the reverse) read the same column."""
        _write_standard(tmp_path / "a", header_prev=_TRADES_COLS_ACCENTED,
                        header_d=_TRADES_COLS_ASCII)
        _write_standard(tmp_path / "b", header_prev=_TRADES_COLS_ASCII,
                        header_d=_TRADES_COLS_ACCENTED)
        assert _ts(tmp_path / "a") == _EXPECTED_SECONDS
        assert _ts(tmp_path / "b") == _EXPECTED_SECONDS

    def test_unaccented_header_exact_seconds(self, tmp_path):
        """GUARD (T-3): the ASCII header fixture already loads exactly."""
        _write_standard(tmp_path, header_prev=_TRADES_COLS_ASCII,
                        header_d=_TRADES_COLS_ASCII)
        assert _ts(tmp_path) == _EXPECTED_SECONDS

    def test_accented_and_unaccented_arrays_identical(self, tmp_path):
        """T-3: the same rows under the two spellings give an identical
        (M, 6) array (all columns, dtype, shape). RED on the unfixed loader
        (the accented side has column [5] = 0)."""
        _write_standard(tmp_path / "ascii", header_prev=_TRADES_COLS_ASCII,
                        header_d=_TRADES_COLS_ASCII)
        ascii_arr = np.asarray(load_omie_trades(_cfg(tmp_path / "ascii")))
        assert ascii_arr[:, TRD_MATCH_TS].tolist() == _EXPECTED_SECONDS
        _write_standard(tmp_path / "acc")
        acc_arr = np.asarray(load_omie_trades(_cfg(tmp_path / "acc")))
        assert acc_arr.dtype == ascii_arr.dtype
        assert acc_arr.shape == ascii_arr.shape
        assert np.array_equal(acc_arr, ascii_arr)


# ---------------------------------------------------------------------------
# T-4 / T-4(b): date-only match time (H-4 / H-4a)
# ---------------------------------------------------------------------------

class TestDateOnlyMidnight:
    def test_date_only_cell_equal_to_d_is_86400(self, tmp_path):
        """T-4 (H-4): ``02/06/2025`` (= D) in file D is D 00:00 = 86,400,
        between a 23:59:59 D-1 row and a 00:00:10 D row (real shape)."""
        _write(
            tmp_path,
            trades_prev=[_trade_line(_ROWS_PREV[0], fecha="01/06/2025")],
            trades_d=[
                _trade_line("01/06/2025 23:59:59"),
                _trade_line("02/06/2025"),
                _trade_line("02/06/2025 00:00:10"),
            ],
        )
        assert _ts(tmp_path) == [59_415, 86_399, MIDNIGHT_D, 86_410]

    @pytest.mark.parametrize(
        "file_side, cell",
        [
            ("prev", "01/06/2025"),  # date-only D-1 in file D-1 (would be 0)
            ("d", "01/06/2025"),     # date-only D-1 in file D (would be 0)
            ("d", "03/06/2025"),     # date-only D+1 in file D (would be 172,800)
        ],
        ids=["d_minus_1_in_file_d_minus_1", "d_minus_1_in_file_d", "d_plus_1_in_file_d"],
    )
    def test_date_only_cell_not_equal_to_d_raises(self, tmp_path, file_side, cell):
        """T-4(b) (H-4a): a date-only cell is accepted only for date D."""
        good_prev = _trade_line(_ROWS_PREV[0], fecha="01/06/2025")
        good_d = _trade_line(_ROWS_D[1])
        if file_side == "prev":
            prev = [good_prev, _trade_line(cell, fecha="01/06/2025")]
            d = [good_d]
        else:
            prev = [good_prev]
            d = [good_d, _trade_line(cell)]
        _write(tmp_path, trades_prev=prev, trades_d=d)
        _assert_raises_timestamp_error(tmp_path, raw=cell)


# ---------------------------------------------------------------------------
# T-5(a-d): load_omie_trades raises OMIETimestampParseError (H-3)
# ---------------------------------------------------------------------------

class TestMatchTimeRaise:
    def test_empty_cell_raises(self, tmp_path):
        """T-5(a): an empty match-time cell raises (no silent 0).

        ``match=`` on the column name only: the raw value of an empty cell
        may reach the message as ``''`` or ``nan`` (not pinned).
        """
        _write_one_bad_cell(tmp_path, "")
        _assert_raises_timestamp_error(tmp_path)

    @pytest.mark.parametrize(
        "raw",
        ["abc", "02/06/2025 25:61:00"],
        ids=["abc", "out_of_range_time"],
    )
    def test_garbage_cell_raises(self, tmp_path, raw):
        """T-5(b): an unparseable cell raises; the message names the column,
        the raw value and the data path (H-3 source)."""
        _write_one_bad_cell(tmp_path, raw)
        exc = _assert_raises_timestamp_error(tmp_path, raw=raw)
        assert str(tmp_path) in str(exc), (
            f"message must name the source data_path {str(tmp_path)!r}: {exc}"
        )

    def test_missing_column_raises(self, tmp_path):
        """T-5(c): trades files without the match-time column raise
        (column name only; the raw value is empty)."""
        _write(
            tmp_path,
            trades_prev=[_trade_line(None, fecha="01/06/2025", with_ts=False)],
            trades_d=[_trade_line(None, with_ts=False)],
            header_prev=_TRADES_COLS_NO_TS,
            header_d=_TRADES_COLS_NO_TS,
        )
        _assert_raises_timestamp_error(tmp_path)

    def test_column_missing_in_file_d_only_raises(self, tmp_path):
        """T-5(d): file D-1 has the column, file D omits it (the concat
        leaves the D rows empty) -> raises."""
        _write(
            tmp_path,
            trades_prev=[_trade_line(_ROWS_PREV[0], fecha="01/06/2025")],
            trades_d=[_trade_line(None, with_ts=False)],
            header_prev=_TRADES_COLS_ACCENTED,
            header_d=_TRADES_COLS_NO_TS,
        )
        _assert_raises_timestamp_error(tmp_path)


# ---------------------------------------------------------------------------
# T-5(e): scope guard -- the raise lives in load_omie_trades only
# ---------------------------------------------------------------------------

class TestScopeGuard:
    def test_orders_metadata_and_builder_still_load(self, tmp_path):
        """GUARD (T-5(e)): the T-5(b) ``abc`` fixture (on which
        ``load_omie_trades`` raises, see TestMatchTimeRaise) still loads
        through the orders loader, the agent metadata and the tape builder."""
        _write_one_bad_cell(tmp_path, "abc")
        cfg = _cfg(tmp_path)
        orders = np.asarray(load_omie_orders(cfg))
        assert orders.shape == (2, 13)
        meta = load_omie_agent_metadata(cfg)
        assert ("BYR", "BU1") in meta.values()
        assert ("SLR", "SU1") in meta.values()
        tapes = np.asarray(build_fixed_tape_set([cfg]))
        assert tapes.ndim == 3 and tapes.shape[0] == 1 and tapes.shape[2] == 13


# ---------------------------------------------------------------------------
# T-8: header normalisation never touches cell values (H-1)
# ---------------------------------------------------------------------------

class TestCellValuesUntouched:
    @pytest.fixture
    def accented_pair_product(self, tmp_path):
        _write(
            tmp_path,
            trades_prev=[_trade_line(_ROWS_PREV[0], fecha="01/06/2025")],
            trades_d=[_trade_line(_ROWS_D[1], buy=("AGÉNTE", "UNÍDAD"))],
        )
        return tmp_path

    def test_agent_metadata_keeps_accented_pair_verbatim(self, accented_pair_product):
        """GUARD (T-8): the trade-only pair stays ``("AGÉNTE", "UNÍDAD")``."""
        pairs = set(load_omie_agent_metadata(_cfg(accented_pair_product)).values())
        assert ("AGÉNTE", "UNÍDAD") in pairs
        assert ("AGENTE", "UNIDAD") not in pairs

    def test_trades_buy_agent_maps_to_accented_pair(self, accented_pair_product):
        """GUARD (T-8): the Contrato filter selects both product rows and
        column [3] of the file-D row is the accented pair's id."""
        cfg = _cfg(accented_pair_product)
        id_to_pair = load_omie_agent_metadata(cfg)
        pair_to_id = {v: k for k, v in id_to_pair.items()}
        trades = np.asarray(load_omie_trades(cfg))
        assert trades.shape[0] == 2
        assert int(trades[1, TRD_BUY_AGENT]) == pair_to_id[("AGÉNTE", "UNÍDAD")]
        assert int(trades[0, TRD_BUY_AGENT]) == pair_to_id[("BYR", "BU1")]


# ---------------------------------------------------------------------------
# Real data: independent reference parse of the raw trades files
# ---------------------------------------------------------------------------

_DATE_ONLY_RE = re.compile(r"\d{2}/\d{2}/\d{4}")


def _have_real(delivery_date):
    d = date.fromisoformat(delivery_date)
    names = []
    for day in (d - timedelta(days=1), d):
        nd = day.strftime("%Y%m%d")
        names += [f"orders_{nd}.1", f"trades_{nd}.1"]
    return REAL_DATA.is_dir() and all((REAL_DATA / n).is_file() for n in names)


def _reference_match_seconds(delivery_date, contract_id):
    """Independent reference: column [5] of product (D, k), file order.

    Reads ``trades_{D-1}.1`` then ``trades_{D}.1`` (Latin-1), keeps rows whose
    ``Contrato`` start is (D, k), and converts the 11th field to seconds since
    D-1 00:00 (wall clock). A date-only cell equal to D is 86,400 (H-4).
    Returns ``(seconds, date_only_row_indices)``.
    """
    d = date.fromisoformat(delivery_date)
    prev = d - timedelta(days=1)
    key = f"{d:%Y%m%d} {contract_id // 4:02d}:{(contract_id % 4) * 15:02d}"
    base = datetime(prev.year, prev.month, prev.day)
    out, date_only = [], []
    for day in (prev, d):
        path = REAL_DATA / f"trades_{day:%Y%m%d}.1"
        lines = path.read_text(encoding="latin-1").splitlines()
        hdr = next(i for i, ln in enumerate(lines) if ln.startswith("Fecha;Contrato;"))
        for ln in lines[hdr + 1:]:
            f = ln.split(";")
            if len(f) < 11:
                continue
            if f[1].strip().split("-")[0].strip() != key:
                continue
            cell = f[10].strip()
            if _DATE_ONLY_RE.fullmatch(cell):
                assert cell == d.strftime("%d/%m/%Y"), cell
                date_only.append(len(out))
                out.append(MIDNIGHT_D)
            else:
                dt = datetime.strptime(cell, "%d/%m/%Y %H:%M:%S")
                out.append(int((dt - base).total_seconds()))
    return out, date_only


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr.astype(np.int32)).tobytes()).hexdigest()


# Pinned on the unfixed loader (main 7d222ee / branch base 3a667e6), H-6 / T-7.
_PIN_20250625_K0_TRADES_SHAPE = (174, 6)
_PIN_20250625_K0_TRADES_COLS_0_4_SHA256 = (
    "8bab06ce336eb94b16ed563563f0ed674523f4b9c8e720aae445b3882bf2f000"
)
_PIN_20250625_K0_TRADES_COLS_0_4_SUMS = [0, 176271434, 5024, 3178, 8900]
_PIN_20250625_K0_ORDERS_SHAPE = (863, 13)
_PIN_20250625_K0_ORDERS_SHA256 = (
    "e1f2ed8782e5d9a678f64447f0b8cf2fe95274f3ddff48957f9d575ec7a52d84"
)


@pytest.fixture(scope="module")
def real_20250625_k0():
    if not _have_real("2025-06-25"):
        pytest.skip("real OMIE data not present under data/rawOMIE")
    cfg = OMIEDataConfig(data_path=str(REAL_DATA), contract_id=0,
                         delivery_date="2025-06-25")
    return cfg, np.asarray(load_omie_trades(cfg))


@pytest.fixture(scope="module")
def real_20250610_k18():
    if not _have_real("2025-06-10"):
        pytest.skip("real OMIE data not present under data/rawOMIE")
    cfg = OMIEDataConfig(data_path=str(REAL_DATA), contract_id=18,
                         delivery_date="2025-06-10")
    return cfg, np.asarray(load_omie_trades(cfg))


@pytest.mark.slow
class TestRealProduct20250625:
    """slow: parses the real 2025-06-24/25 orders + trades session files."""

    def test_match_times_in_frame_and_non_zero(self, real_20250625_k0):
        """T-1: every column-[5] value is in [55,200, 165,600); none is 0."""
        _, t = real_20250625_k0
        ts = t[:, TRD_MATCH_TS]
        assert t.shape[0] > 0
        assert not (ts == 0).any(), f"{int((ts == 0).sum())} rows carry match time 0"
        assert (ts >= FRAME_LO).all(), f"min={int(ts.min())}"
        assert (ts < FRAME_HI).all(), f"max={int(ts.max())}"

    def test_match_times_equal_independent_reference(self, real_20250625_k0):
        """T-1 (exact): column [5] equals the raw ``Momento casación`` cells,
        in file order (file D-1 rows first, no sort)."""
        _, t = real_20250625_k0
        ref, _ = _reference_match_seconds("2025-06-25", 0)
        assert t[:, TRD_MATCH_TS].tolist() == ref

    def test_columns_0_to_4_unchanged(self, real_20250625_k0):
        """GUARD (T-6, H-6): shape and columns [0..4] are byte-identical to
        the pre-fix loader (row order included)."""
        _, t = real_20250625_k0
        assert t.dtype == np.int32
        assert t.shape == _PIN_20250625_K0_TRADES_SHAPE
        assert t[:, :5].sum(axis=0).tolist() == _PIN_20250625_K0_TRADES_COLS_0_4_SUMS
        assert _digest(t[:, :5]) == _PIN_20250625_K0_TRADES_COLS_0_4_SHA256

    def test_orders_array_unchanged(self, real_20250625_k0):
        """GUARD (T-7, H-2): the orders tape of the same product is
        byte-identical to the pre-fix loader."""
        cfg, _ = real_20250625_k0
        o = np.asarray(load_omie_orders(cfg))
        assert o.shape == _PIN_20250625_K0_ORDERS_SHAPE
        assert _digest(o) == _PIN_20250625_K0_ORDERS_SHA256


@pytest.mark.slow
class TestRealProduct20250610K18:
    """slow: parses the real 2025-06-09/10 orders + trades session files.

    Product (2025-06-10, k18) = ``20250610 04:30-20250610 04:45``;
    ``trades_20250610.1`` carries one date-only ``10/06/2025`` row for it.
    """

    def test_loads_in_frame_with_one_midnight(self, real_20250610_k18):
        """T-4 (real guard): loads without raising, column [5] in the frame,
        exactly one 86,400 and it sits on the date-only row."""
        _, t = real_20250610_k18
        ts = t[:, TRD_MATCH_TS]
        _, date_only = _reference_match_seconds("2025-06-10", 18)
        assert len(date_only) == 1  # reference sanity: the defect is present
        assert (ts >= FRAME_LO).all() and (ts < FRAME_HI).all()
        assert int((ts == MIDNIGHT_D).sum()) == 1
        assert int(ts[date_only[0]]) == MIDNIGHT_D

    def test_match_times_equal_independent_reference(self, real_20250610_k18):
        _, t = real_20250610_k18
        ref, _ = _reference_match_seconds("2025-06-10", 18)
        assert t[:, TRD_MATCH_TS].tolist() == ref
