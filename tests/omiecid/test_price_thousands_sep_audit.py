"""Step-5 AUDIT tests for OMIE-PRICE-THOUSANDS-SEP-1 (independent of the locked file).

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-PRICE-THOUSANDS-SEP-1"
(decisions S2-1, S2-1a, S2-2, S2-3; Test-first contract T-DOT / T-RAISE).
Domain: ``docs/omiecid/market-spec.md`` §13 (OMIE flat files: ``;`` delimiter,
Spanish locale numbers, Latin-1), §2.3 / OQ-01 (±9,999 EUR/MWh SIDC band).

Number rule under audit (card "Decided design" + S2-1 / S2-1a):

* a cell that contains ``,`` is Spanish format: ``.`` = thousands separator,
  ``,`` = decimal. It must fully match
  ``[+-]?(\\d{1,3}(\\.\\d{3})+|\\d+),\\d+``, otherwise it raises;
* a cell WITHOUT ``,`` is parsed as-is (synthetic ``'12.5'`` fixtures and the
  dot-decimal ``marginalpdbc`` files);
* REQUIRED cells (orders / trades ``Precio`` and ``Cantidad``, ``marginalpdbc``
  col B) raise ``OMIENumberParseError(ValueError)`` when empty, unparseable or
  non-finite; OPTIONAL cells (``Cantidad reducida``, ``PPD``) default to 0 only
  when empty, and raise on non-empty garbage;
* the error propagates unwrapped through ``build_fixed_tape_set`` and
  ``build_day_ahead_anchors``.

Interfaces used: ``load_omie_orders``, ``load_omie_trades``,
``parse_marginalpdbc_file``, ``day_ahead_anchor``, ``build_fixed_tape_set``,
``build_day_ahead_anchors`` (public loader API) and the strict helper
``_parse_float_comma(val, *, column, source)`` /
``_parse_optional_float(val, default=0.0, *, column, source)``, whose names and
signatures the card pins (S2-1, v5.167).

Runtime: every synthetic test writes a few-row fixture under ``tmp_path`` and
runs in milliseconds (no jit; the loader is eager pandas + ``jnp.array``).
The real-data tests are ``@pytest.mark.slow`` (each loads whole session files
or sweeps the 14 GB archive) and skip when ``data/rawOMIE`` is absent.
"""

from __future__ import annotations

import math
import os
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

import omiecid_simulator.jaxomiecid.omie_data_loader as odl
from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.omiecid.encoding import encode_ppd, encode_price, encode_qty
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig


PC = OMIEProductConfig()
ZERO_PRICE_INT = encode_price(0.0, PC)        # 999,900 = the old silent default
FLOOR_INT = encode_price(-9999.0, PC)         # 0
CAP_INT = encode_price(9999.0, PC)            # 1,999,800

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA = REPO_ROOT / "data" / "rawOMIE"

D = "2025-06-10"           # synthetic product date (no DST); session files 0609 + 0610
D_PREV_NODASH = "20250609"
D_NODASH = "20250610"


def _err():
    """The named error class (looked up lazily, card T-RAISE precedent)."""
    cls = getattr(odl, "OMIENumberParseError", None)
    assert cls is not None, "OMIENumberParseError missing from omie_data_loader (card S2-1)"
    return cls


# ---------------------------------------------------------------------------
# Synthetic fixture writers (real on-disk layout, market-spec §13.1 / §13.2)
# ---------------------------------------------------------------------------

_ORD_META = "OMIE - Mercado de electricidad;Fecha Emisión :01/01/2025 - 00:00;;{fecha};Órdenes enviadas a XBID;;;;"
_ORD_HDR = (
    "Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;Tipo oferta;"
    "Condiciones a la ejecución;Condiciones a la validez;Cantidad reducida;PPD;"
    "Momento de envío;"
)
_TRD_META = "OMIE - Mercado de electricidad;Fecha Emisión :01/01/2025 - 00:00;;{fecha};Transacciones realizadas en XBID;;;;"
_TRD_HDR = (
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;Agente venta;"
    "Unidad venta;Zona venta;Precio;Cantidad;Momento casación;"
)
_ZONE = "10YES-REE------0"


def _contrato(k: int, d_nodash: str = D_NODASH) -> str:
    hh, mm = divmod(15 * k, 60)
    eh, em = divmod(15 * k + 15, 60)
    end_d = d_nodash
    if eh == 24:
        eh = 0
    return f"{d_nodash} {hh:02d}:{mm:02d}-{end_d} {eh:02d}:{em:02d}"


def _fecha(nodash: str) -> str:
    return f"{nodash[6:8]}/{nodash[4:6]}/{nodash[0:4]}"


def order_line(precio, cantidad="1,00", side="V", *, k=0, exec_="", valid="GTD",
               peak="", ppd="", session=D_NODASH, hhmmss="10:00:00",
               agent=("AGA", "AGA01"), contract_date=D_NODASH) -> str:
    fecha = _fecha(session)
    return (
        f"{fecha};{_contrato(k, contract_date)};{_ZONE};{agent[0]};{agent[1]};"
        f"{precio};{cantidad};{side};{exec_};{valid};{peak};{ppd};"
        f"{fecha} {hhmmss};"
    )


def trade_line(precio, cantidad="1,00", *, k=0, session=D_NODASH,
               hhmmss="10:00:00", contract_date=D_NODASH) -> str:
    fecha = _fecha(session)
    return (
        f"{fecha};{_contrato(k, contract_date)};AGB;AGB01;{_ZONE};AGS;AGS01;{_ZONE};"
        f"{precio};{cantidad};{fecha} {hhmmss};"
    )


def write_product(root: Path, *, orders_prev=(), orders_d=(), trades_prev=(),
                  trades_d=(), terminator=True) -> None:
    """Write the four session files of product (D, k) under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    for nodash, o_rows, t_rows in (
        (D_PREV_NODASH, orders_prev, trades_prev),
        (D_NODASH, orders_d, trades_d),
    ):
        o = [_ORD_META.format(fecha=_fecha(nodash)), "", _ORD_HDR, *o_rows]
        t = [_TRD_META.format(fecha=_fecha(nodash)), "", _TRD_HDR, *t_rows]
        if terminator:
            o.append(";" * 13)
            t.append(";" * 11)
        (root / f"orders_{nodash}.1").write_bytes(("\n".join(o) + "\n").encode("latin-1"))
        (root / f"trades_{nodash}.1").write_bytes(("\n".join(t) + "\n").encode("latin-1"))


def write_marginalpdbc(root: Path, col_b: dict, col_a: dict | None = None,
                       nodash: str = D_NODASH) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    y, m, d = nodash[:4], nodash[4:6], nodash[6:8]
    lines = ["MARGINALPDBC;"]
    for h in sorted(col_b):
        a = (col_a or {}).get(h, "50.0")
        lines.append(f"{y};{m};{d};{h};{a};{col_b[h]};")
    lines.append("*")
    p = root / f"marginalpdbc_{nodash}.1"
    p.write_bytes(("\n".join(lines) + "\n").encode("latin-1"))
    return p


def cfg(root: Path, k: int = 0) -> OMIEDataConfig:
    return OMIEDataConfig(data_path=str(root), contract_id=k, delivery_date=D)


def orders_np(root: Path, k: int = 0) -> np.ndarray:
    return np.asarray(odl.load_omie_orders(cfg(root, k)))


def trades_np(root: Path, k: int = 0) -> np.ndarray:
    return np.asarray(odl.load_omie_trades(cfg(root, k)))


def _ts(i: int) -> str:
    """Distinct, increasing D-file send times (keeps loader order = list order)."""
    return f"10:{i // 60:02d}:{i % 60:02d}"


def spanish_ref(raw: str) -> Decimal:
    """Independent reference parse of a Spanish number (exact, Decimal)."""
    s = raw.strip()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    return Decimal(s)


# ---------------------------------------------------------------------------
# 1. The named error class
# ---------------------------------------------------------------------------

class TestErrorClass:
    def test_error_is_a_valueerror_subclass(self):
        assert issubclass(_err(), ValueError)

    def test_error_is_distinct_from_incomplete_product_error(self):
        # Two different failure classes: a missing file vs a bad cell.
        assert not issubclass(_err(), odl.IncompleteProductError)
        assert not issubclass(odl.IncompleteProductError, _err())


# ---------------------------------------------------------------------------
# 2. Strict helper: the number rule cell by cell (card S2-1 / S2-1a)
# ---------------------------------------------------------------------------

def _parse(val):
    return odl._parse_float_comma(val, column="Precio", source="audit-src")


_VALID = [
    # Spanish format with thousands grouping (the defect class)
    ("-9.999,00", -9999.0),
    ("9.999,00", 9999.0),
    ("+9.999,00", 9999.0),
    ("-1.249,90", -1249.9),
    ("1.000,00", 1000.0),
    ("-1.000,01", -1000.01),
    ("4.841,69", 4841.69),
    ("1.234.567,8", 1234567.8),          # multi-group: honest value, no band clip
    ("-1.234.567,89", -1234567.89),
    # comma decimal without grouping
    ("999,99", 999.99),
    ("-999,99", -999.99),
    ("12,34", 12.34),
    ("0,00", 0.0),
    ("-0,00", 0.0),
    ("+0,5", 0.5),
    ("12,345", 12.345),                  # 3 decimals after ',' is NOT grouping
    ("1234,5", 1234.5),                  # ungrouped integer part is allowed
    ("0001,5", 1.5),
    # whitespace around the cell
    ("  -9.999,00  ", -9999.0),
    ("\t12,5\n", 12.5),
    # comma-less cells parse as-is (dot = decimal point)
    ("12.5", 12.5),
    ("36.07", 36.07),
    ("-12.5", -12.5),
    ("100", 100.0),
    ("0", 0.0),
    ("+5", 5.0),
]


class TestHelperValidCells:
    @pytest.mark.parametrize("raw,expected", _VALID, ids=[repr(r) for r, _ in _VALID])
    def test_cell_parses_to_exact_value(self, raw, expected):
        got = _parse(raw)
        assert isinstance(got, float)
        assert got == pytest.approx(expected, abs=1e-9, rel=0)

    @pytest.mark.parametrize("raw", [r for r, _ in _VALID if "." in r and "," in r])
    def test_grouped_cell_never_decodes_to_zero(self, raw):
        assert _parse(raw) != 0.0

    @pytest.mark.parametrize("raw,expected", [(12.5, 12.5), (7, 7.0), (-9999, -9999.0)])
    def test_non_string_numeric_input(self, raw, expected):
        assert _parse(raw) == expected

    def test_floor_and_cap_encode_to_band_edges(self):
        assert encode_price(_parse("-9.999,00"), PC) == FLOOR_INT == 0
        assert encode_price(_parse("9.999,00"), PC) == CAP_INT == 1_999_800


_RAISE = [
    "", "   ", "nan", "NaN", " NAN ", "inf", "-inf", "+inf", "Infinity",
    "1e999",                     # comma-less, finite-looking text, overflows to inf
    "abc", "1,234.5", "12.34,5", "1.23,4", "1.2345,6", ".999,00", "9.999,",
    ",5", "-,5", "12,", "1,2,3", "--5,0", "+-5,0", "9 999,00", "9.999,00€",
    "1..000,0", "1.000.,0", "1.000,0.0", "inf,0", "nan,0", "1,0e3", "1e3,0",
    "9.999,00;", "−9.999,00",    # U+2212 minus sign, not ASCII '-'
]


class TestHelperRejectsCells:
    @pytest.mark.parametrize("raw", _RAISE, ids=[repr(r) for r in _RAISE])
    def test_cell_raises_named_error(self, raw):
        with pytest.raises(_err()):
            _parse(raw)

    @pytest.mark.parametrize("raw", [None, float("nan")])
    def test_missing_sentinels_raise(self, raw):
        with pytest.raises(_err()):
            _parse(raw)

    def test_message_names_column_source_and_raw_value(self):
        with pytest.raises(_err()) as ei:
            odl._parse_float_comma("12.34,5", column="Cantidad", source="file-XYZ")
        msg = str(ei.value)
        assert "Cantidad" in msg and "file-XYZ" in msg and "12.34,5" in msg

    def test_helper_has_no_default_parameter(self):
        # S2-1: the silent default is gone; column/source are keyword-only.
        with pytest.raises(TypeError):
            odl._parse_float_comma("abc", 0.0, column="Precio", source="s")  # noqa
        with pytest.raises(TypeError):
            odl._parse_float_comma("1,0")  # missing kw-only column/source


class TestOptionalHelper:
    @pytest.mark.parametrize("raw", [None, "", "  ", "nan", "NaN", float("nan")])
    def test_empty_optional_returns_default(self, raw):
        got = odl._parse_optional_float(raw, column="PPD", source="s")
        assert got == 0.0
        assert odl._parse_optional_float(raw, 3.5, column="PPD", source="s") == 3.5

    @pytest.mark.parametrize("raw,expected", [("5,00", 5.0), ("-1,00", -1.0),
                                              ("1.000,00", 1000.0), ("0", 0.0)])
    def test_non_empty_optional_uses_strict_rule(self, raw, expected):
        assert odl._parse_optional_float(raw, column="PPD", source="s") == expected

    @pytest.mark.parametrize("raw", ["abc", "inf", "1,234.5", "12.34,5"])
    def test_non_empty_optional_garbage_raises(self, raw):
        with pytest.raises(_err()) as ei:
            odl._parse_optional_float(raw, column="Cantidad reducida", source="s")
        assert "Cantidad reducida" in str(ei.value)


# ---------------------------------------------------------------------------
# 3. load_omie_orders: every call site, end to end on a synthetic product
# ---------------------------------------------------------------------------

_ORDER_PRICES = [
    ("-9.999,00", "C", -9999.0),   # deep passive bid placeholder (dominant real case)
    ("9.999,00", "V", 9999.0),     # deep passive ask placeholder (the phantom-ask case)
    ("9.999,00", "C", 9999.0),     # aggressive BUY at cap (rare)
    ("-9.999,00", "V", -9999.0),
    ("+9.999,00", "V", 9999.0),
    ("-1.249,90", "C", -1249.9),
    ("1.000,00", "V", 1000.0),
    ("-1.000,00", "C", -1000.0),
    ("999,99", "V", 999.99),
    ("12,34", "C", 12.34),
    ("0,00", "V", 0.0),
    ("12.5", "C", 12.5),           # synthetic plain-number fixture keeps working
    ("100", "V", 100.0),
]


class TestOrdersPriceDecode:
    def test_every_price_cell_decodes_on_the_offset_grid(self, tmp_path):
        rows = [order_line(p, side=s, hhmmss=_ts(i)) for i, (p, s, _) in enumerate(_ORDER_PRICES)]
        write_product(tmp_path, orders_d=rows)
        o = orders_np(tmp_path)
        assert o.shape == (len(_ORDER_PRICES), 13)
        assert o[:, 2].tolist() == [encode_price(e, PC) for _, _, e in _ORDER_PRICES]
        assert o[:, 1].tolist() == [0 if s == "C" else 1 for _, s, _ in _ORDER_PRICES]

    def test_placeholders_are_not_stored_at_zero(self, tmp_path):
        rows = [order_line("-9.999,00", side="C", hhmmss=_ts(0)),
                order_line("9.999,00", side="V", hhmmss=_ts(1))]
        write_product(tmp_path, orders_d=rows)
        o = orders_np(tmp_path)
        assert o[:, 2].tolist() == [FLOOR_INT, CAP_INT]
        assert ZERO_PRICE_INT not in o[:, 2].tolist()

    def test_grouped_row_in_file_d_minus_1_decodes(self, tmp_path):
        rows_prev = [order_line("-9.999,00", side="C", session=D_PREV_NODASH, hhmmss="23:00:00")]
        write_product(tmp_path, orders_prev=rows_prev, orders_d=[order_line("50,00")])
        o = orders_np(tmp_path)
        assert o[:, 2].tolist() == [FLOOR_INT, encode_price(50.0, PC)]

    def test_multi_group_value_is_never_silently_zero(self, tmp_path):
        """'1.234.567,8' is outside the ±9,999 band; band clipping is NOT in
        scope (card), so either the honest value loads or a ValueError is
        raised — a silent 0 (or a truncated '1.234') is the only failure."""
        write_product(tmp_path, orders_d=[order_line("1.234.567,8")])
        try:
            o = orders_np(tmp_path)
        except ValueError:
            return
        assert int(o[0, 2]) == encode_price(1234567.8, PC)
        assert int(o[0, 2]) != ZERO_PRICE_INT

    def test_whitespace_padded_cells_decode(self, tmp_path):
        write_product(tmp_path, orders_d=[order_line("  -9.999,00 ", cantidad=" 2,50 ")])
        o = orders_np(tmp_path)
        assert int(o[0, 2]) == FLOOR_INT and int(o[0, 3]) == encode_qty(2.5, PC)

    def test_quantity_with_comma_decimal(self, tmp_path):
        write_product(tmp_path, orders_d=[order_line("10,00", cantidad="144,00"),
                                          order_line("10,00", cantidad="0,10", hhmmss=_ts(1))])
        o = orders_np(tmp_path)
        assert o[:, 3].tolist() == [encode_qty(144.0, PC), encode_qty(0.1, PC)]

    def test_terminator_line_does_not_raise(self, tmp_path):
        # The ';;;...' trailer is the only empty-cell line in real files.
        write_product(tmp_path, orders_d=[order_line("12,34")], terminator=True)
        assert orders_np(tmp_path).shape == (1, 13)


class TestOrdersOptionalColumns:
    def test_iceberg_peak_and_ppd_decode(self, tmp_path):
        rows = [
            order_line("100,00", "144,00", "V", exec_="ICE", peak="5,00", ppd="0", hhmmss=_ts(0)),
            order_line("100,00", "10,00", "V", exec_="ICE", peak="2,50", ppd="-1,00", hhmmss=_ts(1)),
            order_line("100,00", "10,00", "V", exec_="ICE", peak="2,50", ppd="1.000,00", hhmmss=_ts(2)),
        ]
        write_product(tmp_path, orders_d=rows)
        o = orders_np(tmp_path)
        assert o[:, 11].tolist() == [encode_qty(5.0, PC), encode_qty(2.5, PC), encode_qty(2.5, PC)]
        assert o[:, 12].tolist() == [encode_ppd(0.0, PC), encode_ppd(-1.0, PC), encode_ppd(1000.0, PC)]

    def test_empty_optional_columns_default_to_zero(self, tmp_path):
        write_product(tmp_path, orders_d=[order_line("-9.999,00", side="C")])
        o = orders_np(tmp_path)
        assert int(o[0, 11]) == 0 and int(o[0, 12]) == 0

    @pytest.mark.parametrize("peak,ppd,col", [
        ("abc", "0", "Cantidad reducida"),
        ("1,234.5", "0", "Cantidad reducida"),
        ("5,00", "x,y", "PPD"),
        ("5,00", "inf", "PPD"),
        ("5,00", "12.34,5", "PPD"),
    ])
    def test_non_empty_optional_garbage_raises_with_column(self, tmp_path, peak, ppd, col):
        write_product(tmp_path, orders_d=[order_line("100,00", exec_="ICE", peak=peak, ppd=ppd)])
        with pytest.raises(_err()) as ei:
            orders_np(tmp_path)
        assert col in str(ei.value)


_ORDER_BAD = [
    ("Precio", ""), ("Precio", "abc"), ("Precio", "nan"), ("Precio", "inf"),
    ("Precio", "-inf"), ("Precio", "1,234.5"), ("Precio", "12.34,5"),
    ("Precio", "1.2345,6"), ("Precio", "9.999,"), ("Precio", "1e999"),
    ("Cantidad", ""), ("Cantidad", "abc"), ("Cantidad", "nan"),
    ("Cantidad", "inf"), ("Cantidad", "1,0.0"),
]


class TestOrdersRequiredCellsRaise:
    @pytest.mark.parametrize("column,raw", _ORDER_BAD, ids=[f"{c}={r!r}" for c, r in _ORDER_BAD])
    def test_bad_required_cell_raises(self, tmp_path, column, raw):
        line = order_line(raw if column == "Precio" else "10,00",
                          raw if column == "Cantidad" else "1,00")
        write_product(tmp_path, orders_d=[order_line("10,00", hhmmss=_ts(0)), line])
        with pytest.raises(_err()) as ei:
            orders_np(tmp_path)
        msg = str(ei.value)
        assert column in msg
        if raw:
            assert raw in msg
        assert str(tmp_path) in msg   # source names where the bad cell came from

    def test_bad_cell_in_file_d_minus_1_raises(self, tmp_path):
        rows_prev = [order_line("abc", session=D_PREV_NODASH, hhmmss="23:00:00")]
        write_product(tmp_path, orders_prev=rows_prev, orders_d=[order_line("50,00")])
        with pytest.raises(_err()):
            orders_np(tmp_path)

    def test_bad_cell_of_another_product_does_not_poison_this_one(self, tmp_path):
        """Only the requested product's rows reach the parser: a bad cell on
        slot 1, or on the D+1 product carried by file D, is not product (D, 0)."""
        rows = [
            order_line("-9.999,00", side="C", k=0, hhmmss=_ts(0)),
            order_line("abc", k=1, hhmmss=_ts(1)),
            order_line("abc", k=0, contract_date="20250611", hhmmss=_ts(2)),
        ]
        write_product(tmp_path, orders_d=rows)
        o = orders_np(tmp_path, k=0)
        assert o[:, 2].tolist() == [FLOOR_INT]
        with pytest.raises(_err()):
            orders_np(tmp_path, k=1)


# ---------------------------------------------------------------------------
# 4. load_omie_trades
# ---------------------------------------------------------------------------

_TRADE_PRICES = [
    ("4.841,69", 4841.69), ("-1.249,99", -1249.99), ("-9.999,00", -9999.0),
    ("+1.000,00", 1000.0), ("149,50", 149.5), (" -1.195,88 ", -1195.88), ("12.5", 12.5),
]


class TestTrades:
    def test_trade_prices_decode(self, tmp_path):
        trades = [trade_line(p, hhmmss=_ts(i)) for i, (p, _) in enumerate(_TRADE_PRICES)]
        write_product(tmp_path, orders_d=[order_line("10,00")], trades_d=trades)
        t = trades_np(tmp_path)
        assert t[:, 1].tolist() == [encode_price(e, PC) for _, e in _TRADE_PRICES]

    def test_trade_in_file_d_minus_1_decodes(self, tmp_path):
        write_product(tmp_path, orders_d=[order_line("10,00")],
                      trades_prev=[trade_line("4.841,69", "0,60", session=D_PREV_NODASH,
                                              hhmmss="23:00:00")])
        t = trades_np(tmp_path)
        assert t[:, 1].tolist() == [encode_price(4841.69, PC)]
        assert t[:, 2].tolist() == [encode_qty(0.6, PC)]

    @pytest.mark.parametrize("column,raw", [
        ("Precio", ""), ("Precio", "abc"), ("Precio", "nan"), ("Precio", "inf"),
        ("Precio", "1,234.5"), ("Precio", "12.34,5"),
        ("Cantidad", ""), ("Cantidad", "abc"), ("Cantidad", "-inf"),
    ])
    def test_bad_required_trade_cell_raises(self, tmp_path, column, raw):
        line = trade_line(raw if column == "Precio" else "10,00",
                          raw if column == "Cantidad" else "1,00")
        write_product(tmp_path, orders_d=[order_line("10,00")], trades_d=[line])
        with pytest.raises(_err()) as ei:
            trades_np(tmp_path)
        assert column in str(ei.value)
        if raw:
            assert raw in str(ei.value)

    def test_bad_trade_of_another_product_does_not_raise(self, tmp_path):
        write_product(tmp_path, orders_d=[order_line("10,00")],
                      trades_d=[trade_line("12,00", k=0), trade_line("abc", k=5, hhmmss=_ts(1))])
        assert trades_np(tmp_path, k=0)[:, 1].tolist() == [encode_price(12.0, PC)]


# ---------------------------------------------------------------------------
# 5. marginalpdbc (dot-decimal on disk; ',' => Spanish format)
# ---------------------------------------------------------------------------

class TestMarginalpdbc:
    def test_dot_decimal_cells_are_not_stripped(self, tmp_path):
        p = write_marginalpdbc(tmp_path, {1: "36.07", 2: "129.8", 3: "60", 4: "-0.5",
                                          5: "+12.25", 6: "0"})
        got = odl.parse_marginalpdbc_file(str(p))
        assert got == pytest.approx({1: 36.07, 2: 129.8, 3: 60.0, 4: -0.5, 5: 12.25, 6: 0.0})

    def test_comma_cells_use_spanish_rule(self, tmp_path):
        p = write_marginalpdbc(tmp_path, {1: "1.234,5", 2: "12,5", 3: "-9.999,00"})
        got = odl.parse_marginalpdbc_file(str(p))
        assert got == pytest.approx({1: 1234.5, 2: 12.5, 3: -9999.0})

    @pytest.mark.parametrize("raw", ["abc", "nan", "inf", "1,234.5", "1.23,4", "1e999"])
    def test_bad_col_b_raises_and_names_file(self, tmp_path, raw):
        p = write_marginalpdbc(tmp_path, {1: "36.07", 2: raw})
        with pytest.raises(_err()) as ei:
            odl.parse_marginalpdbc_file(str(p))
        assert str(p) in str(ei.value) and raw in str(ei.value)

    def test_bad_hour_far_from_the_anchor_still_raises(self, tmp_path):
        # The whole file is parsed strictly; a bad hour 24 is not silently 0.
        write_marginalpdbc(tmp_path, {1: "36.07", 24: "abc"})
        with pytest.raises(_err()):
            odl.day_ahead_anchor(0, D, str(tmp_path))

    def test_col_a_portugal_is_not_parsed(self, tmp_path):
        # Only col B (ES) is a required site; junk in col A (PT) is ignored.
        p = write_marginalpdbc(tmp_path, {1: "36.07"}, col_a={1: "garbage"})
        assert odl.parse_marginalpdbc_file(str(p)) == pytest.approx({1: 36.07})

    def test_anchor_reads_dot_decimal_value(self, tmp_path):
        write_marginalpdbc(tmp_path, {1: "36.07", 2: "129.8"})
        assert odl.day_ahead_anchor(4, D, str(tmp_path)) == pytest.approx(129.8)


# ---------------------------------------------------------------------------
# 6. Error propagation through the tape builders (unwrapped)
# ---------------------------------------------------------------------------

class TestPropagation:
    def _bad_orders(self, root, *, bad_ppd=False):
        rows = [order_line("10,00", hhmmss=_ts(0))]
        if bad_ppd:
            rows.append(order_line("10,00", exec_="ICE", peak="1,00", ppd="abc", hhmmss=_ts(1)))
        else:
            rows.append(order_line("1,234.5", hhmmss=_ts(1)))
        write_product(root, orders_d=rows)

    def test_build_fixed_tape_set_configs_form(self, tmp_path):
        self._bad_orders(tmp_path)
        with pytest.raises(ValueError) as ei:
            odl.build_fixed_tape_set([cfg(tmp_path)])
        assert ei.type is _err()

    def test_build_fixed_tape_set_pointer_form(self, tmp_path):
        self._bad_orders(tmp_path)
        with pytest.raises(ValueError) as ei:
            odl.build_fixed_tape_set(data_path=str(tmp_path), contract_ids=[0],
                                     delivery_dates=[D])
        assert ei.type is _err()

    def test_bad_optional_cell_propagates_through_tape_builder(self, tmp_path):
        self._bad_orders(tmp_path, bad_ppd=True)
        with pytest.raises(ValueError) as ei:
            odl.build_fixed_tape_set([cfg(tmp_path)])
        assert ei.type is _err()

    def test_one_bad_tape_fails_the_whole_set(self, tmp_path):
        good, bad = tmp_path / "good", tmp_path / "bad"
        write_product(good, orders_d=[order_line("-9.999,00", side="C")])
        self._bad_orders(bad)
        with pytest.raises(ValueError) as ei:
            odl.build_fixed_tape_set([cfg(good), cfg(bad)])
        assert ei.type is _err()

    def test_build_day_ahead_anchors_bad_orders(self, tmp_path):
        self._bad_orders(tmp_path)
        write_marginalpdbc(tmp_path, {1: "36.07"})
        with pytest.raises(ValueError) as ei:
            odl.build_day_ahead_anchors([cfg(tmp_path)])
        assert ei.type is _err()

    def test_build_day_ahead_anchors_bad_marginalpdbc(self, tmp_path):
        write_product(tmp_path, orders_d=[order_line("10,00")])
        write_marginalpdbc(tmp_path, {1: "nan"})
        with pytest.raises(ValueError) as ei:
            odl.build_day_ahead_anchors([cfg(tmp_path)])
        assert ei.type is _err()

    def test_good_product_builds_with_placeholders_at_band_edges(self, tmp_path):
        write_product(tmp_path, orders_d=[order_line("-9.999,00", side="C", hhmmss=_ts(0)),
                                          order_line("9.999,00", side="V", hhmmss=_ts(1))])
        write_marginalpdbc(tmp_path, {1: "36.07"})
        tapes = np.asarray(odl.build_fixed_tape_set([cfg(tmp_path)]))
        assert tapes.shape[0] == 1
        assert tapes[0, :2, 2].tolist() == [FLOOR_INT, CAP_INT]
        anchors = np.asarray(odl.build_day_ahead_anchors([cfg(tmp_path)]))
        assert anchors.tolist() == pytest.approx([36.07])


# ---------------------------------------------------------------------------
# 7. Real data (slow; skip when data/rawOMIE is absent)
# ---------------------------------------------------------------------------

needs_real = pytest.mark.skipif(not REAL_DATA.is_dir(), reason="data/rawOMIE not present")


def _real_files(prefix: str) -> list:
    return sorted(p for p in REAL_DATA.iterdir() if p.name.startswith(prefix + "_"))


@pytest.fixture(scope="module")
def archive_dot_cells():
    """Whole-archive sweep: every orders/trades cell that contains '.'.

    slow: reads the full 14 GB archive once (~5 s warm page cache). Only
    cells containing '.' are extracted (a '.' never appears in dates/times),
    with their 0-based field index.
    """
    found = {}          # (kind, field_idx) -> count
    uniq = set()
    for kind in ("orders", "trades"):
        for p in _real_files(kind):
            b = p.read_bytes()
            i = b.find(b".")
            while i != -1:
                ls = b.rfind(b"\n", 0, i) + 1
                fs = b.rfind(b";", ls, i) + 1
                fe = b.find(b";", i)
                if fe == -1:
                    fe = b.find(b"\n", i)
                key = (kind, b.count(b";", ls, i))
                found[key] = found.get(key, 0) + 1
                uniq.add(b[fs:fe].decode("latin-1"))
                i = b.find(b".", fe)
    return found, uniq


@needs_real
@pytest.mark.slow
class TestRealArchiveSweep:
    """slow: whole-archive sweep of every thousands-separated cell."""

    def test_dots_occur_only_in_price_columns(self, archive_dot_cells):
        found, _ = archive_dot_cells
        # orders Precio = field 5, trades Precio = field 8 (market-spec §13.1/§13.2):
        # Cantidad / Cantidad reducida / PPD never carry a '.'.
        assert set(found) <= {("orders", 5), ("trades", 8)}, found
        assert found.get(("orders", 5), 0) >= 478_932
        assert found.get(("trades", 8), 0) >= 10

    def test_every_dot_cell_parses_to_independent_value(self, archive_dot_cells):
        _, uniq = archive_dot_cells
        bad = []
        for raw in uniq:
            got = odl._parse_float_comma(raw, column="Precio", source="sweep")
            ref = spanish_ref(raw)
            if got == 0.0 or not (1000 <= abs(ref) <= 9999) or abs(Decimal(repr(got)) - ref) > Decimal("1e-9"):
                bad.append((raw, got, ref))
            elif encode_price(got, PC) != int((ref - Decimal("-9999")) / Decimal("0.01")):
                bad.append((raw, got, "encode"))
        assert not bad, bad[:20]

    def test_marginalpdbc_col_b_dot_decimal_everywhere(self):
        files = _real_files("marginalpdbc")
        assert files
        for p in files:
            got = odl.parse_marginalpdbc_file(str(p))
            ref = {}
            for line in p.read_text(encoding="latin-1").splitlines():
                f = [c.strip() for c in line.split(";") if c.strip()]
                if len(f) >= 6 and f[3].isdigit():
                    assert "," not in f[5], (p.name, line)
                    ref[int(f[3])] = float(Decimal(f[5]))
            assert got == ref, p.name
            assert all(math.isfinite(v) and v < 1000 for v in got.values()), p.name


def _ref_product_rows(d_iso: str, k: int):
    """Independent multiset of (side, price_int, qty_int) for product (D, k).

    Reads files D-1 and D directly; a row belongs to the product iff its
    Contrato starts with 'YYYYMMDD HH:MM' of slot k (market-spec §13.4).
    """
    from datetime import date, timedelta
    d = date.fromisoformat(d_iso)
    prefix = f"{d.strftime('%Y%m%d')} {(15 * k) // 60:02d}:{(15 * k) % 60:02d}-"
    out, n_grouped = [], 0
    for day in (d - timedelta(days=1), d):
        p = REAL_DATA / f"orders_{day.strftime('%Y%m%d')}.1"
        for line in p.read_text(encoding="latin-1").splitlines():
            f = line.split(";")
            if len(f) < 8 or not f[1].strip().startswith(prefix):
                continue
            pr = spanish_ref(f[5])
            n_grouped += "." in f[5]
            out.append((
                0 if f[7].strip().upper() == "C" else 1,
                int((pr - Decimal("-9999")) / Decimal("0.01")),
                int(spanish_ref(f[6]) / Decimal("0.1")),
            ))
    return sorted(out), n_grouped


_REAL_PRODUCTS = [("2025-06-24", 51), ("2025-10-15", 56), ("2025-06-25", 0)]


@needs_real
@pytest.mark.slow
class TestRealProductsDecode:
    """slow: each case loads two full real session files (~3-6 s)."""

    @pytest.mark.parametrize("d_iso,k", _REAL_PRODUCTS)
    def test_loaded_prices_equal_independent_raw_parse(self, d_iso, k):
        prev = (np.datetime64(d_iso) - np.timedelta64(1, "D")).astype(str).replace("-", "")
        if not (REAL_DATA / f"orders_{prev}.1").exists():
            pytest.skip(f"orders_{prev}.1 absent")
        o = np.asarray(odl.load_omie_orders(
            OMIEDataConfig(data_path=str(REAL_DATA), contract_id=k, delivery_date=d_iso)))
        ref, n_grouped = _ref_product_rows(d_iso, k)
        assert n_grouped > 0, "fixture product must carry thousands-separated rows"
        assert sorted(map(tuple, o[:, [1, 2, 3]].tolist())) == ref

    @pytest.mark.parametrize("d_iso,k,price,n", [("2025-07-22", 32, 4841.69, 2),
                                                 ("2025-11-25", 11, -1249.99, 2)])
    def test_real_thousands_trades_decode(self, d_iso, k, price, n):
        prev = (np.datetime64(d_iso) - np.timedelta64(1, "D")).astype(str).replace("-", "")
        for kind in ("orders", "trades"):
            for day in (prev, d_iso.replace("-", "")):
                if not (REAL_DATA / f"{kind}_{day}.1").exists():
                    pytest.skip(f"{kind}_{day}.1 absent")
        t = np.asarray(odl.load_omie_trades(
            OMIEDataConfig(data_path=str(REAL_DATA), contract_id=k, delivery_date=d_iso)))
        assert int(np.sum(t[:, 1] == encode_price(price, PC))) == n
