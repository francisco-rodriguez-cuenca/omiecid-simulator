"""Locked behavioural contract tests for OMIE-PRICE-THOUSANDS-SEP-1.

OMIE writes Spanish numbers: ``.`` is the thousands separator and ``,`` the
decimal separator (``'-9.999,00'`` = -9,999.00 EUR/MWh). The day-ahead
``marginalpdbc`` flat file is the exception: it is DOT-decimal (``'36.07'``)
and never contains ``,``.

Sources:
  * docs/omiecid/agile-plan.md, card "Story OMIE-PRICE-THOUSANDS-SEP-1" (v5.167):
    Decided design (S2-1, S2-1a, S2-2, S2-3) and the Test-first contract
    (incl. T-DOT and T-RAISE).
  * docs/omiecid/market-spec.md §13 (CSV format; comma decimal).

Public contract pinned here (the Implementation Agent codes against this):

  * Number rule. A cell that contains ``,`` is Spanish format: it must match
    ``[+-]?(\\d{1,3}(\\.\\d{3})+|\\d+),\\d+``; every ``.`` is removed and the
    ``,`` becomes the decimal point. A cell WITHOUT ``,`` is parsed as-is
    (``'12.5'`` = 12.5, ``'36.07'`` = 36.07, ``'60'`` = 60.0).
  * ``OMIENumberParseError`` is a module-level ``ValueError`` subclass in
    ``omiecid_simulator.jaxomiecid.omie_data_loader``. It is raised when a REQUIRED
    numeric cell (orders ``Precio`` / ``Cantidad``, trades ``Precio`` /
    ``Cantidad``, ``marginalpdbc`` col B) is empty, unparseable, non-finite
    (``'nan'``, ``'inf'``), or contains ``,`` without well-formed Spanish
    grouping. The message names the column and the raw value.
  * Optional order columns (``Cantidad reducida``, ``PPD``): empty -> 0.0;
    a NON-empty unparseable cell raises ``OMIENumberParseError``.
  * The error propagates unwrapped through ``build_fixed_tape_set`` and
    ``build_day_ahead_anchors``.

``OMIENumberParseError`` is looked up with ``getattr`` inside the tests so
this module collects on the unfixed loader (the ``IncompleteProductError``
precedent in ``test_tape_contract_date.py``).

Expected status on the CURRENT (unfixed) loader:
  TestOrdersThousandsSeparator::test_spanish_prices_decode             RED
  TestOrdersThousandsSeparator::test_floor_and_cap_orders_...          RED
  TestOrdersThousandsSeparator::test_comma_decimal_and_plain_prices    GREEN (guard)
  TestTradesThousandsSeparator::test_spanish_prices_decode             RED
  TestTradesThousandsSeparator::test_comma_decimal_prices              GREEN (guard)
  TestMarginalpdbcNumbers::test_thousands_separated_col_b              RED
  TestMarginalpdbcNumbers::test_comma_decimal_col_b                    GREEN (guard)
  TestMarginalpdbcNumbers::test_dot_decimal_col_b (T-DOT i)            GREEN (guard)
  TestMarginalpdbcNumbers::test_real_dot_decimal_file (T-DOT ii)       GREEN (guard)
  TestErrorClass                                                       RED
  TestOrdersRaise / TestTradesRaise / TestMarginalpdbcRaise            RED
  TestOptionalColumns::test_empty_optional_columns_default_to_zero     GREEN (guard)
  TestOptionalColumns::test_spanish_optional_columns_decode            GREEN (guard)
  TestOptionalColumns::test_non_empty_garbage_raises                   RED
  TestPropagation::test_build_*_propagates                             RED
  TestPropagation::test_build_day_ahead_anchors_positive_control       GREEN (guard)
  TestRealProduct20250625::test_reference_has_thousands_separated_cells GREEN (guard)
  TestRealProduct20250625::test_no_thousands_separated_row_decodes_to_zero RED
  TestRealProduct20250625::test_price_multiset_matches_independent_reference RED

Runtime: synthetic tests write tmp_path CSVs (< 1 s each). The real-product
tests are ``@pytest.mark.slow`` (they parse two 30-45 MB session files) and
share one module-scoped load. They skip only when ``data/rawOMIE`` is absent.
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pytest

import omiecid_simulator.jaxomiecid.omie_data_loader as odl
from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    build_day_ahead_anchors,
    build_fixed_tape_set,
    load_omie_orders,
    load_omie_trades,
    parse_marginalpdbc_file,
)
from omiecid_simulator.omiecid.encoding import encode_ppd, encode_price, encode_qty
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORD_PRICE, ORD_QTY, ORD_PEAK, ORD_PPD = 2, 3, 11, 12
TRD_PRICE, TRD_QTY = 1, 2

_PC = OMIEProductConfig()

_REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA = _REPO_ROOT / "data" / "rawOMIE"

# Synthetic product: delivery day D, slot K (10:00-10:15). All rows live in
# session file D; file D-1 exists (completeness, OQ-29) with header only.
D = "2025-06-02"
D_PREV = "2025-06-01"
K = 40
CONTRATO = "20250602 10:00-20250602 10:15"
_ZONE = "10YES-REE------0"

_ORDERS_META = "OMIE - Mercado de electricidad;Fecha Emision :;;;Ordenes enviadas a XBID;;;;"
_ORDERS_COLS = (
    "Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;Tipo oferta;"
    "Condiciones a la ejecucion;Condiciones a la validez;"
    "Cantidad reducida;PPD;Momento de envio;"
)
_TRADES_META = "OMIE - Mercado de electricidad;Fecha Emision :;;;Transacciones realizadas en XBID;;;;"
_TRADES_COLS = (
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;"
    "Agente venta;Unidad venta;Zona venta;Precio;Cantidad;Momento casacion;"
)


# ---------------------------------------------------------------------------
# Accessor for the name the fix introduces
# ---------------------------------------------------------------------------

def _err_cls():
    cls = getattr(odl, "OMIENumberParseError", None)
    if cls is None:
        pytest.fail(
            "omie_data_loader.OMIENumberParseError is missing "
            "(card S2-1: a named, module-level ValueError subclass)"
        )
    assert isinstance(cls, type) and issubclass(cls, ValueError), (
        "OMIENumberParseError must subclass ValueError"
    )
    return cls


def _cfg(data_path, contract_id=K, delivery_date=D):
    return OMIEDataConfig(
        data_path=str(data_path),
        contract_id=int(contract_id),
        delivery_date=delivery_date,
        cache_path=str(data_path),
    )


def _pi(price_eur):
    return int(encode_price(price_eur, _PC))


# ---------------------------------------------------------------------------
# Synthetic session-file writers. Cells are written VERBATIM (raw strings), so
# each test controls the exact text the parser sees.
# ---------------------------------------------------------------------------

def _send(i):
    """Distinct, increasing send times inside session file D (stable order)."""
    return f"02/06/2025 08:{i // 60:02d}:{i % 60:02d}"


def _order_line(precio, cantidad="1,0", side="V", peak="", ppd="", i=0):
    return (
        f"02/06/2025;{CONTRATO};{_ZONE};ABC;U1;{precio};{cantidad};{side};"
        f";;{peak};{ppd};{_send(i)};"
    )


def _trade_line(precio, cantidad="1,0", i=0):
    return (
        f"02/06/2025;{CONTRATO};BYR;BU1;{_ZONE};SLR;SU1;{_ZONE};"
        f"{precio};{cantidad};02/06/2025 09:{i // 60:02d}:{i % 60:02d};"
    )


def _write(dirpath, *, order_lines=None, trade_lines=None):
    """Write a complete product (files D-1 and D for orders and trades).

    Defaults keep each loader's frame non-empty with one well-formed row.
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    if order_lines is None:
        order_lines = [_order_line("50,00")]
    if trade_lines is None:
        trade_lines = [_trade_line("50,00")]
    for iso in (D_PREV, D):
        nd = iso.replace("-", "")
        o = [_ORDERS_META, "", _ORDERS_COLS] + (order_lines if iso == D else [])
        (dirpath / f"orders_{nd}.1").write_text("\n".join(o) + "\n", encoding="latin-1")
        t = [_TRADES_META, _TRADES_COLS] + (trade_lines if iso == D else [])
        (dirpath / f"trades_{nd}.1").write_text("\n".join(t) + "\n", encoding="latin-1")
    return dirpath


def _write_marginalpdbc(path, col_b_by_hour, col_a="50.0"):
    lines = ["MARGINALPDBC;"]
    for h in sorted(col_b_by_hour):
        lines.append(f"2025;06;02;{h};{col_a};{col_b_by_hour[h]};")
    lines.append("*")
    Path(path).write_text("\n".join(lines) + "\n", encoding="latin-1")
    return str(path)


def _orders_prices(tmp_path, cells):
    _write(tmp_path, order_lines=[_order_line(c, i=i) for i, c in enumerate(cells)])
    o = np.asarray(load_omie_orders(_cfg(tmp_path)))
    assert o.shape == (len(cells), 13)
    return [int(v) for v in o[:, ORD_PRICE]]


def _trades_prices(tmp_path, cells):
    _write(tmp_path, trade_lines=[_trade_line(c, i=i) for i, c in enumerate(cells)])
    t = np.asarray(load_omie_trades(_cfg(tmp_path)))
    assert t.shape == (len(cells), 6)
    return [int(v) for v in t[:, TRD_PRICE]]


# ---------------------------------------------------------------------------
# Decoding of Spanish thousands-separated numbers
# ---------------------------------------------------------------------------

class TestOrdersThousandsSeparator:
    def test_spanish_prices_decode(self, tmp_path):
        """Card fixture: main returns 0 EUR for the four thousands-separated cells."""
        cells = ["-9.999,00", "9.999,00", "-1.249,90", "1.000,00", "12,34", "0,00"]
        expected = [-9999.00, 9999.00, -1249.90, 1000.00, 12.34, 0.00]
        assert _orders_prices(tmp_path, cells) == [_pi(p) for p in expected]

    def test_floor_and_cap_orders_are_not_loaded_at_zero(self, tmp_path):
        """A BUY at +9,999 and a SELL at -9,999 keep their price (not 0 EUR)."""
        _write(tmp_path, order_lines=[
            _order_line("9.999,00", side="C", i=0),
            _order_line("-9.999,00", side="V", i=1),
        ])
        o = np.asarray(load_omie_orders(_cfg(tmp_path)))
        assert int(o[0, ORD_PRICE]) == _pi(9999.0) == 1_999_800
        assert int(o[1, ORD_PRICE]) == _pi(-9999.0) == 0
        assert _pi(0.0) not in [int(v) for v in o[:, ORD_PRICE]]

    def test_comma_decimal_and_plain_prices(self, tmp_path):
        """GUARD (green on both): comma decimals, and comma-less cells parsed as-is."""
        cells = ["12,34", "0,00", "-5,5", "100", "12.5", "999,99"]
        expected = [12.34, 0.0, -5.5, 100.0, 12.5, 999.99]
        assert _orders_prices(tmp_path, cells) == [_pi(p) for p in expected]


class TestTradesThousandsSeparator:
    def test_spanish_prices_decode(self, tmp_path):
        cells = ["-1.249,90", "9.999,00", "1.000,00", "12,34"]
        expected = [-1249.90, 9999.00, 1000.00, 12.34]
        assert _trades_prices(tmp_path, cells) == [_pi(p) for p in expected]

    def test_comma_decimal_prices(self, tmp_path):
        """GUARD (green on both)."""
        cells = ["12,34", "0,00", "-5,50", "100"]
        expected = [12.34, 0.0, -5.5, 100.0]
        assert _trades_prices(tmp_path, cells) == [_pi(p) for p in expected]


class TestMarginalpdbcNumbers:
    def test_thousands_separated_col_b(self, tmp_path):
        f = _write_marginalpdbc(tmp_path / "marginalpdbc_20250602.1",
                                {1: "1.234,56", 2: "-1.000,00"})
        curve = parse_marginalpdbc_file(f)
        assert curve[1] == pytest.approx(1234.56)
        assert curve[2] == pytest.approx(-1000.0)

    def test_comma_decimal_col_b(self, tmp_path):
        """GUARD (green on both): existing comma-decimal fixtures keep working."""
        f = _write_marginalpdbc(tmp_path / "marginalpdbc_20250602.1",
                                {1: "36,07", 2: "129,8", 3: "60"})
        curve = parse_marginalpdbc_file(f)
        assert curve == {1: pytest.approx(36.07), 2: pytest.approx(129.8),
                         3: pytest.approx(60.0)}

    def test_dot_decimal_col_b(self, tmp_path):
        """T-DOT (i), GUARD (green on both): a comma-less '.' is a DECIMAL point.

        Fails under an unconditional '.'-strip (36.07 -> 3,607).
        """
        f = _write_marginalpdbc(tmp_path / "marginalpdbc_20250602.1",
                                {1: "36.07", 2: "129.8", 3: "60"})
        curve = parse_marginalpdbc_file(f)
        assert curve == {1: pytest.approx(36.07), 2: pytest.approx(129.8),
                         3: pytest.approx(60.0)}

    @pytest.mark.skipif(
        not (REAL_DATA / "marginalpdbc_20250601.1").is_file(),
        reason="real data/rawOMIE/marginalpdbc_20250601.1 not present",
    )
    def test_real_dot_decimal_file(self):
        """T-DOT (ii), GUARD (green on both): real dot-decimal day-ahead anchors."""
        curve = parse_marginalpdbc_file(str(REAL_DATA / "marginalpdbc_20250601.1"))
        assert curve[1] == pytest.approx(60.0)
        assert curve[2] == pytest.approx(36.07)
        assert curve[3] == pytest.approx(35.01)
        assert set(curve) == set(range(1, 25))
        assert all(abs(v) <= 1000.0 for v in curve.values())  # a "."-strip gives 3,607


# ---------------------------------------------------------------------------
# T-RAISE: the S2-1 / S2-1a raise contract
# ---------------------------------------------------------------------------

class TestErrorClass:
    def test_omie_number_parse_error_is_a_value_error(self):
        cls = _err_cls()
        assert issubclass(cls, ValueError)
        assert cls is not ValueError


def _assert_names(exc, column, raw):
    msg = str(exc.value)
    assert column in msg, f"message must name the column {column!r}: {msg!r}"
    if raw:
        assert raw in msg, f"message must contain the raw value {raw!r}: {msg!r}"


_BAD_PRECIO = ["", "abc", "nan", "inf", "1,234.5", "12.34,5"]
_BAD_PRECIO_IDS = ["empty", "abc", "nan", "inf", "english-grouping", "malformed-grouping"]
_BAD_CANTIDAD = ["", "abc"]
_BAD_CANTIDAD_IDS = ["empty", "abc"]


class TestOrdersRaise:
    @pytest.mark.parametrize("raw", _BAD_PRECIO, ids=_BAD_PRECIO_IDS)
    def test_bad_precio_raises(self, tmp_path, raw):
        err = _err_cls()
        _write(tmp_path, order_lines=[_order_line(raw)])
        with pytest.raises(err) as exc:
            load_omie_orders(_cfg(tmp_path))
        _assert_names(exc, "Precio", raw)

    @pytest.mark.parametrize("raw", _BAD_CANTIDAD, ids=_BAD_CANTIDAD_IDS)
    def test_bad_cantidad_raises(self, tmp_path, raw):
        err = _err_cls()
        _write(tmp_path, order_lines=[_order_line("50,00", cantidad=raw)])
        with pytest.raises(err) as exc:
            load_omie_orders(_cfg(tmp_path))
        _assert_names(exc, "Cantidad", raw)


class TestTradesRaise:
    # An empty trades cell may reach the parser as pandas' 'nan'; for the
    # empty case only the column name is checked.
    @pytest.mark.parametrize("raw", _BAD_PRECIO, ids=_BAD_PRECIO_IDS)
    def test_bad_precio_raises(self, tmp_path, raw):
        err = _err_cls()
        _write(tmp_path, trade_lines=[_trade_line(raw)])
        with pytest.raises(err) as exc:
            load_omie_trades(_cfg(tmp_path))
        _assert_names(exc, "Precio", raw)

    @pytest.mark.parametrize("raw", _BAD_CANTIDAD, ids=_BAD_CANTIDAD_IDS)
    def test_bad_cantidad_raises(self, tmp_path, raw):
        err = _err_cls()
        _write(tmp_path, trade_lines=[_trade_line("50,00", cantidad=raw)])
        with pytest.raises(err) as exc:
            load_omie_trades(_cfg(tmp_path))
        _assert_names(exc, "Cantidad", raw)


class TestMarginalpdbcRaise:
    # No EMPTY col-B case: parse_marginalpdbc_file drops empty ';'-fields
    # before indexing, so an empty col B never reaches the number parser
    # (card T-RAISE (iv)).
    @pytest.mark.parametrize("raw", ["abc", "nan"], ids=["abc", "nan"])
    def test_bad_col_b_raises(self, tmp_path, raw):
        err = _err_cls()
        f = _write_marginalpdbc(tmp_path / "marginalpdbc_20250602.1",
                                {1: "40.0", 2: raw})
        with pytest.raises(err) as exc:
            parse_marginalpdbc_file(f)
        assert raw in str(exc.value)


class TestOptionalColumns:
    def test_empty_optional_columns_default_to_zero(self, tmp_path):
        """GUARD (green on both): empty Cantidad reducida / PPD -> 0.0."""
        _write(tmp_path, order_lines=[_order_line("50,00", cantidad="10,0")])
        o = np.asarray(load_omie_orders(_cfg(tmp_path)))
        assert int(o[0, ORD_PEAK]) == int(encode_qty(0.0, _PC))
        assert int(o[0, ORD_PPD]) == int(encode_ppd(0.0, _PC))

    def test_spanish_optional_columns_decode(self, tmp_path):
        """GUARD (green on both): comma-decimal peak / PPD values."""
        _write(tmp_path, order_lines=[
            _order_line("50,00", cantidad="10,0", peak="5,0", ppd="-0,5"),
        ])
        o = np.asarray(load_omie_orders(_cfg(tmp_path)))
        assert int(o[0, ORD_PEAK]) == int(encode_qty(5.0, _PC))
        assert int(o[0, ORD_PPD]) == int(encode_ppd(-0.5, _PC))

    @pytest.mark.parametrize(
        "column,peak,ppd",
        [("Cantidad reducida", "abc", ""), ("PPD", "5,0", "abc")],
        ids=["cantidad-reducida", "ppd"],
    )
    def test_non_empty_garbage_raises(self, tmp_path, column, peak, ppd):
        err = _err_cls()
        _write(tmp_path, order_lines=[
            _order_line("50,00", cantidad="10,0", peak=peak, ppd=ppd),
        ])
        with pytest.raises(err) as exc:
            load_omie_orders(_cfg(tmp_path))
        _assert_names(exc, column, "abc")


class TestPropagation:
    def test_build_fixed_tape_set_propagates(self, tmp_path):
        """The bad tape raises through the builder (it is not dropped or wrapped)."""
        err = _err_cls()
        good = _write(tmp_path / "good")
        bad = _write(tmp_path / "bad", order_lines=[_order_line("abc")])
        with pytest.raises(err):
            build_fixed_tape_set([_cfg(good), _cfg(bad)])

    def test_build_day_ahead_anchors_propagates(self, tmp_path):
        err = _err_cls()
        _write(tmp_path)  # valid orders for (D, K)
        # anchor(K, D) = marginalpdbc(D, hour K // 4 + 1) = hour 11
        _write_marginalpdbc(
            tmp_path / "marginalpdbc_20250602.1",
            {h: ("abc" if h == K // 4 + 1 else "40.0") for h in range(1, 25)},
        )
        with pytest.raises(err):
            build_day_ahead_anchors([_cfg(tmp_path)])

    def test_build_day_ahead_anchors_positive_control(self, tmp_path):
        """GUARD (green on both): the same fixture with a valid anchor works."""
        _write(tmp_path)
        _write_marginalpdbc(tmp_path / "marginalpdbc_20250602.1",
                            {h: "40.25" for h in range(1, 25)})
        a = np.asarray(build_day_ahead_anchors([_cfg(tmp_path)]))
        assert a.shape == (1,)
        assert float(a[0]) == pytest.approx(40.25)


# ---------------------------------------------------------------------------
# Real product (2025-06-25, slot 0): files orders_20250624.1 + orders_20250625.1
# ---------------------------------------------------------------------------

_REAL_DATE = "2025-06-25"
_REAL_K = 0
_REAL_FILES = ["orders_20250624.1", "orders_20250625.1",
               "trades_20250624.1", "trades_20250625.1"]
_REAL_KEY = "20250625 00:00"


def _have_real():
    return REAL_DATA.is_dir() and all((REAL_DATA / n).is_file() for n in _REAL_FILES)


def _spanish(s):
    """Independent reference parse of a Spanish-format number."""
    return float(s.replace(".", "").replace(",", "."))


def _raw_precio_cells():
    """Raw ``Precio`` cells of product (2025-06-25, 0), both session files."""
    cells = []
    for name in _REAL_FILES[:2]:
        lines = (REAL_DATA / name).read_text(encoding="latin-1").splitlines()
        hdr = next(i for i, ln in enumerate(lines) if ln.startswith("Fecha;Contrato;"))
        for ln in lines[hdr + 1:]:
            f = ln.split(";")
            if len(f) < 8:
                continue
            if f[1].strip().split("-")[0].strip() == _REAL_KEY:
                cells.append(f[5].strip())
    return cells


@pytest.fixture(scope="module")
def real_product():
    if not _have_real():
        pytest.skip("real OMIE data not present under data/rawOMIE")
    cfg = OMIEDataConfig(data_path=str(REAL_DATA), contract_id=_REAL_K,
                         delivery_date=_REAL_DATE)
    o = np.asarray(load_omie_orders(cfg))
    return o, _raw_precio_cells()


@pytest.mark.slow
class TestRealProduct20250625:
    """slow: parses two real 30-45 MB orders session files (shared module load)."""

    def test_reference_has_thousands_separated_cells(self, real_product):
        """GUARD: the product spans the defect (the RED items below are not null)."""
        o, cells = real_product
        assert o.shape[0] == len(cells) > 0
        assert any("." in c and "," in c for c in cells)

    def test_no_thousands_separated_row_decodes_to_zero(self, real_product):
        """Card RED item: zero-price rows == raw cells whose value is exactly 0."""
        o, cells = real_product
        loaded_zero = int(np.sum(o[:, ORD_PRICE] == _pi(0.0)))
        raw_zero = sum(1 for c in cells if _spanish(c) == 0.0)
        assert loaded_zero == raw_zero

    def test_price_multiset_matches_independent_reference(self, real_product):
        o, cells = real_product
        loaded = collections.Counter(int(v) for v in o[:, ORD_PRICE])
        ref = collections.Counter(_pi(_spanish(c)) for c in cells)
        assert loaded == ref
