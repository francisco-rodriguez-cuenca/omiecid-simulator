"""Locked behavioural contract tests for OMIE-TAPE-CONTRACT-DATE-1.

A "(delivery date D, contract k)" tape is ONE delivery product, gathered from
every session file it spans (files D-1 and D ONLY), on a monotone timestamp
axis with no day wrap.

Sources (read in this order):
  * docs/omiecid/agile-plan.md, card "Story OMIE-TAPE-CONTRACT-DATE-1" (v5.160):
    Decided design L1 (i)-(vi) and the Test-first contract TC-1..TC-9 (+TC-1b,
    B-1 12-case TC-6, B-2 zero real-data skips).
  * docs/omiecid/market-spec.md v2.10 §13 / §13.1 / §13.2 (the file date and
    ``Fecha`` are the SESSION day: file ``{orders,trades}_D.1`` holds every row
    sent / matched in [22:20 D-1, 22:00 D)), §13.4 (product = (Contrato start
    date, slot); split across files D-1 and D only), §3.1 (hour-aligned GCT:
    GCT(k) = 60*floor(k/4) - 60 min from D 00:00), OQ-29 (incomplete products
    are excluded), OQ-30 (DST, open).

Public contract pinned here (the Implementation Agent codes against this):

  * ``load_omie_orders(cfg)`` / ``load_omie_trades(cfg)`` /
    ``load_omie_agent_metadata(cfg)`` for ``cfg = OMIEDataConfig(D, k)`` read
    exactly the rows whose ``Contrato`` START is (D, slot k), from session
    files ``*_{D-1}.1`` and ``*_{D}.1``. File ``*_{D+1}.1`` is never opened.
  * Timestamp column (orders col 6, trades col 5) = seconds since D-1 00:00
    local wall clock (== signed seconds from D 00:00 + 86,400), NO modulo.
    Real data lies in [55,200, 165,600).
  * Orders: file D-1 rows then file D rows, stable-sorted by timestamp;
    ``order_id`` = 1..N in that order. Trades: file D-1 rows then file D rows
    in file order, NO sort.
  * The (Agente, Unidad) pair universe is taken over the same product rows
    (both files, orders + trades); one ``agent_id`` per pair.
  * ``IncompleteProductError`` — a module-level ``ValueError`` subclass in
    ``omiecid_simulator.jaxomiecid.omie_data_loader`` — is raised by all three
    loaders when ANY of the four files ``orders_{D-1}.1``, ``orders_D.1``,
    ``trades_{D-1}.1``, ``trades_D.1`` is missing, BEFORE any tolerant parse.
    Its message names the missing file and ``OQ-29``. ``build_fixed_tape_set``
    and ``build_day_ahead_anchors`` propagate it.
  * DST guard: when D is a Europe/Madrid transition day the loader raises a
    ``ValueError`` (message names the DST / transition / OQ-30 cause) unless
    the new ``OMIEDataConfig.allow_dst_transition`` field (default ``False``)
    is ``True``. Under the opt-in, a date-D ``Contrato`` carrying an A/B hour
    suffix raises a ``ValueError`` (no slot mapping is invented).
  * Prefix-first product filter: neighbouring-day ``02:00A`` / ``02:00B`` rows
    inside a session file do not break product D.
  * NO send-window validation in the loader (F-4): the synthetic fixtures here
    deliberately stay inside the real window only for readability; the
    loader must not clip.

TC -> test map and expected status on the CURRENT (unfixed) loader:
  TC-1   TestRealProduct20250625                           RED
  TC-1b  TestRealProduct20250601                           RED
  TC-2   TestRealTimestampAxis                             RED
  TC-3   TestOrdersOneProductPerTape (synthetic)           RED (D+1 guards GREEN)
  TC-4   TestTradesOneProductPerTape (synthetic)           RED (D+1 guard GREEN)
  TC-5   TestAgentIdentityAcrossFiles (synthetic)          RED
  TC-6   TestIncompleteProduct (12 cases + propagation)    RED (positive control GREEN)
  TC-6r  TestIncompleteProductRealData                     RED
  TC-7   TestDstGuard (synthetic + real 2025-10-26)        RED
  TC-8   TestRealByteIdentityOfPreviousDaySends            GREEN on both (guard)
  TC-9   TestPrefixFirstFilter (synthetic + real)          RED

Runtime: synthetic tests are tmp_path CSVs (< 1 s each). Real-data tests are
``@pytest.mark.slow`` (each parses one or two 30-45 MB OMIE session files) and
share module-scoped loads. They skip only when ``data/rawOMIE`` is absent; the
Step-3/4/6 evidence must show 0 skips (card B-2).
"""

from __future__ import annotations

import collections
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import omiecid_simulator.jaxomiecid.omie_data_loader as odl
from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    build_day_ahead_anchors,
    build_fixed_tape_set,
    load_omie_agent_metadata,
    load_omie_orders,
    load_omie_trades,
)
from omiecid_simulator.omiecid.encoding import encode_price
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORD_CID, ORD_PRICE, ORD_OID, ORD_AGENT, ORD_TS = 0, 2, 4, 5, 6
TRD_CID, TRD_PRICE, TRD_BUY, TRD_SELL, TRD_TS = 0, 1, 3, 4, 5

# Real-data send window of any product, seconds since D-1 00:00 (§13.4):
# first sends 15:20 D-1; last GCT (k = 92..95) 22:00 D.
WINDOW_LO = 55_200
WINDOW_HI = 165_600

_PC = OMIEProductConfig()

_REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA = _REPO_ROOT / "data" / "rawOMIE"

_DST_MSG = re.compile(r"(?i)(dst|daylight|transition|OQ-30|allow_dst_transition)")
_SUFFIX_MSG = re.compile(r"(?i)(OQ-30|suffix|02:00[AB]|dst|daylight|transition)")

# Synthetic fixture product day and its neighbours (not a DST day).
D = "2025-06-02"
D_PREV = "2025-06-01"
D_NEXT = "2025-06-03"


# ---------------------------------------------------------------------------
# Accessors for names that the fix introduces (keep the module collectible on
# the unfixed loader; each test fails with a precise message instead).
# ---------------------------------------------------------------------------

def _incomplete_cls():
    cls = getattr(odl, "IncompleteProductError", None)
    if cls is None:
        pytest.fail(
            "omie_data_loader.IncompleteProductError is missing "
            "(card L1 (iv): a named, module-level ValueError subclass)"
        )
    assert issubclass(cls, ValueError), "IncompleteProductError must subclass ValueError"
    return cls


def _cfg(data_path, contract_id, delivery_date, allow_dst=None):
    kw = dict(
        data_path=str(data_path),
        contract_id=int(contract_id),
        delivery_date=delivery_date,
        cache_path=str(data_path),
    )
    if allow_dst is not None:
        kw["allow_dst_transition"] = allow_dst
    try:
        return OMIEDataConfig(**kw)
    except TypeError as exc:
        pytest.fail(
            f"OMIEDataConfig does not accept allow_dst_transition (card L1 (v)): {exc}"
        )


# ---------------------------------------------------------------------------
# Synthetic session-file writers (unaccented headers, as every tmp_path fixture
# in this suite; the accented real header is OMIE-TRADES-MATCH-TS-HEADER-1).
# ---------------------------------------------------------------------------

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
_ZONE = "10YES-REE------0"


def _nodash(iso):
    return iso.replace("-", "")


def _dmy(iso):
    y, m, d = iso.split("-")
    return f"{d}/{m}/{y}"


def contrato(date_iso, k, suffix=""):
    """Quarter-hour Contrato string for (date, slot k); optional DST suffix."""
    h, m = divmod(15 * k, 60)
    h2, m2 = divmod(15 * k + 15, 60)
    end_date = date_iso
    if h2 == 24:
        h2, m2 = 0, 0
        end_date = datetime.fromordinal(
            datetime.fromisoformat(date_iso).toordinal() + 1
        ).date().isoformat()
    return (
        f"{_nodash(date_iso)} {h:02d}:{m:02d}{suffix}-"
        f"{_nodash(end_date)} {h2:02d}:{m2:02d}{suffix}"
    )


def order(product_date, k, price, send, agente="ABC", unidad="U1", side="V",
          suffix="", qty="1,00"):
    """One order row spec. ``send`` = 'DD/MM/YYYY HH:MM:SS'."""
    return dict(c=contrato(product_date, k, suffix), ag=agente, un=unidad,
                price=price, qty=qty, side=side, send=send)


def trade(product_date, k, price, matched, buy=("BYR", "BU1"), sell=("SLR", "SU1"),
          suffix="", qty="1,00"):
    return dict(c=contrato(product_date, k, suffix), buy=buy, sell=sell,
                price=price, qty=qty, matched=matched)


def _fmt_price(p):
    return f"{p:.2f}".replace(".", ",")


def write_orders(dirpath, session_iso, rows):
    lines = [_ORDERS_META, "", _ORDERS_COLS]
    for r in rows:
        lines.append(
            f"{_dmy(session_iso)};{r['c']};{_ZONE};{r['ag']};{r['un']};"
            f"{_fmt_price(r['price'])};{r['qty']};{r['side']};;;;;{r['send']};"
        )
    p = Path(dirpath) / f"orders_{_nodash(session_iso)}.1"
    p.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return p


def write_trades(dirpath, session_iso, rows):
    lines = [_TRADES_META, _TRADES_COLS]
    for r in rows:
        lines.append(
            f"{_dmy(session_iso)};{r['c']};{r['buy'][0]};{r['buy'][1]};{_ZONE};"
            f"{r['sell'][0]};{r['sell'][1]};{_ZONE};"
            f"{_fmt_price(r['price'])};{r['qty']};{r['matched']};"
        )
    p = Path(dirpath) / f"trades_{_nodash(session_iso)}.1"
    p.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return p


def _pi(price_eur):
    return int(encode_price(price_eur, _PC))


def _prices(arr, col):
    """Decode the tag of every row: the encoded price -> the EUR tag."""
    return [int(v) for v in np.asarray(arr)[:, col]]


def _tags(arr, col, tag_of):
    inv = {v: k for k, v in tag_of.items()}
    out = []
    for v in _prices(arr, col):
        out.append(inv.get(v, f"<unknown price_int {v}>"))
    return out


# ---------------------------------------------------------------------------
# The TC-3/4/5 three-session fixture (product day D = 2025-06-02)
#
# Session file D-1 (0601) covers sends [22:20 05-31, 22:00 06-01);
# session file D   (0602) covers sends [22:20 06-01, 22:00 06-02);
# session file D+1 (0603) covers sends [22:20 06-02, 22:00 06-03).
# Each row carries a unique price that serves as its tag.
# ---------------------------------------------------------------------------

ORDER_TAGS = {
    # product (D, 0): GCT 23:00 on D-1
    "A1": 10.01, "A2": 10.02, "A3": 10.03, "A4": 10.04,
    # product (D, 8): GCT 01:00 on D
    "B0": 20.00, "B1": 20.01, "B2": 20.02,
    # product (D, 94): GCT 22:00 on D
    "C1": 30.01, "C2": 30.02, "C3": 30.03, "C4_PLANTED": 30.99,
    # other products sharing slot 0 / slot 94 numbers
    "N1_prev0": 40.01,   # product (D-1, 0) in file D-1
    "N2_next0": 40.02,   # product (D+1, 0) in file D
    "N3_next0": 40.03,   # product (D+1, 0) in file D
    "N5_next0": 40.05,   # product (D+1, 0) in file D+1
    "N6_prev94": 40.06,  # product (D-1, 94) in file D-1
}
ORDER_TAGS_INT = {k: _pi(v) for k, v in ORDER_TAGS.items()}

TRADE_TAGS = {
    "T1": 11.01, "T2": 11.02, "T2b": 11.03,       # product (D, 0)
    "T3": 21.01,                                  # product (D, 8)
    "T8": 31.01, "T5": 31.02, "T6_PLANTED": 31.99,  # product (D, 94)
    "TN_prev0": 41.01, "T4_next0": 41.02, "T7_next0": 41.03,
}
TRADE_TAGS_INT = {k: _pi(v) for k, v in TRADE_TAGS.items()}


def _write_three_sessions(dirpath, *, with_next=True):
    O, T = ORDER_TAGS, TRADE_TAGS
    write_orders(dirpath, D_PREV, [
        order(D, 0, O["A1"], "01/06/2025 15:30:00", agente="AGX", unidad="UX1"),
        order(D_PREV, 0, O["N1_prev0"], "31/05/2025 22:30:00"),
        order(D, 8, O["B0"], "01/06/2025 17:00:00"),
        order(D, 0, O["A2"], "01/06/2025 20:00:00", agente="ABC", unidad="U1"),
        order(D, 94, O["C1"], "01/06/2025 16:00:00"),
        order(D_PREV, 94, O["N6_prev94"], "01/06/2025 18:00:00"),
    ])
    write_orders(dirpath, D, [
        order(D, 0, O["A3"], "01/06/2025 22:25:00", agente="ABC", unidad="U1"),
        order(D, 0, O["A4"], "01/06/2025 22:50:00"),
        order(D, 8, O["B1"], "01/06/2025 23:30:00"),
        order(D, 8, O["B2"], "02/06/2025 00:30:00"),
        order(D_NEXT, 0, O["N2_next0"], "02/06/2025 16:00:00", agente="NOI", unidad="NU1"),
        order(D, 94, O["C2"], "02/06/2025 10:00:00"),
        order(D_NEXT, 0, O["N3_next0"], "02/06/2025 21:00:00"),
        order(D, 94, O["C3"], "02/06/2025 21:59:00"),
    ])
    write_trades(dirpath, D_PREV, [
        trade(D, 0, T["T1"], "01/06/2025 18:00:00", buy=("TBX", "TBU1"), sell=("ABC", "U1")),
        trade(D_PREV, 0, T["TN_prev0"], "31/05/2025 22:40:00"),
        trade(D, 94, T["T8"], "01/06/2025 19:00:00"),
    ])
    write_trades(dirpath, D, [
        # listed out of time order on purpose: trades are NOT sorted
        trade(D, 0, T["T2"], "01/06/2025 22:40:00"),
        trade(D, 0, T["T2b"], "01/06/2025 22:30:00"),
        trade(D, 8, T["T3"], "02/06/2025 00:40:00"),
        trade(D_NEXT, 0, T["T4_next0"], "02/06/2025 17:00:00", buy=("NOT", "NTU1")),
        trade(D, 94, T["T5"], "02/06/2025 21:30:00"),
    ])
    if with_next:
        write_orders(dirpath, D_NEXT, [
            # physically impossible planted product-(D, 94) row: must never load
            order(D, 94, O["C4_PLANTED"], "02/06/2025 22:30:00"),
            order(D_NEXT, 0, O["N5_next0"], "02/06/2025 22:25:00"),
        ])
        write_trades(dirpath, D_NEXT, [
            trade(D, 94, T["T6_PLANTED"], "02/06/2025 22:30:00"),
            trade(D_NEXT, 0, T["T7_next0"], "02/06/2025 22:30:00"),
        ])


@pytest.fixture
def three_sessions(tmp_path):
    _write_three_sessions(tmp_path, with_next=True)
    return tmp_path


@pytest.fixture
def two_sessions_no_next(tmp_path):
    _write_three_sessions(tmp_path, with_next=False)
    return tmp_path


# ===========================================================================
# TC-3 — orders: one product per tape, monotone axis, file D+1 never read
# ===========================================================================


class TestOrdersOneProductPerTape:
    """TC-3 (synthetic). RED on the unfixed loader except the two D+1 guards."""

    def test_contract0_gets_exactly_its_product_rows_from_files_prev_and_d(self, three_sessions):
        arr = load_omie_orders(_cfg(three_sessions, 0, D))
        assert _tags(arr, ORD_PRICE, ORDER_TAGS_INT) == ["A1", "A2", "A3", "A4"]

    def test_contract0_timestamps_are_seconds_since_prev_midnight(self, three_sessions):
        arr = np.asarray(load_omie_orders(_cfg(three_sessions, 0, D)))
        # 15:30, 20:00, 22:25, 22:50 on D-1 -> seconds since D-1 00:00
        assert arr[:, ORD_TS].tolist() == [55_800, 72_000, 80_700, 82_200]

    def test_contract8_spans_midnight_without_wrap(self, three_sessions):
        """Sends 17:00 D-1, 23:30 D-1, 00:30 D are stored 61,200 / 84,600 / 88,200."""
        arr = np.asarray(load_omie_orders(_cfg(three_sessions, 8, D)))
        assert _tags(arr, ORD_PRICE, ORDER_TAGS_INT) == ["B0", "B1", "B2"]
        assert arr[:, ORD_TS].tolist() == [61_200, 84_600, 88_200]

    def test_contract94_last_send_stored_below_window_bound(self, three_sessions):
        """Last send 21:59:00 on D -> 165,540 < 165,600 (GCT of k=94)."""
        arr = np.asarray(load_omie_orders(_cfg(three_sessions, 94, D)))
        assert _tags(arr, ORD_PRICE, ORDER_TAGS_INT) == ["C1", "C2", "C3"]
        assert arr[:, ORD_TS].tolist() == [57_600, 122_400, 165_540]
        assert int(arr[:, ORD_TS].max()) < WINDOW_HI

    def test_next_day_product_gets_its_own_rows(self, three_sessions):
        """Product (D+1, 0) = rows of file D and file D+1, never product D's."""
        arr = np.asarray(load_omie_orders(_cfg(three_sessions, 0, D_NEXT)))
        assert _tags(arr, ORD_PRICE, ORDER_TAGS_INT) == ["N2_next0", "N3_next0", "N5_next0"]
        # 16:00 and 21:00 on D, 22:25 on D -> seconds since D 00:00 (= D+1's D-1)
        assert arr[:, ORD_TS].tolist() == [57_600, 75_600, 80_700]

    def test_order_ids_are_one_to_n_in_timestamp_order(self, three_sessions):
        arr = np.asarray(load_omie_orders(_cfg(three_sessions, 8, D)))
        assert arr[:, ORD_OID].tolist() == [1, 2, 3]
        assert np.all(np.diff(arr[:, ORD_TS]) >= 0)

    def test_contract_id_column_equals_requested_slot(self, three_sessions):
        for k in (0, 8, 94):
            arr = np.asarray(load_omie_orders(_cfg(three_sessions, k, D)))
            assert arr.shape[0] > 0
            assert np.all(arr[:, ORD_CID] == k)

    def test_planted_product_d_row_in_file_d_plus_1_is_never_loaded(self, three_sessions):
        """GUARD (green on the unfixed loader): refutes the 'k >= 93 reads D+1' rule."""
        arr = load_omie_orders(_cfg(three_sessions, 94, D))
        assert "C4_PLANTED" not in _tags(arr, ORD_PRICE, ORDER_TAGS_INT)

    def test_contract94_loads_with_file_d_plus_1_absent(self, two_sessions_no_next):
        """GUARD (green on the unfixed loader): file D+1 is never opened."""
        arr = load_omie_orders(_cfg(two_sessions_no_next, 94, D))
        assert arr.shape[0] > 0
        assert "C4_PLANTED" not in _tags(arr, ORD_PRICE, ORDER_TAGS_INT)

    def test_same_second_ties_keep_file_order(self, tmp_path):
        """Stable sort (card L1 (i)): same-second rows keep file order,
        file D-1 rows before file D rows."""
        tie = "01/06/2025 22:30:00"
        write_orders(tmp_path, D_PREV, [
            order(D, 0, 50.01, "01/06/2025 21:59:59"),
            order(D, 0, 50.02, "01/06/2025 21:59:59"),
        ])
        write_orders(tmp_path, D, [order(D, 0, p, tie) for p in (50.13, 50.11, 50.12)])
        write_trades(tmp_path, D_PREV, [])
        write_trades(tmp_path, D, [])
        arr = np.asarray(load_omie_orders(_cfg(tmp_path, 0, D)))
        assert _prices(arr, ORD_PRICE) == [_pi(p) for p in (50.01, 50.02, 50.13, 50.11, 50.12)]
        assert arr[:, ORD_OID].tolist() == [1, 2, 3, 4, 5]


# ===========================================================================
# TC-4 — trades: same union, file order, no sort, no wrap
# ===========================================================================


class TestTradesOneProductPerTape:
    """TC-4 (synthetic, unaccented ``Momento casacion`` header). RED except the D+1 guard."""

    def test_contract0_trades_are_both_files_in_file_order(self, three_sessions):
        arr = load_omie_trades(_cfg(three_sessions, 0, D))
        assert _tags(arr, TRD_PRICE, TRADE_TAGS_INT) == ["T1", "T2", "T2b"]

    def test_contract0_trade_timestamps_are_seconds_since_prev_midnight_unsorted(self, three_sessions):
        arr = np.asarray(load_omie_trades(_cfg(three_sessions, 0, D)))
        # 18:00, 22:40, 22:30 on D-1 (file order kept, not sorted)
        assert arr[:, TRD_TS].tolist() == [64_800, 81_600, 81_000]

    def test_contract8_trade_after_midnight_is_not_wrapped(self, three_sessions):
        arr = np.asarray(load_omie_trades(_cfg(three_sessions, 8, D)))
        assert _tags(arr, TRD_PRICE, TRADE_TAGS_INT) == ["T3"]
        assert arr[:, TRD_TS].tolist() == [88_800]

    def test_contract94_trades(self, three_sessions):
        arr = np.asarray(load_omie_trades(_cfg(three_sessions, 94, D)))
        assert _tags(arr, TRD_PRICE, TRADE_TAGS_INT) == ["T8", "T5"]
        assert arr[:, TRD_TS].tolist() == [68_400, 163_800]

    def test_next_day_product_trades(self, three_sessions):
        arr = np.asarray(load_omie_trades(_cfg(three_sessions, 0, D_NEXT)))
        assert _tags(arr, TRD_PRICE, TRADE_TAGS_INT) == ["T4_next0", "T7_next0"]
        assert arr[:, TRD_TS].tolist() == [61_200, 81_000]

    def test_planted_product_d_trade_in_file_d_plus_1_is_never_loaded(self, three_sessions):
        """GUARD (green on the unfixed loader)."""
        arr = load_omie_trades(_cfg(three_sessions, 94, D))
        assert "T6_PLANTED" not in _tags(arr, TRD_PRICE, TRADE_TAGS_INT)

    def test_contract94_trades_load_with_file_d_plus_1_absent(self, two_sessions_no_next):
        arr = np.asarray(load_omie_trades(_cfg(two_sessions_no_next, 94, D)))
        assert _tags(arr, TRD_PRICE, TRADE_TAGS_INT) == ["T8", "T5"]


# ===========================================================================
# TC-5 — one agent_id per (Agente, Unidad) across files, orders AND trades
# ===========================================================================


class TestAgentIdentityAcrossFiles:
    """TC-5 (synthetic). RED: pairs seen only in file D-1 are unknown today."""

    def test_order_pair_seen_only_in_prev_file_is_in_metadata_and_on_its_row(self, three_sessions):
        cfg = _cfg(three_sessions, 0, D)
        md = load_omie_agent_metadata(cfg)
        arr = np.asarray(load_omie_orders(cfg))
        assert ("AGX", "UX1") in md.values()
        tags = _tags(arr, ORD_PRICE, ORDER_TAGS_INT)
        a1 = arr[tags.index("A1")]
        assert md[int(a1[ORD_AGENT])] == ("AGX", "UX1")

    def test_pair_present_in_both_files_gets_one_agent_id(self, three_sessions):
        cfg = _cfg(three_sessions, 0, D)
        md = load_omie_agent_metadata(cfg)
        arr = np.asarray(load_omie_orders(cfg))
        tags = _tags(arr, ORD_PRICE, ORDER_TAGS_INT)
        id_a2 = int(arr[tags.index("A2")][ORD_AGENT])
        id_a3 = int(arr[tags.index("A3")][ORD_AGENT])
        assert id_a2 == id_a3
        assert md[id_a2] == ("ABC", "U1")

    def test_trade_only_pair_seen_only_in_prev_file_maps_on_trades(self, three_sessions):
        cfg = _cfg(three_sessions, 0, D)
        md = load_omie_agent_metadata(cfg)
        trd = np.asarray(load_omie_trades(cfg))
        tags = _tags(trd, TRD_PRICE, TRADE_TAGS_INT)
        t1 = trd[tags.index("T1")]
        assert md[int(t1[TRD_BUY])] == ("TBX", "TBU1")
        # the sell leg (ABC, U1) shares the orders' agent_id for that pair
        orders = np.asarray(load_omie_orders(cfg))
        otags = _tags(orders, ORD_PRICE, ORDER_TAGS_INT)
        assert int(t1[TRD_SELL]) == int(orders[otags.index("A2")][ORD_AGENT])

    def test_pair_universe_excludes_other_products_sharing_the_slot(self, three_sessions):
        """The universe is taken over the product's rows only (same union and
        filter as the loaders): pairs that trade only product (D+1, 0) are absent."""
        md = load_omie_agent_metadata(_cfg(three_sessions, 0, D))
        assert ("NOI", "NU1") not in md.values()
        assert ("NOT", "NTU1") not in md.values()

    def test_metadata_is_a_dense_bijection(self, three_sessions):
        md = load_omie_agent_metadata(_cfg(three_sessions, 0, D))
        assert sorted(md.keys()) == list(range(1, len(md) + 1))
        assert len(set(md.values())) == len(md)
        assert set(md.values()) == {
            ("AGX", "UX1"), ("ABC", "U1"),      # orders (D-1 and D)
            ("TBX", "TBU1"),                     # trade buy leg, D-1 only
            ("BYR", "BU1"), ("SLR", "SU1"),      # default trade legs
        }


# ===========================================================================
# TC-6 — incomplete product raises IncompleteProductError (B-1: 12 cases)
# ===========================================================================

_LOADERS = {
    "load_omie_orders": load_omie_orders,
    "load_omie_trades": load_omie_trades,
    "load_omie_agent_metadata": load_omie_agent_metadata,
}
_REQUIRED_FILES = (
    f"orders_{_nodash(D_PREV)}.1",
    f"orders_{_nodash(D)}.1",
    f"trades_{_nodash(D_PREV)}.1",
    f"trades_{_nodash(D)}.1",
)


def _write_complete_product(dirpath):
    write_orders(dirpath, D_PREV, [order(D, 0, 60.01, "01/06/2025 16:00:00")])
    write_orders(dirpath, D, [order(D, 0, 60.02, "01/06/2025 22:30:00")])
    write_trades(dirpath, D_PREV, [trade(D, 0, 60.01, "01/06/2025 16:05:00")])
    write_trades(dirpath, D, [trade(D, 0, 60.02, "01/06/2025 22:35:00")])


class TestIncompleteProduct:
    """TC-6 (synthetic). RED: the unfixed loader silently tolerates a missing file."""

    @pytest.mark.parametrize("loader_name", sorted(_LOADERS))
    def test_complete_product_loads_positive_control(self, tmp_path, loader_name):
        """Anchors the 12 raise cases: the same fixture with all four files loads."""
        _write_complete_product(tmp_path)
        out = _LOADERS[loader_name](_cfg(tmp_path, 0, D))
        assert len(out) > 0

    @pytest.mark.parametrize("missing", _REQUIRED_FILES)
    @pytest.mark.parametrize("loader_name", sorted(_LOADERS))
    def test_each_missing_file_raises_for_each_loader(self, tmp_path, loader_name, missing):
        err = _incomplete_cls()
        _write_complete_product(tmp_path)
        os.remove(tmp_path / missing)
        with pytest.raises(err) as exc:
            _LOADERS[loader_name](_cfg(tmp_path, 0, D))
        msg = str(exc.value)
        assert missing in msg, f"message must name the missing file {missing!r}: {msg!r}"
        assert "OQ-29" in msg, f"message must cite OQ-29: {msg!r}"

    def test_check_precedes_contract_filter_for_an_empty_slot(self, tmp_path):
        """Completeness is checked before any parse: a slot with no rows still raises."""
        err = _incomplete_cls()
        _write_complete_product(tmp_path)
        os.remove(tmp_path / f"trades_{_nodash(D_PREV)}.1")
        with pytest.raises(err):
            load_omie_orders(_cfg(tmp_path, 50, D))

    def test_build_fixed_tape_set_propagates_and_never_drops(self, tmp_path):
        err = _incomplete_cls()
        good = tmp_path / "good"
        bad = tmp_path / "bad"
        good.mkdir()
        bad.mkdir()
        _write_complete_product(good)
        _write_complete_product(bad)
        os.remove(bad / f"orders_{_nodash(D_PREV)}.1")
        with pytest.raises(err):
            build_fixed_tape_set([_cfg(good, 0, D), _cfg(bad, 0, D)])

    def test_build_day_ahead_anchors_propagates(self, tmp_path):
        err = _incomplete_cls()
        _write_complete_product(tmp_path)
        os.remove(tmp_path / f"orders_{_nodash(D_PREV)}.1")
        with pytest.raises(err):
            build_day_ahead_anchors([_cfg(tmp_path, 0, D)])


def _have(*names):
    return REAL_DATA.is_dir() and all((REAL_DATA / n).is_file() for n in names)


_NO_DATA = "real OMIE data not present under data/rawOMIE (card B-2: must not skip at Step 3/4/6)"


@pytest.mark.slow
class TestIncompleteProductRealData:
    """TC-6 real variant. slow: the unfixed loader parses a full session file.
    RED: today both products load half a product silently."""

    @pytest.mark.skipif(not _have("orders_20260301.1", "trades_20260228.1"), reason=_NO_DATA)
    def test_20260301_raises_naming_orders_20260228(self):
        err = _incomplete_cls()
        with pytest.raises(err, match=r"orders_20260228\.1"):
            load_omie_orders(_cfg(REAL_DATA, 0, "2026-03-01"))

    @pytest.mark.skipif(not _have("orders_20250531.1", "trades_20250531.1"), reason=_NO_DATA)
    def test_20250531_raises_naming_orders_20250530(self):
        err = _incomplete_cls()
        with pytest.raises(err, match=r"orders_20250530\.1"):
            load_omie_orders(_cfg(REAL_DATA, 0, "2025-05-31"))


# ===========================================================================
# TC-7 — DST guard
# ===========================================================================

SPRING = "2026-03-29"       # Europe/Madrid spring-forward day (no A/B strings)
SPRING_PREV = "2026-03-28"
FALL = "2025-10-26"         # Europe/Madrid fall-back day (02:00A / 02:00B)
FALL_PREV = "2025-10-25"


def _write_spring_day(dirpath):
    write_orders(dirpath, SPRING_PREV, [order(SPRING, 0, 70.01, "28/03/2026 16:00:00")])
    write_orders(dirpath, SPRING, [order(SPRING, 0, 70.02, "28/03/2026 22:40:00")])
    write_trades(dirpath, SPRING_PREV, [trade(SPRING, 0, 70.01, "28/03/2026 16:10:00")])
    write_trades(dirpath, SPRING, [trade(SPRING, 0, 70.02, "28/03/2026 22:45:00")])


class TestDstGuard:
    """TC-7. RED: no guard, no ``allow_dst_transition`` field today."""

    def test_allow_dst_transition_field_defaults_false(self, tmp_path):
        cfg = OMIEDataConfig(data_path=str(tmp_path), contract_id=0, delivery_date=D)
        assert getattr(cfg, "allow_dst_transition", None) is False

    @pytest.mark.parametrize("loader_name", ["load_omie_orders", "load_omie_trades"])
    def test_transition_day_raises_by_default(self, tmp_path, loader_name):
        _write_spring_day(tmp_path)
        with pytest.raises(ValueError, match=_DST_MSG):
            _LOADERS[loader_name](_cfg(tmp_path, 0, SPRING))

    def test_opt_in_loads_suffix_free_transition_day(self, tmp_path):
        _write_spring_day(tmp_path)
        cfg = _cfg(tmp_path, 0, SPRING, allow_dst=True)
        arr = np.asarray(load_omie_orders(cfg))
        assert _prices(arr, ORD_PRICE) == [_pi(70.01), _pi(70.02)]
        # 16:00 and 22:40 on D-1, local wall clock, seconds since D-1 00:00
        assert arr[:, ORD_TS].tolist() == [57_600, 81_600]
        trd = np.asarray(load_omie_trades(cfg))
        assert _prices(trd, TRD_PRICE) == [_pi(70.01), _pi(70.02)]

    def test_opt_in_with_ab_suffixed_date_d_contrato_raises(self, tmp_path):
        write_orders(tmp_path, FALL_PREV, [order(FALL, 8, 71.01, "25/10/2025 16:00:00")])
        write_orders(tmp_path, FALL, [
            order(FALL, 8, 71.02, "25/10/2025 23:00:00", suffix="A"),
            order(FALL, 8, 71.03, "25/10/2025 23:30:00", suffix="B"),
        ])
        write_trades(tmp_path, FALL_PREV, [])
        write_trades(tmp_path, FALL, [])
        with pytest.raises(ValueError, match=_SUFFIX_MSG):
            load_omie_orders(_cfg(tmp_path, 8, FALL, allow_dst=True))

    @pytest.mark.slow
    @pytest.mark.skipif(
        not _have("orders_20251025.1", "orders_20251026.1",
                  "trades_20251025.1", "trades_20251026.1"),
        reason=_NO_DATA,
    )
    def test_real_20251026_raises_dst_guard(self):
        """slow: real session files (the unfixed loader parses them first)."""
        with pytest.raises(ValueError, match=_DST_MSG):
            load_omie_orders(_cfg(REAL_DATA, 0, "2025-10-26"))


# ===========================================================================
# TC-9 — prefix-first product filter (neighbouring-day A/B strings)
# ===========================================================================


class TestPrefixFirstFilter:
    """TC-9 (synthetic). RED: the unfixed loader parses every slot and raises
    on the neighbour's ``02:00A`` string (2025-10-25), and reads one file."""

    def test_product_before_fall_back_day_loads(self, tmp_path):
        # Product (10-25, 0) = files 1024 + 1025; file 1025 also holds
        # product-10-26 rows with A/B strings (sent on 10-25).
        write_orders(tmp_path, "2025-10-24", [order("2025-10-25", 0, 72.01, "24/10/2025 16:00:00")])
        write_orders(tmp_path, "2025-10-25", [
            order("2025-10-25", 0, 72.02, "24/10/2025 22:30:00"),
            order(FALL, 8, 72.90, "25/10/2025 17:00:00", suffix="A"),
            order(FALL, 8, 72.91, "25/10/2025 17:05:00", suffix="B"),
        ])
        write_trades(tmp_path, "2025-10-24", [trade("2025-10-25", 0, 72.01, "24/10/2025 16:05:00")])
        write_trades(tmp_path, "2025-10-25", [
            trade("2025-10-25", 0, 72.02, "24/10/2025 22:35:00"),
            trade(FALL, 8, 72.90, "25/10/2025 17:10:00", suffix="A"),
        ])
        cfg = _cfg(tmp_path, 0, "2025-10-25")
        arr = np.asarray(load_omie_orders(cfg))
        assert _prices(arr, ORD_PRICE) == [_pi(72.01), _pi(72.02)]
        trd = np.asarray(load_omie_trades(cfg))
        assert _prices(trd, TRD_PRICE) == [_pi(72.01), _pi(72.02)]
        assert len(load_omie_agent_metadata(cfg)) > 0

    def test_product_after_fall_back_day_loads(self, tmp_path):
        # Product (10-27, 0) = files 1026 + 1027; file 1026 holds product-10-26
        # A/B rows (sent 10-25 22:20 .. 10-26 22:00).
        write_orders(tmp_path, FALL, [
            order(FALL, 8, 73.90, "25/10/2025 23:10:00", suffix="A"),
            order("2025-10-27", 0, 73.01, "26/10/2025 16:00:00"),
            order(FALL, 9, 73.91, "25/10/2025 23:20:00", suffix="B"),
        ])
        write_orders(tmp_path, "2025-10-27", [order("2025-10-27", 0, 73.02, "26/10/2025 22:30:00")])
        write_trades(tmp_path, FALL, [
            trade(FALL, 8, 73.90, "25/10/2025 23:15:00", suffix="B"),
            trade("2025-10-27", 0, 73.01, "26/10/2025 16:05:00"),
        ])
        write_trades(tmp_path, "2025-10-27", [trade("2025-10-27", 0, 73.02, "26/10/2025 22:35:00")])
        cfg = _cfg(tmp_path, 0, "2025-10-27")
        arr = np.asarray(load_omie_orders(cfg))
        assert _prices(arr, ORD_PRICE) == [_pi(73.01), _pi(73.02)]
        trd = np.asarray(load_omie_trades(cfg))
        assert _prices(trd, TRD_PRICE) == [_pi(73.01), _pi(73.02)]
        assert len(load_omie_agent_metadata(cfg)) > 0


# ===========================================================================
# Real-data TCs (TC-1, TC-1b, TC-2, TC-8, TC-9 real)
# ===========================================================================


def _session_rows(path):
    """Independent minimal reader: [(Contrato, Momento)] of every data row
    (last non-empty field is the send / match time, as in the OMIE layout)."""
    out = []
    with open(path, encoding="latin-1") as fh:
        lines = fh.readlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("Fecha;Contrato;"))
    for ln in lines[hdr + 1:]:
        ln = ln.strip()
        if not ln:
            continue
        fs = ln.split(";")
        while fs and fs[-1].strip() == "":
            fs.pop()
        if len(fs) < 9:
            continue
        out.append((fs[1].strip(), fs[-1].strip()))
    return out


def _prev_iso(iso):
    return datetime.fromordinal(datetime.fromisoformat(iso).toordinal() - 1).date().isoformat()


def _product_sends(date_iso, k, session_iso):
    """Signed seconds from D 00:00 of every product-(D, k) order row in one session file."""
    h, m = divmod(15 * k, 60)
    prefix = f"{_nodash(date_iso)} {h:02d}:{m:02d}-"
    d0 = datetime.fromisoformat(date_iso)
    path = REAL_DATA / f"orders_{_nodash(session_iso)}.1"
    if not path.is_file():
        pytest.skip(_NO_DATA)
    return [
        int((datetime.strptime(mom, "%d/%m/%Y %H:%M:%S") - d0).total_seconds())
        for c, mom in _session_rows(path)
        if c.startswith(prefix)
    ]


def _files_for(date_iso):
    p = _nodash(_prev_iso(date_iso))
    d = _nodash(date_iso)
    return (f"orders_{p}.1", f"orders_{d}.1", f"trades_{p}.1", f"trades_{d}.1")


_REAL_CACHE: dict = {}


def _real_orders(date_iso, k):
    key = (date_iso, k)
    if key not in _REAL_CACHE:
        if not _have(*_files_for(date_iso)):
            pytest.skip(_NO_DATA)
        _REAL_CACHE[key] = np.asarray(load_omie_orders(_cfg(REAL_DATA, k, date_iso)))
    return _REAL_CACHE[key]


@pytest.mark.slow
class TestRealProduct20250625:
    """TC-1. slow: parses orders_20250624.1 + orders_20250625.1 (~40 MB each).
    RED: today 895 rows across products 06-25 and 06-26."""

    def test_row_count_is_one_product(self):
        assert _real_orders("2025-06-25", 0).shape[0] == 863

    def test_rows_are_exactly_product_0625_from_both_files(self):
        prev = _product_sends("2025-06-25", 0, "2025-06-24")
        cur = _product_sends("2025-06-25", 0, "2025-06-25")
        assert (len(prev), len(cur)) == (387, 476), "raw-data precondition (card TC-1)"
        expected = sorted(s + 86_400 for s in prev + cur)
        got = sorted(int(v) for v in _real_orders("2025-06-25", 0)[:, ORD_TS])
        assert got == expected

    def test_contract_column_is_zero(self):
        assert np.all(_real_orders("2025-06-25", 0)[:, ORD_CID] == 0)


@pytest.mark.slow
class TestRealProduct20250601:
    """TC-1b. slow: parses orders_20250531.1 + orders_20250601.1.
    RED: today 2,201 rows with one timestamp drop."""

    def test_row_count_is_one_product(self):
        assert _real_orders("2025-06-01", 0).shape[0] == 2_546

    def test_rows_are_exactly_product_0601_from_files_0531_and_0601(self):
        prev = _product_sends("2025-06-01", 0, "2025-05-31")
        cur = _product_sends("2025-06-01", 0, "2025-06-01")
        expected = sorted(s + 86_400 for s in prev + cur)
        assert len(expected) == 2_546, "raw-data precondition (card TC-1b)"
        got = sorted(int(v) for v in _real_orders("2025-06-01", 0)[:, ORD_TS])
        assert got == expected

    def test_timestamp_range_and_no_drop(self):
        ts = _real_orders("2025-06-01", 0)[:, ORD_TS].astype(np.int64)
        assert int(ts.min()) == 55_209
        assert int(ts.max()) == 82_769
        assert int(np.sum(np.diff(ts) < 0)) == 0


@pytest.mark.slow
class TestRealTimestampAxis:
    """TC-2. slow: real session files. RED: today one 27,585 s drop on 06-25."""

    @pytest.mark.parametrize("date_iso,k", [("2025-06-25", 0), ("2026-01-12", 81)])
    def test_timestamps_non_decreasing(self, date_iso, k):
        ts = _real_orders(date_iso, k)[:, ORD_TS].astype(np.int64)
        drops = np.diff(ts)[np.diff(ts) < 0]
        assert drops.size == 0, f"{drops.size} drop(s), e.g. {drops[:3].tolist()}"

    @pytest.mark.parametrize("date_iso,k", [("2025-06-25", 0), ("2026-01-12", 81)])
    def test_timestamps_within_send_window(self, date_iso, k):
        ts = _real_orders(date_iso, k)[:, ORD_TS].astype(np.int64)
        assert int(ts.min()) >= WINDOW_LO
        assert int(ts.max()) < WINDOW_HI


@pytest.mark.slow
class TestRealByteIdentityOfPreviousDaySends:
    """TC-8 (GUARD, GREEN on both loaders). slow: real 06-25 files.

    Rows sent on D-1 keep the value the pre-fix loader stored
    (``raw % 86400 == raw + 86400`` for raw in [-86,400, 0)). Pinned as a
    multiset on the 476 product-06-25 rows of file 0625 (all sent on 06-24).
    """

    def test_file_d_rows_of_product_keep_pre_fix_values(self):
        cur = _product_sends("2025-06-25", 0, "2025-06-25")
        assert len(cur) == 476 and all(-86_400 <= s < 0 for s in cur)
        pre_fix = collections.Counter(s % 86_400 for s in cur)
        got = collections.Counter(int(v) for v in _real_orders("2025-06-25", 0)[:, ORD_TS])
        missing = pre_fix - got
        assert not missing, f"{sum(missing.values())} file-D values moved, e.g. {list(missing)[:3]}"


@pytest.mark.slow
class TestPrefixFirstFilterRealData:
    """TC-9 real. slow: real session files around the 2025-10-26 fall-back day.
    RED: 2025-10-25 raises in the slot parser today; 2025-10-27 reads one file."""

    @pytest.mark.parametrize("date_iso", ["2025-10-25", "2025-10-27"])
    def test_dst_neighbour_product_loads_whole(self, date_iso):
        arr = _real_orders(date_iso, 0)
        prev = _product_sends(date_iso, 0, _prev_iso(date_iso))
        cur = _product_sends(date_iso, 0, date_iso)
        assert arr.shape[0] == len(prev) + len(cur) > 0
        assert np.all(np.diff(arr[:, ORD_TS].astype(np.int64)) >= 0)
