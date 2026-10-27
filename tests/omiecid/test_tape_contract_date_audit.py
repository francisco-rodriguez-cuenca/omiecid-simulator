"""Step-5 AUDIT — OMIE-TAPE-CONTRACT-DATE-1 (one delivery product per tape).

Independent, adversarial verification of the finished L1 loader, derived from
the story card (``docs/omiecid/agile-plan.md`` "#### Story
OMIE-TAPE-CONTRACT-DATE-1"), market-spec v2.10 §3.1 / §13 / §13.4 and
open-questions OQ-27..30. Written WITHOUT reading the locked
``test_tape_contract_date.py``.

Contract under audit (black box, public interface only):

  * ``tape(D, k)`` = the rows whose ``Contrato`` START is (D, slot k), read
    from session files D-1 and D ONLY (file date = session day; a session
    file holds sends / matches in [22:20 S-1, 22:00 S)).
  * Orders: file D-1 then file D, product filter, STABLE sort by timestamp
    (same-second ties keep file order, file D-1 first); ``order_id`` 1..N.
  * Trades: file D-1 then file D, file order, NO sort.
  * Timestamp = seconds since D-1 00:00 wall clock, NO modulo. Real data in
    [55,200, 165,600); per product < GCT(k) = 86,400 + 3,600*(k//4 - 1).
    The loader does NOT validate / clip the window (F-4).
  * (Agente, Unidad) pair universe = union over the product's order + trade
    rows from both files; ids 1..N in canonical tuple order.
  * All four files ``{orders,trades}_{D-1,D}.1`` required, checked before any
    parse -> ``IncompleteProductError(ValueError)``; tape-set / anchor
    builders propagate it.
  * DST transition day (Europe/Madrid) -> ``ValueError`` unless
    ``allow_dst_transition=True``; under the opt-in a date-D ``Contrato``
    whose START carries an A/B suffix raises ``ValueError``. Prefix-first
    filter: neighbouring-day A/B strings never break product D.

Incidental constants are deliberately varied (dates across month / year /
leap-day boundaries, slots 0, 3, 4, 47, 48, 92-95, tags, agent codes).

Real-data tests (``data/rawOMIE``) are ``@pytest.mark.slow``: each real load
parses four ~250k-line CSVs (~3.5 s per loader call). They compare the loader
against an independent line-level CSV reference (count, order, timestamps,
side / price / qty, agent pairs).
"""

from __future__ import annotations

import functools
import itertools
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    IncompleteProductError,
    build_day_ahead_anchors,
    build_fixed_tape_set,
    load_omie_agent_metadata,
    load_omie_orders,
    load_omie_trades,
)
from omiecid_simulator.omiecid.encoding import encode_price, encode_qty
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import Side

PC = OMIEProductConfig()

LOW, HIGH = 55_200, 165_600


def gct_l1(k: int) -> int:
    """GCT(k) in seconds since D-1 00:00 (hour-aligned, market-spec §3.1)."""
    return 86_400 + 3_600 * (k // 4 - 1)


# ---------------------------------------------------------------------------
# Synthetic CSV fixture helpers
# ---------------------------------------------------------------------------

_O_META = "OMIE - Mercado de electricidad;Fecha Emision :x;;x;Ordenes enviadas a XBID;;;;"
_O_COLS = (
    "Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;Tipo oferta;"
    "Condiciones a la ejecucion;Condiciones a la validez;Cantidad reducida;PPD;"
    "Momento de envio;"
)
_T_META = "OMIE - Mercado de electricidad;Fecha Emision :x;;x;Transacciones realizadas en XBID;;;;"
_T_COLS = (
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;Agente venta;"
    "Unidad venta;Zona venta;Precio;Cantidad;Momento casacion;"
)
_ZONE = "10YES-REE------0"


def contrato(d: date, k: int) -> str:
    start = datetime.combine(d, time()) + timedelta(minutes=15 * k)
    end = start + timedelta(minutes=15)
    return f"{start:%Y%m%d %H:%M}-{end:%Y%m%d %H:%M}"


def session_of(send: datetime) -> date:
    """Session file date holding a send at ``send`` ([22:20 S-1, 22:00 S))."""
    return send.date() + timedelta(days=1) if send.time() >= time(22, 0) else send.date()


def l1_ts(d: date, send: datetime) -> int:
    """Seconds since (d - 1) 00:00 wall clock (independent reference)."""
    return int((send - datetime.combine(d - timedelta(days=1), time())).total_seconds())


def _num(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


@dataclass
class ORow:
    d: date
    k: int
    send: datetime
    tag: int  # quantity = tag * 0.1 MW -> qty column == tag
    ag: str = "AGA"
    un: str = "UNA"
    side: str = "V"
    price: float = 42.17
    file: Optional[date] = None
    contrato: Optional[str] = None

    def line(self, fdate: date) -> str:
        c = self.contrato or contrato(self.d, self.k)
        return (
            f"{fdate:%d/%m/%Y};{c};{_ZONE};{self.ag};{self.un};{_num(self.price)};"
            f"{_num(self.tag * 0.1)};{self.side};;;;;{self.send:%d/%m/%Y %H:%M:%S};"
        )


@dataclass
class TRow:
    d: date
    k: int
    match: datetime
    tag: int
    buy: tuple = ("BUY", "BU1")
    sell: tuple = ("SEL", "SE1")
    price: float = 37.5
    file: Optional[date] = None
    contrato: Optional[str] = None

    def line(self, fdate: date) -> str:
        c = self.contrato or contrato(self.d, self.k)
        return (
            f"{fdate:%d/%m/%Y};{c};{self.buy[0]};{self.buy[1]};{_ZONE};"
            f"{self.sell[0]};{self.sell[1]};{_ZONE};{_num(self.price)};"
            f"{_num(self.tag * 0.1)};{self.match:%d/%m/%Y %H:%M:%S};"
        )


def write_session_files(root: Path, orders=(), trades=(), dates=()) -> None:
    """Write orders/trades session files; ``dates`` get header-only files too."""
    o_by: dict = {d: [] for d in dates}
    t_by: dict = {d: [] for d in dates}
    for r in orders:
        o_by.setdefault(r.file or session_of(r.send), []).append(r)
    for r in trades:
        t_by.setdefault(r.file or session_of(r.match), []).append(r)
    for fdate, rows in o_by.items():
        p = root / f"orders_{fdate:%Y%m%d}.1"
        p.write_text(
            "\n".join([_O_META, _O_COLS] + [r.line(fdate) for r in rows]) + "\n",
            encoding="latin-1",
        )
    for fdate, rows in t_by.items():
        p = root / f"trades_{fdate:%Y%m%d}.1"
        p.write_text(
            "\n".join([_T_META, _T_COLS] + [r.line(fdate) for r in rows]) + "\n",
            encoding="latin-1",
        )


def cfg(root, d, k, **kw) -> OMIEDataConfig:
    return OMIEDataConfig(
        data_path=str(root),
        contract_id=int(k),
        delivery_date=d.isoformat() if isinstance(d, date) else d,
        **kw,
    )


def np_(x) -> np.ndarray:
    return np.asarray(x)


def D(s: str) -> date:
    return date.fromisoformat(s)


def at(d: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm, ss))


# ---------------------------------------------------------------------------
# 1. Timestamp frame, product gathering, no wrap — varied dates and slots
# ---------------------------------------------------------------------------

_DATES = ["2025-07-01", "2026-01-01", "2028-03-01", "2027-03-01", "2025-11-30"]
_SLOTS = [0, 3, 4, 47, 48, 92, 95]


def _product_timeline(d: date, k: int) -> list:
    """Physically plausible sends of product (d, k), in [15:20 D-1, GCT(k))."""
    d1 = d - timedelta(days=1)
    g = gct_l1(k)
    sends = [at(d1, 15, 20, 0), at(d1, 21, 59, 59), at(d1, 22, 20, 0), at(d1, 22, 59, 59)]
    if g > 86_400 - 1:
        sends.append(at(d1, 23, 59, 59))
    if g > 86_400:
        sends += [at(d, 0, 0, 0), at(d, 0, 0, 1)]
        sends.append(datetime.combine(d1, time()) + timedelta(seconds=g - 1))
    return sends


def _noise_orders(d: date, k: int, tag0: int) -> list:
    """Rows of OTHER products that live in files D-1 / D."""
    d1, d2, dn = d - timedelta(days=1), d - timedelta(days=2), d + timedelta(days=1)
    nb = k + 1 if k < 95 else k - 1
    return [
        ORow(dn, k, at(d, 15, 20, 0), tag0 + 0),  # product (D+1, k) in file D
        ORow(dn, k, at(d, 21, 30, 5), tag0 + 1),  # product (D+1, k) in file D
        ORow(d1, k, at(d1, 8, 0, 0), tag0 + 2),  # product (D-1, k) in file D-1
        ORow(d1, k, at(d2, 22, 30, 0), tag0 + 3),  # product (D-1, k) in file D-1
        ORow(d, nb, at(d1, 16, 0, 0), tag0 + 4),  # neighbour slot, date D
        ORow(d, nb, at(d1, 22, 40, 0), tag0 + 5),  # neighbour slot, date D
    ]


@pytest.mark.parametrize("k", _SLOTS)
@pytest.mark.parametrize("ds", _DATES)
def test_orders_one_product_exact_timestamps_no_wrap(tmp_path, ds, k):
    d = D(ds)
    sends = _product_timeline(d, k)
    own = [ORow(d, k, s, tag=10 + i) for i, s in enumerate(sends)]
    # write each file's own rows in REVERSE chronological order
    rows = list(reversed(own)) + _noise_orders(d, k, 500)
    write_session_files(tmp_path, rows, dates=[d - timedelta(days=1), d])
    out = np_(load_omie_orders(cfg(tmp_path, d, k)))

    exp = sorted(own, key=lambda r: l1_ts(d, r.send))
    assert out.shape == (len(own), 13)
    assert out.dtype == np.int32
    assert out[:, 0].tolist() == [k] * len(own)
    assert out[:, 6].tolist() == [l1_ts(d, r.send) for r in exp]
    assert out[:, 3].tolist() == [r.tag for r in exp]
    assert out[:, 4].tolist() == list(range(1, len(own) + 1))
    ts = out[:, 6]
    assert np.all(np.diff(ts) >= 0)
    assert ts.min() >= LOW and ts.max() < gct_l1(k) <= HIGH


@pytest.mark.parametrize("k", [0, 4, 47, 95])
@pytest.mark.parametrize("ds", ["2026-01-01", "2028-03-01", "2025-10-01"])
def test_trades_one_product_file_order_no_sort(tmp_path, ds, k):
    d = D(ds)
    d1 = d - timedelta(days=1)
    sends = _product_timeline(d, k)
    in_d1 = [s for s in sends if session_of(s) == d1]
    in_d = [s for s in sends if session_of(s) == d]
    # each file lists its trades in REVERSE time order: no sort => kept as is
    w_d1 = list(reversed(in_d1))
    w_d = list(reversed(in_d))
    own = [TRow(d, k, s, tag=20 + i) for i, s in enumerate(w_d1 + w_d)]
    noise = [
        TRow(d + timedelta(days=1), k, at(d, 16, 0, 0), 900),
        TRow(d1, k, at(d1, 9, 0, 0), 901),
        TRow(d, (k + 2) % 96, at(d1, 17, 0, 0), 902),
    ]
    write_session_files(tmp_path, trades=own + noise, dates=[d1, d])
    out = np_(load_omie_trades(cfg(tmp_path, d, k)))
    assert out.shape == (len(own), 6)
    assert out[:, 0].tolist() == [k] * len(own)
    assert out[:, 2].tolist() == [r.tag for r in own]
    assert out[:, 5].tolist() == [l1_ts(d, r.match) for r in own]


def test_midnight_crossing_is_monotone_without_wrap(tmp_path):
    d = D("2026-01-01")  # D-1 = 2025-12-31 (year boundary)
    d1 = d - timedelta(days=1)
    k = 12  # GCT 02:00 on D
    sends = [at(d1, 23, 59, 58), at(d1, 23, 59, 59), at(d, 0, 0, 0), at(d, 0, 0, 1), at(d, 1, 59, 59)]
    own = [ORow(d, k, s, tag=i + 1) for i, s in enumerate(sends)]
    write_session_files(tmp_path, own, dates=[d1, d])
    out = np_(load_omie_orders(cfg(tmp_path, d, k)))
    assert out[:, 6].tolist() == [86_398, 86_399, 86_400, 86_401, 93_599]


def test_ida2_boundary_rows_22_00_and_22_20(tmp_path):
    """21:59:59 D-1 (file D-1) and 22:20:00 D-1 (file D, IDA reopen)."""
    d = D("2025-09-01")  # D-1 = 2025-08-31 (month boundary)
    d1 = d - timedelta(days=1)
    k = 3
    rows = [
        ORow(d, k, at(d1, 22, 20, 0), 1),
        ORow(d, k, at(d1, 22, 20, 0), 2),
        ORow(d, k, at(d1, 21, 59, 59), 3),
    ]
    write_session_files(tmp_path, rows, dates=[d1, d])
    assert (tmp_path / "orders_20250831.1").read_text(encoding="latin-1").count("21:59:59") == 1
    out = np_(load_omie_orders(cfg(tmp_path, d, k)))
    assert out[:, 6].tolist() == [79_199, 80_400, 80_400]
    assert out[:, 3].tolist() == [3, 1, 2]


def test_rows_only_in_file_d_minus_1(tmp_path):
    d = D("2025-08-01")
    d1 = d - timedelta(days=1)
    own = [ORow(d, 1, at(d1, 15, 20 + i, 0), tag=i + 1) for i in range(3)]
    write_session_files(tmp_path, own, dates=[d1, d])
    out = np_(load_omie_orders(cfg(tmp_path, d, 1)))
    assert out[:, 6].tolist() == [55_200, 55_260, 55_320]


def test_rows_only_in_file_d(tmp_path):
    d = D("2025-08-01")
    d1 = d - timedelta(days=1)
    own = [ORow(d, 60, at(d, 9, 0, i), tag=i + 1) for i in range(3)]
    write_session_files(tmp_path, own, dates=[d1, d])
    out = np_(load_omie_orders(cfg(tmp_path, d, 60)))
    assert out[:, 6].tolist() == [86_400 + 32_400 + i for i in range(3)]


def test_loader_does_not_clip_out_of_window_sends(tmp_path):
    """F-4: no send-window validation. A pre-GOT send is kept verbatim."""
    d = D("2025-12-01")
    d1 = d - timedelta(days=1)
    rows = [ORow(d, 0, at(d1, 4, 0, 0), 1), ORow(d, 0, at(d1, 16, 0, 0), 2)]
    write_session_files(tmp_path, rows, dates=[d1, d])
    out = np_(load_omie_orders(cfg(tmp_path, d, 0)))
    assert out[:, 6].tolist() == [14_400, 57_600]


def test_complete_but_empty_product_returns_empty_shapes(tmp_path):
    d = D("2025-12-31")
    d1 = d - timedelta(days=1)
    write_session_files(tmp_path, [ORow(d, 5, at(d1, 16, 0), 1)], dates=[d1, d])
    assert np_(load_omie_orders(cfg(tmp_path, d, 6))).shape == (0, 13)
    assert np_(load_omie_trades(cfg(tmp_path, d, 6))).shape == (0, 6)
    assert load_omie_agent_metadata(cfg(tmp_path, d, 6)) == {}


# ---------------------------------------------------------------------------
# 2. Files D+1 and D-2 are never read
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [92, 93, 94, 95])
def test_file_d_plus_1_never_contributes(tmp_path, k):
    d = D("2025-08-31")  # D+1 = 2025-09-01
    d1, dn, d2 = d - timedelta(days=1), d + timedelta(days=1), d - timedelta(days=2)
    own = [ORow(d, k, at(d, 21, 59, 0), 1), ORow(d, k, at(d1, 15, 21, 0), 2)]
    planted_next = ORow(d, k, at(d, 22, 30, 0), 3, file=dn)  # impossible row in D+1
    planted_prev = ORow(d, k, at(d2, 23, 0, 0), 4, file=d2)  # row in file D-2
    tr_next = TRow(d, k, at(d, 22, 25, 0), 5, file=dn)
    tr_own = TRow(d, k, at(d, 21, 0, 0), 6)
    write_session_files(tmp_path, own + [planted_next, planted_prev], [tr_next, tr_own], dates=[d1, d, dn, d2])
    o = np_(load_omie_orders(cfg(tmp_path, d, k)))
    t = np_(load_omie_trades(cfg(tmp_path, d, k)))
    assert sorted(o[:, 3].tolist()) == [1, 2]
    assert o[:, 6].tolist() == [55_260, 165_540]
    assert t[:, 2].tolist() == [6]


@pytest.mark.parametrize("k", [0, 95])
def test_loads_with_neighbour_files_absent(tmp_path, k):
    d = D("2027-01-01")
    d1 = d - timedelta(days=1)
    write_session_files(tmp_path, [ORow(d, k, at(d1, 18, 0, 0), 1)], dates=[d1, d])
    assert sorted(os.listdir(tmp_path)) == sorted(
        ["orders_20261231.1", "orders_20270101.1", "trades_20261231.1", "trades_20270101.1"]
    )
    assert np_(load_omie_orders(cfg(tmp_path, d, k)))[:, 3].tolist() == [1]


# ---------------------------------------------------------------------------
# 3. Stable order on same-second ties
# ---------------------------------------------------------------------------

def test_same_second_ties_keep_file_order_and_d_minus_1_first(tmp_path):
    d = D("2026-02-01")
    d1 = d - timedelta(days=1)
    k = 40
    rows = [
        ORow(d, k, at(d1, 18, 0, 0), 1, file=d1),
        ORow(d, k, at(d1, 18, 0, 0), 2, file=d1),
        ORow(d, k, at(d1, 18, 0, 0), 3, file=d1),
        ORow(d, k, at(d1, 17, 0, 0), 4, file=d1),
        ORow(d, k, at(d1, 18, 0, 0), 5, file=d1),
        ORow(d, k, at(d1, 18, 0, 0), 6, file=d1),
        ORow(d, k, at(d1, 22, 30, 0), 7, file=d),
        ORow(d, k, at(d1, 22, 25, 0), 8, file=d),
        ORow(d, k, at(d1, 22, 25, 0), 9, file=d),
        # cross-file tie (not physical; loader does not validate) -> D-1 rows first
        ORow(d, k, at(d1, 18, 0, 0), 10, file=d),
    ]
    write_session_files(tmp_path, rows, dates=[d1, d])
    out = np_(load_omie_orders(cfg(tmp_path, d, k)))
    assert out[:, 3].tolist() == [4, 1, 2, 3, 5, 6, 10, 8, 9, 7]
    assert out[:, 4].tolist() == list(range(1, 11))


def test_large_tie_block_is_stable(tmp_path):
    """300 interleaved rows on two timestamps: a non-stable sort would permute."""
    d = D("2025-06-20")
    d1 = d - timedelta(days=1)
    k = 16
    perm = [(i * 37) % 300 + 1 for i in range(300)]  # distinct tags, scrambled
    rows = []
    for i, tag in enumerate(perm):
        send = at(d1, 19, 0, 1) if i % 2 else at(d1, 19, 0, 0)
        rows.append(ORow(d, k, send, tag))
    write_session_files(tmp_path, rows, dates=[d1, d])
    out = np_(load_omie_orders(cfg(tmp_path, d, k)))
    exp = [perm[i] for i in range(0, 300, 2)] + [perm[i] for i in range(1, 300, 2)]
    assert out[:, 3].tolist() == exp


# ---------------------------------------------------------------------------
# 4. (Agente, Unidad) universe across files; orders/trades consistency
# ---------------------------------------------------------------------------

def test_pair_universe_union_over_both_files_orders_and_trades(tmp_path):
    d = D("2026-01-01")
    d1, dn = d - timedelta(days=1), d + timedelta(days=1)
    k = 20
    orders = [
        ORow(d, k, at(d1, 16, 0), 1, ag="MAA", un="U1"),  # D-1 only pair? no, also in D
        ORow(d, k, at(d1, 16, 1), 2, ag="MAA", un="U2"),  # only in file D-1
        ORow(d, k, at(d1, 22, 30), 3, ag="MAA", un="U1"),  # file D
        ORow(d, k, at(d, 1, 0), 4, ag="NBB", un="V1"),  # only in file D
        # other products' pairs (must NOT enter the universe)
        ORow(dn, k, at(d, 16, 0), 5, ag="ZZZ", un="Z9"),
        ORow(d, k + 1, at(d1, 16, 0), 6, ag="YYY", un="Y9"),
    ]
    trades = [
        TRow(d, k, at(d1, 17, 0), 7, buy=("CCC", "W1"), sell=("MAA", "U2")),  # file D-1
        TRow(d, k, at(d, 2, 0), 8, buy=("", ""), sell=("NBB", "V1")),  # file D, blank leg
        TRow(dn, k, at(d, 17, 0), 9, buy=("XXX", "X9"), sell=("MAA", "U1")),  # other product
    ]
    write_session_files(tmp_path, orders, trades, dates=[d1, d])
    c = cfg(tmp_path, d, k)
    meta = load_omie_agent_metadata(c)
    expected_pairs = sorted({("MAA", "U1"), ("MAA", "U2"), ("NBB", "V1"), ("CCC", "W1"), ("", "")})
    assert meta == {i + 1: p for i, p in enumerate(expected_pairs)}

    o = np_(load_omie_orders(c))
    by_tag = {1: ("MAA", "U1"), 2: ("MAA", "U2"), 3: ("MAA", "U1"), 4: ("NBB", "V1")}
    assert sorted(o[:, 3].tolist()) == [1, 2, 3, 4]
    for row in o:
        assert meta[int(row[5])] == by_tag[int(row[3])]
    # one id for the pair that appears in both files
    ids_u1 = {int(r[5]) for r in o if int(r[3]) in (1, 3)}
    assert len(ids_u1) == 1

    t = np_(load_omie_trades(c))
    assert t[:, 2].tolist() == [7, 8]
    assert meta[int(t[0, 3])] == ("CCC", "W1") and meta[int(t[0, 4])] == ("MAA", "U2")
    assert meta[int(t[1, 3])] == ("", "") and meta[int(t[1, 4])] == ("NBB", "V1")
    # orders and trades agree on the id of a shared pair
    oid = {meta[int(r[5])]: int(r[5]) for r in o}
    assert int(t[0, 4]) == oid[("MAA", "U2")]
    assert int(t[1, 4]) == oid[("NBB", "V1")]


def test_trade_only_pair_in_file_d_minus_1_gets_positive_id(tmp_path):
    d = D("2025-07-01")
    d1 = d - timedelta(days=1)
    k = 2
    write_session_files(
        tmp_path,
        [ORow(d, k, at(d, 0, 0) - timedelta(hours=1, minutes=30), 1, ag="OOO", un="O1")],
        [TRow(d, k, at(d1, 18, 0), 2, buy=("TTT", "T1"), sell=("OOO", "O1"))],
        dates=[d1, d],
    )
    c = cfg(tmp_path, d, k)
    meta = load_omie_agent_metadata(c)
    t = np_(load_omie_trades(c))
    assert set(meta.values()) == {("OOO", "O1"), ("TTT", "T1")}
    assert int(t[0, 3]) > 0 and meta[int(t[0, 3])] == ("TTT", "T1")


# ---------------------------------------------------------------------------
# 5. Completeness: every missing-file permutation, all three loaders
# ---------------------------------------------------------------------------

_LOADERS = {
    "orders": load_omie_orders,
    "trades": load_omie_trades,
    "metadata": load_omie_agent_metadata,
}


def _four_files(d: date) -> list:
    d1 = d - timedelta(days=1)
    return [
        f"orders_{d1:%Y%m%d}.1",
        f"orders_{d:%Y%m%d}.1",
        f"trades_{d1:%Y%m%d}.1",
        f"trades_{d:%Y%m%d}.1",
    ]


_SUBSETS = [s for n in range(1, 5) for s in itertools.combinations(range(4), n)]


@pytest.mark.parametrize("loader", list(_LOADERS))
@pytest.mark.parametrize("missing", _SUBSETS, ids=lambda s: "miss" + "".join(map(str, s)))
def test_every_missing_file_subset_raises_incomplete(tmp_path, loader, missing):
    d = D("2026-03-01")  # D-1 = 2026-02-28
    d1 = d - timedelta(days=1)
    write_session_files(
        tmp_path,
        [ORow(d, 8, at(d1, 16, 0), 1)],
        [TRow(d, 8, at(d1, 16, 5), 2)],
        dates=[d1, d, d + timedelta(days=1), d - timedelta(days=2)],
    )
    names = _four_files(d)
    for i in missing:
        (tmp_path / names[i]).unlink()
    with pytest.raises(IncompleteProductError) as ei:
        _LOADERS[loader](cfg(tmp_path, d, 8))
    assert isinstance(ei.value, ValueError)
    msg = str(ei.value)
    assert any(names[i] in msg for i in missing)
    assert not any(names[i] in msg for i in set(range(4)) - set(missing))


@pytest.mark.parametrize("loader", list(_LOADERS))
def test_completeness_checked_before_parsing_garbage(tmp_path, loader):
    d = D("2025-06-10")
    names = _four_files(d)
    for n in names[:3]:
        (tmp_path / n).write_bytes(b"\x00\xff garbage ;;;\n\x01\x02")
    with pytest.raises(IncompleteProductError) as ei:
        _LOADERS[loader](cfg(tmp_path, d, 0))
    assert names[3] in str(ei.value)


def test_incomplete_error_is_module_level_valueerror_subclass():
    assert issubclass(IncompleteProductError, ValueError)
    assert IncompleteProductError.__module__.endswith("omie_data_loader")


def _complete_product(root: Path, d: date, k: int, n: int = 2) -> None:
    d1 = d - timedelta(days=1)
    write_session_files(
        root, [ORow(d, k, at(d1, 16, i), i + 1) for i in range(n)], dates=[d1, d]
    )


def test_build_fixed_tape_set_propagates_incomplete_in_any_position(tmp_path):
    good = D("2025-11-15")
    bad = D("2025-11-20")
    _complete_product(tmp_path, good, 4)
    (tmp_path / f"orders_{bad:%Y%m%d}.1").write_text(_O_META + "\n" + _O_COLS + "\n", encoding="latin-1")
    for order in ([good, bad], [bad, good]):
        with pytest.raises(IncompleteProductError):
            build_fixed_tape_set([cfg(tmp_path, x, 4) for x in order])
    with pytest.raises(IncompleteProductError):
        build_fixed_tape_set(
            data_path=str(tmp_path), contract_ids=[4], delivery_dates=[good.isoformat(), bad.isoformat()]
        )


def test_build_fixed_tape_set_still_drops_complete_empty_tape(tmp_path):
    a, b = D("2025-11-15"), D("2025-11-16")
    _complete_product(tmp_path, a, 4, n=3)
    write_session_files(tmp_path, dates=[b])  # b complete (files 11-15 and 11-16), empty
    tapes = np_(build_fixed_tape_set([cfg(tmp_path, b, 4), cfg(tmp_path, a, 4)]))
    assert tapes.shape[0] == 1
    assert tapes[0, :3, 3].tolist() == [1, 2, 3]


def test_build_day_ahead_anchors_propagates_incomplete(tmp_path):
    good, bad = D("2025-06-05"), D("2025-06-08")
    _complete_product(tmp_path, good, 0)
    with pytest.raises(IncompleteProductError):
        build_day_ahead_anchors([cfg(tmp_path, bad, 0), cfg(tmp_path, good, 0)])
    with pytest.raises(IncompleteProductError):
        build_day_ahead_anchors(data_path=str(tmp_path), contract_ids=[0], delivery_dates=[bad.isoformat()])


# ---------------------------------------------------------------------------
# 6. DST guard, opt-in, A/B suffix, prefix-first filter
# ---------------------------------------------------------------------------

_DST_DAYS = ["2025-10-26", "2026-03-29", "2024-03-31", "2024-10-27", "2027-03-28", "2027-10-31"]
_DST_NEIGHBOURS = ["2025-10-25", "2025-10-27", "2026-03-28", "2026-03-30", "2027-10-30", "2027-11-01"]


@pytest.mark.parametrize("loader", list(_LOADERS))
@pytest.mark.parametrize("ds", _DST_DAYS)
def test_dst_day_raises_without_opt_in(tmp_path, ds, loader):
    d = D(ds)
    _complete_product(tmp_path, d, 0)
    with pytest.raises(ValueError) as ei:
        _LOADERS[loader](cfg(tmp_path, d, 0))
    assert not isinstance(ei.value, IncompleteProductError)


@pytest.mark.parametrize("ds", _DST_NEIGHBOURS)
def test_dst_neighbour_days_are_not_guarded(tmp_path, ds):
    d = D(ds)
    _complete_product(tmp_path, d, 0, n=2)
    assert np_(load_omie_orders(cfg(tmp_path, d, 0))).shape == (2, 13)


@pytest.mark.parametrize("ds", ["2025-10-26", "2026-03-29"])
def test_dst_opt_in_suffix_free_loads(tmp_path, ds):
    d = D(ds)
    d1 = d - timedelta(days=1)
    rows = [ORow(d, 0, at(d1, 16, 0), 1), ORow(d, 0, at(d1, 22, 45), 2)]
    write_session_files(tmp_path, rows, [TRow(d, 0, at(d1, 17, 0), 3)], dates=[d1, d])
    c = cfg(tmp_path, d, 0, allow_dst_transition=True)
    o = np_(load_omie_orders(c))
    assert o[:, 3].tolist() == [1, 2]
    assert o[:, 6].tolist() == [57_600, 81_900]  # both sends on D-1 (no transition yet)
    assert np_(load_omie_trades(c))[:, 2].tolist() == [3]


@pytest.mark.parametrize("suffix_contract", [
    "20251026 02:00A-20251026 02:15A",
    "20251026 02:00B-20251026 02:15B",
    "20251026 02:45A-20251026 02:00B",
])
def test_dst_opt_in_ab_suffixed_start_raises(tmp_path, suffix_contract):
    d = D("2025-10-26")
    d1 = d - timedelta(days=1)
    k = int(suffix_contract[9:11]) * 4 + int(suffix_contract[12:14]) // 15
    write_session_files(
        tmp_path, [ORow(d, k, at(d1, 18, 0), 1, contrato=suffix_contract)], dates=[d1, d]
    )
    with pytest.raises(ValueError):
        load_omie_orders(cfg(tmp_path, d, k, allow_dst_transition=True))


def test_dst_opt_in_suffix_on_end_only_keys_on_start(tmp_path):
    """``01:45-02:00A`` starts at 01:45 (slot 7): no suffix on the start."""
    d = D("2025-10-26")
    d1 = d - timedelta(days=1)
    write_session_files(
        tmp_path,
        [ORow(d, 7, at(d1, 18, 0), 1, contrato="20251026 01:45-20251026 02:00A")],
        dates=[d1, d],
    )
    o = np_(load_omie_orders(cfg(tmp_path, d, 7, allow_dst_transition=True)))
    assert o[:, 3].tolist() == [1]


@pytest.mark.parametrize("ds,noise_file", [("2025-10-27", "2025-10-26"), ("2025-10-25", "2025-10-25")])
def test_prefix_first_neighbour_day_ab_strings_do_not_break_product(tmp_path, ds, noise_file):
    d = D(ds)
    d1 = d - timedelta(days=1)
    nf = D(noise_file)
    ab = [
        "20251026 02:00A-20251026 02:15A",
        "20251026 02:45B-20251026 03:00",
        "20251026 02:15B-20251026 02:30B",
    ]
    noise = [ORow(D("2025-10-26"), 8, at(nf, 12, 0, i), 100 + i, file=nf, contrato=c) for i, c in enumerate(ab)]
    tnoise = [TRow(D("2025-10-26"), 8, at(nf, 12, 1, i), 200 + i, file=nf, contrato=c) for i, c in enumerate(ab)]
    own = [ORow(d, 0, at(d1, 15, 30), 1), ORow(d, 0, at(d1, 22, 21), 2)]
    town = [TRow(d, 0, at(d1, 15, 31), 3)]
    write_session_files(tmp_path, own + noise, town + tnoise, dates=[d1, d])
    c = cfg(tmp_path, d, 0)
    o = np_(load_omie_orders(c))
    assert o[:, 3].tolist() == [1, 2]
    assert o[:, 6].tolist() == [55_800, 80_460]
    assert np_(load_omie_trades(c))[:, 2].tolist() == [3]
    assert len(load_omie_agent_metadata(c)) == 3  # AGA/UNA, BUY/BU1, SEL/SE1


def test_config_allow_dst_transition_field():
    c = OMIEDataConfig(data_path="x", contract_id=0, delivery_date="2025-06-01")
    assert c.allow_dst_transition is False
    with pytest.raises(Exception):
        c.allow_dst_transition = True  # frozen


# ---------------------------------------------------------------------------
# 7. Real data (data/rawOMIE): independent CSV reference
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
_RAW = _REPO / "data" / "rawOMIE"
_needs_raw = pytest.mark.skipif(
    not (_RAW / "orders_20250625.1").exists(), reason="data/rawOMIE absent"
)


@functools.lru_cache(maxsize=None)
def _ref_file(kind: str, nodash: str) -> tuple:
    """(Contrato_start, fields) for every data line of a session file."""
    path = _RAW / f"{kind}_{nodash}.1"
    with open(path, encoding="latin-1") as f:
        lines = f.read().splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("Fecha;Contrato;"))
    out = []
    for ln in lines[hdr + 1:]:
        flds = ln.split(";")
        if len(flds) < 3:
            continue
        start = flds[1].strip().split("-")[0].strip()
        out.append((start, tuple(x.strip() for x in flds)))
    return tuple(out)


def _key(d: date, k: int) -> str:
    return f"{d:%Y%m%d} {k // 4:02d}:{(k % 4) * 15:02d}"


def _ref_rows(kind: str, d: date, k: int) -> list:
    """Product rows from files D-1 then D, file order, with file tag."""
    key = _key(d, k)
    rows = []
    for tag, fd in (("D-1", d - timedelta(days=1)), ("D", d)):
        for start, flds in _ref_file(kind, f"{fd:%Y%m%d}"):
            if start == key:
                assert flds[1].strip()[:8] == f"{d:%Y%m%d}"
                rows.append((tag, flds))
    return rows


def _eur(s: str) -> float:
    """Spanish number format: ``.`` thousands separator, ``,`` decimal."""
    return float(s.replace(".", "").replace(",", "."))


def _ref_orders(d: date, k: int) -> list:
    base = datetime.combine(d - timedelta(days=1), time())
    out = []
    for tag, f in _ref_rows("orders", d, k):
        nonempty = [x for x in f if x != ""]
        ts = int((datetime.strptime(nonempty[-1], "%d/%m/%Y %H:%M:%S") - base).total_seconds())
        out.append(dict(
            file=tag, ts=ts,
            side=int(Side.BUY) if f[7].upper() == "C" else int(Side.SELL),
            raw_price=f[5],
            price=encode_price(_eur(f[5]), PC),
            qty=encode_qty(_eur(f[6]), PC),
            pair=(f[3], f[4]),
        ))
    return sorted(out, key=lambda r: r["ts"])  # Python sort is stable


def _ref_trades(d: date, k: int) -> list:
    out = []
    for tag, f in _ref_rows("trades", d, k):
        out.append(dict(
            file=tag,
            price=encode_price(_eur(f[8]), PC),
            qty=encode_qty(_eur(f[9]), PC),
            buy=(f[2], f[3]), sell=(f[5], f[6]),
        ))
    return out


@functools.lru_cache(maxsize=None)
def _real(kind: str, ds: str, k: int, dst: bool = False):
    c = OMIEDataConfig(data_path=str(_RAW), contract_id=k, delivery_date=ds, allow_dst_transition=dst)
    if kind == "metadata":
        return load_omie_agent_metadata(c)
    return np_((load_omie_orders if kind == "orders" else load_omie_trades)(c))


# June products + later-month / month-start / year-start products
_REAL = [
    ("2025-06-01", 0),
    ("2025-06-25", 0),
    ("2025-06-18", 47),
    ("2025-06-09", 95),
    ("2025-10-15", 81),
    ("2025-12-01", 48),
    ("2026-01-01", 4),
]
_REAL_IDS = [f"{d}-k{k}" for d, k in _REAL]


@pytest.mark.slow
@_needs_raw
class TestRealProducts:
    """slow: each product loads four real session files per loader call (~3.5 s)."""

    @pytest.mark.parametrize("ds,k", _REAL, ids=_REAL_IDS)
    def test_orders_match_independent_reference_row_by_row(self, ds, k):
        d = D(ds)
        o = _real("orders", ds, k)
        ref = _ref_orders(d, k)
        meta = _real("metadata", ds, k)
        assert o.shape == (len(ref), 13) and len(ref) > 0
        assert {r["file"] for r in ref} == {"D-1", "D"}  # product spans both files
        assert o[:, 0].tolist() == [k] * len(ref)
        assert o[:, 6].tolist() == [r["ts"] for r in ref]
        assert o[:, 1].tolist() == [r["side"] for r in ref]
        assert o[:, 2].tolist() == [r["price"] for r in ref]
        assert o[:, 3].tolist() == [r["qty"] for r in ref]
        assert o[:, 4].tolist() == list(range(1, len(ref) + 1))
        assert [meta[int(a)] for a in o[:, 5]] == [r["pair"] for r in ref]

    @pytest.mark.parametrize("ds,k", [("2025-06-25", 0)], ids=["2025-06-25-k0"])
    def test_thousands_separator_prices_are_parsed(self, ds, k):
        o = _real("orders", ds, k)
        ref = _ref_orders(D(ds), k)
        sep = [i for i, r in enumerate(ref) if "." in r["raw_price"]]
        assert sep, "fixture product must contain a thousands-separated price"
        assert o[sep, 2].tolist() == [ref[i]["price"] for i in sep]

    @pytest.mark.parametrize("ds,k", _REAL, ids=_REAL_IDS)
    def test_orders_timestamps_monotone_inside_l1_window(self, ds, k):
        ts = _real("orders", ds, k)[:, 6]
        assert np.all(np.diff(ts) >= 0)
        assert int(ts.min()) >= LOW and int(ts.max()) < HIGH

    @pytest.mark.parametrize("ds,k", _REAL, ids=_REAL_IDS)
    def test_orders_timestamps_before_product_gct(self, ds, k):
        ts = _real("orders", ds, k)[:, 6]
        assert int(ts.max()) < gct_l1(k)

    @pytest.mark.parametrize("ds,k", _REAL, ids=_REAL_IDS)
    def test_trades_match_independent_reference_in_file_order(self, ds, k):
        d = D(ds)
        t = _real("trades", ds, k)
        ref = _ref_trades(d, k)
        meta = _real("metadata", ds, k)
        assert t.shape == (len(ref), 6)
        assert t[:, 0].tolist() == [k] * len(ref)
        assert t[:, 1].tolist() == [r["price"] for r in ref]
        assert t[:, 2].tolist() == [r["qty"] for r in ref]
        assert [meta[int(a)] for a in t[:, 3]] == [r["buy"] for r in ref]
        assert [meta[int(a)] for a in t[:, 4]] == [r["sell"] for r in ref]

    @pytest.mark.parametrize("ds,k", _REAL, ids=_REAL_IDS)
    def test_metadata_is_exact_union_of_product_pairs(self, ds, k):
        d = D(ds)
        meta = _real("metadata", ds, k)
        pairs = {r["pair"] for r in _ref_orders(d, k)}
        for r in _ref_trades(d, k):
            pairs |= {r["buy"], r["sell"]}
        assert meta == {i + 1: p for i, p in enumerate(sorted(pairs))}


@pytest.mark.slow
@_needs_raw
class TestRealDstAndPrefixFirst:
    """slow: real session files around the 2025-10-26 / 2026-03-29 transitions."""

    @pytest.mark.parametrize("ds", ["2025-10-25", "2025-10-27"])
    def test_prefix_first_neighbour_of_fall_back_loads_complete_product(self, ds):
        o = _real("orders", ds, 0)
        ref = _ref_orders(D(ds), 0)
        assert o.shape[0] == len(ref) > 0
        assert o[:, 6].tolist() == [r["ts"] for r in ref]
        assert int(o[:, 6].min()) >= LOW and int(o[:, 6].max()) < gct_l1(0)

    def test_fall_back_day_opt_in_still_raises_on_real_ab_contracts(self):
        with pytest.raises(ValueError) as ei:
            _real("orders", "2025-10-26", 0, True)
        assert not isinstance(ei.value, IncompleteProductError)

    def test_spring_forward_opt_in_loads_slot_7_and_slot_8_is_empty(self):
        o7 = _real("orders", "2026-03-29", 7, True)
        o8 = _real("orders", "2026-03-29", 8, True)
        assert o7.shape[0] > 0 and np.all(np.diff(o7[:, 6]) >= 0)
        assert int(o7[:, 6].min()) >= LOW
        assert o8.shape == (0, 13)  # 02:00-03:00 does not exist that day


@_needs_raw
class TestRealGuards:
    """Cheap real-data guards: raise before any CSV parse."""

    @pytest.mark.parametrize("loader", list(_LOADERS))
    @pytest.mark.parametrize("ds,k,missing", [
        ("2025-05-31", 0, {"orders_20250530.1", "trades_20250530.1"}),
        ("2026-02-28", 3, {"orders_20260228.1"}),
        ("2026-03-01", 47, {"orders_20260228.1"}),
        ("2026-04-06", 48, {"orders_20260406.1", "trades_20260406.1"}),
        ("2026-04-07", 95, {"orders_20260406.1", "trades_20260406.1"}),
        ("2026-04-15", 92, {"orders_20260415.1", "trades_20260415.1"}),
    ])
    def test_real_incomplete_products_raise_naming_missing_file(self, loader, ds, k, missing):
        for name in missing:
            assert not (_RAW / name).exists()
        with pytest.raises(IncompleteProductError) as ei:
            _LOADERS[loader](OMIEDataConfig(data_path=str(_RAW), contract_id=k, delivery_date=ds))
        assert any(name in str(ei.value) for name in missing)

    @pytest.mark.parametrize("ds", ["2025-10-26", "2026-03-29"])
    @pytest.mark.parametrize("loader", list(_LOADERS))
    def test_real_dst_days_raise_without_opt_in(self, ds, loader):
        with pytest.raises(ValueError) as ei:
            _LOADERS[loader](OMIEDataConfig(data_path=str(_RAW), contract_id=0, delivery_date=ds))
        assert not isinstance(ei.value, IncompleteProductError)

    def test_real_2025_06_01_is_complete_and_2025_05_31_files_present(self):
        for n in ("orders_20250531.1", "trades_20250531.1", "orders_20250601.1", "trades_20250601.1"):
            assert (_RAW / n).exists()
