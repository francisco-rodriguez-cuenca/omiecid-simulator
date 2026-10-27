"""Locked behavioural-contract tests for Story OMIE-ICE-INCOMING-HOP-1.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-ICE-INCOMING-HOP-1 — an
incoming iceberg aggresses with its total quantity: sweep it through the shared
loop before registering it, so resting icebergs set the price of every hop"
(certification fix pass, S4 / plan D2; card as amended at Step 2, v5.195).

Rule (OMIE Rules Annex §4.4; market-spec §6.2 item 7; OQ-31 Reading B; domain
Step-2 confirmation recorded in ``sprint-status.md``):

* An incoming iceberg aggresses with its TOTAL quantity, not only its peak.
  Every resting order it meets is passive and sets the trade price; the
  incoming order is the aggressor on EVERY trade row (including the ones that
  would have been its own later slices).
* One trade row per resting counter bid. A resting iceberg's every instance
  (slice) is its own counter bid; the incoming order's peak never splits a row.
* Only the non-executable remainder R rests. It registers with total R and
  shows a first slice of ``min(peak, R)`` at the order's entry stamp
  (``SEC == order.timestamp``, ``NSEC == 0``). An iceberg swept whole takes no
  side-table slot.

Contract (card Test-first contract row, numbering preserved):
  (1)  C15 shrunk seed 244: all 34 qty ticks @ 1005194, incoming oid aggressor.
                                                          RED before the fix.
  (2)  C16 seed 112 contract 10 WITHOUT row 9 (oid 16542010): 5 @ 1003987 ...
       5 @ 1003983, passive 16542007, aggressor 16542014.  RED before the fix.
  (3)  incoming ICE vs resting ICE (incoming peak 2.0 > resting slice 1.0):
       3 x 1.0 @ the resting price; plus the SELL-aggressor mirror and a
       resting-ICEwPI variant (drifted prices 50.00, 50.05, 50.10).
                                                          RED before the fix.
  (3b) the incoming order's later slices keep the aggressor role (4 rows).
                                                          RED before the fix.
  (4)  Guard: a non-crossing ICE entry is byte-identical (book, tape,
       iceberg_meta, n_active, ts_counter; NOT ``LobState.key``, card A5),
       incl. a full-own-side variant whose evicted iceberg slice is healed
       (card A2).                                         GREEN before and after.
  (5)  ``ts_counter`` pinned on (3) (105 -> 107) and (3b) (107 -> 110);
       ``iceberg_meta`` unchanged on both.                RED before the fix.
  (6)  the remainder R registers with total R and shows ``min(peak, R)`` at
       ``SEC == order.timestamp``, ``NSEC == 0``; it does not cross.
                                                          RED before the fix.
       (6c) incoming ICEwPI: the remainder's first slice rests at the BASE
       price with ``slice_count`` 0 -- a MODELLING CHOICE (card A4; domain
       Step-2 ruling nuance (v)), not a sourced rule.     RED before the fix.
       (6d) an iceberg swept whole takes no slot, so a full side-table does not
       set ``truncated`` (card A3).                       RED before the fix.
  (7)  Guard: FILL-REPORT-1 conservation (``SubmitResult.trade_qty`` equals the
       incoming OID's whole-call tape quantity) on (1)-(3b).
                                                          GREEN before and after.
  (8)  Guard: ``_submit_order`` compiles once.            GREEN before and after.

Every expected value was derived by hand from the rules above and cross-checked
by the test-agent against an independent oracle built only from the public
surface of the pre-fix engine (an IOC submission of the total, then a
non-crossing ICE submission of the remainder). Incidental constants vary (both
sides, OIDs of 1-8 digits, off-round prices).

Units: prices in int ticks (``round((eur + 9999) / 0.01)``; 50.00 EUR ->
1004900), quantities in 0.1 MW ticks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_data_loader import _row_to_order_traced
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, _submit_order
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_REMAINING_HIDDEN,
    COL_SLICE_COUNT,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from tests.omiecid.reference.omie_invariants import build_record

# Every test here runs the jitted submit path (T1.1 marker lint).
pytestmark = pytest.mark.jit


REPO_ROOT = Path(__file__).resolve().parents[2]
# The ONE fixture-path constant (same rule as test_ice_price_priority.py: if
# OMIE-CERT-HARNESS-2 moves the fixtures, this line is updated, not a re-baseline).
FIXTURE_DIR = REPO_ROOT / "notebooks" / "certification" / "fixtures"
# OMIE-CERT-HARNESS-2 (J6) moved only the three C15 shrunk counterexamples to
# tests/omiecid/fixtures/cert/; the C14/C16 fixtures stay in FIXTURE_DIR. Path only,
# not a re-baseline (``FIXTURE_DIR / <absolute path>`` is that absolute path).
SHRUNK_FIXTURE_DIR = REPO_ROOT / "tests" / "omiecid" / "fixtures" / "cert"

C15_244 = SHRUNK_FIXTURE_DIR / "C15_shrunk_seed100000244.npz"
C16_C10 = "C16_seed112_contract10.npz"
C16_ROW9_OID = 16542010

BUY, SELL = 0, 1
NON, IOC, FOK, ICE = 0, 1, 2, 3
CONTRACT_ID = 37

# Tape columns (the loader's 13-wide int32 layout).
_T_SIDE, _T_PRICE, _T_QTY, _T_OID, _T_TS, _T_EXEC, _T_PEAK, _T_PPD = 1, 2, 3, 4, 6, 7, 11, 12

# Resting-row layout returned by BookSnapshot.resting(): (P, Q, OID, TID, SEC, NSEC).
_R_P, _R_Q, _R_OID, _R_SEC, _R_NSEC = 0, 1, 2, 4, 5

_DELIVERY_START, _DELIVERY_END, _GCT = 555, 570, 495


# ---------------------------------------------------------------------------
# Tape construction
# ---------------------------------------------------------------------------
def px(eur: float) -> int:
    """EUR/MWh -> jaxob price tick (offset encoding, market-spec §2.3)."""
    return int(round((eur + 9999.0) / 0.01))


def _row(side, price_eur, qty_mw, oid, ts, ex=NON, peak_mw=0.0, ppd_eur=0.0,
         agent=1) -> np.ndarray:
    r = np.zeros(13, dtype=np.int32)
    r[0] = CONTRACT_ID
    r[_T_SIDE] = side
    r[_T_PRICE] = px(price_eur)
    r[_T_QTY] = int(round(qty_mw * 10))
    r[_T_OID] = oid
    r[5] = agent
    r[_T_TS] = ts
    r[_T_EXEC] = ex
    r[_T_PEAK] = int(round(peak_mw * 10))
    r[_T_PPD] = int(round(ppd_eur * 100))
    return r


def _tape(*rows) -> np.ndarray:
    return np.stack(rows).astype(np.int32)


def _load_tape(name: str) -> np.ndarray:
    path = FIXTURE_DIR / name
    assert path.is_file(), f"fixture missing: {path} (update FIXTURE_DIR if moved)"
    tape = np.asarray(np.load(path, allow_pickle=False)["tape"])
    assert tape.dtype == np.int32 and tape.shape[1] == 13
    return tape


def _without_oid(tape: np.ndarray, oid: int) -> np.ndarray:
    keep = tape[:, _T_OID] != oid
    assert int((~keep).sum()) == 1, f"oid {oid} must appear exactly once"
    return tape[keep]


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------
def _fills_of(rec, submission: int) -> List[Tuple[int, int, int, int]]:
    """``(price_tick, qty_tick, passive_oid, aggressor_oid)`` written by one
    submission, in tape order."""
    return [(f.price_int, f.qty_int, f.passive_oid, f.aggressor_oid)
            for f in rec.fills if f.submission == submission]


def _last_submission(rec) -> int:
    return rec.n_submissions - 1


def _rows_of(snap, side: int, oid: int):
    return [r for r in snap.resting(side) if r[_R_OID] == oid]


def _meta_row(snap, oid: int) -> np.ndarray:
    meta = np.asarray(snap.iceberg_meta)
    rows = meta[meta[:, COL_ORDER_ID] == oid] if meta.size else meta
    assert len(rows) == 1, f"oid {oid} must hold exactly one live side-table slot: {rows}"
    return rows[0]


def _new_book(market_config: OMIEMarketConfig = OMIEMarketConfig(),
              cid: int = CONTRACT_ID) -> ContractBook:
    contract = Contract(contract_id=cid, delivery_start_minutes=_DELIVERY_START,
                        delivery_end_minutes=_DELIVERY_END, gct_minutes=_GCT,
                        status=ContractStatus.OPEN)
    return ContractBook(contract, OMIEProductConfig(), market_config)


def _replay_state(tape: np.ndarray, market_config: OMIEMarketConfig = OMIEMarketConfig()):
    """Submit every real row via the public ``ContractBook.submit_order`` on a
    fresh ``book.init()``; return the final state."""
    rows = [r for r in tape if int(r[_T_OID]) >= 1]
    book = _new_book(market_config, int(rows[0][0]))
    pc = OMIEProductConfig()
    state = book.init()
    for r in rows:
        state, res = book.submit_order(state, _row_to_order_traced(jnp.asarray(r), pc))
        assert bool(res.accepted), f"row {r.tolist()} rejected"
    return state


def _state_digest(state) -> Dict[str, str]:
    """sha256 of bids, asks, trades, iceberg_meta (+ n_active) and ts_counter.

    ``LobState.key`` is deliberately excluded (card A5: on a non-crossing
    entry the PRNG key advances by three splits instead of one; only jaxob's
    random cancel modes read it)."""
    def h(*arrs):
        m = hashlib.sha256()
        for a in arrs:
            a = np.ascontiguousarray(np.asarray(a))
            m.update(str((a.dtype.str, a.shape)).encode())
            m.update(a.tobytes())
        return m.hexdigest()[:16]
    return {
        "bids": h(state.lob.bids),
        "asks": h(state.lob.asks),
        "trades": h(state.lob.trades),
        "iceberg_meta": h(state.iceberg_meta, np.asarray(state.n_active)),
        "ts_counter": h(np.asarray(state.ts_counter)),
    }


# ---------------------------------------------------------------------------
# Scenario tapes (synthetic; derivations in the comments)
# ---------------------------------------------------------------------------
# (3) resting SELL ICE 3.0 / 1.0 @ 50.00 (oid 11, ts 100); incoming BUY ICE
# 3.0 / 2.0 @ 50.40 (oid 22, ts 101). The incoming total 3.0 meets oid 11's
# three instances, each its own counter bid: 3 x 1.0 @ 50.00, passive 11,
# aggressor 22. Nothing rests; oid 11 is drained.
T3 = _tape(
    _row(SELL, 50.00, 3.0, 11, 100, ICE, 1.0, agent=11),
    _row(BUY, 50.40, 3.0, 22, 101, ICE, 2.0, agent=22),
)
T3_FILLS = [(px(50.00), 10, 11, 22)] * 3

# (3) mirror: resting BUY ICE 3.0 / 1.0 @ 50.40 (oid 640017); incoming SELL
# ICE 3.0 / 2.0 @ 50.00 (oid 3). 3 x 1.0 @ 50.40, passive 640017.
T3_MIRROR = _tape(
    _row(BUY, 50.40, 3.0, 640017, 40, ICE, 1.0, agent=4),
    _row(SELL, 50.00, 3.0, 3, 47, ICE, 2.0, agent=9),
)
T3_MIRROR_FILLS = [(px(50.40), 10, 640017, 3)] * 3

# (3) resting ICEwPI: SELL ICE 3.0 / 1.0 @ 50.00 PPD +0.05 (oid 7071; slice k
# priced 50.00 + 0.05 k); incoming BUY ICE 3.0 / 2.0 @ 50.40 (oid 29). Every
# drifted slice still crosses 50.40: 1.0 @ 50.00, 1.0 @ 50.05, 1.0 @ 50.10, each
# at the RESTING slice's own (drifted) price.
T3_ICEWPI = _tape(
    _row(SELL, 50.00, 3.0, 7071, 12, ICE, 1.0, 0.05, agent=2),
    _row(BUY, 50.40, 3.0, 29, 18, ICE, 2.0, agent=6),
)
T3_ICEWPI_FILLS = [(px(50.00), 10, 7071, 29), (px(50.05), 10, 7071, 29),
                   (px(50.10), 10, 7071, 29)]

# (3b) resting SELL ICE 5.0 / 1.0 @ 50.00 (oid 11, ts 100); incoming BUY ICE
# 4.0 / 2.0 @ 50.40 (oid 22, ts 101): 4 x 1.0 @ 50.00 (one row per resting
# instance -- not 2 x 2.0 per incoming peak), passive 11, aggressor 22. Oid 11
# keeps its last 1.0 slice (slice index 4, hidden 0); nothing of 22 rests.
T3B = _tape(
    _row(SELL, 50.00, 5.0, 11, 100, ICE, 1.0, agent=11),
    _row(BUY, 50.40, 4.0, 22, 101, ICE, 2.0, agent=22),
)
T3B_FILLS = [(px(50.00), 10, 11, 22)] * 4

# (3b) varied constants, SELL aggressor: resting BUY ICE 5.0 / 1.0 @ 47.83
# (oid 704211); incoming SELL ICE 4.0 / 2.0 @ 47.29 (oid 58).
T3B_SELL = _tape(
    _row(BUY, 47.83, 5.0, 704211, 7, ICE, 1.0, agent=3),
    _row(SELL, 47.29, 4.0, 58, 15, ICE, 2.0, agent=8),
)
T3B_SELL_FILLS = [(px(47.83), 10, 704211, 58)] * 4

# (6a) resting SELL NON 1.5 @ 50.00 (oid 41); incoming BUY ICE 4.0 / 1.0 @
# 50.20 (oid 9312, ts 63). One trade 1.5 @ 50.00; R = 2.5 -> registers total
# 2.5, first slice min(1.0, 2.5) = 1.0 @ 50.20 at (63, 0); hidden 1.5.
# OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B/E-C): now 1.0 + 0.5 rows (instance
# split) and the remainder shows v_1 = 0.5, hidden 2.0.
T6_R_GT_PEAK = _tape(
    _row(SELL, 50.00, 1.5, 41, 5, NON, agent=4),
    _row(BUY, 50.20, 4.0, 9312, 63, ICE, 1.0, agent=7),
)
# (6b) R < peak, SELL side: resting BUY NON 3.2 @ 61.47 (oid 270001); incoming
# SELL ICE 4.0 / 2.0 @ 61.13 (oid 6, ts 18). One trade 3.2 @ 61.47; R = 0.8 ->
# first slice min(2.0, 0.8) = 0.8 @ 61.13 at (18, 0); hidden 0.
T6_R_LT_PEAK = _tape(
    _row(BUY, 61.47, 3.2, 270001, 2, NON, agent=1),
    _row(SELL, 61.13, 4.0, 6, 18, ICE, 2.0, agent=5),
)
# (6c) MODELLING CHOICE (card A4): resting BUY NON 0.5 @ 60.30 (oid 812);
# incoming SELL ICEwPI 3.0 / 1.0 @ 60.00 PPD +0.10 (oid 33417, ts 25). The sweep
# runs at the base limit: 0.5 @ 60.30. R = 2.5 registers with slice_count 0 and
# its first slice rests at the BASE price 60.00: 1.0 visible, 1.5 hidden.
# OMIE-ICE-INCOMING-LADDER-1 (licensed, E-A/E-B): instance 0 is part-used
# (u = 0.5), so the remainder shows v_1 = 0.5 @ 60.00, hidden 2.0.
T6_ICEWPI = _tape(
    _row(BUY, 60.30, 0.5, 812, 9, NON, agent=2),
    _row(SELL, 60.00, 3.0, 33417, 25, ICE, 1.0, 0.10, agent=3),
)

# (4) non-crossing ICE entries (pins measured on the pre-fix engine).
NC_BUY = _tape(
    _row(SELL, 47.31, 2.0, 61, 3, NON, agent=1),
    _row(SELL, 47.12, 1.0, 7003, 4, ICE, 0.3, agent=2),
    _row(BUY, 46.88, 0.7, 512, 5, NON, agent=3),
    _row(BUY, 47.05, 4.2, 880014, 9, ICE, 1.1, agent=4),
)
NC_SELL = _tape(
    _row(BUY, 52.64, 1.5, 4, 11, NON, agent=1),
    _row(BUY, 52.70, 2.4, 33012, 12, ICE, 0.6, -0.05, agent=2),
    _row(SELL, 53.02, 0.9, 777, 13, NON, agent=3),
    _row(SELL, 52.81, 3.3, 9, 20, ICE, 0.8, 0.10, agent=4),
)

# (4) full own side, book_depth = 6: an iceberg slice (oid 9001) is the worst
# own-side row; five NONs fill the side; a non-crossing ICE entry (oid 5555)
# arrives on that full side. jaxob's full-side wipe evicts the iceberg's
# slice; the slot must be healed (card A2), so 9001 shows a slice again.
_FULL_DEPTH = 6


def _full_side_tape(side: int) -> np.ndarray:
    if side == BUY:
        ice_px, non_px, step, opp, opp_px, inc_px = 40.01, 40.03, 0.01, SELL, 41.50, 40.05
    else:
        ice_px, non_px, step, opp, opp_px, inc_px = 60.09, 60.07, -0.01, BUY, 59.00, 60.05
    rows = [_row(side, ice_px, 3.0, 9001, 1, ICE, 0.4, agent=2)]
    rows += [_row(side, round(non_px + step * i, 2), 0.3, 100 + i, 2 + i, NON, agent=3)
             for i in range(5)]
    rows += [_row(opp, opp_px, 1.0, 4242, 20, NON, agent=4),
             _row(side, inc_px, 2.0, 5555, 30, ICE, 0.5, agent=5)]
    return _tape(*rows)


# Pins measured on the PRE-FIX engine (branch base aab502e + Step-2 docs,
# 2026-10-07) with _state_digest(_replay_state(tape, ...)).
BYTE_PINS: Dict[str, Dict[str, str]] = {
    "nc_buy": {
        "bids": "088abc32fef480c2", "asks": "3f63f07c847e9122", "trades": "48d9b7d2a0e91365",
        "iceberg_meta": "771da7334f65e092", "ts_counter": "cf781ace1b450a30"},
    "nc_sell": {
        "bids": "eb3d37dcc4b38e8c", "asks": "d1a42b2fe97afbaf", "trades": "48d9b7d2a0e91365",
        "iceberg_meta": "9fe439c1d9adcceb", "ts_counter": "927e4858c96760ed"},
    "full_buy": {
        "bids": "97e4ee332c33c709", "asks": "2517f7f0f85ab56d", "trades": "48d9b7d2a0e91365",
        "iceberg_meta": "6cda8f706715efc2", "ts_counter": "1c564c840b43ea28"},
    "full_sell": {
        "bids": "adad2710b003bc75", "asks": "7e89bf1248d9b99e", "trades": "48d9b7d2a0e91365",
        "iceberg_meta": "52430f64b654903d", "ts_counter": "1c564c840b43ea28"},
}


# OMIE-SLICE-STAMP-TAPE-SEC-1 (licence (28)-(29)): P / Q / OID / TID columns of
# the full own side, measured on main @ 1dc4456; tip digests of that side are
# frozen from the E1 + E2 tip (e604788, CPU) and cross-checked equal on the
# Step-2 architect's independent E1 + E2 prototype.
_FULL_SIDE_MAIN_PQOT: Dict[str, str] = {
    "full_buy": "510120735e7831d4", "full_sell": "d47dedc2fd2322f7"}
_FULL_SIDE_POST_FIX_DIGEST: Dict[str, str] = {
    "full_buy": "cd6483e5246542a4", "full_sell": "fa51ccbf888049f1"}


def _byte_case(name: str):
    if name == "nc_buy":
        return NC_BUY, OMIEMarketConfig()
    if name == "nc_sell":
        return NC_SELL, OMIEMarketConfig()
    if name == "full_buy":
        return _full_side_tape(BUY), OMIEMarketConfig(book_depth=_FULL_DEPTH)
    if name == "full_sell":
        return _full_side_tape(SELL), OMIEMarketConfig(book_depth=_FULL_DEPTH)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Module-scoped records
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rec_c15_244():
    return build_record(_load_tape(C15_244))


@pytest.fixture(scope="module")
def rec_c16_no_row9():
    return build_record(_without_oid(_load_tape(C16_C10), C16_ROW9_OID))


@pytest.fixture(scope="module")
def rec_t3():
    return build_record(T3)


@pytest.fixture(scope="module")
def rec_t3b():
    return build_record(T3B)


# ===========================================================================
# (1) C15 shrunk seed 244
# ===========================================================================
class TestC15Shrunk244:
    """SELL ICE 3.4 / 0.6 @ 1005194 (oid 3) rests; BUY ICE 6.2 / 3.1 @ 1005245
    (oid 5) arrives. Oid 5 aggresses with its total 6.2: it takes every
    instance of oid 3 (0.6 x 5 + 0.4) at the RESTING price 1005194, one row per
    instance, and rests R = 2.8."""

    def test_c15_shrunk_244_all_at_resting_price(self, rec_c15_244):
        fills = _fills_of(rec_c15_244, _last_submission(rec_c15_244))
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): oid 5 (peak 31 ticks) is a
        # chain of instances; the 6th row (cumulative 30 -> 34) crosses the
        # instance boundary 31 and splits 1 + 3 (spec §4.2.1 Rule 2 item (2)).
        assert fills == [(1005194, 6, 3, 5)] * 5 + [(1005194, 1, 3, 5),
                                                    (1005194, 3, 3, 5)], (
            "every trade of the incoming iceberg must print at the resting "
            "iceberg's price 1005194 with oid 5 the aggressor (never at the "
            "incoming limit 1005245 with roles swapped)")
        assert sum(q for _, q, _, _ in fills) == 34

    def test_c15_shrunk_244_every_row_aggressor_is_incoming(self, rec_c15_244):
        assert rec_c15_244.fills, "the fixture must trade"
        assert all(f.aggressor_oid == 5 and f.passive_oid == 3 for f in rec_c15_244.fills)

    def test_c15_shrunk_244_remainder_rests(self, rec_c15_244):
        fs = rec_c15_244.final_snapshot
        assert fs.resting(SELL) == ()
        assert [(r[_R_P], r[_R_Q]) for r in _rows_of(fs, BUY, 5)] == [(1005245, 28)], (
            "R = 62 - 34 = 28 ticks; first slice min(31, 28) = 28")
        assert int(_meta_row(fs, 5)[COL_REMAINING_HIDDEN]) == 0


# ===========================================================================
# (2) C16 seed 112, contract 10, without row 9
# ===========================================================================
class TestC16Contract10:
    """With row 9 (SELL NON oid 16542010) removed, the resting BUY ICEwPI
    16542007 (2.5 / 0.5 @ 1003987, PPD -1 tick) still holds every instance when
    the SELL ICE 16542014 (5.0 / 2.5 @ 1003974) arrives. Each drifted instance
    crosses 1003974, so the incoming order takes all five, each at its own
    drifted price."""

    def test_c16_contract10_hops_priced_by_resting_iceberg(self, rec_c16_no_row9):
        rec = rec_c16_no_row9
        sub = next(o.submission for o in rec.accepted.values() if o.oid == 16542014)
        want = [(p, 5, 16542007, 16542014)
                for p in (1003987, 1003986, 1003985, 1003984, 1003983)]
        assert _fills_of(rec, sub) == want

    def test_c16_contract10_incoming_remainder_rests_at_entry_stamp(self, rec_c16_no_row9):
        fs = rec_c16_no_row9.final_snapshot
        # R = 50 - 25 = 25 ticks; first slice min(25, 25) = 25 at its tape second.
        rows = _rows_of(fs, SELL, 16542014)
        assert [(r[_R_P], r[_R_Q], r[_R_SEC], r[_R_NSEC]) for r in rows] == [
            (1003974, 25, 34716, 0)]
        assert _rows_of(fs, BUY, 16542007) == [], "16542007 is fully drained"


# ===========================================================================
# (3) Incoming ICE vs resting ICE
# ===========================================================================
class TestIncomingVsRestingIce:

    def test_incoming_ice_vs_resting_ice_minimal(self, rec_t3):
        assert _fills_of(rec_t3, 1) == T3_FILLS, (
            "all three trades @ 50.00 (the resting price), passive 11, aggressor 22; "
            "never @ 50.40")
        fs = rec_t3.final_snapshot
        assert fs.resting(BUY) == () and fs.resting(SELL) == ()
        assert fs.n_active == 0

    def test_incoming_ice_vs_resting_ice_sell_aggressor_mirror(self):
        rec = build_record(T3_MIRROR)
        assert _fills_of(rec, 1) == T3_MIRROR_FILLS
        fs = rec.final_snapshot
        assert fs.resting(BUY) == () and fs.resting(SELL) == ()
        assert fs.n_active == 0

    def test_incoming_ice_vs_resting_icewpi_drifted_prices(self):
        rec = build_record(T3_ICEWPI)
        assert _fills_of(rec, 1) == T3_ICEWPI_FILLS, (
            "each trade prints at the resting ICEwPI slice's own drifted price "
            "(50.00, 50.05, 50.10), resting oid passive")
        fs = rec.final_snapshot
        assert fs.resting(BUY) == () and fs.resting(SELL) == ()


# ===========================================================================
# (3b) Later slices of the incoming iceberg keep the aggressor role
# ===========================================================================
class TestLaterSlicesKeepAggressorRole:

    def test_incoming_later_slices_keep_aggressor_role(self, rec_t3b):
        assert _fills_of(rec_t3b, 1) == T3B_FILLS, (
            "4 rows x 1.0 @ 50.00 (one per resting instance), incoming oid 22 "
            "the aggressor on every row")
        fs = rec_t3b.final_snapshot
        assert [(r[_R_P], r[_R_Q]) for r in _rows_of(fs, SELL, 11)] == [(px(50.00), 10)], (
            "the resting iceberg keeps its last 1.0 slice")
        assert int(_meta_row(fs, 11)[COL_REMAINING_HIDDEN]) == 0
        assert fs.resting(BUY) == (), "nothing of the incoming order rests"
        assert rec_t3b.filled[22] == 40

    def test_incoming_later_slices_keep_aggressor_role_sell_side(self):
        rec = build_record(T3B_SELL)
        assert _fills_of(rec, 1) == T3B_SELL_FILLS
        fs = rec.final_snapshot
        assert [(r[_R_P], r[_R_Q]) for r in _rows_of(fs, BUY, 704211)] == [(px(47.83), 10)]
        assert fs.resting(SELL) == ()


# ===========================================================================
# (4) Guard: a non-crossing ICE entry is byte-identical
# ===========================================================================
class TestNonCrossingByteIdentical:

    @pytest.mark.parametrize("name", ["nc_buy", "nc_sell"])
    def test_non_crossing_ice_entry_byte_identical(self, name):
        tape, mc = _byte_case(name)
        assert _state_digest(_replay_state(tape, mc)) == BYTE_PINS[name], (
            f"{name}: a non-crossing ICE entry must leave book, tape, "
            f"iceberg_meta, n_active and ts_counter bytes equal to the pre-fix pins")

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["full_buy", "full_sell"])
    def test_non_crossing_ice_entry_full_own_side_heals_evicted_slice(self, name):
        """slow: a separate _submit_order compile for book_depth=6 (~5 s).

        Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence (28)-(29)):
        the healed slice is now stamped at the entry second (30) of the
        incoming ICE, after its first slice by (SEC, NSEC), so the own side's
        digest moves (a heal stamp is an engineering-limit choice, C14
        regime, not a §5.2 rule). Tape, iceberg_meta, ts_counter, book_full,
        the other side and the own side's P / Q / OID / TID columns (pinned on
        main @ 1dc4456) stay equal. The moved side's digest differs from main
        and equals the frozen tip digest (``_FULL_SIDE_POST_FIX_DIGEST``)."""
        tape, mc = _byte_case(name)
        state = _replay_state(tape, mc)
        side = BUY if name == "full_buy" else SELL
        moved = "bids" if side == BUY else "asks"
        got, want = _state_digest(state), BYTE_PINS[name]
        assert {k: v for k, v in got.items() if k != moved} == {
            k: v for k, v in want.items() if k != moved}, (
            f"{name}: byte identity broken on a full own side (card A2: the "
            f"evicted iceberg slice must be healed)")
        arr = np.asarray(state.lob.bids if side == BUY else state.lob.asks)
        m = hashlib.sha256()
        a = np.ascontiguousarray(arr[:, [0, 1, 2, 3]])   # P, Q, OID, TID
        m.update(str((a.dtype.str, a.shape)).encode())
        m.update(a.tobytes())
        assert m.hexdigest()[:16] == _FULL_SIDE_MAIN_PQOT[name], (
            "only SEC / NSEC cells may move")
        assert got[moved] != want[moved]
        assert got[moved] == _FULL_SIDE_POST_FIX_DIGEST[name]
        assert bool(state.book_full), "scenario precondition: the side was full"
        # The evicted iceberg (oid 9001) shows a slice again, stamped at the
        # incoming ICE's entry second, after the incoming first slice (5555).
        live = arr[(arr[:, 2] == 9001) & (arr[:, 1] > 0)]
        assert live.shape[0] == 1, "iceberg 9001's evicted slice must be healed"
        first = arr[(arr[:, 2] == 5555) & (arr[:, 1] > 0)]
        assert first.shape[0] == 1
        assert int(live[0][4]) == 30
        assert (int(first[0][4]), int(first[0][5])) < (int(live[0][4]), int(live[0][5]))


# ===========================================================================
# (5) ts_counter on a crossing entry
# ===========================================================================
class TestCrossingCounterPinned:

    def test_crossing_ice_entry_counter_pinned(self, rec_t3):
        """(3): pre-fix 105, adopted design 107 (card v5.195)."""
        assert rec_t3.final_snapshot.ts_counter == 107

    def test_crossing_ice_entry_counter_pinned_later_slices(self, rec_t3b):
        """(3b): pre-fix 107, adopted design 110 (card v5.195)."""
        assert rec_t3b.final_snapshot.ts_counter == 110

    def test_crossing_ice_entry_meta_unchanged(self, rec_t3, rec_t3b):
        """No implied stamp or registered quantity differs on these shapes, so
        ``iceberg_meta`` equals the pre-fix state: (3) empty; (3b) oid 11 with
        hidden 0, peak 10, base 1004900, SELL, agent 11, PPD 0, slice_count 4."""
        assert rec_t3.final_snapshot.n_active == 0
        assert np.asarray(rec_t3b.final_snapshot.iceberg_meta).tolist() == [
            [11, 0, 10, 1004900, SELL, 11, 0, 4]]


# ===========================================================================
# (6) The remainder registers and shows its first slice at the entry stamp
# ===========================================================================
class TestRemainderRegistersAtEntryStamp:

    def test_remainder_registers_first_slice_at_entry_stamp(self):
        rec = build_record(T6_R_GT_PEAK)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-B): the 1.5 MW row crosses the
        # incoming instance boundary at 1.0 (peak), so it splits 1.0 + 0.5
        # (Rule 2 item (2)); M = 15, k_e = 1, u = 5, R = 25 -> the remainder
        # shows v_1 = min(10 - 5, 25) = 5 ticks and hides 20 (item (4)).
        assert _fills_of(rec, 1) == [(px(50.00), 10, 41, 9312),
                                     (px(50.00), 5, 41, 9312)], (
            "one trade per (counter bid, incoming instance): 1.0 + 0.5 @ 50.00")
        fs = rec.final_snapshot
        rows = _rows_of(fs, BUY, 9312)
        assert [(r[_R_P], r[_R_Q], r[_R_SEC], r[_R_NSEC]) for r in rows] == [
            (px(50.20), 5, 63, 0)], "first slice v_1 = min(1.0 - 0.5, 2.5) = 0.5 at (T, 0)"
        meta = _meta_row(fs, 9312)
        assert int(meta[COL_REMAINING_HIDDEN]) == 20, "registered total R = 2.5 = 0.5 + 2.0"
        assert int(meta[COL_SLICE_COUNT]) == 0
        assert fs.best_ask() is None or fs.best_bid() < fs.best_ask(), "book uncrossed"

    def test_remainder_below_peak_shows_whole_remainder(self):
        rec = build_record(T6_R_LT_PEAK)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): the 3.2 MW row crosses the
        # incoming instance boundary at 2.0 (peak) and splits 2.0 + 1.2. The
        # display is unchanged: M = 32, u = 12, v_1 = min(20 - 12, 8) = 8.
        assert _fills_of(rec, 1) == [(px(61.47), 20, 270001, 6),
                                     (px(61.47), 12, 270001, 6)]
        fs = rec.final_snapshot
        rows = _rows_of(fs, SELL, 6)
        assert [(r[_R_P], r[_R_Q], r[_R_SEC], r[_R_NSEC]) for r in rows] == [
            (px(61.13), 8, 18, 0)], "first slice min(2.0, 0.8) = 0.8 at (T, 0)"
        assert int(_meta_row(fs, 6)[COL_REMAINING_HIDDEN]) == 0
        assert fs.resting(BUY) == ()

    def test_incoming_icewpi_remainder_rests_at_base_price(self):
        """Name predates OMIE-ICE-INCOMING-LADDER-1 (kept for node-id
        stability). Spec §4.2.1 Rule 2 *Incoming iceberg* items (4), (6)
        (amended 2026-10-20): the incoming ICEwPI is a chain of instances.
        Instance 0 (limit 60.00) trades 0.5 @ 60.30 and nothing else crosses,
        so M = 5 ticks, k_e = 0, u = 5: the remainder R = 25 continues
        instance 0 at 60.00 + 0 * 0.10 = 60.00 and shows v_1 = min(10 - 5, 25)
        = 5 ticks, hiding 20."""
        rec = build_record(T6_ICEWPI)
        assert _fills_of(rec, 1) == [(px(60.30), 5, 812, 33417)]
        fs = rec.final_snapshot
        rows = _rows_of(fs, SELL, 33417)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-A): display v_1 = min(P - u, R)
        # = 5 ticks at b + k_e d = 60.00 (k_e = 0), not min(P, R) = 10.
        assert [(r[_R_P], r[_R_Q], r[_R_SEC], r[_R_NSEC]) for r in rows] == [
            (px(60.00), 5, 25, 0)]
        meta = _meta_row(fs, 33417)
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-A): hidden R - v_1 = 25 - 5 = 20.
        assert int(meta[COL_REMAINING_HIDDEN]) == 20
        assert int(meta[COL_SLICE_COUNT]) == 0

    @pytest.mark.slow
    def test_iceberg_swept_whole_takes_no_slot(self):
        """slow: a separate _submit_order compile for max_icebergs=2 (~5 s).

        Card A3: the side table (capacity 2) is full of two resting BUY
        icebergs. A SELL NON 2.0 @ 55.00 rests; an incoming BUY ICE 1.5 / 0.5 @
        55.00 is swept whole (R = 0), so it never registers and the call does
        not set ``truncated``."""
        tape = _tape(
            _row(BUY, 40.00, 3.0, 5101, 1, ICE, 1.0, agent=1),
            _row(BUY, 40.10, 3.0, 5102, 2, ICE, 1.0, agent=2),
            _row(SELL, 55.00, 2.0, 5150, 3, NON, agent=3),
            _row(BUY, 55.00, 1.5, 5199, 4, ICE, 0.5, agent=4),
        )
        state = _replay_state(tape, OMIEMarketConfig(max_icebergs=2))
        assert not bool(state.truncated), "an iceberg swept whole takes no slot"
        assert int(state.n_active) == 2
        asks = np.asarray(state.lob.asks)
        assert asks[(asks[:, 2] == 5150) & (asks[:, 1] > 0)][:, 1].tolist() == [5]
        bids = np.asarray(state.lob.bids)
        assert bids[(bids[:, 2] == 5199) & (bids[:, 1] > 0)].shape[0] == 0


# ===========================================================================
# (7) Guard: FILL-REPORT-1 conservation
# ===========================================================================
_CONSERVATION_TAPES = {
    "c15_244": lambda: _load_tape(C15_244),
    "c16_no_row9": lambda: _without_oid(_load_tape(C16_C10), C16_ROW9_OID),
    "t3": lambda: T3,
    "t3_mirror": lambda: T3_MIRROR,
    "t3_icewpi": lambda: T3_ICEWPI,
    "t3b": lambda: T3B,
    "t3b_sell": lambda: T3B_SELL,
}


@pytest.mark.parametrize("name", sorted(_CONSERVATION_TAPES))
def test_fill_report_invariant_kept(name):
    """For every ICE submission, ``SubmitResult.trade_qty`` equals the qty of
    the tape rows that submission wrote involving its own OID, and it writes no
    foreign row."""
    rec = build_record(_CONSERVATION_TAPES[name]())
    ice_subs = [o for o in rec.accepted.values() if o.exec_restriction == ICE]
    reports = {r.submission: r for r in rec.engine_reports}
    assert any(reports[o.submission].traded for o in ice_subs), (
        "fixture must contain a crossing ICE entry")
    for o in ice_subs:
        rows = [f for f in rec.fills if f.submission == o.submission]
        assert all(o.oid in (f.passive_oid, f.aggressor_oid) for f in rows), (
            f"{name}: submission {o.submission} wrote foreign tape rows")
        own = sum(f.qty_int for f in rows)
        reported = int(round(reports[o.submission].trade_qty_mw * 10))
        assert reported == own, (
            f"{name}: oid {o.oid} reported {reported} ticks, tape shows {own}")


# ===========================================================================
# (8) Guard: one compile
# ===========================================================================
@pytest.mark.slow
def test_one_compile_preserved():
    """slow: a fresh _submit_order compile + the (1)-(7) default-config tapes (~10 s).

    The cache is cleared first so entries left by other modules in the same
    (xdist) worker, with other static configs, do not pollute the count."""
    if not hasattr(_submit_order, "_cache_size"):
        pytest.skip("_submit_order does not expose `_cache_size`")
    if hasattr(_submit_order, "clear_cache"):
        _submit_order.clear_cache()
    else:  # pragma: no cover - older jax
        _submit_order._clear_cache()

    tapes = [f() for f in _CONSERVATION_TAPES.values()]
    tapes += [T6_R_GT_PEAK, T6_R_LT_PEAK, T6_ICEWPI, NC_BUY, NC_SELL]
    for t in tapes:
        _replay_state(t)

    size = _submit_order._cache_size()
    assert size == 1, f"expected one _submit_order compile across (1)-(7), got {size}"
