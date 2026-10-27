"""Locked behavioural-contract tests for Story OMIE-ICE-PRICE-PRIORITY-1.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-ICE-PRICE-PRIORITY-1 — a
pass of the NON/IOC and FOK match loops never walks a worse price level while a
better-priced iceberg still has a slice to regenerate" (certification fix pass,
S3 / plan D1).

Rule (market-spec §6.1 price-time priority, a hard rule; §4.2.1 Rule 2; Matching
Algorithm §3.3 / §5.1.3): a price level is not exhausted while an iceberg there
still has hidden quantity. An aggressor must therefore take a better-priced
iceberg's NEXT slice before it trades any worse price level. Each regenerated
slice is ranked by price and time as a NEW entry: an older order resting at the
same price goes first, and an ICEwPI slice ranks at its DRIFTED price
(``base + slice_index * ppd``).

Contract (Test-first contract row, numbering preserved):
  (1) C14 two BUY icebergs fixture -> (50.10, 1.0), (50.10, 0.5), both on the
      first iceberg; plus the SELL-side mirror and the iceberg-over-NON variant.
                                                              RED before the fix.
  (2) C15 shrunk seed 246 -> the trade list equals the price-time reference walk.
                                                              RED before the fix.
  (3) an older same-price NON trades before the iceberg's regenerated slice;
      the worse level is untouched (both sides).              RED before the fix.
  (4) Guard: an ICEwPI slice ranks at its drifted price (both sides).
                                                              GREEN before and after.
  (5) FOK, C15 shrunk seed 249 -> 21, 21, 3 @ 1004403; the NON at 1004456 untouched.
                                                              RED before the fix.
  (6) Guard: no iceberg / no worse crossing level -> final book, trade tape,
      ``iceberg_meta`` and ``ts_counter`` bytes equal pins measured on main.
                                                              GREEN before and after.
  (7) C14c (``C14_seed300000.npz``): price order (P1 drained before P2), and the
      100-pass bound binding is LOUD (``truncated`` True).    RED before the fix.
  (8) Guard: ``_submit_order`` compiles once across (1)-(7).  GREEN before and after.
  (9) Guard: a qualifying opposite iceberg slot whose visible slice is gone
      (full-side wipe) never stalls the loop; I2 holds.       GREEN before and after.

Every expected trade list below was derived by hand from the spec (price
first, then time; a regenerated slice is a new entry) and is cross-checked
against :func:`_reference_walk`, an independent price-time walk written from
the same rules. Incidental constants are varied (both sides, OIDs of 1-6
digits, prices off round numbers).

Units: prices in int ticks (``round((eur + 9999) / 0.01)``, so 50.10 EUR ->
1004910), quantities in 0.1 MW ticks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_data_loader import _row_to_order_traced
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, _submit_order
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from tests.omiecid.reference.omie_invariants import (
    build_record,
    check_i2_quantity_conservation,
)


# Every test here runs the jitted submit path (T1.1 marker lint).
pytestmark = pytest.mark.jit


REPO_ROOT = Path(__file__).resolve().parents[2]
# The ONE fixture-path constant (card, Dependencies: OMIE-CERT-HARNESS-2 may move
# the fixtures to tests/omiecid/fixtures/cert/; whichever story merges second updates
# this line only — not a re-baseline).
FIXTURE_DIR = REPO_ROOT / "notebooks" / "certification" / "fixtures"
# OMIE-CERT-HARNESS-2 (J6) moved only the three C15 shrunk counterexamples to
# tests/omiecid/fixtures/cert/; the C14/C16 fixtures stay in FIXTURE_DIR. Path only,
# not a re-baseline (``FIXTURE_DIR / <absolute path>`` is that absolute path).
SHRUNK_FIXTURE_DIR = REPO_ROOT / "tests" / "omiecid" / "fixtures" / "cert"

C14_TWO_BUY_ICEBERGS = "C14_price_priority_two_buy_icebergs.npz"
C14C = "C14_seed300000.npz"
C15_246 = SHRUNK_FIXTURE_DIR / "C15_shrunk_seed100000246.npz"
C15_249 = SHRUNK_FIXTURE_DIR / "C15_shrunk_seed100000249.npz"
BYTE_IDENTICAL_FIXTURES = (
    "C04_seed100_iceberg_no_cancel.npz",
    "C05_seed0.npz",
    "C10_seed2_req74.npz",
    "C13_seed96.npz",
)

BUY, SELL = 0, 1
NON, IOC, FOK, ICE = 0, 1, 2, 3
CONTRACT_ID = 37

# Tape columns (the loader's 13-wide int32 layout).
_T_SIDE, _T_PRICE, _T_QTY, _T_OID, _T_EXEC, _T_PEAK, _T_PPD = 1, 2, 3, 4, 7, 11, 12

# Contract used by the direct replay (same as the record builder's).
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
    r[6] = ts
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


def _real_rows(tape: np.ndarray) -> List[np.ndarray]:
    return [r for r in tape if int(r[_T_OID]) >= 1]


# ---------------------------------------------------------------------------
# The independent price-time reference walk (written from the rules)
# ---------------------------------------------------------------------------
def _reference_walk(tape: np.ndarray) -> List[Tuple[int, int, int]]:
    """Trades ``(price_tick, qty_tick, passive_oid)`` of the LAST real row of
    ``tape`` when it aggresses against the resting rows before it.

    Scope: the resting rows do not cross each other (true for every case here).
    Rules: best price first; at one price, earliest entry first; an iceberg's
    next slice (``min(peak, hidden)`` at ``base + k * ppd`` for slice ``k``)
    enters as a NEW entry the moment its visible slice is used up. FOK: all or
    nothing against the total (visible + hidden) crossing quantity.
    """
    rows = _real_rows(tape)
    *resting, agg = rows
    book = []   # dicts: side, price, vis, hidden, peak, base, ppd, k, oid, seq
    seq = 0
    for r in resting:
        qty, peak = int(r[_T_QTY]), int(r[_T_PEAK])
        is_ice = int(r[_T_EXEC]) == ICE and peak > 0
        vis = min(peak, qty) if is_ice else qty
        book.append(dict(side=int(r[_T_SIDE]), price=int(r[_T_PRICE]), vis=vis,
                         hidden=qty - vis, peak=peak, base=int(r[_T_PRICE]),
                         ppd=int(r[_T_PPD]), k=0, oid=int(r[_T_OID]), seq=seq))
        seq += 1
    a_side, a_px, a_qty = int(agg[_T_SIDE]), int(agg[_T_PRICE]), int(agg[_T_QTY])

    def crosses(p):
        return p <= a_px if a_side == BUY else p >= a_px

    def rank(e):
        return (e["price"] if a_side == BUY else -e["price"], e["seq"])

    if int(agg[_T_EXEC]) == FOK:
        avail = 0
        for e in book:
            if e["side"] == a_side:
                continue
            if crosses(e["price"]):
                avail += e["vis"]
            hidden, k = e["hidden"], e["k"]
            while hidden > 0:
                k += 1
                s = min(e["peak"], hidden)
                hidden -= s
                if crosses(e["base"] + k * e["ppd"]):
                    avail += s
        if avail < a_qty:
            return []

    trades = []
    remaining = a_qty
    while remaining > 0:
        live = [e for e in book if e["side"] != a_side and e["vis"] > 0
                and crosses(e["price"])]
        if not live:
            break
        e = min(live, key=rank)
        q = min(remaining, e["vis"])
        trades.append((e["price"], q, e["oid"]))
        remaining -= q
        e["vis"] -= q
        if e["vis"] == 0 and e["hidden"] > 0:
            e["k"] += 1
            e["vis"] = min(e["peak"], e["hidden"])
            e["hidden"] -= e["vis"]
            e["price"] = e["base"] + e["k"] * e["ppd"]
            e["seq"] = seq
            seq += 1
    return trades


# ---------------------------------------------------------------------------
# Engine observation helpers
# ---------------------------------------------------------------------------
def _fills(rec, oid: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """``(price_tick, qty_tick, passive_oid)`` in trade-row order, optionally
    only those with aggressor ``oid``."""
    return [(f.price_int, f.qty_int, f.passive_oid) for f in rec.fills
            if oid is None or f.aggressor_oid == oid]


def _visible(snap, oid: int, side: int) -> int:
    return snap.visible_qty(oid, side)


def _accounted(rec, oid: int) -> int:
    cancelled = sum(c.qty_int for c in rec.cancelled_or_expired.get(oid, ()))
    return (rec.filled[oid] + rec.still_resting[oid] + cancelled
            + rec.hidden_remaining[oid] + rec.evicted.get(oid, 0))


def _new_book(cid: int, market_config: OMIEMarketConfig = OMIEMarketConfig()):
    contract = Contract(contract_id=cid, delivery_start_minutes=_DELIVERY_START,
                        delivery_end_minutes=_DELIVERY_END, gct_minutes=_GCT,
                        status=ContractStatus.OPEN)
    return ContractBook(contract, OMIEProductConfig(), market_config)


def _replay_state(tape: np.ndarray):
    """Submit every real row via the public ``ContractBook.submit_order`` on a
    fresh ``book.init()`` (default configs); return the final state."""
    rows = _real_rows(tape)
    book = _new_book(int(rows[0][0]))
    pc = OMIEProductConfig()
    state = book.init()
    for r in rows:
        state, res = book.submit_order(state, _row_to_order_traced(jnp.asarray(r), pc))
        assert bool(res.accepted), f"row {r.tolist()} rejected"
    return state


def _state_digest(state) -> Dict[str, str]:
    """sha256 of the final book, trade tape, iceberg_meta (+ n_active) and ts_counter."""
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
# Scenario tapes (synthetic; spec values derived in the comments)
# ---------------------------------------------------------------------------
# (1b) SELL-side mirror of the C14 fixture: two SELL icebergs 2.0 / 1.0 at 50.37
# (oid 4) and 50.52 (oid 70213); BUY NON 1.5 @ 51.06. Price-time: 1.0 @ 50.37
# (oid 4 slice 1), oid 4's slice 2 (1.0 @ 50.37) appears, 0.5 @ 50.37 from it.
# The 50.52 iceberg is untouched.
SELL_MIRROR = _tape(
    _row(SELL, 50.37, 2.0, 4, 3, ICE, 1.0, agent=2),
    _row(SELL, 50.52, 2.0, 70213, 5, ICE, 1.0, agent=3),
    _row(BUY, 51.06, 1.5, 905, 9, NON, agent=4),
)
SELL_MIRROR_TRADES = [(px(50.37), 10, 4), (px(50.37), 5, 4)]

# (1c) iceberg over a NON level: BUY ICE 2.4 / 1.1 @ 48.73 (oid 6021), BUY NON
# 0.9 @ 48.66 (oid 87), SELL NON 1.7 @ 47.99. Price-time: 1.1 @ 48.73 (slice 1),
# slice 2 = min(1.1, 1.3) = 1.1 @ 48.73 appears, 0.6 from it. NON 87 untouched;
# oid 6021 keeps 0.5 visible + 0.2 hidden.
ICE_OVER_NON = _tape(
    _row(BUY, 48.73, 2.4, 6021, 2, ICE, 1.1, agent=2),
    _row(BUY, 48.66, 0.9, 87, 4, NON, agent=3),
    _row(SELL, 47.99, 1.7, 455512, 8, NON, agent=4),
)
ICE_OVER_NON_TRADES = [(px(48.73), 11, 6021), (px(48.73), 6, 6021)]

# (3) the card's case: BUY ICE 2.0 / 1.0 @ 50.10 ts10 (oid 11), BUY NON 1.0 @
# 50.10 ts12 (oid 33), BUY NON 1.0 @ 50.00 ts13 (oid 44), SELL NON 2.5 @ 49.00
# ts20. At 50.10: oid 11 slice 1 (oldest), then oid 33 (older than the slice 2
# that regenerates after it), then 0.5 from oid 11's slice 2. Oid 44 untouched.
OLDER_SAME_PRICE_BUY = _tape(
    _row(BUY, 50.10, 2.0, 11, 10, ICE, 1.0, agent=1),
    _row(BUY, 50.10, 1.0, 33, 12, NON, agent=2),
    _row(BUY, 50.00, 1.0, 44, 13, NON, agent=3),
    _row(SELL, 49.00, 2.5, 55, 20, NON, agent=4),
)
OLDER_SAME_PRICE_BUY_TRADES = [(px(50.10), 10, 11), (px(50.10), 10, 33), (px(50.10), 5, 11)]

# (3b) SELL-side variant, other constants: SELL ICE 2.0 / 1.0 @ 61.37 (oid 5),
# SELL NON 1.0 @ 61.37 (oid 640022), SELL NON 1.0 @ 61.48 (oid 7071),
# BUY NON 2.5 @ 62.03.
OLDER_SAME_PRICE_SELL = _tape(
    _row(SELL, 61.37, 2.0, 5, 10, ICE, 1.0, agent=3),
    _row(SELL, 61.37, 1.0, 640022, 12, NON, agent=2),
    _row(SELL, 61.48, 1.0, 7071, 13, NON, agent=1),
    _row(BUY, 62.03, 2.5, 98, 20, NON, agent=4),
)
OLDER_SAME_PRICE_SELL_TRADES = [(px(61.37), 10, 5), (px(61.37), 10, 640022), (px(61.37), 5, 5)]

# (4) ICEwPI guard: BUY ICEwPI 3.0 / 1.0 @ 50.10 PPD -0.20 (oid 702), BUY ICE
# 2.0 / 1.0 @ 50.00 (oid 81), SELL IOC 1.5 @ 49.00. After slice 1 the ICEwPI's
# slice 2 sits at its DRIFTED price 49.90 < 50.00, so 0.5 trades at 50.00
# against oid 81. (A cap at the base price 50.10 would stop the IOC at 1.0.)
ICEWPI_BUY = _tape(
    _row(BUY, 50.10, 3.0, 702, 1, ICE, 1.0, -0.20, agent=1),
    _row(BUY, 50.00, 2.0, 81, 2, ICE, 1.0, agent=2),
    _row(SELL, 49.00, 1.5, 33019, 3, IOC, agent=3),
)
ICEWPI_BUY_TRADES = [(px(50.10), 10, 702), (px(50.00), 5, 81)]

# (4b) SELL mirror: SELL ICEwPI 3.0 / 1.0 @ 48.27 PPD +0.20 (oid 3; slice 2 at
# 48.47), SELL ICE 2.0 / 1.0 @ 48.37 (oid 41005), BUY IOC 1.5 @ 49.13.
ICEWPI_SELL = _tape(
    _row(SELL, 48.27, 3.0, 3, 1, ICE, 1.0, 0.20, agent=1),
    _row(SELL, 48.37, 2.0, 41005, 2, ICE, 1.0, agent=2),
    _row(BUY, 49.13, 1.5, 77, 3, IOC, agent=3),
)
ICEWPI_SELL_TRADES = [(px(48.27), 10, 3), (px(48.37), 5, 41005)]

# (6) no-iceberg multi-level sweeps (one per side).
NO_ICE_SWEEP_BUY = _tape(
    _row(SELL, 61.17, 0.7, 12, 1, NON, agent=1),
    _row(SELL, 61.23, 1.3, 3407, 2, NON, agent=2),
    _row(SELL, 61.23, 0.4, 88, 3, NON, agent=3),
    _row(SELL, 61.41, 2.2, 91234, 4, NON, agent=1),
    _row(BUY, 58.02, 0.9, 7, 5, NON, agent=2),
    _row(BUY, 61.30, 2.9, 500017, 6, NON, agent=4),
)
NO_ICE_SWEEP_SELL = _tape(
    _row(BUY, 39.87, 1.1, 610, 1, NON, agent=2),
    _row(BUY, 39.79, 0.6, 29, 2, NON, agent=1),
    _row(BUY, 39.66, 3.0, 400213, 3, NON, agent=3),
    _row(SELL, 42.05, 0.5, 8, 4, NON, agent=1),
    _row(SELL, 39.70, 2.4, 70001, 5, IOC, agent=4),
)

# Pins measured on main (f9ae381) with _state_digest(_replay_state(tape)).
BYTE_PINS: Dict[str, Dict[str, str]] = {
    "C04_seed100_iceberg_no_cancel.npz": {
        "bids": "0013857f03c7119d", "asks": "1404aa613105d249", "trades": "e146a19bb2d1b7a3",
        "iceberg_meta": "81296478bf63839e", "ts_counter": "cf781ace1b450a30"},
    "C05_seed0.npz": {
        "bids": "1404aa613105d249", "asks": "509e26ff48d76279", "trades": "4e6b0f34a497fd10",
        "iceberg_meta": "441f30e2a854cd36", "ts_counter": "f7bc8ea50fe6f532"},
    "C10_seed2_req74.npz": {
        "bids": "1404aa613105d249", "asks": "820abbfbe4043281", "trades": "48d9b7d2a0e91365",
        "iceberg_meta": "441f30e2a854cd36", "ts_counter": "0e1fac1a1dd502c3"},
    "C13_seed96.npz": {
        "bids": "715e3d6ffd612664", "asks": "4c67103484aaa57f", "trades": "4316653962f1a6c9",
        "iceberg_meta": "6be492fd2a690832", "ts_counter": "332013bc7d563d14"},
    "no_ice_sweep_buy": {
        "bids": "3834ae09d2f185df", "asks": "a396cff6e027e098", "trades": "2b5f67dce8c1bd2c",
        "iceberg_meta": "441f30e2a854cd36", "ts_counter": "332013bc7d563d14"},
    "no_ice_sweep_sell": {
        "bids": "cb825a3799fe9480", "asks": "7e940e3ddaf3fa49", "trades": "2183b434e02c10bb",
        "iceberg_meta": "441f30e2a854cd36", "ts_counter": "bc72a98cd17a71a0"},
}


def _byte_case_tape(name: str) -> np.ndarray:
    if name == "no_ice_sweep_buy":
        return NO_ICE_SWEEP_BUY
    if name == "no_ice_sweep_sell":
        return NO_ICE_SWEEP_SELL
    return _load_tape(name)


# ---------------------------------------------------------------------------
# Module-scoped records (one replay per tape, shared by the assertions)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rec_c14_two_buy():
    return build_record(_load_tape(C14_TWO_BUY_ICEBERGS))


@pytest.fixture(scope="module")
def rec_c15_246():
    return build_record(_load_tape(C15_246))


@pytest.fixture(scope="module")
def rec_c15_249():
    return build_record(_load_tape(C15_249))


@pytest.fixture(scope="module")
def rec_c14c():
    return build_record(_load_tape(C14C))


# ===========================================================================
# Reference-walk self-check (pure Python; pins the hand derivations)
# ===========================================================================
class TestReferenceWalkMatchesHandDerivation:
    """The hand-derived expectations below equal the independent walk."""

    @pytest.mark.parametrize("tape,want", [
        (SELL_MIRROR, SELL_MIRROR_TRADES),
        (ICE_OVER_NON, ICE_OVER_NON_TRADES),
        (OLDER_SAME_PRICE_BUY, OLDER_SAME_PRICE_BUY_TRADES),
        (OLDER_SAME_PRICE_SELL, OLDER_SAME_PRICE_SELL_TRADES),
        (ICEWPI_BUY, ICEWPI_BUY_TRADES),
        (ICEWPI_SELL, ICEWPI_SELL_TRADES),
    ], ids=["sell_mirror", "ice_over_non", "older_buy", "older_sell",
            "icewpi_buy", "icewpi_sell"])
    def test_walk_synthetic(self, tape, want):
        assert _reference_walk(tape) == want

    def test_walk_fixtures(self):
        assert _reference_walk(_load_tape(C14_TWO_BUY_ICEBERGS)) == [
            (1004910, 10, 11), (1004910, 5, 11)]
        assert _reference_walk(_load_tape(C15_246)) == (
            [(1005103, 22, 3)] * 4 + [(1005103, 2, 3)] + [(1005101, 32, 5)] * 2)
        assert _reference_walk(_load_tape(C15_249)) == [
            (1004403, 21, 2), (1004403, 21, 2), (1004403, 3, 2)]


# ===========================================================================
# (1) Two icebergs: the better iceberg's next slice before the worse level
# ===========================================================================
class TestBetterSliceFirst:

    def test_two_buy_icebergs_better_slice_first(self, rec_c14_two_buy):
        """C14 fixture: BUY ICE 2.0/1.0 @ 50.10 (oid 11), BUY ICE 2.0/1.0 @ 50.00
        (oid 22), SELL NON 1.5 @ 49.00 (oid 33) -> (50.10, 1.0), (50.10, 0.5),
        both passive on oid 11."""
        rec = rec_c14_two_buy
        assert _fills(rec, 33) == [(1004910, 10, 11), (1004910, 5, 11)], (
            "the second fill must take oid 11's regenerated slice @ 50.10, "
            "not the worse 50.00 level")
        assert rec.filled[33] == 15

    def test_two_buy_icebergs_resting_state(self, rec_c14_two_buy):
        fs = rec_c14_two_buy.final_snapshot
        # oid 11: 2.0 - 1.5 = 0.5 MW left, all in its regenerated slice @ 50.10
        assert _visible(fs, 11, BUY) == 5
        assert fs.hidden_qty(11) == 0
        assert [r[0] for r in fs.resting(BUY) if r[2] == 11] == [1004910]
        # oid 22 untouched: 1.0 visible + 1.0 hidden
        assert (_visible(fs, 22, BUY), fs.hidden_qty(22)) == (10, 10)

    def test_two_buy_icebergs_nothing_rests_above_last_trade(self, rec_c14_two_buy):
        rec = rec_c14_two_buy
        last_px = rec.fills[-1].price_int
        assert rec.final_snapshot.best_bid() <= last_px, (
            "a better-priced bid than the last printed trade still rests: the "
            "aggressor walked past it")

    def test_sell_mirror_better_slice_first(self):
        rec = build_record(SELL_MIRROR)
        assert _fills(rec, 905) == SELL_MIRROR_TRADES
        fs = rec.final_snapshot
        assert (_visible(fs, 4, SELL), fs.hidden_qty(4)) == (5, 0)
        assert (_visible(fs, 70213, SELL), fs.hidden_qty(70213)) == (10, 10)
        assert fs.best_ask() >= rec.fills[-1].price_int

    def test_iceberg_over_non_level_better_slice_first(self):
        rec = build_record(ICE_OVER_NON)
        assert _fills(rec, 455512) == ICE_OVER_NON_TRADES, (
            "the iceberg's slice 2 @ 48.73 must trade before the NON @ 48.66")
        fs = rec.final_snapshot
        assert (_visible(fs, 6021, BUY), fs.hidden_qty(6021)) == (5, 2)
        assert _visible(fs, 87, BUY) == 9, "NON 87 @ 48.66 must be untouched"


# ===========================================================================
# (2) C15 shrunk seed 246: the trade list equals the reference walk
# ===========================================================================
class TestC15Shrunk246:

    def test_c15_shrunk_246_reference_walk(self, rec_c15_246):
        """BUY ICE 9.0/2.2 @ 1005103 (oid 3), BUY ICE 6.4/3.2 @ 1005101 (oid 5),
        SELL IOC 16.8 @ 1005090 (oid 27): oid 3 is drained slice by slice before
        any trade at 1005101; then oid 5's two slices; 1.4 MW IOC remainder killed."""
        tape = _load_tape(C15_246)
        want = [(1005103, 22, 3)] * 4 + [(1005103, 2, 3)] + [(1005101, 32, 5)] * 2
        assert _reference_walk(tape) == want
        assert _fills(rec_c15_246, 27) == want

    def test_c15_shrunk_246_accounting(self, rec_c15_246):
        rec = rec_c15_246
        assert rec.filled[27] == 154
        assert sum(c.qty_int for c in rec.cancelled_or_expired[27]) == 14
        assert rec.final_snapshot.resting(BUY) == ()
        assert rec.final_snapshot.n_active == 0


# ===========================================================================
# (3) An older same-price order goes before the regenerated slice
# ===========================================================================
class TestOlderSamePriceFirst:

    def test_older_same_price_order_before_regenerated_slice(self):
        rec = build_record(OLDER_SAME_PRICE_BUY)
        assert _fills(rec, 55) == OLDER_SAME_PRICE_BUY_TRADES, (
            "1.0 vs oid 11, 1.0 vs oid 33, 0.5 vs oid 11's regenerated slice, "
            "all @ 50.10")
        fs = rec.final_snapshot
        assert _visible(fs, 44, BUY) == 10, "oid 44 @ 50.00 must be untouched"
        assert (_visible(fs, 11, BUY), fs.hidden_qty(11)) == (5, 0)
        assert _visible(fs, 33, BUY) == 0

    def test_older_same_price_order_before_regenerated_slice_sell_side(self):
        rec = build_record(OLDER_SAME_PRICE_SELL)
        assert _fills(rec, 98) == OLDER_SAME_PRICE_SELL_TRADES
        fs = rec.final_snapshot
        assert _visible(fs, 7071, SELL) == 10, "oid 7071 @ 61.48 must be untouched"
        assert (_visible(fs, 5, SELL), fs.hidden_qty(5)) == (5, 0)


# ===========================================================================
# (4) Guard: an ICEwPI slice ranks at its drifted price
# ===========================================================================
class TestIcewpiDriftedPrice:

    def test_icewpi_slice_ranks_at_drifted_price(self):
        rec = build_record(ICEWPI_BUY)
        assert _fills(rec, 33019) == ICEWPI_BUY_TRADES, (
            "the ICEwPI's slice 2 sits at 49.90 (< 50.00): the IOC's 0.5 MW "
            "trades the 50.00 iceberg")
        assert rec.filled[33019] == 15
        fs = rec.final_snapshot
        # ICEwPI slice 2 rests at its drifted price 50.10 - 0.20 = 49.90
        assert [r[0] for r in fs.resting(BUY) if r[2] == 702] == [px(49.90)]

    def test_icewpi_slice_ranks_at_drifted_price_sell_side(self):
        rec = build_record(ICEWPI_SELL)
        assert _fills(rec, 77) == ICEWPI_SELL_TRADES
        assert rec.filled[77] == 15
        fs = rec.final_snapshot
        assert [r[0] for r in fs.resting(SELL) if r[2] == 3] == [px(48.47)]


# ===========================================================================
# (5) FOK, C15 shrunk seed 249
# ===========================================================================
class TestFokC15Shrunk249:

    def test_fok_c15_shrunk_249(self, rec_c15_249):
        """SELL ICE 8.7/2.1 @ 1004403 (oid 2), SELL NON 4.8 @ 1004456 (oid 7),
        BUY FOK 4.5 @ 1004504 (oid 17): fills 2.1, 2.1, 0.3 all on oid 2."""
        rec = rec_c15_249
        assert _fills(rec, 17) == [(1004403, 21, 2), (1004403, 21, 2), (1004403, 3, 2)]
        assert rec.filled[17] == 45

    def test_fok_c15_shrunk_249_non_untouched(self, rec_c15_249):
        fs = rec_c15_249.final_snapshot
        assert _visible(fs, 7, SELL) == 48, "the NON at 1004456 must be untouched"
        # oid 2: 8.7 - 4.5 = 4.2 MW left = slice 3 (2.1 - 0.3 = 1.8) + 2.4 hidden
        assert (_visible(fs, 2, SELL), fs.hidden_qty(2)) == (18, 24)


# ===========================================================================
# (6) Guard: byte-identical where no worse level would be walked
# ===========================================================================
class TestByteIdentical:

    @pytest.mark.parametrize("name", sorted(BYTE_PINS))
    def test_no_iceberg_or_no_worse_level_byte_identical(self, name):
        state = _replay_state(_byte_case_tape(name))
        got = _state_digest(state)
        moved = _E2_MOVED_SIDE.get(name)
        if moved is None:
            assert got == BYTE_PINS[name], (
                f"{name}: final book / trade tape / iceberg_meta / ts_counter bytes "
                f"moved from the pins measured on main")
            return
        # OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence (26)): only the stored
        # [SEC, NSEC] cells of the moved side may change.
        want = BYTE_PINS[name]
        assert {k: v for k, v in got.items() if k != moved} == {
            k: v for k, v in want.items() if k != moved}
        side_arr = np.asarray(getattr(state.lob, moved))
        m = hashlib.sha256()
        a = np.ascontiguousarray(side_arr[:, [0, 1, 2, 3]])   # P, Q, OID, TID
        m.update(str((a.dtype.str, a.shape)).encode())
        m.update(a.tobytes())
        assert m.hexdigest()[:16] == _E2_MOVED_SIDE_PQOT[name], (
            "only SEC / NSEC cells may move")
        assert got[moved] != want[moved], (
            f"{name}: the {moved} stamp cells must move (the regenerated "
            f"slice is stamped at its activation event)")
        assert got[moved] == _E2_POST_FIX_MOVED_DIGEST[name]


# OMIE-SLICE-STAMP-TAPE-SEC-1 re-baseline (licence (26)): on C04 the fixed
# engine moves only the book SEC / NSEC cells of ``bids`` (the BUY iceberg's
# slice, activated by the FOK, is stamped at the FOK's second instead of a
# counter value); tape, iceberg_meta and ts_counter stay equal. The P / Q /
# OID / TID columns of ``bids`` are pinned (measured on main @ 1dc4456). The
# tip ``bids`` digest is frozen from the E1 + E2 tip (e604788, CPU) and
# cross-checked equal on the Step-2 architect's independent E1 + E2 prototype.
_E2_MOVED_SIDE = {"C04_seed100_iceberg_no_cancel.npz": "bids"}
_E2_MOVED_SIDE_PQOT = {"C04_seed100_iceberg_no_cancel.npz": "44ea96a4c084948f"}
_E2_POST_FIX_MOVED_DIGEST: Dict[str, str] = {
    "C04_seed100_iceberg_no_cancel.npz": "4a3c889ce539438f"}


# ===========================================================================
# (7) C14c: price order, and the pass bound binds loudly
# ===========================================================================
class TestC14cPassBound:
    """C14c: four BUY icebergs (1002778 oid 719707 109.6/1.8; 1002765 oid 447964
    175.5/2.9; 1002749 oid 95094 6.1/0.1; 1002737 oid 158898 77.7/1.3) and a
    SELL NON 368.9 @ 1002737 (oid 784545). In price order the SELL drains P1
    (61 slices) before P2; draining P2 too needs more than the engine's
    100-pass bound, so the bound binds — which must be flagged (``truncated``).
    The post-bound NON re-submit (one uncapped call after the bound binds) is
    out of scope, per C14 "Does not certify": nothing is asserted about fills
    or book state it produces (oids 95094 / 158898 below P2 included)."""

    def test_c14c_hits_pass_bound_loudly(self, rec_c14c):
        post = rec_c14c.post_snapshots[-1]
        assert post.truncated is True, (
            "the 100-pass bound bound with work remaining: it must be LOUD")

    def test_c14c_p1_drained_before_p2(self, rec_c14c):
        fs = rec_c14c.final_snapshot
        assert _visible(fs, 719707, BUY) == 0 and fs.hidden_qty(719707) == 0, (
            "P1 (1002778) must be fully drained")

    def test_c14c_fills_in_price_order(self, rec_c14c):
        prices = [f.price_int for f in rec_c14c.fills]
        assert all(a >= b for a, b in zip(prices, prices[1:])), (
            "resting BUY vs SELL aggressor: fill prices must be non-increasing "
            f"in tape order, got {prices}")
        p1 = [i for i, px in enumerate(prices) if px == 1002778]
        p2 = [i for i, px in enumerate(prices) if px == 1002765]
        if p1 and p2:
            assert max(p1) < min(p2), (
                "every P1 (1002778) fill must precede the first P2 (1002765) fill")


# ===========================================================================
# (8) Guard: one compile across (1)-(7)
# ===========================================================================
@pytest.mark.slow
def test_one_compile_preserved():
    """slow: a fresh _submit_order compile + every (1)-(7) tape (~10 s).

    The cache is cleared first so entries left by other modules in the same
    (xdist) worker, with other static configs, do not pollute the count."""
    if not hasattr(_submit_order, "_cache_size"):
        pytest.skip("_submit_order does not expose `_cache_size`")
    if hasattr(_submit_order, "clear_cache"):
        _submit_order.clear_cache()
    else:  # pragma: no cover - older jax
        _submit_order._clear_cache()

    tapes = [_load_tape(n) for n in (C14_TWO_BUY_ICEBERGS, C15_246, C15_249, C14C)]
    tapes += [SELL_MIRROR, ICE_OVER_NON, OLDER_SAME_PRICE_BUY, OLDER_SAME_PRICE_SELL,
              ICEWPI_BUY, ICEWPI_SELL]
    tapes += [_byte_case_tape(n) for n in sorted(BYTE_PINS)]
    for t in tapes:
        _replay_state(t)

    size = _submit_order._cache_size()
    assert size == 1, f"expected one _submit_order compile across (1)-(7), got {size}"


# ===========================================================================
# (9) Guard: a qualifying slot without a visible slice never stalls the loop
# ===========================================================================
# book_depth = 6. Six orders rest at ONE price on one side: iceberg 9001
# (3.0 / 0.4) and five NONs. A seventh, worse-priced NON arrives: the side is
# full, so jaxob wipes the side's worst level, which is the whole (only) level,
# iceberg slice included. Iceberg 9001's slot stays live with 2.6 MW hidden but no visible
# slice, and its next slice price crosses the aggressor that follows, while no
# visible opposite quantity is priced at least as well (V = 0).
_WIPE_DEPTH = 6


def _wipe_tape(side: int) -> Tuple[np.ndarray, int, int]:
    if side == BUY:
        p0, p_worse, p_agg, agg_side = 40.03, 40.01, 39.50, SELL
    else:
        p0, p_worse, p_agg, agg_side = 60.03, 60.05, 60.50, BUY
    rows = [_row(side, p0, 3.0, 9001, 1, ICE, 0.4, agent=2)]
    rows += [_row(side, p0, 0.3, 100 + i, 2 + i, NON, agent=3) for i in range(5)]
    rows += [_row(side, p_worse, 0.2, 4242, 40, NON, agent=1),
             _row(agg_side, p_agg, 1.1, 777, 50, NON, agent=4)]
    return _tape(*rows), px(p0), side


@pytest.mark.slow
@pytest.mark.parametrize("side", [BUY, SELL], ids=["bid_side_wiped", "ask_side_wiped"])
def test_qualifying_slot_without_visible_slice_never_stalls(side):
    """slow: a separate _submit_order compile for book_depth=6 (~5 s)."""
    tape, p0, rest_side = _wipe_tape(side)
    rec = build_record(tape, market_config=OMIEMarketConfig(book_depth=_WIPE_DEPTH))

    # Precondition: right before the aggressor, iceberg 9001 has hidden
    # quantity but no visible slice, and nothing visible at or better than p0.
    pre = rec.pre_snapshots[-1]
    assert pre.hidden_qty(9001) > 0 and _visible(pre, 9001, rest_side) == 0, (
        "scenario precondition: the wipe must leave 9001 with hidden qty and no slice")
    better_or_equal = [r for r in pre.resting(rest_side)
                       if (r[0] >= p0 if rest_side == BUY else r[0] <= p0)]
    assert better_or_equal == [], "scenario precondition: V = 0"

    post = rec.post_snapshots[-1]
    assert post.truncated is False, "the loop must not spin to the pass bound"
    assert _accounted(rec, 777) == 11, "the aggressor's 1.1 MW must be fully accounted"
    assert rec.filled[777] > 0, "the aggressor faces crossing liquidity: it must trade"
    i2 = check_i2_quantity_conservation(rec, allow_eviction=True)
    assert i2.status == "pass", [v.message for v in i2.violations]
