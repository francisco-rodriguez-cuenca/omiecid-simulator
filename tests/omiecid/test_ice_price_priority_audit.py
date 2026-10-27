"""AUDIT tests for OMIE-ICE-PRICE-PRIORITY-1 (Step 5, audit mode).

Independent post-implementation verification that one matching pass of the
NON/IOC and FOK loops never walks a worse opposite price level while a
better-priced resting iceberg still has a slice to regenerate
(market-spec §6.1 price-time priority; §4.2.1 Rule 2; Matching Algorithm
§3.3 / §5.1.3). Regenerated slices rank as NEW entries (an older order at
the same price goes first), and an ICEwPI slice ranks at its drifted price.

Derived from the story card (agile-plan.md v5.194) and market-spec only; the
locked test file was NOT read (audit-mode isolation).

Probe groups:
  A. ``_cap_pass_qty`` helper contract (card signature): P*, V, stall guard,
     n_active masking, same-side / no-hidden / non-crossing exclusions.
  B. Hand-derived end-to-end walks (both sides, NON / IOC / FOK, ICEwPI,
     same-price time priority, regen-vs-older-order ranking), with OIDs of
     1-6 digits and off-round prices.
  C. Seeded reference-walk fuzz: an independent pure-Python price-time
     priority walk (slices re-ranked as new entries) vs the engine tape.
  D. Loud pass bound (``truncated``) and one-compile preservation.

Device policy: CPU-hermetic (conftest pins the cpu platform).
"""

from __future__ import annotations

import random

import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    _cap_pass_qty,
    _submit_order,
)
from omiecid_simulator.omiecid.iceberg_manager import COL_ORDER_ID
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.order_validator import validate_order

# ``submit_order`` drives the module-level jitted ``_submit_order``.
pytestmark = [pytest.mark.jit]

_CID = 52
_BUY = int(Side.BUY)
_SELL = int(Side.SELL)
_NON = ExecRestriction.NON
_IOC = ExecRestriction.IOC
_FOK = ExecRestriction.FOK
_ICE = ExecRestriction.ICE

_P, _Q, _PASS, _AGRS = 0, 1, 2, 3
_NTRADE_CAP = 100
_QTY_ABS = 1e-4


def _contract() -> Contract:
    return Contract(
        contract_id=_CID,
        delivery_start_minutes=14 * 60 + 45,
        delivery_end_minutes=15 * 60,
        gct_minutes=14 * 60 + 45 - 60,
        status=ContractStatus.OPEN,
    )


@pytest.fixture(scope="module")
def pc():
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def book(pc):
    return ContractBook(_contract(), pc, OMIEMarketConfig())


def _mk(pc, side, price, qty, oid, ts, er=_NON, peak=0.0, ppd=0.0):
    o = make_order(
        contract_id=_CID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=oid, timestamp=ts,
        exec_restriction=er, peak_qty=peak, ppd=ppd,
    )
    assert bool(validate_order(o, pc, True)), f"fixture bug: invalid {o}"
    return o


def _off(pc) -> int:
    return int(round(-pc.price_min / pc.price_tick))


def _pint(pc, price) -> int:
    return _off(pc) + int(round(price / pc.price_tick))


def _qint(pc, qty) -> int:
    return int(round(qty / pc.qty_tick))


def _dec(pc, price_int) -> float:
    return round((int(price_int) - _off(pc)) * pc.price_tick, 2)


def _new_rows(before, after) -> np.ndarray:
    b = np.asarray(before)
    a = np.asarray(after)
    return a[np.any(b != a, axis=1)]


def _submit(book, state, order, allow_full_tape=False):
    before = state.lob.trades
    s2, r = book.submit_order(state, order)
    rows = _new_rows(before, s2.lob.trades)
    assert allow_full_tape or rows.shape[0] < _NTRADE_CAP, (
        "fixture bug: tape near saturation")
    return s2, r, rows


def _trades(pc, rows):
    """[(price EUR, qty MW, passive oid, aggressor oid)] in tape order."""
    return [
        (_dec(pc, r[_P]), round(abs(int(r[_Q])) * pc.qty_tick, 4),
         int(r[_PASS]), int(r[_AGRS]))
        for r in rows
    ]


def _merge(seq):
    """Merge consecutive (price, passive) fills (row-split invariance)."""
    out = []
    for p, q, pas in seq:
        if out and out[-1][0] == p and out[-1][2] == pas:
            out[-1] = (p, round(out[-1][1] + q, 4), pas)
        else:
            out.append((p, round(q, 4), pas))
    return out


def _resting(state, side, pc):
    """[(price EUR, qty MW, oid)] of the resting rows of one side."""
    arr = np.asarray(state.lob.bids if side == _BUY else state.lob.asks)
    live = arr[(arr[:, 0] >= 0) & (arr[:, 1] > 0)]
    return [(_dec(pc, r[0]), round(int(r[1]) * pc.qty_tick, 4), int(r[2]))
            for r in live]


def _assert_price_priority_postcondition(pc, state, agg_side, trades):
    """§6.1 postcondition of one aggressive submit.

    Tape prices are monotone (never better after worse), and no opposite
    resting order is strictly better than the worst printed price.
    """
    prices = [t[0] for t in trades]
    if not prices:
        return
    if agg_side == _BUY:
        assert prices == sorted(prices), f"BUY walk not ascending: {prices}"
        worst = max(prices)
        better = [r for r in _resting(state, _SELL, pc) if r[0] < worst]
    else:
        assert prices == sorted(prices, reverse=True), (
            f"SELL walk not descending: {prices}")
        worst = min(prices)
        better = [r for r in _resting(state, _BUY, pc) if r[0] > worst]
    assert not better, (
        f"opposite resting order better than worst print {worst}: {better}")


# ---------------------------------------------------------------------------
# A. _cap_pass_qty helper contract
# ---------------------------------------------------------------------------


class TestCapPassQtyHelper:

    def _cap(self, pc, state, side, price, remaining_mw, n_active=None):
        n = state.n_active if n_active is None else n_active
        return int(_cap_pass_qty(state.lob, side, _pint(pc, price),
                                 _qint(pc, remaining_mw), state.iceberg_meta, n))

    def test_no_iceberg_returns_remaining(self, book, pc):
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _SELL, 33.41, 1.0, 71, 1))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 33.58, 2.0, 72, 2))
        assert self._cap(pc, s, _BUY, 40.00, 2.7) == _qint(pc, 2.7)

    def test_opposite_iceberg_caps_at_visible_up_to_next_slice(self, book, pc):
        """BUY ICE 2.0/0.6 @ 61.37 + NON 0.4 @ 61.37 (newer) + NON 1.0 @ 61.02.
        SELL aggressor 2.5: P* = 61.37, V = 0.6 + 0.4 = 1.0 -> cap 1.0."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 61.37, 2.0, 513, 1, _ICE, 0.6))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 61.37, 0.4, 9, 2))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 61.02, 1.0, 40404, 3))
        assert self._cap(pc, s, _SELL, 58.00, 2.5) == _qint(pc, 1.0)
        # remaining below V is never raised
        assert self._cap(pc, s, _SELL, 58.00, 0.3) == _qint(pc, 0.3)

    def test_same_side_iceberg_ignored(self, book, pc):
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 61.37, 2.0, 513, 1, _ICE, 0.6))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 63.11, 1.0, 8, 2))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 63.47, 1.0, 88, 3))
        assert self._cap(pc, s, _BUY, 64.00, 1.7) == _qint(pc, 1.7)

    def test_n_active_masks_slots(self, book, pc):
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 61.37, 2.0, 513, 1, _ICE, 0.6))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 61.02, 1.0, 40404, 2))
        assert self._cap(pc, s, _SELL, 58.00, 2.5, n_active=0) == _qint(pc, 2.5)

    def test_non_crossing_next_slice_does_not_cap(self, book, pc):
        """SELL ICEwPI base 50.03 ppd +0.30: next slice 50.33 does not cross a
        BUY limit 50.20 -> no cap (the 50.13 level stays walkable)."""
        s = book.init()
        s, _ = book.submit_order(
            s, _mk(pc, _SELL, 50.03, 3.0, 9001, 1, _ICE, 1.0, ppd=0.30))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 50.13, 0.5, 23, 2))
        assert self._cap(pc, s, _BUY, 50.20, 4.0) == _qint(pc, 4.0)
        # with a limit at which the drifted slice crosses, the cap binds at V
        # = 1.0 @ 50.03 + 0.5 @ 50.13 (both at least as good as 50.33)
        assert self._cap(pc, s, _BUY, 50.40, 4.0) == _qint(pc, 1.5)

    def test_icewpi_cap_uses_drifted_not_base_price(self, book, pc):
        """BUY ICEwPI base 50.10 ppd -0.20 (next slice 49.90) + BUY ICE @ 50.00.
        SELL limit 49.00: P* = 50.00 (the ICE; ICEwPI's 49.90 is worse), so V =
        1.0 @ 50.10 + 1.0 @ 50.00 = 2.0 (a base-price P* would give V = 1.0)."""
        s = book.init()
        s, _ = book.submit_order(
            s, _mk(pc, _BUY, 50.10, 3.0, 61, 1, _ICE, 1.0, ppd=-0.20))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 50.00, 2.0, 620, 2, _ICE, 1.0))
        assert self._cap(pc, s, _SELL, 49.00, 3.0) == _qint(pc, 2.0)

    def test_stall_guard_qualifying_slot_without_visible_slice(self, book, pc):
        """A qualifying opposite slot whose visible slice is gone (V = 0)
        returns ``remaining`` unchanged: never a zero-qty pass."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _SELL, 47.13, 2.3, 8, 1, _ICE, 0.7))
        wiped = s.lob._replace(asks=jnp.full_like(s.lob.asks, -1))
        got = int(_cap_pass_qty(wiped, _BUY, _pint(pc, 49.91), _qint(pc, 2.0),
                                s.iceberg_meta, s.n_active))
        assert got == _qint(pc, 2.0)

    def test_cap_always_positive_for_positive_remaining(self, book, pc):
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _SELL, 47.13, 2.3, 8, 1, _ICE, 0.7))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 47.29, 1.0, 704, 2))
        for rem in (0.1, 0.5, 0.7, 0.8, 9.9):
            got = self._cap(pc, s, _BUY, 49.91, rem)
            assert 0 < got <= _qint(pc, rem)


# ---------------------------------------------------------------------------
# B. Hand-derived end-to-end walks
# ---------------------------------------------------------------------------


class TestHandDerivedWalks:

    def test_sell_side_two_icebergs_better_slices_first_ioc(self, book, pc):
        """Asks: ICE 2.3/0.7 @ 47.13 (oid 8), ICE 3.0/1.0 @ 47.29 (oid 70412).
        BUY IOC 2.0 @ 49.91 -> 0.7, 0.7, 0.6 all @ 47.13 vs oid 8."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _SELL, 47.13, 2.3, 8, 1, _ICE, 0.7))
        s, _ = book.submit_order(
            s, _mk(pc, _SELL, 47.29, 3.0, 70412, 2, _ICE, 1.0))
        s, r, rows = _submit(book, s, _mk(pc, _BUY, 49.91, 2.0, 311, 3, _IOC))
        t = _trades(pc, rows)
        assert _merge([(p, q, pa) for p, q, pa, _ in t]) == [(47.13, 2.0, 8)]
        assert all(a == 311 for *_, a in t)
        assert float(r.trade_qty) == pytest.approx(2.0, abs=_QTY_ABS)
        asks = _resting(s, _SELL, pc)
        assert (47.13, 0.1, 8) in asks and (47.29, 1.0, 70412) in asks
        assert not [b for b in _resting(s, _BUY, pc) if b[2] == 311]
        _assert_price_priority_postcondition(pc, s, _BUY, t)

    def test_iceberg_exhausted_before_worse_non_level(self, book, pc):
        """Bids: ICE 1.5/0.4 @ 61.37 (oid 513), NON 0.9 @ 61.02 (oid 44).
        SELL NON 2.2 @ 58.00 -> 1.5 @ 61.37 (all slices) then 0.7 @ 61.02."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 61.37, 1.5, 513, 1, _ICE, 0.4))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 61.02, 0.9, 44, 2))
        s, r, rows = _submit(book, s, _mk(pc, _SELL, 58.00, 2.2, 2906, 3))
        t = _trades(pc, rows)
        assert _merge([(p, q, pa) for p, q, pa, _ in t]) == [
            (61.37, 1.5, 513), (61.02, 0.7, 44)]
        assert _resting(s, _BUY, pc) == [(61.02, 0.2, 44)]
        assert int(s.n_active) == 0
        _assert_price_priority_postcondition(pc, s, _SELL, t)

    def test_icewpi_drift_interleaves_with_non_levels(self, book, pc):
        """Asks: ICEwPI 3.0/1.0 base 50.03 ppd +0.20 (oid 9001), NON 0.5 @
        50.13 (23), NON 1.0 @ 50.31 (77), NON 2.0 @ 50.57 (314).
        BUY NON 4.0 @ 50.50 walks 50.03(ice), 50.13, 50.23(ice), 50.31,
        50.43(ice 0.5 of 1.0); 50.57 untouched; nothing rests on the bid."""
        s = book.init()
        s, _ = book.submit_order(
            s, _mk(pc, _SELL, 50.03, 3.0, 9001, 1, _ICE, 1.0, ppd=0.20))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 50.13, 0.5, 23, 2))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 50.31, 1.0, 77, 3))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 50.57, 2.0, 314, 4))
        s, r, rows = _submit(book, s, _mk(pc, _BUY, 50.50, 4.0, 55, 5))
        t = _trades(pc, rows)
        assert _merge([(p, q, pa) for p, q, pa, _ in t]) == [
            (50.03, 1.0, 9001), (50.13, 0.5, 23), (50.23, 1.0, 9001),
            (50.31, 1.0, 77), (50.43, 0.5, 9001)]
        asks = sorted(_resting(s, _SELL, pc))
        assert asks == [(50.43, 0.5, 9001), (50.57, 2.0, 314)]
        assert _resting(s, _BUY, pc) == []
        _assert_price_priority_postcondition(pc, s, _BUY, t)

    def test_same_price_icebergs_keep_time_priority_then_worse_level(
            self, book, pc):
        """Bids: ICE 2.0/1.0 @ 39.87 (oid 120, ts1), ICE 2.0/1.0 @ 39.87 (oid
        340000, ts2), NON 1.0 @ 39.80 (oid 56). SELL NON 4.6 @ 30.00: 4.0 MW
        @ 39.87 (120 first, then 340000), then 0.6 @ 39.80."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 39.87, 2.0, 120, 1, _ICE, 1.0))
        s, _ = book.submit_order(
            s, _mk(pc, _BUY, 39.87, 2.0, 340000, 2, _ICE, 1.0))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 39.80, 1.0, 56, 3))
        s, r, rows = _submit(book, s, _mk(pc, _SELL, 30.00, 4.6, 7, 4))
        t = _trades(pc, rows)
        assert [x[2] for x in t[:2]] == [120, 340000]
        at_best = [x for x in t if x[0] == 39.87]
        assert sum(x[1] for x in at_best) == pytest.approx(4.0, abs=_QTY_ABS)
        assert t.index(at_best[-1]) < min(
            i for i, x in enumerate(t) if x[0] == 39.80)
        assert _merge([(p, q, pa) for p, q, pa, _ in t])[-1] == (39.80, 0.6, 56)
        assert _resting(s, _BUY, pc) == [(39.80, 0.4, 56)]
        _assert_price_priority_postcondition(pc, s, _SELL, t)

    def test_older_same_price_non_before_regenerated_slice_sell_mirror(
            self, book, pc):
        """Asks: ICE 1.6/0.8 @ 72.19 (oid 4, ts10), NON 0.3 @ 72.19 (oid
        65432, ts12), NON 1.0 @ 72.26 (oid 901, ts13). BUY NON 1.5 @ 75.00:
        0.8 vs 4, 0.3 vs 65432, 0.4 vs 4's regenerated slice, all @ 72.19."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _SELL, 72.19, 1.6, 4, 10, _ICE, 0.8))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 72.19, 0.3, 65432, 12))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 72.26, 1.0, 901, 13))
        s, r, rows = _submit(book, s, _mk(pc, _BUY, 75.00, 1.5, 3030, 20))
        t = _trades(pc, rows)
        assert _merge([(p, q, pa) for p, q, pa, _ in t]) == [
            (72.19, 0.8, 4), (72.19, 0.3, 65432), (72.19, 0.4, 4)]
        assert sorted(_resting(s, _SELL, pc)) == [
            (72.19, 0.4, 4), (72.26, 1.0, 901)]
        _assert_price_priority_postcondition(pc, s, _BUY, t)

    def test_ioc_remainder_killed_after_iceberg_and_crossing_level(
            self, book, pc):
        """Bids: ICE 2.0/0.5 @ 45.55 (oid 7), NON 0.3 @ 45.31 (oid 880),
        NON 3.0 @ 45.10 (oid 81). SELL IOC 2.4 @ 45.20 -> 2.0 @ 45.55, 0.3 @
        45.31, 0.1 killed; 45.10 untouched; nothing rests on the ask."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 45.55, 2.0, 7, 1, _ICE, 0.5))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 45.31, 0.3, 880, 2))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 45.10, 3.0, 81, 3))
        s, r, rows = _submit(book, s, _mk(pc, _SELL, 45.20, 2.4, 2121, 4, _IOC))
        t = _trades(pc, rows)
        assert _merge([(p, q, pa) for p, q, pa, _ in t]) == [
            (45.55, 2.0, 7), (45.31, 0.3, 880)]
        assert float(r.trade_qty) == pytest.approx(2.3, abs=_QTY_ABS)
        assert _resting(s, _SELL, pc) == []
        assert _resting(s, _BUY, pc) == [(45.10, 3.0, 81)]

    def test_fok_iceberg_regen_before_worse_level(self, book, pc):
        """Bids: ICE 2.0/1.0 @ 50.10 (oid 31), NON 1.0 @ 50.00 (oid 4242).
        FOK SELL 1.5 @ 49.00 -> (50.10, 1.0), (50.10, 0.5) both vs oid 31;
        the NON at 50.00 is untouched."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 50.10, 2.0, 31, 1, _ICE, 1.0))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 50.00, 1.0, 4242, 2))
        s, r, rows = _submit(book, s, _mk(pc, _SELL, 49.00, 1.5, 606, 3, _FOK))
        t = _trades(pc, rows)
        assert _merge([(p, q, pa) for p, q, pa, _ in t]) == [(50.10, 1.5, 31)]
        assert (50.00, 1.0, 4242) in _resting(s, _BUY, pc)
        assert float(r.trade_qty) == pytest.approx(1.5, abs=_QTY_ABS)

    def test_fok_icewpi_drift_interleaves_non_buy_aggressor(self, book, pc):
        """Asks: ICEwPI 3.0/1.0 base 70.11 ppd +0.05 (oid 600), NON 1.0 @
        70.14 (oid 61). FOK BUY 2.5 @ 70.20 -> 1.0 @ 70.11, 1.0 @ 70.14,
        0.5 @ 70.16 (ICEwPI slice 2 at its drifted price)."""
        s = book.init()
        s, _ = book.submit_order(
            s, _mk(pc, _SELL, 70.11, 3.0, 600, 1, _ICE, 1.0, ppd=0.05))
        s, _ = book.submit_order(s, _mk(pc, _SELL, 70.14, 1.0, 61, 2))
        s, r, rows = _submit(book, s, _mk(pc, _BUY, 70.20, 2.5, 17, 3, _FOK))
        t = _trades(pc, rows)
        assert _merge([(p, q, pa) for p, q, pa, _ in t]) == [
            (70.11, 1.0, 600), (70.14, 1.0, 61), (70.16, 0.5, 600)]
        _assert_price_priority_postcondition(pc, s, _BUY, t)

    def test_fok_killed_leaves_book_byte_identical(self, book, pc):
        """Bids: ICE 1.0/0.5 @ 50.10, NON 0.4 @ 50.00: matchable 1.4 < 1.5."""
        s = book.init()
        s, _ = book.submit_order(s, _mk(pc, _BUY, 50.10, 1.0, 31, 1, _ICE, 0.5))
        s, _ = book.submit_order(s, _mk(pc, _BUY, 50.00, 0.4, 4242, 2))
        s2, r, rows = _submit(book, s, _mk(pc, _SELL, 49.00, 1.5, 606, 3, _FOK))
        assert rows.shape[0] == 0 and not bool(r.traded)
        np.testing.assert_array_equal(np.asarray(s.lob.bids),
                                      np.asarray(s2.lob.bids))
        np.testing.assert_array_equal(np.asarray(s.iceberg_meta),
                                      np.asarray(s2.iceberg_meta))


# ---------------------------------------------------------------------------
# C. Seeded reference-walk fuzz (independent pure-Python price-time walk)
# ---------------------------------------------------------------------------


def _reference_walk(resting, agg_side, limit_int, qty):
    """Price-time priority walk; a regenerated slice is a NEW entry.

    ``resting``: list of dicts (price, vis, hidden, peak, base, ppd, k, oid)
    in arrival order, all integer ticks. Returns [(price_int, qty, oid)].
    """
    seq = 0
    book = []
    for e in resting:
        book.append(dict(e, seq=seq))
        seq += 1
    out = []
    rem = qty
    better = (lambda a, b: a < b) if agg_side == _BUY else (lambda a, b: a > b)
    while rem > 0:
        cross = [e for e in book
                 if (e["price"] <= limit_int if agg_side == _BUY
                     else e["price"] >= limit_int)]
        if not cross:
            break
        best = cross[0]
        for e in cross[1:]:
            if better(e["price"], best["price"]) or (
                    e["price"] == best["price"] and e["seq"] < best["seq"]):
                best = e
        f = min(rem, best["vis"])
        out.append((best["price"], f, best["oid"]))
        rem -= f
        best["vis"] -= f
        if best["vis"] == 0:
            book.remove(best)
            if best["hidden"] > 0:
                k = best["k"] + 1
                sl = min(best["peak"], best["hidden"])
                book.append(dict(best, vis=sl, hidden=best["hidden"] - sl, k=k,
                                 price=best["base"] + k * best["ppd"], seq=seq))
                seq += 1
    return out


def _gen_scenario(seed, pc):
    """Small one-sided resting book (NON / ICE / correct-sign ICEwPI) and a
    crossing NON / IOC aggressor. Prices off round numbers, OIDs 1-6 digits.
    Every iceberg base sits on its own price level (no same-time regen ties).
    """
    rng = random.Random(seed)
    rest_side = rng.choice([_BUY, _SELL])
    agg_side = _SELL if rest_side == _BUY else _BUY
    anchor = rng.randint(3000, 9000) + rng.choice([7, 13, 29, 41, 53, 67])
    sign = -1 if rest_side == _BUY else 1  # direction of "worse" for resting
    levels = rng.sample(range(0, 40), 6)
    oids = rng.sample(range(2, 999_999), 12)
    resting, orders, ts = [], [], 1
    used_ice_levels = set()
    n_ice = 0
    for i in range(rng.randint(3, 7)):
        lvl = rng.choice(levels)
        p_int_rel = anchor + sign * lvl  # cents
        qty = rng.randint(2, 25)  # 0.1 MW ticks
        oid = oids[i]
        is_ice = n_ice < 3 and lvl not in used_ice_levels and rng.random() < 0.5
        price = p_int_rel / 100.0
        if is_ice:
            n_ice += 1
            used_ice_levels.add(lvl)
            peak = rng.randint(1, max(1, qty // 2))
            ppd_c = rng.choice([0, 0, 0, 2, 5])  # cents of worsening drift
            ppd_c = sign * ppd_c
            o = _mk(pc, rest_side, price, qty / 10, oid, ts, _ICE,
                    peak / 10, ppd_c / 100.0)
            vis = min(peak, qty)
            resting.append(dict(price=p_int_rel, vis=vis, hidden=qty - vis,
                                peak=peak, base=p_int_rel, ppd=ppd_c, k=0,
                                oid=oid))
        else:
            o = _mk(pc, rest_side, price, qty / 10, oid, ts)
            resting.append(dict(price=p_int_rel, vis=qty, hidden=0, peak=0,
                                base=p_int_rel, ppd=0, k=0, oid=oid))
        orders.append(o)
        ts += 1
    worst = max(levels) + 6
    limit_rel = anchor + sign * rng.randint(5, worst)
    agg_qty = rng.randint(5, 80)
    er = rng.choice([_NON, _IOC])
    agg = _mk(pc, agg_side, limit_rel / 100.0, agg_qty / 10, oids[-1], ts, er)
    return orders, resting, agg, agg_side, limit_rel, agg_qty


@pytest.mark.parametrize("seed", list(range(24)))
def test_fuzz_engine_matches_reference_walk(book, pc, seed):
    orders, resting, agg, agg_side, limit_rel, agg_qty = _gen_scenario(
        seed + 7100, pc)
    s = book.init()
    for o in orders:
        s, _ = book.submit_order(s, o)
    s, r, rows = _submit(book, s, agg)
    assert not bool(s.truncated), "fixture bug: pass bound reached"
    got = _merge([(p, q, pa) for p, q, pa, _ in _trades(pc, rows)])
    ref = _merge([(round(p / 100.0, 2), round(q / 10, 4), oid)
                  for p, q, oid in _reference_walk(resting, agg_side,
                                                   limit_rel, agg_qty)])
    assert got == ref, f"seed {seed}: engine {got} != reference {ref}"
    _assert_price_priority_postcondition(pc, s, agg_side, _trades(pc, rows))


# ---------------------------------------------------------------------------
# D. Loud pass bound and one-compile preservation
# ---------------------------------------------------------------------------


def _pass_bound_scenario(book, pc):
    s = book.init()
    s, _ = book.submit_order(s, _mk(pc, _BUY, 50.13, 12.0, 98765, 1, _ICE, 0.1))
    s, _ = book.submit_order(s, _mk(pc, _BUY, 50.02, 1.0, 12, 2))
    # One 0.1 MW trade per pass: the 100-row tape fills exactly at the bound
    # (fresh book), so every row on it belongs to this call.
    s, r, rows = _submit(book, s, _mk(pc, _SELL, 49.00, 12.5, 345, 3),
                         allow_full_tape=True)
    return s, r, _trades(pc, rows)


def test_pass_bound_is_loud_and_capped_passes_stay_on_iceberg(book, pc):
    """BUY ICE 12.0/0.1 @ 50.13 (120 slices) over NON 1.0 @ 50.02. SELL NON
    12.5 @ 49.00: each capped pass takes one 0.1 slice, so the 100-pass bound
    binds and ``truncated`` is set (loud). Every capped pass (the first 99
    rows) prints at 50.13 against the iceberg; the tape is price-monotone."""
    s, r, t = _pass_bound_scenario(book, pc)
    assert bool(s.truncated)
    assert len(t) == _NTRADE_CAP
    assert all(x[0] == 50.13 and x[2] == 98765 for x in t[:99])
    prices = [x[0] for x in t]
    assert prices == sorted(prices, reverse=True)


# Post-bound behaviour is NOT pinned: the uncapped re-submit at the 100-pass
# ceiling is documented degradation (architect ruling (a), 2026-10-06).


def test_one_compile_across_all_submit_shapes(book, pc):
    """Cap adds no recompilation: NON / IOC / FOK / ICE / ICEwPI submits on
    the default book leave ``_submit_order``'s cache size unchanged."""
    s = book.init()
    s, _ = book.submit_order(s, _mk(pc, _BUY, 50.10, 2.0, 31, 1, _ICE, 1.0))
    size0 = _submit_order._cache_size()
    s, _ = book.submit_order(
        s, _mk(pc, _BUY, 50.20, 3.0, 32, 2, _ICE, 1.0, ppd=-0.05))
    s, _ = book.submit_order(s, _mk(pc, _BUY, 50.00, 1.0, 4242, 3))
    s, _ = book.submit_order(s, _mk(pc, _SELL, 49.00, 1.5, 606, 4, _FOK))
    s, _ = book.submit_order(s, _mk(pc, _SELL, 49.00, 1.2, 607, 5, _IOC))
    s, _ = book.submit_order(s, _mk(pc, _SELL, 49.50, 9.0, 608, 6))
    assert _submit_order._cache_size() == size0
    assert int(np.sum(np.asarray(s.iceberg_meta)[:, COL_ORDER_ID] == 31)) <= 1
