"""Locked behavioural-contract tests for Story OMIE-ICEWPI-SIGN-REJECT-1.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-ICEWPI-SIGN-REJECT-1 —
the validator rejects an ICEwPI whose PPD has the wrong sign (BUY PPD > 0,
SELL PPD < 0)" (certification fix pass, S1 / plan D8).

Domain rule: ``docs/omiecid/market-spec.md`` §13.6 (DA-PPD-2026-04) — "ICEwPI buy
orders must have PPD < 0; sell PPD > 0 (hard)"; OMIE Rules Annex §4.5 makes the
sign a condition of entry.

Contract (Test-first contract row, numbering preserved):
  (1) BUY ICE with PPD > 0 is REJECTED; the state is untouched (book bytes,
      trade tape, ``ts_counter``, ``n_active``, ``iceberg_meta``).
  (2) SELL ICE with PPD < 0 is REJECTED; same post-conditions.
  (3) A wrong-sign ICE that would cross makes no trade; the resting
      counter-order is intact.
  (4) ``validate_order`` returns False for (1) and (2), eager and jitted.
  (5) Guard: conforming signs (BUY PPD < 0, SELL PPD > 0) are accepted and
      rest their visible peak.
  (6) Guard: plain ICE (PPD = 0) accepted on both sides.
  (7) Guard: a sub-half-tick PPD (+0.004) rounds to 0 ticks -> plain ICE,
      accepted (pins the Method-A rounding decision).
  (8) Guard: the PPD of a non-ICE order (NON / IOC / FOK) is not checked.
  (9) One-compile invariant: a mixed session leaves
      ``_submit_order._cache_size() == 1``.

(1)-(4) are RED before the fix (the validator never reads ``order.ppd``);
(5)-(9) are guards and are GREEN before and after.

Incidental constants are deliberately varied (both sides, PPD magnitudes
0.01 / 0.10 / 1.00, order IDs of 2-6 digits, prices off round numbers) so the
contract does not hold by accident of a single fixture value.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from omiecid_simulator.omiecid.contract_book import _submit_order
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    make_order,
)
from omiecid_simulator.omiecid.order_validator import validate_order


# Every test here traces or calls the jitted submit path (T1.1 marker lint).
pytestmark = pytest.mark.jit


CONTRACT_ID = 37


def _order(
    side,
    price,
    quantity,
    order_id,
    exec_restriction=ExecRestriction.NON,
    peak_qty=0.0,
    ppd=0.0,
    agent_id=0,
    timestamp=100,
):
    return make_order(
        contract_id=CONTRACT_ID,
        side=side,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=exec_restriction,
        peak_qty=peak_qty,
        ppd=ppd,
    )


def _ice(side, price, quantity, peak, ppd, order_id):
    return _order(
        side, price, quantity, order_id,
        exec_restriction=ExecRestriction.ICE, peak_qty=peak, ppd=ppd,
    )


def _leaves_equal(a, b) -> bool:
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _assert_state_untouched(pre, post):
    """Every observable leaf of the book state equals its pre-submit value."""
    assert np.array_equal(np.asarray(post.lob.bids), np.asarray(pre.lob.bids)), (
        "bid side changed: a rejected order must never enter the SOB"
    )
    assert np.array_equal(np.asarray(post.lob.asks), np.asarray(pre.lob.asks)), (
        "ask side changed: a rejected order must never enter the SOB"
    )
    assert np.array_equal(np.asarray(post.lob.trades), np.asarray(pre.lob.trades)), (
        "trade tape changed: a rejected order must make no trade"
    )
    assert int(post.ts_counter) == int(pre.ts_counter), (
        "ts_counter advanced: a rejected order must take no timestamp"
    )
    assert int(post.n_active) == int(pre.n_active), (
        "n_active changed: a rejected order must register no iceberg"
    )
    assert np.array_equal(
        np.asarray(post.iceberg_meta), np.asarray(pre.iceberg_meta)
    ), "iceberg_meta changed: a rejected order must register no iceberg"
    assert _leaves_equal(pre, post), "some OMIEBookState leaf changed on rejection"


def _bids(book, state):
    return book.get_anonymous_lob(state)[0]


def _asks(book, state):
    return book.get_anonymous_lob(state)[1]


def _level_qty(levels, price):
    return sum(q for p, q in levels if p == pytest.approx(price, abs=1e-6))


# ===========================================================================
# (1) BUY ICE with PPD > 0 is rejected.
# ===========================================================================


class TestBuyPositivePpdRejected:

    def test_buy_positive_ppd_rejected(self, book):
        """Card case: BUY ICE 5.0 @ 50.37 peak 1.0 PPD +0.10 -> rejected,
        state untouched."""
        pre = book.init()
        order = _ice(Side.BUY, 50.37, 5.0, 1.0, 0.10, order_id=4127)
        post, result = book.submit_order(pre, order)
        assert bool(result.accepted) is False
        assert bool(result.traded) is False
        assert float(result.trade_qty) == pytest.approx(0.0)
        _assert_state_untouched(pre, post)

    @pytest.mark.parametrize(
        "ppd,price,order_id",
        [(0.01, 48.91, 63), (1.00, 37.73, 908152)],
    )
    def test_buy_positive_ppd_rejected_other_magnitudes(
        self, book, ppd, price, order_id
    ):
        """PPD magnitudes 0.01 (one tick) and 1.00 are rejected too."""
        pre = book.init()
        order = _ice(Side.BUY, price, 3.4, 0.7, ppd, order_id=order_id)
        post, result = book.submit_order(pre, order)
        assert bool(result.accepted) is False
        assert bool(result.traded) is False
        _assert_state_untouched(pre, post)


# ===========================================================================
# (2) SELL ICE with PPD < 0 is rejected.
# ===========================================================================


class TestSellNegativePpdRejected:

    def test_sell_negative_ppd_rejected(self, book):
        """Card case: SELL ICE 5.0 @ 61.13 peak 1.0 PPD -0.10 -> rejected,
        state untouched."""
        pre = book.init()
        order = _ice(Side.SELL, 61.13, 5.0, 1.0, -0.10, order_id=58)
        post, result = book.submit_order(pre, order)
        assert bool(result.accepted) is False
        assert bool(result.traded) is False
        assert float(result.trade_qty) == pytest.approx(0.0)
        _assert_state_untouched(pre, post)

    @pytest.mark.parametrize(
        "ppd,price,order_id",
        [(-0.01, 72.19, 31004), (-1.00, 44.87, 777)],
    )
    def test_sell_negative_ppd_rejected_other_magnitudes(
        self, book, ppd, price, order_id
    ):
        pre = book.init()
        order = _ice(Side.SELL, price, 2.6, 0.9, ppd, order_id=order_id)
        post, result = book.submit_order(pre, order)
        assert bool(result.accepted) is False
        assert bool(result.traded) is False
        _assert_state_untouched(pre, post)

    def test_wrong_sign_rejected_on_non_empty_book(self, book):
        """Rejection does not depend on an empty book: with resting orders on
        both sides (non-crossing), a SELL PPD < 0 ICE still leaves every leaf
        unchanged."""
        state = book.init()
        state, r1 = book.submit_order(state, _order(Side.BUY, 39.41, 1.3, 2207))
        state, r2 = book.submit_order(state, _order(Side.SELL, 66.08, 2.1, 35))
        assert bool(r1.accepted) and bool(r2.accepted)
        pre = state
        post, result = book.submit_order(
            pre, _ice(Side.SELL, 58.29, 4.0, 1.0, -0.10, order_id=81516)
        )
        assert bool(result.accepted) is False
        _assert_state_untouched(pre, post)


# ===========================================================================
# (3) A wrong-sign ICE that would cross makes no trade.
# ===========================================================================


class TestWrongSignThatWouldCrossMakesNoTrade:

    def test_wrong_sign_that_would_cross_makes_no_trade(self, book):
        """Resting SELL NON 2.0 @ 50.00, then BUY ICE 4.0 @ 50.20 peak 1.0
        PPD +0.05: rejected, tape unchanged, the SELL still rests at 2.0."""
        state = book.init()
        sell_id = 9314
        state, r0 = book.submit_order(state, _order(Side.SELL, 50.00, 2.0, sell_id))
        assert bool(r0.accepted) is True
        pre = state
        post, result = book.submit_order(
            pre, _ice(Side.BUY, 50.20, 4.0, 1.0, 0.05, order_id=26)
        )
        assert bool(result.accepted) is False
        assert bool(result.traded) is False
        assert float(result.trade_qty) == pytest.approx(0.0)
        _assert_state_untouched(pre, post)
        resting = book.get_order(post, order_id=sell_id, side=Side.SELL)
        assert int(resting.order_id) == sell_id
        assert float(resting.quantity) == pytest.approx(2.0)

    def test_wrong_sign_sell_that_would_cross_makes_no_trade(self, book):
        """Mirror: resting BUY NON 1.5 @ 47.63, then SELL ICE 3.0 @ 47.11
        peak 0.5 PPD -1.00: rejected, no trade, the BUY still rests."""
        state = book.init()
        buy_id = 650281
        state, r0 = book.submit_order(state, _order(Side.BUY, 47.63, 1.5, buy_id))
        assert bool(r0.accepted) is True
        pre = state
        post, result = book.submit_order(
            pre, _ice(Side.SELL, 47.11, 3.0, 0.5, -1.00, order_id=4402)
        )
        assert bool(result.accepted) is False
        assert bool(result.traded) is False
        _assert_state_untouched(pre, post)
        resting = book.get_order(post, order_id=buy_id, side=Side.BUY)
        assert int(resting.order_id) == buy_id
        assert float(resting.quantity) == pytest.approx(1.5)


# ===========================================================================
# (4) validate_order returns False for (1) and (2), eager and jitted.
# ===========================================================================


_WRONG_SIGN_CASES = [
    pytest.param(Side.BUY, 50.37, 5.0, 1.0, 0.10, 4127, id="buy_+0.10"),
    pytest.param(Side.SELL, 61.13, 5.0, 1.0, -0.10, 58, id="sell_-0.10"),
    pytest.param(Side.BUY, 48.91, 3.4, 0.7, 0.01, 63, id="buy_+0.01"),
    pytest.param(Side.SELL, 44.87, 2.6, 0.9, -1.00, 777, id="sell_-1.00"),
]


class TestValidateOrderWrongSignFalse:

    @pytest.mark.parametrize("side,price,qty,peak,ppd,oid", _WRONG_SIGN_CASES)
    def test_validate_order_wrong_sign_false_eager(
        self, product_config, side, price, qty, peak, ppd, oid
    ):
        order = _ice(side, price, qty, peak, ppd, order_id=oid)
        assert bool(validate_order(order, product_config, True)) is False

    @pytest.mark.parametrize("side,price,qty,peak,ppd,oid", _WRONG_SIGN_CASES)
    def test_validate_order_wrong_sign_false_jitted(
        self, product_config, side, price, qty, peak, ppd, oid
    ):
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        order = _ice(side, price, qty, peak, ppd, order_id=oid)
        assert bool(jitted(order, product_config, True)) is False

    @pytest.mark.parametrize(
        "side,ppd",
        [(Side.BUY, -0.10), (Side.SELL, 0.10), (Side.BUY, 0.0), (Side.SELL, 0.0)],
    )
    def test_validate_order_conforming_sign_true(self, product_config, side, ppd):
        """Guard companion to (4): conforming / zero PPD validates True,
        eager and jitted."""
        order = _ice(side, 53.71, 5.0, 1.0, ppd, order_id=1093)
        jitted = jax.jit(validate_order, static_argnums=(1, 2))
        assert bool(validate_order(order, product_config, True)) is True
        assert bool(jitted(order, product_config, True)) is True


# ===========================================================================
# (5) Guard: conforming signs are accepted and rest their peak.
# ===========================================================================


class TestConformingSignAccepted:

    @pytest.mark.parametrize(
        "ppd,price,oid",
        [(-0.10, 50.37, 4127), (-0.01, 49.83, 12), (-1.00, 38.27, 830156)],
    )
    def test_conforming_buy_negative_ppd_accepted(self, book, ppd, price, oid):
        pre = book.init()
        post, result = book.submit_order(
            pre, _ice(Side.BUY, price, 5.0, 1.0, ppd, order_id=oid)
        )
        assert bool(result.accepted) is True
        assert bool(result.traded) is False
        assert int(post.n_active) == int(pre.n_active) + 1
        assert int(post.ts_counter) > int(pre.ts_counter)
        assert _level_qty(_bids(book, post), price) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "ppd,price,oid",
        [(0.10, 61.13, 58), (0.01, 70.49, 3301), (1.00, 55.57, 60417)],
    )
    def test_conforming_sell_positive_ppd_accepted(self, book, ppd, price, oid):
        pre = book.init()
        post, result = book.submit_order(
            pre, _ice(Side.SELL, price, 5.0, 1.0, ppd, order_id=oid)
        )
        assert bool(result.accepted) is True
        assert bool(result.traded) is False
        assert int(post.n_active) == int(pre.n_active) + 1
        assert int(post.ts_counter) > int(pre.ts_counter)
        assert _level_qty(_asks(book, post), price) == pytest.approx(1.0)


# ===========================================================================
# (6) Guard: plain ICE (PPD = 0) accepted on both sides.
# ===========================================================================


class TestPlainIcePpdZeroAccepted:

    @pytest.mark.parametrize(
        "side,price,oid",
        [(Side.BUY, 46.19, 81), (Side.SELL, 63.47, 290518)],
    )
    def test_plain_ice_ppd_zero_accepted(self, book, side, price, oid):
        pre = book.init()
        post, result = book.submit_order(
            pre, _ice(side, price, 4.2, 0.6, 0.0, order_id=oid)
        )
        assert bool(result.accepted) is True
        assert bool(result.traded) is False
        assert int(post.n_active) == int(pre.n_active) + 1
        levels = _bids(book, post) if side == Side.BUY else _asks(book, post)
        assert _level_qty(levels, price) == pytest.approx(0.6)


# ===========================================================================
# (7) Guard: sub-half-tick PPD rounds to 0 ticks -> plain ICE, accepted.
# ===========================================================================


class TestSubHalfTickPpdIsPlainIce:

    def test_sub_half_tick_ppd_is_plain_ice(self, book, product_config):
        """BUY PPD +0.004 (< half a 0.01 tick) rounds to 0 ticks and is
        accepted as plain ICE (Method-A rounding, as for price)."""
        order = _ice(Side.BUY, 52.03, 5.0, 1.0, 0.004, order_id=7316)
        assert bool(validate_order(order, product_config, True)) is True
        pre = book.init()
        post, result = book.submit_order(pre, order)
        assert bool(result.accepted) is True
        assert bool(result.traded) is False
        assert int(post.n_active) == int(pre.n_active) + 1

    def test_sub_half_tick_negative_ppd_sell_is_plain_ice(self, book):
        """Mirror: SELL PPD -0.004 rounds to 0 ticks and is accepted."""
        pre = book.init()
        post, result = book.submit_order(
            pre, _ice(Side.SELL, 59.87, 5.0, 1.0, -0.004, order_id=45)
        )
        assert bool(result.accepted) is True
        assert int(post.n_active) == int(pre.n_active) + 1


# ===========================================================================
# (8) Guard: the PPD of a non-ICE order is not checked.
# ===========================================================================


class TestNonIcePpdNotChecked:

    @pytest.mark.parametrize("ppd", [0.10, -0.10])
    @pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
    def test_non_ice_ppd_not_checked_non(self, book, product_config, side, ppd):
        order = _order(side, 51.29, 2.3, 6170, ppd=ppd)
        assert bool(validate_order(order, product_config, True)) is True
        _post, result = book.submit_order(book.init(), order)
        assert bool(result.accepted) is True

    @pytest.mark.parametrize("ppd", [0.10, -0.10])
    @pytest.mark.parametrize(
        "restriction", [ExecRestriction.IOC, ExecRestriction.FOK]
    )
    def test_non_ice_ppd_not_checked_ioc_fok(
        self, book, product_config, restriction, ppd
    ):
        """BUY IOC / FOK with PPD +-0.10 against a resting SELL fully fill:
        accepted and traded (the PPD is ignored)."""
        state = book.init()
        state, r0 = book.submit_order(state, _order(Side.SELL, 50.43, 3.0, 88))
        assert bool(r0.accepted) is True
        order = _order(
            Side.BUY, 50.43, 1.7, 41926, exec_restriction=restriction, ppd=ppd,
        )
        assert bool(validate_order(order, product_config, True)) is True
        _post, result = book.submit_order(state, order)
        assert bool(result.accepted) is True
        assert bool(result.traded) is True
        assert float(result.trade_qty) == pytest.approx(1.7)

    @pytest.mark.parametrize("ppd", [0.10, -0.10])
    @pytest.mark.parametrize(
        "restriction", [ExecRestriction.IOC, ExecRestriction.FOK]
    )
    def test_non_ice_ppd_not_checked_sell_ioc_fok(
        self, book, restriction, ppd
    ):
        state = book.init()
        state, r0 = book.submit_order(state, _order(Side.BUY, 49.67, 3.0, 5502))
        assert bool(r0.accepted) is True
        order = _order(
            Side.SELL, 49.67, 1.1, 19, exec_restriction=restriction, ppd=ppd,
        )
        _post, result = book.submit_order(state, order)
        assert bool(result.accepted) is True
        assert bool(result.traded) is True


# ===========================================================================
# (9) One-compile invariant over a mixed session.
# ===========================================================================


class TestOneCompilePreserved:

    def test_one_compile_preserved(self, book):
        """A session mixing wrong-sign (1)(2), conforming (5), plain ICE (6)
        and non-ICE-with-PPD (8) orders compiles ``_submit_order`` once.

        The cache is cleared first so entries left by other test modules in
        the same worker (different static configs) do not pollute the count.
        """
        if not hasattr(_submit_order, "_cache_size"):
            pytest.skip("_submit_order does not expose `_cache_size`")
        if hasattr(_submit_order, "clear_cache"):
            _submit_order.clear_cache()
        else:  # pragma: no cover - older jax
            _submit_order._clear_cache()

        orders = [
            _ice(Side.BUY, 50.37, 5.0, 1.0, 0.10, order_id=4127),        # (1)
            _ice(Side.SELL, 61.13, 5.0, 1.0, -0.10, order_id=58),        # (2)
            _ice(Side.BUY, 49.83, 5.0, 1.0, -0.10, order_id=12),         # (5)
            _ice(Side.SELL, 70.49, 5.0, 1.0, 0.10, order_id=3301),       # (5)
            _ice(Side.BUY, 46.19, 4.2, 0.6, 0.0, order_id=81),           # (6)
            _ice(Side.SELL, 63.47, 4.2, 0.6, 0.0, order_id=290518),      # (6)
            _order(Side.BUY, 51.29, 2.3, 6170, ppd=0.10),                # (8) NON
            _order(Side.SELL, 51.29, 1.0, 617, exec_restriction=ExecRestriction.IOC,
                   ppd=-0.10),                                           # (8) IOC
            _order(Side.SELL, 51.29, 0.5, 61708, exec_restriction=ExecRestriction.FOK,
                   ppd=0.10),                                            # (8) FOK
        ]
        state = book.init()
        for o in orders:
            state, _ = book.submit_order(state, o)
        for o in orders[:2]:  # wrong-sign also on a fresh book
            book.submit_order(book.init(), o)

        size = _submit_order._cache_size()
        assert size == 1, (
            "The PPD sign predicate must be traced (no new static argument, "
            f"no Python branch): expected one compile, got {size}."
        )
