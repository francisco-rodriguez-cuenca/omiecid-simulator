"""Tests for M3.3 -- IOC/FOK Execution Restrictions.

Behavioral contracts for IOC (Immediate or Cancel) and FOK (Fill or Kill)
handling in ContractBook.submit_order().

All expected values are derived from:
  - docs/omiecid/market-spec.md SS3.2 (execution restrictions)
  - docs/omiecid/market-spec.md SS4.5 (restriction summary table)
  - docs/omiecid/market-spec.md SS5.1 (order lifecycle: IOC/FOK -> CANCELLED)
  - docs/omiecid/market-spec.md SS6.2 (regular matching, steps 4-5)
  - docs/omiecid/migration-plan.md SS M3 (IOC/FOK tests, interface)
  - docs/omiecid/agile-plan.md M3.3 (acceptance criteria)

Architecture decisions (from M3.3 architecture review):
  - IOC: submit as normal limit order (jaxob fills what it can), then
    immediately cancel remainder via existing cancel_order
  - FOK: pre-check matchable volume on opposite side; reject early if
    insufficient, else submit as normal limit order
  - No SubmitResult changes -- existing 5 fields cover all IOC/FOK scenarios
  - Only contract_book.py needs modification

Domain rules (market-spec SS4.5):
  - IOC: partial fill YES, book entry if unmatched NO (cancelled)
  - FOK: partial fill NO (full or cancel), book entry if unmatched NO (cancelled)

These tests are black-box: they import only the public interface and
assert against domain-specified values. No implementation assumptions.

Three acceptance-criteria categories (expanded per-field):
  1. test_ioc_partial_fill_cancels_remainder -- IOC fills available, cancels rest
  2. test_fok_full_fill                      -- FOK fills entirely when liquidity exists
  3. test_fok_insufficient_liquidity_cancelled -- FOK cancelled when insufficient liquidity

Additional edge-case categories:
  4. IOC full fill (all liquidity consumed)
  5. IOC no liquidity (empty book / no crossing)
  6. FOK boundary conditions (exact match, multiple levels)
  7. IOC/FOK interaction with NON resting orders
  8. IOC/FOK SubmitResult field verification
  9. IOC/FOK settlement value correctness
"""

import pytest

from omiecid_simulator.omiecid.order_types import (
    OMIEOrder,
    Side,
    ExecRestriction,
    Validity,
    Status,
    make_order,
)
from omiecid_simulator.omiecid.contract_book import SubmitResult

# --- P2-H2 Category-B helper: get_order returns sentinel OMIEOrder, not None.
from omiecid_simulator.omiecid.jax_constants import (
    SENTINEL_ORDER_ID as _P2H2_SENTINEL_ORDER_ID,  # noqa: E501
)


def _is_absent(order):
    """Return True iff ``order`` is Phase-1 ``None`` OR the post-P2-H2 sentinel.

    The post-P2-H2 ``get_order`` returns a sentinel ``OMIEOrder`` whose
    ``order_id == SENTINEL_ORDER_ID`` for absent ids. This helper bridges
    both representations so callsites that pre-date P2-H2 keep working
    without per-site mechanical rewrites.
    """
    if order is None:
        return True
    try:
        return int(order.order_id) == _P2H2_SENTINEL_ORDER_ID
    except (AttributeError, TypeError):
        return False




# ---------------------------------------------------------------------------
# Helpers: make buy/sell orders with custom fields and exec restrictions
# ---------------------------------------------------------------------------

def _make_buy(order_id=1, price=50.00, quantity=10.0, agent_id=0, timestamp=100,
              exec_restriction=ExecRestriction.NON):
    return make_order(
        contract_id=37, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction,
    )


def _make_sell(order_id=2, price=50.00, quantity=10.0, agent_id=1, timestamp=200,
               exec_restriction=ExecRestriction.NON):
    return make_order(
        contract_id=37, side=Side.SELL, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction,
    )


def _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, agent_id=0, timestamp=500):
    return _make_buy(
        order_id=order_id, price=price, quantity=quantity,
        agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.IOC,
    )


def _make_ioc_sell(order_id=11, price=50.00, quantity=10.0, agent_id=1, timestamp=500):
    return _make_sell(
        order_id=order_id, price=price, quantity=quantity,
        agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.IOC,
    )


def _make_fok_buy(order_id=20, price=50.00, quantity=10.0, agent_id=0, timestamp=500):
    return _make_buy(
        order_id=order_id, price=price, quantity=quantity,
        agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.FOK,
    )


def _make_fok_sell(order_id=21, price=50.00, quantity=10.0, agent_id=1, timestamp=500):
    return _make_sell(
        order_id=order_id, price=price, quantity=quantity,
        agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.FOK,
    )


# ===========================================================================
# 1. TestIOCPartialFillCancelsRemainder
# ===========================================================================

class TestIOCPartialFillCancelsRemainder:
    """IOC order fills what is available, remainder is cancelled (not in book).

    Domain rule (market-spec SS4.5): IOC allows partial fill: YES;
    book entry if unmatched: NO (cancelled).
    Domain rule (market-spec SS6.2 step 5): "If incoming order is partially
    filled and has IOC restriction: residual is cancelled."
    """

    def test_ioc_buy_partial_fill_traded_is_true(self, book, empty_state):
        """Resting sell 5 MW at 50, IOC buy 10 MW at 50 -> traded=True."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True

    def test_ioc_buy_partial_fill_trade_qty_is_available(self, book, empty_state):
        """Resting sell 5 MW, IOC buy 10 MW -> trade_qty = 5.0 (available)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(5.0, abs=0.05)

    def test_ioc_buy_remainder_not_in_book(self, book, empty_state):
        """After IOC partial fill, the unfilled remainder must NOT be in book."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        # IOC remainder (5 MW) must not rest in the book
        assert _is_absent(book.get_order(state, 10, Side.BUY))

    def test_ioc_sell_partial_fill_traded_is_true(self, book, empty_state):
        """Resting buy 5 MW at 50, IOC sell 10 MW at 50 -> traded=True."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_sell(order_id=11, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True

    def test_ioc_sell_partial_fill_trade_qty_is_available(self, book, empty_state):
        """Resting buy 5 MW, IOC sell 10 MW -> trade_qty = 5.0."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_sell(order_id=11, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(5.0, abs=0.05)

    def test_ioc_sell_remainder_not_in_book(self, book, empty_state):
        """After IOC sell partial fill, unfilled remainder not in book."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_ioc_sell(order_id=11, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 11, Side.SELL))

    def test_ioc_partial_fill_resting_order_consumed(self, book, empty_state):
        """The resting NON order is fully consumed by the IOC's partial fill."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        # Resting sell fully consumed
        assert _is_absent(book.get_order(state, 2, Side.SELL))

    def test_ioc_partial_fill_across_two_levels(self, book, empty_state):
        """IOC buy walks through two price levels, cancels remainder.

        Resting: sell 3 MW at 48, sell 4 MW at 50.
        IOC buy 10 MW at 50 -> fills 3+4=7, remainder 3 MW cancelled.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=48.00, quantity=3.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=3, price=50.00, quantity=4.0, timestamp=200)
        )
        state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True
        assert result.trade_qty == pytest.approx(7.0, abs=0.05)
        # Remainder not in book
        assert _is_absent(book.get_order(state, 10, Side.BUY))

    def test_ioc_partial_fill_accepted_is_true(self, book, empty_state):
        """IOC that partially fills is still accepted (passes validation)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.accepted == True

    def test_ioc_partial_fill_preserves_other_resting_orders(self, book, empty_state):
        """IOC partial fill does not affect unrelated orders on the same side.

        Resting: sell 5 MW at 50, sell 5 MW at 60.
        IOC buy 5 MW at 50 -> fills sell at 50, sell at 60 untouched.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=3, price=60.00, quantity=5.0, timestamp=200)
        )
        state, _ = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=5.0, timestamp=500)
        )
        # Sell at 60 is untouched (outside IOC's price range)
        other = book.get_order(state, 3, Side.SELL)
        assert not _is_absent(other)
        assert other.quantity == pytest.approx(5.0, abs=0.05)

    def test_ioc_minimum_qty_partial_fill(self, book, empty_state):
        """IOC at minimum quantity partial fill: 0.1 MW available of 0.2 MW IOC."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=0.1, timestamp=100)
        )
        state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=0.2, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(0.1, abs=0.05)
        assert _is_absent(book.get_order(state, 10, Side.BUY))


# ===========================================================================
# 2. TestFOKFullFill
# ===========================================================================

class TestFOKFullFill:
    """FOK order fills entirely when sufficient liquidity exists.

    Domain rule (market-spec SS4.5): FOK allows partial fill: NO (full or cancel).
    When full quantity is available at crossing prices, the order fills completely.
    """

    def test_fok_buy_full_fill_traded_is_true(self, book, empty_state):
        """Resting sell 10 MW at 50, FOK buy 10 MW at 50 -> traded=True."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True

    def test_fok_buy_full_fill_trade_qty(self, book, empty_state):
        """FOK buy 10 MW against 10 MW resting -> trade_qty = 10.0."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(10.0, abs=0.05)

    def test_fok_buy_full_fill_not_in_book(self, book, empty_state):
        """After FOK full fill, order is not in the book (fully consumed)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 20, Side.BUY))

    def test_fok_buy_full_fill_resting_consumed(self, book, empty_state):
        """After FOK full fill, the resting order is fully consumed."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 2, Side.SELL))

    def test_fok_sell_full_fill_traded_is_true(self, book, empty_state):
        """Resting buy 10 MW at 50, FOK sell 10 MW at 50 -> traded=True."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_sell(order_id=21, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True

    def test_fok_sell_full_fill_trade_qty(self, book, empty_state):
        """FOK sell 10 MW against 10 MW resting buy -> trade_qty = 10.0."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_sell(order_id=21, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(10.0, abs=0.05)

    def test_fok_sell_full_fill_not_in_book(self, book, empty_state):
        """After FOK sell full fill, order is not in the book."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_fok_sell(order_id=21, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 21, Side.SELL))

    def test_fok_buy_accepted_is_true(self, book, empty_state):
        """FOK that fills fully is accepted (passes validation)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.accepted == True

    def test_fok_buy_across_two_price_levels(self, book, empty_state):
        """FOK buy fills across two price levels when total liquidity suffices.

        Resting: sell 5 MW at 48, sell 5 MW at 50.
        FOK buy 10 MW at 50 -> fills 5+5=10, fully filled.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=48.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=3, price=50.00, quantity=5.0, timestamp=200)
        )
        state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=0.05)
        # Both resting sells consumed
        assert _is_absent(book.get_order(state, 2, Side.SELL))
        assert _is_absent(book.get_order(state, 3, Side.SELL))

    def test_fok_sell_across_two_price_levels(self, book, empty_state):
        """FOK sell fills across two price levels when total liquidity suffices.

        Resting: buy 5 MW at 52, buy 5 MW at 50.
        FOK sell 10 MW at 50 -> fills 5+5=10.
        """
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=52.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=3, price=50.00, quantity=5.0, timestamp=200)
        )
        state, result = book.submit_order(
            state, _make_fok_sell(order_id=21, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=0.05)

    def test_fok_buy_excess_liquidity_leaves_residual_resting(self, book, empty_state):
        """FOK buy 5 MW against resting sell 10 MW -> fills 5, resting has 5 left.

        FOK only requires full fill of the FOK order, not of the resting order.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=5.0, timestamp=500)
        )
        assert result.traded == True
        assert result.trade_qty == pytest.approx(5.0, abs=0.05)
        # Resting sell has 5 MW residual
        residual = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(residual)
        assert residual.quantity == pytest.approx(5.0, abs=0.05)


# ===========================================================================
# 3. TestFOKInsufficientLiquidityCancelled
# ===========================================================================

class TestFOKInsufficientLiquidityCancelled:
    """FOK order is cancelled entirely when insufficient liquidity.

    Domain rule (market-spec SS4.5): FOK partial fill: NO (full or cancel).
    If the full quantity cannot be matched, the order is cancelled entirely.
    No partial execution, no book entry.
    """

    def test_fok_buy_insufficient_qty_traded_is_false(self, book, empty_state):
        """Resting sell 5 MW, FOK buy 10 MW -> insufficient, traded=False."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False

    def test_fok_buy_insufficient_qty_trade_qty_zero(self, book, empty_state):
        """FOK rejected -> trade_qty = 0.0 (no partial fill allowed)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(0.0, abs=0.05)

    def test_fok_buy_insufficient_not_in_book(self, book, empty_state):
        """FOK rejected -> order must NOT enter the book."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 20, Side.BUY))

    def test_fok_buy_insufficient_resting_order_untouched(self, book, empty_state):
        """FOK cancelled -> resting orders must NOT be consumed.

        This is critical: FOK is atomic. If it can't fully fill, nothing happens.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        # Resting sell must be completely untouched
        resting = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(resting)
        assert resting.quantity == pytest.approx(5.0, abs=0.05)

    def test_fok_sell_insufficient_qty_traded_is_false(self, book, empty_state):
        """Resting buy 5 MW, FOK sell 10 MW -> insufficient, traded=False."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_sell(order_id=21, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False

    def test_fok_sell_insufficient_not_in_book(self, book, empty_state):
        """FOK sell rejected -> order must NOT enter the book."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_fok_sell(order_id=21, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 21, Side.SELL))

    def test_fok_sell_insufficient_resting_untouched(self, book, empty_state):
        """FOK sell cancelled -> resting buy untouched (atomic semantics)."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_fok_sell(order_id=21, price=50.00, quantity=10.0, timestamp=500)
        )
        resting = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(resting)
        assert resting.quantity == pytest.approx(5.0, abs=0.05)

    def test_fok_into_empty_book_traded_is_false(self, book, empty_state):
        """FOK buy into empty book -> no liquidity, traded=False."""
        _state, result = book.submit_order(
            empty_state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False

    def test_fok_into_empty_book_not_in_book(self, book, empty_state):
        """FOK buy into empty book -> order not in book."""
        state, _ = book.submit_order(
            empty_state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 20, Side.BUY))

    def test_fok_non_crossing_price_cancelled(self, book, empty_state):
        """FOK buy at 45 vs resting sell at 50 -> no crossing, cancelled.

        Even though there is resting liquidity, prices don't cross.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=45.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False

    def test_fok_insufficient_across_multiple_levels(self, book, empty_state):
        """FOK buy 10 MW but only 7 MW available across two levels -> cancelled.

        Resting: sell 3 MW at 48, sell 4 MW at 50.
        FOK buy 10 MW at 50 -> total available 7 < 10, cancelled entirely.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=48.00, quantity=3.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=3, price=50.00, quantity=4.0, timestamp=200)
        )
        state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False
        assert result.trade_qty == pytest.approx(0.0, abs=0.05)
        # Both resting orders untouched
        assert not _is_absent(book.get_order(state, 2, Side.SELL))
        assert not _is_absent(book.get_order(state, 3, Side.SELL))

    def test_fok_sell_insufficient_across_multiple_levels(self, book, empty_state):
        """FOK sell 10 MW but only 8 MW resting buy available -> cancelled.

        Resting: buy 5 MW at 52, buy 3 MW at 50.
        FOK sell 10 MW at 50 -> total available 8 < 10, cancelled.
        """
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=52.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=3, price=50.00, quantity=3.0, timestamp=200)
        )
        state, result = book.submit_order(
            state, _make_fok_sell(order_id=21, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False
        assert result.trade_qty == pytest.approx(0.0, abs=0.05)
        # Both resting buys untouched
        resting1 = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(resting1)
        assert resting1.quantity == pytest.approx(5.0, abs=0.05)
        resting3 = book.get_order(state, 3, Side.BUY)
        assert not _is_absent(resting3)
        assert resting3.quantity == pytest.approx(3.0, abs=0.05)


# ===========================================================================
# 4. TestIOCFullFill
# ===========================================================================

class TestIOCFullFill:
    """IOC order that finds sufficient liquidity fills entirely.

    IOC allows partial fills (SS4.5), but when full liquidity exists the
    IOC fills completely -- identical outcome to a NON order in that case.
    """

    def test_ioc_buy_full_fill_traded_is_true(self, book, empty_state):
        """Resting sell 10 MW, IOC buy 10 MW -> traded=True."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True

    def test_ioc_buy_full_fill_trade_qty(self, book, empty_state):
        """IOC buy 10 MW against 10 MW resting -> trade_qty = 10.0."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(10.0, abs=0.05)

    def test_ioc_buy_full_fill_not_in_book(self, book, empty_state):
        """After IOC full fill, order not in book (fully consumed)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 10, Side.BUY))

    def test_ioc_sell_full_fill_traded_is_true(self, book, empty_state):
        """Resting buy 10 MW, IOC sell 10 MW -> traded=True."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_sell(order_id=11, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True

    def test_ioc_full_fill_across_two_levels(self, book, empty_state):
        """IOC buy fills entirely across two price levels.

        Resting: sell 5 MW at 48, sell 5 MW at 50.
        IOC buy 10 MW at 50 -> fills 5+5=10, fully consumed.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=48.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=3, price=50.00, quantity=5.0, timestamp=200)
        )
        state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=0.05)
        assert _is_absent(book.get_order(state, 10, Side.BUY))


# ===========================================================================
# 5. TestIOCNoLiquidity
# ===========================================================================

class TestIOCNoLiquidity:
    """IOC order into empty book or non-crossing book is cancelled entirely.

    Domain rule: IOC with no available liquidity -> cancelled immediately.
    No book entry (SS4.5: book entry if unmatched = NO).
    """

    def test_ioc_buy_empty_book_traded_is_false(self, book, empty_state):
        """IOC buy into empty book -> no liquidity, traded=False."""
        _state, result = book.submit_order(
            empty_state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False

    def test_ioc_buy_empty_book_trade_qty_zero(self, book, empty_state):
        """IOC buy into empty book -> trade_qty = 0."""
        _state, result = book.submit_order(
            empty_state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(0.0, abs=0.05)

    def test_ioc_buy_empty_book_not_in_book(self, book, empty_state):
        """IOC buy into empty book -> order not in book."""
        state, _ = book.submit_order(
            empty_state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 10, Side.BUY))

    def test_ioc_sell_empty_book_traded_is_false(self, book, empty_state):
        """IOC sell into empty book -> traded=False."""
        _state, result = book.submit_order(
            empty_state, _make_ioc_sell(order_id=11, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False

    def test_ioc_sell_empty_book_not_in_book(self, book, empty_state):
        """IOC sell into empty book -> order not in book."""
        state, _ = book.submit_order(
            empty_state, _make_ioc_sell(order_id=11, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 11, Side.SELL))

    def test_ioc_non_crossing_price_traded_is_false(self, book, empty_state):
        """IOC buy at 45 vs resting sell at 50 -> no crossing, traded=False."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=45.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False

    def test_ioc_non_crossing_not_in_book(self, book, empty_state):
        """IOC buy at non-crossing price -> order not in book (not resting)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=45.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 10, Side.BUY))

    def test_ioc_accepted_even_when_no_fill(self, book, empty_state):
        """IOC into empty book is still accepted (passes validation).

        accepted=True means the order passed validation. traded=False means
        it found no liquidity and was cancelled per IOC rules.
        """
        _state, result = book.submit_order(
            empty_state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.accepted == True


# ===========================================================================
# 6. TestFOKBoundaryConditions
# ===========================================================================

class TestFOKBoundaryConditions:
    """FOK edge cases: exact quantity match, minimum quantity, negative prices."""

    def test_fok_exact_quantity_match_fills(self, book, empty_state):
        """FOK buy 10 MW against exactly 10 MW resting -> fills."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=0.05)

    def test_fok_one_tick_short_of_full_cancelled(self, book, empty_state):
        """FOK buy 10.0 MW vs resting sell 9.9 MW -> 0.1 MW short, cancelled.

        The FOK quantity cannot be rounded or truncated. Full match required.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=9.9, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False

    def test_fok_minimum_quantity_fills(self, book, empty_state):
        """FOK at minimum qty 0.1 MW against 0.1 MW resting -> fills."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=0.1, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=0.1, timestamp=500)
        )
        assert result.traded == True
        assert result.trade_qty == pytest.approx(0.1, abs=0.05)

    def test_fok_with_negative_price_fills(self, book, empty_state):
        """FOK buy at -10.00 vs resting sell at -10.00 -> fills.

        Negative prices are valid in OMIE (SS2.3).
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=-10.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=-10.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True

    def test_fok_aggressive_price_fills(self, book, empty_state):
        """FOK buy at 55 vs resting sell at 50 -> crosses, fills at 50."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=55.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True
        assert result.trade_price == pytest.approx(50.00, abs=0.005)

    def test_fok_partial_level_plus_full_level(self, book, empty_state):
        """FOK buy 8 MW: resting sell 5 MW at 48 + 10 MW at 50 -> fills 5+3.

        Total available across crossing levels >= FOK qty, fills from both.
        Resting sell at 50 should have 7 MW residual.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=48.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=3, price=50.00, quantity=10.0, timestamp=200)
        )
        state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=8.0, timestamp=500)
        )
        assert result.traded == True
        assert result.trade_qty == pytest.approx(8.0, abs=0.05)
        # First sell fully consumed
        assert _is_absent(book.get_order(state, 2, Side.SELL))
        # Second sell partially consumed: 10 - 3 = 7 MW residual
        residual = book.get_order(state, 3, Side.SELL)
        assert not _is_absent(residual)
        assert residual.quantity == pytest.approx(7.0, abs=0.05)


# ===========================================================================
# 7. TestIOCFOKInteractionWithNON
# ===========================================================================

class TestIOCFOKInteractionWithNON:
    """IOC/FOK orders interact correctly with resting NON orders.

    All resting orders in these tests are NON (the standard case).
    Verifies that IOC/FOK behavior is correct against the normal book.
    """

    def test_ioc_does_not_affect_non_resting_orders_on_same_side(self, book, empty_state):
        """IOC buy should not disturb existing NON buy orders.

        Existing NON buy at 45, IOC buy at 50 -> IOC cancelled (no crossing
        sell), existing buy at 45 untouched.
        """
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        # IOC not in book
        assert _is_absent(book.get_order(state, 10, Side.BUY))
        # Existing buy untouched
        existing = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(existing)
        assert existing.quantity == pytest.approx(5.0, abs=0.05)

    def test_fok_does_not_affect_non_resting_orders_on_same_side(self, book, empty_state):
        """FOK buy does not disturb existing NON buy orders."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert _is_absent(book.get_order(state, 20, Side.BUY))
        existing = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(existing)
        assert existing.quantity == pytest.approx(5.0, abs=0.05)

    def test_ioc_partial_fill_leaves_deeper_non_orders(self, book, empty_state):
        """IOC buy fills best ask, leaves non-crossing asks untouched.

        Resting: sell 5 MW at 50, sell 5 MW at 55.
        IOC buy 10 MW at 50 -> fills 5 at 50, but 55 doesn't cross,
        so remainder 5 MW cancelled. Sell at 55 untouched.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=3, price=55.00, quantity=5.0, timestamp=200)
        )
        state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(5.0, abs=0.05)
        # Sell at 55 (non-crossing) untouched
        deeper = book.get_order(state, 3, Side.SELL)
        assert not _is_absent(deeper)
        assert deeper.quantity == pytest.approx(5.0, abs=0.05)

    def test_fok_rejected_leaves_all_non_orders_untouched(self, book, empty_state):
        """FOK rejected: all resting NON orders (both sides) untouched.

        Resting: buy 3 MW at 45, sell 5 MW at 50.
        FOK buy 10 MW at 50 -> insufficient (only 5 MW), cancelled.
        Both resting orders must be untouched.
        """
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, quantity=3.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=200)
        )
        state, _ = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        buy = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(buy)
        assert buy.quantity == pytest.approx(3.0, abs=0.05)
        sell = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(sell)
        assert sell.quantity == pytest.approx(5.0, abs=0.05)

    def test_non_order_after_ioc_enters_book_normally(self, book, empty_state):
        """After an IOC is cancelled, a subsequent NON order enters book normally.

        IOC buy into empty book (cancelled), then NON buy at 50 -> enters book.
        """
        state, _ = book.submit_order(
            empty_state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=600)
        )
        result = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(result)
        assert result.quantity == pytest.approx(10.0, abs=0.05)

    def test_non_order_after_fok_rejected_enters_book_normally(self, book, empty_state):
        """After a FOK is cancelled, a subsequent NON order enters book normally."""
        state, _ = book.submit_order(
            empty_state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=600)
        )
        result = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(result)
        assert result.quantity == pytest.approx(10.0, abs=0.05)


# ===========================================================================
# 8. TestIOCFOKSubmitResultFields
# ===========================================================================

class TestIOCFOKSubmitResultFields:
    """Verify all 5 SubmitResult fields are correctly populated for IOC/FOK.

    SubmitResult: accepted (bool), traded (bool), trade_qty (float, MW),
    trade_price (float, EUR/MWh), settlement_value (float, EUR = Q x p x 0.25).
    """

    # --- IOC partial fill result fields ---

    def test_ioc_partial_fill_result_accepted(self, book, empty_state):
        """IOC partial fill -> accepted=True."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.accepted == True

    def test_ioc_partial_fill_result_traded(self, book, empty_state):
        """IOC partial fill -> traded=True."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True

    def test_ioc_partial_fill_result_trade_price(self, book, empty_state):
        """IOC partial fill at resting price -> trade_price = resting price.

        Resting sell at 50, IOC buy at 55 -> trade at resting price 50.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=55.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_price == pytest.approx(50.00, abs=0.005)

    def test_ioc_partial_fill_result_settlement_value(self, book, empty_state):
        """IOC partial fill settlement: V = Q x p x 0.25.

        5 MW at 50 EUR/MWh -> V = 5 x 50 x 0.25 = 62.50 EUR.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(62.50, abs=0.05)

    # --- IOC no fill result fields ---

    def test_ioc_no_fill_result_accepted(self, book, empty_state):
        """IOC into empty book -> accepted=True (passes validation)."""
        _state, result = book.submit_order(
            empty_state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.accepted == True

    def test_ioc_no_fill_result_traded_false(self, book, empty_state):
        """IOC into empty book -> traded=False."""
        _state, result = book.submit_order(
            empty_state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False

    def test_ioc_no_fill_result_trade_qty_zero(self, book, empty_state):
        """IOC into empty book -> trade_qty=0.0."""
        _state, result = book.submit_order(
            empty_state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(0.0, abs=0.05)

    def test_ioc_no_fill_result_settlement_zero(self, book, empty_state):
        """IOC no fill -> settlement_value=0.0."""
        _state, result = book.submit_order(
            empty_state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(0.0, abs=0.05)

    # --- FOK full fill result fields ---

    def test_fok_full_fill_result_accepted(self, book, empty_state):
        """FOK full fill -> accepted=True."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.accepted == True

    def test_fok_full_fill_result_traded(self, book, empty_state):
        """FOK full fill -> traded=True."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == True

    def test_fok_full_fill_result_trade_price(self, book, empty_state):
        """FOK buy at 55 fills at resting sell price 50 -> trade_price = 50."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=55.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_price == pytest.approx(50.00, abs=0.005)

    def test_fok_full_fill_result_settlement_value(self, book, empty_state):
        """FOK full fill settlement: V = Q x p x 0.25.

        10 MW at 50 EUR/MWh -> V = 10 x 50 x 0.25 = 125.00 EUR.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(125.00, abs=0.05)

    # --- FOK rejected result fields ---

    def test_fok_rejected_result_accepted(self, book, empty_state):
        """FOK rejected (insufficient liquidity) -> accepted=True.

        accepted=True means the order passed validation. traded=False means
        it failed the FOK liquidity check.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.accepted == True

    def test_fok_rejected_result_traded_false(self, book, empty_state):
        """FOK rejected -> traded=False."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.traded == False

    def test_fok_rejected_result_trade_qty_zero(self, book, empty_state):
        """FOK rejected -> trade_qty=0.0."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_qty == pytest.approx(0.0, abs=0.05)

    def test_fok_rejected_result_trade_price_zero(self, book, empty_state):
        """FOK rejected -> trade_price=0.0."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.trade_price == pytest.approx(0.0, abs=0.005)

    def test_fok_rejected_result_settlement_zero(self, book, empty_state):
        """FOK rejected -> settlement_value=0.0."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(0.0, abs=0.05)


# ===========================================================================
# 9. TestIOCFOKSettlementValue
# ===========================================================================

class TestIOCFOKSettlementValue:
    """Settlement value V = Q x p x 0.25 for IOC/FOK trades.

    Same formula as NON orders (market-spec SS2.3). Verifies correctness
    across different IOC/FOK fill scenarios.
    """

    def test_ioc_full_fill_settlement(self, book, empty_state):
        """IOC full fill: 10 MW at 50 -> V = 10 x 50 x 0.25 = 125.00."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(125.00, abs=0.05)

    def test_ioc_partial_fill_settlement_uses_filled_qty(self, book, empty_state):
        """IOC partial fill: only 5 MW filled at 50 -> V = 5 x 50 x 0.25 = 62.50.

        Not 10 x 50 x 0.25 (the requested qty).
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(62.50, abs=0.05)

    def test_fok_full_fill_settlement(self, book, empty_state):
        """FOK full fill: 10 MW at 50 -> V = 10 x 50 x 0.25 = 125.00."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(125.00, abs=0.05)

    def test_fok_negative_price_settlement(self, book, empty_state):
        """FOK at negative price: 10 MW at -20.00 -> V = 10 x (-20) x 0.25 = -50.00.

        Negative settlement value is valid (buyer receives payment).
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=-20.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=-20.00, quantity=10.0, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(-50.00, abs=0.05)

    def test_ioc_multi_level_settlement_uses_vwap(self, book, empty_state):
        """IOC across two levels: VWAP price used in settlement.

        Resting: sell 5 MW at 48, sell 5 MW at 50.
        IOC buy 10 MW at 50 -> fills 5 at 48 + 5 at 50 = 10 MW total.
        VWAP = (5*48 + 5*50) / 10 = 49.00.
        V = 10 x 49.00 x 0.25 = 122.50.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=48.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=3, price=50.00, quantity=5.0, timestamp=200)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(122.50, abs=0.05)

    def test_fok_multi_level_settlement_uses_vwap(self, book, empty_state):
        """FOK across two levels: VWAP in settlement.

        Resting: sell 5 MW at 48, sell 5 MW at 50.
        FOK buy 10 MW at 50 -> fills entirely. VWAP = 49.00.
        V = 10 x 49.00 x 0.25 = 122.50.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=48.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=3, price=50.00, quantity=5.0, timestamp=200)
        )
        _state, result = book.submit_order(
            state, _make_fok_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(122.50, abs=0.05)

    def test_ioc_minimum_qty_settlement(self, book, empty_state):
        """IOC minimum qty: 0.1 MW at 50 -> V = 0.1 x 50 x 0.25 = 1.25."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=0.1, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_ioc_buy(order_id=10, price=50.00, quantity=0.1, timestamp=500)
        )
        assert result.settlement_value == pytest.approx(1.25, abs=0.05)
