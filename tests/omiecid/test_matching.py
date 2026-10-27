"""Tests for M3.2 -- Price-Time Priority Matching.

Behavioral contracts for matching logic in ContractBook.submit_order().
All expected values are derived from:
  - docs/omiecid/market-spec.md SS2.3 (trade value formula V = Q x p x 0.25)
  - docs/omiecid/market-spec.md SS6.1 (price-time priority)
  - docs/omiecid/market-spec.md SS6.2 (regular matching, resting price)
  - docs/omiecid/migration-plan.md SS M3 (matching tests, interface)
  - docs/omiecid/agile-plan.md M3.2 (acceptance criteria)

Architecture decisions (approved by user):
  - SubmitResult expanded: accepted, traded, trade_qty, trade_price, settlement_value
  - accepted remains at index 0 (backward compatible with M3.1)
  - Settlement inline in submit_order: V = Q x p x 0.25 (hardcoded 0.25h)
  - Trade price/qty in OMIE domain units (EUR/MWh, MW)

These tests are black-box: they import only the public interface and
assert against domain-specified values. No implementation assumptions.

Five acceptance-criteria categories (expanded per-field):
  1. test_matching_buy_meets_sell        -- crossing orders produce a trade
  2. test_matching_price_is_resting_order -- trade at resting (maker) price
  3. test_partial_fill_non_restriction    -- NON partial fill, residual in book
  4. test_price_time_priority             -- best price first, then earliest ts
  5. test_trade_settlement_value          -- V = Q x p x 0.25
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
# Helpers: make buy/sell orders with custom fields
# ---------------------------------------------------------------------------

def _make_buy(order_id=1, price=50.00, quantity=10.0, agent_id=0, timestamp=100):
    return make_order(
        contract_id=37, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
    )


def _make_sell(order_id=2, price=50.00, quantity=10.0, agent_id=1, timestamp=200):
    return make_order(
        contract_id=37, side=Side.SELL, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
    )


# ===========================================================================
# 1. TestMatchingBuyMeetsSell
# ===========================================================================

class TestMatchingBuyMeetsSell:
    """When a new order crosses the spread, a trade must execute.

    Domain rule (market-spec SS6.2): incoming order attempts to match
    against best counterpart. Match occurs when sell_price <= buy_price.
    """

    def test_sell_into_resting_buy_produces_trade(self, book, empty_state):
        """Resting buy at 50, incoming sell at 50 -> traded=True."""
        state, _ = book.submit_order(empty_state, _make_buy(price=50.00, timestamp=100))
        _state, result = book.submit_order(state, _make_sell(price=50.00, timestamp=200))
        assert result.traded == True

    def test_buy_into_resting_sell_produces_trade(self, book, empty_state):
        """Resting sell at 50, incoming buy at 50 -> traded=True."""
        state, _ = book.submit_order(empty_state, _make_sell(price=50.00, timestamp=100))
        _state, result = book.submit_order(state, _make_buy(price=50.00, timestamp=200))
        assert result.traded == True

    def test_crossing_orders_trade_qty_equals_matched_amount(self, book, empty_state):
        """Buy 10 MW at 50, sell 10 MW at 50 -> trade_qty = 10.0 MW."""
        state, _ = book.submit_order(empty_state, _make_buy(price=50.00, quantity=10.0))
        _state, result = book.submit_order(state, _make_sell(price=50.00, quantity=10.0))
        assert result.trade_qty == pytest.approx(10.0, abs=0.05)

    def test_crossing_orders_accepted_is_true(self, book, empty_state):
        """A crossing order is accepted (passes validation) before matching."""
        state, _ = book.submit_order(empty_state, _make_buy(price=50.00))
        _state, result = book.submit_order(state, _make_sell(price=50.00))
        assert result.accepted == True

    def test_non_crossing_orders_no_trade(self, book, empty_state):
        """Buy at 45, sell at 55 -> no crossing, traded=False."""
        state, _ = book.submit_order(empty_state, _make_buy(price=45.00))
        _state, result = book.submit_order(state, _make_sell(price=55.00))
        assert result.traded == False

    def test_non_crossing_trade_qty_zero(self, book, empty_state):
        """Non-crossing orders produce trade_qty = 0."""
        state, _ = book.submit_order(empty_state, _make_buy(price=45.00))
        _state, result = book.submit_order(state, _make_sell(price=55.00))
        assert result.trade_qty == pytest.approx(0.0, abs=0.05)

    def test_non_crossing_trade_price_zero(self, book, empty_state):
        """Non-crossing orders produce trade_price = 0."""
        state, _ = book.submit_order(empty_state, _make_buy(price=45.00))
        _state, result = book.submit_order(state, _make_sell(price=55.00))
        assert result.trade_price == pytest.approx(0.0, abs=0.005)

    def test_non_crossing_settlement_value_zero(self, book, empty_state):
        """Non-crossing orders produce settlement_value = 0."""
        state, _ = book.submit_order(empty_state, _make_buy(price=45.00))
        _state, result = book.submit_order(state, _make_sell(price=55.00))
        assert result.settlement_value == pytest.approx(0.0, abs=0.005)

    def test_submit_into_empty_book_no_trade(self, book, empty_state):
        """First order into empty book: accepted but no trade."""
        _state, result = book.submit_order(empty_state, _make_buy(price=50.00))
        assert result.accepted == True
        assert result.traded == False

    def test_same_agent_crossing_still_trades(self, book, empty_state):
        """Same agent on both sides still produces a trade (self-trade).

        OMIE is anonymous; the matching engine does not prevent self-trades
        in the single-zone simulator (market-spec SS7.1).
        """
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, agent_id=0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, agent_id=0, timestamp=200)
        )
        assert result.traded == True

    def test_buy_aggressive_into_resting_sell_trades(self, book, empty_state):
        """Buy at 55 crosses resting sell at 50 -> trade occurs."""
        state, _ = book.submit_order(empty_state, _make_sell(price=50.00, timestamp=100))
        _state, result = book.submit_order(state, _make_buy(price=55.00, timestamp=200))
        assert result.traded == True

    def test_sell_aggressive_into_resting_buy_trades(self, book, empty_state):
        """Sell at 45 crosses resting buy at 50 -> trade occurs."""
        state, _ = book.submit_order(empty_state, _make_buy(price=50.00, timestamp=100))
        _state, result = book.submit_order(state, _make_sell(price=45.00, timestamp=200))
        assert result.traded == True

    def test_resting_order_removed_after_full_match(self, book, empty_state):
        """After a full match, the resting order is no longer in the book."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=200)
        )
        assert _is_absent(book.get_order(state, 1, Side.BUY))

    def test_incoming_order_removed_after_full_match(self, book, empty_state):
        """After a full match, the incoming order is also consumed."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=200)
        )
        assert _is_absent(book.get_order(state, 2, Side.SELL))


# ===========================================================================
# 2. TestMatchingPriceIsRestingOrder
# ===========================================================================

class TestMatchingPriceIsRestingOrder:
    """Trade executes at the resting (maker) order's limit price.

    Domain rule (market-spec SS6.2, step 3): "Trade executes at the
    resting order's limit price (maker pricing)."
    """

    def test_aggressive_buy_trades_at_resting_sell_price(self, book, empty_state):
        """Resting sell at 50, incoming buy at 55 -> trade at 50."""
        state, _ = book.submit_order(
            empty_state, _make_sell(price=50.00, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_buy(price=55.00, timestamp=200)
        )
        assert result.trade_price == pytest.approx(50.00, abs=0.005)

    def test_aggressive_sell_trades_at_resting_buy_price(self, book, empty_state):
        """Resting buy at 55, incoming sell at 50 -> trade at 55."""
        state, _ = book.submit_order(
            empty_state, _make_buy(price=55.00, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(price=50.00, timestamp=200)
        )
        assert result.trade_price == pytest.approx(55.00, abs=0.005)

    def test_exact_price_match_trades_at_that_price(self, book, empty_state):
        """Both at 50.00 -> trade at 50.00."""
        state, _ = book.submit_order(
            empty_state, _make_buy(price=50.00, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(price=50.00, timestamp=200)
        )
        assert result.trade_price == pytest.approx(50.00, abs=0.005)

    def test_resting_price_not_incoming_price(self, book, empty_state):
        """Confirm the trade price is not the aggressive (incoming) price.

        Resting sell at 48.50, incoming buy at 52.00 -> trade at 48.50.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(price=48.50, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_buy(price=52.00, timestamp=200)
        )
        assert result.trade_price == pytest.approx(48.50, abs=0.005)
        assert result.trade_price != pytest.approx(52.00, abs=0.005)

    def test_negative_resting_price_used(self, book, empty_state):
        """Negative prices are valid in OMIE (SS2.3). Trade at resting price.

        Resting sell at -10.00, incoming buy at 0.00 -> trade at -10.00.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(price=-10.00, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_buy(price=0.00, timestamp=200)
        )
        assert result.trade_price == pytest.approx(-10.00, abs=0.005)

    def test_large_price_spread_uses_resting(self, book, empty_state):
        """Resting sell at 10.00, aggressive buy at 9000.00 -> trade at 10.00."""
        state, _ = book.submit_order(
            empty_state, _make_sell(price=10.00, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_buy(price=9000.00, timestamp=200)
        )
        assert result.trade_price == pytest.approx(10.00, abs=0.005)

    def test_resting_buy_price_when_sell_aggresses(self, book, empty_state):
        """Resting buy at 100.00, incoming sell at 90.00 -> trade at 100.00.

        Buyer gets price improvement; seller executes at buyer's (better) price.
        """
        state, _ = book.submit_order(
            empty_state, _make_buy(price=100.00, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(price=90.00, timestamp=200)
        )
        assert result.trade_price == pytest.approx(100.00, abs=0.005)


# ===========================================================================
# 3. TestPartialFillNonRestriction
# ===========================================================================

class TestPartialFillNonRestriction:
    """NON order partially fills; residual stays in the book.

    Domain rule (market-spec SS6.2, step 4): "If incoming order is
    partially filled and has NON restriction: residual enters the book."
    """

    def test_partial_fill_trade_qty_is_smaller_side(self, book, empty_state):
        """Buy 10 MW, sell 5 MW -> trade_qty = 5.0 MW (limited by seller)."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=200)
        )
        assert result.trade_qty == pytest.approx(5.0, abs=0.05)

    def test_partial_fill_residual_buy_remains_in_book(self, book, empty_state):
        """Buy 10 MW resting, sell 5 MW incoming -> 5 MW residual buy remains."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=200)
        )
        residual = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(residual)
        assert residual.quantity == pytest.approx(5.0, abs=0.05)

    def test_partial_fill_residual_sell_remains_in_book(self, book, empty_state):
        """Sell 10 MW resting, buy 5 MW incoming -> 5 MW residual sell remains."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, timestamp=200)
        )
        residual = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(residual)
        assert residual.quantity == pytest.approx(5.0, abs=0.05)

    def test_incoming_non_order_residual_enters_book(self, book, empty_state):
        """Resting sell 5 MW, incoming buy 10 MW NON -> 5 MW residual buy enters book."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=200)
        )
        residual = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(residual)
        assert residual.quantity == pytest.approx(5.0, abs=0.05)

    def test_full_match_no_residual(self, book, empty_state):
        """Equal quantities: no residual on either side."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=200)
        )
        assert _is_absent(book.get_order(state, 1, Side.BUY))
        assert _is_absent(book.get_order(state, 2, Side.SELL))

    def test_partial_fill_traded_is_true(self, book, empty_state):
        """Partial fill still sets traded=True on the result."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=200)
        )
        assert result.traded == True

    def test_partial_fill_accepted_is_true(self, book, empty_state):
        """Partial fill still sets accepted=True on the result."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=200)
        )
        assert result.accepted == True

    def test_minimum_quantity_partial_fill(self, book, empty_state):
        """Minimum qty trade: buy 0.2 MW, sell 0.1 MW -> trade 0.1, residual 0.1."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=0.2, timestamp=100)
        )
        state, result = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=0.1, timestamp=200)
        )
        assert result.trade_qty == pytest.approx(0.1, abs=0.05)
        residual = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(residual)
        assert residual.quantity == pytest.approx(0.1, abs=0.05)

    def test_multiple_partial_fills_drain_resting_order(self, book, empty_state):
        """Resting buy 10 MW, two sells of 5 MW each -> fully consumed."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=200)
        )
        # After first partial fill, 5 MW residual
        residual = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(residual)
        assert residual.quantity == pytest.approx(5.0, abs=0.05)

        # Second sell consumes the rest
        state, result = book.submit_order(
            state, _make_sell(order_id=3, price=50.00, quantity=5.0, timestamp=300)
        )
        assert result.traded == True
        assert result.trade_qty == pytest.approx(5.0, abs=0.05)
        assert _is_absent(book.get_order(state, 1, Side.BUY))

    def test_partial_fill_preserves_other_orders(self, book, empty_state):
        """A partial fill on one order does not affect unrelated orders."""
        # Two buy orders at different prices
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=3, price=45.00, quantity=5.0, timestamp=150)
        )
        # Sell partially fills the best buy
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=200)
        )
        # Second buy at 45 is untouched
        other = book.get_order(state, 3, Side.BUY)
        assert not _is_absent(other)
        assert other.quantity == pytest.approx(5.0, abs=0.05)


# ===========================================================================
# 4. TestPriceTimePriority
# ===========================================================================

class TestPriceTimePriority:
    """Price-time priority: best price first, then earliest timestamp.

    Domain rule (market-spec SS6.1):
    - Buy side: highest price first; ties by earliest timestamp.
    - Sell side: lowest price first; ties by earliest timestamp.
    """

    def test_best_sell_price_matched_first(self, book, empty_state):
        """Two sells at 48 and 50; incoming buy at 52 -> trades at 48 (best ask)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=3, price=48.00, quantity=10.0, timestamp=200)
        )
        _state, result = book.submit_order(
            state, _make_buy(order_id=1, price=52.00, quantity=10.0, timestamp=300)
        )
        # Should trade at 48 (the better sell price)
        assert result.trade_price == pytest.approx(48.00, abs=0.005)

    def test_best_buy_price_matched_first(self, book, empty_state):
        """Two buys at 50 and 55; incoming sell at 45 -> trades at 55 (best bid)."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=3, price=55.00, quantity=10.0, timestamp=200)
        )
        _state, result = book.submit_order(
            state, _make_sell(order_id=2, price=45.00, quantity=10.0, timestamp=300)
        )
        # Should trade at 55 (the best bid price)
        assert result.trade_price == pytest.approx(55.00, abs=0.005)

    def test_same_price_earlier_timestamp_matched_first(self, book, empty_state):
        """Two sells at same price 50; ts=100 matched before ts=200.

        After incoming buy fills first sell (ts=100), second sell (ts=200)
        should remain in book.
        """
        state, _ = book.submit_order(
            empty_state,
            _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100),
        )
        state, _ = book.submit_order(
            state,
            _make_sell(order_id=3, price=50.00, quantity=5.0, timestamp=200),
        )
        # Buy exactly 5 MW -> should consume only the earlier sell (oid=2)
        state, result = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, timestamp=300)
        )
        assert result.traded == True
        # Earlier sell (oid=2) consumed
        assert _is_absent(book.get_order(state, 2, Side.SELL))
        # Later sell (oid=3) still in book
        later = book.get_order(state, 3, Side.SELL)
        assert not _is_absent(later)
        assert later.quantity == pytest.approx(5.0, abs=0.05)

    def test_same_price_earlier_buy_matched_first(self, book, empty_state):
        """Two buys at same price 50; ts=100 matched before ts=200."""
        state, _ = book.submit_order(
            empty_state,
            _make_buy(order_id=1, price=50.00, quantity=5.0, timestamp=100),
        )
        state, _ = book.submit_order(
            state,
            _make_buy(order_id=3, price=50.00, quantity=5.0, timestamp=200),
        )
        # Sell exactly 5 MW -> consumes earlier buy (oid=1)
        state, result = book.submit_order(
            state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=300)
        )
        assert result.traded == True
        assert _is_absent(book.get_order(state, 1, Side.BUY))
        later = book.get_order(state, 3, Side.BUY)
        assert not _is_absent(later)
        assert later.quantity == pytest.approx(5.0, abs=0.05)

    def test_price_priority_overrides_time_priority(self, book, empty_state):
        """Better price beats earlier timestamp.

        Sell at 52 (ts=100), sell at 48 (ts=200). Buy at 52 (ts=300).
        Should match sell at 48 (better price) despite later timestamp.
        """
        state, _ = book.submit_order(
            empty_state,
            _make_sell(order_id=2, price=52.00, quantity=10.0, timestamp=100),
        )
        state, _ = book.submit_order(
            state,
            _make_sell(order_id=3, price=48.00, quantity=10.0, timestamp=200),
        )
        state, result = book.submit_order(
            state, _make_buy(order_id=1, price=52.00, quantity=10.0, timestamp=300)
        )
        assert result.trade_price == pytest.approx(48.00, abs=0.005)
        # Better-priced sell consumed
        assert _is_absent(book.get_order(state, 3, Side.SELL))
        # Worse-priced sell still in book
        assert not _is_absent(book.get_order(state, 2, Side.SELL))

    def test_walk_through_multiple_price_levels(self, book, empty_state):
        """Incoming buy walks through multiple ask levels.

        Sells: 5 MW at 48, 5 MW at 50. Buy 10 MW at 52.
        First fills at 48, then at 50. Total 10 MW traded.
        """
        state, _ = book.submit_order(
            empty_state,
            _make_sell(order_id=2, price=48.00, quantity=5.0, timestamp=100),
        )
        state, _ = book.submit_order(
            state,
            _make_sell(order_id=3, price=50.00, quantity=5.0, timestamp=200),
        )
        state, result = book.submit_order(
            state, _make_buy(order_id=1, price=52.00, quantity=10.0, timestamp=300)
        )
        # Both sells should be consumed
        assert _is_absent(book.get_order(state, 2, Side.SELL))
        assert _is_absent(book.get_order(state, 3, Side.SELL))
        # Incoming buy fully filled, not in book
        assert _is_absent(book.get_order(state, 1, Side.BUY))

    def test_walk_leaves_residual_at_last_level(self, book, empty_state):
        """Buy walks through levels, partially fills last level.

        Sells: 5 MW at 48, 10 MW at 50. Buy 8 MW at 52.
        Fills 5 at 48 + 3 at 50. Residual sell of 7 MW at 50 remains.
        """
        state, _ = book.submit_order(
            empty_state,
            _make_sell(order_id=2, price=48.00, quantity=5.0, timestamp=100),
        )
        state, _ = book.submit_order(
            state,
            _make_sell(order_id=3, price=50.00, quantity=10.0, timestamp=200),
        )
        state, result = book.submit_order(
            state, _make_buy(order_id=1, price=52.00, quantity=8.0, timestamp=300)
        )
        # First sell fully consumed
        assert _is_absent(book.get_order(state, 2, Side.SELL))
        # Second sell partially filled: 10 - 3 = 7 MW residual
        residual = book.get_order(state, 3, Side.SELL)
        assert not _is_absent(residual)
        assert residual.quantity == pytest.approx(7.0, abs=0.05)
        # Incoming buy fully filled
        assert _is_absent(book.get_order(state, 1, Side.BUY))

    def test_incoming_buy_stops_at_uncrossed_level(self, book, empty_state):
        """Buy at 49 matches sell at 48 but not sell at 50.

        Sells: 5 MW at 48, 5 MW at 50. Buy 10 MW at 49.
        Fills 5 at 48 only. 5 MW residual buy at 49 enters book.
        """
        state, _ = book.submit_order(
            empty_state,
            _make_sell(order_id=2, price=48.00, quantity=5.0, timestamp=100),
        )
        state, _ = book.submit_order(
            state,
            _make_sell(order_id=3, price=50.00, quantity=5.0, timestamp=200),
        )
        state, result = book.submit_order(
            state, _make_buy(order_id=1, price=49.00, quantity=10.0, timestamp=300)
        )
        assert result.traded == True
        # Sell at 48 consumed
        assert _is_absent(book.get_order(state, 2, Side.SELL))
        # Sell at 50 untouched (price not crossed)
        uncrossed = book.get_order(state, 3, Side.SELL)
        assert not _is_absent(uncrossed)
        assert uncrossed.quantity == pytest.approx(5.0, abs=0.05)
        # Residual buy enters book
        residual_buy = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(residual_buy)
        assert residual_buy.quantity == pytest.approx(5.0, abs=0.05)


# ===========================================================================
# 5. TestTradeSettlementValue
# ===========================================================================

class TestTradeSettlementValue:
    """Settlement value: V = Q (MW) x p (EUR/MWh) x 0.25 (hours).

    Domain rule (market-spec SS2.3): "V (EUR) = Q (MW) x p (EUR/MWh) x d (h)
    where d = 0.25 h for 15-min contracts."

    Architecture decision: settlement computed inline in submit_order.
    SubmitResult.settlement_value carries the value in EUR.
    """

    def test_settlement_10mw_at_50(self, book, empty_state):
        """V = 10 MW x 50 EUR/MWh x 0.25 h = 125.0 EUR."""
        state, _ = book.submit_order(
            empty_state, _make_buy(price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(price=50.00, quantity=10.0, timestamp=200)
        )
        assert result.settlement_value == pytest.approx(125.0, abs=0.01)

    def test_settlement_5mw_at_100(self, book, empty_state):
        """V = 5 MW x 100 EUR/MWh x 0.25 h = 125.0 EUR."""
        state, _ = book.submit_order(
            empty_state, _make_buy(price=100.00, quantity=5.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(price=100.00, quantity=5.0, timestamp=200)
        )
        assert result.settlement_value == pytest.approx(125.0, abs=0.01)

    def test_settlement_uses_resting_price(self, book, empty_state):
        """V uses resting price (48.50), not incoming price (55.00).

        V = 10 MW x 48.50 EUR/MWh x 0.25 h = 121.25 EUR.
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(price=48.50, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_buy(price=55.00, quantity=10.0, timestamp=200)
        )
        assert result.settlement_value == pytest.approx(121.25, abs=0.01)

    def test_settlement_partial_fill_uses_filled_qty(self, book, empty_state):
        """Partial fill: V uses the filled quantity, not the order quantity.

        Buy 10 MW, sell 5 MW at 50 -> V = 5 x 50 x 0.25 = 62.5 EUR.
        """
        state, _ = book.submit_order(
            empty_state, _make_buy(price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(price=50.00, quantity=5.0, timestamp=200)
        )
        assert result.settlement_value == pytest.approx(62.5, abs=0.01)

    def test_settlement_minimum_trade(self, book, empty_state):
        """Minimum trade: V = 0.1 MW x 0.01 EUR/MWh x 0.25 = 0.00025 EUR."""
        state, _ = book.submit_order(
            empty_state, _make_buy(price=0.01, quantity=0.1, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(price=0.01, quantity=0.1, timestamp=200)
        )
        assert result.settlement_value == pytest.approx(0.00025, abs=1e-6)

    def test_settlement_negative_price(self, book, empty_state):
        """Negative price settlement: V = 10 x (-10.00) x 0.25 = -25.0 EUR.

        Negative settlement is economically valid (buyer is paid to take energy).
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(price=-10.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_buy(price=0.00, quantity=10.0, timestamp=200)
        )
        assert result.settlement_value == pytest.approx(-25.0, abs=0.01)

    def test_settlement_no_trade_zero(self, book, empty_state):
        """No trade -> settlement_value = 0."""
        state, _ = book.submit_order(
            empty_state, _make_buy(price=45.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(price=55.00, quantity=10.0, timestamp=200)
        )
        assert result.settlement_value == pytest.approx(0.0, abs=0.005)

    def test_settlement_large_trade(self, book, empty_state):
        """Large trade: V = 999 MW x 9999 EUR/MWh x 0.25 = 2,497,250.25 EUR."""
        state, _ = book.submit_order(
            empty_state,
            _make_buy(order_id=1, price=9999.00, quantity=999.0, timestamp=100),
        )
        _state, result = book.submit_order(
            state,
            _make_sell(order_id=2, price=9999.00, quantity=999.0, timestamp=200),
        )
        expected = 999.0 * 9999.0 * 0.25
        assert result.settlement_value == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# 6. TestSubmitResultExpanded
# ===========================================================================

class TestSubmitResultExpanded:
    """SubmitResult has the expanded fields from the M3.2 architecture decision.

    Fields: accepted (index 0), traded, trade_qty, trade_price, settlement_value.
    accepted at index 0 preserves backward compatibility with M3.1.
    """

    def test_submit_result_has_accepted_field(self, book, empty_state):
        """SubmitResult has an 'accepted' field."""
        _state, result = book.submit_order(empty_state, _make_buy())
        assert hasattr(result, "accepted")

    def test_submit_result_has_traded_field(self, book, empty_state):
        """SubmitResult has a 'traded' field."""
        _state, result = book.submit_order(empty_state, _make_buy())
        assert hasattr(result, "traded")

    def test_submit_result_has_trade_qty_field(self, book, empty_state):
        """SubmitResult has a 'trade_qty' field."""
        _state, result = book.submit_order(empty_state, _make_buy())
        assert hasattr(result, "trade_qty")

    def test_submit_result_has_trade_price_field(self, book, empty_state):
        """SubmitResult has a 'trade_price' field."""
        _state, result = book.submit_order(empty_state, _make_buy())
        assert hasattr(result, "trade_price")

    def test_submit_result_has_settlement_value_field(self, book, empty_state):
        """SubmitResult has a 'settlement_value' field."""
        _state, result = book.submit_order(empty_state, _make_buy())
        assert hasattr(result, "settlement_value")

    def test_accepted_is_at_index_zero(self, book, empty_state):
        """accepted is the first field (index 0) for backward compatibility."""
        _state, result = book.submit_order(empty_state, _make_buy())
        assert result[0] == result.accepted

    def test_submit_result_is_namedtuple(self, book, empty_state):
        """SubmitResult is a NamedTuple (tuple subclass)."""
        _state, result = book.submit_order(empty_state, _make_buy())
        assert isinstance(result, tuple)
        assert isinstance(result, SubmitResult)

    def test_rejected_order_result_fields(self, book, empty_state):
        """Rejected order: accepted=False, traded=False, all trade fields zero."""
        bad_order = _make_buy(order_id=-1)  # negative ID -> rejected
        _state, result = book.submit_order(empty_state, bad_order)
        assert result.accepted == False
        assert result.traded == False
        assert result.trade_qty == pytest.approx(0.0, abs=0.05)
        assert result.trade_price == pytest.approx(0.0, abs=0.005)
        assert result.settlement_value == pytest.approx(0.0, abs=0.005)

    def test_accepted_no_match_result_fields(self, book, empty_state):
        """Accepted but no match: accepted=True, traded=False, trade fields zero."""
        _state, result = book.submit_order(empty_state, _make_buy(price=45.00))
        assert result.accepted == True
        assert result.traded == False
        assert result.trade_qty == pytest.approx(0.0, abs=0.05)
        assert result.trade_price == pytest.approx(0.0, abs=0.005)
        assert result.settlement_value == pytest.approx(0.0, abs=0.005)

    def test_matched_result_fields(self, book, empty_state):
        """Matched: accepted=True, traded=True, trade fields populated."""
        state, _ = book.submit_order(
            empty_state, _make_buy(price=50.00, quantity=10.0, timestamp=100)
        )
        _state, result = book.submit_order(
            state, _make_sell(price=50.00, quantity=10.0, timestamp=200)
        )
        assert result.accepted == True
        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=0.05)
        assert result.trade_price == pytest.approx(50.00, abs=0.005)
        assert result.settlement_value == pytest.approx(125.0, abs=0.01)
