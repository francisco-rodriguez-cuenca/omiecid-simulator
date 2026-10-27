"""Audit tests for M3.2 -- Price-Time Priority Matching.

Post-implementation verification focusing on edge cases, boundary conditions,
and acceptance criteria gaps:

- Crossing semantics: exact crossing, wide spread, non-crossing no trade
- Resting (maker) price rule: aggressive buy/sell always trades at resting price
- Partial fill residual: NON residual stays in book, correct remaining qty
- Price-time priority: multi-level walk-through, same-price timestamp tie-break
- Settlement formula: V = Q x p x 0.25 for various scenarios
- Self-trade: same agent on both sides
- SubmitResult field completeness: all 5 fields present and correct
- Boundary values: minimum qty match, negative price matching, extreme prices
- No-match scenarios: non-crossing prices, empty book
- Multiple sequential trades draining liquidity

Derived independently from:
- docs/omiecid/market-spec.md §6 (matching rules), §8 (settlement)
- docs/omiecid/agile-plan.md (M3.2 acceptance criteria)
- docs/omiecid/migration-plan.md §M3 (interface)
- omiecid_simulator/omiecid/contract_book.py (public interface only)
"""

import pytest
from pytest import approx

from omiecid_simulator.omiecid.contract_book import SubmitResult
from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    make_order,
)

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




def _buy(contract_id=37, price=50.0, qty=1.0, order_id=1, agent_id=100,
         timestamp=1000):
    return make_order(
        contract_id=contract_id, side=Side.BUY, price=price, quantity=qty,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
    )


def _sell(contract_id=37, price=50.0, qty=1.0, order_id=2, agent_id=200,
          timestamp=2000):
    return make_order(
        contract_id=contract_id, side=Side.SELL, price=price, quantity=qty,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
    )


# ===========================================================================
# 1. Crossing Semantics — when does a trade happen?
# ===========================================================================

class TestCrossingSemantics:
    """Edge cases around when orders do and do not cross the spread."""

    def test_exact_price_crossing(self, book, empty_state):
        """Buy at 50, sell at 50 -> trade (buy_price >= sell_price)."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0))
        state, result = book.submit_order(state, _sell(price=50.0))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)

    def test_buy_higher_than_sell_crosses(self, book, empty_state):
        """Buy at 55, sell at 45 -> trade."""
        state, _ = book.submit_order(empty_state, _sell(price=45.0))
        state, result = book.submit_order(state, _buy(price=55.0, order_id=3))
        assert result.traded == True

    def test_buy_lower_than_sell_no_trade(self, book, empty_state):
        """Buy at 45, sell at 55 -> no trade (buy_price < sell_price)."""
        state, _ = book.submit_order(empty_state, _buy(price=45.0))
        state, result = book.submit_order(state, _sell(price=55.0))
        assert result.traded == False
        assert result.trade_qty == approx(0.0)

    def test_submit_to_empty_book_no_trade(self, book, empty_state):
        """A lone buy into an empty book cannot trade."""
        state, result = book.submit_order(empty_state, _buy(price=50.0))
        assert result.accepted == True
        assert result.traded == False

    def test_submit_sell_to_empty_book_no_trade(self, book, empty_state):
        """A lone sell into an empty book cannot trade."""
        state, result = book.submit_order(empty_state, _sell(price=50.0))
        assert result.accepted == True
        assert result.traded == False

    def test_buy_one_tick_above_sell_crosses(self, book, empty_state):
        """Buy at 50.01, sell at 50.00 -> trade (minimum spread crossing)."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0))
        state, result = book.submit_order(
            state, _buy(price=50.01, order_id=3, timestamp=2001))
        assert result.traded == True

    def test_buy_one_tick_below_sell_no_trade(self, book, empty_state):
        """Buy at 49.99, sell at 50.00 -> no trade."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0))
        state, result = book.submit_order(
            state, _buy(price=49.99, order_id=3, timestamp=2001))
        assert result.traded == False


# ===========================================================================
# 2. Resting (Maker) Price Rule
# ===========================================================================

class TestRestingPriceRule:
    """Trade always executes at the resting (passive/maker) order's price.

    Market-spec §6.2 rule 3: "Trade executes at the resting order's limit
    price (maker pricing)."
    """

    def test_aggressive_buy_trades_at_resting_sell_price(self, book, empty_state):
        """Resting sell at 48.0, aggressive buy at 55.0 -> trade at 48.0."""
        state, _ = book.submit_order(empty_state, _sell(price=48.0))
        state, result = book.submit_order(
            state, _buy(price=55.0, order_id=3, timestamp=3000))
        assert result.trade_price == approx(48.0)

    def test_aggressive_sell_trades_at_resting_buy_price(self, book, empty_state):
        """Resting buy at 55.0, aggressive sell at 48.0 -> trade at 55.0."""
        state, _ = book.submit_order(empty_state, _buy(price=55.0))
        state, result = book.submit_order(
            state, _sell(price=48.0, order_id=3, timestamp=3000))
        assert result.trade_price == approx(55.0)

    def test_exact_match_price_equals_both(self, book, empty_state):
        """Both at 50.0 -> trade at 50.0."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0))
        state, result = book.submit_order(
            state, _buy(price=50.0, order_id=3, timestamp=3000))
        assert result.trade_price == approx(50.0)

    def test_large_spread_uses_resting_price(self, book, empty_state):
        """Resting sell at 10.0, aggressive buy at 9000.0 -> trade at 10.0."""
        state, _ = book.submit_order(empty_state, _sell(price=10.0))
        state, result = book.submit_order(
            state, _buy(price=9000.0, order_id=3, timestamp=3000))
        assert result.trade_price == approx(10.0)

    def test_negative_resting_price(self, book, empty_state):
        """Resting sell at -50.0, aggressive buy at -30.0 -> trade at -50.0."""
        state, _ = book.submit_order(
            empty_state, _sell(price=-50.0, qty=0.5, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=-30.0, qty=0.5, order_id=11, timestamp=3000))
        assert result.trade_price == approx(-50.0)


# ===========================================================================
# 3. Partial Fill + NON Residual
# ===========================================================================

class TestPartialFillResidual:
    """Partial fill: smaller side is fully consumed; NON residual stays.

    Market-spec §6.2 rule 4: "If incoming order is partially filled and has
    NON restriction: residual enters the book."
    """

    def test_incoming_larger_than_resting_residual_enters_book(
        self, book, empty_state
    ):
        """Resting sell 1.0, incoming buy 3.0 -> trade 1.0, buy residual 2.0."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=1.0, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=3.0, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        # Residual buy of 2.0 should be in the book
        remaining = book.get_order(state, 11, Side.BUY)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(2.0)

    def test_resting_larger_than_incoming_residual_stays(
        self, book, empty_state
    ):
        """Resting sell 5.0, incoming buy 2.0 -> trade 2.0, sell residual 3.0."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=5.0, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=2.0, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)
        # Resting sell residual should be 3.0
        remaining = book.get_order(state, 10, Side.SELL)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(3.0)

    def test_full_match_both_removed(self, book, empty_state):
        """Both 1.0 -> trade 1.0, both removed from book."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=1.0, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=1.0, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        assert _is_absent(book.get_order(state, 10, Side.SELL))
        assert _is_absent(book.get_order(state, 11, Side.BUY))

    def test_minimum_qty_partial_fill(self, book, empty_state):
        """Resting sell 0.1, incoming buy 0.5 -> trade 0.1, residual 0.4."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=0.1, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=0.5, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(0.1)
        remaining = book.get_order(state, 11, Side.BUY)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(0.4)

    def test_multiple_partial_fills_drain_resting(self, book, empty_state):
        """Resting sell 3.0, two incoming buys of 1.0 each -> 2 trades, residual 1.0."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=3.0, order_id=10))
        # First buy: 1.0
        state, r1 = book.submit_order(
            state, _buy(price=50.0, qty=1.0, order_id=11, timestamp=3000))
        assert r1.traded == True
        assert r1.trade_qty == approx(1.0)
        # Second buy: 1.0
        state, r2 = book.submit_order(
            state, _buy(price=50.0, qty=1.0, order_id=12, timestamp=4000))
        assert r2.traded == True
        assert r2.trade_qty == approx(1.0)
        # Residual sell: 1.0
        remaining = book.get_order(state, 10, Side.SELL)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(1.0)

    def test_unrelated_orders_preserved_after_match(self, book, empty_state):
        """A resting buy at a lower price is not affected by a match above."""
        # Resting buy at 45.0 (won't match sell at 50.0)
        state, _ = book.submit_order(
            empty_state, _buy(price=45.0, qty=1.0, order_id=5, timestamp=500))
        # Resting sell at 50.0
        state, _ = book.submit_order(
            state, _sell(price=50.0, qty=1.0, order_id=10, timestamp=1000))
        # Aggressive buy at 50.0 matches the sell
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=1.0, order_id=11, timestamp=3000))
        assert result.traded == True
        # The unrelated buy at 45.0 is still in the book
        unrelated = book.get_order(state, 5, Side.BUY)
        assert not _is_absent(unrelated)
        assert unrelated.quantity == approx(1.0)


# ===========================================================================
# 4. Price-Time Priority
# ===========================================================================

class TestPriceTimePriorityAudit:
    """Price-time priority: best price first, then earliest timestamp.

    Market-spec §6.1:
    - Buy side: highest price first; ties by earliest timestamp.
    - Sell side: lowest price first; ties by earliest timestamp.
    """

    def test_best_sell_price_matched_first(self, book, empty_state):
        """Two sells at 48.0 and 52.0. Buy crosses both -> matches 48.0 first."""
        state, _ = book.submit_order(
            empty_state, _sell(price=52.0, qty=1.0, order_id=10, timestamp=100))
        state, _ = book.submit_order(
            state, _sell(price=48.0, qty=1.0, order_id=11, timestamp=200))
        # Aggressive buy that crosses both levels
        state, result = book.submit_order(
            state, _buy(price=55.0, qty=1.0, order_id=20, timestamp=3000))
        # Should match the cheaper sell (48.0)
        assert result.traded == True
        assert result.trade_price == approx(48.0)
        # Sell at 52.0 should still be in the book
        assert not _is_absent(book.get_order(state, 10, Side.SELL))
        assert _is_absent(book.get_order(state, 11, Side.SELL))

    def test_best_buy_price_matched_first(self, book, empty_state):
        """Two buys at 48.0 and 52.0. Sell crosses both -> matches 52.0 first."""
        state, _ = book.submit_order(
            empty_state, _buy(price=48.0, qty=1.0, order_id=10, timestamp=100))
        state, _ = book.submit_order(
            state, _buy(price=52.0, qty=1.0, order_id=11, timestamp=200))
        # Aggressive sell that crosses both levels
        state, result = book.submit_order(
            state, _sell(price=45.0, qty=1.0, order_id=20, timestamp=3000))
        # Should match the higher buy (52.0)
        assert result.traded == True
        assert result.trade_price == approx(52.0)
        assert _is_absent(book.get_order(state, 11, Side.BUY))
        assert not _is_absent(book.get_order(state, 10, Side.BUY))

    def test_same_price_earliest_timestamp_first(self, book, empty_state):
        """Two sells at same price. Earlier timestamp matched first."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=1.0, order_id=10, timestamp=100))
        state, _ = book.submit_order(
            state, _sell(price=50.0, qty=1.0, order_id=11, timestamp=200))
        # Buy matches one sell
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=1.0, order_id=20, timestamp=3000))
        assert result.traded == True
        # Earlier timestamp (order 10) should be consumed first
        assert _is_absent(book.get_order(state, 10, Side.SELL))
        assert not _is_absent(book.get_order(state, 11, Side.SELL))

    def test_price_priority_overrides_time_priority(self, book, empty_state):
        """Earlier sell at 52.0 vs later sell at 48.0. Buy matches cheaper."""
        state, _ = book.submit_order(
            empty_state, _sell(price=52.0, qty=1.0, order_id=10, timestamp=100))
        state, _ = book.submit_order(
            state, _sell(price=48.0, qty=1.0, order_id=11, timestamp=200))
        state, result = book.submit_order(
            state, _buy(price=55.0, qty=1.0, order_id=20, timestamp=3000))
        # Price 48.0 wins even though its timestamp (200) is later
        assert result.trade_price == approx(48.0)
        assert _is_absent(book.get_order(state, 11, Side.SELL))
        assert not _is_absent(book.get_order(state, 10, Side.SELL))

    def test_walk_through_two_price_levels(self, book, empty_state):
        """Aggressive buy walks through two sell levels."""
        state, _ = book.submit_order(
            empty_state, _sell(price=48.0, qty=1.0, order_id=10, timestamp=100))
        state, _ = book.submit_order(
            state, _sell(price=52.0, qty=1.0, order_id=11, timestamp=200))
        # Buy 2.0 at 55.0 walks through both levels
        state, result = book.submit_order(
            state, _buy(price=55.0, qty=2.0, order_id=20, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)
        # Both sells consumed
        assert _is_absent(book.get_order(state, 10, Side.SELL))
        assert _is_absent(book.get_order(state, 11, Side.SELL))

    def test_walk_stops_at_uncrossed_level(self, book, empty_state):
        """Buy at 50.0 walks through sell at 48.0 but stops at sell at 52.0."""
        state, _ = book.submit_order(
            empty_state, _sell(price=48.0, qty=1.0, order_id=10, timestamp=100))
        state, _ = book.submit_order(
            state, _sell(price=52.0, qty=1.0, order_id=11, timestamp=200))
        # Buy at 50.0 can only match sell at 48.0
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=5.0, order_id=20, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        # Sell at 52.0 survives (not crossed)
        assert not _is_absent(book.get_order(state, 11, Side.SELL))
        # Residual buy at 50.0 should be in book
        remaining = book.get_order(state, 20, Side.BUY)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(4.0)

    def test_walk_leaves_partial_residual_at_last_level(self, book, empty_state):
        """Buy 1.5 walks through sell 1.0@48 and partially fills sell 1.0@52."""
        state, _ = book.submit_order(
            empty_state, _sell(price=48.0, qty=1.0, order_id=10, timestamp=100))
        state, _ = book.submit_order(
            state, _sell(price=52.0, qty=1.0, order_id=11, timestamp=200))
        # Buy 1.5 at 55.0: fills all 1.0 at 48, then 0.5 from 52
        state, result = book.submit_order(
            state, _buy(price=55.0, qty=1.5, order_id=20, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(1.5)
        # Sell at 48 fully consumed
        assert _is_absent(book.get_order(state, 10, Side.SELL))
        # Sell at 52 has 0.5 remaining
        remaining = book.get_order(state, 11, Side.SELL)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(0.5)


# ===========================================================================
# 5. Settlement Value: V = Q x p x 0.25
# ===========================================================================

class TestSettlementValueAudit:
    """Settlement formula: V = Q (MW) x p (EUR/MWh) x 0.25 (h).

    Market-spec §8.1: d = 0.25 for 15-min contracts.
    """

    def test_standard_settlement(self, book, empty_state):
        """1.0 MW @ 50.0 EUR/MWh -> V = 1.0 * 50.0 * 0.25 = 12.5."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0))
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=1.0, order_id=3, timestamp=3000))
        assert result.settlement_value == approx(12.5)

    def test_settlement_uses_resting_price_not_aggressive(self, book, empty_state):
        """Sell@48, buy@55 -> trade at 48 -> V = 1.0 * 48.0 * 0.25 = 12.0."""
        state, _ = book.submit_order(
            empty_state, _sell(price=48.0, qty=1.0, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=55.0, qty=1.0, order_id=11, timestamp=3000))
        assert result.settlement_value == approx(1.0 * 48.0 * 0.25)

    def test_settlement_with_partial_fill(self, book, empty_state):
        """Sell 0.5@50, buy 2.0@50 -> trade 0.5 -> V = 0.5 * 50 * 0.25 = 6.25."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=0.5, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=2.0, order_id=11, timestamp=3000))
        assert result.trade_qty == approx(0.5)
        assert result.settlement_value == approx(0.5 * 50.0 * 0.25)

    def test_settlement_minimum_qty(self, book, empty_state):
        """0.1 MW @ 100.0 -> V = 0.1 * 100.0 * 0.25 = 2.5."""
        state, _ = book.submit_order(
            empty_state, _sell(price=100.0, qty=0.1, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=100.0, qty=0.1, order_id=11, timestamp=3000))
        assert result.settlement_value == approx(2.5)

    def test_settlement_negative_price(self, book, empty_state):
        """0.5 MW @ -20.0 -> V = 0.5 * (-20.0) * 0.25 = -2.5."""
        state, _ = book.submit_order(
            empty_state, _sell(price=-20.0, qty=0.5, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=-10.0, qty=0.5, order_id=11, timestamp=3000))
        # Resting price is -20.0
        assert result.trade_price == approx(-20.0)
        assert result.settlement_value == approx(0.5 * (-20.0) * 0.25)

    def test_settlement_zero_when_no_trade(self, book, empty_state):
        """Non-crossing orders -> settlement = 0."""
        state, _ = book.submit_order(
            empty_state, _buy(price=40.0, qty=1.0, order_id=10))
        state, result = book.submit_order(
            state, _sell(price=60.0, qty=1.0, order_id=11, timestamp=3000))
        assert result.traded == False
        assert result.settlement_value == approx(0.0)

    def test_settlement_large_values(self, book, empty_state):
        """Large trade: 100 MW @ 500 EUR/MWh -> V = 100 * 500 * 0.25 = 12500."""
        state, _ = book.submit_order(
            empty_state, _sell(price=500.0, qty=100.0, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=500.0, qty=100.0, order_id=11, timestamp=3000))
        assert result.settlement_value == approx(12500.0)

    def test_settlement_at_price_floor(self, book, empty_state):
        """Trade at price floor: -9999.0 EUR/MWh."""
        state, _ = book.submit_order(
            empty_state, _sell(price=-9999.0, qty=0.1, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=-9999.0, qty=0.1, order_id=11, timestamp=3000))
        assert result.traded == True
        expected = 0.1 * (-9999.0) * 0.25
        assert result.settlement_value == approx(expected)


# ===========================================================================
# 6. Self-Trade (same agent on both sides)
# ===========================================================================

class TestSelfTrade:
    """Market spec does not prohibit self-trade; verify it executes."""

    def test_same_agent_buy_sell_trades(self, book, empty_state):
        """Agent 100 posts buy and sell that cross -> trade allowed."""
        state, _ = book.submit_order(
            empty_state, _buy(price=50.0, agent_id=100, order_id=10))
        state, result = book.submit_order(
            state, _sell(price=50.0, agent_id=100, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)


# ===========================================================================
# 7. SubmitResult Shape and Field Contracts
# ===========================================================================

class TestSubmitResultAudit:
    """Verify SubmitResult NamedTuple shape and field semantics."""

    def test_submit_result_is_namedtuple(self):
        """SubmitResult is a NamedTuple."""
        r = SubmitResult(accepted=True)
        assert isinstance(r, tuple)
        assert hasattr(r, '_fields')

    def test_submit_result_has_five_fields(self):
        """SubmitResult has exactly 5 fields."""
        assert len(SubmitResult._fields) == 5

    def test_accepted_is_index_zero(self):
        """accepted is at position 0 (backward compat with M3.1)."""
        r = SubmitResult(accepted=True, traded=True, trade_qty=1.0,
                         trade_price=50.0, settlement_value=12.5)
        assert r[0] is True
        assert r.accepted == True

    def test_field_names(self):
        """All 5 fields present with correct names."""
        assert SubmitResult._fields == (
            'accepted', 'traded', 'trade_qty', 'trade_price', 'settlement_value'
        )

    def test_rejected_submit_result_all_zeros(self, book, empty_state):
        """Rejected order: accepted=False, traded=False, all values 0."""
        bad = make_order(
            contract_id=37, side=Side.BUY, price=50.0, quantity=1.0,
            order_id=0, agent_id=100, timestamp=1000,  # order_id=0 rejected
        )
        state, result = book.submit_order(empty_state, bad)
        assert result.accepted == False
        assert result.traded == False
        assert result.trade_qty == approx(0.0)
        assert result.trade_price == approx(0.0)
        assert result.settlement_value == approx(0.0)

    def test_accepted_no_match_result(self, book, empty_state):
        """Accepted but no match: accepted=True, traded=False, values 0."""
        state, result = book.submit_order(empty_state, _buy(price=50.0))
        assert result.accepted == True
        assert result.traded == False
        assert result.trade_qty == approx(0.0)
        assert result.trade_price == approx(0.0)
        assert result.settlement_value == approx(0.0)

    def test_matched_result_all_fields_populated(self, book, empty_state):
        """Full match: all 5 fields populated with positive values."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0))
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=1.0, order_id=3, timestamp=3000))
        assert result.accepted == True
        assert result.traded == True
        assert result.trade_qty > 0
        assert result.trade_price != 0.0
        assert result.settlement_value != 0.0


# ===========================================================================
# 8. Boundary Price Matching
# ===========================================================================

class TestBoundaryPriceMatching:
    """Matching at extreme price boundaries."""

    def test_match_at_price_ceiling(self, book, empty_state):
        """Both at +9999.0 -> trade."""
        state, _ = book.submit_order(
            empty_state, _sell(price=9999.0, qty=0.1, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=9999.0, qty=0.1, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_price == approx(9999.0)

    def test_match_at_price_floor(self, book, empty_state):
        """Both at -9999.0 -> trade."""
        state, _ = book.submit_order(
            empty_state, _sell(price=-9999.0, qty=0.1, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=-9999.0, qty=0.1, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_price == approx(-9999.0)

    def test_match_at_zero_price(self, book, empty_state):
        """Both at 0.0 -> trade."""
        state, _ = book.submit_order(
            empty_state, _sell(price=0.0, qty=1.0, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=0.0, qty=1.0, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_price == approx(0.0)
        assert result.settlement_value == approx(0.0)  # V = Q * 0 * 0.25 = 0

    def test_match_buy_at_ceiling_sell_at_floor(self, book, empty_state):
        """Buy at +9999.0, sell at -9999.0 -> crosses, trades at resting."""
        state, _ = book.submit_order(
            empty_state, _sell(price=-9999.0, qty=0.1, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=9999.0, qty=0.1, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_price == approx(-9999.0)  # resting sell price


# ===========================================================================
# 9. Minimum Quantity Matching
# ===========================================================================

class TestMinimumQtyMatching:
    """Matching with minimum allowed quantities."""

    def test_minimum_qty_exact_match(self, book, empty_state):
        """Both 0.1 MW -> trade 0.1."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=0.1, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=0.1, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(0.1)

    def test_minimum_qty_settlement(self, book, empty_state):
        """0.1 MW @ 50.0 -> V = 0.1 * 50 * 0.25 = 1.25."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=0.1, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=0.1, order_id=11, timestamp=3000))
        assert result.settlement_value == approx(1.25)


# ===========================================================================
# 10. Resting order consumed, aggressive enters book
# ===========================================================================

class TestAggressiveResidualEntersBook:
    """When the incoming (aggressive) order has NON restriction and there is
    residual after matching, the residual enters the book as a resting order."""

    def test_aggressive_buy_residual_rests_at_limit_price(self, book, empty_state):
        """Buy 5.0@55, sell 2.0@50 -> trade 2.0, residual buy 3.0@55 in book."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=2.0, order_id=10))
        state, result = book.submit_order(
            state, _buy(price=55.0, qty=5.0, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)
        remaining = book.get_order(state, 11, Side.BUY)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(3.0)
        assert remaining.price == approx(55.0)

    def test_aggressive_sell_residual_rests_at_limit_price(self, book, empty_state):
        """Sell 5.0@45, buy 2.0@50 -> trade 2.0, residual sell 3.0@45 in book."""
        state, _ = book.submit_order(
            empty_state, _buy(price=50.0, qty=2.0, order_id=10))
        state, result = book.submit_order(
            state, _sell(price=45.0, qty=5.0, order_id=11, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)
        remaining = book.get_order(state, 11, Side.SELL)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(3.0)
        assert remaining.price == approx(45.0)


# ===========================================================================
# 11. Multi-trade sequential drain
# ===========================================================================

class TestSequentialDrain:
    """Multiple aggressive orders drain the same side progressively."""

    def test_three_buys_drain_three_sells(self, book, empty_state):
        """Three sells at different prices, three buys consume them in order."""
        state, _ = book.submit_order(
            empty_state, _sell(price=48.0, qty=1.0, order_id=10, timestamp=100))
        state, _ = book.submit_order(
            state, _sell(price=50.0, qty=1.0, order_id=11, timestamp=200))
        state, _ = book.submit_order(
            state, _sell(price=52.0, qty=1.0, order_id=12, timestamp=300))

        # Buy 1 at 55.0 -> matches sell@48
        state, r1 = book.submit_order(
            state, _buy(price=55.0, qty=1.0, order_id=20, timestamp=1000))
        assert r1.traded == True
        assert r1.trade_price == approx(48.0)
        assert _is_absent(book.get_order(state, 10, Side.SELL))

        # Buy 2 at 55.0 -> matches sell@50
        state, r2 = book.submit_order(
            state, _buy(price=55.0, qty=1.0, order_id=21, timestamp=2000))
        assert r2.traded == True
        assert r2.trade_price == approx(50.0)
        assert _is_absent(book.get_order(state, 11, Side.SELL))

        # Buy 3 at 55.0 -> matches sell@52
        state, r3 = book.submit_order(
            state, _buy(price=55.0, qty=1.0, order_id=22, timestamp=3000))
        assert r3.traded == True
        assert r3.trade_price == approx(52.0)
        assert _is_absent(book.get_order(state, 12, Side.SELL))

    def test_fourth_buy_no_liquidity(self, book, empty_state):
        """After draining all sells, another buy does not trade."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=1.0, order_id=10))
        state, _ = book.submit_order(
            state, _buy(price=50.0, qty=1.0, order_id=11, timestamp=3000))
        # Book is now empty on sell side
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=1.0, order_id=12, timestamp=4000))
        assert result.traded == False


# ===========================================================================
# 12. Functional state semantics
# ===========================================================================

class TestFunctionalStateSemantics:
    """Submit returns a new state; original is not mutated."""

    def test_submit_does_not_mutate_original_state(self, book, empty_state):
        """After submit, the original state still shows an empty book."""
        original = empty_state
        state, _ = book.submit_order(original, _buy(price=50.0))
        # Original should still have no order
        assert _is_absent(book.get_order(original, 1, Side.BUY))
        # New state should have the order
        assert not _is_absent(book.get_order(state, 1, Side.BUY))

    def test_trade_does_not_mutate_pre_trade_state(self, book, empty_state):
        """After a matching trade, original pre-trade state is unchanged."""
        state, _ = book.submit_order(
            empty_state, _sell(price=50.0, qty=1.0, order_id=10))
        pre_trade = state
        state, result = book.submit_order(
            state, _buy(price=50.0, qty=1.0, order_id=11, timestamp=3000))
        assert result.traded == True
        # pre_trade state should still show the sell order
        assert not _is_absent(book.get_order(pre_trade, 10, Side.SELL))


# ===========================================================================
# 13. VWAP settlement for multi-level walks
# ===========================================================================

class TestMultiLevelVWAP:
    """When an aggressive order walks through multiple price levels, the
    reported trade_price should be the VWAP and settlement uses it."""

    def test_vwap_two_levels(self, book, empty_state):
        """Buy 2.0@55 matches sell 1.0@48 + sell 1.0@52.
        VWAP = (48*1 + 52*1) / 2 = 50.0.
        Settlement = 2.0 * 50.0 * 0.25 = 25.0.
        """
        state, _ = book.submit_order(
            empty_state, _sell(price=48.0, qty=1.0, order_id=10, timestamp=100))
        state, _ = book.submit_order(
            state, _sell(price=52.0, qty=1.0, order_id=11, timestamp=200))
        state, result = book.submit_order(
            state, _buy(price=55.0, qty=2.0, order_id=20, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)
        assert result.trade_price == approx(50.0)
        assert result.settlement_value == approx(25.0)

    def test_vwap_unequal_quantities(self, book, empty_state):
        """Buy 3.0@55 matches sell 1.0@48 + sell 2.0@52.
        VWAP = (48*1 + 52*2) / 3 = 152/3 ≈ 50.6667.
        Settlement = 3.0 * 50.6667 * 0.25 ≈ 38.0.
        """
        state, _ = book.submit_order(
            empty_state, _sell(price=48.0, qty=1.0, order_id=10, timestamp=100))
        state, _ = book.submit_order(
            state, _sell(price=52.0, qty=2.0, order_id=11, timestamp=200))
        state, result = book.submit_order(
            state, _buy(price=55.0, qty=3.0, order_id=20, timestamp=3000))
        assert result.traded == True
        assert result.trade_qty == approx(3.0)
        # VWAP = (48*1 + 52*2)/3 = 152/3
        expected_vwap = (48.0 * 1 + 52.0 * 2) / 3
        assert result.trade_price == approx(expected_vwap, abs=0.02)
        expected_settlement = 3.0 * expected_vwap * 0.25
        assert result.settlement_value == approx(expected_settlement, abs=0.02)
