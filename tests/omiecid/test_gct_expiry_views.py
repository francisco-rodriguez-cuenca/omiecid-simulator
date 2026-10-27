"""Tests for M3.4 -- GCT Expiry + Views.

Behavioral contracts for expire_at_gct(), get_anonymous_lob(),
get_agent_orders(), and get_agent_trades() on ContractBook.

All expected values are derived from:
  - docs/omiecid/market-spec.md SS3.1 ("Orders on a contract are automatically
    cancelled at GCT if unmatched")
  - docs/omiecid/market-spec.md SS1 (GCT glossary, GFS validity)
  - docs/omiecid/migration-plan.md SS M3 (ContractBook interface)
  - docs/omiecid/agile-plan.md M3.4 (acceptance criteria)
  - docs/omiecid/architecture-review.md (method-by-method design)

Architecture decisions (from M3.4 architecture review):
  - expire_at_gct: direct-clear approach (replace asks/bids with -1 arrays,
    preserve trades and key). O(1), not per-order cancel loop.
  - get_anonymous_lob: wraps jaxob get_L2_state, converts to OMIE units,
    filters empty levels. Returns (bids_list, asks_list) of (price, qty) tuples.
  - get_agent_orders: reads state.asks/bids, filters by TID column, converts
    to OMIEOrder. Returns list of OMIEOrder.
  - get_agent_trades: wraps jaxob get_agent_trades, converts to OMIE units.
    Returns list of trade tuples (price, qty, timestamp) or similar.

These tests are black-box: they import only the public interface and
assert against domain-specified values. No implementation assumptions.

Four acceptance-criteria categories (expanded per-field):
  1. test_expire_at_gct_cancels_all       -- all orders removed, trades preserved
  2. test_anonymous_lob_hides_agent_ids   -- price/qty only, OMIE units, no agent IDs
  3. test_agent_sees_own_orders_only      -- get_agent_orders returns only that agent's
  4. test_agent_sees_own_trades_only      -- get_agent_trades returns only that agent's

Additional edge-case categories:
  5. expire_at_gct on empty book (idempotent)
  6. anonymous LOB ordering (best-to-worst)
  7. agent orders from both sides (buy + sell)
  8. agent trades as passive and aggressor
  9. empty results for agent with no orders/trades
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
# 1. TestExpireAtGCTCancelsAll
# ===========================================================================

class TestExpireAtGCTCancelsAll:
    """expire_at_gct(state) must cancel all remaining orders in the book.

    Domain rule (market-spec SS3.1): "Orders on a contract are automatically
    cancelled at GCT if unmatched."

    Architecture review: direct-clear approach — replace asks/bids with -1
    arrays, preserving trades and RNG key.
    """

    def test_expire_removes_buy_orders(self, book, empty_state):
        """After expiry, buy orders are gone from the book."""
        state, _ = book.submit_order(empty_state, _make_buy(order_id=1, price=45.00))
        state, _ = book.submit_order(state, _make_buy(order_id=3, price=44.00, timestamp=101))
        expired = book.expire_at_gct(state)
        assert _is_absent(book.get_order(expired, 1, Side.BUY))
        assert _is_absent(book.get_order(expired, 3, Side.BUY))

    def test_expire_removes_sell_orders(self, book, empty_state):
        """After expiry, sell orders are gone from the book."""
        state, _ = book.submit_order(empty_state, _make_sell(order_id=2, price=55.00))
        state, _ = book.submit_order(state, _make_sell(order_id=4, price=56.00, timestamp=201))
        expired = book.expire_at_gct(state)
        assert _is_absent(book.get_order(expired, 2, Side.SELL))
        assert _is_absent(book.get_order(expired, 4, Side.SELL))

    def test_expire_removes_orders_from_both_sides(self, book, empty_state):
        """Book with buys and sells: all gone after expiry."""
        state, _ = book.submit_order(empty_state, _make_buy(order_id=1, price=45.00))
        state, _ = book.submit_order(state, _make_sell(order_id=2, price=55.00))
        state, _ = book.submit_order(state, _make_buy(order_id=3, price=44.00, timestamp=101))
        state, _ = book.submit_order(state, _make_sell(order_id=4, price=56.00, timestamp=201))
        expired = book.expire_at_gct(state)
        for oid, side in [(1, Side.BUY), (2, Side.SELL), (3, Side.BUY), (4, Side.SELL)]:
            assert _is_absent(book.get_order(expired, oid, side))

    def test_expire_removes_orders_from_multiple_agents(self, book, empty_state):
        """Orders from different agents are all removed."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=0)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=3, price=44.00, agent_id=2, timestamp=101)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=55.00, agent_id=1)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=4, price=56.00, agent_id=3, timestamp=201)
        )
        expired = book.expire_at_gct(state)
        for oid, side in [(1, Side.BUY), (2, Side.SELL), (3, Side.BUY), (4, Side.SELL)]:
            assert _is_absent(book.get_order(expired, oid, side))

    def test_expire_preserves_trades(self, book, empty_state):
        """Trades executed before GCT are preserved after expiry.

        A trade occurs, then additional resting orders are submitted.
        After expiry the resting orders vanish but the trade remains.
        """
        # Resting sell, then aggressive buy -> trade at 50.00
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        state, result = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, timestamp=200)
        )
        assert result.traded == True
        # Add more resting orders after the trade
        state, _ = book.submit_order(
            state, _make_buy(order_id=5, price=48.00, timestamp=300)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=6, price=52.00, timestamp=301)
        )
        expired = book.expire_at_gct(state)
        # Resting orders are gone
        assert _is_absent(book.get_order(expired, 5, Side.BUY))
        assert _is_absent(book.get_order(expired, 6, Side.SELL))
        # Trades are preserved: agent 0 was involved in the trade
        agent_trades = book.get_agent_trades(expired, 0)
        assert len(agent_trades) > 0

    def test_expire_on_empty_book_succeeds(self, book, empty_state):
        """Expiring an empty book does not raise — returns valid state."""
        expired = book.expire_at_gct(empty_state)
        # Should still be a usable state; anonymous LOB should be empty
        bids, asks = book.get_anonymous_lob(expired)
        assert len(bids) == 0
        assert len(asks) == 0

    def test_expire_is_idempotent(self, book, empty_state):
        """Calling expire_at_gct twice does not raise."""
        state, _ = book.submit_order(empty_state, _make_buy(order_id=1, price=45.00))
        expired = book.expire_at_gct(state)
        double_expired = book.expire_at_gct(expired)
        assert _is_absent(book.get_order(double_expired, 1, Side.BUY))

    def test_expire_returns_lobstate(self, book, empty_state):
        """expire_at_gct returns a LobState (not None or other type)."""
        state, _ = book.submit_order(empty_state, _make_buy(order_id=1, price=45.00))
        expired = book.expire_at_gct(state)
        # The returned value must be usable as a state for further calls
        # (e.g., get_anonymous_lob should work on it)
        bids, asks = book.get_anonymous_lob(expired)
        assert isinstance(bids, list)
        assert isinstance(asks, list)


# ===========================================================================
# 2. TestAnonymousLOBHidesAgentIDs
# ===========================================================================

class TestAnonymousLOBHidesAgentIDs:
    """get_anonymous_lob(state) returns price/qty only, no agent IDs.

    Architecture review: returns (bids_list, asks_list) where each is a
    list of (price_eur, qty_mw) tuples in OMIE units, ordered best-to-worst.
    Empty levels excluded.
    """

    def test_anonymous_lob_returns_two_lists(self, book, empty_state):
        """get_anonymous_lob returns a (bids, asks) pair of lists."""
        state, _ = book.submit_order(empty_state, _make_buy(order_id=1, price=45.00))
        state, _ = book.submit_order(state, _make_sell(order_id=2, price=55.00))
        bids, asks = book.get_anonymous_lob(state)
        assert isinstance(bids, list)
        assert isinstance(asks, list)

    def test_anonymous_lob_bid_contains_price_and_qty(self, book, empty_state):
        """Each bid entry is a (price, quantity) tuple/pair."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, quantity=10.0)
        )
        bids, _ = book.get_anonymous_lob(state)
        assert len(bids) >= 1
        entry = bids[0]
        assert len(entry) == 2  # (price, qty)

    def test_anonymous_lob_ask_contains_price_and_qty(self, book, empty_state):
        """Each ask entry is a (price, quantity) tuple/pair."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=55.00, quantity=5.0)
        )
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) >= 1
        entry = asks[0]
        assert len(entry) == 2  # (price, qty)

    def test_anonymous_lob_no_agent_id_in_bid(self, book, empty_state):
        """Bid entries must not contain agent IDs.

        We submit from agent_id=100 and verify no entry in the returned
        bid data equals 100. (price, qty) tuples have exactly 2 elements.
        """
        state, _ = book.submit_order(
            empty_state,
            _make_buy(order_id=1, price=45.00, quantity=10.0, agent_id=100),
        )
        bids, _ = book.get_anonymous_lob(state)
        for entry in bids:
            assert len(entry) == 2, "bid entry should be (price, qty) only"

    def test_anonymous_lob_no_agent_id_in_ask(self, book, empty_state):
        """Ask entries must not contain agent IDs."""
        state, _ = book.submit_order(
            empty_state,
            _make_sell(order_id=2, price=55.00, quantity=5.0, agent_id=200),
        )
        _, asks = book.get_anonymous_lob(state)
        for entry in asks:
            assert len(entry) == 2, "ask entry should be (price, qty) only"

    def test_anonymous_lob_prices_in_omie_units(self, book, empty_state):
        """Prices are in EUR/MWh (OMIE units), not jaxob int representation."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.50, quantity=10.0)
        )
        bids, _ = book.get_anonymous_lob(state)
        assert len(bids) >= 1
        price, _ = bids[0]
        assert price == pytest.approx(45.50, abs=0.02)

    def test_anonymous_lob_quantities_in_omie_units(self, book, empty_state):
        """Quantities are in MW (OMIE units)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=55.00, quantity=7.5)
        )
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) >= 1
        _, qty = asks[0]
        assert qty == pytest.approx(7.5, abs=0.1)

    def test_anonymous_lob_bids_best_to_worst(self, book, empty_state):
        """Bids are ordered highest (best) price first."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=44.00, quantity=5.0)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=3, price=46.00, quantity=5.0, timestamp=101)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=5, price=45.00, quantity=5.0, timestamp=102)
        )
        bids, _ = book.get_anonymous_lob(state)
        bid_prices = [p for p, _ in bids]
        assert bid_prices == sorted(bid_prices, reverse=True)

    def test_anonymous_lob_asks_best_to_worst(self, book, empty_state):
        """Asks are ordered lowest (best) price first."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=57.00, quantity=5.0)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=4, price=55.00, quantity=5.0, timestamp=201)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=6, price=56.00, quantity=5.0, timestamp=202)
        )
        _, asks = book.get_anonymous_lob(state)
        ask_prices = [p for p, _ in asks]
        assert ask_prices == sorted(ask_prices)

    def test_anonymous_lob_empty_book(self, book, empty_state):
        """Empty book returns empty lists for both sides."""
        bids, asks = book.get_anonymous_lob(empty_state)
        assert len(bids) == 0
        assert len(asks) == 0

    def test_anonymous_lob_aggregates_same_price(self, book, empty_state):
        """Two orders at same price should be aggregated into one level.

        This tests the L2 aggregation behavior: multiple orders at the same
        price level appear as a single (price, total_qty) entry.
        """
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, quantity=5.0, agent_id=0)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=3, price=45.00, quantity=3.0, agent_id=2, timestamp=101)
        )
        bids, _ = book.get_anonymous_lob(state)
        # There should be exactly one price level at 45.00
        prices_at_45 = [(p, q) for p, q in bids if abs(p - 45.00) < 0.02]
        assert len(prices_at_45) == 1
        _, total_qty = prices_at_45[0]
        assert total_qty == pytest.approx(8.0, abs=0.1)

    def test_anonymous_lob_multiple_agents_same_level(self, book, empty_state):
        """Orders from different agents at same price: aggregated, no agent info.

        Agents 100 and 200 both have sells at 55.00. The anonymous LOB shows
        one level with combined quantity. Agent IDs are not exposed.
        """
        state, _ = book.submit_order(
            empty_state,
            _make_sell(order_id=2, price=55.00, quantity=4.0, agent_id=100),
        )
        state, _ = book.submit_order(
            state,
            _make_sell(order_id=4, price=55.00, quantity=6.0, agent_id=200, timestamp=201),
        )
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) >= 1
        entry = asks[0]
        assert len(entry) == 2  # only price and qty, no agent info
        price, qty = entry
        assert price == pytest.approx(55.00, abs=0.02)
        assert qty == pytest.approx(10.0, abs=0.1)

    def test_anonymous_lob_negative_price(self, book, empty_state):
        """Negative prices are returned correctly in OMIE units."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=-10.00, quantity=5.0)
        )
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) >= 1
        price, _ = asks[0]
        assert price == pytest.approx(-10.00, abs=0.02)


# ===========================================================================
# 3. TestAgentSeesOwnOrdersOnly
# ===========================================================================

class TestAgentSeesOwnOrdersOnly:
    """get_agent_orders(state, agent_id) returns only that agent's orders.

    Architecture review: reads state.asks/bids, filters by TID column
    (agent_id). Returns list of OMIEOrder. Asks mapped to Side.SELL,
    bids to Side.BUY.
    """

    def test_agent_sees_own_buy_order(self, book, empty_state):
        """Agent 0 submits a buy; get_agent_orders(0) returns it."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=0)
        )
        orders = book.get_agent_orders(state, 0)
        assert len(orders) == 1

    def test_agent_sees_own_sell_order(self, book, empty_state):
        """Agent 1 submits a sell; get_agent_orders(1) returns it."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=55.00, agent_id=1)
        )
        orders = book.get_agent_orders(state, 1)
        assert len(orders) == 1

    def test_agent_does_not_see_other_agents_orders(self, book, empty_state):
        """Agent 0 submits buy, agent 1 submits sell. Agent 0 sees only buy."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=0)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=55.00, agent_id=1)
        )
        orders_0 = book.get_agent_orders(state, 0)
        assert len(orders_0) == 1
        assert all(o.agent_id == 0 for o in orders_0)

    def test_other_agent_does_not_see_first_agents_orders(self, book, empty_state):
        """Agent 1 does not see agent 0's buy order."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=0)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=55.00, agent_id=1)
        )
        orders_1 = book.get_agent_orders(state, 1)
        assert len(orders_1) == 1
        assert all(o.agent_id == 1 for o in orders_1)

    def test_agent_sees_orders_on_both_sides(self, book, empty_state):
        """Agent 0 has a buy and a sell (non-crossing). Both returned."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=40.00, agent_id=0)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=60.00, agent_id=0, timestamp=200)
        )
        orders = book.get_agent_orders(state, 0)
        assert len(orders) == 2
        sides = {o.side for o in orders}
        assert Side.BUY in sides
        assert Side.SELL in sides

    def test_agent_order_correct_price(self, book, empty_state):
        """Returned order has correct price in OMIE units (EUR/MWh)."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.50, agent_id=0)
        )
        orders = book.get_agent_orders(state, 0)
        assert orders[0].price == pytest.approx(45.50, abs=0.005)

    def test_agent_order_correct_quantity(self, book, empty_state):
        """Returned order has correct quantity in OMIE units (MW)."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, quantity=7.5, agent_id=0)
        )
        orders = book.get_agent_orders(state, 0)
        assert orders[0].quantity == pytest.approx(7.5, abs=0.1)

    def test_agent_order_correct_order_id(self, book, empty_state):
        """Returned order has the submitted order_id."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=42, price=45.00, agent_id=0)
        )
        orders = book.get_agent_orders(state, 0)
        assert orders[0].order_id == 42

    def test_agent_order_correct_side_buy(self, book, empty_state):
        """Buy order returned with side=BUY."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=0)
        )
        orders = book.get_agent_orders(state, 0)
        assert orders[0].side == Side.BUY

    def test_agent_order_correct_side_sell(self, book, empty_state):
        """Sell order returned with side=SELL."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=55.00, agent_id=0, timestamp=200)
        )
        orders = book.get_agent_orders(state, 0)
        assert orders[0].side == Side.SELL

    def test_agent_with_no_orders_returns_empty(self, book, empty_state):
        """Agent 99 has no orders; get_agent_orders returns empty list."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=0)
        )
        orders = book.get_agent_orders(state, 99)
        assert len(orders) == 0

    def test_empty_book_returns_empty(self, book, empty_state):
        """Empty book: any agent gets empty list."""
        orders = book.get_agent_orders(empty_state, 0)
        assert len(orders) == 0

    def test_agent_multiple_orders(self, book, empty_state):
        """Agent with 3 orders sees all 3."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=44.00, agent_id=0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=3, price=45.00, agent_id=0, timestamp=101)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=5, price=46.00, agent_id=0, timestamp=102)
        )
        orders = book.get_agent_orders(state, 0)
        assert len(orders) == 3

    def test_three_agents_isolation(self, book, empty_state):
        """Three agents: each sees only their own orders."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=44.00, agent_id=0, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_sell(order_id=2, price=56.00, agent_id=1, timestamp=200)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=3, price=43.00, agent_id=2, timestamp=101)
        )
        for aid, expected_count in [(0, 1), (1, 1), (2, 1)]:
            orders = book.get_agent_orders(state, aid)
            assert len(orders) == expected_count
            assert all(o.agent_id == aid for o in orders)

    def test_agent_orders_after_cancel(self, book, empty_state):
        """After cancelling an order, agent no longer sees it."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=0)
        )
        state = book.cancel_order(state, 1, Side.BUY)
        orders = book.get_agent_orders(state, 0)
        assert len(orders) == 0

    def test_agent_orders_after_expire(self, book, empty_state):
        """After expire_at_gct, agent sees no orders."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=0)
        )
        expired = book.expire_at_gct(state)
        orders = book.get_agent_orders(expired, 0)
        assert len(orders) == 0


# ===========================================================================
# 4. TestAgentSeesOwnTradesOnly
# ===========================================================================

class TestAgentSeesOwnTradesOnly:
    """get_agent_trades(state, agent_id) returns only that agent's trades.

    Architecture review: wraps jaxob get_agent_trades which filters by
    PASS_TID or AGRS_TID. Returns list of trade tuples/entries in OMIE units.
    Agent may be passive (resting) or aggressor (incoming).
    """

    def test_agent_sees_own_trade_as_aggressor(self, book, empty_state):
        """Agent 0 is aggressor (incoming buy matches resting sell).

        Agent 0 must see this trade.
        """
        # Agent 1 places resting sell
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, agent_id=1, timestamp=100)
        )
        # Agent 0 places aggressive buy (crosses)
        state, result = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, agent_id=0, timestamp=200)
        )
        assert result.traded == True
        trades = book.get_agent_trades(state, 0)
        assert len(trades) > 0

    def test_agent_sees_own_trade_as_passive(self, book, empty_state):
        """Agent 1 is passive (resting sell matched by incoming buy).

        Agent 1 must see this trade.
        """
        # Agent 1 places resting sell
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, agent_id=1, timestamp=100)
        )
        # Agent 0 places aggressive buy (crosses)
        state, result = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, agent_id=0, timestamp=200)
        )
        assert result.traded == True
        trades = book.get_agent_trades(state, 1)
        assert len(trades) > 0

    def test_agent_does_not_see_other_agents_trade(self, book, empty_state):
        """Agent 2 is uninvolved; get_agent_trades(2) returns empty."""
        # Agent 1 sell, agent 0 buy -> trade
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, agent_id=1, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, agent_id=0, timestamp=200)
        )
        trades = book.get_agent_trades(state, 2)
        assert len(trades) == 0

    def test_agent_trade_price_in_omie_units(self, book, empty_state):
        """Trade price is in EUR/MWh (OMIE units)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.50, quantity=5.0, agent_id=1, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.50, quantity=5.0, agent_id=0, timestamp=200)
        )
        trades = book.get_agent_trades(state, 0)
        assert len(trades) > 0
        # Trade entry should have a price field approximately 50.50
        trade = trades[0]
        # The trade is a tuple or similar with price as first element
        price = trade[0]
        assert price == pytest.approx(50.50, abs=0.02)

    def test_agent_trade_quantity_in_omie_units(self, book, empty_state):
        """Trade quantity is in MW (OMIE units), positive."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=7.5, agent_id=1, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=7.5, agent_id=0, timestamp=200)
        )
        trades = book.get_agent_trades(state, 0)
        assert len(trades) > 0
        trade = trades[0]
        qty = trade[1]
        assert qty == pytest.approx(7.5, abs=0.1)
        assert qty > 0  # OMIE-facing quantities are positive (abs of jaxob sign)

    def test_agent_no_trades_returns_empty(self, book, empty_state):
        """Agent with no trades gets empty list."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=0)
        )
        trades = book.get_agent_trades(state, 0)
        assert len(trades) == 0

    def test_empty_book_no_trades(self, book, empty_state):
        """Empty book: any agent gets empty trade list."""
        trades = book.get_agent_trades(empty_state, 0)
        assert len(trades) == 0

    def test_agent_sees_multiple_trades(self, book, empty_state):
        """Agent involved in two separate trades sees both."""
        # Trade 1: agent 0 buys from agent 1
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, agent_id=1, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, agent_id=0, timestamp=200)
        )
        # Trade 2: agent 0 buys from agent 2
        state, _ = book.submit_order(
            state, _make_sell(order_id=4, price=51.00, quantity=3.0, agent_id=2, timestamp=300)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=3, price=51.00, quantity=3.0, agent_id=0, timestamp=400)
        )
        trades = book.get_agent_trades(state, 0)
        assert len(trades) >= 2

    def test_agent_trades_preserved_after_expire(self, book, empty_state):
        """After expire_at_gct, agent still sees their trades."""
        # Create a trade
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, agent_id=1, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, agent_id=0, timestamp=200)
        )
        # Add resting orders then expire
        state, _ = book.submit_order(
            state, _make_buy(order_id=5, price=48.00, agent_id=0, timestamp=300)
        )
        expired = book.expire_at_gct(state)
        # Orders are gone
        assert _is_absent(book.get_order(expired, 5, Side.BUY))
        # But trades persist
        trades_0 = book.get_agent_trades(expired, 0)
        assert len(trades_0) > 0
        trades_1 = book.get_agent_trades(expired, 1)
        assert len(trades_1) > 0

    def test_both_agents_see_same_trade(self, book, empty_state):
        """When agent 0 and agent 1 trade, both see the trade."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, agent_id=1, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, agent_id=0, timestamp=200)
        )
        trades_0 = book.get_agent_trades(state, 0)
        trades_1 = book.get_agent_trades(state, 1)
        assert len(trades_0) > 0
        assert len(trades_1) > 0

    def test_agent_trade_quantity_is_positive(self, book, empty_state):
        """Trade quantities in OMIE are always positive (absolute value).

        jaxob uses signed quantities internally (negative for buy-side match).
        The OMIE-facing method must use abs().
        """
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, agent_id=1, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, agent_id=0, timestamp=200)
        )
        for aid in [0, 1]:
            trades = book.get_agent_trades(state, aid)
            for trade in trades:
                qty = trade[1]
                assert qty > 0, f"Trade qty for agent {aid} should be positive"


# ===========================================================================
# 5. TestAnonymousLOBAfterTrade
# ===========================================================================

class TestAnonymousLOBAfterTrade:
    """Anonymous LOB reflects post-trade state correctly."""

    def test_fully_matched_orders_not_in_lob(self, book, empty_state):
        """After a full match, the consumed orders vanish from the LOB."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, agent_id=1, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, agent_id=0, timestamp=200)
        )
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0

    def test_partial_fill_residual_in_lob(self, book, empty_state):
        """Resting sell 10 MW, buy 5 MW -> residual 5 MW sell in LOB."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, agent_id=1, timestamp=100)
        )
        state, _ = book.submit_order(
            state, _make_buy(order_id=1, price=50.00, quantity=5.0, agent_id=0, timestamp=200)
        )
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0  # buy fully consumed
        assert len(asks) >= 1  # residual sell
        _, qty = asks[0]
        assert qty == pytest.approx(5.0, abs=0.1)


# ===========================================================================
# 6. TestGetAgentOrdersReturnType
# ===========================================================================

class TestGetAgentOrdersReturnType:
    """get_agent_orders returns OMIEOrder instances with all expected fields."""

    def test_returns_omie_order_instances(self, book, empty_state):
        """Each element in the returned list is an OMIEOrder."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=0)
        )
        orders = book.get_agent_orders(state, 0)
        assert len(orders) == 1
        order = orders[0]
        # OMIEOrder is a NamedTuple — check it has the expected fields
        assert hasattr(order, 'price')
        assert hasattr(order, 'quantity')
        assert hasattr(order, 'order_id')
        assert hasattr(order, 'agent_id')
        assert hasattr(order, 'side')

    def test_agent_id_matches(self, book, empty_state):
        """Returned order's agent_id matches the queried agent."""
        state, _ = book.submit_order(
            empty_state, _make_buy(order_id=1, price=45.00, agent_id=7)
        )
        orders = book.get_agent_orders(state, 7)
        assert len(orders) == 1
        assert orders[0].agent_id == 7
