"""Audit tests for M3.4 -- GCT Expiry + Views.

Post-implementation verification focusing on edge cases, boundary conditions,
and acceptance criteria that stress the four new ContractBook methods:

- expire_at_gct: partial-book expiry (only one side populated), expiry after
  trades at multiple price levels, LobState field preservation (key, trades),
  functional semantics (original state not mutated)
- get_anonymous_lob: multi-level ordering verification, spread gap correctness,
  post-expiry returns empty, negative-price book, single-sided book
- get_agent_orders: mixed agents with trades (filled orders should not appear),
  order field fidelity after partial fill, agent with only sell orders
- get_agent_trades: multi-fill trades (walk across levels), trade after IOC,
  quantity always positive regardless of aggressor/passive role, settlement
  consistency between SubmitResult and get_agent_trades

Derived independently from:
- docs/omiecid/market-spec.md §3.1 (GCT), §7.1 (anonymous LOB), §7.2 (trade
  notification), §8.4 (no position close-out at GCT)
- docs/omiecid/agile-plan.md (M3.4 acceptance criteria)
- docs/omiecid/migration-plan.md §M3 (interface)
- omiecid_simulator/omiecid/contract_book.py (public interface only)
"""

import pytest
from pytest import approx

from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.order_types import (
    OMIEOrder,
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




# ---------------------------------------------------------------------------
# Shared fixtures (product_config, market_config from conftest)
# Specialized open_contract overrides conftest (contract_id=55 vs 37)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def open_contract():
    """An OPEN contract suitable for GCT/views tests."""
    return Contract(
        contract_id=55,
        delivery_start_minutes=720,   # 12:00
        delivery_end_minutes=735,     # 12:15
        gct_minutes=660,              # 11:00
        status=ContractStatus.OPEN,
    )


@pytest.fixture
def book(open_contract, product_config, market_config):
    return ContractBook(open_contract, product_config, market_config)


@pytest.fixture
def empty_state(book):
    return book.init()


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _buy(contract_id=55, price=50.0, qty=1.0, order_id=1, agent_id=100,
         timestamp=1000, restriction=ExecRestriction.NON):
    return make_order(
        contract_id=contract_id, side=Side.BUY, price=price, quantity=qty,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=restriction,
    )


def _sell(contract_id=55, price=50.0, qty=1.0, order_id=2, agent_id=200,
          timestamp=1000, restriction=ExecRestriction.NON):
    return make_order(
        contract_id=contract_id, side=Side.SELL, price=price, quantity=qty,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=restriction,
    )


# ===========================================================================
# expire_at_gct — edge cases
# ===========================================================================

class TestExpireAtGCTEdgeCases:
    """Edge cases for expire_at_gct beyond the basic both-sides-cleared test."""

    def test_expire_only_bids_populated(self, book, empty_state):
        """Expiry works when only bids exist (no asks)."""
        state, _ = book.submit_order(empty_state, _buy(price=40.0, order_id=1))
        state, _ = book.submit_order(state, _buy(price=41.0, order_id=2))

        expired = book.expire_at_gct(state)

        # No orders on either side
        bids, asks = book.get_anonymous_lob(expired)
        assert bids == []
        assert asks == []

    def test_expire_only_asks_populated(self, book, empty_state):
        """Expiry works when only asks exist (no bids)."""
        state, _ = book.submit_order(empty_state, _sell(price=60.0, order_id=1))
        state, _ = book.submit_order(state, _sell(price=61.0, order_id=2))

        expired = book.expire_at_gct(state)

        bids, asks = book.get_anonymous_lob(expired)
        assert bids == []
        assert asks == []

    def test_expire_preserves_trades_after_multi_fill(self, book, empty_state):
        """Trades from multiple fills are preserved after GCT expiry."""
        # Create two resting sells at different prices
        state, _ = book.submit_order(empty_state, _sell(price=50.0, order_id=1, agent_id=200))
        state, _ = book.submit_order(state, _sell(price=51.0, order_id=2, agent_id=201))
        # Aggressive buy walks both levels
        state, r = book.submit_order(state, _buy(price=51.0, qty=2.0, order_id=3, agent_id=100))
        assert r.traded

        expired = book.expire_at_gct(state)

        # Agent 100 should still see trades after expiry
        trades = book.get_agent_trades(expired, 100)
        assert len(trades) >= 1

    def test_expire_does_not_mutate_original_state(self, book, empty_state):
        """Functional semantics: original state unchanged after expire_at_gct."""
        state, _ = book.submit_order(empty_state, _buy(price=45.0, order_id=1))
        state, _ = book.submit_order(state, _sell(price=55.0, order_id=2))

        # Snapshot: orders exist
        order_before = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(order_before)

        _ = book.expire_at_gct(state)

        # Original state still has the order
        order_after = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(order_after)

    def test_expire_twice_is_idempotent(self, book, empty_state):
        """Double expiry on an already-expired book produces same result."""
        state, _ = book.submit_order(empty_state, _buy(price=45.0, order_id=1))
        expired_once = book.expire_at_gct(state)
        expired_twice = book.expire_at_gct(expired_once)

        bids1, asks1 = book.get_anonymous_lob(expired_once)
        bids2, asks2 = book.get_anonymous_lob(expired_twice)
        assert bids1 == bids2 == []
        assert asks1 == asks2 == []

    def test_expire_preserves_lob_state_key(self, book, empty_state):
        """The RNG key in LobState is preserved through expiry."""
        state, _ = book.submit_order(empty_state, _buy(price=45.0, order_id=1))
        expired = book.expire_at_gct(state)
        # LobState has a `key` field that must survive (accessed via .lob)
        assert expired.lob.key is not None


# ===========================================================================
# get_anonymous_lob — edge cases
# ===========================================================================

class TestAnonymousLOBEdgeCases:
    """Edge cases for the anonymous LOB view."""

    def test_lob_multi_level_bid_ordering(self, book, empty_state):
        """Bids are returned highest-price first (best to worst)."""
        state, _ = book.submit_order(empty_state, _buy(price=40.0, order_id=1))
        state, _ = book.submit_order(state, _buy(price=42.0, order_id=2))
        state, _ = book.submit_order(state, _buy(price=41.0, order_id=3))

        bids, _ = book.get_anonymous_lob(state)
        bid_prices = [b[0] for b in bids]
        assert bid_prices == sorted(bid_prices, reverse=True)
        assert bid_prices[0] == approx(42.0, abs=0.01)

    def test_lob_multi_level_ask_ordering(self, book, empty_state):
        """Asks are returned lowest-price first (best to worst)."""
        state, _ = book.submit_order(empty_state, _sell(price=62.0, order_id=1))
        state, _ = book.submit_order(state, _sell(price=60.0, order_id=2))
        state, _ = book.submit_order(state, _sell(price=61.0, order_id=3))

        _, asks = book.get_anonymous_lob(state)
        ask_prices = [a[0] for a in asks]
        assert ask_prices == sorted(ask_prices)
        assert ask_prices[0] == approx(60.0, abs=0.01)

    def test_lob_spread_gap_correct(self, book, empty_state):
        """Best bid < best ask when spread exists."""
        state, _ = book.submit_order(empty_state, _buy(price=49.0, order_id=1))
        state, _ = book.submit_order(state, _sell(price=51.0, order_id=2))

        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) >= 1
        assert len(asks) >= 1
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        assert best_bid < best_ask

    def test_lob_post_expiry_empty(self, book, empty_state):
        """Anonymous LOB returns empty lists after GCT expiry."""
        state, _ = book.submit_order(empty_state, _buy(price=49.0, order_id=1))
        state, _ = book.submit_order(state, _sell(price=51.0, order_id=2))
        expired = book.expire_at_gct(state)

        bids, asks = book.get_anonymous_lob(expired)
        assert bids == []
        assert asks == []

    def test_lob_negative_prices(self, book, empty_state):
        """LOB correctly shows negative prices in OMIE units."""
        state, _ = book.submit_order(empty_state, _buy(price=-10.0, order_id=1))
        state, _ = book.submit_order(state, _sell(price=-5.0, order_id=2))

        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) >= 1
        assert len(asks) >= 1
        assert bids[0][0] == approx(-10.0, abs=0.01)
        assert asks[0][0] == approx(-5.0, abs=0.01)

    def test_lob_single_side_only_bids(self, book, empty_state):
        """LOB works with only bids (asks list is empty)."""
        state, _ = book.submit_order(empty_state, _buy(price=45.0, order_id=1))

        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) >= 1
        assert asks == []

    def test_lob_single_side_only_asks(self, book, empty_state):
        """LOB works with only asks (bids list is empty)."""
        state, _ = book.submit_order(empty_state, _sell(price=55.0, order_id=1))

        bids, asks = book.get_anonymous_lob(state)
        assert bids == []
        assert len(asks) >= 1

    def test_lob_tuples_are_two_element(self, book, empty_state):
        """Each LOB entry is exactly (price, qty) — no extra fields."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, qty=2.0, order_id=1))

        bids, _ = book.get_anonymous_lob(state)
        assert len(bids) >= 1
        assert len(bids[0]) == 2

    def test_lob_quantities_in_mw(self, book, empty_state):
        """LOB quantities are in MW, not jaxob internal units."""
        state, _ = book.submit_order(empty_state, _sell(price=55.0, qty=3.5, order_id=1))

        _, asks = book.get_anonymous_lob(state)
        assert len(asks) >= 1
        assert asks[0][1] == approx(3.5, abs=0.1)

    def test_lob_after_partial_fill_shows_residual(self, book, empty_state):
        """After a partial fill, the residual quantity is visible in the LOB."""
        # Resting sell of 5.0 MW
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=5.0, order_id=1, agent_id=200))
        # Aggressive buy of 2.0 MW fills partially
        state, r = book.submit_order(state, _buy(price=50.0, qty=2.0, order_id=2, agent_id=100))
        assert r.traded

        _, asks = book.get_anonymous_lob(state)
        assert len(asks) >= 1
        # Residual should be 3.0 MW
        assert asks[0][1] == approx(3.0, abs=0.1)

    def test_lob_does_not_contain_agent_id(self, book, empty_state):
        """The LOB return structure cannot leak agent IDs.

        Each tuple is (price, qty) only — verifying the structure does not
        grow to include identity information.
        """
        state, _ = book.submit_order(empty_state, _buy(price=50.0, order_id=1, agent_id=999))
        state, _ = book.submit_order(state, _sell(price=55.0, order_id=2, agent_id=888))

        bids, asks = book.get_anonymous_lob(state)
        for entry in bids + asks:
            assert len(entry) == 2, "LOB entry must be (price, qty) only"


# ===========================================================================
# get_agent_orders — edge cases
# ===========================================================================

class TestGetAgentOrdersEdgeCases:
    """Edge cases for per-agent order view."""

    def test_agent_sees_only_sell_orders(self, book, empty_state):
        """Agent with only sell orders gets them back correctly."""
        state, _ = book.submit_order(empty_state, _sell(price=55.0, order_id=1, agent_id=300))
        state, _ = book.submit_order(state, _sell(price=56.0, order_id=2, agent_id=300))

        orders = book.get_agent_orders(state, 300)
        assert len(orders) == 2
        for o in orders:
            assert o.side == Side.SELL
            assert o.agent_id == 300

    def test_agent_orders_after_full_fill_empty(self, book, empty_state):
        """Fully filled orders should NOT appear in get_agent_orders."""
        # Agent 100 places a buy, agent 200 fills it completely
        state, _ = book.submit_order(empty_state, _buy(price=50.0, qty=1.0, order_id=1, agent_id=100))
        state, r = book.submit_order(state, _sell(price=50.0, qty=1.0, order_id=2, agent_id=200))
        assert r.traded

        # Agent 100's buy was fully consumed
        orders_100 = book.get_agent_orders(state, 100)
        assert len(orders_100) == 0

    def test_agent_orders_after_partial_fill_shows_residual(self, book, empty_state):
        """Partially filled order shows in get_agent_orders with reduced qty."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=5.0, order_id=1, agent_id=200))
        state, r = book.submit_order(state, _buy(price=50.0, qty=2.0, order_id=2, agent_id=100))
        assert r.traded

        orders_200 = book.get_agent_orders(state, 200)
        assert len(orders_200) == 1
        # Residual is 3.0 MW
        assert orders_200[0].quantity == approx(3.0, abs=0.1)

    def test_three_agents_isolated(self, book, empty_state):
        """Three agents each see only their own orders."""
        state, _ = book.submit_order(empty_state, _buy(price=40.0, order_id=1, agent_id=100))
        state, _ = book.submit_order(state, _buy(price=41.0, order_id=2, agent_id=200))
        state, _ = book.submit_order(state, _sell(price=60.0, order_id=3, agent_id=300))

        assert len(book.get_agent_orders(state, 100)) == 1
        assert len(book.get_agent_orders(state, 200)) == 1
        assert len(book.get_agent_orders(state, 300)) == 1

        # Each agent sees only their own order_id
        assert book.get_agent_orders(state, 100)[0].order_id == 1
        assert book.get_agent_orders(state, 200)[0].order_id == 2
        assert book.get_agent_orders(state, 300)[0].order_id == 3

    def test_agent_orders_returns_omie_order_instances(self, book, empty_state):
        """Return type is list of OMIEOrder (not raw arrays or tuples)."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, order_id=1, agent_id=100))

        orders = book.get_agent_orders(state, 100)
        assert len(orders) == 1
        assert isinstance(orders[0], OMIEOrder)

    def test_agent_orders_price_in_eur(self, book, empty_state):
        """Order prices are returned in EUR/MWh, not jaxob internal units."""
        state, _ = book.submit_order(empty_state, _buy(price=73.42, order_id=1, agent_id=100))

        orders = book.get_agent_orders(state, 100)
        assert orders[0].price == approx(73.42, abs=0.01)

    def test_agent_orders_quantity_in_mw(self, book, empty_state):
        """Order quantities are returned in MW, not jaxob internal units."""
        state, _ = book.submit_order(empty_state, _sell(price=60.0, qty=7.3, order_id=1, agent_id=100))

        orders = book.get_agent_orders(state, 100)
        assert orders[0].quantity == approx(7.3, abs=0.1)

    def test_agent_orders_empty_for_nonexistent_agent(self, book, empty_state):
        """Querying an agent with no orders returns empty list."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, order_id=1, agent_id=100))

        orders = book.get_agent_orders(state, 999)
        assert orders == []

    def test_agent_orders_after_cancel(self, book, empty_state):
        """Cancelled order does not appear in get_agent_orders."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, order_id=1, agent_id=100))
        state = book.cancel_order(state, 1, Side.BUY)

        orders = book.get_agent_orders(state, 100)
        assert orders == []

    def test_agent_orders_after_expire(self, book, empty_state):
        """After GCT expiry, no agent sees any orders."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, order_id=1, agent_id=100))
        state, _ = book.submit_order(state, _sell(price=60.0, order_id=2, agent_id=200))
        expired = book.expire_at_gct(state)

        assert book.get_agent_orders(expired, 100) == []
        assert book.get_agent_orders(expired, 200) == []

    def test_agent_orders_contract_id_correct(self, book, empty_state):
        """Orders returned include correct contract_id (55)."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, order_id=1, agent_id=100))

        orders = book.get_agent_orders(state, 100)
        assert orders[0].contract_id == 55

    def test_agent_orders_both_sides(self, book, empty_state):
        """Agent with orders on both sides sees all of them."""
        state, _ = book.submit_order(empty_state, _buy(price=40.0, order_id=1, agent_id=100))
        state, _ = book.submit_order(state, _sell(price=60.0, order_id=2, agent_id=100))

        orders = book.get_agent_orders(state, 100)
        assert len(orders) == 2
        sides = {o.side for o in orders}
        assert Side.BUY in sides
        assert Side.SELL in sides


# ===========================================================================
# get_agent_trades — edge cases
# ===========================================================================

class TestGetAgentTradesEdgeCases:
    """Edge cases for per-agent trade view."""

    def test_aggressor_sees_trade(self, book, empty_state):
        """The aggressor (taker) sees the trade in their trade list."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, order_id=1, agent_id=200))
        state, r = book.submit_order(state, _buy(price=50.0, order_id=2, agent_id=100))
        assert r.traded

        trades = book.get_agent_trades(state, 100)
        assert len(trades) >= 1

    def test_passive_sees_trade(self, book, empty_state):
        """The passive (maker) side sees the trade in their trade list."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, order_id=1, agent_id=200))
        state, r = book.submit_order(state, _buy(price=50.0, order_id=2, agent_id=100))
        assert r.traded

        trades = book.get_agent_trades(state, 200)
        assert len(trades) >= 1

    def test_uninvolved_agent_no_trades(self, book, empty_state):
        """Agent not involved in any trade gets empty list."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, order_id=1, agent_id=200))
        state, _ = book.submit_order(state, _buy(price=50.0, order_id=2, agent_id=100))

        trades = book.get_agent_trades(state, 999)
        assert trades == []

    def test_trade_price_in_eur(self, book, empty_state):
        """Trade prices are in EUR/MWh, not jaxob internal units."""
        state, _ = book.submit_order(empty_state, _sell(price=73.50, order_id=1, agent_id=200))
        state, r = book.submit_order(state, _buy(price=73.50, order_id=2, agent_id=100))
        assert r.traded

        trades = book.get_agent_trades(state, 100)
        assert len(trades) >= 1
        price = trades[0][0]
        assert price == approx(73.50, abs=0.01)

    def test_trade_quantity_positive(self, book, empty_state):
        """Trade quantities are always positive regardless of side."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=2.5, order_id=1, agent_id=200))
        state, r = book.submit_order(state, _buy(price=50.0, qty=2.5, order_id=2, agent_id=100))
        assert r.traded

        # Both sides should see positive quantity
        for agent_id in [100, 200]:
            trades = book.get_agent_trades(state, agent_id)
            assert len(trades) >= 1
            qty = trades[0][1]
            assert qty > 0, f"Agent {agent_id} trade qty must be positive"

    def test_trade_quantity_in_mw(self, book, empty_state):
        """Trade quantities are in MW, not jaxob internal units."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=3.7, order_id=1, agent_id=200))
        state, r = book.submit_order(state, _buy(price=50.0, qty=3.7, order_id=2, agent_id=100))
        assert r.traded

        trades = book.get_agent_trades(state, 100)
        qty = trades[0][1]
        assert qty == approx(3.7, abs=0.1)

    def test_trades_preserved_after_expire(self, book, empty_state):
        """Trades survive GCT expiry (market-spec §8.4: matched trades are firm)."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, order_id=1, agent_id=200))
        state, _ = book.submit_order(state, _buy(price=50.0, order_id=2, agent_id=100))
        expired = book.expire_at_gct(state)

        trades_100 = book.get_agent_trades(expired, 100)
        trades_200 = book.get_agent_trades(expired, 200)
        assert len(trades_100) >= 1
        assert len(trades_200) >= 1

    def test_multiple_trades_visible(self, book, empty_state):
        """Agent involved in multiple trades sees all of them."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, agent_id=200))
        state, _ = book.submit_order(state, _sell(price=51.0, qty=1.0, order_id=2, agent_id=201))
        # Aggressive buy walks both levels
        state, r = book.submit_order(state, _buy(price=51.0, qty=2.0, order_id=3, agent_id=100))
        assert r.traded

        trades = book.get_agent_trades(state, 100)
        assert len(trades) >= 2, "Should see trades at both price levels"

    def test_trade_with_negative_price(self, book, empty_state):
        """Trades at negative prices are reported correctly."""
        state, _ = book.submit_order(empty_state, _sell(price=-5.0, order_id=1, agent_id=200))
        state, r = book.submit_order(state, _buy(price=-5.0, order_id=2, agent_id=100))
        assert r.traded

        trades = book.get_agent_trades(state, 100)
        assert len(trades) >= 1
        price = trades[0][0]
        assert price == approx(-5.0, abs=0.01)

    def test_no_trades_on_empty_book(self, book, empty_state):
        """Empty book returns no trades for any agent."""
        trades = book.get_agent_trades(empty_state, 100)
        assert trades == []

    def test_ioc_trade_visible(self, book, empty_state):
        """Trade from an IOC order is visible to both parties."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, agent_id=200))
        state, r = book.submit_order(
            state,
            _buy(price=50.0, qty=2.0, order_id=2, agent_id=100,
                 restriction=ExecRestriction.IOC),
        )
        assert r.traded

        trades_100 = book.get_agent_trades(state, 100)
        trades_200 = book.get_agent_trades(state, 200)
        assert len(trades_100) >= 1
        assert len(trades_200) >= 1
        # Traded qty should be 1.0 MW (limited by resting liquidity)
        assert trades_100[0][1] == approx(1.0, abs=0.1)

    def test_both_agents_see_same_trade(self, book, empty_state):
        """Both sides of a trade see the same price and quantity."""
        state, _ = book.submit_order(empty_state, _sell(price=65.0, qty=4.0, order_id=1, agent_id=200))
        state, r = book.submit_order(state, _buy(price=65.0, qty=4.0, order_id=2, agent_id=100))
        assert r.traded

        t100 = book.get_agent_trades(state, 100)
        t200 = book.get_agent_trades(state, 200)
        assert len(t100) >= 1
        assert len(t200) >= 1
        # Same price and qty
        assert t100[0][0] == approx(t200[0][0], abs=0.01)
        assert t100[0][1] == approx(t200[0][1], abs=0.1)

    def test_trade_tuple_has_at_least_three_elements(self, book, empty_state):
        """Trade return is (price, qty, timestamp) — at least 3 elements."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, order_id=1, agent_id=200))
        state, _ = book.submit_order(state, _buy(price=50.0, order_id=2, agent_id=100))

        trades = book.get_agent_trades(state, 100)
        assert len(trades) >= 1
        assert len(trades[0]) >= 3, "Trade tuple should have (price, qty, timestamp)"


# ===========================================================================
# Cross-feature: expire + views interaction
# ===========================================================================

class TestExpireViewsInteraction:
    """Tests combining expire_at_gct with view methods."""

    def test_expire_clears_orders_but_not_trades(self, book, empty_state):
        """After expiry: orders gone, trades preserved."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, agent_id=200))
        # Partial fill: 0.5 MW filled, 0.5 MW resting
        state, r = book.submit_order(state, _buy(price=50.0, qty=0.5, order_id=2, agent_id=100))
        assert r.traded

        # Before expire: agent 200 has resting order + trade
        assert len(book.get_agent_orders(state, 200)) == 1
        assert len(book.get_agent_trades(state, 200)) >= 1

        expired = book.expire_at_gct(state)

        # After expire: no orders, but trade still visible
        assert book.get_agent_orders(expired, 200) == []
        assert len(book.get_agent_trades(expired, 200)) >= 1

    def test_lob_empty_after_expire_with_mixed_agents(self, book, empty_state):
        """LOB is empty after expiry even with orders from multiple agents."""
        state, _ = book.submit_order(empty_state, _buy(price=40.0, order_id=1, agent_id=100))
        state, _ = book.submit_order(state, _buy(price=41.0, order_id=2, agent_id=200))
        state, _ = book.submit_order(state, _sell(price=60.0, order_id=3, agent_id=300))
        state, _ = book.submit_order(state, _sell(price=61.0, order_id=4, agent_id=400))

        expired = book.expire_at_gct(state)

        bids, asks = book.get_anonymous_lob(expired)
        assert bids == []
        assert asks == []

    def test_agent_orders_empty_after_expire_all_agents(self, book, empty_state):
        """Every agent sees empty orders after expiry."""
        state, _ = book.submit_order(empty_state, _buy(price=40.0, order_id=1, agent_id=100))
        state, _ = book.submit_order(state, _sell(price=60.0, order_id=2, agent_id=200))

        expired = book.expire_at_gct(state)

        for aid in [100, 200, 999]:
            assert book.get_agent_orders(expired, aid) == []
