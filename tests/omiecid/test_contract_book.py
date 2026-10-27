"""Tests for M3.1 -- ContractBook: Submit + Cancel.

Behavioral contracts for ContractBook wrapping jaxob.OrderBook.
All expected values are derived from docs/omiecid/market-spec.md SS2.3, SS4,
docs/omiecid/migration-plan.md SS M3, docs/omiecid/architecture-review.md,
and docs/omiecid/m3-api-mapping.md.

These tests are black-box: they import only the public interface and
assert against domain-specified values. No implementation assumptions.

Four acceptance-criteria categories (expanded per-field):
  1. test_submit_buy_limit_order_enters_book  -- buy orders enter bid side
  2. test_submit_sell_limit_order_enters_book -- sell orders enter ask side
  3. test_cancel_order                        -- cancel removes from book
  4. test_modify_order_resets_timestamp        -- modify = cancel + resubmit

Plus edge-case classes for validation rejection, positive-ID enforcement,
missing-order cancel, SubmitResult contract, and get_order helper.
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
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig
from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult

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
# Fixtures (product_config, market_config, open_contract, book, empty_state from conftest)
# ---------------------------------------------------------------------------

@pytest.fixture
def closed_contract():
    """A CLOSED contract."""
    return Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.CLOSED,
    )


@pytest.fixture
def buy_order():
    """A valid NON buy limit order at 45.50 EUR/MWh, 10.0 MW."""
    return make_order(
        contract_id=37,
        side=Side.BUY,
        price=45.50,
        quantity=10.0,
        order_id=1,
        agent_id=0,
        timestamp=100,
    )


@pytest.fixture
def sell_order():
    """A valid NON sell limit order at 55.00 EUR/MWh, 5.0 MW."""
    return make_order(
        contract_id=37,
        side=Side.SELL,
        price=55.00,
        quantity=5.0,
        order_id=2,
        agent_id=1,
        timestamp=200,
    )


# ---------------------------------------------------------------------------
# Helper: make a buy or sell order with custom fields
# ---------------------------------------------------------------------------

def _make_buy(order_id=1, price=45.50, quantity=10.0, agent_id=0, timestamp=100):
    return make_order(
        contract_id=37, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
    )


def _make_sell(order_id=2, price=55.00, quantity=5.0, agent_id=1, timestamp=200):
    return make_order(
        contract_id=37, side=Side.SELL, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
    )


# ===========================================================================
# 1. TestSubmitBuyLimitOrderEntersBook
# ===========================================================================

class TestSubmitBuyLimitOrderEntersBook:
    """A valid buy limit order must enter the bid side of the book."""

    def test_submit_buy_returns_accepted_true(self, book, empty_state, buy_order):
        """submit_order for a valid buy returns accepted=True."""
        _state, result = book.submit_order(empty_state, buy_order)
        assert result.accepted == True

    def test_submit_buy_order_retrievable(self, book, empty_state, buy_order):
        """After submit, get_order returns the order on the BUY side."""
        state, _ = book.submit_order(empty_state, buy_order)
        retrieved = book.get_order(state, buy_order.order_id, Side.BUY)
        assert not _is_absent(retrieved)

    def test_submit_buy_order_correct_price(self, book, empty_state, buy_order):
        """Retrieved buy order has the submitted price (45.50 EUR/MWh)."""
        state, _ = book.submit_order(empty_state, buy_order)
        retrieved = book.get_order(state, buy_order.order_id, Side.BUY)
        assert retrieved.price == pytest.approx(45.50, abs=0.005)

    def test_submit_buy_order_correct_quantity(self, book, empty_state, buy_order):
        """Retrieved buy order has the submitted quantity (10.0 MW)."""
        state, _ = book.submit_order(empty_state, buy_order)
        retrieved = book.get_order(state, buy_order.order_id, Side.BUY)
        assert retrieved.quantity == pytest.approx(10.0, abs=0.05)

    def test_submit_buy_order_correct_order_id(self, book, empty_state, buy_order):
        """Retrieved buy order has the submitted order_id."""
        state, _ = book.submit_order(empty_state, buy_order)
        retrieved = book.get_order(state, buy_order.order_id, Side.BUY)
        assert retrieved.order_id == buy_order.order_id

    def test_submit_buy_order_correct_agent_id(self, book, empty_state, buy_order):
        """Retrieved buy order has the submitted agent_id."""
        state, _ = book.submit_order(empty_state, buy_order)
        retrieved = book.get_order(state, buy_order.order_id, Side.BUY)
        assert retrieved.agent_id == buy_order.agent_id

    def test_submit_buy_order_correct_timestamp(self, book, empty_state, buy_order):
        """Retrieved buy order has the submitted timestamp."""
        state, _ = book.submit_order(empty_state, buy_order)
        retrieved = book.get_order(state, buy_order.order_id, Side.BUY)
        assert retrieved.timestamp == buy_order.timestamp

    def test_submit_buy_order_not_on_sell_side(self, book, empty_state, buy_order):
        """A buy order must NOT appear on the sell side."""
        state, _ = book.submit_order(empty_state, buy_order)
        retrieved = book.get_order(state, buy_order.order_id, Side.SELL)
        assert _is_absent(retrieved)

    def test_submit_buy_at_negative_price(self, book, empty_state):
        """Buy order at negative price (-50.00 EUR/MWh) enters the book.

        Energy markets allow negative prices (market-spec SS2.3).
        """
        order = _make_buy(order_id=10, price=-50.00, quantity=1.0)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 10, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(-50.00, abs=0.005)

    def test_submit_buy_at_zero_price(self, book, empty_state):
        """Buy order at 0.00 EUR/MWh enters the book."""
        order = _make_buy(order_id=11, price=0.00, quantity=1.0)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 11, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(0.00, abs=0.005)

    def test_submit_buy_at_min_price(self, book, empty_state):
        """Buy order at minimum price (-9999.00) enters the book."""
        order = _make_buy(order_id=12, price=-9999.00, quantity=0.1)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 12, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(-9999.00, abs=0.005)

    def test_submit_buy_at_max_price(self, book, empty_state):
        """Buy order at maximum price (9999.00) enters the book."""
        order = _make_buy(order_id=13, price=9999.00, quantity=0.1)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 13, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(9999.00, abs=0.005)

    def test_submit_buy_at_min_quantity(self, book, empty_state):
        """Buy order at minimum quantity (0.1 MW) enters the book."""
        order = _make_buy(order_id=14, price=50.00, quantity=0.1)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 14, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.quantity == pytest.approx(0.1, abs=0.05)

    def test_submit_buy_at_max_quantity(self, book, empty_state):
        """Buy order at maximum quantity (999.0 MW) enters the book."""
        order = _make_buy(order_id=15, price=50.00, quantity=999.0)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 15, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.quantity == pytest.approx(999.0, abs=0.05)

    def test_submit_multiple_buy_orders(self, book, empty_state):
        """Multiple buy orders at different prices all enter the book."""
        o1 = _make_buy(order_id=20, price=40.00, timestamp=100)
        o2 = _make_buy(order_id=21, price=42.00, timestamp=101)
        o3 = _make_buy(order_id=22, price=44.00, timestamp=102)
        state, r1 = book.submit_order(empty_state, o1)
        state, r2 = book.submit_order(state, o2)
        state, r3 = book.submit_order(state, o3)
        assert r1.accepted == True
        assert r2.accepted == True
        assert r3.accepted == True
        assert not _is_absent(book.get_order(state, 20, Side.BUY))
        assert not _is_absent(book.get_order(state, 21, Side.BUY))
        assert not _is_absent(book.get_order(state, 22, Side.BUY))

    def test_submit_buy_returns_submit_result(self, book, empty_state, buy_order):
        """submit_order returns a SubmitResult (not a plain bool)."""
        _state, result = book.submit_order(empty_state, buy_order)
        assert isinstance(result, SubmitResult)

    def test_submit_buy_returns_new_state(self, book, empty_state, buy_order):
        """submit_order returns a new state object (not the same reference)."""
        new_state, _ = book.submit_order(empty_state, buy_order)
        # The state should differ -- at minimum, the book now has an order.
        # We verify by checking that the order is retrievable in new_state
        # but not in the original empty_state.
        assert not _is_absent(book.get_order(new_state, buy_order.order_id, Side.BUY))
        assert _is_absent(book.get_order(empty_state, buy_order.order_id, Side.BUY))


# ===========================================================================
# 2. TestSubmitSellLimitOrderEntersBook
# ===========================================================================

class TestSubmitSellLimitOrderEntersBook:
    """A valid sell limit order must enter the ask side of the book."""

    def test_submit_sell_returns_accepted_true(self, book, empty_state, sell_order):
        """submit_order for a valid sell returns accepted=True."""
        _state, result = book.submit_order(empty_state, sell_order)
        assert result.accepted == True

    def test_submit_sell_order_retrievable(self, book, empty_state, sell_order):
        """After submit, get_order returns the order on the SELL side."""
        state, _ = book.submit_order(empty_state, sell_order)
        retrieved = book.get_order(state, sell_order.order_id, Side.SELL)
        assert not _is_absent(retrieved)

    def test_submit_sell_order_correct_price(self, book, empty_state, sell_order):
        """Retrieved sell order has the submitted price (55.00 EUR/MWh)."""
        state, _ = book.submit_order(empty_state, sell_order)
        retrieved = book.get_order(state, sell_order.order_id, Side.SELL)
        assert retrieved.price == pytest.approx(55.00, abs=0.005)

    def test_submit_sell_order_correct_quantity(self, book, empty_state, sell_order):
        """Retrieved sell order has the submitted quantity (5.0 MW)."""
        state, _ = book.submit_order(empty_state, sell_order)
        retrieved = book.get_order(state, sell_order.order_id, Side.SELL)
        assert retrieved.quantity == pytest.approx(5.0, abs=0.05)

    def test_submit_sell_order_correct_order_id(self, book, empty_state, sell_order):
        """Retrieved sell order has the submitted order_id."""
        state, _ = book.submit_order(empty_state, sell_order)
        retrieved = book.get_order(state, sell_order.order_id, Side.SELL)
        assert retrieved.order_id == sell_order.order_id

    def test_submit_sell_order_correct_agent_id(self, book, empty_state, sell_order):
        """Retrieved sell order has the submitted agent_id."""
        state, _ = book.submit_order(empty_state, sell_order)
        retrieved = book.get_order(state, sell_order.order_id, Side.SELL)
        assert retrieved.agent_id == sell_order.agent_id

    def test_submit_sell_order_correct_timestamp(self, book, empty_state, sell_order):
        """Retrieved sell order has the submitted timestamp."""
        state, _ = book.submit_order(empty_state, sell_order)
        retrieved = book.get_order(state, sell_order.order_id, Side.SELL)
        assert retrieved.timestamp == sell_order.timestamp

    def test_submit_sell_order_not_on_buy_side(self, book, empty_state, sell_order):
        """A sell order must NOT appear on the buy side."""
        state, _ = book.submit_order(empty_state, sell_order)
        retrieved = book.get_order(state, sell_order.order_id, Side.BUY)
        assert _is_absent(retrieved)

    def test_submit_sell_at_negative_price(self, book, empty_state):
        """Sell order at negative price (-25.00 EUR/MWh) enters the book."""
        order = _make_sell(order_id=30, price=-25.00, quantity=2.0)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 30, Side.SELL)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(-25.00, abs=0.005)

    def test_submit_sell_at_zero_price(self, book, empty_state):
        """Sell order at 0.00 EUR/MWh enters the book."""
        order = _make_sell(order_id=31, price=0.00, quantity=1.0)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 31, Side.SELL)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(0.00, abs=0.005)

    def test_submit_sell_at_min_price(self, book, empty_state):
        """Sell order at minimum price (-9999.00) enters the book."""
        order = _make_sell(order_id=32, price=-9999.00, quantity=0.1)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 32, Side.SELL)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(-9999.00, abs=0.005)

    def test_submit_sell_at_max_price(self, book, empty_state):
        """Sell order at maximum price (9999.00) enters the book."""
        order = _make_sell(order_id=33, price=9999.00, quantity=0.1)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 33, Side.SELL)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(9999.00, abs=0.005)

    def test_submit_sell_at_min_quantity(self, book, empty_state):
        """Sell order at minimum quantity (0.1 MW) enters the book."""
        order = _make_sell(order_id=34, price=50.00, quantity=0.1)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True
        retrieved = book.get_order(state, 34, Side.SELL)
        assert not _is_absent(retrieved)
        assert retrieved.quantity == pytest.approx(0.1, abs=0.05)

    def test_submit_multiple_sell_orders(self, book, empty_state):
        """Multiple sell orders at different prices all enter the book."""
        o1 = _make_sell(order_id=40, price=60.00, timestamp=300)
        o2 = _make_sell(order_id=41, price=62.00, timestamp=301)
        o3 = _make_sell(order_id=42, price=64.00, timestamp=302)
        state, r1 = book.submit_order(empty_state, o1)
        state, r2 = book.submit_order(state, o2)
        state, r3 = book.submit_order(state, o3)
        assert r1.accepted == True
        assert r2.accepted == True
        assert r3.accepted == True
        assert not _is_absent(book.get_order(state, 40, Side.SELL))
        assert not _is_absent(book.get_order(state, 41, Side.SELL))
        assert not _is_absent(book.get_order(state, 42, Side.SELL))

    def test_submit_buy_and_sell_both_sides_populated(self, book, empty_state):
        """Submitting buy and sell (non-crossing) populates both sides."""
        buy = _make_buy(order_id=50, price=40.00, timestamp=100)
        sell = _make_sell(order_id=51, price=60.00, timestamp=200)
        state, _ = book.submit_order(empty_state, buy)
        state, _ = book.submit_order(state, sell)
        assert not _is_absent(book.get_order(state, 50, Side.BUY))
        assert not _is_absent(book.get_order(state, 51, Side.SELL))


# ===========================================================================
# 3. TestCancelOrder
# ===========================================================================

class TestCancelOrder:
    """cancel_order must fully remove an order from the book."""

    def test_cancel_buy_order_removes_from_book(self, book, empty_state, buy_order):
        """Cancelling a buy order makes it no longer retrievable."""
        state, _ = book.submit_order(empty_state, buy_order)
        # Confirm order is present
        assert not _is_absent(book.get_order(state, buy_order.order_id, Side.BUY))
        # Cancel it
        state = book.cancel_order(state, buy_order.order_id, Side.BUY)
        assert _is_absent(book.get_order(state, buy_order.order_id, Side.BUY))

    def test_cancel_sell_order_removes_from_book(self, book, empty_state, sell_order):
        """Cancelling a sell order makes it no longer retrievable."""
        state, _ = book.submit_order(empty_state, sell_order)
        assert not _is_absent(book.get_order(state, sell_order.order_id, Side.SELL))
        state = book.cancel_order(state, sell_order.order_id, Side.SELL)
        assert _is_absent(book.get_order(state, sell_order.order_id, Side.SELL))

    def test_cancel_does_not_affect_other_buy_orders(self, book, empty_state):
        """Cancelling one buy order leaves other buy orders intact."""
        o1 = _make_buy(order_id=60, price=40.00, timestamp=100)
        o2 = _make_buy(order_id=61, price=42.00, timestamp=101)
        state, _ = book.submit_order(empty_state, o1)
        state, _ = book.submit_order(state, o2)
        state = book.cancel_order(state, 60, Side.BUY)
        assert _is_absent(book.get_order(state, 60, Side.BUY))
        assert not _is_absent(book.get_order(state, 61, Side.BUY))

    def test_cancel_does_not_affect_other_sell_orders(self, book, empty_state):
        """Cancelling one sell order leaves other sell orders intact."""
        o1 = _make_sell(order_id=70, price=60.00, timestamp=300)
        o2 = _make_sell(order_id=71, price=62.00, timestamp=301)
        state, _ = book.submit_order(empty_state, o1)
        state, _ = book.submit_order(state, o2)
        state = book.cancel_order(state, 70, Side.SELL)
        assert _is_absent(book.get_order(state, 70, Side.SELL))
        assert not _is_absent(book.get_order(state, 71, Side.SELL))

    def test_cancel_does_not_affect_other_side(self, book, empty_state):
        """Cancelling a buy order does not affect sell orders."""
        buy = _make_buy(order_id=80, price=40.00, timestamp=100)
        sell = _make_sell(order_id=81, price=60.00, timestamp=200)
        state, _ = book.submit_order(empty_state, buy)
        state, _ = book.submit_order(state, sell)
        state = book.cancel_order(state, 80, Side.BUY)
        assert _is_absent(book.get_order(state, 80, Side.BUY))
        assert not _is_absent(book.get_order(state, 81, Side.SELL))

    def test_cancel_returns_lob_state(self, book, empty_state, buy_order):
        """cancel_order returns an OMIEBookState, not a (state, SubmitResult) pair."""
        state, _ = book.submit_order(empty_state, buy_order)
        result = book.cancel_order(state, buy_order.order_id, Side.BUY)
        # Must be an OMIEBookState (has lob with order-book attributes), not a SubmitResult
        assert hasattr(result, 'lob') and hasattr(result.lob, 'asks') and hasattr(result.lob, 'bids')
        assert not hasattr(result, 'accepted')

    def test_cancel_original_state_unchanged(self, book, empty_state, buy_order):
        """cancel_order does not mutate the input state (functional pattern)."""
        state_before, _ = book.submit_order(empty_state, buy_order)
        _state_after = book.cancel_order(state_before, buy_order.order_id, Side.BUY)
        # The order should still be retrievable in the pre-cancel state
        assert not _is_absent(book.get_order(state_before, buy_order.order_id, Side.BUY))

    def test_cancel_buy_order_at_negative_price(self, book, empty_state):
        """Cancel works for buy orders at negative prices."""
        order = _make_buy(order_id=85, price=-100.00, quantity=1.0)
        state, _ = book.submit_order(empty_state, order)
        state = book.cancel_order(state, 85, Side.BUY)
        assert _is_absent(book.get_order(state, 85, Side.BUY))

    def test_cancel_sell_order_at_negative_price(self, book, empty_state):
        """Cancel works for sell orders at negative prices."""
        order = _make_sell(order_id=86, price=-200.00, quantity=1.0)
        state, _ = book.submit_order(empty_state, order)
        state = book.cancel_order(state, 86, Side.SELL)
        assert _is_absent(book.get_order(state, 86, Side.SELL))


# ===========================================================================
# 4. TestModifyOrderResetsTimestamp
# ===========================================================================

class TestModifyOrderResetsTimestamp:
    """modify_order must cancel the old order and submit a new one.

    The new order gets a new timestamp, losing time priority.
    This is the XBID rule from market-spec SS5.3 (via architecture-review SS7.5).
    """

    def test_modify_replaces_order_with_new_timestamp(self, book, empty_state):
        """After modify, the order's timestamp is updated to the new value."""
        original = _make_buy(order_id=90, price=45.00, quantity=10.0, timestamp=100)
        state, _ = book.submit_order(empty_state, original)
        new_order = _make_buy(order_id=91, price=45.00, quantity=10.0, timestamp=500)
        state, result = book.modify_order(state, 90, Side.BUY, new_order)
        assert result.accepted == True
        # Old order should be gone
        assert _is_absent(book.get_order(state, 90, Side.BUY))
        # New order should be present with the new timestamp
        retrieved = book.get_order(state, 91, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.timestamp == 500

    def test_modify_order_returns_accepted(self, book, empty_state):
        """modify_order returns accepted=True when replacement is valid."""
        original = _make_buy(order_id=92, price=50.00, quantity=5.0, timestamp=100)
        state, _ = book.submit_order(empty_state, original)
        new_order = _make_buy(order_id=93, price=50.00, quantity=5.0, timestamp=200)
        state, result = book.modify_order(state, 92, Side.BUY, new_order)
        assert result.accepted == True

    def test_modify_order_returns_submit_result(self, book, empty_state):
        """modify_order returns a SubmitResult (not a plain bool)."""
        original = _make_sell(order_id=94, price=60.00, quantity=3.0, timestamp=100)
        state, _ = book.submit_order(empty_state, original)
        new_order = _make_sell(order_id=95, price=60.00, quantity=3.0, timestamp=200)
        state, result = book.modify_order(state, 94, Side.SELL, new_order)
        assert isinstance(result, SubmitResult)

    def test_modify_can_change_price(self, book, empty_state):
        """modify_order allows changing the price of an order."""
        original = _make_buy(order_id=100, price=45.00, quantity=10.0, timestamp=100)
        state, _ = book.submit_order(empty_state, original)
        new_order = _make_buy(order_id=101, price=47.50, quantity=10.0, timestamp=200)
        state, _ = book.modify_order(state, 100, Side.BUY, new_order)
        retrieved = book.get_order(state, 101, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(47.50, abs=0.005)

    def test_modify_can_change_quantity(self, book, empty_state):
        """modify_order allows changing the quantity of an order."""
        original = _make_sell(order_id=102, price=55.00, quantity=5.0, timestamp=100)
        state, _ = book.submit_order(empty_state, original)
        new_order = _make_sell(order_id=103, price=55.00, quantity=8.0, timestamp=200)
        state, _ = book.modify_order(state, 102, Side.SELL, new_order)
        retrieved = book.get_order(state, 103, Side.SELL)
        assert not _is_absent(retrieved)
        assert retrieved.quantity == pytest.approx(8.0, abs=0.05)

    def test_modify_loses_time_priority(self, book, empty_state):
        """A modified order loses time priority to orders at the same price.

        Submit order A at T=100, order B at T=200, both at price 50.00.
        Modify order A with T=300 (new_order_id). Order A's replacement
        now has timestamp 300 > B's 200, so B has time priority.
        """
        order_a = _make_buy(order_id=110, price=50.00, quantity=5.0, timestamp=100)
        order_b = _make_buy(order_id=111, price=50.00, quantity=5.0, timestamp=200)
        state, _ = book.submit_order(empty_state, order_a)
        state, _ = book.submit_order(state, order_b)
        # Modify A: new timestamp 300
        new_a = _make_buy(order_id=112, price=50.00, quantity=5.0, timestamp=300)
        state, _ = book.modify_order(state, 110, Side.BUY, new_a)
        # Verify new order A has timestamp 300
        retrieved_a = book.get_order(state, 112, Side.BUY)
        assert not _is_absent(retrieved_a)
        assert retrieved_a.timestamp == 300
        # Verify order B still has timestamp 200 (earlier = better priority)
        retrieved_b = book.get_order(state, 111, Side.BUY)
        assert not _is_absent(retrieved_b)
        assert retrieved_b.timestamp == 200
        # B's timestamp < A's new timestamp => B has time priority
        assert retrieved_b.timestamp < retrieved_a.timestamp

    def test_modify_sell_order(self, book, empty_state):
        """modify_order works for sell-side orders."""
        original = _make_sell(order_id=120, price=60.00, quantity=3.0, timestamp=100)
        state, _ = book.submit_order(empty_state, original)
        new_order = _make_sell(order_id=121, price=58.00, quantity=4.0, timestamp=300)
        state, result = book.modify_order(state, 120, Side.SELL, new_order)
        assert result.accepted == True
        assert _is_absent(book.get_order(state, 120, Side.SELL))
        retrieved = book.get_order(state, 121, Side.SELL)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(58.00, abs=0.005)
        assert retrieved.quantity == pytest.approx(4.0, abs=0.05)
        assert retrieved.timestamp == 300

    def test_modify_old_order_removed(self, book, empty_state):
        """After modify, the original order_id is no longer in the book."""
        original = _make_buy(order_id=130, price=45.00, quantity=10.0, timestamp=100)
        state, _ = book.submit_order(empty_state, original)
        new_order = _make_buy(order_id=131, price=46.00, quantity=10.0, timestamp=200)
        state, _ = book.modify_order(state, 130, Side.BUY, new_order)
        assert _is_absent(book.get_order(state, 130, Side.BUY))

    def test_modify_does_not_affect_other_orders(self, book, empty_state):
        """Modifying one order leaves other orders intact."""
        o1 = _make_buy(order_id=140, price=44.00, quantity=5.0, timestamp=100)
        o2 = _make_buy(order_id=141, price=46.00, quantity=5.0, timestamp=101)
        state, _ = book.submit_order(empty_state, o1)
        state, _ = book.submit_order(state, o2)
        new_o1 = _make_buy(order_id=142, price=45.00, quantity=5.0, timestamp=300)
        state, _ = book.modify_order(state, 140, Side.BUY, new_o1)
        # o2 should be unchanged
        retrieved = book.get_order(state, 141, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(46.00, abs=0.005)


# ===========================================================================
# 5. TestSubmitResultContract -- SubmitResult NamedTuple shape
# ===========================================================================

class TestSubmitResultContract:
    """SubmitResult must be a NamedTuple with at least an 'accepted' field."""

    def test_submit_result_has_accepted_field(self):
        """SubmitResult has an 'accepted' attribute."""
        r = SubmitResult(accepted=True)
        assert hasattr(r, 'accepted')

    def test_submit_result_accepted_true(self):
        """SubmitResult(accepted=True).accepted == True."""
        r = SubmitResult(accepted=True)
        assert r.accepted == True

    def test_submit_result_accepted_false(self):
        """SubmitResult(accepted=False).accepted == False."""
        r = SubmitResult(accepted=False)
        assert r.accepted == False

    def test_submit_result_is_namedtuple(self):
        """SubmitResult is a NamedTuple (has _fields attribute)."""
        assert hasattr(SubmitResult, '_fields')
        assert 'accepted' in SubmitResult._fields

    def test_submit_result_tuple_access(self):
        """SubmitResult supports positional (tuple-index) access."""
        r = SubmitResult(accepted=True)
        # 'accepted' should be accessible by index
        assert r[SubmitResult._fields.index('accepted')] is True


# ===========================================================================
# 6. TestValidationRejection -- submit_order rejects invalid orders
# ===========================================================================

class TestValidationRejection:
    """submit_order delegates to order_validator and rejects invalid orders.

    As of P2-L1-HOTFIX-tick-precision (Option C), off-tick floats snap to the
    nearest tick via the encoder; rejection for sub-half-tick offsets is no
    longer a behavioral contract of the submit_order path. Out-of-bounds,
    contract-closed, side-mismatch, and positive-ID rejection tests remain.
    The encoder-layer snap contract lives in
    ``tests/omiecid/test_encoding_snap_semantics.py``.
    """

    def test_reject_price_below_min(self, book, empty_state):
        """Order with price below -9999.0 is rejected."""
        order = _make_buy(order_id=200, price=-9999.01, quantity=1.0)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == False

    def test_reject_price_above_max(self, book, empty_state):
        """Order with price above 9999.0 is rejected."""
        order = _make_sell(order_id=201, price=9999.01, quantity=1.0)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == False

    def test_reject_quantity_below_min(self, book, empty_state):
        """Order with quantity below 0.1 MW is rejected."""
        order = _make_buy(order_id=202, price=50.00, quantity=0.05)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == False

    def test_reject_quantity_above_max(self, book, empty_state):
        """Order with quantity above 999.0 MW is rejected."""
        order = _make_sell(order_id=203, price=50.00, quantity=999.1)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == False

    def test_rejected_order_does_not_enter_book(self, book, empty_state):
        """A rejected order must not appear in the book."""
        order = _make_buy(order_id=206, price=-10000.00, quantity=1.0)
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == False
        assert _is_absent(book.get_order(state, 206, Side.BUY))

    def test_rejected_order_state_unchanged(self, book, empty_state):
        """A rejected order must not change the book state.

        We verify by submitting a valid order, then a rejected order,
        then checking the valid order is still present and unchanged.
        """
        valid = _make_buy(order_id=207, price=50.00, quantity=1.0, timestamp=100)
        state, _ = book.submit_order(empty_state, valid)
        invalid = _make_buy(order_id=208, price=-10000.00, quantity=1.0, timestamp=200)
        state_after, result = book.submit_order(state, invalid)
        assert result.accepted == False
        # Valid order should still be there
        assert not _is_absent(book.get_order(state_after, 207, Side.BUY))

    def test_accept_valid_iceberg_order(self, book, empty_state):
        """Valid ICE orders are accepted (M4.1 removes M1.3 blanket rejection)."""
        order = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=10.0,
            order_id=209, agent_id=0, timestamp=100,
            exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
        )
        state, result = book.submit_order(empty_state, order)
        assert result.accepted == True

    def test_reject_on_closed_contract(self, closed_contract, product_config, market_config):
        """Orders on a CLOSED contract are rejected."""
        closed_book = ContractBook(closed_contract, product_config, market_config)
        state = closed_book.init()
        order = _make_buy(order_id=210, price=50.00, quantity=1.0)
        state, result = closed_book.submit_order(state, order)
        assert result.accepted == False


# ===========================================================================
# 7. TestPositiveOrderIdEnforcement -- reject order_id <= 0
# ===========================================================================

class TestPositiveOrderIdEnforcement:
    """ContractBook must reject orders with order_id <= 0.

    Positive order IDs are required to avoid collisions with jaxob's
    reserved negative IDs (-1, -2, -99, -888888).
    Confirmed by architect open question #3 (answered YES).
    """

    def test_reject_order_id_zero(self, book, empty_state):
        """order_id=0 is rejected."""
        order = _make_buy(order_id=0, price=50.00, quantity=1.0)
        _state, result = book.submit_order(empty_state, order)
        assert result.accepted == False

    def test_reject_order_id_negative(self, book, empty_state):
        """order_id=-1 is rejected."""
        order = _make_buy(order_id=-1, price=50.00, quantity=1.0)
        _state, result = book.submit_order(empty_state, order)
        assert result.accepted == False

    def test_reject_order_id_large_negative(self, book, empty_state):
        """order_id=-999 is rejected."""
        order = _make_sell(order_id=-999, price=50.00, quantity=1.0)
        _state, result = book.submit_order(empty_state, order)
        assert result.accepted == False

    def test_accept_order_id_one(self, book, empty_state):
        """order_id=1 (smallest positive) is accepted."""
        order = _make_buy(order_id=1, price=50.00, quantity=1.0)
        _state, result = book.submit_order(empty_state, order)
        assert result.accepted == True

    def test_reject_zero_id_does_not_enter_book(self, book, empty_state):
        """Order with order_id=0 must not appear in the book."""
        order = _make_buy(order_id=0, price=50.00, quantity=1.0)
        state, _ = book.submit_order(empty_state, order)
        assert _is_absent(book.get_order(state, 0, Side.BUY))


# ===========================================================================
# 8. TestCancelMissingOrder -- silent return on missing order
# ===========================================================================

class TestCancelMissingOrder:
    """Cancelling a non-existent order must return the state unchanged.

    Confirmed by architect open question #2 (answered YES: silent return).
    This protects against jaxob's STRICT_BY_ID corruption on missing orders.
    """

    def test_cancel_nonexistent_buy_returns_state(self, book, empty_state):
        """Cancelling a non-existent buy order returns the state unchanged."""
        state = book.cancel_order(empty_state, 999, Side.BUY)
        # State should be equivalent to empty_state (no corruption)
        assert _is_absent(book.get_order(state, 999, Side.BUY))

    def test_cancel_nonexistent_sell_returns_state(self, book, empty_state):
        """Cancelling a non-existent sell order returns the state unchanged."""
        state = book.cancel_order(empty_state, 999, Side.SELL)
        assert _is_absent(book.get_order(state, 999, Side.SELL))

    def test_cancel_nonexistent_does_not_corrupt_existing(self, book, empty_state):
        """Cancelling a missing order must not corrupt existing orders.

        This guards against the jaxob STRICT_BY_ID bug where a missing-order
        cancel silently corrupts the last row in the book side array
        (architecture-review SS6.3).
        """
        o1 = _make_buy(order_id=300, price=40.00, quantity=5.0, timestamp=100)
        o2 = _make_buy(order_id=301, price=42.00, quantity=3.0, timestamp=101)
        state, _ = book.submit_order(empty_state, o1)
        state, _ = book.submit_order(state, o2)
        # Cancel a non-existent order
        state = book.cancel_order(state, 999, Side.BUY)
        # Both existing orders must still be intact
        r1 = book.get_order(state, 300, Side.BUY)
        r2 = book.get_order(state, 301, Side.BUY)
        assert not _is_absent(r1)
        assert not _is_absent(r2)
        assert r1.price == pytest.approx(40.00, abs=0.005)
        assert r2.price == pytest.approx(42.00, abs=0.005)

    def test_cancel_nonexistent_sell_does_not_corrupt_existing(self, book, empty_state):
        """Same corruption guard for the sell side."""
        o1 = _make_sell(order_id=310, price=60.00, quantity=5.0, timestamp=200)
        o2 = _make_sell(order_id=311, price=62.00, quantity=3.0, timestamp=201)
        state, _ = book.submit_order(empty_state, o1)
        state, _ = book.submit_order(state, o2)
        state = book.cancel_order(state, 999, Side.SELL)
        r1 = book.get_order(state, 310, Side.SELL)
        r2 = book.get_order(state, 311, Side.SELL)
        assert not _is_absent(r1)
        assert not _is_absent(r2)
        assert r1.price == pytest.approx(60.00, abs=0.005)
        assert r2.price == pytest.approx(62.00, abs=0.005)

    def test_cancel_wrong_side_does_not_remove_order(self, book, empty_state):
        """Cancelling with the wrong side does not remove the order.

        A buy order cancelled via Side.SELL should leave the buy intact.
        """
        buy = _make_buy(order_id=320, price=45.00, quantity=5.0, timestamp=100)
        state, _ = book.submit_order(empty_state, buy)
        state = book.cancel_order(state, 320, Side.SELL)
        # Buy order should still be there
        assert not _is_absent(book.get_order(state, 320, Side.BUY))


# ===========================================================================
# 9. TestGetOrder -- read-back helper contract
# ===========================================================================

class TestGetOrder:
    """get_order must look up orders by ID and return them in OMIE units."""

    def test_get_order_returns_none_for_empty_book(self, book, empty_state):
        """get_order returns None when the book is empty."""
        assert _is_absent(book.get_order(empty_state, 1, Side.BUY))
        assert _is_absent(book.get_order(empty_state, 1, Side.SELL))

    def test_get_order_returns_omie_order(self, book, empty_state, buy_order):
        """get_order returns an OMIEOrder (or compatible NamedTuple)."""
        state, _ = book.submit_order(empty_state, buy_order)
        retrieved = book.get_order(state, buy_order.order_id, Side.BUY)
        assert not _is_absent(retrieved)
        # Should have the standard OMIEOrder fields
        assert hasattr(retrieved, 'price')
        assert hasattr(retrieved, 'quantity')
        assert hasattr(retrieved, 'order_id')
        assert hasattr(retrieved, 'agent_id')
        assert hasattr(retrieved, 'timestamp')
        assert hasattr(retrieved, 'side')

    def test_get_order_price_in_eur_mwh(self, book, empty_state):
        """get_order returns price in EUR/MWh (OMIE units, not jaxob int32)."""
        order = _make_buy(order_id=400, price=123.45, quantity=1.0)
        state, _ = book.submit_order(empty_state, order)
        retrieved = book.get_order(state, 400, Side.BUY)
        assert retrieved.price == pytest.approx(123.45, abs=0.005)

    def test_get_order_quantity_in_mw(self, book, empty_state):
        """get_order returns quantity in MW (OMIE units, not jaxob int32)."""
        order = _make_sell(order_id=401, price=50.00, quantity=7.3)
        state, _ = book.submit_order(empty_state, order)
        retrieved = book.get_order(state, 401, Side.SELL)
        assert retrieved.quantity == pytest.approx(7.3, abs=0.05)

    def test_get_order_none_for_wrong_side(self, book, empty_state, buy_order):
        """get_order returns None when looking up the wrong side."""
        state, _ = book.submit_order(empty_state, buy_order)
        assert _is_absent(book.get_order(state, buy_order.order_id, Side.SELL))

    def test_get_order_none_for_wrong_id(self, book, empty_state, buy_order):
        """get_order returns None when looking up a non-existent order_id."""
        state, _ = book.submit_order(empty_state, buy_order)
        assert _is_absent(book.get_order(state, 99999, Side.BUY))


# ===========================================================================
# 10. TestContractBookInit -- constructor and init
# ===========================================================================

class TestContractBookInit:
    """ContractBook constructor and init() must work with default configs."""

    def test_constructor_accepts_contract_and_configs(self, open_contract):
        """ContractBook can be instantiated with contract, product_config, market_config."""
        book = ContractBook(open_contract, OMIEProductConfig(), OMIEMarketConfig())
        assert book is not None

    def test_constructor_default_configs(self, open_contract):
        """ContractBook can be instantiated with default configs."""
        book = ContractBook(open_contract)
        assert book is not None

    def test_init_returns_state(self, book):
        """init() returns a state object."""
        state = book.init()
        assert state is not None

    def test_init_returns_empty_book(self, book):
        """init() returns an empty book (no orders on either side)."""
        state = book.init()
        # No order should be found for any arbitrary ID
        assert _is_absent(book.get_order(state, 1, Side.BUY))
        assert _is_absent(book.get_order(state, 1, Side.SELL))

    def test_init_callable_multiple_times(self, book):
        """init() can be called multiple times to get fresh states."""
        s1 = book.init()
        s2 = book.init()
        assert s1 is not None
        assert s2 is not None


# ===========================================================================
# 11. TestModifyWithInvalidNewOrder -- modify rejects invalid replacement
# ===========================================================================

class TestModifyWithInvalidNewOrder:
    """modify_order should reject if the new_order fails validation."""

    def test_modify_with_invalid_price_rejects(self, book, empty_state):
        """Modify with new_order price out of bounds rejects."""
        original = _make_buy(order_id=500, price=50.00, quantity=5.0, timestamp=100)
        state, _ = book.submit_order(empty_state, original)
        bad_new = _make_buy(order_id=501, price=-10000.00, quantity=5.0, timestamp=200)
        state, result = book.modify_order(state, 500, Side.BUY, bad_new)
        assert result.accepted == False

    def test_modify_with_invalid_quantity_rejects(self, book, empty_state):
        """Modify with new_order quantity out of bounds rejects."""
        original = _make_sell(order_id=502, price=60.00, quantity=5.0, timestamp=100)
        state, _ = book.submit_order(empty_state, original)
        bad_new = _make_sell(order_id=503, price=60.00, quantity=0.05, timestamp=200)
        state, result = book.modify_order(state, 502, Side.SELL, bad_new)
        assert result.accepted == False

    def test_modify_with_zero_order_id_rejects(self, book, empty_state):
        """Modify with new_order having order_id=0 rejects (positive ID rule)."""
        original = _make_buy(order_id=504, price=50.00, quantity=5.0, timestamp=100)
        state, _ = book.submit_order(empty_state, original)
        bad_new = _make_buy(order_id=0, price=50.00, quantity=5.0, timestamp=200)
        state, result = book.modify_order(state, 504, Side.BUY, bad_new)
        assert result.accepted == False
