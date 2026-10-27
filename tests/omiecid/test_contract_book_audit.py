"""Audit tests for M3.1 -- ContractBook: Submit + Cancel.

Post-implementation verification focusing on:
- Adapter layer correctness: price offset encoding round-trip, quantity scaling round-trip
- Dual side-mapping correctness (identified as most likely bug source by architect)
- Cancel semantics: full removal, not partial quantity reduction
- Boundary conditions: extreme prices (-9999.0, +9999.0), minimum/maximum quantities
- Multiple orders interacting: submit multiple, cancel one, verify others intact
- Modify preserving correct fields (only timestamp changes priority)
- Contract status integration: orders on OPEN contract accepted, on non-OPEN rejected

Derived independently from:
- docs/omiecid/market-spec.md (domain rules)
- docs/omiecid/agile-plan.md (M3.1 acceptance criteria)
- docs/omiecid/architecture-review.md (interface contract, open questions)
- docs/omiecid/m3-api-mapping.md (adapter design)
"""

import pytest

from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.order_types import (
    OMIEOrder,
    Side,
    ExecRestriction,
    Validity,
    Status,
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
# Specialized open_contract overrides conftest (contract_id=10 vs 37)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def open_contract():
    """An OPEN contract suitable for testing."""
    return Contract(
        contract_id=10,
        delivery_start_minutes=150,
        delivery_end_minutes=165,
        gct_minutes=90,
        status=ContractStatus.OPEN,
    )


@pytest.fixture
def book(open_contract, product_config, market_config):
    return ContractBook(open_contract, product_config, market_config)


@pytest.fixture
def empty_state(book):
    return book.init()


def _buy_order(price=50.0, quantity=1.0, order_id=1, agent_id=0, timestamp=100):
    return make_order(
        contract_id=10, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
    )


def _sell_order(price=50.0, quantity=1.0, order_id=2, agent_id=0, timestamp=100):
    return make_order(
        contract_id=10, side=Side.SELL, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Price offset encoding round-trip
# ---------------------------------------------------------------------------

class TestPriceOffsetRoundTrip:
    """Verify that prices survive the OMIE -> jaxob -> OMIE round-trip
    through the offset encoding layer. This is the adapter correctness
    test for the price domain (market-spec S2.3, m3-api-mapping S4.3)."""

    def test_positive_price_round_trip(self, book, empty_state):
        state, result = book.submit_order(empty_state, _buy_order(price=123.45))
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(123.45, abs=0.005)

    def test_negative_price_round_trip(self, book, empty_state):
        state, result = book.submit_order(empty_state, _buy_order(price=-50.01))
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(-50.01, abs=0.005)

    def test_zero_price_round_trip(self, book, empty_state):
        state, result = book.submit_order(empty_state, _buy_order(price=0.0))
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(0.0, abs=0.005)

    def test_extreme_negative_price_round_trip(self, book, empty_state):
        """Price at the OMIE floor: -9999.0 EUR/MWh."""
        state, result = book.submit_order(
            empty_state, _buy_order(price=-9999.0, quantity=0.1)
        )
        assert result.accepted
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(-9999.0, abs=0.005)

    def test_extreme_positive_price_round_trip(self, book, empty_state):
        """Price at the OMIE ceiling: +9999.0 EUR/MWh."""
        state, result = book.submit_order(
            empty_state, _sell_order(price=9999.0, quantity=0.1)
        )
        assert result.accepted
        retrieved = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(9999.0, abs=0.005)

    def test_small_positive_price_round_trip(self, book, empty_state):
        """Smallest positive price: 0.01 EUR/MWh (one tick above zero)."""
        state, result = book.submit_order(
            empty_state, _buy_order(price=0.01)
        )
        assert result.accepted
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(0.01, abs=0.005)

    def test_small_negative_price_round_trip(self, book, empty_state):
        """Smallest negative price: -0.01 EUR/MWh (one tick below zero)."""
        state, result = book.submit_order(
            empty_state, _buy_order(price=-0.01)
        )
        assert result.accepted
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.price == pytest.approx(-0.01, abs=0.005)


# ---------------------------------------------------------------------------
# Quantity scaling round-trip
# ---------------------------------------------------------------------------

class TestQuantityScalingRoundTrip:
    """Verify that quantities survive the OMIE -> jaxob -> OMIE round-trip
    through the scaling layer (m3-api-mapping S4.4)."""

    def test_standard_quantity_round_trip(self, book, empty_state):
        state, _ = book.submit_order(empty_state, _buy_order(quantity=5.3))
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.quantity == pytest.approx(5.3, abs=0.05)

    def test_minimum_quantity_round_trip(self, book, empty_state):
        """0.1 MW is the minimum order quantity (market-spec S2.3)."""
        state, _ = book.submit_order(empty_state, _buy_order(quantity=0.1))
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.quantity == pytest.approx(0.1, abs=0.05)

    def test_maximum_quantity_round_trip(self, book, empty_state):
        """999.0 MW is the maximum order quantity (market-spec S2.3)."""
        state, _ = book.submit_order(empty_state, _buy_order(quantity=999.0))
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.quantity == pytest.approx(999.0, abs=0.05)

    def test_one_mw_round_trip(self, book, empty_state):
        state, _ = book.submit_order(empty_state, _buy_order(quantity=1.0))
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.quantity == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Dual side-mapping correctness
# ---------------------------------------------------------------------------

class TestDualSideMapping:
    """The architecture review (S6.4) identified dual side-mapping as the
    most likely bug source. Verify that BUY orders land on the bid side and
    SELL orders land on the ask side, and that get_order uses the correct
    lookup side for each."""

    def test_buy_order_found_on_buy_side(self, book, empty_state):
        state, _ = book.submit_order(empty_state, _buy_order(order_id=10))
        assert not _is_absent(book.get_order(state, 10, Side.BUY))

    def test_buy_order_not_found_on_sell_side(self, book, empty_state):
        state, _ = book.submit_order(empty_state, _buy_order(order_id=10))
        assert _is_absent(book.get_order(state, 10, Side.SELL))

    def test_sell_order_found_on_sell_side(self, book, empty_state):
        state, _ = book.submit_order(empty_state, _sell_order(order_id=20))
        assert not _is_absent(book.get_order(state, 20, Side.SELL))

    def test_sell_order_not_found_on_buy_side(self, book, empty_state):
        state, _ = book.submit_order(empty_state, _sell_order(order_id=20))
        assert _is_absent(book.get_order(state, 20, Side.BUY))

    def test_both_sides_populated_independently(self, book, empty_state):
        """Submit one buy and one sell. Both should be retrievable on their
        respective sides without cross-contamination."""
        state, r1 = book.submit_order(empty_state, _buy_order(order_id=10, price=49.0))
        state, r2 = book.submit_order(state, _sell_order(order_id=20, price=51.0))
        assert r1.accepted
        assert r2.accepted

        buy = book.get_order(state, 10, Side.BUY)
        sell = book.get_order(state, 20, Side.SELL)
        assert not _is_absent(buy)
        assert not _is_absent(sell)
        assert buy.price == pytest.approx(49.0, abs=0.005)
        assert sell.price == pytest.approx(51.0, abs=0.005)

    def test_cancel_buy_does_not_affect_sell(self, book, empty_state):
        """Cancel a buy order; sell order on the other side remains."""
        state, _ = book.submit_order(empty_state, _buy_order(order_id=10, price=49.0))
        state, _ = book.submit_order(state, _sell_order(order_id=20, price=51.0))
        state = book.cancel_order(state, 10, Side.BUY)

        assert _is_absent(book.get_order(state, 10, Side.BUY))
        assert not _is_absent(book.get_order(state, 20, Side.SELL))

    def test_cancel_sell_does_not_affect_buy(self, book, empty_state):
        """Cancel a sell order; buy order on the other side remains."""
        state, _ = book.submit_order(empty_state, _buy_order(order_id=10, price=49.0))
        state, _ = book.submit_order(state, _sell_order(order_id=20, price=51.0))
        state = book.cancel_order(state, 20, Side.SELL)

        assert _is_absent(book.get_order(state, 20, Side.SELL))
        assert not _is_absent(book.get_order(state, 10, Side.BUY))


# ---------------------------------------------------------------------------
# Cancel semantics: full removal
# ---------------------------------------------------------------------------

class TestCancelFullRemoval:
    """OMIE cancel means full order removal, not partial quantity reduction.
    The adapter must look up the full remaining quantity and issue a cancel
    for that exact amount (architecture-review S6)."""

    def test_cancel_removes_order_entirely(self, book, empty_state):
        state, _ = book.submit_order(empty_state, _buy_order(quantity=10.0, order_id=1))
        state = book.cancel_order(state, 1, Side.BUY)
        assert _is_absent(book.get_order(state, 1, Side.BUY))

    def test_cancel_sell_removes_order_entirely(self, book, empty_state):
        state, _ = book.submit_order(empty_state, _sell_order(quantity=10.0, order_id=2))
        state = book.cancel_order(state, 2, Side.SELL)
        assert _is_absent(book.get_order(state, 2, Side.SELL))

    def test_cancel_nonexistent_returns_unchanged_state(self, book, empty_state):
        """Architecture-review open question #2: silent return on cancel of
        missing order."""
        state = book.cancel_order(empty_state, 999, Side.BUY)
        # Should not raise; state should be returned
        assert state is not None

    def test_cancel_nonexistent_does_not_corrupt_existing(self, book, empty_state):
        """STRICT_BY_ID corruption guard: cancelling a nonexistent order must
        not corrupt the last row in the book (architecture-review S6.3)."""
        state, _ = book.submit_order(empty_state, _buy_order(order_id=1))
        state = book.cancel_order(state, 999, Side.BUY)
        # The existing order should still be intact
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.order_id == 1

    def test_cancel_wrong_side_leaves_order_intact(self, book, empty_state):
        """Cancelling with the wrong side should not find the order."""
        state, _ = book.submit_order(empty_state, _buy_order(order_id=1))
        state = book.cancel_order(state, 1, Side.SELL)  # wrong side
        # Order should still be on the buy side
        assert not _is_absent(book.get_order(state, 1, Side.BUY))


# ---------------------------------------------------------------------------
# Multiple orders: submit several, cancel one, verify others intact
# ---------------------------------------------------------------------------

class TestMultipleOrdersInteraction:
    """Test that multiple orders on the same side coexist correctly and that
    cancelling one does not affect others."""

    def test_three_buy_orders_all_retrievable(self, book, empty_state):
        o1 = _buy_order(price=48.0, order_id=1, timestamp=100)
        o2 = _buy_order(price=49.0, order_id=2, timestamp=101)
        o3 = _buy_order(price=50.0, order_id=3, timestamp=102)
        state, _ = book.submit_order(empty_state, o1)
        state, _ = book.submit_order(state, o2)
        state, _ = book.submit_order(state, o3)
        assert not _is_absent(book.get_order(state, 1, Side.BUY))
        assert not _is_absent(book.get_order(state, 2, Side.BUY))
        assert not _is_absent(book.get_order(state, 3, Side.BUY))

    def test_cancel_middle_order_preserves_others(self, book, empty_state):
        o1 = _buy_order(price=48.0, order_id=1, timestamp=100)
        o2 = _buy_order(price=49.0, order_id=2, timestamp=101)
        o3 = _buy_order(price=50.0, order_id=3, timestamp=102)
        state, _ = book.submit_order(empty_state, o1)
        state, _ = book.submit_order(state, o2)
        state, _ = book.submit_order(state, o3)

        state = book.cancel_order(state, 2, Side.BUY)
        assert _is_absent(book.get_order(state, 2, Side.BUY))
        assert not _is_absent(book.get_order(state, 1, Side.BUY))
        assert not _is_absent(book.get_order(state, 3, Side.BUY))

    def test_cancel_first_order_preserves_others(self, book, empty_state):
        o1 = _sell_order(price=51.0, order_id=10, timestamp=100)
        o2 = _sell_order(price=52.0, order_id=11, timestamp=101)
        state, _ = book.submit_order(empty_state, o1)
        state, _ = book.submit_order(state, o2)

        state = book.cancel_order(state, 10, Side.SELL)
        assert _is_absent(book.get_order(state, 10, Side.SELL))
        assert not _is_absent(book.get_order(state, 11, Side.SELL))

    def test_submit_buy_and_sell_at_different_prices_no_interference(self, book, empty_state):
        """Buy at low price, sell at high price. No crossing, both should
        coexist in the book."""
        buy = _buy_order(price=40.0, order_id=1, timestamp=100)
        sell = _sell_order(price=60.0, order_id=2, timestamp=101)
        state, r1 = book.submit_order(empty_state, buy)
        state, r2 = book.submit_order(state, sell)
        assert r1.accepted and r2.accepted

        b = book.get_order(state, 1, Side.BUY)
        s = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(b) and not _is_absent(s)
        assert b.price == pytest.approx(40.0, abs=0.005)
        assert s.price == pytest.approx(60.0, abs=0.005)


# ---------------------------------------------------------------------------
# Modify order: timestamp changes, old order removed
# ---------------------------------------------------------------------------

class TestModifyOrderBehavior:
    """modify_order = cancel + submit. The new order must have the new
    timestamp. The old order must no longer exist in the book."""

    def test_modify_replaces_with_new_timestamp(self, book, empty_state):
        original = _buy_order(price=50.0, order_id=1, timestamp=100)
        state, _ = book.submit_order(empty_state, original)

        replacement = _buy_order(price=50.0, order_id=5, timestamp=200)
        state, result = book.modify_order(state, 1, Side.BUY, replacement)
        assert result.accepted

        # Old order gone
        assert _is_absent(book.get_order(state, 1, Side.BUY))
        # New order present with new timestamp
        new = book.get_order(state, 5, Side.BUY)
        assert not _is_absent(new)
        assert new.timestamp == 200

    def test_modify_can_change_price(self, book, empty_state):
        original = _sell_order(price=55.0, order_id=2, timestamp=100)
        state, _ = book.submit_order(empty_state, original)

        replacement = _sell_order(price=60.0, order_id=3, timestamp=200)
        state, result = book.modify_order(state, 2, Side.SELL, replacement)
        assert result.accepted

        assert _is_absent(book.get_order(state, 2, Side.SELL))
        new = book.get_order(state, 3, Side.SELL)
        assert not _is_absent(new)
        assert new.price == pytest.approx(60.0, abs=0.005)

    def test_modify_can_change_quantity(self, book, empty_state):
        original = _buy_order(price=50.0, quantity=5.0, order_id=1, timestamp=100)
        state, _ = book.submit_order(empty_state, original)

        replacement = _buy_order(price=50.0, quantity=10.0, order_id=4, timestamp=200)
        state, result = book.modify_order(state, 1, Side.BUY, replacement)
        assert result.accepted

        new = book.get_order(state, 4, Side.BUY)
        assert not _is_absent(new)
        assert new.quantity == pytest.approx(10.0, abs=0.05)

    def test_modify_loses_time_priority(self, book, empty_state):
        """XBID rule (market-spec S5.3): modified order loses time priority.
        Submit two orders at the same price. Modify the first with a later
        timestamp. The modified order should now have a later timestamp than
        the second order."""
        o1 = _buy_order(price=50.0, order_id=1, timestamp=100)
        o2 = _buy_order(price=50.0, order_id=2, timestamp=200)
        state, _ = book.submit_order(empty_state, o1)
        state, _ = book.submit_order(state, o2)

        # Modify o1 with timestamp 300 (later than o2's 200)
        replacement = _buy_order(price=50.0, order_id=3, timestamp=300)
        state, result = book.modify_order(state, 1, Side.BUY, replacement)
        assert result.accepted

        new_order = book.get_order(state, 3, Side.BUY)
        second_order = book.get_order(state, 2, Side.BUY)
        assert not _is_absent(new_order)
        assert not _is_absent(second_order)
        # The modified order's timestamp (300) is after the unmodified order (200)
        assert new_order.timestamp > second_order.timestamp

    def test_modify_does_not_affect_other_orders(self, book, empty_state):
        o1 = _buy_order(price=48.0, order_id=1, timestamp=100)
        o2 = _buy_order(price=49.0, order_id=2, timestamp=101)
        state, _ = book.submit_order(empty_state, o1)
        state, _ = book.submit_order(state, o2)

        replacement = _buy_order(price=50.0, order_id=3, timestamp=200)
        state, _ = book.modify_order(state, 1, Side.BUY, replacement)

        # o2 should be unaffected
        o2_retrieved = book.get_order(state, 2, Side.BUY)
        assert not _is_absent(o2_retrieved)
        assert o2_retrieved.price == pytest.approx(49.0, abs=0.005)

    def test_modify_sell_order(self, book, empty_state):
        original = _sell_order(price=55.0, order_id=10, timestamp=100)
        state, _ = book.submit_order(empty_state, original)

        replacement = _sell_order(price=56.0, order_id=11, timestamp=200)
        state, result = book.modify_order(state, 10, Side.SELL, replacement)
        assert result.accepted
        assert _is_absent(book.get_order(state, 10, Side.SELL))
        assert not _is_absent(book.get_order(state, 11, Side.SELL))

    def test_modify_with_invalid_replacement_rejected(self, book, empty_state):
        """If the replacement order is invalid, modify should cancel the old
        order but reject the new one."""
        original = _buy_order(price=50.0, order_id=1, timestamp=100)
        state, _ = book.submit_order(empty_state, original)

        # Invalid price (above max)
        bad_replacement = _buy_order(price=99999.0, order_id=2, timestamp=200)
        state, result = book.modify_order(state, 1, Side.BUY, bad_replacement)
        assert not result.accepted


# ---------------------------------------------------------------------------
# Contract status integration: OPEN vs non-OPEN
# ---------------------------------------------------------------------------

class TestContractStatusGating:
    """Orders must only be accepted when the contract status is OPEN.
    Non-OPEN contracts should reject submissions."""

    def test_pending_contract_rejects_order(self, product_config, market_config):
        contract = Contract(
            contract_id=5, delivery_start_minutes=75,
            delivery_end_minutes=90, gct_minutes=15,
            status=ContractStatus.PENDING,
        )
        book = ContractBook(contract, product_config, market_config)
        state = book.init()
        _, result = book.submit_order(state, _buy_order())
        assert not result.accepted

    def test_closed_contract_rejects_order(self, product_config, market_config):
        contract = Contract(
            contract_id=5, delivery_start_minutes=75,
            delivery_end_minutes=90, gct_minutes=15,
            status=ContractStatus.CLOSED,
        )
        book = ContractBook(contract, product_config, market_config)
        state = book.init()
        _, result = book.submit_order(state, _sell_order())
        assert not result.accepted

    def test_suspended_contract_rejects_order(self, product_config, market_config):
        contract = Contract(
            contract_id=5, delivery_start_minutes=75,
            delivery_end_minutes=90, gct_minutes=15,
            status=ContractStatus.SUSPENDED,
        )
        book = ContractBook(contract, product_config, market_config)
        state = book.init()
        _, result = book.submit_order(state, _buy_order())
        assert not result.accepted

    def test_open_contract_accepts_order(self, book, empty_state):
        _, result = book.submit_order(empty_state, _buy_order())
        assert result.accepted

    def test_rejected_order_does_not_enter_book(self, product_config, market_config):
        contract = Contract(
            contract_id=5, delivery_start_minutes=75,
            delivery_end_minutes=90, gct_minutes=15,
            status=ContractStatus.CLOSED,
        )
        book = ContractBook(contract, product_config, market_config)
        state = book.init()
        state, result = book.submit_order(state, _buy_order())
        assert not result.accepted
        assert _is_absent(book.get_order(state, 1, Side.BUY))


# ---------------------------------------------------------------------------
# Positive order ID enforcement
# ---------------------------------------------------------------------------

class TestPositiveOrderIdEnforcement:
    """Architecture-review open question #3: order_id must be positive.
    IDs <= 0 collide with jaxob reserved values."""

    def test_order_id_zero_rejected(self, book, empty_state):
        order = _buy_order(order_id=0)
        _, result = book.submit_order(empty_state, order)
        assert not result.accepted

    def test_order_id_negative_rejected(self, book, empty_state):
        order = _buy_order(order_id=-1)
        _, result = book.submit_order(empty_state, order)
        assert not result.accepted

    def test_order_id_large_negative_rejected(self, book, empty_state):
        order = _buy_order(order_id=-888888)
        _, result = book.submit_order(empty_state, order)
        assert not result.accepted

    def test_order_id_one_accepted(self, book, empty_state):
        order = _buy_order(order_id=1)
        _, result = book.submit_order(empty_state, order)
        assert result.accepted

    def test_order_id_large_positive_accepted(self, book, empty_state):
        order = _buy_order(order_id=100000)
        _, result = book.submit_order(empty_state, order)
        assert result.accepted


# ---------------------------------------------------------------------------
# Boundary conditions: extreme prices
# ---------------------------------------------------------------------------

class TestBoundaryPrices:
    """Test submission and retrieval at the extreme edges of the OMIE
    price range [-9999.0, +9999.0] (market-spec S2.3)."""

    def test_price_at_floor_accepted(self, book, empty_state):
        order = _buy_order(price=-9999.0, quantity=0.1)
        _, result = book.submit_order(empty_state, order)
        assert result.accepted

    def test_price_at_ceiling_accepted(self, book, empty_state):
        order = _sell_order(price=9999.0, quantity=0.1)
        _, result = book.submit_order(empty_state, order)
        assert result.accepted

    def test_price_just_above_floor_accepted(self, book, empty_state):
        order = _buy_order(price=-9998.99, quantity=0.1)
        _, result = book.submit_order(empty_state, order)
        assert result.accepted

    def test_price_just_below_ceiling_accepted(self, book, empty_state):
        order = _sell_order(price=9998.99, quantity=0.1)
        _, result = book.submit_order(empty_state, order)
        assert result.accepted

    def test_price_below_floor_rejected(self, book, empty_state):
        order = _buy_order(price=-9999.01, quantity=0.1)
        _, result = book.submit_order(empty_state, order)
        assert not result.accepted

    def test_price_above_ceiling_rejected(self, book, empty_state):
        order = _sell_order(price=9999.01, quantity=0.1)
        _, result = book.submit_order(empty_state, order)
        assert not result.accepted


# ---------------------------------------------------------------------------
# Boundary conditions: extreme quantities
# ---------------------------------------------------------------------------

class TestBoundaryQuantities:
    """Test submission and retrieval at the extreme edges of the OMIE
    quantity range [0.1, 999.0] (market-spec S2.3).

    As of P2-L1-HOTFIX-tick-precision (Option C), sub-qty_tick probes
    (e.g. ``0.09``) snap onto ``qty_min`` via the encoder and are accepted.
    The encoder-layer snap contract lives in
    ``tests/omiecid/test_encoding_snap_semantics.py``. Above-maximum and
    zero-quantity rejections remain valid here.
    """

    def test_minimum_quantity_accepted(self, book, empty_state):
        order = _buy_order(quantity=0.1)
        _, result = book.submit_order(empty_state, order)
        assert result.accepted

    def test_maximum_quantity_accepted(self, book, empty_state):
        order = _buy_order(quantity=999.0)
        _, result = book.submit_order(empty_state, order)
        assert result.accepted

    def test_above_maximum_quantity_rejected(self, book, empty_state):
        order = _buy_order(quantity=999.1)
        _, result = book.submit_order(empty_state, order)
        assert not result.accepted

    def test_zero_quantity_rejected(self, book, empty_state):
        order = _buy_order(quantity=0.0)
        _, result = book.submit_order(empty_state, order)
        assert not result.accepted


# ---------------------------------------------------------------------------
# SubmitResult type contract
# ---------------------------------------------------------------------------

class TestSubmitResultType:
    """SubmitResult must be a NamedTuple with accepted: bool."""

    def test_submit_result_is_namedtuple(self):
        r = SubmitResult(accepted=True)
        assert isinstance(r, tuple)

    def test_submit_result_has_accepted_field(self):
        r = SubmitResult(accepted=True)
        assert r.accepted == True

    def test_submit_result_false(self):
        r = SubmitResult(accepted=False)
        assert r.accepted == False

    def test_submit_result_from_submit(self, book, empty_state):
        _, result = book.submit_order(empty_state, _buy_order())
        assert isinstance(result, SubmitResult)
        assert result.accepted == True

    def test_submit_result_from_rejected(self, book, empty_state):
        bad_order = _buy_order(price=-99999.0)
        _, result = book.submit_order(empty_state, bad_order)
        assert isinstance(result, SubmitResult)
        assert result.accepted == False


# ---------------------------------------------------------------------------
# get_order: agent_id and order_id preserved through round-trip
# ---------------------------------------------------------------------------

class TestGetOrderFieldPreservation:
    """get_order must return OMIEOrder with all adapter-translated fields
    correctly round-tripped."""

    def test_agent_id_preserved(self, book, empty_state):
        order = _buy_order(agent_id=42, order_id=7)
        state, _ = book.submit_order(empty_state, order)
        retrieved = book.get_order(state, 7, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.agent_id == 42

    def test_order_id_preserved(self, book, empty_state):
        order = _buy_order(order_id=12345)
        state, _ = book.submit_order(empty_state, order)
        retrieved = book.get_order(state, 12345, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.order_id == 12345

    def test_timestamp_preserved(self, book, empty_state):
        order = _buy_order(timestamp=9999)
        state, _ = book.submit_order(empty_state, order)
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.timestamp == 9999

    def test_side_preserved_for_buy(self, book, empty_state):
        order = _buy_order(order_id=1)
        state, _ = book.submit_order(empty_state, order)
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.side == Side.BUY

    def test_side_preserved_for_sell(self, book, empty_state):
        order = _sell_order(order_id=2)
        state, _ = book.submit_order(empty_state, order)
        retrieved = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(retrieved)
        assert retrieved.side == Side.SELL

    def test_contract_id_preserved(self, book, empty_state):
        order = _buy_order(order_id=1)
        state, _ = book.submit_order(empty_state, order)
        retrieved = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(retrieved)
        assert retrieved.contract_id == 10  # matches the open_contract fixture

    def test_get_order_returns_none_for_nonexistent(self, book, empty_state):
        assert _is_absent(book.get_order(empty_state, 999, Side.BUY))

    def test_get_order_returns_none_for_wrong_side(self, book, empty_state):
        state, _ = book.submit_order(empty_state, _buy_order(order_id=1))
        assert _is_absent(book.get_order(state, 1, Side.SELL))

    def test_get_order_returns_omie_order_type(self, book, empty_state):
        state, _ = book.submit_order(empty_state, _buy_order(order_id=1))
        retrieved = book.get_order(state, 1, Side.BUY)
        assert isinstance(retrieved, OMIEOrder)


# ---------------------------------------------------------------------------
# Functional state: original state unchanged after operations
# ---------------------------------------------------------------------------

class TestFunctionalStateSemantics:
    """ContractBook follows functional state: operations return new state;
    the original state is not mutated."""

    def test_submit_does_not_mutate_original_state(self, book, empty_state):
        new_state, _ = book.submit_order(empty_state, _buy_order(order_id=1))
        # The original empty state should not have the order
        assert _is_absent(book.get_order(empty_state, 1, Side.BUY))
        # The new state should have it
        assert not _is_absent(book.get_order(new_state, 1, Side.BUY))

    def test_cancel_does_not_mutate_original_state(self, book, empty_state):
        state_with_order, _ = book.submit_order(empty_state, _buy_order(order_id=1))
        state_after_cancel = book.cancel_order(state_with_order, 1, Side.BUY)
        # Original state (with order) should still have the order
        assert not _is_absent(book.get_order(state_with_order, 1, Side.BUY))
        # Cancelled state should not
        assert _is_absent(book.get_order(state_after_cancel, 1, Side.BUY))


# ---------------------------------------------------------------------------
# ICE restriction rejected at ContractBook level
# ---------------------------------------------------------------------------

class TestICEOrderAccepted:
    """Valid ICE (iceberg) orders are accepted after M4.1 removes
    M1.3 blanket rejection. Invalid ICE orders are still rejected
    by iceberg validation (peak_qty > 0, peak_qty <= quantity)."""

    def test_ice_buy_order_accepted(self, book, empty_state):
        order = make_order(
            contract_id=10, side=Side.BUY, price=50.0, quantity=10.0,
            order_id=1, agent_id=0, timestamp=100,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=2.0,
        )
        _, result = book.submit_order(empty_state, order)
        assert result.accepted

    def test_ice_sell_order_accepted(self, book, empty_state):
        order = make_order(
            contract_id=10, side=Side.SELL, price=50.0, quantity=10.0,
            order_id=2, agent_id=0, timestamp=100,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=2.0,
        )
        _, result = book.submit_order(empty_state, order)
        assert result.accepted


# ---------------------------------------------------------------------------
# Tick alignment: off-tick orders rejected through ContractBook
# ---------------------------------------------------------------------------

class TestTickAlignmentThroughBook:
    """Verify on-tick acceptance end-to-end through ContractBook.submit_order.

    As of P2-L1-HOTFIX-tick-precision (Option C), sub-half-tick offsets
    snap onto the nearest tick via the encoder; the original off-tick
    rejection probes through the book no longer hold. The encoder-layer
    snap contract lives in ``tests/omiecid/test_encoding_snap_semantics.py``.
    Only the on-tick acceptance sanity checks remain here.
    """

    def test_price_on_tick_accepted(self, book, empty_state):
        order = _buy_order(price=50.01)
        _, result = book.submit_order(empty_state, order)
        assert result.accepted

    def test_quantity_on_tick_accepted(self, book, empty_state):
        order = _buy_order(quantity=1.0)
        _, result = book.submit_order(empty_state, order)
        assert result.accepted


# ---------------------------------------------------------------------------
# Book initialization
# ---------------------------------------------------------------------------

class TestBookInit:
    """ContractBook.init() should produce an empty book state."""

    def test_init_returns_state(self, book):
        state = book.init()
        assert state is not None

    def test_empty_book_get_order_returns_none(self, book, empty_state):
        assert _is_absent(book.get_order(empty_state, 1, Side.BUY))
        assert _is_absent(book.get_order(empty_state, 1, Side.SELL))

    def test_init_can_be_called_multiple_times(self, book):
        s1 = book.init()
        s2 = book.init()
        assert s1 is not None
        assert s2 is not None
