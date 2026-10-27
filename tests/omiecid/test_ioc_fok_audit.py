"""Audit tests for M3.3 -- IOC/FOK Execution Restrictions.

Post-implementation verification focusing on edge cases, boundary conditions,
and acceptance criteria that stress the IOC/FOK implementation:

- IOC with multiple price levels (multi-level walk)
- IOC into exact-match liquidity (full fill, nothing to cancel)
- IOC into empty book and non-crossing prices
- FOK full fill across multiple price levels
- FOK atomicity: resting orders truly untouched on rejection
- FOK exact-quantity boundary (one tick short rejected)
- Interaction with negative prices
- Sequential IOC/FOK operations
- Functional state semantics (original state not mutated)
- Boundary quantities (min qty 0.1 MW, exact qty match)
- SubmitResult field correctness for all IOC/FOK scenarios
- Settlement value V = Q x p x 0.25 for IOC/FOK trades

Derived independently from:
- docs/omiecid/market-spec.md §4.5 (execution restrictions), §6 (matching)
- docs/omiecid/agile-plan.md (M3.3 acceptance criteria)
- docs/omiecid/migration-plan.md §M3 (interface)
- omiecid_simulator/omiecid/contract_book.py (public interface only)
"""

import pytest
from pytest import approx

from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
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




# ---------------------------------------------------------------------------
# Shared fixtures (product_config, market_config from conftest)
# Specialized open_contract overrides conftest (contract_id=42 vs 37)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def open_contract():
    """An OPEN contract suitable for IOC/FOK tests."""
    return Contract(
        contract_id=42,
        delivery_start_minutes=630,   # 10:30
        delivery_end_minutes=645,     # 10:45
        gct_minutes=570,              # 09:30
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

def _buy(contract_id=42, price=50.0, qty=1.0, order_id=1, agent_id=100,
         timestamp=1000, restriction=ExecRestriction.NON):
    return make_order(
        contract_id=contract_id, side=Side.BUY, price=price, quantity=qty,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=restriction,
    )


def _sell(contract_id=42, price=50.0, qty=1.0, order_id=2, agent_id=200,
          timestamp=2000, restriction=ExecRestriction.NON):
    return make_order(
        contract_id=contract_id, side=Side.SELL, price=price, quantity=qty,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=restriction,
    )


def _ioc_buy(contract_id=42, price=50.0, qty=1.0, order_id=10, agent_id=100,
             timestamp=5000):
    return _buy(contract_id=contract_id, price=price, qty=qty,
                order_id=order_id, agent_id=agent_id, timestamp=timestamp,
                restriction=ExecRestriction.IOC)


def _ioc_sell(contract_id=42, price=50.0, qty=1.0, order_id=11, agent_id=200,
              timestamp=5000):
    return _sell(contract_id=contract_id, price=price, qty=qty,
                 order_id=order_id, agent_id=agent_id, timestamp=timestamp,
                 restriction=ExecRestriction.IOC)


def _fok_buy(contract_id=42, price=50.0, qty=1.0, order_id=20, agent_id=100,
             timestamp=6000):
    return _buy(contract_id=contract_id, price=price, qty=qty,
                order_id=order_id, agent_id=agent_id, timestamp=timestamp,
                restriction=ExecRestriction.FOK)


def _fok_sell(contract_id=42, price=50.0, qty=1.0, order_id=21, agent_id=200,
              timestamp=6000):
    return _sell(contract_id=contract_id, price=price, qty=qty,
                 order_id=order_id, agent_id=agent_id, timestamp=timestamp,
                 restriction=ExecRestriction.FOK)


# ===========================================================================
# 1. IOC Multi-Level Walk
# ===========================================================================

class TestIOCMultiLevelWalk:
    """IOC orders that walk through multiple price levels."""

    def test_ioc_buy_walks_two_ask_levels(self, book, empty_state):
        """IOC buy for 2.0 MW walks through asks at 50.0 (1.0) and 51.0 (1.0)."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=51.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _ioc_buy(price=52.0, qty=2.0))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)

    def test_ioc_buy_walks_two_levels_no_remainder_in_book(self, book, empty_state):
        """IOC buy fully filled across 2 levels -> not in book."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=51.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _ioc_buy(price=52.0, qty=2.0))
        assert _is_absent(book.get_order(state, 10, Side.BUY))

    def test_ioc_buy_walks_partial_second_level(self, book, empty_state):
        """IOC buy for 1.5 walks 50.0 (1.0) fully, 51.0 (1.0) partially -> 1.5 filled."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=51.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _ioc_buy(price=52.0, qty=1.5))
        assert result.traded == True
        assert result.trade_qty == approx(1.5)
        # Remainder of second ask still in book
        remaining = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(0.5)

    def test_ioc_buy_partial_walk_remainder_cancelled(self, book, empty_state):
        """IOC buy for 3.0 but only 2.0 available -> 2.0 filled, no IOC in book."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=51.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _ioc_buy(price=52.0, qty=3.0))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)
        assert _is_absent(book.get_order(state, 10, Side.BUY))

    def test_ioc_sell_walks_two_bid_levels(self, book, empty_state):
        """IOC sell for 2.0 MW walks through bids at 51.0 (1.0) and 50.0 (1.0)."""
        state, _ = book.submit_order(empty_state, _buy(price=51.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _buy(price=50.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _ioc_sell(price=49.0, qty=2.0))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)

    def test_ioc_sell_partial_walk_remainder_cancelled(self, book, empty_state):
        """IOC sell for 3.0 but only 1.5 available -> 1.5 filled, no IOC in book."""
        state, _ = book.submit_order(empty_state, _buy(price=51.0, qty=1.5, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_sell(price=49.0, qty=3.0))
        assert result.traded == True
        assert result.trade_qty == approx(1.5)
        assert _is_absent(book.get_order(state, 11, Side.SELL))

    def test_ioc_buy_three_levels_partial_fill(self, book, empty_state):
        """IOC buy walks 3 ask levels, each 0.5 MW, buying 1.2 -> fills 1.2."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=0.5, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=50.5, qty=0.5, order_id=2, timestamp=200))
        state, _ = book.submit_order(state, _sell(price=51.0, qty=0.5, order_id=3, timestamp=300))
        state, result = book.submit_order(state, _ioc_buy(price=51.0, qty=1.2))
        assert result.traded == True
        assert result.trade_qty == approx(1.2)
        assert _is_absent(book.get_order(state, 10, Side.BUY))


# ===========================================================================
# 2. IOC Exact Match (full fill, nothing to cancel)
# ===========================================================================

class TestIOCExactMatch:
    """IOC orders that exactly match available liquidity."""

    def test_ioc_buy_exact_match_single_level(self, book, empty_state):
        """IOC buy 1.0 into exactly 1.0 ask -> full fill, not in book."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=50.0, qty=1.0))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        assert _is_absent(book.get_order(state, 10, Side.BUY))
        assert _is_absent(book.get_order(state, 1, Side.SELL))

    def test_ioc_sell_exact_match_single_level(self, book, empty_state):
        """IOC sell 1.0 into exactly 1.0 bid -> full fill, not in book."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_sell(price=50.0, qty=1.0))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        assert _is_absent(book.get_order(state, 11, Side.SELL))
        assert _is_absent(book.get_order(state, 1, Side.BUY))

    def test_ioc_buy_exact_match_two_levels(self, book, empty_state):
        """IOC buy 2.0 into asks of 1.0@50 + 1.0@51 -> exact full fill."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=51.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _ioc_buy(price=51.0, qty=2.0))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)
        assert _is_absent(book.get_order(state, 10, Side.BUY))

    def test_ioc_minimum_qty_exact_match(self, book, empty_state):
        """IOC buy 0.1 (min qty) into 0.1 ask -> full fill."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=0.1, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=50.0, qty=0.1))
        assert result.traded == True
        assert result.trade_qty == approx(0.1)


# ===========================================================================
# 3. IOC No Liquidity / Non-Crossing
# ===========================================================================

class TestIOCNoLiquidity:
    """IOC orders when no matchable liquidity exists."""

    def test_ioc_buy_empty_book_accepted_no_trade(self, book, empty_state):
        """IOC buy into empty book: accepted, not traded, not in book."""
        state, result = book.submit_order(empty_state, _ioc_buy(price=50.0))
        assert result.accepted == True
        assert result.traded == False
        assert result.trade_qty == approx(0.0)
        assert _is_absent(book.get_order(state, 10, Side.BUY))

    def test_ioc_sell_empty_book_accepted_no_trade(self, book, empty_state):
        """IOC sell into empty book: accepted, not traded, not in book."""
        state, result = book.submit_order(empty_state, _ioc_sell(price=50.0))
        assert result.accepted == True
        assert result.traded == False
        assert _is_absent(book.get_order(state, 11, Side.SELL))

    def test_ioc_buy_non_crossing_price(self, book, empty_state):
        """IOC buy at 49 when ask is at 50 -> no trade, cancelled."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=49.0))
        assert result.traded == False
        assert _is_absent(book.get_order(state, 10, Side.BUY))
        # Resting sell should still be in book
        assert not _is_absent(book.get_order(state, 1, Side.SELL))

    def test_ioc_sell_non_crossing_price(self, book, empty_state):
        """IOC sell at 51 when bid is at 50 -> no trade, cancelled."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_sell(price=51.0))
        assert result.traded == False
        assert _is_absent(book.get_order(state, 11, Side.SELL))
        assert not _is_absent(book.get_order(state, 1, Side.BUY))


# ===========================================================================
# 4. FOK Full Fill — Single and Multi-Level
# ===========================================================================

class TestFOKFullFill:
    """FOK orders that are fully filled."""

    def test_fok_buy_single_level_full_fill(self, book, empty_state):
        """FOK buy 1.0 into 1.0 ask -> full fill."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=50.0, qty=1.0))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        assert _is_absent(book.get_order(state, 20, Side.BUY))

    def test_fok_sell_single_level_full_fill(self, book, empty_state):
        """FOK sell 1.0 into 1.0 bid -> full fill."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_sell(price=50.0, qty=1.0))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        assert _is_absent(book.get_order(state, 21, Side.SELL))

    def test_fok_buy_two_levels_full_fill(self, book, empty_state):
        """FOK buy 2.0 into asks of 1.0@50 + 1.0@51 -> full fill across levels."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=51.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _fok_buy(price=51.0, qty=2.0))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)

    def test_fok_sell_two_levels_full_fill(self, book, empty_state):
        """FOK sell 2.0 into bids of 1.0@51 + 1.0@50 -> full fill across levels."""
        state, _ = book.submit_order(empty_state, _buy(price=51.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _buy(price=50.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _fok_sell(price=50.0, qty=2.0))
        assert result.traded == True
        assert result.trade_qty == approx(2.0)

    def test_fok_buy_excess_liquidity_leaves_residual_resting(self, book, empty_state):
        """FOK buy 1.0 into 2.0 ask -> fills 1.0, resting has 1.0 left."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=2.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=50.0, qty=1.0))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        remaining = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(1.0)

    def test_fok_buy_three_levels(self, book, empty_state):
        """FOK buy 3.0 across three ask levels (1.0 each at 50, 51, 52)."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=51.0, qty=1.0, order_id=2, timestamp=200))
        state, _ = book.submit_order(state, _sell(price=52.0, qty=1.0, order_id=3, timestamp=300))
        state, result = book.submit_order(state, _fok_buy(price=52.0, qty=3.0))
        assert result.traded == True
        assert result.trade_qty == approx(3.0)

    def test_fok_minimum_qty_full_fill(self, book, empty_state):
        """FOK buy 0.1 (min qty) into 0.1 ask -> full fill."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=0.1, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=50.0, qty=0.1))
        assert result.traded == True
        assert result.trade_qty == approx(0.1)


# ===========================================================================
# 5. FOK Atomicity — Resting Orders Untouched on Rejection
# ===========================================================================

class TestFOKAtomicity:
    """FOK rejection must leave resting orders completely untouched."""

    def test_fok_rejected_resting_ask_unchanged(self, book, empty_state):
        """FOK buy 2.0 but only 1.0 available -> resting 1.0 ask untouched."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state_before = state
        state, result = book.submit_order(state, _fok_buy(price=50.0, qty=2.0))
        assert result.traded == False
        resting = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(resting)
        assert resting.quantity == approx(1.0)

    def test_fok_rejected_resting_bid_unchanged(self, book, empty_state):
        """FOK sell 2.0 but only 1.0 available -> resting 1.0 bid untouched."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_sell(price=50.0, qty=2.0))
        assert result.traded == False
        resting = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(resting)
        assert resting.quantity == approx(1.0)

    def test_fok_rejected_multiple_resting_all_untouched(self, book, empty_state):
        """FOK buy 5.0 but only 3.0 across 3 levels -> all 3 resting untouched."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=51.0, qty=1.0, order_id=2, timestamp=200))
        state, _ = book.submit_order(state, _sell(price=52.0, qty=1.0, order_id=3, timestamp=300))
        state, result = book.submit_order(state, _fok_buy(price=52.0, qty=5.0))
        assert result.traded == False
        for oid in [1, 2, 3]:
            resting = book.get_order(state, oid, Side.SELL)
            assert not _is_absent(resting), f"Resting order {oid} should still be in book"
            assert resting.quantity == approx(1.0), f"Resting order {oid} qty should be 1.0"

    def test_fok_rejected_does_not_consume_any_liquidity(self, book, empty_state):
        """After FOK rejection, a subsequent matching order can fill against all resting."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        # FOK for 2.0 rejected (only 1.0 available)
        state, result = book.submit_order(state, _fok_buy(price=50.0, qty=2.0))
        assert result.traded == False
        # Now a NON buy for 1.0 should fill against the untouched ask
        state, result2 = book.submit_order(state, _buy(price=50.0, qty=1.0, order_id=30, timestamp=7000))
        assert result2.traded == True
        assert result2.trade_qty == approx(1.0)

    def test_fok_rejected_empty_book(self, book, empty_state):
        """FOK buy into empty book -> rejected, accepted=True, traded=False."""
        state, result = book.submit_order(empty_state, _fok_buy(price=50.0, qty=1.0))
        assert result.accepted == True
        assert result.traded == False
        assert result.trade_qty == approx(0.0)

    def test_fok_rejected_non_crossing(self, book, empty_state):
        """FOK buy at 49 when ask at 50 -> rejected (no crossing liquidity)."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=49.0, qty=1.0))
        assert result.traded == False
        assert not _is_absent(book.get_order(state, 1, Side.SELL))

    def test_fok_sell_rejected_non_crossing(self, book, empty_state):
        """FOK sell at 51 when bid at 50 -> rejected."""
        state, _ = book.submit_order(empty_state, _buy(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_sell(price=51.0, qty=1.0))
        assert result.traded == False
        assert not _is_absent(book.get_order(state, 1, Side.BUY))

    def test_fok_rejected_one_tick_short(self, book, empty_state):
        """FOK buy 1.1 but only 1.0 available -> rejected (one tick short on qty)."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=50.0, qty=1.1))
        assert result.traded == False
        resting = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(resting)
        assert resting.quantity == approx(1.0)


# ===========================================================================
# 6. Negative Prices
# ===========================================================================

class TestNegativePrices:
    """IOC/FOK behavior with negative EUR/MWh prices."""

    def test_ioc_buy_negative_price(self, book, empty_state):
        """IOC buy at -10.0 into ask at -15.0 (buy >= sell) -> trade."""
        state, _ = book.submit_order(empty_state, _sell(price=-15.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=-10.0, qty=1.0))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        assert result.trade_price == approx(-15.0)

    def test_ioc_sell_negative_price(self, book, empty_state):
        """IOC sell at -20.0 into bid at -10.0 (bid >= sell) -> trade."""
        state, _ = book.submit_order(empty_state, _buy(price=-10.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_sell(price=-20.0, qty=1.0))
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        assert result.trade_price == approx(-10.0)

    def test_fok_buy_negative_price(self, book, empty_state):
        """FOK buy at -5.0 into ask at -10.0 -> trade at -10.0 (resting price)."""
        state, _ = book.submit_order(empty_state, _sell(price=-10.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=-5.0, qty=1.0))
        assert result.traded == True
        assert result.trade_price == approx(-10.0)

    def test_fok_sell_negative_price_rejected(self, book, empty_state):
        """FOK sell at -5.0 when bid at -10.0 -> rejected (sell > bid)."""
        state, _ = book.submit_order(empty_state, _buy(price=-10.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_sell(price=-5.0, qty=1.0))
        assert result.traded == False

    def test_ioc_negative_price_settlement_value(self, book, empty_state):
        """Settlement for negative price: V = 1.0 * (-15.0) * 0.25 = -3.75."""
        state, _ = book.submit_order(empty_state, _sell(price=-15.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=-10.0, qty=1.0))
        assert result.settlement_value == approx(-15.0 * 1.0 * 0.25)

    def test_fok_negative_price_settlement_value(self, book, empty_state):
        """Settlement for negative price FOK: V = Q * p * 0.25."""
        state, _ = book.submit_order(empty_state, _sell(price=-20.0, qty=2.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=-15.0, qty=2.0))
        assert result.traded == True
        assert result.settlement_value == approx(2.0 * -20.0 * 0.25)


# ===========================================================================
# 7. Sequential IOC/FOK Operations
# ===========================================================================

class TestSequentialOperations:
    """Multiple IOC/FOK orders submitted in sequence."""

    def test_two_ioc_buys_drain_liquidity(self, book, empty_state):
        """First IOC fills 1.0, second IOC has nothing left -> no trade."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, r1 = book.submit_order(state, _ioc_buy(price=50.0, qty=1.0, order_id=10, timestamp=5000))
        assert r1.traded == True
        state, r2 = book.submit_order(state, _ioc_buy(price=50.0, qty=1.0, order_id=12, timestamp=5001))
        assert r2.traded == False
        assert r2.trade_qty == approx(0.0)

    def test_fok_after_ioc_partial_drain(self, book, empty_state):
        """IOC drains 1.0 from 2.0 ask, then FOK for 1.0 fills remaining."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=2.0, order_id=1, timestamp=100))
        state, r1 = book.submit_order(state, _ioc_buy(price=50.0, qty=1.0, order_id=10, timestamp=5000))
        assert r1.traded == True
        state, r2 = book.submit_order(state, _fok_buy(price=50.0, qty=1.0, order_id=20, timestamp=6000))
        assert r2.traded == True
        assert r2.trade_qty == approx(1.0)

    def test_fok_rejected_then_ioc_partial(self, book, empty_state):
        """FOK for 2.0 rejected (only 1.0), then IOC for 2.0 fills 1.0."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, r1 = book.submit_order(state, _fok_buy(price=50.0, qty=2.0, order_id=20, timestamp=6000))
        assert r1.traded == False
        state, r2 = book.submit_order(state, _ioc_buy(price=50.0, qty=2.0, order_id=10, timestamp=7000))
        assert r2.traded == True
        assert r2.trade_qty == approx(1.0)

    def test_non_order_after_ioc(self, book, empty_state):
        """After IOC removes itself, a NON order enters the book normally."""
        state, _ = book.submit_order(empty_state, _ioc_buy(price=50.0, qty=1.0, order_id=10, timestamp=5000))
        # IOC had nothing to fill, cancelled -> not in book
        assert _is_absent(book.get_order(state, 10, Side.BUY))
        # NON order enters book
        state, result = book.submit_order(state, _buy(price=50.0, qty=1.0, order_id=30, timestamp=7000))
        assert result.accepted == True
        assert not _is_absent(book.get_order(state, 30, Side.BUY))

    def test_non_order_after_fok_rejection(self, book, empty_state):
        """After FOK rejection, a NON order enters the book normally."""
        state, result = book.submit_order(empty_state, _fok_buy(price=50.0, qty=1.0))
        assert result.traded == False
        state, result2 = book.submit_order(state, _buy(price=50.0, qty=1.0, order_id=30, timestamp=7000))
        assert result2.accepted == True
        assert not _is_absent(book.get_order(state, 30, Side.BUY))

    def test_ioc_does_not_affect_same_side_orders(self, book, empty_state):
        """IOC buy does not disturb existing NON buy orders."""
        state, _ = book.submit_order(empty_state, _buy(price=48.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _ioc_buy(price=50.0, qty=1.0, order_id=10, timestamp=5000))
        # Existing buy should still be there
        assert not _is_absent(book.get_order(state, 1, Side.BUY))
        assert book.get_order(state, 1, Side.BUY).quantity == approx(1.0)

    def test_fok_rejected_does_not_affect_same_side_orders(self, book, empty_state):
        """FOK buy rejected doesn't disturb existing NON buy orders."""
        state, _ = book.submit_order(empty_state, _buy(price=48.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _fok_buy(price=50.0, qty=1.0))
        assert not _is_absent(book.get_order(state, 1, Side.BUY))
        assert book.get_order(state, 1, Side.BUY).quantity == approx(1.0)


# ===========================================================================
# 8. Functional State Semantics (Immutability)
# ===========================================================================

class TestFunctionalStateSemantics:
    """Original state must not be mutated by IOC/FOK operations."""

    def test_ioc_does_not_mutate_original_state(self, book, empty_state):
        """Submitting an IOC does not change the state object passed in."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state_snapshot = state
        new_state, result = book.submit_order(state, _ioc_buy(price=50.0, qty=1.0))
        # Original state should still have the sell order
        assert not _is_absent(book.get_order(state_snapshot, 1, Side.SELL))

    def test_fok_rejected_does_not_mutate_state(self, book, empty_state):
        """FOK rejection returns the same state unchanged."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state_snapshot = state
        new_state, result = book.submit_order(state, _fok_buy(price=50.0, qty=2.0))
        assert result.traded == False
        # Original state should still have the sell order
        assert not _is_absent(book.get_order(state_snapshot, 1, Side.SELL))

    def test_ioc_empty_book_does_not_mutate_state(self, book, empty_state):
        """IOC into empty book returns a valid state; original empty_state intact."""
        new_state, result = book.submit_order(empty_state, _ioc_buy(price=50.0))
        assert result.accepted == True
        # empty_state should still be usable
        state2, r2 = book.submit_order(empty_state, _sell(price=50.0, order_id=30, timestamp=100))
        assert r2.accepted == True


# ===========================================================================
# 9. SubmitResult Field Completeness
# ===========================================================================

class TestSubmitResultFields:
    """Verify all 5 SubmitResult fields for IOC/FOK scenarios."""

    def test_ioc_partial_fill_result_fields(self, book, empty_state):
        """IOC partial fill: accepted=True, traded=True, qty/price/settlement set."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=0.5, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=50.0, qty=1.0))
        assert result.accepted == True
        assert result.traded == True
        assert result.trade_qty == approx(0.5)
        assert result.trade_price == approx(50.0)
        assert result.settlement_value == approx(0.5 * 50.0 * 0.25)

    def test_ioc_no_fill_result_fields(self, book, empty_state):
        """IOC no fill: accepted=True, traded=False, qty/price/settlement=0."""
        state, result = book.submit_order(empty_state, _ioc_buy(price=50.0))
        assert result.accepted == True
        assert result.traded == False
        assert result.trade_qty == approx(0.0)
        assert result.trade_price == approx(0.0)
        assert result.settlement_value == approx(0.0)

    def test_fok_full_fill_result_fields(self, book, empty_state):
        """FOK full fill: accepted=True, traded=True, qty/price/settlement set."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=50.0, qty=1.0))
        assert result.accepted == True
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        assert result.trade_price == approx(50.0)
        assert result.settlement_value == approx(1.0 * 50.0 * 0.25)

    def test_fok_rejected_result_fields(self, book, empty_state):
        """FOK rejected: accepted=True, traded=False, qty/price/settlement=0."""
        state, result = book.submit_order(empty_state, _fok_buy(price=50.0))
        assert result.accepted == True
        assert result.traded == False
        assert result.trade_qty == approx(0.0)
        assert result.trade_price == approx(0.0)
        assert result.settlement_value == approx(0.0)

    def test_ioc_full_fill_result_fields(self, book, empty_state):
        """IOC full fill: same result shape as partial but qty == order qty."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=50.0, qty=1.0))
        assert result.accepted == True
        assert result.traded == True
        assert result.trade_qty == approx(1.0)
        assert result.trade_price == approx(50.0)
        assert result.settlement_value == approx(1.0 * 50.0 * 0.25)

    def test_submit_result_is_namedtuple(self, book, empty_state):
        """SubmitResult is indexable and has named fields."""
        state, result = book.submit_order(empty_state, _ioc_buy(price=50.0))
        assert result[0] == True  # accepted at index 0 (jnp.bool_ post-P2-H2)
        assert hasattr(result, 'accepted')
        assert hasattr(result, 'traded')
        assert hasattr(result, 'trade_qty')
        assert hasattr(result, 'trade_price')
        assert hasattr(result, 'settlement_value')


# ===========================================================================
# 10. Settlement Value — IOC/FOK Specific
# ===========================================================================

class TestSettlementValue:
    """V = Q x p x 0.25 for IOC and FOK trades."""

    def test_ioc_partial_fill_settlement(self, book, empty_state):
        """IOC partial: V = 0.5 * 50.0 * 0.25 = 6.25."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=0.5, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=50.0, qty=1.0))
        assert result.settlement_value == approx(0.5 * 50.0 * 0.25)

    def test_fok_full_fill_settlement(self, book, empty_state):
        """FOK full: V = 2.0 * 60.0 * 0.25 = 30.0."""
        state, _ = book.submit_order(empty_state, _sell(price=60.0, qty=2.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=60.0, qty=2.0))
        assert result.settlement_value == approx(2.0 * 60.0 * 0.25)

    def test_ioc_multi_level_settlement_uses_vwap(self, book, empty_state):
        """IOC across 2 levels: V = total_qty * vwap * 0.25."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=52.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _ioc_buy(price=52.0, qty=2.0))
        vwap = (50.0 * 1.0 + 52.0 * 1.0) / 2.0  # 51.0
        assert result.trade_qty == approx(2.0)
        assert result.settlement_value == approx(2.0 * vwap * 0.25)

    def test_fok_multi_level_settlement_uses_vwap(self, book, empty_state):
        """FOK across 2 levels: V = total_qty * vwap * 0.25."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=1.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=52.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _fok_buy(price=52.0, qty=2.0))
        vwap = (50.0 * 1.0 + 52.0 * 1.0) / 2.0
        assert result.settlement_value == approx(2.0 * vwap * 0.25)

    def test_ioc_minimum_qty_settlement(self, book, empty_state):
        """V = 0.1 * 50.0 * 0.25 = 1.25 for minimum qty IOC."""
        state, _ = book.submit_order(empty_state, _sell(price=50.0, qty=0.1, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=50.0, qty=0.1))
        assert result.settlement_value == approx(0.1 * 50.0 * 0.25)

    def test_ioc_no_fill_settlement_zero(self, book, empty_state):
        """IOC with no fill: settlement = 0."""
        state, result = book.submit_order(empty_state, _ioc_buy(price=50.0))
        assert result.settlement_value == approx(0.0)

    def test_fok_rejected_settlement_zero(self, book, empty_state):
        """FOK rejected: settlement = 0."""
        state, result = book.submit_order(empty_state, _fok_buy(price=50.0))
        assert result.settlement_value == approx(0.0)

    def test_ioc_multi_level_unequal_qty_settlement(self, book, empty_state):
        """IOC across 2 levels with unequal qtys: VWAP weighted correctly."""
        state, _ = book.submit_order(empty_state, _sell(price=40.0, qty=3.0, order_id=1, timestamp=100))
        state, _ = book.submit_order(state, _sell(price=50.0, qty=1.0, order_id=2, timestamp=200))
        state, result = book.submit_order(state, _ioc_buy(price=50.0, qty=4.0))
        vwap = (40.0 * 3.0 + 50.0 * 1.0) / 4.0  # 42.5
        assert result.trade_qty == approx(4.0)
        assert result.settlement_value == approx(4.0 * vwap * 0.25)


# ===========================================================================
# 11. Boundary Prices
# ===========================================================================

class TestBoundaryPrices:
    """IOC/FOK at price bounds."""

    def test_ioc_buy_at_price_ceiling(self, book, empty_state):
        """IOC buy at +9999.0 (max) into ask at +9999.0 -> trade."""
        state, _ = book.submit_order(empty_state, _sell(price=9999.0, qty=0.1, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=9999.0, qty=0.1))
        assert result.traded == True

    def test_ioc_sell_at_price_floor(self, book, empty_state):
        """IOC sell at -9999.0 (min) into bid at -9999.0 -> trade."""
        state, _ = book.submit_order(empty_state, _buy(price=-9999.0, qty=0.1, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_sell(price=-9999.0, qty=0.1))
        assert result.traded == True

    def test_fok_buy_at_price_ceiling(self, book, empty_state):
        """FOK buy at +9999.0 into ask at +9999.0 -> full fill."""
        state, _ = book.submit_order(empty_state, _sell(price=9999.0, qty=0.1, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=9999.0, qty=0.1))
        assert result.traded == True
        assert result.trade_qty == approx(0.1)

    def test_fok_sell_at_price_floor(self, book, empty_state):
        """FOK sell at -9999.0 into bid at -9999.0 -> full fill."""
        state, _ = book.submit_order(empty_state, _buy(price=-9999.0, qty=0.1, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_sell(price=-9999.0, qty=0.1))
        assert result.traded == True
        assert result.trade_qty == approx(0.1)

    def test_ioc_at_zero_price(self, book, empty_state):
        """IOC buy at 0.0 into ask at 0.0 -> trade, settlement = 0."""
        state, _ = book.submit_order(empty_state, _sell(price=0.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _ioc_buy(price=0.0, qty=1.0))
        assert result.traded == True
        assert result.settlement_value == approx(0.0)

    def test_fok_at_zero_price(self, book, empty_state):
        """FOK buy at 0.0 into ask at 0.0 -> trade, settlement = 0."""
        state, _ = book.submit_order(empty_state, _sell(price=0.0, qty=1.0, order_id=1, timestamp=100))
        state, result = book.submit_order(state, _fok_buy(price=0.0, qty=1.0))
        assert result.traded == True
        assert result.settlement_value == approx(0.0)
