"""Audit tests for M6.4 -- Multi-Slice Iceberg Matching Per Order.

Independent audit-phase tests derived from:
  - docs/omiecid/market-spec.md SS4.2.1 (Rule 2: multi-slice matching per aggressive order)
  - docs/omiecid/market-spec.md SS5.1.2 (single incoming order matches multiple iceberg slices)
  - docs/omiecid/market-spec.md SS5.2 (timestamp rules for iceberg slices)
  - docs/omiecid/market-spec.md SS6.2 (regular matching rules, iceberg rule SS6.2.7)
  - docs/omiecid/agile-plan.md M6.4 (acceptance criteria)

Acceptance criteria under test:
  AC-1: Single BUY 10 MW fills entire iceberg (peak=3, total=10) in one submit_order() call.
  AC-2: IOC 6 MW against iceberg (peak=2, total=7) fills 6 MW across 3 slices.
  AC-3: PPD/ICEwPI price improvement applies per slice within the loop.
  AC-4: All existing tests updated and passing (not directly testable here).
  AC-5: market-spec SS4.2.1 Rule 2 updated (not directly testable here).

Focus areas for this audit:
  - Boundary conditions around iceberg total quantity divisibility
  - ICEwPI stop-at-non-crossing for NON and IOC paths
  - Settlement value accumulation correctness
  - Aggressive order residual behavior (NON rests, IOC cancels)
  - Multiple icebergs drained by one aggressive order
  - Symmetry: SELL aggressive vs BUY iceberg
  - Edge case: aggressive qty exactly equals peak (single slice, no loop)
  - Edge case: aggressive qty equals zero hidden (peak == total)
  - FOK regression (multi-slice FOK still works)
"""

import pytest

from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    make_order,
)
from omiecid_simulator.omiecid.contract_book import SubmitResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ice_sell(order_id=10, price=50.00, quantity=10.0, peak_qty=3.0,
              agent_id=1, timestamp=100, ppd=0.0):
    """Standard or ICEwPI SELL iceberg."""
    return make_order(
        contract_id=37, side=Side.SELL, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.ICE, peak_qty=peak_qty, ppd=ppd,
    )


def _ice_buy(order_id=20, price=50.00, quantity=10.0, peak_qty=3.0,
             agent_id=1, timestamp=100, ppd=0.0):
    """Standard or ICEwPI BUY iceberg."""
    return make_order(
        contract_id=37, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.ICE, peak_qty=peak_qty, ppd=ppd,
    )


def _non_buy(order_id=50, price=50.00, quantity=10.0,
             agent_id=2, timestamp=200):
    """NON BUY aggressive order."""
    return make_order(
        contract_id=37, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.NON,
    )


def _non_sell(order_id=60, price=50.00, quantity=10.0,
              agent_id=2, timestamp=200):
    """NON SELL aggressive order."""
    return make_order(
        contract_id=37, side=Side.SELL, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.NON,
    )


def _ioc_buy(order_id=70, price=50.00, quantity=6.0,
             agent_id=2, timestamp=200):
    """IOC BUY aggressive order."""
    return make_order(
        contract_id=37, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.IOC,
    )


def _ioc_sell(order_id=80, price=50.00, quantity=6.0,
              agent_id=2, timestamp=200):
    """IOC SELL aggressive order."""
    return make_order(
        contract_id=37, side=Side.SELL, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.IOC,
    )


def _fok_buy(order_id=90, price=50.00, quantity=10.0,
             agent_id=2, timestamp=200):
    """FOK BUY aggressive order."""
    return make_order(
        contract_id=37, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.FOK,
    )


# ---------------------------------------------------------------------------
# AC-1: NON BUY drains entire iceberg in one submit_order() call
# ---------------------------------------------------------------------------

class TestAC1_NON_FullDrain:
    """AC-1: Single BUY 10 MW fills entire iceberg (peak=3, total=10)."""

    def test_non_buy_10mw_drains_iceberg_peak3_total10(self, book, empty_state):
        """Exact acceptance criterion from agile-plan."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=10.0, price=50.00)
        state, result = book.submit_order(state, buy)

        assert result.accepted == True
        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=1e-6)

    def test_non_buy_drains_iceberg_last_slice_partial(self, book, empty_state):
        """10 MW with peak=3: slices are 3+3+3+1. Last slice is partial (1 MW)."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=10.0, price=50.00)
        state, result = book.submit_order(state, buy)

        # All 10 MW consumed
        assert result.trade_qty == pytest.approx(10.0, abs=1e-6)
        # Settlement = 10 * 50 * 0.25 = 125.0
        assert result.settlement_value == pytest.approx(125.0, abs=0.01)

    def test_non_buy_9mw_drains_iceberg_peak3_total9(self, book, empty_state):
        """Evenly divisible: peak=3, total=9 -> 3 slices of 3."""
        iceberg = _ice_sell(quantity=9.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=9.0, price=50.00)
        state, result = book.submit_order(state, buy)

        assert result.trade_qty == pytest.approx(9.0, abs=1e-6)

    def test_non_buy_peak_equals_total_single_slice(self, book, empty_state):
        """Degenerate iceberg: peak=total=5 MW. No hidden quantity.
        Should fill in a single dispatch, no regeneration loop needed."""
        iceberg = _ice_sell(quantity=5.0, peak_qty=5.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=5.0, price=50.00)
        state, result = book.submit_order(state, buy)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(5.0, abs=1e-6)

    def test_non_buy_minimum_peak_01mw(self, book, empty_state):
        """Minimum peak 0.1 MW, total 0.5 MW -> 5 slices of 0.1."""
        iceberg = _ice_sell(quantity=0.5, peak_qty=0.1, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=0.5, price=50.00)
        state, result = book.submit_order(state, buy)

        assert result.trade_qty == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# AC-2: IOC fills across slices
# ---------------------------------------------------------------------------

class TestAC2_IOC_MultiSlice:
    """AC-2: IOC 6 MW against iceberg (peak=2, total=7) fills 6 MW across 3 slices."""

    def test_ioc_6mw_fills_3_slices(self, book, empty_state):
        """Exact acceptance criterion: IOC BUY 6 MW, iceberg peak=2 total=7."""
        iceberg = _ice_sell(quantity=7.0, peak_qty=2.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        ioc = _ioc_buy(quantity=6.0, price=50.00)
        state, result = book.submit_order(state, ioc)

        assert result.accepted == True
        assert result.traded == True
        assert result.trade_qty == pytest.approx(6.0, abs=1e-6)

    def test_ioc_remainder_cancelled_after_iceberg_exhausted(self, book, empty_state):
        """IOC 10 MW vs iceberg total=7 peak=2: fills 7 MW, cancels 3 MW remainder."""
        iceberg = _ice_sell(quantity=7.0, peak_qty=2.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        ioc = _ioc_buy(quantity=10.0, price=50.00)
        state, result = book.submit_order(state, ioc)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(7.0, abs=1e-6)

    def test_ioc_no_crossing_no_fill(self, book, empty_state):
        """IOC BUY at 49.00 vs iceberg SELL at 50.00: no crossing, no fill."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        ioc = _ioc_buy(quantity=5.0, price=49.00)
        state, result = book.submit_order(state, ioc)

        assert result.traded == False
        assert result.trade_qty == pytest.approx(0.0, abs=1e-6)

    def test_ioc_exact_iceberg_drain(self, book, empty_state):
        """IOC qty exactly equals iceberg total: full drain, zero cancelled."""
        iceberg = _ice_sell(quantity=6.0, peak_qty=2.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        ioc = _ioc_buy(quantity=6.0, price=50.00)
        state, result = book.submit_order(state, ioc)

        assert result.trade_qty == pytest.approx(6.0, abs=1e-6)


# ---------------------------------------------------------------------------
# AC-3: ICEwPI price improvement per slice
# ---------------------------------------------------------------------------

class TestAC3_ICEwPI_PricePerSlice:
    """AC-3: PPD/ICEwPI price improvement applies per slice within the loop."""

    def test_sell_icewpi_ppd_positive_stops_at_noncrossing(self, book, empty_state):
        """SELL iceberg ppd=+1.00, base=50, peak=2, total=8.
        BUY NON at 52.00 for 8 MW.
        Slice prices: 50.00, 51.00, 52.00, 53.00.
        Only 3 slices cross (50, 51, 52 = 6 MW). Slice at 53 does not cross.
        NON fills 6 MW and remainder 2 MW rests as bid."""
        iceberg = _ice_sell(quantity=8.0, peak_qty=2.0, price=50.00, ppd=1.0)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=8.0, price=52.00)
        state, result = book.submit_order(state, buy)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(6.0, abs=1e-6)

    def test_buy_icewpi_ppd_negative_stops_at_noncrossing(self, book, empty_state):
        """BUY iceberg ppd=-1.00, base=52, peak=2, total=8.
        SELL NON at 50.00 for 8 MW.
        Slice prices: 52.00, 51.00, 50.00, 49.00.
        Only 3 slices cross (52, 51, 50 = 6 MW). Slice at 49 does not cross.
        NON SELL fills 6 MW, remainder rests as ask."""
        iceberg = _ice_buy(quantity=8.0, peak_qty=2.0, price=52.00, ppd=-1.0)
        state, _ = book.submit_order(empty_state, iceberg)

        sell = _non_sell(quantity=8.0, price=50.00)
        state, result = book.submit_order(state, sell)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(6.0, abs=1e-6)

    def test_ioc_icewpi_stops_and_cancels(self, book, empty_state):
        """IOC BUY 8 MW vs ICEwPI SELL base=50 ppd=+1.0, peak=2, total=8.
        BUY limit = 51.00 -> crosses slices at 50.00, 51.00 (4 MW).
        Slice at 52.00 doesn't cross. IOC fills 4 MW, cancels 4 MW."""
        iceberg = _ice_sell(quantity=8.0, peak_qty=2.0, price=50.00, ppd=1.0)
        state, _ = book.submit_order(empty_state, iceberg)

        ioc = _ioc_buy(quantity=8.0, price=51.00)
        state, result = book.submit_order(state, ioc)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(4.0, abs=1e-6)

    def test_icewpi_vwap_price_across_slices(self, book, empty_state):
        """Verify VWAP trade price reflects different per-slice prices.
        SELL iceberg base=50 ppd=+0.50, peak=2, total=6.
        Slices: 2@50.00, 2@50.50, 2@51.00.
        BUY NON 6 MW at 51.00.
        VWAP = (2*50 + 2*50.5 + 2*51) / 6 = 302/6 = 50.33..."""
        iceberg = _ice_sell(quantity=6.0, peak_qty=2.0, price=50.00, ppd=0.50)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=6.0, price=51.00)
        state, result = book.submit_order(state, buy)

        assert result.trade_qty == pytest.approx(6.0, abs=1e-6)
        # VWAP = (100 + 101 + 102) / 6 = 50.50
        expected_vwap = (2 * 50.00 + 2 * 50.50 + 2 * 51.00) / 6.0
        assert result.trade_price == pytest.approx(expected_vwap, abs=0.02)

    def test_icewpi_ppd_zero_behaves_like_standard(self, book, empty_state):
        """ICEwPI with ppd=0 is functionally equivalent to standard iceberg."""
        iceberg = _ice_sell(quantity=6.0, peak_qty=2.0, price=50.00, ppd=0.0)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=6.0, price=50.00)
        state, result = book.submit_order(state, buy)

        assert result.trade_qty == pytest.approx(6.0, abs=1e-6)
        assert result.trade_price == pytest.approx(50.00, abs=0.01)


# ---------------------------------------------------------------------------
# Boundary: Aggressive order larger than iceberg total
# ---------------------------------------------------------------------------

class TestAggressiveLargerThanTotal:
    """When aggressive qty exceeds iceberg total, test residual behavior."""

    def test_non_remainder_rests_after_draining_iceberg(self, book, empty_state):
        """NON BUY 15 MW vs iceberg SELL total=10 peak=3.
        Fills 10 MW, remainder 5 MW should rest in the book as a bid."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=15.0, price=50.00)
        state, result = book.submit_order(state, buy)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=1e-6)
        # After fills, 5 MW should rest as bid. Verify by submitting a
        # crossing sell that should match against it.
        sell = _non_sell(order_id=61, quantity=5.0, price=50.00, timestamp=300)
        state, sell_result = book.submit_order(state, sell)
        assert sell_result.traded == True
        assert sell_result.trade_qty == pytest.approx(5.0, abs=1e-6)

    def test_ioc_remainder_cancelled_not_resting(self, book, empty_state):
        """IOC BUY 15 MW vs iceberg SELL total=10 peak=3.
        Fills 10 MW, cancels 5 MW. No residual in book."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        ioc = _ioc_buy(quantity=15.0, price=50.00)
        state, result = book.submit_order(state, ioc)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=1e-6)
        # Verify no residual resting: submit a crossing sell, should NOT trade
        sell = _non_sell(order_id=81, quantity=5.0, price=50.00, timestamp=300)
        state, sell_result = book.submit_order(state, sell)
        assert sell_result.traded == False


# ---------------------------------------------------------------------------
# Settlement value accumulation
# ---------------------------------------------------------------------------

class TestSettlementAccumulation:
    """V = Q x p x 0.25 must be correct across multiple slices."""

    def test_settlement_flat_price_10mw(self, book, empty_state):
        """10 MW at flat 50.00: V = 10 * 50 * 0.25 = 125.0."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=10.0, price=50.00)
        state, result = book.submit_order(state, buy)

        assert result.settlement_value == pytest.approx(125.0, abs=0.01)

    def test_settlement_icewpi_multi_price(self, book, empty_state):
        """ICEwPI SELL base=50 ppd=+0.50, peak=2, total=6.
        Slices: 2@50.00, 2@50.50, 2@51.00.
        V = (2*50 + 2*50.5 + 2*51) * 0.25 = 302 * 0.25 = 75.50."""
        iceberg = _ice_sell(quantity=6.0, peak_qty=2.0, price=50.00, ppd=0.50)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=6.0, price=51.00)
        state, result = book.submit_order(state, buy)

        expected_v = (2 * 50.00 + 2 * 50.50 + 2 * 51.00) * 0.25
        assert result.settlement_value == pytest.approx(expected_v, abs=0.05)

    def test_settlement_ioc_partial(self, book, empty_state):
        """IOC 6 MW against iceberg total=7, peak=2, price=50.
        V = 6 * 50 * 0.25 = 75.0."""
        iceberg = _ice_sell(quantity=7.0, peak_qty=2.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        ioc = _ioc_buy(quantity=6.0, price=50.00)
        state, result = book.submit_order(state, ioc)

        assert result.settlement_value == pytest.approx(75.0, abs=0.01)


# ---------------------------------------------------------------------------
# SELL aggressive vs BUY iceberg (symmetry)
# ---------------------------------------------------------------------------

class TestSellAggressiveVsBuyIceberg:
    """Ensure multi-slice matching works symmetrically for sell aggressors."""

    def test_non_sell_drains_buy_iceberg(self, book, empty_state):
        """NON SELL 10 MW vs BUY iceberg total=10 peak=3 at 50.00."""
        iceberg = _ice_buy(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        sell = _non_sell(quantity=10.0, price=50.00)
        state, result = book.submit_order(state, sell)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=1e-6)

    def test_ioc_sell_partial_drain_buy_iceberg(self, book, empty_state):
        """IOC SELL 6 MW vs BUY iceberg total=7 peak=2 at 50.00."""
        iceberg = _ice_buy(quantity=7.0, peak_qty=2.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        sell = _ioc_sell(quantity=6.0, price=50.00)
        state, result = book.submit_order(state, sell)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(6.0, abs=1e-6)

    def test_sell_icewpi_buy_negative_ppd(self, book, empty_state):
        """BUY iceberg ppd=-0.50, base=52, peak=2, total=6.
        SELL NON at 51.00.
        Slices: 2@52, 2@51.5, 2@51.0. All 3 cross at SELL limit=51.
        Full drain = 6 MW."""
        iceberg = _ice_buy(quantity=6.0, peak_qty=2.0, price=52.00, ppd=-0.50)
        state, _ = book.submit_order(empty_state, iceberg)

        sell = _non_sell(quantity=6.0, price=51.00)
        state, result = book.submit_order(state, sell)

        assert result.trade_qty == pytest.approx(6.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Multiple icebergs at same price
# ---------------------------------------------------------------------------

class TestMultipleIcebergs:
    """Single aggressive order should drain through multiple icebergs."""

    def test_non_drains_two_icebergs(self, book, empty_state):
        """Two SELL icebergs (5 MW each, peak=2) at same price.
        NON BUY 10 MW should fill all 10 MW across both icebergs."""
        ice1 = _ice_sell(order_id=10, quantity=5.0, peak_qty=2.0,
                         price=50.00, timestamp=100)
        state, _ = book.submit_order(empty_state, ice1)

        ice2 = _ice_sell(order_id=11, quantity=5.0, peak_qty=2.0,
                         price=50.00, timestamp=101)
        state, _ = book.submit_order(state, ice2)

        buy = _non_buy(quantity=10.0, price=50.00, timestamp=200)
        state, result = book.submit_order(state, buy)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=1e-6)

    def test_ioc_partial_across_two_icebergs(self, book, empty_state):
        """Two SELL icebergs (4 MW each, peak=2). IOC BUY 6 MW.
        Should fill 6 MW (drains first iceberg 4 MW + 2 MW from second)."""
        ice1 = _ice_sell(order_id=10, quantity=4.0, peak_qty=2.0,
                         price=50.00, timestamp=100)
        state, _ = book.submit_order(empty_state, ice1)

        ice2 = _ice_sell(order_id=11, quantity=4.0, peak_qty=2.0,
                         price=50.00, timestamp=101)
        state, _ = book.submit_order(state, ice2)

        ioc = _ioc_buy(quantity=6.0, price=50.00, timestamp=200)
        state, result = book.submit_order(state, ioc)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(6.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Mixed book: limit + iceberg orders
# ---------------------------------------------------------------------------

class TestMixedLimitAndIceberg:
    """Aggressive order walks through regular limit orders and icebergs."""

    def test_non_walks_limit_then_iceberg(self, book, empty_state):
        """Regular SELL limit 3 MW at 49.00 + SELL iceberg 6 MW peak=2 at 50.00.
        BUY NON 9 MW at 50.00 should fill all 9 MW (3 limit + 6 iceberg)."""
        limit_sell = make_order(
            contract_id=37, side=Side.SELL, price=49.00, quantity=3.0,
            order_id=5, agent_id=3, timestamp=90,
            exec_restriction=ExecRestriction.NON,
        )
        state, _ = book.submit_order(empty_state, limit_sell)

        iceberg = _ice_sell(quantity=6.0, peak_qty=2.0, price=50.00, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        buy = _non_buy(quantity=9.0, price=50.00, timestamp=200)
        state, result = book.submit_order(state, buy)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(9.0, abs=1e-6)

    def test_ioc_walks_limit_then_partial_iceberg(self, book, empty_state):
        """Regular SELL limit 2 MW at 49.00 + SELL iceberg 6 MW peak=2 at 50.00.
        IOC BUY 5 MW at 50.00 should fill 5 MW (2 limit + 2 ice + partial from regen)."""
        limit_sell = make_order(
            contract_id=37, side=Side.SELL, price=49.00, quantity=2.0,
            order_id=5, agent_id=3, timestamp=90,
            exec_restriction=ExecRestriction.NON,
        )
        state, _ = book.submit_order(empty_state, limit_sell)

        iceberg = _ice_sell(quantity=6.0, peak_qty=2.0, price=50.00, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        ioc = _ioc_buy(quantity=5.0, price=50.00, timestamp=200)
        state, result = book.submit_order(state, ioc)

        assert result.traded == True
        # 2 from limit + at least 2 from iceberg peak + 1 from regen = 5
        assert result.trade_qty == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# FOK regression: multi-slice FOK still works
# ---------------------------------------------------------------------------

class TestFOKRegression:
    """FOK path should not be broken by M6.4 changes to NON/IOC path."""

    def test_fok_fills_entire_iceberg(self, book, empty_state):
        """FOK BUY 10 MW vs iceberg total=10 peak=3. Should fill all 10 MW."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        fok = _fok_buy(quantity=10.0, price=50.00)
        state, result = book.submit_order(state, fok)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=1e-6)

    def test_fok_rejected_insufficient_liquidity(self, book, empty_state):
        """FOK BUY 15 MW vs iceberg total=10 peak=3. Rejected (only 10 MW available)."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        fok = _fok_buy(quantity=15.0, price=50.00)
        state, result = book.submit_order(state, fok)

        assert result.traded == False
        assert result.trade_qty == pytest.approx(0.0, abs=1e-6)

    def test_fok_settlement_correct(self, book, empty_state):
        """FOK BUY 10 MW at 50: V = 10 * 50 * 0.25 = 125.0."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        fok = _fok_buy(quantity=10.0, price=50.00)
        state, result = book.submit_order(state, fok)

        assert result.settlement_value == pytest.approx(125.0, abs=0.01)


# ---------------------------------------------------------------------------
# Edge case: iceberg with very large ppd
# ---------------------------------------------------------------------------

class TestICEwPILargePPD:
    """ICEwPI with a large ppd where only the first slice crosses."""

    def test_large_ppd_only_first_slice_crosses(self, book, empty_state):
        """SELL iceberg base=50 ppd=+5.0, peak=2, total=6.
        BUY NON at 52.00.
        Slice 0: 50.00 (crosses). Slice 1: 55.00 (does NOT cross).
        Only 2 MW filled."""
        iceberg = _ice_sell(quantity=6.0, peak_qty=2.0, price=50.00, ppd=5.0)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=6.0, price=52.00)
        state, result = book.submit_order(state, buy)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(2.0, abs=1e-6)

    def test_large_negative_ppd_only_first_slice_crosses(self, book, empty_state):
        """BUY iceberg base=50 ppd=-5.0, peak=2, total=6.
        SELL NON at 48.00.
        Slice 0: 50.00 (crosses). Slice 1: 45.00 (does NOT cross for SELL at 48).
        Only 2 MW filled."""
        iceberg = _ice_buy(quantity=6.0, peak_qty=2.0, price=50.00, ppd=-5.0)
        state, _ = book.submit_order(empty_state, iceberg)

        sell = _non_sell(quantity=6.0, price=48.00)
        state, result = book.submit_order(state, sell)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge case: aggressive order qty equals exactly one peak
# ---------------------------------------------------------------------------

class TestSinglePeakMatch:
    """Aggressive order qty exactly equals one peak -- no loop needed."""

    def test_non_buy_matches_exactly_one_peak(self, book, empty_state):
        """NON BUY 3 MW vs iceberg peak=3 total=10.
        Should fill 3 MW. Iceberg regenerates but aggressive is done."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=3.0, price=50.00)
        state, result = book.submit_order(state, buy)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(3.0, abs=1e-6)

    def test_ioc_buy_matches_exactly_one_peak(self, book, empty_state):
        """IOC BUY 2 MW vs iceberg peak=2 total=7.
        Should fill 2 MW exactly."""
        iceberg = _ice_sell(quantity=7.0, peak_qty=2.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        ioc = _ioc_buy(quantity=2.0, price=50.00)
        state, result = book.submit_order(state, ioc)

        assert result.traded == True
        assert result.trade_qty == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Trade price determination: iceberg is always the price-setter (SS6.2.7)
# ---------------------------------------------------------------------------

class TestIcebergIsPriceSetter:
    """Per SS6.2.7: when a single incoming order matches multiple iceberg slices,
    the iceberg is always the price-setter (passive side)."""

    def test_trade_price_is_iceberg_price_not_aggressive(self, book, empty_state):
        """BUY NON at 55.00 vs SELL iceberg at 50.00.
        Trade price should be 50.00 (iceberg's price), not 55.00."""
        iceberg = _ice_sell(quantity=10.0, peak_qty=3.0, price=50.00)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=10.0, price=55.00)
        state, result = book.submit_order(state, buy)

        assert result.trade_qty == pytest.approx(10.0, abs=1e-6)
        assert result.trade_price == pytest.approx(50.00, abs=0.01)

    def test_icewpi_trade_price_reflects_drifted_prices(self, book, empty_state):
        """BUY NON at 55.00 vs ICEwPI SELL base=50 ppd=+1.0, peak=2, total=4.
        Slices: 2@50.00, 2@51.00. All cross at 55.00.
        VWAP = (2*50 + 2*51) / 4 = 50.50. Not 55.00."""
        iceberg = _ice_sell(quantity=4.0, peak_qty=2.0, price=50.00, ppd=1.0)
        state, _ = book.submit_order(empty_state, iceberg)

        buy = _non_buy(quantity=4.0, price=55.00)
        state, result = book.submit_order(state, buy)

        assert result.trade_qty == pytest.approx(4.0, abs=1e-6)
        expected_vwap = (2 * 50.00 + 2 * 51.00) / 4.0
        assert result.trade_price == pytest.approx(expected_vwap, abs=0.02)
