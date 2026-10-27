"""Tests for M6.4 -- Multi-Slice Iceberg Matching Per Order.

Behavioral contracts for the generalized multi-slice iceberg matching loop
that allows a single aggressive NON or IOC order to consume multiple iceberg
slices within one submit_order() call.

All expected values are derived from:
  - docs/omiecid/market-spec.md SS4.2.1 (Rule 2 -- to be updated by this story)
  - docs/omiecid/architecture-reviews.md §M6.4 (design: regeneration loop)
  - docs/omiecid/agile-plan.md M6.4 (acceptance criteria)
  - Continuous Trading Matching Algorithm SS5.1.2: "a single incoming order
    can match more than one slice of an iceberg order."

Domain rule (SIDC SS5.1.2, corrected behavior -- M6.4):
  - A single aggressive order (NON, IOC, or FOK) can fill multiple iceberg
    slices within one submit_order call.
  - After the initial match consumes visible book quantity, iceberg slices are
    regenerated; if the regenerated slice crosses the aggressive order's resting
    remainder, matching occurs again (Rule 3). This repeats until the aggressive
    order is fully filled or no more regeneration produces fills.
  - For NON orders: unfilled remainder rests in the book after the loop.
  - For IOC orders: unfilled remainder is cancelled after the loop.
  - FOK path is unchanged (already implemented in M6.3).

Acceptance criteria (M6.4):
  1. Single BUY 10 MW fills entire iceberg (peak=3, total=10) in one
     submit_order() call with trade_qty=10.
  2. IOC 6 MW against iceberg (peak=2, total=7) fills 6 MW across 3 slices.
  3. PPD/ICEwPI price improvement still applies per slice within the loop.
  4. All existing tests updated and passing (outside scope of this file).
  5. market-spec.md SS4.2.1 Rule 2 updated (outside scope of this file).

Test categories:
  1. TestNONMultiSliceFullDrain -- NON drains entire iceberg in one call
  2. TestIOCMultiSliceFill -- IOC fills across multiple slices
  3. TestICEwPIPricePerSlice -- PPD price drift applies per slice in the loop
  4. TestPartialLastSlice -- aggressive order partially fills the last slice
  5. TestAggressiveLargerThanIceberg -- aggressive order exceeds iceberg total
  6. TestMixedLimitAndIcebergBook -- limit + iceberg in same book
  7. TestSettlementAccumulation -- settlement value correct across slices
  8. TestMultipleIcebergsAtSamePrice -- aggressive fills across two icebergs
  9. TestBidSideSymmetry -- SELL aggressive against BUY-side icebergs
 10. TestFOKUnchanged -- FOK path remains unaffected by M6.4

These tests are black-box: they import only the public interface and
assert against domain-specified values. No implementation assumptions.
"""

import pytest
from pytest import approx

from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    make_order,
)
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig
from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTRACT_ID = 42

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




def _make_contract():
    return Contract(
        contract_id=_CONTRACT_ID,
        delivery_start_minutes=630,   # 10:30
        delivery_end_minutes=645,     # 10:45
        gct_minutes=570,              # 09:30
        status=ContractStatus.OPEN,
    )


def _make_book(book_depth=100):
    return ContractBook(
        contract=_make_contract(),
        product_config=OMIEProductConfig(),
        market_config=OMIEMarketConfig(book_depth=book_depth),
    )


def _order(side, price, quantity, order_id, agent_id, timestamp,
           exec_restriction=ExecRestriction.NON, peak_qty=0.0, ppd=0.0):
    return make_order(
        contract_id=_CONTRACT_ID, side=side, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction, peak_qty=peak_qty, ppd=ppd,
    )


# ---------------------------------------------------------------------------
# 1. NON order drains entire iceberg in one submit_order call
# ---------------------------------------------------------------------------

class TestNONMultiSliceFullDrain:
    """AC-1: A single NON aggressive order fills an entire iceberg.

    SIDC SS5.1.2: "a single incoming order can match more than one slice of
    an iceberg order already in the order book."
    """

    def test_non_buy_10mw_fills_entire_iceberg_peak3_total10(self):
        """AC-1: Single BUY 10 MW fills entire iceberg (peak=3, total=10).

        Iceberg SELL: total=10 MW, peak=3 MW at 50.00.
        NON BUY 10 MW at 50.00.
        Expected: trade_qty=10.0, trade_price=50.00, iceberg fully drained.

        Match sequence (via regeneration loop):
          Initial dispatch: fills 3 MW (peak1). 7 MW remainder rests.
          Regen slice2 (3@50): crosses resting 7 MW -> fills 3 MW. 4 MW left.
          Regen slice3 (3@50): crosses resting 4 MW -> fills 3 MW. 1 MW left.
          Regen slice4 (1@50): crosses resting 1 MW -> fills 1 MW. Done.
          Total: 3+3+3+1 = 10 MW.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, r_ice = book.submit_order(state, ice)
        assert r_ice.accepted

        non = _order(Side.BUY, 50.00, 10.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.accepted == True
        assert r.traded == True
        assert r.trade_qty == approx(10.0), \
            "AC-1: single NON BUY 10 MW must fill entire iceberg in one call"
        assert r.trade_price == approx(50.00)

        # Iceberg fully consumed -> book empty
        bids, asks = book.get_anonymous_lob(state)
        assert asks == [], "Iceberg fully drained, no asks remain"
        assert bids == [], "NON fully filled, no bids remain"

    def test_non_buy_9mw_fills_entire_iceberg_peak3_total9(self):
        """NON BUY 9 MW fills iceberg (peak=3, total=9) exactly.

        Slices: 3+3+3 = 9 MW. All consumed, book empty.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 9.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 9.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(9.0)
        assert r.trade_price == approx(50.00)

        _, asks = book.get_anonymous_lob(state)
        assert asks == []

    def test_non_buy_7mw_against_iceberg_peak2_total7(self):
        """NON BUY 7 MW against iceberg (peak=2, total=7).

        Slices: 2+2+2+1 = 7 MW. All consumed.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 7.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(7.0)
        assert r.trade_price == approx(50.00)

        _, asks = book.get_anonymous_lob(state)
        assert asks == []

    def test_non_buy_peak_equals_total(self):
        """NON BUY 5 MW against iceberg peak=5, total=5 (one slice only).

        No multi-slice needed. Regression: must still work.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 5.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=5.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 5.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(5.0)

    def test_non_buy_minimum_peak(self):
        """NON BUY 1.0 MW against iceberg peak=0.1, total=1.0.

        10 slices of 0.1 MW each. All consumed.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 1.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=0.1)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 1.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(1.0)

        _, asks = book.get_anonymous_lob(state)
        assert asks == []


# ---------------------------------------------------------------------------
# 2. IOC multi-slice fill
# ---------------------------------------------------------------------------

class TestIOCMultiSliceFill:
    """AC-2: IOC fills across multiple iceberg slices, remainder cancelled.

    IOC aggressive orders now use multi-slice matching (M6.4). After the
    regeneration loop ends, any unfilled remainder is cancelled.
    """

    def test_ioc_6mw_against_iceberg_peak2_total7_fills_6(self):
        """AC-2: IOC BUY 6 MW against iceberg (peak=2, total=7) fills 6 MW.

        Match sequence:
          Initial dispatch: fills 2 MW (peak1). 4 MW IOC remainder rests.
          Regen slice2 (2@50): crosses resting 4 MW -> fills 2 MW. 2 MW left.
          Regen slice3 (2@50): crosses resting 2 MW -> fills 2 MW. Done.
          Total: 2+2+2 = 6 MW.
        IOC: fully filled (6 of 6 requested), no remainder to cancel.
        Iceberg: 7-6 = 1 MW remaining.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.BUY, 50.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.accepted == True
        assert r.traded == True
        assert r.trade_qty == approx(6.0), \
            "AC-2: IOC 6 MW must fill across 3 iceberg slices"
        assert r.trade_price == approx(50.00)

        # IOC remainder must not rest in book
        assert _is_absent(book.get_order(state, 2, Side.BUY)), \
            "IOC remainder must not rest in book"

        # Iceberg: 7-6=1 MW remaining. New slice: min(2, 1)=1 MW visible.
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        assert asks[0][0] == approx(50.00)
        assert asks[0][1] == approx(1.0), \
            "Iceberg last slice: min(peak=2, remaining=1) = 1 MW"

    def test_ioc_fills_entire_iceberg(self):
        """IOC BUY 7 MW against iceberg (peak=2, total=7) fills all 7 MW.

        Slices: 2+2+2+1 = 7 MW. Iceberg fully drained.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.BUY, 50.00, 7.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.traded == True
        assert r.trade_qty == approx(7.0)

        _, asks = book.get_anonymous_lob(state)
        assert asks == [], "Iceberg fully drained"
        bids, _ = book.get_anonymous_lob(state)
        assert bids == [], "IOC fully filled, no remainder"

    def test_ioc_partial_fill_remainder_cancelled(self):
        """IOC BUY 10 MW against iceberg (peak=2, total=7) fills 7 MW, cancels 3.

        IOC fills the entire iceberg (7 MW) then no more crossing -> 3 MW
        unfilled. IOC cancels the 3 MW remainder.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.BUY, 50.00, 10.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.traded == True
        assert r.trade_qty == approx(7.0), \
            "IOC fills entire iceberg (7 MW), remainder cancelled"

        # IOC remainder not in book
        assert _is_absent(book.get_order(state, 2, Side.BUY))
        # Book completely empty
        bids, asks = book.get_anonymous_lob(state)
        assert bids == []
        assert asks == []

    def test_ioc_no_crossing_no_fill(self):
        """IOC BUY at 49.00 against iceberg SELL at 50.00 -> no crossing.

        Regression: IOC that does not cross should still be rejected/no trade.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.BUY, 49.00, 5.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.traded == False


# ---------------------------------------------------------------------------
# 3. ICEwPI price improvement per slice within the loop
# ---------------------------------------------------------------------------

class TestICEwPIPricePerSlice:
    """AC-3: PPD/ICEwPI price improvement still applies per slice.

    Each regenerated slice has a drifted price. The aggressive order fills
    at the iceberg's drifted limit price per the price-determination rule.
    If the drifted price no longer crosses, the loop terminates naturally.
    """

    def test_non_buy_against_icewpi_sell_fills_crossing_slices(self):
        """NON BUY 6@52 against ICEwPI SELL total=6, peak=2, ppd=+0.50 at 50.00.

        Slices at natural prices: 2@50.00, 2@50.50, 2@51.00.
        All cross the BUY limit of 52.00.
        Expected: fills all 6 MW.
        VWAP = (2*50 + 2*50.50 + 2*51) / 6 = (100 + 101 + 102) / 6 = 303/6 = 50.50
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
                     ppd=0.50)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 52.00, 6.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(6.0), \
            "NON fills all 3 crossing slices of ICEwPI iceberg"
        expected_vwap = 303.0 / 6.0  # 50.50
        assert r.trade_price == approx(expected_vwap, abs=0.01), \
            f"VWAP = (2*50 + 2*50.50 + 2*51) / 6 = {expected_vwap:.2f}"

        _, asks = book.get_anonymous_lob(state)
        assert asks == [], "ICEwPI fully drained"

    def test_non_buy_against_icewpi_sell_stops_when_no_crossing(self):
        """NON BUY 6@50.50 against ICEwPI SELL total=6, peak=2, ppd=+0.50 at 50.

        Slices: 2@50.00 (crosses), 2@50.50 (crosses at limit), 2@51.00 (no cross).
        Expected: fills 4 MW (2@50 + 2@50.50). 2 MW NON remainder rests.
        VWAP = (2*50 + 2*50.50) / 4 = (100 + 101) / 4 = 201/4 = 50.25
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
                     ppd=0.50)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.50, 6.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(4.0), \
            "Only 2 slices cross at 50.50: 2@50.00 + 2@50.50 = 4 MW"
        expected_vwap = 201.0 / 4.0  # 50.25
        assert r.trade_price == approx(expected_vwap, abs=0.01)

        # 2 MW NON remainder rests as bid at 50.50
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 1
        assert bids[0][0] == approx(50.50)
        assert bids[0][1] == approx(2.0)
        # Remaining iceberg: 2 MW at 51.00 (not crossing)
        assert len(asks) == 1
        assert asks[0][0] == approx(51.00)
        assert asks[0][1] == approx(2.0)

    def test_ioc_against_icewpi_fills_crossing_cancels_rest(self):
        """IOC BUY 6@50.50 against ICEwPI SELL total=6, peak=2, ppd=+0.50 at 50.

        Same as above but IOC: fills 4 MW, cancels 2 MW remainder.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
                     ppd=0.50)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.BUY, 50.50, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.traded == True
        assert r.trade_qty == approx(4.0), \
            "IOC fills 2 crossing slices (4 MW), cancels 2 MW remainder"

        # IOC remainder not in book
        assert _is_absent(book.get_order(state, 2, Side.BUY))
        # No bids (IOC cancelled)
        bids, _ = book.get_anonymous_lob(state)
        assert bids == []

    def test_non_buy_against_icewpi_buy_negative_ppd(self):
        """NON SELL 4@49 against ICEwPI BUY total=4, peak=2, ppd=-0.50 at 50.

        Buy iceberg: slices at 50.00 (crosses), 49.50 (crosses).
        VWAP = (2*50 + 2*49.50) / 4 = (100+99)/4 = 199/4 = 49.75
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 50.00, 4.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
                     ppd=-0.50)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.SELL, 49.00, 4.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(4.0)
        expected_vwap = 199.0 / 4.0  # 49.75
        assert r.trade_price == approx(expected_vwap, abs=0.01)

        bids, _ = book.get_anonymous_lob(state)
        assert bids == []

    def test_icewpi_large_ppd_only_first_slice_crosses(self):
        """NON BUY 6@50 against ICEwPI SELL total=6, peak=2, ppd=+1.00 at 50.

        Slices: 2@50.00 (crosses at limit), 2@51.00 (no cross), 2@52.00 (no).
        Fills only 2 MW from first slice. 4 MW NON remainder rests.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
                     ppd=1.00)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 6.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(2.0), \
            "Only first slice at 50.00 crosses the BUY limit of 50.00"
        assert r.trade_price == approx(50.00)


# ---------------------------------------------------------------------------
# 4. Partial last slice
# ---------------------------------------------------------------------------

class TestPartialLastSlice:
    """Aggressive order partially fills the last iceberg slice."""

    def test_non_buy_8mw_against_iceberg_peak3_total10(self):
        """NON BUY 8 MW against iceberg (peak=3, total=10).

        Slices consumed: 3+3+2 = 8 MW. Last slice (3 MW) partially filled (2).
        Iceberg: 10-8 = 2 MW remaining. Visible: 1 MW (from partial slice3).
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 8.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(8.0)
        assert r.trade_price == approx(50.00)

        # NON fully filled -> no bids
        bids, asks = book.get_anonymous_lob(state)
        assert bids == []
        # Iceberg: 10-8=2 remaining. Current slice had 3 MW, 2 consumed -> 1 visible.
        assert len(asks) == 1
        assert asks[0][0] == approx(50.00)
        assert asks[0][1] == approx(1.0), \
            "Partial last slice: 3 - 2 = 1 MW visible"

    def test_ioc_5mw_against_iceberg_peak3_total10(self):
        """IOC BUY 5 MW against iceberg (peak=3, total=10).

        Fills: 3 (peak1) + 2 (from peak2, partial) = 5 MW. IOC fully filled.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.BUY, 50.00, 5.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.traded == True
        assert r.trade_qty == approx(5.0)

        # IOC fully filled, no remainder
        assert _is_absent(book.get_order(state, 2, Side.BUY))

    def test_non_buy_1mw_against_iceberg_peak3_total10(self):
        """NON BUY 1 MW against iceberg (peak=3, total=10).

        Fills only 1 MW from peak1 (partial). No multi-slice needed.
        Regression: small order against large iceberg.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 1.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(1.0)

        _, asks = book.get_anonymous_lob(state)
        assert asks[0][0] == approx(50.00)
        assert asks[0][1] == approx(2.0), \
            "Peak partially filled: 3 - 1 = 2 MW visible"


# ---------------------------------------------------------------------------
# 5. Aggressive order larger than iceberg total
# ---------------------------------------------------------------------------

class TestAggressiveLargerThanIceberg:
    """Aggressive order exceeds iceberg total quantity."""

    def test_non_buy_15mw_against_iceberg_total10_fills_10_rests_5(self):
        """NON BUY 15 MW against iceberg (peak=3, total=10) fills 10, 5 rests.

        Multi-slice loop drains entire iceberg (10 MW). 5 MW rests as bid.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 15.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(10.0), \
            "NON fills entire iceberg (10 MW), remainder rests"

        # 5 MW rests as bid
        bids, asks = book.get_anonymous_lob(state)
        assert asks == [], "Iceberg fully drained"
        assert len(bids) == 1
        assert bids[0][0] == approx(50.00)
        assert bids[0][1] == approx(5.0), \
            "NON remainder: 15 - 10 = 5 MW rests as bid"

    def test_ioc_15mw_against_iceberg_total10_fills_10_cancels_5(self):
        """IOC BUY 15 MW against iceberg (peak=3, total=10) fills 10, cancels 5.

        Multi-slice loop drains entire iceberg. IOC cancels 5 MW remainder.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.BUY, 50.00, 15.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.traded == True
        assert r.trade_qty == approx(10.0), \
            "IOC fills entire iceberg (10 MW), cancels remainder"

        # IOC remainder cancelled, book empty
        bids, asks = book.get_anonymous_lob(state)
        assert asks == []
        assert bids == [], "IOC remainder cancelled, no bids"
        assert _is_absent(book.get_order(state, 2, Side.BUY))


# ---------------------------------------------------------------------------
# 6. Mixed limit + iceberg book
# ---------------------------------------------------------------------------

class TestMixedLimitAndIcebergBook:
    """Aggressive order fills limit orders and iceberg slices in one call."""

    def test_non_buy_walks_limit_then_drains_iceberg(self):
        """NON BUY 8@51 against limit SELL 3@50 + iceberg SELL total=9 peak=3 @51.

        Fills: 3@50 (limit consumed) + 3@51 (iceberg peak1) + 2@51 (from peak2).
        Total: 3+3+2 = 8 MW.
        VWAP = (3*50 + 5*51) / 8 = (150 + 255) / 8 = 405/8 = 50.625
        """
        book = _make_book()
        state = book.init()

        lim = _order(Side.SELL, 50.00, 3.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        ice = _order(Side.SELL, 51.00, 9.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 51.00, 8.0, 3, agent_id=3, timestamp=3)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(8.0)
        expected_vwap = 405.0 / 8.0  # 50.625
        assert r.trade_price == approx(expected_vwap, abs=0.01)

        # Limit consumed, iceberg: 9-5=4 remaining, slice2 has 1 MW visible
        bids, asks = book.get_anonymous_lob(state)
        assert bids == [], "NON fully filled"
        assert len(asks) == 1
        assert asks[0][0] == approx(51.00)

    def test_non_buy_fills_limit_only_when_iceberg_at_higher_price(self):
        """NON BUY 3@50 with limit SELL 3@50 + iceberg SELL @51.

        Fills only limit (3@50). Iceberg at 51 does not cross.
        No multi-slice iteration needed.
        """
        book = _make_book()
        state = book.init()

        lim = _order(Side.SELL, 50.00, 3.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        ice = _order(Side.SELL, 51.00, 9.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 3.0, 3, agent_id=3, timestamp=3)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(3.0)
        assert r.trade_price == approx(50.00)

    def test_ioc_fills_limit_then_drains_iceberg_cancels_remainder(self):
        """IOC BUY 20@51 against limit 3@50 + iceberg total=9 peak=3 @51.

        Fills: 3@50 + 9@51 = 12 MW. IOC cancels 8 MW remainder.
        VWAP = (3*50 + 9*51) / 12 = (150+459)/12 = 609/12 = 50.75
        """
        book = _make_book()
        state = book.init()

        lim = _order(Side.SELL, 50.00, 3.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        ice = _order(Side.SELL, 51.00, 9.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.BUY, 51.00, 20.0, 3, agent_id=3, timestamp=3,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.traded == True
        assert r.trade_qty == approx(12.0)
        expected_vwap = 609.0 / 12.0  # 50.75
        assert r.trade_price == approx(expected_vwap, abs=0.01)

        # Book completely empty (limit consumed, iceberg drained, IOC cancelled)
        bids, asks = book.get_anonymous_lob(state)
        assert bids == []
        assert asks == []


# ---------------------------------------------------------------------------
# 7. Settlement value accumulation across slices
# ---------------------------------------------------------------------------

class TestSettlementAccumulation:
    """Settlement value V = Q x p x 0.25 must be correct across all slices.

    For multi-slice fills, the settlement value is the sum of each slice's
    settlement contribution: sum(q_i * p_i * 0.25) across all iterations.
    """

    def test_settlement_non_drains_entire_iceberg(self):
        """NON BUY 10@50 against iceberg (peak=3, total=10) at 50.00.

        V = 10 * 50 * 0.25 = 125.00 EUR.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 10.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.settlement_value == approx(125.00), \
            "V = 10 * 50 * 0.25 = 125.00"

    def test_settlement_ioc_multi_slice(self):
        """IOC BUY 6@50 against iceberg (peak=2, total=7) at 50.00.

        V = 6 * 50 * 0.25 = 75.00 EUR.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.BUY, 50.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.traded == True
        assert r.settlement_value == approx(75.00), \
            "V = 6 * 50 * 0.25 = 75.00"

    def test_settlement_icewpi_multi_price(self):
        """NON BUY 6@52 against ICEwPI SELL total=6, peak=2, ppd=+0.50 at 50.

        Fills: 2@50.00 + 2@50.50 + 2@51.00.
        V = (2*50 + 2*50.50 + 2*51) * 0.25 = 303 * 0.25 = 75.75 EUR.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
                     ppd=0.50)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 52.00, 6.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.settlement_value == approx(75.75), \
            "V = (2*50 + 2*50.50 + 2*51) * 0.25 = 303 * 0.25 = 75.75"

    def test_settlement_mixed_limit_plus_iceberg(self):
        """NON BUY 8@51 against limit 3@50 + iceberg total=9 peak=3 @51.

        Fills: 3@50 + 5@51 = 8 MW.
        V = (3*50 + 5*51) * 0.25 = (150+255) * 0.25 = 405 * 0.25 = 101.25 EUR.
        """
        book = _make_book()
        state = book.init()

        lim = _order(Side.SELL, 50.00, 3.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        ice = _order(Side.SELL, 51.00, 9.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 51.00, 8.0, 3, agent_id=3, timestamp=3)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.settlement_value == approx(101.25), \
            "V = (3*50 + 5*51) * 0.25 = 101.25"


# ---------------------------------------------------------------------------
# 8. Multiple icebergs at the same price
# ---------------------------------------------------------------------------

class TestMultipleIcebergsAtSamePrice:
    """Aggressive order fills across two icebergs at the same price."""

    def test_non_drains_two_icebergs_at_same_price(self):
        """NON BUY 8@50 against two icebergs SELL at 50.00:
        ICE1: total=5, peak=2.  ICE2: total=4, peak=1.
        Total = 9 MW. NON fills 8 MW across both icebergs.

        Price-time priority: ICE1 (ts=1) matched first, then ICE2 (ts=2).
        ICE1 slices: 2+2+1 = 5 MW. ICE2 slices: 1+1+1 = 3 MW. Total: 5+3=8.
        """
        book = _make_book()
        state = book.init()

        ice1 = _order(Side.SELL, 50.00, 5.0, 1, agent_id=1, timestamp=1,
                      exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice1)

        ice2 = _order(Side.SELL, 50.00, 4.0, 2, agent_id=2, timestamp=2,
                      exec_restriction=ExecRestriction.ICE, peak_qty=1.0)
        state, _ = book.submit_order(state, ice2)

        non = _order(Side.BUY, 50.00, 8.0, 3, agent_id=3, timestamp=3)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(8.0), \
            "NON fills 8 MW across two icebergs"
        assert r.trade_price == approx(50.00)

        # 9-8 = 1 MW remaining in one of the icebergs
        _, asks = book.get_anonymous_lob(state)
        total_remaining = sum(q for _, q in asks)
        assert total_remaining == approx(1.0)

    def test_non_drains_both_icebergs_completely(self):
        """NON BUY 9@50 against two icebergs totaling 9 MW. All consumed."""
        book = _make_book()
        state = book.init()

        ice1 = _order(Side.SELL, 50.00, 5.0, 1, agent_id=1, timestamp=1,
                      exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice1)

        ice2 = _order(Side.SELL, 50.00, 4.0, 2, agent_id=2, timestamp=2,
                      exec_restriction=ExecRestriction.ICE, peak_qty=1.0)
        state, _ = book.submit_order(state, ice2)

        non = _order(Side.BUY, 50.00, 9.0, 3, agent_id=3, timestamp=3)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(9.0)

        _, asks = book.get_anonymous_lob(state)
        assert asks == [], "Both icebergs fully drained"


# ---------------------------------------------------------------------------
# 9. Bid-side symmetry (SELL aggressive against BUY icebergs)
# ---------------------------------------------------------------------------

class TestBidSideSymmetry:
    """Multi-slice matching works symmetrically for SELL aggressors."""

    def test_non_sell_drains_buy_iceberg(self):
        """NON SELL 10@50 against iceberg BUY total=10, peak=3 at 50.00.

        Fills 10 MW across all slices. Same as AC-1 but reversed side.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.SELL, 50.00, 10.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(10.0)
        assert r.trade_price == approx(50.00)

        bids, asks = book.get_anonymous_lob(state)
        assert bids == []
        assert asks == []

    def test_ioc_sell_multi_slice_against_buy_iceberg(self):
        """IOC SELL 6@50 against iceberg BUY (peak=2, total=7). Fills 6 MW."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.SELL, 50.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.traded == True
        assert r.trade_qty == approx(6.0)

        # IOC remainder not in book
        assert _is_absent(book.get_order(state, 2, Side.SELL))

    def test_non_sell_against_buy_icewpi(self):
        """NON SELL 4@49 against ICEwPI BUY total=4, peak=2, ppd=-0.50 at 50.

        Slices: 2@50.00, 2@49.50. Both cross the SELL limit of 49.00.
        VWAP = (2*50 + 2*49.50) / 4 = 199/4 = 49.75.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 50.00, 4.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
                     ppd=-0.50)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.SELL, 49.00, 4.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(4.0)
        assert r.trade_price == approx(49.75, abs=0.01)


# ---------------------------------------------------------------------------
# 10. FOK path unchanged (regression)
# ---------------------------------------------------------------------------

class TestFOKUnchanged:
    """FOK path must remain unaffected by M6.4 changes.

    M6.4 does NOT modify _fok_multi_slice_fill(). These are regression tests.
    """

    def test_fok_still_fills_entire_iceberg(self):
        """FOK BUY 7 MW against iceberg (peak=2, total=7) -> fills 7 MW.

        Same as M6.3 test. Must still work.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 7.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r = book.submit_order(state, fok)
        assert r.traded == True
        assert r.trade_qty == approx(7.0)

    def test_fok_rejected_when_total_insufficient(self):
        """FOK BUY 8 MW against iceberg total=7 -> rejected."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 8.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r = book.submit_order(state, fok)
        assert r.traded == False

    def test_fok_settlement_unchanged(self):
        """FOK BUY 3@50 settlement = 3*50*0.25 = 37.50. Regression."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 3.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r = book.submit_order(state, fok)
        assert r.traded == True
        assert r.trade_qty == approx(3.0)
        assert r.settlement_value == approx(37.50)
