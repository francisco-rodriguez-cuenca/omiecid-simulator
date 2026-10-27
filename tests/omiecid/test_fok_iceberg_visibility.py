"""Tests for M6.3 -- FOK Iceberg Full-Quantity Visibility.

Behavioral contracts for the corrected FOK liquidity pre-check that considers
the TOTAL iceberg quantity (visible peak + hidden remainder), not just the
visible peak.

All expected values are derived from:
  - docs/omiecid/market-spec.md SS4.2.1 (Rule 1 -- to be updated by this story)
  - docs/omiecid/architecture-reviews.md §Iceberg-Fidelity (Divergence 1)
  - docs/omiecid/agile-plan.md M6.3 (acceptance criteria)
  - Continuous Trading Matching Algorithm SS3.3: "full quantity is exposed
    to the market for matching."

Domain rule (SIDC SS3.3, corrected behavior):
  - FOK orders see the iceberg's FULL remaining quantity for matching purposes.
  - _matchable_volume() must include hidden iceberg quantity from _iceberg_meta
    for opposite-side icebergs at crossing prices.
  - This applies to the PRE-CHECK only. Actual matching still follows the
    single-slice-per-order rule (M6.4 will address multi-slice matching).

Acceptance criteria (M6.3):
  1. FOK BUY 3 MW accepted against iceberg with peak=2, total=7
  2. FOK BUY 8 MW rejected against same iceberg (total=7)
  3. All existing tests updated and passing (outside scope of this file)
  4. market-spec.md SS4.2.1 Rule 1 updated (outside scope of this file)

IMPORTANT NOTE ON SINGLE-SLICE-PER-ORDER (still active in M6.3):
  M6.3 changes only the FOK PRE-CHECK (visibility). It does NOT change the
  matching behavior. A FOK that passes the pre-check still matches against
  only the visible peak in one dispatch. The FOK is accepted (trade occurs)
  but trade_qty reflects only what the visible book can fill in one pass.
  M6.4 will address multi-slice matching within a single submit_order call.

  Therefore: FOK BUY 3 MW against iceberg peak=2, total=7 is ACCEPTED
  by the pre-check, but the actual trade_qty depends on what jaxob can fill.
  Since only peak=2 is in jaxob, trade_qty=2 and 1 MW remainder is unmatched.
  But FOK semantics require FULL fill or CANCEL. This means M6.3's FOK
  acceptance only makes sense if the implementation also fills the full FOK qty
  (i.e., M6.3 implicitly requires at least FOK-specific multi-slice matching
  for the accepted FOK to actually fill).

  The acceptance criteria say "FOK BUY 3 MW ACCEPTED" -- meaning it trades
  3 MW. We test for trade_qty == 3.0 as the acceptance criterion states.

Test categories:
  1. TestFOKSeesTotalIcebergQuantity -- core acceptance criteria
  2. TestFOKVisibilityEdgeCases -- boundary and edge cases
  3. TestFOKVisibilityWithMixedBook -- iceberg + limit combinations
  4. TestFOKVisibilityBidSide -- bid-side iceberg symmetry
  5. TestFOKVisibilityICEwPI -- ICEwPI icebergs and FOK visibility
  6. TestFOKBookStatePreservation -- rejected FOK does not alter state

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
# 1. Core acceptance criteria: FOK sees total iceberg quantity
# ---------------------------------------------------------------------------

class TestFOKSeesTotalIcebergQuantity:
    """FOK liquidity pre-check considers total iceberg quantity (peak + hidden).

    SIDC Continuous Trading Matching Algorithm SS3.3: "full quantity is exposed
    to the market for matching."
    """

    def test_fok_buy_accepted_against_iceberg_peak2_total7(self):
        """AC-1: FOK BUY 3 MW accepted against iceberg SELL peak=2, total=7.

        Total matchable = 7 MW (2 visible + 5 hidden) >= 3 MW -> accepted.
        FOK fills 3 MW.
        """
        book = _make_book()
        state = book.init()

        # Iceberg SELL: total=7 MW, peak=2 MW at 50.00
        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, r_ice = book.submit_order(state, ice)
        assert r_ice.accepted

        # FOK BUY 3 MW at 50.00 -- total iceberg = 7 >= 3 -> accepted
        fok = _order(Side.BUY, 50.00, 3.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.accepted == True
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(3.0)
        assert r_fok.trade_price == approx(50.00)

    def test_fok_buy_rejected_against_iceberg_peak2_total7_needs8(self):
        """AC-2: FOK BUY 8 MW rejected against iceberg SELL peak=2, total=7.

        Total matchable = 7 MW < 8 MW -> rejected. No trade.
        """
        book = _make_book()
        state = book.init()

        # Iceberg SELL: total=7 MW, peak=2 MW at 50.00
        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, r_ice = book.submit_order(state, ice)
        assert r_ice.accepted

        # FOK BUY 8 MW at 50.00 -- total iceberg = 7 < 8 -> rejected
        fok = _order(Side.BUY, 50.00, 8.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_fok_buy_exact_total_quantity_accepted(self):
        """FOK BUY 7 MW against iceberg SELL peak=2, total=7 -> accepted.

        Boundary: FOK requests exactly the total iceberg quantity.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 7.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.accepted == True
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(7.0)

    def test_fok_buy_one_tick_over_total_rejected(self):
        """FOK BUY 7.1 MW against iceberg total=7 -> rejected.

        Boundary: 0.1 MW over the total quantity.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 7.1, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False


# ---------------------------------------------------------------------------
# 2. Edge cases for FOK visibility
# ---------------------------------------------------------------------------

class TestFOKVisibilityEdgeCases:
    """Edge cases: peak-only icebergs, fully consumed icebergs, etc."""

    def test_fok_accepted_at_peak_only(self):
        """FOK BUY 2 MW against iceberg peak=2, total=7 -> fills at peak.

        Even with total-quantity visibility, a FOK that fits within the peak
        still works. Regression check.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 2.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(2.0)
        assert r_fok.trade_price == approx(50.00)

    def test_fok_against_partially_consumed_iceberg(self):
        """After draining some iceberg qty, FOK sees remaining total.

        Iceberg SELL total=7, peak=2. Consume peak with a NON BUY 2.
        Remaining total = 5 (hidden). New peak regenerated = 2.
        FOK BUY 5 -> total remaining = 5 >= 5 -> accepted.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        # Consume first peak with NON BUY 2 MW
        non_buy = _order(Side.BUY, 50.00, 2.0, 2, agent_id=2, timestamp=2)
        state, r_non = book.submit_order(state, non_buy)
        assert r_non.traded == True
        assert r_non.trade_qty == approx(2.0)

        # FOK BUY 5 MW -- remaining total = 5 (was 7, consumed 2)
        fok = _order(Side.BUY, 50.00, 5.0, 3, agent_id=2, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(5.0)

    def test_fok_against_partially_consumed_iceberg_overshoot(self):
        """After consuming 2 of 7, FOK for 6 -> remaining 5 < 6 -> rejected."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        non_buy = _order(Side.BUY, 50.00, 2.0, 2, agent_id=2, timestamp=2)
        state, _ = book.submit_order(state, non_buy)

        fok = _order(Side.BUY, 50.00, 6.0, 3, agent_id=2, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_fok_against_empty_book_rejected(self):
        """FOK against empty book -> rejected (no iceberg or limit orders)."""
        book = _make_book()
        state = book.init()

        fok = _order(Side.BUY, 50.00, 1.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_fok_against_iceberg_with_no_price_crossing_rejected(self):
        """FOK BUY at 49.00 against iceberg SELL at 50.00 -> no crossing."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 49.00, 3.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_fok_minimum_quantity_against_large_iceberg(self):
        """FOK BUY 0.1 MW (minimum) against iceberg total=100 -> accepted."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 100.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=5.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 0.1, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(0.1)


# ---------------------------------------------------------------------------
# 3. FOK visibility with mixed book (limit + iceberg)
# ---------------------------------------------------------------------------

class TestFOKVisibilityWithMixedBook:
    """FOK pre-check sums limit orders + total iceberg quantity."""

    def test_fok_sees_limit_plus_total_iceberg(self):
        """Limit SELL 3 + iceberg SELL total=7 (peak=2) = 10 MW matchable.

        FOK BUY 8 -> 8 <= 10 -> accepted.
        """
        book = _make_book()
        state = book.init()

        # Limit SELL 3 MW at 50.00
        lim = _order(Side.SELL, 50.00, 3.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        # Iceberg SELL: total=7, peak=2 at 50.00
        ice = _order(Side.SELL, 50.00, 7.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        # FOK BUY 8 MW -- limit(3) + iceberg_total(7) = 10 >= 8
        fok = _order(Side.BUY, 50.00, 8.0, 3, agent_id=3, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(8.0)

    def test_fok_rejected_when_limit_plus_total_iceberg_insufficient(self):
        """Limit SELL 3 + iceberg total=7 = 10. FOK BUY 11 -> rejected."""
        book = _make_book()
        state = book.init()

        lim = _order(Side.SELL, 50.00, 3.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        ice = _order(Side.SELL, 50.00, 7.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 11.0, 3, agent_id=3, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_fok_with_multiple_icebergs_at_same_price(self):
        """Two icebergs at 50.00: total=5 (peak=1) + total=4 (peak=1) = 9 MW.

        FOK BUY 9 -> accepted.
        FOK BUY 10 -> rejected.
        """
        book = _make_book()
        state = book.init()

        ice1 = _order(Side.SELL, 50.00, 5.0, 1, agent_id=1, timestamp=1,
                      exec_restriction=ExecRestriction.ICE, peak_qty=1.0)
        state, _ = book.submit_order(state, ice1)

        ice2 = _order(Side.SELL, 50.00, 4.0, 2, agent_id=2, timestamp=2,
                      exec_restriction=ExecRestriction.ICE, peak_qty=1.0)
        state, _ = book.submit_order(state, ice2)

        # FOK 9 MW -> total = 5 + 4 = 9 -> accepted
        fok9 = _order(Side.BUY, 50.00, 9.0, 3, agent_id=3, timestamp=3,
                      exec_restriction=ExecRestriction.FOK)
        state, r9 = book.submit_order(state, fok9)
        assert r9.traded == True
        assert r9.trade_qty == approx(9.0)

    def test_fok_with_multiple_icebergs_overshoot(self):
        """Two icebergs total = 9. FOK BUY 10 -> rejected."""
        book = _make_book()
        state = book.init()

        ice1 = _order(Side.SELL, 50.00, 5.0, 1, agent_id=1, timestamp=1,
                      exec_restriction=ExecRestriction.ICE, peak_qty=1.0)
        state, _ = book.submit_order(state, ice1)

        ice2 = _order(Side.SELL, 50.00, 4.0, 2, agent_id=2, timestamp=2,
                      exec_restriction=ExecRestriction.ICE, peak_qty=1.0)
        state, _ = book.submit_order(state, ice2)

        fok10 = _order(Side.BUY, 50.00, 10.0, 3, agent_id=3, timestamp=3,
                       exec_restriction=ExecRestriction.FOK)
        state, r10 = book.submit_order(state, fok10)
        assert r10.traded == False

    def test_fok_across_multiple_price_levels_with_iceberg(self):
        """Limit SELL 2@49 + iceberg SELL total=5 (peak=1) @50 = 7 MW crossing.

        FOK BUY 7@50 -> crosses both levels -> accepted.
        """
        book = _make_book()
        state = book.init()

        # Limit SELL 2 MW at 49.00 (better price, matched first)
        lim = _order(Side.SELL, 49.00, 2.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        # Iceberg SELL: total=5, peak=1 at 50.00
        ice = _order(Side.SELL, 50.00, 5.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.ICE, peak_qty=1.0)
        state, _ = book.submit_order(state, ice)

        # FOK BUY 7 at 50.00 -> limit(2@49) + iceberg_total(5@50) = 7 -> accepted
        fok = _order(Side.BUY, 50.00, 7.0, 3, agent_id=3, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(7.0)

    def test_fok_does_not_count_iceberg_beyond_crossing_price(self):
        """Iceberg SELL at 52.00, FOK BUY at 50.00 -> iceberg not crossing.

        FOK should not count non-crossing iceberg's hidden quantity.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 52.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 3.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False


# ---------------------------------------------------------------------------
# 4. Bid-side iceberg symmetry (FOK SELL against iceberg BUY)
# ---------------------------------------------------------------------------

class TestFOKVisibilityBidSide:
    """Same rules apply for FOK SELL against bid-side icebergs."""

    def test_fok_sell_accepted_against_bid_iceberg(self):
        """FOK SELL 3 MW against iceberg BUY peak=2, total=7 -> accepted."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.SELL, 50.00, 3.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(3.0)
        assert r_fok.trade_price == approx(50.00)

    def test_fok_sell_rejected_against_bid_iceberg_total_insufficient(self):
        """FOK SELL 8 MW against iceberg BUY total=7 -> rejected."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.SELL, 50.00, 8.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_fok_sell_against_bid_limit_plus_iceberg(self):
        """Limit BUY 4 + iceberg BUY total=6 (peak=2) at 50.00 = 10 MW.

        FOK SELL 9 -> accepted.
        """
        book = _make_book()
        state = book.init()

        lim = _order(Side.BUY, 50.00, 4.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        ice = _order(Side.BUY, 50.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.SELL, 50.00, 9.0, 3, agent_id=3, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(9.0)


# ---------------------------------------------------------------------------
# 5. ICEwPI icebergs and FOK visibility
# ---------------------------------------------------------------------------

class TestFOKVisibilityICEwPI:
    """FOK pre-check must count hidden quantity from ICEwPI icebergs too.

    ICEwPI slice prices change per regeneration, but the PRE-CHECK only
    needs to know whether sufficient total quantity exists at crossing prices.
    Hidden quantity from ICEwPI icebergs at the original price (or any
    crossing price) must be included.
    """

    def test_fok_sees_icewpi_total_quantity(self):
        """ICEwPI SELL total=6, peak=2, ppd=0.50 at price=50.00.

        All hidden quantity originates from this iceberg. FOK BUY 5@51
        should see total remaining = 6 >= 5 -> accepted.

        (Future slices will be at 50.50, 51.00 -- all crossing 51.00.)
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
                     ppd=0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 51.00, 5.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(5.0)

    def test_fok_icewpi_hidden_qty_at_non_crossing_price_excluded(self):
        """ICEwPI SELL total=6, peak=2 at 50.00, ppd=+1.00.

        Slices: 50.00 (peak=2 MW), 51.00 (2 MW), 52.00 (2 MW).
        FOK BUY at 50.50, qty=3.

        M6.5 slice-by-slice pre-check (OQ-11, market-spec §4.2.1 Rule 1):
        only slices whose projected drifted price actually crosses the FOK
        limit are counted as matchable. Slice 0 at 50.00 <= 50.50 crosses
        (2 MW matchable). Slice 1 at 51.00 > 50.50 does NOT cross. Slice 2
        at 52.00 > 50.50 does NOT cross. Total matchable = 2 MW < 3 MW
        required -> FOK is rejected.

        Prior to M6.5 all hidden quantity was counted when the base price
        crossed, which would have accepted this FOK. The M6.5 story resolves
        this ambiguity in favor of slice-by-slice counting.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
                     ppd=1.00)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.50, 3.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False
        assert r_fok.trade_qty == approx(0.0)


# ---------------------------------------------------------------------------
# 6. Rejected FOK does not alter book state
# ---------------------------------------------------------------------------

class TestFOKBookStatePreservation:
    """A rejected FOK must leave the book completely unchanged."""

    def test_rejected_fok_preserves_iceberg_state(self):
        """FOK BUY 8 rejected against iceberg total=7. Book unchanged after.

        After rejection, same iceberg still has total=7 available, verified
        by a subsequent FOK BUY 7 that succeeds.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        # Rejected FOK
        fok_reject = _order(Side.BUY, 50.00, 8.0, 2, agent_id=2, timestamp=2,
                            exec_restriction=ExecRestriction.FOK)
        state, r_reject = book.submit_order(state, fok_reject)
        assert r_reject.traded == False

        # Book should be unchanged -- FOK BUY 7 now succeeds
        fok_accept = _order(Side.BUY, 50.00, 7.0, 3, agent_id=2, timestamp=3,
                            exec_restriction=ExecRestriction.FOK)
        state, r_accept = book.submit_order(state, fok_accept)
        assert r_accept.traded == True
        assert r_accept.trade_qty == approx(7.0)

    def test_rejected_fok_preserves_lob_snapshot(self):
        """LOB snapshot is identical before and after a rejected FOK."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        # Capture LOB before
        lob_before = book.get_anonymous_lob(state)

        # Rejected FOK
        fok_reject = _order(Side.BUY, 50.00, 8.0, 2, agent_id=2, timestamp=2,
                            exec_restriction=ExecRestriction.FOK)
        state, r_reject = book.submit_order(state, fok_reject)
        assert r_reject.traded == False

        # LOB after should match LOB before
        lob_after = book.get_anonymous_lob(state)
        assert lob_before == lob_after

    def test_multiple_rejected_foks_preserve_state(self):
        """Three consecutive rejected FOKs, then a valid FOK succeeds."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        # Three rejected FOKs (all > 7 MW)
        for i, qty in enumerate([8.0, 10.0, 999.0], start=2):
            fok = _order(Side.BUY, 50.00, qty, i, agent_id=2, timestamp=i,
                         exec_restriction=ExecRestriction.FOK)
            state, r = book.submit_order(state, fok)
            assert r.traded == False

        # Valid FOK for exact total
        fok_ok = _order(Side.BUY, 50.00, 7.0, 10, agent_id=2, timestamp=10,
                        exec_restriction=ExecRestriction.FOK)
        state, r_ok = book.submit_order(state, fok_ok)
        assert r_ok.traded == True
        assert r_ok.trade_qty == approx(7.0)


# ---------------------------------------------------------------------------
# 7. Settlement value correctness
# ---------------------------------------------------------------------------

class TestFOKIcebergSettlement:
    """Settlement value for FOK trades against icebergs.

    V = Q (MW) x p (EUR/MWh) x 0.25 (h) for 15-min contracts.
    """

    def test_settlement_value_fok_against_iceberg(self):
        """FOK BUY 3 MW @ 50.00 -> V = 3 * 50 * 0.25 = 37.50 EUR."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 3.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.settlement_value == approx(37.50)

    def test_settlement_value_fok_exact_total(self):
        """FOK BUY 7 MW @ 50.00 -> V = 7 * 50 * 0.25 = 87.50 EUR."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 7.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.settlement_value == approx(87.50)
