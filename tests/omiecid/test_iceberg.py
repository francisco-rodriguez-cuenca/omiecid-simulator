"""Tests for M4 -- Iceberg Orders (M4.1 Standard + M4.2 ICEwPI).

Behavioral contracts for iceberg order support in ContractBook and
order_validator.

All expected values are derived from:
  - docs/omiecid/market-spec.md §4.2 (iceberg orders, ICEwPI)
  - docs/omiecid/market-spec.md §4.5 (execution restrictions: ICE)
  - docs/omiecid/market-spec.md §5.2 (timestamp rules for iceberg slices)
  - docs/omiecid/market-spec.md §6.2 (regular matching, iceberg rule §6.2.7)
  - docs/omiecid/migration-plan.md §M4 (iceberg implementation, tests)
  - docs/omiecid/agile-plan.md M4.1 + M4.2 (acceptance criteria)

Domain rules:
  - Iceberg orders have exec_restriction=ICE (IntEnum value 3).
  - Only peak_qty is visible in the order book; hidden quantity is tracked
    internally by ContractBook.
  - When a peak slice is fully consumed, a new slice of min(peak_qty,
    remaining_hidden) enters the book with a NEW timestamp.
  - Partial fills of the peak do NOT trigger new slice generation.
  - When total quantity is exhausted (remaining_hidden == 0 and peak
    consumed), no more slices are generated.
  - Icebergs cannot combine with IOC or FOK (ICE replaces NON for icebergs).
  - Validation: peak_qty > 0, peak_qty <= quantity.
  - Anonymous LOB shows only visible peak quantity.

Architecture review decisions (M4.1):
  - ICE submit sends only peak_qty to jaxob, not total quantity.
  - Slice regeneration happens after each submit_order dispatch.
  - New slices reuse the original order_id (previous slice fully consumed).
  - New slices get a fresh, monotonically-increasing timestamp.
  - ContractBook tracks iceberg metadata in a Python side-channel dict.

These tests are black-box: they import only the public interface and
assert against domain-specified values. No implementation assumptions.

Seven acceptance-criteria categories:
  1. test_iceberg_shows_peak_only        -- visible qty = peak_qty, not total
  2. test_iceberg_full_fill_of_peak_generates_new_slice -- new slice after peak consumed
  3. test_iceberg_new_slice_new_timestamp -- regenerated slices have later timestamp
  4. test_iceberg_non_restriction_only   -- ICE+IOC and ICE+FOK rejected
  5. test_iceberg_partial_peak_fill      -- partial fill does NOT trigger new slice
  6. test_iceberg_total_quantity_exhausted -- no more slices when total consumed
  7. test_iceberg_in_anonymous_lob       -- anonymous LOB shows peak only

M4.2 acceptance-criteria category (ICEwPI — Iceberg with Price Increment):
  8. test_iceberg_price_increment_per_slice -- ppd applied to each new slice

M4.2 edge-case categories:
  9. ICEwPI buy direction: negative ppd decreases price per slice
  10. ICEwPI sell direction: positive ppd increases price per slice
  11. ICEwPI ppd=0 backward compat: behaves like standard iceberg
  12. ICEwPI multiple slices accumulate price delta
  13. ICEwPI boundary prices: slice price near price bounds

Additional edge-case categories (M4.1):
  14. Iceberg validation: peak_qty > 0, peak_qty <= quantity
  15. ICE orders now accepted by validator (M1.3 rejection removed)
  16. Iceberg cancel cleans up internal state
  17. Iceberg expire_at_gct cleans up internal state
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
# Helpers: make orders with custom fields
# ---------------------------------------------------------------------------

def _make_iceberg_buy(
    order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
    agent_id=0, timestamp=100, ppd=0.0,
):
    """Create a valid ICE buy order."""
    return make_order(
        contract_id=37, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.ICE, peak_qty=peak_qty, ppd=ppd,
    )


def _make_iceberg_sell(
    order_id=2, price=50.00, quantity=30.0, peak_qty=10.0,
    agent_id=1, timestamp=200, ppd=0.0,
):
    """Create a valid ICE sell order."""
    return make_order(
        contract_id=37, side=Side.SELL, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=ExecRestriction.ICE, peak_qty=peak_qty, ppd=ppd,
    )


def _make_buy(
    order_id=10, price=50.00, quantity=10.0, agent_id=2, timestamp=500,
    exec_restriction=ExecRestriction.NON,
):
    """Create a regular NON buy order (aggressor)."""
    return make_order(
        contract_id=37, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction,
    )


def _make_sell(
    order_id=20, price=50.00, quantity=10.0, agent_id=3, timestamp=600,
    exec_restriction=ExecRestriction.NON,
):
    """Create a regular NON sell order (aggressor)."""
    return make_order(
        contract_id=37, side=Side.SELL, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction,
    )


# ===================================================================
# 1. test_iceberg_shows_peak_only
#    Domain rule: Only peak_qty is visible in the order book, not
#    the total quantity (market-spec §4.2).
# ===================================================================

class TestIcebergShowsPeakOnly:
    """Iceberg visible quantity in book equals peak_qty, not total."""

    def test_buy_iceberg_shows_peak_qty_in_book(self, book, empty_state):
        """Submit a buy iceberg with total=30 MW, peak=10 MW.
        The book should show 10 MW visible, not 30 MW."""
        iceberg = _make_iceberg_buy(quantity=30.0, peak_qty=10.0)
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True

        # Read back the order from the book
        order_in_book = book.get_order(state, iceberg.order_id, Side.BUY)
        assert not _is_absent(order_in_book)
        assert order_in_book.quantity == pytest.approx(10.0, abs=0.01), \
            "Book should show peak_qty (10.0 MW), not total (30.0 MW)"

    def test_sell_iceberg_shows_peak_qty_in_book(self, book, empty_state):
        """Submit a sell iceberg with total=20 MW, peak=5 MW.
        The book should show 5 MW visible, not 20 MW."""
        iceberg = _make_iceberg_sell(quantity=20.0, peak_qty=5.0)
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True

        order_in_book = book.get_order(state, iceberg.order_id, Side.SELL)
        assert not _is_absent(order_in_book)
        assert order_in_book.quantity == pytest.approx(5.0, abs=0.01), \
            "Book should show peak_qty (5.0 MW), not total (20.0 MW)"

    def test_iceberg_peak_equals_total_shows_full_qty(self, book, empty_state):
        """When peak_qty == quantity, entire quantity is visible (edge case)."""
        iceberg = _make_iceberg_buy(quantity=10.0, peak_qty=10.0)
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True

        order_in_book = book.get_order(state, iceberg.order_id, Side.BUY)
        assert not _is_absent(order_in_book)
        assert order_in_book.quantity == pytest.approx(10.0, abs=0.01)


# ===================================================================
# 2. test_iceberg_full_fill_of_peak_generates_new_slice
#    Domain rule: When a slice is exhausted, a new slice of peak size
#    is activated (market-spec §4.2).
# ===================================================================

class TestIcebergFullFillGeneratesNewSlice:
    """When peak is fully consumed, a new slice appears with peak_qty."""

    def test_buy_iceberg_regenerates_after_peak_filled(self, book, empty_state):
        """Submit buy iceberg (total=30, peak=10). Sell 10 MW against it.
        A new slice of 10 MW should appear in the book."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Aggressive sell at 50.00 for 10 MW fills the entire peak
        aggressor = _make_sell(
            order_id=20, price=50.00, quantity=10.0, agent_id=3, timestamp=500,
        )
        state, result = book.submit_order(state, aggressor)
        assert result.traded == True
        assert result.trade_qty == pytest.approx(10.0, abs=0.01)

        # New slice should appear in book (same order_id or new)
        # The book should still have a buy order at this price level
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) > 0, "New slice should have regenerated on bid side"
        bid_at_50 = [qty for price, qty in bids if abs(price - 50.00) < 0.02]
        assert len(bid_at_50) > 0, "Regenerated slice should be at price 50.00"
        assert bid_at_50[0] == pytest.approx(10.0, abs=0.01), \
            "Regenerated slice should show peak_qty = 10.0 MW"

    def test_sell_iceberg_regenerates_after_peak_filled(self, book, empty_state):
        """Submit sell iceberg (total=20, peak=5). Buy 5 MW against it.
        A new slice of 5 MW should appear."""
        iceberg = _make_iceberg_sell(
            order_id=2, price=45.00, quantity=20.0, peak_qty=5.0,
            agent_id=1, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        aggressor = _make_buy(
            order_id=10, price=45.00, quantity=5.0, agent_id=2, timestamp=500,
        )
        state, result = book.submit_order(state, aggressor)
        assert result.traded == True
        assert result.trade_qty == pytest.approx(5.0, abs=0.01)

        # New slice should appear on ask side
        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) > 0, "New slice should have regenerated on ask side"
        ask_at_45 = [qty for price, qty in asks if abs(price - 45.00) < 0.02]
        assert len(ask_at_45) > 0, "Regenerated slice should be at price 45.00"
        assert ask_at_45[0] == pytest.approx(5.0, abs=0.01), \
            "Regenerated slice should show peak_qty = 5.0 MW"

    def test_multiple_slices_regenerate_sequentially(self, book, empty_state):
        """Submit buy iceberg (total=30, peak=10). Fill 3 successive slices
        of 10 MW each. All three fills should succeed."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        total_filled = 0.0
        for i in range(3):
            aggressor = _make_sell(
                order_id=20 + i, price=50.00, quantity=10.0,
                agent_id=3, timestamp=500 + i * 100,
            )
            state, result = book.submit_order(state, aggressor)
            assert result.traded == True, f"Slice {i + 1} should trade"
            assert result.trade_qty == pytest.approx(10.0, abs=0.01)
            total_filled += result.trade_qty

        assert total_filled == pytest.approx(30.0, abs=0.01), \
            "Total filled across all slices should equal original quantity"

    def test_last_slice_uses_remaining_hidden(self, book, empty_state):
        """When remaining_hidden < peak_qty, last slice has reduced quantity.
        Iceberg: total=25, peak=10. After 2 full slices (20 MW), remaining
        hidden = 5 MW. Third slice should be 5 MW, not 10 MW."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=25.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Fill first two full slices
        for i in range(2):
            aggressor = _make_sell(
                order_id=20 + i, price=50.00, quantity=10.0,
                agent_id=3, timestamp=500 + i * 100,
            )
            state, result = book.submit_order(state, aggressor)
            assert result.traded == True

        # Third slice should be 5.0 MW (remaining_hidden)
        bids, _ = book.get_anonymous_lob(state)
        bid_at_50 = [qty for price, qty in bids if abs(price - 50.00) < 0.02]
        assert len(bid_at_50) > 0, "Reduced last slice should be in book"
        assert bid_at_50[0] == pytest.approx(5.0, abs=0.01), \
            "Last slice should be min(peak_qty=10, remaining_hidden=5) = 5.0 MW"


# ===================================================================
# 3. test_iceberg_new_slice_new_timestamp
#    Domain rule: Regenerated slices have a new (later) timestamp,
#    losing time priority (market-spec §5.2).
# ===================================================================

class TestIcebergNewSliceNewTimestamp:
    """Regenerated slices have a new, later timestamp."""

    def test_regenerated_slice_has_later_timestamp(self, book, empty_state):
        """After peak fill, the regenerated slice's timestamp must be
        strictly greater than the original iceberg's timestamp."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Fill the first peak
        aggressor = _make_sell(
            order_id=20, price=50.00, quantity=10.0,
            agent_id=3, timestamp=500,
        )
        state, _ = book.submit_order(state, aggressor)

        # Read back the regenerated slice
        regen_order = book.get_order(state, iceberg.order_id, Side.BUY)
        assert not _is_absent(regen_order), "Regenerated slice should exist in book"
        assert regen_order.timestamp > iceberg.timestamp, \
            f"Regenerated slice timestamp ({regen_order.timestamp}) must be " \
            f"strictly greater than original ({iceberg.timestamp})"

    def test_regenerated_slice_loses_time_priority(self, book, empty_state):
        """A regular order placed AFTER the original iceberg but BEFORE
        the slice regeneration should have time priority over the new slice.

        Setup: iceberg buy at 50.00 (t=100, peak=10), regular buy at 50.00
        (t=200). Sell 10 MW at 50.00 fills the iceberg peak. Regenerated
        slice gets t > 200, so the regular order now has time priority.
        Next sell should fill the regular order, not the iceberg slice."""
        # Iceberg buy
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Regular buy at same price, later timestamp
        regular = _make_buy(
            order_id=5, price=50.00, quantity=10.0,
            agent_id=2, timestamp=200,
        )
        state, _ = book.submit_order(state, regular)

        # Fill iceberg peak (iceberg has time priority: t=100 < t=200)
        sell1 = _make_sell(
            order_id=20, price=50.00, quantity=10.0,
            agent_id=3, timestamp=500,
        )
        state, result1 = book.submit_order(state, sell1)
        assert result1.traded == True

        # Now the regular order (t=200) should have time priority over
        # the regenerated iceberg slice (t > 500). Next sell should
        # fill the regular order.
        sell2 = _make_sell(
            order_id=21, price=50.00, quantity=10.0,
            agent_id=3, timestamp=600,
        )
        state, result2 = book.submit_order(state, sell2)
        assert result2.traded == True

        # The regular order (id=5) should be gone from the book
        regular_in_book = book.get_order(state, 5, Side.BUY)
        assert _is_absent(regular_in_book), \
            "Regular order should have been filled (had time priority " \
            "over regenerated iceberg slice)"

    def test_successive_slices_have_increasing_timestamps(self, book, empty_state):
        """Each successive slice must have a strictly later timestamp
        than the previous one."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        timestamps = []
        for i in range(2):
            # Record timestamp of current slice before filling it
            current = book.get_order(state, iceberg.order_id, Side.BUY)
            assert not _is_absent(current)
            timestamps.append(current.timestamp)

            # Fill the slice
            aggressor = _make_sell(
                order_id=20 + i, price=50.00, quantity=10.0,
                agent_id=3, timestamp=500 + i * 100,
            )
            state, _ = book.submit_order(state, aggressor)

        # Record third slice timestamp
        third = book.get_order(state, iceberg.order_id, Side.BUY)
        assert not _is_absent(third)
        timestamps.append(third.timestamp)

        # Each timestamp should be strictly increasing
        for j in range(len(timestamps) - 1):
            assert timestamps[j + 1] > timestamps[j], \
                f"Slice {j + 2} timestamp ({timestamps[j + 1]}) must be " \
                f"greater than slice {j + 1} timestamp ({timestamps[j]})"


# ===================================================================
# 4. test_iceberg_non_restriction_only
#    Domain rule: Icebergs cannot use IOC/FOK (market-spec §4.2, §4.5).
#    ICE replaces NON for icebergs.
# ===================================================================

class TestIcebergNonRestrictionOnly:
    """Icebergs cannot combine with IOC or FOK execution restrictions."""

    def test_ioc_with_peak_qty_treated_as_regular_ioc(self, book, empty_state):
        """An IOC order with peak_qty set is treated as regular IOC
        (peak_qty ignored for non-ICE orders). No iceberg behavior.
        The constraint is that ICE orders cannot also be IOC/FOK --
        the reverse (IOC with peak_qty) is just a regular IOC."""
        # Place resting sell so IOC has something to match
        resting = _make_sell(
            order_id=20, price=50.00, quantity=5.0,
            agent_id=3, timestamp=100,
            exec_restriction=ExecRestriction.NON,
        )
        state, _ = book.submit_order(empty_state, resting)

        ioc_order = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=30.0,
            order_id=1, agent_id=0, timestamp=200,
            exec_restriction=ExecRestriction.IOC,
            peak_qty=10.0, ppd=0.0,
        )
        state, result = book.submit_order(state, ioc_order)
        # Should fill 5 MW (available), cancel remainder (IOC behavior)
        assert result.traded == True
        assert result.trade_qty == pytest.approx(5.0, abs=0.01)
        # No order should remain in book (IOC cancels remainder)
        order_in_book = book.get_order(state, 1, Side.BUY)
        assert _is_absent(order_in_book), \
            "IOC remainder should be cancelled, not treated as iceberg"

    def test_iceberg_exec_restriction_is_ice_not_ioc(self, book, empty_state):
        """An ICE order is accepted; an IOC iceberg is not a valid concept.
        ICE is the only valid restriction for iceberg behavior."""
        ice_order = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
        )
        state, result = book.submit_order(empty_state, ice_order)
        assert result.accepted == True, "ICE order should be accepted"

    def test_fok_order_with_peak_qty_is_regular_fok(self, book, empty_state):
        """FOK with peak_qty > 0 is treated as regular FOK (peak_qty
        ignored for non-ICE orders). No iceberg behavior."""
        # Place resting sell first so FOK can match
        resting = _make_sell(
            order_id=20, price=50.00, quantity=30.0,
            agent_id=3, timestamp=100,
            exec_restriction=ExecRestriction.NON,
        )
        state, _ = book.submit_order(empty_state, resting)

        fok_order = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=30.0,
            order_id=1, agent_id=0, timestamp=200,
            exec_restriction=ExecRestriction.FOK,
            peak_qty=10.0, ppd=0.0,
        )
        state, result = book.submit_order(state, fok_order)
        # FOK should fill fully (30 MW available), treated as regular FOK
        assert result.traded == True
        assert result.trade_qty == pytest.approx(30.0, abs=0.01)

    def test_ice_order_accepted_by_validator(self, book, empty_state):
        """M4.1 removes the blanket ICE rejection from M1.3 validator.
        Valid ICE orders must now be accepted."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
        )
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True, \
            "ICE orders must be accepted after M4.1 (M1.3 rejection removed)"


# ===================================================================
# 5. test_iceberg_partial_peak_fill
#    Domain rule: Partial fill of peak does NOT trigger new slice
#    (market-spec §4.2 — slice regeneration only on exhaustion).
# ===================================================================

class TestIcebergPartialPeakFill:
    """Partial fill of peak does NOT trigger new slice."""

    def test_partial_fill_leaves_peak_remainder_in_book(self, book, empty_state):
        """Submit buy iceberg (total=30, peak=10). Sell 5 MW. The book
        should still show 5 MW remaining from the SAME slice, not a
        new slice of 10 MW."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Partial fill: sell only 5 MW
        aggressor = _make_sell(
            order_id=20, price=50.00, quantity=5.0,
            agent_id=3, timestamp=500,
        )
        state, result = book.submit_order(state, aggressor)
        assert result.traded == True
        assert result.trade_qty == pytest.approx(5.0, abs=0.01)

        # The remaining visible quantity should be 5 MW, not 10 MW
        order_in_book = book.get_order(state, iceberg.order_id, Side.BUY)
        assert not _is_absent(order_in_book), "Partially filled iceberg should still be in book"
        assert order_in_book.quantity == pytest.approx(5.0, abs=0.01), \
            "Partial fill should leave 5.0 MW remainder, not regenerate a new 10 MW slice"

    def test_partial_fill_preserves_original_timestamp(self, book, empty_state):
        """Partial fill should NOT change the slice's timestamp (no
        regeneration means timestamp is preserved)."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Get timestamp of original slice
        original = book.get_order(state, iceberg.order_id, Side.BUY)
        original_ts = original.timestamp

        # Partial fill
        aggressor = _make_sell(
            order_id=20, price=50.00, quantity=3.0,
            agent_id=3, timestamp=500,
        )
        state, _ = book.submit_order(state, aggressor)

        # Timestamp should be unchanged
        after_partial = book.get_order(state, iceberg.order_id, Side.BUY)
        assert not _is_absent(after_partial)
        assert after_partial.timestamp == original_ts, \
            "Partial fill must not change slice timestamp"

    def test_partial_then_full_fill_triggers_regeneration(self, book, empty_state):
        """Partial fill (5 MW) then remainder fill (5 MW) should trigger
        slice regeneration on the second fill that exhausts the peak."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Partial fill: 5 MW
        sell1 = _make_sell(
            order_id=20, price=50.00, quantity=5.0,
            agent_id=3, timestamp=500,
        )
        state, _ = book.submit_order(state, sell1)

        # Remainder fill: 5 MW (exhausts peak)
        sell2 = _make_sell(
            order_id=21, price=50.00, quantity=5.0,
            agent_id=3, timestamp=600,
        )
        state, _ = book.submit_order(state, sell2)

        # New slice should have regenerated
        regen = book.get_order(state, iceberg.order_id, Side.BUY)
        assert not _is_absent(regen), "New slice should regenerate after peak exhaustion"
        assert regen.quantity == pytest.approx(10.0, abs=0.01), \
            "New slice should have full peak_qty = 10.0 MW"
        assert regen.timestamp > iceberg.timestamp, \
            "Regenerated slice must have a later timestamp"


# ===================================================================
# 6. test_iceberg_total_quantity_exhausted
#    Domain rule: When total quantity is consumed, no more slices
#    are generated.
# ===================================================================

class TestIcebergTotalQuantityExhausted:
    """No new slice when hidden quantity is zero."""

    def test_no_slice_after_total_consumed(self, book, empty_state):
        """Submit buy iceberg (total=20, peak=10). Fill 10 + 10 = 20 MW.
        No more slices should exist in the book."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=20.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Fill first slice
        sell1 = _make_sell(
            order_id=20, price=50.00, quantity=10.0,
            agent_id=3, timestamp=500,
        )
        state, _ = book.submit_order(state, sell1)

        # Fill second (last) slice
        sell2 = _make_sell(
            order_id=21, price=50.00, quantity=10.0,
            agent_id=3, timestamp=600,
        )
        state, _ = book.submit_order(state, sell2)

        # No more slices should be in the book
        order_in_book = book.get_order(state, iceberg.order_id, Side.BUY)
        assert _is_absent(order_in_book), \
            "No slice should remain after total quantity is exhausted"

    def test_no_slice_after_total_consumed_uneven(self, book, empty_state):
        """Iceberg total=25, peak=10. Fill 10 + 10 + 5 = 25 MW.
        After third slice consumed, no more in book."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=25.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        fills = [10.0, 10.0, 5.0]
        for i, qty in enumerate(fills):
            aggressor = _make_sell(
                order_id=20 + i, price=50.00, quantity=qty,
                agent_id=3, timestamp=500 + i * 100,
            )
            state, result = book.submit_order(state, aggressor)
            assert result.traded == True, f"Fill {i + 1} should succeed"

        order_in_book = book.get_order(state, iceberg.order_id, Side.BUY)
        assert _is_absent(order_in_book), \
            "No slice should remain after total quantity exhausted"

    def test_anonymous_lob_empty_after_total_consumed(self, book, empty_state):
        """After total quantity is consumed, the anonymous LOB should
        show no orders at the iceberg price level."""
        iceberg = _make_iceberg_sell(
            order_id=2, price=55.00, quantity=10.0, peak_qty=5.0,
            agent_id=1, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Fill first slice
        buy1 = _make_buy(
            order_id=10, price=55.00, quantity=5.0,
            agent_id=2, timestamp=500,
        )
        state, _ = book.submit_order(state, buy1)

        # Fill second (last) slice
        buy2 = _make_buy(
            order_id=11, price=55.00, quantity=5.0,
            agent_id=2, timestamp=600,
        )
        state, _ = book.submit_order(state, buy2)

        bids, asks = book.get_anonymous_lob(state)
        asks_at_55 = [qty for price, qty in asks if abs(price - 55.00) < 0.02]
        assert len(asks_at_55) == 0, \
            "No asks should remain at iceberg price after total consumed"


# ===================================================================
# 7. test_iceberg_in_anonymous_lob
#    Domain rule: Anonymous LOB shows only visible peak quantity
#    (market-spec §4.2, migration-plan §M4).
# ===================================================================

class TestIcebergInAnonymousLob:
    """Anonymous LOB shows only visible peak quantity."""

    def test_buy_iceberg_anonymous_lob_shows_peak(self, book, empty_state):
        """Buy iceberg (total=30, peak=10) at 50.00: anonymous LOB bid
        side shows 10 MW, not 30 MW."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) > 0, "Bid should appear in anonymous LOB"
        # Find the bid at 50.00
        bid_at_50 = [(p, q) for p, q in bids if abs(p - 50.00) < 0.02]
        assert len(bid_at_50) == 1, "Exactly one bid level at 50.00"
        assert bid_at_50[0][1] == pytest.approx(10.0, abs=0.01), \
            "Anonymous LOB should show peak_qty (10.0 MW), not total (30.0 MW)"

    def test_sell_iceberg_anonymous_lob_shows_peak(self, book, empty_state):
        """Sell iceberg (total=20, peak=5) at 55.00: anonymous LOB ask
        side shows 5 MW, not 20 MW."""
        iceberg = _make_iceberg_sell(
            order_id=2, price=55.00, quantity=20.0, peak_qty=5.0,
            agent_id=1, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) > 0, "Ask should appear in anonymous LOB"
        ask_at_55 = [(p, q) for p, q in asks if abs(p - 55.00) < 0.02]
        assert len(ask_at_55) == 1, "Exactly one ask level at 55.00"
        assert ask_at_55[0][1] == pytest.approx(5.0, abs=0.01), \
            "Anonymous LOB should show peak_qty (5.0 MW), not total (20.0 MW)"

    def test_anonymous_lob_no_agent_ids_for_iceberg(self, book, empty_state):
        """Anonymous LOB must not expose agent IDs, including for icebergs.
        (This is a general LOB rule; verify it holds for icebergs too.)"""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=42, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        bids, asks = book.get_anonymous_lob(state)
        # Each entry is (price, qty) — no agent_id field
        for entry in bids:
            assert len(entry) == 2, \
                "Anonymous LOB entries must be (price, qty) only — no agent ID"

    def test_regenerated_slice_shows_peak_in_lob(self, book, empty_state):
        """After peak fill + regeneration, the new slice should also
        show only peak_qty in the anonymous LOB."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Fill the first peak
        aggressor = _make_sell(
            order_id=20, price=50.00, quantity=10.0,
            agent_id=3, timestamp=500,
        )
        state, _ = book.submit_order(state, aggressor)

        # Check anonymous LOB for regenerated slice
        bids, _ = book.get_anonymous_lob(state)
        bid_at_50 = [(p, q) for p, q in bids if abs(p - 50.00) < 0.02]
        assert len(bid_at_50) == 1, "Regenerated slice should appear in LOB"
        assert bid_at_50[0][1] == pytest.approx(10.0, abs=0.01), \
            "Regenerated slice should show peak_qty in LOB"


# ===================================================================
# 8. Edge cases: Iceberg validation
# ===================================================================

class TestIcebergValidation:
    """Iceberg-specific validation: peak_qty > 0, peak_qty <= quantity."""

    def test_peak_qty_zero_rejected(self, book, empty_state):
        """ICE order with peak_qty=0 must be rejected."""
        bad_iceberg = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=30.0,
            order_id=1, agent_id=0, timestamp=100,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=0.0, ppd=0.0,
        )
        state, result = book.submit_order(empty_state, bad_iceberg)
        assert result.accepted == False, \
            "ICE order with peak_qty=0 must be rejected"

    def test_peak_qty_greater_than_total_rejected(self, book, empty_state):
        """ICE order with peak_qty > quantity must be rejected."""
        bad_iceberg = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=10.0,
            order_id=1, agent_id=0, timestamp=100,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=20.0, ppd=0.0,
        )
        state, result = book.submit_order(empty_state, bad_iceberg)
        assert result.accepted == False, \
            "ICE order with peak_qty > quantity must be rejected"

    def test_peak_qty_negative_rejected(self, book, empty_state):
        """ICE order with peak_qty < 0 must be rejected."""
        bad_iceberg = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=30.0,
            order_id=1, agent_id=0, timestamp=100,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=-5.0, ppd=0.0,
        )
        state, result = book.submit_order(empty_state, bad_iceberg)
        assert result.accepted == False, \
            "ICE order with peak_qty < 0 must be rejected"

    def test_valid_iceberg_accepted(self, book, empty_state):
        """ICE order with valid peak_qty (0 < peak <= quantity) accepted."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
        )
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True

    def test_peak_qty_equals_quantity_accepted(self, book, empty_state):
        """Edge case: peak_qty == quantity is valid (fully visible iceberg)."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=10.0, peak_qty=10.0,
        )
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True

    def test_iceberg_minimum_peak_qty(self, book, empty_state):
        """Peak quantity must respect qty_min (0.1 MW)."""
        iceberg = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=10.0,
            order_id=1, agent_id=0, timestamp=100,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=0.1, ppd=0.0,
        )
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True, \
            "peak_qty=0.1 (qty_min) should be accepted"


# ===================================================================
# 9. Edge case: Iceberg cancel cleans up state
# ===================================================================

class TestIcebergCancel:
    """Cancelling an iceberg should remove it from book and internal state."""

    def test_cancel_iceberg_removes_from_book(self, book, empty_state):
        """Cancel an iceberg: no visible slice should remain."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        state = book.cancel_order(state, iceberg.order_id, Side.BUY)

        order_in_book = book.get_order(state, iceberg.order_id, Side.BUY)
        assert _is_absent(order_in_book), \
            "Cancelled iceberg should not be in book"

    def test_cancel_iceberg_prevents_further_slices(self, book, empty_state):
        """After cancelling an iceberg, no further slices should appear
        even if there was remaining hidden quantity."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)
        state = book.cancel_order(state, iceberg.order_id, Side.BUY)

        # Book should be empty on bid side at 50.00
        bids, _ = book.get_anonymous_lob(state)
        bid_at_50 = [qty for price, qty in bids if abs(price - 50.00) < 0.02]
        assert len(bid_at_50) == 0, \
            "No slices should remain or regenerate after cancel"


# ===================================================================
# 10. Edge case: expire_at_gct cleans up iceberg state
# ===================================================================

class TestIcebergExpireAtGct:
    """GCT expiry should clear iceberg state along with all orders."""

    def test_expire_clears_iceberg(self, book, empty_state):
        """After expire_at_gct, no iceberg slices remain."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        state = book.expire_at_gct(state)

        order_in_book = book.get_order(state, iceberg.order_id, Side.BUY)
        assert _is_absent(order_in_book), \
            "Iceberg should be cleared after GCT expiry"

        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0, "No bids after GCT expiry"
        assert len(asks) == 0, "No asks after GCT expiry"


# ===================================================================
# M4.2 — Iceberg with Price Increment (ICEwPI)
#
# Domain rules (market-spec §4.2):
#   - ICEwPI: each new slice price = original_price + ppd * slice_count
#   - Buy icebergs: negative ppd (price decreases per slice)
#   - Sell icebergs: positive ppd (price increases per slice)
#   - ppd=0 means standard iceberg (no price change) — backward compat
#   - ppd sign is a convention, not a validator-enforced rule
#   - Formula uses multiplicative offset from original price to avoid
#     floating-point drift (architecture review decision)
#
# Architecture review (M4.2):
#   - No validator changes: ppd is not validated for sign
#   - ppd stored in iceberg metadata alongside original_price
#   - slice_price = original_price + ppd * slice_count
#   - Standard icebergs (ppd=0) unchanged: price + 0*N == price
# ===================================================================


# ===================================================================
# 8. test_iceberg_price_increment_per_slice (M4.2 acceptance criterion)
#    Domain rule: ICEwPI adjusts price by ppd on each new slice
#    (market-spec §4.2, migration-plan §M4).
# ===================================================================

class TestIcebergPriceIncrementPerSlice:
    """ICEwPI: each new slice adjusts price by ppd delta."""

    def test_buy_iceberg_ppd_negative_price_decreases(self, book, empty_state):
        """Buy iceberg with ppd=-0.50: each new slice price decreases.
        Original price=50.00. Slice 1 (initial) at 50.00.
        After fill, slice 2 at 50.00 + (-0.50)*1 = 49.50.
        After fill, slice 3 at 50.00 + (-0.50)*2 = 49.00."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100, ppd=-0.50,
        )
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True

        # Slice 1 is at 50.00
        order = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(order)
        assert order.price == pytest.approx(50.00, abs=0.02)

        # Fill slice 1
        sell1 = _make_sell(
            order_id=20, price=50.00, quantity=10.0,
            agent_id=3, timestamp=500,
        )
        state, r1 = book.submit_order(state, sell1)
        assert r1.traded == True

        # Slice 2 should be at 49.50
        order2 = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(order2), "Slice 2 should have regenerated"
        assert order2.price == pytest.approx(49.50, abs=0.02), \
            "Slice 2 price should be 50.00 + (-0.50)*1 = 49.50"

        # Fill slice 2
        sell2 = _make_sell(
            order_id=21, price=49.50, quantity=10.0,
            agent_id=3, timestamp=600,
        )
        state, r2 = book.submit_order(state, sell2)
        assert r2.traded == True

        # Slice 3 should be at 49.00
        order3 = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(order3), "Slice 3 should have regenerated"
        assert order3.price == pytest.approx(49.00, abs=0.02), \
            "Slice 3 price should be 50.00 + (-0.50)*2 = 49.00"

    def test_sell_iceberg_ppd_positive_price_increases(self, book, empty_state):
        """Sell iceberg with ppd=+1.00: each new slice price increases.
        Original price=40.00. Slice 1 at 40.00.
        After fill, slice 2 at 40.00 + 1.00*1 = 41.00.
        After fill, slice 3 at 40.00 + 1.00*2 = 42.00."""
        iceberg = _make_iceberg_sell(
            order_id=2, price=40.00, quantity=30.0, peak_qty=10.0,
            agent_id=1, timestamp=100, ppd=1.00,
        )
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True

        # Slice 1 at 40.00
        order = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(order)
        assert order.price == pytest.approx(40.00, abs=0.02)

        # Fill slice 1
        buy1 = _make_buy(
            order_id=10, price=40.00, quantity=10.0,
            agent_id=2, timestamp=500,
        )
        state, r1 = book.submit_order(state, buy1)
        assert r1.traded == True

        # Slice 2 at 41.00
        order2 = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(order2), "Slice 2 should have regenerated"
        assert order2.price == pytest.approx(41.00, abs=0.02), \
            "Slice 2 price should be 40.00 + 1.00*1 = 41.00"

        # Fill slice 2
        buy2 = _make_buy(
            order_id=11, price=41.00, quantity=10.0,
            agent_id=2, timestamp=600,
        )
        state, r2 = book.submit_order(state, buy2)
        assert r2.traded == True

        # Slice 3 at 42.00
        order3 = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(order3), "Slice 3 should have regenerated"
        assert order3.price == pytest.approx(42.00, abs=0.02), \
            "Slice 3 price should be 40.00 + 1.00*2 = 42.00"

    def test_ppd_zero_behaves_as_standard_iceberg(self, book, empty_state):
        """ppd=0.0 must produce identical behavior to standard iceberg:
        all slices at original price. Backward compatibility with M4.1."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100, ppd=0.0,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        for i in range(3):
            order = book.get_order(state, 1, Side.BUY)
            assert not _is_absent(order), f"Slice {i + 1} should exist"
            assert order.price == pytest.approx(50.00, abs=0.02), \
                f"Slice {i + 1} with ppd=0 must stay at original price 50.00"

            aggressor = _make_sell(
                order_id=20 + i, price=50.00, quantity=10.0,
                agent_id=3, timestamp=500 + i * 100,
            )
            state, result = book.submit_order(state, aggressor)
            assert result.traded == True

        # All 30 MW consumed; no more slices
        remaining = book.get_order(state, 1, Side.BUY)
        assert _is_absent(remaining), "All quantity exhausted"

    def test_multiple_slices_accumulate_price_delta(self, book, empty_state):
        """Buy iceberg: total=50, peak=10, ppd=-0.10. Five slices at:
        50.00, 49.90, 49.80, 49.70, 49.60. Each successive slice must
        reflect the cumulative offset from original price."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=50.0, peak_qty=10.0,
            agent_id=0, timestamp=100, ppd=-0.10,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        expected_prices = [50.00, 49.90, 49.80, 49.70, 49.60]
        for i, expected_price in enumerate(expected_prices):
            order = book.get_order(state, 1, Side.BUY)
            assert not _is_absent(order), f"Slice {i + 1} should exist"
            assert order.price == pytest.approx(expected_price, abs=0.02), \
                f"Slice {i + 1} price should be {expected_price}"

            # Fill the slice (sell at the slice price)
            aggressor = _make_sell(
                order_id=20 + i, price=expected_price, quantity=10.0,
                agent_id=3, timestamp=500 + i * 100,
            )
            state, result = book.submit_order(state, aggressor)
            assert result.traded == True, f"Slice {i + 1} should trade"

    def test_icewpi_slice_appears_in_anonymous_lob_at_new_price(self, book, empty_state):
        """After slice regeneration with ppd, the anonymous LOB should show
        the new slice at the adjusted price, not the original price."""
        iceberg = _make_iceberg_sell(
            order_id=2, price=45.00, quantity=20.0, peak_qty=10.0,
            agent_id=1, timestamp=100, ppd=0.50,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Verify slice 1 in LOB at 45.00
        _, asks = book.get_anonymous_lob(state)
        ask_at_45 = [q for p, q in asks if abs(p - 45.00) < 0.02]
        assert len(ask_at_45) > 0, "Slice 1 should be in LOB at 45.00"

        # Fill slice 1
        buy1 = _make_buy(
            order_id=10, price=45.00, quantity=10.0,
            agent_id=2, timestamp=500,
        )
        state, _ = book.submit_order(state, buy1)

        # Slice 2 should be in LOB at 45.50, NOT at 45.00
        _, asks2 = book.get_anonymous_lob(state)
        ask_at_45_after = [q for p, q in asks2 if abs(p - 45.00) < 0.02]
        assert len(ask_at_45_after) == 0, \
            "No ask at original price 45.00 after slice with ppd regenerates"
        ask_at_45_50 = [q for p, q in asks2 if abs(p - 45.50) < 0.02]
        assert len(ask_at_45_50) > 0, \
            "Slice 2 should appear in LOB at adjusted price 45.50"
        assert ask_at_45_50[0] == pytest.approx(10.0, abs=0.01)

    def test_icewpi_last_slice_reduced_qty_at_adjusted_price(self, book, empty_state):
        """ICEwPI with uneven total: total=25, peak=10, ppd=-0.20.
        Slice 3 has qty=5 (remaining_hidden) at price 50.00 + (-0.20)*2 = 49.60."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=25.0, peak_qty=10.0,
            agent_id=0, timestamp=100, ppd=-0.20,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Fill slice 1 at 50.00
        sell1 = _make_sell(
            order_id=20, price=50.00, quantity=10.0,
            agent_id=3, timestamp=500,
        )
        state, _ = book.submit_order(state, sell1)

        # Fill slice 2 at 49.80
        sell2 = _make_sell(
            order_id=21, price=49.80, quantity=10.0,
            agent_id=3, timestamp=600,
        )
        state, _ = book.submit_order(state, sell2)

        # Slice 3 should be 5 MW at 49.60
        order3 = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(order3), "Reduced last slice should exist"
        assert order3.quantity == pytest.approx(5.0, abs=0.01), \
            "Last slice qty should be min(peak_qty=10, remaining=5) = 5.0"
        assert order3.price == pytest.approx(49.60, abs=0.02), \
            "Last slice price should be 50.00 + (-0.20)*2 = 49.60"

    def test_icewpi_new_slice_gets_new_timestamp(self, book, empty_state):
        """ICEwPI slices must also get new timestamps (same rule as
        standard iceberg, but verify it still holds with ppd)."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100, ppd=-0.50,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        original = book.get_order(state, 1, Side.BUY)
        ts_original = original.timestamp

        # Fill slice 1
        sell1 = _make_sell(
            order_id=20, price=50.00, quantity=10.0,
            agent_id=3, timestamp=500,
        )
        state, _ = book.submit_order(state, sell1)

        regen = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(regen)
        assert regen.timestamp > ts_original, \
            "ICEwPI regenerated slice must have a later timestamp"


# ===================================================================
# M4.2 edge cases: boundary prices
# ===================================================================

class TestIcebergPriceIncrementBoundary:
    """Edge cases for ICEwPI near price bounds and special values."""

    def test_icewpi_buy_ppd_reaching_near_price_min(self, book, empty_state):
        """Buy iceberg with large negative ppd reaching near price_min.
        Slices should still be placed at the computed price as long as
        it is within [-9999, 9999]."""
        # price=-9998.00, ppd=-0.50, peak=0.1, total=0.3
        # Slice 1: -9998.00, Slice 2: -9998.50, Slice 3: -9999.00
        iceberg = _make_iceberg_buy(
            order_id=1, price=-9998.00, quantity=0.3, peak_qty=0.1,
            agent_id=0, timestamp=100, ppd=-0.50,
        )
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True

        # Fill slice 1 at -9998.00
        sell1 = _make_sell(
            order_id=20, price=-9998.00, quantity=0.1,
            agent_id=3, timestamp=500,
        )
        state, _ = book.submit_order(state, sell1)

        # Slice 2 at -9998.50
        order2 = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(order2)
        assert order2.price == pytest.approx(-9998.50, abs=0.02), \
            "Slice 2 price should be -9998.00 + (-0.50)*1 = -9998.50"

    def test_icewpi_sell_ppd_reaching_near_price_max(self, book, empty_state):
        """Sell iceberg with large positive ppd reaching near price_max.
        Slices should still be placed at the computed price as long as
        it is within [-9999, 9999]."""
        # price=9998.00, ppd=+0.50, peak=0.1, total=0.3
        # Slice 1: 9998.00, Slice 2: 9998.50, Slice 3: 9999.00
        iceberg = _make_iceberg_sell(
            order_id=2, price=9998.00, quantity=0.3, peak_qty=0.1,
            agent_id=1, timestamp=100, ppd=0.50,
        )
        state, result = book.submit_order(empty_state, iceberg)
        assert result.accepted == True

        # Fill slice 1
        buy1 = _make_buy(
            order_id=10, price=9998.00, quantity=0.1,
            agent_id=2, timestamp=500,
        )
        state, _ = book.submit_order(state, buy1)

        # Slice 2 at 9998.50
        order2 = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(order2)
        assert order2.price == pytest.approx(9998.50, abs=0.02), \
            "Slice 2 price should be 9998.00 + 0.50*1 = 9998.50"

    def test_icewpi_ppd_on_tick_boundary(self, book, empty_state):
        """ppd=0.01 (minimum tick): price increments by exactly one tick
        per slice. Verifies tick-aligned prices."""
        iceberg = _make_iceberg_sell(
            order_id=2, price=50.00, quantity=0.3, peak_qty=0.1,
            agent_id=1, timestamp=100, ppd=0.01,
        )
        state, _ = book.submit_order(empty_state, iceberg)

        # Fill slice 1 at 50.00
        buy1 = _make_buy(
            order_id=10, price=50.00, quantity=0.1,
            agent_id=2, timestamp=500,
        )
        state, _ = book.submit_order(state, buy1)

        # Slice 2 should be at 50.01
        order2 = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(order2)
        assert order2.price == pytest.approx(50.01, abs=0.005), \
            "Slice 2 price should be 50.00 + 0.01*1 = 50.01"

        # Fill slice 2 at 50.01
        buy2 = _make_buy(
            order_id=11, price=50.01, quantity=0.1,
            agent_id=2, timestamp=600,
        )
        state, _ = book.submit_order(state, buy2)

        # Slice 3 at 50.02
        order3 = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(order3)
        assert order3.price == pytest.approx(50.02, abs=0.005), \
            "Slice 3 price should be 50.00 + 0.01*2 = 50.02"

    def test_icewpi_cancel_clears_ppd_state(self, book, empty_state):
        """After cancelling an ICEwPI iceberg, no further slices
        should appear (ppd metadata cleaned up)."""
        iceberg = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=0, timestamp=100, ppd=-0.50,
        )
        state, _ = book.submit_order(empty_state, iceberg)
        state = book.cancel_order(state, 1, Side.BUY)

        order = book.get_order(state, 1, Side.BUY)
        assert _is_absent(order), "Cancelled ICEwPI should not remain in book"

        bids, _ = book.get_anonymous_lob(state)
        assert len(bids) == 0, "No bids after cancelling ICEwPI iceberg"

    def test_icewpi_expire_at_gct_clears_ppd_state(self, book, empty_state):
        """expire_at_gct must also clear ICEwPI iceberg state."""
        iceberg = _make_iceberg_sell(
            order_id=2, price=40.00, quantity=20.0, peak_qty=10.0,
            agent_id=1, timestamp=100, ppd=1.00,
        )
        state, _ = book.submit_order(empty_state, iceberg)
        state = book.expire_at_gct(state)

        order = book.get_order(state, 2, Side.SELL)
        assert _is_absent(order), "ICEwPI should be cleared after GCT expiry"

        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 0, "No asks after GCT expiry"
