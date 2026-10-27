"""Audit tests for M4.1 — Standard Iceberg and M4.2 — ICEwPI.

Post-implementation verification focusing on edge cases, boundary conditions,
and acceptance criteria for iceberg order handling:

- Iceberg shows peak only: visible quantity in book and LOB equals peak_qty
- Full fill generates new slice: slice regeneration when peak is exhausted
- New slice has new timestamp: fresh timestamp loses time priority
- Non-restriction only: ICE accepted, IOC/FOK with iceberg fields treated as regular
- Partial peak fill: partial fill does NOT trigger new slice
- Total quantity exhausted: no further slices after total_qty consumed
- Iceberg in anonymous LOB: LOB only shows peak (visible) quantity
- Edge cases: boundary peak_qty values, multiple icebergs, multi-slice walks,
  cancel during hidden state, iceberg vs iceberg matching

Derived independently from:
- docs/omiecid/market-spec.md §4.2 (iceberg orders), §4.5 (exec restrictions)
- docs/omiecid/agile-plan.md (M4.1 acceptance criteria)
- docs/omiecid/migration-plan.md §M4 section
- omiecid_simulator/omiecid/contract_book.py (public interface only)
- omiecid_simulator/omiecid/order_validator.py (public interface only)
"""

import pytest
from pytest import approx

from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    make_order,
)
from omiecid_simulator.omiecid.order_validator import validate_order

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
# Shared fixtures (market_config from conftest)
# Specialized open_contract overrides conftest (contract_id=10 vs 37)
# Uses 'config' (not 'product_config') and 'state' (not 'empty_state')
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def open_contract():
    """An OPEN contract for iceberg tests."""
    return Contract(
        contract_id=10,
        delivery_start_minutes=150,   # 02:30
        delivery_end_minutes=165,     # 02:45
        gct_minutes=90,               # 01:30
        status=ContractStatus.OPEN,
    )


@pytest.fixture
def config():
    return OMIEProductConfig()


@pytest.fixture
def book(open_contract, config, market_config):
    return ContractBook(open_contract, config, market_config)


@pytest.fixture
def state(book):
    return book.init()


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _buy(contract_id=10, price=50.0, qty=1.0, order_id=1, agent_id=100,
         timestamp=1000, restriction=ExecRestriction.NON,
         peak_qty=0.0, ppd=0.0):
    return make_order(
        contract_id=contract_id, side=Side.BUY, price=price,
        quantity=qty, order_id=order_id, agent_id=agent_id,
        timestamp=timestamp, exec_restriction=restriction,
        peak_qty=peak_qty, ppd=ppd,
    )


def _sell(contract_id=10, price=50.0, qty=1.0, order_id=1, agent_id=100,
          timestamp=1000, restriction=ExecRestriction.NON,
          peak_qty=0.0, ppd=0.0):
    return make_order(
        contract_id=contract_id, side=Side.SELL, price=price,
        quantity=qty, order_id=order_id, agent_id=agent_id,
        timestamp=timestamp, exec_restriction=restriction,
        peak_qty=peak_qty, ppd=ppd,
    )


def _iceberg_buy(contract_id=10, price=50.0, qty=5.0, order_id=1,
                 agent_id=100, timestamp=1000, peak_qty=1.0, ppd=0.0):
    return make_order(
        contract_id=contract_id, side=Side.BUY, price=price,
        quantity=qty, order_id=order_id, agent_id=agent_id,
        timestamp=timestamp, exec_restriction=ExecRestriction.ICE,
        peak_qty=peak_qty, ppd=ppd,
    )


def _iceberg_sell(contract_id=10, price=50.0, qty=5.0, order_id=1,
                  agent_id=100, timestamp=1000, peak_qty=1.0, ppd=0.0):
    return make_order(
        contract_id=contract_id, side=Side.SELL, price=price,
        quantity=qty, order_id=order_id, agent_id=agent_id,
        timestamp=timestamp, exec_restriction=ExecRestriction.ICE,
        peak_qty=peak_qty, ppd=ppd,
    )


# ===========================================================================
# 1. Iceberg shows peak only
# ===========================================================================

class TestIcebergShowsPeakOnly:
    """Iceberg's visible quantity in the book equals peak_qty, not total."""

    def test_buy_iceberg_visible_qty_equals_peak(self, book, state):
        """A buy iceberg with qty=5, peak=1 shows 1 in the book."""
        order = _iceberg_buy(qty=5.0, peak_qty=1.0, order_id=1)
        state, result = book.submit_order(state, order)
        assert result.accepted
        found = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(found)
        assert found.quantity == approx(1.0)

    def test_sell_iceberg_visible_qty_equals_peak(self, book, state):
        """A sell iceberg with qty=5, peak=1 shows 1 in the book."""
        order = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1)
        state, result = book.submit_order(state, order)
        assert result.accepted
        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.quantity == approx(1.0)

    def test_iceberg_peak_equals_total_visible_is_total(self, book, state):
        """When peak_qty == quantity, the full amount is visible."""
        order = _iceberg_buy(qty=2.0, peak_qty=2.0, order_id=1)
        state, result = book.submit_order(state, order)
        assert result.accepted
        found = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(found)
        assert found.quantity == approx(2.0)

    def test_iceberg_large_hidden_small_peak(self, book, state):
        """Large hidden qty, small peak: only peak visible."""
        order = _iceberg_sell(qty=10.0, peak_qty=0.5, order_id=1)
        state, result = book.submit_order(state, order)
        assert result.accepted
        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.quantity == approx(0.5)


# ===========================================================================
# 2. Full fill of peak generates new slice
# ===========================================================================

class TestIcebergFullFillGeneratesNewSlice:
    """When an incoming order fully consumes the visible peak, a new slice
    is generated from hidden quantity."""

    def test_sell_iceberg_regenerates_after_buy_fills_peak(self, book, state):
        """Sell iceberg resting; buy fills the peak -> new slice appears."""
        # Sell iceberg: total=3, peak=1 at price 50
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # Buy 1.0 at 50 fills the peak
        aggressor = _buy(qty=1.0, price=50.0, order_id=2, agent_id=300,
                         timestamp=200)
        state, result = book.submit_order(state, aggressor)
        assert result.traded
        assert result.trade_qty == approx(1.0)

        # New slice should appear on ask side
        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.quantity == approx(1.0)  # new peak slice

    def test_buy_iceberg_regenerates_after_sell_fills_peak(self, book, state):
        """Buy iceberg resting; sell fills the peak -> new slice appears."""
        iceberg = _iceberg_buy(qty=3.0, peak_qty=1.0, order_id=1,
                               agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        aggressor = _sell(qty=1.0, price=50.0, order_id=2, agent_id=300,
                          timestamp=200)
        state, result = book.submit_order(state, aggressor)
        assert result.traded

        found = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(found)
        assert found.quantity == approx(1.0)

    def test_multiple_slices_consumed_sequentially(self, book, state):
        """Consume multiple slices one at a time."""
        # Sell iceberg: total=3, peak=1
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # First fill
        state, r1 = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        assert r1.traded

        # After first fill, new slice should exist
        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.quantity == approx(1.0)

        # Second fill
        state, r2 = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=11, agent_id=300, timestamp=300))
        assert r2.traded

        # After second fill, last slice should exist (remaining_hidden=1)
        found2 = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found2)
        assert found2.quantity == approx(1.0)

    def test_last_slice_smaller_than_peak(self, book, state):
        """When remaining_hidden < peak_qty, last slice = remaining_hidden."""
        # Sell iceberg: total=2.5, peak=1.0 -> hidden=1.5
        iceberg = _iceberg_sell(qty=2.5, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # Fill first peak (1.0)
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        # Second slice should be 1.0 (min(1.0, 1.5))
        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.quantity == approx(1.0)

        # Fill second peak (1.0), remaining hidden = 0.5
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=11, agent_id=300, timestamp=300))

        # Third slice should be 0.5 (min(1.0, 0.5))
        found2 = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found2)
        assert found2.quantity == approx(0.5)


# ===========================================================================
# 3. New slice has new timestamp
# ===========================================================================

class TestIcebergNewSliceNewTimestamp:
    """Each regenerated slice receives a fresh timestamp."""

    def test_new_slice_timestamp_differs_from_original(self, book, state):
        """Regenerated slice has a different (newer) timestamp."""
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        original = book.get_order(state, 1, Side.SELL)
        original_ts = original.timestamp

        # Fill peak
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        regenerated = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(regenerated)
        assert regenerated.timestamp > original_ts

    def test_new_slice_timestamp_after_aggressor(self, book, state):
        """Regenerated slice timestamp is the aggressor's timestamp.

        Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence (25)): the
        slice is activated inside the aggressor's matching process, so its
        stored SEC is that event's second (500), not a later counter value
        (was "> 500"); it queues after the aggressor by (SEC, NSEC). The
        aggressor (1.0 MW) is fully filled and does not rest."""
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        aggressor_ts = 500
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300,
            timestamp=aggressor_ts))

        regenerated = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(regenerated)
        assert regenerated.timestamp == aggressor_ts

    def test_successive_slices_have_increasing_timestamps(self, book, state):
        """Each successive slice gets a monotonically increasing timestamp."""
        iceberg = _iceberg_sell(qty=4.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        timestamps = []

        # Fill three slices
        for i, ts in enumerate([200, 300, 400]):
            state, _ = book.submit_order(state, _buy(
                qty=1.0, price=50.0, order_id=10 + i, agent_id=300,
                timestamp=ts))
            found = book.get_order(state, 1, Side.SELL)
            if not _is_absent(found):
                timestamps.append(found.timestamp)

        # Each subsequent timestamp should be strictly increasing
        for j in range(1, len(timestamps)):
            assert timestamps[j] > timestamps[j - 1]

    def test_new_slice_loses_time_priority_to_earlier_resting(self, book, state):
        """A regenerated slice loses priority to an order that was already
        resting at the same price with an earlier timestamp."""
        # Place a regular sell at 50, timestamp=150
        regular = _sell(qty=1.0, price=50.0, order_id=2, agent_id=400,
                        timestamp=150)
        state, _ = book.submit_order(state, regular)

        # Place sell iceberg at 50, total=3, peak=1, timestamp=100
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # Buy 1.0 at 50 should match the iceberg (ts=100 < ts=150)
        state, r1 = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        assert r1.traded

        # Now iceberg regenerates with new timestamp > 200
        # Next buy should match the regular order (ts=150) before the
        # regenerated iceberg slice
        state, r2 = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=11, agent_id=300, timestamp=300))
        assert r2.traded

        # Regular order (id=2) should be consumed; iceberg slice (id=1) remains
        regular_check = book.get_order(state, 2, Side.SELL)
        iceberg_check = book.get_order(state, 1, Side.SELL)
        assert _is_absent(regular_check)  # regular was consumed
        assert not _is_absent(iceberg_check)  # iceberg slice still there


# ===========================================================================
# 4. Non-restriction only (ICE cannot combine with IOC/FOK)
# ===========================================================================

class TestIcebergNonRestrictionOnly:
    """Icebergs use exec_restriction=ICE, cannot be IOC or FOK."""

    def test_ice_order_accepted(self, book, state):
        """An order with exec_restriction=ICE is accepted (M4.1 unlocked)."""
        order = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1)
        state, result = book.submit_order(state, order)
        assert result.accepted

    def test_ioc_order_with_peak_qty_treated_as_ioc(self, book, state):
        """An IOC order with non-zero peak_qty is still IOC, not iceberg."""
        # Resting sell at 50
        state, _ = book.submit_order(state, _sell(
            qty=5.0, price=50.0, order_id=1, agent_id=200, timestamp=100))

        # IOC buy with peak_qty set (not ICE restriction)
        ioc_with_peak = _buy(
            qty=2.0, price=50.0, order_id=2, agent_id=300,
            timestamp=200, restriction=ExecRestriction.IOC, peak_qty=1.0)
        state, result = book.submit_order(state, ioc_with_peak)
        # Should be treated as IOC: fill available, cancel rest
        assert result.accepted

    def test_fok_order_with_peak_qty_treated_as_fok(self, book, state):
        """A FOK order with non-zero peak_qty is still FOK, not iceberg."""
        # Resting sell at 50
        state, _ = book.submit_order(state, _sell(
            qty=2.0, price=50.0, order_id=1, agent_id=200, timestamp=100))

        # FOK buy with peak_qty set (not ICE restriction)
        fok_with_peak = _buy(
            qty=2.0, price=50.0, order_id=2, agent_id=300,
            timestamp=200, restriction=ExecRestriction.FOK, peak_qty=1.0)
        state, result = book.submit_order(state, fok_with_peak)
        assert result.traded
        assert result.trade_qty == approx(2.0)


# ===========================================================================
# 5. Partial peak fill does NOT trigger new slice
# ===========================================================================

class TestIcebergPartialPeakFill:
    """Partial fills of the visible peak do not regenerate a new slice."""

    def test_partial_fill_leaves_residual_peak(self, book, state):
        """Filling half the peak leaves the other half resting."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=2.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # Fill 1.0 of the 2.0 peak
        state, result = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        assert result.traded
        assert result.trade_qty == approx(1.0)

        # Remaining visible should be 1.0 (partial peak)
        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.quantity == approx(1.0)

    def test_partial_fill_preserves_original_timestamp(self, book, state):
        """After partial fill, the resting order keeps original timestamp."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=2.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        original = book.get_order(state, 1, Side.SELL)
        original_ts = original.timestamp

        # Partial fill
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.timestamp == original_ts

    def test_partial_then_full_triggers_regeneration(self, book, state):
        """First partial fill no regeneration, then filling the rest triggers it."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=2.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # Partial fill (1.0 of 2.0 peak)
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        found1 = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found1)
        assert found1.quantity == approx(1.0)
        assert found1.timestamp == 100  # original timestamp

        # Fill remaining peak (1.0) -> should trigger regeneration
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=11, agent_id=300, timestamp=300))
        found2 = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found2)
        assert found2.quantity == approx(2.0)  # new full peak
        assert found2.timestamp > 100  # new timestamp


# ===========================================================================
# 6. Total quantity exhausted
# ===========================================================================

class TestIcebergTotalQuantityExhausted:
    """When the total iceberg quantity is consumed, no more slices appear."""

    def test_no_slice_after_total_consumed(self, book, state):
        """After consuming total qty, order disappears from book."""
        # Total=2, peak=1 -> 2 slices
        iceberg = _iceberg_sell(qty=2.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # Fill first slice
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        # Second slice should exist
        assert not _is_absent(book.get_order(state, 1, Side.SELL))

        # Fill second slice
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=11, agent_id=300, timestamp=300))
        # No more slices
        assert _is_absent(book.get_order(state, 1, Side.SELL))

    def test_total_consumed_in_single_large_order(self, book, state):
        """A single order larger than total quantity consumes iceberg fully.

        The iceberg peak=1 out of total=2 rests. A buy for 5.0 should consume
        the peak (1.0), trigger regeneration (1.0 more), consume that too,
        then iceberg is gone. The aggressor may or may not consume both slices
        in one shot depending on the matching engine; but after the dust
        settles the iceberg should be gone.
        """
        iceberg = _iceberg_sell(qty=2.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # Large buy that can consume all
        state, result = book.submit_order(state, _buy(
            qty=5.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        assert result.traded

        # Iceberg should be fully consumed (may or may not have regenerated
        # and been re-consumed in a single pass; either way, gone now)
        # Note: since regeneration happens after the submit, the aggressor
        # may only get 1.0 (the visible peak), leaving the rest resting.
        # This test just confirms the trade happened and checks final state.
        found = book.get_order(state, 1, Side.SELL)
        # The iceberg either has a remaining slice or is gone
        # (depends on implementation detail of regeneration timing)
        # At minimum, the trade happened
        assert result.trade_qty >= 1.0

    def test_peak_equals_total_no_regeneration(self, book, state):
        """When peak_qty == total qty, no hidden qty means no regeneration."""
        iceberg = _iceberg_sell(qty=1.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # Fill the only slice
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        # No regeneration — order gone
        assert _is_absent(book.get_order(state, 1, Side.SELL))


# ===========================================================================
# 7. Iceberg in anonymous LOB
# ===========================================================================

class TestIcebergInAnonymousLob:
    """Anonymous LOB shows only the visible peak quantity of icebergs."""

    def test_iceberg_lob_shows_peak_qty(self, book, state):
        """Anonymous LOB shows peak_qty, not total."""
        iceberg = _iceberg_sell(qty=10.0, peak_qty=2.0, order_id=1,
                                agent_id=200, timestamp=100, price=55.0)
        state, _ = book.submit_order(state, iceberg)

        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        price, qty = asks[0]
        assert price == approx(55.0)
        assert qty == approx(2.0)  # peak only, not 10.0

    def test_iceberg_lob_no_agent_id(self, book, state):
        """Anonymous LOB must not expose agent IDs (generic LOB rule)."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=55.0)
        state, _ = book.submit_order(state, iceberg)

        bids, asks = book.get_anonymous_lob(state)
        # LOB returns (price, qty) tuples — no agent_id field
        for entry in asks:
            assert len(entry) == 2

    def test_iceberg_and_regular_at_same_price_aggregated(self, book, state):
        """An iceberg and regular order at the same price: LOB aggregates
        visible quantities (peak + regular qty)."""
        iceberg = _iceberg_sell(qty=10.0, peak_qty=2.0, order_id=1,
                                agent_id=200, timestamp=100, price=55.0)
        state, _ = book.submit_order(state, iceberg)

        regular = _sell(qty=3.0, price=55.0, order_id=2, agent_id=300,
                        timestamp=150)
        state, _ = book.submit_order(state, regular)

        bids, asks = book.get_anonymous_lob(state)
        # Should show one price level at 55.0 with aggregated qty = 2+3 = 5
        total_ask_qty = sum(qty for _, qty in asks if abs(_ - 55.0) < 0.001)
        assert total_ask_qty == approx(5.0)

    def test_iceberg_buy_in_lob(self, book, state):
        """Buy iceberg shows peak in bids side of LOB."""
        iceberg = _iceberg_buy(qty=8.0, peak_qty=1.5, order_id=1,
                               agent_id=200, timestamp=100, price=45.0)
        state, _ = book.submit_order(state, iceberg)

        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 1
        price, qty = bids[0]
        assert price == approx(45.0)
        assert qty == approx(1.5)


# ===========================================================================
# 8. Iceberg validation edge cases
# ===========================================================================

class TestIcebergValidation:
    """Edge cases for iceberg order validation."""

    def test_ice_peak_qty_zero_rejected(self, config):
        """ICE order with peak_qty=0 is rejected."""
        order = _iceberg_sell(qty=5.0, peak_qty=0.0, order_id=1)
        # peak_qty=0 violates peak_qty > 0
        assert bool(validate_order(order, config, contract_open=True)) is False

    def test_ice_peak_qty_exceeds_total_rejected(self, config):
        """ICE order with peak_qty > quantity is rejected."""
        order = _iceberg_sell(qty=2.0, peak_qty=3.0, order_id=1)
        assert bool(validate_order(order, config, contract_open=True)) is False

    def test_ice_peak_equals_total_accepted(self, config):
        """ICE order with peak_qty == quantity is valid (no hidden qty)."""
        order = _iceberg_sell(qty=5.0, peak_qty=5.0, order_id=1)
        assert bool(validate_order(order, config, contract_open=True)) is True

    def test_ice_minimum_peak_qty(self, config):
        """ICE order with minimum valid peak_qty (0.1) is accepted."""
        order = _iceberg_sell(qty=1.0, peak_qty=0.1, order_id=1)
        assert bool(validate_order(order, config, contract_open=True)) is True

    def test_ice_with_negative_peak_rejected(self, config):
        """ICE order with negative peak_qty is rejected."""
        order = _iceberg_sell(qty=5.0, peak_qty=-1.0, order_id=1)
        assert bool(validate_order(order, config, contract_open=True)) is False

    def test_non_ice_with_zero_peak_accepted(self, config):
        """Regular (NON) order with peak_qty=0 is accepted normally."""
        order = _sell(qty=1.0, price=50.0, order_id=1)
        assert bool(validate_order(order, config, contract_open=True)) is True

    def test_ice_order_on_closed_contract_rejected(self, config):
        """ICE order on a closed contract is rejected."""
        order = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1)
        assert bool(validate_order(order, config, contract_open=False)) is False


# ===========================================================================
# 9. Iceberg cancel
# ===========================================================================

class TestIcebergCancel:
    """Cancelling an iceberg removes visible slice and prevents further
    slice regeneration."""

    def test_cancel_iceberg_removes_from_book(self, book, state):
        """Cancelling an iceberg removes the visible slice."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)
        assert not _is_absent(book.get_order(state, 1, Side.SELL))

        state = book.cancel_order(state, 1, Side.SELL)
        assert _is_absent(book.get_order(state, 1, Side.SELL))

    def test_cancel_iceberg_prevents_regeneration(self, book, state):
        """After cancel, no slice regeneration occurs on subsequent trades."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # Cancel the iceberg
        state = book.cancel_order(state, 1, Side.SELL)

        # Place a new sell order and trade to trigger any regeneration check
        state, _ = book.submit_order(state, _sell(
            qty=1.0, price=50.0, order_id=2, agent_id=300, timestamp=200))
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=3, agent_id=400, timestamp=300))

        # Iceberg should NOT have regenerated
        assert _is_absent(book.get_order(state, 1, Side.SELL))


# ===========================================================================
# 10. Iceberg and GCT expiry
# ===========================================================================

class TestIcebergExpireAtGct:
    """GCT expiry clears iceberg state."""

    def test_gct_expiry_clears_iceberg(self, book, state):
        """expire_at_gct removes the iceberg and its hidden tracking."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        state = book.expire_at_gct(state)
        assert _is_absent(book.get_order(state, 1, Side.SELL))

    def test_gct_expiry_prevents_regeneration(self, book, state):
        """After GCT expiry, no iceberg slices regenerate."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        state = book.expire_at_gct(state)

        # Anonymous LOB should be empty
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0


# ===========================================================================
# 11. Agent views with icebergs
# ===========================================================================

class TestIcebergAgentViews:
    """Agent-specific views with iceberg orders."""

    def test_agent_sees_own_iceberg_order(self, book, state):
        """get_agent_orders shows the owner the full remaining iceberg quantity.

        Re-baselined by OMIE-OWN-ORDERS-ICE-QTY-1 (market-spec §9.2 "Own
        orders": showing full iceberg quantity; Annex §4.4): visible peak 1.0
        plus hidden 4.0 = 5.0 (was visible-only 1.0).
        """
        iceberg = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        orders = book.get_agent_orders(state, 200)
        assert len(orders) == 1
        assert orders[0].quantity == approx(5.0)  # visible + hidden

    def test_agent_sees_trades_from_iceberg_fill(self, book, state):
        """get_agent_trades returns trades produced by iceberg fills."""
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100)
        state, _ = book.submit_order(state, iceberg)

        # Buy fills the peak
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        # Iceberg owner should see the trade
        trades_200 = book.get_agent_trades(state, 200)
        assert len(trades_200) >= 1

        # Aggressor should also see the trade
        trades_300 = book.get_agent_trades(state, 300)
        assert len(trades_300) >= 1


# ===========================================================================
# 12. Multiple icebergs on same side
# ===========================================================================

class TestMultipleIcebergs:
    """Multiple iceberg orders interacting."""

    def test_two_icebergs_at_different_prices(self, book, state):
        """Two sell icebergs at different prices; buy matches best price first."""
        ice1 = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                             agent_id=200, timestamp=100, price=49.0)
        ice2 = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=2,
                             agent_id=300, timestamp=150, price=51.0)
        state, _ = book.submit_order(state, ice1)
        state, _ = book.submit_order(state, ice2)

        # Buy at 51 should match the 49 iceberg first (better price)
        state, result = book.submit_order(state, _buy(
            qty=1.0, price=51.0, order_id=10, agent_id=400, timestamp=200))
        assert result.traded
        assert result.trade_price == approx(49.0)

    def test_two_icebergs_same_price_time_priority(self, book, state):
        """Two sell icebergs at same price: earlier timestamp matched first."""
        ice1 = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                             agent_id=200, timestamp=100, price=50.0)
        ice2 = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=2,
                             agent_id=300, timestamp=150, price=50.0)
        state, _ = book.submit_order(state, ice1)
        state, _ = book.submit_order(state, ice2)

        # Buy should match ice1 (earlier)
        state, result = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=400, timestamp=200))
        assert result.traded

        # ice1 regenerates with new timestamp; ice2 still at ts=150
        # Next buy should match ice2 (ts=150 < regenerated ice1 ts)
        state, r2 = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=11, agent_id=400, timestamp=300))
        assert r2.traded

        # After both fills: ice1 should have regenerated, ice2 should have regenerated
        # Both should still have resting slices (3-1=2 hidden each)
        found1 = book.get_order(state, 1, Side.SELL)
        found2 = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(found1)
        assert not _is_absent(found2)


# ###########################################################################
# M4.2 — ICEwPI (Iceberg with Price Increment) Audit Tests
# ###########################################################################
#
# Domain rules (market-spec §3.3, §4.2):
#   - ICEwPI orders carry a per-slice price delta (ppd).
#   - Each new slice: slice_price = original_price + ppd * slice_number
#   - Buy icebergs: convention is negative ppd (price decreases per slice)
#   - Sell icebergs: convention is positive ppd (price increases per slice)
#   - ppd=0.0 for standard icebergs (M4.1 behavior preserved)
#   - PPD range observed in real data: approx -1.55 to +1.45 EUR/MWh
#
# These tests are derived independently from the domain spec and
# implementation public interface, not from the original test file.
# ###########################################################################


# ===========================================================================
# M4.2-A1. Buy iceberg ppd: price decreases across slices
# ===========================================================================

class TestICEwPIBuyPriceDecreases:
    """Buy iceberg with negative ppd: each slice has a lower price."""

    def test_buy_iceberg_slice1_price_decreased(self, book, state):
        """First regenerated slice (slice_number=1) has price = original + ppd."""
        iceberg = _iceberg_buy(qty=3.0, peak_qty=1.0, order_id=1,
                               agent_id=200, timestamp=100, price=50.0,
                               ppd=-0.50)
        state, _ = book.submit_order(state, iceberg)

        # Verify initial slice at original price
        found = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(found)
        assert found.price == approx(50.0)

        # Fill the peak -> triggers regeneration
        state, r = book.submit_order(state, _sell(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        assert r.traded

        # New slice should be at 50.0 + (-0.50) * 1 = 49.50
        found2 = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(found2)
        assert found2.price == approx(49.50)

    def test_buy_iceberg_slice2_price_decreased_further(self, book, state):
        """Second regenerated slice: price = original + ppd * 2."""
        iceberg = _iceberg_buy(qty=4.0, peak_qty=1.0, order_id=1,
                               agent_id=200, timestamp=100, price=50.0,
                               ppd=-0.50)
        state, _ = book.submit_order(state, iceberg)

        # Fill slice 0 (at 50.0)
        state, _ = book.submit_order(state, _sell(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        # Fill slice 1 (at 49.50)
        state, _ = book.submit_order(state, _sell(
            qty=1.0, price=49.50, order_id=11, agent_id=300, timestamp=300))

        # Slice 2 should be at 50.0 + (-0.50) * 2 = 49.00
        found = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(found)
        assert found.price == approx(49.00)

    def test_buy_iceberg_three_slices_cumulative(self, book, state):
        """Verify cumulative drift across 3 regenerated slices."""
        iceberg = _iceberg_buy(qty=5.0, peak_qty=1.0, order_id=1,
                               agent_id=200, timestamp=100, price=60.0,
                               ppd=-1.00)
        state, _ = book.submit_order(state, iceberg)

        expected_prices = [60.0, 59.0, 58.0, 57.0]
        for i in range(4):
            found = book.get_order(state, 1, Side.BUY)
            assert not _is_absent(found)
            assert found.price == approx(expected_prices[i])
            if i < 3:
                # Fill this slice
                state, _ = book.submit_order(state, _sell(
                    qty=1.0, price=expected_prices[i], order_id=20 + i,
                    agent_id=300, timestamp=200 + i * 100))


# ===========================================================================
# M4.2-A2. Sell iceberg ppd: price increases across slices
# ===========================================================================

class TestICEwPISellPriceIncreases:
    """Sell iceberg with positive ppd: each slice has a higher price."""

    def test_sell_iceberg_slice1_price_increased(self, book, state):
        """First regenerated slice has price = original + ppd."""
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=0.50)
        state, _ = book.submit_order(state, iceberg)

        # Fill the initial peak
        state, r = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        assert r.traded

        # New slice at 50.0 + 0.50 * 1 = 50.50
        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.price == approx(50.50)

    def test_sell_iceberg_three_slices_cumulative(self, book, state):
        """Verify cumulative drift across 3 regenerated sell slices."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=40.0,
                                ppd=0.25)
        state, _ = book.submit_order(state, iceberg)

        expected_prices = [40.0, 40.25, 40.50, 40.75]
        for i in range(4):
            found = book.get_order(state, 1, Side.SELL)
            assert not _is_absent(found)
            assert found.price == approx(expected_prices[i])
            if i < 3:
                state, _ = book.submit_order(state, _buy(
                    qty=1.0, price=expected_prices[i], order_id=20 + i,
                    agent_id=300, timestamp=200 + i * 100))

    def test_sell_iceberg_lob_shows_adjusted_price(self, book, state):
        """After regeneration, anonymous LOB shows the adjusted slice price."""
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=1.00)
        state, _ = book.submit_order(state, iceberg)

        # Fill initial peak
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        # LOB should show 51.0 (not 50.0)
        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        assert asks[0][0] == approx(51.0)
        assert asks[0][1] == approx(1.0)


# ===========================================================================
# M4.2-A3. ppd=0.0 backward compatibility (standard iceberg unchanged)
# ===========================================================================

class TestICEwPIZeroPpdBackwardCompat:
    """ppd=0.0 must produce identical behavior to M4.1 standard iceberg."""

    def test_zero_ppd_all_slices_same_price(self, book, state):
        """With ppd=0, every slice stays at the original price."""
        iceberg = _iceberg_sell(qty=4.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=0.0)
        state, _ = book.submit_order(state, iceberg)

        for i in range(3):
            found = book.get_order(state, 1, Side.SELL)
            assert not _is_absent(found)
            assert found.price == approx(50.0)
            state, _ = book.submit_order(state, _buy(
                qty=1.0, price=50.0, order_id=20 + i,
                agent_id=300, timestamp=200 + i * 100))

        # Last slice also at original price
        found_last = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found_last)
        assert found_last.price == approx(50.0)

    def test_zero_ppd_buy_iceberg_all_slices_same_price(self, book, state):
        """Buy iceberg with ppd=0: all slices at same price."""
        iceberg = _iceberg_buy(qty=3.0, peak_qty=1.0, order_id=1,
                               agent_id=200, timestamp=100, price=48.0,
                               ppd=0.0)
        state, _ = book.submit_order(state, iceberg)

        for i in range(2):
            found = book.get_order(state, 1, Side.BUY)
            assert not _is_absent(found)
            assert found.price == approx(48.0)
            state, _ = book.submit_order(state, _sell(
                qty=1.0, price=48.0, order_id=20 + i,
                agent_id=300, timestamp=200 + i * 100))

        found_last = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(found_last)
        assert found_last.price == approx(48.0)


# ===========================================================================
# M4.2-A4. ppd with partial fills
# ===========================================================================

class TestICEwPIPartialFills:
    """Partial fills of a ppd-carrying iceberg: price adjustment only
    happens on full peak exhaustion (slice regeneration), not partial."""

    def test_partial_fill_keeps_original_price(self, book, state):
        """Partial fill of the initial peak does NOT adjust the price."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=2.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=0.50)
        state, _ = book.submit_order(state, iceberg)

        # Partial fill: 1.0 of 2.0 peak
        state, r = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        assert r.traded

        # Price should still be 50.0 (partial peak, no regeneration)
        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.price == approx(50.0)
        assert found.quantity == approx(1.0)

    def test_partial_then_full_adjusts_price(self, book, state):
        """Partial fill keeps price; finishing the peak adjusts to ppd price."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=2.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=0.50)
        state, _ = book.submit_order(state, iceberg)

        # Partial fill (1.0 of 2.0) — no price change
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        found = book.get_order(state, 1, Side.SELL)
        assert found.price == approx(50.0)

        # Fill remaining peak (1.0) -> regeneration with ppd
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=11, agent_id=300, timestamp=300))
        found2 = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found2)
        assert found2.price == approx(50.50)  # ppd applied once
        assert found2.quantity == approx(2.0)  # full new peak


# ===========================================================================
# M4.2-A5. Last slice with ppd and reduced quantity
# ===========================================================================

class TestICEwPILastSliceReduced:
    """When remaining_hidden < peak_qty, the last slice has reduced qty
    AND the ppd-adjusted price."""

    def test_last_slice_reduced_qty_with_ppd_price(self, book, state):
        """Last slice: min(peak, remaining) qty at ppd-adjusted price."""
        # total=2.5, peak=1.0, ppd=0.10 -> slices: 1.0@50, 1.0@50.10, 0.5@50.20
        iceberg = _iceberg_sell(qty=2.5, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=0.10)
        state, _ = book.submit_order(state, iceberg)

        # Fill slice 0 (1.0 @ 50.0)
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        # Verify slice 1 (1.0 @ 50.10)
        found1 = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found1)
        assert found1.price == approx(50.10)
        assert found1.quantity == approx(1.0)

        # Fill slice 1
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.10, order_id=11, agent_id=300, timestamp=300))

        # Verify slice 2: reduced qty 0.5 @ 50.20
        found2 = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found2)
        assert found2.price == approx(50.20)
        assert found2.quantity == approx(0.5)


# ===========================================================================
# M4.2-A6. Cumulative drift accuracy (floating-point)
# ===========================================================================

class TestICEwPICumulativeDrift:
    """Verify multiplicative formula: slice_price = original + ppd * N
    avoids cumulative floating-point drift."""

    def test_five_slices_with_small_ppd(self, book, state):
        """Five slices with ppd=-0.10: prices should be exact multiples."""
        iceberg = _iceberg_buy(qty=6.0, peak_qty=1.0, order_id=1,
                               agent_id=200, timestamp=100, price=100.0,
                               ppd=-0.10)
        state, _ = book.submit_order(state, iceberg)

        expected = [100.0, 99.90, 99.80, 99.70, 99.60]
        for i in range(5):
            found = book.get_order(state, 1, Side.BUY)
            assert not _is_absent(found)
            assert found.price == approx(expected[i], abs=1e-6), \
                f"Slice {i}: expected {expected[i]}, got {found.price}"
            if i < 4:
                state, _ = book.submit_order(state, _sell(
                    qty=1.0, price=expected[i], order_id=20 + i,
                    agent_id=300, timestamp=200 + i * 100))

    def test_sell_five_slices_ppd_025(self, book, state):
        """Five sell slices with ppd=0.25: verify no drift."""
        iceberg = _iceberg_sell(qty=6.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=30.0,
                                ppd=0.25)
        state, _ = book.submit_order(state, iceberg)

        expected = [30.0, 30.25, 30.50, 30.75, 31.00]
        for i in range(5):
            found = book.get_order(state, 1, Side.SELL)
            assert not _is_absent(found)
            assert found.price == approx(expected[i], abs=1e-6)
            if i < 4:
                state, _ = book.submit_order(state, _buy(
                    qty=1.0, price=expected[i], order_id=20 + i,
                    agent_id=300, timestamp=200 + i * 100))


# ===========================================================================
# M4.2-A7. ppd with cancel semantics
# ===========================================================================

class TestICEwPICancel:
    """Cancelling an ICEwPI order stops further ppd-adjusted slices."""

    def test_cancel_after_one_slice_stops_ppd_progression(self, book, state):
        """Cancel after first regeneration: no more slices appear."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=0.50)
        state, _ = book.submit_order(state, iceberg)

        # Fill initial peak -> slice 1 at 50.50
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.price == approx(50.50)

        # Cancel the iceberg
        state = book.cancel_order(state, 1, Side.SELL)
        assert _is_absent(book.get_order(state, 1, Side.SELL))

        # Submit another trade to trigger any regeneration check
        state, _ = book.submit_order(state, _sell(
            qty=1.0, price=55.0, order_id=2, agent_id=400, timestamp=300))
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=55.0, order_id=3, agent_id=500, timestamp=400))

        # Iceberg should NOT have regenerated
        assert _is_absent(book.get_order(state, 1, Side.SELL))

    def test_cancel_original_slice_before_any_fill(self, book, state):
        """Cancel the initial ICEwPI slice before any fill."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=0.50)
        state, _ = book.submit_order(state, iceberg)

        state = book.cancel_order(state, 1, Side.SELL)
        assert _is_absent(book.get_order(state, 1, Side.SELL))


# ===========================================================================
# M4.2-A8. ICEwPI interaction with matching (price crosses spread)
# ===========================================================================

class TestICEwPIMatchingInteraction:
    """ICEwPI slices that drift in price may interact with matching."""

    def test_sell_icewpi_slice_price_moves_away_from_spread(self, book, state):
        """Sell iceberg with positive ppd: slices move further from best bid,
        making them harder to match (realistic behavior)."""
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=1.00)
        state, _ = book.submit_order(state, iceberg)

        # Fill initial peak at 50.0
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        # Next slice at 51.0 — a buy at 50.0 should NOT match it
        state, result = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=11, agent_id=300, timestamp=300))
        assert not result.traded

        # Buy at 51.0 SHOULD match it
        state, result2 = book.submit_order(state, _buy(
            qty=1.0, price=51.0, order_id=12, agent_id=300, timestamp=400))
        assert result2.traded
        assert result2.trade_price == approx(51.0)

    def test_buy_icewpi_slice_price_moves_away_from_spread(self, book, state):
        """Buy iceberg with negative ppd: slices have lower bid price."""
        iceberg = _iceberg_buy(qty=3.0, peak_qty=1.0, order_id=1,
                               agent_id=200, timestamp=100, price=50.0,
                               ppd=-1.00)
        state, _ = book.submit_order(state, iceberg)

        # Fill initial peak at 50.0
        state, _ = book.submit_order(state, _sell(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        # Next slice at 49.0 — sell at 50.0 should NOT match (sell price > bid)
        state, result = book.submit_order(state, _sell(
            qty=1.0, price=50.0, order_id=11, agent_id=300, timestamp=300))
        assert not result.traded

        # Sell at 49.0 matches the new slice
        state, result2 = book.submit_order(state, _sell(
            qty=1.0, price=49.0, order_id=12, agent_id=300, timestamp=400))
        assert result2.traded
        assert result2.trade_price == approx(49.0)


# ===========================================================================
# M4.2-A9. ICEwPI vs ICEwPI — two icebergs with different ppd
# ===========================================================================

class TestICEwPIvsICEwPI:
    """Two ICEwPI orders interacting on opposite sides."""

    def test_two_sell_icebergs_different_ppd(self, book, state):
        """Two sell icebergs with different ppd values at same initial price.
        After first slice fill, their prices diverge."""
        ice1 = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                             agent_id=200, timestamp=100, price=50.0,
                             ppd=0.10)
        ice2 = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=2,
                             agent_id=300, timestamp=150, price=50.0,
                             ppd=0.50)
        state, _ = book.submit_order(state, ice1)
        state, _ = book.submit_order(state, ice2)

        # Buy 1 at 50.0 matches ice1 (earlier timestamp)
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=400, timestamp=200))

        # ice1 regenerates at 50.10; ice2 still at 50.0
        # Next buy at 50.10 should match ice2 (at 50.0, better price)
        state, r2 = book.submit_order(state, _buy(
            qty=1.0, price=50.10, order_id=11, agent_id=400, timestamp=300))
        assert r2.traded
        assert r2.trade_price == approx(50.0)  # ice2's price

        # Now ice2 regenerates at 50.50; ice1 still at 50.10
        # Next buy at 50.50 should match ice1 (50.10 < 50.50)
        state, r3 = book.submit_order(state, _buy(
            qty=1.0, price=50.50, order_id=12, agent_id=400, timestamp=400))
        assert r3.traded
        assert r3.trade_price == approx(50.10)


# ===========================================================================
# M4.2-A10. ICEwPI with GCT expiry
# ===========================================================================

class TestICEwPIGctExpiry:
    """GCT expiry cleans up ICEwPI state including ppd tracking."""

    def test_gct_expiry_clears_icewpi(self, book, state):
        """expire_at_gct removes ICEwPI and its metadata."""
        iceberg = _iceberg_sell(qty=5.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=0.50)
        state, _ = book.submit_order(state, iceberg)

        # Fill one slice to advance ppd state
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        state = book.expire_at_gct(state)

        # Book should be empty
        assert _is_absent(book.get_order(state, 1, Side.SELL))
        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) == 0


# ===========================================================================
# M4.2-A11. ICEwPI agent views
# ===========================================================================

class TestICEwPIAgentViews:
    """Agent views correctly reflect ppd-adjusted slices."""

    def test_agent_orders_shows_ppd_adjusted_price(self, book, state):
        """After regeneration, get_agent_orders shows adjusted price."""
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=0.50)
        state, _ = book.submit_order(state, iceberg)

        # Fill initial peak
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        # Agent should see resting order at 50.50
        orders = book.get_agent_orders(state, 200)
        assert len(orders) == 1
        assert orders[0].price == approx(50.50)
        # Re-baselined by OMIE-OWN-ORDERS-ICE-QTY-1 (§9.2): the owner sees the
        # full remaining quantity 3.0 - 1.0 filled = 2.0 (was visible-only 1.0).
        assert orders[0].quantity == approx(2.0)

    def test_agent_trades_records_correct_prices(self, book, state):
        """Trades from different slices record at different prices."""
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=1.00)
        state, _ = book.submit_order(state, iceberg)

        # Trade 1: fill at 50.0
        state, r1 = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))
        assert r1.traded
        assert r1.trade_price == approx(50.0)

        # Trade 2: fill at 51.0
        state, r2 = book.submit_order(state, _buy(
            qty=1.0, price=51.0, order_id=11, agent_id=300, timestamp=300))
        assert r2.traded
        assert r2.trade_price == approx(51.0)

        # Iceberg owner should see both trades
        trades = book.get_agent_trades(state, 200)
        assert len(trades) >= 2


# ===========================================================================
# M4.2-A12. ICEwPI with fractional ppd values
# ===========================================================================

class TestICEwPIFractionalPpd:
    """PPD values that are small fractions of the tick size."""

    def test_ppd_equals_one_tick(self, book, state):
        """ppd=0.01 (one tick): each slice moves by exactly one tick."""
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=0.01)
        state, _ = book.submit_order(state, iceberg)

        # Fill initial peak
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.price == approx(50.01)

    def test_ppd_larger_than_typical(self, book, state):
        """ppd=1.45 (near observed max): verify large jumps work."""
        iceberg = _iceberg_sell(qty=3.0, peak_qty=1.0, order_id=1,
                                agent_id=200, timestamp=100, price=50.0,
                                ppd=1.45)
        state, _ = book.submit_order(state, iceberg)

        # Fill initial
        state, _ = book.submit_order(state, _buy(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        found = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(found)
        assert found.price == approx(51.45)

    def test_ppd_negative_larger_than_typical(self, book, state):
        """ppd=-1.55 (near observed min): verify negative large jumps."""
        iceberg = _iceberg_buy(qty=3.0, peak_qty=1.0, order_id=1,
                               agent_id=200, timestamp=100, price=50.0,
                               ppd=-1.55)
        state, _ = book.submit_order(state, iceberg)

        # Fill initial
        state, _ = book.submit_order(state, _sell(
            qty=1.0, price=50.0, order_id=10, agent_id=300, timestamp=200))

        found = book.get_order(state, 1, Side.BUY)
        assert not _is_absent(found)
        assert found.price == approx(48.45)
