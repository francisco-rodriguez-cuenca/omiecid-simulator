"""Audit tests for M6.1 — End-to-End Scripted Scenarios.

Post-implementation verification focusing on edge cases, boundary conditions,
and acceptance criteria gaps for the full Phase 1 simulator integration.

Derived independently from:
- docs/omiecid/market-spec.md §2.3 (price/qty constraints), §3.1 (GCT), §4 (order types)
- docs/omiecid/migration-plan.md §M6 (scenario scripts with expected values)
- docs/omiecid/agile-plan.md (M6.1 acceptance criteria)
- omiecid_simulator/omiecid/ public interfaces only

Audit areas:
1. Full scenario PnL zero-sum invariant across participants
2. FOK boundary: exact liquidity match vs one-tick-short rejection
3. IOC with zero matchable liquidity (nothing to fill)
4. Iceberg with price increment (ICEwPI) in end-to-end flow
5. Multi-price-level walk: settlement value consistency with VWAP
6. GCT expiry preserves trade history (trades survive, orders don't)
7. Interleaved agent orders: time priority across agents
8. Negative-price scenario end-to-end
9. Minimum tick quantity and price in full flow
10. FOK + IOC combined scenario: FOK rejected, then IOC partial in same book
"""

import jax.numpy as jnp
import pytest
from pytest import approx

from omiecid_simulator.omiecid.contract_book import SubmitResult
from omiecid_simulator.omiecid.encoding import encode_price
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    make_order,
)
from omiecid_simulator.omiecid.settlement import compute_trade_value, compute_agent_pnl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = OMIEProductConfig()
_QTY_TICK = _CFG.qty_tick  # 0.1 MW

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




def _make_order(side, price, quantity, order_id, agent_id, timestamp,
                exec_restriction=ExecRestriction.NON, peak_qty=0.0, ppd=0.0):
    """Convenience wrapper with contract_id=37 (matching migration-plan §M6)."""
    return make_order(
        contract_id=37, side=side, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction, peak_qty=peak_qty, ppd=ppd,
    )


def _agent_tape(raw_trades, *, agent_id, is_seller, counterparty=-9):
    """Build a synthetic (M, 6) int32 tape for P2-L2's tape-centric API.

    Post-P2-L2, ``compute_agent_pnl`` reads an ``(M, 6) int32`` tape with
    an offset-encoded ``price`` column (migration-plan §Decoding contract,
    v5.4). ``get_agent_trades`` drops buyer/seller columns, but each test
    here knows ``agent_id``'s role by construction, so we rebuild the tape
    placing ``agent_id`` on the known leg and a sentinel on the other.
    """
    rows = []
    for price_eur, qty_mw, *_rest in raw_trades:
        p_col = encode_price(float(price_eur), _CFG)
        q_col = round(float(qty_mw) / _QTY_TICK)
        if is_seller:
            rows.append([0, p_col, q_col, counterparty, agent_id, 0])
        else:
            rows.append([0, p_col, q_col, agent_id, counterparty, 0])
    if not rows:
        rows.append([0, 0, 0, -1, -1, 0])
    return jnp.asarray(rows, dtype=jnp.int32)


def _pnl_one_sided(raw_trades, *, agent_id, is_seller):
    """Back-compat shim: compute agent PnL via the new tape-centric API."""
    tape = _agent_tape(raw_trades, agent_id=agent_id, is_seller=is_seller)
    return float(compute_agent_pnl(tape, jnp.int32(agent_id)))


# ---------------------------------------------------------------------------
# Shared fixtures (product_config, market_config, open_contract, book from conftest)
# ---------------------------------------------------------------------------

@pytest.fixture
def state(book):
    """Alias: tests in this file use 'state' rather than 'empty_state'."""
    return book.init()


# ===========================================================================
# 1. PnL zero-sum invariant: all trades across two agents must net to zero
# ===========================================================================

class TestPnlZeroSumInvariant:
    """Domain rule: in a closed system with 2 participants, total PnL = 0.

    This verifies the invariant across a multi-trade scenario where
    both sides trade at different prices.
    """

    def test_pnl_sums_to_zero_across_agents(self, book, state):
        """Multiple trades at different prices; A sells, B buys. Sum PnL = 0."""
        AGENT_A, AGENT_B = 1, 2

        # A posts two sell orders at different prices
        sell1 = _make_order(Side.SELL, 45.00, 5.0, 1, AGENT_A, 100)
        state, r1 = book.submit_order(state, sell1)
        assert r1.accepted

        sell2 = _make_order(Side.SELL, 47.50, 3.0, 2, AGENT_A, 200)
        state, r2 = book.submit_order(state, sell2)
        assert r2.accepted

        # B sweeps both levels with a single buy
        buy = _make_order(Side.BUY, 48.00, 8.0, 3, AGENT_B, 300)
        state, r3 = book.submit_order(state, buy)
        assert r3.traded
        assert r3.trade_qty == approx(8.0)

        # Compute PnL for each agent
        a_trades = book.get_agent_trades(state, AGENT_A)
        b_trades = book.get_agent_trades(state, AGENT_B)

        a_pnl = _pnl_one_sided(a_trades, agent_id=AGENT_A, is_seller=True)
        b_pnl = _pnl_one_sided(b_trades, agent_id=AGENT_B, is_seller=False)

        # Zero-sum: A's profit + B's cost = 0
        assert a_pnl + b_pnl == approx(0.0)
        # Also verify individual values are non-trivial
        assert a_pnl > 0  # A sold
        assert b_pnl < 0  # B bought

    def test_pnl_matches_hand_computed_values(self, book, state):
        """Verify PnL matches hand-computed values from the scenario script.

        Reproduces migration-plan §M6 scenario with exact expected values.
        A: sold 5 MW@48.50 + 3 MW@49.00 = 60.625 + 36.75 = 97.375 EUR
        B: bought same = -97.375 EUR
        """
        AGENT_A, AGENT_B = 1, 2

        # Step 2: A sells 10 MW @ 50.00
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 10.0, 1, AGENT_A, 100))
        # Step 3: A sells 5 MW @ 48.50
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 48.50, 5.0, 2, AGENT_A, 200))
        # Step 4: B buys 8 MW @ 49.00 -> matches 5 MW@48.50
        state, r4 = book.submit_order(state, _make_order(
            Side.BUY, 49.00, 8.0, 3, AGENT_B, 300))
        assert r4.traded
        assert r4.trade_qty == approx(5.0)
        assert r4.trade_price == approx(48.50)

        # Step 5: A sells 3 MW @ 48.00 IOC -> matches 3 MW@49.00
        state, r5 = book.submit_order(state, _make_order(
            Side.SELL, 48.00, 3.0, 4, AGENT_A, 400,
            exec_restriction=ExecRestriction.IOC))
        assert r5.traded
        assert r5.trade_qty == approx(3.0)
        assert r5.trade_price == approx(49.00)  # resting price

        a_trades = book.get_agent_trades(state, AGENT_A)
        b_trades = book.get_agent_trades(state, AGENT_B)

        a_pnl = _pnl_one_sided(a_trades, agent_id=AGENT_A, is_seller=True)
        b_pnl = _pnl_one_sided(b_trades, agent_id=AGENT_B, is_seller=False)

        assert a_pnl == approx(97.375)
        assert b_pnl == approx(-97.375)


# ===========================================================================
# 2. FOK boundary: exact liquidity vs one-tick-short
# ===========================================================================

class TestFokBoundaryConditions:
    """FOK must reject when liquidity is even slightly insufficient."""

    def test_fok_rejected_one_tick_short(self, book, state):
        """FOK BUY for 5.1 MW rejected when only 5.0 MW available."""
        # Resting sell
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 5.0, 1, 1, 100))

        # FOK for 5.1 MW (one qty tick more than available)
        fok = _make_order(Side.BUY, 50.00, 5.1, 2, 2, 200,
                          exec_restriction=ExecRestriction.FOK)
        state, result = book.submit_order(state, fok)

        # Passed validation (accepted=True) but FOK check failed
        assert result.accepted == True
        assert result.traded == False

        # Resting order should still be in book
        remaining = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(remaining)
        assert remaining.quantity == approx(5.0)

    def test_fok_accepted_exact_match(self, book, state):
        """FOK BUY for exactly 5.0 MW fills when exactly 5.0 MW available."""
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 5.0, 1, 1, 100))

        fok = _make_order(Side.BUY, 50.00, 5.0, 2, 2, 200,
                          exec_restriction=ExecRestriction.FOK)
        state, result = book.submit_order(state, fok)

        assert result.accepted == True
        assert result.traded == True
        assert result.trade_qty == approx(5.0)
        assert result.trade_price == approx(50.00)

        # Book should be empty
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0

    def test_fok_across_multiple_levels(self, book, state):
        """FOK BUY 5 MW fills across two price levels (2+3 MW)."""
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 48.00, 2.0, 1, 1, 100))
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 49.00, 3.0, 2, 1, 200))

        fok = _make_order(Side.BUY, 49.00, 5.0, 3, 2, 300,
                          exec_restriction=ExecRestriction.FOK)
        state, result = book.submit_order(state, fok)

        assert result.accepted == True
        assert result.traded == True
        assert result.trade_qty == approx(5.0)

        # Book should be empty (both levels consumed)
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0


# ===========================================================================
# 3. IOC with zero available liquidity
# ===========================================================================

class TestIocZeroLiquidity:
    """IOC into an empty book: nothing fills, order does not rest."""

    def test_ioc_no_match_empty_book(self, book, state):
        """IOC BUY into empty book: accepted but not traded, no resting order."""
        ioc = _make_order(Side.BUY, 50.00, 5.0, 1, 1, 100,
                          exec_restriction=ExecRestriction.IOC)
        state, result = book.submit_order(state, ioc)

        assert result.accepted == True
        assert result.traded == False
        assert result.trade_qty == approx(0.0)

        # IOC must NOT rest in book
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0

    def test_ioc_no_match_wrong_side(self, book, state):
        """IOC BUY when only BUY orders in book: nothing crosses."""
        # Place a resting buy (same side as IOC)
        state, _ = book.submit_order(state, _make_order(
            Side.BUY, 50.00, 5.0, 1, 1, 100))

        # IOC buy cannot match other buys
        ioc = _make_order(Side.BUY, 50.00, 5.0, 2, 2, 200,
                          exec_restriction=ExecRestriction.IOC)
        state, result = book.submit_order(state, ioc)

        assert result.traded == False

        # Original resting buy should still be there, IOC should not
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 1  # original resting buy
        assert len(asks) == 0

    def test_ioc_partial_fill_settlement_value(self, book, state):
        """IOC partial fill: settlement value matches filled portion only.

        SELL 3 MW@50 resting. IOC BUY 8 MW@50 -> 3 MW filled.
        Settlement = 3 * 50 * 0.25 = 37.50 EUR.
        """
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 3.0, 1, 1, 100))

        ioc = _make_order(Side.BUY, 50.00, 8.0, 2, 2, 200,
                          exec_restriction=ExecRestriction.IOC)
        state, result = book.submit_order(state, ioc)

        assert result.traded == True
        assert result.trade_qty == approx(3.0)
        assert result.trade_price == approx(50.00)
        assert result.settlement_value == approx(37.50)

        # Book should be empty: sell consumed, IOC remainder cancelled
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0


# ===========================================================================
# 4. Iceberg with price increment (ICEwPI) end-to-end
# ===========================================================================

class TestIcebergWithPriceIncrementE2E:
    """ICEwPI sell: each new slice price increments by ppd.

    SELL 9 MW, peak=3 MW, ppd=+0.50, starting at 50.00.
    Slice 0: 3 MW @ 50.00
    Slice 1: 3 MW @ 50.50
    Slice 2: 3 MW @ 51.00
    """

    def test_icewpi_sell_three_slices(self, book, state):
        """Verify price increments across 3 iceberg slices."""
        SELLER, BUYER = 1, 2

        # Submit ICEwPI SELL
        ice = _make_order(Side.SELL, 50.00, 9.0, 1, SELLER, 100,
                          exec_restriction=ExecRestriction.ICE,
                          peak_qty=3.0, ppd=0.50)
        state, r1 = book.submit_order(state, ice)
        assert r1.accepted

        # Verify initial LOB: 3 MW @ 50.00 visible
        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        assert asks[0] == approx((50.00, 3.0))

        # Fill slice 0: BUY 3 MW @ 50.00
        buy1 = _make_order(Side.BUY, 50.00, 3.0, 2, BUYER, 200)
        state, r2 = book.submit_order(state, buy1)
        assert r2.traded
        assert r2.trade_qty == approx(3.0)
        assert r2.trade_price == approx(50.00)

        # After slice 0 exhausted, new slice at 50.50
        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        assert asks[0][0] == approx(50.50)  # price incremented
        assert asks[0][1] == approx(3.0)

        # Fill slice 1: BUY 3 MW @ 50.50
        buy2 = _make_order(Side.BUY, 50.50, 3.0, 3, BUYER, 300)
        state, r3 = book.submit_order(state, buy2)
        assert r3.traded
        assert r3.trade_qty == approx(3.0)
        assert r3.trade_price == approx(50.50)

        # After slice 1, new slice at 51.00
        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        assert asks[0][0] == approx(51.00)
        assert asks[0][1] == approx(3.0)

        # Fill slice 2: BUY 3 MW @ 51.00
        buy3 = _make_order(Side.BUY, 51.00, 3.0, 4, BUYER, 400)
        state, r4 = book.submit_order(state, buy3)
        assert r4.traded

        # Book should be empty (9 MW exhausted)
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0

    def test_icewpi_buy_negative_delta(self, book, state):
        """ICEwPI BUY: ppd is negative (price decreases per slice).

        BUY 6 MW, peak=3 MW, ppd=-0.50, starting at 50.00.
        Slice 0: 3 MW @ 50.00
        Slice 1: 3 MW @ 49.50
        """
        SELLER, BUYER = 1, 2

        # Submit ICEwPI BUY
        ice = _make_order(Side.BUY, 50.00, 6.0, 1, BUYER, 100,
                          exec_restriction=ExecRestriction.ICE,
                          peak_qty=3.0, ppd=-0.50)
        state, _ = book.submit_order(state, ice)

        # Verify initial LOB: 3 MW @ 50.00 on bid side
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 1
        assert bids[0] == approx((50.00, 3.0))

        # Fill slice 0: SELL 3 MW @ 50.00
        sell1 = _make_order(Side.SELL, 50.00, 3.0, 2, SELLER, 200)
        state, r2 = book.submit_order(state, sell1)
        assert r2.traded

        # After slice 0 exhausted, new slice at 49.50
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 1
        assert bids[0][0] == approx(49.50)
        assert bids[0][1] == approx(3.0)

        # Fill slice 1: SELL 3 MW @ 49.50
        sell2 = _make_order(Side.SELL, 49.50, 3.0, 3, SELLER, 300)
        state, r3 = book.submit_order(state, sell2)
        assert r3.traded

        # Book empty
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0


# ===========================================================================
# 5. Multi-price-level walk: VWAP and settlement consistency
# ===========================================================================

class TestMultiPriceLevelWalkDetails:
    """Verify VWAP calculation and settlement value when walking price levels."""

    def test_settlement_equals_vwap_times_qty_times_duration(self, book, state):
        """Settlement V = trade_qty * trade_price(VWAP) * 0.25.

        3 levels: SELL 2@48, 3@49, 5@50. BUY 7@50 walks all 3.
        VWAP = (2*48 + 3*49 + 2*50) / 7 = 343/7 = 49.0
        V = 7 * 49 * 0.25 = 85.75
        """
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 48.00, 2.0, 1, 1, 100))
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 49.00, 3.0, 2, 1, 200))
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 5.0, 3, 1, 300))

        buy = _make_order(Side.BUY, 50.00, 7.0, 4, 2, 400)
        state, result = book.submit_order(state, buy)

        assert result.traded == True
        assert result.trade_qty == approx(7.0)
        assert result.trade_price == approx(49.0, abs=0.01)
        assert result.settlement_value == approx(85.75, abs=0.01)

    def test_residual_after_walk(self, book, state):
        """After walk, residual sell quantity stays in book at correct level."""
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 48.00, 2.0, 1, 1, 100))
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 49.00, 3.0, 2, 1, 200))
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 5.0, 3, 1, 300))

        buy = _make_order(Side.BUY, 50.00, 7.0, 4, 2, 400)
        state, _ = book.submit_order(state, buy)

        # Only 3 MW residual @ 50.00 should remain
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 1
        assert asks[0] == approx((50.00, 3.0))


# ===========================================================================
# 6. GCT expiry preserves trade history
# ===========================================================================

class TestGctExpiryPreservesTrades:
    """After expire_at_gct, orders are gone but trade history remains."""

    def test_trades_survive_expiry(self, book, state):
        """Trade records persist after GCT expiry clears the book."""
        AGENT_A, AGENT_B = 1, 2

        # Create and fill a trade
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 5.0, 1, AGENT_A, 100))
        state, result = book.submit_order(state, _make_order(
            Side.BUY, 50.00, 5.0, 2, AGENT_B, 200))
        assert result.traded

        # Record trades before expiry
        a_trades_before = book.get_agent_trades(state, AGENT_A)
        b_trades_before = book.get_agent_trades(state, AGENT_B)
        assert len(a_trades_before) > 0
        assert len(b_trades_before) > 0

        # Expire at GCT
        state = book.expire_at_gct(state)

        # Book is empty
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0

        # Trades are preserved
        a_trades_after = book.get_agent_trades(state, AGENT_A)
        b_trades_after = book.get_agent_trades(state, AGENT_B)
        assert len(a_trades_after) == len(a_trades_before)
        assert len(b_trades_after) == len(b_trades_before)

    def test_unfilled_orders_cancelled_at_gct(self, book, state):
        """Unfilled resting orders are removed at GCT."""
        # Place orders that don't cross
        state, _ = book.submit_order(state, _make_order(
            Side.BUY, 45.00, 5.0, 1, 1, 100))
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 55.00, 3.0, 2, 2, 200))

        # Both should be in book
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 1
        assert len(asks) == 1

        # Expire
        state = book.expire_at_gct(state)

        # All gone
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0


# ===========================================================================
# 7. Interleaved agent orders: time priority across agents
# ===========================================================================

class TestInterleavedAgentTimePriority:
    """When two agents place orders at the same price, time priority governs."""

    def test_earlier_order_fills_first(self, book, state):
        """Agent A posts SELL@50 at ts=100, B posts SELL@50 at ts=200.
        Aggressive BUY@50 for 3 MW fills against A (earlier timestamp)."""
        AGENT_A, AGENT_B, AGENT_C = 1, 2, 3

        # A posts sell first
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 5.0, 1, AGENT_A, 100))
        # B posts sell at same price, later
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 5.0, 2, AGENT_B, 200))

        # C buys 3 MW
        state, result = book.submit_order(state, _make_order(
            Side.BUY, 50.00, 3.0, 3, AGENT_C, 300))
        assert result.traded
        assert result.trade_qty == approx(3.0)

        # A's order should be partially filled (2 MW left)
        a_order = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(a_order)
        assert a_order.quantity == approx(2.0)

        # B's order should be untouched (5 MW)
        b_order = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(b_order)
        assert b_order.quantity == approx(5.0)

    def test_three_agents_sequential_trades(self, book, state):
        """Three agents interleave orders, verifying correct trade attribution."""
        A, B, C = 1, 2, 3

        # A: SELL 5 MW @ 50
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 5.0, 1, A, 100))
        # B: BUY 2 MW @ 50 (matches A)
        state, r1 = book.submit_order(state, _make_order(
            Side.BUY, 50.00, 2.0, 2, B, 200))
        assert r1.traded
        assert r1.trade_qty == approx(2.0)

        # C: BUY 3 MW @ 50 (matches remaining from A)
        state, r2 = book.submit_order(state, _make_order(
            Side.BUY, 50.00, 3.0, 3, C, 300))
        assert r2.traded
        assert r2.trade_qty == approx(3.0)

        # A's order fully consumed
        a_order = book.get_order(state, 1, Side.SELL)
        assert _is_absent(a_order)

        # Verify A has 2 trades
        a_trades = book.get_agent_trades(state, A)
        assert len(a_trades) == 2


# ===========================================================================
# 8. Negative-price end-to-end scenario
# ===========================================================================

class TestNegativePriceScenario:
    """OMIE allows negative prices (min -9999 EUR/MWh). Market-spec §2.3."""

    def test_negative_price_trade_and_settlement(self, book, state):
        """Trade at negative price: settlement value is negative."""
        AGENT_A, AGENT_B = 1, 2

        # A sells at -10 EUR/MWh
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, -10.00, 5.0, 1, AGENT_A, 100))
        # B buys at -10 EUR/MWh
        state, result = book.submit_order(state, _make_order(
            Side.BUY, -10.00, 5.0, 2, AGENT_B, 200))

        assert result.traded
        assert result.trade_qty == approx(5.0)
        assert result.trade_price == approx(-10.00)
        # V = 5 * (-10) * 0.25 = -12.50
        assert result.settlement_value == approx(-12.50)

        # Seller's PnL from selling at negative price: loss
        a_trades = book.get_agent_trades(state, AGENT_A)
        a_pnl = _pnl_one_sided(a_trades, agent_id=AGENT_A, is_seller=True)
        # Seller sold at -10: V = -12.50 -> PnL = -12.50 (loss)
        assert a_pnl == approx(-12.50)

        # Buyer's PnL from buying at negative price: gain
        b_trades = book.get_agent_trades(state, AGENT_B)
        b_pnl = _pnl_one_sided(b_trades, agent_id=AGENT_B, is_seller=False)
        # Buyer bought at -10: cost = -12.50 -> PnL = -(-12.50) = 12.50 (gain)
        assert b_pnl == approx(12.50)


# ===========================================================================
# 9. Minimum tick quantity and price in full flow
# ===========================================================================

class TestMinimumTickValues:
    """End-to-end with minimum allowed price tick and quantity tick."""

    def test_min_quantity_trade(self, book, state):
        """Trade 0.1 MW (minimum quantity tick) at a normal price."""
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 0.1, 1, 1, 100))
        state, result = book.submit_order(state, _make_order(
            Side.BUY, 50.00, 0.1, 2, 2, 200))

        assert result.traded
        assert result.trade_qty == approx(0.1)
        # V = 0.1 * 50 * 0.25 = 1.25
        assert result.settlement_value == approx(1.25)

    def test_min_price_tick_trade(self, book, state):
        """Trade at 0.01 EUR/MWh (minimum positive price tick)."""
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 0.01, 1.0, 1, 1, 100))
        state, result = book.submit_order(state, _make_order(
            Side.BUY, 0.01, 1.0, 2, 2, 200))

        assert result.traded
        assert result.trade_qty == approx(1.0)
        assert result.trade_price == approx(0.01)
        # V = 1.0 * 0.01 * 0.25 = 0.0025
        assert result.settlement_value == approx(0.0025)


# ===========================================================================
# 10. Combined restriction scenario: FOK rejected, then IOC partial
# ===========================================================================

class TestCombinedRestrictionScenario:
    """FOK fails, then IOC partially fills in same book state."""

    def test_fok_reject_then_ioc_partial(self, book, state):
        """Resting SELL 3 MW@50. FOK BUY 5 MW rejected.
        IOC BUY 5 MW partially fills 3 MW, remainder cancelled."""
        SELLER, BUYER = 1, 2

        # Resting sell
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 3.0, 1, SELLER, 100))

        # FOK for 5 MW (needs 5, only 3 available) -> rejected
        fok = _make_order(Side.BUY, 50.00, 5.0, 2, BUYER, 200,
                          exec_restriction=ExecRestriction.FOK)
        state, fok_result = book.submit_order(state, fok)
        assert fok_result.accepted == True  # passed validation
        assert fok_result.traded == False    # FOK liquidity check failed

        # Resting sell still there
        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1

        # IOC for 5 MW -> partial fill 3 MW, remainder cancelled
        ioc = _make_order(Side.BUY, 50.00, 5.0, 3, BUYER, 300,
                          exec_restriction=ExecRestriction.IOC)
        state, ioc_result = book.submit_order(state, ioc)
        assert ioc_result.accepted == True
        assert ioc_result.traded == True
        assert ioc_result.trade_qty == approx(3.0)
        assert ioc_result.settlement_value == approx(3.0 * 50.0 * 0.25)

        # Book should be empty
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0


# ===========================================================================
# 11. Iceberg standard end-to-end: 4 successive fills
# ===========================================================================

class TestIcebergStandardE2E:
    """Standard iceberg SELL 10 MW, peak=3 MW, across 4 fills (3+3+3+1)."""

    def test_iceberg_four_fills_total_exhaustion(self, book, state):
        """Iceberg SELL 10 MW peak=3: fills 3, 3, 3, 1 MW then empty."""
        SELLER, BUYER = 1, 2

        ice = _make_order(Side.SELL, 50.00, 10.0, 1, SELLER, 100,
                          exec_restriction=ExecRestriction.ICE,
                          peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        # Verify initial visibility
        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        assert asks[0][1] == approx(3.0)

        total_filled = 0.0
        expected_fills = [3.0, 3.0, 3.0, 1.0]
        for i, expected_qty in enumerate(expected_fills):
            buy = _make_order(Side.BUY, 50.00, expected_qty, 10 + i, BUYER,
                              200 + i * 100)
            state, result = book.submit_order(state, buy)
            assert result.traded, f"Fill {i} should trade"
            assert result.trade_qty == approx(expected_qty), \
                f"Fill {i}: expected {expected_qty}, got {result.trade_qty}"
            total_filled += result.trade_qty

        assert total_filled == approx(10.0)

        # Book should be empty after total exhaustion
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        assert len(asks) == 0

    def test_iceberg_settlement_total(self, book, state):
        """Total settlement value across all iceberg fills equals expected."""
        SELLER, BUYER = 1, 2

        ice = _make_order(Side.SELL, 50.00, 10.0, 1, SELLER, 100,
                          exec_restriction=ExecRestriction.ICE,
                          peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        total_settlement = 0.0
        fills = [3.0, 3.0, 3.0, 1.0]
        for i, qty in enumerate(fills):
            buy = _make_order(Side.BUY, 50.00, qty, 10 + i, BUYER,
                              200 + i * 100)
            state, result = book.submit_order(state, buy)
            total_settlement += result.settlement_value

        # Total: 10 MW * 50 EUR/MWh * 0.25 h = 125.00 EUR
        assert total_settlement == approx(125.00)


# ===========================================================================
# 12. Resting order survives non-crossing aggressor
# ===========================================================================

class TestNonCrossingOrders:
    """Orders that don't cross the spread should both rest in the book."""

    def test_bid_below_ask_both_rest(self, book, state):
        """BUY@45 and SELL@50 do not cross; both rest."""
        state, r1 = book.submit_order(state, _make_order(
            Side.SELL, 50.00, 5.0, 1, 1, 100))
        assert r1.accepted
        assert r1.traded == False

        state, r2 = book.submit_order(state, _make_order(
            Side.BUY, 45.00, 3.0, 2, 2, 200))
        assert r2.accepted
        assert r2.traded == False

        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 1
        assert len(asks) == 1
        assert bids[0] == approx((45.00, 3.0))
        assert asks[0] == approx((50.00, 5.0))

    def test_trade_executes_at_resting_price(self, book, state):
        """Domain rule: trade price is the resting order's price, not aggressor's.

        SELL 5 MW @ 48.00 resting. BUY 5 MW @ 52.00 aggressive.
        Trade at 48.00 (resting price), NOT 52.00.
        """
        state, _ = book.submit_order(state, _make_order(
            Side.SELL, 48.00, 5.0, 1, 1, 100))
        state, result = book.submit_order(state, _make_order(
            Side.BUY, 52.00, 5.0, 2, 2, 200))

        assert result.traded
        assert result.trade_price == approx(48.00)
        # Settlement at resting price
        assert result.settlement_value == approx(5.0 * 48.0 * 0.25)
