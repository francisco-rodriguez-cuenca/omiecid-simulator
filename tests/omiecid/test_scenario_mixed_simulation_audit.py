"""Audit tests for M6.2 — Mixed Order-Type Simulation (updated M6.4).

Post-implementation verification focusing on edge cases, boundary conditions,
and combinatorial interactions between order types in the same book.

Derived independently from:
- docs/omiecid/market-spec.md §4 (order types), §4.5 (exec restrictions), §6 (matching)
- docs/omiecid/agile-plan.md (M6.2 acceptance criteria)
- omiecid_simulator/omiecid/ public interfaces only

Critical domain rules tested (M6.4 update):
1. FOK pre-check sees total iceberg quantity (visible peak + hidden remainder)
   at crossing prices (market-spec §4.2.1 Rule 1, updated M6.3).
2. All aggressive order types (FOK, NON, IOC) use multi-slice matching:
   fills across iceberg slice regenerations within a single submit_order call
   (market-spec §4.2.1 Rule 2, updated M6.4).
3. IOC cancels unfilled remainder after the multi-slice loop ends.

Audit areas:
A. FOK acceptance against iceberg using total-quantity visibility (M6.3)
B. Multi-slice iceberg consumption (now within single submit_order, M6.4)
C. ICEwPI price drift across slices in mixed book
D. IOC multi-slice fill against iceberg (M6.4 update)
E. Cancel mid-iceberg: hidden qty evaporates, no phantom orders
F. 3-4 agent zero-sum PnL across mixed order scenario
G. LOB snapshot correctness: only visible iceberg peak shown
H. NON order walking through mixed limit + iceberg levels
I. FOK exact-match at iceberg peak boundary
J. Cancel + re-submit mid-sequence preserves book integrity
"""

import jax.numpy as jnp
import pytest
from pytest import approx

from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.encoding import encode_price
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig
from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    make_order,
)
from omiecid_simulator.omiecid.settlement import compute_trade_value, compute_agent_pnl


_PCFG = OMIEProductConfig()
_QTY_TICK = _PCFG.qty_tick  # 0.1 MW

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




def _tape_from_trades(trades):
    """Build an (M, 6) int32 tape for P2-L2's tape-centric API.

    ``trades`` is a list of ``(price_eur, qty_mw, buyer, seller)`` tuples.
    Post-P2-L2, ``compute_agent_pnl`` reads an ``(M, 6) int32`` tape whose
    ``price`` column is offset-encoded via ``encode_price``
    (migration-plan §Decoding contract, v5.4).
    """
    rows = []
    for price_eur, qty_mw, buyer, seller in trades:
        p_col = encode_price(float(price_eur), _PCFG)
        q_col = round(float(qty_mw) / _QTY_TICK)
        rows.append([0, p_col, q_col, buyer, seller, 0])
    if not rows:
        rows.append([0, 0, 0, -1, -1, 0])
    return jnp.asarray(rows, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTRACT_ID = 42

def _make_contract():
    return Contract(
        contract_id=_CONTRACT_ID,
        delivery_start_minutes=630,
        delivery_end_minutes=645,
        gct_minutes=570,
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
# A. FOK acceptance against iceberg (total-quantity visibility, M6.3)
# ---------------------------------------------------------------------------

class TestFOKIcebergVisibility:
    """FOK liquidity check sees total iceberg quantity (visible + hidden).

    Updated M6.3: market-spec §4.2.1 Rule 1. FOK pre-check uses total
    iceberg quantity at crossing prices, not just visible peak.
    """

    def test_fok_accepted_when_total_sufficient(self):
        """FOK BUY 3 MW against iceberg SELL with peak=2 MW, total=10 MW -> FILLS.

        M6.3: total=10 >= 3, pre-check passes. Multi-slice fill: 2 MW from
        peak1, regen peak2 (2 MW), fill 1 MW from peak2. Total: 3@50.
        """
        book = _make_book()
        state = book.init()

        # Iceberg SELL: total=10 MW, peak=2 MW at 50.00
        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, r = book.submit_order(state, ice)
        assert r.accepted

        # FOK BUY 3 MW at 50.00 — total=10 >= 3 -> fills via multi-slice
        fok = _order(Side.BUY, 50.00, 3.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r = book.submit_order(state, fok)
        assert r.traded == True
        assert r.trade_qty == approx(3.0)
        assert r.trade_price == approx(50.00)

    def test_fok_accepted_when_peak_exactly_matches(self):
        """FOK BUY 2 MW against iceberg SELL with peak=2 MW -> fills."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 2.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r = book.submit_order(state, fok)
        assert r.traded == True
        assert r.trade_qty == approx(2.0)
        assert r.trade_price == approx(50.00)

    def test_fok_accepted_fractional_within_total(self):
        """FOK BUY 2.1 MW against iceberg peak=2, total=10 -> FILLS (M6.3).

        Pre-check: total=10 >= 2.1. Multi-slice fill: 2 MW from peak1,
        regen peak2 (2 MW), fill 0.1 from peak2. Total: 2.1@50.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 2.1, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r = book.submit_order(state, fok)
        assert r.traded == True
        assert r.trade_qty == approx(2.1)
        assert r.trade_price == approx(50.00)

    def test_fok_sees_limit_plus_iceberg_total(self):
        """FOK 5 MW against limit 3 MW + iceberg total 10 MW = 13 MW matchable -> fills."""
        book = _make_book()
        state = book.init()

        # Limit SELL 3 MW at 50.00
        lim = _order(Side.SELL, 50.00, 3.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        # Iceberg SELL total=10 MW, peak=2 MW at 50.00
        ice = _order(Side.SELL, 50.00, 10.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0)
        state, _ = book.submit_order(state, ice)

        # FOK BUY 5 MW at 50.00 -> 3 (limit) + 10 (iceberg total) = 13 matchable -> fills
        fok = _order(Side.BUY, 50.00, 5.0, 3, agent_id=3, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r = book.submit_order(state, fok)
        assert r.traded == True
        assert r.trade_qty == approx(5.0)


# ---------------------------------------------------------------------------
# B. Multi-slice iceberg consumption (separate aggressive orders needed)
# ---------------------------------------------------------------------------

class TestIcebergMultiSliceConsumption:
    """M6.4: A single aggressive order fills multiple iceberg slices."""

    def test_non_order_fills_multiple_slices_in_one_call(self):
        """NON BUY 5 MW against iceberg SELL peak=3 MW, total=10 MW.
        M6.4 multi-slice: dispatch fills 3 MW (peak1). 2 MW rests. Regen
        slice2 (3@50) crosses resting 2 MW bid -> fills 2 MW. Total: 5 MW.
        NON fully filled, no bids remain."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 5.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, non)
        assert r.traded == True
        assert r.trade_qty == approx(5.0)  # M6.4: fills across slices

        # NON fully filled, no bids remain.
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 0
        # Iceberg: consumed 5 of 10. Remaining=5. Slice2 had 3, consumed 2, so 1 visible.
        assert len(asks) >= 1

    def test_single_order_consumes_10mw_iceberg_peak3(self):
        """M6.4: 10 MW iceberg with peak=3: one NON BUY 10 drains it all."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        buy = _order(Side.BUY, 50.00, 10.0, 10, agent_id=2, timestamp=10)
        state, r = book.submit_order(state, buy)
        assert r.traded == True
        assert r.trade_qty == approx(10.0)

        # Iceberg fully consumed — no asks remain
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 0

    def test_ioc_fills_multiple_slices_cancels_remainder(self):
        """M6.4: IOC BUY 5 MW against iceberg peak=3 MW -> fills 5 MW."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        ioc = _order(Side.BUY, 50.00, 5.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.IOC)
        state, r = book.submit_order(state, ioc)
        assert r.traded == True
        assert r.trade_qty == approx(5.0)  # M6.4: fills across slices

        # No resting buy order (IOC fully filled)
        bids, _ = book.get_anonymous_lob(state)
        assert len(bids) == 0


# ---------------------------------------------------------------------------
# C. ICEwPI price drift across slices
# ---------------------------------------------------------------------------

class TestICEwPIPriceDrift:
    """Iceberg with Price Increment: each new slice price shifts by ppd."""

    def test_sell_iceberg_price_increases_per_slice(self):
        """Sell ICEwPI: total=6, peak=2, ppd=+0.50. Slices at 50, 50.50, 51.00."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        # Slice 1: price=50.00, qty=2
        buy1 = _order(Side.BUY, 55.00, 2.0, 10, agent_id=2, timestamp=10)
        state, r1 = book.submit_order(state, buy1)
        assert r1.traded and r1.trade_price == approx(50.00)

        # Slice 2: price=50.50, qty=2
        buy2 = _order(Side.BUY, 55.00, 2.0, 11, agent_id=2, timestamp=11)
        state, r2 = book.submit_order(state, buy2)
        assert r2.traded and r2.trade_price == approx(50.50)

        # Slice 3: price=51.00, qty=2
        buy3 = _order(Side.BUY, 55.00, 2.0, 12, agent_id=2, timestamp=12)
        state, r3 = book.submit_order(state, buy3)
        assert r3.traded and r3.trade_price == approx(51.00)

        # Fully consumed
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 0

    def test_buy_iceberg_price_decreases_per_slice(self):
        """Buy ICEwPI: total=4, peak=2, ppd=-0.50. Slices at 50, 49.50."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 50.00, 4.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=2.0, ppd=-0.50)
        state, _ = book.submit_order(state, ice)

        # Slice 1: price=50.00
        sell1 = _order(Side.SELL, 45.00, 2.0, 10, agent_id=2, timestamp=10)
        state, r1 = book.submit_order(state, sell1)
        assert r1.traded and r1.trade_price == approx(50.00)

        # Slice 2: price=49.50
        sell2 = _order(Side.SELL, 45.00, 2.0, 11, agent_id=2, timestamp=11)
        state, r2 = book.submit_order(state, sell2)
        assert r2.traded and r2.trade_price == approx(49.50)

        # Fully consumed
        bids, _ = book.get_anonymous_lob(state)
        assert len(bids) == 0


# ---------------------------------------------------------------------------
# D. LOB snapshot: only iceberg peak visible
# ---------------------------------------------------------------------------

class TestLOBIcebergVisibility:
    """Anonymous LOB must show only the visible iceberg peak, not hidden qty."""

    def test_iceberg_shows_peak_only(self):
        """Iceberg SELL total=10, peak=3 -> LOB shows 3 MW."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        assert asks[0][1] == approx(3.0)  # visible peak only

    def test_mixed_limit_and_iceberg_same_price(self):
        """Limit 5 MW + iceberg peak 3 MW at same price -> LOB aggregates 8 MW."""
        book = _make_book()
        state = book.init()

        lim = _order(Side.SELL, 50.00, 5.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        ice = _order(Side.SELL, 50.00, 10.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        _, asks = book.get_anonymous_lob(state)
        # At 50.00: 5 + 3 = 8 MW visible
        total_at_50 = sum(q for p, q in asks if abs(p - 50.00) < 0.001)
        assert total_at_50 == approx(8.0)

    def test_iceberg_regenerated_slice_visible_after_fill(self):
        """After consuming one slice, new slice appears in LOB."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        # Consume first slice
        buy = _order(Side.BUY, 50.00, 3.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, buy)
        assert r.trade_qty == approx(3.0)

        # New slice should appear (3 MW)
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) >= 1
        assert any(abs(q - 3.0) < 0.01 for _, q in asks)


# ---------------------------------------------------------------------------
# E. Cancel mid-iceberg: hidden qty evaporates
# ---------------------------------------------------------------------------

class TestCancelIceberg:
    """Cancelling an iceberg removes both visible and hidden quantity."""

    def test_cancel_iceberg_removes_all(self):
        """Cancel an iceberg with hidden qty -> no asks, no phantom orders."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        state = book.cancel_order(state, 1, Side.SELL)

        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 0

        # Submitting a buy should not match anything
        buy = _order(Side.BUY, 50.00, 1.0, 2, agent_id=2, timestamp=2)
        state, r = book.submit_order(state, buy)
        assert r.traded == False

    def test_cancel_mid_consumption_stops_regeneration(self):
        """Consume one slice, cancel iceberg, verify no further slices."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 9.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE, peak_qty=3.0)
        state, _ = book.submit_order(state, ice)

        # Consume first slice
        buy1 = _order(Side.BUY, 50.00, 3.0, 2, agent_id=2, timestamp=2)
        state, _ = book.submit_order(state, buy1)

        # Now cancel the iceberg (new slice has been placed with oid=1)
        state = book.cancel_order(state, 1, Side.SELL)

        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 0

        # No further matching possible
        buy2 = _order(Side.BUY, 50.00, 3.0, 3, agent_id=2, timestamp=3)
        state, r = book.submit_order(state, buy2)
        assert r.traded == False


# ---------------------------------------------------------------------------
# F. Large mixed scenario: 4 agents, ~25 orders, zero-sum PnL
# ---------------------------------------------------------------------------

class TestLargeMixedScenario:
    """Full integration: mixed resting book, aggressive orders, cancel, sweep."""

    def test_mixed_book_scenario_pnl_zero_sum(self):
        """4 agents, mixed order types, verify PnL sums to zero.

        Phase 1: Build mixed book (limits + icebergs on both sides)
        Phase 2: Aggressive orders (NON walk, FOK, IOC)
        Phase 3: Cancel + re-submit
        Phase 4: Final sweep + PnL check

        Hand-computed expected values below.
        """
        book = _make_book()
        state = book.init()
        ts = 1

        # === Phase 1: Build resting book ===
        # Agent 1: SELL limit 3 MW @ 50.00
        state, r = book.submit_order(state, _order(
            Side.SELL, 50.00, 3.0, 101, agent_id=1, timestamp=ts)); ts += 1
        assert r.accepted

        # Agent 1: SELL iceberg 9 MW (peak=3) @ 51.00
        state, r = book.submit_order(state, _order(
            Side.SELL, 51.00, 9.0, 102, agent_id=1, timestamp=ts,
            exec_restriction=ExecRestriction.ICE, peak_qty=3.0)); ts += 1
        assert r.accepted

        # Agent 2: BUY limit 4 MW @ 48.00
        state, r = book.submit_order(state, _order(
            Side.BUY, 48.00, 4.0, 201, agent_id=2, timestamp=ts)); ts += 1
        assert r.accepted

        # Agent 2: BUY iceberg 6 MW (peak=2) @ 47.00
        state, r = book.submit_order(state, _order(
            Side.BUY, 47.00, 6.0, 202, agent_id=2, timestamp=ts,
            exec_restriction=ExecRestriction.ICE, peak_qty=2.0)); ts += 1
        assert r.accepted

        # Agent 3: SELL limit 2 MW @ 52.00
        state, r = book.submit_order(state, _order(
            Side.SELL, 52.00, 2.0, 301, agent_id=3, timestamp=ts)); ts += 1
        assert r.accepted

        # LOB check: asks at 50.00(3), 51.00(3 visible), 52.00(2)
        #            bids at 48.00(4), 47.00(2 visible)
        bids, asks = book.get_anonymous_lob(state)
        ask_prices = [p for p, _ in asks]
        bid_prices = [p for p, _ in bids]
        assert 50.00 in [round(p, 2) for p in ask_prices]
        assert 48.00 in [round(p, 2) for p in bid_prices]

        # === Phase 2: Aggressive orders ===

        # Agent 4: FOK BUY 3 MW @ 50.00 -> fills against Agent1 limit
        state, r = book.submit_order(state, _order(
            Side.BUY, 50.00, 3.0, 401, agent_id=4, timestamp=ts,
            exec_restriction=ExecRestriction.FOK)); ts += 1
        assert r.traded and r.trade_qty == approx(3.0) and r.trade_price == approx(50.00)
        # Trade: A4 buys 3 MW @ 50.00 from A1
        # settlement = 3 * 50 * 0.25 = 37.50

        # Agent 4: FOK BUY 4 MW @ 51.00 -> total iceberg=9 MW >= 4 -> FILLS (M6.3)
        # Multi-slice: 3 MW (slice1) + regen slice2 (3 MW) + 1 MW from slice2
        state, r = book.submit_order(state, _order(
            Side.BUY, 51.00, 4.0, 402, agent_id=4, timestamp=ts,
            exec_restriction=ExecRestriction.FOK)); ts += 1
        assert r.traded == True
        assert r.trade_qty == approx(4.0)
        assert r.trade_price == approx(51.00)
        # Trade: A4 buys 4 MW @ 51.00 from A1 (iceberg multi-slice)
        # settlement = 4 * 51 * 0.25 = 51.00

        # Iceberg: 9-4=5 MW remaining. Visible slice2 has 2 MW left.
        # Agent 3: IOC BUY 5 MW @ 51.00 -> M6.4 multi-slice:
        #   Fills 2 MW (slice2 remainder). Regen slice3 (3@51). Crosses IOC (3 MW).
        #   Fills 3 MW. Total: 5 MW. IOC fully filled.
        state, r = book.submit_order(state, _order(
            Side.BUY, 51.00, 5.0, 302, agent_id=3, timestamp=ts,
            exec_restriction=ExecRestriction.IOC)); ts += 1
        assert r.traded and r.trade_qty == approx(5.0)  # M6.4: fills across slices
        # Trade: A3 buys 5 MW @ 51.00 from A1 (iceberg multi-slice)

        # === Phase 3: Cancel + re-submit ===

        # Cancel Agent 2's buy limit (oid=201)
        state = book.cancel_order(state, 201, Side.BUY)
        # Verify no phantom
        order_check = book.get_order(state, 201, Side.BUY)
        assert _is_absent(order_check)

        # Agent 2 re-submits: BUY 4 MW @ 49.00
        state, r = book.submit_order(state, _order(
            Side.BUY, 49.00, 4.0, 203, agent_id=2, timestamp=ts)); ts += 1
        assert r.accepted and r.traded == False

        # === Phase 4: Sell-side sweep ===

        # Agent 3: SELL 5 MW @ 47.00 (crosses buy side: 49.00(4) then 47.00(2 visible))
        state, r = book.submit_order(state, _order(
            Side.SELL, 47.00, 5.0, 303, agent_id=3, timestamp=ts)); ts += 1
        assert r.traded
        # Fills: 4 MW @ 49.00 (A2 limit) + 1 MW @ 47.00 (A2 iceberg peak, partial)
        # Actually the aggressive sell walks: best bid is 49.00 (4 MW), then 47.00 (2 MW visible)
        # 5 MW fills: 4 @ 49.00 + 1 @ 47.00
        assert r.trade_qty == approx(5.0)

        # === PnL verification ===
        # All trades so far:
        # T1: A4 buys 3 MW @ 50.00 from A1 -> V = 3*50*0.25 = 37.50
        # T2: A4 buys 4 MW @ 51.00 from A1 -> V = 4*51*0.25 = 51.00 (M6.3 multi-slice)
        # T3: A3 buys 5 MW @ 51.00 from A1 -> V = 5*51*0.25 = 63.75 (M6.4 IOC multi-slice)
        # T4: A3 sells 4 MW @ 49.00 to A2  -> V = 4*49*0.25 = 49.00
        # T5: A3 sells 1 MW @ 47.00 to A2  -> V = 1*47*0.25 = 11.75

        # Agent 1: sold 3@50 + 4@51 + 5@51 = 12 MW (iceberg fully drained: 3+4+5=12>9?)
        # Wait: A1 limit was 3@50, iceberg was 9@51. FOK 4@51 + IOC 5@51 = 9@51. Total: 12 MW.
        # PnL1 = sell_value = 37.50 + 51.00 + 63.75 = 152.25

        # Agent 2: bought 4@49 + 1@47 = 5 MW
        # PnL2 = -buy_value = -(49.00 + 11.75) = -60.75

        # Agent 3: bought 5@51, sold 4@49 + 1@47
        # PnL3 = sell_value - buy_value = (49.00 + 11.75) - 63.75 = -3.00

        # Agent 4: bought 3@50 + 4@51 = 7 MW
        # PnL4 = -buy_value = -(37.50 + 51.00) = -88.50

        # Sum: 152.25 - 60.75 + (-3.00) - 88.50 = 0.00

        # Tape of all closed trades (T1–T5 above).
        # Columns: (price_eur, qty_mw, buyer, seller).
        trade_tape = _tape_from_trades([
            (50.00, 3.0, 4, 1),  # T1: A4 buys from A1
            (51.00, 4.0, 4, 1),  # T2: A4 buys from A1 (M6.3 multi-slice)
            (51.00, 5.0, 3, 1),  # T3: A3 buys from A1 (M6.4 IOC multi-slice)
            (49.00, 4.0, 2, 3),  # T4: A3 sells to A2
            (47.00, 1.0, 2, 3),  # T5: A3 sells to A2
        ])

        pnl1 = float(compute_agent_pnl(trade_tape, jnp.int32(1)))
        pnl2 = float(compute_agent_pnl(trade_tape, jnp.int32(2)))
        pnl3 = float(compute_agent_pnl(trade_tape, jnp.int32(3)))
        pnl4 = float(compute_agent_pnl(trade_tape, jnp.int32(4)))

        assert pnl1 == approx(152.25)
        assert pnl2 == approx(-60.75)
        assert pnl3 == approx(-3.00)
        assert pnl4 == approx(-88.50)
        # Closed symmetric book — 4 agents, all on the tape — bilateral zero-sum.
        assert (pnl1 + pnl2 + pnl3 + pnl4) == approx(0.0, abs=1e-3)

        # === Final LOB: no phantom orders for cancelled/filled orders ===
        bids, asks = book.get_anonymous_lob(state)
        # Remaining ask: A1 iceberg fully drained (9-4-5=0, M6.4).
        #   A3 limit 2 MW @ 52.00 still resting.
        # Remaining bid: A2 iceberg had 1 MW consumed of peak, so partial.
        #   A2 iceberg: 6 MW total, peak=2 MW. 1 MW consumed from visible slice
        #   -> 1 MW remains of current peak visible

        # Verify no phantom: oid=201 was cancelled
        assert _is_absent(book.get_order(state, 201, Side.BUY))


# ---------------------------------------------------------------------------
# G. NON walking through mixed limit + iceberg at different price levels
# ---------------------------------------------------------------------------

class TestNONWalkMixedBook:
    """NON order walking through multiple price levels with mixed types."""

    def test_non_buy_walks_limit_then_fills_iceberg_multi_slice(self):
        """M6.4: NON BUY walks through limit level, then fills multiple iceberg slices.
        Fills 2@50 (limit) + 3@51 (iceberg peak1) = 5 MW initially. 1 MW rests.
        Regen slice2 (3@51) crosses resting 1 MW bid. Fills 1 MW via Rule 3.
        Multi-slice loop reports total: 6 MW. No resting bid."""
        book = _make_book()
        state = book.init()

        # Limit SELL 2 MW @ 50.00
        state, _ = book.submit_order(state, _order(
            Side.SELL, 50.00, 2.0, 1, agent_id=1, timestamp=1))
        # Iceberg SELL 9 MW (peak=3) @ 51.00
        state, _ = book.submit_order(state, _order(
            Side.SELL, 51.00, 9.0, 2, agent_id=2, timestamp=2,
            exec_restriction=ExecRestriction.ICE, peak_qty=3.0))

        # NON BUY 6 MW @ 51.00 -> M6.4 multi-slice: fills 2@50 + 3@51 + 1@51 = 6 MW
        state, r = book.submit_order(state, _order(
            Side.BUY, 51.00, 6.0, 3, agent_id=3, timestamp=3))
        assert r.traded
        assert r.trade_qty == approx(6.0)  # M6.4: includes Rule 3 regen fill

        # No bids remain (NON fully filled via multi-slice)
        bids, _ = book.get_anonymous_lob(state)
        assert len(bids) == 0


# ---------------------------------------------------------------------------
# H. FOK with limit + iceberg combined at boundary
# ---------------------------------------------------------------------------

class TestFOKLimitPlusIcebergBoundary:
    """FOK where limit + iceberg total crosses boundary (M6.3 total-quantity)."""

    def test_fok_exact_at_combined_visible(self):
        """Limit 2 MW + iceberg total 10 MW = 12 MW matchable. FOK 5 -> fills."""
        book = _make_book()
        state = book.init()

        state, _ = book.submit_order(state, _order(
            Side.SELL, 50.00, 2.0, 1, agent_id=1, timestamp=1))
        state, _ = book.submit_order(state, _order(
            Side.SELL, 50.00, 10.0, 2, agent_id=2, timestamp=2,
            exec_restriction=ExecRestriction.ICE, peak_qty=3.0))

        state, r = book.submit_order(state, _order(
            Side.BUY, 50.00, 5.0, 3, agent_id=3, timestamp=3,
            exec_restriction=ExecRestriction.FOK))
        assert r.traded and r.trade_qty == approx(5.0)

    def test_fok_one_over_combined_visible_fills_with_total(self):
        """Limit 2 + iceberg total 10 = 12 matchable. FOK 5.1 -> fills (M6.3).

        Under M6.3 total-quantity pre-check: 2 + 10 = 12 >= 5.1.
        Multi-slice fill: 2 (limit) + 3 (iceberg peak1) + 0.1 (peak2 partial) = 5.1.
        """
        book = _make_book()
        state = book.init()

        state, _ = book.submit_order(state, _order(
            Side.SELL, 50.00, 2.0, 1, agent_id=1, timestamp=1))
        state, _ = book.submit_order(state, _order(
            Side.SELL, 50.00, 10.0, 2, agent_id=2, timestamp=2,
            exec_restriction=ExecRestriction.ICE, peak_qty=3.0))

        state, r = book.submit_order(state, _order(
            Side.BUY, 50.00, 5.1, 3, agent_id=3, timestamp=3,
            exec_restriction=ExecRestriction.FOK))
        assert r.traded == True
        assert r.trade_qty == approx(5.1)

    def test_fok_over_total_rejected(self):
        """Limit 2 + iceberg total 10 = 12 matchable. FOK 12.1 -> rejected (M6.3).

        Even with total-quantity visibility, 12 < 12.1.
        """
        book = _make_book()
        state = book.init()

        state, _ = book.submit_order(state, _order(
            Side.SELL, 50.00, 2.0, 1, agent_id=1, timestamp=1))
        state, _ = book.submit_order(state, _order(
            Side.SELL, 50.00, 10.0, 2, agent_id=2, timestamp=2,
            exec_restriction=ExecRestriction.ICE, peak_qty=3.0))

        state, r = book.submit_order(state, _order(
            Side.BUY, 50.00, 12.1, 3, agent_id=3, timestamp=3,
            exec_restriction=ExecRestriction.FOK))
        assert r.traded == False


# ---------------------------------------------------------------------------
# I. Cancel + re-submit mid-sequence
# ---------------------------------------------------------------------------

class TestCancelResubmit:
    """Cancel then re-submit preserves book integrity."""

    def test_cancel_limit_resubmit_different_price(self):
        """Cancel a limit order, re-submit at different price, verify LOB."""
        book = _make_book()
        state = book.init()

        state, _ = book.submit_order(state, _order(
            Side.SELL, 50.00, 5.0, 1, agent_id=1, timestamp=1))
        state = book.cancel_order(state, 1, Side.SELL)

        # No asks
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 0

        # Re-submit at 52.00
        state, r = book.submit_order(state, _order(
            Side.SELL, 52.00, 5.0, 2, agent_id=1, timestamp=2))
        assert r.accepted
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        assert asks[0][0] == approx(52.00)

    def test_cancel_nonexistent_is_noop(self):
        """Cancelling a non-existent order does nothing."""
        book = _make_book()
        state = book.init()

        state, _ = book.submit_order(state, _order(
            Side.SELL, 50.00, 5.0, 1, agent_id=1, timestamp=1))
        state_before = state
        state = book.cancel_order(state, 999, Side.SELL)

        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1  # original order still there


# ---------------------------------------------------------------------------
# J. Settlement value consistency
# ---------------------------------------------------------------------------

class TestSettlementConsistency:
    """Settlement values in SubmitResult match V = Q * p * 0.25."""

    def test_settlement_on_simple_fill(self):
        book = _make_book()
        state = book.init()

        state, _ = book.submit_order(state, _order(
            Side.SELL, 60.00, 4.0, 1, agent_id=1, timestamp=1))

        state, r = book.submit_order(state, _order(
            Side.BUY, 60.00, 4.0, 2, agent_id=2, timestamp=2))
        assert r.traded
        expected = 4.0 * 60.00 * 0.25
        assert r.settlement_value == approx(expected)

    def test_settlement_matches_compute_trade_value(self):
        """SubmitResult.settlement_value == compute_trade_value(qty, price)."""
        book = _make_book()
        state = book.init()

        state, _ = book.submit_order(state, _order(
            Side.SELL, 75.50, 2.5, 1, agent_id=1, timestamp=1))

        state, r = book.submit_order(state, _order(
            Side.BUY, 75.50, 2.5, 2, agent_id=2, timestamp=2))
        assert r.traded
        expected = compute_trade_value(2.5, 75.50, 0.25)
        assert r.settlement_value == approx(expected)
