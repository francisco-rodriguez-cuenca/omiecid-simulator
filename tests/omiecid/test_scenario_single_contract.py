"""Tests for M6.1 — End-to-End Scripted Scenarios.

Integration tests exercising the full Phase 1 simulator without any RL agent.
Each test chains the complete flow: config -> contract -> orders -> matching
-> settlement, composing all modules delivered in M1--M5.

All expected values are hand-computed from:
  - docs/omiecid/migration-plan.md §M6 (scenario scripts with expected values)
  - docs/omiecid/market-spec.md §2.3 (price/qty constraints, tick sizes)
  - docs/omiecid/market-spec.md §4.5 (execution restrictions: NON, IOC, FOK, ICE)
  - docs/omiecid/market-spec.md §6.2 (matching: price-time priority, resting price)
  - docs/omiecid/market-spec.md §8 (settlement: V = Q x p x d)
  - docs/omiecid/architecture-reviews.md §M6.1 (scenario design, interface mapping)

Five scripted scenarios:
  1. test_full_single_contract_scenario — 2 participants, multiple orders,
     GCT expiry, PnL verification (migration-plan §M6 verbatim)
  2. test_fok_scenario — FOK rejected (insufficient liquidity), then accepted
  3. test_ioc_partial_fill_scenario — IOC partially fills, remainder cancelled
  4. test_iceberg_scenario — iceberg order fills across 4 slices
  5. test_multiple_price_levels_scenario — aggressive order walks 3 price levels

These tests are black-box: they import only the public interface and assert
against hand-computed domain-specified values. No implementation assumptions.

Trade-side limitation: get_agent_trades() returns (price, qty, timestamp)
without a side field. In these scenarios, each agent's role (buyer/seller)
is known by construction. This assumption is documented per-scenario.
"""

import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.encoding import encode_price
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    OMIEOrder,
    Side,
    ExecRestriction,
    Validity,
    Status,
    make_order,
)
from omiecid_simulator.omiecid.contract_book import SubmitResult
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
    """Convenience wrapper for make_order with contract_id=37."""
    return make_order(
        contract_id=37, side=side, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction, peak_qty=peak_qty, ppd=ppd,
    )


def _agent_tape(raw_trades, *, agent_id, is_seller, counterparty=-9):
    """Build a synthetic (M, 6) int32 trade tape for the new P2-L2 API.

    ``get_agent_trades`` returns ``list[(price_eur, qty_mw, timestamp)]`` —
    it drops the buyer/seller columns. Post-P2-L2, ``compute_agent_pnl``
    reads an ``(M, 6) int32`` tape whose ``price`` column is offset-encoded
    via ``encode_price`` (migration-plan §Decoding contract, v5.4).

    These tests already know ``agent_id``'s role on each trade by
    construction. We rebuild the tape placing ``agent_id`` on the known
    leg and a distinct sentinel ``counterparty`` on the other leg — the
    sentinel is chosen to not collide with any real ``agent_id`` so it
    does not perturb other agents' PnL in the same test.
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
        rows.append([0, 0, 0, -1, -1, 0])  # empty-tape padding
    return jnp.asarray(rows, dtype=jnp.int32)


def _pnl_one_sided(raw_trades, *, agent_id, is_seller):
    """Back-compat shim: compute agent PnL under the new tape-centric API."""
    tape = _agent_tape(raw_trades, agent_id=agent_id, is_seller=is_seller)
    return float(compute_agent_pnl(tape, jnp.int32(agent_id)))


# ===========================================================================
# 1. test_full_single_contract_scenario
#
# Migration-plan §M6 verbatim scenario: two participants trading on
# contract Q37. Agent A (id=1) is always the seller, Agent B (id=2) is
# always the buyer.
#
# Script:
#   1. A submits SELL 10 MW @ 50.00 (NON)
#   2. A submits SELL 5 MW @ 48.50 (NON)
#   3. B submits BUY 8 MW @ 49.00 (NON)
#      -> matches SELL@48.50 (5 MW), residual 3 MW enters book as BUY@49.00
#   4. A submits SELL 3 MW @ 48.00 (IOC)
#      -> matches BUY@49.00 (3 MW at resting price 49.00), IOC fully consumed
#   5. LOB: only SELL 10 MW @ 50.00 remains
#   6. Expire at GCT -> SELL 10 MW cancelled
#   7. PnL:
#      A: sold 5@48.50 + sold 3@49.00 = 5*48.50*0.25 + 3*49.00*0.25
#         = 60.625 + 36.75 = 97.375 EUR
#      B: bought 5@48.50 + bought 3@49.00 = -97.375 EUR
#   8. Trade book: 2 trades per agent
# ===========================================================================

class TestFullSingleContractScenario:
    """Full single-contract scenario: 2 participants, multiple orders,
    GCT expiry, PnL verification."""

    def test_full_single_contract_scenario(self, book, empty_state):
        """Scripted scenario from migration-plan §M6.

        Hand-computed expected values:
          Trade 1: 5 MW @ 48.50 -> V = 5 * 48.50 * 0.25 = 60.625 EUR
          Trade 2: 3 MW @ 49.00 -> V = 3 * 49.00 * 0.25 = 36.75 EUR
          A (seller) PnL: 60.625 + 36.75 = 97.375 EUR
          B (buyer)  PnL: -97.375 EUR
        """
        # Agent IDs: A=1 (seller), B=2 (buyer)
        AGENT_A = 1
        AGENT_B = 2

        # Step 1: A submits SELL 10 MW @ 50.00 (NON)
        state, r1 = book.submit_order(
            empty_state,
            _make_order(Side.SELL, 50.00, 10.0, order_id=1, agent_id=AGENT_A, timestamp=100),
        )
        assert r1.accepted == True
        assert r1.traded == False, "No crossing orders in book yet"

        # Step 2: A submits SELL 5 MW @ 48.50 (NON)
        state, r2 = book.submit_order(
            state,
            _make_order(Side.SELL, 48.50, 5.0, order_id=2, agent_id=AGENT_A, timestamp=200),
        )
        assert r2.accepted == True
        assert r2.traded == False, "Still no buy orders to cross"

        # Step 3: B submits BUY 8 MW @ 49.00 (NON)
        # Crosses SELL@48.50 (5 MW filled at resting price 48.50)
        # Residual 3 MW enters book as BUY@49.00
        state, r3 = book.submit_order(
            state,
            _make_order(Side.BUY, 49.00, 8.0, order_id=3, agent_id=AGENT_B, timestamp=300),
        )
        assert r3.accepted == True
        assert r3.traded == True
        assert r3.trade_qty == pytest.approx(5.0, abs=0.05), \
            "Only 5 MW available at SELL@48.50; BUY@49.00 < SELL@50.00"
        assert r3.trade_price == pytest.approx(48.50, abs=0.01), \
            "Trade at resting sell price 48.50"

        # Verify residual 3 MW BUY@49.00 entered the book
        residual = book.get_order(state, 3, Side.BUY)
        assert not _is_absent(residual), "Residual 3 MW should rest in book"
        assert residual.quantity == pytest.approx(3.0, abs=0.05)

        # Step 4: A submits SELL 3 MW @ 48.00 (IOC)
        # Crosses BUY@49.00 (3 MW at resting price 49.00), IOC fully consumed
        state, r4 = book.submit_order(
            state,
            _make_order(Side.SELL, 48.00, 3.0, order_id=4, agent_id=AGENT_A,
                        timestamp=400, exec_restriction=ExecRestriction.IOC),
        )
        assert r4.accepted == True
        assert r4.traded == True
        assert r4.trade_qty == pytest.approx(3.0, abs=0.05)
        assert r4.trade_price == pytest.approx(49.00, abs=0.01), \
            "Trade at resting buy price 49.00, not incoming sell price 48.00"

        # Step 5: Verify LOB state — only SELL 10 MW @ 50.00 remains
        bids, asks = book.get_anonymous_lob(state)
        assert bids == [], "All buy orders consumed or matched"
        assert len(asks) == 1, "Only one sell order should remain"
        ask_price, ask_qty = asks[0][0], asks[0][1]
        assert ask_price == pytest.approx(50.00, abs=0.01)
        assert ask_qty == pytest.approx(10.0, abs=0.05)

        # Step 6: Expire at GCT -> SELL 10 MW @ 50.00 cancelled
        expired_state = book.expire_at_gct(state)
        bids_exp, asks_exp = book.get_anonymous_lob(expired_state)
        assert bids_exp == [], "All orders cancelled after GCT"
        assert asks_exp == [], "SELL 10@50.00 cancelled after GCT"

        # Step 7: Verify PnL
        # Agent A is seller in both trades. get_agent_trades returns both.
        a_trades_raw = book.get_agent_trades(expired_state, AGENT_A)
        assert len(a_trades_raw) == 2, "Agent A involved in 2 trades"
        a_pnl = _pnl_one_sided(a_trades_raw, agent_id=AGENT_A, is_seller=True)
        assert a_pnl == pytest.approx(97.375), \
            "A: 5*48.50*0.25 + 3*49.00*0.25 = 60.625 + 36.75 = 97.375"

        # Agent B is buyer in both trades.
        b_trades_raw = book.get_agent_trades(expired_state, AGENT_B)
        assert len(b_trades_raw) == 2, "Agent B involved in 2 trades"
        b_pnl = _pnl_one_sided(b_trades_raw, agent_id=AGENT_B, is_seller=False)
        assert b_pnl == pytest.approx(-97.375), \
            "B: -(5*48.50*0.25 + 3*49.00*0.25) = -97.375"

        # Zero-sum check: A's gain = B's loss
        assert a_pnl + b_pnl == pytest.approx(0.0)


# ===========================================================================
# 2. test_fok_scenario
#
# FOK (Fill or Kill): rejected when insufficient liquidity, accepted when
# sufficient. No partial fills allowed.
#
# Script:
#   1. Resting SELL 5 MW @ 50.00
#   2. FOK BUY 10 MW @ 50.00 -> rejected (only 5 available, need 10)
#   3. FOK BUY 5 MW @ 50.00 -> filled
#   4. Book is empty
# ===========================================================================

class TestFokScenario:
    """FOK order: rejected when insufficient liquidity, filled when exact."""

    def test_fok_scenario(self, book, empty_state):
        """FOK rejected then accepted scenario.

        Step 1: Add resting SELL 5 MW @ 50.00
        Step 2: FOK BUY 10 MW @ 50.00 -> rejected (need 10, only 5 available)
        Step 3: FOK BUY 5 MW @ 50.00 -> filled entirely
        Step 4: Book is empty
        """
        SELLER = 1
        BUYER = 2

        # Step 1: Resting sell
        state, r_sell = book.submit_order(
            empty_state,
            _make_order(Side.SELL, 50.00, 5.0, order_id=1, agent_id=SELLER, timestamp=100),
        )
        assert r_sell.accepted == True

        # Step 2: FOK BUY 10 MW @ 50.00 -> rejected (insufficient liquidity)
        state, r_fok_reject = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 10.0, order_id=2, agent_id=BUYER,
                        timestamp=200, exec_restriction=ExecRestriction.FOK),
        )
        assert r_fok_reject.accepted == True, \
            "FOK passes validation (accepted=True), but fails to fill"
        assert r_fok_reject.traded == False, \
            "FOK with insufficient liquidity must not trade"
        assert r_fok_reject.trade_qty == pytest.approx(0.0, abs=0.05), \
            "No quantity traded on rejected FOK"

        # Verify the resting sell is still intact (FOK did not consume any)
        resting = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(resting), "Resting SELL must survive rejected FOK"
        assert resting.quantity == pytest.approx(5.0, abs=0.05)

        # Step 3: FOK BUY 5 MW @ 50.00 -> fully filled
        state, r_fok_fill = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 5.0, order_id=3, agent_id=BUYER,
                        timestamp=300, exec_restriction=ExecRestriction.FOK),
        )
        assert r_fok_fill.accepted == True
        assert r_fok_fill.traded == True
        assert r_fok_fill.trade_qty == pytest.approx(5.0, abs=0.05)
        assert r_fok_fill.trade_price == pytest.approx(50.00, abs=0.01)

        # Step 4: Book is empty after FOK fill
        bids, asks = book.get_anonymous_lob(state)
        assert bids == [], "No buy orders should be in book"
        assert asks == [], "Resting sell fully consumed by FOK"


# ===========================================================================
# 3. test_ioc_partial_fill_scenario
#
# IOC (Immediate or Cancel): partially fills available liquidity,
# remainder is cancelled (does not rest in book).
#
# Script:
#   1. Resting SELL 3 MW @ 50.00
#   2. IOC BUY 8 MW @ 50.00 -> partial fill (3 MW), remainder cancelled
#   3. Book is empty (sell consumed, IOC remainder not in book)
# ===========================================================================

class TestIocPartialFillScenario:
    """IOC order: partial fill, remainder cancelled."""

    def test_ioc_partial_fill_scenario(self, book, empty_state):
        """IOC partial fill scenario.

        Resting SELL 3 MW @ 50.00. IOC BUY 8 MW @ 50.00 fills 3 MW,
        remainder (5 MW) is cancelled. Book is empty afterward.
        """
        SELLER = 1
        BUYER = 2

        # Step 1: Resting sell
        state, r_sell = book.submit_order(
            empty_state,
            _make_order(Side.SELL, 50.00, 3.0, order_id=1, agent_id=SELLER, timestamp=100),
        )
        assert r_sell.accepted == True

        # Step 2: IOC BUY 8 MW @ 50.00 -> partial fill
        state, r_ioc = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 8.0, order_id=2, agent_id=BUYER,
                        timestamp=200, exec_restriction=ExecRestriction.IOC),
        )
        assert r_ioc.accepted == True
        assert r_ioc.traded == True
        assert r_ioc.trade_qty == pytest.approx(3.0, abs=0.05), \
            "Only 3 MW available; IOC fills what it can"
        assert r_ioc.trade_price == pytest.approx(50.00, abs=0.01)

        # Settlement value for the partial fill: 3 MW * 50.00 * 0.25 = 37.50
        assert r_ioc.settlement_value == pytest.approx(37.50, abs=0.01)

        # Step 3: Book is empty
        bids, asks = book.get_anonymous_lob(state)
        assert asks == [], "Resting sell fully consumed"
        assert bids == [], "IOC remainder must NOT enter book"

        # Explicitly verify the IOC order is not in the book
        assert _is_absent(book.get_order(state, 2, Side.BUY)), \
            "IOC remainder (5 MW) must not rest in book"


# ===========================================================================
# 4. test_iceberg_scenario
#
# Iceberg order fills across multiple slices. Each slice gets a new
# timestamp (loses time priority). Total quantity exhausted after 4 fills.
#
# Script:
#   1. ICE SELL 10 MW, peak=3 MW @ 50.00 (agent A)
#   2. LOB shows 3 MW visible
#   3. BUY 3 MW @ 50.00 -> fills peak (slice 1)
#   4. New slice 3 MW appears (slice 2)
#   5. BUY 3 MW @ 50.00 -> fills slice 2
#   6. New slice 3 MW appears (slice 3)
#   7. BUY 3 MW @ 50.00 -> fills slice 3
#   8. Final slice 1 MW appears (10 - 3*3 = 1 remaining)
#   9. BUY 1 MW @ 50.00 -> fills final slice
#   10. Book is empty (total 10 MW exhausted)
# ===========================================================================

class TestIcebergScenario:
    """Iceberg order fills across multiple slices with correct timestamps."""

    def test_iceberg_scenario(self, book, empty_state):
        """Iceberg multi-slice fill scenario.

        ICE SELL total=10 MW, peak=3 MW @ 50.00.
        Four successive BUY orders consume 3 + 3 + 3 + 1 = 10 MW total.
        Each new slice gets a fresh timestamp > previous.
        """
        ICEBERG_SELLER = 1
        BUYER = 2

        # Step 1: Submit iceberg SELL 10 MW, peak=3 MW @ 50.00
        state, r_ice = book.submit_order(
            empty_state,
            _make_order(Side.SELL, 50.00, 10.0, order_id=1, agent_id=ICEBERG_SELLER,
                        timestamp=100, exec_restriction=ExecRestriction.ICE,
                        peak_qty=3.0),
        )
        assert r_ice.accepted == True

        # Step 2: LOB shows only 3 MW visible (peak), not 10 MW
        _bids, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        assert asks[0][1] == pytest.approx(3.0, abs=0.05), \
            "LOB should show peak_qty (3 MW), not total (10 MW)"

        # Steps 3-10: Four successive fills consuming the iceberg
        total_traded = 0.0
        fill_sizes = [3.0, 3.0, 3.0, 1.0]  # 3+3+3+1 = 10 MW total

        for i, expected_fill in enumerate(fill_sizes):
            buy_ts = 200 + i * 100  # 200, 300, 400, 500
            state, r_buy = book.submit_order(
                state,
                _make_order(Side.BUY, 50.00, expected_fill, order_id=10 + i,
                            agent_id=BUYER, timestamp=buy_ts),
            )
            assert r_buy.traded == True, f"Fill {i+1}: should trade"
            assert r_buy.trade_qty == pytest.approx(expected_fill, abs=0.05), \
                f"Fill {i+1}: expected {expected_fill} MW"
            assert r_buy.trade_price == pytest.approx(50.00, abs=0.01), \
                f"Fill {i+1}: trade at resting iceberg price"
            total_traded += expected_fill

            if i < len(fill_sizes) - 1:
                # Verify a new slice appeared
                _bids, asks = book.get_anonymous_lob(state)
                assert len(asks) >= 1, f"After fill {i+1}: new slice should appear"
                next_expected = min(3.0, 10.0 - total_traded)
                assert asks[0][1] == pytest.approx(next_expected, abs=0.05), \
                    f"After fill {i+1}: next slice should be {next_expected} MW"

        # Step 10: Book is empty (total 10 MW exhausted)
        assert total_traded == pytest.approx(10.0, abs=0.05)
        bids, asks = book.get_anonymous_lob(state)
        assert asks == [], "Iceberg fully exhausted; book should be empty"
        assert bids == [], "No buy orders should remain"

        # Verify total settlement: 10 MW * 50.00 * 0.25 = 125.00 EUR
        buyer_trades = book.get_agent_trades(state, BUYER)
        assert len(buyer_trades) == 4, "Buyer had 4 separate fills"
        buyer_total_value = sum(
            compute_trade_value(quantity_mw=qty, price_eur_mwh=price)
            for price, qty, *_rest in buyer_trades
        )
        assert buyer_total_value == pytest.approx(125.0, abs=0.01), \
            "Total settlement = 10 * 50.00 * 0.25 = 125.00 EUR"


# ===========================================================================
# 5. test_multiple_price_levels_scenario
#
# Aggressive BUY order walks through multiple resting SELL price levels.
# Price-time priority: best (lowest) ask matched first.
#
# Script:
#   1. Resting sells: 2 MW@48, 3 MW@49, 5 MW@50
#   2. BUY 7 MW @ 50.00 -> walks 3 levels: 2@48 + 3@49 + 2@50
#   3. SubmitResult: traded=True, trade_qty=7.0
#   4. trade_price = VWAP = (2*48 + 3*49 + 2*50) / 7 = 343/7 = 49.0
#   5. LOB: only SELL 3 MW @ 50.00 remains (5-2=3)
#   6. settlement_value = 7 * 49.0 * 0.25 = 85.75
# ===========================================================================

class TestMultiplePriceLevelsScenario:
    """Aggressive order walks through multiple resting price levels."""

    def test_multiple_price_levels_scenario(self, book, empty_state):
        """Multi-price-level walk scenario.

        Three resting sells at 48, 49, 50. Aggressive BUY 7 MW @ 50.00
        fills across all three levels: 2@48 + 3@49 + 2@50 = 7 MW total.

        Hand-computed:
          VWAP = (2*48 + 3*49 + 2*50) / 7 = (96 + 147 + 100) / 7 = 343/7 = 49.0
          settlement = 7 * 49.0 * 0.25 = 85.75 EUR
        """
        SELLER_1 = 1
        SELLER_2 = 2
        SELLER_3 = 3
        BUYER = 4

        # Step 1: Resting sells at 3 price levels (price-time priority)
        state, r1 = book.submit_order(
            empty_state,
            _make_order(Side.SELL, 48.00, 2.0, order_id=1, agent_id=SELLER_1, timestamp=100),
        )
        assert r1.accepted == True

        state, r2 = book.submit_order(
            state,
            _make_order(Side.SELL, 49.00, 3.0, order_id=2, agent_id=SELLER_2, timestamp=200),
        )
        assert r2.accepted == True

        state, r3 = book.submit_order(
            state,
            _make_order(Side.SELL, 50.00, 5.0, order_id=3, agent_id=SELLER_3, timestamp=300),
        )
        assert r3.accepted == True

        # Step 2: Aggressive BUY 7 MW @ 50.00 walks through 3 levels
        state, r_buy = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 7.0, order_id=4, agent_id=BUYER, timestamp=400),
        )

        # Step 3: SubmitResult verification
        assert r_buy.accepted == True
        assert r_buy.traded == True
        assert r_buy.trade_qty == pytest.approx(7.0, abs=0.05), \
            "Total filled = 2 + 3 + 2 = 7 MW"

        # Step 4: VWAP = (2*48 + 3*49 + 2*50) / 7 = 343/7 = 49.0
        expected_vwap = (2 * 48.0 + 3 * 49.0 + 2 * 50.0) / 7.0
        assert expected_vwap == pytest.approx(49.0, abs=0.001)  # sanity check
        assert r_buy.trade_price == pytest.approx(expected_vwap, abs=0.01), \
            "trade_price should be VWAP across filled levels"

        # Step 5: LOB verification — only SELL 3 MW @ 50.00 remains
        bids, asks = book.get_anonymous_lob(state)
        assert bids == [], "Aggressive buy fully consumed (no residual)"
        assert len(asks) == 1, "Only one sell level should remain"
        assert asks[0][0] == pytest.approx(50.00, abs=0.01), \
            "Remaining sell at 50.00"
        assert asks[0][1] == pytest.approx(3.0, abs=0.05), \
            "5 MW - 2 MW filled = 3 MW remaining"

        # Step 6: Settlement value = 7 * 49.0 * 0.25 = 85.75 EUR
        assert r_buy.settlement_value == pytest.approx(85.75, abs=0.01), \
            "settlement_value = trade_qty * VWAP * 0.25"
