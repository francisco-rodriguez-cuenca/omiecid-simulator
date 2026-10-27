"""Tests for M6.2 — Mixed Order-Type Simulation (updated M6.3, M6.4).

Large scripted integration tests exercising combinatorial interactions between
order types in the same book. 3-4 agents, 20-30 orders per test.

All expected values are hand-computed from:
  - docs/omiecid/market-spec.md §2.3 (price/qty constraints, tick sizes)
  - docs/omiecid/market-spec.md §4.2 (iceberg orders, ICEwPI price increment)
  - docs/omiecid/market-spec.md §4.2.1 (iceberg interaction semantics, M6.3/M6.4 rules)
  - docs/omiecid/market-spec.md §4.5 (execution restrictions: NON, IOC, FOK, ICE)
  - docs/omiecid/market-spec.md §6.2 (matching: price-time priority, resting price)
  - docs/omiecid/market-spec.md §8 (settlement: V = Q x p x d)

CRITICAL DOMAIN RULES (revision v5, M6.4 update):

  1. FOK vs Iceberg visibility (Rule 1 — M6.3):
     FOK pre-check sees TOTAL iceberg quantity (visible peak + hidden remainder)
     at crossing prices. A FOK BUY for N MW passes if total matchable quantity
     (visible limit orders + total iceberg quantity) >= N. After passing,
     FOK uses multi-slice matching: fills visible, regenerates iceberg slices,
     re-fills, until the full FOK quantity is matched within one submit_order.

  2. Multi-slice matching for all order types (Rule 2 — M6.4 update):
     All aggressive order types (NON, IOC, FOK) can fill multiple iceberg
     slices within one submit_order call. After the initial match, iceberg
     slices are regenerated; if the regenerated slice crosses the aggressive
     order's resting remainder (Rule 3), matching occurs again. This repeats
     until no more regeneration produces fills or the aggressive order is
     fully filled. IOC cancels any unfilled remainder after the loop.

     Implications:
       - NON hitting iceberg: fills across multiple slices via regen loop,
         unfilled remainder rests in book
       - IOC hitting iceberg: fills across multiple slices, remainder cancelled
       - FOK hitting iceberg: pre-check uses total qty; if it passes,
         multi-slice fill across regenerations within one call
       - A single NON/IOC/FOK order can drain an entire iceberg

Two scripted scenarios:
  1. test_mixed_order_type_simulation — 4 agents, 4 phases, ~25 orders:
     Phase A: build mixed resting book (limit + iceberg + ICEwPI on both sides)
     Phase B: aggressive orders (NON, FOK total-qty check, IOC partial)
     Phase C: cancel + re-submit mid-sequence
     Phase D: final sweep + PnL verification + zero-sum check
  2. test_iceberg_fok_ioc_interaction — focused interaction between iceberg
     resting liquidity and FOK/IOC aggressors, testing the total-quantity
     FOK pre-check (Rule 1) and single-slice NON/IOC rule (Rule 2)

These tests are black-box: they import only the public interface and assert
against hand-computed domain-specified values. No implementation assumptions.

Trade-side limitation: get_agent_trades() returns (price, qty, timestamp)
without a side field. Each agent's role (buyer/seller) is known by construction.
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
    """Build a synthetic (M, 6) int32 tape for P2-L2's tape-centric API.

    See ``test_scenario_single_contract.py::_agent_tape`` for rationale.
    ``get_agent_trades`` drops buyer/seller columns; this tests already
    knows ``agent_id``'s role by construction, so we rebuild the tape
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


# ===========================================================================
# 1. test_mixed_order_type_simulation
#
# 4 agents, 4 phases exercising combinatorial order type interactions.
#
# Agents:
#   A1 (id=1) — seller (limit orders)
#   A2 (id=2) — buyer (limit + iceberg)
#   A3 (id=3) — seller (iceberg + ICEwPI)
#   A4 (id=4) — aggressive buyer/seller
#
# Phase A — Build mixed resting book (7 orders, no trades):
#   O1:  A1 SELL  5 MW @ 52.00 (NON)          ts=100
#   O2:  A1 SELL  3 MW @ 54.00 (NON)          ts=110
#   O3:  A3 SELL  9 MW @ 50.00 (ICE, peak=3)  ts=120  [visible: 3@50]
#   O4:  A2 BUY   4 MW @ 47.00 (NON)          ts=130
#   O5:  A2 BUY   6 MW @ 46.00 (NON)          ts=140
#   O6:  A2 BUY   8 MW @ 45.00 (ICE, peak=3)  ts=150  [visible: 3@45]
#   O7:  A3 SELL  6 MW @ 51.00 (ICE, peak=2, ppd=0.50) ts=160
#        ICEwPI sell: slice1@51.00, slice2@51.50, slice3@52.00
#        [visible: 2@51]
#
#   LOB visible after Phase A:
#     ASKS: 3@50.00, 2@51.00, 5@52.00, 3@54.00
#     BIDS: 4@47.00, 6@46.00, 3@45.00
#   Best bid=47 < best ask=50 -> no crossing.
#
# Phase B — Aggressive orders (6 orders):
#   O8:  A4 BUY 3 MW @ 50.00 (NON) — hits O3 iceberg          ts=200
#        Fills exactly the visible peak = 3@50 (order qty = peak qty).
#        O3: 9-3=6 MW remaining. New slice: 3@50 visible.
#        Trade: 3 MW @ 50.00.
#
#   O8b: A4 BUY 3 MW @ 50.00 (NON) — hits O3 iceberg slice2   ts=210
#        Fills 3@50 (slice2 peak). O3: 6-3=3 MW remaining. New slice: 3@50.
#        Trade: 3 MW @ 50.00.
#
#   O9:  A4 BUY 10 MW @ 55.00 (FOK) — total-quantity pre-check (M6.3): ts=300
#        MATCHABLE asks <=55 (M6.3: visible + hidden iceberg):
#          O3: 3 MW total @50.00 (last slice, no hidden)
#          O7: 6 MW total @51.00 (2 visible + 4 hidden)
#          O1: 5 MW @52.00 (regular limit)
#          O2: 3 MW @54.00 (regular limit)
#          Total matchable = 3+6+5+3 = 17 MW >= 10. Pre-check passes.
#        Execution in strict price-time priority (market-spec §6.1,
#        §4.2.1 Rule 2; re-baselined by OMIE-ICE-PRICE-PRIORITY-1): a
#        worse price level is never walked while a better-priced iceberg
#        still has a slice to regenerate, and each regenerated slice is
#        ranked as a NEW entry (it queues behind older orders at its price):
#          3@50.00 (O3 exhausted)
#          2@51.00 (O7 slice1 consumed -> slice2 2@51.50 regenerates)
#          2@51.50 (O7 slice2 consumed -> slice3 2@52.00 regenerates,
#                   ranked behind O1, which rests at 52.00 since ts=100)
#          3@52.00 (from O1, the older order at 52.00)
#          Total = 3+2+2+3 = 10 MW.
#        VWAP = (3*50 + 2*51 + 2*51.50 + 3*52) / 10
#             = (150 + 102 + 103 + 156) / 10 = 511/10 = 51.10
#        (Before the fix the engine sold O1's 5@52.00 before O7's slice2
#        @51.50 existed: VWAP 512/10 = 51.20 — a §6.1 price-priority breach.)
#        O3 exhausted. O7: 6-2-2=2 MW remaining, slice3 2@52.00 visible, no hidden.
#        O1: 5-3=2 MW left @52.00. O2 still resting (3@54).
#        Best ask level after O9: 52.00 with 2 (O1) + 2 (O7 slice3) = 4 MW.
#
#   O10: A4 SELL 5 MW @ 46.50 (IOC) — matches resting bids:    ts=400
#        Best bid: O4 A2 BUY 4 MW@47.00. Cross. Fill 4 MW@47.00.
#        Next: O5 A2 BUY 6 MW@46.00. 46.50 > 46.00, no cross.
#        IOC: 4 MW filled, 1 MW cancelled.
#        Trade: 4 MW @ 47.00. O4 consumed.
#
#   O11: A4 SELL 8 MW @ 44.00 (FOK) — total-quantity pre-check (M6.3): ts=500
#        MATCHABLE bids >=44 (M6.3: visible + hidden iceberg):
#          O5: 6 MW @46.00 (regular limit)
#          O6: 8 MW total @45.00 (3 visible + 5 hidden)
#          Total matchable = 6+8 = 14 MW >= 8. Pre-check passes.
#        Execution (visible=9 >= 8, fills in first pass):
#          6@46.00 (O5 consumed)
#          2@45.00 (O6 slice1 partial: fill 2 from 3 visible)
#          Total: 6+2 = 8 MW.
#        VWAP = (6*46 + 2*45) / 8 = (276 + 90) / 8 = 366/8 = 45.75
#        O5 consumed. O6: slice1 has 1 MW remaining visible, 8-2=6 total remaining.
#
#   O12: A2 BUY 2 MW @ 51.50 (FOK) — total-quantity pre-check (M6.3): ts=600
#        MATCHABLE asks <=51.50: none (O7's slice2 @51.50 already sold
#        inside O9; best ask is 52.00). Total matchable = 0 < 2.
#        FOK KILLED: no trade, nothing rests. Book unchanged:
#        asks 4@52.00 (O1 2 + O7 slice3 2), 3@54.00 (O2).
#
# Phase C — Cancel + re-submit (3 orders):
#   C1: A1 cancels O2 (SELL 3@54.00, still resting)           ts=700
#       O2 removed from book. No phantom order.
#   C2: A1 submits new O13: SELL 4 MW@53.00 (NON)             ts=710
#       Enters book (no crossing, best bid is 45 from O6).
#   C3: A2 cancels O6 (BUY iceberg, 6 MW remaining@45.00)     ts=720
#       O6 removed from book entirely. No phantom order.
#
#   LOB visible after Phase C:
#     ASKS: 4@52.00 (O1 2 MW, then O7 slice3 2 MW), 4@53.00(O13)
#     BIDS: (empty — all bids consumed or cancelled)
#
# Phase D — Final sweep + PnL (1 order):
#   O14: A2 BUY 6 MW @ 53.00 (NON)                            ts=800
#        Price-time at 52.00: O1 (older) first, then O7 slice3.
#        Fills 2@52.00 (O1 consumed) + 2@52.00 (O7 slice3, O7 exhausted)
#        + 2@53.00 (O13 partial). Total: 2+2+2 = 6 MW.
#        VWAP = (2*52 + 2*52 + 2*53) / 6 = (104+104+106)/6 = 314/6 ≈ 52.3333
#
#   LOB after Phase D: no bids; O13 keeps 2@53.00 on the ask side.
#
# Trade summary (all trades, chronological):
#   T1:  A4 buys 3@50.00 from A3     V=3*50*0.25=37.50     (O8)
#   T2:  A4 buys 3@50.00 from A3     V=3*50*0.25=37.50     (O8b)
#   T3:  A4 buys 3@50.00 from A3     V=3*50*0.25=37.50     (O9: O3 slice3)
#   T4:  A4 buys 2@51.00 from A3     V=2*51*0.25=25.50     (O9: O7 slice1)
#   T5:  A4 buys 2@51.50 from A3     V=2*51.50*0.25=25.75  (O9: O7 slice2)
#   T6:  A4 buys 3@52.00 from A1     V=3*52*0.25=39.00     (O9: O1)
#   T7:  A4 sells 4@47.00 to A2      V=4*47*0.25=47.00     (O10)
#   T8:  A4 sells 6@46.00 to A2      V=6*46*0.25=69.00     (O11: O5)
#   T9:  A4 sells 2@45.00 to A2      V=2*45*0.25=22.50     (O11: O6 slice1)
#        (O12 FOK killed: no trade)
#   T10: A2 buys 2@52.00 from A1     V=2*52*0.25=26.00     (O14: O1)
#   T11: A2 buys 2@52.00 from A3     V=2*52*0.25=26.00     (O14: O7 slice3)
#   T12: A2 buys 2@53.00 from A1     V=2*53*0.25=26.50     (O14: O13)
#
# PnL per agent (sell_revenue - buy_cost):
#   A1: sells 3@52(T6) + 2@52(T10) + 2@53(T12) = 7 MW
#     Revenue: 39.00 + 26.00 + 26.50 = 91.50
#     PnL_A1 = +91.50
#
#   A2: buys 4@47(T7) + 6@46(T8) + 2@45(T9) + 2@52(T10) + 2@52(T11) + 2@53(T12)
#     Note: T7,T8,T9 are where A4 sells TO A2, so A2 is buyer at those prices
#     Cost: 47.00 + 69.00 + 22.50 + 26.00 + 26.00 + 26.50 = 217.00
#     PnL_A2 = -217.00
#
#   A3: sells 3@50(T1) + 3@50(T2) + 3@50(T3) + 2@51(T4) + 2@51.50(T5) + 2@52(T11)
#     Revenue: 37.50 + 37.50 + 37.50 + 25.50 + 25.75 + 26.00 = 189.75
#     PnL_A3 = +189.75  (unchanged: O7's slice2 now sells inside O9, not O12)
#
#   A4: buys 3@50+3@50+3@50+2@51+2@51.50+3@52
#         (cost=37.50+37.50+37.50+25.50+25.75+39.00=202.75)
#       sells 4@47+6@46+2@45 (revenue=47+69+22.5=138.50)
#     PnL_A4 = 138.50 - 202.75 = -64.25
#
# Zero-sum check: 91.50 + (-217.00) + 189.75 + (-64.25) = 0.00 ✓
#
# ===========================================================================

class TestMixedOrderTypeSimulation:
    """4-phase mixed order-type simulation: 4 agents, ~25 orders."""

    def test_mixed_order_type_simulation(self, book, empty_state):
        """Full mixed-type scenario with hand-computed expected values.

        Phase A: 7 orders build mixed resting book (limit + ICE + ICEwPI)
        Phase B: 6 aggressive orders (NON, FOK visible-only check, IOC, FOK)
        Phase C: cancel + re-submit
        Phase D: final sweep + PnL zero-sum verification

        Hand-computed PnL (re-baselined by OMIE-ICE-PRICE-PRIORITY-1):
          A1: +91.50  A2: -217.00  A3: +189.75  A4: -64.25
          Sum: 0.00 (zero-sum)
        """
        A1, A2, A3, A4 = 1, 2, 3, 4

        # =================================================================
        # Phase A — Build mixed resting book (7 orders, no trades)
        # =================================================================

        # O1: A1 SELL 5 MW @ 52.00 (NON)
        state, r = book.submit_order(
            empty_state,
            _make_order(Side.SELL, 52.00, 5.0, order_id=1, agent_id=A1, timestamp=100),
        )
        assert r.accepted == True
        assert r.traded == False

        # O2: A1 SELL 3 MW @ 54.00 (NON)
        state, r = book.submit_order(
            state,
            _make_order(Side.SELL, 54.00, 3.0, order_id=2, agent_id=A1, timestamp=110),
        )
        assert r.accepted == True
        assert r.traded == False

        # O3: A3 SELL 9 MW @ 50.00 (ICE, peak=3) — standard iceberg
        state, r = book.submit_order(
            state,
            _make_order(Side.SELL, 50.00, 9.0, order_id=3, agent_id=A3, timestamp=120,
                        exec_restriction=ExecRestriction.ICE, peak_qty=3.0),
        )
        assert r.accepted == True
        assert r.traded == False

        # O4: A2 BUY 4 MW @ 47.00 (NON)
        state, r = book.submit_order(
            state,
            _make_order(Side.BUY, 47.00, 4.0, order_id=4, agent_id=A2, timestamp=130),
        )
        assert r.accepted == True
        assert r.traded == False

        # O5: A2 BUY 6 MW @ 46.00 (NON)
        state, r = book.submit_order(
            state,
            _make_order(Side.BUY, 46.00, 6.0, order_id=5, agent_id=A2, timestamp=140),
        )
        assert r.accepted == True
        assert r.traded == False

        # O6: A2 BUY 8 MW @ 45.00 (ICE, peak=3) — standard iceberg
        state, r = book.submit_order(
            state,
            _make_order(Side.BUY, 45.00, 8.0, order_id=6, agent_id=A2, timestamp=150,
                        exec_restriction=ExecRestriction.ICE, peak_qty=3.0),
        )
        assert r.accepted == True
        assert r.traded == False

        # O7: A3 SELL 6 MW @ 51.00 (ICEwPI, peak=2, ppd=+0.50)
        # Slices: slice1@51.00, slice2@51.50, slice3@52.00
        state, r = book.submit_order(
            state,
            _make_order(Side.SELL, 51.00, 6.0, order_id=7, agent_id=A3, timestamp=160,
                        exec_restriction=ExecRestriction.ICE, peak_qty=2.0, ppd=0.50),
        )
        assert r.accepted == True
        assert r.traded == False

        # Verify LOB visible state after Phase A
        bids, asks = book.get_anonymous_lob(state)
        # ASKS: 3@50.00(O3 ice peak), 2@51.00(O7 ice peak), 5@52.00(O1), 3@54.00(O2)
        assert len(asks) == 4, \
            f"Expected 4 visible ask levels, got {len(asks)}"
        assert asks[0][0] == pytest.approx(50.00, abs=0.01)
        assert asks[0][1] == pytest.approx(3.0, abs=0.05), \
            "O3 iceberg: only peak 3 MW visible"
        assert asks[1][0] == pytest.approx(51.00, abs=0.01)
        assert asks[1][1] == pytest.approx(2.0, abs=0.05), \
            "O7 ICEwPI: only peak 2 MW visible"
        assert asks[2][0] == pytest.approx(52.00, abs=0.01)
        assert asks[2][1] == pytest.approx(5.0, abs=0.05)
        assert asks[3][0] == pytest.approx(54.00, abs=0.01)
        assert asks[3][1] == pytest.approx(3.0, abs=0.05)

        # BIDS: 4@47.00(O4), 6@46.00(O5), 3@45.00(O6 ice peak)
        assert len(bids) == 3, \
            f"Expected 3 visible bid levels, got {len(bids)}"
        assert bids[0][0] == pytest.approx(47.00, abs=0.01)
        assert bids[0][1] == pytest.approx(4.0, abs=0.05)
        assert bids[1][0] == pytest.approx(46.00, abs=0.01)
        assert bids[1][1] == pytest.approx(6.0, abs=0.05)
        assert bids[2][0] == pytest.approx(45.00, abs=0.01)
        assert bids[2][1] == pytest.approx(3.0, abs=0.05), \
            "O6 iceberg: only peak 3 MW visible"

        # =================================================================
        # Phase B — Aggressive orders against mixed book
        # =================================================================

        # O8: A4 BUY 3 MW @ 50.00 (NON) — fills O3 iceberg slice1 exactly
        # Fills peak = 3@50. No residual (order qty = peak qty).
        state, r8 = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 3.0, order_id=8, agent_id=A4, timestamp=200),
        )
        assert r8.accepted == True
        assert r8.traded == True
        assert r8.trade_qty == pytest.approx(3.0, abs=0.05), \
            "O8: fills exactly O3 slice1 peak = 3 MW"
        assert r8.trade_price == pytest.approx(50.00, abs=0.01)

        # O3: 9-3=6 MW remaining. New slice2: 3@50 visible.
        _, asks_b1 = book.get_anonymous_lob(state)
        assert asks_b1[0][0] == pytest.approx(50.00, abs=0.01)
        assert asks_b1[0][1] == pytest.approx(3.0, abs=0.05), \
            "O3 slice2 reloaded: 3 MW visible"

        # O8b: A4 BUY 3 MW @ 50.00 (NON) — fills O3 iceberg slice2
        state, r8b = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 3.0, order_id=80, agent_id=A4, timestamp=210),
        )
        assert r8b.accepted == True
        assert r8b.traded == True
        assert r8b.trade_qty == pytest.approx(3.0, abs=0.05)
        assert r8b.trade_price == pytest.approx(50.00, abs=0.01)

        # O3: 9-6=3 MW remaining. New slice3: 3@50 visible.
        _, asks_b1b = book.get_anonymous_lob(state)
        assert asks_b1b[0][0] == pytest.approx(50.00, abs=0.01)
        assert asks_b1b[0][1] == pytest.approx(3.0, abs=0.05), \
            "O3 slice3 reloaded: min(3, 3)=3 MW visible"

        # O9: A4 BUY 10 MW @ 55.00 (FOK) — total-quantity pre-check (M6.3)
        # Matchable asks <=55: O3(3@50 total) + O7(6@51 total) + O1(5@52) + O2(3@54) = 17
        # Price-time (§6.1, §4.2.1 Rule 2; OMIE-ICE-PRICE-PRIORITY-1): O7's slice2
        # @51.50 trades before any 52.00 order; O7's slice3 @52.00 is a new entry
        # ranked behind O1 (ts=100).
        # Execution: 3@50(O3 exhausted) + 2@51(O7 slice1) + 2@51.50(O7 slice2)
        #            + 3@52(O1) = 10 MW
        # VWAP = (3*50 + 2*51 + 2*51.50 + 3*52)/10 = (150+102+103+156)/10 = 511/10 = 51.10
        state, r9 = book.submit_order(
            state,
            _make_order(Side.BUY, 55.00, 10.0, order_id=9, agent_id=A4, timestamp=300,
                        exec_restriction=ExecRestriction.FOK),
        )
        assert r9.accepted == True
        assert r9.traded == True, \
            "FOK: 17 MW matchable (13 visible + 4 hidden) >= 10 MW, pre-check passes"
        assert r9.trade_qty == pytest.approx(10.0, abs=0.05)
        expected_vwap_o9 = 511.0 / 10.0  # = 51.10
        assert r9.trade_price == pytest.approx(expected_vwap_o9, abs=0.01), \
            f"FOK VWAP = 511/10 = {expected_vwap_o9:.2f}"

        # After O9: O3 exhausted, O7 slices 1 and 2 consumed -> slice3 2@52.00
        # visible (no hidden left); O1 has 2 MW left @52.00; O2(3@54) remains
        _, asks_b2 = book.get_anonymous_lob(state)
        # Visible levels: 52.00 (O1 2 + O7 slice3 2 = 4 MW) + O2 (3@54)
        assert len(asks_b2) == 2, \
            f"Expected 2 ask levels after FOK, got {len(asks_b2)}"
        assert asks_b2[0][0] == pytest.approx(52.00, abs=0.01), \
            "Best ask after O9: 52.00 (O1 remainder + O7 slice3 = 51.00 + 2*0.50)"
        assert asks_b2[0][1] == pytest.approx(4.0, abs=0.05), \
            "52.00 level: O1 2 MW + O7 slice3 2 MW = 4 MW visible"
        assert asks_b2[1][0] == pytest.approx(54.00, abs=0.01)
        assert asks_b2[1][1] == pytest.approx(3.0, abs=0.05)

        # O10: A4 SELL 5 MW @ 46.50 (IOC) — partial fill against bids
        # Best bid: O4 A2 BUY 4@47.00. Cross. Fill 4@47.
        # Next: O5 A2 BUY 6@46.00 < 46.50. No cross.
        # IOC: 4 MW filled, 1 MW cancelled.
        state, r10 = book.submit_order(
            state,
            _make_order(Side.SELL, 46.50, 5.0, order_id=10, agent_id=A4, timestamp=400,
                        exec_restriction=ExecRestriction.IOC),
        )
        assert r10.accepted == True
        assert r10.traded == True
        assert r10.trade_qty == pytest.approx(4.0, abs=0.05), \
            "IOC: only 4 MW available at >=46.50 (O4@47.00)"
        assert r10.trade_price == pytest.approx(47.00, abs=0.01), \
            "Trade at resting bid price 47.00"

        # IOC remainder must not rest in book
        assert _is_absent(book.get_order(state, 10, Side.SELL)), \
            "IOC remainder must NOT enter book"

        # O11: A4 SELL 8 MW @ 44.00 (FOK) — total-quantity pre-check on bids (M6.3)
        # Matchable bids >=44: O5(6@46) + O6(8@45 total) = 14 MW >= 8.
        # Visible alone = 6+3 = 9 >= 8 -> fills in first pass (no multi-slice needed)
        # Execution: 6@46(O5 consumed) + 2@45(O6 slice1 partial, 2 from 3 visible) = 8 MW
        # VWAP = (6*46 + 2*45)/8 = (276+90)/8 = 366/8 = 45.75
        state, r11 = book.submit_order(
            state,
            _make_order(Side.SELL, 44.00, 8.0, order_id=11, agent_id=A4, timestamp=500,
                        exec_restriction=ExecRestriction.FOK),
        )
        assert r11.accepted == True
        assert r11.traded == True, \
            "FOK: 14 MW matchable (9 visible + 5 hidden) >= 8 MW, pre-check passes"
        assert r11.trade_qty == pytest.approx(8.0, abs=0.05)
        assert r11.trade_price == pytest.approx(45.75, abs=0.01), \
            "FOK VWAP = (6*46+2*45)/8 = 45.75"

        # O5 consumed. O6: slice1 had 3 visible, 2 filled -> 1 MW remaining visible.
        # O6 total: 8-2=6 MW remaining.
        bids_b3, _ = book.get_anonymous_lob(state)
        assert len(bids_b3) == 1, \
            f"Expected 1 bid level (O6 ice), got {len(bids_b3)}"
        assert bids_b3[0][0] == pytest.approx(45.00, abs=0.01)
        assert bids_b3[0][1] == pytest.approx(1.0, abs=0.05), \
            "O6 iceberg: slice1 had 3-2=1 MW remaining visible"

        # O12: A2 BUY 2 MW @ 51.50 (FOK) — total-quantity pre-check (M6.3)
        # Matchable asks <=51.50: none (O7 slice2 @51.50 already sold inside O9;
        # best ask is 52.00). 0 < 2 -> FOK KILLED: no trade, nothing rests.
        state, r12 = book.submit_order(
            state,
            _make_order(Side.BUY, 51.50, 2.0, order_id=12, agent_id=A2, timestamp=600,
                        exec_restriction=ExecRestriction.FOK),
        )
        assert r12.accepted == True
        assert r12.traded == False, "FOK: no ask <= 51.50 remains -> killed"
        assert r12.trade_qty == pytest.approx(0.0, abs=0.05)
        assert r12.trade_price == pytest.approx(0.0, abs=0.01)

        # After O12 (killed): book unchanged — 52.00 (O1 2 + O7 slice3 2), O2 at 54.00
        _, asks_b4 = book.get_anonymous_lob(state)
        assert asks_b4[0][0] == pytest.approx(52.00, abs=0.01), \
            "Best ask: 52.00 (O1 remainder + O7 slice3 = 51.00 + 2*0.50)"
        assert asks_b4[0][1] == pytest.approx(4.0, abs=0.05), \
            "52.00 level: O1 2 MW + O7 slice3 2 MW = 4 MW visible"

        # =================================================================
        # Phase C — Cancel + re-submit
        # =================================================================

        # C1: Cancel O2 (A1 SELL 3@54, still resting)
        state = book.cancel_order(state, 2, Side.SELL)
        assert _is_absent(book.get_order(state, 2, Side.SELL)), \
            "O2 must be removed after cancel — no phantom order"

        # C2: A1 submits new O13: SELL 4 MW @ 53.00 (NON)
        state, r13 = book.submit_order(
            state,
            _make_order(Side.SELL, 53.00, 4.0, order_id=13, agent_id=A1, timestamp=710),
        )
        assert r13.accepted == True
        assert r13.traded == False, "Best bid 45 < 53, no crossing"

        # C3: Cancel O6 (A2 BUY iceberg, 6 MW remaining@45)
        state = book.cancel_order(state, 6, Side.BUY)
        assert _is_absent(book.get_order(state, 6, Side.BUY)), \
            "O6 iceberg must be fully removed after cancel — no phantom order"

        # Verify LOB after Phase C
        bids_c, asks_c = book.get_anonymous_lob(state)
        assert bids_c == [], "All bids consumed or cancelled"
        assert len(asks_c) == 2, \
            f"Expected 2 ask levels: 52.00 (O1 2 + O7s3 2) + O13(4@53), got {len(asks_c)}"
        assert asks_c[0][0] == pytest.approx(52.00, abs=0.01)
        assert asks_c[0][1] == pytest.approx(4.0, abs=0.05)
        assert asks_c[1][0] == pytest.approx(53.00, abs=0.01)
        assert asks_c[1][1] == pytest.approx(4.0, abs=0.05)

        # =================================================================
        # Phase D — Final sweep + PnL verification
        # =================================================================

        # O14: A2 BUY 6 MW @ 53.00 (NON)
        # Price-time at 52.00: O1 (older) then O7 slice3 (O7 exhausted), then O13.
        # Fills 2@52(O1) + 2@52(O7 slice3) + 2@53(O13 partial). Total: 6 MW.
        # VWAP = (2*52 + 2*52 + 2*53)/6 = (104+104+106)/6 = 314/6 ≈ 52.3333
        state, r14 = book.submit_order(
            state,
            _make_order(Side.BUY, 53.00, 6.0, order_id=14, agent_id=A2, timestamp=800),
        )
        assert r14.accepted == True
        assert r14.traded == True
        assert r14.trade_qty == pytest.approx(6.0, abs=0.05)
        expected_vwap_o14 = 314.0 / 6.0
        assert r14.trade_price == pytest.approx(expected_vwap_o14, abs=0.01), \
            f"VWAP = (2*52+2*52+2*53)/6 = {expected_vwap_o14:.4f}"

        # No bids remain; O13 keeps 2 MW @ 53.00 on the ask side
        bids_final, asks_final = book.get_anonymous_lob(state)
        assert bids_final == [], "No bids should remain"
        assert len(asks_final) == 1, f"Only O13's remainder should rest, got {asks_final}"
        assert asks_final[0][0] == pytest.approx(53.00, abs=0.01)
        assert asks_final[0][1] == pytest.approx(2.0, abs=0.05), \
            "O13: 4 - 2 = 2 MW left @ 53.00"

        # =================================================================
        # PnL verification — zero-sum check
        # =================================================================

        # A1 (id=1): seller only. Sold: 3@52(T6) + 2@52(T10) + 2@53(T12) = 7 MW
        # Revenue: 3*52*0.25 + 2*52*0.25 + 2*53*0.25 = 39.00 + 26.00 + 26.50 = 91.50
        a1_trades_raw = book.get_agent_trades(state, A1)
        a1_pnl = _pnl_one_sided(a1_trades_raw, agent_id=A1, is_seller=True)
        assert a1_pnl == pytest.approx(91.50, abs=0.01), \
            "A1 sell revenue: 39.00 + 26.00 + 26.50 = 91.50"

        # A2 (id=2): buyer only (counterparty to A4's sells + buys from A1/A3)
        # Bought: 4@47(T7) + 6@46(T8) + 2@45(T9) + 2@52(T10) + 2@52(T11) + 2@53(T12)
        # Cost: 47+69+22.5+26+26+26.5 = 217.00  (O12 was killed: no trade)
        a2_trades_raw = book.get_agent_trades(state, A2)
        a2_pnl = _pnl_one_sided(a2_trades_raw, agent_id=A2, is_seller=False)
        assert a2_pnl == pytest.approx(-217.00, abs=0.01), \
            "A2 buy cost: 47+69+22.5+26+26+26.5 = 217.00"

        # A3 (id=3): seller only. Sold via icebergs.
        # 3@50(T1) + 3@50(T2) + 3@50(T3) + 2@51(T4) + 2@51.50(T5) + 2@52(T11) = 15 MW
        # Revenue: 37.50+37.50+37.50+25.50+25.75+26.00 = 189.75
        # (unchanged: O7's slice2 now sells inside O9 instead of O12)
        a3_trades_raw = book.get_agent_trades(state, A3)
        a3_pnl = _pnl_one_sided(a3_trades_raw, agent_id=A3, is_seller=True)
        assert a3_pnl == pytest.approx(189.75, abs=0.01), \
            "A3 sell revenue: 37.50+37.50+37.50+25.50+25.75+26.00 = 189.75"

        # A4 (id=4): buys and sells.
        # Buys: 3@50+3@50+3@50+2@51+2@51.50+3@52 = 16 MW
        #   (cost=37.50+37.50+37.50+25.50+25.75+39.00=202.75)
        # Sells: 4@47+6@46+2@45 = 12 MW (revenue=47+69+22.5=138.50)
        # PnL = 138.50 - 202.75 = -64.25
        # Since we can't separate buy/sell from get_agent_trades, use zero-sum:
        a4_trades_raw = book.get_agent_trades(state, A4)
        expected_a4_pnl = -(a1_pnl + a2_pnl + a3_pnl)
        assert expected_a4_pnl == pytest.approx(-64.25, abs=0.01), \
            "A4 PnL by zero-sum: -(91.50 + (-217.00) + 189.75) = -64.25"

        # Verify total trade count per agent
        assert len(a1_trades_raw) >= 2, \
            "A1 involved in at least 2 fills (O1 in O9 and O14, O13 in O14)"
        assert len(a2_trades_raw) >= 6, \
            "A2 involved in at least 6 fills"
        assert len(a3_trades_raw) >= 6, \
            "A3 involved in at least 6 fills (iceberg multi-order)"
        assert len(a4_trades_raw) >= 7, \
            "A4 involved in at least 7 fills (buys + sells)"


# ===========================================================================
# 2. test_iceberg_fok_ioc_interaction
#
# Focused test: iceberg resting liquidity interacting with FOK and IOC,
# specifically exercising the total-quantity FOK pre-check (Rule 1, M6.3)
# and multi-slice matching for all order types (Rule 2, M6.4).
#
# Domain rules (M6.4 update):
#   - FOK pre-check uses TOTAL iceberg quantity (visible + hidden) — Rule 1
#   - FOK uses multi-slice matching after passing pre-check — Rule 1
#   - NON/IOC also use multi-slice matching via regen loop (M6.4) — Rule 2
#   - IOC fills across slices, remainder cancelled after loop
#   - A single aggressive order can drain an entire iceberg
#
# Agents:
#   A1 (id=1) — iceberg seller
#   A2 (id=2) — FOK/IOC buyer
#   A3 (id=3) — limit seller (provides extra liquidity)
#
# Script (11 orders):
#   O1: A1 SELL 7 MW@50.00 (ICE, peak=2)         ts=100
#       Visible: 2@50. Hidden: 5.
#   O2: A3 SELL 3 MW@51.00 (NON)                  ts=110
#   O3: A3 SELL 2 MW@52.00 (NON)                  ts=120
#
#   LOB visible: 2@50, 3@51, 2@52 (total visible=7, true total=12)
#
#   O4: A2 BUY 3 MW@50.00 (FOK)                   ts=200
#       M6.3: total iceberg at <=50 = 7 MW >= 3. Passes.
#       Multi-slice: 2@50 (peak1) + regen peak2 (2@50) + 1@50 from peak2.
#       Trade: 3@50. O1: 7-3=4 MW remaining. Peak2: 1 MW visible, 3 hidden.
#
#   O5: A2 BUY 2 MW@50.00 (FOK)                   ts=300
#       M6.3: total at <=50 = 4 MW >= 2. Passes.
#       Multi-slice: 1@50 (peak2 rem) + regen peak3 (2@50) + 1@50 from peak3.
#       Trade: 2@50. O1: 4-2=2 MW remaining. Peak3: 1 MW visible, 1 hidden.
#
#   O6: A2 BUY 3 MW@50.00 (FOK)                   ts=400
#       Total at <=50 = 2 MW < 3. REJECTED.
#
#   O7: A2 BUY 3 MW@50.00 (IOC)                   ts=500
#       M6.4 multi-slice: fills 1@50 (peak3 remainder). Regen peak4:
#       min(2,1)=1@50. Crosses IOC remainder (2 MW). Fills 1@50 from peak4.
#       O1 exhausted (2-1-1=0). IOC: 2 MW filled, 1 MW cancelled.
#
#   O7b: A2 BUY 1 MW@50.00 (NON)                  ts=510
#       Iceberg exhausted. No crossing. 1 MW rests as bid@50.
#
#   O8: A2 BUY 4 MW@51.00 (FOK)                   ts=600
#       Available at <=51 = O2(3@51) = 3 MW. 3 < 4. REJECTED.
#
#   O9: A2 BUY 3 MW@51.00 (FOK)                   ts=700
#       Available at <=51 = 3 MW >= 3. Passes. Fills 3@51. O2 consumed.
#
#   O10: A2 BUY 5 MW@52.00 (IOC)                  ts=800
#       Available at <=52: O3(2@52) = 2 MW. IOC fills 2@52, cancels 3.
#       O3 consumed.
#
#   Book should be completely empty.
#
# PnL:
#   A1 sells: 3@50(O4) + 2@50(O5) + 2@50(O7, M6.4 multi-slice) = 7 MW@50
#     Revenue: 7*50*0.25 = 87.50
#   A2 buys: 3@50 + 2@50 + 2@50 + 3@51 + 2@52 = 12 MW
#     (O7b does not trade — iceberg exhausted, rests as bid)
#     Cost: 37.50 + 25.00 + 25.00 + 38.25 + 26.00 = 151.75
#   A3 sells: 3@51 + 2@52
#     Revenue: 3*51*0.25 + 2*52*0.25 = 38.25 + 26.00 = 64.25
#   Zero-sum: 87.50 + (-151.75) + 64.25 = 0.00
# ===========================================================================

class TestIcebergFokIocInteraction:
    """Focused test: iceberg resting liquidity with FOK/IOC aggressors.

    Exercises the total-quantity FOK pre-check (M6.3 Rule 1) and
    multi-slice matching for all order types (M6.4 Rule 2). All aggressive
    order types (FOK, IOC, NON) use multi-slice matching.
    """

    def test_iceberg_fok_ioc_interaction(self, book, empty_state):
        """Iceberg vs FOK/IOC interaction scenario.

        Hand-computed PnL:
          A1: +87.50  A2: -151.75  A3: +64.25
          Sum: 0.00 (zero-sum)
        """
        A1, A2, A3 = 1, 2, 3

        # --- Build resting book ---

        # O1: A1 SELL 7 MW@50 (ICE, peak=2)
        state, r1 = book.submit_order(
            empty_state,
            _make_order(Side.SELL, 50.00, 7.0, order_id=1, agent_id=A1, timestamp=100,
                        exec_restriction=ExecRestriction.ICE, peak_qty=2.0),
        )
        assert r1.accepted == True

        # O2: A3 SELL 3 MW@51 (NON)
        state, r2 = book.submit_order(
            state,
            _make_order(Side.SELL, 51.00, 3.0, order_id=2, agent_id=A3, timestamp=110),
        )
        assert r2.accepted == True

        # O3: A3 SELL 2 MW@52 (NON)
        state, r3 = book.submit_order(
            state,
            _make_order(Side.SELL, 52.00, 2.0, order_id=3, agent_id=A3, timestamp=120),
        )
        assert r3.accepted == True

        # LOB visible: 2@50, 3@51, 2@52
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 3
        assert asks[0] == pytest.approx((50.00, 2.0), abs=0.05), \
            "Iceberg peak 2 MW visible"
        assert asks[1] == pytest.approx((51.00, 3.0), abs=0.05)
        assert asks[2] == pytest.approx((52.00, 2.0), abs=0.05)

        # --- O4: FOK BUY 3 MW@50 — FILLS (M6.3: total=7 >= 3, multi-slice) ---
        # Multi-slice: 2 MW from peak1, regen peak2 (2 MW), 1 MW from peak2.
        state, r4 = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 3.0, order_id=4, agent_id=A2, timestamp=200,
                        exec_restriction=ExecRestriction.FOK),
        )
        assert r4.accepted == True, "FOK passes validation (accepted=True)"
        assert r4.traded == True, \
            "FOK 3@50: total iceberg=7 MW at <=50, pre-check passes (M6.3)"
        assert r4.trade_qty == pytest.approx(3.0, abs=0.05), \
            "FOK multi-slice fill: 2 (peak1) + 1 (peak2 partial) = 3 MW"
        assert r4.trade_price == pytest.approx(50.00, abs=0.01)

        # Iceberg: 7-3=4 MW remaining. Peak2 has 1 MW visible, 3 MW hidden.
        _, asks = book.get_anonymous_lob(state)
        assert asks[0] == pytest.approx((50.00, 1.0), abs=0.05), \
            "Peak2 partial: 2 MW slice minus 1 MW filled = 1 MW visible"

        # --- O5: FOK BUY 2 MW@50 — succeeds (M6.3: total=4 >= 2, multi-slice) ---
        # Multi-slice: 1 MW from peak2 remainder, regen peak3 (2 MW), 1 from peak3.
        state, r5 = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 2.0, order_id=5, agent_id=A2, timestamp=300,
                        exec_restriction=ExecRestriction.FOK),
        )
        assert r5.accepted == True
        assert r5.traded == True, \
            "FOK 2@50: total iceberg=4 MW, pre-check passes (M6.3)"
        assert r5.trade_qty == pytest.approx(2.0, abs=0.05)
        assert r5.trade_price == pytest.approx(50.00, abs=0.01)

        # O1: 4-2=2 MW remaining. Peak3 has 1 MW visible, 1 MW hidden.
        _, asks = book.get_anonymous_lob(state)
        assert asks[0][0] == pytest.approx(50.00, abs=0.01)
        assert asks[0][1] == pytest.approx(1.0, abs=0.05), \
            "Peak3 partial: 2 MW slice minus 1 MW filled = 1 MW visible"

        # --- O6: FOK BUY 3 MW@50 — REJECTED (total=2 < 3 needed) ---
        state, r6 = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 3.0, order_id=6, agent_id=A2, timestamp=400,
                        exec_restriction=ExecRestriction.FOK),
        )
        assert r6.accepted == True
        assert r6.traded == False, \
            "FOK 3@50: total iceberg=2 MW at <=50, insufficient for 3 MW"
        assert r6.trade_qty == pytest.approx(0.0, abs=0.05)

        # Verify iceberg still intact after rejected FOK
        _, asks = book.get_anonymous_lob(state)
        assert asks[0] == pytest.approx((50.00, 1.0), abs=0.05), \
            "Rejected FOK must not alter iceberg state (1 MW visible)"

        # --- O7: IOC BUY 3 MW@50 — multi-slice fill (2 MW, 1 cancelled) ---
        # M6.4: IOC uses multi-slice matching. Fills 1 MW from peak3 remainder.
        # Regen peak4: min(2, 1)=1 MW@50. Crosses resting IOC remainder (2 MW).
        # Fills 1 MW from peak4. O1 exhausted (2-1-1=0). IOC: 2 filled, 1 cancelled.
        state, r7 = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 3.0, order_id=7, agent_id=A2, timestamp=500,
                        exec_restriction=ExecRestriction.IOC),
        )
        assert r7.accepted == True
        assert r7.traded == True
        assert r7.trade_qty == pytest.approx(2.0, abs=0.05), \
            "IOC multi-slice (M6.4): fills 1 MW (peak3 rem) + 1 MW (peak4) = 2 MW"
        assert r7.trade_price == pytest.approx(50.00, abs=0.01)

        # IOC remainder (1 MW) must not be in book
        assert _is_absent(book.get_order(state, 7, Side.BUY)), \
            "IOC remainder (1 MW) must not rest in book"

        # O1 iceberg exhausted (7-3-2-2=0). Visible asks: 3@51, 2@52
        _, asks = book.get_anonymous_lob(state)
        assert len(asks) == 2, "Iceberg exhausted after IOC multi-slice, 2 ask levels remain"
        assert asks[0] == pytest.approx((51.00, 3.0), abs=0.05)
        assert asks[1] == pytest.approx((52.00, 2.0), abs=0.05)

        # --- O7b: NON BUY 1 MW@50 — iceberg exhausted, no crossing, rests ---
        # Iceberg O1 is fully consumed. NON enters book as resting bid@50.
        state, r7b = book.submit_order(
            state,
            _make_order(Side.BUY, 50.00, 1.0, order_id=70, agent_id=A2, timestamp=510),
        )
        assert r7b.accepted == True
        assert r7b.traded == False, \
            "Iceberg exhausted, no asks at <=50.00. NON rests as bid."

        # LOB: bid at 50 (O7b), asks at 51 and 52
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 1
        assert bids[0] == pytest.approx((50.00, 1.0), abs=0.05), \
            "O7b rests as bid: 1 MW@50"
        assert len(asks) == 2
        assert asks[0] == pytest.approx((51.00, 3.0), abs=0.05)
        assert asks[1] == pytest.approx((52.00, 2.0), abs=0.05)

        # --- O8: FOK BUY 4 MW@51 — REJECTED (visible=3 < 4 needed) ---
        state, r8 = book.submit_order(
            state,
            _make_order(Side.BUY, 51.00, 4.0, order_id=8, agent_id=A2, timestamp=600,
                        exec_restriction=ExecRestriction.FOK),
        )
        assert r8.accepted == True
        assert r8.traded == False, \
            "FOK 4@51: only 3 MW available at <=51 (O2 limit), must reject"

        # --- O9: FOK BUY 3 MW@51 — fills exactly ---
        state, r9 = book.submit_order(
            state,
            _make_order(Side.BUY, 51.00, 3.0, order_id=9, agent_id=A2, timestamp=700,
                        exec_restriction=ExecRestriction.FOK),
        )
        assert r9.accepted == True
        assert r9.traded == True
        assert r9.trade_qty == pytest.approx(3.0, abs=0.05)
        assert r9.trade_price == pytest.approx(51.00, abs=0.01)

        # O2 consumed. Asks: only O3 (2@52) remains. Bids: O7b (1@50).
        bids, asks = book.get_anonymous_lob(state)
        assert len(asks) == 1
        assert asks[0] == pytest.approx((52.00, 2.0), abs=0.05)
        assert len(bids) == 1, "O7b still resting as bid"
        assert bids[0] == pytest.approx((50.00, 1.0), abs=0.05)

        # --- O10: IOC BUY 5 MW@52 — partial fill (2 MW, cancel 3) ---
        state, r10 = book.submit_order(
            state,
            _make_order(Side.BUY, 52.00, 5.0, order_id=10, agent_id=A2, timestamp=800,
                        exec_restriction=ExecRestriction.IOC),
        )
        assert r10.accepted == True
        assert r10.traded == True
        assert r10.trade_qty == pytest.approx(2.0, abs=0.05), \
            "IOC: only 2 MW available at <=52"
        assert r10.trade_price == pytest.approx(52.00, abs=0.01)

        # All asks consumed. O7b (BUY 1@50) still rests as bid.
        bids, asks = book.get_anonymous_lob(state)
        assert len(bids) == 1, "O7b still resting as bid@50"
        assert bids[0] == pytest.approx((50.00, 1.0), abs=0.05)
        assert asks == [], "All asks consumed"

        # IOC remainder not in book
        assert _is_absent(book.get_order(state, 10, Side.BUY))

        # --- PnL verification ---
        # A1: sold 3@50(O4) + 2@50(O5) + 2@50(O7, M6.4 multi-slice) = 7@50
        # Revenue = 7*50*0.25 = 87.50
        a1_raw = book.get_agent_trades(state, A1)
        a1_pnl = _pnl_one_sided(a1_raw, agent_id=A1, is_seller=True)
        assert a1_pnl == pytest.approx(87.50, abs=0.01), \
            "A1: 7*50*0.25 = 87.50"

        # A3: sold 3@51(O9) + 2@52(O10). Revenue = 38.25 + 26.00 = 64.25
        a3_raw = book.get_agent_trades(state, A3)
        a3_pnl = _pnl_one_sided(a3_raw, agent_id=A3, is_seller=True)
        assert a3_pnl == pytest.approx(64.25, abs=0.01), \
            "A3: 3*51*0.25 + 2*52*0.25 = 38.25 + 26.00 = 64.25"

        # A2: bought 3@50(O4) + 2@50(O5) + 2@50(O7, M6.4) + 3@51(O9) + 2@52(O10) = 12 MW
        # O7b does NOT trade (iceberg exhausted, rests as bid)
        # Cost = 37.50 + 25.00 + 25.00 + 38.25 + 26.00 = 151.75
        a2_raw = book.get_agent_trades(state, A2)
        a2_pnl = _pnl_one_sided(a2_raw, agent_id=A2, is_seller=False)
        assert a2_pnl == pytest.approx(-151.75, abs=0.01), \
            "A2: -(37.50+25.00+25.00+38.25+26.00) = -151.75"

        # Zero-sum check
        assert a1_pnl + a2_pnl + a3_pnl == pytest.approx(0.0, abs=0.01), \
            "N-agent zero-sum: all trade values cancel out"

        # Verify trade counts
        assert len(a1_raw) >= 3, \
            "A1: at least 3 fills from iceberg slicing (O4, O5, O7 multi-slice)"
        assert len(a2_raw) >= 4, \
            "A2: fills from O4, O5, O7(multi-slice), O9, O10"
        assert len(a3_raw) == 2, "A3: exactly 2 fills (O2@51, O3@52)"
