"""Audit tests for OMIE-OWN-ORDERS-ICE-QTY-1 (S6, D5).

Independent post-implementation verification of
``ContractBook.get_agent_orders``: the owner's own-orders view shows each
iceberg's FULL remaining quantity (visible slice + hidden remainder), while
anonymous views keep showing only the visible peak.

Derived independently from:
- docs/omiecid/market-spec.md §9.2 "Own orders" ("Agent's own active orders
  (showing full iceberg quantity)"); OMIE Rules Annex §4.4 (the owner "will
  also be able to see the total quantity"); anonymity of the public book.
- docs/omiecid/agile-plan.md, card OMIE-OWN-ORDERS-ICE-QTY-1 (v5.200).
- omiecid_simulator/omiecid/contract_book.py (public interface only).

Focus (edge cases not named in the card's test-first contract):
- conservation: owner quantity == submitted - traded, through a sequence of
  partial fills that cross slice boundaries;
- the last slice (no hidden remainder left) and full exhaustion;
- cancel / GCT expiry drop the iceberg from the owner's view;
- OID reuse after an iceberg is gone (no stale hidden quantity leaks in);
- same OID on opposite sides (hidden quantity is keyed by side too);
- incoming iceberg that partially crosses before resting (Annex §4.4 sweep);
- several icebergs of one owner, on both sides, at once;
- ICEwPI: drifted price + full remaining quantity together;
- read-only: the view does not mutate state and is idempotent;
- anonymity: anonymous L2 volume never includes hidden quantity, and other
  agents never see the owner's orders.

All tests are synthetic, CPU, no jax.jit, well under the 1 s budget.
"""

import pytest
from pytest import approx

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order


CID = 10
OWNER = 200
OTHER = 300
THIRD = 400


@pytest.fixture(scope="module")
def open_contract():
    return Contract(
        contract_id=CID,
        delivery_start_minutes=150,
        delivery_end_minutes=165,
        gct_minutes=90,
        status=ContractStatus.OPEN,
    )


@pytest.fixture
def book(open_contract, market_config):
    return ContractBook(open_contract, OMIEProductConfig(), market_config)


@pytest.fixture
def state(book):
    return book.init()


def _order(side, price, qty, oid, agent, ts, restriction=ExecRestriction.NON,
           peak=0.0, ppd=0.0):
    return make_order(
        contract_id=CID, side=side, price=price, quantity=qty, order_id=oid,
        agent_id=agent, timestamp=ts, exec_restriction=restriction,
        peak_qty=peak, ppd=ppd,
    )


def _ice(side, price, qty, oid, agent=OWNER, ts=100, peak=1.0, ppd=0.0):
    restriction = ExecRestriction.ICE
    return _order(side, price, qty, oid, agent, ts, restriction, peak, ppd)


def _by_oid(orders):
    return {int(o.order_id): o for o in orders}


def _lob_volume(levels):
    """Sum of quantities in an anonymous L2 side (list of (price, qty)-likes)."""
    total = 0.0
    for lvl in levels:
        qty = lvl[1] if not hasattr(lvl, "quantity") else lvl.quantity
        total += float(qty)
    return total


# ===========================================================================
# 1. Conservation: owner quantity == submitted - traded
# ===========================================================================

class TestConservationThroughFills:

    def test_sequence_of_fills_crossing_slice_boundaries(self, book, state):
        """SELL ICE 5.0 peak 1.0; buys 0.3, 0.7, 1.5, 0.2: owner sees 5 - cum."""
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 5.0, oid=1))
        cum = 0.0
        for k, q in enumerate([0.3, 0.7, 1.5, 0.2]):
            state, _ = book.submit_order(
                state, _order(Side.BUY, 50.0, q, oid=100 + k, agent=OTHER,
                              ts=200 + k))
            cum += q
            orders = book.get_agent_orders(state, OWNER)
            assert len(orders) == 1, f"after fill {k}"
            assert orders[0].quantity == approx(5.0 - cum, abs=1e-6), (
                f"after cumulative fill {cum}")

    def test_fill_spanning_two_slices_in_one_call(self, book, state):
        """BUY ICE 4.0 peak 1.0; one SELL 1.5 eats a slice and half the next."""
        state, _ = book.submit_order(state, _ice(Side.BUY, 50.0, 4.0, oid=7))
        state, _ = book.submit_order(
            state, _order(Side.SELL, 50.0, 1.5, oid=8, agent=OTHER, ts=200))
        orders = book.get_agent_orders(state, OWNER)
        assert len(orders) == 1
        assert orders[0].quantity == approx(2.5, abs=1e-6)

    def test_owner_quantity_never_below_visible_peak_in_lob(self, book, state):
        """The owner's view is >= what the anonymous book shows at the level."""
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 3.0, oid=1,
                                                 peak=0.5))
        state, _ = book.submit_order(
            state, _order(Side.BUY, 50.0, 0.2, oid=2, agent=OTHER, ts=200))
        _, asks = book.get_anonymous_lob(state)
        own = book.get_agent_orders(state, OWNER)
        assert own[0].quantity >= _lob_volume(asks) - 1e-9
        assert own[0].quantity == approx(2.8, abs=1e-6)

    def test_quantity_is_on_qty_tick_grid(self, book, state):
        """Reported quantity is an integer multiple of qty_tick (0.1 MW)."""
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 7.3, oid=1,
                                                 peak=1.1))
        state, _ = book.submit_order(
            state, _order(Side.BUY, 50.0, 1.4, oid=2, agent=OTHER, ts=200))
        q = book.get_agent_orders(state, OWNER)[0].quantity
        assert q == approx(5.9, abs=1e-6)
        assert abs(q / 0.1 - round(q / 0.1)) < 1e-6


# ===========================================================================
# 2. Last slice and exhaustion
# ===========================================================================

class TestLastSliceAndExhaustion:

    def test_last_slice_reports_visible_only(self, book, state):
        """With no hidden remainder left, owner sees the resting slice only."""
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 3.0, oid=1))
        state, _ = book.submit_order(
            state, _order(Side.BUY, 50.0, 2.4, oid=2, agent=OTHER, ts=200))
        orders = book.get_agent_orders(state, OWNER)
        assert len(orders) == 1
        assert orders[0].quantity == approx(0.6, abs=1e-6)

    def test_fully_exhausted_iceberg_disappears(self, book, state):
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 3.0, oid=1))
        state, _ = book.submit_order(
            state, _order(Side.BUY, 50.0, 3.0, oid=2, agent=OTHER, ts=200))
        assert book.get_agent_orders(state, OWNER) == []

    def test_peak_equal_total_reports_total(self, book, state):
        """Degenerate iceberg (peak == total): no hidden part, total shown."""
        state, _ = book.submit_order(state, _ice(Side.BUY, 50.0, 2.0, oid=1,
                                                 peak=2.0))
        orders = book.get_agent_orders(state, OWNER)
        assert len(orders) == 1
        assert orders[0].quantity == approx(2.0, abs=1e-6)


# ===========================================================================
# 3. Removal: cancel and GCT expiry
# ===========================================================================

class TestRemoval:

    def test_cancelled_iceberg_not_in_owner_view(self, book, state):
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 5.0, oid=1))
        state = book.cancel_order(state, 1, Side.SELL)
        assert book.get_agent_orders(state, OWNER) == []

    def test_gct_expiry_clears_owner_view(self, book, state):
        state, _ = book.submit_order(state, _ice(Side.BUY, 50.0, 5.0, oid=1))
        state = book.expire_at_gct(state)
        assert book.get_agent_orders(state, OWNER) == []


# ===========================================================================
# 4. Keying: no stale or cross-side hidden quantity
# ===========================================================================

class TestHiddenKeying:

    def test_oid_reused_by_non_after_iceberg_exhausted(self, book, state):
        """A NON reusing a dead iceberg's OID shows its own quantity only."""
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 3.0, oid=1))
        state, _ = book.submit_order(
            state, _order(Side.BUY, 50.0, 3.0, oid=2, agent=OTHER, ts=200))
        state, _ = book.submit_order(
            state, _order(Side.SELL, 51.0, 2.0, oid=1, agent=OWNER, ts=300))
        orders = book.get_agent_orders(state, OWNER)
        assert len(orders) == 1
        assert orders[0].quantity == approx(2.0, abs=1e-6)

    def test_oid_reused_by_non_after_iceberg_cancelled(self, book, state):
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 5.0, oid=1))
        state = book.cancel_order(state, 1, Side.SELL)
        state, _ = book.submit_order(
            state, _order(Side.SELL, 52.0, 1.3, oid=1, agent=OWNER, ts=300))
        orders = book.get_agent_orders(state, OWNER)
        assert len(orders) == 1
        assert orders[0].quantity == approx(1.3, abs=1e-6)

    def test_same_oid_opposite_side_non_gets_no_hidden(self, book, state):
        """ICE SELL oid=1 (owner) and NON BUY oid=1 (other agent) coexist.

        The BUY row must not inherit the SELL iceberg's hidden remainder.
        """
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 5.0, oid=1))
        state, _ = book.submit_order(
            state, _order(Side.BUY, 40.0, 1.2, oid=1, agent=OTHER, ts=200))
        other = book.get_agent_orders(state, OTHER)
        own = book.get_agent_orders(state, OWNER)
        assert [o.quantity for o in other] == [approx(1.2, abs=1e-6)]
        assert [o.quantity for o in own] == [approx(5.0, abs=1e-6)]
        assert other[0].side == Side.BUY and own[0].side == Side.SELL

    def test_same_owner_icebergs_both_sides_and_non(self, book, state):
        """One owner: BUY ICE, SELL ICE, NON SELL — each reported correctly."""
        state, _ = book.submit_order(state, _ice(Side.BUY, 45.0, 4.0, oid=1,
                                                 peak=0.5, ts=100))
        state, _ = book.submit_order(state, _ice(Side.SELL, 55.0, 6.0, oid=2,
                                                 peak=1.5, ts=101))
        state, _ = book.submit_order(
            state, _order(Side.SELL, 56.0, 0.7, oid=3, agent=OWNER, ts=102))
        got = _by_oid(book.get_agent_orders(state, OWNER))
        assert set(got) == {1, 2, 3}
        assert got[1].quantity == approx(4.0, abs=1e-6)
        assert got[2].quantity == approx(6.0, abs=1e-6)
        assert got[3].quantity == approx(0.7, abs=1e-6)
        assert got[1].side == Side.BUY
        assert got[2].side == Side.SELL and got[3].side == Side.SELL

    def test_two_owners_icebergs_same_level_independent(self, book, state):
        """Two agents' icebergs at the same price: each sees only its own total."""
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 5.0, oid=1,
                                                 agent=OWNER, ts=100))
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 3.0, oid=2,
                                                 agent=THIRD, ts=101))
        # fill only the first (time-priority) peak partially
        state, _ = book.submit_order(
            state, _order(Side.BUY, 50.0, 0.4, oid=3, agent=OTHER, ts=200))
        a = book.get_agent_orders(state, OWNER)
        b = book.get_agent_orders(state, THIRD)
        assert [int(o.order_id) for o in a] == [1]
        assert [int(o.order_id) for o in b] == [2]
        assert a[0].quantity == approx(4.6, abs=1e-6)
        assert b[0].quantity == approx(3.0, abs=1e-6)


# ===========================================================================
# 5. Incoming iceberg that partially crosses (Annex §4.4 total sweep)
# ===========================================================================

class TestIncomingIcebergRemainder:

    def test_incoming_ice_partially_crossing_shows_remainder(self, book, state):
        """BUY NON 1.5 rests; SELL ICE 5.0 peak 1.0 at the same price.

        1.5 trades; the owner sees the 3.5 that remains (visible + hidden).
        """
        state, _ = book.submit_order(
            state, _order(Side.BUY, 50.0, 1.5, oid=1, agent=OTHER, ts=100))
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 5.0, oid=2,
                                                 ts=200))
        orders = book.get_agent_orders(state, OWNER)
        assert len(orders) == 1
        assert orders[0].quantity == approx(3.5, abs=1e-6)
        _, asks = book.get_anonymous_lob(state)
        assert _lob_volume(asks) <= 1.0 + 1e-6  # public sees at most a peak


# ===========================================================================
# 6. ICEwPI: drifted price and full quantity together
# ===========================================================================

class TestICEwPI:

    def test_icewpi_after_two_slices_price_and_quantity(self, book, state):
        """SELL ICEwPI 4.0 peak 1.0 ppd +0.5 at 50: two slices filled."""
        state, _ = book.submit_order(
            state, _ice(Side.SELL, 50.0, 4.0, oid=1, peak=1.0, ppd=0.5))
        state, _ = book.submit_order(
            state, _order(Side.BUY, 50.0, 1.0, oid=2, agent=OTHER, ts=200))
        state, _ = book.submit_order(
            state, _order(Side.BUY, 50.5, 1.0, oid=3, agent=OTHER, ts=201))
        orders = book.get_agent_orders(state, OWNER)
        assert len(orders) == 1
        assert orders[0].price == approx(51.0, abs=1e-6)
        assert orders[0].quantity == approx(2.0, abs=1e-6)
        assert orders[0].side == Side.SELL
        assert int(orders[0].order_id) == 1
        assert int(orders[0].agent_id) == OWNER


# ===========================================================================
# 7. Read-only, anonymity
# ===========================================================================

class TestReadOnlyAndAnonymity:

    def test_view_is_idempotent_and_does_not_mutate_lob(self, book, state):
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 5.0, oid=1))
        lob_before = book.get_anonymous_lob(state)
        first = book.get_agent_orders(state, OWNER)
        second = book.get_agent_orders(state, OWNER)
        assert [(o.order_id, o.quantity, o.price) for o in first] == \
            [(o.order_id, o.quantity, o.price) for o in second]
        assert book.get_anonymous_lob(state) == lob_before

    def test_anonymous_lob_never_includes_hidden(self, book, state):
        """Public volume = sum of visible peaks, never the owners' totals."""
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 5.0, oid=1,
                                                 peak=1.0, ts=100))
        state, _ = book.submit_order(state, _ice(Side.SELL, 50.0, 4.0, oid=2,
                                                 agent=THIRD, peak=0.5,
                                                 ts=101))
        _, asks = book.get_anonymous_lob(state)
        assert _lob_volume(asks) == approx(1.5, abs=1e-6)
        owner_total = sum(o.quantity for o in book.get_agent_orders(state, OWNER))
        third_total = sum(o.quantity for o in book.get_agent_orders(state, THIRD))
        assert owner_total + third_total == approx(9.0, abs=1e-6)

    def test_non_owner_and_unknown_agent_see_nothing(self, book, state):
        state, _ = book.submit_order(state, _ice(Side.BUY, 50.0, 5.0, oid=1))
        assert book.get_agent_orders(state, OTHER) == []
        assert book.get_agent_orders(state, 987654) == []

    def test_empty_book_returns_empty_list(self, book, state):
        assert book.get_agent_orders(state, OWNER) == []
