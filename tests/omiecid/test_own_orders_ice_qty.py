"""Locked behavioural-contract tests for Story OMIE-OWN-ORDERS-ICE-QTY-1.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-OWN-ORDERS-ICE-QTY-1 —
``get_agent_orders`` shows the owner each iceberg's full remaining quantity
(visible + hidden)" (certification fix pass, S6 / plan D5; card as amended at
v5.200).

Rule (market-spec §9.2 "Own orders": "Agent's own active orders (showing full
iceberg quantity)"; OMIE Rules Annex §4.4: "the agent who entered the bid will
also be able to see the total quantity"). ``ContractBook.get_agent_orders`` is
the §9.2 surface (domain-analyst ruling 2026-10-06). For each of the owner's
resting rows it must report

    quantity = visible slice + the iceberg's hidden remainder

and a NON row (no iceberg) is unchanged. Anonymity is untouched: a non-owner
never sees another agent's orders, and the anonymous LOB still shows only the
visible peak. Price, side, OID, agent and timestamp are unchanged; the order
is still a plain ``make_order`` record (no exec-restriction / peak / ppd fields
are added).

Contract (Test-first contract row, numbering preserved):
  (1) ``test_c13_seed96_full_remaining`` — C13 seed-96 replay, qty ticks:
      203 -> {82783: 989}, 547 -> {53226: 177}, 731 -> {63633: 40, 92707: 5},
      930 -> {}.                                     RED before the fix (174, 18).
  (2) ``test_fresh_iceberg_shows_total`` — SELL ICE 5.0 / peak 1.0 -> 5.0
      (plus a BUY mirror with other constants).       RED before the fix.
  (3) ``test_after_partial_fill_visible_plus_hidden`` — BUY ICE 4.0 / peak 1.0,
      then SELL 0.4 -> 3.6 (plus a SELL mirror, a post-regeneration case and
      ICEwPI drifted-slice cases).                    RED before the fix.
  (4) ``test_non_orders_unchanged`` (guard) — NON rows report their resting
      quantity.                                       GREEN before and after.
  (5) ``test_never_more_than_owner_may_see`` (guard) — a non-owner's view has
      none of the owner's orders; ``get_anonymous_lob`` shows only the peak.
                                                      GREEN before and after.
  (6) ``test_other_fields_unchanged`` (guard) — price (incl. an ICEwPI drifted
      slice), side, OID, agent, timestamp equal the values measured on main.
                                                      GREEN before and after.

Incidental constants are varied (both sides, OIDs of 1-6 digits, prices off
round numbers, several agents). Units: quantities are compared in 0.1 MW qty
ticks (``round(mw / 0.1)``) so float noise in the decode cannot mask a defect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import jax.numpy as jnp
import numpy as np
import pytest
from pytest import approx

from omiecid_simulator.jaxomiecid.omie_data_loader import _row_to_order_traced
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order


# Every test here runs the jitted ``_submit_order`` path (T1.1 marker lint).
pytestmark = pytest.mark.jit


REPO_ROOT = Path(__file__).resolve().parents[2]
# The ONE fixture-path constant (OMIE-CERT-HARNESS-2 may move the fixtures to
# tests/omiecid/fixtures/cert/; whichever story merges second updates this line
# only — not a re-baseline).
FIXTURE_DIR = REPO_ROOT / "notebooks" / "certification" / "fixtures"
C13_SEED96 = "C13_seed96.npz"

QTY_TICK_MW = 0.1


def _ticks(mw: float) -> int:
    return int(round(mw / QTY_TICK_MW))


# ---------------------------------------------------------------------------
# Book construction
# ---------------------------------------------------------------------------
CID = 23


def _new_book(cid: int = CID) -> ContractBook:
    contract = Contract(contract_id=cid, delivery_start_minutes=555,
                        delivery_end_minutes=570, gct_minutes=495,
                        status=ContractStatus.OPEN)
    return ContractBook(contract, OMIEProductConfig(), OMIEMarketConfig())


def _order(side, price, qty, oid, agent, ts, ex=ExecRestriction.NON,
           peak=0.0, ppd=0.0):
    return make_order(contract_id=CID, side=side, price=price, quantity=qty,
                      order_id=oid, agent_id=agent, timestamp=ts,
                      exec_restriction=ex, peak_qty=peak, ppd=ppd)


def _submit_all(book: ContractBook, orders):
    state = book.init()
    for o in orders:
        state, res = book.submit_order(state, o)
        assert bool(res.accepted), f"order {o.order_id} rejected"
    return state


def _own_qty_ticks(book, state, agent: int) -> Dict[int, int]:
    """{oid: qty ticks} of ``get_agent_orders`` for ``agent`` (one row per oid)."""
    out: Dict[int, int] = {}
    for o in book.get_agent_orders(state, agent):
        oid = int(o.order_id)
        assert oid not in out, f"oid {oid} listed twice for agent {agent}"
        out[oid] = _ticks(float(o.quantity))
    return out


def _lob_ticks(book, state):
    bids, asks = book.get_anonymous_lob(state)
    return ([(round(p, 2), _ticks(q)) for p, q in bids],
            [(round(p, 2), _ticks(q)) for p, q in asks])


# ---------------------------------------------------------------------------
# C13 seed-96 replay (module-scoped: one replay shared by the assertions)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def c13():
    tape = np.load(FIXTURE_DIR / C13_SEED96)["tape"]
    rows = [r for r in tape if int(r[4]) > 0]
    assert len(rows) == 7
    book = _new_book(int(rows[0][0]))
    pc = OMIEProductConfig()
    state = book.init()
    for r in rows:
        state, res = book.submit_order(state, _row_to_order_traced(jnp.asarray(r), pc))
        assert bool(res.accepted), f"row {r.tolist()} rejected"
    return book, state


C13_EXPECTED = {
    203: {82783: 989},
    547: {53226: 177},
    731: {63633: 40, 92707: 5},
    930: {},
}


# ===========================================================================
# (1) C13 seed 96 — owner sees each iceberg's full remaining quantity
# ===========================================================================
class TestC13Seed96:

    def test_c13_seed96_full_remaining(self, c13):
        """Card (1): every agent sees its full remaining quantity (qty ticks)."""
        book, state = c13
        got = {a: _own_qty_ticks(book, state, a) for a in C13_EXPECTED}
        assert got == C13_EXPECTED

    @pytest.mark.parametrize("agent,oid,full", [(203, 82783, 989), (547, 53226, 177)])
    def test_c13_seed96_iceberg_owner_sees_total(self, c13, agent, oid, full):
        """Card (1), per iceberg: 82783 = 174 visible + 815 hidden = 989;
        53226 = 18 visible + 159 hidden = 177 (main shows 174 and 18)."""
        book, state = c13
        assert _own_qty_ticks(book, state, agent).get(oid) == full

    def test_c13_seed96_anonymous_lob_still_visible_only(self, c13):
        """Card (5) on the C13 book: the LOB view shows only peaks (1.8, 17.4)."""
        book, state = c13
        bids, asks = _lob_ticks(book, state)
        assert bids == [(248.95, 5)]
        assert asks == [(249.05, 18), (249.11, 174), (249.13, 40)]


# ===========================================================================
# (2) Fresh iceberg — owner sees the total
# ===========================================================================
class TestFreshIceberg:

    def test_fresh_iceberg_shows_total(self):
        """Card (2): SELL ICE 5.0 / peak 1.0 -> owner sees 5.0."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.SELL, 50.0, 5.0, 1, 200, 100, ExecRestriction.ICE, 1.0),
        ])
        orders = book.get_agent_orders(state, 200)
        assert len(orders) == 1
        assert _ticks(orders[0].quantity) == 50
        assert orders[0].quantity == approx(5.0)

    def test_fresh_buy_iceberg_shows_total(self):
        """Card (2) BUY mirror, other constants: BUY ICE 7.3 / peak 0.6 -> 7.3."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.BUY, 38.47, 7.3, 640281, 17, 41, ExecRestriction.ICE, 0.6),
        ])
        assert _own_qty_ticks(book, state, 17) == {640281: 73}

    def test_two_icebergs_same_owner_each_full(self):
        """Each of the owner's icebergs carries its own hidden remainder
        (BUY 3.1/0.4 and SELL 2.2/0.7, plus a non-crossing SELL 1.5/0.5)."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.BUY, 44.81, 3.1, 52, 9, 3, ExecRestriction.ICE, 0.4),
            _order(Side.SELL, 45.93, 2.2, 7714, 9, 5, ExecRestriction.ICE, 0.7),
            _order(Side.SELL, 46.02, 1.5, 309, 9, 6, ExecRestriction.ICE, 0.5),
        ])
        assert _own_qty_ticks(book, state, 9) == {52: 31, 7714: 22, 309: 15}

    def test_hidden_not_attributed_to_other_agents_iceberg(self):
        """Two owners' icebergs on the same side and price: each owner's view
        holds only its own iceberg's total (4.4 and 1.9)."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.SELL, 52.66, 4.4, 811, 31, 10, ExecRestriction.ICE, 1.1),
            _order(Side.SELL, 52.66, 1.9, 9046, 77, 12, ExecRestriction.ICE, 0.3),
        ])
        assert _own_qty_ticks(book, state, 31) == {811: 44}
        assert _own_qty_ticks(book, state, 77) == {9046: 19}


# ===========================================================================
# (3) After fills — visible + hidden
# ===========================================================================
class TestAfterFills:

    def test_after_partial_fill_visible_plus_hidden(self):
        """Card (3): BUY ICE 4.0 / peak 1.0, then SELL 0.4 -> owner sees 3.6."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.BUY, 50.0, 4.0, 1, 200, 100, ExecRestriction.ICE, 1.0),
            _order(Side.SELL, 50.0, 0.4, 2, 300, 200),
        ])
        orders = book.get_agent_orders(state, 200)
        assert len(orders) == 1
        assert _ticks(orders[0].quantity) == 36
        assert orders[0].quantity == approx(3.6)

    def test_after_partial_fill_sell_mirror(self):
        """SELL ICE 2.9 / peak 0.8 @ 61.04, BUY 0.3 -> owner sees 2.6."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.SELL, 61.04, 2.9, 33157, 4, 7, ExecRestriction.ICE, 0.8),
            _order(Side.BUY, 61.20, 0.3, 8, 5, 9),
        ])
        assert _own_qty_ticks(book, state, 4) == {33157: 26}

    def test_after_slice_regeneration_visible_plus_hidden(self):
        """BUY ICE 4.0 / peak 1.0 @ 49.37, SELL 1.4 @ 49.30: one full slice plus
        0.4 of the next -> 2.6 remaining (0.6 visible + 2.0 hidden)."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.BUY, 49.37, 4.0, 4410, 61, 20, ExecRestriction.ICE, 1.0),
            _order(Side.SELL, 49.30, 1.4, 92, 62, 25),
        ])
        assert _own_qty_ticks(book, state, 61) == {4410: 26}

    def test_icewpi_sell_after_slice_fill_shows_remaining(self):
        """SELL ICEwPI 3.0 / peak 1.0 / ppd +0.50 @ 50.00, BUY 1.0 @ 50.00:
        the drifted slice rests at 50.50 and the owner sees 2.0."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.SELL, 50.0, 3.0, 1, 200, 100, ExecRestriction.ICE, 1.0, 0.50),
            _order(Side.BUY, 50.0, 1.0, 10, 300, 200),
        ])
        orders = book.get_agent_orders(state, 200)
        assert len(orders) == 1
        assert orders[0].price == approx(50.50)
        assert _ticks(orders[0].quantity) == 20

    def test_icewpi_buy_after_slice_fill_shows_remaining(self):
        """BUY ICEwPI 2.0 / peak 0.8 / ppd -0.07 @ 47.22, SELL 0.8 @ 47.22:
        the drifted slice rests at 47.15 and the owner sees 1.2."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.BUY, 47.22, 2.0, 5503, 12, 30, ExecRestriction.ICE, 0.8, -0.07),
            _order(Side.SELL, 47.22, 0.8, 6, 13, 31),
        ])
        orders = book.get_agent_orders(state, 12)
        assert len(orders) == 1
        assert orders[0].price == approx(47.15)
        assert _ticks(orders[0].quantity) == 12

    def test_last_slice_no_hidden_shows_visible(self):
        """Guard: hidden exhausted (SELL ICE 2.0 / 1.0, BUY 1.3) -> the last
        slice's 0.7 is the whole remainder.          GREEN before and after."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.SELL, 55.55, 2.0, 77, 8, 2, ExecRestriction.ICE, 1.0),
            _order(Side.BUY, 55.55, 1.3, 78, 3, 4),
        ])
        assert _own_qty_ticks(book, state, 8) == {77: 7}

    def test_fully_filled_iceberg_not_listed(self):
        """Guard: an iceberg filled in full leaves nothing in the owner's view.
                                                     GREEN before and after."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.BUY, 42.18, 1.5, 120, 44, 2, ExecRestriction.ICE, 0.5),
            _order(Side.SELL, 42.00, 1.5, 121, 45, 3),
        ])
        assert _own_qty_ticks(book, state, 44) == {}

    def test_cancelled_iceberg_not_listed(self):
        """Guard: a cancelled iceberg (visible and hidden) is not listed.
                                                     GREEN before and after."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.SELL, 58.08, 3.3, 2718, 70, 6, ExecRestriction.ICE, 1.2),
        ])
        state = book.cancel_order(state, 2718, Side.SELL)
        assert _own_qty_ticks(book, state, 70) == {}


# ===========================================================================
# (4) NON rows unchanged (guard)
# ===========================================================================
class TestNonOrdersUnchanged:

    def test_non_orders_unchanged(self):
        """Card (4): NON rows report their resting quantity, both sides,
        including a partially filled one.            GREEN before and after."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.BUY, 40.13, 2.5, 3, 15, 1),
            _order(Side.SELL, 63.71, 0.7, 48120, 15, 2),
            _order(Side.SELL, 39.99, 0.9, 5, 16, 3),   # fills 0.9 of oid 3
        ])
        assert _own_qty_ticks(book, state, 15) == {3: 16, 48120: 7}

    def test_non_row_beside_owner_iceberg_unchanged(self):
        """The owner's NON row keeps its own quantity next to its iceberg
        (NON 1.2 -> 12 ticks; the iceberg's hidden is not added to it)."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.BUY, 41.07, 3.6, 601, 22, 1, ExecRestriction.ICE, 0.9),
            _order(Side.BUY, 41.07, 1.2, 602, 22, 2),
        ])
        assert _own_qty_ticks(book, state, 22)[602] == 12

    def test_same_oid_opposite_side_non_not_inflated(self):
        """Guard (Step-2 R2): the hidden remainder is matched by OID and side.
        A BUY NON 0.4 sharing OID 9 with the owner's SELL iceberg reports 0.4.
                                                     GREEN before and after."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.SELL, 57.40, 4.0, 9, 26, 1, ExecRestriction.ICE, 1.0),
            _order(Side.BUY, 51.90, 0.4, 9, 26, 2),
        ])
        buys = [o for o in book.get_agent_orders(state, 26) if o.side == Side.BUY]
        assert len(buys) == 1
        assert _ticks(buys[0].quantity) == 4


# ===========================================================================
# (5) Never more than the owner may see (guard)
# ===========================================================================
class TestNeverMoreThanOwnerMaySee:

    def test_never_more_than_owner_may_see(self):
        """Card (5): a non-owner's view contains none of the owner's orders,
        and the anonymous LOB shows only the visible peak.
                                                     GREEN before and after."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.SELL, 50.0, 5.0, 1, 200, 100, ExecRestriction.ICE, 1.0),
            _order(Side.BUY, 48.6, 4.2, 35, 200, 101, ExecRestriction.ICE, 0.7),
            _order(Side.BUY, 48.1, 0.6, 2, 300, 102),
        ])
        other = book.get_agent_orders(state, 300)
        assert {int(o.order_id) for o in other} == {2}
        assert all(int(o.agent_id) == 300 for o in other)
        assert book.get_agent_orders(state, 999) == []
        bids, asks = _lob_ticks(book, state)
        assert bids == [(48.6, 7), (48.1, 6)]
        assert asks == [(50.0, 10)]

    def test_owner_view_returns_only_owner_agent_id(self):
        """Every row in an owner's view carries that owner's agent id."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.SELL, 53.21, 2.4, 404, 5, 1, ExecRestriction.ICE, 0.6),
            _order(Side.SELL, 53.21, 1.1, 405, 6, 2, ExecRestriction.ICE, 0.2),
        ])
        for agent in (5, 6):
            assert all(int(o.agent_id) == agent
                       for o in book.get_agent_orders(state, agent))


# ===========================================================================
# (6) Other fields unchanged (guard) — pins measured on main @ 0ef39e0
# ===========================================================================
# (oid, side, price, agent, timestamp, contract) of the drifted slice, measured
# on main @ 0ef39e0 (the regenerated slice carried a fresh counter stamp, 17).
# Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence (27)): the slice is
# stamped at its activation event, the second of the BUY that exhausted the
# previous slice (16). The RL observation does not read stored stamps.
PIN_ICEWPI_DRIFTED = (40917, 1, 61.5, 58, 16, CID)


class TestOtherFieldsUnchanged:

    @staticmethod
    def _fields(o):
        return (int(o.order_id), int(o.side), round(float(o.price), 2),
                int(o.agent_id), int(o.timestamp), int(o.contract_id))

    def test_other_fields_unchanged(self, c13):
        """Card (6): price, side, OID, agent, timestamp, contract on C13 seed 96."""
        book, state = c13
        got = sorted(self._fields(o) for a in C13_EXPECTED
                     for o in book.get_agent_orders(state, a))
        assert got == sorted([
            (82783, 1, 249.11, 203, 1, 0),
            (53226, 1, 249.05, 547, 4, 0),
            (92707, 0, 248.95, 731, 2, 0),
            (63633, 1, 249.13, 731, 3, 0),
        ])

    def test_other_fields_unchanged_icewpi_drifted_slice(self):
        """Card (6): an ICEwPI drifted slice keeps its drifted price (61.50),
        side, OID, agent and timestamp."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.SELL, 61.37, 2.7, 40917, 58, 14, ExecRestriction.ICE, 0.9, 0.13),
            _order(Side.BUY, 61.37, 0.9, 3, 59, 16),
        ])
        orders = book.get_agent_orders(state, 58)
        assert len(orders) == 1
        assert self._fields(orders[0]) == PIN_ICEWPI_DRIFTED

    def test_order_is_plain_record(self):
        """Card: still built with ``make_order``; no exec-restriction, peak or
        ppd fields are added to the own-orders record."""
        book = _new_book()
        state = _submit_all(book, [
            _order(Side.BUY, 36.66, 2.8, 1907, 3, 8, ExecRestriction.ICE, 0.7, -0.05),
        ])
        (o,) = book.get_agent_orders(state, 3)
        assert o.exec_restriction == ExecRestriction.NON
        assert float(o.peak_qty) == 0.0
        assert float(o.ppd) == 0.0

