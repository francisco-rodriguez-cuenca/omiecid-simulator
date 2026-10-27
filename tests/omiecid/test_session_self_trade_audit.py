"""Independent audit twin for Story REL-06 (Self-trade accounting).

Derived from ``docs/release/release-plan.md`` §2.1 ("Self-trade accounting"),
§4.4 ("Cash", "Self-trades"), the REL-06 card of
``docs/release/release-backlog.md`` and ``docs/omiecid/market-spec.md`` §6.6
(no self-trade prevention at XBID level: matching is pure price-time priority,
an own buy can match an own sell), §6.1-6.2 (price-time priority, maker
pricing) and §8.1/§8.4 (V = Q x p x d; the buy leg books -V, the sell leg +V).
Written black-box against the public interface; the locked contract file
``tests/omiecid/test_session_self_trade.py`` was not consulted.

Contract under audit:
  * When the participant's order matches its own resting order, BOTH legs are
    the participant's: the aggressive leg and the passive leg are attributed,
    so the self-matched quantity nets to zero inventory and zero cash.
  * ``state.agent_cash`` holds realized EUR (aggressive + passive, unscaled) and
    agrees with ``compute_agent_pnl`` over the book's trade tape rows involving
    the participant (the exact settlement).
  * Architecture ruling (2026-10-26): the per-step aggregated ``agent_trades``
    passive row is exact only on steps WITHOUT passive fills in both directions.
    ``compute_agent_pnl(agent_trades) == agent_cash`` is therefore asserted only
    in scenarios free of mixed-direction passive steps.

Tolerances. ``agent_cash`` / ``agent_inventory`` are float32 accumulators whose
legs are decoded by different float paths (book ``trade_qty`` vs qty ticks x
0.1 MW); the plan itself says exact settlement is ``compute_agent_pnl`` over the
tape. "Nets to zero" is read as zero to float32 rounding: |inventory| < 1e-4 MW
(1/1000 of the 0.1 MW qty tick) and |cash| < 1e-3 EUR (1/10 of a cent).
``compute_agent_pnl`` on its own (int64 tick space) is asserted EXACTLY where
both legs of a self-match sit on the tape (they must cancel to 0.0 exactly).

The book's 8-column trade tape is converted to the ``compute_agent_pnl``
layout ``[ts, price_offset, qty_tick, buyer, seller, contract]`` by mapping
each trade's passive / aggressive ORDER ID to the side the test itself
submitted (tape rows and the participant's stamped ``report.order_id``), so the
buyer / seller attribution does not depend on the engine's Q-sign convention.

Cost model: every scenario lives on ONE session that stacks several scenario
tapes (selected with ``state.replace(tape_idx=i)``, plan §4.4 "Choosing a
tape"), so the module compiles ``step_env`` once (plus one vmapped compile).
Tests that dispatch a compiled step carry ``@pytest.mark.slow`` (slow: shares a
jitted Session step compile, several seconds cold on CPU).
"""

from __future__ import annotations

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob.jaxob_constants import EMPTY_SLOT, TradesFeat
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.omie_session import (
    AGENT_TRADER_ID,
    AgentOrder,
    Session,
)
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from omiecid_simulator.omiecid.settlement import compute_agent_pnl

BUY = int(Side.BUY)
SELL = int(Side.SELL)
NON = int(ExecRestriction.NON)
IOC = int(ExecRestriction.IOC)
ICE = int(ExecRestriction.ICE)

D_H = 0.25           # 15-minute contract (market-spec §8.1)
BG_TID = 777         # background trader id on every scenario tape
N_ROWS = 8           # tape length; orders_per_step = 1 -> 8 steps per episode
SPAN = 100_000.0     # tape seconds mapped onto [0, GCT]: ts <= 10 stays at t=0
KEY = jax.random.PRNGKey(0)

INV_TOL = 1e-4       # MW
CASH_TOL = 1e-3      # EUR

_PC = OMIEProductConfig()


def _row(side, price, qty, oid, ts, er=NON):
    return [
        0, int(side),
        int(round((price - _PC.price_min) / _PC.price_tick)),
        int(round(qty / _PC.qty_tick)),
        int(oid), BG_TID, int(ts), int(er), 0, 0, 0, 0, 0,
    ]


_PAD = [0, 0, 0, 0, -1, 0, -1, NON, 0, 0, 0, 0, 0]


def _tape(rows_by_step):
    """rows_by_step: {step_index: row}; other steps are inert pad rows."""
    return [rows_by_step.get(i, list(_PAD)) for i in range(N_ROWS)]


# Background orders per scenario tape: oid -> side (for tape-to-pnl conversion).
T_EMPTY = 0          # no background at all
T_PARTIAL = 1        # step1: bg SELL 6 @ 50.5
T_BG_BETTER = 2      # step0: bg SELL 5 @ 49.5 (better price than own 50)
T_OWN_FIRST = 3      # step1: bg SELL 5 @ 50 arrives AFTER own SELL 5 @ 50
T_BG_FIRST = 4       # step0: bg SELL 5 @ 50 arrives BEFORE own SELL 5 @ 50
T_MIXED = 5          # step2: bg SELL 3 @ 48 IOC (hits own bid)
T_PASSIVE = 6        # step1: bg SELL 4 @ 49 (hits own bid; control)
T_AGGR = 7           # step0: bg SELL 5 @ 51 (own BUY lifts it; control)
T_REMAINDER = 8      # step2: bg SELL 6 @ 50 (hits own resting remainder)

_BG_ROWS = {
    T_EMPTY: {},
    T_PARTIAL: {1: (SELL, 50.5, 6.0, 11, 10)},
    T_BG_BETTER: {0: (SELL, 49.5, 5.0, 21, 5)},
    T_OWN_FIRST: {1: (SELL, 50.0, 5.0, 31, 10)},
    T_BG_FIRST: {0: (SELL, 50.0, 5.0, 41, 5)},
    T_MIXED: {2: (SELL, 48.0, 3.0, 51, 10, IOC)},
    T_PASSIVE: {1: (SELL, 49.0, 4.0, 61, 10)},
    T_AGGR: {0: (SELL, 51.0, 5.0, 71, 5)},
    T_REMAINDER: {2: (SELL, 50.0, 6.0, 81, 10)},
}

BG_SIDES = {
    spec[3]: spec[0] for rows in _BG_ROWS.values() for spec in rows.values()
}


def _build_tapes():
    tapes = []
    for idx in range(len(_BG_ROWS)):
        rows = {s: _row(*spec) for s, spec in _BG_ROWS[idx].items()}
        tapes.append(_tape(rows))
    return np.asarray(tapes, dtype=np.int32)


@pytest.fixture(scope="module")
def session():
    return Session({
        "tapes": _build_tapes(),
        "orders_per_step": 1,
        "tape_seconds_span": SPAN,
    })


class _Run:
    """Drive one scenario; record per-step reports and own order sides."""

    def __init__(self, session, tape_idx):
        self.s = session
        self.state = session.reset(KEY).replace(tape_idx=jnp.int32(tape_idx))
        self.reports = []
        self.sides = dict(BG_SIDES)

    def step(self, order=None):
        self.state, rep, done = self.s.step(KEY, self.state, order)
        assert not bool(done), "scenario must stay before gate closure"
        if bool(rep.submitted):
            self.sides[int(rep.order_id)] = int(order.side)
        self.reports.append(rep)
        return rep

    @property
    def inv(self):
        return float(self.state.agent_inventory[0])

    @property
    def cash(self):
        return float(self.state.agent_cash[0])

    def book_trades(self):
        t = np.asarray(self.state.book.lob.trades)
        return t[t[:, TradesFeat.P.value] != EMPTY_SLOT]

    def book_tape_pnl_rows(self):
        """Book trade tape -> compute_agent_pnl (M, 6) rows, side by order id."""
        out = []
        for r in self.book_trades():
            p_oid = int(r[TradesFeat.PASS_OID.value])
            a_oid = int(r[TradesFeat.AGRS_OID.value])
            p_side, a_side = self.sides[p_oid], self.sides[a_oid]
            assert p_side != a_side, "a trade must join a buy and a sell"
            p_tid = int(r[TradesFeat.PASS_TID.value])
            a_tid = int(r[TradesFeat.AGRS_TID.value])
            buyer, seller = (p_tid, a_tid) if p_side == BUY else (a_tid, p_tid)
            out.append([
                int(r[TradesFeat.SEC.value]), int(r[TradesFeat.P.value]),
                abs(int(r[TradesFeat.Q.value])), buyer, seller, 0,
            ])
        return np.asarray(out, dtype=np.int32).reshape(-1, 6)

    def participant_rows(self):
        rows = self.book_tape_pnl_rows()
        mask = (rows[:, 3] == AGENT_TRADER_ID) | (rows[:, 4] == AGENT_TRADER_ID)
        return rows[mask]

    def tape_pnl(self):
        return float(compute_agent_pnl(
            jnp.asarray(self.participant_rows()), jnp.int32(AGENT_TRADER_ID)))

    def agent_trades_pnl(self):
        return float(compute_agent_pnl(
            jnp.asarray(self.state.agent_trades).reshape(-1, 6),
            jnp.int32(AGENT_TRADER_ID)))


def _o(side, price, qty, **kw):
    return AgentOrder.create(side, price, qty, **kw)


# ---------------------------------------------------------------------------
# 1. Full self-match: own resting order consumed by own aggressive order
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestFullSelfMatch:
    """slow: shares a jitted Session step compile."""

    def _run(self, session, rest_side, aggr_side):
        run = _Run(session, T_EMPTY)
        run.step(_o(rest_side, 50.0, 10.0))
        rep = run.step(_o(aggr_side, 50.0, 10.0))
        return run, rep

    def test_own_sell_crossed_by_own_buy_matches_fully(self, session):
        run, rep = self._run(session, SELL, BUY)
        assert bool(rep.submit.traded)
        assert float(rep.submit.trade_qty) == pytest.approx(10.0, abs=INV_TOL)
        assert float(rep.submit.trade_price) == pytest.approx(50.0, abs=1e-4)

    def test_passive_leg_of_self_match_is_attributed(self, session):
        run, rep = self._run(session, SELL, BUY)
        assert float(rep.passive_qty_signed) == pytest.approx(-10.0, abs=INV_TOL)
        assert float(rep.passive_value_eur) == pytest.approx(
            10.0 * 50.0 * D_H, abs=CASH_TOL)

    def test_step_inventory_and_cash_deltas_net_to_zero(self, session):
        _, rep = self._run(session, SELL, BUY)
        assert abs(float(rep.inventory_delta)) < INV_TOL
        assert abs(float(rep.cash_delta_eur)) < CASH_TOL

    def test_cumulative_inventory_and_cash_net_to_zero(self, session):
        run, _ = self._run(session, SELL, BUY)
        assert abs(run.inv) < INV_TOL
        assert abs(run.cash) < CASH_TOL

    def test_mirror_own_buy_crossed_by_own_sell_nets_to_zero(self, session):
        run, rep = self._run(session, BUY, SELL)
        assert float(rep.passive_qty_signed) == pytest.approx(10.0, abs=INV_TOL)
        assert float(rep.passive_value_eur) == pytest.approx(
            -10.0 * 50.0 * D_H, abs=CASH_TOL)
        assert abs(run.inv) < INV_TOL
        assert abs(run.cash) < CASH_TOL

    def test_book_is_empty_after_full_self_match(self, session):
        run, _ = self._run(session, SELL, BUY)
        tob = session.top_of_book(run.state)
        assert not bool(tob.has_bid)
        assert not bool(tob.has_ask)

    def test_book_tape_records_participant_on_both_legs(self, session):
        run, _ = self._run(session, SELL, BUY)
        trades = run.book_trades()
        assert trades.shape[0] >= 1
        assert np.all(trades[:, TradesFeat.PASS_TID.value] == AGENT_TRADER_ID)
        assert np.all(trades[:, TradesFeat.AGRS_TID.value] == AGENT_TRADER_ID)

    def test_book_tape_pnl_is_exactly_zero_and_matches_cash(self, session):
        run, _ = self._run(session, SELL, BUY)
        assert run.tape_pnl() == 0.0
        assert run.cash == pytest.approx(run.tape_pnl(), abs=CASH_TOL)

    def test_agent_trades_pnl_is_exactly_zero_and_matches_cash(self, session):
        run, _ = self._run(session, SELL, BUY)
        assert run.agent_trades_pnl() == 0.0
        assert run.cash == pytest.approx(run.agent_trades_pnl(), abs=CASH_TOL)

    def test_step_trade_rows_carry_both_legs_and_cancel(self, session):
        _, rep = self._run(session, SELL, BUY)
        rows = np.asarray(rep.trade_rows)
        # aggressive leg: participant buys; passive leg: participant sells.
        assert rows[0, 2] == 100 and rows[0, 3] == AGENT_TRADER_ID
        assert rows[1, 2] == 100 and rows[1, 4] == AGENT_TRADER_ID
        assert float(compute_agent_pnl(
            jnp.asarray(rows), jnp.int32(AGENT_TRADER_ID))) == 0.0


# ---------------------------------------------------------------------------
# 2. Partial self-match
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestPartialSelfMatch:
    """slow: shares a jitted Session step compile."""

    def test_remainder_fills_against_background(self, session):
        # own SELL 4 @ 50 rests; bg SELL 6 @ 50.5 arrives; own BUY 10 @ 51
        # takes own 4 @ 50 (better price) then bg 6 @ 50.5.
        run = _Run(session, T_PARTIAL)
        run.step(_o(SELL, 50.0, 4.0))
        rep = run.step(_o(BUY, 51.0, 10.0))
        assert float(rep.submit.trade_qty) == pytest.approx(10.0, abs=INV_TOL)
        assert float(rep.passive_qty_signed) == pytest.approx(-4.0, abs=INV_TOL)
        # only the background leg survives the netting
        assert run.inv == pytest.approx(6.0, abs=INV_TOL)
        assert run.cash == pytest.approx(-6.0 * 50.5 * D_H, abs=CASH_TOL)
        assert float(rep.inventory_delta) == pytest.approx(6.0, abs=INV_TOL)
        assert float(rep.cash_delta_eur) == pytest.approx(
            -6.0 * 50.5 * D_H, abs=CASH_TOL)

    def test_remainder_against_background_cash_matches_tapes(self, session):
        run = _Run(session, T_PARTIAL)
        run.step(_o(SELL, 50.0, 4.0))
        run.step(_o(BUY, 51.0, 10.0))
        assert run.tape_pnl() == pytest.approx(-6.0 * 50.5 * D_H, abs=1e-6)
        assert run.cash == pytest.approx(run.tape_pnl(), abs=CASH_TOL)
        assert run.cash == pytest.approx(run.agent_trades_pnl(), abs=CASH_TOL)

    def test_remainder_rests_then_background_hits_it(self, session):
        # own SELL 4 @ 50 rests; own BUY 10 @ 50 self-matches 4, rests 6 bid;
        # step2 bg SELL 6 @ 50 hits the resting remainder (passive buy).
        run = _Run(session, T_REMAINDER)
        run.step(_o(SELL, 50.0, 4.0))
        rep1 = run.step(_o(BUY, 50.0, 10.0))
        assert abs(float(rep1.inventory_delta)) < INV_TOL
        assert abs(float(rep1.cash_delta_eur)) < CASH_TOL
        tob = session.top_of_book(run.state)
        assert bool(tob.has_bid) and not bool(tob.has_ask)
        assert float(tob.p_bid) == pytest.approx(50.0, abs=1e-4)
        assert float(tob.q_bid) == pytest.approx(6.0, abs=INV_TOL)
        rep2 = run.step(None)
        assert float(rep2.passive_qty_signed) == pytest.approx(6.0, abs=INV_TOL)
        assert run.inv == pytest.approx(6.0, abs=INV_TOL)
        assert run.cash == pytest.approx(-6.0 * 50.0 * D_H, abs=CASH_TOL)
        assert run.cash == pytest.approx(run.tape_pnl(), abs=CASH_TOL)
        assert run.cash == pytest.approx(run.agent_trades_pnl(), abs=CASH_TOL)


# ---------------------------------------------------------------------------
# 3. Own iceberg and own second price level
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestSelfMatchStructure:
    """slow: shares a jitted Session step compile."""

    def test_self_match_against_own_iceberg_nets_to_zero(self, session):
        run = _Run(session, T_EMPTY)
        rep0 = run.step(_o(SELL, 50.0, 10.0, exec_restriction=ICE, peak_qty=2.0))
        assert bool(rep0.submit.accepted)
        rep = run.step(_o(BUY, 50.0, 10.0))
        filled = float(rep.submit.trade_qty)
        assert filled > 0.0
        # whatever the iceberg yielded, every filled MW is attributed both ways
        assert float(rep.passive_qty_signed) == pytest.approx(-filled, abs=INV_TOL)
        assert abs(run.inv) < INV_TOL
        assert abs(run.cash) < CASH_TOL
        assert run.tape_pnl() == 0.0
        assert run.agent_trades_pnl() == 0.0

    def test_iceberg_slices_all_carry_participant_on_both_legs(self, session):
        run = _Run(session, T_EMPTY)
        run.step(_o(SELL, 50.0, 10.0, exec_restriction=ICE, peak_qty=2.0))
        run.step(_o(BUY, 50.0, 10.0))
        trades = run.book_trades()
        assert trades.shape[0] >= 1
        assert np.all(trades[:, TradesFeat.PASS_TID.value] == AGENT_TRADER_ID)
        assert np.all(trades[:, TradesFeat.AGRS_TID.value] == AGENT_TRADER_ID)
        total = np.abs(trades[:, TradesFeat.Q.value]).sum() * _PC.qty_tick
        assert total == pytest.approx(
            float(run.reports[-1].submit.trade_qty), abs=INV_TOL)

    def test_self_match_across_two_own_price_levels(self, session):
        run = _Run(session, T_EMPTY)
        run.step(_o(SELL, 50.0, 5.0))
        run.step(_o(SELL, 51.0, 5.0))
        rep = run.step(_o(BUY, 51.0, 10.0))
        assert float(rep.submit.trade_qty) == pytest.approx(10.0, abs=INV_TOL)
        assert float(rep.passive_qty_signed) == pytest.approx(-10.0, abs=INV_TOL)
        assert float(rep.passive_value_eur) == pytest.approx(
            (5.0 * 50.0 + 5.0 * 51.0) * D_H, abs=CASH_TOL)
        assert abs(run.inv) < INV_TOL
        assert abs(run.cash) < CASH_TOL
        assert run.tape_pnl() == 0.0
        # single-direction passive step, tick-aligned average: exact per ruling
        assert run.agent_trades_pnl() == 0.0
        tob = session.top_of_book(run.state)
        assert not bool(tob.has_ask)


# ---------------------------------------------------------------------------
# 4. Price-time priority decides which order is hit (market-spec §6.6, §6.1)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestPriorityWithOwnOrders:
    """slow: shares a jitted Session step compile."""

    def test_better_priced_background_is_hit_before_own_order(self, session):
        # bg SELL 5 @ 49.5 and own SELL 5 @ 50 rest; own BUY 5 @ 50 hits bg.
        run = _Run(session, T_BG_BETTER)
        run.step(_o(SELL, 50.0, 5.0))
        rep = run.step(_o(BUY, 50.0, 5.0))
        assert float(rep.submit.trade_price) == pytest.approx(49.5, abs=1e-4)
        assert float(rep.passive_qty_signed) == 0.0
        assert run.inv == pytest.approx(5.0, abs=INV_TOL)
        assert run.cash == pytest.approx(-5.0 * 49.5 * D_H, abs=CASH_TOL)
        trades = run.book_trades()
        assert np.all(trades[:, TradesFeat.PASS_TID.value] == BG_TID)
        # own ask still rests at 50
        tob = session.top_of_book(run.state)
        assert float(tob.p_ask) == pytest.approx(50.0, abs=1e-4)
        assert float(tob.q_ask) == pytest.approx(5.0, abs=INV_TOL)
        assert run.cash == pytest.approx(run.tape_pnl(), abs=CASH_TOL)
        assert run.cash == pytest.approx(run.agent_trades_pnl(), abs=CASH_TOL)

    def test_own_order_then_self_match_after_background_is_gone(self, session):
        run = _Run(session, T_BG_BETTER)
        run.step(_o(SELL, 50.0, 5.0))
        run.step(_o(BUY, 50.0, 5.0))
        rep = run.step(_o(BUY, 50.0, 5.0))      # now hits own ask
        assert float(rep.passive_qty_signed) == pytest.approx(-5.0, abs=INV_TOL)
        assert abs(float(rep.inventory_delta)) < INV_TOL
        assert abs(float(rep.cash_delta_eur)) < CASH_TOL
        assert run.inv == pytest.approx(5.0, abs=INV_TOL)
        assert run.cash == pytest.approx(-5.0 * 49.5 * D_H, abs=CASH_TOL)
        assert run.cash == pytest.approx(run.tape_pnl(), abs=CASH_TOL)
        assert run.cash == pytest.approx(run.agent_trades_pnl(), abs=CASH_TOL)

    def test_earlier_own_order_at_same_price_is_hit_first(self, session):
        # own SELL 5 @ 50 at step0; bg SELL 5 @ 50 at step1 (later arrival);
        # own BUY 5 @ 50 at step1 -> time priority hits the own order.
        run = _Run(session, T_OWN_FIRST)
        run.step(_o(SELL, 50.0, 5.0))
        rep = run.step(_o(BUY, 50.0, 5.0))
        assert float(rep.passive_qty_signed) == pytest.approx(-5.0, abs=INV_TOL)
        assert abs(run.inv) < INV_TOL
        assert abs(run.cash) < CASH_TOL
        tob = session.top_of_book(run.state)
        assert float(tob.q_ask) == pytest.approx(5.0, abs=INV_TOL)  # bg left
        assert run.tape_pnl() == 0.0
        assert run.agent_trades_pnl() == 0.0

    def test_earlier_background_at_same_price_is_hit_first(self, session):
        # bg SELL 5 @ 50 replayed at step0 BEFORE own SELL 5 @ 50 (step0);
        # own BUY 5 @ 50 at step1 -> hits the background, no self-match.
        run = _Run(session, T_BG_FIRST)
        run.step(_o(SELL, 50.0, 5.0))
        rep = run.step(_o(BUY, 50.0, 5.0))
        assert float(rep.passive_qty_signed) == 0.0
        assert run.inv == pytest.approx(5.0, abs=INV_TOL)
        assert run.cash == pytest.approx(-5.0 * 50.0 * D_H, abs=CASH_TOL)
        trades = run.book_trades()
        assert np.all(trades[:, TradesFeat.PASS_TID.value] == BG_TID)
        assert run.cash == pytest.approx(run.tape_pnl(), abs=CASH_TOL)


# ---------------------------------------------------------------------------
# 5. Mixed-direction passive step (architecture ruling: agent_trades excluded)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestMixedDirectionPassiveStep:
    """slow: shares a jitted Session step compile."""

    def _run(self, session):
        # own BUY 3 @ 48 rests (s0); own SELL 3 @ 52 rests (s1); s2: bg SELL
        # 3 @ 48 IOC hits own bid (passive buy), then own BUY 3 @ 52 lifts own
        # ask (self-match: aggressive buy + passive sell).
        run = _Run(session, T_MIXED)
        run.step(_o(BUY, 48.0, 3.0))
        run.step(_o(SELL, 52.0, 3.0))
        rep = run.step(_o(BUY, 52.0, 3.0))
        return run, rep

    def test_inventory_and_cash_fold_both_passive_directions(self, session):
        run, rep = self._run(session)
        assert abs(float(rep.passive_qty_signed)) < INV_TOL   # +3 - 3
        assert float(rep.passive_value_eur) == pytest.approx(
            (-3.0 * 48.0 + 3.0 * 52.0) * D_H, abs=CASH_TOL)
        assert run.inv == pytest.approx(3.0, abs=INV_TOL)
        assert run.cash == pytest.approx(-3.0 * 48.0 * D_H, abs=CASH_TOL)

    def test_cash_matches_book_tape_settlement(self, session):
        run, _ = self._run(session)
        assert run.tape_pnl() == pytest.approx(-3.0 * 48.0 * D_H, abs=1e-6)
        assert run.cash == pytest.approx(run.tape_pnl(), abs=CASH_TOL)


# ---------------------------------------------------------------------------
# 6. Controls: no self-match
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestControlsNoSelfMatch:
    """slow: shares a jitted Session step compile."""

    def test_aggressive_only_fill_against_background(self, session):
        run = _Run(session, T_AGGR)
        rep = run.step(_o(BUY, 51.0, 5.0))
        assert float(rep.passive_qty_signed) == 0.0
        assert float(rep.passive_value_eur) == 0.0
        assert run.inv == pytest.approx(5.0, abs=INV_TOL)
        assert run.cash == pytest.approx(-5.0 * 51.0 * D_H, abs=CASH_TOL)
        assert run.cash == pytest.approx(run.tape_pnl(), abs=CASH_TOL)
        assert run.cash == pytest.approx(run.agent_trades_pnl(), abs=CASH_TOL)

    def test_passive_only_fill_by_background(self, session):
        run = _Run(session, T_PASSIVE)
        run.step(_o(BUY, 49.0, 4.0))
        rep = run.step(None)
        assert not bool(rep.submitted)
        assert float(rep.passive_qty_signed) == pytest.approx(4.0, abs=INV_TOL)
        assert run.inv == pytest.approx(4.0, abs=INV_TOL)
        assert run.cash == pytest.approx(-4.0 * 49.0 * D_H, abs=CASH_TOL)
        assert run.cash == pytest.approx(run.tape_pnl(), abs=CASH_TOL)
        assert run.cash == pytest.approx(run.agent_trades_pnl(), abs=CASH_TOL)

    def test_resting_own_orders_on_both_sides_without_cross_do_not_trade(
            self, session):
        run = _Run(session, T_EMPTY)
        run.step(_o(BUY, 49.0, 5.0))
        rep = run.step(_o(SELL, 51.0, 5.0))
        assert not bool(rep.submit.traded)
        assert float(rep.passive_qty_signed) == 0.0
        assert run.inv == 0.0 and run.cash == 0.0
        assert run.book_trades().shape[0] == 0
        assert np.all(np.asarray(run.state.agent_trades) == 0)

    def test_empty_steps_leave_accounts_untouched(self, session):
        run = _Run(session, T_EMPTY)
        for _ in range(3):
            rep = run.step(None)
            assert float(rep.inventory_delta) == 0.0
            assert float(rep.cash_delta_eur) == 0.0
        assert run.inv == 0.0 and run.cash == 0.0


# ---------------------------------------------------------------------------
# 7. vmap: the self-match accounting holds batched
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_self_match_nets_to_zero_under_vmap(session):
    """slow: separate vmapped step_env compile.

    Runs under an outer ``enable_x64`` (the engine's int64 fill accumulator,
    OQ-18): the plain-vmap lowering failure is a pre-existing REL-05 audit
    finding (``test_session_audit.py``), not the subject of this story.
    """
    with jax.experimental.enable_x64():
        _vmapped_self_match(session)


def _vmapped_self_match(session):
    keys = jax.random.split(KEY, 2)
    states = jax.vmap(session.reset)(keys)
    states = states.replace(tape_idx=jnp.array([T_EMPTY, T_AGGR], jnp.int32))
    vstep = jax.vmap(session.step_env, in_axes=(None, 0, 0))

    def batch(*orders):
        return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *orders)

    # env 0: own SELL rests then own BUY self-matches; env 1: plain buy fill.
    states, _, _ = vstep(KEY, states, batch(_o(SELL, 50.0, 10.0),
                                            _o(BUY, 51.0, 5.0)))
    states, reps, _ = vstep(KEY, states, batch(_o(BUY, 50.0, 10.0),
                                               AgentOrder.none()))
    assert float(reps.passive_qty_signed[0]) == pytest.approx(-10.0, abs=INV_TOL)
    assert abs(float(states.agent_inventory[0, 0])) < INV_TOL
    assert abs(float(states.agent_cash[0, 0])) < CASH_TOL
    assert float(states.agent_inventory[1, 0]) == pytest.approx(5.0, abs=INV_TOL)
    assert float(states.agent_cash[1, 0]) == pytest.approx(
        -5.0 * 51.0 * D_H, abs=CASH_TOL)
