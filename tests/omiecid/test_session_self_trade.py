"""Locked tests for story REL-06 -- self-trade accounting in ``Session``.

Source of truth: ``docs/release/release-backlog.md`` card "Story REL-06";
``docs/release/release-plan.md`` §2.1 "Self-trade accounting", §4.4 "Cash" and
"Self-trades", §5.2 (self-trade row), §5.3 (``test_session_self_trade.py``
row); ``docs/omiecid/market-spec.md`` §6.6 (XBID/M7 has NO self-trade
prevention: an agent's buy can match its own sell, pure price-time priority).

Requirement under test (plan §4.4): "OMIE has no self-trade prevention. The
passive side of a participant's self-match is attributed, so inventory and cash
net to zero." ``agent_cash`` holds realized EUR (aggressive plus passive,
unscaled); exact settlement is ``compute_agent_pnl`` over the trade tape.

Acceptance criteria locked here:
  AC1  An own resting BUY crossed by the participant's own later SELL nets to
       zero inventory and zero cash; both legs are attributed and the
       ``FillReport`` legs are consistent.
  AC2  ``compute_agent_pnl`` over the session's trade tape equals
       ``state.agent_cash`` in a session holding a self-match plus ordinary
       (aggressive and passive) fills -- and both equal the analytic value in
       which the self-match contributes exactly zero.
  Plus a control (no self-match -> only the aggressive leg is attributed,
  unchanged behaviour) and a partial self-match (own resting order fully
  consumed by an own aggressive order whose remainder fills against a
  background order).

Interpretation choices:
  I1  "Both legs attributed" is read through the public ``FillReport``: on a
      self-match step the passive leg appears in ``passive_qty_signed`` (+MW
      on the participant's resting BUY) and ``passive_value_eur`` (realized
      EUR, + on a sell / - on a buy, per the ``FillReport`` docstring), and
      ``inventory_delta`` / ``cash_delta_eur`` are aggressive + passive. This
      follows the card's fix description ("the passive-fill diff also covers
      the participant's own submission").
  I2  Trade-row consistency is checked layout-agnostically: summing over the
      step's ``trade_rows`` (+qty where the participant is buyer, -qty where it
      is seller) must equal ``inventory_delta``, and ``compute_agent_pnl`` over
      those rows must equal ``cash_delta_eur``. The test does not pin whether
      the self-match is recorded as two rows or one buyer==seller row.
  I3  AC2 alone would be satisfied on the current tip (tape and cash both omit
      the passive leg consistently), so it also pins the analytic value. That
      is what makes it red for the right reason.
  I4  Matching itself (no STP) is checked via ``top_of_book``: the
      participant's resting bid is consumed by its own order.

Device policy: CPU. Every class compiles one shared Session (``@pytest.mark
.slow``: one jitted Session compile, > 1 s); all tests share that config.

Run::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python3 -m pytest \
        tests/omiecid/test_session_self_trade.py -n0 -q
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.omie_session import (
    AGENT_TRADER_ID,
    AgentOrder,
    Session,
)
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    Status,
    Validity,
)
from omiecid_simulator.omiecid.settlement import compute_agent_pnl


pytestmark = pytest.mark.slow  # slow: one jitted Session compile (> 1 s)

# ---------------------------------------------------------------------------
# Small synthetic world (one tape, one compile)
# ---------------------------------------------------------------------------
_PC = OMIEProductConfig()
_MC = OMIEMarketConfig(book_depth=10, nTrades=16, max_icebergs=16)
_OPS = 4
_KEY = jax.random.PRNGKey(0)
_D = float(_PC.delivery_duration_hours)       # 0.25 h

BUY = int(Side.BUY)
SELL = int(Side.SELL)
NON = int(ExecRestriction.NON)
IOC = int(ExecRestriction.IOC)

_BG = 777
_MW = 1e-4
_EUR = 1e-3
_PAD = [0, 0, 0, 0, -1, 0, -1, NON, int(Validity.GFS), 0,
        int(Status.ACTIVE), 0, 0]


def _pint(price: float) -> int:
    return int(round((price - _PC.price_min) / _PC.price_tick))


def _qint(qty: float) -> int:
    return int(round(qty / _PC.qty_tick))


def _r(side, price, qty, oid, ts, er=NON):
    return [0, int(side), _pint(price), _qint(qty), int(oid), _BG, int(ts),
            int(er), int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0]


# 24 rows = 6 steps of 4.
#   step 0: background bids BUY 40.00 x5 (101), BUY 49.00 x5 (103) and ask
#           SELL 60.00 x5 (102).
#   steps 1-3: no background flow.
#   step 4: background BUY IOC 59.00 x2 (104) -- hits a participant SELL
#           resting at <= 59.00 if any, else is killed (best ask 60.00).
#   step 5: a far BUY at the max timestamp, so ``done`` fires on step 5.
_TAPE = (
    [_r(BUY, 40.0, 5.0, 101, 10), _r(SELL, 60.0, 5.0, 102, 20),
     _r(BUY, 49.0, 5.0, 103, 30), _PAD]
    + [_PAD] * 4 + [_PAD] * 4 + [_PAD] * 4
    + [_r(BUY, 59.0, 2.0, 104, 500, IOC), _PAD, _PAD, _PAD]
    + [_r(BUY, 1.0, 0.1, 199, 1000), _PAD, _PAD, _PAD]
)
assert len(_TAPE) == 24
_TAPES = np.asarray([_TAPE], dtype=np.int32)


@functools.lru_cache(maxsize=None)
def _session():
    return Session({"tapes": _TAPES, "orders_per_step": _OPS,
                    "market_config": _MC})


def _step(state, order=None):
    return _session().step(_KEY, state, order)


def _start():
    return _session().reset(_KEY)


def _rest_buy_50x3():
    """BUY 50.00 x3 NON: rests as best bid (best ask 60.00)."""
    return AgentOrder.create(side=BUY, price=50.0, quantity=3.0)


def _f(x) -> float:
    return float(np.asarray(x))


def _rows_inventory(rows) -> float:
    """Signed MW of the participant over trade rows (+buyer / -seller)."""
    rows = np.asarray(rows).reshape(-1, 6)
    q = rows[:, 2].astype(np.int64)
    sign = ((rows[:, 3] == AGENT_TRADER_ID).astype(np.int64)
            - (rows[:, 4] == AGENT_TRADER_ID).astype(np.int64))
    return float(np.sum(sign * q)) * _PC.qty_tick


def _rows_pnl(rows) -> float:
    rows = jnp.asarray(np.asarray(rows).reshape(-1, 6), dtype=jnp.int32)
    return _f(compute_agent_pnl(rows, jnp.int32(AGENT_TRADER_ID)))


# ---------------------------------------------------------------------------
# Scenario runners (each returns the per-step reports and the final state)
# ---------------------------------------------------------------------------
def _run_full_self_match():
    """Step 0: own resting BUY 50 x3. Step 1: own SELL 50 x3 crosses it."""
    s = _start()
    s, r0, _ = _step(s, _rest_buy_50x3())
    s, r1, _ = _step(s, AgentOrder.create(side=SELL, price=50.0, quantity=3.0))
    return s, r0, r1


def _run_partial_self_match():
    """Step 0: own resting BUY 50 x3. Step 1: own SELL IOC 49 x5 takes the
    own 3 MW at 50.00 first, then 2 MW of background bid 103 at 49.00."""
    s = _start()
    s, r0, _ = _step(s, _rest_buy_50x3())
    s, r1, _ = _step(s, AgentOrder.create(side=SELL, price=49.0, quantity=5.0,
                                          exec_restriction=IOC))
    return s, r0, r1


def _run_mixed_session():
    """Self-match plus ordinary fills, played to ``done``.

    step 0: own resting BUY 50 x3          (no fill)
    step 1: own SELL 50 x3                 (self-match: nets to zero)
    step 2: own BUY IOC 60 x1              (ordinary aggressive: +1 MW, -15.0)
    step 3: own SELL 59 x2 NON             (rests as best ask)
    step 4: background BUY IOC 59 x2       (ordinary passive: -2 MW, +29.5)
    step 5: no order                       (done)
    """
    s = _start()
    orders = [
        _rest_buy_50x3(),
        AgentOrder.create(side=SELL, price=50.0, quantity=3.0),
        AgentOrder.create(side=BUY, price=60.0, quantity=1.0,
                          exec_restriction=IOC),
        AgentOrder.create(side=SELL, price=59.0, quantity=2.0),
        None,
        None,
    ]
    reports = []
    done = False
    for o in orders:
        s, r, done = _step(s, o)
        reports.append(r)
    return s, reports, bool(np.asarray(done))


# ---------------------------------------------------------------------------
# AC1 -- full self-match nets to zero
# ---------------------------------------------------------------------------
class TestFullSelfMatch:

    def test_own_sell_matches_own_resting_buy_no_self_trade_prevention(self):
        """market-spec §6.6: the own SELL trades against the own bid."""
        s, r0, r1 = _run_full_self_match()
        assert not bool(r0.submit.traded)
        assert bool(r1.submit.traded)
        assert _f(r1.submit.trade_qty) == pytest.approx(3.0, abs=_MW)
        assert _f(r1.submit.trade_price) == pytest.approx(50.0, abs=1e-4)
        tob = _session().top_of_book(s)
        # The own 50.00 bid is consumed; the background 49.00 bid is best.
        assert _f(tob.p_bid) == pytest.approx(49.0, abs=1e-4)
        assert _f(tob.q_bid) == pytest.approx(5.0, abs=_MW)

    def test_self_match_nets_to_zero_inventory(self):
        s, _r0, _r1 = _run_full_self_match()
        assert _f(s.agent_inventory[0]) == pytest.approx(0.0, abs=_MW)

    def test_self_match_nets_to_zero_cash(self):
        s, _r0, _r1 = _run_full_self_match()
        assert _f(s.agent_cash[0]) == pytest.approx(0.0, abs=_EUR)

    def test_self_match_step_report_deltas_are_zero(self):
        _s, _r0, r1 = _run_full_self_match()
        assert _f(r1.inventory_delta) == pytest.approx(0.0, abs=_MW)
        assert _f(r1.cash_delta_eur) == pytest.approx(0.0, abs=_EUR)

    def test_self_match_passive_leg_is_attributed_in_report(self):
        """I1: the own resting BUY's fill is the passive leg: +3 MW and
        -3 * 50.00 * 0.25 = -37.5 EUR."""
        _s, _r0, r1 = _run_full_self_match()
        assert _f(r1.passive_qty_signed) == pytest.approx(3.0, abs=_MW)
        assert _f(r1.passive_value_eur) == pytest.approx(-3.0 * 50.0 * _D,
                                                         abs=_EUR)

    def test_self_match_report_legs_are_consistent(self):
        """Aggressive leg (-3 MW, +37.5) plus passive leg (+3 MW, -37.5)
        add up to the reported deltas."""
        _s, _r0, r1 = _run_full_self_match()
        aggr_inv = -_f(r1.submit.trade_qty)                    # own SELL
        aggr_cash = _f(r1.submit.settlement_value)             # + on a sell
        assert aggr_cash == pytest.approx(3.0 * 50.0 * _D, abs=_EUR)
        assert _f(r1.inventory_delta) == pytest.approx(
            aggr_inv + _f(r1.passive_qty_signed), abs=_MW)
        assert _f(r1.cash_delta_eur) == pytest.approx(
            aggr_cash + _f(r1.passive_value_eur), abs=_EUR)

    def test_self_match_trade_rows_record_both_legs(self):
        """I2: the step's trade rows carry 3 MW on the buyer side and 3 MW on
        the seller side for the participant, at 50.00."""
        _s, _r0, r1 = _run_full_self_match()
        rows = np.asarray(r1.trade_rows).reshape(-1, 6)
        bought = rows[rows[:, 3] == AGENT_TRADER_ID]
        sold = rows[rows[:, 4] == AGENT_TRADER_ID]
        assert int(bought[:, 2].sum()) == _qint(3.0)
        assert int(sold[:, 2].sum()) == _qint(3.0)
        assert set(bought[:, 1].tolist()) == {_pint(50.0)}
        assert set(sold[:, 1].tolist()) == {_pint(50.0)}

    def test_self_match_trade_rows_agree_with_report(self):
        _s, _r0, r1 = _run_full_self_match()
        assert _rows_inventory(r1.trade_rows) == pytest.approx(
            _f(r1.inventory_delta), abs=_MW)
        assert _rows_pnl(r1.trade_rows) == pytest.approx(
            _f(r1.cash_delta_eur), abs=_EUR)

    def test_self_match_tape_settles_to_zero(self):
        s, _r0, _r1 = _run_full_self_match()
        assert _rows_pnl(s.agent_trades) == pytest.approx(0.0, abs=_EUR)
        assert _rows_inventory(s.agent_trades) == pytest.approx(0.0, abs=_MW)


# ---------------------------------------------------------------------------
# Partial self-match: own 3 MW at 50.00, then 2 MW against background at 49.00
# ---------------------------------------------------------------------------
class TestPartialSelfMatch:

    def test_aggressive_order_walks_own_bid_then_background_bid(self):
        s, _r0, r1 = _run_partial_self_match()
        assert _f(r1.submit.trade_qty) == pytest.approx(5.0, abs=_MW)
        assert _f(r1.submit.settlement_value) == pytest.approx(
            (3.0 * 50.0 + 2.0 * 49.0) * _D, abs=_EUR)
        tob = _session().top_of_book(s)
        assert _f(tob.p_bid) == pytest.approx(49.0, abs=1e-4)
        assert _f(tob.q_bid) == pytest.approx(3.0, abs=_MW)

    def test_only_background_remainder_moves_inventory(self):
        """Net -2 MW: the own 3 MW nets out, the 2 MW sold to background stays."""
        s, _r0, r1 = _run_partial_self_match()
        assert _f(r1.inventory_delta) == pytest.approx(-2.0, abs=_MW)
        assert _f(s.agent_inventory[0]) == pytest.approx(-2.0, abs=_MW)

    def test_only_background_remainder_moves_cash(self):
        """Net +2 * 49.00 * 0.25 = +24.5 EUR."""
        s, _r0, r1 = _run_partial_self_match()
        assert _f(r1.cash_delta_eur) == pytest.approx(2.0 * 49.0 * _D, abs=_EUR)
        assert _f(s.agent_cash[0]) == pytest.approx(2.0 * 49.0 * _D, abs=_EUR)

    def test_partial_self_match_passive_leg_is_own_quantity_only(self):
        _s, _r0, r1 = _run_partial_self_match()
        assert _f(r1.passive_qty_signed) == pytest.approx(3.0, abs=_MW)
        assert _f(r1.passive_value_eur) == pytest.approx(-3.0 * 50.0 * _D,
                                                         abs=_EUR)

    def test_partial_self_match_tape_equals_cash_and_inventory(self):
        s, _r0, r1 = _run_partial_self_match()
        assert _rows_inventory(r1.trade_rows) == pytest.approx(
            _f(r1.inventory_delta), abs=_MW)
        assert _rows_pnl(s.agent_trades) == pytest.approx(
            _f(s.agent_cash[0]), abs=_EUR)


# ---------------------------------------------------------------------------
# AC2 -- compute_agent_pnl over the tape equals agent_cash (mixed session)
# ---------------------------------------------------------------------------
class TestTapeSettlementEqualsCash:

    _EXPECTED_CASH = -1.0 * 60.0 * _D + 2.0 * 59.0 * _D    # -15.0 + 29.5
    _EXPECTED_INV = 1.0 - 2.0

    def test_mixed_session_reaches_done_with_ordinary_fills(self):
        """Sanity of the scenario: the ordinary aggressive and passive fills
        happen (independently of the self-match fix)."""
        _s, reps, done = _run_mixed_session()
        assert done
        assert bool(reps[2].submit.traded)
        assert _f(reps[2].submit.trade_qty) == pytest.approx(1.0, abs=_MW)
        assert _f(reps[4].passive_qty_signed) == pytest.approx(-2.0, abs=_MW)
        assert _f(reps[4].passive_value_eur) == pytest.approx(2.0 * 59.0 * _D,
                                                              abs=_EUR)

    def test_compute_agent_pnl_equals_agent_cash(self):
        s, _reps, _done = _run_mixed_session()
        assert _rows_pnl(s.agent_trades) == pytest.approx(
            _f(s.agent_cash[0]), abs=_EUR)

    def test_agent_cash_equals_analytic_value_self_match_contributes_zero(self):
        s, _reps, _done = _run_mixed_session()
        assert _f(s.agent_cash[0]) == pytest.approx(self._EXPECTED_CASH,
                                                    abs=_EUR)

    def test_compute_agent_pnl_equals_analytic_value(self):
        s, _reps, _done = _run_mixed_session()
        assert _rows_pnl(s.agent_trades) == pytest.approx(self._EXPECTED_CASH,
                                                          abs=_EUR)

    def test_inventory_equals_analytic_value_and_tape(self):
        s, _reps, _done = _run_mixed_session()
        assert _f(s.agent_inventory[0]) == pytest.approx(self._EXPECTED_INV,
                                                         abs=_MW)
        assert _rows_inventory(s.agent_trades) == pytest.approx(
            self._EXPECTED_INV, abs=_MW)

    def test_cash_is_sum_of_report_cash_deltas(self):
        s, reps, _done = _run_mixed_session()
        total = sum(_f(r.cash_delta_eur) for r in reps)
        assert _f(s.agent_cash[0]) == pytest.approx(total, abs=_EUR)


# ---------------------------------------------------------------------------
# Controls -- no self-match, behaviour unchanged
# ---------------------------------------------------------------------------
class TestNoSelfMatchControl:

    def test_aggressive_only_fill_attributes_only_the_aggressive_leg(self):
        """Own BUY IOC 60 x1 against the background ask: +1 MW, -15.0 EUR, no
        passive leg."""
        s = _start()
        s, r, _ = _step(s, AgentOrder.create(side=BUY, price=60.0, quantity=1.0,
                                             exec_restriction=IOC))
        assert bool(r.submit.traded)
        assert _f(r.passive_qty_signed) == 0.0
        assert _f(r.passive_value_eur) == 0.0
        assert _f(r.inventory_delta) == pytest.approx(1.0, abs=_MW)
        assert _f(r.cash_delta_eur) == pytest.approx(-60.0 * _D, abs=_EUR)
        rows = np.asarray(r.trade_rows).reshape(-1, 6)
        live = rows[rows[:, 2] != 0]
        assert live.shape[0] == 1
        assert int(live[0, 3]) == AGENT_TRADER_ID
        assert int(live[0, 4]) != AGENT_TRADER_ID
        assert _f(s.agent_inventory[0]) == pytest.approx(1.0, abs=_MW)
        assert _rows_pnl(s.agent_trades) == pytest.approx(
            _f(s.agent_cash[0]), abs=_EUR)

    def test_own_aggressive_fill_against_background_with_own_order_resting_elsewhere(self):
        """Own resting BUY 50 x3 does not interact with an own BUY IOC 60 x1
        that fills against the background ask: no passive leg."""
        s = _start()
        s, _r0, _ = _step(s, _rest_buy_50x3())
        s, r1, _ = _step(s, AgentOrder.create(side=BUY, price=60.0,
                                              quantity=1.0,
                                              exec_restriction=IOC))
        assert _f(r1.passive_qty_signed) == 0.0
        assert _f(r1.passive_value_eur) == 0.0
        assert _f(r1.inventory_delta) == pytest.approx(1.0, abs=_MW)
        assert _f(r1.cash_delta_eur) == pytest.approx(-60.0 * _D, abs=_EUR)
        tob = _session().top_of_book(s)
        assert _f(tob.p_bid) == pytest.approx(50.0, abs=1e-4)
        assert _f(tob.q_bid) == pytest.approx(3.0, abs=_MW)

    def test_resting_only_order_moves_nothing(self):
        s = _start()
        s, r, _ = _step(s, _rest_buy_50x3())
        assert not bool(r.submit.traded)
        assert _f(r.inventory_delta) == 0.0
        assert _f(r.cash_delta_eur) == 0.0
        assert not np.any(np.asarray(r.trade_rows))
