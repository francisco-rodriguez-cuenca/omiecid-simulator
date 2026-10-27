"""Locked behavioural tests for OMIE-AGENT-TIME-PRIORITY-1.

Contract (agile-plan.md §Story OMIE-AGENT-TIME-PRIORITY-1, v5.158,
"Test-first contract"; market-spec.md v2.10 §5.2, global-queue rule):

    "All timestamps ... are assigned from a single global queue (unique,
    sequential)."  Inside the book that queue is ``ts_counter``. An agent
    order is therefore stamped from the POST-BACKGROUND book's arrival
    counter (the counter after that step's background replay, before the
    agent submit), NOT from the minutes-from-GOT env clock ``state.time``.

Consequences pinned here (black-box, through ``Session.step_env``; the
participant's orders are public ``AgentOrder`` values, the session stamps the
order id, participant id and timestamp):

* an agent resting order queues BEHIND every earlier same-price resting
  order, on both sides, whichever path it rested by (rest-only or
  fill-then-rest);
* successive agent orders get strictly increasing stamps (no tie);
* the stamp equals the post-background counter whatever the clock / tape
  seconds / GCT / ``tape_seconds_span``;
* the agent-aggressor trade ``SEC`` carries the same counter value.

Guards (identical on the unfixed base and on the fix): the no-order world
is the pure replay, fully-filled agent orders score identically, the clock
and (where no priority outcome differs) the top of book are unchanged, the
agent stays ahead of a LATER-second arrival, and the one-compile invariant.

Two strict xfails characterise residual limitations that only a tape-side
global sequence (option (iii)) would remove: (5b) O1 counter drift and
(7) R1 raw-second tape stamps.

Every literal below was re-derived by the test-agent on the unfixed code
(worktree @ 434b3a4 == scheduled base e557fa4 for the env/book code) and on
a scratch copy with only the card's one-line change applied, CPU.

RED on the unfixed code (fail until the fix lands):
    (1a/1b/1c/1d), (2) stamp, (2b), (3) x5, (3b) step-1 stamp,
    (6) fill split, (8) x2, and (5b) (strict xfail -> XPASS(strict) = FAIL).
GREEN on the unfixed code (guards):
    (2) step-3 guard, (3b) step-0 stamp, (4) x2, (5), (6) over-correction
    guard, (7) (strict xfail on both), (9) clock + top of book, (10).

Layouts:
    book side row (jaxob ``OrderSideFeat``): [P, Q, OID, TID, SEC, NSEC]
    trades row (jaxob ``TradesFeat``):
        [P, Q, PASS_OID, AGRS_OID, SEC, NSEC, PASS_TID, AGRS_TID]
    trade ``Q`` is SIGNED (negative when the aggressor is a BUY).
    13-column tape row: [contract, side, price_tick, qty_tick, oid,
        agent_id, ts, exec_restriction, validity, expiry, status,
        peak_tick, ppd]

Device policy: CPU (conftest). Every test builds and steps a jitted env
(> 1 s compile), so the whole file carries ``@pytest.mark.slow``. Rollouts
are memoised per fixture and the JAX caches are cleared after each fresh
rollout so several env compiles can share one process.

Run:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python3 -m pytest \
        tests/omiecid/test_agent_time_priority.py -n0 -q
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat, TradesFeat
from omiecid_simulator.jaxomiecid.omie_data_loader import _replay_orders
from omiecid_simulator.omiecid.contract_book import _submit_order
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.omie_session import AgentOrder, Session
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    Status,
    Validity,
)

pytestmark = [pytest.mark.slow]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PC = OMIEProductConfig()

BUY = int(Side.BUY)
SELL = int(Side.SELL)
NON = int(ExecRestriction.NON)
IOC = int(ExecRestriction.IOC)
ICE = int(ExecRestriction.ICE)

AGENT_OID0 = 2_000_000_000          # agent OID = 2_000_000_000 + step_counter
BG_TRADER = 777                     # anonymous background trader id

# Participant orders: (side, price EUR/MWh, quantity MW), all resting NON
# GFS; ``None`` submits nothing. Each price is the touch the order joins
# (own side) or crosses (opposite side) on the post-background book of the
# step it is submitted on.
NOOP = None


def _o(side, price, qty=3.0):
    return (int(side), float(price), float(qty))


def _agent_order(spec):
    if spec is None:
        return AgentOrder.none()
    side, price, qty = spec
    return AgentOrder.create(side=side, price=price, quantity=qty)

_B_P = OrderSideFeat.P.value
_B_Q = OrderSideFeat.Q.value
_B_OID = OrderSideFeat.OID.value
_B_SEC = OrderSideFeat.SEC.value

_T_P = TradesFeat.P.value
_T_Q = TradesFeat.Q.value
_T_PASS = TradesFeat.PASS_OID.value
_T_AGRS = TradesFeat.AGRS_OID.value
_T_SEC = TradesFeat.SEC.value

_CASH_ABS = 0.02
_QTY_ABS = 1e-3

_PAD = [0, 0, 0, 0, -1, 0, -1, NON, int(Validity.GFS), 0,
        int(Status.ACTIVE), 0, 0]


def _ticks(qty_mw: float) -> int:
    return int(round(qty_mw / float(_PC.qty_tick)))


def _price_int(price: float) -> int:
    return int(round((price - float(_PC.price_min)) / float(_PC.price_tick)))


def _row(side, price, qty, oid, ts, er=NON, peak=0.0):
    return [0, int(side), _price_int(price), _ticks(qty), int(oid), BG_TRADER,
            int(ts), int(er), int(Validity.GFS), 0, int(Status.ACTIVE),
            _ticks(peak), 0]


def _agent(step: int) -> int:
    return AGENT_OID0 + step


# ---------------------------------------------------------------------------
# Fixtures (in-source tapes). name -> (rows, orders, orders_per_step, extra)
# ---------------------------------------------------------------------------
_R = _row

_ROWS_1A = [
    _R(BUY, 50.00, 10.0, 311, 600),
    _R(SELL, 50.40, 5.0, 509, 660),
    _R(SELL, 50.00, 12.0, 347, 900, IOC),
]


def _shift(rows, sec):
    return [r[:6] + [r[6] + sec] + r[7:] for r in rows]


# The Session ignores calls once ``done`` (time >= GCT) fires. On the
# late-second tapes below the span-scaled clock already reads 60 on step 0,
# so a 60-minute horizon would end the session before the participant's
# orders. A 61-minute horizon over a 56,250 s span keeps every slice of
# these tapes at minute 60 (the same clock) and the session open.
_OPEN_AT_60 = {"gct_minutes_from_got": 61, "tape_seconds_span": 56_250}


_FIXTURES = {
    # (1a) op1, BUY side.
    "1a": (_ROWS_1A, [NOOP, _o(BUY, 50.00), NOOP], 1, {}),
    # (1b) op1 mirror, SELL side, off-round prices.
    "1b": ([
        _R(SELL, 61.37, 8.0, 2203, 1210),
        _R(BUY, 60.90, 4.0, 2207, 1300),
        _R(BUY, 61.37, 9.5, 2219, 1500, IOC),
    ], [NOOP, _o(SELL, 61.37), NOOP], 1, {}),
    # (1c) op4, agent submits at step 0 (base clock == 0 there).
    "1c": ([
        _R(BUY, 50.00, 10.0, 311, 600),
        _R(SELL, 50.40, 5.0, 509, 660),
        _R(BUY, 49.50, 1.0, 313, 700),
        _R(SELL, 51.00, 1.0, 511, 720),
        _R(SELL, 50.00, 12.0, 347, 900, IOC), _PAD, _PAD, _PAD,
    ], [_o(BUY, 50.00), NOOP], 4, {}),
    # (1d) op4, several same-price arrivals in one slice.
    "1d": ([
        _R(BUY, 50.00, 2.0, 801, 600),
        _R(BUY, 50.00, 2.0, 802, 610),
        _R(BUY, 50.00, 2.0, 803, 620),
        _R(SELL, 50.40, 1.0, 804, 630),
        _R(SELL, 50.00, 7.5, 805, 900, IOC), _PAD, _PAD, _PAD,
    ], [_o(BUY, 50.00), NOOP], 4, {}),
    # (2) fill-then-rest (step 1 crosses the 50.40 ask) + rest-only (step 2
    # joins the agent's own 50.40 bid).
    "2": ([
        _R(SELL, 50.40, 1.0, 4127, 600),
        _R(BUY, 49.80, 2.0, 4133, 700),
        _R(BUY, 49.10, 1.0, 4139, 760),
        _R(SELL, 50.40, 3.0, 4141, 900, IOC),
    ], [NOOP, _o(BUY, 50.40), _o(BUY, 50.40), NOOP], 1, {}),
    # (2b) two agent rests separated by far non-crossing rows at the SAME raw
    # second (clock does not move between the two submits).
    "2b": ([
        _R(BUY, 50.00, 10.0, 311, 600),
        _R(SELL, 70.00, 1.0, 9901, 600),
        _R(SELL, 70.10, 1.0, 9903, 600),
        _R(SELL, 50.00, 12.0, 347, 900, IOC),
    ], [NOOP, _o(BUY, 50.00), _o(BUY, 50.00), NOOP], 1, {}),
    # (3) variants of (1a).
    "3_shift54600": (_shift(_ROWS_1A, 54_600), [NOOP, _o(BUY, 50.00), NOOP], 1,
                     _OPEN_AT_60),
    "3_shift12345": (_shift(_ROWS_1A, 12_345), [NOOP, _o(BUY, 50.00), NOOP], 1, {}),
    "3_gct45": (_ROWS_1A, [NOOP, _o(BUY, 50.00), NOOP], 1,
                {"gct_minutes_from_got": 45}),
    "3_span86400": (_ROWS_1A, [NOOP, _o(BUY, 50.00), NOOP], 1,
                    {"tape_seconds_span": 86400}),
    # (4) no-order rollout of (1a).
    "4_noop": (_ROWS_1A, [NOOP, NOOP, NOOP], 1, {}),
    # (5) fully-filled agent orders crossing the 52.10 ask.
    "5": ([
        _R(SELL, 52.10, 20.0, 7301, 55_300),
        _R(SELL, 52.30, 20.0, 7303, 55_420),
        _R(BUY, 51.80, 5.0, 7305, 55_480),
        _R(BUY, 51.70, 5.0, 7307, 55_540),
    ], [NOOP, _o(BUY, 52.10), _o(BUY, 52.10), _o(BUY, 52.10)], 1, _OPEN_AT_60),
    # (5b) iceberg regen vs a later tape SELL.
    "5b": ([
        _R(SELL, 52.30, 3.0, 7303, 55_300, ICE, peak=1.0),
        _R(SELL, 52.10, 20.0, 7301, 55_300),
        _R(BUY, 51.80, 5.0, 7305, 55_300),
        _R(BUY, 52.30, 15.0, 7307, 55_300, IOC),
        _R(SELL, 52.30, 1.0, 7311, 55_304),
        _R(BUY, 52.30, 1.0, 7313, 55_305, IOC),
    ], [NOOP, _o(BUY, 52.10), _o(BUY, 52.10), NOOP, NOOP, NOOP], 1,
        _OPEN_AT_60),
    # (6) later-second same-price arrival (353 @ 800 > agent stamp 661).
    "6": ([
        _R(BUY, 50.00, 10.0, 311, 600),
        _R(SELL, 50.40, 5.0, 509, 660),
        _R(BUY, 50.00, 2.0, 353, 800),
        _R(SELL, 50.00, 12.0, 347, 900, IOC),
    ], [NOOP, _o(BUY, 50.00), NOOP, NOOP], 1, {}),
    # (7) as (6) but 355 arrives LATER at a raw second (660) below the agent's
    # counter stamp (661).
    "7": ([
        _R(BUY, 50.00, 10.0, 311, 600),
        _R(SELL, 50.40, 5.0, 509, 660),
        _R(BUY, 50.00, 2.0, 355, 660),
        _R(SELL, 50.00, 12.0, 347, 900, IOC),
    ], [NOOP, _o(BUY, 50.00), NOOP, NOOP], 1, {}),
}


def _make_env(name):
    rows, _orders, ops, extra = _FIXTURES[name]
    rows = list(rows) + [_PAD] * (8 * ops)
    cfg = {
        "tapes": np.asarray(rows, dtype=np.int32)[None],
        "orders_per_step": ops,
        "gct_minutes_from_got": 60,
    }
    cfg.update(extra)
    return Session(cfg), np.asarray(rows, dtype=np.int32)


def _live(arr):
    arr = np.asarray(arr)
    return arr[(arr[:, _B_OID] > 0) & (arr[:, _B_Q] > 0)]


class _Step:
    """numpy record of one ``step_env`` call."""

    def __init__(self, env, pre, post, report, done, bg_counter):
        tr0 = np.asarray(pre.book.lob.trades)
        tr1 = np.asarray(post.book.lob.trades)
        new = [j for j in range(tr1.shape[0])
               if tr0[j, _T_PASS] == -1 and tr1[j, _T_PASS] != -1]
        self.trades = tr1[new]                      # new trades, in order
        self.pre_time = int(pre.time)
        self.time = int(post.time)
        self.cash_delta = float(report.cash_delta_eur)   # realized EUR
        self.done = bool(done)
        tob = env.top_of_book(post)
        # p_bid, p_ask, q_bid, q_ask, step_counter, inventory
        self.market = np.asarray([
            float(tob.p_bid), float(tob.p_ask), float(tob.q_bid),
            float(tob.q_ask),
            float(post.step_counter), float(post.agent_inventory[0]),
        ], dtype=np.float64)
        self.inventory = float(post.agent_inventory[0])
        self.ts_counter = int(post.book.ts_counter)
        self.bg_counter = int(bg_counter)          # counter after bg replay
        self.bids = _live(post.book.lob.bids)
        self.asks = _live(post.book.lob.asks)
        step = int(pre.step_counter)
        self.agent_trades = np.asarray(post.agent_trades)[step]


@functools.lru_cache(maxsize=None)
def _rollout(name):
    env, rows = _make_env(name)
    _rows, orders, ops, _extra = _FIXTURES[name]
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    steps = []
    for spec in orders:
        cur = int(state.tape_cursor)
        bg_book, _slab = _replay_orders(
            state.book, jnp.asarray(rows[cur:cur + ops]),
            env.product_config, env.market_config,
        )
        nxt, report, done = env.step_env(key, state, _agent_order(spec))
        steps.append(_Step(env, state, nxt, report, done,
                           bg_book.ts_counter))
        state = nxt
    jax.clear_caches()
    return steps


def _trade_pairs(step):
    """[(PASS_OID, signed Q ticks), ...] of the new trades of a step."""
    return [(int(t[_T_PASS]), int(t[_T_Q])) for t in step.trades]


def _abs_pairs(step):
    return [(p, abs(q)) for p, q in _trade_pairs(step)]


def _row_of(side_arr, oid):
    hit = side_arr[side_arr[:, _B_OID] == oid]
    assert hit.shape[0] == 1, f"oid {oid}: expected one live row, got {hit}"
    return hit[0]


def _has(side_arr, oid):
    return bool(np.any(side_arr[:, _B_OID] == oid))


# ---------------------------------------------------------------------------
# (1) RED — the agent rest queues behind an earlier same-price order.
# ---------------------------------------------------------------------------
# name -> (submit_step, fill_step, agent side, expected stamp,
#          expected fill-step pairs (PASS_OID, |Q| ticks),
#          inventory, fill-step realized cash (EUR), agent residual MW, price)
_CASE1 = {
    "1a": (1, 2, BUY, 661, [(311, _ticks(10.0)), (_agent(1), _ticks(2.0))],
           2.0, -25.00, 1.0, 50.00),
    "1b": (1, 2, SELL, 1301, [(2203, _ticks(8.0)), (_agent(1), _ticks(1.5))],
           -1.5, 23.01375, 1.5, 61.37),
    "1c": (0, 1, BUY, 721, [(311, _ticks(10.0)), (_agent(0), _ticks(2.0))],
           2.0, -25.00, 1.0, 50.00),
    "1d": (0, 1, BUY, 631, [(801, _ticks(2.0)), (802, _ticks(2.0)),
                            (803, _ticks(2.0)), (_agent(0), _ticks(1.5))],
           1.5, -18.75, 1.5, 50.00),
}


class TestAgentRestQueuesBehindEarlierOrders:
    """(1) Base: the agent is stamped with the minute clock (40 / 48 / 0 / 0),
    outranks every same-price tape order and is filled FIRST."""

    @pytest.mark.parametrize("name", ["1a", "1b", "1c", "1d"])
    def test_agent_rest_queues_behind_earlier_same_price_order(self, name):
        (sub, fill, side, stamp, pairs, inv, cash, resid,
         price) = _CASE1[name]
        steps = _rollout(name)
        own = steps[sub].bids if side == BUY else steps[sub].asks
        row = _row_of(own, _agent(sub))
        assert int(row[_B_SEC]) == stamp, (
            f"[{name}] agent resting SEC must be the post-background counter "
            f"{stamp}, got {int(row[_B_SEC])} (base stamps the minute clock)"
        )
        assert int(row[_B_SEC]) == steps[sub].bg_counter
        assert _abs_pairs(steps[fill]) == pairs, (
            f"[{name}] fill order must serve the earlier tape order(s) first"
        )
        # Trade Q is signed: negative for a BUY aggressor.
        sign = -1 if side == SELL else 1   # aggressor = opposite of the agent
        assert all(np.sign(q) == sign for _p, q in _trade_pairs(steps[fill]))
        assert steps[fill].inventory == pytest.approx(inv, abs=_QTY_ABS)
        assert steps[fill].cash_delta == pytest.approx(cash, abs=_CASH_ABS)
        own_after = steps[fill].bids if side == BUY else steps[fill].asks
        arow = _row_of(own_after, _agent(sub))
        assert int(arow[_B_Q]) == _ticks(resid)
        assert int(arow[_B_P]) == _price_int(price)
        assert int(arow[_B_SEC]) == stamp

    def test_1a_earlier_order_fully_consumed(self):
        steps = _rollout("1a")
        assert not _has(steps[2].bids, 311), "311 must be fully filled"

    def test_1d_every_earlier_arrival_fully_consumed(self):
        steps = _rollout("1d")
        for oid in (801, 802, 803):
            assert not _has(steps[1].bids, oid)


# ---------------------------------------------------------------------------
# (2) rest-only and fill-then-rest share one sequence.
# ---------------------------------------------------------------------------
class TestOneSequenceAcrossRestPaths:

    def test_rest_only_and_fill_then_rest_share_one_sequence(self):
        """RED lock. Base: 40 / 47."""
        steps = _rollout("2")
        s1 = int(_row_of(steps[1].bids, _agent(1))[_B_SEC])   # fill-then-rest
        s2 = int(_row_of(steps[2].bids, _agent(2))[_B_SEC])   # rest-only
        assert (s1, s2) == (701, 761), f"got {(s1, s2)}; base 40/47"
        assert s1 == steps[1].bg_counter
        assert s2 == steps[2].bg_counter
        # strictly greater than every stamp resting at submit (700 / 760)
        assert s1 > 700 and s2 > 760
        assert s2 > s1

    def test_step3_fill_order_guard(self):
        """GUARD (identical on base and fix)."""
        steps = _rollout("2")
        assert _abs_pairs(steps[3]) == [(_agent(1), _ticks(2.0)),
                                        (_agent(2), _ticks(1.0))]
        row = _row_of(steps[3].bids, _agent(2))
        assert int(row[_B_Q]) == _ticks(2.0)
        assert int(row[_B_P]) == _price_int(50.40)
        assert not _has(steps[3].bids, _agent(1))
        assert steps[3].inventory == pytest.approx(4.0, abs=_QTY_ABS)
        assert steps[3].cash_delta == pytest.approx(-37.80, abs=_CASH_ABS)


# ---------------------------------------------------------------------------
# (2b) RED — two agent rests at the same clock value do not tie.
# ---------------------------------------------------------------------------
def test_two_agent_rests_separated_by_non_crossing_rows_do_not_tie():
    """Base: both stamped 40 (tie -> slot), rows (…001 3.0), (…002 3.0),
    (311 6.0), inventory 6.0, -75.00."""
    steps = _rollout("2b")
    s1 = int(_row_of(steps[1].bids, _agent(1))[_B_SEC])
    s2 = int(_row_of(steps[2].bids, _agent(2))[_B_SEC])
    assert steps[1].pre_time == steps[2].pre_time == 40   # clock did not move
    assert (s1, s2) == (601, 602)
    assert s1 == steps[1].bg_counter and s2 == steps[2].bg_counter
    assert s2 > s1
    assert _abs_pairs(steps[3]) == [(311, _ticks(10.0)),
                                    (_agent(1), _ticks(2.0))]
    assert steps[3].inventory == pytest.approx(2.0, abs=_QTY_ABS)
    assert steps[3].cash_delta == pytest.approx(-25.00, abs=_CASH_ABS)
    assert int(_row_of(steps[3].bids, _agent(2))[_B_Q]) == _ticks(3.0)
    assert int(_row_of(steps[3].bids, _agent(1))[_B_Q]) == _ticks(1.0)


# ---------------------------------------------------------------------------
# (3) RED — the stamp reads the book, not the clock.
# ---------------------------------------------------------------------------
# name -> expected stamp (base stamp in the comment)
_CASE3 = {
    "1a": 661,              # base 40
    "3_shift54600": 55_261,  # base 60
    "3_shift12345": 13_006,  # base 59
    "3_gct45": 661,          # base 30
    "3_span86400": 661,      # base 0
}


@pytest.mark.parametrize("name", list(_CASE3))
def test_agent_stamp_reads_the_book_not_the_clock(name):
    steps = _rollout(name)
    row = _row_of(steps[1].bids, _agent(1))
    assert int(row[_B_SEC]) == steps[1].bg_counter, (
        f"[{name}] agent SEC {int(row[_B_SEC])} != post-background counter "
        f"{steps[1].bg_counter}"
    )
    assert int(row[_B_SEC]) == _CASE3[name]
    # the (1a) fill split is unchanged in every variant
    assert _abs_pairs(steps[2]) == [(311, _ticks(10.0)),
                                    (_agent(1), _ticks(2.0))]
    assert steps[2].inventory == pytest.approx(2.0, abs=_QTY_ABS)
    assert steps[2].cash_delta == pytest.approx(-25.00, abs=_CASH_ABS)


class TestStampTimestamp:
    """(3b) call ``env._stamp_order`` directly (no submit)."""

    @staticmethod
    def _env_and_states():
        env, _rows = _make_env("1a")
        key = jax.random.PRNGKey(0)
        s0 = env.reset(key)
        s1, *_ = env.step_env(key, s0, AgentOrder.none())
        # No order is submitted, so s2.book is the step-1 post-background
        # book.
        s2, *_ = env.step_env(key, s1, AgentOrder.none())
        return env, s0, s1, s2

    def test_stamp_on_step0_empty_book_is_int32_zero(self):
        """GUARD (equal on base and fix)."""
        env, s0, _s1, _s2 = self._env_and_states()
        order, _noop, _side = env._stamp_order(
            _agent_order(_o(BUY, 0.0)), s0.book, s0)
        assert order.timestamp.dtype == jnp.int32
        assert int(order.timestamp) == 0

    def test_stamp_on_step1_book_reads_counter(self):
        """RED. Base: 40 (the pre-step clock)."""
        env, _s0, s1, s2 = self._env_and_states()
        order, _noop, _side = env._stamp_order(
            _agent_order(_o(BUY, 50.00)), s2.book, s1)
        assert order.timestamp.dtype == jnp.int32
        assert int(s2.book.ts_counter) == 661
        assert int(order.timestamp) == 661


# ---------------------------------------------------------------------------
# (4) GUARD — a no-order episode is the pure replay.
# ---------------------------------------------------------------------------
def _assert_book_equal(a, b):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _noop_until_done(env, rows):
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    tape = np.asarray(env.tapes[int(state.tape_idx)]) if rows is None else rows
    for _ in range(env.steps_per_episode):
        state, _report, done = env.step_env(key, state, AgentOrder.none())
        if bool(done):
            break
    assert bool(done), "fixture never reached done"
    consumed = int(state.tape_cursor)
    ref, _slab = _replay_orders(
        env._empty_book(), jnp.asarray(tape[:consumed]),
        env.product_config, env.market_config,
    )
    return state.book, ref


class TestNoopEpisodeIsPureReplay:

    def test_noop_episode_identical_to_pure_replay_1a(self):
        env, rows = _make_env("1a")
        book, ref = _noop_until_done(env, rows)
        _assert_book_equal(book, ref)
        jax.clear_caches()

    def test_noop_episode_identical_to_pure_replay_default_env(self):
        env = Session.from_test_config()
        book, ref = _noop_until_done(env, None)
        _assert_book_equal(book, ref)
        jax.clear_caches()


# ---------------------------------------------------------------------------
# (5) GUARD — fully-filled agent orders: scored legs unchanged.
# ---------------------------------------------------------------------------
def test_fully_filled_agent_orders_scored_legs_unchanged():
    steps = _rollout("5")
    for k, inv in ((1, 3.0), (2, 6.0), (3, 9.0)):
        assert steps[k].cash_delta == pytest.approx(-39.075, abs=_CASH_ABS)
        assert steps[k].inventory == pytest.approx(inv, abs=_QTY_ABS)
        assert _trade_pairs(steps[k]) == [(7301, -_ticks(3.0))]
        np.testing.assert_array_equal(
            steps[k].agent_trades,
            np.asarray([[60, _price_int(52.10), _ticks(3.0), 1_000_000,
                         2_000_000_000, 0], [0, 0, 0, 0, 0, 0]]),
        )
    final = steps[3]
    got_b = [(int(r[_B_OID]), int(r[_B_Q]), int(r[_B_P])) for r in final.bids]
    got_a = [(int(r[_B_OID]), int(r[_B_Q]), int(r[_B_P])) for r in final.asks]
    assert sorted(got_b) == sorted([(7305, _ticks(5.0), _price_int(51.80)),
                                    (7307, _ticks(5.0), _price_int(51.70))])
    assert sorted(got_a) == sorted([(7301, _ticks(11.0), _price_int(52.10)),
                                    (7303, _ticks(20.0), _price_int(52.30))])


# ---------------------------------------------------------------------------
# (5b) O1 counter drift reorders regen vs a later tape order. Formerly a STRICT
# XFAIL recording known deviation O1; OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence
# (30)) stamps the regen slice at its activation event, which fixes O1: the
# marker is dropped and the arrival-correct assertions are kept unchanged.
# ---------------------------------------------------------------------------
def test_counter_drift_reorders_background_regen_vs_later_tape():
    """Arrival-correct outcome: the regen slice of 7303 (activated at step 3,
    before 7311 arrived at 55,304) fills 7313. Passes at the E1 + E2 tip: the
    regen slice is stamped at its activation event (SEC 55,303 < 55,304), not
    at a drifted counter value (55,305, which would let 7311 fill first)."""
    steps = _rollout("5b")
    assert _abs_pairs(steps[5]) == [(7303, _ticks(1.0))]
    assert _has(steps[5].asks, 7311)
    assert steps[5].market[3] == pytest.approx(2.0, abs=_QTY_ABS)   # q_ask


# ---------------------------------------------------------------------------
# (6) over-correction: the agent stays AHEAD of a later-second arrival.
# ---------------------------------------------------------------------------
class TestNoOverCorrection:

    def test_agent_stays_ahead_of_later_second_arrival(self):
        """GUARD (holds on base and fix): 353 (ts 800) is never served before
        the agent and keeps its full 2.0."""
        steps = _rollout("6")
        passes = [p for p, _q in _trade_pairs(steps[3])]
        assert 353 not in passes
        assert _agent(1) in passes
        assert int(_row_of(steps[3].bids, 353)[_B_Q]) == _ticks(2.0)

    def test_later_second_arrival_fill_split(self):
        """RED. Base: (agent 3.0), (311 9.0), inventory 3.0, -37.50."""
        steps = _rollout("6")
        assert _abs_pairs(steps[3]) == [(311, _ticks(10.0)),
                                        (_agent(1), _ticks(2.0))]
        assert steps[3].inventory == pytest.approx(2.0, abs=_QTY_ABS)
        assert steps[3].cash_delta == pytest.approx(-25.00, abs=_CASH_ABS)
        assert int(_row_of(steps[3].bids, _agent(1))[_B_Q]) == _ticks(1.0)


# ---------------------------------------------------------------------------
# (7) STRICT XFAIL — R1: a later tape order at a raw second <= the stamp.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="R1: raw-second tape stamps vs counter; option (iii), a tape-side "
           "global sequence, would flip this to XPASS",
)
def test_r1_later_raw_second_arrival_ranks_ahead_of_agent():
    """Arrival-correct outcome: the agent (arrived at step 1) is served
    before 355 (arrived at step 2). With the fix 355 (SEC 660 < 661) ranks
    ahead: rows (311 10.0), (355 2.0), agent unfilled, inventory 0.0."""
    steps = _rollout("7")
    assert _abs_pairs(steps[3]) == [(311, _ticks(10.0)),
                                    (_agent(1), _ticks(2.0))]
    assert int(_row_of(steps[3].bids, 355)[_B_Q]) == _ticks(2.0)


# ---------------------------------------------------------------------------
# (8) RED — the agent-aggressor trade SEC is the counter.
# ---------------------------------------------------------------------------
class TestAgentAggressorTradeSec:

    def test_fill_then_rest_trade_sec_is_counter(self):
        """Base: 40."""
        steps = _rollout("2")
        rows = steps[1].trades[steps[1].trades[:, _T_AGRS] == _agent(1)]
        assert rows.shape[0] == 1
        assert int(rows[0][_T_SEC]) == 701 == steps[1].bg_counter
        assert int(rows[0][_T_PASS]) == 4127

    def test_fully_filled_trade_secs_are_counter(self):
        """Base: 60 / 60 / 60 and final ts_counter 55,542."""
        steps = _rollout("5")
        secs = []
        for k in (1, 2, 3):
            rows = steps[k].trades[steps[k].trades[:, _T_AGRS] == _agent(k)]
            assert rows.shape[0] == 1
            secs.append(int(rows[0][_T_SEC]))
            assert secs[-1] == steps[k].bg_counter
        assert secs == [55_421, 55_481, 55_541]
        assert steps[3].ts_counter == 55_543


# ---------------------------------------------------------------------------
# (9) GUARD — clock and top of book unchanged.
# ---------------------------------------------------------------------------
_CLOCK = {
    "1a": [40, 44, 60],
    "1b": [48, 52, 60],
    "1c": [48, 60],
    "1d": [42, 60],
    "2": [40, 47, 51, 60],
    "2b": [40, 40, 40, 60],
    "3_shift54600": [60, 60, 60],
    "3_shift12345": [59, 59, 60],
    "3_gct45": [30, 33, 45],
    "3_span86400": [0, 0, 1],
    "5": [60, 60, 60, 60],
    "5b": [60, 60, 60, 60, 60, 60],
    "6": [40, 44, 53, 60],
    "7": [40, 44, 44, 60],
}


@pytest.mark.parametrize("name", list(_CLOCK))
def test_clock_unchanged(name):
    steps = _rollout(name)
    assert [s.time for s in steps] == _CLOCK[name]


# Per step: p_bid, p_ask, q_bid, q_ask, time_to_gct, step_counter, inventory,
# day_ahead_anchor. Compared: the four ``Session.top_of_book`` fields,
# ``state.step_counter`` and ``state.agent_inventory`` (columns 0-3, 5, 6);
# the clock is pinned by ``test_clock_unchanged``.
_OBS = {
    "4_noop": [
        [50.0, 0.0, 10.0, 0.0, 20.0, 1.0, 0.0, 0.0],
        [50.0, 50.4, 10.0, 5.0, 16.0, 2.0, 0.0, 0.0],
        [0.0, 50.4, 0.0, 5.0, 0.0, 3.0, 0.0, 0.0],
    ],
    "5": [
        [0.0, 52.1, 0.0, 20.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 52.1, 0.0, 17.0, 0.0, 2.0, 3.0, 0.0],
        [51.8, 52.1, 5.0, 14.0, 0.0, 3.0, 6.0, 0.0],
        [51.8, 52.1, 5.0, 11.0, 0.0, 4.0, 9.0, 0.0],
    ],
}


@pytest.mark.parametrize("name", list(_OBS))
def test_market_view_unchanged_where_no_priority_outcome_differs(name):
    steps = _rollout(name)
    got = np.stack([s.market for s in steps])
    np.testing.assert_allclose(
        got, np.asarray(_OBS[name])[:, [0, 1, 2, 3, 5, 6]], atol=1e-3)


# ---------------------------------------------------------------------------
# (10) GUARD — one-compile invariant.
# ---------------------------------------------------------------------------
@pytest.mark.jit
def test_no_new_submit_order_cache_entry():
    env, _rows = _make_env("1a")
    key = jax.random.PRNGKey(0)
    s0 = env.reset(key)
    s1, *_ = env.step_env(key, s0, AgentOrder.none())
    s2, *_ = env.step_env(key, s1, AgentOrder.none())
    _submit_order.clear_cache()
    assert _submit_order._cache_size() == 0
    # Per book: join the bid, cross to the ask (no ask on s1 -> 0.0), join
    # the ask (no ask on s1 -> 0.0).
    specs = {
        0: (_o(BUY, 50.00), _o(BUY, 0.0), _o(SELL, 0.0)),
        1: (_o(BUY, 50.00), _o(BUY, 50.40), _o(SELL, 50.40)),
    }
    for i, (book, state) in enumerate(((s1.book, s0), (s2.book, s1))):
        for spec in specs[i]:
            order, _noop, _side = env._stamp_order(
                _agent_order(spec), book, state)
            _nb, res = _submit_order(book, order, env.product_config,
                                     env.market_config)
            _ = bool(res.accepted)
    assert _submit_order._cache_size() == 1
    jax.clear_caches()
