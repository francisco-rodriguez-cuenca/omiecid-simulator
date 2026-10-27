"""Independent audit twin for Story REL-05 (Strip the environment into Session).

Derived from ``docs/release/release-plan.md`` §4.1 (public imports), §4.2 (kept
names), §4.4 (Session interface, Behaviour, Limitations), §5.3 (Session
additions) and ``docs/omiecid/market-spec.md`` (§5.2 timestamps / global queue,
§6.1-6.2 price-time priority and maker pricing, §7.1 anonymous book, §8.1/8.4
settlement V = Q x p x d, sell +V / buy -V). Written black-box against the
public interface only; the locked contract file was not consulted.

Cost model. Every distinct ``Session`` configuration compiles its own
``step_env`` (tens of seconds on CPU). To keep the compile count small, almost
every scenario lives on ONE session that stacks several scenario tapes and
selects a tape with ``state.replace(tape_idx=i)`` (plan §4.4 "Choosing a
tape"). Extra compiles: the vmapped step, the real-clock session and the
small-``nTrades`` session. Tests that dispatch a compiled step carry
``@pytest.mark.slow`` (slow: shares a jitted Session step compile, > 5 s cold).
Pure-type / import tests stay in the fast suite.

Self-trades are REL-06 and are deliberately avoided: no scenario lets the
participant's order cross its own resting order.
"""

from __future__ import annotations

import importlib
import importlib.util

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.omiecid.omie_session import (
    AGENT_ORDER_ID_BASE,
    AGENT_TRADER_ID,
    AgentOrder,
    FillReport,
    Session,
    TopOfBook,
)
from omiecid_simulator.omiecid.omie_states import OMIEContractState
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult
from omiecid_simulator.omiecid.encoding import encode_price
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from omiecid_simulator.omiecid.settlement import compute_agent_pnl
from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder, stack_tapes

BUY = int(Side.BUY)
SELL = int(Side.SELL)
NON = int(ExecRestriction.NON)
IOC = int(ExecRestriction.IOC)
ICE = int(ExecRestriction.ICE)

BG = 7            # background participant id on every scenario tape
D_H = 0.25        # 15-minute contract delivery duration (market-spec §8.1)
KEY = jax.random.PRNGKey(0)
SPAN = 1000       # tape seconds mapped onto [0, GCT]; only ts=1000 reaches GCT
EUR = dict(abs=1e-3)

SLOW = pytest.mark.slow  # slow: jitted Session step compile (> 5 s cold)


# ---------------------------------------------------------------------------
# Scenario tapes (one Session, one compile)
# ---------------------------------------------------------------------------
def _tape(rows):
    tb = TapeBuilder()
    for r in rows:
        r = dict(r)
        r.setdefault("agent_id", BG)
        tb.add(**r)
    return tb.build(pad_extra=0)


def _o(side, price, qty, **kw):
    return dict(side=side, price_eur=price, qty_mw=qty, **kw)


SCENARIOS = {
    # Two-level ask ladder and two-level bid ladder, built in 4 steps.
    "agg": [_o(SELL, 50.0, 1.0), _o(SELL, 51.0, 1.0),
            _o(BUY, 48.0, 3.0), _o(BUY, 47.0, 2.0)],
    # Negative prices: resting bid -10, resting ask -8 (no cross).
    "neg": [_o(BUY, -10.0, 2.0), _o(SELL, -8.0, 2.0)],
    # Agent queues BEHIND an earlier same-price arrival of the same step.
    "prio_behind": [_o(SELL, 50.0, 2.0), _o(BUY, 50.0, 2.0), _o(BUY, 50.0, 2.0)],
    # Agent queues AHEAD of a later same-price arrival.
    "prio_ahead": [_o(BUY, 40.0, 1.0), _o(SELL, 50.0, 2.0), _o(BUY, 50.0, 2.0)],
    # Mixed aggressive + passive session for cash consistency.
    "mixed": [_o(SELL, 52.0, 5.0), _o(BUY, 50.0, 2.0), _o(BUY, 55.0, 5.0),
              _o(BUY, 45.0, 2.0), _o(SELL, 44.0, 1.0), _o(SELL, 46.0, 2.0)],
    # Gate closure: the third row carries the tape's GCT timestamp.
    "done": [_o(BUY, 48.0, 2.0), _o(SELL, 52.0, 2.0),
             _o(SELL, 53.0, 1.0, timestamp=SPAN)],
    # Top-of-book: aggregated bid level, ask level plus a foreign iceberg.
    "tob": [_o(BUY, 48.0, 3.0), _o(BUY, 48.0, 2.0), _o(BUY, 47.0, 4.0),
            _o(SELL, 51.0, 4.0), _o(SELL, 52.0, 1.0),
            _o(SELL, 51.0, 10.0, exec_restriction=ICE, peak_qty_mw=2.0)],
}
TAPE_IDX = {name: i for i, name in enumerate(SCENARIOS)}


@pytest.fixture(scope="module")
def session():
    tapes = stack_tapes([_tape(rows) for rows in SCENARIOS.values()], pad_extra=8)
    return Session({"tapes": tapes, "orders_per_step": 1,
                    "tape_seconds_span": SPAN, "gct_minutes_from_got": 60})


def _start(session, name):
    return session.reset(KEY).replace(tape_idx=jnp.int32(TAPE_IDX[name]))


def _run(session, state, orders):
    """Apply a list of orders (None or AgentOrder); return state, reports, dones."""
    reports, dones = [], []
    for o in orders:
        state, rep, done = session.step(KEY, state, o)
        reports.append(rep)
        dones.append(bool(done))
    return state, reports, dones


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def _trees_equal(a, b):
    la, lb = _leaves(a), _leaves(b)
    return (jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
            and len(la) == len(lb)
            and all(x.shape == y.shape and x.dtype == y.dtype
                    and np.array_equal(x, y) for x, y in zip(la, lb)))


def _signature(tree):
    return (jax.tree_util.tree_structure(tree),
            [(np.shape(x), np.asarray(x).dtype) for x in jax.tree_util.tree_leaves(tree)])


def _book_api(session):
    contract = Contract(contract_id=0, delivery_start_minutes=120,
                        delivery_end_minutes=135, gct_minutes=60,
                        status=ContractStatus.OPEN)
    return ContractBook(contract, session.product_config, session.market_config)


def _pnl(rows):
    return float(compute_agent_pnl(jnp.asarray(rows, jnp.int32).reshape(-1, 6),
                                   jnp.int32(AGENT_TRADER_ID)))


QTY_COL, BUYER_COL, SELLER_COL, PRICE_COL = 2, 3, 4, 1


# ===========================================================================
# 1. Public imports and kept names (plan §4.1, §4.2) — fast
# ===========================================================================
class TestPublicImports:
    def test_the_four_public_imports_of_section_4_1(self):
        from omiecid_simulator.omiecid.omie_session import Session as S, AgentOrder as A
        from omiecid_simulator.omiecid.contract_book import ContractBook as CB
        from omiecid_simulator.jaxomiecid.omie_data_loader import load_omie_orders
        from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder as TB
        assert all(x is not None for x in (S, A, CB, load_omie_orders, TB))
        assert callable(load_omie_orders)

    def test_kept_names_of_section_4_2(self):
        mod = importlib.import_module("omiecid_simulator.omiecid.omie_session")
        assert isinstance(mod.AGENT_TRADER_ID, int)
        assert isinstance(mod.AGENT_ORDER_ID_BASE, int)
        assert callable(getattr(Session, "reset"))
        assert callable(getattr(Session, "step_env"))
        assert callable(getattr(Session, "step"))
        assert callable(getattr(Session, "top_of_book"))
        assert OMIEContractState.__name__ == "OMIEContractState"

    def test_fill_report_and_top_of_book_fields_match_the_plan(self):
        assert FillReport._fields == (
            "submitted", "order_id", "submit", "passive_qty_signed",
            "passive_value_eur", "inventory_delta", "cash_delta_eur",
            "trade_rows", "trades_saturated")
        assert TopOfBook._fields == (
            "has_bid", "has_ask", "p_bid", "p_ask", "mid", "q_bid", "q_ask")
        assert set(SubmitResult._fields) >= {
            "accepted", "traded", "trade_qty", "trade_price", "settlement_value"}


# ===========================================================================
# 2. No RL surface (plan §4.4, REL-05 AC 4) — fast
# ===========================================================================
class TestNoRLSurface:
    RL_ATTRS = ("action_space", "observation_space", "reward", "rewards",
                "get_obs", "_get_obs", "_extract_action", "_time_to_gct",
                "_counterparty_top_of_book", "_validate_obs_step_scale",
                "agents", "observation")

    def test_session_has_no_base_class(self):
        assert Session.__bases__ == (object,)

    def test_session_class_has_no_rl_attributes(self):
        for name in self.RL_ATTRS:
            assert not hasattr(Session, name), name

    def test_session_instance_has_no_rl_attributes(self, session):
        for name in self.RL_ATTRS:
            assert not hasattr(session, name), name

    def test_one_participant_per_session(self):
        assert Session.num_agents == 1

    def test_states_module_has_no_observation_type(self):
        states = importlib.import_module("omiecid_simulator.omiecid.omie_states")
        assert not hasattr(states, "OMIEAgentObs")

    def test_session_module_has_no_spaces_or_multiagent_base(self):
        mod = importlib.import_module("omiecid_simulator.omiecid.omie_session")
        for name in ("spaces", "MultiAgentEnv", "OMIEAgentObs", "OMIEEnv"):
            assert not hasattr(mod, name), name

    @pytest.mark.parametrize("pkg", ["omiecid_simulator.jaxen",
                                     "omiecid_simulator.jaxrl"])
    def test_rl_packages_are_absent(self, pkg):
        assert importlib.util.find_spec(pkg) is None

    def test_reset_returns_the_state_only(self, session):
        out = session.reset(KEY)
        assert isinstance(out, OMIEContractState)


# ===========================================================================
# 3. AgentOrder (plan §4.4 interface) — fast
# ===========================================================================
EXPECTED_DTYPES = {
    "active": jnp.bool_, "side": jnp.int32, "price": jnp.float32,
    "quantity": jnp.float32, "exec_restriction": jnp.int32,
    "validity": jnp.int32, "expiry_time": jnp.int32,
    "peak_qty": jnp.float32, "ppd": jnp.float32,
}


class TestAgentOrder:
    def test_fields_are_economic_only(self):
        assert AgentOrder._fields == tuple(EXPECTED_DTYPES)
        for stamped in ("order_id", "agent_id", "timestamp", "contract_id"):
            assert stamped not in AgentOrder._fields

    def test_create_casts_python_scalars(self):
        o = AgentOrder.create(SELL, 50, 3)
        for f, dt in EXPECTED_DTYPES.items():
            v = getattr(o, f)
            assert np.shape(v) == (), f
            assert jnp.asarray(v).dtype == dt, f
        assert bool(o.active) is True
        assert float(o.price) == 50.0 and float(o.quantity) == 3.0
        assert int(o.exec_restriction) == NON

    def test_create_casts_numpy_wide_dtypes(self):
        o = AgentOrder.create(np.int64(BUY), np.float64(49.99), np.float64(1.5),
                              exec_restriction=np.int64(IOC))
        for f, dt in EXPECTED_DTYPES.items():
            assert jnp.asarray(getattr(o, f)).dtype == dt, f
        assert int(o.exec_restriction) == IOC

    def test_none_is_inactive_with_the_create_signature(self):
        n, c = AgentOrder.none(), AgentOrder.create(BUY, 50.0, 1.0)
        assert bool(n.active) is False
        assert _signature(n) == _signature(c)


# ===========================================================================
# 4. reset (plan §4.4) — reset compile is cheap
# ===========================================================================
class TestReset:
    def test_reset_starts_flat_at_got(self, session):
        s = session.reset(KEY)
        assert int(s.time) == 0
        assert int(s.step_counter) == 0
        assert np.all(np.asarray(s.agent_inventory) == 0.0)
        assert np.all(np.asarray(s.agent_cash) == 0.0)
        assert np.asarray(s.agent_cash).dtype == np.float32

    @SLOW
    def test_reset_book_is_empty(self, session):
        tob = session.top_of_book(session.reset(KEY))
        assert not bool(tob.has_bid) and not bool(tob.has_ask)
        for f in ("p_bid", "p_ask", "mid", "q_bid", "q_ask"):
            assert float(getattr(tob, f)) == 0.0, f


# ===========================================================================
# 5. Stamping and time priority (plan §4.4 "Stamping"; spec §5.2, §6.1)
# ===========================================================================
@SLOW
class TestStampingAndTimePriority:
    def test_agent_rests_behind_earlier_same_price_arrival(self, session):
        s0 = _start(session, "prio_behind")
        s1, r1, _ = session.step(KEY, s0, AgentOrder.create(SELL, 50.0, 2.0))
        assert bool(r1.submitted) and bool(r1.submit.accepted)
        assert not bool(r1.submit.traded)
        # Step 2: a background BUY 50 x2 must fill the EARLIER background seller.
        s2, r2, _ = session.step(KEY, s1, None)
        assert float(r2.passive_qty_signed) == pytest.approx(0.0, **EUR)
        assert float(r2.passive_value_eur) == pytest.approx(0.0, **EUR)
        tob = session.top_of_book(s2)
        assert bool(tob.has_ask) and float(tob.p_ask) == pytest.approx(50.0, **EUR)
        assert float(tob.q_ask) == pytest.approx(2.0, **EUR)   # the agent's order
        # Step 3: the next background BUY 50 x2 fills the agent at 50.
        _, r3, _ = session.step(KEY, s2, None)
        assert float(r3.passive_qty_signed) == pytest.approx(-2.0, **EUR)
        assert float(r3.passive_value_eur) == pytest.approx(2.0 * 50.0 * D_H, **EUR)

    def test_agent_rests_ahead_of_later_same_price_arrival(self, session):
        s0 = _start(session, "prio_ahead")
        s1, r1, _ = session.step(KEY, s0, AgentOrder.create(SELL, 50.0, 2.0))
        assert not bool(r1.submit.traded)
        # Step 2: background SELL 50 x2 arrives later, rests behind the agent.
        s2, r2, _ = session.step(KEY, s1, None)
        assert float(r2.passive_qty_signed) == 0.0
        # Step 3: background BUY 50 x2 fills the agent (time priority).
        _, r3, _ = session.step(KEY, s2, None)
        assert float(r3.passive_qty_signed) == pytest.approx(-2.0, **EUR)
        assert float(r3.passive_value_eur) == pytest.approx(25.0, **EUR)

    def test_stamp_is_post_background_arrival_counter(self, session):
        s0 = _start(session, "prio_behind")
        s_none, _, _ = session.step(KEY, s0, None)
        post_background_counter = int(s_none.book.ts_counter)
        s1, r1, _ = session.step(KEY, s0, AgentOrder.create(SELL, 50.0, 2.0))
        cb = _book_api(session)
        mine = cb.get_agent_orders(s1.book, AGENT_TRADER_ID)
        assert len(mine) == 1
        assert int(mine[0].timestamp) == post_background_counter
        others = cb.get_agent_orders(s1.book, BG)
        assert others and all(int(mine[0].timestamp) > int(o.timestamp) for o in others)

    def test_session_stamps_participant_id_and_order_id(self, session):
        s0 = _start(session, "prio_behind")
        s1, r1, _ = session.step(KEY, s0, AgentOrder.create(SELL, 50.0, 2.0))
        mine = _book_api(session).get_agent_orders(s1.book, AGENT_TRADER_ID)
        assert len(mine) == 1
        assert int(mine[0].agent_id) == AGENT_TRADER_ID
        assert int(mine[0].order_id) == int(r1.order_id) > 0

    def test_successive_submissions_get_increasing_stamps_and_distinct_ids(self, session):
        s0 = _start(session, "prio_ahead")
        s1, r1, _ = session.step(KEY, s0, AgentOrder.create(SELL, 55.0, 1.0))
        s2, r2, _ = session.step(KEY, s1, AgentOrder.create(SELL, 55.0, 1.0))
        assert int(r1.order_id) != int(r2.order_id)
        mine = {int(o.order_id): int(o.timestamp)
                for o in _book_api(session).get_agent_orders(s2.book, AGENT_TRADER_ID)}
        assert set(mine) == {int(r1.order_id), int(r2.order_id)}
        assert mine[int(r1.order_id)] < mine[int(r2.order_id)]

    def test_no_order_reports_order_id_zero_and_not_submitted(self, session):
        _, r, _ = session.step(KEY, _start(session, "prio_behind"), None)
        assert not bool(r.submitted)
        assert int(r.order_id) == 0
        assert not bool(r.submit.accepted) and not bool(r.submit.traded)

    def test_cancellation_between_steps_with_report_order_id(self, session):
        """Plan §4.4: cancellation between steps, eagerly, using report.order_id."""
        s0 = _start(session, "prio_ahead")
        s1, r1, _ = session.step(KEY, s0, AgentOrder.create(SELL, 50.0, 2.0))
        book = _book_api(session).cancel_order(s1.book, int(r1.order_id), SELL)
        s1 = s1.replace(book=book)
        assert _book_api(session).get_agent_orders(s1.book, AGENT_TRADER_ID) == []
        _, reps, _ = _run(session, s1, [None, None])
        assert float(reps[1].passive_qty_signed) == 0.0   # background seller filled


# ===========================================================================
# 6. FillReport arithmetic — aggressive fills (spec §6.2 maker price, §8)
# ===========================================================================
@pytest.fixture(scope="module")
def agg_state(session):
    s, _, _ = _run(session, _start(session, "agg"), [None] * 4)
    return s


@SLOW
class TestAggressiveFill:
    def test_buy_walks_two_ask_levels(self, session, agg_state):
        s, r, _ = session.step(KEY, agg_state, AgentOrder.create(BUY, 52.0, 2.0))
        value = (50.0 + 51.0) * 1.0 * D_H                  # 25.25 EUR
        assert bool(r.submitted) and bool(r.submit.accepted) and bool(r.submit.traded)
        assert float(r.submit.trade_qty) == pytest.approx(2.0, **EUR)
        assert float(r.submit.trade_price) == pytest.approx(50.5, **EUR)
        assert float(r.submit.settlement_value) == pytest.approx(value, **EUR)
        assert float(r.inventory_delta) == pytest.approx(2.0, **EUR)
        assert float(r.cash_delta_eur) == pytest.approx(-value, **EUR)
        assert float(r.passive_qty_signed) == 0.0 and float(r.passive_value_eur) == 0.0
        assert float(s.agent_inventory[0]) == pytest.approx(2.0, **EUR)
        assert float(s.agent_cash[0]) == pytest.approx(-value, **EUR)

    def test_buy_trade_rows_aggressive_leg(self, session, agg_state):
        _, r, _ = session.step(KEY, agg_state, AgentOrder.create(BUY, 52.0, 2.0))
        rows = np.asarray(r.trade_rows)
        assert rows.shape == (2, 6) and rows.dtype == np.int32
        q_tick = session.product_config.qty_tick
        assert rows[0, QTY_COL] == round(2.0 / q_tick)
        assert rows[0, BUYER_COL] == AGENT_TRADER_ID
        assert rows[0, SELLER_COL] != AGENT_TRADER_ID
        assert rows[1, QTY_COL] == 0                       # no passive leg
        assert _pnl(rows) == pytest.approx(float(r.cash_delta_eur), **EUR)

    def test_sell_hits_bid_cash_positive(self, session, agg_state):
        s, r, _ = session.step(KEY, agg_state, AgentOrder.create(SELL, 47.5, 3.0))
        value = 3.0 * 48.0 * D_H                           # maker price 48
        assert float(r.submit.trade_price) == pytest.approx(48.0, **EUR)
        assert float(r.inventory_delta) == pytest.approx(-3.0, **EUR)
        assert float(r.cash_delta_eur) == pytest.approx(value, **EUR)
        rows = np.asarray(r.trade_rows)
        assert rows[0, SELLER_COL] == AGENT_TRADER_ID
        assert rows[0, BUYER_COL] != AGENT_TRADER_ID
        assert rows[0, PRICE_COL] == encode_price(48.0, session.product_config)
        assert _pnl(rows) == pytest.approx(value, **EUR)
        assert float(s.agent_cash[0]) == pytest.approx(value, **EUR)

    def test_ioc_residual_is_cancelled(self, session, agg_state):
        s, r, _ = session.step(
            KEY, agg_state, AgentOrder.create(BUY, 50.5, 4.0, exec_restriction=IOC))
        assert float(r.submit.trade_qty) == pytest.approx(1.0, **EUR)
        assert float(r.cash_delta_eur) == pytest.approx(-50.0 * D_H, **EUR)
        tob = session.top_of_book(s)
        assert float(tob.p_bid) == pytest.approx(48.0, **EUR)  # no agent residual
        assert float(tob.p_ask) == pytest.approx(51.0, **EUR)

    def test_non_residual_rests_and_is_visible(self, session, agg_state):
        s, r, _ = session.step(KEY, agg_state, AgentOrder.create(BUY, 50.5, 4.0))
        assert float(r.submit.trade_qty) == pytest.approx(1.0, **EUR)
        assert float(r.inventory_delta) == pytest.approx(1.0, **EUR)
        tob = session.top_of_book(s)
        assert float(tob.p_bid) == pytest.approx(50.5, **EUR)
        assert float(tob.q_bid) == pytest.approx(3.0, **EUR)

    def test_non_crossing_order_rests_without_cash(self, session, agg_state):
        s, r, _ = session.step(KEY, agg_state, AgentOrder.create(BUY, 49.0, 1.0))
        assert bool(r.submit.accepted) and not bool(r.submit.traded)
        assert float(r.cash_delta_eur) == 0.0 and float(r.inventory_delta) == 0.0
        assert int(np.asarray(r.trade_rows)[:, QTY_COL].sum()) == 0
        tob = session.top_of_book(s)                      # anonymous book shows it
        assert float(tob.p_bid) == pytest.approx(49.0, **EUR)
        assert float(tob.q_bid) == pytest.approx(1.0, **EUR)

    def test_rejected_order_leaves_book_as_no_order(self, session, agg_state):
        s_rej, r, _ = session.step(KEY, agg_state, AgentOrder.create(BUY, 10000.0, 1.0))
        s_none, _, _ = session.step(KEY, agg_state, None)
        assert not bool(r.submit.accepted) and not bool(r.submit.traded)
        assert float(r.cash_delta_eur) == 0.0 and float(r.inventory_delta) == 0.0
        assert _trees_equal(s_rej.book, s_none.book)

    def test_negative_price_sell_cash_is_signed_value(self, session):
        s, _, _ = _run(session, _start(session, "neg"), [None, None])
        _, r, _ = session.step(KEY, s, AgentOrder.create(SELL, -20.0, 1.0))
        # Sell: +V with V = Q*p*d = 1 * (-10) * 0.25 = -2.5 (spec §8.1, §8.4).
        assert float(r.submit.trade_price) == pytest.approx(-10.0, **EUR)
        assert float(r.cash_delta_eur) == pytest.approx(-2.5, **EUR)
        assert float(r.inventory_delta) == pytest.approx(-1.0, **EUR)
        assert _pnl(r.trade_rows) == pytest.approx(-2.5, **EUR)

    def test_negative_price_buy_cash_is_minus_value(self, session):
        s, _, _ = _run(session, _start(session, "neg"), [None, None])
        _, r, _ = session.step(KEY, s, AgentOrder.create(BUY, -5.0, 1.0))
        # Buy: -V with V = 1 * (-8) * 0.25 = -2.0  =>  cash +2.0.
        assert float(r.submit.trade_price) == pytest.approx(-8.0, **EUR)
        assert float(r.cash_delta_eur) == pytest.approx(2.0, **EUR)
        assert float(r.inventory_delta) == pytest.approx(1.0, **EUR)
        assert _pnl(r.trade_rows) == pytest.approx(2.0, **EUR)


# ===========================================================================
# 7. Passive fills, both legs and cash consistency (plan §4.4 "Cash")
# ===========================================================================
MIXED_ORDERS = [
    AgentOrder.create(SELL, 50.0, 3.0),   # 1: rests at 50 (best ask)
    None,                                 # 2: bg BUY 50x2 -> passive -2 @50
    AgentOrder.create(BUY, 53.0, 1.0),    # 3: bg BUY 55x5 -> passive -1 @50; agent buys 1 @52
    AgentOrder.create(SELL, 44.0, 1.0),   # 4: bg BUY 45x2 rests; agent sells 1 @45
    AgentOrder.create(BUY, 46.0, 2.0),    # 5: bg SELL 44x1 hits bg bid; agent bid rests
    None,                                 # 6: bg SELL 46x2 -> passive +2 @46
]
MIXED_EXPECT = [  # (inventory_delta, cash_delta_eur)
    (0.0, 0.0),
    (-2.0, 2 * 50.0 * D_H),
    (-1.0 + 1.0, 1 * 50.0 * D_H - 1 * 52.0 * D_H),
    (-1.0, 1 * 45.0 * D_H),
    (0.0, 0.0),
    (2.0, -2 * 46.0 * D_H),
]


@pytest.fixture(scope="module")
def mixed_run(session):
    return _run(session, _start(session, "mixed"), MIXED_ORDERS)


@SLOW
class TestPassiveFillsAndCash:
    def test_per_step_inventory_and_cash_deltas(self, mixed_run):
        _, reps, _ = mixed_run
        for i, (rep, (inv, cash)) in enumerate(zip(reps, MIXED_EXPECT)):
            assert float(rep.inventory_delta) == pytest.approx(inv, **EUR), i
            assert float(rep.cash_delta_eur) == pytest.approx(cash, **EUR), i

    def test_passive_sell_fields(self, mixed_run):
        _, reps, _ = mixed_run
        r = reps[1]
        assert not bool(r.submitted)
        assert float(r.passive_qty_signed) == pytest.approx(-2.0, **EUR)
        assert float(r.passive_value_eur) == pytest.approx(25.0, **EUR)
        rows = np.asarray(r.trade_rows)
        assert rows[0, QTY_COL] == 0
        assert rows[1, QTY_COL] == 20
        assert rows[1, SELLER_COL] == AGENT_TRADER_ID
        assert rows[1, BUYER_COL] != AGENT_TRADER_ID

    def test_passive_buy_fields(self, mixed_run):
        _, reps, _ = mixed_run
        r = reps[5]
        assert float(r.passive_qty_signed) == pytest.approx(2.0, **EUR)
        assert float(r.passive_value_eur) == pytest.approx(-23.0, **EUR)
        rows = np.asarray(r.trade_rows)
        assert rows[1, BUYER_COL] == AGENT_TRADER_ID
        assert _pnl(rows) == pytest.approx(-23.0, **EUR)

    def test_step_with_both_legs_records_both_rows(self, mixed_run):
        _, reps, _ = mixed_run
        r = reps[2]
        rows = np.asarray(r.trade_rows)
        assert rows[0, QTY_COL] == 10 and rows[0, BUYER_COL] == AGENT_TRADER_ID
        assert rows[1, QTY_COL] == 10 and rows[1, SELLER_COL] == AGENT_TRADER_ID
        assert float(r.passive_qty_signed) == pytest.approx(-1.0, **EUR)
        assert float(r.submit.trade_price) == pytest.approx(52.0, **EUR)
        assert _pnl(rows) == pytest.approx(float(r.cash_delta_eur), **EUR)

    def test_each_report_rows_settle_to_its_cash_delta(self, mixed_run):
        _, reps, _ = mixed_run
        for i, r in enumerate(reps):
            assert _pnl(r.trade_rows) == pytest.approx(float(r.cash_delta_eur), **EUR), i

    def test_agent_cash_equals_compute_agent_pnl_over_trade_tape(self, mixed_run):
        s, reps, _ = mixed_run
        expected = sum(c for _, c in MIXED_EXPECT)          # 12.75 EUR
        assert float(s.agent_cash[0]) == pytest.approx(expected, **EUR)
        assert _pnl(s.agent_trades) == pytest.approx(float(s.agent_cash[0]), **EUR)

    def test_state_accumulates_report_deltas(self, mixed_run):
        s, reps, _ = mixed_run
        assert float(s.agent_cash[0]) == pytest.approx(
            sum(float(r.cash_delta_eur) for r in reps), **EUR)
        assert float(s.agent_inventory[0]) == pytest.approx(
            sum(float(r.inventory_delta) for r in reps), **EUR)
        assert float(s.agent_inventory[0]) == pytest.approx(-1.0, **EUR)


# ===========================================================================
# 8. Gate closure and after-done no-op (plan §2.1 "Gate closure", §4.4)
# ===========================================================================
@pytest.fixture(scope="module")
def done_run(session):
    return _run(session, _start(session, "done"), [None, None, None])


@SLOW
class TestAfterDone:
    def test_done_fires_when_clock_reaches_gct(self, done_run):
        s, _, dones = done_run
        assert dones == [False, False, True]
        assert int(s.time) >= 60

    @pytest.mark.parametrize("kind", ["crossing", "resting", "none_obj", "none"])
    def test_call_after_done_is_ignored(self, session, done_run, kind):
        s, _, _ = done_run
        order = {"crossing": AgentOrder.create(BUY, 60.0, 1.0),
                 "resting": AgentOrder.create(BUY, 40.0, 1.0),
                 "none_obj": AgentOrder.none(),
                 "none": None}[kind]
        s2, r, done = session.step(KEY, s, order)
        assert bool(done)
        assert _trees_equal(s2, s)                        # whole state incl. tapes
        assert np.array_equal(np.asarray(s2.agent_trades), np.asarray(s.agent_trades))
        assert np.array_equal(np.asarray(s2.book.lob.trades), np.asarray(s.book.lob.trades))
        for leaf in _leaves(r):
            assert not np.any(leaf), r                    # empty report

    def test_repeated_calls_after_done_stay_fixed(self, session, done_run):
        s, _, _ = done_run
        cur = s
        for _ in range(3):
            cur, _, done = session.step_env(KEY, cur, AgentOrder.create(BUY, 60.0, 1.0))
            assert bool(done)
        assert _trees_equal(cur, s)


# ===========================================================================
# 9. top_of_book vs the anonymous book (plan §4.4; spec §7.1)
# ===========================================================================
@SLOW
class TestTopOfBook:
    def test_two_sided_book_with_foreign_iceberg(self, session):
        s, _, _ = _run(session, _start(session, "tob"), [None] * 6)
        tob = session.top_of_book(s)
        assert bool(tob.has_bid) and bool(tob.has_ask)
        assert float(tob.p_bid) == pytest.approx(48.0, **EUR)
        assert float(tob.q_bid) == pytest.approx(5.0, **EUR)       # 3 + 2 aggregated
        assert float(tob.p_ask) == pytest.approx(51.0, **EUR)
        assert float(tob.q_ask) == pytest.approx(4.0 + 2.0, **EUR)  # iceberg peak only
        assert float(tob.mid) == pytest.approx(49.5, **EUR)

    def test_matches_contract_book_anonymous_view(self, session):
        s, _, _ = _run(session, _start(session, "tob"), [None] * 6)
        tob = session.top_of_book(s)
        bids, asks = _book_api(session).get_anonymous_lob(s.book)
        best_bid = max(bids, key=lambda x: x[0])
        best_ask = min(asks, key=lambda x: x[0])
        assert float(tob.p_bid) == pytest.approx(best_bid[0], **EUR)
        assert float(tob.q_bid) == pytest.approx(best_bid[1], **EUR)
        assert float(tob.p_ask) == pytest.approx(best_ask[0], **EUR)
        assert float(tob.q_ask) == pytest.approx(best_ask[1], **EUR)

    def test_one_sided_book_has_zero_mid(self, session):
        s, _, _ = _run(session, _start(session, "tob"), [None] * 3)
        tob = session.top_of_book(s)
        assert bool(tob.has_bid) and not bool(tob.has_ask)
        assert float(tob.p_ask) == 0.0 and float(tob.q_ask) == 0.0
        assert float(tob.mid) == 0.0

    def test_output_dtypes(self, session):
        tob = session.top_of_book(session.reset(KEY))
        assert jnp.asarray(tob.has_bid).dtype == jnp.bool_
        for f in ("p_bid", "p_ask", "mid", "q_bid", "q_ask"):
            assert jnp.asarray(getattr(tob, f)).dtype == jnp.float32, f
            assert np.shape(getattr(tob, f)) == ()

    def test_choosing_a_tape_selects_its_flow(self, session):
        s_neg, _, _ = _run(session, _start(session, "neg"), [None])
        s_agg, _, _ = _run(session, _start(session, "agg"), [None])
        t_neg, t_agg = session.top_of_book(s_neg), session.top_of_book(s_agg)
        assert bool(t_neg.has_bid) and float(t_neg.p_bid) == pytest.approx(-10.0, **EUR)
        assert bool(t_agg.has_ask) and float(t_agg.p_ask) == pytest.approx(50.0, **EUR)


# ===========================================================================
# 10. One compiled program; jit / vmap shape-dtype stability (plan §4.4)
# ===========================================================================
@SLOW
class TestCompileAndShapes:
    def test_none_empty_and_real_orders_share_one_program(self, session, agg_state):
        session.step(KEY, agg_state, AgentOrder.create(BUY, 52.0, 2.0))
        before = Session.step_env._cache_size()
        session.step(KEY, agg_state, None)
        session.step(KEY, agg_state, AgentOrder.none())
        session.step(KEY, agg_state, AgentOrder.create(SELL, 47.5, 3.0,
                                                       exec_restriction=IOC))
        # An uncast AgentOrder built from Python scalars: step casts it.
        session.step(KEY, agg_state, AgentOrder(active=True, side=BUY,
                                                price=49, quantity=1))
        assert Session.step_env._cache_size() == before

    def test_uncast_order_matches_create(self, session, agg_state):
        a = session.step(KEY, agg_state, AgentOrder(True, BUY, 52, 2))
        b = session.step(KEY, agg_state, AgentOrder.create(BUY, 52.0, 2.0))
        assert _trees_equal(a, b)

    def test_report_signature_is_order_independent(self, session, agg_state):
        _, r_none, d_none = session.step(KEY, agg_state, None)
        _, r_real, d_real = session.step(KEY, agg_state, AgentOrder.create(BUY, 52.0, 2.0))
        assert _signature(r_none) == _signature(r_real)
        assert np.shape(d_none) == () and jnp.asarray(d_none).dtype == jnp.bool_

    def test_state_signature_is_carry_stable(self, session, agg_state):
        s0 = session.reset(KEY)
        s1, _, _ = session.step(KEY, agg_state, AgentOrder.create(BUY, 52.0, 2.0))
        assert _signature(s1) == _signature(s0)

    def test_report_dtypes(self, session, agg_state):
        _, r, _ = session.step(KEY, agg_state, AgentOrder.create(BUY, 52.0, 2.0))
        assert jnp.asarray(r.submitted).dtype == jnp.bool_
        assert jnp.asarray(r.trades_saturated).dtype == jnp.bool_
        assert jnp.issubdtype(jnp.asarray(r.order_id).dtype, jnp.integer)
        for f in ("passive_qty_signed", "passive_value_eur",
                  "inventory_delta", "cash_delta_eur"):
            assert jnp.asarray(getattr(r, f)).dtype == jnp.float32, f
            assert np.shape(getattr(r, f)) == (), f
        assert np.asarray(r.trade_rows).shape == (2, 6)
        assert np.asarray(r.trade_rows).dtype == np.int32

    def test_vmapped_step_matches_eager_per_element(self, session):
        """Batched lanes run under an outer ``enable_x64`` (the engine's int64
        fill accumulator, OQ-18; same convention as the REL-03 parity harness)."""
        with jax.experimental.enable_x64():
            self._check_vmapped_step(session)

    @pytest.mark.xfail(strict=True, reason=(
        "AUDIT FINDING: plan section 4.4 calls step_env 'vmappable' with no "
        "precondition, but a plain jax.vmap (x64 off) fails to lower: the "
        "engine's extract_fills int64 accumulator (trade_extraction.py) needs "
        "an outer enable_x64. Inherited from the research engine (OQ-18), not "
        "introduced by REL-05; the precondition must be documented or fixed."))
    def test_plain_vmap_without_x64_context(self, session):
        self._check_vmapped_step(session)

    @staticmethod
    def _check_vmapped_step(session):
        names = ["agg", "neg"]   # row 2 keeps a 50 ask / a -10 bid live
        orders = [AgentOrder.create(BUY, 51.0, 1.0), AgentOrder.create(SELL, -20.0, 1.0)]
        starts = [_start(session, n) for n in names]
        stack = lambda xs: jax.tree_util.tree_map(lambda *a: jnp.stack(a), *xs)
        # Build the batched state via one step each (book needs the first row).
        eager_pre = [session.step(KEY, s, None)[0] for s in starts]
        keys = jnp.stack([KEY, KEY])
        out = jax.vmap(session.step_env)(keys, stack(eager_pre), stack(orders))
        eager = [session.step_env(KEY, s, o) for s, o in zip(eager_pre, orders)]
        for i in range(2):
            elem = jax.tree_util.tree_map(lambda x: x[i], out)
            assert _trees_equal(elem, eager[i]), names[i]
        b_state, b_rep, b_done = out
        assert np.shape(b_done) == (2,)
        assert np.asarray(b_rep.trade_rows).shape == (2, 2, 6)
        assert float(b_rep.cash_delta_eur[0]) == pytest.approx(-50.0 * D_H, **EUR)
        assert float(b_rep.cash_delta_eur[1]) == pytest.approx(-2.5, **EUR)

    def test_vmapped_reset_has_leading_batch_axis(self, session):
        keys = jax.random.split(KEY, 3)
        states = jax.vmap(session.reset)(keys)
        single = session.reset(KEY)
        for a, b in zip(_leaves(states), _leaves(single)):
            assert a.shape == (3,) + b.shape and a.dtype == b.dtype


# ===========================================================================
# 11. Real clock: session_clock="got_relative" (plan §4.4 "Real data")
# ===========================================================================
GOT = 55_200


@pytest.fixture(scope="module")
def real_clock_session():
    tb = TapeBuilder(first_timestamp=GOT + 30, timestamp_step=60)
    tb.add(side=SELL, price_eur=50.0, qty_mw=2.0, agent_id=BG)   # minute 0
    tb.add(side=BUY, price_eur=48.0, qty_mw=1.0, agent_id=BG)    # minute 1
    tb.add(side=SELL, price_eur=51.0, qty_mw=1.0, agent_id=BG)   # minute 2
    tapes = stack_tapes([tb.build(pad_extra=0)], pad_extra=8)
    return Session({"tapes": tapes, "orders_per_step": 1,
                    "session_clock": "got_relative", "got_seconds": GOT,
                    "gct_minutes_from_got": 60})


class TestRealClock:
    def test_unknown_clock_is_rejected(self):
        tapes = stack_tapes([_tape(SCENARIOS["agg"])])
        with pytest.raises(ValueError):
            Session({"tapes": tapes, "session_clock": "wallclock"})

    def test_synthetic_frame_tape_is_rejected_under_got_relative(self):
        tapes = stack_tapes([_tape(SCENARIOS["agg"])])          # ts 1..4 < GOT
        with pytest.raises(ValueError):
            Session({"tapes": tapes, "orders_per_step": 1,
                     "session_clock": "got_relative", "got_seconds": GOT})

    @SLOW
    def test_clock_follows_tape_and_ends_on_exhaustion(self, real_clock_session):
        sess = real_clock_session
        s = sess.reset(KEY)
        times, dones = [], []
        for _ in range(4):
            s, _, d = sess.step(KEY, s, None)
            times.append(int(s.time))
            dones.append(bool(d))
        assert times[:3] == [0, 1, 2]
        assert dones == [False, False, False, True]
        assert times[3] >= 60

    @SLOW
    def test_participant_trades_under_real_clock(self, real_clock_session):
        sess = real_clock_session
        s, r, done = sess.step(KEY, sess.reset(KEY), AgentOrder.create(BUY, 51.0, 1.0))
        assert not bool(done)
        assert float(r.submit.trade_price) == pytest.approx(50.0, **EUR)
        assert float(r.cash_delta_eur) == pytest.approx(-12.5, **EUR)
        assert float(s.agent_inventory[0]) == pytest.approx(1.0, **EUR)
        assert _pnl(s.agent_trades) == pytest.approx(float(s.agent_cash[0]), **EUR)

    @SLOW
    def test_after_done_is_ignored_under_real_clock(self, real_clock_session):
        sess = real_clock_session
        s, _, _ = _run(sess, sess.reset(KEY), [None] * 4)
        s2, r, done = sess.step(KEY, s, AgentOrder.create(BUY, 60.0, 1.0))
        assert bool(done) and _trees_equal(s2, s)
        assert all(not np.any(x) for x in _leaves(r))


# ===========================================================================
# 12. Trade-buffer saturation flag (plan §4.4 "Trade-buffer saturation")
# ===========================================================================
@SLOW
class TestTradesSaturated:
    def test_flag_rises_when_buffer_is_full(self):
        rows = []
        for _ in range(3):                       # one background trade per pair
            rows += [_o(SELL, 50.0, 1.0), _o(BUY, 50.0, 1.0)]
        tapes = stack_tapes([_tape(rows)], pad_extra=8)
        sess = Session({"tapes": tapes, "orders_per_step": 1,
                        "tape_seconds_span": SPAN,
                        "market_config": OMIEMarketConfig(nTrades=3)})
        _, reps, _ = _run(sess, sess.reset(KEY), [None] * 6)
        flags = [bool(r.trades_saturated) for r in reps]
        assert flags[:5] == [False] * 5          # 0, 0, 1, 1, 2 trades stored
        assert flags[5] is True                  # 3 trades = every row filled
