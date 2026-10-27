"""Locked contract tests for the public ``Session`` (story REL-04, for REL-05).

Source of truth: ``docs/release/release-plan.md`` §4.4 (interface block,
"Behaviour", "Limitations") and §5.3 (``AgentOrder``, ``FillReport``,
``TopOfBook``, ``Session.step``, ``step_env`` additions); acceptance criteria
1-7 of card REL-04 in ``docs/release/release-backlog.md``.

Public surface exercised (plan §4.1 / §4.4)::

    from omiecid_simulator.omiecid.omie_session import Session, AgentOrder
    omie_session.FillReport, omie_session.TopOfBook   (module attributes)
    Session(config); Session.reset(key) -> OMIEContractState
    Session.step_env(key, state, order) -> (state, FillReport, done)
    Session.step(key, state, order=None) -> (state, FillReport, done)
    Session.top_of_book(state) -> TopOfBook

Interpretation choices (where §4.4 / §5.3 do not name something):

  I1  Config constructor. ``Session(config)`` takes the existing config dict;
      only kept mechanics keys are used here: ``tapes`` (an
      ``(n_tapes, n_rows, 13)`` int32 array), ``orders_per_step`` and
      ``market_config`` (``OMIEMarketConfig``). No RL key is passed.
  I2  Tape selection uses ``state.replace(tape_idx=i)`` (§4.4 "Choosing a
      tape").
  I3  Eager cancel entry point. §4.4 says "between steps, eagerly, using
      ``report.order_id``" without naming a Session method. The minimal
      reading is the existing public book API:
      ``ContractBook(...).cancel_order(state.book, int(report.order_id), side)``
      then ``state.replace(book=...)``. No new Session method is required.
  I4  "Empty report" (after ``done``) = every leaf of the ``FillReport``
      (including ``order_id``, ``trades_saturated`` and every ``submit``
      field) is zero / False.
  I5  "Stamping" is checked through the book: the agent's resting row has
      ``OID == report.order_id``, ``TID == AGENT_TRADER_ID`` and
      ``SEC ==`` the book's arrival counter after the background replay. That
      counter is observed black-box as ``book.ts_counter`` after a no-order
      step from the same state (the no-order step's book is the
      post-background book). "User-supplied ids are ignored" is checked
      structurally: ``AgentOrder`` has only the economic fields of §4.4 and
      ``AgentOrder.create`` rejects ``order_id`` / ``agent_id`` /
      ``timestamp`` / ``contract_id`` keywords. The contract stamp is not
      stored per row in the book; it is checked as acceptance of the order.
  I6  Saturation. "Full" means every one of the ``nTrades`` rows of the
      book's trade buffer holds a trade; the flag is checked both at exactly
      full and after an overwrite, and is False below capacity.
  I7  "One compiled program" is counted with JAX's lowering event
      (``/jax/core/compile/jaxpr_to_mlir_module_duration``), filtering on
      programs whose name contains ``step`` (so tiny eager primitives are
      not counted). After one warm-up call, ``None``, ``AgentOrder.none()``,
      a different real order and an ``AgentOrder`` built directly with
      Python scalars (``Session.step`` "casts types") lower nothing new.
  I8  ``TopOfBook`` on an empty side follows the existing internal helper
      that §5.3 says is reused: ``has_* = False``, ``p_* = 0.0``,
      ``q_* = 0.0``, and ``mid = 0.0`` unless both sides are present.
      Prices in these tests sit on whole ticks so ``mid`` is exact.
  I9  ``FillReport`` content checked here (§4.4 "Cash"): ``inventory_delta``
      is signed MW (+buy), ``cash_delta_eur`` is realized, unscaled EUR
      (``-Q*p*d`` on a buy), ``passive_qty_signed`` is signed MW (+buy).
      ``passive_value_eur``'s sign is not pinned.
  I10 vmap-safety is checked with the vmapped ``step_env`` call wrapped in
      ``jax.experimental.enable_x64()``: the repository's existing convention
      for vmapping the engine (``extract_fills`` accumulates in int64; without
      the outer context the vmapped lowering fails, see
      ``test_omie_env.py::TestOneCompilePreservationUnderVmap``). REL-05
      keeps the mechanics unchanged, so the same caller context applies.
      ``trade_rows`` must still come out int32 inside that context.

Device policy: CPU. Every class that compiles ``reset`` / ``step_env`` /
``top_of_book`` carries ``@pytest.mark.slow`` (slow: one jitted Session
compile, > 1 s). All such tests share one Session configuration so a worker
compiles it once (the compile-count test uses a distinct one on purpose).

Run::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python3 -m pytest \
        tests/omiecid/test_session.py -n0 -q
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

import jax
import jax.experimental  # enable_x64 for the vmap leg (I10)
import jax.numpy as jnp

from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.omie_states import OMIEContractState
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    Status,
    Validity,
    make_order,
)

# The public interface (plan §4.1). Imported defensively so that the file
# collects on a tree that does not implement it yet; every test then FAILS
# (not skips) with the import error via the autouse fixture below.
try:
    from omiecid_simulator.omiecid import omie_session as _om
    from omiecid_simulator.omiecid.omie_session import (  # noqa: F401
        AGENT_TRADER_ID,
        AgentOrder,
        Session,
    )
    _IMPORT_ERROR = None
except ImportError as _exc:  # pragma: no cover - RED path before REL-05
    _IMPORT_ERROR = _exc
    _om = None
    AgentOrder = None
    Session = None
    AGENT_TRADER_ID = 1_000_000


@pytest.fixture(autouse=True)
def _require_public_interface():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"public Session interface (plan §4.1/§4.4) missing: "
                    f"{_IMPORT_ERROR!r}")


# ---------------------------------------------------------------------------
# Small synthetic world
# ---------------------------------------------------------------------------
_PC = OMIEProductConfig()
_MC = OMIEMarketConfig(book_depth=10, nTrades=4, max_icebergs=16)
_OPS = 4                       # orders_per_step
_KEY = jax.random.PRNGKey(0)

BUY = int(Side.BUY)
SELL = int(Side.SELL)
NON = int(ExecRestriction.NON)
IOC = int(ExecRestriction.IOC)

_B_P = OrderSideFeat.P.value
_B_Q = OrderSideFeat.Q.value
_B_OID = OrderSideFeat.OID.value
_B_TID = OrderSideFeat.TID.value
_B_SEC = OrderSideFeat.SEC.value

_PAD = [0, 0, 0, 0, -1, 0, -1, NON, int(Validity.GFS), 0,
        int(Status.ACTIVE), 0, 0]
_BG = 777                       # anonymous background trader id
_MW = 1e-4                      # MW tolerance
_EUR = 1e-3                     # EUR tolerance


def _pint(price: float) -> int:
    return int(round((price - _PC.price_min) / _PC.price_tick))


def _qint(qty: float) -> int:
    return int(round(qty / _PC.qty_tick))


def _r(side, price, qty, oid, ts, er=NON):
    return [0, int(side), _pint(price), _qint(qty), int(oid), _BG, int(ts),
            int(er), int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0]


def _tape(slice2, slice3):
    """20 rows = 5 steps of 4. Step 0: BUY 49.00 x5 (101), SELL 51.00 x5 (102).
    Step 1: BUY 49.00 x5 (103). Steps 2-3: tape specific. Step 4: a far BUY
    at the max timestamp, so ``done`` fires on step 4 = the last step."""
    rows = ([_r(BUY, 49.0, 5.0, 101, 10), _r(SELL, 51.0, 5.0, 102, 20),
             _PAD, _PAD,
             _r(BUY, 49.0, 5.0, 103, 30), _PAD, _PAD, _PAD]
            + slice2 + slice3
            + [_r(BUY, 1.0, 0.1, 199, 1000), _PAD, _PAD, _PAD])
    assert len(rows) == 20
    return rows


# Tape 0: step 2 a SELL IOC 49.00 x10 takes exactly 101 + 103 (2 trades).
T_QUEUE = 0
# Tape 1: step 2 a SELL IOC 49.00 x12 takes 101 + 103 and 2 MW more.
T_OVER = 1
# Tape 2: step 2 makes 4 trades (buffer exactly full at nTrades=4), step 3
# one more (overwrite).
T_SAT = 2
_TAPES = np.asarray([
    _tape([_r(SELL, 49.0, 10.0, 104, 40, IOC), _PAD, _PAD, _PAD], [_PAD] * 4),
    _tape([_r(SELL, 49.0, 12.0, 104, 40, IOC), _PAD, _PAD, _PAD], [_PAD] * 4),
    _tape([_r(SELL, 51.0, 1.0, 201, 40), _r(SELL, 51.0, 1.0, 202, 41),
           _r(SELL, 51.0, 1.0, 203, 42), _r(BUY, 51.0, 8.0, 204, 43, IOC)],
          [_r(SELL, 51.0, 1.0, 205, 50), _r(BUY, 51.0, 1.0, 206, 51, IOC),
           _PAD, _PAD]),
], dtype=np.int32)
DONE_STEP = 4


def _config(tapes=_TAPES):
    return {"tapes": tapes, "orders_per_step": _OPS, "market_config": _MC}


@functools.lru_cache(maxsize=None)
def _session():
    return Session(_config())


def _book_api():
    contract = Contract(contract_id=0, delivery_start_minutes=120,
                        delivery_end_minutes=135, gct_minutes=60,
                        status=ContractStatus.OPEN)
    return ContractBook(contract, _PC, _MC)


def _passive_buy():
    """BUY 49.00 x3 NON: rests behind 101/103 (best ask is 51.00)."""
    return AgentOrder.create(side=BUY, price=49.0, quantity=3.0)


def _cross_buy():
    """BUY 51.00 x1 IOC: takes 1 MW of the resting ask 102."""
    return AgentOrder.create(side=BUY, price=51.0, quantity=1.0,
                             exec_restriction=IOC)


def _start(tape_idx):
    state = _session().reset(_KEY)
    return state.replace(tape_idx=jnp.int32(tape_idx))


def _advance(state, n):
    """n no-order steps."""
    for _ in range(n):
        state, _rep, _done = _session().step(_KEY, state, None)
    return state


def _live(side_arr):
    a = np.asarray(side_arr)
    return a[(a[:, _B_OID] > 0) & (a[:, _B_Q] > 0)]


def _rows_with_oid(side_arr, oid):
    a = _live(side_arr)
    return a[a[:, _B_OID] == int(oid)]


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def _trees_equal(a, b):
    if jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b):
        return False
    return all(x.shape == y.shape and np.array_equal(x, y)
               for x, y in zip(_leaves(a), _leaves(b)))


def _signature(tree):
    return (jax.tree_util.tree_structure(tree),
            [(x.shape, x.dtype) for x in _leaves(tree)])


# ---------------------------------------------------------------------------
# Public type shapes (fast: no compile)
# ---------------------------------------------------------------------------
_AGENT_ORDER_FIELDS = {"active", "side", "price", "quantity",
                       "exec_restriction", "validity", "expiry_time",
                       "peak_qty", "ppd"}
_FILL_REPORT_FIELDS = {"submitted", "order_id", "submit", "passive_qty_signed",
                       "passive_value_eur", "inventory_delta", "cash_delta_eur",
                       "trade_rows", "trades_saturated"}
_TOB_FIELDS = {"has_bid", "has_ask", "p_bid", "p_ask", "mid", "q_bid", "q_ask"}
_AGENT_ORDER_DTYPES = {
    "active": jnp.bool_, "side": jnp.int32, "price": jnp.float32,
    "quantity": jnp.float32, "exec_restriction": jnp.int32,
    "validity": jnp.int32, "expiry_time": jnp.int32,
    "peak_qty": jnp.float32, "ppd": jnp.float32,
}


class TestPublicTypes:
    def test_agent_order_has_only_economic_fields(self):
        assert set(AgentOrder._fields) == _AGENT_ORDER_FIELDS

    def test_fill_report_fields(self):
        assert set(_om.FillReport._fields) == _FILL_REPORT_FIELDS

    def test_top_of_book_fields(self):
        assert set(_om.TopOfBook._fields) == _TOB_FIELDS


class TestAgentOrderCreate:
    def test_create_casts_every_field_to_its_dtype(self):
        order = AgentOrder.create(side=1, price=50, quantity=3)
        for name, dtype in _AGENT_ORDER_DTYPES.items():
            leaf = getattr(order, name)
            assert hasattr(leaf, "dtype"), f"{name} is not an array"
            assert leaf.dtype == jnp.dtype(dtype), (name, leaf.dtype)
            assert leaf.shape == (), (name, leaf.shape)
        assert bool(order.active) is True
        assert int(order.side) == 1
        assert float(order.price) == 50.0
        assert float(order.quantity) == pytest.approx(3.0)

    def test_create_defaults_are_non_gfs(self):
        order = AgentOrder.create(side=BUY, price=50.0, quantity=1.0)
        assert int(order.exec_restriction) == NON
        assert int(order.validity) == int(Validity.GFS)
        assert int(order.expiry_time) == 0
        assert float(order.peak_qty) == 0.0
        assert float(order.ppd) == 0.0

    def test_create_casts_wide_numpy_inputs_down(self):
        order = AgentOrder.create(side=np.int64(0), price=np.float64(49.25),
                                  quantity=np.float64(2.5),
                                  exec_restriction=ExecRestriction.IOC)
        assert order.side.dtype == jnp.int32
        assert order.price.dtype == jnp.float32
        assert order.quantity.dtype == jnp.float32
        assert order.exec_restriction.dtype == jnp.int32
        assert int(order.exec_restriction) == IOC

    def test_none_is_inactive_with_the_same_signature_as_create(self):
        none = AgentOrder.none()
        real = AgentOrder.create(side=SELL, price=51.0, quantity=1.0)
        assert bool(none.active) is False
        assert _signature(none) == _signature(real)

    @pytest.mark.parametrize("kw", ["order_id", "agent_id", "timestamp",
                                    "contract_id"])
    def test_create_rejects_session_stamped_fields(self, kw):
        with pytest.raises(TypeError):
            AgentOrder.create(side=BUY, price=49.0, quantity=1.0, **{kw: 7})


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestReset:
    def test_reset_returns_the_state_only(self):
        out = _session().reset(_KEY)
        assert not isinstance(out, tuple)
        assert isinstance(out, OMIEContractState)

    def test_reset_state_starts_flat_at_got(self):
        state = _session().reset(_KEY)
        assert int(state.step_counter) == 0
        assert int(state.tape_cursor) == 0
        assert 0 <= int(state.tape_idx) < _TAPES.shape[0]
        assert float(np.asarray(state.agent_inventory)[0]) == 0.0
        assert float(np.asarray(state.agent_cash)[0]) == 0.0


# ---------------------------------------------------------------------------
# (1) Stamping
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestStamping:
    def _submit_at_step1(self, tape_idx):
        s1 = _advance(_start(tape_idx), 1)
        s2_none, _r0, _d0 = _session().step(_KEY, s1, None)
        counter = int(s2_none.book.ts_counter)     # post-background counter
        s2, rep, _done = _session().step(_KEY, s1, _passive_buy())
        return s1, s2, rep, counter

    def test_resting_row_carries_report_order_id_and_agent_participant_id(self):
        _s1, s2, rep, _c = self._submit_at_step1(T_QUEUE)
        assert bool(rep.submitted)
        assert bool(rep.submit.accepted)          # contract stamped: accepted
        rows = _rows_with_oid(s2.book.lob.bids, rep.order_id)
        assert rows.shape[0] == 1, "agent order must rest under report.order_id"
        assert int(rows[0][_B_TID]) == AGENT_TRADER_ID
        assert int(rows[0][_B_P]) == _pint(49.0)
        assert int(rows[0][_B_Q]) == _qint(3.0)
        assert int(rep.order_id) > 0

    def test_order_id_does_not_collide_with_background_ids(self):
        _s1, _s2, rep, _c = self._submit_at_step1(T_QUEUE)
        background = set(int(x) for x in _TAPES[:, :, 4].ravel() if x > 0)
        assert int(rep.order_id) not in background

    def test_timestamp_is_the_post_background_arrival_counter(self):
        _s1, s2, rep, counter = self._submit_at_step1(T_QUEUE)
        row = _rows_with_oid(s2.book.lob.bids, rep.order_id)[0]
        assert int(row[_B_SEC]) == counter

    def test_timestamp_is_behind_every_earlier_same_price_arrival(self):
        _s1, s2, rep, _c = self._submit_at_step1(T_QUEUE)
        bids = _live(s2.book.lob.bids)
        agent = bids[bids[:, _B_OID] == int(rep.order_id)][0]
        earlier = bids[(bids[:, _B_P] == _pint(49.0))
                       & (bids[:, _B_OID] != int(rep.order_id))]
        assert set(int(x) for x in earlier[:, _B_OID]) == {101, 103}
        assert np.all(earlier[:, _B_SEC] < agent[_B_SEC])

    def test_order_queues_behind_earlier_arrivals_not_filled_first(self):
        # SELL x10 at step 2 is exactly 101 + 103: the agent is not touched.
        _s1, s2, rep, _c = self._submit_at_step1(T_QUEUE)
        s3, rep3, _d = _session().step(_KEY, s2, None)
        assert float(rep3.passive_qty_signed) == pytest.approx(0.0, abs=_MW)
        rows = _rows_with_oid(s3.book.lob.bids, rep.order_id)
        assert rows.shape[0] == 1 and int(rows[0][_B_Q]) == _qint(3.0)
        assert _rows_with_oid(s3.book.lob.bids, 101).shape[0] == 0
        assert _rows_with_oid(s3.book.lob.bids, 103).shape[0] == 0

    def test_order_fills_after_earlier_arrivals_are_exhausted(self):
        # SELL x12 at step 2: 101 + 103 take 10, the agent gets the last 2.
        _s1, s2, rep, _c = self._submit_at_step1(T_OVER)
        s3, rep3, _d = _session().step(_KEY, s2, None)
        assert float(rep3.passive_qty_signed) == pytest.approx(2.0, abs=_MW)
        rows = _rows_with_oid(s3.book.lob.bids, rep.order_id)
        assert rows.shape[0] == 1 and int(rows[0][_B_Q]) == _qint(1.0)

    def test_identical_economic_orders_get_distinct_ids_and_increasing_stamps(self):
        _s1, s2, rep_a, _c = self._submit_at_step1(T_QUEUE)
        s3, rep_b, _d = _session().step(_KEY, s2, _passive_buy())
        assert int(rep_a.order_id) != int(rep_b.order_id)
        row_a = _rows_with_oid(s3.book.lob.bids, rep_a.order_id)
        row_b = _rows_with_oid(s3.book.lob.bids, rep_b.order_id)
        assert row_a.shape[0] == 1 and row_b.shape[0] == 1
        assert int(row_b[0][_B_TID]) == AGENT_TRADER_ID
        assert int(row_b[0][_B_SEC]) > int(row_a[0][_B_SEC])


# ---------------------------------------------------------------------------
# (2) After done
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestAfterDone:
    @staticmethod
    def _to_done():
        s = _advance(_start(T_QUEUE), DONE_STEP)
        s_done, rep, done = _session().step(_KEY, s, _cross_buy())
        return s, s_done, rep, done

    def test_done_fires_at_gate_closure_with_a_trade_on_the_last_step(self):
        _s, s_done, rep, done = self._to_done()
        assert bool(done)
        assert bool(rep.submit.traded)

    def test_not_done_before_gate_closure_and_the_call_acts(self):
        s = _advance(_start(T_QUEUE), DONE_STEP - 1)
        s_next, rep, done = _session().step(_KEY, s, _cross_buy())
        assert not bool(done)
        assert bool(rep.submitted) and bool(rep.submit.traded)
        assert int(s_next.step_counter) == int(s.step_counter) + 1
        assert not _trees_equal(s, s_next)

    @pytest.mark.parametrize("order_kind", ["real", "none"])
    def test_call_after_done_returns_the_same_state(self, order_kind):
        _s, s_done, _rep, _done = self._to_done()
        order = _cross_buy() if order_kind == "real" else None
        s_after, _rep2, done2 = _session().step(_KEY, s_done, order)
        assert bool(done2)
        assert _trees_equal(s_after, s_done)

    def test_call_after_done_returns_an_empty_report(self):
        _s, s_done, _rep, _done = self._to_done()
        _s_after, rep2, _d = _session().step(_KEY, s_done, _cross_buy())
        for leaf in _leaves(rep2):
            assert not np.any(leaf), f"non-empty report leaf after done: {rep2}"

    def test_call_after_done_does_not_overwrite_the_trade_tapes(self):
        _s, s_done, _rep, _done = self._to_done()
        assert np.any(np.asarray(s_done.agent_trades)[-1] != 0), (
            "precondition: the done step wrote the last agent-trade row")
        s_after, _r2, _d = _session().step(_KEY, s_done, _cross_buy())
        np.testing.assert_array_equal(np.asarray(s_after.agent_trades),
                                      np.asarray(s_done.agent_trades))
        np.testing.assert_array_equal(np.asarray(s_after.book.lob.trades),
                                      np.asarray(s_done.book.lob.trades))

    def test_step_env_after_done_is_also_a_no_op(self):
        _s, s_done, _rep, _done = self._to_done()
        s_after, rep2, done2 = _session().step_env(_KEY, s_done, _cross_buy())
        assert bool(done2)
        assert _trees_equal(s_after, s_done)
        assert not any(np.any(x) for x in _leaves(rep2))


# ---------------------------------------------------------------------------
# (3) Saturation flag
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestTradesSaturated:
    def test_flag_set_when_the_buffer_is_exactly_full(self):
        s = _advance(_start(T_SAT), 2)
        _s3, rep, _d = _session().step(_KEY, s, None)
        assert bool(rep.trades_saturated)

    def test_flag_set_when_trades_overwrite_the_last_row(self):
        s = _advance(_start(T_SAT), 3)
        _s4, rep, _d = _session().step(_KEY, s, None)
        assert bool(rep.trades_saturated)

    def test_flag_clear_with_no_trades(self):
        s = _advance(_start(T_SAT), 1)
        _s2, rep, _d = _session().step(_KEY, s, None)
        assert not bool(rep.trades_saturated)

    def test_flag_clear_below_capacity(self):
        s = _advance(_start(T_QUEUE), 2)          # step 2: 2 of 4 rows used
        _s3, rep, _d = _session().step(_KEY, s, None)
        assert not bool(rep.trades_saturated)


# ---------------------------------------------------------------------------
# (4) top_of_book
# ---------------------------------------------------------------------------
def _eager_view(book_state):
    """Reference from the eager public book view."""
    bids, asks = _book_api().get_anonymous_lob(book_state)
    out = {"has_bid": bool(bids), "has_ask": bool(asks),
           "p_bid": 0.0, "p_ask": 0.0, "q_bid": 0.0, "q_ask": 0.0, "mid": 0.0}
    if bids:
        p = max(px for px, _q in bids)
        out["p_bid"] = p
        out["q_bid"] = sum(q for px, q in bids if px == p)
    if asks:
        p = min(px for px, _q in asks)
        out["p_ask"] = p
        out["q_ask"] = sum(q for px, q in asks if px == p)
    if bids and asks:
        out["mid"] = (out["p_bid"] + out["p_ask"]) / 2.0
    return out


def _with_orders(orders):
    """Session state whose book holds ``orders`` = [(side, price, qty), ...]."""
    state = _session().reset(_KEY)
    book, bstate = _book_api(), state.book
    for i, (side, price, qty) in enumerate(orders):
        bstate, res = book.submit_order(bstate, make_order(
            contract_id=0, side=Side(side), price=price, quantity=qty,
            order_id=500 + i, agent_id=_BG, timestamp=i + 1))
        assert bool(res.accepted)
    return state.replace(book=bstate)


def _assert_tob_matches(tob, ref):
    assert bool(tob.has_bid) == ref["has_bid"]
    assert bool(tob.has_ask) == ref["has_ask"]
    for name in ("p_bid", "p_ask", "mid"):
        assert float(getattr(tob, name)) == pytest.approx(ref[name], abs=_EUR), name
    for name in ("q_bid", "q_ask"):
        assert float(getattr(tob, name)) == pytest.approx(ref[name], abs=_MW), name


_TOB_CASES = {
    "empty": [],
    "bid_only": [(BUY, 49.0, 5.0), (BUY, 49.0, 3.0), (BUY, 48.0, 4.0)],
    "ask_only": [(SELL, 51.0, 2.0), (SELL, 52.0, 1.0)],
    "both": [(BUY, 49.0, 5.0), (BUY, 48.0, 1.0), (SELL, 51.0, 2.0),
             (SELL, 51.0, 0.5)],
}


@pytest.mark.slow
class TestTopOfBook:
    @pytest.mark.parametrize("case", sorted(_TOB_CASES))
    def test_top_of_book_equals_eager_book_view(self, case):
        state = _with_orders(_TOB_CASES[case])
        tob = _session().top_of_book(state)
        assert isinstance(tob, _om.TopOfBook)
        _assert_tob_matches(tob, _eager_view(state.book))

    def test_empty_book_reads_all_zero(self):
        tob = _session().top_of_book(_session().reset(_KEY))
        assert not bool(tob.has_bid) and not bool(tob.has_ask)
        for name in ("p_bid", "p_ask", "mid", "q_bid", "q_ask"):
            assert float(getattr(tob, name)) == 0.0, name

    def test_one_sided_book_has_zero_mid_and_zero_empty_side(self):
        tob = _session().top_of_book(_with_orders(_TOB_CASES["bid_only"]))
        assert bool(tob.has_bid) and not bool(tob.has_ask)
        assert float(tob.p_ask) == 0.0 and float(tob.q_ask) == 0.0
        assert float(tob.mid) == 0.0

    def test_two_sided_book_mid_and_touch_volume(self):
        tob = _session().top_of_book(_with_orders(_TOB_CASES["both"]))
        assert float(tob.mid) == pytest.approx(50.0, abs=_EUR)
        assert float(tob.q_ask) == pytest.approx(2.5, abs=_MW)  # aggregated
        assert float(tob.q_bid) == pytest.approx(5.0, abs=_MW)  # touch only

    def test_top_of_book_after_a_session_step(self):
        s1 = _advance(_start(T_QUEUE), 1)
        _assert_tob_matches(_session().top_of_book(s1), _eager_view(s1.book))

    @pytest.mark.jit
    def test_top_of_book_is_jit_and_vmap_safe(self):
        states = [_with_orders(_TOB_CASES[c]) for c in ("both", "empty")]
        jitted = jax.jit(_session().top_of_book)(states[0])
        _assert_tob_matches(jitted, _eager_view(states[0].book))
        batched = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), *states)
        vt = jax.vmap(_session().top_of_book)(batched)
        assert np.asarray(vt.has_bid).tolist() == [True, False]
        assert np.asarray(vt.mid).shape == (2,)


# ---------------------------------------------------------------------------
# (5) One compiled program
# ---------------------------------------------------------------------------
def _count_step_lowerings(fn):
    from jax._src import dispatch, monitoring

    names = []

    def _listener(event, _duration, **kwargs):
        if event == dispatch.JAXPR_TO_MLIR_MODULE_EVENT:
            names.append(str(kwargs.get("fun_name", "")))

    jax.monitoring.register_event_duration_secs_listener(_listener)
    try:
        fn()
    finally:
        monitoring._unregister_event_duration_listener_by_callback(_listener)
    return [n for n in names if "step" in n]


@pytest.mark.slow
class TestOneCompiledProgram:
    def test_none_empty_and_real_orders_share_one_program(self):
        # A distinct static configuration (extra tape) so the warm-up call is a
        # genuine first compile in this process.
        session = Session(_config(np.concatenate([_TAPES, _TAPES[:1]], axis=0)))
        state = session.reset(_KEY).replace(tape_idx=jnp.int32(T_QUEUE))

        def _run(order):
            out = session.step(_KEY, state, order)
            jax.block_until_ready(jax.tree_util.tree_leaves(out))

        warm = _count_step_lowerings(lambda: _run(_passive_buy()))
        assert len(warm) >= 1, "counter sanity: the first call must lower"

        raw = AgentOrder(active=True, side=SELL, price=51.0, quantity=2.0)
        later = _count_step_lowerings(lambda: (
            _run(None),
            _run(AgentOrder.none()),
            _run(_cross_buy()),
            _run(raw),
        ))
        assert later == [], f"extra compiled programs: {later}"

    def test_step_none_equals_step_agent_order_none(self):
        s1 = _advance(_start(T_QUEUE), 1)
        a = _session().step(_KEY, s1, None)
        b = _session().step(_KEY, s1, AgentOrder.none())
        assert _trees_equal(a, b)

    def test_inactive_order_submits_nothing(self):
        s1 = _advance(_start(T_QUEUE), 1)
        inactive = _passive_buy()._replace(active=jnp.bool_(False))
        s2, rep, _d = _session().step(_KEY, s1, inactive)
        s2_none, _rep_n, _dn = _session().step(_KEY, s1, None)
        assert not bool(rep.submitted)
        assert not np.any(np.asarray(rep.trade_rows))
        assert _trees_equal(s2, s2_none)


# ---------------------------------------------------------------------------
# (6) FillReport shapes and content
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestFillReportShapes:
    def _reports(self):
        s1 = _advance(_start(T_QUEUE), 1)
        _a, rep_none, _ = _session().step(_KEY, s1, None)
        _b, rep_rest, _ = _session().step(_KEY, s1, _passive_buy())
        _c, rep_fill, _ = _session().step(_KEY, s1, _cross_buy())
        return rep_none, rep_rest, rep_fill

    def test_trade_rows_is_2x6_int32(self):
        for rep in self._reports():
            assert isinstance(rep, _om.FillReport)
            tr = np.asarray(rep.trade_rows)
            assert tr.shape == (2, 6) and tr.dtype == np.int32

    def test_report_shape_is_fixed_across_order_kinds(self):
        sigs = [_signature(r) for r in self._reports()]
        assert sigs[0] == sigs[1] == sigs[2]

    def test_trade_rows_zero_without_a_fill_and_set_on_an_aggressive_fill(self):
        rep_none, rep_rest, rep_fill = self._reports()
        assert not np.any(np.asarray(rep_none.trade_rows))
        assert not np.any(np.asarray(rep_rest.trade_rows))
        tr = np.asarray(rep_fill.trade_rows)
        assert np.any(tr[0] != 0), "aggressive leg row must be written"
        assert not np.any(tr[1]), "no passive leg this step"

    def test_passive_fill_writes_the_passive_leg_row(self):
        s1 = _advance(_start(T_OVER), 1)
        s2, _rep, _d = _session().step(_KEY, s1, _passive_buy())
        _s3, rep3, _d = _session().step(_KEY, s2, None)
        tr = np.asarray(rep3.trade_rows)
        assert not np.any(tr[0]) and np.any(tr[1] != 0)

    @pytest.mark.jit
    def test_step_env_is_jit_safe(self):
        s1 = _advance(_start(T_QUEUE), 1)
        out = jax.jit(_session().step_env)(_KEY, s1, _passive_buy())
        s2, rep, done = out
        assert isinstance(s2, OMIEContractState)
        assert np.asarray(rep.trade_rows).shape == (2, 6)
        assert np.asarray(done).shape == ()

    def test_step_env_is_vmap_safe(self):
        keys = jax.random.split(_KEY, 2)
        states = jax.vmap(_session().reset)(keys)
        states = states.replace(tape_idx=jnp.asarray([T_QUEUE, T_OVER], jnp.int32))
        orders = jax.tree_util.tree_map(
            lambda a, b: jnp.stack([a, b]), _passive_buy(), AgentOrder.none())
        # I10: the existing caller-scoped x64 convention for vmapping the
        # engine (extract_fills' int64 leg; see test_omie_env.py
        # TestOneCompilePreservationUnderVmap). Mechanics are unchanged in
        # REL-05, so the same outer context applies to the Session.
        with jax.experimental.enable_x64():
            s2, rep, done = jax.vmap(_session().step_env)(keys, states, orders)
        tr = np.asarray(rep.trade_rows)
        assert tr.shape == (2, 2, 6) and tr.dtype == np.int32
        assert np.asarray(done).shape == (2,)
        for leaf in _leaves(rep):
            assert leaf.shape[:1] == (2,)
        assert np.asarray(rep.submitted).tolist() == [True, False]

    def test_aggressive_buy_reports_inventory_and_unscaled_cash(self):
        s1 = _advance(_start(T_QUEUE), 1)
        s2, rep, _d = _session().step(_KEY, s1, _cross_buy())
        assert bool(rep.submit.traded)
        assert float(rep.inventory_delta) == pytest.approx(1.0, abs=_MW)
        assert float(rep.cash_delta_eur) == pytest.approx(-12.75, abs=_EUR)
        d_cash = float(np.asarray(s2.agent_cash)[0] - np.asarray(s1.agent_cash)[0])
        d_inv = float(np.asarray(s2.agent_inventory)[0]
                      - np.asarray(s1.agent_inventory)[0])
        assert d_cash == pytest.approx(-12.75, abs=_EUR)
        assert d_inv == pytest.approx(1.0, abs=_MW)
        assert float(rep.passive_qty_signed) == pytest.approx(0.0, abs=_MW)

    def test_passive_buy_fill_reports_inventory_and_cash(self):
        s1 = _advance(_start(T_OVER), 1)
        s2, _rep, _d = _session().step(_KEY, s1, _passive_buy())
        s3, rep3, _d = _session().step(_KEY, s2, None)
        assert float(rep3.passive_qty_signed) == pytest.approx(2.0, abs=_MW)
        assert float(rep3.inventory_delta) == pytest.approx(2.0, abs=_MW)
        assert float(rep3.cash_delta_eur) == pytest.approx(-24.5, abs=_EUR)
        d_cash = float(np.asarray(s3.agent_cash)[0] - np.asarray(s2.agent_cash)[0])
        assert d_cash == pytest.approx(-24.5, abs=_EUR)


# ---------------------------------------------------------------------------
# (7) Eager cancel between steps
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestEagerCancel:
    @staticmethod
    def _rest_agent_bid():
        s1 = _advance(_start(T_OVER), 1)
        s2, rep, _d = _session().step(_KEY, s1, _passive_buy())
        assert _rows_with_oid(s2.book.lob.bids, rep.order_id).shape[0] == 1
        return s2, rep

    def test_cancel_by_report_order_id_removes_the_order(self):
        s2, rep = self._rest_agent_bid()
        book = _book_api().cancel_order(s2.book, int(rep.order_id), BUY)
        s2c = s2.replace(book=book)
        assert _rows_with_oid(s2c.book.lob.bids, rep.order_id).shape[0] == 0
        assert _book_api().get_agent_orders(s2c.book, AGENT_TRADER_ID) == []

    def test_cancelled_order_is_not_filled_on_the_next_step(self):
        s2, rep = self._rest_agent_bid()
        s2c = s2.replace(book=_book_api().cancel_order(
            s2.book, int(rep.order_id), BUY))
        s3, rep3, _d = _session().step(_KEY, s2c, None)
        assert float(rep3.passive_qty_signed) == pytest.approx(0.0, abs=_MW)
        assert float(np.asarray(s3.agent_inventory)[0]) == pytest.approx(0.0, abs=_MW)

    def test_uncancelled_order_is_filled_on_the_next_step(self):
        s2, _rep = self._rest_agent_bid()
        _s3, rep3, _d = _session().step(_KEY, s2, None)
        assert float(rep3.passive_qty_signed) == pytest.approx(2.0, abs=_MW)

    def test_cancel_with_another_id_leaves_the_order_resting(self):
        s2, rep = self._rest_agent_bid()
        wrong = int(rep.order_id) + 1
        s2c = s2.replace(book=_book_api().cancel_order(s2.book, wrong, BUY))
        assert _rows_with_oid(s2c.book.lob.bids, rep.order_id).shape[0] == 1
        _s3, rep3, _d = _session().step(_KEY, s2c, None)
        assert float(rep3.passive_qty_signed) == pytest.approx(2.0, abs=_MW)
