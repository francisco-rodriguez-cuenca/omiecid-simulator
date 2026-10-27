"""OrderBook adapter — module-level pure-function lift.

The 15 ``@jax.jit`` / ``@partial(jax.jit, …)`` decorators that previously sat
on ``self``-bound instance methods are lifted to module-level pure functions
named ``_<method>(...)`` taking explicit ``state`` and ``cfg`` arguments.
``OrderBook`` retains an instance-method API as thin trampolines so existing
call sites in ``omiecid_simulator/omiecid/`` need not change.  The lift unblocks
persistent-JAX-cache reuse across distinct ``OrderBook`` instances.

Modified by Francisco Rodriguez-Cuenca, 2026
"""
from functools import partial
from typing import Dict, NamedTuple, Optional, Tuple

import chex
import jax
from jax import numpy as jnp

import omiecid_simulator.jaxob.JaxOrderBookArrays as job
from omiecid_simulator.jaxob.jaxob_config import JAXLOB_Configuration


class LobState(NamedTuple):
    asks: jnp.ndarray
    bids: jnp.ndarray
    trades: jnp.ndarray
    key: chex.PRNGKey


# --- Lifted module-level pure functions (the jit surfaces) ---------------


def _side_array(state: LobState, side) -> jax.Array:
    """Select bids/asks under a traced ``side`` (1=bid, 0=ask) via lax.cond."""
    return jax.lax.cond(side == 1, lambda a, b: b, lambda a, b: a, state.asks, state.bids)


@partial(jax.jit, static_argnames=("cfg",))
def _init(cfg: JAXLOB_Configuration) -> LobState:
    asks = (jnp.ones((cfg.nOrders, 6)) * -1).astype(jnp.int32)
    bids = (jnp.ones((cfg.nOrders, 6)) * -1).astype(jnp.int32)
    trades = (jnp.ones((cfg.nTrades, 8)) * -1).astype(jnp.int32)
    key = jax.random.PRNGKey(cfg.seed)
    return LobState(asks, bids, trades, key)


@partial(jax.jit, static_argnames=("cfg",))
def _reset_empty(cfg: JAXLOB_Configuration, time: jnp.ndarray) -> LobState:
    del time  # symmetric with _reset_from_l2; unused on the empty path
    return _init(cfg)


@partial(jax.jit, static_argnames=("cfg",))
def _reset_from_l2(l2_book, time, cfg: JAXLOB_Configuration) -> LobState:
    state = _init(cfg)
    msgs = job.init_msgs_from_l2(cfg, l2_book, time=time)
    return _process_orders_array(state, msgs, cfg)


@partial(jax.jit, static_argnames=("cfg",))
def _process_order_array(state: LobState, quote, cfg) -> LobState:
    (*ordersides, key) = state
    key, split_key = jax.random.split(key)
    ordersides, _ = job.cond_type_side(cfg, ordersides, (split_key, quote))
    return LobState(*ordersides, key)


@partial(jax.jit, static_argnames=("cfg",))
def _process_orders_array(state: LobState, msgs, cfg) -> LobState:
    (*ordersides, key) = state
    key, split_key = jax.random.split(key)
    return LobState(*job.scan_through_entire_array(cfg, split_key, msgs, tuple(ordersides)), key)


@partial(jax.jit, static_argnames=("cfg", "n_levels"))
def _process_orders_array_l2(state: LobState, msgs, n_levels, cfg) -> Tuple[LobState, jax.Array]:
    (*ordersides, key) = state
    key, split_key = jax.random.split(key)
    all_asks, all_bids, trades = job.scan_through_entire_array_save_states(
        cfg, split_key, msgs, tuple(ordersides), msgs.shape[0])
    state = LobState(all_asks[-1], all_bids[-1], trades, key)
    l2_states = jax.vmap(job.get_L2_state, (0, 0, None, None), 0)(all_asks, all_bids, n_levels, cfg)
    return state, l2_states


@partial(jax.jit, static_argnames=("cfg", "side"))
def _get_volume_at_price(state, price, side, cfg) -> int:
    del cfg  # static-arg-passed for trampoline symmetry; not consumed here
    side_array = state.asks if side == 0 else state.bids
    return job.get_volume_at_price(side_array, price)


@partial(jax.jit, static_argnames=("cfg", "side"))
def _get_init_volume_at_price(state, price, side, cfg) -> int:
    side_array = state.asks if side == 0 else state.bids
    return job.get_init_volume_at_price(side_array, price, cfg)


@partial(jax.jit, static_argnames=("cfg",))
def _get_best_price(state, side, cfg) -> int:
    return jax.lax.cond(side == 1,
                       lambda s: job.get_best_bid(cfg, s.bids),
                       lambda s: job.get_best_ask(cfg, s.asks),
                       state)


@partial(jax.jit, static_argnames=("cfg",))
def _get_best_bid(state, cfg) -> int:
    return job.get_best_bid(cfg, state.bids)


@partial(jax.jit, static_argnames=("cfg",))
def _get_best_ask(state, cfg) -> int:
    return job.get_best_ask(cfg, state.asks)


@partial(jax.jit, static_argnames=("cfg",))
def _get_best_bid_and_ask_inclQuants(state, cfg) -> Tuple[jax.Array, jax.Array]:
    return job.get_best_bid_and_ask_inclQuants(cfg, state.asks, state.bids)


@partial(jax.jit, static_argnames=("cfg", "n_levels"))
def _get_L2_state(state, n_levels, cfg) -> jax.Array:
    return job.get_L2_state(state.asks, state.bids, n_levels, cfg)


@jax.jit
def _get_side_ids(state, side) -> jax.Array:
    return job.get_order_ids(_side_array(state, side))


# get_order / get_order_at_time: the Python ``price is not None`` dispatch
# lives at the trampoline (not traceable).  Underlying job.* helpers carry no
# cfg arg, so the lifted bodies use plain @jax.jit.

@jax.jit
def _get_order_by_id(state, side, order_id) -> jax.Array:
    return job.get_order_by_id(_side_array(state, side), order_id)


@jax.jit
def _get_order_by_id_and_price(state, side, order_id, price) -> jax.Array:
    return job.get_order_by_id_and_price(_side_array(state, side), order_id, price)


@jax.jit
def _get_order_by_time(state, side, time_s, time_ns) -> jax.Array:
    return job.get_order_by_time(_side_array(state, side), time_s, time_ns)


@jax.jit
def _get_order_by_time_and_price(state, side, time_s, time_ns, price) -> jax.Array:
    return job.get_order_by_time_and_price(_side_array(state, side), time_s, time_ns, price)


@partial(jax.jit, static_argnames=("cfg", "side"))
def _get_next_executable_order(state, side, cfg):
    return job.get_next_executable_order(cfg, side, _side_array(state, side))


# --- OrderBook: thin trampoline class over the lifted helpers ------------


class OrderBook():
    def __init__(
            self: 'OrderBook',
            cfg: Optional[JAXLOB_Configuration] = None,
        ) -> None:
        self.cfg = cfg if cfg is not None else JAXLOB_Configuration()

    def init(self): return _init(self.cfg)

    def reset(self, l2_book=None, time=None):
        if time is None:
            time = jnp.array([0, 0])
        if l2_book is None:
            return _reset_empty(self.cfg, time)
        return _reset_from_l2(l2_book, time, self.cfg)

    def process_order(self, state, quote: Dict, from_data: bool = False, verbose: bool = False):
        '''Dict-quote variant: build msg array and dispatch to _process_order_array.'''
        inttype, intside = 5, -1
        if quote['side'] == 'bid':
            intside = 1
        if quote['type'] == 'limit':
            inttype = 1
        elif quote['type'] == 'cancel':
            inttype = 2
        elif quote['type'] == 'delete':
            inttype = 2
        elif quote['type'] == 'market':
            inttype = 1
            intside = intside * -1
        msg = jnp.array([
            inttype, intside, quote['quantity'], quote['price'],
            quote['trade_id'], quote['order_id'],
            int(quote['timestamp'].split('.')[0]),
            int(quote['timestamp'].split('.')[1])
        ], dtype=jnp.int32)
        return _process_order_array(state, msg, self.cfg)

    def process_order_array(self, state, quote, from_data: bool = False, verbose: bool = False):
        return _process_order_array(state, quote, self.cfg)

    def process_orders_array(self, state, msgs):
        return _process_orders_array(state, msgs, self.cfg)

    def process_orders_array_l2(self, state, msgs, n_levels):
        return _process_orders_array_l2(state, msgs, n_levels, self.cfg)

    def get_volume_at_price(self, state, side, price, init_only: bool = False):
        if init_only:
            return _get_init_volume_at_price(state, price, side, self.cfg)
        return _get_volume_at_price(state, price, side, self.cfg)

    def get_best_price(self, state, side): return _get_best_price(state, side, self.cfg)
    def get_best_bid(self, state): return _get_best_bid(state, self.cfg)
    def get_best_ask(self, state): return _get_best_ask(state, self.cfg)
    def get_best_bid_and_ask_inclQuants(self, state): return _get_best_bid_and_ask_inclQuants(state, self.cfg)
    def get_L2_state(self, state, n_levels): return _get_L2_state(state, n_levels, self.cfg)
    def get_side_ids(self, state, side): return _get_side_ids(state, side)

    def get_order(self, state, side, order_id, price: Optional[int] = None):
        if price is not None:
            return _get_order_by_id_and_price(state, side, order_id, price)
        return _get_order_by_id(state, side, order_id)

    def get_order_at_time(self, state, side, time_s, time_ns, price: Optional[int] = None):
        if price is not None:
            return _get_order_by_time_and_price(state, side, time_s, time_ns, price)
        return _get_order_by_time(state, side, time_s, time_ns)

    def get_next_executable_order(self, state, side):
        return _get_next_executable_order(state, side, self.cfg)

    #Flatten and Unflatten functions so that methods can be appropriately jitted.
    def _tree_flatten(self: 'OrderBook'):
        children = ()  # arrays / dynamic values
        aux_data = {'cfg': self.cfg,}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


jax.tree_util.register_pytree_node(
    OrderBook,
    OrderBook._tree_flatten,
    OrderBook._tree_unflatten
)


if __name__ == "__main__":
    print("""Testing functionality of the jax OrderBook object in jorderbook.py
                acts as an object wrapper around the JaxOrderBookArrays functions""")
    ob = OrderBook()
    dict_quote = {'type': 'limit', 'side': 'bid', 'quantity': 99, 'price': 346000, 'trade_id': 8888, 'order_id': 8888, 'timestamp': '3400.005000000'}
    l2init = jnp.array([354200, 452, 350100, 89, 361200, 100, 344000, 400, 362900, 100, 343100, 100, 364000, 400, 338700, 100, 371900, 1100, 337100, 1000, 372200, 100, 336400, 1000, 372300, 200, 336000, 300, 372800, 1000, 333600, 1000, 374600, 1000, 332500, 100, 376700, 100, 331600, 100])
    state = ob.reset(l2init)
    print(state)
    print("Message dict quote processing", ob.process_order(state, dict_quote))
    msg_array = jnp.array([1, 1, 99, 346000, 8888, 8888, 3400, 5000000])
    msgs_array = jnp.array([[1, 1, 99, 346000, 8888, 8888, 3400, 5000000],
                            [1, -1, 2, 346000, 8777, 8777, 3401, 5060000]])
    print("Message array processing", ob.process_order_array(state, msg_array))
    print("Process array of messages and return N_levels of L2 state", ob.process_orders_array_l2(state, msgs_array, 10))
    print("Volume at price test:", ob.get_volume_at_price(state, 1, 346000, True))
    print("Next executable order test:", ob.get_next_executable_order(state, 1))
    print("Best bid price:", ob.get_best_price(state, 1))
    print("Best ask price:", ob.get_best_price(state, 0))
    print("Best bid and ask prices + quants:", ob.get_best_bid_and_ask_inclQuants(state))
