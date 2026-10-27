"""OMIE single-contract trading session.

A ``Session`` replays a historical (or synthetic) order tape for ONE 15-minute
OMIE contract on the market clock, from gate opening (GOT) to gate closure
(GCT), and lets one participant submit its own orders between the background
arrivals.

Public interface::

    session = Session(config)
    state = session.reset(key)                        # OMIEContractState
    state, report, done = session.step(key, state, order)   # order: AgentOrder | None
    state, report, done = session.step_env(key, state, order)  # compiled core
    tob = session.top_of_book(state)                  # TopOfBook

One step:
  * **Background.** A fixed ``orders_per_step`` count of consecutive tape rows,
    read at the carried ``state.tape_cursor``, is replayed into the book with
    ``_replay_orders``. A resting order of the participant that this flow hits
    is attributed as a passive fill. So is a resting order of the participant
    hit by its own order this step: OMIE has no self-trade prevention.
  * **Participant.** The ``AgentOrder`` carries the economic fields only. The
    session stamps the order id, the participant id (``AGENT_TRADER_ID``), the
    contract and the time-priority timestamp (the book's arrival counter after
    the background replay) and submits it with ``_submit_order``. An inactive
    order (``AgentOrder.none()`` or ``None``) submits nothing; real and empty
    orders share one compiled program.
  * **Clock.** ``time`` advances to the slice's latest timestamp, in minutes
    from GOT; ``done = time >= GCT``. Calls after ``done`` return the same state
    and an empty ``FillReport``.

``state.agent_cash`` holds realized EUR (aggressive plus passive fills,
unscaled); ``state.agent_inventory`` the signed net MW; ``state.agent_trades``
the participant's trade tape in the ``compute_agent_pnl`` column layout, one
aggressive row and one passive row per step. The passive row aggregates all of
the step's passive fills (net direction, total quantity, quantity-weighted
price). ``agent_cash``, ``agent_inventory`` and the book trade tape are exact;
``compute_agent_pnl(agent_trades)`` is an approximation on steps with passive
fills in both directions.

``reset`` / ``step_env`` / ``top_of_book`` are ``@partial(jax.jit,
static_argnums=(0,))`` and vmappable; ``(product_config, market_config)`` are
static frozen instances, so the ``_submit_order`` compile cache holds one entry.
The module carries no device pin.
"""

from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from omiecid_simulator.jaxob import JaxOrderBookArrays as job
from omiecid_simulator.jaxob import jaxob_constants as cst

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    SubmitResult,
    _make_jaxob_cfg,
    _submit_order,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.omie_states import OMIEContractState
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, Status, Validity
from omiecid_simulator.jaxomiecid.omie_data_loader import _replay_orders


# The OMIE order id the participant's submitted orders carry. Positive
# (the order validator rejects order_id <= 0) and distinct from the background
# tape's ids so the participant's fills are attributable in the whole-row diff.
AGENT_ORDER_ID_BASE = 2_000_000_000
# Anonymous participant trader id (anonymity is a HARD RULE, market-spec §7/§9.2;
# it exists only for the book's internal order-id space).
AGENT_TRADER_ID = 1_000_000


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
class AgentOrder(NamedTuple):
    """The participant's order: economic fields only.

    The session stamps order id, participant id, contract and timestamp. Every
    field is a 0-d array of a fixed dtype (``create`` / ``none`` cast), so real
    and empty orders have one pytree signature and share one compiled step.
    ``active=False`` means "no order".
    """

    active: jnp.ndarray            # bool
    side: jnp.ndarray              # int32, Side.BUY / Side.SELL
    price: jnp.ndarray             # float32, EUR/MWh
    quantity: jnp.ndarray          # float32, MW
    exec_restriction: jnp.ndarray = int(ExecRestriction.NON)   # int32
    validity: jnp.ndarray = int(Validity.GFS)                  # int32
    expiry_time: jnp.ndarray = 0                               # int32
    peak_qty: jnp.ndarray = 0.0                                # float32, MW
    ppd: jnp.ndarray = 0.0                                     # float32, EUR/MWh

    @classmethod
    def create(cls, side, price, quantity,
               exec_restriction=int(ExecRestriction.NON),
               validity=int(Validity.GFS), expiry_time=0,
               peak_qty=0.0, ppd=0.0) -> "AgentOrder":
        """An active order, every field cast to its dtype."""
        return _cast_agent_order(cls(
            active=True, side=side, price=price, quantity=quantity,
            exec_restriction=exec_restriction, validity=validity,
            expiry_time=expiry_time, peak_qty=peak_qty, ppd=ppd,
        ))

    @classmethod
    def none(cls) -> "AgentOrder":
        """The empty order (``active=False``), same signature as ``create``."""
        return _cast_agent_order(cls(active=False, side=0, price=0.0, quantity=0.0))


_AGENT_ORDER_DTYPES = (
    jnp.bool_, jnp.int32, jnp.float32, jnp.float32,
    jnp.int32, jnp.int32, jnp.int32, jnp.float32, jnp.float32,
)


def _cast_agent_order(order: AgentOrder) -> AgentOrder:
    return AgentOrder(*(jnp.asarray(v, dtype=d).reshape(())
                        for v, d in zip(order, _AGENT_ORDER_DTYPES)))


class FillReport(NamedTuple):
    """Per-step result (fixed shape, jit/vmap-safe).

    ``submitted``: the step carried an active order. ``order_id``: its stamped
    id (0 when none). ``submit``: the book's ``SubmitResult`` for it (all zero
    when none). ``passive_qty_signed`` (MW, +buy) and ``passive_value_eur``
    (realized EUR, + on a sell): the participant's resting orders hit this
    step, by the background flow or by the participant's own order (the
    passive leg of a self-match). ``inventory_delta`` (MW, +buy) and
    ``cash_delta_eur`` (realized EUR, aggressive plus passive). ``trade_rows``:
    the ``(2, 6)`` int32 trade-tape rows written this step (aggressive leg,
    passive leg; the passive row aggregates all of the step's passive fills).
    ``trades_saturated``: every row of the book's trade buffer
    holds a trade (later trades overwrite the last row).
    """

    submitted: jnp.ndarray
    order_id: jnp.ndarray
    submit: SubmitResult
    passive_qty_signed: jnp.ndarray
    passive_value_eur: jnp.ndarray
    inventory_delta: jnp.ndarray
    cash_delta_eur: jnp.ndarray
    trade_rows: jnp.ndarray
    trades_saturated: jnp.ndarray


class TopOfBook(NamedTuple):
    """Anonymous best bid / ask (EUR/MWh), mid and visible touch volume (MW).

    An empty side reads ``has_* = False``, price and volume 0.0; ``mid`` is 0.0
    unless both sides are present.
    """

    has_bid: jnp.ndarray
    has_ask: jnp.ndarray
    p_bid: jnp.ndarray
    p_ask: jnp.ndarray
    mid: jnp.ndarray
    q_bid: jnp.ndarray
    q_ask: jnp.ndarray


class Session:
    """Single-contract OMIE trading session for one participant.

    Constructed from a config dict (``tapes``, ``orders_per_step``,
    ``product_config``, ``market_config``, ``session_clock``, ``got_seconds``,
    ``gct_minutes_from_got``, ``tape_seconds_span``); the zero-arg path ships a
    self-contained synthetic tape set.
    """

    # One participant per session.
    num_agents = 1

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    def __init__(self, config: Optional[dict] = None):
        config = dict(config) if config else {}

        # --- static frozen configs (the one-compile invariant depends on these
        # being the SAME instances across every reset/step call) ---------------
        self.product_config: OMIEProductConfig = config.get(
            "product_config", OMIEProductConfig()
        )
        self.market_config: OMIEMarketConfig = config.get(
            "market_config", OMIEMarketConfig()
        )

        # --- step granularity -------------------------------------------------
        self.orders_per_step: int = int(config.get("orders_per_step", 4))

        # --- the fixed tape set (loaded once) --------------------------------
        # A stackable (n_tapes, N_orders_padded, 13) int32 array indexable by the
        # traced ``tape_idx``. The synthetic default set below is used unless a
        # tape set is supplied.
        tapes = config.get("tapes", None)
        if tapes is None:
            tapes = _default_fixed_tape_set(self.product_config)
        self.tapes = jnp.asarray(tapes, dtype=jnp.int32)
        # Digest the tape content ONCE here so the
        # static-jit-key identity is O(1) per dispatch (no per-call ``tobytes``).
        self._tape_digest = hash(self.tapes.tobytes())
        assert self.tapes.ndim == 3 and self.tapes.shape[-1] == 13, (
            "tapes must be a (n_tapes, N_orders_padded, 13) int32 array"
        )
        self.n_tapes: int = int(self.tapes.shape[0])
        self.n_orders_padded: int = int(self.tapes.shape[1])
        # Alias read by test helpers that derive the done step.
        self.tape_len: int = self.n_orders_padded

        # --- contract / horizon (GOT = origin, GCT in minutes-from-GOT) ------
        # The session measures ``time`` as minutes-from-GOT (GOT is the origin,
        # time==0 at reset). GCT is (delivery_start - 60) min per market-spec §3.1;
        # in the minutes-from-GOT frame it is a fixed horizon. The synthetic tape
        # timestamps (session-relative seconds) are scaled to this horizon.
        self.gct_minutes_from_got: int = int(
            config.get("gct_minutes_from_got", 60)
        )
        # Attribute aliases the test's GCT-discovery helper probes.
        self.gct_from_got = float(self.gct_minutes_from_got)
        self.gct = float(self.gct_minutes_from_got)
        # Tape-second -> minutes-from-GOT scale (``span_scaled`` clock): tape
        # timestamps are assumed to be seconds from GOT spread over [0, S], with
        # S = ``tape_seconds_span``, and [0, S] is mapped onto [0, gct] minutes;
        # a slice's time is its max timestamp scaled and ROUNDED TO THE NEAREST
        # minute (see ``_slice_time_minutes``). S defaults to the max timestamp
        # over the WHOLE tape set (set-wide, not per tape), so a shorter tape in
        # a stack may never reach GCT. Because of the rounding, ``done`` fires
        # at the first slice within half a scaled minute of S, possibly BEFORE
        # the last real slice, and the rows after the done slice are then not
        # replayed; tapes whose timestamps are offset or clustered (not spread
        # over [0, S] from GOT) reach GCT earlier still. For a full-tape replay
        # set ``tape_seconds_span >= 2 * max timestamp over the set`` (raising
        # ``gct_minutes_from_got`` does not help: a slice holding S always
        # reaches GCT); for real data use ``session_clock="got_relative"``.
        span_override = config.get("tape_seconds_span", None)
        if span_override is not None:
            self.tape_seconds_span: float = float(span_override)
        else:
            max_ts = int(jnp.max(self.tapes[:, :, 6]))
            self.tape_seconds_span = float(max(max_ts, 1))

        # --- session clock ---------------------------------------------------
        # ``session_clock`` selects how a slice's max timestamp maps to
        # ``state.time`` (minutes-from-GOT):
        #   * ``span_scaled`` (DEFAULT): the tape's [0, tape_seconds_span] is
        #     scaled onto [0, gct], rounded to the nearest minute (an
        #     all-pad slice keeps the prior time and does NOT end the session).
        #   * ``got_relative`` (opt-in, real per-product tapes): ``time = max(0,
        #     (max_ts - got_seconds) // 60)`` and the first ALL-PAD slice (tape
        #     exhausted) sets ``time = gct_minutes_from_got``, so
        #     ``done = time >= gct`` covers both clauses and stays sticky.
        # ``got_seconds`` is GOT in the per-product tape frame (55,200 = 15:20 D-1,
        # market-spec §3.1). Both are Python-static (trace-time branch in
        # ``_slice_time_minutes``) and join ``_jit_identity``.
        self.session_clock: str = str(config.get("session_clock", "span_scaled"))
        _VALID_SESSION_CLOCKS = ("span_scaled", "got_relative")
        if self.session_clock not in _VALID_SESSION_CLOCKS:
            raise ValueError(
                f"session_clock must be one of {_VALID_SESSION_CLOCKS}; got "
                f"{self.session_clock!r}"
            )
        self.got_seconds: int = int(config.get("got_seconds", 55_200))
        if self.session_clock == "got_relative":
            self._check_got_relative_tapes()

        # --- derived replay length (compile-time constant) -------------------
        # steps_per_episode = ceil(N_orders_padded / orders_per_step)
        self.steps_per_episode: int = int(
            math.ceil(self.n_orders_padded / self.orders_per_step)
        )

        # Cached jaxob config (used for best-bid/ask/volume reads).
        self._jaxob_cfg = _make_jaxob_cfg(self.market_config)

        # --- int-domain price decode (FMA-immune) ----------------------------
        # Integer tick offset of price 0.0, the SAME quantity computed at
        # ``contract_book.py:206`` / ``settlement.py:100``. int32 by
        # construction so the subtraction in ``_decode_price_eur`` cannot
        # promote to int64 (OQ-18: no x64 scope is opened). For the default
        # SIDC band this is 999_900, and the max legal tick 1_999_800; both,
        # their difference and their sum (3_999_600) sit two orders of
        # magnitude inside int32.
        self._price_offset_int = jnp.int32(
            round(-self.product_config.price_min / self.product_config.price_tick)
        )

    # ---------------------------------------------------------------------
    # Content-based hash/eq — fixes a cross-call segfault.
    #
    # ``reset`` / ``step_env`` are ``@partial(jax.jit, static_argnums=(0,))``, so
    # ``self`` is a STATIC jit key. Without a value-based ``__hash__`` / ``__eq__``
    # the default object-identity hash makes every distinct ``Session`` INSTANCE a
    # distinct static key ⇒ its OWN freshly-compiled XLA executable, even when two
    # instances are byte-identical. Repeatedly constructing sessions in ONE process
    # (a driver's per-tape sessions + the test helpers' per-call sessions) thus
    # churned an unbounded number of distinct compiled ``step_env`` / ``reset``
    # executables. On a many-core CPU host XLA's parallel LLVM ORC-JIT then
    # exhausts a compile-time resource ("LLVM compilation error: Cannot allocate
    # memory", cascading "JITDylib defunct") and the process takes a native
    # SIGSEGV/SIGABRT — the cross-call segfault on the 2nd+ driver call.
    #
    # Value-based identity makes two byte-identical sessions share ONE static jit key
    # ⇒ ONE compiled executable, so the distinct-executable count is bounded by
    # the number of distinct CONFIGS rather than the number of instances. This is
    # the standard fix for ``static_argnums`` on a custom class and changes no
    # observable behavior (equal-config sessions are functionally interchangeable).
    # The key covers every field the compiled jaxpr closes over.
    # ---------------------------------------------------------------------
    def _jit_identity(self):
        return (
            # Tape identity: cached O(1) content digest + shape + dtype string.
            # (No per-dispatch ``tobytes``.)
            tuple(self.tapes.shape),
            self._tape_digest,
            str(self.tapes.dtype),
            int(self.num_agents),
            int(self.orders_per_step),
            int(self.gct_minutes_from_got),
            float(self.tape_seconds_span),
            # Session clock: Python-static trace-time
            # branch + GOT origin, so distinct clocks never share an executable.
            str(self.session_clock),
            int(self.got_seconds),
            # Frozen dataclasses ⇒ value-hashable at O(1); the compiled jaxpr
            # closes over both (product tick/bound math + market book reads), so
            # functionally-different sessions must never collide onto one executable.
            self.product_config,
            self.market_config,
        )

    def __hash__(self):
        return hash(self._jit_identity())

    def __eq__(self, other):
        return (
            isinstance(other, Session)
            and self._jit_identity() == other._jit_identity()
        )

    # ---------------------------------------------------------------------
    # Test-factory escape hatches (zero-config construction).
    # ---------------------------------------------------------------------
    @classmethod
    def from_test_config(cls) -> "Session":
        """Zero-config test session with the self-contained synthetic tape set."""
        return cls({})

    @classmethod
    def for_testing(cls) -> "Session":
        return cls({})

    # ---------------------------------------------------------------------
    # reset
    # ---------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        """Sample ONE tape from the fixed set; init an empty book at GOT.

        Returns the ``OMIEContractState`` with ``tape_cursor == 0``,
        ``time == GOT`` (== 0 in minutes-from-GOT), inventory and cash 0.
        """
        tape_idx = jax.random.randint(
            key, shape=(), minval=0, maxval=self.n_tapes
        ).astype(jnp.int32)

        book0 = self._empty_book()
        state = OMIEContractState(
            book=book0,
            time=jnp.int32(0),                       # GOT origin (minutes-from-GOT)
            contract_status=jnp.int32(int(ContractStatus.OPEN)),
            step_counter=jnp.int32(0),
            agent_inventory=jnp.zeros((self.num_agents,), dtype=jnp.float32),
            agent_cash=jnp.zeros((self.num_agents,), dtype=jnp.float32),
            tape_idx=tape_idx,
            tape_cursor=jnp.int32(0),
            # Agent trade tape, two-slot layout:
            # ``(steps_per_episode, 2, 6)`` — TWO (all-zero, inert) rows per session
            # step (slot 0 aggressive / own-submit leg, slot 1 passive / resting-NON
            # leg). Sized to the derived replay length so every step's fills write
            # at ``step_counter`` without growing (jit-safe fixed shape). Zero rows
            # (``qty_tick == 0``) contribute nothing to ``compute_agent_pnl``;
            # reshaped ``(-1, 6)`` the tape feeds ``compute_agent_pnl`` directly.
            agent_trades=jnp.zeros(
                (self.steps_per_episode, 2, 6), dtype=jnp.int32
            ),
        )
        return state

    # ---------------------------------------------------------------------
    # step — user entry point
    # ---------------------------------------------------------------------
    def step(self, key, state, order: Optional[AgentOrder] = None):
        """One session step. ``order=None`` submits nothing.

        Casts the order's fields to their dtypes and calls the compiled
        ``step_env``; returns ``(state, FillReport, done)``.
        """
        order = AgentOrder.none() if order is None else _cast_agent_order(order)
        return self.step_env(key, state, order)

    # ---------------------------------------------------------------------
    # step_env — inline background-then-participant interleave core
    # ---------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key, state, order: AgentOrder):
        """One step: background slice, then the participant's order.

        Returns ``(next_state, FillReport, done)``. After ``done`` the call is
        ignored: the same state and an empty report are returned.
        """
        already_done = state.time >= jnp.int32(self.gct_minutes_from_got)

        # -- 1. background: the fixed-orders_per_step tape slice at tape_cursor --
        # ``lax.dynamic_slice_in_dim`` clamps the start so the final short step
        # reads inert pad rows, never out of bounds.
        tape = self.tapes[state.tape_idx]                     # (N_padded, 13) int32
        step_slice = jax.lax.dynamic_slice_in_dim(
            tape, state.tape_cursor, self.orders_per_step, axis=0,
        )
        book_after_background, _slab = _replay_orders(        # loader's jitted replay
            state.book, step_slice, self.product_config, self.market_config,
        )

        # -- 1b. passive-leg fill attribution ----------------------------------
        # A resting NON order placed on a PRIOR step survives in the carried
        # ``state.book`` and can be hit by this step's background flow. That fill
        # lands INSIDE the background replay (the discarded ``_slab``), with the
        # participant on the PASSIVE (maker) leg. The passive fills are credited
        # (inventory + realized cash + agent_trades tape) via
        # ``PASS_TID == AGENT_TRADER_ID`` on a diff of the book trades tape that
        # spans BOTH the background replay and the participant's own submission
        # (computed after step 2). OMIE has no self-trade prevention, so an own
        # order hitting an own resting order is a real trade, and its passive
        # leg is attributed here (the aggressive leg comes from the submit).

        # -- 2. participant: stamp the AgentOrder -> OMIEOrder, submit ----------
        # The whole-row diff wraps the participant's call specifically, taken on
        # the single carry AFTER the background slice folded in, so background
        # fills never leak into the participant's own fill.
        agent_order, is_noop, side = self._stamp_order(
            order, book_after_background, state,
        )

        book_after_agent, submit_result = _submit_order(     # shared jitted submit
            book_after_background, agent_order,
            self.product_config, self.market_config,
        )
        # An inactive order submits nothing: keep the background book unchanged.
        book_next = jax.tree_util.tree_map(
            lambda a, b: jnp.where(is_noop, a, b),
            book_after_background, book_after_agent,
        )
        passive_inv_delta, passive_realized_eur, passive_trade_row = (
            self._extract_passive_fill(
                state.book.lob.trades, book_next.lob.trades, state.time))

        # -- realized cash = sign * settlement_value (settlement_value UNSIGNED;
        # +1 SELL / -1 BUY, settlement.py:141-143). No order => 0.0. ---------
        # sign: SELL(=1) -> +1, BUY(=0) -> -1  =>  sign = 2*side - 1.
        sign = jnp.float32(2) * side.astype(jnp.float32) - jnp.float32(1)
        raw_value = sign * jnp.asarray(submit_result.settlement_value, jnp.float32)
        aggressive_realized = jnp.where(is_noop, jnp.float32(0.0), raw_value)
        # The passive-leg realized EUR (background replay plus the passive leg
        # of a self-match); exactly 0.0 when no resting order of the
        # participant filled.
        passive_realized = passive_realized_eur
        realized_cash = aggressive_realized + passive_realized

        # -- inventory update (signed net MW; no order leaves it unchanged) ----
        # trade_qty is UNSIGNED MW; signed net position moves +qty on BUY,
        # -qty on SELL: inv_delta = -sign * trade_qty. The passive-leg signed
        # fill (background replay plus the passive leg of a self-match) folds
        # into the same net position.
        traded = jnp.asarray(submit_result.traded, jnp.bool_) & (~is_noop)
        trade_qty = jnp.asarray(submit_result.trade_qty, jnp.float32)
        inv_delta = jnp.where(traded, -sign * trade_qty, jnp.float32(0.0))
        agent_inventory = state.agent_inventory.at[0].add(
            inv_delta + passive_inv_delta
        )

        # -- agent trade tape rows (jit-safe int32). Encode the participant's own
        # fill into the exact (6,) column layout ``compute_agent_pnl`` consumes:
        # [timestamp, price_offset_int, qty_tick, buyer, seller, contract_id]
        # (settlement.py:46-49). A no-fill step writes an all-zero (qty_tick==0)
        # inert row. ``compute_agent_pnl`` itself is NEVER called here (OQ-18:
        # its int64 accumulator would truncate under trace). ------------------
        aggressive_trade_row = self._agent_trade_row(
            submit_result, side, traded, state.time,
        )
        # Two-slot layout ``(steps_per_episode, 2, 6)``: slot 0 the AGGRESSIVE
        # (own-submit) leg, slot 1 the PASSIVE (resting-NON) leg, so a step with
        # both fills records both. Slot 1 aggregates ALL of the step's passive
        # fills into one row (net direction, total quantity, quantity-weighted
        # price): exact for single-direction steps up to the price rounding, an
        # approximation when passive fills go both ways. agent_cash /
        # agent_inventory and the book trade tape stay exact. Both rows are
        # already int32.
        step_rows = jnp.stack(
            [aggressive_trade_row, passive_trade_row], axis=0
        )                                                    # (2, 6) int32
        write_idx = jnp.clip(
            state.step_counter, jnp.int32(0), jnp.int32(self.steps_per_episode - 1)
        )
        agent_trades = state.agent_trades.at[write_idx].set(step_rows)

        # -- time / cursor / step advance -------------------------------------
        new_cursor = (state.tape_cursor + jnp.int32(self.orders_per_step)).astype(jnp.int32)
        new_step = (state.step_counter + jnp.int32(1)).astype(jnp.int32)
        # time = max timestamp in the slice (session-relative seconds) converted
        # to minutes-from-GOT; monotone because the tape is timestamp-sorted.
        new_time = self._slice_time_minutes(step_slice, state.time)

        # -- done = (time >= GCT) ----------------------------------------------
        done = new_time >= jnp.int32(self.gct_minutes_from_got)
        done = done.astype(jnp.bool_)

        agent_cash = state.agent_cash.at[0].add(realized_cash)

        next_state = OMIEContractState(
            book=book_next,
            time=new_time,
            contract_status=state.contract_status,
            step_counter=new_step,
            agent_inventory=agent_inventory,
            agent_cash=agent_cash,
            tape_idx=state.tape_idx,
            tape_cursor=new_cursor,
            agent_trades=agent_trades,
        )

        # -- FillReport ---------------------------------------------------------
        submitted = ~is_noop
        submit = jax.tree_util.tree_map(
            lambda x: jnp.where(submitted, x, jnp.zeros_like(x)),
            SubmitResult(*(jnp.asarray(x) for x in submit_result)),
        )
        trade_price_col = book_next.lob.trades[:, cst.TradesFeat.P.value]
        report = FillReport(
            submitted=submitted,
            order_id=jnp.where(submitted, agent_order.order_id, jnp.int32(0)),
            submit=submit,
            passive_qty_signed=passive_inv_delta,
            passive_value_eur=passive_realized_eur,
            inventory_delta=inv_delta + passive_inv_delta,
            cash_delta_eur=realized_cash,
            trade_rows=step_rows,
            trades_saturated=jnp.all(trade_price_col != jnp.int32(cst.EMPTY_SLOT)),
        )

        # -- after done: the call is ignored (same state, empty report) --------
        next_state = jax.tree_util.tree_map(
            lambda old, new: jnp.where(already_done, old, new), state, next_state,
        )
        report = jax.tree_util.tree_map(
            lambda x: jnp.where(already_done, jnp.zeros_like(x), x), report,
        )
        done = done | already_done
        return next_state, report, done

    # ---------------------------------------------------------------------
    # top_of_book — compiled market reader
    # ---------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def top_of_book(self, state: OMIEContractState) -> TopOfBook:
        """Anonymous best bid/ask, mid and aggregated VISIBLE touch volume.

        Read from ``book.lob`` only: a foreign iceberg's hidden quantity never
        moves ``q_bid`` / ``q_ask`` (``iceberg_meta`` is not read).
        """
        lob = state.book.lob
        best_ask_int, best_bid_int, has_ask, has_bid, p_ask, p_bid, mid = (
            self._top_of_book(lob)
        )
        qty_tick = jnp.float32(self.product_config.qty_tick)
        q_ask = job.get_volume_at_price(lob.asks, best_ask_int).astype(jnp.float32) * qty_tick
        q_bid = job.get_volume_at_price(lob.bids, best_bid_int).astype(jnp.float32) * qty_tick
        q_ask = jnp.where(has_ask, q_ask, jnp.float32(0.0))
        q_bid = jnp.where(has_bid, q_bid, jnp.float32(0.0))
        return TopOfBook(has_bid=has_bid, has_ask=has_ask, p_bid=p_bid,
                         p_ask=p_ask, mid=mid, q_bid=q_bid, q_ask=q_ask)

    # ---------------------------------------------------------------------
    # Order stamping — AgentOrder -> OMIEOrder (jit-safe).
    # ---------------------------------------------------------------------
    def _stamp_order(self, agent_order, book, state):
        """Stamp the participant's ``AgentOrder`` into an ``OMIEOrder``.

        The economic fields come from ``agent_order``; the session sets the
        order id (``AGENT_ORDER_ID_BASE + step_counter``), the participant id
        (``AGENT_TRADER_ID``), the contract (0) and the timestamp, taken from
        the post-background book's arrival counter (``book.ts_counter``,
        market-spec §5.2 global queue).

        Returns ``(order, is_noop, side)`` where ``side`` is a 0-d int32 in
        {BUY=0, SELL=1} and ``is_noop`` a 0-d bool (the order is inactive).
        """
        is_noop = jnp.logical_not(agent_order.active)
        side = agent_order.side

        # Unique positive order id per step so the participant's fills are
        # attributable and the validator's order_id > 0 guard passes. Anonymous
        # participant trader id.
        order_id = jnp.int32(AGENT_ORDER_ID_BASE) + state.step_counter.astype(jnp.int32)
        agent_id = jnp.int32(AGENT_TRADER_ID)
        # Time priority: stamp the order from the
        # post-background book's arrival counter, NOT the minutes-from-GOT
        # clock. market-spec §5.2 (v2.10 :325): "All timestamps ... are
        # assigned from a single global queue (unique, sequential)" -- inside the
        # book, ``ts_counter`` is that queue. Invariant: ``ts_counter`` is
        # strictly greater than the SEC of every order resting in the book (no-fill
        # rests, restamped fill-then-rest residuals and iceberg slices all leave the counter
        # >= stamp+1), so the order rests strictly BEHIND every prior arrival at
        # its price, and successive submits get strictly increasing stamps.
        # The rest-only path dispatches at this stamp; the fill-then-rest residual
        # is restamped to it (``_restamp_resting_residual``). ``state.time``
        # is the session clock only.
        timestamp = book.ts_counter.astype(jnp.int32)

        order = _make_traced_order(
            contract_id=jnp.int32(0),
            side=side,
            price=agent_order.price,
            quantity=agent_order.quantity,
            order_id=order_id,
            agent_id=agent_id,
            timestamp=timestamp,
            exec_restriction=agent_order.exec_restriction,
            validity=agent_order.validity,
            expiry_time=agent_order.expiry_time,
            status=jnp.int32(int(Status.ACTIVE)),
            peak_qty=agent_order.peak_qty,
            ppd=agent_order.ppd,
        )
        return order, is_noop, side

    # ---------------------------------------------------------------------
    # Small helpers
    # ---------------------------------------------------------------------
    def _top_of_book(self, lob):
        """Shared anonymous top-of-book decode.

        Reads the best bid/ask int ticks from the LOB, decodes them to EUR/MWh,
        and derives the side-neutral mid. Empty sides (best == -1) are masked to
        a neutral 0.0 (there is no resting liquidity to price against). ``mid`` is
        0.0 unless BOTH sides are present.

        Returns ``(best_ask_int, best_bid_int, has_ask, has_bid, p_ask, p_bid,
        mid)`` where ``p_ask``/``p_bid`` are the masked EUR/MWh prices.
        """
        best_ask_int = job.get_best_ask(self._jaxob_cfg, lob.asks)   # int tick, -1 if empty
        best_bid_int = job.get_best_bid(self._jaxob_cfg, lob.bids)   # int tick, -1 if empty
        has_ask = best_ask_int >= 0
        has_bid = best_bid_int >= 0
        # Int-domain decode (FMA-immune, backend-identical).
        p_ask = jnp.where(
            has_ask, self._decode_price_eur(best_ask_int), jnp.float32(0.0)
        )
        p_bid = jnp.where(
            has_bid, self._decode_price_eur(best_bid_int), jnp.float32(0.0)
        )
        # The MID anchor is computed in the INT
        # domain and decoded through the SAME helper, so the anchor is always
        # ON-TICK and therefore device-independent end-to-end. Tie rule: FLOOR
        # TOWARD THE BID -- ``//`` on two NON-NEGATIVE int32 tick codes is plain
        # exact floor (no negative-operand subtlety), and the sum's maximum
        # 2 * 1_999_800 = 3_999_600 is two orders inside int32 (OQ-18 holds).
        # Floor is not an arbitrary convention: it is what the CPU pipeline has
        # always submitted, because the naive decode's uniform negative bias
        # pushed every half-tick down. The mask and the both-empty 0.0 fallback
        # are byte-unchanged.
        mid_int = (best_bid_int + best_ask_int) // jnp.int32(2)
        mid = jnp.where(
            has_ask & has_bid, self._decode_price_eur(mid_int), jnp.float32(0.0)
        )
        return best_ask_int, best_bid_int, has_ask, has_bid, p_ask, p_bid, mid

    def _decode_price_eur(self, price_int):
        """Decode an int32 price tick to EUR/MWh in the INT domain (FMA-immune).

        This is the repo's own
        pre-existing convention, used verbatim at ``contract_book.py:354`` and
        ``settlement.py:100-101``::

            (price_int - price_offset_int).astype(float32) * price_tick

        It replaces the naive ``price_int * price_tick + price_min``, which is
        catastrophic cancellation by construction (both operands are ~200x the
        answer) AND is a multiply-add that XLA:CPU contracts into an FMA while
        XLA:GPU (at these shapes) does not — so the two backends decoded the
        same tick to different float32 bits. The int-domain form has no
        multiply-add to contract (the subtraction happens in the INT domain),
        so it is bitwise stable eager/jit and CPU/GPU.

        Args:
            price_int: int32 tick code (array or scalar).

        Returns:
            float32 EUR/MWh price of the same shape.
        """
        return (
            jnp.asarray(price_int, jnp.int32) - self._price_offset_int
        ).astype(jnp.float32) * jnp.float32(self.product_config.price_tick)

    def _extract_passive_fill(self, trades_before, trades_after, timestamp):
        """Attribute the participant's PASSIVE-leg fills of this step.

        Diffs the jaxob 8-column trades tape (``TradesFeat``) across the whole
        step and credits any NEW trade whose ``PASS_TID`` (the resting/maker
        leg) carries ``AGENT_TRADER_ID``. The window spans the background
        replay AND the participant's own submission. A resting order of the
        participant can therefore be hit by the background flow or by the
        participant's own order; OMIE has no self-trade prevention, so the
        latter is a real trade and its passive leg is attributed here. The
        aggressive leg of the participant's own order is accounted from the
        submit result, not by this function, so matching ``PASS_TID`` only
        counts each leg once.

        Returns ``(passive_inv_delta, passive_realized_eur,
        passive_trade_row)``:

          * ``passive_inv_delta``   0-d float32 signed MW (+buy / -sell),
          * ``passive_realized_eur`` 0-d float32 realized cash in EUR (+ on a
            passive sell, - on a passive buy; V = Q*p*d, same as the aggressive
            settlement),
          * ``passive_trade_row``   (6,) int32 in the ``compute_agent_pnl`` column
            layout ``[timestamp, price_offset_int, qty_tick, buyer, seller,
            contract_id]``, aggregating the step's passive fills (net direction,
            qty-weighted price). Inert all-zero row when no passive fill.

        Every value is EXACTLY zero / inert when the participant has no resting
        order in the carried book. Fully jit-safe: fixed shapes, vectorised
        masks, no Python branch on the traced tape.

        Note: the trade row aggregates all of the step's passive fills into
        one row (net direction, total quantity, quantity-weighted price). One
        row cannot represent fills in both directions (net quantity near zero
        with non-zero cash), so on such steps it is an approximation, and
        ``compute_agent_pnl`` over ``agent_trades`` differs from
        ``agent_cash``. The cash and inventory folds are exact per row
        regardless; the exact record is ``agent_cash`` / ``agent_inventory``
        and the book trade tape.
        """
        P = cst.TradesFeat.P.value                 # 0  resting (maker) price offset
        Q = cst.TradesFeat.Q.value                 # 1  signed qty ticks
        PASS_TID = cst.TradesFeat.PASS_TID.value   # 6  passive/maker trader id
        aid = jnp.int32(AGENT_TRADER_ID)

        # New-fill detector: ring-buffer-immune whole-row diff (extract_fills
        # idiom). A row changes iff a new fill was written to that slot.
        new_mask = jnp.any(trades_before != trades_after, axis=1)   # (nTrades,) bool

        # Only the passive (maker) leg is attributed here; the participant's
        # aggressive leg comes from the submit result (see docstring), so we
        # match ``PASS_TID`` only.
        agent_leg = new_mask & (trades_after[:, PASS_TID] == aid)

        # buy/sell disambiguation: jaxob signs the trade Q by the AGGRESSOR
        # side, so on the passive leg Q>=0 is a buy and Q<0 a sell.
        q_signed = trades_after[:, Q]
        is_buy = agent_leg & (q_signed >= 0)
        is_sell = agent_leg & (q_signed < 0)

        qty_ticks = jnp.abs(q_signed).astype(jnp.int32)
        price_ticks = trades_after[:, P].astype(jnp.int32)

        qty_tick = jnp.float32(self.product_config.qty_tick)
        d = jnp.float32(self.product_config.delivery_duration_hours)

        # signed inventory delta (MW): +buy / -sell (per-row exact).
        inv_dir = jnp.where(
            is_buy, jnp.int32(1),
            jnp.where(is_sell, jnp.int32(-1), jnp.int32(0)),
        )
        net_qty_ticks = jnp.sum(inv_dir * qty_ticks)               # signed int
        passive_inv_delta = net_qty_ticks.astype(jnp.float32) * qty_tick

        # realized cash (EUR): + on a sell, - on a buy; V = Q_mw * p_eur * d
        # (per-row exact, matching the aggressive settlement convention).
        qty_mw_row = qty_ticks.astype(jnp.float32) * qty_tick
        # Int-domain decode, shared helper.
        price_eur_row = self._decode_price_eur(price_ticks)
        cash_dir = jnp.where(
            is_sell, jnp.float32(1.0),
            jnp.where(is_buy, jnp.float32(-1.0), jnp.float32(0.0)),
        )
        passive_realized_eur = jnp.sum(cash_dir * qty_mw_row * price_eur_row * d)

        # aggregate ONE (6,) row: total abs qty, qty-weighted price offset, the
        # participant on the net-direction leg (seller when net short).
        active = is_buy | is_sell
        total_qty_ticks = jnp.sum(jnp.where(active, qty_ticks, jnp.int32(0)))
        # --- qty-weighted average price offset, OVERFLOW-SAFE (OQ-18) ---
        # The naive accumulator
        #   wsum = Σ_active q_i · p_i          (int32)
        # overflows once a large aggregated passive fill has q_tick·price_off
        # > 2^31 (e.g. 2995 · 1_004_700 ≈ 3.0e9 > 2.147e9), wrapping negative and
        # corrupting the aggregated passive-row PRICE on the trade tape (and
        # hence compute_agent_pnl). The per-row cash (passive_realized_eur) and
        # inventory (passive_inv_delta) are float32 per-row-exact above and
        # unaffected.
        #
        # We CANNOT fix this with int64 / jax.experimental.enable_x64: this runs
        # inside the jitted/vmapped step_env, where an in-body enable_x64 no-ops
        # under trace and an int64 accumulator TRUNCATES — the exact reason
        # compute_agent_pnl (which relies on int64) is never called inside
        # step_env (OQ-18).
        #
        # x64-free EXACT int32 formulation: subtract a per-step reference offset
        # p_ref = min active price offset BEFORE weighting. A resting order sits
        # at a single price level, so within one step the active fills cluster in
        # price and (p_i − p_ref) is small (0 for a single-level resting order);
        # q_i·(p_i − p_ref) and their sum therefore stay far inside int32
        # (bounded by total_qty_ticks · intra-step price spread, orders of
        # magnitude below the ~1e6·1e3 unshifted product). The qty-weighted mean
        # is reconstructed exactly as
        #   avg = p_ref + round(Σ q_i·(p_i − p_ref) / Σ q_i).
        _BIG_OFF = jnp.int32(1 << 30)  # sentinel for inactive rows in the min
        p_ref = jnp.min(jnp.where(active, price_ticks, _BIG_OFF))
        off = jnp.where(active, price_ticks - p_ref, jnp.int32(0))
        wsum_off = jnp.sum(qty_ticks * off)                        # int32-safe
        avg_price_off = jnp.where(
            total_qty_ticks > jnp.int32(0),
            p_ref + jnp.round(
                wsum_off.astype(jnp.float32)
                / jnp.maximum(total_qty_ticks, jnp.int32(1)).astype(jnp.float32)
            ).astype(jnp.int32),
            jnp.int32(0),
        )
        is_net_sell = net_qty_ticks < jnp.int32(0)
        counterparty = jnp.int32(AGENT_ORDER_ID_BASE)  # distinct anonymous id
        buyer = jnp.where(is_net_sell, counterparty, aid).astype(jnp.int32)
        seller = jnp.where(is_net_sell, aid, counterparty).astype(jnp.int32)
        passive_present = total_qty_ticks > jnp.int32(0)
        row = jnp.stack([
            timestamp.astype(jnp.int32),   # [0] timestamp
            avg_price_off,                 # [1] price offset int
            total_qty_ticks,               # [2] qty tick
            buyer,                         # [3] buyer agent id
            seller,                        # [4] seller agent id
            jnp.int32(0),                  # [5] contract_id
        ]).astype(jnp.int32)
        zero_row = jnp.zeros((6,), dtype=jnp.int32)
        passive_trade_row = jnp.where(passive_present, row, zero_row).astype(jnp.int32)

        return (
            passive_inv_delta,
            passive_realized_eur,
            passive_trade_row,
        )

    def _agent_trade_row(self, submit_result, side, traded, timestamp):
        """Encode the participant's own fill into the (6,) int32
        ``compute_agent_pnl`` row: ``[timestamp, price_offset_int, qty_tick,
        buyer, seller, contract_id]`` (settlement.py:46-49).

        The participant leg id is ``AGENT_TRADER_ID`` and the counterparty is a
        distinct anonymous background id, placed on the buyer/seller slot per the
        order side (SELL => participant is seller, BUY => participant is buyer).
        ``compute_agent_pnl`` then applies +1 (seller) / -1 (buyer), so the eager
        reference decodes to ``sign · (Q · p · d)``, matching the per-step
        ``sign · settlement_value`` cash within float32 tick-rounding.

        A no-fill step (``traded == False``) yields an all-zero row
        (``qty_tick == 0``) that contributes nothing to ``compute_agent_pnl``.
        Fully jit-safe: no Python branch on the traced ``side`` / ``traded``.
        """
        price_tick = jnp.float32(self.product_config.price_tick)
        price_min = jnp.float32(self.product_config.price_min)
        qty_tick = jnp.float32(self.product_config.qty_tick)

        trade_price = jnp.asarray(submit_result.trade_price, jnp.float32)
        trade_qty = jnp.asarray(submit_result.trade_qty, jnp.float32)

        # Tick-encode price offset + quantity exactly as the loader / settlement
        # tape does (round to the nearest tick).
        price_offset_int = jnp.round(
            (trade_price - price_min) / price_tick
        ).astype(jnp.int32)
        qty_tick_int = jnp.round(trade_qty / qty_tick).astype(jnp.int32)

        agent_id = jnp.int32(AGENT_TRADER_ID)
        counterparty = jnp.int32(AGENT_ORDER_ID_BASE)  # distinct anonymous id
        is_sell = side == jnp.int32(int(Side.SELL))
        buyer = jnp.where(is_sell, counterparty, agent_id).astype(jnp.int32)
        seller = jnp.where(is_sell, agent_id, counterparty).astype(jnp.int32)

        row = jnp.stack([
            timestamp.astype(jnp.int32),      # [0] timestamp
            price_offset_int,                 # [1] price offset int
            qty_tick_int,                     # [2] qty tick
            buyer,                            # [3] buyer agent id
            seller,                           # [4] seller agent id
            jnp.int32(0),                     # [5] contract_id
        ]).astype(jnp.int32)
        # Inert (all-zero) row on a no-fill step.
        zero_row = jnp.zeros((6,), dtype=jnp.int32)
        return jnp.where(traded, row, zero_row).astype(jnp.int32)

    def _empty_book(self) -> OMIEBookState:
        """An empty ``OMIEBookState`` at GOT (matches ``ContractBook.init``)."""
        lob0 = ContractBook(
            _synthetic_contract(self.gct_minutes_from_got),
            self.product_config, self.market_config,
        ).init()
        return lob0

    def _check_got_relative_tapes(self):
        """Constructor guards for ``session_clock="got_relative"``.

        Raises ``ValueError`` (naming the offending key) when the loaded tapes
        cannot carry the GOT-relative clock:
          * a real row (ts > 0) before ``got_seconds`` — wrong frame (older
            per-session or synthetic tapes);
          * a real row at or after ``got_seconds + 60 * gct_minutes_from_got``
            — the GCT does not match the loaded contract (every row of a
            product is sent in [GOT, GCT), market-spec §13.4);
          * fewer than ``orders_per_step`` trailing pad rows on the longest
            tape — the clamped ``dynamic_slice`` would re-read real rows, the
            first all-pad slice would never come and the session would never
            terminate;
          * an all-pad aligned slice (rows ``k*ops .. k*ops+ops-1``, every row
            ts <= 0) before the slice holding a tape's last real row
            — the first all-pad
            slice ends the session, so the real rows after it would never
            replay. Pad rows inside a slice that also holds a real row are
            harmless and accepted; a tape with no real row is exempt.
        """
        ts = np.asarray(self.tapes[:, :, 6])
        real = ts > 0
        if bool(real.any()):
            real_ts = ts[real]
            min_ts, max_ts = int(real_ts.min()), int(real_ts.max())
            if min_ts < self.got_seconds:
                raise ValueError(
                    f"session_clock='got_relative': a real tape row has ts "
                    f"{min_ts} < got_seconds {self.got_seconds} (wrong tape "
                    "frame? got_relative needs session-day-frame tapes (ts in "
                    "seconds since D-1 00:00))"
                )
            gct_second = self.got_seconds + 60 * self.gct_minutes_from_got
            if max_ts >= gct_second:
                raise ValueError(
                    f"session_clock='got_relative': a real tape row has ts "
                    f"{max_ts} >= got_seconds + 60*gct_minutes_from_got = "
                    f"{gct_second}; gct_minutes_from_got="
                    f"{self.gct_minutes_from_got} does not match the loaded "
                    "contract (no row can be sent at or after GCT)"
                )
        n = ts.shape[1]
        idx = np.arange(n)[None, :]
        last_real = np.where(real, idx, -1).max(axis=1)        # -1: no real row
        min_trailing = int((n - 1 - last_real).min())
        if min_trailing < self.orders_per_step:
            raise ValueError(
                f"session_clock='got_relative' needs at least orders_per_step="
                f"{self.orders_per_step} trailing pad rows on the longest tape "
                f"(found {min_trailing}); otherwise the first all-pad slice "
                "never comes and the episode never terminates"
            )
        ops = int(self.orders_per_step)
        for b in range(ts.shape[0]):
            L = int(last_real[b])
            n_full = L // ops if L >= 0 else 0      # slices strictly before L's slice
            if n_full == 0:
                continue
            slice_has_real = real[b, : n_full * ops].reshape(n_full, ops).any(axis=1)
            if not bool(slice_has_real.all()):
                k = int(np.argmin(slice_has_real))  # first all-pad slice
                raise ValueError(
                    f"session_clock='got_relative': tape {b} has an all-pad "
                    f"slice at step {k} (rows {k*ops}..{k*ops+ops-1}, "
                    f"orders_per_step={ops}) before its last real row {L}; the "
                    "first all-pad slice ends the episode (time jumps to "
                    "gct_minutes_from_got), so the real rows after it would "
                    "never replay. Interior pad blocks are not supported under "
                    "got_relative (use session_clock='span_scaled' or remove "
                    "the gap rows)"
                )

    def _slice_time_minutes(self, step_slice, prior_time):
        """Advance ``time`` to the max timestamp in the slice (minutes-from-GOT).

        Column [6] is the timestamp. Pad/inert rows
        carry a non-positive timestamp; we ignore them (``jnp.where`` masks to
        the prior time so a slice of only pad rows keeps time monotone, never
        regressing).

        Default ``span_scaled`` clock: timestamps are assumed to be seconds
        from GOT spread over [0, S], with S = ``tape_seconds_span`` (by default
        the max timestamp over the WHOLE tape set, so S is set-wide). The slice
        time is ``t_k = max(t_{k-1}, round(m_k * gct / S))``, with ``m_k`` the
        slice's max real timestamp, ROUNDED TO THE NEAREST minute, and
        ``done = t_k >= gct``. ``done`` therefore fires at the first slice with
        ``m_k >= S * (gct - 1/2) / gct``: at the first slice within half a
        scaled minute of S, possibly before the last real slice; rows after
        the done slice are then not replayed (calls after ``done`` are ignored). Offset
        or clustered timestamps reach GCT earlier still, and a tape whose max
        is below S may never reach it. There is no promise that ``done`` fires
        exactly at the last real slice. For a full-tape replay set
        ``tape_seconds_span >= 2 * max timestamp over the set`` (raising
        ``gct_minutes_from_got`` does not help: a slice holding S always
        reaches GCT). The exact condition is ``round(M * gct / S) < gct``,
        i.e. ``S > M * gct / (gct - 1/2)`` with M the set's max timestamp;
        the factor 2 is a margin against float32 rounding.

        Under ``session_clock="got_relative"`` (a
        Python-static branch, so the default jaxpr is unchanged) the slice time
        is instead ``max(0, (max_ts - got_seconds) // 60)`` (int32, pad rows
        masked BEFORE the subtraction) and the first ALL-PAD slice sets it to
        ``gct_minutes_from_got``. Every real row precedes GCT, so tape
        exhaustion is the only terminator on real data.
        """
        ts = step_slice[:, 6]                                   # (orders_per_step,) int32
        valid = ts > jnp.int32(0)
        masked = jnp.where(valid, ts, jnp.int32(-1))
        max_ts = jnp.max(masked)                                # -1 if all pad rows
        if self.session_clock == "got_relative":
            # All-pad slice (tape exhausted) jumps to GCT; the monotone max
            # below keeps it there (sticky ``done``).
            since_got = jnp.maximum(
                (max_ts - jnp.int32(self.got_seconds)) // jnp.int32(60),
                jnp.int32(0),
            )
            new_time = jnp.where(
                max_ts >= 0, since_got, jnp.int32(self.gct_minutes_from_got)
            )
            return jnp.maximum(new_time, prior_time.astype(jnp.int32)).astype(jnp.int32)
        scale = jnp.float32(self.gct_minutes_from_got) / jnp.float32(
            max(self.tape_seconds_span, 1.0)
        )
        slice_time = jnp.round(max_ts.astype(jnp.float32) * scale).astype(jnp.int32)
        # A slice of only pad rows (max_ts < 0) keeps the prior time.
        new_time = jnp.where(max_ts >= 0, slice_time, prior_time.astype(jnp.int32))
        # Never regress (monotone).
        return jnp.maximum(new_time, prior_time.astype(jnp.int32)).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Traced order construction — build an OMIEOrder whose fields are 0-d traced
# scalars (the shared jitted _submit_order consumes traced float/int fields).
# ---------------------------------------------------------------------------
def _make_traced_order(
    contract_id, side, price, quantity, order_id, agent_id, timestamp,
    exec_restriction, validity, expiry_time, status, peak_qty, ppd,
):
    from omiecid_simulator.omiecid.order_types import OMIEOrder
    return OMIEOrder(
        contract_id=jnp.asarray(contract_id, jnp.int32),
        side=jnp.asarray(side, jnp.int32),
        price=jnp.asarray(price, jnp.float32),
        quantity=jnp.asarray(quantity, jnp.float32),
        order_id=jnp.asarray(order_id, jnp.int32),
        agent_id=jnp.asarray(agent_id, jnp.int32),
        timestamp=jnp.asarray(timestamp, jnp.int32),
        exec_restriction=jnp.asarray(exec_restriction, jnp.int32),
        validity=jnp.asarray(validity, jnp.int32),
        expiry_time=jnp.asarray(expiry_time, jnp.int32),
        status=jnp.asarray(status, jnp.int32),
        peak_qty=jnp.asarray(peak_qty, jnp.float32),
        ppd=jnp.asarray(ppd, jnp.float32),
    )



def _synthetic_contract(gct_minutes_from_got: int) -> Contract:
    """A minimal OPEN contract for building the empty book.

    The session measures time in minutes-from-GOT (GOT is the origin), so the exact
    absolute minutes are irrelevant to the book — only the OPEN status and the
    presence of the fields matter for ``ContractBook.init``.
    """
    return Contract(
        contract_id=0,
        delivery_start_minutes=gct_minutes_from_got + 60,
        delivery_end_minutes=gct_minutes_from_got + 75,
        gct_minutes=gct_minutes_from_got,
        status=ContractStatus.OPEN,
    )


# ---------------------------------------------------------------------------
# Synthetic default fixed-tape set — a self-contained set so Session() (zero-arg)
# constructs without data (used by the behavioural tests). Real tapes come from
# the loader (``omiecid_simulator.jaxomiecid``); this default is used only when
# no ``tapes`` is supplied.
# ---------------------------------------------------------------------------
def _default_fixed_tape_set(product_config: OMIEProductConfig):
    """Build a small (n_tapes, N_orders_padded, 13) int32 fixed-tape set.

    Each tape is a chronologically-sorted stream of resting limit orders that
    build a two-sided book plus a few crossing orders (so the background flow
    produces fills and the book has a non-trivial top-of-book). Rows are the
    loader's 13-column int32 layout (contract_id, side, price_int, qty_int,
    order_id, agent_id, timestamp, exec_restriction, validity, expiry_time,
    status, peak_qty_int, ppd_int); prices/quantities are tick-encoded exactly
    as the loader emits them. Trailing rows are inert pad sentinels
    (timestamp <= 0, order_id <= 0) that ``_replay_orders`` tolerates (an
    empty/inert row produces no fill).
    """
    import numpy as np

    price_min = product_config.price_min
    price_tick = product_config.price_tick
    qty_tick = product_config.qty_tick

    def enc_price(p):
        return int(round((p - price_min) / price_tick))

    def enc_qty(q):
        return int(round(q / qty_tick))

    def row(side, price, qty, oid, ts, exec_r=int(ExecRestriction.NON)):
        return [
            0,                       # contract_id
            int(side),               # side (BUY=0/SELL=1)
            enc_price(price),        # price int tick
            enc_qty(qty),            # qty int tick
            int(oid),                # order_id (>0)
            777,                     # background agent_id (anonymous, != agent)
            int(ts),                 # timestamp (session-relative seconds > 0)
            int(exec_r),             # exec_restriction
            int(Validity.GFS),       # validity
            0,                       # expiry_time
            int(Status.ACTIVE),      # status
            0,                       # peak_qty int
            0,                       # ppd int
        ]

    BUY = int(Side.BUY)
    SELL = int(Side.SELL)

    tapes = []
    # A couple of distinct tapes (a fixed *set*; reset samples one). All tapes
    # share the SAME timestamp sequence (only prices/ids differ) so every tape's
    # last real order maps to exactly GCT — the session terminates at the same
    # derived step regardless of which tape reset sampled.
    for base in (0, 1):
        oid = 100 + base * 1000
        ts = 60                # session-relative seconds; strictly increasing
        rows = []
        # Build a resting two-sided book (bids below, asks above a ~50 mid).
        for lvl in range(3):
            rows.append(row(BUY, 49.0 - lvl - base * 0.1, 10.0 + lvl, oid, ts)); oid += 1; ts += 30
            rows.append(row(SELL, 51.0 + lvl + base * 0.1, 10.0 + lvl, oid, ts)); oid += 1; ts += 30
        # A few crossing IOC orders (aggress the resting book => background fills).
        rows.append(row(BUY, 52.0, 5.0, oid, ts, exec_r=int(ExecRestriction.IOC))); oid += 1; ts += 30
        rows.append(row(SELL, 48.0, 5.0, oid, ts, exec_r=int(ExecRestriction.IOC))); oid += 1; ts += 30
        # More resting refills so later steps still have a live book.
        for lvl in range(3):
            rows.append(row(BUY, 49.5 - lvl, 8.0, oid, ts)); oid += 1; ts += 30
            rows.append(row(SELL, 50.5 + lvl, 8.0, oid, ts)); oid += 1; ts += 30
        tapes.append(rows)

    n_real = max(len(t) for t in tapes)
    # Pad to a common N_orders_padded that is a multiple of a reasonable
    # orders_per_step so the derived replay length is clean, and add a couple of
    # trailing pad steps so the final slice comfortably crosses GCT.
    n_padded = n_real + 8
    # Inert pad sentinel row: order_id <= 0 and timestamp <= 0 => no fill, and
    # ``_slice_time_minutes`` masks it out (it does not advance time).
    pad = [0, 0, 0, 0, -1, 0, -1, int(ExecRestriction.NON), int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0]

    padded = []
    for t in tapes:
        rows = list(t) + [list(pad) for _ in range(n_padded - len(t))]
        padded.append(rows)

    return np.asarray(padded, dtype=np.int32)


__all__ = ["Session", "AgentOrder", "FillReport", "TopOfBook"]
