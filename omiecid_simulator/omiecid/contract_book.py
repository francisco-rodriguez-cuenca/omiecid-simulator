"""OMIE Continuous Intraday per-contract order book.

Wraps jaxob.OrderBook with OMIE-specific adapter logic:
- Price offset encoding (negative EUR/MWh -> positive jaxob int32)
- Quantity scaling (float MW -> int32)
- Side encoding (BUY=0/SELL=1 -> +1/-1)
- Timestamp mapping (single int -> (seconds, 0))
- Full-removal cancel semantics (jaxob subtracts quantity)
- Positive order ID enforcement

Domain rules from docs/omiecid/market-spec.md.

``submit_order`` is hoisted to a module-level
``@partial(jax.jit, static_argnames=("product_config", "market_config"))``
function ``_submit_order`` that dispatches the four ExecRestriction
variants (NON / IOC / FOK / ICE) via ``jax.lax.switch`` on a traced
``jnp.int32`` discriminator and runs the two multi-slice matching
loops as a bounded ``jax.lax.fori_loop`` (a fixed-trip loop rather
than a ``jax.lax.while_loop``) whose per-trip body is the
loop body gated by a ``lax.cond`` early-out; the fixed trip
count ``_TRIP_BOUND`` (100) is DECOUPLED from the truncation cap
``OMIEMarketConfig.max_match_iterations`` (200) — see the module-level
``_TRIP_BOUND`` note. ICEwPI is the ICE arm with ``order.ppd != 0``
(no separate switch arm). The IOC and ICE arms
share ONE NON/IOC loop instance (an IOC sweep of the order's total quantity)
run under a ``lax.cond`` above the four-arm switch; the ICE arm then
registers the unexecuted remainder. The trip-count exit ORs into
``OMIEBookState.truncated`` (the truncation carry-flag — no new flag).

The ``SubmitResult`` fields are 0-d ``jnp.*`` scalars.
``get_order`` returns a sentinel ``OMIEOrder`` (``order_id ==
SENTINEL_ORDER_ID``) on miss instead of ``None``. ``cancel_order``
(``msg_type=2``) is a separate public method outside the jitted submit
path; it remains Python-side.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.jaxob import JaxOrderBookArrays as job
from omiecid_simulator.jaxob.jaxob_config import JAXLOB_Configuration
from omiecid_simulator.jaxob.jorderbook import LobState, OrderBook
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.jax_constants import (
    MAX_ICEBERGS,
    SENTINEL_ORDER_ID,
    make_sentinel_order,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction, OMIEOrder, Side, make_order,
)
from omiecid_simulator.omiecid.order_validator import validate_order
from omiecid_simulator.omiecid.encoding import (
    decode_price as _jaxob_price_to_omie,
    decode_qty as _jaxob_qty_to_omie,
    encode_side as _omie_side_to_jaxob,
    encode_side_lookup as _omie_side_to_lookup,
)
from omiecid_simulator.omiecid.trade_extraction import extract_fills, RawFills


# ---------------------------------------------------------------------------
# SubmitResult — return type for submit_order and modify_order
# ---------------------------------------------------------------------------

class SubmitResult(NamedTuple):
    """Result of a submit_order or modify_order call.

    Fields are 0-d JAX scalars:
        accepted: 0-d ``jnp.bool_`` — True if the order passed validation
            and was submitted.
        traded: 0-d ``jnp.bool_`` — True if the order produced at least
            one trade.
        trade_qty: 0-d ``jnp.float32`` — total matched quantity in MW
            (0.0 if no trade).
        trade_price: 0-d ``jnp.float32`` — trade price in EUR/MWh
            (0.0 if no trade). For walked levels this is the
            quantity-weighted average price.
        settlement_value: 0-d ``jnp.float32`` — V = Q (MW) x p (EUR/MWh)
            x d (h). For 15-min contracts, d = 0.25 h (market-spec SS2.3).

    ``accepted`` remains at index 0 for backward compatibility.
    The four trailing fields carry zero defaults so the existing
    direct-construction callsites (e.g., ``SubmitResult(accepted=True)``)
    continue to work — the tracer accepts Python ``bool`` / ``float``
    just as readily as 0-d ``jnp.*`` scalars; the production
    construction sites in the jitted body always pass concrete 0-d
    ``jnp.*`` scalars.
    """
    accepted: jnp.ndarray
    traded: jnp.ndarray = False
    trade_qty: jnp.ndarray = 0.0
    trade_price: jnp.ndarray = 0.0
    settlement_value: jnp.ndarray = 0.0


# ---------------------------------------------------------------------------
# OMIEBookState — complete state for one OMIE contract book
# ---------------------------------------------------------------------------

class OMIEBookState(NamedTuple):
    """Complete state for one OMIE contract book.

    Bundles the jaxob LobState with the iceberg side-channel state
    and monotonic timestamp counter. All ContractBook public methods
    accept and return this instead of raw LobState.

    ``iceberg_meta`` is a fixed-size ``jnp.ndarray`` of
    shape ``(MAX_ICEBERGS, 8)`` (int32). ``ts_counter`` is a scalar
    ``jnp.int32``. ``truncated`` is a 0-d ``jnp.bool_`` carry-flag
    (truncation backstop): set to True when an iceberg is registered whose
    ``ceil(total_qty / peak_qty) > MAX_SLICES_PER_REGEN``, surfacing
    the silent-truncation hazard documented in ``jax_constants.py``
    (MAX_SLICES_PER_REGEN). The pytree contains only ``jnp.ndarray`` leaves — no Python
    ``dict`` / ``list`` / ``tuple`` / ``None`` leaves.

    ``n_active`` is a 0-d ``jnp.int32`` counting the number
    of live iceberg slots at the head of ``iceberg_meta``. The
    iceberg layout is **packed and contiguous** — live rows occupy
    indices ``[0, n_active)`` and sentinel rows occupy indices
    ``[n_active, MAX_ICEBERGS)``. This is the load-bearing structural
    invariant maintained by append-at-tail register and swap-with-last delete in
    ``IcebergManager.register`` / ``IcebergManager.cleanup``. It lets
    the outer ``lax.scan`` / ``lax.dynamic_slice`` bound in
    ``regenerate_slices`` / ``matchable_hidden_volume`` /
    ``truncation_flag`` be ``n_active`` instead of ``MAX_ICEBERGS``,
    making cost data-dependent rather than cap-dependent.

    ``book_full`` is a 0-d ``jnp.bool_``
    sticky carry-flag, **distinct** from ``truncated`` (``truncated`` can
    already be set by iceberg saturation on busy sessions, so a
    reused signal would be hollow exactly where observability matters).
    It is set ``True`` when a resting limit-add fires jaxob's full-side
    eviction (the silent worst-price level-wipe in ``bid_lim`` /
    ``ask_lim``'s ``check_book_fill=True`` branch). Detected without
    changing any jaxob signature, by re-deriving jaxob's own ``full_book_flag`` predicate
    ``jnp.all(side[:, OrderSideFeat.P.value] >= 0)`` on the **pre-dispatch**
    side the order rests on, at the five ``_dispatch_jaxob_msg`` sites.
    The flag makes the cap-breach loss **OBSERVABLE, NOT CORRECT**:
    ``book_depth`` stays at 100 and the level-wipe still drops resting
    liquidity past the cap — that gap is tracked under OQ-10.
    """
    lob: LobState                    # jaxob order book arrays
    iceberg_meta: jnp.ndarray        # (MAX_ICEBERGS, 8) int32 — iceberg side-channel
    ts_counter: jnp.ndarray          # 0-d int32 — monotonic timestamp counter
    truncated: jnp.ndarray           # 0-d bool — sticky truncation backstop flag
    n_active: jnp.ndarray            # 0-d int32 — count of live iceberg slots
    # Appended LAST with a False default so existing
    # 5-arg keyword constructions (synthetic-state fixtures, audit tests) stay
    # valid — the leaf defaults wherever a constructor omits it.
    book_full: jnp.ndarray = False   # 0-d bool — sticky full-side-eviction diagnostic


def _build_jaxob_msg(
    msg_type: int,
    side: int,
    quantity: int,
    price: int,
    order_id: int,
    agent_id: int,
    time_s: int,
    time_ns: int,
) -> jnp.ndarray:
    """Construct the 8-element int32 array that jaxob expects."""
    return jnp.array(
        [msg_type, side, quantity, price, order_id, agent_id, time_s, time_ns],
        dtype=jnp.int32,
    )


def _build_jaxob_msg_traced(
    msg_type, side, quantity, price, order_id, agent_id, time_s, time_ns,
) -> jnp.ndarray:
    """Like ``_build_jaxob_msg`` but every component is a traced 0-d int32 scalar.

    Used inside the jitted submit_order body where the message fields
    are derived from traced ``order`` pytree values.
    """
    return jnp.stack([
        jnp.asarray(msg_type, jnp.int32),
        jnp.asarray(side, jnp.int32),
        jnp.asarray(quantity, jnp.int32),
        jnp.asarray(price, jnp.int32),
        jnp.asarray(order_id, jnp.int32),
        jnp.asarray(agent_id, jnp.int32),
        jnp.asarray(time_s, jnp.int32),
        jnp.asarray(time_ns, jnp.int32),
    ])


# IcebergManager import is placed AFTER _build_jaxob_msg so that
# iceberg_manager.py can re-import _build_jaxob_msg from this module
# without creating a circular-import failure.
from omiecid_simulator.omiecid.iceberg_manager import (  # noqa: E402
    COL_BASE_PRICE_INT, COL_ORDER_ID, COL_PPD, COL_REMAINING_HIDDEN, COL_SIDE,
    COL_SLICE_COUNT, IcebergManager,
)


# ---------------------------------------------------------------------------
# JIT-side encoders / decoders (operate on traced scalars; configs static)
# ---------------------------------------------------------------------------

def _jit_encode_price(price_eur, product_config: OMIEProductConfig):
    """Round-to-tick offset encoding of a traced price to int32 ticks."""
    price_scale = jnp.float32(round(1.0 / product_config.price_tick))
    price_offset = jnp.int32(round(-product_config.price_min / product_config.price_tick))
    px = jnp.asarray(price_eur, jnp.float32)
    return jnp.round(px * price_scale).astype(jnp.int32) + price_offset


def _jit_encode_qty(qty_mw, product_config: OMIEProductConfig):
    """Round-to-tick scaling of a traced quantity to int32 ticks."""
    qty_scale = jnp.float32(round(1.0 / product_config.qty_tick))
    q = jnp.asarray(qty_mw, jnp.float32)
    return jnp.round(q * qty_scale).astype(jnp.int32)


def _jit_decode_qty(qty_int, product_config: OMIEProductConfig):
    """Decode a 0-d int32 quantity tick back into a float32 MW scalar."""
    qty_tick = jnp.float32(product_config.qty_tick)
    return jnp.asarray(qty_int, jnp.float32) * qty_tick


def _jit_encode_side(side):
    """BUY=0 -> +1; SELL=1 -> -1 (jaxob convention). Formula: 1 - 2 * side."""
    s = jnp.asarray(side, jnp.int32)
    return jnp.int32(1) - jnp.int32(2) * s


# ---------------------------------------------------------------------------
# JAXLOB config derivation — pure function, deterministic from market_config.
# ---------------------------------------------------------------------------

def _make_jaxob_cfg(market_config: OMIEMarketConfig) -> JAXLOB_Configuration:
    return JAXLOB_Configuration(
        book_depth=market_config.book_depth,
        nOrders=market_config.book_depth,
        nTrades=market_config.nTrades,
        cancel_mode=cst.CancelMode.STRICT_BY_ID.value,
        type_4_interpretation=cst.Type4Interpretation.IOC.value,
        simulator_mode=cst.SimulatorMode.GENERAL_EXCHANGE.value,
    )


def _dispatch_jaxob_msg(lob: LobState, msg: jnp.ndarray, jaxob_cfg) -> LobState:
    """JIT-side dispatch of a single jaxob message into the order book.

    Hides the private ``_process_order_array`` lift behind a public name
    so production code does not import lifted private symbols (audited
    by ``tests/omiecid/test_jit_cache_defrag_audit.py`` §"no private lifted
    import outside jaxob"). Implementation parity with
    ``OrderBook.process_order_array`` is intentional and audited.
    """
    (*ordersides, key) = lob
    key, split_key = jax.random.split(key)
    ordersides, _ = job.cond_type_side(jaxob_cfg, ordersides, (split_key, msg))
    return LobState(*ordersides, key)


def _resting_side_full(lob: LobState, side) -> jnp.ndarray:
    """Pre-dispatch re-derivation of jaxob's ``full_book_flag`` (no jaxob change).

    Returns a 0-d ``jnp.bool_``: ``True`` iff the side a limit order rests on
    is already full at dispatch time — i.e. a subsequent resting add will
    fire jaxob's silent worst-price level-wipe
    (``JaxOrderBookArrays.py`` ``bid_lim:396`` / ``ask_lim:485``
    ``full_book_flag = jnp.all(side[:, P] >= 0)``).

    Efficiency constraints:
      - (e1) one evaluation, on the **single** side the order rests on
        (BUY rests on ``bids``, SELL rests on ``asks``);
      - (e2) reuses the ``lob`` already in scope pre-dispatch — no extra
        gather/copy of the book array;
      - (e3) a **single** ``jnp.all`` reduction over a ``(book_depth,)``
        price column; the resting side is selected by a scalar-broadcast
        ``jnp.where`` mask on ``side``, **not** a ``lax.cond`` / Python
        ``if`` (which would defeat the one-compile invariant);
      - (e4) the caller OR-s the result into the existing ``book_full``
        leaf — no new control flow, no extra state plumbing.

    Recompute-vs-signature tension: this is **exactly** the ``full_book_flag``
    jaxob already computes internally and discards. The zero-cost
    alternative (have jaxob return it) is an ``add_order`` / ``bid_lim`` /
    ``ask_lim`` / ``cond_type_side`` signature change, which the jaxob
    core is kept free of, so the one
    redundant reduction per resting-add is the accepted cost.

    Args:
        lob: pre-dispatch ``LobState``.
        side: 0-d ``jnp.int32`` OMIE side (``Side.BUY`` == 0, ``Side.SELL``
            == 1).

    Returns:
        0-d ``jnp.bool_``.
    """
    p_col = cst.OrderSideFeat.P.value
    is_buy = jnp.asarray(side, jnp.int32) == jnp.int32(int(Side.BUY))
    # Select the resting side's price column (BUY->bids, SELL->asks) with a
    # scalar-broadcast mask, then a single jnp.all over the (book_depth,)
    # column. No control flow.
    resting_prices = jnp.where(is_buy, lob.bids[:, p_col], lob.asks[:, p_col])
    return jnp.all(resting_prices >= 0)


# ---------------------------------------------------------------------------
# Trade result conversion (RawFills -> SubmitResult, JIT-safe scalars)
# ---------------------------------------------------------------------------

def _fills_to_submit_result(
    fills: RawFills,
    product_config: OMIEProductConfig,
    accepted_scalar: jnp.ndarray,
) -> SubmitResult:
    """Convert raw jaxob-int fills into an OMIE-domain SubmitResult.

    Args:
        fills: Raw fill aggregation from ``extract_fills()``.
        product_config: OMIE product config for unit conversion.
        accepted_scalar: 0-d ``jnp.bool_`` carry — what to put in the
            ``accepted`` field of the returned ``SubmitResult``.

    Returns:
        A ``SubmitResult`` whose five fields are 0-d ``jnp.*`` scalars.
        ``accepted`` / ``traded`` are 0-d ``jnp.bool_``; the three
        EUR/MWh fields are 0-d ``jnp.float32``. The price decode is
        rewritten to ``(int - offset) * tick`` (subtract the offset in
        integer space FIRST, then multiply by the tick) so float32
        precision is preserved — the naive ``int * tick + price_min``
        suffered from the "large + small" float subtraction that
        amplified the float32 ULP at decode time. Mathematically
        equivalent; numerically much tighter.
    """
    total_qty_f = jnp.asarray(fills.total_qty_int, jnp.float32)
    weighted_price_f = jnp.asarray(fills.weighted_price_int, jnp.float32)
    has_fill = total_qty_f > jnp.float32(0.0)

    qty_tick = jnp.float32(product_config.qty_tick)
    price_tick = jnp.float32(product_config.price_tick)
    price_offset_int = jnp.int32(
        round(-product_config.price_min / product_config.price_tick)
    )
    delivery_duration_hours = jnp.float32(product_config.delivery_duration_hours)

    # Safe divisor for VWAP (float32).
    safe_denom = jnp.where(has_fill, total_qty_f, jnp.float32(1.0))
    vwap_jaxob_f = weighted_price_f / safe_denom
    trade_qty_mw = total_qty_f * qty_tick

    # Price decode: subtract the integer offset BEFORE multiplying by the
    # tick. This avoids the "1004910 * 0.01 - 9999.0" float32 cancellation
    # that produces ``50.0996`` instead of ``50.10`` at the float32 ULP
    # boundary — the rewritten form ``(1004910 - 999900) * 0.01 = 5010 *
    # 0.01 = 50.10`` is exact at float32.
    vwap_int = jnp.round(vwap_jaxob_f).astype(jnp.int32)
    trade_price_eur = jnp.asarray(vwap_int - price_offset_int, jnp.float32) * price_tick

    # Settlement: compute the centered (no-offset) price * qty sum in
    # float32 to avoid int32 overflow on large trades. The
    # ``weighted_price_int`` field is int64 from ``extract_fills`` (the
    # product can reach ~2e10 for a 999-MW @ 9999-EUR trade — beyond
    # int32). Casting to float32 first absorbs the magnitude; the
    # ``omie_value = (weighted_price_int - offset * qty_int)`` form
    # then cancels the encoding offset cleanly.
    price_offset_f = jnp.float32(price_offset_int)
    omie_value_f = weighted_price_f - price_offset_f * total_qty_f
    exact_omie_value = omie_value_f * qty_tick * price_tick
    settlement = exact_omie_value * delivery_duration_hours

    zero_f = jnp.float32(0.0)
    trade_qty_out = jnp.where(has_fill, trade_qty_mw, zero_f)
    trade_price_out = jnp.where(has_fill, trade_price_eur, zero_f)
    settlement_out = jnp.where(has_fill, settlement, zero_f)

    return SubmitResult(
        accepted=jnp.asarray(accepted_scalar, jnp.bool_),
        traded=has_fill,
        trade_qty=trade_qty_out,
        trade_price=trade_price_out,
        settlement_value=settlement_out,
    )


def _rejected_result() -> SubmitResult:
    """Build the canonical REJECTED result (all-zero scalars)."""
    return SubmitResult(
        accepted=jnp.bool_(False),
        traded=jnp.bool_(False),
        trade_qty=jnp.float32(0.0),
        trade_price=jnp.float32(0.0),
        settlement_value=jnp.float32(0.0),
    )


def _accepted_no_trade_result() -> SubmitResult:
    """Build the ACCEPTED-NO-TRADE result (accepted=True, rest zero)."""
    return SubmitResult(
        accepted=jnp.bool_(True),
        traded=jnp.bool_(False),
        trade_qty=jnp.float32(0.0),
        trade_price=jnp.float32(0.0),
        settlement_value=jnp.float32(0.0),
    )


def _aggregated_result(total_qty, total_value, product_config) -> SubmitResult:
    """Build an ACCEPTED SubmitResult from a multi-slice loop's aggregate fills.

    Shared by ``_non_or_ioc_path`` and ``_fok_path``: both wrap the
    ``(total_qty, total_value)`` accumulators returned by their matching
    loop into a synthetic single ``RawFills`` (``num_fills`` = 1 iff any
    quantity matched) and decode it to an ``accepted=True`` result. The
    ``num_fills`` predicate only needs to distinguish traded / no-trade —
    the exact count is not consumed downstream by ``_fills_to_submit_result``.
    """
    agg_fills = RawFills(
        total_qty_int=total_qty,
        weighted_price_int=total_value,
        num_fills=jnp.where(
            total_qty > jnp.float32(0.0), jnp.int32(1), jnp.int32(0),
        ),
    )
    return _fills_to_submit_result(agg_fills, product_config, jnp.bool_(True))


# ---------------------------------------------------------------------------
# JIT helpers — cancel-by-id and matchable-volume inline (traced).
# ---------------------------------------------------------------------------

def _jit_cancel_by_id(lob, order_id, side, iceberg_meta, n_active, ts_counter,
                     prior_trunc, iceberg_mgr, jaxob_cfg):
    """Cancel a resting order whose id == order_id from ``lob``.

    Mirrors the eager ``cancel_order`` semantics:
      - look up the order on the side
      - if not found (oid <= 0), return lob unchanged
      - otherwise dispatch a CANCEL message with the remaining qty
      - clean up iceberg metadata for that order_id.

    The return tuple is the 5-tuple
    ``(lob, meta, n_active, ts, trunc)`` so ``n_active`` is threaded
    through the cancel path.
    """
    lookup_side = jnp.int32(1) - jnp.asarray(side, jnp.int32)
    order_row = job.get_order_by_id(
        jax.lax.cond(
            lookup_side == jnp.int32(1),
            lambda lob_: lob_.bids,
            lambda lob_: lob_.asks,
            lob,
        ),
        order_id,
    )
    # order_row layout: [price, qty, oid, tid, sec, nsec].
    found = order_row[2] > jnp.int32(0)
    remaining_qty = order_row[1]
    price = order_row[0]
    jaxob_side = _jit_encode_side(side)

    msg = _build_jaxob_msg_traced(
        msg_type=jnp.int32(2),
        side=jaxob_side,
        quantity=remaining_qty,
        price=price,
        order_id=order_id,
        agent_id=jnp.int32(0),
        time_s=jnp.int32(0),
        time_ns=jnp.int32(0),
    )

    new_lob = jax.lax.cond(
        found,
        lambda args: _dispatch_jaxob_msg(args[0], args[1], jaxob_cfg),
        lambda args: args[0],
        (lob, msg),
    )
    new_meta, new_n_active = iceberg_mgr.cleanup(iceberg_meta, n_active, order_id)
    return new_lob, new_meta, new_n_active, ts_counter, prior_trunc


def _jit_matchable_volume(lob, side, jaxob_price, iceberg_meta, n_active,
                          iceberg_mgr):
    """Return total matchable quantity on the opposite side at crossing prices.

    Returns a 0-d int32 — visible + hidden iceberg quantity.

    ``n_active`` is threaded through to the
    bounded ``matchable_hidden_volume`` call so the hidden-volume scan
    is cap-independent.
    """
    side_i = jnp.asarray(side, jnp.int32)
    price_i = jnp.asarray(jaxob_price, jnp.int32)
    is_buy = side_i == jnp.int32(int(Side.BUY))

    asks = lob.asks
    bids = lob.bids
    asks_valid = (asks[:, 2] > 0) & (asks[:, 0] <= price_i)
    bids_valid = (bids[:, 2] > 0) & (bids[:, 0] >= price_i)
    asks_total = jnp.sum(jnp.where(asks_valid, asks[:, 1], jnp.int32(0)))
    bids_total = jnp.sum(jnp.where(bids_valid, bids[:, 1], jnp.int32(0)))
    total_visible = jnp.where(is_buy, asks_total, bids_total)

    hidden_total = iceberg_mgr.matchable_hidden_volume(
        iceberg_meta, n_active, side_i, price_i,
    )
    return total_visible + hidden_total


# ---------------------------------------------------------------------------
# Multi-slice OUTER matching-loop helpers (bounded
# lax.fori_loop + lax.cond-identity early-out; the ``_while_loop_match*``
# names are retained for the existing test/caller surface).
# ---------------------------------------------------------------------------

# The OMIE OUTER multi-slice / iceberg-regeneration loop
# is a bounded ``fori_loop`` (converted from a data-dependent ``while_loop``).
# ``TRIP_BOUND`` is a PYTHON-STATIC int (NOT a traced ``jnp.int32``) so the
# ``fori_loop`` is a fixed-trip-count primitive that compiles to one artifact —
# the one-compile invariant holds regardless of how deep the real loop runs.
# Flattening also removes the vmap batch-max amplification (the whole vmapped
# batch would otherwise run to the deepest lane's realized trip count); the fixed
# masked-pass count is flat across the batch instead.
#
# DECOUPLING: the loop TRIP COUNT is a DISTINCT concept
# from the truncation cap ``max_match_iterations`` (200, kept in config). Only
# the ``fori_loop`` iteration count is ``TRIP_BOUND = 100``, chosen because
# 100 >= the realized outer-loop max (67, measured by a probe) AND >= the test
# suite's N_RESTING_ICEBERGS=70 scenario — a safety margin over the empirical max while
# HALVING the masked-trip overhead vs 200 (which regressed short-tape
# throughput). Since 100 >= every realized/tested trip count, the loop completes
# all REAL trips then runs identity no-ops, so every observable leaf is
# byte-identical to the 200-trip run. Do NOT raise this above
# ``max_match_iterations``.
#
# STRUCTURAL CEILING (why the ``hit_cap``-vs-``TRIP_BOUND`` guard is a LIVE,
# loud backstop and never dead code): the outer-loop trip count is structurally
# bounded by ``max_icebergs * MAX_SLICES_PER_REGEN`` (the
# runtime iceberg cap is the configured ``market_config.max_icebergs``,
# bounded above by the ``MAX_ICEBERGS`` = 8192 hard structural ceiling in
# ``omiecid_simulator/omiecid/jax_constants.py``; ``MAX_SLICES_PER_REGEN`` = 64 there
# too — worst case 8192 * 64 = 524288 at the ceiling). This FAR exceeds both
# ``TRIP_BOUND = 100`` and ``max_match_iterations = 200``. So maxing out the
# loop is NOT impossible by construction — it is impossible only on
# reachable/realistic inputs (realized max 67; see the ``hit_cap`` comment in
# ``_while_loop_match`` for the replay and fuzz maxima with the per-pass
# price-priority cap); an adversarial input CAN exceed 100 (C14c does, by design). The
# ``hit_cap``-against-``TRIP_BOUND`` guard therefore
# stays a live, "loud, never silent" truncation backstop (it sets
# ``truncated``), never a silent drop. No runtime assertion is added: it would
# be false-by-design on such adversarial inputs.
_TRIP_BOUND = 100


def _run_bounded_match_loop(still_running, body_fn, init_carry):
    """Run the OMIE OUTER matching loop as a bounded ``fori_loop`` early-out.

    Shared by ``_while_loop_match`` (NON/IOC) and ``_while_loop_match_fok``;
    the ``body_fn`` / ``still_running`` predicate differ per site so both are
    passed in as arguments (each closes over its own site-local state), keeping
    the loop bound ``_TRIP_BOUND`` a Python-static int (one-compile preserved).

    ``masked_body`` wraps the EXISTING ``body_fn`` verbatim in a ``lax.cond``
    early-out: once ``still_running(carry)`` is False (the original
    ``while_loop`` ``cond_fn`` would have exited), every remaining trip is a
    pure identity no-op returning the WHOLE carry unchanged — no accumulator
    double-count, no state carried past termination. The ``it`` counter
    increments ONLY inside ``body_fn`` (real, unmasked trips), so the caller's
    post-loop ``iter_f`` still means "realized real trip count" for its
    ``hit_cap`` computation.

    See the module-level ``_TRIP_BOUND`` note for the trip-count/truncation
    decoupling and the structural-ceiling rationale for why the caller's
    ``hit_cap``-vs-``_TRIP_BOUND`` guard is a live, loud backstop.
    """
    def masked_body(_i, carry):
        return jax.lax.cond(
            still_running(carry),
            body_fn,
            lambda c: c,
            carry,
        )

    return jax.lax.fori_loop(0, _TRIP_BOUND, masked_body, init_carry)

def _restamp_resting_residual(lob, order_id, side, timestamp, filled):
    """Restore the book-visible entry stamp of a resting NON residual.

    Market-spec v2.9 §5.2: partial execution is
    not a modification and assigns no new timestamp, so the unexecuted
    residual of a NON incoming order rests with its ORIGINAL entry
    timestamp. The counter dispatch stamps used inside
    ``_while_loop_match`` are internal fill disambiguators (the whole-row
    ``extract_fills`` needs distinct tape rows under saturation); the
    book-visible stamp of a resting NON residual is restored to
    ``order.timestamp`` here, per market-spec §5.2.

    Masked column write on the incoming order's OWN side: every row whose
    ``OID == order_id`` gets ``SEC = timestamp`` and ``NSEC = 0``, gated on
    ``filled`` (the call traded). The gate keeps the no-fill path a strict
    byte no-op, so an older resting row sharing the ``order_id`` is never
    restamped by a non-crossing submit. A miss is a no-op. ``side`` is
    traced; both sides are computed and selected with ``jnp.where``.
    ``ts_counter`` is neither read nor written.
    """
    oid_col = cst.OrderSideFeat.OID.value
    sec_col = cst.OrderSideFeat.SEC.value
    nsec_col = cst.OrderSideFeat.NSEC.value
    order_id = jnp.asarray(order_id, jnp.int32)
    filled = jnp.asarray(filled, jnp.bool_)

    def _restamp(side_arr):
        hit = (side_arr[:, oid_col] == order_id) & filled
        side_arr = side_arr.at[:, sec_col].set(
            jnp.where(hit, jnp.asarray(timestamp, side_arr.dtype),
                      side_arr[:, sec_col]))
        return side_arr.at[:, nsec_col].set(
            jnp.where(hit, jnp.zeros((), side_arr.dtype),
                      side_arr[:, nsec_col]))

    is_buy = jnp.asarray(side, jnp.int32) == jnp.int32(Side.BUY)
    bids = jnp.where(is_buy, _restamp(lob.bids), lob.bids)
    asks = jnp.where(is_buy, lob.asks, _restamp(lob.asks))
    return lob._replace(bids=bids, asks=asks)


def _rank_nsec(lob, order_id, side, price_int, sec):
    """Rank a resting entry behind its same-second peers via NSEC.

    Unique stamps across prices and sides
    (market-spec §5.2 "single global queue, unique, sequential"; §6.1 time
    priority; Matching Algorithm §4.3, §5.5). Tape stamps have second
    resolution, so two orders resting in the same second would otherwise
    store identical (SEC, NSEC) keys and jaxob would break a same-price tie by
    array slot (a cancelled order's freed slot lets a later arrival jump the
    queue). The stored stamp is made unique per contract, across prices and
    sides, and strictly increasing in the simulator's processing order.

    *Own* rows: live rows (``Q > 0``) on the order's own side with ``OID ==
    order_id``, ``P == price_int`` and ``SEC == sec``. *Peers*: every other
    live row with ``SEC == sec``, on either side. If any peer exists, write
    ``NSEC = max(peer NSEC) + 1`` on the own rows, else leave the book as is.
    A miss (the order did not rest) is a byte no-op, as is a rest with no peer.
    ``side`` is traced; both sides are computed and masked by side. SEC and
    ``ts_counter`` are neither read nor written.

    Applied at the NON post-loop (after ``_restamp_resting_residual``) and at
    the first-slice rest of an incoming iceberg in ``_ice_path``. Slices
    regenerated within a submit are not ranked here: they are restamped once
    per submit, at the activation event, by ``_stamp_regenerated_slices``,
    which ranks them on the same
    per-contract NSEC key behind every row then resting in their second. A
    later tape order arriving in that second is ranked behind them by this
    helper at its own rest site ("processing order").
    """
    oid_col = cst.OrderSideFeat.OID.value
    p_col = cst.OrderSideFeat.P.value
    q_col = cst.OrderSideFeat.Q.value
    sec_col = cst.OrderSideFeat.SEC.value
    nsec_col = cst.OrderSideFeat.NSEC.value
    order_id = jnp.asarray(order_id, jnp.int32)

    is_buy = jnp.asarray(side, jnp.int32) == jnp.int32(Side.BUY)

    def _keys(side_arr, own_side):
        dt = side_arr.dtype
        same_sec = ((side_arr[:, q_col] > 0)
                    & (side_arr[:, sec_col] == jnp.asarray(sec, dt)))
        own = (same_sec & own_side
               & (side_arr[:, p_col] == jnp.asarray(price_int, dt))
               & (side_arr[:, oid_col] == order_id))
        peer = same_sec & ~own
        m = jnp.max(jnp.where(peer, side_arr[:, nsec_col],
                              jnp.asarray(-1, dt)))
        return own, peer, m

    own_b, peer_b, m_b = _keys(lob.bids, is_buy)
    own_a, peer_a, m_a = _keys(lob.asks, ~is_buy)
    any_peer = jnp.any(peer_b) | jnp.any(peer_a)
    m = jnp.maximum(m_b, m_a.astype(m_b.dtype))

    def _write(side_arr, own):
        dt = side_arr.dtype
        hit = own & any_peer
        return side_arr.at[:, nsec_col].set(
            jnp.where(hit, m.astype(dt) + jnp.asarray(1, dt),
                      side_arr[:, nsec_col]))

    bids = _write(lob.bids, own_b)
    asks = _write(lob.asks, own_a)
    return lob._replace(bids=bids, asks=asks)


def _stamp_regenerated_slices(lob, c0, c1, event_sec):
    """Stamp the slices a submit activated at that submit's event second.

    Slice activation stamps (market-spec §5.2 "a new timestamp is
    assigned on ... activation of a new iceberg slice"; §4.2; §4.2.1 Rule 4
    eager activation; §6.1). ``regenerate_slices`` stamps each new slice
    from ``ts_counter`` (``time_ns = 0``), which runs ahead of the tape
    clock; this helper moves those stored stamps back to the matching
    process that activated them, so a slice queues after the order whose
    process activated it, behind every order then resting at its price and
    ahead of every later arrival.

    Called ONCE per submit, after the submit's last dispatch and after the
    aggressor's own rows are restamped and ranked (never inside a match
    loop body). ``c0`` is the submit's ``ts_init`` =
    ``update_timestamp(ts_counter, event_sec)``; ``c1`` is the counter after
    the submit's last regeneration.

    **Precondition:** every SEC resting at entry is strictly below
    ``ts_counter``, so every row present at entry, and ``event_sec`` itself,
    lies below ``c0``. Only under that invariant does the range selection
    below pick out exactly the slices this submit regenerated.

    *Selected* rows: live rows (``Q > 0``) on either side with
    ``c0 <= SEC < c1``. *Pre-existing* rows: live rows with ``SEC < c0``
    (they include the aggressor's restamped remainder or first slice at
    ``(event_sec, n)``). For a selected row r at price p on side s:

    * ``SEC_r = max(event_sec, max SEC of pre-existing rows on s at p)`` —
      the inner max binds only when an agent order (stamped from the
      counter) already rests at p, so the slice never jumps ahead of it.
    * ``NSEC_r = b(SEC_r) + (old SEC_r - c0)``, where ``b(S)`` is ``1 +`` the
      max NSEC of pre-existing rows on either side with ``SEC == S``, or 0
      if none. The offset keeps the exhaustion order of the counter stamps;
      NSEC may have gaps (only order and uniqueness matter).

    With nothing selected the book is returned byte-identical. Pure: P, Q,
    OID, TID, the trades tape and ``ts_counter`` are neither read for
    writing nor written.
    """
    p_col = cst.OrderSideFeat.P.value
    q_col = cst.OrderSideFeat.Q.value
    sec_col = cst.OrderSideFeat.SEC.value
    nsec_col = cst.OrderSideFeat.NSEC.value
    c0 = jnp.asarray(c0, jnp.int32)
    c1 = jnp.asarray(c1, jnp.int32)
    e = jnp.asarray(event_sec, jnp.int32)

    both = jnp.concatenate([lob.bids, lob.asks], axis=0)
    both_live = both[:, q_col] > 0
    both_sec = both[:, sec_col].astype(jnp.int32)
    both_pre = both_live & (both_sec < c0)

    def _stamp(side_arr):
        dt = side_arr.dtype
        live = side_arr[:, q_col] > 0
        sec = side_arr[:, sec_col].astype(jnp.int32)
        sel = live & (sec >= c0) & (sec < c1)
        pre = live & (sec < c0)
        # S_r: latest pre-existing SEC at the slice's price on its side.
        same_p = side_arr[:, None, p_col] == side_arr[None, :, p_col]
        pre_sec_at_p = jnp.max(
            jnp.where(same_p & pre[None, :], sec[None, :], jnp.int32(-1)),
            axis=1)
        new_sec = jnp.maximum(e, pre_sec_at_p)
        # b(S_r): next free NSEC in second S_r across both sides.
        at_s = both_pre[None, :] & (both_sec[None, :] == new_sec[:, None])
        m = jnp.max(jnp.where(at_s, both[None, :, nsec_col].astype(jnp.int32),
                              jnp.int32(-1)), axis=1)
        new_nsec = (m + jnp.int32(1)) + (sec - c0)
        side_arr = side_arr.at[:, sec_col].set(
            jnp.where(sel, new_sec.astype(dt), side_arr[:, sec_col]))
        return side_arr.at[:, nsec_col].set(
            jnp.where(sel, new_nsec.astype(dt), side_arr[:, nsec_col]))

    return lob._replace(bids=_stamp(lob.bids), asks=_stamp(lob.asks))


def _cap_pass_qty(lob, side, price_int, remaining_int, iceberg_meta, n_active):
    """Cap one matching pass so it never walks past a better iceberg's next slice.

    Iceberg price priority (market-spec §6.1, §4.2.1 Rule 2; Matching
    Algorithm §3.3 / §5.1.3): a price level is not exhausted while an iceberg
    there still has hidden quantity. jaxob walks every crossing level in one
    call and slices regenerate only between passes, so each pass is stopped at
    the last level at least as good as the next slice a resting iceberg is
    about to show.

    P* is the best next-slice price, ``base + (slice_count + 1) * ppd`` (the
    ``regenerate_slices`` formula), over opposite-side active slots
    (index < ``n_active``) with hidden remainder > 0 whose next slice crosses
    the aggressor's limit ``price_int``. V is the visible opposite-side
    quantity priced at least as well as P*. Returns ``min(remaining_int, V)``
    when V > 0, else ``remaining_int`` unchanged (no qualifying slot, or a
    qualifying slot without a visible slice: the stall guard, so every pass
    dispatches a positive quantity). Vectorised masked reductions only;
    ``side`` is traced and both sides are selected with ``jnp.where``.
    """
    side = jnp.asarray(side, jnp.int32)
    price_int = jnp.asarray(price_int, jnp.int32)
    remaining_int = jnp.asarray(remaining_int, jnp.int32)
    is_buy = side == jnp.int32(Side.BUY)

    next_px = (iceberg_meta[:, COL_BASE_PRICE_INT]
               + iceberg_meta[:, COL_PPD]
               * (iceberg_meta[:, COL_SLICE_COUNT] + jnp.int32(1)))
    live = jnp.arange(iceberg_meta.shape[0], dtype=jnp.int32) < jnp.asarray(
        n_active, jnp.int32)
    crosses = jnp.where(is_buy, next_px <= price_int, next_px >= price_int)
    qualifies = (live & (iceberg_meta[:, COL_SIDE] != side)
                 & (iceberg_meta[:, COL_REMAINING_HIDDEN] > jnp.int32(0))
                 & crosses)
    big = jnp.int32(jnp.iinfo(jnp.int32).max)
    # BUY aggressor: best = lowest ask-side next slice; SELL: highest bid-side.
    p_star = jnp.where(
        is_buy,
        jnp.min(jnp.where(qualifies, next_px, big)),
        jnp.max(jnp.where(qualifies, next_px, -big)),
    )

    p_col = cst.OrderSideFeat.P.value
    q_col = cst.OrderSideFeat.Q.value
    opp = jnp.where(is_buy, lob.asks, lob.bids)
    opp_p = opp[:, p_col]
    opp_q = opp[:, q_col]
    at_least_as_good = jnp.where(is_buy, opp_p <= p_star, opp_p >= p_star)
    visible = (opp_p >= 0) & (opp_q > 0) & at_least_as_good
    v = jnp.sum(jnp.where(visible, opp_q, jnp.int32(0))).astype(jnp.int32)

    apply_cap = jnp.any(qualifies) & (v > jnp.int32(0))
    return jnp.where(apply_cap, jnp.minimum(remaining_int, v), remaining_int)


def _while_loop_match(
    state_lob, state_iceberg, state_n_active, state_ts, state_trunc,
    order, product_config, market_config,
    iceberg_mgr, jaxob_cfg,
    is_ioc,
):
    """The shared NON/IOC multi-slice matching loop.

    ``n_active`` is in the carry
    pytree alongside ``iceberg_meta`` and threaded through every
    cancel / regenerate / cleanup branch.

    A 0-d ``jnp.bool_`` ``book_full`` accumulator
    rides the carry as the final element. Before each resting limit-add
    dispatch it ORs in ``_resting_side_full(lob_in, side)`` (the
    pre-dispatch full-side predicate), guarded so a no-dispatch iteration
    (``nothing_left``) contributes nothing. The accumulator is returned to
    the caller, which OR-s it into the ``book_full`` state leaf.

    Carries:
        lob, iceberg_meta, n_active, ts_counter, prior_trunc,
        remaining_qty_int (int64), total_qty_int (int64),
        total_value_int (int64), dispatch_ts (int32),
        aggressive_cancelled (bool), iter (int32), terminate (bool),
        book_full (bool)

    Remainder timestamp: pass 1 dispatches at ``order.timestamp``;
    later passes (and ``do_resubmit``) dispatch at counter stamps, which
    are internal fill disambiguators. On the NON branch, once the call
    has filled, ``_restamp_resting_residual`` writes ``SEC = order.timestamp``
    and ``NSEC = 0`` back on the resting residual (market-spec §5.2).
    The trades tape and ``ts_counter`` are unaffected.

    Iceberg price priority (market-spec §6.1, §4.2.1 Rule 2): each pass
    dispatches ``_cap_pass_qty(...)`` instead of the whole remainder, so a
    pass stops at the last opposite level at least as good as the next
    slice of a resting opposite iceberg. That slice regenerates between
    passes and is crossed by the next pass before any worse level. Tapes
    with no such iceberg, or no worse crossing level, are unchanged. The
    post-loop NON re-submit (``do_resubmit``) is not capped: it only runs
    after the trip bound binds, where ``truncated`` already flags the call.

    Returns the final carry tuple, from which the caller builds the
    SubmitResult.
    """
    max_iters = jnp.int32(market_config.max_match_iterations)

    quantity_int = _jit_encode_qty(order.quantity, product_config)
    price_int = _jit_encode_price(order.price, product_config)
    side = jnp.asarray(order.side, jnp.int32)
    jaxob_side = _jit_encode_side(side)
    order_id = jnp.asarray(order.order_id, jnp.int32)
    agent_id = jnp.asarray(order.agent_id, jnp.int32)
    timestamp = jnp.asarray(order.timestamp, jnp.int32)

    # Initial dispatch timestamp = order.timestamp; ts_counter updated via
    # iceberg_mgr.update_timestamp().
    ts_init = iceberg_mgr.update_timestamp(state_ts, timestamp)

    # Incoming ICEwPI ladder (market-spec §4.2.1 Rule 2, *Incoming
    # iceberg*): an incoming ICEwPI sweeps as a chain of instances. Instance
    # k shows at most P ticks at limit L_k = b + k*d, so each pass is bounded
    # by what is left of the current instance and is dispatched at L_k; a
    # pass with no fill ends the sweep (the first instance that cannot
    # trade). Traced only on the IOC sweep (the shared IOC / ICE instance),
    # so the NON trace is unchanged. ``stepped`` is False for IOC and for a
    # standard ICE (d == 0), whose passes keep the old inputs bit for bit.
    # d is the encoder ``register`` writes to ``COL_PPD`` and P is encoded as
    # in ``_ice_path``, so L_k equals the price regeneration gives instance k.
    if is_ioc:
        ladder_peak_int = _jit_encode_qty(order.peak_qty, product_config)
        ladder_ppd_int = iceberg_mgr._encode_ppd_int(order.ppd)
        stepped = (
            (jnp.asarray(order.exec_restriction, jnp.int32)
             == jnp.int32(ExecRestriction.ICE))
            & (ladder_peak_int > jnp.int32(0))
            & (ladder_ppd_int != jnp.int32(0))
        )
        ladder_peak_div = jnp.maximum(ladder_peak_int, jnp.int32(1))

    # Accumulators stay as float32 inside the trace to avoid the
    # int32 ↔ int64 split that ``trade_extraction.extract_fills`` returns
    # (its int64 fields are produced under a scoped ``enable_x64`` —
    # casting them down here keeps the carry pytree dtype-stable under
    # the surrounding default-x32 trace).
    init_remaining = jnp.asarray(quantity_int, jnp.float32)
    init_total_qty = jnp.float32(0.0)
    init_total_value = jnp.float32(0.0)
    init_dispatch_ts = timestamp
    init_aggressive_cancelled = jnp.bool_(False)
    init_iter = jnp.int32(0)
    init_terminate = jnp.bool_(False)
    init_book_full = jnp.bool_(False)

    init_carry = (
        state_lob, state_iceberg, state_n_active, ts_init, state_trunc,
        init_remaining, init_total_qty, init_total_value,
        init_dispatch_ts, init_aggressive_cancelled,
        init_iter, init_terminate, init_book_full,
    )

    def still_running(carry):
        # The fori_loop early-out predicate. Same
        # predicate a while_loop ``cond_fn`` would test, MINUS the
        # ``it < max_iters`` clause — that clause is subsumed by the fixed
        # ``fori_loop`` trip count ``TRIP_BOUND`` (a real body trip can only
        # execute TRIP_BOUND=100 times). The truncation cap is enforced
        # separately in the ``hit_cap`` computation below, which fires against
        # BOTH ``max_iters`` (200) and ``TRIP_BOUND`` (100) so exhausting the
        # loop's trips with work remaining sets ``truncated`` — never silent.
        (_lob, _meta, _na, _ts, _trunc,
         remaining, _tq, _tv,
         _dts, _ac,
         _it, terminate, _bf) = carry
        return (remaining > jnp.float32(0.0)) & (~terminate)

    def body_fn(carry):
        (lob, iceberg_meta, n_active, ts_counter, prior_trunc,
         remaining, total_qty, total_value,
         dispatch_ts, aggressive_cancelled,
         it, terminate, book_full) = carry

        # Iceberg price priority: cap the pass so it never walks a worse
        # level while a better-priced opposite iceberg can regenerate.
        pass_price, pass_qty = price_int, remaining
        if is_ioc:
            # Incoming ICEwPI ladder: F filled ticks so far,
            # k = F // P, l = P(k+1) - F ticks left in instance k,
            # L_k = b + k*d. One cap call; its limit and quantity inputs go
            # through ``jnp.where(stepped, new, old)``.
            filled_int = jnp.round(total_qty).astype(jnp.int32)
            inst_k = filled_int // ladder_peak_div
            # Multi-instance pass: with p*
            # the best visible opposite price (``_cap_pass_qty``'s validity
            # test), the pass covers instances k..j, j = k + floor(g / |d|),
            # g = L_k - p* (BUY) or p* - L_k (SELL), when liquidity exists and
            # g >= 0 (else j = k). It is dispatched at L_j = b + j d, the least
            # aggressive limit of k..j, so every fill is allowed for each of
            # them; a shortfall leaves F inside an earlier instance and the
            # next pass resumes at that instance's own limit.
            is_buy = side == jnp.int32(Side.BUY)
            opp = jnp.where(is_buy, lob.asks, lob.bids)
            opp_p = opp[:, cst.OrderSideFeat.P.value]
            opp_valid = (opp_p >= 0) & (opp[:, cst.OrderSideFeat.Q.value] > 0)
            big = jnp.int32(jnp.iinfo(jnp.int32).max)
            p_star = jnp.where(
                is_buy,
                jnp.min(jnp.where(opp_valid, opp_p, big)),
                jnp.max(jnp.where(opp_valid, opp_p, -big)),
            )
            limit_k = price_int + inst_k * ladder_ppd_int
            # Liquidity is the guard: with no visible opposite order p* is
            # the +/-INT32_MAX sentinel and the raw gap wraps.
            has_liq = jnp.any(opp_valid)
            gap = jnp.where(
                has_liq,
                jnp.where(is_buy, limit_k - p_star, p_star - limit_k),
                jnp.int32(-1),
            )
            abs_d = jnp.maximum(jnp.abs(ladder_ppd_int), jnp.int32(1))
            # int32 guard: j <= k + ceil(remaining / P). Value-neutral, since
            # the pass quantity is min(remaining, P(j+1) - F).
            remaining_int = jnp.round(remaining).astype(jnp.int32)
            j_cap = inst_k + (remaining_int + ladder_peak_div - jnp.int32(1)) \
                // ladder_peak_div
            inst_j = jnp.where(
                gap >= jnp.int32(0),
                jnp.minimum(inst_k + gap // abs_d, j_cap),
                inst_k,
            )
            inst_left = (ladder_peak_int * (inst_j + jnp.int32(1))
                         - filled_int)
            pass_price = jnp.where(
                stepped, price_int + inst_j * ladder_ppd_int, price_int)
            pass_qty = jnp.where(
                stepped,
                jnp.minimum(remaining, inst_left.astype(jnp.float32)),
                remaining,
            )
        d_qty_int = _cap_pass_qty(lob, side, pass_price, pass_qty,
                                  iceberg_meta, n_active)
        # If d_qty_int <= 0 we should not dispatch; instead, terminate.
        nothing_left = d_qty_int <= jnp.int32(0)

        # Re-derive jaxob's full_book_flag on the
        # pre-dispatch ``lob`` resting side. Rest-guarded by ``~nothing_left``
        # so a no-dispatch iteration cannot false-positive. The mask is a
        # ``&``, not a ``lax.cond`` (e3).
        book_full = book_full | (_resting_side_full(lob, side) & (~nothing_left))

        d_time_s = jnp.asarray(dispatch_ts, jnp.int32)
        d_time_ns = jnp.int32(0)

        d_msg = _build_jaxob_msg_traced(
            msg_type=jnp.int32(1),  # LIMIT
            side=jaxob_side,
            quantity=d_qty_int,
            price=pass_price,
            order_id=order_id,
            agent_id=agent_id,
            time_s=d_time_s,
            time_ns=d_time_ns,
        )

        trades_before = lob.trades

        # Dispatch only if there's something left.
        lob_after = jax.lax.cond(
            nothing_left,
            lambda args: args[0],
            lambda args: _dispatch_jaxob_msg(args[0], args[1], jaxob_cfg),
            (lob, d_msg),
        )

        trades_after = lob_after.trades
        fills = extract_fills(trades_before, trades_after)
        # Cast int64 -> float32 to keep the carry dtype-stable (see
        # init_total_qty comment above for rationale).
        f_qty = jnp.asarray(fills.total_qty_int, jnp.float32)
        f_num = jnp.asarray(fills.num_fills, jnp.int32)
        f_wpx = jnp.asarray(fills.weighted_price_int, jnp.float32)

        any_fills = (f_num > jnp.int32(0)) & (~nothing_left)

        zero_f = jnp.float32(0.0)
        new_total_qty = total_qty + jnp.where(any_fills, f_qty, zero_f)
        new_total_value = total_value + jnp.where(any_fills, f_wpx, zero_f)
        new_remaining = remaining - jnp.where(any_fills, f_qty, zero_f)

        # If no fills (and we did dispatch): the aggressive order rests in
        # the book. Regenerate slices once and terminate.
        # If fills: cancel aggressive remainder, regenerate slices,
        # advance dispatch_ts, continue.
        def no_fill_path(args):
            lob_in, iceberg_meta_in, n_active_in, ts_in, trunc_in = args
            # The regen is gated out of no-fill
            # iterations. With cascade quiescence guaranteed at the
            # source (regenerate_slices' bounded fixed-point + the
            # _ice_path/fill_path regen callers), no active slot is left
            # in peak_consumed=True between aggressor arrivals, so a
            # no-fill iteration's regen is a guaranteed no-op and is
            # skipped. ``no_fill_path`` is reached only as the false arm
            # of the outer ``lax.cond(any_fills, fill_path, no_fill_path)``
            # below, i.e. ``any_fills`` is False by construction here — so
            # the state passes through unchanged (identity regen).
            return (lob_in, iceberg_meta_in, n_active_in, ts_in, trunc_in,
                    jnp.bool_(False), jnp.bool_(True))

        def fill_path(args):
            lob_in, iceberg_meta_in, n_active_in, ts_in, trunc_in = args
            # Cancel aggressive remainder.
            lob_c, meta_c, n_active_c, ts_c, trunc_c = _jit_cancel_by_id(
                lob_in, order_id, side, iceberg_meta_in, n_active_in,
                ts_in, trunc_in, iceberg_mgr, jaxob_cfg,
            )
            # Regenerate. Exhaustion-order stamps: pass the
            # pre-dispatch book so the slices of peaks exhausted in this
            # pass are stamped in exhaustion order (eager activation,
            # market-spec v2.14 §4.2.1 Rule 4).
            lob_r, meta_r, n_active_r, ts_r = iceberg_mgr.regenerate_slices(
                lob_c, meta_c, n_active_c, ts_c, prior_lob=lob,
                aggressor_side=side,
            )
            # Advance dispatch_ts via next_timestamp.
            # The counter dispatch stamps are
            # internal fill disambiguators (the whole-row ``extract_fills``
            # needs distinct tape rows under saturation); the book-visible
            # stamp of a resting NON residual is restored to
            # ``order.timestamp`` per market-spec §5.2 after the loop
            # (``_restamp_resting_residual``).
            _, ts_advanced = iceberg_mgr.next_timestamp(ts_r)
            return (lob_r, meta_r, n_active_r, ts_advanced, trunc_c,
                    jnp.bool_(True), jnp.bool_(False))

        (new_lob_after, new_meta_after, new_n_active_after,
         new_ts_after, new_trunc_after, ac_after, term_after) = jax.lax.cond(
            any_fills,
            fill_path,
            no_fill_path,
            (lob_after, iceberg_meta, n_active, ts_counter, prior_trunc),
        )

        terminate_flag = term_after | nothing_left

        # When fill_path was taken, dispatch_ts advances (ts_advanced - 1);
        # on no_fill or nothing_left, dispatch_ts stays.
        new_dispatch_ts = jnp.where(
            any_fills,
            jnp.asarray(new_ts_after - jnp.int32(1), jnp.int32),
            dispatch_ts,
        )

        return (
            new_lob_after,
            new_meta_after,
            new_n_active_after,
            new_ts_after,
            new_trunc_after,
            new_remaining,
            new_total_qty,
            new_total_value,
            new_dispatch_ts,
            ac_after,
            it + jnp.int32(1),
            terminate_flag,
            book_full,
        )

    # Run the OUTER multi-slice / iceberg-regeneration
    # loop as a bounded ``fori_loop`` + ``lax.cond``-identity early-out (shared
    # ``_run_bounded_match_loop``; see the module-level ``_TRIP_BOUND`` note for
    # the trip-count / truncation decoupling and the structural-ceiling
    # rationale).
    final_carry = _run_bounded_match_loop(still_running, body_fn, init_carry)

    # Truncation: if we exhaust the loop's trips (or the config truncation cap)
    # and there's still remaining quantity, set the truncated flag.
    #
    # HARD CONSTRAINT: ``hit_cap`` fires against BOTH the
    # config truncation cap ``max_iters`` (== max_match_iterations, 200) AND the
    # loop's own ``_TRIP_BOUND`` (100). The ``_TRIP_BOUND`` guard is the safety
    # fix for the decoupled trip count: because the loop runs only 100 real
    # trips, a hypothetical input needing >100 outer trips would stop at 100 with
    # remaining > 0 but ``iter_f >= max_iters`` (keyed to 200) would be False — a
    # SILENT truncation with no ``truncated`` flag. OR-ing in ``iter_f >=
    # _TRIP_BOUND`` makes any such over-run LOUD (flagged), never silent. Since
    # _TRIP_BOUND=100 <= max_iters=200, this reduces to ``(iter_f >= _TRIP_BOUND)
    # & (remaining > 0)`` in practice; both terms are written so the intent is
    # explicit. ``iter_f`` counts only real (unmasked) trips. _TRIP_BOUND=100
    # is above every realized trip count: 67 (per-step batch-max on a short
    # vmapped tape, throughput measurement) and, with the incoming ICEwPI
    # instance-chain sweep and its multi-instance pass, 13 (replay
    # contract 81, 2026-01-12), 27 (contract 58), 50 (the ten contracts of the
    # highest-instance real ICEwPI entries, 2025-10-10 / 2025-10-27; max over
    # session-ring and emptied-ring runs), 14 (all 2,200 C15 fuzz draws) and 16
    # (all C16 draws); and above the tested 70. On those inputs
    # ``iter_f`` never reaches 100, so this is byte-identical to a
    # 200-trip while_loop's ``hit_cap``. The structural ceiling (524288, see
    # ``_TRIP_BOUND``) keeps the guard a LIVE, loud backstop: the adversarial
    # C14c construction (``C14_seed300000``, four stacked icebergs) reaches the
    # bound by design under the cap and is flagged ``truncated``.
    (_lob_f, _meta_f, _na_f, _ts_f, trunc_f,
     remaining_f, _tq_f, _tv_f,
     _dts_f, _ac_f,
     iter_f, _term_f, _bf_f) = final_carry
    hit_cap = (
        (iter_f >= max_iters) | (iter_f >= jnp.int32(_TRIP_BOUND))
    ) & (remaining_f > jnp.float32(0.0))
    trunc_final = jnp.logical_or(trunc_f, hit_cap)
    final_carry = (
        _lob_f, _meta_f, _na_f, _ts_f, trunc_final,
        remaining_f, _tq_f, _tv_f,
        _dts_f, _ac_f,
        iter_f, _term_f, _bf_f,
    )

    # Post-loop:
    # - IOC: cancel any residual at the original order_id.
    # - NON: if aggressive_cancelled and remaining_qty > 0, re-submit
    #   the residual.
    (lob_f, meta_f, n_active_f, ts_f, trunc_f,
     remaining_f, total_qty_f, total_value_f,
     _dts_f, ac_f,
     _iter_f, _term_f, book_full_f) = final_carry

    # IOC path: cancel residual unconditionally.
    def ioc_post(args):
        lob_, meta_, n_active_, ts_, trunc_ = args
        return _jit_cancel_by_id(
            lob_, order_id, side, meta_, n_active_, ts_, trunc_,
            iceberg_mgr, jaxob_cfg,
        )

    def non_post(args):
        lob_, meta_, n_active_, ts_, trunc_ = args
        # NON: re-submit residual if aggressive_cancelled and remaining > 0.
        need_resubmit = ac_f & (remaining_f > jnp.float32(0.0))
        rest_qty_int = jnp.asarray(jnp.round(remaining_f), jnp.int32)
        # If quantity is 0 in tick units, skip dispatch even if need_resubmit.
        will_dispatch = need_resubmit & (rest_qty_int > jnp.int32(0))

        def do_resubmit(inner_args):
            lob_in, meta_in, n_active_in, ts_in, trunc_in = inner_args
            rest_ts, ts_next = iceberg_mgr.next_timestamp(ts_in)
            rest_msg = _build_jaxob_msg_traced(
                msg_type=jnp.int32(1),
                side=jaxob_side,
                quantity=rest_qty_int,
                price=price_int,
                order_id=order_id,
                agent_id=agent_id,
                time_s=jnp.asarray(rest_ts, jnp.int32),
                time_ns=jnp.int32(0),
            )
            trades_before = lob_in.trades
            # The residual re-submit is a genuine
            # resting add — re-derive the pre-dispatch full-side predicate.
            resubmit_full = _resting_side_full(lob_in, side)
            lob_out = _dispatch_jaxob_msg(lob_in, rest_msg, jaxob_cfg)
            trades_after = lob_out.trades
            return (lob_out, meta_in, n_active_in, ts_next, trunc_in,
                    trades_before, trades_after, resubmit_full)

        def no_resubmit(inner_args):
            lob_in, meta_in, n_active_in, ts_in, trunc_in = inner_args
            # Returned trades pair = (lob_in.trades, lob_in.trades) so
            # extract_fills yields zeros. No resting add -> no book_full.
            return (lob_in, meta_in, n_active_in, ts_in, trunc_in,
                    lob_in.trades, lob_in.trades, jnp.bool_(False))

        (lob_post, meta_post, n_active_post, ts_post, trunc_post,
         tb, ta, bf_post) = jax.lax.cond(
            will_dispatch, do_resubmit, no_resubmit,
            (lob_, meta_, n_active_, ts_, trunc_),
        )
        # Account for any fills that the re-submit produced.
        return (lob_post, meta_post, n_active_post, ts_post, trunc_post,
                tb, ta, bf_post)

    if is_ioc:
        # IOC post = cancel residual (a removal, never a resting add):
        # contributes no book_full.
        (lob_after_post, meta_after_post, n_active_after_post,
         ts_after_post, trunc_after_post) = ioc_post(
            (lob_f, meta_f, n_active_f, ts_f, trunc_f)
        )
        # IOC has no NON re-submit branch; trades_before == trades_after.
        trades_before_post = lob_after_post.trades
        trades_after_post = lob_after_post.trades
        book_full_post = jnp.bool_(False)
    else:
        (lob_after_post, meta_after_post, n_active_after_post,
         ts_after_post, trunc_after_post,
         trades_before_post, trades_after_post,
         book_full_post) = non_post(
            (lob_f, meta_f, n_active_f, ts_f, trunc_f)
        )

    post_fills = extract_fills(trades_before_post, trades_after_post)
    total_qty_final = total_qty_f + jnp.asarray(
        post_fills.total_qty_int, jnp.float32
    )
    total_value_final = total_value_f + jnp.asarray(
        post_fills.weighted_price_int, jnp.float32
    )

    book_full_final = book_full_f | book_full_post

    if not is_ioc:
        # Remainder timestamp: a NON residual that rests after a fill
        # (natural last-pass rest or ``do_resubmit``) keeps its ORIGINAL
        # entry timestamp (market-spec §5.2). Gated on the call having
        # filled, so the no-fill path stays a byte no-op.
        lob_after_post = _restamp_resting_residual(
            lob_after_post, order_id, side, timestamp,
            total_qty_final > jnp.float32(0.0),
        )
        # Unique stamp, NON site: a resting NON entry (fill-then-rest
        # residual or no-fill rest) is ranked behind its same-(price, SEC)
        # peers (market-spec §5.2). Ungated: a no-op when nothing rested.
        lob_after_post = _rank_nsec(
            lob_after_post, order_id, side, price_int, timestamp,
        )
        # Slice activation stamps (NON site): slices this submit
        # activated are stamped at the event second, after the aggressor's
        # own rows. Byte no-op when nothing regenerated.
        lob_after_post = _stamp_regenerated_slices(
            lob_after_post, ts_init, ts_after_post, timestamp,
        )

    return (
        lob_after_post, meta_after_post, n_active_after_post,
        ts_after_post, trunc_after_post,
        total_qty_final, total_value_final,
        book_full_final,
    )


def _while_loop_match_fok(
    state_lob, state_iceberg, state_n_active, state_ts, state_trunc,
    order, product_config, market_config,
    iceberg_mgr, jaxob_cfg,
):
    """The FOK multi-slice matching loop (the jitted form of an earlier
    eager ``_fok_multi_slice_fill``).

    Only invoked when the FOK pre-check has accepted the order.

    ``n_active`` is in the carry
    pytree and threaded through the cancel / regenerate calls.

    A ``book_full`` accumulator rides the carry
    (final element) and ORs in the pre-dispatch full-side predicate at the
    FOK dispatch site, rest-guarded by ``~nothing_left``. Returned to the
    caller for OR-ing into the ``book_full`` state leaf.

    Iceberg price priority (market-spec §6.1): each pass dispatches
    ``_cap_pass_qty(...)``, the same per-pass cap as ``_while_loop_match``,
    so the FOK never walks a worse level while a better-priced resting
    iceberg still has a slice to regenerate.
    """
    max_iters = jnp.int32(market_config.max_match_iterations)
    quantity_int = _jit_encode_qty(order.quantity, product_config)
    price_int = _jit_encode_price(order.price, product_config)
    side = jnp.asarray(order.side, jnp.int32)
    jaxob_side = _jit_encode_side(side)
    order_id = jnp.asarray(order.order_id, jnp.int32)
    agent_id = jnp.asarray(order.agent_id, jnp.int32)
    timestamp = jnp.asarray(order.timestamp, jnp.int32)

    ts_init = iceberg_mgr.update_timestamp(state_ts, timestamp)

    init_carry = (
        state_lob, state_iceberg, state_n_active, ts_init, state_trunc,
        jnp.asarray(quantity_int, jnp.float32),       # remaining
        jnp.float32(0.0),                             # total_qty
        jnp.float32(0.0),                             # total_value
        timestamp,                                    # dispatch_ts
        jnp.int32(0),                                 # iter
        jnp.bool_(False),                             # terminate
        jnp.bool_(False),                             # book_full
    )

    def still_running(carry):
        # Fori_loop early-out predicate — a while_loop
        # ``cond_fn`` minus the ``it < max_iters`` clause (subsumed by the
        # fixed ``TRIP_BOUND=100``; truncation enforced in ``hit_cap`` against
        # both max_iters and TRIP_BOUND — see ``_while_loop_match``).
        (_lob, _meta, _na, _ts, _trunc, remaining,
         _tq, _tv, _dts, _it, terminate, _bf) = carry
        return (remaining > jnp.float32(0.0)) & (~terminate)

    def body_fn(carry):
        (lob, iceberg_meta, n_active, ts_counter, prior_trunc,
         remaining, total_qty, total_value,
         dispatch_ts, it, terminate, book_full) = carry

        # Iceberg price priority: same per-pass cap as the NON/IOC loop.
        d_qty_int = _cap_pass_qty(lob, side, price_int,
                                  jnp.asarray(jnp.round(remaining), jnp.int32),
                                  iceberg_meta, n_active)
        nothing_left = d_qty_int <= jnp.int32(0)

        # Re-derive the pre-dispatch full-side
        # predicate, rest-guarded by ~nothing_left. The FOK dispatch fires
        # jaxob's worst-price eviction whenever the resting side is full at
        # dispatch time (even though FOK then cancels the remainder), so the
        # corruption is observable.
        book_full = book_full | (_resting_side_full(lob, side) & (~nothing_left))

        d_msg = _build_jaxob_msg_traced(
            msg_type=jnp.int32(1),
            side=jaxob_side,
            quantity=d_qty_int,
            price=price_int,
            order_id=order_id,
            agent_id=agent_id,
            time_s=jnp.asarray(dispatch_ts, jnp.int32),
            time_ns=jnp.int32(0),
        )

        trades_before = lob.trades
        lob_after = jax.lax.cond(
            nothing_left,
            lambda args: args[0],
            lambda args: _dispatch_jaxob_msg(args[0], args[1], jaxob_cfg),
            (lob, d_msg),
        )
        trades_after = lob_after.trades
        fills = extract_fills(trades_before, trades_after)
        f_qty = jnp.asarray(fills.total_qty_int, jnp.float32)
        f_num = jnp.asarray(fills.num_fills, jnp.int32)
        f_wpx = jnp.asarray(fills.weighted_price_int, jnp.float32)

        any_fills = (f_num > jnp.int32(0)) & (~nothing_left)

        zero_f = jnp.float32(0.0)
        new_total_qty = total_qty + jnp.where(any_fills, f_qty, zero_f)
        new_total_value = total_value + jnp.where(any_fills, f_wpx, zero_f)
        new_remaining = remaining - jnp.where(any_fills, f_qty, zero_f)

        # FOK always cancels the resting remainder after each dispatch.
        lob_c, meta_c, n_active_c, ts_c, trunc_c = _jit_cancel_by_id(
            lob_after, order_id, side, iceberg_meta, n_active,
            ts_counter, prior_trunc, iceberg_mgr, jaxob_cfg,
        )
        # Gate the regen out of no-fill
        # iterations. Cascade quiescence (regenerate_slices' bounded
        # fixed-point at the source callers) guarantees no slot is left
        # peak_consumed=True between aggressor arrivals, so a no-fill
        # FOK iteration's regen is a guaranteed no-op.
        lob_r, meta_r, n_active_r, ts_r = jax.lax.cond(
            any_fills,
            # Exhaustion-order stamps.
            lambda c: iceberg_mgr.regenerate_slices(
                *c, prior_lob=lob, aggressor_side=side,
            ),
            lambda c: c,
            (lob_c, meta_c, n_active_c, ts_c),
        )

        # Termination conditions:
        #   - nothing_left (we did not dispatch)
        #   - any_fills was False (no progress this iter; no more matches)
        terminate_flag = nothing_left | (~any_fills)

        # Advance dispatch_ts via next_timestamp for next iter.
        _, ts_advanced = iceberg_mgr.next_timestamp(ts_r)
        new_dispatch_ts = jnp.asarray(ts_r, jnp.int32)

        return (
            lob_r, meta_r, n_active_r, ts_advanced, trunc_c,
            new_remaining, new_total_qty, new_total_value,
            new_dispatch_ts,
            it + jnp.int32(1),
            terminate_flag,
            book_full,
        )

    # Run the FOK OUTER loop as a bounded ``fori_loop`` +
    # ``lax.cond``-identity early-out (shared ``_run_bounded_match_loop``, same
    # restructure as ``_while_loop_match``; see the module-level ``_TRIP_BOUND``
    # note for the trip-count / truncation decoupling and structural ceiling).
    final_carry = _run_bounded_match_loop(still_running, body_fn, init_carry)
    (lob_f, meta_f, n_active_f, ts_f, trunc_f,
     remaining_f, total_qty_f, total_value_f,
     _dts_f, iter_f, _term_f, book_full_f) = final_carry
    # HARD CONSTRAINT: ``hit_cap`` fires against BOTH the
    # config truncation cap ``max_iters`` (== max_match_iterations, 200) AND the
    # loop's own ``_TRIP_BOUND`` (100) — the safety guard for the decoupled trip
    # count. Because the loop runs only 100 real trips, an input needing >100
    # outer trips would stop at 100 with remaining > 0; OR-ing in ``iter_f >=
    # _TRIP_BOUND`` makes that over-run LOUD (flagged ``truncated``), never a
    # silent drop. _TRIP_BOUND=100 is above every realized trip count (67 on
    # a short vmapped tape; with the per-pass price-priority cap, 14 /
    # 24 on replay contracts 81 / 58 and 14 on the C15 fuzz draws) and the
    # tested 70, so on those inputs this is byte-identical to a 200-trip
    # while_loop's ``hit_cap``; the structural ceiling (524288) keeps the guard a LIVE
    # backstop on adversarial inputs. See ``_while_loop_match`` /
    # ``_TRIP_BOUND`` for the full rationale.
    hit_cap = (
        (iter_f >= max_iters) | (iter_f >= jnp.int32(_TRIP_BOUND))
    ) & (remaining_f > jnp.float32(0.0))
    trunc_final = jnp.logical_or(trunc_f, hit_cap)
    # Slice activation stamps (FOK site): once, after the loop.
    lob_f = _stamp_regenerated_slices(lob_f, ts_init, ts_f, timestamp)
    return (lob_f, meta_f, n_active_f, ts_f, trunc_final,
            total_qty_f, total_value_f, book_full_f)


# ---------------------------------------------------------------------------
# Module-level jitted submit_order
# ---------------------------------------------------------------------------


def _submit_order_body(
    state: OMIEBookState,
    order: OMIEOrder,
    product_config: OMIEProductConfig,
    market_config: OMIEMarketConfig,
):
    """The body of submit_order — JIT-safe, no Python-side branches on
    traced order fields.

    Returns a ``(state, SubmitResult)`` tuple where SubmitResult fields
    are 0-d ``jnp.*`` scalars. The four ``ExecRestriction`` variants
    are dispatched via ``jax.lax.switch`` on a traced ``jnp.int32``
    discriminator so a single compiled artifact covers all four.

    Args:
        state: ``OMIEBookState``.
        order: ``OMIEOrder`` whose float / int fields are traced.
        product_config: static ``OMIEProductConfig`` (passed as
            ``static_argnames`` from the outer jitted callable).
        market_config: static ``OMIEMarketConfig``.

    Notes:
        - ``contract_open`` is baked to ``True`` inside the JIT body
          (the class-method trampoline does the OPEN guard Python-side
          per the caller-contract invariant in ``jax_constants.py``).
        - Iceberg / OrderBook configuration objects are reconstructed
          inside the body from the static configs; they are stateless
          and compile-time-resolvable.
    """
    jaxob_cfg = _make_jaxob_cfg(market_config)
    # OrderBook is a pytree with empty children + aux_data={cfg}. We do
    # not need the instance for any operation in the JIT body since we
    # call the public-named ``_dispatch_jaxob_msg(lob, msg, cfg)`` helper
    # directly (no private lifted-function import).
    ob = OrderBook(cfg=jaxob_cfg)
    iceberg_mgr = IcebergManager(ob, product_config)

    # --- Pre-validation ----------------------------------------------------
    pos_id = jnp.asarray(order.order_id, jnp.int32) > jnp.int32(0)
    # contract_open is implicit-True inside the JIT body — the class-method
    # trampoline gates this Python-side per the caller-contract invariant.
    validator_ok = validate_order(order, product_config, True)
    valid = pos_id & validator_ok

    rejected = _rejected_result()

    def rejected_branch(state_order):
        st, _ord = state_order
        return st, rejected

    def accepted_branch(state_order):
        st, ord_ = state_order

        # Dispatch on exec_restriction via lax.switch.
        discriminator = jnp.asarray(ord_.exec_restriction, jnp.int32)

        # ONE IOC sweep instance, shared
        # by the IOC and ICE arms and hoisted above the switch. Under ``vmap``
        # ``lax.switch`` traces every arm, so a separate sweep inside the ICE
        # arm would be a third traced copy of ``_while_loop_match`` and pushes
        # a vmapped CPU compile past ``vm.max_map_count``. NON and FOK take
        # the passthrough arm (state unchanged, zero accumulators).
        def do_sweep(state_order_):
            s_, o_ = state_order_
            return _run_match_loop(s_, o_, product_config, market_config,
                                   iceberg_mgr, jaxob_cfg, is_ioc=True)

        def no_sweep(state_order_):
            s_, _o = state_order_
            return (s_.lob, s_.iceberg_meta,
                    jnp.asarray(s_.n_active, jnp.int32),
                    jnp.asarray(s_.ts_counter, jnp.int32),
                    jnp.asarray(s_.truncated, jnp.bool_),
                    jnp.float32(0.0), jnp.float32(0.0), jnp.bool_(False))

        sweep = jax.lax.cond(
            (discriminator == jnp.int32(ExecRestriction.IOC))
            | (discriminator == jnp.int32(ExecRestriction.ICE)),
            do_sweep, no_sweep, (st, ord_),
        )

        def non_branch(args):
            return _non_or_ioc_path(
                args[0], args[1], product_config, market_config,
                iceberg_mgr, jaxob_cfg, is_ioc=False,
            )

        def ioc_branch(args):
            # Slice activation stamps (IOC site): restamp the sweep's
            # slices here, not inside the shared sweep (the ICE arm shares
            # it and must rank its first slice before any sweep slice).
            st_, o_, sw_ = args
            c0_ = iceberg_mgr.update_timestamp(st_.ts_counter, o_.timestamp)
            sw_ = (_stamp_regenerated_slices(sw_[0], c0_, sw_[3],
                                             o_.timestamp),) + tuple(sw_[1:])
            return _state_and_result_from_loop(
                st_, sw_, product_config, iceberg_mgr)

        def fok_branch(args):
            return _fok_path(args[0], args[1], product_config, market_config,
                             iceberg_mgr, jaxob_cfg)

        def ice_branch(args):
            return _ice_path(args[0], args[1], args[2], product_config,
                             market_config, iceberg_mgr, jaxob_cfg)

        return jax.lax.switch(
            discriminator,
            (non_branch, ioc_branch, fok_branch, ice_branch),
            (st, ord_, sweep),
        )

    return jax.lax.cond(valid, accepted_branch, rejected_branch, (state, order))


# ---------------------------------------------------------------------------
# Per-variant branch bodies — each returns (OMIEBookState, SubmitResult).
# ---------------------------------------------------------------------------


def _non_or_ioc_path(
    state, order, product_config, market_config, iceberg_mgr, jaxob_cfg, is_ioc,
):
    """NON / IOC path: standard limit-order multi-slice matching loop.

    The two arms differ only in the post-loop residual handling, which is
    parameterised inside ``_while_loop_match`` via the Python-static
    ``is_ioc`` flag (NON re-submits the residual; IOC cancels it).

    ``_submit_order_body`` calls this for NON only.
    The IOC arm reads the shared sweep run above the switch
    (``_state_and_result_from_loop``); ``is_ioc=True`` stays supported for
    direct callers.
    """
    loop_out = _run_match_loop(state, order, product_config, market_config,
                               iceberg_mgr, jaxob_cfg, is_ioc=is_ioc)
    return _state_and_result_from_loop(state, loop_out, product_config,
                                       iceberg_mgr)


def _fok_path(state, order, product_config, market_config, iceberg_mgr, jaxob_cfg):
    """FOK path: pre-check liquidity; if sufficient, fill; else accept-no-trade."""
    lob = state.lob
    iceberg_meta = state.iceberg_meta
    ts_counter = state.ts_counter
    prior_trunc = state.truncated
    n_active = state.n_active
    prior_book_full = state.book_full

    required_qty_int = _jit_encode_qty(order.quantity, product_config)
    jaxob_price = _jit_encode_price(order.price, product_config)
    available = _jit_matchable_volume(
        lob, jnp.asarray(order.side, jnp.int32), jaxob_price,
        iceberg_meta, n_active, iceberg_mgr,
    )

    sufficient = jnp.asarray(available, jnp.int32) >= required_qty_int

    def accepted_no_trade(_args):
        # Accept-no-trade does not add to the book: no per-call book_full.
        new_state = _build_state_jit(
            lob, iceberg_meta, n_active, ts_counter, prior_trunc, iceberg_mgr,
            overflow=jnp.bool_(False),
            prior_book_full=prior_book_full, call_book_full=jnp.bool_(False),
        )
        return new_state, _accepted_no_trade_result()

    def fok_fill(_args):
        (lob_f, meta_f, n_active_f, ts_f, trunc_f,
         total_qty, total_value, call_book_full) = _while_loop_match_fok(
            lob, iceberg_meta, n_active, ts_counter, prior_trunc,
            order, product_config, market_config,
            iceberg_mgr, jaxob_cfg,
        )
        result = _aggregated_result(total_qty, total_value, product_config)
        new_state = _build_state_jit(
            lob_f, meta_f, n_active_f, ts_f, trunc_f, iceberg_mgr,
            overflow=jnp.bool_(False),
            prior_book_full=prior_book_full, call_book_full=call_book_full,
        )
        return new_state, result

    return jax.lax.cond(sufficient, fok_fill, accepted_no_trade, None)


def _run_match_loop(state, order, product_config, market_config, iceberg_mgr,
                    jaxob_cfg, is_ioc):
    """Run ``_while_loop_match`` from ``state``; return its 8-tuple output."""
    return _while_loop_match(
        state.lob, state.iceberg_meta, state.n_active, state.ts_counter,
        state.truncated, order, product_config, market_config,
        iceberg_mgr, jaxob_cfg, is_ioc=is_ioc,
    )


def _state_and_result_from_loop(state, loop_out, product_config, iceberg_mgr):
    """NON / IOC tail: build (new_state, result) from a match-loop output.

    The IOC arm applies this to the shared sweep run once above the switch;
    ``_non_or_ioc_path`` applies it to
    its own loop run.
    """
    (lob_f, meta_f, n_active_f, ts_f, trunc_f,
     total_qty, total_value, call_book_full) = loop_out
    result = _aggregated_result(total_qty, total_value, product_config)
    new_state = _build_state_jit(
        lob_f, meta_f, n_active_f, ts_f, trunc_f, iceberg_mgr,
        overflow=jnp.bool_(False),
        prior_book_full=state.book_full, call_book_full=call_book_full,
    )
    return new_state, result


def _split_incoming_rows(trades_before, trades_after, peak_int, order_id):
    """Split an incoming iceberg's trade rows at its instance boundaries.

    Market-spec §4.2.1 Rule 2, *Incoming
    iceberg* item (2): one trade per (resting counter, incoming instance).
    Pure and fixed-shape; returns one ``(N, 8)`` int32 trade array.

    The call's new rows are contiguous from the first free slot ``n0`` of
    ``trades_before``. Only own rows (``AGRS_OID == order_id``) are split.
    With ``c_i`` the cumulative ``|q|`` of the own rows in slot order and
    ``P = max(peak_int, 1)``, own row i becomes
    ``max(floor((c_i - 1)/P) - floor(c_{i-1}/P) + 1, 1)`` pieces; piece r
    covers ``[max(c_{i-1}, mP), min(c_i, (m+1)P))`` with
    ``m = floor(c_{i-1}/P) + r``, has ``Q = sign(q) * length`` and copies
    every other column. Foreign rows (cascade rows of another aggressor) stay
    one piece each, in place in the sequence.

    All or nothing: the split applies only if ``n0 < N`` and
    ``n0 + sum(pieces) <= N``; otherwise ``trades_after`` is returned
    unchanged. Row sums per (price, OIDs, TIDs) are preserved, so
    ``extract_fills`` totals are unchanged.
    """
    trades_after = jnp.asarray(trades_after, jnp.int32)
    n_rows = trades_after.shape[0]
    # Emptiness column: trade column 4, ``TradesFeat.SEC`` (jaxob spells the
    # same index ``LOBMSGFEAT.OID`` in its first-free-slot test).
    sec_col = cst.TradesFeat.SEC.value
    q_col = cst.TradesFeat.Q.value
    agrs_col = cst.TradesFeat.AGRS_OID.value

    idx = jnp.arange(n_rows, dtype=jnp.int32)
    empty_before = jnp.asarray(trades_before)[:, sec_col] == jnp.int32(-1)
    n0 = jnp.where(jnp.any(empty_before),
                   jnp.argmax(empty_before).astype(jnp.int32),
                   jnp.int32(n_rows))
    is_new = (idx >= n0) & (trades_after[:, sec_col] != jnp.int32(-1))
    own = is_new & (trades_after[:, agrs_col] == jnp.asarray(order_id, jnp.int32))

    p = jnp.maximum(jnp.asarray(peak_int, jnp.int32), jnp.int32(1))
    q = trades_after[:, q_col]
    abs_q = jnp.where(own, jnp.abs(q), jnp.int32(0))
    c_hi = jnp.cumsum(abs_q).astype(jnp.int32)
    c_lo = c_hi - abs_q
    own_pieces = jnp.maximum(
        (c_hi - jnp.int32(1)) // p - c_lo // p + jnp.int32(1), jnp.int32(1))
    pieces = jnp.where(own, own_pieces,
                       jnp.where(is_new, jnp.int32(1), jnp.int32(0)))
    cum_pieces = jnp.cumsum(pieces).astype(jnp.int32)
    total_pieces = cum_pieces[-1]
    fits = (n0 < jnp.int32(n_rows)) & (n0 + total_pieces <= jnp.int32(n_rows))

    # Output slot j >= n0 takes piece k = j - n0: source row via searchsorted.
    k = idx - n0
    src = jnp.searchsorted(cum_pieces, k, side="right").astype(jnp.int32)
    src = jnp.clip(src, 0, n_rows - 1)
    r = k - (cum_pieces[src] - pieces[src])
    m = c_lo[src] // p + r
    lo = jnp.maximum(c_lo[src], m * p)
    hi = jnp.minimum(c_hi[src], (m + jnp.int32(1)) * p)
    piece_q = jnp.sign(q[src]) * (hi - lo)
    piece_rows = trades_after[src]
    piece_rows = piece_rows.at[:, q_col].set(
        jnp.where(own[src], piece_q, q[src]))

    in_prefix = idx < n0
    in_pieces = (k >= jnp.int32(0)) & (k < total_pieces)
    split = jnp.where(in_prefix[:, None], trades_after,
                      jnp.where(in_pieces[:, None], piece_rows,
                                jnp.full_like(trades_after, -1)))
    return jnp.where(fits, split, trades_after)


def _ice_path(state, order, sweep, product_config, market_config, iceberg_mgr,
              jaxob_cfg):
    """ICE arm: sweep the TOTAL, register the remainder, show its first slice.

    OMIE Rules Annex §4.4; market-spec §6.2 item 7;
    OQ-31 Reading B. Order of operations:

    (a) ``sweep`` is the shared IOC pass of ``_while_loop_match`` over the
        order's **total** quantity at its base limit, run above the switch
        before anything is registered (an incoming ICEwPI starts its sweep
        at its base limit). The loop cancels the aggressor's
        remainder before each regen, so a resting iceberg regenerates against
        a remainder-free book and is re-crossed by the next pass as passive
        liquidity. The incoming iceberg is therefore the aggressor on every
        trade and every trade prints at the resting entry's price. The IOC
        post-path removes the unexecuted remainder. As the last write of
        the call, the own rows are split at the
        incoming instance boundaries (one row per (resting counter, incoming
        instance), ``_split_incoming_rows``).
    (b) If the remainder R > 0, register an order view with ``quantity = R``
        (hidden = R - slice). An iceberg swept whole takes no slot and never
        sets ``truncated`` (overflow only when R > 0).
    (c) Dispatch the first slice at ``order.timestamp`` (``NSEC`` 0).
        It continues the instance chain,
        ``v_1 = min(P - u, R)`` at ``b + k_e d``;
        the registered row is rebased to ``b + k_e d`` with hidden
        ``R - v_1`` and ``slice_count`` 0. It cannot cross: the sweep stopped
        at the first instance that could not trade, and it took every
        crossing regenerated slice.
    (d) Run ``regenerate_slices`` once, unconditionally: it heals a
        resting iceberg slice evicted by a full own side during the sweep's
        resting pass or the slice dispatch, keeping a non-crossing entry
        byte-identical to the pre-sweep engine.
    (e) Snapshot ``extract_fills`` over the whole call (the fill report
        covers the sweep and the first slice).

    ``book_full`` is the sweep's own flag OR the pre-dispatch full-side check
    of step (c).
    """
    prior_book_full = state.book_full
    trades_before = state.lob.trades
    (lob_s, meta_s, n_active_s, ts_s, trunc_s,
     total_qty, _total_value, sweep_book_full) = sweep

    # (b) Register the remainder R (in qty ticks) only when R > 0.
    quantity_int = _jit_encode_qty(order.quantity, product_config)
    peak_int = _jit_encode_qty(order.peak_qty, product_config)
    rem_int = quantity_int - jnp.round(total_qty).astype(jnp.int32)
    has_rem = rem_int > jnp.int32(0)
    rem_view = order._replace(quantity=_jit_decode_qty(rem_int, product_config))
    meta_reg, n_active_reg, overflow_reg = iceberg_mgr.register(
        meta_s, n_active_s, rem_view,
    )
    # Market-spec §4.2.1 Rule 2, *Incoming
    # iceberg* item (4): the remainder continues the instance chain. With
    # M swept ticks, k_e = M // P and u = M - k_e P, it shows
    # v_1 = min(P - u, R) at b + k_e d and hides R - v_1 (rebase: the
    # registered row's base becomes b + k_e d, slice_count stays 0, so
    # regeneration prices b + (k_e + 1) d, ...). d = 0 keeps the base.
    swept_int = jnp.round(total_qty).astype(jnp.int32)
    k_e = swept_int // jnp.maximum(peak_int, jnp.int32(1))
    first_qty = jnp.minimum(peak_int - (swept_int - k_e * peak_int), rem_int)
    registered = (~overflow_reg) & has_rem
    reg_slot = jnp.minimum(jnp.asarray(n_active_s, jnp.int32),
                           jnp.int32(meta_s.shape[0] - 1))
    meta_rebased = (
        meta_reg
        .at[reg_slot, COL_REMAINING_HIDDEN].set(
            jnp.maximum(rem_int - first_qty, jnp.int32(0)))
        .at[reg_slot, COL_BASE_PRICE_INT].set(
            meta_reg[reg_slot, COL_BASE_PRICE_INT]
            + k_e * meta_reg[reg_slot, COL_PPD])
    )
    meta_reg = jnp.where(registered, meta_rebased, meta_reg)
    meta_r = jnp.where(has_rem, meta_reg, meta_s)
    n_active_r = jnp.where(has_rem, n_active_reg, n_active_s)
    ice_overflow = overflow_reg & has_rem
    # First slice price: the registered (rebased) row; on an overflowed
    # table the same b + k_e d, unregistered (``truncated`` is set).
    first_price = jnp.where(
        registered, meta_reg[reg_slot, COL_BASE_PRICE_INT],
        _jit_encode_price(order.price, product_config)
        + k_e * iceberg_mgr._encode_ppd_int(order.ppd))

    # (c) First slice v_1 = min(P - u, R) at b + k_e d, at the entry stamp.
    side = jnp.asarray(order.side, jnp.int32)
    msg = _build_jaxob_msg_traced(
        msg_type=jnp.int32(1),
        side=_jit_encode_side(side),
        quantity=first_qty,
        price=first_price,
        order_id=jnp.asarray(order.order_id, jnp.int32),
        agent_id=jnp.asarray(order.agent_id, jnp.int32),
        time_s=jnp.asarray(order.timestamp, jnp.int32),
        time_ns=jnp.int32(0),
    )
    # The slice dispatch is a resting add, guarded
    # by R > 0 (no dispatch, no full-side contribution).
    slice_full = _resting_side_full(lob_s, side) & has_rem
    lob_f = jax.lax.cond(
        has_rem,
        lambda a: _dispatch_jaxob_msg(a[0], a[1], jaxob_cfg),
        lambda a: a[0],
        (lob_s, msg),
    )
    # Unique stamp, ICE site (a'): the first slice rests at
    # (order.timestamp, 0); rank it behind same-(price, SEC) peers
    # (market-spec §5.2). Gated on R > 0, before the heal regen.
    ranked = _rank_nsec(
        lob_f, order.order_id, side, first_price, order.timestamp,
    )
    lob_f = lob_f._replace(
        bids=jnp.where(has_rem, ranked.bids, lob_f.bids),
        asks=jnp.where(has_rem, ranked.asks, lob_f.asks),
    )

    # (d) Heal regen.
    lob_g, meta_g, n_active_g, ts_g = iceberg_mgr.regenerate_slices(
        lob_f, meta_r, n_active_r, ts_s,
    )
    # Slice activation stamps (ICE site): one call covering the sweep
    # slices [c0, ts_s) and the heal slices [ts_s, ts_g), after site (a').
    lob_g = _stamp_regenerated_slices(
        lob_g, iceberg_mgr.update_timestamp(state.ts_counter, order.timestamp),
        ts_g, order.timestamp,
    )

    # The last write of the call. Split the
    # incoming iceberg's own rows at its instance boundaries (all or nothing
    # on the ring); only ``lob.trades`` changes, and the sums do not.
    lob_g = lob_g._replace(trades=_split_incoming_rows(
        trades_before, lob_g.trades, peak_int, order.order_id))

    # (e) Whole-call tape diff.
    fills = extract_fills(trades_before, lob_g.trades)
    result = _fills_to_submit_result(fills, product_config, jnp.bool_(True))
    new_state = _build_state_jit(
        lob_g, meta_g, n_active_g, ts_g, trunc_s, iceberg_mgr,
        overflow=ice_overflow,
        prior_book_full=prior_book_full,
        call_book_full=sweep_book_full | slice_full,
    )
    return new_state, result


def _build_state_jit(lob, iceberg_meta, n_active, ts_counter, prior_truncated,
                     iceberg_mgr, overflow, prior_book_full=False,
                     call_book_full=False):
    """JIT-side analogue of ContractBook._build_state.

    OR the prior_truncated with the iceberg manager's live truncation flag
    and the per-call overflow flag.

    ``n_active`` is a positional argument
    after ``iceberg_meta`` and is propagated through to the bounded
    ``truncation_flag`` call so the sticky-flag accumulator stays
    cap-independent.

    Composes the sticky ``book_full`` leaf as
    ``prior_book_full | call_book_full`` — the prior state's flag (so it
    persists across the per-order scan) OR-ed with the per-call full-side
    predicate accumulated by the dispatching path. Defaults keep the
    leaf False for any caller that omits the flags.
    """
    live = iceberg_mgr.truncation_flag(iceberg_meta, n_active)
    return OMIEBookState(
        lob=lob,
        iceberg_meta=iceberg_meta,
        ts_counter=ts_counter,
        truncated=jnp.logical_or(
            jnp.logical_or(prior_truncated, live),
            jnp.asarray(overflow, jnp.bool_),
        ),
        n_active=jnp.asarray(n_active, jnp.int32),
        book_full=jnp.logical_or(
            jnp.asarray(prior_book_full, jnp.bool_),
            jnp.asarray(call_book_full, jnp.bool_),
        ),
    )


# Module-level jitted lift: one compile per (product_config, market_config).
_submit_order = jax.jit(
    _submit_order_body,
    static_argnames=("product_config", "market_config"),
)


# ---------------------------------------------------------------------------
# ContractBook
# ---------------------------------------------------------------------------


class ContractBook:
    """Per-contract OMIE order book wrapping jaxob.OrderBook.

    Manages: order validation, OMIE<->jaxob field translation,
    cancel-as-full-removal semantics, and modify-as-cancel-resubmit.

    The book itself is stateless — OMIEBookState flows through method
    arguments and return values (functional pattern, JAX-compatible).

    Args:
        contract: The Contract this book serves.
        product_config: OMIE product parameters (price/qty bounds and ticks).
        market_config: Simulator-level parameters (book depth, etc.).
    """

    def __init__(
        self,
        contract: Contract,
        product_config: OMIEProductConfig = OMIEProductConfig(),
        market_config: OMIEMarketConfig = OMIEMarketConfig(),
    ):
        # A runtime book_depth above the
        # hard structural ceiling would let the resting book outgrow the
        # iceberg side-channel array, opening a silent-eviction surface.
        # Fail fast at __init__ rather than allowing the
        # per-call slot-overflow path to absorb every saturating registration.
        # MAX_ICEBERGS was raised from an interim 100 to 8192.
        #
        # The runtime iceberg cap is the configured
        # `market_config.max_icebergs` (the first dim of `iceberg_meta`), which
        # may never exceed MAX_ICEBERGS (= 8192), the hard structural ceiling
        # the packed-layout / regen loops are sized against — so this adds the
        # `max_icebergs <= MAX_ICEBERGS` ceiling check.
        #
        # NOTE: the chained assertion
        # `book_depth <= max_icebergs <= MAX_ICEBERGS` is deliberately NOT
        # used. book_depth (resting orders per side) and max_icebergs (iceberg
        # slots) are independent axes: book_depth=MAX_ICEBERGS=8192 with the
        # DEFAULT max_icebergs=1024 is valid, and so is (book_depth=100,
        # max_icebergs=8), whose 9th iceberg overflows loudly via
        # `truncated`. The checks are the
        # `book_depth <= MAX_ICEBERGS` boundary PLUS the ceiling check
        # `max_icebergs <= MAX_ICEBERGS`.
        assert market_config.max_icebergs <= MAX_ICEBERGS, (
            f"max_icebergs ({market_config.max_icebergs}) must be "
            f"<= MAX_ICEBERGS ({MAX_ICEBERGS}), the hard structural ceiling; "
            f"raising the ceiling requires the "
            f"MAX_ICEBERGS cap raise (jax_constants.py)."
        )
        assert market_config.book_depth <= MAX_ICEBERGS, (
            f"book_depth ({market_config.book_depth}) must be "
            f"<= MAX_ICEBERGS ({MAX_ICEBERGS}); raising this requires "
            f"the MAX_ICEBERGS cap raise (jax_constants.py)."
        )
        self._contract = contract
        self._product_config = product_config
        self._market_config = market_config
        jaxob_cfg = _make_jaxob_cfg(market_config)
        self._ob = OrderBook(cfg=jaxob_cfg)

        # Iceberg logic delegated to IcebergManager (stateless).
        self._iceberg_mgr = IcebergManager(self._ob, product_config)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self) -> OMIEBookState:
        """Initialize an empty order book state for this contract."""
        return OMIEBookState(
            lob=self._ob.init(),
            # Size the meta from the configured
            # `max_icebergs`: shape (market_config.max_icebergs, 8); MAX_ICEBERGS
            # stays the hard structural ceiling asserted at __init__.
            iceberg_meta=IcebergManager.empty_meta(
                self._market_config.max_icebergs
            ),
            ts_counter=jnp.int32(0),
            truncated=jnp.bool_(False),
            n_active=jnp.int32(0),
            book_full=jnp.bool_(False),  # sticky full-side-eviction diagnostic
        )

    # ------------------------------------------------------------------
    # Internal helper — propagate the sticky truncation flag through
    # every state-reconstruction site in cancel / etc.
    # ------------------------------------------------------------------

    def _build_state(
        self, lob, iceberg_meta, ts_counter, prior_truncated,
        n_active, overflow=False, prior_book_full=False,
    ):
        """Construct a new ``OMIEBookState`` with the sticky truncation
        carry-flag updated.

        Every state-reconstruction site
        threads ``n_active`` alongside ``iceberg_meta``. The bounded
        ``truncation_flag(meta, n_active)`` is called here so the
        sticky-flag accumulator stays cap-independent.

        ``prior_book_full`` is propagated through
        unchanged. The Python-side cancel / modify / regenerate callers of
        this method are removals or no-ops on the resting book, so none of
        them sets ``book_full``; they only carry the sticky prior value
        forward.
        """
        live = self._iceberg_mgr.truncation_flag(iceberg_meta, n_active)
        overflow_arr = jnp.asarray(overflow, jnp.bool_)
        return OMIEBookState(
            lob=lob,
            iceberg_meta=iceberg_meta,
            ts_counter=ts_counter,
            truncated=jnp.logical_or(
                jnp.logical_or(prior_truncated, live), overflow_arr,
            ),
            n_active=jnp.asarray(n_active, jnp.int32),
            book_full=jnp.asarray(prior_book_full, jnp.bool_),
        )

    # ------------------------------------------------------------------
    # Submit — trampoline to the module-level jitted lift.
    # ------------------------------------------------------------------

    def submit_order(
        self, state: OMIEBookState, order: OMIEOrder
    ) -> tuple:
        """Submit an OMIE order to the book.

        Trampoline to the module-level ``_submit_order`` jitted function.
        The contract_open guard is
        evaluated Python-side per the caller-contract invariant
        (see ``jax_constants.py``); when the contract is not OPEN, the
        method returns the canonical REJECTED result without entering
        the jitted body.
        """
        # Caller-contract invariant (see ``jax_constants.py``):
        # ``contract_open`` is resolved Python-side via a single
        # ``== ContractStatus.OPEN`` comparison. The result is the
        # closed-over scalar that gates entry into the jitted body.
        # (This Python-side read is allowlisted by the test suite's
        # JIT-scope audit.)
        contract_open = self._contract.status == ContractStatus.OPEN
        if not contract_open:
            return (state, _rejected_result())
        return _submit_order(
            state, order, self._product_config, self._market_config,
        )

    # ------------------------------------------------------------------
    # Cancel  (Python-side, outside the jitted submit path)
    # ------------------------------------------------------------------

    def cancel_order(
        self, state: OMIEBookState, order_id: int, side: int
    ) -> OMIEBookState:
        """Cancel an order by ID (full removal).

        This method remains Python-side (it is not part of the jitted
        submit path).
        """
        lob = state.lob
        iceberg_meta = state.iceberg_meta
        ts_counter = state.ts_counter
        prior_trunc = state.truncated
        n_active = state.n_active

        lookup_side = _omie_side_to_lookup(side)
        order_row = self._ob.get_order(lob, lookup_side, order_id)

        if int(order_row[2]) <= 0:
            return state

        remaining_qty = int(order_row[1])
        price = int(order_row[0])
        jaxob_side = _omie_side_to_jaxob(side)

        msg = _build_jaxob_msg(
            msg_type=2,
            side=jaxob_side,
            quantity=remaining_qty,
            price=price,
            order_id=order_id,
            agent_id=0,
            time_s=0,
            time_ns=0,
        )
        lob = self._ob.process_order_array(lob, msg)
        iceberg_meta, n_active = self._iceberg_mgr.cleanup(
            iceberg_meta, n_active, order_id,
        )
        return self._build_state(
            lob, iceberg_meta, ts_counter, prior_trunc, n_active,
            prior_book_full=state.book_full,
        )

    # ------------------------------------------------------------------
    # Modify
    # ------------------------------------------------------------------

    def modify_order(
        self,
        state: OMIEBookState,
        order_id: int,
        side: int,
        new_order: OMIEOrder,
    ) -> tuple:
        """Modify an order: cancel old, submit new with fresh timestamp."""
        state = self.cancel_order(state, order_id, side)
        state, result = self.submit_order(state, new_order)
        return (state, result)

    # ------------------------------------------------------------------
    # Read-back helper — returns sentinel OMIEOrder on miss.
    # ------------------------------------------------------------------

    def get_order(
        self, state: OMIEBookState, order_id: int, side: int
    ) -> OMIEOrder:
        """Look up an order by ID and return it in OMIE units.

        Returns:
            ``OMIEOrder`` with price/quantity in OMIE units when found;
            otherwise the canonical sentinel ``OMIEOrder`` (with
            ``order_id == SENTINEL_ORDER_ID``) — never ``None``. This
            keeps the return type fixed (no Optional), which jitted callers need.
        """
        lob = state.lob
        lookup_side = _omie_side_to_lookup(side)
        order_row = self._ob.get_order(lob, lookup_side, order_id)

        if int(order_row[2]) <= 0:
            return make_sentinel_order()

        price_eur = _jaxob_price_to_omie(int(order_row[0]), self._product_config)
        qty_mw = _jaxob_qty_to_omie(int(order_row[1]), self._product_config)
        retrieved_order_id = int(order_row[2])
        agent_id = int(order_row[3])
        timestamp = int(order_row[4])

        return make_order(
            contract_id=self._contract.contract_id,
            side=side,
            price=price_eur,
            quantity=qty_mw,
            order_id=retrieved_order_id,
            agent_id=agent_id,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # GCT Expiry
    # ------------------------------------------------------------------

    def expire_at_gct(self, state: OMIEBookState) -> OMIEBookState:
        """Cancel all remaining orders at Gate Closure Time (GCT).

        Resetting ``iceberg_meta`` to the
        all-sentinel empty array also resets ``n_active`` to 0 — the
        contiguous-prefix invariant requires the live count and the
        meta to agree.
        """
        lob = state.lob
        empty_asks = jnp.ones_like(lob.asks) * -1
        empty_bids = jnp.ones_like(lob.bids) * -1
        # Size the reset meta from the configured
        # `max_icebergs` so the `iceberg_meta` leaf shape is invariant across
        # a step (matching `init()`); a `clear_all()` at the MAX_ICEBERGS
        # ceiling would change the pytree leaf shape when the runtime cap is
        # below the ceiling and break the one-compile / vmap invariant.
        new_iceberg_meta = IcebergManager.empty_meta(
            self._market_config.max_icebergs
        )
        new_lob = LobState(
            asks=empty_asks,
            bids=empty_bids,
            trades=lob.trades,
            key=lob.key,
        )
        return OMIEBookState(
            lob=new_lob,
            iceberg_meta=new_iceberg_meta,
            ts_counter=state.ts_counter,
            truncated=state.truncated,
            n_active=jnp.int32(0),
            # book_full is a sticky session-level
            # diagnostic — preserve it across the GCT empty, like truncated.
            book_full=state.book_full,
        )

    # ------------------------------------------------------------------
    # Anonymous LOB view (stays Python-side; see jax_constants.py)
    # ------------------------------------------------------------------

    def get_anonymous_lob(self, state: OMIEBookState) -> tuple:
        """Return anonymous L2 order book: price/qty only, no agent IDs."""
        lob = state.lob
        cfg = self._ob.cfg
        n_levels = cfg.book_depth

        l2 = job.get_L2_state(lob.asks, lob.bids, n_levels, cfg)
        l2_reshaped = l2.reshape(n_levels, 4)

        asks_list = []
        for i in range(n_levels):
            raw_price = int(l2_reshaped[i, 0])
            raw_vol = int(l2_reshaped[i, 1])
            if raw_price >= cfg.maxint or raw_vol <= 0:
                continue
            price_eur = _jaxob_price_to_omie(raw_price, self._product_config)
            qty_mw = _jaxob_qty_to_omie(raw_vol, self._product_config)
            asks_list.append((price_eur, qty_mw))

        bids_list = []
        for i in range(n_levels):
            raw_price = int(l2_reshaped[i, 2])
            raw_vol = int(l2_reshaped[i, 3])
            if raw_price <= -cfg.maxint or raw_vol <= 0:
                continue
            price_eur = _jaxob_price_to_omie(raw_price, self._product_config)
            qty_mw = _jaxob_qty_to_omie(raw_vol, self._product_config)
            bids_list.append((price_eur, qty_mw))

        return (bids_list, asks_list)

    # ------------------------------------------------------------------
    # Agent orders view (stays Python-side; see jax_constants.py)
    # ------------------------------------------------------------------

    def get_agent_orders(self, state: OMIEBookState, agent_id: int) -> list:
        """Return all resting orders belonging to a specific agent.

        market-spec §9.2 "Own orders" (OMIE Rules Annex §4.4): the owner
        sees each iceberg's FULL remaining quantity. For a row that is an
        iceberg's visible slice, ``quantity`` = visible ``Q`` + the hidden
        remainder (``COL_REMAINING_HIDDEN``) of the live slot (index <
        ``n_active``) whose order id AND side match the row. NON rows (no
        slot) report their resting quantity unchanged. Price, side, OID,
        agent and timestamp are the row's own; anonymous LOB views are
        untouched (they still show only the visible peak).
        """
        lob = state.lob
        # Host-side, once per call: (order_id, side) -> hidden ticks of live slots.
        meta = np.asarray(state.iceberg_meta)[: int(state.n_active)]
        hidden_by_oid_side = {
            (int(r[COL_ORDER_ID]), int(r[COL_SIDE])): int(r[COL_REMAINING_HIDDEN])
            for r in meta
        }
        OID_COL = cst.OrderSideFeat.OID.value
        TID_COL = cst.OrderSideFeat.TID.value
        P_COL = cst.OrderSideFeat.P.value
        Q_COL = cst.OrderSideFeat.Q.value
        SEC_COL = cst.OrderSideFeat.SEC.value

        orders = []

        for i in range(lob.bids.shape[0]):
            row = lob.bids[i]
            oid = int(row[OID_COL])
            tid = int(row[TID_COL])
            if oid > 0 and tid == agent_id:
                price_eur = _jaxob_price_to_omie(int(row[P_COL]), self._product_config)
                qty_int = int(row[Q_COL]) + hidden_by_oid_side.get(
                    (oid, int(Side.BUY)), 0)
                qty_mw = _jaxob_qty_to_omie(qty_int, self._product_config)
                timestamp = int(row[SEC_COL])
                orders.append(make_order(
                    contract_id=self._contract.contract_id,
                    side=Side.BUY,
                    price=price_eur,
                    quantity=qty_mw,
                    order_id=oid,
                    agent_id=tid,
                    timestamp=timestamp,
                ))

        for i in range(lob.asks.shape[0]):
            row = lob.asks[i]
            oid = int(row[OID_COL])
            tid = int(row[TID_COL])
            if oid > 0 and tid == agent_id:
                price_eur = _jaxob_price_to_omie(int(row[P_COL]), self._product_config)
                qty_int = int(row[Q_COL]) + hidden_by_oid_side.get(
                    (oid, int(Side.SELL)), 0)
                qty_mw = _jaxob_qty_to_omie(qty_int, self._product_config)
                timestamp = int(row[SEC_COL])
                orders.append(make_order(
                    contract_id=self._contract.contract_id,
                    side=Side.SELL,
                    price=price_eur,
                    quantity=qty_mw,
                    order_id=oid,
                    agent_id=tid,
                    timestamp=timestamp,
                ))

        return orders

    # ------------------------------------------------------------------
    # Agent trades view (stays Python-side; see jax_constants.py)
    # ------------------------------------------------------------------

    def get_agent_trades(self, state: OMIEBookState, agent_id: int) -> list:
        """Return all trades involving a specific agent."""
        lob = state.lob
        P = cst.TradesFeat.P.value
        Q = cst.TradesFeat.Q.value
        SEC = cst.TradesFeat.SEC.value
        PASS_TID = cst.TradesFeat.PASS_TID.value
        AGRS_TID = cst.TradesFeat.AGRS_TID.value

        agent_trades_arr, _ = job.get_agent_trades(lob.trades, agent_id)

        trades_list = []
        for i in range(agent_trades_arr.shape[0]):
            row = agent_trades_arr[i]
            raw_price = int(row[P])
            raw_qty = int(row[Q])
            pass_tid = int(row[PASS_TID])
            agrs_tid = int(row[AGRS_TID])

            if pass_tid == 0 and agrs_tid == 0 and raw_price == 0 and raw_qty == 0:
                continue

            price_eur = _jaxob_price_to_omie(raw_price, self._product_config)
            qty_mw = _jaxob_qty_to_omie(abs(raw_qty), self._product_config)
            timestamp = int(row[SEC])
            trades_list.append((price_eur, qty_mw, timestamp))

        return trades_list

    # ------------------------------------------------------------------
    # Iceberg slice regeneration (thin delegation to IcebergManager)
    # ------------------------------------------------------------------

    def _regenerate_iceberg_slices(self, state: OMIEBookState) -> OMIEBookState:
        """Delegate iceberg slice regeneration to IcebergManager."""
        lob = state.lob
        iceberg_meta = state.iceberg_meta
        ts_counter = state.ts_counter
        prior_trunc = state.truncated
        n_active = state.n_active
        lob, iceberg_meta, n_active, ts_counter = (
            self._iceberg_mgr.regenerate_slices(
                lob, iceberg_meta, n_active, ts_counter,
            )
        )
        return self._build_state(
            lob, iceberg_meta, ts_counter, prior_trunc, n_active,
            prior_book_full=state.book_full,
        )
