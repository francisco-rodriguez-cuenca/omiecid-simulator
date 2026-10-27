"""Locked behavioural test for P2-PERF-8 — A1 fused: skip
``regenerate_slices`` on no-fill loop iterations.

This file pins the byte-identity invariant the production gating must
preserve: wrapping ``regenerate_slices`` in
``lax.cond(any_fills, regen, identity)`` on the NON / IOC and FOK
no-fill code paths must produce final OMIE book state byte-identical
to an unconditional-regen reference path on the canonical OMIE replay
surface.

Test design — locked by the P2-PERF-8 architect review (sprint-status §
"P2-PERF-8 — Architecture Review (2026-05-26)") beat 4(a):

    Production path:  imports the live ``_replay_orders`` from
        ``omiecid_simulator/omiecid/contract_book.py`` (pre- or post-
        P2-PERF-8 — the byte-identity assertion is invariant on which).
    Reference path:   an inline copy of the pre-P2-PERF-8 production
        body (``_ref_while_loop_match`` / ``_ref_while_loop_match_fok``
        / ``_ref_submit_order_body`` / ``_ref_replay_orders``).  These
        re-use every helper from production via direct import (encoders,
        ``_dispatch_jaxob_msg``, ``extract_fills``, ``_jit_cancel_by_id``,
        ``_jit_matchable_volume``, ``_fills_to_submit_result``,
        ``_build_state_jit``, ``_rejected_result``,
        ``_accepted_no_trade_result``, ``_make_jaxob_cfg``,
        ``IcebergManager``, ``OrderBook``) so the reference stays small
        and the only thing the test asserts is the structural-license
        invariant on which production gates ``regenerate_slices``.

Byte-identity tuple — locked at architect §4 (8-tuple, extended from
the card's 6-tuple by the additional ``n_active`` and ``ts_counter``
scalar fields that ``regenerate_slices`` reads/writes):

    (n_fills, total_qty_ticks,
     hash(lob.bids), hash(lob.asks), hash(lob.trades),
     hash(iceberg_meta),
     n_active, ts_counter)

Hashing uses ``hashlib.sha256`` over the raw ``np.asarray(...).tobytes()``
bytes of each int32 array.  Deterministic given a fixed dtype + memory
layout; the comparison is exact-bytes between two runs of the same code
path on the same inputs.

Locked-test coverage tier — narrowed by the orchestrator at Step 3
gate ("P2-PERF-8 — Scope narrowing: locked-test coverage tier
(2026-05-26)").  This file authors **only fast tests**; no
``@pytest.mark.slow`` test under P2-PERF-8:

  1. ``test_synthetic_mixed_byte_identity`` — small constructed
     ``OMIEBookState`` fixture with mixed ICE / NON / IOC / FOK orders
     exercising both the no-fill subset (3 of 6 orders) and the with-
     fill subset (3 of 6 orders).  Locks the 8-tuple byte-identity
     between the gated production path and the inline unconditional-
     regen reference path.  Uses fresh book construction (the existing
     ``conftest.py`` ``book`` fixture is module-scoped which is not
     suitable for a destructive replay; we build fresh per test).

  2. ``test_truncated_real_data_byte_identity`` — bounded slice of
     50 orders from cid 81 on delivery date 2026-01-12 loaded via the
     canonical ``load_omie_orders`` loader path.  Same 8-tuple byte-
     identity invariant against the inline reference.  The 50-order
     slice is the smallest prefix that still exercises both the no-
     fill subset (~89 % of the canonical session) and at least one
     fill iteration.

Both tests must pass today (pre-implementation) because the live
production code is itself unconditional-regen — both the production
path and the inline reference traverse the same loop body.  After the
developer applies the gating in a follow-up commit, the byte-identity
invariant verifies the gated production path still produces identical
state on every replayed input.  This is the correct shape: the test
locks the **invariant**, not the absence of the optimization.

References:
  * docs/omiecid/agile-plan.md §"Story P2-PERF-8 — A1 fused: skip
    `regenerate_slices` on no-fill loop iterations" (story card).
  * docs/omiecid/sprint-status.md §"P2-PERF-8 — Architecture Review
    (2026-05-26)" (locked refinements: 8-tuple, A/B inline, surgical
    sites at contract_book.py:545-575 / :791-794, ``any_fills``
    discriminant at lines 534 / 779).
  * docs/omiecid/sprint-status.md §"P2-PERF-8 — Scope narrowing: locked-
    test coverage tier (2026-05-26)" (fast-tier-only decision).
  * docs/omiecid/market-spec.md §4.2 Rule 3 (regenerated-slice
    observability — extra-regen with nothing to do produces no
    observable trade).
  * omiecid_simulator/omiecid/iceberg_manager.py:367-372 (the
    ``peak_consumed`` / ``need_regen`` / ``need_cleanup`` structural-
    license derivation underpinning the no-op claim).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Production imports — the (potentially-gated) live code path under test.
from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.jaxob.jorderbook import OrderBook
from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    _replay_orders,
    _row_to_order_traced,
    load_omie_orders,
)
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    RawFills,
    _accepted_no_trade_result,
    _build_jaxob_msg_traced,
    _build_state_jit,
    _dispatch_jaxob_msg,
    _fills_to_submit_result,
    _jit_cancel_by_id,
    _jit_encode_price,
    _jit_encode_qty,
    _jit_encode_side,
    _jit_matchable_volume,
    _make_jaxob_cfg,
    _rejected_result,
    _resting_side_full,
    _submit_order,
    extract_fills,
)
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_PEAK_QTY,
    COL_REMAINING_HIDDEN,
    COL_SIDE,
    COL_SLICE_COUNT,
    IcebergManager,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    Status,
    Validity,
)
from omiecid_simulator.omiecid.order_validator import validate_order


# ---------------------------------------------------------------------------
# Module-level pytest marker
# ---------------------------------------------------------------------------
#
# This file jit-compiles two ``_submit_order`` / ``_replay_orders`` lift
# bodies at module scope (``_ref_submit_order`` / ``_ref_replay_orders``
# below) for the byte-identity A/B.  The module-level
# ``pytestmark = pytest.mark.jit`` conforms to the T1.1 jit-marker
# discipline enforced by ``test_jit_marker_lint`` (not an exception — the
# A/B legitimately compiles lift bodies at import time).  Per Cascade
# Adjudication §6 (sprint-status.md §"P2-PERF-8 — Cascade Adjudication
# (2026-05-26)").
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NTRADE_CAP = cst.NTRADE_CAP

# Canonical real-data fixture (architect §3 pick for the locked test
# inputs).  Only the canonical contract (cid 81) is exercised here at
# the fast tier; cid 82 is deferred per the scope-narrowing decision.
RAW_OMIE_DIR = Path(__file__).resolve().parents[2] / "data" / "rawOMIE"
DELIVERY_DATE = "2026-01-12"
CANONICAL_CONTRACT_ID = 81

# Slice size for the truncated-replay test.  50 orders chosen to keep
# the test inside the fast-tier wall-clock budget on GPU 1.  Wall-clock
# is dominated by (a) the pandas CSV loader (~2.4 s, fixed regardless
# of slice), (b) two ``jax.jit`` warm compiles of the full submit-order
# body (~3–4 s combined under the JAX persistent cache hook in
# ``conftest.py``), and (c) the device-side scan over the slice (~ms
# at 50 orders).  Slice size affects (c) only; we keep it at 50 to
# minimise (c) while still empirically transiting both the no-fill
# subset (the 89 % no-fill ratio in the canonical session means ~45
# of 50 orders take the no-fill regen branch) and the with-fill subset
# (~5 orders).
TRUNCATED_REPLAY_SLICE = 50


# ---------------------------------------------------------------------------
# Test helpers — Contract + book construction
# ---------------------------------------------------------------------------

def _make_open_contract(cid: int) -> Contract:
    """Build an OPEN ``Contract`` instance for the given contract id.

    Mirrors ``demos/benchmark_single_contract.make_open_contract``:
    every quarter-hour OMIE contract is ``contract_id * 15`` minutes
    after midnight, with ``gct_minutes = delivery_start - 60``.

    A standalone helper is required because ``conftest.py``'s
    ``open_contract`` is fixed to contract 37 (mid-day) — these tests
    need cid 81 (real-data alignment) and a synthetic OPEN contract
    that is sufficiently in-window for ``validate_order``.
    """
    start = cid * 15
    end = start + 15
    return Contract(
        contract_id=cid,
        delivery_start_minutes=start,
        delivery_end_minutes=end,
        gct_minutes=start - 60,
        status=ContractStatus.OPEN,
    )


# ---------------------------------------------------------------------------
# 8-tuple byte-identity computation
# ---------------------------------------------------------------------------

def _hash_array(arr) -> str:
    """SHA-256 over the raw int32 bytes of a JAX/NumPy array.

    Architect §4 fixes the byte-identity tuple's hash fields to operate
    on raw array bytes.  SHA-256 makes the tuple printable and renders
    any divergence reproducible via the failure message.

    Deterministic given a fixed dtype + memory layout.  JAX device
    arrays are converted to NumPy first via ``np.asarray``; the
    conversion is no-op for host-resident arrays and a single copy for
    device-resident arrays.
    """
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


def _byte_identity_tuple(
    state: OMIEBookState,
    n_fills: int,
    total_qty_ticks: int,
) -> tuple:
    """Compute the byte-identity tuple per architect §4.

    Tuple shape (architect's locked 8-tuple, extended by the
    P3-PREP-REPLAY-1 Part B ``book_full`` leaf as a 9th element):

        (n_fills, total_qty_ticks,
         hash(lob.bids), hash(lob.asks), hash(lob.trades),
         hash(iceberg_meta),
         n_active, ts_counter, book_full)

    ``n_fills`` and ``total_qty_ticks`` are aggregate matching outcomes
    on the replay buffer (architect §4: "aggregate matching outcome"
    pins) computed by the caller; the state-derived elements come from
    ``state``.  ``truncated`` is intentionally omitted — architect
    confirms regen is not a producer of ``OMIEBookState.truncated``;
    including it would add no signal (architect §4 / §6(c)).

    P3-PREP-REPLAY-1 Part B: ``book_full`` (the new 6th ``OMIEBookState``
    leaf) IS a producer of observable state on the resting-add dispatch
    sites the reference now threads symmetrically with production, so it
    is pinned here.  The reference's ``book_full`` is composed identically
    to production's (``prior | per-call``, OR-ing
    ``_resting_side_full(lob, side) & (~nothing_left)`` at each resting
    add); the gate now verifies the two agree byte-for-byte on this leaf
    too — not weakened, strengthened.
    """
    return (
        int(n_fills),
        int(total_qty_ticks),
        _hash_array(state.lob.bids),
        _hash_array(state.lob.asks),
        _hash_array(state.lob.trades),
        _hash_array(state.iceberg_meta),
        int(state.n_active),
        int(state.ts_counter),
        bool(state.book_full),
    )


def _aggregate_replay_buf(buf: jnp.ndarray) -> tuple[int, int]:
    """Compute ``(n_fills, total_qty_ticks)`` over a ``_replay_orders`` buffer.

    The buffer has shape ``(N_orders, NTRADE_CAP, 3) int32`` with
    sentinel ``-1`` in column 0 for unfilled slabs.  ``total_qty_ticks``
    sums column 2 only at non-sentinel rows.
    """
    fill_mask = buf[..., 0] != -1
    n_fills = int(jnp.sum(fill_mask))
    total_qty_ticks = int(jnp.sum(jnp.where(fill_mask, buf[..., 2], 0)))
    return n_fills, total_qty_ticks


# ---------------------------------------------------------------------------
# Inline reference path — unconditional-regen body
# ---------------------------------------------------------------------------
#
# The functions below mirror the pre-P2-PERF-8 production source
# (``omiecid_simulator/omiecid/contract_book.py`` at branch HEAD =
# ``main @ 052d095``) verbatim except for cosmetic renaming (the
# ``_ref_`` prefix).  They invoke ``regenerate_slices`` **unconditionally**
# at the same call sites the gating will protect (architect §1:
# NON / IOC ``no_fill_path`` at production line 547, FOK at production
# lines 791-794).  This is the A/B reference per architect §4(a).
#
# When the developer applies the gating to production, the bodies
# below intentionally do NOT change.  That is the load-bearing point of
# the A/B: production picks up the gate; reference stays unconditional;
# byte-identity verifies the gate is observably equivalent.


def _ref_while_loop_match(
    state_lob, state_iceberg, state_n_active, state_ts, state_trunc,
    order, product_config, market_config,
    iceberg_mgr, jaxob_cfg,
    is_ioc,
):
    """Reference NON/IOC ``while_loop`` body with unconditional regen.

    Carry layout mirrors production verbatim (see
    ``contract_book._while_loop_match`` at branch HEAD).  Locks against
    drift on the structural-license site: on no-fill iterations, the
    ``no_fill_path`` branch calls ``regenerate_slices`` unconditionally.

    P3-PREP-REPLAY-1 Part B: a 0-d ``jnp.bool_`` ``book_full`` accumulator
    rides the carry as the final element, mirroring production verbatim
    (``contract_book._while_loop_match`` lines 544/578/685/795).  Before
    each resting limit-add dispatch it ORs in
    ``_resting_side_full(lob, side) & (~nothing_left)``; the NON residual
    re-submit ORs in its own pre-dispatch predicate.  Returned as the
    8th element so the reference's ``book_full`` leaf stays byte-identical
    to production's.
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
        jnp.asarray(quantity_int, jnp.float32),  # remaining
        jnp.float32(0.0),                         # total_qty
        jnp.float32(0.0),                         # total_value
        timestamp,                                # dispatch_ts
        jnp.bool_(False),                         # aggressive_cancelled
        jnp.int32(0),                             # iter
        jnp.bool_(False),                         # terminate
        jnp.bool_(False),                         # book_full
    )

    def cond_fn(carry):
        (_lob, _meta, _na, _ts, _trunc,
         remaining, _tq, _tv,
         _dts, _ac, it, terminate, _bf) = carry
        return (it < max_iters) & (remaining > jnp.float32(0.0)) & (~terminate)

    def body_fn(carry):
        (lob, iceberg_meta, n_active, ts_counter, prior_trunc,
         remaining, total_qty, total_value,
         dispatch_ts, aggressive_cancelled,
         it, terminate, book_full) = carry

        d_qty_int = jnp.asarray(remaining, jnp.int32)
        nothing_left = d_qty_int <= jnp.int32(0)
        # P3-PREP-REPLAY-1 Part B: re-derive the pre-dispatch full-side
        # predicate, rest-guarded by ``~nothing_left`` (mirrors production
        # _while_loop_match line 578).
        book_full = book_full | (_resting_side_full(lob, side) & (~nothing_left))
        d_time_s = jnp.asarray(dispatch_ts, jnp.int32)

        d_msg = _build_jaxob_msg_traced(
            msg_type=jnp.int32(1),
            side=jaxob_side,
            quantity=d_qty_int,
            price=price_int,
            order_id=order_id,
            agent_id=agent_id,
            time_s=d_time_s,
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

        # No-fill path — unconditional regen (architect-locked reference
        # behaviour; production v5.32 calls this verbatim today).
        def no_fill_path(args):
            lob_in, iceberg_meta_in, n_active_in, ts_in, trunc_in = args
            lob_r, meta_r, n_active_r, ts_r = iceberg_mgr.regenerate_slices(
                lob_in, iceberg_meta_in, n_active_in, ts_in,
            )
            return (lob_r, meta_r, n_active_r, ts_r, trunc_in,
                    jnp.bool_(False), jnp.bool_(True))

        def fill_path(args):
            lob_in, iceberg_meta_in, n_active_in, ts_in, trunc_in = args
            lob_c, meta_c, n_active_c, ts_c, trunc_c = _jit_cancel_by_id(
                lob_in, order_id, side, iceberg_meta_in, n_active_in,
                ts_in, trunc_in, iceberg_mgr, jaxob_cfg,
            )
            # OMIE-REGEN-EXHAUSTION-ORDER-1 (card D6, licensed mirror
            # edit): mirrors production's exhaustion-order stamping keywords.
            lob_r, meta_r, n_active_r, ts_r = iceberg_mgr.regenerate_slices(
                lob_c, meta_c, n_active_c, ts_c, prior_lob=lob,
                aggressor_side=side,
            )
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

        new_dispatch_ts = jnp.where(
            any_fills,
            jnp.asarray(new_ts_after - jnp.int32(1), jnp.int32),
            dispatch_ts,
        )

        return (
            new_lob_after, new_meta_after, new_n_active_after,
            new_ts_after, new_trunc_after,
            new_remaining, new_total_qty, new_total_value,
            new_dispatch_ts, ac_after,
            it + jnp.int32(1),
            terminate_flag,
            book_full,
        )

    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    (_lob_f, _meta_f, _na_f, _ts_f, trunc_f,
     remaining_f, _tq_f, _tv_f,
     _dts_f, _ac_f, iter_f, _term_f, _bf_f) = final_carry
    hit_cap = (iter_f >= max_iters) & (remaining_f > jnp.float32(0.0))
    trunc_final = jnp.logical_or(trunc_f, hit_cap)
    final_carry = (
        _lob_f, _meta_f, _na_f, _ts_f, trunc_final,
        remaining_f, _tq_f, _tv_f,
        _dts_f, _ac_f, iter_f, _term_f, _bf_f,
    )

    (lob_f, meta_f, n_active_f, ts_f, trunc_f,
     remaining_f, total_qty_f, total_value_f,
     _dts_f, ac_f, _iter_f, _term_f, book_full_f) = final_carry

    def ioc_post(args):
        lob_, meta_, n_active_, ts_, trunc_ = args
        return _jit_cancel_by_id(
            lob_, order_id, side, meta_, n_active_, ts_, trunc_,
            iceberg_mgr, jaxob_cfg,
        )

    def non_post(args):
        lob_, meta_, n_active_, ts_, trunc_ = args
        need_resubmit = ac_f & (remaining_f > jnp.float32(0.0))
        rest_qty_int = jnp.asarray(jnp.round(remaining_f), jnp.int32)
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
            # P3-PREP-REPLAY-1 Part B: the residual re-submit is a genuine
            # resting add — re-derive the pre-dispatch full-side predicate
            # (mirrors production _while_loop_match line 746).
            resubmit_full = _resting_side_full(lob_in, side)
            lob_out = _dispatch_jaxob_msg(lob_in, rest_msg, jaxob_cfg)
            trades_after = lob_out.trades
            return (lob_out, meta_in, n_active_in, ts_next, trunc_in,
                    trades_before, trades_after, resubmit_full)

        def no_resubmit(inner_args):
            lob_in, meta_in, n_active_in, ts_in, trunc_in = inner_args
            # No resting add -> no book_full (mirrors production line 757).
            return (lob_in, meta_in, n_active_in, ts_in, trunc_in,
                    lob_in.trades, lob_in.trades, jnp.bool_(False))

        (lob_post, meta_post, n_active_post, ts_post, trunc_post,
         tb, ta, bf_post) = jax.lax.cond(
            will_dispatch, do_resubmit, no_resubmit,
            (lob_, meta_, n_active_, ts_, trunc_),
        )
        return (lob_post, meta_post, n_active_post, ts_post, trunc_post,
                tb, ta, bf_post)

    if is_ioc:
        # IOC post = cancel residual (a removal, never a resting add):
        # contributes no book_full (mirrors production line 778).
        (lob_after_post, meta_after_post, n_active_after_post,
         ts_after_post, trunc_after_post) = ioc_post(
            (lob_f, meta_f, n_active_f, ts_f, trunc_f)
        )
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
        # OMIE-STAMP-UNIQUE-1 site (a) mirror (licensed re-baseline,
        # agile-plan v5.198 Deliverable (2)): a resting NON entry is ranked
        # NSEC = 1 + max NSEC among live same-(price, SEC) rows on its side
        # (market-spec §5.2). Calls the PRODUCTION helper; looked up lazily so
        # this module still imports before the helper exists.
        from omiecid_simulator.omiecid import contract_book as _cb
        lob_after_post = _cb._rank_nsec(
            lob_after_post, order_id, side, price_int, timestamp,
        )

    return (
        lob_after_post, meta_after_post, n_active_after_post,
        ts_after_post, trunc_after_post,
        total_qty_final, total_value_final,
        book_full_final,
    )


def _ref_while_loop_match_fok(
    state_lob, state_iceberg, state_n_active, state_ts, state_trunc,
    order, product_config, market_config,
    iceberg_mgr, jaxob_cfg,
):
    """Reference FOK ``while_loop`` body with unconditional regen.

    Mirrors production ``_while_loop_match_fok`` verbatim including
    the regen call at lines 791-794 (architect §1).  When P2-PERF-8
    lands, production wraps lines 791-794 in
    ``lax.cond(any_fills, regen, identity)``; this reference does
    NOT.  The byte-identity invariant verifies they produce identical
    state on the no-fill iterations the FOK loop transits.

    P3-PREP-REPLAY-1 Part B: a 0-d ``jnp.bool_`` ``book_full`` accumulator
    rides the carry as the final element (mirrors production
    ``_while_loop_match_fok`` lines 842/863/928/938), ORing in the
    pre-dispatch full-side predicate rest-guarded by ``~nothing_left``.
    Returned as the 8th element so the reference's ``book_full`` leaf
    stays byte-identical to production's.
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
        jnp.asarray(quantity_int, jnp.float32),
        jnp.float32(0.0),
        jnp.float32(0.0),
        timestamp,
        jnp.int32(0),
        jnp.bool_(False),
        jnp.bool_(False),                             # book_full
    )

    def cond_fn(carry):
        (_lob, _meta, _na, _ts, _trunc, remaining,
         _tq, _tv, _dts, it, terminate, _bf) = carry
        return (it < max_iters) & (remaining > jnp.float32(0.0)) & (~terminate)

    def body_fn(carry):
        (lob, iceberg_meta, n_active, ts_counter, prior_trunc,
         remaining, total_qty, total_value,
         dispatch_ts, it, terminate, book_full) = carry

        d_qty_int = jnp.asarray(jnp.round(remaining), jnp.int32)
        nothing_left = d_qty_int <= jnp.int32(0)

        # P3-PREP-REPLAY-1 Part B: re-derive the pre-dispatch full-side
        # predicate, rest-guarded by ``~nothing_left`` (mirrors production
        # _while_loop_match_fok line 863).
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

        # FOK cancels every iteration (load-bearing per architect §6(b)).
        lob_c, meta_c, n_active_c, ts_c, trunc_c = _jit_cancel_by_id(
            lob_after, order_id, side, iceberg_meta, n_active,
            ts_counter, prior_trunc, iceberg_mgr, jaxob_cfg,
        )
        # Unconditional regen — load-bearing surgical site for the
        # A/B (architect §1, production lines 791-794).
        # OMIE-REGEN-EXHAUSTION-ORDER-1 (card D6, licensed mirror edit):
        # mirrors production's exhaustion-order stamping keywords.
        lob_r, meta_r, n_active_r, ts_r = iceberg_mgr.regenerate_slices(
            lob_c, meta_c, n_active_c, ts_c, prior_lob=lob,
            aggressor_side=side,
        )

        terminate_flag = nothing_left | (~any_fills)

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

    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    (lob_f, meta_f, n_active_f, ts_f, trunc_f,
     remaining_f, total_qty_f, total_value_f,
     _dts_f, _iter_f, _term_f, book_full_f) = final_carry

    return (lob_f, meta_f, n_active_f, ts_f, trunc_f,
            total_qty_f, total_value_f, book_full_f)


def _ref_non_or_ioc_path(
    state, order, product_config, market_config, iceberg_mgr, jaxob_cfg, is_ioc,
):
    """Reference NON/IOC branch.  Delegates to ``_ref_while_loop_match``."""
    # P3-PREP-REPLAY-1 Part B: 6-leaf OMIEBookState unpack (was 5).  The
    # prior ``book_full`` is carried forward and OR-ed with the per-call
    # accumulator (mirrors production ``_non_or_ioc_path`` lines 1045-1069).
    lob, iceberg_meta, ts_counter, prior_trunc, n_active, prior_book_full = state
    (lob_f, meta_f, n_active_f, ts_f, trunc_f,
     total_qty, total_value, call_book_full) = _ref_while_loop_match(
        lob, iceberg_meta, n_active, ts_counter, prior_trunc,
        order, product_config, market_config,
        iceberg_mgr, jaxob_cfg, is_ioc=is_ioc,
    )
    agg_fills = RawFills(
        total_qty_int=total_qty,
        weighted_price_int=total_value,
        num_fills=jnp.where(
            total_qty > jnp.float32(0.0), jnp.int32(1), jnp.int32(0),
        ),
    )
    result = _fills_to_submit_result(agg_fills, product_config, jnp.bool_(True))
    new_state = _build_state_jit(
        lob_f, meta_f, n_active_f, ts_f, trunc_f, iceberg_mgr,
        overflow=jnp.bool_(False),
        prior_book_full=prior_book_full, call_book_full=call_book_full,
    )
    return new_state, result


def _ref_fok_path(state, order, product_config, market_config, iceberg_mgr, jaxob_cfg):
    """Reference FOK branch.  Delegates to ``_ref_while_loop_match_fok``."""
    # P3-PREP-REPLAY-1 Part B: 6-leaf OMIEBookState unpack (was 5); carry the
    # prior ``book_full`` forward (mirrors production ``_fok_path``
    # lines 1075-1119).
    lob, iceberg_meta, ts_counter, prior_trunc, n_active, prior_book_full = state

    required_qty_int = _jit_encode_qty(order.quantity, product_config)
    jaxob_price = _jit_encode_price(order.price, product_config)
    available = _jit_matchable_volume(
        lob, jnp.asarray(order.side, jnp.int32), jaxob_price,
        iceberg_meta, n_active, iceberg_mgr,
    )
    sufficient = jnp.asarray(available, jnp.int32) >= required_qty_int

    def accepted_no_trade(_args):
        # Accept-no-trade does not add to the book: no per-call book_full
        # (mirrors production line 1096).
        new_state = _build_state_jit(
            lob, iceberg_meta, n_active, ts_counter, prior_trunc, iceberg_mgr,
            overflow=jnp.bool_(False),
            prior_book_full=prior_book_full, call_book_full=jnp.bool_(False),
        )
        return new_state, _accepted_no_trade_result()

    def fok_fill(_args):
        (lob_f, meta_f, n_active_f, ts_f, trunc_f,
         total_qty, total_value, call_book_full) = _ref_while_loop_match_fok(
            lob, iceberg_meta, n_active, ts_counter, prior_trunc,
            order, product_config, market_config,
            iceberg_mgr, jaxob_cfg,
        )
        agg_fills = RawFills(
            total_qty_int=total_qty,
            weighted_price_int=total_value,
            num_fills=jnp.where(
                total_qty > jnp.float32(0.0), jnp.int32(1), jnp.int32(0),
            ),
        )
        result = _fills_to_submit_result(agg_fills, product_config, jnp.bool_(True))
        new_state = _build_state_jit(
            lob_f, meta_f, n_active_f, ts_f, trunc_f, iceberg_mgr,
            overflow=jnp.bool_(False),
            prior_book_full=prior_book_full, call_book_full=call_book_full,
        )
        return new_state, result

    return jax.lax.cond(sufficient, fok_fill, accepted_no_trade, None)


def _ref_ice_path(state, order, product_config, market_config, iceberg_mgr, jaxob_cfg):
    """Reference ICE branch.  Identical to production — ICE has no no-
    fill no-op regen risk (always fills the peak or rests it, both
    requiring regen).  Included so the A/B's ``lax.switch`` arm
    coverage matches production verbatim.

    P3-PREP-REPLAY-1 Part B: 6-leaf OMIEBookState unpack (was 5); the
    iceberg peak dispatch is a resting add, so the pre-dispatch full-side
    predicate is re-derived and OR-ed into the prior ``book_full`` (mirrors
    production ``_ice_path`` lines 1125-1178).
    """
    lob, iceberg_meta, ts_counter, prior_trunc, n_active, prior_book_full = state

    iceberg_meta_new, n_active_new, ice_overflow = iceberg_mgr.register(
        iceberg_meta, n_active, order,
    )
    ts_counter_new = iceberg_mgr.update_timestamp(ts_counter, order.timestamp)

    peak_qty_int = _jit_encode_qty(order.peak_qty, product_config)
    price_int = _jit_encode_price(order.price, product_config)
    side = jnp.asarray(order.side, jnp.int32)
    jaxob_side = _jit_encode_side(side)
    order_id = jnp.asarray(order.order_id, jnp.int32)
    agent_id = jnp.asarray(order.agent_id, jnp.int32)
    timestamp = jnp.asarray(order.timestamp, jnp.int32)

    msg = _build_jaxob_msg_traced(
        msg_type=jnp.int32(1),
        side=jaxob_side,
        quantity=peak_qty_int,
        price=price_int,
        order_id=order_id,
        agent_id=agent_id,
        time_s=timestamp,
        time_ns=jnp.int32(0),
    )

    trades_before = lob.trades
    # P3-PREP-REPLAY-1 Part B: the iceberg peak dispatch is a resting add —
    # re-derive the pre-dispatch full-side predicate on the resting side
    # (mirrors production line 1163).
    call_book_full = _resting_side_full(lob, side)
    lob_after = _dispatch_jaxob_msg(lob, msg, jaxob_cfg)
    trades_after = lob_after.trades
    fills = extract_fills(trades_before, trades_after)

    lob_r, meta_r, n_active_r, ts_r = iceberg_mgr.regenerate_slices(
        lob_after, iceberg_meta_new, n_active_new, ts_counter_new,
    )

    result = _fills_to_submit_result(fills, product_config, jnp.bool_(True))
    new_state = _build_state_jit(
        lob_r, meta_r, n_active_r, ts_r, prior_trunc, iceberg_mgr,
        overflow=ice_overflow,
        prior_book_full=prior_book_full, call_book_full=call_book_full,
    )
    return new_state, result


def _ref_submit_order_body(
    state: OMIEBookState,
    order: OMIEOrder,
    product_config: OMIEProductConfig,
    market_config: OMIEMarketConfig,
):
    """Reference ``_submit_order_body`` — unconditional-regen everywhere.

    Verbatim copy of production's ``_submit_order_body`` (modulo
    ``_ref_*`` branch routing).  When P2-PERF-8 lands, production's
    ``_while_loop_match`` and ``_while_loop_match_fok`` gate the regen
    call; the reference's ``_ref_while_loop_match`` /
    ``_ref_while_loop_match_fok`` do NOT.  Byte-identity verifies the
    gating is observably equivalent.
    """
    jaxob_cfg = _make_jaxob_cfg(market_config)
    ob = OrderBook(cfg=jaxob_cfg)
    iceberg_mgr = IcebergManager(ob, product_config)

    pos_id = jnp.asarray(order.order_id, jnp.int32) > jnp.int32(0)
    validator_ok = validate_order(order, product_config, True)
    valid = pos_id & validator_ok

    rejected = _rejected_result()

    def rejected_branch(state_order):
        st, _ord = state_order
        return st, rejected

    def accepted_branch(state_order):
        st, ord_ = state_order
        discriminator = jnp.asarray(ord_.exec_restriction, jnp.int32)

        def non_branch(args):
            return _ref_non_or_ioc_path(
                args[0], args[1], product_config, market_config,
                iceberg_mgr, jaxob_cfg, is_ioc=False,
            )

        def ioc_branch(args):
            return _ref_non_or_ioc_path(
                args[0], args[1], product_config, market_config,
                iceberg_mgr, jaxob_cfg, is_ioc=True,
            )

        def fok_branch(args):
            return _ref_fok_path(args[0], args[1], product_config, market_config,
                                 iceberg_mgr, jaxob_cfg)

        def ice_branch(args):
            return _ref_ice_path(args[0], args[1], product_config, market_config,
                                 iceberg_mgr, jaxob_cfg)

        return jax.lax.switch(
            discriminator,
            (non_branch, ioc_branch, fok_branch, ice_branch),
            (st, ord_),
        )

    return jax.lax.cond(valid, accepted_branch, rejected_branch, (state, order))


# Module-level jit lift of the reference body — same idiom shape as
# production ``_submit_order``.  Cached separately from the production
# lift's compile cache so an A/B run does not contaminate either.
_ref_submit_order = jax.jit(
    _ref_submit_order_body,
    static_argnames=("product_config", "market_config"),
)


def _ref_replay_orders_body(state, orders, product_config, market_config):
    """Reference ``_replay_orders_body`` — drives the unconditional-regen lift.

    Identical scan shape to production's ``_replay_orders_body`` but
    calls ``_ref_submit_order`` instead of the production
    ``_submit_order``.
    """
    P_COL = cst.TradesFeat.P.value
    Q_COL = cst.TradesFeat.Q.value

    def scan_step(carry_state, row):
        order = _row_to_order_traced(row, product_config)
        trades_before = carry_state.lob.trades
        new_state, _result = _ref_submit_order(
            carry_state, order, product_config, market_config,
        )
        trades_after = new_state.lob.trades

        new_mask = jnp.any(trades_before != trades_after, axis=1)
        aggressor_oid = order.order_id.astype(jnp.int32)
        oid_col = jnp.where(new_mask, aggressor_oid, jnp.int32(-1))
        price_col = jnp.where(
            new_mask, trades_after[:, P_COL].astype(jnp.int32), jnp.int32(0),
        )
        qty_col = jnp.where(
            new_mask, jnp.abs(trades_after[:, Q_COL]).astype(jnp.int32), jnp.int32(0),
        )
        step_slab = jnp.stack([oid_col, price_col, qty_col], axis=1)
        return new_state, step_slab

    final_state, replayed_trades_buf = jax.lax.scan(
        scan_step, state, orders,
    )
    return final_state, replayed_trades_buf


_ref_replay_orders = jax.jit(
    _ref_replay_orders_body,
    static_argnames=("product_config", "market_config"),
)


# ---------------------------------------------------------------------------
# Synthetic fixture builder — for the mixed ICE/NON/IOC/FOK byte-identity test
# ---------------------------------------------------------------------------

def _synthetic_mixed_orders(pc: OMIEProductConfig, oid_base: int = 0) -> jnp.ndarray:
    """Construct a small ``(N, 13) int32`` session that exercises both
    the no-fill regen site and the with-fill regen site, across all
    four ``ExecRestriction`` types (NON, IOC, FOK, ICE).

    Order list (chronological; contract id 37 — mid-day OPEN window):

      Row 0: ICE SELL @ 60 EUR/MWh, qty 1.0, peak 0.2.
               -> rests; registers an iceberg slice (with-fill site
                  via the ICE branch's mandatory regen).
      Row 1: NON SELL @ 55 EUR/MWh, qty 0.1.
               -> rests; no fill (transits NON ``no_fill_path``
                  branch -> regen site).
      Row 2: NON BUY  @ 50 EUR/MWh, qty 0.1.
               -> rests; no fill (transits NON ``no_fill_path``
                  branch -> regen site).
      Row 3: IOC BUY  @ 45 EUR/MWh, qty 0.1.
               -> does not cross; IOC residual cancel after no_fill_path
                  (transits IOC ``no_fill_path`` branch -> regen site).
      Row 4: NON BUY  @ 55 EUR/MWh, qty 0.05.
               -> CROSSES the resting sell at 55: with-fill (transits
                  NON ``fill_path`` branch -> regen site is mandatory).
      Row 5: FOK BUY  @ 60 EUR/MWh, qty 0.2.
               -> CROSSES the ICE peak at 60: with-fill (transits
                  the FOK loop body -> regen site at production
                  lines 791-794).

    The session is intentionally small (6 orders) so the test fits the
    fast-suite < 1 s budget per the test-agent runtime rules.  The
    coverage breakdown by branch / no-fill-vs-fill:

      - ICE  with-fill (mandatory regen): row 0.
      - NON  no-fill:                       rows 1, 2.
      - IOC  no-fill:                       row 3.
      - NON  with-fill:                     row 4.
      - FOK  with-fill:                     row 5.

    This is the minimum input set that exercises all four
    ``ExecRestriction`` arms of the ``lax.switch`` in
    ``_submit_order_body`` AND both arms of the inner
    ``lax.cond(any_fills, fill_path, no_fill_path, ...)`` regen-gate
    on the NON/IOC body.
    """
    p_low = int(round((45.00 - pc.price_min) / pc.price_tick))
    p_mid_low = int(round((50.00 - pc.price_min) / pc.price_tick))
    p_mid = int(round((55.00 - pc.price_min) / pc.price_tick))
    p_high = int(round((60.00 - pc.price_min) / pc.price_tick))
    q_small = int(round(0.05 / pc.qty_tick))
    q_med = int(round(0.1 / pc.qty_tick))
    q_full = int(round(1.0 / pc.qty_tick))
    q_peak = int(round(0.2 / pc.qty_tick))

    # FOLLOWUP-2 fixture-OID realism sweep (2026-07-09): ``oid_base``
    # offsets every OID (and its agent_id) so this byte-identity
    # constructor can be driven with realistic OIDs (base 1000) as well
    # as the original tiny OIDs (base 0).  The prod-vs-ref byte-identity
    # comparison is a self-consistency check with NO pinned scalar, so
    # the offset cannot change any pinned expected value — it only
    # proves the identity holds when OIDs are far above every qty in
    # ticks (1.0 MW = 10 ticks), catching the OID-vs-qty-column
    # confusion class.
    b = oid_base
    rows = [
        # Row 0: ICE SELL @ 60, qty 1.0, peak 0.2 (rests; registers iceberg).
        [37, int(Side.SELL), p_high, q_full, b + 1, b + 1, 100,
         int(ExecRestriction.ICE), int(Validity.GFS), 0,
         int(Status.ACTIVE), q_peak, 0],
        # Row 1: NON SELL @ 55, qty 0.1 (rests; no fill).
        [37, int(Side.SELL), p_mid, q_med, b + 2, b + 2, 101,
         int(ExecRestriction.NON), int(Validity.GFS), 0,
         int(Status.ACTIVE), 0, 0],
        # Row 2: NON BUY @ 50, qty 0.1 (rests; no fill).
        [37, int(Side.BUY), p_mid_low, q_med, b + 3, b + 3, 102,
         int(ExecRestriction.NON), int(Validity.GFS), 0,
         int(Status.ACTIVE), 0, 0],
        # Row 3: IOC BUY @ 45, qty 0.1 (does not cross; IOC residual cancel).
        [37, int(Side.BUY), p_low, q_med, b + 4, b + 4, 103,
         int(ExecRestriction.IOC), int(Validity.GFS), 0,
         int(Status.ACTIVE), 0, 0],
        # Row 4: NON BUY @ 55, qty 0.05 (crosses NON sell @ 55 from row 1).
        [37, int(Side.BUY), p_mid, q_small, b + 5, b + 5, 104,
         int(ExecRestriction.NON), int(Validity.GFS), 0,
         int(Status.ACTIVE), 0, 0],
        # Row 5: FOK BUY @ 60, qty 0.2 (crosses ICE peak from row 0).
        [37, int(Side.BUY), p_high, q_peak, b + 6, b + 6, 105,
         int(ExecRestriction.FOK), int(Validity.GFS), 0,
         int(Status.ACTIVE), 0, 0],
    ]
    return jnp.array(rows, dtype=jnp.int32)


# FOLLOWUP-2 fixture-OID realism sweep (2026-07-09): the synthetic
# byte-identity + quiescence-surface constructors above take an
# ``oid_base`` offset; this pair drives them at the original tiny OIDs
# (base 0 → 1..N, numerically inside the qty tick range) and at
# realistic OIDs (base 1000 → 1001..N, far above every qty in ticks).
# The matching/quiescence assertions must hold identically at both
# magnitudes, catching the OID-vs-qty-column confusion class the
# QUIESCENCE-1 audit found.
SWEEP_OID_BASES = [
    pytest.param(0, id="small-oids"),
    pytest.param(1000, id="realistic-oids"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestP2Perf8RegenGateByteIdentity:
    """Locked byte-identity invariant for P2-PERF-8.

    The class holds the two fast tests that pin the gating contract:

      1. ``test_synthetic_mixed_byte_identity``: small synthetic session
         covering both the no-fill subset (3 orders) and the with-fill
         subset (3 orders) across all four ExecRestriction arms.
      2. ``test_truncated_real_data_byte_identity``: 100-order prefix
         of the canonical cid 81 / 2026-01-12 OMIE session, loaded
         via the production loader path.

    Both tests assert the 8-tuple
    ``(n_fills, total_qty_ticks, hash(lob.bids), hash(lob.asks),
       hash(lob.trades), hash(iceberg_meta), n_active, ts_counter)``
    is byte-identical between the production ``_replay_orders``
    (potentially-gated, post-P2-PERF-8) and the inline reference
    ``_ref_replay_orders`` (unconditional regen).

    Today (pre-implementation) production IS the unconditional-regen
    path, so both paths trivially match.  After the developer applies
    the ``lax.cond(any_fills, regen, identity)`` gate, the byte-
    identity invariant gains its load-bearing role.
    """

    @pytest.mark.parametrize("oid_base", SWEEP_OID_BASES)
    def test_synthetic_mixed_byte_identity(self, oid_base):
        """Synthetic 6-order ICE/NON/IOC/FOK session: byte-identity on
        the 8-tuple between production and the unconditional reference.

        Exercises every ``ExecRestriction`` arm of the ``lax.switch``
        in ``_submit_order_body`` AND both arms of the inner
        ``lax.cond(any_fills, fill_path, no_fill_path, ...)`` regen-
        gate on the NON/IOC body.  See ``_synthetic_mixed_orders``
        docstring for the full coverage matrix.

        FOLLOWUP-2 sweep: parametrized over ``SWEEP_OID_BASES`` — the
        prod-vs-ref byte-identity is a self-consistency check (no pinned
        scalar) and must hold at realistic OIDs too.
        """
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        contract = Contract(
            contract_id=37,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.OPEN,
        )
        book_prod = ContractBook(contract, pc, mc)
        book_ref = ContractBook(contract, pc, mc)

        orders = _synthetic_mixed_orders(pc, oid_base)

        # Production (potentially-gated post-P2-PERF-8) path.
        state_prod, buf_prod = _replay_orders(
            book_prod.init(), orders, pc, mc,
        )
        for leaf in jax.tree_util.tree_leaves((state_prod, buf_prod)):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        n_fills_prod, qty_prod = _aggregate_replay_buf(buf_prod)
        tup_prod = _byte_identity_tuple(state_prod, n_fills_prod, qty_prod)

        # Reference (unconditional-regen) path.
        state_ref, buf_ref = _ref_replay_orders(
            book_ref.init(), orders, pc, mc,
        )
        for leaf in jax.tree_util.tree_leaves((state_ref, buf_ref)):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        n_fills_ref, qty_ref = _aggregate_replay_buf(buf_ref)
        tup_ref = _byte_identity_tuple(state_ref, n_fills_ref, qty_ref)

        assert tup_prod == tup_ref, (
            "P2-PERF-8 byte-identity invariant violated on synthetic "
            "mixed ICE/NON/IOC/FOK session:\n"
            f"  production (gated)       ={tup_prod!r}\n"
            f"  reference (unconditional)={tup_ref!r}\n"
            f"  per-field equality:\n"
            f"    n_fills        : prod={tup_prod[0]} ref={tup_ref[0]} "
            f"(equal={tup_prod[0] == tup_ref[0]})\n"
            f"    total_qty_ticks: prod={tup_prod[1]} ref={tup_ref[1]} "
            f"(equal={tup_prod[1] == tup_ref[1]})\n"
            f"    hash(bids)     : equal={tup_prod[2] == tup_ref[2]}\n"
            f"    hash(asks)     : equal={tup_prod[3] == tup_ref[3]}\n"
            f"    hash(trades)   : equal={tup_prod[4] == tup_ref[4]}\n"
            f"    hash(iceberg)  : equal={tup_prod[5] == tup_ref[5]}\n"
            f"    n_active       : prod={tup_prod[6]} ref={tup_ref[6]} "
            f"(equal={tup_prod[6] == tup_ref[6]})\n"
            f"    ts_counter     : prod={tup_prod[7]} ref={tup_ref[7]} "
            f"(equal={tup_prod[7] == tup_ref[7]})\n"
            f"    book_full      : prod={tup_prod[8]} ref={tup_ref[8]} "
            f"(equal={tup_prod[8] == tup_ref[8]})"
        )

    def test_truncated_real_data_byte_identity(self):
        """Truncated real-data slice (50 orders from cid 81 / 2026-01-12):
        byte-identity on the 8-tuple between production and the
        unconditional reference.

        Preserves real-data coverage at the fast tier without the
        wall-clock cost of the 6,954-order session (deferred to a
        follow-up slow-tier story per the scope-narrowing decision).
        The 50-order prefix is empirically sufficient to transit both
        the no-fill subset (the 89 % no-fill ratio in the canonical
        session means ~45 of 50 orders take the no-fill regen branch)
        and the with-fill subset (~5 orders).

        Skipped if the raw OMIE orders file is absent — the fast-tier
        invariant is skip-on-data-absent so the suite remains green
        on data-less CI.
        """
        orders_path = (
            RAW_OMIE_DIR / f"orders_{DELIVERY_DATE.replace('-', '')}.1"
        )
        if not orders_path.exists():
            pytest.skip(
                f"OMIE raw orders file missing: {orders_path}. The "
                f"truncated real-data invariant is skip-on-data-absent."
            )

        cfg = OMIEDataConfig(
            data_path=str(RAW_OMIE_DIR),
            contract_id=CANONICAL_CONTRACT_ID,
            delivery_date=DELIVERY_DATE,
        )
        orders_full = load_omie_orders(cfg)
        if int(orders_full.shape[0]) == 0:
            pytest.skip(
                f"Loader returned 0 orders for cid={CANONICAL_CONTRACT_ID} "
                f"on {DELIVERY_DATE}; canonical fixture unavailable."
            )

        # Truncate to the locked slice size.  The slice is a contiguous
        # prefix of the full session so the temporal ordering of orders
        # (and hence the iceberg-meta evolution) is preserved.
        n_take = min(TRUNCATED_REPLAY_SLICE, int(orders_full.shape[0]))
        orders = orders_full[:n_take]

        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()

        book_prod = ContractBook(_make_open_contract(CANONICAL_CONTRACT_ID), pc, mc)
        book_ref = ContractBook(_make_open_contract(CANONICAL_CONTRACT_ID), pc, mc)

        # Production (gated, post-P2-PERF-8) path.
        state_prod, buf_prod = _replay_orders(
            book_prod.init(), orders, pc, mc,
        )
        for leaf in jax.tree_util.tree_leaves((state_prod, buf_prod)):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        n_fills_prod, qty_prod = _aggregate_replay_buf(buf_prod)
        tup_prod = _byte_identity_tuple(state_prod, n_fills_prod, qty_prod)

        # Reference (unconditional-regen) path.
        state_ref, buf_ref = _ref_replay_orders(
            book_ref.init(), orders, pc, mc,
        )
        for leaf in jax.tree_util.tree_leaves((state_ref, buf_ref)):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        n_fills_ref, qty_ref = _aggregate_replay_buf(buf_ref)
        tup_ref = _byte_identity_tuple(state_ref, n_fills_ref, qty_ref)

        assert tup_prod == tup_ref, (
            f"P2-PERF-8 byte-identity invariant violated on truncated "
            f"real-data replay (cid={CANONICAL_CONTRACT_ID}, "
            f"delivery_date={DELIVERY_DATE}, n_orders={n_take}):\n"
            f"  production (gated)       ={tup_prod!r}\n"
            f"  reference (unconditional)={tup_ref!r}\n"
            f"  per-field equality:\n"
            f"    n_fills        : prod={tup_prod[0]} ref={tup_ref[0]} "
            f"(equal={tup_prod[0] == tup_ref[0]})\n"
            f"    total_qty_ticks: prod={tup_prod[1]} ref={tup_ref[1]} "
            f"(equal={tup_prod[1] == tup_ref[1]})\n"
            f"    hash(bids)     : equal={tup_prod[2] == tup_ref[2]}\n"
            f"    hash(asks)     : equal={tup_prod[3] == tup_ref[3]}\n"
            f"    hash(trades)   : equal={tup_prod[4] == tup_ref[4]}\n"
            f"    hash(iceberg)  : equal={tup_prod[5] == tup_ref[5]}\n"
            f"    n_active       : prod={tup_prod[6]} ref={tup_ref[6]} "
            f"(equal={tup_prod[6] == tup_ref[6]})\n"
            f"    ts_counter     : prod={tup_prod[7]} ref={tup_ref[7]} "
            f"(equal={tup_prod[7] == tup_ref[7]})\n"
            f"    book_full      : prod={tup_prod[8]} ref={tup_ref[8]} "
            f"(equal={tup_prod[8] == tup_ref[8]})"
        )


# ---------------------------------------------------------------------------
# Cascade-quiescence invariant (P2-PERF-8 enlarged scope — decision iii)
# ---------------------------------------------------------------------------
#
# Domain law: market-spec.md §4.2.1 Rule 4 (2026-05-26 P2-PERF-8 ruling).
# A regenerated iceberg slice that is immediately re-matched by resting
# opposite-side liquidity MUST resolve to a fixed point WITHIN the
# originating ``submit_order`` matching process; the order book is never
# left in a crossed state (no active iceberg slot left in a
# ``peak_consumed=True`` cascade-leftover state) between aggressor
# arrivals.  This is the precondition that makes the no-fill gate
# provably safe (sprint-status.md §"P2-PERF-8 — Cascade Adjudication
# (2026-05-26)" §1/§2).
#
# Quiescence check (per the card Test-first contract (b)): for every
# active iceberg slot ``i in [0, n_active)``, the slot's order id must be
# present in the resting side array with a positive visible peak qty —
# i.e. ``get_order(lob, lookup_side, order_id)[2] > 0`` where
# ``lookup_side = 1 - side``.  When the OID is absent from the side
# (its peak was consumed and not regenerated within the call),
# ``get_order`` returns the ``NEGATIVE_RETURN_ID = -99`` sentinel row, so
# column 2 (visible qty) reads ``-99 <= 0`` → the slot is in the
# spec-violating ``peak_consumed=True`` cascade-leftover state.
#
# This invariant is RED on the current branch HEAD: the production
# ``_ice_path`` regen ([contract_book.py:1059]) drains the cascade across
# ``submit_order`` calls, leaving a slot in ``peak_consumed=True`` after
# the originating call (the order-20→21 cascade on the cid 81 truncated
# slice — sprint-status.md §"P2-PERF-8 — Implementation (2026-05-26)"
# NON/IOC empirical-fail trace).  It turns GREEN only after the developer
# lands the bounded fixed-point cascade-chase in ``regenerate_slices``
# (enlarged Deliverable 1).  This is the genuine red→green TDD pin.


def _quiescence_violations(state: OMIEBookState) -> list[dict]:
    """Return the list of active iceberg slots that violate Rule 4.

    For every active slot ``i in [0, n_active)`` we look up the slot's
    resting peak via ``OrderBook.get_order(lob, lookup_side, order_id)``
    and flag the slot if ``order_row[2] <= 0`` (the ``-99`` sentinel from
    an absent OID, or an otherwise non-positive visible peak qty) — i.e.
    the slot is in a ``peak_consumed=True`` cascade-leftover state.

    Returns a list of per-slot diagnostic dicts (empty if quiescent).
    The lookup mirrors ``IcebergManager.regenerate_slices``'
    ``peak_consumed`` derivation (iceberg_manager.py:367-369) exactly:
    ``lookup_side = 1 - side`` and ``order_row = get_order(lob, lookup_side,
    order_id)``; ``peak_consumed = order_row[2] <= 0``.
    """
    # A fresh OrderBook for the read-only ``get_order`` query; the cfg
    # only affects mutating dispatch, not the id-lookup the query does.
    ob = OrderBook(cfg=_make_jaxob_cfg(OMIEMarketConfig()))

    meta = np.asarray(state.iceberg_meta)
    n_active = int(state.n_active)

    violations: list[dict] = []
    for i in range(n_active):
        order_id = int(meta[i, COL_ORDER_ID])
        side = int(meta[i, COL_SIDE])
        lookup_side = 1 - side
        order_row = np.asarray(
            ob.get_order(state.lob, jnp.int32(lookup_side), jnp.int32(order_id))
        )
        visible_qty = int(order_row[2])
        if visible_qty <= 0:
            violations.append({
                "slot": i,
                "order_id": order_id,
                "side": side,
                "lookup_side": lookup_side,
                "visible_qty": visible_qty,
                "remaining_hidden": int(meta[i, COL_REMAINING_HIDDEN]),
                "peak_qty": int(meta[i, COL_PEAK_QTY]),
                "slice_count": int(meta[i, COL_SLICE_COUNT]),
            })
    return violations


def _cascade_synthetic_orders(pc: OMIEProductConfig, oid_base: int = 0) -> jnp.ndarray:
    """Construct a session that deterministically triggers the
    regen-immediate-rematch cascade on the canonical sell-side ICE path.

    Mirrors the cid 81 order-20→21 cascade (sprint-status.md §"P2-PERF-8
    — Implementation (2026-05-26)" NON/IOC empirical-fail trace) in
    miniature:

      Rows 0-2: three NON BUY @ 60 EUR/MWh, qty 0.2 each — build deep
                resting bid liquidity (0.6 MW total) at the price the
                ICE iceberg will rest/cross.
      Row 3:    ICE SELL @ 60 EUR/MWh, total qty 1.0, peak 0.2.
                ``_ice_path`` dispatches the peak (0.2 MW) as a LIMIT
                SELL @ 60 → it fully crosses one resting bid (0.2 MW) →
                ``order_id`` consumed → the post-dispatch ``regenerate_
                slices`` sees ``peak_consumed=True`` → dispatches slice 2
                (0.2 MW @ 60, ``process_order_array`` = match-then-rest)
                → slice 2 ALSO crosses the next resting bid (still 0.4 MW
                of bid depth) → consumed AGAIN within the same regen
                call.  The single-visit ``while_loop`` advances past the
                slot, leaving it ``peak_consumed=True`` — the cascade
                leftover.  Under cascade quiescence the chase must drain
                until a slice rests visibly (or the iceberg exhausts),
                so no slot is left ``peak_consumed=True`` after row 3's
                ``submit_order`` returns.
      Row 4:    NON SELL @ 70 EUR/MWh, qty 0.1 — a no-fill order (rests
                on the ask; nothing crosses at 70).  This is the order
                whose ``no_fill_path`` the gate protects; the quiescence
                invariant must hold AFTER row 3 already (the gate would
                otherwise diverge here, exactly as cid 81 order 21 does).

    The session is intentionally small (5 orders) to fit the fast-suite
    < 1 s budget.  The deep resting bid liquidity (3 × 0.2 MW) guarantees
    the regenerated slice is itself re-consumed, forcing the multi-hop
    cascade Rule 4 names — not merely the single-hop Rule 3 case.
    """
    p_cross = int(round((60.00 - pc.price_min) / pc.price_tick))
    p_high = int(round((70.00 - pc.price_min) / pc.price_tick))
    q_peak = int(round(0.2 / pc.qty_tick))
    q_small = int(round(0.1 / pc.qty_tick))
    q_full = int(round(1.0 / pc.qty_tick))

    # FOLLOWUP-2 fixture-OID realism sweep (2026-07-09): ``oid_base``
    # offsets every OID.  This is the canonical quiescence-surface
    # cascade whose original tiny OIDs (1..5) sit numerically WITHIN the
    # qty tick range (q_peak=2, q_full=10) — exactly the small-OID class
    # the QUIESCENCE-1 audit found masked a wrong-column gate.  Driving
    # it at base 1000 proves the per-submit quiescence invariant is
    # OID-magnitude-independent.
    b = oid_base
    rows = [
        # Rows 0-2: deep resting bid liquidity @ 60 (0.2 MW each).
        [37, int(Side.BUY), p_cross, q_peak, b + 1, b + 1, 100,
         int(ExecRestriction.NON), int(Validity.GFS), 0,
         int(Status.ACTIVE), 0, 0],
        [37, int(Side.BUY), p_cross, q_peak, b + 2, b + 2, 101,
         int(ExecRestriction.NON), int(Validity.GFS), 0,
         int(Status.ACTIVE), 0, 0],
        [37, int(Side.BUY), p_cross, q_peak, b + 3, b + 3, 102,
         int(ExecRestriction.NON), int(Validity.GFS), 0,
         int(Status.ACTIVE), 0, 0],
        # Row 3: ICE SELL @ 60, total 1.0, peak 0.2 — peak crosses, regen
        # slice re-crosses → cascade leftover under current production.
        [37, int(Side.SELL), p_cross, q_full, b + 4, b + 4, 103,
         int(ExecRestriction.ICE), int(Validity.GFS), 0,
         int(Status.ACTIVE), q_peak, 0],
        # Row 4: NON SELL @ 70, qty 0.1 — no-fill order whose no_fill_path
        # the gate protects; quiescence must already hold after row 3.
        [37, int(Side.SELL), p_high, q_small, b + 5, b + 5, 104,
         int(ExecRestriction.NON), int(Validity.GFS), 0,
         int(Status.ACTIVE), 0, 0],
    ]
    return jnp.array(rows, dtype=jnp.int32)


def _replay_with_quiescence_check(
    init_state: OMIEBookState,
    orders: jnp.ndarray,
    pc: OMIEProductConfig,
    mc: OMIEMarketConfig,
) -> tuple[int, list[dict]]:
    """Replay ``orders`` order-by-order via the production ``_submit_order``
    lift, checking the Rule 4 cascade-quiescence invariant after EVERY
    ``submit_order`` call (not just no-fill iterations — per the card's
    Test-first contract (b)).

    Returns ``(first_bad_order_index, violations)`` — the index of the
    first order after whose ``submit_order`` an active slot was left in
    ``peak_consumed=True`` state (and the violating-slot diagnostics), or
    ``(-1, [])`` if the whole replay is quiescent.

    The host-side per-order loop is required (rather than the device-side
    ``lax.scan`` in ``_replay_orders``) because the invariant must be
    inspected at every intermediate state — ``lax.scan`` exposes only the
    final carry and the per-step output slab, not the per-step
    ``OMIEBookState``.
    """
    state = init_state
    n_orders = int(orders.shape[0])
    for k in range(n_orders):
        row = orders[k]
        order = _row_to_order_traced(row, pc)
        state, _result = _submit_order(state, order, pc, mc)
        # Block so the read-only ``get_order`` queries observe the
        # settled device state.
        for leaf in jax.tree_util.tree_leaves(state):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        violations = _quiescence_violations(state)
        if violations:
            return k, violations
    return -1, []


class TestP2Perf8CascadeQuiescence:
    """Rule 4 cascade-quiescence invariant (P2-PERF-8 enlarged scope).

    After every ``submit_order`` call in the replay, no active iceberg
    slot may be in a ``peak_consumed=True`` cascade-leftover state
    (``get_order(lob, lookup_side, order_id)[2] > 0`` for all active
    slots).  This pins the divergence-at-order-21 regression directly
    and is the precondition that makes the no-fill gate provably safe.

    RED on current branch HEAD (production drains the cascade across
    calls); GREEN only after the developer lands the bounded fixed-point
    cascade-chase in ``regenerate_slices``.
    """

    @pytest.mark.parametrize("oid_base", SWEEP_OID_BASES)
    def test_synthetic_ice_cascade_quiescent_after_every_submit(self, oid_base):
        """Synthetic 5-order session reproducing the cid 81 order-20→21
        cascade in miniature: after every ``submit_order`` the book must
        be cascade-quiescent (no active slot in ``peak_consumed=True``).

        See ``_cascade_synthetic_orders`` for the deterministic cascade
        construction (deep resting bids re-consume the regenerated ICE
        slice within the originating call).

        FOLLOWUP-2 sweep: parametrized over ``SWEEP_OID_BASES``.  The
        original tiny OIDs (1..5) sit inside the qty tick range — this
        is precisely the small-OID class the QUIESCENCE-1 audit found
        masked a wrong-column gate; the realistic-OID variant proves the
        per-submit quiescence invariant is OID-magnitude-independent.
        """
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        contract = Contract(
            contract_id=37,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.OPEN,
        )
        book = ContractBook(contract, pc, mc)
        orders = _cascade_synthetic_orders(pc, oid_base)

        bad_idx, violations = _replay_with_quiescence_check(
            book.init(), orders, pc, mc,
        )

        assert bad_idx == -1, (
            "P2-PERF-8 Rule 4 cascade-quiescence invariant violated on the "
            "synthetic ICE-cascade session: after submit_order #"
            f"{bad_idx} (order row index {bad_idx}) an active iceberg "
            "slot is in peak_consumed=True (cascade-leftover) state — its "
            "OID is absent from the resting side (get_order col-2 visible "
            "qty <= 0, the NEGATIVE_RETURN_ID=-99 sentinel), meaning a "
            "regenerated slice was left crossed and undrained across the "
            "submit cycle — an engineering-invariant violation (OQ-23; "
            "not a Rule 4 breach — the resting book is uncrossed).\n"
            f"  violating slots: {violations!r}"
        )

    def test_truncated_real_data_cascade_quiescent_after_every_submit(self):
        """Truncated real-data slice (cid 81 / 2026-01-12): after every
        ``submit_order`` the book must be cascade-quiescent.

        This is the fixture that originally exposed the cascade at order
        20 (ICE SELL @ 999900, registers iceberg order_id=21) → order 21
        (NON SELL @ 1013729), where the gated path diverges from the
        unconditional reference (sprint-status.md §"P2-PERF-8 —
        Implementation (2026-05-26)" NON/IOC empirical-fail trace).  The
        invariant pins that regression directly.

        Skipped if the raw OMIE orders file is absent — the fast-tier
        invariant is skip-on-data-absent so the suite remains green on
        data-less CI.
        """
        orders_path = (
            RAW_OMIE_DIR / f"orders_{DELIVERY_DATE.replace('-', '')}.1"
        )
        if not orders_path.exists():
            pytest.skip(
                f"OMIE raw orders file missing: {orders_path}. The "
                f"cascade-quiescence real-data invariant is "
                f"skip-on-data-absent."
            )

        cfg = OMIEDataConfig(
            data_path=str(RAW_OMIE_DIR),
            contract_id=CANONICAL_CONTRACT_ID,
            delivery_date=DELIVERY_DATE,
        )
        orders_full = load_omie_orders(cfg)
        if int(orders_full.shape[0]) == 0:
            pytest.skip(
                f"Loader returned 0 orders for cid={CANONICAL_CONTRACT_ID} "
                f"on {DELIVERY_DATE}; canonical fixture unavailable."
            )

        n_take = min(TRUNCATED_REPLAY_SLICE, int(orders_full.shape[0]))
        orders = orders_full[:n_take]

        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        book = ContractBook(_make_open_contract(CANONICAL_CONTRACT_ID), pc, mc)

        bad_idx, violations = _replay_with_quiescence_check(
            book.init(), orders, pc, mc,
        )

        assert bad_idx == -1, (
            "P2-PERF-8 Rule 4 cascade-quiescence invariant violated on the "
            f"truncated cid {CANONICAL_CONTRACT_ID} / {DELIVERY_DATE} "
            f"replay ({n_take} orders): after submit_order #{bad_idx} an "
            "active iceberg slot is in peak_consumed=True (cascade-leftover) "
            "state — its OID is absent from the resting side (get_order "
            "col-2 visible qty <= 0, the NEGATIVE_RETURN_ID=-99 sentinel), "
            "meaning a regenerated slice was left crossed and undrained "
            "across the submit cycle — an engineering-invariant violation "
            "(OQ-23; not a Rule 4 breach — the resting book is uncrossed). "
            f"This is the order-20→21 regression.\n"
            f"  violating slots: {violations!r}"
        )
