"""Iceberg order side-channel state and slice regeneration (JIT-safe).

Owns the iceberg side-channel logic that was previously embedded in
ContractBook. All methods are stateless — they accept ``iceberg_meta``
(a fixed-size ``jnp.ndarray`` of shape ``(MAX_ICEBERGS, 8)``) and
``ts_counter`` (a ``jnp.int32`` scalar) as parameters and return updated
versions. This makes the module compatible with the functional
``OMIEBookState`` pattern under ``jax.jit``.

State schema
------------
``iceberg_meta`` is an ``(MAX_ICEBERGS, 8)`` int32 array. Each row is
either an empty slot (column 0 == ``SENTINEL_ORDER_ID``) or an active
iceberg. Columns:

    [order_id, remaining_hidden, peak_qty,
     base_price_int, side, agent_id, ppd, slice_count]

where ``base_price_int`` is the **immutable** offset-encoded base price
(``encoding.encode_price``) captured at register time. The drifted slice
price for slice N is recomputed at every read as
``base_price_int + ppd * slice_count`` in encoded space (``ppd`` is
signed int-ticks, see ``encoding.encode_ppd``). ``remaining_hidden`` and
``peak_qty`` are quantity ticks (``encoding.encode_qty``).

Bounded control flow
--------------------
* The outer iteration over ``MAX_ICEBERGS`` slots in
  ``regenerate_slices`` is a ``jax.lax.scan`` ordered by slot index.
  Empty slots (``order_id == SENTINEL_ORDER_ID``) are no-ops.
* The inner ICEwPI slice-projection loop in ``matchable_hidden_volume``
  is a ``jax.lax.while_loop`` bounded by ``MAX_SLICES_PER_REGEN``.

Truncation backstop
-------------------
Carry flag (active in every build). ``truncation_flag(meta)`` returns a
``jnp.bool_`` that becomes True if any active slot would require more
than ``MAX_SLICES_PER_REGEN`` total slices to drain
(``ceil(remaining_hidden / peak_qty) + slice_count > MAX_SLICES_PER_REGEN``).
The detection is recomputed on demand from the meta array so it does
not occupy a state column.

Encoding boundary
-----------------
``register(meta, order)`` performs the OQ-16 round-to-tick encoding of
``order.price`` / ``order.peak_qty`` / ``order.ppd`` / ``order.quantity``
from float to int32 ticks. ``OMIEOrder`` field annotations remain float
(OQ-17 stays open).

Constants: ``omiecid_simulator/omiecid/jax_constants.py`` (MAX_ICEBERGS,
MAX_SLICES_PER_REGEN, SENTINEL_ORDER_ID).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat
from omiecid_simulator.jaxob.jorderbook import OrderBook, LobState
from omiecid_simulator.omiecid.jax_constants import (
    MAX_ICEBERGS,
    MAX_SLICES_PER_REGEN,
    SENTINEL_ORDER_ID,
)
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import OMIEOrder, Side


# ---------------------------------------------------------------------------
# Schema column indices — fixed; every reader depends on this order.
# ---------------------------------------------------------------------------

COL_ORDER_ID = 0
COL_REMAINING_HIDDEN = 1
COL_PEAK_QTY = 2
COL_BASE_PRICE_INT = 3
COL_SIDE = 4
COL_AGENT_ID = 5
COL_PPD = 6
COL_SLICE_COUNT = 7
N_COLS = 8

# jaxob resting-row columns (``OrderSideFeat``: [P, Q, OID, TID, SEC, NSEC]),
# read by the exhaustion-order ranked-stamp compares (see regenerate_slices).
_P = OrderSideFeat.P.value
_Q = OrderSideFeat.Q.value
_OID = OrderSideFeat.OID.value
_SEC = OrderSideFeat.SEC.value
_NSEC = OrderSideFeat.NSEC.value


def _sentinel_row() -> jnp.ndarray:
    """Return a single freed-slot row: zeros with ``COL_ORDER_ID = SENTINEL_ORDER_ID``.

    Shared by ``cleanup`` (free a slot by id) and the cleanup branch of
    ``regenerate_slices`` (free a slot whose hidden volume has been fully
    drained).
    """
    zero_row = jnp.zeros((N_COLS,), dtype=jnp.int32)
    return zero_row.at[COL_ORDER_ID].set(jnp.int32(SENTINEL_ORDER_ID))


class IcebergManager:
    """JIT-safe iceberg side-channel logic and slice regeneration.

    All methods accept an array-shaped ``iceberg_meta`` (shape
    ``(MAX_ICEBERGS, 8)``, dtype int32) and a scalar ``ts_counter``
    (``jnp.int32``) and return updated versions. No mutable state is
    stored on the instance.

    Args:
        ob: The jaxob OrderBook instance (shared with ContractBook).
            Needed for ``get_order()`` lookups and
            ``process_order_array()`` during slice regeneration.
        product_config: OMIE product config for encoding conversions.
            Used at construction time to derive trace-time-constant
            scalars (``price_offset_int``, ``price_scale_inv``, etc.)
            so jitted code paths receive only static numbers.
    """

    def __init__(self, ob: OrderBook, product_config: OMIEProductConfig):
        self._ob = ob
        self._product_config = product_config
        # Trace-time-constant scalars used inside the jitted methods. We
        # materialise them as Python floats / ints so they enter the jaxpr
        # as constants (not traced arrays).
        cfg = product_config
        self._price_offset = float(-cfg.price_min / cfg.price_tick)  # e.g. 999900.0
        self._price_scale_inv = float(1.0 / cfg.price_tick)          # e.g. 100.0
        self._qty_scale_inv = float(1.0 / cfg.qty_tick)              # e.g. 10.0

    # ------------------------------------------------------------------
    # Empty-state factory (used by ContractBook.init)
    # ------------------------------------------------------------------

    @staticmethod
    def empty_meta(max_icebergs: int = MAX_ICEBERGS) -> jnp.ndarray:
        """Return an empty ``iceberg_meta`` array (all slots sentinel).

        Shape ``(max_icebergs, N_COLS)``, dtype int32, all rows zero except
        column 0 (``order_id``) set to ``SENTINEL_ORDER_ID`` to mark every
        slot available.

        ``max_icebergs`` is defaulted to
        ``MAX_ICEBERGS`` (the hard structural ceiling) so the zero-arg call
        ``empty_meta()`` stays source-compatible for the direct-construction
        callers — its shape remains ``(MAX_ICEBERGS, N_COLS)``.
        ``ContractBook.init()`` passes ``market_config.max_icebergs`` to size
        the meta from the runtime cap.
        """
        meta = jnp.zeros((max_icebergs, N_COLS), dtype=jnp.int32)
        meta = meta.at[:, COL_ORDER_ID].set(jnp.int32(SENTINEL_ORDER_ID))
        return meta

    # ------------------------------------------------------------------
    # Internal round-to-tick encoders (JIT-safe; produce int32 tick scalars)
    # ------------------------------------------------------------------

    def _encode_price_int(self, price_eur) -> jnp.ndarray:
        """Offset encoding: ``round((price - price_min) / price_tick)``.

        Equivalent to ``encoding.encode_price`` but stays inside JAX so
        the result is a traced int32 scalar when called under ``jit``.
        """
        return jnp.round(
            jnp.asarray(price_eur, jnp.float32) * self._price_scale_inv
            + self._price_offset
        ).astype(jnp.int32)

    def _encode_qty_int(self, qty_mw) -> jnp.ndarray:
        return jnp.round(
            jnp.asarray(qty_mw, jnp.float32) * self._qty_scale_inv
        ).astype(jnp.int32)

    def _encode_ppd_int(self, ppd_eur) -> jnp.ndarray:
        # Signed int-tick encoding; matches ``encoding.encode_ppd``.
        return jnp.round(
            jnp.asarray(ppd_eur, jnp.float32) * self._price_scale_inv
        ).astype(jnp.int32)

    # ------------------------------------------------------------------
    # Registration  (packed layout: append-at-tail)
    # ------------------------------------------------------------------

    def register(
        self,
        iceberg_meta: jnp.ndarray,
        n_active,
        order: OMIEOrder,
    ) -> tuple:
        """Register a new iceberg by appending it at slot ``n_active``.

        Encodes ``order.price`` / ``order.peak_qty`` / ``order.ppd`` /
        ``order.quantity`` from float to int32 ticks (round-to-tick,
        encode-on-entry) and writes them at index ``n_active``
        of the packed-and-contiguous iceberg meta array. Then increments
        ``n_active``. No slot search is required — the layout invariant
        guarantees ``meta[n_active]`` is a sentinel row.

        Saturation / slot-overflow contract:
            If ``n_active >= iceberg_meta.shape[0]`` (the configured
            ``max_icebergs``; saturated), ``overflow`` is True,
            ``iceberg_meta`` is returned unchanged, and ``n_active`` is NOT
            incremented. Callers must propagate this overflow flag through
            ``ContractBook._build_state`` onto the sticky
            ``OMIEBookState.truncated`` field.

        Per-op JAX cost:
            * 1 ``.at[n_active].set(new_row)`` write (gated by overflow)
            * 1 int32 add (``n_active += 1``)
            * Overflow predicate: ``n_active >= iceberg_meta.shape[0]``
              (single int32 compare against the configured ``max_icebergs``,
              read as a trace-static shape; replaces the prior
              ``~jnp.any(sentinels)`` reduction).

        Args:
            iceberg_meta: Current ``(MAX_ICEBERGS, 8)`` int32 array.
            n_active: 0-d int32 — count of live iceberg slots at the
                head of the array. Live rows occupy ``[0, n_active)``;
                sentinels occupy ``[n_active, MAX_ICEBERGS)``.
            order: The iceberg order being registered. Float fields are
                encoded to int32 ticks here; integer fields are cast.

        Returns:
            ``(updated_meta, new_n_active, overflow)`` — ``updated_meta``
            has the same shape/dtype as the input; ``new_n_active`` is
            the post-write live count (``n_active + 1`` on success,
            ``n_active`` on overflow); ``overflow`` is a 0-d
            ``jnp.bool_`` that is True iff the call was made on a
            saturated meta (no slot was available).
        """
        n_active_i = jnp.asarray(n_active, jnp.int32)

        # Encode the order fields to int32 ticks (round-to-tick).
        peak_int = self._encode_qty_int(order.peak_qty)
        qty_int = self._encode_qty_int(order.quantity)
        base_price_int = self._encode_price_int(order.price)
        ppd_int = self._encode_ppd_int(order.ppd)

        remaining_hidden = qty_int - peak_int
        # Defensive clamp: remaining_hidden can never go negative under
        # validated input (validator enforces peak <= quantity), but
        # JAX rounding edge cases at the float-to-int boundary should
        # not produce negatives either.
        remaining_hidden = jnp.maximum(remaining_hidden, jnp.int32(0))

        # Packed layout: overflow is a single int32 compare.
        # The saturation ceiling is the meta's OWN first
        # dimension (the configured ``max_icebergs``), read as a trace-static
        # shape from the ``iceberg_meta`` argument — NOT the ``MAX_ICEBERGS``
        # hard-ceiling constant. This makes the loud-cap overflow follow the
        # runtime cap (a ``max_icebergs=8`` book overflows on the 9th
        # registration) while staying a single int32 compare.
        overflow = n_active_i >= jnp.int32(iceberg_meta.shape[0])
        not_overflow = jnp.logical_not(overflow)

        # Build the new row in column order.
        new_row = jnp.stack([
            jnp.asarray(order.order_id, jnp.int32),
            remaining_hidden,
            peak_int,
            base_price_int,
            jnp.asarray(order.side, jnp.int32),
            jnp.asarray(order.agent_id, jnp.int32),
            ppd_int,
            jnp.int32(0),  # slice_count starts at 0
        ])
        # The write target index. On overflow, clamp to a safe in-bounds
        # index (0) — the gated write below will discard the result so
        # the row written is irrelevant; the clamp only prevents
        # out-of-bounds indexing under saturated states.
        slot_idx = jnp.where(not_overflow, n_active_i, jnp.int32(0))
        written_meta = iceberg_meta.at[slot_idx].set(new_row)
        # Gate the write on overflow — if saturated, return iceberg_meta
        # unchanged (no silent overwrite at slot 0).
        updated_meta = jnp.where(not_overflow, written_meta, iceberg_meta)
        new_n_active = n_active_i + jnp.where(
            not_overflow, jnp.int32(1), jnp.int32(0)
        )
        return (updated_meta, new_n_active, overflow)

    # ------------------------------------------------------------------
    # Timestamp management
    # ------------------------------------------------------------------

    def update_timestamp(self, ts_counter, incoming_ts):
        """Advance the monotonic counter past ``incoming_ts`` if needed.

        JIT-safe: works on either Python ints or JAX scalars. Returns
        ``max(ts_counter, incoming_ts + 1)``.
        """
        ts = jnp.asarray(ts_counter, jnp.int32)
        ts_in = jnp.asarray(incoming_ts, jnp.int32)
        return jnp.maximum(ts, ts_in + jnp.int32(1))

    def next_timestamp(self, ts_counter):
        """Return ``(ts, ts_counter + 1)`` as a JAX-friendly pair.

        For backward compatibility with the older dict-based API, callers may
        receive Python ints when they pass Python ints; the contract
        is that ``ts_counter`` is a 0-d int32.
        """
        ts = jnp.asarray(ts_counter, jnp.int32)
        return (ts, ts + jnp.int32(1))

    # ------------------------------------------------------------------
    # Slice regeneration — outer ``lax.scan`` over MAX_ICEBERGS slots.
    # ------------------------------------------------------------------

    def regenerate_slices(
        self,
        lob: LobState,
        iceberg_meta: jnp.ndarray,
        n_active,
        ts_counter,
        max_iterations: int = None,
        prior_lob: LobState = None,
        aggressor_side=None,
    ) -> tuple:
        """For every active slot whose visible peak has been consumed,
        regenerate one new slice and update the meta row, re-running the
        slot scan until the regen-immediate-rematch cascade is fully
        drained (a bounded fixed point).

        The inner slot scan is bounded by
        ``n_active`` via ``lax.while_loop`` (equivalent to ``lax.scan``
        over a ``lax.dynamic_slice``-d prefix; chosen because the
        write-back to ``iceberg_meta`` is in-place via
        ``at[slot_idx].set(new_row)`` rather than an output-stack
        assembly, which fits the while_loop carry naturally). The cost
        scales with ``n_active``, NOT ``MAX_ICEBERGS`` — this is the
        mechanism that makes the cost independent of the cap.

        Packed-layout invariant: rows at indices ``[n_active,
        MAX_ICEBERGS)`` are guaranteed sentinels (live region is a
        contiguous prefix). The bound ``i < n_active`` is therefore
        sufficient — no extra "is this row active?" mask is needed.

        Mutually-exclusive lifecycle per slot:
          (a) ``need_regen``: visible peak gone AND remaining > 0.
              Produce a new slice, decrement ``remaining_hidden``, bump
              ``slice_count``. The slot stays live.
          (b) ``need_cleanup``: visible peak gone AND remaining == 0.
              The iceberg is fully drained; perform swap-with-last to
              the tail and decrement ``n_active``. Slot index ``i`` then
              carries the previously-tail row, which the next iteration
              (``i + 1``) does NOT visit because the outer bound shrunk.

        Bounded fixed-point cascade-chase (market-spec §4.2.1 Rule 4):
        a regenerated slice dispatched via ``process_order_array``
        (a match-then-rest path, line ~397) may itself be immediately
        re-consumed by still-resting opposite-side liquidity within the
        same call, leaving the slot in a ``peak_consumed=True`` cascade-
        leftover state that the single-visit inner scan does not chase
        (the inner loop advances past the slot at line ~454). Rule 4
        mandates the cascade resolve to a fixed point WITHIN the
        originating ``submit_order`` call. We therefore wrap the inner
        slot scan in an OUTER ``lax.while_loop`` that re-runs the scan
        until no active slot reports ``peak_consumed=True`` (quiescence),
        capped by ``max_iterations`` (defaults to
        ``OMIEMarketConfig.max_match_iterations`` = 200). On the
        pathological non-quiescent-at-cap exit the slot is left with a
        ``slice_count`` far exceeding ``MAX_SLICES_PER_REGEN`` (each
        re-run that dispatches bumps ``slice_count``), so the existing
        ``truncation_flag`` backstop catches it and ORs into
        ``OMIEBookState.truncated`` at ``_build_state_jit`` — the SAME
        truncation flag, no new flag. The post-condition is: after
        ``regenerate_slices`` returns, no active slot is in
        ``peak_consumed=True``.

        Cross-slot quiescence (OQ-23): the
        post-condition above holds, including the
        CROSS-slot case. Without the re-arm below, the ``dirty`` flag reflected only
        SELF-OID re-consumption (``slice_re_consumed``): a regen slice
        for slot *j* that PARTIALLY rested while consuming an
        already-visited, lower-index, opposite-side slot *k*'s visible
        peak did not set ``dirty``, so the outer loop could exit with
        *k* left ``peak_consumed=True`` (the OQ-23 latent quiescence
        gap — an engineering-invariant violation, NOT a Rule 4 breach;
        the resting book stays uncrossed). The fix: at the end of every
        pass in which a dispatched regen slice actually TRADED (the only
        way a pass can remove another slot's resting peak), re-derive
        ``any_active_peak_consumed`` across ``meta[:n_active]`` (the
        same ``get_order(...)[2] <= 0`` derivation the test-suite
        quiescence probe uses) and OR it into the pass's ``dirty`` verdict — see
        ``_single_pass``. Non-quiescent-at-cap exits remain possible
        only via the ``max_iterations`` cap-trip (truncation backstop
        above). This guarantee is what ``contract_book.py``'s
        ``no_fill_path`` regen-skip relies on between
        aggressors. Refs: docs/omiecid/open-questions.md §OQ-23.

        Args:
            lob: Current jaxob LobState.
            iceberg_meta: ``(MAX_ICEBERGS, 8)`` int32 array.
            n_active: 0-d int32 — count of live iceberg slots.
            ts_counter: scalar int32 timestamp counter.
            max_iterations: Python int cap on the outer fixed-point
                re-runs. ``None`` (the default used by every call site)
                falls back to ``OMIEMarketConfig.max_match_iterations``
                (200), so the call sites' 4-arg signature is unchanged.
            prior_lob: Exhaustion-order stamping (market-spec v2.14
                §4.2.1 Rule 4 *Slice activation timing*, eager
                activation). The book **before** the matching pass that
                preceded this call. ``None`` (default) keeps the unranked
                traced program exactly (static Python branch): every
                dispatch is stamped in slot-visit order. When given, the
                slots whose peak rested in ``prior_lob`` and is gone from
                ``lob`` (with hidden remainder > 0, on the side opposite
                ``aggressor_side``) are *ranked* by their ``prior_lob``
                priority key ``(±P, SEC, NSEC)`` (price negated for BUY),
                which is the order the pass exhausted them. In the primed
                pass only, a ranked slot's slice is stamped
                ``ts_p + rank`` and every other dispatch
                ``ts_p + k0 + n_dyn`` (visit order), so the stamps are a
                permutation of the unranked stamps. Dispatch order, meta
                writes and the ``ts_counter`` increments are unchanged.
                Cascade passes keep visit-order stamping.
            aggressor_side: OMIE side (0 BUY / 1 SELL) of the aggressor
                of that pass; slots on this side are never ranked.
                ``None`` ranks both sides (BUY ranks first, then SELL).

        Returns:
            ``(lob, iceberg_meta, n_active, ts_counter)`` after the
            cascade has been drained to a fixed point (or the cap hit).
        """
        if max_iterations is None:
            from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig
            max_iterations = OMIEMarketConfig().max_match_iterations
        max_iters = jnp.int32(max_iterations)
        ts_counter = jnp.asarray(ts_counter, jnp.int32)
        n_active_i = jnp.asarray(n_active, jnp.int32)
        sentinel_row = _sentinel_row()

        if prior_lob is not None:
            prior_bids = jnp.asarray(prior_lob.bids, jnp.int32)
            prior_asks = jnp.asarray(prior_lob.asks, jnp.int32)
            if aggressor_side is not None:
                aggressor_side_i = jnp.asarray(aggressor_side, jnp.int32)

        def cond_fn(carry):
            _n_active, i = carry[2], carry[4]
            # NOTE: the bound uses the live ``_n_active`` carry, which
            # can shrink mid-loop when a slot is cleaned up via
            # swap-with-last. The slot index ``i`` does NOT advance when
            # cleanup happens (we re-process the freshly-swapped tail
            # row at the same ``i``).
            return i < _n_active

        def body_fn(carry):
            return _body(carry, None)

        def _make_body_fn(rank_ctx):
            """Build the slot-scan body. ``rank_ctx is None`` is the unranked
            body (visit-order stamps, 7-element carry); otherwise it is
            ``(ts_p, k0, k0_buy, marks_b, marks_a)`` from
            ``_rank_context`` and the carry gains an int32 ``n_dyn``."""
            def body_fn(carry):
                return _body(carry, rank_ctx)
            return body_fn

        def _ranked_stamp(rank_ctx, side, order_id, n_dyn):
            """Exhaustion-order stamping: the stamp of slot
            ``order_id``'s dispatch in the primed ranked pass, and whether
            the slot is ranked. Plain vectorised compares on the
            ``prior_lob`` side arrays (no ``get_order`` / ``lax.cond``);
            every reduction is int32 (under the outer ``enable_x64``)."""
            ts_p, k0, k0_buy, marks_b, marks_a = rank_ctx
            is_buy = side == jnp.int32(int(Side.BUY))
            side_arr = jnp.where(is_buy, prior_bids, prior_asks)
            marks = jnp.where(is_buy, marks_b, marks_a)
            hit = (side_arr[:, _OID] == order_id) & (marks >= jnp.int32(0))
            is_ranked = jnp.any(hit)
            idx = jnp.argmax(hit)
            row = side_arr[idx]
            my_mark = marks[idx]
            kp_all = jnp.where(is_buy, -side_arr[:, _P], side_arr[:, _P])
            kp = jnp.where(is_buy, -row[_P], row[_P])
            ks, kn = row[_SEC], row[_NSEC]
            better = (
                (kp_all < kp)
                | ((kp_all == kp) & (
                    (side_arr[:, _SEC] < ks)
                    | ((side_arr[:, _SEC] == ks) & (
                        (side_arr[:, _NSEC] < kn)
                        | ((side_arr[:, _NSEC] == kn) & (marks < my_mark))
                    ))
                ))
            )
            rank = jnp.sum(
                (marks >= jnp.int32(0)) & better, dtype=jnp.int32,
            ) + jnp.where(is_buy, jnp.int32(0), k0_buy)
            stamp = jnp.where(
                is_ranked, ts_p + rank, ts_p + k0 + n_dyn,
            ).astype(jnp.int32)
            return stamp, is_ranked

        def _body(carry, rank_ctx):
            if rank_ctx is None:
                lob_c, meta_c, n_active_c, ts_c, i, dirty, dispatched = carry
            else:
                (lob_c, meta_c, n_active_c, ts_c, i, dirty, dispatched,
                 n_dyn) = carry
            # Read slot[i] via dynamic_index_in_dim.
            slot = jax.lax.dynamic_index_in_dim(meta_c, i, axis=0, keepdims=False)
            order_id = slot[COL_ORDER_ID]
            side = slot[COL_SIDE]
            remaining = slot[COL_REMAINING_HIDDEN]
            peak = slot[COL_PEAK_QTY]
            base_price_int = slot[COL_BASE_PRICE_INT]
            agent_id = slot[COL_AGENT_ID]
            ppd = slot[COL_PPD]
            slice_count = slot[COL_SLICE_COUNT]

            # Slots in [0, n_active) are guaranteed live under the
            # packed-layout invariant — no sentinel filter needed.
            lookup_side = jnp.int32(1) - side
            order_row = self._ob.get_order(lob_c, lookup_side, order_id)
            peak_consumed = order_row[2] <= jnp.int32(0)

            need_regen = peak_consumed & (remaining > jnp.int32(0))
            need_cleanup = peak_consumed & (remaining <= jnp.int32(0))

            slice_qty = jnp.minimum(peak, remaining)
            new_slice_count = slice_count + jnp.int32(1)
            new_remaining = remaining - slice_qty
            slice_price_int = base_price_int + ppd * new_slice_count

            # Build the regen LIMIT message. Dispatch is gated by
            # ``need_regen`` via ``lax.cond``; the cond is mandatory —
            # we cannot unconditionally call ``process_order_array``
            # without changing observable behaviour.
            jaxob_side = jnp.int32(1) - jnp.int32(2) * side  # BUY=0->+1, SELL=1->-1
            if rank_ctx is None:
                stamp = ts_c
            else:
                stamp, is_ranked = _ranked_stamp(
                    rank_ctx, side, order_id, n_dyn,
                )
            msg = jnp.stack([
                jnp.int32(1),       # msg_type LIMIT
                jaxob_side,
                slice_qty,
                slice_price_int,
                order_id,
                agent_id,
                stamp,              # time_s
                jnp.int32(0),       # time_ns
            ])

            new_lob = jax.lax.cond(
                need_regen,
                lambda lob_msg: self._ob.process_order_array(*lob_msg),
                lambda lob_msg: lob_msg[0],
                (lob_c, msg),
            )

            # Cascade-continuation detection: after
            # dispatching the regen slice, re-check whether this slot's
            # OID is still absent/consumed in the post-dispatch book. If
            # the freshly-dispatched slice rested visibly, the slot is
            # quiescent (no re-run needed for it). If the slice was
            # itself immediately re-consumed by still-resting opposite
            # liquidity, the slot remains peak_consumed=True and the
            # outer fixed-point must run another pass. ``need_cleanup``
            # (slot removed) and inactive slots never carry forward.
            post_row = self._ob.get_order(new_lob, lookup_side, order_id)
            slice_re_consumed = need_regen & (post_row[2] <= jnp.int32(0))

            regen_row = (
                slot
                .at[COL_REMAINING_HIDDEN].set(new_remaining)
                .at[COL_SLICE_COUNT].set(new_slice_count)
            )

            # Mechanism (2a) swap-with-last on cleanup: read the tail
            # row at index ``n_active - 1``, write it into the freed
            # slot ``i``, then clear the old tail. Done branchlessly
            # via ``jnp.where`` between the regen and cleanup row plans.
            tail_idx = n_active_c - jnp.int32(1)
            tail_row = jax.lax.dynamic_index_in_dim(
                meta_c, tail_idx, axis=0, keepdims=False,
            )

            # The row to write at slot ``i``:
            #   need_regen  -> regen_row (regenerated; same slot)
            #   need_cleanup -> tail_row (swap-with-last)
            #   else         -> slot (unchanged)
            new_slot_at_i = jnp.where(
                need_regen,
                regen_row,
                jnp.where(need_cleanup, tail_row, slot),
            )
            meta_after_i = meta_c.at[i].set(new_slot_at_i)

            # On cleanup, ALSO write the sentinel into the old tail
            # slot. We use a dynamic_update_slice via .at[tail_idx].set
            # gated by need_cleanup so the unchanged path leaves the
            # tail untouched. Note: when ``i == tail_idx``
            # (single-iceberg self-swap), the tail-clear correctly
            # overwrites the row we just wrote (matches the (2a)
            # self-swap-then-clear contract).
            meta_after_tail_clear = meta_after_i.at[tail_idx].set(sentinel_row)
            new_meta = jnp.where(
                need_cleanup, meta_after_tail_clear, meta_after_i,
            )

            # Update n_active: decrement on cleanup, else unchanged.
            new_n_active = n_active_c - jnp.where(
                need_cleanup, jnp.int32(1), jnp.int32(0),
            )

            # Update ts_counter only when a regen actually dispatched.
            new_ts = jnp.where(
                need_regen, ts_c + jnp.int32(1), ts_c,
            ).astype(jnp.int32)

            # Advance ``i`` only when we did NOT clean up — on cleanup,
            # the swap-with-last places a previously-unprocessed row at
            # ``i`` and shrinks the outer bound; we want to re-visit
            # ``i`` on the next iteration.
            new_i = i + jnp.where(need_cleanup, jnp.int32(0), jnp.int32(1))

            # OR-accumulate whether this pass LEFT any slot still
            # peak-consumed after its regen dispatch (a cascade
            # continuation). A slot that regenerated and rested visibly,
            # a cleaned-up slot, and an inactive slot all contribute
            # False — so a non-cascading regen does NOT force a wasteful
            # extra confirmation pass. Only a genuine re-consumption
            # (``slice_re_consumed``) re-arms the outer fixed-point via
            # this self-OID channel; CROSS-slot consumption is caught by
            # the post-pass re-derivation in ``_single_pass`` (OQ-23).
            new_dirty = dirty | slice_re_consumed

            # Track whether this pass dispatched a regen slice that
            # actually TRADED (``post_row[1] < slice_qty`` — some of the
            # dispatched quantity did not rest) — the gate for the OQ-23
            # cross-slot re-derivation. Soundness of the skip: cleanup
            # iterations mutate only ``meta`` (never the book), and a
            # dispatched slice that rested IN FULL removed nothing from
            # the opposite side — so a trade-free pass cannot newly
            # consume any slot's peak. Only a pass that traded can have
            # aggressed another slot's resting peak.
            new_dispatched = dispatched | (
                need_regen & (post_row[1] < slice_qty)
            )

            if rank_ctx is None:
                return (
                    new_lob, new_meta, new_n_active, new_ts, new_i,
                    new_dirty, new_dispatched,
                )
            new_n_dyn = n_dyn + jnp.where(
                need_regen & ~is_ranked, jnp.int32(1), jnp.int32(0),
            )
            return (
                new_lob, new_meta, new_n_active, new_ts, new_i,
                new_dirty, new_dispatched, new_n_dyn,
            )

        # ----- Bounded fixed-point cascade-chase -----
        # Outer loop: re-run the inner slot scan until a full pass
        # observes no ``peak_consumed`` slot (quiescence), capped by
        # ``max_iters``. The cap-trip exit (still-dirty at the cap) is
        # not surfaced via a new flag — the leftover slot's inflated
        # ``slice_count`` trips the existing ``truncation_flag``
        # at ``_build_state_jit``, ORing into ``OMIEBookState.truncated``.
        def _rank_context(lob_p, meta_p, n_active_p, ts_p):
            """Exhaustion-order ranking: one ``fori_loop`` over
            ``[0, n_active)`` marks the ``prior_lob`` row of every *ranked*
            slot (peak absent from the pass-start book, hidden remainder
            > 0, side != ``aggressor_side``, live row with its OID on its
            side in ``prior_lob``) with the slot index. Returns
            ``(ts_p, k0, k0_buy, marks_b, marks_a)``; ``marks_*`` are
            int32 per-row slot indices (-1 = unmarked)."""
            n_rows_b = prior_bids.shape[0]
            n_rows_a = prior_asks.shape[0]
            cur_bids = jnp.asarray(lob_p.bids, jnp.int32)
            cur_asks = jnp.asarray(lob_p.asks, jnp.int32)

            def mark_body(k, acc):
                marks_b, marks_a = acc
                slot_k = jax.lax.dynamic_index_in_dim(
                    meta_p, k, axis=0, keepdims=False,
                )
                oid = slot_k[COL_ORDER_ID]
                side_k = slot_k[COL_SIDE]
                is_buy = side_k == jnp.int32(int(Side.BUY))
                cur = jnp.where(is_buy, cur_bids, cur_asks)
                present_now = jnp.any(
                    (cur[:, _OID] == oid) & (cur[:, _Q] > jnp.int32(0))
                )
                prior = jnp.where(is_buy, prior_bids, prior_asks)
                hit = (prior[:, _OID] == oid) & (prior[:, _Q] > jnp.int32(0))
                is_r = (
                    ~present_now
                    & (slot_k[COL_REMAINING_HIDDEN] > jnp.int32(0))
                    & jnp.any(hit)
                )
                if aggressor_side is not None:
                    is_r = is_r & (side_k != aggressor_side_i)
                idx = jnp.argmax(hit)
                k_i = jnp.asarray(k, jnp.int32)
                marks_b = marks_b.at[idx].set(
                    jnp.where(is_r & is_buy, k_i, marks_b[idx]))
                marks_a = marks_a.at[idx].set(
                    jnp.where(is_r & ~is_buy, k_i, marks_a[idx]))
                return marks_b, marks_a

            marks_b, marks_a = jax.lax.fori_loop(
                jnp.int32(0), n_active_p, mark_body,
                (jnp.full((n_rows_b,), -1, jnp.int32),
                 jnp.full((n_rows_a,), -1, jnp.int32)),
            )
            k0_buy = jnp.sum(marks_b >= jnp.int32(0), dtype=jnp.int32)
            k0 = k0_buy + jnp.sum(marks_a >= jnp.int32(0), dtype=jnp.int32)
            return (ts_p, k0, k0_buy, marks_b, marks_a)

        def _single_pass(lob_p, meta_p, n_active_p, ts_p, rank=False):
            if rank:
                rank_ctx = _rank_context(lob_p, meta_p, n_active_p, ts_p)
                pass_carry = (
                    lob_p, meta_p, n_active_p, ts_p,
                    jnp.int32(0), jnp.bool_(False), jnp.bool_(False),
                    jnp.int32(0),
                )
                (
                    lob_o, meta_o, n_active_o, ts_o, _i_o, dirty_o,
                    dispatched_o, _n_dyn_o,
                ) = jax.lax.while_loop(
                    cond_fn, _make_body_fn(rank_ctx), pass_carry,
                )
            else:
                pass_carry = (
                    lob_p, meta_p, n_active_p, ts_p,
                    jnp.int32(0), jnp.bool_(False), jnp.bool_(False),
                )
                (
                    lob_o, meta_o, n_active_o, ts_o, _i_o, dirty_o,
                    dispatched_o,
                ) = jax.lax.while_loop(cond_fn, body_fn, pass_carry)

            # ---- OQ-23 cross-slot re-arm ----
            # The body's ``slice_re_consumed`` channel observes only each
            # slot's OWN OID, so a regen slice that PARTIALLY RESTS while
            # consuming an already-visited, lower-index, opposite-side
            # slot's visible peak leaves that slot ``peak_consumed=True``
            # without setting ``dirty``. Re-derive ``any_active_peak_
            # consumed`` across the packed prefix at pass end — the SAME
            # ``get_order(...)[2] <= 0`` derivation the test-suite
            # ``_quiescence_violations`` probe uses — and OR it into the
            # pass's ``dirty`` verdict.
            #
            # Cost gating (gate tightened to the trade
            # predicate to keep the warm-time overhead <=10%):
            #   * run the re-derivation ONLY when the pass dispatched a
            #     regen slice that TRADED (``dispatched_o``; see
            #     ``new_dispatched`` in the body) — cleanup-only work
            #     never mutates the book and a slice that rested in full
            #     removed nothing from the opposite side, so a trade-free
            #     pass cannot create a new consumed peak; the quiescent
            #     common path (and the common rest-without-crossing regen
            #     path) pays one boolean; and
            #   * skip it when ``dirty_o`` is already True — another pass
            #     runs regardless and re-evaluates at ITS end, so the
            #     extra reads would be unobservable.
            # The scan is ``n_active``-bounded (fori_loop +
            # dynamic_index_in_dim, the ``matchable_hidden_volume``
            # pattern) — cost tracks the live count, not MAX_ICEBERGS.
            def _any_active_peak_consumed():
                def rearm_body(k, acc):
                    slot_k = jax.lax.dynamic_index_in_dim(
                        meta_o, k, axis=0, keepdims=False,
                    )
                    lookup_k = jnp.int32(1) - slot_k[COL_SIDE]
                    row_k = self._ob.get_order(
                        lob_o, lookup_k, slot_k[COL_ORDER_ID],
                    )
                    return acc | (row_k[2] <= jnp.int32(0))

                return jax.lax.fori_loop(
                    jnp.int32(0), n_active_o, rearm_body, jnp.bool_(False),
                )

            cross_slot_consumed = jax.lax.cond(
                dispatched_o & ~dirty_o,
                _any_active_peak_consumed,
                lambda: jnp.bool_(False),
            )
            return lob_o, meta_o, n_active_o, ts_o, dirty_o | cross_slot_consumed

        def outer_cond(carry):
            _lob, _meta, _na, _ts, dirty, it = carry
            return dirty & (it < max_iters)

        def outer_body(carry):
            lob_o, meta_o, n_active_o, ts_o, _dirty, it = carry
            lob_n, meta_n, n_active_n, ts_n, dirty_n = _single_pass(
                lob_o, meta_o, n_active_o, ts_o,
            )
            return (lob_n, meta_n, n_active_n, ts_n, dirty_n, it + jnp.int32(1))

        # Prime the fixed point with one pass so the outer ``dirty``
        # predicate reflects the first scan's observation.
        # Exhaustion-order ranking: only the primed pass is
        # ranked, and only when the caller passed ``prior_lob``.
        lob_1, meta_1, n_active_1, ts_1, dirty_1 = _single_pass(
            lob, iceberg_meta, n_active_i, ts_counter,
            rank=prior_lob is not None,
        )
        init_carry = (lob_1, meta_1, n_active_1, ts_1, dirty_1, jnp.int32(1))
        lob_out, meta_out, n_active_out, ts_out, _dirty_out, _it_out = (
            jax.lax.while_loop(outer_cond, outer_body, init_carry)
        )
        return (lob_out, meta_out, n_active_out, ts_out)

    # ------------------------------------------------------------------
    # Hidden volume query — JIT-safe scan of opposite-side icebergs.
    # ------------------------------------------------------------------

    def matchable_hidden_volume(
        self,
        iceberg_meta: jnp.ndarray,
        n_active,
        side,
        limit_price_int,
    ) -> jnp.ndarray:
        """Return total hidden iceberg qty (jaxob int) crossing the price.

        For each opposite-side iceberg in the packed prefix
        ``meta[:n_active]``, project the future slice sequence using a
        bounded ``while_loop``. A slice whose drifted price
        (``base_price_int + ppd * slice_idx``) crosses the aggressor's
        ``limit_price_int`` contributes its quantity to the total.

        The outer scan over icebergs is
        bounded by ``n_active`` — we ``lax.fori_loop`` over the prefix
        rather than vmap over the full ``(MAX_ICEBERGS, 8)`` meta. The
        old vmap-over-full-meta gave correctness (the ``active`` mask
        zeroed sentinel contributions) but ran over all ``MAX_ICEBERGS``
        rows regardless of live count — the bounded form is the actual
        perf-independence-from-cap mechanism for the hidden-volume
        query path.

        Args:
            iceberg_meta: ``(MAX_ICEBERGS, 8)`` int32 array.
            n_active: 0-d int32 — count of live iceberg slots.
            side: incoming aggressor's ``Side`` (0=BUY, 1=SELL).
            limit_price_int: aggressor's offset-encoded jaxob price.

        Returns:
            Total hidden matchable quantity in jaxob int units (0-d
            int32 array).
        """
        side_i = jnp.asarray(side, jnp.int32)
        limit_i = jnp.asarray(limit_price_int, jnp.int32)
        n_active_i = jnp.asarray(n_active, jnp.int32)
        # opposite_side: BUY (0) -> SELL (1); SELL (1) -> BUY (0)
        opposite_side = jnp.int32(1) - side_i
        # Aggressor-side crossing direction is loop-invariant.
        #   BUY aggressor: ask price <= buy limit
        #   SELL aggressor: bid price >= sell limit
        is_buy = side_i == jnp.int32(int(Side.BUY))

        def slot_contribution(slot):
            slot_side = slot[COL_SIDE]
            remaining = slot[COL_REMAINING_HIDDEN]
            peak = slot[COL_PEAK_QTY]
            base_price = slot[COL_BASE_PRICE_INT]
            ppd = slot[COL_PPD]
            slice_count = slot[COL_SLICE_COUNT]

            # Under the packed-layout invariant, rows visited here are
            # already live (i < n_active); the only side-channel filter
            # left is the opposite-side check.
            active = (
                (slot_side == opposite_side)
                & (remaining > jnp.int32(0))
                & (peak > jnp.int32(0))
            )

            # Bounded while_loop projecting up to MAX_SLICES_PER_REGEN
            # future slices. Carry: (remaining_left, slice_idx, total).
            def cond_fn(carry):
                rem_left, slice_idx, _total = carry
                return (rem_left > jnp.int32(0)) & (
                    slice_idx <= jnp.int32(MAX_SLICES_PER_REGEN)
                )

            def body_fn(carry):
                rem_left, slice_idx, total = carry
                slice_qty = jnp.minimum(peak, rem_left)
                slice_price = base_price + ppd * slice_idx
                crosses = jnp.where(
                    is_buy, slice_price <= limit_i, slice_price >= limit_i
                )
                contrib = jnp.where(crosses, slice_qty, jnp.int32(0))
                return (
                    rem_left - slice_qty,
                    slice_idx + jnp.int32(1),
                    total + contrib,
                )

            init = (
                jnp.where(active, remaining, jnp.int32(0)),
                slice_count + jnp.int32(1),
                jnp.int32(0),
            )
            _, _, contrib_total = jax.lax.while_loop(cond_fn, body_fn, init)
            return contrib_total

        # Bounded outer loop over the packed prefix [0, n_active). We
        # use ``lax.fori_loop`` over ``i`` and read ``meta[i]`` via
        # ``dynamic_index_in_dim`` — this lowers to a ``while_loop``
        # whose trip-count tracks ``n_active``, not ``MAX_ICEBERGS``.
        def outer_body(i, total):
            slot = jax.lax.dynamic_index_in_dim(
                iceberg_meta, i, axis=0, keepdims=False,
            )
            return total + slot_contribution(slot)

        total = jax.lax.fori_loop(
            jnp.int32(0), n_active_i, outer_body, jnp.int32(0),
        )
        return total.astype(jnp.int32)

    # ------------------------------------------------------------------
    # Cleanup — swap-with-last delete by order_id (packed layout).
    # ------------------------------------------------------------------

    def cleanup(
        self,
        iceberg_meta: jnp.ndarray,
        n_active,
        order_id,
    ) -> tuple:
        """Free the slot whose ``order_id`` equals the given id.

        Packed-layout swap-with-last:
            1. Find the slot index whose ``COL_ORDER_ID == order_id``.
               If no match, this is a no-op (returns the meta and
               ``n_active`` unchanged).
            2. Read the live-tail row at ``meta[n_active - 1]``.
            3. Write the tail row into the freed slot (swap into place).
            4. Clear the old tail to ``SENTINEL_ROW``.
            5. Decrement ``n_active`` by 1.

        Per-op JAX cost:
            * 2 ``.at[...].set(...)`` writes (swap + tail-clear)
            * 1 int32 subtract (``n_active -= 1``)

        Self-swap case (``slot == n_active - 1``): the tail row is
        written into itself (no-op write), then immediately cleared to
        sentinel. Net effect: the single live slot is cleared. This is
        the only case where slot-reuse-after-cleanup tests on a
        single-iceberg cycle observe the same shape as an in-place
        clear; with more live slots the tail row moves into the freed slot.

        Preconditions:
            ``order_id == SENTINEL_ORDER_ID`` is an explicit no-op
            under a defensive guard:
            ``any_match`` is conjoined with ``oid != SENTINEL_ORDER_ID``
            so that a sentinel order-id short-circuits to a strict
            no-op (meta unchanged, ``n_active`` unchanged). This
            prevents argmax-on-match-mask from matching tail sentinel
            rows and silently corrupting state.

        Args:
            iceberg_meta: ``(MAX_ICEBERGS, 8)`` int32 array.
            n_active: 0-d int32 — count of live iceberg slots.
            order_id: order id to free.

        Returns:
            ``(updated_meta, new_n_active)`` — both are JAX arrays with
            the same shape/dtype as the inputs.
        """
        n_active_i = jnp.asarray(n_active, jnp.int32)
        oid = jnp.asarray(order_id, jnp.int32)
        sentinel_row = _sentinel_row()

        # Find the slot index of the matching order_id. We use the
        # argmax-on-match-mask trick — argmax returns the first True
        # index (deterministic).
        match_mask = iceberg_meta[:, COL_ORDER_ID] == oid
        any_match = jnp.any(match_mask) & (oid != jnp.int32(SENTINEL_ORDER_ID))
        slot_idx = jnp.argmax(match_mask.astype(jnp.int32))

        # Read the live-tail row at n_active - 1. Clamp to 0 if
        # n_active==0 to avoid out-of-bounds indexing; the gate below
        # discards the result in that case.
        tail_idx = jnp.where(
            n_active_i > jnp.int32(0), n_active_i - jnp.int32(1), jnp.int32(0),
        )
        tail_row = jax.lax.dynamic_index_in_dim(
            iceberg_meta, tail_idx, axis=0, keepdims=False,
        )

        # Swap-with-last: write tail_row into slot_idx, then clear the
        # old tail to sentinel.
        meta_after_swap = iceberg_meta.at[slot_idx].set(tail_row)
        meta_after_clear = meta_after_swap.at[tail_idx].set(sentinel_row)

        # Gate everything on a successful match: if no match, return
        # meta and n_active unchanged.
        updated_meta = jnp.where(any_match, meta_after_clear, iceberg_meta)
        new_n_active = n_active_i - jnp.where(
            any_match, jnp.int32(1), jnp.int32(0),
        )
        return (updated_meta, new_n_active)

    def clear_all(self) -> jnp.ndarray:
        """Return a fresh empty meta array (all slots sentinel).

        Sized at the ``MAX_ICEBERGS`` hard ceiling (``empty_meta()``'s
        zero-arg default) — this is the legacy no-arg clear used by callers
        that do not thread the runtime ``max_icebergs`` cap.
        """
        return self.empty_meta()

    # ------------------------------------------------------------------
    # Truncation backstop (carry-flag accessor).
    # ------------------------------------------------------------------

    def truncation_flag(
        self,
        iceberg_meta: jnp.ndarray,
        n_active=None,
    ) -> jnp.ndarray:
        """Return True if any active slot would exceed MAX_SLICES_PER_REGEN.

        Truncation backstop (active in every build): scans the meta and
        returns ``True`` if any active slot's worst-case slice count
        (``ceil(remaining_hidden / peak_qty) + slice_count``) exceeds
        ``MAX_SLICES_PER_REGEN``. This is the "silent truncation
        hazard" guard documented in ``jax_constants.py`` (MAX_SLICES_PER_REGEN).

        The check is intentionally pessimistic — it triggers as soon as
        an iceberg with too-large hidden volume is registered, even
        before any regeneration call. Callers that need to detect
        truncation post-regenerate may invoke this on the returned meta.

        When ``n_active`` is supplied, the
        scan is bounded by ``n_active`` instead of ``MAX_ICEBERGS``.
        Under the packed-layout invariant, rows in
        ``[n_active, MAX_ICEBERGS)`` are sentinels — both forms give
        the same OR-reduction result, but the bounded form is
        cap-independent in cost. When ``n_active`` is ``None`` (legacy
        call site), the scan covers the full meta; correctness is
        preserved either way because the sentinel ``active`` mask zeros
        their contribution.

        Args:
            iceberg_meta: ``(MAX_ICEBERGS, 8)`` int32 array.
            n_active: optional 0-d int32 — count of live iceberg slots.
                When ``None``, the scan ranges over the full meta.

        Returns:
            0-d ``jnp.bool_`` — True iff any slot would truncate.
        """
        if n_active is None:
            order_id = iceberg_meta[:, COL_ORDER_ID]
            active = order_id != jnp.int32(SENTINEL_ORDER_ID)
            remaining = iceberg_meta[:, COL_REMAINING_HIDDEN]
            peak = iceberg_meta[:, COL_PEAK_QTY]
            slice_count = iceberg_meta[:, COL_SLICE_COUNT]

            # ceil(remaining / peak) — guard against peak == 0 by clamping.
            safe_peak = jnp.maximum(peak, jnp.int32(1))
            slices_needed = (remaining + safe_peak - jnp.int32(1)) // safe_peak
            worst_case_total = slice_count + slices_needed
            per_slot = active & (
                worst_case_total > jnp.int32(MAX_SLICES_PER_REGEN)
            )
            return jnp.any(per_slot)

        # Bounded form: scan only the [0, n_active) prefix via
        # lax.fori_loop. Cost is independent of MAX_ICEBERGS.
        n_active_i = jnp.asarray(n_active, jnp.int32)

        def body(i, found):
            slot = jax.lax.dynamic_index_in_dim(
                iceberg_meta, i, axis=0, keepdims=False,
            )
            remaining = slot[COL_REMAINING_HIDDEN]
            peak = slot[COL_PEAK_QTY]
            slice_count = slot[COL_SLICE_COUNT]
            safe_peak = jnp.maximum(peak, jnp.int32(1))
            slices_needed = (remaining + safe_peak - jnp.int32(1)) // safe_peak
            worst_case_total = slice_count + slices_needed
            this_truncates = worst_case_total > jnp.int32(MAX_SLICES_PER_REGEN)
            return jnp.logical_or(found, this_truncates)

        return jax.lax.fori_loop(
            jnp.int32(0), n_active_i, body, jnp.bool_(False),
        )
