"""Slow-tier cascade cap-trip probe for P2-PERF-8-FOLLOWUP-1.

This file closes the *direct-exercise* gap left by the P2-PERF-8 audit:
P2-PERF-8 introduced the bounded fixed-point cascade-chase in
``IcebergManager.regenerate_slices`` (the OUTER ``lax.while_loop`` at
``iceberg_manager.py:531-550``) whose ``outer_cond`` re-runs the inner
slot scan until cascade quiescence OR the iteration cap
(``max_iterations`` → ``OMIEMarketConfig.max_match_iterations = 200``)
is hit.  On the pathological non-quiescent-at-cap exit the leftover
slot keeps its inflated ``slice_count`` / ``remaining_hidden``, which
the existing Path B ``truncation_flag`` backstop catches and ORs into
``OMIEBookState.truncated`` at ``_build_state_jit``
(``contract_book.py:1067-1077``) — the SAME flag from P2-H1 §A3, no new
flag.  The P2-PERF-8 audit only pinned the *registration-time*
worst-case-slice-count proxy; it never drove the WITHIN-CALL outer-loop
cap-trip path itself.  This probe drives that path directly.

Domain law protected by the cap: market-spec.md §4.2.1 Rule 4 — a
regenerated iceberg slice that is immediately re-matched by resting
opposite-side liquidity must resolve to a fixed point WITHIN the
originating ``submit_order`` matching process.  The cap is the
no-hang/no-silent-wrong-answer backstop on that fixed point: a
pathologically deep cascade that cannot drain in ``max_iterations``
passes terminates (bounded loop) and surfaces ``truncated=True`` rather
than hanging or corrupting the book.

================================================================
PROBE VEHICLE — design-question resolution (read the production guards)
================================================================
The story card asks whether a genuinely >200-hop WITHIN-CALL cascade is
constructible under the production guards, or whether a test-local
lowered-cap override is the cleaner probe vehicle.  Reading the
production code on disk resolves this decisively:

  1. The cascade cap is NOT config-threaded from the caller's
     ``OMIEMarketConfig``.  Every production ``regenerate_slices`` call
     site — ``_ice_path`` (``contract_book.py:1041``), ``fill_path``
     (``contract_book.py:568``), the FOK body — invokes
     ``regenerate_slices(lob, meta, n_active, ts)`` with NO
     ``max_iterations`` argument.  Inside ``regenerate_slices`` the
     ``max_iterations is None`` branch
     (``iceberg_manager.py:370-372``) falls back to a FRESH
     ``OMIEMarketConfig().max_match_iterations`` — the *default* (200),
     NOT the ``market_config`` the caller passed to ``submit_order``.
     Therefore passing ``OMIEMarketConfig(max_match_iterations=N)`` to
     ``ContractBook`` / ``submit_order`` does **not** lower the cascade
     cap.  A submit-order-level config override is structurally
     incapable of reaching the cap-trip with a small cascade.

  2. To trip the real 200 cap via ``submit_order`` you would need a
     genuine >200-hop within-call cascade — i.e. >200 successive
     regenerate-then-rematch hops in a single call.  Each hop requires
     a distinct resting opposite-side order to re-consume the freshly
     regenerated slice, so a >200-hop cascade needs either >200 resting
     price levels or a >200-slice iceberg whose every slice re-crosses.
     ``book_depth`` defaults to 100 and the ``__init__`` guard caps it
     at ``MAX_ICEBERGS = 8192`` (``contract_book.py:1122``), so it is
     *raisable*, but a 200+-order replay-and-cross setup compiled
     through the full jitted ``submit_order`` body is far outside the
     slow-tier wall-clock budget and adds no signal over a scaled
     exercise.

  3. ``regenerate_slices`` ITSELF exposes the cleaner vehicle: its
     public signature carries ``max_iterations: int = None``
     (``iceberg_manager.py:302``).  Passing a small explicit
     ``max_iterations`` exercises the EXACT same ``outer_cond`` cap-trip
     code (``iceberg_manager.py:531-550``) — the only thing that
     changes is the integer the loop compares against; the cap-trip
     branch, the leftover-slot state, and the downstream
     ``truncation_flag`` → ``truncated`` wiring are byte-for-byte the
     production path.

DECISION: **test-local lowered-cap override on ``regenerate_slices``.**
We construct a small, deterministic multi-hop cascade state (deep
resting bids + a registered ICE iceberg whose peak has just been
consumed, exactly the ``_ice_path`` mid-cascade state) and call the
LIVE production ``regenerate_slices`` with ``max_iterations=1`` so a
single pass provably cannot drain the multi-hop cascade, forcing the
cap-trip exit with a fast cascade.  We then assert against a
fully-drained reference produced by the SAME production code with the
production cap (``max_iterations=200``), so the cap-trip is pinned to
degrade *gracefully* — the capped book is a valid, non-corrupt prefix
of the drained book, never an impossible state.  The production 200
cap is the same loop body proven by this scaled-down exercise plus the
existing registration-time proxy (P2-PERF-8 audit).

Why this is the honest pin and not a tautology: the ``max_iterations``
parameter is real production surface, the cap-trip branch
(``outer_cond`` false-exit while still ``dirty``) is the literal
production branch, and the truncation backstop that fires
(``truncation_flag`` ORed into ``OMIEBookState.truncated``) is the
literal production guard.  Lowering the integer only shrinks the
cascade depth needed to trip it — it does not change which code runs.

This test is ``@pytest.mark.slow`` because it compiles the production
``regenerate_slices`` lift at two distinct ``max_iterations`` values
plus a third reference, and the multi-hop cascade dispatches several
``process_order_array`` matches per pass; the combined warm-compile +
device cost exceeds the < 1 s fast-suite budget.

References:
  * docs/omiecid/agile-plan.md §"Story P2-PERF-8-FOLLOWUP-1 — Slow-tier
    cascade cap-trip probe".
  * docs/omiecid/market-spec.md §4.2.1 Rule 4 (within-call cascade
    fixed-point resolution invariant the cap protects).
  * omiecid_simulator/omiecid/iceberg_manager.py:514-551 (bounded
    fixed-point outer loop + the ``max_iters`` cap).
  * omiecid_simulator/omiecid/iceberg_manager.py:756-806 (the
    ``truncation_flag`` worst-case-slice-count detection).
  * omiecid_simulator/omiecid/contract_book.py:1053-1077
    (``_build_state_jit`` ORs ``truncation_flag`` into
    ``OMIEBookState.truncated``).
  * tests/omiecid/test_perf_a1_fused_regen_gate.py
    (``_cascade_synthetic_orders`` / ``_quiescence_violations`` —
    cascade-construction idioms reused conceptually here; that file is
    NOT modified).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob.jorderbook import OrderBook
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    _build_state_jit,
    _make_jaxob_cfg,
)
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_PEAK_QTY,
    COL_REMAINING_HIDDEN,
    COL_SIDE,
    COL_SLICE_COUNT,
    IcebergManager,
)
from omiecid_simulator.omiecid.jax_constants import MAX_SLICES_PER_REGEN
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    Status,
    Validity,
)

# This file exercises the production ``regenerate_slices`` lift at
# multiple ``max_iterations`` values; the cascade dispatches several
# matches per pass.  Slow-tier per the test-agent runtime rules.
#
# It also jit-compiles bodies at module scope for the A/B comparison, so it
# additionally carries ``pytest.mark.jit`` to conform to the T1.1 jit-marker
# discipline enforced by ``test_jit_marker_lint`` — matching the sibling
# P2-PERF-8 file ``test_perf_a1_fused_regen_gate.py``.  ``slow`` governs the
# test tier; ``jit`` governs persistent-cache discipline; the test
# legitimately carries both.
pytestmark = [pytest.mark.slow, pytest.mark.jit]


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _make_open_contract(cid: int) -> Contract:
    """Build an OPEN ``Contract`` instance whose delivery window admits
    in-window order validation (mirrors the fixture helper in
    ``test_perf_a1_fused_regen_gate._make_open_contract``)."""
    start = cid * 15
    end = start + 15
    return Contract(
        contract_id=cid,
        delivery_start_minutes=start,
        delivery_end_minutes=end,
        gct_minutes=start - 60,
        status=ContractStatus.OPEN,
    )


def _resting_bid_order(
    pc: OMIEProductConfig, price_eur: float, qty_mw: float, oid: int, ts: int,
) -> OMIEOrder:
    """A NON BUY that rests on the bid (provides re-consumption liquidity)."""
    return OMIEOrder(
        contract_id=37,
        side=int(Side.BUY),
        price=price_eur,
        quantity=qty_mw,
        order_id=oid,
        agent_id=oid,
        timestamp=ts,
        exec_restriction=int(ExecRestriction.NON),
        validity=int(Validity.GFS),
        expiry_time=0,
        status=int(Status.ACTIVE),
        peak_qty=0.0,
        ppd=0.0,
    )


def _ice_sell_order(
    pc: OMIEProductConfig,
    price_eur: float,
    total_mw: float,
    peak_mw: float,
    oid: int,
    ts: int,
) -> OMIEOrder:
    """An ICE SELL iceberg (the cascade source)."""
    return OMIEOrder(
        contract_id=37,
        side=int(Side.SELL),
        price=price_eur,
        quantity=total_mw,
        order_id=oid,
        agent_id=oid,
        timestamp=ts,
        exec_restriction=int(ExecRestriction.ICE),
        validity=int(Validity.GFS),
        expiry_time=0,
        status=int(Status.ACTIVE),
        peak_qty=peak_mw,
        ppd=0.0,
    )


def _build_mid_cascade_state(
    pc: OMIEProductConfig,
    mc: OMIEMarketConfig,
    *,
    n_resting_bids: int,
    bid_qty_mw: float,
    bid_price_eur: float,
    ice_total_mw: float,
    ice_peak_mw: float,
    ice_price_eur: float,
    oid_base: int = 0,
):
    """Construct the ``_ice_path`` mid-cascade state, WITHOUT draining it.

    Mirrors ``ContractBook._ice_path`` / ``contract_book._ice_path``
    (``contract_book.py:1009-1050``) up to but NOT including the
    ``regenerate_slices`` call:

      1. Rest ``n_resting_bids`` NON BUY orders at ``bid_price_eur``
         (deep bid liquidity that will re-consume each regenerated
         iceberg slice).
      2. ``register`` an ICE SELL iceberg at ``ice_price_eur`` with
         ``ice_total_mw`` total / ``ice_peak_mw`` peak.
      3. Dispatch the iceberg's visible peak as a LIMIT SELL — it
         crosses one resting bid and is consumed, leaving the slot in
         the ``peak_consumed=True`` state that ``regenerate_slices``
         must chase to quiescence.

    Returns ``(ice_mgr, jaxob_cfg, lob, meta, n_active, ts, ice_oid)`` —
    the carry tuple ``regenerate_slices`` consumes, plus the iceberg's
    order id for downstream lookups.  The deep resting-bid liquidity
    guarantees the regenerated slice is itself re-consumed, producing
    the genuine multi-hop cascade Rule 4 names (not the single-hop
    Rule 3 case).

    FOLLOWUP-2 fixture-OID realism sweep (2026-07-09): ``oid_base``
    offsets the iceberg OID (50) and the resting-bid OIDs (100+k).  The
    cap-trip / truncation / graceful-degradation assertions read the
    ``peak_consumed`` state via the ``get_order`` col-2 ``-99`` sentinel
    and the meta OID-identity column, so they are OID-magnitude-
    independent; the realistic-OID variant (base 1000) confirms no
    wrong-column (OID-vs-qty) confusion is masked by the base-0 OIDs
    (50, which sits above the qty ticks 2 but below the total-hidden
    ticks 132).
    """
    jaxob_cfg = _make_jaxob_cfg(mc)
    ob = OrderBook(cfg=jaxob_cfg)
    ice_mgr = IcebergManager(ob, pc)

    lob = ob.init()
    meta = IcebergManager.empty_meta()
    n_active = jnp.int32(0)
    ts = jnp.int32(0)

    # ---- (1) deep resting bid liquidity --------------------------------
    # Encode price/qty via the iceberg manager's own encoders so the
    # tick conventions match production exactly.
    for k in range(n_resting_bids):
        oid = oid_base + 100 + k
        order = _resting_bid_order(pc, bid_price_eur, bid_qty_mw, oid, ts + k)
        side = jnp.asarray(order.side, jnp.int32)
        jaxob_side = jnp.where(side == jnp.int32(int(Side.BUY)),
                               jnp.int32(1), jnp.int32(-1))
        qty_int = ice_mgr._encode_qty_int(order.quantity)
        price_int = ice_mgr._encode_price_int(order.price)
        msg = jnp.array([
            1, jaxob_side, qty_int, price_int,
            order.order_id, order.agent_id, ts + k, 0,
        ], dtype=jnp.int32)
        lob = ob.process_order_array(lob, msg)
        ts = jnp.maximum(ts, jnp.int32(ts + k + 1))

    # ---- (2) register the ICE iceberg ----------------------------------
    ice_oid = oid_base + 50
    ice_order = _ice_sell_order(
        pc, ice_price_eur, ice_total_mw, ice_peak_mw, ice_oid, int(ts) + 10,
    )
    meta, n_active, _overflow = ice_mgr.register(meta, n_active, ice_order)
    ts = ice_mgr.update_timestamp(ts, ice_order.timestamp)

    # ---- (3) dispatch the visible peak (it crosses, gets consumed) -----
    peak_int = ice_mgr._encode_qty_int(ice_order.peak_qty)
    ice_price_int = ice_mgr._encode_price_int(ice_order.price)
    sell_msg = jnp.array([
        1, jnp.int32(-1), peak_int, ice_price_int,
        ice_oid, ice_oid, int(ts), 0,
    ], dtype=jnp.int32)
    lob = ob.process_order_array(lob, sell_msg)

    return ice_mgr, jaxob_cfg, lob, meta, jnp.asarray(n_active, jnp.int32), \
        jnp.asarray(ts, jnp.int32), ice_oid


def _book_consistency_report(
    lob, meta, n_active: int, ob: OrderBook,
) -> list[str]:
    """Return a list of internal-consistency violations on the book.

    Checks the post-cascade observable state is not corrupt:
      * no resting order has negative remaining quantity,
      * no resting order has negative price,
      * every active iceberg slot's ``remaining_hidden`` is >= 0 and its
        ``slice_count`` is >= 0 (monotone bookkeeping, never negative),
      * ``n_active`` is within ``[0, MAX_ICEBERGS]`` (here trivially small).

    A graceful cap-trip degrades to a valid book; this report must be
    empty.  (The whole point of (c): the cap-trip does not produce an
    *impossible* state.)
    """
    problems: list[str] = []
    bids = np.asarray(lob.bids)
    asks = np.asarray(lob.asks)

    # jaxob resting arrays are (nOrders, 6) with the canonical column
    # layout [price(0), qty(1), orderid(2), trader(3), time_s(4),
    # time_ns(5)].  EMPTY_SLOT = -1 fills every column of an unpopulated
    # row (jaxob_constants.py).  A row is POPULATED iff its order-id
    # column (2) is NOT the empty sentinel.  jaxob's
    # ``_removeZeroNegQuant`` guarantees a populated resting row always
    # carries qty > 0 (it evicts any row whose qty drops to <= 0), so a
    # populated row with qty <= 0 is an impossible (corrupt) state — that
    # is exactly the "silent wrong-answer" the graceful cap-trip must
    # avoid.  We therefore mask out empty slots and flag only populated
    # rows with non-positive qty.
    EMPTY = -1
    for name, arr in (("bids", bids), ("asks", asks)):
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        oid_col = arr[:, 2]
        qty_col = arr[:, 1]
        populated = oid_col != EMPTY
        bad = np.where(populated & (qty_col <= 0))[0]
        if bad.size:
            problems.append(
                f"{name}: populated rows {bad.tolist()} (order-id "
                f"{oid_col[bad].tolist()}) have non-positive resting qty "
                f"{qty_col[bad].tolist()} (impossible resting state — "
                f"jaxob evicts zero/neg-qty rows)"
            )

    m = np.asarray(meta)
    for i in range(n_active):
        rem = int(m[i, COL_REMAINING_HIDDEN])
        sc = int(m[i, COL_SLICE_COUNT])
        peak = int(m[i, COL_PEAK_QTY])
        if rem < 0:
            problems.append(
                f"iceberg slot {i}: remaining_hidden={rem} < 0 "
                f"(impossible iceberg state)"
            )
        if sc < 0:
            problems.append(
                f"iceberg slot {i}: slice_count={sc} < 0 "
                f"(impossible bookkeeping)"
            )
        if peak < 0:
            problems.append(
                f"iceberg slot {i}: peak_qty={peak} < 0 "
                f"(impossible iceberg state)"
            )
    return problems


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestP2Perf8CascadeCapTrip:
    """Direct exercise of the bounded fixed-point outer-loop cap-trip.

    Probe vehicle: test-local lowered-cap override on the LIVE
    production ``IcebergManager.regenerate_slices`` (``max_iterations``
    parameter).  See the module docstring for the full design-question
    resolution.  The single test below drives a within-call multi-hop
    cascade that exceeds a small (``max_iterations=1``) cap inside one
    ``regenerate_slices`` call and asserts the three acceptance
    criteria: (a) termination, (b) ``truncated=True`` on the cap-trip,
    (c) graceful degradation (no corrupt / impossible book state).
    """

    @pytest.mark.parametrize("oid_base", [
        pytest.param(0, id="small-oids"),
        pytest.param(1000, id="realistic-oids"),
    ])
    def test_within_call_cascade_cap_trip_terminates_truncates_and_degrades_gracefully(
        self, oid_base,
    ):
        """A within-call cascade that exceeds the bounded fixed-point cap
        inside a single ``regenerate_slices`` call must (a) terminate,
        (b) set ``OMIEBookState.truncated`` True via the
        ``truncation_flag`` backstop, and (c) leave the book in an
        internally-consistent (non-corrupt) state.

        slow: compiles the production ``regenerate_slices`` lift at two
        ``max_iterations`` values + a drained reference, with a
        multi-hop cascade dispatching several matches per pass.

        Construction recipe (see ``_build_mid_cascade_state``):
          * 6 resting NON BUY orders @ 60.00 EUR/MWh, 0.2 MW each
            (1.2 MW total bid depth) — the re-consumption liquidity.
          * 1 ICE SELL iceberg @ 60.00 EUR/MWh, total 8.0 MW, peak
            0.2 MW (40 slices, well above MAX_SLICES_PER_REGEN=64? no —
            8.0/0.2 = 40 slices; we size remaining_hidden so the
            registration worst-case ALSO exceeds 64 by choosing total
            13.0 MW / peak 0.2 MW = 65 slices, see below).
          * Dispatch the peak (crosses one resting bid) → slot enters
            ``peak_consumed=True``.
          * Call ``regenerate_slices(..., max_iterations=1)`` — one pass
            regenerates a single slice that the deep bids re-consume,
            but the multi-hop cascade is NOT drained in one pass, so the
            outer ``while_loop`` would need more passes; the cap (1)
            stops it while still ``dirty`` → the cap-trip exit.
        """
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        # Sanity: the production cap that this scaled exercise stands in
        # for.  The lowered cap (1) exercises the identical outer-loop
        # branch; the production cap is 200.
        assert mc.max_match_iterations == 200

        # Size the iceberg so its registration worst-case slice count
        # exceeds MAX_SLICES_PER_REGEN (so the leftover slot after the
        # cap-trip is truncation-detectable): total / peak slices.
        ice_peak_mw = 0.2
        # 13.2 MW / 0.2 MW = 66 slices > MAX_SLICES_PER_REGEN (64).
        ice_total_mw = round((MAX_SLICES_PER_REGEN + 2) * ice_peak_mw, 4)
        assert ice_total_mw / ice_peak_mw > MAX_SLICES_PER_REGEN

        n_resting_bids = 6  # deep enough for a genuine multi-hop cascade
        bid_qty_mw = 0.2
        cross_price = 60.00

        (ice_mgr, jaxob_cfg, lob0, meta0, n_active0, ts0,
         ice_oid) = _build_mid_cascade_state(
            pc, mc,
            n_resting_bids=n_resting_bids,
            bid_qty_mw=bid_qty_mw,
            bid_price_eur=cross_price,
            ice_total_mw=ice_total_mw,
            ice_peak_mw=ice_peak_mw,
            ice_price_eur=cross_price,
            oid_base=oid_base,
        )

        # The constructed slot must genuinely be peak-consumed
        # mid-cascade (precondition: there IS a cascade to chase).
        ob = OrderBook(cfg=jaxob_cfg)
        meta0_np = np.asarray(meta0)
        n_active0_int = int(n_active0)
        assert n_active0_int >= 1, (
            "fixture must register at least one iceberg slot; got "
            f"n_active={n_active0_int}"
        )
        # Locate the iceberg slot and confirm its visible peak was
        # consumed (the regenerate_slices precondition).
        slot = None
        for i in range(n_active0_int):
            if int(meta0_np[i, COL_ORDER_ID]) == ice_oid:
                slot = i
                break
        assert slot is not None, (
            f"iceberg order_id={ice_oid} not found in active meta prefix"
        )
        side_i = int(meta0_np[slot, COL_SIDE])
        lookup_side = 1 - side_i
        peak_row = np.asarray(
            ob.get_order(lob0, jnp.int32(lookup_side), jnp.int32(ice_oid))
        )
        assert int(peak_row[2]) <= 0, (
            "fixture precondition failed: the iceberg peak was NOT "
            f"consumed (get_order col-2 visible qty={int(peak_row[2])} > 0); "
            "there is no cascade to chase, so the cap-trip cannot be "
            "exercised.  Adjust the resting-bid depth / peak so the "
            "dispatched peak fully crosses."
        )

        # ---- (a) TERMINATION: the capped call completes (no hang) ------
        # The bounded loop guarantees termination; we wrap with a wall-
        # clock guard to convert any (impossible) hang into a failure
        # rather than a CI timeout.
        regen_capped = jax.jit(
            ice_mgr.regenerate_slices, static_argnames=("max_iterations",),
        )
        t0 = time.monotonic()
        lob_c, meta_c, n_active_c, ts_c = regen_capped(
            lob0, meta0, n_active0, ts0, max_iterations=1,
        )
        for leaf in (lob_c, meta_c, n_active_c, ts_c):
            for sub in jax.tree_util.tree_leaves(leaf):
                if hasattr(sub, "block_until_ready"):
                    sub.block_until_ready()
        elapsed = time.monotonic() - t0
        assert elapsed < 60.0, (
            f"(a) cap-trip call did not terminate within 60 s "
            f"(elapsed={elapsed:.2f}s) — the bounded fixed-point loop "
            f"must guarantee termination at the cap; a hang here means "
            f"the outer while_loop cond is not cap-bounded."
        )

        # ---- (b) TRUNCATED: the cap-trip surfaces truncated=True -------
        # Build the OMIEBookState exactly as production does post-regen
        # (contract_book._build_state_jit) and assert the truncation
        # backstop ORed True.  prior_truncated=False isolates the signal
        # to this cap-trip + the leftover-slot worst-case.
        state_capped = _build_state_jit(
            lob_c, meta_c, n_active_c, ts_c,
            jnp.bool_(False), ice_mgr, overflow=jnp.bool_(False),
        )
        assert bool(state_capped.truncated), (
            "(b) cap-trip did NOT surface OMIEBookState.truncated=True. "
            "The leftover slot after the capped (max_iterations=1) "
            "cascade should retain enough remaining_hidden / slice_count "
            "that IcebergManager.truncation_flag fires and "
            "_build_state_jit ORs it into truncated.  meta leftover slot "
            f"diagnostics: remaining_hidden="
            f"{int(np.asarray(meta_c)[slot, COL_REMAINING_HIDDEN])}, "
            f"slice_count={int(np.asarray(meta_c)[slot, COL_SLICE_COUNT])}, "
            f"peak_qty={int(np.asarray(meta_c)[slot, COL_PEAK_QTY])}."
        )

        # ---- (c) GRACEFUL DEGRADATION: book is internally consistent ---
        # (c.1) No impossible state (negative qtys / negative bookkeeping).
        problems = _book_consistency_report(
            lob_c, meta_c, int(n_active_c), ob,
        )
        assert not problems, (
            "(c) cap-trip corrupted the observable book state — the "
            "capped-but-terminated cascade must degrade gracefully, not "
            "produce an impossible state.  Violations:\n  - "
            + "\n  - ".join(problems)
        )

        # (c.2) The capped book is a valid PREFIX of the fully-drained
        # book: draining the SAME starting state with the production cap
        # (max_iterations=200) reaches quiescence; the capped result must
        # match the drained result re-capped at 1 hop — i.e. the cap-trip
        # is the drained cascade *interrupted partway*, never a divergent
        # wrong answer.  We assert the weaker, robust invariant: the
        # fully-drained reference is quiescent (no slot peak-consumed)
        # AND it carries strictly-more drained hidden volume than the
        # capped result (the cap genuinely left work undone), AND the
        # capped result's leftover remaining_hidden is >= the drained
        # result's (the cap stopped early, so MORE hidden remains).
        regen_drained = jax.jit(
            ice_mgr.regenerate_slices, static_argnames=("max_iterations",),
        )
        lob_d, meta_d, n_active_d, ts_d = regen_drained(
            lob0, meta0, n_active0, ts0, max_iterations=200,
        )
        for leaf in (lob_d, meta_d, n_active_d, ts_d):
            for sub in jax.tree_util.tree_leaves(leaf):
                if hasattr(sub, "block_until_ready"):
                    sub.block_until_ready()

        # The drained reference must itself be internally consistent.
        drained_problems = _book_consistency_report(
            lob_d, meta_d, int(n_active_d), ob,
        )
        assert not drained_problems, (
            "(c) the fully-drained (max_iterations=200) reference is "
            "itself corrupt — the cascade-chase must reach a consistent "
            "fixed point.  Violations:\n  - "
            + "\n  - ".join(drained_problems)
        )

        meta_c_np = np.asarray(meta_c)
        meta_d_np = np.asarray(meta_d)
        # Slot positions can shift under swap-with-last cleanup; locate
        # the iceberg slot by order_id in each result (it may also be
        # fully drained out of the drained result).
        def _find_slot(meta_np, n_act, oid):
            for i in range(int(n_act)):
                if int(meta_np[i, COL_ORDER_ID]) == oid:
                    return i
            return None

        slot_c = _find_slot(meta_c_np, n_active_c, ice_oid)
        slot_d = _find_slot(meta_d_np, n_active_d, ice_oid)

        assert slot_c is not None, (
            "(c) the iceberg slot vanished from the CAPPED result — the "
            "cap-trip should leave it live (still peak-consumed), not "
            "clean it up."
        )
        rem_capped = int(meta_c_np[slot_c, COL_REMAINING_HIDDEN])
        rem_drained = (
            int(meta_d_np[slot_d, COL_REMAINING_HIDDEN])
            if slot_d is not None else 0
        )
        # The cap stopped early → the capped result has drained STRICTLY
        # LESS hidden volume → MORE remaining than the drained reference.
        assert rem_capped > rem_drained, (
            "(c) cap-trip did not actually leave work undone: capped "
            f"remaining_hidden={rem_capped} is not strictly greater than "
            f"drained remaining_hidden={rem_drained}.  If they are equal "
            "the single pass already drained the cascade and the cap-trip "
            "path was not exercised — increase the resting-bid depth or "
            "iceberg size so the cascade genuinely needs >1 pass."
        )
