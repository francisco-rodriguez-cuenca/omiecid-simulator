"""Locked behavioural tests for P3-PERF-REGEN-QUIESCENCE-1 — re-arm the
``regenerate_slices`` outer fixed-point on cross-slot peak consumption.

THE FIX UNDER CONTRACT (OQ-23, filed + ruled 2026-07-09): in
``IcebergManager.regenerate_slices`` (iceberg_manager.py:306-561 on
``main``) the outer fixed-point's ``dirty`` re-arm reflects only
SELF-OID re-consumption (the body re-checks slot *j*'s own OID at
:456-457 and ORs ``slice_re_consumed`` at :520).  If slot *j*'s regen
slice PARTIALLY RESTS while consuming an already-visited, lower-index,
opposite-side slot *k*'s visible peak, *j* does not set ``dirty`` and
the outer loop (:536-561) can exit with slot *k* left
``peak_consumed=True`` — violating the documented post-condition at
iceberg_manager.py:352-364, on which ``contract_book.py``'s
``no_fill_path`` P2-PERF-8 regen-skip (:721-730) RELIES.  The fix
re-arms the outer fixed-point on CROSS-slot peak consumption so the
post-condition holds on the OQ-23 gap shape too.  This is a BEHAVIOR
CHANGE on the gap shape (closes it) and must be byte-preserving on all
NON-gap shapes.

================================================================
CONSTRUCTOR DECISION (probe results recorded in sprint-status.md)
================================================================
The story card names two candidate OQ-23 constructors:

  * PRIMARY — the ICEwPI downward-drift variant (ICEwPI SELL with
    ``ppd = -0.20``).  REJECTED as the locked constructor: market-spec
    §13.6 (DA-PPD-2026-04) rules the ICEwPI sign convention HARD —
    "ICEwPI buy orders must have PPD < 0; ICEwPI sell orders must have
    PPD > 0" — so a downward-drifting SELL is not valid OMIE order
    flow.  (The simulator's ``validate_order`` does not currently
    inspect ``ppd`` sign, and the authoring-time probe confirmed the
    spec-invalid variant DOES mechanically construct the gap; it is
    excluded here so the locked contract never pins behaviour on
    inputs the domain forbids.)  Per the card's own fallback clause,
    the sign convention "is validated to forbid" the drift → fallback.

  * FALLBACK — the standard-iceberg (PPD=0) registration-dispatch
    constructor, LITERALLY as given in open-questions.md §OQ-23
    ("ICE SELL @60.00 peak 0.2 then ICE BUY @60.00 peak 0.6").
    Probed at authoring time: it SELF-HEALS (no violation) — the
    registration-consumed slot 0 is visited FIRST by the regen pass,
    and every subsequent cross is a FULL consumption that re-arms
    ``dirty`` via the self-OID check, so the fixed point drains.

  * LOCKED CONSTRUCTOR — a refinement of the registration-dispatch
    family using only spec-valid signs, sized so BOTH regen dispatches
    partially rest (the ``dirty``-invisibility discriminant):

      Order 1: ICEwPI BUY  k @ 60.20, ppd = -0.20 (valid: buy < 0),
               total 1.0 MW, peak 0.3 MW  -> slot 0; bid 0.3 @ 60.20.
      Order 2: ICE SELL    j @ 60.00, ppd = 0, total 1.0 MW,
               peak 0.4 MW -> ``_ice_path`` dispatches the peak, which
               crosses k's bid (trade 0.3 @ 60.20) and rests 0.1 ask
               @ 60.00; j registers at slot 1.  The post-registration
               ``regenerate_slices`` pass then:
                 slot 0 (k): peak consumed -> regen slice 2 = BUY 0.3
                   @ 60.20 + (-0.20)*1 = 60.00 -> crosses j's resting
                   0.1 ask -> trade 0.1, RESTS 0.2 bid @ 60.00
                   (partial rest -> ``slice_re_consumed`` False).
                 slot 1 (j): its ask was consumed by k's slice ->
                   regen slice 2 = SELL 0.4 @ 60.00 -> crosses k's
                   fresh 0.2 bid -> trade 0.2, RESTS 0.2 ask @ 60.00
                   (partial rest -> ``slice_re_consumed`` False).
               ``dirty`` is never set; the outer loop exits with slot
               0 (k) ``peak_consumed=True`` — the OQ-23 gap, produced
               by a REGEN-pass dispatch (slot 1) consuming an
               already-visited slot's peak.  Both orders pass
               ``validate_order`` and honour the §13.6 sign rule.

RED/GREEN EXPECTATION (recorded red-proof, orchestrator ruling 1):
  * (a)+(b) gap-closure/quiescence-surface tests are RED on pre-fix
    ``main`` (the constructor genuinely trips ``_quiescence_violations``
    — slot 0 left ``peak_consumed=True`` with 0.4 MW hidden) and turn
    GREEN only when the developer lands the cross-slot re-arm.
  * (c)/(d)/(e) pin PRESERVED behaviour and are GREEN pre-fix; the
    developer must NOT perturb them.

NON-GAP REAL-TAPE PARITY (orchestrator ruling 2): the Deliverable-2
cross-version real-tape evidence is executed by the DEVELOPER at
implementation time and is NOT locked here — real tapes may
legitimately diverge wherever a gap event fired.  This file locks only
the synthetic NON-gap parity pins (c) and provides
``per_submit_tuple_stream`` as the harness helper: it captures the
per-submit state-tuple stream so the FIRST diverging order of any
cross-version replay is locatable, plus the final per-contract tuple.

References:
  * docs/omiecid/agile-plan.md §"Story P3-PERF-REGEN-QUIESCENCE-1" (card).
  * docs/omiecid/open-questions.md §OQ-23 (mechanism, probe, Q1/Q2 rulings).
  * docs/omiecid/market-spec.md §4.2.1 Rule 4 + Scope clarification
    (the gap is an ENGINEERING invariant, NOT a Rule-4 breach — the
    resting book stays uncrossed), §13.6 (hard PPD sign rule).
  * tests/omiecid/test_perf_a1_fused_regen_gate.py (the LOCKED quiescence
    probe ``_quiescence_violations`` and byte-identity tuple helpers,
    imported below so the derivations cannot drift apart).
  * tests/omiecid/test_perf_a1_cascade_cap_trip.py (UNMODIFIED guard: the
    cap-trip -> sticky ``truncated`` wiring; cited by (d)).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob.jorderbook import OrderBook
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    _replay_orders,
    _row_to_order_traced,
)
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    _make_jaxob_cfg,
    _submit_order,
)
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_REMAINING_HIDDEN,
    COL_SIDE,
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

# Locked derivations imported from the LOCKED P2-PERF-8 file so the
# quiescence probe and the byte-identity tuple used here are the SAME
# derivations the existing invariant runs on (card Test-first contract
# (a)/(b)/(c); precedent for cross-test import:
# test_jit_marker_lint_audit.py:35).
from tests.omiecid.test_perf_a1_fused_regen_gate import (
    _aggregate_replay_buf,
    _byte_identity_tuple,
    _hash_array,
    _quiescence_violations,
    _replay_with_quiescence_check,
)

# This module drives the jitted ``_submit_order`` / ``_replay_orders``
# lifts and compiles ``regenerate_slices`` at two ``max_iterations``
# values in the (d) cap-binding probe — jit-marker discipline per T1.1.
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Fixture construction — all values on-tick and inside the OQ-01
# envelopes (±9,999 EUR/MWh, 0.1–999 MW); every constructor order is
# checked against ``validate_order`` (see TestOQ23GapClosure).
# ---------------------------------------------------------------------------

def _open_contract_37() -> Contract:
    """Mid-day OPEN quarter-hour contract (mirrors conftest contract 37)."""
    return Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


def _p(pc: OMIEProductConfig, price_eur: float) -> int:
    """Offset price encoding — mirrors ``encoding.encode_price``."""
    return int(round((price_eur - pc.price_min) / pc.price_tick))


def _q(pc: OMIEProductConfig, qty_mw: float) -> int:
    return int(round(qty_mw / pc.qty_tick))


def _ppd(pc: OMIEProductConfig, ppd_eur: float) -> int:
    """SIGNED ppd tick encoding (no offset) — mirrors ``encoding.encode_ppd``."""
    return int(round(ppd_eur / pc.price_tick))


def _order_row(
    pc: OMIEProductConfig,
    side: Side,
    price_eur: float,
    qty_mw: float,
    oid: int,
    ts: int,
    exec_r: ExecRestriction = ExecRestriction.NON,
    peak_mw: float = 0.0,
    ppd_eur: float = 0.0,
) -> list[int]:
    """One loader-layout ``Int32[13]`` order row (see ``_row_to_order_traced``)."""
    return [
        37, int(side), _p(pc, price_eur), _q(pc, qty_mw), oid, oid, ts,
        int(exec_r), int(Validity.GFS), 0, int(Status.ACTIVE),
        _q(pc, peak_mw), _ppd(pc, ppd_eur),
    ]


def _gap_constructor_orders(
    pc: OMIEProductConfig,
    oids: tuple[int, int] = (1, 2),
) -> jnp.ndarray:
    """The LOCKED OQ-23 gap constructor (spec-valid signs; see module
    docstring §CONSTRUCTOR DECISION for the full mechanism trace).

    Two orders; the gap fires inside order 2's ``_ice_path``
    post-registration ``regenerate_slices`` pass: slot 1's regen slice
    consumes already-visited slot 0's freshly-rested peak and partially
    rests, so neither slot sets ``dirty`` and the pre-fix outer loop
    exits with slot 0 ``peak_consumed=True`` (0.4 MW hidden stranded).

    LOCKED AMENDMENT (audit-failure ruling item 2, 2026-07-09): the gap
    shape is oid-INDEPENDENT, so the gap-closure contracts parametrize
    ``oids`` over BOTH a small pair (1, 2) and a realistic pair
    (1001, 1002 — far above every quantity in ticks).  The Step-5 audit
    proved the pre-existing defective gate at iceberg_manager.py:556
    compares the get_order ORDER-ID column against a quantity, so the
    small-oid fixture passed by luck; the contract must not be
    satisfiable by any oid-dependent implementation.
    """
    oid_k, oid_j = oids
    rows = [
        # Order 1 — iceberg k: ICEwPI BUY @ 60.20, ppd = -0.20 (§13.6
        # valid: buy drifts DOWN), total 1.0, peak 0.3 -> slot 0.
        _order_row(pc, Side.BUY, 60.20, 1.0, oid_k, 100,
                   ExecRestriction.ICE, peak_mw=0.3, ppd_eur=-0.20),
        # Order 2 — iceberg j: standard ICE SELL @ 60.00 (ppd 0),
        # total 1.0, peak 0.4 -> registration peak crosses k (partial
        # rest), then the regen pass constructs the gap.
        _order_row(pc, Side.SELL, 60.00, 1.0, oid_j, 101,
                   ExecRestriction.ICE, peak_mw=0.4),
    ]
    return jnp.array(rows, dtype=jnp.int32)


# Locked oid-magnitude parametrization for the OQ-23 gap-closure
# contracts (amendment above): small pair (the original fixture) and a
# realistic, validate_order-clean pair much larger than any slice qty
# in ticks (1.0 MW = 20 qty ticks << 1001).
GAP_OID_PAIRS = [
    pytest.param((1, 2), id="small-oids"),
    pytest.param((1001, 1002), id="realistic-oids"),
]


# FOLLOWUP-2 fixture-OID realism sweep (2026-07-09): the NON-gap parity
# fixtures below take an ``oid_base`` offset so their pinned matching
# outcomes (fill count, qty ticks, n_active, ts_counter, book_full) can
# be shown OID-magnitude-INDEPENDENT — a wrong-column (OID-vs-qty)
# implementation regression cannot be masked by tiny fixture OIDs.  The
# base pair mirrors GAP_OID_PAIRS: 0 (original small OIDs 1..5) and 1000
# (realistic OIDs 1001..1005, all far above every qty/peak in ticks —
# 1.0 MW = 10 ticks, 0.5 MW = 5 ticks).  Timestamps are UNCHANGED (the
# ts_counter pin derives from the fixed 100+k timestamps, not the OIDs).
NONGAP_OID_BASES = [
    pytest.param(0, id="small-oids"),
    pytest.param(1000, id="realistic-oids"),
]


def _forward_heal_orders(pc: OMIEProductConfig, oid_base: int = 0) -> jnp.ndarray:
    """NON-gap sibling of the gap constructor: FORWARD (higher-index)
    consumption that the ascending sequential pass heals SAME-pass.

    Identical to ``_gap_constructor_orders`` except order 2 is an
    ICEwPI SELL with ppd = +0.20 (§13.6 valid: sell drifts UP), so in
    the regen pass slot 0's slice still consumes slot 1's resting ask
    (forward consumption — slot 1 is visited LATER in the same pass and
    healed), but slot 1's own regen slice re-prices to 60.20 and rests
    WITHOUT crossing slot 0's fresh 60.00 bid.  Pass ends quiescent —
    no violation pre-fix, and the post-fix re-arm must observe nothing
    to chase (byte-identical outcome).
    """
    rows = [
        _order_row(pc, Side.BUY, 60.20, 1.0, oid_base + 1, 100,
                   ExecRestriction.ICE, peak_mw=0.3, ppd_eur=-0.20),
        _order_row(pc, Side.SELL, 60.00, 1.0, oid_base + 2, 101,
                   ExecRestriction.ICE, peak_mw=0.4, ppd_eur=+0.20),
    ]
    return jnp.array(rows, dtype=jnp.int32)


def _self_oid_cascade_orders(pc: OMIEProductConfig, oid_base: int = 0) -> jnp.ndarray:
    """NON-gap fixture: the canonical SELF-OID cascade (drains normally).

    Same shape as the LOCKED ``_cascade_synthetic_orders`` in
    test_perf_a1_fused_regen_gate.py: deep resting bids re-consume the
    ICE SELL's regenerated slices FULLY each pass, so the self-OID
    ``slice_re_consumed`` re-arm drives the fixed point to quiescence —
    the pre-fix path already handles this; the fix must not perturb it.
    """
    rows = [
        _order_row(pc, Side.BUY, 60.00, 0.2, oid_base + 1, 100),
        _order_row(pc, Side.BUY, 60.00, 0.2, oid_base + 2, 101),
        _order_row(pc, Side.BUY, 60.00, 0.2, oid_base + 3, 102),
        _order_row(pc, Side.SELL, 60.00, 1.0, oid_base + 4, 103,
                   ExecRestriction.ICE, peak_mw=0.2),
        _order_row(pc, Side.SELL, 70.00, 0.1, oid_base + 5, 104),
    ]
    return jnp.array(rows, dtype=jnp.int32)


def _no_iceberg_orders(pc: OMIEProductConfig, oid_base: int = 0) -> jnp.ndarray:
    """NON-gap fixture: iceberg-free NON/IOC/FOK matching sequence
    (``regenerate_slices`` has zero active slots on every call)."""
    rows = [
        _order_row(pc, Side.BUY, 60.00, 0.5, oid_base + 1, 100),
        _order_row(pc, Side.SELL, 60.00, 0.3, oid_base + 2, 101),   # trade 0.3
        _order_row(pc, Side.SELL, 61.00, 0.2, oid_base + 3, 102),   # rests
        _order_row(pc, Side.BUY, 59.00, 0.1, oid_base + 4, 103, ExecRestriction.IOC),
        _order_row(pc, Side.SELL, 60.00, 0.5, oid_base + 5, 104, ExecRestriction.FOK),
    ]
    return jnp.array(rows, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _fresh_book() -> ContractBook:
    return ContractBook(_open_contract_37(), OMIEProductConfig(), OMIEMarketConfig())


def _replay_final_state(
    orders: jnp.ndarray,
    pc: OMIEProductConfig,
    mc: OMIEMarketConfig,
) -> OMIEBookState:
    """Replay ``orders`` through the production ``_submit_order`` lift."""
    state = _fresh_book().init()
    for k in range(int(orders.shape[0])):
        order = _row_to_order_traced(orders[k], pc)
        state, _result = _submit_order(state, order, pc, mc)
    for leaf in jax.tree_util.tree_leaves(state):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return state


def per_submit_tuple_stream(
    init_state: OMIEBookState,
    orders: jnp.ndarray,
    pc: OMIEProductConfig,
    mc: OMIEMarketConfig,
) -> list[tuple]:
    """Gap-classifier harness helper (orchestrator ruling 2) — NOT a
    locked real-tape gate.

    Replays ``orders`` order-by-order through the production
    ``_submit_order`` lift and captures, AFTER EVERY submit, the
    state-derived identity tuple::

        (order_idx, accepted, traded,
         hash(lob.bids), hash(lob.asks), hash(lob.trades),
         hash(iceberg_meta), n_active, ts_counter, truncated, book_full)

    Running this on the SAME tape under two code versions and comparing
    element-wise locates the FIRST diverging order (divergence is
    guilty until proven gap-attributable: the developer then checks
    whether ``_quiescence_violations`` was non-empty at that order on
    the pre-fix version).  The final entry doubles as the per-contract
    final-state tuple.
    """
    stream: list[tuple] = []
    state = init_state
    for k in range(int(orders.shape[0])):
        order = _row_to_order_traced(orders[k], pc)
        state, result = _submit_order(state, order, pc, mc)
        for leaf in jax.tree_util.tree_leaves(state):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        stream.append((
            k,
            bool(result.accepted),
            bool(result.traded),
            _hash_array(state.lob.bids),
            _hash_array(state.lob.asks),
            _hash_array(state.lob.trades),
            _hash_array(state.iceberg_meta),
            int(state.n_active),
            int(state.ts_counter),
            bool(state.truncated),
            bool(state.book_full),
        ))
    return stream


# ===========================================================================
# (a) GAP-CLOSURE — the behavior change.  RED pre-fix, GREEN post-fix.
# ===========================================================================

class TestOQ23GapClosure:
    """Card contract (a): after the gap constructor's final
    ``submit_order`` returns, the ``regenerate_slices`` post-condition
    (iceberg_manager.py:352-364) HOLDS — ``_quiescence_violations`` is
    EMPTY.  Pre-fix this is the exact state the OQ-23 probe flags
    (slot 0 left ``peak_consumed=True``); the fix closes it.

    The stranded state is an engineering-invariant violation (OQ-23;
    not a Rule 4 breach — the resting book is uncrossed), but it is
    correctness-affecting because ``no_fill_path`` (contract_book.py
    :721-730) relies on the post-condition between aggressors.
    """

    @pytest.mark.parametrize("oids", GAP_OID_PAIRS)
    def test_gap_constructor_orders_are_valid_omie_inputs(self, oids):
        """Fixture hygiene (GREEN pre-fix): every constructor order
        passes ``validate_order`` on an OPEN contract (at BOTH oid
        magnitudes), and the ICEwPI ppd sign honours the market-spec
        §13.6 HARD sign rule (buy < 0; sell > 0; standard icebergs 0)."""
        pc = OMIEProductConfig()
        orders = _gap_constructor_orders(pc, oids)
        for k in range(int(orders.shape[0])):
            order = _row_to_order_traced(orders[k], pc)
            accepted = bool(validate_order(order, pc, jnp.bool_(True)))
            assert accepted, (
                f"gap-constructor order idx {k} (row "
                f"{np.asarray(orders[k]).tolist()}) must pass "
                f"validate_order — the locked OQ-23 constructor may only "
                f"use valid OMIE inputs (OQ-01 envelopes, on-tick)."
            )
        rows = np.asarray(orders)
        for k in range(rows.shape[0]):
            side, exec_r, ppd_int = int(rows[k, 1]), int(rows[k, 7]), int(rows[k, 12])
            if exec_r == int(ExecRestriction.ICE) and ppd_int != 0:
                if side == int(Side.BUY):
                    assert ppd_int < 0, (
                        f"order idx {k}: ICEwPI BUY must carry ppd < 0 "
                        f"(market-spec §13.6 hard sign rule); got {ppd_int}."
                    )
                else:
                    assert ppd_int > 0, (
                        f"order idx {k}: ICEwPI SELL must carry ppd > 0 "
                        f"(market-spec §13.6 hard sign rule); got {ppd_int}."
                    )

    @pytest.mark.parametrize("oids", GAP_OID_PAIRS)
    def test_gap_constructor_quiescent_after_final_submit(self, oids):
        """THE behavior-change contract (RED pre-fix): after order 2's
        ``submit_order`` returns, no active iceberg slot may be left
        ``peak_consumed=True`` — ``_quiescence_violations(state)`` is
        EMPTY.  Pre-fix the outer fixed-point exits ``dirty=False``
        with slot 0 stranded (visible peak consumed by slot 1's
        partially-resting regen slice; 0.4 MW hidden not re-exposed).
        Parametrized over oid magnitude so the contract cannot be
        satisfied by an oid-dependent implementation (locked amendment,
        audit-failure ruling item 2)."""
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        state = _replay_final_state(_gap_constructor_orders(pc, oids), pc, mc)

        violations = _quiescence_violations(state)
        assert violations == [], (
            "OQ-23 gap NOT closed: after the gap constructor's final "
            "submit_order an active iceberg slot is left "
            "peak_consumed=True (its OID absent from the resting side — "
            "get_order col-2 <= 0).  The regenerate_slices outer "
            "fixed-point exited without re-arming on CROSS-slot peak "
            "consumption, violating the post-condition at "
            "iceberg_manager.py:352-364 that no_fill_path "
            "(contract_book.py:721-730) relies on.  This is the "
            "engineering-invariant violation ruled in OQ-23 (not a "
            f"Rule 4 breach).\n  violating slots: {violations!r}"
        )

    @pytest.mark.parametrize("oids", GAP_OID_PAIRS)
    def test_gap_constructor_is_nonvacuous(self, oids):
        """Structural pins of the ``submit_order`` constructor: two live
        icebergs survive the session, the session trades 6 qty ticks in 3
        rows (OMIE-ICE-INCOMING-LADDER-1 row split), and slot 1's iceberg
        (oid ``oid_j``) rests a visible ask.

        OMIE-ICE-INCOMING-HOP-1 (S4) re-baseline: the fill count is 2, not
        3 (the incoming total sweeps k before registering), and this
        constructor no longer reaches the OQ-23 gap; see
        ``TestOQ23DirectRegenConstructor`` for the non-vacuous one.

        FOLLOWUP-2 fixture-OID realism sweep (2026-07-09): parametrized
        over ``GAP_OID_PAIRS`` (small + realistic) so the structural
        pins cannot be satisfied by an OID-magnitude-dependent
        implementation — the fill count / n_active / partial-rest pins
        must hold identically for OIDs far above every qty in ticks."""
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        _oid_k, oid_j = oids
        orders = _gap_constructor_orders(pc, oids)

        final_state, buf = _replay_orders(_fresh_book().init(), orders, pc, mc)
        n_fills, total_qty = _aggregate_replay_buf(buf)

        assert int(final_state.n_active) == 2, (
            f"gap constructor must leave BOTH icebergs live (n_active=2); "
            f"got {int(final_state.n_active)} — the fixture no longer "
            f"builds the OQ-23 two-slot shape."
        )
        # OMIE-ICE-INCOMING-HOP-1 (S4) re-baseline (card v5.195 Deliverable
        # (3)): order 2 now sweeps its total before registering, taking k's
        # slices 0.3 @ 60.20 and 0.3 @ 60.00 (2 fills / 6 ticks; was 3 / 6),
        # then rests R = 0.4. No regenerated slice trades, so through
        # ``submit_order`` this constructor NO LONGER reaches the cross-slot
        # re-arm. Non-vacuity of the OQ-23 gap now rests on the direct
        # ``regenerate_slices`` constructor (``TestOQ23DirectRegenConstructor``).
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): j's peak is 0.4, so k's
        # second fill (cumulative 0.3 -> 0.6) crosses j's instance boundary
        # 0.4 and prints as 0.1 + 0.2 (spec §4.2.1 Rule 2 item (2)): 3 fills,
        # still 6 qty ticks.
        assert (n_fills, total_qty) == (3, 6), (
            f"gap constructor must produce exactly 3 fills / 6 qty ticks "
            f"(the incoming total sweeps k's 0.3 @ 60.20 and 0.3 @ 60.00, the "
            f"second split 0.1 + 0.2 at j's instance boundary); "
            f"got ({n_fills}, {total_qty})."
        )
        # Slot 1's iceberg (oid 2, SELL) must be resting a visible ask:
        # its regen slice partially rested, which is what kept ``dirty``
        # off pre-fix.  (True pre- and post-fix.)
        ob = OrderBook(cfg=_make_jaxob_cfg(mc))
        meta = np.asarray(final_state.iceberg_meta)
        slot_j = next(
            i for i in range(int(final_state.n_active))
            if int(meta[i, COL_ORDER_ID]) == oid_j
        )
        lookup_side = 1 - int(meta[slot_j, COL_SIDE])
        row = np.asarray(
            ob.get_order(final_state.lob, jnp.int32(lookup_side), jnp.int32(oid_j))
        )
        assert int(row[2]) > 0, (
            f"iceberg j (oid {oid_j}) must be resting a visible "
            f"partial-rest surplus; get_order col-2 = {int(row[2])}."
        )


# ===========================================================================
# (a') OQ-23 reachability after OMIE-ICE-INCOMING-HOP-1 (S4) — the direct
#      ``regenerate_slices`` constructor (card v5.195 Deliverable (3)).
#      GREEN before and after S4 (no ``regenerate_slices`` change in scope).
# ===========================================================================

def _direct_regen_gap_inputs(oids: tuple[int, int]):
    """Resting icebergs only, unit level: an EMPTY book (``book.init().lob``)
    and a hand-built side table in which BOTH visible peaks are already
    consumed (``n_active = 2``, ``ts = 200``).

      slot 0, k: BUY ICEwPI, base 60.20, ppd -0.20, peak 0.3, hidden 0.7,
                 slice_count 0.
      slot 1, j: SELL ICE,   base 60.00, ppd  0,    peak 0.4, hidden 0.6,
                 slice_count 0.

    Pass 1: k's slice 1 is 0.3 @ 60.00 (60.20 - 0.20) and rests on the empty
    book; j's slice 1 (0.4 @ 60.00) takes it (0.3, passive k) and partially
    rests 0.1. Both slices partially rest, so the self-OID re-arm stays
    clean: only the CROSS-slot re-arm can re-visit k (the OQ-23 gap).
    """
    pc = OMIEProductConfig()
    mc = OMIEMarketConfig()
    oid_k, oid_j = oids
    base = _fresh_book().init()
    meta = np.array(base.iceberg_meta)
    meta[0] = [oid_k, _q(pc, 0.7), _q(pc, 0.3), _p(pc, 60.20), int(Side.BUY),
               oid_k, _ppd(pc, -0.20), 0]
    meta[1] = [oid_j, _q(pc, 0.6), _q(pc, 0.4), _p(pc, 60.00), int(Side.SELL),
               oid_j, 0, 0]
    mgr = IcebergManager(OrderBook(cfg=_make_jaxob_cfg(mc)), pc)
    return base, mgr, jnp.asarray(meta, jnp.int32), jnp.int32(2), jnp.int32(200)


def _live_rows(lob_side, oid: int) -> list[tuple[int, int]]:
    arr = np.asarray(lob_side)
    rows = arr[(arr[:, 2] == oid) & (arr[:, 1] > 0)]
    return [(int(r[0]), int(r[1])) for r in rows]


class TestOQ23DirectRegenConstructor:
    """OQ-23 reachability (card OMIE-ICE-INCOMING-HOP-1, Deliverable (3)).

    After S4 an incoming iceberg sweeps its total before it registers, so the
    ``submit_order`` gap constructor above no longer reaches the cross-slot
    re-arm in ``regenerate_slices``. This constructor reaches it directly,
    with resting icebergs only; the re-arm stays as defence in depth."""

    @pytest.mark.parametrize("oids", GAP_OID_PAIRS)
    def test_direct_regen_constructor_is_nonvacuous(self, oids):
        """With ``max_iterations=1`` the single pass strands slot 0 (k): the
        OQ-23 gap, reached only via a cross-slot peak consumption."""
        oid_k, _oid_j = oids
        base, mgr, meta, n_active, ts = _direct_regen_gap_inputs(oids)
        lob_c, meta_c, na_c, ts_c = mgr.regenerate_slices(
            base.lob, meta, n_active, ts, max_iterations=1,
        )
        state = base._replace(lob=lob_c, iceberg_meta=meta_c, n_active=na_c,
                              ts_counter=ts_c)
        violations = [(v["slot"], v["order_id"]) for v in _quiescence_violations(state)]
        assert violations == [(0, oid_k)], (
            f"the direct constructor must strand slot 0 (k) after one pass; "
            f"got {violations!r}")
        assert np.asarray(meta_c)[0].tolist() == [
            oid_k, 4, 3, 1005920, int(Side.BUY), oid_k, -20, 1]

    @pytest.mark.parametrize("oids", GAP_OID_PAIRS)
    def test_direct_regen_constructor_quiescent_with_default_cap(self, oids):
        """With the default cap the cross-slot re-arm re-visits k: one trade
        (0.3 @ 60.00, passive k, aggressor j); k ends at slice_count 2 showing
        0.3 @ 59.80 with 0.1 hidden; j shows 0.1 @ 60.00 with 0.2 hidden;
        ts = 203 (three slice dispatches)."""
        oid_k, oid_j = oids
        base, mgr, meta, n_active, ts = _direct_regen_gap_inputs(oids)
        lob_d, meta_d, na_d, ts_d = mgr.regenerate_slices(base.lob, meta, n_active, ts)
        state = base._replace(lob=lob_d, iceberg_meta=meta_d, n_active=na_d,
                              ts_counter=ts_d)
        assert _quiescence_violations(state) == []
        trades = np.asarray(lob_d.trades)
        used = trades[trades[:, 4] != -1]
        assert [(int(r[0]), abs(int(r[1])), int(r[2]), int(r[3])) for r in used] == [
            (1005900, 3, oid_k, oid_j)]
        m = np.asarray(meta_d)
        assert int(na_d) == 2
        assert m[0].tolist() == [oid_k, 1, 3, 1005920, int(Side.BUY), oid_k, -20, 2]
        assert m[1].tolist() == [oid_j, 2, 4, 1005900, int(Side.SELL), oid_j, 0, 1]
        assert _live_rows(lob_d.bids, oid_k) == [(1005880, 3)]
        assert _live_rows(lob_d.asks, oid_j) == [(1005900, 1)]
        assert int(ts_d) == 203


# ===========================================================================
# (b) Quiescence surface — the OQ-23 constructor joins the LOCKED
#     invariant's surface.  RED pre-fix, GREEN post-fix.
# ===========================================================================

class TestOQ23QuiescenceSurface:
    """Card contract (b): the LOCKED ``TestP2Perf8CascadeQuiescence``
    invariant (test_perf_a1_fused_regen_gate.py — which must stay GREEN
    unmodified) is made NON-VACUOUS by adding the OQ-23 constructor to
    the quiescence surface: a future regression re-opening the gap
    fails THIS test loudly, using the SAME per-submit
    ``_replay_with_quiescence_check`` derivation the locked invariant
    runs (imported, not copied)."""

    @pytest.mark.parametrize("oids", GAP_OID_PAIRS)
    def test_gap_constructor_quiescent_after_every_submit(self, oids):
        """Per-submit quiescence over the WHOLE gap-constructor session
        (RED pre-fix at submit index 1, the ICE registration whose
        regen pass constructs the cross-slot consumption).  Parametrized
        over oid magnitude (locked amendment, audit-failure ruling
        item 2)."""
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _gap_constructor_orders(pc, oids)

        bad_idx, violations = _replay_with_quiescence_check(
            _fresh_book().init(), orders, pc, mc,
        )
        assert bad_idx == -1, (
            "OQ-23 quiescence-surface regression: after submit_order #"
            f"{bad_idx} of the gap-constructor session an active iceberg "
            "slot is left peak_consumed=True — the cross-slot re-arm is "
            "absent or re-broken (engineering-invariant violation, OQ-23; "
            "not a Rule 4 breach — the resting book is uncrossed).\n"
            f"  violating slots: {violations!r}"
        )


# ===========================================================================
# (c) NON-gap matching preserved — fixed-expectation byte-parity pins.
#     GREEN pre-fix; the developer must NOT perturb them.
# ===========================================================================

class TestNonGapByteParityPins:
    """Card contract (c): the fix perturbs ONLY the gap outcome.  These
    three NON-gap fixtures pin the 9-tuple byte-identity derivation
    (imported from the LOCKED byte-identity anchor) with exact-value
    scalar pins computed on the PRE-fix code (2026-07-09 authoring run)
    plus a two-run hash-determinism check.  Any post-fix change to a
    pinned scalar or any cross-run hash instability is a parity break.

    Real-tape NON-gap parity is the DEVELOPER's Deliverable-2 evidence
    (see ``per_submit_tuple_stream``) — deliberately NOT locked here.
    """

    def _run_twice(self, orders: jnp.ndarray) -> tuple[tuple, tuple]:
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        tuples = []
        for _ in range(2):
            final_state, buf = _replay_orders(
                _fresh_book().init(), orders, pc, mc,
            )
            n_fills, total_qty = _aggregate_replay_buf(buf)
            tuples.append(_byte_identity_tuple(final_state, n_fills, total_qty))
        return tuples[0], tuples[1]

    def _assert_pins(self, tup: tuple, tup2: tuple, pins: dict, label: str):
        assert tup == tup2, (
            f"{label}: 9-tuple not deterministic across two identical "
            f"replays —\n  run1={tup!r}\n  run2={tup2!r}"
        )
        observed = {
            "n_fills": tup[0], "total_qty_ticks": tup[1],
            "n_active": tup[6], "ts_counter": tup[7], "book_full": tup[8],
        }
        assert observed == pins, (
            f"{label}: NON-gap fixed-expectation pin broken — the fix "
            f"must be byte-preserving on non-gap shapes.\n"
            f"  expected (pre-fix pins): {pins!r}\n  observed: {observed!r}"
        )

    @pytest.mark.parametrize("oid_base", NONGAP_OID_BASES)
    def test_self_oid_cascade_pinned(self, oid_base):
        """Self-OID cascade that drains normally: the existing
        ``slice_re_consumed`` re-arm already chases it; pinned pre-fix
        scalars must survive the fix unchanged.

        FOLLOWUP-2 sweep: parametrized over ``NONGAP_OID_BASES`` — the
        matching-outcome pins are OID-magnitude-independent, so the
        realistic-OID variant must produce the SAME pinned scalars."""
        pc = OMIEProductConfig()
        tup, tup2 = self._run_twice(_self_oid_cascade_orders(pc, oid_base))
        # OMIE-ICE-INCOMING-HOP-1 (S4) re-baseline: the incoming ICE SELL
        # sweeps its total 1.0 MW in ONE pass (three 0.2 rows, one per
        # resting bid) instead of peak + two regenerated slices, so the
        # counter advances two fewer steps: ts_counter 107 -> 105.
        self._assert_pins(tup, tup2, {
            "n_fills": 3, "total_qty_ticks": 6,
            "n_active": 1, "ts_counter": 105, "book_full": False,
        }, "self-OID cascade")

    @pytest.mark.parametrize("oid_base", NONGAP_OID_BASES)
    def test_no_iceberg_session_pinned(self, oid_base):
        """Iceberg-free NON/IOC/FOK sequence: ``regenerate_slices``
        scans zero slots; nothing for the re-arm to observe.

        FOLLOWUP-2 sweep: parametrized over ``NONGAP_OID_BASES`` — the
        matching-outcome pins must be OID-magnitude-independent."""
        pc = OMIEProductConfig()
        tup, tup2 = self._run_twice(_no_iceberg_orders(pc, oid_base))
        self._assert_pins(tup, tup2, {
            "n_fills": 1, "total_qty_ticks": 3,
            "n_active": 0, "ts_counter": 104, "book_full": False,
        }, "no-iceberg session")

    @pytest.mark.parametrize("oid_base", NONGAP_OID_BASES)
    def test_forward_heal_same_pass_pinned_and_quiescent(self, oid_base):
        """FORWARD (higher-index) consumption healed SAME-pass by the
        ascending scan: slot 0's regen slice consumes slot 1's peak,
        but slot 1 is visited later in the same pass and its ICEwPI
        slice re-prices upward and rests non-crossing.  Quiescent
        pre-fix (GREEN) — the re-arm must find nothing to chase and the
        pinned scalars must survive unchanged.

        FOLLOWUP-2 sweep: parametrized over ``NONGAP_OID_BASES`` — both
        the per-submit quiescence and the matching-outcome pins must be
        OID-magnitude-independent."""
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _forward_heal_orders(pc, oid_base)

        bad_idx, violations = _replay_with_quiescence_check(
            _fresh_book().init(), orders, pc, mc,
        )
        assert bad_idx == -1, (
            f"forward-heal fixture must be quiescent after every submit "
            f"(the sequential pass heals higher-index consumption "
            f"same-pass); violation after submit #{bad_idx}: {violations!r}"
        )

        tup, tup2 = self._run_twice(orders)
        # OMIE-ICE-INCOMING-HOP-1 (S4) re-baseline: order 2 (SELL ICEwPI
        # 1.0 / 0.4 @ 60.00) sweeps its total before registering: k's slice
        # 0.3 @ 60.20, then k's drifted slice 0.3 @ 60.00 (both k passive);
        # k's next slice (59.80) does not cross, so R = 0.4 registers.
        # total_qty_ticks 4 -> 6, ts_counter 104 -> 106.
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-A): order 2 is a chain of
        # 0.4 instances with limits 60.00, 60.20, ... (spec §4.2.1 Rule 2
        # item (6)). Instance 0 takes 0.3 @ 60.20 (k regenerates 0.3 @ 60.00)
        # and 0.1 @ 60.00; instance 1 (limit 60.20) cannot trade, so the sweep
        # stops: 2 fills / 4 ticks. R = 0.6 rests as instance 1 (0.4 @ 60.20,
        # uncrossed vs k's 0.2 @ 60.00), n_active 2. ts_counter: two fill
        # passes and one regenerated slice, the pass sequence of an IOC SELL
        # 0.4 @ 60.00 on the same book (105 on the unchanged IOC path), so 105.
        self._assert_pins(tup, tup2, {
            "n_fills": 2, "total_qty_ticks": 4,
            "n_active": 2, "ts_counter": 105, "book_full": False,
        }, "forward-heal same-pass")

    def test_per_submit_stream_helper_shape(self):
        """Smoke-pin for the developer's gap-classifier harness helper:
        one entry per submitted order, 11 fields per entry, entries
        deterministic across two identical replays (so cross-version
        comparison is well-defined)."""
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _forward_heal_orders(pc)

        s1 = per_submit_tuple_stream(_fresh_book().init(), orders, pc, mc)
        s2 = per_submit_tuple_stream(_fresh_book().init(), orders, pc, mc)
        assert len(s1) == int(orders.shape[0])
        assert all(len(entry) == 11 for entry in s1)
        assert s1 == s2, (
            "per_submit_tuple_stream must be deterministic on identical "
            "inputs — cross-version first-divergence location depends on it."
        )


# ===========================================================================
# (d) Truncation semantics preserved.  GREEN pre-fix.
# ===========================================================================

class TestTruncationSemanticsPreserved:
    """Card contract (d).  The cap-trip -> sticky ``truncated`` wiring
    itself is locked by the UNMODIFIED guard
    ``test_perf_a1_cascade_cap_trip.py`` (must-pass set).  Here we pin
    the two properties the re-arm could plausibly break:

      1. the ``max_iterations`` cap still BINDS (the re-arm predicate
         must stay inside ``dirty & (it < max_iters)`` — extra passes
         cannot loop past the cap), and
      2. the re-arm's extra passes on the gap shape are BOUNDED far
         below the cap (well-founded progress: each re-armed pass
         either strictly decreases some slot's remaining hidden volume
         or cleans a slot up), so a clearly-below-cap gap fixture
         terminates WITHOUT tripping ``truncated``.

    Orchestrator ruling 3: ``truncated`` is pinned ONLY on the
    clearly-below-cap fixture (a near-cap fixture may legitimately flip
    False->True post-fix and is deliberately not authored)."""

    @pytest.mark.slow  # slow: compiles the regenerate_slices lift at two max_iterations values (~2 s)
    @pytest.mark.parametrize("oid_base", [
        pytest.param(0, id="small-oids"),
        pytest.param(1000, id="realistic-oids"),
    ])
    def test_iteration_cap_still_binds_under_rearm(self, oid_base):
        """With ``max_iterations=1`` a multi-pass cascade must exit
        after its single pass and leave the leftover consumed slot —
        pre-fix AND post-fix (the re-arm may only OR into ``dirty``,
        never bypass the ``it < max_iters`` conjunct).  With the
        production default cap the SAME state drains to quiescence.

        FOLLOWUP-2 sweep: parametrized over ``oid_base`` — the
        ``consumed_slots`` probe reads the ``get_order`` col-2 sentinel
        (``-99`` when the OID is absent), so the cap-binding /
        drain-to-quiescence assertions are OID-magnitude-independent;
        the realistic-OID variant (ice OID 1050, bids 1100+k) confirms
        no wrong-column confusion is masked by the base OIDs."""
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        jaxob_cfg = _make_jaxob_cfg(mc)
        ob = OrderBook(cfg=jaxob_cfg)
        mgr = IcebergManager(ob, pc)

        # Mid-cascade state (mirrors the UNMODIFIED cap-trip guard's
        # ``_build_mid_cascade_state``): 3 deep resting bids @ 60.00
        # (0.2 MW each), an ICE SELL @ 60.00 total 1.0 / peak 0.2
        # registered, its visible peak dispatched and consumed.
        lob = ob.init()
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        for k in range(3):
            bid_oid = oid_base + 100 + k
            msg = jnp.array([
                1, 1, mgr._encode_qty_int(0.2), mgr._encode_price_int(60.00),
                bid_oid, bid_oid, k, 0,
            ], dtype=jnp.int32)
            lob = ob.process_order_array(lob, msg)
        ice_oid = oid_base + 50
        ice = OMIEOrder(
            contract_id=37, side=int(Side.SELL), price=60.00, quantity=1.0,
            order_id=ice_oid, agent_id=ice_oid, timestamp=10,
            exec_restriction=int(ExecRestriction.ICE),
            validity=int(Validity.GFS), expiry_time=0,
            status=int(Status.ACTIVE), peak_qty=0.2, ppd=0.0,
        )
        meta, n_active, _ovf = mgr.register(meta, n_active, ice)
        ts = mgr.update_timestamp(jnp.int32(3), ice.timestamp)
        peak_msg = jnp.array([
            1, -1, mgr._encode_qty_int(0.2), mgr._encode_price_int(60.00),
            ice_oid, ice_oid, int(ts), 0,
        ], dtype=jnp.int32)
        lob = ob.process_order_array(lob, peak_msg)

        def consumed_slots(lob_x, meta_x, n_active_x) -> list[int]:
            m = np.asarray(meta_x)
            out = []
            for i in range(int(n_active_x)):
                oid = int(m[i, COL_ORDER_ID])
                lookup = 1 - int(m[i, COL_SIDE])
                row = np.asarray(
                    ob.get_order(lob_x, jnp.int32(lookup), jnp.int32(oid))
                )
                if int(row[2]) <= 0:
                    out.append(i)
            return out

        assert consumed_slots(lob, meta, n_active) == [0], (
            "fixture self-check: the mid-cascade state must start with "
            "slot 0 peak_consumed=True"
        )

        # Capped at 1: the single primed pass dispatches slice 2, which
        # the deep bids re-consume; the outer loop must NOT run again.
        lob_c, meta_c, na_c, _ts_c = mgr.regenerate_slices(
            lob, meta, n_active, ts, max_iterations=1,
        )
        assert consumed_slots(lob_c, meta_c, na_c) == [0], (
            "max_iterations=1 must still BIND under the re-arm: the "
            "multi-pass cascade cannot drain in one pass, so slot 0 must "
            "remain peak_consumed=True at the capped exit (graceful "
            "cap-trip; the sticky truncated wiring is locked by "
            "test_perf_a1_cascade_cap_trip.py)."
        )

        # Production default cap: the same state drains to quiescence.
        lob_d, meta_d, na_d, _ts_d = mgr.regenerate_slices(
            lob, meta, n_active, ts,
        )
        assert consumed_slots(lob_d, meta_d, na_d) == [], (
            "the production-default cap must fully drain the 3-hop "
            "cascade to quiescence (fixed-point termination)."
        )

    @pytest.mark.parametrize("oids", GAP_OID_PAIRS)
    def test_gap_fixture_terminates_below_cap_without_truncation(self, oids):
        """Boundedness on the gap shape (clearly below cap): the
        2-order gap constructor replays to completion with
        ``truncated`` False — pre-fix (no extra passes) AND post-fix
        (the re-arm adds a bounded number of healing passes, nowhere
        near ``max_match_iterations=200``).

        FOLLOWUP-2 sweep: parametrized over ``GAP_OID_PAIRS`` — the
        ``truncated`` boundedness pin must be OID-magnitude-independent
        (the pass count depends on the cascade shape, not the OIDs)."""
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        state = _replay_final_state(_gap_constructor_orders(pc, oids), pc, mc)
        assert not bool(state.truncated), (
            "gap-constructor replay must NOT trip the sticky truncated "
            "flag: the fixture is clearly below the 200-pass cap, so the "
            "re-arm's extra passes must terminate by well-founded "
            "progress without reaching the truncation backstop."
        )


# ===========================================================================
# (e) One-compile invariant preserved.  GREEN pre-fix.
# ===========================================================================

class TestOneCompilePreserved:
    """Card contract (e): the rewrite adds no ``_submit_order`` cache
    entry.  Asserted as an INTRA-test cache-size DELTA (warm the exact
    shapes, snapshot, re-run, assert zero growth) — this is immune to
    the known allowlisted one-compile xdist flake
    (test_jit_cache_defrag.py ``KNOWN_XDIST_FLAKES`` /
    test-perf-gate.md row 129), which is a race on the ABSOLUTE
    cache-count observed across tests, not on within-test growth."""

    def test_no_new_submit_order_cache_entry_on_gap_session(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        if not hasattr(_submit_order, "_cache_size"):
            pytest.skip(
                "_submit_order does not expose `_cache_size`; trace-count "
                "probe unavailable on this JAX version."
            )
        orders = _gap_constructor_orders(pc)

        # Warm pass: compiles (or reuses) the single _submit_order entry
        # for this state/order pytree shape.
        _replay_final_state(orders, pc, mc)
        cache_before = _submit_order._cache_size()

        # Second pass over the same shapes: the gap session (icebergs +
        # regen cascade + the re-armed passes post-fix) must not retrace.
        _replay_final_state(orders, pc, mc)
        cache_after = _submit_order._cache_size()

        assert cache_after == cache_before, (
            f"_submit_order retraced on the warm gap session: cache grew "
            f"{cache_before} -> {cache_after}.  The re-arm must live "
            f"inside the existing jitted body (outer-loop predicate "
            f"only); a new cache entry means the fix changed the traced "
            f"signature or specialised on a value."
        )
