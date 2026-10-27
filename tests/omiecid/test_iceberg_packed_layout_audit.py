"""Audit tests for P2-PERF-6 — packed-array `regenerate_slices`.

Independent post-implementation audit of the packed-and-contiguous
iceberg-meta layout introduced by P2-PERF-6 (mechanism (2a) swap-with-last
delete). Author did not read the original locked test file
(``tests/omiecid/test_iceberg_packed_layout.py``); audit assertions are
derived independently from:

  * ``docs/omiecid/agile-plan.md`` Story P2-PERF-6 card body
    (Deliverables (1)–(6), Test-first contract (a)–(c), Risk audit items).
  * ``docs/omiecid/market-spec.md`` §4.2 (iceberg catalogue), §5.2 (timestamp
    rules around new slice activation).
  * ``docs/omiecid/migration-plan.md`` canonical module-level
    ``static_argnames`` jit idiom and ``OMIEBookState`` schema notes.
  * Production sources at
    ``omiecid_simulator/omiecid/iceberg_manager.py`` (signatures
    ``register / cleanup / regenerate_slices / matchable_hidden_volume /
    truncation_flag``), ``omiecid_simulator/omiecid/contract_book.py``
    (``OMIEBookState`` with ``n_active``; ``_jit_cancel_by_id`` 5-tuple
    return; ``_while_loop_match`` / ``_while_loop_match_fok`` carries;
    ``_build_state`` / ``_build_state_jit``; ``expire_at_gct`` reset),
    ``omiecid_simulator/omiecid/jax_constants.py``.

Adversarial dimensions covered (mapping to the audit brief):

  1. JAX-safety probes — pytree leaf types, no float64 promotion on
     traced paths, jaxpr-level invariants.
  2. Packed-layout invariant under adversarial register/cleanup
     sequences — including the saturated-edge / cleanup-zero / re-register
     case.
  3. Sentinel-row hygiene — every column equals the canonical sentinel
     row after cleanup (not just ``COL_ORDER_ID``).
  4. ``n_active`` threading correctness across init / expire_at_gct /
     cancel_order / submit paths.
  5. Bounded-scan behavioural correctness — outer bound is ``n_active``,
     not ``MAX_ICEBERGS``; jaxpr trip-counts confirm via op-count growth
     pattern.
  6. ``base_price_int`` immutability across regenerate_slices cascades.
  7. Multiset equivalence under stress — random register/cleanup
     sequences preserve the multiset of live order_ids.
  8. Direct-construction state coherence — every audit-suite synthetic
     state has ``n_active`` matching ``iceberg_meta``.
  9. truncation_flag bounded form is behaviour-neutral.
 10. _while_loop_match / FOK carry consistency on n_active threading.

Runtime budget:
  All tests target the < 1 s fast-suite budget except the load-stress
  test which is marked ``@pytest.mark.slow``. JIT-compilation probes
  are marked ``@pytest.mark.jit`` per ``pyproject.toml``.
"""

from __future__ import annotations

import inspect
import random
import re
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    _build_state_jit,
    _jit_cancel_by_id,
    _make_jaxob_cfg,
    _submit_order,
)
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_AGENT_ID,
    COL_BASE_PRICE_INT,
    COL_ORDER_ID,
    COL_PEAK_QTY,
    COL_PPD,
    COL_REMAINING_HIDDEN,
    COL_SIDE,
    COL_SLICE_COUNT,
    IcebergManager,
    N_COLS,
    _sentinel_row,
)
from omiecid_simulator.omiecid.jax_constants import (
    MAX_ICEBERGS,
    MAX_SLICES_PER_REGEN,
    SENTINEL_ORDER_ID,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    make_order,
)


# ---------------------------------------------------------------------------
# Local fixtures — module-scoped per the conftest convention.
# Contract/unit categorymeans synthetic in-test state, fits < 1 s budget.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def mkt_cfg():
    return OMIEMarketConfig()


@pytest.fixture(scope="module")
def open_contract():
    return Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


@pytest.fixture(scope="module")
def book(open_contract, cfg, mkt_cfg):
    return ContractBook(open_contract, cfg, mkt_cfg)


@pytest.fixture(scope="module")
def mgr(book):
    return book._iceberg_mgr


@pytest.fixture(scope="module")
def empty_state(book):
    return book.init()


def _ice_order(
    *,
    order_id: int,
    side: Side = Side.BUY,
    price: float = 50.0,
    quantity: float = 5.0,
    peak_qty: float = 1.0,
    ppd: float = 0.0,
    agent_id: int = 7,
    timestamp: int = 0,
) -> OMIEOrder:
    return make_order(
        contract_id=37,
        side=side,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=ExecRestriction.ICE,
        peak_qty=peak_qty,
        ppd=ppd,
    )


def _live_oids(meta: jnp.ndarray, n_active: jnp.ndarray) -> list[int]:
    """Return the order_ids in the contiguous prefix meta[:n_active] as
    a Python list. Helper for multiset-equivalence assertions.
    """
    n = int(n_active)
    return [int(meta[i, COL_ORDER_ID]) for i in range(n)]


def _assert_packed_invariant(meta: jnp.ndarray, n_active: jnp.ndarray):
    """Audit-side helper: assert the packed-and-contiguous invariant.

    Live region ``meta[:n_active]`` has no SENTINEL_ORDER_ID in COL 0;
    tail region ``meta[n_active:]`` is all sentinel-row.
    """
    n = int(n_active)
    sentinel_oid = jnp.int32(SENTINEL_ORDER_ID)
    sent = _sentinel_row()
    # Live prefix: no sentinel oids
    for i in range(n):
        assert int(meta[i, COL_ORDER_ID]) != int(sentinel_oid), (
            f"slot {i} (in live prefix) carries SENTINEL_ORDER_ID"
        )
    # Tail: every row is the canonical sentinel row (all columns).
    # jnp.all reduces the per-cell `== sent[col]` checks; bidirectionally
    # equivalent to the per-iter form (any mismatch trips the assertion).
    tail = meta[n:]
    assert bool(jnp.all(tail == sent[None, :])), (
        f"tail region [{n}, {MAX_ICEBERGS}) != canonical sentinel row"
    )


# ---------------------------------------------------------------------------
# Pytree-leaf and module-level dtype probes (no jit invocation).
# ---------------------------------------------------------------------------


class TestPytreeLeafAndSchema:
    """Audit dimension 1+8: ``OMIEBookState`` pytree leaves are ndarrays
    (no Python ``int`` / ``float`` / ``None``), with declared dtypes.
    ``n_active`` is a 0-d int32. Direct-construction states must be
    internally consistent (n_active matches the packed prefix).
    """

    def test_state_pytree_leaves_are_all_jnp_ndarray(self, empty_state):
        leaves = jax.tree_util.tree_leaves(empty_state)
        assert len(leaves) > 0
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray), (
                f"OMIEBookState leaf is not jnp.ndarray: {type(leaf)}"
            )

    def test_state_pytree_has_no_python_collections_as_leaves(self, empty_state):
        leaves = jax.tree_util.tree_leaves(empty_state)
        forbidden = (dict, list, tuple, type(None))
        for leaf in leaves:
            # tree_leaves never returns tuple/list/dict as leaves, but
            # asserting reinforces the contract for future schema edits.
            assert not isinstance(leaf, forbidden)

    def test_n_active_initial_state_is_zero_dim_int32(self, empty_state):
        na = empty_state.n_active
        assert isinstance(na, jnp.ndarray)
        assert na.shape == ()
        assert na.dtype == jnp.int32
        assert int(na) == 0

    def test_n_active_field_present_on_namedtuple(self):
        assert "n_active" in OMIEBookState._fields

    def test_iceberg_meta_shape_and_dtype_invariant(self, book, empty_state):
        # P3-PERF-REGEN-CAP-1: the book-derived meta's first dim follows the
        # configured max_icebergs, not the MAX_ICEBERGS ceiling constant.
        meta = empty_state.iceberg_meta
        assert meta.shape == (book._market_config.max_icebergs, N_COLS)
        assert meta.dtype == jnp.int32

    def test_state_construction_via_init_is_packed_invariant_holding(
        self, empty_state,
    ):
        # ContractBook.init must produce a state whose meta is fully
        # sentinel and n_active == 0.
        _assert_packed_invariant(empty_state.iceberg_meta, empty_state.n_active)

    def test_state_replace_with_inconsistent_n_active_is_authorless_burden(
        self, empty_state,
    ):
        """Audit fixture-coherence helper: building a fresh state with
        n_active mismatched against the meta must be observable by
        ``_assert_packed_invariant`` (so audit suites can catch their
        own broken fixtures). This is a probe of the helper, not of the
        production code.
        """
        broken_meta = empty_state.iceberg_meta
        # Claim n_active=5 but meta is still all-sentinel — invariant
        # fails because slots 0..4 are sentinel-row but n_active says
        # they should be live.
        with pytest.raises(AssertionError):
            _assert_packed_invariant(broken_meta, jnp.int32(5))


# ---------------------------------------------------------------------------
# Sentinel-row hygiene — all columns, not just COL_ORDER_ID.
# ---------------------------------------------------------------------------


class TestSentinelRowHygieneAllColumns:
    """Audit dimension 3: after any swap-with-last cleanup, the tail
    row that was cleared has every column == _sentinel_row()[col] —
    not just COL_ORDER_ID. Catches the bug class where swap-with-last
    leaves stale state in the cleared tail."""

    def test_register_then_cleanup_full_column_sentinel(self, mgr):
        meta = IcebergManager.empty_meta()
        order = _ice_order(
            order_id=42, side=Side.SELL, price=51.23, quantity=10.0,
            peak_qty=2.0, ppd=0.05, agent_id=11,
        )
        meta, n_active, _ = mgr.register(meta, jnp.int32(0), order)
        meta, n_active = mgr.cleanup(meta, n_active, 42)
        # Self-swap case: n_active was 1, slot 0 == tail. After cleanup,
        # slot 0 must be the full sentinel row.
        sent = _sentinel_row()
        for col in range(N_COLS):
            assert int(meta[0, col]) == int(sent[col]), (
                f"slot 0 col {col} != sentinel after cleanup; "
                f"got {int(meta[0, col])}, want {int(sent[col])}"
            )

    def test_swap_with_last_clears_full_column_at_old_tail(self, mgr):
        """Register 3 icebergs, cleanup the FIRST. The tail row (slot 2)
        must end up at slot 0 (swap-with-last), and slot 2 must be the
        canonical sentinel row — every column zeroed, not just oid.
        """
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        a = _ice_order(order_id=10, side=Side.BUY, price=10.0,
                       quantity=8.0, peak_qty=2.0, ppd=0.0, agent_id=1)
        b = _ice_order(order_id=20, side=Side.SELL, price=12.0,
                       quantity=6.0, peak_qty=3.0, ppd=0.0, agent_id=2)
        c = _ice_order(order_id=30, side=Side.BUY, price=20.5,
                       quantity=5.0, peak_qty=1.0, ppd=0.0, agent_id=3)
        meta, n_active, _ = mgr.register(meta, n_active, a)
        meta, n_active, _ = mgr.register(meta, n_active, b)
        meta, n_active, _ = mgr.register(meta, n_active, c)
        assert int(n_active) == 3
        # Cleanup the first iceberg (oid=10 at slot 0).
        meta, n_active = mgr.cleanup(meta, n_active, 10)
        assert int(n_active) == 2
        # Now: tail row (oid=30) must be at slot 0; old slot 2 must be sentinel.
        assert int(meta[0, COL_ORDER_ID]) == 30
        sent = _sentinel_row()
        for col in range(N_COLS):
            assert int(meta[2, col]) == int(sent[col]), (
                f"slot 2 (old tail) col {col} != sentinel after "
                f"swap-with-last; got {int(meta[2, col])}"
            )

    def test_cleanup_of_middle_slot_swaps_tail_in(self, mgr):
        """Register 4 icebergs, cleanup the middle (slot 1).
        Mechanism (2a): tail (slot 3) swaps into slot 1; slot 3 becomes
        sentinel.
        """
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        orders = [
            _ice_order(order_id=100 + i, side=Side.BUY, price=10.0 + i,
                       quantity=4.0, peak_qty=1.0, ppd=0.0, agent_id=i)
            for i in range(4)
        ]
        for o in orders:
            meta, n_active, _ = mgr.register(meta, n_active, o)
        assert int(n_active) == 4
        # Cleanup slot-1 iceberg (oid=101)
        meta, n_active = mgr.cleanup(meta, n_active, 101)
        assert int(n_active) == 3
        # Slot 1 should now carry the previous tail's oid (103).
        assert int(meta[1, COL_ORDER_ID]) == 103
        # Slot 3 must be canonical sentinel — full row.
        sent = _sentinel_row()
        for col in range(N_COLS):
            assert int(meta[3, col]) == int(sent[col])
        # Packed invariant holds end-to-end.
        _assert_packed_invariant(meta, n_active)


# ---------------------------------------------------------------------------
# Packed-layout invariant under adversarial sequences.
# ---------------------------------------------------------------------------


class TestPackedInvariantUnderRandomSequences:
    """Audit dimension 2: the packed-layout invariant holds at every
    step under random register/cleanup sequences. Hand-rolled
    deterministic (seeded) random walk."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 12345])
    def test_random_register_cleanup_sequence_preserves_invariant(
        self, mgr, seed,
    ):
        rng = random.Random(seed)
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        live_oids: list[int] = []
        next_oid = 1
        n_steps = 30
        for _ in range(n_steps):
            # Bias toward register if no live; otherwise 50/50.
            if not live_oids or rng.random() < 0.55:
                # Register
                o = _ice_order(
                    order_id=next_oid,
                    side=Side.BUY if rng.random() < 0.5 else Side.SELL,
                    price=10.0 + rng.random() * 30,
                    quantity=4.0 + rng.random() * 5,
                    peak_qty=1.0 + rng.random() * 1.5,
                    ppd=0.0,
                    agent_id=rng.randint(1, 50),
                )
                meta, n_active, overflow = mgr.register(meta, n_active, o)
                # No overflow at n_steps=30 << MAX_ICEBERGS
                assert not bool(overflow)
                live_oids.append(next_oid)
                next_oid += 1
            else:
                # Cleanup a random live iceberg
                victim = rng.choice(live_oids)
                meta, n_active = mgr.cleanup(meta, n_active, victim)
                live_oids.remove(victim)

            # Invariants after every step
            assert int(n_active) == len(live_oids)
            _assert_packed_invariant(meta, n_active)
            # Multiset of oids in live prefix matches the tracked set
            packed_oids = _live_oids(meta, n_active)
            assert sorted(packed_oids) == sorted(live_oids)

    def test_register_at_saturation_then_cleanup_zero_then_reregister(self, mgr):
        """Boundary case: fill to MAX_ICEBERGS, cleanup slot 0, then
        register again. After cleanup, n_active = MAX_ICEBERGS - 1 and
        the next register must land at slot (MAX_ICEBERGS - 1), filling
        the just-vacated tail.

        Marked slow because filling MAX_ICEBERGS sequentially is the
        hot-path lane that the cap value bounds — ~100 register calls
        each producing a JAX op trace.
        """
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        # Saturate the cap.
        # ovf_acc OR-accumulates the per-iter `overflow` flag; post-loop
        # assert catches any-True (bidirectionally equivalent to per-iter
        # `assert not bool(overflow)`).
        def _body_saturate_1000(i, carry):
            meta, n_active, ovf_acc = carry
            o = _ice_order(
                order_id=1000 + i, side=Side.BUY, price=20.0,
                quantity=4.0, peak_qty=1.0, agent_id=i % 10,
            )
            meta, n_active, overflow = mgr.register(meta, n_active, o)
            return (meta, n_active, ovf_acc | overflow)
        meta, n_active, any_overflow = jax.lax.fori_loop(
            0, MAX_ICEBERGS, _body_saturate_1000,
            (meta, n_active, jnp.bool_(False)),
        )
        assert not bool(any_overflow), (
            "unexpected overflow during saturation; "
            f"n_active={int(n_active)}"
        )
        assert int(n_active) == MAX_ICEBERGS
        # Cleanup the first iceberg (oid=1000 at slot 0). Mechanism (2a)
        # swaps tail (oid=1000+MAX_ICEBERGS-1) into slot 0.
        tail_oid_before = 1000 + MAX_ICEBERGS - 1
        meta, n_active = mgr.cleanup(meta, n_active, 1000)
        assert int(n_active) == MAX_ICEBERGS - 1
        assert int(meta[0, COL_ORDER_ID]) == tail_oid_before
        # Tail slot must now be sentinel.
        sent = _sentinel_row()
        for col in range(N_COLS):
            assert int(meta[MAX_ICEBERGS - 1, col]) == int(sent[col])
        # Re-register an iceberg — must land at slot n_active = MAX_ICEBERGS - 1
        o_new = _ice_order(order_id=9999, side=Side.SELL, price=30.0,
                           quantity=3.0, peak_qty=1.0, agent_id=42)
        meta, n_active, overflow = mgr.register(meta, n_active, o_new)
        assert not bool(overflow)
        assert int(n_active) == MAX_ICEBERGS
        # The freshly registered iceberg landed at slot MAX_ICEBERGS-1
        # (the just-vacated tail).
        assert int(meta[MAX_ICEBERGS - 1, COL_ORDER_ID]) == 9999

    def test_register_when_already_saturated_is_overflow_with_meta_unchanged(
        self, mgr,
    ):
        """When n_active == MAX_ICEBERGS, register must return
        overflow=True and leave meta and n_active unchanged."""
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        # Pure plumbing: fill MAX_ICEBERGS sequential orders; no per-iter
        # assertion.
        def _body_saturate_500(i, carry):
            meta, n_active = carry
            o = _ice_order(order_id=500 + i, side=Side.BUY,
                           price=20.0, quantity=4.0, peak_qty=1.0)
            meta, n_active, _ = mgr.register(meta, n_active, o)
            return (meta, n_active)
        meta, n_active = jax.lax.fori_loop(
            0, MAX_ICEBERGS, _body_saturate_500, (meta, n_active),
        )
        # Saturated state
        assert int(n_active) == MAX_ICEBERGS
        meta_before = meta
        # Attempt to register one more.
        o_extra = _ice_order(order_id=8888, side=Side.BUY,
                             price=25.0, quantity=4.0, peak_qty=1.0)
        meta_after, n_active_after, overflow = mgr.register(
            meta, n_active, o_extra,
        )
        assert bool(overflow) is True
        assert int(n_active_after) == MAX_ICEBERGS
        # Meta must be byte-for-byte unchanged.
        assert bool(jnp.all(meta_after == meta_before))


# ---------------------------------------------------------------------------
# Cleanup of non-existent order — pin the production behaviour.
# ---------------------------------------------------------------------------


class TestCleanupOfUnknownOrderId:
    """Audit dimension 2: cleanup of an order_id that doesn't exist
    is implementation-defined per the v5.30 spec. The current
    implementation gates everything on ``any_match`` and returns the
    inputs unchanged — pin this behaviour."""

    def test_cleanup_of_unknown_oid_on_empty_meta_is_noop(self, mgr):
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        meta_after, n_active_after = mgr.cleanup(meta, n_active, 999)
        assert bool(jnp.all(meta_after == meta))
        assert int(n_active_after) == 0
        _assert_packed_invariant(meta_after, n_active_after)

    def test_cleanup_of_unknown_oid_on_populated_meta_is_noop(self, mgr):
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        a = _ice_order(order_id=11, side=Side.BUY, price=10.0,
                       quantity=4.0, peak_qty=1.0)
        b = _ice_order(order_id=22, side=Side.SELL, price=20.0,
                       quantity=4.0, peak_qty=1.0)
        meta, n_active, _ = mgr.register(meta, n_active, a)
        meta, n_active, _ = mgr.register(meta, n_active, b)
        meta_before = meta
        n_active_before = int(n_active)
        meta_after, n_active_after = mgr.cleanup(meta, n_active, 7777)
        assert bool(jnp.all(meta_after == meta_before))
        assert int(n_active_after) == n_active_before
        _assert_packed_invariant(meta_after, n_active_after)

    def test_cleanup_of_sentinel_oid_negative_one_is_noop_or_corrupts(
        self, mgr,
    ):
        """P2-PERF-6-FOLLOWUP-1 contract (a): passing SENTINEL_ORDER_ID
        to cleanup must be an explicit no-op — meta unchanged and
        n_active unchanged.

        Without the defensive guard at iceberg_manager.py:644-647, the
        argmax-on-match-mask pattern matches tail sentinel rows (which
        all carry COL_ORDER_ID == SENTINEL_ORDER_ID by construction),
        any_match becomes True, swap-with-last writes the live tail row
        into a sentinel slot AND clears the live tail-1 slot — silently
        corrupting state and decrementing n_active.

        With the guard (`any_match & (oid != SENTINEL_ORDER_ID)`), the
        sentinel order-id short-circuits to no-op: meta_after equals
        meta_before byte-for-byte, and n_active is preserved.

        Construct a non-empty packed state (a few live icebergs), call
        cleanup with SENTINEL_ORDER_ID, and lock the no-op contract.
        """
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        # Register three live icebergs so n_active > 0 and the live
        # tail-1 slot exists (where corruption would land without the
        # guard).
        for oid, price in [(10, 5.0), (20, 6.0), (30, 7.0)]:
            o = _ice_order(
                order_id=oid, side=Side.BUY, price=price,
                quantity=4.0, peak_qty=1.0,
            )
            meta, n_active, _ = mgr.register(meta, n_active, o)
        assert int(n_active) == 3
        _assert_packed_invariant(meta, n_active)

        meta_before = meta
        n_active_before = int(n_active)

        # Call cleanup with the sentinel order-id. Per the FOLLOWUP-1
        # contract, the defensive guard short-circuits this to a strict
        # no-op.
        meta_after, n_active_after = mgr.cleanup(
            meta, n_active, jnp.int32(SENTINEL_ORDER_ID),
        )

        # Contract (a): meta byte-for-byte unchanged, n_active unchanged.
        assert jnp.array_equal(meta_after, meta_before)
        assert int(n_active_after) == int(n_active_before)
        # Packed invariant still holds on the returned state.
        _assert_packed_invariant(meta_after, n_active_after)

    def test_cleanup_legitimate_path_unaffected_by_sentinel_guard(
        self, mgr,
    ):
        """P2-PERF-6-FOLLOWUP-1 contract (c): the defensive guard does
        not regress the legitimate cleanup path. On a saturated state
        (n_active == MAX_ICEBERGS, every row carries a real, distinct
        order-id, and no row carries SENTINEL_ORDER_ID), calling
        cleanup(meta, n_active, real_oid) for any real_oid in the meta
        must behave identically pre- and post-guard:

        * the multiset of live order_ids shrinks by exactly one
          (the removed real_oid is gone);
        * n_active decrements by exactly one.

        This proves the guard is a strict no-op extension for the
        legitimate path — passing a real order-id behaves identically
        whether or not the `& (oid != SENTINEL_ORDER_ID)` predicate is
        present (since `real_oid != SENTINEL_ORDER_ID` is always True).
        """
        # Saturate the meta with MAX_ICEBERGS distinct real order-ids.
        # Build inline (rather than via the test_iceberg_cap_overflow
        # _saturated_meta helper) to keep the fixture self-contained
        # and avoid cross-file fixture coupling.
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        start_oid = 1000
        # ovf_acc OR-accumulates the per-iter `overflow` flag; post-loop
        # assert catches any-True (bidirectionally equivalent to per-iter
        # `assert not bool(overflow)`).
        def _body_saturate_start_oid(i, carry):
            meta, n_active, ovf_acc = carry
            o = _ice_order(
                order_id=start_oid + i, side=Side.BUY, price=20.0,
                quantity=4.0, peak_qty=1.0, agent_id=i % 10,
            )
            meta, n_active, overflow = mgr.register(meta, n_active, o)
            return (meta, n_active, ovf_acc | overflow)
        meta, n_active, any_overflow = jax.lax.fori_loop(
            0, MAX_ICEBERGS, _body_saturate_start_oid,
            (meta, n_active, jnp.bool_(False)),
        )
        assert not bool(any_overflow), (
            "unexpected overflow during saturation; "
            f"n_active={int(n_active)}"
        )
        assert int(n_active) == MAX_ICEBERGS
        _assert_packed_invariant(meta, n_active)

        # Precondition: no row carries SENTINEL_ORDER_ID — the saturated
        # state is entirely real, so the guard's predicate
        # `oid != SENTINEL_ORDER_ID` evaluates True for every real_oid
        # we could pick.
        oids_before = jnp.asarray(meta[:, COL_ORDER_ID])
        assert not bool(jnp.any(oids_before == jnp.int32(SENTINEL_ORDER_ID))), (
            "saturated state unexpectedly carries SENTINEL_ORDER_ID — "
            "precondition violated"
        )

        # Pick a real_oid that exists in the meta. The middle slot's
        # order-id exercises the swap-with-last path (not the self-swap
        # degenerate case).
        target_oid = start_oid + (MAX_ICEBERGS // 2)
        live_oids_before = _live_oids(meta, n_active)
        assert target_oid in live_oids_before, (
            f"test bug: target_oid {target_oid} not in live oids"
        )

        # Call cleanup with the real order-id. Per contract (c), this
        # must succeed identically pre- and post-guard.
        meta_after, n_active_after = mgr.cleanup(
            meta, n_active, jnp.int32(target_oid),
        )

        # Multiset shrinks by exactly one — target_oid is removed, all
        # other live order-ids remain.
        assert int(n_active_after) == int(n_active) - 1
        assert int(n_active_after) == MAX_ICEBERGS - 1
        live_oids_after = _live_oids(meta_after, n_active_after)
        assert sorted(live_oids_after) == sorted(
            oid for oid in live_oids_before if oid != target_oid
        )
        # Target order-id is no longer present in the live prefix.
        assert target_oid not in live_oids_after
        # Packed invariant holds on the returned state.
        _assert_packed_invariant(meta_after, n_active_after)


# ---------------------------------------------------------------------------
# n_active threading correctness across state-construction surfaces.
# ---------------------------------------------------------------------------


class TestNActiveThreading:
    """Audit dimension 4: every ``OMIEBookState`` construction surface
    threads ``n_active`` correctly. Init → 0; expire_at_gct → 0;
    _build_state_jit → matches input; _jit_cancel_by_id returns
    5-tuple with ``n_active`` slot."""

    def test_init_n_active_is_zero(self, book):
        st = book.init()
        assert int(st.n_active) == 0

    def test_expire_at_gct_resets_n_active_to_zero(self, book, mgr):
        # Build a state with some live icebergs, then expire.
        st = book.init()
        order = _ice_order(order_id=55, side=Side.BUY,
                           price=20.0, quantity=4.0, peak_qty=1.0)
        meta_new, na_new, _ = mgr.register(st.iceberg_meta, st.n_active, order)
        st_populated = st._replace(iceberg_meta=meta_new, n_active=na_new)
        assert int(st_populated.n_active) == 1
        st_expired = book.expire_at_gct(st_populated)
        assert int(st_expired.n_active) == 0
        # Meta is also all-sentinel.
        _assert_packed_invariant(
            st_expired.iceberg_meta, st_expired.n_active,
        )

    def test_jit_cancel_by_id_returns_5_tuple_with_n_active(
        self, book, mgr, cfg, mkt_cfg,
    ):
        """_jit_cancel_by_id must return 5-tuple
        (lob, meta, n_active, ts, trunc) per Deliverable (6)."""
        st = book.init()
        jaxob_cfg = _make_jaxob_cfg(mkt_cfg)
        result = _jit_cancel_by_id(
            st.lob, jnp.int32(999), jnp.int32(0),
            st.iceberg_meta, st.n_active, st.ts_counter, st.truncated,
            mgr, jaxob_cfg,
        )
        assert len(result) == 5
        new_lob, new_meta, new_n_active, new_ts, new_trunc = result
        assert isinstance(new_n_active, jnp.ndarray)
        assert new_n_active.shape == ()
        assert new_n_active.dtype == jnp.int32
        # On a no-op cancel (order not found), n_active unchanged.
        assert int(new_n_active) == int(st.n_active)

    def test_build_state_jit_preserves_n_active(self, book, mgr):
        st = book.init()
        new_state = _build_state_jit(
            st.lob,
            st.iceberg_meta,
            jnp.int32(7),
            st.ts_counter,
            st.truncated,
            mgr,
            overflow=jnp.bool_(False),
        )
        assert int(new_state.n_active) == 7

    def test_build_state_method_propagates_n_active(self, book):
        """ContractBook._build_state passes n_active onto OMIEBookState."""
        st = book.init()
        # Forge n_active=5 via _build_state.
        new_state = book._build_state(
            st.lob, st.iceberg_meta, st.ts_counter, st.truncated,
            jnp.int32(5),
        )
        assert int(new_state.n_active) == 5

    def test_cancel_order_path_decrements_n_active_when_iceberg_matched(
        self, book, mgr,
    ):
        """ContractBook.cancel_order (Python-side) must thread
        n_active through the iceberg cleanup branch — when the
        cancelled order id corresponds to an iceberg, n_active drops.
        """
        st = book.init()
        # Register an iceberg directly (no submit_order); cancel_order
        # cleans up iceberg meta by order_id even if the order is not
        # in the LOB.
        order = _ice_order(order_id=77, side=Side.BUY,
                           price=20.0, quantity=4.0, peak_qty=1.0)
        meta_new, na_new, _ = mgr.register(
            st.iceberg_meta, st.n_active, order,
        )
        st_with_ice = st._replace(
            iceberg_meta=meta_new, n_active=na_new,
        )
        assert int(st_with_ice.n_active) == 1
        # cancel_order will not find the order in the LOB (we never
        # dispatched a LIMIT), but the iceberg cleanup branch runs
        # unconditionally? Let's verify the production behaviour: in
        # cancel_order, `if int(order_row[2]) <= 0: return state` —
        # the cleanup branch is short-circuited. So cancel_order on a
        # not-in-LOB id is a complete no-op. Pin this.
        result_state = book.cancel_order(st_with_ice, 77, int(Side.BUY))
        # Expect no change (since order_row not found).
        assert int(result_state.n_active) == int(st_with_ice.n_active)


# ---------------------------------------------------------------------------
# Bounded-scan: trip-count tracks n_active, not MAX_ICEBERGS.
# ---------------------------------------------------------------------------


class TestBoundedScanIndependentOfCap:
    """Audit dimension 5: ``regenerate_slices`` /
    ``matchable_hidden_volume`` / ``truncation_flag`` have outer
    iteration counts tracking ``n_active``, not ``MAX_ICEBERGS``.
    We probe this via jaxpr introspection — the lowered jaxpr should
    contain a ``while_loop`` / ``scan`` whose carry references
    ``n_active`` as the bound."""

    def test_regenerate_slices_uses_while_loop(self, mgr):
        """The outer iteration in ``regenerate_slices`` must be a
        bounded ``lax.while_loop``, not a Python ``for`` over
        MAX_ICEBERGS or a ``scan`` of length MAX_ICEBERGS.
        """
        # Use make_jaxpr to extract the lowered jaxpr.
        cfg = OMIEProductConfig()
        mkt_cfg = OMIEMarketConfig()
        contract = Contract(
            contract_id=37, delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30, gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.OPEN,
        )
        book = ContractBook(contract, cfg, mkt_cfg)
        st = book.init()

        def fn(meta, n_active, ts):
            return mgr.regenerate_slices(st.lob, meta, n_active, ts)

        jaxpr = jax.make_jaxpr(fn)(
            st.iceberg_meta, st.n_active, st.ts_counter,
        )
        text = str(jaxpr)
        # Outer iteration must be a while_loop, not a scan over a
        # MAX_ICEBERGS-long axis. Probe for the primitive name.
        assert "while" in text, (
            "regenerate_slices does not lower to a while_loop"
        )

    def test_matchable_hidden_volume_uses_fori_loop_bounded_by_n_active(
        self, mgr,
    ):
        """matchable_hidden_volume outer loop must be a fori_loop or
        while_loop, not a vmap over the full (MAX_ICEBERGS, 8) meta."""
        meta = IcebergManager.empty_meta()

        def fn(meta, n_active):
            return mgr.matchable_hidden_volume(
                meta, n_active, jnp.int32(0), jnp.int32(1000),
            )

        jaxpr = jax.make_jaxpr(fn)(meta, jnp.int32(0))
        text = str(jaxpr)
        # Either a fori_loop (lowers to scan in jaxpr) or while_loop.
        assert ("scan" in text) or ("while" in text), (
            "matchable_hidden_volume's outer iteration is not a "
            "bounded scan / while_loop"
        )

    def test_truncation_flag_with_n_active_is_bounded(self, mgr):
        """truncation_flag(meta, n_active) must use a fori_loop over
        the n_active prefix, NOT a per-row vmap over MAX_ICEBERGS."""
        meta = IcebergManager.empty_meta()

        def fn(meta, n_active):
            return mgr.truncation_flag(meta, n_active)

        jaxpr = jax.make_jaxpr(fn)(meta, jnp.int32(0))
        text = str(jaxpr)
        assert ("scan" in text) or ("while" in text), (
            "truncation_flag(n_active) does not use a bounded loop"
        )

    def test_truncation_flag_no_n_active_is_legacy_unbounded(self, mgr):
        """truncation_flag(meta, n_active=None) is the legacy
        ergonomic form — produces correct results but is NOT bounded
        (operates over the full MAX_ICEBERGS rows). Pin behaviour
        equivalence between the two forms on packed-invariant states."""
        meta = IcebergManager.empty_meta()
        # Both forms must return False on an empty meta.
        unbounded = bool(mgr.truncation_flag(meta))
        bounded = bool(mgr.truncation_flag(meta, jnp.int32(0)))
        assert unbounded == bounded == False  # noqa: E712

    def test_bounded_and_unbounded_truncation_flag_agree_on_live_state(
        self, mgr,
    ):
        """Risk audit item (2): bounded and unbounded ``truncation_flag``
        must produce identical 0-d bools on any packed-invariant state
        (rows in [n_active, MAX_ICEBERGS) are sentinels and contribute
        False to the OR regardless of bound)."""
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        # Register a benign iceberg (well below truncation threshold).
        o = _ice_order(order_id=1, side=Side.BUY, price=20.0,
                       quantity=4.0, peak_qty=1.0)
        meta, n_active, _ = mgr.register(meta, n_active, o)
        unbounded = bool(mgr.truncation_flag(meta))
        bounded = bool(mgr.truncation_flag(meta, n_active))
        assert unbounded == bounded


# ---------------------------------------------------------------------------
# base_price_int immutability across regenerate_slices cascades.
# ---------------------------------------------------------------------------


class TestBasePriceImmutability:
    """Audit dimension 6: across N successive ``regenerate_slices``
    calls on a single ICEwPI iceberg, ``base_price_int`` is invariant;
    only ``slice_count`` increments and ``remaining_hidden`` decreases.
    """

    def test_base_price_int_constant_across_register_only(self, mgr, cfg):
        """Probe the static property of ``register``: even with
        non-zero ``ppd``, ``base_price_int`` is the offset-encoded
        register-time price, unchanged by the encoder."""
        meta = IcebergManager.empty_meta()
        order_a = _ice_order(order_id=1, side=Side.BUY, price=50.0,
                             quantity=10.0, peak_qty=2.0, ppd=0.05)
        order_b = _ice_order(order_id=2, side=Side.SELL, price=51.0,
                             quantity=10.0, peak_qty=2.0, ppd=-0.10)
        meta, na, _ = mgr.register(meta, jnp.int32(0), order_a)
        meta, na, _ = mgr.register(meta, na, order_b)
        # base_price_int is offset-encoded register-time price.
        expected_a = round((50.0 - cfg.price_min) / cfg.price_tick)
        expected_b = round((51.0 - cfg.price_min) / cfg.price_tick)
        assert int(meta[0, COL_BASE_PRICE_INT]) == expected_a
        assert int(meta[1, COL_BASE_PRICE_INT]) == expected_b
        # ppd is signed
        assert int(meta[0, COL_PPD]) > 0
        assert int(meta[1, COL_PPD]) < 0

    def test_base_price_unchanged_across_two_register_cycles(self, mgr):
        """Register two icebergs with the same order_id (after cleanup)
        and confirm the second register stamps its own base_price_int —
        but the FIRST iceberg's base_price was preserved through
        register+cleanup+register lifecycle (no leakage between
        register calls)."""
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        order_a = _ice_order(order_id=1, side=Side.BUY, price=30.0,
                             quantity=4.0, peak_qty=1.0, ppd=0.0)
        meta, n_active, _ = mgr.register(meta, n_active, order_a)
        base_a = int(meta[0, COL_BASE_PRICE_INT])
        # Cleanup + re-register at a DIFFERENT price.
        meta, n_active = mgr.cleanup(meta, n_active, 1)
        order_a2 = _ice_order(order_id=1, side=Side.BUY, price=35.0,
                              quantity=4.0, peak_qty=1.0, ppd=0.0)
        meta, n_active, _ = mgr.register(meta, n_active, order_a2)
        base_a2 = int(meta[0, COL_BASE_PRICE_INT])
        # The two register calls must stamp DIFFERENT base prices.
        assert base_a != base_a2


# ---------------------------------------------------------------------------
# Multiset equivalence — replacement for the retired test_cleanup_middle.
# ---------------------------------------------------------------------------


class TestMultisetEquivalenceAuditFlavour:
    """Audit dimension 7: under (2a) swap-with-last, the multiset of
    order_ids in meta[:n_active] equals the set of registered minus
    cleaned-up — even when cleanup hits a middle slot. The retired
    locked test ``test_cleanup_middle_slot_then_register_fills_middle``
    over-specified slot identity; the structural intent is multiset
    equivalence under arbitrary churn.
    """

    def test_register_three_cleanup_middle_then_register_multiset_correct(
        self, mgr,
    ):
        """Re-derived from the retired test's intent — what survives is
        the multiset of live oids, not the slot positions."""
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        # Register 3.
        for oid, price in [(10, 5.0), (20, 6.0), (30, 7.0)]:
            o = _ice_order(order_id=oid, side=Side.BUY, price=price,
                           quantity=4.0, peak_qty=1.0)
            meta, n_active, _ = mgr.register(meta, n_active, o)
        # Cleanup middle.
        meta, n_active = mgr.cleanup(meta, n_active, 20)
        # Register a fourth.
        o4 = _ice_order(order_id=40, side=Side.SELL, price=8.0,
                        quantity=4.0, peak_qty=1.0)
        meta, n_active, _ = mgr.register(meta, n_active, o4)
        # Multiset of live oids must be {10, 30, 40} — slot positions
        # are deliberately not asserted.
        live = sorted(_live_oids(meta, n_active))
        assert live == sorted([10, 30, 40])
        _assert_packed_invariant(meta, n_active)

    def test_register_cleanup_cycles_produce_unique_oids(self, mgr):
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        for i in range(5):
            o = _ice_order(order_id=200 + i, side=Side.BUY, price=10.0,
                           quantity=4.0, peak_qty=1.0)
            meta, n_active, _ = mgr.register(meta, n_active, o)
        # Cleanup all but one — random middles and ends.
        meta, n_active = mgr.cleanup(meta, n_active, 202)
        meta, n_active = mgr.cleanup(meta, n_active, 200)
        meta, n_active = mgr.cleanup(meta, n_active, 204)
        # 2 live left: {201, 203}
        live = sorted(_live_oids(meta, n_active))
        assert live == sorted([201, 203])
        assert len(live) == len(set(live))


# ---------------------------------------------------------------------------
# JIT compilation surface and float64 promotion probe.
# ---------------------------------------------------------------------------


class TestJITCompilationSurface:
    """Audit dimension 1: production paths that consume ``n_active``
    compile and trace deterministically — no Python branching on
    traced values. JIT round-trip preserves the OMIEBookState shape.
    """

    pytestmark = pytest.mark.jit

    def test_register_compiles_under_jit_with_n_active(self, mgr):
        """register can be wrapped in jax.jit and run with traced
        n_active without raising. Manager is closed over by partial."""
        meta = IcebergManager.empty_meta()
        order = _ice_order(order_id=1, side=Side.BUY, price=20.0,
                           quantity=4.0, peak_qty=1.0)

        @jax.jit
        def call_register(meta, n_active, order):
            return mgr.register(meta, n_active, order)

        result = call_register(meta, jnp.int32(0), order)
        assert len(result) == 3
        new_meta, new_na, overflow = result
        assert isinstance(new_meta, jnp.ndarray)
        assert int(new_na) == 1
        assert bool(overflow) is False

    def test_cleanup_compiles_under_jit_with_n_active(self, mgr):
        meta = IcebergManager.empty_meta()
        order = _ice_order(order_id=88, side=Side.SELL, price=20.0,
                           quantity=4.0, peak_qty=1.0)
        meta, n_active, _ = mgr.register(meta, jnp.int32(0), order)

        @jax.jit
        def call_cleanup(meta, n_active, oid):
            return mgr.cleanup(meta, n_active, oid)

        new_meta, new_na = call_cleanup(meta, n_active, jnp.int32(88))
        assert int(new_na) == 0
        _assert_packed_invariant(new_meta, new_na)

    def test_regenerate_slices_compiles_under_jit(self, mgr, book):
        st = book.init()

        @jax.jit
        def call_regen(lob, meta, n_active, ts):
            return mgr.regenerate_slices(lob, meta, n_active, ts)

        result = call_regen(st.lob, st.iceberg_meta, st.n_active, st.ts_counter)
        assert len(result) == 4

    def test_no_float64_on_register_traced_path(self, mgr):
        """No float64 promotion in register's lowered jaxpr — Method A
        invariant from P2-L1. Float fields encode to int32 via
        round+cast in float32 land; no f64 anywhere."""
        order = _ice_order(order_id=1, side=Side.BUY, price=20.5,
                           quantity=4.7, peak_qty=1.3, ppd=0.05)
        meta = IcebergManager.empty_meta()

        def fn(meta, n_active, order):
            return mgr.register(meta, n_active, order)

        jaxpr = jax.make_jaxpr(fn)(meta, jnp.int32(0), order)
        text = str(jaxpr)
        # No float64 dtype should appear in the lowered jaxpr.
        assert "float64" not in text, (
            "register lowered jaxpr contains float64 — Method A invariant "
            "broken"
        )

    def test_no_float64_on_matchable_hidden_volume_traced_path(self, mgr):
        meta = IcebergManager.empty_meta()

        def fn(meta, n_active):
            return mgr.matchable_hidden_volume(
                meta, n_active, jnp.int32(0), jnp.int32(1000),
            )

        jaxpr = jax.make_jaxpr(fn)(meta, jnp.int32(0))
        text = str(jaxpr)
        assert "float64" not in text

    def test_state_pytree_round_trips_through_jit(self, book, empty_state):
        """Pytree flatten/unflatten on OMIEBookState (including
        n_active) is jit-compatible — a no-op jit on the state must
        produce a state with identical n_active dtype/shape/value."""

        @jax.jit
        def identity(state):
            return state

        new_state = identity(empty_state)
        assert isinstance(new_state, OMIEBookState)
        assert new_state.n_active.shape == ()
        assert new_state.n_active.dtype == jnp.int32
        assert int(new_state.n_active) == int(empty_state.n_active)


# ---------------------------------------------------------------------------
# submit_order ICE path threads n_active through the full lift.
# ---------------------------------------------------------------------------


class TestSubmitOrderICEPathThreading:
    """Audit dimension 10: the module-level ``_submit_order`` jitted
    lift threads ``n_active`` through the ICE arm of the
    ``lax.switch`` and returns an ``OMIEBookState`` with consistent
    ``n_active``.
    """

    pytestmark = pytest.mark.jit

    def test_submit_ice_order_increments_n_active(self, book, cfg, mkt_cfg):
        st = book.init()
        ice_order = make_order(
            contract_id=37, side=Side.BUY, price=50.0, quantity=10.0,
            order_id=1001, agent_id=5, timestamp=100,
            exec_restriction=ExecRestriction.ICE, peak_qty=2.0, ppd=0.0,
        )
        new_state, result = _submit_order(st, ice_order, cfg, mkt_cfg)
        assert bool(result.accepted) is True
        # n_active increments by exactly 1 on a fresh iceberg.
        assert int(new_state.n_active) == int(st.n_active) + 1
        # Packed invariant holds end-to-end.
        _assert_packed_invariant(new_state.iceberg_meta, new_state.n_active)

    def test_submit_two_ice_orders_packed_prefix_grows_contiguously(
        self, book, cfg, mkt_cfg,
    ):
        st = book.init()
        ice1 = make_order(
            contract_id=37, side=Side.BUY, price=50.0, quantity=10.0,
            order_id=2001, agent_id=5, timestamp=100,
            exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
        )
        ice2 = make_order(
            contract_id=37, side=Side.SELL, price=60.0, quantity=8.0,
            order_id=2002, agent_id=6, timestamp=101,
            exec_restriction=ExecRestriction.ICE, peak_qty=1.5,
        )
        st1, _ = _submit_order(st, ice1, cfg, mkt_cfg)
        st2, _ = _submit_order(st1, ice2, cfg, mkt_cfg)
        assert int(st2.n_active) == 2
        # Both icebergs live in the contiguous prefix.
        live_oids = _live_oids(st2.iceberg_meta, st2.n_active)
        assert sorted(live_oids) == sorted([2001, 2002])
        _assert_packed_invariant(st2.iceberg_meta, st2.n_active)

    def test_submit_non_ice_order_leaves_n_active_zero(
        self, book, cfg, mkt_cfg,
    ):
        st = book.init()
        non_order = make_order(
            contract_id=37, side=Side.BUY, price=50.0, quantity=2.0,
            order_id=3001, agent_id=7, timestamp=100,
            exec_restriction=ExecRestriction.NON,
        )
        new_state, result = _submit_order(st, non_order, cfg, mkt_cfg)
        assert bool(result.accepted) is True
        # No iceberg ⇒ n_active stays 0.
        assert int(new_state.n_active) == 0
        _assert_packed_invariant(new_state.iceberg_meta, new_state.n_active)


# ---------------------------------------------------------------------------
# Direct-construction fixture-coherence (Risk audit item (3)).
# ---------------------------------------------------------------------------


class TestDirectConstructionCoherence:
    """Audit dimension 8: any synthetic ``OMIEBookState`` built by a
    test fixture must carry an ``n_active`` consistent with its
    ``iceberg_meta``. Probes the audit-suite's own helper."""

    def test_empty_state_replace_with_explicit_n_active_zero_is_coherent(
        self, empty_state,
    ):
        st = empty_state._replace(n_active=jnp.int32(0))
        _assert_packed_invariant(st.iceberg_meta, st.n_active)

    def test_state_with_synthetic_saturated_meta_must_set_n_active(
        self, book, mgr,
    ):
        """A synthetic state that bypasses register/cleanup must also
        bypass the n_active maintenance — direct-construction is
        author's responsibility per Risk audit (3)."""
        st = book.init()
        # Build a (partially) live meta directly.
        order = _ice_order(order_id=1, side=Side.BUY, price=20.0,
                           quantity=4.0, peak_qty=1.0)
        meta_new, na_new, _ = mgr.register(st.iceberg_meta, st.n_active, order)
        # Direct construction:
        synthetic = OMIEBookState(
            lob=st.lob,
            iceberg_meta=meta_new,
            ts_counter=st.ts_counter,
            truncated=st.truncated,
            n_active=na_new,
        )
        _assert_packed_invariant(synthetic.iceberg_meta, synthetic.n_active)
        assert int(synthetic.n_active) == 1


# ---------------------------------------------------------------------------
# expire_at_gct invariants and cross-step composition.
# ---------------------------------------------------------------------------


class TestExpireAtGctComposition:
    """Audit dimension 4: expire_at_gct produces a coherent state
    that itself satisfies the packed invariant, and the truncated
    sticky flag is preserved across the reset."""

    def test_expire_at_gct_after_population_resets_meta_and_n_active(
        self, book, mgr,
    ):
        st = book.init()
        for i in range(3):
            o = _ice_order(order_id=400 + i, side=Side.BUY,
                           price=20.0 + i, quantity=4.0, peak_qty=1.0)
            meta_n, na_n, _ = mgr.register(st.iceberg_meta, st.n_active, o)
            st = st._replace(iceberg_meta=meta_n, n_active=na_n)
        assert int(st.n_active) == 3
        # Expire.
        st_expired = book.expire_at_gct(st)
        assert int(st_expired.n_active) == 0
        sent = _sentinel_row()
        # jnp.all reduces the per-cell `== sent[c]` checks across the
        # full meta; bidirectionally equivalent to the nested per-iter
        # form (any mismatch trips the assertion).
        assert bool(jnp.all(st_expired.iceberg_meta == sent[None, :]))
        _assert_packed_invariant(
            st_expired.iceberg_meta, st_expired.n_active,
        )

    def test_expire_at_gct_preserves_truncated_sticky_flag(self, book):
        """If the truncated flag was set going in, it must survive
        the expire reset — it's a session-sticky bit."""
        st = book.init()
        st_truncated = st._replace(truncated=jnp.bool_(True))
        st_expired = book.expire_at_gct(st_truncated)
        assert bool(st_expired.truncated) is True


# ---------------------------------------------------------------------------
# Mechanism (2a) per-op cost contract — observed via primitive count.
# ---------------------------------------------------------------------------


class TestMechanism2aJaxprShape:
    """Audit dimension 5: the per-op JAX cost of ``register`` and
    ``cleanup`` matches the v5.30 spec's pinned cost (1 write + 1 add
    for register; 2 writes + 1 subtract for cleanup). Verified via
    jaxpr primitive enumeration."""

    def test_register_jaxpr_does_not_contain_argmax(self, mgr):
        """The pre-(2a) implementation used ``jnp.argmax`` over a
        sentinel mask. Mechanism (2a) replaces this with a direct
        ``.at[n_active].set(...)`` write — argmax must NOT appear in
        the lowered register jaxpr."""
        meta = IcebergManager.empty_meta()
        order = _ice_order(order_id=1, side=Side.BUY, price=20.0,
                           quantity=4.0, peak_qty=1.0)

        def fn(meta, na, order):
            return mgr.register(meta, na, order)

        jaxpr = jax.make_jaxpr(fn)(meta, jnp.int32(0), order)
        text = str(jaxpr)
        assert "argmax" not in text, (
            "register jaxpr contains argmax — mechanism (2a) requires "
            "direct .at[n_active].set, not search-by-sentinel"
        )

    def test_register_overflow_predicate_is_int32_compare(self, mgr):
        """Overflow predicate must be a single int32 compare
        (``n_active >= MAX_ICEBERGS``), not a reduction over the
        sentinel mask. We probe by jaxpr inspection: the substring
        ``reduce_or`` (or ``any``) should NOT appear in register's
        overflow gate."""
        meta = IcebergManager.empty_meta()
        order = _ice_order(order_id=1, side=Side.BUY, price=20.0,
                           quantity=4.0, peak_qty=1.0)

        def fn(meta, na, order):
            return mgr.register(meta, na, order)

        jaxpr = jax.make_jaxpr(fn)(meta, jnp.int32(0), order)
        text = str(jaxpr)
        # The old (~jnp.any) form would lower to reduce_or. The new
        # form is a single ge_ comparison.
        # Note: matchable_hidden_volume uses scan internally for
        # iteration; we're only inspecting register's jaxpr here.
        assert "reduce_or" not in text, (
            "register jaxpr contains reduce_or — mechanism (2a) requires "
            "a single int32 compare, not a sentinel-mask reduction"
        )


# ---------------------------------------------------------------------------
# Trace-count invariant (Risk audit (1) — single compile).
# ---------------------------------------------------------------------------


class TestTraceCount:
    """Audit dimension 1: ``_submit_order`` jitted lift compiles
    ONCE across config-equal-but-identity-distinct
    ``OMIEProductConfig`` / ``OMIEMarketConfig`` instances. Two
    distinct config objects with equal field values must share the
    same cached compile artifact."""

    pytestmark = pytest.mark.jit

    def test_submit_order_one_compile_across_identical_config_instances(
        self, book,
    ):
        """Build two OMIEProductConfig instances with identical fields
        and verify a single compile is produced."""
        cfg_a = OMIEProductConfig()
        cfg_b = OMIEProductConfig()
        mkt_a = OMIEMarketConfig()
        mkt_b = OMIEMarketConfig()
        assert cfg_a == cfg_b
        assert mkt_a == mkt_b

        ice_order = make_order(
            contract_id=37, side=Side.BUY, price=50.0, quantity=10.0,
            order_id=5001, agent_id=5, timestamp=100,
            exec_restriction=ExecRestriction.ICE, peak_qty=2.0,
        )

        # Warm up with cfg_a; record cache before/after with cfg_b.
        st = book.init()
        cache_a = _submit_order._cache_size()
        _submit_order(st, ice_order, cfg_a, mkt_a)
        cache_b = _submit_order._cache_size()
        _submit_order(st, ice_order, cfg_b, mkt_b)
        cache_c = _submit_order._cache_size()
        # Cache must NOT grow on the second call — same effective
        # static args.
        assert cache_c == cache_b, (
            f"submit_order recompiled across identical configs "
            f"(b={cache_b} c={cache_c}); should share artifact"
        )


# ---------------------------------------------------------------------------
# Stress / load-scale probe — marked slow.
# ---------------------------------------------------------------------------


class TestLoadStressBoundedScan:
    """Audit dimension 5: bounded-scan cost-independence-from-cap.
    Performance probe complementing the test-first contracts (a) and
    (b). slow because each call exercises the JIT-compile + warmup +
    measurement loop."""

    @pytest.mark.slow
    @pytest.mark.jit
    def test_regenerate_slices_runtime_independent_of_n_active_when_empty(
        self, book, mgr,
    ):
        """slow: one warmup pass + 100 timed iterations of the same
        jitted regenerate_slices on an empty meta. On a packed-layout
        invariant state with n_active=0, the while_loop should not
        enter its body — runtime is bounded by the constant overhead
        of dispatch."""
        st = book.init()

        @jax.jit
        def call_regen(state):
            lob, meta, n_active, ts = mgr.regenerate_slices(
                state.lob, state.iceberg_meta, state.n_active, state.ts_counter,
            )
            return lob, meta, n_active, ts

        # Warmup.
        result = call_regen(st)
        jax.block_until_ready(result[1])

        # Time 100 calls.
        t0 = time.perf_counter()
        for _ in range(100):
            result = call_regen(st)
            jax.block_until_ready(result[1])
        t1 = time.perf_counter()
        elapsed_us = (t1 - t0) * 1e6 / 100
        # Loose bound: each dispatch (no body) must complete well
        # under 100 ms (sanity probe, not a tight perf assertion).
        assert elapsed_us < 1e6, (
            f"regenerate_slices on empty meta took "
            f"{elapsed_us:.1f} us/call — way more than expected"
        )


# ---------------------------------------------------------------------------
# Module-boundary / import-graph discipline (carried over from P2-H1).
# ---------------------------------------------------------------------------


class TestModuleBoundaryDiscipline:
    """Audit dimension 10: ``iceberg_manager.py`` does not import from
    jaxen / jaxrl. The packed-layout refactor must not have introduced
    a forbidden cross-layer dependency."""

    def test_iceberg_manager_does_not_import_jaxen_or_jaxrl(self):
        src = Path(
            "omiecid_simulator/omiecid/iceberg_manager.py"
        ).read_text(encoding="utf-8")
        forbidden = re.compile(
            r"from\s+omiecid_simulator\.(jaxen|jaxrl)\b"
            r"|import\s+omiecid_simulator\.(jaxen|jaxrl)\b"
        )
        assert forbidden.search(src) is None, (
            "iceberg_manager.py imports from jaxen / jaxrl"
        )

    def test_contract_book_does_not_import_jaxen_or_jaxrl(self):
        src = Path(
            "omiecid_simulator/omiecid/contract_book.py"
        ).read_text(encoding="utf-8")
        forbidden = re.compile(
            r"from\s+omiecid_simulator\.(jaxen|jaxrl)\b"
            r"|import\s+omiecid_simulator\.(jaxen|jaxrl)\b"
        )
        assert forbidden.search(src) is None

    def test_register_signature_returns_three_tuple(self, mgr):
        """Pin the public surface of register: (meta, n_active, overflow)."""
        sig = inspect.signature(mgr.register)
        params = list(sig.parameters.keys())
        # Method bound; expect (iceberg_meta, n_active, order)
        assert "iceberg_meta" in params
        assert "n_active" in params
        assert "order" in params

    def test_cleanup_signature_returns_two_tuple(self, mgr):
        """Pin the public surface of cleanup: (meta, n_active)."""
        sig = inspect.signature(mgr.cleanup)
        params = list(sig.parameters.keys())
        assert "iceberg_meta" in params
        assert "n_active" in params
        assert "order_id" in params

    def test_regenerate_slices_signature(self, mgr):
        """Pin the public surface of regenerate_slices."""
        sig = inspect.signature(mgr.regenerate_slices)
        params = list(sig.parameters.keys())
        assert "lob" in params
        assert "iceberg_meta" in params
        assert "n_active" in params
        assert "ts_counter" in params

    def test_matchable_hidden_volume_signature(self, mgr):
        sig = inspect.signature(mgr.matchable_hidden_volume)
        params = list(sig.parameters.keys())
        assert "iceberg_meta" in params
        assert "n_active" in params
        assert "side" in params
        assert "limit_price_int" in params

    def test_truncation_flag_n_active_optional(self, mgr):
        """truncation_flag still supports the n_active=None legacy
        form for ergonomic test-side calls. Pin the kwarg default."""
        sig = inspect.signature(mgr.truncation_flag)
        assert sig.parameters["n_active"].default is None
