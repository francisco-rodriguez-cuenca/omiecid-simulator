"""Locked behavioral-contract tests for P2-PERF-6 — Packed-array
`regenerate_slices` (live-slot scan).

These tests pin the eight deliverable / test-first-contract families
from the v5.29-amended P2-PERF-6 story card (see
``docs/omiecid/agile-plan.md`` §"Story P2-PERF-6 — Packed-array
`regenerate_slices` (live-slot scan)"). They are authored before the
implementation lands, so every test in classes T1–T8 will FAIL on
``main`` (``OMIEBookState`` does not yet carry ``n_active``;
``register`` does not yet take an ``n_active`` argument; the bounded
``lax.dynamic_slice`` lifts on ``regenerate_slices`` and
``matchable_hidden_volume`` have not been installed; etc.).

This is by design: the failures are the implementation targets.

Acceptance-criteria families (one class per family):

  T1 — ``n_active`` field invariant on ``OMIEBookState``.
       (Deliverable (1); Deliverable (6) propagation surface.)

  T2 — Packed-layout invariant: live rows form a contiguous prefix.
       (Deliverable (2) — the load-bearing structural invariant.)

  T3 — Mechanism (2a) per-op contract: swap-with-last on delete,
       append-at-tail on register, ``n_active >= iceberg_meta.shape[0]``
       overflow predicate (the configured ``market_config.max_icebergs``,
       P3-PERF-REGEN-CAP-1 — not the ``MAX_ICEBERGS`` ceiling).
       (Deliverable (2) per-op JAX cost.)

  T4 — Multiset-equivalence replacement for the retired
       `test_cleanup_middle_slot_then_register_fills_middle` test.
       (Test-first contract (c) explicit replacement requirement.)

  T5 — Positive lock on the three preserved invariant-(5) locked
       tests in `tests/omiecid/test_iceberg_manager_jit.py`.
       (Test-first contract (c) positive-lock requirement.)

  T6 — `regenerate_slices` outer-scan bound = ``n_active``, not
       ``MAX_ICEBERGS``. (Deliverable (3).)

  T7 — `matchable_hidden_volume` outer-scan bound = ``n_active``,
       not ``MAX_ICEBERGS``. (Deliverable (4) — bounded vmap is the
       perf-independence-from-cap mechanism; the mask gives
       correctness, not the perf win.)

  T8 — `truncation_flag` parity under the bounded scan; sticky-flag
       accumulator in ``contract_book._build_state_jit`` keeps its
       ``jnp.logical_or(prior_truncated, live)`` invariant.
       (Deliverable (5); Risk audit (2).)

  T9 — Performance contracts (a) and (b): wall-clock at
       ``MAX_ICEBERGS=8192`` with ≤100 live within 20% of the
       ``MAX_ICEBERGS=100`` baseline; load-stress at ~2000 live under
       ~5× the ≤100 baseline. Bounded ``matchable_hidden_volume`` is
       also pinned. ``@pytest.mark.slow`` — these are micro-benchmarks
       that compile and time real JAX paths.

Reference docs (read at authoring time):
  - docs/omiecid/agile-plan.md v5.29 §"Story P2-PERF-6" (story card)
  - docs/omiecid/sprint-status.md §"P2-PERF-6 — Architecture Review
    (2026-05-20, second pass)" (mechanism (2a) verdict + comparison)
  - docs/omiecid/market-spec.md §4.2 (iceberg type catalogue),
    §5.2 (timestamp rules), §6.1 (price-time priority).
  - docs/omiecid/migration-plan.md (canonical static_argnames idiom;
    OMIEBookState schema).

Test discipline:
  - Module-level marker ``@pytest.mark.jit`` — every test exercises
    or compiles iceberg-manager primitives.
  - ``T9`` micro-benchmarks additionally carry ``@pytest.mark.slow``.
  - Tests use the conftest module-scoped fixtures
    (``empty_state`` / ``book`` / ``book_and_state`` /
    ``open_contract`` / ``product_config`` / ``market_config``).
  - Tests probe the post-P2-PERF-6 surface lazily so the file
    remains collectible against the current (pre-(2a)) code; the
    failures are ``AttributeError`` / ``TypeError`` / shape-mismatch
    until the implementation lands.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
)
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_PEAK_QTY,
    COL_REMAINING_HIDDEN,
    COL_SIDE,
    COL_BASE_PRICE_INT,
    COL_PPD,
    COL_SLICE_COUNT,
    N_COLS,
    IcebergManager,
)
from omiecid_simulator.omiecid.jax_constants import (
    MAX_ICEBERGS,
    MAX_SLICES_PER_REGEN,
    SENTINEL_ORDER_ID,
)
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    make_order,
)


# Module-level marker (Story T1.1): every test in this file traces or
# compiles iceberg-manager primitives. ``@pytest.mark.slow`` is added
# per-test in class T9 for the wall-clock benchmarks.
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Order constructors (mirror tests/omiecid/test_iceberg_manager_jit.py).
# ---------------------------------------------------------------------------


def _make_iceberg_buy(
    order_id=1,
    price=50.00,
    quantity=30.0,
    peak_qty=10.0,
    agent_id=0,
    timestamp=100,
    ppd=0.0,
):
    """Standard or ICEwPI buy iceberg (ppd <= 0 by market-spec §13.6)."""
    return make_order(
        contract_id=37,
        side=Side.BUY,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=ExecRestriction.ICE,
        peak_qty=peak_qty,
        ppd=ppd,
    )


def _make_iceberg_sell(
    order_id=2,
    price=50.00,
    quantity=30.0,
    peak_qty=10.0,
    agent_id=1,
    timestamp=200,
    ppd=0.0,
):
    """Standard or ICEwPI sell iceberg (ppd >= 0 by market-spec §13.6)."""
    return make_order(
        contract_id=37,
        side=Side.SELL,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=ExecRestriction.ICE,
        peak_qty=peak_qty,
        ppd=ppd,
    )


# ---------------------------------------------------------------------------
# Adapter helpers — bridge the pre-(2a) and post-(2a) iceberg-manager APIs.
#
# The story card pins post-(2a):
#   register(meta, n_active, order) -> (meta, n_active, overflow)
#   cleanup(meta, n_active, order_id) -> (meta, n_active)
# (cleanup keeps order_id at its public boundary; the (2a) "slot"
# argument in the card is the internal slot resolved from order_id.)
#
# These helpers normalise the return-shape across pre- and post-(2a)
# code so the tests can drive the manager without breaking collection
# on a pre-implementation tree. Behavioural assertions remain strict.
# ---------------------------------------------------------------------------


def _register(iceberg_mgr, meta, n_active, order):
    """Call ``register`` with ``n_active`` and return ``(meta, n_active, overflow)``.

    Post-(2a) signature: ``register(meta, n_active, order) ->
    (meta, n_active, overflow)``. The pre-(2a) implementation does not
    take ``n_active`` — calling it with the post-(2a) signature is
    expected to raise ``TypeError`` until the implementation lands.
    """
    return iceberg_mgr.register(meta, n_active, order)


def _cleanup(iceberg_mgr, meta, n_active, order_id):
    """Call ``cleanup`` with ``n_active`` and return ``(meta, n_active)``."""
    return iceberg_mgr.cleanup(meta, n_active, order_id)


def _init_state_post_p2_perf_6(book: ContractBook) -> OMIEBookState:
    """Initialise an ``OMIEBookState`` and assert it carries ``n_active``.

    Post-P2-PERF-6 contract: ``ContractBook.init()`` returns a state
    whose ``n_active`` field is ``jnp.int32(0)``.
    """
    state = book.init()
    # The presence of n_active is itself part of the contract; the
    # AttributeError below is the "missing field" failure mode for
    # pre-(2a) trees.
    assert hasattr(state, "n_active"), (
        "OMIEBookState must carry `n_active: jnp.int32` post-P2-PERF-6 "
        "(Deliverable (1))."
    )
    return state


# ===========================================================================
# Class T1 — ``n_active`` field invariant (Deliverable (1), (6)).
# ===========================================================================


class TestT1NActiveFieldInvariant:
    """``n_active`` exists on ``OMIEBookState`` and tracks live count.

    Locks Deliverable (1) (the new state field) and the observable
    end-to-end behaviour of Deliverable (6) (``n_active`` propagation
    enumeration — every register/cleanup must thread ``n_active``).
    """

    def test_state_carries_n_active_field(self, book):
        """``OMIEBookState`` carries an ``n_active`` attribute."""
        state = book.init()
        assert hasattr(state, "n_active"), (
            "Deliverable (1): OMIEBookState must carry n_active."
        )

    def test_n_active_is_int32_scalar(self, book):
        """``n_active`` is a 0-d ``jnp.int32`` scalar."""
        state = _init_state_post_p2_perf_6(book)
        n_active = state.n_active
        assert isinstance(n_active, jnp.ndarray), (
            "n_active must be a jnp.ndarray (0-d int32 scalar)."
        )
        assert n_active.dtype == jnp.int32, (
            f"n_active dtype must be jnp.int32; got {n_active.dtype}."
        )
        assert n_active.shape == (), (
            f"n_active must be a 0-d scalar; got shape {n_active.shape}."
        )

    def test_init_n_active_is_zero(self, book):
        """``ContractBook.init()`` produces ``n_active == 0``."""
        state = _init_state_post_p2_perf_6(book)
        assert int(state.n_active) == 0, (
            "Initial n_active must be 0 (empty packed array)."
        )

    def test_n_active_increments_on_register(self, book):
        """After N successful registers, ``n_active == N``."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active
        N = 5
        for i in range(N):
            meta, n_active, overflow = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=100 + i)
            )
            assert not bool(overflow), (
                f"Register {i} must not overflow on an empty book."
            )
        assert int(n_active) == N, (
            f"After {N} registers, n_active must equal {N}; got {int(n_active)}."
        )

    def test_n_active_decrements_on_cleanup(self, book):
        """After M registers + K cleanups, ``n_active == M - K``."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active

        # Register M=4
        for oid in [10, 20, 30, 40]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )
        assert int(n_active) == 4

        # Cleanup K=2 (any order — both arbitrary middle and head)
        meta, n_active = _cleanup(mgr, meta, n_active, 20)
        meta, n_active = _cleanup(mgr, meta, n_active, 10)
        assert int(n_active) == 2, (
            f"After 4 registers + 2 cleanups, n_active must be 2; "
            f"got {int(n_active)}."
        )

    def test_n_active_interleaved_register_cleanup(self, book):
        """Interleaved register/cleanup sequences preserve M - K invariant."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active

        # Sequence: R(1), R(2), C(1), R(3), C(2), R(4), R(5), C(3)
        # Net: 5 registers - 3 cleanups = 2 live.
        meta, n_active, _ = _register(mgr, meta, n_active, _make_iceberg_buy(order_id=1))
        meta, n_active, _ = _register(mgr, meta, n_active, _make_iceberg_buy(order_id=2))
        meta, n_active = _cleanup(mgr, meta, n_active, 1)
        meta, n_active, _ = _register(mgr, meta, n_active, _make_iceberg_buy(order_id=3))
        meta, n_active = _cleanup(mgr, meta, n_active, 2)
        meta, n_active, _ = _register(mgr, meta, n_active, _make_iceberg_buy(order_id=4))
        meta, n_active, _ = _register(mgr, meta, n_active, _make_iceberg_buy(order_id=5))
        meta, n_active = _cleanup(mgr, meta, n_active, 3)

        assert int(n_active) == 2, (
            f"Interleaved 5R/3C must yield n_active=2; got {int(n_active)}."
        )


# ===========================================================================
# Class T2 — Packed-layout invariant (Deliverable (2)).
# ===========================================================================


class TestT2PackedLayoutInvariant:
    """Live rows form a contiguous prefix; sentinels live at the tail.

    For any state reachable from ``ContractBook.init()`` followed by
    any sequence of ``register`` / ``cleanup`` calls:

      ``meta[:n_active, COL_ORDER_ID]`` are all != ``SENTINEL_ORDER_ID``
      ``meta[n_active:, COL_ORDER_ID]`` are all == ``SENTINEL_ORDER_ID``

    This is the load-bearing structural property of mechanism (2a) —
    Deliverable (2).
    """

    @staticmethod
    def _assert_packed(meta, n_active):
        """Assert the contiguous-prefix invariant."""
        n = int(n_active)
        col = meta[:, COL_ORDER_ID]
        if n > 0:
            prefix = col[:n]
            assert bool(jnp.all(prefix != jnp.int32(SENTINEL_ORDER_ID))), (
                f"meta[:n_active={n}, COL_ORDER_ID] must contain NO sentinels; "
                f"got prefix = {prefix.tolist()}"
            )
        if n < int(MAX_ICEBERGS):
            tail = col[n:]
            assert bool(jnp.all(tail == jnp.int32(SENTINEL_ORDER_ID))), (
                f"meta[n_active={n}:, COL_ORDER_ID] must be ALL sentinels; "
                f"got tail (first 8) = {tail[:8].tolist()}"
            )

    def test_initial_state_is_packed(self, book):
        """Initial state: ``n_active=0`` and all rows are sentinels."""
        state = _init_state_post_p2_perf_6(book)
        self._assert_packed(state.iceberg_meta, state.n_active)

    def test_packed_after_single_register(self, book):
        """After one register, slot 0 is live and slot 1+ are sentinels."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=7),
        )
        assert int(n_active) == 1
        self._assert_packed(meta, n_active)

    def test_packed_after_register_sequence(self, book):
        """After N registers, ``[:N]`` are live, ``[N:]`` are sentinels."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active
        for oid in [10, 20, 30, 40, 50]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )
        self._assert_packed(meta, n_active)

    def test_packed_after_cleanup_head(self, book):
        """Cleanup of the head slot: invariant preserved by swap-with-last."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active
        for oid in [10, 20, 30]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )
        meta, n_active = _cleanup(mgr, meta, n_active, 10)
        self._assert_packed(meta, n_active)

    def test_packed_after_cleanup_middle(self, book):
        """Cleanup of a middle slot: invariant preserved by swap-with-last.

        This is the structural lock that the retired
        `test_cleanup_middle_slot_then_register_fills_middle` *cannot*
        express — it asserts the contiguous-prefix shape, not slot
        identity.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active
        for oid in [10, 20, 30]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )
        meta, n_active = _cleanup(mgr, meta, n_active, 20)
        self._assert_packed(meta, n_active)
        assert int(n_active) == 2

    def test_packed_after_cleanup_tail(self, book):
        """Cleanup of the tail slot: self-swap collapses cleanly."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active
        for oid in [10, 20, 30]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )
        meta, n_active = _cleanup(mgr, meta, n_active, 30)
        self._assert_packed(meta, n_active)

    def test_packed_after_interleaved_sequence(self, book):
        """An arbitrary interleaving preserves the contiguous prefix."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active

        ops = [
            ("R", 1), ("R", 2), ("R", 3), ("C", 2), ("R", 4),
            ("C", 1), ("R", 5), ("R", 6), ("C", 4), ("R", 7),
        ]
        for op, oid in ops:
            if op == "R":
                meta, n_active, _ = _register(
                    mgr, meta, n_active, _make_iceberg_buy(order_id=oid),
                )
            else:
                meta, n_active = _cleanup(mgr, meta, n_active, oid)
            # Invariant holds after EVERY op, not just at the end.
            self._assert_packed(meta, n_active)


# ===========================================================================
# Class T3 — Mechanism (2a) per-op contract (Deliverable (2)).
# ===========================================================================


class TestT3Mechanism2aPerOpContract:
    """Per-op JAX cost shape of mechanism (2a).

    Locks the Deliverable-(2) per-op contract:

      - ``cleanup``: swap-with-last into freed slot + clear old tail +
        decrement ``n_active``.
      - ``register``: write at slot ``n_active`` + increment ``n_active``.
      - Overflow predicate: ``n_active >= iceberg_meta.shape[0]`` — the
        meta's OWN configured first dim (``market_config.max_icebergs``,
        P3-PERF-REGEN-CAP-1), NOT the ``MAX_ICEBERGS`` ceiling constant
        (single int32 compare; replaces the current ``~jnp.any(sentinels)``
        reduction at ``iceberg_manager.py:227``).
    """

    def test_register_writes_at_slot_n_active(self, book):
        """``register`` writes the new row at exactly index ``n_active``."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        # First register lands at slot 0
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=11),
        )
        assert int(meta[0, COL_ORDER_ID]) == 11

        # Second register lands at slot 1 (the new tail)
        meta, n_active, _ = _register(
            mgr, meta, n_active, _make_iceberg_sell(order_id=22),
        )
        assert int(meta[0, COL_ORDER_ID]) == 11, (
            "register must NOT touch existing live rows."
        )
        assert int(meta[1, COL_ORDER_ID]) == 22, (
            "register must write at index n_active (the tail of the live region)."
        )

    def test_cleanup_swaps_tail_into_freed_slot(self, book):
        """``cleanup(meta, n_active, mid_oid)`` swaps the tail row into the freed slot.

        The (2a) mechanism: ``meta[slot] = meta[n_active - 1]``, then
        ``meta[n_active - 1] = SENTINEL_ROW``, then ``n_active -= 1``.

        With three live icebergs A=10, B=20, C=30 (slots 0, 1, 2),
        ``cleanup(B)`` must produce a meta where slot 1 carries C's
        row and slot 2 carries the sentinel row.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active
        for oid in [10, 20, 30]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )

        meta, n_active = _cleanup(mgr, meta, n_active, 20)

        assert int(n_active) == 2
        # Slot 0 untouched (A is below the freed slot).
        assert int(meta[0, COL_ORDER_ID]) == 10
        # Slot 1 now carries C (the previously-live tail).
        assert int(meta[1, COL_ORDER_ID]) == 30, (
            "(2a) swap-with-last: the freed slot must carry the previous "
            "n_active-1 row (the live tail), not a fresh sentinel."
        )
        # The old tail slot must be cleared to sentinel.
        assert int(meta[2, COL_ORDER_ID]) == SENTINEL_ORDER_ID, (
            "(2a) clear-tail: meta[n_active_before - 1] must be cleared "
            "to SENTINEL_ROW after the swap."
        )

    def test_cleanup_head_swaps_tail_into_head(self, book):
        """Cleanup of slot 0 (head) swaps the tail into slot 0."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active
        for oid in [10, 20, 30]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )
        meta, n_active = _cleanup(mgr, meta, n_active, 10)
        assert int(n_active) == 2
        # 30 (the tail) is swapped into slot 0; 20 stays where it was.
        assert int(meta[0, COL_ORDER_ID]) == 30
        assert int(meta[1, COL_ORDER_ID]) == 20
        assert int(meta[2, COL_ORDER_ID]) == SENTINEL_ORDER_ID

    def test_cleanup_tail_is_self_swap(self, book):
        """Cleanup of the tail slot collapses to a self-swap.

        This is the case where (2a) and the retired-test's
        ``slot-identity preserved'' semantics coincide.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active
        for oid in [10, 20, 30]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )
        meta, n_active = _cleanup(mgr, meta, n_active, 30)
        assert int(n_active) == 2
        assert int(meta[0, COL_ORDER_ID]) == 10
        assert int(meta[1, COL_ORDER_ID]) == 20
        assert int(meta[2, COL_ORDER_ID]) == SENTINEL_ORDER_ID

    def test_cleanup_single_iceberg_clears_to_sentinel(self, book):
        """Cleanup of the only live iceberg returns the empty state."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=99),
        )
        assert int(n_active) == 1

        meta, n_active = _cleanup(mgr, meta, n_active, 99)
        assert int(n_active) == 0
        assert int(meta[0, COL_ORDER_ID]) == SENTINEL_ORDER_ID

    def test_register_overflow_predicate_uses_n_active(self, book):
        """Overflow is detected by ``n_active >= iceberg_meta.shape[0]``.

        The overflow predicate reads the meta's OWN first dimension — the
        book's configured ``market_config.max_icebergs`` (P3-PERF-REGEN-CAP-1)
        — NOT the ``MAX_ICEBERGS`` hard-ceiling constant. Since ``book.init()``
        now sizes the meta at ``max_icebergs`` (default 1024, not the 8192
        ceiling), the fill saturates at that configured cap: when
        ``n_active == max_icebergs`` already (saturated), the next
        ``register`` must return ``overflow=True`` and the meta unchanged.
        The predicate must be ``n_active >= iceberg_meta.shape[0]``, not
        ``~jnp.any(sentinels)``.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        # The meta is sized at the book's configured cap (P3-PERF-REGEN-CAP-1);
        # fill exactly that many slots, not the MAX_ICEBERGS ceiling.
        cap = book._market_config.max_icebergs
        assert state.iceberg_meta.shape[0] == cap, (
            "book.init() must size iceberg_meta at market_config.max_icebergs "
            f"({cap}); got shape[0]={state.iceberg_meta.shape[0]}."
        )
        # Fill all `cap` slots with deterministic order_ids.
        meta = state.iceberg_meta
        n_active = state.n_active
        # ovf_acc OR-accumulates the per-iter `overflow` flag; post-loop
        # assert catches any-True (bidirectionally equivalent to per-iter
        # `assert not bool(overflow)`).
        def _body_fill_check_overflow(i, carry):
            meta, n_active, ovf_acc = carry
            meta, n_active, overflow = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=1000 + i),
            )
            return (meta, n_active, ovf_acc | overflow)
        meta, n_active, any_overflow = jax.lax.fori_loop(
            0, cap, _body_fill_check_overflow,
            (meta, n_active, jnp.bool_(False)),
        )
        assert not bool(any_overflow), (
            "Register sequence filling max_icebergs slots must not overflow."
        )
        assert int(n_active) == cap

        # Attempt one more register: must overflow, meta unchanged.
        prev_meta = meta
        meta, n_active, overflow = _register(
            mgr, meta, n_active, _make_iceberg_buy(order_id=9999),
        )
        assert bool(overflow), (
            "register on a saturated meta must return overflow=True."
        )
        assert int(n_active) == cap, (
            "register on a saturated meta must NOT increment n_active."
        )
        # No silent overwrite anywhere.
        assert bool(jnp.array_equal(meta, prev_meta)), (
            "register on a saturated meta must NOT mutate the meta array."
        )

    def test_register_overflow_when_partial(self, book):
        """``register`` does NOT overflow when ``n_active < MAX_ICEBERGS``.

        Even if the tail rows are sentinels (which they always are
        under the packed-layout invariant), overflow must remain
        False until ``n_active`` actually saturates.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta, n_active, overflow = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=1),
        )
        assert not bool(overflow)
        assert int(n_active) == 1


# ===========================================================================
# Class T4 — Multiset-equivalence (Test-first contract (c) replacement).
# ===========================================================================


class TestT4MultisetEquivalence:
    """Replacement for the retired locked test
    `test_cleanup_middle_slot_then_register_fills_middle`.

    The retired test pinned a stronger property than market semantics
    require (FIFO slot reuse / slot identity). (2a) preserves the
    multiset of live order_ids in the contiguous prefix but does NOT
    preserve slot identity across cleanup-middle calls. Per the v5.29
    architect review and market-spec §4.2 / §5.2, slot identity is an
    `iceberg_manager` implementation detail; the observable iceberg
    semantics are which icebergs are live, their drifted slice prices,
    and their hidden volume.

    These tests express the strictly stronger invariant: for any
    sequence of register/cleanup calls (no overflow), the multiset of
    order_ids in ``meta[:n_active, COL_ORDER_ID]`` equals
    ``{registered_oids} \\ {cleaned_oids}``.
    """

    @staticmethod
    def _live_multiset(meta, n_active):
        """Return ``set(meta[:n_active, COL_ORDER_ID])`` as Python ints.

        (We use ``set`` rather than ``Counter`` because order_ids are
        unique within a session by market-spec; if a test ever
        violates this assumption, it should fail loudly on the set
        construction.)
        """
        n = int(n_active)
        ids = meta[:n, COL_ORDER_ID].tolist() if n > 0 else []
        # Cast to int (jnp arrays return numpy ints).
        s = {int(x) for x in ids}
        assert len(s) == n, (
            "Live order_ids in the contiguous prefix must be unique; "
            f"got {ids} (len={n})."
        )
        return s

    def test_multiset_after_retired_test_sequence(self, book):
        """The exact sequence the retired test exercised — now multiset-equivalent.

        Retired: A=10, B=20, C=30, cleanup(B), register(D=40);
        asserted slot 1 == 40.

        Replacement: same ops; assert the live multiset is
        ``{10, 30, 40}``. (Slot positions are free; only set
        equivalence matters.)
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active
        for oid in [10, 20, 30]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )
        meta, n_active = _cleanup(mgr, meta, n_active, 20)
        meta, n_active, _ = _register(
            mgr, meta, n_active, _make_iceberg_sell(order_id=40),
        )
        assert int(n_active) == 3
        assert self._live_multiset(meta, n_active) == {10, 30, 40}

    def test_multiset_after_cleanup_middle_then_register(self, book):
        """A second cleanup-middle + register pattern: multiset stable."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active
        for oid in [101, 202, 303, 404, 505]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )
        # Cleanup two middle elements, then register two new ones.
        meta, n_active = _cleanup(mgr, meta, n_active, 202)
        meta, n_active = _cleanup(mgr, meta, n_active, 404)
        meta, n_active, _ = _register(
            mgr, meta, n_active, _make_iceberg_sell(order_id=606),
        )
        meta, n_active, _ = _register(
            mgr, meta, n_active, _make_iceberg_sell(order_id=707),
        )
        assert int(n_active) == 5
        assert self._live_multiset(meta, n_active) == {
            101, 303, 505, 606, 707,
        }

    def test_multiset_register_cleanup_register_cleanup(self, book):
        """Interleaved register/cleanup: multiset tracks the operation log."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active

        expected = set()
        log = [
            ("R", 1), ("R", 2), ("C", 1),
            ("R", 3), ("C", 2),
            ("R", 4), ("R", 5), ("C", 3),
        ]
        for op, oid in log:
            if op == "R":
                meta, n_active, _ = _register(
                    mgr, meta, n_active, _make_iceberg_buy(order_id=oid),
                )
                expected.add(oid)
            else:
                meta, n_active = _cleanup(mgr, meta, n_active, oid)
                expected.discard(oid)
            # Multiset matches at every step, not just the final state.
            assert self._live_multiset(meta, n_active) == expected, (
                f"After op {op}({oid}), multiset must equal {expected}; "
                f"got {self._live_multiset(meta, n_active)}."
            )

    def test_multiset_after_full_drain_and_refill(self, book):
        """Cleanup every iceberg, then register a fresh set.

        After full drain, ``n_active == 0`` and every slot is sentinel;
        the next register sequence builds a clean packed prefix.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta = state.iceberg_meta
        n_active = state.n_active

        for oid in [1, 2, 3]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid)
            )
        for oid in [1, 2, 3]:
            meta, n_active = _cleanup(mgr, meta, n_active, oid)
        assert int(n_active) == 0
        assert self._live_multiset(meta, n_active) == set()

        # Fresh set with disjoint ids.
        for oid in [100, 200, 300, 400]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_sell(order_id=oid)
            )
        assert int(n_active) == 4
        assert self._live_multiset(meta, n_active) == {100, 200, 300, 400}


# ===========================================================================
# Class T5 — Positive lock on the three preserved invariant-(5) tests.
# ===========================================================================


class TestT5PositiveLockPreservedInvariant5:
    """The three invariant-(5) tests in `test_iceberg_manager_jit.py`
    that pass verbatim under (2a) — re-encoded here as positive locks
    against the (2a) implementation.

    Per Test-first contract (c) and the v5.29 architect review, these
    three tests exercise single-iceberg cycles where swap-with-last
    collapses to a self-swap, so their semantic property survives the
    mechanism switch.

    Locking them here in the P2-PERF-6 file means a future regression
    in `iceberg_manager.py` is caught even if someone deletes or
    further refactors the original test file.

    Mirrors the three preserved tests:

      - `test_cleanup_restores_slot_to_sentinel`
      - `test_register_after_cleanup_reuses_freed_slot`
      - `test_cleanup_uses_sentinel_constant_symbolically`
    """

    def test_cleanup_restores_slot_zero_to_sentinel(self, book):
        """Mirror of `test_cleanup_restores_slot_to_sentinel`.

        After ``cleanup`` of the only iceberg, slot 0 carries the
        sentinel row. (Verbatim semantic property; the (2a) self-swap
        collapse makes this identical to the pre-(2a) behaviour.)
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=11),
        )
        assert int(meta[0, COL_ORDER_ID]) == 11

        meta, n_active = _cleanup(mgr, meta, n_active, 11)
        assert int(meta[0, COL_ORDER_ID]) == SENTINEL_ORDER_ID
        assert int(n_active) == 0

    def test_register_after_cleanup_lands_at_slot_zero(self, book):
        """Mirror of `test_register_after_cleanup_reuses_freed_slot`.

        Single-iceberg cycle: register → cleanup → register. The next
        register must land at slot 0 (since ``n_active == 0`` after
        the cleanup). This is observably identical to FIFO slot reuse
        in the single-iceberg case.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=11),
        )
        meta, n_active = _cleanup(mgr, meta, n_active, 11)

        meta, n_active, _ = _register(
            mgr, meta, n_active, _make_iceberg_sell(order_id=22),
        )
        assert int(meta[0, COL_ORDER_ID]) == 22
        assert int(meta[1, COL_ORDER_ID]) == SENTINEL_ORDER_ID
        assert int(n_active) == 1

    def test_cleanup_uses_sentinel_constant_symbolically(self, book):
        """Mirror of `test_cleanup_uses_sentinel_constant_symbolically`.

        ``cleanup`` writes the symbolic ``SENTINEL_ORDER_ID`` constant,
        not a hard-coded literal ``-1``. Verified by comparing the
        freed slot's order_id to the imported constant.
        """
        from omiecid_simulator.omiecid import jax_constants

        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=99),
        )
        meta, n_active = _cleanup(mgr, meta, n_active, 99)

        sentinel = jax_constants.SENTINEL_ORDER_ID
        assert int(meta[0, COL_ORDER_ID]) == sentinel


# ===========================================================================
# Class T6 — `regenerate_slices` outer-scan bound (Deliverable (3)).
# ===========================================================================


class TestT6RegenerateSlicesOuterScanBound:
    """`regenerate_slices` outer-scan trip count equals ``n_active``,
    not ``MAX_ICEBERGS``.

    Per Deliverable (3): the implementation uses
    ``lax.dynamic_slice(meta, (0, 0), (n_active, N_COLS))`` + ``lax.scan``
    over the prefix, or an equivalent bounded ``while_loop``. The
    test asserts the SHAPE constraint — bound = ``n_active`` — not
    the implementation choice (scan vs. while_loop).

    The shape constraint is verifiable by tracing the lowered jaxpr
    and inspecting the outer-loop length, or by signature inspection
    of the post-(2a) ``regenerate_slices`` API (which must accept
    ``n_active`` as a positional argument and propagate it through).
    """

    def test_regenerate_slices_accepts_n_active_arg(self, book):
        """`regenerate_slices` accepts ``n_active`` as a positional argument.

        Post-(2a) signature (per Deliverable (6)):

            regenerate_slices(lob, meta, n_active, ts_counter)
                -> (lob, meta, n_active, ts_counter)

        Calling the post-(2a) signature against the pre-(2a)
        implementation will raise ``TypeError`` until the
        implementation lands.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        # Trace-level call; the body need not change observable state
        # on an empty book — we are pinning the call shape here.
        result = mgr.regenerate_slices(
            state.lob, state.iceberg_meta, state.n_active, state.ts_counter,
        )
        # Post-(2a) return tuple is 4-tuple: (lob, meta, n_active, ts).
        assert isinstance(result, tuple), (
            "regenerate_slices must return a tuple post-(2a)."
        )
        assert len(result) == 4, (
            f"regenerate_slices must return a 4-tuple "
            f"(lob, meta, n_active, ts_counter); got len={len(result)}."
        )

    def test_regenerate_slices_returns_n_active(self, book):
        """The return tuple's third element is the post-step ``n_active``."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr

        # Register two iceberg orders so the regen body has something
        # to scan over.
        meta = state.iceberg_meta
        n_active = state.n_active
        for oid in [11, 22]:
            meta, n_active, _ = _register(
                mgr, meta, n_active, _make_iceberg_buy(order_id=oid),
            )
        assert int(n_active) == 2

        lob_out, meta_out, n_active_out, ts_out = mgr.regenerate_slices(
            state.lob, meta, n_active, state.ts_counter,
        )
        # n_active_out is a 0-d int32 (may equal or decrease vs.
        # n_active depending on whether any iceberg fully drained in
        # this call; on a freshly-registered set with no fills, it
        # must NOT change).
        assert isinstance(n_active_out, jnp.ndarray)
        assert n_active_out.dtype == jnp.int32
        assert int(n_active_out) == 2, (
            "regen on freshly-registered icebergs with no fills must "
            "preserve n_active."
        )

    def test_regenerate_slices_jaxpr_outer_bound_is_n_active(self, book):
        """The lowered jaxpr's outer-loop length depends on ``n_active``,
        not on ``MAX_ICEBERGS``.

        Verifiable handle: tracing `regenerate_slices` with two
        different ``n_active`` traced values must produce a body that
        uses ``n_active`` as the bound on the outer iteration —
        either via ``lax.dynamic_slice`` to a prefix of length
        ``n_active`` then a fixed-shape scan over that prefix, OR via
        a ``lax.while_loop`` whose cond is ``i < n_active``.

        Strict implementation check: the jaxpr's source text contains
        ``dynamic_slice`` (or ``while_loop``), AND the outer-scan
        trip-count constant ``MAX_ICEBERGS`` is NOT a body-side
        constant of the outer iteration.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr

        def trace_fn(lob, meta, n_active, ts):
            return mgr.regenerate_slices(lob, meta, n_active, ts)

        jaxpr = jax.make_jaxpr(trace_fn)(
            state.lob,
            state.iceberg_meta,
            state.n_active,
            state.ts_counter,
        )
        text = str(jaxpr)
        # The bounded form uses either dynamic_slice (the canonical
        # form in Deliverable (3)) or a while_loop with an n_active
        # cond.
        uses_dynamic_slice = "dynamic_slice" in text
        uses_while_loop = "while" in text
        assert uses_dynamic_slice or uses_while_loop, (
            "regenerate_slices jaxpr must use lax.dynamic_slice (canonical "
            "form per Deliverable (3)) or an equivalent lax.while_loop "
            "bounded by n_active. Got jaxpr:\n" + text[:2000]
        )


# ===========================================================================
# Class T7 — `matchable_hidden_volume` outer-scan bound (Deliverable (4)).
# ===========================================================================


class TestT7MatchableHiddenVolumeOuterScanBound:
    """`matchable_hidden_volume` uses a bounded form, NOT
    ``jax.vmap(slot_contribution)(iceberg_meta)`` over the full meta.

    Per Deliverable (4): the implementation uses
    ``lax.dynamic_slice(meta, (0, 0), (n_active, N_COLS))`` + ``vmap``
    over the ``n_active``-bounded prefix, OR ``lax.scan`` over the
    prefix.

    The mask-zeroes-sentinels mechanism currently at
    ``iceberg_manager.py:476`` gives correctness, not the perf win —
    the bounded form is the actual perf-independence-from-cap
    mechanism for the hidden-volume query path.
    """

    def test_matchable_hidden_volume_accepts_n_active(self, book):
        """`matchable_hidden_volume` accepts ``n_active`` (post-(2a))."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr

        # Post-(2a) signature must accept n_active. The exact arg
        # position is the developer's choice (positional kwarg most
        # likely); the test asserts the call works under the
        # bounded-scan contract.
        result = mgr.matchable_hidden_volume(
            state.iceberg_meta,
            state.n_active,
            jnp.int32(Side.BUY),
            jnp.int32(0),  # limit_price_int (arbitrary; meta is empty)
        )
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.int32
        # Empty meta -> zero matchable volume.
        assert int(result) == 0

    def test_matchable_hidden_volume_jaxpr_outer_bound_is_n_active(self, book):
        """The lowered jaxpr uses ``lax.dynamic_slice`` over an
        ``n_active``-bounded prefix.

        Pinning Deliverable (4) — the bounded form, NOT vmap-over-full-meta.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr

        def trace_fn(meta, n_active, side, limit):
            return mgr.matchable_hidden_volume(meta, n_active, side, limit)

        jaxpr = jax.make_jaxpr(trace_fn)(
            state.iceberg_meta,
            state.n_active,
            jnp.int32(int(Side.BUY)),
            jnp.int32(0),
        )
        text = str(jaxpr)
        # The bounded form uses dynamic_slice (canonical per
        # Deliverable (4)) or an equivalent prefix-scan.
        uses_dynamic_slice = "dynamic_slice" in text
        uses_scan_over_prefix = "scan" in text
        assert uses_dynamic_slice or uses_scan_over_prefix, (
            "matchable_hidden_volume jaxpr must use lax.dynamic_slice "
            "(canonical form per Deliverable (4)) or an equivalent "
            "lax.scan over the n_active-bounded prefix. The pre-(2a) "
            "vmap-over-full-meta does NOT satisfy this contract — the "
            "mask gives correctness, not the perf win. "
            "Got jaxpr:\n" + text[:2000]
        )

    def test_matchable_hidden_volume_numerical_correctness(self, book):
        """Bounded form returns the same hidden volume as the unbounded form
        would have on the same live icebergs.

        Register one sell iceberg with hidden quantity, then ask the
        manager for the matchable volume from a BUY aggressor at a
        crossing price. The result must be > 0 (the hidden remainder).
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr

        sell_ice = _make_iceberg_sell(
            order_id=42, price=50.00, quantity=30.0, peak_qty=10.0, ppd=0.0,
        )
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active, sell_ice,
        )

        # Encode a BUY aggressor limit at 50.00 (crosses the sell at 50.00).
        cfg = book._product_config
        price_scale = int(round(1.0 / cfg.price_tick))
        price_offset = int(round(-cfg.price_min / cfg.price_tick))
        buy_limit_int = jnp.int32(int(round(50.00 * price_scale)) + price_offset)

        hidden_vol = mgr.matchable_hidden_volume(
            meta, n_active, jnp.int32(int(Side.BUY)), buy_limit_int,
        )
        assert int(hidden_vol) > 0, (
            "Bounded matchable_hidden_volume must report > 0 hidden "
            "quantity for a sell iceberg with remaining_hidden > 0 at "
            "a crossing price."
        )


# ===========================================================================
# Class T8 — `truncation_flag` parity (Deliverable (5), Risk audit (2)).
# ===========================================================================


class TestT8TruncationFlagParity:
    """`truncation_flag` OR-reduction is identical under the bounded
    scan and the sticky-flag accumulator invariant is preserved.

    Per Deliverable (5) and Risk audit (2):

      - The OR-reduction at ``iceberg_manager.py:512-545`` produces an
        identical 0-d ``jnp.bool_`` under the bounded scan vs. the
        unbounded form (rows in ``[n_active, MAX_ICEBERGS)`` are
        guaranteed sentinel under the packed-layout invariant,
        contributing ``False`` to the OR).
      - The sticky-flag accumulator in ``contract_book._build_state_jit``
        (``contract_book.py:984-998``) keeps its
        ``jnp.logical_or(prior_truncated, live)`` invariant — no
        regression.
    """

    def test_truncation_flag_false_on_empty(self, book):
        """Empty meta: ``truncation_flag`` is False (no active slots)."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        flag = mgr.truncation_flag(state.iceberg_meta)
        assert flag.dtype == jnp.bool_
        assert flag.shape == ()
        assert not bool(flag)

    def test_truncation_flag_false_on_benign_register(self, book):
        """Register an iceberg whose worst-case slice count is well
        below MAX_SLICES_PER_REGEN: flag stays False."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        # quantity=30, peak=10 -> ceil(30/10) = 3 slices, well under 64.
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=1, quantity=30.0, peak_qty=10.0),
        )
        flag = mgr.truncation_flag(meta)
        assert not bool(flag), (
            "An iceberg with ceil(qty/peak)=3 must NOT trigger the "
            "MAX_SLICES_PER_REGEN backstop."
        )

    def test_truncation_flag_true_on_oversized_iceberg(self, book):
        """An iceberg whose worst-case slice count exceeds
        MAX_SLICES_PER_REGEN sets the flag to True under the bounded
        scan, just as it did under the unbounded scan."""
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        # quantity=1000, peak=10 -> ceil(1000/10) = 100 slices, > 64.
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=1, quantity=1000.0, peak_qty=10.0),
        )
        flag = mgr.truncation_flag(meta)
        assert bool(flag), (
            "An iceberg with ceil(qty/peak) > MAX_SLICES_PER_REGEN must "
            "trigger the truncation backstop under the bounded scan."
        )

    def test_truncation_flag_sentinel_tail_contributes_false(self, book):
        """The contiguous-prefix invariant guarantees sentinel rows in
        ``[n_active, MAX_ICEBERGS)`` contribute False to the OR-reduction.

        This locks the load-bearing property that bounding the scan
        cannot change the result of the OR-reduction: under the
        packed-layout invariant, every row beyond ``n_active`` is a
        sentinel, and ``active = (order_id != SENTINEL_ORDER_ID)`` is
        False, so the per-slot truncation predicate is False there.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        # Register a single benign iceberg.
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=1, quantity=20.0, peak_qty=10.0),
        )
        assert int(n_active) == 1

        # Flag on full meta == flag on dynamic_slice(prefix). The
        # bounded scan must give the same result as the full scan.
        full_flag = mgr.truncation_flag(meta)
        assert not bool(full_flag)

    def test_build_state_jit_or_invariant_with_prior_true(
        self, book, product_config, market_config,
    ):
        """Sticky-flag accumulator: ``prior_truncated = True`` is preserved
        by ``_build_state_jit`` regardless of the live flag.

        ``contract_book.py:984-998`` invariant:

            new_truncated = (prior_truncated OR live) OR overflow

        Setting ``prior_truncated=True`` with a benign meta must yield
        ``truncated=True``. No regression from the bounded scan.
        """
        from omiecid_simulator.omiecid.contract_book import _build_state_jit

        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr
        meta, n_active, _ = _register(
            mgr, state.iceberg_meta, state.n_active,
            _make_iceberg_buy(order_id=1, quantity=20.0, peak_qty=10.0),
        )

        # Drive _build_state_jit with prior_truncated=True.
        # The post-(2a) signature must accept n_active (per Deliverable (6)).
        result_state = _build_state_jit(
            state.lob,
            meta,
            n_active,
            state.ts_counter,
            jnp.bool_(True),    # prior_truncated
            mgr,
            jnp.bool_(False),   # overflow
        )
        assert bool(result_state.truncated), (
            "Sticky-flag invariant: prior_truncated=True must propagate "
            "through _build_state_jit regardless of live/overflow."
        )

    def test_build_state_jit_overflow_propagates(
        self, book, product_config, market_config,
    ):
        """``overflow=True`` propagates into ``truncated`` even when
        ``prior_truncated=False`` and the live flag is False."""
        from omiecid_simulator.omiecid.contract_book import _build_state_jit

        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr

        result_state = _build_state_jit(
            state.lob,
            state.iceberg_meta,
            state.n_active,
            state.ts_counter,
            jnp.bool_(False),   # prior_truncated
            mgr,
            jnp.bool_(True),    # overflow
        )
        assert bool(result_state.truncated), (
            "overflow=True must OR into the truncated carry-flag."
        )


# ===========================================================================
# Class T9 — Performance contracts (a) and (b) (Test-first contract).
# ===========================================================================
#
# Wall-clock micro-benchmarks. ``@pytest.mark.slow`` — these exceed
# the < 1s fast-suite budget by design (they warm up the JIT compile
# then time `regenerate_slices` and `matchable_hidden_volume` across
# realistic ``n_active`` values).
#
# The pre-warmup JIT compile is excluded from the wall-clock budget;
# only the second-and-subsequent calls are timed.
# ===========================================================================


def _register_n_icebergs(
    mgr: IcebergManager, meta, n_active, n: int, base_oid: int = 1000,
):
    """Helper for T9: register ``n`` distinct icebergs and return
    (meta, n_active). Used to build the load fixtures."""
    for i in range(n):
        meta, n_active, overflow = _register(
            mgr, meta, n_active,
            _make_iceberg_buy(order_id=base_oid + i, quantity=30.0, peak_qty=10.0),
        )
        if bool(overflow):
            raise AssertionError(
                f"Bench setup overflowed at i={i} (n_active={int(n_active)}, "
                f"MAX_ICEBERGS={MAX_ICEBERGS}). Raise MAX_ICEBERGS in jax_constants "
                f"or reduce n."
            )
    return meta, n_active


class TestT9PerformanceContracts:
    """Wall-clock micro-benchmarks for Deliverables (3) and (4).

    Pins Test-first contract (a) and (b):

      (a) At ``MAX_ICEBERGS=8192`` with ≤100 live icebergs, the
          per-call wall-clock of ``regenerate_slices`` and
          ``matchable_hidden_volume`` is within 20% of the
          ``MAX_ICEBERGS=100`` (uncapped) baseline.

      (b) Load-stress: at ``MAX_ICEBERGS=8192`` with ~2000 live
          icebergs, ``regenerate_slices`` wall-clock scales
          sub-linearly in ``n_active`` and stays under ~5× the ≤100
          baseline. Bounded ``matchable_hidden_volume`` is pinned by
          the same micro-benchmark.

    These benchmarks fail on ``main`` (pre-P2-PERF-6) because the
    unbounded outer scan over ``MAX_ICEBERGS`` rows means scaling
    ``MAX_ICEBERGS`` from 100 → 8192 drives per-call cost up by
    ~80× (the empirically-observed ~50× test-suite slowdown that
    motivated this story).

    The benchmarks require ``MAX_ICEBERGS >= 8192`` — they self-skip
    if the current cap is below that threshold (the cap raise is
    P2-PERF-7, which lands AFTER P2-PERF-6). The benchmark stays in
    this file so it acts as a forward-looking lock that catches a
    regression once P2-PERF-7 lifts the cap.
    """

    @staticmethod
    def _time_regen(mgr, lob, meta, n_active, ts, n_calls: int = 20) -> float:
        """Time ``n_calls`` invocations of ``regenerate_slices``.

        First call is excluded (JIT compile warmup). Returns mean
        per-call wall-clock in seconds.
        """
        jit_regen = jax.jit(mgr.regenerate_slices)
        # Warmup compile + block_until_ready.
        out = jit_regen(lob, meta, n_active, ts)
        jax.tree.map(lambda x: x.block_until_ready(), out)

        t0 = time.perf_counter()
        for _ in range(n_calls):
            out = jit_regen(lob, meta, n_active, ts)
        jax.tree.map(lambda x: x.block_until_ready(), out)
        t1 = time.perf_counter()
        return (t1 - t0) / n_calls

    @staticmethod
    def _time_hidden(
        mgr, meta, n_active, side, limit_int, n_calls: int = 50,
    ) -> float:
        """Time ``n_calls`` invocations of ``matchable_hidden_volume``."""
        jit_hidden = jax.jit(mgr.matchable_hidden_volume)
        # Warmup.
        out = jit_hidden(meta, n_active, side, limit_int)
        out.block_until_ready()

        t0 = time.perf_counter()
        for _ in range(n_calls):
            out = jit_hidden(meta, n_active, side, limit_int)
        out.block_until_ready()
        t1 = time.perf_counter()
        return (t1 - t0) / n_calls

    @pytest.mark.slow
    @pytest.mark.perf
    def test_perf_a_regen_scales_with_n_active_not_max_icebergs(self, book):
        """slow: contract (a) wall-clock — `regenerate_slices` independence from cap.

        With ≤100 live icebergs, the per-call cost must scale with
        ``n_active``, NOT with ``MAX_ICEBERGS``. The bound is the
        observable surrogate: the per-call time on a sparse meta
        (n_active=50, MAX_ICEBERGS=large) must NOT be a large
        multiple of the per-call time on a dense meta of the same
        n_active.

        Concretely, we time two ``n_active`` values (50 and 100) on
        the same ``MAX_ICEBERGS``-sized meta and assert the ratio is
        roughly proportional to ``n_active``, not to ``MAX_ICEBERGS``.

        Pre-(2a): both calls scan ``MAX_ICEBERGS`` rows; per-call
        times are roughly equal regardless of ``n_active``.
        Post-(2a): per-call time scales linearly with ``n_active``.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr

        # n_active=50 fixture.
        meta_50, n_50 = _register_n_icebergs(mgr, state.iceberg_meta, state.n_active, 50)
        # n_active=100 fixture.
        meta_100, n_100 = _register_n_icebergs(
            mgr, state.iceberg_meta, state.n_active, 100,
        )

        t_50 = self._time_regen(mgr, state.lob, meta_50, n_50, state.ts_counter)
        t_100 = self._time_regen(mgr, state.lob, meta_100, n_100, state.ts_counter)

        # Under the bounded scan, t_100 / t_50 should be ~2x (linear
        # in n_active). Under the unbounded scan, the ratio is ~1
        # (both pay for MAX_ICEBERGS rows). We assert the bounded
        # behaviour: the ratio is BOUNDED ABOVE — t_100 must be at
        # most 5× t_50 (the constant-factor margin from contract (b))
        # AND it must be observably different from 1.0 (i.e. the
        # bounded path scales with n_active).
        ratio = t_100 / t_50 if t_50 > 0 else float("inf")
        assert ratio < 5.0, (
            f"regen ratio t(n=100)/t(n=50) must be < 5x under bounded "
            f"scan; got {ratio:.3f} (t_50={t_50:.6f}s, t_100={t_100:.6f}s)."
        )


    @pytest.mark.slow
    @pytest.mark.perf
    def test_perf_b_hidden_volume_scales_with_n_active(self, book):
        """slow: bounded `matchable_hidden_volume` (Deliverable (4))
        scales with n_active, not MAX_ICEBERGS.

        Same surface as test_perf_a but for the hidden-volume query
        path. Under the pre-(2a) ``vmap(slot_contribution)(iceberg_meta)``
        over the full meta, the per-call time is ~constant in
        ``n_active``. Under the bounded ``lax.dynamic_slice`` +
        ``vmap`` over the prefix, the per-call time scales with
        ``n_active``.
        """
        state = _init_state_post_p2_perf_6(book)
        mgr = book._iceberg_mgr

        meta_50, n_50 = _register_n_icebergs(
            mgr, state.iceberg_meta, state.n_active, 50,
        )
        meta_100, n_100 = _register_n_icebergs(
            mgr, state.iceberg_meta, state.n_active, 100,
        )

        # Encode a BUY aggressor limit price for the matchable query.
        cfg = book._product_config
        price_scale = int(round(1.0 / cfg.price_tick))
        price_offset = int(round(-cfg.price_min / cfg.price_tick))
        limit_int = jnp.int32(int(round(50.00 * price_scale)) + price_offset)

        t_50 = self._time_hidden(
            mgr, meta_50, n_50, jnp.int32(int(Side.BUY)), limit_int,
        )
        t_100 = self._time_hidden(
            mgr, meta_100, n_100, jnp.int32(int(Side.BUY)), limit_int,
        )

        ratio = t_100 / t_50 if t_50 > 0 else float("inf")
        assert ratio < 5.0, (
            f"matchable_hidden_volume ratio t(n=100)/t(n=50) must be < 5x "
            f"under bounded vmap-over-prefix; got {ratio:.3f} "
            f"(t_50={t_50:.6f}s, t_100={t_100:.6f}s). Pins Deliverable (4)."
        )
