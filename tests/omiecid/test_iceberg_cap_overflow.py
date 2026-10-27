"""Locked behavioral-contract tests for P2-HOTFIX-iceberg-cap-overflow.

These tests pin the correctness-machinery deliverables of the contracted
hotfix scope (per `docs/omiecid/sprint-status.md` §"Scope contraction —
defer cap raise + demo regen post-H3 (orchestrator-elected, 2026-05-04)"
and §"Test-contract clarification (orchestrator-elected, option c)").
They are authored before the implementation lands, so most assertions
WILL FAIL until the `register → (updated_meta, overflow_bool)` signature,
the `book_depth <= MAX_ICEBERGS` `__init__` assertion, and the
`_build_state` overflow-OR wiring all ship — this is by design.

Locked invariants encoded in this file (numbering matches the agile-plan
story-card "Test-first contract" row, after the 2026-05-04 scope
contraction):

  (1) Anti-regression for the silent-eviction surface — the four binding
      assertions on the post-call `iceberg_meta` returned from a
      saturated-`+1` `register` call (option-c clarification, sprint-status):
        (a) the returned `overflow` flag is True;
        (b) non-sentinel cardinality of `meta_post` equals `MAX_ICEBERGS`;
        (c) sorted/sentinel-filtered surviving order_ids in `meta_post`
            equal — as multisets — the pre-call sorted/sentinel-filtered
            ids, after excluding the new 101st `order_id` exactly once
            (equality, NOT superset);
        (d) the new 101st `order_id` is absent from `meta_post[:, 0]`.
      Slot of the surviving 100 ids is intentionally NOT asserted.

  (3) `book_depth <= MAX_ICEBERGS` `__init__` assertion fires (Python-side,
      trace-time): constructing `OMIEMarketConfig(book_depth =
      MAX_ICEBERGS + 1)` and then `ContractBook(...)` raises.

  (4) `OMIEBookState.truncated` discriminates slot-overflow from
      slice-overflow — independent triggers, independent assertions.
      Slice-overflow path uses the existing P2-H1 `MAX_SLICES_PER_REGEN`
      Path B mechanism. Slot-overflow path uses the new
      `register`-side overflow that propagates through `_build_state`.

Tests EXPLICITLY NOT authored here (per the scope contraction —
deferred to `P2-HOTFIX-iceberg-cap-overflow-FOLLOWUP`):
  (2) Faithful `_hidden_total` on canonical 2026-01-12 cid=81 session.
  (5) Demo HTML oracle.

Test-first contract item (6) — "MAX_SLICES_PER_REGEN parity" — is
satisfied **negatively**: the existing P2-H1 locked tests in
`tests/omiecid/test_iceberg_manager_jit.py` (esp. invariant 6,
`TestInvariant6TruncationBackstop`) already cover the slice-overflow
mechanism and must remain green at the new `register` signature. This
file does not duplicate that coverage. Item (6) compliance is
verified by the orchestrator running the full OMIE suite green.

Architectural references (read at authoring time):
  - docs/omiecid/agile-plan.md (story P2-HOTFIX-iceberg-cap-overflow card,
    Test-first contract row, with 2026-05-04 scope-contraction strikes
    on items (2) and (5)).
  - docs/omiecid/sprint-status.md §"P2-HOTFIX-iceberg-cap-overflow —
    Architecture Review", §"Test-contract clarification
    (orchestrator-elected, option c)", §"Scope contraction — defer cap
    raise + demo regen post-H3", §"Demo fully out of scope on this branch".
  - omiecid_simulator/omiecid/iceberg_manager.py (current `register`
    signature; mechanism-locked to `(updated_meta, overflow_bool)`
    post-fix).
  - omiecid_simulator/omiecid/contract_book.py (`OMIEBookState.truncated` is
    the sticky surface; no new sibling field).
  - omiecid_simulator/omiecid/jax_constants.py §B1 (MAX_ICEBERGS == 100 on
    this branch; cap raise deferred to FOLLOWUP).

Mechanism-locked invariants (binding from sprint-status):
  - `register` returns the 2-tuple `(updated_meta, overflow_bool)`.
  - The sticky surface is `OMIEBookState.truncated` (not a new field).
  - `MAX_ICEBERGS` stays at 100 on this branch.

Test discipline:
  - Tests that invoke `jax.jit` carry `@pytest.mark.jit` (T1.1).
  - Saturated meta is constructed directly via `jnp.array` manipulation
    rather than 100 successive `register` calls — the synthetic fixture
    approach matches the < 1s fast-suite budget per T1.1 guidance.
"""

from __future__ import annotations

import collections
import dataclasses

import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, OMIEBookState
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_PEAK_QTY,
    COL_REMAINING_HIDDEN,
    COL_SIDE,
    COL_SLICE_COUNT,
    IcebergManager,
    N_COLS,
)
from omiecid_simulator.omiecid.jax_constants import (
    MAX_ICEBERGS,
    MAX_SLICES_PER_REGEN,
    SENTINEL_ORDER_ID,
)
from omiecid_simulator.omiecid.omie_config import (
    OMIEMarketConfig,
    OMIEProductConfig,
)
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    make_order,
)


# ---------------------------------------------------------------------------
# Fixtures — narrow, synthetic; budget < 1s per test where possible.
# ---------------------------------------------------------------------------


@pytest.fixture
def product_config() -> OMIEProductConfig:
    return OMIEProductConfig()


@pytest.fixture
def market_config() -> OMIEMarketConfig:
    return OMIEMarketConfig()


@pytest.fixture(scope="module")
def open_contract() -> Contract:
    """OPEN contract 37 (09:15-09:30) — same as conftest.py default."""
    return Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


@pytest.fixture
def book(open_contract, product_config, market_config) -> ContractBook:
    return ContractBook(open_contract, product_config, market_config)


@pytest.fixture
def empty_state(book) -> OMIEBookState:
    return book.init()


@pytest.fixture
def iceberg_mgr(book) -> IcebergManager:
    """The IcebergManager owned by the ContractBook fixture."""
    return getattr(book, "_iceberg_mgr", getattr(book, "iceberg_mgr", None))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_iceberg_buy(
    order_id: int = 1,
    price: float = 50.00,
    quantity: float = 30.0,
    peak_qty: float = 10.0,
    agent_id: int = 0,
    timestamp: int = 100,
    ppd: float = 0.0,
):
    """Standard iceberg BUY order."""
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


def _saturated_meta(start_oid: int = 1000) -> jnp.ndarray:
    """Build a saturated `iceberg_meta` of shape `(MAX_ICEBERGS, 8)`.

    All `MAX_ICEBERGS` slots carry distinct, non-sentinel order_ids in
    column 0; remaining columns carry plausible non-zero values so the
    rows are clearly "active" (not sentinels). The exact non-id values
    are not asserted on — only the order_id column is contractually
    relevant for the no-eviction property.
    """
    meta = jnp.zeros((MAX_ICEBERGS, N_COLS), dtype=jnp.int32)
    oids = jnp.arange(
        start_oid, start_oid + MAX_ICEBERGS, dtype=jnp.int32
    )
    meta = meta.at[:, COL_ORDER_ID].set(oids)
    # Set non-zero peak/remaining so rows look active.  Side alternates
    # so both sides are exercised, but again — the contract is on the
    # order_id column only.
    meta = meta.at[:, COL_PEAK_QTY].set(jnp.int32(100))
    meta = meta.at[:, COL_REMAINING_HIDDEN].set(jnp.int32(200))
    meta = meta.at[:, COL_SIDE].set(
        (jnp.arange(MAX_ICEBERGS, dtype=jnp.int32) % jnp.int32(2))
    )
    return meta


def _non_sentinel_oids(meta: jnp.ndarray) -> list:
    """Return the sorted list of non-sentinel order_ids in `meta[:, 0]`."""
    col = meta[:, COL_ORDER_ID].tolist()
    return sorted(oid for oid in col if oid != SENTINEL_ORDER_ID)


# ===========================================================================
# Test-first contract item (1) — Anti-regression for silent eviction.
# Encoded per option-c binding clarification (sprint-status).
# ===========================================================================


class TestSlotOverflowNoEviction:
    """`register` on a saturated meta returns `(meta, overflow=True)` and
    does NOT evict any pre-existing iceberg.

    Mechanism-locked: `register(meta, order) -> (updated_meta, overflow)`.
    The four binding assertions (option-c, sprint-status) all execute on
    the same `meta_post` returned from the saturated-`+1` call.
    """

    def test_overflow_flag_true_on_saturated_register(self, iceberg_mgr):
        """(1)(a): `overflow` is True when registering into a saturated meta."""
        meta_pre = _saturated_meta(start_oid=1000)
        new_oid = 1000 + MAX_ICEBERGS  # 101st distinct id (post-fix: dropped)
        order = _make_iceberg_buy(order_id=new_oid)

        # P2-PERF-6 canonical signature: register(meta, n_active, order)
        # -> (meta, n_active, overflow). The saturated-meta fixture has
        # all MAX_ICEBERGS slots live, so n_active = MAX_ICEBERGS.
        out = iceberg_mgr.register(meta_pre, jnp.int32(MAX_ICEBERGS), order)

        assert isinstance(out, tuple) and len(out) == 3, (
            "register must return (updated_meta, n_active, overflow_bool) "
            "post-(2a)."
        )
        meta_post, _n_active_post, overflow = out

        assert bool(overflow) is True, (
            "overflow flag must be True when register is called on a "
            "saturated meta (101st registration into a 100-slot array)."
        )

    def test_cardinality_preserved_on_saturated_register(self, iceberg_mgr):
        """(1)(b): non-sentinel cardinality of `meta_post` == MAX_ICEBERGS.

        No slot was freed and no slot was double-allocated by the
        overflowing register call.
        """
        meta_pre = _saturated_meta(start_oid=1000)
        new_oid = 1000 + MAX_ICEBERGS
        order = _make_iceberg_buy(order_id=new_oid)

        meta_post, _n_active_post, _overflow = iceberg_mgr.register(
            meta_pre, jnp.int32(MAX_ICEBERGS), order,
        )

        non_sentinel_count = int(
            jnp.sum(meta_post[:, COL_ORDER_ID] != jnp.int32(SENTINEL_ORDER_ID))
        )
        assert non_sentinel_count == MAX_ICEBERGS, (
            f"saturated register must preserve cardinality; got "
            f"{non_sentinel_count} non-sentinel slots, expected {MAX_ICEBERGS}."
        )

    def test_no_eviction_equality_against_pre_call_ids(self, iceberg_mgr):
        """(1)(c): sorted surviving ids equal pre-call ids (NOT superset).

        After excluding the new 101st `order_id` from `meta_post`'s
        order_ids exactly once, the multiset must equal the pre-call
        multiset of order_ids. This is the tightened option-c
        assertion: equality, not superset. Equality forbids any
        hypothetical implementation that evicts a pre-existing id and
        writes the new one in its place.
        """
        meta_pre = _saturated_meta(start_oid=1000)
        new_oid = 1000 + MAX_ICEBERGS
        order = _make_iceberg_buy(order_id=new_oid)

        meta_post, _n_active_post, _overflow = iceberg_mgr.register(
            meta_pre, jnp.int32(MAX_ICEBERGS), order,
        )

        pre_ids = _non_sentinel_oids(meta_pre)
        post_ids = _non_sentinel_oids(meta_post)
        # Exclude the new 101st order_id from post_ids exactly once if
        # present (post-fix it must NOT be present, but exclusion-once
        # is the option-c construction; assertion (1)(d) tests absence
        # explicitly).
        post_ids_filtered = list(post_ids)
        if new_oid in post_ids_filtered:
            post_ids_filtered.remove(new_oid)

        assert (
            collections.Counter(post_ids_filtered)
            == collections.Counter(pre_ids)
        ), (
            "no-eviction equality violated — the sorted/sentinel-filtered "
            "surviving order_ids in meta_post (after excluding the new "
            "101st id once) must equal the pre-call multiset. Any "
            "inequality means a pre-existing iceberg was overwritten."
        )

    def test_new_oid_absent_from_post_meta(self, iceberg_mgr):
        """(1)(d): the new 101st `order_id` is NOT present in `meta_post`.

        The behavioral inverse of the silent-eviction bug: when
        `overflow` is True, the 101st registration is dropped (not
        silently written into slot 0 or any other slot).
        """
        meta_pre = _saturated_meta(start_oid=1000)
        new_oid = 1000 + MAX_ICEBERGS
        order = _make_iceberg_buy(order_id=new_oid)

        meta_post, _n_active_post, _overflow = iceberg_mgr.register(
            meta_pre, jnp.int32(MAX_ICEBERGS), order,
        )

        post_oids = meta_post[:, COL_ORDER_ID].tolist()
        assert new_oid not in post_oids, (
            f"overflow register must drop the 101st iceberg; found "
            f"new_oid={new_oid} written into meta_post (silent eviction "
            f"behavior would do this)."
        )


# ===========================================================================
# Test-first contract item (3) — book_depth <= MAX_ICEBERGS __init__ assert.
# ===========================================================================


class TestBookDepthAssertion:
    """`ContractBook.__init__` enforces `book_depth <= MAX_ICEBERGS`."""

    def test_init_raises_when_book_depth_exceeds_max_icebergs(
        self, open_contract, product_config
    ):
        """Constructing `ContractBook` with `book_depth = MAX_ICEBERGS + 1`
        must raise (Python-side, trace-time).

        Implementation freedom: the raise can be `AssertionError` (from
        a literal `assert book_depth <= MAX_ICEBERGS, ...`) or any other
        Python `Exception` subclass — what matters is that the
        Python-side construction call fails fast at __init__ time
        rather than silently allowing a runtime overflow at
        slot-allocation time. We accept any non-trivial Exception.
        """
        bad_market_config = dataclasses.replace(
            OMIEMarketConfig(),
            book_depth=MAX_ICEBERGS + 1,
        )

        with pytest.raises(Exception):
            ContractBook(open_contract, product_config, bad_market_config)

    def test_init_does_not_raise_at_book_depth_equal_max_icebergs(
        self, open_contract, product_config
    ):
        """The boundary case `book_depth == MAX_ICEBERGS` must NOT raise.

        Pins that the assertion is `<=`, not `<`. The story-card
        deliverable language is `assert book_depth <= MAX_ICEBERGS`.
        """
        ok_market_config = dataclasses.replace(
            OMIEMarketConfig(),
            book_depth=MAX_ICEBERGS,
        )

        # Should construct without raising.
        ContractBook(open_contract, product_config, ok_market_config)


# ===========================================================================
# Test-first contract item (4) — OMIEBookState.truncated discriminates
# slot-overflow from slice-overflow.
# ===========================================================================


@pytest.mark.jit
class TestTruncatedDiscriminatesOverflowKinds:
    """`OMIEBookState.truncated` becomes True under EITHER overflow path,
    independently.

    Slice-overflow trigger uses the existing P2-H1
    `MAX_SLICES_PER_REGEN` Path B mechanism (an iceberg whose
    `ceil(remaining_hidden / peak_qty) > MAX_SLICES_PER_REGEN`).
    Slot-overflow trigger uses the new `register`-side overflow
    (101st iceberg into a saturated meta), wired through `_build_state`
    onto the sticky `state.truncated` field.

    Each trigger has its OWN test (independent assertions, independent
    fixtures) so a regression in one mechanism does not mask the other.
    """

    def test_slice_overflow_sets_truncated(
        self, book, empty_state, product_config
    ):
        """Slice-overflow trigger (P2-H1 Path B) sets `state.truncated`.

        Submit an iceberg with `total_qty / peak_qty` >
        `MAX_SLICES_PER_REGEN` — the existing `truncation_flag`-based
        mechanism must surface True on the resulting state's
        `truncated` field. This pins that P2-H1's slice-overflow path
        keeps working under the post-fix `register` signature.
        """
        peak_mw = 0.1
        slices = MAX_SLICES_PER_REGEN + 8
        total_qty = peak_mw * slices  # > MAX_SLICES_PER_REGEN
        assert (total_qty / peak_mw) > MAX_SLICES_PER_REGEN

        order = _make_iceberg_buy(
            order_id=1,
            price=50.00,
            quantity=total_qty,
            peak_qty=peak_mw,
            ppd=0.0,
        )

        new_state, _result = book.submit_order(empty_state, order)
        assert bool(new_state.truncated) is True, (
            "slice-overflow trigger (total_qty/peak_qty > "
            "MAX_SLICES_PER_REGEN) must set state.truncated = True via "
            "the P2-H1 Path B carry-flag mechanism."
        )

    def test_slot_overflow_sets_truncated(
        self, book, empty_state, iceberg_mgr
    ):
        """Slot-overflow trigger (new) sets `state.truncated`.

        Construct a state whose `iceberg_meta` is saturated at
        `MAX_ICEBERGS` distinct ids, then submit a 101st iceberg
        through `book.submit_order`. The `register → (meta, overflow)`
        contract must propagate overflow=True through `_build_state`
        onto `state.truncated`.

        This is the NEW behavior introduced by this hotfix; pre-fix
        the silent-eviction bug means truncated stays False even
        though an iceberg was overwritten.
        """
        # Synthesise a saturated state directly so we don't pay 100x
        # submit_order traces (the < 1s budget rules out that
        # construction). The fields outside `iceberg_meta` come from
        # the empty-init state; they are unrelated to the overflow path.
        sat_meta = _saturated_meta(start_oid=1000)
        # P2-PERF-6: synthetic saturated state must carry the matching
        # n_active = MAX_ICEBERGS so the contiguous-prefix invariant
        # holds (every row is live; no sentinel tail).
        sat_state = empty_state._replace(
            iceberg_meta=sat_meta, n_active=jnp.int32(MAX_ICEBERGS),
        )

        new_oid = 1000 + MAX_ICEBERGS
        order = _make_iceberg_buy(order_id=new_oid)

        new_state, _result = book.submit_order(sat_state, order)

        assert bool(new_state.truncated) is True, (
            "slot-overflow trigger (101st register into saturated meta) "
            "must set state.truncated = True. Pre-fix this stays False "
            "because the silent-eviction bug overwrites slot 0 without "
            "raising; post-fix `register → (meta, overflow)` propagates "
            "overflow=True through `_build_state` onto the sticky flag."
        )

    def test_truncated_is_sticky_across_subsequent_clean_ops(
        self, book, empty_state
    ):
        """Once set by either mechanism, `truncated` stays True across
        subsequent state transitions that would not themselves trigger
        it. Pins the "sticky" semantics promised by the hotfix.

        We use the slice-overflow trigger (cheaper than constructing a
        saturated meta + submit) and then issue a second, well-formed
        non-iceberg order; `truncated` must remain True.
        """
        # First: trigger slice-overflow.
        peak_mw = 0.1
        slices = MAX_SLICES_PER_REGEN + 8
        big_iceberg = _make_iceberg_buy(
            order_id=1,
            price=50.00,
            quantity=peak_mw * slices,
            peak_qty=peak_mw,
            ppd=0.0,
        )
        state1, _ = book.submit_order(empty_state, big_iceberg)
        assert bool(state1.truncated) is True, (
            "preconditioning step: slice-overflow must set truncated."
        )

        # Second: submit a benign order that does not itself trigger
        # any overflow.  truncated MUST stay True (sticky).
        benign = make_order(
            contract_id=37,
            side=Side.SELL,
            price=60.00,
            quantity=1.0,
            order_id=2,
            agent_id=1,
            timestamp=200,
            exec_restriction=ExecRestriction.NON,
        )
        state2, _ = book.submit_order(state1, benign)
        assert bool(state2.truncated) is True, (
            "truncated must remain True (sticky) across subsequent "
            "non-overflow state transitions."
        )
