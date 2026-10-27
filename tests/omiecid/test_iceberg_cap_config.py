"""Locked behavioral contract for P3-PERF-REGEN-CAP-1 — config-driven
iceberg capacity (``OMIEMarketConfig.max_icebergs``, default 1024).

Story: `P3-PERF-REGEN-CAP-1` (agile-plan.md §"Story P3-PERF-REGEN-CAP-1 —
Config-driven iceberg capacity"). This file is the NEW locked contract the
developer implements against; it will FAIL against current `main` (the
``max_icebergs`` field does not exist yet) — that is correct. The file must
be COLLECTIBLE (imports cleanly, no syntax errors) even while red: every
red comes from a runtime call inside a test body, never from import time.

What this file pins (the four locked concerns (a)-(d) + default + one-compile):

  (a) A ``ContractBook`` built with ``max_icebergs=8`` overflows on the 9th
      iceberg registration → ``overflow=True`` from ``register`` → sticky
      ``truncated=True`` on the state (the loud-cap contract at a small
      config value). The overflow surface is ``OMIEBookState.truncated``
      (never a silent eviction) per the P2-HOTFIX-iceberg-cap-overflow
      contract carried forward by this card.
  (b) ``ContractBook.init()`` meta shape follows ``market_config.max_icebergs``
      (build with 1024 and with 256; assert
      ``state.book.iceberg_meta.shape == (that_value, N_COLS)``). This is the
      amended P2-H3 AC-4 shape contract: constant → config-derived, with
      ``MAX_ICEBERGS`` retained as the hard structural ceiling.
  (c) ``ContractBook.__init__`` bounds each capacity axis INDEPENDENTLY by
      the hard ceiling: it rejects ``book_depth > MAX_ICEBERGS`` AND
      ``max_icebergs > MAX_ICEBERGS`` (both raise). ``book_depth`` (resting-
      order LOB capacity, the ``(book_depth, 6)`` LOB arrays) and
      ``max_icebergs`` (iceberg side-channel capacity, the
      ``(max_icebergs, 8)`` ``iceberg_meta`` rows) are ORTHOGONAL axes —
      there is NO ``book_depth <= max_icebergs`` coupling, so a book with
      ``book_depth > max_icebergs`` (both ``<= MAX_ICEBERGS``) is a legal
      configuration and must construct. The __init__ boundary is
      ``book_depth <= MAX_ICEBERGS`` AND ``max_icebergs <= MAX_ICEBERGS``.
      (Corrected acceptance criterion — Step-4 STOP resolution; the card's
      chained ``book_depth <= max_icebergs <= MAX_ICEBERGS`` form was an
      internally-unsatisfiable AC error, contradicted by the locked suite.)
  (d) ``IcebergManager.empty_meta()`` zero-arg default remains
      ``(MAX_ICEBERGS, N_COLS)`` — back-compat pin for the direct callers
      the card lists (``empty_meta(max_icebergs=MAX_ICEBERGS)`` with a
      defaulted param keeps direct-construction callers source-compatible).
  + default: ``OMIEMarketConfig().max_icebergs == 1024``.
  + one-compile: constructing books at two different ``max_icebergs`` values
      does not break the one-compile invariant for a fixed config instance
      (``_submit_order._cache_size()`` stays 1 across a submit sweep on a
      single fixed book), matching the style of the existing one-compile
      tests (``test_full_domain_jit_smoke.py::TestACThreeOneCompileAcrossFiveVariants``).

Domain grounding: market-spec §4.2.1 (iceberg/regen semantics); migration-plan
P2-H3 AC-4 (the ``(MAX_ICEBERGS, 8)`` schema this card amends to
``(market_config.max_icebergs, 8)`` with ``MAX_ICEBERGS`` as the ceiling);
§P3.2 one-compile/vmap invariant.

Device: CPU-hermetic per docs/omiecid/device-policy.md (this story is CPU-bound).
Run with ``CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python3 -m pytest``.
Budget: every test in this file is a small synthetic fixture (contract/unit
category) and targets the < 1s fast-suite budget; none is marked slow.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, OMIEBookState
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    IcebergManager,
    N_COLS,
)
from omiecid_simulator.omiecid.jax_constants import (
    MAX_ICEBERGS,
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
# Fixtures + helpers — narrow, synthetic; < 1s per test.
# ---------------------------------------------------------------------------


@pytest.fixture
def product_config() -> OMIEProductConfig:
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def open_contract() -> Contract:
    """OPEN contract 37 (09:15-09:30) — same as the conftest default."""
    return Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


def _market_config(**overrides) -> OMIEMarketConfig:
    """Build an ``OMIEMarketConfig`` with ``max_icebergs`` (and any other
    field) overridden.

    Against current `main` this raises ``TypeError`` because ``max_icebergs``
    is not a field of the frozen dataclass yet — that is the intended red;
    it fires at call time inside a test body, so this module still collects.
    """
    return OMIEMarketConfig(**overrides)


def _make_book(product_config, open_contract, **market_overrides) -> ContractBook:
    """Construct a ``ContractBook`` whose market config carries the given
    ``max_icebergs`` / ``book_depth`` overrides."""
    return ContractBook(
        open_contract, product_config, _market_config(**market_overrides),
    )


def _ice_buy(
    order_id: int,
    *,
    price: float = 50.00,
    quantity: float = 30.0,
    peak_qty: float = 10.0,
    agent_id: int = 0,
    timestamp: int = 100,
    ppd: float = 0.0,
):
    """A standard ICE iceberg BUY that rests (crosses nothing on an empty
    book) and therefore consumes exactly one iceberg slot on registration."""
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


# ===========================================================================
# Default value — OMIEMarketConfig().max_icebergs == 1024.
# ===========================================================================


class TestDefaultMaxIcebergs:
    """The new frozen field defaults to 1024 (≥2× headroom over the
    observed 257-512 short-RL-tape peak live n_active; the 96-contract
    peak sweep is the empirical backstop that may bump to 2048)."""

    def test_default_max_icebergs_is_1024(self):
        """``OMIEMarketConfig().max_icebergs`` must default to 1024."""
        assert OMIEMarketConfig().max_icebergs == 1024, (
            "P3-PERF-REGEN-CAP-1: the frozen OMIEMarketConfig gains a "
            "`max_icebergs: int = 1024` field. Default 1024 gives ≥2x "
            "headroom over the observed short-RL-tape peak (257-512)."
        )

    def test_max_icebergs_field_is_present_on_dataclass(self):
        """``max_icebergs`` must be a declared field of the frozen config
        (not an ad-hoc attribute), so it is hashable / static-arg safe."""
        field_names = {f.name for f in dataclasses.fields(OMIEMarketConfig)}
        assert "max_icebergs" in field_names, (
            "max_icebergs must be a declared dataclass field of "
            "OMIEMarketConfig (frozen → hashable → static_argnames-safe)."
        )

    def test_config_remains_frozen_and_hashable(self):
        """The config stays a frozen (hashable) dataclass after the field
        add — required for `static_argnames` at every jit lift (one-compile
        preserved)."""
        cfg = _market_config(max_icebergs=1024)
        # Frozen: mutation raises.
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_icebergs = 2048  # type: ignore[misc]
        # Hashable: usable as a static arg / dict key.
        assert hash(cfg) == hash(_market_config(max_icebergs=1024))


# ===========================================================================
# (b) init() meta shape follows market_config.max_icebergs.
# ===========================================================================


class TestInitMetaShapeFollowsConfig:
    """(b): ``ContractBook.init()`` builds ``iceberg_meta`` with first
    dimension ``market_config.max_icebergs`` (amended P2-H3 AC-4 contract:
    ``(MAX_ICEBERGS, 8)`` → ``(market_config.max_icebergs, 8)``)."""

    @pytest.mark.parametrize("cap", [1024, 256])
    def test_init_meta_first_dim_equals_configured_max_icebergs(
        self, product_config, open_contract, cap,
    ):
        """Build a book with ``max_icebergs=cap`` and assert the init
        state's ``iceberg_meta`` shape is ``(cap, N_COLS)``."""
        book = _make_book(product_config, open_contract, max_icebergs=cap)
        state = book.init()
        assert state.iceberg_meta.shape == (cap, N_COLS), (
            f"init() iceberg_meta shape must follow max_icebergs={cap}; "
            f"got {state.iceberg_meta.shape}. P3-PERF-REGEN-CAP-1 amends "
            f"the P2-H3 AC-4 (MAX_ICEBERGS, 8) contract to "
            f"(market_config.max_icebergs, 8)."
        )

    def test_init_meta_dtype_and_columns_unchanged(
        self, product_config, open_contract,
    ):
        """Only the first dim is config-driven: dtype int32 and the
        8-column schema are unchanged from P2-H1/P2-H3."""
        book = _make_book(product_config, open_contract, max_icebergs=256)
        state = book.init()
        assert state.iceberg_meta.dtype == jnp.int32
        assert state.iceberg_meta.shape[1] == N_COLS == 8

    def test_init_meta_all_slots_sentinel_at_configured_cap(
        self, product_config, open_contract,
    ):
        """Every one of the ``cap`` slots starts sentinel (the packed-layout
        empty invariant holds at the configured cap, not the ceiling)."""
        cap = 256
        book = _make_book(product_config, open_contract, max_icebergs=cap)
        state = book.init()
        order_ids = state.iceberg_meta[:, COL_ORDER_ID]
        assert order_ids.shape == (cap,)
        assert int(jnp.sum(order_ids == SENTINEL_ORDER_ID)) == cap

    def test_book_market_config_carries_the_cap(
        self, product_config, open_contract,
    ):
        """The book exposes its configured cap on ``_market_config`` — the
        book-derived value the amended shape assertions across the suite
        assert against."""
        book = _make_book(product_config, open_contract, max_icebergs=512)
        assert book._market_config.max_icebergs == 512


# ===========================================================================
# (a) Loud-cap overflow at a small config value.
# ===========================================================================


class TestSmallCapOverflowIsLoud:
    """(a): a ``ContractBook`` built with ``max_icebergs=8`` overflows on
    the 9th iceberg registration → ``overflow=True`` → sticky
    ``truncated=True`` on the state. Overflow is never a silent eviction
    (P2-HOTFIX-iceberg-cap-overflow contract, carried by this card)."""

    def test_ninth_registration_overflows_at_cap_8(
        self, product_config, open_contract,
    ):
        """Register 8 distinct icebergs through ``book.submit_order`` (all
        fit, ``truncated`` stays False), then the 9th overflows and sets
        the sticky ``truncated`` flag.

        Runtime: cap-8 keeps the meta tiny; 9 submits is well under the 1s
        budget (contract/unit category — synthetic in-test build)."""
        book = _make_book(product_config, open_contract, max_icebergs=8)
        state = book.init()

        # 8 registrations fill the 8-slot meta exactly; each ICE order rests
        # (price 40.00 buy against an empty book — nothing to cross).
        for i in range(8):
            state, _result = book.submit_order(
                state, _ice_buy(order_id=100 + i, price=40.00),
            )
            assert bool(state.truncated) is False, (
                f"registration {i + 1} of 8 must fit within max_icebergs=8 "
                f"without setting truncated; got truncated=True at i={i}."
            )

        # 9th registration overflows the 8-slot cap → sticky truncated.
        state9, _result = book.submit_order(
            state, _ice_buy(order_id=200, price=40.00),
        )
        assert bool(state9.truncated) is True, (
            "the 9th iceberg registration into a max_icebergs=8 book must "
            "overflow (register overflow predicate reads the meta's own "
            "shape[0]=8) and propagate overflow=True through _build_state "
            "onto the sticky OMIEBookState.truncated flag — loud, never a "
            "silent eviction."
        )

    def test_register_overflow_flag_true_on_saturated_small_cap(
        self, product_config, open_contract,
    ):
        """Manager-level probe: ``register`` on a meta saturated at the
        configured ``max_icebergs=8`` returns ``overflow=True`` — the
        overflow predicate is driven by the meta's own first dimension, not
        the ``MAX_ICEBERGS`` ceiling constant."""
        book = _make_book(product_config, open_contract, max_icebergs=8)
        mgr = book._iceberg_mgr

        # Build a saturated 8-row meta directly (all 8 slots non-sentinel).
        cap = 8
        meta = jnp.zeros((cap, N_COLS), dtype=jnp.int32)
        oids = jnp.arange(1000, 1000 + cap, dtype=jnp.int32)
        meta = meta.at[:, COL_ORDER_ID].set(oids)

        order = _ice_buy(order_id=9999, price=40.00)
        out = mgr.register(meta, jnp.int32(cap), order)
        assert isinstance(out, tuple) and len(out) == 3, (
            "register must return (updated_meta, n_active, overflow)."
        )
        _meta_post, _n_active_post, overflow = out
        assert bool(overflow) is True, (
            "register into an 8-slot meta saturated at n_active=8 must set "
            "overflow=True; the predicate reads iceberg_meta.shape[0]=8, "
            "not MAX_ICEBERGS."
        )

    def test_truncated_is_sticky_after_small_cap_overflow(
        self, product_config, open_contract,
    ):
        """Once overflow sets ``truncated`` at the small cap, a subsequent
        benign (non-overflowing) order must leave ``truncated`` True."""
        book = _make_book(product_config, open_contract, max_icebergs=8)
        state = book.init()
        for i in range(8):
            state, _ = book.submit_order(
                state, _ice_buy(order_id=100 + i, price=40.00),
            )
        # Overflow.
        state, _ = book.submit_order(
            state, _ice_buy(order_id=200, price=40.00),
        )
        assert bool(state.truncated) is True

        # A benign non-iceberg sell that rests (does not itself overflow).
        benign = make_order(
            contract_id=37,
            side=Side.SELL,
            price=90.00,
            quantity=1.0,
            order_id=300,
            agent_id=1,
            timestamp=200,
            exec_restriction=ExecRestriction.NON,
        )
        state2, _ = book.submit_order(state, benign)
        assert bool(state2.truncated) is True, (
            "truncated must be sticky once set by the small-cap overflow."
        )


# ===========================================================================
# (c) __init__ rejects book_depth > max_icebergs AND max_icebergs > MAX_ICEBERGS.
# ===========================================================================


class TestInitRejectsInvalidCaps:
    """(c): the ``ContractBook.__init__`` boundary bounds each capacity axis
    INDEPENDENTLY by the hard ceiling — ``book_depth <= MAX_ICEBERGS`` AND
    ``max_icebergs <= MAX_ICEBERGS``. ``book_depth`` (LOB capacity) and
    ``max_icebergs`` (iceberg side-channel capacity) are orthogonal axes;
    there is NO ``book_depth <= max_icebergs`` coupling. A ceiling violation
    on EITHER axis raises; a ``book_depth > max_icebergs`` config with both
    within the ceiling is legal and constructs.

    (Step-4 STOP resolution: the card's chained
    ``book_depth <= max_icebergs <= MAX_ICEBERGS`` AC was an internally-
    unsatisfiable error — contradicted by the must-pass-unmodified guard
    ``test_iceberg_cap_raise_followup.py::test_book_depth_assertion_at_boundary``
    (``book_depth=MAX_ICEBERGS=8192`` against default ``max_icebergs=1024``,
    NO raise) and by the sibling ``TestSmallCapOverflowIsLoud``
    (``book_depth=100, max_icebergs=8``, must construct). The implemented
    ``__init__`` bounds the two axes independently.)"""

    def test_book_depth_may_exceed_max_icebergs_when_both_within_ceiling(
        self, product_config, open_contract,
    ):
        """``book_depth > max_icebergs`` is a LEGAL, independent-axes config
        and must construct WITHOUT raising when both are ``<= MAX_ICEBERGS``.

        ``book_depth`` sizes the resting-order LOB arrays (``(book_depth, 6)``);
        ``max_icebergs`` sizes the iceberg side-channel meta rows
        (``(max_icebergs, 8)``). They are orthogonal — a deeper resting book
        than the iceberg side-channel is perfectly valid (the side-channel
        counts outstanding-with-hidden icebergs, NOT LOB-resident orders).
        There is NO ``book_depth <= max_icebergs`` requirement. Both
        ``book_depth=32`` and ``max_icebergs=16`` are well within the
        ``MAX_ICEBERGS`` ceiling, so construction must succeed."""
        book = _make_book(
            product_config, open_contract,
            max_icebergs=16, book_depth=32,
        )
        assert book._market_config.max_icebergs == 16
        assert book._market_config.book_depth == 32
        # The two orthogonal axes size their own arrays independently: the
        # iceberg side-channel meta follows max_icebergs (16), unaffected by
        # the larger book_depth (32).
        state = book.init()
        assert state.iceberg_meta.shape == (16, N_COLS)

    def test_max_icebergs_greater_than_ceiling_raises(
        self, product_config, open_contract,
    ):
        """``max_icebergs > MAX_ICEBERGS`` must raise — ``MAX_ICEBERGS``
        remains the hard structural ceiling (the runtime cap may not exceed
        the ceiling the packed-layout / regen loops are sized against)."""
        with pytest.raises(AssertionError):
            _make_book(
                product_config, open_contract,
                max_icebergs=MAX_ICEBERGS + 1,
            )

    def test_book_depth_equal_to_max_icebergs_is_accepted(
        self, product_config, open_contract,
    ):
        """The boundary is ``<=`` (not ``<``): ``book_depth == max_icebergs``
        must construct successfully."""
        book = _make_book(
            product_config, open_contract,
            max_icebergs=64, book_depth=64,
        )
        assert book._market_config.max_icebergs == 64

    def test_max_icebergs_equal_to_ceiling_is_accepted(
        self, product_config, open_contract,
    ):
        """``max_icebergs == MAX_ICEBERGS`` is the ceiling boundary and must
        construct (the sanctioned high end; e.g. the golden-parity
        max_icebergs=8192 replay in the Deliverables)."""
        book = _make_book(
            product_config, open_contract,
            max_icebergs=MAX_ICEBERGS,
        )
        assert book._market_config.max_icebergs == MAX_ICEBERGS


# ===========================================================================
# (d) empty_meta() zero-arg default remains (MAX_ICEBERGS, N_COLS).
# ===========================================================================


class TestEmptyMetaZeroArgBackCompat:
    """(d): ``IcebergManager.empty_meta()`` called with no args stays a
    ``(MAX_ICEBERGS, N_COLS)`` array — the back-compat pin for the direct
    callers the card preserves via ``empty_meta(max_icebergs=MAX_ICEBERGS)``
    (defaulted param)."""

    def test_empty_meta_zero_arg_shape_is_ceiling(self):
        """``empty_meta()`` (no args) → ``(MAX_ICEBERGS, N_COLS)``."""
        meta = IcebergManager.empty_meta()
        assert meta.shape == (MAX_ICEBERGS, N_COLS), (
            f"empty_meta() zero-arg default must stay (MAX_ICEBERGS="
            f"{MAX_ICEBERGS}, N_COLS={N_COLS}) for source-compatible direct "
            f"callers; got {meta.shape}."
        )

    def test_empty_meta_zero_arg_all_sentinel(self):
        """``empty_meta()`` initialises every slot to ``SENTINEL_ORDER_ID``
        (the empty-slot marker) — unchanged by the parameterisation."""
        meta = IcebergManager.empty_meta()
        order_ids = meta[:, COL_ORDER_ID]
        assert int(jnp.sum(order_ids == SENTINEL_ORDER_ID)) == MAX_ICEBERGS

    def test_empty_meta_accepts_max_icebergs_kwarg(self):
        """``empty_meta(max_icebergs=cap)`` returns a ``(cap, N_COLS)`` array
        — the new defaulted parameter the card adds (so ``init()`` can size
        the meta from ``market_config.max_icebergs``)."""
        meta = IcebergManager.empty_meta(max_icebergs=256)
        assert meta.shape == (256, N_COLS)
        order_ids = meta[:, COL_ORDER_ID]
        assert int(jnp.sum(order_ids == SENTINEL_ORDER_ID)) == 256


# ===========================================================================
# One-compile invariant — two caps do not break one-compile for a fixed book.
# ===========================================================================


class TestOneCompileAcrossCaps:
    """The config field enters as a static hashable member of the already
    ``static_argnames`` frozen ``OMIEMarketConfig``, so the meta shape is a
    compile-time constant per config instance. A submit sweep on a single
    fixed book must produce exactly ONE ``_submit_order`` cache entry — the
    parameterisation must not add a retrace hazard.

    Style mirrors
    ``test_full_domain_jit_smoke.py::TestACThreeOneCompileAcrossFiveVariants``
    (uses the public ``_submit_order._cache_size()`` probe). Books built at
    two DIFFERENT caps are two different (static) config instances → they may
    legitimately compile separately; the locked invariant is that a SINGLE
    fixed config instance still one-compiles across a submit sweep.
    """

    def test_single_book_submit_sweep_is_one_compile(
        self, product_config, open_contract,
    ):
        """A sweep of submits on ONE book (fixed ``max_icebergs``) yields
        ``_submit_order._cache_size() == 1``."""
        from omiecid_simulator.omiecid.contract_book import _submit_order

        if not hasattr(_submit_order, "_cache_size"):
            pytest.skip(
                "_submit_order does not expose `_cache_size`; trace-count "
                "probe unavailable on this JAX version."
            )

        book = _make_book(product_config, open_contract, max_icebergs=256)
        state = book.init()
        baseline = _submit_order._cache_size()

        # Several submits through the SAME fixed-config book: a resting ICE,
        # a resting NON, and another ICE — all share the jitted artefact.
        state, _ = book.submit_order(state, _ice_buy(order_id=1, price=40.00))
        state, _ = book.submit_order(
            state,
            make_order(
                contract_id=37, side=Side.SELL, price=90.00, quantity=1.0,
                order_id=2, agent_id=1, timestamp=200,
                exec_restriction=ExecRestriction.NON,
            ),
        )
        state, _ = book.submit_order(state, _ice_buy(order_id=3, price=41.00))

        after = _submit_order._cache_size()
        assert after - baseline <= 1, (
            "submitting several orders through a single fixed-config book "
            "must add at most ONE _submit_order cache entry (one-compile). "
            f"Cache grew by {after - baseline} (baseline={baseline}, "
            f"after={after}) — a config-instance-dependent meta shape must "
            "remain a compile-time constant, not a retrace trigger."
        )
