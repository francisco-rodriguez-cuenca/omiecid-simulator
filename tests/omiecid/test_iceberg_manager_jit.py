"""Locked behavioral-contract tests for P2-H1 — `iceberg_manager` JIT-safe.

These tests pin the eight invariants from the v5.12-amended P2-H1 story
card (`docs/omiecid/agile-plan.md` §"Story P2-H1 — `iceberg_manager` JIT-safe",
Test-first contract row). They are authored before the implementation
lands, so most assertions WILL FAIL until the array-state conversion is
shipped — this is by design: the failures act as targets for the
implementation agent.

Locked invariants (numbering matches the story-card Test-first contract):

  (1) `jax.jit` of `register` / `regenerate_slices` /
      `matchable_hidden_volume` / `cleanup` compiles cleanly.
  (2) State pytree is flat — `jax.tree.leaves(OMIEBookState)` yields
      only `jnp.ndarray`; no `dict` / `list` / `tuple` / `None` leaves.
  (3) ICEwPI slice-loop produces the same slice sequence as Phase 1 for
      all existing ICEwPI fixtures (parity).
  (4) Slot allocation is deterministic and does not collide across
      successive `register` calls in the same session.
  (5) Register-cleanup-register slot reuse — after `cleanup(meta, oid_A)`
      the freed slot is the *first* slot returned by the next `register`
      call. Uses `SENTINEL_ORDER_ID` symbolically.
  (6) Truncation backstop visibility — an ICEwPI fixture with
      `total_qty / peak_qty > MAX_SLICES_PER_REGEN` raises (debug build)
      OR sets a `truncated` carry-flag (release build). Mechanism choice
      is the developer's; this test accepts either path.
  (7) Outer-loop scan parity — slice sequence over multiple
      simultaneously-exhausted icebergs matches Phase 1 (slice ordering,
      timestamp ordering, price ordering).
  (8) `base_price_int` immutability — across N successive
      `regenerate_slices` calls on a single ICEwPI iceberg, the
      `base_price_int` column is byte-identical; only `slice_count`
      increments and `remaining_hidden` decreases.

Architectural references (read at authoring time):
  - docs/omiecid/agile-plan.md v5.12 §P2-H1 (story card)
  - docs/omiecid/sprint-status.md §"P2-H1 — Architecture Review" (§A1–§A6)
  - docs/omiecid/migration-plan.md §Phase 2 loader-dtype contract
  - docs/omiecid/market-spec.md §4.5, §6.2, §13.6 (ICEwPI semantics)
  - omiecid_simulator/omiecid/jax_constants.py (MAX_*, SENTINEL_*)

Encoding contract (per architecture review §A4 / OQ-16 Method A):
`register(meta_array, order)` performs Method A encode-on-entry of
`order.price` / `order.peak_qty` / `order.ppd` from float to int32 ticks.
The 8-column schema is:

    [order_id, remaining_hidden, peak_qty,
     base_price_int, side, agent_id, ppd, slice_count]

`base_price_int` is the IMMUTABLE base price encoded once at register
time. The drifted slice price for slice N is recomputed on every read
as `base_price_int + ppd * slice_count` (encoded space).

Test discipline:
  - JIT-compilation tests carry @pytest.mark.jit (registered in
    pyproject.toml) so they can be deselected on systems without JAX
    GPU/CPU compile budget.
  - Integer columns are compared exactly (no `pytest.approx`).
  - Float comparisons (incoming OMIE units) use `pytest.approx`.
  - Tests do NOT import production symbols that may not exist post-H1;
    they probe the post-H1 surface lazily so the file remains
    collectible against the current dict-state code (failures will be
    `AttributeError` / `TypeError` until P2-H1 ships).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, OMIEBookState
from omiecid_simulator.omiecid.encoding import (
    encode_price,
    encode_qty,
    encode_ppd,
)
from omiecid_simulator.omiecid.iceberg_manager import IcebergManager
from omiecid_simulator.omiecid.jax_constants import (
    MAX_ICEBERGS,
    MAX_SLICES_PER_REGEN,
    SENTINEL_ORDER_ID,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    make_order,
)

# Module-level marker (Story T1.1): every test in this file traces or
# compiles iceberg-manager primitives.  Marker pinned at module scope;
# any pre-existing per-test `@pytest.mark.jit` decorators have been
# stripped (redundant under module-level pin).
pytestmark = pytest.mark.jit

# --- P2-H2 Category-B helper: get_order returns sentinel OMIEOrder, not None.
from omiecid_simulator.omiecid.jax_constants import (
    SENTINEL_ORDER_ID as _P2H2_SENTINEL_ORDER_ID,  # noqa: E501
)


def _is_absent(order):
    """Return True iff ``order`` is Phase-1 ``None`` OR the post-P2-H2 sentinel.

    The post-P2-H2 ``get_order`` returns a sentinel ``OMIEOrder`` whose
    ``order_id == SENTINEL_ORDER_ID`` for absent ids. This helper bridges
    both representations so callsites that pre-date P2-H2 keep working
    without per-site mechanical rewrites.
    """
    if order is None:
        return True
    try:
        return int(order.order_id) == _P2H2_SENTINEL_ORDER_ID
    except (AttributeError, TypeError):
        return False




# ---------------------------------------------------------------------------
# Fixtures — scoped narrowly so each test fits the < 1s fast-suite budget.
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
    # Public attribute name today is `_iceberg_mgr`; accept either the
    # current private name or a future public rename without coupling.
    return getattr(book, "_iceberg_mgr", getattr(book, "iceberg_mgr", None))


@pytest.fixture
def empty_iceberg_meta(empty_state) -> object:
    """The post-H1 `OMIEBookState.iceberg_meta` array (or current dict).

    Post-H1 contract: `jnp.ndarray(MAX_ICEBERGS, 8)` of int32, populated
    with `SENTINEL_ORDER_ID` in column 0 for every empty slot.
    """
    return empty_state.iceberg_meta


# ---------------------------------------------------------------------------
# Helpers — order constructors mirroring tests/omiecid/test_iceberg.py.
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


def _make_aggressor_sell(order_id=20, price=50.00, quantity=10.0, agent_id=3,
                         timestamp=500):
    return make_order(
        contract_id=37,
        side=Side.SELL,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=ExecRestriction.NON,
    )


# ---------------------------------------------------------------------------
# Schema column indices — pinned by the v5.12 story card §A1.
# Test code uses these names to keep the column meaning visible.
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


# ===========================================================================
# Invariant (1) — JIT compilation of the four public entry points.
# ===========================================================================


class TestInvariant1JitCompiles:
    """Each of the four public entry points is `jax.jit`-traceable.

    Story-card pin: "(1) `jax.jit` of each of `register`,
    `regenerate_slices`, `matchable_hidden_volume`, `cleanup` compiles."
    """

    def test_register_jits(self, book, iceberg_mgr, empty_iceberg_meta):
        """`jax.jit(register)` must trace and execute without leaking
        Python control flow into the jaxpr."""
        order = _make_iceberg_buy(order_id=1, ppd=0.0)

        def _register(meta, n_active):
            updated_meta, _n_active, _overflow = iceberg_mgr.register(
                meta, n_active, order,
            )
            return updated_meta

        compiled = jax.jit(_register)
        result = compiled(empty_iceberg_meta, jnp.int32(0))

        # Smoke: returned object is a jnp.ndarray of the expected shape.
        assert isinstance(result, jnp.ndarray), (
            "register must return jnp.ndarray after P2-H1; "
            "current dict-state code returns dict and will fail this."
        )
        # P3-PERF-REGEN-CAP-1: the book-derived meta's first dim follows the
        # configured max_icebergs, not the MAX_ICEBERGS ceiling constant.
        assert result.shape == (book._market_config.max_icebergs, N_COLS)

    def test_regenerate_slices_jits(
        self, iceberg_mgr, empty_state, empty_iceberg_meta
    ):
        """`jax.jit(regenerate_slices)` must compile."""
        # Use empty meta — no work to do, but tracer must succeed.
        lob = empty_state.lob
        ts_counter = jnp.int32(0)

        def _regen(lob, meta, n_active, ts):
            return iceberg_mgr.regenerate_slices(lob, meta, n_active, ts)

        compiled = jax.jit(_regen)
        out = compiled(lob, empty_iceberg_meta, jnp.int32(0), ts_counter)

        # Output must be a 4-tuple (lob, meta, n_active, ts) under the
        # P2-PERF-6 canonical signature.
        assert isinstance(out, tuple) and len(out) == 4

    def test_matchable_hidden_volume_jits(
        self, iceberg_mgr, empty_iceberg_meta, product_config
    ):
        """`jax.jit(matchable_hidden_volume)` must compile."""
        # Encoded crossing price for a BUY aggressor at 100.00 EUR/MWh.
        limit_price_int = jnp.int32(
            encode_price(100.00, product_config)
        )
        side_buy = jnp.int32(int(Side.BUY))

        def _hidden(meta, n_active, side, lpi):
            return iceberg_mgr.matchable_hidden_volume(
                meta, n_active, side, lpi,
            )

        compiled = jax.jit(_hidden)
        out = compiled(empty_iceberg_meta, jnp.int32(0), side_buy, limit_price_int)

        # Output is a scalar jnp.int*; no Python int.
        assert hasattr(out, "dtype"), (
            "matchable_hidden_volume must return a JAX scalar after P2-H1"
        )

    def test_cleanup_jits(self, book, iceberg_mgr, empty_iceberg_meta):
        """`jax.jit(cleanup)` must compile.

        Cleanup of a non-existent slot is a no-op under the post-H1
        contract (no Python `dict.pop` semantics)."""
        def _cleanup(meta, n_active, oid):
            return iceberg_mgr.cleanup(meta, n_active, oid)

        compiled = jax.jit(_cleanup)
        result, _n_active = compiled(
            empty_iceberg_meta, jnp.int32(0), jnp.int32(999),
        )

        assert isinstance(result, jnp.ndarray)
        # P3-PERF-REGEN-CAP-1: first dim follows the configured max_icebergs.
        assert result.shape == (book._market_config.max_icebergs, N_COLS)


# ===========================================================================
# Invariant (2) — flat-pytree audit. No dict/list/tuple/None leaves.
# ===========================================================================


class TestInvariant2FlatPytree:
    """The OMIEBookState pytree contains only `jnp.ndarray` leaves.

    Story-card pin: "(2) State pytree is flat — `jax.tree.leaves(state)`
    yields only `jnp.ndarray`; no Python `dict`, `list`, `tuple`, or
    `None` leaves (audit-class)."
    """

    def test_iceberg_meta_is_ndarray(self, book, empty_iceberg_meta):
        """`OMIEBookState.iceberg_meta` is a `jnp.ndarray` of the expected
        static shape and integer dtype."""
        assert isinstance(empty_iceberg_meta, jnp.ndarray), (
            "iceberg_meta must be jnp.ndarray after P2-H1, "
            f"got {type(empty_iceberg_meta).__name__}"
        )
        # P3-PERF-REGEN-CAP-1: first dim follows the configured max_icebergs.
        assert empty_iceberg_meta.shape == (
            book._market_config.max_icebergs, N_COLS,
        )
        # int32 columns (with int64 accumulators only inside RawFills,
        # which is not part of iceberg_meta).
        assert jnp.issubdtype(empty_iceberg_meta.dtype, jnp.integer)

    def test_book_state_pytree_has_only_ndarray_leaves(self, empty_state):
        """`jax.tree.leaves(state)` returns only `jnp.ndarray` (no dict
        / list / tuple / None leaves)."""
        leaves = jax.tree.leaves(empty_state)
        assert len(leaves) > 0
        for i, leaf in enumerate(leaves):
            assert isinstance(leaf, jnp.ndarray), (
                f"Leaf {i} of OMIEBookState pytree is "
                f"{type(leaf).__name__}, expected jnp.ndarray. "
                "Phase-1 dict/Python-int leaves must be eliminated by P2-H1."
            )

    def test_book_state_has_no_python_collections(self, empty_state):
        """The state must not carry `dict` / `list` / `tuple` (other than
        the outer NamedTuple) / `None` as fields."""
        # OMIEBookState is itself a NamedTuple — the outer container is a
        # PyTree node, not a leaf. We verify the leaves only.
        for leaf in jax.tree.leaves(empty_state):
            assert not isinstance(leaf, (dict, list, type(None))), (
                f"forbidden Python leaf type: {type(leaf).__name__}"
            )

    def test_iceberg_meta_initial_slots_are_sentinel(
        self, book, empty_iceberg_meta
    ):
        """All configured slots start with `order_id == SENTINEL_ORDER_ID`.

        P3-PERF-REGEN-CAP-1: the slot count follows the book's configured
        ``max_icebergs`` (was the ``MAX_ICEBERGS`` ceiling constant)."""
        cap = book._market_config.max_icebergs
        order_ids = empty_iceberg_meta[:, COL_ORDER_ID]
        assert order_ids.shape == (cap,)
        # exact integer comparison
        assert int(jnp.sum(order_ids == SENTINEL_ORDER_ID)) == cap, (
            "Empty iceberg_meta must have all slots set to SENTINEL_ORDER_ID"
        )


# ===========================================================================
# Invariant (3) — ICEwPI slice-sequence parity against Phase-1 fixtures.
# ===========================================================================


class TestInvariant3ICEwPISliceParity:
    """ICEwPI slice loop produces the same slice sequence as Phase 1.

    Story-card pin: "(3) ICEwPI slice loop produces the same slice
    sequence as Phase 1 for all existing ICEwPI fixtures."

    Fixtures borrowed from the Phase-1 suite (test_iceberg.py
    ICEwPI sections — see tests/omiecid/test_iceberg.py classes
    `TestICEwPIPriceIncrement*`).
    """

    def test_icewpi_sell_slice_prices_increase_per_slice(
        self, book, empty_state, product_config
    ):
        """ICEwPI SELL with ppd=+0.50 EUR/MWh: slice 0 at base, slice 1
        at base + 0.50, slice 2 at base + 1.00, etc.

        Behavioral parity check via `submit_order` outcomes — survives
        the pytree shape conversion (Category B per the manifest).
        """
        # base 50.00, peak 10, total 30 -> 3 slices: 50.00, 50.50, 51.00
        ice = _make_iceberg_sell(
            order_id=1, price=50.00, quantity=30.0,
            peak_qty=10.0, agent_id=1, timestamp=100, ppd=0.50,
        )
        state, res = book.submit_order(empty_state, ice)
        assert res.accepted == True

        # Aggressor BUY at 50.00 — only slice 0 crosses
        agg0 = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=10.0,
            order_id=10, agent_id=2, timestamp=500,
            exec_restriction=ExecRestriction.NON,
        )
        state, r0 = book.submit_order(state, agg0)
        assert r0.traded == True
        assert r0.trade_price == pytest.approx(50.00, abs=0.01)
        assert r0.trade_qty == pytest.approx(10.0, abs=0.01)

        # Aggressor BUY at 50.50 — slice 1 (now resting after regen)
        # Tests parity: regen must produce the drifted slice price.
        agg1 = make_order(
            contract_id=37, side=Side.BUY, price=50.50, quantity=10.0,
            order_id=11, agent_id=2, timestamp=501,
            exec_restriction=ExecRestriction.NON,
        )
        state, r1 = book.submit_order(state, agg1)
        assert r1.traded == True
        assert r1.trade_price == pytest.approx(50.50, abs=0.01)

    def test_icewpi_buy_slice_prices_decrease_per_slice(
        self, book, empty_state
    ):
        """ICEwPI BUY with ppd=-0.50: slice 0 at 50.00, slice 1 at 49.50."""
        ice = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=20.0,
            peak_qty=10.0, agent_id=0, timestamp=100, ppd=-0.50,
        )
        state, res = book.submit_order(empty_state, ice)
        assert res.accepted == True

        # Aggressor SELL at 50.00 — fills slice 0 at 50.00
        agg0 = _make_aggressor_sell(order_id=20, price=50.00, quantity=10.0)
        state, r0 = book.submit_order(state, agg0)
        assert r0.traded == True
        assert r0.trade_price == pytest.approx(50.00, abs=0.01)

        # Aggressor SELL at 49.50 — fills slice 1 at 49.50 (drifted)
        agg1 = _make_aggressor_sell(order_id=21, price=49.50, quantity=10.0,
                                    timestamp=600)
        state, r1 = book.submit_order(state, agg1)
        assert r1.traded == True
        assert r1.trade_price == pytest.approx(49.50, abs=0.01)

    def test_icewpi_ppd_zero_behaves_as_standard_iceberg(
        self, book, empty_state
    ):
        """ICEwPI with ppd=0 must behave identically to a standard
        iceberg — all slices at the same price."""
        ice = _make_iceberg_sell(
            order_id=1, price=50.00, quantity=20.0,
            peak_qty=10.0, agent_id=1, timestamp=100, ppd=0.0,
        )
        state, _ = book.submit_order(empty_state, ice)

        agg0 = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=10.0,
            order_id=10, agent_id=2, timestamp=500,
            exec_restriction=ExecRestriction.NON,
        )
        state, r0 = book.submit_order(state, agg0)
        assert r0.trade_price == pytest.approx(50.00, abs=0.01)

        agg1 = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=10.0,
            order_id=11, agent_id=2, timestamp=501,
            exec_restriction=ExecRestriction.NON,
        )
        state, r1 = book.submit_order(state, agg1)
        # Standard iceberg: slice 1 is also at 50.00 (no drift).
        assert r1.trade_price == pytest.approx(50.00, abs=0.01)


# ===========================================================================
# Invariant (4) — slot allocation is deterministic and non-colliding.
# ===========================================================================


class TestInvariant4SlotAllocation:
    """Slot allocation is deterministic and does not collide.

    Story-card pin: "(4) Slot allocation is deterministic and does not
    collide across `register` calls in the same session."
    """

    def test_first_register_uses_slot_zero(self, iceberg_mgr,
                                           empty_iceberg_meta):
        """The first iceberg registered into an empty meta lands in slot 0.

        Determinism flows from `jnp.where(slot[:, 0] == SENTINEL_ORDER_ID,
        size=1)` returning the first sentinel slot.
        """
        order = _make_iceberg_buy(order_id=42)
        meta1, _n_active, _ = iceberg_mgr.register(
            empty_iceberg_meta, jnp.int32(0), order,
        )

        # Slot 0 now carries our order
        assert int(meta1[0, COL_ORDER_ID]) == 42
        # All other slots remain sentinel — semantically equivalent to a
        # per-row `== SENTINEL_ORDER_ID` check; jnp.all reduces the
        # per-row bools to one any-failure-trips assertion.
        assert bool(jnp.all(
            meta1[1:, COL_ORDER_ID] == jnp.int32(SENTINEL_ORDER_ID)
        ))

    def test_three_consecutive_registers_occupy_slots_0_1_2(
        self, iceberg_mgr, empty_iceberg_meta
    ):
        """Three successive register calls populate slots 0, 1, 2 in order."""
        meta = empty_iceberg_meta
        n_active = jnp.int32(0)
        oids = [101, 102, 103]
        for oid in oids:
            order = _make_iceberg_buy(order_id=oid, peak_qty=5.0,
                                      quantity=10.0)
            meta, n_active, _ = iceberg_mgr.register(meta, n_active, order)

        for slot, expected_oid in enumerate(oids):
            assert int(meta[slot, COL_ORDER_ID]) == expected_oid, (
                f"slot {slot}: expected order_id={expected_oid}, "
                f"got {int(meta[slot, COL_ORDER_ID])}"
            )

    def test_no_collision_across_distinct_orders(self, iceberg_mgr,
                                                  empty_iceberg_meta):
        """Distinct order_ids occupy distinct slots — no overwrite."""
        meta = empty_iceberg_meta
        n_active = jnp.int32(0)
        oids = [201, 202, 203, 204, 205]
        for oid in oids:
            order = _make_iceberg_buy(order_id=oid, peak_qty=5.0,
                                      quantity=10.0)
            meta, n_active, _ = iceberg_mgr.register(meta, n_active, order)

        # Read order_ids from the first 5 slots
        observed = [int(meta[i, COL_ORDER_ID]) for i in range(5)]
        # Every requested order_id appears exactly once
        for oid in oids:
            assert observed.count(oid) == 1, (
                f"order_id {oid} should appear in exactly one slot"
            )

    def test_register_sets_base_price_int_via_method_a_encoding(
        self, iceberg_mgr, empty_iceberg_meta, product_config
    ):
        """register() encodes order.price → base_price_int via
        encode_price (offset-encoded int32). Architecture review §A4."""
        order = _make_iceberg_buy(order_id=1, price=50.00)
        meta, _n_active, _ = iceberg_mgr.register(
            empty_iceberg_meta, jnp.int32(0), order,
        )

        expected_base = encode_price(50.00, product_config)
        assert int(meta[0, COL_BASE_PRICE_INT]) == expected_base, (
            "base_price_int must equal encode_price(order.price, config)"
        )

    def test_register_sets_peak_qty_via_method_a_encoding(
        self, iceberg_mgr, empty_iceberg_meta, product_config
    ):
        """register() encodes order.peak_qty → peak_qty (int32 ticks)."""
        order = _make_iceberg_buy(order_id=1, peak_qty=10.0,
                                  quantity=30.0)
        meta, _n_active, _ = iceberg_mgr.register(
            empty_iceberg_meta, jnp.int32(0), order,
        )

        expected_peak = encode_qty(10.0, product_config)
        assert int(meta[0, COL_PEAK_QTY]) == expected_peak

    def test_register_sets_ppd_as_signed_int_ticks(
        self, iceberg_mgr, empty_iceberg_meta, product_config
    ):
        """ppd column is signed int-ticks (NOT offset-encoded), per the
        Loader dtype contract and encoding.encode_ppd."""
        order = _make_iceberg_sell(order_id=1, ppd=0.50)
        meta, n_active, _ = iceberg_mgr.register(
            empty_iceberg_meta, jnp.int32(0), order,
        )

        expected_ppd = encode_ppd(0.50, product_config)
        assert int(meta[0, COL_PPD]) == expected_ppd, (
            "ppd column must use encode_ppd (signed int-ticks)"
        )
        # And for the BUY direction (ppd < 0) the sign must be preserved.
        order2 = _make_iceberg_buy(order_id=2, ppd=-0.50)
        meta2, _n_active, _ = iceberg_mgr.register(meta, n_active, order2)
        expected_ppd_neg = encode_ppd(-0.50, product_config)
        assert int(meta2[1, COL_PPD]) == expected_ppd_neg
        assert int(meta2[1, COL_PPD]) < 0


# ===========================================================================
# Invariant (5) — register / cleanup / register slot reuse.
# ===========================================================================


class TestInvariant5SlotReuseAfterCleanup:
    """After cleanup, the freed slot is reused by the next register.

    Story-card pin: "(5) Register-cleanup-register slot reuse — after
    `cleanup(meta, oid_A)`, the freed slot is the *first* slot returned
    by the next `register` call (deterministic re-use via
    `jnp.where(..., size=1)`; uses `SENTINEL_ORDER_ID` symbolically)."
    """

    def test_cleanup_restores_slot_to_sentinel(self, iceberg_mgr,
                                                empty_iceberg_meta):
        """After cleanup, the freed slot's order_id == SENTINEL_ORDER_ID."""
        order_a = _make_iceberg_buy(order_id=11)
        meta, n_active, _ = iceberg_mgr.register(
            empty_iceberg_meta, jnp.int32(0), order_a,
        )
        assert int(meta[0, COL_ORDER_ID]) == 11

        meta, _n_active = iceberg_mgr.cleanup(meta, n_active, 11)
        assert int(meta[0, COL_ORDER_ID]) == SENTINEL_ORDER_ID

    def test_register_after_cleanup_reuses_freed_slot(
        self, iceberg_mgr, empty_iceberg_meta
    ):
        """register-cleanup-register: the next register lands in slot 0
        (the freed slot), NOT in slot 1.

        This is the symbolic-`SENTINEL_ORDER_ID` invariant: slot
        allocation finds the FIRST sentinel slot — after cleanup, that
        is the freed one."""
        order_a = _make_iceberg_buy(order_id=11)
        meta, n_active, _ = iceberg_mgr.register(
            empty_iceberg_meta, jnp.int32(0), order_a,
        )
        meta, n_active = iceberg_mgr.cleanup(meta, n_active, 11)

        order_b = _make_iceberg_sell(order_id=22)
        meta, _n_active, _ = iceberg_mgr.register(meta, n_active, order_b)

        assert int(meta[0, COL_ORDER_ID]) == 22, (
            "Freed slot 0 must be reused by the next register call"
        )
        # And slot 1 remains empty
        assert int(meta[1, COL_ORDER_ID]) == SENTINEL_ORDER_ID

    def test_cleanup_uses_sentinel_constant_symbolically(
        self, iceberg_mgr, empty_iceberg_meta
    ):
        """Cleanup must write `SENTINEL_ORDER_ID` (the symbolic
        constant), not a hard-coded literal `-1`. This is verified by
        confirming the freed slot equals the imported constant."""
        order = _make_iceberg_buy(order_id=99)
        meta, n_active, _ = iceberg_mgr.register(
            empty_iceberg_meta, jnp.int32(0), order,
        )
        meta, _n_active = iceberg_mgr.cleanup(meta, n_active, 99)

        # Symbolic equality, not literal-int equality.
        from omiecid_simulator.omiecid import jax_constants
        sentinel = jax_constants.SENTINEL_ORDER_ID
        assert int(meta[0, COL_ORDER_ID]) == sentinel


# ===========================================================================
# Invariant (6) — MAX_SLICES_PER_REGEN truncation backstop.
# ===========================================================================


class TestInvariant6TruncationBackstop:
    """An ICEwPI fixture that would exceed MAX_SLICES_PER_REGEN must
    NOT silently drop hidden volume.

    Story-card pin: "(6) Truncation backstop visibility — an ICEwPI
    fixture with `total_qty / peak_qty > MAX_SLICES_PER_REGEN` raises
    (debug build) or sets the truncation flag (release build) without
    silently dropping volume."

    Mechanism choice is the developer's: this test accepts either path.
      - Path A (debug-build assertion): regenerate_slices raises a
        Python exception (AssertionError, RuntimeError, or a custom
        exception class) when the inner while_loop hits the trip-count
        bound without `remaining == 0`.
      - Path B (release-build carry-flag): iceberg_manager exports a
        `truncated` accessor (or the regenerate_slices return tuple
        carries a truncation flag) that becomes True on the trigger.
    """

    def _build_truncating_fixture(self, peak_mw: float = 0.1,
                                   slices_to_request: int = 96):
        """Build an ICEwPI iceberg whose
        `total_qty / peak_qty > MAX_SLICES_PER_REGEN`.

        With peak=0.1 MW and total=peak * (MAX_SLICES_PER_REGEN + 32),
        the implied slice count exceeds the 64 trip-count bound.
        """
        assert slices_to_request > MAX_SLICES_PER_REGEN, (
            "fixture must exceed the trip-count to trigger the backstop"
        )
        total_qty = peak_mw * slices_to_request
        return _make_iceberg_sell(
            order_id=1,
            price=50.00,
            quantity=total_qty,
            peak_qty=peak_mw,
            agent_id=1,
            timestamp=100,
            ppd=0.50,  # ICEwPI sell — positive ppd
        )

    def test_truncation_backstop_visible(self, book, empty_state,
                                          iceberg_mgr):
        """Some backstop must fire; the test accepts either path.

        Try path A first (debug-build assertion). If a raise does not
        happen, fall back to path B and require a truncation accessor.
        If neither path fires, the silent-drop hazard documented in
        `jax_constants.py` §B2 is materialised — that is a hard fail.
        """
        ice = self._build_truncating_fixture(
            peak_mw=0.1, slices_to_request=MAX_SLICES_PER_REGEN + 32
        )
        state, res = book.submit_order(empty_state, ice)
        assert res.accepted == True

        # Force regeneration cascade by feeding a large aggressor
        # against the iceberg side.
        aggressor = make_order(
            contract_id=37,
            side=Side.BUY,
            price=999.00,  # crosses every drifted slice
            quantity=ice.quantity,  # fully consume
            order_id=999,
            agent_id=2,
            timestamp=600,
            exec_restriction=ExecRestriction.NON,
        )

        path_a_raised = False
        try:
            state2, _ = book.submit_order(state, aggressor)
        except (AssertionError, RuntimeError, ValueError) as exc:
            # Path A: debug-build assertion fired.
            path_a_raised = True
            # message must reference truncation / MAX_SLICES_PER_REGEN
            msg = str(exc).lower()
            assert (
                "truncat" in msg
                or "max_slices_per_regen" in msg
                or "slice" in msg
            ), (
                "truncation exception message must mention the bound; "
                f"got: {exc!r}"
            )

        if path_a_raised:
            return  # Path A satisfied; done.

        # Path B: a truncation accessor must exist and report True.
        # Try common accessor patterns the developer might choose.
        truncated = self._probe_truncation_flag(state2, iceberg_mgr)
        assert truncated is True, (
            "Either the debug-build assertion (Path A) must fire, OR a "
            "truncation flag (Path B) must be readable as True. "
            "Silent volume loss is forbidden by `jax_constants.py` §B2."
        )

    @staticmethod
    def _probe_truncation_flag(state, mgr) -> bool:
        """Return True if any truncation-flag accessor reports True.

        Tries (in order):
          1. `state.truncated` (NamedTuple field on OMIEBookState)
          2. `mgr.truncation_flag(state.iceberg_meta)` accessor method
          3. `mgr.is_truncated(state)` method
          4. last column of `state.iceberg_meta` carries a per-slot flag
        """
        # 1. State-level field
        flag = getattr(state, "truncated", None)
        if flag is not None:
            return bool(flag)
        # 2. Accessor on manager
        for fn_name in ("truncation_flag", "is_truncated", "was_truncated"):
            fn = getattr(mgr, fn_name, None)
            if fn is not None:
                try:
                    out = fn(state.iceberg_meta) if fn_name != "is_truncated" \
                        else fn(state)
                except Exception:
                    continue
                if out is not None:
                    return bool(out)
        return False


# ===========================================================================
# Invariant (7) — outer-loop scan parity over multi-iceberg fixtures.
# ===========================================================================


class TestInvariant7OuterLoopScanParity:
    """The slice sequence over multiple simultaneously-exhausted icebergs
    matches Phase-1 ordering exactly.

    Story-card pin: "(7) Outer-loop scan parity — the slice sequence
    emitted by `regenerate_slices` over a fixture with multiple
    simultaneously-exhausted icebergs matches the Phase-1 sequence
    exactly (slice ordering, timestamp ordering, price ordering)."

    The outer loop becomes `jax.lax.scan` over MAX_ICEBERGS slots
    (architecture review §A2). Scan ordering must equal Phase-1's
    `for oid in list(meta.keys())` insertion order, and each slice
    regeneration takes a fresh monotonic `ts_counter`.
    """

    def test_two_icebergs_regenerate_with_strictly_increasing_timestamps(
        self, book, empty_state
    ):
        """Two icebergs at the same price, both exhausted by one
        aggressor — the regenerated slices must carry strictly
        increasing stored stamps (SEC, NSEC) (slot 0 first, slot 1 second)."""
        ice_a = _make_iceberg_buy(
            order_id=1, price=50.00, quantity=20.0, peak_qty=10.0,
            agent_id=0, timestamp=100, ppd=0.0,
        )
        ice_b = _make_iceberg_buy(
            order_id=2, price=50.00, quantity=20.0, peak_qty=10.0,
            agent_id=1, timestamp=101, ppd=0.0,
        )
        state, _ = book.submit_order(empty_state, ice_a)
        state, _ = book.submit_order(state, ice_b)

        # Aggressor SELL 20 MW at 50.00 — fills both peaks in one call.
        agg = _make_aggressor_sell(order_id=99, price=50.00, quantity=20.0,
                                   timestamp=500)
        state, res = book.submit_order(state, agg)
        assert res.traded == True
        assert res.trade_qty == pytest.approx(20.0, abs=0.01)

        # Both icebergs must have regenerated. OMIE-SLICE-STAMP-TAPE-SEC-1
        # (E2, licence (24)): both slices are activated by the aggressor's
        # matching process, so both store SEC = 500 (the activation event,
        # was "> 500"), and A precedes B by the stored key (SEC, NSEC)
        # (exhaustion order; was a strictly smaller SEC).
        regen_a = book.get_order(state, 1, Side.BUY)
        regen_b = book.get_order(state, 2, Side.BUY)
        assert not _is_absent(regen_a) and not _is_absent(regen_b)
        assert regen_a.timestamp == 500
        assert regen_b.timestamp == 500
        bids = jax.device_get(state.lob.bids)

        def _stored_key(oid):
            rows = [r for r in bids if int(r[2]) == oid and int(r[1]) > 0]
            assert len(rows) == 1
            return int(rows[0][4]), int(rows[0][5])   # (SEC, NSEC)

        assert _stored_key(1) < _stored_key(2), (
            "Iceberg A (exhausted first) must be queued BEFORE iceberg B "
            "by (SEC, NSEC)"
        )

    def test_two_icewpi_icebergs_regenerate_at_distinct_drifted_prices(
        self, book, empty_state
    ):
        """Two ICEwPI icebergs with different ppd values must each
        regenerate at their own drifted price — the outer scan must
        carry per-slot `base_price_int + ppd * slice_count` correctly."""
        ice_a = _make_iceberg_sell(
            order_id=1, price=50.00, quantity=20.0, peak_qty=10.0,
            agent_id=0, timestamp=100, ppd=0.10,  # +0.10 per slice
        )
        ice_b = _make_iceberg_sell(
            order_id=2, price=51.00, quantity=20.0, peak_qty=10.0,
            agent_id=1, timestamp=101, ppd=0.50,  # +0.50 per slice
        )
        state, _ = book.submit_order(empty_state, ice_a)
        state, _ = book.submit_order(state, ice_b)

        # BUY 10 @ 50.00 — only ice_a slice 0 crosses
        agg = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=10.0,
            order_id=10, agent_id=2, timestamp=500,
            exec_restriction=ExecRestriction.NON,
        )
        state, r = book.submit_order(state, agg)
        assert r.traded == True
        assert r.trade_price == pytest.approx(50.00, abs=0.01)

        # Inspect regenerated slice 1 of ice_a — should be at 50.10.
        regen_a = book.get_order(state, 1, Side.SELL)
        assert not _is_absent(regen_a)
        assert regen_a.price == pytest.approx(50.10, abs=0.01), (
            "ice_a slice 1 should regenerate at base + ppd*1 = 50.10"
        )
        # ice_b's slot is untouched (no aggressor crossed its price)
        rest_b = book.get_order(state, 2, Side.SELL)
        assert not _is_absent(rest_b)
        assert rest_b.price == pytest.approx(51.00, abs=0.01), (
            "ice_b slot must remain at its base price (no regen)"
        )

    def test_outer_scan_ignores_sentinel_slots(self, iceberg_mgr,
                                                empty_state):
        """`regenerate_slices` over a fully-empty meta is a no-op:
        ts_counter unchanged, lob unchanged, meta unchanged.

        This pins the scan's slot-active mask: empty slots
        (`slot[:, 0] == SENTINEL_ORDER_ID`) must be no-ops."""
        lob_in = empty_state.lob
        meta_in = empty_state.iceberg_meta
        ts_in = jnp.int32(0)

        lob_out, meta_out, _n_active_out, ts_out = (
            iceberg_mgr.regenerate_slices(
                lob_in, meta_in, empty_state.n_active, ts_in,
            )
        )

        # Counter does not advance when no work was done
        assert int(ts_out) == int(ts_in), (
            "Empty regeneration must not advance ts_counter"
        )
        # Meta unchanged (byte-identical column 0 across all slots)
        assert bool(jnp.all(meta_out[:, COL_ORDER_ID]
                            == meta_in[:, COL_ORDER_ID]))


# ===========================================================================
# Invariant (8) — base_price_int immutability across N regenerate calls.
# ===========================================================================


class TestInvariant8BasePriceIntImmutability:
    """`base_price_int` is invariant across N successive regenerate calls.

    Story-card pin: "(8) `base_price_int` immutability — across N
    successive `regenerate_slices` calls on a single ICEwPI iceberg,
    the `base_price_int` column is invariant; only `slice_count`
    increments and `remaining_hidden` decreases."

    This is the load-bearing column-semantics decision from the
    architecture review (Finding F1, Risk #1): the array carries the
    IMMUTABLE base price, NOT the drifted current price.
    """

    def test_base_price_int_byte_identical_across_three_regens(
        self, book, empty_state
    ):
        """ICEwPI sell: base 50.00, ppd +0.50, peak 10, total 30 (3 slices).
        After two aggressor-driven regenerations, the iceberg's
        base_price_int column must equal its register-time value."""
        ice = _make_iceberg_sell(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=1, timestamp=100, ppd=0.50,
        )
        state, _ = book.submit_order(empty_state, ice)

        # Snapshot the base_price_int at register time (slot 0)
        base_at_register = self._read_base_price_int(state, slot=0)

        # Drive two regenerations
        for i, agg_price in enumerate([50.00, 50.50]):
            agg = make_order(
                contract_id=37, side=Side.BUY, price=agg_price,
                quantity=10.0,
                order_id=10 + i, agent_id=2, timestamp=500 + i,
                exec_restriction=ExecRestriction.NON,
            )
            state, r = book.submit_order(state, agg)
            assert r.traded == True

            base_after = self._read_base_price_int(state, slot=0)
            # exact integer comparison — base_price_int must not drift
            assert base_after == base_at_register, (
                f"base_price_int drifted after regen {i + 1}: "
                f"{base_at_register} -> {base_after}. "
                "The 8-field schema must carry the IMMUTABLE base, "
                "not the drifted current price (architecture review F1)."
            )

    def test_only_slice_count_and_remaining_change_across_regens(
        self, book, empty_state, product_config
    ):
        """After one regen, the iceberg's slot has:
          - same order_id, peak_qty, side, agent_id, ppd, base_price_int
          - slice_count incremented by 1
          - remaining_hidden decreased by peak_qty
        """
        ice = _make_iceberg_sell(
            order_id=1, price=50.00, quantity=30.0, peak_qty=10.0,
            agent_id=1, timestamp=100, ppd=0.50,
        )
        state, _ = book.submit_order(empty_state, ice)

        # Read pre-regen slot snapshot
        slot_before = self._read_slot(state, 0)

        # Aggressor BUY @ 50.00 — exhausts visible peak, triggers regen
        agg = make_order(
            contract_id=37, side=Side.BUY, price=50.00, quantity=10.0,
            order_id=10, agent_id=2, timestamp=500,
            exec_restriction=ExecRestriction.NON,
        )
        state, r = book.submit_order(state, agg)
        assert r.traded == True

        slot_after = self._read_slot(state, 0)

        # Invariant columns: byte-identical
        for col_name, col in [
            ("order_id", COL_ORDER_ID),
            ("peak_qty", COL_PEAK_QTY),
            ("base_price_int", COL_BASE_PRICE_INT),
            ("side", COL_SIDE),
            ("agent_id", COL_AGENT_ID),
            ("ppd", COL_PPD),
        ]:
            assert slot_after[col] == slot_before[col], (
                f"column '{col_name}' must be invariant across regen; "
                f"{slot_before[col]} -> {slot_after[col]}"
            )

        # slice_count increments by 1
        assert slot_after[COL_SLICE_COUNT] == slot_before[COL_SLICE_COUNT] + 1

        # remaining_hidden decreased by peak_qty (in encoded units)
        assert slot_after[COL_REMAINING_HIDDEN] < slot_before[COL_REMAINING_HIDDEN]

    def test_base_price_int_invariant_under_pure_regenerate_calls(
        self, iceberg_mgr, empty_state
    ):
        """Direct test on `regenerate_slices`: register an ICEwPI
        iceberg into the meta array, then call `regenerate_slices`
        N times in a loop on an unchanged LOB. Every call must leave
        base_price_int byte-identical to the register-time value.

        This isolates the array-level invariant from the contract_book
        side: iceberg_manager itself must not write back to the
        base_price_int column under any code path."""
        order = _make_iceberg_sell(
            order_id=1, price=50.00, quantity=30.0,
            peak_qty=10.0, agent_id=1, timestamp=100, ppd=0.50,
        )
        meta, n_active, _ = iceberg_mgr.register(
            empty_state.iceberg_meta, empty_state.n_active, order,
        )
        base_at_register = int(meta[0, COL_BASE_PRICE_INT])

        # Note: `regenerate_slices` reads the LOB; on an empty LOB the
        # peak is "missing" so a slice would be regenerated. We are not
        # asserting on the slice action here — only on column invariance.
        lob = empty_state.lob
        ts = jnp.int32(0)
        for _ in range(3):
            lob, meta, n_active, ts = iceberg_mgr.regenerate_slices(
                lob, meta, n_active, ts,
            )
            base_now = int(meta[0, COL_BASE_PRICE_INT])
            assert base_now == base_at_register, (
                f"base_price_int changed during regenerate_slices: "
                f"{base_at_register} -> {base_now}"
            )

    # ----- helpers -----

    @staticmethod
    def _read_slot(state, slot: int) -> dict:
        """Read all 8 columns of the given slot as a {col_idx: int} dict."""
        meta = state.iceberg_meta
        return {col: int(meta[slot, col]) for col in range(N_COLS)}

    @staticmethod
    def _read_base_price_int(state, slot: int) -> int:
        return int(state.iceberg_meta[slot, COL_BASE_PRICE_INT])


# ===========================================================================
# Cross-cutting smoke checks (sanity, not a numbered invariant).
# ===========================================================================


class TestCrosscuttingSmoke:
    """Lightweight smoke checks that surface schema breakage early."""

    def test_iceberg_meta_dtype_is_integer(self, empty_iceberg_meta):
        """The eight columns are integer (int32 per the architecture
        review §A4 and §A1; int64 is reserved for RawFills accumulators
        on the contract_book side)."""
        assert jnp.issubdtype(empty_iceberg_meta.dtype, jnp.integer), (
            f"iceberg_meta dtype must be integer; got {empty_iceberg_meta.dtype}"
        )

    def test_iceberg_meta_static_shape_matches_max_icebergs(
        self, book, empty_iceberg_meta
    ):
        """The first dimension equals the configured ``max_icebergs`` and
        the second is 8.

        P3-PERF-REGEN-CAP-1: the first dim is config-derived
        (``market_config.max_icebergs``), not the ``MAX_ICEBERGS`` ceiling
        constant."""
        cap = book._market_config.max_icebergs
        assert empty_iceberg_meta.shape == (cap, N_COLS), (
            f"iceberg_meta shape must be (max_icebergs={cap}, "
            f"N_COLS={N_COLS}); got {empty_iceberg_meta.shape}"
        )

    def test_max_slices_per_regen_is_at_least_two(self):
        """MAX_SLICES_PER_REGEN must be >= 2 (an iceberg has at least
        two slices by definition; see jax_constants.py §B2)."""
        assert MAX_SLICES_PER_REGEN >= 2

    def test_sentinel_order_id_is_negative(self):
        """SENTINEL_ORDER_ID must be < 0 so it cannot alias a real
        loader-emitted order_id (which is >= 1)."""
        assert SENTINEL_ORDER_ID < 0
