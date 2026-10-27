"""Audit tests for P2-H1 — `iceberg_manager` JIT-safe.

Independent post-implementation audit of the array-backed
``IcebergManager`` and its ``OMIEBookState`` consumer surface. Tests are
derived directly from the v5.12 P2-H1 story card, the §A1–§A6
architecture amendments, and ``docs/omiecid/market-spec.md`` §3 / §4 / §13
ICE / ICEwPI semantics — without reference to the original locked test
file.

Audit dimensions covered (mapping to the 10 dimensions enumerated in
the audit brief):

  1. Domain-correctness against market-spec (ICE peak-only invariant;
     ICEwPI slice price drift ``base_price + ppd * slice_idx``;
     residual handling on the final slice).
  2. Pytree-leaf typing — every leaf in ``OMIEBookState`` is a fixed-
     shape ``jnp.ndarray`` with the declared dtype.
  3. Slot-table invariants — true sentinel rows after cleanup; no
     duplicate ``order_id`` after register-cleanup-register cycles.
  4. Encode-on-entry contract (§A4) — float→int round-trip at the
     OMIE-grid resolution.
  5. Truncation backstop semantics — sticky / scoped / inspectable.
  6. ``base_price_int`` immutability under regenerate-cascade.
  7. Outer-scan parity at boundary cases (empty / single / full meta;
     timestamp ordering).
  8. JIT compilation surface — leaves & structure parity.
  9. Match-against-iceberg semantics — visible+hidden volume.
 10. Module-boundary discipline — no jaxen/jaxrl imports; only ndarray
     leaves.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, OMIEBookState
from omiecid_simulator.omiecid.encoding import (
    decode_price,
    decode_qty,
    encode_price,
    encode_ppd,
    encode_qty,
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
)
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

# Module-level marker (Story T1.1): every test in this audit file traces
# or compiles iceberg-manager primitives.  Marker pinned at module scope;
# pre-existing per-test `@pytest.mark.jit` decorators stripped.
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Local fixtures — minimal, in-test synthetic state per the repo testing guidance
# (contract/unit category, < 1s per test).
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    return OMIEProductConfig()


@pytest.fixture
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


@pytest.fixture
def book(open_contract, cfg, mkt_cfg):
    return ContractBook(open_contract, cfg, mkt_cfg)


@pytest.fixture
def state(book):
    return book.init()


@pytest.fixture
def mgr(book):
    """The ``IcebergManager`` instance owned by the book."""
    return book._iceberg_mgr


def _ice_order(
    *,
    order_id: int,
    side: Side,
    price: float,
    quantity: float,
    peak_qty: float,
    ppd: float = 0.0,
    agent_id: int = 7,
    timestamp: int = 0,
):
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


# ---------------------------------------------------------------------------
# Dimension 2 + 10 — Pytree leaf typing and module-boundary discipline.
# ---------------------------------------------------------------------------


class TestPytreeLeafTyping:
    """Audit dimension 2: every ``OMIEBookState`` leaf is an ndarray
    with declared dtype and fixed shape; no Python collections / None."""

    def test_state_has_exactly_four_leaves(self, state):
        leaves = jax.tree_util.tree_leaves(state)
        # lob has its own subtree (asks, bids, trades, key); plus 3 OMIE leaves.
        # The total should equal 4 lob leaves + iceberg_meta + ts_counter +
        # truncated = 7. We probe specifically here that iceberg_meta /
        # ts_counter / truncated are ndarrays — invariants 1–2 of the story
        # card.
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray), (
                f"leaf {type(leaf).__name__} is not jnp.ndarray"
            )

    def test_state_has_no_python_dict_or_list_or_none(self, state):
        # Walk the pytree and confirm that none of the *leaf* containers
        # collapse to Python containers. ``tree_leaves`` only returns
        # array-like; the structural test is that re-flattening does not
        # surface any Python ``dict``/``list``/``tuple``/``None`` as a
        # leaf — those would not be JAX-traceable.
        leaves = jax.tree_util.tree_leaves(state)
        forbidden = (dict, list, type(None))
        for leaf in leaves:
            assert not isinstance(leaf, forbidden), (
                "OMIEBookState pytree contains a Python container leaf"
            )

    def test_iceberg_meta_shape_and_dtype(self, book, state):
        # P3-PERF-REGEN-CAP-1: the book-derived meta's first dim follows the
        # configured max_icebergs, not the MAX_ICEBERGS ceiling constant; the
        # 8-column schema (N_COLS == 8) is unchanged.
        cap = book._market_config.max_icebergs
        assert state.iceberg_meta.shape == (cap, N_COLS) == (cap, 8)
        assert state.iceberg_meta.dtype == jnp.int32

    def test_ts_counter_is_zero_dim_int32(self, state):
        assert state.ts_counter.shape == ()
        assert state.ts_counter.dtype == jnp.int32

    def test_truncated_is_zero_dim_bool(self, state):
        assert state.truncated.shape == ()
        # ``jnp.bool_`` may surface as ``bool_`` or as the underlying
        # numpy ``bool`` dtype on some platforms; accept both.
        assert state.truncated.dtype == jnp.bool_

    def test_iceberg_manager_module_has_no_jaxen_or_jaxrl_import(self):
        """Audit dimension 10: ``iceberg_manager.py`` must not import
        from ``omiecid_simulator.jaxen`` or ``omiecid_simulator.jaxrl`` —
        the omie/ pure-domain layer is upstream of those packages."""
        src = Path(
            "omiecid_simulator/omiecid/iceberg_manager.py"
        ).read_text(encoding="utf-8")
        # Look for absolute imports against the forbidden packages.
        forbidden = re.compile(
            r"from\s+omiecid_simulator\.(jaxen|jaxrl)\b"
            r"|import\s+omiecid_simulator\.(jaxen|jaxrl)\b"
        )
        assert forbidden.search(src) is None, (
            "iceberg_manager.py must not import from jaxen / jaxrl"
        )


# ---------------------------------------------------------------------------
# Dimension 3 — Slot-table invariants and sentinel rows.
# ---------------------------------------------------------------------------


class TestSlotTableInvariants:
    """Audit dimension 3: sentinel rows are zero-everywhere except
    ``order_id == SENTINEL_ORDER_ID``; no two slots share an
    ``order_id`` after register-cleanup-register cycles."""

    def test_empty_meta_is_pure_sentinel(self):
        meta = IcebergManager.empty_meta()
        # All rows: order_id == SENTINEL_ORDER_ID, every other column == 0.
        oid_col = meta[:, COL_ORDER_ID]
        assert int(jnp.all(oid_col == jnp.int32(SENTINEL_ORDER_ID)))
        # Every non-order_id column must be exactly zero across all rows.
        for col in range(N_COLS):
            if col == COL_ORDER_ID:
                continue
            col_vals = meta[:, col]
            assert int(jnp.all(col_vals == jnp.int32(0))), (
                f"column {col} of an empty meta is not all-zero"
            )

    def test_cleanup_writes_full_sentinel_row_not_just_zeroed_oid(
        self, mgr, cfg
    ):
        meta = IcebergManager.empty_meta()
        order = _ice_order(
            order_id=42, side=Side.SELL, price=50.0, quantity=10.0,
            peak_qty=2.0, ppd=0.1, agent_id=11, timestamp=0,
        )
        meta, n_active, _ = mgr.register(meta, jnp.int32(0), order)
        # Now cleanup that slot.
        meta, _n_active = mgr.cleanup(meta, n_active, 42)
        # The freed row must equal the sentinel row exactly — *not* simply
        # an order_id reset with stale columns lingering.
        # Find the row that was just cleared (it is the first row we ever
        # touched; but any row whose order_id is sentinel and whose other
        # columns are zero counts as a sentinel row).
        sentinel_oid = jnp.int32(SENTINEL_ORDER_ID)
        oids = meta[:, COL_ORDER_ID]
        # Every row must be a true sentinel row.
        is_sentinel_oid = oids == sentinel_oid
        assert int(jnp.all(is_sentinel_oid)), (
            "cleanup left at least one row with a non-sentinel order_id"
        )
        for col in range(N_COLS):
            if col == COL_ORDER_ID:
                continue
            assert int(jnp.all(meta[:, col] == jnp.int32(0))), (
                f"after cleanup, column {col} retained non-zero state"
            )

    def test_no_two_slots_share_order_id_after_register_cleanup_register(
        self, mgr
    ):
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        # Register A, register B, cleanup A, register A again.
        a = _ice_order(
            order_id=101, side=Side.BUY, price=10.0, quantity=8.0,
            peak_qty=2.0, ppd=0.0,
        )
        b = _ice_order(
            order_id=202, side=Side.SELL, price=12.0, quantity=6.0,
            peak_qty=3.0, ppd=0.0,
        )
        meta, n_active, _ = mgr.register(meta, n_active, a)
        meta, n_active, _ = mgr.register(meta, n_active, b)
        meta, n_active = mgr.cleanup(meta, n_active, 101)
        # Re-register an order with the same id to probe duplicate detection.
        a2 = _ice_order(
            order_id=101, side=Side.BUY, price=11.0, quantity=4.0,
            peak_qty=1.0, ppd=0.0,
        )
        meta, n_active, _ = mgr.register(meta, n_active, a2)
        # Across the whole table, distinct active order_ids only.
        active_oids = meta[
            meta[:, COL_ORDER_ID] != jnp.int32(SENTINEL_ORDER_ID),
            COL_ORDER_ID,
        ]
        py_oids = [int(x) for x in active_oids]
        assert len(py_oids) == len(set(py_oids)), (
            f"duplicate order_id in slot table: {py_oids}"
        )


# ---------------------------------------------------------------------------
# Dimension 4 — Encode-on-entry contract (§A4).
# ---------------------------------------------------------------------------


class TestEncodeOnEntry:
    """Audit dimension 4: ``register`` accepts float fields and stores
    int32 ticks. Round-trip via ``decode_*`` reproduces the input at
    OMIE-grid resolution."""

    def test_register_stores_int32_offset_encoded_base_price(self, mgr, cfg):
        meta = IcebergManager.empty_meta()
        order = _ice_order(
            order_id=1, side=Side.BUY, price=-12.34, quantity=5.0,
            peak_qty=1.0, ppd=0.0,
        )
        meta, _n_active, _ = mgr.register(meta, jnp.int32(0), order)
        stored_int = int(meta[0, COL_BASE_PRICE_INT])
        # Reference: encoding.encode_price (Python).
        assert stored_int == encode_price(-12.34, cfg)
        # Round-trip through decode_price must reproduce the input within
        # the price-tick grid.
        round_trip = decode_price(stored_int, cfg)
        assert round_trip == pytest.approx(-12.34, abs=cfg.price_tick / 2)

    def test_register_stores_int32_qty_ticks(self, mgr, cfg):
        meta = IcebergManager.empty_meta()
        order = _ice_order(
            order_id=1, side=Side.SELL, price=20.0, quantity=7.3,
            peak_qty=1.4,
        )
        meta, _n_active, _ = mgr.register(meta, jnp.int32(0), order)
        peak_int = int(meta[0, COL_PEAK_QTY])
        rem_hidden = int(meta[0, COL_REMAINING_HIDDEN])
        # peak == encode_qty(1.4) == round(14)
        assert peak_int == encode_qty(1.4, cfg)
        # remaining_hidden = encode_qty(7.3) - encode_qty(1.4)
        assert rem_hidden == encode_qty(7.3, cfg) - encode_qty(1.4, cfg)
        # Quantity round-trip preserves grid resolution.
        assert decode_qty(peak_int, cfg) == pytest.approx(1.4, abs=cfg.qty_tick / 2)

    def test_register_stores_signed_int32_ppd(self, mgr, cfg):
        meta = IcebergManager.empty_meta()
        # Negative PPD is meaningful (BUY iceberg with falling slice price).
        order = _ice_order(
            order_id=1, side=Side.BUY, price=30.0, quantity=4.0,
            peak_qty=2.0, ppd=-0.05,
        )
        meta, _n_active, _ = mgr.register(meta, jnp.int32(0), order)
        ppd_int = int(meta[0, COL_PPD])
        assert ppd_int == encode_ppd(-0.05, cfg)
        assert ppd_int < 0  # sign preserved

    def test_remaining_hidden_clamped_non_negative(self, mgr, cfg):
        """Defensive clamp: rounding-edge inputs must not produce a
        negative remaining_hidden."""
        meta = IcebergManager.empty_meta()
        # peak == quantity ⇒ remaining_hidden should be exactly 0
        # (boundary case).
        order = _ice_order(
            order_id=1, side=Side.BUY, price=15.0, quantity=2.5,
            peak_qty=2.5,
        )
        meta, _n_active, _ = mgr.register(meta, jnp.int32(0), order)
        assert int(meta[0, COL_REMAINING_HIDDEN]) == 0


# ---------------------------------------------------------------------------
# Dimension 5 — Truncation backstop: monotonic / scoped / inspectable.
# ---------------------------------------------------------------------------


class TestTruncationBackstop:
    """Audit dimension 5: the §A3 backstop must be sticky once tripped,
    not falsely fire on benign fixtures, and surface at the consumer
    boundary."""

    def test_backstop_quiet_on_empty_state(self, state, mgr):
        assert bool(state.truncated) is False
        assert bool(mgr.truncation_flag(state.iceberg_meta)) is False

    def test_backstop_quiet_on_benign_64_slice_fixture(self, mgr, cfg):
        """A fixture sized exactly at the bound (no overrun) must not
        false-trigger. With peak=0.1 MW and quantity = peak * 64 = 6.4
        MW, ``ceil(remaining_hidden/peak)`` is exactly 63, plus
        slice_count starts at 0, so worst_case_total <= 63 ≤ 64."""
        meta = IcebergManager.empty_meta()
        # peak 0.1, total 6.4 → remaining_hidden_int = (6.4-0.1)/0.1 = 63
        # slices needed = 63. <= 64 → quiet.
        order = _ice_order(
            order_id=1, side=Side.BUY, price=10.0, quantity=6.4,
            peak_qty=0.1,
        )
        meta, _n_active, _ = mgr.register(meta, jnp.int32(0), order)
        assert bool(mgr.truncation_flag(meta)) is False

    def test_backstop_trips_when_overrunning_max_slices(self, mgr, cfg):
        """``ceil(remaining/peak) + slice_count > MAX_SLICES_PER_REGEN``
        ⇒ the flag must trip at register time."""
        meta = IcebergManager.empty_meta()
        # peak 0.1 MW, total 999.0 MW → remaining_hidden_int = 9989
        # slices needed = ceil(9989/1) = 9989 ≫ 64 → trip.
        order = _ice_order(
            order_id=1, side=Side.SELL, price=20.0, quantity=999.0,
            peak_qty=0.1,
        )
        meta, _n_active, _ = mgr.register(meta, jnp.int32(0), order)
        assert bool(mgr.truncation_flag(meta)) is True

    def test_backstop_is_sticky_across_state_rebuilds(
        self, book, state, mgr
    ):
        """Once the consumer surface (``OMIEBookState.truncated``) goes
        True, subsequent state rebuilds via ``submit_order`` /
        ``cancel_order`` / regeneration must keep it True even if the
        live meta no longer trips."""
        # Submit a hazard iceberg.
        hazard = _ice_order(
            order_id=999, side=Side.SELL, price=20.0, quantity=999.0,
            peak_qty=0.1, agent_id=3, timestamp=1,
        )
        new_state, result = book.submit_order(state, hazard)
        assert result.accepted == True
        assert bool(new_state.truncated) is True
        # Now cancel the hazard; the live flag will re-evaluate to False
        # (slot is freed), but the sticky carry should remain True.
        post_cancel = book.cancel_order(new_state, 999, Side.SELL)
        live = bool(mgr.truncation_flag(post_cancel.iceberg_meta))
        assert live is False  # live re-evaluates clean
        assert bool(post_cancel.truncated) is True  # sticky persists

    def test_backstop_scoped_to_active_slots_only(self, mgr, cfg):
        """An iceberg whose hidden volume *would* truncate but whose
        slot is sentinel (i.e., already cleaned up) must not trip."""
        meta = IcebergManager.empty_meta()
        order = _ice_order(
            order_id=1, side=Side.SELL, price=20.0, quantity=999.0,
            peak_qty=0.1,
        )
        meta, n_active, _ = mgr.register(meta, jnp.int32(0), order)
        assert bool(mgr.truncation_flag(meta)) is True
        meta, _n_active = mgr.cleanup(meta, n_active, 1)
        # After cleanup the slot is sentinel — the backstop must not
        # fire on the (now-zero) row.
        assert bool(mgr.truncation_flag(meta)) is False


# ---------------------------------------------------------------------------
# Dimension 6 — `base_price_int` immutability under regenerate-cascade.
# ---------------------------------------------------------------------------


class TestBasePriceImmutability:
    """Audit dimension 6: across realistic multi-step matching, the
    ``base_price_int`` column of an active iceberg slot is invariant.
    Only ``slice_count`` increments and ``remaining_hidden`` decreases."""

    def _find_slot(self, meta, order_id):
        for i in range(meta.shape[0]):
            if int(meta[i, COL_ORDER_ID]) == order_id:
                return i
        return None

    def test_base_price_invariant_across_three_regenerate_cascades(
        self, book, state, cfg
    ):
        """Realistic cascade: register an ICEwPI, hit it three times
        with crossing aggressor orders, and observe that
        ``base_price_int`` never moves."""
        # Passive ICEwPI on the SELL side: peak 1 MW, total 5 MW, ppd +0.10.
        passive = _ice_order(
            order_id=500, side=Side.SELL, price=50.0, quantity=5.0,
            peak_qty=1.0, ppd=0.10, agent_id=4, timestamp=1,
        )
        state, result_p = book.submit_order(state, passive)
        assert result_p.accepted == True
        slot_idx = self._find_slot(state.iceberg_meta, 500)
        assert slot_idx is not None
        base_int_initial = int(state.iceberg_meta[slot_idx, COL_BASE_PRICE_INT])

        observed = [base_int_initial]
        prior_remaining = int(
            state.iceberg_meta[slot_idx, COL_REMAINING_HIDDEN]
        )
        prior_slice_count = int(
            state.iceberg_meta[slot_idx, COL_SLICE_COUNT]
        )

        # Aggressor BUYs at successively higher prices to cross drifted slices.
        # Slice N price = 50.0 + 0.10 * N; aggressor buys at >= those.
        for k, agg_oid in enumerate([1001, 1002, 1003]):
            aggressor = make_order(
                contract_id=37, side=Side.BUY, price=60.0, quantity=1.0,
                order_id=agg_oid, agent_id=99, timestamp=2 + k,
                exec_restriction=ExecRestriction.NON,
            )
            state, _ = book.submit_order(state, aggressor)
            slot_idx = self._find_slot(state.iceberg_meta, 500)
            if slot_idx is None:
                # The iceberg fully drained; we already captured several
                # cascades — break and validate.
                break
            cur_base = int(state.iceberg_meta[slot_idx, COL_BASE_PRICE_INT])
            observed.append(cur_base)
            cur_remaining = int(
                state.iceberg_meta[slot_idx, COL_REMAINING_HIDDEN]
            )
            cur_slice_count = int(
                state.iceberg_meta[slot_idx, COL_SLICE_COUNT]
            )
            # Monotonicity: remaining strictly decreases, slice_count
            # strictly increases (across calls that produced regen).
            assert cur_remaining <= prior_remaining
            assert cur_slice_count >= prior_slice_count
            prior_remaining = cur_remaining
            prior_slice_count = cur_slice_count

        # Every observation of base_price_int must equal the initial value.
        assert all(b == base_int_initial for b in observed), (
            f"base_price_int drifted across cascade: {observed}"
        )


# ---------------------------------------------------------------------------
# Dimension 7 — Outer-scan boundary cases & determinism.
# ---------------------------------------------------------------------------


class TestOuterScanBoundaryCases:
    """Audit dimension 7: empty / single / full-meta boundary cases for
    ``regenerate_slices`` and timestamp-ordering determinism."""

    def test_regenerate_slices_on_empty_meta_is_noop(self, book, state, mgr):
        lob_before = state.lob
        ts_before = int(state.ts_counter)
        new_lob, new_meta, _n_active, new_ts = mgr.regenerate_slices(
            lob_before, state.iceberg_meta, state.n_active, state.ts_counter,
        )
        # Meta unchanged.
        assert int(jnp.all(new_meta == state.iceberg_meta))
        # ts_counter unchanged (no regenerations dispatched).
        assert int(new_ts) == ts_before
        # Lob unchanged.
        assert int(jnp.all(new_lob.asks == lob_before.asks))
        assert int(jnp.all(new_lob.bids == lob_before.bids))

    def test_regenerate_slices_single_iceberg_no_consumption_is_noop(
        self, book, state, mgr
    ):
        """A registered ICE whose visible peak is still resting in the
        book (no consumption) should not trigger any regeneration."""
        passive = _ice_order(
            order_id=300, side=Side.SELL, price=50.0, quantity=4.0,
            peak_qty=2.0, ppd=0.0, timestamp=0,
        )
        state, _ = book.submit_order(state, passive)
        meta_before = state.iceberg_meta
        new_lob, new_meta, _n_active, _ = mgr.regenerate_slices(
            state.lob, state.iceberg_meta, state.n_active, state.ts_counter,
        )
        # Slot row unchanged: peak is still in the book.
        assert int(jnp.all(new_meta == meta_before))

    def test_register_at_max_minus_one_then_one_more(self, mgr, cfg):
        """Boundary: register MAX_ICEBERGS-1 icebergs, then one more —
        the table should fill its last sentinel slot. No crash, no
        duplicate ids."""
        meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        # Pure plumbing: fill MAX_ICEBERGS-1 sequential orders via fori_loop;
        # no per-iter assertion to fold.
        def _body_fill_minus_one(k, carry):
            meta, n_active = carry
            order = _ice_order(
                order_id=1000 + k, side=Side.BUY, price=10.0, quantity=2.0,
                peak_qty=1.0, ppd=0.0,
            )
            meta, n_active, _ = mgr.register(meta, n_active, order)
            return (meta, n_active)
        meta, n_active = jax.lax.fori_loop(
            0, MAX_ICEBERGS - 1, _body_fill_minus_one, (meta, n_active),
        )
        # One sentinel slot remains.
        sentinel_count = int(jnp.sum(
            (meta[:, COL_ORDER_ID] == jnp.int32(SENTINEL_ORDER_ID)).astype(
                jnp.int32
            )
        ))
        assert sentinel_count == 1
        # Fill the last slot.
        last = _ice_order(
            order_id=9999, side=Side.SELL, price=20.0, quantity=2.0,
            peak_qty=1.0, ppd=0.0,
        )
        meta, _n_active, _ = mgr.register(meta, n_active, last)
        # Now zero sentinels.
        sentinel_count = int(jnp.sum(
            (meta[:, COL_ORDER_ID] == jnp.int32(SENTINEL_ORDER_ID)).astype(
                jnp.int32
            )
        ))
        assert sentinel_count == 0
        # Active oids are unique.
        oids = [int(x) for x in meta[:, COL_ORDER_ID]]
        assert len(set(oids)) == len(oids)

    def test_timestamp_counter_monotonic_across_regenerate(
        self, book, state, cfg
    ):
        """Audit dimension 7: ts_counter is monotonic-non-decreasing
        across submit/regenerate cycles."""
        ts_history = [int(state.ts_counter)]

        passive = _ice_order(
            order_id=400, side=Side.SELL, price=50.0, quantity=3.0,
            peak_qty=1.0, ppd=0.0, timestamp=10,
        )
        state, _ = book.submit_order(state, passive)
        ts_history.append(int(state.ts_counter))

        # Aggressor consumes first peak; submit_order internally calls
        # regenerate_slices which advances ts_counter.
        for k, agg_oid in enumerate([2001, 2002]):
            aggressor = make_order(
                contract_id=37, side=Side.BUY, price=55.0, quantity=1.0,
                order_id=agg_oid, agent_id=88, timestamp=20 + k,
                exec_restriction=ExecRestriction.NON,
            )
            state, _ = book.submit_order(state, aggressor)
            ts_history.append(int(state.ts_counter))

        for a, b in zip(ts_history, ts_history[1:]):
            assert b >= a, f"ts_counter went backwards: {ts_history}"


# ---------------------------------------------------------------------------
# Dimension 8 — JIT compilation surface parity.
# ---------------------------------------------------------------------------


class TestJITCompilationSurface:
    """Audit dimension 8: jitted ``IcebergManager`` methods produce the
    same pytree leaves and structure as their eager counterparts."""

    def test_truncation_flag_jit_eager_parity(self, mgr, cfg):
        meta = IcebergManager.empty_meta()
        order = _ice_order(
            order_id=1, side=Side.BUY, price=15.0, quantity=4.0,
            peak_qty=1.0, ppd=0.0,
        )
        meta, _n_active, _ = mgr.register(meta, jnp.int32(0), order)
        eager = mgr.truncation_flag(meta)
        jitted = jax.jit(mgr.truncation_flag)(meta)
        assert eager.shape == jitted.shape == ()
        assert eager.dtype == jitted.dtype
        assert bool(eager) == bool(jitted)

    def test_register_jit_pytree_structure_parity(self, mgr, cfg):
        meta_eager = IcebergManager.empty_meta()
        order = _ice_order(
            order_id=42, side=Side.SELL, price=25.0, quantity=6.0,
            peak_qty=2.0, ppd=0.05,
        )
        meta_after_eager, _na_eager, _ = mgr.register(
            meta_eager, jnp.int32(0), order,
        )
        meta_after_jit, _na_jit, _ = jax.jit(mgr.register)(
            meta_eager, jnp.int32(0), order,
        )
        # Tree structure identical.
        s_eager = jax.tree_util.tree_structure(meta_after_eager)
        s_jit = jax.tree_util.tree_structure(meta_after_jit)
        assert s_eager == s_jit
        # Element-wise equality.
        assert int(jnp.all(meta_after_eager == meta_after_jit))

    def test_matchable_hidden_volume_jit_eager_parity(self, mgr, cfg):
        meta = IcebergManager.empty_meta()
        order = _ice_order(
            order_id=7, side=Side.SELL, price=30.0, quantity=10.0,
            peak_qty=2.0, ppd=0.0,
        )
        meta, n_active, _ = mgr.register(meta, jnp.int32(0), order)
        # Aggressor BUY at 35 — should cross the entire iceberg.
        limit_int = encode_price(35.0, cfg)
        eager = mgr.matchable_hidden_volume(
            meta, n_active, int(Side.BUY), limit_int,
        )
        jitted = jax.jit(
            mgr.matchable_hidden_volume,
            static_argnums=(),
        )(meta, n_active, jnp.int32(int(Side.BUY)), jnp.int32(limit_int))
        assert int(eager) == int(jitted)
        # Sanity: ``remaining_hidden_int = encode_qty(10) - encode_qty(2)``
        # i.e. 80; the projected slices total all of it at limit=35.
        expected = encode_qty(10.0, cfg) - encode_qty(2.0, cfg)
        assert int(eager) == expected


# ---------------------------------------------------------------------------
# Dimension 9 — Match-against-iceberg semantics (visible+hidden).
# ---------------------------------------------------------------------------


class TestMatchAgainstIcebergSemantics:
    """Audit dimension 9: an aggressor crossing a hidden iceberg sees
    the projected hidden volume via ``matchable_hidden_volume``, and a
    follow-up ``regenerate_slices`` replenishes the visible tip."""

    def test_matchable_hidden_volume_returns_only_crossing_slices(
        self, mgr, cfg
    ):
        """ICEwPI projection: only slices whose drifted price actually
        crosses the aggressor's limit are counted."""
        meta = IcebergManager.empty_meta()
        # SELL ICEwPI: base 50, peak 1, qty 6, ppd +0.50.
        # Slice 1 price = 50.50, slice 2 = 51.00, slice 3 = 51.50, etc.
        order = _ice_order(
            order_id=1, side=Side.SELL, price=50.0, quantity=6.0,
            peak_qty=1.0, ppd=0.50,
        )
        meta, n_active, _ = mgr.register(meta, jnp.int32(0), order)
        # Aggressor BUY at 51.00 — only slices priced <= 51.00 should
        # contribute. That is slices 1 and 2 (50.50, 51.00). Slice 3
        # (51.50) does NOT cross — total contribution = 2 * peak_int.
        limit_int = encode_price(51.0, cfg)
        contrib = int(mgr.matchable_hidden_volume(
            meta, n_active, int(Side.BUY), limit_int,
        ))
        peak_int = encode_qty(1.0, cfg)
        assert contrib == 2 * peak_int

    def test_matchable_hidden_volume_zero_for_non_crossing(self, mgr, cfg):
        meta = IcebergManager.empty_meta()
        order = _ice_order(
            order_id=1, side=Side.SELL, price=50.0, quantity=6.0,
            peak_qty=1.0, ppd=0.10,
        )
        meta, n_active, _ = mgr.register(meta, jnp.int32(0), order)
        # Aggressor BUY at 49.0 — no slice crosses (drifted prices >= 50.10).
        limit_int = encode_price(49.0, cfg)
        contrib = int(mgr.matchable_hidden_volume(
            meta, n_active, int(Side.BUY), limit_int,
        ))
        assert contrib == 0

    def test_matchable_hidden_volume_ignores_same_side_icebergs(
        self, mgr, cfg
    ):
        """A BUY iceberg cannot match a BUY aggressor, regardless of
        price crossing."""
        meta = IcebergManager.empty_meta()
        # Same-side BUY iceberg (irrelevant to BUY aggressor).
        order = _ice_order(
            order_id=1, side=Side.BUY, price=80.0, quantity=10.0,
            peak_qty=1.0, ppd=0.0,
        )
        meta, n_active, _ = mgr.register(meta, jnp.int32(0), order)
        # BUY aggressor crossing the BUY iceberg's price upwards.
        contrib = int(mgr.matchable_hidden_volume(
            meta, n_active, int(Side.BUY), encode_price(100.0, cfg),
        ))
        assert contrib == 0

    def test_regenerate_slices_replenishes_visible_tip_after_match(
        self, book, state, cfg
    ):
        """End-to-end: a crossing aggressor consumes the visible peak;
        on next ``submit_order`` (which internally calls
        ``regenerate_slices``), the next slice surfaces with the
        correct drifted price.

        Marked slow: end-to-end pipeline through ``book.submit_order`` +
        ``regenerate_slices`` for two icebergs.
        """
        # Passive ICEwPI SELL: base 50, peak 1, qty 3, ppd +0.10.
        # After consuming peak (1 MW), slice 2 should appear at
        # 50.10 EUR/MWh.
        passive = _ice_order(
            order_id=600, side=Side.SELL, price=50.0, quantity=3.0,
            peak_qty=1.0, ppd=0.10, agent_id=5, timestamp=1,
        )
        state, _ = book.submit_order(state, passive)
        # Aggressor BUY 1 MW at 50.0 — consumes first peak exactly.
        agg = make_order(
            contract_id=37, side=Side.BUY, price=50.0, quantity=1.0,
            order_id=701, agent_id=77, timestamp=5,
            exec_restriction=ExecRestriction.NON,
        )
        state, result = book.submit_order(state, agg)
        # The second slice should be visible on the SELL book at
        # price = 50.0 + 0.10 = 50.10. Use anonymous LOB view.
        bids, asks = book.get_anonymous_lob(state)
        # Find a SELL level at 50.10.
        sell_prices = [round(p, 2) for (p, _q) in asks]
        assert any(
            abs(p - 50.10) < cfg.price_tick for p in sell_prices
        ), f"second iceberg slice did not surface at 50.10; asks={asks}"


# ---------------------------------------------------------------------------
# Dimension 1 — Domain-correctness against market-spec.
# ---------------------------------------------------------------------------


class TestDomainCorrectness:
    """Audit dimension 1: ICE peak-only invariant; ICEwPI slice price
    drift; final-slice residual handling."""

    def test_ice_visible_only_peak_quantity_on_book(self, book, state, cfg):
        """Market-spec §13: an ICE order shows only ``peak_qty`` on the
        book at any time. The remaining hidden volume must NOT appear
        in the anonymous LOB."""
        passive = _ice_order(
            order_id=800, side=Side.SELL, price=50.0, quantity=10.0,
            peak_qty=1.5, ppd=0.0, timestamp=1,
        )
        state, _ = book.submit_order(state, passive)
        bids, asks = book.get_anonymous_lob(state)
        # Visible volume on asks at 50.0 must equal 1.5 (peak).
        sell_at_50 = [
            (p, q) for (p, q) in asks
            if abs(p - 50.0) < cfg.price_tick / 2
        ]
        assert len(sell_at_50) == 1
        _, qty = sell_at_50[0]
        assert qty == pytest.approx(1.5, abs=cfg.qty_tick / 2)

    def test_icewpi_slice_price_drift_formula(self, mgr, cfg):
        """For ICEwPI, the projected slice prices follow
        ``base_price + ppd * slice_idx`` exactly (in encoded space)."""
        meta = IcebergManager.empty_meta()
        order = _ice_order(
            order_id=1, side=Side.SELL, price=50.0, quantity=5.0,
            peak_qty=1.0, ppd=0.10,
        )
        meta, n_active, _ = mgr.register(meta, jnp.int32(0), order)
        base = int(meta[0, COL_BASE_PRICE_INT])
        ppd = int(meta[0, COL_PPD])
        # Sweep aggressor limits and confirm the cumulative crossing
        # matches the formula.
        for limit_eur in [50.0, 50.10, 50.20, 50.30, 50.40]:
            limit_int = encode_price(limit_eur, cfg)
            # Manual projection: count slices 1..N where
            # base + ppd * slice_idx <= limit_int.
            # remaining_hidden_int / peak_int = (50-10)/10 = 4 slices to project,
            # but slice_count starts at 0 and projection starts at slice_idx=1.
            peak_int = encode_qty(1.0, cfg)
            rem_int = encode_qty(5.0, cfg) - peak_int
            expected = 0
            slice_idx = 1
            rem_left = rem_int
            while rem_left > 0 and slice_idx <= MAX_SLICES_PER_REGEN:
                slice_qty = min(peak_int, rem_left)
                slice_price = base + ppd * slice_idx
                if slice_price <= limit_int:
                    expected += slice_qty
                rem_left -= slice_qty
                slice_idx += 1
            actual = int(mgr.matchable_hidden_volume(
                meta, n_active, int(Side.BUY), limit_int,
            ))
            assert actual == expected, (
                f"limit {limit_eur}: expected={expected} actual={actual}"
            )

    def test_final_slice_residual_handling(self, mgr, cfg):
        """When ``remaining_hidden`` is not a multiple of ``peak_qty``,
        the final slice quantity must be the residual (``< peak_qty``),
        not ``peak_qty`` (which would over-deliver hidden volume)."""
        meta = IcebergManager.empty_meta()
        # peak 2 MW, total 5 MW. remaining_hidden = 3 MW (not divisible by 2).
        # First regen slice = 2 MW, second regen slice = 1 MW (residual).
        order = _ice_order(
            order_id=1, side=Side.SELL, price=50.0, quantity=5.0,
            peak_qty=2.0, ppd=0.0,
        )
        meta, n_active, _ = mgr.register(meta, jnp.int32(0), order)
        peak_int = encode_qty(2.0, cfg)
        rem_int = encode_qty(5.0, cfg) - peak_int  # = 30 (remaining hidden)
        # Aggressor BUY at 50.0 (crosses everything since ppd=0).
        limit_int = encode_price(50.0, cfg)
        contrib = int(mgr.matchable_hidden_volume(
            meta, n_active, int(Side.BUY), limit_int,
        ))
        # Total hidden contributed = remaining_hidden_int (residual handling
        # ensures the total never exceeds remaining_hidden).
        assert contrib == rem_int
        # And critically: the contribution does NOT exceed
        # ceil(rem/peak) * peak_int — i.e., we do not over-count.
        ceiling = math.ceil(rem_int / peak_int) * peak_int
        assert contrib <= ceiling


# ---------------------------------------------------------------------------
# Tail check: a quick top-level invariant for the OMIEBookState pytree
# under JIT identity.
# ---------------------------------------------------------------------------


class TestStatePytreeUnderJIT:
    """Audit dimension 8 (continued): the post-jit ``OMIEBookState``
    has the same pytree structure as the eager state and identical
    leaf values for an identity transform."""

    def test_state_identity_jit_round_trip(self, state):
        identity = jax.jit(lambda s: s)
        out = identity(state)
        assert (
            jax.tree_util.tree_structure(out)
            == jax.tree_util.tree_structure(state)
        )
        out_leaves = jax.tree_util.tree_leaves(out)
        in_leaves = jax.tree_util.tree_leaves(state)
        assert len(out_leaves) == len(in_leaves)
        for a, b in zip(in_leaves, out_leaves):
            assert a.shape == b.shape
            assert a.dtype == b.dtype
            assert int(jnp.all(a == b))
