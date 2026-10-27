"""Audit tests for P2-HOTFIX-iceberg-cap-overflow.

Independent additional probes against the finished implementation. Author
did not read the original locked test file; tests are derived directly
from the story contract in `docs/omiecid/agile-plan.md` and the architecture
review in `docs/omiecid/sprint-status.md`.

Audit beats covered:

  1. Slot-overflow `overflow_bool` is correctly derived from the
     "no sentinel slot exists" predicate — boundary cases at the
     last-free-slot and the first-saturated-call.
  2. `OMIEBookState.truncated` stickiness across mixed sequences of
     slot-overflow + slice-overflow + benign ops.
  3. `__init__` assertion message conveys actionable diagnostic info
     (cites the offending values, not bare AssertionError).
  4. §B1 docstring contains corrected steady-state population analysis
     and explicitly references the FOLLOWUP story by id.
  5. Cross-mechanism: slot-overflow does not corrupt previously-valid
     `iceberg_meta` rows (full-array diff, not just sorted-id equality).
  6. When both slice-overflow and slot-overflow trigger in the same step,
     `truncated` is True (OR semantics, not XOR).
  7. Out-of-scope confirmations: `MAX_ICEBERGS == 100`; `OMIEOrder` field
     dtypes unchanged.
  8. Public-API shape pinning: `register` returns 2-tuple of
     (jnp.ndarray, jnp.bool_); `_build_state` `overflow` kwarg defaults
     to False.

All tests target the < 1s fast-suite budget. None invoke jax.jit
directly, so no @pytest.mark.jit marker is required.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_REMAINING_HIDDEN,
    COL_PEAK_QTY,
    COL_SLICE_COUNT,
    IcebergManager,
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
    Status,
    Validity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ice_order(order_id: int, *, peak: float = 1.0, qty: float = 5.0,
               price: float = 50.0, side: int = int(Side.BUY),
               agent_id: int = 1) -> OMIEOrder:
    """Build a minimally-valid iceberg OMIEOrder."""
    return OMIEOrder(
        contract_id=37,
        side=side,
        price=price,
        quantity=qty,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=0,
        exec_restriction=int(ExecRestriction.ICE),
        validity=int(Validity.GFS),
        expiry_time=0,
        status=int(Status.ACTIVE),
        peak_qty=peak,
        ppd=0.0,
    )


@pytest.fixture
def manager(product_config):
    """Bare IcebergManager not bound to a ContractBook (no LOB needed for
    register/cleanup/truncation_flag tests)."""
    from omiecid_simulator.jaxob.jorderbook import OrderBook
    from omiecid_simulator.jaxob.jaxob_config import JAXLOB_Configuration
    cfg = JAXLOB_Configuration(book_depth=10, nOrders=10, nTrades=10)
    ob = OrderBook(cfg=cfg)
    return IcebergManager(ob, product_config)


# ---------------------------------------------------------------------------
# 1. overflow_bool boundary semantics
# ---------------------------------------------------------------------------


class TestOverflowBoolBoundaries:
    """`overflow_bool == ~any(sentinels)` — pin the boundary."""

    def test_one_free_slot_left_no_overflow(self, manager):
        """With exactly one sentinel slot remaining, register must succeed
        (overflow=False) and consume the last slot."""
        meta = manager.empty_meta()
        n_active = jnp.int32(0)
        # Saturate all but one slot.
        # ovf_acc OR-accumulates the per-iter `ovf` flag; post-loop assert
        # catches any-True (bidirectionally equivalent to per-iter
        # `assert bool(ovf) is False`).
        def _body_saturate_minus_one(i, carry):
            meta, n_active, ovf_acc = carry
            meta, n_active, ovf = manager.register(
                meta, n_active, _ice_order(order_id=100 + i)
            )
            return (meta, n_active, ovf_acc | ovf)
        meta, n_active, any_overflow = jax.lax.fori_loop(
            0, MAX_ICEBERGS - 1, _body_saturate_minus_one,
            (meta, n_active, jnp.bool_(False)),
        )
        assert bool(any_overflow) is False
        # Sanity: exactly one sentinel slot remains.
        sentinels = (meta[:, COL_ORDER_ID] == jnp.int32(SENTINEL_ORDER_ID))
        assert int(jnp.sum(sentinels.astype(jnp.int32))) == 1

        meta_final, n_active, ovf = manager.register(
            meta, n_active, _ice_order(order_id=999)
        )
        assert bool(ovf) is False
        # No sentinels left now.
        sentinels_after = (
            meta_final[:, COL_ORDER_ID] == jnp.int32(SENTINEL_ORDER_ID)
        )
        assert int(jnp.sum(sentinels_after.astype(jnp.int32))) == 0

    def test_zero_free_slots_overflow_on_next_register(self, manager):
        """With zero sentinels, the next register reports overflow=True
        and leaves meta untouched."""
        meta = manager.empty_meta()
        n_active = jnp.int32(0)
        # Pure plumbing: fill MAX_ICEBERGS sequential orders; no per-iter
        # assertion (the post-loop sentinel-count check validates the
        # whole sequence).
        def _body_saturate(i, carry):
            meta, n_active = carry
            meta, n_active, _ = manager.register(
                meta, n_active, _ice_order(order_id=100 + i)
            )
            return (meta, n_active)
        meta, n_active = jax.lax.fori_loop(
            0, MAX_ICEBERGS, _body_saturate, (meta, n_active),
        )
        # Saturated: no sentinels.
        sentinels = (meta[:, COL_ORDER_ID] == jnp.int32(SENTINEL_ORDER_ID))
        assert int(jnp.sum(sentinels.astype(jnp.int32))) == 0

        meta_after, _n_active, ovf = manager.register(
            meta, n_active, _ice_order(order_id=42424242)
        )
        assert bool(ovf) is True
        # Meta is byte-identical to the saturated meta — no silent eviction.
        assert jnp.array_equal(meta, meta_after)
        # The new order id is NOT present anywhere.
        assert not bool(jnp.any(meta_after[:, COL_ORDER_ID] == jnp.int32(42424242)))

    def test_overflow_independent_of_meta_content(self, manager):
        """`overflow_bool` is derived from sentinel availability, not from
        any other meta column. A saturated meta whose rows hold weird /
        edge-case values still flips overflow=True."""
        meta = manager.empty_meta()
        n_active = jnp.int32(0)
        # Pure plumbing with traced-from-`i` parametric fields
        # (side / peak / qty); no per-iter assertion.
        buy = jnp.int32(int(Side.BUY))
        sell = jnp.int32(int(Side.SELL))
        def _body_varied_fill(i, carry):
            meta, n_active = carry
            side = jnp.where(i % 2 == 0, buy, sell)
            peak = 0.5 + (i % 5).astype(jnp.float32) * 0.1
            qty = 5.0 + i.astype(jnp.float32) * 0.01
            meta, n_active, _ = manager.register(
                meta, n_active,
                _ice_order(order_id=1000 + i, peak=peak,
                           qty=qty, side=side),
            )
            return (meta, n_active)
        meta, n_active = jax.lax.fori_loop(
            0, MAX_ICEBERGS, _body_varied_fill, (meta, n_active),
        )
        meta_after, _n_active, ovf = manager.register(
            meta, n_active, _ice_order(order_id=7777)
        )
        assert bool(ovf) is True
        assert jnp.array_equal(meta, meta_after)


# ---------------------------------------------------------------------------
# 2. Mixed slot-overflow + slice-overflow + benign-op stickiness
# ---------------------------------------------------------------------------


class TestTruncationStickiness:
    """`OMIEBookState.truncated` is sticky across all flavors of overflow."""

    def test_slot_overflow_then_clean_ops_stay_sticky(
        self, book_and_state, manager
    ):
        """Trigger slot overflow via _build_state, then run many clean
        _build_state calls — `truncated` must remain True."""
        book, state = book_and_state
        # Inject a saturated meta by force.
        saturated_meta = manager.empty_meta()
        sat_n_active = jnp.int32(0)
        # Pure plumbing: fill MAX_ICEBERGS sequential orders; no per-iter
        # assertion.
        def _body_inject_saturate(i, carry):
            meta, n_active = carry
            meta, n_active, _ = manager.register(
                meta, n_active, _ice_order(order_id=200 + i)
            )
            return (meta, n_active)
        saturated_meta, sat_n_active = jax.lax.fori_loop(
            0, MAX_ICEBERGS, _body_inject_saturate,
            (saturated_meta, sat_n_active),
        )

        s1 = book._build_state(
            state.lob, saturated_meta, state.ts_counter,
            state.truncated, sat_n_active, overflow=True,
        )
        assert bool(s1.truncated) is True

        # Now many clean rebuilds with overflow=False — sticky must hold.
        s_curr = s1
        for _ in range(20):
            s_curr = book._build_state(
                s_curr.lob, s_curr.iceberg_meta, s_curr.ts_counter,
                s_curr.truncated, s_curr.n_active, overflow=False,
            )
        assert bool(s_curr.truncated) is True

    def test_slice_overflow_then_clean_ops_stay_sticky(
        self, book_and_state, manager
    ):
        """Trigger slice-overflow path (Path B) by registering an iceberg
        whose hidden fan-out exceeds MAX_SLICES_PER_REGEN. truncated must
        latch True via _build_state's `live` term."""
        book, state = book_and_state
        # peak=0.1, qty=(MAX_SLICES_PER_REGEN+5)*peak -> slices_needed > cap.
        big_qty = float((MAX_SLICES_PER_REGEN + 5) * 0.1 + 0.1)
        meta, n_active, ovf = manager.register(
            state.iceberg_meta, state.n_active,
            _ice_order(order_id=501, peak=0.1, qty=big_qty),
        )
        assert bool(ovf) is False
        # truncation_flag must already trip on this meta.
        assert bool(manager.truncation_flag(meta)) is True

        s1 = book._build_state(
            state.lob, meta, state.ts_counter, state.truncated, n_active,
            overflow=False,
        )
        assert bool(s1.truncated) is True

        # Clean rebuilds with a *cleaned* meta — but the prior_truncated
        # carry should keep it sticky.
        clean_meta = manager.empty_meta()
        s_curr = s1
        for _ in range(5):
            s_curr = book._build_state(
                s_curr.lob, clean_meta, s_curr.ts_counter,
                s_curr.truncated, jnp.int32(0), overflow=False,
            )
        assert bool(s_curr.truncated) is True

    def test_simultaneous_slot_and_slice_overflow_or_semantics(
        self, book_and_state, manager
    ):
        """When both slice-overflow (live truncation_flag True) AND
        slot-overflow (overflow kwarg True) trigger in the same
        _build_state call, truncated is True. OR, not XOR."""
        book, state = book_and_state
        # Build a meta that simultaneously is saturated AND contains a
        # slice-overflowing slot. Saturate first, then surgically replace
        # one row with a slice-overflowing row.
        meta = manager.empty_meta()
        n_active = jnp.int32(0)
        # Pure plumbing: fill MAX_ICEBERGS sequential orders.
        def _body_saturate_600(i, carry):
            meta, n_active = carry
            meta, n_active, _ = manager.register(
                meta, n_active, _ice_order(order_id=600 + i)
            )
            return (meta, n_active)
        meta, n_active = jax.lax.fori_loop(
            0, MAX_ICEBERGS, _body_saturate_600, (meta, n_active),
        )
        # Replace slot 0 with a slice-overflowing iceberg row — set
        # remaining_hidden huge and peak tiny so worst-case > cap.
        meta = (
            meta
            .at[0, COL_REMAINING_HIDDEN].set(jnp.int32(10_000))
            .at[0, COL_PEAK_QTY].set(jnp.int32(1))
            .at[0, COL_SLICE_COUNT].set(jnp.int32(0))
        )
        assert bool(manager.truncation_flag(meta)) is True

        s = book._build_state(
            state.lob, meta, state.ts_counter, state.truncated, n_active,
            overflow=True,
        )
        assert bool(s.truncated) is True


# ---------------------------------------------------------------------------
# 3. __init__ assertion message diagnostics
# ---------------------------------------------------------------------------


class TestInitAssertionMessage:
    def test_assertion_message_cites_offending_values(self, product_config):
        """The book_depth <= MAX_ICEBERGS assertion must include both
        numeric values (actionable diagnostic) — not a bare AssertionError."""
        bad_market = OMIEMarketConfig(book_depth=MAX_ICEBERGS + 1)
        contract = Contract(
            contract_id=1,
            delivery_start_minutes=600,
            delivery_end_minutes=615,
            gct_minutes=540,
            status=ContractStatus.OPEN,
        )
        with pytest.raises(AssertionError) as excinfo:
            ContractBook(contract, product_config, bad_market)
        msg = str(excinfo.value)
        assert str(MAX_ICEBERGS + 1) in msg, (
            f"assertion message missing offending book_depth: {msg!r}"
        )
        assert str(MAX_ICEBERGS) in msg, (
            f"assertion message missing MAX_ICEBERGS bound: {msg!r}"
        )


# ---------------------------------------------------------------------------
# 4. §B1 docstring corrections
# ---------------------------------------------------------------------------


class TestB1DocstringContent:
    """The story card explicitly requires the §B1 docstring to be
    rewritten with the corrected population analysis and to cite the
    FOLLOWUP story id."""

    @pytest.fixture
    def jax_constants_text(self) -> str:
        path = Path(
            "omiecid_simulator/omiecid/jax_constants.py"
        ).resolve()
        return path.read_text()

    def test_followup_story_id_referenced(self, jax_constants_text):
        # The cap-history passage must record the raise from 100 to 8192.
        start = jax_constants_text.find("Cap history")
        assert start != -1, "jax_constants.py must keep its cap-history passage"
        end = jax_constants_text.find("\n\n", start)
        passage = jax_constants_text[start: end if end != -1 else None]
        assert re.search(r"\b100\b", passage), passage
        assert re.search(r"\b8192\b", passage), passage

    def test_population_decomposition_present(self, jax_constants_text):
        """Verify the corrected three-term decomposition is documented:
        population = registers - cleanups - full_drains."""
        # Look for the three component names — exact phrasing locked
        # by the story-card audit row.
        assert "registers_cumulative" in jax_constants_text
        assert "cleanups_cumulative" in jax_constants_text
        assert "full_drains_cumulative" in jax_constants_text

    def test_old_book_depth_derivation_repudiated(self, jax_constants_text):
        """The fixed docstring must call out that the prior
        `book_depth`-tied derivation was incorrect — keyword cues."""
        lower = jax_constants_text.lower()
        # Either "incorrect" or "conflated" must appear in the rewritten §B1.
        assert ("incorrect" in lower) or ("conflated" in lower)


# ---------------------------------------------------------------------------
# 5. Cross-mechanism: no corruption of valid rows on overflow
# ---------------------------------------------------------------------------


class TestNoCorruptionOnOverflow:
    def test_full_meta_unchanged_on_overflow(self, manager):
        """Saturated-meta + overflowing-register call must return the
        SAME meta — element-wise equality across every cell, not just
        a sorted-id check."""
        meta = manager.empty_meta()
        n_active = jnp.int32(0)
        # Pure plumbing with traced-from-`i` parametric fields
        # (peak / qty); no per-iter assertion.
        def _body_saturate_varied(i, carry):
            meta, n_active = carry
            peak = 0.5 + (i % 7).astype(jnp.float32) * 0.1
            qty = 3.0 + i.astype(jnp.float32) * 0.05
            meta, n_active, _ = manager.register(
                meta, n_active,
                _ice_order(order_id=300 + i, peak=peak, qty=qty),
            )
            return (meta, n_active)
        meta, n_active = jax.lax.fori_loop(
            0, MAX_ICEBERGS, _body_saturate_varied, (meta, n_active),
        )
        meta_before = meta  # JAX arrays are immutable — safe alias.
        meta_after, _n_active, ovf = manager.register(
            meta, n_active, _ice_order(order_id=99999)
        )
        assert bool(ovf) is True
        # Full-array element-wise equality.
        assert jnp.array_equal(meta_before, meta_after)
        # And shapes/dtypes preserved.
        assert meta_after.shape == (MAX_ICEBERGS, 8)
        assert meta_after.dtype == jnp.int32


# ---------------------------------------------------------------------------
# 6. Out-of-scope confirmations (negative tests)
# ---------------------------------------------------------------------------


class TestOutOfScopeUnchanged:
    def test_max_icebergs_in_sanctioned_set(self):
        """The cap raise was deferred from P2-HOTFIX to P2-PERF-7 (which
        re-targets the cap at 8192 after P2-PERF-6 made
        ``regenerate_slices`` runtime independent of the cap value).

        Originally this test pinned the literal value ``MAX_ICEBERGS == 100``
        as a P2-HOTFIX-era audit guardrail. P2-PERF-7 (v5.31 agile-plan
        card) widens it to membership in the sanctioned set
        ``{100, 2048, 4096, 8192}``:
          * ``100`` is the pre-P2-PERF-7 cap (preserved for backwards
            compatibility while the cap raise is in-flight).
          * ``{2048, 4096, 8192}`` is the orchestrator-sanctioned set
            inherited from the retired
            ``P2-HOTFIX-iceberg-cap-overflow-FOLLOWUP`` story (recovered
            from commit ``0a89cfa``).
          * ``8192`` is the P2-PERF-7 pinned target value (next power
            of 2 above the empirical session-max of 5007
            ICE-with-hidden submissions on 2025-10-18 cid=62).
        Any value outside this set indicates an unauthorised drift.
        """
        assert MAX_ICEBERGS in {100, 2048, 4096, 8192}, (
            f"MAX_ICEBERGS={MAX_ICEBERGS} not in sanctioned set "
            f"{{100, 2048, 4096, 8192}} "
            f"(P2-PERF-7 v5.31 / agile-plan)"
        )

    def test_omie_order_field_dtypes_unchanged(self):
        """`OMIEOrder` field annotations must be unchanged by this
        hotfix (audit guardrail)."""
        annotations = OMIEOrder.__annotations__
        assert annotations["order_id"] is int
        assert annotations["agent_id"] is int
        assert annotations["price"] is float
        assert annotations["quantity"] is float
        assert annotations["peak_qty"] is float
        assert annotations["ppd"] is float
        assert annotations["timestamp"] is int


# ---------------------------------------------------------------------------
# 7. Public-API shape pinning
# ---------------------------------------------------------------------------


class TestPublicAPIShape:
    def test_register_returns_3_tuple(self, manager):
        # P2-PERF-6 canonical signature: register(meta, n_active, order)
        # -> (meta, n_active, overflow). The pre-(2a) 2-tuple shape is
        # retired together with the polymorphic dispatch shim.
        result = manager.register(
            manager.empty_meta(), jnp.int32(0), _ice_order(order_id=1)
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_register_return_dtypes(self, manager):
        meta_out, n_active_out, ovf = manager.register(
            manager.empty_meta(), jnp.int32(0), _ice_order(order_id=1)
        )
        assert isinstance(meta_out, jnp.ndarray)
        assert meta_out.dtype == jnp.int32
        assert meta_out.shape == (MAX_ICEBERGS, 8)
        # n_active is a 0-d JAX int32 scalar.
        assert isinstance(n_active_out, jnp.ndarray)
        assert n_active_out.dtype == jnp.int32
        assert n_active_out.shape == ()
        # overflow is a 0-d JAX bool scalar.
        assert isinstance(ovf, jnp.ndarray)
        assert ovf.dtype == jnp.bool_
        assert ovf.shape == ()

    def test_build_state_overflow_kwarg_defaults_to_false(self, book_and_state):
        """Pre-existing call sites that don't pass `overflow` must still
        be valid — i.e. the kwarg has a default."""
        book, state = book_and_state
        sig = inspect.signature(book._build_state)
        param = sig.parameters.get("overflow")
        assert param is not None, "_build_state must accept `overflow` kwarg"
        assert param.default is False, (
            f"`overflow` default must be False (got {param.default!r}) so "
            f"pre-existing call sites remain valid"
        )

        # And functionally: calling without overflow on a clean meta
        # produces truncated=False.
        s = book._build_state(
            state.lob, state.iceberg_meta, state.ts_counter,
            jnp.bool_(False), state.n_active,
        )
        assert bool(s.truncated) is False
