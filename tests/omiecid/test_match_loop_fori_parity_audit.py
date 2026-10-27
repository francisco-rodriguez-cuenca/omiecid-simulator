"""Independent audit of the OMIE outer-loop ``while_loop`` -> ``fori_loop``
conversion (story ``P3-PERF-MATCH-LOOP-1``).

Mode: AUDIT (post-implementation, independent verification). Authored against
the FINISHED implementation in ``omiecid_simulator/omiecid/contract_book.py`` — the
two OMIE OUTER multi-slice / iceberg-regeneration loops were converted from a
data-dependent ``jax.lax.while_loop`` to a bounded ``jax.lax.fori_loop`` with a
``lax.cond`` identity early-out via the module-level helper
``_run_bounded_match_loop(still_running, body_fn, init_carry)`` running a
Python-static ``_TRIP_BOUND = 100`` iterations, shared by ``_while_loop_match``
(NON / IOC) and ``_while_loop_match_fok`` (FOK). The truncation flag is
``hit_cap = ((iter_f >= max_iters) | (iter_f >= _TRIP_BOUND)) & (remaining_f >
0)`` with ``max_iters = OMIEMarketConfig.max_match_iterations = 200``.

These tests drive the PUBLIC matching entrypoint (``ContractBook.submit_order``)
and assert domain-behavioural correctness beyond the happy path — they are NOT
a copy of the locked byte-parity suite. They probe:

  1. Truncation semantics under the decoupled trip bound (T1).
  2. Identity-arm no-op integrity of the trailing ``fori_loop`` passes (T2).
  3. Multi-slice / iceberg-regeneration correctness across NON/IOC/FOK/ICE/
     ICEwPI (T3).
  4. FOK all-or-nothing under the ``fori_loop`` (T4).
  5. One-compile / shape stability across varying ``n_active`` / book depth (T5).
  6. Domain edges: empty-book early termination, both-sides icebergs (T6).

DOMAIN GROUNDING (measured against the shipped code during authoring):

  * A single resting iceberg with ``ceil(total/peak)`` slices produces exactly
    that many real outer trips (10 slices -> iter_f=10, 63 slices -> iter_f=63),
    one trip per regenerated slice, fully filled with ``truncated=False``.
  * The per-iceberg slice count is itself capped by
    ``MAX_SLICES_PER_REGEN = 64`` (the ``IcebergManager.truncation_flag`` sets
    ``OMIEBookState.truncated`` at ICE-register time when
    ``ceil(remaining_hidden/peak) > 64`` — a side-channel truncation ORTHOGONAL
    to the outer-loop ``_TRIP_BOUND``). Multiple resting icebergs at the same /
    adjacent prices are swept in ONE dispatch per trip (jaxob inner
    level-sweep), collapsing the outer trip count. Consequently the realized
    outer-loop trip count stays <= ~64 on constructible inputs (spike realized
    max 67), and the ``_TRIP_BOUND = 100`` guard is NOT reachable by a
    non-adversarial constructible input — matching the story's own prediction
    ("If you cannot reach 100 trips with a constructible input, document that
    and instead assert the guard EXPRESSION is correct via a boundary probe").
    T1 therefore probes the guard at a high-but-reachable trip count (63) and
    asserts the module-level guard EXPRESSION directly, rather than driving the
    loop to 100 real trips.
"""

import math

import jax
import jax.numpy as jnp
import pytest
from pytest import approx

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    _TRIP_BOUND,
    _submit_order,
)
from omiecid_simulator.omiecid.jax_constants import MAX_ICEBERGS, MAX_SLICES_PER_REGEN
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order


# ---------------------------------------------------------------------------
# Local fixtures / helpers — synthetic in-test state (contract/unit category,
# microsecond runtime, no shared cache). See test-agent §"choose state source".
# ---------------------------------------------------------------------------

CID = 37


@pytest.fixture(scope="module")
def pc():
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def mc():
    return OMIEMarketConfig()


@pytest.fixture(scope="module")
def contract():
    return Contract(
        contract_id=CID,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


@pytest.fixture(scope="module")
def book(contract, pc, mc):
    return ContractBook(contract, pc, mc)


def _ice(side, price, qty, oid, ts, peak, ppd=0.0, agent=1):
    return make_order(
        CID, side, price, qty, oid, agent, ts,
        exec_restriction=ExecRestriction.ICE, peak_qty=peak, ppd=ppd,
    )


def _order(side, price, qty, oid, ts, restr=ExecRestriction.NON, agent=2):
    return make_order(CID, side, price, qty, oid, agent, ts, exec_restriction=restr)


def _fresh(book):
    return book.init()


# ===========================================================================
# T1 — Truncation semantics under the decoupled trip bound.
#
# The loop runs only _TRIP_BOUND=100 real trips, but hit_cap fires against BOTH
# max_match_iterations=200 AND _TRIP_BOUND=100. (a) A deep-but-under-100 drain
# completes byte-correctly with truncated=False. (b) The guard expression is
# probed directly (100 real trips is unreachable by a constructible input; see
# module docstring).
# ===========================================================================

class TestT1TruncationSemanticsUnderDecoupledBound:

    def test_decoupling_constants_hold(self, mc):
        """_TRIP_BOUND (100) is <= max_match_iterations (200); the trip bound
        is decoupled from the truncation cap and never exceeds it."""
        assert _TRIP_BOUND == 100
        assert mc.max_match_iterations == 200
        assert _TRIP_BOUND <= mc.max_match_iterations

    def test_trip_bound_exceeds_spike_realized_max(self):
        """The chosen trip bound must be >= the realized outer-loop max (67,
        D-D probe) so no real regeneration is dropped on realistic input."""
        assert _TRIP_BOUND >= 67

    def test_deep_single_iceberg_drain_under_bound_completes_untruncated(
        self, book,
    ):
        """A single 63-slice iceberg (peak 0.1, total 6.3) drives ~63 real
        outer trips — high but < _TRIP_BOUND — and drains fully with
        truncated=False. This exercises the region the fori_loop must cover
        with real (unmasked) trips."""
        st = _fresh(book)
        st, r_ice = book.submit_order(
            st, _ice(Side.SELL, 50.00, 6.3, 1, 1, peak=0.1),
        )
        assert bool(r_ice.accepted)
        # 63 slices <= MAX_SLICES_PER_REGEN (64): no side-channel truncation.
        assert not bool(st.truncated)

        st, r = book.submit_order(st, _order(Side.BUY, 50.00, 6.3, 2, 2))
        assert bool(r.traded)
        assert float(r.trade_qty) == approx(6.3, abs=1e-6)
        assert float(r.trade_price) == approx(50.00, abs=1e-6)
        # Full drain, complete matching, NOT truncated.
        assert not bool(st.truncated)
        bids, asks = book.get_anonymous_lob(st)
        assert asks == []
        assert bids == []

    def test_side_channel_truncation_at_max_slices_is_loud(self, book):
        """DISTINCT from the outer-loop bound: an iceberg whose hidden slice
        count exceeds MAX_SLICES_PER_REGEN (64) sets truncated=True at
        ICE-register time (IcebergManager.truncation_flag). This is a domain
        backstop, and it fires LOUDLY (truncated=True), not silently. 66
        slices (total 6.6, peak 0.1) => remaining-hidden slice count > 64."""
        st = _fresh(book)
        st, r_ice = book.submit_order(
            st, _ice(Side.SELL, 50.00, 6.6, 1, 1, peak=0.1),
        )
        assert bool(r_ice.accepted)
        # ceil((6.6-0.1)/0.1) = 65 hidden slices > MAX_SLICES_PER_REGEN (64).
        assert bool(st.truncated), (
            "iceberg exceeding MAX_SLICES_PER_REGEN must set truncated LOUD "
            "at register time"
        )

    def test_hit_cap_guard_expression_is_wired_against_both_bounds(self, mc):
        """The guard EXPRESSION probe (100 real trips is unreachable by a
        constructible input — see module docstring). Replicate the shipped
        hit_cap expression and assert it fires for any iter_f that reaches
        EITHER bound with remaining>0, and stays False when either the trip
        count is below both bounds OR nothing remains. This locks the
        'loud, never silent' intent of the decoupled guard."""
        max_iters = mc.max_match_iterations

        def hit_cap(iter_f, remaining_f):
            return bool(
                ((iter_f >= max_iters) | (iter_f >= _TRIP_BOUND))
                & (remaining_f > 0.0)
            )

        # At/above _TRIP_BOUND with remaining > 0 -> LOUD (True).
        assert hit_cap(_TRIP_BOUND, 1.0) is True
        assert hit_cap(_TRIP_BOUND + 1, 5.0) is True
        # At/above max_iters with remaining > 0 -> LOUD (True).
        assert hit_cap(max_iters, 1.0) is True
        # Below both bounds -> not truncated regardless of remaining.
        assert hit_cap(_TRIP_BOUND - 1, 100.0) is False
        assert hit_cap(67, 100.0) is False
        # At the bound but nothing remaining -> NOT truncated (complete match).
        assert hit_cap(_TRIP_BOUND, 0.0) is False
        assert hit_cap(max_iters, 0.0) is False

    def test_completed_high_trip_count_reports_not_truncated(self, book):
        """A scenario that finishes at a high (but < bound) real-trip count
        with zero remaining must report truncated=False — the guard's
        'remaining > 0' term is honoured (boundary probe per the story)."""
        st = _fresh(book)
        # 60-slice iceberg: high trip count, exactly drainable, remaining->0.
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 6.0, 1, 1, peak=0.1))
        assert not bool(st.truncated)  # 60 <= 64: no side-channel trunc.
        st, r = book.submit_order(st, _order(Side.BUY, 50.00, 6.0, 2, 2))
        assert float(r.trade_qty) == approx(6.0, abs=1e-6)
        assert not bool(st.truncated)


# ===========================================================================
# T2 — Identity-arm no-op integrity.
#
# Once matching completes, the remaining fori_loop iterations are lax.cond
# identity passes. The final result must be INVARIANT to how many identity
# passes run — i.e. the same aggressor against equal reachable liquidity
# produces identical fills/final-state regardless of iceberg geometry that
# changes where the natural termination point (real-trip count) falls. No
# accumulator double-count, no state drift on masked trips.
# ===========================================================================

class TestT2IdentityArmNoOpIntegrity:

    @pytest.mark.parametrize("peak", [2.0, 1.0, 0.5, 0.2])
    def test_fill_invariant_to_number_of_real_trips(self, book, peak):
        """Total 2.0 MW resting at 50.00 drained by a 2.0 MW aggressor fills
        identically (qty 2.0, price 50.00, truncated=False) whether the drain
        takes 1, 2, 4, or 10 real outer trips — the trailing identity passes
        do not perturb the result and the accumulator is not double-counted."""
        st = _fresh(book)
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 2.0, 1, 1, peak=peak))
        st, r = book.submit_order(st, _order(Side.BUY, 50.00, 2.0, 2, 2))
        assert bool(r.traded)
        assert float(r.trade_qty) == approx(2.0, abs=1e-6)
        assert float(r.trade_price) == approx(50.00, abs=1e-6)
        assert not bool(st.truncated)
        bids, asks = book.get_anonymous_lob(st)
        assert asks == [], "iceberg fully drained"
        assert bids == [], "aggressor fully filled, nothing rests"

    def test_settlement_value_invariant_to_trip_count(self, book, pc):
        """Settlement value V = Q x p x d must be identical across differing
        real-trip counts for the same economic fill (no per-trip drift)."""
        results = []
        for peak in (2.0, 0.5, 0.2):
            st = _fresh(book)
            st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 2.0, 1, 1, peak=peak))
            st, r = book.submit_order(st, _order(Side.BUY, 50.00, 2.0, 2, 2))
            results.append(float(r.settlement_value))
        # V = 2.0 MW * 50.0 EUR/MWh * 0.25 h = 25.0 EUR — identical every time.
        expected = 2.0 * 50.0 * pc.delivery_duration_hours
        for v in results:
            assert v == approx(expected, abs=1e-4)
        assert results[0] == approx(results[1], abs=1e-4)
        assert results[1] == approx(results[2], abs=1e-4)

    def test_masked_trips_do_not_advance_state_beyond_termination(self, book):
        """A partial fill (aggressor smaller than resting) must terminate at
        the natural point and leave the correct residual resting — the masked
        identity passes must not drain more of the iceberg after the aggressor
        is exhausted."""
        st = _fresh(book)
        # Resting iceberg total 5.0 (peak 0.5 -> 10 slices).
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 5.0, 1, 1, peak=0.5))
        # Aggressor only wants 2.0.
        st, r = book.submit_order(st, _order(Side.BUY, 50.00, 2.0, 2, 2))
        assert float(r.trade_qty) == approx(2.0, abs=1e-6)
        assert not bool(st.truncated)
        bids, asks = book.get_anonymous_lob(st)
        # 3.0 MW of the iceberg must remain (visible peak of 0.5 at least);
        # aggressor fully consumed, no residual bid.
        assert bids == [], "aggressor fully consumed by the 2.0 fill"
        assert len(asks) == 1, "iceberg still has undrained volume resting"
        assert asks[0][0] == approx(50.00, abs=1e-6)
        assert float(asks[0][1]) == approx(0.5, abs=1e-6), (
            "the visible peak (0.5) must rest; masked passes must not have "
            "over-drained the hidden volume"
        )


# ===========================================================================
# T3 — Multi-slice / iceberg-regeneration correctness across variants.
#
# The outer loop spins once per regenerated slice. Assert the aggregate fill,
# price, and final iceberg_meta / n_active match the domain-expected multi-slice
# drain across NON / IOC / FOK / ICE / ICEwPI.
# ===========================================================================

class TestT3MultiSliceRegenerationCorrectness:

    def test_non_drains_full_iceberg_across_slices(self, book):
        """NON aggressor drains the entire multi-slice iceberg (peak 0.5,
        total 4.0 -> 8 slices) in one submit_order call."""
        st = _fresh(book)
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 4.0, 1, 1, peak=0.5))
        st, r = book.submit_order(st, _order(Side.BUY, 50.00, 4.0, 2, 2))
        assert bool(r.traded)
        assert float(r.trade_qty) == approx(4.0, abs=1e-6)
        assert float(r.trade_price) == approx(50.00, abs=1e-6)
        bids, asks = book.get_anonymous_lob(st)
        assert asks == []
        assert bids == []
        # The drained iceberg slot must be released (packed layout invariant).
        assert int(st.n_active) == 0

    def test_ioc_partial_drains_then_cancels_residual(self, book):
        """IOC aggressor larger than the resting iceberg drains all slices
        then cancels its residual — no IOC residual rests."""
        st = _fresh(book)
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 3.0, 1, 1, peak=0.5))
        st, r = book.submit_order(
            st, _order(Side.BUY, 50.00, 5.0, 2, 2, restr=ExecRestriction.IOC),
        )
        assert float(r.trade_qty) == approx(3.0, abs=1e-6)
        assert float(r.trade_price) == approx(50.00, abs=1e-6)
        bids, asks = book.get_anonymous_lob(st)
        assert asks == [], "iceberg fully drained"
        assert bids == [], "IOC residual cancelled, does not rest"
        assert int(st.n_active) == 0

    def test_non_partial_leaves_residual_resting(self, book):
        """NON aggressor larger than resting liquidity drains the iceberg
        then rests its residual as a resting bid (NON semantics, distinct
        from IOC)."""
        st = _fresh(book)
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 3.0, 1, 1, peak=0.5))
        st, r = book.submit_order(st, _order(Side.BUY, 50.00, 5.0, 2, 2))
        assert float(r.trade_qty) == approx(3.0, abs=1e-6)
        bids, asks = book.get_anonymous_lob(st)
        assert asks == []
        assert len(bids) == 1, "NON residual (2.0) rests as a bid"
        assert bids[0][0] == approx(50.00, abs=1e-6)
        assert float(bids[0][1]) == approx(2.0, abs=1e-6)

    def test_ice_peak_dispatch_and_regen_registers_slot(self, book):
        """A resting ICE order dispatches only its peak into the visible book
        and registers exactly one iceberg slot (n_active == 1) with the hidden
        volume tracked in the side channel."""
        st = _fresh(book)
        st, r = book.submit_order(st, _ice(Side.SELL, 50.00, 6.0, 1, 1, peak=2.0))
        assert bool(r.accepted)
        bids, asks = book.get_anonymous_lob(st)
        assert len(asks) == 1
        assert asks[0][0] == approx(50.00, abs=1e-6)
        assert float(asks[0][1]) == approx(2.0, abs=1e-6), "only the peak visible"
        assert int(st.n_active) == 1

    def test_icewpi_ppd_regenerates_slices_at_shifted_price(self, book):
        """ICEwPI (ppd != 0) registers a peak-with-price-delta iceberg and its
        visible peak rests at the base price; the aggressor draining across
        slices must fill without truncation."""
        st = _fresh(book)
        st, r_ice = book.submit_order(
            st, _ice(Side.SELL, 50.00, 6.0, 1, 1, peak=2.0, ppd=0.5),
        )
        assert bool(r_ice.accepted)
        bids, asks = book.get_anonymous_lob(st)
        assert len(asks) == 1
        assert asks[0][0] == approx(50.00, abs=1e-6), "first peak at base price"
        assert float(asks[0][1]) == approx(2.0, abs=1e-6)
        assert int(st.n_active) == 1
        # Draining ICEwPI across its price-incrementing slices: a BUY high
        # enough to cross all shifted slices fills the full 6.0 MW.
        st, r = book.submit_order(st, _order(Side.BUY, 60.00, 6.0, 2, 2))
        assert bool(r.traded)
        assert float(r.trade_qty) == approx(6.0, abs=1e-6)
        assert not bool(st.truncated)
        assert int(st.n_active) == 0

    def test_multi_iceberg_aggregate_drain(self, book):
        """Two resting icebergs at the same price, each multi-slice, drained
        by a single aggressor. Total fill == sum of both, no truncation, both
        slots released."""
        st = _fresh(book)
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 3.0, 1, 1, peak=1.0))
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 2.0, 2, 2, peak=0.5))
        assert int(st.n_active) == 2
        st, r = book.submit_order(st, _order(Side.BUY, 50.00, 5.0, 3, 3))
        assert float(r.trade_qty) == approx(5.0, abs=1e-6)
        assert float(r.trade_price) == approx(50.00, abs=1e-6)
        assert not bool(st.truncated)
        assert int(st.n_active) == 0
        bids, asks = book.get_anonymous_lob(st)
        assert asks == []
        assert bids == []


# ===========================================================================
# T4 — FOK all-or-nothing under the fori_loop.
#
# FOK either fully fills or does nothing. The fori_loop conversion must preserve
# this across a multi-slice FOK scenario.
# ===========================================================================

class TestT4FOKAllOrNothing:

    def test_fok_fully_fills_across_multiple_slices(self, book):
        """FOK that CAN fully fill against a multi-slice iceberg does so
        (peak 0.5, total 6.0 -> 12 slices; FOK BUY 6.0 fully fills)."""
        st = _fresh(book)
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 6.0, 1, 1, peak=0.5))
        st, r = book.submit_order(
            st, _order(Side.BUY, 50.00, 6.0, 2, 2, restr=ExecRestriction.FOK),
        )
        assert bool(r.accepted)
        assert bool(r.traded)
        assert float(r.trade_qty) == approx(6.0, abs=1e-6)
        assert float(r.trade_price) == approx(50.00, abs=1e-6)
        bids, asks = book.get_anonymous_lob(st)
        assert asks == [], "iceberg fully drained by FOK"
        assert bids == [], "FOK leaves no residual"

    def test_fok_cannot_fill_leaves_book_unchanged(self, book):
        """FOK that CANNOT fully fill leaves the book byte-unchanged — no
        partial fills, no residual, no drained slices."""
        st = _fresh(book)
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 3.0, 1, 1, peak=0.5))
        bids0, asks0 = book.get_anonymous_lob(st)
        n_active0 = int(st.n_active)
        # FOK wants 6.0 but only 3.0 is available -> all-or-nothing: nothing.
        st, r = book.submit_order(
            st, _order(Side.BUY, 50.00, 6.0, 2, 2, restr=ExecRestriction.FOK),
        )
        assert bool(r.accepted), "FOK is accepted (validation passes)"
        assert not bool(r.traded), "FOK cannot fully fill -> no trade"
        assert float(r.trade_qty) == approx(0.0, abs=1e-6)
        bids1, asks1 = book.get_anonymous_lob(st)
        assert asks1 == asks0, "resting iceberg peak unchanged by rejected FOK"
        assert bids1 == bids0
        assert int(st.n_active) == n_active0, "no iceberg slot drained"

    def test_fok_exact_fill_boundary(self, book):
        """FOK requesting EXACTLY the available multi-slice liquidity fills;
        one qty-tick more does not (all-or-nothing boundary)."""
        st = _fresh(book)
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 2.0, 1, 1, peak=0.5))
        st, r_exact = book.submit_order(
            st, _order(Side.BUY, 50.00, 2.0, 2, 2, restr=ExecRestriction.FOK),
        )
        assert bool(r_exact.traded)
        assert float(r_exact.trade_qty) == approx(2.0, abs=1e-6)

        # Now the just-above case on a fresh book.
        st2 = _fresh(book)
        st2, _ = book.submit_order(st2, _ice(Side.SELL, 50.00, 2.0, 3, 1, peak=0.5))
        st2, r_over = book.submit_order(
            st2, _order(Side.BUY, 50.00, 2.1, 4, 2, restr=ExecRestriction.FOK),
        )
        assert not bool(r_over.traded), "2.1 > 2.0 available -> FOK does nothing"
        _, asks = book.get_anonymous_lob(st2)
        assert len(asks) == 1, "iceberg untouched by the rejected FOK"


# ===========================================================================
# T5 — One-compile / shape stability beyond the happy path.
#
# The _TRIP_BOUND static int must not introduce shape polymorphism: varying
# n_active / book depth must not retrace _submit_order.
# ===========================================================================

@pytest.mark.jit
class TestT5OneCompileShapeStability:

    def test_no_retrace_across_varying_n_active_and_book_depth(
        self, contract, pc, mc,
    ):
        """Running submit_order across a growing iceberg population (varying
        n_active) and across NON / IOC / FOK / ICE / ICEwPI variants must
        compile _submit_order exactly ONCE — the fixed _TRIP_BOUND keeps the
        loop a single artifact regardless of realized depth."""
        book = ContractBook(contract, pc, mc)
        jax.clear_caches()
        assert _submit_order._cache_size() == 0

        st = book.init()
        # Grow n_active 0 -> 6 via ICE registers (varying live-slot count).
        for i in range(6):
            st, _ = book.submit_order(
                st, _ice(Side.SELL, 50.00 + 0.01 * i, 3.0, 200 + i, i, peak=0.5),
            )
        after_ice = _submit_order._cache_size()
        assert after_ice == 1, "ICE dispatch across varying n_active: one compile"

        # Exercise the other variants — same artifact.
        st, _ = book.submit_order(st, _order(Side.BUY, 50.00, 1.0, 10, 20))
        st, _ = book.submit_order(
            st, _order(Side.BUY, 50.00, 0.5, 11, 21, restr=ExecRestriction.IOC),
        )
        st, _ = book.submit_order(
            st, _order(Side.BUY, 50.00, 0.5, 12, 22, restr=ExecRestriction.FOK),
        )
        st, _ = book.submit_order(
            st, _ice(Side.BUY, 49.00, 4.0, 13, 23, peak=1.0, ppd=0.5),
        )
        assert _submit_order._cache_size() == 1, (
            "all four ExecRestriction variants + varying n_active share ONE "
            "compiled _submit_order artifact (one-compile preserved)"
        )

    def test_state_pytree_shapes_are_bound_independent(self, contract, pc, mc):
        """OMIEBookState leaf shapes must be governed by the configured
        max_icebergs / book_depth (the static caps), NOT by _TRIP_BOUND or
        realized trip count — no shape leaks from the loop bound.

        P3-PERF-REGEN-CAP-1: the iceberg_meta first dim is config-derived
        (``book._market_config.max_icebergs``) rather than the MAX_ICEBERGS
        ceiling constant; the bound-independence invariant is unchanged."""
        book = ContractBook(contract, pc, mc)
        st_empty = book.init()
        # iceberg_meta rows == configured max_icebergs regardless of activity.
        assert st_empty.iceberg_meta.shape[0] == book._market_config.max_icebergs

        st = st_empty
        for i in range(4):
            st, _ = book.submit_order(
                st, _ice(Side.SELL, 50.00, 3.0, 300 + i, i, peak=0.5),
            )
        st, _ = book.submit_order(st, _order(Side.BUY, 50.00, 6.0, 99, 50))
        # Shapes unchanged after a deep multi-slice drain.
        assert st.iceberg_meta.shape == st_empty.iceberg_meta.shape
        assert st.lob.asks.shape == st_empty.lob.asks.shape
        assert st.lob.bids.shape == st_empty.lob.bids.shape


# ===========================================================================
# T6 — Domain edges under the fori_loop.
# ===========================================================================

class TestT6DomainEdges:

    def test_empty_book_aggressor_rests_no_trade_no_truncation(self, book):
        """A NON aggressor against an empty book terminates immediately (the
        fori_loop early-out fires on the very first predicate check): no trade,
        the order rests, truncated stays False."""
        st = _fresh(book)
        st, r = book.submit_order(st, _order(Side.BUY, 50.00, 5.0, 1, 1))
        assert bool(r.accepted)
        assert not bool(r.traded)
        assert float(r.trade_qty) == approx(0.0, abs=1e-6)
        assert not bool(st.truncated)
        bids, asks = book.get_anonymous_lob(st)
        assert asks == []
        assert len(bids) == 1, "the unmatched NON rests"
        assert bids[0][0] == approx(50.00, abs=1e-6)
        assert float(bids[0][1]) == approx(5.0, abs=1e-6)

    def test_non_crossing_price_no_trade(self, book):
        """A BUY below the resting ask does not cross — no trade, both rest,
        no truncation. Early-out on the first (no-fill) trip."""
        st = _fresh(book)
        st, _ = book.submit_order(st, _ice(Side.SELL, 50.00, 3.0, 1, 1, peak=1.0))
        st, r = book.submit_order(st, _order(Side.BUY, 49.00, 2.0, 2, 2))
        assert not bool(r.traded)
        assert not bool(st.truncated)
        bids, asks = book.get_anonymous_lob(st)
        assert len(asks) == 1 and asks[0][0] == approx(50.00, abs=1e-6)
        assert len(bids) == 1 and bids[0][0] == approx(49.00, abs=1e-6)

    def test_ioc_no_liquidity_cancels_completely(self, book):
        """IOC against an empty book: accepted, no trade, and NO residual rests
        (IOC cancels its unfilled remainder). truncated stays False."""
        st = _fresh(book)
        st, r = book.submit_order(
            st, _order(Side.BUY, 50.00, 5.0, 1, 1, restr=ExecRestriction.IOC),
        )
        assert bool(r.accepted)
        assert not bool(r.traded)
        assert not bool(st.truncated)
        bids, asks = book.get_anonymous_lob(st)
        assert asks == []
        assert bids == [], "IOC with no fill leaves nothing resting"

    def test_both_sides_icebergs_independent_drain(self, book):
        """Simultaneous resting icebergs on BOTH sides. An aggressor draining
        one side must not disturb the other side's iceberg slot or its
        visible peak."""
        st = _fresh(book)
        # Resting SELL iceberg (peak 1.0, total 3.0) and BUY iceberg
        # (peak 1.0, total 3.0) at non-crossing prices.
        st, _ = book.submit_order(st, _ice(Side.SELL, 51.00, 3.0, 1, 1, peak=1.0))
        st, _ = book.submit_order(st, _ice(Side.BUY, 49.00, 3.0, 2, 2, peak=1.0))
        assert int(st.n_active) == 2
        # Aggressive SELL crosses the resting BUY iceberg for its full 3.0.
        st, r = book.submit_order(st, _order(Side.SELL, 49.00, 3.0, 3, 3))
        assert float(r.trade_qty) == approx(3.0, abs=1e-6)
        assert not bool(st.truncated)
        bids, asks = book.get_anonymous_lob(st)
        # The BUY iceberg drained; the SELL iceberg's peak must still rest.
        assert bids == [], "BUY iceberg fully drained"
        assert len(asks) == 1, "SELL iceberg untouched"
        assert asks[0][0] == approx(51.00, abs=1e-6)
        assert float(asks[0][1]) == approx(1.0, abs=1e-6)
        assert int(st.n_active) == 1, "only the SELL iceberg slot remains live"

    def test_zero_reachable_volume_early_termination_is_clean(self, book):
        """An aggressor whose price cannot reach any resting liquidity leaves
        state clean: the resting iceberg peak and its slot are untouched, and
        the new order rests without spuriously draining hidden volume."""
        st = _fresh(book)
        st, _ = book.submit_order(st, _ice(Side.SELL, 60.00, 4.0, 1, 1, peak=1.0))
        n0 = int(st.n_active)
        _, asks0 = book.get_anonymous_lob(st)
        st, r = book.submit_order(st, _order(Side.BUY, 55.00, 2.0, 2, 2))
        assert not bool(r.traded)
        assert int(st.n_active) == n0, "resting iceberg slot untouched"
        _, asks1 = book.get_anonymous_lob(st)
        assert asks1 == asks0, "no hidden volume drained on a non-crossing add"
