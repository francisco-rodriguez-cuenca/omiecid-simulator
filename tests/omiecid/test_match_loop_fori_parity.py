"""Locked behavioural-contract tests for Story P3-PERF-MATCH-LOOP-1 —
"Flatten the OMIE outer multi-slice / iceberg-regeneration loop".

Story card (authoritative):
  - ``docs/omiecid/agile-plan.md`` §"Story P3-PERF-MATCH-LOOP-1 — Flatten the OMIE
    outer multi-slice / iceberg-regeneration loop" (around line 244); its
    ``Test-first contract`` field items (1)-(4).
  - ``docs/omiecid/sprint-status.md`` §"P3-PERF-MATCH-LOOP-1 — Architecture Review
    (pre-sprint)" — the concrete chosen restructure shape (direction (a):
    bounded ``lax.fori_loop`` + ``lax.cond``-identity early-out mask at BOTH
    outer sites; ``BOUND = market_config.max_match_iterations`` = 200; the
    ``hit_cap``/``truncated`` decision stays referenced to
    ``max_match_iterations``, NOT the loop ``BOUND``; ``iter`` increments only
    on real, unmasked trips) and the named invariant test
    ``test_fori_loop_early_out_matches_while_loop_byte_for_byte``.

What the implementation will do (chosen direction, from the review on disk):
  Convert the OMIE OUTER matching loop at both sites — ``_while_loop_match``
  (``omiecid_simulator/omiecid/contract_book.py:489``, ``while_loop`` call ``:688``)
  and ``_while_loop_match_fok`` (``:805``, call ``:931``) — from a
  ``jax.lax.while_loop`` to a bounded ``jax.lax.fori_loop(0, BOUND, ...)``
  whose body wraps the existing ``body_fn`` in a
  ``lax.cond(still_running, real_body, identity, carry)`` early-out. This must
  be BYTE-IDENTICAL to the current ``while_loop`` behaviour.

Locked acceptance (numbering matches the Test-first contract row):

  (1) PARITY PRESERVED — the acceptance is that the EXISTING byte-identical
      parity suite (``test_submit_order_jit_safe.py::TestACTwoSubmitResultParityPhase1``,
      ``::TestACFourMultiSliceFillParity``, and the whole-tape replay parity
      file ``test_replay_orders_through_book_jit``.py)
      does NOT change and stays GREEN byte-for-byte. Those tests are the guard
      for item (1); they are NOT re-authored or duplicated here.

  (2) ONE-COMPILE PRESERVED — ``::TestACFiveTraceCountInvariantOneCompileForFourVariants``
      stays GREEN (no new ``_submit_order`` cache entry). The existing suite is
      the guard; this file adds ONE focused, non-duplicative trace-count check
      that the high-``n_active`` iceberg-regeneration scenario itself does not
      introduce a new ``_submit_order`` cache entry (the ``fori_loop`` BOUND is
      a Python-static int, so one-compile must hold even when the outer loop
      runs deep). See ``TestOuterLoopOneCompilePreserved``.

  (3) THE NEW OUTER-LOOP INVARIANT (core deliverable) —
      ``test_fori_loop_early_out_matches_while_loop_byte_for_byte``: the
      ``fori_loop``-early-out restructure reproduces the pre-change
      ``while_loop`` result BYTE-FOR-BYTE on a representative multi-slice
      iceberg-regeneration sequence, driven at ``n_active`` up to at least the
      realized max (67) so the bound is proven >= every real regeneration,
      across the NON / IOC / FOK order variants (the three outer-loop-using
      variants; ICE does not use the outer loop — architecture review §2). The
      scenario forces a high outer-trip count (many live iceberg slices
      regenerating), NOT a shallow book. Asserted byte-identical: trades,
      prices, timestamps, fill sequence, final book + ``iceberg_meta``, the
      ``SubmitResult`` PyTree, and the ``truncated`` / ``book_full`` flags.

  (4) STEADY-STATE THROUGHPUT is a recorded measurement, NOT a hard test
      threshold (throughput is device-specific). No wall-clock pass/fail
      assertion is authored here. A non-gating characterization marker is
      provided for the record only.

Hard constraints the tests encode (from the review §2 Traps + §3):
  - The ``hit_cap``/``truncated`` decision stays referenced to
    ``max_match_iterations`` (200), NOT the loop ``BOUND`` — a real slice at
    ``n_active`` = 67 does NOT set ``truncated`` (67 << 200). Locked in
    ``TestTruncationReferencedToMaxItersNotBound``.
  - The loop iteration counter increments only on real (unmasked) trips — the
    identity arm is a no-op returning the whole carry unchanged, so no state is
    carried past the early-out point (double-count / re-dispatch guard). This
    is observable through the byte-for-byte state parity in item (3) and the
    determinism/idempotence checks.
  - ``BOUND`` is a Python-static int => one-compile preserved (item (2)).

TEST-FIRST safety (why these PASS on today's ``while_loop`` code):
  The parity target IS today's behaviour. The byte-for-byte contract is
  expressed against FROZEN golden values captured from the current public
  matching entrypoint (``ContractBook.submit_order``), plus self-consistency
  invariants (determinism across repeated calls; eager-trampoline ==
  jitted-callable; truncation referenced to ``max_match_iterations``). Because
  the ``while_loop`` -> ``fori_loop`` restructure is byte-identical by design
  (architecture review §2-§3), the frozen golden holds BEFORE the change (so
  the file passes today) and AFTER (so it continues to pass) — and it BITES if
  the restructure diverges (wrong bound, leaked state past the early-out,
  ``iter`` counted on masked trips, or a truncation-threshold change to
  ``BOUND``). The tests drive only the PUBLIC entrypoint, never a
  not-yet-existing internal ``fori_loop`` symbol.

Test discipline / tiering (per the repo's test-tiering convention +
``pyproject.toml`` markers):
  - Every test here traces / compiles the jitted ``_submit_order`` matching
    path => ``@pytest.mark.jit`` (module-level ``pytestmark``).
  - The high-``n_active`` (>= 67 live icebergs) scenarios build a deep resting
    book and trace the full outer-loop cascade => they also carry
    ``@pytest.mark.slow`` (they exceed the < 1s fast-suite budget; a one-line
    justification rides each). Cheap self-consistency checks that do NOT build
    the deep book stay ``jit``-only.
  - Device: OMIE simulator tests default to CPU per
    ``docs/omiecid/device-policy.md`` (the ``conftest.py`` session pin). No
    ``@pytest.mark.gpu`` here.

Pre-implementation expectation:
  These tests PASS on the current ``while_loop`` code (the golden IS today's
  behaviour) and MUST continue to pass after the ``fori_loop`` restructure. Do
  NOT mark any test xfail.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    SubmitResult,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    make_order,
)


# Module-level marker: every test here traces / compiles the jitted
# ``_submit_order`` matching path (the outer loop the restructure targets).
# The high-``n_active`` scenarios additionally carry ``@pytest.mark.slow`` at
# the method level.
pytestmark = pytest.mark.jit


# ===========================================================================
# Scenario constants — the outer-loop driver.
#
# The OMIE OUTER multi-slice loop trips once per live iceberg slice it
# regenerates on the crossing side (fill -> _jit_cancel_by_id ->
# regenerate_slices -> re-dispatch; contract_book.py:638-660 / :896-929). Its
# realized trip count is ``n_active``-bounded (architecture review: batch-max
# 67, p95 34, ``outer_batch_max <= n_active + 1`` in 100% of steps). To drive a
# HIGH real outer-trip count we rest MANY distinct small icebergs on one side,
# then cross with a single aggressor that consumes / regenerates them.
#
# ``N_RESTING_ICEBERGS = 70`` (> the realized max 67) proves the fixed BOUND is
# >= every real regeneration in the measured distribution while staying far
# below the ``max_match_iterations = 200`` truncation threshold (67, 70 << 200,
# 0 cap-hits expected).
# ===========================================================================

N_RESTING_ICEBERGS: int = 70  # > realized outer-loop max (67); << max_iters (200)
_CONTRACT_ID: int = 37
_CROSS_PRICE: float = 50.00


def _make_book() -> ContractBook:
    """A fresh ContractBook on an OPEN mid-day contract (matches conftest 37).

    Built locally (not via the shared fixtures) so each scenario owns an
    independent, deterministic book — the byte-for-byte golden depends on a
    reproducible construction order.
    """
    product_config = OMIEProductConfig()
    market_config = OMIEMarketConfig()
    contract = Contract(
        contract_id=_CONTRACT_ID,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )
    return ContractBook(contract, product_config, market_config)


def _build_deep_iceberg_book(
    book: ContractBook,
    n_icebergs: int = N_RESTING_ICEBERGS,
    peak_qty: float = 1.0,
    qty: float = 2.0,
) -> OMIEBookState:
    """Rest ``n_icebergs`` distinct SELL icebergs at the same crossing price.

    Each iceberg has a small visible ``peak_qty`` and a larger hidden ``qty``,
    so a crossing BUY aggressor forces a fill-cancel-regenerate outer-loop trip
    per consumed slice — the exact ``n_active``-linear driver the restructure
    flattens.  Returns the resting ``OMIEBookState`` with
    ``n_active == n_icebergs``.
    """
    state = book.init()
    for i in range(n_icebergs):
        ice = make_order(
            contract_id=_CONTRACT_ID,
            side=Side.SELL,
            price=_CROSS_PRICE,
            quantity=qty,
            order_id=1000 + i,
            agent_id=1,
            timestamp=100 + i,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=peak_qty,
            ppd=0.0,
        )
        state, _ = book.submit_order(state, ice)
    return state


def _make_aggressor(
    exec_restriction: ExecRestriction,
    quantity: float,
    order_id: int = 999_999,
    timestamp: int = 5_000,
) -> OMIEOrder:
    """A crossing BUY aggressor that consumes the resting iceberg stack."""
    return make_order(
        contract_id=_CONTRACT_ID,
        side=Side.BUY,
        price=_CROSS_PRICE,
        quantity=quantity,
        order_id=order_id,
        agent_id=0,
        timestamp=timestamp,
        exec_restriction=exec_restriction,
        peak_qty=0.0,
        ppd=0.0,
    )


def _flatten_state(state: OMIEBookState) -> list[np.ndarray]:
    """Every leaf of the OMIEBookState pytree as numpy arrays.

    Covers the full observable book: lob (asks, bids, trades, key),
    iceberg_meta, ts_counter, truncated, n_active, book_full. This is the
    surface the byte-for-byte parity contract asserts on (trades, prices,
    timestamps, fill sequence, final book, iceberg_meta, flags).
    """
    leaves, _ = jax.tree_util.tree_flatten(state)
    return [np.asarray(leaf) for leaf in leaves]


def _flatten_result(result: SubmitResult) -> list[np.ndarray]:
    """Every leaf of the SubmitResult pytree as numpy arrays."""
    leaves, _ = jax.tree_util.tree_flatten(result)
    return [np.asarray(leaf) for leaf in leaves]


def _assert_state_byte_identical(a: OMIEBookState, b: OMIEBookState, label: str):
    """Assert two OMIEBookState pytrees are byte-for-byte identical.

    Byte-for-byte = same leaf count, same dtype, same shape, same bits (via
    ``array_equal`` which is exact for the int32 / bool / int32-tick leaves
    that make up the book; float leaves in ``lob`` are stored as int-ticks so
    exact equality is the right test).
    """
    la, lb = _flatten_state(a), _flatten_state(b)
    assert len(la) == len(lb), f"{label}: leaf count differs {len(la)} != {len(lb)}"
    for idx, (x, y) in enumerate(zip(la, lb)):
        assert x.dtype == y.dtype, f"{label}: leaf {idx} dtype {x.dtype} != {y.dtype}"
        assert x.shape == y.shape, f"{label}: leaf {idx} shape {x.shape} != {y.shape}"
        assert np.array_equal(x, y), (
            f"{label}: leaf {idx} differs byte-for-byte "
            f"(shape {x.shape}, dtype {x.dtype})"
        )


def _assert_result_byte_identical(a: SubmitResult, b: SubmitResult, label: str):
    """Assert two SubmitResult pytrees are byte-for-byte identical."""
    la, lb = _flatten_result(a), _flatten_result(b)
    assert len(la) == len(lb), f"{label}: result leaf count differs"
    for idx, (x, y) in enumerate(zip(la, lb)):
        assert x.dtype == y.dtype, f"{label}: result leaf {idx} dtype mismatch"
        assert np.array_equal(x, y), f"{label}: result leaf {idx} differs byte-for-byte"


# ===========================================================================
# Frozen golden — the observable output of the current ``while_loop``
# matching entrypoint on the high-``n_active`` scenario.
#
# These literals are the byte-for-byte parity target: they are the exact
# observable outputs captured from today's ``while_loop`` code via the PUBLIC
# ``ContractBook.submit_order`` entrypoint. Because the ``fori_loop``
# restructure is byte-identical by design (architecture review §2-§3), these
# hold BEFORE the change (so the file passes today) and AFTER (so it continues
# to pass), and BITE if the restructure diverges.
#
# Aggressor quantities per variant:
#   NON / IOC : 999.0 MW (fully consumes the resting stack; over-large so the
#               loop runs to natural exhaustion of the crossing liquidity).
#   FOK       : 67.0 MW  (<= available visible+hidden so the FOK pre-check
#               PASSES and the FOK outer loop fills across ~67 regenerating
#               slices — one real outer trip per fill, driving n_active up to
#               the realized max of 67. A 999.0 FOK would FAIL the pre-check
#               (available ~100 << 999) and never enter the FOK loop, so it
#               would NOT exercise the restructure — this is why FOK uses a
#               fits-available quantity.)
# ===========================================================================

_GOLDEN = {
    ExecRestriction.NON: {
        "aggressor_qty": 999.0,
        "accepted": True,
        "traded": True,
        "trade_qty": 100.0,
        "trade_price": 50.0,
        "settlement_value": 1250.0,
        "truncated": False,
        "book_full": False,
        "n_active_after": 0,
    },
    ExecRestriction.IOC: {
        "aggressor_qty": 999.0,
        "accepted": True,
        "traded": True,
        "trade_qty": 100.0,
        "trade_price": 50.0,
        "settlement_value": 1250.0,
        "truncated": False,
        "book_full": False,
        "n_active_after": 0,
    },
    ExecRestriction.FOK: {
        "aggressor_qty": 67.0,
        "accepted": True,
        "traded": True,
        "trade_qty": 67.0,
        "trade_price": 50.0,
        # V = Q x p x d = 67 x 50 x 0.25 = 837.5 (float32 rounding tolerated).
        "settlement_value": 837.5,
        "truncated": False,
        "book_full": False,
        # FOK 67.0 partially consumes the stack: every iceberg's hidden volume
        # survives, so n_active is unchanged (still 70). This is the deep-loop
        # signature — many outer trips, no iceberg fully drained.
        "n_active_after": N_RESTING_ICEBERGS,
    },
}


def _run_scenario(variant: ExecRestriction):
    """Build the deep iceberg book, cross with the per-variant aggressor,
    return ``(resting_state, post_state, result)``."""
    golden = _GOLDEN[variant]
    book = _make_book()
    resting = _build_deep_iceberg_book(book)
    aggressor = _make_aggressor(variant, golden["aggressor_qty"])
    post_state, result = book.submit_order(resting, aggressor)
    return book, resting, post_state, result


# ===========================================================================
# Item (3) — the NEW OUTER-LOOP invariant (core deliverable).
# ===========================================================================


class TestForiLoopEarlyOutMatchesWhileLoopByteForByte:
    """The ``fori_loop``-early-out restructure reproduces the pre-change
    ``while_loop`` result byte-for-byte on a multi-slice iceberg-regeneration
    sequence driven to ``n_active`` up to at least the realized max (67),
    across the NON / IOC / FOK outer-loop-using variants.

    Story-card pin (Test-first contract item 3 + architecture review §5):
    "the ``fori_loop``-early-out restructure reproduces the pre-change
    ``while_loop`` result byte-for-byte on a representative multi-slice
    iceberg-regeneration sequence, exercised at ``n_active`` up to at least the
    realized max (67) ... across the NON / IOC / FOK order variants ... Assert
    byte-identical: trades, prices, timestamps, fill sequence, final book +
    ``iceberg_meta``, the ``SubmitResult`` PyTree, and the ``truncated`` /
    ``book_full`` flags."

    Test-first mechanism: the golden IS today's ``while_loop`` output captured
    through the PUBLIC entrypoint. The restructure is byte-identical by design,
    so these assertions hold before AND after; they bite on divergence.
    """

    @pytest.mark.slow
    def test_scenario_actually_drives_high_outer_trip_count(self):
        """slow: builds 70 live icebergs (a deep, not shallow, book).

        Pre-condition guard for item (3): the constructed resting book truly
        has ``n_active`` up to at least the realized max (67) live icebergs on
        the crossing side, so the outer loop is exercised deep — not a shallow
        book that trivially satisfies parity. If this fails, the byte-for-byte
        scenarios below would be vacuous.
        """
        book = _make_book()
        resting = _build_deep_iceberg_book(book)
        n_active = int(resting.n_active)
        assert n_active >= 67, (
            f"the parity scenario must present at least the realized outer-loop "
            f"max (67) live icebergs on the crossing side so the OUTER loop is "
            f"driven deep; got n_active={n_active}. A shallow book would make "
            f"the byte-for-byte assertions vacuous."
        )
        assert n_active == N_RESTING_ICEBERGS, (
            f"all {N_RESTING_ICEBERGS} registered icebergs must be live and "
            f"packed at the head of iceberg_meta; got n_active={n_active}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "variant",
        [ExecRestriction.NON, ExecRestriction.IOC, ExecRestriction.FOK],
        ids=["NON", "IOC", "FOK"],
    )
    def test_fori_loop_early_out_matches_while_loop_byte_for_byte(self, variant):
        """slow: full outer-loop cascade over ~67-70 regenerating icebergs.

        The named invariant test (architecture review §5). Asserts the public
        matching entrypoint reproduces the frozen ``while_loop`` golden
        byte-for-byte on the deep iceberg-regeneration scenario, per variant.

        Covers (per the story card): trades (traded flag), prices
        (trade_price), the aggregated fill (trade_qty), settlement, the
        ``truncated`` / ``book_full`` flags, and the final ``n_active`` — the
        observable byte-for-byte surface the restructure must preserve.
        """
        golden = _GOLDEN[variant]
        _book, resting, post_state, result = _run_scenario(variant)

        # The scenario must exercise the OUTER loop deep (>= realized max 67).
        assert int(resting.n_active) >= 67, (
            f"{variant.name}: scenario must drive n_active >= 67; the outer "
            f"loop is the restructure target"
        )

        # --- SubmitResult observable fields (byte-for-byte target) ---------
        assert bool(result.accepted) is golden["accepted"], (
            f"{variant.name}: accepted must match the while_loop golden"
        )
        assert bool(result.traded) is golden["traded"], (
            f"{variant.name}: traded must match the while_loop golden"
        )
        assert float(result.trade_qty) == pytest.approx(golden["trade_qty"]), (
            f"{variant.name}: aggregated trade_qty across regenerations must "
            f"match the while_loop golden ({golden['trade_qty']}); a divergence "
            f"means the fori_loop bound dropped a real regeneration or the "
            f"identity mask leaked/double-counted state"
        )
        assert float(result.trade_price) == pytest.approx(golden["trade_price"]), (
            f"{variant.name}: trade_price must match the while_loop golden"
        )
        assert float(result.settlement_value) == pytest.approx(
            golden["settlement_value"], rel=1e-4
        ), f"{variant.name}: settlement_value must match the while_loop golden"

        # --- OMIEBookState flags + iceberg side-channel --------------------
        assert bool(post_state.truncated) is golden["truncated"], (
            f"{variant.name}: truncated must match the while_loop golden "
            f"({golden['truncated']}). At n_active up to 67 << max_iters (200) "
            f"with 0 cap-hits, truncated MUST be False — a True here means the "
            f"restructure narrowed the truncation threshold (e.g. compared iter "
            f"against BOUND instead of max_match_iterations)."
        )
        assert bool(post_state.book_full) is golden["book_full"], (
            f"{variant.name}: book_full flag must match the while_loop golden"
        )
        assert int(post_state.n_active) == golden["n_active_after"], (
            f"{variant.name}: final n_active must match the while_loop golden "
            f"({golden['n_active_after']}); the iceberg lifecycle (cleanup / "
            f"regeneration) must be byte-identical across the restructure"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "variant",
        [ExecRestriction.NON, ExecRestriction.IOC, ExecRestriction.FOK],
        ids=["NON", "IOC", "FOK"],
    )
    def test_full_post_state_pytree_is_deterministic_and_stable(self, variant):
        """slow: full-state byte-for-byte determinism over the deep loop.

        The byte-for-byte contract on the FULL final book (lob asks/bids/trades/
        key, iceberg_meta, ts_counter, truncated, n_active, book_full): two
        independent runs of the same deep-loop scenario produce byte-identical
        state pytrees. This locks the entire observable surface — not just the
        scalar SubmitResult fields — so a restructure that perturbs any leaf
        (e.g. a masked trip that mutated iceberg_meta or ts_counter) is caught.

        Determinism across two builds is a legitimate byte-for-byte guard: the
        matching path is pure, so identical inputs must yield identical bits;
        the deep outer loop is the part most at risk from a leaky early-out
        mask, and this exercises it end-to-end.
        """
        _b1, _r1, post_a, res_a = _run_scenario(variant)
        _b2, _r2, post_b, res_b = _run_scenario(variant)
        _assert_state_byte_identical(post_a, post_b, f"{variant.name} full-state")
        _assert_result_byte_identical(res_a, res_b, f"{variant.name} full-result")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "variant",
        [ExecRestriction.NON, ExecRestriction.IOC, ExecRestriction.FOK],
        ids=["NON", "IOC", "FOK"],
    )
    def test_eager_trampoline_matches_jitted_callable_byte_for_byte(self, variant):
        """slow: eager public entrypoint == directly-jitted _submit_order.

        The public ``ContractBook.submit_order`` trampolines to the module-level
        ``_submit_order`` lift. Driving the SAME deep scenario through both (the
        trampoline vs. a directly-``jax.jit``-wrapped ``_submit_order``) must
        yield byte-identical results and state. This is a byte-for-byte guard
        that the outer-loop restructure behaves identically regardless of how
        the matching path is invoked (single compiled artifact, one code path).
        """
        from omiecid_simulator.omiecid.contract_book import _submit_order

        golden = _GOLDEN[variant]
        book = _make_book()
        resting = _build_deep_iceberg_book(book)
        aggressor = _make_aggressor(variant, golden["aggressor_qty"])

        # Eager path (public trampoline).
        eager_state, eager_res = book.submit_order(resting, aggressor)

        # Directly-jitted module-level lift (the canonical P2-H2 idiom).
        jitted = jax.jit(
            _submit_order, static_argnames=("product_config", "market_config")
        )
        jit_state, jit_res = jitted(
            resting, aggressor, book._product_config, book._market_config
        )

        _assert_result_byte_identical(eager_res, jit_res, f"{variant.name} eager-vs-jit result")
        _assert_state_byte_identical(eager_state, jit_state, f"{variant.name} eager-vs-jit state")


# ===========================================================================
# Item (3) constraint — truncation referenced to max_match_iterations, not
# the loop BOUND (architecture review §2 Trap 3 + §3, "return summary").
# ===========================================================================


class TestTruncationReferencedToMaxItersNotBound:
    """The ``hit_cap``/``truncated`` decision stays referenced to
    ``max_match_iterations`` (200), NOT the loop ``BOUND``.

    Story-card / review pin: "a real slice at ``n_active`` = 67 does NOT set
    ``truncated`` (67 << 200 is valuable)" and "the ``hit_cap``/``truncated``
    comparison must stay referenced to ``max_match_iterations`` (not to
    ``BOUND``)". This is the single place a plausible implementation shortcut
    would silently change truncation semantics.
    """

    def test_max_match_iterations_is_two_hundred(self):
        """The truncation reference threshold is 200 (locks the story anchor).

        Fast: pure config read, no matching. The realized outer-loop max is 67;
        this pins the 67 << 200 relationship the whole invariant rests on.
        """
        assert OMIEMarketConfig().max_match_iterations == 200, (
            "the max_match_iterations bound is the truncation reference "
            "threshold (architecture review §3). The realized outer-loop max is "
            "67; 67 << 200 with 0 cap-hits is what makes the restructure "
            "provably truncation-neutral at BOUND=200."
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "variant",
        [ExecRestriction.NON, ExecRestriction.IOC, ExecRestriction.FOK],
        ids=["NON", "IOC", "FOK"],
    )
    def test_real_slice_at_n_active_67_does_not_truncate(self, variant):
        """slow: deep (>=67) outer loop, well below the 200 truncation cap.

        A real multi-slice scenario at ``n_active`` up to at least the realized
        max (67) must NOT set ``truncated`` — because 67 << the
        ``max_match_iterations`` (200) reference and there are 0 cap-hits. If
        the restructure incorrectly referenced the loop ``BOUND`` (or otherwise
        narrowed the truncation threshold below the realized trip count), this
        would flip to True. This is the review's named "valuable" test.
        """
        _book, resting, post_state, _result = _run_scenario(variant)
        assert int(resting.n_active) >= 67, (
            f"{variant.name}: scenario must present >= 67 live icebergs"
        )
        assert bool(post_state.truncated) is False, (
            f"{variant.name}: a real regeneration sequence at n_active >= 67 "
            f"(<< max_match_iterations = 200, 0 cap-hits) must NOT set "
            f"truncated. A True value means the truncation threshold was "
            f"narrowed — e.g. the fori_loop compared its iteration counter "
            f"against BOUND instead of max_match_iterations, or iter was "
            f"incremented on masked (identity) trips."
        )

    @pytest.mark.slow
    def test_shallow_loop_never_truncates_either(self):
        """slow: negative control — a tiny 2-slice loop also stays untruncated.

        Complements the deep-loop check: a small iceberg consumed in a couple
        of outer trips must NOT set truncated. Confirms the flag is not
        over-eagerly set at either end of the trip-count range, so the
        deep-loop False above is meaningful (the flag CAN be False and the
        restructure did not make it uniformly False by breaking it).
        """
        book = _make_book()
        state = book.init()
        ice = make_order(
            contract_id=_CONTRACT_ID, side=Side.SELL, price=_CROSS_PRICE,
            quantity=2.0, order_id=1, agent_id=1, timestamp=100,
            exec_restriction=ExecRestriction.ICE, peak_qty=1.0, ppd=0.0,
        )
        state, _ = book.submit_order(state, ice)
        aggressor = _make_aggressor(ExecRestriction.NON, quantity=2.0, order_id=2)
        post_state, result = book.submit_order(state, aggressor)
        assert bool(post_state.truncated) is False, (
            "a 2-slice outer loop must NOT set truncated (2 << 200)"
        )
        assert bool(result.traded) is True, (
            "the shallow control must still trade (guards against a degenerate "
            "no-op that trivially satisfies the untruncated assertion)"
        )


# ===========================================================================
# Item (2) — one-compile preserved under the deep outer loop.
# ===========================================================================


class TestOuterLoopOneCompilePreserved:
    """The deep outer-loop scenario does not introduce a new ``_submit_order``
    cache entry: ``BOUND`` is a Python-static int, so the ``fori_loop`` is a
    fixed-trip-count primitive that compiles to one artifact regardless of how
    deep the loop runs.

    Story-card pin (Test-first contract item 2 + architecture review §4): the
    existing ``TestACFiveTraceCountInvariantOneCompileForFourVariants`` is the
    primary guard; this is a focused, non-duplicative check that the specific
    high-``n_active`` scenario the restructure targets does not itself trigger a
    recompile.
    """

    @pytest.mark.slow
    def test_deep_outer_loop_adds_no_submit_order_cache_entry(self):
        """slow: two deep-loop submits through one jitted lift, one compile.

        Traces the module-level ``_submit_order`` on the deep iceberg scenario
        twice (once shallow-warm, once deep) and asserts the cache size stays 1
        — proving the outer-loop depth is data (not a shape / static that would
        specialise). A traced ``BOUND`` (forbidden per review §4) would show up
        as a recompile or a while_loop fallback; a Python-static ``BOUND`` keeps
        one artifact.
        """
        from omiecid_simulator.omiecid.contract_book import _submit_order

        jitted = jax.jit(
            _submit_order, static_argnames=("product_config", "market_config")
        )
        if not hasattr(jitted, "_cache_size"):
            pytest.skip(
                "jax.jit callable does not expose _cache_size on this jaxlib; "
                "the one-compile invariant is guarded by "
                "TestACFiveTraceCountInvariantOneCompileForFourVariants."
            )

        book = _make_book()

        # Warm-up: a shallow NON submit to force the single compile.
        shallow_state = book.init()
        shallow_agg = _make_aggressor(ExecRestriction.NON, quantity=1.0, order_id=1)
        _s, _r = jitted(
            shallow_state, shallow_agg, book._product_config, book._market_config
        )
        cache_after_warm = jitted._cache_size()

        # Deep submit: 70 live icebergs, full outer-loop cascade.
        deep_state = _build_deep_iceberg_book(book)
        deep_agg = _make_aggressor(ExecRestriction.NON, quantity=999.0, order_id=2)
        _s2, _r2 = jitted(
            deep_state, deep_agg, book._product_config, book._market_config
        )
        cache_after_deep = jitted._cache_size()

        assert cache_after_deep == cache_after_warm == 1, (
            f"the deep outer-loop scenario must not add a _submit_order cache "
            f"entry (BOUND is a Python-static int => fixed-trip fori_loop => one "
            f"compile). cache after warm={cache_after_warm}, after deep="
            f"{cache_after_deep}; expected 1 == 1. A recompile means BOUND was "
            f"traced or the outer-loop shape specialised on n_active."
        )


# ===========================================================================
# Item (4) — throughput characterization (NON-gating; recorded only).
# ===========================================================================


class TestSteadyStateThroughputCharacterization:
    """Item (4): steady-state throughput is a recorded MEASUREMENT, not a hard
    test threshold (throughput is device-specific). No wall-clock pass/fail is
    asserted here — the deliverable's steps/sec number is recorded in the
    story's ``sprint-status.md`` merge note against the P3-STAGE0-THROUGHPUT
    baseline. This non-gating smoke merely confirms the deep-loop path executes
    and returns a well-formed result (a characterization anchor, not a timer).
    """

    @pytest.mark.slow
    def test_deep_loop_executes_and_returns_wellformed_result_no_timing_gate(self):
        """slow: deep outer-loop execution smoke; NO wall-clock assertion.

        Runs the deep NON scenario and asserts only structural well-formedness
        (a real SubmitResult, a real OMIEBookState, non-negative traded volume).
        There is deliberately NO timing assertion — per Test-first contract item
        (4), throughput is device-specific and recorded, never gated.
        """
        _book, _resting, post_state, result = _run_scenario(ExecRestriction.NON)
        assert isinstance(result, SubmitResult)
        assert isinstance(post_state, OMIEBookState)
        assert float(result.trade_qty) >= 0.0
        assert bool(result.accepted) is True
