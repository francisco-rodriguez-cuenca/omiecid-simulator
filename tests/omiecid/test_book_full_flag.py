"""Locked behavioural-contract tests for Story ``P3-PREP-REPLAY-1`` — Part B
(the diagnostic-flag deliverable).

Story card: ``docs/omiecid/agile-plan.md`` §"Story P3-PREP-REPLAY-1" (v5.46),
the **Test-first contract** cell + **Acceptance (A)–(H)** + **Out-of-scope
(O1)–(O7)**. Architect rulings that pin the deliverable shape:
``docs/omiecid/sprint-status.md`` §"Architect Ruling on Part A (book_depth
sizing)" (Q1–Q5) and §"Q6 — Are Part B tests necessary?" (the
reduced-but-locked subset ruling).

What the story delivers (and what these tests lock)
---------------------------------------------------
Part B adds a NEW distinct **``book_full``** 0-d ``jnp.bool_`` leaf to
``OMIEBookState``. It is set ``True`` when a resting-add fires the
full-side eviction (the silent worst-price level-wipe in jaxob's
``bid_lim`` / ``ask_lim`` ``check_book_fill=True`` branch). The flag is
set via the **O1-compliant** pre-dispatch predicate
``jnp.all(side[:, OrderSideFeat.P.value] >= 0)`` re-derived at the OMIE
dispatch sites and ORed into the per-call flag composition in
``_build_state_jit``.

CRITICAL — the flag makes the loss **OBSERVABLE, NOT CORRECT** (N5
honesty invariant). Part A found the true peak intractable (27–59 % of
order count; cid 81 = 1147/side; window-max ≥ 5556/side), so
``book_depth`` and ``nTrades`` are **both left at 100** and the
busy-contract under-fill is **reclassified to OQ-10 — NOT fixed here**.

Why these tests FAIL on the current (pre-implementation) tree
-------------------------------------------------------------
The ``book_full`` leaf does NOT exist yet. Every test that asserts
against ``state.book_full`` will raise ``AttributeError`` on the
unmodified branch (``OMIEBookState`` is a 5-leaf ``NamedTuple``:
``lob / iceberg_meta / ts_counter / truncated / n_active``) and will
pass once the developer adds the ``book_full`` leaf and wires the
predicate. This is the locked test-first contract: the tests are the
behavioural specification the Implementation Agent must satisfy.

The field is referenced by the name the card uses (``book_full``) so the
developer implements that exact name.

Test coverage map (card §Test-first contract ↔ test classes)
------------------------------------------------------------
  (1)  full-side add is FLAGGED, not silent
       → ``TestFullSideAddIsFlagged``
  (1b) multi-order level-wipe captured (flag fires on the default
       ``check_book_fill=True`` path; count-drop > 1 rides free as a C1
       characterization)
       → ``TestMultiOrderLevelWipeCaptured``
  (2)  no silent liquidity loss below cap (false-positive / off-by-one
       guard)
       → ``TestNoSilentLossBelowCap``
  (3)  busy-cap-breach known-limitation lock (XFAIL — OQ-10 residual)
       → ``TestBusyContractKnownLimitationOQ10``
  (4)  flag observable end-to-end through ``_build_state_jit``
       → ``TestFlagObservableEndToEnd``
  (4b) flag distinct on a ``truncated``-saturated session
       → ``TestFlagDistinctFromTruncated``
  (5)  OPTIONAL real-data slice (skip-if-no-data, non-gating)
       → ``TestRealDataSliceOptional``

NOT authored here (already-locked tripwires that must stay green —
leave untouched): the P2-H3 one-compile invariant
(``test_submit_order_jit_safe.py`` / ``test_full_domain_jit_smoke.py``)
and ``test_replay_orders_through_book_jit.py::
test_final_state_tree_leaf_equal_to_legacy``.

Test discipline
---------------
  - Device: CPU-default per ``docs/omiecid/device-policy.md``. Run with
    ``CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=.
    .venv/bin/python3 -m pytest tests/omiecid/test_book_full_flag.py``.
  - Runtime budget: contract/unit tests build a small synthetic fixture
    in-test and submit through ``ContractBook.submit_order`` in a Python
    loop at the default ``book_depth=100``. The FIRST ``submit_order`` in
    each test pays a one-time XLA compile of the jitted OMIE submit path
    (~1.5–2.2 s wall-clock observed on first ``ContractBook.submit_order``),
    which pushes the per-test time past the < 1 s fast-suite budget; the
    in-loop submits that follow are microseconds. The whole module
    therefore carries ``pytestmark = pytest.mark.jit`` (compile-heavy;
    deselectable with ``-m "not jit"``) — the same idiom
    ``test_book_full_flag_audit.py`` uses. None of these contract/unit
    tests are ``@pytest.mark.slow``; the OPTIONAL real-data slice (test 5)
    is the only ``@pytest.mark.slow`` test here and is unaffected by the
    jit marker.
  - Black-box: imports only the public interface
    (``ContractBook`` / ``OMIEBookState`` / ``make_order`` /
    ``replay_orders_through_book`` / ``OMIEMarketConfig`` /
    ``OrderSideFeat``). No mocks, no Hypothesis. No jaxob-core internals.
  - Resting-count is read from the public state via the same price-column
    ``>= 0`` predicate the spike note and jaxob's own ``full_book_flag``
    use (``OrderSideFeat.P.value``).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, OMIEBookState
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    make_order,
)


# Module-level jit marker (P3-PREP-REPLAY-1-FOLLOWUP-2, folded into
# P3-PREP-TEST-1): each test's first ``submit_order`` triggers a one-time
# XLA compile of the jitted OMIE submit path (~1.5–2.2 s wall-clock),
# exceeding the < 1 s fast-suite budget. Marking the module lets the fast
# feedback loop deselect the compile cost via ``-m "not jit"`` — the same
# idiom ``test_book_full_flag_audit.py`` uses.
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Module constants / data-path probe (test (5) gating)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_OMIE_DIR = _REPO_ROOT / "data" / "rawOMIE"
# The finding's contract / window (book-depth-cap-finding.md). Only used
# by the OPTIONAL, skip-if-no-data test (5).
FINDING_OMIE_DATE = "2025-06-02"
FINDING_CONTRACT_ID = 81
FINDING_ORDERS_FILE = (
    RAW_OMIE_DIR / f"orders_{FINDING_OMIE_DATE.replace('-', '')}.1"
)

CONTRACT_ID = 37  # mirrors the conftest open_contract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def product_config() -> OMIEProductConfig:
    return OMIEProductConfig()


@pytest.fixture
def open_contract() -> Contract:
    """An OPEN contract so submit_order enters the jitted body."""
    return Contract(
        contract_id=CONTRACT_ID,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_book(open_contract, product_config, book_depth=None, nTrades=None):
    """Build a ContractBook, optionally overriding book_depth / nTrades.

    The override path is used only by the busy-contract / known-limitation
    test; the default-config path is used by the cap-boundary tests so
    they exercise the production book_depth=100 the card leaves unchanged.
    """
    mc = OMIEMarketConfig()
    if book_depth is not None or nTrades is not None:
        import dataclasses

        mc = dataclasses.replace(
            mc,
            book_depth=book_depth if book_depth is not None else mc.book_depth,
            nTrades=nTrades if nTrades is not None else mc.nTrades,
        )
    return ContractBook(open_contract, product_config, mc)


def _make_resting_sell(order_id, price, quantity=1.0, agent_id=1, timestamp=1000):
    """A NON sell limit order that rests on the ask side.

    SELL orders submitted at strictly *ascending* distinct prices never
    cross one another, so each one rests — building a clean one-sided book.
    """
    return make_order(
        contract_id=CONTRACT_ID,
        side=Side.SELL,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=ExecRestriction.NON,
    )


def _make_resting_buy(order_id, price, quantity=1.0, agent_id=1, timestamp=1000):
    """A NON buy limit order that rests on the bid side.

    BUY orders submitted at strictly *descending* distinct prices never
    cross one another, so each one rests.
    """
    return make_order(
        contract_id=CONTRACT_ID,
        side=Side.BUY,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=ExecRestriction.NON,
    )


def _ask_resting_count(state: OMIEBookState) -> int:
    """Count resting orders on the ask side.

    Uses the exact price-column ``>= 0`` predicate jaxob computes for its
    own ``full_book_flag`` (``JaxOrderBookArrays.py:485``) and that Part A
    used to measure the true peak. A live slot has a non-negative price in
    column ``OrderSideFeat.P.value``; empty slots carry the ``-1`` sentinel.
    """
    asks = state.lob.asks
    return int(jnp.sum(asks[:, OrderSideFeat.P.value] >= 0))


def _bid_resting_count(state: OMIEBookState) -> int:
    """Count resting orders on the bid side (price-column ``>= 0`` predicate)."""
    bids = state.lob.bids
    return int(jnp.sum(bids[:, OrderSideFeat.P.value] >= 0))


def _submit_all(book, state, orders):
    """Submit a list of orders sequentially, threading state. Returns final state."""
    for order in orders:
        state, _ = book.submit_order(state, order)
    return state


# ---------------------------------------------------------------------------
# (1) full-side add is FLAGGED, not silent
# ---------------------------------------------------------------------------

class TestFullSideAddIsFlagged:
    """Card test (1): submitting ``book_depth + 1`` non-matching resting
    orders on one side must set the ``book_full`` flag.

    Today no flag exists — this is the new contract. The (book_depth+1)th
    add fires the worst-price eviction; ``book_full`` must go True.
    """

    def test_full_side_add_sets_book_full_flag(self, open_contract, product_config):
        book = _make_book(open_contract, product_config)
        book_depth = book._market_config.book_depth  # 100, default

        # book_depth + 1 SELL orders at strictly ascending distinct prices.
        # None cross (all sells, ascending), so each one wants to rest;
        # the (book_depth+1)th fires the full-side eviction.
        orders = [
            _make_resting_sell(order_id=i + 1, price=50.00 + 0.01 * i)
            for i in range(book_depth + 1)
        ]
        state = book.init()
        state = _submit_all(book, state, orders)

        # CONTRACT: the full-side add must be observable via book_full.
        # AttributeError on the unmodified tree; True after the leaf lands.
        assert bool(state.book_full) is True, (
            "submitting book_depth+1 non-matching resting orders fires the "
            "full-side eviction; book_full must be True (today no flag fires "
            "— the bug this story makes observable)."
        )

    def test_full_side_add_on_bid_side_sets_book_full_flag(
        self, open_contract, product_config
    ):
        """Symmetric coverage: the bid-side eviction path (``bid_lim``,
        ``JaxOrderBookArrays.py:395-401``) must set the same flag."""
        book = _make_book(open_contract, product_config)
        book_depth = book._market_config.book_depth

        # Descending distinct buy prices: none cross, each wants to rest.
        orders = [
            _make_resting_buy(order_id=i + 1, price=50.00 - 0.01 * i)
            for i in range(book_depth + 1)
        ]
        state = book.init()
        state = _submit_all(book, state, orders)

        assert bool(state.book_full) is True, (
            "the bid-side full-book eviction must also set book_full."
        )


# ---------------------------------------------------------------------------
# (1b) multi-order level-wipe captured
# ---------------------------------------------------------------------------

class TestMultiOrderLevelWipeCaptured:
    """Card test (1b): fill a side to ``book_depth`` with MULTIPLE orders
    sharing the worst price, submit one more, assert the flag fires.

    Load-bearing assertion (architect Q6 ruling): **the flag fires on the
    ``check_book_fill=True`` (default / level-wipe) path** that real
    replays exercise. The "count-drop > 1" assertion is a *characterization*
    of untouched jaxob C1 level-wipe behaviour
    (``JaxOrderBookArrays.py:398``/``:487`` set every row at ``worst_price``
    to the ``-1`` sentinel) — included as a secondary assertion, but the
    primary lock is "flag fires on the default eviction path".
    """

    def _fill_with_shared_worst_level(self, book):
        """Fill the ask side to exactly ``book_depth`` with the worst
        (highest) price level holding multiple orders.

        Layout: (book_depth - K) orders at distinct cheap prices, then K
        orders all sharing the single worst (highest) price. The next add
        that crosses the cap will wipe the entire worst-price level (all K).
        """
        book_depth = book._market_config.book_depth
        k_at_worst = 3  # multiple orders share the worst price level
        n_cheap = book_depth - k_at_worst

        worst_price = 50.00 + 0.01 * (n_cheap + 1)  # strictly above all cheap
        orders = []
        oid = 1
        for i in range(n_cheap):
            orders.append(
                _make_resting_sell(order_id=oid, price=50.00 + 0.01 * i)
            )
            oid += 1
        # k_at_worst orders all at the SAME worst price (distinct order_ids).
        for _ in range(k_at_worst):
            orders.append(
                _make_resting_sell(order_id=oid, price=worst_price)
            )
            oid += 1

        state = book.init()
        state = _submit_all(book, state, orders)
        return state, k_at_worst

    def test_flag_fires_on_default_check_book_fill_path(
        self, open_contract, product_config
    ):
        """PRIMARY (load-bearing): the flag fires on the default
        ``check_book_fill=True`` level-wipe path the real replays run."""
        book = _make_book(open_contract, product_config)
        state, _k = self._fill_with_shared_worst_level(book)
        assert _ask_resting_count(state) == book._market_config.book_depth

        # One more resting add forces the worst-price level-wipe.
        cheaper = _make_resting_sell(
            order_id=10_000, price=49.50, timestamp=2000,
        )
        state_after, _ = book.submit_order(state, cheaper)

        assert bool(state_after.book_full) is True, (
            "the predicate must be wired to the check_book_fill=True "
            "(default / level-wipe) eviction path real replays exercise, "
            "not only the check_book_fill=False fallback."
        )

    def test_level_wipe_drops_more_than_one_order(
        self, open_contract, product_config
    ):
        """SECONDARY (C1 characterization, rides free on the same fixture):
        the eviction removes the WHOLE worst-price level (count drop > 1),
        not a single order. This documents untouched jaxob C1 behaviour;
        the developer should not re-derive level-wipe internals."""
        book = _make_book(open_contract, product_config)
        state, k_at_worst = self._fill_with_shared_worst_level(book)
        count_before = _ask_resting_count(state)
        assert count_before == book._market_config.book_depth

        cheaper = _make_resting_sell(
            order_id=10_000, price=49.50, timestamp=2000,
        )
        state_after, _ = book.submit_order(state, cheaper)
        count_after = _ask_resting_count(state_after)

        # The whole worst-price level (k_at_worst orders) is wiped, then the
        # one new cheaper order is added: net count drop is (k_at_worst - 1).
        # The load-bearing claim is simply that more than one order left.
        net_drop = count_before - count_after
        assert net_drop >= 1, (
            "a multi-order worst-price level wipe drops more than one "
            "resting order (C1: JaxOrderBookArrays.py:487 sets every row at "
            f"worst_price to -1); k_at_worst={k_at_worst}, net_drop={net_drop}."
        )
        # And the flag must still observe the event.
        assert bool(state_after.book_full) is True


# ---------------------------------------------------------------------------
# (2) no silent liquidity loss below cap (false-positive / off-by-one guard)
# ---------------------------------------------------------------------------

class TestNoSilentLossBelowCap:
    """Card test (2): submit EXACTLY ``book_depth`` non-matching resting
    orders; assert all ``book_depth`` are present AND ``book_full`` is False.

    This is the direct guard against the off-by-one and the rest-guard
    false-positive: a predicate that fires one order early, or fires on a
    matched-away / IOC residual that never rests, would pass test (1) and
    only fail here.
    """

    def test_exactly_book_depth_orders_all_present_and_flag_false(
        self, open_contract, product_config
    ):
        book = _make_book(open_contract, product_config)
        book_depth = book._market_config.book_depth

        orders = [
            _make_resting_sell(order_id=i + 1, price=50.00 + 0.01 * i)
            for i in range(book_depth)
        ]
        state = book.init()
        state = _submit_all(book, state, orders)

        assert _ask_resting_count(state) == book_depth, (
            "exactly book_depth non-matching resting orders must all be "
            "present — none evicted below the cap."
        )
        assert bool(state.book_full) is False, (
            "filling exactly to book_depth (not over) must NOT set "
            "book_full — the predicate must not fire one order early "
            "(off-by-one on the price-column >= 0 count)."
        )

    def test_ioc_residual_into_full_side_does_not_false_positive(
        self, open_contract, product_config
    ):
        """An IOC aggressor whose residual never rests must NOT set the
        flag, even when the resting side is exactly full.

        The rest-guard mask is the single most error-prone line in the
        diff: a matched-away / type-4-IOC residual that never adds must
        not false-positive book_full.
        """
        book = _make_book(open_contract, product_config)
        book_depth = book._market_config.book_depth

        # Fill the ask side to exactly book_depth with resting sells.
        sells = [
            _make_resting_sell(order_id=i + 1, price=50.00 + 0.01 * i)
            for i in range(book_depth)
        ]
        state = book.init()
        state = _submit_all(book, state, sells)
        assert _ask_resting_count(state) == book_depth
        assert bool(state.book_full) is False

        # An IOC BUY that crosses the best ask: it matches, and any residual
        # is cancelled (IOC) — it never rests on the bid side and does not
        # add to the full ask side. book_full must remain False.
        ioc_buy = make_order(
            contract_id=CONTRACT_ID,
            side=Side.BUY,
            price=50.00,            # crosses the cheapest resting ask
            quantity=0.5,
            order_id=10_001,
            agent_id=2,
            timestamp=3000,
            exec_restriction=ExecRestriction.IOC,
        )
        state_after, _ = book.submit_order(state, ioc_buy)

        assert bool(state_after.book_full) is False, (
            "an IOC residual that matches-and-cancels (never rests) must "
            "NOT set book_full — the rest-guard must gate the predicate on "
            "an order actually resting."
        )


# ---------------------------------------------------------------------------
# (3) busy-contract known-limitation lock (XFAIL — OQ-10 residual)
# ---------------------------------------------------------------------------

class TestBusyContractKnownLimitationOQ10:
    """Card test (3), XFAIL form (Part A resolved the fork to intractable).

    Replays a synthetic cid-81-class busy scenario that breaches the
    ``book_depth=100`` cap mid-session. Two assertions:

      (a) the diagnostic flag fires on the breach — this is the part the
          story DELIVERS and that must pass; and
      (b) the residual busy-contract under-fill (silent liquidity loss
          past the cap) is NOT fixed — ``book_depth`` stays at 100 and the
          level-wipe still drops liquidity. This is the documented OQ-10
          known limitation.

    The XFAIL marks (b): the test asserts the *honesty invariant* that the
    flag makes the loss OBSERVABLE but NOT CORRECT. The story does not
    raise ``book_depth`` and does not prevent the level-wipe; that work is
    delegated to OQ-10. ``strict=True`` so that if a future change actually
    closes the gap, this lock turns into an XPASS and forces a re-read.
    """

    def _busy_breach_state(self, book):
        """Submit >book_depth non-matching resting sells; the cap binds and
        liquidity is silently dropped past 100 (the mid-session breach)."""
        book_depth = book._market_config.book_depth
        n = book_depth + 50  # breach the cap mid-session
        orders = [
            _make_resting_sell(order_id=i + 1, price=50.00 + 0.01 * i)
            for i in range(n)
        ]
        state = book.init()
        state = _submit_all(book, state, orders)
        return state, n

    def test_flag_fires_on_busy_cap_breach(self, open_contract, product_config):
        """The DELIVERED half: the flag observes the breach (must PASS)."""
        book = _make_book(open_contract, product_config)
        state, _n = self._busy_breach_state(book)
        assert bool(state.book_full) is True, (
            "a busy session that breaches book_depth=100 mid-session must "
            "set book_full — this is the diagnostic the story delivers."
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "OQ-10 known limitation: book_depth stays at 100 and the "
            "level-wipe still silently drops resting liquidity past the cap. "
            "The book_full flag makes the loss OBSERVABLE, NOT CORRECT "
            "(N5 honesty invariant). Closing the busy-contract under-fill "
            "is reclassified to OQ-10 / replay-fidelity work, not fixed here. "
            "An XPASS here means the gap was closed — re-read the card."
        ),
    )
    def test_busy_contract_underfill_not_fixed_oq10(
        self, open_contract, product_config
    ):
        """The UNFIXED half (XFAIL): resting liquidity past the cap is still
        lost — the side is pinned at book_depth, fewer than submitted."""
        book = _make_book(open_contract, product_config)
        state, n_submitted = self._busy_breach_state(book)

        resting = _ask_resting_count(state)
        # The XFAIL expectation: the busy contract IS still under-filled —
        # the side is pinned at the cap, strictly below what was submitted.
        # We phrase the assertion as the CORRECT behaviour we do NOT yet
        # have, so it fails today (and after Part B) → XFAIL, and would
        # XPASS only if the gap were genuinely closed.
        assert resting == n_submitted, (
            "no resting liquidity should be silently lost (the correctness "
            f"goal). Today resting={resting} pinned at the cap while "
            f"{n_submitted} were submitted — the OQ-10 gap is open."
        )


# ---------------------------------------------------------------------------
# (4) flag observable end-to-end through _build_state_jit
# ---------------------------------------------------------------------------

class TestFlagObservableEndToEnd:
    """Card test (4): a replay that fires the full-book branch surfaces
    ``book_full`` on the returned ``OMIEBookState``.

    Locks the full wiring path: per-call predicate → ``_build_state_jit``
    flag composition → state-pytree leaf the consumer reads. (1)/(1b)/(2)
    can be authored against an intermediate predicate; (4) is the only
    test that pins the predicate → ``_build_state_jit`` → pytree path
    end-to-end through ``replay_orders_through_book``.
    """

    def test_book_full_present_on_initial_state(
        self, open_contract, product_config
    ):
        """Structural lock: the leaf exists on the freshly-initialised
        state and is False before any order is submitted."""
        book = _make_book(open_contract, product_config)
        state = book.init()
        # AttributeError on the unmodified tree; present + False after.
        assert bool(state.book_full) is False, (
            "book_full must exist as a 0-d bool leaf on the initial "
            "OMIEBookState and default to False."
        )

    def test_replay_surfaces_book_full_end_to_end(
        self, open_contract, product_config
    ):
        """A ``replay_orders_through_book`` session that overflows one side
        surfaces ``book_full`` True on the final state.

        Imported lazily so collection survives even if the loader import
        graph shifts; the assertion (not the import) is the contract.
        """
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            replay_orders_through_book,
        )
        from omiecid_simulator.omiecid.encoding import encode_price, encode_qty

        book = _make_book(open_contract, product_config)
        book_depth = book._market_config.book_depth
        pc = product_config

        # Build a (N, 13) int32 loader-schema orders tensor: book_depth + 1
        # non-matching resting sells at strictly ascending distinct prices.
        def _row(order_id, price, ts):
            return [
                CONTRACT_ID,
                int(Side.SELL),
                int(encode_price(price, pc)),
                int(encode_qty(1.0, pc)),
                order_id,
                1,                       # agent_id
                ts,
                int(ExecRestriction.NON),
                0,                       # validity (GFS)
                0,                       # expiry_time
                0,                       # status ACTIVE
                0,                       # peak_qty ticks
                0,                       # ppd ticks
            ]

        rows = [
            _row(i + 1, 50.00 + 0.01 * i, 1000 + i)
            for i in range(book_depth + 1)
        ]
        orders = jnp.array(rows, dtype=jnp.int32)
        hist_trades = jnp.full((1, 6), -1, dtype=jnp.int32)

        result = replay_orders_through_book(orders, book, hist_trades)

        # The final state must carry book_full True. ReplayResult exposes
        # the terminal book state; resolve it by attribute or fall back to
        # asserting the leaf on whatever final-state object is returned.
        final_state = _resolve_final_state(result)
        assert final_state is not None, (
            "could not locate the final OMIEBookState on the ReplayResult; "
            "the developer must surface the terminal book state so the "
            "book_full leaf is observable end-to-end."
        )
        assert bool(final_state.book_full) is True, (
            "a replay that overflows one side must surface book_full=True "
            "through _build_state_jit on the returned final OMIEBookState."
        )


def _resolve_final_state(replay_result):
    """Locate the terminal ``OMIEBookState`` on a ``ReplayResult``.

    Tries the most likely attribute names; returns the first value that
    looks like an ``OMIEBookState`` (has a ``book_full`` or ``lob`` leaf),
    else ``None``. Kept tolerant so the test fails on the *assertion*
    (a clear message) rather than on attribute archaeology.
    """
    for name in ("final_state", "book_state", "state", "final_book_state"):
        val = getattr(replay_result, name, None)
        if val is not None and (
            hasattr(val, "lob") or hasattr(val, "book_full")
        ):
            return val
    # Some result tuples may carry the state positionally; scan fields.
    fields = getattr(replay_result, "_fields", ())
    for name in fields:
        val = getattr(replay_result, name, None)
        if val is not None and (
            hasattr(val, "lob") or hasattr(val, "book_full")
        ):
            return val
    return None


# ---------------------------------------------------------------------------
# (4b) flag distinct on a `truncated`-saturated session
# ---------------------------------------------------------------------------

class TestFlagDistinctFromTruncated:
    """Card test (4b): ``book_full`` is observable INDEPENDENTLY even when
    ``truncated`` is already True from iceberg-overflow.

    This is the test that JUSTIFIES the distinct leaf over reusing
    ``truncated`` (architect Q3). On busy sessions ``truncated`` is already
    sticky-True from the interim iceberg cap (C2-caveat), so ORing the
    full-book event into ``truncated`` would be a hollow, indistinguishable
    signal exactly where observability is the whole deliverable.
    """

    def test_truncated_alone_does_not_set_book_full(
        self, open_contract, product_config
    ):
        """Sanity precondition: an oversized iceberg sets ``truncated`` True
        WITHOUT touching ``book_full`` (the two flags are distinct leaves)."""
        book = _make_book(open_contract, product_config)
        state = book.init()

        # quantity=998.0 stays below qty_max=999.0 so the iceberg registers;
        # ceil(998 / 10) = 100 slices > MAX_SLICES_PER_REGEN (64) → truncated.
        oversized_iceberg = make_order(
            contract_id=CONTRACT_ID,
            side=Side.SELL,
            price=60.00,
            quantity=998.0,
            order_id=1,
            agent_id=1,
            timestamp=1000,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=10.0,
        )
        state, _ = book.submit_order(state, oversized_iceberg)

        assert bool(state.truncated) is True, (
            "an iceberg with ceil(total/peak) > MAX_SLICES_PER_REGEN must "
            "set truncated (precondition for the distinctness test)."
        )
        assert bool(state.book_full) is False, (
            "iceberg truncation must NOT set book_full — the two are "
            "distinct leaves (no aliasing)."
        )

    def test_book_full_observable_while_truncated_already_true(
        self, open_contract, product_config
    ):
        """The load-bearing distinctness lock: with ``truncated`` already
        sticky-True from iceberg overflow, a subsequent full-side eviction
        sets ``book_full`` True — observable INDEPENDENTLY of ``truncated``.

        The book_depth here is reduced so the one-sided fill is cheap; the
        iceberg-overflow path that pre-saturates ``truncated`` is unchanged.
        """
        # Small book_depth keeps the one-sided fill microsecond-cheap while
        # still exercising the full-side eviction → book_full path. The
        # distinctness behaviour does not depend on the production value.
        book = _make_book(open_contract, product_config, book_depth=8)
        book_depth = book._market_config.book_depth
        state = book.init()

        # 1) Pre-saturate truncated via an oversized iceberg on the BID side
        #    (so it does not consume ask-side resting slots). quantity=998.0
        #    stays below qty_max=999.0 so the iceberg registers;
        #    ceil(998 / 10) = 100 slices > MAX_SLICES_PER_REGEN (64) → truncated.
        oversized_iceberg = make_order(
            contract_id=CONTRACT_ID,
            side=Side.BUY,
            price=10.00,            # far below ask prices → does not cross
            quantity=998.0,
            order_id=1,
            agent_id=1,
            timestamp=1000,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=10.0,
        )
        state, _ = book.submit_order(state, oversized_iceberg)
        assert bool(state.truncated) is True, (
            "precondition: truncated must already be sticky-True from the "
            "iceberg overflow before the full-side eviction."
        )

        # 2) Now overflow the ASK side with book_depth + 1 resting sells.
        sells = [
            _make_resting_sell(
                order_id=100 + i, price=50.00 + 0.01 * i, timestamp=2000 + i,
            )
            for i in range(book_depth + 1)
        ]
        state = _submit_all(book, state, sells)

        # Both flags are independently observable: truncated stayed True
        # (sticky) AND book_full is now True — distinct, not aliased.
        assert bool(state.truncated) is True, (
            "truncated must remain sticky-True across the full-side eviction."
        )
        assert bool(state.book_full) is True, (
            "book_full must be observable INDEPENDENTLY even when truncated "
            "is already True from iceberg overflow — this is the entire "
            "justification for the distinct book_full leaf (architect Q3)."
        )


# ---------------------------------------------------------------------------
# (5) OPTIONAL real-data slice (skip-if-no-data, non-gating)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not FINDING_ORDERS_FILE.exists(),
    reason=(
        "OPTIONAL real-data slice: data/rawOMIE orders file for the finding's "
        "cid-81 / 2025-06-02 window is absent. Skipped cleanly per the "
        "no-assume-data invariant (non-gating)."
    ),
)
@pytest.mark.slow
class TestRealDataSliceOptional:
    """Card test (5), OPTIONAL: replay a tiny real-data slice and assert the
    flag surfaces.

    slow: replays a real-data slice through the book (loader + scan), which
    exceeds the < 1 s fast-suite budget; gated to skip cleanly when
    ``data/rawOMIE`` is absent. Non-gating per the Q6 ruling.
    """

    def test_real_busy_slice_surfaces_book_full(self, open_contract, product_config):
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            OMIEDataConfig,
            load_omie_orders,
            replay_orders_through_book,
        )

        cfg = OMIEDataConfig(
            data_path=str(RAW_OMIE_DIR),
            delivery_date=FINDING_OMIE_DATE,
            contract_id=FINDING_CONTRACT_ID,
        )
        orders = load_omie_orders(cfg)
        # Use a slice large enough to breach the 100/side cap (the finding's
        # first breach is at order #123). Keep it bounded so the slow test
        # stays a slice, not a full session.
        slice_n = min(int(orders.shape[0]), 200)
        orders = orders[:slice_n]

        contract = Contract(
            contract_id=FINDING_CONTRACT_ID,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.OPEN,
        )
        book = ContractBook(contract, product_config, OMIEMarketConfig())
        hist_trades = jnp.full((1, 6), -1, dtype=jnp.int32)

        result = replay_orders_through_book(orders, book, hist_trades)
        final_state = _resolve_final_state(result)
        assert final_state is not None
        assert bool(final_state.book_full) is True, (
            "a real busy slice that breaches book_depth=100 mid-session must "
            "surface book_full=True (the diagnostic the story delivers)."
        )
