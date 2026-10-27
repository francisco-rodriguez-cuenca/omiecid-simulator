"""Audit tests for the P3-PREP-REPLAY-1 Part B ``book_full`` diagnostic flag.

Independent post-implementation verification pass. These tests are authored
from the v5.46 card (``docs/omiecid/agile-plan.md`` ``#### Story P3-PREP-REPLAY-1``),
the committed architect rulings (Q1-Q7) and Part-A spike note in
``docs/omiecid/sprint-status.md``, and the FINISHED implementation's *public
interface* (``omiecid_simulator/omiecid/contract_book.py`` ``OMIEBookState.book_full``
/ ``_resting_side_full``; ``omiecid_simulator/jaxomiecid/omie_data_loader.py``
``ReplayResult.final_state``).

They probe domain-correctness angles a first-pass suite might under-cover,
DERIVED INDEPENDENTLY from the same spec (this file does not read the locked
``test_book_full_flag.py`` and does not mirror its fixtures):

  - Boundary precision: independent re-derivation of the off-by-one
    (exactly ``book_depth`` -> False; ``book_depth + 1`` -> True) on a
    distinct fixture (descending non-crossing asks).
  - Stickiness across steps: once True the flag STAYS True through later
    submits that do not fill the side (sticky-OR, like ``truncated``), and
    across the ``expire_at_gct`` session boundary; and conversely the flag
    is False before any breach.
  - Per-side correctness: a full ASK side sets the flag while a
    simultaneously non-full BID side neither sets nor clears it; the
    predicate keys off the side the order ACTUALLY rests on. The symmetric
    bid-only fill is also checked.
  - No false-positive on non-resting orders: a FOK that kills (no rest)
    and a cancel (type-2 removal) do NOT set ``book_full``. (The original
    suite is documented as covering the IOC angle; this adds FOK-kill and
    cancel.)
  - Distinctness under combined pressure (NO aliasing either direction):
    ``truncated`` True (iceberg overflow) must not imply ``book_full``, and
    ``book_full`` True (full-side eviction) must not imply ``truncated``;
    a session that triggers BOTH carries both leaves True, each readable
    independently.
  - JIT / vmap shape stability: ``book_full`` is a 0-d ``jnp.bool_`` that
    survives ``jax.jit`` (the public ``_submit_order`` lift) and a small
    ``jax.vmap`` over the leaf (relevant to the forthcoming RL vmap
    topology).
  - The honesty invariant as an executable lock: the busy-contract
    under-fill is NOT fixed by the flag (``book_depth`` stays 100; the
    full-side eviction still silently drops resting liquidity past the
    cap) -- the OQ-10 known-limitation, locked as a strict ``xfail``.

Device policy: CPU-default (``CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu``).
The fixtures are small synthetic per-contract books, but every
``ContractBook.submit_order`` call drives the production ``_submit_order``
``jax.jit`` lift, so the FIRST submit in each test pays a one-time XLA
compile (~1.5 s wall-clock observed) that pushes the per-test time past the
< 1 s fast-suite budget. The whole module therefore carries
``pytestmark = pytest.mark.jit`` (compile-heavy; deselectable with
``-m "not jit"``) — the same idiom the OMIE submit-path test suites already
use. There is no real-data / loader integration here, so nothing is marked
``slow``.
"""

import jax
import jax.numpy as jnp
import pytest

# Module-level jit marker: each test's first ``submit_order`` (or the direct
# ``_submit_order`` lift) triggers a one-time XLA compile of the jitted
# OMIE submit path, exceeding the < 1 s fast-suite budget. Marking the module
# lets the fast feedback loop deselect the compile cost via ``-m "not jit"``.
pytestmark = pytest.mark.jit

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    _submit_order,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order


# ---------------------------------------------------------------------------
# Fixtures — small synthetic per-contract book.
#
# A deliberately SMALL ``book_depth`` (8) keeps every contract/unit test in
# the microsecond regime and makes the boundary fixtures readable. The cap is
# a simultaneous-resting count (book-depth-cap-finding.md), so a small value
# exercises the identical eviction path the production default (100) does.
# ---------------------------------------------------------------------------

_BOOK_DEPTH = 8
_CONTRACT_ID = 37
_P_COL = cst.OrderSideFeat.P.value


@pytest.fixture
def product_config() -> OMIEProductConfig:
    return OMIEProductConfig()


@pytest.fixture
def market_config() -> OMIEMarketConfig:
    # nTrades left at the default 100; book_depth shrunk to 8 for cheap
    # boundary fixtures. nTrades=100 >> any fill count these fixtures
    # produce, so the trades buffer never confounds the resting-book cap.
    return OMIEMarketConfig(book_depth=_BOOK_DEPTH, nTrades=100)


@pytest.fixture
def open_contract() -> Contract:
    """An OPEN mid-day contract (submit_order's OPEN guard must pass)."""
    return Contract(
        contract_id=_CONTRACT_ID,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


@pytest.fixture
def book(open_contract, product_config, market_config) -> ContractBook:
    return ContractBook(open_contract, product_config, market_config)


# ---------------------------------------------------------------------------
# Helpers — non-crossing resting orders. Asks far above bids so nothing ever
# matches; every submit is a pure resting add (the path the flag guards).
# ---------------------------------------------------------------------------

def _resting_ask(order_id: int, price: float, ts: int):
    return make_order(
        contract_id=_CONTRACT_ID, side=Side.SELL, price=price, quantity=1.0,
        order_id=order_id, agent_id=1, timestamp=ts,
    )


def _resting_bid(order_id: int, price: float, ts: int):
    return make_order(
        contract_id=_CONTRACT_ID, side=Side.BUY, price=price, quantity=1.0,
        order_id=order_id, agent_id=1, timestamp=ts,
    )


def _count_resting(side_arr: jnp.ndarray) -> int:
    """Number of live rows on a jaxob side array (price column >= 0)."""
    return int(jnp.sum(side_arr[:, _P_COL] >= 0))


def _fill_side_to_cap(book: ContractBook, state, *, side: int, base_price: float,
                      start_oid: int = 1, start_ts: int = 10):
    """Submit exactly ``book_depth`` non-crossing resting orders on ``side``.

    Asks ascend above ``base_price``; bids descend below it; both kept clear
    of the opposite side so nothing matches. Returns ``(state, next_oid)``.
    """
    oid = start_oid
    for i in range(book._market_config.book_depth):
        if side == int(Side.SELL):
            order = _resting_ask(oid, base_price + i, start_ts + i)
        else:
            order = _resting_bid(oid, base_price - 0.5 * i, start_ts + i)
        state, _ = book.submit_order(state, order)
        oid += 1
    return state, oid


# ===========================================================================
# Angle 1 — Boundary precision (independent off-by-one re-derivation)
# ===========================================================================

class TestBoundaryPrecision:
    """Exactly ``book_depth`` resting orders -> False; one more -> True.

    Independent fixture from any first-pass test: this builds the cap with
    DESCENDING ask prices (worst price last) so the level-wipe target is
    unambiguous, and asserts on the live-row count directly rather than only
    the flag.
    """

    def test_exactly_book_depth_resting_asks_flag_false(self, book):
        state = book.init()
        # Descending asks: 200, 199, ... still all > 0 ticks, none crossing.
        oid = 1
        for i in range(_BOOK_DEPTH):
            state, _ = book.submit_order(state, _resting_ask(oid, 200.0 - i, 10 + i))
            oid += 1
        assert _count_resting(state.lob.asks) == _BOOK_DEPTH
        assert bool(state.book_full) is False, (
            "exactly book_depth resting asks must NOT trip the full-side "
            "predicate (off-by-one: the side is full only AFTER book_depth, "
            "not AT it)"
        )

    def test_book_depth_plus_one_resting_ask_flag_true(self, book):
        state = book.init()
        oid = 1
        for i in range(_BOOK_DEPTH):
            state, _ = book.submit_order(state, _resting_ask(oid, 200.0 - i, 10 + i))
            oid += 1
        assert bool(state.book_full) is False  # boundary precondition
        # The (book_depth + 1)-th resting add fires the eviction predicate.
        state, _ = book.submit_order(state, _resting_ask(oid, 150.0, 500))
        assert bool(state.book_full) is True, (
            "the resting add that arrives at a full side must set book_full"
        )
        # The cap is a simultaneous-resting slot count: the side stays pinned
        # at book_depth (the eviction dropped, did not grow).
        assert _count_resting(state.lob.asks) == _BOOK_DEPTH


# ===========================================================================
# Angle 2 — Stickiness across steps (sticky-OR, like truncated)
# ===========================================================================

class TestStickiness:
    def test_flag_false_on_fresh_init(self, book):
        state = book.init()
        assert hasattr(state, "book_full"), "OMIEBookState must carry book_full"
        assert bool(state.book_full) is False

    def test_flag_false_while_below_cap_across_many_submits(self, book):
        """No spurious True before any breach, over several resting adds."""
        state = book.init()
        oid = 1
        for i in range(_BOOK_DEPTH - 1):  # stay strictly below the cap
            state, _ = book.submit_order(state, _resting_ask(oid, 200.0 + i, 10 + i))
            oid += 1
            assert bool(state.book_full) is False, (
                f"book_full went True after {i + 1} resting asks (< book_depth)"
            )

    def test_flag_stays_true_through_non_filling_submits(self, book):
        """Once True, later submits that do NOT fill the side keep it True."""
        state = book.init()
        # Breach the ASK side.
        oid = 1
        for i in range(_BOOK_DEPTH + 1):
            state, _ = book.submit_order(state, _resting_ask(oid, 200.0 + i, 10 + i))
            oid += 1
        assert bool(state.book_full) is True

        # Subsequent BID adds (a different, non-full side) must not clear it.
        for i in range(3):
            state, _ = book.submit_order(state, _resting_bid(oid, 50.0 - i, 600 + i))
            oid += 1
            assert bool(state.book_full) is True, (
                "book_full must be sticky-OR: a later non-filling submit "
                "cannot reset a once-True flag"
            )

    def test_flag_preserved_across_gct_expiry(self, book):
        """book_full is a session-level diagnostic — survives expire_at_gct."""
        state = book.init()
        oid = 1
        for i in range(_BOOK_DEPTH + 1):
            state, _ = book.submit_order(state, _resting_ask(oid, 200.0 + i, 10 + i))
            oid += 1
        assert bool(state.book_full) is True
        post_gct = book.expire_at_gct(state)
        # GCT empties the book but the sticky session diagnostic persists,
        # mirroring the truncated carry.
        assert bool(post_gct.book_full) is True
        assert _count_resting(post_gct.lob.asks) == 0


# ===========================================================================
# Angle 3 — Per-side correctness (predicate keys off the RESTING side)
# ===========================================================================

class TestPerSideCorrectness:
    def test_full_ask_side_sets_flag_empty_bid_side_does_not_clear(self, book):
        """A full ASK sets the flag; the simultaneously-empty BID is irrelevant."""
        state = book.init()
        oid = 1
        for i in range(_BOOK_DEPTH + 1):
            state, _ = book.submit_order(state, _resting_ask(oid, 200.0 + i, 10 + i))
            oid += 1
        assert bool(state.book_full) is True
        # BID side is entirely empty (non-full) — the flag is about the ASK
        # side the orders rested on, not the BID side.
        assert _count_resting(state.lob.bids) == 0

    def test_full_bid_side_independently_sets_flag(self, book):
        """Symmetric: filling only the BID side past the cap also sets the flag."""
        state = book.init()
        oid = 1
        # book_depth bids -> still False.
        state, oid = _fill_side_to_cap(
            book, state, side=int(Side.BUY), base_price=50.0, start_oid=oid
        )
        assert bool(state.book_full) is False
        # one more bid -> eviction predicate fires on the bids side.
        state, _ = book.submit_order(state, _resting_bid(oid, 60.0, 500))
        assert bool(state.book_full) is True
        assert _count_resting(state.lob.bids) == _BOOK_DEPTH

    def test_full_one_side_does_not_flag_a_resting_add_on_the_other_side(self, book):
        """A resting add on the NON-full opposite side must not itself flip the
        flag (the flag is sticky from the prior breach, but the new add's
        per-call contribution must be False — the predicate selects the
        resting side, not the matched-against side)."""
        # Fill ASK side to EXACTLY the cap (flag still False).
        state = book.init()
        oid = 1
        for i in range(_BOOK_DEPTH):
            state, _ = book.submit_order(state, _resting_ask(oid, 200.0 + i, 10 + i))
            oid += 1
        assert bool(state.book_full) is False  # ask side full, bid empty, no breach yet
        # Now add a resting BID at a non-crossing price. The BID side is empty
        # (non-full), so the per-call predicate on the bid side is False and
        # the flag must remain False even though the ASK side is at the cap.
        state, _ = book.submit_order(state, _resting_bid(oid, 50.0, 500))
        assert bool(state.book_full) is False, (
            "a resting add on the empty BID side must not set book_full just "
            "because the ASK side is at the cap — the predicate keys off the "
            "side the order rests on"
        )


# ===========================================================================
# Angle 4 — No false-positive on non-resting orders (FOK-kill + cancel)
# ===========================================================================

class TestNoFalsePositiveOnNonRestingOrders:
    def test_fok_kill_on_empty_book_does_not_set_flag(self, book):
        """A FOK with no matchable liquidity is killed (accepted-no-trade,
        no rest) and must NOT set book_full."""
        state = book.init()
        fok = make_order(
            contract_id=_CONTRACT_ID, side=Side.BUY, price=50.0, quantity=5.0,
            order_id=1, agent_id=1, timestamp=10,
            exec_restriction=ExecRestriction.FOK,
        )
        new_state, result = book.submit_order(state, fok)
        assert bool(result.accepted) is True
        assert bool(result.traded) is False
        assert bool(new_state.book_full) is False, (
            "a FOK that finds no liquidity never rests, so it cannot fire the "
            "full-side eviction predicate"
        )

    def test_fok_kill_on_full_book_does_not_set_flag(self, book):
        """A FOK killed for insufficient liquidity against a FULL opposite
        side still rests nothing, so it must not set book_full.

        Build a full ASK side WITHOUT breaching it (exactly book_depth asks
        -> flag False), then a FOK BUY for more volume than is available at a
        crossing price. The FOK pre-check fails (accept-no-trade), nothing
        rests, and the flag stays False."""
        state = book.init()
        oid = 1
        # Exactly book_depth asks, all at price 100..107, 1 MW each => 8 MW
        # total visible. Flag still False (we did not breach the cap).
        for i in range(_BOOK_DEPTH):
            state, _ = book.submit_order(state, _resting_ask(oid, 100.0 + i, 10 + i))
            oid += 1
        assert bool(state.book_full) is False
        # FOK BUY for 50 MW @ 130 (crosses all asks but only 8 MW available
        # < 50 MW required) => killed, accept-no-trade, no rest.
        fok = make_order(
            contract_id=_CONTRACT_ID, side=Side.BUY, price=130.0, quantity=50.0,
            order_id=oid, agent_id=1, timestamp=500,
            exec_restriction=ExecRestriction.FOK,
        )
        new_state, result = book.submit_order(state, fok)
        assert bool(result.traded) is False, "FOK should be killed (insufficient liquidity)"
        assert bool(new_state.book_full) is False, (
            "a killed FOK rests nothing and must not set book_full"
        )

    def test_cancel_is_a_removal_and_does_not_set_flag(self, book):
        """A type-2 cancel removes a resting order — never a resting add — so
        it must not set book_full."""
        state = book.init()
        bid = _resting_bid(7, 50.0, 10)
        state, _ = book.submit_order(state, bid)
        assert bool(state.book_full) is False
        state = book.cancel_order(state, order_id=7, side=int(Side.BUY))
        assert bool(state.book_full) is False, (
            "cancel is a removal, not a resting add; it must carry the prior "
            "(False) flag forward unchanged"
        )

    def test_cancel_preserves_a_prior_true_flag(self, book):
        """A removal must not CLEAR an already-True sticky flag either."""
        state = book.init()
        oid = 1
        for i in range(_BOOK_DEPTH + 1):
            state, _ = book.submit_order(state, _resting_ask(oid, 200.0 + i, 10 + i))
            oid += 1
        assert bool(state.book_full) is True
        # Cancel the first resting ask (order_id 1). The flag is sticky.
        state = book.cancel_order(state, order_id=1, side=int(Side.SELL))
        assert bool(state.book_full) is True, (
            "cancel must propagate the sticky book_full forward, neither "
            "setting nor clearing it"
        )


# ===========================================================================
# Angle 5 — Distinctness under combined pressure (NO aliasing, either way)
# ===========================================================================

class TestDistinctFromTruncated:
    """``book_full`` and ``truncated`` are independent 0-d leaves.

    The whole rationale for the distinct leaf (architect Q3) is that
    ``truncated`` is already iceberg-saturated on busy sessions; reusing it
    would make the full-side signal hollow. These tests lock the two-way
    no-aliasing property.
    """

    def _oversized_iceberg(self, side: int, price: float, oid: int, ts: int):
        # quantity=998.0 < qty_max=999.0 registers; encodes to 9980 ticks;
        # ceil(9980/100)=100 > MAX_SLICES_PER_REGEN=64 -> truncated=True.
        # (Construction validated by the committed Q7 ruling probe.)
        return make_order(
            contract_id=_CONTRACT_ID, side=side, price=price, quantity=998.0,
            order_id=oid, agent_id=1, timestamp=ts,
            exec_restriction=ExecRestriction.ICE, peak_qty=10.0,
        )

    def test_truncated_true_does_not_imply_book_full(self, book):
        """Iceberg overflow sets truncated WITHOUT setting book_full."""
        state = book.init()
        ice = self._oversized_iceberg(int(Side.BUY), 10.0, oid=1, ts=10)
        state, _ = book.submit_order(state, ice)
        assert bool(state.truncated) is True, "oversized iceberg must set truncated"
        assert bool(state.book_full) is False, (
            "truncated True (iceberg overflow) must NOT alias to book_full — "
            "no full-side eviction has occurred"
        )

    def test_book_full_true_does_not_imply_truncated(self, book):
        """Full-side eviction sets book_full WITHOUT setting truncated."""
        state = book.init()
        oid = 1
        for i in range(_BOOK_DEPTH + 1):
            state, _ = book.submit_order(state, _resting_ask(oid, 200.0 + i, 10 + i))
            oid += 1
        assert bool(state.book_full) is True
        assert bool(state.truncated) is False, (
            "book_full True (full-side eviction) must NOT alias to truncated — "
            "no iceberg slice-overflow has occurred on this fixture"
        )

    def test_both_leaves_true_and_independently_readable(self, book):
        """A session that triggers BOTH carries both leaves True, each read
        independently (no aliasing in either direction)."""
        state = book.init()
        # 1) oversized iceberg on the BID side -> truncated True, book_full False.
        ice = self._oversized_iceberg(int(Side.BUY), 10.0, oid=1, ts=10)
        state, _ = book.submit_order(state, ice)
        assert bool(state.truncated) is True
        assert bool(state.book_full) is False
        # 2) breach the ASK side -> book_full True; truncated stays sticky-True.
        oid = 2
        for i in range(_BOOK_DEPTH + 1):
            state, _ = book.submit_order(state, _resting_ask(oid, 100.0 + i, 20 + i))
            oid += 1
        assert bool(state.truncated) is True
        assert bool(state.book_full) is True
        # The two leaves are distinct objects/values, not the same array.
        assert state.truncated is not state.book_full


# ===========================================================================
# Angle 6 — JIT / vmap shape stability
# ===========================================================================

class TestJitVmapShapeStability:
    @pytest.mark.jit
    def test_book_full_is_zero_d_bool_after_jit_submit(self, book, product_config,
                                                       market_config):
        """The jitted ``_submit_order`` lift returns a state whose book_full
        leaf is a 0-d ``jnp.bool_`` (the new leaf does not break tracing)."""
        state = book.init()
        order = _resting_ask(1, 100.0, 10)
        new_state, _result = _submit_order(state, order, product_config, market_config)
        assert isinstance(new_state, OMIEBookState)
        leaf = new_state.book_full
        assert jnp.asarray(leaf).shape == (), "book_full must be a 0-d scalar"
        assert jnp.asarray(leaf).dtype == jnp.bool_, "book_full must be jnp.bool_"

    @pytest.mark.jit
    def test_book_full_set_under_jit_on_full_side(self, book, product_config,
                                                  market_config):
        """The flag is correctly set True through the jitted lift on a breach,
        confirming the predicate threads through jit (not only the Python
        trampoline)."""
        state = book.init()
        oid = 1
        for i in range(_BOOK_DEPTH):
            state, _ = _submit_order(
                state, _resting_ask(oid, 200.0 + i, 10 + i),
                product_config, market_config,
            )
            oid += 1
        assert bool(state.book_full) is False
        state, _ = _submit_order(
            state, _resting_ask(oid, 150.0, 500), product_config, market_config,
        )
        assert bool(state.book_full) is True

    def test_book_full_leaf_survives_vmap(self):
        """A 0-d bool leaf batches cleanly under jax.vmap — relevant to the
        forthcoming RL vmap topology where every batched env shares a traced
        shape. Lightweight: no env rollout, just the leaf transform."""
        batch = jnp.array([False, True, False, True])

        def sticky_or(prior, call):
            # Mirrors the production sticky-OR composition on the leaf.
            return jnp.logical_or(jnp.asarray(prior, jnp.bool_),
                                  jnp.asarray(call, jnp.bool_))

        calls = jnp.array([False, False, True, False])
        out = jax.vmap(sticky_or)(batch, calls)
        assert out.shape == (4,)
        assert out.dtype == jnp.bool_
        # Sticky-OR semantics preserved per element.
        assert out.tolist() == [False, True, True, True]


# ===========================================================================
# Angle 7 — The honesty invariant as an executable lock
# ===========================================================================

class TestHonestyInvariantNotFixed:
    """The flag makes the loss OBSERVABLE, NOT CORRECT (N5 / Q1 / Q2).

    The busy-contract under-fill is reclassified to OQ-10 and is NOT closed
    by this card: ``book_depth`` stays at its configured value, the side is
    pinned at the cap, and the surplus resting liquidity is silently dropped.
    """

    def test_full_side_eviction_silently_drops_resting_liquidity(self, book):
        """Submitting far more than ``book_depth`` non-crossing resting orders
        leaves exactly ``book_depth`` resting (the surplus is silently lost),
        even though book_full fires. This is the OBSERVABLE-not-CORRECT
        property: the flag rises but the liquidity is still gone."""
        state = book.init()
        oid = 1
        n_submitted = _BOOK_DEPTH + 50
        for i in range(n_submitted):
            state, _ = book.submit_order(state, _resting_ask(oid, 100.0 + i, 10 + i))
            oid += 1
        assert bool(state.book_full) is True, "the breach must be observable"
        resting = _count_resting(state.lob.asks)
        assert resting == _BOOK_DEPTH, (
            f"resting count pinned at the cap ({_BOOK_DEPTH}); the "
            f"{n_submitted - _BOOK_DEPTH} surplus orders were silently dropped"
        )
        # book_depth is NOT grown to accommodate the breach (shape stays fixed).
        assert state.lob.asks.shape == (_BOOK_DEPTH, 6)
        assert book._market_config.book_depth == _BOOK_DEPTH

    @pytest.mark.xfail(
        strict=True,
        reason="OQ-10 known limitation: the book_full flag is a DIAGNOSTIC. "
        "Busy-contract replay is NOT made correct by it — the full-side "
        "eviction still drops liquidity past book_depth. This xfail locks "
        "the honesty invariant; an XPASS means someone made busy-contract "
        "replay lossless and this card's framing must be re-read.",
    )
    def test_busy_replay_underfill_is_fixed_oq10_xfail(self, book):
        """ASSERTS (incorrectly, hence xfail) that no resting liquidity is
        lost on a busy breach. This is the OQ-10 gap the flag does NOT close;
        the assertion must FAIL so the xfail passes."""
        state = book.init()
        oid = 1
        n_submitted = _BOOK_DEPTH + 25
        for i in range(n_submitted):
            state, _ = book.submit_order(state, _resting_ask(oid, 100.0 + i, 10 + i))
            oid += 1
        # If the busy-contract gap were CORRECT (not merely observable), every
        # submitted non-crossing order would still be resting. It is not.
        assert _count_resting(state.lob.asks) == n_submitted
