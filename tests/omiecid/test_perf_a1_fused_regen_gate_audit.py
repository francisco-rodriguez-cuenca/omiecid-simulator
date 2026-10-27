"""Independent audit tests for Story P2-PERF-8 — A1 fused regen-gate /
cascade-quiescence (decision iii).

Mode: AUDIT. These tests are authored independently of the original
locked test file (`test_perf_a1_fused_regen_gate.py`, NOT read by this
author) and probe the PUBLIC behavioural contract of the finished
implementation against the spec directly:

  * docs/omiecid/market-spec.md §4.2.1 Rule 4 — the regen-immediate-rematch
    cascade MUST resolve to a fixed point WITHIN the originating
    `submit_order` call (no slot left peak-consumed; the book is never
    left crossed between aggressor arrivals).
  * agile-plan.md §"Story P2-PERF-8" Test-first contract (b) — after
    EVERY `submit_order` call, no active iceberg slot is peak-consumed:
    `book._ob.get_order(state.lob, lookup_side, oid)[2] > 0` for all
    `i in [0, n_active)` (a missing OID returns the all-`-99`
    NEGATIVE_RETURN_ID sentinel row, so `[2] <= 0`).

The three probe families (orchestrator brief):
  1. Cascade-quiescence post-condition — adversarial multi-iceberg /
     deep-book / BID+ASK / many-hop fixtures, all DIFFERENT from the
     synthetic + truncated-cid-81 fixtures the original author used.
  2. Within-call resolution correctness — volume conservation, no
     phantom fills, idempotence, and the `truncated`-flag cap-wiring.
  3. Gate equivalence — regen on a quiescent no-fill state is a true
     no-op (the invariant that makes `lax.cond(any_fills, regen,
     identity)` observably equivalent to unconditional regen).

ROUTING NOTE (P2-PERF-5 guard compliance): every probe routes through
the PUBLIC ``ContractBook.submit_order`` surface (and, for the gate-
equivalence no-op probe, the PUBLIC ``IcebergManager.regenerate_slices``
method). This audit file does NOT reference the internal scan-lift
symbol guarded by the P2-PERF-5 caller-grep, so it requires no
file-pinned allowlist entry in that guard.

FAST TESTS ONLY — synthetic fixtures, microsecond matching, no
`@pytest.mark.slow` (user election, sprint-status §"P2-PERF-8 — Scope
narrowing"). `pytestmark = pytest.mark.jit` is set at module scope
because `submit_order` invokes the module-level jitted lift (conforms
to `test_jit_marker_lint::test_omie_suite_jit_marker_discipline`).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.encoding import encode_side_lookup
from omiecid_simulator.omiecid.order_types import make_order, Side, ExecRestriction

pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Shared probe helpers
# ---------------------------------------------------------------------------

def _quiescence_violations(book: ContractBook, state):
    """Return the list of cascade-quiescence violations in ``state``.

    A violation is an active iceberg slot (index in ``[0, n_active)``)
    whose registered OID is NOT present-and-resting in the book: the
    raw jaxob lookup row's column ``[2]`` (the order_id column) is
    ``<= 0``. A missing OID returns the all-``-99`` NEGATIVE_RETURN_ID
    sentinel row, so ``[2] == -99 <= 0`` flags it. A present resting
    order has ``[2] == oid > 0``.

    Returns a list of ``(slot_index, oid, side, lookup_row_col2)`` for
    every offending slot — empty list means quiescent. This is exactly
    the post-condition Rule 4 / agile-plan contract (b) mandate.
    """
    n_active = int(state.n_active)
    meta = state.iceberg_meta
    violations = []
    for i in range(n_active):
        oid = int(meta[i, 0])
        side = int(meta[i, 4])
        lookup_side = encode_side_lookup(side)
        row = book._ob.get_order(state.lob, lookup_side, oid)
        col2 = int(row[2])
        if col2 <= 0:
            violations.append((i, oid, side, col2))
    return violations


def _state_tuple(state):
    """A reduced observable state fingerprint for idempotence /
    no-op comparisons.

    Not the full 8-tuple byte-identity of the locked test (that is the
    original author's job); this is an INDEPENDENT coarse fingerprint
    over the same observable surfaces — sums over the resting book,
    trade tape, and iceberg meta plus the scalar carries. Two replays
    that agree on this tuple agree on all aggregate observables.
    """
    return (
        bool(state.truncated),
        int(state.n_active),
        int(state.ts_counter),
        int(jnp.sum(jnp.abs(state.lob.bids))),
        int(jnp.sum(jnp.abs(state.lob.asks))),
        int(jnp.sum(jnp.abs(state.lob.trades))),
        int(jnp.sum(jnp.abs(state.iceberg_meta))),
    )


def _fresh(book: ContractBook):
    return book.init()


# ---------------------------------------------------------------------------
# Probe family 1 — cascade-quiescence post-condition (Rule 4 / contract b)
# ---------------------------------------------------------------------------

class TestCascadeQuiescencePostCondition:
    """After every ``submit_order`` no active slot is peak-consumed.

    Fixtures are deliberately distinct from the original author's
    (multi-iceberg same-side cascades, deep-book crossing, BID + ASK
    symmetry, many-hop drain).
    """

    def test_single_iceberg_drained_by_large_non_leaves_no_peak_consumed_slot(
        self, book,
    ):
        """A NON order that drains a whole iceberg (many regen hops) must
        leave the slot either cleaned up or visibly resting — never
        peak-consumed.

        SELL iceberg peak=2 total=10 at 50.00; BUY NON 20 MW crosses and
        consumes all five slices within one call (Rule 2 multi-slice).
        Post-condition: the iceberg is fully drained (slot removed), so
        n_active drops and there is nothing left peak-consumed.
        """
        state = _fresh(book)
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 10.0, 201, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
        )
        state, result = book.submit_order(
            state,
            make_order(37, Side.BUY, 50.00, 20.0, 202, 2, 101,
                       ExecRestriction.NON),
        )
        assert _quiescence_violations(book, state) == []
        # Fully drained -> slot removed.
        assert int(state.n_active) == 0
        assert float(result.trade_qty) == pytest.approx(10.0)

    def test_bid_side_iceberg_cascade_quiescent(self, book):
        """BID-side symmetry: resting BUY iceberg drained by aggressive
        SELL must also leave no peak-consumed slot.

        Probes that the cascade fix is side-agnostic (the original
        author may have anchored only on the ASK-side cid-81 slice).
        """
        state = _fresh(book)
        state, _ = book.submit_order(
            state,
            make_order(37, Side.BUY, 50.00, 10.0, 301, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
        )
        state, result = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 20.0, 302, 2, 101,
                       ExecRestriction.NON),
        )
        assert _quiescence_violations(book, state) == []
        assert int(state.n_active) == 0
        assert float(result.trade_qty) == pytest.approx(10.0)

    def test_two_same_side_icebergs_cross_consumed_remain_quiescent(
        self, book,
    ):
        """Two SELL icebergs at the same price; a crossing BUY iceberg
        consumes peaks and triggers regen on BOTH resting sell slots,
        each of which must stay quiescent (present-and-resting) after the
        cascade.

        Multi-iceberg chain on one side: a fixture class the original
        single-iceberg cid-81-slice contract is unlikely to cover. The
        crossing BUY iceberg (total=6) fully self-consumes against the
        two regenerating SELL icebergs and is cleaned up; the two SELL
        slots retain hidden volume and remain live and quiescent.
        """
        state = _fresh(book)
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 10.0, 101, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
        )
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 8.0, 102, 1, 101,
                       ExecRestriction.ICE, peak_qty=2.0),
        )
        # BUY iceberg (peak=2, total=6) crosses both resting SELL
        # icebergs, regenerating their slices within the call.
        state, _ = book.submit_order(
            state,
            make_order(37, Side.BUY, 50.00, 6.0, 103, 2, 102,
                       ExecRestriction.ICE, peak_qty=2.0),
        )
        # The contract being probed: NO active slot is peak-consumed.
        assert _quiescence_violations(book, state) == []
        # The two SELL icebergs retain hidden volume and remain live; the
        # crossing BUY iceberg fully self-consumed (6 MW) and cleaned up.
        assert int(state.n_active) == 2
        live_oids = {int(state.iceberg_meta[i, 0]) for i in range(2)}
        assert live_oids == {101, 102}

    def test_deep_book_iceberg_crossed_across_multiple_levels_quiescent(
        self, book,
    ):
        """A regenerated slice that must cross multiple resting price
        levels still resolves to quiescence within the call.

        Resting BUY ladder at 49.00 / 49.50 / 50.00; a SELL iceberg
        peak=2 total=8 at 49.00 drains across all of them. After the
        call, if any iceberg slot remains it must be resting visibly.
        """
        state = _fresh(book)
        for oid, px in ((401, 49.00), (402, 49.50), (403, 50.00)):
            state, _ = book.submit_order(
                state,
                make_order(37, Side.BUY, px, 3.0, oid, 1, 100 + oid,
                           ExecRestriction.NON),
            )
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 49.00, 8.0, 500, 2, 200,
                       ExecRestriction.ICE, peak_qty=2.0),
        )
        assert _quiescence_violations(book, state) == []

    def test_rule3_regen_slice_crosses_resting_aggressor_remainder_quiescent(
        self, book,
    ):
        """Rule 3 single-hop: a NON partial fill leaves a resting BUY
        remainder; the regenerated SELL-iceberg slice crosses it. The
        slot must end quiescent, not peak-consumed.

        SELL iceberg peak=2 total=6; BUY NON 5 MW at the same price.
        """
        state = _fresh(book)
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 6.0, 601, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
        )
        state, _ = book.submit_order(
            state,
            make_order(37, Side.BUY, 50.00, 5.0, 602, 2, 101,
                       ExecRestriction.NON),
        )
        assert _quiescence_violations(book, state) == []

    def test_quiescence_holds_after_every_call_in_a_mixed_session(self, book):
        """The contract is "after EVERY submit_order call" — not just the
        final one. Assert quiescence at each step of a mixed NON / IOC /
        FOK / ICE session with crossing and non-crossing orders.
        """
        state = _fresh(book)
        session = [
            make_order(37, Side.SELL, 50.00, 10.0, 1, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
            make_order(37, Side.BUY, 49.00, 3.0, 2, 2, 101,
                       ExecRestriction.NON),          # rests, no fill
            make_order(37, Side.BUY, 50.00, 1.0, 3, 2, 102,
                       ExecRestriction.IOC),          # crosses peak partially
            make_order(37, Side.BUY, 50.00, 4.0, 4, 3, 103,
                       ExecRestriction.NON),          # crosses, regen
            make_order(37, Side.SELL, 60.00, 2.0, 5, 3, 104,
                       ExecRestriction.NON),          # rests far away
            make_order(37, Side.BUY, 50.00, 2.0, 6, 4, 105,
                       ExecRestriction.FOK),          # FOK against iceberg
        ]
        for o in session:
            state, _ = book.submit_order(state, o)
            assert _quiescence_violations(book, state) == [], (
                f"quiescence violated after order {o.order_id}"
            )

    def test_iceberg_requiring_many_regen_hops_drains_without_truncation(
        self, book,
    ):
        """An iceberg whose hidden volume needs many small slices to
        drain (peak=1, total=40) consumed by one big NON. Within the
        MAX_SLICES_PER_REGEN budget this stays quiescent and does NOT
        trip ``truncated``.
        """
        state = _fresh(book)
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 40.0, 700, 1, 100,
                       ExecRestriction.ICE, peak_qty=1.0),
        )
        state, result = book.submit_order(
            state,
            make_order(37, Side.BUY, 50.00, 50.0, 701, 2, 101,
                       ExecRestriction.NON),
        )
        assert _quiescence_violations(book, state) == []
        assert int(state.n_active) == 0
        assert float(result.trade_qty) == pytest.approx(40.0)
        assert bool(state.truncated) is False


# ---------------------------------------------------------------------------
# Probe family 2 — within-call resolution correctness
# ---------------------------------------------------------------------------

class TestWithinCallResolutionCorrectness:
    """Volume conservation, no phantom fills, idempotence, truncated cap."""

    def test_total_traded_volume_does_not_exceed_iceberg_hidden_liquidity(
        self, book,
    ):
        """An aggressor cannot extract more volume than the iceberg's
        total hidden quantity (no phantom fills from the cascade chase).

        SELL iceberg total=10; BUY NON 100 MW. Traded qty must be exactly
        10 (the full iceberg), never more.
        """
        state = _fresh(book)
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 10.0, 801, 1, 100,
                       ExecRestriction.ICE, peak_qty=3.0),
        )
        state, result = book.submit_order(
            state,
            make_order(37, Side.BUY, 50.00, 100.0, 802, 2, 101,
                       ExecRestriction.NON),
        )
        assert float(result.trade_qty) <= 10.0 + 1e-6
        assert float(result.trade_qty) == pytest.approx(10.0)

    def test_no_phantom_fill_when_aggressor_does_not_cross(self, book):
        """A non-crossing aggressor against a resting iceberg produces
        zero fills — the cascade chase must not invent a trade.
        """
        state = _fresh(book)
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 10.0, 901, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
        )
        state, result = book.submit_order(
            state,
            make_order(37, Side.BUY, 40.00, 5.0, 902, 2, 101,
                       ExecRestriction.NON),
        )
        assert bool(result.traded) is False
        assert float(result.trade_qty) == pytest.approx(0.0)
        # The iceberg is untouched and quiescent.
        assert _quiescence_violations(book, state) == []
        assert int(state.n_active) == 1

    def test_replaying_identical_session_twice_yields_identical_state(
        self, book,
    ):
        """Idempotence: the within-call resolution is deterministic.
        Running the same order sequence from a fresh book twice yields
        an identical observable state fingerprint.
        """
        session = [
            make_order(37, Side.SELL, 50.00, 10.0, 1, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
            make_order(37, Side.BUY, 50.00, 7.0, 2, 2, 101,
                       ExecRestriction.NON),
            make_order(37, Side.BUY, 49.50, 3.0, 3, 2, 102,
                       ExecRestriction.NON),
            make_order(37, Side.SELL, 50.50, 4.0, 4, 3, 103,
                       ExecRestriction.NON),
        ]

        def run():
            s = _fresh(book)
            for o in session:
                s, _ = book.submit_order(s, o)
            return _state_tuple(s)

        assert run() == run()

    def test_truncated_flag_stays_false_for_within_budget_cascade(self, book):
        """On a normal in-budget fixture the bounded fixed-point
        terminates and the ``truncated`` Path B flag stays False.

        This validates the cap is wired but does not false-positive on
        well-behaved input. (The cap-TRIP case — a cascade needing more
        than MAX_SLICES_PER_REGEN=200 hops — is not fast-tier-probeable:
        constructing >200 distinct chained regenerations within one call
        requires either >200 resting levels or a >200-slice iceberg, both
        of which exceed the microsecond fast-suite budget and the
        book_depth<=MAX_ICEBERGS=100 __init__ guard. The cap-trip is left
        to the integration/slow tier; here we pin the non-trip semantics.)
        """
        state = _fresh(book)
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 20.0, 1101, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
        )
        state, _ = book.submit_order(
            state,
            make_order(37, Side.BUY, 50.00, 25.0, 1102, 2, 101,
                       ExecRestriction.NON),
        )
        assert bool(state.truncated) is False

    def test_truncated_flag_set_when_iceberg_total_exceeds_slice_budget(
        self, book,
    ):
        """The Path B truncation flag DOES fire when an iceberg's worst-
        case slice count exceeds MAX_SLICES_PER_REGEN — proving the
        cap detection is live and ORed into ``truncated`` at registration.

        peak=0.1 MW, total=999 MW -> ceil(999/0.1) ~= 9990 slices,
        far above the 200 budget. Registering it must set ``truncated``.
        """
        state = _fresh(book)
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 999.0, 1201, 1, 100,
                       ExecRestriction.ICE, peak_qty=0.1),
        )
        assert bool(state.truncated) is True


# ---------------------------------------------------------------------------
# Probe family 3 — gate equivalence (regen on quiescent no-fill is no-op)
# ---------------------------------------------------------------------------

class TestGateEquivalenceOnQuiescentInvariant:
    """`lax.cond(any_fills, regen, identity)` is observably equivalent to
    unconditional regen because regen on a quiescent state is a no-op.

    We probe the invariant directly via the PUBLIC
    ``IcebergManager.regenerate_slices`` method: forcing a regen on a
    quiescent state returns the lob/meta/n_active/ts_counter unchanged.
    This is exactly what the ``identity`` branch of the gate also
    returns, so the two branches coincide on the quiescent invariant.
    """

    def test_forced_regen_on_quiescent_state_is_a_true_noop(self, book):
        """Build a quiescent state (resting iceberg + non-crossing bid),
        then call ``regenerate_slices`` directly. Every component of the
        returned 4-tuple must equal the input.
        """
        state = _fresh(book)
        for o in (
            make_order(37, Side.SELL, 50.00, 10.0, 201, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
            make_order(37, Side.BUY, 49.00, 3.0, 202, 2, 101,
                       ExecRestriction.NON),
        ):
            state, _ = book.submit_order(state, o)
        assert _quiescence_violations(book, state) == []

        lob2, meta2, na2, ts2 = book._iceberg_mgr.regenerate_slices(
            state.lob, state.iceberg_meta, state.n_active, state.ts_counter,
        )
        assert bool(jnp.array_equal(lob2.bids, state.lob.bids))
        assert bool(jnp.array_equal(lob2.asks, state.lob.asks))
        assert bool(jnp.array_equal(lob2.trades, state.lob.trades))
        assert bool(jnp.array_equal(meta2, state.iceberg_meta))
        assert int(na2) == int(state.n_active)
        # No spurious timestamp bump on a no-regen pass.
        assert int(ts2) == int(state.ts_counter)

    def test_no_fill_order_through_gate_preserves_book_and_quiescence(
        self, book,
    ):
        """A no-fill order routed through the gated production
        ``submit_order`` path leaves the resting iceberg and trade tape
        unchanged on the iceberg-bearing surfaces and stays quiescent.

        The gate's ``identity`` branch (no-fill) must not perturb the
        iceberg meta, the resting asks, or the trade tape — only add the
        new resting bid. Compares the iceberg-relevant observables
        before/after.
        """
        state = _fresh(book)
        for o in (
            make_order(37, Side.SELL, 50.00, 10.0, 301, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
            make_order(37, Side.BUY, 49.00, 3.0, 302, 2, 101,
                       ExecRestriction.NON),
        ):
            state, _ = book.submit_order(state, o)

        asks_before = state.lob.asks
        trades_before = state.lob.trades
        meta_before = state.iceberg_meta
        na_before = int(state.n_active)

        # No-fill BUY far below market -> rests, never crosses.
        state, result = book.submit_order(
            state,
            make_order(37, Side.BUY, 10.00, 1.0, 999, 5, 200,
                       ExecRestriction.NON),
        )
        assert bool(result.traded) is False
        # The gate skipped regen; iceberg surfaces are untouched.
        assert bool(jnp.array_equal(state.lob.asks, asks_before))
        assert bool(jnp.array_equal(state.lob.trades, trades_before))
        assert bool(jnp.array_equal(state.iceberg_meta, meta_before))
        assert int(state.n_active) == na_before
        assert _quiescence_violations(book, state) == []

    def test_repeated_no_fill_orders_do_not_drift_iceberg_state(self, book):
        """Many no-fill orders in a row must leave the iceberg meta
        bit-stable — the gate's identity branch is a pure pass-through,
        so the iceberg side-channel never drifts across no-fill calls.
        """
        state = _fresh(book)
        state, _ = book.submit_order(
            state,
            make_order(37, Side.SELL, 50.00, 10.0, 401, 1, 100,
                       ExecRestriction.ICE, peak_qty=2.0),
        )
        meta_ref = state.iceberg_meta
        na_ref = int(state.n_active)
        for k in range(5):
            state, result = book.submit_order(
                state,
                make_order(37, Side.BUY, 5.00, 1.0, 500 + k, 6, 200 + k,
                           ExecRestriction.NON),
            )
            assert bool(result.traded) is False
            assert bool(jnp.array_equal(state.iceberg_meta, meta_ref))
            assert int(state.n_active) == na_ref
            assert _quiescence_violations(book, state) == []
