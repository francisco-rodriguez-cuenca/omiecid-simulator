"""OMIE-CERT-INVARIANTS-1 — locked broken-fixture tests for I1–I9 and M1–M6.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-CERT-INVARIANTS-1", binding
decisions V1–V16 (row "Step-2 decisions (v5.185, binding)").

Rule 1 of the scenario-ladder plan: a check nobody has seen fail is not evidence.
Every test in the "broken" classes below starts from a **real record built by the
engine** (``build_record``), applies one minimal ``dataclasses.replace`` mutation,
and asserts that the check **fails** through ``assert_check_fails`` (V6, V14).
Positive controls (the unbroken record passes) sit beside them, so a check that
fails on everything is caught too.

The body-inspecting guard (``tests/omiecid/reference/battery_guard.py``, V14) reads
this file: every ``assert_check_fails(<call to the check>, "<ID>", field=...)`` is a
plain statement of a non-skipped ``test_*`` body. Do not wrap those calls in ``if``,
``try``, ``with`` or an inner ``def`` — the guard would (rightly) not count them.

Submission indices are 0-based positions among the tape's real rows (pad rows are
skipped, V2). Order ids come from ``TapeBuilder`` (1, 2, 3, ... in row order).
All runs are on CPU (device policy).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_tape_synth import CapacityEnvelope, TapeBuilder
from omiecid_simulator.omiecid.contract import ContractStatus
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side

try:  # the module under test does not exist before Step 4; keep collection green
    from tests.omiecid.reference.omie_invariants import (
        Cancellation,
        assert_check_fails,
        build_record,
        check_i1_volume_balance,
        check_i2_quantity_conservation,
        check_i3_uncrossed_on_return,
        check_i4_priority,
        check_i5_price_setter,
        check_i6_settlement,
        check_i7_no_manufactured_liquidity,
        check_i8_capacity_headroom,
        check_i9_non_vacuity,
        check_m1_tie_permutation,
        check_m2_monotonicity,
        check_m3_ioc_non_equivalence,
        check_m4_fok_dichotomy,
        check_m5_split_aggressor,
        check_m6_ownership_blindness,
        check_record,
    )

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only before implementation
    _IMPORT_ERROR = exc


def _need() -> None:
    if _IMPORT_ERROR is not None:
        pytest.fail(
            "tests/omiecid/reference/omie_invariants.py is not importable: "
            f"{_IMPORT_ERROR!r}"
        )


@pytest.fixture(autouse=True)
def _require_reference_module():
    _need()


# ---------------------------------------------------------------------------
# Tape helpers (integer ticks; price_int is offset-encoded, 50.00 EUR = 1_004_900)
# ---------------------------------------------------------------------------
BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, IOC, FOK, ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                      int(ExecRestriction.FOK), int(ExecRestriction.ICE))

P40 = 1_003_900  # 40.00 EUR/MWh
P41 = 1_004_000  # 41.00
P50 = 1_004_900  # 50.00
P51 = 1_005_000  # 51.00
P52 = 1_005_100  # 52.00


def _row(side, price_int, qty_int, agent_id, **kw):
    return dict(side=side, price_int=price_int, qty_int=qty_int, agent_id=agent_id, **kw)


def _tape(*rows):
    b = TapeBuilder()
    for r in rows:
        b.add(**r)
    return b.build()


def _swap(seq, i, new):
    """``seq`` (any sequence) as a tuple with element ``i`` replaced."""
    out = list(seq)
    out[i] = new
    return tuple(out)


def _with_key(mapping, key, value):
    d = dict(mapping)
    d[key] = value
    return d


def _without_key(mapping, key):
    d = dict(mapping)
    d.pop(key, None)
    return d


def _add_bid_row(snapshot, price_int, qty_int=10, oid=999, agent=9):
    """A copy of ``snapshot`` with one extra resting bid row in a free slot."""
    bids = np.array(snapshot.bids, copy=True)
    free = np.where(bids[:, 2] <= 0)[0]
    assert free.size, "no free bid row to inject into"
    bids[free[0]] = [price_int, qty_int, oid, agent, 1, 0]
    return replace(snapshot, bids=bids)


# BASIC: four sells (one an iceberg), one rejected buy, one sweeping buy, one
# resting buy. Submission 5 (oid 6) writes trade rows 0..4:
#   (P50,10,maker 1) (P50,10,maker 2) (P51,10,maker 3) (P52,10,maker 4) (P52,5,maker 4)
BASIC_ROWS = (
    _row(SELL, P50, 10, 1),                                   # s0 oid1
    _row(SELL, P50, 10, 2),                                   # s1 oid2
    _row(SELL, P51, 10, 2),                                   # s2 oid3
    _row(SELL, P52, 50, 4, exec_restriction=ICE, peak_int=10),  # s3 oid4
    _row(BUY, P41, 0, 9),                                     # s4 oid5 rejected (qty 0)
    _row(BUY, P52, 45, 3),                                    # s5 oid6 aggressor
    _row(BUY, P40, 20, 5),                                    # s6 oid7 rests
)
BASIC_AGG_SUB = 5


@pytest.fixture(scope="module")
def basic():
    _need()
    return build_record(_tape(*BASIC_ROWS))


@pytest.fixture(scope="module")
def pair():
    """One crossing pair: one fill of 10 at P50 (submission 1)."""
    _need()
    return build_record(_tape(_row(SELL, P50, 10, 1), _row(BUY, P50, 10, 2)))


@pytest.fixture(scope="module")
def nocross():
    _need()
    return build_record(_tape(_row(SELL, P51, 10, 1), _row(BUY, P50, 10, 2)))


@pytest.fixture(scope="module")
def closed():
    _need()
    return build_record(_tape(_row(SELL, P50, 10, 1), _row(BUY, P50, 10, 2)),
                        contract_status=ContractStatus.CLOSED)


@pytest.fixture(scope="module")
def hop_base():
    """Sell iceberg (oid1, 50 @P50, peak 10); far resting buy (oid2, 5 @P40-ish);
    aggressor buy (oid3, 25 @P51), submission 2, fills: ice 10, 10, 5 at P50."""
    _need()
    return build_record(_tape(
        _row(SELL, P50, 50, 1, exec_restriction=ICE, peak_int=10),
        _row(BUY, 1_003_000, 5, 2),
        _row(BUY, P51, 25, 3),
    ))


@pytest.fixture(scope="module")
def icewpi():
    """Sell ICE with ppd +5 (ladder P50, P50+5, P50+10; 3 slices); buy 25 @P50+10.
    Fills at P50, P50+5, P50+10 (submission 1)."""
    _need()
    return build_record(_tape(
        _row(SELL, P50, 30, 1, exec_restriction=ICE, peak_int=10, ppd_int=5),
        _row(BUY, P50 + 10, 25, 2),
    ))


def _fills_of(rec, submission):
    return [f for f in rec.fills if f.submission == submission]


P55 = 1_005_400  # 55.00


@pytest.fixture(scope="module")
def d3_01():
    """D3-01 record (v5.187): sell ICE j (oid1, 10 @P50, peak 2, ppd 0); sell NON k
    (oid2, 5 @P50); buy NON o (oid3, 10 @P55). Submission 2 fills (engine, probed
    2026-10-02): j 2, k 5, j 2, j 1 at P50, aggressor 3, timestamps 3, 3, 5, 7."""
    _need()
    return build_record(_tape(
        _row(SELL, P50, 10, 1, exec_restriction=ICE, peak_int=2),
        _row(SELL, P50, 5, 2),
        _row(BUY, P55, 10, 3),
    ))


def _as_hop(f, ice_oid, own_oid):
    """``f`` relabelled as a Rule-3 hop fill: the iceberg ``ice_oid`` aggresses
    against the submitting order ``own_oid``'s resting remainder. The buyer and
    seller legs are unchanged; passive/aggressor ids and agents swap, and the
    signed quantity flips (its sign follows the aggressor's side)."""
    return replace(f, passive_oid=own_oid, aggressor_oid=ice_oid,
                   passive_agent=f.aggressor_agent, aggressor_agent=f.passive_agent,
                   qty_signed=-f.qty_signed)


def _d3_hop_control(rec):
    """Positive control: keep ``j`` 2 and ``k`` 5; relabel the last two rows
    (``j`` 2, ``j`` 1) as hop fills (passive 3, aggressor 1)."""
    f = tuple(rec.fills)
    return replace(rec, fills=(f[0], f[1], _as_hop(f[2], 1, 3), _as_hop(f[3], 1, 3)))


@pytest.fixture(scope="module")
def ice_buyer_neg_ppd():
    """Sell NON (oid1, 10 @P50); buy ICE (oid2, 30 @P50, peak 10, ppd -5): one fill
    of 10 at P50, the ICE buyer aggressing. Its ladder is P50, P50-5, P50-10, so
    its most permissive rung (the maximum, for a buyer) is ``base`` = P50."""
    _need()
    return build_record(_tape(
        _row(SELL, P50, 10, 1),
        _row(BUY, P50, 30, 2, exec_restriction=ICE, peak_int=10, ppd_int=-5),
    ))


@pytest.fixture(scope="module")
def m5_hop_pair():
    """Whole: sell ICE j (oid1, 10 @P50, peak 2); buy oid2 4 @P55 -> two plain
    fills against j. Split: the same iceberg; buy oid2 2 @P55 and buy oid3 2 @P55
    -> one plain fill each."""
    _need()
    ice = _row(SELL, P50, 10, 1, exec_restriction=ICE, peak_int=2)
    whole = build_record(_tape(ice, _row(BUY, P55, 4, 2)))
    split = build_record(_tape(ice, _row(BUY, P55, 2, 2), _row(BUY, P55, 2, 3)))
    return whole, split


# ---------------------------------------------------------------------------
# Positive controls — the unbroken records pass
# ---------------------------------------------------------------------------
class TestHealthyRecordsPass:
    def test_basic_record_has_the_expected_shape(self, basic):
        assert len(basic.fills) == 5
        assert {f.submission for f in basic.fills} == {BASIC_AGG_SUB}
        assert [f.passive_oid for f in basic.fills] == [1, 2, 3, 4, 4]

    @pytest.mark.parametrize("name", [
        "check_record", "check_i1_volume_balance", "check_i2_quantity_conservation",
        "check_i3_uncrossed_on_return", "check_i4_priority", "check_i5_price_setter",
        "check_i6_settlement", "check_i7_no_manufactured_liquidity",
        "check_i8_capacity_headroom",
    ])
    def test_basic_record_passes(self, basic, name):
        res = globals()[name](basic)
        assert res.status == "pass", res.violations

    def test_basic_record_passes_i9_expecting_a_trade(self, basic):
        assert check_i9_non_vacuity(basic, expects_trade=True).status == "pass"

    def test_hop_base_passes_i5_with_no_hop(self, hop_base):
        res = check_i5_price_setter(hop_base)
        assert res.status == "pass", res.violations
        assert res.evidence["rule3_hop_fills"] == 0

    def test_icewpi_record_passes_i5_on_its_declared_ladder(self, icewpi):
        assert sorted(f.price_int for f in icewpi.fills) == [P50, P50 + 5, P50 + 10]
        res = check_i5_price_setter(icewpi)
        assert res.status == "pass", res.violations


# ---------------------------------------------------------------------------
# I1 — matched buy volume == matched sell volume, each side read independently
# ---------------------------------------------------------------------------
class TestI1Broken:
    def test_i1_fails_when_both_legs_of_a_fill_are_on_the_sell_side(self, basic):
        f = basic.fills[0]
        broken = replace(basic, fills=_swap(basic.fills, 0, replace(f, buyer_oid=f.seller_oid)))
        res = check_i1_volume_balance(broken)
        assert_check_fails(res, "I1")

    def test_i1_fails_when_a_fill_names_an_order_that_was_never_accepted(self, basic):
        f = basic.fills[1]
        broken = replace(basic, fills=_swap(basic.fills, 1, replace(f, seller_oid=99)))
        res = check_i1_volume_balance(broken)
        assert_check_fails(res, "I1")

    def test_i1_fails_when_a_fill_names_the_rejected_order(self, basic):
        f = basic.fills[0]
        broken = replace(basic, fills=_swap(basic.fills, 0, replace(f, buyer_oid=5)))
        assert_check_fails(check_i1_volume_balance(broken), "I1")

    def test_i1_fails_on_a_fill_whose_buyer_and_seller_legs_are_swapped(self, basic):
        # V7 as amended at Step 3 (v5.186): a swapped fill leaves BOTH totals by
        # the same qty, so the totals stay equal; only the per-fill leg check
        # (field="fill_sides") can see it.
        f = basic.fills[0]
        swapped = replace(f, buyer_oid=f.seller_oid, seller_oid=f.buyer_oid,
                          buyer_agent=f.seller_agent, seller_agent=f.buyer_agent)
        broken = replace(basic, fills=_swap(basic.fills, 0, swapped))
        res = check_i1_volume_balance(broken)
        assert_check_fails(res, "I1", field="fill_sides")
        # Why the totals alone are blind: recomputed in test-local arithmetic.
        acc = broken.accepted
        buy_total = sum(x.qty_int for x in broken.fills
                        if x.buyer_oid in acc and int(acc[x.buyer_oid].side) == BUY)
        sell_total = sum(x.qty_int for x in broken.fills
                         if x.seller_oid in acc and int(acc[x.seller_oid].side) == SELL)
        assert buy_total == sell_total
        sides = [v for v in res.violations if v.field == "fill_sides"]
        assert any(v.submission == f.submission and v.fill_row == f.row_index
                   and v.oid in (f.buyer_oid, f.seller_oid) for v in sides), sides


# ---------------------------------------------------------------------------
# I2 — quantity conservation per accepted order, buckets from engine state
# ---------------------------------------------------------------------------
class TestI2Broken:
    def test_i2_fails_when_an_accepted_quantity_is_not_accounted_for(self, basic):
        a = basic.accepted[6]
        broken = replace(basic, accepted=_with_key(basic.accepted, 6, replace(a, qty_int=46)))
        res = check_i2_quantity_conservation(broken)
        assert_check_fails(res, "I2")

    def test_i2_fails_when_the_hidden_bucket_is_lost(self, basic):
        broken = replace(basic, hidden_remaining=_with_key(basic.hidden_remaining, 4, 0))
        res = check_i2_quantity_conservation(broken)
        assert_check_fails(res, "I2")

    def test_i2_fails_on_a_non_zero_eviction_outside_c14(self, basic):
        # The identity still balances (20 moved from resting to evicted); the
        # failure must come from "evicted must be 0 outside C14".
        broken = replace(
            basic,
            still_resting=_with_key(basic.still_resting, 7, 0),
            evicted=_with_key(basic.evicted, 7, 20),
        )
        res = check_i2_quantity_conservation(broken)
        assert_check_fails(res, "I2", field="evicted")

    def test_i2_accepts_the_same_balanced_eviction_when_eviction_is_allowed(self, basic):
        broken = replace(
            basic,
            still_resting=_with_key(basic.still_resting, 7, 0),
            evicted=_with_key(basic.evicted, 7, 20),
        )
        res = check_i2_quantity_conservation(broken, allow_eviction=True)
        assert res.status == "pass", res.violations

    def test_i2_fails_when_an_eviction_is_filed_as_a_cancellation(self, basic):
        # oid 7 disappears "by cancellation" though no cancel event names it.
        broken = replace(
            basic,
            still_resting=_with_key(basic.still_resting, 7, 0),
            cancelled_or_expired=_with_key(
                basic.cancelled_or_expired, 7,
                (Cancellation(oid=7, cause="cancellation", qty_int=20, event_index=0),),
            ),
        )
        res = check_i2_quantity_conservation(broken)
        assert_check_fails(res, "I2", field="cancelled_or_expired")

    def test_i2_fails_when_a_non_order_is_given_an_ioc_remainder(self, basic):
        # NON and ICE orders get no cancellation of their own (V4).
        broken = replace(
            basic,
            still_resting=_with_key(basic.still_resting, 7, 0),
            cancelled_or_expired=_with_key(
                basic.cancelled_or_expired, 7,
                (Cancellation(oid=7, cause="ioc_remainder", qty_int=20, event_index=6),),
            ),
        )
        res = check_i2_quantity_conservation(broken)
        assert_check_fails(res, "I2", field="cancelled_or_expired")

    def test_i2_fails_on_a_cause_outside_the_section_5_1_list(self, basic):
        broken = replace(
            basic,
            still_resting=_with_key(basic.still_resting, 7, 0),
            cancelled_or_expired=_with_key(
                basic.cancelled_or_expired, 7,
                (Cancellation(oid=7, cause="eviction", qty_int=20, event_index=0),),
            ),
        )
        res = check_i2_quantity_conservation(broken)
        assert_check_fails(res, "I2", field="cancelled_or_expired")

    def test_i2_fails_when_filled_double_counts(self, basic):
        broken = replace(basic, filled=_with_key(basic.filled, 1, basic.filled[1] + 10))
        res = check_i2_quantity_conservation(broken)
        assert_check_fails(res, "I2")


# ---------------------------------------------------------------------------
# I3 — never left crossed after any return, the last submission included
# ---------------------------------------------------------------------------
class TestI3Broken:
    def test_i3_fails_on_a_crossed_book_after_the_last_submission_only(self, basic):
        last = len(basic.post_snapshots) - 1
        snap = _add_bid_row(basic.post_snapshots[last], P52 + 100)
        broken = replace(basic, post_snapshots=_swap(basic.post_snapshots, last, snap))
        res = check_i3_uncrossed_on_return(broken)
        assert_check_fails(res, "I3")

    def test_i3_fails_on_a_crossed_book_after_a_middle_submission(self, basic):
        snap = _add_bid_row(basic.post_snapshots[3], P51)
        broken = replace(basic, post_snapshots=_swap(basic.post_snapshots, 3, snap))
        res = check_i3_uncrossed_on_return(broken)
        assert_check_fails(res, "I3")

    def test_i3_fails_on_a_locked_book_best_bid_equal_best_ask(self, basic):
        last = len(basic.post_snapshots) - 1
        snap = _add_bid_row(basic.post_snapshots[last], P52)
        broken = replace(basic, post_snapshots=_swap(basic.post_snapshots, last, snap))
        res = check_i3_uncrossed_on_return(broken)
        assert_check_fails(res, "I3")

    def test_i3_fails_on_a_crossed_final_snapshot(self, basic):
        broken = replace(basic, final_snapshot=_add_bid_row(basic.final_snapshot, P52 + 100))
        res = check_i3_uncrossed_on_return(broken)
        assert_check_fails(res, "I3")


# ---------------------------------------------------------------------------
# I4 — price-time priority against the book before each submission
# ---------------------------------------------------------------------------
class TestI4Broken:
    def test_i4_fails_when_a_worse_priced_maker_fills_while_a_better_one_has_depth(self, basic):
        f = basic.fills[0]  # was maker oid1 @P50; now "maker oid3 @P51"
        bad = replace(f, passive_oid=3, seller_oid=3, price_int=P51)
        broken = replace(basic, fills=_swap(basic.fills, 0, bad))
        res = check_i4_priority(broken)
        assert_check_fails(res, "I4")

    def test_i4_fails_when_a_later_equal_priced_maker_fills_first(self, basic):
        f = basic.fills[0]  # was oid1 (earlier); now oid2 (later, same price)
        bad = replace(f, passive_oid=2, seller_oid=2, passive_agent=2, seller_agent=2)
        broken = replace(basic, fills=_swap(basic.fills, 0, bad))
        res = check_i4_priority(broken)
        assert_check_fails(res, "I4")

    # --- V8 as amended at Step 5 (v5.187), D3-01: priority on the iceberg's side
    # at the Rule-3 hop ---
    def test_d3_01_real_record_has_the_probed_fills_and_passes_i4_and_i5(self, d3_01):
        got = [(f.submission, f.passive_oid, f.aggressor_oid, f.qty_int, f.price_int,
                f.sec) for f in d3_01.fills]
        assert got == [(2, 1, 3, 2, P50, 3), (2, 2, 3, 5, P50, 3),
                       (2, 1, 3, 2, P50, 5), (2, 1, 3, 1, P50, 7)]
        r4 = check_i4_priority(d3_01)
        assert r4.status == "pass", r4.violations
        r5 = check_i5_price_setter(d3_01)
        assert r5.status == "pass", r5.violations

    def test_i4_fails_on_a_hop_while_another_entry_on_the_icebergs_side_has_depth(self, d3_01):
        # Keep the first j 2 row; replace the rest by four hop fills of 2 at P50
        # (passive 3, aggressor 1, buyer 3, seller 1). k (oid2, 5 @P50) still has
        # depth 5 on the iceberg's (sell) side at a price <= the fill price.
        f = tuple(d3_01.fills)
        hop = _as_hop(replace(f[2], qty_int=2, qty_signed=-2, price_int=P50), 1, 3)
        assert (hop.passive_oid, hop.aggressor_oid, hop.buyer_oid, hop.seller_oid) == (3, 1, 3, 1)
        hops = tuple(replace(hop, row_index=f[0].row_index + i) for i in range(1, 5))
        broken = replace(d3_01, fills=(f[0],) + hops)
        res = check_i4_priority(broken)
        assert_check_fails(res, "I4", field="hop_priority")
        hp = [v for v in res.violations if v.field == "hop_priority"]
        assert any(v.oid == 2 and v.submission == 2 for v in hp), hp

    def test_i4_accepts_a_hop_once_the_icebergs_side_is_exhausted(self, d3_01):
        ctl = _d3_hop_control(d3_01)
        res = check_i4_priority(ctl)
        assert res.status == "pass", res.violations
        assert res.evidence["hop_fills"] == 2


# ---------------------------------------------------------------------------
# I5 — every trade price equals its price-setter's limit (Rule-3 hop: the iceberg)
# ---------------------------------------------------------------------------
class TestI5Broken:
    def test_i5_fails_on_a_trade_at_the_aggressors_limit(self, basic):
        f = basic.fills[0]  # maker oid1 declared P50; aggressor oid6 declared P52
        broken = replace(basic, fills=_swap(basic.fills, 0, replace(f, price_int=P52)))
        res = check_i5_price_setter(broken)
        assert_check_fails(res, "I5")

    def test_i5_fails_on_a_price_that_merely_rests_on_that_side(self, basic):
        # Rejected formulation: "any price resting on that side of the book".
        f = basic.fills[2]  # maker oid3 declared P51; P50 also rests on the ask side
        broken = replace(basic, fills=_swap(basic.fills, 2, replace(f, price_int=P50)))
        res = check_i5_price_setter(broken)
        assert_check_fails(res, "I5")

    def test_i5_fails_on_an_iceberg_slice_off_its_flat_ladder(self, basic):
        f = basic.fills[3]  # maker oid4, ICE @P52 with ppd 0: ladder is {P52}
        broken = replace(basic, fills=_swap(basic.fills, 3, replace(f, price_int=P52 + 1)))
        res = check_i5_price_setter(broken)
        assert_check_fails(res, "I5")

    def test_i5_fails_on_a_ladder_step_beyond_the_declared_slice_count(self, icewpi):
        # ceil(30/10) = 3 slices: k in {0,1,2}; k = 3 is off the declared ladder.
        i = [f.price_int for f in icewpi.fills].index(P50 + 5)
        bad = replace(icewpi.fills[i], price_int=P50 + 15)
        broken = replace(icewpi, fills=_swap(icewpi.fills, i, bad))
        res = check_i5_price_setter(broken)
        assert_check_fails(res, "I5")

    def test_i5_fails_on_a_price_between_ladder_steps(self, icewpi):
        i = [f.price_int for f in icewpi.fills].index(P50 + 5)
        bad = replace(icewpi.fills[i], price_int=P50 + 2)
        broken = replace(icewpi, fills=_swap(icewpi.fills, i, bad))
        res = check_i5_price_setter(broken)
        assert_check_fails(res, "I5")

    def test_i5_accepts_any_step_of_the_declared_ladder(self, icewpi):
        # The narrow exemption: base + k*ppd with 0 <= k < ceil(qty/peak).
        i = [f.price_int for f in icewpi.fills].index(P50 + 5)
        moved = replace(icewpi.fills[i], price_int=P50 + 10)
        res = check_i5_price_setter(replace(icewpi, fills=_swap(icewpi.fills, i, moved)))
        assert res.status == "pass", res.violations

    def test_i5_fails_on_a_rule3_hop_priced_at_the_aggressors_limit(self, hop_base):
        # Submission 2 (oid3). A resting iceberg slice (oid1) aggresses against
        # oid3's own resting remainder: the Rule-3 hop. OQ-31 Reading B: it
        # prices at the iceberg (P50), never at oid3's limit (P51).
        f = _fills_of(hop_base, 2)[1]
        i = hop_base.fills.index(f)
        hop = replace(f, aggressor_oid=1, passive_oid=3, price_int=P51)
        broken = replace(hop_base, fills=_swap(hop_base.fills, i, hop))
        res = check_i5_price_setter(broken)
        assert_check_fails(res, "I5", field="rule3_hop")

    def test_i5_accepts_a_rule3_hop_priced_at_the_iceberg_and_counts_it(self, hop_base):
        f = _fills_of(hop_base, 2)[1]
        i = hop_base.fills.index(f)
        hop = replace(f, aggressor_oid=1, passive_oid=3, price_int=P50)
        res = check_i5_price_setter(replace(hop_base, fills=_swap(hop_base.fills, i, hop)))
        assert res.status == "pass", res.violations
        assert res.evidence["rule3_hop_fills"] == 1

    def test_i5_fails_on_an_unruled_hop_against_another_resting_order(self, hop_base):
        # The slice aggresses against oid2, not the submitting order's remainder:
        # no ruling covers it, so I5 fails it (even at the iceberg's price).
        f = _fills_of(hop_base, 2)[1]
        i = hop_base.fills.index(f)
        hop = replace(f, aggressor_oid=1, passive_oid=2, price_int=P50)
        broken = replace(hop_base, fills=_swap(hop_base.fills, i, hop))
        res = check_i5_price_setter(broken)
        assert_check_fails(res, "I5", field="unruled_hop")

    # --- V9 as amended at Step 5 (v5.187): D3-02 limit_cross, D3-03, D3-04 ---
    def test_i5_accepts_the_d3_01_hop_control_and_counts_two_hops(self, d3_01):
        res = check_i5_price_setter(_d3_hop_control(d3_01))
        assert res.status == "pass", res.violations
        assert res.evidence["rule3_hop_fills"] == 2

    def test_i5_fails_a_hop_whose_aggressor_is_not_an_ice_order(self, d3_01):
        # D3-03: the hop branch needs aggressor_oid to be an accepted ICE order on
        # the side opposite o. k (oid2) is a NON sell.
        ctl = _d3_hop_control(d3_01)
        bad = replace(ctl.fills[3], aggressor_oid=2)
        broken = replace(ctl, fills=_swap(ctl.fills, 3, bad))
        res = check_i5_price_setter(broken)
        assert_check_fails(res, "I5", field="unruled_hop")

    def test_the_unruled_hop_message_names_the_probable_causes(self, hop_base):
        # D3-04: the reworded message cites section 3.3 and Rule 4.
        f = _fills_of(hop_base, 2)[1]
        i = hop_base.fills.index(f)
        hop = replace(f, aggressor_oid=1, passive_oid=2, price_int=P50)
        res = check_i5_price_setter(replace(hop_base, fills=_swap(hop_base.fills, i, hop)))
        msgs = [v.message for v in res.violations if v.field == "unruled_hop"]
        assert msgs, res.violations
        assert any("§3.3" in m and "Rule 4" in m for m in msgs), msgs

    def test_i5_fails_a_fill_above_the_aggressing_buyers_limit(self, basic):
        # basic.fills[0]: buyer oid6 is the aggressor, price P50, setter oid1 @P50
        # (the price-setter rule still holds). Lower the buyer's declared limit to
        # one tick below the fill price: the fill no longer crosses.
        f = basic.fills[0]
        assert f.aggressor_oid == f.buyer_oid == 6
        a = basic.accepted[6]
        broken = replace(basic, accepted=_with_key(
            basic.accepted, 6, replace(a, price_int=f.price_int - 1)))
        res = check_i5_price_setter(broken)
        assert_check_fails(res, "I5", field="limit_cross")
        lc = [v for v in res.violations if v.field == "limit_cross"]
        assert any(v.oid == 6 and v.submission == f.submission
                   and v.fill_row == f.row_index for v in lc), lc

    def test_i5_ice_buyer_with_negative_ppd_crosses_at_its_base(self, ice_buyer_neg_ppd):
        # The ICE leg's limit is its most permissive rung: the maximum for a buyer,
        # which is base when ppd < 0.
        (f,) = ice_buyer_neg_ppd.fills
        assert (f.buyer_oid, f.price_int) == (2, P50)
        res = check_i5_price_setter(ice_buyer_neg_ppd)
        assert res.status == "pass", res.violations

    def test_i5_ice_buyer_with_negative_ppd_fails_limit_cross_one_tick_above_base(
            self, ice_buyer_neg_ppd):
        # Move the fill to base + 1 and the setter's (seller oid1) declaration with
        # it, so the price-setter rule still holds and only the ICE buyer's leg
        # fails to cross.
        rec = ice_buyer_neg_ppd
        f = rec.fills[0]
        acc = _with_key(rec.accepted, 1, replace(rec.accepted[1], price_int=P50 + 1))
        broken = replace(rec, accepted=acc,
                         fills=_swap(rec.fills, 0, replace(f, price_int=P50 + 1)))
        res = check_i5_price_setter(broken)
        assert_check_fails(res, "I5", field="limit_cross")
        assert {v.field for v in res.violations} == {"limit_cross"}, res.violations
        assert {v.oid for v in res.violations} == {2}


# ---------------------------------------------------------------------------
# I6 — (a) settlements sum to zero within tolerance; (b) reports agree with rows
# ---------------------------------------------------------------------------
class TestI6Broken:
    def test_i6_fails_when_one_participants_settlement_leg_is_dropped(self, basic):
        traded_agents = {f.buyer_agent for f in basic.fills} | {f.seller_agent for f in basic.fills}
        assert 4 in traded_agents
        broken = replace(basic, engine_settlements=_without_key(basic.engine_settlements, 4))
        res = check_i6_settlement(broken)
        assert_check_fails(res, "I6", field="engine_settlements")

    def test_i6_fails_when_settlements_do_not_sum_to_zero(self, basic):
        broken = replace(
            basic,
            engine_settlements=_with_key(
                basic.engine_settlements, 1, basic.engine_settlements[1] + 1.0),
        )
        res = check_i6_settlement(broken)
        assert_check_fails(res, "I6", field="engine_settlements")

    def test_i6_fails_when_the_reported_settlement_value_disagrees(self, basic):
        er = basic.engine_reports[BASIC_AGG_SUB]
        bad = replace(er, settlement_value_eur=er.settlement_value_eur + 1.0)
        broken = replace(basic, engine_reports=_swap(basic.engine_reports, BASIC_AGG_SUB, bad))
        res = check_i6_settlement(broken)
        assert_check_fails(res, "I6", field="engine_reports")

    def test_i6_fails_when_the_reported_quantity_is_one_tick_off(self, basic):
        er = basic.engine_reports[BASIC_AGG_SUB]
        bad = replace(er, trade_qty_mw=er.trade_qty_mw + 0.1)
        broken = replace(basic, engine_reports=_swap(basic.engine_reports, BASIC_AGG_SUB, bad))
        res = check_i6_settlement(broken)
        assert_check_fails(res, "I6", field="engine_reports")

    def test_i6_fails_when_a_trading_submission_reports_no_trade(self, basic):
        er = basic.engine_reports[BASIC_AGG_SUB]
        broken = replace(basic, engine_reports=_swap(
            basic.engine_reports, BASIC_AGG_SUB, replace(er, traded=False)))
        res = check_i6_settlement(broken)
        assert_check_fails(res, "I6", field="engine_reports")

    def test_i6_fails_when_a_non_trading_submission_reports_a_trade(self, basic):
        er = basic.engine_reports[0]
        broken = replace(basic, engine_reports=_swap(
            basic.engine_reports, 0, replace(er, traded=True)))
        res = check_i6_settlement(broken)
        assert_check_fails(res, "I6", field="engine_reports")

    def test_i6_fails_when_a_trade_rows_buyer_is_not_the_declared_agent(self, basic):
        f = basic.fills[0]
        broken = replace(basic, fills=_swap(basic.fills, 0, replace(f, buyer_agent=99)))
        res = check_i6_settlement(broken)
        assert_check_fails(res, "I6")


# ---------------------------------------------------------------------------
# I7 — total traded <= min(total buy accepted, total sell accepted)
# ---------------------------------------------------------------------------
class TestI7Broken:
    def test_i7_fails_on_manufactured_liquidity(self, pair):
        f = pair.fills[0]
        extra = replace(f, row_index=f.row_index + 1)
        broken = replace(pair, fills=tuple(pair.fills) + (extra,))
        res = check_i7_no_manufactured_liquidity(broken)
        assert_check_fails(res, "I7")

    def test_i7_fails_when_the_buy_side_declared_less_than_traded(self, basic):
        acc = _with_key(basic.accepted, 6, replace(basic.accepted[6], qty_int=20))
        acc = _with_key(acc, 7, replace(acc[7], qty_int=1))
        broken = replace(basic, accepted=acc)  # buy declared 21 < traded 45
        res = check_i7_no_manufactured_liquidity(broken)
        assert_check_fails(res, "I7")


# ---------------------------------------------------------------------------
# I8 — capacity headroom: a breach is INVALID, not failed
# ---------------------------------------------------------------------------
class TestI8Broken:
    def test_i8_is_invalid_when_book_full_was_raised(self, basic):
        broken = replace(basic, flags=_with_key(basic.flags, "book_full", True))
        res = check_i8_capacity_headroom(broken)
        assert_check_fails(res, "I8", status="invalid")

    def test_i8_is_invalid_when_truncated_was_raised(self, basic):
        broken = replace(basic, flags=_with_key(basic.flags, "truncated", True))
        res = check_i8_capacity_headroom(broken)
        assert_check_fails(res, "I8", status="invalid")

    def test_i8_is_invalid_when_the_trade_buffer_saturated(self, basic):
        broken = replace(basic, flags=_with_key(basic.flags, "trade_buffer_saturated", True))
        res = check_i8_capacity_headroom(broken)
        assert_check_fails(res, "I8", status="invalid")

    def test_i8_is_invalid_when_a_snapshot_is_truncated_though_flags_are_clear(self, basic):
        # V11 as amended at Step 5 (v5.187): I8 does not trust rec.flags.
        assert not any(basic.flags.values())
        last = len(basic.post_snapshots) - 1
        snap = replace(basic.post_snapshots[last], truncated=True)
        broken = replace(basic, post_snapshots=_swap(basic.post_snapshots, last, snap))
        res = check_i8_capacity_headroom(broken)
        assert_check_fails(res, "I8", status="invalid")
        assert res.evidence["flags"]["truncated"] is True

    def test_i8_is_invalid_when_the_hop_bound_exceeds_the_margin(self, basic):
        broken = replace(basic, capacity_bounds=_with_key(basic.capacity_bounds, "hops", 161))
        res = check_i8_capacity_headroom(broken)
        assert_check_fails(res, "I8", status="invalid")

    def test_i8_is_invalid_when_resting_per_side_exceeds_a_tight_envelope(self, basic):
        # BASIC peaks at 4 resting asks (before submission 5).
        res = check_i8_capacity_headroom(basic, envelope=CapacityEnvelope(max_resting_per_side=3))
        assert_check_fails(res, "I8", status="invalid")

    def test_i8_passes_at_resting_per_side_equal_to_the_margin(self, basic):
        res = check_i8_capacity_headroom(basic, envelope=CapacityEnvelope(max_resting_per_side=4))
        assert res.status == "pass", res.violations

    def test_i8_is_invalid_when_trades_exceed_a_tight_envelope(self, basic):
        # BASIC writes 5 trade rows.
        res = check_i8_capacity_headroom(basic, envelope=CapacityEnvelope(max_trades=4))
        assert_check_fails(res, "I8", status="invalid")

    def test_i8_passes_at_trades_equal_to_the_margin(self, basic):
        res = check_i8_capacity_headroom(basic, envelope=CapacityEnvelope(max_trades=5))
        assert res.status == "pass", res.violations

    def test_i8_breach_is_not_applicable_in_c14_mode_with_full_evidence(self, basic):
        broken = replace(basic, flags=_with_key(basic.flags, "book_full", True))
        invalid = check_i8_capacity_headroom(broken)
        na = check_i8_capacity_headroom(broken, allow_breach=True)
        assert invalid.status == "invalid"
        assert na.status == "not_applicable"
        assert set(invalid.evidence) <= set(na.evidence)


# ---------------------------------------------------------------------------
# I9 — non-vacuity
# ---------------------------------------------------------------------------
class TestI9Broken:
    def test_i9_fails_when_nothing_was_accepted(self, closed):
        assert closed.accepted == {} or len(closed.accepted) == 0
        res = check_i9_non_vacuity(closed, expects_trade=False)
        assert_check_fails(res, "I9")

    def test_i9_fails_when_a_trade_is_expected_and_none_happened(self, nocross):
        res = check_i9_non_vacuity(nocross, expects_trade=True)
        assert_check_fails(res, "I9")

    def test_i9_passes_on_the_same_run_when_no_trade_is_expected(self, nocross):
        assert check_i9_non_vacuity(nocross, expects_trade=False).status == "pass"

    def test_i9_fails_when_the_fills_are_dropped(self, basic):
        broken = replace(basic, fills=())
        res = check_i9_non_vacuity(broken, expects_trade=True)
        assert_check_fails(res, "I9")


# ---------------------------------------------------------------------------
# M1–M6 — metamorphic relations over real pairs
# ---------------------------------------------------------------------------
M1_ROWS = (
    _row(SELL, P50, 10, 1, timestamp=5, order_id=1),
    _row(SELL, P50, 10, 2, timestamp=5, order_id=2),
    _row(BUY, P51, 15, 3, timestamp=6, order_id=3),
)


@pytest.fixture(scope="module")
def m1_pair():
    _need()
    base = build_record(_tape(*M1_ROWS))
    permuted = build_record(_tape(M1_ROWS[1], M1_ROWS[0], M1_ROWS[2]))
    return base, permuted


@pytest.fixture(scope="module")
def m2_size_pair():
    _need()
    rests = (_row(SELL, P50, 10, 1), _row(SELL, P51, 10, 2))
    base = build_record(_tape(*rests, _row(BUY, P52, 10, 3)))
    variant = build_record(_tape(*rests, _row(BUY, P52, 15, 3)))
    return base, variant


@pytest.fixture(scope="module")
def m2_limit_pair():
    _need()
    rests = (_row(SELL, P50, 10, 1), _row(SELL, P51, 10, 2))
    base = build_record(_tape(*rests, _row(BUY, P52, 20, 3)))
    variant = build_record(_tape(*rests, _row(BUY, P50, 20, 3)))
    return base, variant


@pytest.fixture(scope="module")
def m3_pair():
    _need()
    ioc = build_record(_tape(_row(SELL, P50, 10, 1),
                             _row(BUY, P50, 20, 2, exec_restriction=IOC)))
    non = build_record(_tape(_row(SELL, P50, 10, 1), _row(BUY, P50, 20, 2)))
    return ioc, non


@pytest.fixture(scope="module")
def m4_killed():
    _need()
    return build_record(_tape(_row(SELL, P50, 10, 1),
                              _row(BUY, P50, 20, 2, exec_restriction=FOK)))


@pytest.fixture(scope="module")
def m4_filled():
    _need()
    return build_record(_tape(_row(SELL, P50, 20, 1),
                              _row(BUY, P50, 20, 2, exec_restriction=FOK)))


@pytest.fixture(scope="module")
def m5_pair():
    _need()
    rests = (_row(SELL, P50, 10, 1), _row(SELL, P51, 10, 2))
    whole = build_record(_tape(*rests, _row(BUY, P52, 20, 3)))
    split = build_record(_tape(*rests, _row(BUY, P52, 10, 3), _row(BUY, P52, 10, 4)))
    return whole, split


@pytest.fixture(scope="module")
def m6_pair():
    _need()
    base = build_record(_tape(_row(SELL, P50, 10, 1), _row(BUY, P50, 10, 2)))
    same_owner = build_record(_tape(_row(SELL, P50, 10, 7), _row(BUY, P50, 10, 7)))
    return base, same_owner


class TestMetamorphicHealthyPairsPass:
    def test_m1_healthy(self, m1_pair):
        assert check_m1_tie_permutation(*m1_pair, tie_oids=(1, 2)).status == "pass"

    def test_m2_size_healthy(self, m2_size_pair):
        res = check_m2_monotonicity(*m2_size_pair, aggressor_oid=3, relation="size_increased")
        assert res.status == "pass", res.violations

    def test_m2_limit_healthy(self, m2_limit_pair):
        res = check_m2_monotonicity(*m2_limit_pair, aggressor_oid=3, relation="limit_tightened")
        assert res.status == "pass", res.violations

    def test_m3_healthy(self, m3_pair):
        assert check_m3_ioc_non_equivalence(*m3_pair, oid=2).status == "pass"

    def test_m4_killed_healthy(self, m4_killed):
        assert check_m4_fok_dichotomy(m4_killed, fok_oid=2).status == "pass"

    def test_m4_filled_healthy(self, m4_filled):
        assert check_m4_fok_dichotomy(m4_filled, fok_oid=2).status == "pass"

    def test_m5_healthy(self, m5_pair):
        res = check_m5_split_aggressor(*m5_pair, whole_oid=3, split_oids=(3, 4))
        assert res.status == "pass", res.violations

    def test_m5_counts_a_hop_against_the_iceberg_like_a_plain_fill(self, m5_hop_pair):
        # V12 as amended at Step 5 (v5.187): the M5 map is keyed by counterparty,
        # the iceberg aggressor_oid at a hop. Whole: two plain fills against j
        # (oid1). Split: oid3's fill against j relabelled as a hop (passive 3,
        # aggressor 1). The same trades, so the maps agree.
        whole, split = m5_hop_pair
        assert [(f.passive_oid, f.aggressor_oid, f.qty_int, f.price_int)
                for f in whole.fills] == [(1, 2, 2, P50), (1, 2, 2, P50)]
        assert [(f.submission, f.passive_oid, f.aggressor_oid) for f in split.fills] == [
            (1, 1, 2), (2, 1, 3)]
        hopped = replace(split, fills=_swap(split.fills, 1, _as_hop(split.fills[1], 1, 3)))
        res = check_m5_split_aggressor(whole, hopped, whole_oid=2, split_oids=(2, 3))
        assert res.status == "pass", res.violations

    def test_m6_healthy(self, m6_pair):
        res = check_m6_ownership_blindness(*m6_pair, agent_map={1: 7, 2: 7})
        assert res.status == "pass", res.violations


class TestMetamorphicBroken:
    def test_m1_fails_when_the_permuted_run_matches_less(self, m1_pair):
        base, permuted = m1_pair
        broken = replace(permuted, fills=tuple(permuted.fills)[:1])
        res = check_m1_tie_permutation(base, broken, tie_oids=(1, 2))
        assert_check_fails(res, "M1")

    def test_m2_fails_when_a_larger_aggressor_matches_less(self, m2_size_pair):
        base, variant = m2_size_pair  # base matches 10, variant 15
        broken = replace(variant, fills=tuple(variant.fills)[1:])  # variant now 5
        res = check_m2_monotonicity(base, broken, aggressor_oid=3, relation="size_increased")
        assert_check_fails(res, "M2")

    def test_m2_fails_when_a_tighter_limit_matches_more(self, m2_limit_pair):
        base, variant = m2_limit_pair  # base matches 20, variant 10
        f = variant.fills[0]
        extra = replace(f, row_index=f.row_index + 1, qty_int=15, passive_oid=2,
                        seller_oid=2, price_int=P51)
        broken = replace(variant, fills=tuple(variant.fills) + (extra,))  # variant now 25
        res = check_m2_monotonicity(base, broken, aggressor_oid=3, relation="limit_tightened")
        assert_check_fails(res, "M2")

    def test_m3_fails_when_the_ioc_remainder_rests(self, m3_pair):
        ioc, non = m3_pair
        broken = replace(ioc, still_resting=_with_key(ioc.still_resting, 2, 10))
        res = check_m3_ioc_non_equivalence(broken, non, oid=2)
        assert_check_fails(res, "M3")

    def test_m3_fails_when_the_ioc_remainder_appears_in_a_post_snapshot(self, m3_pair):
        ioc, non = m3_pair
        snap = _add_bid_row(ioc.post_snapshots[1], P50 - 100, qty_int=10, oid=2, agent=2)
        broken = replace(ioc, post_snapshots=_swap(ioc.post_snapshots, 1, snap))
        res = check_m3_ioc_non_equivalence(broken, non, oid=2)
        assert_check_fails(res, "M3")

    def test_m3_fails_when_the_trade_lists_differ(self, m3_pair):
        ioc, non = m3_pair
        broken = replace(ioc, fills=_swap(ioc.fills, 0, replace(ioc.fills[0], qty_int=5)))
        res = check_m3_ioc_non_equivalence(broken, non, oid=2)
        assert_check_fails(res, "M3")

    def test_m4_fails_when_a_killed_fok_leaves_the_book_changed(self, m4_killed):
        assert len(m4_killed.fills) == 0
        snap = m4_killed.post_snapshots[1]
        asks = np.array(snap.asks, copy=True)
        live = np.where(asks[:, 2] > 0)[0][0]
        asks[live, 1] -= 1
        broken = replace(m4_killed, post_snapshots=_swap(
            m4_killed.post_snapshots, 1, replace(snap, asks=asks)))
        res = check_m4_fok_dichotomy(broken, fok_oid=2)
        assert_check_fails(res, "M4")

    def test_m4_fails_when_a_fok_fills_partially(self, m4_filled):
        f = m4_filled.fills[0]
        broken = replace(m4_filled, fills=_swap(m4_filled.fills, 0, replace(f, qty_int=10)))
        res = check_m4_fok_dichotomy(broken, fok_oid=2)
        assert_check_fails(res, "M4")

    def test_m4_fails_when_a_killed_fok_changed_the_iceberg_side_channel(self, m4_killed):
        snap = m4_killed.post_snapshots[1]
        broken = replace(m4_killed, post_snapshots=_swap(
            m4_killed.post_snapshots, 1, replace(snap, n_active=snap.n_active + 1)))
        res = check_m4_fok_dichotomy(broken, fok_oid=2)
        assert_check_fails(res, "M4")

    def test_m5_fails_when_a_split_child_takes_a_different_price(self, m5_pair):
        whole, split = m5_pair
        i = [f.passive_oid for f in split.fills].index(2)
        bad = replace(split.fills[i], price_int=P50)
        broken = replace(split, fills=_swap(split.fills, i, bad))
        res = check_m5_split_aggressor(whole, broken, whole_oid=3, split_oids=(3, 4))
        assert_check_fails(res, "M5")

    def test_m6_fails_when_relabelling_changes_an_agent_beyond_the_map(self, m6_pair):
        base, relabelled = m6_pair
        f = relabelled.fills[0]
        broken = replace(relabelled, fills=_swap(relabelled.fills, 0, replace(f, buyer_agent=8)))
        res = check_m6_ownership_blindness(base, broken, agent_map={1: 7, 2: 7})
        assert_check_fails(res, "M6")

    def test_m6_fails_when_relabelling_changes_the_trade_quantity(self, m6_pair):
        base, relabelled = m6_pair
        f = relabelled.fills[0]
        broken = replace(relabelled, fills=_swap(relabelled.fills, 0, replace(f, qty_int=9)))
        res = check_m6_ownership_blindness(base, broken, agent_map={1: 7, 2: 7})
        assert_check_fails(res, "M6")
