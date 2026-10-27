"""AUDIT tests for OMIE-REGEN-EXHAUSTION-ORDER-1 — slices regenerated in one
matching pass are stamped in the order their peaks were exhausted.

Independent post-implementation verification (audit mode, orchestrate.md
Step 5). Derived from the story card in ``agile-plan.md`` (v5.206, D1-D6,
R1-R7) and the domain reading: market-spec v2.14 §4.2.1 Rule 4 (*Slice
activation timing*: eager), OMIE Matching Algorithm §3.3 (a new slice is
made available at the trade that exhausts the shown quantity), §4.3 / §5.5
(the slice gets the next timestamp from one global, sequential queue) and
§4.4 (same price: oldest timestamp first), as confirmed in the AC (8)
domain block of ``sprint-status.md``. The locked test file was NOT read.

Strategy:

* An independent pure-Python **eager** reference (one price level, a FIFO
  queue in which an exhausted iceberg's next slice is appended to the tail
  at the exhausting trade) is replayed against the engine on seeded
  sessions mixing resting ICE / NON arrivals with NON, IOC, FOK (filled and
  killed) and incoming-ICE aggressors, on both sides. The engine's
  per-submit (passive OID, quantity) trade sequence must equal the
  reference's exactly.
* Deterministic variants with incidental constants deliberately varied
  (lesson of OQ-23 / ``P3-PERF-REGEN-MASK-1``): off-round and negative
  prices, peaks other than 1 MW, OIDs non-monotonic in arrival order and of
  mixed magnitude, both registration orders.
* Cross-price global ranking (ICEwPI slices from two different prices that
  coincide), both sides.
* Direct ``IcebergManager.regenerate_slices`` contract (card D1/D3/D4):
  stamps are a permutation of the pre-fix stamps, meta / ``n_active`` /
  ``ts_counter`` / every non-SEC column unchanged; ``prior_lob=None`` is the
  visit-order behaviour; an aggressor-side slot is never ranked; an unranked
  dispatch is stamped after every ranked one; jit + vmap under the outer
  ``enable_x64`` (R7) keeps int32 and matches the unbatched result.

Tape seconds of incoming orders are spaced 1000 s apart so that the
engine's counter-derived slice stamps never overtake the next tape second
(an incidental of the stamp clock outside this story's scope).

Device policy: CPU (conftest pins the cpu platform).
"""

from __future__ import annotations

import jax
import jax.experimental  # noqa: F401  (enable_x64 outer context, R7 / OQ-18)
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.order_validator import validate_order

# submit_order drives the module-level jitted matcher, and the direct
# regenerate_slices contract tests jit-compile wrappers — T1.1 discipline.
pytestmark = [pytest.mark.jit]

_CID = 81
BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, IOC, FOK, ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                      int(ExecRestriction.FOK), int(ExecRestriction.ICE))

_P = cst.OrderSideFeat.P.value
_Q = cst.OrderSideFeat.Q.value
_OID = cst.OrderSideFeat.OID.value
_SEC = cst.OrderSideFeat.SEC.value
_NSEC = cst.OrderSideFeat.NSEC.value
_T_PASS = cst.TradesFeat.PASS_OID.value
_T_Q = cst.TradesFeat.Q.value

_DT = 1000  # tape-second spacing between submits (see module docstring)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pc():
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def bk(pc):
    contract = Contract(
        contract_id=_CID,
        delivery_start_minutes=17 * 60,
        delivery_end_minutes=17 * 60 + 15,
        gct_minutes=17 * 60 - 60,
        status=ContractStatus.OPEN,
    )
    return ContractBook(contract, pc, OMIEMarketConfig())


def _o(pc, side, price, qty, oid, ts, er=NON, peak=0.0, ppd=0.0, agent=None):
    order = make_order(
        contract_id=_CID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=agent if agent is not None else 7 + oid % 911,
        timestamp=ts, exec_restriction=er, peak_qty=peak, ppd=ppd,
    )
    assert bool(validate_order(order, pc, True)), f"fixture bug: {order}"
    return order


def _new_rows(before, after):
    b = np.asarray(before.lob.trades)
    a = np.asarray(after.lob.trades)
    return a[np.any(a != b, axis=1)]


def _submit(bk, state, order):
    """Submit; return (state, [(passive_oid, |qty| in 0.1 MW)] in tape order)."""
    s2, _ = bk.submit_order(state, order)
    rows = _new_rows(state, s2)
    return s2, [(int(r[_T_PASS]), abs(int(r[_T_Q]))) for r in rows]


def _run(bk, state, orders):
    for x in orders:
        state, _ = bk.submit_order(state, x)
    return state


def _side_arr(lob, side):
    return np.asarray(lob.bids if side == BUY else lob.asks)


def _row(lob, side, oid):
    a = _side_arr(lob, side)
    r = a[(a[:, _OID] == oid) & (a[:, _Q] > 0)]
    assert r.shape[0] == 1, f"expected one live row for oid {oid}, got {r}"
    return r[0]


def _assert_book_invariants(state):
    """Unique (P, SEC, NSEC) among live rows per side; counter above every
    resting SEC (market-spec §5.5 unique sequential stamps; S5-A2)."""
    ts = int(state.ts_counter)
    for side in (BUY, SELL):
        a = _side_arr(state.lob, side)
        live = a[a[:, _Q] > 0]
        keys = [(int(r[_P]), int(r[_SEC]), int(r[_NSEC])) for r in live]
        assert len(keys) == len(set(keys)), f"duplicate keys side {side}: {keys}"
        if live.shape[0]:
            assert ts > int(live[:, _SEC].max()), (
                f"ts_counter {ts} not above resting SEC {live[:, _SEC].max()}")


# ---------------------------------------------------------------------------
# Independent eager reference (one price level)
# ---------------------------------------------------------------------------

class _EagerLevel:
    """FIFO queue at one price. Entries ``[oid, shown, hidden, peak]`` in
    0.1 MW units. A trade that exhausts an iceberg's shown quantity appends
    its next slice to the tail at that trade (MA §3.3, §4.3, §5.5)."""

    def __init__(self):
        self.q = []

    def avail(self):
        return sum(e[1] + e[2] for e in self.q)

    def rest(self, oid, total, peak):
        shown = min(total, peak)
        self.q.append([oid, shown, total - shown, peak])

    def take(self, qty, aggressor_peak=None):
        """Trades of an aggressor of ``qty``. ``aggressor_peak`` (0.1 MW) is
        set for an incoming ICE aggressor: it is a chain of instances of that
        size, and a trade row also ends where one incoming instance ends
        (market-spec §4.2.1 Rule 2 *Incoming iceberg* item (2), amended
        2026-10-20; OMIE-ICE-INCOMING-LADDER-1, licensed E-C oracle change).
        The split changes rows only, never which entry trades or how much."""
        trades, exhausted = [], 0
        while qty > 0 and self.q:
            e = self.q[0]
            t = min(qty, e[1])
            e[1] -= t
            qty -= t
            trades.append((e[0], t))
            if e[1] == 0:
                self.q.pop(0)
                if e[2] > 0:
                    exhausted += 1
                    nxt = min(e[3], e[2])
                    self.q.append([e[0], nxt, e[2] - nxt, e[3]])
        if aggressor_peak is not None:
            trades = _split_at_instances(trades, aggressor_peak)
        return trades, exhausted


def _split_at_instances(trades, peak):
    """Split ``(passive_oid, qty)`` rows of ONE incoming-ICE aggressor at the
    multiples of its peak, counted over its cumulative traded quantity."""
    peak = max(int(peak), 1)
    out, c = [], 0
    for oid, q in trades:
        lo, hi = c, c + q
        m = lo // peak
        while max(lo, m * peak) < hi:
            out.append((oid, min(hi, (m + 1) * peak) - max(lo, m * peak)))
            m += 1
        c = hi
    return out


def _seeded_session(bk, pc, seed, pside, price, n_steps=36):
    """Replay a seeded session against engine and reference; return the
    number of aggressor calls that regenerated >= 2 slices (non-vacuity)."""
    rng = np.random.default_rng(seed)
    aside = 1 - pside
    ref = _EagerLevel()
    state = bk.init()
    used = set()
    multi = 0
    ts = 500 + 37 * seed
    for step in range(n_steps):
        oid = int(rng.integers(2, 9_000_000))
        while oid in used:
            oid = int(rng.integers(2, 9_000_000))
        used.add(oid)
        avail = ref.avail()
        if avail < 6 or rng.random() < 0.35:
            if rng.random() < 0.8:
                peak = int(rng.integers(2, 9))
                total = peak * int(rng.integers(2, 5)) + int(rng.integers(0, peak))
                order = _o(pc, pside, price, total / 10, oid, ts, ICE, peak / 10)
                ref.rest(oid, total, peak)
            else:
                qty = int(rng.integers(1, 8))
                order = _o(pc, pside, price, qty / 10, oid, ts)
                ref.rest(oid, qty, qty)
            expected, n_ex = [], 0
            label = "rest"
        else:
            er = [NON, IOC, FOK, ICE][int(rng.integers(0, 4))]
            qty = int(rng.integers(1, min(avail, 30) + 1))
            if er == FOK and rng.random() < 0.3:
                qty = avail + int(rng.integers(1, 5))  # killed FOK
            peak = max(1, qty // 3) / 10 if er == ICE else 0.0
            order = _o(pc, aside, price, qty / 10, oid, ts, er, peak)
            if er == FOK and qty > avail:
                expected, n_ex = [], 0
            else:
                # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): an incoming ICE
                # aggressor's rows split at its instance boundaries (oracle).
                expected, n_ex = ref.take(
                    qty, max(1, qty // 3) if er == ICE else None)
            label = f"aggr er={er} qty={qty}"
        state, got = _submit(bk, state, order)
        assert got == expected, (
            f"seed {seed} side {pside} step {step} ({label}): engine trades "
            f"{got} != eager reference {expected}")
        _assert_book_invariants(state)
        multi += int(n_ex >= 2)
        ts += _DT
    return multi


# ---------------------------------------------------------------------------
# 1. Seeded sessions against the eager reference
# ---------------------------------------------------------------------------

class TestEagerReferenceSessions:
    """Every trade of every submit equals the eager-activation queue."""

    @pytest.mark.parametrize("seed", [3, 17, 2026])
    def test_sell_side_icebergs_match_eager_reference(self, bk, pc, seed):
        multi = _seeded_session(bk, pc, seed, SELL, 61.37)
        assert multi >= 2, "session did not exercise multi-slice passes"

    @pytest.mark.parametrize("seed", [5, 404])
    def test_buy_side_icebergs_match_eager_reference(self, bk, pc, seed):
        multi = _seeded_session(bk, pc, seed, BUY, -14.83)
        assert multi >= 2, "session did not exercise multi-slice passes"


# ---------------------------------------------------------------------------
# 2. Deterministic variants (varied incidental constants)
# ---------------------------------------------------------------------------

def _two_icebergs(bk, pc, pside, price, first, second, peak=0.4, total=4.0):
    """Rest ``first`` then ``second`` (both ICE, same price); consume
    ``first``'s peak so its next slice queues behind ``second``'s peak."""
    aside = 1 - pside
    s = _run(bk, bk.init(), [
        _o(pc, pside, price, total, first, 1000, ICE, peak),
        _o(pc, pside, price, total, second, 2000, ICE, peak),
    ])
    s, tr = _submit(bk, s, _o(pc, aside, price, peak, 777001, 3000, IOC))
    assert tr == [(first, int(round(peak * 10)))], tr
    return s


class TestReproducerVariants:

    @pytest.mark.parametrize("first,second", [(904417, 12), (12, 904417)])
    def test_sell_reproducer_next_fill_goes_to_earlier_exhausted(
            self, bk, pc, first, second):
        """Queue: second's peak, first's slice 2. One BUY takes both (second
        exhausted first). Next same-price BUY must fill ``second``."""
        price, peak = 73.19, 0.4
        s = _two_icebergs(bk, pc, SELL, price, first, second, peak)
        s, tr = _submit(bk, s, _o(pc, BUY, price, 2 * peak, 31, 4000))
        assert tr == [(second, 4), (first, 4)], tr
        s, tr = _submit(bk, s, _o(pc, BUY, price, peak, 32, 5000))
        assert tr == [(second, 4)], tr
        _assert_book_invariants(s)

    @pytest.mark.parametrize("first,second", [(55, 3_000_001), (3_000_001, 55)])
    def test_buy_mirror_negative_price(self, bk, pc, first, second):
        price, peak = -12.34, 0.7
        s = _two_icebergs(bk, pc, BUY, price, first, second, peak, total=3.5)
        s, tr = _submit(bk, s, _o(pc, SELL, price, 2 * peak, 31, 4000))
        assert tr == [(second, 7), (first, 7)], tr
        s, tr = _submit(bk, s, _o(pc, SELL, price, peak, 32, 5000, IOC))
        assert tr == [(second, 7)], tr

    @pytest.mark.parametrize("er", [NON, IOC, FOK, ICE])
    def test_middle_aggressor_type_does_not_matter(self, bk, pc, er):
        """The aggressor that exhausts both peaks may be NON, IOC, FOK or an
        incoming ICE (AC 4b; domain nuance iv)."""
        price, peak = 88.08, 0.3
        first, second = 640, 9
        s = _two_icebergs(bk, pc, SELL, price, first, second, peak, total=2.4)
        middle_peak = 0.2 if er == ICE else 0.0
        s, tr = _submit(bk, s, _o(pc, BUY, price, 2 * peak, 4321, 4000, er,
                                  middle_peak))
        # OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): an incoming ICE middle
        # aggressor (peak 0.2) splits its rows at its instance boundaries 0.2
        # and 0.4: (second, 3) -> 2 + 1, (first, 3) -> 1 + 2 (Rule 2 item (2)).
        want = ([(second, 2), (second, 1), (first, 1), (first, 2)] if er == ICE
                else [(second, 3), (first, 3)])
        assert sorted(tr) == sorted(want), tr
        s, tr = _submit(bk, s, _o(pc, BUY, price, peak, 4322, 5000))
        assert tr == [(second, 3)], tr

    def test_killed_fok_leaves_queue_untouched(self, bk, pc):
        """A killed FOK trades nothing, so nothing is exhausted (nuance iv)."""
        price, peak = 88.08, 0.3
        s = _two_icebergs(bk, pc, SELL, price, 640, 9, peak, total=0.9)
        before_asks = np.asarray(s.lob.asks).copy()
        s, tr = _submit(bk, s, _o(pc, BUY, price, 5.0, 4323, 4000, FOK))
        assert tr == []
        np.testing.assert_array_equal(np.asarray(s.lob.asks), before_asks)

    def test_single_aggressor_interleaves_exhaustion_order(self, bk, pc):
        """One large BUY: A1, B2, A2, B3, A3 ... (AC 3 generalised, peak
        0.6, totals that leave a partial last slice)."""
        price, peak = 45.67, 0.6
        a, b = 2_718_281, 314
        s = _run(bk, bk.init(), [
            _o(pc, SELL, price, 2.0, b, 1000, ICE, peak),   # B: 6+6+6+2
            _o(pc, SELL, price, 2.0, a, 2000, ICE, peak),   # A: 6+6+6+2
        ])
        s, tr = _submit(bk, s, _o(pc, BUY, price, peak, 1, 3000, IOC))
        assert tr == [(b, 6)]
        # Queue now A1, B2. Take everything.
        s, tr = _submit(bk, s, _o(pc, BUY, price, 3.4, 2, 4000))
        assert tr == [(a, 6), (b, 6), (a, 6), (b, 6), (a, 6), (b, 2), (a, 2)], tr
        assert int(s.n_active) == 0

    def test_three_icebergs_rotation_with_unequal_peaks(self, bk, pc):
        """Slot order X, Y, Z; queue rotated to Z, X, Y with distinct peaks;
        one BUY exhausts all three; the next slices must keep Z, X, Y."""
        price = 102.5
        x, y, z = 77, 4_400_000, 1203
        s = _run(bk, bk.init(), [
            _o(pc, SELL, price, 3.0, x, 1000, ICE, 0.5),
            _o(pc, SELL, price, 3.0, y, 2000, ICE, 0.3),
            _o(pc, SELL, price, 3.0, z, 3000, ICE, 0.2),
        ])
        s, tr = _submit(bk, s, _o(pc, BUY, price, 0.5, 1, 4000))
        assert tr == [(x, 5)]
        s, tr = _submit(bk, s, _o(pc, BUY, price, 0.3, 2, 5000))
        assert tr == [(y, 3)]
        # Queue: Z1(2), X2(5), Y2(3).
        s, tr = _submit(bk, s, _o(pc, BUY, price, 1.0, 3, 6000))
        assert tr == [(z, 2), (x, 5), (y, 3)], tr
        s, tr = _submit(bk, s, _o(pc, BUY, price, 1.0, 4, 7000, IOC))
        assert tr == [(z, 2), (x, 5), (y, 3)], tr
        _assert_book_invariants(s)


# ---------------------------------------------------------------------------
# 3. Cross-price global ranking (ICEwPI slices that coincide)
# ---------------------------------------------------------------------------

class TestCrossPriceGlobalRank:
    """Ranking spans the whole pass, not one price level (domain nuance i)."""

    @pytest.mark.parametrize("worse_first", [True, False])
    def test_sell_icewpi_slices_coincide(self, bk, pc, worse_first):
        """Better-priced B (69.85, PPD +0.50) and worse-priced A (70.10, PPD
        +0.25) both drift to 70.35; B is exhausted first whatever the
        registration (slot / SEC) order."""
        t_a, t_b = (1000, 2000) if worse_first else (2000, 1000)
        worse = _o(pc, SELL, 70.10, 2.0, 5005, t_a, ICE, 0.5, ppd=0.25)
        better = _o(pc, SELL, 69.85, 2.0, 61, t_b, ICE, 0.5, ppd=0.50)
        s = _run(bk, bk.init(), [worse, better] if worse_first
                 else [better, worse])
        s, tr = _submit(bk, s, _o(pc, BUY, 70.10, 1.0, 1, 3000))
        assert tr == [(61, 5), (5005, 5)], tr
        assert int(_row(s.lob, SELL, 61)[_P]) == int(_row(s.lob, SELL, 5005)[_P])
        s, tr = _submit(bk, s, _o(pc, BUY, 70.35, 0.5, 2, 4000, IOC))
        assert tr == [(61, 5)], tr

    def test_buy_icewpi_slices_coincide(self, bk, pc):
        s = _run(bk, bk.init(), [
            _o(pc, BUY, 40.00, 2.0, 9_999, 1000, ICE, 0.4, ppd=-0.25),
            _o(pc, BUY, 40.25, 2.0, 18, 2000, ICE, 0.4, ppd=-0.50),
        ])
        s, tr = _submit(bk, s, _o(pc, SELL, 40.00, 0.8, 1, 3000))
        assert tr == [(18, 4), (9_999, 4)], tr
        s, tr = _submit(bk, s, _o(pc, SELL, 39.75, 0.4, 2, 4000, IOC))
        assert tr == [(18, 4)], tr


# ---------------------------------------------------------------------------
# 4. Direct regenerate_slices contract (card D1 / D3 / D4)
# ---------------------------------------------------------------------------

_JIT_CACHE = {}


def _regen(bk, lob, meta, n, ts, prior=None, side="absent"):
    """Jitted regenerate_slices; ``prior=None`` -> the D1 no-keyword call;
    ``side=None`` -> aggressor_side omitted with a prior book."""
    mgr = bk._iceberg_mgr
    key = (id(mgr), prior is None, side is None, side == "absent")
    if key not in _JIT_CACHE:
        if prior is None and side == "absent":
            f = jax.jit(lambda l, m, k, t: mgr.regenerate_slices(l, m, k, t))
        elif prior is None:
            f = jax.jit(lambda l, m, k, t, a: mgr.regenerate_slices(
                l, m, k, t, prior_lob=None, aggressor_side=a))
        elif side is None:
            f = jax.jit(lambda l, m, k, t, p: mgr.regenerate_slices(
                l, m, k, t, prior_lob=p))
        else:
            f = jax.jit(lambda l, m, k, t, p, a: mgr.regenerate_slices(
                l, m, k, t, prior_lob=p, aggressor_side=a))
        _JIT_CACHE[key] = f
    f = _JIT_CACHE[key]
    if prior is None and side == "absent":
        return f(lob, meta, n, ts)
    if prior is None:
        return f(lob, meta, n, ts, jnp.int32(side))
    if side is None:
        return f(lob, meta, n, ts, prior)
    return f(lob, meta, n, ts, prior, jnp.int32(side))


def _blank(lob, side, oids):
    """Remove the live rows of ``oids`` from ``side`` (simulate consumed
    peaks): jaxob empty rows are all -1."""
    a = _side_arr(lob, side).copy()
    for oid in oids:
        a[a[:, _OID] == oid] = -1
    arr = jnp.asarray(a, jnp.int32)
    return lob._replace(bids=arr) if side == BUY else lob._replace(asks=arr)


@pytest.fixture(scope="module")
def rotated(bk, pc):
    """SELL ICEs X (slot 0), Y (slot 1), Z (slot 2) at 33.33; X's first
    peak consumed, so book priority is Y, Z, X (slot order X, Y, Z)."""
    x, y, z = 8_100_009, 46, 3_210
    s = _run(bk, bk.init(), [
        _o(pc, SELL, 33.33, 2.0, x, 1000, ICE, 0.5),
        _o(pc, SELL, 33.33, 2.0, y, 2000, ICE, 0.5),
        _o(pc, SELL, 33.33, 2.0, z, 3000, ICE, 0.5),
    ])
    s, tr = _submit(bk, s, _o(pc, BUY, 33.33, 0.5, 1, 4000, IOC))
    assert tr == [(x, 5)]
    return s, (x, y, z)


def _secs(lob, side, oids):
    return [int(_row(lob, side, o)[_SEC]) for o in oids]


class TestDirectRegenContract:

    def test_ranked_stamps_follow_prior_priority(self, bk, rotated):
        s, (x, y, z) = rotated
        cur = _blank(s.lob, SELL, (x, y, z))
        tp = int(s.ts_counter)
        out = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter,
                     prior=s.lob, side=BUY)
        assert _secs(out[0], SELL, (y, z, x)) == [tp, tp + 1, tp + 2]

    def test_prior_none_is_visit_order(self, bk, rotated):
        s, (x, y, z) = rotated
        cur = _blank(s.lob, SELL, (x, y, z))
        tp = int(s.ts_counter)
        out = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter)
        assert _secs(out[0], SELL, (x, y, z)) == [tp, tp + 1, tp + 2]
        # aggressor_side without a prior book changes nothing (D1 static arm)
        out2 = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter,
                      prior=None, side=BUY)
        for a_, b_ in zip(jax.tree_util.tree_leaves(out),
                          jax.tree_util.tree_leaves(out2)):
            np.testing.assert_array_equal(np.asarray(a_), np.asarray(b_))

    def test_only_stamps_are_permuted(self, bk, rotated):
        """D4/R2: meta, n_active, ts_counter, trades, bids and every non-SEC
        ask column identical to the visit-order call; SEC multiset equal."""
        s, (x, y, z) = rotated
        cur = _blank(s.lob, SELL, (x, y, z))
        base = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter)
        new = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter,
                     prior=s.lob, side=BUY)
        np.testing.assert_array_equal(np.asarray(base[1]), np.asarray(new[1]))
        assert int(base[2]) == int(new[2])
        assert int(base[3]) == int(new[3]) == int(s.ts_counter) + 3
        np.testing.assert_array_equal(np.asarray(base[0].bids),
                                      np.asarray(new[0].bids))
        np.testing.assert_array_equal(np.asarray(base[0].trades),
                                      np.asarray(new[0].trades))
        ab, an = np.asarray(base[0].asks), np.asarray(new[0].asks)
        cols = [c for c in range(ab.shape[1]) if c != _SEC]
        np.testing.assert_array_equal(ab[:, cols], an[:, cols])
        assert sorted(ab[:, _SEC].tolist()) == sorted(an[:, _SEC].tolist())

    def test_aggressor_side_slots_never_ranked(self, bk, rotated):
        s, (x, y, z) = rotated
        cur = _blank(s.lob, SELL, (x, y, z))
        base = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter)
        same = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter,
                      prior=s.lob, side=SELL)
        np.testing.assert_array_equal(np.asarray(base[0].asks),
                                      np.asarray(same[0].asks))

    def test_unranked_dispatch_stamped_after_ranked(self, bk, rotated):
        """Z's peak is absent from the prior book too (e.g. an evicted row):
        Z is unranked and goes after every ranked slot (D4)."""
        s, (x, y, z) = rotated
        cur = _blank(s.lob, SELL, (x, y, z))
        prior = _blank(s.lob, SELL, (z,))
        tp = int(s.ts_counter)
        out = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter,
                     prior=prior, side=BUY)
        assert _secs(out[0], SELL, (y, x, z)) == [tp, tp + 1, tp + 2]
        assert int(out[3]) == tp + 3

    def test_unexhausted_slot_not_dispatched_and_stamps_contiguous(
            self, bk, rotated):
        """Only X and Z exhausted; Y keeps its row; stamps tp, tp+1 in
        prior priority (Z older than X's slice 2)."""
        s, (x, y, z) = rotated
        y_row = _row(s.lob, SELL, y)
        cur = _blank(s.lob, SELL, (x, z))
        tp = int(s.ts_counter)
        out = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter,
                     prior=s.lob, side=BUY)
        assert _secs(out[0], SELL, (z, x)) == [tp, tp + 1]
        np.testing.assert_array_equal(_row(out[0], SELL, y), y_row)
        assert int(out[3]) == tp + 2

    def test_global_rank_price_before_time(self, bk, pc):
        """Older peak at a worse price ranks after a younger peak at the
        better price (key (+P, SEC, NSEC) for SELL, (-P, ...) for BUY)."""
        s = _run(bk, bk.init(), [
            _o(pc, SELL, 50.02, 2.0, 701, 1000, ICE, 0.5),
            _o(pc, SELL, 50.00, 2.0, 33, 2000, ICE, 0.5),
            _o(pc, BUY, 49.00, 2.0, 702, 3000, ICE, 0.5),
            _o(pc, BUY, 49.01, 2.0, 34, 4000, ICE, 0.5),
        ])
        tp = int(s.ts_counter)
        cur = _blank(_blank(s.lob, SELL, (701, 33)), BUY, (702, 34))
        o_s = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter,
                     prior=s.lob, side=BUY)
        assert _secs(o_s[0], SELL, (33, 701)) == [tp, tp + 1]
        assert sorted(_secs(o_s[0], BUY, (702, 34))) == [tp + 2, tp + 3]
        o_b = _regen(bk, cur, s.iceberg_meta, s.n_active, s.ts_counter,
                     prior=s.lob, side=SELL)
        assert _secs(o_b[0], BUY, (34, 702)) == [tp, tp + 1]

    def test_jit_vmap_under_x64_matches_unbatched(self, bk, rotated):
        """R7: under the outer enable_x64, jit(vmap(regen)) keeps int32 and
        reproduces the per-example results."""
        s, (x, y, z) = rotated
        cur_a = _blank(s.lob, SELL, (x, y, z))
        cur_b = _blank(s.lob, SELL, (x, z))
        ref_a = _regen(bk, cur_a, s.iceberg_meta, s.n_active, s.ts_counter,
                       prior=s.lob, side=BUY)
        ref_b = _regen(bk, cur_b, s.iceberg_meta, s.n_active, s.ts_counter,
                       prior=s.lob, side=BUY)
        stack = lambda a, b: jax.tree_util.tree_map(  # noqa: E731
            lambda u, v: jnp.stack([u, v]), a, b)
        mgr = bk._iceberg_mgr
        prior_flag = bool(jax.config.jax_enable_x64)
        try:
            with jax.experimental.enable_x64():
                f = jax.jit(jax.vmap(
                    lambda l, m, k, t, p, a: mgr.regenerate_slices(
                        l, m, k, t, prior_lob=p, aggressor_side=a)))
                out = f(stack(cur_a, cur_b),
                        stack(s.iceberg_meta, s.iceberg_meta),
                        stack(s.n_active, s.n_active),
                        stack(s.ts_counter, s.ts_counter),
                        stack(s.lob, s.lob),
                        jnp.asarray([BUY, BUY], jnp.int32))
                out = jax.device_get(out)
        finally:
            jax.config.update("jax_enable_x64", prior_flag)
        assert np.asarray(out[3]).dtype == np.int32
        assert np.asarray(out[0].asks).dtype == np.int32
        for i, ref in enumerate((ref_a, ref_b)):
            np.testing.assert_array_equal(np.asarray(out[0].asks)[i],
                                          np.asarray(ref[0].asks))
            np.testing.assert_array_equal(np.asarray(out[1])[i],
                                          np.asarray(ref[1]))
            assert int(np.asarray(out[3])[i]) == int(ref[3])
