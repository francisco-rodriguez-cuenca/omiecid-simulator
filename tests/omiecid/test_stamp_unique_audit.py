"""AUDIT tests for OMIE-STAMP-UNIQUE-1 — stored entry stamps are unique among
same-price, same-second resting orders (NSEC rank on each resting add).

Independent post-implementation verification (audit mode, orchestrate.md
Step 5). Derived from the story card in ``agile-plan.md`` (v5.198, incl. the
S5-A1..A3 Step-2 amendment) and ``market-spec.md`` §5.2 ("All timestamps ...
are assigned from a single global queue (unique, sequential)"; partial
execution does NOT assign a new timestamp; modification DOES) and §6.1
(price-time priority). The locked test file was NOT read.

Incidental constants are deliberately varied (lesson of OQ-23): both sides,
OIDs from 2 to 7 digits and NON-monotonic in arrival order (so a "lowest OID
first" or "lowest slot first" tie-break cannot pass by accident), off-round
prices, tape seconds both above and below the book counter.

Book rows are read through the jaxob row layout ``[P, Q, OID, TID, SEC,
NSEC]`` (``OrderSideFeat``); empty rows are all -1. Who fills is read from
the trade tape's passive-OID column.

Device policy: CPU (conftest pins the cpu platform).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, _rank_nsec
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.order_validator import validate_order

# submit_order drives the module-level jitted matcher; one test jit-compiles
# ``_rank_nsec`` directly — T1.1 jit-marker discipline.
pytestmark = [pytest.mark.jit]

_CID = 58
BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, IOC, FOK, ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                      int(ExecRestriction.FOK), int(ExecRestriction.ICE))

_P = cst.OrderSideFeat.P.value
_Q = cst.OrderSideFeat.Q.value
_OID = cst.OrderSideFeat.OID.value
_SEC = cst.OrderSideFeat.SEC.value
_NSEC = cst.OrderSideFeat.NSEC.value
_T_PASS = cst.TradesFeat.PASS_OID.value


def _pint(price_eur: float) -> int:
    """Method-A encoding, re-derived from the spec defaults (tick 0.01,
    floor -9999.00): ticks above the floor."""
    return int(round(price_eur * 100)) + 999900


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
        delivery_start_minutes=14 * 60 + 45,
        delivery_end_minutes=15 * 60,
        gct_minutes=14 * 60 + 45 - 60,
        status=ContractStatus.OPEN,
    )
    return ContractBook(contract, pc, OMIEMarketConfig())


def _o(pc, side, price, qty, oid, ts, er=NON, peak=0.0):
    order = make_order(
        contract_id=_CID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=1000 + (oid % 977), timestamp=ts,
        exec_restriction=er, peak_qty=peak,
    )
    assert bool(validate_order(order, pc, True)), f"fixture bug: {order}"
    return order


def _side(state, side):
    return np.asarray(state.lob.bids if side == BUY else state.lob.asks)


def _row(state, side, oid):
    a = _side(state, side)
    r = a[a[:, _OID] == oid]
    assert r.shape[0] == 1, f"expected one row for oid {oid}, got {r}"
    return r[0]


def _stamp(state, side, oid):
    r = _row(state, side, oid)
    return int(r[_SEC]), int(r[_NSEC])


def _run(bk, state, orders):
    for x in orders:
        state, _ = bk.submit_order(state, x)
    return state


def _passive_oids(before, after):
    """Passive OIDs of trade rows added by one call, in tape-row order."""
    b = np.asarray(before.lob.trades)
    a = np.asarray(after.lob.trades)
    changed = np.any(a != b, axis=1)
    return [int(x) for x in a[changed][:, _T_PASS]]


def _take(bk, pc, state, side, price, qty, oid, ts):
    """IOC take (never rests); returns (state, passive OIDs it hit)."""
    s2, _ = bk.submit_order(state, _o(pc, side, price, qty, oid, ts, er=IOC))
    return s2, _passive_oids(state, s2)


def _assert_unique_keys(state):
    for side in (BUY, SELL):
        a = _side(state, side)
        live = a[a[:, _Q] > 0]
        keys = [(int(r[_P]), int(r[_SEC]), int(r[_NSEC])) for r in live]
        assert len(keys) == len(set(keys)), (
            f"duplicate (P, SEC, NSEC) on side {side}: {sorted(keys)}")


# ---------------------------------------------------------------------------
# 1. Pure helper semantics (_rank_nsec), on hand-built books
# ---------------------------------------------------------------------------

def _book(bk, bid_rows=(), ask_rows=()):
    lob = bk.init().lob
    bids = np.asarray(lob.bids).copy()
    asks = np.asarray(lob.asks).copy()
    for i, r in enumerate(bid_rows):
        bids[i] = r
    for i, r in enumerate(ask_rows):
        asks[i] = r
    return lob._replace(bids=jnp.asarray(bids), asks=jnp.asarray(asks))


class TestRankNsecHelper:
    P = _pint(61.37)

    def test_max_plus_one_not_count_of_peers(self, bk):
        """Peers at NSEC 0 and 5 (a gap left by cancels): the new entry gets
        6 = max + 1, not 2 = count; uniqueness must survive gaps."""
        P = self.P
        lob = _book(bk, bid_rows=[
            [P, 10, 31, 31, 400, 0],
            [P, 10, 4471, 4471, 400, 5],
            [P, 10, 902, 902, 400, 0],   # the order being ranked
        ])
        out = _rank_nsec(lob, 902, BUY, P, 400)
        b = np.asarray(out.bids)
        assert int(b[2, _NSEC]) == 6
        assert np.array_equal(b[:2], np.asarray(lob.bids)[:2])
        assert np.array_equal(np.asarray(out.asks), np.asarray(lob.asks))

    @pytest.mark.parametrize("label,peer", [
        ("other_price", [_pint(61.36), 10, 31, 31, 400, 3]),
        ("other_second", [_pint(61.37), 10, 31, 31, 399, 3]),
        ("dead_row_q0", [_pint(61.37), 0, 31, 31, 400, 3]),
    ])
    def test_non_peers_do_not_rank(self, bk, label, peer):
        """Only live rows (Q > 0) in the same second are peers; anything else
        leaves the entry at NSEC 0 (byte no-op).

        Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E1, licence (3), the
        ``other_price`` case only): uniqueness is per contract, across
        prices, so a live row at another price in the same second IS a peer;
        the entry ranks above it and nothing else changes."""
        P = self.P
        lob = _book(bk, ask_rows=[peer, [P, 10, 902, 902, 400, 0]])
        out = _rank_nsec(lob, 902, SELL, P, 400)
        if label == "other_price":
            a, a0 = np.asarray(out.asks), np.asarray(lob.asks)
            assert np.array_equal(a[0], a0[0]), "the peer row is never written"
            cols = [c for c in range(a.shape[1]) if c != _NSEC]
            assert np.array_equal(a[:, cols], a0[:, cols])
            assert int(a[1, _NSEC]) > int(a0[0, _NSEC])
            assert np.array_equal(a[2:], a0[2:])
            assert np.array_equal(np.asarray(out.bids), np.asarray(lob.bids))
            return
        assert np.array_equal(np.asarray(out.asks), np.asarray(lob.asks)), label

    def test_peer_on_opposite_side_is_ignored(self, bk):
        """Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E1, licence (4)): the
        stamp is unique per contract across sides, so a live same-second row
        on the opposite side IS a peer: the SELL entry ranks above it (NSEC
        > 7); the BUY row and every other cell are untouched."""
        P = self.P
        lob = _book(bk, bid_rows=[[P, 10, 31, 31, 400, 7]],
                    ask_rows=[[P, 10, 902, 902, 400, 0]])
        out = _rank_nsec(lob, 902, SELL, P, 400)
        a, a0 = np.asarray(out.asks), np.asarray(lob.asks)
        cols = [c for c in range(a.shape[1]) if c != _NSEC]
        assert np.array_equal(a[:, cols], a0[:, cols])
        assert int(a[0, _NSEC]) > 7
        assert np.array_equal(a[1:], a0[1:])
        assert np.array_equal(np.asarray(out.bids), np.asarray(lob.bids))

    def test_order_not_resting_is_byte_noop(self, bk):
        """Miss: the oid is not on the book (it fully filled / was IOC).
        Peers must NOT be touched."""
        P = self.P
        lob = _book(bk, bid_rows=[[P, 10, 31, 31, 400, 0],
                                  [P, 10, 44, 44, 400, 1]])
        out = _rank_nsec(lob, 902, BUY, P, 400)
        assert np.array_equal(np.asarray(out.bids), np.asarray(lob.bids))

    def test_sec_and_other_columns_never_written(self, bk):
        P = self.P
        lob = _book(bk, bid_rows=[[P, 10, 31, 31, 400, 2],
                                  [P, 30, 5, 5, 400, 0]])
        out = _rank_nsec(lob, 5, BUY, P, 400)
        b, b0 = np.asarray(out.bids), np.asarray(lob.bids)
        cols = [c for c in range(b.shape[1]) if c != _NSEC]
        assert np.array_equal(b[:, cols], b0[:, cols])
        assert int(b[1, _NSEC]) == 3

    @pytest.mark.jit
    def test_traced_side_matches_eager_on_both_sides(self, bk):
        """``side`` is traced inside the matcher; the jitted helper must give
        the eager result for BUY and SELL under one compile."""
        P = self.P
        rows = [[P, 10, 31, 31, 400, 4], [P, 10, 902, 902, 400, 0]]
        lob = _book(bk, bid_rows=rows, ask_rows=rows)
        f = jax.jit(_rank_nsec)
        for side in (BUY, SELL):
            e = _rank_nsec(lob, 902, side, P, 400)
            j = f(lob, jnp.int32(902), jnp.int32(side), jnp.int32(P),
                  jnp.int32(400))
            assert np.array_equal(np.asarray(e.bids), np.asarray(j.bids))
            assert np.array_equal(np.asarray(e.asks), np.asarray(j.asks))
        j = f(lob, jnp.int32(902), jnp.int32(BUY), jnp.int32(P), jnp.int32(400))
        assert int(np.asarray(j.bids)[1, _NSEC]) == 5
        assert int(np.asarray(j.asks)[1, _NSEC]) == 0


# ---------------------------------------------------------------------------
# 2. No-fill rests: processing order, not OID order, not slot order
# ---------------------------------------------------------------------------

class TestNoFillRestRanking:
    @pytest.mark.parametrize("side", [BUY, SELL])
    def test_four_same_second_rests_rank_in_processing_order(self, bk, pc, side):
        price = 37.83 if side == BUY else 81.09
        oids = [918273, 12, 5004, 77]          # non-monotonic, 2..6 digits
        s = _run(bk, bk.init(),
                 [_o(pc, side, price, 0.3 + 0.1 * i, oid, 913)
                  for i, oid in enumerate(oids)])
        assert [_stamp(s, side, o) for o in oids] == [
            (913, 0), (913, 1), (913, 2), (913, 3)]
        _assert_unique_keys(s)

    @pytest.mark.parametrize("side", [BUY, SELL])
    def test_takes_fill_same_second_peers_in_arrival_order(self, bk, pc, side):
        """§6.1: equal price and equal second -> earlier arrival first."""
        price = 37.83 if side == BUY else 81.09
        oids = [918273, 12, 5004]
        qtys = [0.4, 0.7, 1.1]
        s = _run(bk, bk.init(), [_o(pc, side, price, q, oid, 913)
                                 for q, oid in zip(qtys, oids)])
        opp = SELL if side == BUY else BUY
        hit = []
        for k, q in enumerate(qtys):
            s, h = _take(bk, pc, s, opp, price, q, 60000 + k, 913)
            hit += h
        assert hit == oids

    def test_distinct_seconds_or_prices_stay_at_nsec_zero(self, bk, pc):
        """No same-second peer -> NSEC stays 0 (byte no-op path).

        Re-baselined by OMIE-SLICE-STAMP-TAPE-SEC-1 (E1, licence (5)): peers
        are per contract, across prices and sides, so 303 (other price) and
        304 (other side) rank behind 302 in the same second, in processing
        order (ordering, not literal NSEC values)."""
        s = _run(bk, bk.init(), [
            _o(pc, SELL, 52.01, 1.0, 301, 40),
            _o(pc, SELL, 52.01, 1.0, 302, 41),   # same price, next second
            _o(pc, SELL, 52.02, 1.0, 303, 41),   # same second, other price
            _o(pc, BUY, 51.99, 1.0, 304, 41),    # same second, other side
        ])
        assert _stamp(s, SELL, 301) == (40, 0)
        assert _stamp(s, SELL, 302) == (41, 0)
        assert _stamp(s, SELL, 303)[0] == 41 and _stamp(s, BUY, 304)[0] == 41
        assert _stamp(s, SELL, 302) < _stamp(s, SELL, 303) < _stamp(s, BUY, 304)

    def test_tape_second_below_counter_still_ranks(self, bk, pc):
        """A tape second already passed by ts_counter (counter ran ahead) is
        still ranked by processing order — the rank is independent of the
        counter."""
        s = _run(bk, bk.init(), [_o(pc, BUY, 12.34, 1.0, 9, 500)])
        assert int(s.ts_counter) > 300
        s = _run(bk, s, [_o(pc, BUY, 12.35, 1.0, 88, 300),
                         _o(pc, BUY, 12.35, 1.0, 7, 300)])
        assert _stamp(s, BUY, 88) == (300, 0)
        assert _stamp(s, BUY, 7) == (300, 1)


# ---------------------------------------------------------------------------
# 3. Slot reuse after a cancel must not let a later arrival jump the queue
# ---------------------------------------------------------------------------

class TestSlotReuse:
    def test_buy_side_probe_a_mirror(self, bk, pc):
        """Mirror of C05 probe A on the BUY side with different OIDs/prices:
        decoy takes bid slot 0; SELL r rests; BUY aggressor fills r and
        rests its residual; decoy cancelled; BUY interloper at the same
        price/second lands in the freed slot. The residual must fill first."""
        s = _run(bk, bk.init(), [
            _o(pc, BUY, 3.17, 1.0, 640017, 7),             # decoy, slot 0
            _o(pc, SELL, 44.44, 2.7, 23, 8),               # r
            _o(pc, BUY, 44.49, 5.2, 31337, 9),             # fills 2.7, rests 2.5
        ])
        assert _row(s, BUY, 31337)[_Q] == 25
        s = bk.cancel_order(s, 640017, BUY)
        s = _run(bk, s, [_o(pc, BUY, 44.49, 16.6, 2, 9)])  # interloper
        assert _stamp(s, BUY, 31337) == (9, 0)
        assert _stamp(s, BUY, 2) == (9, 1)
        s, hit = _take(bk, pc, s, SELL, 44.49, 2.5, 4242, 9)
        assert hit == [31337]
        assert int(_row(s, BUY, 2)[_Q]) == 166

    def test_cancel_of_last_peer_then_new_arrival_still_ranks_last(self, bk, pc):
        """Peers 0,1,2; the NSEC-2 peer is cancelled; a later same-second
        arrival must still rank after the survivors (any value > 1 and
        unique), and fill last."""
        s = _run(bk, bk.init(), [
            _o(pc, SELL, 70.07, 0.5, 501, 66),
            _o(pc, SELL, 70.07, 0.6, 50, 66),
            _o(pc, SELL, 70.07, 0.7, 5, 66),
        ])
        s = bk.cancel_order(s, 5, SELL)
        s = _run(bk, s, [_o(pc, SELL, 70.07, 0.8, 5000000, 66)])
        assert _stamp(s, SELL, 5000000)[1] > _stamp(s, SELL, 50)[1]
        _assert_unique_keys(s)
        hit = []
        for k, q in enumerate([0.5, 0.6, 0.8]):
            s, h = _take(bk, pc, s, BUY, 70.07, q, 900 + k, 66)
            hit += h
        assert hit == [501, 50, 5000000]

    def test_cancel_of_middle_peer_keeps_order(self, bk, pc):
        s = _run(bk, bk.init(), [
            _o(pc, BUY, 9.99, 0.5, 71, 3),
            _o(pc, BUY, 9.99, 0.6, 72, 3),
            _o(pc, BUY, 9.99, 0.7, 73, 3),
        ])
        s = bk.cancel_order(s, 72, BUY)
        s = _run(bk, s, [_o(pc, BUY, 9.99, 0.8, 70, 3)])  # reuses slot 1
        _assert_unique_keys(s)
        hit = []
        for k, q in enumerate([0.5, 0.7, 0.8]):
            s, h = _take(bk, pc, s, SELL, 9.99, q, 800 + k, 3)
            hit += h
        assert hit == [71, 73, 70]


# ---------------------------------------------------------------------------
# 4. §5.2 timestamp semantics around the rank
# ---------------------------------------------------------------------------

class TestTimestampSemantics:
    def test_partially_executed_resting_order_keeps_its_rank(self, bk, pc):
        """§5.2: partial execution of a RESTING order is not a modification;
        it keeps (SEC, NSEC) and its queue position."""
        s = _run(bk, bk.init(), [
            _o(pc, SELL, 33.33, 2.0, 404, 120),
            _o(pc, SELL, 33.33, 2.0, 99, 120),
        ])
        s, hit = _take(bk, pc, s, BUY, 33.33, 0.9, 1, 121)
        assert hit == [404]
        assert _stamp(s, SELL, 404) == (120, 0)
        assert _stamp(s, SELL, 99) == (120, 1)
        s, hit = _take(bk, pc, s, BUY, 33.33, 1.1, 2, 122)
        assert hit == [404]

    def test_modification_in_same_second_loses_priority(self, bk, pc):
        """§5.2/§5.3: modification assigns a new timestamp. Re-entered in
        the same second, the modified order ranks behind its former
        follower."""
        s = _run(bk, bk.init(), [
            _o(pc, BUY, 28.41, 1.0, 3141, 77),
            _o(pc, BUY, 28.41, 1.0, 2718, 77),
        ])
        s, _ = bk.modify_order(s, 3141, BUY,
                               _o(pc, BUY, 28.41, 1.5, 3141, 77))
        assert _stamp(s, BUY, 2718) == (77, 1)
        assert _stamp(s, BUY, 3141)[1] > 1
        s, hit = _take(bk, pc, s, SELL, 28.41, 1.0, 5, 77)
        assert hit == [2718]

    def test_ioc_and_fok_never_rest_and_never_rerank_peers(self, bk, pc):
        """An IOC/FOK at a peer's (price, second) that does not rest must not
        touch any resting NSEC."""
        s = _run(bk, bk.init(), [
            _o(pc, BUY, 45.06, 1.0, 61, 200),
            _o(pc, BUY, 45.06, 1.0, 62, 200),
            _o(pc, SELL, 45.10, 0.5, 63, 200),
        ])
        before = _side(s, BUY).copy()
        s = _run(bk, s, [
            _o(pc, BUY, 45.06, 0.4, 64, 200, er=IOC),   # no cross, discarded
            _o(pc, BUY, 45.10, 0.9, 65, 200, er=IOC),   # fills 0.5, rest killed
            _o(pc, BUY, 45.06, 0.4, 66, 200, er=FOK),   # killed
        ])
        assert np.array_equal(_side(s, BUY), before)

    def test_incoming_ice_first_slice_ranked_behind_non_peer(self, bk, pc):
        """Site (a'): an incoming iceberg's first slice resting at a (P, SEC)
        already held by a NON peer is ranked behind it."""
        s = _run(bk, bk.init(), [
            _o(pc, SELL, 66.61, 0.8, 7007, 250),
            _o(pc, SELL, 66.61, 3.0, 12, 250, er=ICE, peak=1.0),
        ])
        assert _stamp(s, SELL, 7007) == (250, 0)
        assert _stamp(s, SELL, 12) == (250, 1)
        s, hit = _take(bk, pc, s, BUY, 66.61, 0.8, 1, 250)
        assert hit == [7007]

    def test_non_ranked_behind_earlier_incoming_ice_slice(self, bk, pc):
        """Reverse order: ICE slice rests first at (P, T), the NON at (P, T)
        ranks behind it."""
        s = _run(bk, bk.init(), [
            _o(pc, BUY, 19.73, 2.0, 845, 260, er=ICE, peak=0.6),
            _o(pc, BUY, 19.73, 0.6, 3, 260),
        ])
        assert _stamp(s, BUY, 845) == (260, 0)
        assert _stamp(s, BUY, 3) == (260, 1)
        s, hit = _take(bk, pc, s, SELL, 19.73, 0.6, 9, 260)
        assert hit == [845]

    def test_ice_that_sweeps_then_rests_is_ranked_for_later_peer(self, bk, pc):
        """Incoming ICE sweeps a counter order, rests its first slice at
        (order.timestamp, 0); a NON at the same (P, T) ranks behind it."""
        s = _run(bk, bk.init(), [
            _o(pc, SELL, 55.50, 0.7, 101, 299),
            _o(pc, BUY, 55.55, 4.0, 8080, 300, er=ICE, peak=1.0),
        ])
        assert _stamp(s, BUY, 8080) == (300, 0)
        s = _run(bk, s, [_o(pc, BUY, 55.55, 1.0, 17, 300)])
        assert _stamp(s, BUY, 17) == (300, 1)

    def test_tape_order_at_counter_second_ranks_behind_regen_slice(self, bk, pc):
        """Probe S5-P2 on the BUY side. Re-baselined by
        OMIE-SLICE-STAMP-TAPE-SEC-1 (E2, licence (12)): the regenerated slice
        is stamped at its activation event, the second of the SELL that
        exhausted slice 1 (101), not from the counter; a later tape order at
        that very second ranks behind it by (SEC, NSEC), and the slice keeps
        its stamp."""
        s = _run(bk, bk.init(), [
            _o(pc, BUY, 47.37, 3.0, 4021, 100, er=ICE, peak=1.0),
            _o(pc, SELL, 47.37, 1.0, 77, 101),             # fills slice 1
        ])
        slice_key = _stamp(s, BUY, 4021)
        slice_sec = slice_key[0]
        assert slice_sec == 101                           # activation second
        assert int(s.ts_counter) > 101                    # counter ran ahead
        s = _run(bk, s, [_o(pc, BUY, 47.37, 2.0, 909, slice_sec)])
        assert _stamp(s, BUY, 909)[0] == slice_sec
        assert _stamp(s, BUY, 909) > slice_key
        assert _stamp(s, BUY, 4021) == slice_key
        s, hit = _take(bk, pc, s, SELL, 47.37, 1.0, 6, slice_sec)
        assert hit == [4021]


# ---------------------------------------------------------------------------
# 5. Session-level invariants against an independent reference queue
# ---------------------------------------------------------------------------

def _random_non_session(rng, n):
    """Non-crossing NON adds (bids <= 40.00 < 60.00 <= asks) on a handful of
    price levels with heavy same-second bunching, interleaved with cancels."""
    ops, live, oid, t = [], [], 0, 1000
    bid_px = [39.97, 39.98, 40.00]
    ask_px = [60.00, 60.01, 60.04]
    for _ in range(n):
        if rng.random() < 0.3:
            t += int(rng.integers(1, 3))
        if live and rng.random() < 0.25:
            k = int(rng.integers(len(live)))
            ops.append(("cancel",) + live.pop(k))
            continue
        oid += int(rng.integers(1, 9000))
        side = BUY if rng.random() < 0.5 else SELL
        px = float(rng.choice(bid_px if side == BUY else ask_px))
        qty = round(0.1 * int(rng.integers(1, 30)), 1)
        ops.append(("add", oid, side, px, qty, t))
        live.append((oid, side, px, qty, t))
    return ops, t


@pytest.mark.parametrize("seed", [3, 17, 2026])
def test_random_session_unique_keys_and_fifo_drain(bk, pc, seed):
    """After every op no two live rows on a side share (P, SEC, NSEC); and
    draining each side with exact-size IOC takes fills orders in the
    reference price-time order (best price, earlier second, earlier
    arrival), regardless of slot reuse after cancels."""
    rng = np.random.default_rng(seed)
    ops, t_last = _random_non_session(rng, 40)
    s = bk.init()
    ref = {}   # oid -> (side, px, qty, sec, arrival)
    for arr, op in enumerate(ops):
        if op[0] == "add":
            _, oid, side, px, qty, t = op
            s, _ = bk.submit_order(s, _o(pc, side, px, qty, oid, t))
            ref[oid] = (side, px, qty, t, arr)
        else:
            oid, side = op[1], op[2]
            s = bk.cancel_order(s, oid, side)
            ref.pop(oid)
        _assert_unique_keys(s)

    take_oid = 10_000_000
    for side, opp in ((BUY, SELL), (SELL, BUY)):
        sgn = -1 if side == BUY else 1
        expected = sorted((o for o, v in ref.items() if v[0] == side),
                          key=lambda o: (sgn * ref[o][1], ref[o][3], ref[o][4]))
        got = []
        for o in expected:
            take_oid += 1
            s, h = _take(bk, pc, s, opp, ref[o][1], ref[o][2], take_oid,
                         t_last)
            got += h
        assert got == expected, f"seed {seed} side {side}"


def test_c05_shape_store_keys_without_cancel(bk, pc):
    """C05 'same_second' shape: residual (fill-then-rest) and a later
    interloper at the same (P, second) store distinct keys, residual first,
    even without slot reuse."""
    s = _run(bk, bk.init(), [
        _o(pc, BUY, 100.45, 2.7, 1, 2),
        _o(pc, SELL, 100.44, 5.2, 20, 3),          # fills 2.7, rests 2.5
        _o(pc, SELL, 100.44, 16.6, 300, 3),        # interloper
    ])
    assert _stamp(s, SELL, 20) == (3, 0)
    assert _stamp(s, SELL, 300) == (3, 1)
