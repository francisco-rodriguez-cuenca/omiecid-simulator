"""Audit tests for OMIE-JAXOB-LOOKUP-MISS-1 (S2, D3).

Independent post-implementation verification (Step 5). Derived from the
story card in docs/omiecid/agile-plan.md ("Certification fix pass") and from
market-spec cancel-by-id semantics (a cancel removes exactly the named
order, nothing else; I2 conservation; §6.1 an iceberg with hidden volume
keeps a visible slice).

Contract under audit: ``get_order_by_id``, ``get_order_by_tid`` and
``get_order_by_id_and_price`` return ``NEGATIVE_RETURN_ID * ones(6)``
(int32) on a miss -- never the side's last physical row -- and return the
first matching row on a hit.

Layers probed:
  A. jaxob primitives on synthetic side arrays (shapes, duplicates, tiny
     sides, traced ids under an outer jit, vmap over mixed hit/miss).
  B. ``ContractBook`` paths that reach the lookups on a side whose last
     physical row is occupied (book_depth = 100 resting orders):
     Python ``get_order`` / ``cancel_order``, the jitted NON / IOC / FOK
     post-fill cancel-by-id of the aggressor, and iceberg regeneration.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob import JaxOrderBookArrays as job
from omiecid_simulator.jaxob.jaxob_constants import NEGATIVE_RETURN_ID, EMPTY_SLOT
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.jax_constants import SENTINEL_ORDER_ID
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order


SENTINEL_ROW = np.full((6,), NEGATIVE_RETURN_ID, dtype=np.int32)
DEPTH = 100


# ---------------------------------------------------------------------------
# Synthetic side-array helpers (layout: [price, qty, oid, tid, sec, nsec])
# ---------------------------------------------------------------------------

def _empty_side(n=DEPTH):
    return np.full((n, 6), EMPTY_SLOT, dtype=np.int32)


def _row(price, qty, oid, tid, sec=1, nsec=0):
    return np.array([price, qty, oid, tid, sec, nsec], dtype=np.int32)


def _full_side(n=DEPTH, oid_base=1000, tid_base=5000):
    side = _empty_side(n)
    for i in range(n):
        side[i] = _row(10_000 + i, 10 + i, oid_base + i, tid_base + i, sec=i)
    return side


def _np(x):
    return np.asarray(x)


# ===========================================================================
# A. jaxob primitives
# ===========================================================================

class TestMissOnFullSide:
    """Every one of the 100 rows occupied; the queried key is nowhere."""

    def test_by_id_miss_full_side(self):
        side = jnp.asarray(_full_side())
        out = _np(job.get_order_by_id(side, 4242))
        np.testing.assert_array_equal(out, SENTINEL_ROW)

    def test_by_tid_miss_full_side(self):
        side = jnp.asarray(_full_side())
        out = _np(job.get_order_by_tid(side, 4242))
        np.testing.assert_array_equal(out, SENTINEL_ROW)

    def test_by_id_and_price_miss_full_side(self):
        side = jnp.asarray(_full_side())
        out = _np(job.get_order_by_id_and_price(side, 4242, 10_050))
        np.testing.assert_array_equal(out, SENTINEL_ROW)

    def test_by_id_and_price_price_present_oid_absent(self):
        """Row 99's price matches but its oid does not -> miss."""
        side = _full_side()
        out = _np(job.get_order_by_id_and_price(
            jnp.asarray(side), 4242, int(side[99, 0])))
        np.testing.assert_array_equal(out, SENTINEL_ROW)

    def test_by_id_and_price_oid_at_row99_other_price(self):
        """Row 99 holds the oid but at a different price -> miss."""
        side = _full_side()
        out = _np(job.get_order_by_id_and_price(
            jnp.asarray(side), int(side[99, 2]), int(side[99, 0]) + 1))
        np.testing.assert_array_equal(out, SENTINEL_ROW)

    def test_by_id_does_not_match_on_tid_column(self):
        """A key equal to a tid (not an oid) must miss in get_order_by_id."""
        side = _full_side()
        out = _np(job.get_order_by_id(jnp.asarray(side), int(side[99, 3])))
        np.testing.assert_array_equal(out, SENTINEL_ROW)

    def test_by_tid_does_not_match_on_oid_column(self):
        side = _full_side()
        out = _np(job.get_order_by_tid(jnp.asarray(side), int(side[99, 2])))
        np.testing.assert_array_equal(out, SENTINEL_ROW)


class TestHitsAndDuplicates:

    @pytest.mark.parametrize("row", [0, 1, 50, 98, 99])
    def test_by_id_hit_returns_exact_row(self, row):
        side = _full_side()
        out = _np(job.get_order_by_id(jnp.asarray(side), int(side[row, 2])))
        np.testing.assert_array_equal(out, side[row])

    @pytest.mark.parametrize("row", [0, 99])
    def test_by_tid_hit_returns_exact_row(self, row):
        side = _full_side()
        out = _np(job.get_order_by_tid(jnp.asarray(side), int(side[row, 3])))
        np.testing.assert_array_equal(out, side[row])

    @pytest.mark.parametrize("row", [0, 99])
    def test_by_id_and_price_hit_returns_exact_row(self, row):
        side = _full_side()
        out = _np(job.get_order_by_id_and_price(
            jnp.asarray(side), int(side[row, 2]), int(side[row, 0])))
        np.testing.assert_array_equal(out, side[row])

    def test_duplicate_oid_returns_first_match(self):
        side = _full_side()
        side[99, 2] = side[7, 2]
        out = _np(job.get_order_by_id(jnp.asarray(side), int(side[7, 2])))
        np.testing.assert_array_equal(out, side[7])

    def test_duplicate_oid_price_disambiguates_to_row99(self):
        """Same oid at rows 7 and 99, the price picks row 99 (a real hit)."""
        side = _full_side()
        side[99, 2] = side[7, 2]
        out = _np(job.get_order_by_id_and_price(
            jnp.asarray(side), int(side[7, 2]), int(side[99, 0])))
        np.testing.assert_array_equal(out, side[99])


class TestShapesAndSmallSides:

    @pytest.mark.parametrize("n", [1, 2, 7])
    def test_tiny_full_side_miss(self, n):
        side = jnp.asarray(_full_side(n))
        for out in (job.get_order_by_id(side, 4242),
                    job.get_order_by_tid(side, 4242),
                    job.get_order_by_id_and_price(side, 4242, 10_000)):
            out = _np(out)
            assert out.shape == (6,)
            assert out.dtype == np.int32
            np.testing.assert_array_equal(out, SENTINEL_ROW)

    def test_single_row_side_hit(self):
        side = _full_side(1)
        out = _np(job.get_order_by_id(jnp.asarray(side), int(side[0, 2])))
        np.testing.assert_array_equal(out, side[0])

    def test_empty_side_query_empty_slot_value_not_reported_as_order(self):
        """Querying oid == EMPTY_SLOT on an empty side must not look like a
        resting order to callers that test ``row[2] > 0``."""
        side = jnp.asarray(_empty_side())
        out = _np(job.get_order_by_id(side, EMPTY_SLOT))
        assert out.shape == (6,)
        assert out[2] <= 0


class TestTracedAndBatched:

    @pytest.mark.jit
    def test_traced_order_id_under_outer_jit(self):
        side = _full_side()

        @jax.jit
        def f(s, oid):
            return job.get_order_by_id(s, oid)

        s = jnp.asarray(side)
        np.testing.assert_array_equal(_np(f(s, jnp.int32(4242))), SENTINEL_ROW)
        np.testing.assert_array_equal(_np(f(s, jnp.int32(side[99, 2]))), side[99])
        np.testing.assert_array_equal(_np(f(s, jnp.int32(side[3, 2]))), side[3])

    def test_vmap_mixed_hit_and_miss(self):
        sides = np.stack([_full_side(), _full_side(oid_base=2000)])
        oids = jnp.asarray([1099, 1099], dtype=jnp.int32)  # hit in 0, miss in 1
        out = _np(jax.vmap(job.get_order_by_id)(jnp.asarray(sides), oids))
        np.testing.assert_array_equal(out[0], sides[0, 99])
        np.testing.assert_array_equal(out[1], SENTINEL_ROW)

    def test_vmap_by_id_and_price_mixed(self):
        side = _full_side()
        sides = jnp.asarray(np.stack([side, side]))
        oids = jnp.asarray([side[99, 2], side[99, 2]], dtype=jnp.int32)
        prices = jnp.asarray([side[99, 0], side[99, 0] + 1], dtype=jnp.int32)
        out = _np(jax.vmap(job.get_order_by_id_and_price)(sides, oids, prices))
        np.testing.assert_array_equal(out[0], side[99])
        np.testing.assert_array_equal(out[1], SENTINEL_ROW)


# ===========================================================================
# B. ContractBook paths with a full side (last physical row occupied)
# ===========================================================================

CID = 37


@pytest.fixture(scope="module")
def cb():
    contract = Contract(
        contract_id=CID,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )
    mc = OMIEMarketConfig()
    assert mc.book_depth == DEPTH
    return ContractBook(contract, OMIEProductConfig(), mc)


def _order(side, price, qty, oid, ts, er=ExecRestriction.NON, peak=0.0):
    return make_order(contract_id=CID, side=side, price=price, quantity=qty,
                      order_id=oid, agent_id=1, timestamp=ts,
                      exec_restriction=er, peak_qty=peak)


ASK_OIDS = list(range(1000, 1000 + DEPTH))


@pytest.fixture(scope="module")
def full_asks_state(cb):
    """100 resting SELLs (1.0 MW each, 100.00..199.00), ask side full."""
    st = cb.init()
    for i, oid in enumerate(ASK_OIDS):
        st, r = cb.submit_order(st, _order(Side.SELL, 100.0 + i, 1.0, oid, 10 + i))
        assert bool(r.accepted)
    return st


SPARSE_ASK_OIDS = ASK_OIDS[1:]  # ask 1000 (row 0) cancelled


@pytest.fixture(scope="module")
def sparse_asks_state(cb, full_asks_state):
    """99 resting SELLs: row 0 vacated, last physical row (99) still occupied.

    The aggressor tests use this rather than a FULL side so that they isolate
    the D3 lookup-miss (cancel-by-id of a non-resting aggressor) from the
    separate, document-only D6 full-side worst-level wipe
    (OMIE-CAPACITY-DOCS-1), which fires on any dispatch against a full side.
    """
    st = cb.cancel_order(full_asks_state, ASK_OIDS[0], Side.SELL)
    asks = _np(st.lob.asks)
    assert int(asks[DEPTH - 1, 2]) > 0, "construction: row 99 must be occupied"
    assert int((asks[:, 2] > 0).sum()) == DEPTH - 1, "construction: side not full"
    return st


def _assert_asks_intact(cb, st, oids=SPARSE_ASK_OIDS):
    for oid in oids:
        o = cb.get_order(st, oid, Side.SELL)
        assert int(o.order_id) == oid, f"ask {oid} lost"
        assert o.quantity == pytest.approx(1.0, abs=1e-6), f"ask {oid} qty"
    _, asks = cb.get_anonymous_lob(st)
    assert sum(q for _, q in asks) == pytest.approx(float(len(oids)), abs=1e-4)


@pytest.mark.slow
@pytest.mark.jit
class TestContractBookFullSide:
    """slow: _submit_order compile + ~100 submissions to fill one side."""

    def test_get_order_absent_on_full_side_is_sentinel(self, cb, full_asks_state):
        o = cb.get_order(full_asks_state, 777_777, Side.SELL)
        assert int(o.order_id) == SENTINEL_ORDER_ID

    def test_python_cancel_absent_oid_leaves_full_side_untouched(self, cb, full_asks_state):
        st = cb.cancel_order(full_asks_state, 777_777, Side.SELL)
        np.testing.assert_array_equal(_np(st.lob.asks), _np(full_asks_state.lob.asks))
        np.testing.assert_array_equal(_np(st.lob.bids), _np(full_asks_state.lob.bids))

    def test_python_cancel_wrong_side_oid_is_noop(self, cb, full_asks_state):
        """An ask oid cancelled as BUY: the (empty) bid side misses; the
        asks must not be touched either."""
        st = cb.cancel_order(full_asks_state, ASK_OIDS[-1], Side.BUY)
        np.testing.assert_array_equal(_np(st.lob.asks), _np(full_asks_state.lob.asks))

    @pytest.mark.parametrize("victim", [ASK_OIDS[0], ASK_OIDS[50], ASK_OIDS[-1]])
    def test_python_cancel_present_oid_removes_exactly_that_order(self, cb, full_asks_state, victim):
        st = cb.cancel_order(full_asks_state, victim, Side.SELL)
        assert int(cb.get_order(st, victim, Side.SELL).order_id) == SENTINEL_ORDER_ID
        for oid in ASK_OIDS:
            if oid == victim:
                continue
            assert int(cb.get_order(st, oid, Side.SELL).order_id) == oid
        _, asks = cb.get_anonymous_lob(st)
        assert sum(q for _, q in asks) == pytest.approx(DEPTH - 1.0, abs=1e-4)

    def test_non_sell_fully_filled_keeps_every_resting_ask(self, cb, sparse_asks_state):
        st, r = cb.submit_order(sparse_asks_state, _order(Side.BUY, 50.0, 1.0, 1, 500))
        st, r = cb.submit_order(st, _order(Side.SELL, 50.0, 1.0, 5000, 501))
        assert float(r.trade_qty) == pytest.approx(1.0, abs=1e-6)
        _assert_asks_intact(cb, st)
        assert int(cb.get_order(st, 5000, Side.SELL).order_id) == SENTINEL_ORDER_ID

    def test_ioc_sell_residual_cancel_keeps_every_resting_ask(self, cb, sparse_asks_state):
        st, _ = cb.submit_order(sparse_asks_state, _order(Side.BUY, 50.0, 0.5, 2, 500))
        st, r = cb.submit_order(
            st, _order(Side.SELL, 50.0, 1.0, 5001, 501, er=ExecRestriction.IOC))
        assert float(r.trade_qty) == pytest.approx(0.5, abs=1e-6)
        _assert_asks_intact(cb, st)
        assert int(cb.get_order(st, 5001, Side.SELL).order_id) == SENTINEL_ORDER_ID

    def test_fok_sell_keeps_every_resting_ask(self, cb, sparse_asks_state):
        st, _ = cb.submit_order(sparse_asks_state, _order(Side.BUY, 50.0, 1.0, 3, 500))
        st, r = cb.submit_order(
            st, _order(Side.SELL, 50.0, 1.0, 5002, 501, er=ExecRestriction.FOK))
        assert float(r.trade_qty) == pytest.approx(1.0, abs=1e-6)
        _assert_asks_intact(cb, st)

    def test_iceberg_on_full_bid_side_regenerates_and_stays_matchable(self, cb):
        """§6.1: a consumed peak with hidden volume must regenerate even when
        the bid side's last physical row is occupied."""
        st = cb.init()
        st, r = cb.submit_order(
            st, _order(Side.BUY, 50.0, 1.0, 9000, 1, er=ExecRestriction.ICE, peak=0.1))
        assert bool(r.accepted)
        for i in range(DEPTH - 1):
            st, r = cb.submit_order(st, _order(Side.BUY, 10.0 + 0.01 * i, 0.1, 100 + i, 2 + i))
            assert bool(r.accepted)
        # consume the visible peak
        st, r = cb.submit_order(st, _order(Side.SELL, 50.0, 0.1, 7000, 200))
        assert float(r.trade_qty) == pytest.approx(0.1, abs=1e-6)
        # the iceberg must still show a visible slice at 50.00
        bids, _ = cb.get_anonymous_lob(st)
        at50 = sum(q for p, q in bids if p == pytest.approx(50.0, abs=1e-6))
        assert at50 == pytest.approx(0.1, abs=1e-6)
        # and the hidden remainder must be reachable by a later SELL
        st, r = cb.submit_order(st, _order(Side.SELL, 50.0, 0.3, 7001, 201))
        assert float(r.trade_qty) == pytest.approx(0.3, abs=1e-6)
        # 99 plain bids untouched
        for i in range(DEPTH - 1):
            assert int(cb.get_order(st, 100 + i, Side.BUY).order_id) == 100 + i
