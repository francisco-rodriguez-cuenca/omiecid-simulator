"""Locked behavioural-contract tests for Story OMIE-JAXOB-LOOKUP-MISS-1.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-JAXOB-LOOKUP-MISS-1 —
jaxob ``get_order_by_id`` / ``get_order_by_tid`` / ``get_order_by_id_and_price``
return "not found" on a miss, not the side's last physical row" (certification
fix pass, S2 / plan D3).

Rule: a lookup MISS returns ``NEGATIVE_RETURN_ID * ones(6)`` (the documented
"not found" sentinel), whatever sits in the side's last physical row. Anchors:
I2 quantity conservation (a miss must not lead to a cancel that silently
deletes another order) and market-spec §6.1 / §4.2 (an iceberg whose slice is
consumed must regenerate, not be stranded behind a phantom "found" peak).

Contract (Test-first contract row, numbering preserved):
  (1) ``get_order_by_id`` miss -> sentinel, with the last row (row 99) occupied;
      both book sides; queried OID absent / present elsewhere (n/a for an
      OID-only lookup, so: absent with the side partly or fully occupied, and
      an empty side).                                         RED before the fix.
  (2) same for ``get_order_by_tid``.                           RED before the fix.
  (3) same for ``get_order_by_id_and_price`` (OID present at another price ->
      miss).                                                  RED before the fix.
  (4) Guard: a hit in row 0, a middle row and row 99 returns exactly that row.
                                                              GREEN before and after.
  (5) ``C14_last_row_lookup_crossing_non.npz`` replayed through
      ``ContractBook.submit_order``: oid 199 still rests with 1.0 MW and I2
      holds.                                                  RED before the fix.
  (6) ``C14_last_row_lookup_iceberg_stranded.npz``: iceberg 9000 regenerates a
      visible slice and the later SELL 9100 trades 0.5 MW against it.
                                                              RED before the fix.
  (7) Guard: the three functions keep output shape ``(6,)`` / int32 on hit and
      miss, and replaying (5)-(6) compiles ``_submit_order`` once.
                                                              GREEN before and after.

Incidental constants are varied (OIDs of 3-7 digits, TIDs, prices off round
numbers, both sides, partly and fully occupied sides) so the contract does not
hold by accident of a single fixture value.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import omiecid_simulator.jaxob.JaxOrderBookArrays as job
from omiecid_simulator.jaxob.jaxob_constants import NEGATIVE_RETURN_ID
from omiecid_simulator.jaxob.jorderbook import OrderBook
from omiecid_simulator.jaxomiecid.omie_data_loader import _row_to_order_traced
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, _submit_order
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import Side


REPO_ROOT = Path(__file__).resolve().parents[2]
# The ONE fixture-path constant (card: OMIE-CERT-HARNESS-2 may move the C14
# fixtures to tests/omiecid/fixtures/cert/; whichever story merges second updates
# this line only).
FIXTURE_DIR = REPO_ROOT / "notebooks" / "certification" / "fixtures"
CROSSING_NON = FIXTURE_DIR / "C14_last_row_lookup_crossing_non.npz"
ICEBERG_STRANDED = FIXTURE_DIR / "C14_last_row_lookup_iceberg_stranded.npz"

DEPTH = 100          # default book_depth; row 99 is the last physical row
LAST = DEPTH - 1
SENTINEL = np.full((6,), NEGATIVE_RETURN_ID, dtype=np.int32)

# jaxob side codes used by OrderBook.get_order (1 = bid, 0 = ask).
JAXOB_BID, JAXOB_ASK = 1, 0


# ---------------------------------------------------------------------------
# Synthetic (100, 6) side arrays. Columns: P, Q, OID, TID, SEC, NSEC.
# Empty rows are all -1 (jaxob's empty-row convention).
# ---------------------------------------------------------------------------
def _row(i: int, *, base_price: int, oid_base: int, tid_base: int) -> list:
    return [base_price + 7 * i, 3 + (i % 11), oid_base + i, tid_base + 13 * i, 1_000 + i, 17 * i]


def _side(occupied, *, base_price=1_004_137, oid_base=100, tid_base=55_001,
          overrides=None) -> np.ndarray:
    """A (100, 6) int32 side with ``occupied`` row indices filled; ``overrides``
    maps a row index to an explicit 6-vector."""
    arr = np.full((DEPTH, 6), -1, dtype=np.int32)
    for i in occupied:
        arr[i] = _row(i, base_price=base_price, oid_base=oid_base, tid_base=tid_base)
    for i, r in (overrides or {}).items():
        arr[i] = np.asarray(r, dtype=np.int32)
    return arr


# Layouts whose last physical row (99) is OCCUPIED (the defect trigger).
LAST_ROW_LAYOUTS = {
    # only row 99 occupied (oid 199 in the last row, as in the C14 probes)
    "only_last_row": dict(occupied=[LAST]),
    # a sparse side: a few rows plus row 99
    "sparse_with_last_row": dict(occupied=[0, 4, 37, 62, LAST]),
    # a full side (100 resting orders), different incidental constants
    "full_side": dict(occupied=range(DEPTH), base_price=987_653, oid_base=4_200_001,
                      tid_base=7_000_003),
}


def _lob_with(side_arr: np.ndarray, jaxob_side: int):
    """A LobState with ``side_arr`` on the requested side, the other side empty."""
    lob = OrderBook().init()
    a = jnp.asarray(side_arr)
    if jaxob_side == JAXOB_BID:
        return lob._replace(bids=a)
    return lob._replace(asks=a)


def _np(x) -> np.ndarray:
    return np.asarray(jax.device_get(x))


# ===========================================================================
# (1) get_order_by_id: a miss returns the sentinel
# ===========================================================================
class TestGetOrderByIdMiss:

    @pytest.mark.parametrize("layout", sorted(LAST_ROW_LAYOUTS))
    @pytest.mark.parametrize("absent_oid", [199_999, 5, 31_337_021])
    def test_get_order_by_id_miss_returns_sentinel(self, layout, absent_oid):
        side = _side(**LAST_ROW_LAYOUTS[layout])
        assert absent_oid not in side[:, 2]
        out = _np(job.get_order_by_id(jnp.asarray(side), absent_oid))
        np.testing.assert_array_equal(
            out, SENTINEL,
            err_msg=f"{layout}: a miss must return NEGATIVE_RETURN_ID*ones(6), "
                    f"not the last physical row {side[LAST].tolist()}")

    def test_get_order_by_id_miss_on_empty_side_returns_sentinel(self):
        side = _side([])
        out = _np(job.get_order_by_id(jnp.asarray(side), 4127))
        np.testing.assert_array_equal(out, SENTINEL)

    @pytest.mark.parametrize("jaxob_side", [JAXOB_BID, JAXOB_ASK], ids=["bids", "asks"])
    @pytest.mark.parametrize("layout", sorted(LAST_ROW_LAYOUTS))
    def test_orderbook_get_order_miss_returns_sentinel_both_sides(self, jaxob_side, layout):
        """Via the jorderbook trampoline (the path ContractBook and the
        IcebergManager use), with the occupied last row on either side."""
        side = _side(**LAST_ROW_LAYOUTS[layout])
        lob = _lob_with(side, jaxob_side)
        out = _np(OrderBook().get_order(lob, jaxob_side, 8_888_888))
        np.testing.assert_array_equal(out, SENTINEL)

    @pytest.mark.parametrize("jaxob_side", [JAXOB_BID, JAXOB_ASK], ids=["bids", "asks"])
    def test_orderbook_get_order_miss_when_oid_rests_on_other_side(self, jaxob_side):
        """The queried OID rests on the OTHER side only: a miss on this side."""
        side = _side([3, LAST], oid_base=600)            # oids 603, 699
        lob = OrderBook().init()
        other = _side([LAST], oid_base=4_000)              # oid 4099 on the other side
        if jaxob_side == JAXOB_BID:
            lob = lob._replace(bids=jnp.asarray(side), asks=jnp.asarray(other))
        else:
            lob = lob._replace(asks=jnp.asarray(side), bids=jnp.asarray(other))
        out = _np(OrderBook().get_order(lob, jaxob_side, 4_099))
        np.testing.assert_array_equal(out, SENTINEL)


# ===========================================================================
# (2) get_order_by_tid: a miss returns the sentinel
# ===========================================================================
class TestGetOrderByTidMiss:

    @pytest.mark.parametrize("layout", sorted(LAST_ROW_LAYOUTS))
    @pytest.mark.parametrize("absent_tid", [2, 123_457, 99_999_991])
    def test_get_order_by_tid_miss_returns_sentinel(self, layout, absent_tid):
        side = _side(**LAST_ROW_LAYOUTS[layout])
        assert absent_tid not in side[:, 3]
        out = _np(job.get_order_by_tid(jnp.asarray(side), absent_tid))
        np.testing.assert_array_equal(
            out, SENTINEL,
            err_msg=f"{layout}: a TID miss must return the sentinel, "
                    f"not the last physical row {side[LAST].tolist()}")

    def test_get_order_by_tid_miss_when_value_is_an_oid_not_a_tid(self):
        """The queried value occurs in the OID column, never in the TID column."""
        side = _side([0, 50, LAST], oid_base=300, tid_base=70_001)
        probe = int(side[50, 2])                                  # an OID
        assert probe not in side[:, 3]
        out = _np(job.get_order_by_tid(jnp.asarray(side), probe))
        np.testing.assert_array_equal(out, SENTINEL)

    def test_get_order_by_tid_miss_on_empty_side_returns_sentinel(self):
        out = _np(job.get_order_by_tid(jnp.asarray(_side([])), 61))
        np.testing.assert_array_equal(out, SENTINEL)


# ===========================================================================
# (3) get_order_by_id_and_price: a miss returns the sentinel
# ===========================================================================
class TestGetOrderByIdAndPriceMiss:

    @pytest.mark.parametrize("layout", sorted(LAST_ROW_LAYOUTS))
    def test_oid_present_at_another_price_is_a_miss(self, layout):
        side = _side(**LAST_ROW_LAYOUTS[layout])
        occ = [i for i in range(DEPTH) if side[i, 2] > 0]
        # the OID of the FIRST occupied row (never row 99 unless it is the only one)
        i = occ[0]
        oid, price = int(side[i, 2]), int(side[i, 0])
        wrong_price = price + 3
        assert not np.any((side[:, 2] == oid) & (side[:, 0] == wrong_price))
        out = _np(job.get_order_by_id_and_price(jnp.asarray(side), oid, wrong_price))
        np.testing.assert_array_equal(
            out, SENTINEL,
            err_msg=f"{layout}: OID {oid} rests at {price}, not {wrong_price}: a miss")

    @pytest.mark.parametrize("layout", sorted(LAST_ROW_LAYOUTS))
    def test_last_row_oid_at_another_price_is_a_miss(self, layout):
        """The last row's own OID queried at a different price."""
        side = _side(**LAST_ROW_LAYOUTS[layout])
        oid, price = int(side[LAST, 2]), int(side[LAST, 0])
        out = _np(job.get_order_by_id_and_price(jnp.asarray(side), oid, price - 1))
        np.testing.assert_array_equal(out, SENTINEL)

    @pytest.mark.parametrize("layout", sorted(LAST_ROW_LAYOUTS))
    def test_price_present_but_oid_absent_is_a_miss(self, layout):
        side = _side(**LAST_ROW_LAYOUTS[layout])
        price = int(side[LAST, 0])                       # a price that does rest
        out = _np(job.get_order_by_id_and_price(jnp.asarray(side), 777_777, price))
        np.testing.assert_array_equal(out, SENTINEL)

    @pytest.mark.parametrize("jaxob_side", [JAXOB_BID, JAXOB_ASK], ids=["bids", "asks"])
    def test_orderbook_get_order_with_price_miss_both_sides(self, jaxob_side):
        side = _side([2, 40, LAST], oid_base=1_000)
        lob = _lob_with(side, jaxob_side)
        oid, price = int(side[40, 2]), int(side[40, 0])
        out = _np(OrderBook().get_order(lob, jaxob_side, oid, price + 10))
        np.testing.assert_array_equal(out, SENTINEL)


# ===========================================================================
# (4) Guard: a hit returns exactly that row
# ===========================================================================
class TestHitReturnsRow:

    @pytest.mark.parametrize("row", [0, 46, LAST], ids=["row0", "middle", "row99"])
    @pytest.mark.parametrize("full", [False, True], ids=["sparse", "full"])
    def test_hit_returns_row(self, row, full):
        occupied = range(DEPTH) if full else sorted({0, 46, 71, LAST})
        side = _side(occupied, base_price=1_003_311, oid_base=20_000, tid_base=90_001)
        want = side[row]
        a = jnp.asarray(side)
        np.testing.assert_array_equal(_np(job.get_order_by_id(a, int(want[2]))), want)
        np.testing.assert_array_equal(_np(job.get_order_by_tid(a, int(want[3]))), want)
        np.testing.assert_array_equal(
            _np(job.get_order_by_id_and_price(a, int(want[2]), int(want[0]))), want)

    @pytest.mark.parametrize("jaxob_side", [JAXOB_BID, JAXOB_ASK], ids=["bids", "asks"])
    @pytest.mark.parametrize("row", [0, 46, LAST], ids=["row0", "middle", "row99"])
    def test_orderbook_get_order_hit_returns_row(self, jaxob_side, row):
        side = _side(sorted({0, 46, LAST}), oid_base=512)
        lob = _lob_with(side, jaxob_side)
        want = side[row]
        ob = OrderBook()
        np.testing.assert_array_equal(_np(ob.get_order(lob, jaxob_side, int(want[2]))), want)
        np.testing.assert_array_equal(
            _np(ob.get_order(lob, jaxob_side, int(want[2]), int(want[0]))), want)

    def test_duplicate_oid_returns_first_row(self):
        """Documented CAVE: with a repeated OID the first matching row is returned."""
        side = _side([5, 30, LAST], oid_base=900)
        side[30, 2] = side[LAST, 2] = 905
        out = _np(job.get_order_by_id(jnp.asarray(side), 905))
        np.testing.assert_array_equal(out, side[5])


# ===========================================================================
# (7a) Guard: output shape / dtype on hit and miss
# ===========================================================================
class TestShapesAndDtype:

    @pytest.mark.parametrize("hit", [True, False], ids=["hit", "miss"])
    def test_output_shape_6_int32(self, hit):
        side = _side([0, 46, LAST])
        a = jnp.asarray(side)
        r = side[46]
        oid = int(r[2]) if hit else 3_141_592
        tid = int(r[3]) if hit else 2_718_281
        price = int(r[0])
        for out in (job.get_order_by_id(a, oid),
                    job.get_order_by_tid(a, tid),
                    job.get_order_by_id_and_price(a, oid, price)):
            assert out.shape == (6,)
            assert out.dtype == jnp.int32

    @pytest.mark.jit
    def test_traced_order_id_miss_returns_sentinel_under_jit(self):
        """The miss decision is traced (not a Python branch): a jitted caller
        with a traced OID gets the sentinel on a miss and the row on a hit."""
        side = jnp.asarray(_side([0, LAST], oid_base=7_000))

        @jax.jit
        def f(s, oid):
            return job.get_order_by_id(s, oid)

        np.testing.assert_array_equal(_np(f(side, jnp.int32(123))), SENTINEL)
        np.testing.assert_array_equal(_np(f(side, jnp.int32(7_099))), _np(side[LAST]))


# ===========================================================================
# (5)-(6), (7b): C14 last-row fixtures replayed through ContractBook
# ===========================================================================
def _load_tape(path: Path) -> np.ndarray:
    assert path.is_file(), f"fixture missing: {path} (update FIXTURE_DIR if moved)"
    z = np.load(path, allow_pickle=False)
    tape = np.asarray(z["tape"])
    assert tape.dtype == np.int32 and tape.shape[1] == 13
    return tape


def _new_book():
    contract = Contract(contract_id=37, delivery_start_minutes=555,
                        delivery_end_minutes=570, gct_minutes=495,
                        status=ContractStatus.OPEN)
    return ContractBook(contract, OMIEProductConfig(), OMIEMarketConfig())


def _replay_direct(book: ContractBook, tape: np.ndarray):
    """Submit every real row (order_id >= 1) via ContractBook.submit_order."""
    pc = OMIEProductConfig()
    state = book.init()
    for row in tape:
        if int(row[4]) < 1:
            continue
        state, res = book.submit_order(state, _row_to_order_traced(jnp.asarray(row), pc))
        assert bool(res.accepted), f"fixture row {row.tolist()} rejected"
    return state


@pytest.fixture(scope="module")
def crossing_record():
    from tests.omiecid.reference.omie_invariants import build_record
    return build_record(_load_tape(CROSSING_NON))


@pytest.fixture(scope="module")
def iceberg_record():
    from tests.omiecid.reference.omie_invariants import build_record
    return build_record(_load_tape(ICEBERG_STRANDED))


@pytest.mark.jit
@pytest.mark.slow
class TestLastRowCrossingNon:
    """slow: one _submit_order compile + a 103-row replay (~10-15 s, shared)."""

    def test_last_row_crossing_non_keeps_order(self, crossing_record):
        rec = crossing_record
        assert rec.still_resting[199] == 10, (
            "oid 199 (the side's last physical row) must still rest with 1.0 MW "
            f"after the crossing NON 9000 fully executes; got {rec.still_resting[199]}")

    def test_last_row_crossing_non_i2_holds(self, crossing_record):
        from tests.omiecid.reference.omie_invariants import check_i2_quantity_conservation
        i2 = check_i2_quantity_conservation(crossing_record)
        assert i2.status == "pass", [v.message for v in i2.violations]

    def test_last_row_crossing_non_loses_no_resting_bid(self, crossing_record):
        rec = crossing_record
        lost = [o for o in range(101, 200) if rec.still_resting[o] != 10]
        assert lost == [], f"resting 1.0 MW bids lost: {lost}"
        assert not any(rec.evicted.values())
        assert rec.flags["book_full"] is False

    def test_last_row_crossing_non_trades_as_expected(self, crossing_record):
        got = [(f.price_int, f.qty_int, f.passive_oid, f.aggressor_oid)
               for f in crossing_record.fills]
        assert got == [(1004900, 1, 100, 5001), (1005900, 1, 5000, 9000)]
        assert crossing_record.still_resting[9000] == 0

    def test_last_row_order_visible_via_contract_book_get_order(self):
        """Direct ContractBook replay: the public get_order still finds oid 199."""
        book = _new_book()
        state = _replay_direct(book, _load_tape(CROSSING_NON))
        o = book.get_order(state, 199, Side.BUY)
        assert int(o.order_id) == 199
        assert float(o.quantity) == pytest.approx(1.0)


@pytest.mark.jit
@pytest.mark.slow
class TestLastRowIcebergRegenerates:
    """slow: one _submit_order compile + a 104-row replay (~10-15 s, shared)."""

    def test_last_row_iceberg_regenerates(self, iceberg_record):
        """After its peak trades against SELL 5000, iceberg 9000 must show a
        visible slice (0.1 MW) with 0.8 MW hidden -- not 0 visible / 0.9 hidden."""
        rec = iceberg_record
        assert 9000 in rec.accepted
        post = rec.post_snapshots[rec.accepted[9000].submission]
        assert (post.visible_qty(9000, Side.BUY), post.hidden_qty(9000)) == (1, 8)

    def test_later_sell_trades_half_mw_against_iceberg(self, iceberg_record):
        rec = iceberg_record
        assert rec.filled[9100] == 5, (
            f"SELL 9100 0.5 MW @60.00 must trade fully against iceberg 9000; "
            f"filled {rec.filled[9100]}, hidden {rec.hidden_remaining[9000]}")
        legs = [(f.price_int, f.qty_int, f.passive_oid)
                for f in rec.fills if f.aggressor_oid == 9100]
        assert legs == [(1005900, 1, 9000)] * 5

    def test_iceberg_remaining_after_session(self, iceberg_record):
        rec = iceberg_record
        # 1.0 MW total - 0.1 to 5000 - 0.5 to 9100 = 0.4 MW = 1 visible + 3 hidden
        assert (rec.still_resting[9000], rec.hidden_remaining[9000]) == (1, 3)

    def test_iceberg_replay_keeps_other_bids_and_i2(self, iceberg_record):
        from tests.omiecid.reference.omie_invariants import check_i2_quantity_conservation
        rec = iceberg_record
        assert [o for o in range(101, 200) if rec.still_resting[o] != 10] == []
        assert not any(rec.evicted.values())
        i2 = check_i2_quantity_conservation(rec)
        assert i2.status == "pass", [v.message for v in i2.violations]


@pytest.mark.jit
@pytest.mark.slow
def test_jit_shapes_and_one_compile():
    """slow: one fresh _submit_order compile + two ~100-row replays (~10-15 s).

    Replaying both C14 last-row fixtures compiles ``_submit_order`` exactly
    once. The cache is cleared first so entries left by other modules in the
    same (xdist) worker, with different static configs, do not pollute the count.
    """
    if not hasattr(_submit_order, "_cache_size"):
        pytest.skip("_submit_order does not expose `_cache_size`")
    if hasattr(_submit_order, "clear_cache"):
        _submit_order.clear_cache()
    else:  # pragma: no cover - older jax
        _submit_order._clear_cache()

    for path in (CROSSING_NON, ICEBERG_STRANDED):
        book = _new_book()
        _replay_direct(book, _load_tape(path))

    size = _submit_order._cache_size()
    assert size == 1, f"expected one _submit_order compile across (5)-(6), got {size}"

    side = jnp.asarray(_side([0, LAST]))
    for out in (job.get_order_by_id(side, 1),
                job.get_order_by_tid(side, 1),
                job.get_order_by_id_and_price(side, 1, 1)):
        assert out.shape == (6,) and out.dtype == jnp.int32
