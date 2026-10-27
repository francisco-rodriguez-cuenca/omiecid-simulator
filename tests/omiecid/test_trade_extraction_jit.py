"""Locked behavioural-contract tests for story P2-L4 — `trade_extraction` JIT-safe.

These tests express the JIT-contract that `omiecid_simulator.omiecid.trade_extraction`
must satisfy after the P2-L4 rewrite. They are authored as black-box
contract assertions; they import only the public surface
(`extract_fills`, `RawFills`, `NO_FILLS`).

Acceptance criteria captured here (from the P2-L4 story card "Test-first
contract" plus the architect review of 2026-04-27):

  1. `jax.jit(extract_fills)` compiles cleanly on representative inputs.
  2. The return value is a `RawFills` NamedTuple of three 0-d
     `jnp.ndarray` scalars, with dtype split
     `(int64, int64, int32)` for
     `(total_qty_int, weighted_price_int, num_fills)`.
  3. `jax.tree.leaves(result)` yields only `jnp.ndarray` scalars
     (no Python `int`, no `None`, no `dict`).
  4. Eager-vs-jit field-by-field parity on the same inputs.
  5. `NO_FILLS` is a NamedTuple of 0-d JAX scalars matching the same
     dtype split, NOT Python ints.
  6. Empty / no-change LOB diff produces field values `(0, 0, 0)` with
     the correct dtype split.
  7. Whole-row diff detection
     (`jnp.any(trades_before != trades_after, axis=1)`) is preserved
     under JIT — the JIT'd path produces the same fills as the eager
     path on a saturated ring buffer (this file does NOT duplicate
     `test_trade_extraction_ring_buffer.py`'s assertions; it asserts
     parity under `jax.jit` on a saturation fixture).

These tests are FAIL-by-default against the pre-P2-L4 implementation:
the current `trade_extraction.py` body uses `int(jnp.sum(...))` and a
Python `for`-loop, both of which raise `TracerArrayConversionError`
under `jax.jit`. That is the intended state at author time.

Test tier: most cases carry `@pytest.mark.jit` because they invoke
`jax.jit` compilation. The saturated-ring-buffer parity case also
carries `@pytest.mark.slow` because saturation requires a realistic
replay through `submit_order` (NTRADE_CAP+ matches), parallelling
the helper used by the locked HOTFIX regression suite.

References:
  * `docs/omiecid/agile-plan.md` §"Story P2-L4"
  * `docs/omiecid/sprint-status.md` §"P2-L4 — Architecture Review" (2026-04-27)
  * `docs/omiecid/migration-plan.md` §820 (whole-row diff detector invariant),
    §853 (jit-contract dtype-split adjudication).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    make_order,
)
from omiecid_simulator.omiecid.trade_extraction import NO_FILLS, RawFills, extract_fills

# Module-level marker (Story T1.1): every test in this file traces or
# compiles `extract_fills` / `RawFills`.  Module-level pin replaces the
# stripped per-test `@pytest.mark.jit` decorators.
pytestmark = pytest.mark.jit


NTRADE_CAP = cst.NTRADE_CAP
TRADES_NCOLS = 8  # P, Q, PASS_OID, AGRS_OID, SEC, NSEC, PASS_TID, AGRS_TID


# ---------------------------------------------------------------------------
# Synthetic trade-array fixtures
# ---------------------------------------------------------------------------

def _empty_trades() -> jnp.ndarray:
    """All-sentinel trades buffer: PASS_OID == -1 in every slot."""
    arr = jnp.full((NTRADE_CAP, TRADES_NCOLS), -1, dtype=jnp.int32)
    return arr


def _trades_with_one_new_row(
    *,
    price_int: int,
    qty_int: int,
    pass_oid: int = 555,
    agrs_oid: int = 666,
    sec: int = 100,
    slot: int = 0,
) -> jnp.ndarray:
    """An empty buffer with a single populated trade row at `slot`."""
    arr = _empty_trades()
    row = jnp.array(
        [price_int, qty_int, pass_oid, agrs_oid, sec, 0, pass_oid, agrs_oid],
        dtype=jnp.int32,
    )
    arr = arr.at[slot].set(row)
    return arr


def _trades_with_k_new_rows(
    rows: list[tuple[int, int]],
) -> jnp.ndarray:
    """An empty buffer with `len(rows)` populated trade rows.

    Each row is `(price_int, qty_int)`; the other columns are filled
    with deterministic dummy values so each row is distinct.
    """
    arr = _empty_trades()
    for i, (p, q) in enumerate(rows):
        full_row = jnp.array(
            [p, q, 1000 + i, 2000 + i, 100 + i, 0, 1000 + i, 2000 + i],
            dtype=jnp.int32,
        )
        arr = arr.at[i].set(full_row)
    return arr


# ---------------------------------------------------------------------------
# Saturation helper (mirrors locked HOTFIX style — kept private to this file)
# ---------------------------------------------------------------------------

def _open_contract(contract_id: int = 37) -> Contract:
    return Contract(
        contract_id=contract_id,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


def _fresh_book_and_state(contract_id: int = 37):
    book = ContractBook(
        _open_contract(contract_id),
        OMIEProductConfig(),
        OMIEMarketConfig(),
    )
    return book, book.init()


def _mk_buy(order_id, price, quantity, *, agent_id, timestamp,
            exec_restriction=ExecRestriction.NON, contract_id=37):
    return make_order(
        contract_id=contract_id, side=Side.BUY,
        price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction,
    )


def _mk_sell(order_id, price, quantity, *, agent_id, timestamp,
             exec_restriction=ExecRestriction.NON, contract_id=37):
    return make_order(
        contract_id=contract_id, side=Side.SELL,
        price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction,
    )


def _saturate_trades_via_replay(book, state, *, agent_a=10, agent_b=11,
                                base_ts=1000, base_oid=1000):
    """Realistic replay saturation — same shape as the HOTFIX helper.

    Each iteration submits a resting SELL then a crossing BUY, producing
    one trade row per match. Repeats until the trades ring has fully
    saturated (>= NTRADE_CAP fills).
    """
    oid = base_oid
    ts = base_ts
    for _ in range(NTRADE_CAP + 5):
        sell = _mk_sell(oid, 50.00, 0.1, agent_id=agent_a, timestamp=ts)
        state, _ = book.submit_order(state, sell)
        oid += 1
        ts += 1
        buy = _mk_buy(oid, 50.00, 0.1, agent_id=agent_b, timestamp=ts)
        state, _ = book.submit_order(state, buy)
        oid += 1
        ts += 1

    OID_COL = cst.TradesFeat.PASS_OID.value
    occ = int(jnp.sum(state.lob.trades[:, OID_COL] != -1))
    assert occ == NTRADE_CAP, (
        f"saturation helper failed to fully saturate: occ={occ}/{NTRADE_CAP}"
    )
    return state


# ---------------------------------------------------------------------------
# Category A1 — JIT compilation gate
# ---------------------------------------------------------------------------

class TestExtractFillsCompilesUnderJit:
    """`jax.jit(extract_fills)` must compile cleanly. Acceptance: bullet 1."""

    def test_jit_compiles_on_empty_inputs(self):
        before = _empty_trades()
        after = _empty_trades()
        jit_extract = jax.jit(extract_fills)
        # The compile happens on first call; we just need it not to raise.
        result = jit_extract(before, after)
        # Smoke: the result is a RawFills with our three named fields.
        assert hasattr(result, "total_qty_int")
        assert hasattr(result, "weighted_price_int")
        assert hasattr(result, "num_fills")

    def test_jit_compiles_on_single_new_fill(self):
        before = _empty_trades()
        after = _trades_with_one_new_row(price_int=1500, qty_int=300)
        jit_extract = jax.jit(extract_fills)
        result = jit_extract(before, after)
        assert hasattr(result, "num_fills")

    def test_jit_compiles_on_multi_row_fixture(self):
        before = _empty_trades()
        after = _trades_with_k_new_rows([
            (1500, 300),
            (1600, 200),
            (1400, 100),
        ])
        jit_extract = jax.jit(extract_fills)
        result = jit_extract(before, after)
        assert hasattr(result, "num_fills")


# ---------------------------------------------------------------------------
# Category A2 — RawFills tree shape and dtype split
# ---------------------------------------------------------------------------

class TestRawFillsTreeShapeAndDtype:
    """Acceptance bullets 2, 3, 5: NamedTuple of 0-d JAX scalars,
    dtype split (int64, int64, int32), no Python leaves.
    """

    def test_eager_result_fields_are_jax_scalars(self):
        """All three fields must be `jnp.ndarray`, not Python ints."""
        before = _empty_trades()
        after = _trades_with_one_new_row(price_int=1500, qty_int=300)
        result = extract_fills(before, after)
        assert isinstance(result.total_qty_int, jnp.ndarray), (
            f"total_qty_int must be jnp.ndarray, got {type(result.total_qty_int)}"
        )
        assert isinstance(result.weighted_price_int, jnp.ndarray), (
            f"weighted_price_int must be jnp.ndarray, "
            f"got {type(result.weighted_price_int)}"
        )
        assert isinstance(result.num_fills, jnp.ndarray), (
            f"num_fills must be jnp.ndarray, got {type(result.num_fills)}"
        )

    def test_eager_result_fields_are_zero_dim(self):
        before = _empty_trades()
        after = _trades_with_one_new_row(price_int=1500, qty_int=300)
        result = extract_fills(before, after)
        assert result.total_qty_int.shape == (), result.total_qty_int.shape
        assert result.weighted_price_int.shape == (), result.weighted_price_int.shape
        assert result.num_fills.shape == (), result.num_fills.shape

    def test_eager_dtype_split_int64_int64_int32(self):
        """Acceptance bullet 2: dtype split (int64, int64, int32)."""
        before = _empty_trades()
        after = _trades_with_one_new_row(price_int=1500, qty_int=300)
        result = extract_fills(before, after)
        assert result.total_qty_int.dtype == jnp.int64, (
            f"total_qty_int dtype must be int64, got {result.total_qty_int.dtype}"
        )
        assert result.weighted_price_int.dtype == jnp.int64, (
            f"weighted_price_int dtype must be int64, "
            f"got {result.weighted_price_int.dtype}"
        )
        assert result.num_fills.dtype == jnp.int32, (
            f"num_fills dtype must be int32, got {result.num_fills.dtype}"
        )

    def test_jit_result_has_same_tree_shape_and_dtype_split(self):
        """Same dtype split must hold under jax.jit, not just eager."""
        before = _empty_trades()
        after = _trades_with_one_new_row(price_int=1500, qty_int=300)
        result = jax.jit(extract_fills)(before, after)
        assert isinstance(result.total_qty_int, jnp.ndarray)
        assert result.total_qty_int.dtype == jnp.int64
        assert result.weighted_price_int.dtype == jnp.int64
        assert result.num_fills.dtype == jnp.int32
        assert result.total_qty_int.shape == ()
        assert result.weighted_price_int.shape == ()
        assert result.num_fills.shape == ()

    def test_jax_tree_leaves_are_three_ndarray_scalars(self):
        """Acceptance bullet 3: jax.tree.leaves(result) yields only
        jnp.ndarray scalars; no Python ints, no None, no dict.
        """
        before = _empty_trades()
        after = _trades_with_one_new_row(price_int=1500, qty_int=300)
        result = extract_fills(before, after)
        leaves = jax.tree.leaves(result)
        assert len(leaves) == 3, (
            f"expected 3 leaves (NamedTuple of three scalars), got {len(leaves)}"
        )
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray), (
                f"every leaf must be jnp.ndarray, got {type(leaf)}"
            )
            assert leaf.shape == (), (
                f"every leaf must be 0-d, got shape {leaf.shape}"
            )

    def test_result_is_named_tuple_with_field_order(self):
        """RawFills must remain a NamedTuple-shaped pytree with the three
        documented fields in declaration order. Architect Q4 / story-card
        deliverable.
        """
        before = _empty_trades()
        after = _trades_with_one_new_row(price_int=1500, qty_int=300)
        result = extract_fills(before, after)
        # NamedTuple gives us _fields in the declared order.
        assert hasattr(result, "_fields"), (
            "RawFills must be a NamedTuple (have a _fields attribute)"
        )
        assert result._fields == (
            "total_qty_int",
            "weighted_price_int",
            "num_fills",
        ), f"unexpected RawFills field order: {result._fields}"


# ---------------------------------------------------------------------------
# Category A3 — Empty / no-change LOB diff produces (0, 0, 0)
# ---------------------------------------------------------------------------

class TestNoChangeProducesZeroFills:
    """Acceptance bullet 6: empty/no-change diff yields field values
    `(0, 0, 0)` with the correct dtype split.
    """

    def test_no_change_eager_returns_zero_with_dtype_split(self):
        buf = _empty_trades()
        result = extract_fills(buf, buf)
        assert int(result.total_qty_int) == 0
        assert int(result.weighted_price_int) == 0
        assert int(result.num_fills) == 0
        # Dtype split must hold even on the zero path.
        assert result.total_qty_int.dtype == jnp.int64
        assert result.weighted_price_int.dtype == jnp.int64
        assert result.num_fills.dtype == jnp.int32

    def test_no_change_jit_returns_zero_with_dtype_split(self):
        buf = _empty_trades()
        result = jax.jit(extract_fills)(buf, buf)
        assert int(result.total_qty_int) == 0
        assert int(result.weighted_price_int) == 0
        assert int(result.num_fills) == 0
        assert result.total_qty_int.dtype == jnp.int64
        assert result.weighted_price_int.dtype == jnp.int64
        assert result.num_fills.dtype == jnp.int32

    def test_no_change_on_populated_buffer_returns_zero(self):
        """Even on a fully-populated buffer, before == after must yield zero."""
        populated = _trades_with_k_new_rows([
            (1500, 300),
            (1600, 200),
        ])
        result = extract_fills(populated, populated)
        assert int(result.num_fills) == 0
        assert int(result.total_qty_int) == 0
        assert int(result.weighted_price_int) == 0


# ---------------------------------------------------------------------------
# Category A4 — Eager / JIT field-by-field parity
# ---------------------------------------------------------------------------

class TestEagerJitParity:
    """Acceptance bullet 4: eager and jit paths must produce identical
    field values on the same inputs.
    """

    def test_parity_single_fill(self):
        before = _empty_trades()
        after = _trades_with_one_new_row(price_int=1500, qty_int=300)
        eager = extract_fills(before, after)
        jitted = jax.jit(extract_fills)(before, after)
        assert int(eager.total_qty_int) == int(jitted.total_qty_int)
        assert int(eager.weighted_price_int) == int(jitted.weighted_price_int)
        assert int(eager.num_fills) == int(jitted.num_fills)

    def test_parity_multi_fill(self):
        before = _empty_trades()
        after = _trades_with_k_new_rows([
            (1500, 300),
            (1600, 200),
            (1400, 100),
        ])
        eager = extract_fills(before, after)
        jitted = jax.jit(extract_fills)(before, after)
        assert int(eager.total_qty_int) == int(jitted.total_qty_int)
        assert int(eager.weighted_price_int) == int(jitted.weighted_price_int)
        assert int(eager.num_fills) == int(jitted.num_fills)

    def test_parity_no_change(self):
        buf = _empty_trades()
        eager = extract_fills(buf, buf)
        jitted = jax.jit(extract_fills)(buf, buf)
        assert int(eager.total_qty_int) == int(jitted.total_qty_int)
        assert int(eager.weighted_price_int) == int(jitted.weighted_price_int)
        assert int(eager.num_fills) == int(jitted.num_fills)

    def test_parity_negative_q_column_is_absolute_valued(self):
        """Q is signed in jaxob (negative for incoming-buy matches).
        `extract_fills` aggregates `abs(Q)`. Eager and jit must agree.
        """
        before = _empty_trades()
        # Manually craft a row with negative Q.
        after = _empty_trades()
        row = jnp.array([1500, -300, 555, 666, 100, 0, 555, 666],
                        dtype=jnp.int32)
        after = after.at[0].set(row)

        eager = extract_fills(before, after)
        jitted = jax.jit(extract_fills)(before, after)

        # abs(Q)=300, P=1500 → total_qty_int=300, weighted=450000.
        assert int(eager.total_qty_int) == 300
        assert int(eager.weighted_price_int) == 1500 * 300
        assert int(eager.num_fills) == 1
        assert int(eager.total_qty_int) == int(jitted.total_qty_int)
        assert int(eager.weighted_price_int) == int(jitted.weighted_price_int)
        assert int(eager.num_fills) == int(jitted.num_fills)


# ---------------------------------------------------------------------------
# Category A5 — Whole-row diff detector preserved under JIT
# ---------------------------------------------------------------------------

class TestWholeRowDiffPreservedUnderJit:
    """Acceptance bullet 7: the ring-buffer-immune whole-row diff detector
    introduced by P2-HOTFIX-ioc-fill-detection must survive the JAX
    conversion. This file does NOT duplicate the locked HOTFIX
    assertions — it asserts that on a saturated ring buffer, the
    JIT'd path produces the same result as the eager path
    (i.e. the detector continues to fire under jit).

    See the locked file `tests/omiecid/test_trade_extraction_ring_buffer.py`
    for the underlying behavioural lock; this file only asserts the
    eager == jit equivalence on a saturation fixture.
    """

    @pytest.mark.slow
    def test_jit_eager_parity_on_saturated_buffer_single_fill(self):
        """slow: realistic replay saturation (>= NTRADE_CAP matches).

        Drive the ring buffer to full occupancy via `submit_order` replay,
        then submit one final crossing match. The eager path must report
        a positive fill (locked by HOTFIX); the JIT path must report
        the same field values.
        """
        book, state = _fresh_book_and_state()
        state = _saturate_trades_via_replay(book, state)

        # Park a sell at 60.00; then a crossing buy.
        state, _ = book.submit_order(
            state, _mk_sell(90000, 60.00, 0.3, agent_id=20, timestamp=9000),
        )
        before = state.lob.trades
        state, _ = book.submit_order(
            state, _mk_buy(90001, 60.00, 0.3, agent_id=21, timestamp=9001),
        )
        after = state.lob.trades

        eager = extract_fills(before, after)
        jitted = jax.jit(extract_fills)(before, after)

        # Eager path must detect the fill (HOTFIX invariant).
        assert int(eager.num_fills) >= 1, (
            "eager whole-row-diff detector must fire on saturated buffer; "
            "if this trips, the rewrite reverted to the sentinel-transition mask"
        )

        # JIT path must agree field-by-field.
        assert int(eager.total_qty_int) == int(jitted.total_qty_int), (
            f"JIT total_qty_int diverged from eager under saturation: "
            f"eager={int(eager.total_qty_int)}, jit={int(jitted.total_qty_int)}"
        )
        assert int(eager.weighted_price_int) == int(jitted.weighted_price_int), (
            f"JIT weighted_price_int diverged from eager under saturation: "
            f"eager={int(eager.weighted_price_int)}, "
            f"jit={int(jitted.weighted_price_int)}"
        )
        assert int(eager.num_fills) == int(jitted.num_fills), (
            f"JIT num_fills diverged from eager under saturation: "
            f"eager={int(eager.num_fills)}, jit={int(jitted.num_fills)}"
        )


# ---------------------------------------------------------------------------
# Category A6 — NO_FILLS module sentinel migration
# ---------------------------------------------------------------------------

class TestNoFillsSentinelIsJaxScalarTuple:
    """Acceptance bullet 5 (also story-card Deliverables): `NO_FILLS`
    is a NamedTuple of 0-d JAX scalars matching the dtype split, NOT
    a tuple of Python ints.
    """

    def test_no_fills_fields_are_jax_scalars(self):
        assert isinstance(NO_FILLS.total_qty_int, jnp.ndarray), (
            f"NO_FILLS.total_qty_int must be jnp.ndarray, got "
            f"{type(NO_FILLS.total_qty_int)} (Python int sentinel "
            f"is the pre-P2-L4 state)"
        )
        assert isinstance(NO_FILLS.weighted_price_int, jnp.ndarray), (
            f"NO_FILLS.weighted_price_int must be jnp.ndarray, got "
            f"{type(NO_FILLS.weighted_price_int)}"
        )
        assert isinstance(NO_FILLS.num_fills, jnp.ndarray), (
            f"NO_FILLS.num_fills must be jnp.ndarray, got "
            f"{type(NO_FILLS.num_fills)}"
        )

    def test_no_fills_dtype_split_matches_extract_fills(self):
        """NO_FILLS dtype split must match the live `extract_fills` return
        type: int64, int64, int32.
        """
        assert NO_FILLS.total_qty_int.dtype == jnp.int64, (
            f"NO_FILLS.total_qty_int dtype must be int64, got "
            f"{NO_FILLS.total_qty_int.dtype}"
        )
        assert NO_FILLS.weighted_price_int.dtype == jnp.int64, (
            f"NO_FILLS.weighted_price_int dtype must be int64, got "
            f"{NO_FILLS.weighted_price_int.dtype}"
        )
        assert NO_FILLS.num_fills.dtype == jnp.int32, (
            f"NO_FILLS.num_fills dtype must be int32, got "
            f"{NO_FILLS.num_fills.dtype}"
        )

    def test_no_fills_fields_are_zero_dim(self):
        assert NO_FILLS.total_qty_int.shape == ()
        assert NO_FILLS.weighted_price_int.shape == ()
        assert NO_FILLS.num_fills.shape == ()

    def test_no_fills_field_values_are_zero(self):
        """The sentinel must still represent (0, 0, 0)."""
        assert int(NO_FILLS.total_qty_int) == 0
        assert int(NO_FILLS.weighted_price_int) == 0
        assert int(NO_FILLS.num_fills) == 0

    def test_no_fills_equals_extract_fills_on_no_change(self):
        """`NO_FILLS` and `extract_fills` on a no-change diff must agree
        field-by-field (same dtype split, same zero values).
        """
        buf = _empty_trades()
        result = extract_fills(buf, buf)
        assert int(NO_FILLS.total_qty_int) == int(result.total_qty_int)
        assert int(NO_FILLS.weighted_price_int) == int(result.weighted_price_int)
        assert int(NO_FILLS.num_fills) == int(result.num_fills)
        assert NO_FILLS.total_qty_int.dtype == result.total_qty_int.dtype
        assert NO_FILLS.weighted_price_int.dtype == result.weighted_price_int.dtype
        assert NO_FILLS.num_fills.dtype == result.num_fills.dtype

    def test_no_fills_tree_leaves_are_jax_scalars(self):
        """`NO_FILLS` is a pytree of three 0-d JAX scalars."""
        leaves = jax.tree.leaves(NO_FILLS)
        assert len(leaves) == 3
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray)
            assert leaf.shape == ()
