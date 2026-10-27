"""Independent audit suite for story P2-L4 — `trade_extraction` JIT-safe.

Audit pass for the JIT-safe rewrite of `extract_fills` / `RawFills` /
`NO_FILLS`. The auditor has NOT read the original locked test file
(`tests/omiecid/test_trade_extraction_jit.py`) and does NOT have visibility
into the developer's reasoning. Coverage is derived independently from:

  - `docs/omiecid/agile-plan.md` §"Story P2-L4" (test-first contract,
    deliverables incl. dtype split (int64, int64, int32) and the scoped
    `enable_x64()` requirement).
  - `docs/omiecid/migration-plan.md` §"Trade extraction detector invariant"
    + RawFills dtype-split addendum.
  - `docs/omiecid/sprint-status.md` §"P2-L4 — Architecture Review" and
    §"P2-L4 — Mid-sprint adjudication" (dtype-split contract, x64
    boundary discipline, OQ-18 narrowed clarification).
  - Public interface of `omiecid_simulator.omiecid.trade_extraction`.
  - Public interface signatures observed at the boundary in
    `omiecid_simulator.omiecid.contract_book` (consumer call sites).

This file is **separate from** `test_trade_extraction_ring_buffer_audit.py`
(the HOTFIX-era audit) — focus areas here are domain-VWAP correctness,
NTRADE_CAP cardinality coverage, dtype-split contract, NO_FILLS value
equivalence, and a structural / AST audit of `extract_fills`'s body.

All audit tests must pass against the current `trade_extraction.py`.
"""

from __future__ import annotations

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.omiecid import trade_extraction
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.trade_extraction import (
    NO_FILLS,
    RawFills,
    extract_fills,
)

# Module-level marker (Story T1.1): audit suite for P2-L4
# `extract_fills` JIT contract; fully-JIT.  Per-test markers stripped.
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Constants pulled from the public domain layer (no implementation peeking)
# ---------------------------------------------------------------------------

P_COL = cst.TradesFeat.P.value
Q_COL = cst.TradesFeat.Q.value
OID_COL = cst.TradesFeat.PASS_OID.value
NTRADE_CAP = cst.NTRADE_CAP
TRADE_FEAT = cst.TRADE_FEAT  # 8


# ---------------------------------------------------------------------------
# Synthetic-fixture helpers (in-test, visible inputs, microsecond runtime)
# ---------------------------------------------------------------------------

def _sentinel_trades(n: int = NTRADE_CAP) -> np.ndarray:
    """All-(-1) trades buffer: jaxob's empty/initial state."""
    return -np.ones((n, TRADE_FEAT), dtype=np.int32)


def _apply_fills(
    base: np.ndarray,
    rows: list[tuple[int, int, int, int]],
) -> jnp.ndarray:
    """Stamp `(slot_idx, price_int, signed_qty_int, oid)` rows onto a copy."""
    arr = base.copy()
    for slot, p, q, oid in rows:
        arr[slot, P_COL] = p
        arr[slot, Q_COL] = q
        arr[slot, OID_COL] = oid
        arr[slot, 3] = oid + 1_000_000  # AGRS_OID — distinct
        arr[slot, 4] = slot  # SEC stub
    return jnp.asarray(arr)


def _expected_aggregates(rows: list[tuple[int, int, int, int]]):
    """Pure-Python ground truth: (total_qty, weighted_price, n)."""
    total_qty = sum(abs(q) for _, _, q, _ in rows)
    weighted_price = sum(p * abs(q) for _, p, q, _ in rows)
    return total_qty, weighted_price, len(rows)


# ===========================================================================
# Focus area 1 — Domain correctness of fill aggregation across cardinalities
# ===========================================================================

class TestAggregationCardinalities:
    """`total_qty_int` and `weighted_price_int` must equal the spec sum
    Σ|q_i| and Σ p_i·|q_i| in tick space, computed in pure Python from
    the same row inputs (no re-running of `extract_fills`)."""

    def test_one_fill_matches_python_ground_truth(self):
        before_np = _sentinel_trades()
        rows = [(7, 1234, 5, 9001)]
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)
        eq, ew, en = _expected_aggregates(rows)
        assert int(f.total_qty_int) == eq
        assert int(f.weighted_price_int) == ew
        assert int(f.num_fills) == en

    def test_two_fills_distinct_slots(self):
        before_np = _sentinel_trades()
        rows = [(0, 500, 3, 1), (50, 700, -4, 2)]  # signed: |-4| = 4
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)
        eq, ew, en = _expected_aggregates(rows)
        assert int(f.total_qty_int) == eq  # 3 + 4 = 7
        assert int(f.weighted_price_int) == ew  # 500*3 + 700*4 = 4300
        assert int(f.num_fills) == en

    def test_five_fills_mixed_signs(self):
        before_np = _sentinel_trades()
        rows = [
            (3, 100, 1, 10),
            (11, 250, -2, 20),
            (42, 999, 7, 30),
            (77, 1, 9, 40),
            (95, 80_000, -6, 50),
        ]
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)
        eq, ew, en = _expected_aggregates(rows)
        assert int(f.total_qty_int) == eq
        assert int(f.weighted_price_int) == ew
        assert int(f.num_fills) == 5

    def test_ntrade_cap_minus_one_fills(self):
        """NTRADE_CAP - 1 distinct fills — boundary case just below the cap."""
        before_np = _sentinel_trades()
        rows = [
            (i, 1000 + i, (i % 9) + 1, 5000 + i)
            for i in range(NTRADE_CAP - 1)
        ]
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)
        eq, ew, en = _expected_aggregates(rows)
        assert int(f.total_qty_int) == eq
        assert int(f.weighted_price_int) == ew
        assert int(f.num_fills) == NTRADE_CAP - 1


# ===========================================================================
# Focus area 2 — Sign correctness of the trade-tape contract
# ===========================================================================

class TestSignCorrectness:
    """Per the public docstring: jaxob Q is signed (negative for
    incoming-buy match, positive for incoming-sell). `extract_fills`
    must take `abs(q)` before aggregation, and `weighted_price_int`
    must remain non-negative for any legal-envelope input (price >= 0
    in tick space, since OMIE encoding is offset so jaxob-int >= 0)."""

    def test_negative_q_does_not_flip_sign_of_total(self):
        before_np = _sentinel_trades()
        rows = [(0, 600, -7, 11)]  # negative signed qty
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)
        assert int(f.total_qty_int) == 7  # |−7|, NOT −7
        assert int(f.total_qty_int) > 0

    def test_weighted_price_non_negative_when_signed_q_negative(self):
        before_np = _sentinel_trades()
        rows = [(5, 1500, -10, 22)]
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)
        # 1500 * |-10| = 15000, NOT -15000
        assert int(f.weighted_price_int) == 15_000
        assert int(f.weighted_price_int) >= 0

    def test_mixed_sign_qs_all_aggregate_as_absolute(self):
        before_np = _sentinel_trades()
        rows = [
            (0, 100, +5, 1),
            (10, 200, -3, 2),
            (20, 300, +8, 3),
            (30, 400, -1, 4),
        ]
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)
        # |5| + |-3| + |8| + |-1| = 17
        assert int(f.total_qty_int) == 17
        # 100*5 + 200*3 + 300*8 + 400*1 = 3900
        assert int(f.weighted_price_int) == 3900


# ===========================================================================
# Focus area 3 — NTRADE_CAP edge cases + dtype-split survival
# ===========================================================================

class TestRingBufferEdges:
    """Build inputs that fill exactly 0, 1, half, and the full ring
    buffer. Assert the `(int64, int64, int32)` dtype split survives,
    `num_fills` matches, and values stay within int64 bounds.

    Use `int(...)` casts on int64 LHS in any literal comparison >= int32
    (the documented Category-B adapter pattern from sprint-status
    §"P2-L4 — Mid-sprint adjudication")."""

    def _assert_dtype_split(self, f):
        assert isinstance(f, RawFills)
        assert isinstance(f.total_qty_int, jnp.ndarray)
        assert isinstance(f.weighted_price_int, jnp.ndarray)
        assert isinstance(f.num_fills, jnp.ndarray)
        assert f.total_qty_int.shape == ()
        assert f.weighted_price_int.shape == ()
        assert f.num_fills.shape == ()
        assert f.total_qty_int.dtype == jnp.int64
        assert f.weighted_price_int.dtype == jnp.int64
        assert f.num_fills.dtype == jnp.int32

    def test_zero_fills_empty_buffer(self):
        before = jnp.asarray(_sentinel_trades())
        f = extract_fills(before, before)
        self._assert_dtype_split(f)
        assert int(f.num_fills) == 0

    def test_exactly_one_fill(self):
        before_np = _sentinel_trades()
        after = _apply_fills(before_np, [(0, 12, 4, 1)])
        f = extract_fills(jnp.asarray(before_np), after)
        self._assert_dtype_split(f)
        assert int(f.num_fills) == 1

    def test_half_buffer_fills(self):
        half = NTRADE_CAP // 2
        before_np = _sentinel_trades()
        rows = [(i, 100 + i, 2, 9000 + i) for i in range(half)]
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)
        self._assert_dtype_split(f)
        assert int(f.num_fills) == half
        # value stays within int64 bounds (use int() to dodge int32-literal
        # canonicalisation per mid-sprint adjudication)
        INT64_MAX = 2**63 - 1
        assert int(f.total_qty_int) < INT64_MAX
        assert int(f.weighted_price_int) < INT64_MAX

    def test_full_buffer_at_envelope_extremes_within_int64(self):
        """Full NTRADE_CAP slots saturated at the OMIE int-envelope
        worst case: q_max=9990, p_max≈1,999,800. Per-row product
        ~2e10 (overflows int32); summed over NTRADE_CAP rows ~2e12.
        Must fit int64 comfortably."""
        before_np = _sentinel_trades()
        # use distinct prices so each row is different (otherwise the
        # whole-row diff has nothing to detect for slot[0] vs sentinel)
        rows = [
            (i, 1_999_800 - i, 9990, 70_000 + i)
            for i in range(NTRADE_CAP)
        ]
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)
        self._assert_dtype_split(f)
        assert int(f.num_fills) == NTRADE_CAP

        # Pure-Python ground truth — exact integer arithmetic
        eq, ew, _ = _expected_aggregates(
            [(slot, p, q, oid) for (slot, p, q, oid) in rows]
        )
        assert int(f.total_qty_int) == eq
        assert int(f.weighted_price_int) == ew
        # Within int64 bounds (use int() LHS cast for the over-int32
        # literal — Category-B adapter pattern)
        INT64_MAX = 2**63 - 1
        assert int(f.weighted_price_int) < INT64_MAX
        assert int(f.weighted_price_int) > 2**31  # exceeds int32, by design


# ===========================================================================
# Focus area 4 — NO_FILLS sentinel domain semantics
# ===========================================================================

class TestNoFillsSentinel:
    """Independent audit of NO_FILLS semantics, derived from the
    deliverable spec ('NO_FILLS module sentinel migrated to a tuple
    of 0-d JAX scalars')."""

    def test_no_fills_total_qty_is_zero(self):
        assert int(NO_FILLS.total_qty_int) == 0

    def test_no_fills_weighted_price_is_zero(self):
        assert int(NO_FILLS.weighted_price_int) == 0

    def test_no_fills_num_fills_is_zero(self):
        assert int(NO_FILLS.num_fills) == 0

    def test_no_fills_dtype_split(self):
        assert NO_FILLS.total_qty_int.dtype == jnp.int64
        assert NO_FILLS.weighted_price_int.dtype == jnp.int64
        assert NO_FILLS.num_fills.dtype == jnp.int32

    def test_extract_fills_on_no_change_is_value_equivalent_to_no_fills(self):
        """Value-level equivalence (not just structural): every leaf
        equals the corresponding NO_FILLS leaf."""
        before = jnp.asarray(_sentinel_trades())
        f = extract_fills(before, before)
        assert int(f.total_qty_int) == int(NO_FILLS.total_qty_int)
        assert int(f.weighted_price_int) == int(NO_FILLS.weighted_price_int)
        assert int(f.num_fills) == int(NO_FILLS.num_fills)

    def test_extract_fills_on_saturated_no_change_matches_no_fills(self):
        """Same buffer in/out, but saturated (no -1 sentinels): still
        value-equivalent to NO_FILLS."""
        arr = np.zeros((NTRADE_CAP, TRADE_FEAT), dtype=np.int32)
        for i in range(NTRADE_CAP):
            arr[i, P_COL] = 100 + i
            arr[i, Q_COL] = 3
            arr[i, OID_COL] = 1000 + i
        buf = jnp.asarray(arr)
        f = extract_fills(buf, buf)
        assert int(f.total_qty_int) == 0
        assert int(f.weighted_price_int) == 0
        assert int(f.num_fills) == 0

    def test_no_fills_reusable_across_multiple_reads(self):
        """Defensive contract: callers may read NO_FILLS multiple times
        (JAX scalars are immutable, but assert the contract). After
        repeated reads, values are unchanged."""
        for _ in range(5):
            assert int(NO_FILLS.total_qty_int) == 0
            assert int(NO_FILLS.weighted_price_int) == 0
            assert int(NO_FILLS.num_fills) == 0

    def test_no_fills_pytree_has_three_leaves(self):
        leaves = jax.tree_util.tree_leaves(NO_FILLS)
        assert len(leaves) == 3


# ===========================================================================
# Focus area 5 — Weighted-average-price reconstruction in EUR/MWh
# ===========================================================================

class TestVWAPReconstructionEurPerMWh:
    """Audit-side check that the int64 accumulator semantics are
    domain-correct: from `total_qty_int`, `weighted_price_int`, and the
    OMIEProductConfig tick scales, reconstruct EUR/MWh VWAP and compare
    against pure-Python ground truth.

    The integer aggregate is exact in tick space; only the final EUR/MWh
    float carries rounding, hence `pytest.approx`.

    Encoding (per omiecid_simulator/omiecid/encoding.py):
        price_eur = jaxob_int * price_tick + price_min
        qty_mw    = jaxob_int * qty_tick
    VWAP price (EUR/MWh) = Σ(p_i * q_i) / Σ(q_i)
        in EUR space:
        avg_jaxob_p = weighted_price_int / total_qty_int
        avg_eur     = avg_jaxob_p * price_tick + price_min
    """

    def test_uniform_price_vwap_equals_that_price(self):
        cfg = OMIEProductConfig()
        # Two fills at the same jaxob price → VWAP equals that price decoded.
        before_np = _sentinel_trades()
        rows = [(0, 500_000, 4, 1), (10, 500_000, 6, 2)]
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)
        avg_jaxob = int(f.weighted_price_int) / int(f.total_qty_int)
        vwap_eur = avg_jaxob * cfg.price_tick + cfg.price_min
        # Decoded value of jaxob-price 500_000:
        expected_eur = 500_000 * cfg.price_tick + cfg.price_min
        assert vwap_eur == pytest.approx(expected_eur, rel=0, abs=1e-9)

    def test_mixed_price_vwap_matches_python_ground_truth(self):
        cfg = OMIEProductConfig()
        before_np = _sentinel_trades()
        # rows in jaxob units; build the python-side ground truth in EUR
        rows = [
            (0, 1_000_000, 3, 1),    # ~ 1e6 ticks
            (5, 1_500_000, 7, 2),
            (12, 2_000, 5, 3),
        ]
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)

        # Reconstruct via aggregates
        avg_jaxob = int(f.weighted_price_int) / int(f.total_qty_int)
        vwap_eur = avg_jaxob * cfg.price_tick + cfg.price_min

        # Python ground truth: decode each row's price, weight by qty (MW)
        total_value_eur_mw = 0.0
        total_qty_mw = 0.0
        for _, p_int, q_int, _ in rows:
            p_eur = p_int * cfg.price_tick + cfg.price_min
            q_mw = abs(q_int) * cfg.qty_tick
            total_value_eur_mw += p_eur * q_mw
            total_qty_mw += q_mw
        expected_vwap = total_value_eur_mw / total_qty_mw

        assert vwap_eur == pytest.approx(expected_vwap, rel=1e-9, abs=1e-6)

    def test_extreme_envelope_vwap_int_aggregate_exact(self):
        """At the OMIE envelope worst case the per-row product overflows
        int32 (~2e10) — the int64 accumulator must keep the aggregate
        exact. Reconstruction via aggregates must match the per-row
        pure-Python sum exactly in jaxob tick space (no rounding)."""
        before_np = _sentinel_trades()
        rows = [
            (0, 1_999_800, 9990, 1),   # ~2e10 — overflows int32
            (1, 1_999_700, 9990, 2),
            (2, 1_999_600, 9990, 3),
        ]
        after = _apply_fills(before_np, rows)
        f = extract_fills(jnp.asarray(before_np), after)

        # Pure-int Python ground truth — exact in tick space
        expected_total_qty = sum(abs(q) for _, _, q, _ in rows)
        expected_weighted = sum(p * abs(q) for _, p, q, _ in rows)
        assert int(f.total_qty_int) == expected_total_qty
        assert int(f.weighted_price_int) == expected_weighted
        # And the per-row product really does exceed int32:
        assert rows[0][1] * abs(rows[0][2]) > 2**31


# ===========================================================================
# Focus area 6 — Whole-row diff under saturation, audit-level
# ===========================================================================

class TestSaturatedRingBufferDetector:
    """Audit-level check that the JIT'd `extract_fills` produces the
    spec-correct fill set on a saturated ring buffer where the
    pre-HOTFIX `(-1 → valid)` sentinel-transition mask would mis-detect.

    Different abstraction from the HOTFIX-era audit: build the saturated
    tape directly via a fixture-style helper, then JIT the function and
    assert outputs against pure-Python ground truth. Does not duplicate
    the HOTFIX-era ring-buffer-immune detector tests."""

    @pytest.fixture
    def saturated_buf(self):
        arr = np.zeros((NTRADE_CAP, TRADE_FEAT), dtype=np.int32)
        for i in range(NTRADE_CAP):
            arr[i, P_COL] = 5_000 + i
            arr[i, Q_COL] = 4
            arr[i, OID_COL] = 100 + i
            arr[i, 3] = 100 + i + 50_000
            arr[i, 4] = i
        return arr

    def test_jit_extract_fills_on_saturated_overwrite(self, saturated_buf):
        before = jnp.asarray(saturated_buf)
        # overwrite three slots with new (price, qty, oid) — note: every
        # OID stays != -1 throughout, so a sentinel-transition detector
        # would see zero fills. The whole-row diff must catch all three.
        after_np = saturated_buf.copy()
        overwrites = [
            (10, 9_000, 6, 99_001),
            (50, 9_500, 8, 99_002),
            (90, 9_900, 2, 99_003),
        ]
        for slot, p, q, oid in overwrites:
            after_np[slot, P_COL] = p
            after_np[slot, Q_COL] = q
            after_np[slot, OID_COL] = oid
            after_np[slot, 3] = oid + 1_000_000
        after = jnp.asarray(after_np)

        jitted = jax.jit(extract_fills)
        f = jitted(before, after)

        # Pure-Python ground truth
        eq = sum(abs(q) for _, _, q, _ in overwrites)
        ew = sum(p * abs(q) for _, p, q, _ in overwrites)
        assert int(f.num_fills) == 3
        assert int(f.total_qty_int) == eq
        assert int(f.weighted_price_int) == ew

    def test_jit_eager_parity_field_by_field(self, saturated_buf):
        """Audit: JIT and eager paths must agree on every leaf for the
        same input. (Architectural invariant from story-card test
        contract: 'eager-vs-jit parity holds field-by-field'.)"""
        before = jnp.asarray(saturated_buf)
        after_np = saturated_buf.copy()
        after_np[42, P_COL] = 8_888
        after_np[42, Q_COL] = -3
        after_np[42, OID_COL] = 77_777
        after = jnp.asarray(after_np)

        eager = extract_fills(before, after)
        jitted = jax.jit(extract_fills)(before, after)

        assert int(eager.total_qty_int) == int(jitted.total_qty_int)
        assert int(eager.weighted_price_int) == int(jitted.weighted_price_int)
        assert int(eager.num_fills) == int(jitted.num_fills)
        # dtype parity too
        assert eager.total_qty_int.dtype == jitted.total_qty_int.dtype
        assert eager.weighted_price_int.dtype == jitted.weighted_price_int.dtype
        assert eager.num_fills.dtype == jitted.num_fills.dtype


# ===========================================================================
# Focus area 7 — AST / static-import boundary audit
# ===========================================================================

class TestStaticBodyAudit:
    """AST-level structural audit (not regex). The story-card invariants
    require that the JIT-blocking idioms be eliminated from
    `extract_fills`'s body:

      - no Python `for ... in range(...)` loop
      - no `int(...)` cast on internal arrays / traced values inside the
        function body (the `int(...)` boundary lives in callers, not
        the callee)
      - no `return None`
    """

    def _extract_fills_ast(self) -> ast.FunctionDef:
        src = inspect.getsource(extract_fills)
        # dedent in case getsource is module-level (it is; no-op for top-level)
        module = ast.parse(src)
        fn = module.body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert fn.name == "extract_fills"
        return fn

    def test_no_python_for_range_loop_in_extract_fills_body(self):
        fn = self._extract_fills_ast()
        for node in ast.walk(fn):
            if isinstance(node, ast.For):
                # is the iterator a `range(...)` call?
                it = node.iter
                if (
                    isinstance(it, ast.Call)
                    and isinstance(it.func, ast.Name)
                    and it.func.id == "range"
                ):
                    pytest.fail(
                        "extract_fills body contains a Python "
                        "`for ... in range(...)` loop; the JIT-safe "
                        "rewrite must use vectorised operations."
                    )

    def test_no_int_cast_inside_extract_fills_body(self):
        """`int(...)` calls on traced values inside the function body
        force concrete materialisation and break tracing. The boundary
        cast lives in callers (`contract_book.py`), per the mid-sprint
        adjudication."""
        fn = self._extract_fills_ast()
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "int"
            ):
                pytest.fail(
                    "extract_fills body contains an `int(...)` cast on a "
                    "value — this concretises a traced array and breaks "
                    "JIT compatibility."
                )

    def test_no_none_return_in_extract_fills(self):
        fn = self._extract_fills_ast()
        for node in ast.walk(fn):
            if isinstance(node, ast.Return):
                v = node.value
                # bare `return` → v is None; `return None` → ast.Constant(None)
                if v is None:
                    pytest.fail("extract_fills has a bare `return` statement.")
                if isinstance(v, ast.Constant) and v.value is None:
                    pytest.fail(
                        "extract_fills returns `None` — the contract requires "
                        "a `RawFills` NamedTuple of three 0-d JAX scalars."
                    )

    def test_extract_fills_module_exports_public_surface(self):
        """Public surface check: the three names required by the
        deliverable contract are importable."""
        assert hasattr(trade_extraction, "extract_fills")
        assert hasattr(trade_extraction, "RawFills")
        assert hasattr(trade_extraction, "NO_FILLS")
