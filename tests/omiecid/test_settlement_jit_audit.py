"""Audit tests for P2-L2 — JIT-safe settlement primitives.

Authored independently from the original P2-L2 locked test suite. The
original test file is deliberately NOT read; audit assertions are derived
from:
  - docs/omiecid/agile-plan.md §P2-L2 (v5.8)
  - docs/omiecid/migration-plan.md §Phase 2 — Settlement API pin (v5.6)
  - docs/omiecid/market-spec.md §2.3, §8 (bilateral zero-sum)
  - inspection of the public surface of
    `omiecid_simulator/omiecid/settlement.py` and friends — not its internals.

Audit focus areas:
  (A) Trace-mode parity: eager == jax.vmap(...) under enable_x64 context.
  (B) Decoding contract: offset-encoded price column, p_eur = offset *
      price_tick + price_min, bilateral zero-sum holds exactly.
  (C) Int64 exactness envelope: exact integer arithmetic at the OMIE
      ceiling (999 MW x 9999 EUR/MWh) with no float drift; probes just
      under and at the envelope.
  (D) Float32 return boundary: dtype is strictly float32, 0-d shape.
  (E) Padding / sentinel immunity: zero-qty rows and absent-agent rows
      contribute EXACTLY zero (integer equality, not approx).
  (F) Scalar compute_trade_value Phase-1 parity: known market scenarios.
  (G) Module-import side-effect audit: importing the settlement module
      MUST NOT flip the global jax_enable_x64 flag.
  (H) SENTINEL_AGENT_ID = -1 isolation: never matches a real agent row.
  (I) jax.vmap(..., in_axes=(None, 0)) over agent_id: shape + zero-sum
      across the full agent batch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

# Import encoding / config BEFORE settlement so we can observe the
# x64 flag before and after the settlement import for focus area (G).
from omiecid_simulator.omiecid.encoding import encode_price, encode_qty
from omiecid_simulator.omiecid.jax_constants import SENTINEL_AGENT_ID
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig


# ---------------------------------------------------------------------------
# Module-import side-effect audit (focus G) — executed at collection time.
# ---------------------------------------------------------------------------

_X64_BEFORE_IMPORT = jax.config.read("jax_enable_x64")
from omiecid_simulator.omiecid import settlement  # noqa: E402
from omiecid_simulator.omiecid.settlement import (  # noqa: E402
    compute_agent_pnl,
    compute_trade_value,
)
_X64_AFTER_IMPORT = jax.config.read("jax_enable_x64")

# Module-level marker (Story T1.1): every test in this file audits the
# P2-L2 settlement JIT contract.  Marker pinned at module scope.
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = OMIEProductConfig()
_COL_TS, _COL_PRICE, _COL_QTY, _COL_BUYER, _COL_SELLER, _COL_CID = range(6)


def _row(buyer: int, seller: int, price_eur: float, qty_mw: float,
         ts: int = 0, cid: int = 81) -> list:
    return [
        int(ts),
        int(encode_price(price_eur, _CFG)),
        int(encode_qty(qty_mw, _CFG)),
        int(buyer),
        int(seller),
        int(cid),
    ]


def _tape(rows) -> jnp.ndarray:
    if not rows:
        return jnp.zeros((0, 6), dtype=jnp.int32)
    return jnp.asarray(rows, dtype=jnp.int32)


def _pnl_reference_eur(rows, agent_id: int) -> float:
    """Pure-Python float reference using the decoded EUR/MWh pipeline."""
    total = 0.0
    for r in rows:
        price_eur = r[_COL_PRICE] * _CFG.price_tick + _CFG.price_min
        qty_mw = r[_COL_QTY] * _CFG.qty_tick
        if r[_COL_SELLER] == agent_id and r[_COL_BUYER] != agent_id:
            total += qty_mw * price_eur * _CFG.delivery_duration_hours
        elif r[_COL_BUYER] == agent_id and r[_COL_SELLER] != agent_id:
            total -= qty_mw * price_eur * _CFG.delivery_duration_hours
    return total


# ===========================================================================
# (G) Module import side-effects
# ===========================================================================

class TestImportSideEffects:
    def test_settlement_import_did_not_flip_x64_flag(self):
        # Architectural invariant from migration-plan v5.6: settlement module
        # must NOT flip the global jax_enable_x64 flag (P2-L1 Method A).
        assert _X64_BEFORE_IMPORT == _X64_AFTER_IMPORT, (
            "Importing omiecid_simulator.omiecid.settlement flipped the global "
            "jax_enable_x64 flag (before=%r, after=%r) — this violates the "
            "P2-L1 Method A int32 validator invariant."
            % (_X64_BEFORE_IMPORT, _X64_AFTER_IMPORT)
        )

    def test_x64_flag_is_off_after_eager_call(self):
        # The eager path uses a scoped enable_x64 context. After the call
        # returns, the global flag must still be False (or at least
        # unchanged).
        before = jax.config.read("jax_enable_x64")
        trades = _tape([_row(1, 2, 50.0, 10.0)])
        _ = compute_agent_pnl(trades, jnp.int32(1))
        after = jax.config.read("jax_enable_x64")
        assert before == after


# ===========================================================================
# (D) / (F) scalar compute_trade_value
# ===========================================================================

class TestTradeValueScalar:
    def test_returns_float32_zero_dim(self):
        v = compute_trade_value(10.0, 50.0)
        assert v.dtype == jnp.float32
        assert v.shape == ()

    @pytest.mark.parametrize(
        "q_mw,p_eur,d_h,expected",
        [
            (10.0, 50.0, 0.25, 125.0),
            (1.0, 100.0, 0.25, 25.0),
            (999.0, 9999.0, 0.25, 999.0 * 9999.0 * 0.25),
            (0.0, 123.45, 0.25, 0.0),
            (5.0, -10.0, 0.25, -12.5),      # negative EUR/MWh
            (2.0, 40.0, 1.0, 80.0),         # non-default duration
        ],
    )
    def test_known_scenarios(self, q_mw, p_eur, d_h, expected):
        v = compute_trade_value(q_mw, p_eur, d_h)
        assert float(v) == pytest.approx(expected, rel=1e-5, abs=1e-3)

    def test_jit_compilable(self):
        f = jax.jit(compute_trade_value)
        v = f(10.0, 50.0, 0.25)
        assert v.dtype == jnp.float32
        assert float(v) == pytest.approx(125.0)


# ===========================================================================
# (D) / (E) compute_agent_pnl — basic eager contract
# ===========================================================================

class TestPnLReturnContract:
    def test_returns_float32_scalar(self):
        trades = _tape([_row(1, 2, 50.0, 10.0)])
        v = compute_agent_pnl(trades, jnp.int32(2))
        assert v.dtype == jnp.float32
        assert v.shape == ()

    def test_empty_tape_returns_zero(self):
        trades = _tape([])
        v = compute_agent_pnl(trades, jnp.int32(7))
        assert float(v) == 0.0

    def test_absent_agent_returns_exact_zero(self):
        # Agent 999 does not appear in any row: must be exactly 0.0, not
        # merely near-zero.
        rows = [
            _row(1, 2, 50.0, 10.0),
            _row(3, 4, -20.0, 5.0),
            _row(2, 1, 75.25, 0.3),
        ]
        v = compute_agent_pnl(_tape(rows), jnp.int32(999))
        assert float(v) == 0.0

    def test_padding_rows_contribute_exactly_zero(self):
        # Zero-qty rows (loader padding). Add them between real rows; PnL
        # must equal that of the same tape without padding, to the bit.
        real = [_row(1, 2, 50.0, 10.0), _row(1, 3, 60.0, 5.0)]
        pad = [_row(0, 0, 0.0, 0.0) for _ in range(7)]  # qty_tick == 0
        v_real = compute_agent_pnl(_tape(real), jnp.int32(1))
        v_padded = compute_agent_pnl(_tape(real + pad), jnp.int32(1))
        assert float(v_real) == float(v_padded)

    def test_sentinel_agent_id_never_matches_real_row(self):
        # SENTINEL_AGENT_ID = -1 must never match any real agent (loader
        # emits agent_ids >= 0). Real rows should not leak into its PnL.
        assert SENTINEL_AGENT_ID == -1
        rows = [
            _row(0, 1, 50.0, 10.0),
            _row(2, 3, 60.0, 7.0),
            _row(4, 5, -25.0, 2.0),
        ]
        v = compute_agent_pnl(_tape(rows), jnp.int32(SENTINEL_AGENT_ID))
        assert float(v) == 0.0


# ===========================================================================
# (B) Decoding contract invariants
# ===========================================================================

class TestDecodingContract:
    def test_seller_receives_positive_pnl(self):
        rows = [_row(buyer=1, seller=2, price_eur=50.0, qty_mw=10.0)]
        v = compute_agent_pnl(_tape(rows), jnp.int32(2))
        expected = 10.0 * 50.0 * 0.25  # = 125 EUR
        assert float(v) == pytest.approx(expected, rel=1e-5)

    def test_buyer_pays_negative_pnl(self):
        rows = [_row(buyer=1, seller=2, price_eur=50.0, qty_mw=10.0)]
        v = compute_agent_pnl(_tape(rows), jnp.int32(1))
        assert float(v) == pytest.approx(-125.0, rel=1e-5)

    def test_bilateral_zero_sum_single_row(self):
        rows = [_row(1, 2, 73.49, 17.3)]
        v_buyer = compute_agent_pnl(_tape(rows), jnp.int32(1))
        v_seller = compute_agent_pnl(_tape(rows), jnp.int32(2))
        # Exact zero-sum in tick space; after float32 decode the sum may
        # carry one ULP, but for a single small row it should be bitwise.
        assert float(v_buyer + v_seller) == 0.0

    def test_bilateral_zero_sum_multi_row(self):
        rows = [
            _row(1, 2, 50.0, 10.0),
            _row(2, 1, -25.5, 3.0),   # roles flipped + negative price
            _row(1, 2, 9999.0, 999.0),
            _row(1, 2, -9999.0, 100.0),
        ]
        v1 = compute_agent_pnl(_tape(rows), jnp.int32(1))
        v2 = compute_agent_pnl(_tape(rows), jnp.int32(2))
        # Zero-sum is the product of exact-integer tick arithmetic; float32
        # decoding afterwards may not be bitwise zero but must be tiny
        # relative to the envelope (O(999*9999*0.25) ~= 2.5e6).
        s = float(v1 + v2)
        envelope = 999.0 * 9999.0 * 0.25
        assert abs(s) <= envelope * 1e-5

    def test_price_decode_formula_single_row(self):
        # Verify decoding formula: p_eur = offset * price_tick + price_min.
        # Build a tape by-hand (bypassing encode_price) so the offset
        # column is visible.
        offset = 1_234_567  # -> price_eur = 1_234_567 * 0.01 + (-9999.0)
        price_eur = offset * _CFG.price_tick + _CFG.price_min
        qty_mw = 5.0
        qty_tick_int = int(qty_mw / _CFG.qty_tick)
        tape = jnp.asarray(
            [[0, offset, qty_tick_int, 10, 20, 81]], dtype=jnp.int32,
        )
        v = compute_agent_pnl(tape, jnp.int32(20))
        expected = qty_mw * price_eur * _CFG.delivery_duration_hours
        assert float(v) == pytest.approx(expected, rel=1e-5)

    def test_negative_price_crossing_zero(self):
        # Negative EUR/MWh is legal per market-spec; confirm sign handling
        # in offset-decoded arithmetic.
        rows = [_row(1, 2, -500.0, 4.0)]
        v_seller = compute_agent_pnl(_tape(rows), jnp.int32(2))
        v_buyer = compute_agent_pnl(_tape(rows), jnp.int32(1))
        # Seller of a negative-priced trade PAYS (receives negative PnL).
        assert float(v_seller) < 0.0
        assert float(v_buyer) > 0.0
        assert float(v_seller + v_buyer) == pytest.approx(0.0, abs=1e-3)


# ===========================================================================
# (C) Int64 exactness envelope
# ===========================================================================

class TestInt64Exactness:
    def test_single_row_at_envelope_ceiling(self):
        # Extreme OMIE row: 999 MW x 9999 EUR/MWh. The per-row tick product
        # Q_tick (9990) * p_offset (~1_999_800) ~= 2e10 > int32 ceiling.
        rows = [_row(1, 2, 9999.0, 999.0)]
        v = compute_agent_pnl(_tape(rows), jnp.int32(2))
        expected = 999.0 * 9999.0 * 0.25  # ~ 2.5e6 EUR
        assert float(v) == pytest.approx(expected, rel=1e-5)

    def test_row_at_negative_envelope(self):
        rows = [_row(1, 2, -9999.0, 999.0)]
        v = compute_agent_pnl(_tape(rows), jnp.int32(2))
        expected = 999.0 * -9999.0 * 0.25
        assert float(v) == pytest.approx(expected, rel=1e-5)

    def test_many_rows_at_envelope_would_overflow_int32(self):
        # 1000 rows each at the envelope. The aggregate tick accumulator
        # would be ~2e13, which is far beyond int32's 2.15e9 ceiling.
        # int64 must keep this exact; float64 decoding preserves precision
        # since 2e13 is still well within float64's 2^53 ~ 9e15 ceiling.
        N = 1000
        rows = [_row(1, 2, 9999.0, 999.0) for _ in range(N)]
        v = compute_agent_pnl(_tape(rows), jnp.int32(2))
        expected = N * 999.0 * 9999.0 * 0.25
        assert float(v) == pytest.approx(expected, rel=1e-5)

    def test_reference_pipeline_parity_large_mixed_tape(self):
        # Deterministic pseudo-random tape of 2000 rows across 16 agents,
        # mixed signs. Compare against the pure-Python float reference
        # (focus-area safety net).
        rng = np.random.default_rng(20260424)
        N = 2000
        rows = []
        for _ in range(N):
            b, s = rng.integers(0, 16, size=2)
            while s == b:
                s = int(rng.integers(0, 16))
            price = float(rng.uniform(-100.0, 200.0))
            qty = float(rng.uniform(0.1, 50.0))
            # Round qty to tick grid so encoding is exact.
            qty = round(qty / _CFG.qty_tick) * _CFG.qty_tick
            price = round(price / _CFG.price_tick) * _CFG.price_tick
            rows.append(_row(int(b), int(s), price, qty))

        tape = _tape(rows)
        for aid in range(16):
            v = float(compute_agent_pnl(tape, jnp.int32(aid)))
            ref = _pnl_reference_eur(rows, aid)
            assert v == pytest.approx(ref, rel=1e-4, abs=1e-2), (
                f"agent {aid}: impl={v} vs ref={ref}"
            )


# ===========================================================================
# (A) / (I) Trace-mode parity and vmap
# ===========================================================================

class TestVmapParity:
    """jax.vmap over agent_id under enable_x64 context — Option 3a contract."""

    def _build_tape(self):
        rows = [
            _row(0, 1, 50.0, 10.0),
            _row(2, 0, 45.5, 3.2),
            _row(1, 2, -12.5, 7.7),
            _row(3, 0, 9999.0, 999.0),
            _row(0, 3, -9999.0, 500.0),
            _row(0, 0, 30.0, 1.0),  # self-match => contributes zero
        ]
        return rows, _tape(rows)

    def test_vmap_matches_eager_under_enable_x64(self):
        rows, tape = self._build_tape()
        agent_ids = jnp.arange(6, dtype=jnp.int32)

        # Eager reference — one call per agent.
        eager = jnp.stack(
            [compute_agent_pnl(tape, aid) for aid in agent_ids]
        )

        with enable_x64():
            vmapped = jax.vmap(
                compute_agent_pnl, in_axes=(None, 0)
            )(tape, agent_ids)

        assert vmapped.dtype == jnp.float32
        assert vmapped.shape == (6,)
        # Bitwise-close: both paths share the same int64 accumulator and
        # float64 decode; rounding to float32 should be the same.
        np.testing.assert_allclose(
            np.asarray(vmapped), np.asarray(eager), rtol=1e-6, atol=1e-3,
        )

    def test_vmap_zero_sum_over_all_agents(self):
        # Sum of PnL over ALL agents on a closed tape must be ~zero.
        rows, tape = self._build_tape()
        # Universe of distinct agent_ids appearing in the tape.
        aids = sorted({r[_COL_BUYER] for r in rows}
                      | {r[_COL_SELLER] for r in rows})
        agent_ids = jnp.asarray(aids, dtype=jnp.int32)

        with enable_x64():
            pnls = jax.vmap(compute_agent_pnl, in_axes=(None, 0))(
                tape, agent_ids
            )

        envelope = 999.0 * 9999.0 * 0.25
        assert abs(float(jnp.sum(pnls))) <= envelope * 1e-5

    def test_self_match_row_contributes_zero(self):
        # A row where buyer == seller == agent_id must contribute exactly
        # zero (sign = +1 - 1 = 0). Not allowed by market semantics in
        # practice, but the primitive must handle it cleanly.
        rows = [_row(7, 7, 50.0, 10.0)]
        v = compute_agent_pnl(_tape(rows), jnp.int32(7))
        assert float(v) == 0.0


# ===========================================================================
# Shape / tree hygiene on vmap output
# ===========================================================================

class TestShapeHygiene:
    def test_vmap_batched_output_is_1d_float32(self):
        rows = [_row(1, 2, 50.0, 10.0), _row(1, 3, 20.0, 5.0)]
        tape = _tape(rows)
        with enable_x64():
            out = jax.vmap(compute_agent_pnl, in_axes=(None, 0))(
                tape, jnp.asarray([1, 2, 3], dtype=jnp.int32),
            )
        assert out.shape == (3,)
        assert out.dtype == jnp.float32
        # PyTree structure: just a single array leaf.
        leaves = jax.tree_util.tree_leaves(out)
        assert len(leaves) == 1

    def test_2d_vmap_over_tapes_and_agents(self):
        # Stack two tapes with shared shape, vmap agent_id outer. This
        # exercises the traced path at higher rank and confirms no shape
        # leak from internal reductions.
        tape_a = _tape([_row(1, 2, 50.0, 10.0), _row(0, 0, 0.0, 0.0)])
        tape_b = _tape([_row(2, 1, 10.0, 5.0), _row(3, 1, 20.0, 1.5)])
        stacked = jnp.stack([tape_a, tape_b], axis=0)  # (2, M, 6)

        with enable_x64():
            # Outer vmap over tapes, inner vmap over agents.
            per_tape = jax.vmap(
                lambda t: jax.vmap(compute_agent_pnl, in_axes=(None, 0))(
                    t, jnp.asarray([1, 2, 3], dtype=jnp.int32),
                )
            )(stacked)

        assert per_tape.shape == (2, 3)
        assert per_tape.dtype == jnp.float32
