"""Behavioral contracts for P2-L2 — ``settlement`` JIT-safe.

Story: P2-L2 (agile-plan.md §"Story P2-L2 — `settlement` JIT-safe", v5.6).

Tests the two public settlement primitives:

  1. ``compute_trade_value(Q_mw: f32, p_eur_mwh: f32, d_h: f32 = 0.25) -> f32``
     — scalar JAX-traceable primitive. 0-d ``jnp.float32`` return. Phase-1
     semantics unchanged (scalar decoded-float signature).

  2. ``compute_agent_pnl(trades: Int32[M, 6], agent_id: i32) -> f32``
     — tape-centric JAX-traceable API. Operates directly on the
     **offset-encoded** int32 ``price`` column of the ``(M, 6)`` tape
     produced by the loader. No per-row ``decode_price`` call: the
     decomposition ``p_eur = p_tick_offset × price_tick + price_min``
     is done once internally, via an int64 two-term accumulator:

         pnl_ticks_offset = sum(sign × Q_ticks × p_tick_offset) [int64]
         pnl_ticks_min    = sum(sign × Q_ticks)                 [int64]
         pnl_eur = (pnl_ticks_offset × price_tick × qty_tick × d_h)
                 + (pnl_ticks_min    × price_min  × qty_tick × d_h)

     The float32 boundary is at the returned PnL only.

Trade tape layout (per P2-L2 story card + migration-plan §"Settlement API pin"):
    columns = [timestamp, price, quantity_tick,
               buyer_agent_id, seller_agent_id, contract_id]

    where ``price`` is the **offset-encoded** int32
    ``encode_price(p_eur) = round((p_eur - price_min) / price_tick)``.

Bilateral zero-sum invariant (market-spec v2.4 §2.3 + §8):

    sum_over_agents(compute_agent_pnl(tape, aid)) == 0

over any closed trade book (every trade's buyer and seller are present
in ``agent_ids``). There is no simulator/third-party revenue term.

Test file is black-box: imports only the public interface of
``omiecid_simulator.omiecid.settlement`` plus the canonical encoder.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.omiecid.encoding import encode_price
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.settlement import (
    compute_agent_pnl,
    compute_trade_value,
)

# Module-level marker (Story T1.1): every test in this file traces or
# compiles JAX code (settlement primitives).  Marker pinned at module
# scope per the T1.1 lint discipline.
pytestmark = pytest.mark.jit


# ---------------------------------------------------------------------------
# Tape layout constants (per P2-L2 story card + migration-plan §Settlement API pin)
# ---------------------------------------------------------------------------

# Column indices into the (M, 6) int32 trade tape.
COL_TIMESTAMP = 0
COL_PRICE = 1             # offset-encoded int32 (encode_price)
COL_QUANTITY_TICK = 2
COL_BUYER_AGENT_ID = 3
COL_SELLER_AGENT_ID = 4
COL_CONTRACT_ID = 5

# OMIE default grid (OMIEProductConfig defaults).
_CFG = OMIEProductConfig()
QTY_TICK = _CFG.qty_tick                   # 0.1 MW per tick
PRICE_TICK = _CFG.price_tick               # 0.01 EUR/MWh per tick
PRICE_MIN = _CFG.price_min                 # -9999.0 EUR/MWh
DURATION_HOURS = _CFG.delivery_duration_hours  # 0.25

# Sentinel for padding rows (quantity_tick=0 zeros out contribution regardless
# of agent_id matching logic).
PAD_AGENT_ID = -1


# ---------------------------------------------------------------------------
# Helpers: build small synthetic tapes using the canonical offset encoder
# ---------------------------------------------------------------------------


def _enc(price_eur: float) -> int:
    """Canonical offset-encoded price column value."""
    return encode_price(price_eur, _CFG)


def _row(
    *,
    ts: int = 0,
    price_eur: float,
    quantity_tick: int,
    buyer: int,
    seller: int,
    contract_id: int = 0,
) -> list[int]:
    """Build one (6,) trade-tape row.

    ``price_eur`` is the *decoded* EUR/MWh price; the helper runs it
    through ``encode_price`` to populate the offset-encoded ``price`` column
    that ``compute_agent_pnl`` expects per the pinned contract.
    """
    return [ts, _enc(price_eur), quantity_tick, buyer, seller, contract_id]


def _row_raw_price(
    *,
    ts: int = 0,
    price_col: int,
    quantity_tick: int,
    buyer: int,
    seller: int,
    contract_id: int = 0,
) -> list[int]:
    """Build a row with an already-encoded int price column.

    Used by padding rows (where we explicitly want price=0) and by the
    int64-overflow probes at the offset-encoded envelope bound.
    """
    return [ts, price_col, quantity_tick, buyer, seller, contract_id]


def _tape(rows: list[list[int]]) -> jnp.ndarray:
    """Stack rows into an (M, 6) int32 jnp array."""
    return jnp.asarray(rows, dtype=jnp.int32)


def _empty_tape(m: int = 4) -> jnp.ndarray:
    """Build an empty padded (M, 6) int32 tape with zero-qty rows."""
    rows = [
        _row_raw_price(
            price_col=0, quantity_tick=0, buyer=PAD_AGENT_ID, seller=PAD_AGENT_ID
        )
        for _ in range(m)
    ]
    return _tape(rows)


def _expected_pnl(
    trades: list[tuple[float, int, int, int]],  # (price_eur, qty_tick, buyer, seller)
    agent_id: int,
) -> float:
    """Independent reference implementation of compute_agent_pnl.

    Uses decoded EUR prices and the Phase-1 ``V = Q_mw × p_eur × d_h`` formula,
    with sign = +1 on seller leg, -1 on buyer leg.
    """
    total = 0.0
    for price_eur, qty_tick, buyer, seller in trades:
        q_mw = qty_tick * QTY_TICK
        v = q_mw * price_eur * DURATION_HOURS
        if agent_id == seller:
            total += v
        if agent_id == buyer:
            total -= v
    return total


# ===========================================================================
# SECTION 1 — compute_trade_value: scalar primitive contracts (Category B)
# ===========================================================================


class TestTradeValueContract:
    """Phase-1 parity + scalar dtype contract for ``compute_trade_value``."""

    def test_hand_computed_15min_example(self):
        """10 MW × 50 EUR/MWh × 0.25h = 125.00 EUR (migration-plan §M5)."""
        v = compute_trade_value(10.0, 50.0)
        assert float(v) == pytest.approx(125.0)

    def test_default_duration_is_quarter_hour(self):
        """Default ``d_h`` = 0.25."""
        v = compute_trade_value(1.0, 100.0)
        assert float(v) == pytest.approx(25.0)

    def test_explicit_one_hour_duration(self):
        """10 MW × 50 × 1.0h = 500 EUR."""
        v = compute_trade_value(10.0, 50.0, 1.0)
        assert float(v) == pytest.approx(500.0)

    def test_half_hour_duration(self):
        """10 MW × 50 × 0.5h = 250 EUR."""
        v = compute_trade_value(10.0, 50.0, 0.5)
        assert float(v) == pytest.approx(250.0)

    def test_zero_quantity_yields_zero(self):
        v = compute_trade_value(0.0, 50.0)
        assert float(v) == pytest.approx(0.0)

    def test_negative_price_valid(self):
        """OMIE admits negative prices (market-spec §2.3, min = -9999)."""
        v = compute_trade_value(10.0, -50.0)
        assert float(v) == pytest.approx(-125.0)

    def test_minimum_tick_values(self):
        """0.1 MW × 0.01 EUR/MWh × 0.25h = 0.00025 EUR."""
        v = compute_trade_value(0.1, 0.01)
        assert float(v) == pytest.approx(0.00025, abs=1e-9)

    def test_maximum_omie_bounds(self):
        """999 MW × 9999 EUR/MWh × 0.25h — phase-1 parity at the OMIE ceiling."""
        v = compute_trade_value(999.0, 9999.0)
        assert float(v) == pytest.approx(999.0 * 9999.0 * 0.25, rel=1e-5)


class TestTradeValueDtypeAndScalar:
    """Return is a 0-d ``jnp.float32`` scalar, JAX-traceable."""

    def test_returns_jax_array(self):
        v = compute_trade_value(10.0, 50.0)
        assert isinstance(v, jnp.ndarray), f"expected jnp.ndarray, got {type(v)}"

    def test_returns_zero_dim(self):
        v = compute_trade_value(10.0, 50.0)
        assert jnp.asarray(v).ndim == 0

    def test_returns_float32(self):
        v = compute_trade_value(10.0, 50.0)
        assert jnp.asarray(v).dtype == jnp.float32


class TestTradeValueJitCompiles:
    """``jax.jit(compute_trade_value)`` must compile and produce a 0-d float32."""

    def test_jit_compiles_and_matches(self):
        jitted = jax.jit(compute_trade_value)
        v = jitted(10.0, 50.0, 0.25)
        assert jnp.asarray(v).dtype == jnp.float32
        assert jnp.asarray(v).ndim == 0
        assert float(v) == pytest.approx(125.0)

    def test_jit_default_duration(self):
        jitted = jax.jit(compute_trade_value)
        v = jitted(1.0, 100.0)
        assert float(v) == pytest.approx(25.0)


# ===========================================================================
# SECTION 2 — compute_agent_pnl: tape-centric API shape/dtype contracts
# ===========================================================================


class TestAgentPnlShapeAndDtype:
    """``compute_agent_pnl(trades, agent_id)`` returns 0-d jnp.float32."""

    def test_returns_jax_array(self):
        tape = _empty_tape(4)
        out = compute_agent_pnl(tape, jnp.int32(0))
        assert isinstance(out, jnp.ndarray)

    def test_returns_zero_dim(self):
        tape = _empty_tape(4)
        out = compute_agent_pnl(tape, jnp.int32(0))
        assert jnp.asarray(out).ndim == 0

    def test_returns_float32(self):
        tape = _empty_tape(4)
        out = compute_agent_pnl(tape, jnp.int32(0))
        assert jnp.asarray(out).dtype == jnp.float32

    def test_accepts_python_int_agent_id(self):
        """agent_id may be passed as a Python int, promoted internally."""
        tape = _empty_tape(4)
        out = compute_agent_pnl(tape, jnp.int32(0))
        assert jnp.asarray(out).dtype == jnp.float32


class TestAgentPnlVmapUnderX64:
    """Per Option 3a (migration-plan §Phase 2 — Settlement API pin
    §jit-contract adjudication, v5.6; agile-plan §P2-L2 v5.8 item (g)):

    ``compute_agent_pnl`` no longer carries a public ``jax.jit`` contract.
    The supported traced-usage contract is ``jax.vmap`` under an
    ``enable_x64`` context, which preserves the int64 two-term accumulator
    across the trace. Prior ``TestAgentPnlJitCompiles`` tests have been
    removed.
    """

    def test_vmap_under_enable_x64_context_compiles_and_zero_sums(self):
        """vmap over agent_id under ``jax.experimental.enable_x64()``.

        Symmetric 2-agent, 1-trade closed book → exact zero-sum on the
        ``(2,) float32`` result (IEEE-754 negation is exact for the
        symmetric case; precedent: ``test_zero_sum_two_agents_single_trade``
        / ``test_single_matched_trade_two_agents`` above).
        """
        tape = _tape([_row(price_eur=50.0, quantity_tick=100, buyer=0, seller=1)])
        agent_ids = jnp.asarray([0, 1], dtype=jnp.int32)
        with jax.experimental.enable_x64():
            pnls = jax.vmap(compute_agent_pnl, in_axes=(None, 0))(tape, agent_ids)
        assert pnls.shape == (2,)
        assert pnls.dtype == jnp.float32
        total = float(jnp.sum(pnls))
        assert total == 0.0, f"expected exact 0.0, got {total}"


# ===========================================================================
# SECTION 3 — Bilateral zero-sum invariant (market-spec §2.3 + §8)
# ===========================================================================


class TestBilateralZeroSumInvariant:
    """sum_over_agents(compute_agent_pnl(tape, aid)) == 0 — exactly."""

    def test_single_matched_trade_two_agents(self):
        """2 agents, 1 matched trade: exact cancellation.

        Int64 two-term accumulation makes (+V, −V) cancel exactly in int64
        tick space before the float32 decode boundary.
        """
        tape = _tape([_row(price_eur=50.0, quantity_tick=100, buyer=0, seller=1)])
        agent_ids = jnp.asarray([0, 1], dtype=jnp.int32)
        pnls = jax.vmap(compute_agent_pnl, in_axes=(None, 0))(tape, agent_ids)
        total = float(jnp.sum(pnls))
        assert total == 0.0, f"expected exact 0.0, got {total}"

    def test_multi_trade_closed_book_three_agents(self):
        """Three agents, three trades. Each trade is a closed transfer."""
        tape = _tape([
            _row(price_eur=50.0, quantity_tick=100, buyer=0, seller=1),
            _row(price_eur=62.0, quantity_tick=50, buyer=2, seller=0),
            _row(price_eur=48.0, quantity_tick=75, buyer=1, seller=2),
        ])
        agent_ids = jnp.asarray([0, 1, 2], dtype=jnp.int32)
        pnls = jax.vmap(compute_agent_pnl, in_axes=(None, 0))(tape, agent_ids)
        total = float(jnp.sum(pnls))
        assert total == 0.0, f"expected exact 0.0, got {total}"

    def test_zero_sum_with_negative_prices(self):
        """Closed trade book with negative prices still sums to zero."""
        tape = _tape([
            _row(price_eur=-30.0, quantity_tick=100, buyer=0, seller=1),
            _row(price_eur=-5.0, quantity_tick=50, buyer=1, seller=0),
        ])
        agent_ids = jnp.asarray([0, 1], dtype=jnp.int32)
        pnls = jax.vmap(compute_agent_pnl, in_axes=(None, 0))(tape, agent_ids)
        total = float(jnp.sum(pnls))
        assert total == 0.0, f"expected exact 0.0, got {total}"

    def test_zero_sum_many_agents_many_trades(self):
        """Randomly generated asymmetric closed trade book; sum ~= 0 within float32 ULP.

        Fixed RNG seed for determinism; all trades matched between agents
        in ``[0, N_AGENTS)``. Decoded prices drawn from the valid OMIE range.

        Note on tolerance: the int64 two-term accumulation makes zero-sum
        exact in int64 tick space, but ``compute_agent_pnl`` returns
        ``float32`` at the Python boundary (migration-plan §Settlement API
        pin, v5.5). For **asymmetric** closed books the per-agent float32
        PnLs no longer cancel bitwise — residual is ~1 ULP at the per-agent
        PnL scale. The other four zero-sum tests in this class are
        symmetric (2-agent, ceiling-symmetric, padding, negative-price)
        where IEEE-754 negation is exact, so they stay strict
        ``== 0.0``. This asymmetric case is relaxed to the same float32
        tolerance (1e-3 EUR) used elsewhere in the file.
        """
        rng = np.random.default_rng(seed=42)
        n_agents = 8
        n_trades = 20
        rows = []
        for _ in range(n_trades):
            b = int(rng.integers(0, n_agents))
            s = int(rng.integers(0, n_agents))
            if b == s:
                s = (s + 1) % n_agents
            price_eur = float(rng.integers(-2000, 8000)) * 0.01  # stay on tick grid
            qty = int(rng.integers(1, 500))
            rows.append(_row(price_eur=price_eur, quantity_tick=qty, buyer=b, seller=s))
        tape = _tape(rows)
        agent_ids = jnp.arange(n_agents, dtype=jnp.int32)
        pnls = jax.vmap(compute_agent_pnl, in_axes=(None, 0))(tape, agent_ids)
        total = float(jnp.sum(pnls))
        assert abs(total) < 1e-3, f"expected |total| < 1e-3 (float32 ULP), got {total}"

    def test_zero_sum_survives_padding_rows(self):
        """Padding rows (quantity_tick=0) must not perturb the invariant."""
        rows = [
            _row(price_eur=50.0, quantity_tick=100, buyer=0, seller=1),
            _row_raw_price(
                price_col=0, quantity_tick=0, buyer=PAD_AGENT_ID, seller=PAD_AGENT_ID
            ),
            _row_raw_price(
                price_col=0, quantity_tick=0, buyer=PAD_AGENT_ID, seller=PAD_AGENT_ID
            ),
        ]
        tape = _tape(rows)
        agent_ids = jnp.asarray([0, 1], dtype=jnp.int32)
        pnls = jax.vmap(compute_agent_pnl, in_axes=(None, 0))(tape, agent_ids)
        total = float(jnp.sum(pnls))
        assert total == 0.0


# ===========================================================================
# SECTION 4 — int64 internal accumulation: no overflow at OMIE bounds
# ===========================================================================


class TestInt64AccumulatorNoOverflow:
    """Internal accumulation must not overflow at the OMIE envelope.

    OMIE envelope in tick units:
      - quantity_tick max ~ 999 MW / 0.1 = 9990
      - offset-encoded price max ~ (9999 − (−9999)) / 0.01 = 1_999_800
      - product per trade ~ 9990 × 1_999_800 ≈ 2.0e10 > int32 max (~2.15e9).

    The story card pins int64-internal accumulation over a two-term
    decomposition (Q_ticks × p_tick_offset) and (Q_ticks) separately.
    """

    def test_single_trade_at_omie_ceiling(self):
        """One trade at 999 MW × 9999 EUR/MWh × 0.25h."""
        tape = _tape([
            _row(price_eur=9999.0, quantity_tick=9990, buyer=0, seller=1),
        ])
        expected = 999.0 * 9999.0 * 0.25  # EUR
        pnl_seller = float(compute_agent_pnl(tape, jnp.int32(1)))
        pnl_buyer = float(compute_agent_pnl(tape, jnp.int32(0)))
        assert pnl_seller == pytest.approx(expected, rel=1e-5)
        assert pnl_buyer == pytest.approx(-expected, rel=1e-5)

    def test_aggregate_exceeds_int32_range(self):
        """Sum of several near-ceiling trades exceeds int32 tick-product space.

        Ten trades at the OMIE ceiling — aggregate per-term products are
        comfortably outside int32 range.
        """
        n_trades = 10
        rows = [
            _row(price_eur=9999.0, quantity_tick=9990, buyer=0, seller=1)
            for _ in range(n_trades)
        ]
        tape = _tape(rows)
        expected_per = 999.0 * 9999.0 * 0.25
        pnl_seller = float(compute_agent_pnl(tape, jnp.int32(1)))
        assert pnl_seller == pytest.approx(n_trades * expected_per, rel=1e-5)

    def test_aggregate_zero_sum_at_ceiling(self):
        """Bilateral zero-sum survives the overflow probe."""
        n_trades = 10
        rows = [
            _row(price_eur=9999.0, quantity_tick=9990, buyer=0, seller=1)
            for _ in range(n_trades)
        ]
        tape = _tape(rows)
        agent_ids = jnp.asarray([0, 1], dtype=jnp.int32)
        pnls = jax.vmap(compute_agent_pnl, in_axes=(None, 0))(tape, agent_ids)
        total = float(jnp.sum(pnls))
        assert total == 0.0, f"expected 0.0 at OMIE ceiling, got {total}"

    def test_negative_ceiling_no_overflow(self):
        """Near-negative-ceiling price (−9999 EUR/MWh) × 999 MW still exact.

        Offset-encoded price column here is ``encode_price(-9999) = 0`` —
        this also exercises the ``p_tick_offset = 0`` branch of the
        two-term decomposition where only the ``price_min`` term
        contributes.
        """
        tape = _tape([
            _row(price_eur=-9999.0, quantity_tick=9990, buyer=0, seller=1),
        ])
        expected = -999.0 * 9999.0 * 0.25
        pnl_seller = float(compute_agent_pnl(tape, jnp.int32(1)))
        assert pnl_seller == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# SECTION 5 — Phase 1 / 1.5 parity on agent PnL (Category C rewrites)
# ===========================================================================


class TestAgentPnlPhase1Parity:
    """Semantically equivalent Phase-1 scenarios, re-expressed on the tape."""

    def test_symmetric_self_trade_yields_zero(self):
        """Agent 0 sells then buys same qty at same price → PnL exactly 0."""
        tape = _tape([
            _row(price_eur=50.0, quantity_tick=100, buyer=1, seller=0),
            _row(price_eur=50.0, quantity_tick=100, buyer=0, seller=1),
        ])
        pnl = float(compute_agent_pnl(tape, jnp.int32(0)))
        assert pnl == 0.0

    def test_buy_at_40_sell_at_50_profit(self):
        """Buy 10 MW @ 40, sell 10 MW @ 50 → +25 EUR."""
        tape = _tape([
            _row(price_eur=40.0, quantity_tick=100, buyer=0, seller=1),
            _row(price_eur=50.0, quantity_tick=100, buyer=1, seller=0),
        ])
        pnl_0 = float(compute_agent_pnl(tape, jnp.int32(0)))
        assert pnl_0 == pytest.approx(25.0, rel=1e-5)
        assert pnl_0 > 0.0

    def test_buy_at_60_sell_at_50_loss(self):
        """Buy 10 MW @ 60, sell 10 MW @ 50 → −25 EUR."""
        tape = _tape([
            _row(price_eur=60.0, quantity_tick=100, buyer=0, seller=1),
            _row(price_eur=50.0, quantity_tick=100, buyer=1, seller=0),
        ])
        pnl_0 = float(compute_agent_pnl(tape, jnp.int32(0)))
        assert pnl_0 == pytest.approx(-25.0, rel=1e-5)
        assert pnl_0 < 0.0

    def test_no_trades_yields_zero(self):
        tape = _empty_tape(8)
        pnl = float(compute_agent_pnl(tape, jnp.int32(0)))
        assert pnl == 0.0

    def test_only_sells_positive(self):
        tape = _tape([
            _row(price_eur=50.0, quantity_tick=100, buyer=1, seller=0),  # 10 MW @ 50
        ])
        pnl = float(compute_agent_pnl(tape, jnp.int32(0)))
        assert pnl == pytest.approx(125.0, rel=1e-5)
        assert pnl > 0.0

    def test_only_buys_negative(self):
        tape = _tape([
            _row(price_eur=50.0, quantity_tick=100, buyer=0, seller=1),  # 10 MW @ 50
        ])
        pnl = float(compute_agent_pnl(tape, jnp.int32(0)))
        assert pnl == pytest.approx(-125.0, rel=1e-5)
        assert pnl < 0.0

    def test_multiple_trades_aggregated(self):
        """sells: (5 MW × 55 × 0.25 = 68.75), (3 MW × 60 × 0.25 = 45.00) = 113.75
        buys:  (5 MW × 45 × 0.25 = 56.25), (3 MW × 50 × 0.25 = 37.50) = 93.75
        PnL = 113.75 − 93.75 = 20.00
        """
        tape = _tape([
            _row(price_eur=55.0, quantity_tick=50, buyer=1, seller=0),
            _row(price_eur=60.0, quantity_tick=30, buyer=1, seller=0),
            _row(price_eur=45.0, quantity_tick=50, buyer=0, seller=1),
            _row(price_eur=50.0, quantity_tick=30, buyer=0, seller=1),
        ])
        pnl = float(compute_agent_pnl(tape, jnp.int32(0)))
        assert pnl == pytest.approx(20.0, rel=1e-5)

    def test_matches_reference_impl_on_mixed_book(self):
        """Parity against an independent decoded-price reference implementation.

        Exercises the two-term int64 decomposition on a mixed trade book
        that includes negative prices, asserting the production routine
        matches the decoded Phase-1 formula within float32 tolerance.
        """
        trades = [
            (50.0, 100, 0, 1),
            (-20.0, 50, 1, 0),
            (123.45, 37, 2, 0),
            (9999.0, 9, 0, 2),
        ]
        rows = [
            _row(price_eur=p, quantity_tick=q, buyer=b, seller=s)
            for (p, q, b, s) in trades
        ]
        tape = _tape(rows)
        for aid in [0, 1, 2, 99]:
            got = float(compute_agent_pnl(tape, jnp.int32(aid)))
            expected = _expected_pnl(trades, aid)
            assert got == pytest.approx(expected, rel=1e-5, abs=1e-3), (
                f"agent {aid}: got {got}, expected {expected}"
            )


# ===========================================================================
# SECTION 6 — agent_id isolation / absent agents
# ===========================================================================


class TestAgentIdIsolation:
    """Only rows touching ``agent_id`` contribute to that agent's PnL."""

    def test_absent_agent_has_zero_pnl(self):
        """An agent_id that never appears in the tape has PnL = 0."""
        tape = _tape([
            _row(price_eur=50.0, quantity_tick=100, buyer=0, seller=1),
            _row(price_eur=60.0, quantity_tick=50, buyer=2, seller=3),
        ])
        pnl_absent = float(compute_agent_pnl(tape, jnp.int32(99)))
        assert pnl_absent == 0.0

    def test_third_party_trades_do_not_leak(self):
        """Agent 0's PnL unaffected by trades between agents 1 and 2."""
        tape = _tape([
            _row(price_eur=50.0, quantity_tick=100, buyer=0, seller=1),  # agent 0 buys
            _row(price_eur=80.0, quantity_tick=200, buyer=1, seller=2),  # unrelated
            _row(price_eur=30.0, quantity_tick=400, buyer=2, seller=1),  # unrelated
        ])
        pnl_0 = float(compute_agent_pnl(tape, jnp.int32(0)))
        # Agent 0 only appears once, as buyer of 10 MW @ 50 → −125 EUR.
        assert pnl_0 == pytest.approx(-125.0, rel=1e-5)

    def test_zero_quantity_row_contributes_nothing(self):
        """A row with quantity_tick = 0 contributes 0 to any agent."""
        tape = _tape([
            _row(price_eur=50.0, quantity_tick=0, buyer=0, seller=1),
            _row(price_eur=50.0, quantity_tick=100, buyer=0, seller=1),
        ])
        pnl_0 = float(compute_agent_pnl(tape, jnp.int32(0)))
        assert pnl_0 == pytest.approx(-125.0, rel=1e-5)
