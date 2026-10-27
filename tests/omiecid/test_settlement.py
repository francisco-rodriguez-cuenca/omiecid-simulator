"""Tests for M5.1 — Trade Value + Agent PnL (Settlement).

Behavioral contracts for ``compute_trade_value`` and ``compute_agent_pnl``
in ``omiecid_simulator.omiecid.settlement``.

All expected values are derived from:
  - docs/omiecid/market-spec.md §2.3 (trade value formula: V = Q x p x d)
  - docs/omiecid/market-spec.md §8 (settlement and PnL rules)
  - docs/omiecid/migration-plan.md §"Settlement API pin" (P2-L2, v5.5)
  - docs/omiecid/agile-plan.md §P2-L2 (v5.7)

Domain rules:
  - Trade value: V (EUR) = Q (MW) x p (EUR/MWh) x d (h).
  - For 15-min contracts: d = 0.25 h (default).
  - Bilateral PnL: seller leg gets +V, buyer leg gets -V.
  - Buying and selling same qty at same price -> PnL = 0.

Post-P2-L2 note:
  ``compute_agent_pnl`` is tape-centric: it takes an ``(M, 6) int32`` trade
  tape whose ``price`` column is **offset-encoded** via ``encode_price``
  (migration-plan §Decoding contract, v5.4). The Phase-1 list-of-tuples
  signature was retired; tests here build a synthetic tape via
  ``encode_price(p_eur, OMIEProductConfig())`` and pin a sentinel
  counterparty on the non-agent leg.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.encoding import encode_price
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.settlement import compute_agent_pnl, compute_trade_value


# ---------------------------------------------------------------------------
# Tape helpers (per migration-plan §Settlement API pin, v5.5)
# ---------------------------------------------------------------------------

_CFG = OMIEProductConfig()
_QTY_TICK = _CFG.qty_tick                 # 0.1 MW
_DURATION = _CFG.delivery_duration_hours  # 0.25 h

AGENT = 1          # "this" agent under test (seller or buyer per call)
COUNTERPARTY = 2   # the opposite leg — constant sentinel, irrelevant to AGENT's PnL


def _qty_ticks(qty_mw: float) -> int:
    return round(qty_mw / _QTY_TICK)


def _row(price_eur: float, qty_mw: float, buyer: int, seller: int) -> list[int]:
    """Build one (6,) tape row using the canonical offset encoder."""
    return [0, encode_price(price_eur, _CFG), _qty_ticks(qty_mw), buyer, seller, 0]


def _tape_from_legs(
    sell_trades: list[tuple[float, float, float]],
    buy_trades: list[tuple[float, float, float]],
) -> jnp.ndarray:
    """Rebuild an (M, 6) int32 tape from the retired list-of-tuples inputs.

    ``sell_trades`` / ``buy_trades`` items are ``(qty_mw, price_eur, duration_h)``.
    Duration is read from ``OMIEProductConfig().delivery_duration_hours`` (0.25)
    in the production path — the per-row ``duration_h`` column is dropped here
    and each row must carry ``duration_h == _DURATION`` to preserve the
    Phase-1 semantics.
    """
    rows: list[list[int]] = []
    for qty_mw, price_eur, d in sell_trades:
        assert d == _DURATION, (
            f"Phase-2 tape carries a single fixed delivery duration "
            f"({_DURATION}h); got {d}h — test must match."
        )
        # AGENT is seller on this leg.
        rows.append(_row(price_eur, qty_mw, buyer=COUNTERPARTY, seller=AGENT))
    for qty_mw, price_eur, d in buy_trades:
        assert d == _DURATION, (
            f"Phase-2 tape carries a single fixed delivery duration "
            f"({_DURATION}h); got {d}h — test must match."
        )
        # AGENT is buyer on this leg.
        rows.append(_row(price_eur, qty_mw, buyer=AGENT, seller=COUNTERPARTY))
    if not rows:
        # Empty-tape fallback: one zero-qty padding row that contributes 0.
        rows.append([0, 0, 0, -1, -1, 0])
    return jnp.asarray(rows, dtype=jnp.int32)


def _pnl(
    sell_trades: list[tuple[float, float, float]] | None = None,
    buy_trades: list[tuple[float, float, float]] | None = None,
) -> float:
    """Back-compat shim: rebuild the tape and invoke the new API for AGENT."""
    tape = _tape_from_legs(sell_trades or [], buy_trades or [])
    return float(compute_agent_pnl(tape, jnp.int32(AGENT)))


# ---------------------------------------------------------------------------
# 1. test_trade_value_15min — acceptance criterion
# ---------------------------------------------------------------------------

class TestTradeValue15Min:
    """V = Q x p x d; for 15-min contracts d = 0.25h (default)."""

    def test_hand_computed_example(self):
        """10 MW x 50 EUR/MWh x 0.25h = 125.00 EUR (migration-plan §M5)."""
        result = compute_trade_value(10.0, 50.0)
        assert float(result) == pytest.approx(125.0)

    def test_default_duration_is_quarter_hour(self):
        """Default duration_hours must be 0.25 (15-min contract)."""
        result = compute_trade_value(1.0, 100.0)
        assert float(result) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# 2. test_pnl_net_zero_for_self_trade — acceptance criterion
# ---------------------------------------------------------------------------

class TestPnlNetZeroForSelfTrade:
    """Buying and selling same qty at same price -> PnL = 0."""

    def test_symmetric_trade(self):
        """Buy 10 MW at 50, sell 10 MW at 50 -> PnL = 0."""
        sell_trades = [(10.0, 50.0, 0.25)]
        buy_trades = [(10.0, 50.0, 0.25)]
        result = _pnl(sell_trades, buy_trades)
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_symmetric_multiple_trades(self):
        """Multiple buys and sells at same prices cancel out."""
        sell_trades = [(5.0, 50.0, 0.25), (5.0, 50.0, 0.25)]
        buy_trades = [(5.0, 50.0, 0.25), (5.0, 50.0, 0.25)]
        result = _pnl(sell_trades, buy_trades)
        assert result == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 3. test_pnl_profit_on_sell — acceptance criterion
# ---------------------------------------------------------------------------

class TestPnlProfitOnSell:
    """Buy low, sell high -> positive PnL."""

    def test_buy_at_40_sell_at_50(self):
        """Buy 10 MW at 40, sell 10 MW at 50 -> PnL = +25 EUR."""
        sell_trades = [(10.0, 50.0, 0.25)]
        buy_trades = [(10.0, 40.0, 0.25)]
        result = _pnl(sell_trades, buy_trades)
        assert result == pytest.approx(25.0, rel=1e-5)
        assert result > 0


# ---------------------------------------------------------------------------
# 4. test_pnl_loss_on_buy — acceptance criterion
# ---------------------------------------------------------------------------

class TestPnlLossOnBuy:
    """Buy high, sell low -> negative PnL."""

    def test_buy_at_60_sell_at_50(self):
        """Buy 10 MW at 60, sell 10 MW at 50 -> PnL = -25 EUR."""
        sell_trades = [(10.0, 50.0, 0.25)]
        buy_trades = [(10.0, 60.0, 0.25)]
        result = _pnl(sell_trades, buy_trades)
        assert result == pytest.approx(-25.0, rel=1e-5)
        assert result < 0


# ---------------------------------------------------------------------------
# 5. Edge case: explicit duration
# ---------------------------------------------------------------------------

class TestTradeValueExplicitDuration:
    """Trade value with non-default duration_hours (scalar primitive)."""

    def test_one_hour_duration(self):
        """10 MW x 50 EUR/MWh x 1.0h = 500 EUR."""
        result = compute_trade_value(10.0, 50.0, 1.0)
        assert float(result) == pytest.approx(500.0)

    def test_half_hour_duration(self):
        """10 MW x 50 EUR/MWh x 0.5h = 250 EUR."""
        result = compute_trade_value(10.0, 50.0, 0.5)
        assert float(result) == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# 6. Edge case: zero quantity
# ---------------------------------------------------------------------------

class TestTradeValueZeroQuantity:
    """Zero quantity produces zero trade value."""

    def test_zero_quantity(self):
        result = compute_trade_value(0.0, 50.0)
        assert float(result) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. Edge case: negative price (valid in OMIE: min = -9999 EUR/MWh)
# ---------------------------------------------------------------------------

class TestTradeValueNegativePrice:
    """Negative prices are valid in OMIE (market-spec §2.3)."""

    def test_negative_price(self):
        """10 MW x (-50) EUR/MWh x 0.25h = -125 EUR."""
        result = compute_trade_value(10.0, -50.0)
        assert float(result) == pytest.approx(-125.0)


# ---------------------------------------------------------------------------
# 8. Edge case: empty trade lists / one-sided books
# ---------------------------------------------------------------------------

class TestPnlEmptyTrades:
    """No trades -> PnL = 0; one-sided books preserve sign."""

    def test_no_sells_no_buys(self):
        """Empty sell and buy lists -> PnL = 0."""
        result = _pnl([], [])
        assert result == pytest.approx(0.0)

    def test_only_sells(self):
        """Sells only -> PnL = +V (positive)."""
        result = _pnl(sell_trades=[(10.0, 50.0, 0.25)])
        assert result == pytest.approx(125.0, rel=1e-5)
        assert result > 0

    def test_only_buys(self):
        """Buys only -> PnL = -V (negative)."""
        result = _pnl(buy_trades=[(10.0, 50.0, 0.25)])
        assert result == pytest.approx(-125.0, rel=1e-5)
        assert result < 0


# ---------------------------------------------------------------------------
# 9. Edge case: multiple trades aggregation
# ---------------------------------------------------------------------------

class TestPnlMultipleTrades:
    """PnL aggregates across multiple trades."""

    def test_multiple_sells_and_buys(self):
        """sells: (5 MW x 55 x 0.25 = 68.75), (3 MW x 60 x 0.25 = 45.00)
        buys:  (5 MW x 45 x 0.25 = 56.25), (3 MW x 50 x 0.25 = 37.50)
        PnL = 113.75 - 93.75 = 20.00
        """
        sell_trades = [(5.0, 55.0, 0.25), (3.0, 60.0, 0.25)]
        buy_trades = [(5.0, 45.0, 0.25), (3.0, 50.0, 0.25)]
        result = _pnl(sell_trades, buy_trades)
        assert result == pytest.approx(20.0, rel=1e-5)


# ---------------------------------------------------------------------------
# 10. Edge case: minimum/maximum tick values
# ---------------------------------------------------------------------------

class TestTradeValueMinTickValues:
    """Trade value at minimum/maximum tick sizes (market-spec §2.3)."""

    def test_minimum_quantity_and_price_tick(self):
        """0.1 MW x 0.01 EUR/MWh x 0.25h = 0.00025 EUR."""
        result = compute_trade_value(0.1, 0.01)
        assert float(result) == pytest.approx(0.00025, abs=1e-9)

    def test_maximum_quantity_and_price(self):
        """999 MW x 9999 EUR/MWh x 0.25h — OMIE ceiling."""
        result = compute_trade_value(999.0, 9999.0)
        assert float(result) == pytest.approx(999.0 * 9999.0 * 0.25, rel=1e-5)
