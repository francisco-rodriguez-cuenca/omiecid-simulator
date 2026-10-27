"""Tests for M1.1 — OMIE Configuration Dataclasses.

Behavioral contracts for OMIEProductConfig and OMIEMarketConfig.
All expected values are derived from docs/omiecid/market-spec.md §2.1–2.3
and docs/omiecid/migration-plan.md §M1.1.

These tests are black-box: they import only the public interface and
assert against domain-specified defaults.
"""

import pytest
from dataclasses import FrozenInstanceError

from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig


# ---------------------------------------------------------------------------
# test_config_defaults — verify default values match market-spec §2.3
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """Default field values must match OMIE market rules exactly."""

    # -- OMIEProductConfig defaults (market-spec §2.1, §2.3) --

    def test_mtu_minutes(self, product_config):
        """MTU is 15 minutes for quarter-hourly product (§2.1)."""
        assert product_config.mtu_minutes == 15

    def test_price_min(self, product_config):
        """Minimum price is -9999 EUR/MWh (§2.3)."""
        assert product_config.price_min == -9999.0

    def test_price_max(self, product_config):
        """Maximum price is +9999 EUR/MWh (§2.3)."""
        assert product_config.price_max == 9999.0

    def test_price_tick(self, product_config):
        """Price tick is 0.01 EUR/MWh — two decimal places (§2.3)."""
        assert product_config.price_tick == 0.01

    def test_qty_min(self, product_config):
        """Minimum quantity is 0.1 MW (§2.3)."""
        assert product_config.qty_min == 0.1

    def test_qty_max(self, product_config):
        """Maximum quantity is 999 MW (§2.3)."""
        assert product_config.qty_max == 999.0

    def test_qty_tick(self, product_config):
        """Quantity tick is 0.1 MW — one decimal place (§2.3)."""
        assert product_config.qty_tick == 0.1

    def test_delivery_duration_hours(self, product_config):
        """Delivery duration is 0.25 h (15 min) for trade value formula (§2.3)."""
        assert product_config.delivery_duration_hours == 0.25

    # -- OMIEMarketConfig defaults (market-spec §2.2, migration-plan §M1.1) --

    def test_num_contracts(self, market_config):
        """96 quarter-hourly contracts per delivery day (§2.2)."""
        assert market_config.num_contracts == 96

    def test_num_agents(self, market_config):
        """Default number of agents is 2 (migration-plan §M1.1)."""
        assert market_config.num_agents == 2

    def test_book_depth(self, market_config):
        """Default book depth is 100 orders per side (migration-plan §M1.1)."""
        assert market_config.book_depth == 100


# ---------------------------------------------------------------------------
# test_config_immutability — verify dataclasses are frozen
# ---------------------------------------------------------------------------

class TestConfigImmutability:
    """Both config dataclasses must be frozen (immutable after creation)."""

    def test_product_config_frozen(self, product_config):
        """OMIEProductConfig must reject attribute assignment."""
        with pytest.raises(FrozenInstanceError):
            product_config.price_min = 0.0

    def test_market_config_frozen(self, market_config):
        """OMIEMarketConfig must reject attribute assignment."""
        with pytest.raises(FrozenInstanceError):
            market_config.num_contracts = 48


# ---------------------------------------------------------------------------
# test_price_tick_precision — verify tick divides price range evenly
# ---------------------------------------------------------------------------

class TestPriceTickPrecision:
    """Price and quantity ticks must divide their respective ranges
    into a whole number of steps, ensuring no fractional-tick prices
    or quantities are forced by the range boundaries."""

    def test_price_range_divisible_by_tick(self, product_config):
        """The price range [-9999, +9999] must be evenly divisible by 0.01."""
        price_range = product_config.price_max - product_config.price_min
        # Number of ticks must be an integer
        n_ticks = price_range / product_config.price_tick
        assert n_ticks == int(n_ticks), (
            f"Price range {price_range} not evenly divisible by "
            f"tick {product_config.price_tick}: {n_ticks} ticks"
        )
        assert n_ticks > 0

    def test_qty_range_divisible_by_tick(self, product_config):
        """The quantity range [0.1, 999] must be evenly divisible by 0.1."""
        qty_range = product_config.qty_max - product_config.qty_min
        n_ticks = qty_range / product_config.qty_tick
        assert n_ticks == pytest.approx(int(round(n_ticks))), (
            f"Quantity range {qty_range} not evenly divisible by "
            f"tick {product_config.qty_tick}: {n_ticks} ticks"
        )
        assert n_ticks > 0

    def test_price_boundaries_on_tick(self, product_config):
        """Both price_min and price_max must themselves sit on a tick boundary
        (i.e., be expressible as integer multiples of price_tick).

        Uses division-based check instead of modulo to avoid IEEE 754
        float modulo precision issues with large dividends.
        """
        for value in (product_config.price_min, product_config.price_max):
            ratio = value / product_config.price_tick
            assert ratio == pytest.approx(round(ratio), abs=1e-9), (
                f"{value} is not on a {product_config.price_tick} tick boundary"
            )

    def test_qty_boundaries_on_tick(self, product_config):
        """Both qty_min and qty_max must sit on a tick boundary.

        Uses division-based check instead of modulo to avoid IEEE 754
        float modulo precision issues with large dividends.
        """
        for value in (product_config.qty_min, product_config.qty_max):
            ratio = value / product_config.qty_tick
            assert ratio == pytest.approx(round(ratio), abs=1e-9), (
                f"{value} is not on a {product_config.qty_tick} tick boundary"
            )
