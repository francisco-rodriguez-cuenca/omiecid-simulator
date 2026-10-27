"""Tests for M7.3 — NPZ Cache Layer.

Behavioral contracts for the OMIE NPZ caching module (omie_cache.py).
The cache layer follows the jaxlobster pattern: first load parses raw
files and saves NPZ; subsequent loads read from cache directly.

Two required tests per migration-plan section M7.3:
  - test_cache_roundtrip: save to NPZ, reload, arrays are identical
  - test_cache_speedup: cached load is faster than raw parse

All expected behavior is derived from:
  docs/omiecid/migration-plan.md section M7.3
  docs/omiecid/agile-plan.md story M7.3
  omiecid_simulator/jaxlobster/lobster_loader.py (reference caching pattern)

These tests are black-box: they import only the public interface and
assert against specification-mandated behavior.
"""

import os
import shutil
import tempfile
import time

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig

# ---------------------------------------------------------------------------
# Guarded imports for M7.3 — skip gracefully if not yet implemented
# ---------------------------------------------------------------------------
try:
    from omiecid_simulator.jaxomiecid.omie_cache import (
        save_omie_cache,
        load_omie_cache,
    )
    _M73_MISSING = False
except ImportError:
    _M73_MISSING = True

try:
    import jax.numpy as jnp
    from omiecid_simulator.jaxomiecid.omie_data_loader import (
        load_omie_orders,
        load_omie_trades,
    )
    _M72_MISSING = False
except ImportError:
    _M72_MISSING = True

_m73_skip = pytest.mark.skipif(
    _M73_MISSING,
    reason="M7.3 omie_cache not yet implemented",
)

_m72_skip = pytest.mark.skipif(
    _M72_MISSING,
    reason="M7.2 omie_data_loader not yet implemented",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary directory for cache files, cleaned up after test."""
    d = tmp_path / "omie_cache"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_orders():
    """Synthetic (N, 13) order array matching OMIEOrder 13-field layout.

    This is a minimal array for roundtrip verification — it does not need
    to contain semantically valid order data, only structurally correct
    shapes and dtypes.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 1000, size=(25, 13)).astype(np.float32)


@pytest.fixture
def sample_trades():
    """Synthetic (M, 6) trade array matching OMIE trade 6-field layout."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 500, size=(10, 6)).astype(np.float32)


@pytest.fixture
def data_config(cache_dir):
    """OMIEDataConfig pointing at the temporary cache directory."""
    return OMIEDataConfig(
        data_path="/tmp/omie_data",
        contract_id=2,
        delivery_date="2025-04-01",
        cache_path=cache_dir,
    )


# ---------------------------------------------------------------------------
# TestCacheRoundtrip — save to NPZ, reload, arrays are identical
# ---------------------------------------------------------------------------

@_m73_skip
class TestCacheRoundtrip:
    """M7.3 requirement: save arrays to NPZ, reload, and verify identity.

    Per migration-plan section M7.3 and agile-plan M7.3:
    'test_cache_roundtrip() — save to NPZ, reload, arrays are identical.'
    """

    def test_roundtrip_orders_shape_preserved(self, data_config, sample_orders):
        """Saved and reloaded order array has the same shape."""
        save_omie_cache(data_config, orders=sample_orders)
        loaded = load_omie_cache(data_config)
        assert loaded["orders"].shape == sample_orders.shape

    def test_roundtrip_orders_values_identical(self, data_config, sample_orders):
        """Saved and reloaded order array values are bitwise identical."""
        save_omie_cache(data_config, orders=sample_orders)
        loaded = load_omie_cache(data_config)
        np.testing.assert_array_equal(loaded["orders"], sample_orders)

    def test_roundtrip_trades_shape_preserved(self, data_config, sample_trades):
        """Saved and reloaded trade array has the same shape."""
        save_omie_cache(data_config, trades=sample_trades)
        loaded = load_omie_cache(data_config)
        assert loaded["trades"].shape == sample_trades.shape

    def test_roundtrip_trades_values_identical(self, data_config, sample_trades):
        """Saved and reloaded trade array values are bitwise identical."""
        save_omie_cache(data_config, trades=sample_trades)
        loaded = load_omie_cache(data_config)
        np.testing.assert_array_equal(loaded["trades"], sample_trades)

    def test_roundtrip_both_arrays_together(
        self, data_config, sample_orders, sample_trades
    ):
        """When both orders and trades are saved, both survive roundtrip."""
        save_omie_cache(data_config, orders=sample_orders, trades=sample_trades)
        loaded = load_omie_cache(data_config)
        np.testing.assert_array_equal(loaded["orders"], sample_orders)
        np.testing.assert_array_equal(loaded["trades"], sample_trades)

    def test_roundtrip_dtype_preserved(self, data_config, sample_orders):
        """Array dtype is preserved through save/load cycle."""
        save_omie_cache(data_config, orders=sample_orders)
        loaded = load_omie_cache(data_config)
        assert loaded["orders"].dtype == sample_orders.dtype

    def test_roundtrip_empty_orders(self, data_config):
        """An empty (0, 13) order array survives roundtrip."""
        empty = np.empty((0, 13), dtype=np.float32)
        save_omie_cache(data_config, orders=empty)
        loaded = load_omie_cache(data_config)
        assert loaded["orders"].shape == (0, 13)

    def test_roundtrip_single_order(self, data_config):
        """A single-row (1, 13) order array survives roundtrip."""
        single = np.ones((1, 13), dtype=np.float32)
        save_omie_cache(data_config, orders=single)
        loaded = load_omie_cache(data_config)
        np.testing.assert_array_equal(loaded["orders"], single)

    def test_cache_file_created_on_disk(self, data_config, sample_orders):
        """save_omie_cache creates at least one .npz file under cache_path."""
        save_omie_cache(data_config, orders=sample_orders)
        npz_files = [
            f for f in os.listdir(data_config.cache_path) if f.endswith(".npz")
        ]
        assert len(npz_files) >= 1, "Expected at least one .npz file in cache_path"

    def test_cache_keyed_by_config(self, cache_dir, sample_orders):
        """Different configs (different contract_id) produce independent caches."""
        cfg_a = OMIEDataConfig(
            data_path="/tmp/omie_data",
            contract_id=2,
            delivery_date="2025-04-01",
            cache_path=cache_dir,
        )
        cfg_b = OMIEDataConfig(
            data_path="/tmp/omie_data",
            contract_id=37,
            delivery_date="2025-04-01",
            cache_path=cache_dir,
        )
        orders_a = sample_orders
        orders_b = sample_orders * 2  # different content

        save_omie_cache(cfg_a, orders=orders_a)
        save_omie_cache(cfg_b, orders=orders_b)

        loaded_a = load_omie_cache(cfg_a)
        loaded_b = load_omie_cache(cfg_b)

        np.testing.assert_array_equal(loaded_a["orders"], orders_a)
        np.testing.assert_array_equal(loaded_b["orders"], orders_b)

    def test_cache_keyed_by_delivery_date(self, cache_dir, sample_orders):
        """Different delivery_date values produce independent caches."""
        cfg_a = OMIEDataConfig(
            data_path="/tmp/omie_data",
            contract_id=2,
            delivery_date="2025-04-01",
            cache_path=cache_dir,
        )
        cfg_b = OMIEDataConfig(
            data_path="/tmp/omie_data",
            contract_id=2,
            delivery_date="2025-04-02",
            cache_path=cache_dir,
        )
        orders_a = sample_orders
        orders_b = sample_orders + 1

        save_omie_cache(cfg_a, orders=orders_a)
        save_omie_cache(cfg_b, orders=orders_b)

        loaded_a = load_omie_cache(cfg_a)
        loaded_b = load_omie_cache(cfg_b)

        np.testing.assert_array_equal(loaded_a["orders"], orders_a)
        np.testing.assert_array_equal(loaded_b["orders"], orders_b)

    def test_overwrite_existing_cache(self, data_config, sample_orders):
        """Saving again with different data overwrites the old cache."""
        save_omie_cache(data_config, orders=sample_orders)
        new_orders = sample_orders * 3
        save_omie_cache(data_config, orders=new_orders)
        loaded = load_omie_cache(data_config)
        np.testing.assert_array_equal(loaded["orders"], new_orders)

    def test_load_nonexistent_cache_returns_none(self, data_config):
        """Loading from a config with no saved cache returns None."""
        result = load_omie_cache(data_config)
        assert result is None


# ---------------------------------------------------------------------------
# TestCacheSpeedup — cached load is faster than raw parse
# ---------------------------------------------------------------------------

@_m73_skip
@_m72_skip
class TestCacheSpeedup:
    """M7.3 requirement: cached load is faster than raw parse.

    Per migration-plan section M7.3:
    'test_cache_speedup() — cached load is faster than raw parse.'

    These tests require real OMIE data files to be present.  They are
    skipped when the M7.2 loader or M7.3 cache module is unavailable.
    The speedup assertion is deliberately generous (cache must be at
    least 2x faster) to avoid flaky timing on CI.
    """

    @pytest.fixture
    def real_data_config(self, cache_dir):
        """Config pointing at real OMIE data.

        Uses environment variable OMIE_DATA_PATH if set, otherwise
        falls back to a conventional path.  Skips if data is absent.
        """
        data_path = os.environ.get(
            "OMIE_DATA_PATH", "data/rawOMIE"
        )
        cfg = OMIEDataConfig(
            data_path=data_path,
            contract_id=2,
            delivery_date="2025-06-14",
            cache_path=cache_dir,
        )
        # Verify raw data files exist — skip if not
        orders_glob = os.path.join(data_path, "orders_*.1")
        import glob as glob_mod
        if not glob_mod.glob(orders_glob):
            pytest.skip(
                f"No OMIE order files found under {data_path}; "
                "set OMIE_DATA_PATH to run cache speedup tests"
            )
        return cfg

    def test_cached_load_faster_than_raw_parse(self, real_data_config):
        """Cached load must be at least 2x faster than raw parse.

        Procedure:
        1. Time raw parse via load_omie_orders (no cache).
        2. Save the result to cache via save_omie_cache.
        3. Time load_omie_cache.
        4. Assert cache load < raw parse / 2.
        """
        cfg = real_data_config

        # Step 1: time the raw parse
        t0 = time.perf_counter()
        orders_raw = load_omie_orders(cfg)
        t_raw = time.perf_counter() - t0

        # Step 2: save to cache
        orders_np = np.asarray(orders_raw)
        save_omie_cache(cfg, orders=orders_np)

        # Step 3: time the cached load
        t0 = time.perf_counter()
        loaded = load_omie_cache(cfg)
        t_cache = time.perf_counter() - t0

        # Step 4: assert speedup (generous 2x threshold)
        assert t_cache < t_raw / 2, (
            f"Cache load ({t_cache:.4f}s) was not at least 2x faster "
            f"than raw parse ({t_raw:.4f}s)"
        )

        # Bonus: verify the cached data is identical
        np.testing.assert_array_equal(loaded["orders"], orders_np)


# ---------------------------------------------------------------------------
# TestCacheDirectoryHandling — edge cases around directory creation
# ---------------------------------------------------------------------------

@_m73_skip
class TestCacheDirectoryHandling:
    """Cache layer should handle directory creation and path edge cases."""

    def test_creates_cache_directory_if_missing(self, tmp_path, sample_orders):
        """save_omie_cache creates the cache_path directory if it doesn't exist."""
        nested = str(tmp_path / "deep" / "nested" / "cache")
        cfg = OMIEDataConfig(
            data_path="/tmp/omie_data",
            contract_id=0,
            delivery_date="2025-04-01",
            cache_path=nested,
        )
        # Directory should not exist yet
        assert not os.path.exists(nested)
        save_omie_cache(cfg, orders=sample_orders)
        assert os.path.isdir(nested)

        # And the data should be retrievable
        loaded = load_omie_cache(cfg)
        np.testing.assert_array_equal(loaded["orders"], sample_orders)

    def test_cache_path_with_trailing_slash(self, tmp_path, sample_orders):
        """Cache path works regardless of trailing slash presence."""
        d = str(tmp_path / "cache_trail")
        cfg_with_slash = OMIEDataConfig(
            data_path="/tmp/omie_data",
            contract_id=5,
            delivery_date="2025-04-01",
            cache_path=d + "/",
        )
        save_omie_cache(cfg_with_slash, orders=sample_orders)
        loaded = load_omie_cache(cfg_with_slash)
        np.testing.assert_array_equal(loaded["orders"], sample_orders)
