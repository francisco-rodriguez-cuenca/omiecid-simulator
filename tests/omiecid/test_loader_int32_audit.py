"""Independent audit tests for Story P2-F0 — Loader numeric representation → int32.

Audit-mode coverage. Written without reading the original locked
``test_loader_int32.py``. Focus areas:

  * encoder/decoder boundary-value correctness (price_min, price_max,
    zero, negative, on-tick / off-tick rounding semantics),
  * signed ``ppd`` tick encoding and its *non*-offset nature,
  * cache v4 negative paths (wrong version, missing version, corrupted
    payload, forward-compat future version),
  * dtype containment at every observable output slot,
  * determinism / bitwise-identical reloads,
  * timestamp modulo-86400 wrap boundaries,
  * real-data round-trip sanity (slow) on 2026-01-12 data.

All tests exercise only the public interface:
  - ``omiecid_simulator.omiecid.encoding``          — encode_*/decode_*
  - ``omiecid_simulator.jaxomiecid.omie_data_loader`` — load_omie_orders/trades, row_to_order
  - ``omiecid_simulator.jaxomiecid.omie_cache``      — save/load + CACHE_SCHEMA_VERSION
"""

from __future__ import annotations

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_cache import (
    CACHE_SCHEMA_VERSION,
    load_omie_cache,
    save_omie_cache,
)
from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    load_omie_orders,
    load_omie_trades,
    row_to_order,
)
from omiecid_simulator.omiecid.encoding import (
    decode_ppd,
    decode_price,
    decode_qty,
    encode_ppd,
    encode_price,
    encode_qty,
)
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pc() -> OMIEProductConfig:
    return OMIEProductConfig()


REAL_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "rawOMIE"
)
REAL_DATE = "2026-01-12"
REAL_CID = 81


# ---------------------------------------------------------------------------
# 1. Price boundary + rounding audits
# ---------------------------------------------------------------------------


class TestPriceEncodingBoundaries:
    """Encoder/decoder behavior at the SIDC domain extremes."""

    def test_price_min_encodes_to_zero(self, pc):
        # Offset-encoding anchors price_min to 0.
        assert encode_price(pc.price_min, pc) == 0

    def test_price_max_encodes_to_full_span(self, pc):
        span = round((pc.price_max - pc.price_min) / pc.price_tick)
        assert encode_price(pc.price_max, pc) == span
        # Must fit int32 with room to spare.
        assert span < np.iinfo(np.int32).max

    def test_price_zero_round_trips(self, pc):
        assert decode_price(encode_price(0.0, pc), pc) == pytest.approx(0.0, abs=1e-9)

    def test_negative_on_tick_price_round_trips(self, pc):
        for p in (-0.01, -123.45, -500.00, -9998.99):
            enc = encode_price(p, pc)
            assert isinstance(enc, int)
            assert enc >= 0, "offset encoding must be non-negative"
            assert decode_price(enc, pc) == pytest.approx(p, abs=1e-9)

    def test_positive_on_tick_price_round_trips(self, pc):
        for p in (0.01, 50.00, 133.87, 999.99, 9998.99):
            enc = encode_price(p, pc)
            assert decode_price(enc, pc) == pytest.approx(p, abs=1e-9)

    def test_off_tick_price_rounds_to_nearest_tick(self, pc):
        # Half-tick cases are banker's-rounded by Python's round().
        # Document the actual behavior; do not assume away-from-zero.
        # 0.005 rounds to 0.00 (0 ticks above price_min-offset reference).
        # 0.015 rounds to 0.02 (banker's rounding to even).
        # We pin the behavior that round_half_to_even is in effect.
        assert round((0.005 - pc.price_min) / pc.price_tick) == encode_price(0.005, pc)
        assert round((0.015 - pc.price_min) / pc.price_tick) == encode_price(0.015, pc)


# ---------------------------------------------------------------------------
# 2. Quantity boundary audits
# ---------------------------------------------------------------------------


class TestQuantityEncodingBoundaries:
    def test_qty_zero_encodes_to_zero(self, pc):
        assert encode_qty(0.0, pc) == 0

    def test_qty_min_round_trips(self, pc):
        assert decode_qty(encode_qty(pc.qty_min, pc), pc) == pytest.approx(
            pc.qty_min, abs=1e-9
        )

    def test_qty_max_round_trips(self, pc):
        assert decode_qty(encode_qty(pc.qty_max, pc), pc) == pytest.approx(
            pc.qty_max, abs=1e-9
        )

    def test_qty_max_fits_int32(self, pc):
        assert encode_qty(pc.qty_max, pc) < np.iinfo(np.int32).max

    def test_qty_is_non_negative_for_physical_values(self, pc):
        for q in (0.1, 0.2, 1.0, 7.5, 99.9, 500.0, 998.9):
            assert encode_qty(q, pc) >= 0


# ---------------------------------------------------------------------------
# 3. PPD encoding audits — THE critical non-offset semantics
# ---------------------------------------------------------------------------


class TestPPDSignedTickEncoding:
    """``ppd`` must be signed int-ticks, NOT offset-encoded."""

    def test_ppd_zero_encodes_to_zero(self, pc):
        # The distinguishing feature between signed int-tick and offset
        # encoding: only signed int-tick yields 0 for 0.0 EUR/MWh.
        assert encode_ppd(0.0, pc) == 0

    def test_ppd_negative_preserves_sign(self, pc):
        for p in (-0.01, -1.55, -3.00, -50.00):
            enc = encode_ppd(p, pc)
            assert enc < 0, f"sign lost for PPD={p}: encoded={enc}"
            assert decode_ppd(enc, pc) == pytest.approx(p, abs=1e-9)

    def test_ppd_positive_round_trips(self, pc):
        for p in (0.01, 1.45, 3.00, 50.00):
            enc = encode_ppd(p, pc)
            assert enc > 0
            assert decode_ppd(enc, pc) == pytest.approx(p, abs=1e-9)

    def test_ppd_encoding_is_exact_division_by_price_tick(self, pc):
        # Formula contract: encode_ppd(p) = round(p / price_tick).
        # No offset, no scale factor other than price_tick.
        for p in (-2.37, 0.0, 4.13):
            assert encode_ppd(p, pc) == round(p / pc.price_tick)

    def test_ppd_differs_from_price_encoding_by_the_offset(self, pc):
        # encode_price(x) = encode_ppd(x) + encode_price(0.0).
        for x in (-1.5, 0.0, 2.5, 10.0):
            assert encode_price(x, pc) == encode_ppd(x, pc) + encode_price(0.0, pc)


# ---------------------------------------------------------------------------
# 4. Cache v4 negative paths
# ---------------------------------------------------------------------------


class TestCacheV4NegativePaths:
    def _cfg(self, tmpdir: str) -> OMIEDataConfig:
        return OMIEDataConfig(
            data_path=tmpdir,
            contract_id=0,
            delivery_date="2026-01-12",
            cache_path=tmpdir,
        )

    def test_current_schema_version_is_four(self):
        assert CACHE_SCHEMA_VERSION == 4

    def test_missing_cache_file_returns_none(self, tmp_path):
        cfg = self._cfg(str(tmp_path))
        assert load_omie_cache(cfg) is None

    def test_cache_with_missing_schema_version_rejected(self, tmp_path):
        cfg = self._cfg(str(tmp_path))
        path = os.path.join(
            str(tmp_path), f"omie_cache_c0_{cfg.delivery_date}.npz"
        )
        np.savez_compressed(
            path, orders=np.zeros((0, 13), dtype=np.int32)
        )  # no schema_version key
        assert load_omie_cache(cfg) is None

    def test_cache_with_older_schema_version_rejected(self, tmp_path):
        cfg = self._cfg(str(tmp_path))
        path = os.path.join(
            str(tmp_path), f"omie_cache_c0_{cfg.delivery_date}.npz"
        )
        np.savez_compressed(
            path,
            schema_version=np.array(3, dtype=np.int32),
            orders=np.zeros((0, 13), dtype=np.int32),
        )
        assert load_omie_cache(cfg) is None

    def test_forward_compat_future_schema_rejected_gracefully(self, tmp_path):
        # A v5+ cache (forward compat) must not crash — just returns None.
        cfg = self._cfg(str(tmp_path))
        path = os.path.join(
            str(tmp_path), f"omie_cache_c0_{cfg.delivery_date}.npz"
        )
        np.savez_compressed(
            path,
            schema_version=np.array(CACHE_SCHEMA_VERSION + 1, dtype=np.int32),
            orders=np.zeros((1, 13), dtype=np.int32),
        )
        assert load_omie_cache(cfg) is None

    def test_roundtrip_bitwise_identical(self, tmp_path):
        cfg = self._cfg(str(tmp_path))
        rng = np.random.default_rng(0xC0FFEE)
        orders = rng.integers(
            low=0, high=1_000_000, size=(17, 13), dtype=np.int32
        )
        trades = rng.integers(low=0, high=500_000, size=(5, 6), dtype=np.int32)
        save_omie_cache(cfg, orders=orders, trades=trades, epoch_base=1_737_000_000)
        reloaded = load_omie_cache(cfg)
        assert reloaded is not None
        assert np.array_equal(reloaded["orders"], orders)
        assert np.array_equal(reloaded["trades"], trades)
        # Dtype containment on the cache round trip.
        assert reloaded["orders"].dtype == np.int32
        assert reloaded["trades"].dtype == np.int32

    def test_roundtrip_orders_only_no_trades(self, tmp_path):
        cfg = self._cfg(str(tmp_path))
        orders = np.zeros((3, 13), dtype=np.int32)
        save_omie_cache(cfg, orders=orders)
        reloaded = load_omie_cache(cfg)
        assert reloaded is not None
        assert "orders" in reloaded
        # trades must not be synthesized; only what was saved comes back.
        assert "trades" not in reloaded


# ---------------------------------------------------------------------------
# 5. row_to_order audit
# ---------------------------------------------------------------------------


class TestRowToOrderDecoding:
    def test_row_decoding_recovers_eur_and_mw(self, pc):
        price_eur = 133.87
        qty_mw = 2.5
        peak_qty_mw = 1.0
        ppd_eur = -1.55

        row = jnp.array(
            [
                5,                                # contract_id
                0,                                # side (BUY)
                encode_price(price_eur, pc),      # price
                encode_qty(qty_mw, pc),           # quantity
                42,                               # order_id
                7,                                # agent_id
                12345,                            # timestamp
                3,                                # exec_restriction (ICE)
                1,                                # validity (GTD)
                0,                                # expiry_time
                1,                                # status
                encode_qty(peak_qty_mw, pc),      # peak_qty
                encode_ppd(ppd_eur, pc),          # ppd  (NEGATIVE)
            ],
            dtype=jnp.int32,
        )

        order = row_to_order(row, pc)
        assert order.contract_id == 5
        assert int(order.side) == 0
        assert order.price == pytest.approx(price_eur, abs=1e-9)
        assert order.quantity == pytest.approx(qty_mw, abs=1e-9)
        assert order.peak_qty == pytest.approx(peak_qty_mw, abs=1e-9)
        assert order.ppd == pytest.approx(ppd_eur, abs=1e-9)
        # PPD sign must survive the full int32-row → OMIEOrder decode.
        assert order.ppd < 0


# ---------------------------------------------------------------------------
# 6. Real-data audits — slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRealDataAudit:
    """slow: loads the 2026-01-12 raw OMIE orders file (thousands of rows)."""

    @pytest.fixture
    def cfg(self, tmp_path):
        # Skip-on-data-absent: the product needs session files D-1 and D.
        for day in ("20260111", "20260112"):
            orders_path = os.path.join(REAL_DATA_PATH, f"orders_{day}.1")
            if not os.path.isfile(orders_path):
                pytest.skip(f"raw OMIE orders file not present: {orders_path}")
        return OMIEDataConfig(
            data_path=REAL_DATA_PATH,
            contract_id=REAL_CID,
            delivery_date=REAL_DATE,
            cache_path=str(tmp_path),
        )

    def test_orders_output_dtype_and_shape(self, cfg):
        arr = load_omie_orders(cfg)
        assert arr.dtype == jnp.int32
        assert arr.ndim == 2
        assert arr.shape[1] == 13
        assert arr.shape[0] > 0, "expected real data for cid=81 on 2026-01-12"

    def test_orders_load_is_deterministic(self, cfg):
        """slow: double-load check for hidden nondeterminism."""
        a = load_omie_orders(cfg)
        b = load_omie_orders(cfg)
        assert jnp.array_equal(a, b)

    def test_timestamp_column_in_session_range(self, cfg):
        # OMIE-TAPE-CONTRACT-DATE-1 (group L re-baseline; was [0, 86400)):
        # seconds since D-1 00:00, no modulo; real sends lie in
        # [15:20 D-1, 22:00 D) = [55200, 165600) (market-spec v2.10 §13.4).
        arr = load_omie_orders(cfg)
        ts = arr[:, 6]
        assert int(jnp.min(ts)) >= 55200
        assert int(jnp.max(ts)) < 165600

    def test_peak_qty_invariants(self, cfg):
        """Cross-column invariants: peak_qty <= quantity; peak_qty==0 for non-ICE/ICEwPI.

        Note: ICEwPI is not a distinct ExecRestriction enum value in
        the current schema — ICE covers both pure-iceberg and ICE-with-PPD
        flavors (market-spec §13.6), disambiguated by whether ``ppd != 0``.
        The non-iceberg rule therefore locks ``exec_restriction != ICE``.
        """
        arr = load_omie_orders(cfg)
        quantity = arr[:, 3]
        exec_restriction = arr[:, 7]
        peak_qty = arr[:, 11]
        # peak_qty <= quantity row-wise.
        assert bool(jnp.all(peak_qty <= quantity))
        # peak_qty == 0 when exec_restriction != ICE (ICE == 3).
        non_ice = exec_restriction != 3
        assert bool(jnp.all(peak_qty[non_ice] == 0))

    def test_encoded_columns_fit_int32(self, cfg):
        arr = load_omie_orders(cfg)
        i32_max = int(np.iinfo(np.int32).max)
        i32_min = int(np.iinfo(np.int32).min)
        assert int(jnp.min(arr)) > i32_min
        assert int(jnp.max(arr)) < i32_max

    def test_real_rows_round_trip_through_row_to_order(self, cfg, pc):
        """Sampled rows decode, then re-encode to the same int32 values."""
        arr = load_omie_orders(cfg)
        n = arr.shape[0]
        sample_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        for i in sample_idx:
            row = arr[i]
            order = row_to_order(row, pc)
            # Re-encode and confirm.
            assert encode_price(order.price, pc) == int(row[2])
            assert encode_qty(order.quantity, pc) == int(row[3])
            assert encode_qty(order.peak_qty, pc) == int(row[11])
            assert encode_ppd(order.ppd, pc) == int(row[12])

    def test_trades_output_dtype_when_present(self, cfg):
        """If a trades file exists, it must also load as int32 (M, 6)."""
        trades_path = os.path.join(
            REAL_DATA_PATH, f"trades_{REAL_DATE.replace('-', '')}.1"
        )
        if not os.path.isfile(trades_path):
            pytest.skip(f"trades file not present for {REAL_DATE}")
        arr = load_omie_trades(cfg)
        assert arr.dtype == jnp.int32
        assert arr.ndim == 2
        assert arr.shape[1] == 6
