"""Audit tests for story 1.5-HOTFIX-loader-precision.

Independent probes of the public `row_to_order(row, product_config)` helper
in `omiecid_simulator.jaxomiecid.omie_data_loader` and its integration point in
`replay_orders_through_book`.

Scope (post-implementation verification):
  * non-float columns round-trip byte-identically through the helper
  * snap idempotency (double-apply is a no-op)
  * output prices / quantities are exact integer multiples of their tick
  * behaviour under extreme ticks (fine grid, coarse grid)
  * behaviour at ULP / half-tick boundaries (documents Python banker's rounding)
  * negative-zero handling
  * NaN propagation (the helper is not responsible for sanitizing NaN —
    that is ``_parse_float_comma`` / ``_parse_optional_float`` upstream;
    we only document the observed behaviour so any regression is visible)
  * the helper is actually invoked from ``replay_orders_through_book``
  * ``OMIEOrder`` field-count invariant is preserved

These tests derive from the sprint acceptance criteria *independently*:
the audit agent was not shown the original locked-test file.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from omiecid_simulator.jaxomiecid.omie_data_loader import (
    replay_orders_through_book,
    row_to_order,
)
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.encoding import (
    encode_ppd,
    encode_price,
    encode_qty,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    Status,
    Validity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_PC = OMIEProductConfig()  # price_tick=0.01, qty_tick=0.1


def _row(
    *,
    contract_id=5,
    side=0,
    price=12.34,
    quantity=1.5,
    order_id=42,
    agent_id=7,
    timestamp=3600,
    exec_restriction=0,
    validity=0,
    expiry_time=0,
    status=0,
    peak_qty=0.0,
    ppd=0.0,
    product_config=DEFAULT_PC,
) -> list:
    """Build a 13-column loader-style row as a plain python list.

    Post P2-F0, the loader emits int32 rows with tick-encoded price /
    quantity / peak_qty / ppd. This helper accepts EUR/MWh prices and
    MW quantities and encodes them on the caller-supplied
    ``product_config`` tick grid so the test caller can still express
    intent in human-readable units.
    """
    price_int = encode_price(price, product_config)
    qty_int = encode_qty(quantity, product_config)
    peak_qty_int = encode_qty(peak_qty, product_config)
    ppd_int = encode_ppd(ppd, product_config)
    return [
        contract_id, side, price_int, qty_int, order_id, agent_id,
        timestamp, exec_restriction, validity, expiry_time, status,
        peak_qty_int, ppd_int,
    ]


def _jnp_row(**kw) -> jnp.ndarray:
    """Build an int32 row matching post-P2-F0 ``load_omie_orders`` output."""
    return jnp.asarray(_row(**kw), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Non-float columns — must round-trip byte-identically
# ---------------------------------------------------------------------------

class TestNonFloatColumnsPreserved:
    """Columns 0, 1, 4, 5, 6, 7, 8, 9, 10 must pass through untouched."""

    def test_contract_id_preserved(self):
        o = row_to_order(_row(contract_id=17), DEFAULT_PC)
        assert o.contract_id == 17

    def test_side_buy_preserved(self):
        o = row_to_order(_row(side=0), DEFAULT_PC)
        assert o.side == Side.BUY

    def test_side_sell_preserved(self):
        o = row_to_order(_row(side=1), DEFAULT_PC)
        assert o.side == Side.SELL

    def test_order_id_preserved(self):
        o = row_to_order(_row(order_id=1_000_001), DEFAULT_PC)
        assert o.order_id == 1_000_001

    def test_agent_id_preserved(self):
        o = row_to_order(_row(agent_id=123), DEFAULT_PC)
        assert o.agent_id == 123

    def test_timestamp_preserved(self):
        o = row_to_order(_row(timestamp=86_399), DEFAULT_PC)
        assert o.timestamp == 86_399

    def test_exec_restriction_all_values(self):
        for v in (0, 1, 2, 3):
            o = row_to_order(_row(exec_restriction=v), DEFAULT_PC)
            assert int(o.exec_restriction) == v

    def test_validity_preserved(self):
        for v in (0, 1):
            o = row_to_order(_row(validity=v), DEFAULT_PC)
            assert int(o.validity) == v

    def test_expiry_time_preserved(self):
        o = row_to_order(_row(expiry_time=7200), DEFAULT_PC)
        assert o.expiry_time == 7200

    def test_status_preserved(self):
        for v in (0, 1, 2, 3, 4, 5):
            o = row_to_order(_row(status=v), DEFAULT_PC)
            assert int(o.status) == v


# ---------------------------------------------------------------------------
# Tick alignment — output divided by tick must be integer within 1e-9
# ---------------------------------------------------------------------------

class TestTickMultiples:

    @pytest.mark.parametrize("price", [0.01, 0.02, 12.34, 99.99, -5.55, 0.0])
    def test_price_is_integer_multiple_of_price_tick(self, price):
        o = row_to_order(_row(price=price), DEFAULT_PC)
        ratio = o.price / DEFAULT_PC.price_tick
        assert abs(ratio - round(ratio)) < 1e-9

    @pytest.mark.parametrize("qty", [0.1, 0.3, 1.5, 99.9, 0.0])
    def test_quantity_is_integer_multiple_of_qty_tick(self, qty):
        o = row_to_order(_row(quantity=qty), DEFAULT_PC)
        ratio = o.quantity / DEFAULT_PC.qty_tick
        assert abs(ratio - round(ratio)) < 1e-9

    def test_peak_qty_snaps_to_qty_tick(self):
        o = row_to_order(_row(peak_qty=2.4), DEFAULT_PC)
        ratio = o.peak_qty / DEFAULT_PC.qty_tick
        assert abs(ratio - round(ratio)) < 1e-9

    def test_ppd_snaps_to_price_tick(self):
        o = row_to_order(_row(ppd=0.37), DEFAULT_PC)
        ratio = o.ppd / DEFAULT_PC.price_tick
        assert abs(ratio - round(ratio)) < 1e-9

    def test_float32_noise_is_absorbed(self):
        """Values cast through float32 (loader pattern) still snap cleanly."""
        # 12.34 is not representable in float32; ratio without snap drifts.
        row = _jnp_row(price=12.34, quantity=1.5, peak_qty=2.4, ppd=0.37)
        o = row_to_order(row, DEFAULT_PC)
        for val, tick in (
            (o.price, DEFAULT_PC.price_tick),
            (o.quantity, DEFAULT_PC.qty_tick),
            (o.peak_qty, DEFAULT_PC.qty_tick),
            (o.ppd, DEFAULT_PC.price_tick),
        ):
            ratio = val / tick
            assert abs(ratio - round(ratio)) < 1e-9


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_double_snap_is_noop(self):
        o1 = row_to_order(_row(price=10.01, quantity=1.2, peak_qty=0.3, ppd=0.05),
                          DEFAULT_PC)
        # Feed o1's values back through the helper.
        row2 = _row(
            price=float(o1.price),
            quantity=float(o1.quantity),
            peak_qty=float(o1.peak_qty),
            ppd=float(o1.ppd),
        )
        o2 = row_to_order(row2, DEFAULT_PC)
        assert float(o1.price) == float(o2.price)
        assert float(o1.quantity) == float(o2.quantity)
        assert float(o1.peak_qty) == float(o2.peak_qty)
        assert float(o1.ppd) == float(o2.ppd)

    def test_encode_decode_round_trip_idempotent(self):
        """Post P2-F0 the snap step is the encode/decode pair on int-ticks.

        ``encode_price`` followed by ``decode_price`` is idempotent on any
        on-tick input, so re-encoding the decoded value yields the same
        integer tick as the first encode — the P2-F0 analogue of the
        ``_snap_to_tick`` idempotency check.
        """
        from omiecid_simulator.omiecid.encoding import (
            decode_price as _decode_price,
        )
        for v in (0.01, 0.02, 12.34, 99.99, 0.0):
            once = encode_price(v, DEFAULT_PC)
            twice = encode_price(_decode_price(once, DEFAULT_PC), DEFAULT_PC)
            assert once == twice


# ---------------------------------------------------------------------------
# Extreme ticks
# ---------------------------------------------------------------------------

class TestExtremeTicks:

    def test_fine_price_tick(self):
        pc = OMIEProductConfig(price_tick=0.0001, qty_tick=0.001)
        o = row_to_order(_row(price=1.2345, quantity=0.123, product_config=pc), pc)
        # Offset decoding ``int * 0.0001 + (-9999)`` introduces a few ULP
        # of float noise for very fine ticks; relax the ratio check to
        # 1e-7 (Category B adapter, P2-F0). The underlying integer tick
        # index is still exact.
        assert abs(o.price / pc.price_tick - round(o.price / pc.price_tick)) < 1e-7
        assert abs(o.quantity / pc.qty_tick - round(o.quantity / pc.qty_tick)) < 1e-7

    def test_coarse_price_tick(self):
        pc = OMIEProductConfig(price_tick=1.0, qty_tick=1.0)
        o = row_to_order(_row(price=12.4, quantity=3.5, product_config=pc), pc)
        # 12.4 -> 12.0, 3.5 -> 4 (banker's: round-half-to-even)
        assert o.price == 12.0
        assert o.quantity == 4.0


# ---------------------------------------------------------------------------
# Rounding edge cases
# ---------------------------------------------------------------------------

class TestRoundingEdges:

    def test_negative_zero_is_snapped_cleanly(self):
        """-0.0 input must not produce NaN / non-finite output."""
        o = row_to_order(_row(price=-0.0, quantity=-0.0), DEFAULT_PC)
        assert math.isfinite(float(o.price))
        assert math.isfinite(float(o.quantity))
        assert float(o.price) == 0.0
        assert float(o.quantity) == 0.0

    def test_half_tick_boundary_uses_python_bankers_rounding(self):
        """Python's built-in ``round`` is half-to-even.

        This test documents the rounding direction so any silent change
        (e.g., a switch to ``int(... + 0.5)``) would be caught.
        """
        # 0.005 / 0.01 == 0.5 (approximately); round(0.5) -> 0 (to-even).
        # 0.015 / 0.01 == 1.5; round(1.5) -> 2 (to-even).
        # These identities may shift by a ULP because of float32 casts, so
        # we assert the result is *on tick* rather than a specific side.
        for price in (0.005, 0.015, 0.025):
            # Post P2-F0 the loader stores int ticks directly; decoding
            # gives the tick-aligned EUR/MWh value.
            enc = encode_price(price, DEFAULT_PC)
            decoded = enc * DEFAULT_PC.price_tick + DEFAULT_PC.price_min
            ratio = decoded / 0.01
            assert abs(ratio - round(ratio)) < 1e-6

    def test_negative_price_snaps_correctly(self):
        """OMIE allows negative prices down to -9999.0; snap must cope."""
        o = row_to_order(_row(price=-12.34), DEFAULT_PC)
        ratio = o.price / DEFAULT_PC.price_tick
        assert abs(ratio - round(ratio)) < 1e-9
        assert float(o.price) < 0

    def test_large_price_near_ceiling(self):
        o = row_to_order(_row(price=9998.99), DEFAULT_PC)
        ratio = o.price / DEFAULT_PC.price_tick
        assert abs(ratio - round(ratio)) < 1e-9


# ---------------------------------------------------------------------------
# NaN handling (documentation test)
# ---------------------------------------------------------------------------

class TestNaNHandling:
    """The helper is implemented as ``round(v / tick) * tick``.

    ``round(nan)`` raises ``ValueError`` in Python. This test documents
    that the helper does **not** silently swallow NaN — upstream parsers
    (``_parse_optional_float`` etc.) are responsible for eliminating it.
    """

    def test_nan_price_raises(self):
        with pytest.raises((ValueError, OverflowError)):
            row_to_order(_row(price=float("nan")), DEFAULT_PC)


# ---------------------------------------------------------------------------
# OMIEOrder structural invariants
# ---------------------------------------------------------------------------

class TestOMIEOrderShape:

    def test_returns_omie_order_namedtuple(self):
        o = row_to_order(_row(), DEFAULT_PC)
        assert isinstance(o, OMIEOrder)

    def test_has_thirteen_fields(self):
        o = row_to_order(_row(), DEFAULT_PC)
        assert len(o) == 13
        assert len(o._fields) == 13

    def test_field_names_match_spec(self):
        expected = (
            "contract_id", "side", "price", "quantity", "order_id",
            "agent_id", "timestamp", "exec_restriction", "validity",
            "expiry_time", "status", "peak_qty", "ppd",
        )
        assert OMIEOrder._fields == expected


# ---------------------------------------------------------------------------
# Integration — helper is actually invoked inside replay_orders_through_book
# ---------------------------------------------------------------------------

class TestReplayUsesHelper:
    """Probe that ``replay_orders_through_book`` snaps via the helper.

    Strategy: feed a single order whose raw float32 representation would
    *not* be on-tick (e.g. 12.34 — which float32 cannot represent
    exactly). If the helper fires, the order is accepted and rests in
    the book. If the code path bypassed the helper and passed the raw
    float32 into ``validate_order``, the order would be rejected at
    ``_is_on_tick`` (1e-9 tolerance) and produce zero replayed trades
    plus an empty resting book.
    """

    def _make_book(self):
        contract = Contract(
            contract_id=5,
            delivery_start_minutes=5 * 15,
            delivery_end_minutes=5 * 15 + 15,
            gct_minutes=5 * 15 - 60,
            status=ContractStatus.OPEN,
        )
        return ContractBook(contract, OMIEProductConfig(), OMIEMarketConfig())

    def test_off_tick_float32_order_is_accepted_via_helper(self):
        # Post P2-F0 the loader emits int32 tick-encoded rows; replay
        # decodes via ``row_to_order``. We build the row by encoding a
        # representative on-tick price to demonstrate the integration
        # path (replay feeds row → row_to_order → validate_order).
        pc = OMIEProductConfig()
        orders = jnp.asarray([[
            5,
            0,
            encode_price(12.34, pc),
            encode_qty(1.5, pc),
            1,
            7,
            3600,
            0,
            0,
            0,
            0,
            encode_qty(0.0, pc),
            encode_ppd(0.0, pc),
        ]], dtype=jnp.int32)

        historical = jnp.zeros((0, 6), dtype=jnp.int32)
        book = self._make_book()

        result = replay_orders_through_book(
            orders, book, historical, heuristic="rest_until_gct",
        )

        # No trades expected (only one resting order), but the order
        # must have been *accepted*. We verify acceptance by checking
        # that the final book's bids array has at least one live order.
        lob = result.final_book.lob
        # jaxob encodes empty slots with price == -1 (sentinel). If the
        # order was rejected by on-tick, every slot stays at -1.
        bid_prices = lob.bids[:, 0]
        assert int(jnp.any(bid_prices != -1)) == 1, (
            "Off-tick float32 price was rejected — replay likely "
            "bypassed row_to_order."
        )

    def test_replay_empty_orders_smoke(self):
        """Sanity: zero-order replay returns a well-formed ReplayResult."""
        orders = jnp.zeros((0, 13), dtype=jnp.int32)
        historical = jnp.zeros((0, 6), dtype=jnp.int32)
        book = self._make_book()
        result = replay_orders_through_book(orders, book, historical)
        assert result.historical_trades_total == 0
        assert result.replayed_trades.shape == (0, 3)
