"""Tests for M1.3 — Order Validator.

Behavioral contracts for OMIE order validation rules.
All expected values are derived from docs/omiecid/market-spec.md §2.3, §4.5
and docs/omiecid/agile-plan.md M1.3.

These tests are black-box: they import only the public interface and
assert against domain-specified rules. No implementation assumptions.

The validator must be callable without instantiating a book.
"""

import jax
import pytest

from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    Validity,
    Status,
    make_order,
)
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_validator import validate_order


# ---------------------------------------------------------------------------
# Device-parametrized fixture (P3-OPS-01a, 2026-05-28)
# ---------------------------------------------------------------------------
#
# OMIE simulator tests default to CPU under the 2026-05-28 device-policy
# ruling (see ``docs/omiecid/device-policy.md`` and the session-level
# ``jax_platform_name`` pin in ``tests/omiecid/conftest.py``). The OQ-16
# Method A tick-precision invariant — which this file probes — was
# GPU-verified at P2-L1-HOTFIX-tick-precision (sprint-status §"P2-L1-HOTFIX-
# tick-precision — Implementation (Option C)") and is the one invariant
# in the suite that legitimately needs preserved dual-device coverage.
#
# This fixture is autouse + function-scoped + parametrized over
# ``("cpu", "gpu")``. Each test runs once per device. The GPU
# parametrization auto-skips on CUDA-hidden hosts. Per-test override
# of the session CPU pin is via ``jax.default_device(...)`` context
# manager — newly-created arrays within the wrapped test inherit that
# device, overriding the session ``jax_platform_name=cpu`` default.

@pytest.fixture(autouse=True, params=["cpu", "gpu"])
def jax_device(request):
    backend = request.param
    try:
        devices = jax.devices(backend)
    except RuntimeError:
        pytest.skip(f"{backend.upper()} device not visible")
    with jax.default_device(devices[0]):
        yield backend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Default OMIE product config (market-spec §2.3 values)."""
    return OMIEProductConfig()


def _make_valid_buy(contract_id=37, price=45.50, quantity=10.0, **overrides):
    """Helper: a valid NON buy order on an OPEN contract."""
    kwargs = dict(
        contract_id=contract_id,
        side=Side.BUY,
        price=price,
        quantity=quantity,
        order_id=1,
        agent_id=0,
        timestamp=100,
    )
    kwargs.update(overrides)
    return make_order(**kwargs)


def _make_valid_sell(contract_id=37, price=50.00, quantity=5.0, **overrides):
    """Helper: a valid NON sell order on an OPEN contract."""
    kwargs = dict(
        contract_id=contract_id,
        side=Side.SELL,
        price=price,
        quantity=quantity,
        order_id=2,
        agent_id=1,
        timestamp=200,
    )
    kwargs.update(overrides)
    return make_order(**kwargs)


# ---------------------------------------------------------------------------
# Test: Valid orders accepted
# ---------------------------------------------------------------------------

class TestValidOrderAccepted:
    """A well-formed order that satisfies all OMIE rules must be accepted."""

    def test_valid_buy_order_accepted(self, config):
        """A standard NON buy order with valid price/qty on an OPEN contract."""
        order = _make_valid_buy(price=45.50, quantity=10.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_valid_sell_order_accepted(self, config):
        """A standard NON sell order with valid price/qty on an OPEN contract."""
        order = _make_valid_sell(price=50.00, quantity=5.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_valid_ioc_order_accepted(self, config):
        """An IOC order with valid fields is accepted."""
        order = _make_valid_buy(exec_restriction=ExecRestriction.IOC)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_valid_fok_order_accepted(self, config):
        """A FOK order with valid fields is accepted."""
        order = _make_valid_sell(exec_restriction=ExecRestriction.FOK)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_valid_order_at_min_price(self, config):
        """Order at exactly the minimum price (-9999.0) is accepted."""
        order = _make_valid_buy(price=-9999.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_valid_order_at_max_price(self, config):
        """Order at exactly the maximum price (9999.0) is accepted."""
        order = _make_valid_sell(price=9999.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_valid_order_at_min_quantity(self, config):
        """Order at exactly the minimum quantity (0.1 MW) is accepted."""
        order = _make_valid_buy(quantity=0.1)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_valid_order_at_max_quantity(self, config):
        """Order at exactly the maximum quantity (999.0 MW) is accepted."""
        order = _make_valid_sell(quantity=999.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_valid_order_negative_price(self, config):
        """Negative prices are valid in OMIE (market-spec §2.3)."""
        order = _make_valid_buy(price=-50.01)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_valid_order_zero_price(self, config):
        """Zero price is valid and on tick (0.0 / 0.01 is integer)."""
        order = _make_valid_buy(price=0.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True


# ---------------------------------------------------------------------------
# Test: Price below minimum rejected
# ---------------------------------------------------------------------------

class TestPriceBelowMinRejected:
    """Orders with price < -9999.0 must be rejected (market-spec §2.3)."""

    def test_price_below_min_rejected(self, config):
        order = _make_valid_buy(price=-9999.01)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False

    def test_price_far_below_min_rejected(self, config):
        order = _make_valid_buy(price=-20000.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False


# ---------------------------------------------------------------------------
# Test: Price above maximum rejected
# ---------------------------------------------------------------------------

class TestPriceAboveMaxRejected:
    """Orders with price > 9999.0 must be rejected (market-spec §2.3)."""

    def test_price_above_max_rejected(self, config):
        order = _make_valid_sell(price=9999.01)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False

    def test_price_far_above_max_rejected(self, config):
        order = _make_valid_sell(price=20000.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False


# ---------------------------------------------------------------------------
# Test: Quantity below minimum rejected
# ---------------------------------------------------------------------------

class TestQtyBelowMinRejected:
    """Orders with quantity < 0.1 MW must be rejected (market-spec §2.3).

    As of P2-L1-HOTFIX-tick-precision (Option C), sub-qty_tick probes
    (e.g. ``0.09``) snap onto ``qty_min`` via the encoder and are accepted
    by the validator. The sub-qty_tick rejection assertion therefore no
    longer holds at the validator layer. The encoder-layer snap contract
    lives in ``tests/omiecid/test_encoding_snap_semantics.py``. The
    hard-zero and negative-quantity rejections survive here because those
    inputs snap to int values strictly below ``qty_int_min``.
    """

    def test_qty_below_min_rejected(self, config):
        order = _make_valid_buy(quantity=0.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False

    def test_qty_negative_rejected(self, config):
        order = _make_valid_buy(quantity=-1.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False


# ---------------------------------------------------------------------------
# Test: Quantity above maximum rejected
# ---------------------------------------------------------------------------

class TestQtyAboveMaxRejected:
    """Orders with quantity > 999.0 MW must be rejected (market-spec §2.3)."""

    def test_qty_above_max_rejected(self, config):
        order = _make_valid_sell(quantity=999.1)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False

    def test_qty_far_above_max_rejected(self, config):
        order = _make_valid_sell(quantity=5000.0)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False


# ---------------------------------------------------------------------------
# Test: Price not on tick rejected
# ---------------------------------------------------------------------------

class TestPriceNotOnTickRejected:
    """Price tick-alignment contract (market-spec §2.3).

    As of P2-L1-HOTFIX-tick-precision (Option C), off-tick rejection is the
    encoder's responsibility — sub-half-tick offsets snap to the nearest
    tick via ``jnp.round``. The validator bounds-checks the resulting int32.
    The only assertion that survives here is the on-tick acceptance
    sanity-check; the sub-half-tick off-tick probes have moved to
    ``tests/omiecid/test_encoding_snap_semantics.py``.
    """

    def test_price_on_tick_accepted(self, config):
        """45.51 is on 0.01 tick — accepted (sanity check)."""
        order = _make_valid_buy(price=45.51)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True


# ---------------------------------------------------------------------------
# Test: Quantity not on tick rejected
# ---------------------------------------------------------------------------

class TestQtyNotOnTickRejected:
    """Quantity tick-alignment contract (market-spec §2.3).

    As of P2-L1-HOTFIX-tick-precision (Option C), off-tick rejection is the
    encoder's responsibility — sub-half-tick offsets snap to the nearest
    tick via ``jnp.round``. The validator bounds-checks the resulting int32.
    The only assertion that survives here is the on-tick acceptance
    sanity-check; the sub-half-tick off-tick probes have moved to
    ``tests/omiecid/test_encoding_snap_semantics.py``.
    """

    def test_qty_on_tick_accepted(self, config):
        """10.3 is on 0.1 tick — accepted (sanity check)."""
        order = _make_valid_buy(quantity=10.3)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True


# ---------------------------------------------------------------------------
# Test: Order on closed contract rejected
# ---------------------------------------------------------------------------

class TestOrderOnClosedContractRejected:
    """Orders on a contract that is not OPEN must be rejected.

    The validator receives a contract_open flag. When False, the contract
    is past GCT (CLOSED) or not yet open (PENDING).
    """

    def test_order_on_closed_contract_rejected(self, config):
        """A valid order on a closed contract is rejected."""
        order = _make_valid_buy()
        result = validate_order(order, config, contract_open=False)
        assert bool(result) is False

    def test_sell_order_on_closed_contract_rejected(self, config):
        """A valid sell order on a closed contract is also rejected."""
        order = _make_valid_sell()
        result = validate_order(order, config, contract_open=False)
        assert bool(result) is False


# ---------------------------------------------------------------------------
# Test: ICE (iceberg) order rejected
# ---------------------------------------------------------------------------

class TestIcebergOrderValidation:
    """ICE orders accepted with valid peak_qty after M4.1 (blanket
    rejection removed). Invalid ICE orders still rejected by iceberg
    validation: peak_qty > 0, peak_qty <= quantity (market-spec §4.2).
    """

    def test_iceberg_order_accepted(self, config):
        """A valid ICE order with peak_qty > 0 and peak_qty <= quantity
        is accepted after M4.1."""
        order = _make_valid_buy(
            exec_restriction=ExecRestriction.ICE,
            peak_qty=5.0,
            ppd=0.0,
        )
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_iceberg_sell_order_accepted(self, config):
        """A valid sell ICE order is also accepted."""
        order = _make_valid_sell(
            exec_restriction=ExecRestriction.ICE,
            peak_qty=2.0,
            ppd=0.1,
        )
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_iceberg_with_zero_peak_qty_rejected(self, config):
        """An ICE order with peak_qty=0 is rejected (iceberg validation
        requires peak_qty > 0)."""
        order = _make_valid_buy(
            exec_restriction=ExecRestriction.ICE,
            peak_qty=0.0,
            ppd=0.0,
        )
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False


# ---------------------------------------------------------------------------
# Test: Side validation
# ---------------------------------------------------------------------------

class TestSideValidation:
    """Side must be a valid Side enum value (BUY=0 or SELL=1)."""

    def test_buy_side_accepted(self, config):
        order = _make_valid_buy()
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_sell_side_accepted(self, config):
        order = _make_valid_sell()
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is True

    def test_invalid_side_rejected(self, config):
        """An order with an invalid side value (not 0 or 1) is rejected."""
        order = make_order(
            contract_id=37,
            side=99,  # invalid side
            price=45.50,
            quantity=10.0,
            order_id=1,
            agent_id=0,
            timestamp=100,
        )
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False


# ---------------------------------------------------------------------------
# Test: Multiple violations — validator rejects (no specific order required)
# ---------------------------------------------------------------------------

class TestMultipleViolations:
    """When multiple rules are violated, the order is still rejected."""

    def test_price_and_qty_both_invalid(self, config):
        """Price out of bounds AND quantity off tick — still rejected."""
        order = _make_valid_buy(price=99999.0, quantity=10.05)
        result = validate_order(order, config, contract_open=True)
        assert bool(result) is False

    def test_ice_on_closed_contract(self, config):
        """ICE order on a closed contract — rejected."""
        order = _make_valid_buy(exec_restriction=ExecRestriction.ICE, peak_qty=5.0)
        result = validate_order(order, config, contract_open=False)
        assert bool(result) is False
