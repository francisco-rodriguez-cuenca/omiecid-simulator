"""Audit tests for M1.2 — Order Type Definitions.

Post-implementation verification focusing on edge cases, boundary
conditions, and domain compliance beyond the original test suite.
Derived independently from market-spec §4, §4.5, §5.1, §13
and agile-plan M1.2 acceptance criteria.
"""

import pytest

from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    Validity,
    Status,
    OMIEOrder,
    make_order,
)


# ---------------------------------------------------------------------------
# IntEnum JAX compatibility — values must behave as plain integers
# ---------------------------------------------------------------------------

class TestEnumIntArithmetic:
    """IntEnum values must participate in integer arithmetic and comparison
    without explicit .value access, ensuring JAX array compatibility."""

    def test_side_buy_usable_in_arithmetic(self):
        assert Side.BUY + 1 == Side.SELL

    def test_exec_restriction_comparable_to_int(self):
        assert ExecRestriction.ICE == 3
        assert ExecRestriction.NON == 0

    def test_validity_usable_as_index(self):
        labels = ["GFS", "GTD"]
        assert labels[Validity.GFS] == "GFS"
        assert labels[Validity.GTD] == "GTD"

    def test_status_values_are_contiguous_from_zero(self):
        """Status values should be 0..5 for dense array indexing."""
        values = sorted(s.value for s in Status)
        assert values == list(range(6))

    def test_exec_restriction_values_are_contiguous_from_zero(self):
        """ExecRestriction values should be 0..3 for dense array indexing."""
        values = sorted(er.value for er in ExecRestriction)
        assert values == list(range(4))


# ---------------------------------------------------------------------------
# Enum round-trip construction from int
# ---------------------------------------------------------------------------

class TestEnumFromInt:
    """Enums must be constructible from their integer values (round-trip)."""

    def test_side_from_int(self):
        assert Side(0) is Side.BUY
        assert Side(1) is Side.SELL

    def test_exec_restriction_from_int(self):
        for er in ExecRestriction:
            assert ExecRestriction(er.value) is er

    def test_validity_from_int(self):
        for v in Validity:
            assert Validity(v.value) is v

    def test_status_from_int(self):
        for s in Status:
            assert Status(s.value) is s

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            Side(99)

    def test_invalid_exec_restriction_raises(self):
        with pytest.raises(ValueError):
            ExecRestriction(99)

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            Status(-1)


# ---------------------------------------------------------------------------
# NamedTuple structural properties
# ---------------------------------------------------------------------------

class TestOMIEOrderStructure:
    """OMIEOrder must be a proper NamedTuple with 13 fields."""

    def test_is_tuple_subclass(self):
        order = make_order(
            contract_id=0, side=Side.BUY, price=10.0, quantity=1.0,
            order_id=1, agent_id=0, timestamp=0,
        )
        assert isinstance(order, tuple)

    def test_has_13_fields_via_len(self):
        order = make_order(
            contract_id=0, side=Side.BUY, price=10.0, quantity=1.0,
            order_id=1, agent_id=0, timestamp=0,
        )
        assert len(order) == 13

    def test_fields_attribute(self):
        """OMIEOrder._fields must list all 13 field names in order."""
        expected = (
            "contract_id", "side", "price", "quantity", "order_id",
            "agent_id", "timestamp", "exec_restriction", "validity",
            "expiry_time", "status", "peak_qty", "ppd",
        )
        assert OMIEOrder._fields == expected

    def test_positional_access_matches_named_access(self):
        """Index-based access must match attribute access for all fields."""
        order = make_order(
            contract_id=5, side=Side.SELL, price=42.0, quantity=3.0,
            order_id=99, agent_id=1, timestamp=500,
            exec_restriction=ExecRestriction.IOC,
            validity=Validity.GTD,
            expiry_time=600,
            status=Status.PARTIALLY_FILLED,
            peak_qty=0.0,
            ppd=0.0,
        )
        for i, field_name in enumerate(OMIEOrder._fields):
            assert order[i] == getattr(order, field_name), (
                f"Field {field_name}: index {i} = {order[i]}, "
                f"attr = {getattr(order, field_name)}"
            )

    def test_order_is_immutable(self):
        """NamedTuple instances must be immutable."""
        order = make_order(
            contract_id=0, side=Side.BUY, price=10.0, quantity=1.0,
            order_id=1, agent_id=0, timestamp=0,
        )
        with pytest.raises(AttributeError):
            order.price = 99.0

    def test_tuple_unpacking(self):
        """Order must support tuple unpacking to all 13 fields."""
        order = make_order(
            contract_id=1, side=Side.BUY, price=20.0, quantity=5.0,
            order_id=10, agent_id=0, timestamp=50,
        )
        (cid, side, price, qty, oid, aid, ts,
         er, val, exp, st, pq, ppd) = order
        assert cid == 1
        assert side == Side.BUY
        assert price == 20.0
        assert qty == 5.0
        assert oid == 10
        assert aid == 0
        assert ts == 50
        assert er == ExecRestriction.NON
        assert val == Validity.GFS
        assert exp == 0
        assert st == Status.ACTIVE
        assert pq == 0.0
        assert ppd == 0.0


# ---------------------------------------------------------------------------
# Negative price support — OMIE allows negative prices (§2.3)
# ---------------------------------------------------------------------------

class TestNegativePriceOrder:
    """OMIE allows prices in [-9999, +9999]. Orders with negative prices
    must be representable (market-spec §2.3)."""

    def test_negative_price_stored(self):
        order = make_order(
            contract_id=10, side=Side.BUY, price=-50.25,
            quantity=1.0, order_id=1, agent_id=0, timestamp=0,
        )
        assert order.price == -50.25

    def test_zero_price_stored(self):
        order = make_order(
            contract_id=10, side=Side.SELL, price=0.0,
            quantity=1.0, order_id=1, agent_id=0, timestamp=0,
        )
        assert order.price == 0.0

    def test_extreme_negative_price(self):
        order = make_order(
            contract_id=0, side=Side.BUY, price=-9999.0,
            quantity=0.1, order_id=1, agent_id=0, timestamp=0,
        )
        assert order.price == -9999.0


# ---------------------------------------------------------------------------
# Boundary contract_id values — 96 contracts (0..95)
# ---------------------------------------------------------------------------

class TestContractIdBoundary:
    """Contract IDs range from 0 to 95 for 96 quarter-hourly contracts."""

    def test_first_contract(self):
        order = make_order(
            contract_id=0, side=Side.BUY, price=10.0, quantity=1.0,
            order_id=1, agent_id=0, timestamp=0,
        )
        assert order.contract_id == 0

    def test_last_contract(self):
        order = make_order(
            contract_id=95, side=Side.SELL, price=10.0, quantity=1.0,
            order_id=1, agent_id=0, timestamp=0,
        )
        assert order.contract_id == 95


# ---------------------------------------------------------------------------
# make_order with all explicit args — no default leakage
# ---------------------------------------------------------------------------

class TestMakeOrderExplicitArgs:
    """When all 13 args are provided explicitly, every field must match."""

    def test_all_args_explicit(self):
        order = make_order(
            contract_id=50,
            side=Side.SELL,
            price=123.45,
            quantity=77.7,
            order_id=9999,
            agent_id=3,
            timestamp=1000000,
            exec_restriction=ExecRestriction.ICE,
            validity=Validity.GTD,
            expiry_time=2000000,
            status=Status.CANCELLED,
            peak_qty=15.5,
            ppd=-0.25,
        )
        assert order.contract_id == 50
        assert order.side == Side.SELL
        assert order.price == 123.45
        assert order.quantity == 77.7
        assert order.order_id == 9999
        assert order.agent_id == 3
        assert order.timestamp == 1000000
        assert order.exec_restriction == ExecRestriction.ICE
        assert order.validity == Validity.GTD
        assert order.expiry_time == 2000000
        assert order.status == Status.CANCELLED
        assert order.peak_qty == 15.5
        assert order.ppd == -0.25


# ---------------------------------------------------------------------------
# OMIEOrder direct construction (bypassing factory)
# ---------------------------------------------------------------------------

class TestOMIEOrderDirectConstruction:
    """OMIEOrder can be constructed directly without make_order."""

    def test_direct_construction_positional(self):
        order = OMIEOrder(1, Side.BUY, 10.0, 5.0, 100, 0, 50,
                          ExecRestriction.NON, Validity.GFS, 0,
                          Status.ACTIVE, 0.0, 0.0)
        assert order.contract_id == 1
        assert order.ppd == 0.0

    def test_direct_construction_keyword(self):
        order = OMIEOrder(
            contract_id=2, side=Side.SELL, price=20.0, quantity=3.0,
            order_id=200, agent_id=1, timestamp=100,
            exec_restriction=ExecRestriction.FOK, validity=Validity.GTD,
            expiry_time=150, status=Status.REJECTED,
            peak_qty=0.0, ppd=0.0,
        )
        assert order.status == Status.REJECTED
        assert order.validity == Validity.GTD


# ---------------------------------------------------------------------------
# ICEwPI negative PPD — buy icebergs use negative delta (§4.2)
# ---------------------------------------------------------------------------

class TestICEwPINegativePPD:
    """ICEwPI buy icebergs use a negative PPD (price decreases per slice).
    The order type layer must accept negative ppd values (§4.2)."""

    def test_negative_ppd_for_buy_iceberg(self):
        order = make_order(
            contract_id=37, side=Side.BUY, price=100.0, quantity=500.0,
            order_id=7001, agent_id=0, timestamp=700,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=50.0,
            ppd=-1.00,
        )
        assert order.ppd == -1.00
        assert order.peak_qty == 50.0
        assert order.exec_restriction == ExecRestriction.ICE
        assert order.side == Side.BUY


# ---------------------------------------------------------------------------
# Order equality and identity
# ---------------------------------------------------------------------------

class TestOrderEquality:
    """Two orders with identical fields must compare as equal (tuple semantics)."""

    def test_same_fields_are_equal(self):
        a = make_order(
            contract_id=1, side=Side.BUY, price=10.0, quantity=1.0,
            order_id=1, agent_id=0, timestamp=0,
        )
        b = make_order(
            contract_id=1, side=Side.BUY, price=10.0, quantity=1.0,
            order_id=1, agent_id=0, timestamp=0,
        )
        assert a == b

    def test_different_fields_are_not_equal(self):
        a = make_order(
            contract_id=1, side=Side.BUY, price=10.0, quantity=1.0,
            order_id=1, agent_id=0, timestamp=0,
        )
        b = make_order(
            contract_id=1, side=Side.SELL, price=10.0, quantity=1.0,
            order_id=1, agent_id=0, timestamp=0,
        )
        assert a != b


# ---------------------------------------------------------------------------
# make_order return type
# ---------------------------------------------------------------------------

class TestMakeOrderReturnType:
    """make_order must return an OMIEOrder instance."""

    def test_return_type_is_omie_order(self):
        order = make_order(
            contract_id=0, side=Side.BUY, price=10.0, quantity=1.0,
            order_id=1, agent_id=0, timestamp=0,
        )
        assert isinstance(order, OMIEOrder)


# ---------------------------------------------------------------------------
# Status enum — all non-terminal and terminal states present
# ---------------------------------------------------------------------------

class TestStatusEnumCompleteness:
    """Verify that specific domain-required statuses are present
    and accessible by name (market-spec §5.1)."""

    def test_active_present(self):
        assert hasattr(Status, "ACTIVE")

    def test_partially_filled_present(self):
        assert hasattr(Status, "PARTIALLY_FILLED")

    def test_filled_present(self):
        assert hasattr(Status, "FILLED")

    def test_cancelled_present(self):
        assert hasattr(Status, "CANCELLED")

    def test_rejected_present(self):
        assert hasattr(Status, "REJECTED")

    def test_expired_present(self):
        assert hasattr(Status, "EXPIRED")
