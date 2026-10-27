"""Tests for M1.2 — Order Type Definitions.

Behavioral contracts for OMIE order representation and associated enums.
All expected values are derived from docs/omiecid/market-spec.md §4, §4.5,
§5.1, §13 and docs/omiecid/agile-plan.md M1.2.

These tests are black-box: they import only the public interface and
assert against domain-specified values. No implementation assumptions.
"""

import pytest

from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    Validity,
    Status,
    make_order,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def non_order():
    """A regular (NON) limit buy order with typical field values."""
    return make_order(
        contract_id=37,
        side=Side.BUY,
        price=45.50,
        quantity=10.0,
        order_id=1001,
        agent_id=0,
        timestamp=100,
    )


@pytest.fixture
def ioc_order():
    """An IOC (immediate-or-cancel) sell order."""
    return make_order(
        contract_id=10,
        side=Side.SELL,
        price=52.30,
        quantity=5.0,
        order_id=2001,
        agent_id=1,
        timestamp=200,
        exec_restriction=ExecRestriction.IOC,
    )


@pytest.fixture
def fok_order():
    """A FOK (fill-or-kill) buy order."""
    return make_order(
        contract_id=20,
        side=Side.BUY,
        price=48.00,
        quantity=20.0,
        order_id=3001,
        agent_id=0,
        timestamp=300,
        exec_restriction=ExecRestriction.FOK,
    )


@pytest.fixture
def ice_order():
    """An iceberg (ICE) sell order with visible peak and price increment."""
    return make_order(
        contract_id=42,
        side=Side.SELL,
        price=55.00,
        quantity=100.0,
        order_id=4001,
        agent_id=1,
        timestamp=400,
        exec_restriction=ExecRestriction.ICE,
        peak_qty=10.0,
        ppd=0.50,
    )


# ---------------------------------------------------------------------------
# TestSideEnum — market-spec §4.1 (BUY/SELL)
# ---------------------------------------------------------------------------

class TestSideEnum:
    """Side enum must define BUY and SELL as integer values."""

    def test_buy_is_integer(self):
        """Side.BUY must be usable as an integer."""
        assert isinstance(Side.BUY.value, int)

    def test_sell_is_integer(self):
        """Side.SELL must be usable as an integer."""
        assert isinstance(Side.SELL.value, int)

    def test_buy_and_sell_are_distinct(self):
        """BUY and SELL must have different integer values."""
        assert Side.BUY.value != Side.SELL.value

    def test_side_has_exactly_two_members(self):
        """Side enum must have exactly BUY and SELL — no extras."""
        assert len(Side) == 2
        assert set(Side.__members__.keys()) == {"BUY", "SELL"}


# ---------------------------------------------------------------------------
# TestExecRestrictionEnum — market-spec §4.5 (NON, IOC, FOK, ICE)
# ---------------------------------------------------------------------------

class TestExecRestrictionEnum:
    """ExecRestriction enum must define NON, IOC, FOK, ICE with
    integer values matching the OMIE data encoding (§4.5, §13)."""

    def test_non_is_integer(self):
        assert isinstance(ExecRestriction.NON.value, int)

    def test_ioc_is_integer(self):
        assert isinstance(ExecRestriction.IOC.value, int)

    def test_fok_is_integer(self):
        assert isinstance(ExecRestriction.FOK.value, int)

    def test_ice_value_is_3(self):
        """ICE must have value 3 per agile-plan M1.2 and data encoding."""
        assert ExecRestriction.ICE.value == 3

    def test_all_values_are_distinct(self):
        """All four restriction values must be unique integers."""
        values = [er.value for er in ExecRestriction]
        assert len(values) == len(set(values))

    def test_has_exactly_four_members(self):
        """ExecRestriction must have exactly NON, IOC, FOK, ICE."""
        assert len(ExecRestriction) == 4
        assert set(ExecRestriction.__members__.keys()) == {
            "NON", "IOC", "FOK", "ICE"
        }


# ---------------------------------------------------------------------------
# TestValidityEnum — market-spec §4.6 (GFS, GTD)
# ---------------------------------------------------------------------------

class TestValidityEnum:
    """Validity enum must define GFS and GTD as integer values."""

    def test_gfs_is_integer(self):
        assert isinstance(Validity.GFS.value, int)

    def test_gtd_is_integer(self):
        assert isinstance(Validity.GTD.value, int)

    def test_gfs_and_gtd_are_distinct(self):
        assert Validity.GFS.value != Validity.GTD.value

    def test_has_exactly_two_members(self):
        assert len(Validity) == 2
        assert set(Validity.__members__.keys()) == {"GFS", "GTD"}


# ---------------------------------------------------------------------------
# TestStatusEnum — market-spec §5.1 (order lifecycle states)
# ---------------------------------------------------------------------------

class TestStatusEnum:
    """Status enum must define all order lifecycle states from §5.1."""

    EXPECTED_MEMBERS = {
        "ACTIVE", "PARTIALLY_FILLED", "FILLED",
        "CANCELLED", "REJECTED", "EXPIRED",
    }

    def test_all_status_values_are_integers(self):
        """Every Status member must have an integer value."""
        for status in Status:
            assert isinstance(status.value, int), (
                f"Status.{status.name} value is not int: {type(status.value)}"
            )

    def test_all_status_values_are_distinct(self):
        """All status values must be unique."""
        values = [s.value for s in Status]
        assert len(values) == len(set(values))

    def test_has_expected_members(self):
        """Status enum must have exactly the 6 lifecycle states."""
        assert len(Status) == 6
        assert set(Status.__members__.keys()) == self.EXPECTED_MEMBERS


# ---------------------------------------------------------------------------
# TestNonOrderFields — basic regular (NON) limit order (§4.1)
# ---------------------------------------------------------------------------

class TestNonOrderFields:
    """A regular NON order must carry all 13 fields with correct values."""

    def test_contract_id(self, non_order):
        assert non_order.contract_id == 37

    def test_side(self, non_order):
        assert non_order.side == Side.BUY

    def test_price(self, non_order):
        assert non_order.price == 45.50

    def test_quantity(self, non_order):
        assert non_order.quantity == 10.0

    def test_order_id(self, non_order):
        assert non_order.order_id == 1001

    def test_agent_id(self, non_order):
        assert non_order.agent_id == 0

    def test_timestamp(self, non_order):
        assert non_order.timestamp == 100

    def test_exec_restriction_defaults_to_non(self, non_order):
        """Regular orders default to NON execution restriction (§4.1)."""
        assert non_order.exec_restriction == ExecRestriction.NON

    def test_validity_defaults_to_gfs(self, non_order):
        """Regular orders default to GFS validity (§4.6)."""
        assert non_order.validity == Validity.GFS

    def test_expiry_time_defaults_to_zero(self, non_order):
        """expiry_time=0 means 'not set / use contract GCT'."""
        assert non_order.expiry_time == 0

    def test_status_defaults_to_active(self, non_order):
        """Newly created orders start as ACTIVE."""
        assert non_order.status == Status.ACTIVE

    def test_peak_qty_defaults_to_zero(self, non_order):
        """Non-iceberg orders must have peak_qty=0.0."""
        assert non_order.peak_qty == 0.0

    def test_ppd_defaults_to_zero(self, non_order):
        """Non-iceberg orders must have ppd=0.0."""
        assert non_order.ppd == 0.0


# ---------------------------------------------------------------------------
# TestIOCOrderFields — IOC execution restriction (§4.5)
# ---------------------------------------------------------------------------

class TestIOCOrderFields:
    """IOC orders: partial fill allowed, no book entry if unmatched (§4.5).
    At the type-definition level, the order must carry IOC restriction
    and default iceberg fields (peak_qty=0.0, ppd=0.0)."""

    def test_exec_restriction_is_ioc(self, ioc_order):
        assert ioc_order.exec_restriction == ExecRestriction.IOC

    def test_side(self, ioc_order):
        assert ioc_order.side == Side.SELL

    def test_price(self, ioc_order):
        assert ioc_order.price == 52.30

    def test_quantity(self, ioc_order):
        assert ioc_order.quantity == 5.0

    def test_peak_qty_zero_for_ioc(self, ioc_order):
        """IOC orders are not icebergs — peak_qty must be 0.0."""
        assert ioc_order.peak_qty == 0.0

    def test_ppd_zero_for_ioc(self, ioc_order):
        """IOC orders are not icebergs — ppd must be 0.0."""
        assert ioc_order.ppd == 0.0

    def test_validity_defaults_to_gfs(self, ioc_order):
        assert ioc_order.validity == Validity.GFS


# ---------------------------------------------------------------------------
# TestFOKOrderFields — FOK execution restriction (§4.5)
# ---------------------------------------------------------------------------

class TestFOKOrderFields:
    """FOK orders: full quantity or cancel, no book entry (§4.5).
    At the type-definition level, the order must carry FOK restriction
    and default iceberg fields."""

    def test_exec_restriction_is_fok(self, fok_order):
        assert fok_order.exec_restriction == ExecRestriction.FOK

    def test_side(self, fok_order):
        assert fok_order.side == Side.BUY

    def test_price(self, fok_order):
        assert fok_order.price == 48.00

    def test_quantity(self, fok_order):
        assert fok_order.quantity == 20.0

    def test_peak_qty_zero_for_fok(self, fok_order):
        """FOK orders are not icebergs — peak_qty must be 0.0."""
        assert fok_order.peak_qty == 0.0

    def test_ppd_zero_for_fok(self, fok_order):
        """FOK orders are not icebergs — ppd must be 0.0."""
        assert fok_order.ppd == 0.0


# ---------------------------------------------------------------------------
# TestICEOrderFields — iceberg order (§4.2, §4.5, §13)
# ---------------------------------------------------------------------------

class TestICEOrderFields:
    """ICE (iceberg) orders carry visible peak_qty and optional price
    increment (ppd). ExecRestriction must be ICE (§4.5, §13)."""

    def test_exec_restriction_is_ice(self, ice_order):
        assert ice_order.exec_restriction == ExecRestriction.ICE

    def test_side(self, ice_order):
        assert ice_order.side == Side.SELL

    def test_price(self, ice_order):
        assert ice_order.price == 55.00

    def test_total_quantity(self, ice_order):
        """quantity field carries the total (hidden) quantity for icebergs."""
        assert ice_order.quantity == 100.0

    def test_peak_qty_nonzero(self, ice_order):
        """Iceberg orders must have a non-zero peak_qty (visible slice)."""
        assert ice_order.peak_qty == 10.0

    def test_ppd(self, ice_order):
        """PPD carries the per-slice price increment (ICEwPI, §4.2)."""
        assert ice_order.ppd == 0.50

    def test_ice_with_zero_ppd(self):
        """A standard iceberg (no price increment) has ppd=0.0 (§4.2)."""
        order = make_order(
            contract_id=42,
            side=Side.SELL,
            price=55.00,
            quantity=100.0,
            order_id=4002,
            agent_id=1,
            timestamp=401,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=10.0,
            ppd=0.0,
        )
        assert order.ppd == 0.0
        assert order.peak_qty == 10.0
        assert order.exec_restriction == ExecRestriction.ICE


# ---------------------------------------------------------------------------
# TestOrderDefaultIcebergFields — non-iceberg orders (§4.5, agile-plan M1.2)
# ---------------------------------------------------------------------------

class TestOrderDefaultIcebergFields:
    """Non-iceberg orders (NON, IOC, FOK) must default peak_qty=0.0
    and ppd=0.0. These fields are forward-compatible placeholders for
    M4 iceberg logic and M7 data parsing."""

    def test_non_order_defaults(self, non_order):
        assert non_order.peak_qty == 0.0
        assert non_order.ppd == 0.0

    def test_ioc_order_defaults(self, ioc_order):
        assert ioc_order.peak_qty == 0.0
        assert ioc_order.ppd == 0.0

    def test_fok_order_defaults(self, fok_order):
        assert fok_order.peak_qty == 0.0
        assert fok_order.ppd == 0.0

    def test_make_order_without_iceberg_args(self):
        """Calling make_order without peak_qty/ppd must produce 0.0 defaults."""
        order = make_order(
            contract_id=1,
            side=Side.BUY,
            price=10.0,
            quantity=1.0,
            order_id=1,
            agent_id=0,
            timestamp=0,
        )
        assert order.peak_qty == 0.0
        assert order.ppd == 0.0


# ---------------------------------------------------------------------------
# TestOrderFieldAccess — 13 fields accessible by name
# ---------------------------------------------------------------------------

class TestOrderFieldAccess:
    """All 13 order fields must be accessible by name (attribute access)."""

    EXPECTED_FIELDS = [
        "contract_id", "side", "price", "quantity", "order_id",
        "agent_id", "timestamp", "exec_restriction", "validity",
        "expiry_time", "status", "peak_qty", "ppd",
    ]

    def test_all_fields_accessible(self, non_order):
        """Every field in the 13-field spec must be accessible by name."""
        for field_name in self.EXPECTED_FIELDS:
            assert hasattr(non_order, field_name), (
                f"Order missing field: {field_name}"
            )

    def test_field_count(self, non_order):
        """Order must have exactly 13 named fields."""
        # Count fields that match the spec (ignoring any internal methods/attrs)
        present = [f for f in self.EXPECTED_FIELDS if hasattr(non_order, f)]
        assert len(present) == 13


# ---------------------------------------------------------------------------
# TestOrderWithGTDValidity — GTD orders (§4.6)
# ---------------------------------------------------------------------------

class TestOrderWithGTDValidity:
    """Orders may specify GTD validity with an explicit expiry_time."""

    def test_gtd_order_validity(self):
        order = make_order(
            contract_id=5,
            side=Side.SELL,
            price=30.00,
            quantity=2.0,
            order_id=5001,
            agent_id=0,
            timestamp=500,
            validity=Validity.GTD,
            expiry_time=600,
        )
        assert order.validity == Validity.GTD
        assert order.expiry_time == 600

    def test_gtd_order_still_has_default_iceberg_fields(self):
        """GTD validity does not imply iceberg fields."""
        order = make_order(
            contract_id=5,
            side=Side.SELL,
            price=30.00,
            quantity=2.0,
            order_id=5002,
            agent_id=0,
            timestamp=501,
            validity=Validity.GTD,
            expiry_time=700,
        )
        assert order.peak_qty == 0.0
        assert order.ppd == 0.0
