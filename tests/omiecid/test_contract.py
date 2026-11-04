"""Tests for M2.1 + M2.2 — Contract Dataclass, Status Transitions, and Factory.

Behavioral contracts for ContractStatus enum, Contract dataclass, and the
generate_contracts factory function.
All expected values are derived from docs/omiecid/market-spec.md §2.1–2.2, §3.1
and docs/omiecid/migration-plan.md §M2.

These tests are black-box: they import only the public interface and
assert against domain-specified values.

M2.1 tests: TestContractStatusEnum through TestContractSuspended (lines ~60–373).
M2.2 tests: TestFactoryReturns96Contracts onward (appended below M2.1 tests).
"""

import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus, generate_contracts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def contract_0():
    """Contract 0: first quarter-hour of the day (00:00-00:15)."""
    return Contract(
        contract_id=0,
        delivery_start_minutes=0,
        delivery_end_minutes=15,
        gct_minutes=0 - 60,  # -60: delivery starts at midnight, GCT 1h before
        status=ContractStatus.PENDING,
    )


@pytest.fixture
def contract_95():
    """Contract 95: last quarter-hour of the day (23:45-24:00)."""
    return Contract(
        contract_id=95,
        delivery_start_minutes=23 * 60 + 45,  # 1425
        delivery_end_minutes=24 * 60,          # 1440
        gct_minutes=60 * (95 // 4) - 60,      # 1320: hour-aligned, closes with contracts 92-94 (§3.1 v2.10)
        status=ContractStatus.PENDING,
    )


@pytest.fixture
def contract_37():
    """Contract 37: 09:15-09:30 (mid-day reference)."""
    return Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,   # 555
        delivery_end_minutes=9 * 60 + 30,      # 570
        gct_minutes=60 * (37 // 4) - 60,       # 480: hour-aligned, closes with contracts 36-39 (§3.1 v2.10)
        status=ContractStatus.PENDING,
    )


# ---------------------------------------------------------------------------
# TestContractStatusEnum — ContractStatus values and properties
# ---------------------------------------------------------------------------

class TestContractStatusEnum:
    """ContractStatus enum must have the 4 lifecycle states with correct integer values."""

    def test_pending_value(self):
        """PENDING = 0 (before GOT)."""
        assert ContractStatus.PENDING == 0

    def test_open_value(self):
        """OPEN = 1 (accepting orders, matching active)."""
        assert ContractStatus.OPEN == 1

    def test_suspended_value(self):
        """SUSPENDED = 2 (IDA clearing window)."""
        assert ContractStatus.SUSPENDED == 2

    def test_closed_value(self):
        """CLOSED = 3 (past GCT)."""
        assert ContractStatus.CLOSED == 3

    def test_member_count(self):
        """Exactly 4 status values."""
        assert len(ContractStatus) == 4

    def test_all_values_are_integers(self):
        """All members are integers (IntEnum contract)."""
        for member in ContractStatus:
            assert isinstance(member, int)

    def test_all_values_distinct(self):
        """All member values are distinct."""
        values = [m.value for m in ContractStatus]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# TestContractIdToDeliveryTime — contract_id maps to correct delivery windows
# ---------------------------------------------------------------------------

class TestContractIdToDeliveryTime:
    """Contract ID determines the delivery window: ID * 15 min from midnight."""

    def test_contract_0_delivery_start(self, contract_0):
        """Contract 0 delivery starts at 00:00 (0 minutes from midnight)."""
        assert contract_0.delivery_start_minutes == 0

    def test_contract_0_delivery_end(self, contract_0):
        """Contract 0 delivery ends at 00:15 (15 minutes from midnight)."""
        assert contract_0.delivery_end_minutes == 15

    def test_contract_95_delivery_start(self, contract_95):
        """Contract 95 delivery starts at 23:45 (1425 minutes from midnight)."""
        assert contract_95.delivery_start_minutes == 1425

    def test_contract_95_delivery_end(self, contract_95):
        """Contract 95 delivery ends at 24:00 (1440 minutes from midnight)."""
        assert contract_95.delivery_end_minutes == 1440

    def test_contract_37_delivery_start(self, contract_37):
        """Contract 37 delivery starts at 09:15 (555 minutes from midnight)."""
        assert contract_37.delivery_start_minutes == 555

    def test_contract_37_delivery_end(self, contract_37):
        """Contract 37 delivery ends at 09:30 (570 minutes from midnight)."""
        assert contract_37.delivery_end_minutes == 570

    def test_delivery_window_is_15_minutes(self, contract_0):
        """Every contract spans exactly 15 minutes (1 MTU)."""
        assert contract_0.delivery_end_minutes - contract_0.delivery_start_minutes == 15

    def test_delivery_window_is_15_minutes_last(self, contract_95):
        """Last contract also spans exactly 15 minutes."""
        assert contract_95.delivery_end_minutes - contract_95.delivery_start_minutes == 15

    def test_delivery_window_is_15_minutes_mid(self, contract_37):
        """Mid-day contract also spans exactly 15 minutes."""
        assert contract_37.delivery_end_minutes - contract_37.delivery_start_minutes == 15

    def test_contract_id_field(self, contract_0):
        """Contract stores its contract_id."""
        assert contract_0.contract_id == 0

    def test_contract_id_field_last(self, contract_95):
        """Last contract stores contract_id=95."""
        assert contract_95.contract_id == 95


# ---------------------------------------------------------------------------
# TestGCTIsHourAligned — GCT(k) = 60 * (k // 4) - 60
# ---------------------------------------------------------------------------

class TestGCTIsHourAligned:
    """GCT is hour-aligned (market-spec §3.1, v2.10 correction; §11.1).

    GCT(k) = 60 * floor(k / 4) - 60 minutes from D 00:00: one hour before the
    start of the delivery *hour* containing quarter-hour k, so the four
    quarter-hours of an hour close together.
    """

    def test_gct_contract_0(self, contract_0):
        """Contract 0: GCT = 0 - 60 = -60 minutes (before midnight on previous day)."""
        assert contract_0.gct_minutes == -60

    def test_gct_contract_95(self, contract_95):
        """Contract 95 (23:45): GCT = 60 * 23 - 60 = 1320 minutes (22:00)."""
        assert contract_95.gct_minutes == 1320

    def test_gct_contract_37(self, contract_37):
        """Contract 37 (09:15): GCT = 60 * 9 - 60 = 480 minutes (08:00)."""
        assert contract_37.gct_minutes == 480

    def test_gct_is_hour_aligned(self, contract_0):
        """GCT formula: 60 * (contract_id // 4) - 60."""
        assert contract_0.gct_minutes == 60 * (contract_0.contract_id // 4) - 60

    def test_gct_is_hour_aligned_last(self, contract_95):
        """GCT formula holds for last contract."""
        assert contract_95.gct_minutes == 60 * (contract_95.contract_id // 4) - 60

    def test_gct_is_hour_aligned_mid(self, contract_37):
        """GCT formula holds for mid-day contract."""
        assert contract_37.gct_minutes == 60 * (contract_37.contract_id // 4) - 60

    def test_early_contracts_have_negative_gct(self):
        """Contracts 0-3 (00:00-01:00) have GCT before midnight (negative minutes).

        This is expected: these contracts close on day D-1 before midnight.
        """
        for cid in range(4):
            start = cid * 15
            gct = start - 60
            c = Contract(
                contract_id=cid,
                delivery_start_minutes=start,
                delivery_end_minutes=start + 15,
                gct_minutes=gct,
                status=ContractStatus.PENDING,
            )
            assert c.gct_minutes < 0


# ---------------------------------------------------------------------------
# TestContractTransitionsPendingToOpen — PENDING -> OPEN at GOT
# ---------------------------------------------------------------------------

class TestContractTransitionsPendingToOpen:
    """Contracts begin in PENDING state and transition to OPEN at GOT."""

    def test_initial_status_is_pending(self, contract_0):
        """A newly created contract starts in PENDING status."""
        assert contract_0.status == ContractStatus.PENDING

    def test_initial_status_is_pending_last(self, contract_95):
        """Last contract also starts PENDING."""
        assert contract_95.status == ContractStatus.PENDING

    def test_pending_to_open_transition(self, contract_0):
        """Contract can transition from PENDING to OPEN."""
        opened = Contract(
            contract_id=contract_0.contract_id,
            delivery_start_minutes=contract_0.delivery_start_minutes,
            delivery_end_minutes=contract_0.delivery_end_minutes,
            gct_minutes=contract_0.gct_minutes,
            status=ContractStatus.OPEN,
        )
        assert opened.status == ContractStatus.OPEN

    def test_open_status_value(self, contract_0):
        """OPEN status has integer value 1."""
        opened = Contract(
            contract_id=contract_0.contract_id,
            delivery_start_minutes=contract_0.delivery_start_minutes,
            delivery_end_minutes=contract_0.delivery_end_minutes,
            gct_minutes=contract_0.gct_minutes,
            status=ContractStatus.OPEN,
        )
        assert opened.status == 1

    def test_pending_is_not_open(self, contract_0):
        """PENDING and OPEN are distinct states."""
        assert contract_0.status != ContractStatus.OPEN

    def test_delivery_times_unchanged_after_open(self, contract_37):
        """Opening a contract does not change its delivery window or GCT."""
        opened = Contract(
            contract_id=contract_37.contract_id,
            delivery_start_minutes=contract_37.delivery_start_minutes,
            delivery_end_minutes=contract_37.delivery_end_minutes,
            gct_minutes=contract_37.gct_minutes,
            status=ContractStatus.OPEN,
        )
        assert opened.delivery_start_minutes == contract_37.delivery_start_minutes
        assert opened.delivery_end_minutes == contract_37.delivery_end_minutes
        assert opened.gct_minutes == contract_37.gct_minutes


# ---------------------------------------------------------------------------
# TestContractTransitionsOpenToClosed — OPEN -> CLOSED at GCT
# ---------------------------------------------------------------------------

class TestContractTransitionsOpenToClosed:
    """Contracts transition from OPEN to CLOSED at GCT."""

    def test_open_to_closed_transition(self, contract_0):
        """Contract can transition from OPEN to CLOSED."""
        closed = Contract(
            contract_id=contract_0.contract_id,
            delivery_start_minutes=contract_0.delivery_start_minutes,
            delivery_end_minutes=contract_0.delivery_end_minutes,
            gct_minutes=contract_0.gct_minutes,
            status=ContractStatus.CLOSED,
        )
        assert closed.status == ContractStatus.CLOSED

    def test_closed_status_value(self):
        """CLOSED status has integer value 3."""
        c = Contract(
            contract_id=50,
            delivery_start_minutes=50 * 15,
            delivery_end_minutes=50 * 15 + 15,
            gct_minutes=50 * 15 - 60,
            status=ContractStatus.CLOSED,
        )
        assert c.status == 3

    def test_closed_is_not_open(self, contract_0):
        """CLOSED and OPEN are distinct states."""
        assert ContractStatus.CLOSED != ContractStatus.OPEN

    def test_delivery_times_unchanged_after_close(self, contract_95):
        """Closing a contract does not change its delivery window or GCT."""
        closed = Contract(
            contract_id=contract_95.contract_id,
            delivery_start_minutes=contract_95.delivery_start_minutes,
            delivery_end_minutes=contract_95.delivery_end_minutes,
            gct_minutes=contract_95.gct_minutes,
            status=ContractStatus.CLOSED,
        )
        assert closed.delivery_start_minutes == contract_95.delivery_start_minutes
        assert closed.delivery_end_minutes == contract_95.delivery_end_minutes
        assert closed.gct_minutes == contract_95.gct_minutes


# ---------------------------------------------------------------------------
# TestContractFields — field access and dataclass properties
# ---------------------------------------------------------------------------

class TestContractFields:
    """Contract exposes all 5 fields with correct types and named access."""

    def test_has_contract_id_field(self, contract_37):
        """Contract has contract_id field."""
        assert hasattr(contract_37, 'contract_id')

    def test_has_delivery_start_minutes_field(self, contract_37):
        """Contract has delivery_start_minutes field."""
        assert hasattr(contract_37, 'delivery_start_minutes')

    def test_has_delivery_end_minutes_field(self, contract_37):
        """Contract has delivery_end_minutes field."""
        assert hasattr(contract_37, 'delivery_end_minutes')

    def test_has_gct_minutes_field(self, contract_37):
        """Contract has gct_minutes field."""
        assert hasattr(contract_37, 'gct_minutes')

    def test_has_status_field(self, contract_37):
        """Contract has status field."""
        assert hasattr(contract_37, 'status')

    def test_contract_id_is_int(self, contract_37):
        """contract_id is an integer."""
        assert isinstance(contract_37.contract_id, int)

    def test_delivery_start_is_int(self, contract_37):
        """delivery_start_minutes is an integer."""
        assert isinstance(contract_37.delivery_start_minutes, int)

    def test_delivery_end_is_int(self, contract_37):
        """delivery_end_minutes is an integer."""
        assert isinstance(contract_37.delivery_end_minutes, int)

    def test_gct_is_int(self, contract_37):
        """gct_minutes is an integer."""
        assert isinstance(contract_37.gct_minutes, int)

    def test_status_is_contract_status(self, contract_37):
        """status is a ContractStatus member."""
        assert isinstance(contract_37.status, ContractStatus)


# ---------------------------------------------------------------------------
# TestContractSuspended — SUSPENDED state exists (IDA clearing window)
# ---------------------------------------------------------------------------

class TestContractSuspended:
    """SUSPENDED state is available for IDA clearing windows (§3.3)."""

    def test_suspended_status_exists(self):
        """ContractStatus has a SUSPENDED member."""
        assert hasattr(ContractStatus, 'SUSPENDED')

    def test_can_create_suspended_contract(self):
        """A contract can be created with SUSPENDED status."""
        c = Contract(
            contract_id=10,
            delivery_start_minutes=10 * 15,
            delivery_end_minutes=10 * 15 + 15,
            gct_minutes=10 * 15 - 60,
            status=ContractStatus.SUSPENDED,
        )
        assert c.status == ContractStatus.SUSPENDED

    def test_suspended_is_distinct_from_open_and_closed(self):
        """SUSPENDED is not OPEN and not CLOSED."""
        assert ContractStatus.SUSPENDED != ContractStatus.OPEN
        assert ContractStatus.SUSPENDED != ContractStatus.CLOSED
        assert ContractStatus.SUSPENDED != ContractStatus.PENDING


# ===========================================================================
# M2.2 — Contract Factory Tests
#
# Behavioral contracts for generate_contracts() factory function.
# Domain source: market-spec.md §2.1–2.2 (96 quarter-hourly contracts per
# delivery day, Q1–Q96 covering 00:00–24:00 CET, GCT = delivery_start − 60).
# Architecture source: migration-plan.md §M2.
# ===========================================================================


# ---------------------------------------------------------------------------
# Fixtures for factory tests
# ---------------------------------------------------------------------------

@pytest.fixture
def all_contracts():
    """Generate all 96 contracts using the factory function."""
    return generate_contracts()


# ---------------------------------------------------------------------------
# TestFactoryReturns96Contracts — correct count and return type
# ---------------------------------------------------------------------------

class TestFactoryReturns96Contracts:
    """generate_contracts() must return exactly 96 Contract instances."""

    def test_returns_96_contracts(self, all_contracts):
        """Factory produces exactly 96 contracts (§2.1: 96 quarter-hourly)."""
        assert len(all_contracts) == 96

    def test_returns_sequence(self, all_contracts):
        """Factory returns a sequence (list or tuple) of contracts."""
        assert hasattr(all_contracts, '__len__')
        assert hasattr(all_contracts, '__getitem__')

    def test_all_elements_are_contract_instances(self, all_contracts):
        """Every element in the result is a Contract instance."""
        for c in all_contracts:
            assert isinstance(c, Contract)

    def test_no_none_elements(self, all_contracts):
        """No element in the result is None."""
        for c in all_contracts:
            assert c is not None


# ---------------------------------------------------------------------------
# TestFactoryContractIds — IDs are 0..95 in order
# ---------------------------------------------------------------------------

class TestFactoryContractIds:
    """Each contract must have contract_id matching its position (0–95)."""

    def test_first_contract_id_is_0(self, all_contracts):
        """First contract has contract_id=0."""
        assert all_contracts[0].contract_id == 0

    def test_last_contract_id_is_95(self, all_contracts):
        """Last contract has contract_id=95."""
        assert all_contracts[95].contract_id == 95

    def test_ids_are_sequential(self, all_contracts):
        """contract_id values are 0, 1, 2, ..., 95 in order."""
        ids = [c.contract_id for c in all_contracts]
        assert ids == list(range(96))

    def test_all_ids_unique(self, all_contracts):
        """All contract_id values are distinct."""
        ids = [c.contract_id for c in all_contracts]
        assert len(ids) == len(set(ids))

    def test_mid_contract_id(self, all_contracts):
        """Contract at index 48 has contract_id=48 (mid-day: 12:00–12:15)."""
        assert all_contracts[48].contract_id == 48


# ---------------------------------------------------------------------------
# TestFactoryDeliveryWindows — correct start/end for all 96 contracts
# ---------------------------------------------------------------------------

class TestFactoryDeliveryWindows:
    """Each contract's delivery window must be [id*15, id*15+15] minutes."""

    def test_contract_0_delivery_start(self, all_contracts):
        """Contract 0: delivery starts at 00:00 (0 minutes)."""
        assert all_contracts[0].delivery_start_minutes == 0

    def test_contract_0_delivery_end(self, all_contracts):
        """Contract 0: delivery ends at 00:15 (15 minutes)."""
        assert all_contracts[0].delivery_end_minutes == 15

    def test_contract_95_delivery_start(self, all_contracts):
        """Contract 95: delivery starts at 23:45 (1425 minutes)."""
        assert all_contracts[95].delivery_start_minutes == 1425

    def test_contract_95_delivery_end(self, all_contracts):
        """Contract 95: delivery ends at 24:00 (1440 minutes)."""
        assert all_contracts[95].delivery_end_minutes == 1440

    def test_contract_48_delivery_start(self, all_contracts):
        """Contract 48 (12:00–12:15): delivery starts at 720 minutes."""
        assert all_contracts[48].delivery_start_minutes == 48 * 15

    def test_contract_48_delivery_end(self, all_contracts):
        """Contract 48 (12:00–12:15): delivery ends at 735 minutes."""
        assert all_contracts[48].delivery_end_minutes == 48 * 15 + 15

    def test_all_delivery_starts_correct(self, all_contracts):
        """Every contract's delivery_start = contract_id * 15."""
        for c in all_contracts:
            assert c.delivery_start_minutes == c.contract_id * 15

    def test_all_delivery_ends_correct(self, all_contracts):
        """Every contract's delivery_end = contract_id * 15 + 15."""
        for c in all_contracts:
            assert c.delivery_end_minutes == c.contract_id * 15 + 15

    def test_all_windows_span_15_minutes(self, all_contracts):
        """Every contract spans exactly 15 minutes (1 MTU)."""
        for c in all_contracts:
            assert c.delivery_end_minutes - c.delivery_start_minutes == 15

    def test_first_delivery_starts_at_midnight(self, all_contracts):
        """First delivery period starts at 00:00 (minute 0)."""
        assert all_contracts[0].delivery_start_minutes == 0

    def test_last_delivery_ends_at_midnight(self, all_contracts):
        """Last delivery period ends at 24:00 (minute 1440)."""
        assert all_contracts[95].delivery_end_minutes == 1440

    def test_contracts_cover_full_day_contiguously(self, all_contracts):
        """Delivery windows cover 00:00–24:00 with no gaps or overlaps."""
        for i in range(95):
            assert all_contracts[i].delivery_end_minutes == all_contracts[i + 1].delivery_start_minutes


# ---------------------------------------------------------------------------
# TestFactoryGCTs — GCT(k) = 60 * (k // 4) - 60 for all contracts
# ---------------------------------------------------------------------------

class TestFactoryGCTs:
    """Every factory-generated contract has the hour-aligned GCT.

    GCT(k) = 60 * floor(k / 4) - 60 (market-spec §3.1, v2.10 correction; §11.1).
    """

    def test_contract_0_gct(self, all_contracts):
        """Contract 0: GCT = 0 - 60 = -60 (previous day)."""
        assert all_contracts[0].gct_minutes == -60

    def test_contract_4_gct(self, all_contracts):
        """Contract 4 (01:00–01:15): GCT = 60 - 60 = 0 (midnight exactly)."""
        assert all_contracts[4].gct_minutes == 0

    def test_contract_95_gct(self, all_contracts):
        """Contract 95 (23:45): GCT = 60 * 23 - 60 = 1320 (22:00)."""
        assert all_contracts[95].gct_minutes == 1320

    def test_all_gcts_correct(self, all_contracts):
        """Every contract's GCT = 60 * (contract_id // 4) - 60."""
        for c in all_contracts:
            assert c.gct_minutes == 60 * (c.contract_id // 4) - 60

    def test_first_4_contracts_have_negative_gct(self, all_contracts):
        """Contracts 0–3 (delivery before 01:00) have negative GCT (previous day)."""
        for i in range(4):
            assert all_contracts[i].gct_minutes < 0

    def test_contract_4_onwards_have_nonnegative_gct(self, all_contracts):
        """Contracts 4–95 (delivery at or after 01:00) have GCT >= 0."""
        for i in range(4, 96):
            assert all_contracts[i].gct_minutes >= 0

    def test_gcts_equal_within_hour_strictly_increasing_across_hours(
        self, all_contracts
    ):
        """The four quarter-hours of an hour share one GCT; GCTs increase hour to hour."""
        gcts = [c.gct_minutes for c in all_contracts]
        for i in range(len(gcts) - 1):
            if (i + 1) % 4 == 0:
                assert gcts[i] < gcts[i + 1]
            else:
                assert gcts[i] == gcts[i + 1]


# ---------------------------------------------------------------------------
# TestFactoryInitialStatus — all contracts start in PENDING
# ---------------------------------------------------------------------------

class TestFactoryInitialStatus:
    """All factory-generated contracts must start in PENDING status."""

    def test_first_contract_is_pending(self, all_contracts):
        """Contract 0 starts PENDING."""
        assert all_contracts[0].status == ContractStatus.PENDING

    def test_last_contract_is_pending(self, all_contracts):
        """Contract 95 starts PENDING."""
        assert all_contracts[95].status == ContractStatus.PENDING

    def test_all_contracts_are_pending(self, all_contracts):
        """Every contract from the factory starts in PENDING status."""
        for c in all_contracts:
            assert c.status == ContractStatus.PENDING

    def test_no_contract_is_open(self, all_contracts):
        """No factory-generated contract is OPEN (they haven't reached GOT yet)."""
        for c in all_contracts:
            assert c.status != ContractStatus.OPEN

    def test_no_contract_is_closed(self, all_contracts):
        """No factory-generated contract is CLOSED."""
        for c in all_contracts:
            assert c.status != ContractStatus.CLOSED


# ---------------------------------------------------------------------------
# TestFactoryCallableMultipleTimes — idempotency
# ---------------------------------------------------------------------------

class TestFactoryCallableMultipleTimes:
    """Factory can be called multiple times and produces consistent results."""

    def test_two_calls_produce_same_count(self):
        """Two calls to generate_contracts() both return 96 contracts."""
        c1 = generate_contracts()
        c2 = generate_contracts()
        assert len(c1) == len(c2) == 96

    def test_two_calls_produce_same_ids(self):
        """Two calls produce contracts with identical IDs."""
        c1 = generate_contracts()
        c2 = generate_contracts()
        ids1 = [c.contract_id for c in c1]
        ids2 = [c.contract_id for c in c2]
        assert ids1 == ids2

    def test_two_calls_produce_same_delivery_times(self):
        """Two calls produce contracts with identical delivery windows."""
        c1 = generate_contracts()
        c2 = generate_contracts()
        for a, b in zip(c1, c2):
            assert a.delivery_start_minutes == b.delivery_start_minutes
            assert a.delivery_end_minutes == b.delivery_end_minutes

    def test_two_calls_produce_same_gcts(self):
        """Two calls produce contracts with identical GCTs."""
        c1 = generate_contracts()
        c2 = generate_contracts()
        for a, b in zip(c1, c2):
            assert a.gct_minutes == b.gct_minutes
