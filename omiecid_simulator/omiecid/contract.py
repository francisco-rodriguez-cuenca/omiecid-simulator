"""OMIE Continuous Intraday contract lifecycle definitions.

Domain values from docs/omiecid/market-spec.md §2.1–2.2, §3.1.

Provides:
- ContractStatus IntEnum (PENDING/OPEN/SUSPENDED/CLOSED)
- Contract dataclass with contract_id, delivery times, GCT, and status
"""

from dataclasses import dataclass
from enum import IntEnum


class ContractStatus(IntEnum):
    """Contract lifecycle status.

    PENDING: before GOT — contract exists but not yet tradable.
    OPEN: accepting orders, matching active.
    SUSPENDED: IDA clearing window — no submission or matching (§3.3).
    CLOSED: past GCT — no further trading.
    """
    PENDING = 0
    OPEN = 1
    SUSPENDED = 2
    CLOSED = 3


@dataclass
class Contract:
    """A single OMIE quarter-hourly delivery contract.

    Each contract represents one 15-minute delivery period (1 MTU).
    Times are expressed in minutes from midnight CET of the delivery day.

    Attributes:
        contract_id: 0–95 identifying the quarter-hour slot.
        delivery_start_minutes: Start of delivery period (minutes from midnight).
        delivery_end_minutes: End of delivery period (delivery_start + 15).
        gct_minutes: Gate Closure Time, aligned to the delivery hour:
            60 * (contract_id // 4) - 60. The four quarter-hours of one hour
            close together, one hour before that hour starts (market-spec.md
            §3.1; market-model.md "Trading window and gate closure"). Negative
            for the first hour's contracts (GCT falls on the previous calendar
            day).
        status: Current lifecycle status.
    """
    contract_id: int
    delivery_start_minutes: int
    delivery_end_minutes: int
    gct_minutes: int
    status: ContractStatus


def generate_contracts() -> list["Contract"]:
    """Generate all 96 quarter-hourly contracts for a delivery day.

    Each contract covers one 15-minute MTU (Market Time Unit):
      - Contract 0:  00:00–00:15
      - Contract 95: 23:45–24:00

    GCT (Gate Closure Time) is aligned to the delivery hour: the four
    quarter-hours of hour h (contracts 4h..4h+3) all close one hour before
    that hour starts, GCT(k) = 60 * (k // 4) - 60 minutes from midnight of
    the delivery day, i.e. 60, 75, 90 and 105 min before their own delivery
    starts; negative (the previous day) for contracts 0-3
    (market-spec.md §3.1, §11.1; market-model.md "Trading window and gate
    closure").
    All contracts start in PENDING status.

    Returns:
        List of 96 Contract instances ordered by contract_id (0–95).
    """
    contracts = []
    for cid in range(96):
        start = cid * 15
        contracts.append(Contract(
            contract_id=cid,
            delivery_start_minutes=start,
            delivery_end_minutes=start + 15,
            gct_minutes=60 * (cid // 4) - 60,
            status=ContractStatus.PENDING,
        ))
    return contracts
