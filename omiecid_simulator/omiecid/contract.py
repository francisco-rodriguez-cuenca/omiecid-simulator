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
        gct_minutes: Gate Closure Time (delivery_start - 60). May be negative
            for early-morning contracts (GCT falls on previous calendar day).
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

    GCT (Gate Closure Time) is 60 minutes before delivery start (§3.1).
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
            gct_minutes=start - 60,
            status=ContractStatus.PENDING,
        ))
    return contracts
