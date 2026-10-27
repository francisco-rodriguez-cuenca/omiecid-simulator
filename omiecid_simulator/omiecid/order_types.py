"""OMIE Continuous Intraday order type definitions.

Domain values from docs/omiecid/market-spec.md §4, §4.5, §5.1, §13.

Provides:
- IntEnum classes for Side, ExecRestriction, Validity, Status
- OMIEOrder NamedTuple with 13 fields
- make_order() factory with OMIE-appropriate defaults
"""

from enum import IntEnum
from typing import NamedTuple


class Side(IntEnum):
    """Order side (market-spec §4.1)."""
    BUY = 0
    SELL = 1


class ExecRestriction(IntEnum):
    """Execution restriction (market-spec §4.5, §13).

    ICE=3 matches real OMIE data encoding where iceberg orders use
    the ICE execution restriction value rather than NON.
    """
    NON = 0
    IOC = 1
    FOK = 2
    ICE = 3


class Validity(IntEnum):
    """Order validity condition (market-spec §4.6).

    GFS: valid until contract's GCT (default).
    GTD: valid until agent-specified time <= GCT.
    """
    GFS = 0
    GTD = 1


class Status(IntEnum):
    """Order lifecycle state (market-spec §5.1)."""
    ACTIVE = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4
    EXPIRED = 5


class OMIEOrder(NamedTuple):
    """13-field OMIE order representation.

    Field order is fixed (positional construction relies on it).
    peak_qty and ppd carry the iceberg parameters (peak size and
    per-period price delta) used by iceberg logic and data parsing.
    """
    contract_id: int
    side: int          # Side IntEnum value
    price: float
    quantity: float
    order_id: int
    agent_id: int
    timestamp: int
    exec_restriction: int   # ExecRestriction IntEnum value
    validity: int           # Validity IntEnum value
    expiry_time: int        # 0 = not set / use contract GCT
    status: int             # Status IntEnum value
    peak_qty: float         # visible slice size for icebergs; 0.0 otherwise
    ppd: float              # per-slice price delta for ICEwPI; 0.0 otherwise


def make_order(
    contract_id: int,
    side: Side,
    price: float,
    quantity: float,
    order_id: int,
    agent_id: int,
    timestamp: int,
    exec_restriction: ExecRestriction = ExecRestriction.NON,
    validity: Validity = Validity.GFS,
    expiry_time: int = 0,
    status: Status = Status.ACTIVE,
    peak_qty: float = 0.0,
    ppd: float = 0.0,
) -> OMIEOrder:
    """Create an OMIE order with domain-appropriate defaults.

    Required fields: contract_id, side, price, quantity, order_id,
    agent_id, timestamp.

    Optional fields default to: exec_restriction=NON, validity=GFS,
    expiry_time=0 (use contract GCT), status=ACTIVE, peak_qty=0.0,
    ppd=0.0.
    """
    return OMIEOrder(
        contract_id=contract_id,
        side=side,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=exec_restriction,
        validity=validity,
        expiry_time=expiry_time,
        status=status,
        peak_qty=peak_qty,
        ppd=ppd,
    )
