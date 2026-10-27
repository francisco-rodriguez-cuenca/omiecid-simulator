"""OMIE Continuous Intraday market configuration dataclasses.

Domain values from docs/omiecid/market-spec.md §2.1–2.3.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OMIEProductConfig:
    """Configuration for a single OMIE 15-minute energy product.

    All defaults reflect OMIE/SIDC market rules for the Iberian zone.
    Price and quantity constraints come from market-spec §2.3.
    """
    mtu_minutes: int = 15                    # Market Time Unit (minutes)
    price_min: float = -9999.0               # EUR/MWh — hard SIDC floor
    price_max: float = 9999.0                # EUR/MWh — hard SIDC ceiling
    price_tick: float = 0.01                 # EUR/MWh — minimum increment
    qty_min: float = 0.1                     # MW — minimum order quantity
    qty_max: float = 999.0                   # MW — maximum order quantity
    qty_tick: float = 0.1                    # MW — minimum quantity increment
    delivery_duration_hours: float = 0.25    # hours — for trade value formula V = Q × p × d


@dataclass(frozen=True)
class OMIEMarketConfig:
    """Simulator-level configuration for an OMIE continuous intraday session.

    Controls how the simulator is sized: number of order books,
    agents, and book capacity.
    """
    num_contracts: int = 96     # quarter-hourly contracts per delivery day
    num_agents: int = 2         # agent identities (configurable)
    book_depth: int = 100       # max resting orders per side per contract
    nTrades: int = 100          # trades array capacity for jaxob (per matching call)
    max_match_iterations: int = 200  # truncation cap; inert while > `_TRIP_BOUND` (100), which is the effective outer pass bound
    # Config-driven iceberg side-channel capacity — the
    # first dim of the (max_icebergs, 8) `iceberg_meta` array. Default 1024
    # gives >=2x headroom over the observed 257-512 short-tape peak live
    # n_active. `regenerate_slices` cost is linear in this dim, so shrinking
    # it from the MAX_ICEBERGS=8192 hard ceiling to the runtime cap is the
    # perf lever. Frozen+int keeps hashability → `static_argnames`-safe at
    # every jit lift, so the meta shape stays a compile-time constant per
    # config instance (one-compile / vmap invariant preserved). Bounded by
    # MAX_ICEBERGS (the hard structural ceiling) at ContractBook.__init__.
    max_icebergs: int = 1024
