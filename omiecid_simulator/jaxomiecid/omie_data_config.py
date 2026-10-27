"""OMIE Data Configuration.

Frozen dataclass holding all parameters needed to locate and cache
OMIE continuous intraday market data for a single delivery date and
contract.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OMIEDataConfig:
    """Immutable configuration for OMIE market data loading.

    Attributes:
        data_path: Root directory containing raw OMIE data files.
        contract_id: Quarter-hour contract index (0-95, where 0 = 00:00-00:15
            and 95 = 23:45-24:00).
        delivery_date: ISO-8601 date string (YYYY-MM-DD) for the delivery day
            D of the product. Product (D, k) is read from session files
            D-1 and D (market-spec §13.4).
        cache_path: Directory for serialized/cached data artifacts.
            Defaults to ``"saved_npz/omie/"``.
        allow_dst_transition: Opt in to loading a product whose delivery day
            is a Europe/Madrid DST transition day. Default ``False``: the
            loaders raise ``ValueError`` on such days (open-questions OQ-30).
            Under the opt-in, a date-D ``Contrato`` with an ``A`` / ``B``
            hour suffix still raises (no slot mapping is defined).

    Note:
        Per market-spec.md §13.4, the market area (``Zona`` / ``Zona compra``
        / ``Zona venta``) column is parsed from OMIE files but is not used
        for filtering. The loaded dataset spans the full Iberian participant
        universe (Spain + Portugal) for every contract.
    """

    data_path: str
    contract_id: int
    delivery_date: str
    cache_path: str = "saved_npz/omie/"
    allow_dst_transition: bool = False
