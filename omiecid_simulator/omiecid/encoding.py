"""Stateless OMIE <-> jaxob field conversion functions.

Pure functions at module scope with no state dependencies.
Used by ContractBook and trade_extraction to convert
between OMIE domain values and jaxob internal int32 representation.

Encoding scheme:
- Price: offset encoding so negative EUR/MWh maps to positive int32.
- Quantity: scale by inverse of qty_tick (0.1 MW -> 1 unit).
- Side: BUY=0 -> +1, SELL=1 -> -1 (jaxob convention).
- Timestamp: (seconds, 0) tuple for jaxob lexicographic ordering.

Domain rules: docs/omiecid/market-spec.md.
"""

from omiecid_simulator.omiecid.omie_config import OMIEProductConfig


def encode_price(price_eur: float, config: OMIEProductConfig) -> int:
    """Offset encoding: int_price = round((price_eur - price_min) / price_tick).

    With defaults: round((price_eur - (-9999.0)) / 0.01)
    Maps [-9999.0, +9999.0] -> [0, 1_999_800].
    Always positive; safe for jaxob's -1 sentinel.
    """
    return round((price_eur - config.price_min) / config.price_tick)


def decode_price(jaxob_int: int, config: OMIEProductConfig) -> float:
    """Inverse: price_eur = jaxob_int * price_tick + price_min."""
    return jaxob_int * config.price_tick + config.price_min


def encode_qty(qty_mw: float, config: OMIEProductConfig) -> int:
    """Scale: int_qty = round(qty_mw / qty_tick).

    With defaults: round(qty_mw / 0.1) = round(qty_mw * 10).
    Maps [0.1, 999.0] -> [1, 9990].
    """
    return round(qty_mw / config.qty_tick)


def decode_qty(jaxob_int: int, config: OMIEProductConfig) -> float:
    """Inverse: qty_mw = jaxob_int * qty_tick."""
    return jaxob_int * config.qty_tick


def encode_ppd(ppd_eur: float, config: OMIEProductConfig) -> int:
    """Signed int-tick encoding for Per-Period Delta (ICEwPI).

    NOT offset-encoded. Sign is preserved so downstream arithmetic
    (``base_price + ppd * slice_idx``) remains exact in tick units.

    Formula (pinned; exact in tick units):
        encode_ppd(ppd_eur) = int32(round(ppd_eur / price_tick))
    """
    return round(ppd_eur / config.price_tick)


def decode_ppd(ppd_int: int, config: OMIEProductConfig) -> float:
    """Inverse of :func:`encode_ppd`: ppd_eur = ppd_int * price_tick."""
    return ppd_int * config.price_tick


def encode_side(side: int) -> int:
    """BUY=0 -> +1; SELL=1 -> -1. Formula: 1 - 2 * side."""
    return 1 - 2 * side


def encode_side_lookup(side: int) -> int:
    """OMIE side -> jaxob get_order lookup side.

    BUY=0 -> 1 (bids), SELL=1 -> 0 (asks).
    Formula: 1 - side.

    jaxob's get_order selects bids when side==1, asks otherwise.
    """
    return 1 - side


def encode_timestamp(ts: int) -> tuple:
    """Return (ts, 0). Nanosecond field is zero.

    jaxob uses (seconds, nanoseconds) lexicographic ordering.
    Since OMIE timestamps are monotonic integers, mapping to
    (ts, 0) preserves temporal ordering.
    """
    return (ts, 0)
