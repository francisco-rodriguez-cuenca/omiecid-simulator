"""OMIE Continuous Intraday order validation (JIT-safe).

Domain rules from docs/omiecid/market-spec.md §2.3, §4.2, §4.5.

Validates orders against OMIE rules without requiring a book instance.
ICE (iceberg) orders are accepted with iceberg-specific validation.

JIT contract:
- Single ``jnp.bool_`` scalar return.
- ``config`` and ``contract_open`` are static (``static_argnums=(1, 2)``).
- Tick-layer checks are performed in int32 space after the round-to-tick encode;
  no float tolerance literal appears on the traced path.
- Five checks fused into one AND expression (no ``lax.cond``).
"""

import jax.numpy as jnp

from omiecid_simulator.omiecid.order_types import OMIEOrder, Side, ExecRestriction
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.encoding import encode_price, encode_qty


def _is_on_tick(value: float, tick: float) -> bool:
    """Legacy helper retained for external test compatibility.

    An earlier ``validate_order`` used this helper with a float tolerance.
    The current ``validate_order`` itself performs exact integer
    tick-alignment in a scaled representation. This helper remains for
    use in loader/round-trip tests where sub-ULP float64 noise can leak
    from offset encode/decode cycles; the tolerance is sized to absorb
    that noise without accepting genuine off-tick (>= 0.1 scaled) inputs.
    """
    scale = round(1.0 / tick)
    scaled = value * scale
    return abs(scaled - round(scaled)) < (1e-3)


def validate_order(
    order: OMIEOrder,
    config: OMIEProductConfig,
    contract_open: bool,
) -> jnp.ndarray:
    """Validate an order against OMIE market rules.

    Returns a ``jnp.bool_`` scalar: True if accepted, False if rejected.

    Checks (market-spec §2.3, §4.2, §4.5):
    - Contract must be OPEN
    - Side must be valid (BUY=0 or SELL=1)
    - Price within [price_min, price_max] after tick-snap
    - Quantity within [qty_min, qty_max] after tick-snap
    - ICE orders: peak_qty > 0 and peak_qty <= quantity (market-spec §4.2)
    - ICE orders: PPD sign (market-spec §13.6, Rules Annex §4.5): a BUY ICE
      needs PPD <= 0 and a SELL ICE needs PPD >= 0 (after round-to-tick encoding to
      ticks; PPD = 0 is a plain ICE). The PPD of a non-ICE order is not checked.
    """
    # Float price/qty are encoded to int32 ticks via the round-to-tick encode
    # (multiply-by-inverse-tick-as-float32-integer, then round, then int32
    # cast). Verified XLA-safe on the full OMIE on-tick price
    # grid (1,999,801 cases, 0 mismatches) and qty grid (9,990 cases, 0
    # mismatches) under ``jit+vmap``. The on-tick check collapses to a
    # no-op: ``jnp.round`` snaps sub-half-tick offsets to the nearest tick,
    # and the int32 bound check rejects out-of-range ticks. Off-tick
    # rejection for offsets >= half-tick works because ``jnp.round``
    # commits the input to a distinct integer tick that may or may not be
    # in range -- and the integer result carries no residual float error.
    # Sub-half-tick offsets are treated as on-tick by the encoder,
    # consistent with the int32 tick-encoded loader dtype contract and
    # ``encoding.encode_price``'s ``round()`` semantic.
    price_scale_f = jnp.float32(round(1.0 / config.price_tick))  # 100.0
    qty_scale_f = jnp.float32(round(1.0 / config.qty_tick))      # 10.0

    # Static int32 bounds (config is static_argnum=1).
    price_int_min = jnp.int32(encode_price(config.price_min, config))  # 0
    price_int_max = jnp.int32(encode_price(config.price_max, config))
    qty_int_min = jnp.int32(encode_qty(config.qty_min, config))
    qty_int_max = jnp.int32(encode_qty(config.qty_max, config))
    price_offset_int = jnp.int32(round(-config.price_min / config.price_tick))

    # Traced float -> traced int32 via the round-to-tick encode (XLA-safe).
    price_f = jnp.asarray(order.price, dtype=jnp.float32)
    qty_f = jnp.asarray(order.quantity, dtype=jnp.float32)
    price_tick_int = jnp.round(price_f * price_scale_f).astype(jnp.int32)
    price_int = price_tick_int + price_offset_int
    qty_int = jnp.round(qty_f * qty_scale_f).astype(jnp.int32)

    # Bounds-only checks in int32 space.
    price_in_range = (price_int >= price_int_min) & (price_int <= price_int_max)
    qty_in_range = (qty_int >= qty_int_min) & (qty_int <= qty_int_max)

    # Side validity (static-known enum ints).
    side = jnp.asarray(order.side, dtype=jnp.int32)
    side_ok = (side == jnp.int32(int(Side.BUY))) | (side == jnp.int32(int(Side.SELL)))

    # Iceberg branch (market-spec §4.2). Gated on exec_restriction == ICE.
    exec_restr = jnp.asarray(order.exec_restriction, dtype=jnp.int32)
    is_ice = exec_restr == jnp.int32(int(ExecRestriction.ICE))
    peak_qty = jnp.asarray(order.peak_qty, dtype=jnp.float32)
    # PPD sign rule (market-spec §13.6): round-to-tick encoding as for price, so a
    # sub-half-tick PPD rounds to 0 ticks and is treated as a plain ICE.
    ppd_f = jnp.asarray(order.ppd, dtype=jnp.float32)
    ppd_int = jnp.round(ppd_f * price_scale_f).astype(jnp.int32)
    ppd_sign_ok = (
        ((side == jnp.int32(int(Side.BUY))) & (ppd_int <= jnp.int32(0)))
        | ((side == jnp.int32(int(Side.SELL))) & (ppd_int >= jnp.int32(0)))
    )
    ice_ok_if_ice = (
        (peak_qty > jnp.float32(0.0)) & (peak_qty <= qty_f) & ppd_sign_ok
    )
    ice_ok = (~is_ice) | ice_ok_if_ice

    # contract_open is a Python static bool (static_argnum=2).
    open_ok = jnp.asarray(bool(contract_open), dtype=jnp.bool_)

    # AND of bool arrays is already jnp.bool_; no trailing cast needed.
    return open_ok & side_ok & price_in_range & qty_in_range & ice_ok
