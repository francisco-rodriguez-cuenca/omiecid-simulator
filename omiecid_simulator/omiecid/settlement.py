"""JIT-safe settlement primitives.

Two public JAX-traceable primitives:

1. ``compute_trade_value(Q_mw, p_eur_mwh, d_h=0.25) -> f32``
   Scalar trade value formula V = Q x p x d, returning a 0-d float32
   JAX scalar. JIT-compilable.

2. ``compute_agent_pnl(trades: Int32[M, 6], agent_id: i32) -> f32``
   Tape-centric per-agent PnL over the offset-encoded int32 trade tape
   emitted by the loader. Uses an int64 two-term accumulator

       p_eur = p_tick_offset x price_tick + price_min

   so that bilateral zero-sum holds exactly in integer tick space before
   the float32 decode boundary.

   **Contract (scoped-x64, no public jit contract):**

     * ``compute_agent_pnl`` is JAX-traceable and supports ``jax.vmap``.
     * ``compute_agent_pnl`` **does not** carry a public ``jax.jit``
       contract. JAX issue jax-ml/jax#5982 makes int64 dtypes collapse to
       int32 inside ``jit`` when the global x64 flag is off; this is
       tracked in OQ-18.
     * Traced callers (``jax.vmap`` / any future traced composition) **must**
       supply an active ``with jax.experimental.enable_x64():`` context at
       the call site. Wrapping ``enable_x64`` inside a ``jit``-traced body
       is a no-op (the flag is read at trace time), which is why the jit
       contract is dropped rather than patched.
     * For **eager** (non-traced) calls the function body opens a scoped
       ``enable_x64`` context internally so the int64 accumulator works
       without the caller having to wrap every direct invocation. This is
       safe: in eager mode ``enable_x64()`` behaves as a true scoped
       context, touches no global state that outlives the call, and leaves
       the int32 order-validator jaxpr untouched.
     * The module deliberately does **not** flip the global
       ``jax.config.update("jax_enable_x64", True)`` flag — the
       order validator's jaxpr must remain strictly int32 (a global
       flag would change every traced program).

Domain: market-spec v2.4 §2.3, §8 (zero-sum bilateral transfer).

Trade tape column layout (M, 6):
    [timestamp, price (offset-encoded int32), quantity_tick,
     buyer_agent_id, seller_agent_id, contract_id]
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental import enable_x64 as _enable_x64

from omiecid_simulator.omiecid.omie_config import OMIEProductConfig

# Column indices into the (M, 6) int32 trade tape.
_COL_PRICE = 1
_COL_QTY_TICK = 2
_COL_BUYER = 3
_COL_SELLER = 4

_CFG = OMIEProductConfig()
_QTY_TICK = float(_CFG.qty_tick)
_PRICE_TICK = float(_CFG.price_tick)
_PRICE_MIN = float(_CFG.price_min)
_DURATION_HOURS = float(_CFG.delivery_duration_hours)

# Precomputed decode coefficients (Python floats, promoted to float64 inside
# the scoped x64 context at call time). Folded once here to avoid repeating
# the (qty_tick * duration_hours) product on every invocation.
_SCALE_QTY_DUR = _QTY_TICK * _DURATION_HOURS
_COEF_OFFSET = _PRICE_TICK * _SCALE_QTY_DUR  # EUR per (tick_offset * qty_tick_unit)
_COEF_MIN = _PRICE_MIN * _SCALE_QTY_DUR      # EUR per (qty_tick_unit)


def compute_trade_value(
    quantity_mw,
    price_eur_mwh,
    duration_hours: float = 0.25,
):
    """Scalar trade-value primitive: V = Q x p x d (EUR).

    Returns a 0-d ``jnp.float32`` array. JIT-compilable.
    """
    q = jnp.asarray(quantity_mw, dtype=jnp.float32)
    p = jnp.asarray(price_eur_mwh, dtype=jnp.float32)
    d = jnp.asarray(duration_hours, dtype=jnp.float32)
    return q * p * d


def compute_agent_pnl(trades, agent_id):
    """Per-agent PnL over an offset-encoded int32 trade tape.

    Implements the two-term int64 decomposition:

        pnl_ticks_offset = sum(sign * Q_ticks * p_tick_offset)   [int64]
        pnl_ticks_min    = sum(sign * Q_ticks)                   [int64]
        pnl_eur = (pnl_ticks_offset * price_tick * qty_tick * d_h)
                + (pnl_ticks_min    * price_min  * qty_tick * d_h)

    sign = +1 when ``agent_id`` is the seller leg, -1 when buyer leg, 0
    otherwise. Padding rows (quantity_tick == 0) contribute zero.

    Int64 is structurally required: per-row ``Q_tick x p_tick_offset`` at
    the OMIE envelope reaches ~2e10, overflowing int32 (~2.15e9).

    Call-site contract (see module docstring for the full scoped-x64
    rationale):

      - Eager calls: body opens its own scoped ``enable_x64`` context, so
        direct invocation works without caller ceremony.
      - ``jax.vmap`` calls: the caller **must** wrap the vmap invocation in
        ``with jax.experimental.enable_x64():``. An in-body context cannot
        promote dtypes under tracing.
      - ``jax.jit``: not supported (OQ-18; JAX issue jax-ml/jax#5982).

    Args:
        trades: Int32 array of shape ``(M, 6)`` — the trade tape.
        agent_id: int32 scalar — agent to settle.

    Returns:
        0-d ``jnp.float32`` PnL in EUR.
    """
    aid = agent_id

    buyer = trades[:, _COL_BUYER]
    seller = trades[:, _COL_SELLER]

    # Scoped x64 context — active on the eager path; a no-op under tracing.
    # Traced callers (jax.vmap) must supply their own outer enable_x64
    # context. The global x64 flag is intentionally NOT flipped
    # so the order-validator jaxpr stays strictly int32.
    with _enable_x64():
        price_offset = trades[:, _COL_PRICE].astype(jnp.int64)
        qty_tick = trades[:, _COL_QTY_TICK].astype(jnp.int64)

        # sign: +1 seller leg, -1 buyer leg, 0 otherwise. A row where
        # buyer == seller == agent_id collapses to zero (legal self-match).
        is_seller = (seller == aid).astype(jnp.int64)
        is_buyer = (buyer == aid).astype(jnp.int64)
        sign = is_seller - is_buyer  # int64, in {-1, 0, +1}

        signed_q = sign * qty_tick                           # int64
        pnl_ticks_offset = jnp.sum(signed_q * price_offset)  # int64
        pnl_ticks_min = jnp.sum(signed_q)                    # int64

        # Decode in float64 so the int64 tick-space accumulators are not
        # truncated on the way out (100e6-range tick products exceed
        # float32's 2^24 exact-integer ceiling). Float32 boundary is
        # applied only on the final EUR result, outside the scope.
        coef_offset = jnp.asarray(_COEF_OFFSET, dtype=jnp.float64)
        coef_min = jnp.asarray(_COEF_MIN, dtype=jnp.float64)
        pnl_eur_f64 = (
            pnl_ticks_offset.astype(jnp.float64) * coef_offset
            + pnl_ticks_min.astype(jnp.float64) * coef_min
        )

    return pnl_eur_f64.astype(jnp.float32)


__all__ = ["compute_trade_value", "compute_agent_pnl"]
