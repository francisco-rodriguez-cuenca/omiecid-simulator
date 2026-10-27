"""Trade-diff and VWAP accumulation for OMIE order book matching.

JIT-safe: the function body is fully JAX-traceable
— no `int(...)` materialisation, no Python `for`-loop, no dynamic-shape
indexing. Fills are detected by a whole-row diff of the trades buffer
(ring-buffer-immune; also catches IOC fills whose residual is cancelled).

Returns raw jaxob-int quantities and prices so that callers can perform
domain conversion (or accumulate across iterations) without rounding
divergence.

Dtype split contract:
``(total_qty_int, weighted_price_int, num_fills) -> (int64, int64, int32)``.
The int64 fields are required because, on a saturated buffer, the per-row
``price_int * qty_int`` product reaches ~2e10 (price ~2e6 x qty ~1e4),
exceeding int32 (~2.15e9). Aggregation across NTRADE_CAP rows
compounds this.

Because the rest of the project runs with ``jax_enable_x64=False`` (the
order validator's jaxpr must stay strictly int32), this module **does not
flip the global flag**. Instead, both the body of ``extract_fills`` and the construction
of ``NO_FILLS`` are wrapped in a scoped
``with jax.experimental.enable_x64():`` context, which is honored both
in eager and in jit-traced execution and leaves the global config
untouched. Same scoped-x64 pattern used by ``omie.settlement``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import enable_x64 as _enable_x64

from omiecid_simulator.jaxob import jaxob_constants as cst


class RawFills(NamedTuple):
    """Raw fill results in jaxob internal int units.

    All three fields are 0-d ``jnp.ndarray`` scalars. Dtype split:
    ``(int64, int64, int32)`` for ``(total_qty_int, weighted_price_int,
    num_fills)``.

    This NamedTuple is a JAX pytree; ``jax.tree.leaves(fills)`` yields
    exactly the three scalar leaves.

    Attributes:
        total_qty_int: Sum of absolute fill quantities (jaxob int units).
        weighted_price_int: Sum of (price_i * qty_i) across all fills
            (jaxob int units). Divide by ``total_qty_int`` to get VWAP.
        num_fills: Number of new trade rows detected.
    """
    total_qty_int: jnp.ndarray
    weighted_price_int: jnp.ndarray
    num_fills: jnp.ndarray


def _build_no_fills() -> RawFills:
    """Construct the zero sentinel under a scoped x64 context so the
    int64 fields are not silently truncated to int32.
    """
    with _enable_x64():
        return RawFills(
            total_qty_int=jnp.zeros((), dtype=jnp.int64),
            weighted_price_int=jnp.zeros((), dtype=jnp.int64),
            num_fills=jnp.zeros((), dtype=jnp.int32),
        )


#: Sentinel value for "no fills occurred" — JAX scalars matching the
#: dtype split of `extract_fills`.
NO_FILLS = _build_no_fills()


def extract_fills(
    trades_before: jnp.ndarray,
    trades_after: jnp.ndarray,
) -> RawFills:
    """Diff two trades arrays to detect new fills (JIT-safe).

    Detects new trades via whole-row inequality between ``trades_before``
    and ``trades_after``. This detector is ring-buffer-immune: once the
    NTRADE_CAP-slot jaxob trades buffer saturates, new fills overwrite
    existing slots (PASS_OID stays != -1), so a sentinel-transition mask
    `(before_oids == -1) & (after_oids != -1)` would fail to fire. A row
    changes iff a new fill was written to that slot, so whole-row
    inequality is the correct detector.

    Trade quantity is the absolute value of the Q column (which is signed
    in jaxob: negative for incoming-buy matches, positive for
    incoming-sell).

    The aggregation is a single vectorised pass over all NTRADE_CAP rows:
    ``jnp.sum(jnp.where(new_mask, ..., 0))``. Aggregates are cast to int64
    via ``jax.lax.convert_element_type`` inside a scoped
    ``enable_x64`` context so the sums hold extreme price*qty products
    without overflow. The global x64 flag is intentionally NOT flipped
    (keeps the order validator's jaxpr strictly int32).

    See `docs/omiecid/market-spec.md` §4.5 (IOC cancels residual, never
    rests): an IOC fill leaves no resting order, so it is detected from
    the trades buffer, not from the book.

    Args:
        trades_before: Trades array snapshot taken *before* dispatch.
        trades_after: Trades array snapshot taken *after* dispatch.

    Returns:
        A ``RawFills`` of three 0-d JAX scalars with dtype split
        (int64, int64, int32). If no new trades occurred, returns the
        zero-valued equivalent of ``NO_FILLS``.
    """
    P = cst.TradesFeat.P.value
    Q = cst.TradesFeat.Q.value

    with _enable_x64():
        # Whole-row diff detector (ring-buffer-immune). DO NOT revert to the
        # sentinel-transition mask — it fails to fire under saturation.
        new_mask = jnp.any(trades_before != trades_after, axis=1)

        # Per-row signed→abs qty and raw price (int32 from the trades
        # buffer); promote to int64 for accumulation so price*qty sums
        # do not overflow under saturation with extreme values.
        raw_qtys = jnp.abs(trades_after[:, Q])
        raw_prices = trades_after[:, P]
        qtys64 = jax.lax.convert_element_type(raw_qtys, jnp.int64)
        prices64 = jax.lax.convert_element_type(raw_prices, jnp.int64)

        # Vectorised aggregation over all rows; non-fill rows contribute 0.
        # Multiply by the int64-cast mask rather than `jnp.where(..., 0)`
        # — under default x64-off, the literal `0` would be canonicalised to
        # int32 by the outer jit and break MLIR type-checking on the int64
        # branch.
        mask64 = jax.lax.convert_element_type(new_mask, jnp.int64)
        total_qty_int = jnp.sum(qtys64 * mask64)
        weighted_price_int = jnp.sum(prices64 * qtys64 * mask64)
        num_fills = jax.lax.convert_element_type(
            jnp.sum(new_mask), jnp.int32
        )

    return RawFills(
        total_qty_int=total_qty_int,
        weighted_price_int=weighted_price_int,
        num_fills=num_fills,
    )
