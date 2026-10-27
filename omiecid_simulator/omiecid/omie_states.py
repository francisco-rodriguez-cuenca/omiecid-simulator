"""OMIE single-contract session state.

Defines the JAX-compatible ``@struct.dataclass`` state that the ``Session``
(``omiecid_simulator/omiecid/omie_session.py``) produces and consumes. It is a
pure state definition module — it contains no ``reset``/``step`` logic.

``OMIEContractState``
    The full JAX state for one OMIE contract session. It **composes** the
    ``OMIEBookState`` (from ``omiecid_simulator/omiecid/contract_book.py``) as a
    nested ``book`` field — it never re-shapes, re-wraps, or replaces it, so the
    ``iceberg_meta`` ``jnp.ndarray`` leaf (shape ``(MAX_ICEBERGS, 8)``, int32)
    crosses into the session state byte-for-byte unchanged. Around
    the composed book it adds the session-level fields ``time``,
    ``contract_status``, ``step_counter``, ``agent_inventory``, ``agent_cash``,
    ``tape_idx``, ``tape_cursor`` and ``agent_trades``.

Device policy: this module is device-agnostic — it defines pure pytree types and
carries no device-coupling primitive, inheriting whatever default platform the
importing harness pins.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct

from omiecid_simulator.omiecid.contract_book import OMIEBookState


# ---------------------------------------------------------------------------
# OMIEContractState — JAX state for one OMIE single-contract session.
# Composes the OMIEBookState (never re-shapes it) and adds the
# session-level fields.
# ---------------------------------------------------------------------------

@struct.dataclass
class OMIEContractState:
    """Full JAX state for one OMIE single-contract session.

    Composes (never re-shapes/replaces) the ``OMIEBookState`` as the
    ``book`` field.

    Fields:
        book: The composed ``OMIEBookState`` (the jaxob LOB + iceberg
            side-channel + counters). Carried unchanged — its ``iceberg_meta``
            ``jnp.ndarray`` leaf keeps its ``(MAX_ICEBERGS, 8)`` int32 shape.
        time: 0-d int — current simulated time (minutes from GOT).
        contract_status: 0-d int — OPEN=1, CLOSED=3.
        step_counter: 0-d int — Session step index within the replay.
        agent_inventory: ``(num_agents,)`` float32 — signed net MW per agent
            (=~0 at GCT by design).
        agent_cash: ``(num_agents,)`` float32 — realized cash (EUR) accumulator.
        tape_idx: 0-d int — which tape of the fixed set this replay uses.
        tape_cursor: 0-d int32 — the carried background-tape order offset the
            per-step tape-slice mechanism reads (each step replays the next
            ``orders_per_step`` tape rows). Initialised
            to 0 in ``reset`` and advanced by ``orders_per_step`` each
            ``step_env``. Additive and
            boundary-preserving — it does NOT touch the composed
            ``OMIEBookState`` (its ``iceberg_meta`` leaf still crosses
            byte-for-byte) and is a pure 0-d int leaf, so the
            dataclass stays ``@struct.dataclass``-registerable and
            JIT/``vmap``-passable with no adapter. Appended LAST with a default
            so 7-field keyword constructions stay valid.
    """

    book: OMIEBookState            # composed book state (not replaced)
    time: jnp.ndarray             # 0-d int — simulated time (minutes from GOT)
    contract_status: jnp.ndarray  # 0-d int — OPEN=1, CLOSED=3
    step_counter: jnp.ndarray     # 0-d int — Session step index within the replay
    agent_inventory: jnp.ndarray  # (num_agents,) float32 — signed net MW per agent
    agent_cash: jnp.ndarray       # (num_agents,) float32 — realized cash (EUR) accumulator
    tape_idx: jnp.ndarray         # 0-d int — replayed tape index within the set
    # Additive 0-d int32 background-tape order offset. Appended
    # LAST with a default so 7-field keyword constructions stay valid —
    # the leaf defaults wherever a constructor omits it.
    tape_cursor: jnp.ndarray = struct.field(default_factory=lambda: jnp.int32(0))
    # Additive (steps_per_episode, 2, 6) int32 AGENT
    # trade tape — the running record of the agent's OWN fills in the exact column
    # layout ``compute_agent_pnl`` consumes ([timestamp, price_offset_int,
    # qty_tick, buyer, seller, contract_id]; settlement.py:46-49). TWO (6,) rows
    # are written per step at ``step_counter``: slot 0 the AGGRESSIVE (own-submit)
    # leg, slot 1 the PASSIVE (resting-NON) leg, so a simultaneous aggressive +
    # passive fill records BOTH legs (a single slot would lose one). NOOP /
    # no-fill steps leave both slots all-zero (inert, ``qty_tick == 0``), which
    # contribute nothing to ``compute_agent_pnl``. Consumers reshape ``(-1, 6)``
    # and run the EAGER ``compute_agent_pnl`` over it (``with
    # jax.experimental.enable_x64():`` — NEVER inside the vmapped jit, OQ-18).
    # Appended LAST with a default (a single inert row) so fixed-arg
    # keyword constructions stay valid; additive and boundary-preserving — it
    # does NOT touch the composed ``OMIEBookState``.
    agent_trades: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros((1, 2, 6), dtype=jnp.int32)
    )


__all__ = [
    "OMIEContractState",
]
