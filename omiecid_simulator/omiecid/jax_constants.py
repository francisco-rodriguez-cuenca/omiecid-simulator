"""Compile-time JAX shape constants and domain conventions for the OMIE domain.

These constants parameterise the fixed-size state arrays of the static-shape
JAX engine. They must be concrete Python ``int``
values (not ``jnp.int32`` / ``jnp.ndarray``) so that ``jax.jit`` treats them
as compile-time static arguments — any traced/array form would leak into
``static_argnums`` and break downstream consumers (``iceberg_manager.py``
for MAX_ICEBERGS/MAX_SLICES_PER_REGEN and ``settlement.py`` /
``contract_book.py`` for MAX_AGENTS).

Public API (frozen; the test suite pins these names and values):

    from omiecid_simulator.omiecid.jax_constants import (
        MAX_ICEBERGS,
        MAX_SLICES_PER_REGEN,
        MAX_AGENTS,
        SENTINEL_ORDER_ID,
        SENTINEL_AGENT_ID,
        make_sentinel_order,
    )

-----------------------------------------------------------------------------
MAX_ICEBERGS
-----------------------------------------------------------------------------
Chosen value: 8192.

Steady-state population analysis:

The size of the iceberg side-channel ``OMIEBookState.iceberg_meta`` is
NOT bounded by the resting-book depth. An earlier derivation that tied
``MAX_ICEBERGS`` to ``OMIEMarketConfig.book_depth`` was incorrect: it
conflated the LOB-resident visible-peak population (which IS bounded by
``book_depth`` per side) with the iceberg-meta population (which is the
**outstanding-with-hidden** population — every iceberg with
``remaining_hidden > 0`` consumes one slot, regardless of whether its
visible peak is currently in the LOB).

Per-session, the ``iceberg_meta`` population at time t equals:

    population(t) = registers_cumulative(t)
                  - cleanups_cumulative(t)
                  - full_drains_cumulative(t)

where ``registers_cumulative`` is the count of ICE submissions accepted
so far, ``cleanups_cumulative`` is the count of ICE cancels, and
``full_drains_cumulative`` is the count of icebergs whose
``remaining_hidden`` reached zero via the regen lifecycle (slot freed
by ``regenerate_slices``' cleanup branch). The LOB-resident visible-peak
count is a strict subset of this population — at any moment, the
visible-peak count is bounded by ``min(active_iceberg_count,
book_depth_per_side)``, but the side-channel meta must size for the
**outstanding-with-hidden** count, which is unbounded by ``book_depth``.

The cap therefore must be sized against the session-max **ICE-with-hidden**
peak, derived from the canonical session ICE registration / cancel /
full-drain trace — not against ``book_depth``. Empirically (audit of
2026-01-12 cid=81), realistic sessions exceeded an interim cap
of 100, and the silent-eviction surface (a 101st ``register`` overwriting
slot 0 without raising) is what motivated the overflow handling below.

Cap history:
    The cap was first set at an interim **100**, then raised from 100 to 8192 once
    the fully jitted engine could be benchmarked against the larger
    ``iceberg_meta`` array (the re-derived cap materially expands the size
    of every traced ``iceberg_meta`` array, so compile time and step time
    had to be measured). The cap is now **8192** (see ``MAX_ICEBERGS``
    below). The interim caveat — ``OMIEBookState.truncated`` being
    sticky-True on most realistic sessions because of the 100-slot
    ceiling — no longer holds. ``truncated`` should now only assert on
    genuinely pathological sessions that exceed the configured
    outstanding-with-hidden capacity, which is an error condition rather
    than a routine caveat.

Slot-overflow handling:
    ``IcebergManager.register`` returns ``(updated_meta, overflow_bool)``;
    when the meta is saturated at registration time, ``overflow_bool``
    is True and the meta is returned unchanged (no slot is silently
    overwritten). The overflow flag propagates through
    ``ContractBook._build_state`` onto the sticky
    ``OMIEBookState.truncated`` field — the same field used by the
    slice-overflow carry-flag backstop. ``ContractBook.__init__`` also
    enforces ``book_depth <= MAX_ICEBERGS`` so that runtime book-depth
    raises cannot outpace the side-channel size.

Cost of revising: changes the row-count of
``OMIEBookState.iceberg_meta: jnp.ndarray(MAX_ICEBERGS, 8)``.
Every ``.at[slot].set(...)`` and ``lax.dynamic_slice`` path in
``iceberg_manager.py`` is shape-affected. No cache invalidation is required
(the v4 NPZ cache stores loader output, which is ``(N, 13)``-shaped and
independent of ``MAX_ICEBERGS``). The ``ContractBook.__init__``
assertion ``book_depth <= MAX_ICEBERGS`` couples the runtime book depth
to the side-channel ceiling.

-----------------------------------------------------------------------------
MAX_SLICES_PER_REGEN
-----------------------------------------------------------------------------
Chosen value: 64.

Derivation: analytical bound for the ICEwPI ``while remaining > 0`` loop
(the slice projection in ``matchable_hidden_volume`` and ``regenerate_slices``
in ``omiecid_simulator/omiecid/iceberg_manager.py``). An
iceberg's slice count is bounded by ``ceil(total_quantity / peak_qty)``.
With ``qty_max = 999.0 MW`` and ``qty_min = 0.1 MW`` the theoretical max is
9990, which is impractical as a ``jax.lax.while_loop`` max-trip-count.
Empirically the per-regenerate call only generates slices until the book
matches or the peak is unfilled, and for realistic ``peak_qty >= 0.5 MW``
values the iteration count stays well below 64. 64 is chosen as a safe,
tractable ceiling that preserves iceberg semantics (``>= 2`` at minimum)
while keeping the compiled ``while_loop`` trip-count small enough for XLA.

HAZARD — truncation-silent bound: if a session ever produces an iceberg
with ``remaining_hidden / peak_qty > MAX_SLICES_PER_REGEN``, the
``jax.lax.while_loop`` terminates early and the remaining hidden volume is
silently dropped from matching. An uncapped Python ``while remaining > 0``
loop would not have this hazard. The engine therefore carries an explicit
flag that detects it: ``IcebergManager.truncation_flag`` ORs into the
sticky ``OMIEBookState.truncated`` field.

Cost of revising: changes the ``jax.lax.while_loop`` max-trip-count for
ICEwPI slice generation in ``iceberg_manager.py``. No cache invalidation.
No downstream shape change beyond the loop-bound itself.

-----------------------------------------------------------------------------
MAX_AGENTS
-----------------------------------------------------------------------------
Chosen value: 512.

Derivation: empirical, from three representative sessions in
``data/rawOMIE/`` loaded via ``load_omie_agent_metadata`` on the canonical
``contract_id=81``. Seasonal spread (summer / autumn / winter) guards
against a seasonal bias in the distinct ``(Agente, Unidad)`` pair count:

    * 2025-06-01 cid=81 -> 158 distinct pairs (summer)
    * 2025-10-15 cid=81 -> 217 distinct pairs (autumn shoulder; max)
    * 2026-01-12 cid=81 -> 185 distinct pairs (winter, regression date)

Observed cross-session max = 217. Safety margin = 2x observed, rounded up
to the next power of 2 -> 512.

Re-measured after the switch to per-product tapes (2026-09-30; a tape is
now ONE delivery product gathered from session files D-1 and D, so the pair
universe changed):

    * 2025-06-01 cid=81 -> 146 distinct pairs (was 158)
    * 2025-06-02 cid=81 -> 201 distinct pairs (was 229)
    * 2025-10-15 cid=81 -> 237 distinct pairs (was 217; new max)
    * 2026-01-12 cid=81 -> 185 distinct pairs (was 186)

New cross-session max = 237; 2 x 237 = 474 <= 512, so MAX_AGENTS is
unchanged. The 2x margin covers growth from new
entrants over a year; the power-of-2 rounding helps XLA array layout.
Per-row cost in any ``(MAX_AGENTS, K)`` settlement array is measured in
hundreds of bytes, so doubling is effectively free.

Cost of revising: changes the per-agent pad dimension used by
``compute_agent_pnl`` and the optional per-agent query arrays in
``contract_book.get_agent_orders`` / ``get_agent_trades`` (see the
query-method JIT-scope decision below). No cache invalidation required — the v4 NPZ
cache stores loader output sized ``(N, 13)`` / ``(M, 6)``, which is
independent of ``MAX_AGENTS``. Re-derivation is a human-driven decision
triggered by (a) significantly newer data, (b) an observed session whose
pair universe approaches the chosen bound, or (c) the safety margin being
judged insufficient in a later review.

-----------------------------------------------------------------------------
SENTINEL_ORDER_ID / SENTINEL_AGENT_ID
-----------------------------------------------------------------------------
Chosen values:

    SENTINEL_ORDER_ID = -1
    SENTINEL_AGENT_ID = -1

Both are Python ``int`` (hashable; usable as ``static_argnums`` values or
dict keys). ``bool`` is disallowed — although ``bool`` is a subclass of
``int``, downstream ``static_argnums`` consumers disambiguate on identity,
so sentinels must be pure ``int``.

Derivation: the loader emits ``(Agente, Unidad)`` pair indices
as ``int32`` ``>= 0`` and order IDs as ``int32`` ``>= 1`` (positive-ID
enforcement in ``ContractBook.submit_order`` rejects ``<= 0``). Therefore
``-1`` cannot alias a real agent or a real order and is a safe sentinel
for both "not found" paths.

HAZARD — single-discriminator invariant: ``OMIEOrder`` has several fields
whose zero-value is a legal real state:

    * ``side == 0`` maps to ``Side.BUY`` (not "no side").
    * ``status == 0`` maps to ``Status.ACTIVE`` (not "no status").
    * ``peak_qty == 0.0`` means "not an iceberg" (a legal real state).
    * ``ppd == 0.0`` means "no slice delta" (a legal real state).

The ONLY reliable discriminator is ``order_id == SENTINEL_ORDER_ID``.
Consumers must not branch on any other field to detect the sentinel.
``make_sentinel_order()`` (below) therefore sets ``order_id = -1`` and
zero-fills the rest — the zero fields are deliberately ambiguous with
real state, and the discriminator lives entirely in ``order_id``.

Callsite adoption: this module provides the sentinel SURFACE (this constant +
the factory below). Callsites (e.g. ``contract_book.get_order``) return
``make_sentinel_order()`` instead of ``None``.

-----------------------------------------------------------------------------
JIT-scope of query methods
-----------------------------------------------------------------------------
**Decision:** ``get_anonymous_lob``, ``get_agent_orders``, and
``get_agent_trades`` remain Python-side helpers that operate on
``OMIEBookState``. Rationale: (a) all three return variable-length Python
sequences built from masked fixed-size JAX arrays; converting to
fixed-size returns would force every consumer to carry per-row validity
masks, pushing filtering complexity into the caller without removing it.
(b) None of the three methods are on the per-match hot path — they are
diagnostic / inspection utilities invoked per-step at most,
not per-match. (c) The underlying JAX primitives (``job.get_L2_state``,
``job.get_agent_trades``, ``lob.asks``/``lob.bids`` masking) are already
JIT-compatible; the engine only requires that
``submit_order`` JITs end-to-end, not that the query helpers do. (d) If
a JIT-traceable agent view is later required, a new method
``get_agent_orders_array(state, agent_id) ->
jnp.ndarray`` can be added as a JIT-safe sibling without disturbing the
Python-side query API. The three methods are therefore
untouched except for the ``get_order`` sentinel return (see above).

Cross-reference: ``contract.py`` status transitions rely on this
decision to remain Python-side, because transitions are only
consulted from Python-side wrappers (see the next decision).

-----------------------------------------------------------------------------
Contract status transitions stay Python-side
-----------------------------------------------------------------------------
**Decision ("no change required"):** the ``ContractStatus``
IntEnum, the ``Contract`` dataclass, and the ``generate_contracts()``
factory in ``omiecid_simulator/omiecid/contract.py`` remain Python-side. Contract
status transitions are NOT JIT-traced — there is no JIT conversion of
``contract.py``. This is the load-bearing verdict: `contract.py`
remains Python-side, status transitions are not traced.

Rationale: the contract
lifecycle is static within a session (one ``Contract`` instance per
``contract_id``, status fixed at construction time). The only in-tree
reader of ``Contract.status`` inside a JIT'd function is
the ``submit_order`` guard in ``contract_book.py`` (which tests
``self._contract.status`` against ``ContractStatus.OPEN``). Converting the IntEnum
to a traced array would buy nothing because that guard only needs a
boolean decision; the decision can be taken Python-side and passed in as a
closed-over scalar.

**Caller-contract invariant (load-bearing):**
The jitted ``submit_order`` receives the status guard across the JIT
boundary as a ``contract_open: bool`` closed-over / pre-computed scalar
argument — NOT as a live read of ``self._contract.status`` inside traced
code. ``self._contract.status`` / any ``.status`` comparison inside a
jitted function body is forbidden; the sanctioned pattern is: compute
``contract_open = bool(self._contract.status is ContractStatus.OPEN)`` (or the equivalent ``==`` comparison)
Python-side, then pass ``contract_open`` into the traced inner function.
This keeps the decision coherent with the jitted ``submit_order`` without
re-opening the JIT scope of ``contract.py``.

**Cross-references:**
  * The "JIT-scope of query methods" decision (above) — established the
    Python-side-helper pattern for ``get_anonymous_lob`` /
    ``get_agent_orders`` / ``get_agent_trades`` that this decision extends to
    ``contract.py``.
  * ``docs/omiecid/market-spec.md`` §2.1–2.2 — the four-state contract
    lifecycle (OPEN / SUSPENDED / CLOSED / SETTLED) whose semantics in
    this engine are static-per-session and therefore do not require JIT
    tracing. Sections 2.1 and 2.2 are the authoritative source for the
    lifecycle rules this decision locks in.

**Trigger for reopening:** this decision must
be revisited if and only if per-step auto-transitions
of contract status driven by simulation time are
introduced (for example auto-SUSPEND at the IDA session boundary, or
auto-CLOSE at GCT). At that point status becomes a per-step dynamic
quantity rather than a session-static one, and the closed-over-scalar
adapter pattern no longer suffices; the decision must be re-examined,
not silently patched.

**DAG annotation (import-graph constraint):** JIT-scope modules (those
wrapped by ``jax.jit`` / ``jax.vmap``, notably the jitted
``submit_order`` in ``contract_book.py``) MUST NOT import ``Contract`` or
``ContractStatus`` into a traced code path. The only sanctioned workaround
is the adapter / closed-over-scalar pattern described above: resolve the
status comparison Python-side, pass the resulting ``contract_open`` bool
(or an equivalent pre-computed scalar) across the JIT boundary. Plain
``from omiecid_simulator.omiecid.contract import Contract, ContractStatus``
inside a module whose functions are jitted — without going through the
adapter — violates this decision and must be caught at review time.

-----------------------------------------------------------------------------
Module placement
-----------------------------------------------------------------------------
This module lives under ``omiecid_simulator/omiecid/`` (the pure-domain layer),
NOT under ``omiecid_simulator/jaxomiecid/`` (the I/O-boundary layer). The
constants parameterise domain state-array shapes consumed by
``iceberg_manager``, ``contract_book``, and ``settlement`` — all of which
sit under ``omiecid/``. Placing this file under ``jaxomiecid/`` would invert
the existing ``jaxomiecid/ -> omiecid/`` dependency direction.

The sentinel constants and ``make_sentinel_order()`` live here alongside
the shape constants — no sibling ``jax_conventions.py`` module is introduced.
"""

from __future__ import annotations

from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    Status,
    Validity,
)

# See module docstring for derivation and revision cost.
# This constant is the HARD STRUCTURAL CEILING for the
# iceberg side-channel — the maximum first-dim the packed-layout / regen loops
# are sized against and the upper bound `ContractBook.__init__` asserts on
# `market_config.max_icebergs`. The RUNTIME per-book iceberg capacity is the
# config-driven `OMIEMarketConfig.max_icebergs` (default 1024); the meta array
# is built at that runtime cap, not this ceiling. The numeric value STAYS 8192
# (its role is the "hard ceiling"); the test suite
# asserts this module constant is still 8192.
MAX_ICEBERGS: int = 8192

# See module docstring for derivation, revision cost, and the
# truncation-silent hazard (caught by the truncation-flag backstop).
MAX_SLICES_PER_REGEN: int = 64

# See module docstring for derivation and revision cost.
MAX_AGENTS: int = 512

# "Order not found" sentinel. Python int (hashable; usable as
# static_argnums / dict key). THE discriminator in make_sentinel_order() —
# see module docstring hazard note on zero-ambiguous fields (side, status).
SENTINEL_ORDER_ID: int = -1

# Sibling convention to SENTINEL_ORDER_ID. The loader emits
# (Agente, Unidad) pair indices >= 0, so -1 does not alias a real agent.
SENTINEL_AGENT_ID: int = -1


def make_sentinel_order() -> OMIEOrder:
    """Construct the canonical "order not found" sentinel OMIEOrder.

    Contract (pinned by the test suite):
      * ``order_id == SENTINEL_ORDER_ID`` — this is the SOLE discriminator.
      * Every field is a concrete (non-None) JAX-compatible scalar; each
        survives ``jnp.asarray(...)`` as a 0-d array.
      * Deterministic / idempotent: two successive calls return equal
        ``OMIEOrder`` instances (no ``time.time()``-style defaults).

    Usage: jitted downstream consumers close
    over this sentinel as their "not found" return value instead of
    ``None`` — avoiding Optional-typed branches that would make the tracer
    see a non-JAX type.

    Zero-field ambiguity: ``side``, ``status``, ``peak_qty``, ``ppd`` are
    all zero here, which ALIAS legal real states (side=0 -> BUY, status=0
    -> ACTIVE, peak_qty=0 -> not an iceberg, ppd=0 -> no slice delta).
    Consumers MUST test ``order.order_id == SENTINEL_ORDER_ID`` rather
    than branching on any of these fields — see the module docstring
    (SENTINEL_ORDER_ID section).

    Returns:
        An ``OMIEOrder`` with ``order_id = SENTINEL_ORDER_ID``,
        ``agent_id = SENTINEL_AGENT_ID``, and every remaining field a
        concrete zero of the appropriate Python scalar type.
    """
    return OMIEOrder(
        contract_id=0,
        side=int(Side.BUY),
        price=0.0,
        quantity=0.0,
        order_id=SENTINEL_ORDER_ID,
        agent_id=SENTINEL_AGENT_ID,
        timestamp=0,
        exec_restriction=int(ExecRestriction.NON),
        validity=int(Validity.GFS),
        expiry_time=0,
        status=int(Status.ACTIVE),
        peak_qty=0.0,
        ppd=0.0,
    )


__all__ = [
    "MAX_ICEBERGS",
    "MAX_SLICES_PER_REGEN",
    "MAX_AGENTS",
    "SENTINEL_ORDER_ID",
    "SENTINEL_AGENT_ID",
    "make_sentinel_order",
]
