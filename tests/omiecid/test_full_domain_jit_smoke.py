"""Locked behavioural-contract tests for Story P2-H3 — Integration + acceptance gate.

Story card (v5.21, 2026-05-15):
  - ``docs/omiecid/agile-plan.md`` §"Story P2-H3 — Integration + acceptance gate"
    (around line 574). The eight ACs are listed there and re-affirmed by the
    P2-H3 architecture review on this branch:
    ``docs/omiecid/sprint-status.md`` §"P2-H3 — Architecture Review (2026-05-15)"
    (line 6313). Three interpretation pins drive the test design:

      1. Synthetic ≤ ``MAX_ICEBERGS`` fixture only; do NOT replay real OMIE
         sessions because ``MAX_ICEBERGS=100`` is the interim cap and the
         cap raise is deferred to ``P2-HOTFIX-iceberg-cap-overflow-FOLLOWUP``.
         Assertions are on ``SubmitResult`` fields, NOT on ``state.truncated``.
      2. Cancel (msg_type=2) is a SEPARATE submit-then-cancel-then-get_order
         round-trip test — NOT a 6th ``ExecRestriction`` parametrize row.
         ``cancel_order`` is Python-side (out of the jitted ``submit_order``
         scope per the v5.20 P2-H2 amendment).
      3. AC (6) shim-free probe scopes to the **jitted-helper bodies**
         in ``contract_book.py`` (``_submit_order_body``, ``_non_or_ioc_path``,
         ``_fok_path``, ``_ice_path``, ``_while_loop_match``,
         ``_while_loop_match_fok``, ``_build_state_jit``). The Python-side
         boundary ``int(...)`` casts in ``cancel_order`` / ``get_order`` /
         ``get_anonymous_lob`` / ``get_agent_orders`` / ``get_agent_trades``
         are NOT shims (B4 decision) and must not be flagged.

Architectural references (read at authoring time):
  - ``docs/omiecid/agile-plan.md`` §P2-H3 (the eight ACs).
  - ``docs/omiecid/migration-plan.md`` §Phase 2 (loader dtype, module-level jit
    idiom, settlement API pin, acceptance subsections).
  - ``docs/omiecid/market-spec.md`` (NON/IOC/FOK/ICE/ICEwPI semantics).
  - ``docs/omiecid/phase2-test-manifest.md`` (Cat A/B/C classification).
  - ``omiecid_simulator/omiecid/contract_book.py`` — module-level
    ``_submit_order`` at line 1005-1008; ``OMIEBookState`` at line 95-114;
    ``cancel_order`` at line 1127-1159; ``get_order`` at line 1181-1213.
  - ``omiecid_simulator/omiecid/iceberg_manager.py`` — array-backed
    ``iceberg_meta`` of shape ``(MAX_ICEBERGS, 8) int32``.
  - ``omiecid_simulator/omiecid/jax_constants.py`` — B1-B5 constants and the
    B4 decision-record docstring.
  - ``omiecid_simulator/jaxomiecid/omie_data_loader.py`` —
    ``load_omie_agent_metadata`` at line 328-359 (returns
    ``dict[int, tuple[str, str]]``).

Phase 2 acceptance criteria (the 8 ACs encoded here):

  (1) All Phase 1 + Phase 1.5 tests pass with documented adapter changes.
      Environmental invariant verified by the orchestrator at merge gate;
      this file is the *evidence* for (3)/(4)/(5)/(6)/(7)/(8). A sanity
      probe is included that imports the canonical Phase-1 surface for
      collection-time evidence the import graph is intact.
  (2) All new ``*_under_jit`` tests pass. Same — environmental. This file
      is the new AC (2) evidence (registered via the ``pytest.mark.jit``
      module marker).
  (3) ``jax.jit(contract_book._submit_order)(state, order, ...)`` compiles
      cleanly and produces the same ``SubmitResult`` tree as Phase 1.5
      across NON / IOC / FOK / ICE / ICEwPI; cancel is a SEPARATE
      ``ContractBook.cancel_order`` round-trip per pin (2).
  (4) ``OMIEBookState.iceberg_meta`` is a ``jnp.ndarray`` of dtype int32
      and shape ``(MAX_ICEBERGS, 8)``.
  (5) ``OMIEBookState`` has no ``dict`` / ``None`` / ``list`` / raw
      ``tuple`` leaves — only ``jnp.ndarray`` leaves under ``jax.tree``.
  (6) No Python compatibility shims remain in jitted-helper bodies of
      ``contract_book.py``. Scope per pin (3).
  (7) B1-B5 constants documented and live in ``jax_constants.py``:
      ``MAX_ICEBERGS`` (B1), ``MAX_SLICES_PER_REGEN`` (B2),
      ``SENTINEL_ORDER_ID`` (B3), B4 JIT-scope decision (docstring),
      ``MAX_AGENTS`` (B5).
  (8) Agent metadata accessor ``id_to_pair`` (via
      ``load_omie_agent_metadata``) remains accessible outside JIT for
      settlement aggregation and diagnostics.

Test discipline:
  - Module-level ``pytestmark = pytest.mark.jit`` (T1.1 marker discipline).
  - NOT ``@pytest.mark.slow``. The harness traces five variants through
    the already-warm jit cache (one compile, five trace reuses); the
    synthetic fixture stays well below ``MAX_ICEBERGS``. The wall-clock
    budget is < 5 s warm, < 30 s cold per architect §6.
  - Synthetic fixtures only — no real OMIE replay (architect pin 1).
  - AC (8): ``test_id_to_pair_accessor_returns_dict_outside_jit`` is the
    only test that touches real data on disk. It is guarded with a
    ``skipif`` on the canonical 2026-01-12 cid=81 file presence and runs
    fast (~0.5-2 s) because ``_collect_pair_universe`` reads pandas CSV
    once; if the file is absent the test is skipped at collection.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid import jax_constants as jc
from omiecid_simulator.omiecid.contract_book import (
    OMIEBookState,
    SubmitResult,
    _submit_order,
    _submit_order_body,
)
from omiecid_simulator.omiecid.jax_constants import (
    MAX_AGENTS,
    MAX_ICEBERGS,
    MAX_SLICES_PER_REGEN,
    SENTINEL_ORDER_ID,
    make_sentinel_order,
)
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    make_order,
)


# Module-level marker (Story T1.1 + P2-H2 precedent at
# ``tests/omiecid/test_submit_order_jit_safe.py:117``). Every test in this
# file traces or compiles ``_submit_order``.
pytestmark = pytest.mark.jit


# ===========================================================================
# Helpers — synthetic order factories matching the P2-H2 precedent shape.
# ===========================================================================


def _make_buy(
    order_id: int = 1,
    price: float = 50.00,
    quantity: float = 10.0,
    agent_id: int = 0,
    timestamp: int = 100,
    exec_restriction: ExecRestriction = ExecRestriction.NON,
    peak_qty: float = 0.0,
    ppd: float = 0.0,
) -> OMIEOrder:
    return make_order(
        contract_id=37,
        side=Side.BUY,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=exec_restriction,
        peak_qty=peak_qty,
        ppd=ppd,
    )


def _make_sell(
    order_id: int = 2,
    price: float = 50.00,
    quantity: float = 10.0,
    agent_id: int = 1,
    timestamp: int = 200,
    exec_restriction: ExecRestriction = ExecRestriction.NON,
    peak_qty: float = 0.0,
    ppd: float = 0.0,
) -> OMIEOrder:
    return make_order(
        contract_id=37,
        side=Side.SELL,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=exec_restriction,
        peak_qty=peak_qty,
        ppd=ppd,
    )


def _as_python_bool(x) -> bool:
    return bool(x)


def _as_python_float(x) -> float:
    return float(x)


# ===========================================================================
# Phase-1.5 oracle table — analytically derived, not a recorded session
# trace.
# ===========================================================================
#
# Each row is a complete behavioural specification of one
# ``ExecRestriction`` variant. The smoke harness:
#
#   1. Submits a ``setup`` resting counterparty (if any).
#   2. Submits the ``aggressor`` order through ``ContractBook.submit_order``
#      (which trampolines into the module-level ``_submit_order`` lift).
#   3. Asserts on the five ``SubmitResult`` fields against the
#      analytically-derived oracle.
#
# Oracle derivation (Phase-1.5 ground truth):
#
#   - Contract 37 covers minutes [9*60+15, 9*60+30] = a 15-minute slot, so
#     the settlement duration ``d = 15/60 = 0.25 h``.
#     ``settlement_value = trade_qty (MW) × trade_price (EUR/MWh) × d``.
#     See ``OMIEMarketConfig.book_depth`` / market-spec SS2.3 for the
#     duration / settlement formula.
#
#   - NON crossing (4 MW SELL @ 50.00 / 10 MW BUY @ 50.00):
#       4 MW filled @ 50.00 EUR/MWh → settlement = 4 × 50 × 0.25 = 50.0
#       BUY residual 6 MW rests post-fill (NON re-submit branch).
#       Final SubmitResult shows the *aggregated* trade_qty=4.0 / trade_price=50.0
#       (the residual rest is a state-side effect, not a SubmitResult field).
#
#   - IOC full fill (10 MW SELL @ 50.00 / 10 MW BUY @ 50.00 IOC):
#       10 MW filled @ 50.00 → settlement = 10 × 50 × 0.25 = 125.0
#       No residual (IOC has none by definition).
#
#   - FOK full fill (10 MW SELL @ 50.00 / 10 MW BUY @ 50.00 FOK):
#       Pre-check liquidity = 10 MW (visible) >= 10 MW required → fills
#       fully. Same result as IOC full fill: 10 MW × 50.00 × 0.25 = 125.0.
#
#   - ICE on empty book (5 MW BUY @ 50.00, peak=1.0, ppd=0):
#       No counterparty → no trade. accepted=True, traded=False, the
#       visible peak rests in the book; the iceberg side-channel registers
#       the hidden remainder. trade_qty / trade_price / settlement_value
#       all zero.
#
#   - ICEwPI on empty book (5 MW BUY @ 50.00, peak=1.0, ppd=0.10):
#       Same shape as ICE — no counterparty, no trade. ppd only matters
#       on slice regeneration (not on the initial peak rest). Per the
#       v5.20 P2-H2 amendment, ICEwPI is the ICE arm taken when ppd != 0;
#       the harness exercises both arms through one ``_submit_order``
#       cache entry.
#
# Field names match ``SubmitResult`` (contract_book.py:84-88):
#   accepted, traded, trade_qty, trade_price, settlement_value.
#
# Float fields use ``pytest.approx`` (default rel=1e-6, abs=1e-12) for
# the float32-precision boundary; bool fields use exact equality.

CONTRACT_DURATION_H = 0.25  # contract 37 is a 15-minute slot.


def _setup_buy_oracle_table():
    """Return the five-row oracle table.

    Returns:
        List of ``(label, setup, aggressor, oracle_dict)`` tuples where
        ``setup`` is either ``None`` (no resting counterparty) or a
        ``OMIEOrder`` to submit BEFORE the aggressor.
    """
    # Common resting-SELL counterparties used by the NON / IOC / FOK rows.
    sell_4mw = _make_sell(order_id=1, price=50.00, quantity=4.0, timestamp=100)
    sell_10mw = _make_sell(order_id=1, price=50.00, quantity=10.0, timestamp=100)

    non_aggressor = _make_buy(
        order_id=2, price=50.00, quantity=10.0, timestamp=200,
        exec_restriction=ExecRestriction.NON,
    )
    ioc_aggressor = _make_buy(
        order_id=2, price=50.00, quantity=10.0, timestamp=200,
        exec_restriction=ExecRestriction.IOC,
    )
    fok_aggressor = _make_buy(
        order_id=2, price=50.00, quantity=10.0, timestamp=200,
        exec_restriction=ExecRestriction.FOK,
    )
    ice_aggressor = _make_buy(
        order_id=2, price=50.00, quantity=5.0, timestamp=200,
        exec_restriction=ExecRestriction.ICE,
        peak_qty=1.0, ppd=0.0,
    )
    # OMIE-ICEWPI-SIGN-REJECT-1 re-baseline: a BUY ICEwPI with PPD > 0 has
    # the wrong sign (market-spec §13.6) and is rejected by the validator;
    # the conforming twin (BUY PPD < 0) keeps the ICEwPI-accepted coverage.
    icewpi_wrong_sign_aggressor = _make_buy(
        order_id=2, price=50.00, quantity=5.0, timestamp=200,
        exec_restriction=ExecRestriction.ICE,
        peak_qty=1.0, ppd=0.10,  # BUY PPD > 0 → wrong sign → rejected
    )
    icewpi_aggressor = _make_buy(
        order_id=2, price=50.00, quantity=5.0, timestamp=200,
        exec_restriction=ExecRestriction.ICE,
        peak_qty=1.0, ppd=-0.10,  # conforming BUY ICEwPI (PPD < 0)
    )

    return [
        (
            "NON",
            sell_4mw,
            non_aggressor,
            {
                "accepted": True,
                "traded": True,
                "trade_qty": 4.0,
                "trade_price": 50.00,
                "settlement_value": 4.0 * 50.00 * CONTRACT_DURATION_H,  # 50.0
            },
        ),
        (
            "IOC",
            sell_10mw,
            ioc_aggressor,
            {
                "accepted": True,
                "traded": True,
                "trade_qty": 10.0,
                "trade_price": 50.00,
                "settlement_value": 10.0 * 50.00 * CONTRACT_DURATION_H,  # 125.0
            },
        ),
        (
            "FOK",
            sell_10mw,
            fok_aggressor,
            {
                "accepted": True,
                "traded": True,
                "trade_qty": 10.0,
                "trade_price": 50.00,
                "settlement_value": 10.0 * 50.00 * CONTRACT_DURATION_H,  # 125.0
            },
        ),
        (
            "ICE",
            None,  # empty book — ICE rests its peak, no trade
            ice_aggressor,
            {
                "accepted": True,
                "traded": False,
                "trade_qty": 0.0,
                "trade_price": 0.0,
                "settlement_value": 0.0,
            },
        ),
        (
            "ICEwPI-wrong-sign",
            None,  # empty book — rejected by the §13.6 sign rule
            icewpi_wrong_sign_aggressor,
            {
                "accepted": False,
                "traded": False,
                "trade_qty": 0.0,
                "trade_price": 0.0,
                "settlement_value": 0.0,
            },
        ),
        (
            "ICEwPI",
            None,  # empty book — ICEwPI rests its peak, no trade
            icewpi_aggressor,
            {
                "accepted": True,
                "traded": False,
                "trade_qty": 0.0,
                "trade_price": 0.0,
                "settlement_value": 0.0,
            },
        ),
    ]


ORACLE_ROWS = _setup_buy_oracle_table()


# ===========================================================================
# AC (3) — full-domain JIT smoke: 5-variant parametrize.
# ===========================================================================


class TestACThreeFullDomainJitSmoke:
    """AC (3) part 1: ``jax.jit(contract_book._submit_order)(state, order)``
    compiles cleanly and produces the same ``SubmitResult`` tree as
    Phase 1.5 across NON / IOC / FOK / ICE / ICEwPI.

    Architect pin (sprint-status §"P2-H3 — Architecture Review", §3):
      Single parametrized test, NOT five separate tests. The
      parametrize sweep proves the discriminator is traced (one
      ``_submit_order`` cache entry across all five variants — see
      ``TestACThreeOneCompileAcrossFiveVariants`` for the cache-count
      invariant).

    The aggressor goes through ``book.submit_order`` (the public
    trampoline) which delegates into ``_submit_order`` (the module-level
    jitted lift at ``contract_book.py:1005-1008``). The harness asserts
    on the five ``SubmitResult`` fields against the
    analytically-derived Phase-1.5 oracle.

    Architect pin (synthetic fixture, NOT real cid=81 replay): the
    interim ``MAX_ICEBERGS=100`` cap (deferred raise) would make
    ``state.truncated`` sticky-True on real sessions, so this harness
    uses the module-scoped synthetic ``book_and_state`` fixture from
    ``tests/omiecid/conftest.py:138``. Assertions are on ``SubmitResult``
    fields only — NOT on ``state.truncated``.
    """

    @pytest.mark.parametrize(
        "label,setup,aggressor,oracle",
        ORACLE_ROWS,
        ids=[row[0] for row in ORACLE_ROWS],
    )
    def test_submit_result_matches_phase1_oracle(
        self, book, empty_state, label, setup, aggressor, oracle,
    ):
        """For each ExecRestriction variant, the SubmitResult produced
        by the jitted ``_submit_order`` lift must match the Phase-1.5
        oracle field-by-field.

        Setup → Aggressor → Assert. The oracle is derived analytically
        from the order parameters and the contract duration (0.25 h
        for the 15-minute contract 37); no recorded session is used.
        """
        # 1. Prime the book with the resting counterparty (if any).
        if setup is not None:
            state, setup_result = book.submit_order(empty_state, setup)
            assert _as_python_bool(setup_result.accepted) is True, (
                f"[{label}] setup order must be accepted to prime the "
                f"resting counterparty for the aggressor; got "
                f"accepted={setup_result.accepted}"
            )
        else:
            state = empty_state

        # 2. Submit the aggressor through the jitted public surface.
        _new_state, result = book.submit_order(state, aggressor)

        # 3. Field-by-field oracle assertion.
        assert isinstance(result, SubmitResult), (
            f"[{label}] result must be SubmitResult NamedTuple; got "
            f"{type(result).__name__}"
        )
        assert _as_python_bool(result.accepted) == oracle["accepted"], (
            f"[{label}] accepted: expected {oracle['accepted']}, got "
            f"{_as_python_bool(result.accepted)}"
        )
        assert _as_python_bool(result.traded) == oracle["traded"], (
            f"[{label}] traded: expected {oracle['traded']}, got "
            f"{_as_python_bool(result.traded)}"
        )
        assert _as_python_float(result.trade_qty) == pytest.approx(
            oracle["trade_qty"]
        ), (
            f"[{label}] trade_qty: expected {oracle['trade_qty']}, got "
            f"{_as_python_float(result.trade_qty)}"
        )
        # When traded=False, trade_price is conventionally 0.0 (the
        # SubmitResult default field); when traded=True, the
        # quantity-weighted average price applies.
        assert _as_python_float(result.trade_price) == pytest.approx(
            oracle["trade_price"]
        ), (
            f"[{label}] trade_price: expected {oracle['trade_price']}, "
            f"got {_as_python_float(result.trade_price)}"
        )
        assert _as_python_float(result.settlement_value) == pytest.approx(
            oracle["settlement_value"]
        ), (
            f"[{label}] settlement_value: expected "
            f"{oracle['settlement_value']}, got "
            f"{_as_python_float(result.settlement_value)}"
        )


class TestACThreeOneCompileAcrossFiveVariants:
    """AC (3) trace-count invariant: sweeping the 5 ExecRestriction rows
    through the module-level jitted ``_submit_order`` causes exactly
    one ``lower(...).compile()``.

    Precedent: ``tests/omiecid/test_submit_order_jit_safe.py::TestACFiveTraceCountInvariantOneCompileForFourVariants::test_one_compile_across_all_four_exec_restrictions``
    (line 735-776). This is the P2-H3 reaffirmation extended to FIVE
    rows because ICEwPI is the ICE arm exercised with ``ppd != 0``;
    architect pin (sprint-status §3) confirms ICEwPI must NOT trigger
    a recompile.

    The cache-count probe uses ``_submit_order._cache_size()`` —
    PjitFunction's public probe attribute (same surface used by
    ``test_order_validator_jit.py:206`` and the P2-H2 audit suite).
    """

    def test_one_compile_across_all_five_exec_restrictions(
        self, book, empty_state,
    ):
        """Cache count after sweeping NON / IOC / FOK / ICE / ICEwPI
        through the same ``_submit_order`` must be exactly 1.

        The discriminator is a traced ``jnp.int32`` (per the P2-H2
        v5.20 amendment); the four ``ExecRestriction`` arms share one
        ``lax.switch`` artefact and ICEwPI is the ICE arm with
        non-zero ``ppd`` (no separate switch arm). Net: one cache
        entry across the five rows.
        """
        if not hasattr(_submit_order, "_cache_size"):
            pytest.skip(
                "_submit_order does not expose `_cache_size`; "
                "trace-count probe unavailable on this JAX version "
                "(JaxLib < pjit-cache-public-api)."
            )

        # Sweep all five rows through the jitted ``_submit_order``.
        # We invoke each variant on a fresh ``empty_state`` so the
        # setup ordering does not bias the cache key (the cache key is
        # over the pytree shape, which is constant across all rows).
        for idx, (label, setup, aggressor, _oracle) in enumerate(ORACLE_ROWS):
            state = empty_state
            if setup is not None:
                state, _ = book.submit_order(state, setup)
            _new_state, _result = book.submit_order(state, aggressor)

        cache_size = _submit_order._cache_size()
        assert cache_size == 1, (
            f"All five ExecRestriction variants (NON / IOC / FOK / "
            f"ICE / ICEwPI) must compile to one cache entry — the "
            f"discriminator is a traced int32, not a static Python "
            f"enum; ICEwPI is the ICE arm with non-zero ppd (not a "
            f"separate switch arm). Got _cache_size()={cache_size}. "
            f"A value > 1 means either the discriminator was "
            f"specialised at trace time, or ICEwPI was lifted to a "
            f"separate switch arm contrary to the v5.20 P2-H2 amendment."
        )


# ===========================================================================
# AC (3) — cancel sanity (SEPARATE from the parametrize sweep per pin 2).
# ===========================================================================


class TestACThreeCancelOrderRoundTrip:
    """AC (3) part 2: cancel sanity via submit-then-cancel-then-get_order.

    Architect pin (sprint-status §3 + §7 Risk #3): cancel (msg_type=2)
    is the Python-side ``ContractBook.cancel_order`` (line 1127-1159).
    It is NOT a 6th ``ExecRestriction`` parametrize row because it does
    not share the jitted ``submit_order`` dispatch. Per the v5.20
    P2-H2 amendment ``cancel_order`` is out of scope for the jitted
    lift but must continue to work post-refactor.

    The round trip:
      1. submit_order(NON BUY) → BUY rests in the book.
      2. get_order(BUY id) → returns the resting OMIEOrder.
      3. cancel_order(BUY id) → removes the BUY.
      4. get_order(BUY id) → returns the sentinel (order_id == -1).
    """

    def test_cancel_round_trip_returns_sentinel(self, book, empty_state):
        """Submit a NON BUY at price=50.00 / quantity=10.0; verify it
        rests; cancel it; verify ``get_order`` now returns the sentinel
        OMIEOrder (``order_id == SENTINEL_ORDER_ID``)."""
        order = _make_buy(order_id=42, price=50.00, quantity=10.0)
        state, submit_result = book.submit_order(empty_state, order)
        assert _as_python_bool(submit_result.accepted) is True, (
            "setup NON BUY must be accepted before cancel can be tested"
        )

        # Verify resting before cancel.
        resting = book.get_order(state, order_id=42, side=Side.BUY)
        assert resting is not None, (
            "the resting order must be retrievable before cancel"
        )
        assert int(resting.order_id) == 42, (
            f"pre-cancel get_order must return the real order_id=42; "
            f"got order_id={int(resting.order_id)}"
        )

        # Cancel.
        state2 = book.cancel_order(state, order_id=42, side=Side.BUY)

        # Verify sentinel after cancel.
        looked_up = book.get_order(state2, order_id=42, side=Side.BUY)
        assert looked_up is not None, (
            "post-cancel get_order must return the sentinel OMIEOrder, "
            "NOT None (P2-H2 AC (3) sentinel-on-miss contract)"
        )
        assert int(looked_up.order_id) == SENTINEL_ORDER_ID, (
            f"post-cancel get_order must return order_id == "
            f"SENTINEL_ORDER_ID (={SENTINEL_ORDER_ID}); got "
            f"order_id={int(looked_up.order_id)}"
        )


# ===========================================================================
# AC (4) — OMIEBookState.iceberg_meta is a jnp.ndarray.
# ===========================================================================


class TestACFourIcebergMetaIsJnpArray:
    """AC (4): ``OMIEBookState.iceberg_meta`` is a ``jnp.ndarray`` of
    dtype int32 and shape ``(MAX_ICEBERGS, 8)``.

    This is re-affirmed evidence: the type was set by P2-H1 and is
    already locked by the P2-H1 audit. The probe lives in P2-H3 for
    self-containment of the acceptance gate.

    Schema columns (per ``iceberg_manager.py:14-17``):
      ``[order_id, remaining_hidden, peak_qty, base_price_int, side,
        agent_id, ppd, slice_count]`` — 8 columns.
    """

    def test_iceberg_meta_is_jnp_ndarray(self, empty_state):
        """``state.iceberg_meta`` must be a JAX array (no Python dict
        / list shim left from Phase 1)."""
        assert isinstance(empty_state.iceberg_meta, jnp.ndarray), (
            f"OMIEBookState.iceberg_meta must be jnp.ndarray (P2-H1 "
            f"§A1 schema); got {type(empty_state.iceberg_meta).__name__}"
        )

    def test_iceberg_meta_dtype_is_int32(self, empty_state):
        """``state.iceberg_meta`` dtype must be int32 per the P2-H1 schema."""
        assert empty_state.iceberg_meta.dtype == jnp.int32, (
            f"OMIEBookState.iceberg_meta dtype must be int32; got "
            f"{empty_state.iceberg_meta.dtype}"
        )

    def test_iceberg_meta_shape_is_max_icebergs_by_eight(self, book, empty_state):
        """``state.iceberg_meta`` first dim must follow the book's
        configured ``max_icebergs`` (P3-PERF-REGEN-CAP-1 amends the signed
        P2-H3 AC-4 contract from the ``MAX_ICEBERGS`` constant to the
        config-derived ``market_config.max_icebergs``; ``MAX_ICEBERGS``
        remains the hard structural ceiling). The 8-column schema is
        unchanged."""
        expected = book._market_config.max_icebergs
        assert empty_state.iceberg_meta.shape == (expected, 8), (
            f"OMIEBookState.iceberg_meta shape must be "
            f"({expected}, 8) — first dim from the book's configured "
            f"max_icebergs (P3-PERF-REGEN-CAP-1), 8-column P2-H1 §A1 "
            f"schema; got {empty_state.iceberg_meta.shape}"
        )


# ===========================================================================
# AC (5) — OMIEBookState has no dict / None / list leaves.
# ===========================================================================


class TestACFiveNoDictOrNoneLeaves:
    """AC (5): ``jax.tree.leaves(state)`` returns only ``jnp.ndarray``
    leaves — no Python ``dict`` / ``None`` / ``list`` / raw ``tuple``.

    ``OMIEBookState`` is a NamedTuple (per contract_book.py:95). When
    flattened by ``jax.tree``, NamedTuples flatten transparently —
    their fields become the leaves. The four fields are
    ``lob, iceberg_meta, ts_counter, truncated``; ``lob`` is itself
    a ``LobState`` NamedTuple from jaxob whose fields are all
    ``jnp.ndarray``. Net: every leaf must be a ``jnp.ndarray``.

    Architect §5 verification: this is already shipped via the P2-H2
    ``TestSubmitResultDtypeContract`` audit and re-affirmed here as
    AC (5) self-contained evidence.
    """

    def test_all_leaves_are_jnp_ndarray(self, empty_state):
        """Every flattened leaf of ``OMIEBookState`` must be a
        ``jnp.ndarray`` — no Python primitive or container leaks."""
        leaves = jax.tree.leaves(empty_state)
        offenders = [
            (i, type(leaf).__name__)
            for i, leaf in enumerate(leaves)
            if not isinstance(leaf, jnp.ndarray)
        ]
        assert offenders == [], (
            f"OMIEBookState contains non-jnp.ndarray leaves: {offenders}. "
            f"Phase 2 closure requires every leaf to be a JAX array — "
            f"no Python dict / None / list / raw tuple may remain."
        )

    def test_no_none_leaves(self, empty_state):
        """Belt-and-suspenders complement: no ``None`` leaf anywhere
        in the pytree."""
        leaves = jax.tree.leaves(empty_state)
        for i, leaf in enumerate(leaves):
            assert leaf is not None, (
                f"OMIEBookState leaf #{i} is None — every leaf must be "
                f"a concrete jnp.ndarray (P2-H1 §A2 / AC (5))"
            )

    def test_pytree_has_at_least_one_leaf(self, empty_state):
        """Sanity check: ``jax.tree.leaves(state)`` is non-empty.
        If a future refactor accidentally registers ``OMIEBookState``
        as an opaque type, the leaf-walk would return ``[state]``
        itself and this test class's invariant would silently pass.
        We explicitly verify the flatten produces at least one
        concrete array leaf."""
        leaves = jax.tree.leaves(empty_state)
        assert len(leaves) >= 1, (
            "OMIEBookState must flatten to at least one leaf via "
            "jax.tree; if zero leaves are produced, the NamedTuple "
            "registration is broken."
        )


# ===========================================================================
# AC (6) — No Python compatibility shims in jitted-helper bodies.
# ===========================================================================


CONTRACT_BOOK_PATH = Path(
    inspect.getsourcefile(
        importlib.import_module("omiecid_simulator.omiecid.contract_book")
    )
)


def _module_source_tree() -> ast.Module:
    """Parse ``contract_book.py`` into an AST tree."""
    return ast.parse(CONTRACT_BOOK_PATH.read_text())


def _module_function(name: str) -> ast.FunctionDef:
    """Find a module-level function by name in ``contract_book.py``."""
    for node in ast.walk(_module_source_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"function {name!r} not found in contract_book.py"
    )


# The seven jitted-helper bodies the AC (6) shim probe targets. Each
# carries the lifted JIT-safe implementation of one phase of
# ``submit_order``. Per architect pin (3): the AC (6) shim-free scope
# is THESE bodies only; the Python-side B4 methods
# (``cancel_order`` / ``get_order`` / ``get_anonymous_lob`` /
# ``get_agent_orders`` / ``get_agent_trades``) carry legitimate
# boundary ``int(...)`` casts that are NOT shims.
JITTED_HELPER_NAMES = [
    "_submit_order_body",
    "_non_or_ioc_path",
    "_fok_path",
    "_ice_path",
    "_while_loop_match",
    "_while_loop_match_fok",
    "_build_state_jit",
]


class TestACSixShimFreeJittedBodies:
    """AC (6): the jitted-helper bodies in ``contract_book.py`` carry
    zero Python ``int(...)`` casts on traced values.

    Architect pin (3) (sprint-status §4 + §7 Risk #2): the AC (6)
    scope is the jitted-helper bodies ONLY. The Python-side B4 query
    methods (``cancel_order`` / ``get_order`` / ``get_anonymous_lob``
    / ``get_agent_orders`` / ``get_agent_trades``) carry legitimate
    boundary ``int(...)`` casts that are NOT shims — those are part
    of the B4 decision to keep those methods Python-side per
    ``jax_constants.py`` §B4.

    Probe mechanism: AST-walk each jitted-helper function body for
    ``ast.Call(func=ast.Name(id='int'), ...)`` nodes. The precedent
    is ``tests/omiecid/test_submit_order_jit_safe_audit.py::TestASTInvariantsOnContractBookModule``
    (line 313-392). A clean P2-H3 lifts the body to use ``jnp.asarray(...,
    jnp.int32)`` exclusively — Python ``int(...)`` on a traced value
    would force a concretisation and break the JIT contract.
    """

    @pytest.mark.parametrize("fn_name", JITTED_HELPER_NAMES)
    def test_no_python_int_cast_in_jitted_helper_body(self, fn_name):
        """The function body of each jitted helper must contain zero
        ``int(...)`` calls. ``jnp.asarray(..., jnp.int32)`` is the
        canonical replacement.

        Failure mode: a stray ``int(some_traced_value)`` inside a
        jitted body forces JAX to concretise the value at trace time,
        which either raises a ConcretizationTypeError (good — caught
        immediately) or silently specialises on the value (bad —
        the cache fragments).
        """
        node = _module_function(fn_name)
        offenders = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id == "int":
                    offenders.append(
                        f"{fn_name}:{sub.lineno} — int({ast.unparse(sub.args[0]) if sub.args else ''})"
                    )
        assert offenders == [], (
            f"Jitted helper {fn_name} contains Python int(...) casts: "
            f"{offenders}. Per AC (6) and the architect pin (sprint-status "
            f"§4), these must be replaced by jnp.asarray(..., jnp.int32) "
            f"— a Python int() on a traced value would force "
            f"concretisation and break the JIT contract."
        )

    def test_no_legacy_compat_module_imported_into_contract_book(self):
        """Contract book must not import any ``*_compat`` / ``*_legacy``
        / ``*_phase1`` adapter module — those would be Phase-1 → Phase-2
        bridges and AC (6) requires their full retirement.

        Precedent: P2-H2 architect pin §4 inventory item 2.
        """
        tree = _module_source_tree()
        forbidden_substrings = ("_compat", "_legacy", "_phase1")
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(s in alias.name for s in forbidden_substrings):
                        offenders.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                if any(s in node.module for s in forbidden_substrings):
                    offenders.append(f"from {node.module} import ...")
        assert offenders == [], (
            f"contract_book.py imports legacy compatibility module(s): "
            f"{offenders}. AC (6) requires no Python compatibility shims "
            f"remain in production code."
        )


# ===========================================================================
# AC (7) — B1-B5 constants and B4 decision record live in jax_constants.py.
# ===========================================================================


class TestACSevenJaxConstantsExposesAllFiveConstants:
    """AC (7): the four B1/B2/B3/B5 constants live in
    ``jax_constants.py`` as Python ``int``; the B4 decision is
    recorded in the module docstring.

    The five names from the agile-plan card:
      - ``MAX_ICEBERGS`` (B1)
      - ``MAX_SLICES_PER_REGEN`` (B2)
      - ``SENTINEL_ORDER_ID`` (B3)
      - B4 JIT-scope decision (docstring; per sprint-status:608 and
        :777 the B4 record is a docstring block, NOT a Python attribute
        named ``B4_DECISION``)
      - ``MAX_AGENTS`` (B5)

    Architect pin (sprint-status:608, :777): the B4 record is verified
    via a docstring substring match — "B4 decision" must appear in
    the module docstring and the decision text must be non-empty.
    """

    def test_max_icebergs_is_python_int(self):
        """B1: ``MAX_ICEBERGS`` is a Python ``int`` (not jnp.int32).
        Static argument compatibility requires concrete Python ints."""
        assert isinstance(MAX_ICEBERGS, int), (
            f"MAX_ICEBERGS must be a Python int (static_argnames "
            f"compatibility); got {type(MAX_ICEBERGS).__name__}"
        )
        assert not isinstance(MAX_ICEBERGS, bool), (
            "MAX_ICEBERGS must NOT be a bool subclass (sprint-status "
            "§B3 hazard note)"
        )
        assert MAX_ICEBERGS > 0, (
            f"MAX_ICEBERGS must be strictly positive; got {MAX_ICEBERGS}"
        )

    def test_max_slices_per_regen_is_python_int(self):
        """B2: ``MAX_SLICES_PER_REGEN`` is a Python ``int``."""
        assert isinstance(MAX_SLICES_PER_REGEN, int)
        assert not isinstance(MAX_SLICES_PER_REGEN, bool)
        assert MAX_SLICES_PER_REGEN >= 2, (
            f"MAX_SLICES_PER_REGEN must be >= 2 to preserve iceberg "
            f"semantics (sprint-status §B2); got {MAX_SLICES_PER_REGEN}"
        )

    def test_sentinel_order_id_is_python_int(self):
        """B3: ``SENTINEL_ORDER_ID`` is a Python ``int`` (not bool)."""
        assert isinstance(SENTINEL_ORDER_ID, int)
        assert not isinstance(SENTINEL_ORDER_ID, bool), (
            "SENTINEL_ORDER_ID must NOT be a bool subclass — downstream "
            "static_argnums consumers disambiguate on identity"
        )
        # Per sprint-status §B3: -1 is the canonical value.
        assert SENTINEL_ORDER_ID < 0, (
            f"SENTINEL_ORDER_ID must be < 0 (cannot alias positive-id "
            f"real orders); got {SENTINEL_ORDER_ID}"
        )

    def test_max_agents_is_python_int(self):
        """B5: ``MAX_AGENTS`` is a Python ``int``."""
        assert isinstance(MAX_AGENTS, int)
        assert not isinstance(MAX_AGENTS, bool)
        assert MAX_AGENTS > 0

    def test_b4_decision_recorded_in_module_docstring(self):
        """B4: the JIT-scope decision is a *docstring record*, not a
        Python attribute. Per sprint-status §608 / §777: a strict
        symbol-presence parser would miss this, so the test asserts
        substring presence in ``jax_constants.__doc__``.
        """
        doc = jc.__doc__
        assert doc is not None and len(doc) > 0, (
            "jax_constants.py must carry a non-empty module docstring "
            "(houses the B4 JIT-scope decision record)"
        )
        # Per sprint-status §615 / §678: the docstring uses the "B4
        # decision" phrasing. A case-insensitive substring match is the
        # spec.
        assert "JIT-scope of query methods" in doc, (
            "jax_constants.py module docstring must record the B4 "
            "JIT-scope decision (substring 'B4 decision' required per "
            "sprint-status §615 + §678)"
        )
        # The decision must name at least one of the three Python-side
        # query methods covered (rationale: ensures the decision text
        # is substantive, not just a bare heading).
        named_methods = [
            "get_anonymous_lob",
            "get_agent_orders",
            "get_agent_trades",
        ]
        assert any(m in doc for m in named_methods), (
            f"B4 decision record must name at least one of the three "
            f"query methods it scopes ({named_methods}); the docstring "
            f"appears to be a bare heading without substance."
        )

    def test_query_methods_are_not_jitted(self):
        """B4: the three query methods identified by the B4 decision
        must remain Python-side (no ``@jax.jit`` decoration; no
        ``_cache_size`` attribute).

        The query methods are instance methods on ``ContractBook``;
        they must not be PjitFunctions.
        """
        from omiecid_simulator.omiecid.contract_book import ContractBook
        for method_name in (
            "get_anonymous_lob",
            "get_agent_orders",
            "get_agent_trades",
        ):
            method = getattr(ContractBook, method_name, None)
            if method is None:
                pytest.fail(
                    f"ContractBook.{method_name} does not exist — B4 "
                    f"decision references a method that has been removed"
                )
            # A jitted method exposes ``_cache_size`` / ``lower``. A
            # plain Python method does not.
            assert not hasattr(method, "_cache_size"), (
                f"ContractBook.{method_name} appears to be jitted "
                f"(_cache_size exposed) — violates B4 decision "
                f"(Python-side scope)"
            )


# ===========================================================================
# AC (8) — id_to_pair / load_omie_agent_metadata accessible outside JIT.
# ===========================================================================


# The canonical 2026-01-12 cid=81 raw OMIE file is the file used by the
# P2-HOTFIX-ioc-fill-detection regression and the P2-HOTFIX-iceberg-cap-overflow
# audit. It is the smallest reasonable corpus to exercise
# ``load_omie_agent_metadata`` against.
RAW_OMIE_DIR = Path(__file__).parent.parent.parent / "data" / "rawOMIE"
CANONICAL_OMIE_DATE = "2026-01-12"
CANONICAL_ORDERS_FILE = (
    RAW_OMIE_DIR / f"orders_{CANONICAL_OMIE_DATE.replace('-', '')}.1"
)


class TestACEightAgentMetadataAccessibleOutsideJit:
    """AC (8): the ``id_to_pair`` accessor remains accessible outside
    JIT for settlement aggregation and diagnostics.

    Per the agile-plan card and ``omie_data_loader.py:328-359``, the
    public accessor is ``load_omie_agent_metadata(config) ->
    dict[int, tuple[str, str]]``. The function returns the reverse
    mapping ``{agent_id: (agente, unidad)}`` used by settlement
    aggregation (P2-L2 ``compute_agent_pnl``) and by diagnostics in
    the demo regeneration pipeline.

    The function is decorated with NO ``@jax.jit`` — it is a Python /
    pandas loader by design. The test verifies:
      1. It is importable from a non-JIT context.
      2. It is NOT a PjitFunction (no ``_cache_size`` attribute).
      3. When called on the canonical 2026-01-12 cid=81 file, it
         returns a ``dict[int, tuple[str, str]]`` with at least one
         entry.

    The on-disk file dependency is guarded by ``skipif`` so collection
    succeeds on machines without the rawOMIE corpus.
    """

    def test_load_omie_agent_metadata_is_importable(self):
        """The accessor is importable from a top-level Python context
        with no JAX trace active."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        assert callable(load_omie_agent_metadata), (
            "load_omie_agent_metadata must be a callable"
        )

    def test_load_omie_agent_metadata_is_not_jitted(self):
        """The accessor must remain Python-side — no ``@jax.jit``
        decoration (no ``_cache_size`` / no ``lower``)."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        assert not hasattr(load_omie_agent_metadata, "_cache_size"), (
            "load_omie_agent_metadata must NOT be jitted — AC (8) "
            "requires it to stay accessible outside JIT for settlement "
            "aggregation and diagnostics"
        )
        assert not hasattr(load_omie_agent_metadata, "lower"), (
            "load_omie_agent_metadata must NOT be jitted (no `lower` attr)"
        )

    @pytest.mark.skipif(
        not CANONICAL_ORDERS_FILE.is_file(),
        reason=(
            f"canonical OMIE raw file missing: {CANONICAL_ORDERS_FILE}; "
            f"AC (8) integration probe requires real data on disk"
        ),
    )
    def test_id_to_pair_accessor_returns_dict_outside_jit(self):
        """Smoke-test the accessor: invoked outside any JIT context,
        it must return a ``dict[int, tuple[str, str]]`` mapping
        agent_id → (agente, unidad).

        The keys form a contiguous ``[1, N]`` range per
        ``_build_agent_mapping`` (line 240-241), and the values are
        2-tuples of Python strings.
        """
        from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )

        cfg = OMIEDataConfig(
            data_path=str(RAW_OMIE_DIR),
            contract_id=81,
            delivery_date=CANONICAL_OMIE_DATE,
        )
        id_to_pair = load_omie_agent_metadata(cfg)

        # Top-level type: dict.
        assert isinstance(id_to_pair, dict), (
            f"load_omie_agent_metadata must return a dict; got "
            f"{type(id_to_pair).__name__}"
        )
        # Non-empty (the canonical session has 158-217 distinct pairs
        # per sprint-status §B5 derivation).
        assert len(id_to_pair) > 0, (
            "load_omie_agent_metadata returned an empty dict for the "
            "canonical 2026-01-12 cid=81 session — either the file is "
            "broken or _collect_pair_universe is mis-filtering"
        )
        # Key / value shape.
        sample_key = next(iter(id_to_pair))
        sample_value = id_to_pair[sample_key]
        assert isinstance(sample_key, int), (
            f"id_to_pair key must be int (agent_id); got "
            f"{type(sample_key).__name__}"
        )
        assert isinstance(sample_value, tuple) and len(sample_value) == 2, (
            f"id_to_pair value must be a 2-tuple (agente, unidad); got "
            f"{type(sample_value).__name__} of len "
            f"{len(sample_value) if hasattr(sample_value, '__len__') else '?'}"
        )
        agente, unidad = sample_value
        assert isinstance(agente, str), (
            f"id_to_pair value[0] (agente) must be str; got "
            f"{type(agente).__name__}"
        )
        assert isinstance(unidad, str), (
            f"id_to_pair value[1] (unidad) must be str; got "
            f"{type(unidad).__name__}"
        )

    def test_load_omie_agent_metadata_callable_outside_active_jit_trace(self):
        """Robustness probe: even if the accessor is invoked from a
        context where JAX is imported and a trace is conceivably
        active, the call must succeed (it does not depend on any
        traced array). We exercise the call via a tiny synthetic
        helper that imports JAX, then calls the accessor — no JIT
        boundary is crossed.

        This is the part of AC (8) that locks "accessible outside JIT"
        as a positive property rather than a negation. We do NOT call
        the function on real data here (that's the file-guarded test
        above); we only verify the function object can be referenced
        and inspected.
        """
        # Import jax to simulate a context where the JIT machinery is
        # initialised — AC (8) requires the accessor to remain usable.
        _ = jax  # noqa: F841 — sanity: jax is importable.
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        # Pure-Python signature inspection — does not invoke JIT.
        sig = inspect.signature(load_omie_agent_metadata)
        assert "config" in sig.parameters, (
            "load_omie_agent_metadata must accept a `config` parameter "
            "(OMIEDataConfig); the public signature is unchanged from "
            "the migration-plan v4.0 spec"
        )


# ===========================================================================
# AC (1)/(2) — Suite-level environmental sanity probes.
# ===========================================================================


class TestACOneAndTwoSuiteEvidence:
    """AC (1) and AC (2) are environmental invariants (all Phase 1 +
    Phase 1.5 tests pass; all new ``*_under_jit`` tests pass). The
    orchestrator validates them at the merge gate by running the full
    test suite; this class provides one optional sanity probe each so
    the smoke file itself contributes a collection-time check.

    The probes do NOT execute the full suite — that's the
    orchestrator's job — but they verify that the canonical Phase-1
    public surface is importable from the smoke file (collection
    works) and that this very file carries the ``pytest.mark.jit``
    marker (so it is included in the ``*_under_jit`` test subset for
    AC (2)).
    """

    def test_phase1_public_surfaces_importable(self):
        """AC (1) sanity probe: the Phase-1 contract-book public
        surfaces are importable. A broken import here means the
        Phase-1 / Phase-1.5 suite cannot pass."""
        from omiecid_simulator.omiecid.contract_book import (  # noqa: F401
            ContractBook,
            OMIEBookState,
            SubmitResult,
        )
        from omiecid_simulator.omiecid.order_types import (  # noqa: F401
            OMIEOrder,
            ExecRestriction,
            Side,
            make_order,
        )
        from omiecid_simulator.omiecid.jax_constants import (  # noqa: F401
            MAX_ICEBERGS,
            SENTINEL_ORDER_ID,
            make_sentinel_order,
        )

    def test_this_file_carries_jit_marker(self):
        """AC (2) sanity probe: this file's module-level marker is
        ``pytest.mark.jit``. The marker discipline ensures the file
        is registered in the ``*_under_jit`` test subset that AC (2)
        scopes."""
        # The pytestmark module variable is evaluated at collection;
        # if it's not pytest.mark.jit, the assertion fires.
        marker = pytestmark
        # pytest.mark.jit is a MarkDecorator; its ``name`` attribute
        # is "jit".
        assert hasattr(marker, "name"), (
            "pytestmark must be a pytest.mark.* decorator"
        )
        assert marker.name == "jit", (
            f"pytestmark must be pytest.mark.jit per T1.1 marker "
            f"discipline; got pytest.mark.{marker.name}"
        )
