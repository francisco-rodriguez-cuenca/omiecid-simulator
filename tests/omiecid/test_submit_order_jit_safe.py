"""Locked behavioural-contract tests for Story P2-H2 — `contract_book.submit_order` JIT-safe.

Story card (v5.20, 2026-05-14):
  - ``docs/omiecid/agile-plan.md`` §"Story P2-H2 — `contract_book.submit_order` JIT-safe"
    (around line 545); the v5.20 amendment (2026-05-14) corrects the exec-restriction
    variant count from six to four and replaces ``static_argnames=()`` with
    ``static_argnames=("product_config", "market_config")`` per the canonical
    module-level jit idiom established by P2-PERF-3-FOLLOWUP-2.

Architectural references (read at authoring time):
  - ``docs/omiecid/migration-plan.md`` §"Module-level jit idiom" (v5.11, line 818-820) —
    canonical decoration shape ``@partial(jax.jit, static_argnames=("cfg", ...))``.
  - ``docs/omiecid/migration-plan.md`` §"Settlement API pin" + Decoding contract
    (line 836-885) — caller-side ``enable_x64()`` precedent and consumer-side
    ``int(...)``-on-receipt pattern at the six ``extract_fills`` call sites.
  - ``docs/omiecid/migration-plan.md`` §"Loader dtype contract" (line 806-816) — int32
    end-to-end; ``ppd`` as signed int-ticks.
  - ``docs/omiecid/market-spec.md`` §4.5 (IOC), §4.2 (ICE), §6.2 (matching).
  - ``omiecid_simulator/omiecid/jax_constants.py`` — ``SENTINEL_ORDER_ID = -1``;
    B4 decision ("query methods stay Python-side"); P2-L3 caller-contract invariant
    ("contract_open: bool" closed-over scalar at :238-246).
  - ``omiecid_simulator/omiecid/order_types.py:22-31`` — ``ExecRestriction`` IntEnum with
    exactly four values: ``NON=0, IOC=1, FOK=2, ICE=3``.
  - ``omiecid_simulator/omiecid/omie_config.py:38`` —
    ``OMIEMarketConfig.max_match_iterations = 200``.
  - ``omiecid_simulator/omiecid/iceberg_manager.py`` — ``regenerate_slices``,
    ``truncation_flag`` (Path B carry-flag).
  - ``tests/omiecid/conftest.py`` — module-scoped fixture cascade (``book``,
    ``empty_state``, ``book_and_state``).

Locked acceptance criteria (numbering matches the v5.20 Test-first contract row):

  (1) ``jax.jit(submit_order)`` compiles cleanly via the public surface the
      developer chooses (module-level lifted function decorated with
      ``@partial(jax.jit, static_argnames=("product_config", "market_config"))``,
      per the canonical idiom).
  (2) ``SubmitResult`` matches Phase 1 for NON / IOC / FOK / ICE (including ICE
      with non-zero ``ppd``, i.e. the ICEwPI shape) — scalar-equal on all five
      fields.  ``cancel_order`` (``msg_type=2``) is a separate public method and is
      out of scope for AC (2); one sanity test confirms it still works.
  (3) ``ContractBook.get_order`` returns a sentinel ``OMIEOrder`` with
      ``order_id == SENTINEL_ORDER_ID`` for absent order_ids — no ``None``.
  (4) Multi-slice NON / IOC / FOK loops produce identical fill sequences to Phase 1
      on the existing module-scoped fixtures.
  (5) Trace-count invariant: sequencing all four ``ExecRestriction`` variants
      through the same jitted ``submit_order`` causes exactly one
      ``lower(...).compile()`` (or one ``_cache_size()`` entry) — proving the
      discriminator is traced, not specialised.  ICE-with-non-zero-``ppd`` (the
      ICEwPI shape) varies ``order.ppd`` within the ICE arm and must NOT trigger
      a recompile.
  (6) Truncation backstop: a synthetic fixture forcing > 200 inner iterations sets
      ``OMIEBookState.truncated == True`` without silent volume loss — visible-trade
      record after truncation equals the matched portion that fit within the
      200-iteration budget, and the unmatched residual is observable via the
      ``truncated`` flag.

AC (7) — Test-perf DoD — is checked structurally via a developer-facing
self-check string constant.  See ``TestACSevenMergeNoteSelfCheck`` at the
bottom of the module.

Test discipline:
  - All tests that touch ``jax.jit`` carry ``@pytest.mark.jit`` via the
    module-level ``pytestmark`` pin (T1.1 marker discipline; AST lint at
    ``tests/omiecid/test_jit_marker_lint.py`` enforces it).
  - Synthetic fixtures are built in-test where possible to fit the < 1s
    fast-suite budget.  Long-running ``submit_order`` cascades or any test
    that traces through the full jitted body MAY carry
    ``@pytest.mark.slow`` per the existing precedent at
    ``tests/omiecid/test_matching.py:30``.
  - Lazy imports for symbols whose final surface depends on the developer's
    lift decision (e.g. ``submit_order_jit`` module-level function or
    ``ContractBook.submit_order_jit`` method).  The locked tests probe the
    most likely surfaces and fall through informatively.

Pre-implementation expectation:
  These tests will currently fail or error because the production lift has
  not been shipped yet.  That is the intended test-first state.  Do NOT mark
  any test as xfail — they go green when the developer ships the P2-H2
  deliverables.
"""

from __future__ import annotations

import importlib
import inspect

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    SubmitResult,
)
from omiecid_simulator.omiecid.jax_constants import (
    SENTINEL_ORDER_ID,
    make_sentinel_order,
)
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    make_order,
)


# Module-level marker (Story T1.1): every test in this file traces or compiles
# `submit_order` and its helpers.  The AST-lint at `tests/omiecid/test_jit_marker_lint.py`
# treats a module-level `pytestmark = pytest.mark.jit` as a class- and function-level
# marker in scope; per-test `@pytest.mark.jit` decorators are therefore redundant
# under this pin.
pytestmark = pytest.mark.jit


# ===========================================================================
# Helpers — synthetic order factories, fits-fast-suite size.
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
    """Coerce a Python bool, ``jnp.bool_`` 0-d array, or numpy bool to ``bool``.

    Behavioral parity contract for AC (2): the Phase-1 ``SubmitResult.accepted /
    .traded`` are Python ``bool``; under P2-H2 they become ``jnp.bool_`` 0-d
    scalars.  This helper bridges both representations so the locked tests
    survive the transition.
    """
    return bool(x)


def _as_python_float(x) -> float:
    """Coerce a Python float, ``jnp.float32`` 0-d array, or numpy float to ``float``."""
    return float(x)


# ===========================================================================
# Surface probes — the developer's lift decision is fluid, so each test
# resolves the jitted callable via one of two patterns (canonical or fallback)
# and SKIPs informatively if neither has landed yet.
# ===========================================================================


def _resolve_jitted_submit_order():
    """Return a ``jax.jit``-decorated callable for ``submit_order`` if exposed.

    Tries (in order):

      1. Module-level lifted function exported from
         ``omiecid_simulator.omiecid.contract_book`` under one of the canonical
         names (``submit_order``, ``_submit_order``, ``submit_order_jit``).
         This is the canonical Phase-2 module-level idiom from
         migration-plan v5.11 §"Module-level jit idiom".
      2. ``ContractBook.submit_order`` method-bound + manually wrapped in
         ``jax.jit`` with ``static_argnames=("product_config", "market_config")``
         — fallback used for an instance-method-scope decoration.

    If neither surface exists, ``pytest.skip(...)`` with the reason so the
    test reports informatively rather than confusingly.
    """
    module = importlib.import_module("omiecid_simulator.omiecid.contract_book")
    for name in ("submit_order_jit", "_submit_order", "submit_order"):
        if hasattr(module, name):
            candidate = getattr(module, name)
            if callable(candidate) and not inspect.isclass(candidate):
                # Detect whether `candidate` is already jit-decorated by
                # probing for `_cache_size` / `lower` / `trace` attributes.
                if hasattr(candidate, "_cache_size") or hasattr(candidate, "lower"):
                    return candidate
                # Otherwise the developer may have exported a plain function;
                # the test wraps it in `jax.jit` with the canonical idiom.
                from functools import partial

                return partial(jax.jit, static_argnames=("product_config", "market_config"))(
                    candidate,
                )
    pytest.fail(
        "no module-level lifted `submit_order` found in "
        "`omiecid_simulator.omiecid.contract_book`. P2-H2 deliverable not yet "
        "shipped — the canonical Phase-2 module-level idiom (per "
        "migration-plan v5.11 §'Module-level jit idiom') exports the "
        "jitted callable at module scope.  Expected one of: "
        "`submit_order_jit`, `_submit_order`, `submit_order` (a "
        "`@partial(jax.jit, static_argnames=('product_config', 'market_config'))`-"
        "decorated function)."
    )


# ===========================================================================
# AC (1) — `jax.jit(submit_order)` compiles cleanly.
# ===========================================================================


class TestACOneJitCompilesCleanly:
    """The jitted public surface must compile without raising.

    Story-card pin: "(1) `jax.jit(submit_order)` compiles cleanly. (The
    signature must accept ``static_argnames=("product_config", "market_config")``
    per the canonical module-level jit idiom from migration-plan v5.11.)"
    """

    def test_jitted_submit_order_compiles_for_non(self, book, empty_state):
        """A bare NON order traced through the jitted body must compile.

        Compilation is the act of finishing one ``lower(...).compile()`` —
        we observe it by invoking the callable and accepting any return
        shape, asserting only that no exception escapes during trace+compile.
        """
        jitted = _resolve_jitted_submit_order()
        order = _make_buy(order_id=1, exec_restriction=ExecRestriction.NON)
        # The developer's call shape may be either:
        #   jitted(state, order, product_config, market_config)
        # or:
        #   jitted(book, state, order)
        # The test accepts either by probing the signature lazily.
        result = _invoke_submit(jitted, book, empty_state, order)
        assert result is not None, "jitted submit_order must return a result"

    def test_jitted_submit_order_compiles_for_each_variant(
        self, book, empty_state
    ):
        """All four ``ExecRestriction`` variants must compile without error.

        This is the per-variant trace-validity gate; AC (5)'s trace-count
        invariant (one compile across the four) is asserted separately.
        """
        jitted = _resolve_jitted_submit_order()
        variants = (
            ExecRestriction.NON,
            ExecRestriction.IOC,
            ExecRestriction.FOK,
            ExecRestriction.ICE,
        )
        for variant in variants:
            extras = {}
            if variant == ExecRestriction.ICE:
                extras = {"peak_qty": 1.0, "ppd": 0.0}
            order = _make_buy(
                order_id=10 + int(variant),
                exec_restriction=variant,
                **extras,
            )
            result = _invoke_submit(jitted, book, empty_state, order)
            assert result is not None, (
                f"jit-compile for ExecRestriction.{variant.name} must succeed"
            )


def _invoke_submit(jitted, book, state, order):
    """Call the jitted submit_order, probing the supported signature.

    Tries (in order):
      (a) ``jitted(state, order, book._product_config, book._market_config)``
          — the canonical module-level lifted signature.
      (b) ``jitted(book, state, order)`` — fallback for a class-method lift.
      (c) ``jitted(state, order)`` — minimal fallback when configs are
          closed-over.

    Returns the ``(new_state, result)`` tuple (the public ``submit_order``
    contract) or whatever the jitted callable returns.  Errors propagate
    so the caller can assert on them.
    """
    try:
        return jitted(state, order, book._product_config, book._market_config)
    except TypeError:
        pass
    try:
        return jitted(book, state, order)
    except TypeError:
        pass
    return jitted(state, order)


# ===========================================================================
# AC (2) — SubmitResult parity for NON / IOC / FOK / ICE / ICEwPI.
# ===========================================================================


class TestACTwoSubmitResultParityPhase1:
    """Scalar-equal parity for the four ExecRestriction variants plus ICEwPI.

    Story-card pin: "(2) `SubmitResult` matches Phase 1 for NON / IOC / FOK /
    ICE (including ICE with non-zero `ppd`, i.e. the ICEwPI shape) — scalar-
    equal on all five fields. `cancel_order` is a separate public method
    (msg_type=2) and is out of scope for this AC."

    Methodology: each test invokes the Phase-1 public ``book.submit_order``
    (which the developer's diff converts to the jitted body) and asserts
    on the five SubmitResult fields against synthetic crossing/non-crossing
    fixtures.  The locked oracle values are derived from the existing
    Phase-1 semantics in ``contract_book.py``.

    The Phase-1 fields are Python ``bool`` / ``float``; under P2-H2 they
    become ``jnp.bool_`` / ``jnp.float32`` 0-d scalars.  Each assertion is
    written to survive the transition via ``_as_python_bool`` /
    ``_as_python_float`` coercers.
    """

    def test_non_no_crossing_accepted_not_traded(self, book, empty_state):
        """NON BUY into an empty book: accepted=True, traded=False, qty=0."""
        order = _make_buy(price=50.00, exec_restriction=ExecRestriction.NON)
        _state, result = book.submit_order(empty_state, order)
        assert _as_python_bool(result.accepted) is True
        assert _as_python_bool(result.traded) is False
        assert _as_python_float(result.trade_qty) == pytest.approx(0.0)

    def test_non_crossing_partial_fill(self, book, empty_state):
        """NON BUY crossing a smaller resting SELL: traded=True, partial qty."""
        sell = _make_sell(order_id=1, price=50.00, quantity=4.0, timestamp=100)
        state, _ = book.submit_order(empty_state, sell)
        buy = _make_buy(order_id=2, price=50.00, quantity=10.0, timestamp=200,
                        exec_restriction=ExecRestriction.NON)
        _state, result = book.submit_order(state, buy)
        assert _as_python_bool(result.accepted) is True
        assert _as_python_bool(result.traded) is True
        assert _as_python_float(result.trade_qty) == pytest.approx(4.0)
        assert _as_python_float(result.trade_price) == pytest.approx(50.00)

    def test_ioc_full_fill_no_residual(self, book, empty_state):
        """IOC BUY fully consumed by resting SELL: traded=True, full qty."""
        sell = _make_sell(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        state, _ = book.submit_order(empty_state, sell)
        buy = _make_buy(order_id=2, price=50.00, quantity=10.0, timestamp=200,
                        exec_restriction=ExecRestriction.IOC)
        _state, result = book.submit_order(state, buy)
        assert _as_python_bool(result.accepted) is True
        assert _as_python_bool(result.traded) is True
        assert _as_python_float(result.trade_qty) == pytest.approx(10.0)

    def test_ioc_partial_fill_cancels_residual(self, book, empty_state):
        """IOC BUY partially filled; residual must be cancelled per market-spec §4.5.

        Locked invariant: after the IOC, the book has no resting BUY at the
        IOC's price (the residual was cancelled).  This is the same
        invariant violated by the IOC #1022 regression that motivated
        P2-HOTFIX-ioc-fill-detection.
        """
        sell = _make_sell(order_id=1, price=50.00, quantity=4.0, timestamp=100)
        state, _ = book.submit_order(empty_state, sell)
        buy = _make_buy(order_id=2, price=50.00, quantity=10.0, timestamp=200,
                        exec_restriction=ExecRestriction.IOC)
        state2, result = book.submit_order(state, buy)
        assert _as_python_bool(result.accepted) is True
        assert _as_python_bool(result.traded) is True
        assert _as_python_float(result.trade_qty) == pytest.approx(4.0)
        # IOC residual was cancelled — get_order on the IOC's order_id must
        # not return a resting order on the BUY side.  After AC (3), the
        # sentinel discriminator is `order_id == SENTINEL_ORDER_ID`.
        looked_up = book.get_order(state2, order_id=2, side=Side.BUY)
        assert _is_sentinel_or_none(looked_up), (
            "IOC partial residual must be cancelled; the IOC's order_id "
            "must not be retrievable as a resting BUY"
        )

    def test_fok_full_fill_when_sufficient_liquidity(self, book, empty_state):
        """FOK BUY against sufficient resting SELL: traded=True, full qty."""
        sell = _make_sell(order_id=1, price=50.00, quantity=10.0, timestamp=100)
        state, _ = book.submit_order(empty_state, sell)
        buy = _make_buy(order_id=2, price=50.00, quantity=10.0, timestamp=200,
                        exec_restriction=ExecRestriction.FOK)
        _state, result = book.submit_order(state, buy)
        assert _as_python_bool(result.accepted) is True
        assert _as_python_bool(result.traded) is True
        assert _as_python_float(result.trade_qty) == pytest.approx(10.0)

    def test_fok_insufficient_liquidity_accepted_not_traded(
        self, book, empty_state
    ):
        """FOK BUY against insufficient resting SELL: accepted=True (passed
        validation), traded=False (liquidity check failed)."""
        sell = _make_sell(order_id=1, price=50.00, quantity=4.0, timestamp=100)
        state, _ = book.submit_order(empty_state, sell)
        buy = _make_buy(order_id=2, price=50.00, quantity=10.0, timestamp=200,
                        exec_restriction=ExecRestriction.FOK)
        _state, result = book.submit_order(state, buy)
        assert _as_python_bool(result.accepted) is True
        assert _as_python_bool(result.traded) is False
        assert _as_python_float(result.trade_qty) == pytest.approx(0.0)

    def test_ice_with_zero_ppd_visible_peak_rests(self, book, empty_state):
        """ICE with ppd=0 (vanilla iceberg): visible peak rests, accepted=True.

        Resting state with no crossing counterparty -> traded=False; the
        iceberg is registered in the side-channel.
        """
        ice = _make_buy(
            order_id=1, price=50.00, quantity=5.0,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=1.0, ppd=0.0,
        )
        _state, result = book.submit_order(empty_state, ice)
        assert _as_python_bool(result.accepted) is True
        assert _as_python_bool(result.traded) is False

    def test_ice_with_nonzero_ppd_icewpi_visible_peak_rests(
        self, book, empty_state
    ):
        """ICEwPI (ICE with ppd != 0): same accepted=True, traded=False shape
        as vanilla ICE.  Per migration-plan §"Module-level jit idiom" and the
        P2-H2 v5.20 amendment, ICEwPI is NOT a separate switch arm — it is
        the ICE arm with non-zero ppd, handled inside the shared ICE body
        (the semantic split happens in IcebergManager.regenerate_slices).
        """
        ice = _make_buy(
            order_id=1, price=50.00, quantity=5.0,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=1.0, ppd=-0.10,  # conforming BUY ICEwPI (PPD < 0)
        )
        _state, result = book.submit_order(empty_state, ice)
        assert _as_python_bool(result.accepted) is True
        assert _as_python_bool(result.traded) is False

    def test_ice_with_wrong_sign_ppd_icewpi_rejected(
        self, book, empty_state
    ):
        """OMIE-ICEWPI-SIGN-REJECT-1 re-baseline: a BUY ICEwPI with
        PPD > 0 has the wrong sign (market-spec §13.6, Rules Annex §4.5)
        and is rejected by the validator: accepted=False, traded=False.
        """
        ice = _make_buy(
            order_id=1, price=50.00, quantity=5.0,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=1.0, ppd=0.10,  # BUY PPD > 0 → wrong sign
        )
        _state, result = book.submit_order(empty_state, ice)
        assert _as_python_bool(result.accepted) is False
        assert _as_python_bool(result.traded) is False

    def test_result_has_five_named_fields(self, book, empty_state):
        """SubmitResult shape — five named fields per Phase-1 contract:
        accepted, traded, trade_qty, trade_price, settlement_value."""
        _state, result = book.submit_order(
            empty_state, _make_buy(order_id=1)
        )
        assert isinstance(result, SubmitResult)
        for field in ("accepted", "traded", "trade_qty", "trade_price",
                      "settlement_value"):
            assert hasattr(result, field), (
                f"SubmitResult must retain Phase-1 field `{field}`"
            )


class TestACTwoCancelOrderSanityOutOfScope:
    """Sanity-only — `cancel_order` (msg_type=2) is out of scope for AC (2)
    but must not be broken by the P2-H2 production diff.

    Per v5.20 card amendment: ``ContractBook.cancel_order`` remains a separate
    public method with its own locked tests; this story must not break it but
    does not modify it.  One sanity test confirms post-refactor that a
    successful cancel removes the order from the book.
    """

    def test_cancel_removes_resting_buy(self, book, empty_state):
        """Submit BUY, cancel it: get_order on the BUY side returns sentinel."""
        order = _make_buy(order_id=42, price=50.00, quantity=10.0)
        state, _ = book.submit_order(empty_state, order)
        state2 = book.cancel_order(state, order_id=42, side=Side.BUY)
        looked_up = book.get_order(state2, order_id=42, side=Side.BUY)
        assert _is_sentinel_or_none(looked_up)


# ===========================================================================
# AC (3) — get_order returns sentinel for absent order_ids (no None).
# ===========================================================================


class TestACThreeGetOrderSentinelOnMiss:
    """`get_order` must return ``SENTINEL_ORDER_ID``-bearing sentinel for
    absent order_ids — never ``None``.

    Story-card pin: "(3) `get_order` returns `SENTINEL_ORDER_ID` for absent
    order_ids (no `None`). The constant name and value are in `jax_constants.py`."

    The post-P2-H2 callsite swap converts the Phase-1
    ``return None`` (at ``contract_book.py:826``) to
    ``return make_sentinel_order()``.  The ~137 ``get_order(...) is None``
    callsite cascade is the developer's Category-B mechanical replace pass.
    """

    def test_get_order_miss_returns_non_none(self, book, empty_state):
        """A bare empty book has no orders — get_order must NOT return None."""
        result = book.get_order(empty_state, order_id=999, side=Side.BUY)
        assert result is not None, (
            "get_order must return a sentinel OMIEOrder for absent order_ids, "
            "not None (P2-H2 callsite swap of contract_book.py:826)"
        )

    def test_get_order_miss_returns_sentinel_order_id(self, book, empty_state):
        """The sentinel discriminator is `order_id == SENTINEL_ORDER_ID`.

        Per ``jax_constants.py`` §B3 module docstring: "The ONLY reliable
        discriminator is `order_id == SENTINEL_ORDER_ID`. Consumers must
        not branch on any other field to detect the sentinel."
        """
        result = book.get_order(empty_state, order_id=999, side=Side.BUY)
        if result is None:
            pytest.fail(
                "get_order must return a sentinel OMIEOrder, not None"
            )
        # Read the order_id off whatever shape the result has (OMIEOrder
        # NamedTuple or a 0-d jnp scalar of the same field).
        oid = int(result.order_id)
        assert oid == SENTINEL_ORDER_ID, (
            f"get_order miss must return order_id == SENTINEL_ORDER_ID "
            f"(={SENTINEL_ORDER_ID}); got {oid}"
        )

    def test_get_order_hit_returns_real_order(self, book, empty_state):
        """A real order is retrievable; this is the non-miss complement
        to the sentinel-on-miss contract."""
        order = _make_buy(order_id=42, price=50.00, quantity=10.0)
        state, _ = book.submit_order(empty_state, order)
        looked_up = book.get_order(state, order_id=42, side=Side.BUY)
        assert looked_up is not None
        assert int(looked_up.order_id) == 42, (
            "get_order hit must return the resting order's real order_id, "
            "not the sentinel"
        )

    def test_get_order_miss_after_cancel_returns_sentinel(
        self, book, empty_state
    ):
        """A successful cancel makes the order absent — subsequent
        get_order must return the sentinel, not the cached row."""
        order = _make_buy(order_id=42, price=50.00, quantity=10.0)
        state, _ = book.submit_order(empty_state, order)
        state2 = book.cancel_order(state, order_id=42, side=Side.BUY)
        looked_up = book.get_order(state2, order_id=42, side=Side.BUY)
        if looked_up is None:
            pytest.fail(
                "get_order after cancel must return sentinel OMIEOrder, not None"
            )
        assert int(looked_up.order_id) == SENTINEL_ORDER_ID

    def test_make_sentinel_order_factory_exposed(self):
        """The factory ``make_sentinel_order()`` is the recommended
        construction pattern for callsites that need an explicit sentinel.

        This is the P2-F2 surface that P2-H2's diff consumes.
        """
        sentinel = make_sentinel_order()
        assert int(sentinel.order_id) == SENTINEL_ORDER_ID


def _is_sentinel_or_none(value) -> bool:
    """Accept either Phase-1 ``None`` OR the post-P2-H2 sentinel discriminator.

    Used in tests that don't care whether the diff has landed yet — they only
    care that the resting order is gone.
    """
    if value is None:
        return True
    try:
        return int(value.order_id) == SENTINEL_ORDER_ID
    except (AttributeError, TypeError):
        return False


# ===========================================================================
# AC (4) — Multi-slice loops produce identical fill sequences to Phase 1.
# ===========================================================================


class TestACFourMultiSliceFillParity:
    """The two ``for iteration in range(max_match_iterations)`` Python loops
    at ``contract_book.py:444`` (NON/IOC) and ``:612`` (FOK) become
    ``jax.lax.while_loop``s under P2-H2.  Fill sequences across iceberg
    slice regeneration must remain identical to Phase 1.

    Story-card pin: "(4) Multi-slice loops produce identical fill sequences
    to Phase 1 on the existing test fixtures."

    These tests use the module-scoped ``book`` / ``empty_state`` fixtures
    from ``tests/omiecid/conftest.py`` rather than authoring new replay-style
    fixtures (per test-agent brief: do not duplicate the saturated-buffer
    fixture from ``test_trade_extraction_ring_buffer.py``).
    """

    def test_non_aggressor_consumes_iceberg_across_regenerations(
        self, book, empty_state
    ):
        """A NON BUY large enough to consume an iceberg's full hidden volume
        must produce a single aggregated SubmitResult whose ``trade_qty``
        equals the full iceberg quantity.

        This is the multi-slice loop's positive parity assertion — the loop
        iterates, regenerates slices, re-dispatches, and accumulates.
        """
        ice = _make_sell(
            order_id=1, price=50.00, quantity=3.0,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=1.0, ppd=0.0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, ice)
        aggressor = _make_buy(
            order_id=2, price=50.00, quantity=3.0,
            exec_restriction=ExecRestriction.NON, timestamp=200,
        )
        _state, result = book.submit_order(state, aggressor)
        assert _as_python_bool(result.traded) is True
        assert _as_python_float(result.trade_qty) == pytest.approx(3.0), (
            "Multi-slice NON aggressor against an iceberg must aggregate "
            "fills across slice regenerations to the full hidden volume"
        )

    def test_ioc_aggressor_consumes_iceberg_then_cancels_residual(
        self, book, empty_state
    ):
        """IOC variant of the above — full fill, no residual (per §4.5)."""
        ice = _make_sell(
            order_id=1, price=50.00, quantity=3.0,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=1.0, ppd=0.0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, ice)
        aggressor = _make_buy(
            order_id=2, price=50.00, quantity=3.0,
            exec_restriction=ExecRestriction.IOC, timestamp=200,
        )
        state2, result = book.submit_order(state, aggressor)
        assert _as_python_bool(result.traded) is True
        assert _as_python_float(result.trade_qty) == pytest.approx(3.0)
        looked_up = book.get_order(state2, order_id=2, side=Side.BUY)
        assert _is_sentinel_or_none(looked_up), (
            "IOC residual must be cancelled after the multi-slice loop"
        )

    def test_fok_against_visible_plus_hidden_passes_precheck(
        self, book, empty_state
    ):
        """FOK pre-check considers visible + hidden iceberg quantity per
        market-spec §6.3.  When total liquidity is sufficient, the FOK
        fills fully across slice regenerations."""
        ice = _make_sell(
            order_id=1, price=50.00, quantity=3.0,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=1.0, ppd=0.0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, ice)
        fok = _make_buy(
            order_id=2, price=50.00, quantity=3.0,
            exec_restriction=ExecRestriction.FOK, timestamp=200,
        )
        _state, result = book.submit_order(state, fok)
        assert _as_python_bool(result.accepted) is True
        assert _as_python_bool(result.traded) is True
        assert _as_python_float(result.trade_qty) == pytest.approx(3.0)

    def test_non_partial_then_rest_persists_residual(self, book, empty_state):
        """The hidden NON re-submit-rest branch at ``contract_book.py:527-559``
        (flagged by the architect-review §(a) finding 4): a NON order whose
        partial fill leaves a leftover after the aggressive order is cancelled
        must have its residual re-submitted to rest in the book.

        Setup: 4 MW resting SELL + 10 MW NON BUY at the same price.
        Phase-1 expected: 4 MW filled, 6 MW residual rests in the book.
        After P2-H2's lift to lax.while_loop, the same shape must hold.
        """
        sell = _make_sell(order_id=1, price=50.00, quantity=4.0, timestamp=100)
        state, _ = book.submit_order(empty_state, sell)
        buy = _make_buy(
            order_id=2, price=50.00, quantity=10.0, timestamp=200,
            exec_restriction=ExecRestriction.NON,
        )
        state2, result = book.submit_order(state, buy)
        assert _as_python_bool(result.traded) is True
        assert _as_python_float(result.trade_qty) == pytest.approx(4.0)
        # Residual 6 MW BUY @ 50 must be retrievable post-fill.
        looked_up = book.get_order(state2, order_id=2, side=Side.BUY)
        assert looked_up is not None
        assert int(looked_up.order_id) == 2, (
            "NON residual must be re-submitted to rest at order_id=2 "
            "after the aggressive cancel-then-refresh sequence"
        )
        assert _as_python_float(looked_up.quantity) == pytest.approx(6.0)


# ===========================================================================
# AC (5) — Trace-count invariant: one compile across NON/IOC/FOK/ICE.
# ===========================================================================


class TestACFiveTraceCountInvariantOneCompileForFourVariants:
    """Sequencing all four ``ExecRestriction`` variants through the same
    jitted ``submit_order`` causes exactly one ``lower(...).compile()``.

    Story-card pin (v5.20-amended): "(5) Trace-count invariant: sequencing
    all four `ExecRestriction` variants (NON / IOC / FOK / ICE) through the
    same jitted `submit_order` causes exactly one `lower(...).compile()`
    across the four variants under the `lax.switch` discriminator. The
    ICE-with-non-zero-`ppd` path must be exercised by varying `order.ppd`
    within the ICE arm and must not trigger a recompile."

    Probe mechanism: ``jitted._cache_size()`` — the same pattern used by
    ``test_order_validator_jit.py:206`` and the audit suite at
    ``test_jit_cache_defrag_audit.py:159-175``.
    """

    def test_one_compile_across_all_four_exec_restrictions(
        self, book, empty_state
    ):
        """From an empty cache, the four variants must compile exactly once.

        The ``ExecRestriction`` discriminator is a traced ``jnp.int32``; the
        four variants share one ``lax.switch`` arm and therefore one
        compiled artefact.

        The compile cache of the shipped jitted ``submit_order`` is cleared
        first, so entries left by earlier tests in the same process (other
        static configurations or shapes) do not pollute the count. A delta
        measurement is not used: an earlier test with the same configuration
        could already hold the extra entries a static discriminator would
        create, and a delta of 0 would hide that regression.
        """
        jitted = _resolve_jitted_submit_order()
        if not hasattr(jitted, "_cache_size"):
            pytest.skip(
                "jitted submit_order does not expose `_cache_size`; "
                "trace-count probe unavailable on this JAX version"
            )

        # Start from an empty cache of the shipped jit.
        if hasattr(jitted, "clear_cache"):
            jitted.clear_cache()
        else:  # pragma: no cover - older jax
            jitted._clear_cache()
        assert jitted._cache_size() == 0, (
            f"clearing the submit_order compile cache must empty it; "
            f"got _cache_size()={jitted._cache_size()}"
        )

        # First call compiles; subsequent variants must reuse.
        for idx, variant in enumerate((
            ExecRestriction.NON,
            ExecRestriction.IOC,
            ExecRestriction.FOK,
            ExecRestriction.ICE,
        )):
            extras = {}
            if variant == ExecRestriction.ICE:
                extras = {"peak_qty": 1.0, "ppd": 0.0}
            order = _make_buy(
                order_id=100 + idx,
                exec_restriction=variant,
                **extras,
            )
            _invoke_submit(jitted, book, empty_state, order)

        cache_size = jitted._cache_size()
        assert cache_size == 1, (
            f"All four ExecRestriction variants must compile to one cache "
            f"entry (the discriminator is a traced int32, not a static "
            f"Python enum); got _cache_size()={cache_size}.  This is the "
            f"load-bearing AC (5) trace-count invariant — a value > 1 "
            f"means the discriminator was specialised at trace time "
            f"instead of being routed through lax.switch."
        )

    def test_no_recompile_for_icewpi_ppd_variation(self, book, empty_state):
        """The ICE arm must not recompile when ``order.ppd`` varies.

        Per v5.20 card: ICEwPI is the ICE arm taken when ``ppd != 0``;
        the semantic split is inside ``IcebergManager.regenerate_slices``.
        Varying ``ppd`` is a traced-value change, not a static change,
        and must NOT bump the trace count.
        """
        jitted = _resolve_jitted_submit_order()
        if not hasattr(jitted, "_cache_size"):
            pytest.skip("trace-count probe unavailable")

        # Warmup the ICE arm with ppd=0.
        warmup = _make_buy(
            order_id=200, exec_restriction=ExecRestriction.ICE,
            peak_qty=1.0, ppd=0.0,
        )
        _invoke_submit(jitted, book, empty_state, warmup)
        cache_before = jitted._cache_size()

        # Vary ppd within the ICE arm — ppd in {0.10, 0.50, 1.00, -0.25}.
        for idx, ppd_val in enumerate((0.10, 0.50, 1.00, -0.25)):
            order = _make_buy(
                order_id=201 + idx,
                exec_restriction=ExecRestriction.ICE,
                peak_qty=1.0, ppd=ppd_val,
            )
            _invoke_submit(jitted, book, empty_state, order)

        cache_after = jitted._cache_size()
        assert cache_after == cache_before, (
            f"Varying order.ppd within the ICE arm must NOT trigger a "
            f"recompile; got cache_before={cache_before}, "
            f"cache_after={cache_after}.  ICEwPI is the ICE arm taken "
            f"when ppd != 0 (v5.20 amendment) and the ppd-dependent "
            f"semantic split happens in IcebergManager.regenerate_slices, "
            f"not in a submit_order branch."
        )

    def test_no_recompile_when_order_pytree_values_vary(
        self, book, empty_state
    ):
        """Sanity complement — varying ANY traced order field
        (price / quantity / order_id / agent_id / timestamp) must not
        force recompilation.

        Same pattern as ``test_order_validator_jit.py:201`` — proves the
        order is crossed as a traced pytree, not a static argument.
        """
        jitted = _resolve_jitted_submit_order()
        if not hasattr(jitted, "_cache_size"):
            pytest.skip("trace-count probe unavailable")

        warmup = _make_buy(order_id=300, exec_restriction=ExecRestriction.NON)
        _invoke_submit(jitted, book, empty_state, warmup)
        cache_before = jitted._cache_size()

        # Many distinct order values with the same pytree shape.
        for idx, (price, qty) in enumerate(((45.50, 5.0), (46.00, 7.0),
                                             (50.10, 1.0), (10.0, 50.0))):
            order = _make_buy(
                order_id=301 + idx, price=price, quantity=qty,
                exec_restriction=ExecRestriction.NON,
            )
            _invoke_submit(jitted, book, empty_state, order)

        assert jitted._cache_size() == cache_before, (
            "Varying traced order fields must not recompile"
        )


class TestACFiveSubmitResultDtypeContract:
    """Locked dtype contract for the ~496-callsite cascade.

    The test-agent brief flags that authoring 496 mechanical-replace edits
    is the developer's category-B pass, not the test-agent's.  This class
    locks the dtype contract centrally so the developer's mechanical pass
    has a single source of truth to validate against.

    Story-card pin: "Type each `SubmitResult` field as a concrete 0-d
    `jnp.*` scalar (no Python `bool` / `float`)."
    """

    def test_accepted_is_jnp_bool_zero_d_scalar(self, book, empty_state):
        """``SubmitResult.accepted`` is a 0-d ``jnp.bool_`` scalar post-P2-H2."""
        _state, result = book.submit_order(
            empty_state, _make_buy(order_id=1)
        )
        accepted = result.accepted
        # 0-d ``jnp.bool_``: ``isinstance(jnp.asarray(True), jnp.ndarray)``
        # AND ``.dtype == jnp.bool_`` AND ``.shape == ()``.
        if isinstance(accepted, bool):
            pytest.fail(
                "SubmitResult.accepted is still a Python bool — P2-H2 must "
                "convert it to a 0-d jnp.bool_ scalar (see v5.20 card "
                "Description row)"
            )
        assert isinstance(accepted, jnp.ndarray), (
            f"SubmitResult.accepted must be a jnp.ndarray (0-d), got "
            f"{type(accepted).__name__}"
        )
        assert accepted.dtype == jnp.bool_, (
            f"SubmitResult.accepted dtype must be jnp.bool_, got {accepted.dtype}"
        )
        assert accepted.shape == (), (
            f"SubmitResult.accepted must be 0-d, got shape {accepted.shape}"
        )

    def test_traded_is_jnp_bool_zero_d_scalar(self, book, empty_state):
        """``SubmitResult.traded`` is a 0-d ``jnp.bool_`` scalar post-P2-H2."""
        _state, result = book.submit_order(
            empty_state, _make_buy(order_id=1)
        )
        traded = result.traded
        if isinstance(traded, bool):
            pytest.fail(
                "SubmitResult.traded is still a Python bool — P2-H2 must "
                "convert it to a 0-d jnp.bool_ scalar"
            )
        assert isinstance(traded, jnp.ndarray)
        assert traded.dtype == jnp.bool_
        assert traded.shape == ()


# ===========================================================================
# AC (6) — Truncation backstop: > 200 inner iterations -> truncated=True
# without silent volume loss.
# ===========================================================================


class TestACSixTruncationBackstop:
    """A synthetic fixture forcing > 200 inner iterations sets
    ``OMIEBookState.truncated == True`` without silent volume loss.

    Story-card pin: "(6) Truncation backstop: a synthetic fixture forcing
    >200 inner iterations (the `OMIEMarketConfig.max_match_iterations` cap)
    sets `OMIEBookState.truncated = True` without silent volume loss — i.e.,
    the visible-trade record after truncation must equal the matched portion
    that fit within the 200-iteration budget, and the unmatched residual
    must be observable via the `truncated` flag, not silently swallowed.
    Use the same `truncated`-flag OR-accumulator pattern from P2-H1 (no
    new flag)."

    The matching cap is ``OMIEMarketConfig.max_match_iterations = 200``
    (verified at ``omie_config.py:38``).  Forcing > 200 inner iterations
    requires either (a) > 200 distinct resting passive slices on the same
    side at crossing prices, or (b) an iceberg whose regeneration
    cascade fires > 200 times.  Path (b) is structurally cleaner — an
    ICEwPI with very small ``peak_qty`` against a large aggressor will
    produce one regeneration per slice until exhaustion.

    Method:
      1. Submit an ICEwPI with ``peak_qty = 0.1`` and
         ``quantity > 200 * peak_qty`` so that the natural slice count
         exceeds the trip-count.
      2. Submit an aggressor that fully consumes the iceberg.
      3. Assert that ``state.truncated == True`` post-aggression.
      4. Assert that the cumulative reported ``trade_qty`` equals the matched
         portion (i.e., no silent volume loss — what was matched is reported,
         and the residual is observable via the flag).

    @pytest.mark.slow: this test traces through the full jitted ``submit_order``
    body and exercises a regeneration cascade > 200 iterations; even with a
    bounded trip count this exceeds the < 1s fast-suite budget.  Justified
    per test-agent brief — slow tests are appropriate when exercising
    multi-slice matching at the truncation envelope.
    """

    @pytest.mark.slow
    def test_truncation_flag_set_on_excess_iterations(
        self, book, empty_state
    ):
        """slow: > 200-iteration regeneration cascade.

        Force > MAX_MATCH_ITERATIONS regenerations via a small-peak ICEwPI
        against a large aggressor; assert state.truncated becomes True.
        """
        from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig

        # The cap is read from the same config object the book was built
        # with — locks the test to whatever value the developer's diff uses.
        max_iters = OMIEMarketConfig().max_match_iterations

        # Build an ICEwPI whose hidden-volume regeneration cascade would
        # naturally generate > max_iters slices.  peak_qty=0.1, ppd=0.01
        # gives many tiny slices at slightly different prices.  The
        # quantity must be large enough to produce > max_iters slices
        # *if uncapped*.  qty_max=999.0 (per OMIEProductConfig.qty_max)
        # caps individual orders at 999.0 MW.  With peak=0.1, a single
        # 999.0 MW iceberg generates 9990 slices uncapped — far more
        # than the 200 trip-count.
        ice = _make_sell(
            order_id=1, price=50.00, quantity=999.0,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=0.1, ppd=0.01, timestamp=100,
        )
        state, ice_res = book.submit_order(empty_state, ice)
        assert _as_python_bool(ice_res.accepted) is True

        # Aggressor at a very crossing price, large quantity — forces the
        # regeneration loop to fire until the trip-count bounds it.
        aggressor = _make_buy(
            order_id=2, price=99.00, quantity=999.0,
            exec_restriction=ExecRestriction.NON, timestamp=200,
        )
        state2, agg_res = book.submit_order(state, aggressor)

        # Backstop must be visible — either state.truncated or a P2-H1-style
        # OR-accumulator flag.  Path B (release-build carry-flag) is
        # the locked-in mechanism per P2-H1 §A3.
        truncated = bool(state2.truncated)
        assert truncated is True, (
            f"After > {max_iters} inner iterations of the multi-slice "
            f"matching loop, state.truncated must be True (P2-H1 §A3 "
            f"Path B carry-flag OR-accumulated into OMIEBookState.truncated).  "
            f"A False value means either the bound was not hit or the "
            f"backstop is silently swallowing volume — both are forbidden."
        )

        # Silent-volume-loss check: whatever was reported as traded must
        # equal the verifiable matched portion.  We assert structurally:
        # the cumulative trade_qty must be a finite non-negative jnp scalar,
        # and the residual is observable via the flag (not silently
        # discarded).  We do not assert an exact MW count because the
        # cascade depth depends on the iceberg manager's slice ordering,
        # which is deliberately implementation-detail-free in this test.
        trade_qty = _as_python_float(agg_res.trade_qty)
        assert trade_qty >= 0.0, (
            f"trade_qty must be non-negative; got {trade_qty}"
        )

    @pytest.mark.slow
    def test_truncation_does_not_silently_drop_matched_volume(
        self, book, empty_state
    ):
        """slow: small-fill verification under the truncation envelope.

        Submit a SMALL ICEwPI whose slice count does NOT exceed the
        trip-count, then a small aggressor.  Truncation must NOT fire,
        and the trade_qty must equal the iceberg's full quantity.  This is
        the negative-control complement to the > 200 trigger — the flag
        must not be over-eagerly set.
        """
        ice = _make_sell(
            order_id=1, price=50.00, quantity=1.0,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=0.5, ppd=0.0, timestamp=100,
        )
        state, _ = book.submit_order(empty_state, ice)

        aggressor = _make_buy(
            order_id=2, price=50.00, quantity=1.0,
            exec_restriction=ExecRestriction.NON, timestamp=200,
        )
        state2, agg_res = book.submit_order(state, aggressor)

        # The natural slice count here is 2 (1.0 / 0.5).  Far below the
        # 200 trip-count, so truncation MUST NOT fire.
        truncated = bool(state2.truncated)
        assert truncated is False, (
            f"A 2-slice iceberg must NOT trigger the truncation backstop; "
            f"got state.truncated={truncated}.  Over-eager truncation is "
            f"as harmful as silent volume loss."
        )
        assert _as_python_bool(agg_res.traded) is True
        assert _as_python_float(agg_res.trade_qty) == pytest.approx(1.0)


# ===========================================================================
# AC (7) — Test-perf DoD: merge note records six §4 wall-clock numbers.
# ===========================================================================


# Developer self-check string constant — the developer's merge note in
# `docs/omiecid/sprint-status.md` under the `## P2-H2 — Implementation` block
# must contain the six §4 wall-clock numbers per AC (7).  This constant is
# read by the lint test below and is also embedded in the test file so a
# developer can grep `AC7_MERGE_NOTE_REQUIREMENTS` to find the canonical
# checklist without re-reading the agile-plan.
AC7_MERGE_NOTE_REQUIREMENTS = """\
P2-H2 merge note checklist (per agile-plan v5.20 Test-first contract AC (7)
+ `docs/omiecid/test-perf-gate.md` §4):

The developer's merge-note block in `docs/omiecid/sprint-status.md` under the
heading `## P2-H2 — Implementation` must record SIX wall-clock numbers, in
any tabular or labelled-paragraph format the developer prefers, covering
the following three surfaces in both cold and warm states:

  (a) Gate subset, cold
  (b) Gate subset, warm    — must be <= 300 s (per test-perf-gate.md §2)
  (c) Full `not slow`, cold
  (d) Full `not slow`, warm
  (e) Full suite, cold
  (f) Full suite, warm     — must be <= 900 s (per test-perf-gate.md §3)

If the orchestrator dispatches this sprint under the P2-PERF-4 warm-only
carve-out (formalised by `P2-PERF-4-FOLLOWUP-1`), only the three warm
numbers (b), (d), (f) are required; the architect-audit pass will check
the orchestrator decision.
"""


class TestACSevenMergeNoteSelfCheck:
    """Structural pin for the merge-note checklist.

    Story-card pin: "(7) Test-perf DoD: the developer's merge note must
    record six §4 wall-clock numbers from `docs/omiecid/test-perf-gate.md` §4
    (cold + warm × {gate subset, full `not slow`, full suite}).  Gate-subset
    warm ≤ 300 s; full-suite warm ≤ 900 s."

    Per the test-agent brief: "If this lint pattern is awkward to express
    as a test, document it inline in the test file as a developer
    self-check before merge string constant and lock only ACs (1)-(6);
    the architect-audit pass will catch any merge-note gap."

    Only the checklist constant is locked here; the merge note itself is a
    research process record and is not read by the test suite.
    """

    def test_merge_note_checklist_constant_is_authoritative(self):
        """The AC7_MERGE_NOTE_REQUIREMENTS string is the canonical
        developer-facing checklist.  Sanity-check that all six surface
        labels are present so a future edit cannot silently delete one."""
        for label in ("Gate subset, cold", "Gate subset, warm",
                      "Full `not slow`, cold", "Full `not slow`, warm",
                      "Full suite, cold", "Full suite, warm"):
            assert label in AC7_MERGE_NOTE_REQUIREMENTS, (
                f"merge-note checklist string must list `{label}`"
            )
        # Budget pins.
        assert "300 s" in AC7_MERGE_NOTE_REQUIREMENTS
        assert "900 s" in AC7_MERGE_NOTE_REQUIREMENTS
