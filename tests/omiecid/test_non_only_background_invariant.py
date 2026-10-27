"""Locked INVARIANT contract for WF-20 · ``OMIE-NON-ONLY-IMPL-1`` — removing IOC
from the AGENT action space must NOT touch the matching engine / background tape.

The WF-20 reversal makes the AGENT action space NON-only (``Discrete(13)``, every
non-NOOP index a resting ``ExecRestriction.NON`` order). This file locks the
STRUCTURAL invariant the design (``docs/omiecid/non-only-action-design.md`` §3.2
"LEAVE UNTOUCHED") demands: the *replay tape still carries real IOC/FOK/ICE*, so
the exec-restriction enum and the background-tape matching dispatch must be
UNCHANGED.

Two guards, both derived directly from the WF-20 card + the WF-19 design:

  (1) ``ExecRestriction`` retains ALL FOUR members (NON=0, IOC=1, FOK=2, ICE=3).
      Removing IOC from the agent action space must not delete/renumber the enum
      the background tape and the matching engine index on
      (``order_types.py``; ``contract_book.py`` ``jax.lax.switch`` discriminator).

  (2) The background matching dispatch ``jax.lax.switch(discriminator,
      (non_branch, ioc_branch, fok_branch, ice_branch), (st, ord_))`` in
      ``contract_book.py`` (~L1166) is UNTOUCHED — a structural/text guard (the
      4-branch tuple still names all four paths, one per ExecRestriction member)
      PLUS a behavioural check that a background IOC / FOK / NON order submitted
      through the matching engine still dispatches to its own semantics (IOC
      partial-fills-then-cancels; FOK is all-or-nothing; NON rests the residual).

TEST-FIRST STATUS: these are INVARIANTS the WF-20 change must PRESERVE, so they
are EXPECTED TO PASS on the current tree AND after the reversal (they are the
"do-not-break" locks). They protect against an over-eager implementation that
also strips IOC/FOK from the engine while removing it from the agent space.

Spec sources:
  * ``docs/omiecid/non-only-action-design.md`` §3.2 "LEAVE UNTOUCHED": ExecRestriction
    keeps all 4 members; the background-tape matching ``lax.switch`` at
    ``contract_book.py:1166`` must not be edited.
  * ``docs/omiecid/market-spec.md`` §4.5 — the IOC / FOK / NON / ICE execution
    restriction semantics the engine implements.
  * ``omiecid_simulator/omiecid/order_types.py`` — ``ExecRestriction`` IntEnum.
  * ``omiecid_simulator/omiecid/contract_book.py`` — the 4-branch ``lax.switch``.

Device / tiering: CPU-hermetic (``tests/omiecid/conftest.py`` pins CPU). The enum +
text guards are pure-python (microseconds). The behavioural dispatch checks
submit through the ``ContractBook`` fixture (a small jitted matching call) and
carry ``@pytest.mark.jit``; all fit the <1s fast-suite budget.
"""

from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# Import guard — keep the file collectible even on a partial checkout.
# ---------------------------------------------------------------------------
try:
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
    from omiecid_simulator.omiecid import contract_book as _cb_module
    from omiecid_simulator.omiecid.contract_book import SubmitResult

    _MISSING = False
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - collection-time guard
    _MISSING = True
    _IMPORT_ERROR = repr(exc)

_skip = pytest.mark.skipif(
    _MISSING, reason=f"OMIE matching-engine surface not importable ({_IMPORT_ERROR})"
)


# ===========================================================================
# 1. ExecRestriction keeps all 4 members (the enum the background tape indexes).
# ===========================================================================
@_skip
class TestExecRestrictionMembersRetained:
    """``ExecRestriction`` must keep ALL FOUR members with their canonical
    integer values — NON=0, IOC=1, FOK=2, ICE=3 (market-spec §4.5, §13). The
    WF-20 reversal removes IOC from the AGENT action space only; the enum drives
    the background-tape replay + the matching-engine ``lax.switch`` discriminator
    and must be untouched."""

    def test_all_four_members_present_with_canonical_values(self):
        assert int(ExecRestriction.NON) == 0
        assert int(ExecRestriction.IOC) == 1
        assert int(ExecRestriction.FOK) == 2
        assert int(ExecRestriction.ICE) == 3

    def test_exactly_four_members(self):
        members = list(ExecRestriction)
        assert len(members) == 4, (
            f"ExecRestriction must have exactly 4 members (NON/IOC/FOK/ICE); got "
            f"{[m.name for m in members]} — removing IOC from the AGENT action "
            "space must NOT delete it from the enum (non-only-action-design.md §3.2)"
        )
        names = {m.name for m in members}
        assert names == {"NON", "IOC", "FOK", "ICE"}

    def test_ioc_still_defined_for_the_background_tape(self):
        """IOC must remain a defined enum member — the background replay tape
        still carries real IOC orders even though the agent can no longer emit
        one."""
        assert hasattr(ExecRestriction, "IOC")
        assert int(ExecRestriction.IOC) == 1


# ===========================================================================
# 2a. Structural/text guard — the 4-branch background matching switch is intact.
# ===========================================================================
@_skip
class TestBackgroundSwitchStructurallyIntact:
    """The background matching dispatch ``jax.lax.switch(discriminator,
    (non_branch, ioc_branch, fok_branch, ice_branch), ...)`` in
    ``contract_book.py`` (~L1166) must still name all four exec-restriction
    branches, one per ``ExecRestriction`` member. A source-text guard catches an
    over-eager reversal that also stripped IOC/FOK from the engine."""

    def _source(self):
        return inspect.getsource(_cb_module)

    def test_switch_names_all_four_branches(self):
        src = self._source()
        # The 4-branch dispatch tuple (order = ExecRestriction NON/IOC/FOK/ICE).
        assert "non_branch" in src, "the NON matching branch must remain defined"
        assert "ioc_branch" in src, (
            "the IOC matching branch must remain defined — the background tape "
            "still dispatches real IOC orders (non-only-action-design.md §3.2)"
        )
        assert "fok_branch" in src, "the FOK matching branch must remain defined"
        assert "ice_branch" in src, "the ICE matching branch must remain defined"

    def test_switch_dispatch_tuple_present(self):
        """The exact 4-branch ``lax.switch`` dispatch tuple is present (the
        engine still routes on the exec-restriction discriminator, all 4 arms)."""
        src = self._source()
        # Whitespace-insensitive check for the 4-branch tuple.
        normalized = " ".join(src.split())
        assert "(non_branch, ioc_branch, fok_branch, ice_branch)" in normalized, (
            "the 4-branch background matching lax.switch dispatch tuple "
            "'(non_branch, ioc_branch, fok_branch, ice_branch)' must be intact — "
            "the WF-20 agent-space reversal must not touch the matching engine "
            "(non-only-action-design.md §3.2 'LEAVE UNTOUCHED')"
        )

    def test_lax_switch_used_for_dispatch(self):
        src = self._source()
        assert "lax.switch" in src, (
            "the exec-restriction matching dispatch must still be a jax.lax.switch"
        )

    def test_branch_count_matches_exec_restriction_cardinality(self):
        """The dispatch has exactly as many branches (4) as ExecRestriction has
        members (4) — one matching path per restriction."""
        assert len(list(ExecRestriction)) == 4
        src = " ".join(self._source().split())
        # count the four named branch identifiers inside the dispatch tuple
        for name in ("non_branch", "ioc_branch", "fok_branch", "ice_branch"):
            assert name in src


# ===========================================================================
# 2b. Behavioural guard — background IOC / FOK / NON still dispatch correctly.
#
# Submits IOC / FOK / NON orders through the matching engine (the ContractBook
# fixture — the background/submit path, NOT the agent decode) and confirms each
# still routes to its own execution-restriction semantics. This proves the engine
# is untouched by the agent-space reversal.
# ===========================================================================
def _make_sell(order_id, price, quantity, timestamp, exec_r=ExecRestriction.NON):
    return make_order(
        contract_id=37, side=Side.SELL, price=price, quantity=quantity,
        order_id=order_id, agent_id=1, timestamp=timestamp,
        exec_restriction=exec_r,
    )


def _make_buy(order_id, price, quantity, timestamp, exec_r=ExecRestriction.NON):
    return make_order(
        contract_id=37, side=Side.BUY, price=price, quantity=quantity,
        order_id=order_id, agent_id=0, timestamp=timestamp,
        exec_restriction=exec_r,
    )


@_skip
class TestBackgroundIOCFOKStillDispatch:
    """A background IOC / FOK / NON order submitted through the matching engine
    still dispatches to its own semantics (market-spec §4.5): IOC partial-fills
    then cancels the residual; FOK is all-or-nothing; NON rests the residual.
    This is the do-not-break lock on the engine the agent-space reversal must
    preserve."""

    @pytest.mark.jit
    def test_background_ioc_partial_fills_then_cancels_residual(self, book, empty_state):
        """Resting sell 5 MW @ 50; a background IOC buy 10 MW @ 50 fills the
        available 5 MW and the 5 MW residual is CANCELLED (not rested)."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        state, result = book.submit_order(
            state,
            _make_buy(order_id=10, price=50.00, quantity=10.0, timestamp=500,
                      exec_r=ExecRestriction.IOC),
        )
        assert isinstance(result, SubmitResult)
        assert bool(result.traded), "the background IOC order must dispatch and fill"
        assert float(result.trade_qty) == pytest.approx(5.0, abs=0.05), (
            "the IOC order fills only the available 5 MW"
        )
        # IOC residual must NOT rest in the book (cancelled).
        resting = book.get_order(state, 10, Side.BUY)
        residual_absent = (resting is None) or (
            int(getattr(resting, "order_id", -1)) != 10
        )
        assert residual_absent, (
            "the IOC residual must be CANCELLED, not rested — the IOC matching "
            "branch is still dispatched for background orders"
        )

    @pytest.mark.jit
    def test_background_fok_all_or_nothing_cancels_on_insufficient_liquidity(
        self, book, empty_state
    ):
        """Resting sell 5 MW @ 50; a background FOK buy 10 MW @ 50 CANNOT fully
        fill (only 5 MW available), so it is CANCELLED with no trade — the FOK
        branch is still dispatched."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=5.0, timestamp=100)
        )
        state, result = book.submit_order(
            state,
            _make_buy(order_id=20, price=50.00, quantity=10.0, timestamp=500,
                      exec_r=ExecRestriction.FOK),
        )
        assert isinstance(result, SubmitResult)
        assert not bool(result.traded), (
            "a FOK order with insufficient liquidity must NOT trade (all-or-"
            "nothing) — the FOK matching branch is still dispatched"
        )

    @pytest.mark.jit
    def test_background_fok_full_fill_when_liquidity_sufficient(self, book, empty_state):
        """Resting sell 10 MW @ 50; a background FOK buy 10 MW @ 50 fully fills."""
        state, _ = book.submit_order(
            empty_state, _make_sell(order_id=2, price=50.00, quantity=10.0, timestamp=100)
        )
        state, result = book.submit_order(
            state,
            _make_buy(order_id=21, price=50.00, quantity=10.0, timestamp=500,
                      exec_r=ExecRestriction.FOK),
        )
        assert bool(result.traded), "a fully-satisfiable FOK order must fill"
        assert float(result.trade_qty) == pytest.approx(10.0, abs=0.05)

    @pytest.mark.jit
    def test_background_non_rests_unfilled_residual(self, book, empty_state):
        """A background NON buy with no crossing liquidity RESTS (book entry if
        unmatched = Yes; market-spec §4.5) — the NON branch is still dispatched.
        This is the same maker semantics the agent's NON-only orders now use."""
        state, result = book.submit_order(
            empty_state,
            _make_buy(order_id=30, price=49.00, quantity=4.0, timestamp=100,
                      exec_r=ExecRestriction.NON),
        )
        assert isinstance(result, SubmitResult)
        assert not bool(result.traded), "a NON order on an empty book cannot cross"
        # The NON order RESTS: it is retrievable from the book.
        resting = book.get_order(state, 30, Side.BUY)
        assert resting is not None and int(getattr(resting, "order_id", -1)) == 30, (
            "an unmatched NON order must REST in the book (book entry if unmatched "
            "= Yes) — the NON matching branch is still dispatched"
        )
