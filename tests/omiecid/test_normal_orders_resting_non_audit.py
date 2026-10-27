"""Audit tests for WF-20 · ``OMIE-NON-ONLY-IMPL-1`` — the participant's
resting-maker (``ExecRestriction.NON``) orders and their passive-fill
attribution.

Independent, post-implementation verification pass (orchestrate.md Step 5). This
file is AUTO-TAGGED ``audit`` by ``tests/omiecid/conftest.py`` (the ``*_audit.py``
suffix), so routine gates deselect it with ``-m "not audit"``; it is run on
demand / at architect-audit.

A participant order submitted as a resting ``ExecRestriction.NON`` (maker)
order persists in the carried book; when later background flow crosses it,
the passive leg is attributed to the participant.

Scope (derived independently from the WF-20 card + ``docs/omiecid/non-only-action-
design.md``, NOT from the original locked test file): probe the *public
behaviour and edge cases* of resting NON orders beyond the happy path — sign
correctness both directions, no-fill / no-phantom-fill, multiple coexisting
resting orders, aggressive(cross)/passive separability, and the eager
``compute_agent_pnl`` reconciliation.

Implementation surfaces inspected (read-only, for designing adversarial cases):
  * ``omiecid_simulator/omiecid/omie_session.py`` — ``step_env``'s passive-leg
    attribution (``_extract_passive_fill`` via ``PASS_TID == AGENT_TRADER_ID``;
    folds into inventory + realized cash + the ``agent_trades`` tape; surfaced
    as ``FillReport.passive_value_eur``).

Controlled fixtures: every behavioural scenario builds a SMALL synthetic 13-column
tape in-test (single tape ⇒ ``reset`` is deterministic; the participant's
prices are set against a hand-built two-sided book) so the passive-fill path is
exercised deterministically. ``FillReport.cash_delta_eur`` is the pure
realized (aggressive + passive) EUR, keeping the sign / magnitude assertions
exact.

Device / tiering: CPU-hermetic (``tests/omiecid/conftest.py`` pins CPU). Tests that
construct + step the jitted env (``reset`` / ``step_env`` compiles + rollouts)
carry ``@pytest.mark.slow`` with a one-line justification.

Run:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python3 -m pytest \
        tests/omiecid/test_normal_orders_resting_non_audit.py -v
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import guard — keep the file collectible even on a partial checkout.
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp

    from omiecid_simulator.omiecid.omie_session import (
        AgentOrder,
        Session,
        AGENT_TRADER_ID,
    )
    from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
    from omiecid_simulator.omiecid.order_types import (
        ExecRestriction,
        Side,
        Status,
        Validity,
    )
    from omiecid_simulator.omiecid.settlement import compute_agent_pnl

    _MISSING = False
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - collection-time guard
    _MISSING = True
    _IMPORT_ERROR = repr(exc)

_skip = pytest.mark.skipif(
    _MISSING, reason=f"WF-11 surfaces not importable ({_IMPORT_ERROR})"
)


# ---------------------------------------------------------------------------
# Tape-building helpers (13-column loader layout, tick-encoded).
# ---------------------------------------------------------------------------
_P = OMIEProductConfig()


def _enc_price(p):
    return int(round((p - _P.price_min) / _P.price_tick))


def _enc_qty(q):
    return int(round(q / _P.qty_tick))


def _row(side, price, qty, oid, ts, exec_r=None):
    """One 13-column background-order row (defaults to a resting NON limit)."""
    if exec_r is None:
        exec_r = int(ExecRestriction.NON)
    return [
        0,                       # contract_id
        int(side),               # side (BUY=0 / SELL=1)
        _enc_price(price),       # price int tick
        _enc_qty(qty),           # qty int tick
        int(oid),                # order_id (>0)
        777,                     # anonymous background trader id (!= agent)
        int(ts),                 # timestamp (session-relative seconds > 0)
        int(exec_r),             # exec_restriction
        int(Validity.GFS),       # validity
        0,                       # expiry_time
        int(Status.ACTIVE),      # status
        0,                       # peak_qty int
        0,                       # ppd int
    ]


def _pad_row():
    """Inert pad sentinel (order_id<=0, timestamp<=0 ⇒ no fill, no time advance)."""
    return [0, 0, 0, 0, -1, 0, -1, int(ExecRestriction.NON),
            int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0]


def _build_tape(rows, n_padded=16):
    """A single (1, n_padded, 13) int32 tape — n_tapes==1 ⇒ deterministic reset."""
    rows = list(rows) + [_pad_row() for _ in range(n_padded - len(rows))]
    return np.asarray([rows], dtype=np.int32)


def _make_env(tape, orders_per_step=2, **overrides):
    """Session over a controlled tape. ``FillReport.cash_delta_eur`` is the
    step's realized EUR, aggressive + passive."""
    config = dict(
        tapes=tape,
        orders_per_step=orders_per_step,
        tape_seconds_span=100.0,
    )
    config.update(overrides)
    return Session(config)


# Participant orders (side, price EUR/MWh, quantity MW) against the 45/55
# opener below: every order is a resting ``ExecRestriction.NON`` GFS order of
# one 0.1-MW qty tick. ``None`` submits nothing.
NOOP = None
NON_BUY_MID = (0, 50.0, 0.1)        # BUY at the 45/55 mid, rests as inside bid
NON_SELL_MID = (1, 50.0, 0.1)       # SELL at the 45/55 mid, rests as inside ask
NON_SELL_AT_55 = (1, 55.0, 0.1)     # SELL joining the 55 best ask
# A marketable NON BUY crossing the opposite touch (fills on submit, rests any
# residual): the participant's aggressive-fill route.
NON_BUY_CROSS_55 = (0, 55.0, 0.1)   # crosses the opener's 55 ask
NON_BUY_CROSS_52 = (0, 52.0, 0.1)   # crosses a fresh 52 ask


def _agent_order(spec):
    if spec is None:
        return AgentOrder.none()
    side, price, qty = spec
    return AgentOrder.create(side=side, price=price, quantity=qty)


def _key():
    return jax.random.PRNGKey(0)


def _step(env, state, spec):
    """One step; returns ``(state, FillReport, done)``."""
    return env.step_env(_key(), state, _agent_order(spec))


# A two-sided-book opener consumed by most scenarios: a resting bid@45 and
# ask@55 (mid 50) so a NON order at 50 rests as a fresh inside quote.
def _opener():
    return [_row(Side.BUY, 45.0, 10.0, 101, 10),
            _row(Side.SELL, 55.0, 10.0, 102, 20)]


# ===========================================================================
# Passive-fill sign correctness — BOTH directions (audit dimension 1).
# The happy-path locked test covers one side; here we pin both.
# ===========================================================================
@_skip
class TestPassiveFillSignBothDirections:

    @pytest.mark.slow  # slow: env reset + 2 step_env compiles + rollout
    def test_resting_non_sell_fills_positive_cash_and_reduces_inventory(self):
        """A resting NON SELL hit by later background BUY flow: passive realized
        cash > 0 (a sell earns), inventory goes NEGATIVE (short)."""
        tape = _build_tape(
            _opener() + [_row(Side.BUY, 60.0, 5.0, 103, 40, int(ExecRestriction.IOC)),
                         _pad_row()]
        )
        env = _make_env(tape)
        state = env.reset(_key())
        # step 0: place a resting NON SELL at the mid (50), rests as inside ask.
        state, rep0, _d0 = _step(env, state, NON_SELL_MID)
        assert float(rep0.passive_value_eur) == 0.0  # not hit yet
        # step 1: background BUY@60 crosses the resting SELL ⇒ passive fill.
        state, rep1, _d1 = _step(env, state, NOOP)
        passive = float(rep1.passive_value_eur)
        inv = float(state.agent_inventory[0])
        assert passive > 0.0, "a passive SELL fill must earn positive realized PnL"
        assert inv < 0.0, "a passive SELL fill must reduce (short) inventory"
        # The whole realized cash on this no-order step IS the passive term.
        assert float(rep1.cash_delta_eur) == pytest.approx(passive, abs=1e-4)

    @pytest.mark.slow  # slow: env reset + 2 step_env compiles + rollout
    def test_resting_non_buy_fills_negative_cash_and_raises_inventory(self):
        """The MIRROR of the happy path: a resting NON BUY hit by later
        background SELL flow: passive realized cash < 0 (a buy costs),
        inventory goes POSITIVE (long)."""
        tape = _build_tape(
            _opener() + [_row(Side.SELL, 40.0, 5.0, 103, 40, int(ExecRestriction.IOC)),
                         _pad_row()]
        )
        env = _make_env(tape)
        state = env.reset(_key())
        # step 0: place a resting NON BUY at the mid (50), rests as inside bid.
        state, rep0, _d0 = _step(env, state, NON_BUY_MID)
        assert float(rep0.passive_value_eur) == 0.0
        # step 1: background SELL@40 crosses the resting BUY ⇒ passive fill.
        state, rep1, _d1 = _step(env, state, NOOP)
        passive = float(rep1.passive_value_eur)
        inv = float(state.agent_inventory[0])
        assert passive < 0.0, "a passive BUY fill must cost (negative realized PnL)"
        assert inv > 0.0, "a passive BUY fill must raise (long) inventory"
        assert float(rep1.cash_delta_eur) == pytest.approx(passive, abs=1e-4)


# ===========================================================================
# Passive-fill no-op cases — never fills / no phantom fills (audit dimension 2).
# ===========================================================================
@_skip
class TestPassiveFillNeverFills:

    @pytest.mark.slow  # slow: env reset + rollout to GCT
    def test_unhit_resting_non_contributes_zero_and_leaves_no_tape_row(self):
        """A resting NON SELL that no background order ever crosses contributes
        EXACTLY zero passive cash + zero inventory across the whole episode and
        writes NO passive ``agent_trades`` row (no phantom fill)."""
        # Background BUY@48 never reaches the SELL resting near the ask (~50/55).
        tape = _build_tape(
            _opener() + [_row(Side.BUY, 48.0, 5.0, 103, 40), _pad_row()]
        )
        env = _make_env(tape)
        state = env.reset(_key())
        # step 0: resting NON SELL at the own-side touch — parks on the 55 ask.
        state, report, done = _step(env, state, NON_SELL_AT_55)
        total_passive = float(report.passive_value_eur)
        for _ in range(env.steps_per_episode):
            if bool(done):
                break
            state, report, done = _step(env, state, NOOP)
            total_passive += float(report.passive_value_eur)
        assert total_passive == 0.0, "an unhit resting NON must earn nothing"
        assert float(state.agent_inventory[0]) == 0.0, (
            "an unhit resting NON must not move inventory"
        )
        # (steps, 2, 6) two-slot tape (WF-11-FOLLOWUP-1: slot 0 aggressive, slot 1
        # passive) — flatten to (-1, 6) BEFORE column-indexing so `at[:, 2]` is the
        # qty_tick column across BOTH legs, not the size-2 slot axis.
        at = np.asarray(state.agent_trades).reshape(-1, 6)
        n_fill_rows = int((at[:, 2] != 0).sum())  # col 2 == qty_tick
        assert n_fill_rows == 0, (
            "an unhit resting NON must leave NO passive agent_trades row "
            "(no phantom fill)"
        )

    @pytest.mark.slow  # slow: env reset + no-order rollout on the default tape
    def test_no_order_rollout_has_zero_passive_every_step(self):
        """No-maker-leg guard: on the default synthetic tape a participant that
        never submits an order can never be the passive/maker leg of a background
        trade — ``FillReport.passive_value_eur`` is EXACTLY 0.0 on every step. (A
        submitted NON order may rest its residual and later fill passively, so
        the no-order session is the clean zero-passive baseline.)"""
        env = Session.from_test_config()
        state = env.reset(_key())
        done = False
        for _ in range(env.steps_per_episode):
            if bool(done):
                break
            state, report, done = _step(env, state, NOOP)
            assert float(report.passive_value_eur) == 0.0, (
                "a session with no orders must never produce a passive fill"
            )


# ===========================================================================
# Multiple coexisting resting orders (audit dimension: unique per-step id,
# constant AGENT_TRADER_ID ⇒ joint attribution).
# ===========================================================================
@_skip
class TestMultipleRestingOrdersJointAttribution:

    @pytest.mark.slow  # slow: env reset + 3 step rollout
    def test_two_resting_sells_from_different_steps_fill_jointly(self):
        """Two resting NON SELLs placed on DIFFERENT steps (distinct per-step
        order ids) survive together in the carried book and are BOTH attributed
        to the agent when a single later background BUY sweeps them — because they
        share the constant ``AGENT_TRADER_ID`` (design §2)."""
        tape = _build_tape(
            _opener()
            + [_pad_row(), _pad_row()]                       # step 1 quiet
            + [_row(Side.BUY, 60.0, 50.0, 103, 40, int(ExecRestriction.IOC)),
               _pad_row()]                                   # step 2 big sweep
        )
        env = _make_env(tape)
        state = env.reset(_key())
        # step 0: first resting SELL (at the mid, 50).
        state, _rep, _d = _step(env, state, NON_SELL_MID)
        # step 1: second resting SELL joining the own-side touch (the agent's
        # own 50 ask) — coexists.
        state, _rep, _d = _step(env, state, NON_SELL_MID)
        # aggregated visible ask volume == both rests
        q_ask_two = float(env.top_of_book(state).q_ask)
        # step 2: one background BUY sweeps BOTH agent SELLs.
        state, rep2, _d2 = _step(env, state, NOOP)
        passive = float(rep2.passive_value_eur)
        inv = float(state.agent_inventory[0])
        assert q_ask_two > 0.0, "both resting SELLs must be visible before the sweep"
        assert passive > 0.0
        # Both 0.1-MW SELLs filled ⇒ combined short of ~0.2 MW.
        assert inv == pytest.approx(-0.2, abs=1e-4), (
            f"joint attribution of two resting SELLs expected inv ~= -0.2, got {inv}"
        )
        assert float(rep2.cash_delta_eur) == pytest.approx(passive, abs=1e-4)


# ===========================================================================
# Aggressive/passive separability + no cross-contamination (audit dimension 3).
# ===========================================================================
@_skip
class TestAggressivePassiveSeparability:

    @pytest.mark.slow  # slow: env reset + 2 step rollout
    def test_pure_aggressive_step_has_zero_passive_value(self):
        """An aggressive NON fill (own-submit, marketable on submit) with NO
        prior resting order surfaces its PnL in ``cash_delta_eur`` while
        ``passive_value_eur`` stays EXACTLY 0.0 — the two attribution channels
        do not cross. The agent's NON BUY crosses the resting ask in full
        (qty << resting volume), so no residual rests to fill passively."""
        # Book with a resting ask the agent's NON BUY CROSS can cross on step 0.
        tape = _build_tape(_opener())
        env = _make_env(tape)
        state = env.reset(_key())
        state, rep0, _d0 = _step(env, state, NON_BUY_CROSS_55)
        assert float(rep0.passive_value_eur) == 0.0, (
            "a pure aggressive fill must not register as a passive fill"
        )
        # The realized term is the aggressive PnL (a BUY ⇒ <= 0), no passive leak.
        assert float(rep0.cash_delta_eur) == pytest.approx(
            float(state.agent_cash[0]), abs=1e-4
        )

    @pytest.mark.slow  # slow: env reset + 2 step rollout, simultaneous fills
    def test_simultaneous_aggressive_and_passive_separate_correctly(self):
        """A single step carrying BOTH a passive fill (a prior resting SELL hit by
        background flow) AND an aggressive fill (the agent's own NON BUY crossing
        the ask this step): ``cash_delta_eur == aggressive + passive``, the passive
        component is isolated in ``passive_value_eur``, and inventory folds BOTH
        signed legs (no double-count, no cross-contamination)."""
        # step0 opener rests SELL@50; step1 background BUY@60 hits it passively
        # AND refills a fresh ask@52 the agent's own NON BUY@52 then crosses.
        tape = _build_tape(
            _opener()
            + [_row(Side.BUY, 60.0, 5.0, 103, 40, int(ExecRestriction.IOC)),
               _row(Side.SELL, 52.0, 10.0, 104, 45)]
        )
        env = _make_env(tape)
        state = env.reset(_key())
        state, _rep0, _d0 = _step(env, state, NON_SELL_MID)  # rest SELL
        inv_before = float(state.agent_inventory[0])
        cash_before = float(state.agent_cash[0])
        state, rep1, _d1 = _step(env, state, NON_BUY_CROSS_52)
        passive = float(rep1.passive_value_eur)
        realized = float(rep1.cash_delta_eur)
        inv_after = float(state.agent_inventory[0])
        assert passive > 0.0, "the prior resting SELL must fill passively this step"
        # aggressive = realized - passive; the agent's crossing NON BUY (a cost).
        aggressive = realized - passive
        assert aggressive < 0.0, "the own crossing NON BUY leg must be a cost (negative)"
        # the cash accumulator moves by realized == aggressive + passive.
        assert float(state.agent_cash[0]) - cash_before == pytest.approx(
            realized, abs=1e-4)
        # inventory folds BOTH legs: passive SELL (-) + aggressive BUY (+).
        # Equal 0.1-MW legs ⇒ they net back toward the pre-step inventory.
        assert inv_after == pytest.approx(inv_before, abs=1e-4), (
            "a matched passive-SELL + aggressive-BUY of equal qty must net to no "
            f"inventory change (before={inv_before}, after={inv_after})"
        )

    @pytest.mark.slow  # slow: env reset + 2 step rollout
    def test_report_fields_present_and_finite(self):
        """The separability contract: ``FillReport.passive_value_eur`` is present,
        a 0-d float32, and finite; ``cash_delta_eur`` and ``passive_value_eur``
        are distinct fields."""
        tape = _build_tape(
            _opener() + [_row(Side.BUY, 60.0, 5.0, 103, 40, int(ExecRestriction.IOC)),
                         _pad_row()]
        )
        env = _make_env(tape)
        state = env.reset(_key())
        state, report, _d = _step(env, state, NON_SELL_MID)
        assert "passive_value_eur" in report._fields
        pfr = jnp.asarray(report.passive_value_eur)
        assert pfr.shape == (), "passive_value_eur must be a 0-d scalar"
        assert pfr.dtype == jnp.float32
        assert bool(jnp.isfinite(pfr))
        # Distinct, separable channels (design §4b(2)).
        for k in ("cash_delta_eur", "passive_value_eur"):
            assert k in report._fields, f"FillReport must surface the separable field {k!r}"


# ===========================================================================
# Eager compute_agent_pnl reconciliation (audit dimension: the trade tape
# reconciles with the total realized cash).
# ===========================================================================
@_skip
class TestEagerPnlReconciliation:

    def _rollout(self, tape, orders):
        env = _make_env(tape)
        state = env.reset(_key())
        sum_realized = 0.0
        done = False
        for spec in orders:
            if bool(done):
                break
            state, report, done = _step(env, state, spec)
            sum_realized += float(report.cash_delta_eur)
        # (steps, 2, 6) two-slot tape (WF-11-FOLLOWUP-1) → flat (-1, 6) that
        # compute_agent_pnl consumes (inert slots are all-zero rows contributing
        # zero).
        flat = np.asarray(state.agent_trades).reshape(-1, 6)
        pnl = float(
            compute_agent_pnl(jnp.asarray(flat),
                              jnp.int32(AGENT_TRADER_ID))
        )
        return sum_realized, pnl, flat

    @pytest.mark.slow  # slow: env reset + rollout + eager x64 pnl
    def test_passive_only_episode_reconciles_with_compute_agent_pnl(self):
        """The core WF-11 reconciliation claim (design §4b(3)): on a passive fill
        that lands on a quiet (no-order) step, the eager ``compute_agent_pnl``
        over the ``agent_trades`` tape (which carries the passive-leg row) equals
        the summed ``cash_delta_eur`` within float32 tolerance."""
        tape = _build_tape(
            _opener() + [_row(Side.BUY, 60.0, 5.0, 103, 40, int(ExecRestriction.IOC)),
                         _pad_row()]
        )
        sum_realized, pnl, _at = self._rollout(tape, [NON_SELL_MID, NOOP])
        assert sum_realized == pytest.approx(pnl, abs=1e-3, rel=1e-4), (
            f"passive-only reconciliation broke: sum_realized={sum_realized}, "
            f"compute_agent_pnl={pnl}"
        )
        assert pnl > 0.0

    @pytest.mark.slow  # slow: env reset + rollout, simultaneous-fill tape probe
    def test_simultaneous_fill_step_tape_records_both_legs_and_reconciles(self):
        """RESOLVED behaviour of the WF-11-FOLLOWUP-1 two-slot tape.

        On a step carrying BOTH a passive fill and an aggressive own-submit fill,
        the ``agent_trades`` tape is now ``(steps, 2, 6)`` — slot 0 the aggressive
        leg, slot 1 the passive leg — so a simultaneous-fill step records BOTH
        legs instead of dropping the aggressive one. (The earlier v1 one-row-per-
        step tape kept only the passive row and DIVERGED from the folded realized
        reward; this test previously characterised that limitation. The two-slot
        widening resolved it, exactly as design §4b(3) asks — the tape now carries
        the passive rows 'the same way the aggressive fills are'.)

        Consequence — the eager ``compute_agent_pnl`` over the FLATTENED
        ``(-1, 6)`` tape now RECONCILES with ``cash_delta_eur`` even when the
        aggressive and passive fills COINCIDE on one step, because both legs are on
        the tape. Column-indexing MUST reshape(-1, 6) first; indexing the raw 3-D
        tape hits the size-2 slot axis, not the qty column.
        """
        tape = _build_tape(
            _opener()
            + [_row(Side.BUY, 60.0, 5.0, 103, 40, int(ExecRestriction.IOC)),
               _row(Side.SELL, 52.0, 10.0, 104, 45)]
        )
        # `at` is the flat (-1, 6) tape from `_rollout` (both slots of the
        # (steps, 2, 6) two-slot tape flattened); col 2 == qty_tick.
        sum_realized, pnl, at = self._rollout(
            tape, [NON_SELL_MID, NON_BUY_CROSS_52]
        )
        fill_rows = at[at[:, 2] != 0]  # non-inert rows across BOTH legs
        assert fill_rows.shape[0] == 2, (
            "the two-slot tape records BOTH legs of the simultaneous fill "
            "(aggressive own-submit + passive resting) — one row per slot"
        )
        # One row is the AGGRESSIVE leg (agent BUYER, col 3) and one is the PASSIVE
        # leg (agent SELLER, col 4); both are the agent's fills.
        buyers = {int(r[3]) for r in fill_rows}
        sellers = {int(r[4]) for r in fill_rows}
        assert int(AGENT_TRADER_ID) in buyers, (
            "the aggressive crossing NON BUY leg (agent as buyer) must be on the tape"
        )
        assert int(AGENT_TRADER_ID) in sellers, (
            "the passive resting-NON-SELL leg (agent as seller) must be on the tape"
        )
        # RECONCILIATION: with both legs on the tape the eager PnL now EQUALS the
        # folded realized cash on a simultaneous-fill step (float32 envelope).
        assert abs(pnl) > 1e-3, "episode produced no fills — cross-check vacuous"
        assert sum_realized == pytest.approx(pnl, abs=1e-3, rel=1e-4), (
            "with both legs recorded the eager tape PnL must reconcile with the "
            f"folded realized cash (sum_realized={sum_realized}, pnl={pnl})"
        )
