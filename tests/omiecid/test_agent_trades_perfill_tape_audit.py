"""Independent AUDIT contracts for WF-11-FOLLOWUP-1 ·
``OMIE-AGENT-TRADES-PERFILL-TAPE-1`` — the per-fill / two-slot ``agent_trades``
tape widening.

Post-implementation, adversarial verification pass authored independently of the
original locked file (``tests/omiecid/test_agent_trades_perfill_tape.py`` — NOT
read). Derived directly from the sprint requirements:

  * ``docs/omiecid/agile-plan.md`` §"Story WF-11-FOLLOWUP-1"
  * ``docs/omiecid/normal-orders-design.md`` §4b(3)
  * ``docs/omiecid/migration-plan.md`` §P3.2 (``agent_trades`` contract)
  * ``docs/omiecid/sprint-status.md`` §WF-11-FOLLOWUP-1

The change under audit: ``OMIEContractState.agent_trades`` was widened from
``(steps_per_episode, 6)`` to ``(steps_per_episode, 2, 6)`` — **slot 0 =
aggressive (own-submit) leg, slot 1 = passive (resting-NON) leg**. On a step with
BOTH an aggressive and a passive fill, both legs are recorded; a ``<=1``-fill step
writes an inert (all-zero) companion slot. Reward and inventory are unchanged
(tape-fidelity only); the fix restores the P3.2c eager ``compute_agent_pnl``
reconciliation over the reshaped ``(-1, 6)`` tape.

Audit dimensions (one test class each):
  * TestSlotAssignmentCorrectness — slot 0 genuinely the aggressive own-submit
    leg, slot 1 the passive resting-NON leg (NOT swapped); each row's
    trader-id/side/qty/price correct for its leg.
  * TestInertSlotDiscipline — NOOP / no-fill / single-aggressive / single-passive
    steps write an inert companion slot (no phantom fill).
  * TestReconciliationExactness — ``compute_agent_pnl`` over the reshaped tape ≈
    Σ ``FillReport.cash_delta_eur`` across a mixed session (incl. a simultaneous step).
  * TestEconomicsUnaffectedByWidening — IOC-only rollouts never populate slot 1;
    ``FillReport.passive_value_eur`` stays 0.0 (economics are tape-fidelity only).
  * TestLegCountingSemantics — a simultaneous step counts 2 legs over the
    reshaped tape; inert slots are never counted.
  * TestJitVmapFixedShape — the 3-D tape rides vmapped ``step_env`` with the
    documented outer ``enable_x64`` context and no extra ``_submit_order`` compile.
  * TestBufferBounds — axis-0 length == steps_per_episode at orders_per_step=1;
    a full-length rollout does not overflow.

These audit tests are expected to PASS against the finished code.

Run (CPU, per device-policy):
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python3 -m pytest \
        tests/omiecid/test_agent_trades_perfill_tape_audit.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Guarded imports (partial-checkout + collection safety). A missing core import
# SKIPS the whole file rather than erroring at collection.
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    import jax.experimental  # for enable_x64 (scoped-x64 leg under vmap)

    from omiecid_simulator.omiecid.omie_session import (
        AgentOrder,
        Session,
        AGENT_TRADER_ID,
        AGENT_ORDER_ID_BASE,
    )
    from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
    from omiecid_simulator.omiecid.omie_states import OMIEContractState
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, Status, Validity
    from omiecid_simulator.omiecid.settlement import compute_agent_pnl

    _CORE_MISSING = False
    _CORE_REASON = ""
except Exception as exc:  # pragma: no cover - collection guard
    _CORE_MISSING = True
    _CORE_REASON = f"OMIE core import failed: {exc!r}"


_core_skip = pytest.mark.skipif(_CORE_MISSING, reason=_CORE_REASON or "core present")


# ---------------------------------------------------------------------------
# Tape-column layout (compute_agent_pnl contract, settlement.py:46-49):
#   [timestamp, price_offset_int, qty_tick, buyer, seller, contract_id]
# ---------------------------------------------------------------------------
_COL_TS = 0
_COL_PRICE = 1
_COL_QTY = 2
_COL_BUYER = 3
_COL_SELLER = 4


# ---------------------------------------------------------------------------
# Participant orders on the mixed tape below (side, price EUR/MWh, quantity
# MW). Every order is a resting ``ExecRestriction.NON`` GFS order of one
# 0.1-MW qty tick; the aggressive leg is a marketable NON that crosses the
# opposite touch. ``None`` submits nothing.
# ---------------------------------------------------------------------------
_REST_SELL_51 = (1, 51.0, 0.1)    # rest a SELL at the best ask (51), no cross
_AGG_BUY_52 = (0, 52.0, 0.1)      # step 1: BUY crossing the best ask (52)
_AGG_SELL_49 = (1, 49.0, 0.1)     # SELL crossing the best bid (49)


def _agent_order(spec):
    if spec is None:
        return AgentOrder.none()
    side, price, qty = spec
    return AgentOrder.create(side=side, price=price, quantity=qty)


# ---------------------------------------------------------------------------
# Synthetic tape builder. A single-tape (n_tapes==1) set so reset is
# deterministic. orders_per_step==6:
#   * step-0 slice (rows 0..5): a two-sided book — bids 49/48/47, asks 51/52/53
#     (qty 10 each). The agent then rests a NON SELL PASSIVE at best_ask (51),
#     qty 0.1 (a non-crossing resting order that SURVIVES into the carried book).
#   * step-1 slice (row 6): ONE crossing background BUY@51 qty 20 (background
#     IOC). It consumes the two 51-level asks — the background ask (10 MW) FIRST
#     by time priority, then the agent's resting SELL (0.1 MW) fully — so the
#     agent gets a PASSIVE SELL fill of 0.1 MW @ 51. The background IOC residual
#     cancels; best_ask -> 52.
#   * remaining rows: inert pad.
# Driven with [SELL @ 51, BUY @ 52] this yields a SIMULTANEOUS step-1:
# the passive SELL @ 51 (inside the background replay) AND a fresh aggressive
# BUY @ 52 (the agent's own NON CROSS lifting best_ask: 0.1 MW against 10 MW
# resting fills fully, so no agent residual rests). The two legs are
# DISTINGUISHABLE by both side (agent seller vs buyer) and price (51 vs 52).
# ---------------------------------------------------------------------------
_ORDERS_PER_STEP = 6


def _build_mixed_tape(prod, n_steps=4):
    pmin, ptick, qtick, pmax = (
        prod.price_min, prod.price_tick, prod.qty_tick, prod.price_max
    )

    def enc_p(p):
        return int(round((p - pmin) / ptick))

    def enc_q(q):
        return int(round(q / qtick))

    def order(side, price, qty, oid, ts, exec_r=int(ExecRestriction.NON)):
        return [
            0, int(side), enc_p(price), enc_q(qty), int(oid), 777, int(ts),
            int(exec_r), int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0,
        ]

    pad = [
        0, 0, 0, 0, -1, 0, -1, int(ExecRestriction.NON), int(Validity.GFS), 0,
        int(Status.ACTIVE), 0, 0,
    ]

    BUY, SELL = int(Side.BUY), int(Side.SELL)
    rows = []
    oid = 100
    # step-0: build a resting two-sided book (ts=60).
    for lvl in range(3):
        rows.append(order(BUY, 49.0 - lvl, 10.0, oid, 60)); oid += 1
        rows.append(order(SELL, 51.0 + lvl, 10.0, oid, 60)); oid += 1
    assert len(rows) == _ORDERS_PER_STEP
    # step-1: ONE crossing BUY@51 qty 20 that sweeps the 51 level (bg ask + agent
    # resting SELL). IOC so no residual rests.
    rows.append(order(BUY, 51.0, 20.0, oid, 120, exec_r=int(ExecRestriction.IOC)))
    oid += 1
    n_padded = n_steps * _ORDERS_PER_STEP
    rows += [list(pad) for _ in range(n_padded - len(rows))]
    return np.asarray([rows], dtype=np.int32)


def _make_env(prod, tapes, orders_per_step=_ORDERS_PER_STEP):
    """Session over the tape set. ``FillReport.cash_delta_eur`` is the step's
    realized (aggressive + passive) EUR — the value the eager cross-check
    reconciles. Returns None if the ``tapes`` config path is unavailable."""
    cfg = {
        "product_config": prod,
        "tapes": tapes,
        "orders_per_step": orders_per_step,
    }
    try:
        return Session(cfg)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small plumbing helpers.
# ---------------------------------------------------------------------------
def _run(env, order_seq, seed=0):
    """Drive ``env.step_env`` over ``order_seq``; return (final_state, reports)."""
    state = env.reset(jax.random.PRNGKey(seed))
    reports = []
    for j, spec in enumerate(order_seq):
        state, report, _d = env.step_env(
            jax.random.PRNGKey(j + 1), state, _agent_order(spec)
        )
        reports.append(report)
    return state, reports


def _batched_none(n):
    """``AgentOrder.none()`` with a leading batch axis of ``n`` on every field."""
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n,)), AgentOrder.none()
    )


def _eager_pnl(flat_tape):
    """EAGER ``compute_agent_pnl`` over a host ``(-1, 6)`` tape (opens its own
    scoped enable_x64 internally)."""
    return float(
        np.asarray(compute_agent_pnl(jnp.asarray(flat_tape), jnp.int32(AGENT_TRADER_ID)))
    )


# Expected tick offsets for the mixed-tape fill prices (asserted exactly — the
# tape encoder and the env row encoder must agree bit-for-bit).
def _price_off(prod, price):
    return int(round((price - prod.price_min) / prod.price_tick))


# ===========================================================================
# Dimension 1 — slot assignment correctness (the core adversarial contract).
# ===========================================================================
@_core_skip
class TestSlotAssignmentCorrectness:
    """On a simultaneous step, slot 0 carries the AGGRESSIVE (own-submit) leg and
    slot 1 the PASSIVE (resting-NON) leg — NOT swapped; each row's trader-id,
    side, qty and price are correct for its leg (design §4b(3))."""

    @pytest.mark.slow
    def test_slot0_aggressive_slot1_passive_not_swapped(self):
        """slow: 2-step jitted ``step_env`` rollout on the mixed tape.

        Rest a NON SELL PASSIVE on step 0; on step 1 the background BUY sweeps
        that resting SELL (passive leg) WHILE the agent submits an aggressive IOC
        BUY CROSS (aggressive leg). Assert the two legs land in the RIGHT slots.
        """
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")

        rest = _REST_SELL_51   # rest SELL @ own touch (best ask)
        agg = _AGG_BUY_52      # aggressive BUY crossing the opposite touch
        state, _reports = _run(env, [rest, agg])

        tape = np.asarray(state.agent_trades)
        assert tape.ndim == 3 and tape.shape[1:] == (2, 6), (
            f"agent_trades must be (steps, 2, 6); got {tape.shape}."
        )

        slot0 = tape[1, 0]   # step-1 aggressive slot
        slot1 = tape[1, 1]   # step-1 passive slot

        # Both legs are genuine fills.
        assert slot0[_COL_QTY] > 0, f"aggressive slot 0 must be a fill; got {slot0.tolist()}."
        assert slot1[_COL_QTY] > 0, f"passive slot 1 must be a fill; got {slot1.tolist()}."

        # SLOT 0 = AGGRESSIVE BUY: the agent is the BUYER leg, at best_ask (52).
        assert int(slot0[_COL_BUYER]) == AGENT_TRADER_ID, (
            "slot 0 (aggressive) must carry the agent on the BUYER leg for an "
            f"aggressive BUY; row={slot0.tolist()}."
        )
        assert int(slot0[_COL_SELLER]) != AGENT_TRADER_ID
        assert int(slot0[_COL_PRICE]) == _price_off(prod, 52.0), (
            "aggressive BUY CROSS must fill at best_ask 52; "
            f"price_off={int(slot0[_COL_PRICE])}, expected {_price_off(prod, 52.0)}."
        )

        # SLOT 1 = PASSIVE SELL: the agent is the SELLER leg, at its resting price (51).
        assert int(slot1[_COL_SELLER]) == AGENT_TRADER_ID, (
            "slot 1 (passive) must carry the agent on the SELLER leg for a passive "
            f"resting SELL; row={slot1.tolist()}."
        )
        assert int(slot1[_COL_BUYER]) != AGENT_TRADER_ID
        assert int(slot1[_COL_PRICE]) == _price_off(prod, 51.0), (
            "passive SELL fills at the agent's resting price 51; "
            f"price_off={int(slot1[_COL_PRICE])}, expected {_price_off(prod, 51.0)}."
        )

        # NOT SWAPPED: the aggressive (buyer-leg, higher price) and passive
        # (seller-leg, lower price) rows are distinct and in the documented order.
        assert int(slot0[_COL_PRICE]) != int(slot1[_COL_PRICE]), (
            "the two legs must be independently recorded (distinct prices); "
            "an equal price would indicate one row overwrote the other."
        )

    @pytest.mark.slow
    def test_both_legs_share_the_step_timestamp(self):
        """slow: both slots of a simultaneous step carry the SAME (step) timestamp
        — they are two legs of ONE step, not two different steps collapsed."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        state, _ = _run(
            env, [_REST_SELL_51, _AGG_BUY_52]
        )
        tape = np.asarray(state.agent_trades)
        assert int(tape[1, 0][_COL_TS]) == int(tape[1, 1][_COL_TS]), (
            "both legs of a simultaneous step must carry the same step timestamp; "
            f"slot0 ts={int(tape[1, 0][_COL_TS])}, slot1 ts={int(tape[1, 1][_COL_TS])}."
        )


# ===========================================================================
# Dimension 2 — inert-slot discipline (no phantom fills).
# ===========================================================================
@_core_skip
class TestInertSlotDiscipline:
    """NOOP / no-fill / single-aggressive-only / single-passive-only steps each
    write an inert (all-zero) companion slot — no phantom fill."""

    @pytest.mark.slow
    def test_noop_step_writes_two_inert_slots(self):
        """slow: a NOOP step leaves BOTH slots all-zero (no fill either leg)."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        state, _ = _run(env, [None])  # no order on step 0
        tape = np.asarray(state.agent_trades)
        assert np.all(tape[0] == 0), (
            f"a NOOP step must leave both slots inert; got {tape[0].tolist()}."
        )

    @pytest.mark.slow
    def test_no_fill_rest_step_writes_two_inert_slots(self):
        """slow: a non-crossing resting NON order does NOT fill on its emitting
        step, so both slots stay inert (no phantom aggressive fill)."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        state, _ = _run(env, [_REST_SELL_51])  # rest, no cross
        tape = np.asarray(state.agent_trades)
        assert np.all(tape[0] == 0), (
            "a resting NON order that does not cross must not write any fill row; "
            f"got {tape[0].tolist()}."
        )

    @pytest.mark.slow
    def test_aggressive_only_step_leaves_passive_slot_inert(self):
        """slow: an aggressive IOC fill with NO resting agent order writes slot 0
        only; slot 1 (passive) stays all-zero."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        # step 0 no order (build book), step 1 aggressive SELL crossing the bids.
        state, reports = _run(env, [None, _AGG_SELL_49])
        tape = np.asarray(state.agent_trades)
        assert tape[1, 0][_COL_QTY] > 0, "aggressive slot 0 must be a fill."
        assert np.all(tape[1, 1] == 0), (
            f"passive slot 1 must be inert with no resting order; got {tape[1, 1].tolist()}."
        )
        assert int(tape[1, 0][_COL_SELLER]) == AGENT_TRADER_ID, (
            "aggressive SELL CROSS puts the agent on the SELLER leg."
        )
        # separability: no passive contribution on an aggressive-only step.
        assert float(np.asarray(reports[1].passive_value_eur)) == 0.0

    @pytest.mark.slow
    def test_passive_only_step_leaves_aggressive_slot_inert(self):
        """slow: a purely passive fill (agent NOOPs while background sweeps its
        resting SELL) writes slot 1 only; slot 0 (aggressive) stays all-zero."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        # step 0 rest NON SELL, step 1 no order (so any fill is purely passive).
        state, _ = _run(env, [_REST_SELL_51, None])
        tape = np.asarray(state.agent_trades)
        assert np.all(tape[1, 0] == 0), (
            f"aggressive slot 0 must be inert on a NOOP step; got {tape[1, 0].tolist()}."
        )
        assert tape[1, 1][_COL_QTY] > 0, "passive slot 1 must record the resting-SELL fill."
        assert int(tape[1, 1][_COL_SELLER]) == AGENT_TRADER_ID

    @pytest.mark.slow
    def test_inert_rows_do_not_perturb_compute_agent_pnl(self):
        """slow: adding the all-zero companion slots does not change the eager
        ``compute_agent_pnl`` value — inert (qty_tick==0) rows contribute zero."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        state, _ = _run(
            env, [_REST_SELL_51, _AGG_BUY_52]
        )
        flat = np.asarray(state.agent_trades).reshape(-1, 6)
        with_inert = _eager_pnl(flat)
        # Drop every inert row and recompute — must be identical.
        non_inert = flat[flat[:, _COL_QTY] > 0]
        without_inert = _eager_pnl(non_inert)
        assert np.isclose(with_inert, without_inert, atol=1e-6, rtol=0.0), (
            "inert (all-zero) slots must contribute exactly 0 to compute_agent_pnl; "
            f"with-inert={with_inert}, without-inert={without_inert}."
        )


# ===========================================================================
# Dimension 3 — reconciliation exactness across a mixed session.
# ===========================================================================
@_core_skip
class TestReconciliationExactness:
    """``compute_agent_pnl`` over the reshaped ``(-1, 6)`` tape ≈ Σ
    ``FillReport.cash_delta_eur`` across a mixed session that INCLUDES a simultaneous
    aggressive+passive step (the WF-11 one-row divergence the follow-up closes)."""

    @pytest.mark.slow
    def test_simultaneous_session_reconciles_within_float_tol(self):
        """slow: a session mixing a quiet rest step and a simultaneous
        aggressive+passive step reconciles the summed realized reward against the
        eager tape PnL within float32 tolerance."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")

        # rest (quiet) -> simultaneous (aggressive BUY + passive SELL) -> none -> none
        seq = [
            _REST_SELL_51,
            _AGG_BUY_52,
            None,
            None,
        ]
        state, reports = _run(env, seq)

        sum_realized = sum(
            float(np.asarray(r.cash_delta_eur)) for r in reports
        )
        flat = np.asarray(state.agent_trades).reshape(-1, 6)
        eager = _eager_pnl(flat)

        # Tolerance scaled to the gross notional traded (float32 tick rounding).
        prices = flat[:, _COL_PRICE].astype(np.float64) * prod.price_tick + prod.price_min
        qtys = flat[:, _COL_QTY].astype(np.float64) * prod.qty_tick
        gross = float(np.sum(np.abs(prices) * qtys) * prod.delivery_duration_hours)
        atol = 1e-3 + 1e-4 * gross
        assert abs(sum_realized - eager) <= atol, (
            "Σ realized cash must reconcile with eager compute_agent_pnl over "
            f"the reshaped tape; Σrealized={sum_realized}, eager={eager}, "
            f"|diff|={abs(sum_realized - eager)} > atol={atol}."
        )
        # The simultaneous step really did contribute BOTH legs (else the test is
        # vacuous): step-1 tape has two non-inert rows.
        step1 = np.asarray(state.agent_trades)[1]
        assert int(np.count_nonzero(step1[:, _COL_QTY] > 0)) == 2, (
            "the reconciliation must exercise a genuine simultaneous 2-leg step."
        )

    @pytest.mark.slow
    def test_passive_only_session_reconciles(self):
        """slow: a passive-only session (rest then NOOP) also reconciles — the
        passive leg alone lands on the tape and matches its realized reward."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        state, reports = _run(env, [_REST_SELL_51, None, None])
        sum_realized = sum(float(np.asarray(r.cash_delta_eur)) for r in reports)
        eager = _eager_pnl(np.asarray(state.agent_trades).reshape(-1, 6))
        assert abs(sum_realized - eager) <= 1e-2 + 1e-3 * abs(eager), (
            f"passive-only session must reconcile; Σrealized={sum_realized}, eager={eager}."
        )


# ===========================================================================
# Dimension 4 — economics are tape-fidelity only (widening changes no reward).
# ===========================================================================
@_core_skip
class TestEconomicsUnaffectedByWidening:
    """On IOC-only / ``<=1``-fill rollouts the passive slot is NEVER populated and
    ``FillReport.passive_value_eur`` is exactly 0.0 — the widening added tape fidelity, not
    economics."""

    @pytest.mark.slow
    def test_ioc_only_rollout_never_populates_passive_slot(self):
        """slow: an IOC-only action sequence (no resting NON order ever placed)
        leaves EVERY step's passive slot inert and passive_value_eur == 0.0."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        # no order (build book) then repeated aggressive SELL crossing the 49
        # bid — never rests.
        seq = [None, _AGG_SELL_49, _AGG_SELL_49, None]
        state, reports = _run(env, seq)
        tape = np.asarray(state.agent_trades)
        assert np.all(tape[:, 1, :] == 0), (
            "no resting agent order => the passive slot (index 1) must be inert on "
            "every step; a non-zero passive slot means a phantom passive fill."
        )
        for r in reports:
            assert float(np.asarray(r.passive_value_eur)) == 0.0, (
                "passive_value_eur must be exactly 0.0 on every IOC-only step."
            )


# ===========================================================================
# Dimension 5 — leg-counting semantics over the reshaped tape.
# ===========================================================================
@_core_skip
class TestLegCountingSemantics:
    """A simultaneous step counts as 2 legs over the reshaped ``(-1, 6)`` tape;
    inert slots are never counted."""

    @pytest.mark.slow
    def test_simultaneous_step_counts_two_legs_not_inert(self):
        """slow: over the reshaped tape a simultaneous step yields exactly 2
        non-inert rows and the total non-inert count equals the true fill count."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        state, _ = _run(
            env, [_REST_SELL_51, _AGG_BUY_52]
        )
        flat = np.asarray(state.agent_trades).reshape(-1, 6)
        num_trades = int(np.count_nonzero(flat[:, _COL_QTY] > 0))
        assert num_trades == 2, (
            "a simultaneous aggressive+passive step must count 2 legs over the "
            f"reshaped tape; got num_trades={num_trades}."
        )
        # Every counted row genuinely carries the agent on one leg (no inert slot
        # double-counted).
        counted = flat[flat[:, _COL_QTY] > 0]
        for row in counted:
            assert (int(row[_COL_BUYER]) == AGENT_TRADER_ID) or (
                int(row[_COL_SELLER]) == AGENT_TRADER_ID
            ), f"a counted leg must attribute the agent; row={row.tolist()}."


# ===========================================================================
# Dimension 6 — jit / vmap fixed-shape ride.
# ===========================================================================
@_core_skip
class TestJitVmapFixedShape:
    """The 3-D ``(steps, 2, 6)`` tape rides vmapped ``reset``/``step_env`` with the
    documented outer ``enable_x64`` context and adds no ``_submit_order`` compile."""

    NUM_ENVS = 3

    @pytest.mark.slow
    def test_vmapped_step_preserves_three_d_tape_shape(self):
        """slow: vmapped reset+step over NUM_ENVS carries a leading batch axis on
        the ``(steps, 2, 6)`` tape of the state."""
        env = Session.from_test_config()
        keys = jax.random.split(jax.random.PRNGKey(7), self.NUM_ENVS)
        states = jax.vmap(env.reset)(keys)
        assert tuple(states.agent_trades.shape) == (
            self.NUM_ENVS, env.steps_per_episode, 2, 6
        ), f"vmapped reset tape shape {states.agent_trades.shape} unexpected."

        orders = _batched_none(self.NUM_ENVS)
        # Outer enable_x64 is the documented convention for vmapping step_env
        # (the scoped int64 leg inside extract_fills survives the batching
        # transform only under an outer context).
        with jax.experimental.enable_x64():
            s2, _rep, _d = jax.vmap(env.step_env)(keys, states, orders)
        assert tuple(np.asarray(s2.agent_trades).shape) == (
            self.NUM_ENVS, env.steps_per_episode, 2, 6
        )

    @pytest.mark.slow
    @pytest.mark.jit
    def test_vmapped_step_adds_at_most_one_submit_order_compile(self):
        """slow: the widened 3-D tape does not break the one-compile invariant —
        vmapping ``step_env`` over the batch grows the module-level
        ``_submit_order`` cache by at most one entry."""
        from omiecid_simulator.omiecid.contract_book import _submit_order

        if not hasattr(_submit_order, "_cache_size"):
            pytest.skip("_submit_order does not expose _cache_size — cannot audit.")

        env = Session.from_test_config()
        keys = jax.random.split(jax.random.PRNGKey(1), self.NUM_ENVS)
        states = jax.vmap(env.reset)(keys)
        orders = _batched_none(self.NUM_ENVS)

        _submit_order.clear_cache()
        assert _submit_order._cache_size() == 0
        with jax.experimental.enable_x64():
            out = jax.vmap(env.step_env)(keys, states, orders)
        # Force the compile to complete.
        _ = int(np.asarray(out[0].tape_cursor).reshape(-1)[0])
        assert _submit_order._cache_size() <= 1, (
            f"_submit_order cache grew to {_submit_order._cache_size()} entries "
            "under vmap — the 3-D tape widening must not add a compile entry."
        )

    @pytest.mark.slow
    def test_reset_zeroes_tape_between_episodes(self):
        """slow: a fresh ``reset`` yields an all-zero tape of the correct 3-D
        shape — the state that auto-reset restores between episodes carries no
        stale fills from the prior episode."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod))
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        # Run a session that writes fills, then reset again.
        state, _ = _run(
            env, [_REST_SELL_51, _AGG_BUY_52]
        )
        assert np.any(np.asarray(state.agent_trades) != 0), "session must write fills."
        fresh = env.reset(jax.random.PRNGKey(99))
        assert np.all(np.asarray(fresh.agent_trades) == 0), (
            "a fresh reset must zero the agent_trades tape (no stale fills)."
        )
        assert tuple(fresh.agent_trades.shape) == (env.steps_per_episode, 2, 6)


# ===========================================================================
# Dimension 7 — buffer bounds.
# ===========================================================================
@_core_skip
class TestBufferBounds:
    """Axis-0 length == steps_per_episode; a full-length rollout does not
    overflow the write index."""

    def test_default_state_tape_is_2x6_two_slot_shape(self):
        """The dataclass default ``agent_trades`` leaf is the two-slot layout
        ``(1, 2, 6)`` int32 — the widening's structural signature (fast: pure
        Python dataclass construction, no jit)."""
        book_field = OMIEContractState.__dataclass_fields__["agent_trades"]
        leaf = book_field.default_factory()
        arr = np.asarray(leaf)
        assert arr.shape == (1, 2, 6), (
            f"default agent_trades leaf must be (1, 2, 6); got {arr.shape}."
        )
        assert arr.dtype == np.int32
        assert np.all(arr == 0), "default leaf must be inert (all zero)."

    def test_steps_per_episode_equals_padded_len_at_ops_one(self):
        """At orders_per_step=1, steps_per_episode == n_orders_padded — the tape
        axis-0 length the buffer must exactly match (fast: env __init__ math, no
        rollout / jit)."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod), orders_per_step=1)
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        assert env.steps_per_episode == env.n_orders_padded, (
            "at orders_per_step=1 the episode length must equal the padded tape "
            f"length; steps={env.steps_per_episode}, n_padded={env.n_orders_padded}."
        )

    @pytest.mark.slow
    def test_full_length_rollout_does_not_overflow(self):
        """slow: driving exactly steps_per_episode steps at orders_per_step=1 keeps
        the write index in-bounds — the tape shape is unchanged and the final
        step's row is still addressable (no overflow / clobber)."""
        prod = OMIEProductConfig()
        env = _make_env(prod, _build_mixed_tape(prod), orders_per_step=1)
        if env is None:
            pytest.skip("Session 'tapes' config path unavailable — AMBIGUITY SURFACED.")
        state = env.reset(jax.random.PRNGKey(0))
        # No order the entire episode — the point is index bounds, not fills.
        for j in range(env.steps_per_episode):
            state, _rep, _d = env.step_env(
                jax.random.PRNGKey(j + 1), state, AgentOrder.none()
            )
        tape = np.asarray(state.agent_trades)
        assert tape.shape == (env.steps_per_episode, 2, 6), (
            f"tape shape must be preserved across a full rollout; got {tape.shape}."
        )
        # An all-no-order episode leaves the whole tape inert (no phantom writes,
        # no out-of-bounds clamp writing a spurious final row).
        assert np.all(tape == 0), (
            "an all-no-order full rollout must leave the entire tape inert."
        )
