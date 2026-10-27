"""Locked behavioral contract tests for WF-11-FOLLOWUP-1 ·
``OMIE-AGENT-TRADES-PERFILL-TAPE-1`` — restore the eager cross-check invariant
under a simultaneous aggressive + passive fill by widening the per-step
``agent_trades`` tape to carry BOTH legs.

The defect being closed
-----------------------
WF-11 added passive-leg fill attribution: a resting NON order's passive fill is
folded into realized cash (``FillReport.cash_delta_eur``) + inventory (economics CORRECT). BUT the v1
``OMIEContractState.agent_trades`` tape carries ONE row per step
(``(steps_per_episode, 6)`` int32). On a step with BOTH an aggressive (own-submit)
fill AND a passive (resting-NON) fill, the passive row takes precedence
(``omie_session.py`` ~L622-624) and the AGGRESSIVE leg is DROPPED from the tape. So
the P3.2c eager ``compute_agent_pnl`` tape-reconciliation diagnostic DIVERGES on
that step (measured: passive-SELL(+1.25)/aggressive-BUY(-1.30) → per-step
``realized cash == -0.05``, but ``compute_agent_pnl`` over the tape
== +1.25). Reward/inventory are correct — only the tape-based reconciliation
breaks.

The fix (developer-owned; NOT in this file)
--------------------------------------------
Widen the per-step ``agent_trades`` slot to carry BOTH legs — a two-slot /
per-fill row (``(steps_per_episode, 2, 6)`` or an equivalent per-fill layout) —
so both the aggressive AND the passive leg land on the tape and the eager
``compute_agent_pnl`` reconciles the summed realized cash (``FillReport.cash_delta_eur``) again. Must
stay jit/vmap-safe (fixed shapes, no ragged rows) and preserve the byte-identical
IOC-only / <=1-fill reward + inventory (tape-fidelity only).

SUPERSEDES (not edited here): the WF-11 characterization test
``tests/omiecid/test_normal_orders_resting_non_audit.py::TestEagerPnlReconciliation
::test_simultaneous_fill_step_tape_records_only_passive_leg`` pins the ACTUAL
(defective) one-row-per-step behaviour + the divergence. When this follow-up
lands, the developer/implementation supersedes that characterization; this file
does NOT edit it.

Contract map (3 locked contracts)
----------------------------------
1. Simultaneous-fill reconciliation — ``TestSimultaneousFillReconciliation``:
   on a real multi-step ``step_env`` rollout that produces a step with BOTH an
   aggressive fill AND a passive (resting-NON) fill, the eager
   ``compute_agent_pnl`` over the widened tape reconciles the summed per-step
   realized cash (``FillReport.cash_delta_eur``) within float32 tolerance (the WF-11 divergence is
   CLOSED) and BOTH legs appear on the tape. RED now: the aggressive leg is
   dropped, so the tape PnL (passive only) diverges from the folded realized sum.
2. IOC-only / <=1-fill byte-identity — ``TestSingleFillByteIdentity``: a step
   with no fill or a single fill produces a tape byte-identical to the WF-11
   baseline MODULO the inert added slot, and reward + inventory are unchanged.
   RED now: the tape has not yet gained the second per-step slot (capacity check).
3. jit/vmap one-compile — ``TestJitVmapOneCompile``: the widened tape rides the
   jitted/vmapped ``step_env`` with fixed shapes (no ragged rows), adding no new
   ``_submit_order`` compile entry. RED now: the batched tape lacks the widened
   per-step capacity.

Layout-agnostic capacity contract
---------------------------------
The card permits EITHER a ``(steps_per_episode, 2, 6)`` layout OR an equivalent
two-slot / per-fill row. Rather than pin one, the widened-capacity assertion is
expressed layout-agnostically as::

    state.agent_trades.reshape(-1, 6).shape[0] == 2 * env.steps_per_episode

which holds for ``(steps, 2, 6)`` (→ ``2*steps`` rows) and for a flat
``(steps, 12)`` per-fill layout (→ ``2*steps`` rows) alike, and FAILS for the v1
``(steps, 6)`` tape (→ ``steps`` rows). The reconciliation tests likewise reshape
the tape to ``(-1, 6)`` before calling ``compute_agent_pnl`` so they work under
either layout (identity for the v1 tape, the downstream-reader update for the
widened tape).

Device / tiering: CPU-hermetic (``tests/omiecid/conftest.py`` pins CPU). Every test
constructs + steps the jitted env, so all carry ``@pytest.mark.slow`` (env reset
+ ``step_env`` compile + rollout, > 1s); the vmap one-compile tests additionally
carry ``@pytest.mark.jit``.

Run:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python3 -m pytest \
        tests/omiecid/test_agent_trades_perfill_tape.py -v
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
    from omiecid_simulator.omiecid.contract_book import _submit_order

    _MISSING = False
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - collection-time guard
    _MISSING = True
    _IMPORT_ERROR = repr(exc)

_skip = pytest.mark.skipif(
    _MISSING, reason=f"WF-11-FOLLOWUP-1 surfaces not importable ({_IMPORT_ERROR})"
)


# ---------------------------------------------------------------------------
# Tape-building helpers (13-column loader layout, tick-encoded). Mirrors the
# WF-11 audit fixtures so the simultaneous-fill scenario is reproduced exactly;
# this file OWNS its own copy (the Test Agent is the sole author of its tests).
# ---------------------------------------------------------------------------
_P = OMIEProductConfig() if not _MISSING else None


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
    """Inert pad sentinel (order_id<=0, timestamp<=0 => no fill, no time advance)."""
    return [0, 0, 0, 0, -1, 0, -1, int(ExecRestriction.NON),
            int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0]


def _build_tape(rows, n_padded=16):
    """A single (1, n_padded, 13) int32 tape — n_tapes==1 => deterministic reset."""
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


# Participant orders (side, price EUR/MWh, quantity MW): every order is a
# resting ``ExecRestriction.NON`` GFS order of one 0.1-MW qty tick. The
# aggressive leg is a marketable NON that crosses resting opposite liquidity
# (and would rest any residual), NOT a taker IOC. ``None`` submits nothing.
NOOP = None
_QTY = 0.1
NON_SELL_MID = (1, 50.0, _QTY)          # rest a SELL at the 45/55 mid
NON_BUY_CROSS_52 = (0, 52.0, _QTY)      # BUY crossing the fresh 52 ask
NON_BUY_CROSS_55 = (0, 55.0, _QTY)      # BUY crossing the opener's 55 ask
NON_SELL_MID_48_5 = (1, 48.5, _QTY)     # SELL at the 45/52 mid


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


def _opener():
    """A two-sided book opener: resting bid@45 + ask@55 (mid 50)."""
    return [_row(Side.BUY, 45.0, 10.0, 101, 10),
            _row(Side.SELL, 55.0, 10.0, 102, 20)]


def _eager_pnl(agent_trades):
    """Eager ``compute_agent_pnl`` over the agent tape, layout-agnostic.

    Reshapes to ``(-1, 6)`` so it works for the v1 ``(steps, 6)`` tape (identity)
    AND for the widened ``(steps, 2, 6)`` / flat ``(steps, 12)`` per-fill layouts
    (the downstream-reader update). ``compute_agent_pnl`` opens its own scoped x64
    context on the eager path, so no outer ceremony is needed here.
    """
    tape = jnp.asarray(agent_trades).reshape(-1, 6)
    return float(compute_agent_pnl(tape, jnp.int32(AGENT_TRADER_ID)))


def _sum_realized(env, state, orders):
    """Roll out ``orders``, returning (sum realized cash, per-step passive
    cash, final state). Stops early on ``done``."""
    sum_realized = 0.0
    passives = []
    done = False
    for spec in orders:
        if bool(done):
            break
        state, report, done = _step(env, state, spec)
        sum_realized += float(report.cash_delta_eur)
        passives.append(float(report.passive_value_eur))
    return sum_realized, passives, state


# The tape that produces a SIMULTANEOUS aggressive + passive fill on step 1:
#   step 0: agent rests a NON SELL at the mid (50).
#   step 1: a background BUY@60 sweeps the resting SELL (agent = PASSIVE seller),
#           AND refills a fresh ask@52 that the agent's own NON BUY@52 then crosses
#           (agent = AGGRESSIVE buyer). Both legs fill on the SAME step.
def _simultaneous_fill_tape():
    return _build_tape(
        _opener()
        + [_row(Side.BUY, 60.0, 5.0, 103, 40, int(ExecRestriction.IOC)),
           _row(Side.SELL, 52.0, 10.0, 104, 45)]
    )


_SIMUL_ORDERS = [NON_SELL_MID, NON_BUY_CROSS_52]


# ===========================================================================
# Contract 1 — simultaneous-fill reconciliation (the load-bearing contract).
# The WF-11 divergence must be CLOSED: eager compute_agent_pnl over the widened
# tape reconciles the summed per-step realized cash, and BOTH legs are on
# the tape.
# ===========================================================================
@_skip
class TestSimultaneousFillReconciliation:
    """Contract 1. RED now: the aggressive leg is dropped from the one-row-per-step
    tape, so the eager tape PnL (passive only) diverges from the folded realized
    reward (aggressive + passive) on the simultaneous-fill step."""

    @pytest.mark.slow  # slow: env reset + 2 step_env compiles + rollout + eager x64 pnl
    def test_simultaneous_fill_reconciles_with_compute_agent_pnl(self):
        """On a real rollout with a step carrying BOTH an aggressive (own NON
        BUY-CROSS, marketable cross-and-rest) AND a passive (resting NON SELL hit
        by background flow) fill, the eager
        ``compute_agent_pnl`` over the widened tape equals the summed
        realized cash (``FillReport.cash_delta_eur``) within float32 ``atol``/``rtol``.

        This is the WF-11-FOLLOWUP-1 acceptance claim. On the v1 one-row-per-step
        tape the aggressive leg is overwritten by the passive row, so the eager
        reference captures the passive leg alone and DIVERGES (measured:
        realized == -0.05 vs compute_agent_pnl == +1.25).
        """
        env = _make_env(_simultaneous_fill_tape())
        state = env.reset(_key())
        # step 0: rest the NON SELL.
        state, rep0, _d0 = _step(env, state, NON_SELL_MID)
        # step 1: the simultaneous aggressive + passive fill step.
        state, rep1, _d1 = _step(env, state, NON_BUY_CROSS_52)

        # -- Guard: the step genuinely drove a SIMULTANEOUS fill (else the test is
        #    vacuous). Passive SELL leg present (>0) AND an aggressive BUY cost
        #    folded into realized (aggressive = realized - passive < 0).
        passive = float(rep1.passive_value_eur)
        realized_step1 = float(rep1.cash_delta_eur)
        aggressive = realized_step1 - passive
        assert passive > 0.0, (
            "scenario guard: the prior resting SELL must fill PASSIVELY on this "
            f"step (passive_value_eur={passive})"
        )
        assert aggressive < 0.0, (
            "scenario guard: the agent's own NON BUY-CROSS must fill AGGRESSIVELY as a "
            f"cost on this step (aggressive={aggressive})"
        )

        # -- The reconciliation contract: eager tape PnL == summed realized over
        #    the whole (2-step) episode.
        sum_realized = float(rep0.cash_delta_eur) + realized_step1
        pnl = _eager_pnl(state.agent_trades)
        assert sum_realized == pytest.approx(pnl, abs=1e-3, rel=1e-4), (
            "WF-11-FOLLOWUP-1: the eager compute_agent_pnl over the widened tape "
            "must reconcile the summed realized cash on a simultaneous "
            f"aggressive+passive fill step (sum_realized={sum_realized}, "
            f"compute_agent_pnl={pnl}); divergence => the aggressive leg is still "
            "being dropped from the tape."
        )

    @pytest.mark.slow  # slow: env reset + 2 step_env compiles + rollout
    def test_both_legs_present_on_the_simultaneous_step(self):
        """The widened tape records BOTH legs of the simultaneous step: an
        aggressive BUY row (agent as BUYER, settlement col 3 == AGENT_TRADER_ID)
        AND a passive SELL row (agent as SELLER, settlement col 4 ==
        AGENT_TRADER_ID). RED now: the v1 tape keeps only the passive SELL row.

        Column layout (settlement.py): [ts, price_off, qty_tick, buyer, seller,
        contract_id]; a non-inert fill row has qty_tick (col 2) != 0.
        """
        env = _make_env(_simultaneous_fill_tape())
        state = env.reset(_key())
        _sum, _passives, final_state = _sum_realized(env, state, _SIMUL_ORDERS)

        tape = np.asarray(jnp.asarray(final_state.agent_trades).reshape(-1, 6))
        fill_rows = tape[tape[:, 2] != 0]  # non-inert (qty_tick != 0)

        agent = int(AGENT_TRADER_ID)
        agent_is_buyer = fill_rows[:, 3] == agent   # aggressive BUY leg
        agent_is_seller = fill_rows[:, 4] == agent  # passive SELL leg
        assert agent_is_buyer.any(), (
            "the AGGRESSIVE BUY leg (agent as buyer) must appear on the widened "
            f"tape; recorded fill rows:\n{fill_rows}"
        )
        assert agent_is_seller.any(), (
            "the PASSIVE SELL leg (agent as seller) must appear on the widened "
            f"tape; recorded fill rows:\n{fill_rows}"
        )


# ===========================================================================
# Contract 2 — IOC-only / <=1-fill byte-identity (modulo the inert added slot).
# The fix is tape-fidelity only: reward + inventory are UNCHANGED, and a <=1-fill
# step's non-inert content is byte-identical to the WF-11 baseline; the tape only
# gains a second (inert) per-step slot.
# ===========================================================================
@_skip
class TestSingleFillByteIdentity:
    """Contract 2. The reward/inventory + reconciliation invariants HOLD on the
    v1 tape already (tape-fidelity-only fix); the RED assertion is the widened
    per-step capacity (the added inert slot). Expressed layout-agnostically as
    ``reshape(-1, 6).shape[0] == 2 * steps_per_episode`` so it accepts a
    ``(steps, 2, 6)`` OR a flat per-fill layout and rejects the v1 ``(steps, 6)``.
    """

    def _ioc_only_env_state(self):
        # A book with a resting ask@55 the agent's NON BUY-CROSS crosses on step 0;
        # no resting agent order => NO passive fill on any step (IOC-only world).
        env = _make_env(_build_tape(_opener()))
        state = env.reset(_key())
        return env, state

    @pytest.mark.slow  # slow: env reset + step_env compile + rollout
    def test_ioc_only_cash_and_inventory_unchanged(self):
        """Byte-identity of the ECONOMICS: on a single aggressive NON BUY-CROSS fill the
        step realized cash == the cash accumulator, inventory rises by the filled
        qty, and NO passive fill is registered. These must NOT change under the
        widening (tape-fidelity only). Holds on the v1 tape and must stay held.
        """
        env, state = self._ioc_only_env_state()
        state, rep0, _d0 = _step(env, state, NON_BUY_CROSS_55)
        assert float(rep0.passive_value_eur) == 0.0, (
            "an IOC-only step must register NO passive fill"
        )
        assert float(rep0.cash_delta_eur) == pytest.approx(
            float(state.agent_cash[0]), abs=1e-4
        ), "the step's realized cash must equal the cash accumulator — economics unchanged"
        inv = float(state.agent_inventory[0])
        assert inv == pytest.approx(0.1, abs=1e-4), (
            f"an aggressive BUY of one 0.1-MW qty tick must raise inventory to "
            f"+0.1 MW (got {inv}) — inventory fold unchanged"
        )

    @pytest.mark.slow  # slow: env reset + step_env compile + rollout + eager x64 pnl
    def test_ioc_only_tape_reconciles_and_records_single_fill(self):
        """The IOC-only reconciliation still holds AND the non-inert tape content
        is exactly ONE fill row (the aggressive BUY, agent as buyer). Reshaped to
        ``(-1, 6)`` this holds on both the v1 and the widened tape (the widened
        tape's extra slot is inert => dropped by the qty_tick!=0 filter).
        """
        env, state = self._ioc_only_env_state()
        sum_realized, passives, final_state = _sum_realized(
            env, state, [NON_BUY_CROSS_55, NOOP, NOOP]
        )
        assert all(p == 0.0 for p in passives), "no passive fills in the IOC world"
        pnl = _eager_pnl(final_state.agent_trades)
        assert sum_realized == pytest.approx(pnl, abs=1e-3, rel=1e-4), (
            "IOC-only reconciliation must be preserved (tape-fidelity-only fix)"
        )
        tape = np.asarray(jnp.asarray(final_state.agent_trades).reshape(-1, 6))
        fill_rows = tape[tape[:, 2] != 0]
        assert fill_rows.shape[0] == 1, (
            f"an IOC-only single-fill episode must record exactly one non-inert "
            f"row (got {fill_rows.shape[0]}):\n{fill_rows}"
        )
        assert int(fill_rows[0, 3]) == int(AGENT_TRADER_ID), (
            "the recorded row is the aggressive BUY (agent as buyer, col 3)"
        )

    @pytest.mark.slow  # slow: env reset + step_env compile + rollout
    def test_tape_gained_second_per_step_slot(self):
        """RED contract: the widened tape carries TWO 6-column rows per step (the
        aggressive slot + the passive slot), so a ``<=1``-fill step keeps one inert
        companion slot. Layout-agnostic capacity check — FAILS on the v1
        ``(steps, 6)`` tape (only ``steps`` rows).
        """
        env, state = self._ioc_only_env_state()
        state, _rep0, _d0 = _step(env, state, NON_BUY_CROSS_55)
        total_rows = int(jnp.asarray(state.agent_trades).reshape(-1, 6).shape[0])
        assert total_rows == 2 * env.steps_per_episode, (
            "the widened agent_trades tape must hold TWO (6,) rows per step "
            f"(aggressive + passive slot): expected {2 * env.steps_per_episode} "
            f"total rows, got {total_rows}. On the v1 one-row-per-step tape this "
            "is exactly steps_per_episode — the widening has not landed."
        )
        assert jnp.asarray(state.agent_trades).dtype == jnp.int32, (
            "the widened tape must stay int32"
        )


# ===========================================================================
# Contract 3 — jit/vmap one-compile with fixed (non-ragged) shapes.
# The widened tape must ride the jitted/vmapped step_env adding no new
# _submit_order compile entry, with a static per-env shape.
# ===========================================================================
@_skip
class TestJitVmapOneCompile:
    """Contract 3. The one-compile invariant HOLDS on the v1 tape already; the RED
    assertion is the widened per-step capacity surviving vmap with a fixed shape.
    """

    NUM_ENVS = 4

    def _batched_orders(self, spec):
        # Broadcast one AgentOrder across NUM_ENVS (a leading batch axis on
        # every field).
        one = _agent_order(spec)
        return jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (self.NUM_ENVS,)), one
        )

    @pytest.mark.slow  # slow: env reset + step_env compile
    @pytest.mark.jit
    def test_no_new_submit_order_compile_entry(self):
        """The tape widening is pure per-step assembly fused into the SAME jitted
        ``step_env`` body — it must NOT add a ``_submit_order`` compile-cache
        entry (one-compile invariant). Holds now; must stay held under the fix.
        """
        assert hasattr(_submit_order, "_cache_size"), (
            "_submit_order must expose _cache_size()"
        )
        _submit_order.clear_cache()
        assert _submit_order._cache_size() == 0
        env = _make_env(_simultaneous_fill_tape())
        state = env.reset(_key())
        for spec in (NON_SELL_MID, NON_BUY_CROSS_52, NOOP, NON_SELL_MID_48_5):
            state, _rep, _d = _step(env, state, spec)
        assert _submit_order._cache_size() <= 1, (
            f"_submit_order cache grew to {_submit_order._cache_size()} — the "
            "tape widening must NOT add a compile entry (one-compile invariant)."
        )

    @pytest.mark.slow  # slow: env reset + vmap step_env compile over NUM_ENVS
    @pytest.mark.jit
    def test_widened_tape_rides_vmap_fixed_shape_no_extra_compile(self):
        """The widened tape rides ``jax.vmap(step_env)`` over NUM_ENVS with a fixed
        (non-ragged) per-env shape and no extra ``_submit_order`` compile entry.

        RED contract: the batched tape must carry ``2 * steps_per_episode`` rows
        per env (the two per-step slots). On the v1 tape it carries only
        ``steps_per_episode`` per env. The one-compile + finite-reward invariants
        hold now and must stay held.
        """
        _submit_order.clear_cache()
        env = _make_env(_simultaneous_fill_tape())
        keys = jax.random.split(jax.random.PRNGKey(0), self.NUM_ENVS)
        state_b = jax.vmap(env.reset)(keys)
        orders_b = self._batched_orders(NON_BUY_CROSS_55)
        step_keys = jax.random.split(jax.random.PRNGKey(1), self.NUM_ENVS)
        with jax.experimental.enable_x64():
            out = jax.vmap(env.step_env)(step_keys, state_b, orders_b)
        sb, report_b, _db = out

        # realized-cash leaf stays a batched 0-d-per-env float32 (no
        # dtype/shape leak).
        rleaf = jnp.asarray(report_b.cash_delta_eur)
        assert rleaf.shape[0] == self.NUM_ENVS, (
            f"batched cash leading axis {rleaf.shape} != NUM_ENVS"
        )
        assert rleaf.dtype == jnp.float32

        # one-compile invariant preserved under vmap.
        assert _submit_order._cache_size() <= 1, (
            f"_submit_order cache grew to {_submit_order._cache_size()} under "
            "vmap — the tape widening must preserve the one-compile invariant."
        )

        # RED: the batched widened tape carries 2 rows/step/env with a fixed shape.
        batched = jnp.asarray(sb.agent_trades)
        assert batched.shape[0] == self.NUM_ENVS, (
            f"batched agent_trades leading axis {batched.shape} != NUM_ENVS"
        )
        per_env_rows = int(batched.reshape(self.NUM_ENVS, -1, 6).shape[1])
        assert per_env_rows == 2 * env.steps_per_episode, (
            "the widened agent_trades tape must hold TWO (6,) rows per step per "
            f"env under vmap: expected {2 * env.steps_per_episode} rows/env, got "
            f"{per_env_rows}. The v1 one-row-per-step tape yields "
            "steps_per_episode — the widening has not landed."
        )
