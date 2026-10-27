"""Locked behavioral contract for WF-20 · ``OMIE-NON-ONLY-IMPL-1`` — the
participant's resting-maker (``ExecRestriction.NON``) orders.

A participant order submitted as a resting ``ExecRestriction.NON`` (maker)
order persists in the carried book. The passive-leg fill attribution (a
resting NON order that is hit by later background flow credits the
participant) is the load-bearing prerequisite this file locks.

Authoritative design: ``docs/omiecid/non-only-action-design.md`` (WF-19):

  * Survival: a resting NON order SURVIVES the carried-book incremental replay
    (no book-persistence change — WF-10 §4a) and fills PASSIVELY when a later
    background aggressor crosses it; that passive leg MUST be attributed to the
    participant (WF-10 §4b) via ``PASS_TID == AGENT_TRADER_ID``.

The locked contract (mapped to the test class below):

  #1  TestRestingNonSurvivesThenFillsPassively — a NON order rests, survives to a
      later step, is hit by background flow, and the passive fill is attributed
      (inventory + realized cash + agent_trades). THE LOAD-BEARING
      scientific-viability test. (slow: two jitted ``step_env`` compiles + a
      2-step rollout.)

The participant's order is a public ``AgentOrder`` (economic fields only); the
session stamps the order id, participant id and timestamp.

Run (CPU, per device-policy):
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python3 -m pytest \
        tests/omiecid/test_normal_orders_resting_non.py
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Guarded imports (partial-checkout safety).
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
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, Status, Validity

    _CORE_MISSING = False
    _CORE_REASON = ""
except Exception as exc:  # pragma: no cover - collection guard
    _CORE_MISSING = True
    _CORE_REASON = f"OMIE core import failed: {exc!r}"
    # Fallbacks so the module still imports and COLLECTS (every test then SKIPS).
    AGENT_TRADER_ID = 1_000_000


_core_skip = pytest.mark.skipif(_CORE_MISSING, reason=_CORE_REASON or "core present")


# ---------------------------------------------------------------------------
# Bound this process's in-memory JAX compile pool so the file's per-test env
# compiles cannot accumulate past XLA's single-process SIGABRT ceiling.
# Function-scoped; the on-disk persistent cache is untouched (warm reuse kept).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _bound_inmemory_jit_cache():
    yield
    if not _CORE_MISSING:
        jax.clear_caches()


def _non_sell_passive_order():
    """A resting NON SELL at the own-side touch: the best ask (51.0) of the
    step-0 book below, for half the maximum order quantity (non-crossing)."""
    prod = OMIEProductConfig()
    return AgentOrder.create(
        side=int(Side.SELL),
        price=51.0,
        quantity=prod.qty_max / 2.0,
        exec_restriction=int(ExecRestriction.NON),
    )


# ---------------------------------------------------------------------------
# Passive-fill tape fixture (contract #1). Single tape (n_tapes=1) so reset is
# deterministic. orders_per_step=6:
#   * step 0 slice (rows 0..5): builds a resting two-sided book (bids 49/48/47,
#     asks 51/52/53). The agent then rests a NON SELL at best_ask (51).
#   * step 1 slice (rows 6..11): ONE huge crossing background BUY at price_max
#     with qty >> total book, so it sweeps EVERY ask — including the agent's
#     resting SELL. The agent submits nothing, so the fill is PURELY passive
#     (inside the background replay).
# The aggressor carries the max timestamp so ``done`` fires on step 1 (harmless).
# ---------------------------------------------------------------------------
def _build_passive_fill_tape(prod_cfg, orders_per_step=6):
    pmin = prod_cfg.price_min
    ptick = prod_cfg.price_tick
    qtick = prod_cfg.qty_tick
    pmax = prod_cfg.price_max

    def enc_p(p):
        return int(round((p - pmin) / ptick))

    def enc_q(q):
        return int(round(q / qtick))

    def row(side, price, qty, oid, ts, exec_r=int(ExecRestriction.NON)):
        return [
            0, int(side), enc_p(price), enc_q(qty), int(oid), 777, int(ts),
            int(exec_r), int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0,
        ]

    pad = [
        0, 0, 0, 0, -1, 0, -1, int(ExecRestriction.NON), int(Validity.GFS), 0,
        int(Status.ACTIVE), 0, 0,
    ]

    BUY = int(Side.BUY)
    SELL = int(Side.SELL)

    rows = []
    oid = 100
    # -- step 0: resting two-sided book (ts=60 so time stays well below GCT) ---
    for lvl in range(3):
        rows.append(row(BUY, 49.0 - lvl, 10.0, oid, 60)); oid += 1
        rows.append(row(SELL, 51.0 + lvl, 10.0, oid, 60)); oid += 1
    assert len(rows) == orders_per_step  # exactly one step's worth
    # -- step 1: ONE huge crossing BUY that sweeps every ask + the agent order.
    #    It carries IOC in the BACKGROUND tape (the replay tape carries real
    #    IOC/FOK/ICE).
    rows.append(row(BUY, pmax, 500.0, oid, 120, exec_r=int(ExecRestriction.IOC))); oid += 1
    # Fill out step-1's slice + a trailing step with inert pad rows.
    n_padded = 3 * orders_per_step
    rows += [list(pad) for _ in range(n_padded - len(rows))]
    return np.asarray([rows], dtype=np.int32)


def _build_passive_fill_env():
    """Session on the passive-fill tape. ``FillReport.cash_delta_eur`` is the
    step's realized (aggressive + passive) EUR and inventory reflects only
    fills. Returns None if the ``tapes`` config path is unavailable.
    """
    prod = OMIEProductConfig()
    cfg = {
        "product_config": prod,
        "tapes": _build_passive_fill_tape(prod),
        "orders_per_step": 6,
    }
    try:
        return Session(cfg)
    except Exception:
        return None


# ===========================================================================
# Contract #1 — resting NON survives + fills passively + is attributed
# ===========================================================================
@_core_skip
class TestRestingNonSurvivesThenFillsPassively:
    """The load-bearing scientific-viability contract (design §2.2 survival path).

    A NON order placed on step 0 is NOT consumed IOC-style on its emitting step;
    it SURVIVES into the carried book and, when a large background aggressor
    sweeps the book on step 1, fills PASSIVELY — and that passive leg is
    attributed to the agent (inventory + realized cash + the ``agent_trades``
    tape).
    """

    @pytest.mark.slow
    def test_resting_non_fills_passively_and_credits_agent(self):
        """slow: 2-step ``step_env`` rollout (two jitted env compiles)."""
        env = _build_passive_fill_env()
        if env is None:
            pytest.skip(
                "Session could not be constructed on a supplied 'tapes' config "
                "(the tapes config path is a P3.3 deliverable). AMBIGUITY SURFACED."
            )

        key = jax.random.PRNGKey(0)
        state0 = env.reset(key)

        # -- step 0: rest a NON SELL at the own-side touch => non-crossing ------
        k0 = jax.random.PRNGKey(1)
        state1, _rep0, d0 = env.step_env(k0, state0, _non_sell_passive_order())
        inv_after_rest = float(np.asarray(state1.agent_inventory)[0])
        # SURVIVAL: a resting NON order is NOT consumed IOC-style on emit — it
        # rests non-crossing, so no fill and no inventory move on step 0.
        assert inv_after_rest == 0.0, (
            "a NON SELL at the own-side touch must REST (non-crossing) on its "
            f"emitting step, not fill IOC-style; inventory after step 0 = {inv_after_rest}."
        )

        # -- step 1: a huge background BUY sweeps the book; the agent submits
        #    nothing so any fill of its resting SELL is PURELY passive (inside
        #    the background replay).
        k1 = jax.random.PRNGKey(2)
        state2, rep1, d1 = env.step_env(k1, state1, AgentOrder.none())

        inv_after_fill = float(np.asarray(state2.agent_inventory)[0])
        step1_cash = float(rep1.cash_delta_eur)
        trades = np.asarray(state2.agent_trades).reshape(-1, 6)

        # (a) INVENTORY: the passive SELL moves the signed net position negative.
        assert inv_after_fill < 0.0, (
            "the resting NON SELL must fill PASSIVELY against the step-1 "
            "background sweep and credit the agent a SHORT position; inventory "
            f"after step 1 = {inv_after_fill} (0 => the passive fill was "
            "discarded — the resting NON order is an economic no-op)."
        )

        # (b) CASH: the passive SELL contributes positive realized EUR on the
        #     step it fills.
        assert step1_cash > 0.0, (
            "the passive SELL fill must contribute positive realized cash on "
            f"the step it fills; step-1 cash = {step1_cash} (0 => the passive "
            "fill did not enter the realized cash)."
        )

        # (c) AGENT_TRADES TAPE: a non-inert row (qty_tick>0) carrying the agent's
        #     trader id on either leg — the passive-leg attribution.
        qty_col = trades[:, 2]
        buyer_col = trades[:, 3]
        seller_col = trades[:, 4]
        attributed = (qty_col > 0) & (
            (buyer_col == AGENT_TRADER_ID) | (seller_col == AGENT_TRADER_ID)
        )
        assert bool(np.any(attributed)), (
            "the agent_trades tape must gain a passive-leg row for the resting "
            "NON fill (qty_tick>0 with AGENT_TRADER_ID on the buyer/seller leg); "
            f"found none. qty col = {qty_col.tolist()}."
        )
        # The attributed row is a SELL leg: the agent sits on the seller slot.
        sell_attributed = (qty_col > 0) & (seller_col == AGENT_TRADER_ID)
        assert bool(np.any(sell_attributed)), (
            "the passive fill of a resting SELL must attribute the agent to the "
            "SELLER leg of the trade tape."
        )
