"""Step-5 audit for OMIE-ICEWPI-SIGN-REJECT-1 (S1, D8, XS).

Independent of the locked file ``tests/omiecid/test_icewpi_sign_reject.py``
(not read). Derived from the story card (agile-plan.md, "Certification fix
pass") and market-spec.md §13.6 (DA-PPD-2026-04, sign rule, hard):

    ICEwPI buy orders must have PPD < 0; sell orders PPD > 0.
    PPD = 0 is a plain ICE (accepted). Non-ICE PPD is not checked.
    A sub-half-tick PPD rounds to 0 ticks (Method A) -> plain ICE.

Probe areas (beyond the card's nine locked cases):
  A. rounding boundary just above / below half a tick, both sides;
  B. large magnitudes (real-data -3.00, near the derived 19,998 ceiling);
  C. sign rule composes with other checks (never overrides a peak failure);
  D. vmap over a mixed batch gives the per-lane verdict;
  E. a rejected wrong-sign order leaves the state byte-identical, so a later
     conforming submit behaves exactly as on a fresh book (ts, n_active,
     iceberg_meta, resting views);
  F. a wrong-sign ICE never disturbs an already-resting iceberg or crosses;
  G. non-ICE wrong-sign PPD still trades (not checked);
  H. traced PPD sign changes do not recompile the jitted validator.

Incidental constants are varied (sides, OIDs of 2-7 digits, off-round
prices, agent ids). CPU default; synthetic fixtures; fast suite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    make_order,
)
from omiecid_simulator.omiecid.order_validator import validate_order

pytestmark = pytest.mark.jit

CID = 53
PCFG = OMIEProductConfig()
MCFG = OMIEMarketConfig()


@pytest.fixture(scope="module")
def book() -> ContractBook:
    contract = Contract(
        contract_id=CID,
        delivery_start_minutes=13 * 60 + 15,
        delivery_end_minutes=13 * 60 + 30,
        gct_minutes=12 * 60 + 15,
        status=ContractStatus.OPEN,
    )
    return ContractBook(contract, PCFG, MCFG)


def _ice(side, price, qty, peak, ppd, oid, agent=7) -> OMIEOrder:
    return make_order(
        contract_id=CID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=agent, timestamp=0,
        exec_restriction=ExecRestriction.ICE, peak_qty=peak, ppd=ppd,
    )


def _plain(side, price, qty, oid, er=ExecRestriction.NON, ppd=0.0, agent=9):
    return make_order(
        contract_id=CID, side=side, price=price, quantity=qty,
        order_id=oid, agent_id=agent, timestamp=0,
        exec_restriction=er, ppd=ppd,
    )


def _leaves_equal(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb)
    )


def _assert_rejected(res):
    assert bool(res.accepted) is False
    assert bool(res.traded) is False
    assert float(res.trade_qty) == 0.0
    assert float(res.settlement_value) == 0.0


_jit_validate = jax.jit(validate_order, static_argnums=(1, 2))


def _valid(order, jitted=False) -> bool:
    fn = _jit_validate if jitted else validate_order
    return bool(fn(order, PCFG, True))


# ---------------------------------------------------------------------------
# A. Rounding boundary (Method A: round(ppd * 100) ticks)
# ---------------------------------------------------------------------------


class TestRoundingBoundary:
    @pytest.mark.parametrize("jitted", [False, True])
    def test_buy_ppd_just_over_half_tick_rejected(self, jitted):
        # +0.006 -> 1 tick (> 0) -> wrong sign for BUY.
        assert _valid(_ice(Side.BUY, 47.83, 6.3, 1.1, 0.006, 4821), jitted) is False

    @pytest.mark.parametrize("jitted", [False, True])
    def test_sell_ppd_just_under_minus_half_tick_rejected(self, jitted):
        assert _valid(_ice(Side.SELL, 62.19, 3.7, 0.9, -0.006, 913), jitted) is False

    @pytest.mark.parametrize("jitted", [False, True])
    def test_sell_sub_half_tick_negative_ppd_is_plain_ice(self, jitted):
        # Mirror of the card's BUY +0.004 guard, on the SELL side.
        assert _valid(_ice(Side.SELL, 62.19, 3.7, 0.9, -0.004, 914), jitted) is True

    @pytest.mark.parametrize("jitted", [False, True])
    def test_one_tick_conforming_ppd_accepted(self, jitted):
        assert _valid(_ice(Side.BUY, 47.83, 6.3, 1.1, -0.01, 33), jitted) is True
        assert _valid(_ice(Side.SELL, 47.83, 6.3, 1.1, 0.01, 34), jitted) is True

    @pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
    def test_negative_zero_ppd_is_plain_ice(self, side):
        assert _valid(_ice(side, 51.07, 2.4, 0.8, -0.0, 77)) is True


# ---------------------------------------------------------------------------
# B. Large magnitudes (spec: no magnitude bound; only the sign is ruled)
# ---------------------------------------------------------------------------


class TestMagnitudes:
    @pytest.mark.parametrize("ppd", [-3.00, -19998.0])
    def test_buy_large_negative_ppd_accepted(self, ppd):
        # -3.00 observed on real ICEwPI BUY rows (market-spec §13.6).
        assert _valid(_ice(Side.BUY, 49.61, 8.2, 2.1, ppd, 600123)) is True

    @pytest.mark.parametrize("ppd", [3.00, 19998.0])
    def test_buy_large_positive_ppd_rejected(self, ppd):
        assert _valid(_ice(Side.BUY, 49.61, 8.2, 2.1, ppd, 600124)) is False

    @pytest.mark.parametrize("ppd", [1.55, 19998.0])
    def test_sell_large_positive_ppd_accepted(self, ppd):
        assert _valid(_ice(Side.SELL, 70.43, 8.2, 2.1, ppd, 6001255)) is True

    @pytest.mark.parametrize("ppd", [-1.55, -19998.0])
    def test_sell_large_negative_ppd_rejected(self, ppd):
        assert _valid(_ice(Side.SELL, 70.43, 8.2, 2.1, ppd, 6001256)) is False


# ---------------------------------------------------------------------------
# C. Composition with the other checks
# ---------------------------------------------------------------------------


class TestComposition:
    def test_conforming_sign_does_not_rescue_peak_above_total(self):
        assert _valid(_ice(Side.BUY, 44.44, 2.0, 3.0, -0.25, 19)) is False
        assert _valid(_ice(Side.SELL, 44.44, 2.0, 3.0, 0.25, 20)) is False

    def test_conforming_sign_does_not_rescue_zero_peak(self):
        assert _valid(_ice(Side.SELL, 44.44, 2.0, 0.0, 0.25, 21)) is False

    def test_conforming_sign_does_not_rescue_out_of_range_price(self):
        assert _valid(_ice(Side.BUY, 10000.5, 2.0, 1.0, -0.25, 22)) is False

    def test_wrong_sign_rejected_on_closed_contract_too(self):
        assert bool(validate_order(
            _ice(Side.BUY, 44.44, 2.0, 1.0, 0.25, 23), PCFG, False)) is False

    def test_conforming_ice_on_closed_contract_rejected(self):
        # The sign rule must not weaken the contract-open gate.
        assert bool(validate_order(
            _ice(Side.BUY, 44.44, 2.0, 1.0, -0.25, 24), PCFG, False)) is False


# ---------------------------------------------------------------------------
# D. vmap over a mixed batch
# ---------------------------------------------------------------------------


def test_vmapped_validator_gives_per_lane_sign_verdict():
    cases = [
        (Side.BUY, ExecRestriction.ICE, 0.10, False),
        (Side.BUY, ExecRestriction.ICE, -0.10, True),
        (Side.SELL, ExecRestriction.ICE, -1.00, False),
        (Side.SELL, ExecRestriction.ICE, 1.00, True),
        (Side.BUY, ExecRestriction.ICE, 0.0, True),
        (Side.SELL, ExecRestriction.NON, -0.10, True),
        (Side.BUY, ExecRestriction.IOC, 0.10, True),
        (Side.BUY, ExecRestriction.FOK, 0.10, True),
    ]
    n = len(cases)
    batch = OMIEOrder(
        contract_id=jnp.full(n, CID, jnp.int32),
        side=jnp.array([int(c[0]) for c in cases], jnp.int32),
        price=jnp.linspace(41.17, 58.93, n, dtype=jnp.float32).round(2),
        quantity=jnp.full(n, 4.3, jnp.float32),
        order_id=jnp.arange(1001, 1001 + n, dtype=jnp.int32),
        agent_id=jnp.full(n, 3, jnp.int32),
        timestamp=jnp.zeros(n, jnp.int32),
        exec_restriction=jnp.array([int(c[1]) for c in cases], jnp.int32),
        validity=jnp.zeros(n, jnp.int32),
        expiry_time=jnp.zeros(n, jnp.int32),
        status=jnp.zeros(n, jnp.int32),
        peak_qty=jnp.array(
            [1.2 if c[1] == ExecRestriction.ICE else 0.0 for c in cases],
            jnp.float32),
        ppd=jnp.array([c[2] for c in cases], jnp.float32),
    )
    out = jax.vmap(lambda o: validate_order(o, PCFG, True))(batch)
    assert [bool(x) for x in np.asarray(out)] == [c[3] for c in cases]


# ---------------------------------------------------------------------------
# E. Rejection is a true no-op on the state (submit path)
# ---------------------------------------------------------------------------


class TestRejectionIsNoOp:
    @pytest.mark.parametrize(
        "side,price,ppd,oid",
        [(Side.BUY, 52.71, 0.01, 88), (Side.BUY, 52.71, 1.00, 4402),
         (Side.SELL, 66.09, -0.01, 731905), (Side.SELL, 66.09, -1.00, 51)],
    )
    def test_whole_state_byte_identical_after_wrong_sign(
        self, book, side, price, ppd, oid
    ):
        s0 = book.init()
        s1, res = book.submit_order(s0, _ice(side, price, 5.0, 1.0, ppd, oid))
        _assert_rejected(res)
        assert _leaves_equal(s0, s1)

    def test_rejection_then_conforming_matches_fresh_book(self, book):
        conforming = _ice(Side.SELL, 63.29, 4.0, 1.5, 0.20, 70112, agent=11)
        fresh, r_fresh = book.submit_order(book.init(), conforming)
        s, r_bad = book.submit_order(
            book.init(), _ice(Side.SELL, 63.29, 4.0, 1.5, -0.20, 70111, agent=11))
        _assert_rejected(r_bad)
        after, r_ok = book.submit_order(s, conforming)
        assert bool(r_ok.accepted) is True and bool(r_fresh.accepted) is True
        assert _leaves_equal(fresh, after)

    def test_rejected_order_not_visible_in_any_view(self, book):
        s0 = book.init()
        s1, res = book.submit_order(
            s0, _ice(Side.BUY, 48.36, 6.0, 2.0, 0.30, 27, agent=41))
        _assert_rejected(res)
        assert book.get_anonymous_lob(s1) == ([], [])
        assert book.get_agent_orders(s1, 41) == []
        assert int(book.get_order(s1, 27, int(Side.BUY)).order_id) != 27


# ---------------------------------------------------------------------------
# F. Interaction with resting liquidity
# ---------------------------------------------------------------------------


class TestRestingInteraction:
    def test_wrong_sign_ice_does_not_touch_existing_iceberg(self, book):
        s = book.init()
        s, r = book.submit_order(
            s, _ice(Side.BUY, 45.18, 6.0, 1.5, -0.05, 312, agent=5))
        assert bool(r.accepted) is True
        n_before = int(s.n_active)
        assert n_before >= 1
        s2, r2 = book.submit_order(
            s, _ice(Side.BUY, 45.18, 6.0, 1.5, 0.05, 313, agent=6))
        _assert_rejected(r2)
        assert int(s2.n_active) == n_before
        assert _leaves_equal(s, s2)

    def test_wrong_sign_sell_ice_that_would_cross_makes_no_trade(self, book):
        s = book.init()
        s, r = book.submit_order(s, _plain(Side.BUY, 57.45, 3.0, 64))
        assert bool(r.accepted) is True
        bids_before, asks_before = book.get_anonymous_lob(s)
        s2, r2 = book.submit_order(
            s, _ice(Side.SELL, 57.10, 5.0, 1.0, -0.07, 65))
        _assert_rejected(r2)
        assert book.get_anonymous_lob(s2) == (bids_before, asks_before)
        assert np.array_equal(np.asarray(s.lob.trades), np.asarray(s2.lob.trades))


# ---------------------------------------------------------------------------
# G. Non-ICE PPD is not checked (it still trades)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("er", [ExecRestriction.NON, ExecRestriction.IOC,
                                ExecRestriction.FOK])
def test_non_ice_wrong_sign_ppd_still_trades(book, er):
    s = book.init()
    s, _ = book.submit_order(s, _plain(Side.SELL, 50.13, 2.0, 501))
    s2, res = book.submit_order(
        s, _plain(Side.BUY, 50.13, 2.0, 502, er=er, ppd=0.35))
    assert bool(res.accepted) is True
    assert bool(res.traded) is True
    assert float(res.trade_qty) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# H. Recompile pattern unchanged by traced PPD sign
# ---------------------------------------------------------------------------


def test_jitted_validator_single_compile_across_signs():
    fn = jax.jit(validate_order, static_argnums=(1, 2))
    outs = []
    for side, ppd in [(Side.BUY, 0.4), (Side.BUY, -0.4), (Side.SELL, -0.4),
                      (Side.SELL, 0.4), (Side.BUY, 0.0), (Side.SELL, 0.003)]:
        outs.append(bool(fn(_ice(side, 55.55, 3.3, 1.1, ppd, 9), PCFG, True)))
    assert outs == [False, True, False, True, True, True]
    assert fn._cache_size() == 1
