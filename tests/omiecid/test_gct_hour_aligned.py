"""Hour-aligned gate closure written by ``generate_contracts()``.

Behavioural contract (docs/omiecid/market-spec.md §3.1, v2.10 correction, and
§11.1; docs/market-model.md §"Trading window and gate closure"):

    GCT(k) = 60 * floor(k / 4) - 60   minutes from D 00:00

The four quarter-hours of a delivery hour close together, one hour before the
start of that hour. The per-quarter-hour rule (delivery_start - 60) agrees
only when k mod 4 == 0.

  (1) For all 96 contracts, gct_minutes == 60 * (k // 4) - 60.
  (2) Spot values from the spec's data evidence: k=3 -> -60, k=51 -> 660,
      k=92..95 -> 1320.
  (3) Contracts with k mod 4 == 0 keep the value delivery_start - 60.
  (4) Session state, fill reports and done are byte-identical whatever value
      Contract.gct_minutes carries: no package code reads the field (the
      session closes on its configured ``gct_minutes_from_got``).

Public interfaces only: omiecid_simulator.omiecid.contract (Contract,
generate_contracts) and omiecid_simulator.omiecid.omie_session (Session,
AgentOrder; zero-arg synthetic session, reset / step).

Device policy: CPU (pinned by tests/omiecid/conftest.py). Tiering: the session
invariance tests force a retrace and compile reset/step_env (a few seconds on
CPU), so they carry ``@pytest.mark.slow`` and ``@pytest.mark.jit``; the factory
tests are pure Python and run in microseconds.
"""

import dataclasses

import numpy as np
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus, generate_contracts


def _hour_aligned_gct(k: int) -> int:
    return 60 * (k // 4) - 60


@pytest.fixture(scope="module")
def contracts():
    return generate_contracts()


# ---------------------------------------------------------------------------
# (1) Formula for all 96 contracts
# ---------------------------------------------------------------------------

class TestAllContractsFollowHourAlignedFormula:
    """generate_contracts() writes GCT(k) = 60 * (k // 4) - 60 for every k."""

    def test_factory_returns_96_contracts(self, contracts):
        assert len(contracts) == 96

    def test_every_contract_gct_is_hour_aligned(self, contracts):
        wrong = [
            (c.contract_id, c.gct_minutes, _hour_aligned_gct(c.contract_id))
            for c in contracts
            if c.gct_minutes != _hour_aligned_gct(c.contract_id)
        ]
        assert not wrong, f"(k, got, expected) mismatches: {wrong}"

    @pytest.mark.parametrize("k", [k for k in range(96) if k % 4 != 0])
    def test_non_hour_start_contract_closes_with_its_hour(self, contracts, k):
        """Quarter-hours 2, 3, 4 of an hour share the GCT of the hour's first one."""
        first_of_hour = contracts[k - (k % 4)]
        assert contracts[k].gct_minutes == first_of_hour.gct_minutes

    @pytest.mark.parametrize("hour", range(24))
    def test_four_quarter_hours_of_an_hour_share_one_gct(self, contracts, hour):
        gcts = {contracts[4 * hour + q].gct_minutes for q in range(4)}
        assert gcts == {60 * hour - 60}

    def test_gct_is_one_hour_before_delivery_hour_start(self, contracts):
        """GCT = (start of the delivery hour containing k) - 60."""
        for c in contracts:
            hour_start = (c.delivery_start_minutes // 60) * 60
            assert c.gct_minutes == hour_start - 60, c.contract_id

    def test_gct_never_after_delivery_start_minus_60(self, contracts):
        """Hour-aligned closure is never later than the per-quarter rule."""
        for c in contracts:
            assert c.gct_minutes <= c.delivery_start_minutes - 60, c.contract_id

    def test_gct_offset_from_old_rule_is_0_15_30_or_45(self, contracts):
        for c in contracts:
            offset = (c.delivery_start_minutes - 60) - c.gct_minutes
            assert offset == 15 * (c.contract_id % 4), c.contract_id

    def test_gct_is_int(self, contracts):
        assert all(type(c.gct_minutes) is int for c in contracts)

    def test_other_fields_unchanged(self, contracts):
        """Only gct_minutes follows the new rule; ids, windows and status do not move."""
        for k, c in enumerate(contracts):
            assert c.contract_id == k
            assert c.delivery_start_minutes == 15 * k
            assert c.delivery_end_minutes == 15 * k + 15
            assert c.status == ContractStatus.PENDING


# ---------------------------------------------------------------------------
# (2) Spot values from the spec's data evidence
# ---------------------------------------------------------------------------

class TestSpotValues:
    """market-spec §3.1 v2.10 data evidence."""

    def test_contract_3_gct_is_minus_60(self, contracts):
        """k=3 (00:45-01:00) closes at 23:00 on D-1, with contracts 0-2."""
        assert contracts[3].gct_minutes == -60

    def test_contract_51_gct_is_660(self, contracts):
        """k=51 (12:45-13:00) closes at 11:00."""
        assert contracts[51].gct_minutes == 660

    @pytest.mark.parametrize("k", [92, 93, 94, 95])
    def test_last_hour_contracts_gct_is_1320(self, contracts, k):
        """k=92..95 (23:00-24:00) all close at 22:00."""
        assert contracts[k].gct_minutes == 1320

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_first_hour_contracts_gct_is_minus_60(self, contracts, k):
        assert contracts[k].gct_minutes == -60


# ---------------------------------------------------------------------------
# (3) k mod 4 == 0: unchanged from today's delivery_start - 60
# ---------------------------------------------------------------------------

class TestHourStartContractsUnchanged:

    @pytest.mark.parametrize("k", list(range(0, 96, 4)))
    def test_hour_start_contract_gct_equals_delivery_start_minus_60(self, contracts, k):
        c = contracts[k]
        assert c.gct_minutes == c.delivery_start_minutes - 60

    def test_contract_52_gct_is_720(self, contracts):
        """k=52 (13:00), the tutorial reference contract: both rules give 720."""
        assert contracts[52].gct_minutes == 720


# ---------------------------------------------------------------------------
# (4) Session unaffected by Contract.gct_minutes (no package reader)
# ---------------------------------------------------------------------------

try:  # guarded so the factory tests above still collect if the session regresses
    import jax
    from omiecid_simulator.omiecid.order_types import Side
    from omiecid_simulator.omiecid.omie_session import AgentOrder, Session
    _SESSION_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - collection guard
    _SESSION_IMPORT_ERROR = exc


_PERTURBATIONS = (7777, -1)  # wild, off-by-one


def _order_for(t, seed):
    """A fixed, non-trivial order schedule: idle steps, resting and crossing orders."""
    phase = (3 * t + seed) % 5
    if phase == 0:
        return None
    side = int(Side.BUY) if phase in (1, 3) else int(Side.SELL)
    # phase 1 / 2 rest inside the spread; phase 3 / 4 cross the synthetic ~50 mid.
    if phase == 1:
        price = 49.5
    elif phase == 2:
        price = 50.5
    elif phase == 3:
        price = 52.0
    else:
        price = 48.0
    return AgentOrder.create(side=side, price=price, quantity=1.0 + (t % 3))


def _rollout(session, seed):
    """Full session under a fixed order schedule.

    Returns host-side numpy copies of every (state, report, done) leaf, per
    step, plus the reset state.
    """
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    state = session.reset(k0)
    trace = [("reset", jax.tree_util.tree_map(np.asarray, state))]
    for t in range(int(session.steps_per_episode)):
        key, k = jax.random.split(key)
        state, report, done = session.step(k, state, _order_for(t, seed))
        trace.append(("step", jax.tree_util.tree_map(
            np.asarray, (state, report, done))))
    return trace


def _assert_traces_byte_identical(base, pert):
    assert len(base) == len(pert)
    for t, ((tag_a, a), (tag_b, b)) in enumerate(zip(base, pert)):
        assert tag_a == tag_b
        leaves_a, tree_a = jax.tree_util.tree_flatten(a)
        leaves_b, tree_b = jax.tree_util.tree_flatten(b)
        assert tree_a == tree_b, f"step {t}: pytree structure differs"
        for j, (x, y) in enumerate(zip(leaves_a, leaves_b)):
            assert x.dtype == y.dtype and x.shape == y.shape, (t, j)
            assert x.tobytes() == y.tobytes(), (
                f"step {t} leaf {j}: differs when Contract.gct_minutes is perturbed"
            )


def _clear_session_jit():
    """Drop Session's reset/step_env compile caches.

    Session is a static jit argument hashed by content (``__hash__`` /
    ``__eq__``), so a second ``Session()`` re-uses the first one's trace and
    would never build a Contract under the perturbation. Clearing forces a
    retrace, so the perturbed Contract really reaches the compiled code.
    """
    Session.reset.clear_cache()
    Session.step_env.clear_cache()


@pytest.fixture(scope="module")
def baseline_traces():
    if _SESSION_IMPORT_ERROR is not None:
        pytest.skip(f"Session not importable ({_SESSION_IMPORT_ERROR})")
    session = Session()
    return session, {seed: _rollout(session, seed) for seed in (0, 1)}


@pytest.fixture
def retrace(baseline_traces):
    """Force a retrace inside the test and leave no perturbed trace behind."""
    _clear_session_jit()
    yield
    _clear_session_jit()


def _install_gct_perturbation(monkeypatch, offset):
    """Every Contract built while installed carries gct_minutes + offset.

    Returns a call counter so the test can prove the session built a Contract
    under the perturbation (i.e. the perturbation reached its construction path).
    """
    calls = {"n": 0}
    original_init = Contract.__init__

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        calls["n"] += 1
        self.gct_minutes = int(self.gct_minutes) + offset

    monkeypatch.setattr(Contract, "__init__", _init)
    return calls


class TestSessionIgnoresContractGCT:
    """State, fill reports and done do not depend on Contract.gct_minutes.

    The session closes on its configured ``gct_minutes_from_got``; this pins that
    the hour-aligned fix (a value change of the field) cannot move any output.
    """

    def test_perturbation_mechanism_changes_the_field(self, monkeypatch):
        calls = _install_gct_perturbation(monkeypatch, 7777)
        c = Contract(contract_id=5, delivery_start_minutes=75,
                     delivery_end_minutes=90, gct_minutes=0,
                     status=ContractStatus.PENDING)
        assert c.gct_minutes == 7777 and calls["n"] == 1
        assert dataclasses.is_dataclass(c)

    # slow: each case retraces + compiles reset/step_env (a few seconds on CPU)
    @pytest.mark.slow
    @pytest.mark.jit
    @pytest.mark.parametrize("offset", _PERTURBATIONS)
    def test_rollout_byte_identical_under_gct_perturbation(
        self, baseline_traces, retrace, monkeypatch, offset
    ):
        _session0, base = baseline_traces
        calls = _install_gct_perturbation(monkeypatch, offset)
        session = Session()
        for seed, base_trace in base.items():
            pert_trace = _rollout(session, seed)
            _assert_traces_byte_identical(base_trace, pert_trace)
        assert calls["n"] > 0, (
            "the session built no Contract under the perturbation; the test "
            "would be vacuous"
        )

    @pytest.mark.slow  # slow: retrace + compile of reset/step_env (a few seconds on CPU)
    @pytest.mark.jit
    def test_rollout_byte_identical_when_session_built_before_perturbation(
        self, baseline_traces, retrace, monkeypatch
    ):
        """Perturbing only at reset/step time (session already constructed) also changes nothing."""
        session0, base = baseline_traces
        calls = _install_gct_perturbation(monkeypatch, -60_000)
        for seed, base_trace in base.items():
            _assert_traces_byte_identical(base_trace, _rollout(session0, seed))
        assert calls["n"] > 0, "no Contract built under the perturbation (vacuous)"

    @pytest.mark.slow  # slow: pays the baseline_traces compile (a few seconds on CPU)
    @pytest.mark.jit
    def test_baseline_rollout_reaches_done_and_trades(self, baseline_traces):
        """Non-vacuity: the compared session runs to done and the participant trades."""
        _session, base = baseline_traces
        for trace in base.values():
            _tag, (_state, _report, done) = trace[-1]
            assert bool(np.all(done)), "the compared session must reach done on its last step"
            submitted = [bool(np.asarray(r.submitted)) for tag, (_s, r, _d) in
                         [tr for tr in trace if tr[0] == "step"]]
            assert any(submitted), "the schedule must submit orders"
            final_state = trace[-1][1][0]
            assert np.any(np.asarray(final_state.agent_trades) != 0), (
                "the participant must trade at least once (fills exercise the book)"
            )
