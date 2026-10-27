"""Engine-confirmed capacity bounds for the synthetic tape generator (S7).

Story card: ``docs/omiecid/agile-plan.md`` -> "Story OMIE-CERT-SYNTH-1", decision S7
("Engine-confirmed test") and AC (11). Sibling of ``test_omie_tape_synth.py``.

Every registry key (seeds 0-1), plus ``C15.random_session`` on every length-160 fuzz
cell with ``cross_prob`` 0.6 (seed 0), is replayed **row by row** through a
``ContractBook`` on an OPEN contract, on CPU. After every submission the realised
resting rows per side and the trade-ring occupancy are read off the book state.
The test asserts **bounds only, never outcomes** (outcomes are certification):

* realised trade-ring rows   <= ``capacity_bounds["trades"]``
* realised resting rows/side <= ``capacity_bounds["resting_per_side"]`` (at every step)
* ``book_full`` is False and ``truncated`` is False at the end of the replay

The parametrisation reads ``CONSTRUCTIONS`` when the module imports, so a mode added
later by a certification card (S2) is picked up without editing this file. The
archived generator fails this test (its C15 bounds were estimates: 91, 84 and 100
realised trade rows against a claimed <= 80), which is the check seen failing; the
self-test at the bottom shows the same checker flagging an under-stated bound on a
hand-built tape, without the generator.

All tests here are ``slow`` (they replay through the jitted submit path; the first
call pays the compile) and ``jit``.
"""

from __future__ import annotations

import math
import types

import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat, TradesFeat
from omiecid_simulator.jaxomiecid.omie_data_loader import row_to_order
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, Status

try:  # guarded: the file must collect before the module exists
    from omiecid_simulator.jaxomiecid import omie_tape_synth as synth
    _IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - exercised only pre-implementation
    synth = None
    _IMPORT_ERROR = repr(_exc)

pytestmark = [pytest.mark.slow, pytest.mark.jit]

PRODUCT = OMIEProductConfig()
MARKET = OMIEMarketConfig()
C_OID = 4

S1_KEYS = (
    "C00.empty", "C00.pad_only", "C00.status_gate", "C01.validation",
    "C02.crossing_pair", "C03.priority_ladder", "C03.same_owner_pair",
    "C04.cancellation", "C04.iceberg_cancel_fok", "C05.aggressor_residual",
    "C05.resting_residual", "C06.multi_level_walk", "C06.multi_level_walk_stress",
    "C07.ioc_regimes", "C08.iceberg_cascade", "C08.rule3", "C08.incoming_iceberg",
    "C09.icewpi_drift", "C09.icewpi_wrong_sign", "C10.fok_visible",
    "C10.fok_hidden", "C10.fok_icewpi", "C15.random_session",
)


def S():
    if synth is None:
        pytest.fail(
            "omiecid_simulator.jaxomiecid.omie_tape_synth is not importable "
            f"({_IMPORT_ERROR}); OMIE-CERT-SYNTH-1 is not implemented yet."
        )
    return synth


def _registry_keys():
    if synth is not None and hasattr(synth, "CONSTRUCTIONS"):
        return tuple(sorted(synth.CONSTRUCTIONS))
    return tuple(sorted(S1_KEYS))


@pytest.fixture(scope="module")
def book():
    contract = Contract(
        contract_id=0, delivery_start_minutes=600, delivery_end_minutes=615,
        gct_minutes=540, status=ContractStatus.OPEN,
    )
    return ContractBook(contract, PRODUCT, MARKET)


def _resting(state):
    p = OrderSideFeat.P.value
    return (int(jnp.sum(state.lob.bids[:, p] >= 0)),
            int(jnp.sum(state.lob.asks[:, p] >= 0)))


def _trade_rows(state):
    return int(jnp.sum(state.lob.trades[:, TradesFeat.PASS_OID.value] != -1))


def _engine_findings(book, draw) -> list:
    """Replay ``draw.tape``'s real rows; report every realised bound breach."""
    tape = np.asarray(draw.tape)
    bounds = draw.capacity_bounds
    state = book.init()
    peak_bid = peak_ask = 0
    for row in tape[tape[:, C_OID] >= 1]:
        state, _ = book.submit_order(state, row_to_order(row, PRODUCT))
        b, a = _resting(state)
        peak_bid, peak_ask = max(peak_bid, b), max(peak_ask, a)
    f = []
    n_tr = _trade_rows(state)
    if n_tr > int(bounds["trades"]):
        f.append(f"realised trade rows {n_tr} > bound {bounds['trades']}")
    if max(peak_bid, peak_ask) > int(bounds["resting_per_side"]):
        f.append(f"realised resting rows bid={peak_bid} ask={peak_ask} > bound "
                 f"{bounds['resting_per_side']}")
    if bool(state.book_full):
        f.append("book_full raised")
    if bool(state.truncated):
        f.append("truncated raised")
    return f


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("key", _registry_keys())
def test_engine_confirms_the_capacity_bounds_of_every_registered_key(book, key, seed):
    """slow: replays through the jitted ContractBook submit path (compile on first call)."""
    draw = S().CONSTRUCTIONS[key](seed)
    assert _engine_findings(book, draw) == [], (key, seed)


def _c15_cells():
    if synth is None:
        return [pytest.param(None, id="module-missing")]
    cells = [r for r in synth.fuzz_grid()
             if r.n_orders == 160 and math.isclose(r.cross_prob, 0.6)]
    return [pytest.param(r, id=f"cell{i}") for i, r in enumerate(cells)]


@pytest.mark.parametrize("regime", _c15_cells())
def test_engine_confirms_c15_bounds_on_every_160_order_high_cross_cell(book, regime):
    """slow: 160-order fuzz session replayed row by row (the archive breached here)."""
    s = S()
    if regime is None:
        pytest.fail("fuzz grid unavailable")
    draw = s.make_random_session(0, regime)
    assert draw.capacity_bounds["trades"] <= s.DEFAULT_ENVELOPE.max_trades
    assert _engine_findings(book, draw) == [], regime


def test_c15_cells_cover_every_mix_at_both_tie_densities():
    """The selection above is non-empty: 5 mixes x 2 tie densities."""
    cells = [r for r in S().fuzz_grid()
             if r.n_orders == 160 and math.isclose(r.cross_prob, 0.6)]
    assert len(cells) == 10


def _hand_row(oid, ts, side, price_int, qty_int):
    return [0, int(side), price_int, qty_int, oid, 1, ts, int(ExecRestriction.NON),
            0, 0, int(Status.ACTIVE), 0, 0]


def test_engine_checker_flags_an_understated_bound(book):
    """Rule 1: five asks swept by one buy -> 5 trade rows, 5 resting asks at peak.

    Built by hand (no generator), so it runs before the module exists. The true
    bounds pass; each bound under-stated by one is flagged.
    """
    rows = [_hand_row(i + 1, i + 1, Side.SELL, 1_005_000 + i, 10) for i in range(5)]
    rows.append(_hand_row(6, 6, Side.BUY, 1_005_010, 50))
    tape = np.asarray(rows, dtype=np.int32)
    ok = {"resting_per_side": 5, "trades": 5, "hops": 9, "concurrent_icebergs": 0}
    assert _engine_findings(book, types.SimpleNamespace(tape=tape, capacity_bounds=ok)) == []
    for axis in ("trades", "resting_per_side"):
        bad = dict(ok, **{axis: ok[axis] - 1})
        found = _engine_findings(book, types.SimpleNamespace(tape=tape, capacity_bounds=bad))
        assert found != [], axis


# ---------------------------------------------------------------------------
# OMIE-CERT-HARNESS-2 (S8), test-first contract (3): guard
# ---------------------------------------------------------------------------
SOUNDNESS_SEEDS = tuple(range(20))


@pytest.mark.parametrize("key", _registry_keys())
def test_bounds_still_sound(book, key):
    """slow: 20 draws per registry key replayed row by row through the jitted submit path.

    Guard for OMIE-CERT-HARNESS-2: once the generator stops counting rejectable rows
    and drops the ``+ 2`` hop term for rows without a crossable counterparty, the
    engine's realised peaks (trade rows, resting rows per side) must still stay within
    the tightened bounds, with neither ``book_full`` nor ``truncated`` raised.
    """
    s = S()
    bad = {}
    for seed in SOUNDNESS_SEEDS:
        found = _engine_findings(book, s.CONSTRUCTIONS[key](seed))
        if found:
            bad[seed] = found
    assert bad == {}, (key, bad)
