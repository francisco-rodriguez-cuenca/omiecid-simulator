"""OMIE-CERT-HARNESS-2 (S8) -- locked behavioural contract.

Story card: ``docs/omiecid/agile-plan.md`` -> "Story OMIE-CERT-HARNESS-2" (v5.203),
test-first contract items (1), (2), (4), (5), (6). Item (3), the engine-replay
soundness guard, lives in ``tests/omiecid/test_omie_tape_synth_capacity.py``
(``test_bounds_still_sound``), next to the replay checker it reuses.

* (J3) ``omie_tape_synth.tape_capacity_bounds(tape)`` stops counting rows the
  market will reject (market-spec §5.4 range/side conditions; for ICE rows only:
  peak <= 0, peak > qty and the §13.6 PPD sign rule) in all four bounds, and adds
  the ``+ 2`` hop term only for a row crossable with at least one other counted
  real row. Where the generator cannot tell, the row is counted (upper bounds).
* (J6) ``cert_support.SHRUNK_FIXTURE_DIR`` = ``tests/omiecid/fixtures/cert``; shrunk
  counterexamples resolve there unless ``OMIE_CERT_FIXTURE_DIR`` overrides it.
  ``FIXTURE_DIR`` itself is unchanged (it serves fixtures that stay). The three
  ``C15_shrunk_*`` files are moved there byte-for-byte.
* (J7) ``cert_support.GATE_CELL_TIMEOUT_S`` maps scenario ids to a per-cell gate
  timeout (default 1,800 s; C16 at least 2 x 6,106 s), and the merge gate
  ``test_certified_notebooks.py::test_certified_notebook_still_passes`` hands the
  per-scenario value to ``execute_notebook``.

Everything here is pure numpy / file-system / a monkeypatched executor: no JAX
compile, no notebook execution. Device: CPU.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import types
from pathlib import Path

import pytest

from omiecid_simulator.jaxomiecid import omie_tape_synth as synth
from omiecid_simulator.omiecid.encoding import encode_price, encode_qty
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from notebooks.certification import cert_support as cs

REPO_ROOT = Path(__file__).resolve().parents[3]
NEW_SHRUNK_DIR = REPO_ROOT / "tests" / "omiecid" / "fixtures" / "cert"
OLD_FIXTURE_DIR = REPO_ROOT / "notebooks" / "certification" / "fixtures"
NB_DIR = REPO_ROOT / "notebooks" / "certification"

CFG = OMIEProductConfig()
PRICE_MIN_INT = encode_price(CFG.price_min, CFG)       # 0
PRICE_MAX_INT = encode_price(CFG.price_max, CFG)
QTY_MIN_INT = encode_qty(CFG.qty_min, CFG)
QTY_MAX_INT = encode_qty(CFG.qty_max, CFG)

BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, IOC, ICE = int(ExecRestriction.NON), int(ExecRestriction.IOC), int(ExecRestriction.ICE)

P_MID = encode_price(50.0, CFG)                        # a well-inside price

AXES = ("resting_per_side", "trades", "hops", "concurrent_icebergs")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _valid_pair(b: synth.TapeBuilder) -> None:
    """Two valid NON rows that cross: SELL 10 @ P_MID, then BUY 10 @ P_MID + 5."""
    b.add(side=SELL, price_int=P_MID, qty_int=10, exec_restriction=NON)
    b.add(side=BUY, price_int=P_MID + 5, qty_int=10, exec_restriction=NON)


def _bounds(b: synth.TapeBuilder) -> dict:
    return dict(synth.tape_capacity_bounds(b.build()))


def _pair_bounds() -> dict:
    b = synth.TapeBuilder(CFG)
    _valid_pair(b)
    return _bounds(b)


# One rejectable row each; every one would be crossable with the valid pair (or
# rests on a side) if counted, so counting it moves at least one bound.
REJECTABLE_ROWS = {
    # ICE: peak > qty (§4.2 / validator ICE gate)
    "ice_peak_above_qty": dict(side=SELL, price_int=P_MID - 2, qty_int=10,
                               exec_restriction=ICE, peak_int=20),
    # ICE: peak <= 0
    "ice_peak_zero": dict(side=SELL, price_int=P_MID - 2, qty_int=10,
                          exec_restriction=ICE, peak_int=0),
    "ice_peak_negative": dict(side=SELL, price_int=P_MID - 2, qty_int=10,
                              exec_restriction=ICE, peak_int=-3),
    # §5.4: price above price_max
    "price_above_max": dict(side=BUY, price_int=PRICE_MAX_INT + 1, qty_int=10,
                            exec_restriction=NON),
    # §5.4: price below price_min
    "price_below_min": dict(side=SELL, price_int=PRICE_MIN_INT - 1, qty_int=10,
                            exec_restriction=NON),
    # §5.4: qty above qty_max
    "qty_above_max": dict(side=BUY, price_int=P_MID + 5, qty_int=QTY_MAX_INT + 1,
                          exec_restriction=NON),
    # §5.4: qty below qty_min
    "qty_below_min": dict(side=BUY, price_int=P_MID + 5, qty_int=QTY_MIN_INT - 1,
                          exec_restriction=NON),
    # §13.6: BUY ICE(wPI) needs PPD <= 0
    "buy_ice_ppd_positive": dict(side=BUY, price_int=P_MID + 5, qty_int=30,
                                 exec_restriction=ICE, peak_int=10, ppd_int=3),
    # §13.6: SELL ICE(wPI) needs PPD >= 0
    "sell_ice_ppd_negative": dict(side=SELL, price_int=P_MID - 2, qty_int=30,
                                  exec_restriction=ICE, peak_int=10, ppd_int=-3),
}


# ---------------------------------------------------------------------------
# (1) rejectable rows are not counted
# ---------------------------------------------------------------------------
class TestRejectedRowsNotCounted:

    def test_rejected_rows_not_counted(self):
        """Card (1): 2 valid crossing rows + 3 rejectable rows (ICE peak > qty;
        price_int > encode(price_max); BUY ICE PPD > 0) bound exactly like the
        2-row tape."""
        b = synth.TapeBuilder(CFG)
        _valid_pair(b)
        b.add(**REJECTABLE_ROWS["ice_peak_above_qty"])
        b.add(**REJECTABLE_ROWS["price_above_max"])
        b.add(**REJECTABLE_ROWS["buy_ice_ppd_positive"])
        assert _bounds(b) == _pair_bounds()

    def test_rejected_rows_interleaved_before_the_valid_pair_not_counted(self):
        """Order of submission is irrelevant: rejectable rows first, then the pair."""
        b = synth.TapeBuilder(CFG)
        b.add(**REJECTABLE_ROWS["buy_ice_ppd_positive"])
        b.add(**REJECTABLE_ROWS["ice_peak_above_qty"])
        b.add(**REJECTABLE_ROWS["price_above_max"])
        _valid_pair(b)
        assert _bounds(b) == _pair_bounds()

    @pytest.mark.parametrize("name", sorted(REJECTABLE_ROWS))
    def test_each_rejection_condition_is_excluded_from_all_four_bounds(self, name):
        """One rejectable row added to the valid pair leaves all four bounds unchanged."""
        b = synth.TapeBuilder(CFG)
        _valid_pair(b)
        b.add(**REJECTABLE_ROWS[name])
        assert _bounds(b) == _pair_bounds(), name

    def test_invalid_side_row_is_not_counted(self):
        """§5.4 / validator side check: a side that is neither BUY nor SELL is rejectable."""
        b = synth.TapeBuilder(CFG)
        _valid_pair(b)
        b.add(side=7, price_int=P_MID, qty_int=10, exec_restriction=NON)
        assert _bounds(b) == _pair_bounds()

    def test_tape_of_only_rejectable_rows_has_all_bounds_zero(self):
        b = synth.TapeBuilder(CFG)
        for name in sorted(REJECTABLE_ROWS):
            b.add(**REJECTABLE_ROWS[name])
        assert _bounds(b) == {a: 0 for a in AXES}

    def test_rejectable_iceberg_not_counted_as_concurrent_iceberg(self):
        b = synth.TapeBuilder(CFG)
        _valid_pair(b)
        b.add(**REJECTABLE_ROWS["sell_ice_ppd_negative"])
        b.add(**REJECTABLE_ROWS["ice_peak_zero"])
        assert _bounds(b)["concurrent_icebergs"] == 0

    # --- guards: valid edge rows ARE counted (bounds must stay upper bounds) ---

    def test_boundary_valid_rows_are_still_counted(self):
        """Guard: price == encode(price_max), qty == qty_min and qty == qty_max are valid."""
        for row in (
            dict(side=BUY, price_int=PRICE_MAX_INT, qty_int=10, exec_restriction=NON),
            dict(side=SELL, price_int=PRICE_MIN_INT, qty_int=10, exec_restriction=NON),
            dict(side=BUY, price_int=P_MID + 5, qty_int=QTY_MIN_INT, exec_restriction=NON),
            dict(side=BUY, price_int=P_MID + 5, qty_int=QTY_MAX_INT, exec_restriction=NON),
        ):
            b = synth.TapeBuilder(CFG)
            _valid_pair(b)
            b.add(**row)
            assert _bounds(b) != _pair_bounds(), row

    def test_valid_iceberg_edges_are_still_counted(self):
        """Guard: ICE with peak == qty, BUY ICE PPD == 0, BUY ICE PPD < 0 and
        SELL ICE PPD > 0 are valid and counted as icebergs."""
        rows = (
            dict(side=SELL, price_int=P_MID - 2, qty_int=10, exec_restriction=ICE, peak_int=10),
            dict(side=BUY, price_int=P_MID + 5, qty_int=30, exec_restriction=ICE,
                 peak_int=10, ppd_int=0),
            dict(side=BUY, price_int=P_MID + 5, qty_int=30, exec_restriction=ICE,
                 peak_int=10, ppd_int=-3),
            dict(side=SELL, price_int=P_MID - 2, qty_int=30, exec_restriction=ICE,
                 peak_int=10, ppd_int=3),
        )
        for row in rows:
            b = synth.TapeBuilder(CFG)
            _valid_pair(b)
            b.add(**row)
            assert _bounds(b)["concurrent_icebergs"] == 1, row

    def test_peak_and_ppd_rules_apply_to_ice_rows_only(self):
        """Guard: a NON row carrying a peak > qty or a wrong-sign PPD is not rejectable
        on those grounds (the validator gates them on ICE only), so it is counted."""
        for row in (
            dict(side=BUY, price_int=P_MID + 5, qty_int=10, exec_restriction=NON, peak_int=20),
            dict(side=BUY, price_int=P_MID + 5, qty_int=10, exec_restriction=NON, ppd_int=3),
        ):
            b = synth.TapeBuilder(CFG)
            _valid_pair(b)
            b.add(**row)
            got = _bounds(b)
            assert got["resting_per_side"] == 2, row
            assert got["trades"] > _pair_bounds()["trades"], row

    def test_generator_does_not_import_the_engine_validator(self):
        """Rejected alternative R1: the predicate is spec-derived, not imported."""
        tree = ast.parse(Path(synth.__file__).read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
                mods.update(f"{node.module}.{a.name}" for a in node.names)
        assert not any("order_validator" in m for m in mods), sorted(mods)


# ---------------------------------------------------------------------------
# (2) the + 2 hop term needs a crossable counterparty
# ---------------------------------------------------------------------------
class TestNoCounterpartyNoPlusTwo:

    def test_no_counterparty_no_plus_two(self):
        """Card (2): a single non-crossable NON row: hops == units (== 1)."""
        b = synth.TapeBuilder(CFG)
        b.add(side=SELL, price_int=P_MID, qty_int=10, exec_restriction=NON)
        assert _bounds(b)["hops"] == 1

    @pytest.mark.parametrize("side", [BUY, SELL])
    def test_single_row_either_side(self, side):
        b = synth.TapeBuilder(CFG)
        b.add(side=side, price_int=P_MID, qty_int=10, exec_restriction=NON)
        assert _bounds(b)["hops"] == 1

    def test_lone_iceberg_hops_equal_its_slice_count(self):
        """A lone ICE 35 / peak 10 has 4 units and no counterparty: hops == 4."""
        b = synth.TapeBuilder(CFG)
        b.add(side=BUY, price_int=P_MID, qty_int=35, exec_restriction=ICE, peak_int=10)
        assert _bounds(b)["hops"] == 4

    def test_same_side_rows_never_get_the_plus_two(self):
        b = synth.TapeBuilder(CFG)
        for k in range(3):
            b.add(side=BUY, price_int=P_MID + k, qty_int=10, exec_restriction=NON)
        assert _bounds(b)["hops"] == 1

    def test_opposite_rows_that_do_not_cross_by_price_get_no_plus_two(self):
        """BUY @ P_MID - 10 and SELL @ P_MID: opposite sides, never cross."""
        b = synth.TapeBuilder(CFG)
        b.add(side=BUY, price_int=P_MID - 10, qty_int=10, exec_restriction=NON)
        b.add(side=SELL, price_int=P_MID, qty_int=10, exec_restriction=NON)
        assert _bounds(b)["hops"] == 1

    def test_only_counterparty_is_rejectable_no_plus_two(self):
        """The lone valid row's only crossable partner is rejectable (not counted)."""
        b = synth.TapeBuilder(CFG)
        b.add(side=SELL, price_int=P_MID, qty_int=10, exec_restriction=NON)
        b.add(side=BUY, price_int=PRICE_MAX_INT + 1, qty_int=10, exec_restriction=NON)
        assert _bounds(b)["hops"] == 1

    def test_crossable_pair_keeps_the_plus_two(self):
        """Guard: the second row of a crossing NON pair crosses 1 earlier unit:
        hops == 1 + 1 + 2 == 4 (unchanged from main)."""
        assert _pair_bounds()["hops"] == 4

    def test_crossable_with_a_later_row_keeps_the_plus_two(self):
        """Guard: the crossability test is the trades one (any other row, earlier
        or later). A resting ICE 35 / peak 10 (4 units) followed by a crossing
        BUY NON: the iceberg's own term is 0 + 4 + 2 = 6; the buy's is 4 + 1 + 2 = 7."""
        b = synth.TapeBuilder(CFG)
        b.add(side=SELL, price_int=P_MID, qty_int=35, exec_restriction=ICE, peak_int=10)
        b.add(side=BUY, price_int=P_MID + 1, qty_int=1, exec_restriction=NON)
        assert _bounds(b)["hops"] == 7

    def test_pair_bounds_unchanged_guard(self):
        """Guard: the valid crossing pair bounds as on main (1 resting per side,
        2 trade units, hops 4, no icebergs)."""
        assert _pair_bounds() == {"resting_per_side": 1, "trades": 2, "hops": 4,
                                  "concurrent_icebergs": 0}


# ---------------------------------------------------------------------------
# (4) shrunk fixture directory
# ---------------------------------------------------------------------------
class TestShrunkFixtureDir:

    def test_shrunk_fixture_dir_constant(self):
        assert Path(cs.SHRUNK_FIXTURE_DIR).resolve() == NEW_SHRUNK_DIR.resolve()

    def test_shrunk_fixture_dir(self, monkeypatch):
        """Card (4): unset env -> tests/omiecid/fixtures/cert/; set -> the env dir."""
        monkeypatch.delenv(cs.FIXTURE_ENV, raising=False)
        got = cs._env_dir(cs.FIXTURE_ENV, cs.SHRUNK_FIXTURE_DIR) / "C15_shrunk_seed1.npz"
        assert got.resolve().parent == NEW_SHRUNK_DIR.resolve()

    def test_shrunk_fixture_dir_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(cs.FIXTURE_ENV, str(tmp_path))
        got = cs._env_dir(cs.FIXTURE_ENV, cs.SHRUNK_FIXTURE_DIR) / "C15_shrunk_seed1.npz"
        assert got.resolve().parent == tmp_path.resolve()

    def test_fixture_env_name_unchanged(self):
        assert cs.FIXTURE_ENV == "OMIE_CERT_FIXTURE_DIR"

    def test_general_fixture_dir_not_repointed(self):
        """Guard: FIXTURE_DIR still serves the fixtures that stay (C04/C05/C10/C13/C14/C16)."""
        assert Path(cs.FIXTURE_DIR).resolve() == OLD_FIXTURE_DIR.resolve()


# ---------------------------------------------------------------------------
# (5) the three C15 shrunk fixtures are relocated byte-for-byte
# ---------------------------------------------------------------------------
C15_SHRUNK_SHA256 = {
    "C15_shrunk_seed100000244.npz":
        "7b4b7730dbc0bc08ecb481485637fd462dc7106487e446e616252e72772fb8b7",
    "C15_shrunk_seed100000246.npz":
        "ad6b74689d51f8b136bd3a0ed867059b4daabcc52b64902cfeeb8d6d48ed92f2",
    "C15_shrunk_seed100000249.npz":
        "97116805ec0ef62fbb6ffc0a84365970b75068a37ac364bbe2a5603ac58c1293",
}

STAYING_FIXTURES = (
    "C04_seed100_iceberg_no_cancel.npz", "C05_seed0.npz", "C05_seed0_same_second.npz",
    "C10_seed2_req74.npz", "C13_seed96.npz", "C14_last_row_lookup_crossing_non.npz",
    "C14_last_row_lookup_iceberg_stranded.npz", "C14_price_priority_two_buy_icebergs.npz",
    "C14_seed300000.npz", "C15_seed100000244.npz", "C16_seed112_contract10.npz",
    "C16_seed112.npz",
)


class TestC15ShrunkFixturesRelocated:

    @pytest.mark.parametrize("name", sorted(C15_SHRUNK_SHA256))
    def test_c15_shrunk_fixtures_relocated(self, name):
        """Card (5): present under tests/omiecid/fixtures/cert/, absent from
        notebooks/certification/fixtures/, bytes unchanged (sha256 pinned)."""
        new = NEW_SHRUNK_DIR / name
        assert new.is_file(), f"missing: {new}"
        assert not (OLD_FIXTURE_DIR / name).exists(), f"still at old path: {name}"
        assert hashlib.sha256(new.read_bytes()).hexdigest() == C15_SHRUNK_SHA256[name]

    @pytest.mark.parametrize("name", STAYING_FIXTURES)
    def test_other_fixtures_stay_put(self, name):
        """Guard: only the three C15_shrunk_* files move."""
        assert (OLD_FIXTURE_DIR / name).is_file(), name
        assert not (NEW_SHRUNK_DIR / name).exists(), name


# ---------------------------------------------------------------------------
# (6) per-scenario gate timeout
# ---------------------------------------------------------------------------
def _gate_module():
    from tests.omiecid.certification import test_certified_notebooks as gate
    return gate


def _record_gate_call(monkeypatch, notebook: Path) -> dict:
    """Run the merge-gate test body against a recording stand-in for execute_notebook."""
    calls = []

    def fake_execute_notebook(path, **kwargs):
        calls.append({"path": Path(path), **kwargs})
        return types.SimpleNamespace(passed=True, executed=True, results={"status": "pass"},
                                     results_errors=[], error=None)

    monkeypatch.setattr(cs, "execute_notebook", fake_execute_notebook)
    _gate_module().test_certified_notebook_still_passes(notebook, None)
    assert len(calls) == 1
    return calls[0]


class TestGateTimeoutPerScenario:

    def test_gate_timeout_per_scenario(self, monkeypatch):
        """Card (6): C16's timeout is at least 2 x the measured 6,106 s, and the gate
        passes it to execute_notebook."""
        assert int(cs.GATE_CELL_TIMEOUT_S["C16"]) >= 2 * 6106
        call = _record_gate_call(monkeypatch, NB_DIR / "C16_multi_contract_isolation.ipynb")
        assert call.get("timeout") == cs.GATE_CELL_TIMEOUT_S["C16"]

    def test_c16_timeout_is_the_designed_value(self):
        """Design decision (d): C16 = 12,600 s (about 2 x 6,106 s)."""
        assert int(cs.GATE_CELL_TIMEOUT_S["C16"]) == 12600

    @pytest.mark.parametrize("nb_name", ["C00_inertness_and_status.ipynb",
                                         "C15_randomised_fuzz.ipynb"])
    def test_other_scenarios_keep_the_default_1800(self, monkeypatch, nb_name):
        """Rejected alternative R2: no global raise; scenarios not in the map run at 1,800 s."""
        call = _record_gate_call(monkeypatch, NB_DIR / nb_name)
        assert int(call.get("timeout", 1800)) == 1800

    def test_gate_timeout_map_is_a_mapping_of_scenario_ids_to_positive_ints(self):
        m = cs.GATE_CELL_TIMEOUT_S
        assert "C16" in m
        for k in m:
            assert isinstance(k, str) and k.startswith("C"), k
            assert int(m[k]) > 0, k

    def test_execute_notebook_default_timeout_unchanged(self):
        """Guard: the global per-cell default stays 1,800 s (R2 rejected)."""
        sig = inspect.signature(cs.execute_notebook)
        assert sig.parameters["timeout"].default == 1800

    def test_gate_still_passes_the_notebook_path(self, monkeypatch):
        nb = NB_DIR / "C16_multi_contract_isolation.ipynb"
        call = _record_gate_call(monkeypatch, nb)
        assert call["path"] == nb

