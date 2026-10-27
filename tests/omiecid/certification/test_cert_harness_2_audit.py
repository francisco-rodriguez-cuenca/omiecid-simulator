"""Step-5 audit for OMIE-CERT-HARNESS-2 (S8): generator bounds, C15 fixture move, gate timeout.

Derived independently from the story card (``docs/omiecid/agile-plan.md``, "Story
OMIE-CERT-HARNESS-2"), the capacity-validity precondition (``docs/omiecid/scenario-ladder.md``
v2.3) and ``market-spec.md`` §5.4 / §13.6. The story's locked test file was not read.

Groups:

* ``TestRejectablePredicate`` / ``TestBoundsExcludeRejectableRows`` /
  ``TestHopPlusTwoOnlyWhenCrossable``: pure numpy probes of ``tape_capacity_bounds``
  on hand-built tapes (fast).
* ``TestEngineAgreesWithPredicate`` / ``TestBoundsSoundOnEngine``: hand-built and
  seeded random tapes replayed through ``ContractBook`` (slow + jit: the jitted submit
  path compiles on first use). The predicate must never call a row rejectable that the
  engine accepts (that would make a bound unsound), and the engine's realised trade
  rows, resting rows per side and live icebergs must stay within the bounds.
* ``TestShrunkFixtureDir`` / ``TestGateTimeout``: ``cert_support`` wiring (fast; the
  nbclient client is replaced by a recorder, so no kernel starts).
* ``TestRelocatedC15Fixtures``: the three moved files load and are byte-identical to the
  git blobs recorded on ``main`` before the move (fast), and their real rows replay
  within their bounds (slow + jit).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid import omie_tape_synth as synth
from omiecid_simulator.omiecid.encoding import encode_price, encode_qty
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCT = OMIEProductConfig()

BUY, SELL = 0, 1
NON, IOC, FOK, ICE = 0, 1, 2, 3

P_MIN = encode_price(PRODUCT.price_min, PRODUCT)
P_MAX = encode_price(PRODUCT.price_max, PRODUCT)
Q_MIN = encode_qty(PRODUCT.qty_min, PRODUCT)
Q_MAX = encode_qty(PRODUCT.qty_max, PRODUCT)
MID = encode_price(37.25, PRODUCT)  # incidental mid price, deliberately not 50.00
AXES = ("resting_per_side", "trades", "hops", "concurrent_icebergs")


def row(oid, side, price, qty, ex=NON, peak=0, ppd=0, ts=None, contract=0):
    """One 13-wide int32 tape row in the loader's column order."""
    ts = oid if ts is None else ts
    return [contract, side, price, qty, oid, 100 + oid, ts, ex, 0, 0, 0, peak, ppd]


def tape(*rows, pad=0):
    t = [list(r) for r in rows] + [[0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0]] * pad
    return np.asarray(t, dtype=np.int32).reshape(-1, 13)


def bounds(t):
    return dict(synth.tape_capacity_bounds(t))


ZERO = {a: 0 for a in AXES}

# Rejectable rows (market-spec §5.4; §13.6 sign rule), one per condition.
REJECTABLE = {
    "price_above_max": row(91, SELL, P_MAX + 1, 5),
    "price_below_min": row(92, BUY, P_MIN - 1, 5),
    "qty_zero": row(93, BUY, MID, 0),
    "qty_above_max": row(94, SELL, MID, Q_MAX + 1),
    "ice_peak_zero": row(95, SELL, MID, 30, ICE, 0, 0),
    "ice_peak_negative": row(96, BUY, MID, 30, ICE, -4, 0),
    "ice_peak_above_qty": row(97, SELL, MID, 30, ICE, 31, 0),
    "ice_buy_ppd_positive": row(98, BUY, MID, 30, ICE, 5, 3),
    "ice_sell_ppd_negative": row(99, SELL, MID, 30, ICE, 5, -3),
}

# Rows on the accepted side of every boundary.
VALID_EDGE = {
    "price_at_max": row(81, SELL, P_MAX, 5),
    "price_at_min": row(82, BUY, P_MIN, 5),
    "qty_at_min": row(83, BUY, MID, Q_MIN),
    "qty_at_max": row(84, SELL, MID, Q_MAX),
    "ice_peak_equals_qty": row(85, SELL, MID, 30, ICE, 30, 0),
    "ice_buy_ppd_zero": row(86, BUY, MID, 30, ICE, 7, 0),
    "ice_sell_ppd_zero": row(87, SELL, MID, 30, ICE, 7, 0),
    "ice_buy_ppd_negative": row(88, BUY, MID, 30, ICE, 7, -2),
    "ice_sell_ppd_positive": row(89, SELL, MID, 30, ICE, 7, 2),
    # The ICE-only checks must not leak onto NON/IOC/FOK rows (validator gates them on ICE).
    "non_with_junk_peak_and_ppd": row(80, BUY, MID, 5, NON, 99, 9),
    "ioc_with_negative_peak": row(79, SELL, MID, 5, IOC, -1, -9),
    "fok_with_wrong_sign_ppd": row(78, BUY, MID, 5, FOK, 0, 4),
}


# ===========================================================================
# (a) the predicate, pure
# ===========================================================================
class TestRejectablePredicate:
    @pytest.mark.parametrize("name", sorted(REJECTABLE))
    def test_spec_rejection_conditions_are_rejectable(self, name):
        assert synth._row_is_rejectable(np.asarray(REJECTABLE[name], np.int32)) is True

    @pytest.mark.parametrize("name", sorted(VALID_EDGE))
    def test_boundary_valid_rows_are_not_rejectable(self, name):
        assert synth._row_is_rejectable(np.asarray(VALID_EDGE[name], np.int32)) is False

    @pytest.mark.parametrize("side", [2, -1, 7])
    def test_invalid_side_is_rejectable(self, side):
        assert synth._row_is_rejectable(np.asarray(row(1, side, MID, 5), np.int32)) is True

    def test_config_argument_is_honoured(self):
        """The range checks read the given product config, not a hard-coded default."""
        tight = OMIEProductConfig(price_max=40.0)
        r = np.asarray(row(1, BUY, encode_price(41.0, tight), 5), np.int32)
        assert synth._row_is_rejectable(r, tight) is True
        assert synth._row_is_rejectable(r) is False

    def test_generator_does_not_import_the_engine_validator(self):
        src = Path(synth.__file__).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+\S*order_validator", src, re.M)


# ===========================================================================
# (a) rejectable rows are excluded from all four bounds
# ===========================================================================
class TestBoundsExcludeRejectableRows:
    BASE = (row(1, SELL, MID, 4, ICE, 2, 0), row(2, BUY, MID + 3, 4))

    @pytest.mark.parametrize("name", sorted(REJECTABLE))
    def test_appending_a_rejectable_row_changes_no_bound(self, name):
        assert bounds(tape(*self.BASE, REJECTABLE[name])) == bounds(tape(*self.BASE))

    @pytest.mark.parametrize("name", sorted(REJECTABLE))
    def test_prepending_a_rejectable_row_changes_no_bound(self, name):
        """An earlier rejected order must not feed the later row's hop sum either."""
        assert bounds(tape(REJECTABLE[name], *self.BASE)) == bounds(tape(*self.BASE))

    def test_all_rejectable_rows_together_change_no_bound(self):
        rej = [REJECTABLE[k] for k in sorted(REJECTABLE)]
        mixed = tape(rej[0], self.BASE[0], *rej[1:5], self.BASE[1], *rej[5:])
        assert bounds(mixed) == bounds(tape(*self.BASE))

    def test_only_rejectable_rows_bound_to_zero(self):
        assert bounds(tape(*REJECTABLE.values(), pad=3)) == ZERO

    def test_rejectable_counterparty_gives_no_trades_and_no_plus_two(self):
        """A valid NON whose only crossing counterparty is rejectable: no trade, hops = 1."""
        t = tape(row(1, BUY, MID, 5), row(2, SELL, MID - 10, 5, ICE, 1, -3))  # wrong-sign SELL ICE
        assert bounds(t) == {"resting_per_side": 1, "trades": 0, "hops": 1,
                             "concurrent_icebergs": 0}

    def test_rejected_ice_not_counted_as_iceberg_or_resting(self):
        t = tape(row(1, BUY, MID, 3), REJECTABLE["ice_peak_above_qty"],
                 REJECTABLE["ice_sell_ppd_negative"])
        b = bounds(t)
        assert b["concurrent_icebergs"] == 0 and b["resting_per_side"] == 1

    @pytest.mark.parametrize("name", sorted(VALID_EDGE))
    def test_boundary_valid_rows_are_still_counted(self, name):
        """Guard: a valid row adds to at least one axis (bounds stay upper bounds)."""
        r = VALID_EDGE[name]
        t_with = tape(*self.BASE, r)
        b0, b1 = bounds(tape(*self.BASE)), bounds(t_with)
        assert all(b1[a] >= b0[a] for a in AXES)
        assert b1 != b0 or r[7] in (IOC, FOK)  # IOC/FOK rows never rest, may not cross here
        if r[7] == ICE:
            assert b1["concurrent_icebergs"] == b0["concurrent_icebergs"] + 1

    def test_input_tape_not_mutated(self):
        t = tape(*self.BASE, *REJECTABLE.values(), pad=2)
        before = t.copy()
        bounds(t)
        np.testing.assert_array_equal(t, before)


# ===========================================================================
# (b) the + 2 hop term only for crossable rows
# ===========================================================================
class TestHopPlusTwoOnlyWhenCrossable:
    @pytest.mark.parametrize("total,peak", [(1, 1), (7, 7), (9, 2), (40, 1)])
    def test_lone_iceberg_hops_equal_its_units(self, total, peak):
        b = bounds(tape(row(1, SELL, MID, total, ICE, peak, 0)))
        assert b["hops"] == -(-total // peak)
        assert b["trades"] == 0

    def test_same_side_rows_never_get_plus_two(self):
        t = tape(row(1, BUY, MID, 5), row(2, BUY, MID + 50, 9, ICE, 3, -1), row(3, BUY, MID - 1, 2))
        assert bounds(t)["hops"] == 3  # the iceberg's own units only

    def test_opposite_rows_that_do_not_cross_get_no_plus_two(self):
        t = tape(row(1, SELL, MID + 1, 5, ICE, 1, 0), row(2, BUY, MID, 4, ICE, 2, 0))
        assert bounds(t) == {"resting_per_side": 1, "trades": 0, "hops": 5,
                             "concurrent_icebergs": 2}

    def test_icewpi_drifting_away_does_not_create_a_counterparty(self):
        """A valid BUY ICEwPI (PPD < 0) drifts down; its reach is its first price."""
        t = tape(row(1, BUY, MID, 10, ICE, 1, -5), row(2, SELL, MID + 1, 3))
        assert bounds(t)["trades"] == 0 and bounds(t)["hops"] == 10

    def test_crossing_pair_gets_plus_two_on_both_rows(self):
        """Crossable with a LATER row counts too: the first row gets units + 2."""
        t = tape(row(1, SELL, MID, 6, ICE, 1, 0), row(2, BUY, MID, 1))
        # row 1: 6 + 2; row 2: 6 (earlier crossable units) + 1 + 2
        assert bounds(t)["hops"] == 9
        t_first_only = tape(row(1, SELL, MID, 6, ICE, 1, 0), row(2, BUY, MID - 1, 1))
        assert bounds(t_first_only)["hops"] == 6

    def test_plus_two_restored_when_a_valid_counterparty_is_added(self):
        lone = tape(row(1, BUY, MID, 4, ICE, 1, 0))
        paired = tape(row(1, BUY, MID, 4, ICE, 1, 0), row(2, SELL, MID - 2, 1, IOC))
        assert bounds(lone)["hops"] == 4
        assert bounds(paired)["hops"] == 4 + 1 + 2


# ===========================================================================
# (a)/(b) soundness against the engine
# ===========================================================================
def _book(market=None):
    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.contract_book import ContractBook
    c = Contract(contract_id=0, delivery_start_minutes=600, delivery_end_minutes=615,
                 gct_minutes=540, status=ContractStatus.OPEN)
    return ContractBook(c, PRODUCT, market or OMIEMarketConfig())


def _replay(book, t):
    """Replay real rows; return (accepted flags, realised peaks, final state)."""
    import jax.numpy as jnp
    from omiecid_simulator.jaxob.jaxob_constants import OrderSideFeat, TradesFeat
    from omiecid_simulator.jaxomiecid.omie_data_loader import row_to_order
    p = OrderSideFeat.P.value
    st = book.init()
    acc, peak_rest, peak_ice = [], 0, 0
    real = t[t[:, 4] >= 1]
    for r in real:
        st, res = book.submit_order(st, row_to_order(r, PRODUCT))
        acc.append(bool(res.accepted))
        peak_rest = max(peak_rest, int(jnp.sum(st.lob.bids[:, p] >= 0)),
                        int(jnp.sum(st.lob.asks[:, p] >= 0)))
        peak_ice = max(peak_ice, int(st.n_active))
    n_trades = int(jnp.sum(st.lob.trades[:, TradesFeat.PASS_OID.value] != -1))
    return acc, {"resting_per_side": peak_rest, "trades": n_trades,
                 "concurrent_icebergs": peak_ice}, st


def _breaches(t, realised, st):
    b = bounds(t)
    out = [f"{a}: realised {realised[a]} > bound {b[a]}" for a in realised if realised[a] > b[a]]
    if bool(st.truncated):
        out.append("truncated")
    if bool(st.book_full):
        out.append("book_full")
    return out


@pytest.fixture(scope="module")
def default_book():
    return _book()


def _random_tape(seed, n=10):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        side = int(rng.integers(0, 2))
        price = MID + int(rng.integers(-6, 7))
        ex = int(rng.choice([NON, NON, IOC, FOK, ICE, ICE]))
        qty = int(rng.integers(1, 25))
        peak = ppd = 0
        if ex == ICE:
            peak = int(rng.integers(1, qty + 1))
            ppd = int(rng.integers(0, 3)) * (-1 if side == BUY else 1)
        r = row(i + 1, side, price, qty, ex, peak, ppd)
        kind = rng.random()
        if kind < 0.35:  # turn the row into a rejectable one that would otherwise trade
            fault = int(rng.integers(0, 5))
            if fault == 0:
                r[2] = P_MAX + 1 + int(rng.integers(0, 50)) if side == BUY else P_MIN - 1
            elif fault == 1:
                r[3] = Q_MAX + 1 + int(rng.integers(0, 9))
            elif fault == 2:
                r[7], r[11], r[12] = ICE, qty + 1 + int(rng.integers(0, 4)), 0
            elif fault == 3:
                r[7], r[11], r[12] = ICE, max(1, qty // 3), (1 if side == BUY else -1) * int(rng.integers(1, 4))
            else:
                r[7], r[11], r[12] = ICE, 0, 0
        rows.append(r)
    return tape(*rows, pad=2)


@pytest.mark.slow
@pytest.mark.jit
class TestEngineAgreesWithPredicate:
    """slow: jitted ContractBook submit path (compile on first call), ~30 submissions."""

    @pytest.mark.parametrize("name", sorted(REJECTABLE))
    def test_engine_rejects_every_rejectable_row(self, default_book, name):
        acc, _, _ = _replay(default_book, tape(REJECTABLE[name]))
        assert acc == [False], name

    @pytest.mark.parametrize("name", sorted(VALID_EDGE))
    def test_engine_accepts_every_boundary_valid_row(self, default_book, name):
        """Soundness direction: the predicate never drops a row the engine accepts."""
        acc, _, _ = _replay(default_book, tape(VALID_EDGE[name]))
        assert acc == [True], name


@pytest.mark.slow
@pytest.mark.jit
class TestBoundsSoundOnEngine:
    """slow: seeded random tapes with rejectable rows replayed through the jitted submit path."""

    @pytest.mark.parametrize("seed", range(40))
    def test_random_mixed_tape_stays_within_bounds(self, default_book, seed):
        t = _random_tape(7_000 + seed)
        acc, realised, st = _replay(default_book, t)
        predicted = [not synth._row_is_rejectable(r) for r in t[t[:, 4] >= 1]]
        assert acc == predicted, "predicate and engine disagree on acceptance"
        assert _breaches(t, realised, st) == [], seed

    def test_rejected_wrong_sign_ice_against_resting_book_trades_nothing(self, default_book):
        """The excluded row really is inert: bounds and engine both say zero trades."""
        t = tape(row(1, SELL, MID, 4), row(2, SELL, MID + 1, 4),
                 row(3, BUY, MID + 5, 20, ICE, 2, 4))  # BUY ICE with PPD > 0
        acc, realised, st = _replay(default_book, t)
        assert acc == [True, True, False]
        assert realised["trades"] == 0 == bounds(t)["trades"]
        assert _breaches(t, realised, st) == []

    def test_non_crossable_tape_untruncated_at_tightest_admissible_ceiling(self):
        """slow: own compile. With max_match_iterations = hop bound + 1 (the tightest
        ceiling the envelope rule admits), a tape with no crossable pair is not truncated."""
        t = tape(row(1, SELL, MID + 3, 6, ICE, 2, 0), row(2, BUY, MID, 5, ICE, 5, -1),
                 row(3, SELL, MID + 4, 2), row(4, BUY, MID - 2, 1))
        b = bounds(t)
        assert b["trades"] == 0 and b["hops"] == 3
        acc, realised, st = _replay(_book(OMIEMarketConfig(max_match_iterations=b["hops"] + 1)), t)
        assert all(acc) and realised["trades"] == 0
        assert not bool(st.truncated)


# ===========================================================================
# (c) SHRUNK_FIXTURE_DIR
# ===========================================================================
@pytest.fixture()
def cs():
    from notebooks.certification import cert_support
    return cert_support


class TestShrunkFixtureDir:
    def test_constant_points_at_tests_fixtures_cert(self, cs):
        assert Path(cs.SHRUNK_FIXTURE_DIR) == REPO_ROOT / "tests" / "omiecid" / "fixtures" / "cert"
        assert Path(cs.SHRUNK_FIXTURE_DIR).is_absolute()
        assert Path(cs.SHRUNK_FIXTURE_DIR).is_dir()

    def test_general_fixture_dir_not_repointed(self, cs):
        assert Path(cs.FIXTURE_DIR) == REPO_ROOT / "notebooks" / "certification" / "fixtures"
        assert Path(cs.FIXTURE_DIR) != Path(cs.SHRUNK_FIXTURE_DIR)

    def test_unset_and_empty_env_resolve_to_shrunk_dir(self, cs, monkeypatch):
        monkeypatch.delenv(cs.FIXTURE_ENV, raising=False)
        assert cs._env_dir(cs.FIXTURE_ENV, cs.SHRUNK_FIXTURE_DIR) == cs.SHRUNK_FIXTURE_DIR
        monkeypatch.setenv(cs.FIXTURE_ENV, "")
        assert cs._env_dir(cs.FIXTURE_ENV, cs.SHRUNK_FIXTURE_DIR) == cs.SHRUNK_FIXTURE_DIR

    def test_env_override_wins(self, cs, monkeypatch, tmp_path):
        monkeypatch.setenv(cs.FIXTURE_ENV, str(tmp_path))
        assert cs._env_dir(cs.FIXTURE_ENV, cs.SHRUNK_FIXTURE_DIR) == tmp_path

    def test_c15_notebook_default_matches_constant(self):
        """The C15 notebook's own SHRUNK_DIR default names the same directory."""
        nb = json.loads((REPO_ROOT / "notebooks/certification/C15_randomised_fuzz.ipynb")
                        .read_text(encoding="utf-8"))
        src = "".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
        m = re.search(r"SHRUNK_DIR\s*=.*", src)
        assert m, "C15 notebook defines no SHRUNK_DIR"
        assert 'OMIE_CERT_FIXTURE_DIR' in m.group(0)
        assert re.search(r'REPO_ROOT\s*/\s*"tests"\s*/\s*"omiecid"\s*/\s*"fixtures"\s*/\s*"cert"',
                         m.group(0))

    def test_gate_run_does_not_write_into_the_shrunk_dir(self, cs, monkeypatch, tmp_path):
        """execute_notebook without fixture_dir points the kernel at a fresh temp dir."""
        seen = {}

        class FakeClient:
            def __init__(self, nb, **kw):
                seen["timeout"] = kw.get("timeout")

            def execute(self, env=None):
                seen["fixture_env"] = env[cs.FIXTURE_ENV]

        monkeypatch.setattr(cs.nbclient, "NotebookClient", FakeClient)
        nbp = _tiny_notebook(tmp_path)
        cs.execute_notebook(nbp, results_dir=tmp_path / "r")
        fdir = Path(seen["fixture_env"]).resolve()
        assert fdir != Path(cs.SHRUNK_FIXTURE_DIR).resolve()
        assert Path(cs.SHRUNK_FIXTURE_DIR).resolve() not in fdir.parents


def _tiny_notebook(tmp_path, name="C99_probe.ipynb"):
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("x = 1"))
    p = tmp_path / name
    nbformat.write(nb, p)
    return p


# ===========================================================================
# (d) GATE_CELL_TIMEOUT_S
# ===========================================================================
class TestGateTimeout:
    def test_map_shape(self, cs):
        m = cs.GATE_CELL_TIMEOUT_S
        assert all(isinstance(k, str) and re.fullmatch(r"C\d{2}", k) for k in m)
        assert all(isinstance(v, int) and not isinstance(v, bool) for v in m.values())

    def test_c16_at_least_twice_the_measured_runtime(self, cs):
        assert cs.GATE_CELL_TIMEOUT_S["C16"] >= 2 * 6106

    def test_c16_bounded_so_a_runaway_still_dies(self, cs):
        """Rejected alternative R2 (raise everything): the C16 entry stays finite and sane."""
        assert cs.GATE_CELL_TIMEOUT_S["C16"] <= 24 * 3600

    def test_no_scenario_below_the_default(self, cs):
        assert all(v >= 1800 for v in cs.GATE_CELL_TIMEOUT_S.values())

    def test_only_c16_overridden(self, cs):
        """R2: no blanket raise; the other scenarios keep the 1,800 s default."""
        assert set(cs.GATE_CELL_TIMEOUT_S) == {"C16"}

    def test_every_key_names_an_existing_notebook(self, cs):
        names = {p.name[:3] for p in (REPO_ROOT / "notebooks/certification").glob("C??_*.ipynb")}
        assert set(cs.GATE_CELL_TIMEOUT_S) <= names

    def test_execute_notebook_default_unchanged(self, cs):
        assert inspect.signature(cs.execute_notebook).parameters["timeout"].default == 1800

    @pytest.mark.parametrize("timeout", [1800, 12600, 77])
    def test_execute_notebook_hands_timeout_to_nbclient(self, cs, monkeypatch, tmp_path, timeout):
        seen = {}

        class FakeClient:
            def __init__(self, nb, **kw):
                seen["timeout"] = kw.get("timeout")

            def execute(self, env=None):
                return None

        monkeypatch.setattr(cs.nbclient, "NotebookClient", FakeClient)
        cs.execute_notebook(_tiny_notebook(tmp_path), timeout=timeout, results_dir=tmp_path / "r",
                            fixture_dir=tmp_path / "f")
        assert seen["timeout"] == timeout

    @pytest.mark.parametrize(
        "nb_name", sorted(p.name for p in (REPO_ROOT / "notebooks/certification").glob("C??_*.ipynb")))
    def test_gate_passes_per_scenario_timeout_for_every_real_notebook(self, cs, monkeypatch, nb_name):
        """Call the gate test body directly with a recording executor (no kernel starts)."""
        from tests.omiecid.certification import test_certified_notebooks as gate
        seen = {}

        class G:
            passed, executed, results, results_errors, error = True, True, {}, [], None

        def fake_execute(path, **kw):
            seen["timeout"] = kw.get("timeout")
            return G()

        monkeypatch.setattr(cs, "execute_notebook", fake_execute)
        gate.test_certified_notebook_still_passes(
            REPO_ROOT / "notebooks/certification" / nb_name, None)
        assert seen["timeout"] == cs.GATE_CELL_TIMEOUT_S.get(nb_name[:3], 1800)


# ===========================================================================
# (c) the relocated C15 fixtures
# ===========================================================================
# Git blob ids of the three files as recorded on ``main`` at
# ``notebooks/certification/fixtures/`` before the move (``git ls-tree main``).
C15_BLOBS = {
    "C15_shrunk_seed100000244.npz": "6e9ed3052afe241b08194dfffb470f40e5259770",
    "C15_shrunk_seed100000246.npz": "7b6e92b6226abc1b8b997d671189343f80def804",
    "C15_shrunk_seed100000249.npz": "d800a97cf829d3ea6864163b7bbb50ed59b1b73e",
}
NEW_DIR = REPO_ROOT / "tests" / "omiecid" / "fixtures" / "cert"
OLD_DIR = REPO_ROOT / "notebooks" / "certification" / "fixtures"


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class TestRelocatedC15Fixtures:
    @pytest.mark.parametrize("name", sorted(C15_BLOBS))
    def test_byte_identical_to_recorded_blob(self, name):
        assert _git_blob_id((NEW_DIR / name).read_bytes()) == C15_BLOBS[name]

    @pytest.mark.parametrize("name", sorted(C15_BLOBS))
    def test_not_left_at_old_location(self, name):
        assert not (OLD_DIR / name).exists()

    def test_no_shrunk_fixture_anywhere_under_notebooks(self):
        assert sorted(p.name for p in (REPO_ROOT / "notebooks").rglob("*_shrunk_*.npz")) == []

    def test_staying_fixtures_stay(self):
        """Only the shrunk files move; the unshrunk C15 draw and C14/C16 fixtures remain."""
        for n in ("C15_seed100000244.npz", "C14_seed300000.npz", "C16_seed112.npz"):
            assert (OLD_DIR / n).is_file(), n

    @pytest.mark.parametrize("name", sorted(C15_BLOBS))
    def test_loads_without_pickle_and_has_a_tape(self, name):
        with np.load(NEW_DIR / name, allow_pickle=False) as d:
            t = d["tape"]
            params = json.loads(str(d["params_json"]))
        assert t.dtype == np.int32 and t.ndim == 2 and t.shape[1] == 13
        assert (t[:, 4] >= 1).sum() >= 1
        assert params.get("shrunk") is True
        assert name.endswith(f"seed{int(name.split('seed')[1][:9])}.npz")

    @pytest.mark.slow
    @pytest.mark.jit
    @pytest.mark.parametrize("name", sorted(C15_BLOBS))
    def test_replay_stays_within_bounds(self, default_book, name):
        """slow: jitted ContractBook submit path. Bound axes only (outcomes are certification)."""
        with np.load(NEW_DIR / name, allow_pickle=False) as d:
            t = np.asarray(d["tape"])
        acc, realised, st = _replay(default_book, t)
        predicted = [not synth._row_is_rejectable(r) for r in t[t[:, 4] >= 1]]
        assert acc == predicted
        b = bounds(t)
        assert all(realised[a] <= b[a] for a in realised), (realised, b)
        assert not bool(st.book_full)
