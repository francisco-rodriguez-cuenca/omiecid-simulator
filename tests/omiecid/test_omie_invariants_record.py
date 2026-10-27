"""OMIE-CERT-INVARIANTS-1 — locked tests for the record builder and RB.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-CERT-INVARIANTS-1", binding
decisions V2–V5 (entry points, record types, I2 buckets, fills) and V14 (RB).

Three groups:

* **Builder behaviour** — what ``build_record`` / ``record_from_draw`` /
  ``compare_records`` put in a ``RunRecord``, on small real engine runs. Where the
  engine's value differs from anything derivable from the tape (an iceberg's visible
  slice vs its declared total; a maker's price vs the aggressor's limit), the test
  pins the **engine** value (plan rule 2).
* **Pure readers on hand-built engine arrays** — ``BookSnapshot.resting``,
  ``best_bid``, ``best_ask``, ``trade_rows_used``.
* **RB broken fixtures** — one per V14 record field: a real record, one
  ``dataclasses.replace`` mutation, and ``assert_check_fails(check_record(...),
  "RB", field=<field>)``. The guard (V14) reads this file together with
  ``test_omie_invariants_checks.py``; keep those calls plain statements of the
  test body.

Submission indices are 0-based positions among real rows. CPU only.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_tape_synth import (
    AXES,
    CONSTRUCTIONS,
    TapeBuilder,
    tape_capacity_bounds,
)
from omiecid_simulator.omiecid.contract import ContractStatus
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
from omiecid_simulator.omiecid.settlement import compute_agent_pnl

try:  # the module under test does not exist before Step 4; keep collection green
    from tests.omiecid.reference.omie_invariants import (
        AcceptedOrder,
        BookSnapshot,
        Cancellation,
        EngineReport,
        Fill,
        Rejection,
        RecordBuildError,
        RunRecord,
        assert_check_fails,
        build_record,
        check_record,
        compare_records,
        record_from_draw,
    )

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only before implementation
    _IMPORT_ERROR = exc


def _need() -> None:
    if _IMPORT_ERROR is not None:
        pytest.fail(
            "tests/omiecid/reference/omie_invariants.py is not importable: "
            f"{_IMPORT_ERROR!r}"
        )


@pytest.fixture(autouse=True)
def _require_reference_module():
    _need()


REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "tests" / "omiecid" / "reference" / "omie_invariants.py"

BUY, SELL = int(Side.BUY), int(Side.SELL)
NON, IOC, FOK, ICE = (int(ExecRestriction.NON), int(ExecRestriction.IOC),
                      int(ExecRestriction.FOK), int(ExecRestriction.ICE))
P40 = 1_003_900
P41 = 1_004_000
P50 = 1_004_900
P51 = 1_005_000
P52 = 1_005_100


def _row(side, price_int, qty_int, agent_id, **kw):
    return dict(side=side, price_int=price_int, qty_int=qty_int, agent_id=agent_id, **kw)


def _tape(*rows):
    b = TapeBuilder()
    for r in rows:
        b.add(**r)
    return b.build()


def _swap(seq, i, new):
    out = list(seq)
    out[i] = new
    return tuple(out)


def _with_key(mapping, key, value):
    d = dict(mapping)
    d[key] = value
    return d


def _without_key(mapping, key):
    d = dict(mapping)
    d.pop(key, None)
    return d


def _resting_oids(snapshot, side):
    return sorted(int(r[2]) for r in snapshot.resting(side))


def _visible(snapshot, oid, side):
    return sum(int(r[1]) for r in snapshot.resting(side) if int(r[2]) == oid)


def _lower_visible(snapshot, oid, side, by=1):
    """A copy of ``snapshot`` with ``oid``'s first resting row on ``side`` lowered by
    ``by`` (the entry is kept: quantity stays > 0)."""
    name = "bids" if side == BUY else "asks"
    arr = np.array(getattr(snapshot, name), copy=True)
    idx = np.where((arr[:, 2] == oid) & (arr[:, 1] > 0))[0]
    assert idx.size, f"oid {oid} not resting on side {side}"
    assert arr[idx[0], 1] > by
    arr[idx[0], 1] -= by
    return replace(snapshot, **{name: arr})


# BASIC (identical to the checks file): submission 5 (oid6) sweeps rows 0..4.
BASIC_ROWS = (
    _row(SELL, P50, 10, 1),                                     # s0 oid1
    _row(SELL, P50, 10, 2),                                     # s1 oid2
    _row(SELL, P51, 10, 2),                                     # s2 oid3
    _row(SELL, P52, 50, 4, exec_restriction=ICE, peak_int=10),  # s3 oid4
    _row(BUY, P41, 0, 9),                                       # s4 oid5 rejected (qty 0)
    _row(BUY, P52, 45, 3),                                      # s5 oid6 aggressor
    _row(BUY, P40, 20, 5),                                      # s6 oid7 rests
)
BASIC_TAPE = _tape(*BASIC_ROWS)
BASIC_AGG_SUB = 5
BASIC_REJ_SUB = 4


@pytest.fixture(scope="module")
def basic():
    _need()
    return build_record(BASIC_TAPE)


@pytest.fixture(scope="module")
def sell_aggressor():
    """Resting buy (oid1, agent 1) hit by a sell (oid2, agent 2): Q > 0 row."""
    _need()
    return build_record(_tape(_row(BUY, P50, 10, 1), _row(SELL, P50 - 100, 10, 2)))


@pytest.fixture(scope="module")
def ioc_fok():
    """s0 sell 10 (oid1); s1 IOC buy 20 (oid2) fills 10; s2 sell 10 (oid3);
    s3 FOK buy 20 (oid4) killed; s4 sell 10 @P51 (oid5)."""
    _need()
    return build_record(_tape(
        _row(SELL, P50, 10, 1),
        _row(BUY, P50, 20, 2, exec_restriction=IOC),
        _row(SELL, P50, 10, 1),
        _row(BUY, P50, 20, 2, exec_restriction=FOK),
        _row(SELL, P51, 10, 6),
    ))


@pytest.fixture(scope="module")
def cancelled():
    """oid1 cancelled before oid3's row; oid3 then fills against oid2."""
    _need()
    tape = _tape(_row(SELL, P50, 10, 1), _row(SELL, P50, 10, 2), _row(BUY, P51, 10, 3))
    return build_record(tape, cancels_before={3: (1,)})


@pytest.fixture(scope="module")
def expired():
    """Unfilled iceberg (oid1, 50, peak 10), a bid (oid2, 20), an ask (oid3, 10);
    no cross; GCT expiry applied."""
    _need()
    tape = _tape(
        _row(SELL, P51, 50, 1, exec_restriction=ICE, peak_int=10),
        _row(BUY, P50, 20, 2),
        _row(SELL, P52, 10, 3),
    )
    return build_record(tape, expire_at_gct=True)


# ---------------------------------------------------------------------------
# V1 — import allow-list of the reference module
# ---------------------------------------------------------------------------
_ALLOWED_PREFIXES = ("numpy", "jax", "omiecid_simulator.omiecid")
_ALLOWED_EXACT = {
    "omiecid_simulator.jaxob.jaxob_constants": None,
    "omiecid_simulator.jaxomiecid.omie_data_loader": {"_row_to_order_traced"},
    "omiecid_simulator.jaxomiecid.omie_tape_synth": {
        "CapacityEnvelope", "DEFAULT_ENVELOPE", "tape_capacity_bounds", "Draw"},
}


def _module_allowed(mod: str) -> bool:
    top = mod.split(".")[0]
    if top in sys.stdlib_module_names or mod == "__future__":
        return True
    if any(mod == p or mod.startswith(p + ".") for p in _ALLOWED_PREFIXES):
        return True
    return mod in _ALLOWED_EXACT


class TestImportAllowList:
    def test_reference_module_imports_only_the_allow_list(self):
        assert REFERENCE.is_file(), f"{REFERENCE} missing"
        tree = ast.parse(REFERENCE.read_text())
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _ALLOWED_EXACT.get(alias.name) is not None:
                        # a name-restricted module may only be imported by name
                        bad.append(f"import {alias.name} (whole module)")
                    elif not _module_allowed(alias.name):
                        bad.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    bad.append(f"relative import from {'.' * node.level}{mod}")
                    continue
                if not _module_allowed(mod):
                    bad.append(f"from {mod} import ...")
                    continue
                names = _ALLOWED_EXACT.get(mod)
                if names is not None:
                    extra = {a.name for a in node.names} - names
                    if extra:
                        bad.append(f"from {mod} import {sorted(extra)}")
        assert not bad, bad


# ---------------------------------------------------------------------------
# V2 — entry points, driving, errors
# ---------------------------------------------------------------------------
class TestBuildRecordEntryPoint:
    def test_returns_a_run_record_with_one_snapshot_pair_per_real_row(self, basic):
        assert isinstance(basic, RunRecord)
        assert basic.n_submissions == 7
        assert len(basic.pre_snapshots) == 7
        assert len(basic.post_snapshots) == 7
        assert len(basic.engine_reports) == 7

    def test_pad_rows_are_skipped_wherever_they_sit(self):
        tape = _tape(_row(SELL, P50, 10, 1), _row(BUY, P50, 10, 2))
        pad = tape[-1].copy()
        assert pad[4] < 1
        with_mid_pad = np.ascontiguousarray(np.vstack([tape[:1], pad[None, :], tape[1:]]),
                                            dtype=np.int32)
        rec = build_record(with_mid_pad)
        assert rec.n_submissions == 2
        assert len(rec.fills) == 1

    def test_record_keeps_the_tape_and_the_configs(self, basic):
        assert np.array_equal(np.asarray(basic.tape), BASIC_TAPE)
        assert basic.jit is True
        assert basic.product_config is not None and basic.market_config is not None

    def test_empty_tape_builds_an_empty_record(self):
        rec = build_record(CONSTRUCTIONS["C00.empty"](0).tape)
        assert rec.n_submissions == 0
        assert tuple(rec.fills) == ()
        assert len(rec.accepted) == 0
        assert rec.final_snapshot is not None
        assert rec.final_snapshot.resting(BUY) == () or len(rec.final_snapshot.resting(BUY)) == 0

    def test_record_types_are_frozen(self, basic):
        with pytest.raises(FrozenInstanceError):
            basic.n_submissions = 0  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            basic.fills[0].qty_int = 0  # type: ignore[misc]

    def test_record_build_error_is_a_value_error(self):
        assert issubclass(RecordBuildError, ValueError)

    def test_rejects_a_tape_with_the_wrong_width(self):
        with pytest.raises(RecordBuildError):
            build_record(BASIC_TAPE[:, :12])

    def test_rejects_a_tape_with_the_wrong_dtype(self):
        with pytest.raises(RecordBuildError):
            build_record(BASIC_TAPE.astype(np.int64))

    def test_rejects_a_one_dimensional_tape(self):
        with pytest.raises(RecordBuildError):
            build_record(BASIC_TAPE[0])

    def test_rejects_mixed_contract_ids_among_real_rows(self):
        tape = _tape(_row(SELL, P50, 10, 1), _row(BUY, P50, 10, 2, contract_id=3))
        with pytest.raises(RecordBuildError):
            build_record(tape)

    def test_rejects_a_cancel_of_an_oid_never_submitted(self):
        with pytest.raises(RecordBuildError):
            build_record(BASIC_TAPE, cancels_before={6: (99,)})

    def test_rejects_a_cancel_of_a_rejected_oid(self):
        with pytest.raises(RecordBuildError):
            build_record(BASIC_TAPE, cancels_before={6: (5,)})

    @pytest.mark.parametrize("status", [ContractStatus.PENDING, ContractStatus.SUSPENDED,
                                        ContractStatus.CLOSED])
    def test_non_open_contract_does_not_raise_and_rejects_everything(self, status):
        rec = build_record(_tape(_row(SELL, P50, 10, 1), _row(BUY, P50, 10, 2)),
                           contract_status=status)
        assert len(rec.accepted) == 0
        assert len(rec.rejections) == 2
        assert {r.reason for r in rec.rejections} == {"contract_not_open"}
        assert tuple(rec.fills) == ()
        assert all(er.accepted is False for er in rec.engine_reports)

    def test_capacity_bounds_default_to_the_tape_bounds(self, basic):
        assert dict(basic.capacity_bounds) == tape_capacity_bounds(BASIC_TAPE)

    def test_capacity_bounds_are_taken_when_supplied(self):
        bounds = {"resting_per_side": 9, "trades": 9, "hops": 9, "concurrent_icebergs": 9}
        rec = build_record(_tape(_row(SELL, P50, 10, 1)), capacity_bounds=bounds)
        assert dict(rec.capacity_bounds) == bounds


class TestRecordFromDraw:
    def test_forwards_the_draws_capacity_bounds(self):
        draw = CONSTRUCTIONS["C02.crossing_pair"](0)
        rec = record_from_draw(draw)
        assert dict(rec.capacity_bounds) == dict(draw.capacity_bounds)
        assert rec.n_submissions == draw.n_real_rows()

    def test_forwards_the_draws_cancel_before_meta(self):
        draw = CONSTRUCTIONS["C04.cancellation"](0)
        rec = record_from_draw(draw)
        real_oids = [int(o) for o in draw.tape[:, 4] if o >= 1]
        before = real_oids.index(int(draw.meta["cancel_before_row_of_oid"]))
        cancel_oid = int(draw.params["cancel_oid"])
        assert tuple(tuple(e) for e in rec.cancel_events) == ((before, cancel_oid),)
        causes = [c.cause for c in rec.cancelled_or_expired.get(cancel_oid, ())]
        assert causes == ["cancellation"]

    def test_iceberg_cancel_counts_visible_plus_hidden_and_the_fok_is_killed(self):
        draw = CONSTRUCTIONS["C04.iceberg_cancel_fok"](0)
        p = draw.params
        rec = record_from_draw(draw)
        ice = rec.cancelled_or_expired[int(p["ice_oid"])]
        assert [(c.cause, c.qty_int) for c in ice] == [("cancellation", int(p["total_int"]))]
        fok = rec.cancelled_or_expired[int(p["fok_oid"])]
        assert [(c.cause, c.qty_int) for c in fok] == [("fok_kill", int(p["req_int"]))]


class TestCompareRecords:
    def test_identical_records_compare_equal(self, basic):
        assert compare_records(basic, basic) == ()

    def test_an_integer_field_difference_is_named(self, basic):
        other = replace(basic, filled=_with_key(basic.filled, 1, basic.filled[1] + 1))
        assert "filled" in compare_records(basic, other)

    def test_a_fill_difference_is_named(self, basic):
        other = replace(basic, fills=_swap(basic.fills, 0, replace(basic.fills[0], qty_int=9)))
        assert "fills" in compare_records(basic, other)

    def test_float_settlements_are_excluded(self, basic):
        other = replace(basic, engine_settlements={k: v + 1e-3 for k, v in
                                                   basic.engine_settlements.items()})
        assert compare_records(basic, other) == ()

    def test_float_euro_report_fields_are_excluded(self, basic):
        er = basic.engine_reports[BASIC_AGG_SUB]
        other = replace(basic, engine_reports=_swap(
            basic.engine_reports, BASIC_AGG_SUB,
            replace(er, settlement_value_eur=er.settlement_value_eur + 1e-3,
                    trade_price_eur=er.trade_price_eur + 1e-3)))
        assert compare_records(basic, other) == ()

    def test_returns_a_tuple_of_strings(self, basic):
        other = replace(basic, still_resting=_with_key(basic.still_resting, 7, 0))
        out = compare_records(basic, other)
        assert isinstance(out, tuple) and all(isinstance(n, str) for n in out)

    @pytest.mark.slow
    def test_eager_and_jitted_records_agree_on_integer_fields(self, basic):
        """slow: the eager build runs every submission under jax.disable_jit (~1 s)."""
        eager = build_record(BASIC_TAPE, jit=False)
        assert eager.jit is False
        assert compare_records(eager, basic) == ()


# ---------------------------------------------------------------------------
# V3 — record types: accepted, rejections, reports, participants, flags
# ---------------------------------------------------------------------------
class TestAcceptedAndRejected:
    def test_accepted_carries_the_tape_declaration(self, basic):
        a = basic.accepted[4]
        assert isinstance(a, AcceptedOrder)
        assert (a.oid, a.submission, a.side, a.price_int, a.qty_int, a.agent,
                a.exec_restriction, a.peak_int, a.ppd_int) == (4, 3, SELL, P52, 50, 4, ICE, 10, 0)
        assert a.tape_ts == int(BASIC_TAPE[3, 6])

    def test_accepted_excludes_the_rejected_order(self, basic):
        assert set(basic.accepted) == {1, 2, 3, 4, 6, 7}

    def test_validator_rejection_is_recorded_with_its_row(self, basic):
        assert len(basic.rejections) == 1
        r = basic.rejections[0]
        assert isinstance(r, Rejection)
        assert (r.submission, r.oid, r.reason) == (BASIC_REJ_SUB, 5, "validator")
        assert tuple(int(x) for x in r.row) == tuple(int(x) for x in BASIC_TAPE[4])

    def test_the_validators_own_reason_is_reported_unobserved(self, basic):
        assert "rejection_reason" in basic.unobserved

    def test_rejected_submission_has_identical_pre_and_post_books(self, basic):
        pre, post = basic.pre_snapshots[BASIC_REJ_SUB], basic.post_snapshots[BASIC_REJ_SUB]
        for name in ("bids", "asks", "trades", "iceberg_meta"):
            assert np.array_equal(getattr(pre, name), getattr(post, name)), name

    def test_engine_reports_are_python_scalars_in_submission_order(self, basic):
        for s, er in enumerate(basic.engine_reports):
            assert isinstance(er, EngineReport)
            assert er.submission == s
            assert type(er.accepted) is bool and type(er.traded) is bool
            for v in (er.trade_qty_mw, er.trade_price_eur, er.settlement_value_eur):
                assert type(v) is float
        assert basic.engine_reports[BASIC_REJ_SUB].accepted is False
        agg = basic.engine_reports[BASIC_AGG_SUB]
        assert agg.accepted is True and agg.traded is True
        assert agg.trade_qty_mw == pytest.approx(4.5, abs=1e-5)

    def test_participants_are_the_agents_of_accepted_orders(self, basic):
        assert basic.participants == frozenset({1, 2, 3, 4, 5})

    def test_flags_have_exactly_the_three_keys_and_are_clear(self, basic):
        assert dict(basic.flags) == {"book_full": False, "truncated": False,
                                     "trade_buffer_saturated": False}

    def test_observed_capacity_is_read_from_the_snapshots(self, basic):
        oc = dict(basic.observed_capacity)
        assert set(oc) <= set(AXES)
        assert {"resting_per_side", "trades", "concurrent_icebergs"} <= set(oc)
        assert oc["resting_per_side"] == 4   # 4 asks rest before submission 5
        assert oc["trades"] == 5
        assert oc["concurrent_icebergs"] == 1
        # V11: the hop axis is the analytic bound, never an observation
        assert oc.get("hops", basic.capacity_bounds["hops"]) == basic.capacity_bounds["hops"]


# ---------------------------------------------------------------------------
# V5 — fills: whole-row diff, submission tag, maker id, buyer/seller from sign
# ---------------------------------------------------------------------------
class TestFills:
    def test_fills_are_tagged_with_the_submission_that_wrote_them(self, basic):
        assert [(f.row_index, f.submission) for f in basic.fills] == [
            (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]

    def test_fill_fields_come_from_the_trade_rows(self, basic):
        trades = np.asarray(basic.post_snapshots[BASIC_AGG_SUB].trades)
        for f in basic.fills:
            row = trades[f.row_index]
            assert isinstance(f, Fill)
            assert f.price_int == int(row[0])
            assert f.qty_signed == int(row[1])
            assert f.qty_int == abs(int(row[1]))
            assert f.passive_oid == int(row[2])
            assert f.aggressor_oid == int(row[3])
            assert (f.sec, f.nsec) == (int(row[4]), int(row[5]))
            assert f.passive_agent == int(row[6])
            assert f.aggressor_agent == int(row[7])

    def test_trade_price_is_the_makers_not_the_aggressors(self, basic):
        # Tape-derived from the aggressor would be P52 for every row.
        assert [f.price_int for f in basic.fills] == [P50, P50, P51, P52, P52]
        assert [f.qty_int for f in basic.fills] == [10, 10, 10, 10, 5]
        assert [f.passive_oid for f in basic.fills] == [1, 2, 3, 4, 4]
        assert {f.aggressor_oid for f in basic.fills} == {6}

    def test_a_buying_aggressor_is_the_buyer(self, basic):
        f = basic.fills[0]
        assert f.qty_signed < 0
        assert (f.buyer_oid, f.seller_oid) == (6, 1)
        assert (f.buyer_agent, f.seller_agent) == (3, 1)

    def test_a_selling_aggressor_makes_the_passive_the_buyer(self, sell_aggressor):
        (f,) = sell_aggressor.fills
        assert f.qty_signed > 0
        assert (f.buyer_oid, f.seller_oid) == (1, 2)
        assert (f.buyer_agent, f.seller_agent) == (1, 2)
        assert f.price_int == P50

    def test_pre_snapshot_follows_the_previous_post_snapshot(self, basic):
        for i in range(1, basic.n_submissions):
            for name in ("bids", "asks", "trades"):
                assert np.array_equal(getattr(basic.pre_snapshots[i], name),
                                      getattr(basic.post_snapshots[i - 1], name)), (i, name)

    def test_snapshots_are_copies_not_views_of_one_buffer(self, basic):
        assert not np.array_equal(basic.pre_snapshots[BASIC_AGG_SUB].trades,
                                  basic.post_snapshots[BASIC_AGG_SUB].trades)

    @pytest.mark.slow
    def test_trade_buffer_saturation_is_flagged(self):
        """slow: a non-default nTrades recompiles the submit path (~15 s)."""
        tape = _tape(*[_row(SELL, P50, 10, i + 1) for i in range(5)], _row(BUY, P50, 50, 9))
        rec = build_record(tape, market_config=OMIEMarketConfig(nTrades=4))
        assert rec.flags["trade_buffer_saturated"] is True
        assert rec.final_snapshot.trade_rows_used() == 4


# ---------------------------------------------------------------------------
# V4 — the I2 buckets, read from engine state
# ---------------------------------------------------------------------------
class TestBuckets:
    def test_filled_is_summed_from_the_trade_rows(self, basic):
        assert {o: basic.filled.get(o, 0) for o in basic.accepted} == {
            1: 10, 2: 10, 3: 10, 4: 15, 6: 45, 7: 0}

    def test_iceberg_resting_is_the_visible_slice_not_the_declared_total(self, basic):
        assert basic.still_resting.get(4, 0) == 5
        assert basic.hidden_remaining.get(4, 0) == 30

    def test_still_resting_of_a_plain_order(self, basic):
        assert basic.still_resting.get(7, 0) == 20
        assert basic.still_resting.get(6, 0) == 0
        assert basic.hidden_remaining.get(7, 0) == 0

    def test_no_cancellation_and_no_eviction_on_basic(self, basic):
        assert all(len(v) == 0 for v in basic.cancelled_or_expired.values())
        assert all(v == 0 for v in basic.evicted.values())
        assert tuple(basic.cancel_events) == ()
        assert basic.expired_at_gct is False

    def test_ioc_remainder_is_accepted_minus_its_own_fills(self, ioc_fok):
        (c,) = ioc_fok.cancelled_or_expired[2]
        assert isinstance(c, Cancellation)
        assert (c.oid, c.cause, c.qty_int) == (2, "ioc_remainder", 10)
        assert ioc_fok.still_resting.get(2, 0) == 0

    def test_fok_kill_is_the_whole_accepted_quantity(self, ioc_fok):
        (c,) = ioc_fok.cancelled_or_expired[4]
        assert (c.oid, c.cause, c.qty_int) == (4, "fok_kill", 20)
        assert ioc_fok.filled.get(4, 0) == 0
        assert ioc_fok.engine_reports[3].accepted is True
        assert ioc_fok.engine_reports[3].traded is False

    def test_non_orders_get_no_cancellation_of_their_own(self, ioc_fok):
        for oid in (1, 3, 5):
            assert len(ioc_fok.cancelled_or_expired.get(oid, ())) == 0

    def test_cancellation_is_read_before_cancel_and_logged_as_an_event(self, cancelled):
        assert tuple(tuple(e) for e in cancelled.cancel_events) == ((2, 1),)
        (c,) = cancelled.cancelled_or_expired[1]
        assert (c.oid, c.cause, c.qty_int) == (1, "cancellation", 10)
        assert 1 not in _resting_oids(cancelled.pre_snapshots[2], SELL)
        assert [f.passive_oid for f in cancelled.fills] == [2]

    def test_gate_expiry_counts_visible_plus_hidden_per_order(self, expired):
        assert expired.expired_at_gct is True
        got = {o: [(c.cause, c.qty_int) for c in expired.cancelled_or_expired.get(o, ())]
               for o in (1, 2, 3)}
        assert got == {1: [("gate_expiry", 50)], 2: [("gate_expiry", 20)],
                       3: [("gate_expiry", 10)]}

    def test_final_snapshot_is_taken_after_expiry(self, expired):
        assert len(expired.final_snapshot.resting(BUY)) == 0
        assert len(expired.final_snapshot.resting(SELL)) == 0
        assert int(expired.final_snapshot.n_active) == 0
        assert {o: expired.still_resting.get(o, 0) for o in (1, 2, 3)} == {1: 0, 2: 0, 3: 0}
        assert expired.hidden_remaining.get(1, 0) == 0
        # the last post-snapshot is the book *before* expiry
        assert len(expired.post_snapshots[-1].resting(SELL)) == 2


# ---------------------------------------------------------------------------
# Pure readers on hand-built engine arrays
# ---------------------------------------------------------------------------
def _side_array(rows, depth=6):
    a = -np.ones((depth, 6), dtype=np.int32)
    for i, r in enumerate(rows):
        a[i] = r
    return a


def _snapshot(bids, asks, trades=None):
    if trades is None:
        trades = -np.ones((4, 8), dtype=np.int32)
    return BookSnapshot(
        bids=bids, asks=asks, trades=trades,
        iceberg_meta=np.zeros((0, 8), dtype=np.int32),
        n_active=0, ts_counter=0, book_full=False, truncated=False,
    )


class TestPureReaders:
    def test_resting_keeps_rows_with_positive_oid_and_quantity(self):
        bids = _side_array([
            [P50, 10, 11, 1, 3, 0],
            [P51, 0, 12, 1, 4, 0],     # zero quantity: not resting
            [P52, 10, -1, 1, 5, 0],    # no order id: not resting
            [P40, 7, 13, 2, 6, 9],
        ])
        snap = _snapshot(bids, _side_array([]))
        assert sorted(snap.resting(BUY)) == sorted([(P50, 10, 11, 1, 3, 0),
                                                    (P40, 7, 13, 2, 6, 9)])
        assert len(snap.resting(SELL)) == 0

    def test_best_bid_and_best_ask_come_from_the_resting_rows(self):
        bids = _side_array([[P40, 5, 1, 1, 1, 0], [P50, 5, 2, 1, 2, 0], [P52, 0, 3, 1, 3, 0]])
        asks = _side_array([[P52, 5, 4, 1, 4, 0], [P51, 5, 5, 1, 5, 0], [P50 - 1, 5, -1, 1, 6, 0]])
        snap = _snapshot(bids, asks)
        assert snap.best_bid() == P50
        assert snap.best_ask() == P51

    def test_best_prices_are_none_on_an_empty_side(self):
        snap = _snapshot(_side_array([]), _side_array([]))
        assert snap.best_bid() is None
        assert snap.best_ask() is None

    def test_trade_rows_used_tests_column_four(self):
        trades = -np.ones((4, 8), dtype=np.int32)
        trades[0] = [P50, -10, 1, 2, 3, 0, 1, 2]
        trades[1, 0] = P50            # price set but SEC == -1: unused
        trades[2] = [-1, 5, -1, -1, 7, 0, -1, -1]  # SEC set: used
        snap = _snapshot(_side_array([]), _side_array([]), trades)
        assert snap.trade_rows_used() == 2


# ---------------------------------------------------------------------------
# RB — record consistency, one broken fixture per V14 field
# ---------------------------------------------------------------------------
class TestRecordCheckHealthy:
    @pytest.mark.parametrize("name", ["basic", "ioc_fok", "cancelled", "expired",
                                      "sell_aggressor"])
    def test_real_records_pass_rb(self, request, name):
        res = check_record(request.getfixturevalue(name))
        assert res.status == "pass", res.violations


class TestRecordCheckBroken:
    def test_rb_fails_when_a_pre_snapshot_is_missing(self, basic):
        broken = replace(basic, pre_snapshots=tuple(basic.pre_snapshots)[:-1])
        res = check_record(broken)
        assert_check_fails(res, "RB", field="pre_snapshots")

    def test_rb_fails_when_the_last_post_snapshot_is_missing(self, basic):
        broken = replace(basic, post_snapshots=tuple(basic.post_snapshots)[:-1])
        res = check_record(broken)
        assert_check_fails(res, "RB", field="post_snapshots")

    def test_rb_fails_on_a_fill_tagged_with_an_out_of_range_submission(self, basic):
        bad = replace(basic.fills[0], submission=basic.n_submissions)
        broken = replace(basic, fills=_swap(basic.fills, 0, bad))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="fills")

    def test_rb_fails_on_fills_out_of_row_order(self, basic):
        f = tuple(basic.fills)
        broken = replace(basic, fills=(f[1], f[0]) + f[2:])
        res = check_record(broken)
        assert_check_fails(res, "RB", field="fills")

    def test_rb_fails_on_a_fill_outside_its_submissions_diff(self, basic):
        # Submission 0 wrote no trade row.
        bad = replace(basic.fills[0], submission=0)
        broken = replace(basic, fills=_swap(basic.fills, 0, bad))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="fills")

    def test_rb_fails_on_a_fill_whose_quantity_disagrees_with_its_row(self, basic):
        bad = replace(basic.fills[2], qty_int=9, qty_signed=-9)
        broken = replace(basic, fills=_swap(basic.fills, 2, bad))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="fills")

    def test_rb_fails_when_filled_disagrees_with_the_fills(self, basic):
        broken = replace(basic, filled=_with_key(basic.filled, 1, basic.filled[1] + 1))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="filled")

    def test_rb_fails_when_still_resting_disagrees_with_the_final_book(self, basic):
        broken = replace(basic, still_resting=_with_key(basic.still_resting, 7, 19))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="still_resting")

    def test_rb_fails_when_hidden_remaining_disagrees_with_the_side_channel(self, basic):
        broken = replace(basic, hidden_remaining=_with_key(basic.hidden_remaining, 4, 29))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="hidden_remaining")

    def test_rb_fails_when_a_rejection_is_also_accepted(self, basic):
        r = basic.rejections[0]
        broken = replace(basic, rejections=_swap(basic.rejections, 0, replace(r, oid=1)))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="rejections")

    def test_rb_fails_when_a_rejected_submission_changed_the_book(self, basic):
        changed = basic.post_snapshots[BASIC_AGG_SUB]
        broken = replace(basic, post_snapshots=_swap(basic.post_snapshots, BASIC_REJ_SUB, changed))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="rejections")

    def test_rb_fails_when_a_rejection_is_reported_accepted(self, basic):
        er = basic.engine_reports[BASIC_REJ_SUB]
        broken = replace(basic, engine_reports=_swap(
            basic.engine_reports, BASIC_REJ_SUB, replace(er, accepted=True)))
        res = check_record(broken)
        assert_check_fails(res, "RB")
        assert {v.field for v in res.violations} & {"rejections", "engine_reports"}

    def test_rb_fails_when_an_engine_report_is_missing(self, basic):
        broken = replace(basic, engine_reports=tuple(basic.engine_reports)[:-1])
        res = check_record(broken)
        assert_check_fails(res, "RB", field="engine_reports")

    def test_rb_fails_when_an_accepted_order_is_reported_rejected(self, basic):
        er = basic.engine_reports[0]
        broken = replace(basic, engine_reports=_swap(
            basic.engine_reports, 0, replace(er, accepted=False)))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="engine_reports")

    def test_rb_fails_on_an_eviction_without_the_level_wipe_signature(self, basic):
        # oid 7 is still resting in the final book: not an eviction.
        broken = replace(basic, evicted=_with_key(basic.evicted, 7, 20))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="evicted")

    def test_rb_fails_on_an_eviction_of_an_order_that_filled(self, basic):
        # oid 1 disappeared by a fill of submission 5, not by eviction.
        broken = replace(basic, evicted=_with_key(basic.evicted, 1, 10))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="evicted")

    def test_rb_fails_when_observed_capacity_disagrees_with_the_snapshots(self, basic):
        oc = dict(basic.observed_capacity)
        oc["resting_per_side"] += 1
        broken = replace(basic, observed_capacity=oc)
        res = check_record(broken)
        assert_check_fails(res, "RB", field="observed_capacity")

    def test_rb_fails_when_observed_trades_disagree(self, basic):
        oc = dict(basic.observed_capacity)
        oc["trades"] -= 1
        broken = replace(basic, observed_capacity=oc)
        res = check_record(broken)
        assert_check_fails(res, "RB", field="observed_capacity")

    def test_rb_fails_when_a_trade_row_agent_has_no_settlement(self, basic):
        broken = replace(basic, engine_settlements=_without_key(basic.engine_settlements, 3))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="engine_settlements")

    # --- V14 as amended at Step 5 (v5.187): cancelled_or_expired, flags,
    # accepted, participants ---
    def test_rb_fails_when_a_cancellation_disagrees_with_the_pre_cancel_book(self, cancelled):
        # The cancel of oid1 happens before row s = 2; post_snapshots[1] is the
        # state immediately before it. Lower oid1's visible qty there (entry kept):
        # the recorded cancellation of 10 no longer matches the engine's 9.
        (s, oid), = tuple(tuple(e) for e in cancelled.cancel_events)
        snap = _lower_visible(cancelled.post_snapshots[s - 1], oid, SELL)
        assert _visible(snap, oid, SELL) == 9
        broken = replace(cancelled, post_snapshots=_swap(cancelled.post_snapshots, s - 1, snap))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="cancelled_or_expired")

    def test_rb_fails_when_a_gate_expiry_disagrees_with_the_pre_expiry_book(self, expired):
        # post_snapshots[-1] is the state immediately before expire_at_gct.
        last = len(expired.post_snapshots) - 1
        snap = _lower_visible(expired.post_snapshots[last], 2, BUY)
        assert _visible(snap, 2, BUY) == 19
        broken = replace(expired, post_snapshots=_swap(expired.post_snapshots, last, snap))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="cancelled_or_expired")

    def test_rb_fails_when_an_ioc_remainder_is_dropped(self, ioc_fok):
        assert [c.cause for c in ioc_fok.cancelled_or_expired[2]] == ["ioc_remainder"]
        broken = replace(ioc_fok, cancelled_or_expired=_with_key(
            ioc_fok.cancelled_or_expired, 2, ()))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="cancelled_or_expired")

    def test_rb_fails_when_book_full_is_flagged_on_clean_snapshots(self, basic):
        broken = replace(basic, flags=_with_key(basic.flags, "book_full", True))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="flags")

    def test_rb_fails_when_a_truncated_snapshot_is_not_flagged(self, basic):
        assert not any(basic.flags.values())
        snap = replace(basic.post_snapshots[BASIC_AGG_SUB], truncated=True)
        broken = replace(basic, post_snapshots=_swap(basic.post_snapshots, BASIC_AGG_SUB, snap))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="flags")

    def test_rb_fails_when_an_accepted_quantity_differs_from_the_tape(self, basic):
        a = basic.accepted[7]
        broken = replace(basic, accepted=_with_key(basic.accepted, 7,
                                                   replace(a, qty_int=a.qty_int + 1)))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="accepted")

    def test_rb_fails_when_a_participant_is_missing(self, basic):
        assert 5 in basic.participants
        broken = replace(basic, participants=frozenset(basic.participants - {5}))
        res = check_record(broken)
        assert_check_fails(res, "RB", field="participants")


# ---------------------------------------------------------------------------
# I6(a) input — engine_settlements is compute_agent_pnl on the (M, 6) flattening
# ---------------------------------------------------------------------------
class TestEngineSettlements:
    def test_settlements_cover_every_trade_row_agent(self, basic):
        agents = {f.buyer_agent for f in basic.fills} | {f.seller_agent for f in basic.fills}
        assert agents == {1, 2, 3, 4}
        assert agents <= set(basic.engine_settlements)

    def test_settlements_are_compute_agent_pnl_on_the_flattened_fills(self, basic):
        t = np.array([[f.sec, f.price_int, f.qty_int, f.buyer_agent, f.seller_agent, 0]
                      for f in basic.fills], dtype=np.int32)
        assert t.shape == (5, 6)
        for a in (1, 2, 3, 4):
            want = float(compute_agent_pnl(jnp.asarray(t), a))
            assert basic.engine_settlements[a] == pytest.approx(want, rel=1e-6, abs=1e-6)

    def test_settlement_values_are_python_floats(self, basic):
        assert all(type(v) is float for v in basic.engine_settlements.values())
