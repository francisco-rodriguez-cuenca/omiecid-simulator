"""OMIE-CERT-INVARIANTS-1 — locked clean-run, API-surface and tolerance-pin tests.

Story card: ``docs/omiecid/agile-plan.md`` §"Story OMIE-CERT-INVARIANTS-1", binding
decisions V6 (check API), V9 (``RULE3_HOP_PRICE_SETTER``), V10 (the I6 tolerance),
V11 (I8), V13 (unobserved vocabulary) and V15 (clean runs).

* V15(a): the full battery passes on both ``_default_fixed_tape_set`` tapes (AC 5).
* V15(b): every ``CONSTRUCTIONS`` key x seeds 0-7, contract OPEN, gives no
  ``"fail"`` or ``"invalid"`` from RB and I1-I8.
* V15(c): the derived tolerance
  $\\tau_s = k\\,u\\,(c_s W_s + O\\,Q_s + 5\\lvert D_s\\rvert)(1 + (c_s + 8)u)$
  (V10 as amended at Step 5, v5.187; $c_s$ = the number of distinct engine
  timestamps ``(sec, nsec)`` among the submission's fill rows, the settlement
  chunks) is pinned **behaviourally**: a reported settlement just inside
  $\\tau_s$ passes I6, just outside fails. The pin goes through
  ``check_i6_settlement`` itself (rel. margin 1e-6 of $\\tau$), on three real
  records: $m = c = 1$; $m = 3$, $c = 1$; and $m = 3$, $c = 2$.
* V15(d): one passing real-engine instance per M check.

The ``assert_check_fails`` meta-tests live here (not in the guarded files) because
they wrap the helper in ``pytest.raises``. CPU only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from omiecid_simulator.jaxomiecid.omie_tape_synth import CONSTRUCTIONS, CapacityEnvelope, TapeBuilder
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.omie_session import _default_fixed_tape_set
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side

try:  # the module under test does not exist before Step 4; keep collection green
    from tests.omiecid.reference.omie_invariants import (
        ALL_CHECKS,
        CHECKS,
        RULE3_HOP_PRICE_SETTER,
        STATEMENTS,
        BatteryReport,
        CheckResult,
        Violation,
        assert_check_fails,
        build_record,
        check_i1_volume_balance,
        check_i2_quantity_conservation,
        check_i3_uncrossed_on_return,
        check_i4_priority,
        check_i5_price_setter,
        check_i6_settlement,
        check_i7_no_manufactured_liquidity,
        check_i8_capacity_headroom,
        check_i9_non_vacuity,
        check_m1_tie_permutation,
        check_m2_monotonicity,
        check_m3_ioc_non_equivalence,
        check_m4_fok_dichotomy,
        check_m5_split_aggressor,
        check_m6_ownership_blindness,
        check_record,
        record_from_draw,
        run_battery,
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

BUY, SELL = int(Side.BUY), int(Side.SELL)
IOC, FOK, ICE = int(ExecRestriction.IOC), int(ExecRestriction.FOK), int(ExecRestriction.ICE)
P40 = 1_003_900
P41 = 1_004_000
P50 = 1_004_900
P51 = 1_005_000
P52 = 1_005_100

EXPECTED_ALL_CHECKS = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9",
                       "M1", "M2", "M3", "M4", "M5", "M6")
BATTERY_ORDER = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")
STATUSES = {"pass", "fail", "invalid", "not_applicable"}
UNOBSERVED_VOCAB = {"rejection_reason", "queue_timestamp_correctness",
                    "independent_valuation", "matching_iterations"}


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


BASIC_ROWS = (
    _row(SELL, P50, 10, 1),
    _row(SELL, P50, 10, 2),
    _row(SELL, P51, 10, 2),
    _row(SELL, P52, 50, 4, exec_restriction=ICE, peak_int=10),
    _row(BUY, P41, 0, 9),
    _row(BUY, P52, 45, 3),
    _row(BUY, P40, 20, 5),
)


@pytest.fixture(scope="module")
def basic():
    _need()
    return build_record(_tape(*BASIC_ROWS))


@pytest.fixture(scope="module")
def nocross():
    _need()
    return build_record(_tape(_row(SELL, P51, 10, 1), _row(BUY, P50, 10, 2)))


@pytest.fixture(scope="module")
def garbage(basic):
    """A record whose only fill names orders, agents and nothing that exists."""
    f = basic.fills[0]
    bad = replace(f, passive_oid=999, aggressor_oid=998, buyer_oid=998, seller_oid=999,
                  buyer_agent=99, seller_agent=98, passive_agent=98, aggressor_agent=99)
    return replace(basic, fills=(bad,))


# ---------------------------------------------------------------------------
# V15(a) — the default tape set runs clean (AC 5)
# ---------------------------------------------------------------------------
class TestDefaultTapeSet:
    @pytest.mark.parametrize("index", [0, 1])
    def test_battery_passes_on_each_default_tape(self, index):
        tapes = _default_fixed_tape_set(OMIEProductConfig())
        assert len(tapes) == 2
        rec = build_record(tapes[index])
        report = run_battery(rec, expects_trade=True)
        assert isinstance(report, BatteryReport)
        got = {r.check_id: r.status for r in report.results}
        assert got == {cid: "pass" for cid in BATTERY_ORDER}, [
            (r.check_id, r.status, r.violations) for r in report.results if r.status != "pass"]
        assert report.passed is True
        assert report.valid is True


# ---------------------------------------------------------------------------
# V15(b) — every registry construction x seeds 0-7 (no fail / invalid)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(CONSTRUCTIONS))
def test_registry_construction_runs_clean_for_seeds_0_to_7(key):
    bad = []
    for seed in range(8):
        rec = record_from_draw(CONSTRUCTIONS[key](seed))
        for cid in ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"):
            res = CHECKS[cid](rec)
            if res.status in ("fail", "invalid"):
                bad.append((seed, cid, res.status, res.violations[:3]))
    assert not bad, bad


# ---------------------------------------------------------------------------
# V15(c) — the derived I6 tolerance, pinned behaviourally
# ---------------------------------------------------------------------------
U = 2.0 ** -24


def _tau_and_value(rows, pc=OMIEProductConfig(), chunks=None):
    """(tau_s, V_s) for fill rows [(price_int, qty_int), ...] of one submission.

    ``chunks`` is c_s, the number of distinct (sec, nsec) among the rows; when
    None it is taken as ``len(rows)`` (one chunk per row, the loosest case)."""
    k = float(pc.qty_tick) * float(pc.price_tick) * float(pc.delivery_duration_hours)
    O = round(-pc.price_min / pc.price_tick)
    W = sum(p * q for p, q in rows)
    Q = sum(q for _, q in rows)
    D = W - O * Q
    c = len(rows) if chunks is None else int(chunks)
    return k * U * (c * W + O * Q + 5 * abs(D)) * (1 + (c + 8) * U), k * D


PIN1_ROWS = [(1_004_598, 20)]
PIN3_ROWS = [(1_004_598, 10), (1_004_650, 7), (1_004_701, 13)]
PIN2C_ROWS = [(P50, 10), (P50, 20), (P50, 10)]


@pytest.fixture(scope="module")
def pin1():
    _need()
    return build_record(_tape(_row(SELL, 1_004_598, 20, 1), _row(BUY, 1_004_598, 20, 2)))


@pytest.fixture(scope="module")
def pin3():
    _need()
    return build_record(_tape(
        _row(SELL, 1_004_598, 10, 1), _row(SELL, 1_004_650, 7, 2),
        _row(SELL, 1_004_701, 13, 3), _row(BUY, 1_004_800, 30, 4)))


@pytest.fixture(scope="module")
def pin2c():
    """Two settlement chunks (V15(c), v5.187): ice 10 and NON 20 at one engine
    timestamp, then ice 10 at a later one (probe 2026-10-02: (3,0), (3,0), (5,0))."""
    _need()
    return build_record(_tape(
        _row(SELL, P50, 50, 1, exec_restriction=ICE, peak_int=10),
        _row(SELL, P50, 20, 2),
        _row(BUY, P50, 40, 3)))


# which -> (fill rows, trading submission, c_s)
PINS = {"pin1": (PIN1_ROWS, 1, 1), "pin3": (PIN3_ROWS, 3, 1), "pin2c": (PIN2C_ROWS, 2, 2)}


def _with_settlement_value(rec, submission, value):
    er = rec.engine_reports[submission]
    return replace(rec, engine_reports=_swap(
        rec.engine_reports, submission, replace(er, settlement_value_eur=float(value))))


def _i6_report_fields(res):
    return {v.field for v in res.violations}


class TestToleranceFormula:
    def test_offset_constant_is_999_900(self):
        pc = OMIEProductConfig()
        assert round(-pc.price_min / pc.price_tick) == 999_900

    def test_card_example_tau(self):
        tau, value = _tau_and_value(PIN1_ROWS)
        assert tau == pytest.approx(6.043878e-4, rel=1e-6)
        assert value == pytest.approx(23.49, rel=1e-12)

    def test_three_row_tau_matches_the_formula_term_by_term(self):
        # A single-dispatch sweep: m = 3 rows, c = 1 chunk.
        tau, _ = _tau_and_value(PIN3_ROWS, chunks=1)
        W = 1_004_598 * 10 + 1_004_650 * 7 + 1_004_701 * 13
        O, Q = 999_900, 30
        k = 0.1 * 0.01 * 0.25
        want = k * U * (1 * W + O * Q + 5 * abs(W - O * Q)) * (1 + 9 * U)
        assert tau == pytest.approx(want, rel=1e-12)

    def test_two_chunk_tau_matches_the_formula_term_by_term(self):
        # m = 3 rows, c = 2 chunks.
        tau, _ = _tau_and_value(PIN2C_ROWS, chunks=2)
        W = P50 * 10 + P50 * 20 + P50 * 10
        O, Q = 999_900, 40
        k = 0.1 * 0.01 * 0.25
        want = k * U * (2 * W + O * Q + 5 * abs(W - O * Q)) * (1 + 10 * U)
        assert tau == pytest.approx(want, rel=1e-12)

    def test_pin3_fills_share_one_engine_timestamp(self, pin3):
        assert len({(f.sec, f.nsec) for f in pin3.fills}) == 1

    def test_pin2c_has_the_constructed_fills_in_exactly_two_chunks(self, pin2c):
        assert [(f.price_int, f.qty_int, f.passive_oid, f.submission)
                for f in pin2c.fills] == [(P50, 10, 1, 2), (P50, 20, 2, 2), (P50, 10, 1, 2)]
        assert len({(f.sec, f.nsec) for f in pin2c.fills}) == 2

    def test_pin_records_have_the_constructed_fills(self, pin1, pin3):
        assert [(f.price_int, f.qty_int, f.submission) for f in pin1.fills] == [
            (1_004_598, 20, 1)]
        assert [(f.price_int, f.qty_int, f.submission) for f in pin3.fills] == [
            (p, q, 3) for p, q in PIN3_ROWS]

    def test_real_engine_settlement_is_inside_tau(self, pin1, pin3, pin2c):
        assert check_i6_settlement(pin1).status == "pass"
        assert check_i6_settlement(pin3).status == "pass"
        assert check_i6_settlement(pin2c).status == "pass"


class TestToleranceBehaviour:
    @pytest.mark.parametrize("which", ["pin1", "pin3", "pin2c"])
    def test_just_inside_tau_passes(self, request, which):
        rec = request.getfixturevalue(which)
        rows, s, c = PINS[which]
        tau, value = _tau_and_value(rows, chunks=c)
        for sign in (+1, -1):
            res = check_i6_settlement(_with_settlement_value(rec, s, value + sign * tau * (1 - 1e-6)))
            assert res.status == "pass", (sign, res.violations)

    @pytest.mark.parametrize("which", ["pin1", "pin3", "pin2c"])
    def test_just_outside_tau_fails_on_engine_reports(self, request, which):
        rec = request.getfixturevalue(which)
        rows, s, c = PINS[which]
        tau, value = _tau_and_value(rows, chunks=c)
        for sign in (+1, -1):
            res = check_i6_settlement(_with_settlement_value(rec, s, value + sign * tau * (1 + 1e-6)))
            assert res.status == "fail", sign
            assert "engine_reports" in _i6_report_fields(res)

    def test_settlement_sum_just_inside_the_summed_tau_passes(self, pin1):
        tau, value = _tau_and_value(PIN1_ROWS)
        es = {1: value, 2: -value + tau * (1 - 1e-6)}
        res = check_i6_settlement(replace(pin1, engine_settlements=es))
        assert res.status == "pass", res.violations

    def test_settlement_sum_just_outside_the_summed_tau_fails(self, pin1):
        tau, value = _tau_and_value(PIN1_ROWS)
        es = {1: value, 2: -value + tau * (1 + 1e-6)}
        res = check_i6_settlement(replace(pin1, engine_settlements=es))
        assert res.status == "fail"
        assert "engine_settlements" in _i6_report_fields(res)

    def test_reported_quantity_must_be_exact_in_ticks(self, pin1):
        er = pin1.engine_reports[1]
        ok = replace(pin1, engine_reports=_swap(pin1.engine_reports, 1,
                                                replace(er, trade_qty_mw=2.0 + 0.04)))
        bad = replace(pin1, engine_reports=_swap(pin1.engine_reports, 1,
                                                 replace(er, trade_qty_mw=2.0 + 0.06)))
        assert check_i6_settlement(ok).status == "pass"
        assert check_i6_settlement(bad).status == "fail"


# ---------------------------------------------------------------------------
# V15(d) — one passing real-engine instance per M check
# ---------------------------------------------------------------------------
def _first_seed(key, pred, limit=64):
    for seed in range(limit):
        d = CONSTRUCTIONS[key](seed)
        if pred(d):
            return d
    raise AssertionError(f"no seed < {limit} of {key} satisfies the predicate")


class TestMetamorphicRealInstances:
    @pytest.mark.parametrize("regime", ["none", "partial", "full"])
    def test_m3_on_c07_ioc_regimes_vs_its_counterfactual(self, regime):
        draw = _first_seed("C07.ioc_regimes", lambda d: d.params["regime"] == regime)
        ioc = record_from_draw(draw)
        non = build_record(draw.aux_tapes["counterfactual"])
        res = check_m3_ioc_non_equivalence(ioc, non, oid=int(draw.params["agg_oid"]))
        assert res.status == "pass", res.violations

    def test_m6_on_c03_same_owner_pair(self):
        draw = CONSTRUCTIONS["C03.same_owner_pair"](0)
        base_tape = draw.aux_tapes["counterfactual"]
        agent_map = {}
        for b_row, s_row in zip(base_tape, draw.tape):
            if b_row[4] >= 1:
                agent_map[int(b_row[5])] = int(s_row[5])
        assert set(agent_map.values()) == {int(draw.params["owner_agent_id"])}
        base = build_record(base_tape)
        same = record_from_draw(draw)
        assert len(same.fills) >= 1  # §6.6: it must trade with its own owner
        res = check_m6_ownership_blindness(base, same, agent_map=agent_map)
        assert res.status == "pass", res.violations

    def test_m4_on_c04_iceberg_cancel_fok(self):
        draw = CONSTRUCTIONS["C04.iceberg_cancel_fok"](0)
        rec = record_from_draw(draw)
        res = check_m4_fok_dichotomy(rec, fok_oid=int(draw.params["fok_oid"]))
        assert res.status == "pass", res.violations

    def test_m1_on_a_tie_permutation(self):
        rows = (_row(SELL, P50, 10, 1, timestamp=5, order_id=1),
                _row(SELL, P50, 10, 2, timestamp=5, order_id=2),
                _row(BUY, P51, 15, 3, timestamp=6, order_id=3))
        base = build_record(_tape(*rows))
        permuted = build_record(_tape(rows[1], rows[0], rows[2]))
        res = check_m1_tie_permutation(base, permuted, tie_oids=(1, 2))
        assert res.status == "pass", res.violations

    def test_m2_on_size_and_limit_variants(self):
        rests = (_row(SELL, P50, 10, 1), _row(SELL, P51, 10, 2))
        small = build_record(_tape(*rests, _row(BUY, P52, 10, 3)))
        large = build_record(_tape(*rests, _row(BUY, P52, 15, 3)))
        tight = build_record(_tape(*rests, _row(BUY, P50, 15, 3)))
        assert check_m2_monotonicity(small, large, aggressor_oid=3,
                                     relation="size_increased").status == "pass"
        assert check_m2_monotonicity(large, tight, aggressor_oid=3,
                                     relation="limit_tightened").status == "pass"

    def test_m5_on_a_split_aggressor(self):
        rests = (_row(SELL, P50, 10, 1), _row(SELL, P51, 10, 2))
        whole = build_record(_tape(*rests, _row(BUY, P52, 20, 3)))
        split = build_record(_tape(*rests, _row(BUY, P52, 10, 3), _row(BUY, P52, 10, 4)))
        res = check_m5_split_aggressor(whole, split, whole_oid=3, split_oids=(3, 4))
        assert res.status == "pass", res.violations


# ---------------------------------------------------------------------------
# V6 / V9 / V13 — the API surface
# ---------------------------------------------------------------------------
_NAMED = {
    "RB": "check_record", "I1": "check_i1_volume_balance",
    "I2": "check_i2_quantity_conservation", "I3": "check_i3_uncrossed_on_return",
    "I4": "check_i4_priority", "I5": "check_i5_price_setter", "I6": "check_i6_settlement",
    "I7": "check_i7_no_manufactured_liquidity", "I8": "check_i8_capacity_headroom",
    "I9": "check_i9_non_vacuity", "M1": "check_m1_tie_permutation",
    "M2": "check_m2_monotonicity", "M3": "check_m3_ioc_non_equivalence",
    "M4": "check_m4_fok_dichotomy", "M5": "check_m5_split_aggressor",
    "M6": "check_m6_ownership_blindness",
}


class TestApiSurface:
    def test_all_checks_order(self):
        assert tuple(ALL_CHECKS) == EXPECTED_ALL_CHECKS

    def test_checks_maps_every_id_to_its_named_function(self):
        assert set(CHECKS) == set(EXPECTED_ALL_CHECKS)
        g = globals()
        for cid, name in _NAMED.items():
            assert CHECKS[cid] is g[name], cid

    def test_rule3_hop_switch_is_pinned_to_the_iceberg(self):
        assert RULE3_HOP_PRICE_SETTER == "iceberg"

    def test_check_results_carry_their_id_and_a_known_status(self, basic):
        for cid in BATTERY_ORDER:
            res = CHECKS[cid](basic) if cid != "I9" else CHECKS[cid](basic, expects_trade=True)
            assert isinstance(res, CheckResult)
            assert res.check_id == cid
            assert res.status in STATUSES
            assert set(res.unobserved) <= UNOBSERVED_VOCAB

    def test_fixed_unobserved_vocabulary(self, basic):
        assert tuple(check_i4_priority(basic).unobserved) == ("queue_timestamp_correctness",)
        assert tuple(check_i6_settlement(basic).unobserved) == ("independent_valuation",)
        assert tuple(check_i8_capacity_headroom(basic).unobserved) == ("matching_iterations",)

    def test_i4_is_not_applicable_on_a_run_with_no_fills(self, nocross):
        res = check_i4_priority(nocross)
        assert res.status == "not_applicable"
        assert res.evidence["reason"]

    def test_i9_requires_the_expects_trade_keyword(self, basic):
        with pytest.raises(TypeError):
            check_i9_non_vacuity(basic)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            check_i9_non_vacuity(basic, True)  # type: ignore[misc]

    def test_run_battery_requires_the_expects_trade_keyword(self, basic):
        with pytest.raises(TypeError):
            run_battery(basic)  # type: ignore[call-arg]

    def test_checks_never_raise_on_content(self, garbage):
        for cid in BATTERY_ORDER:
            res = CHECKS[cid](garbage) if cid != "I9" else CHECKS[cid](garbage, expects_trade=True)
            assert isinstance(res, CheckResult), cid
            if cid != "I8":
                assert res.status != "invalid", cid

    def test_violations_name_their_check_and_field(self, garbage):
        res = check_i1_volume_balance(garbage)
        assert res.status == "fail"
        for v in res.violations:
            assert isinstance(v, Violation)
            assert v.check_id == "I1"
            assert isinstance(v.field, str) and v.field
            assert isinstance(v.message, str)


class TestRunBattery:
    def test_runs_rb_and_i1_to_i9_in_order(self, basic):
        report = run_battery(basic, expects_trade=True)
        assert tuple(r.check_id for r in report.results) == BATTERY_ORDER
        assert report.passed is True and report.valid is True

    def test_does_not_short_circuit_on_failure(self, garbage):
        report = run_battery(garbage, expects_trade=True)
        assert tuple(r.check_id for r in report.results) == BATTERY_ORDER
        assert report.passed is False

    def test_expects_trade_is_forwarded_to_i9(self, nocross):
        on = {r.check_id: r.status for r in run_battery(nocross, expects_trade=True).results}
        off = {r.check_id: r.status for r in run_battery(nocross, expects_trade=False).results}
        assert on["I9"] == "fail" and off["I9"] == "pass"

    def test_a_capacity_breach_makes_the_report_invalid(self, basic):
        broken = replace(basic, flags=_with_key(basic.flags, "book_full", True))
        report = run_battery(broken, expects_trade=True)
        assert report.valid is False
        assert report.passed is False

    def test_allowing_the_breach_makes_i8_not_applicable(self, basic):
        broken = replace(basic, flags=_with_key(basic.flags, "book_full", True))
        report = run_battery(broken, expects_trade=True, allow_capacity_breach=True)
        i8 = {r.check_id: r for r in report.results}["I8"]
        assert i8.status == "not_applicable"
        assert report.valid is True

    def test_the_envelope_is_forwarded_to_i8(self, basic):
        report = run_battery(basic, expects_trade=True, envelope=CapacityEnvelope(max_trades=4))
        i8 = {r.check_id: r for r in report.results}["I8"]
        assert i8.status == "invalid"
        assert report.valid is False


class TestAssertCheckFails:
    @staticmethod
    def _res(cid="I1", status="fail", fields=("fills",)):
        vs = tuple(Violation(check_id=cid, field=f, message="m") for f in fields)
        return CheckResult(check_id=cid, status=status, violations=vs, unobserved=(),
                           evidence={})

    def test_returns_the_result_on_a_matching_failure(self):
        r = self._res()
        assert assert_check_fails(r, "I1") is r
        assert assert_check_fails(r, "I1", field="fills") is r

    def test_raises_on_a_pass(self):
        with pytest.raises(AssertionError):
            assert_check_fails(self._res(status="pass", fields=()), "I1")

    def test_raises_on_a_not_applicable(self):
        with pytest.raises(AssertionError):
            assert_check_fails(self._res(status="not_applicable", fields=()), "I1")

    def test_raises_on_the_wrong_check_id(self):
        with pytest.raises(AssertionError):
            assert_check_fails(self._res(cid="I2"), "I1")

    def test_raises_when_the_field_is_absent(self):
        with pytest.raises(AssertionError):
            assert_check_fails(self._res(fields=("fills",)), "I1", field="evicted")

    def test_status_invalid_must_be_asked_for(self):
        r = self._res(cid="I8", status="invalid", fields=("flags",))
        assert assert_check_fails(r, "I8", status="invalid") is r
        with pytest.raises(AssertionError):
            assert_check_fails(r, "I8")

    def test_raises_on_a_non_result(self):
        with pytest.raises(AssertionError):
            assert_check_fails({"check_id": "I1", "status": "fail"}, "I1")  # type: ignore[arg-type]
