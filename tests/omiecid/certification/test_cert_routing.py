"""OMIE-CERT-HARNESS-1 -- the K5 routing rule (decision H9; AC (4)).

The rule is **imported, not copied**: every assertion below runs against
``notebooks.certification.cert_support.route_k5`` itself. Expected routes are
written as literal per-case values, never computed by a test-side function (the
first attempt's test asserted against its own copy of the rule, which proves
nothing about the rule the cycle uses).
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_SUPPORT = REPO_ROOT / "notebooks" / "certification" / "cert_support.py"


@pytest.fixture(scope="module")
def rule():
    """Import the production rule by its documented path (H1, H9)."""
    from notebooks.certification.cert_support import K5Outcome, Route, result_word, route_k5
    return route_k5, Route, K5Outcome, result_word


def _outcome(K5Outcome, **over):
    base = dict(notebook_executed=True, results_status="PASS", audit_twin_passed=True,
                suite_regressed=False, independence_ok=True)
    base.update(over)
    return K5Outcome(**base)


def test_import_path_is_the_documented_one(rule):
    import notebooks.certification.cert_support as cs
    route_k5, Route, K5Outcome, result_word = rule
    assert route_k5 is cs.route_k5
    assert route_k5.__module__ == "notebooks.certification.cert_support"


def test_cert_support_defines_the_rule_exactly_once():
    tree = ast.parse(CERT_SUPPORT.read_text(encoding="utf-8"))
    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "route_k5"]
    assert len(defs) == 1, "route_k5 must be defined exactly once in cert_support.py"
    # no other function reads the suite-regression flag: the decision lives only in route_k5
    readers = sorted({
        fn.name for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name != "route_k5"
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute) and node.attr == "suite_regressed"
    })
    assert readers == [], f"routing logic duplicated in: {readers}"


def test_this_test_module_defines_no_copy_of_the_rule():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "route_k5" not in names and not any(n.startswith("_route") for n in names)


class TestRouteEnum:
    def test_exactly_three_members(self, rule):
        _, Route, _, _ = rule
        assert {m.name: m.value for m in Route} == {
            "K6": "K6",
            "K6F_MERGE_EVIDENCE": "K6-F merge-evidence",
            "K6F_RECORD_ONLY": "K6-F record-only",
        }

    def test_route_is_a_str_enum(self, rule):
        _, Route, _, _ = rule
        assert isinstance(Route.K6, str) and Route.K6 == "K6"

    def test_merges_property(self, rule):
        _, Route, _, _ = rule
        assert Route.K6.merges is True
        assert Route.K6F_MERGE_EVIDENCE.merges is True
        assert Route.K6F_RECORD_ONLY.merges is False

    def test_signs_certificate_property(self, rule):
        _, Route, _, _ = rule
        assert Route.K6.signs_certificate is True
        assert Route.K6F_MERGE_EVIDENCE.signs_certificate is False
        assert Route.K6F_RECORD_ONLY.signs_certificate is False


class TestK5Outcome:
    def test_frozen(self, rule):
        _, _, K5Outcome, _ = rule
        o = _outcome(K5Outcome)
        with pytest.raises(dataclasses.FrozenInstanceError):
            o.suite_regressed = True  # type: ignore[misc]

    def test_fields(self, rule):
        _, _, K5Outcome, _ = rule
        assert [f.name for f in dataclasses.fields(K5Outcome)] == [
            "notebook_executed", "results_status", "audit_twin_passed", "suite_regressed", "independence_ok"]


class TestRouting:
    def test_all_pass_routes_k6(self, rule):
        route_k5, Route, K5Outcome, _ = rule
        assert route_k5(_outcome(K5Outcome)) is Route.K6

    @pytest.mark.parametrize("over", [
        {"notebook_executed": False},
        {"notebook_executed": False, "results_status": None},
        {"results_status": "NOT CERTIFIED"},
        {"results_status": None},
        {"results_status": "pass"},
        {"audit_twin_passed": False},
        {"independence_ok": False},
    ], ids=["not-executed", "crashed-no-results", "not-certified", "no-results", "status-case", "audit-twin",
            "independence"])
    def test_each_single_failure_routes_merge_evidence(self, rule, over):
        route_k5, Route, K5Outcome, _ = rule
        assert route_k5(_outcome(K5Outcome, **over)) is Route.K6F_MERGE_EVIDENCE

    def test_all_failures_without_regression_route_merge_evidence(self, rule):
        route_k5, Route, K5Outcome, _ = rule
        o = _outcome(K5Outcome, notebook_executed=False, results_status=None, audit_twin_passed=False,
                     independence_ok=False)
        assert route_k5(o) is Route.K6F_MERGE_EVIDENCE

    def test_regression_alone_routes_record_only(self, rule):
        route_k5, Route, K5Outcome, _ = rule
        assert route_k5(_outcome(K5Outcome, suite_regressed=True)) is Route.K6F_RECORD_ONLY

    @pytest.mark.parametrize("over", [
        {"notebook_executed": False},
        {"results_status": "NOT CERTIFIED"},
        {"results_status": None},
        {"audit_twin_passed": False},
        {"independence_ok": False},
        {"notebook_executed": False, "results_status": None, "audit_twin_passed": False,
         "independence_ok": False},
    ], ids=["not-executed", "not-certified", "no-results", "audit-twin", "independence", "everything"])
    def test_regression_with_other_failures_routes_record_only(self, rule, over):
        route_k5, Route, K5Outcome, _ = rule
        assert route_k5(_outcome(K5Outcome, suite_regressed=True, **over)) is Route.K6F_RECORD_ONLY

    def test_route_is_always_one_of_three(self, rule):
        """There is no stop / await member: every outcome yields a route that continues the ladder."""
        route_k5, Route, K5Outcome, _ = rule
        seen = set()
        for executed in (True, False):
            for status in ("PASS", "NOT CERTIFIED", None):
                for twin in (True, False):
                    for reg in (True, False):
                        for ind in (True, False):
                            r = route_k5(K5Outcome(executed, status, twin, reg, ind))
                            assert isinstance(r, Route)
                            seen.add(r)
        assert seen == set(Route)


class TestResultWord:
    def test_k6(self, rule):
        _, Route, _, result_word = rule
        assert result_word(Route.K6) == "CERTIFIED"

    def test_k6_provisional_keeps_given_order(self, rule):
        _, Route, _, result_word = rule
        assert result_word(Route.K6, provisional_on=("C05", "C02")) == "CERTIFIED — provisional on C05, C02"
        assert result_word(Route.K6, ("C02",)) == "CERTIFIED — provisional on C02"

    @pytest.mark.parametrize("name", ["K6F_MERGE_EVIDENCE", "K6F_RECORD_ONLY"])
    def test_k6f_is_not_certified(self, rule, name):
        _, Route, _, result_word = rule
        assert result_word(Route[name]) == "NOT CERTIFIED"
        assert result_word(Route[name], provisional_on=("C02",)) == "NOT CERTIFIED"


class TestConstants:
    def test_human_verdicts(self):
        from notebooks.certification.cert_support import HUMAN_VERDICTS
        assert HUMAN_VERDICTS == ("REPEAT", "DEFECT-FOUND", "SCOPE-CHANGE", "INVALID")

    def test_ladder(self):
        from notebooks.certification.cert_support import LADDER
        assert LADDER == tuple(f"C{i:02d}" for i in range(18))
