"""OMIE-CERT-INVARIANTS-1 — locked tests for the body-inspecting guard (V14, AC 2).

``battery_guard.uncovered(source)`` parses a test module with ``ast`` and returns
the ``REQUIRED_BROKEN_FIXTURES`` entries that the module does **not** cover. A body
covers ``(ID, field)`` only if it is a non-skipped ``test_*`` function (module level
or inside a ``Test*`` class) containing the statement
``assert_check_fails(<arg0>, "<ID>", field="<field>")`` as a **top-level statement**
of its body (V14 as amended at v5.186: never inside any compound statement —
``if``, ``try``, ``with``, ``for``, ``while``, ``match``, inner ``def`` — and never
after a top-level ``return``/``raise``/``pytest.skip``/``xfail``/``fail``), whose
``<arg0>`` is a call to ``CHECKS[ID]``'s function or a name the same body bound
from one by an earlier top-level assignment. Test **names are never read**. I8's
entry is covered only with ``status="invalid"``.

The guard is first seen failing on decoy sources (rule 1 applies to the guard
too), then run over the two locked broken-fixture files, which together must
leave nothing uncovered. CPU only; no engine run.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

try:  # the module under test does not exist before Step 4; keep collection green
    from tests.omiecid.reference.battery_guard import REQUIRED_BROKEN_FIXTURES, uncovered

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only before implementation
    _IMPORT_ERROR = exc


@pytest.fixture(autouse=True)
def _require_guard_module():
    if _IMPORT_ERROR is not None:
        pytest.fail(
            "tests/omiecid/reference/battery_guard.py is not importable: "
            f"{_IMPORT_ERROR!r}"
        )


HERE = Path(__file__).resolve().parent
CHECKS_FILE = HERE / "test_omie_invariants_checks.py"
RECORD_FILE = HERE / "test_omie_invariants_record.py"

HEADER = textwrap.dedent("""\
    import pytest
    from tests.omiecid.reference.omie_invariants import (
        assert_check_fails,
        check_i1_volume_balance,
        check_i2_quantity_conservation,
        check_i8_capacity_headroom,
        check_record,
    )
""")

RB_FIELDS = ("pre_snapshots", "post_snapshots", "fills", "filled", "still_resting",
             "hidden_remaining", "rejections", "engine_reports", "evicted",
             "observed_capacity", "engine_settlements",
             # V14 as amended at Step 5 (v5.187)
             "cancelled_or_expired", "flags", "accepted", "participants")


def _src(body: str) -> str:
    return HEADER + "\n" + textwrap.dedent(body)


def _uncov(body: str):
    return uncovered(_src(body))


I1 = ("I1", None)


# ---------------------------------------------------------------------------
# The required list
# ---------------------------------------------------------------------------
class TestRequiredList:
    def test_every_check_needs_a_broken_fixture(self):
        req = set(REQUIRED_BROKEN_FIXTURES)
        for cid in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9",
                    "M1", "M2", "M3", "M4", "M5", "M6"):
            assert (cid, None) in req, cid

    def test_the_named_sub_claims_are_required(self):
        req = set(REQUIRED_BROKEN_FIXTURES)
        for item in (("I2", "evicted"), ("I2", "cancelled_or_expired"),
                     ("I6", "engine_settlements"), ("I6", "engine_reports"),
                     ("I5", "unruled_hop"), ("I5", "rule3_hop"),
                     ("I1", "fill_sides"),
                     # V8 / V9 as amended at Step 5 (v5.187), D3-01 / D3-02
                     ("I4", "hop_priority"), ("I5", "limit_cross")):
            assert item in req, item

    def test_every_record_field_is_required(self):
        req = set(REQUIRED_BROKEN_FIXTURES)
        for f in RB_FIELDS:
            assert ("RB", f) in req, f

    def test_an_empty_module_covers_nothing(self):
        assert uncovered("") == set(REQUIRED_BROKEN_FIXTURES)

    def test_uncovered_returns_a_set_of_pairs(self):
        out = uncovered(_src("def test_x():\n    pass\n"))
        assert isinstance(out, set)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in out)


# ---------------------------------------------------------------------------
# What counts
# ---------------------------------------------------------------------------
class TestCountedBodies:
    def test_an_assigned_result_counts(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i1_volume_balance(rec)
                assert_check_fails(res, "I1")
        """)
        assert I1 not in out

    def test_a_direct_call_counts(self):
        out = _uncov("""
            def test_anything(rec):
                assert_check_fails(check_i1_volume_balance(rec), "I1")
        """)
        assert I1 not in out

    def test_a_method_of_a_test_class_counts(self):
        out = _uncov("""
            class TestSomething:
                def test_anything(self, rec):
                    res = check_i1_volume_balance(rec)
                    assert_check_fails(res, "I1")
        """)
        assert I1 not in out

    def test_test_names_are_never_read(self):
        out = _uncov("""
            def test_zzz_unrelated_name(rec):
                res = check_i1_volume_balance(rec)
                assert_check_fails(res, "I1")
        """)
        assert I1 not in out

    def test_a_field_keyword_covers_that_field(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i2_quantity_conservation(rec)
                assert_check_fails(res, "I2", field="evicted")
        """)
        assert ("I2", "evicted") not in out

    def test_a_record_field_is_covered_through_check_record(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_record(rec)
                assert_check_fails(res, "RB", field="filled")
        """)
        assert ("RB", "filled") not in out

    def test_i8_counts_with_status_invalid(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i8_capacity_headroom(rec)
                assert_check_fails(res, "I8", status="invalid")
        """)
        assert ("I8", None) not in out

    def test_a_non_skip_decorator_does_not_hide_a_body(self):
        out = _uncov("""
            @pytest.mark.parametrize("x", [1])
            def test_anything(rec, x):
                res = check_i1_volume_balance(rec)
                assert_check_fails(res, "I1")
        """)
        assert I1 not in out

    def test_a_top_level_call_after_a_with_block_that_does_not_contain_it_counts(self):
        # V14 as amended at Step 3 (v5.186): a preceding compound statement does
        # not disqualify a later top-level call.
        out = _uncov("""
            import jax

            def test_anything(rec):
                with jax.disable_jit():
                    warm = rec
                res = check_i1_volume_balance(warm)
                assert_check_fails(res, "I1")
        """)
        assert I1 not in out


# ---------------------------------------------------------------------------
# Decoys — the guard must be seen failing
# ---------------------------------------------------------------------------
class TestDecoys:
    def test_a_test_named_for_i1_whose_body_asserts_a_pass(self):
        out = _uncov("""
            def test_i1_catches_a_same_side_match(rec):
                res = check_i1_volume_balance(rec)
                assert res.status == "pass"
        """)
        assert I1 in out

    def test_the_helper_under_if_false(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i1_volume_balance(rec)
                if False:
                    assert_check_fails(res, "I1")
        """)
        assert I1 in out

    def test_an_i2_result_passed_as_i1(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i2_quantity_conservation(rec)
                assert_check_fails(res, "I1")
        """)
        assert I1 in out

    def test_an_i2_call_passed_directly_as_i1(self):
        out = _uncov("""
            def test_anything(rec):
                assert_check_fails(check_i2_quantity_conservation(rec), "I1")
        """)
        assert I1 in out

    @pytest.mark.parametrize("decorator", [
        '@pytest.mark.skip(reason="later")',
        '@pytest.mark.skipif(True, reason="later")',
        "@pytest.mark.xfail",
        '@pytest.mark.xfail(reason="later")',
    ])
    def test_a_skip_or_xfail_decorated_test(self, decorator):
        out = _uncov(f"""
{decorator}
def test_anything(rec):
    res = check_i1_volume_balance(rec)
    assert_check_fails(res, "I1")
""")
        assert I1 in out

    def test_the_helper_inside_try(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i1_volume_balance(rec)
                try:
                    assert_check_fails(res, "I1")
                except AssertionError:
                    pass
        """)
        assert I1 in out

    def test_the_helper_inside_an_inner_def(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i1_volume_balance(rec)
                def inner():
                    assert_check_fails(res, "I1")
        """)
        assert I1 in out

    def test_a_non_test_function(self):
        out = _uncov("""
            def helper_for_i1(rec):
                res = check_i1_volume_balance(rec)
                assert_check_fails(res, "I1")
        """)
        assert I1 in out

    def test_a_test_method_of_a_non_test_class(self):
        out = _uncov("""
            class Helpers:
                def test_anything(self, rec):
                    res = check_i1_volume_balance(rec)
                    assert_check_fails(res, "I1")
        """)
        assert I1 in out

    def test_a_name_assigned_in_another_body(self):
        out = _uncov("""
            def test_one(rec):
                res = check_i1_volume_balance(rec)

            def test_two(rec, res):
                assert_check_fails(res, "I1")
        """)
        assert I1 in out

    def test_an_argument_that_is_not_a_check_result(self):
        out = _uncov("""
            def test_anything(rec, res):
                assert_check_fails(res, "I1")
        """)
        assert I1 in out

    def test_the_wrong_field_does_not_cover(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i2_quantity_conservation(rec)
                assert_check_fails(res, "I2", field="filled")
        """)
        assert ("I2", "evicted") in out

    def test_i8_without_status_invalid(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i8_capacity_headroom(rec)
                assert_check_fails(res, "I8")
        """)
        assert ("I8", None) in out

    def test_a_record_field_asserted_through_the_wrong_check(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i1_volume_balance(rec)
                assert_check_fails(res, "RB", field="filled")
        """)
        assert ("RB", "filled") in out

    # --- V14 as amended at Step 3 (v5.186): only top-level statements count ---
    def test_the_helper_under_pytest_raises(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i1_volume_balance(rec)
                with pytest.raises(AssertionError):
                    assert_check_fails(res, "I1")
        """)
        assert I1 in out

    def test_the_helper_under_an_empty_for_loop(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i1_volume_balance(rec)
                for _ in ():
                    assert_check_fails(res, "I1")
        """)
        assert I1 in out

    def test_the_helper_after_a_top_level_return(self):
        out = _uncov("""
            def test_anything(rec):
                res = check_i1_volume_balance(rec)
                return
                assert_check_fails(res, "I1")
        """)
        assert I1 in out


# ---------------------------------------------------------------------------
# The guard over the locked suite
# ---------------------------------------------------------------------------
class TestLockedSuiteIsCovered:
    def test_both_files_exist(self):
        assert CHECKS_FILE.is_file()
        assert RECORD_FILE.is_file()

    def test_nothing_required_is_left_uncovered(self):
        missing = uncovered(CHECKS_FILE.read_text()) & uncovered(RECORD_FILE.read_text())
        assert missing == set(), sorted(missing, key=lambda t: (t[0], t[1] or ""))

    def test_the_checks_file_covers_every_check(self):
        out = uncovered(CHECKS_FILE.read_text())
        for cid in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9",
                    "M1", "M2", "M3", "M4", "M5", "M6"):
            assert (cid, None) not in out, cid

    def test_the_record_file_covers_every_record_field(self):
        out = uncovered(RECORD_FILE.read_text())
        for f in RB_FIELDS:
            assert ("RB", f) not in out, f
