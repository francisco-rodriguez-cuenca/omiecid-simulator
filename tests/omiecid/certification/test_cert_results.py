"""OMIE-CERT-HARNESS-1 -- results JSON schema, ``validate_results`` and the
``Certification`` accumulator (decisions H5, H6).

Locked behavioural contract:

* the validator accepts a well-formed payload and rejects each semantic
  inconsistency of H5 (``failure_kinds`` must equal the set derived from the
  payload; PASS <=> no failure kind <=> null explanation; discard rate);
* the accumulator records every failure without raising, an I8 ``invalid``
  result discards the draw, and ``valid < requested`` derives ``capacity_flag``.

Payloads are produced by the accumulator itself (the production path), then
mutated one field at a time, so each rejection isolates one rule.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.omiecid.reference.omie_invariants import BatteryReport, CheckResult, Violation

try:  # collection must succeed before the harness exists; the tests then fail
    from notebooks.certification import cert_support as cs
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    cs = None
    _IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "notebooks" / "certification" / "results.schema.json"

CHECK_IDS = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")
TOP_LEVEL_KEYS = (
    "schema_version", "cert_id", "notebook", "scenario", "status", "failure_kinds", "known_result",
    "oracle_kind", "draws", "seeds", "generator", "spec", "citations", "assertions", "invariants",
    "capacity", "eager_jit_identical", "wall_clock_s", "runtime_forecast_s", "failure_explanation",
    "environment",
)
FAILURE_KINDS = ("oracle_mismatch", "invariant_violation", "eager_jit_disagreement", "capacity_flag",
                 "vacuous_run", "quote_drift")


@pytest.fixture(autouse=True)
def _require_harness():
    if cs is None:
        pytest.fail(f"notebooks.certification.cert_support is not importable: {_IMPORT_ERROR!r}")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _good_quote():
    return cs.SpecQuote("market-spec.md", "§6.2 item 3",
                        "Trade executes at the **resting order's limit price** (maker pricing).")


def _bad_quote():
    return cs.SpecQuote("market-spec.md", "§6.2 item 3", "This sentence is in no source document at all.")


def _report(**status) -> BatteryReport:
    results = []
    for cid in CHECK_IDS:
        st = status.get(cid, "pass")
        viol = (Violation(cid, "probe", "deliberately broken input"),) if st in ("fail", "invalid") else ()
        results.append(CheckResult(check_id=cid, status=st, violations=viol, unobserved=(), evidence={}))
    return BatteryReport(results=tuple(results))


def _cert(*, requested=1, quotes=None, cert_id="C98", constructions=(), citations=()):
    return cs.Certification(
        cert_id=cert_id, notebook=f"{cert_id}_probe.ipynb", scenario="Probe scenario",
        known_result="p_trade = p_rest", oracle_kind="exact-parametric",
        quotes=quotes if quotes is not None else (_good_quote(),), citations=citations,
        constructions=constructions, requested_draws=requested,
    )


def _passing(**kw):
    cert = _cert(**kw)
    cert.check("trade price equals resting price", True, seed=0, expected=100, observed=100, unit="price_int")
    cert.set_eager_jit(0, True)
    cert.add_battery(_report(), seed=0)
    return cert


def _with_failure(kind: str):
    """A certification exhibiting exactly the named failure kind."""
    if kind == "capacity_flag":
        cert = _cert(requested=2)
        cert.check("trade price equals resting price", True, seed=0)
        cert.add_battery(_report(), seed=0)
        return cert
    if kind == "quote_drift":
        cert = _cert(quotes=(_bad_quote(),))
        cert.check("trade price equals resting price", True, seed=0)
        cert.add_battery(_report(), seed=0)
        return cert
    cert = _cert()
    cert.check("trade price equals resting price", kind != "oracle_mismatch", seed=0, expected=100,
               observed=100 if kind != "oracle_mismatch" else 101, unit="price_int")
    cert.set_eager_jit(0, kind != "eager_jit_disagreement")
    battery = {"invariant_violation": {"I3": "fail"}, "vacuous_run": {"I9": "fail"}}.get(kind, {})
    cert.add_battery(_report(**battery), seed=0)
    return cert


def _two_failures(kind: str):
    """Oracle mismatch plus ``kind`` (or plus eager-jit when ``kind`` is the oracle itself)."""
    other = "eager_jit_disagreement" if kind == "oracle_mismatch" else kind
    if other == "capacity_flag":
        cert = _cert(requested=2)
    elif other == "quote_drift":
        cert = _cert(quotes=(_bad_quote(),))
    else:
        cert = _cert()
    cert.check("trade price equals resting price", False, seed=0, expected=100, observed=101, unit="price_int")
    cert.set_eager_jit(0, other != "eager_jit_disagreement")
    battery = {"invariant_violation": {"I3": "fail"}, "vacuous_run": {"I9": "fail"}}.get(other, {})
    cert.add_battery(_report(**battery), seed=0)
    return cert


def _row(payload, check_id):
    rows = [r for r in payload["invariants"] if r["check_id"] == check_id]
    assert len(rows) == 1, f"expected one invariants row for {check_id}"
    return rows[0]


# ---------------------------------------------------------------------------
# Schema file
# ---------------------------------------------------------------------------
class TestSchemaFile:
    def test_schema_is_draft_2020_12_and_valid(self):
        import jsonschema
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        assert "2020-12" in schema.get("$schema", "")
        jsonschema.Draft202012Validator.check_schema(schema)

    def test_schema_version_constant(self):
        assert cs.RESULTS_SCHEMA_VERSION == 1

    def test_schema_accepts_accumulator_payload_and_requires_every_key(self):
        import jsonschema
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        v = jsonschema.Draft202012Validator(schema)
        payload = _passing().payload()
        assert list(v.iter_errors(payload)) == []
        broken = dict(payload)
        del broken["draws"]
        assert list(v.iter_errors(broken)) != []


# ---------------------------------------------------------------------------
# Accumulator: well-formed payloads
# ---------------------------------------------------------------------------
class TestPassingPayload:
    def test_valid_and_pass(self):
        cert = _passing()
        p = cert.payload()
        assert cs.validate_results(p) == []
        assert p["status"] == "PASS"
        assert p["failure_kinds"] == []
        assert p["failure_explanation"] is None
        assert cert.verdict_line() == "PASS"
        assert cert.failure_section_markdown() == ""

    def test_top_level_keys(self):
        p = _passing().payload()
        assert set(TOP_LEVEL_KEYS) <= set(p)

    def test_identity_fields(self):
        p = _passing().payload()
        assert p["schema_version"] == 1
        assert p["cert_id"] == "C98"
        assert p["notebook"] == "C98_probe.ipynb"
        assert p["scenario"] == "Probe scenario"
        assert p["known_result"] == "p_trade = p_rest"
        assert p["oracle_kind"] == "exact-parametric"

    def test_draws_block(self):
        p = _passing().payload()
        assert p["draws"] == {"requested": 1, "valid": 1, "discarded": 0, "discard_rate": 0}

    def test_payload_is_json_serialisable(self):
        json.dumps(_passing().payload(), allow_nan=False)

    def test_eager_jit_identical(self):
        p = _passing().payload()
        assert p["eager_jit_identical"] is True
        eager = [a for a in p["assertions"] if a["kind"] == "eager_jit"]
        assert len(eager) == 1 and eager[0]["status"] == "pass"

    def test_spec_block_and_quote_digest(self):
        p = _passing().payload()
        quotes = p["spec"]["quotes"]
        assert len(quotes) == 1
        q = quotes[0]
        assert q["source"] == "market-spec.md" and q["section"] == "§6.2 item 3"
        assert q["block"] is None
        assert q["ok"] is True and q["reason"] == "ok"
        assert q["digest"] == cs.check_quote(_good_quote()).digest
        joined = "".join(x["digest"] for x in quotes)
        assert p["spec"]["quote_digest"] == hashlib.blake2b(joined.encode("ascii"), digest_size=16).hexdigest()

    def test_citations_recorded_unchecked(self):
        c = cs.Citation("OMIE Rules §38.2", "Price bounds: regulatory corpus text")
        p = _passing(citations=(c,)).payload()
        assert p["citations"] == [{"source": "OMIE Rules §38.2",
                                   "text": "Price bounds: regulatory corpus text", "checked": False}]
        assert cs.validate_results(p) == []

    def test_generator_digests(self):
        from omiecid_simulator.jaxomiecid import omie_tape_synth as synth
        p = _passing(constructions=("C02.crossing_pair",)).payload()
        g = p["generator"]
        assert g["module"] == "omiecid_simulator.jaxomiecid.omie_tape_synth"
        assert g["digest_schema_version"] == synth.DIGEST_SCHEMA_VERSION
        assert g["digests"] == {"C02.crossing_pair": synth.construction_digest("C02.crossing_pair")}

    def test_invariant_rows_one_per_check_id(self):
        p = _passing().payload()
        assert sorted(r["check_id"] for r in p["invariants"]) == sorted(CHECK_IDS)
        row = _row(p, "I3")
        assert row["status"] == "pass" and row["n_pass"] == 1 and row["n_fail"] == 0
        assert row["failing_seeds"] == []

    def test_not_applicable_row(self):
        cert = _cert(requested=2)
        cert.add_battery(_report(I7="not_applicable"), seed=0)
        cert.add_battery(_report(I7="not_applicable"), seed=1)
        p = cert.payload()
        row = _row(p, "I7")
        assert row["status"] == "not_applicable" and row["n_not_applicable"] == 2
        assert cs.validate_results(p) == []
        assert p["status"] == "PASS"

    def test_capacity_peaks_elementwise_max(self):
        cert = _passing()
        cert.record_capacity({"resting_per_side": 3, "trades": 1, "hops": 2, "concurrent_icebergs": 0})
        cert.record_capacity({"resting_per_side": 1, "trades": 9, "hops": 2, "concurrent_icebergs": 4})
        p = cert.payload()
        assert p["capacity"]["peaks"] == {"resting_per_side": 3, "trades": 9, "hops": 2, "concurrent_icebergs": 4}
        assert set(p["capacity"]["envelope"]) == {"resting_per_side", "trades", "hops", "concurrent_icebergs"}

    def test_unobserved_peaks_are_null(self):
        p = _passing().payload()
        assert set(p["capacity"]["peaks"]) == {"resting_per_side", "trades", "hops", "concurrent_icebergs"}
        assert all(v is None for v in p["capacity"]["peaks"].values())

    def test_runtime_forecast_default_null(self):
        assert _passing().payload()["runtime_forecast_s"] is None

    def test_environment_block(self):
        env = _passing().payload()["environment"]
        assert set(env) >= {"python", "jax", "backend"}

    def test_table_lists_checks(self):
        t = _passing().table()
        assert isinstance(t, str) and "trade price equals resting price" in t

    def test_emit_writes_validated_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OMIE_CERT_RESULTS_DIR", str(tmp_path))
        path = _passing().emit()
        assert Path(path) == tmp_path / "C98.json"
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert cs.validate_results(data) == []
        assert data["status"] == "PASS"


# ---------------------------------------------------------------------------
# Accumulator: failures are recorded, never raised
# ---------------------------------------------------------------------------
class TestFailuresRecorded:
    def test_failed_check_does_not_raise(self):
        cert = _cert()
        cert.check("deliberately wrong", False, seed=7, expected=100, observed=101, unit="price_int",
                   detail="off by one tick", rule="§6.2 item 3")
        cert.add_battery(_report(), seed=7)
        p = cert.payload()
        a = [x for x in p["assertions"] if x["name"] == "deliberately wrong"][0]
        assert a == {"name": "deliberately wrong", "kind": "oracle", "status": "fail", "seed": 7,
                     "expected": 100, "observed": 101, "unit": "price_int", "detail": "off by one tick",
                     "rule": "§6.2 item 3"}
        assert p["status"] == "NOT CERTIFIED" and p["failure_kinds"] == ["oracle_mismatch"]
        assert cert.verdict_line() == "NOT CERTIFIED"
        assert 7 in p["seeds"]["failing"]
        assert cs.validate_results(p) == []

    def test_no_short_circuit(self):
        cert = _cert()
        for i, ok in enumerate((False, True, False, False)):
            cert.check(f"check-{i}", ok, seed=i)
        cert.add_battery(_report(), seed=0)
        p = cert.payload()
        statuses = {a["name"]: a["status"] for a in p["assertions"] if a["kind"] == "oracle"}
        assert statuses == {"check-0": "fail", "check-1": "pass", "check-2": "fail", "check-3": "fail"}

    def test_check_equal(self):
        cert = _cert()
        cert.check_equal("equal", 5, 5, unit="qty_int")
        cert.check_equal("unequal", 6, 5, unit="qty_int", seed=3)
        cert.add_battery(_report(), seed=0)
        rows = {a["name"]: a for a in cert.payload()["assertions"]}
        assert rows["equal"]["status"] == "pass"
        assert rows["unequal"]["status"] == "fail"
        assert rows["unequal"]["observed"] == 6 and rows["unequal"]["expected"] == 5
        assert rows["unequal"]["unit"] == "qty_int" and rows["unequal"]["seed"] == 3

    def test_eager_jit_disagreement(self):
        cert = _with_failure("eager_jit_disagreement")
        p = cert.payload()
        assert p["failure_kinds"] == ["eager_jit_disagreement"]
        assert p["eager_jit_identical"] is False
        assert cs.validate_results(p) == []

    @pytest.mark.parametrize("cid", ["RB", "I1", "I3", "I5"])
    def test_invariant_fail_is_invariant_violation(self, cid):
        cert = _cert()
        cert.check("oracle", True, seed=5)
        cert.add_battery(_report(**{cid: "fail"}), seed=5)
        p = cert.payload()
        assert p["failure_kinds"] == ["invariant_violation"]
        row = _row(p, cid)
        assert row["status"] == "fail" and row["n_fail"] == 1 and row["failing_seeds"] == [5]
        assert 5 in p["seeds"]["failing"]
        assert cs.validate_results(p) == []

    def test_i9_fail_is_vacuous_run_only(self):
        p = _with_failure("vacuous_run").payload()
        assert p["failure_kinds"] == ["vacuous_run"]
        assert _row(p, "I9")["status"] == "fail"
        assert cs.validate_results(p) == []

    def test_quote_drift(self):
        p = _with_failure("quote_drift").payload()
        assert p["failure_kinds"] == ["quote_drift"]
        assert p["spec"]["quotes"][0]["ok"] is False
        assert p["spec"]["quotes"][0]["reason"] == "not_found"
        assert cs.validate_results(p) == []

    def test_several_kinds_sorted_and_unique(self):
        cert = _cert(quotes=(_bad_quote(),), requested=2)
        cert.check("a", False, seed=0)
        cert.check("b", False, seed=0)
        cert.set_eager_jit(0, False)
        cert.add_battery(_report(I3="fail", I9="fail"), seed=0)
        p = cert.payload()
        assert p["failure_kinds"] == sorted(FAILURE_KINDS)
        assert cs.validate_results(p) == []

    def test_failure_explanation_shape(self):
        cert = _with_failure("oracle_mismatch")
        p = cert.payload()
        fe = p["failure_explanation"]
        assert set(fe) >= {"what_failed", "smallest_case", "rule", "hypothesis", "downstream", "paragraph"}
        assert fe["hypothesis"] is None
        assert isinstance(fe["paragraph"], str) and fe["paragraph"]
        assert "Why this certification failed" in cert.failure_section_markdown()


class TestDrawAccounting:
    def test_i8_invalid_discards_the_draw(self):
        cert = _cert(requested=1)
        cert.check("oracle", True, seed=0)
        cert.add_battery(_report(I8="invalid", I3="fail"), seed=0)
        p = cert.payload()
        assert p["draws"]["valid"] == 0 and p["draws"]["discarded"] == 1
        # the discarded draw's own results are not recorded as invariant outcomes
        assert "invariant_violation" not in p["failure_kinds"]
        assert all(r["status"] != "fail" for r in p["invariants"])
        assert p["failure_kinds"] == ["capacity_flag"]
        assert cs.validate_results(p) == []

    def test_discard_then_valid_draw_meets_request(self):
        cert = _cert(requested=1)
        cert.check("oracle", True, seed=1)
        cert.add_battery(_report(I8="invalid"), seed=0)
        cert.add_battery(_report(), seed=1)
        p = cert.payload()
        assert p["draws"] == {"requested": 1, "valid": 1, "discarded": 1, "discard_rate": 0.5}
        assert p["status"] == "PASS"
        assert cs.validate_results(p) == []

    def test_explicit_discard(self):
        cert = _cert(requested=3)
        for s in range(3):
            cert.add_battery(_report(), seed=s)
        cert.discard(9, "construction rejected the seed")
        p = cert.payload()
        assert p["draws"]["valid"] == 3 and p["draws"]["discarded"] == 1
        assert p["draws"]["discard_rate"] == pytest.approx(0.25)
        assert p["status"] == "PASS"

    def test_valid_below_requested_is_capacity_flag(self):
        p = _with_failure("capacity_flag").payload()
        assert p["draws"]["valid"] == 1 and p["draws"]["requested"] == 2
        assert p["failure_kinds"] == ["capacity_flag"]
        assert p["status"] == "NOT CERTIFIED"
        assert cs.validate_results(p) == []

    def test_invariants_count_valid_draws_only(self):
        cert = _cert(requested=2)
        cert.add_battery(_report(), seed=0)
        cert.add_battery(_report(I8="invalid"), seed=1)
        cert.add_battery(_report(I3="fail"), seed=2)
        p = cert.payload()
        row = _row(p, "I3")
        assert row["n_pass"] == 1 and row["n_fail"] == 1 and row["failing_seeds"] == [2]


# ---------------------------------------------------------------------------
# Validator: each semantic inconsistency rejected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", FAILURE_KINDS)
def test_two_failure_payload_is_valid(kind):
    p = _two_failures(kind).payload()
    expected = sorted({"oracle_mismatch", kind if kind != "oracle_mismatch" else "eager_jit_disagreement"})
    assert p["failure_kinds"] == expected
    assert cs.validate_results(p) == []


@pytest.mark.parametrize("kind", FAILURE_KINDS)
def test_dropping_a_derived_kind_is_rejected(kind):
    p = _two_failures(kind).payload()
    p["failure_kinds"] = [k for k in p["failure_kinds"] if k != kind]
    assert p["failure_kinds"], "the other failure keeps the payload NOT CERTIFIED"
    assert cs.validate_results(p) != []


class TestValidatorRejects:
    def _fail(self):
        return _with_failure("oracle_mismatch").payload()

    def test_validator_accepts_well_formed(self):
        assert cs.validate_results(_passing().payload()) == []
        assert cs.validate_results(self._fail()) == []

    def test_kind_not_derivable(self):
        p = self._fail()
        p["failure_kinds"] = ["eager_jit_disagreement", "oracle_mismatch"]
        assert cs.validate_results(p) != []

    def test_invariant_violation_instead_of_vacuous_run(self):
        p = _with_failure("vacuous_run").payload()
        p["failure_kinds"] = ["invariant_violation"]
        assert cs.validate_results(p) != []

    def test_unsorted_kinds(self):
        p = _two_failures("quote_drift").payload()
        p["failure_kinds"] = sorted(p["failure_kinds"], reverse=True)
        assert cs.validate_results(p) != []

    def test_duplicate_kinds(self):
        p = self._fail()
        p["failure_kinds"] = ["oracle_mismatch", "oracle_mismatch"]
        assert cs.validate_results(p) != []

    def test_unknown_kind(self):
        p = self._fail()
        p["failure_kinds"] = ["oracle_mismatch", "zz_unknown"]
        assert cs.validate_results(p) != []

    def test_pass_status_with_failures(self):
        p = self._fail()
        p["status"] = "PASS"
        assert cs.validate_results(p) != []

    def test_not_certified_without_failures(self):
        p = _passing().payload()
        p["status"] = "NOT CERTIFIED"
        p["failure_explanation"] = self._fail()["failure_explanation"]
        assert cs.validate_results(p) != []

    def test_explanation_missing_on_not_certified(self):
        p = self._fail()
        p["failure_explanation"] = None
        assert cs.validate_results(p) != []

    def test_explanation_present_on_pass(self):
        p = _passing().payload()
        p["failure_explanation"] = self._fail()["failure_explanation"]
        assert cs.validate_results(p) != []

    def test_wrong_discard_rate(self):
        cert = _cert(requested=1)
        cert.add_battery(_report(I8="invalid"), seed=0)
        cert.add_battery(_report(), seed=1)
        p = cert.payload()
        assert cs.validate_results(p) == []
        p["draws"]["discard_rate"] = 0.25
        assert cs.validate_results(p) != []

    def test_discard_rate_nonzero_with_no_draws(self):
        p = _passing().payload()
        p["draws"] = {"requested": 0, "valid": 0, "discarded": 0, "discard_rate": 0.5}
        assert cs.validate_results(p) != []

    def test_capacity_flag_missing(self):
        p = _passing().payload()
        p["draws"]["requested"] = 2
        assert cs.validate_results(p) != []

    def test_failing_quote_without_quote_drift(self):
        p = _passing().payload()
        p["spec"]["quotes"][0]["ok"] = False
        p["spec"]["quotes"][0]["reason"] = "not_found"
        assert cs.validate_results(p) != []

    def test_failing_eager_jit_assertion_without_kind(self):
        p = _passing().payload()
        for a in p["assertions"]:
            if a["kind"] == "eager_jit":
                a["status"] = "fail"
        assert cs.validate_results(p) != []

    @pytest.mark.parametrize("cert_id", ["C1", "C001", "X01", "c01", "C9a"])
    def test_bad_cert_id(self, cert_id):
        p = _passing().payload()
        p["cert_id"] = cert_id
        assert cs.validate_results(p) != []

    @pytest.mark.parametrize("cert_id", ["C00", "C17", "C90", "C99"])
    def test_good_cert_id(self, cert_id):
        p = _passing().payload()
        p["cert_id"] = cert_id
        assert cs.validate_results(p) == []

    def test_bad_status_word(self):
        p = _passing().payload()
        p["status"] = "CERTIFIED"
        assert cs.validate_results(p) != []

    def test_bad_schema_version(self):
        p = _passing().payload()
        p["schema_version"] = 2
        assert cs.validate_results(p) != []

    @pytest.mark.parametrize("key", TOP_LEVEL_KEYS)
    def test_missing_top_level_key(self, key):
        p = _passing().payload()
        del p[key]
        assert cs.validate_results(p) != []

    def test_bad_assertion_kind(self):
        p = _passing().payload()
        p["assertions"][0]["kind"] = "magic"
        assert cs.validate_results(p) != []

    def test_bad_assertion_unit(self):
        p = _passing().payload()
        p["assertions"][0]["unit"] = "euros"
        assert cs.validate_results(p) != []

    def test_invariant_row_status_invalid_is_not_a_row_status(self):
        p = _passing().payload()
        p["invariants"][0]["status"] = "invalid"
        assert cs.validate_results(p) != []

    def test_citation_marked_checked(self):
        c = cs.Citation("OMIE Rules §38.2", "Price bounds: regulatory corpus text")
        p = _passing(citations=(c,)).payload()
        p["citations"][0]["checked"] = True
        assert cs.validate_results(p) != []

    def test_garbage_reports_instead_of_raising(self):
        assert cs.validate_results({}) != []

    def test_mutation_does_not_leak_into_accumulator(self):
        cert = _passing()
        p = cert.payload()
        q = copy.deepcopy(p)
        p["status"] = "garbage"
        assert cert.payload()["status"] == q["status"] == "PASS"


# ---------------------------------------------------------------------------
# Step-5 re-close ruling (agile-plan v5.189, R3b / R3c)
# ---------------------------------------------------------------------------
def _other_hex(h: str) -> str:
    """A different 32-hex value."""
    return ("0" if h[0] != "0" else "1") + h[1:]


class TestQuoteDigestCheck:
    """V1 -- ``spec.quote_digest`` must equal blake2b-128 over the ASCII concatenation of
    ``spec.quotes[*].digest``, in order (digest of b"" for an empty list)."""

    def test_v1_replaced_quote_digest_rejected(self):
        p = _passing().payload()
        p["spec"]["quote_digest"] = _other_hex(p["spec"]["quote_digest"])
        assert cs.validate_results(p) != []

    def test_v1_replaced_per_quote_digest_rejected(self):
        p = _passing().payload()
        p["spec"]["quotes"][0]["digest"] = _other_hex(p["spec"]["quotes"][0]["digest"])
        assert cs.validate_results(p) != []

    def test_v1_empty_quotes_with_digest_of_empty_accepted(self):
        p = _passing(quotes=()).payload()
        assert p["spec"]["quotes"] == []
        p["spec"]["quote_digest"] = hashlib.blake2b(b"", digest_size=16).hexdigest()
        assert cs.validate_results(p) == []


class TestInvariantRowConsistency:
    """V2 -- fail iff n_fail > 0; pass iff n_fail == 0 and n_pass > 0; not_applicable iff
    both are 0. ``failing_seeds`` is not tied to ``n_fail``."""

    def test_v2_pass_with_n_fail_rejected(self):
        p = _passing().payload()
        row = _row(p, "I3")
        assert row["status"] == "pass"
        row["n_fail"] = 1
        assert cs.validate_results(p) != []

    def test_v2_fail_with_zero_n_fail_rejected(self):
        p = _with_failure("invariant_violation").payload()
        assert cs.validate_results(p) == []
        row = _row(p, "I3")
        assert row["status"] == "fail"
        row["n_fail"] = 0
        row["n_pass"] = 1
        assert p["failure_kinds"] == ["invariant_violation"]  # kinds still consistent with a fail row
        assert cs.validate_results(p) != []

    def test_v2_not_applicable_with_n_pass_rejected(self):
        p = _passing().payload()
        row = _row(p, "I3")
        row["status"] = "not_applicable"
        row["n_pass"] = 1
        row["n_fail"] = 0
        assert cs.validate_results(p) != []

    def test_v2_pass_with_no_counts_rejected(self):
        p = _passing().payload()
        row = _row(p, "I3")
        row["n_pass"] = 0
        row["n_fail"] = 0
        assert row["status"] == "pass"
        assert cs.validate_results(p) != []

    def test_v2_unseeded_fail_row_accepted(self):
        cert = _cert()
        cert.check("trade price equals resting price", True, seed=0, expected=100, observed=100,
                   unit="price_int")
        cert.set_eager_jit(0, True)
        cert.add_battery(_report(I3="fail"), seed=None)
        p = cert.payload()
        row = _row(p, "I3")
        assert row["status"] == "fail" and row["n_fail"] == 1 and row["failing_seeds"] == []
        assert cs.validate_results(p) == []
