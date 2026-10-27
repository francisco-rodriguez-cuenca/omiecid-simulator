"""OMIE-CERT-HARNESS-1 -- Step-5 fresh-context audit.

Derived independently from the story card (agile-plan.md, H1-H16),
scenario-ladder-plan.md (§Notebook architecture, §Agent independence rules)
and orchestrate.md (§Certification cycle). The original locked test files were
not read.

Every test works on scratch copies in ``tmp_path``; the real
``certification-log.md``, ``certificate.md`` and ``results/`` are hashed before
and after the module and must be byte-identical.

Fast tests build their inputs in-test. The ``slow`` tests start a Jupyter
kernel (2-30 s each) and are marked as such.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import shutil
from itertools import product
from pathlib import Path

import nbformat
import numpy as np
import pytest

from notebooks.certification import cert_support as cs
from notebooks.certification.cert_support import (
    ELISION_SPAN,
    HUMAN_VERDICTS,
    LADDER,
    Citation,
    GateSelectionError,
    K5Outcome,
    LogParseError,
    Route,
    SpecQuote,
    check_quote,
    check_quotes,
    close_scenario,
    execute_notebook,
    gated_notebooks,
    next_scenario,
    notebook_structure_errors,
    parse_log,
    quote_digest,
    render_quotes,
    result_word,
    results_digest,
    route_k5,
    validate_results,
)

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs" / "omiecid"
REAL_LOG = DOCS / "certification-log.md"
REAL_CERT = DOCS / "certificate.md"
NB_DIR = REPO / "notebooks" / "certification"
REAL_RESULTS = NB_DIR / "results"
TEMPLATE = NB_DIR / "_template.ipynb"


def _blake(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


def _tree_hash(paths):
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
        for f in files:
            h.update(str(f).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module", autouse=True)
def _real_artifacts_untouched():
    before = _tree_hash([REAL_LOG, REAL_CERT, REAL_RESULTS, NB_DIR / "fixtures", TEMPLATE])
    yield
    after = _tree_hash([REAL_LOG, REAL_CERT, REAL_RESULTS, NB_DIR / "fixtures", TEMPLATE])
    assert before == after, "the real log / certificate / results / fixtures / template changed"


# ---------------------------------------------------------------------------
# scratch documents for the quote checker
# ---------------------------------------------------------------------------
SPEC = """# Scratch spec

**Status:** v9.1

### Version history

| v9.1 | a version history line that must never be quotable |

## 6. Matching

### 6.1 Priority

Orders are ranked by price first and then by time of entry into the book.

### 6.2 Execution

Opening text of section six point two, before the list.

1. The first item of the list says hello world.
2. The second item of the list is about partial fills.
3. Trade executes at the **resting order's limit price** (maker pricing).

```
## 6.9 Fenced heading that is not a heading
fenced text line that sits inside a code block
```

## Unlabelled tail

Text after an unlabelled level-two heading pops section six.

## 7. Long section

{long}

### R-1 — a ruling

The ruling text of R-1 says the citation is wrong but the rule is right.
"""

FILLER = " ".join(f"word{i:03d}" for i in range(120))   # ~960 characters of unique words
SPEC = SPEC.replace("{long}", "ALPHA-START marker sentence. " + FILLER + " OMEGA-END marker sentence.")

OQ = """# Scratch open questions

**Status:** a long status line

## OQ-25 — scratch question

Preamble of the question, which no block lead introduces.

**Question.** What happens to the agent's order in the queue?

**CORRECTION (2026-10-01).** The agent queues at the back was a false premise here.

**CORRECTION (2026-10-02).** A second correction block that shares the prefix.

#### Resolution heading block

Heading-led block text that runs to the next heading of the same level.

## OQ-26 — another question

**Question.** Unrelated text of question twenty six.
"""


@pytest.fixture()
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "market-spec.md").write_text(SPEC, encoding="utf-8")
    (d / "open-questions.md").write_text(OQ, encoding="utf-8")
    (d / "scenario-ladder-review.md").write_text(SPEC.replace("v9.1", "v3.3"), encoding="utf-8")
    return d


def _reason(docs, section, text, source="market-spec.md", block=None):
    return check_quote(SpecQuote(source, section, text, block=block), docs_dir=docs).reason


class TestQuoteNormalisationIsStrict:
    """H3: only the listed normalisations; no Unicode folding of any kind."""

    def test_plain_passage_passes(self, docs):
        assert _reason(docs, "§6.2 item 3", "Trade executes at the resting order's limit price") == "ok"

    def test_emphasis_in_source_ignored(self, docs):
        assert _reason(docs, "§6.2", "at the resting order's limit price (maker pricing).") == "ok"

    def test_emphasis_in_quote_ignored(self, docs):
        assert _reason(docs, "§6.2", "**Opening** text of `section` six *point* two") == "ok"

    def test_case_is_significant(self, docs):
        assert _reason(docs, "§6.2", "TRADE executes at the resting order's") == "not_found"

    def test_typographic_apostrophe_not_folded(self, docs):
        assert _reason(docs, "§6.2", "Trade executes at the resting order’s") == "not_found"

    def test_internal_nbsp_significant(self, docs):
        assert _reason(docs, "§6.2", "Trade executes at the resting order's") == "not_found"

    def test_trailing_nbsp_significant(self, docs):
        # H3: whitespace collapse is ASCII-only ([ \t\r\n]+) and U+00A0 stays significant;
        # a trailing U+00A0 is not in the source (which has an ASCII space there).
        assert _reason(docs, "§6.2", "Trade executes at the ") != "ok"

    def test_leading_nbsp_significant(self, docs):
        assert _reason(docs, "§6.2", " Trade executes at the") != "ok"

    def test_trailing_em_space_significant(self, docs):
        assert _reason(docs, "§6.2", "Trade executes at the ") != "ok"

    def test_nbsp_at_elision_boundary_significant(self, docs):
        assert _reason(docs, "§6.2", "Trade executes at the […] limit price (maker pricing).") != "ok"

    def test_ascii_dots_are_not_an_elision(self, docs):
        assert _reason(docs, "§6.2", "Trade executes [...] (maker pricing).") == "not_found"

    def test_bare_ellipsis_is_not_an_elision(self, docs):
        assert _reason(docs, "§6.2", "Trade executes … (maker pricing).") == "not_found"

    def test_over_quote_beyond_passage_fails(self, docs):
        assert _reason(docs, "§6.2", "limit price (maker pricing). And then more words.") != "ok"


class TestQuoteEmptyAndFragments:
    def test_empty(self, docs):
        assert _reason(docs, "§6.2", "") == "empty"

    def test_whitespace_only(self, docs):
        assert _reason(docs, "§6.2", " \n\t ") == "empty"

    def test_emphasis_only(self, docs):
        assert _reason(docs, "§6.2", "** ** `` *") == "empty"

    def test_only_elision(self, docs):
        assert _reason(docs, "§6.2", "[…]") == "degenerate_fragment"

    def test_leading_elision_empty_fragment(self, docs):
        assert _reason(docs, "§6.2", "[…] Trade executes at the resting") == "degenerate_fragment"

    def test_fragment_of_seven_chars_degenerate(self, docs):
        assert _reason(docs, "§6.2", "Trade executes at the […]maker p") == "degenerate_fragment"

    def test_fragment_of_eight_chars_accepted(self, docs):
        assert _reason(docs, "§6.2", "Trade executes at the […]maker pr") == "ok"

    def test_fragments_in_order(self, docs):
        assert _reason(docs, "§6.2", "Trade executes […] (maker pricing).") == "ok"

    def test_fragments_reordered(self, docs):
        assert _reason(docs, "§6.2", "(maker pricing). […] Trade executes") == "out_of_order"


class TestElisionSpanBoundary:
    """H3: span from the start of the first fragment to the end of the last <= 400."""

    def _frags(self, span):
        hay = cs.normalise(FILLER)
        first = "word000 word001"
        start = hay.index(first)
        # the last fragment is the 15 characters ending exactly at start + span
        end = start + span
        last = hay[end - 15:end]
        assert hay.count(last) == 1
        return first, last

    def test_span_exactly_limit_passes(self, docs):
        a, b = self._frags(ELISION_SPAN)
        assert _reason(docs, "§7", f"{a} […] {b}") == "ok"

    def test_span_one_over_limit_fails(self, docs):
        a, b = self._frags(ELISION_SPAN + 1)
        assert _reason(docs, "§7", f"{a} […] {b}") == "span_exceeded"

    def test_long_section_markers_far_apart(self, docs):
        assert _reason(docs, "§7", "ALPHA-START marker […] OMEGA-END marker") == "span_exceeded"


class TestLabels:
    def test_innermost_label_required(self, docs):
        # §6.2 text cited as its parent §6
        assert _reason(docs, "§6", "Opening text of section six point two") == "wrong_section"

    def test_sibling_label_rejected(self, docs):
        assert _reason(docs, "§6.1", "Opening text of section six point two") == "wrong_section"

    def test_quote_cannot_span_a_heading(self, docs):
        assert _reason(docs, "§6.1", "time of entry into the book. Opening text") != "ok"

    def test_unlabelled_heading_pops_section(self, docs):
        assert _reason(docs, "§6", "Text after an unlabelled level-two heading") != "ok"

    def test_version_history_unquotable(self, docs):
        for label in ("§6", "§6.2", "§7"):
            assert _reason(docs, label, "a version history line that must never") != "ok"

    def test_fenced_heading_is_not_a_label(self, docs):
        assert _reason(docs, "§6.9", "fenced text line that sits inside") == "section_not_found"

    def test_fenced_text_belongs_to_enclosing_section(self, docs):
        assert _reason(docs, "§6.2", "fenced text line that sits inside") == "ok"

    def test_id_label(self, docs):
        assert _reason(docs, "R-1", "The ruling text of R-1 says") == "ok"

    def test_id_label_text_not_in_parent(self, docs):
        assert _reason(docs, "§7", "The ruling text of R-1 says") == "wrong_section"

    @pytest.mark.parametrize("label", ["6.2", "§ 6.2", "§6.2 ", "§6.2 item 0", "§6.2 item", "§6.2 item x",
                                       "r-1", "R-1 ", "§6.2, bullet 2", "", "§6.2item 3"])
    def test_bad_labels(self, docs, label):
        assert _reason(docs, label, "Opening text of section six point two") == "bad_label"

    def test_unknown_section_label(self, docs):
        assert _reason(docs, "§9.9", "Opening text of section six point two") == "section_not_found"

    def test_item_right(self, docs):
        assert _reason(docs, "§6.2 item 2", "second item of the list is about partial fills") == "ok"

    def test_item_wrong(self, docs):
        assert _reason(docs, "§6.2 item 1", "second item of the list is about partial fills") == "wrong_section"

    def test_item_region_excludes_section_preamble(self, docs):
        assert _reason(docs, "§6.2 item 1", "Opening text of section six point two") == "wrong_section"

    def test_item_quote_cannot_cross_into_next_item(self, docs):
        assert _reason(docs, "§6.2 item 1", "says hello world. 2. The second item") != "ok"


class TestSourcesAndBlocks:
    def test_unknown_source(self, docs):
        assert _reason(docs, "§6.2", "Opening text of section six", source="market-spec") == "unknown_source"

    def test_source_with_path_prefix_unknown(self, docs):
        assert _reason(docs, "§6.2", "Opening text of section six",
                       source="docs/omiecid/market-spec.md") == "unknown_source"

    def test_market_spec_quote_attributed_to_review_doc_fails_when_absent(self, docs):
        (docs / "scenario-ladder-review.md").write_text("# R\n\n**Status:** v1\n\n## 6.2 X\n\nother.\n")
        assert _reason(docs, "§6.2", "Opening text of section six point two",
                       source="scenario-ladder-review.md") == "not_found"

    def test_oq_text_attributed_to_market_spec_fails(self, docs):
        r = _reason(docs, "OQ-25", "The agent queues at the back was a false premise")
        assert r != "ok"

    def test_block_missing(self, docs):
        assert _reason(docs, "OQ-25", "The agent queues at the back", source="open-questions.md") == "block_missing"

    def test_block_empty_string_missing(self, docs):
        assert _reason(docs, "OQ-25", "The agent queues at the back", source="open-questions.md",
                       block="") == "block_missing"

    def test_block_forbidden(self, docs):
        assert _reason(docs, "§6.2", "Opening text of section six", block="x") == "block_forbidden"

    def test_block_forbidden_even_if_empty(self, docs):
        assert _reason(docs, "§6.2", "Opening text of section six", block="") == "block_forbidden"

    def test_oq_section_label_must_be_oq(self, docs):
        assert _reason(docs, "§6.2", "The agent queues at the back", source="open-questions.md",
                       block="CORRECTION (2026-10-01") == "bad_label"

    def test_oq_block_unique_prefix(self, docs):
        assert _reason(docs, "OQ-25", "The agent queues at the back was a false premise",
                       source="open-questions.md", block="CORRECTION (2026-10-01") == "ok"

    def test_oq_block_ambiguous(self, docs):
        assert _reason(docs, "OQ-25", "The agent queues at the back", source="open-questions.md",
                       block="CORRECTION") == "block_ambiguous"

    def test_oq_quote_in_other_block_fails(self, docs):
        assert _reason(docs, "OQ-25", "A second correction block that shares", source="open-questions.md",
                       block="CORRECTION (2026-10-01") != "ok"

    def test_oq_preamble_outside_any_block_fails(self, docs):
        assert _reason(docs, "OQ-25", "Preamble of the question, which", source="open-questions.md",
                       block="Question.") != "ok"

    def test_oq_block_not_found(self, docs):
        assert _reason(docs, "OQ-25", "The agent queues at the back", source="open-questions.md",
                       block="RESOLUTION") == "block_not_found"

    def test_oq_block_in_other_oq_fails(self, docs):
        assert _reason(docs, "OQ-26", "The agent queues at the back", source="open-questions.md",
                       block="CORRECTION (2026-10-01") == "block_not_found"

    def test_oq_heading_led_block(self, docs):
        assert _reason(docs, "OQ-25", "Heading-led block text that runs", source="open-questions.md",
                       block="Resolution heading") == "ok"

    def test_oq_version_string(self, docs):
        c = check_quote(SpecQuote("open-questions.md", "OQ-25", "The agent queues at the back",
                                  block="CORRECTION (2026-10-01"), docs_dir=docs)
        assert c.version == "OQ-25 · CORRECTION (2026-10-01"

    def test_unversioned_source(self, docs):
        (docs / "market-spec.md").write_text(SPEC.replace("**Status:** v9.1", "Status: v9.1"))
        assert _reason(docs, "§6.2", "Opening text of section six point two") == "unversioned_source"

    def test_version_from_status_line(self, docs):
        c = check_quote(SpecQuote("market-spec.md", "§6.2", "Opening text of section six"), docs_dir=docs)
        assert c.version == "v9.1"
        c = check_quote(SpecQuote("scenario-ladder-review.md", "§6.2", "Opening text of section six"),
                        docs_dir=docs)
        assert c.version == "v3.3"

    def test_missing_source_file_not_ok(self, tmp_path):
        c = check_quote(SpecQuote("market-spec.md", "§6.2", "Opening text of section six"), docs_dir=tmp_path)
        assert c.ok is False

    def test_never_raises_on_mismatching_strings(self, docs):
        for q in (SpecQuote("", "", ""), SpecQuote("market-spec.md", "§", "x" * 20),
                  SpecQuote("open-questions.md", "OQ-99", "x" * 20, block="**"),
                  SpecQuote("market-spec.md", "§6.2 item 99", "x" * 20),
                  SpecQuote("market-spec.md", "§6.2", "[…][…][…]"), SpecQuote("market-spec.md", "§6.2", "\x00" * 9)):
            c = check_quote(q, docs_dir=docs)
            assert c.ok is False and c.reason in cs.REASONS

    def test_spec_edit_after_check_is_detected(self, docs):
        q = SpecQuote("market-spec.md", "§6.2", "Opening text of section six point two")
        assert check_quote(q, docs_dir=docs).ok
        p = docs / "market-spec.md"
        p.write_text(p.read_text().replace("Opening text of section six point two",
                                           "Opening words of section six point two"))
        assert check_quote(q, docs_dir=docs).ok is False


class TestRealSourceAttribution:
    """The template's quote, against the real documents."""

    TEMPLATE_QUOTE = "Trade executes at the **resting order's limit price** (maker pricing)."

    def test_template_quote_passes_on_real_spec(self):
        c = check_quote(SpecQuote("market-spec.md", "§6.2 item 3", self.TEMPLATE_QUOTE))
        assert c.ok, c.reason
        status = re.search(r"^\*\*Status:\*\*\s*(v\d+(?:\.\d+)*)", (DOCS / "market-spec.md").read_text(), re.M)
        assert c.version == status.group(1)

    def test_template_quote_wrong_item_fails(self):
        assert check_quote(SpecQuote("market-spec.md", "§6.2 item 2", self.TEMPLATE_QUOTE)).ok is False

    def test_template_quote_wrong_document_fails(self):
        for src in ("scenario-ladder-review.md",):
            assert check_quote(SpecQuote(src, "§6.2 item 3", self.TEMPLATE_QUOTE)).ok is False

    def test_template_quote_parent_label_fails(self):
        assert check_quote(SpecQuote("market-spec.md", "§6", self.TEMPLATE_QUOTE)).reason == "wrong_section"


class TestDigestAndRender:
    def test_digest_formula(self):
        q = SpecQuote("open-questions.md", "OQ-25", "a b  c", block="CORRECTION")
        assert quote_digest(q) == _blake("open-questions.md\x00OQ-25\x00CORRECTION\x00a b  c")
        q2 = SpecQuote("market-spec.md", "§6.2", "x")
        assert quote_digest(q2) == _blake("market-spec.md\x00§6.2\x00\x00x")

    def test_digest_over_text_as_written(self):
        a = SpecQuote("market-spec.md", "§6.2", "Trade executes at the")
        b = SpecQuote("market-spec.md", "§6.2", "**Trade** executes  at the")
        assert quote_digest(a) != quote_digest(b)

    def test_digest_independent_of_note(self):
        a = SpecQuote("market-spec.md", "§6.2", "Trade executes", note="one")
        b = SpecQuote("market-spec.md", "§6.2", "Trade executes", note="two")
        assert quote_digest(a) == quote_digest(b)

    def test_check_carries_digest(self, docs):
        q = SpecQuote("market-spec.md", "§6.2", "Opening text of section six point two")
        assert check_quote(q, docs_dir=docs).digest == quote_digest(q)

    def test_render_label_and_drift(self, docs):
        good = SpecQuote("market-spec.md", "§6.2 item 3", "Trade executes at the resting order's", note="N1")
        bad = SpecQuote("market-spec.md", "§6.1", "Opening text of section six point two")
        md = render_quotes(check_quotes([good, bad], docs_dir=docs), [Citation("OMIE Rules §58", "cited text")])
        assert "§6.2 item 3 — market-spec.md v9.1" in md
        assert "> Trade executes at the resting order's" in md
        assert "N1" in md
        assert "**DRIFT: wrong_section**" in md
        assert md.count("DRIFT") == 1
        assert "cited text" in md

    def test_check_quotes_preserves_order(self, docs):
        qs = [SpecQuote("market-spec.md", "§6.1", "Orders are ranked by price"),
              SpecQuote("market-spec.md", "§6.2", ""),
              SpecQuote("market-spec.md", "§6.2", "Opening text of section six")]
        assert [c.reason for c in check_quotes(qs, docs_dir=docs)] == ["ok", "empty", "ok"]


# ---------------------------------------------------------------------------
# H5 -- results validator
# ---------------------------------------------------------------------------
def _pass_payload(cid="C93"):
    return {
        "schema_version": 1, "cert_id": cid, "notebook": f"{cid}_x.ipynb", "scenario": "s",
        "status": "PASS", "failure_kinds": [], "known_result": "k", "oracle_kind": "o",
        "draws": {"requested": 2, "valid": 2, "discarded": 0, "discard_rate": 0.0},
        "seeds": {"first": 0, "last": 1, "count": 2, "failing": []},
        "generator": {"module": "m", "digest_schema_version": 1, "digests": {}},
        "spec": {"quotes": [{"source": "market-spec.md", "section": "§6.2 item 3", "block": None,
                             "version": "v2.13", "digest": "a" * 32, "ok": True, "reason": "ok"}],
                 "quote_digest": _blake("a" * 32)},
        "citations": [],
        "assertions": [{"name": "price", "kind": "oracle", "status": "pass", "seed": 0, "expected": 2,
                        "observed": 2, "unit": "price_int", "detail": "", "rule": "§6.2 item 3"}],
        "invariants": [{"check_id": "I9", "status": "pass", "n_pass": 2, "n_fail": 0, "n_not_applicable": 0,
                        "failing_seeds": [], "unobserved": []}],
        "capacity": {"envelope": {"resting_per_side": 80, "trades": 80, "hops": 160, "concurrent_icebergs": 800},
                     "peaks": {"resting_per_side": 1, "trades": 1, "hops": None, "concurrent_icebergs": None}},
        "eager_jit_identical": True, "wall_clock_s": 0.1, "runtime_forecast_s": None,
        "failure_explanation": None, "environment": {"python": "3.10", "jax": "0.4", "backend": "cpu"},
    }


_EXPL = {"what_failed": "w", "smallest_case": None, "rule": "r", "hypothesis": None, "downstream": "d",
         "paragraph": "C93 NOT CERTIFIED: one paragraph."}


def _fail(p, kinds):
    p = copy.deepcopy(p)
    p["status"] = "NOT CERTIFIED"
    p["failure_kinds"] = kinds
    p["failure_explanation"] = dict(_EXPL)
    return p


def _fail_payload(cid="C93"):
    p = _pass_payload(cid)
    p["assertions"][0].update(status="fail", observed=1)
    p["seeds"]["failing"] = [0]
    return _fail(p, ["oracle_mismatch"])


class TestResultsValidator:
    def test_pass_payload_valid(self):
        assert validate_results(_pass_payload()) == []

    def test_fail_payload_valid(self):
        assert validate_results(_fail_payload()) == []

    def test_never_raises_on_garbage(self):
        for x in (None, [], "PASS", 3, {"status": "PASS"}, {"failure_kinds": None}):
            assert validate_results(x) != []

    def test_extra_top_level_key_rejected(self):
        p = _pass_payload()
        p["extra"] = 1
        assert validate_results(p) != []

    @pytest.mark.parametrize("key", ["spec", "seeds", "generator", "environment", "draws", "citations",
                                     "wall_clock_s", "runtime_forecast_s", "failure_explanation"])
    def test_required_key(self, key):
        p = _pass_payload()
        del p[key]
        assert validate_results(p) != []

    @pytest.mark.parametrize("field", ["source", "section", "block", "version", "digest", "ok", "reason"])
    def test_rule6_quote_fields_required(self, field):
        p = _pass_payload()
        del p["spec"]["quotes"][0][field]
        assert validate_results(p) != []

    def test_rule6_quote_digest_required(self):
        p = _pass_payload()
        del p["spec"]["quote_digest"]
        assert validate_results(p) != []

    def test_rule6_digest_must_be_hex128(self):
        p = _pass_payload()
        p["spec"]["quotes"][0]["digest"] = "XYZ"
        assert validate_results(p) != []

    def test_environment_backend_required(self):
        p = _pass_payload()
        del p["environment"]["backend"]
        assert validate_results(p) != []

    def test_citation_checked_must_be_false(self):
        p = _pass_payload()
        p["citations"] = [{"source": "OMIE Rules", "text": "t", "checked": True}]
        assert validate_results(p) != []

    def test_pass_with_drifted_quote_rejected(self):
        p = _pass_payload()
        p["spec"]["quotes"][0].update(ok=False, reason="not_found")
        assert validate_results(p) != []

    def test_quote_drift_derived(self):
        p = _pass_payload()
        p["spec"]["quotes"][0].update(ok=False, reason="not_found")
        assert validate_results(_fail(p, ["quote_drift"])) == []

    def test_pass_with_failing_eager_jit_rejected(self):
        p = _pass_payload()
        p["assertions"].append({"name": "eager", "kind": "eager_jit", "status": "fail", "seed": 0,
                                "expected": True, "observed": False, "unit": "bool", "detail": "", "rule": None})
        assert validate_results(p) != []
        assert validate_results(_fail(p, ["eager_jit_disagreement"])) == []
        assert validate_results(_fail(p, ["oracle_mismatch"])) != []

    def test_vacuous_run_is_not_invariant_violation(self):
        p = _pass_payload()
        p["invariants"][0].update(status="fail", n_fail=1, n_pass=1, failing_seeds=[1])
        assert validate_results(_fail(p, ["vacuous_run"])) == []
        assert validate_results(_fail(p, ["invariant_violation"])) != []
        assert validate_results(_fail(p, ["invariant_violation", "vacuous_run"])) != []

    def test_other_invariant_fail_is_violation(self):
        p = _pass_payload()
        p["invariants"].append({"check_id": "I5", "status": "fail", "n_pass": 1, "n_fail": 1,
                                "n_not_applicable": 0, "failing_seeds": [1], "unobserved": []})
        assert validate_results(_fail(p, ["invariant_violation"])) == []

    def test_capacity_flag_iff_short_of_requested(self):
        p = _pass_payload()
        p["draws"].update(valid=1, discarded=1, discard_rate=0.5)
        assert validate_results(p) != []
        assert validate_results(_fail(p, ["capacity_flag"])) == []
        q = _pass_payload()
        assert validate_results(_fail(q, ["capacity_flag"])) != []

    def test_discard_rate_formula(self):
        p = _pass_payload()
        p["draws"].update(valid=2, discarded=1, discard_rate=1 / 3)
        assert validate_results(p) == []
        p["draws"]["discard_rate"] = 0.5   # discarded / requested would be wrong
        assert validate_results(p) != []

    def test_discard_rate_zero_when_nothing_drawn(self):
        p = _pass_payload()
        p["draws"].update(requested=0, valid=0, discarded=0, discard_rate=0)
        assert validate_results(p) == []

    def test_failure_kinds_unsorted_rejected(self):
        p = _fail_payload()
        p["spec"]["quotes"][0].update(ok=False, reason="not_found")
        assert validate_results(_fail(p, ["oracle_mismatch", "quote_drift"])) == []
        assert validate_results(_fail(p, ["quote_drift", "oracle_mismatch"])) != []

    def test_failure_kinds_duplicates_rejected(self):
        assert validate_results(_fail(_fail_payload(), ["oracle_mismatch", "oracle_mismatch"])) != []

    def test_not_certified_without_explanation_rejected(self):
        p = _fail_payload()
        p["failure_explanation"] = None
        assert validate_results(p) != []

    def test_pass_with_explanation_rejected(self):
        p = _pass_payload()
        p["failure_explanation"] = dict(_EXPL)
        assert validate_results(p) != []

    def test_status_word_exact(self):
        for word in ("pass", "CERTIFIED", "NOT_CERTIFIED", "FAIL"):
            p = _pass_payload()
            p["status"] = word
            assert validate_results(p) != []

    def test_cert_id_pattern(self):
        for cid in ("C1", "C001", "c02", "X02"):
            assert validate_results(_pass_payload(cid)) != []


# ---------------------------------------------------------------------------
# H6 -- the accumulator (imports the generator, ~0.5 s once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def battery_types():
    from tests.omiecid.reference.omie_invariants import BatteryReport, CheckResult
    return BatteryReport, CheckResult


def _report(battery_types, **statuses):
    BatteryReport, CheckResult = battery_types
    return BatteryReport(results=tuple(CheckResult(check_id=k, status=v, violations=(), unobserved=(),
                                                   evidence={}) for k, v in statuses.items()))


def _cert(docs, quotes=None, requested=2, cid="C93"):
    q = quotes if quotes is not None else (SpecQuote("market-spec.md", "§6.2", "Opening text of section six"),)
    return cs.Certification(cid, f"{cid}_x.ipynb", "scratch", "k", "exact", q, (Citation("Rules", "c"),),
                            (), requested_draws=requested, docs_dir=docs)


class TestAccumulator:
    def test_all_pass(self, docs, battery_types):
        c = _cert(docs)
        for s in (0, 1):
            assert c.add_battery(_report(battery_types, I1="pass", I9="pass"), seed=s)
            c.check_equal("x", 1, 1, seed=s)
        p = c.payload()
        assert validate_results(p) == []
        assert p["status"] == "PASS" and c.verdict_line() == "PASS"
        assert c.failure_section_markdown() == ""

    def test_failures_all_recorded_never_raise(self, docs, battery_types):
        c = _cert(docs)
        c.add_battery(_report(battery_types, I5="fail", I9="pass"), seed=0)
        c.add_battery(_report(battery_types, I5="pass", I9="fail"), seed=1)
        assert c.check("a", False, seed=0) is False
        assert c.check_equal("b", 1, 2, seed=1) is False
        c.set_eager_jit(1, False, "diff")
        p = c.payload()
        assert validate_results(p) == []
        assert p["failure_kinds"] == ["eager_jit_disagreement", "invariant_violation", "oracle_mismatch",
                                      "vacuous_run"]
        assert len([a for a in p["assertions"] if a["status"] == "fail"]) == 3
        assert p["seeds"]["failing"] == [0, 1]
        assert p["eager_jit_identical"] is False
        assert c.verdict_line() == "NOT CERTIFIED"

    def test_i8_invalid_discards(self, docs, battery_types):
        c = _cert(docs, requested=2)
        assert c.add_battery(_report(battery_types, I5="fail", I8="invalid"), seed=0) is False
        assert c.add_battery(_report(battery_types, I5="pass", I8="pass"), seed=1)
        assert c.add_battery(_report(battery_types, I5="pass", I8="pass"), seed=2)
        p = c.payload()
        assert p["draws"] == {"requested": 2, "valid": 2, "discarded": 1, "discard_rate": pytest.approx(1 / 3)}
        # the discarded draw's I5 fail is not recorded
        assert p["status"] == "PASS", p["failure_kinds"]

    def test_capacity_flag_when_short(self, docs, battery_types):
        c = _cert(docs, requested=3)
        c.add_battery(_report(battery_types, I8="invalid"), seed=0)
        c.add_battery(_report(battery_types, I8="pass"), seed=1)
        p = c.payload()
        assert p["failure_kinds"] == ["capacity_flag"]
        assert validate_results(p) == []

    def test_quote_drift_makes_not_certified(self, docs, battery_types):
        c = _cert(docs, quotes=(SpecQuote("market-spec.md", "§6.1", "Opening text of section six"),))
        c.add_battery(_report(battery_types, I9="pass"), seed=0)
        c.add_battery(_report(battery_types, I9="pass"), seed=1)
        p = c.payload()
        assert p["failure_kinds"] == ["quote_drift"]
        assert p["spec"]["quotes"][0]["reason"] == "wrong_section"

    def test_rule6_fields(self, docs):
        qs = (SpecQuote("market-spec.md", "§6.2", "Opening text of section six"),
              SpecQuote("open-questions.md", "OQ-25", "The agent queues at the back",
                        block="CORRECTION (2026-10-01"))
        p = _cert(docs, quotes=qs).payload()
        rows = p["spec"]["quotes"]
        assert [(r["source"], r["section"], r["block"]) for r in rows] == [
            ("market-spec.md", "§6.2", None), ("open-questions.md", "OQ-25", "CORRECTION (2026-10-01")]
        assert [r["digest"] for r in rows] == [quote_digest(q) for q in qs]
        assert p["spec"]["quote_digest"] == _blake("".join(quote_digest(q) for q in qs))
        assert rows[0]["version"] == "v9.1"
        assert p["citations"] == [{"source": "Rules", "text": "c", "checked": False}]
        assert set(p["environment"]) >= {"python", "jax", "backend"}

    def test_record_capacity_elementwise_max(self, docs):
        c = _cert(docs)
        c.record_capacity({"resting_per_side": 3, "trades": None})
        c.record_capacity({"resting_per_side": 2, "trades": 5, "hops": 1})
        assert c.payload()["capacity"]["peaks"] == {"resting_per_side": 3, "trades": 5, "hops": 1,
                                                     "concurrent_icebergs": None}

    def test_emit_respects_env_dirs_and_writes_fixture(self, docs, battery_types, tmp_path, monkeypatch):
        rd, fd = tmp_path / "r", tmp_path / "f"
        monkeypatch.setenv("OMIE_CERT_RESULTS_DIR", str(rd))
        monkeypatch.setenv("OMIE_CERT_FIXTURE_DIR", str(fd))
        c = _cert(docs, requested=1)
        c.add_battery(_report(battery_types, I9="pass"), seed=7)
        tape = np.zeros((4, 13), dtype=np.int64)
        tape[:2, 4] = 1
        c.record_case(7, tape, {"p": 1})
        c.check_equal("price", 1, 2, seed=7, unit="price_int")
        path = c.emit()
        assert path == rd / "C93.json"
        data = json.loads(path.read_text())
        assert validate_results(data) == []
        assert data["status"] == "NOT CERTIFIED"
        fx = fd / "C93_seed7.npz"
        assert fx.exists()
        with np.load(fx) as z:
            assert z["tape"].dtype == np.int32
            assert json.loads(str(z["params_json"])) == {"p": 1}
        assert data["failure_explanation"]["smallest_case"]["seed"] == 7


# ---------------------------------------------------------------------------
# H9 -- routing
# ---------------------------------------------------------------------------
class TestRouting:
    def test_exactly_three_routes_no_stop(self):
        assert {r.value for r in Route} == {"K6", "K6-F merge-evidence", "K6-F record-only"}
        assert not any(re.search(r"stop|await|wait|halt", r.name + r.value, re.I) for r in Route)

    def test_route_properties(self):
        assert Route.K6.merges and Route.K6.signs_certificate
        assert Route.K6F_MERGE_EVIDENCE.merges and not Route.K6F_MERGE_EVIDENCE.signs_certificate
        assert not Route.K6F_RECORD_ONLY.merges and not Route.K6F_RECORD_ONLY.signs_certificate

    @pytest.mark.parametrize("executed,status,twin,regressed,indep",
                             list(product([True, False], ["PASS", "NOT CERTIFIED", None, "pass"],
                                          [True, False], [True, False], [True, False])))
    def test_exhaustive(self, executed, status, twin, regressed, indep):
        r = route_k5(K5Outcome(executed, status, twin, regressed, indep))
        assert isinstance(r, Route)
        if regressed:
            assert r is Route.K6F_RECORD_ONLY
        elif executed and status == "PASS" and twin and indep:
            assert r is Route.K6
        else:
            assert r is Route.K6F_MERGE_EVIDENCE

    def test_outcome_frozen(self):
        o = K5Outcome(True, "PASS", True, False, True)
        with pytest.raises(Exception):
            o.suite_regressed = True

    def test_result_words(self):
        assert result_word(Route.K6) == "CERTIFIED"
        assert result_word(Route.K6, ("C05", "C02")) == "CERTIFIED — provisional on C05, C02"
        assert result_word(Route.K6F_MERGE_EVIDENCE, ("C02",)) == "NOT CERTIFIED"
        assert result_word(Route.K6F_RECORD_ONLY) == "NOT CERTIFIED"

    def test_agents_never_produce_human_verdict_words(self):
        words = {result_word(r, p) for r in Route for p in ((), ("C01",))}
        for w in words:
            assert not any(h in w for h in HUMAN_VERDICTS)

    def test_ladder(self):
        assert LADDER == tuple(f"C{i:02d}" for i in range(18))


# ---------------------------------------------------------------------------
# H10 -- parsing the log, gating, next scenario
# ---------------------------------------------------------------------------
HEAD = "# Log\n\n## Entry template\n\n```markdown\n### <N> · <YYYY-MM-DD> · `C<nn>` — <name>\n\n" \
       "- **Result:** CERTIFIED\n- **Verdict:** ______ · **rationale:** ______\n```\n\n## Entries\n\n"


def _entry(n, cid, result, verdict="______", date="2026-10-02"):
    return (f"### {n} · {date} · `{cid}` — scen {cid}\n\n- **Card:** x\n- **Result:** {result}\n"
            f"- **Verdict:** {verdict} · **rationale:** {'______' if verdict == '______' else 'why'}\n\n")


def _log(tmp_path, *entries, tail="## Appendix\n\n### 99 · 2026-10-02 · `C09` — not an entry\n"):
    p = tmp_path / "log.md"
    p.write_text(HEAD + "".join(entries) + tail, encoding="utf-8")
    return p


def _nbs(tmp_path, *cids):
    d = tmp_path / "nbs"
    d.mkdir(exist_ok=True)
    for c in cids:
        (d / f"{c}_x.ipynb").write_text("{}")
    return d


class TestLogParsing:
    def test_text_after_entries_section_ignored(self, tmp_path):
        assert [e.n for e in parse_log(_log(tmp_path, _entry(1, "C00", "CERTIFIED", "CERTIFIED")))] == [1]

    @pytest.mark.parametrize("bad", ["CERTIFIED - provisional on C02", "CERTIFIED — provisional on C2",
                                     "CERTIFIED — provisional on C02,", "CERTIFIED — provisional on C02 C05",
                                     "CERTIFIED — provisional", "CERTIFIED (provisional)", "certified",
                                     "NOT  CERTIFIED", "PASS", ""])
    def test_malformed_result_raises(self, tmp_path, bad):
        with pytest.raises(LogParseError):
            parse_log(_log(tmp_path, _entry(1, "C00", bad)))

    def test_provisional_multi_parses(self, tmp_path):
        e = parse_log(_log(tmp_path, _entry(1, "C06", "CERTIFIED — provisional on C02, C05", "CERTIFIED")))
        assert e[0].result == "CERTIFIED — provisional on C02, C05"

    @pytest.mark.parametrize("heading", ["### 1 - 2026-10-02 - `C00` — x", "### 1 · 2026-10-02 · `C00` - x",
                                         "### 1 · 2026-10-2 · `C00` — x", "### 1 · 2026-10-02 · C00 — x"])
    def test_malformed_heading_raises(self, tmp_path, heading):
        p = tmp_path / "log.md"
        p.write_text(HEAD + heading + "\n\n- **Result:** CERTIFIED\n- **Verdict:** CERTIFIED\n")
        with pytest.raises(LogParseError):
            parse_log(p)

    def test_missing_verdict_raises(self, tmp_path):
        p = tmp_path / "log.md"
        p.write_text(HEAD + "### 1 · 2026-10-02 · `C00` — x\n\n- **Result:** CERTIFIED\n")
        with pytest.raises(LogParseError):
            parse_log(p)

    def test_equal_numbers_raise(self, tmp_path):
        with pytest.raises(LogParseError):
            parse_log(_log(tmp_path, _entry(1, "C00", "CERTIFIED", "CERTIFIED"),
                           _entry(1, "C01", "CERTIFIED", "CERTIFIED")))

    def test_blank_verdict_parses_empty(self, tmp_path):
        e = parse_log(_log(tmp_path, _entry(1, "C00", "NOT CERTIFIED")))
        assert e[0].verdict == ""

    def test_fenced_entry_inside_entries_ignored(self, tmp_path):
        fenced = "```markdown\n" + _entry(5, "C03", "CERTIFIED", "CERTIFIED") + "```\n\n"
        e = parse_log(_log(tmp_path, _entry(1, "C00", "NOT CERTIFIED"), fenced))
        assert [x.cert_id for x in e] == ["C00"]


class TestGating:
    def test_not_certified_skipped_certified_gated(self, tmp_path):
        log = _log(tmp_path, _entry(1, "C00", "CERTIFIED", "CERTIFIED"), _entry(2, "C01", "NOT CERTIFIED"))
        d = _nbs(tmp_path, "C00", "C01")
        assert gated_notebooks(log, d) == [d / "C00_x.ipynb"]

    def test_not_certified_with_any_human_verdict_skipped(self, tmp_path):
        for v in HUMAN_VERDICTS:
            log = _log(tmp_path, _entry(1, "C01", "NOT CERTIFIED", v))
            assert gated_notebooks(log, _nbs(tmp_path, "C01")) == []

    def test_provisional_gated(self, tmp_path):
        log = _log(tmp_path, _entry(1, "C01", "NOT CERTIFIED"),
                   _entry(2, "C02", "CERTIFIED — provisional on C01", "CERTIFIED"))
        d = _nbs(tmp_path, "C01", "C02")
        assert gated_notebooks(log, d) == [d / "C02_x.ipynb"]

    def test_provisional_with_repeat_verdict_ungated(self, tmp_path):
        log = _log(tmp_path, _entry(2, "C02", "CERTIFIED — provisional on C01", "REPEAT"))
        assert gated_notebooks(log, _nbs(tmp_path, "C02")) == []

    def test_latest_is_highest_n_not_file_order(self, tmp_path):
        # retroactive correction entry after a pass
        log = _log(tmp_path, _entry(1, "C02", "CERTIFIED", "CERTIFIED"), _entry(2, "C03", "CERTIFIED", "CERTIFIED"),
                   _entry(3, "C02", "NOT CERTIFIED"))
        d = _nbs(tmp_path, "C02", "C03")
        assert gated_notebooks(log, d) == [d / "C03_x.ipynb"]

    def test_recertified_after_failure_gated(self, tmp_path):
        log = _log(tmp_path, _entry(1, "C02", "NOT CERTIFIED", "REPEAT"), _entry(2, "C02", "CERTIFIED", "CERTIFIED"))
        d = _nbs(tmp_path, "C02")
        assert gated_notebooks(log, d) == [d / "C02_x.ipynb"]

    def test_two_notebooks_for_one_id_raise(self, tmp_path):
        log = _log(tmp_path, _entry(1, "C02", "CERTIFIED", "CERTIFIED"))
        d = _nbs(tmp_path, "C02")
        (d / "C02_other.ipynb").write_text("{}")
        with pytest.raises(GateSelectionError):
            gated_notebooks(log, d)

    def test_missing_gated_notebook_raises_but_missing_ungated_ok(self, tmp_path):
        log = _log(tmp_path, _entry(1, "C02", "NOT CERTIFIED"))
        assert gated_notebooks(log, _nbs(tmp_path)) == []
        log = _log(tmp_path, _entry(1, "C02", "CERTIFIED", "CERTIFIED"))
        with pytest.raises(GateSelectionError):
            gated_notebooks(log, _nbs(tmp_path))

    def test_gated_sorted(self, tmp_path):
        log = _log(tmp_path, _entry(1, "C05", "CERTIFIED", "CERTIFIED"), _entry(2, "C01", "CERTIFIED", "CERTIFIED"))
        d = _nbs(tmp_path, "C05", "C01")
        assert gated_notebooks(log, d) == [d / "C01_x.ipynb", d / "C05_x.ipynb"]

    def test_real_log_selects_no_template(self):
        gated = gated_notebooks(REAL_LOG, NB_DIR)
        assert all(re.match(r"^C\d{2}_", p.name) for p in gated)
        assert TEMPLATE not in gated


class TestNextScenario:
    def test_failure_does_not_hold_ladder(self, tmp_path):
        log = _log(tmp_path, _entry(1, "C00", "CERTIFIED", "CERTIFIED"), _entry(2, "C01", "NOT CERTIFIED"))
        assert next_scenario(log) == "C02"

    def test_gap_is_first_unseen(self, tmp_path):
        log = _log(tmp_path, _entry(1, "C00", "CERTIFIED", "CERTIFIED"), _entry(2, "C02", "NOT CERTIFIED"))
        assert next_scenario(log) == "C01"

    def test_exhausted(self, tmp_path):
        log = _log(tmp_path, *[_entry(i + 1, c, "NOT CERTIFIED") for i, c in enumerate(LADDER)])
        assert next_scenario(log) is None

    def test_real_log_starts_at_c00(self, tmp_path):
        # The live log grows through C17, so "starts at C00" is asserted on the real log's
        # preamble/template reset to the empty state, and on the ladder order itself.
        log = tmp_path / "certification-log.md"
        log.write_text(_skeleton_log_text(), encoding="utf-8")
        assert len(parse_log(log)) == 0
        assert next_scenario(log) == "C00"
        assert LADDER[0] == "C00"


# ---------------------------------------------------------------------------
# H11 -- writing the log and the certificate (on scratch copies of the real skeletons)
# ---------------------------------------------------------------------------
TOC_PLACEHOLDER = ("| — | — | *(no entries yet — the ladder runs in a later session, after the three "
                   "`OMIE-CERT-*` prerequisite stories)* | — | — | — | — |")
AWAIT_OPEN = "**Awaiting the human (K7-U):** "
AWAIT_CLOSE = " — every"


def _skeleton_log_text(real_log: Path = None) -> str:
    """The real log's preamble and entry template with entries, TOC rows and the awaiting
    list reset to the empty state -- independent of how many real entries exist."""
    text = Path(real_log or REAL_LOG).read_text(encoding="utf-8")
    head, sep, _ = text.partition("\n## Entries\n")
    assert sep, "the real log must carry a '## Entries' heading"
    text = head + "\n## Entries\n\n*(none yet)*\n"
    i = text.index(AWAIT_OPEN) + len(AWAIT_OPEN)
    j = text.index(AWAIT_CLOSE, i)
    text = text[:i] + "*(none yet)*" + text[j:]
    lines = text.split("\n")
    k = next(n for n, ln in enumerate(lines) if ln.startswith("| # | Date | Scenario"))
    end = k + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    lines[k + 2:end] = [TOC_PLACEHOLDER]
    return "\n".join(lines)


def _skeleton_cert_text(real_cert: Path = None) -> str:
    """The real certificate with every coverage-matrix Status/Date/Entry cell reset to '—' and
    every scenario section removed (the intro paragraph of '## Scenario sections' is kept)."""
    text = Path(real_cert or REAL_CERT).read_text(encoding="utf-8")
    lines = text.split("\n")
    for n, ln in enumerate(lines):
        if re.match(r"^\| C\d+ ", ln):
            cells = ln.split("|")
            assert len(cells) == 7, ln
            cells[3:6] = [" — ", " — ", " — "]
            lines[n] = "|".join(cells)
    text = "\n".join(lines)
    a = text.index("\n## Scenario sections\n")
    first = re.search(r"^### C\d+ — ", text[a:], re.M)
    if first:
        start = a + first.start()
        stop = text.index("\n---\n", start)
        text = text[:start].rstrip("\n") + "\n" + text[stop:]
    return text


@pytest.fixture()
def scratch(tmp_path):
    log, cert = tmp_path / "certification-log.md", tmp_path / "certificate.md"
    log.write_text(_skeleton_log_text(), encoding="utf-8")
    cert.write_text(_skeleton_cert_text(), encoding="utf-8")
    return log, cert


def _close(scratch, route, results, cid, **kw):
    log, cert = scratch
    args = dict(route=route, results=results, log_path=log, certificate_path=cert, card=f"OMIE-CERT-{cid}-1",
                branch=f"omie/cert-{cid.lower()}-x", commit="c0ffee1", scenario=f"Scenario {cid}",
                date="2026-10-02")
    args.update(kw)
    return close_scenario(**args)


def _entries_section(text):
    return text[text.index("\n## Entries"):]


def _entry_blocks(text):
    body = _entries_section(text)
    return re.findall(r"^### \d+ · .*?(?=^### |\Z)", body, re.M | re.S)


def _verdict_lines(text):
    return [ln for ln in _entries_section(text).splitlines() if ln.startswith("- **Verdict:**")]


def _cert_section(text, cid):
    m = re.search(rf"^### {cid} — .*?(?=^### |^---|^## |\Z)", text, re.M | re.S)
    return m.group(0) if m else None


def _matrix_row(text, num):
    return next(ln for ln in text.splitlines() if ln.startswith(f"| C{num} "))


class TestCloseAppendOnly:
    def test_prior_entries_byte_identical(self, scratch):
        log, _ = scratch
        _close(scratch, Route.K6, _pass_payload("C00"), "C00")
        first = _entry_blocks(log.read_text())
        _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C01"), "C01")
        _close(scratch, Route.K6F_RECORD_ONLY, _fail_payload("C02"), "C02", tip_sha="deadbee")
        after = _entry_blocks(log.read_text())
        assert len(after) == 3
        assert after[0].rstrip() == first[0].rstrip()

    def test_entries_section_only_grows(self, scratch):
        log, _ = scratch
        _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C00"), "C00")
        s1 = _entries_section(log.read_text()).rstrip()
        _close(scratch, Route.K6, _pass_payload("C01"), "C01")
        assert _entries_section(log.read_text()).startswith(s1)

    def test_header_and_template_untouched(self, scratch):
        log, _ = scratch
        orig = log.read_text()
        tpl = orig[orig.index("## Entry template"):orig.index("\n## Entries")]
        head = orig[:orig.index("## Table of contents")]
        _close(scratch, Route.K6, _pass_payload("C00"), "C00")
        _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C01"), "C01")
        new = log.read_text()
        assert tpl in new and new.startswith(head)

    def test_human_filled_verdict_preserved(self, scratch):
        log, _ = scratch
        _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C00"), "C00")
        t = log.read_text()
        k = t.index("\n## Entries")
        t = t[:k] + t[k:].replace("- **Verdict:** ______ · **rationale:** ______",
                                  "- **Verdict:** DEFECT-FOUND · **rationale:** engine wrong", 1)
        log.write_text(t)
        _close(scratch, Route.K6, _pass_payload("C01"), "C01")
        e = parse_log(log)
        assert e[0].verdict == "DEFECT-FOUND" and e[1].verdict == "CERTIFIED"
        assert "DEFECT-FOUND · **rationale:** engine wrong" in log.read_text()

    def test_toc_rows_accumulate(self, scratch):
        log, _ = scratch
        _close(scratch, Route.K6, _pass_payload("C00"), "C00")
        _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C01"), "C01")
        t = log.read_text()
        assert "*(no entries yet" not in t
        assert re.search(r"^\| 1 \| 2026-10-02 \| `C00`", t, re.M)
        assert re.search(r"^\| 2 \| 2026-10-02 \| `C01`", t, re.M)


class TestCloseVerdictVocabulary:
    def test_k6_verdict_certified(self, scratch):
        log, _ = scratch
        _close(scratch, Route.K6, _pass_payload("C00"), "C00")
        assert parse_log(log)[-1].verdict == "CERTIFIED"
        assert _verdict_lines(log.read_text()) == ["- **Verdict:** CERTIFIED · **rationale:** K5 pass"]

    def test_provisional_verdict_certified_result_provisional(self, scratch):
        log, cert = scratch
        _close(scratch, Route.K6, _pass_payload("C03"), "C03", provisional_on=("C01", "C02"))
        e = parse_log(log)[-1]
        assert e.result == "CERTIFIED — provisional on C01, C02" and e.verdict == "CERTIFIED"
        sec = _cert_section(cert.read_text(), "C03")
        assert "provisional on C01, C02" in sec
        assert "results blake2b-128" in sec
        assert "provisional on C01, C02" in _matrix_row(cert.read_text(), 3)

    @pytest.mark.parametrize("route", [Route.K6F_MERGE_EVIDENCE, Route.K6F_RECORD_ONLY])
    def test_failure_verdict_blank(self, scratch, route):
        log, _ = scratch
        _close(scratch, route, _fail_payload("C00"), "C00", tip_sha="deadbee")
        e = parse_log(log)[-1]
        assert e.result == "NOT CERTIFIED" and e.verdict == ""
        line = _verdict_lines(log.read_text())[-1]
        assert not any(w in line for w in HUMAN_VERDICTS + ("CERTIFIED",))

    @pytest.mark.parametrize("route", [Route.K6F_MERGE_EVIDENCE, Route.K6F_RECORD_ONLY])
    def test_failure_with_provisional_still_not_certified(self, scratch, route):
        log, _ = scratch
        _close(scratch, route, _fail_payload("C00"), "C00", provisional_on=("C01",))
        assert parse_log(log)[-1].result == "NOT CERTIFIED"

    def test_no_human_word_written_anywhere_new(self, scratch):
        log, cert = scratch
        before_l, before_c = log.read_text(), cert.read_text()
        _close(scratch, Route.K6, _pass_payload("C00"), "C00", provisional_on=("C05",))
        _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C01"), "C01")
        _close(scratch, Route.K6F_RECORD_ONLY, _fail_payload("C02"), "C02")
        for before, after in ((before_l, log.read_text()), (before_c, cert.read_text())):
            for w in HUMAN_VERDICTS:
                assert after.count(w) == before.count(w), w


class TestCloseCertificate:
    def test_k6_signs_with_results_digest(self, scratch):
        _, cert = scratch
        res = _pass_payload("C02")
        out = _close(scratch, Route.K6, res, "C02")
        assert out.signed and out.merged and out.entry_n == 1
        sec = _cert_section(cert.read_text(), "C02")
        assert re.search(rf"- \*\*Signed:\*\* 2026-10-02 · log entry 1 · results blake2b-128 "
                         rf"{results_digest(res)}", sec)
        assert re.fullmatch(r"[0-9a-f]{32}", results_digest(res))
        row = _matrix_row(cert.read_text(), 2)
        assert "| CERTIFIED | 2026-10-02 | 1 |" in row

    def test_merge_evidence_unsigned(self, scratch):
        _, cert = scratch
        out = _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C05"), "C05")
        assert out.merged and not out.signed
        sec = _cert_section(cert.read_text(), "C05")
        assert "unsigned" in sec and "blake2b" not in sec
        assert "NOT CERTIFIED" in _matrix_row(cert.read_text(), 5)

    def test_record_only_certificate_byte_identical(self, scratch):
        _, cert = scratch
        before = cert.read_bytes()
        out = _close(scratch, Route.K6F_RECORD_ONLY, _fail_payload("C05"), "C05", tip_sha="deadbee")
        assert not out.merged and not out.signed
        assert cert.read_bytes() == before

    def test_record_only_merged_line(self, scratch):
        log, _ = scratch
        _close(scratch, Route.K6F_RECORD_ONLY, _fail_payload("C05"), "C05", tip_sha="deadbee")
        merged = next(ln for ln in _entries_section(log.read_text()).splitlines()
                      if ln.startswith("- **Merged:**"))
        assert "NOT merged" in merged and "omie/cert-c05-x" in merged and "deadbee" in merged

    def test_c1_row_does_not_touch_c10_to_c17(self, scratch):
        _, cert = scratch
        before = cert.read_text()
        _close(scratch, Route.K6, _pass_payload("C01"), "C01")
        after = cert.read_text()
        for num in range(10, 17):  # the release certificate's matrix ends at C16 (C17 is not shipped)
            assert _matrix_row(after, num) == _matrix_row(before, num)
        assert _matrix_row(after, 1) != _matrix_row(before, 1)

    def test_c10_updates_c10_only(self, scratch):
        _, cert = scratch
        before = cert.read_text()
        _close(scratch, Route.K6, _pass_payload("C10"), "C10")
        after = cert.read_text()
        assert _matrix_row(after, 1) == _matrix_row(before, 1)
        assert "CERTIFIED" in _matrix_row(after, 10)

    def test_harness_id_leaves_matrix_untouched(self, scratch):
        _, cert = scratch
        before = cert.read_text()
        _close(scratch, Route.K6, _pass_payload("C90"), "C90")
        after = cert.read_text()
        cov = lambda t: t[t.index("## Coverage matrix"):t.index("## Conformance gaps")]  # noqa: E731
        assert cov(after) == cov(before)
        assert _cert_section(after, "C90") is not None

    def test_sections_land_under_scenario_sections_before_appendix(self, scratch):
        _, cert = scratch
        _close(scratch, Route.K6, _pass_payload("C00"), "C00")
        _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C01"), "C01")
        t = cert.read_text()
        a, b = t.index("## Scenario sections"), t.index("## Appendix A")
        assert a < t.index("### C00 — ") < t.index("### C01 — ") < b
        sec = _cert_section(t, "C00")
        order = [sec.index(f"- **{k}:**") for k in ("Status", "Signed", "Certifies", "Evidence")]
        assert order == sorted(order)

    def test_k6_without_results_does_not_sign(self, scratch):
        # A K6 signature attests a K5 pass recorded in a results JSON; with no results there is
        # nothing to sign. Either refuse, or at least never emit a "results blake2b-128" signature.
        _, cert = scratch
        try:
            _close(scratch, Route.K6, None, "C02")
        except Exception:
            return
        sec = _cert_section(cert.read_text(), "C02")
        assert sec is None or "blake2b" not in sec

    def test_k6_with_not_certified_results_does_not_sign(self, scratch):
        _, cert = scratch
        try:
            _close(scratch, Route.K6, _fail_payload("C02"), "C02")
        except Exception:
            return
        sec = _cert_section(cert.read_text(), "C02")
        assert sec is None or "blake2b" not in sec


class TestCloseCrashAndAwaiting:
    def test_crash_entry(self, scratch):
        log, cert = scratch
        tb = "\x1b[31mTraceback (most recent call last):\x1b[0m\n  File x\n  line 2\nRuntimeError: boom\n"
        out = _close(scratch, Route.K6F_MERGE_EVIDENCE, None, "C04", traceback=tb)
        t = log.read_text()
        why = next(ln for ln in t.splitlines() if ln.startswith("- **Why it failed:**"))
        assert "explanation missing — notebook crashed; traceback: Traceback (most recent call last): " \
               "… RuntimeError: boom" in why
        assert "\x1b" not in t
        assert parse_log(log)[-1].result == "NOT CERTIFIED"
        assert out.next_scenario == "C00"
        assert "unsigned" in _cert_section(cert.read_text(), "C04")

    def test_why_it_failed_is_paragraph(self, scratch):
        log, _ = scratch
        _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C04"), "C04")
        why = next(ln for ln in log.read_text().splitlines() if ln.startswith("- **Why it failed:**"))
        assert why == "- **Why it failed:** " + _EXPL["paragraph"]

    def test_k6_has_no_why_line(self, scratch):
        log, _ = scratch
        _close(scratch, Route.K6, _pass_payload("C04"), "C04")
        assert "- **Why it failed" not in _entries_section(log.read_text())

    def test_awaiting_list(self, scratch):
        log, _ = scratch
        _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C00"), "C00")
        _close(scratch, Route.K6, _pass_payload("C01"), "C01")
        _close(scratch, Route.K6F_RECORD_ONLY, _fail_payload("C02"), "C02")
        t = log.read_text()
        m = re.search(r"\*\*Awaiting the human \(K7-U\):\*\* (.*?) — every", t, re.S)
        assert m.group(1) == "`C00` (entry 1), `C02` (entry 3)"

    def test_field_order_follows_template(self, scratch):
        log, _ = scratch
        _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C00"), "C00")
        t = log.read_text()
        tpl = t[t.index("```markdown"):t.index("```", t.index("```markdown") + 3)]
        want = re.findall(r"^- \*\*([^*]+?)\*\*", tpl, re.M)
        got = re.findall(r"^- \*\*([^*]+?)\*\*", _entry_blocks(t)[0], re.M)
        assert [w.rstrip(":") for w in got] == [w.rstrip(":") for w in want]

    def test_next_scenario_uses_order(self, scratch):
        out = _close(scratch, Route.K6F_MERGE_EVIDENCE, _fail_payload("C90"), "C90", order=("C90", "C91"))
        assert out.next_scenario == "C91"
        out = _close(scratch, Route.K6, _pass_payload("C91"), "C91", order=("C90", "C91"))
        assert out.next_scenario is None


class TestCloseNeverCorruptsLog:
    """An append-only log cannot be repaired by editing; a writer must not leave it unparseable."""

    @pytest.mark.parametrize("kw", [{"provisional_on": ("C2",)}, {"provisional_on": ("REPEAT",)},
                                    {"date": "2026-10-2"}, {"scenario": "two\nlines"}])
    def test_invalid_input_leaves_log_parseable(self, scratch, kw):
        log, cert = scratch
        _close(scratch, Route.K6, _pass_payload("C00"), "C00")
        before_log, before_cert = log.read_bytes(), cert.read_bytes()
        try:
            _close(scratch, Route.K6, _pass_payload("C01"), "C01", **kw)
        except Exception:
            assert log.read_bytes() == before_log, "raised after writing the log"
            assert cert.read_bytes() == before_cert, "raised after writing the certificate"
            return
        parse_log(log)   # must not raise


# ---------------------------------------------------------------------------
# H8 -- notebook structure, template content, append_failure_section
# ---------------------------------------------------------------------------
def _template():
    return nbformat.read(TEMPLATE, as_version=4)


class TestTemplateAndStructure:
    def test_template_well_formed(self):
        assert notebook_structure_errors(_template()) == []

    def test_template_parameters(self):
        cells = [c for c in _template().cells if "parameters" in c.metadata.get("tags", [])]
        assert len(cells) == 1
        vals = {}
        for node in ast.parse(cells[0].source).body:
            if isinstance(node, ast.Assign):
                vals[node.targets[0].id] = ast.literal_eval(node.value)
        assert vals == {"CERT_ID": "C99", "DRAWS": 3, "SEED0": 0, "INJECT_FAILURE": False}

    def test_template_stripped_on_disk(self):
        for c in _template().cells:
            if c.cell_type == "code":
                assert c.outputs == [] and c.execution_count is None

    def test_template_quotes_spec_6_2_item_3(self):
        src = "\n".join(c.source for c in _template().cells)
        assert '"§6.2 item 3"' in src and "market-spec.md" in src
        assert "Trade executes at the **resting order's limit price** (maker pricing)." in src

    def test_last_cell_render(self):
        last = _template().cells[-1]
        assert last.cell_type == "code" and "section-9-render" in last.metadata.get("tags", [])

    def test_missing_heading_detected(self):
        nb = _template()
        for c in nb.cells:
            if c.cell_type == "markdown" and "## 5. Known result" in c.source:
                c.source = c.source.replace("## 5. Known result", "## 5. Expected")
        assert notebook_structure_errors(nb) != []

    def test_swapped_sections_detected(self):
        nb = _template()
        i5 = next(i for i, c in enumerate(nb.cells) if "section-5" in c.metadata.get("tags", []))
        i6 = next(i for i, c in enumerate(nb.cells) if "section-6" in c.metadata.get("tags", []))
        nb.cells[i5], nb.cells[i6] = nb.cells[i6], nb.cells[i5]
        assert notebook_structure_errors(nb) != []

    def test_untagged_cell_detected(self):
        nb = _template()
        nb.cells.insert(3, nbformat.v4.new_code_cell("x = 1"))
        assert notebook_structure_errors(nb) != []

    def test_static_section9_before_render_detected(self):
        nb = _template()
        nb.cells.insert(len(nb.cells) - 1, nbformat.v4.new_markdown_cell(
            "## 9. Why this certification failed", metadata={"tags": ["section-9"]}))
        assert notebook_structure_errors(nb) != []

    def test_section9_subheadings_out_of_order_detected(self):
        nb = _template()
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "\n\n".join(["## 9. Why this certification failed", "### 9.2 The smallest failing case",
                         "### 9.1 What failed", "### 9.3 Which rule it concerns",
                         "### 9.4 Where the fault most plausibly lies", "### 9.5 What it affects downstream"]),
            metadata={"tags": ["section-9"]}))
        assert notebook_structure_errors(nb) != []


class TestAppendFailureSection:
    def test_appends_last_and_keeps_sections_1_to_8(self, tmp_path):
        p = tmp_path / "C91_x.ipynb"
        shutil.copy(TEMPLATE, p)
        before = [(c.cell_type, c.source, c.metadata.get("tags")) for c in nbformat.read(p, as_version=4).cells]
        res = _fail_payload("C91")
        orig = copy.deepcopy(res)
        upd = cs.append_failure_section(p, res, hypothesis="H-one")
        assert res == orig, "the caller's payload was mutated"
        nb = nbformat.read(p, as_version=4)
        assert [(c.cell_type, c.source, c.metadata.get("tags")) for c in nb.cells[:-1]] == before
        last = nb.cells[-1]
        assert last.cell_type == "markdown" and last.metadata.get("tags") == ["section-9"]
        heads = ["## 9. Why this certification failed", "### 9.1 What failed", "### 9.2 The smallest failing case",
                 "### 9.3 Which rule it concerns", "### 9.4 Where the fault most plausibly lies",
                 "### 9.5 What it affects downstream"]
        pos = [last.source.index(h) for h in heads]
        assert pos == sorted(pos)
        after94 = last.source[pos[4]:pos[5]]
        assert "Hypothesis: H-one" in after94
        assert notebook_structure_errors(nb) == []
        assert upd["failure_explanation"]["hypothesis"] == "H-one"
        assert validate_results(upd) == []

    def test_second_call_replaces(self, tmp_path):
        p = tmp_path / "C91_x.ipynb"
        shutil.copy(TEMPLATE, p)
        cs.append_failure_section(p, _fail_payload("C91"), hypothesis="first")
        cs.append_failure_section(p, _fail_payload("C91"), hypothesis="second")
        nb = nbformat.read(p, as_version=4)
        s9 = [c for c in nb.cells if "section-9" in c.metadata.get("tags", [])]
        assert len(s9) == 1 and "second" in s9[0].source and "first" not in s9[0].source
        assert notebook_structure_errors(nb) == []


# ---------------------------------------------------------------------------
# H7 -- parameter injection errors (raised before any kernel starts)
# ---------------------------------------------------------------------------
class TestInjectionRejections:
    @pytest.mark.parametrize("value", [np.int64(3), [1], (1,), {"a": 1}, b"x", float("nan"), float("inf"),
                                       Path("x")])
    def test_non_literal_rejected(self, tmp_path, value):
        p = tmp_path / "C93_x.ipynb"
        shutil.copy(TEMPLATE, p)
        with pytest.raises(ValueError):
            execute_notebook(p, parameters={"DRAWS": value})

    def test_bad_name_rejected(self, tmp_path):
        p = tmp_path / "C93_x.ipynb"
        shutil.copy(TEMPLATE, p)
        with pytest.raises(ValueError):
            execute_notebook(p, parameters={"DRAWS = 1; import os": 1})

    def test_zero_parameter_cells_rejected(self, tmp_path):
        nb = _template()
        for c in nb.cells:
            c.metadata["tags"] = [t for t in c.metadata.get("tags", []) if t != "parameters"]
        p = tmp_path / "C93_x.ipynb"
        nbformat.write(nb, p)
        with pytest.raises(ValueError):
            execute_notebook(p, parameters={"DRAWS": 1})


# ---------------------------------------------------------------------------
# H7 -- the headless gate on small scratch notebooks (kernel start: slow)
# ---------------------------------------------------------------------------
def _scratch_nb(path, *cells):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for src, tags in cells:
        nb.cells.append(nbformat.v4.new_code_cell(src, metadata={"tags": list(tags)}))
    nbformat.write(nb, path)
    return path


def _writer(payload, name="C93.json"):
    return ("import os, pathlib\n"
            f"pathlib.Path(os.environ['OMIE_CERT_RESULTS_DIR'], {name!r}).write_text({json.dumps(payload)!r})\n")


@pytest.mark.slow
class TestGateAdversarial:
    """slow: each test starts one Jupyter kernel (~2-4 s)."""

    def test_cell_tagged_raises_exception_still_fails_gate(self, tmp_path):
        # nbclient tolerates an error in a cell tagged ``raises-exception``; the gate must not.
        p = _scratch_nb(tmp_path / "C93_x.ipynb", (_writer(_pass_payload()), ()),
                        ("raise RuntimeError('a cell raised')", ("raises-exception",)))
        g = execute_notebook(p)
        assert g.passed is False
        assert g.executed is False

    def test_stale_results_json_does_not_pass(self, tmp_path):
        # results_dir already holds a PASS JSON from an earlier run; this run writes nothing.
        rd = tmp_path / "res"
        rd.mkdir()
        (rd / "C93.json").write_text(json.dumps(_pass_payload()))
        p = _scratch_nb(tmp_path / "C93_x.ipynb", ("x = 1", ()))
        g = execute_notebook(p, results_dir=rd)
        assert g.passed is False

    def test_kernel_death_after_emit_fails(self, tmp_path):
        p = _scratch_nb(tmp_path / "C93_x.ipynb", (_writer(_pass_payload()), ()), ("import os; os._exit(0)", ()),
                        ("y = 2", ()))
        g = execute_notebook(p, timeout=60)
        assert g.executed is False and g.passed is False

    def test_system_exit_after_emit_fails(self, tmp_path):
        p = _scratch_nb(tmp_path / "C93_x.ipynb", (_writer(_pass_payload()), ()), ("raise SystemExit(0)", ()),
                        ("y = 2", ()))
        g = execute_notebook(p)
        assert g.passed is False

    def test_two_results_files_fail(self, tmp_path):
        p = _scratch_nb(tmp_path / "C93_x.ipynb", (_writer(_pass_payload()), ()),
                        (_writer(_pass_payload("C94"), "C94.json"), ()))
        g = execute_notebook(p)
        assert g.executed is True and g.passed is False and g.results_errors

    def test_invalid_pass_json_fails(self, tmp_path):
        bad = _pass_payload()
        bad["spec"]["quotes"][0].update(ok=False, reason="not_found")   # PASS despite drift
        p = _scratch_nb(tmp_path / "C93_x.ipynb", (_writer(bad), ()))
        g = execute_notebook(p)
        assert g.executed is True and g.passed is False and g.results_errors

    def test_default_dirs_are_temporary(self, tmp_path):
        p = _scratch_nb(tmp_path / "C93_x.ipynb", (_writer(_pass_payload()), ()),
                        ("import os; print(os.environ['OMIE_CERT_RESULTS_DIR']); "
                         "print(os.environ['OMIE_CERT_FIXTURE_DIR'])", ()))
        g = execute_notebook(p)
        assert g.passed is True
        out = g.notebook.cells[-1].outputs[0]["text"].split()
        for d in out:
            assert Path(d).resolve() not in (REAL_RESULTS.resolve(), (NB_DIR / "fixtures").resolve())


# ---------------------------------------------------------------------------
# the template end to end, one draw (rule-6 fields of a real emitted JSON)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestTemplateEndToEnd:
    """slow: kernel start + one crossing-pair draw with a jit compile (~20 s)."""

    def test_emitted_json_rule6_and_environment(self, tmp_path):
        p = tmp_path / "C92_audit.ipynb"
        shutil.copy(TEMPLATE, p)
        rd, fd = tmp_path / "r", tmp_path / "f"
        g = execute_notebook(p, parameters={"CERT_ID": "C92", "DRAWS": 1}, results_dir=rd, fixture_dir=fd)
        assert g.executed, g.error
        assert g.passed, g.results_errors
        r = g.results
        assert r["cert_id"] == "C92" and r["notebook"] == "C92_audit.ipynb"
        q = SpecQuote("market-spec.md", "§6.2 item 3",
                      "Trade executes at the **resting order's limit price** (maker pricing).")
        assert r["spec"]["quotes"][0]["digest"] == quote_digest(q)
        assert r["spec"]["quote_digest"] == _blake(quote_digest(q))
        assert r["spec"]["quotes"][0]["version"] == check_quote(q).version
        assert r["environment"]["backend"] == "cpu"
        assert r["draws"]["requested"] == 1 and r["draws"]["valid"] == 1
        assert "C02.crossing_pair" in r["generator"]["digests"]
        assert r["eager_jit_identical"] is True
        assert list(fd.iterdir()) == []

    def test_injected_failure_is_oracle_only(self, tmp_path):
        p = tmp_path / "C92_audit.ipynb"
        shutil.copy(TEMPLATE, p)
        g = execute_notebook(p, parameters={"CERT_ID": "C92", "DRAWS": 1, "INJECT_FAILURE": True},
                             results_dir=tmp_path / "r", fixture_dir=tmp_path / "f")
        assert g.executed and not g.passed
        r = g.results
        assert r["failure_kinds"] == ["oracle_mismatch"]
        failing = [a for a in r["assertions"] if a["status"] == "fail"]
        assert [a["unit"] for a in failing] == ["price_int"]
        assert failing[0]["observed"] == failing[0]["expected"] - 1
        sc = r["failure_explanation"]["smallest_case"]
        assert sc and Path(sc["fixture"]).exists()
        assert Path(sc["fixture"]).parent == (tmp_path / "f").resolve()
