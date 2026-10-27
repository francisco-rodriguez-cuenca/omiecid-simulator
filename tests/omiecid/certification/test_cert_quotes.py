"""OMIE-CERT-HARNESS-1 -- the spec-quote checker (decisions H3, H4; AC (3)).

Locked behavioural contract. Every reason code of H3/H4 is seen failing on a
scratch document written into ``tmp_path``; the four Unicode code points of the
test-first contract (U+2212, U+00D7, U+00B7, U+2014) pass byte-copied and fail
on their ASCII look-alike, on scratch documents and on one real
``market-spec.md`` passage each; real ``open-questions.md`` (OQ-25) and
``scenario-ladder-review.md`` (R-1) quotes pass against their named file and
fail when attributed to ``market-spec.md``.

Public interface under test (``notebooks.certification.cert_support``):
``SpecQuote``, ``Citation``, ``QuoteCheck``, ``check_quote``, ``check_quotes``,
``render_quotes``, ``ELISION_SPAN``, ``DOCS_DIR``.

The real-document tests are deliberately coupled to the quoted passages: a
domain-analyst edit of one of them turns the test red, which is the same drift
signal the notebooks carry (sprint-status Step 2, risks).
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path

import pytest

try:  # collection must succeed before the harness exists; the tests then fail
    from notebooks.certification import cert_support as cs
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    cs = None
    _IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_DOCS = REPO_ROOT / "docs" / "omiecid"
ELL = "[…]"  # the one elision marker: U+2026 in square brackets

ALL_REASONS = frozenset({
    "ok", "empty", "degenerate_fragment", "unknown_source", "block_missing",
    "block_forbidden", "bad_label", "unversioned_source", "section_not_found",
    "block_not_found", "block_ambiguous", "not_found", "wrong_section",
    "out_of_order", "span_exceeded",
})


@pytest.fixture(autouse=True)
def _require_harness():
    if cs is None:
        pytest.fail(f"notebooks.certification.cert_support is not importable: {_IMPORT_ERROR!r}")


# ---------------------------------------------------------------------------
# Scratch documents
# ---------------------------------------------------------------------------
_FILLER = "filler " * 70  # 490 normalised characters
SCRATCH_SPEC = f"""# Scratch market spec

**Status:** v9.1 — scratch copy for the quote-checker tests
**Date:** 2026-10-02

### Version history

| Version | Changes |
|---|---|
| v9.1 | The history-only sentence lives here and nowhere else. |

## 4. Order Types

Plain section-four preamble text.

### 4.2 Iceberg orders

- Only the peak size is shown in the order book.
- When a slice is exhausted, a new slice of peak size is activated.

### 4.2.1 Iceberg interaction semantics

**Rule 1 — FOK visibility.** A Fill-or-Kill order considers the total iceberg quantity.

## 6. Matching Rules and Priority

### 6.1 Price-Time Priority

- **Buy side:** ranked highest price first; ties broken by earliest timestamp.
- A match occurs when best bid price ≥ best ask price.

### 6.2 Regular matching

1. Incoming order attempts to match against best counterpart in the book.
2. Match is possible if orders are opposite sides and `sell_price ≤ buy_price`.
3. Trade executes at the **resting order's limit price** (maker pricing).
4. If incoming order is partially filled, the residual enters the book.

Trailing prose of section six-two, after the numbered list.

### 6.7 Elision probes

ALPHA-FRAGMENT opens the probe. The middle clause sits between them. OMEGA-FRAGMENT closes the probe.

REPEAT-FRAGMENT appears first here, far away. {_FILLER}REPEAT-FRAGMENT appears again near. TARGET-FRAGMENT follows closely.

SPAN-START {"x" * 380} SPAN-END

LONG-START {"y" * 381} LONG-END

### 6.8 Unicode probes

- The buy leg books −V as a cash outflow on every trade.
- The trade value is Q × p × d for every quarter-hour.
- The gate closes at 60·⌊k/4⌋ minutes for every round.
- Phase 2 — the clearing window suspends submission.
- The café accent is composed, and 0.25 h uses a plain space.
- The flag `book_full` keeps its underscore.

### 6.9 Fenced probes

Prose before the fence in six-nine.

```text
## 9. Not a heading inside a fence
fenced line content stays in six-nine
```

Prose after the fence still belongs to six-nine.

### 6.10 Blockquote probes

> Data feed encoding: iceberg orders are identified by the
> execution restriction field.
"""

SCRATCH_OQ = """# Scratch open questions

**Status:** scratch

## OQ-07 — Scratch question

**Question.** Is the scratch rule stated anywhere at all?

**Ruling (2026-01-01).** The first ruling text stands as written.

**Ruling (2026-02-02).** The second ruling text supersedes it.

**CORRECTION (2026-03-03, CARD-1).** This block corrects the first ruling premise.
- **(a) Fact.** The sub bullet stays inside the correction block.

### CLOSURE NOTE (2026-04-04)

**Closure paragraph.** Heading-led block text continues here.

More closure prose inside the heading-led block.

## OQ-08 — Next question

**Question.** Another question body text for OQ-08.
"""

SCRATCH_REVIEW = """# Scratch review

**Status:** v4.2 — scratch

## 12. Rulings for the notebooks to quote

Numbered-section prose for twelve only.

### R-9 — scratch ruling

**Ruling (2026-10-02).** The scratch ruling text is quoted from R-9.

#### D9-01 — Low — a nested finding

The nested finding text belongs to D9-01.
"""


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    (tmp_path / "market-spec.md").write_text(SCRATCH_SPEC, encoding="utf-8")
    (tmp_path / "open-questions.md").write_text(SCRATCH_OQ, encoding="utf-8")
    (tmp_path / "scenario-ladder-review.md").write_text(SCRATCH_REVIEW, encoding="utf-8")
    return tmp_path


def _spec(section, text, **kw):
    return cs.SpecQuote("market-spec.md", section, text, **kw)


def _check(q, docs_dir):
    return cs.check_quote(q, docs_dir=docs_dir)


def _assert_reason(q, docs_dir, reason):
    res = _check(q, docs_dir)
    assert res.reason == reason, f"expected {reason!r}, got {res.reason!r} for {q!r}"
    assert res.ok is (reason == "ok")
    return res


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------
class TestShapes:
    def test_spec_quote_is_frozen_and_defaults(self):
        q = _spec("§6.2 item 3", "Trade executes at the resting order's limit price")
        assert q.block is None and q.note == ""
        with pytest.raises(dataclasses.FrozenInstanceError):
            q.text = "other"  # type: ignore[misc]

    def test_unknown_source_constructs_without_raising(self):
        # the checker reports it; construction must not pre-empt the report
        cs.SpecQuote("README.md", "§1", "whatever text here")

    def test_quote_check_fields(self, docs):
        q = _spec("§6.1", "ranked highest price first; ties broken by earliest timestamp")
        res = _check(q, docs)
        assert res.quote == q
        assert res.ok is True and res.reason == "ok"
        assert res.version == "v9.1"
        assert re.fullmatch(r"[0-9a-f]{32}", res.digest)

    def test_elision_span_constant(self):
        assert cs.ELISION_SPAN == 400

    def test_default_docs_dir_is_repo_docs_omie(self):
        assert Path(cs.DOCS_DIR).resolve() == REAL_DOCS.resolve()

    def test_citation_shape(self):
        c = cs.Citation("OMIE Rules §38.2", "Price bounds are -9,999 to +9,999")
        assert c.source == "OMIE Rules §38.2"
        assert c.text == "Price bounds are -9,999 to +9,999"


# ---------------------------------------------------------------------------
# Normalisation: emphasis, backticks, blockquotes, whitespace -- and nothing else
# ---------------------------------------------------------------------------
class TestNormalisation:
    def test_quote_without_emphasis_matches_bold_source(self, docs):
        _assert_reason(_spec("§6.2 item 3",
                             "Trade executes at the resting order's limit price (maker pricing)."),
                       docs, "ok")

    def test_quote_with_emphasis_matches_bold_source(self, docs):
        _assert_reason(_spec("§6.2 item 3",
                             "Trade executes at the **resting order's limit price** (maker pricing)."),
                       docs, "ok")

    def test_bold_lead_in_stripped(self, docs):
        _assert_reason(_spec("§6.1", "Buy side: ranked highest price first"), docs, "ok")

    def test_single_asterisk_emphasis_in_quote_stripped(self, docs):
        _assert_reason(_spec("§6.1", "*Buy side:* ranked highest price first"), docs, "ok")

    def test_backticks_stripped_both_ways(self, docs):
        _assert_reason(_spec("§6.2 item 2", "opposite sides and sell_price ≤ buy_price"),
                       docs, "ok")
        _assert_reason(_spec("§6.2 item 2", "opposite sides and `sell_price ≤ buy_price`"),
                       docs, "ok")

    def test_quote_spanning_a_line_wrap(self, docs):
        _assert_reason(_spec("§6.10", "Data feed encoding: iceberg orders are identified by "
                                           "the execution restriction field."), docs, "ok")

    def test_ascii_whitespace_runs_collapse(self, docs):
        _assert_reason(_spec("§6.1", "ranked   highest\tprice\nfirst"), docs, "ok")

    def test_underscore_kept(self, docs):
        _assert_reason(_spec("§6.8", "The flag book_full keeps its underscore."), docs, "ok")

    @pytest.mark.parametrize("text", [
        "The flag bookfull keeps its underscore.",
        "The flag book full keeps its underscore.",
    ])
    def test_underscore_is_significant(self, docs, text):
        assert not _check(_spec("§6.8", text), docs).ok

    def test_no_case_folding(self, docs):
        assert not _check(_spec("§6.2 item 3",
                                "trade executes at the resting order's limit price"), docs).ok

    def test_nbsp_is_significant(self, docs):
        assert not _check(_spec("§6.8", "0.25 h uses a plain space"), docs).ok

    def test_no_unicode_normalisation(self, docs):
        # NFD "e + combining acute" must not match the NFC source character
        assert not _check(_spec("§6.8", "The café accent is composed"), docs).ok
        _assert_reason(_spec("§6.8", "The café accent is composed"), docs, "ok")


# ---------------------------------------------------------------------------
# Unicode look-alikes (V3-06): pass byte-copied, fail on the ASCII look-alike
# ---------------------------------------------------------------------------
_SCRATCH_UNICODE = [
    ("U+2212", "The buy leg books −V as a cash outflow", "−", "-"),
    ("U+00D7", "The trade value is Q × p × d for every quarter-hour", "×", "x"),
    ("U+00B7", "The gate closes at 60·⌊k/4⌋ minutes", "·", "."),
    ("U+2014", "Phase 2 — the clearing window suspends submission", "—", "-"),
]

_REAL_UNICODE = [
    ("U+2212", "§8.4", "Buy agent PnL per trade: −V (cash outflow)", "−", "-"),
    ("U+00D7", "§11.1", "Trade value = Q × p × d; d = 0.25 h for 15-min contracts",
     "×", "x"),
    ("U+00B7", "§13.4", "to its GCT (60·⌊k/4⌋ − 60 min from D 00:00, §3.1)",
     "·", "."),
    ("U+2014", "§3.3", "Phase 2 — IDA clearing window (GCT to results, ~20 min):",
     "—", "-"),
]


class TestUnicodeLookAlikes:
    @pytest.mark.parametrize("cp,text,char,ascii_", _SCRATCH_UNICODE, ids=[r[0] for r in _SCRATCH_UNICODE])
    def test_scratch_byte_copied_passes(self, docs, cp, text, char, ascii_):
        assert char in text
        _assert_reason(_spec("§6.8", text), docs, "ok")

    @pytest.mark.parametrize("cp,text,char,ascii_", _SCRATCH_UNICODE, ids=[r[0] for r in _SCRATCH_UNICODE])
    def test_scratch_ascii_lookalike_fails(self, docs, cp, text, char, ascii_):
        res = _check(_spec("§6.8", text.replace(char, ascii_)), docs)
        assert not res.ok
        assert res.reason == "not_found"

    @pytest.mark.parametrize("cp,section,text,char,ascii_", _REAL_UNICODE, ids=[r[0] for r in _REAL_UNICODE])
    def test_real_market_spec_byte_copied_passes(self, cp, section, text, char, ascii_):
        res = cs.check_quote(_spec(section, text))
        assert res.ok, res.reason
        assert res.reason == "ok"

    @pytest.mark.parametrize("cp,section,text,char,ascii_", _REAL_UNICODE, ids=[r[0] for r in _REAL_UNICODE])
    def test_real_market_spec_ascii_lookalike_fails(self, cp, section, text, char, ascii_):
        res = cs.check_quote(_spec(section, text.replace(char, ascii_)))
        assert not res.ok
        assert res.reason == "not_found"


# ---------------------------------------------------------------------------
# Elision
# ---------------------------------------------------------------------------
class TestElision:
    def test_in_order_fragments_within_span_pass(self, docs):
        _assert_reason(_spec("§6.7", f"ALPHA-FRAGMENT opens the probe. {ELL} OMEGA-FRAGMENT closes the probe."),
                       docs, "ok")

    def test_fragments_out_of_order(self, docs):
        _assert_reason(_spec("§6.7", f"OMEGA-FRAGMENT closes the probe. {ELL} ALPHA-FRAGMENT opens the probe."),
                       docs, "out_of_order")

    def test_span_exactly_400_passes(self, docs):
        _assert_reason(_spec("§6.7", f"SPAN-START {ELL} SPAN-END"), docs, "ok")

    def test_span_401_exceeded(self, docs):
        _assert_reason(_spec("§6.7", f"LONG-START {ELL} LONG-END"), docs, "span_exceeded")

    def test_far_fragments_span_exceeded(self, docs):
        _assert_reason(_spec("§6.7", f"ALPHA-FRAGMENT opens the probe. {ELL} LONG-END"),
                       docs, "span_exceeded")

    def test_any_occurrence_chain_suffices(self, docs):
        # first REPEAT occurrence is > 400 chars from TARGET; the second is close
        _assert_reason(_spec("§6.7", f"REPEAT-FRAGMENT appears {ELL} TARGET-FRAGMENT follows closely."),
                       docs, "ok")

    def test_ascii_three_dots_is_not_an_elision_marker(self, docs):
        assert not _check(_spec("§6.7", "ALPHA-FRAGMENT opens the probe. ... OMEGA-FRAGMENT closes the probe."),
                          docs).ok

    def test_bare_ellipsis_is_not_an_elision_marker(self, docs):
        assert not _check(_spec("§6.7", "ALPHA-FRAGMENT opens the probe. … OMEGA-FRAGMENT closes the probe."),
                          docs).ok


# ---------------------------------------------------------------------------
# Empty and degenerate quotes
# ---------------------------------------------------------------------------
class TestEmptyAndDegenerate:
    def test_empty_quote_refused(self, docs):
        _assert_reason(_spec("§6.1", ""), docs, "empty")

    def test_whitespace_only_quote_refused(self, docs):
        _assert_reason(_spec("§6.1", "  \n\t "), docs, "empty")

    def test_marker_only_quote_refused(self, docs):
        assert not _check(_spec("§6.1", ELL), docs).ok

    def test_short_whole_quote_degenerate(self, docs):
        _assert_reason(_spec("§6.2 item 3", "maker"), docs, "degenerate_fragment")

    def test_seven_characters_degenerate(self, docs):
        _assert_reason(_spec("§6.1", "Buy sid"), docs, "degenerate_fragment")

    def test_eight_characters_accepted(self, docs):
        _assert_reason(_spec("§6.1", "Buy side"), docs, "ok")

    def test_short_fragment_degenerate(self, docs):
        _assert_reason(_spec("§6.2 item 3", f"Trade executes at the {ELL} price"), docs,
                       "degenerate_fragment")


# ---------------------------------------------------------------------------
# Section labels (H4)
# ---------------------------------------------------------------------------
class TestLabels:
    def test_section_quote_ok(self, docs):
        _assert_reason(_spec("§6.1", "A match occurs when best bid price ≥ best ask price."), docs, "ok")

    def test_6_1_text_cited_as_6_2_wrong_section(self, docs):
        _assert_reason(_spec("§6.2", "A match occurs when best bid price ≥ best ask price."),
                       docs, "wrong_section")

    def test_4_2_1_quote_cited_as_4_2_wrong_section(self, docs):
        _assert_reason(_spec("§4.2", "A Fill-or-Kill order considers the total iceberg quantity."),
                       docs, "wrong_section")

    def test_4_2_1_quote_cited_as_4_wrong_section(self, docs):
        _assert_reason(_spec("§4", "A Fill-or-Kill order considers the total iceberg quantity."),
                       docs, "wrong_section")

    def test_4_2_1_quote_cited_as_4_2_1_ok(self, docs):
        _assert_reason(_spec("§4.2.1", "A Fill-or-Kill order considers the total iceberg quantity."),
                       docs, "ok")

    def test_4_2_text_cited_as_4_wrong_section(self, docs):
        _assert_reason(_spec("§4", "Only the peak size is shown in the order book."), docs, "wrong_section")

    def test_parent_own_text_quotable(self, docs):
        _assert_reason(_spec("§4", "Plain section-four preamble text."), docs, "ok")

    def test_version_history_text_cannot_be_quoted(self, docs):
        _assert_reason(_spec("§4", "The history-only sentence lives here and nowhere else."),
                       docs, "wrong_section")

    def test_unknown_section(self, docs):
        _assert_reason(_spec("§7.7", "A match occurs when best bid price"), docs, "section_not_found")

    def test_text_nowhere_not_found(self, docs):
        _assert_reason(_spec("§6.1", "This sentence occurs in no scratch document."), docs, "not_found")

    def test_heading_inside_fence_is_not_a_heading(self, docs):
        _assert_reason(_spec("§6.9", "Prose after the fence still belongs to six-nine."), docs, "ok")

    def test_fenced_heading_is_no_section(self, docs):
        _assert_reason(_spec("§9", "Prose after the fence still belongs to six-nine."), docs,
                       "section_not_found")

    @pytest.mark.parametrize("label", [
        "6.2", "§6.2 bullet 3", "§6.2 item", "section 6.2", "§ 6.2",
        "§6.2 item three", "R1", "",
    ])
    def test_bad_labels(self, docs, label):
        _assert_reason(_spec(label, "A match occurs when best bid price"), docs, "bad_label")


class TestItemLabels:
    def test_item_right(self, docs):
        _assert_reason(_spec("§6.2 item 3", "Trade executes at the resting order's limit price"), docs, "ok")

    def test_item_wrong(self, docs):
        _assert_reason(_spec("§6.2 item 4", "Trade executes at the resting order's limit price"),
                       docs, "wrong_section")

    def test_section_label_covers_its_items(self, docs):
        _assert_reason(_spec("§6.2", "Trade executes at the resting order's limit price"), docs, "ok")

    def test_missing_item_fails(self, docs):
        assert not _check(_spec("§6.2 item 9", "Trade executes at the resting order's limit price"),
                          docs).ok

    def test_item_on_bulleted_section_fails(self, docs):
        assert not _check(_spec("§6.1 item 1", "ranked highest price first"), docs).ok


class TestIdLabels:
    def _rv(self, section, text):
        return cs.SpecQuote("scenario-ladder-review.md", section, text)

    def test_id_heading_label(self, docs):
        res = _assert_reason(self._rv("R-9", "The scratch ruling text is quoted from R-9."), docs, "ok")
        assert res.version == "v4.2"

    def test_innermost_labelled_heading_wins(self, docs):
        _assert_reason(self._rv("§12", "The scratch ruling text is quoted from R-9."), docs, "wrong_section")

    def test_nested_id_heading(self, docs):
        _assert_reason(self._rv("D9-01", "The nested finding text belongs to D9-01."), docs, "ok")
        _assert_reason(self._rv("R-9", "The nested finding text belongs to D9-01."), docs, "wrong_section")

    def test_numbered_parent_own_text(self, docs):
        _assert_reason(self._rv("§12", "Numbered-section prose for twelve only."), docs, "ok")


# ---------------------------------------------------------------------------
# Sources, blocks and versions
# ---------------------------------------------------------------------------
class TestSources:
    def test_unknown_source(self, docs):
        _assert_reason(cs.SpecQuote("README.md", "§6.1", "ranked highest price first"), docs,
                       "unknown_source")

    def test_block_missing_for_open_questions(self, docs):
        _assert_reason(cs.SpecQuote("open-questions.md", "OQ-07", "The first ruling text stands as written."),
                       docs, "block_missing")

    def test_block_forbidden_for_market_spec(self, docs):
        _assert_reason(_spec("§6.1", "ranked highest price first", block="Buy side"), docs,
                       "block_forbidden")

    def test_block_forbidden_for_review(self, docs):
        _assert_reason(cs.SpecQuote("scenario-ladder-review.md", "R-9", "The scratch ruling text is quoted",
                                    block="Ruling"), docs, "block_forbidden")

    def test_unversioned_market_spec(self, tmp_path):
        (tmp_path / "market-spec.md").write_text(SCRATCH_SPEC.replace("**Status:**", "**State:**"),
                                                 encoding="utf-8")
        _assert_reason(_spec("§6.1", "ranked highest price first"), tmp_path, "unversioned_source")

    def test_unversioned_review(self, tmp_path):
        (tmp_path / "scenario-ladder-review.md").write_text(
            SCRATCH_REVIEW.replace("**Status:** v4.2", "**Status:** draft"), encoding="utf-8")
        _assert_reason(cs.SpecQuote("scenario-ladder-review.md", "R-9", "The scratch ruling text is quoted"),
                       tmp_path, "unversioned_source")

    def test_market_spec_version_from_status_line(self, docs):
        assert _check(_spec("§6.1", "ranked highest price first"), docs).version == "v9.1"


class TestOpenQuestionBlocks:
    def _oq(self, text, block, section="OQ-07"):
        return cs.SpecQuote("open-questions.md", section, text, block=block)

    def test_bold_led_block(self, docs):
        res = _assert_reason(self._oq("This block corrects the first ruling premise.", "CORRECTION (2026-03-03"),
                             docs, "ok")
        assert res.version == "OQ-07 · CORRECTION (2026-03-03"

    def test_bold_led_block_includes_sub_bullets(self, docs):
        _assert_reason(self._oq("The sub bullet stays inside the correction block.", "CORRECTION (2026-03-03"),
                       docs, "ok")

    def test_quote_outside_the_named_block_fails(self, docs):
        assert not _check(self._oq("The second ruling text supersedes it.", "CORRECTION (2026-03-03"), docs).ok

    def test_heading_ends_a_bold_led_block(self, docs):
        assert not _check(self._oq("Heading-led block text continues here.", "CORRECTION (2026-03-03"), docs).ok

    def test_heading_led_block_spans_inner_bold_leads(self, docs):
        _assert_reason(self._oq("More closure prose inside the heading-led block.", "CLOSURE NOTE (2026-04-04"),
                       docs, "ok")

    def test_unique_block_prefix(self, docs):
        _assert_reason(self._oq("The first ruling text stands as written.", "Ruling (2026-01"), docs, "ok")

    def test_block_ambiguous(self, docs):
        _assert_reason(self._oq("The first ruling text stands as written.", "Ruling (2026"), docs,
                       "block_ambiguous")

    def test_block_not_found(self, docs):
        _assert_reason(self._oq("The first ruling text stands as written.", "VERDICT (1999"), docs,
                       "block_not_found")

    def test_block_belongs_to_its_oq(self, docs):
        assert not _check(self._oq("Another question body text for OQ-08.", "Question.", section="OQ-07"),
                          docs).ok
        _assert_reason(self._oq("Another question body text for OQ-08.", "Question.", section="OQ-08"),
                       docs, "ok")

    def test_oq_not_present(self, docs):
        _assert_reason(self._oq("The first ruling text stands as written.", "Ruling (2026-01", section="OQ-99"),
                       docs, "section_not_found")

    @pytest.mark.parametrize("label", ["§5.2", "R-1", "OQ7", "§6.2 item 3"])
    def test_open_questions_accepts_only_oq_labels(self, docs, label):
        _assert_reason(self._oq("The first ruling text stands as written.", "Ruling (2026-01", section=label),
                       docs, "bad_label")


# ---------------------------------------------------------------------------
# Real source documents (V3-07)
# ---------------------------------------------------------------------------
OQ25_TEXT = "This block corrects the Verdict's premise"
OQ25_BLOCK = "CORRECTION (2026-10-01"
R1_TEXT = "The quote is verbatim and correctly located, but it proves something else."


class TestRealSources:
    def test_real_oq25_correction_passes(self):
        res = cs.check_quote(cs.SpecQuote("open-questions.md", "OQ-25", OQ25_TEXT, block=OQ25_BLOCK))
        assert res.ok, res.reason
        assert res.version == f"OQ-25 · {OQ25_BLOCK}"

    def test_real_oq25_fails_when_attributed_to_market_spec(self):
        res = cs.check_quote(_spec("§5.2", OQ25_TEXT))
        assert not res.ok
        assert res.reason == "not_found"

    def test_real_oq25_fails_in_another_oq(self):
        assert not cs.check_quote(cs.SpecQuote("open-questions.md", "OQ-24", OQ25_TEXT, block=OQ25_BLOCK)).ok

    def test_real_review_r1_passes(self):
        res = cs.check_quote(cs.SpecQuote("scenario-ladder-review.md", "R-1", R1_TEXT))
        assert res.ok, res.reason
        status = (REAL_DOCS / "scenario-ladder-review.md").read_text(encoding="utf-8")
        assert re.fullmatch(r"v\d+(\.\d+)*", res.version)
        assert f"**Status:** {res.version}" in status

    def test_real_review_r1_fails_when_attributed_to_market_spec(self):
        res = cs.check_quote(_spec("§3.3", R1_TEXT))
        assert not res.ok
        assert res.reason == "not_found"

    def test_real_review_r1_cited_as_its_numbered_parent_fails(self):
        res = cs.check_quote(cs.SpecQuote("scenario-ladder-review.md", "§12", R1_TEXT))
        assert res.reason == "wrong_section"

    def test_real_market_spec_template_quote(self):
        res = cs.check_quote(_spec("§6.2 item 3",
                                   "Trade executes at the **resting order's limit price** (maker pricing)."))
        assert res.ok, res.reason
        text = (REAL_DOCS / "market-spec.md").read_text(encoding="utf-8")
        assert f"**Status:** {res.version}" in text

    def test_real_sources_carry_no_elision_marker(self):
        for name in ("market-spec.md", "open-questions.md", "scenario-ladder-review.md"):
            assert ELL not in (REAL_DOCS / name).read_text(encoding="utf-8"), name


# ---------------------------------------------------------------------------
# Digest, batch checking, rendering
# ---------------------------------------------------------------------------
def _expected_digest(source, section, block, text):
    blob = "\x00".join([source, section, block or "", text]).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


class TestDigestAndBatch:
    def test_digest_formula(self, docs):
        q = _spec("§6.1", "ranked highest  price first")  # text exactly as written, not normalised
        assert _check(q, docs).digest == _expected_digest("market-spec.md", "§6.1", None, q.text)

    def test_digest_includes_block(self, docs):
        q = cs.SpecQuote("open-questions.md", "OQ-07", "This block corrects the first ruling premise.",
                         block="CORRECTION (2026-03-03")
        assert _check(q, docs).digest == _expected_digest(q.source, q.section, q.block, q.text)

    def test_digest_differs_on_any_field(self, docs):
        a = _check(_spec("§6.1", "ranked highest price first"), docs).digest
        b = _check(_spec("§6.1", "ranked highest price first "), docs).digest
        assert a != b

    def test_check_quotes_reports_in_order_and_never_raises(self, docs):
        quotes = (
            _spec("§6.1", "ranked highest price first"),
            cs.SpecQuote("nowhere.md", "§1", "whatever text here"),
            _spec("§6.1", ""),
            _spec("bullet 3", "ranked highest price first"),
            _spec("§6.2", "ranked highest price first"),
        )
        res = cs.check_quotes(quotes, docs_dir=docs)
        assert isinstance(res, tuple)
        assert [r.quote for r in res] == list(quotes)
        assert [r.reason for r in res] == ["ok", "unknown_source", "empty", "bad_label", "wrong_section"]

    def test_check_quote_never_raises_on_missing_file(self, tmp_path):
        res = cs.check_quote(_spec("§6.1", "ranked highest price first"), docs_dir=tmp_path)
        assert not res.ok


class TestRender:
    def test_render_passing_quote(self, docs):
        q = _spec("§6.2 item 3", "Trade executes at the resting order's limit price (maker pricing).",
                  note="the maker-pricing rule")
        md = cs.render_quotes(cs.check_quotes((q,), docs_dir=docs))
        lines = md.splitlines()
        assert any(ln.startswith("**") and "§6.2 item 3 — market-spec.md v9.1" in ln for ln in lines)
        assert any(ln.startswith(">") and "resting order's limit price" in ln for ln in lines)
        assert "the maker-pricing rule" in md
        assert "DRIFT" not in md

    def test_render_failing_quote_flags_drift(self, docs):
        q = _spec("§6.1", "This sentence occurs in no scratch document.")
        md = cs.render_quotes(cs.check_quotes((q,), docs_dir=docs))
        assert "**DRIFT: not_found**" in md

    def test_render_citations(self, docs):
        c = cs.Citation("OMIE Rules §38.2", "Price bounds: regulatory corpus text")
        md = cs.render_quotes((), citations=(c,))
        assert "Price bounds: regulatory corpus text" in md


def test_this_file_exercises_every_reason_code():
    """Self-check: each of the fifteen H3 reason codes is asserted somewhere above."""
    src = Path(__file__).read_text(encoding="utf-8")
    seen = {r for r in ALL_REASONS if f'"{r}"' in src.split("def test_this_file_exercises_every_reason_code")[0]}
    assert seen == ALL_REASONS
