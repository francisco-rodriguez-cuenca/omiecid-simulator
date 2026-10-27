"""The certification harness: everything the scenario notebooks share.

Story ``OMIE-CERT-HARNESS-1`` (``docs/omiecid/agile-plan.md``, binding decisions
H1-H16). Import path everywhere, the notebook kernel included:
``notebooks.certification.cert_support``.

Contents, in file order:

* **The quote checker** (H3, H4): :class:`SpecQuote`, :class:`Citation`,
  :class:`QuoteCheck`, :func:`check_quote`, :func:`check_quotes`,
  :func:`render_quotes`. A quote names its source document and is checked,
  byte-strictly after a fixed normalisation, against the labelled region of that
  document. The checker never raises on a mismatch; it reports a reason code.
* **The results schema and the accumulator** (H5, H6):
  :data:`RESULTS_SCHEMA_VERSION`, :func:`validate_results`,
  :class:`Certification`. No check raises; every assertion is recorded.
* **Notebook structure** (H8): :func:`notebook_structure_errors`,
  :func:`append_failure_section`.
* **The headless gate** (H7): :func:`execute_notebook` -> :class:`GateResult`,
  driving ``nbclient`` directly on an in-memory copy of the notebook.
* **The K5 routing rule** (H9), defined once: :class:`Route`,
  :class:`K5Outcome`, :func:`route_k5`, :func:`result_word`.
* **The log and the certificate** (H10, H11): :func:`parse_log`,
  :func:`gated_notebooks`, :func:`next_scenario`, :func:`close_scenario`,
  :func:`unsign_scenario`.

Import boundary (H2): at module level only the standard library, ``numpy``,
``jsonschema``, ``nbformat`` and ``nbclient`` are imported. The generator, the
engine and the invariant battery are imported inside the functions that need
them, so the quote, routing and log surfaces never load JAX.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import jsonschema
import nbclient
import nbformat
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
DOCS_DIR = REPO_ROOT / "docs" / "omiecid"
RESULTS_DIR = NOTEBOOK_DIR / "results"
FIXTURE_DIR = NOTEBOOK_DIR / "fixtures"
#: Shrunk regression tapes (plan of record, scenario-ladder-plan.md:1303). Resolve
#: through ``_env_dir(FIXTURE_ENV, SHRUNK_FIXTURE_DIR)`` so ``OMIE_CERT_FIXTURE_DIR``
#: still overrides it (gate runs write to a temp dir). OMIE-CERT-HARNESS-2 (J6).
SHRUNK_FIXTURE_DIR = REPO_ROOT / "tests" / "omiecid" / "fixtures" / "cert"
SCHEMA_PATH = NOTEBOOK_DIR / "results.schema.json"

RESULTS_ENV = "OMIE_CERT_RESULTS_DIR"
FIXTURE_ENV = "OMIE_CERT_FIXTURE_DIR"

#: Per-scenario per-cell gate timeout in seconds (OMIE-CERT-HARNESS-2, J7), read by
#: ``test_certified_notebooks.py::test_certified_notebook_still_passes`` as
#: ``GATE_CELL_TIMEOUT_S.get("Cnn", 1800)``. Scenarios not listed keep the 1,800 s
#: default of ``execute_notebook``. C16 measured 6,106 s at K3 (superlinear ``vmap``
#: per-lane cost), so it gets about 2x that.
GATE_CELL_TIMEOUT_S: Dict[str, int] = {"C16": 12600}


# ===========================================================================
# H3 / H4 -- the quote checker
# ===========================================================================
SOURCES = ("market-spec.md", "open-questions.md", "scenario-ladder-review.md")
_VERSIONED_SOURCES = ("market-spec.md", "scenario-ladder-review.md")

#: The one elision marker: U+2026 in square brackets.
ELISION = "[…]"
#: Maximum normalised span from the start of the first fragment to the end of the last.
ELISION_SPAN = 400
#: Minimum normalised length of a quote fragment.
MIN_FRAGMENT = 8

REASONS = (
    "ok", "empty", "degenerate_fragment", "unknown_source", "block_missing", "block_forbidden",
    "bad_label", "unversioned_source", "section_not_found", "block_not_found", "block_ambiguous",
    "not_found", "wrong_section", "out_of_order", "span_exceeded",
)

_STATUS_RE = re.compile(r"^\*\*Status:\*\*\s*(v\d+(?:\.\d+)*)", re.M)
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(```|~~~)")
_NUM_LABEL_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
_ID_LABEL_RE = re.compile(r"^([A-Z][A-Za-z0-9]*-\d+)\b")
_LABEL_SECTION_RE = re.compile(r"^§(\d+(?:\.\d+)*)$")
_LABEL_ITEM_RE = re.compile(r"^§(\d+(?:\.\d+)*) item ([1-9]\d*)$")
_LABEL_ID_RE = re.compile(r"^[A-Z][A-Za-z0-9]*-\d+$")
_LABEL_OQ_RE = re.compile(r"^OQ-\d+$")
_WS_RE = re.compile(r"[ \t\r\n]+")
_BQ_RE = re.compile(r"^> ?", re.M)


@dataclass(frozen=True)
class SpecQuote:
    """One verbatim passage of a repository source document.

    ``source`` is one of :data:`SOURCES`, resolved under ``docs_dir``.
    ``section`` is a label: ``§<n>(.<n>)*``, ``§<n>(.<n>)* item <N>`` or an ID
    heading token (``R-1``, ``V3-06``); ``open-questions.md`` accepts only
    ``OQ-<nn>``. ``block`` (required for ``open-questions.md``, forbidden
    otherwise) is the prefix of the block lead the quote sits in, e.g.
    ``"CORRECTION (2026-10-01"``. ``text`` is byte-copied; ``[…]`` elides.
    """

    source: str
    section: str
    text: str
    block: Optional[str] = None
    note: str = ""


@dataclass(frozen=True)
class Citation:
    """Regulatory-corpus text: rendered, never machine-checked (``checked: false``)."""

    source: str
    text: str


@dataclass(frozen=True)
class QuoteCheck:
    """The outcome of checking one :class:`SpecQuote` (``reason`` in :data:`REASONS`)."""

    quote: SpecQuote
    ok: bool
    reason: str
    version: Optional[str]
    digest: str


def normalise(text: str) -> str:
    """H3 normalisation: blockquote markers, emphasis and backticks dropped,
    ASCII whitespace runs collapsed, ends stripped. Nothing else."""
    s = _BQ_RE.sub("", text)
    s = s.replace("**", "").replace("*", "").replace("`", "")
    return _WS_RE.sub(" ", s).strip(" ")


def _blake(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def quote_digest(q: SpecQuote) -> str:
    """blake2b-128 over ``source \\x00 section \\x00 block-or-empty \\x00 text``."""
    return _blake("\x00".join([q.source, q.section, q.block or "", q.text]))


@dataclass
class _Heading:
    line: int
    level: int
    text: str
    label: Optional[str]


def _heading_label(text: str) -> Optional[str]:
    m = _NUM_LABEL_RE.match(text + " ")
    if m:
        return "§" + m.group(1)
    m = _ID_LABEL_RE.match(text)
    if m:
        return m.group(1)
    return None


class _Doc:
    """A source document split into headings and per-line innermost labels."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.split("\n")
        self.headings: List[_Heading] = []
        self.heading_at: Dict[int, _Heading] = {}
        self.line_label: List[Optional[str]] = []
        stack: List[_Heading] = []
        in_fence = False
        for i, ln in enumerate(self.lines):
            if _FENCE_RE.match(ln):
                in_fence = not in_fence
                self.line_label.append(self._innermost(stack))
                continue
            m = None if in_fence else _HEADING_RE.match(ln)
            if m:
                level = len(m.group(1))
                htext = m.group(2).strip()
                h = _Heading(i, level, htext, _heading_label(htext))
                while stack and stack[-1].level >= level:
                    stack.pop()
                stack.append(h)
                self.headings.append(h)
                self.heading_at[i] = h
                self.line_label.append(None)   # heading lines are not quotable text
                continue
            self.line_label.append(self._innermost(stack))

    @staticmethod
    def _innermost(stack: Sequence[_Heading]) -> Optional[str]:
        for h in reversed(stack):
            if h.label is not None:
                return h.label
        return None

    def runs_where(self, lo: int, hi: int, pred) -> List[Tuple[int, int]]:
        """Maximal ``[start, end)`` runs of non-heading lines in ``[lo, hi)`` satisfying ``pred``."""
        out: List[Tuple[int, int]] = []
        start = None
        for i in range(lo, hi):
            mine = i not in self.heading_at and pred(i)
            if mine and start is None:
                start = i
            elif not mine and start is not None:
                out.append((start, i))
                start = None
        if start is not None:
            out.append((start, hi))
        return out

    def runs(self, label: Optional[str]) -> List[Tuple[int, int]]:
        """Contiguous ``[start, end)`` line runs whose innermost label is ``label``."""
        return self.runs_where(0, len(self.lines), lambda i: self.line_label[i] == label)

    def all_runs(self) -> List[Tuple[int, int]]:
        """Every maximal run of non-heading lines (any label, or none)."""
        return self.runs_where(0, len(self.lines), lambda i: True)

    def run_text(self, start: int, end: int) -> str:
        return normalise("\n".join(self.lines[start:end]))

    def has_label(self, label: str) -> bool:
        return any(h.label == label for h in self.headings)

    def heading_extent(self, label: str) -> Optional[Tuple[_Heading, int]]:
        """The first heading carrying ``label`` and the line its subtree ends at."""
        for h in self.headings:
            if h.label == label:
                end = len(self.lines)
                for g in self.headings:
                    if g.line > h.line and g.level <= h.level:
                        end = g.line
                        break
                return h, end
        return None


def _occurrences(hay: str, needle: str) -> List[int]:
    out, i = [], hay.find(needle)
    while i >= 0:
        out.append(i)
        i = hay.find(needle, i + 1)
    return out


def _chain_status(hay: str, frags: Sequence[str]) -> str:
    """``ok`` | ``span_exceeded`` | ``out_of_order`` | ``absent`` for one region text."""
    occ = [_occurrences(hay, f) for f in frags]
    if any(not o for o in occ):
        return "absent"
    complete = False
    for start in occ[0]:
        end = start + len(frags[0])
        chained = True
        for k in range(1, len(frags)):
            nxt = next((p for p in occ[k] if p >= end), None)
            if nxt is None:
                chained = False
                break
            end = nxt + len(frags[k])
        if chained:
            complete = True
            if end - start <= ELISION_SPAN:
                return "ok"
    return "span_exceeded" if complete else "out_of_order"


def _match_regions(texts: Sequence[str], frags: Sequence[str]) -> str:
    """Combine :func:`_chain_status` over the region's runs."""
    statuses = [_chain_status(t, frags) for t in texts]
    if "ok" in statuses:
        return "ok"
    if "span_exceeded" in statuses:
        return "span_exceeded"
    return "out_of_order" if all(any(f in t for t in texts) for f in frags) else "absent"


def _item_runs(doc: _Doc, label: str, item: int) -> List[Tuple[int, int]]:
    """The ``item N`` region inside the labelled section's own text."""
    ext = doc.heading_extent(label)
    if ext is None:
        return []
    h, _ = ext
    own_end = next((g.line for g in doc.headings if g.line > h.line), len(doc.lines))
    start = None
    for i in range(h.line + 1, own_end):
        if re.match(rf"^{item}\. ", doc.lines[i]):
            start = i
            break
    if start is None:
        return []
    end = own_end
    for i in range(start + 1, own_end):
        if re.match(r"^\d+\. ", doc.lines[i]):
            end = i
            break
    return [(start, end)]


def _block_region(doc: _Doc, oq: str, block: str):
    """Locate the open-questions block; returns ``(reason, runs)``."""
    ext = doc.heading_extent(oq)
    if ext is None:
        return "section_not_found", []
    h, end = ext
    leads = []   # (line, kind, level)
    in_fence = False
    for i in range(h.line + 1, end):
        ln = doc.lines[i]
        if _FENCE_RE.match(ln):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if i in doc.heading_at and doc.heading_at[i].level >= 3:
            leads.append((i, "heading", doc.heading_at[i].level))
        elif ln.startswith("**"):
            leads.append((i, "bold", 0))
    want = normalise(block)
    hits = []
    for (i, kind, level) in leads:
        lead_text = normalise(doc.heading_at[i].text if kind == "heading" else doc.lines[i])
        if want and lead_text.startswith(want):
            hits.append((i, kind, level))
    if not hits:
        return "block_not_found", []
    if len(hits) > 1:
        return "block_ambiguous", []
    i, kind, level = hits[0]
    if kind == "bold":
        stop = next((j for (j, _, _) in leads if j > i), end)
        return "ok", [(i, stop)]
    stop = end
    for g in doc.headings:
        if i < g.line < end and g.level <= level:
            stop = g.line
            break
    # heading lines inside the block are excluded from the quotable text
    return "ok", doc.runs_where(i + 1, stop, lambda j: True)


@lru_cache(maxsize=32)
def _load_doc(path: str, mtime_ns: int, size: int) -> _Doc:
    return _Doc(Path(path).read_text(encoding="utf-8"))


def _doc_for(path: Path) -> Optional[_Doc]:
    try:
        st = path.stat()
        return _load_doc(str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _valid_label(q: SpecQuote) -> bool:
    if q.source == "open-questions.md":
        return bool(_LABEL_OQ_RE.match(q.section))
    return bool(_LABEL_SECTION_RE.match(q.section) or _LABEL_ITEM_RE.match(q.section)
                or _LABEL_ID_RE.match(q.section))


def check_quote(q: SpecQuote, *, docs_dir: Path = DOCS_DIR) -> QuoteCheck:
    """Check one quote against the document it names. Never raises on a mismatch."""
    digest = quote_digest(q)

    def done(reason: str, version: Optional[str] = None) -> QuoteCheck:
        return QuoteCheck(quote=q, ok=reason == "ok", reason=reason, version=version, digest=digest)

    if q.source not in SOURCES:
        return done("unknown_source")
    if q.source == "open-questions.md" and not q.block:
        return done("block_missing")
    if q.source != "open-questions.md" and q.block is not None:
        return done("block_forbidden")
    if not isinstance(q.section, str) or not _valid_label(q):
        return done("bad_label")
    if not isinstance(q.text, str) or normalise(q.text) == "":
        return done("empty")
    frags = [normalise(f) for f in q.text.split(ELISION)]
    if any(len(f) < MIN_FRAGMENT for f in frags):
        return done("degenerate_fragment")

    doc = _doc_for(Path(docs_dir) / q.source)
    if doc is None:
        return done("not_found")

    if q.source in _VERSIONED_SOURCES:
        m = _STATUS_RE.search(doc.text)
        if not m:
            return done("unversioned_source")
        version: Optional[str] = m.group(1)
    else:
        version = f"{q.section} · {q.block}"

    if q.source == "open-questions.md":
        reason, runs = _block_region(doc, q.section, q.block or "")
        if reason != "ok":
            return done(reason, version)
    else:
        mi = _LABEL_ITEM_RE.match(q.section)
        if mi:
            label = "§" + mi.group(1)
            runs = _item_runs(doc, label, int(mi.group(2)))
            if not doc.has_label(label) or not runs:
                return done("section_not_found", version)
        else:
            if not doc.has_label(q.section):
                return done("section_not_found", version)
            runs = doc.runs(q.section)

    status = _match_regions([doc.run_text(a, b) for a, b in runs], frags)
    if status != "absent":   # ok | span_exceeded | out_of_order
        return done(status, version)
    anywhere = _match_regions([doc.run_text(a, b) for a, b in doc.all_runs()], frags)
    return done("wrong_section" if anywhere == "ok" else "not_found", version)


def check_quotes(quotes: Iterable[SpecQuote], *, docs_dir: Path = DOCS_DIR) -> Tuple[QuoteCheck, ...]:
    """:func:`check_quote` over ``quotes``, in order. Never raises on a mismatch."""
    return tuple(check_quote(q, docs_dir=docs_dir) for q in quotes)


def render_quotes(checks: Sequence[QuoteCheck], citations: Sequence[Citation] = ()) -> str:
    """Section 2's markdown: label line, blockquote, note, and a DRIFT flag on failure."""
    out: List[str] = []
    for c in checks:
        q = c.quote
        version = c.version or "(unversioned)"
        if q.source == "open-questions.md":
            label = f"**{version} — {q.source}**"
        else:
            label = f"**{q.section} — {q.source} {version}**"
        out.append(label)
        out.append("")
        out.extend(_blockquote(q.text))
        if q.note:
            out.append("")
            out.append(f"*{q.note}*")
        if not c.ok:
            out.append("")
            out.append(f"**DRIFT: {c.reason}**")
        out.append("")
    for cit in citations:
        out.append(f"**{cit.source}** *(regulatory corpus; cited, not machine-checked)*")
        out.append("")
        out.extend(_blockquote(cit.text))
        out.append("")
    return "\n".join(out)


def _blockquote(text: str) -> List[str]:
    return [f"> {line.strip()}" for line in text.strip().splitlines() or [""]]


# ===========================================================================
# H5 -- the results schema and its validator
# ===========================================================================
RESULTS_SCHEMA_VERSION = 1
FAILURE_KINDS = ("capacity_flag", "eager_jit_disagreement", "invariant_violation", "oracle_mismatch",
                 "quote_drift", "vacuous_run")
CAPACITY_AXES = ("resting_per_side", "trades", "hops", "concurrent_icebergs")
GENERATOR_MODULE = "omiecid_simulator.jaxomiecid.omie_tape_synth"


@lru_cache(maxsize=1)
def _schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def derive_failure_kinds(payload: Mapping[str, Any]) -> List[str]:
    """The failure kinds implied by a payload's own content (sorted, unique)."""
    kinds = set()
    for a in payload.get("assertions", []):
        if a.get("status") == "fail":
            if a.get("kind") == "oracle":
                kinds.add("oracle_mismatch")
            elif a.get("kind") == "eager_jit":
                kinds.add("eager_jit_disagreement")
    if any(not q.get("ok") for q in payload.get("spec", {}).get("quotes", [])):
        kinds.add("quote_drift")
    draws = payload.get("draws", {})
    if draws.get("valid", 0) < draws.get("requested", 0):
        kinds.add("capacity_flag")
    for row in payload.get("invariants", []):
        if row.get("status") == "fail":
            kinds.add("vacuous_run" if row.get("check_id") == "I9" else "invariant_violation")
    return sorted(kinds)


def validate_results(payload: Any) -> List[str]:
    """JSON-Schema errors plus the H5 semantic checks. Empty list = valid. Never raises."""
    try:
        validator = jsonschema.Draft202012Validator(_schema())
        errors = [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
                  for e in validator.iter_errors(payload)]
    except Exception as exc:  # noqa: BLE001
        return [f"schema validation could not run: {exc!r}"]
    if errors:
        return errors
    try:
        derived = derive_failure_kinds(payload)
        if payload["failure_kinds"] != derived:
            errors.append(f"failure_kinds {payload['failure_kinds']!r} != derived {derived!r}")
        if (payload["status"] == "PASS") != (payload["failure_kinds"] == []):
            errors.append(f"status {payload['status']!r} inconsistent with failure_kinds "
                          f"{payload['failure_kinds']!r}")
        if (payload["failure_explanation"] is None) != (payload["status"] == "PASS"):
            errors.append("failure_explanation must be null exactly when status is PASS")
        d = payload["draws"]
        denom = d["valid"] + d["discarded"]
        want = d["discarded"] / denom if denom else 0
        if not math.isclose(float(d["discard_rate"]), want, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(f"discard_rate {d['discard_rate']!r} != discarded/(valid+discarded) = {want!r}")
        # R3b: the aggregate quote digest. The per-quote digests cannot be recomputed here,
        # because the payload does not carry the quote text.
        spec = payload["spec"]
        want_qd = _blake("".join(q["digest"] for q in spec["quotes"]))
        if spec["quote_digest"] != want_qd:
            errors.append(f"spec.quote_digest {spec['quote_digest']!r} != blake2b-128 of the quote digests "
                          f"{want_qd!r}")
        # R3c: each invariant row's status follows from its counts. failing_seeds is not tied
        # to n_fail: an unseeded battery (add_battery(seed=None)) records a failure with no seed.
        for row in payload["invariants"]:
            n_pass, n_fail = row["n_pass"], row["n_fail"]
            want_st = "fail" if n_fail > 0 else "pass" if n_pass > 0 else "not_applicable"
            if row["status"] != want_st:
                errors.append(f"invariant {row['check_id']}: status {row['status']!r} inconsistent with "
                              f"n_pass={n_pass}, n_fail={n_fail} (expected {want_st!r})")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"semantic check failed: {exc!r}")
    return errors


def _jsonable(x: Any) -> Any:
    """Plain-JSON copy (numpy scalars and arrays converted; NaN refused as a string)."""
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if x is None or isinstance(x, (bool, int, str)):
        return x
    return repr(x)


def results_digest(results: Mapping[str, Any]) -> str:
    """blake2b-128 of the canonical JSON of a results payload (the certificate signature)."""
    return _blake(json.dumps(results, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


# ===========================================================================
# H6 -- the accumulator
# ===========================================================================
_INVARIANT_ORDER = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")


def _env_dir(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else default


class Certification:
    """Accumulates one scenario's evidence and writes its results JSON.

    ``started_at`` (a ``time.time()`` value, default: construction time) is where
    ``wall_clock_s`` is measured from; a notebook stamps it in its first code cell
    so the measured wall clock covers the whole run, not only section 7.

    Nothing here raises on a failed check: every assertion is evaluated and
    recorded, and the verdict is derived from the record at :meth:`payload` time.
    """

    def __init__(self, cert_id: str, notebook: str, scenario: str, known_result: str, oracle_kind: str,
                 quotes: Sequence[SpecQuote], citations: Sequence[Citation] = (),
                 constructions: Sequence[str] = (), *, requested_draws: int,
                 runtime_forecast_s: Optional[float] = None, downstream: str = "",
                 docs_dir: Path = DOCS_DIR, started_at: Optional[float] = None):
        self.cert_id = cert_id
        self.notebook = notebook
        self.scenario = scenario
        self.known_result = known_result
        self.oracle_kind = oracle_kind
        self.quotes = tuple(quotes)
        self.citations = tuple(citations)
        self.constructions = tuple(constructions)
        self.requested_draws = int(requested_draws)
        self.runtime_forecast_s = runtime_forecast_s
        self.downstream = downstream
        self.quote_checks = check_quotes(self.quotes, docs_dir=docs_dir)
        self._assertions: List[Dict[str, Any]] = []
        self._rows: Dict[str, Dict[str, Any]] = {}
        self._valid = 0
        self._discards: List[Tuple[Any, str]] = []
        self._seeds: set = set()
        self._peaks: Dict[str, Optional[int]] = {k: None for k in CAPACITY_AXES}
        self._cases: Dict[int, Tuple[np.ndarray, Dict[str, Any]]] = {}
        # wall clock runs from ``started_at`` (the notebook stamps it in its first code cell)
        self._t0 = time.time() if started_at is None else float(started_at)

    # -- recording ---------------------------------------------------------
    def _note_seed(self, seed) -> None:
        if seed is not None:
            self._seeds.add(int(seed))

    def check(self, name: str, ok: bool, *, kind: str = "oracle", seed: Optional[int] = None,
              expected: Any = None, observed: Any = None, unit: str = "other", detail: str = "",
              rule: Optional[str] = None) -> bool:
        """Record one assertion; returns ``ok``. Never raises on a failed check."""
        ok = bool(ok)
        self._note_seed(seed)
        self._assertions.append({
            "name": str(name), "kind": kind, "status": "pass" if ok else "fail",
            "seed": None if seed is None else int(seed),
            "expected": _jsonable(expected), "observed": _jsonable(observed),
            "unit": unit, "detail": str(detail), "rule": rule,
        })
        return ok

    def check_equal(self, name: str, observed: Any, expected: Any, **kw) -> bool:
        """:meth:`check` with ``ok = observed == expected``."""
        try:
            ok = bool(observed == expected)
        except Exception:  # noqa: BLE001
            ok = False
        if not ok and not kw.get("detail"):
            kw["detail"] = f"observed {observed!r} != expected {expected!r}"
        return self.check(name, ok, expected=expected, observed=observed, **kw)

    def add_battery(self, report, *, seed: Optional[int]) -> bool:
        """Fold a ``BatteryReport``; an I8 ``invalid`` discards the draw. Returns validity."""
        self._note_seed(seed)
        results = list(report.results)
        for r in results:
            self._rows.setdefault(r.check_id, {"n_pass": 0, "n_fail": 0, "n_not_applicable": 0,
                                               "failing_seeds": [], "unobserved": set()})
        if any(r.check_id == "I8" and r.status == "invalid" for r in results):
            self._discards.append((seed, "I8 invalid: generated draw exceeds the capacity envelope"))
            return False
        self._valid += 1
        for r in results:
            row = self._rows[r.check_id]
            if r.status == "fail":
                row["n_fail"] += 1
                if seed is not None and int(seed) not in row["failing_seeds"]:
                    row["failing_seeds"].append(int(seed))
            elif r.status == "not_applicable":
                row["n_not_applicable"] += 1
            else:
                row["n_pass"] += 1
            row["unobserved"].update(str(u) for u in r.unobserved)
        return True

    def discard(self, seed: Optional[int], reason: str) -> None:
        """Record a draw discarded for ``reason`` (it counts towards the discard rate)."""
        self._note_seed(seed)
        self._discards.append((seed, str(reason)))

    def record_capacity(self, peaks: Mapping[str, Optional[int]]) -> None:
        """Element-wise maximum of the observed capacity peaks (``None`` = unobserved)."""
        for k, v in peaks.items():
            if k not in self._peaks or v is None:
                continue
            cur = self._peaks[k]
            self._peaks[k] = int(v) if cur is None else max(cur, int(v))

    def set_eager_jit(self, seed: Optional[int], identical: bool, detail: str = "") -> bool:
        """Record the eager-vs-jitted comparison of one draw as an ``eager_jit`` assertion."""
        return self.check(f"eager == jitted (seed {seed})", identical, kind="eager_jit", seed=seed,
                          expected=True, observed=bool(identical), unit="bool", detail=detail)

    def record_case(self, seed: int, tape, params: Mapping[str, Any]) -> None:
        """Keep a draw's tape and parameters, so a failing seed can be frozen as a fixture."""
        self._cases[int(seed)] = (np.asarray(tape, dtype=np.int32).copy(), dict(params))

    # -- derived -----------------------------------------------------------
    def _discard_rate(self) -> float:
        denom = self._valid + len(self._discards)
        return len(self._discards) / denom if denom else 0

    def _invariant_rows(self) -> List[Dict[str, Any]]:
        order = [c for c in _INVARIANT_ORDER if c in self._rows] + sorted(
            c for c in self._rows if c not in _INVARIANT_ORDER)
        out = []
        for cid in order:
            r = self._rows[cid]
            if r["n_fail"]:
                status = "fail"
            elif r["n_pass"]:
                status = "pass"
            else:
                status = "not_applicable"
            out.append({"check_id": cid, "status": status, "n_pass": r["n_pass"], "n_fail": r["n_fail"],
                        "n_not_applicable": r["n_not_applicable"],
                        "failing_seeds": sorted(r["failing_seeds"]), "unobserved": sorted(r["unobserved"])})
        return out

    def _failing_seeds(self, rows) -> List[int]:
        s = {a["seed"] for a in self._assertions if a["status"] == "fail" and a["seed"] is not None}
        for r in rows:
            s.update(r["failing_seeds"])
        return sorted(s)

    def _fixture_path(self, seed: int) -> Path:
        return _env_dir(FIXTURE_ENV, FIXTURE_DIR) / f"{self.cert_id}_seed{seed}.npz"

    def _smallest_case(self, failing: Sequence[int]) -> Optional[Dict[str, Any]]:
        cands = [s for s in failing if s in self._cases]
        if not cands:
            return None
        def rows(s):
            tape = self._cases[s][0]
            return int(np.sum(tape[:, 4] >= 1)) if tape.ndim == 2 and tape.shape[1] > 4 else int(len(tape))
        best = min(cands, key=lambda s: (rows(s), s))
        return {"seed": best, "fixture": str(self._fixture_path(best)), "rows": rows(best)}

    def _explanation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """The K5-time failure explanation (the hypothesis is completed later, at K6-F)."""
        kinds = payload["failure_kinds"]
        bad_a = [a for a in payload["assertions"] if a["status"] == "fail"]
        bad_i = [r for r in payload["invariants"] if r["status"] == "fail"]
        bad_q = [q for q in payload["spec"]["quotes"] if not q["ok"]]
        parts = []
        if bad_a:
            first = bad_a[0]
            parts.append(f"{len(bad_a)} assertion(s) failed, first '{first['name']}' on seed {first['seed']}: "
                         f"expected {first['expected']!r}, observed {first['observed']!r} ({first['unit']})")
        if bad_i:
            parts.append("invariant check(s) failed: " + ", ".join(
                f"{r['check_id']} on seeds {r['failing_seeds']}" for r in bad_i))
        if bad_q:
            parts.append("quote drift: " + ", ".join(f"{q['section']} of {q['source']} ({q['reason']})"
                                                       for q in bad_q))
        d = payload["draws"]
        if "capacity_flag" in kinds:
            parts.append(f"only {d['valid']} of {d['requested']} requested draws were valid "
                         f"({d['discarded']} discarded)")
        what = _one_line("; ".join(parts) + ".") if parts else "unexplained failure."
        sc = self._smallest_case(payload["seeds"]["failing"])
        rule_list = [a["rule"] for a in bad_a if a.get("rule")]
        rule = rule_list[0] if rule_list else ", ".join(
            f"{q['source']} {q['section']}" for q in payload["spec"]["quotes"]) or "—"
        downstream = self.downstream or "not stated by the card"
        sc_text = f"the smallest failing case is {_case_text(sc)}" if sc else _case_text(sc)
        paragraph = _one_line(f"{payload['cert_id']} NOT CERTIFIED ({', '.join(kinds)}). What failed: {what} "
                              f"Smallest case: {sc_text}. Rule: {rule}. Fault: hypothesis not yet written "
                              f"(completed at K6-F). Downstream: {downstream}.")
        return {"what_failed": what, "smallest_case": sc, "rule": rule, "hypothesis": None,
                "downstream": downstream, "paragraph": paragraph}

    def payload(self) -> Dict[str, Any]:
        """The results payload (a fresh dict on every call)."""
        from omiecid_simulator.jaxomiecid import omie_tape_synth as synth

        rows = self._invariant_rows()
        quotes = [{"source": c.quote.source, "section": c.quote.section, "block": c.quote.block,
                   "version": c.version, "digest": c.digest, "ok": c.ok, "reason": c.reason}
                  for c in self.quote_checks]
        discarded = len(self._discards)
        seeds = sorted(self._seeds)
        failing = self._failing_seeds(rows)
        eager = [a for a in self._assertions if a["kind"] == "eager_jit"]
        env = synth.DEFAULT_ENVELOPE
        p: Dict[str, Any] = {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "cert_id": self.cert_id,
            "notebook": self.notebook,
            "scenario": self.scenario,
            "status": "PASS",
            "failure_kinds": [],
            "known_result": self.known_result,
            "oracle_kind": self.oracle_kind,
            "draws": {"requested": self.requested_draws, "valid": self._valid, "discarded": discarded,
                      "discard_rate": self._discard_rate()},
            "seeds": {"first": seeds[0] if seeds else None, "last": seeds[-1] if seeds else None,
                      "count": len(seeds), "failing": failing},
            "generator": {"module": GENERATOR_MODULE, "digest_schema_version": synth.DIGEST_SCHEMA_VERSION,
                          "digests": {k: synth.construction_digest(k) for k in self.constructions}},
            "spec": {"quotes": quotes,
                     "quote_digest": _blake("".join(q["digest"] for q in quotes))},
            "citations": [{"source": c.source, "text": c.text, "checked": False} for c in self.citations],
            "assertions": copy.deepcopy(self._assertions),
            "invariants": rows,
            "capacity": {"envelope": {"resting_per_side": env.max_resting_per_side, "trades": env.max_trades,
                                      "hops": env.max_hops, "concurrent_icebergs": env.max_concurrent_icebergs},
                         "peaks": dict(self._peaks)},
            "eager_jit_identical": (all(a["status"] == "pass" for a in eager) if eager else None),
            "wall_clock_s": round(time.time() - self._t0, 3),
            "runtime_forecast_s": self.runtime_forecast_s,
            "failure_explanation": None,
            "environment": _environment(),
        }
        p["failure_kinds"] = derive_failure_kinds(p)
        if p["failure_kinds"]:
            p["status"] = "NOT CERTIFIED"
            p["failure_explanation"] = self._explanation(p)
        return p

    # -- rendering and emission --------------------------------------------
    def table(self) -> str:
        """Markdown pass/fail table: assertions, then invariant rows, then draw accounting."""
        lines = ["| check | kind | status | seed | expected | observed | unit | detail |",
                 "|---|---|---|---|---|---|---|---|"]
        for a in self._assertions:
            lines.append(f"| {a['name']} | {a['kind']} | {a['status'].upper()} | {a['seed']} | "
                         f"{a['expected']} | {a['observed']} | {a['unit']} | {a['detail']} |")
        for r in self._invariant_rows():
            lines.append(f"| {r['check_id']} | invariant | {r['status'].upper()} | "
                         f"fail seeds {r['failing_seeds']} | pass {r['n_pass']} | fail {r['n_fail']} | "
                         f"n/a {r['n_not_applicable']} | unobserved {', '.join(r['unobserved']) or '-'} |")
        rate = self._discard_rate()
        lines.append("")
        lines.append(f"Draws: requested {self.requested_draws}, valid {self._valid}, discarded "
                     f"{len(self._discards)} (discard rate {rate:.3f}). Capacity peaks: "
                     + ", ".join(f"{k} {v if v is not None else 'unobserved'}" for k, v in self._peaks.items()))
        return "\n".join(lines)

    def emit(self) -> Path:
        """Validate the payload, write ``<results dir>/<cert_id>.json`` (and, on NOT
        CERTIFIED, the smallest failing case's fixture), and return the JSON path."""
        p = self.payload()
        errors = validate_results(p)
        if errors:
            raise ValueError("results payload failed validation: " + "; ".join(errors))
        sc = (p["failure_explanation"] or {}).get("smallest_case")
        if sc and sc.get("fixture"):
            tape, params = self._cases[int(sc["seed"])]
            fx = Path(sc["fixture"])
            fx.parent.mkdir(parents=True, exist_ok=True)
            np.savez(fx, tape=np.asarray(tape, dtype=np.int32),
                     params_json=np.array(json.dumps(_jsonable(params), sort_keys=True)))
        out_dir = _env_dir(RESULTS_ENV, RESULTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.cert_id}.json"
        path.write_text(json.dumps(p, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        return path

    def verdict_line(self) -> str:
        """Exactly ``PASS`` or ``NOT CERTIFIED``."""
        return "PASS" if not self.payload()["failure_kinds"] else "NOT CERTIFIED"

    def failure_section_markdown(self) -> str:
        """Section 9 as rendered at execution time (empty string on PASS)."""
        p = self.payload()
        if p["status"] == "PASS":
            return ""
        return failure_section_text(p["failure_explanation"])


def _environment() -> Dict[str, Any]:
    try:
        import importlib.metadata as md
        jax_version = md.version("jax")
    except Exception:  # noqa: BLE001
        jax_version = None
    backend = None
    if "jax" in sys.modules:
        try:
            backend = sys.modules["jax"].default_backend()
        except Exception:  # noqa: BLE001
            backend = None
    if backend is None:
        backend = os.environ.get("JAX_PLATFORMS") or None
    return {"python": platform.python_version(), "jax": jax_version, "backend": backend}


# ===========================================================================
# H8 -- notebook structure and section 9
# ===========================================================================
HEADINGS = ("## 1. Purpose", "## 2. The rule, quoted", "## 3. Configuration", "## 4. Dataset",
            "## 5. Known result", "## 6. Run", "## 7. Certification", "## 8. Verdict")
SECTION9 = ("## 9. Why this certification failed", "### 9.1 What failed", "### 9.2 The smallest failing case",
            "### 9.3 Which rule it concerns", "### 9.4 Where the fault most plausibly lies",
            "### 9.5 What it affects downstream")
_SECTION_TAG_RE = re.compile(r"^section-(\d)$")


def failure_section_text(fe: Mapping[str, Any]) -> str:
    """The section-9 markdown for a ``failure_explanation`` object."""
    sc = fe.get("smallest_case")
    if sc:
        case = (f"Seed {sc.get('seed')}, {sc.get('rows')} real tape rows; frozen fixture "
                f"`{sc.get('fixture')}` (keys `tape`, `params_json`).")
    else:
        case = "No counterexample fixture was recorded (the failure is not seed-specific)."
    hyp = fe.get("hypothesis")
    hyp_text = (f"Hypothesis: {hyp}" if hyp else
                "Hypothesis: *not yet written. A fresh test-agent completes this item at K6-F, "
                "as a hypothesis with its evidence, never a verdict.*")
    return "\n\n".join([
        SECTION9[0],
        SECTION9[1], str(fe.get("what_failed") or "—"),
        SECTION9[2], case,
        SECTION9[3], str(fe.get("rule") or "—"),
        SECTION9[4], hyp_text,
        SECTION9[5], str(fe.get("downstream") or "—"),
    ]) + "\n"


def _cell_tags(cell) -> List[str]:
    return list(cell.get("metadata", {}).get("tags", []))


def notebook_structure_errors(nb) -> List[str]:
    """Every H8 structural rule that ``nb`` breaks (empty list = well-formed)."""
    errs: List[str] = []
    cells = list(nb.cells)
    if not cells:
        return ["notebook has no cells"]

    # headings: each exactly once, in order, in a cell tagged with its section
    positions = []
    for k, h in enumerate(HEADINGS, start=1):
        hits = [i for i, c in enumerate(cells) if c.cell_type == "markdown"
                and re.search(rf"^{re.escape(h)}\s*$", c.source, re.M)]
        if len(hits) != 1:
            errs.append(f"heading {h!r} found {len(hits)} times (expected exactly once)")
            continue
        positions.append(hits[0])
        if f"section-{k}" not in _cell_tags(cells[hits[0]]):
            errs.append(f"heading {h!r} is in a cell not tagged section-{k}")
    if positions != sorted(positions):
        errs.append("section headings are out of order")

    # the render cell and the optional persisted section 9
    render = [i for i, c in enumerate(cells) if "section-9-render" in _cell_tags(c)]
    s9 = [i for i, c in enumerate(cells) if "section-9" in _cell_tags(c)]
    if len(render) != 1:
        errs.append(f"expected exactly one section-9-render cell, found {len(render)}")
    else:
        r = render[0]
        if cells[r].cell_type != "code":
            errs.append("the section-9-render cell must be a code cell")
        tail = cells[r + 1:]
        if len(s9) > 1:
            errs.append(f"expected at most one section-9 cell, found {len(s9)}")
        if s9:
            if s9[-1] != len(cells) - 1 or s9[-1] != r + 1:
                errs.append("the section-9 cell must be the last cell, right after the render cell")
        elif tail:
            errs.append("the section-9-render cell must be the last cell")
    for i in s9:
        c = cells[i]
        if c.cell_type != "markdown":
            errs.append("the section-9 cell must be markdown")
            continue
        pos = [c.source.find(h) for h in SECTION9]
        if any(p < 0 for p in pos) or pos != sorted(pos):
            errs.append("the section-9 cell must hold 9 and 9.1-9.5 in order")

    # section tags on every other cell, non-decreasing
    current = 0
    for i, c in enumerate(cells):
        tags = _cell_tags(c)
        if "section-9-render" in tags or "section-9" in tags:
            continue
        secs = [int(m.group(1)) for m in (_SECTION_TAG_RE.match(t) for t in tags) if m]
        secs = [k for k in secs if 1 <= k <= 8]
        if len(secs) != 1:
            errs.append(f"cell {i} carries {len(secs)} section-<k> tags (expected exactly one)")
            continue
        if secs[0] < current:
            errs.append(f"cell {i} tagged section-{secs[0]} after section-{current}")
        current = max(current, secs[0])

    # the parameter cell
    params = [c for c in cells if "parameters" in _cell_tags(c)]
    if len(params) != 1:
        errs.append(f"expected exactly one parameters cell, found {len(params)}")
    else:
        if params[0].cell_type != "code":
            errs.append("the parameters cell must be a code cell")
        if "section-3" not in _cell_tags(params[0]):
            errs.append("the parameters cell must be tagged section-3")
    return errs


PASS_TWIN_LINE = "The notebook passed; its audit twin failed"


def append_failure_section(notebook_path, results: Mapping[str, Any], *, hypothesis: str,
                           downstream: Optional[str] = None,
                           explanation: Optional[str] = None) -> Dict[str, Any]:
    """Persist section 9 as the notebook's last cell (K6-F) and return the updated payload.

    Any earlier ``section-9`` cell is replaced. The returned payload carries the
    hypothesis (and ``downstream``, when given) with its paragraph regenerated;
    the caller writes it back to the results JSON.

    When ``explanation`` (the audit twin's failure text) is given, section 9 is written
    from it -- 9.1 opens with "The notebook passed; its audit twin failed" -- and the
    results payload is returned **unchanged** (a PASS payload keeps its null
    ``failure_explanation``; ``validate_results``' PASS <=> null rule is untouched).
    """
    path = Path(notebook_path)
    if explanation is not None:
        _write_section9(path, {"what_failed": f"{PASS_TWIN_LINE}: {_one_line(explanation)}",
                               "smallest_case": None, "rule": "—",
                               "hypothesis": hypothesis, "downstream": downstream})
        return copy.deepcopy(dict(results))
    updated = copy.deepcopy(dict(results))
    fe = dict(updated.get("failure_explanation") or {})
    fe["hypothesis"] = hypothesis
    if downstream is not None:
        fe["downstream"] = downstream
    fe["paragraph"] = _one_line(
        f"{updated.get('cert_id')} NOT CERTIFIED ({', '.join(updated.get('failure_kinds', []))}). "
        f"What failed: {fe.get('what_failed') or '—'} "
        f"Smallest case: {_case_text(fe.get('smallest_case'))}. Rule: {fe.get('rule') or '—'}. "
        f"Fault (hypothesis): {hypothesis}. Downstream: {fe.get('downstream') or '—'}.")
    updated["failure_explanation"] = fe
    _write_section9(path, fe)
    return updated


def _write_section9(path: Path, fe: Mapping[str, Any]) -> None:
    """Replace any earlier ``section-9`` cell with one rendered from ``fe``, appended last."""
    nb = nbformat.read(path, as_version=4)
    nb.cells = [c for c in nb.cells if "section-9" not in _cell_tags(c)]
    nb.cells.append(nbformat.v4.new_markdown_cell(failure_section_text(fe), metadata={"tags": ["section-9"]}))
    nbformat.write(nb, path)


def _case_text(sc) -> str:
    if not sc:
        return "no counterexample fixture was recorded"
    return f"seed {sc.get('seed')} ({sc.get('rows')} rows), frozen as {Path(str(sc.get('fixture'))).name}"


# ===========================================================================
# H7 -- the headless gate
# ===========================================================================
@dataclass
class GateResult:
    """Outcome of one headless execution (``passed`` is the gate's verdict)."""

    path: Path
    executed: bool
    error: Optional[str]
    results: Optional[Dict[str, Any]]
    results_errors: List[str]
    notebook: Any

    @property
    def passed(self) -> bool:
        return bool(self.executed and self.results is not None and self.results_errors == []
                    and self.results.get("status") == "PASS")


_PARAM_TYPES = (bool, int, float, str, type(None))


def _injected_source(parameters: Mapping[str, Any]) -> str:
    lines = ["# Parameters injected by cert_support.execute_notebook"]
    for name, value in parameters.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"parameter name {name!r} is not an identifier")
        if type(value) not in _PARAM_TYPES:
            raise ValueError(f"parameter {name}={value!r}: only int, float, str, bool or None may be injected")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"parameter {name}={value!r}: non-finite floats cannot be injected")
        lines.append(f"{name} = {value!r}")
    return "\n".join(lines) + "\n"


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def execute_notebook(path, *, parameters: Optional[Mapping[str, Any]] = None, results_dir=None,
                     fixture_dir=None, timeout: int = 1800, startup_timeout: int = 120) -> GateResult:
    """Execute an in-memory copy of the notebook through ``nbclient`` (H7).

    The file on disk is never written. The kernel is ``python3`` (the caller's
    interpreter) with ``PYTHONPATH`` headed by the repository root, CPU-only JAX,
    and ``OMIE_CERT_RESULTS_DIR`` / ``OMIE_CERT_FIXTURE_DIR`` set to the given
    directories or fresh temporaries. Raises ``ValueError`` only for an
    unusable parameter request.
    """
    path = Path(path)
    nb = nbformat.read(path, as_version=4)
    # R3a: the id the results JSON must carry -- the injected CERT_ID, else the file name's
    # ``C<nn>_`` prefix, else none (the template and the gate probes are tied to no id).
    if parameters is not None and "CERT_ID" in parameters:
        want_id = parameters["CERT_ID"]
    else:
        m_id = re.match(r"^(C\d{2})_", path.name)
        want_id = m_id.group(1) if m_id else None
    if parameters is not None:
        src = _injected_source(parameters)
        idx = [i for i, c in enumerate(nb.cells) if "parameters" in _cell_tags(c)]
        if len(idx) != 1:
            raise ValueError(f"parameter injection needs exactly one 'parameters' cell, found {len(idx)}")
        cell = nbformat.v4.new_code_cell(src, metadata={"tags": ["injected-parameters"]})
        nb.cells.insert(idx[0] + 1, cell)
    # nbclient tolerates an error in a cell tagged ``raises-exception``; the gate does not.
    for c in nb.cells:
        tags = c.get("metadata", {}).get("tags")
        if tags and "raises-exception" in tags:
            c.metadata["tags"] = [t for t in tags if t != "raises-exception"]

    rdir = Path(results_dir) if results_dir is not None else Path(tempfile.mkdtemp(prefix="omie_cert_results_"))
    fdir = Path(fixture_dir) if fixture_dir is not None else Path(tempfile.mkdtemp(prefix="omie_cert_fixtures_"))
    rdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + old if old else "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"
    env[RESULTS_ENV] = str(rdir.resolve())
    env[FIXTURE_ENV] = str(fdir.resolve())

    # Results JSONs already present (e.g. from an earlier run) are not this run's evidence.
    before = {q.name: (q.stat().st_mtime_ns, q.read_bytes()) for q in rdir.glob("*.json")}

    client = nbclient.NotebookClient(nb, timeout=timeout, startup_timeout=startup_timeout,
                                     kernel_name="python3",
                                     resources={"metadata": {"path": str(path.parent)}})
    executed, error = True, None
    try:
        client.execute(env=env)
    except Exception as exc:  # noqa: BLE001 - CellExecutionError, DeadKernelError, timeouts
        executed = False
        error = _ANSI_RE.sub("", f"{type(exc).__name__}: {exc}")
    if executed:
        errs = [o for c in nb.cells if c.get("cell_type") == "code" for o in c.get("outputs", [])
                if o.get("output_type") == "error"]
        if errs:
            executed = False
            error = _ANSI_RE.sub("", f"{errs[0].get('ename')}: {errs[0].get('evalue')}")

    def fresh(q: Path) -> bool:
        prev = before.get(q.name)
        return prev is None or q.stat().st_mtime_ns != prev[0] or q.read_bytes() != prev[1]

    results, results_errors = None, []
    found = sorted(q for q in rdir.glob("*.json") if fresh(q))
    if len(found) > 1:
        results_errors = [f"more than one results JSON in {rdir}: {[p.name for p in found]}"]
    elif len(found) == 1:
        try:
            results = json.loads(found[0].read_text(encoding="utf-8"))
            results_errors = validate_results(results)
        except Exception as exc:  # noqa: BLE001
            results_errors = [f"unreadable results JSON {found[0].name}: {exc!r}"]
        # R3a: the payload must carry the id this run was meant to certify.
        if want_id is not None and isinstance(results, dict) and results.get("cert_id") != want_id:
            results_errors.append(f"results cert_id {results.get('cert_id')!r} != expected {want_id!r}")
    else:
        results_errors = [f"no results JSON written by this run in {rdir}"
                          + (f" (stale, ignored: {sorted(before)})" if before else "")]
    return GateResult(path=path, executed=executed, error=error, results=results,
                      results_errors=results_errors, notebook=nb)


# ===========================================================================
# H9 -- the K5 routing rule (defined once, here)
# ===========================================================================
class Route(str, Enum):
    """How a scenario closes after K5. There is no stop or await member."""

    K6 = "K6"
    K6F_MERGE_EVIDENCE = "K6-F merge-evidence"
    K6F_RECORD_ONLY = "K6-F record-only"

    @property
    def merges(self) -> bool:
        return self is not Route.K6F_RECORD_ONLY

    @property
    def signs_certificate(self) -> bool:
        return self is Route.K6


@dataclass(frozen=True)
class K5Outcome:
    """The K5 facts the route is computed from.

    ``suite_regressed``: a per-test regression against the ``main`` baseline
    anywhere in the merge-gate suite, previously certified notebooks included.
    """

    notebook_executed: bool
    results_status: Optional[str]
    audit_twin_passed: bool
    suite_regressed: bool
    independence_ok: bool


def route_k5(outcome: K5Outcome) -> Route:
    """The single-sourced K5 routing rule (orchestrate.md §Certification cycle)."""
    if outcome.suite_regressed:
        return Route.K6F_RECORD_ONLY
    if (outcome.notebook_executed and outcome.results_status == "PASS" and outcome.audit_twin_passed
            and outcome.independence_ok):
        return Route.K6
    return Route.K6F_MERGE_EVIDENCE


def result_word(route: Route, provisional_on: Sequence[str] = ()) -> str:
    """The log's Result word for a route."""
    if route is not Route.K6:
        return "NOT CERTIFIED"
    if provisional_on:
        return "CERTIFIED — provisional on " + ", ".join(provisional_on)
    return "CERTIFIED"


HUMAN_VERDICTS = ("REPEAT", "DEFECT-FOUND", "SCOPE-CHANGE", "INVALID")
LADDER = tuple(f"C{i:02d}" for i in range(18))


# ===========================================================================
# H10 -- reading the log
# ===========================================================================
class LogParseError(ValueError):
    """The certification log is malformed (never a skip)."""


class GateSelectionError(LookupError):
    """A gated scenario does not resolve to exactly one notebook."""


@dataclass(frozen=True)
class LogEntry:
    n: int
    date: str
    cert_id: str
    scenario: str
    result: str
    verdict: str


_ENTRY_RE = re.compile(r"^### (\d+) · (\d{4}-\d{2}-\d{2}) · `(C\d{2})` — (.+)$")
_RESULT_RE = re.compile(r"^(CERTIFIED|CERTIFIED — provisional on C\d{2}(?:, C\d{2})*|NOT CERTIFIED)$")
_RESULT_LINE_RE = re.compile(r"^- \*\*Result:\*\*(.*)$")
_VERDICT_LINE_RE = re.compile(r"^- \*\*Verdict:\*\*(.*)$")
_PLACEHOLDER_VERDICT = re.compile(r"^_*$")


def _entries_lines(text: str) -> List[str]:
    """Lines under ``## Entries`` up to the next ``## `` heading, fenced code skipped."""
    lines = text.split("\n")
    out: List[str] = []
    inside, fence = False, False
    for ln in lines:
        if _FENCE_RE.match(ln):
            fence = not fence
            continue
        if fence:
            continue
        if not inside:
            if ln.rstrip() == "## Entries":
                inside = True
            continue
        if ln.startswith("## "):
            break
        out.append(ln)
    return out


def parse_log(path) -> Tuple[LogEntry, ...]:
    """Parse the entries of ``certification-log.md`` (H10). Raises :class:`LogParseError`."""
    return _parse_log_text(Path(path).read_text(encoding="utf-8"))


def _parse_log_text(text: str) -> Tuple[LogEntry, ...]:
    entries: List[LogEntry] = []
    cur: Optional[Dict[str, Any]] = None

    def finish(c):
        if c is None:
            return
        if c["result"] is None:
            raise LogParseError(f"entry {c['n']}: missing Result line")
        if not _RESULT_RE.match(c["result"]):
            raise LogParseError(f"entry {c['n']}: ill-formed Result {c['result']!r}")
        if c["verdict"] is None:
            raise LogParseError(f"entry {c['n']}: missing Verdict line")
        if entries and c["n"] <= entries[-1].n:
            raise LogParseError(f"entry numbers must strictly increase: {entries[-1].n} then {c['n']}")
        entries.append(LogEntry(c["n"], c["date"], c["cert_id"], c["scenario"], c["result"], c["verdict"]))

    for ln in _entries_lines(text):
        m = _ENTRY_RE.match(ln)
        if m:
            finish(cur)
            cur = {"n": int(m.group(1)), "date": m.group(2), "cert_id": m.group(3),
                   "scenario": m.group(4).strip(), "result": None, "verdict": None}
            continue
        if ln.startswith("### "):
            if re.match(r"^### \d+ ", ln):
                raise LogParseError(f"ill-formed entry heading {ln!r}")
            finish(cur)
            cur = None
            continue
        if cur is None:
            continue
        mr = _RESULT_LINE_RE.match(ln)
        if mr:
            if cur["result"] is not None:
                raise LogParseError(f"entry {cur['n']}: two Result lines")
            cur["result"] = mr.group(1).strip()
            continue
        mv = _VERDICT_LINE_RE.match(ln)
        if mv:
            if cur["verdict"] is not None:
                raise LogParseError(f"entry {cur['n']}: two Verdict lines")
            body = mv.group(1)
            word = body.split("·", 1)[0].strip() if "**rationale:**" in body else body.strip()
            cur["verdict"] = "" if _PLACEHOLDER_VERDICT.match(word) else word
    finish(cur)
    return tuple(entries)


def _latest(entries: Sequence[LogEntry]) -> Dict[str, LogEntry]:
    out: Dict[str, LogEntry] = {}
    for e in entries:
        if e.cert_id not in out or e.n > out[e.cert_id].n:
            out[e.cert_id] = e
    return out


def _is_gated(e: LogEntry) -> bool:
    """The H10 predicate on a scenario's latest entry: Result CERTIFIED*, Verdict not human."""
    return e.result.startswith("CERTIFIED") and e.verdict not in HUMAN_VERDICTS


def gated_notebooks(log_path, notebook_dir) -> List[Path]:
    """Notebooks the merge gate re-executes: latest Result CERTIFIED*, Verdict not human."""
    nbdir = Path(notebook_dir)
    out = []
    for cid, e in sorted(_latest(parse_log(log_path)).items()):
        if not _is_gated(e):
            continue
        hits = sorted(nbdir.glob(f"{cid}_*.ipynb"))
        if len(hits) != 1:
            raise GateSelectionError(f"gated scenario {cid} matches {len(hits)} notebooks in {nbdir}")
        out.append(hits[0])
    return sorted(out)


def next_scenario(log_path, order: Sequence[str] = LADDER) -> Optional[str]:
    """The first id in ``order`` with no log entry at all (None when exhausted)."""
    seen = {e.cert_id for e in parse_log(log_path)}
    return next((cid for cid in order if cid not in seen), None)


# ===========================================================================
# H11 -- writing the log and the certificate
# ===========================================================================
@dataclass(frozen=True)
class CloseOutcome:
    entry_n: int
    route: Route
    merged: bool
    signed: bool
    next_scenario: Optional[str]


_AWAIT_OPEN = "**Awaiting the human (K7-U):** "
_AWAIT_CLOSE = " — every"
_NONE_YET = "*(none yet)*"
_DASH = "—"


def _template_labels(log_text: str) -> List[str]:
    """Field labels, in order, from the log's own ``## Entry template`` block."""
    i = log_text.find("## Entry template")
    if i < 0:
        raise LogParseError("the log has no '## Entry template' section")
    a = log_text.find("```markdown", i)
    if a < 0:
        raise LogParseError("the entry template has no ```markdown block")
    b = log_text.find("```", a + len("```markdown"))
    block = log_text[a:b]
    return [m.group(1).rstrip(":") for m in re.finditer(r"^- \*\*([^*]+?)\*\*", block, re.M)]


_NO_RESULTS_EXPLANATION = "explanation missing — the results carry no failure explanation"


def _why_it_failed(results, *, twin_why: Optional[str], traceback: Optional[str]) -> str:
    """The non-K6 "Why it failed" text, by precedence: the results paragraph, else the
    audit twin's text (``twin_why``), else -- results present -- the "no explanation" line,
    else -- no results JSON -- the crash text."""
    if results and results.get("failure_explanation"):
        return _one_line(results["failure_explanation"]["paragraph"])
    if twin_why is not None and twin_why.strip():
        return _one_line(twin_why)
    if results:
        return _NO_RESULTS_EXPLANATION
    return _crash_explanation(traceback)


def _crash_explanation(traceback: Optional[str]) -> str:
    lines = [ln.strip() for ln in _ANSI_RE.sub("", traceback or "").splitlines() if ln.strip()]
    if not lines:
        tb = "unavailable"
    elif len(lines) == 1:
        tb = lines[0]
    else:
        tb = f"{lines[0]} … {lines[-1]}"
    return _one_line(f"explanation missing — notebook crashed; traceback: {tb}")


def _certifies(results) -> str:
    if not results:
        return _DASH
    qs = results.get("spec", {}).get("quotes", [])
    if not qs:
        return _DASH
    return "; ".join(f"{q['section']} ({q['source']} {q.get('version') or 'unversioned'})" for q in qs)


def _field_lines(route: Route, results, *, card, branch, commit, cid, provisional_on, traceback,
                 revealed, tip_sha, twin_why: Optional[str] = None) -> Dict[str, str]:
    r = results
    nb = r.get("notebook") if r else None
    f: Dict[str, str] = {
        "Card": f"- **Card:** `{card}` · **Branch:** `{branch}` · **Commit:** `{commit}`",
        "Notebook": (f"- **Notebook:** `notebooks/certification/{nb}` · **Results:** `results/{cid}.json`"
                     if nb else f"- **Notebook:** {_DASH} · **Results:** {_DASH}"),
        "Certifies": f"- **Certifies:** {_certifies(r)}",
    }
    if r:
        d, s, g = r["draws"], r["seeds"], r["generator"]
        digests = ", ".join(f"{k} {v[:16]}" for k, v in g.get("digests", {}).items()) or "no construction digest"
        f["Dataset"] = (f"- **Dataset:** {', '.join(g.get('digests', {})) or 'no registered construction'}; "
                        f"{d['valid']} valid of {d['requested']} requested draws; seeds {s['first']}–{s['last']}; "
                        f"generator digest {digests}")
        f["Known result"] = f"- **Known result:** {r['known_result']}"
        orc = [a for a in r["assertions"] if a["kind"] == "oracle"]
        inv = r["invariants"]
        unobs = sorted({u for row in inv for u in row.get("unobserved", [])})
        f["Outcome"] = (f"- **Outcome:** {sum(a['status'] == 'pass' for a in orc)}/{len(orc)} oracle assertions "
                        f"passed; invariants {sum(x['status'] == 'pass' for x in inv)} pass / "
                        f"{sum(x['status'] == 'fail' for x in inv)} fail / "
                        f"{sum(x['status'] == 'not_applicable' for x in inv)} not applicable; unobserved: "
                        f"{', '.join(unobs) or 'none'}; discard rate {float(d['discard_rate']):.3f}")
        pk, ev = r["capacity"]["peaks"], r["capacity"]["envelope"]
        fmt = lambda v: "unobserved" if v is None else str(v)  # noqa: E731
        f["Capacity peaks"] = ("- **Capacity peaks:** " + ", ".join(
            f"{k} {fmt(pk.get(k))}" for k in CAPACITY_AXES) + " — against the "
            + "/".join(str(ev.get(k)) for k in CAPACITY_AXES) + " envelope")
        ej = r["eager_jit_identical"]
        f["Eager vs jitted"] = "- **Eager vs jitted:** " + (
            "identical" if ej is True else "not compared" if ej is None else "diverged: " + "; ".join(
                f"seed {a['seed']}: {a['detail'] or 'differs'}" for a in r["assertions"]
                if a["kind"] == "eager_jit" and a["status"] == "fail"))
        fc = r.get("runtime_forecast_s")
        f["Wall clock"] = f"- **Wall clock:** {r['wall_clock_s']} (forecast {fc if fc is not None else _DASH})"
    else:
        for lab in ("Dataset", "Known result", "Outcome", "Capacity peaks", "Eager vs jitted"):
            f[lab] = f"- **{lab}:** {_DASH}"
        f["Wall clock"] = f"- **Wall clock:** {_DASH} (forecast {_DASH})"
    f["Anything the run revealed that the card did not anticipate"] = (
        f"- **Anything the run revealed that the card did not anticipate:** {_one_line(revealed)}")
    f["Result"] = f"- **Result:** {result_word(route, provisional_on)}"
    if route is not Route.K6:
        why = _why_it_failed(r, twin_why=twin_why, traceback=traceback)
        f["Why it failed"] = f"- **Why it failed:** {why}"
    if route is Route.K6:
        merged = "branch merged"
    elif route is Route.K6F_MERGE_EVIDENCE:
        merged = "evidence merged, certificate section unsigned"
    else:
        merged = (f"branch NOT merged — the existing OMIE suite regressed; entry written on main "
                  f"(branch `{branch}`, tip `{tip_sha or 'unknown'}`)")
    f["Merged"] = f"- **Merged:** {merged}"
    f["Verdict"] = ("- **Verdict:** CERTIFIED · **rationale:** K5 pass" if route is Route.K6
                    else "- **Verdict:** ______ · **rationale:** ______")
    return f


def _section_span(text: str, heading: str, until: str = r"^## ") -> Optional[Tuple[int, int]]:
    """``(end of the heading match, start of the next ``until`` match or EOF)``, or None."""
    m = re.search(heading, text, re.M)
    if not m:
        return None
    nxt = re.search(until, text[m.end():], re.M)
    return m.end(), (m.end() + nxt.start()) if nxt else len(text)


def _entries_bounds(text: str) -> Tuple[int, int]:
    span = _section_span(text, r"^## Entries[ \t]*$")
    if span is None:
        raise LogParseError("the log has no '## Entries' heading")
    return span


def _write_log(text: str, entry: str, toc_row: str, *, require_awaiting: bool) -> str:
    # (1) the entry, at the end of the Entries section
    a, b = _entries_bounds(text)
    body = text[a:b]
    stripped = body.strip()
    if stripped == _NONE_YET or stripped == "":
        new_body = "\n\n" + entry.rstrip() + "\n"
    else:
        new_body = body.rstrip() + "\n\n" + entry.rstrip() + "\n"
    if b < len(text):
        new_body += "\n"
    text = text[:a] + new_body + text[b:]

    # (2) the table of contents
    lines = text.split("\n")
    k = next((i for i, ln in enumerate(lines) if ln.startswith("| # | Date | Scenario")), None)
    if k is None:
        raise LogParseError("the log has no table-of-contents table")
    end = k + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    rows = [ln for ln in lines[k + 2:end] if "*(no entries yet" not in ln]
    rows.append(toc_row)
    lines[k + 2:end] = rows
    text = "\n".join(lines)

    # (3) the awaiting-the-human list, recomputed from the new log (R1d): every NOT CERTIFIED
    # entry whose Verdict is still blank, in entry order.
    i = text.find(_AWAIT_OPEN)
    if i < 0:
        if require_awaiting:
            raise LogParseError("the log has no 'Awaiting the human (K7-U)' list")
        return text
    i += len(_AWAIT_OPEN)
    j = text.find(_AWAIT_CLOSE, i)
    if j < 0:
        raise LogParseError("the log's 'Awaiting the human (K7-U)' list has no closing ' — every'")
    pending = [f"`{e.cert_id}` (entry {e.n})" for e in _parse_log_text(text)
               if e.result == "NOT CERTIFIED" and e.verdict == ""]
    return text[:i] + (", ".join(pending) or _NONE_YET) + text[j:]


_SECTIONS_RE = r"^## Scenario sections[ \t]*$"
_SECTIONS_HEAD = ("## Scenario sections\n\nOne section per scenario, replaced in place when the scenario is "
                  "re-closed (the log keeps every run); written by "
                  "`notebooks.certification.cert_support.close_scenario`, unsigned by "
                  "`notebooks.certification.cert_support.unsign_scenario`, never by hand.\n\n---\n\n")
_REVOKED_STATUS = "REVOKED — section unsigned pending re-run"
_UNSIGNED_PREFIX = "*unsigned"


def _scenario_section_spans(text: str, cid: str) -> List[Tuple[int, int]]:
    """Spans ``(heading start, section end)`` of every ``### <cid> — `` section inside the
    ``## Scenario sections`` region. Shared by ``close_scenario`` and ``unsign_scenario``.

    A section runs from its heading line to the next ``### `` heading, or to the end of the
    region, not counting the region's trailing ``---``. The span includes the section's
    trailing blank lines, which a replacement preserves.
    """
    span = _section_span(text, _SECTIONS_RE)
    if span is None:
        return []
    start, end = span
    stripped = text[start:end].rstrip()
    limit = start + len(stripped) - 3 if stripped.endswith("---") else end
    heads = [start + m.start() for m in re.finditer(r"^### ", text[start:limit], re.M)]
    return [(h, heads[k + 1] if k + 1 < len(heads) else limit)
            for k, h in enumerate(heads) if text.startswith(f"### {cid} — ", h)]


def _one_section(text: str, cid: str) -> Optional[Tuple[int, int]]:
    """The id's single section span (None when it has none). ValueError on a duplicate."""
    spans = _scenario_section_spans(text, cid)
    if len(spans) > 1:
        raise ValueError(f"the certificate holds {len(spans)} sections for {cid}; expected at most one")
    return spans[0] if spans else None


def _replace_span(text: str, span: Tuple[int, int], new: str) -> str:
    """Replace ``text[span]`` by ``new``, keeping the old span's trailing whitespace."""
    a, b = span
    old = text[a:b]
    return text[:a] + new.rstrip() + old[len(old.rstrip()):] + text[b:]


def _set_matrix_row(text: str, cid: str, status: str, date: str, n: int) -> str:
    """Overwrite Status, Date and Entry of the id's coverage-matrix row, when it has one."""
    num = int(cid[1:])
    cm = _section_span(text, r"^## Coverage matrix", until=r"^#{2,3} ")
    if not cm:
        return text
    cstart, cend = cm
    lines = text[cstart:cend].split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(f"| C{num} "):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) >= 5:
                cells[2], cells[3], cells[4] = status, date, str(n)
                lines[i] = "| " + " | ".join(cells) + " |"
            break
    return text[:cstart] + "\n".join(lines) + text[cend:]


def _write_certificate(text: str, *, cid: str, scenario: str, route: Route, results, date: str, n: int,
                       provisional_on, branch: str, commit: str, tip_sha: Optional[str]) -> str:
    """The certificate text after closing ``cid`` (H11 as amended by R1). Raises ValueError
    when the id already has two or more sections."""
    existing = _one_section(text, cid)
    if route is Route.K6F_RECORD_ONLY and existing is None:
        return text  # a record-only first run leaves the certificate byte-identical
    word = result_word(route, provisional_on)
    if route is Route.K6:
        status, row_status = word, word
        signed = f"{date} · log entry {n} · results blake2b-128 {results_digest(results)}"
    else:
        status, row_status = "NOT CERTIFIED — section unsigned", "NOT CERTIFIED (section unsigned)"
        signed = "*unsigned — NOT CERTIFIED; awaiting the human (K7-U)*"
    if route is Route.K6F_RECORD_ONLY:
        evidence = f"record only — branch `{branch}` NOT merged · tip `{tip_sha or 'unknown'}` · log entry {n}"
    elif results:
        evidence = (f"`notebooks/certification/results/{cid}.json` · log entry {n} · "
                    f"branch `{branch}` · commit `{commit}`")
    else:
        evidence = (f"no results JSON (the notebook crashed) · log entry {n} · branch `{branch}` · "
                    f"commit `{commit}`")
    section = "\n".join([
        f"### {cid} — {scenario}", "",
        f"- **Status:** {status}",
        f"- **Signed:** {signed}",
        f"- **Certifies:** {_certifies(results)}",
        f"- **Evidence:** {evidence}",
    ])

    if existing is not None:
        # (1a) a re-close replaces the section in place (R1a, R1b); no supersede marker
        text = _replace_span(text, existing, section)
    else:
        # (1b) a first close appends at the end of "## Scenario sections" (created if absent)
        if _section_span(text, _SECTIONS_RE) is None:
            anchor = re.search(r"^## Appendix A", text, re.M)
            pos = anchor.start() if anchor else len(text)
            text = text[:pos] + _SECTIONS_HEAD + text[pos:]
        start, end = _section_span(text, _SECTIONS_RE)
        region = text[start:end].rstrip()
        sep = region.endswith("---")
        if sep:
            region = region[:-3].rstrip()
        new_region = region + "\n\n" + section + "\n\n" + ("---\n\n" if sep else "")
        text = text[:start] + new_region + text[end:]

    # (2) the coverage-matrix row, when the id has one
    return _set_matrix_row(text, cid, row_status, date, n)


def unsign_scenario(*, cert_id: str, log_path, certificate_path) -> bool:
    """Unsign ``cert_id``'s certificate section once the human has revoked it (R2).

    Call it after the human's correction entry (e.g. a retroactive ``REPEAT``) is in the log
    and before the re-run starts. Returns True iff the certificate changed.

    Raises ``ValueError`` for a malformed id, an id with no log entry, an id the log still
    gates (H10 predicate: only a verdict the human typed can revoke a section) or a duplicate
    section; ``LogParseError`` propagates. An id with no section, or whose section is already
    unsigned, is a no-op. Certifies and Evidence are kept byte-identical. Never writes the
    log; never prompts.
    """
    if not isinstance(cert_id, str) or not _CID_RE.match(cert_id):
        raise ValueError(f"cert_id must match C<nn>, got {cert_id!r}")
    latest = _latest(parse_log(log_path)).get(cert_id)
    if latest is None:
        raise ValueError(f"{cert_id} has no entry in the certification log")
    if _is_gated(latest):
        raise ValueError(f"{cert_id} is still gated by log entry {latest.n} (Result {latest.result!r}, "
                         f"Verdict {latest.verdict!r}); the human's verdict must be in the log first")
    certificate_path = Path(certificate_path)
    text = certificate_path.read_text(encoding="utf-8")
    span = _one_section(text, cert_id)
    if span is None:
        return False
    sec = text[span[0]:span[1]]
    m_signed = re.search(r"^- \*\*Signed:\*\* ?(.*)$", sec, re.M)
    if m_signed is None or m_signed.group(1).startswith(_UNSIGNED_PREFIX):
        return False
    status_line = f"- **Status:** {_REVOKED_STATUS}"
    signed_line = f"- **Signed:** *unsigned — revoked by the human (log entry {latest.n}); awaiting re-run*"
    sec = re.sub(r"^- \*\*Status:\*\*.*$", lambda _m: status_line, sec, count=1, flags=re.M)
    sec = re.sub(r"^- \*\*Signed:\*\*.*$", lambda _m: signed_line, sec, count=1, flags=re.M)
    text = text[:span[0]] + sec + text[span[1]:]
    text = _set_matrix_row(text, cert_id, "REVOKED (section unsigned)", latest.date, latest.n)
    certificate_path.write_text(text, encoding="utf-8")
    return True


def _cert_id_of(card: str, branch: str, results) -> str:
    if results and re.match(r"^C\d{2}$", str(results.get("cert_id", ""))):
        return results["cert_id"]
    for src, pat in ((card, r"OMIE-CERT-(C\d{2})-"), (branch, r"cert-c(\d{2})")):
        m = re.search(pat, src or "")
        if m:
            return m.group(1) if m.group(1).startswith("C") else "C" + m.group(1)
    raise ValueError(f"cannot determine the scenario id from card {card!r} / branch {branch!r}")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CID_RE = re.compile(r"^C\d{2}$")


def _validate_close_inputs(route: Route, results, *, cid: str, scenario: str, date: str,
                           provisional_on: Sequence[str]) -> None:
    """Reject inputs that would write an unparseable or unjustified entry. Raises ValueError
    before anything is written."""
    if not isinstance(date, str) or not _DATE_RE.match(date):
        raise ValueError(f"date must be YYYY-MM-DD, got {date!r}")
    if not isinstance(scenario, str) or not scenario.strip() or any(c in scenario for c in "\r\n"):
        raise ValueError(f"scenario must be a non-empty single line, got {scenario!r}")
    bad = [p for p in provisional_on if not isinstance(p, str) or not _CID_RE.match(p)]
    if bad:
        raise ValueError(f"provisional_on must hold scenario ids C<nn>, got {bad!r}")
    if route is Route.K6:
        # A K6 signature attests a K5 pass recorded in a results JSON: nothing to sign otherwise.
        if results is None:
            raise ValueError("Route K6 requires a results payload (nothing to sign without one)")
        if results.get("status") != "PASS":
            raise ValueError(f"Route K6 requires a PASS results payload, got status {results.get('status')!r}")


def close_scenario(*, route: Route, results: Optional[Mapping[str, Any]], log_path, certificate_path, card: str,
                   branch: str, commit: str, scenario: str, date: str, provisional_on: Sequence[str] = (),
                   traceback: Optional[str] = None, revealed: str = "nothing", tip_sha: Optional[str] = None,
                   order: Sequence[str] = LADDER, twin_why: Optional[str] = None) -> CloseOutcome:
    """Close a scenario at K6 / K6-F (H11): log entry, table of contents, awaiting list,
    certificate section and coverage-matrix row. Never asks; never waits.

    ``twin_why`` is the audit twin's failure text. On a non-K6 route "Why it failed" takes
    the results paragraph, else ``twin_why``, else (results present) "explanation missing —
    the results carry no failure explanation", else (no results JSON) the crash text."""
    route = Route(route)
    log_path, certificate_path = Path(log_path), Path(certificate_path)
    results = dict(results) if results is not None else None
    cid = _cert_id_of(card, branch, results)
    _validate_close_inputs(route, results, cid=cid, scenario=scenario, date=date,
                           provisional_on=tuple(provisional_on))

    log_text = log_path.read_text(encoding="utf-8")
    existing = parse_log(log_path)
    n = (existing[-1].n + 1) if existing else 1
    labels = _template_labels(log_text)
    fields = _field_lines(route, results, card=card, branch=branch, commit=commit, cid=cid,
                          provisional_on=tuple(provisional_on), traceback=traceback, revealed=revealed,
                          tip_sha=tip_sha, twin_why=twin_why)
    body = [fields.get(lab, f"- **{lab}:** {_DASH}") for lab in labels if lab in fields or lab != "Why it failed"]
    entry = f"### {n} · {date} · `{cid}` — {scenario}\n\n" + "\n".join(body) + "\n"

    word = result_word(route, tuple(provisional_on))
    draws = str(results["draws"]["valid"]) if results else _DASH
    if route is Route.K6:
        note = "auto-certified at K6"
    elif route is Route.K6F_MERGE_EVIDENCE:
        note = ("crashed before section 9" if results is None
                else ", ".join(results.get("failure_kinds", [])) or "K5 failure") + "; evidence merged"
    else:
        note = "record only: the existing OMIE suite regressed"
    verdict_cell = "CERTIFIED" if route is Route.K6 else "______"
    toc_row = (f"| {n} | {date} | `{cid}` — {scenario.replace('|', '/')} | {draws} | {word} | "
               f"{verdict_cell} | {note.replace('|', '/')} |")
    # Build both documents in memory (R1c) and prove the new log parses before touching disk:
    # the log is append-only, so a writer must never leave it unparseable (H11).
    new_log = _write_log(log_text, entry, toc_row, require_awaiting=route is not Route.K6)
    if [e.n for e in _parse_log_text(new_log)] != [e.n for e in existing] + [n]:
        raise LogParseError(f"the new entry {n} does not round-trip through parse_log")
    cert_text = new_cert = None
    if route.merges or certificate_path.exists():
        # Record-only reads it too: an existing section must be unsigned (R1b), and a
        # duplicate section raises ValueError here, before anything is written.
        cert_text = certificate_path.read_text(encoding="utf-8")
        new_cert = _write_certificate(cert_text, cid=cid, scenario=scenario, route=route, results=results,
                                      date=date, n=n, provisional_on=tuple(provisional_on), branch=branch,
                                      commit=commit, tip_sha=tip_sha)

    if new_cert is not None and new_cert != cert_text:
        certificate_path.write_text(new_cert, encoding="utf-8")
    log_path.write_text(new_log, encoding="utf-8")

    return CloseOutcome(entry_n=n, route=route, merged=route.merges, signed=route.signs_certificate,
                        next_scenario=next_scenario(log_path, order))
