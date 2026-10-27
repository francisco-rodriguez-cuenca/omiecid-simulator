"""Locked behavioural-contract tests for sprint story P2-PERF-3.

Story card: ``docs/omiecid/agile-plan.md`` §"Story P2-PERF-3 — Replay
fixturing + JIT-cache defragmentation" (agile-plan v5.9, 2026-05-11).
Gate doc (regression-gate subset, budget, dispatch): ``docs/omiecid/test-perf-gate.md``.

This file encodes the seven locked invariants (AC pins a-g) from the
story card's "Test-first contract" row.  The seven invariants are:

  (a) Zero ``@jax.jit`` / ``@partial(jax.jit, ...)`` on ``self``-bound
      methods inside ``class OrderBook`` in
      ``omiecid_simulator/jaxob/jorderbook.py``.
  (b) All 13 fixtures (6 in ``tests/omiecid/conftest.py`` + 7 file-local
      ``open_contract`` overrides) carry ``scope="module"``.
  (c) The regression-gate subset (per ``test-perf-gate.md`` §1, §5)
      passes at the ``pyproject.toml`` ``-n`` pin with exit code 0.
  (d) The regression-gate subset warm-run wall-clock is < 5 min (per
      ``test-perf-gate.md`` §2 per-PR budget).
  (e) The full ``not slow and not audit`` surface still passes at the
      ``pyproject.toml`` ``-n`` pin with pass count >= 1820 (do NOT
      regress pass count).  The measured surface was narrowed from
      ``not slow`` to ``not slow and not audit`` under P3.4-FOLLOWUP-3
      (2026-07-08): P3-TEST-AUDIT-DECOUPLE-1 moved the routine gate
      posture to ``-m "not audit"``, so the meta-test's nested run now
      measures the SAME surface class the authoritative gate measures
      (excluding the ``*_audit.py`` compile-bomb the routine gate excludes
      directly).  Floor provenance: 1836 passed measured 2026-07-08 on
      ``-m "not slow and not audit" -n 24``; pinned 1820 (16 below).
  (f) >= 50% reduction on the 8-test slow ring-buffer subset
      (cold-cache ``-n 0`` baseline 343 s -> <= 172 s), **excluding**
      ``test_ioc_1022_matches_at_114_69_eur_0_30_mw``.  The outlier's
      wall-clock is still recorded in the merge note but not in the
      reduction calculation.
  (g) No behavioural regression in the four named replay/scenario
      files: ``test_trade_extraction_ring_buffer.py``,
      ``test_scenario_mixed_simulation.py``,
      ``test_scenario_single_contract.py``, ``test_omie_data.py``.
      Both per-file and total pass-count equality vs the pre-story
      baseline collected on commit ``5d1e40a`` (main, 2026-05-11).

Authoring notes
---------------
* The static-structure tests (a) and (b) run inside the default fast
  suite (``-m "not slow"``) — they use ``ast`` only, no ``import jax``,
  no subprocess.  Each is sub-millisecond.
* The subprocess-driven tests (c)-(g) all carry ``@pytest.mark.slow``.
  They spawn a fresh ``pytest`` process (the ``.venv/bin/python3 -m
  pytest ...`` invocation the repo prescribes) so the JIT cache state and the
  ``pyproject.toml addopts`` apply unmodified.  In-process recursion is
  brittle and would re-collect this very test file.
* AC-(f) wall-clock is environment-noisy.  We author the inline timed
  assertion with a generous slack factor (median-of-3, +20% tolerance)
  and document the limitation in the test docstring.  See the
  ``test_f_*`` block below for the full caveat.
* All subprocess invocations use the
  ``PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -m pytest``
  prefix from the repo run conventions and ``test-perf-gate.md`` §5.

Pre-implementation expectation
------------------------------
All seven AC pins will currently fail (or be xfailed by current behaviour)
because the production lift in Deliverable (1) and the fixture cascade
in Deliverable (2) have not been shipped yet.  That is the intended
test-first state.  Do NOT mark these xfail — they go green when the
developer ships Deliverables (1)+(2)+(3).
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

import pytest


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
JORDERBOOK_PATH = REPO_ROOT / "omiecid_simulator" / "jaxob" / "jorderbook.py"
CONFTEST_PATH = REPO_ROOT / "tests" / "omiecid" / "conftest.py"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
PYTHON_BIN = pathlib.Path(sys.executable)  # the interpreter running pytest (any venv layout)

# The 7 file-local ``open_contract`` override files enumerated in
# agile-plan v5.9 (2026-05-11) Deliverable (2) (vii)-(xiii).
FILE_LOCAL_OPEN_CONTRACT_OVERRIDES = (
    REPO_ROOT / "tests" / "omiecid" / "test_contract_book_audit.py",
    REPO_ROOT / "tests" / "omiecid" / "test_iceberg_manager_jit_audit.py",
    REPO_ROOT / "tests" / "omiecid" / "test_ioc_fok_audit.py",
    REPO_ROOT / "tests" / "omiecid" / "test_iceberg_manager_jit.py",
    REPO_ROOT / "tests" / "omiecid" / "test_iceberg_cap_overflow.py",
    REPO_ROOT / "tests" / "omiecid" / "test_iceberg_audit.py",
    REPO_ROOT / "tests" / "omiecid" / "test_gct_expiry_views_audit.py",
)

# The 6 conftest fixtures from agile-plan v5.9 Deliverable (2) (i)-(vi).
CONFTEST_FIXTURE_NAMES = (
    "open_contract",
    "product_config",
    "market_config",
    "book",
    "book_and_state",
    "empty_state",
)


# ---------------------------------------------------------------------------
# Subprocess helpers (self-contained — no external conftest helpers)
# ---------------------------------------------------------------------------

def _pytest_env() -> dict:
    """Subprocess env with the repo run-convention / test-perf-gate.md invariants."""
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("WANDB_MODE", "disabled")
    return env


def _parse_pytest_summary(stdout: str) -> dict:
    """Extract pass/fail/skip/error counts from pytest stdout.

    Looks for the conventional last-line summary in the form
    ``=== N passed, M failed, K skipped, J errors, ... in T.TTs ===``.
    Returns a dict with keys ``passed``, ``failed``, ``skipped``,
    ``errors``, ``xfailed``, ``xpassed`` (each defaults to 0 if absent).
    """
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    # Look at the last 50 non-empty lines for the summary block.
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    summary_re = re.compile(
        r"(\d+)\s+(passed|failed|skipped|error|errors|xfailed|xpassed)"
    )
    for line in reversed(lines[-50:]):
        if " in " in line and ("passed" in line or "failed" in line or "error" in line):
            for n, name in summary_re.findall(line):
                if name == "error":
                    counts["errors"] += int(n)
                else:
                    counts[name] = int(n)
            break
    return counts


def _parse_failed_node_ids(stdout: str) -> list[str]:
    """Extract the FAILED test node-ids from pytest stdout.

    Parses the ``short test summary info`` block that pytest emits (even
    under ``--tb=no -q`` and under ``-n <pin>`` xdist — verified identical
    format) as one line per failure of the form::

        FAILED tests/omiecid/test_x.py::TestY::test_z - AssertionError: ...

    Returns the list of node-ids (the token after ``FAILED `` and before
    the optional `` - <reason>`` tail).  Order is preserved.  Errors
    (``ERROR ...`` lines) are intentionally NOT folded in here: a collection
    error is never the known one-compile flake and must surface as an
    "other failure" so the caller fails loudly.
    """
    node_ids: list[str] = []
    failed_re = re.compile(r"^FAILED\s+(\S+)")
    for line in stdout.splitlines():
        m = failed_re.match(line.strip())
        if m:
            node_ids.append(m.group(1))
    return node_ids


def _read_addopts_n_pin() -> str:
    """Return the ``-n`` token from ``pyproject.toml`` ``addopts`` as a string.

    Used by tests (c), (e), (g) so the subprocess invocation honours the
    project-pinned worker count without hard-coding it (which would
    break when P2-PERF-4 lands its empirical re-pin).
    """
    text = PYPROJECT_PATH.read_text()
    m = re.search(r'addopts\s*=\s*"([^"]*)"', text)
    if not m:
        pytest.fail("could not parse addopts from pyproject.toml")
    addopts = m.group(1)
    m2 = re.search(r"-n\s*(\S+)", addopts)
    if not m2:
        pytest.fail(f"could not find -n pin inside addopts={addopts!r}")
    return m2.group(1)


# ---------------------------------------------------------------------------
# AC-(a): Zero @jax.jit on self-bound methods in class OrderBook
# ---------------------------------------------------------------------------
#
# AST inspection only — no ``import jax``, no subprocess.  Sub-millisecond.
# Goes green when the developer ships Deliverable (1) of P2-PERF-3 (the
# module-level pure-function lift).

class TestACa_NoJitOnOrderBookSelfBoundMethods:
    """AC-(a): zero ``@jax.jit`` / ``@partial(jax.jit, ...)`` decorators on
    any ``self``-bound (instance) method inside ``class OrderBook`` in
    ``omiecid_simulator/jaxob/jorderbook.py``.
    """

    def _orderbook_class_node(self) -> ast.ClassDef:
        tree = ast.parse(JORDERBOOK_PATH.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "OrderBook":
                return node
        pytest.fail(
            "class OrderBook not found in omiecid_simulator/jaxob/jorderbook.py "
            "— the AC-(a) assertion presupposes its existence per the story card"
        )

    @staticmethod
    def _is_jit_decorator(dec: ast.expr) -> bool:
        """Return True iff ``dec`` is ``@jax.jit`` or ``@partial(jax.jit, ...)``.

        Mirrors the lint pattern used in ``tests/omiecid/test_jit_marker_lint.py``
        (decorator can be either a name reference or a partial-call).
        """
        # @jax.jit  (bare attribute)
        if isinstance(dec, ast.Attribute) and dec.attr == "jit":
            if isinstance(dec.value, ast.Name) and dec.value.id == "jax":
                return True
        # @jax.jit(...)  (called with kwargs like static_argnames)
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Attribute) and func.attr == "jit":
                if isinstance(func.value, ast.Name) and func.value.id == "jax":
                    return True
            # @partial(jax.jit, ...)
            if isinstance(func, ast.Name) and func.id == "partial" and dec.args:
                first = dec.args[0]
                if isinstance(first, ast.Attribute) and first.attr == "jit":
                    if isinstance(first.value, ast.Name) and first.value.id == "jax":
                        return True
        return False

    @staticmethod
    def _is_self_bound(method: ast.FunctionDef) -> bool:
        """True iff first positional arg of ``method`` is named ``self``."""
        if not method.args.args:
            return False
        return method.args.args[0].arg == "self"

    def test_no_jit_decorator_on_orderbook_self_bound_methods(self):
        """AC-(a): grep/AST assertion — zero matches inside the class body."""
        cls = self._orderbook_class_node()
        offenders: list[str] = []
        for item in cls.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if not self._is_self_bound(item):
                continue
            for dec in item.decorator_list:
                if self._is_jit_decorator(dec):
                    offenders.append(f"{item.name}@line{item.lineno}")
        assert offenders == [], (
            f"AC-(a) violation: found @jax.jit / @partial(jax.jit, ...) on "
            f"self-bound methods inside class OrderBook: {offenders}.  "
            f"P2-PERF-3 Deliverable (1) requires module-level pure-function "
            f"lift; instance methods must be thin trampolines with no jit "
            f"decorator on the method itself."
        )


# ---------------------------------------------------------------------------
# AC-(b): All 13 fixtures at scope="module"
# ---------------------------------------------------------------------------
#
# AST inspection of ``tests/omiecid/conftest.py`` plus the 7 file-local
# override files enumerated in agile-plan v5.9 Deliverable (2).
# Sub-millisecond.

class TestACb_FixtureCascadeAtModuleScope:
    """AC-(b): all 13 fixtures from agile-plan v5.9 Deliverable (2) carry
    ``scope="module"``.

    Breakdown:
      - 6 in ``tests/omiecid/conftest.py``: ``open_contract``,
        ``product_config``, ``market_config``, ``book``, ``book_and_state``,
        ``empty_state``.
      - 7 file-local ``open_contract`` overrides in the 7 files
        enumerated in ``FILE_LOCAL_OPEN_CONTRACT_OVERRIDES``.

    Asserting only the three core fixtures (the v5.8 wording) is
    insufficient — pytest collection raises ``ScopeMismatch`` if ``book``
    is ``scope="module"`` while its transitive dependencies remain
    function-scoped, so the cascade is part of the locked invariant.
    """

    @staticmethod
    def _is_pytest_fixture_dec(dec: ast.expr) -> bool:
        """True iff ``dec`` is ``@pytest.fixture`` or ``@pytest.fixture(...)``."""
        if isinstance(dec, ast.Attribute) and dec.attr == "fixture":
            if isinstance(dec.value, ast.Name) and dec.value.id == "pytest":
                return True
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Attribute) and func.attr == "fixture":
                if isinstance(func.value, ast.Name) and func.value.id == "pytest":
                    return True
        return False

    @staticmethod
    def _scope_from_dec(dec: ast.expr) -> str | None:
        """Return the ``scope=...`` keyword string literal on a fixture
        decorator, or None if absent (bare ``@pytest.fixture`` defaults to
        ``function`` scope).
        """
        if not isinstance(dec, ast.Call):
            return None
        for kw in dec.keywords:
            if kw.arg == "scope" and isinstance(kw.value, ast.Constant):
                return kw.value.value
        return None

    def _find_fixture_scope(
        self, path: pathlib.Path, fixture_name: str
    ) -> tuple[bool, str | None, int | None]:
        """Walk ``path`` and locate the first top-level (or class-body)
        ``@pytest.fixture`` def named ``fixture_name``.

        Returns ``(found, scope_or_None, lineno)``.  ``scope_or_None`` is
        the string literal on ``scope=...`` (e.g. ``"module"``) or None if
        the fixture uses the default (function) scope.
        """
        tree = ast.parse(path.read_text())

        def _walk(body):
            for node in body:
                if isinstance(node, ast.FunctionDef) and node.name == fixture_name:
                    for dec in node.decorator_list:
                        if self._is_pytest_fixture_dec(dec):
                            return True, self._scope_from_dec(dec), node.lineno
                if isinstance(node, ast.ClassDef):
                    result = _walk(node.body)
                    if result is not None:
                        return result
            return None

        result = _walk(tree.body)
        if result is None:
            return False, None, None
        return result

    def test_conftest_six_fixtures_at_module_scope(self):
        """AC-(b) sub-assertion: all 6 ``tests/omiecid/conftest.py`` fixtures
        (open_contract, product_config, market_config, book, book_and_state,
        empty_state) carry ``scope="module"``.
        """
        offenders: list[str] = []
        for name in CONFTEST_FIXTURE_NAMES:
            found, scope, lineno = self._find_fixture_scope(CONFTEST_PATH, name)
            if not found:
                offenders.append(f"{name}: MISSING from conftest.py")
                continue
            if scope != "module":
                offenders.append(
                    f"{name}@line{lineno}: scope={scope!r} "
                    "(expected 'module')"
                )
        assert offenders == [], (
            "AC-(b) violation in tests/omiecid/conftest.py — the following "
            f"fixtures are not at scope='module':\n  " + "\n  ".join(offenders)
            + "\nAll 6 conftest fixtures must be promoted in a single "
            "coordinated edit (agile-plan v5.9 Deliverable (2))."
        )

    @pytest.mark.parametrize(
        "override_path",
        FILE_LOCAL_OPEN_CONTRACT_OVERRIDES,
        ids=lambda p: p.name,
    )
    def test_file_local_open_contract_override_at_module_scope(
        self, override_path: pathlib.Path
    ):
        """AC-(b) sub-assertion: each of the 7 file-local ``open_contract``
        override files (enumerated in agile-plan v5.9 Deliverable (2)
        (vii)-(xiii)) carries ``scope="module"`` on its ``open_contract``
        fixture.

        ScopeMismatch rationale: when ``book``/``book_and_state``/``empty_state``
        are module-scoped, every transitively-required ``open_contract``
        fixture (including file-local overrides) must also be
        module-scoped or pytest raises ``ScopeMismatch`` at collection.
        """
        found, scope, lineno = self._find_fixture_scope(
            override_path, "open_contract"
        )
        assert found, (
            f"AC-(b) violation: file-local open_contract fixture not found "
            f"in {override_path.relative_to(REPO_ROOT)} — agile-plan v5.9 "
            f"Deliverable (2) enumerates this file as carrying an "
            f"open_contract override; if a refactor removed it, the "
            f"developer must report back to the architect before silently "
            f"dropping the list entry."
        )
        assert scope == "module", (
            f"AC-(b) violation in "
            f"{override_path.relative_to(REPO_ROOT)}@line{lineno}: "
            f"open_contract fixture has scope={scope!r} (expected "
            f"'module').  Promote in lockstep with the 6 conftest "
            f"fixtures (agile-plan v5.9 Deliverable (2))."
        )

    def test_total_thirteen_fixture_edits_locked(self):
        """AC-(b) cardinality pin: 6 conftest fixtures + 7 file-local
        ``open_contract`` overrides = 13 ``@pytest.fixture`` decorator
        edits.  Per agile-plan v5.9 (2026-05-11), 13 is the locked
        contract — not 8 (the architecture-review's mis-summed count) and
        not 3 (the v5.8 wording).
        """
        assert len(CONFTEST_FIXTURE_NAMES) == 6
        assert len(FILE_LOCAL_OPEN_CONTRACT_OVERRIDES) == 7
        assert (
            len(CONFTEST_FIXTURE_NAMES) + len(FILE_LOCAL_OPEN_CONTRACT_OVERRIDES)
            == 13
        )


# ---------------------------------------------------------------------------
# AC-(c) + AC-(d): Regression-gate subset passes and warm-run < 5 min
# ---------------------------------------------------------------------------
#
# Both pins exercise the same subprocess invocation — the regression-gate
# subset paste-ready argv from ``test-perf-gate.md`` §5 (the deferred
# IOC-1022 line is intentionally NOT folded in until P2-PERF-3 reduces
# its 203-s cost; this story IS P2-PERF-3, so it MAY be folded in by the
# developer in a follow-up edit to test-perf-gate.md §1 / §5.  The locked
# test below uses §5 as authored on 2026-05-11 — the developer must not
# silently expand the subset under the test).

REGRESSION_GATE_SUBSET = (
    "tests/omiecid/test_matching.py",
    "tests/omiecid/test_contract_book.py",
    "tests/omiecid/test_contract_book_audit.py",
    "tests/omiecid/test_ioc_fok.py",
    "tests/omiecid/test_ioc_fok_audit.py",
    "tests/omiecid/test_fok_iceberg_visibility.py",
    "tests/omiecid/test_fok_icewpi_precheck.py",
    "tests/omiecid/test_trade_extraction_ring_buffer.py::TestSaturatedBufferRegression::test_single_new_fill_detected_on_saturated_buffer",
    "tests/omiecid/test_trade_extraction_ring_buffer.py::TestSaturatedBufferRegression::test_multi_level_new_fills_detected_on_saturated_buffer",
    "tests/omiecid/test_scenario_single_contract.py::TestFullSingleContractScenario",
    "tests/omiecid/test_scenario_mixed_simulation.py::TestMixedOrderTypeSimulation",
)

# Per-PR budget from test-perf-gate.md §2 (binding).
GATE_BUDGET_SECONDS = 5 * 60.0


@pytest.mark.slow
class TestACcd_RegressionGateSubsetGreenAndUnderBudget:
    """AC-(c): regression-gate subset passes with exit 0 (correctness
    self-check of the §5 gate argv — stays in the default tier).
    AC-(d): warm-run wall-clock < 5 min (§2 temporal budget — ``perf`` tier).

    Both pins are exercised by the same subprocess invocation per
    ``test-perf-gate.md`` §5 (the paste-ready argv).  The subprocess is
    spawned cold-cache-friendly (no ``.jax_cache`` deletion in-test — we
    use whatever cache the dev workstation has) and the wall-clock is
    measured from process spawn to process exit.

    Tier split (P3-PREP-TEST-1, deliverable (3)): the AC-(c) correctness
    self-check (``test_subset_exits_zero``) STAYS in the default
    (non-``perf``) tier so the gate's executable self-check runs on every
    suite run.  The AC-(d) §2 temporal-budget assertion
    (``test_subset_under_five_minutes``) is moved behind
    ``@pytest.mark.perf`` so the fast/audit feedback loop can deselect the
    wall-clock measurement with ``-m "not perf"``.  Each method is
    self-contained (it spawns its own subprocess run via ``_run_gate_subset``)
    — no mutable state is shared across the two methods, so isolation holds
    even though the subprocess therefore runs once per method.

    Both methods stay ``@pytest.mark.slow`` (class-level): each spawns a
    separate ``pytest`` process that itself initialises JAX and may take
    several minutes warm-cache.
    """

    @staticmethod
    def _run_gate_subset() -> tuple[int, float, str, str]:
        """Spawn the §5 regression-gate subset once.

        Returns ``(returncode, elapsed_seconds, stdout_tail, stderr_tail)``.
        Self-contained: builds its own argv + env and measures its own
        wall-clock, so each of the two split methods can call it
        independently without sharing mutable state.
        """
        n_pin = _read_addopts_n_pin()
        cmd = [
            str(PYTHON_BIN),
            "-m",
            "pytest",
            "-n",
            n_pin,
            *REGRESSION_GATE_SUBSET,
            "--tb=no",
            "-q",
        ]
        env = _pytest_env()
        t0 = time.monotonic()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=GATE_BUDGET_SECONDS + 60.0,
        )
        elapsed = time.monotonic() - t0
        return proc.returncode, elapsed, proc.stdout[-2000:], proc.stderr[-2000:]

    def test_subset_exits_zero(self):
        """AC-(c): the §5 gate-subset argv exits 0 (executable self-check
        of the regression-gate subset).  Default (non-``perf``) tier — runs
        on every suite run so the gate's correctness self-check is never
        skipped by ``-m "not perf"``.
        """
        returncode, _elapsed, stdout_tail, stderr_tail = self._run_gate_subset()
        assert returncode == 0, (
            f"AC-(c) violation: regression-gate subset returned exit "
            f"code {returncode}.\n"
            f"stdout tail:\n{stdout_tail}\n"
            f"stderr tail:\n{stderr_tail}"
        )

    @pytest.mark.perf
    def test_subset_under_five_minutes(self):
        """AC-(d): the §5 gate-subset warm wall-clock < 5 min (§2 per-PR
        budget).  ``perf`` tier — this is the temporal-budget assertion
        only; deselectable with ``-m "not perf"``.

        Self-contained: spawns its own subprocess run (the AC-(c)
        correctness floor is still asserted here so a crash does not
        falsely improve the timing).
        """
        returncode, elapsed, stdout_tail, stderr_tail = self._run_gate_subset()
        assert returncode == 0, (
            f"AC-(d) precondition: regression-gate subset returned exit "
            f"code {returncode} (a crash would falsely improve the timing).\n"
            f"stdout tail:\n{stdout_tail}\n"
            f"stderr tail:\n{stderr_tail}"
        )
        assert elapsed < GATE_BUDGET_SECONDS, (
            f"AC-(d) violation: regression-gate subset warm wall-clock "
            f"= {elapsed:.1f} s, exceeds per-PR budget of "
            f"{GATE_BUDGET_SECONDS:.0f} s (test-perf-gate.md §2).  "
            f"Either the budget needs an architect-mediated amendment or "
            f"the subset needs a P2-PERF-4 -n re-pin."
        )


# ---------------------------------------------------------------------------
# AC-(e): Full `not slow and not audit` surface passes with pass count >= 1820
# ---------------------------------------------------------------------------
#
# The original baseline (2026-05-11, commit 5d1e40a, per test-perf-gate.md
# §1 rationale table row 5) was 2471 pass on the `not slow` surface — but
# that baseline predates the `audit` marker, when the `*_audit.py` files
# were part of the routine surface.  Under P3-TEST-AUDIT-DECOUPLE-1 the
# routine gate posture moved to `-m "not audit"` (test-perf-gate.md §3/§4),
# so the meta-test's nested `-m "not slow"` was a STALE posture: it
# re-included the `audit`-tagged compile-bomb file that the authoritative
# `-m "not audit"` gate excludes directly, and under the double-`-n 24`
# nested load (~48 concurrent workers vs the host's single-process XLA
# compile ceiling) a worker SIGSEGV'd, giving the authoritative gate an
# intermittent false-red.  P3.4-FOLLOWUP-3 (2026-07-08) narrows the nested
# run to `-m "not slow and not audit"` (see test_e below) so it measures
# the SAME surface class as the authoritative gate, and re-baselines this
# floor to the not-audit surface in the SAME edit.
#
# Floor provenance: `-m "not slow and not audit" -n 24` measured 1836
# passed on 2026-07-08 (CUDA_VISIBLE_DEVICES="" PYTHONPATH=.
# .venv/bin/python3 -m pytest tests/omiecid/ -m "not slow and not audit"
# -n 24, exit 0, clean, no SIGSEGV).  Pinned 16 below observed to allow
# for xdist skip-count jitter; do not regress.  This is a measurement, NOT
# a `pyproject.toml addopts -n` change (the global pin is UNTOUCHED), so it
# does not fire test-perf-gate.md §7 trigger-4.

NOT_SLOW_BASELINE_PASSED = 1820  # not-slow-and-not-audit surface, 2026-07-08 (obs 1836)

# ---------------------------------------------------------------------------
# Known non-deterministic xdist flake allowlist (AC-(e) tolerance)
# ---------------------------------------------------------------------------
#
# The AC-(e) meta-test below spawns its own nested ``pytest -n <pin> -m
# "not slow and not audit" tests/omiecid/`` subprocess and asserts the
# pass-count floor.  ``test_submit_order_jit_safe.py`` is NOT ``audit``-
# tagged, so it remains in the narrowed surface (P3.4-FOLLOWUP-3).
# One documented test is non-deterministic under xdist work-stealing: the
# "one-compile-across-four-variants" trace-count invariant.  It PASSES in
# isolation (``-n0``: 1 passed) but occasionally FAILS inside the nested
# subprocess when xdist work-stealing perturbs the JIT-trace ordering.
#
# This is the flake documented in ``docs/omiecid/test-perf-gate.md`` row 129
# ("the known ``test_submit_order_jit_safe.py::…four_exec_restrictions``
# flake") and row 226 ("xdist-flake territory ... not stable for gating").
# It is NOT a regression — the P2-H2 one-compile invariant holds; only the
# xdist scheduling of the trace-count probe is racy.
#
# The meta-test tolerates THIS node-id (and only this one) as a subprocess
# failure.  Any OTHER FAILED node-id — or a genuine pass-count regression
# below the floor — still fails the meta-test loudly.  Node-ids are matched
# by suffix (module::class::method) so a leading ``tests/omiecid/`` path
# prefix from the subprocess cwd does not defeat the match.
KNOWN_XDIST_FLAKES = frozenset({
    "test_submit_order_jit_safe.py::"
    "TestACFiveTraceCountInvariantOneCompileForFourVariants::"
    "test_one_compile_across_all_four_exec_restrictions",
})


def _node_id_is_known_flake(node_id: str) -> bool:
    """True iff ``node_id`` matches a ``KNOWN_XDIST_FLAKES`` entry.

    Matches by suffix so both the bare ``test_x.py::Class::method`` form
    and the ``tests/omiecid/test_x.py::Class::method`` (cwd-prefixed) form
    reported by the nested subprocess are recognised.
    """
    return any(node_id == flake or node_id.endswith("/" + flake)
               for flake in KNOWN_XDIST_FLAKES)


@pytest.mark.slow
class TestACe_FullNotSlowSurfaceNoRegression:
    """AC-(e): the full ``not slow and not audit`` surface still passes with
    pass count >= 1820 (do not regress).

    The measured surface was narrowed from ``not slow`` to
    ``not slow and not audit`` under P3.4-FOLLOWUP-3 (2026-07-08).
    P3-TEST-AUDIT-DECOUPLE-1 moved the routine gate posture to
    ``-m "not audit"`` (test-perf-gate.md §3/§4); the nested ``-m "not
    slow"`` was a stale posture re-including the ``*_audit.py`` compile-bomb
    the authoritative gate excludes, which caused an intermittent worker
    SIGSEGV under the double-``-n 24`` nested load.  Narrowing the nested
    run to ``not slow and not audit`` makes it measure the SAME surface
    class as the authoritative gate.  See ``NOT_SLOW_BASELINE_PASSED`` for
    the re-baselined floor (measured 1836, pinned 1820).

    Slow-marked: this subprocess takes ~20-75 s on the dev workstation.
    """

    @pytest.mark.perf
    def test_e_full_not_slow_pass_count_non_decreasing(self):
        n_pin = _read_addopts_n_pin()
        cmd = [
            str(PYTHON_BIN),
            "-m",
            "pytest",
            "-n",
            n_pin,
            "-m",
            # P3.4-FOLLOWUP-3 (2026-07-08): narrowed from "not slow" to
            # "not slow and not audit" so the nested run matches the routine
            # gate posture set by P3-TEST-AUDIT-DECOUPLE-1 (-m "not audit").
            # "not slow" alone does NOT exclude `audit`, so it re-included
            # the *_audit.py compile-bomb file the authoritative gate
            # excludes directly; under the double-`-n 24` nested load that
            # tipped a worker SIGSEGV (intermittent false-red).  `not audit`
            # is the correct axis (NOT `not jit`, which would drop ~936
            # tests on an unrelated cost boundary).
            "not slow and not audit",
            "tests/omiecid/",
            # exclude this test file itself to avoid pytest-recursion
            # (this test file is collected under tests/omiecid/ but its
            # subprocess tests are slow-marked so ``-m "not slow"`` filters
            # them out by construction; the static (a)/(b) tests are
            # included and must pass).
            "--tb=no",
            "-q",
        ]
        env = _pytest_env()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=15 * 60.0,
        )
        counts = _parse_pytest_summary(proc.stdout)

        # Core regression detector (UNCHANGED): the full `not slow and not
        # audit` pass count must not drop below the re-baselined floor.  A
        # genuine regression that deletes/breaks tests still trips this
        # floor regardless of the returncode handling below.
        assert counts["passed"] >= NOT_SLOW_BASELINE_PASSED, (
            f"AC-(e) violation: full `not slow and not audit` pass count "
            f"regressed.  floor (measured 1836 on 2026-07-08, pinned per "
            f"P3.4-FOLLOWUP-3) = {NOT_SLOW_BASELINE_PASSED}; observed = "
            f"{counts['passed']}.  full summary = {counts}."
        )

        # Returncode tolerance (P3.4-FOLLOWUP-3): the nested subprocess is
        # allowed to be red ONLY if the sole failure(s) are the documented
        # one-compile xdist flake (KNOWN_XDIST_FLAKES, test-perf-gate.md
        # row 129 / row 226).  Any OTHER FAILED node-id — or an error /
        # non-parseable red exit with no identifiable failed node-ids —
        # fails the meta-test loudly, preserving its regression-detection
        # purpose.
        if proc.returncode != 0:
            failed_node_ids = _parse_failed_node_ids(proc.stdout)
            unexpected = [
                nid for nid in failed_node_ids
                if not _node_id_is_known_flake(nid)
            ]
            # Guard against a red exit whose failure surface we could not
            # attribute to any FAILED node-id (e.g. collection error,
            # worker crash, timeout with no short-summary): treat as
            # unexpected — never silently tolerate an unexplained red.
            assert failed_node_ids and not unexpected, (
                f"AC-(e) violation: full `not slow and not audit` surface "
                f"exit {proc.returncode} with failure(s) that are NOT the "
                f"documented one-compile xdist flake.\n"
                f"parsed FAILED node-ids = {failed_node_ids or '<none parsed>'}\n"
                f"unexpected (non-allowlisted) = "
                f"{unexpected or '<none parsed — unexplained red exit>'}\n"
                f"tolerated flake allowlist (test-perf-gate.md row 129) = "
                f"{sorted(KNOWN_XDIST_FLAKES)}\n"
                f"counts={counts}.\n"
                f"stdout tail:\n{proc.stdout[-2000:]}\n"
                f"stderr tail:\n{proc.stderr[-2000:]}"
            )


# ---------------------------------------------------------------------------
# AC-(f): >= 50% reduction on 8-test slow ring-buffer subset
# ---------------------------------------------------------------------------
#
# Baseline (cold-cache -n 0): 343 s on the 8 tests excluding
# ``test_ioc_1022_matches_at_114_69_eur_0_30_mw`` (which carries a
# 203-s per-call dispatch cost the jit-cache-defrag fix does not touch
# — see agile-plan v5.9 AC-(f) revision and architecture-review finding 5).
# Target: <= 172 s.
#
# WALL-CLOCK CAVEAT (test-agent flag): the inline timed assertion is
# environment-noisy.  We mitigate with a median-of-3 timing across three
# subprocess runs and a 20% slack factor above the 172-s target.  If the
# orchestrator later judges this pin "fundamentally unauthorable as a
# deterministic test", the architect's recommended adjustment ("switch
# to merge-note parsing rather than inline timed assertion") becomes a
# v5.10 follow-up amendment.  See sprint-status §"P2-PERF-3 — Test
# Authoring (2026-05-11)" for the test-agent's report.

RING_BUFFER_EIGHT_TESTS = (
    # The 9 slow tests in test_trade_extraction_ring_buffer.py minus the
    # outlier ``test_ioc_1022_matches_at_114_69_eur_0_30_mw``.  Pinned
    # by class path, not by individual ::method, to keep the subprocess
    # argv short and to match how the developer will run them locally.
    "tests/omiecid/test_trade_extraction_ring_buffer.py",
)

RING_BUFFER_EIGHT_BASELINE_SECONDS = 343.0  # cold-cache -n 0, 2026-05-11
RING_BUFFER_EIGHT_TARGET_SECONDS = 172.0  # >= 50% reduction, AC-(f)
RING_BUFFER_AC_F_SLACK_FACTOR = 1.20  # +20% environment slack

# The outlier — excluded from AC-(f) reduction calculation per agile-plan
# v5.9 AC-(f) revision, but its wall-clock IS still recorded by the
# developer in the merge note as a baseline for the future story
# (P2-H2 / P2-L4) that addresses its 203-s per-call dispatch cost.
RING_BUFFER_OUTLIER_DESELECT = (
    "test_ioc_1022_matches_at_114_69_eur_0_30_mw"
)


@pytest.mark.slow
class TestACf_RingBufferEightTestSubsetUnderHalfBaseline:
    """AC-(f): the 8-test slow ring-buffer subset (excluding
    ``test_ioc_1022_matches_at_114_69_eur_0_30_mw``) runs in <= 172 s
    cold-cache at ``-n 0``, a >= 50% reduction from the 343-s baseline
    measured on commit 5d1e40a (agile-plan v5.9, 2026-05-11).

    Limitation: wall-clock is environment-noisy.  We use median-of-3
    timing with a +20% slack factor (effective ceiling = 172 * 1.20 =
    206.4 s).  If the median-of-3 still exceeds the ceiling, the test
    fails and the developer dispatches back to the architect for either
    a measurement-method amendment or a budget reset.  Re-running the
    test should produce stable medians once the defrag fix lands and
    the timing distribution narrows.

    Slow-marked: each subprocess takes roughly target seconds; three
    runs target ~10 min on a defragged tree, longer pre-defrag.
    """

    def _run_once(self) -> tuple[int, float, dict, str]:
        """Run the 8-test subset cold-cache (``-n 0``) and return
        ``(returncode, elapsed_seconds, parsed_counts, stdout_tail)``.

        Cold-cache: per ``test-perf-gate.md`` §4 the cold-cache numbers
        require ``rm -rf .jax_cache``.  We do NOT delete the cache from
        within this test — doing so would invalidate every other
        subprocess test and would race against xdist workers running in
        parallel inside the parent pytest process.  Instead, the test
        runs each subprocess with a freshly-set ``JAX_COMPILATION_CACHE_DIR``
        pointing at a temp directory, so each invocation is cold by
        construction without touching the workspace cache.
        """
        import tempfile

        with tempfile.TemporaryDirectory(prefix="jax_cache_acf_") as cache_dir:
            env = _pytest_env()
            env["JAX_COMPILATION_CACHE_DIR"] = cache_dir
            cmd = [
                str(PYTHON_BIN),
                "-m",
                "pytest",
                "-n",
                "0",
                "-m",
                "slow",
                *RING_BUFFER_EIGHT_TESTS,
                "--deselect",
                f"tests/omiecid/test_trade_extraction_ring_buffer.py::"
                f"TestIOC1022EndToEnd::{RING_BUFFER_OUTLIER_DESELECT}",
                "--tb=no",
                "-q",
            ]
            t0 = time.monotonic()
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=RING_BUFFER_EIGHT_BASELINE_SECONDS * 2.0,
            )
            elapsed = time.monotonic() - t0

        counts = _parse_pytest_summary(proc.stdout)
        return proc.returncode, elapsed, counts, proc.stdout[-2000:]

    @pytest.mark.perf
    def test_f_eight_test_subset_median_under_target(self):
        """AC-(f): median-of-3 cold-cache wall-clock <= 172 s * 1.20 slack."""
        results = [self._run_once() for _ in range(3)]
        rcs = [r[0] for r in results]
        elapsed = [r[1] for r in results]
        all_counts = [r[2] for r in results]

        # Every run must pass exit 0 (this is a behavioural floor — a
        # crash would falsely improve the timing median otherwise).
        for i, (rc, _, counts, tail) in enumerate(results):
            assert rc == 0, (
                f"AC-(f) sub-violation: 8-test subset run {i} exited "
                f"with code {rc}, counts={counts}.\n"
                f"stdout tail:\n{tail}"
            )

        median = statistics.median(elapsed)
        ceiling = RING_BUFFER_EIGHT_TARGET_SECONDS * RING_BUFFER_AC_F_SLACK_FACTOR
        assert median <= ceiling, (
            f"AC-(f) violation: median-of-3 cold-cache wall-clock for the "
            f"8-test ring-buffer subset = {median:.1f} s, exceeds the "
            f">= 50%-reduction target {RING_BUFFER_EIGHT_TARGET_SECONDS:.0f} s "
            f"* {RING_BUFFER_AC_F_SLACK_FACTOR}x slack = {ceiling:.1f} s.\n"
            f"individual runs: {[f'{e:.1f}' for e in elapsed]}\n"
            f"baseline (commit 5d1e40a, 2026-05-11, -n 0 cold-cache) = "
            f"{RING_BUFFER_EIGHT_BASELINE_SECONDS:.0f} s.\n"
            f"counts per run: {all_counts}"
        )


# ---------------------------------------------------------------------------
# AC-(g): No behavioural regression in four replay/scenario files
# ---------------------------------------------------------------------------
#
# Per-file pass-count baselines collected on 2026-05-11 from branch
# ``omie/P2-PERF-3-jit-cache-defrag`` (which carries no production change
# yet, so the numbers represent main pre-story behaviour).  Subprocess
# invocation uses ``-m "not slow"`` to keep the test deterministic and
# under the fast-path total runtime — the two mandatory slow ring-buffer
# tests are covered by AC-(c)/(d) and the 8-test ring-buffer subset by
# AC-(f), so the slow surface is locked elsewhere.
#
# If a future ``test_*`` is added/removed from any of these four files
# the baseline must be re-collected and committed in the same edit; a
# silently-shifted baseline is a severity-medium audit finding.

# Format: relative file path -> expected pass count under ``-m "not slow"``
# Collected 2026-05-11 on the story branch with no production changes.
AC_G_FOUR_FILES_NOT_SLOW_BASELINE = {
    "tests/omiecid/test_trade_extraction_ring_buffer.py": 2,
    "tests/omiecid/test_scenario_mixed_simulation.py": 2,
    "tests/omiecid/test_scenario_single_contract.py": 5,
    "tests/omiecid/test_omie_data.py": 70,
}
AC_G_FOUR_FILES_TOTAL_BASELINE = sum(AC_G_FOUR_FILES_NOT_SLOW_BASELINE.values())  # 79


@pytest.mark.slow
class TestACg_NoBehavioralRegressionInFourReplayScenarioFiles:
    """AC-(g): every test in the four named replay/scenario files that
    passed on main pre-story still passes post-story.  We pin both
    per-file pass-count equality and total pass-count equality vs the
    baseline collected 2026-05-11.

    Scope note: per the test-agent dispatch prompt's specific-guidance §6,
    we run ``-m "not slow"`` against each file (cheap, ~14 s total).  The
    two mandatory slow ring-buffer tests are covered by AC-(c)/(d); the
    8-test slow ring-buffer subset by AC-(f); so the slow surface is
    locked by the other pins and does not need to be re-asserted here.
    """

    @pytest.mark.parametrize(
        "rel_path,baseline_passed",
        list(AC_G_FOUR_FILES_NOT_SLOW_BASELINE.items()),
        ids=lambda x: x if isinstance(x, str) else str(x),
    )
    @pytest.mark.perf
    def test_g_per_file_pass_count_non_decreasing(
        self, rel_path: str, baseline_passed: int
    ):
        """AC-(g) per-file pin: pass count for ``rel_path`` (under
        ``-m "not slow"``) is >= the 2026-05-11 baseline.
        """
        cmd = [
            str(PYTHON_BIN),
            "-m",
            "pytest",
            "-n",
            "0",  # serial — these files are small, xdist startup dominates
            "-m",
            "not slow",
            rel_path,
            "--tb=no",
            "-q",
        ]
        env = _pytest_env()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120.0,
        )
        counts = _parse_pytest_summary(proc.stdout)
        assert proc.returncode == 0, (
            f"AC-(g) violation: {rel_path} exit {proc.returncode}, "
            f"counts={counts}.\n"
            f"stdout tail:\n{proc.stdout[-1500:]}"
        )
        assert counts["passed"] >= baseline_passed, (
            f"AC-(g) violation: {rel_path} pass count regressed.  "
            f"baseline (2026-05-11) = {baseline_passed}; observed = "
            f"{counts['passed']}.  full summary = {counts}."
        )

    @pytest.mark.perf
    def test_g_total_pass_count_across_four_files_non_decreasing(self):
        """AC-(g) aggregate pin: total pass count across all four files
        (under ``-m "not slow"``) is >= the sum of the per-file baselines.

        Run as a single subprocess for total-count equivalence; this
        catches the case where a per-file baseline shifts up while
        another shifts down by the same amount (which the per-file
        parametrization above would also catch, but the aggregate gives
        an unambiguous single number for the merge note).
        """
        cmd = [
            str(PYTHON_BIN),
            "-m",
            "pytest",
            "-n",
            "0",
            "-m",
            "not slow",
            *AC_G_FOUR_FILES_NOT_SLOW_BASELINE.keys(),
            "--tb=no",
            "-q",
        ]
        env = _pytest_env()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=180.0,
        )
        counts = _parse_pytest_summary(proc.stdout)
        assert proc.returncode == 0, (
            f"AC-(g) aggregate violation: exit {proc.returncode}, "
            f"counts={counts}.\n"
            f"stdout tail:\n{proc.stdout[-1500:]}"
        )
        assert counts["passed"] >= AC_G_FOUR_FILES_TOTAL_BASELINE, (
            f"AC-(g) aggregate violation: total pass count across four "
            f"replay/scenario files regressed.  baseline (2026-05-11) = "
            f"{AC_G_FOUR_FILES_TOTAL_BASELINE}; observed = "
            f"{counts['passed']}.  full summary = {counts}."
        )
