"""Shared pytest fixtures for tests/omiecid/.

Provides the four standard fixtures used across most OMIE test files:
  - product_config  — default OMIEProductConfig()
  - market_config   — default OMIEMarketConfig()
  - open_contract   — Contract 37 (09:15-09:30), OPEN status
  - book_and_state  — (ContractBook, initial OMIEBookState) tuple from the above

Derivative convenience fixtures also provided:
  - book        — ContractBook(open_contract, product_config, market_config)
  - empty_state — book.init()  (returns OMIEBookState)

Test files that need a *different* contract (e.g. contract_id=10 or 42)
override open_contract locally; pytest's scoping rules ensure the local
fixture wins.

This conftest also installs a session-scoped JAX persistent compilation
cache hook (``pytest_configure``) that points the JAX cache at a
workspace-local ``.jax_cache/`` directory. See the docstring on that
function for the xdist concurrency model and the threshold-defaults
gotcha.
"""

import logging
import os


# ---------------------------------------------------------------------------
# cuInit / CUDA_ERROR_NO_DEVICE plugin-discovery noise suppression
# (P3-PREP-HARNESS-1, H-2; 2026-06-16)
#
# MUST run at conftest-import time, BEFORE the OMIE module imports below — the
# `from omiecid_simulator.omiecid.contract_book import ContractBook` import chain
# transitively initializes the JAX backend (proven: the `cpu` backend is
# already live after that import), which is what fires the plugin-discovery
# noise. `pytest_configure` runs too late to catch that first discovery in the
# xdist controller. So the helper is defined and CALLED here, ahead of the
# imports. `pytest_configure` calls it again (idempotent) to cover xdist
# workers whose conftest import order differs.
# ---------------------------------------------------------------------------

def _silence_cuinit_plugin_noise() -> None:
    """Silence the harmless CUDA-plugin discovery traceback on CPU-hermetic runs.

    Under the CPU-hermetic harness convention (``CUDA_VISIBLE_DEVICES=""`` —
    see ``docs/omiecid/device-policy.md`` §"JAX_PLATFORMS vs ...") the installed
    ``jax_plugins.xla_cuda12`` plugin's ``initialize()`` raises while probing
    for a device, and JAX's plugin-discovery loop logs the failure via
    ``logger.exception(...)`` on the ``jax._src.xla_bridge`` logger. This emits
    a multi-line ::

        ERROR:jax._src.xla_bridge:... Jax plugin configuration error: Exception
          when calling jax_plugins.xla_cuda12.initialize()
        Traceback (most recent call last):
          ...
        RuntimeError: ... operation cuInit(0) failed: CUDA_ERROR_NO_DEVICE
        WARNING:jax._src.xla_bridge:... An NVIDIA GPU may be present ... Falling
          back to cpu.

    JAX then recovers correctly to the CPU backend and the suite is green —
    this is pure log noise that can be mistaken for a real failure during a
    perf-gate run.

    NARROWNESS (HARD CONSTRAINT — see P3-PREP-HARNESS-1 deliverable (b)):
      * We raise the level of EXACTLY ONE logger — ``jax._src.xla_bridge`` —
        to ``CRITICAL``, so only that logger's ERROR (the cuInit traceback)
        and WARNING (the "Falling back to cpu" line) records are dropped. A
        genuine ``CRITICAL`` on the same logger STILL propagates, so a real
        catastrophic CUDA/JAX init failure is NOT masked. (The cuInit
        traceback is attached via ``exc_info`` on a ``logger.exception(...)``
        ERROR record — its text is not in ``record.getMessage()`` — so a
        message-substring ``logging.Filter`` does NOT catch it; raising the
        logger's threshold is the mechanism that works.)
      * It is gated on ``CUDA_VISIBLE_DEVICES == ""`` (the CPU-hermetic path),
        so it is a no-op on any CUDA-visible host.
      * It lives ONLY in ``tests/omiecid/conftest.py`` and so applies ONLY to the
        ``tests/omiecid/`` collection. It does NOT touch the root logger, does NOT
        call ``logging.disable``, and adds NO stderr/warnings filter — so it
        CANNOT mask a real CUDA/JAX error in the GPU RL-training process
        (``omiecid_simulator/jaxrl/MARL/``), which keeps the GPU default per
        ``docs/omiecid/device-policy.md``, runs in a different process, and does
        not set ``CUDA_VISIBLE_DEVICES=""``.

    Sequencing: the JAX backend is initialized as a side effect of the OMIE
    module imports at the top of this conftest (the ``contract_book`` import
    chain), which is when the plugin-discovery noise fires. This helper is
    therefore called at conftest-import time *before* those imports (see the
    module-level call below); ``pytest_configure`` calls it again so xdist
    workers are covered regardless of conftest import ordering. The call is
    idempotent (it only sets a logger level).
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        return
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)


# Raise the level BEFORE the OMIE imports trigger JAX backend initialization.
_silence_cuinit_plugin_noise()


import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig
from omiecid_simulator.omiecid.contract_book import ContractBook


# ---------------------------------------------------------------------------
# JAX persistent compilation cache (Story T1.1)
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Enable the JAX persistent compilation cache for the OMIE test suite.

    Three knobs are set, in order:

      1. ``jax_compilation_cache_dir = ".jax_cache"`` — workspace-local
         directory where JAX writes its cached compile artifacts (HLO +
         lowered XLA modules).  ``.jax_cache/`` is gitignored.

      2. ``jax_persistent_cache_min_entry_size_bytes = -1`` — disable the
         minimum-entry-size threshold (default skips small cache entries).

      3. ``jax_persistent_cache_min_compile_time_secs = 0.0`` — disable
         the minimum-compile-time threshold (default skips compiles that
         finish below ~1 s).

    Why the aggressive (-1, 0.0) thresholds?
      The OMIE test suite consists overwhelmingly of *leaf-module*
      compiles (validate_order, compute_trade_value, extract_fills,
      register / regenerate_slices, _status_after_*) that are each
      individually fast (<1 s).  Under JAX's default thresholds these
      compiles fall *below* both the size and the time floor and so are
      never persisted to disk — meaning the warm-cache speedup we are
      trying to achieve evaporates.  Pinning both knobs to their
      "always-cache" sentinel values forces every compile to persist,
      which is exactly the behaviour we want for a test suite where
      cumulative compile cost (not single-call cost) dominates wall-clock.

    xdist concurrency model:
      JAX writes cache entries via temp-file-plus-atomic-rename
      (per the JAX persistent-cache documentation).  Multiple xdist
      workers compiling the same key produce byte-identical content
      (the cache key is content-addressed over HLO + jaxlib version +
      compile options) and the last writer wins safely on the
      ``rename(2)`` boundary.  This is process-safe on POSIX and on
      Windows ``ReplaceFileW`` semantics.  No file-locking is required
      and no explicit serialisation among workers is needed.

    Failure mode:
      If ``jax`` is not importable for any reason (e.g. a non-JAX-only
      pytest invocation), the hook is a silent no-op and pytest startup
      proceeds normally.  This keeps the default ``pytest -m "not jit"``
      fast-path JAX-import-free if a future caller wants it.
    """
    # Belt-and-braces re-assertion of the cuInit-noise suppression for xdist
    # workers (the primary suppression is the module-level call above, which
    # runs before the OMIE imports init the JAX backend). Idempotent. See the
    # helper docstring for the narrowness rationale (P3-PREP-HARNESS-1, H-2).
    _silence_cuinit_plugin_noise()

    try:
        import jax
    except ImportError:
        return

    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    # Device policy (P3-OPS-01a, 2026-05-28): pin CPU as the default
    # platform for the OMIE test suite. RL training entrypoints
    # (omiecid_simulator/jaxrl/MARL/) keep GPU as default and are not
    # affected by this pin — it is scoped to tests/omiecid/ via this
    # conftest, not a repo-wide one. Per-test GPU coverage for the
    # OQ-16 Method A tick-precision invariant pair is restored by the
    # device-parametrized fixture in
    # ``test_order_validator.py`` / ``test_order_validator_tick_precision_audit.py``
    # via per-call ``jax.default_device(...)`` context managers (which
    # override this session-level platform pin for the wrapped tests).
    #
    # MUST be set AFTER the jax_compilation_cache_dir knobs above so the
    # cache directory is fixed before any platform-dependent compile.
    # See docs/omiecid/device-policy.md for the JAX_PLATFORMS-vs-
    # ``jax_platform_name`` convention.
    jax.config.update("jax_platform_name", "cpu")


# ---------------------------------------------------------------------------
# @pytest.mark.gpu — auto-skip on CUDA-hidden hosts (P3-OPS-01a)
# ---------------------------------------------------------------------------

def _cuda_visible() -> bool:
    """Return True iff a CUDA GPU is visible to this process.

    Implementation: try to enumerate GPU devices via ``jax.devices('gpu')``
    after the conftest has pinned the default platform to CPU. The pin
    does NOT prevent ``jax.devices(...)`` from probing other backends —
    it only controls the default platform for un-pinned ops. So this
    probe still detects a CUDA device when one is visible.

    Returns False when ``CUDA_VISIBLE_DEVICES=""`` or when no NVIDIA
    device is present (jax raises ``RuntimeError`` enumerating the
    unavailable backend).
    """
    try:
        import jax
    except ImportError:
        return False
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# @pytest.mark.audit — auto-tag tests/omiecid/*_audit.py (P3-TEST-AUDIT-DECOUPLE-1)
# ---------------------------------------------------------------------------
#
# §1 regression-gate members that carry the ``_audit.py`` suffix but MUST stay
# in routine runs (test-perf-gate.md §1 categories (ii)/(iii)). These are NOT
# tagged ``audit``, so ``-m "not audit"`` never deselects them. See the
# 2026-05-11 rollback for why a blanket filename exclusion is unsafe.
#
# GUARDRAIL: any ``*_audit.py`` file promoted to §1 must be added here, or
# ``-m "not audit"`` will silently drop it from routine gates (test-perf-gate.md §1).
_AUDIT_MARKER_EXEMPT = frozenset({
    "test_contract_book_audit.py",
    "test_ioc_fok_audit.py",
})


def pytest_collection_modifyitems(config, items):
    """Auto-tag ``*_audit.py`` tests with ``audit``, then auto-skip
    ``@pytest.mark.gpu`` tests when no CUDA device is visible.

    Audit tagging (P3-TEST-AUDIT-DECOUPLE-1): per-story independent-
    verification files (``tests/omiecid/*_audit.py``, orchestrate.md Step 5)
    are tagged ``audit`` so routine gates can deselect them with
    ``-m "not audit"`` while a bare ``pytest tests/omiecid/`` still collects
    everything (the marker is additive). The two §1 gate members are
    exempt (``_AUDIT_MARKER_EXEMPT``) and always run. This tagging MUST
    run BEFORE the GPU early-``return`` below, or it silently no-ops on
    CUDA-visible dev workstations. pytest's own ``-m`` filter
    (``_pytest.mark.deselect_by_mark``) is itself a
    ``pytest_collection_modifyitems`` hook that runs AFTER conftest hooks,
    so a marker added here IS seen by ``-m "not audit"`` (verified on
    pytest 9.0.2).

    GPU auto-skip: per the 2026-05-28 device-policy ruling, the OMIE test
    suite must stay green on CUDA-hidden hosts (CI determinism). Tests
    that legitimately need a GPU carry ``@pytest.mark.gpu`` and skip
    cleanly here when CUDA is not available, rather than erroring inside
    the test body.
    """
    # Audit auto-tag — MUST precede the GPU early-return.
    for item in items:
        fname = item.path.name  # e.g. "test_omie_env_audit.py"
        if fname.endswith("_audit.py") and fname not in _AUDIT_MARKER_EXEMPT:
            item.add_marker(pytest.mark.audit)

    # GPU auto-skip.
    if _cuda_visible():
        return
    skip_gpu = pytest.mark.skip(reason="GPU not visible (CUDA_VISIBLE_DEVICES empty or no NVIDIA device)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def product_config():
    """Default OMIE product config."""
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def market_config():
    """Default OMIE market config."""
    return OMIEMarketConfig()


# ---------------------------------------------------------------------------
# Contract fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def open_contract():
    """An OPEN contract (mid-day, contract 37: 09:15-09:30)."""
    return Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


# ---------------------------------------------------------------------------
# Book + state fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def book(open_contract, product_config, market_config):
    """A ContractBook backed by an OPEN contract with default configs."""
    return ContractBook(open_contract, product_config, market_config)


@pytest.fixture(scope="module")
def empty_state(book):
    """An initialized empty OMIEBookState."""
    return book.init()


@pytest.fixture(scope="module")
def book_and_state(book):
    """Convenience tuple: (ContractBook, initial OMIEBookState)."""
    return book, book.init()
