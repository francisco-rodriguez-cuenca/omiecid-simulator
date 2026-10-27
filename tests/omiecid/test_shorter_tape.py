"""Locked behavioral contracts for Story P3-PERF-TAPE-1 — Shorter liquid-hour tape.

These are the LOCKED tests the P3-PERF-TAPE-1 developer implements against,
derived from ``docs/omiecid/agile-plan.md`` §"Story P3-PERF-TAPE-1 — Shorter
liquid-hour tape / smaller ``n_orders_padded``" (Test-first contract 1-3) and the
governing domain ruling ``docs/omiecid/market-spec.md`` §14 "Contract order-flow
magnitude and tape-selection semantics (DA-TAPE-2026-07)" — in particular §14.4
(the non-degenerate-fills criterion behind test (2)).

--------------------------------------------------------------------------------
DEVELOPER -> TEST INTERFACE (the locked config-path/key contract)
--------------------------------------------------------------------------------
The exact shorter window (which ``contract_id`` / ``delivery_date``, or head-cap
size) is NOT pinnable in advance: it depends on what OMIE data is on disk and
which candidate clears §14.4 (see §14.5 "No-assume-data"). The developer surveys
``data/rawOMIE/`` and picks the concrete window. These tests therefore assert the
**criteria** (steps < baseline; non-degenerate fills; config resolves), reading
whatever shorter-tape config the developer produces — they do NOT hard-code a
specific chosen ``contract_id``/date as the expected shorter window.

The one thing frozen here (so the developer->test contract is unambiguous) is
**where the shorter-tape config lives**:

  * NEW ADDITIVE config file: ``config/env_configs/omie_short_tape.json``
    (preferred additive path per the orchestrator's additive decision + §14.5 —
    do NOT mutate ``omie_single_contract.json``; the D-D contract_id-16 /
    2025-06-01..03 window must stay intact as the reproducibility anchor).

  * It carries the SAME two-block schema as ``omie_single_contract.json``:
      - an ``omie_env_config`` block (the Session config-dict knobs;
        ``orders_per_step`` in particular — used to derive ``steps_per_episode``);
      - a ``data_source`` block (``loader``/``data_path``/``contract_ids``/
        ``delivery_dates``/``cache_path``/``pad_extra`` [+ OPTIONALLY a new
        per-tape order cap or explicit ``n_orders_padded`` key if the developer
        applied the §14.3(2) head-cap fallback]) that resolves through
        ``omiecid_simulator.jaxomiecid.omie_data_loader.build_fixed_tape_set`` into the
        ``(n_tapes, N_orders_padded, 13)`` int32 tapes array Session consumes.

  * The ``data_source`` block MUST resolve to a set whose ``n_orders_padded`` and
    derived ``steps_per_episode = ceil(n_orders_padded / orders_per_step)`` are
    MATERIALLY SMALLER than the fixed 5459/1365 baseline (contract_id 16 /
    2025-06 window), per test (1) below.

  * If the developer applies a §14.3(2) head cap via a NEW ``build_fixed_tape_set``
    argument (e.g. a per-tape ``max_orders`` / ``head_orders`` / ``n_orders_padded``
    override) rather than pure small-``contract_id`` selection, the ``data_source``
    block must express that arg (any of the recognized key names probed by
    ``_data_source_to_tapes`` below) so this test file's resolver forwards it.

Baseline anchor (FIXED / KNOWN — the comparison target test (1) checks against):
  * ``contract_id 16`` / ``2025-06-01..03`` loads ``(3, 5459, 13)`` ⇒
    ``n_orders_padded == 5459`` ⇒ ``steps_per_episode == 1365`` at
    ``orders_per_step == 4`` (report §2 Setup;
    ``outputs/cid_rl/runs/stage0_20260703_585f775/report.md``). These figures are
    hard-coded below as the baseline to beat.

--------------------------------------------------------------------------------
DOMAIN FLOOR for test (2) (market-spec §14.4 — authoritative)
--------------------------------------------------------------------------------
The chosen (contract, date) shorter tape must clear ALL of:
  (a) >= 1 executed historical trade for that ``contract_id`` in the OMIE
      ``trades_YYYYMMDD.1`` file (via ``load_omie_trades`` — the book demonstrably
      crossed at least once historically);
  (b) both order sides present in the loaded tape — >= 1 BUY (Side.BUY==0) and
      >= 1 SELL (Side.SELL==1) row (§13.1 column 8 / loader column [1]);
  (c) a scripted-agent rollout over the shortened tape produces NON-ZERO
      simulated fills (the agent-reachable confirmation).
(a)+(b) are facts about the real tape; (c) is the behavioral confirmation over a
scripted replay. These are non-degeneracy engineering thresholds, NOT market
rules (§14.4). ">= 1 historical trade + both sides + non-zero rollout fills" is
the FLOOR.

--------------------------------------------------------------------------------
OPERATIONAL FORM of §14.4(c) — the scripted rollout is the PRODUCTION REPLAY, NOT
a Discrete(13) crossing agent (test-agent finding; ambiguity surfaced)
--------------------------------------------------------------------------------
§14.4(c) names "a scripted-agent rollout produces a non-zero simulated fill
count." The natural reading — an ``Session`` agent submitting a marketable
``Discrete(13)`` order that crosses the resting book — is NOT a reliable
non-degeneracy signal, for a reason that is a fixed property of the ENV action
space, not of the tape: the action decoder prices the agent's order at
``mid + price_offsets[bucket]`` with ``price_offsets == (-1, 0, +1)`` EUR
(``omie_session.py:_decode_action`` / the config's ``price_offsets`` knob), so the
agent can only cross a book whose half-spread is <= 1 EUR. Real OMIE books are
far wider: on the verification window (contract_id 4 / 2025-06-02) the MINIMUM
top-of-book spread over the whole episode was 5.50 EUR, so a ``Discrete(13)``
agent NEVER crosses and its own fill count is structurally 0 regardless of how
non-degenerate the tape is. Asserting on the agent's own fills would therefore
make test (2c) a test of the (fixed) action geometry, not of tape
non-degeneracy.

The faithful, robust operational form the test-agent LOCKS for §14.4(c) is:
**the deterministic (scripted) production replay of the shortened tape's real
orders through a ``ContractBook`` produces NON-ZERO simulated fills**
(``replay_orders_through_book(orders, book, historical_trades).replayed_trades``
count >= 1). This is the same production ``_replay_orders`` path ``Session``
folds the background tape through every step; a non-zero replayed-fill count
confirms the book actually crosses in the simulator and therefore presents
real, agent-reachable resting liquidity — the "NOT a degenerate empty book"
condition the card and §14.4 demand. The "rest_until_gct" (H1) heuristic is the
scripted replay policy. (On the verification window this yielded 115 simulated
fills / 32 historical matches — see the test-authoring note in sprint-status.)
This reading is within §14.4's grant that "the test-agent will assert" the
operational form and "may set a stricter numeric threshold"; ">= 1 replayed
fill" is exactly the non-zero floor.

--------------------------------------------------------------------------------
TEST TIERING (per the test-agent runtime budget + conftest/pyproject markers)
--------------------------------------------------------------------------------
  * ``TestConfigLoads`` (test (3), the config-parse half): pure JSON parsing — no
    JAX compile — fast suite (NO marker). The ``build_fixed_tape_set`` resolution
    half of test (3) reads real OMIE CSVs / warm cache ⇒ ``@pytest.mark.slow``.
  * ``TestShorterTapeReducesSteps`` (test (1)): reads real OMIE data via the
    loader AND compiles/steps ``Session`` (reset -> step -> done) ⇒
    ``pytestmark = [pytest.mark.jit, pytest.mark.slow]`` at class scope.
  * ``TestShorterTapeStillExercisesRealFills`` (test (2)): reads real OMIE data
    (orders + trades) AND runs a scripted-agent rollout that compiles/steps
    ``step_env`` ⇒ ``pytestmark = [pytest.mark.jit, pytest.mark.slow]``.

PRE-IMPLEMENTATION STATE: the developer has not yet created
``config/env_configs/omie_short_tape.json``, so every test that depends on it
SKIPs cleanly (config file absent / data absent / builder-signature mismatch)
rather than erroring — mirroring ``test_omie_config_smoke.py``. The domain ruling
(§14) RESOLVED the domain ambiguity, so there is NO ``pytest.mark.skip`` for
domain ambiguity here — skips fire only on the not-yet-created config artifact,
absent real data, or an unmatched builder signature.

OQ-18 x64 TRAP: any test that vmaps/jits ``step_env`` wraps the trace-triggering
call sites in ``with jax.experimental.enable_x64():`` (mirroring the P3.3 smoke
tests / ``ippo_rnn_JAXMARL._outer_x64_context``), or it fails with the
int64-``extract_fills`` promotion ValueError. This file steps ``step_env`` under
that outer context.

RELEASE NOTE (REL-07): the research config files
``config/env_configs/omie_short_tape.json`` / ``omie_single_contract.json`` are
not shipped. The shorter-tape window and its session keys are inlined below
(``SHORT_TAPE_CONFIG``); the two tests whose subject was the files themselves
(the shorter config parses; the baseline file was not mutated) were removed.
The participant submits nothing (``AgentOrder.none()``) when stepping to done.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Repository-root-relative paths.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The shorter-tape window of the research config
# ``config/env_configs/omie_short_tape.json`` (not shipped), inlined: its
# session keys and its data_source block (RL keys dropped).
SHORT_TAPE_CONFIG = {
    "omie_env_config": {
        "orders_per_step": 4,
        "gct_minutes_from_got": 60,
    },
    "data_source": {
        "loader": "omie",
        "data_path": "data/rawOMIE",
        "contract_ids": [0],
        "delivery_dates": ["2025-06-01", "2025-06-02", "2025-06-03"],
        "cache_path": "saved_npz/omie/",
        "pad_extra": 8,
    },
}

OMIE_DATA_ROOT = _REPO_ROOT / "data" / "rawOMIE"

# Baseline anchor (FIXED / KNOWN — report §2 Setup). The shorter tape must beat
# BOTH of these. These are the comparison targets, not a chosen window.
BASELINE_N_ORDERS_PADDED = 5459
BASELINE_STEPS_PER_EPISODE = 1365
BASELINE_ORDERS_PER_STEP = 4

# The OMIEOrder 13-column int32 layout (omie_data_loader.py / omie_session.py).
TAPE_WIDTH = 13

# Loaded-order side encoding (order_types.Side): column [1] of the (N,13) tape.
SIDE_COL = 1
SIDE_BUY = 0
SIDE_SELL = 1

# The scripted-rollout fill signal (§14.4(c)) is the count of simulated fills the
# production H1 replay of the shortened tape's real orders produces in the
# ContractBook (ReplayResult.replayed_trades). See the module-docstring
# "OPERATIONAL FORM of §14.4(c)" for why the Discrete(13) agent's OWN fills are
# NOT used (±1 EUR price_offsets cannot cross realistic OMIE spreads).


# ---------------------------------------------------------------------------
# Import guards — keep the file collectible before P3-PERF-TAPE-1 lands.
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    import jax.experimental  # for enable_x64 (OQ-18 outer context)

    from omiecid_simulator.omiecid.omie_session import AgentOrder, Session
    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.omie_config import (
        OMIEProductConfig,
        OMIEMarketConfig,
    )
    from omiecid_simulator.omiecid.contract_book import ContractBook

    _JAX_MISSING = False
    _JAX_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - collection-time guard
    _JAX_MISSING = True
    _JAX_IMPORT_ERROR = repr(exc)

try:
    from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
    from omiecid_simulator.jaxomiecid import omie_data_loader as _loader_mod

    _LOADER_MISSING = False
    _LOADER_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - collection-time guard
    _LOADER_MISSING = True
    _LOADER_IMPORT_ERROR = repr(exc)


_jax_skip = pytest.mark.skipif(
    _JAX_MISSING,
    reason=f"Session / jax not importable ({_JAX_IMPORT_ERROR})",
)


# ---------------------------------------------------------------------------
# Config discovery / data-source resolution helpers (black-box).
#
# The shorter-tape config path is FROZEN (SHORT_TAPE_CONFIG_PATH) but the
# concrete window inside it is the developer's data-driven choice, so these
# helpers read whatever the developer wrote and resolve its `data_source` block
# through the (already-shipped, P3.3) `build_fixed_tape_set`. They forward the
# recognized `data_source` keys — including any NEW §14.3(2) head-cap arg the
# developer may add — and SKIP (never error) when a prerequisite is absent.
# ---------------------------------------------------------------------------

# Recognized head-cap / explicit-length key names in the data_source block. The
# §14.3(2) head-cap axis requires a NEW build_fixed_tape_set arg whose exact name
# is a developer deliverable not frozen here; probe the documented candidates.
_HEAD_CAP_KEYS = (
    "n_orders_padded",   # explicit common padded length (already supported)
    "max_orders",        # candidate per-tape head cap
    "max_orders_per_tape",
    "head_orders",
    "order_cap",
    "first_k_orders",
)


def _short_config_or_skip():
    """Return a copy of the inlined shorter-tape config."""
    return copy.deepcopy(SHORT_TAPE_CONFIG)


def _data_source_block(cfg):
    """Return the `data_source` sub-block, or SKIP if the schema is missing it."""
    ds = cfg.get("data_source")
    if not isinstance(ds, dict):
        pytest.skip(
            "shorter-tape config has no `data_source` block — the config must "
            "carry the two-block (omie_env_config + data_source) schema so the "
            "loader-glue resolves it through build_fixed_tape_set"
        )
    return ds


def _orders_per_step(cfg):
    """Resolve orders_per_step from the config (env default 4 if unset)."""
    env_block = cfg.get("omie_env_config", {})
    if isinstance(env_block, dict) and "orders_per_step" in env_block:
        return int(env_block["orders_per_step"])
    if "orders_per_step" in cfg:
        return int(cfg["orders_per_step"])
    return 4  # Session default (omie_session.py:159)


def _forward_head_cap_kwargs(ds):
    """Collect any recognized head-cap / explicit-length kwargs from data_source.

    Returns a dict of the kwargs to forward to build_fixed_tape_set. Only keys
    the builder actually accepts are forwarded (probed against its signature) so
    a developer-added arg the builder does not yet accept degrades to a SKIP at
    the call site rather than a spurious TypeError.
    """
    import inspect

    forwarded = {}
    if _LOADER_MISSING:
        return forwarded
    try:
        params = set(inspect.signature(_loader_mod.build_fixed_tape_set).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        params = set()
    for key in _HEAD_CAP_KEYS:
        if key in ds and key in params:
            forwarded[key] = ds[key]
    return forwarded


def _data_available_for(ds) -> bool:
    """True iff the orders+trades files for the config's window exist on disk."""
    data_path = Path(ds.get("data_path", str(OMIE_DATA_ROOT)))
    if not data_path.is_absolute():
        data_path = _REPO_ROOT / data_path
    dates = ds.get("delivery_dates") or (
        [ds["delivery_date"]] if "delivery_date" in ds else []
    )
    if not dates:
        return False
    for d in dates:
        nodash = str(d).replace("-", "")
        if not (data_path / f"orders_{nodash}.1").exists():
            return False
    return True


def _build_tapes_from_data_source(ds):
    """Resolve the `data_source` block into a (n_tapes, N_padded, 13) tapes array.

    Uses the shipped build_fixed_tape_set pointer form, forwarding whatever
    contract_ids / delivery_dates / pad_extra / cache_path / head-cap args the
    developer wrote. SKIPs (never errors) when the builder is absent, the data
    is absent, or the builder rejects the forwarded kwargs.
    """
    if _LOADER_MISSING:
        pytest.skip(
            f"omiecid_simulator.jaxomiecid.omie_data_loader not importable "
            f"({_LOADER_IMPORT_ERROR})"
        )
    builder = getattr(_loader_mod, "build_fixed_tape_set", None)
    if not callable(builder):
        pytest.skip(
            "build_fixed_tape_set absent from the loader module — P3.3 loader-glue "
            "(min-GREEN gate for P3-PERF-TAPE-1)"
        )
    if not _data_available_for(ds):
        pytest.skip(
            "real OMIE data for the shorter-tape window is absent under "
            f"{OMIE_DATA_ROOT} (gitignored, local-only) — cannot resolve the "
            "data_source block"
        )

    data_path = ds.get("data_path", str(OMIE_DATA_ROOT))
    if not Path(data_path).is_absolute():
        data_path = str(_REPO_ROOT / data_path)
    contract_ids = ds.get("contract_ids") or (
        [ds["contract_id"]] if "contract_id" in ds else None
    )
    delivery_dates = ds.get("delivery_dates") or (
        [ds["delivery_date"]] if "delivery_date" in ds else None
    )
    if not contract_ids or not delivery_dates:
        pytest.skip(
            "shorter-tape data_source block lacks contract_ids / delivery_dates "
            "— cannot resolve through build_fixed_tape_set"
        )

    kwargs = dict(
        data_path=data_path,
        contract_ids=[int(c) for c in contract_ids],
        delivery_dates=[str(d) for d in delivery_dates],
    )
    if "cache_path" in ds:
        kwargs["cache_path"] = ds["cache_path"]
    if "pad_extra" in ds:
        kwargs["pad_extra"] = int(ds["pad_extra"])
    kwargs.update(_forward_head_cap_kwargs(ds))

    try:
        tapes = jnp.asarray(builder(**kwargs), dtype=jnp.int32)
    except TypeError as exc:
        pytest.skip(
            "build_fixed_tape_set rejected the forwarded data_source kwargs "
            f"({exc!r}) — if the developer added a NEW head-cap arg its name is "
            f"not in this file's probe list {_HEAD_CAP_KEYS}; surfaced ambiguity"
        )
    except ValueError as exc:
        # e.g. every requested (contract, date) loaded ZERO orders — the chosen
        # window resolves to no data. Surface it, don't mask it as a pass.
        pytest.fail(
            "build_fixed_tape_set raised ValueError resolving the shorter-tape "
            f"window ({exc!r}) — the chosen data_source resolves to an empty / "
            "invalid tape set; the developer must pick a window that loads data"
        )
    return tapes


def _first_data_config(ds):
    """Build ONE OMIEDataConfig for the shorter tape's (first) contract/date.

    Used by the §14.4 (a)/(b) real-data checks (historical trades / both sides).
    Returns the config, or SKIPs when the schema/data is unavailable.
    """
    if _LOADER_MISSING:
        pytest.skip("loader module not importable")
    if not _data_available_for(ds):
        pytest.skip("real OMIE data for the shorter-tape window absent")
    data_path = ds.get("data_path", str(OMIE_DATA_ROOT))
    if not Path(data_path).is_absolute():
        data_path = str(_REPO_ROOT / data_path)
    contract_ids = ds.get("contract_ids") or (
        [ds["contract_id"]] if "contract_id" in ds else None
    )
    delivery_dates = ds.get("delivery_dates") or (
        [ds["delivery_date"]] if "delivery_date" in ds else None
    )
    if not contract_ids or not delivery_dates:
        pytest.skip("shorter-tape data_source lacks contract_ids / delivery_dates")
    cache_path = ds.get("cache_path", "saved_npz/omie/")
    return OMIEDataConfig(
        data_path=str(data_path),
        contract_id=int(contract_ids[0]),
        delivery_date=str(delivery_dates[0]),
        cache_path=str(cache_path),
    )


# ---------------------------------------------------------------------------
# Scripted production-replay helper (test (2), criterion (c)).
#
# Per the module-docstring "OPERATIONAL FORM of §14.4(c)" finding, the scripted
# rollout is the deterministic H1 ("rest_until_gct") PRODUCTION replay of the
# shortened tape's real orders through a ContractBook — the same _replay_orders
# path Session folds the background tape through — and the non-degeneracy signal
# is a NON-ZERO simulated fill count (replayed_trades). A Discrete(13) crossing
# agent is NOT used: its ±1 EUR price_offsets cannot cross realistic OMIE spreads
# (verification window min spread 5.50 EUR), so its fill count would test the
# fixed action geometry, not tape non-degeneracy.
# ---------------------------------------------------------------------------
def _contract_book_for(contract_id: int):
    """Build an OPEN ContractBook for a contract_id (default OMIE configs).

    ``delivery_start_minutes = contract_id * 15`` and ``gct_minutes = start - 60``
    per market-spec §13.4 / §3.1 (contract.py:generate_contracts). Default
    product/market configs (the same the loader/env use). OPEN status so the
    replay scan is entered (P2-L3 caller-contract).
    """
    start = int(contract_id) * 15
    contract = Contract(
        contract_id=int(contract_id),
        delivery_start_minutes=start,
        delivery_end_minutes=start + 15,
        gct_minutes=start - 60,
        status=ContractStatus.OPEN,
    )
    return ContractBook(contract, OMIEProductConfig(), OMIEMarketConfig())


def _replayed_fill_count(orders, contract_id: int, historical_trades) -> int:
    """Return the NON-ZERO-able simulated fill count of the scripted H1 replay.

    Runs ``replay_orders_through_book`` (the production replay, "rest_until_gct"
    heuristic) over the shortened tape's real orders through an OPEN ContractBook
    for ``contract_id`` and returns ``replayed_trades`` count — the number of
    fills the simulated book crossed to produce. >= 1 confirms the book is not a
    degenerate empty book (§14.4(c)).
    """
    replay_fn = getattr(_loader_mod, "replay_orders_through_book", None)
    if not callable(replay_fn):
        pytest.skip("replay_orders_through_book absent from the loader module")
    book = _contract_book_for(contract_id)
    result = replay_fn(orders, book, historical_trades)
    replayed = jnp.asarray(result.replayed_trades)
    return int(replayed.shape[0])


# ===========================================================================
# Contract (3): test_config_loads — the shorter-tape config parses and its
# data_source block resolves through build_fixed_tape_set.
# ===========================================================================
class TestConfigLoads:
    """Test-first contract (3): the shorter-tape config parses (fast) and its
    data_source block resolves through build_fixed_tape_set (slow, real data).

    The PARSE half is pure JSON — fast suite (no marker). The RESOLVE half reads
    real OMIE CSVs / warm cache (the production data path) so it is an
    INTEGRATION test carrying @pytest.mark.slow (a synthetic fixture would defeat
    the purpose per the test-agent integration-test rule).
    """

    # ---- fast: pure config checks -----------------------------------------
    def test_short_tape_config_has_two_block_schema(self):
        """It carries the omie_env_config + data_source two-block schema (P3.3)."""
        cfg = _short_config_or_skip()
        assert "data_source" in cfg and isinstance(cfg["data_source"], dict), (
            "shorter-tape config must carry a `data_source` block (the loader-glue "
            "pointer resolved by build_fixed_tape_set)"
        )
        ds = cfg["data_source"]
        # OMIE-native loader, NEVER LOBSTER (§14.5).
        assert str(ds.get("loader", "omie")).lower() == "omie", (
            f"shorter-tape data_source.loader must be 'omie' (OMIE-native, NEVER "
            f"LOBSTER, §14.5); found {ds.get('loader')!r}"
        )
        has_contract = ("contract_ids" in ds) or ("contract_id" in ds)
        has_delivery = ("delivery_dates" in ds) or ("delivery_date" in ds)
        assert has_contract and has_delivery, (
            "data_source must carry contract_ids + delivery_dates the loader-glue "
            f"resolves; found contract={has_contract}, delivery={has_delivery}"
        )

    # ---- slow: real-data resolution through build_fixed_tape_set -----------
    @_jax_skip
    @pytest.mark.slow
    def test_short_tape_data_source_resolves_through_builder(self):
        """The data_source block resolves through build_fixed_tape_set to a
        (n_tapes, N_orders_padded, 13) int32 tapes array (the loaded contract).

        slow: reads real OMIE data (data/rawOMIE/) — the production data path;
        SKIPs cleanly pre-implementation (config absent), when the data is absent,
        or when a developer-added head-cap arg is outside the probe list.
        """
        cfg = _short_config_or_skip()
        ds = _data_source_block(cfg)
        tapes = _build_tapes_from_data_source(ds)
        assert tapes.ndim == 3, (
            f"resolved tape set must be 3-D (n_tapes, N_orders_padded, 13); got "
            f"{tapes.shape}"
        )
        assert tapes.shape[-1] == TAPE_WIDTH, (
            f"tape trailing width must be {TAPE_WIDTH} (OMIEOrder layout); got "
            f"{tapes.shape[-1]}"
        )
        assert tapes.dtype == jnp.int32, (
            f"resolved tape set must be int32 (Session consumes int32 tapes); got "
            f"{tapes.dtype}"
        )
        assert tapes.shape[0] >= 1, "resolved set must contain >= 1 tape"


# ===========================================================================
# Contract (1): test_shorter_tape_reduces_steps_per_episode — the chosen window
# yields SMALLER n_orders_padded AND steps_per_episode than the 5459/1365
# baseline, and the env still resets/steps to `done` at GCT on the shorter tape.
# ===========================================================================
@_jax_skip
class TestShorterTapeReducesSteps:
    """Test-first contract (1): fewer padded orders ⇒ fewer steps than baseline,
    AND the env still resets -> steps -> done at GCT on the shorter tape.

    slow+jit: resolves the tape set from real OMIE data AND compiles/steps
    Session (reset -> step_env under the OQ-18 outer enable_x64 context, driven to
    done). Reads the production data path so a synthetic fixture would defeat the
    purpose; the vmap-free single-env step still triggers an XLA compile ⇒ jit.
    SKIPs cleanly pre-implementation (config / data / builder-signature absent).
    """

    pytestmark = [pytest.mark.jit, pytest.mark.slow]

    def _short_env(self):
        """Build an Session over the resolved shorter-tape set (config-driven)."""
        cfg = _short_config_or_skip()
        ds = _data_source_block(cfg)
        tapes = _build_tapes_from_data_source(ds)
        ops = _orders_per_step(cfg)
        env = Session({"tapes": tapes, "orders_per_step": ops})
        return env, tapes, ops

    def test_shorter_tape_reduces_n_orders_padded_below_baseline(self):
        """The resolved shorter tape has a SMALLER n_orders_padded than 5459.

        §14.5(ii): "materially smaller than the 5459/1365 baseline". This is the
        primary throughput lever — fewer padded orders per tape.
        """
        env, tapes, _ops = self._short_env()
        assert env.n_orders_padded == tapes.shape[1]
        assert env.n_orders_padded < BASELINE_N_ORDERS_PADDED, (
            f"shorter tape n_orders_padded={env.n_orders_padded} must be < the "
            f"{BASELINE_N_ORDERS_PADDED}-order baseline (contract_id 16 / 2025-06); "
            "the whole point of the story is a SHORTER window (§14.2/§14.3)"
        )

    def test_shorter_tape_reduces_steps_per_episode_below_baseline(self):
        """steps_per_episode = ceil(n_orders_padded/orders_per_step) < 1365.

        The derived episode length (compile-time constant, omie_session.py:237-239)
        must beat the 1365-step baseline — this is the metric the throughput
        re-run measures against (§14.5(ii)).
        """
        env, _tapes, ops = self._short_env()
        expected = math.ceil(env.n_orders_padded / ops)
        assert env.steps_per_episode == expected, (
            "steps_per_episode must equal ceil(n_orders_padded/orders_per_step) "
            f"(={expected}); env reports {env.steps_per_episode}"
        )
        assert env.steps_per_episode < BASELINE_STEPS_PER_EPISODE, (
            f"shorter-tape steps_per_episode={env.steps_per_episode} must be < the "
            f"{BASELINE_STEPS_PER_EPISODE}-step baseline (5459 orders / "
            f"{BASELINE_ORDERS_PER_STEP} orders_per_step); the story's throughput "
            "goal is fewer steps/episode"
        )

    def test_shorter_tape_env_resets_cleanly(self):
        """The env resets to a valid state at GOT (time==0, cursor==0)."""
        env, _tapes, _ops = self._short_env()
        with jax.experimental.enable_x64():
            state = env.reset(jax.random.PRNGKey(0))
        assert state is not None
        assert int(jnp.asarray(state.tape_cursor).reshape(-1)[0]) == 0, (
            "reset must start at tape_cursor 0 (GOT origin)"
        )
        assert int(jnp.asarray(state.time).reshape(-1)[0]) == 0, (
            "reset must start at time 0 (minutes-from-GOT origin)"
        )

    def test_shorter_tape_steps_to_done_at_gct(self):
        """Stepping the shorter-tape env to episode end reaches done == True at GCT.

        The done contract (omie_session.py:401-402): done == (time >= gct_minutes_from_got).
        Driving the env through its (now-smaller) steps_per_episode NOOP steps must
        eventually set done True — proving the shorter tape still terminates at GCT
        rather than never finishing. Runs under the OQ-18 outer enable_x64 context.
        Submits no order (``AgentOrder.none()``) — this test asserts TERMINATION,
        not fills.
        """
        env, _tapes, _ops = self._short_env()
        # No order — termination is independent of the participant's order.
        with jax.experimental.enable_x64():
            state = env.reset(jax.random.PRNGKey(0))
            order = AgentOrder.none()
            reached_done = False
            # steps_per_episode is the exact horizon; +1 guards the boundary.
            for i in range(env.steps_per_episode + 1):
                key = jax.random.PRNGKey(1 + i)
                state, _report, done = env.step_env(key, state, order)
                if bool(jnp.asarray(done).reshape(-1)[0]):
                    reached_done = True
                    break
        assert reached_done, (
            "the shorter-tape env must reach done == True within "
            f"steps_per_episode (+1) = {env.steps_per_episode + 1} steps "
            "(done fires at time >= GCT); it never terminated"
        )
        # done fired because time crossed GCT — confirm the terminal time bound.
        final_time = int(jnp.asarray(state.time).reshape(-1)[0])
        assert final_time >= env.gct_minutes_from_got, (
            f"done fired but terminal time {final_time} < GCT "
            f"{env.gct_minutes_from_got}; done must be time-at-GCT driven "
            "(omie_session.py:401-402)"
        )


# ===========================================================================
# Contract (2): test_shorter_tape_still_exercises_real_fills — the shortened
# tape produces NON-TRIVIAL agent-reachable fills (NOT a degenerate empty book),
# per the market-spec §14.4 non-degeneracy floor:
#   (a) >= 1 historical trade in trades_YYYYMMDD.1 for the contract_id;
#   (b) both order sides (>=1 BUY, >=1 SELL) present in the loaded tape;
#   (c) a scripted-agent rollout produces NON-ZERO simulated fills.
# ===========================================================================
@_jax_skip
class TestShorterTapeStillExercisesRealFills:
    """Test-first contract (2): the shortened tape is non-degenerate per §14.4.

    slow+jit: (a)/(b) read real OMIE orders+trades files (the production data
    path); (c) runs the scripted H1 production replay (replay_orders_through_book)
    which compiles the jitted _replay_orders scan over the tape. All three SKIP
    cleanly pre-implementation (config / data absent). The §14.4 floor is a
    NON-DEGENERACY ENGINEERING THRESHOLD, not a market rule (§14.4 final para).
    """

    pytestmark = [pytest.mark.jit, pytest.mark.slow]

    def _ds_or_skip(self):
        cfg = _short_config_or_skip()
        return _data_source_block(cfg), cfg

    # ---- §14.4(a): >= 1 executed historical trade for the contract_id -------
    def test_historical_trades_file_has_at_least_one_trade(self):
        """§14.4(a): the OMIE trades_YYYYMMDD.1 file has >= 1 executed trade for
        the chosen contract_id — the historical book demonstrably crossed.

        Read directly from real data via load_omie_trades (an (M,6) array filtered
        by contract_id); M >= 1 is the domain floor. A contract whose historical
        flow produced zero trades is provably degenerate (§14.4) and fails the
        experiment's premise regardless of throughput.
        """
        ds, _cfg = self._ds_or_skip()
        data_cfg = _first_data_config(ds)
        loader = getattr(_loader_mod, "load_omie_trades", None)
        if not callable(loader):
            pytest.skip("load_omie_trades absent from the loader module")
        trades = jnp.asarray(loader(data_cfg))
        n_trades = int(trades.shape[0])
        assert n_trades >= 1, (
            f"§14.4(a) FLOOR: contract_id {data_cfg.contract_id} / "
            f"{data_cfg.delivery_date} has {n_trades} historical trades in "
            f"trades_{data_cfg.delivery_date.replace('-', '')}.1 — the domain "
            "requires >= 1 (the book must demonstrably have crossed); the chosen "
            "shorter window is degenerate and must escalate per §14.5"
        )

    # ---- §14.4(b): both order sides present in the loaded tape --------------
    def test_loaded_tape_has_both_order_sides(self):
        """§14.4(b): the loaded tape contains >= 1 BUY and >= 1 SELL order, so a
        crossing is structurally possible.

        Read from real data via load_omie_orders ((N,13) tape); column [1] is the
        side (Side.BUY==0 / Side.SELL==1). Both present ⇒ resting bid AND ask
        liquidity can meet. A one-sided tape is degenerate (§14.4).
        """
        ds, _cfg = self._ds_or_skip()
        data_cfg = _first_data_config(ds)
        loader = getattr(_loader_mod, "load_omie_orders", None)
        if not callable(loader):
            pytest.skip("load_omie_orders absent from the loader module")
        orders = jnp.asarray(loader(data_cfg))
        assert orders.shape[0] >= 1, "the loaded tape must have >= 1 real order"
        sides = orders[:, SIDE_COL]
        n_buy = int(jnp.sum(sides == SIDE_BUY))
        n_sell = int(jnp.sum(sides == SIDE_SELL))
        assert n_buy >= 1 and n_sell >= 1, (
            f"§14.4(b) FLOOR: the loaded tape for contract_id "
            f"{data_cfg.contract_id} / {data_cfg.delivery_date} must contain BOTH "
            f"sides (>=1 BUY, >=1 SELL) so a crossing is structurally possible; "
            f"found n_buy={n_buy}, n_sell={n_sell} — a one-sided tape is degenerate"
        )

    # ---- §14.4(c): scripted production replay produces NON-ZERO fills -------
    def test_scripted_replay_produces_nonzero_simulated_fills(self):
        """§14.4(c): the scripted H1 production replay of the shorter tape produces
        NON-ZERO simulated fills — the book crosses in the simulator, so the tape
        presents real, agent-reachable liquidity (NOT a degenerate empty book).

        BEHAVIORAL confirmation on top of the (a)/(b) real-data facts. The scripted
        rollout is the deterministic "rest_until_gct" (H1) production replay of the
        shortened tape's real orders through an OPEN ContractBook — the same
        ``_replay_orders`` path ``Session`` folds the background tape through — and
        the non-degeneracy signal is a NON-ZERO ``replayed_trades`` count. See the
        module docstring "OPERATIONAL FORM of §14.4(c)": a ``Discrete(13)`` crossing
        agent is NOT the operational form, because its ±1 EUR ``price_offsets``
        cannot cross realistic OMIE spreads (verification-window min spread 5.50
        EUR) — its own fill count would test the fixed action geometry, not tape
        non-degeneracy. The replay reads the FIRST (contract_id, date) of the
        shorter-tape window; >= 1 simulated fill is the non-zero floor (§14.4).
        """
        ds, _cfg = self._ds_or_skip()
        data_cfg = _first_data_config(ds)
        orders_loader = getattr(_loader_mod, "load_omie_orders", None)
        trades_loader = getattr(_loader_mod, "load_omie_trades", None)
        if not callable(orders_loader) or not callable(trades_loader):
            pytest.skip("load_omie_orders / load_omie_trades absent from loader")
        orders = jnp.asarray(orders_loader(data_cfg))
        historical_trades = jnp.asarray(trades_loader(data_cfg))
        n_fills = _replayed_fill_count(
            orders, data_cfg.contract_id, historical_trades
        )
        assert n_fills >= 1, (
            "§14.4(c) FLOOR: the scripted H1 production replay of the shorter tape "
            f"(contract_id {data_cfg.contract_id} / {data_cfg.delivery_date}) "
            f"produced {n_fills} simulated fills — the domain requires NON-ZERO "
            "(>= 1) agent-reachable fills (the book must cross in the simulator). "
            "Zero fills means the shortened tape is a degenerate (near-)empty book "
            "and fails the learnability premise (§14.4); the window must escalate "
            "per §14.5"
        )
