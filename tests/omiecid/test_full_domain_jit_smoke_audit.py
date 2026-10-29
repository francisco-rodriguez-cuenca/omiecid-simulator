"""Audit tests for P2-H3 — Integration + acceptance gate (Phase 2 closure).

This file is an *independent* post-implementation audit of the integrated
OMIE domain stack on branch ``omie/P2-H3-integration-acceptance``. It does
NOT read or import from the original locked test file
(``tests/omiecid/test_full_domain_jit_smoke.py``). Any overlap with assertions
over there is incidental — every test below is derived independently from:

  * the P2-H3 story card (``docs/omiecid/agile-plan.md`` §"Story P2-H3 —
    Integration + acceptance gate" — 8 acceptance criteria);
  * the canonical Phase 2 module-level jit idiom pinned in
    ``docs/omiecid/migration-plan.md`` v5.11 §"Module-level jit idiom"
    (``static_argnames=("cfg", ...)``);
  * the Phase 2 acceptance subsection at migration-plan.md:888 (the 8 AC
    list — single source of truth);
  * the loader-dtype contract at migration-plan.md:806 (int32, ppd
    signed-int-ticks, ``decode_price`` / ``decode_qty`` retained for the
    Python boundary);
  * inspection of the finished implementation's PUBLIC interface only
    (class signatures, module-level symbols, AST of jit-targeted modules
    — never reverse-engineering of behaviour we'd be probing).

This audit deliberately probes ANGLES the original smoke harness is
unlikely to have covered, structured by 14 dimensions:

  1.  End-to-end determinism / reproducibility (two-run bitwise equality).
  2.  Cross-module pytree invariants (jnp.ndarray-only leaves; stable tree
      structure).
  3.  dtype invariants of integrated state (int32 / bool_ enforced).
  4.  ``SubmitResult`` field stability across the five ExecRestriction
      variants (NON / IOC / FOK / ICE / ICEwPI).
  5.  B1-B5 constants reachable from JIT scope (Python ``int`` — not
      ``jnp.int32`` arrays).
  6.  Phase 2 closure invariants documented in migration-plan.md.
  7.  AST invariants on the integration surface (no instance-method jit;
      no Python for/while inside jitted bodies; no dict/list ctor in
      jit scope).
  8.  Trace-cache hygiene across the full domain (one compile per
      module-level jit entry after exercising the full public API).
  9.  Empirical zero-sum settlement (P2-L2 invariant at the integrated
      layer via the matching loop).
 10.  No regression on P2-H1 truncation backstop (under-cap stays
      ``truncated == False``).
 11.  ``load_omie_agent_metadata`` contract (AC 8 deep probe — structure /
      types of the returned dict).
 12.  Module-level jit idiom conformance (``static_argnames`` tuple form).
 13.  (Removed: the demo-script import contract; the demo was
      deleted and tutorial T03 replaces it.)
 14.  Phase 3 surface untouched (deferral-respect).

Marker discipline (Story T1.1): every test traces or compiles
``_submit_order`` (or probes a module that closes over the jit cache);
the marker is pinned at module scope.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import re
import subprocess
from typing import List, Tuple

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    SubmitResult,
    _submit_order,
    _submit_order_body,
)
from omiecid_simulator.omiecid.iceberg_manager import IcebergManager
from omiecid_simulator.omiecid.jax_constants import (
    MAX_AGENTS,
    MAX_ICEBERGS,
    MAX_SLICES_PER_REGEN,
    SENTINEL_AGENT_ID,
    SENTINEL_ORDER_ID,
    make_sentinel_order,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    Status,
    Validity,
    make_order,
)
from omiecid_simulator.omiecid.settlement import compute_agent_pnl
from omiecid_simulator.omiecid.trade_extraction import RawFills, extract_fills


# Module-level marker (Story T1.1 — every test compiles or traces a jitted
# entry, or probes jit-cache hygiene).
pytestmark = pytest.mark.jit


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OMIE_DIR = REPO_ROOT / "omiecid_simulator" / "omiecid"
CONTRACT_BOOK_PATH = OMIE_DIR / "contract_book.py"
ICEBERG_MANAGER_PATH = OMIE_DIR / "iceberg_manager.py"
SETTLEMENT_PATH = OMIE_DIR / "settlement.py"
TRADE_EXTRACTION_PATH = OMIE_DIR / "trade_extraction.py"
ORDER_VALIDATOR_PATH = OMIE_DIR / "order_validator.py"
JAX_CONSTANTS_PATH = OMIE_DIR / "jax_constants.py"
ENCODING_PATH = OMIE_DIR / "encoding.py"


# Modules whose AST is probed for forbidden idioms in dimension 7. We
# DELIBERATELY exclude `*_demo.py` and `tests/` and `jaxomie/` per the
# audit scope ("the jit-safe domain modules only"). order_types.py,
# omie_config.py, jax_constants.py are pure data definitions and carry
# no jitted bodies; they are listed only for the "no @jax.jit on
# instance methods" probe.
JIT_DOMAIN_MODULES = [
    OMIE_DIR / "contract_book.py",
    OMIE_DIR / "iceberg_manager.py",
    OMIE_DIR / "settlement.py",
    OMIE_DIR / "trade_extraction.py",
    OMIE_DIR / "order_validator.py",
    OMIE_DIR / "encoding.py",
]


# ---------------------------------------------------------------------------
# Local fixtures (self-contained; uses the canonical conftest names where
# possible for module-scoped amortisation, but does not depend on them
# semantically — the audit is reproducible by re-execution alone).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def product_cfg() -> OMIEProductConfig:
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def market_cfg() -> OMIEMarketConfig:
    return OMIEMarketConfig()


@pytest.fixture(scope="module")
def open_contract_local() -> Contract:
    return Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


@pytest.fixture(scope="module")
def book_open(open_contract_local, product_cfg, market_cfg) -> ContractBook:
    return ContractBook(open_contract_local, product_cfg, market_cfg)


@pytest.fixture(scope="module")
def empty_state(book_open) -> OMIEBookState:
    return book_open.init()


def _mk_order(**overrides) -> OMIEOrder:
    """Build a default-valid OMIE order; overrides override the defaults."""
    base = dict(
        contract_id=37,
        side=int(Side.BUY),
        price=50.00,
        quantity=1.0,
        order_id=1,
        agent_id=1,
        timestamp=0,
        exec_restriction=int(ExecRestriction.NON),
        validity=int(Validity.GFS),
        expiry_time=0,
        status=int(Status.ACTIVE),
        peak_qty=0.0,
        ppd=0.0,
    )
    base.update(overrides)
    return make_order(**base)


def _mk_iceberg(**overrides) -> OMIEOrder:
    """Build a default-valid ICE order — total qty 5, peak 1."""
    base = dict(
        contract_id=37,
        side=int(Side.BUY),
        price=50.00,
        quantity=5.0,
        order_id=1,
        agent_id=1,
        timestamp=0,
        exec_restriction=int(ExecRestriction.ICE),
        validity=int(Validity.GFS),
        expiry_time=0,
        status=int(Status.ACTIVE),
        peak_qty=1.0,
        ppd=0.0,
    )
    base.update(overrides)
    return make_order(**base)


def _module_ast(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _walk_function_defs(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            yield node


def _walk_class_defs(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node


def _module_function(path: pathlib.Path, name: str) -> ast.FunctionDef:
    for node in _walk_function_defs(_module_ast(path)):
        if node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in {path}")


# ---------------------------------------------------------------------------
# Dimension 1. End-to-end determinism / reproducibility
# ---------------------------------------------------------------------------


def _state_leaves(state) -> list:
    return jax.tree.leaves(state)


def _states_bitwise_equal(s1, s2) -> bool:
    leaves1 = _state_leaves(s1)
    leaves2 = _state_leaves(s2)
    if len(leaves1) != len(leaves2):
        return False
    for a, b in zip(leaves1, leaves2):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not bool(jnp.all(a == b)):
            return False
    return True


def _result_leaves_equal(r1: SubmitResult, r2: SubmitResult) -> bool:
    return all(
        bool(jnp.all(jnp.asarray(a) == jnp.asarray(b)))
        for a, b in zip(r1, r2)
    )


class TestEndToEndDeterminism:
    """Two independent runs of the same submit_order sequence on the same
    initial state must yield bitwise-identical state and result trees.
    """

    @pytest.mark.parametrize(
        "exec_restriction",
        [
            int(ExecRestriction.NON),
            int(ExecRestriction.IOC),
            int(ExecRestriction.FOK),
            int(ExecRestriction.ICE),
        ],
        ids=["NON", "IOC", "FOK", "ICE"],
    )
    def test_two_runs_same_sequence_yield_identical_state(
        self, book_open, empty_state, exec_restriction
    ):
        if exec_restriction == int(ExecRestriction.ICE):
            o = _mk_iceberg(order_id=11)
        else:
            o = _mk_order(order_id=11, exec_restriction=exec_restriction)
        s1, r1 = book_open.submit_order(empty_state, o)
        s2, r2 = book_open.submit_order(empty_state, o)
        assert _states_bitwise_equal(s1, s2)
        assert _result_leaves_equal(r1, r2)

    def test_two_runs_full_sequence_yield_identical_state(
        self, book_open, empty_state
    ):
        """Submit 5 orders in sequence twice; final state must match."""

        def run_sequence(state):
            for i, er in enumerate(
                [
                    int(ExecRestriction.NON),
                    int(ExecRestriction.IOC),
                    int(ExecRestriction.FOK),
                    int(ExecRestriction.NON),
                    int(ExecRestriction.NON),
                ]
            ):
                # Mix buy/sell, varying prices so some cross and some rest.
                side = int(Side.BUY) if i % 2 == 0 else int(Side.SELL)
                price = 50.0 + (i * 0.5)
                o = _mk_order(
                    order_id=20 + i,
                    side=side,
                    price=price,
                    quantity=1.0,
                    exec_restriction=er,
                    timestamp=i,
                )
                state, _ = book_open.submit_order(state, o)
            return state

        s1 = run_sequence(empty_state)
        s2 = run_sequence(empty_state)
        assert _states_bitwise_equal(s1, s2)

    def test_iceberg_sequence_deterministic_across_runs(
        self, book_open, empty_state
    ):
        """Iceberg-with-ppd path is the most stateful — verify determinism."""

        def run(state):
            o1 = _mk_iceberg(order_id=30, quantity=5.0, peak_qty=1.0, price=50.0)
            o2 = _mk_iceberg(order_id=31, quantity=3.0, peak_qty=0.5, ppd=0.01,
                             price=49.5, timestamp=1)
            state, _ = book_open.submit_order(state, o1)
            state, _ = book_open.submit_order(state, o2)
            return state

        s1 = run(empty_state)
        s2 = run(empty_state)
        assert _states_bitwise_equal(s1, s2)


# ---------------------------------------------------------------------------
# Dimension 2. Cross-module pytree invariants
# ---------------------------------------------------------------------------


class TestPytreeInvariants:
    """Every flattened leaf of every state pytree in the integrated stack
    must be a ``jnp.ndarray`` (no Python int/float/bool, no dict/None/list).
    """

    def test_empty_state_leaves_are_all_jnp_ndarray(self, empty_state):
        leaves = jax.tree.leaves(empty_state)
        assert len(leaves) >= 1
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray), (
                f"OMIEBookState leaf type is {type(leaf).__name__}, "
                "expected jnp.ndarray"
            )

    def test_empty_state_has_no_python_scalar_leaves(self, empty_state):
        """Python int/float/bool would be silently treated as leaves and
        would not specialize correctly under jit."""
        leaves = jax.tree.leaves(empty_state)
        for leaf in leaves:
            assert not isinstance(leaf, (int, float, bool)), (
                f"OMIEBookState contains Python scalar leaf: {leaf!r}"
            )

    def test_empty_state_has_no_none_leaves(self, empty_state):
        flat, _ = jax.tree.flatten(empty_state)
        # `flatten` filters None nodes; leaves should never contain None.
        for leaf in flat:
            assert leaf is not None

    def test_iceberg_meta_pytree_structure_is_an_ndarray_leaf(self, empty_state):
        # iceberg_meta is the bare jnp.ndarray field; tree-flatten over it
        # alone should produce exactly one leaf, itself.
        flat = jax.tree.leaves(empty_state.iceberg_meta)
        assert len(flat) == 1
        assert flat[0] is empty_state.iceberg_meta

    def test_lobstate_pytree_leaves_all_jnp_ndarray(self, empty_state):
        for leaf in jax.tree.leaves(empty_state.lob):
            assert isinstance(leaf, jnp.ndarray)

    @pytest.mark.parametrize(
        "exec_restriction",
        [
            int(ExecRestriction.NON),
            int(ExecRestriction.IOC),
            int(ExecRestriction.FOK),
            int(ExecRestriction.ICE),
        ],
        ids=["NON", "IOC", "FOK", "ICE"],
    )
    def test_submit_result_leaves_are_jnp_ndarray(
        self, book_open, empty_state, exec_restriction
    ):
        if exec_restriction == int(ExecRestriction.ICE):
            o = _mk_iceberg(order_id=40)
        else:
            o = _mk_order(order_id=40, exec_restriction=exec_restriction)
        _, result = book_open.submit_order(empty_state, o)
        for leaf in jax.tree.leaves(result):
            assert isinstance(leaf, jnp.ndarray)

    def test_pytree_structure_stable_across_submit_calls(
        self, book_open, empty_state
    ):
        """The treedef of OMIEBookState must not change shape across
        submit_order calls (a regression here would break vmap / scan
        callers)."""
        treedef_initial = jax.tree.structure(empty_state)
        o = _mk_order(order_id=50)
        state2, _ = book_open.submit_order(empty_state, o)
        treedef_after = jax.tree.structure(state2)
        assert treedef_initial == treedef_after

    def test_pytree_structure_stable_across_all_exec_restrictions(
        self, book_open, empty_state
    ):
        treedef_initial = jax.tree.structure(empty_state)
        for er in [
            int(ExecRestriction.NON),
            int(ExecRestriction.IOC),
            int(ExecRestriction.FOK),
            int(ExecRestriction.ICE),
        ]:
            if er == int(ExecRestriction.ICE):
                o = _mk_iceberg(order_id=60 + er)
            else:
                o = _mk_order(order_id=60 + er, exec_restriction=er)
            state2, _ = book_open.submit_order(empty_state, o)
            assert jax.tree.structure(state2) == treedef_initial, (
                f"treedef changed after submit_order with exec_restriction={er}"
            )


# ---------------------------------------------------------------------------
# Dimension 3. dtype invariants of integrated state
# ---------------------------------------------------------------------------


class TestStateDtypeInvariants:
    """Every leaf of the integrated OMIEBookState carries a fixed dtype.
    Catches silent f32 leakage and accidental int64 promotion."""

    def test_iceberg_meta_dtype_is_int32(self, empty_state):
        assert empty_state.iceberg_meta.dtype == jnp.int32

    def test_ts_counter_dtype_is_int32(self, empty_state):
        assert jnp.asarray(empty_state.ts_counter).dtype == jnp.int32

    def test_truncated_dtype_is_bool(self, empty_state):
        assert jnp.asarray(empty_state.truncated).dtype == jnp.bool_

    def test_lob_asks_dtype_is_int32(self, empty_state):
        assert empty_state.lob.asks.dtype == jnp.int32

    def test_lob_bids_dtype_is_int32(self, empty_state):
        assert empty_state.lob.bids.dtype == jnp.int32

    def test_lob_trades_dtype_is_int32(self, empty_state):
        assert empty_state.lob.trades.dtype == jnp.int32

    @pytest.mark.parametrize(
        "exec_restriction",
        [
            int(ExecRestriction.NON),
            int(ExecRestriction.IOC),
            int(ExecRestriction.FOK),
            int(ExecRestriction.ICE),
        ],
        ids=["NON", "IOC", "FOK", "ICE"],
    )
    def test_state_dtypes_preserved_after_submit(
        self, book_open, empty_state, exec_restriction
    ):
        if exec_restriction == int(ExecRestriction.ICE):
            o = _mk_iceberg(order_id=70)
        else:
            o = _mk_order(order_id=70, exec_restriction=exec_restriction)
        state2, _ = book_open.submit_order(empty_state, o)
        assert state2.iceberg_meta.dtype == jnp.int32
        assert jnp.asarray(state2.ts_counter).dtype == jnp.int32
        assert jnp.asarray(state2.truncated).dtype == jnp.bool_
        assert state2.lob.asks.dtype == jnp.int32
        assert state2.lob.bids.dtype == jnp.int32
        assert state2.lob.trades.dtype == jnp.int32


# ---------------------------------------------------------------------------
# Dimension 4. SubmitResult field stability across the five variants
# ---------------------------------------------------------------------------


class TestSubmitResultStability:
    """SubmitResult pytree must have exactly 5 leaves and identical
    treedef across all 5 ExecRestriction variants (NON / IOC / FOK / ICE
    / ICEwPI)."""

    @pytest.fixture(scope="class")
    def results_by_variant(self, book_open, empty_state):
        out = {}
        for label, er, mk in [
            ("NON", int(ExecRestriction.NON), lambda i: _mk_order(order_id=i)),
            ("IOC", int(ExecRestriction.IOC), lambda i: _mk_order(
                order_id=i, exec_restriction=int(ExecRestriction.IOC))),
            ("FOK", int(ExecRestriction.FOK), lambda i: _mk_order(
                order_id=i, exec_restriction=int(ExecRestriction.FOK))),
            ("ICE", int(ExecRestriction.ICE), lambda i: _mk_iceberg(order_id=i)),
            ("ICEwPI", int(ExecRestriction.ICE), lambda i: _mk_iceberg(
                order_id=i, ppd=0.01)),
        ]:
            _, r = book_open.submit_order(empty_state, mk(80 + er))
            out[label] = r
        return out

    @pytest.mark.parametrize(
        "label", ["NON", "IOC", "FOK", "ICE", "ICEwPI"]
    )
    def test_each_variant_returns_submit_result(self, results_by_variant, label):
        r = results_by_variant[label]
        assert isinstance(r, SubmitResult), (
            f"variant {label} returned {type(r).__name__}, expected SubmitResult"
        )

    @pytest.mark.parametrize(
        "label", ["NON", "IOC", "FOK", "ICE", "ICEwPI"]
    )
    def test_submit_result_has_exactly_five_leaves(
        self, results_by_variant, label
    ):
        r = results_by_variant[label]
        leaves = jax.tree.leaves(r)
        assert len(leaves) == 5

    def test_treedef_is_identical_across_all_five_variants(
        self, results_by_variant
    ):
        ref = jax.tree.structure(results_by_variant["NON"])
        for label in ("IOC", "FOK", "ICE", "ICEwPI"):
            assert jax.tree.structure(results_by_variant[label]) == ref, (
                f"SubmitResult treedef diverged at variant {label}"
            )

    @pytest.mark.parametrize(
        "label", ["NON", "IOC", "FOK", "ICE", "ICEwPI"]
    )
    def test_accepted_is_bool_scalar(self, results_by_variant, label):
        r = results_by_variant[label]
        arr = jnp.asarray(r.accepted)
        assert arr.dtype == jnp.bool_
        assert arr.shape == ()

    @pytest.mark.parametrize(
        "label", ["NON", "IOC", "FOK", "ICE", "ICEwPI"]
    )
    def test_traded_is_bool_scalar(self, results_by_variant, label):
        r = results_by_variant[label]
        arr = jnp.asarray(r.traded)
        assert arr.dtype == jnp.bool_
        assert arr.shape == ()

    @pytest.mark.parametrize("field", ["trade_qty", "trade_price", "settlement_value"])
    @pytest.mark.parametrize(
        "label", ["NON", "IOC", "FOK", "ICE", "ICEwPI"]
    )
    def test_float_fields_are_float32_scalars(
        self, results_by_variant, label, field
    ):
        r = results_by_variant[label]
        arr = jnp.asarray(getattr(r, field))
        assert arr.dtype == jnp.float32
        assert arr.shape == ()


# ---------------------------------------------------------------------------
# Dimension 5. B1-B5 constants are bare Python ints (not jnp arrays)
# ---------------------------------------------------------------------------


class TestB1ToB5ConstantsArePythonInts:
    """The Phase 2 shape constants live in ``jax_constants.py`` as bare
    Python ``int`` values so they specialize statically when used as
    array shapes or ``static_argnums``. A regression to ``jnp.int32``
    would silently break ``static_argnums`` callers."""

    def test_max_icebergs_is_python_int(self):
        assert type(MAX_ICEBERGS) is int, (
            f"MAX_ICEBERGS type is {type(MAX_ICEBERGS).__name__}, "
            "must be Python int"
        )

    def test_max_slices_per_regen_is_python_int(self):
        assert type(MAX_SLICES_PER_REGEN) is int

    def test_max_agents_is_python_int(self):
        assert type(MAX_AGENTS) is int

    def test_sentinel_order_id_is_python_int(self):
        assert type(SENTINEL_ORDER_ID) is int

    def test_sentinel_agent_id_is_python_int(self):
        assert type(SENTINEL_AGENT_ID) is int

    def test_constants_are_not_bool(self):
        """``bool`` is an ``int`` subclass; must be excluded explicitly."""
        for v in (MAX_ICEBERGS, MAX_SLICES_PER_REGEN, MAX_AGENTS,
                  SENTINEL_ORDER_ID, SENTINEL_AGENT_ID):
            assert not isinstance(v, bool)

    def test_constants_values_match_documented_decisions(self):
        # Pinned values per jax_constants.py module docstring.
        # MAX_ICEBERGS: originally pinned at the literal value ``100``
        # (P2-HOTFIX-era), widened to a sanctioned set by P2-PERF-7
        # (v5.31 agile-plan card). Membership in the set
        # ``{100, 2048, 4096, 8192}`` accepts (a) the pre-P2-PERF-7
        # baseline of 100, (b) the orchestrator-sanctioned values
        # ``{2048, 4096, 8192}`` recovered from the retired
        # ``P2-HOTFIX-iceberg-cap-overflow-FOLLOWUP`` story
        # (commit ``0a89cfa``), and (c) the P2-PERF-7 pinned target
        # value 8192. Any other value indicates an unauthorised drift.
        assert MAX_ICEBERGS in {100, 2048, 4096, 8192}, (
            f"MAX_ICEBERGS={MAX_ICEBERGS} not in sanctioned set "
            f"{{100, 2048, 4096, 8192}} "
            f"(P2-PERF-7 v5.31 / agile-plan)"
        )
        assert MAX_SLICES_PER_REGEN == 64
        assert MAX_AGENTS == 512
        assert SENTINEL_ORDER_ID == -1
        assert SENTINEL_AGENT_ID == -1

    def test_constants_are_hashable(self):
        """Python ``int`` values are hashable — they can be used as
        ``static_argnums`` argument values and as dict keys."""
        d = {MAX_ICEBERGS: 0, MAX_SLICES_PER_REGEN: 1, MAX_AGENTS: 2}
        assert len(d) == 3


# ---------------------------------------------------------------------------
# Dimension 6. Phase 2 closure invariants from migration-plan.md
# ---------------------------------------------------------------------------


class TestMigrationPlanPhase2Invariants:
    """The migration-plan §Phase 2 Acceptance Criteria + §Loader dtype
    contract make specific factual claims about the production code.
    Each test here pins one of those claims directly. A drift means
    either the implementation regressed or the migration plan is stale."""

    def test_omie_book_state_iceberg_meta_is_jnp_ndarray_field(self, empty_state):
        # AC (4)
        assert isinstance(empty_state.iceberg_meta, jnp.ndarray)

    def test_omie_book_state_has_no_dict_or_none_leaves(self, empty_state):
        # AC (5)
        for leaf in jax.tree.leaves(empty_state):
            assert not isinstance(leaf, dict)
            assert leaf is not None

    def test_encoding_module_retains_decode_price_and_decode_qty(self):
        # AC (8) re: migration-plan.md:814 — "encoding.py retains
        # decode_price / decode_qty for the Python boundary".
        encoding = importlib.import_module("omiecid_simulator.omiecid.encoding")
        assert hasattr(encoding, "decode_price")
        assert hasattr(encoding, "decode_qty")
        assert callable(encoding.decode_price)
        assert callable(encoding.decode_qty)

    def test_encoding_module_retains_encode_price_and_encode_qty(self):
        encoding = importlib.import_module("omiecid_simulator.omiecid.encoding")
        assert hasattr(encoding, "encode_price")
        assert hasattr(encoding, "encode_qty")

    def test_load_omie_agent_metadata_remains_python_side(self):
        """AC (7) — `id_to_pair` lookup is Python-side (not JIT-traced)."""
        from omiecid_simulator.jaxomiecid import omie_data_loader
        f = omie_data_loader.load_omie_agent_metadata
        # A jit-wrapped function exposes _cache_size / lower; a Python
        # function does not.
        assert not hasattr(f, "_cache_size"), (
            "load_omie_agent_metadata is jit-wrapped — must stay Python-side "
            "per AC (8)"
        )

    def test_submit_order_module_level_lift_exists(self):
        """AC (3) — `_submit_order = jax.jit(...)` at module scope."""
        from omiecid_simulator.omiecid import contract_book
        assert hasattr(contract_book, "_submit_order")
        assert callable(contract_book._submit_order)
        assert hasattr(contract_book._submit_order, "_cache_size")

    def test_submit_order_body_is_separate_from_lift(self):
        """The un-jitted body must coexist with the lift so audits and
        debug callers can introspect."""
        from omiecid_simulator.omiecid import contract_book
        assert hasattr(contract_book, "_submit_order_body")
        assert contract_book._submit_order_body is not contract_book._submit_order

    def test_get_order_returns_sentinel_not_none(self, book_open, empty_state):
        """B3 — `get_order` returns SENTINEL_ORDER_ID, not Python None."""
        # Lookup an order_id that has never been submitted.
        result = book_open.get_order(empty_state, order_id=99999, side=int(Side.BUY))
        assert result is not None, (
            "get_order returned Python None — B3 sentinel contract violated"
        )
        assert isinstance(result, OMIEOrder)
        assert int(result.order_id) == SENTINEL_ORDER_ID

    def test_make_sentinel_order_discriminator_is_order_id(self):
        s = make_sentinel_order()
        assert int(s.order_id) == SENTINEL_ORDER_ID
        # zero-ambiguous fields must NOT be relied on for discriminator
        # purposes; the docstring guarantees the order_id field carries it.
        assert int(s.agent_id) == SENTINEL_AGENT_ID

    def test_ppd_signed_int_ticks_rule_documented(self):
        """migration-plan.md:812 pins `ppd` as signed-int-ticks (NOT
        offset-encoded). We assert the documentation contains the rule;
        production code's `_encode_ppd_int` uses `round(ppd * scale)` —
        no `+ price_offset` term. AST-check that the function does not
        add a price_offset."""
        ice_src = ICEBERG_MANAGER_PATH.read_text()
        # Locate the _encode_ppd_int method body.
        tree = ast.parse(ice_src)
        found = False
        for cls in _walk_class_defs(tree):
            if cls.name != "IcebergManager":
                continue
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_encode_ppd_int":
                    body_src = ast.unparse(fn)
                    found = True
                    assert "price_offset" not in body_src, (
                        "_encode_ppd_int adds price_offset — violates "
                        "migration-plan.md:812 (ppd is signed-int-ticks, "
                        "not offset-encoded)"
                    )
        assert found, "_encode_ppd_int not located in IcebergManager"


# ---------------------------------------------------------------------------
# Dimension 7. AST invariants on the integration surface
# ---------------------------------------------------------------------------


class TestASTInvariantsOnDomainModules:
    """No instance-method jit; no Python for/while in jitted bodies; no
    dict/list constructors inside jit scope; no `isinstance(x, float)`
    or `x is None` patterns inside jitted bodies."""

    @pytest.mark.parametrize(
        "module_path",
        JIT_DOMAIN_MODULES,
        ids=[p.name for p in JIT_DOMAIN_MODULES],
    )
    def test_no_jit_decorator_on_any_bound_method(self, module_path):
        """Class methods (instance or class) carry zero `@jax.jit` /
        `@partial(jax.jit, ...)` decorators. The lift always sits at
        module scope (canonical Phase 2 idiom)."""
        tree = _module_ast(module_path)
        offenders: List[str] = []
        for cls in _walk_class_defs(tree):
            for item in cls.body:
                if isinstance(item, ast.FunctionDef):
                    for d in item.decorator_list:
                        src = ast.unparse(d)
                        if "jax.jit" in src or src == "jit":
                            offenders.append(
                                f"{module_path.name}::{cls.name}.{item.name}: {src}"
                            )
        assert offenders == [], (
            f"jit decorator on bound method(s): {offenders}. P2-H2 / P2-H3 "
            "require module-level lifts only."
        )

    def test_no_python_for_in_lifted_submit_helpers(self):
        """The lifted submit_order helpers are pure jit bodies; Python
        `for` would unroll under tracing and inflate compile time."""
        for fn_name in [
            "_submit_order_body",
            "_non_or_ioc_path",
            "_fok_path",
            "_ice_path",
            "_while_loop_match",
            "_while_loop_match_fok",
            "_build_state_jit",
        ]:
            node = _module_function(CONTRACT_BOOK_PATH, fn_name)
            for sub in ast.walk(node):
                if isinstance(sub, ast.For):
                    raise AssertionError(
                        f"{fn_name} contains a Python `for` loop at line {sub.lineno}"
                    )

    def test_no_python_while_in_lifted_submit_helpers(self):
        """No Python `while` either — must be `lax.while_loop`."""
        for fn_name in [
            "_submit_order_body",
            "_non_or_ioc_path",
            "_fok_path",
            "_ice_path",
            "_while_loop_match",
            "_while_loop_match_fok",
            "_build_state_jit",
        ]:
            node = _module_function(CONTRACT_BOOK_PATH, fn_name)
            for sub in ast.walk(node):
                if isinstance(sub, ast.While):
                    raise AssertionError(
                        f"{fn_name} contains a Python `while` loop at line {sub.lineno}"
                    )

    def test_no_isinstance_float_or_int_in_lifted_submit_body(self):
        """`isinstance(x, float)` inside a jitted body would fail —
        traced values are JAX arrays. Even if the call is guarded by an
        eager check, it would break under jit-trace expansion."""
        body_src = ast.unparse(_module_function(CONTRACT_BOOK_PATH, "_submit_order_body"))
        assert "isinstance" not in body_src or (
            "isinstance" in body_src and "float" not in body_src
        ), "_submit_order_body contains isinstance(..., float)"
        # Stricter — the `int` form is also forbidden inside the jit body.
        assert not re.search(r"isinstance\([^,]+,\s*int\b", body_src)
        assert not re.search(r"isinstance\([^,]+,\s*float\b", body_src)

    def test_no_is_none_pattern_in_lifted_submit_body(self):
        """`x is None` patterns are forbidden inside jit scope — use
        sentinel comparisons (`order_id == SENTINEL_ORDER_ID`) instead."""
        body_src = ast.unparse(_module_function(CONTRACT_BOOK_PATH, "_submit_order_body"))
        assert "is None" not in body_src

    def test_no_is_none_pattern_in_lifted_branch_paths(self):
        for fn_name in ["_non_or_ioc_path", "_fok_path", "_ice_path"]:
            body_src = ast.unparse(_module_function(CONTRACT_BOOK_PATH, fn_name))
            assert "is None" not in body_src, (
                f"`is None` pattern present in jitted body {fn_name}"
            )

    def test_no_dict_constructor_in_lifted_submit_body(self):
        """Python `dict(...)` calls inside jit scope would produce a
        non-pytree object — must use NamedTuple."""
        body_src = ast.unparse(_module_function(CONTRACT_BOOK_PATH, "_submit_order_body"))
        # The `dict` call would appear as `dict(...)` or `{...}` ast.Dict.
        node = _module_function(CONTRACT_BOOK_PATH, "_submit_order_body")
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name) and sub.func.id == "dict":
                    raise AssertionError(
                        f"dict(...) call in _submit_order_body at line {sub.lineno}"
                    )
            if isinstance(sub, ast.Dict):
                raise AssertionError(
                    f"`{{...}}` dict literal in _submit_order_body at line {sub.lineno}"
                )

    def test_no_list_constructor_in_lifted_submit_body(self):
        """`list(...)` would produce a non-pytree object — must use
        jnp.ndarray or jnp.stack."""
        node = _module_function(CONTRACT_BOOK_PATH, "_submit_order_body")
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name) and sub.func.id == "list":
                    raise AssertionError(
                        f"list(...) call in _submit_order_body at line {sub.lineno}"
                    )

    def test_no_print_or_breakpoint_calls_in_jit_domain_modules(self):
        """Debugger / print residue must never ship to a jit-targeted
        module — under tracing they materialise the traced value, breaking
        the jit cache key."""
        for path in JIT_DOMAIN_MODULES:
            tree = _module_ast(path)
            for sub in ast.walk(tree):
                if isinstance(sub, ast.Call):
                    func_src = (
                        ast.unparse(sub.func)
                        if not isinstance(sub.func, ast.Name)
                        else sub.func.id
                    )
                    assert func_src not in ("print", "breakpoint"), (
                        f"{path.name} contains call to {func_src}() at line "
                        f"{sub.lineno}"
                    )


# ---------------------------------------------------------------------------
# Dimension 8. Trace-cache hygiene across the full domain
# ---------------------------------------------------------------------------


class TestTraceCacheHygiene:
    """After exercising the public submit_order API across all five
    ExecRestriction variants on a single contract, the cache size of the
    module-level `_submit_order` is exactly 1 (one compile covers the
    full domain)."""

    def test_one_compile_covers_all_five_exec_restriction_variants(
        self, product_cfg, market_cfg
    ):
        """Use a fresh contract to keep cache attribution clean. The
        module-level cache is shared, but the count delta is what we
        measure."""
        contract = Contract(
            contract_id=99,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.OPEN,
        )
        book = ContractBook(contract, product_cfg, market_cfg)
        state = book.init()

        before = _submit_order._cache_size()
        # Five rows; ICE arm taken twice (once ppd=0 ICE, once ppd!=0 ICEwPI).
        variants = [
            _mk_order(order_id=1, exec_restriction=int(ExecRestriction.NON)),
            _mk_order(order_id=2, exec_restriction=int(ExecRestriction.IOC)),
            _mk_order(order_id=3, exec_restriction=int(ExecRestriction.FOK)),
            _mk_iceberg(order_id=4),
            _mk_iceberg(order_id=5, ppd=0.01),
        ]
        for o in variants:
            state, _ = book.submit_order(state, o)

        after = _submit_order._cache_size()
        delta = after - before
        assert delta <= 1, (
            f"Module-level _submit_order acquired {delta} new cache entries "
            "after exercising all 5 ExecRestriction variants on one "
            "(product_cfg, market_cfg) tuple — expected at most 1 (one "
            "compile shared by the four-arm lax.switch)."
        )

    def test_iceberg_ppd_variation_does_not_trigger_recompile(
        self, product_cfg, market_cfg
    ):
        """`ppd` is a traced order field; varying it within the ICE arm
        must NOT bump the cache count — the ICE/ICEwPI distinction is
        handled inside the jitted body."""
        contract = Contract(
            contract_id=98,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.OPEN,
        )
        book = ContractBook(contract, product_cfg, market_cfg)
        state = book.init()

        # First an ICE order to lock the cache.
        state, _ = book.submit_order(state, _mk_iceberg(order_id=10))
        before = _submit_order._cache_size()

        # Now vary ppd within the ICE arm.
        for i, ppd in enumerate([0.0, 0.01, -0.02, 0.05, -0.1]):
            o = _mk_iceberg(order_id=20 + i, ppd=ppd)
            state, _ = book.submit_order(state, o)

        after = _submit_order._cache_size()
        assert after == before, (
            f"varying ppd within the ICE arm bumped _submit_order's cache "
            f"count from {before} to {after}"
        )

    def test_qty_variation_does_not_trigger_recompile(
        self, book_open, empty_state
    ):
        """Quantity is traced — variation must not bump cache count."""
        # Warm the cache.
        s, _ = book_open.submit_order(empty_state, _mk_order(order_id=30))
        before = _submit_order._cache_size()

        for i, q in enumerate([0.5, 1.5, 3.0, 7.0, 99.0]):
            s, _ = book_open.submit_order(s, _mk_order(order_id=31 + i, quantity=q))
        after = _submit_order._cache_size()
        assert after == before


# ---------------------------------------------------------------------------
# Dimension 9. Empirical zero-sum settlement via the matching loop
# ---------------------------------------------------------------------------


class TestEmpiricalZeroSumSettlement:
    """Generate a synthetic closed trade tape via the OMIE matching loop
    (NON BUY meets NON SELL), then assert bilateral PnL sum is zero."""

    def _build_trade_tape_from_lob(self, lob, contract_id: int) -> jnp.ndarray:
        """Convert jaxob's `lob.trades` ring buffer into the (M, 6) int32
        trade tape expected by `compute_agent_pnl`.

        jaxob trades layout (cst.TradesFeat): P, Q, AGRS_OID, PASS_OID,
        AGRS_TID, PASS_TID, SEC, NSEC.

        Settlement tape layout (M, 6): [timestamp, price_int_offset,
        qty_tick, buyer_agent_id, seller_agent_id, contract_id].

        We reuse the offset-encoded price column from jaxob trades as the
        offset-encoded price column the settlement tape expects, since both
        flow through the same Method A encoder.
        """
        from omiecid_simulator.jaxob import jaxob_constants as cst

        P = cst.TradesFeat.P.value
        Q = cst.TradesFeat.Q.value
        AGRS_TID = cst.TradesFeat.AGRS_TID.value
        PASS_TID = cst.TradesFeat.PASS_TID.value
        SEC = cst.TradesFeat.SEC.value

        rows = []
        for i in range(lob.trades.shape[0]):
            row = lob.trades[i]
            qty_signed = int(row[Q])  # jaxob signs qty by passive side
            price = int(row[P])
            if qty_signed == 0 or price <= 0:
                continue
            agrs = int(row[AGRS_TID])
            passv = int(row[PASS_TID])
            ts = int(row[SEC])
            # On a BUY-aggressor / SELL-passive cross (qty_signed < 0):
            #   AGRS_TID is the buyer, PASS_TID is the seller.
            # On a SELL-aggressor / BUY-passive cross (qty_signed > 0):
            #   AGRS_TID is the seller, PASS_TID is the buyer.
            if qty_signed < 0:
                buyer, seller = agrs, passv
            else:
                buyer, seller = passv, agrs
            qty = abs(qty_signed)
            rows.append([ts, price, qty, buyer, seller, contract_id])
        if not rows:
            return jnp.zeros((0, 6), dtype=jnp.int32)
        return jnp.asarray(rows, dtype=jnp.int32)

    def test_two_agents_buy_sell_meet_zero_sum(
        self, product_cfg, market_cfg
    ):
        """Agent A submits SELL at 50; agent B submits a crossing BUY at
        50. The crossing trade must produce equal-and-opposite PnL."""
        # Use a fresh, larger book to ensure independent state across runs.
        contract = Contract(
            contract_id=42,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.OPEN,
        )
        book = ContractBook(contract, product_cfg, market_cfg)
        state = book.init()

        aid_a = 7
        aid_b = 11

        # Seller rests first.
        sell = _mk_order(
            order_id=100, agent_id=aid_a, side=int(Side.SELL),
            price=50.00, quantity=2.0, timestamp=0,
        )
        state, _ = book.submit_order(state, sell)

        # Buyer crosses.
        buy = _mk_order(
            order_id=101, agent_id=aid_b, side=int(Side.BUY),
            price=50.00, quantity=2.0, timestamp=1,
        )
        state, _ = book.submit_order(state, buy)

        tape = self._build_trade_tape_from_lob(state.lob, contract_id=42)
        assert tape.shape[0] >= 1, "matching loop produced no trade tape rows"

        # Eager (outside any jit / vmap) path is the sanctioned settlement
        # call site per migration-plan §867 ("eager calls: body opens its
        # own scoped enable_x64 context").
        pnl_a = float(compute_agent_pnl(tape, jnp.int32(aid_a)))
        pnl_b = float(compute_agent_pnl(tape, jnp.int32(aid_b)))

        # Float32 EUR returns — tolerance for float-decode roundoff at the
        # f32 boundary (int64 invariant is exact in tick space).
        assert pnl_a + pnl_b == pytest.approx(0.0, abs=1e-3), (
            f"bilateral PnL non-zero: pnl_a={pnl_a}, pnl_b={pnl_b}, "
            f"sum={pnl_a + pnl_b}"
        )

    def test_uninvolved_third_agent_has_zero_pnl(
        self, product_cfg, market_cfg
    ):
        """An agent who never appears on the trade tape must have zero PnL."""
        contract = Contract(
            contract_id=43,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.OPEN,
        )
        book = ContractBook(contract, product_cfg, market_cfg)
        state = book.init()

        state, _ = book.submit_order(state, _mk_order(
            order_id=200, agent_id=1, side=int(Side.SELL),
            price=50.00, quantity=1.0, timestamp=0,
        ))
        state, _ = book.submit_order(state, _mk_order(
            order_id=201, agent_id=2, side=int(Side.BUY),
            price=50.00, quantity=1.0, timestamp=1,
        ))
        tape = self._build_trade_tape_from_lob(state.lob, contract_id=43)

        # Agent 999 never traded — must be zero EUR regardless of tape size.
        pnl_third = float(compute_agent_pnl(tape, jnp.int32(999)))
        assert pnl_third == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dimension 10. Truncation backstop — under-cap must stay `truncated=False`
# ---------------------------------------------------------------------------


class TestTruncationBackstopUnderCap:
    """Per the architect's pin 1 on this audit: the smoke harness must NOT
    exercise the >100 iceberg path (it's deferred to
    P2-HOTFIX-iceberg-cap-overflow-FOLLOWUP). We probe the *under-cap*
    direction: with N icebergs <= MAX_ICEBERGS, `state.truncated` stays
    False."""

    def test_empty_state_truncated_is_false(self, empty_state):
        assert bool(jnp.asarray(empty_state.truncated)) is False

    def test_single_ice_submit_does_not_set_truncated(
        self, book_open, empty_state
    ):
        o = _mk_iceberg(order_id=400)
        state2, _ = book_open.submit_order(empty_state, o)
        assert bool(jnp.asarray(state2.truncated)) is False

    def test_under_cap_iceberg_burst_stays_under_truncated_flag(
        self, product_cfg, market_cfg
    ):
        """Submit 20 distinct icebergs (< MAX_ICEBERGS=100, well under
        book_depth=100 / 2 sides). truncated must remain False."""
        contract = Contract(
            contract_id=44,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.OPEN,
        )
        book = ContractBook(contract, product_cfg, market_cfg)
        state = book.init()

        # 20 BUY icebergs at distinct prices (so none cross or rest at the
        # same level). Each one carries a peak < total to actually drain
        # the hidden lifecycle through `register`.
        for i in range(20):
            price = 40.0 + (i * 0.1)
            o = _mk_iceberg(
                order_id=500 + i, agent_id=1, side=int(Side.BUY),
                price=price, quantity=2.0, peak_qty=0.5, timestamp=i,
            )
            state, _ = book.submit_order(state, o)

        assert bool(jnp.asarray(state.truncated)) is False, (
            "truncated flag spuriously set under MAX_ICEBERGS cap "
            "(20 registrations of 100 capacity)"
        )

    def test_truncation_flag_is_sticky_once_set(self, empty_state):
        """If `prior_truncated` is True, the next state-build must keep
        it True (no clear-on-empty). We construct a synthetic state with
        truncated=True and pass it through _build_state."""
        cfg = OMIEProductConfig()
        ob = ContractBook(
            Contract(45, 0, 15, -60, ContractStatus.OPEN),
            cfg, OMIEMarketConfig(),
        )
        # Build a fake "after-submit" state with truncated=True.
        # P2-PERF-6: OMIEBookState now carries n_active alongside
        # iceberg_meta — propagate the empty_state's n_active (== 0)
        # since the meta is also the empty/sentinel array.
        state_truncated = OMIEBookState(
            lob=empty_state.lob,
            iceberg_meta=empty_state.iceberg_meta,
            ts_counter=empty_state.ts_counter,
            truncated=jnp.bool_(True),
            n_active=empty_state.n_active,
        )
        # Submit a NON order — truncated must remain True.
        new_state, _ = ob.submit_order(
            state_truncated, _mk_order(order_id=600),
        )
        assert bool(jnp.asarray(new_state.truncated)) is True


# ---------------------------------------------------------------------------
# Dimension 11. `load_omie_agent_metadata` contract (AC 8 deep probe)
# ---------------------------------------------------------------------------


class TestLoadOmieAgentMetadataContract:
    """AC (8): `id_to_pair` is accessible outside JIT for settlement
    aggregation and diagnostics. Beyond "the function exists", we probe
    its signature and (data-presence permitting) the structure of its
    return value on canonical 2026-01-12 cid=81 data."""

    def test_function_is_importable(self):
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        assert callable(load_omie_agent_metadata)

    def test_function_signature_takes_config_returns_dict(self):
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        sig = inspect.signature(load_omie_agent_metadata)
        params = list(sig.parameters)
        assert params == ["config"], (
            f"load_omie_agent_metadata signature is {sig}, expected "
            "(config: OMIEDataConfig)"
        )
        # Return annotation: dict[int, tuple[str, str]] (string form
        # acceptable per the PEP 563 deferred-eval convention).
        ra = sig.return_annotation
        ra_str = str(ra)
        assert "dict" in ra_str.lower() or ra_str.startswith("dict"), (
            f"return annotation lacks `dict`: {ra_str!r}"
        )

    def test_function_is_not_jit_wrapped(self):
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        # A jit wrapper exposes `_cache_size`; a Python function does not.
        assert not hasattr(load_omie_agent_metadata, "_cache_size")
        assert not hasattr(load_omie_agent_metadata, "lower")

    def test_metadata_dict_structure_on_canonical_data(self):
        """Probe the actual return value on canonical 2026-01-12 cid=81
        data. Skip if the raw OMIE file is absent on this machine."""
        raw_path = REPO_ROOT / "data" / "rawOMIE" / "orders_20260112.1"
        if not raw_path.is_file():
            pytest.skip(f"raw OMIE file absent: {raw_path}")
        from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        cfg = OMIEDataConfig(
            data_path=str(REPO_ROOT / "data" / "rawOMIE"),
            contract_id=81,
            delivery_date="2026-01-12",
        )
        id_to_pair = load_omie_agent_metadata(cfg)
        assert isinstance(id_to_pair, dict)
        assert len(id_to_pair) >= 1, "id_to_pair is empty on canonical data"
        # Keys are 1-indexed contiguous integers per migration-plan
        # §Phase 1.5 (line 356-357 of omie_data_loader.py: "Keys form [1, N]").
        keys = sorted(id_to_pair.keys())
        assert keys[0] == 1, (
            f"first key is {keys[0]}, expected 1-indexed start"
        )
        assert keys == list(range(1, max(keys) + 1)), (
            "id_to_pair keys are not a contiguous 1-indexed range"
        )
        # Values are (str, str) tuples.
        for v in id_to_pair.values():
            assert isinstance(v, tuple)
            assert len(v) == 2
            assert isinstance(v[0], str)
            assert isinstance(v[1], str)


# ---------------------------------------------------------------------------
# Dimension 12. Module-level jit idiom conformance
# ---------------------------------------------------------------------------


class TestModuleLevelJitIdiomConformance:
    """Per migration-plan v5.11 §"Module-level jit idiom": the canonical
    lift uses `static_argnames=(...)` as a tuple, NOT `static_argnums`.
    AST-probe `contract_book.py` to confirm the lift matches this shape
    exactly."""

    def test_contract_book_submit_order_lift_uses_static_argnames(self):
        tree = _module_ast(CONTRACT_BOOK_PATH)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "_submit_order"
                for t in node.targets
            ):
                continue
            assert isinstance(node.value, ast.Call)
            call = node.value
            kw_names = [k.arg for k in call.keywords]
            assert "static_argnames" in kw_names
            assert "static_argnums" not in kw_names, (
                "_submit_order lift uses numeric static_argnums — "
                "canonical idiom requires static_argnames"
            )
            return
        raise AssertionError("_submit_order assignment not found in contract_book.py")

    def test_contract_book_submit_order_static_argnames_is_tuple(self):
        tree = _module_ast(CONTRACT_BOOK_PATH)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "_submit_order"
                for t in node.targets
            ):
                continue
            call = node.value
            sa = next(k for k in call.keywords if k.arg == "static_argnames")
            assert isinstance(sa.value, ast.Tuple), (
                f"static_argnames value is {type(sa.value).__name__}; "
                "canonical idiom requires a tuple even for single entries"
            )
            elts = [
                ast.literal_eval(e) for e in sa.value.elts
                if isinstance(e, ast.Constant)
            ]
            assert elts == ["product_config", "market_config"], (
                f"static_argnames tuple has unexpected contents: {elts}"
            )
            return
        raise AssertionError("_submit_order assignment not found")

    def test_jorderbook_module_level_jits_use_static_argnames(self):
        """The precedent jorderbook.py lift also uses `static_argnames=...`
        per migration-plan v5.11. Probe a sample of decorators."""
        jorderbook_path = (
            REPO_ROOT / "omiecid_simulator" / "jaxob" / "jorderbook.py"
        )
        tree = _module_ast(jorderbook_path)
        partial_jit_decorators_found = 0
        offenders: List[str] = []
        for fn in _walk_function_defs(tree):
            for d in fn.decorator_list:
                if not isinstance(d, ast.Call):
                    continue
                # We only care about `partial(jax.jit, ...)` decorators —
                # the canonical Phase 2 idiom shape.
                func_src = ast.unparse(d.func)
                if func_src != "partial":
                    continue
                if not d.args:
                    continue
                first_arg_src = ast.unparse(d.args[0])
                if first_arg_src != "jax.jit":
                    continue
                kw_names = [k.arg for k in d.keywords]
                if "static_argnums" in kw_names:
                    offenders.append(f"{fn.name}: static_argnums detected")
                if "static_argnames" in kw_names:
                    partial_jit_decorators_found += 1
        # Canonical idiom states 15 of 18 carry static_argnames; we don't
        # pin the exact count but assert at least 5 are seen and none use
        # the forbidden numeric form.
        assert partial_jit_decorators_found >= 5, (
            f"Only {partial_jit_decorators_found} partial(jax.jit, "
            "static_argnames=...) decorators found in jorderbook.py — "
            "regression vs the v5.11 idiom?"
        )
        assert offenders == [], (
            f"jorderbook.py contains static_argnums (forbidden): {offenders}"
        )


# ---------------------------------------------------------------------------
# Dimension 14. Phase 3 surface untouched
# ---------------------------------------------------------------------------


class TestPhase3SurfaceUntouched:
    """`omiecid_simulator/jaxen/` and `omiecid_simulator/jaxrl/` are unmodified
    relative to `main` at the P2-H2 merge SHA (0e0855e). This is the
    deferral-respect probe — P2-H3 is the acceptance gate, not a Phase 3
    jumpstart.

    NOTE: We cannot reliably probe diffs against a specific historical SHA
    at audit time (the SHA may not be reachable from a shallow clone),
    so we probe a weaker but still meaningful invariant: the jaxen and
    jaxrl trees do NOT import any P2-H3 production surface that did not
    exist before. Specifically, neither tree imports `_submit_order`
    from contract_book (a P2-H2 deliverable) — they should still use
    the public `ContractBook.submit_order` method via instance call, or
    not use it at all in Phase 2.
    """

    def test_no_jaxen_file_imports_module_level_lift(self):
        jaxen_dir = REPO_ROOT / "omiecid_simulator" / "jaxen"
        if not jaxen_dir.is_dir():
            pytest.skip("jaxen/ tree absent on this machine")
        for py_path in jaxen_dir.rglob("*.py"):
            src = py_path.read_text()
            assert "from omiecid_simulator.omiecid.contract_book import _submit_order" not in src, (
                f"{py_path.relative_to(REPO_ROOT)} imports _submit_order — "
                "Phase 3 must use the public ContractBook trampoline"
            )

    def test_no_jaxrl_file_imports_module_level_lift(self):
        jaxrl_dir = REPO_ROOT / "omiecid_simulator" / "jaxrl"
        if not jaxrl_dir.is_dir():
            pytest.skip("jaxrl/ tree absent on this machine")
        for py_path in jaxrl_dir.rglob("*.py"):
            src = py_path.read_text()
            assert "from omiecid_simulator.omiecid.contract_book import _submit_order" not in src

    def test_phase3_trees_have_no_p2_h3_jit_constants_drift(self):
        """`jax_constants.py` is a Phase 2 surface — Phase 3 trees must
        not have copies / forks of it. Probe for the specific names."""
        for subtree_name in ("jaxen", "jaxrl"):
            subtree = REPO_ROOT / "omiecid_simulator" / subtree_name
            if not subtree.is_dir():
                continue
            for py_path in subtree.rglob("*.py"):
                src = py_path.read_text()
                # P2-H3-internal constants must not be redefined outside omie/.
                forbidden = [
                    "MAX_ICEBERGS = ",
                    "MAX_SLICES_PER_REGEN = ",
                    "SENTINEL_ORDER_ID = ",
                ]
                for pat in forbidden:
                    assert pat not in src, (
                        f"{py_path.relative_to(REPO_ROOT)} redefines `{pat}` — "
                        "P2-H3 constant must live only in jax_constants.py"
                    )


# ---------------------------------------------------------------------------
# Bonus dimension — cancel_order / get_order Python-side correctness
# (these are documented as out-of-scope for the jit lift but must
# continue to behave correctly per AC (3)'s "cancel" sub-clause).
# ---------------------------------------------------------------------------


class TestCancelOrderRoundTripPythonSide:
    """`ContractBook.cancel_order` is Python-side per the P2-H2 explicit
    deferral; AC (3) requires that it produce the same SubmitResult tree
    on full NON / IOC / FOK / ICE / ICEwPI / **cancel** sequences."""

    def test_cancel_unknown_order_id_is_noop(self, book_open, empty_state):
        """Cancel on a non-existent order_id must return the same state
        (Phase-1 semantics preserved)."""
        state_after = book_open.cancel_order(
            empty_state, order_id=99999, side=int(Side.BUY),
        )
        # Empty book; the cancel branch returns `state` unchanged.
        assert _states_bitwise_equal(state_after, empty_state)

    def test_get_order_on_empty_book_returns_sentinel(
        self, book_open, empty_state
    ):
        result = book_open.get_order(
            empty_state, order_id=12345, side=int(Side.BUY),
        )
        assert isinstance(result, OMIEOrder)
        assert int(result.order_id) == SENTINEL_ORDER_ID

    def test_cancel_then_get_order_returns_sentinel(
        self, book_open, empty_state
    ):
        """Submit a resting BUY, cancel it, then `get_order` must miss."""
        o = _mk_order(order_id=700, side=int(Side.BUY), price=49.0, quantity=1.0)
        state, _ = book_open.submit_order(empty_state, o)
        # Resting order should be retrievable.
        before = book_open.get_order(state, order_id=700, side=int(Side.BUY))
        assert int(before.order_id) == 700
        # Cancel.
        state2 = book_open.cancel_order(state, order_id=700, side=int(Side.BUY))
        # After cancel: must miss and return sentinel.
        after = book_open.get_order(state2, order_id=700, side=int(Side.BUY))
        assert int(after.order_id) == SENTINEL_ORDER_ID
