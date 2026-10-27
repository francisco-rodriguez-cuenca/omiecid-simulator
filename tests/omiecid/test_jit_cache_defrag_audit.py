"""Audit tests for P2-PERF-3 — Replay fixturing + JIT-cache defragmentation.

This file is an *independent* audit of the post-implementation state on
branch ``omie/P2-PERF-3-jit-cache-defrag``.  It does NOT read or import
from the original locked test file (``test_jit_cache_defrag.py``).  Any
overlap with assertions over there is incidental; everything below is
derived from the v5.9 story-card pins and from inspection of the
implementation's public surface.

What this file audits (one heading per claim from the story card):

1. **Cache-key stability across distinct ``OrderBook`` instances**
   — the actual claim of the story.  Two structurally-equal instances
   must share a single JIT cache entry per lifted function.

2. **Trampoline parity** — every public ``OrderBook`` method's return
   value is ``jnp.array_equal``-equal to a direct call into the
   corresponding module-level lifted function with ``self.cfg`` threaded.

3. **``static_argnames`` form on the three nuanced sites** (card pin 1b,
   1c, v5.9, 2026-05-11): keyword form ``static_argnames=("cfg", "side")``
   on ``_get_volume_at_price`` / ``_get_init_volume_at_price``; static
   argument values trigger a recompile.

4. **``reset`` split-helper consistency** (card pin 1a): two separate
   module-level functions ``_reset_empty`` and ``_reset_from_l2`` exist
   and the trampoline dispatches at the Python ``None``-branch level —
   no ``lax.cond`` on ``None`` inside ``OrderBook.reset``.

5. **Fixture-scope cascade integrity** — the 13 fixtures pinned in v5.9
   are at ``scope="module"`` and no function-scope fixture in the 8 named
   files transitively consumes a module-scope one.

6. **Saturated-buffer fixture semantics** — the module-scoped fixtures
   ``saturated_book_and_state`` / ``saturated_book_state_oid_ts`` return
   immutable ``NamedTuple`` state; consumer tests cannot corrupt them
   for sibling tests through ``submit_order`` mutation paths.

7. **Out-of-scope deferral integrity** — no private ``_<name>`` lifted
   function is imported outside ``omiecid_simulator/jaxob/``.

8. **Honest-delta sanity probe** — class body is compact, no ``@jax.jit``
   appears inside ``class OrderBook``, module-level decorator count is
   sane.

9. **Module-level function purity** — each ``_<name>`` lifted function
   takes ``state`` and/or ``cfg`` as positional parameters and uses no
   ``nonlocal``/``global`` statements.

All tests are AST-/introspection-based or run a fixed-size synthetic
``OrderBook`` interaction.  Wall-clock budget per test is well under
1 s on a warm cache; no test is marked ``slow``.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.jaxob import jorderbook
from omiecid_simulator.jaxob.jorderbook import (
    LobState,
    OrderBook,
    _get_best_ask,
    _get_best_bid,
    _get_best_bid_and_ask_inclQuants,
    _get_best_price,
    _get_init_volume_at_price,
    _get_L2_state,
    _get_next_executable_order,
    _get_order_by_id,
    _get_order_by_id_and_price,
    _get_order_by_time,
    _get_order_by_time_and_price,
    _get_side_ids,
    _get_volume_at_price,
    _init,
    _process_order_array,
    _process_orders_array,
    _process_orders_array_l2,
    _reset_empty,
    _reset_from_l2,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
JORDERBOOK_PATH = REPO_ROOT / "omiecid_simulator" / "jaxob" / "jorderbook.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# A small, reproducible L2 snapshot used by every functional test below.
_L2_INIT = jnp.array(
    [354200, 452, 350100, 89, 361200, 100, 344000, 400, 362900, 100,
     343100, 100, 364000, 400, 338700, 100, 371900, 1100, 337100, 1000,
     372200, 100, 336400, 1000, 372300, 200, 336000, 300, 372800, 1000,
     333600, 1000, 374600, 1000, 332500, 100, 376700, 100, 331600, 100],
    dtype=jnp.int32,
)


_LIFTED_BY_TRAMPOLINE = {
    # public OrderBook method -> module-level lifted function it delegates to
    "init": jorderbook._init,
    "process_order_array": jorderbook._process_order_array,
    "process_orders_array": jorderbook._process_orders_array,
    "process_orders_array_l2": jorderbook._process_orders_array_l2,
    "get_best_price": jorderbook._get_best_price,
    "get_best_bid": jorderbook._get_best_bid,
    "get_best_ask": jorderbook._get_best_ask,
    "get_best_bid_and_ask_inclQuants": jorderbook._get_best_bid_and_ask_inclQuants,
    "get_L2_state": jorderbook._get_L2_state,
    "get_side_ids": jorderbook._get_side_ids,
    "get_next_executable_order": jorderbook._get_next_executable_order,
    # multi-dispatch trampolines split into 2 lifted functions:
    # - reset:  _reset_empty / _reset_from_l2
    # - get_volume_at_price: _get_volume_at_price / _get_init_volume_at_price
    # - get_order: _get_order_by_id / _get_order_by_id_and_price
    # - get_order_at_time: _get_order_by_time / _get_order_by_time_and_price
    # parity checks for those are exercised explicitly in dedicated tests below.
}


def _module_source_tree() -> ast.Module:
    return ast.parse(JORDERBOOK_PATH.read_text())


def _orderbook_classdef() -> ast.ClassDef:
    for node in ast.walk(_module_source_tree()):
        if isinstance(node, ast.ClassDef) and node.name == "OrderBook":
            return node
    raise AssertionError("class OrderBook not found in jorderbook.py")


# ---------------------------------------------------------------------------
# 1. Cache-key stability across distinct OrderBook instances
# ---------------------------------------------------------------------------


class TestCacheKeyStabilityAcrossInstances:
    """The headline claim of P2-PERF-3.

    Two structurally-equal ``OrderBook`` instances must produce a single
    JIT trace per lifted function, not two — that is what unblocks the
    persistent-compilation-cache reuse path.
    """

    def test_init_lifted_cache_size_stays_one_across_two_instances(self):
        _init._clear_cache()
        ob1 = OrderBook()
        ob2 = OrderBook()
        assert ob1 is not ob2  # genuinely distinct Python objects

        _ = ob1.init()
        size_after_first = _init._cache_size()
        _ = ob2.init()
        size_after_second = _init._cache_size()

        assert size_after_first == 1
        assert size_after_second == 1, (
            "_init JIT cache must not grow when called via a second "
            "structurally-equal OrderBook instance — that is the "
            "fragmentation regression P2-PERF-3 was filed to prevent."
        )

    def test_process_order_array_lifted_cache_size_stays_one(self):
        _process_order_array._clear_cache()
        ob1 = OrderBook()
        ob2 = OrderBook()
        state1 = ob1.init()
        state2 = ob2.init()
        msg = jnp.array(
            [1, 1, 99, 346000, 8888, 8888, 3400, 5000000], dtype=jnp.int32
        )

        _ = ob1.process_order_array(state1, msg)
        _ = ob2.process_order_array(state2, msg)

        assert _process_order_array._cache_size() == 1

    def test_lowered_hlo_byte_identical_across_two_instances(self):
        """Belt-and-braces: byte-equal HLO is the strongest form of
        cache-key stability (it is what JAX's persistent cache content-
        addresses on)."""
        ob1 = OrderBook()
        ob2 = OrderBook()
        state1 = ob1.init()
        state2 = ob2.init()
        msg = jnp.array(
            [1, 1, 99, 346000, 8888, 8888, 3400, 5000000], dtype=jnp.int32
        )

        lo1 = _process_order_array.lower(state1, msg, ob1.cfg).compiler_ir(
            dialect="hlo"
        ).as_hlo_text()
        lo2 = _process_order_array.lower(state2, msg, ob2.cfg).compiler_ir(
            dialect="hlo"
        ).as_hlo_text()

        assert lo1 == lo2
        # length sanity: must not be empty / degenerate
        assert len(lo1) > 100


# ---------------------------------------------------------------------------
# 2. Trampoline parity
# ---------------------------------------------------------------------------


class TestTrampolineParity:
    """For each public ``OrderBook`` method whose lift is a single
    delegation, the trampoline's output must equal the lifted function's
    output called directly with ``self.cfg`` threaded.

    This proves trampolines are pure delegators — no silent transform,
    no behavioural drift.
    """

    @pytest.fixture(scope="class")
    def ob_and_state(self):
        ob = OrderBook()
        return ob, ob.reset(_L2_INIT)

    def test_init_parity(self):
        ob = OrderBook()
        s_tramp = ob.init()
        s_lifted = _init(ob.cfg)
        assert jnp.array_equal(s_tramp.asks, s_lifted.asks)
        assert jnp.array_equal(s_tramp.bids, s_lifted.bids)
        assert jnp.array_equal(s_tramp.trades, s_lifted.trades)

    def test_reset_empty_parity(self):
        ob = OrderBook()
        s_tramp = ob.reset()  # l2_book=None path -> _reset_empty
        s_lifted = _reset_empty(ob.cfg, jnp.array([0, 0]))
        assert jnp.array_equal(s_tramp.asks, s_lifted.asks)
        assert jnp.array_equal(s_tramp.bids, s_lifted.bids)

    def test_reset_from_l2_parity(self):
        ob = OrderBook()
        s_tramp = ob.reset(_L2_INIT)
        s_lifted = _reset_from_l2(_L2_INIT, jnp.array([0, 0]), ob.cfg)
        assert jnp.array_equal(s_tramp.asks, s_lifted.asks)
        assert jnp.array_equal(s_tramp.bids, s_lifted.bids)

    def test_process_order_array_parity(self, ob_and_state):
        ob, state = ob_and_state
        msg = jnp.array(
            [1, 1, 99, 346000, 8888, 8888, 3400, 5000000], dtype=jnp.int32
        )
        s_tramp = ob.process_order_array(state, msg)
        s_lifted = _process_order_array(state, msg, ob.cfg)
        assert jnp.array_equal(s_tramp.asks, s_lifted.asks)
        assert jnp.array_equal(s_tramp.bids, s_lifted.bids)

    def test_get_best_bid_parity(self, ob_and_state):
        ob, state = ob_and_state
        assert jnp.array_equal(ob.get_best_bid(state), _get_best_bid(state, ob.cfg))

    def test_get_best_ask_parity(self, ob_and_state):
        ob, state = ob_and_state
        assert jnp.array_equal(ob.get_best_ask(state), _get_best_ask(state, ob.cfg))

    def test_get_best_price_both_sides_parity(self, ob_and_state):
        ob, state = ob_and_state
        for side in (0, 1):
            assert jnp.array_equal(
                ob.get_best_price(state, side),
                _get_best_price(state, side, ob.cfg),
            )

    def test_get_best_bid_and_ask_inclquants_parity(self, ob_and_state):
        ob, state = ob_and_state
        a, b = ob.get_best_bid_and_ask_inclQuants(state)
        a2, b2 = _get_best_bid_and_ask_inclQuants(state, ob.cfg)
        assert jnp.array_equal(a, a2) and jnp.array_equal(b, b2)

    def test_get_L2_state_parity(self, ob_and_state):
        ob, state = ob_and_state
        assert jnp.array_equal(
            ob.get_L2_state(state, 5),
            _get_L2_state(state, 5, ob.cfg),
        )

    def test_get_side_ids_parity(self, ob_and_state):
        ob, state = ob_and_state
        for side in (0, 1):
            assert jnp.array_equal(
                ob.get_side_ids(state, side),
                _get_side_ids(state, side),
            )

    def test_get_volume_at_price_parity_both_dispatch_branches(self, ob_and_state):
        """Trampoline's ``init_only`` Python branch must dispatch to the
        correct lifted function (this proves card pin 1c's split is
        coherent)."""
        ob, state = ob_and_state
        # default (init_only=False)
        v_tramp = ob.get_volume_at_price(state, 1, 343100)
        v_lifted = _get_volume_at_price(state, 343100, 1, ob.cfg)
        assert jnp.array_equal(v_tramp, v_lifted)
        # init_only=True
        v_tramp_i = ob.get_volume_at_price(state, 1, 343100, init_only=True)
        v_lifted_i = _get_init_volume_at_price(state, 343100, 1, ob.cfg)
        assert jnp.array_equal(v_tramp_i, v_lifted_i)

    def test_get_order_parity_both_dispatch_branches(self, ob_and_state):
        """Trampoline's ``price is not None`` Python branch (card pin 1c)
        must dispatch to the correct lifted function."""
        ob, state = ob_and_state
        # no price
        a = ob.get_order(state, 1, 99)
        b = _get_order_by_id(state, 1, 99)
        assert jnp.array_equal(a, b)
        # with price
        a2 = ob.get_order(state, 1, 99, price=343100)
        b2 = _get_order_by_id_and_price(state, 1, 99, 343100)
        assert jnp.array_equal(a2, b2)

    def test_get_order_at_time_parity_both_dispatch_branches(self, ob_and_state):
        ob, state = ob_and_state
        a = ob.get_order_at_time(state, 1, 3400, 5000000)
        b = _get_order_by_time(state, 1, 3400, 5000000)
        assert jnp.array_equal(a, b)
        a2 = ob.get_order_at_time(state, 1, 3400, 5000000, price=343100)
        b2 = _get_order_by_time_and_price(state, 1, 3400, 5000000, 343100)
        assert jnp.array_equal(a2, b2)

    def test_get_next_executable_order_parity(self, ob_and_state):
        """Verifies trampoline parity for ``_get_next_executable_order`` after the v5.9 lift."""
        ob, state = ob_and_state
        for side in (0, 1):
            assert jnp.array_equal(
                ob.get_next_executable_order(state, side),
                _get_next_executable_order(state, side, ob.cfg),
            )


# ---------------------------------------------------------------------------
# 3. static_argnames correctness on the three nuanced sites
# ---------------------------------------------------------------------------


def _decorator_keywords(func_node: ast.FunctionDef) -> List[ast.keyword]:
    """Return the keyword list of the @partial(jax.jit, …) decorator on
    ``func_node`` — or an empty list if the function uses bare @jax.jit."""
    for d in func_node.decorator_list:
        # @partial(jax.jit, static_argnames=(...))
        if isinstance(d, ast.Call) and ast.unparse(d.func) == "partial":
            return list(d.keywords)
    return []


def _module_function(node_name: str) -> ast.FunctionDef:
    for node in ast.walk(_module_source_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == node_name:
            return node
    raise AssertionError(f"function {node_name} not found")


class TestStaticArgnamesForm:
    """Card pin v5.9, 2026-05-11 — the three nuanced sites must use
    ``static_argnames`` (keyword form), not ``static_argnums``.

    Note on lift-shape deviation: the v5.9 card pin 1c specifies
    ``_get_order`` / ``_get_order_at_time`` with
    ``static_argnames=("cfg", "init_only")``.  The implementation chose
    a different (still card-permitted) lift shape: split into four
    separate ``@jax.jit`` functions (``_get_order_by_id`` /
    ``_get_order_by_id_and_price`` / ``_get_order_by_time`` /
    ``_get_order_by_time_and_price``) with the ``price`` and
    ``init_only`` Python branches at the trampoline.  This audit accepts
    that lift shape because the behavioural parity tests in
    ``TestTrampolineParity`` cover it; we still verify the two ``side``
    sites carry the keyword form as the card pins.
    """

    def test_get_volume_at_price_uses_static_argnames_keyword(self):
        kwds = _decorator_keywords(_module_function("_get_volume_at_price"))
        arg_names = [k.arg for k in kwds]
        assert "static_argnames" in arg_names, (
            "card pin 1b requires keyword `static_argnames`, not numeric `static_argnums`"
        )
        # And static_argnums must NOT appear (keyword-form supersedes it).
        assert "static_argnums" not in arg_names

        # The keyword's value must include both 'cfg' and 'side'.
        sa_kw = [k for k in kwds if k.arg == "static_argnames"][0]
        # Tuple value or single-string value
        val_src = ast.unparse(sa_kw.value)
        assert "cfg" in val_src and "side" in val_src

    def test_get_init_volume_at_price_uses_static_argnames_keyword(self):
        kwds = _decorator_keywords(_module_function("_get_init_volume_at_price"))
        arg_names = [k.arg for k in kwds]
        assert "static_argnames" in arg_names
        assert "static_argnums" not in arg_names
        sa_kw = [k for k in kwds if k.arg == "static_argnames"][0]
        val_src = ast.unparse(sa_kw.value)
        assert "cfg" in val_src and "side" in val_src

    def test_static_side_flip_triggers_recompile(self):
        """Behavioural cross-check on the AST claim: changing the static
        ``side`` value must produce a second cache entry (and the third
        call with the same side==0 must NOT add a third)."""
        _get_volume_at_price._clear_cache()
        ob = OrderBook()
        state = ob.reset(_L2_INIT)

        _ = ob.get_volume_at_price(state, 0, 350100)
        assert _get_volume_at_price._cache_size() == 1
        _ = ob.get_volume_at_price(state, 1, 350100)
        assert _get_volume_at_price._cache_size() == 2, (
            "flipping `side` from 0 to 1 must add a new cache entry — "
            "this proves `side` is a static argument"
        )
        # And calling again with side=0 but a different (traced) price must NOT recompile.
        _ = ob.get_volume_at_price(state, 0, 354200)
        assert _get_volume_at_price._cache_size() == 2, (
            "varying the traced `price` argument must not recompile — "
            "if it does, side has been incorrectly marked traced"
        )


# ---------------------------------------------------------------------------
# 4. reset split-helper consistency (card pin 1a)
# ---------------------------------------------------------------------------


class TestResetSplitHelperConsistency:

    def test_reset_empty_module_function_exists_and_is_jitted(self):
        assert callable(_reset_empty)
        # PjitFunction (jaxlib._jax.PjitFunction) is what jax.jit returns
        assert hasattr(_reset_empty, "_cache_size")

    def test_reset_from_l2_module_function_exists_and_is_jitted(self):
        assert callable(_reset_from_l2)
        assert hasattr(_reset_from_l2, "_cache_size")

    def test_trampoline_dispatches_at_python_none_level_no_lax_cond(self):
        """The reset trampoline must use a plain Python ``if l2_book is None``
        — not a ``lax.cond`` with sentinel construction (card pin 1a
        Option (b))."""
        src = inspect.getsource(OrderBook.reset)
        assert "lax.cond" not in src, (
            "card pin 1a requires Python-level None dispatch, not lax.cond"
        )
        assert "lax.switch" not in src
        # Positive structural assertion: both helpers are referenced in the trampoline body.
        assert "_reset_empty" in src
        assert "_reset_from_l2" in src


# ---------------------------------------------------------------------------
# 5. Fixture-scope cascade integrity (v5.9, 13 fixtures)
# ---------------------------------------------------------------------------


# The 13-edit cascade enumerated in agile-plan v5.9.
CASCADE_FILES_AND_FIXTURES: List[Tuple[str, Tuple[str, ...]]] = [
    ("tests/omiecid/conftest.py", (
        "open_contract", "product_config", "market_config",
        "book", "book_and_state", "empty_state",
    )),
    ("tests/omiecid/test_contract_book_audit.py", ("open_contract",)),
    ("tests/omiecid/test_iceberg_manager_jit_audit.py", ("open_contract",)),
    ("tests/omiecid/test_ioc_fok_audit.py", ("open_contract",)),
    ("tests/omiecid/test_iceberg_manager_jit.py", ("open_contract",)),
    ("tests/omiecid/test_iceberg_cap_overflow.py", ("open_contract",)),
    ("tests/omiecid/test_iceberg_audit.py", ("open_contract",)),
    ("tests/omiecid/test_gct_expiry_views_audit.py", ("open_contract",)),
]


def _fixture_scope(file_src: str, fixture_name: str) -> str:
    """Return the scope string for ``fixture_name`` defined in ``file_src``.

    Looks for ``def <fixture_name>(...)`` with a ``@pytest.fixture(...)``
    decorator above it (or with no args, meaning function scope by default).
    """
    tree = ast.parse(file_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fixture_name:
            for d in node.decorator_list:
                if isinstance(d, ast.Call):
                    deco_src = ast.unparse(d.func)
                    if "pytest.fixture" in deco_src or deco_src.endswith("fixture"):
                        for kw in d.keywords:
                            if kw.arg == "scope":
                                return ast.literal_eval(kw.value)
                        return "function"
                elif isinstance(d, ast.Attribute):
                    if ast.unparse(d) in ("pytest.fixture",):
                        return "function"
    raise AssertionError(
        f"fixture {fixture_name!r} not found in source"
    )


class TestFixtureCascadeIntegrity:

    @pytest.mark.parametrize(
        "rel_path,fixture_name",
        [
            (rel, name)
            for rel, names in CASCADE_FILES_AND_FIXTURES
            for name in names
        ],
    )
    def test_cascade_entry_at_module_scope(self, rel_path, fixture_name):
        src = (REPO_ROOT / rel_path).read_text()
        scope = _fixture_scope(src, fixture_name)
        assert scope == "module", (
            f"{rel_path}::{fixture_name} expected scope='module' "
            f"(v5.9 cascade); found scope={scope!r}"
        )

    def test_cascade_total_count_is_thirteen(self):
        total = sum(len(names) for _, names in CASCADE_FILES_AND_FIXTURES)
        assert total == 13

    def test_no_module_scope_fixture_consumes_function_scope_fixture(self):
        """Negative integrity check (pytest scope rule):

          - a module-scope fixture MAY NOT depend on a function-scope
            fixture (pytest raises ``ScopeMismatch`` at collection),
          - a function-scope fixture MAY depend on a module-scope one
            (broader-to-narrower is always permitted).

        After the v5.9 cascade, the 13 named fixtures are module-scope.
        Inspect each of the 8 cascade files plus the two ring-buffer files
        (which carry the Deliverable-(3) saturated fixtures) for any
        module-scope fixture that names a function-scope fixture as a
        parameter — that would re-trigger the ``ScopeMismatch`` the
        cascade was designed to prevent.
        """
        # Build the set of function-scope fixture names that exist in any
        # of the inspected files (so we can detect leakage into module-scope).
        inspected_files = [
            rel for rel, _ in CASCADE_FILES_AND_FIXTURES
        ] + [
            "tests/omiecid/test_trade_extraction_ring_buffer.py",
            "tests/omiecid/test_trade_extraction_ring_buffer_audit.py",
        ]

        # First pass: collect every fixture (name, scope, file).
        fixture_scope_by_name: dict[str, str] = {}
        for rel in inspected_files:
            src = (REPO_ROOT / rel).read_text()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                scope = None
                for d in node.decorator_list:
                    deco_src = ast.unparse(d)
                    if "pytest.fixture" in deco_src or deco_src.endswith(
                        "fixture"
                    ):
                        scope = "function"
                        if isinstance(d, ast.Call):
                            for kw in d.keywords:
                                if kw.arg == "scope":
                                    scope = ast.literal_eval(kw.value)
                        break
                if scope is not None:
                    # If the same name exists in two files with different
                    # scopes, the narrower wins for ScopeMismatch detection
                    # (the override determines the local resolution).
                    prev = fixture_scope_by_name.get(node.name)
                    if prev is None or prev == "function":
                        fixture_scope_by_name[node.name] = scope
                    # else: keep the wider scope; file-local overrides
                    # win in pytest, but our concern is whether any
                    # module-scope fixture *anywhere* could resolve to a
                    # function-scope dep, which the override-aware
                    # resolution above captures conservatively.

        function_scope_names = {
            n for n, s in fixture_scope_by_name.items() if s == "function"
        }

        # Second pass: assert no module-scope fixture consumes one.
        offenders: list[str] = []
        for rel in inspected_files:
            src = (REPO_ROOT / rel).read_text()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                is_fixture = False
                scope = "function"
                for d in node.decorator_list:
                    deco_src = ast.unparse(d)
                    if "pytest.fixture" in deco_src or deco_src.endswith(
                        "fixture"
                    ):
                        is_fixture = True
                        if isinstance(d, ast.Call):
                            for kw in d.keywords:
                                if kw.arg == "scope":
                                    scope = ast.literal_eval(kw.value)
                if not is_fixture:
                    continue
                if scope not in ("module", "session"):
                    continue
                arg_names = {a.arg for a in node.args.args}
                bad = arg_names & function_scope_names
                if bad:
                    offenders.append(
                        f"{rel}::{node.name} (scope={scope!r}) depends on "
                        f"function-scope fixture(s) {sorted(bad)!r}"
                    )
        assert offenders == [], (
            "ScopeMismatch risk: " + "; ".join(offenders)
        )


# ---------------------------------------------------------------------------
# 6. Saturated-buffer fixture semantics
# ---------------------------------------------------------------------------


class TestSaturatedBufferFixtureScope:

    def test_saturated_book_and_state_fixture_is_module_scope(self):
        src = (REPO_ROOT / "tests/omiecid/test_trade_extraction_ring_buffer.py").read_text()
        assert _fixture_scope(src, "saturated_book_and_state") == "module"

    def test_saturated_book_state_oid_ts_fixture_is_module_scope(self):
        src = (REPO_ROOT / "tests/omiecid/test_trade_extraction_ring_buffer_audit.py").read_text()
        assert _fixture_scope(src, "saturated_book_state_oid_ts") == "module"

    def test_omie_book_state_is_a_namedtuple(self):
        """Mutation-safety guarantee from the v5.9 card.  ``OMIEBookState``
        is a ``NamedTuple``: any attempt to mutate produces ``AttributeError``."""
        from omiecid_simulator.omiecid.contract_book import (
            OMIEBookState, ContractBook,
        )
        from omiecid_simulator.omiecid.contract import Contract, ContractStatus
        from omiecid_simulator.omiecid.omie_config import (
            OMIEProductConfig, OMIEMarketConfig,
        )

        assert issubclass(OMIEBookState, tuple), (
            "OMIEBookState must be a NamedTuple — module-scope re-use "
            "across consumer tests depends on immutability"
        )
        # And it must reject in-place mutation.
        contract = Contract(
            contract_id=1, delivery_start_minutes=0, delivery_end_minutes=15,
            gct_minutes=-60, status=ContractStatus.OPEN,
        )
        book = ContractBook(contract, OMIEProductConfig(), OMIEMarketConfig())
        state = book.init()
        # NamedTuple fields are read-only.
        with pytest.raises(AttributeError):
            state.trades = state.trades  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 7. Out-of-scope deferral integrity
# ---------------------------------------------------------------------------


class TestOutOfScopeDeferralIntegrity:
    """No module outside ``omiecid_simulator/jaxob/`` may import a
    private-prefixed ``_<name>`` lifted function from ``jorderbook`` —
    that would silently widen the surface the trampolines were designed
    to preserve.
    """

    def test_no_private_lifted_import_outside_jaxob(self):
        """No ``from omiecid_simulator.jaxob.jorderbook import _xxx`` style
        import may exist outside ``omiecid_simulator/jaxob/`` and ``tests/``.

        ``tests/`` is excluded because this audit file itself imports
        private names intentionally — that is the entire point of the
        audit.  Production code must not.
        """
        offenders: List[str] = []
        for path in REPO_ROOT.glob("omiecid_simulator/**/*.py"):
            if path.is_relative_to(REPO_ROOT / "omiecid_simulator" / "jaxob"):
                continue
            try:
                text = path.read_text()
            except Exception:
                continue
            if "jorderbook" not in text:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                # Some unrelated files in jaxrl have py3.10 f-string
                # edge cases; skip them — they cannot contain a private
                # jorderbook import if the file does not even parse,
                # and we explicitly searched for the string above.
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if "jorderbook" in node.module:
                        for alias in node.names:
                            if alias.name.startswith("_"):
                                offenders.append(f"{path}: {node.module}::{alias.name}")
        assert offenders == [], (
            "Private lifted-function imports leaked outside "
            f"omiecid_simulator/jaxob/: {offenders}"
        )


# ---------------------------------------------------------------------------
# 8. Honest-delta sanity probe
# ---------------------------------------------------------------------------


class TestHonestDeltaSanityProbe:

    def test_orderbook_class_body_is_compact(self):
        classdef = _orderbook_classdef()
        body_loc = classdef.end_lineno - classdef.lineno + 1
        # Post-lift the class body should be trampolines only.  The
        # implementation merge note records the class opens at line 163;
        # we assert a generous upper bound that still catches an
        # accidental re-introduction of a fat method body.
        assert body_loc < 200, (
            f"OrderBook class body grew to {body_loc} LOC — post-lift it "
            "should consist of trampolines only.  Re-check whether a "
            "@jax.jit'd body sneaked back in."
        )

    def test_no_jit_decorators_inside_orderbook_class(self):
        classdef = _orderbook_classdef()
        for item in classdef.body:
            if isinstance(item, ast.FunctionDef):
                for d in item.decorator_list:
                    deco_src = ast.unparse(d)
                    assert "jax.jit" not in deco_src, (
                        f"{item.name} carries a @jax.jit decorator INSIDE "
                        f"class OrderBook ({deco_src!r}) — this defeats "
                        "the entire point of P2-PERF-3."
                    )

    def test_module_level_jit_decorator_count_matches_lifted_function_count(self):
        """Each ``_<name>`` lifted function with a JIT-y decorator should
        match the count of decorated module-level functions.  This catches
        an accidental dangling decorator or a duplicated lift.
        """
        tree = _module_source_tree()
        jit_decorated = 0
        lifted_funcs = 0
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                has_jit = any(
                    "jax.jit" in ast.unparse(d) for d in node.decorator_list
                )
                if has_jit:
                    jit_decorated += 1
                if node.name.startswith("_") and node.name not in ("_side_array",):
                    # _side_array is the deliberate non-decorated helper
                    # extracted in the refactor beat (it inlines under
                    # tracing inside each jit'd caller).
                    if has_jit:
                        lifted_funcs += 1
        assert jit_decorated == lifted_funcs, (
            f"@jax.jit decorator count ({jit_decorated}) does not match "
            f"private-lifted-function count ({lifted_funcs}) — possible "
            "dangling decorator or non-private jit at module level."
        )


# ---------------------------------------------------------------------------
# 9. Module-level function purity
# ---------------------------------------------------------------------------


class TestModuleLevelFunctionPurity:

    def test_each_lifted_function_has_state_or_cfg_positional_argument(self):
        """Every lifted ``_<name>`` taking a JIT decorator must have
        either ``state`` or ``cfg`` (or both) in its parameter list.
        This is the structural sign of a properly-lifted pure function.
        """
        tree = _module_source_tree()
        offenders: List[str] = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("_") or node.name == "_side_array":
                continue
            has_jit = any("jax.jit" in ast.unparse(d) for d in node.decorator_list)
            if not has_jit:
                continue
            arg_names = {a.arg for a in node.args.args}
            if not (("state" in arg_names) or ("cfg" in arg_names) or
                    ("l2_book" in arg_names)):  # _reset_from_l2 takes l2_book + cfg
                offenders.append(node.name)
        assert offenders == [], (
            f"Lifted functions missing state/cfg/l2_book positional arg: "
            f"{offenders}"
        )

    def test_no_lifted_function_uses_nonlocal_or_global(self):
        tree = _module_source_tree()
        offenders: List[str] = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("_"):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Nonlocal, ast.Global)):
                    offenders.append(f"{node.name}: {type(sub).__name__}")
        assert offenders == [], (
            f"Lifted functions reference module-mutable state via "
            f"nonlocal/global: {offenders}"
        )

    def test_no_lifted_function_first_arg_is_named_self(self):
        """A trace-time regression-catcher: if a lifted function is
        accidentally restored to take ``self`` as its first arg, the
        cache-key would once again partition by instance identity."""
        tree = _module_source_tree()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("_"):
                continue
            if not node.args.args:
                continue
            first = node.args.args[0].arg
            assert first != "self", (
                f"lifted function {node.name} takes `self` as its first "
                "argument — this is the very fragmentation regression "
                "P2-PERF-3 was filed to prevent."
            )
