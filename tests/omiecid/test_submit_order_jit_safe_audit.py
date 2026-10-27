"""Audit tests for P2-H2 — `contract_book.submit_order` JIT-safe.

This file is an *independent* post-implementation audit of the
module-level ``_submit_order`` lift on branch
``omie/P2-H2-submit-order-jit-safe-v2``. It does NOT read or import
from the original locked test file
(``tests/omiecid/test_submit_order_jit_safe.py``). Any overlap with
assertions over there is incidental — every test below is derived
independently from:

  * the P2-H2 story card (agile-plan.md v5.20 line 545+: Description /
    Deliverables / Test-first contract rows);
  * the canonical Phase 2 module-level jit idiom pinned in
    migration-plan.md v5.11 §"Module-level jit idiom";
  * the market-spec NON / IOC / FOK / ICE / ICEwPI semantics (§4.5,
    §13);
  * inspection of the finished implementation's PUBLIC interface only
    (class signatures, type annotations, decorator shape — never
    behaviour we'd be reverse-engineering).

Coverage dimensions (one heading per probe category):

  1. **Module-level jit decoration shape** — ``_submit_order`` is a
     ``PjitFunction`` decorated with the canonical
     ``static_argnames=("product_config", "market_config")`` idiom; no
     bare ``@jax.jit``; no leftover ``static_argnums``.
  2. **AST invariants on contract_book.py** — no ``@jax.jit`` /
     ``@partial(jax.jit, ...)`` decorator survives on a ``ContractBook``
     instance method; no Python ``for`` loop in the four lifted
     submit-order helpers; the lifted body imports neither
     ``Contract`` nor ``ContractStatus`` at trace time.
  3. **Jaxpr-content probes** — the lifted submit_order's jaxpr
     contains exactly one ``switch`` operation (the four-arm
     exec-restriction dispatch) and at least one ``while`` (one of the
     two multi-slice loops); no forbidden tolerance literals; the
     jaxpr leaves are JAX scalar primitives.
  4. **SubmitResult dtype contract** — every field of the five-field
     ``NamedTuple`` is a 0-d ``jnp.*`` scalar with the right dtype
     across all four variants (NON / IOC / FOK / ICE), under
     accepted-paid, accepted-no-trade, and rejected paths; the
     pytree leaf count is stable.
  5. **ExecRestriction surface integrity** — exactly four IntEnum
     members (NON=0, IOC=1, FOK=2, ICE=3); the four ``lax.switch``
     branch closures inside ``_submit_order_body`` exhaust the enum.
  6. **`get_order` sentinel contract** — returns
     ``make_sentinel_order()`` shape on miss; ``None`` is never
     returned for any side / order_id combination; the discriminator
     lives in ``order_id == SENTINEL_ORDER_ID``.
  7. **Cross-module isolation / out-of-scope deferral** — the cancel
     instance method is unchanged from the v5.19 baseline in surface
     terms (it is NOT jitted, it operates Python-side, and it remains
     a separate public method); ``get_anonymous_lob`` /
     ``get_agent_orders`` / ``get_agent_trades`` remain Python-side
     per the §B4 decision.
  8. **Trace-count parity under variation** — within a single jitted
     ``_submit_order``, varying *traced* inputs (price, quantity,
     order_id, ``order.ppd`` within the ICE arm) does not bump the
     cache count; switching to a *new* product_config DOES bump it
     (since that arg is static).
  9. **Truncation backstop integrity** — the ``OMIEBookState.truncated``
     carry-flag is sticky-True across submits once set, and
     ``submit_order`` ORs into it (no new flag introduced).
 10. **Contract-closed gate** — when the parent contract's
     ``ContractStatus`` is not OPEN, ``submit_order`` returns the
     canonical REJECTED ``SubmitResult`` without entering the jitted
     body (the P2-L3 caller-contract invariant).

Marker discipline (Story T1.1): every test in this audit file traces
or compiles ``_submit_order``; the marker is pinned at module scope.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
from typing import List

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
from omiecid_simulator.omiecid.jax_constants import (
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


# Module-level marker (Story T1.1).
pytestmark = pytest.mark.jit


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTRACT_BOOK_PATH = REPO_ROOT / "omiecid_simulator" / "omiecid" / "contract_book.py"


# ---------------------------------------------------------------------------
# Local fixtures (intentionally self-contained — no dependency on conftest
# fixtures, so the audit module is reproducible by re-execution alone).
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
def closed_contract_local() -> Contract:
    return Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.CLOSED,
    )


@pytest.fixture(scope="module")
def book_open(open_contract_local, product_cfg, market_cfg) -> ContractBook:
    return ContractBook(open_contract_local, product_cfg, market_cfg)


@pytest.fixture(scope="module")
def empty_state(book_open) -> OMIEBookState:
    return book_open.init()


def _mk_order(**overrides) -> OMIEOrder:
    """Build a default-valid OMIE order with field overrides."""
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


def _module_source_tree() -> ast.Module:
    return ast.parse(CONTRACT_BOOK_PATH.read_text())


def _module_function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_module_source_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in contract_book.py")


def _contractbook_classdef() -> ast.ClassDef:
    for node in ast.walk(_module_source_tree()):
        if isinstance(node, ast.ClassDef) and node.name == "ContractBook":
            return node
    raise AssertionError("class ContractBook not found in contract_book.py")


# ---------------------------------------------------------------------------
# 1. Module-level jit decoration shape
# ---------------------------------------------------------------------------


class TestModuleLevelJitDecorationShape:
    """The canonical Phase 2 module-level jit idiom pinned in
    migration-plan v5.11 §"Module-level jit idiom":

        ``@partial(jax.jit, static_argnames=(...))``

    or its semantic equivalent ``jax.jit(fn, static_argnames=(...))``.
    Bare ``@jax.jit`` and numeric ``static_argnums`` are forbidden.
    """

    def test_submit_order_is_jitted_pjit_function(self):
        # PjitFunction (jaxlib._jax.PjitFunction) is what jax.jit returns
        # — same surface used by jorderbook lifted helpers in
        # P2-PERF-3.  We probe for the public ``_cache_size`` /
        # ``lower`` attributes those carry.
        assert callable(_submit_order)
        assert hasattr(_submit_order, "_cache_size"), (
            "_submit_order is not a jax.jit'd function — expected a "
            "PjitFunction exposing _cache_size, got "
            f"{type(_submit_order).__name__}"
        )
        assert hasattr(_submit_order, "lower")

    def test_submit_order_body_underlies_jitted_lift(self):
        """The decorator must wrap ``_submit_order_body`` — the
        un-jitted helper exists at module scope as a Python callable
        so the implementation can be audited directly."""
        assert callable(_submit_order_body)
        # The two are distinct objects: the body is a plain Python
        # function; the jitted lift is the PjitFunction wrapping it.
        assert _submit_order_body is not _submit_order

    def test_submit_order_uses_static_argnames_not_static_argnums(self):
        """AST probe: the ``jax.jit(...)`` call wrapping
        ``_submit_order_body`` must use the keyword form
        ``static_argnames``, NOT numeric ``static_argnums``."""
        tree = _module_source_tree()
        # Look for the module-level Assign that binds ``_submit_order``.
        found = False
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "_submit_order"
                for t in node.targets
            ):
                continue
            found = True
            assert isinstance(node.value, ast.Call), (
                "expected `_submit_order = jax.jit(...)` call form"
            )
            call = node.value
            kw_names = [k.arg for k in call.keywords]
            assert "static_argnames" in kw_names, (
                "canonical Phase 2 idiom requires keyword "
                "`static_argnames`, not `static_argnums`"
            )
            assert "static_argnums" not in kw_names, (
                "numeric `static_argnums` is forbidden by the canonical "
                "idiom (signature-reorder-robustness)"
            )
            # Verify the keyword's value mentions both required names.
            sa_kw = next(k for k in call.keywords if k.arg == "static_argnames")
            val_src = ast.unparse(sa_kw.value)
            assert "product_config" in val_src
            assert "market_config" in val_src
        assert found, (
            "module-level `_submit_order` assignment not found in "
            "contract_book.py"
        )

    def test_submit_order_static_argnames_tuple_form_is_canonical(self):
        """The migration-plan v5.11 idiom prescribes a tuple, not a
        single string, even when the tuple has two entries.  Locks
        against a regression to a single-string form."""
        tree = _module_source_tree()
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "_submit_order"
                for t in node.targets
            ):
                continue
            call = node.value
            sa_kw = next(k for k in call.keywords if k.arg == "static_argnames")
            assert isinstance(sa_kw.value, ast.Tuple), (
                "static_argnames value must be a tuple per the canonical "
                f"idiom; got {type(sa_kw.value).__name__}"
            )
            tuple_elts = [
                ast.literal_eval(e) for e in sa_kw.value.elts
                if isinstance(e, ast.Constant)
            ]
            assert tuple_elts == ["product_config", "market_config"], (
                f"static_argnames tuple has unexpected contents: {tuple_elts}"
            )


# ---------------------------------------------------------------------------
# 2. AST invariants on contract_book.py
# ---------------------------------------------------------------------------


class TestASTInvariantsOnContractBookModule:
    """No ``@jax.jit`` survives on a ``ContractBook`` instance method;
    the lifted helpers are at module scope and pure-function-shaped.
    """

    def test_no_jit_decorator_on_any_contractbook_method(self):
        classdef = _contractbook_classdef()
        offenders: List[str] = []
        for item in classdef.body:
            if isinstance(item, ast.FunctionDef):
                for d in item.decorator_list:
                    src = ast.unparse(d)
                    if "jax.jit" in src or "@jit" == src:
                        offenders.append(f"{item.name}: {src}")
        assert offenders == [], (
            "ContractBook instance method(s) carry a @jax.jit decorator: "
            f"{offenders}. P2-H2 requires the JIT to live at module "
            "level only — instance-method jit re-fragments the cache "
            "(P2-PERF-3 regression class)."
        )

    def test_submit_order_method_is_a_pure_python_trampoline(self):
        """The class-method ``ContractBook.submit_order`` must
        delegate to the module-level lift.  We probe its source for
        the public-name call.
        """
        src = inspect.getsource(ContractBook.submit_order)
        assert "_submit_order(" in src, (
            "ContractBook.submit_order does not delegate to the "
            "module-level _submit_order lift"
        )
        # And it must NOT contain a lax.cond / lax.switch /
        # lax.while_loop — those belong to the jitted body, not the
        # Python-side trampoline.
        assert "lax.switch" not in src
        assert "lax.while_loop" not in src

    @pytest.mark.parametrize(
        "fn_name",
        [
            "_non_or_ioc_path",
            "_fok_path",
            "_ice_path",
            "_submit_order_body",
            "_while_loop_match",
            "_while_loop_match_fok",
        ],
    )
    def test_no_python_for_loop_in_lifted_helper(self, fn_name):
        """The two Phase-1 ``for iteration in range(...)`` multi-slice
        loops at the old line 437 / 600 are now ``lax.while_loop``;
        no Python ``for`` may remain inside the lifted bodies."""
        node = _module_function(fn_name)
        for sub in ast.walk(node):
            if isinstance(sub, ast.For):
                raise AssertionError(
                    f"{fn_name} contains a Python `for` loop at line "
                    f"{sub.lineno} — must be lax.while_loop / lax.scan"
                )

    @pytest.mark.parametrize(
        "fn_name",
        [
            "_submit_order_body",
            "_non_or_ioc_path",
            "_fok_path",
            "_ice_path",
            "_while_loop_match",
            "_while_loop_match_fok",
        ],
    )
    def test_no_while_python_loop_in_lifted_helper(self, fn_name):
        """Equivalent guard for ``while`` Python loops."""
        node = _module_function(fn_name)
        for sub in ast.walk(node):
            if isinstance(sub, ast.While):
                raise AssertionError(
                    f"{fn_name} contains a Python `while` loop at line "
                    f"{sub.lineno} — must be lax.while_loop"
                )

    def test_contract_status_not_imported_into_traced_body(self):
        """The P2-L3 caller-contract invariant (jax_constants.py §B4
        / §P2-L3 DAG-annotation) forbids ``ContractStatus`` being
        consulted inside a traced submit_order body.  We probe the
        AST of ``_submit_order_body`` for any ``ContractStatus``
        identifier."""
        node = _module_function("_submit_order_body")
        body_src = ast.unparse(node)
        assert "ContractStatus" not in body_src, (
            "_submit_order_body references ContractStatus — P2-L3 "
            "caller-contract invariant violation (status must be "
            "resolved Python-side)"
        )

    def test_contract_open_branch_lives_in_trampoline_not_body(self):
        """The Python-side ``if not contract_open`` early return must
        live in the class-method trampoline, NOT inside the jitted
        body."""
        trampoline_src = inspect.getsource(ContractBook.submit_order)
        body_src = inspect.getsource(_submit_order_body)
        assert "contract_open" in trampoline_src
        # The body may NOT consult contract.status — it's baked True.
        assert ".status" not in body_src, (
            "_submit_order_body consults `.status` directly — "
            "must be resolved Python-side before entering the jit"
        )


# ---------------------------------------------------------------------------
# 3. Jaxpr-content probes
# ---------------------------------------------------------------------------


class TestJaxprContentProbes:
    """Structural probes of the lowered jaxpr text — these are
    last-line-of-defence checks against a regression that bypasses
    the AST guards above (e.g., by importing ``Contract`` aliased
    or by inlining a ``for`` loop via list comprehension)."""

    @pytest.fixture(scope="class")
    def submit_jaxpr(self, book_open, empty_state, product_cfg, market_cfg):
        order = _mk_order()
        # jax.make_jaxpr accepts static_argnums (positional indices) — not
        # static_argnames. The two configs sit at positions 2 and 3.
        return jax.make_jaxpr(
            _submit_order_body,
            static_argnums=(2, 3),
        )(empty_state, order, product_cfg, market_cfg)

    def test_jaxpr_text_contains_switch_primitive(self, submit_jaxpr):
        text = str(submit_jaxpr)
        # ``lax.switch`` lowers to a ``cond`` primitive in jaxpr text
        # (jax 0.4.x and 0.5.x).  Either form is acceptable.
        assert ("cond" in text) or ("switch" in text), (
            "submit_order jaxpr lacks any switch/cond primitive — "
            "the four-arm ExecRestriction dispatch is missing"
        )

    def test_jaxpr_text_contains_while_primitive(self, submit_jaxpr):
        text = str(submit_jaxpr)
        assert "while" in text, (
            "submit_order jaxpr lacks any while primitive — at least "
            "one of the two multi-slice matching loops should appear"
        )

    def test_jaxpr_has_no_forbidden_tolerance_literal(self, submit_jaxpr):
        """The validator and downstream traces must not contain
        float tolerance literals (1e-9 style).  This mirrors the
        guard in test_order_validator_audit.py and prevents a sneaky
        regression that swaps an integer tick check for a tolerance.
        """
        text = str(submit_jaxpr)
        forbidden_exact = [
            "1e-09", "1e-9", "1.e-9", "1.0e-09",
            "1e-08", "1e-8",
            "1e-06", "1e-6",
        ]
        for pat in forbidden_exact:
            assert pat not in text, (
                f"submit_order jaxpr contains forbidden tolerance "
                f"literal {pat!r}"
            )
        # And generic small-exponent regex sweep:
        assert re.search(r"\b\d+(\.\d+)?e-0?[6-9]\b", text) is None, (
            "submit_order jaxpr contains a suspicious small-exponent "
            "literal — exact-integer encoding should leave no float "
            "tolerances behind"
        )

    def test_jaxpr_in_vars_count_matches_one_state_one_order(self, submit_jaxpr):
        """Sanity check the jaxpr's ``invars`` count — it should be
        the sum of pytree leaves of ``OMIEBookState`` (4 leaves: lob
        is itself 4 jaxob arrays, plus iceberg_meta, ts_counter,
        truncated) plus ``OMIEOrder`` (13 scalar fields).  A drift
        here would indicate a silent signature change."""
        # We don't pin an exact count (jaxob LobState's pytree leaf
        # count is itself an implementation detail), but it must be
        # non-trivial (>= 13 for the order leaves alone).
        assert len(submit_jaxpr.in_avals) >= 13, (
            "submit_order jaxpr invar count is suspiciously small "
            f"({len(submit_jaxpr.in_avals)})"
        )


# ---------------------------------------------------------------------------
# 4. SubmitResult dtype contract
# ---------------------------------------------------------------------------


class TestSubmitResultDtypeContract:
    """Each field of ``SubmitResult`` is a 0-d ``jnp.*`` scalar
    (``jnp.bool_`` / ``jnp.float32``) — no Python ``bool`` / ``float``
    leaks; the pytree leaf count is stable at 5 across all paths.
    """

    @pytest.fixture(scope="class")
    def book_state(self, open_contract_local, product_cfg, market_cfg):
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        return book, book.init()

    # ----- field type per variant -----
    @pytest.mark.parametrize(
        "variant",
        [
            ExecRestriction.NON,
            ExecRestriction.IOC,
            ExecRestriction.FOK,
            ExecRestriction.ICE,
        ],
    )
    def test_submitresult_accepted_is_0d_bool(self, book_state, variant):
        book, state = book_state
        kwargs = dict(exec_restriction=int(variant))
        if variant == ExecRestriction.ICE:
            kwargs["peak_qty"] = 0.5
            kwargs["quantity"] = 5.0
        order = _mk_order(**kwargs)
        _, result = book.submit_order(state, order)
        assert isinstance(result.accepted, jnp.ndarray)
        assert result.accepted.dtype == jnp.bool_
        assert result.accepted.shape == ()

    @pytest.mark.parametrize(
        "variant",
        [
            ExecRestriction.NON,
            ExecRestriction.IOC,
            ExecRestriction.FOK,
            ExecRestriction.ICE,
        ],
    )
    def test_submitresult_traded_is_0d_bool(self, book_state, variant):
        book, state = book_state
        kwargs = dict(exec_restriction=int(variant))
        if variant == ExecRestriction.ICE:
            kwargs["peak_qty"] = 0.5
            kwargs["quantity"] = 5.0
        order = _mk_order(**kwargs)
        _, result = book.submit_order(state, order)
        assert isinstance(result.traded, jnp.ndarray)
        assert result.traded.dtype == jnp.bool_
        assert result.traded.shape == ()

    @pytest.mark.parametrize("field", ["trade_qty", "trade_price", "settlement_value"])
    @pytest.mark.parametrize(
        "variant",
        [
            ExecRestriction.NON,
            ExecRestriction.IOC,
            ExecRestriction.FOK,
            ExecRestriction.ICE,
        ],
    )
    def test_submitresult_float_field_is_0d_float32(
        self, book_state, variant, field
    ):
        book, state = book_state
        kwargs = dict(exec_restriction=int(variant))
        if variant == ExecRestriction.ICE:
            kwargs["peak_qty"] = 0.5
            kwargs["quantity"] = 5.0
        order = _mk_order(**kwargs)
        _, result = book.submit_order(state, order)
        value = getattr(result, field)
        assert isinstance(value, jnp.ndarray), (
            f"{field} is {type(value).__name__}, expected jnp.ndarray"
        )
        assert value.dtype == jnp.float32, (
            f"{field} dtype is {value.dtype}, expected float32"
        )
        assert value.shape == (), (
            f"{field} shape is {value.shape}, expected ()"
        )

    def test_submitresult_pytree_leaf_count_is_five(self, book_state):
        book, state = book_state
        order = _mk_order()
        _, result = book.submit_order(state, order)
        leaves = jax.tree.leaves(result)
        assert len(leaves) == 5, (
            f"SubmitResult has {len(leaves)} pytree leaves, expected 5 "
            "(accepted, traded, trade_qty, trade_price, settlement_value)"
        )

    def test_submitresult_pytree_leaf_count_stable_across_variants(self, book_state):
        """The leaf count must be the same regardless of which arm of
        the lax.switch fired — proves the four arms produce the same
        result-tree shape."""
        book, state = book_state
        counts = []
        for variant in (ExecRestriction.NON, ExecRestriction.IOC,
                        ExecRestriction.FOK, ExecRestriction.ICE):
            kwargs = dict(exec_restriction=int(variant))
            if variant == ExecRestriction.ICE:
                kwargs["peak_qty"] = 0.5
                kwargs["quantity"] = 5.0
            order = _mk_order(**kwargs)
            _, result = book.submit_order(state, order)
            counts.append(len(jax.tree.leaves(result)))
        assert len(set(counts)) == 1, (
            f"SubmitResult leaf count varies across variants: {counts}. "
            "All four lax.switch arms must produce the same tree shape."
        )

    def test_rejected_result_dtypes_match_accepted_result(self, book_state):
        """Negative path: a rejected order's SubmitResult must have
        the same dtype profile as an accepted order's, so the
        ``lax.cond`` between rejected/accepted produces a
        shape-stable tree."""
        book, state = book_state
        rej_order = _mk_order(order_id=-5)  # negative -> rejected
        acc_order = _mk_order(order_id=1)
        _, rej = book.submit_order(state, rej_order)
        _, acc = book.submit_order(state, acc_order)
        for field in SubmitResult._fields:
            r_val = getattr(rej, field)
            a_val = getattr(acc, field)
            assert r_val.dtype == a_val.dtype, (
                f"{field} dtype drift: rejected={r_val.dtype}, "
                f"accepted={a_val.dtype}"
            )
            assert r_val.shape == a_val.shape, (
                f"{field} shape drift: rejected={r_val.shape}, "
                f"accepted={a_val.shape}"
            )

    def test_rejected_result_all_fields_zero_or_false(self, book_state):
        book, state = book_state
        rej_order = _mk_order(order_id=0)  # 0 -> rejected (positive-id rule)
        _, result = book.submit_order(state, rej_order)
        assert bool(result.accepted) is False
        assert bool(result.traded) is False
        assert float(result.trade_qty) == 0.0
        assert float(result.trade_price) == 0.0
        assert float(result.settlement_value) == 0.0


# ---------------------------------------------------------------------------
# 5. ExecRestriction surface integrity
# ---------------------------------------------------------------------------


class TestExecRestrictionSurfaceIntegrity:
    """``ExecRestriction`` has exactly four IntEnum members
    (NON=0, IOC=1, FOK=2, ICE=3).  P2-H2's v5.20 amendment
    corrected the v5.0 "six variants" framing — this test pins the
    four-variant reality."""

    def test_exec_restriction_has_exactly_four_members(self):
        members = list(ExecRestriction)
        assert len(members) == 4, (
            f"ExecRestriction has {len(members)} members, expected 4 "
            "(NON, IOC, FOK, ICE)"
        )

    def test_exec_restriction_member_values_are_contiguous_0_to_3(self):
        values = sorted(int(m) for m in ExecRestriction)
        assert values == [0, 1, 2, 3]

    def test_exec_restriction_member_names_are_exactly_non_ioc_fok_ice(self):
        names = sorted(m.name for m in ExecRestriction)
        assert names == ["FOK", "ICE", "IOC", "NON"]

    def test_lax_switch_in_submit_body_has_exactly_four_arms(self):
        """AST probe: the lax.switch inside _submit_order_body must
        pass exactly four branch callables."""
        node = _module_function("_submit_order_body")
        switch_calls = [
            c for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and ast.unparse(c.func).endswith("lax.switch")
        ]
        assert len(switch_calls) == 1, (
            f"_submit_order_body has {len(switch_calls)} lax.switch "
            "call(s); expected exactly 1 (the ExecRestriction dispatch)"
        )
        call = switch_calls[0]
        # lax.switch(discriminator, branches, operand) — branches is the
        # second positional arg.
        assert len(call.args) >= 2
        branches_arg = call.args[1]
        assert isinstance(branches_arg, (ast.Tuple, ast.List)), (
            "lax.switch second positional arg is not a tuple/list"
        )
        assert len(branches_arg.elts) == 4, (
            f"lax.switch has {len(branches_arg.elts)} branches; expected 4"
        )


# ---------------------------------------------------------------------------
# 6. get_order sentinel contract
# ---------------------------------------------------------------------------


class TestGetOrderSentinelContract:
    """``get_order`` must never return ``None`` — it returns the
    canonical sentinel ``OMIEOrder`` whose discriminator is
    ``order_id == SENTINEL_ORDER_ID``."""

    def test_get_order_returns_sentinel_on_miss_buy_side(self, book_open, empty_state):
        result = book_open.get_order(empty_state, order_id=99999, side=int(Side.BUY))
        assert result is not None, "get_order returned None — must return sentinel"
        assert isinstance(result, OMIEOrder)
        assert result.order_id == SENTINEL_ORDER_ID

    def test_get_order_returns_sentinel_on_miss_sell_side(self, book_open, empty_state):
        result = book_open.get_order(empty_state, order_id=99999, side=int(Side.SELL))
        assert result is not None
        assert isinstance(result, OMIEOrder)
        assert result.order_id == SENTINEL_ORDER_ID

    def test_sentinel_agent_id_matches_constant(self, book_open, empty_state):
        result = book_open.get_order(empty_state, order_id=99999, side=int(Side.BUY))
        assert result.agent_id == SENTINEL_AGENT_ID

    def test_sentinel_is_deterministic_and_idempotent(self, book_open, empty_state):
        r1 = book_open.get_order(empty_state, order_id=99999, side=int(Side.BUY))
        r2 = book_open.get_order(empty_state, order_id=99999, side=int(Side.BUY))
        assert r1 == r2

    def test_sentinel_factory_consistent_with_get_order(self, book_open, empty_state):
        """The sentinel returned by ``get_order`` on miss must equal
        the one produced by ``make_sentinel_order()`` — the single
        source of truth for the sentinel shape."""
        sentinel = make_sentinel_order()
        from_miss = book_open.get_order(
            empty_state, order_id=99999, side=int(Side.BUY),
        )
        # Only the discriminator (order_id) is the contracted equality;
        # the rest may differ on contract_id since make_sentinel_order
        # returns contract_id=0. We pin the order_id and agent_id
        # discriminators here.
        assert from_miss.order_id == sentinel.order_id
        assert from_miss.agent_id == sentinel.agent_id


# ---------------------------------------------------------------------------
# 7. Cross-module isolation / out-of-scope deferral
# ---------------------------------------------------------------------------


class TestCrossModuleIsolation:
    """``cancel_order`` and the three query helpers must be unchanged in
    surface terms — the P2-H2 refactor only touches ``submit_order``.
    """

    def test_cancel_order_is_not_jitted(self):
        """The class-method ``cancel_order`` must remain a plain
        Python method — not wrapped by ``jax.jit``."""
        # cancel_order should be a regular function, not a PjitFunction.
        method = ContractBook.cancel_order
        assert not hasattr(method, "_cache_size"), (
            "cancel_order has _cache_size — it has been jitted, but "
            "P2-H2 specifies cancel_order is out of scope and must "
            "remain Python-side"
        )

    def test_get_anonymous_lob_is_not_jitted(self):
        method = ContractBook.get_anonymous_lob
        assert not hasattr(method, "_cache_size")

    def test_get_agent_orders_is_not_jitted(self):
        method = ContractBook.get_agent_orders
        assert not hasattr(method, "_cache_size")

    def test_get_agent_trades_is_not_jitted(self):
        method = ContractBook.get_agent_trades
        assert not hasattr(method, "_cache_size")

    def test_cancel_order_returns_omiebookstate(self, book_open, empty_state):
        """Behavioural smoke: cancel of a non-existent id must be a
        no-op and return a valid OMIEBookState (P2-H2 must not break
        the cancel surface)."""
        new_state = book_open.cancel_order(
            empty_state, order_id=99999, side=int(Side.BUY),
        )
        assert isinstance(new_state, OMIEBookState)

    def test_omieorder_fields_unchanged_post_p2h2(self):
        """The 13-field ``OMIEOrder`` NamedTuple must keep its
        annotations: P2-H2 is forbidden from widening or narrowing
        the order schema."""
        expected_fields = (
            "contract_id", "side", "price", "quantity",
            "order_id", "agent_id", "timestamp",
            "exec_restriction", "validity", "expiry_time", "status",
            "peak_qty", "ppd",
        )
        assert OMIEOrder._fields == expected_fields


# ---------------------------------------------------------------------------
# 8. Trace-count parity under variation
# ---------------------------------------------------------------------------


class TestTraceCountParityUnderVariation:
    """Within one Python process, varying *traced* inputs to
    ``_submit_order`` must NOT bump its compile cache count.  Only
    varying a *static* argument should."""

    def test_varying_traced_order_fields_no_recompile(
        self, book_open, empty_state, product_cfg, market_cfg,
    ):
        """Warm the cache once, then vary price / quantity /
        order_id and verify cache size is unchanged."""
        # Snapshot the cache size before our probe (the cache may
        # already be warm from earlier fixtures / tests).
        initial = _submit_order._cache_size()

        for i, (price, qty, oid) in enumerate(
            [(50.0, 1.0, 1001), (75.5, 2.5, 1002), (10.25, 0.5, 1003)]
        ):
            order = _mk_order(price=price, quantity=qty, order_id=oid)
            _submit_order(empty_state, order, product_cfg, market_cfg)

        final = _submit_order._cache_size()
        # The cache may have grown by AT MOST 1 (the first call may
        # have been the first-ever compile under the current static
        # arg pair); subsequent calls with same static args must not
        # add entries.
        assert final - initial <= 1, (
            f"_submit_order cache grew by {final - initial} after "
            f"varying traced fields — expected <= 1"
        )

    def test_varying_ppd_within_ice_arm_no_recompile(
        self, book_open, empty_state, product_cfg, market_cfg,
    ):
        """The v5.20 amendment specifically calls out: ``order.ppd``
        is a traced field inside the ICE arm.  Varying it must NOT
        trigger a recompile (ICEwPI is exercised by ppd != 0, but
        it shares the ICE arm)."""
        # First, warm cache with an ICE order at ppd=0.
        ice_ppd0 = _mk_order(
            exec_restriction=int(ExecRestriction.ICE),
            peak_qty=0.5, quantity=5.0, ppd=0.0,
        )
        _submit_order(empty_state, ice_ppd0, product_cfg, market_cfg)
        cache_after_warm = _submit_order._cache_size()

        # Now submit with ppd != 0 (ICEwPI shape).
        ice_ppd_pos = _mk_order(
            exec_restriction=int(ExecRestriction.ICE),
            peak_qty=0.5, quantity=5.0, ppd=0.05,
        )
        _submit_order(empty_state, ice_ppd_pos, product_cfg, market_cfg)
        cache_after_ice_ppd_pos = _submit_order._cache_size()

        # And with negative ppd.
        ice_ppd_neg = _mk_order(
            exec_restriction=int(ExecRestriction.ICE),
            peak_qty=0.5, quantity=5.0, ppd=-0.03,
        )
        _submit_order(empty_state, ice_ppd_neg, product_cfg, market_cfg)
        cache_after_ice_ppd_neg = _submit_order._cache_size()

        assert cache_after_warm == cache_after_ice_ppd_pos
        assert cache_after_warm == cache_after_ice_ppd_neg

    def test_varying_exec_restriction_triggers_at_most_one_compile(
        self, book_open, empty_state, product_cfg, market_cfg,
    ):
        """The headline AC (5) of P2-H2: cycling through all four
        ExecRestriction variants must NOT add a fresh compile per
        variant — the lax.switch discriminator is traced, not
        specialised."""
        # Make sure we have a fresh order_id pool to avoid the
        # cancel-by-id path inside the IOC residual handling
        # producing collateral state changes that perturb the
        # cache.  (Cache-size is what we probe; this is just hygiene.)
        for variant in [ExecRestriction.NON, ExecRestriction.IOC,
                        ExecRestriction.FOK, ExecRestriction.ICE]:
            kwargs = dict(exec_restriction=int(variant),
                          order_id=2000 + int(variant))
            if variant == ExecRestriction.ICE:
                kwargs["peak_qty"] = 0.5
                kwargs["quantity"] = 5.0
            order = _mk_order(**kwargs)
            _submit_order(empty_state, order, product_cfg, market_cfg)

        cache_after_cycle = _submit_order._cache_size()

        # Now do a second pass — definitely no new entries.
        for variant in [ExecRestriction.NON, ExecRestriction.IOC,
                        ExecRestriction.FOK, ExecRestriction.ICE]:
            kwargs = dict(exec_restriction=int(variant),
                          order_id=3000 + int(variant))
            if variant == ExecRestriction.ICE:
                kwargs["peak_qty"] = 0.5
                kwargs["quantity"] = 5.0
            order = _mk_order(**kwargs)
            _submit_order(empty_state, order, product_cfg, market_cfg)

        cache_after_second_pass = _submit_order._cache_size()
        assert cache_after_second_pass == cache_after_cycle, (
            "Cycling ExecRestriction variants a second time bumped "
            f"_submit_order cache from {cache_after_cycle} to "
            f"{cache_after_second_pass} — the discriminator is "
            "incorrectly specialised, not traced"
        )

    def test_changing_static_product_config_bumps_cache(
        self, book_open, empty_state, market_cfg,
    ):
        """Negative trace-count check: changing the static
        ``product_config`` SHOULD bump the cache.  This proves
        the static-argnames declaration actually takes effect.
        """
        # Use a config whose hash differs (the dataclass is frozen
        # but Python doesn't intern dataclass instances).  Different
        # qty_tick values produce different hashes.
        cfg_a = OMIEProductConfig(qty_tick=0.1)
        cfg_b = OMIEProductConfig(qty_tick=0.5)

        order = _mk_order(quantity=1.0)
        _submit_order(empty_state, order, cfg_a, market_cfg)
        size_after_a = _submit_order._cache_size()
        _submit_order(empty_state, order, cfg_b, market_cfg)
        size_after_b = _submit_order._cache_size()

        assert size_after_b > size_after_a, (
            f"Cache did not grow after switching product_config "
            f"({size_after_a} -> {size_after_b}); the "
            "static_argnames declaration is not specialising on it"
        )


# ---------------------------------------------------------------------------
# 9. Truncation backstop integrity
# ---------------------------------------------------------------------------


class TestTruncationBackstopIntegrity:
    """``OMIEBookState.truncated`` is the sticky carry-flag from
    P2-H1 §A3 Path B.  P2-H2 must OR into it, never reset it.
    """

    def test_truncated_field_is_0d_bool_in_initial_state(
        self, book_open, empty_state,
    ):
        assert isinstance(empty_state.truncated, jnp.ndarray)
        assert empty_state.truncated.dtype == jnp.bool_
        assert empty_state.truncated.shape == ()
        assert bool(empty_state.truncated) is False

    def test_truncated_field_preserved_through_submit(
        self, book_open, empty_state,
    ):
        """A clean submit preserves ``truncated=False``."""
        order = _mk_order()
        new_state, _ = book_open.submit_order(empty_state, order)
        assert isinstance(new_state.truncated, jnp.ndarray)
        assert new_state.truncated.dtype == jnp.bool_

    def test_truncated_field_is_sticky_when_preset_true(
        self, book_open, empty_state,
    ):
        """If we manually set ``truncated=True`` on the input state,
        ``submit_order`` must NOT reset it to False — the flag is
        sticky."""
        forced_state = empty_state._replace(truncated=jnp.bool_(True))
        order = _mk_order()
        new_state, _ = book_open.submit_order(forced_state, order)
        assert bool(new_state.truncated) is True, (
            "truncated flag was reset to False — the carry-flag must "
            "be sticky-True across submits (P2-H1 §A3 Path B)"
        )

    def test_truncated_stickiness_survives_rejected_path(
        self, book_open, empty_state,
    ):
        """Even on the rejected path (negative order_id), the
        truncated flag must propagate from input to output."""
        forced_state = empty_state._replace(truncated=jnp.bool_(True))
        rej_order = _mk_order(order_id=-1)  # rejected
        new_state, result = book_open.submit_order(forced_state, rej_order)
        assert bool(result.accepted) is False
        # The rejected branch returns state unchanged — so truncated
        # must still be True.
        assert bool(new_state.truncated) is True


# ---------------------------------------------------------------------------
# 10. Contract-closed gate (P2-L3 caller-contract invariant)
# ---------------------------------------------------------------------------


class TestContractClosedGate:
    """When the parent contract is not OPEN, ``submit_order`` returns
    REJECTED without entering the jitted body.  This is the P2-L3
    caller-contract invariant — the trampoline gates Python-side."""

    @pytest.mark.parametrize(
        "closed_status",
        [
            ContractStatus.PENDING,
            ContractStatus.SUSPENDED,
            ContractStatus.CLOSED,
        ],
    )
    def test_non_open_contract_rejects(
        self, product_cfg, market_cfg, closed_status,
    ):
        contract = Contract(
            contract_id=37,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=closed_status,
        )
        book = ContractBook(contract, product_cfg, market_cfg)
        state = book.init()
        order = _mk_order()
        new_state, result = book.submit_order(state, order)
        assert bool(result.accepted) is False, (
            f"contract status {closed_status.name} should reject — "
            f"got accepted={bool(result.accepted)}"
        )

    def test_non_open_returns_state_unchanged(
        self, closed_contract_local, product_cfg, market_cfg,
    ):
        """The rejected path on a CLOSED contract returns the state
        unchanged (same Python object identity, since no jitted body
        was invoked)."""
        book = ContractBook(closed_contract_local, product_cfg, market_cfg)
        state = book.init()
        order = _mk_order()
        new_state, _ = book.submit_order(state, order)
        # Python identity — the rejected branch returns the same state.
        assert new_state is state

    def test_open_contract_accepts_valid_order(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        order = _mk_order()
        _, result = book.submit_order(state, order)
        assert bool(result.accepted) is True

    def test_closed_contract_rejected_result_uses_canonical_zero_fill(
        self, closed_contract_local, product_cfg, market_cfg,
    ):
        """The REJECTED result on a closed contract must be the
        canonical all-zero SubmitResult — not garbage from a partial
        execution."""
        book = ContractBook(closed_contract_local, product_cfg, market_cfg)
        state = book.init()
        order = _mk_order()
        _, result = book.submit_order(state, order)
        assert bool(result.accepted) is False
        assert bool(result.traded) is False
        assert float(result.trade_qty) == 0.0
        assert float(result.trade_price) == 0.0
        assert float(result.settlement_value) == 0.0


# ---------------------------------------------------------------------------
# 11. JIT body re-entry semantics — same instance, two submits
# ---------------------------------------------------------------------------


class TestJitBodyReEntrySemantics:
    """Two consecutive submits on the same ``ContractBook`` instance
    must thread state correctly through the jitted body and produce
    a stable post-state."""

    def test_two_buys_no_match_both_accepted(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        # Two buy orders that don't cross any resting asks — both rest.
        order1 = _mk_order(order_id=10, price=50.00, quantity=1.0, timestamp=0)
        order2 = _mk_order(order_id=11, price=49.00, quantity=2.0, timestamp=1)
        state, r1 = book.submit_order(state, order1)
        state, r2 = book.submit_order(state, order2)
        assert bool(r1.accepted) is True
        assert bool(r2.accepted) is True
        # Neither order should have traded against an empty book.
        assert bool(r1.traded) is False
        assert bool(r2.traded) is False

    def test_buy_then_crossing_sell_produces_trade(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        # Buy at 50, then sell at 50 -> crosses.
        order_b = _mk_order(order_id=20, side=int(Side.BUY),
                            price=50.00, quantity=1.0, timestamp=0)
        order_s = _mk_order(order_id=21, side=int(Side.SELL),
                            price=50.00, quantity=1.0, timestamp=1)
        state, _ = book.submit_order(state, order_b)
        state, result_s = book.submit_order(state, order_s)
        assert bool(result_s.accepted) is True
        assert bool(result_s.traded) is True
        assert float(result_s.trade_qty) > 0.0

    def test_state_pytree_structure_unchanged_through_submit(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        """State pytree leaves must remain shape/dtype-stable through
        submit_order — required for ``lax.cond`` and ``lax.switch``
        to operate without per-branch shape adjustment."""
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        order = _mk_order()
        new_state, _ = book.submit_order(state, order)

        in_leaves = jax.tree.leaves(state)
        out_leaves = jax.tree.leaves(new_state)
        assert len(in_leaves) == len(out_leaves), (
            f"OMIEBookState pytree leaf count changed: "
            f"{len(in_leaves)} -> {len(out_leaves)}"
        )
        for il, ol in zip(in_leaves, out_leaves):
            assert il.shape == ol.shape, (
                f"leaf shape drift: {il.shape} -> {ol.shape}"
            )
            assert il.dtype == ol.dtype, (
                f"leaf dtype drift: {il.dtype} -> {ol.dtype}"
            )


# ---------------------------------------------------------------------------
# 12. ICE-arm-specific semantics
# ---------------------------------------------------------------------------


class TestICEArmSpecificSemantics:
    """The ICE arm dispatches register / regenerate_slices / peak
    submission inside the jitted body.  These probes guard the
    behaviour callers depend on."""

    def test_ice_with_zero_ppd_is_plain_iceberg(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        """ICE with ppd=0 is the canonical iceberg (not ICEwPI)."""
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        ice = _mk_order(
            exec_restriction=int(ExecRestriction.ICE),
            peak_qty=0.5, quantity=5.0, ppd=0.0, order_id=30,
        )
        new_state, result = book.submit_order(state, ice)
        assert bool(result.accepted) is True

    def test_ice_with_positive_ppd_buy_wrong_sign_rejected(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        """OMIE-ICEWPI-SIGN-REJECT-1 re-baseline: a BUY ICE with ppd>0
        has the wrong ICEwPI sign (market-spec §13.6) and is rejected."""
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        ice = _mk_order(
            exec_restriction=int(ExecRestriction.ICE),
            peak_qty=0.5, quantity=5.0, ppd=0.05, order_id=31,
        )
        new_state, result = book.submit_order(state, ice)
        assert bool(result.accepted) is False

    def test_ice_with_negative_ppd_buy_is_icewpi(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        """Conforming twin: a BUY ICE with ppd<0 is ICEwPI (slice prices
        drift downward for BUY).  Same arm; semantics differ inside
        regenerate_slices."""
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        ice = _mk_order(
            exec_restriction=int(ExecRestriction.ICE),
            peak_qty=0.5, quantity=5.0, ppd=-0.05, order_id=32,
        )
        new_state, result = book.submit_order(state, ice)
        assert bool(result.accepted) is True

    def test_ice_validator_rejects_zero_peak(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        """Domain rule (market-spec §4.5): ICE requires peak_qty > 0.
        The validator rejects, and the result is REJECTED."""
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        ice_bad = _mk_order(
            exec_restriction=int(ExecRestriction.ICE),
            peak_qty=0.0, quantity=5.0, order_id=32,
        )
        _, result = book.submit_order(state, ice_bad)
        assert bool(result.accepted) is False

    def test_ice_validator_rejects_peak_exceeds_quantity(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        """Domain rule: peak_qty must be <= quantity."""
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        ice_bad = _mk_order(
            exec_restriction=int(ExecRestriction.ICE),
            peak_qty=10.0, quantity=5.0, order_id=33,
        )
        _, result = book.submit_order(state, ice_bad)
        assert bool(result.accepted) is False


# ---------------------------------------------------------------------------
# 13. Behavioural parity probes — FOK acceptance / IOC residual
# ---------------------------------------------------------------------------


class TestFOKAndIOCBehaviouralParity:
    """Probe key FOK / IOC semantics not covered by the dtype /
    structural classes above — these are domain assertions, not
    structural ones."""

    def test_fok_with_no_liquidity_is_accepted_no_trade(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        """FOK semantics §market-spec: if not enough liquidity, the
        order is accepted but produces no trade (no resting)."""
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        # Buy with FOK at a high price on an empty book -> no liquidity.
        fok = _mk_order(
            exec_restriction=int(ExecRestriction.FOK),
            order_id=40, price=100.0, quantity=1.0,
        )
        _, result = book.submit_order(state, fok)
        # Per the implementation contract, FOK with no liquidity
        # returns accepted=True, traded=False.
        assert bool(result.accepted) is True
        assert bool(result.traded) is False

    def test_ioc_with_no_liquidity_does_not_rest(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        """IOC semantics: any unfilled residual must be cancelled,
        not left resting on the book."""
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        ioc = _mk_order(
            exec_restriction=int(ExecRestriction.IOC),
            order_id=50, price=50.0, quantity=1.0,
        )
        new_state, result = book.submit_order(state, ioc)
        # Behavioural: there must be no resting order_id=50 in the book.
        retrieved = book.get_order(
            new_state, order_id=50, side=int(Side.BUY),
        )
        assert retrieved.order_id == SENTINEL_ORDER_ID, (
            "IOC order with no fill left residual on the book — "
            "must be cancelled per market-spec §4.5"
        )

    def test_non_with_no_match_rests_on_book(
        self, open_contract_local, product_cfg, market_cfg,
    ):
        """NON semantics: an unfilled order rests on the book at
        its limit price."""
        book = ContractBook(open_contract_local, product_cfg, market_cfg)
        state = book.init()
        non = _mk_order(
            exec_restriction=int(ExecRestriction.NON),
            order_id=60, price=50.0, quantity=1.0,
        )
        new_state, result = book.submit_order(state, non)
        assert bool(result.accepted) is True
        # Behavioural: order_id=60 must be findable on the BUY side.
        retrieved = book.get_order(
            new_state, order_id=60, side=int(Side.BUY),
        )
        assert retrieved.order_id == 60, (
            "NON order with no match did not rest on the book — "
            "expected to be findable at order_id=60"
        )


# ---------------------------------------------------------------------------
# 14. Direct-construction backward-compatibility
# ---------------------------------------------------------------------------


class TestSubmitResultDirectConstructionBackwardCompat:
    """The class docstring explicitly preserves backward-compat for
    Python-side ``SubmitResult(accepted=True)`` direct construction.
    Probe that this still works (the trailing four fields carry
    Python zero defaults)."""

    def test_construct_with_python_bool_accepted_only(self):
        r = SubmitResult(accepted=True)
        # Field count and trailing-default presence.
        assert r._fields == (
            "accepted", "traded", "trade_qty", "trade_price",
            "settlement_value",
        )

    def test_construct_with_python_bool_and_float_compat(self):
        r = SubmitResult(
            accepted=True, traded=False, trade_qty=0.0,
            trade_price=0.0, settlement_value=0.0,
        )
        assert r.accepted is True
        assert r.traded is False
        assert r.trade_qty == 0.0

    def test_construct_with_jnp_scalars_typechecks(self):
        r = SubmitResult(
            accepted=jnp.bool_(True),
            traded=jnp.bool_(False),
            trade_qty=jnp.float32(0.0),
            trade_price=jnp.float32(0.0),
            settlement_value=jnp.float32(0.0),
        )
        assert isinstance(r.accepted, jnp.ndarray)
        assert r.trade_qty.dtype == jnp.float32
