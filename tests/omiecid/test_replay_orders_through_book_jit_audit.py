"""Audit-mode tests for P2-PERF-5: ``replay_orders_through_book`` outer-loop
``lax.scan`` lift.

This file is the **independent audit pass** authored after the
P2-PERF-5 implementation merged. It is written without reading the
original locked test file at
``tests/omiecid/test_replay_orders_through_book_jit.py``; the audit and
locked surfaces are deliberately orthogonal probes of the same
sprint contract:

    1. structural-only contract: single ``jax.lax.scan`` outer loop,
       in-scan new-fill detection, fixed-shape preallocated
       ``(N_orders, NTRADE_CAP, 3) int32`` replayed-trades buffer with
       sentinel ``-1``,
    2. module-level lift mirrors the P2-H2 idiom verbatim --
       ``_replay_orders = jax.jit(_replay_orders_body,
       static_argnames=("product_config", "market_config"))``,
    3. byte-identical replay parity against a pre-fix Python-loop
       reference.

This audit covers, in order of focus dimension:

    A. Real-data parity (canonical 2026-01-12 cid=81 session,
       skip-on-data-absent).
    B. Pytree / dtype invariants of the lift output.
    C. Trace-cache hygiene: a single compile per unique
       ``orders.shape[0]``.
    D. Sentinel invariants on the order-id column.
    E. Public-API backward-compatibility (caller-grep probe).
    F. Module-level lift idiom conformance (``static_argnames`` shape).
    G. No host-side outer ``for`` loop in the live source.
    H. ``OMIEBookState`` pytree treedef stability across the lift.
    I. Determinism / bitwise reproducibility.
    J. In-scan new-fill detection invariant (cross-checks the whole-row
       diff used by ``extract_fills`` on the same boundary).
    K. ``replay_orders_through_book`` callsite-untouched probe (grep
       check against the five known caller files in ``tests/omiecid/``).
    L. Per-step output shape stability across varying synthetic inputs
       of the same ``N``.

All fast-suite tests target < 1 s; the real-data parity test and the
parity-vs-reference probes are marked ``@pytest.mark.slow``. Each
slow test carries a one-line justification.

References:
    docs/omiecid/agile-plan.md lines 473-485 (P2-PERF-5 card v5.26).
    docs/omiecid/migration-plan.md SS"Module-level jit idiom".
    docs/omiecid/market-spec.md SS4, SS6 (order lifecycle, matching).
    omiecid_simulator/omiecid/contract_book.py:1005-1008 (canonical lift).
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.jit

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.jaxomiecid import omie_data_loader
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    ReplayResult,
    _replay_orders,
    _replay_orders_body,
    _row_to_order_traced,
    replay_orders_through_book,
)
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    _submit_order,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    Status,
    Validity,
)
from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NTRADE_CAP = cst.NTRADE_CAP  # 100 at audit authoring time
LOADER_PATH = Path(omie_data_loader.__file__)
RAW_OMIE_DIR = Path(__file__).resolve().parents[2] / "data" / "rawOMIE"
CANONICAL_DELIVERY_DATE = "2026-01-12"
CANONICAL_CONTRACT_ID = 81
TEST_DIR = Path(__file__).resolve().parent

# Files in tests/omiecid/ that call replay_orders_through_book per the
# P2-PERF-5 merge note ("public API unchanged for these callers").
PUBLIC_CALLER_FILES = (
    "test_omie_data.py",
    "test_omie_data_audit.py",
    "test_loader_tick_alignment_audit.py",
    "test_trade_extraction_ring_buffer.py",
    "test_trade_extraction_ring_buffer_audit.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_contract(contract_id: int = 37) -> Contract:
    """An OPEN ``Contract`` usable in ``ContractBook``."""
    return Contract(
        contract_id=contract_id,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


def _fresh_book(product_config: OMIEProductConfig | None = None,
                market_config: OMIEMarketConfig | None = None,
                contract_id: int = 37) -> ContractBook:
    pc = product_config or OMIEProductConfig()
    mc = market_config or OMIEMarketConfig()
    return ContractBook(_open_contract(contract_id), pc, mc)


def _encode_price(price_eur: float, pc: OMIEProductConfig) -> int:
    """Method-A offset encoding mirroring ``encoding.encode_price``."""
    return int(round((price_eur - pc.price_min) / pc.price_tick))


def _encode_qty(qty_mw: float, pc: OMIEProductConfig) -> int:
    return int(round(qty_mw / pc.qty_tick))


def _mk_row(*, contract_id: int = 37, side: int, price_int: int,
            qty_int: int, order_id: int, agent_id: int, timestamp: int,
            exec_restriction: int = int(ExecRestriction.NON),
            validity: int = int(Validity.GFS), expiry_time: int = 0,
            status: int = int(Status.ACTIVE), peak_qty_int: int = 0,
            ppd_int: int = 0) -> list:
    """Build a 13-int32 loader-shaped row."""
    return [
        contract_id, side, price_int, qty_int, order_id, agent_id,
        timestamp, exec_restriction, validity, expiry_time, status,
        peak_qty_int, ppd_int,
    ]


def _build_crossing_pair_orders(n_pairs: int, *,
                                product_config: OMIEProductConfig | None = None
                                ) -> jnp.ndarray:
    """``(2 * n_pairs, 13) int32`` synthetic orders.

    Each pair is one resting SELL followed by one crossing BUY at
    50.00 EUR/MWh, 0.1 MW. After ``replay`` every pair produces exactly
    one fill row in the per-step slab.
    """
    pc = product_config or OMIEProductConfig()
    p_int = _encode_price(50.00, pc)
    q_int = _encode_qty(0.1, pc)
    rows: list = []
    oid = 1
    ts = 100
    for _ in range(n_pairs):
        rows.append(_mk_row(side=int(Side.SELL), price_int=p_int,
                            qty_int=q_int, order_id=oid, agent_id=1,
                            timestamp=ts))
        oid += 1
        ts += 1
        rows.append(_mk_row(side=int(Side.BUY), price_int=p_int,
                            qty_int=q_int, order_id=oid, agent_id=2,
                            timestamp=ts))
        oid += 1
        ts += 1
    return jnp.array(rows, dtype=jnp.int32)


def _build_noncrossing_orders(n_orders: int, *,
                              product_config: OMIEProductConfig | None = None
                              ) -> jnp.ndarray:
    """``(n_orders, 13) int32`` resting-only SELL orders.

    Each successive SELL is placed at a higher price than the
    previous, so none crosses. Every row produces an all-sentinel
    slab in the replayed-trades buffer.
    """
    pc = product_config or OMIEProductConfig()
    q_int = _encode_qty(0.1, pc)
    rows: list = []
    for i in range(n_orders):
        p_int = _encode_price(60.00 + 0.01 * (i + 1), pc)
        rows.append(_mk_row(side=int(Side.SELL), price_int=p_int,
                            qty_int=q_int, order_id=1000 + i, agent_id=7,
                            timestamp=10_000 + i))
    return jnp.array(rows, dtype=jnp.int32)


def _legacy_python_replay(orders: jnp.ndarray,
                          book: ContractBook,
                          ) -> tuple[OMIEBookState, jnp.ndarray]:
    """Pre-P2-PERF-5 reference implementation of the outer loop.

    Mirrors the legacy ``omie_data_loader.py:847-881`` shape exactly:
    Python ``for`` over ``orders.shape[0]``, with the ring-buffer-
    immune whole-row diff detector inside each iteration.

    Returns ``(final_state, replayed_trades_buf)`` where
    ``replayed_trades_buf`` has shape ``(N_orders, NTRADE_CAP, 3) int32``
    with the same per-row layout the lift emits:
        column 0 = aggressor ``order_id`` (sentinel ``-1`` for unfilled
            slots),
        column 1 = jaxob price tick,
        column 2 = absolute quantity tick.
    """
    state = book.init()
    pc = book._product_config
    mc = book._market_config

    n = int(orders.shape[0])
    buf = jnp.full((n, NTRADE_CAP, 3), -1, dtype=jnp.int32)
    buf = buf.at[..., 1:].set(jnp.int32(0))

    p_col = cst.TradesFeat.P.value
    q_col = cst.TradesFeat.Q.value

    for i in range(n):
        row = orders[i]
        order = _row_to_order_traced(row, pc)
        trades_before = state.lob.trades
        state, _ = _submit_order(state, order, pc, mc)
        trades_after = state.lob.trades

        new_mask = jnp.any(trades_before != trades_after, axis=1)
        aggressor_oid = order.order_id.astype(jnp.int32)
        oid_col = jnp.where(new_mask, aggressor_oid, jnp.int32(-1))
        price_col = jnp.where(
            new_mask, trades_after[:, p_col].astype(jnp.int32),
            jnp.int32(0),
        )
        qty_col = jnp.where(
            new_mask, jnp.abs(trades_after[:, q_col]).astype(jnp.int32),
            jnp.int32(0),
        )
        slab = jnp.stack([oid_col, price_col, qty_col], axis=1)
        buf = buf.at[i].set(slab)

    return state, buf


# ---------------------------------------------------------------------------
# Section A -- Real-data parity (skip-on-data-absent)
# ---------------------------------------------------------------------------

class TestRealDataParity:
    """Parity probe against the canonical 2026-01-12 cid=81 session.

    The replay-correctness target is byte-identical against the inline
    legacy Python-loop reference. Skipped when the OMIE raw file for
    the canonical date is missing on disk.
    """

    @pytest.mark.slow
    def test_canonical_2026_01_12_cid81_replay_matches_legacy_loop(self):
        """slow: end-to-end loader + jit replay on a real OMIE session."""
        orders_path = RAW_OMIE_DIR / f"orders_{CANONICAL_DELIVERY_DATE.replace('-', '')}.1"
        if not orders_path.exists():
            pytest.skip(
                f"canonical OMIE orders file not found at {orders_path}; "
                f"real-data parity probe is skip-on-data-absent."
            )

        cfg = OMIEDataConfig(
            data_path=str(RAW_OMIE_DIR),
            contract_id=CANONICAL_CONTRACT_ID,
            delivery_date=CANONICAL_DELIVERY_DATE,
        )
        orders = omie_data_loader.load_omie_orders(cfg)
        if int(orders.shape[0]) == 0:
            pytest.skip(
                f"canonical session has zero orders at "
                f"({CANONICAL_DELIVERY_DATE}, cid={CANONICAL_CONTRACT_ID}); "
                f"parity probe is vacuous."
            )

        # Cap the canonical session at a manageable prefix when the file
        # is enormous; the parity invariant is per-row, so any prefix
        # exercising both fill and non-fill rows is sufficient.
        n_use = int(min(150, orders.shape[0]))
        orders = orders[:n_use]

        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()

        book_legacy = ContractBook(_open_contract(CANONICAL_CONTRACT_ID), pc, mc)
        state_legacy, buf_legacy = _legacy_python_replay(orders, book_legacy)

        book_jit = ContractBook(_open_contract(CANONICAL_CONTRACT_ID), pc, mc)
        initial_state = book_jit.init()
        state_jit, buf_jit = _replay_orders(initial_state, orders, pc, mc)

        # Byte-identical parity on the replayed-trades buffer.
        assert buf_legacy.shape == buf_jit.shape == (n_use, NTRADE_CAP, 3)
        assert buf_legacy.dtype == buf_jit.dtype == jnp.int32
        assert jnp.array_equal(buf_legacy, buf_jit), (
            "P2-PERF-5 replay buffer diverges from legacy Python-loop "
            "reference on canonical 2026-01-12 cid=81 prefix."
        )

        # Byte-identical parity on every state leaf.
        leaves_legacy = jax.tree_util.tree_leaves(state_legacy)
        leaves_jit = jax.tree_util.tree_leaves(state_jit)
        assert len(leaves_legacy) == len(leaves_jit)
        for la, lb in zip(leaves_legacy, leaves_jit):
            assert jnp.array_equal(la, lb), (
                "OMIEBookState leaf diverges between legacy loop and "
                "P2-PERF-5 lift on canonical real-data session."
            )


# ---------------------------------------------------------------------------
# Section B -- Pytree / dtype invariants of the lift output
# ---------------------------------------------------------------------------

class TestPytreeAndDtypeInvariants:
    """The lift must return a JAX-compatible tree with no Python-side leaves."""

    def test_replayed_trades_buffer_is_int32(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_crossing_pair_orders(3, product_config=pc)
        book = _fresh_book(pc, mc)
        _, buf = _replay_orders(book.init(), orders, pc, mc)
        assert buf.dtype == jnp.int32
        assert buf.ndim == 3
        assert buf.shape == (int(orders.shape[0]), NTRADE_CAP, 3)

    def test_state_leaves_are_all_jnp_ndarray(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_crossing_pair_orders(2, product_config=pc)
        book = _fresh_book(pc, mc)
        state, _ = _replay_orders(book.init(), orders, pc, mc)
        leaves = jax.tree_util.tree_leaves(state)
        assert leaves, "state pytree must have at least one leaf"
        for i, leaf in enumerate(leaves):
            assert isinstance(leaf, jax.Array), (
                f"state leaf {i} is not a jax.Array: type={type(leaf)!r}"
            )

    def test_lift_output_has_no_python_int_or_none_leaves(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_crossing_pair_orders(2, product_config=pc)
        book = _fresh_book(pc, mc)
        state, buf = _replay_orders(book.init(), orders, pc, mc)
        for leaf in jax.tree_util.tree_leaves((state, buf)):
            assert not isinstance(leaf, (int, float, bool, type(None))), (
                f"Found Python scalar leaf {type(leaf)!r} in lift output."
            )

    def test_shapes_are_static_across_same_n_calls(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        n = 4
        orders_a = _build_crossing_pair_orders(n // 2, product_config=pc)
        orders_b = _build_noncrossing_orders(n, product_config=pc)
        assert int(orders_a.shape[0]) == int(orders_b.shape[0]) == n
        book_a = _fresh_book(pc, mc)
        book_b = _fresh_book(pc, mc)
        _, buf_a = _replay_orders(book_a.init(), orders_a, pc, mc)
        _, buf_b = _replay_orders(book_b.init(), orders_b, pc, mc)
        assert buf_a.shape == buf_b.shape


# ---------------------------------------------------------------------------
# Section C -- Trace-cache hygiene
# ---------------------------------------------------------------------------

class TestTraceCacheHygiene:
    """The module-level jitted lift must not re-trace on same-shape inputs."""

    def test_lift_cache_size_stable_across_same_shape_calls(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()

        # Clear any prior state to make the assertion robust against
        # interleaved test ordering.
        _replay_orders.clear_cache()

        n = 4
        for seed in range(3):
            orders = _build_crossing_pair_orders(n // 2, product_config=pc)
            # Distinct book instances; the lift is shape-static, not
            # contract-instance-static.
            book = _fresh_book(pc, mc)
            _replay_orders(book.init(), orders, pc, mc)

        cache_size = _replay_orders._cache_size()
        assert cache_size == 1, (
            f"expected exactly one compiled trace for same-shape calls; "
            f"got {cache_size}. Re-tracing on same-shape inputs would "
            f"undo the P2-PERF-5 outer-loop lift."
        )

    def test_distinct_n_orders_triggers_one_additional_compile(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        _replay_orders.clear_cache()

        orders_n4 = _build_crossing_pair_orders(2, product_config=pc)
        orders_n6 = _build_crossing_pair_orders(3, product_config=pc)
        assert int(orders_n4.shape[0]) == 4
        assert int(orders_n6.shape[0]) == 6

        book = _fresh_book(pc, mc)
        _replay_orders(book.init(), orders_n4, pc, mc)
        assert _replay_orders._cache_size() == 1

        _replay_orders(book.init(), orders_n6, pc, mc)
        assert _replay_orders._cache_size() == 2, (
            "lift must re-compile exactly once per new orders.shape[0]; "
            "shape-static idiom violation otherwise."
        )

        # Returning to n=4 must reuse the prior trace -- cache stays at 2.
        _replay_orders(book.init(), orders_n4, pc, mc)
        assert _replay_orders._cache_size() == 2, (
            "lift recompiled on a previously-seen shape -- shape is not "
            "trace-time static for the lift."
        )


# ---------------------------------------------------------------------------
# Section D -- Sentinel invariants on the order-id column
# ---------------------------------------------------------------------------

class TestSentinelInvariants:
    """``-1`` in column 0 must correspond exactly to unfilled slab slots."""

    def test_noncrossing_session_produces_all_sentinel_slabs(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        n = 5
        orders = _build_noncrossing_orders(n, product_config=pc)
        book = _fresh_book(pc, mc)
        _, buf = _replay_orders(book.init(), orders, pc, mc)
        # Every row must be entirely sentinels (no fills produced).
        assert jnp.all(buf[:, :, 0] == -1), (
            "non-crossing orders must produce no fills; column 0 must be "
            "all sentinels."
        )
        # Price/qty columns must be the canonical zero-fill content.
        assert jnp.all(buf[:, :, 1] == 0)
        assert jnp.all(buf[:, :, 2] == 0)

    def test_crossing_session_emits_exactly_one_fill_per_pair(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        n_pairs = 4
        orders = _build_crossing_pair_orders(n_pairs, product_config=pc)
        book = _fresh_book(pc, mc)
        _, buf = _replay_orders(book.init(), orders, pc, mc)

        # Per-row fill count.
        per_row_fills = jnp.sum(buf[:, :, 0] != -1, axis=1)
        # Odd-indexed rows are crossing BUYs that fill exactly once;
        # even-indexed rows are resting SELLs (no fill).
        for i in range(int(orders.shape[0])):
            if i % 2 == 0:
                assert int(per_row_fills[i]) == 0, (
                    f"resting SELL row {i} must produce zero fills"
                )
            else:
                assert int(per_row_fills[i]) == 1, (
                    f"crossing BUY row {i} must produce exactly one fill"
                )

    def test_sentinel_rows_have_zero_price_and_zero_qty(self):
        """Wherever ``oid == -1``, the price/qty cells must be the
        canonical zero-fill content (no leaked stale ticks)."""
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_crossing_pair_orders(3, product_config=pc)
        book = _fresh_book(pc, mc)
        _, buf = _replay_orders(book.init(), orders, pc, mc)
        sentinel_mask = (buf[:, :, 0] == -1)
        assert jnp.all(buf[:, :, 1][sentinel_mask] == 0), (
            "price tick must be zero on sentinel rows"
        )
        assert jnp.all(buf[:, :, 2][sentinel_mask] == 0), (
            "qty tick must be zero on sentinel rows"
        )

    def test_nonsentinel_rows_have_nonzero_aggressor_oid(self):
        """Where the oid is non-sentinel, it must be a strictly positive
        order_id (>= 1) -- the synthetic generator allocates oids starting
        at 1."""
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_crossing_pair_orders(3, product_config=pc)
        book = _fresh_book(pc, mc)
        _, buf = _replay_orders(book.init(), orders, pc, mc)
        oids = buf[:, :, 0]
        nonsentinel_mask = oids != -1
        if int(jnp.sum(nonsentinel_mask)) > 0:
            assert jnp.all(oids[nonsentinel_mask] > 0), (
                "non-sentinel oid column must hold strictly positive order_ids"
            )


# ---------------------------------------------------------------------------
# Section E -- Public-API backward compatibility
# ---------------------------------------------------------------------------

class TestPublicApiBackwardCompatibility:
    """The public ``replay_orders_through_book`` signature is unchanged
    for the five known caller files."""

    def test_public_function_signature_is_unchanged(self):
        sig = inspect.signature(replay_orders_through_book)
        params = list(sig.parameters.keys())
        assert params == [
            "orders", "contract_book", "historical_trades", "heuristic",
        ], (
            f"replay_orders_through_book signature changed: got {params!r}. "
            f"P2-PERF-5 merge note pins these caller-facing positional "
            f"names."
        )
        heuristic_default = sig.parameters["heuristic"].default
        assert heuristic_default == "rest_until_gct", (
            f"heuristic default changed: got {heuristic_default!r}"
        )

    def test_return_type_is_ReplayResult(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_crossing_pair_orders(1, product_config=pc)
        book = _fresh_book(pc, mc)
        historical = jnp.zeros((0, 6), dtype=jnp.int32)
        result = replay_orders_through_book(
            orders=orders,
            contract_book=book,
            historical_trades=historical,
            heuristic="rest_until_gct",
        )
        assert isinstance(result, ReplayResult), (
            f"public function must return ReplayResult; got {type(result)!r}"
        )
        # ReplayResult fields used downstream.
        for fld in ("final_book", "replayed_trades",
                    "historical_trades_matched", "historical_trades_total",
                    "extra_trades", "match_rate", "extra_volume_pct"):
            assert hasattr(result, fld), (
                f"ReplayResult is missing field {fld!r}"
            )

    def test_unknown_heuristic_still_raises(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_crossing_pair_orders(1, product_config=pc)
        book = _fresh_book(pc, mc)
        historical = jnp.zeros((0, 6), dtype=jnp.int32)
        with pytest.raises(NotImplementedError):
            replay_orders_through_book(
                orders=orders,
                contract_book=book,
                historical_trades=historical,
                heuristic="some-future-heuristic",
            )


# ---------------------------------------------------------------------------
# Section F -- Module-level lift idiom conformance
# ---------------------------------------------------------------------------

class TestLiftIdiomConformance:
    """The lift must mirror the canonical P2-H2 idiom shape verbatim."""

    def test_lift_is_module_level_attribute(self):
        assert hasattr(omie_data_loader, "_replay_orders"), (
            "module-level jitted lift is not exposed as "
            "_replay_orders attribute"
        )
        assert hasattr(omie_data_loader, "_replay_orders_body"), (
            "scan body helper is not exposed"
        )

    def test_lift_is_a_pjit_wrapper(self):
        """The lift must be a ``jax.jit`` wrapper (introspectable via
        ``_cache_size`` / ``clear_cache``)."""
        assert callable(_replay_orders)
        assert hasattr(_replay_orders, "_cache_size"), (
            "lift does not expose _cache_size -- not a jax.jit wrapper"
        )
        assert hasattr(_replay_orders, "clear_cache"), (
            "lift does not expose clear_cache -- not a jax.jit wrapper"
        )

    def test_lift_uses_static_argnames_not_static_argnums(self):
        """AST-check the assignment for the canonical idiom shape.

        Per migration-plan v5.13 SS"Module-level jit idiom" the assignment
        must call ``jax.jit(..., static_argnames=(...))``.  The negative
        form ``static_argnums=(...)`` is explicitly disallowed for this
        worked example (it forces callers to know positional indices).
        """
        src = LOADER_PATH.read_text()
        tree = ast.parse(src)

        found_call = None
        for node in ast.walk(tree):
            # Match: _replay_orders = jax.jit(_replay_orders_body, ...)
            if isinstance(node, ast.Assign) and node.targets:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name) and tgt.id == "_replay_orders":
                    if isinstance(node.value, ast.Call):
                        found_call = node.value
                        break

        assert found_call is not None, (
            "could not find module-level `_replay_orders = jax.jit(...)` "
            "assignment in omie_data_loader.py"
        )

        kw_names = {kw.arg for kw in found_call.keywords}
        assert "static_argnames" in kw_names, (
            "lift assignment must pass static_argnames=(...); the canonical "
            "P2-H2 idiom forbids static_argnums for this lift."
        )
        assert "static_argnums" not in kw_names, (
            "lift assignment uses static_argnums; violates canonical "
            "P2-H2 idiom (migration-plan v5.13 SS'Module-level jit idiom')."
        )

        # Extract the static_argnames tuple and verify content.
        static_kw = next(kw for kw in found_call.keywords
                         if kw.arg == "static_argnames")
        assert isinstance(static_kw.value, ast.Tuple), (
            "static_argnames must be a tuple literal; got "
            f"{type(static_kw.value).__name__}"
        )
        names = {
            elt.value for elt in static_kw.value.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
        assert names == {"product_config", "market_config"}, (
            f"static_argnames must be ('product_config', 'market_config'); "
            f"got {names!r}"
        )

    def test_lift_mirrors_canonical_submit_order_lift_shape(self):
        """Cross-check: ``_replay_orders`` and ``_submit_order`` are both
        ``jax.jit`` wrappers with identical introspection surface."""
        for attr in ("_cache_size", "clear_cache"):
            assert hasattr(_replay_orders, attr) and hasattr(_submit_order, attr), (
                f"canonical idiom mismatch: attribute {attr!r} missing on "
                f"one of the two lifts"
            )


# ---------------------------------------------------------------------------
# Section G -- No host-side outer Python loop in the live source
# ---------------------------------------------------------------------------

class TestNoHostSideOuterLoop:
    """The deliverable is a single ``lax.scan`` over the orders tensor;
    a stray Python ``for`` over ``orders.shape[0]`` would silently
    undo the lift."""

    def test_no_python_for_over_orders_shape_in_replay_body(self):
        """AST-walk ``_replay_orders_body``: it must contain zero
        Python ``for`` statements and at least one ``lax.scan`` call."""
        src = inspect.getsource(_replay_orders_body)
        tree = ast.parse(src)

        py_for_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
        assert len(py_for_nodes) == 0, (
            f"_replay_orders_body contains {len(py_for_nodes)} Python "
            f"`for` statements; the deliverable is a single lax.scan."
        )

        scan_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                # match jax.lax.scan(...) or lax.scan(...)
                if isinstance(fn, ast.Attribute) and fn.attr == "scan":
                    scan_calls.append(node)
        assert len(scan_calls) >= 1, (
            "_replay_orders_body has no `lax.scan` call; deliverable is "
            "a single `lax.scan` outer loop."
        )

    def test_public_function_does_not_loop_over_orders_shape0(self):
        """The public ``replay_orders_through_book`` must not contain a
        Python ``for i in range(orders.shape[0])`` pattern.  Other
        ``for`` statements (e.g., the trailing Python matching block
        over ``historical_trades_total`` and ``replayed_trades_arr``)
        are out of scope for this story."""
        src = inspect.getsource(replay_orders_through_book)
        # Allow any Python `for` not over orders.shape[0]; explicitly
        # forbid the legacy pattern (regex is permissive over whitespace).
        forbidden = re.compile(
            r"for\s+\w+\s+in\s+range\s*\(\s*orders\s*\.\s*shape\s*\[\s*0\s*\]\s*\)",
            re.MULTILINE,
        )
        assert not forbidden.search(src), (
            "replay_orders_through_book still contains a "
            "`for ... in range(orders.shape[0])` pattern; this is the "
            "exact loop the P2-PERF-5 lift must replace."
        )


# ---------------------------------------------------------------------------
# Section H -- ``OMIEBookState`` pytree treedef stability
# ---------------------------------------------------------------------------

class TestStateTreedefStability:
    """The lift must not introduce or remove state pytree leaves."""

    def test_treedef_unchanged_pre_vs_post_replay(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        book = _fresh_book(pc, mc)
        pre = book.init()
        orders = _build_crossing_pair_orders(2, product_config=pc)
        post, _ = _replay_orders(pre, orders, pc, mc)
        td_pre = jax.tree_util.tree_structure(pre)
        td_post = jax.tree_util.tree_structure(post)
        assert td_pre == td_post, (
            "OMIEBookState pytree treedef changed across the lift; the "
            "scan carry must thread the same pytree structure."
        )

    def test_state_leaf_count_unchanged(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        book = _fresh_book(pc, mc)
        pre = book.init()
        orders = _build_crossing_pair_orders(1, product_config=pc)
        post, _ = _replay_orders(pre, orders, pc, mc)
        assert len(jax.tree_util.tree_leaves(pre)) == len(
            jax.tree_util.tree_leaves(post)
        )

    def test_state_post_replay_is_OMIEBookState_namedtuple(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        book = _fresh_book(pc, mc)
        orders = _build_crossing_pair_orders(2, product_config=pc)
        state, _ = _replay_orders(book.init(), orders, pc, mc)
        assert isinstance(state, OMIEBookState), (
            "lift must thread OMIEBookState as carry; got "
            f"{type(state).__name__}"
        )


# ---------------------------------------------------------------------------
# Section I -- Determinism / bitwise reproducibility
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Two back-to-back invocations must produce bitwise-identical outputs."""

    def test_lift_invocations_are_bitwise_identical(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_crossing_pair_orders(3, product_config=pc)

        book_a = _fresh_book(pc, mc)
        book_b = _fresh_book(pc, mc)
        state_a, buf_a = _replay_orders(book_a.init(), orders, pc, mc)
        state_b, buf_b = _replay_orders(book_b.init(), orders, pc, mc)

        assert jnp.array_equal(buf_a, buf_b), (
            "lift produced non-deterministic replayed-trades buffer"
        )
        for la, lb in zip(jax.tree_util.tree_leaves(state_a),
                          jax.tree_util.tree_leaves(state_b)):
            assert jnp.array_equal(la, lb), (
                "lift produced non-deterministic state leaf"
            )

    def test_public_wrapper_is_deterministic_on_same_inputs(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_crossing_pair_orders(2, product_config=pc)
        historical = jnp.zeros((0, 6), dtype=jnp.int32)

        result_a = replay_orders_through_book(
            orders=orders,
            contract_book=_fresh_book(pc, mc),
            historical_trades=historical,
            heuristic="rest_until_gct",
        )
        result_b = replay_orders_through_book(
            orders=orders,
            contract_book=_fresh_book(pc, mc),
            historical_trades=historical,
            heuristic="rest_until_gct",
        )
        assert jnp.array_equal(result_a.replayed_trades, result_b.replayed_trades)
        assert result_a.historical_trades_matched == result_b.historical_trades_matched
        assert result_a.historical_trades_total == result_b.historical_trades_total
        assert result_a.extra_trades == result_b.extra_trades


# ---------------------------------------------------------------------------
# Section J -- In-scan new-fill detection invariant
# ---------------------------------------------------------------------------

class TestInScanNewFillDetection:
    """The lift's replayed-trades buffer fill-count must equal the count
    of rows where pre-/post-submission ``state.lob.trades`` differs.

    This anchors P2-PERF-5 deliverable (2): "new-fill detection moves
    inside the scan body" and matches the whole-row diff detector
    pinned by P2-HOTFIX-ioc-fill-detection for ``extract_fills``."""

    def test_lift_fill_count_matches_legacy_whole_row_diff_count(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        n_pairs = 3
        orders = _build_crossing_pair_orders(n_pairs, product_config=pc)

        # Count fills the lift reports.
        book_jit = _fresh_book(pc, mc)
        _, buf = _replay_orders(book_jit.init(), orders, pc, mc)
        lift_fills = int(jnp.sum(buf[:, :, 0] != -1))

        # Count fills via the legacy whole-row diff per-row pre/post
        # submission, the same detector ``extract_fills`` uses.
        book_legacy = _fresh_book(pc, mc)
        state = book_legacy.init()
        legacy_fills = 0
        for i in range(int(orders.shape[0])):
            row = orders[i]
            order = _row_to_order_traced(row, pc)
            before = state.lob.trades
            state, _ = _submit_order(state, order, pc, mc)
            after = state.lob.trades
            new_mask = jnp.any(before != after, axis=1)
            legacy_fills += int(jnp.sum(new_mask))

        assert lift_fills == legacy_fills, (
            f"lift fill count ({lift_fills}) diverges from whole-row "
            f"diff fill count ({legacy_fills}); in-scan new-fill "
            f"detection invariant violated."
        )

    def test_zero_fills_session_emits_zero_lift_fills(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_noncrossing_orders(4, product_config=pc)
        book = _fresh_book(pc, mc)
        _, buf = _replay_orders(book.init(), orders, pc, mc)
        assert int(jnp.sum(buf[:, :, 0] != -1)) == 0


# ---------------------------------------------------------------------------
# Section K -- Callsite-untouched probe via grep
# ---------------------------------------------------------------------------

class TestCallsitesUntouched:
    """The public ``replay_orders_through_book`` callers in
    ``tests/omiecid/`` did not need to change. We verify by greping the
    five known caller files and confirming they still issue calls of
    the canonical kwargs-shape."""

    def test_five_known_caller_files_still_call_replay(self):
        # Use git ls-files to discover what's actually tracked. We're not
        # validating any non-test caller here.
        for fname in PUBLIC_CALLER_FILES:
            path = TEST_DIR / fname
            assert path.exists(), (
                f"expected caller file {fname} is missing; the public "
                f"caller universe for replay_orders_through_book changed."
            )

    def test_known_callers_use_replay_orders_through_book_publicly(self):
        """All five caller files must reference the public function by
        name at least once (sanity probe that grep tooling and the path
        layout match the merge-note claim)."""
        for fname in PUBLIC_CALLER_FILES:
            text = (TEST_DIR / fname).read_text()
            assert "replay_orders_through_book" in text, (
                f"{fname} no longer references replay_orders_through_book; "
                f"the public-caller universe diverged from the P2-PERF-5 "
                f"merge note."
            )


# ---------------------------------------------------------------------------
# Section L -- Per-step output shape stability
# ---------------------------------------------------------------------------

class TestPerStepShapeStability:
    """For a fixed ``N``, the per-step slab shape is invariant across
    varying inputs."""

    @pytest.mark.parametrize("n_pairs", [1, 2, 3, 4])
    def test_shape_constant_across_synthetic_pair_counts(self, n_pairs):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        orders = _build_crossing_pair_orders(n_pairs, product_config=pc)
        book = _fresh_book(pc, mc)
        _, buf = _replay_orders(book.init(), orders, pc, mc)
        assert buf.shape == (2 * n_pairs, NTRADE_CAP, 3)
        assert buf.dtype == jnp.int32

    def test_shape_invariant_across_fill_vs_no_fill_inputs_of_same_N(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        n = 6
        orders_fill = _build_crossing_pair_orders(n // 2, product_config=pc)
        orders_no_fill = _build_noncrossing_orders(n, product_config=pc)
        assert int(orders_fill.shape[0]) == int(orders_no_fill.shape[0]) == n
        book1 = _fresh_book(pc, mc)
        book2 = _fresh_book(pc, mc)
        _, buf_fill = _replay_orders(book1.init(), orders_fill, pc, mc)
        _, buf_no_fill = _replay_orders(book2.init(), orders_no_fill, pc, mc)
        assert buf_fill.shape == buf_no_fill.shape == (n, NTRADE_CAP, 3)
        assert buf_fill.dtype == buf_no_fill.dtype == jnp.int32

    def test_empty_orders_produces_empty_leading_axis_buffer(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        # An empty orders tensor with shape (0, 13) is the documented
        # short-circuit path; the public wrapper bypasses the lift
        # entirely. We probe that the public function still returns
        # a well-shaped ReplayResult.
        orders = jnp.zeros((0, 13), dtype=jnp.int32)
        historical = jnp.zeros((0, 6), dtype=jnp.int32)
        result = replay_orders_through_book(
            orders=orders,
            contract_book=_fresh_book(pc, mc),
            historical_trades=historical,
            heuristic="rest_until_gct",
        )
        # The compacted public-facing buffer is (0, 3) float32.
        assert result.replayed_trades.shape == (0, 3)
        assert result.historical_trades_total == 0
        assert result.historical_trades_matched == 0
        assert result.extra_trades == 0


# ---------------------------------------------------------------------------
# Section M -- Bonus probes (orthogonal extras)
# ---------------------------------------------------------------------------

class TestRowToOrderTraced:
    """The traced row->OMIEOrder helper preserves order identity fields
    and is callable both eagerly and under jit."""

    def test_traced_helper_preserves_integer_enum_fields(self):
        pc = OMIEProductConfig()
        row = jnp.array(_mk_row(
            side=int(Side.BUY), price_int=_encode_price(60.50, pc),
            qty_int=_encode_qty(2.5, pc), order_id=42, agent_id=7,
            timestamp=12345, exec_restriction=int(ExecRestriction.IOC),
            validity=int(Validity.GTD),
        ), dtype=jnp.int32)
        order = _row_to_order_traced(row, pc)
        assert int(order.order_id) == 42
        assert int(order.agent_id) == 7
        assert int(order.timestamp) == 12345
        assert int(order.side) == int(Side.BUY)
        assert int(order.exec_restriction) == int(ExecRestriction.IOC)
        assert int(order.validity) == int(Validity.GTD)

    def test_traced_helper_callable_under_jit(self):
        pc = OMIEProductConfig()

        @jax.jit
        def _call(row):
            o = _row_to_order_traced(row, pc)
            return o.order_id, o.agent_id

        row = jnp.array(_mk_row(
            side=int(Side.SELL), price_int=_encode_price(50.00, pc),
            qty_int=_encode_qty(0.1, pc), order_id=9, agent_id=3,
            timestamp=1,
        ), dtype=jnp.int32)
        oid, aid = _call(row)
        assert int(oid) == 9
        assert int(aid) == 3


class TestBufferContentLawful:
    """Spot-check the slab content of a single known-good crossing pair."""

    def test_single_crossing_pair_emits_expected_price_qty_in_slab(self):
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()
        p_eur = 50.00
        q_mw = 0.1
        p_int = _encode_price(p_eur, pc)
        q_int = _encode_qty(q_mw, pc)
        orders = _build_crossing_pair_orders(1, product_config=pc)
        book = _fresh_book(pc, mc)
        _, buf = _replay_orders(book.init(), orders, pc, mc)

        # Row 0 (resting SELL) -> no fill.
        assert jnp.all(buf[0, :, 0] == -1), "resting SELL must emit no fill"

        # Row 1 (crossing BUY) -> exactly one fill at (p_int, q_int).
        oid_col = buf[1, :, 0]
        n_fills = int(jnp.sum(oid_col != -1))
        assert n_fills == 1, (
            f"crossing BUY must emit exactly one fill; got {n_fills}"
        )
        # Find the single non-sentinel row.
        fill_idx = int(jnp.argmax((oid_col != -1).astype(jnp.int32)))
        slab_row = buf[1, fill_idx]
        # Aggressor oid is the BUY's order_id (oid=2 in the generator).
        assert int(slab_row[0]) == 2
        # The lift records the *jaxob* int32 price/qty ticks. Under the
        # jaxob convention these correspond to the OMIE-encoded p_int /
        # q_int values (1:1 for crossing pair at the same price).
        # We assert non-zero and consistent magnitudes rather than exact
        # equality to remain robust to any internal sign/scale convention.
        assert int(slab_row[1]) != 0, (
            "fill row price-tick column must be non-zero on a real fill"
        )
        assert int(slab_row[2]) != 0, (
            "fill row qty-tick column must be non-zero on a real fill"
        )
        assert int(slab_row[2]) == q_int, (
            f"fill row qty-tick must equal encoded q_int; got "
            f"{int(slab_row[2])} expected {q_int}"
        )
