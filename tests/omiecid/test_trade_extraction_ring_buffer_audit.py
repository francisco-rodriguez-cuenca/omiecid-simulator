"""Audit tests for P2-HOTFIX-ioc-fill-detection (ring-buffer-immune detector).

Independent audit pass. The auditor has NOT read the original locked test
file (`tests/omiecid/test_trade_extraction_ring_buffer.py`) and derives
coverage independently from the story's acceptance boundary and from the
public interfaces of:

  - ``omiecid_simulator.omiecid.trade_extraction.extract_fills`` / ``RawFills``
  - ``omiecid_simulator.omiecid.contract_book.ContractBook.submit_order``
  - ``omiecid_simulator.jaxomiecid.omie_data_loader.replay_orders_through_book``

Focus areas (per orchestrator brief):

  1. Signature / NamedTuple-ish stability of ``RawFills`` across buffer
     saturation states.
  2. Detector correctness boundaries (whole-row diff, zero-change, JAX
     jittability of the ``RawFills`` payload).
  3. Absence of the pre-hotfix sentinel-transition mask anywhere under
     ``omiecid_simulator/omiecid`` or ``omiecid_simulator/jaxomiecid`` production code.
  4. Market-spec §4.5 invariant (IOC cancels residual, never rests) —
     property-style sweep against a saturated trades buffer.
  5. Consumer parity: ``replay_orders_through_book`` aggregates fills
     consistent with ``contract_book.submit_order`` on the same input
     sequence.
  6. Dtype discipline: aggregate totals stay integer-valued.
  7. Bounds: on a saturated buffer with extreme int32 trade values, totals
     do not silently overflow int32.

All tests target the post-fix code and MUST pass.
"""

from __future__ import annotations

import ast
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order
from omiecid_simulator.omiecid.trade_extraction import (
    NO_FILLS,
    RawFills,
    extract_fills,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic trades buffers
# ---------------------------------------------------------------------------

P_COL = cst.TradesFeat.P.value
Q_COL = cst.TradesFeat.Q.value
OID_COL = cst.TradesFeat.PASS_OID.value
NTRADE_CAP = cst.NTRADE_CAP
N_TRADE_COLS = 8  # TRADE_FEAT


def _empty_trades(n: int = NTRADE_CAP) -> jnp.ndarray:
    return (jnp.ones((n, N_TRADE_COLS), dtype=jnp.int32) * -1)


def _saturated_trades(base_oid: int = 1000, price: int = 100,
                      qty: int = 5, n: int = NTRADE_CAP) -> jnp.ndarray:
    """Every slot occupied by a distinct valid fill (no -1 sentinels)."""
    arr = np.zeros((n, N_TRADE_COLS), dtype=np.int32)
    for i in range(n):
        arr[i, P_COL] = price + i  # distinct prices so rows differ
        arr[i, Q_COL] = qty
        arr[i, OID_COL] = base_oid + i
        arr[i, 3] = base_oid + i + 10000  # AGRS_OID
        arr[i, 4] = i  # SEC
    return jnp.asarray(arr)


# ---------------------------------------------------------------------------
# 1. Signature / RawFills stability
# ---------------------------------------------------------------------------

class TestRawFillsShape:
    def test_no_fills_sentinel_fields_and_values(self):
        assert NO_FILLS.num_fills == 0
        assert NO_FILLS.total_qty_int == 0
        assert NO_FILLS.weighted_price_int == 0

    def test_raw_fills_has_three_public_int_fields(self):
        # Post-P2-L4: RawFills fields are 0-d jnp.ndarray scalars with the
        # dtype split (int64, int64, int32). This is a Category B adapter
        # update — value semantics are preserved (qty/wpx/n still equal),
        # only the leaf representation moved from Python int to JAX scalar.
        before = _empty_trades()
        after_np = np.array(before)
        after_np[0, P_COL] = 14
        after_np[0, Q_COL] = 3
        after_np[0, OID_COL] = 1
        after = jnp.asarray(after_np)
        f = extract_fills(before, after)
        # value semantics
        assert int(f.total_qty_int) == 3
        assert int(f.weighted_price_int) == 42
        assert int(f.num_fills) == 1
        # representation: 0-d JAX scalars with the locked dtype split
        for field, expected_dtype in (
            (f.total_qty_int, jnp.int64),
            (f.weighted_price_int, jnp.int64),
            (f.num_fills, jnp.int32),
        ):
            assert isinstance(field, jnp.ndarray)
            assert field.dtype == expected_dtype
            assert field.shape == ()

    @pytest.mark.parametrize("buffer_builder,label", [
        (lambda: _empty_trades(), "empty"),
        (lambda: _saturated_trades(), "saturated"),
    ])
    def test_same_buffer_before_and_after_yields_zero_fills(
        self, buffer_builder, label
    ):
        buf = buffer_builder()
        fills = extract_fills(buf, buf)
        assert fills.num_fills == 0, label
        assert fills.total_qty_int == 0, label
        assert fills.weighted_price_int == 0, label


# ---------------------------------------------------------------------------
# 2. Detector correctness boundaries
# ---------------------------------------------------------------------------

class TestDetectorCorrectness:
    def test_identical_empty_buffers(self):
        b = _empty_trades()
        assert extract_fills(b, b).num_fills == 0

    def test_single_row_flip_from_sentinel_to_valid(self):
        before = _empty_trades()
        after_np = np.array(before)
        after_np[0, P_COL] = 500
        after_np[0, Q_COL] = -3  # jaxob signed: abs() must be applied
        after_np[0, OID_COL] = 42
        after = jnp.asarray(after_np)

        fills = extract_fills(before, after)
        assert fills.num_fills == 1
        assert fills.total_qty_int == 3
        assert fills.weighted_price_int == 500 * 3

    def test_ring_buffer_overwrite_detected_via_row_diff(self):
        """Saturated buffer: a NEW fill overwrites an existing slot.

        The pre-hotfix `(-1 -> valid)` sentinel mask would see zero
        changes here because all OIDs stay != -1. The post-hotfix
        whole-row diff must detect the overwrite.
        """
        before = _saturated_trades(base_oid=1000, price=100, qty=5)
        after_np = np.array(before)
        # overwrite slot 17 with a brand-new fill
        after_np[17, P_COL] = 777
        after_np[17, Q_COL] = 9
        after_np[17, OID_COL] = 99999
        after = jnp.asarray(after_np)

        fills = extract_fills(before, after)
        assert fills.num_fills == 1
        assert fills.total_qty_int == 9
        assert fills.weighted_price_int == 777 * 9

    def test_multi_row_overwrite_on_saturated_buffer(self):
        before = _saturated_trades()
        after_np = np.array(before)
        # overwrite three non-contiguous slots
        spec = [(3, 200, 4), (50, 300, 7), (99, 400, 2)]
        for idx, price, qty in spec:
            after_np[idx, P_COL] = price
            after_np[idx, Q_COL] = qty
            after_np[idx, OID_COL] = 50000 + idx
        after = jnp.asarray(after_np)

        fills = extract_fills(before, after)
        assert fills.num_fills == 3
        assert fills.total_qty_int == 4 + 7 + 2
        assert fills.weighted_price_int == 200 * 4 + 300 * 7 + 400 * 2

    def test_signed_quantity_uses_absolute_value(self):
        before = _empty_trades()
        after_np = np.array(before)
        # incoming-buy match leaves a negative Q in jaxob
        after_np[0] = np.array([100, -8, 7, 17, 0, 0, 0, 0])
        after_np[1] = np.array([110, 3, 8, 18, 0, 0, 0, 0])
        after = jnp.asarray(after_np)

        fills = extract_fills(before, after)
        assert fills.num_fills == 2
        assert fills.total_qty_int == 8 + 3
        assert fills.weighted_price_int == 100 * 8 + 110 * 3

    def test_final_state_count_not_internal_match_count(self):
        """Documents hotfix limitation as a positive assertion.

        If a single `submit_order` wrote row X and then overwrote row X
        again in the same dispatch, the detector only sees the final
        row's change and reports num_fills consistent with visible-row-
        delta, not with how many matches jaxob internally computed.
        """
        before = _empty_trades()
        after_np = np.array(before)
        # one net visible row change
        after_np[5, P_COL] = 123
        after_np[5, Q_COL] = 1
        after_np[5, OID_COL] = 777
        after = jnp.asarray(after_np)

        fills = extract_fills(before, after)
        assert fills.num_fills == 1  # NOT 2, even if jaxob matched twice

    @pytest.mark.jit
    def test_raw_fills_is_jittable_payload(self):
        """RawFills must be constructible from pure-Python ints inside
        a callable that itself can be JIT-compiled around extract_fills.

        extract_fills contains Python-int accumulation (not traceable),
        so we verify jittability by jitting a lambda that consumes only
        the integer fields of RawFills — ensuring those fields survive
        the JAX boundary without dtype/structure surprises."""
        before = _empty_trades()
        after_np = np.array(before)
        after_np[0, P_COL] = 50
        after_np[0, Q_COL] = 2
        after_np[0, OID_COL] = 11
        after = jnp.asarray(after_np)

        fills = extract_fills(before, after)

        @jax.jit
        def _consume(qty, wpx, n):
            return jnp.int32(qty) * 1 + jnp.int32(wpx) + jnp.int32(n)

        out = _consume(fills.total_qty_int,
                       fills.weighted_price_int,
                       fills.num_fills)
        # expected: 2 + 100 + 1 = 103
        assert int(out) == 2 + 100 + 1


# ---------------------------------------------------------------------------
# 3. No residual sentinel-transition logic in production code
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _production_py_files():
    for sub in ("omiecid_simulator/omiecid", "omiecid_simulator/jaxomiecid"):
        d = _REPO_ROOT / sub
        for p in d.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


def _strip_comments_and_docstrings(src: str) -> str:
    """Return executable source only — comments and string-literal
    expression statements (module/class/function docstrings or standalone
    string literals used as explanatory notes) are removed.

    This keeps commentary about the old sentinel-transition mask (which
    must survive as explanation) from triggering the regression check.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    # Collect docstring-style string-constant lines to blank out
    blank_lines: set[int] = set()

    class _DocVisitor(ast.NodeVisitor):
        def _maybe_strip_docstring(self, node):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant
            ) and isinstance(body[0].value.value, str):
                first = body[0]
                for ln in range(first.lineno, first.end_lineno + 1):
                    blank_lines.add(ln)

        def visit_Module(self, node):
            self._maybe_strip_docstring(node)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            self._maybe_strip_docstring(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self._maybe_strip_docstring(node)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            self._maybe_strip_docstring(node)
            self.generic_visit(node)

    _DocVisitor().visit(tree)

    out_lines = []
    for i, line in enumerate(src.splitlines(), start=1):
        if i in blank_lines:
            out_lines.append("")
            continue
        # strip inline comments: find first '#' not inside a string
        # (approximate but good enough given project style)
        stripped = line
        in_s = None
        buf = []
        j = 0
        while j < len(stripped):
            ch = stripped[j]
            if in_s:
                buf.append(ch)
                if ch == "\\" and j + 1 < len(stripped):
                    buf.append(stripped[j + 1])
                    j += 2
                    continue
                if ch == in_s:
                    in_s = None
            else:
                if ch in ("'", '"'):
                    in_s = ch
                    buf.append(ch)
                elif ch == "#":
                    break
                else:
                    buf.append(ch)
            j += 1
        out_lines.append("".join(buf))
    return "\n".join(out_lines)


class TestNoResidualSentinelLogic:
    def test_no_sentinel_transition_pattern_in_executable_code(self):
        forbidden_fragments = [
            # any textual shape of the old mask
            "(before_oids == -1)",
            "before_oids == -1",
            "after_oids != -1",
        ]
        offenders: list[str] = []
        for path in _production_py_files():
            src = path.read_text(encoding="utf-8")
            code = _strip_comments_and_docstrings(src)
            for frag in forbidden_fragments:
                if frag in code:
                    offenders.append(f"{path}: contains {frag!r}")
        assert not offenders, (
            "Residual sentinel-transition logic found in production code: "
            + "; ".join(offenders)
        )

    def test_no_arrayed_minus_one_transition_mask_in_executable_code(self):
        """More structural check via AST: search for a BinOp of form
        `(X == -1) & (Y != -1)` anywhere in production code."""
        offenders: list[str] = []
        for path in _production_py_files():
            src = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.BinOp):
                    continue
                if not isinstance(node.op, ast.BitAnd):
                    continue
                left, right = node.left, node.right
                if (_is_eq_neg_one(left) and _is_neq_neg_one(right)) or (
                    _is_neq_neg_one(left) and _is_eq_neg_one(right)
                ):
                    offenders.append(f"{path}:{getattr(node, 'lineno', '?')}")
        assert not offenders, (
            "AST-level (x == -1) & (y != -1) pattern still present: "
            + "; ".join(offenders)
        )


def _is_eq_neg_one(node) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return False
    return _is_neg_one_const(node.comparators[0])


def _is_neq_neg_one(node) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.NotEq):
        return False
    return _is_neg_one_const(node.comparators[0])


def _is_neg_one_const(node) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and node.operand.value == 1
    if isinstance(node, ast.Constant):
        return node.value == -1
    return False


# ---------------------------------------------------------------------------
# 4. Market-spec §4.5 IOC invariant (property-style sweep)
# ---------------------------------------------------------------------------

def _book_with_saturated_trades(base_price_eur=50.0):
    """Build a ContractBook whose trades ring-buffer is pre-saturated by
    replaying NTRADE_CAP+ matches before the IOC-under-test arrives.

    This is the real bug trigger: once all slots are full, the old
    sentinel detector stops firing on new fills and IOCs appear to
    "not trade" which in turn made them rest (violating §4.5).
    """
    contract = Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )
    pc = OMIEProductConfig()
    mc = OMIEMarketConfig()
    book = ContractBook(contract, pc, mc)
    state = book.init()

    next_oid = 1

    def _rest_sell(price, qty, ts):
        nonlocal next_oid
        o = make_order(
            contract_id=37, side=Side.SELL, price=price, quantity=qty,
            order_id=next_oid, agent_id=next_oid + 100, timestamp=ts,
        )
        next_oid += 1
        return book.submit_order(state, o)

    # Saturate: drop N+5 matched lots by alternating resting sell and
    # aggressive buy at the same price.
    n_saturate = NTRADE_CAP + 5
    ts = 1000
    for i in range(n_saturate):
        # rest sell
        o_sell = make_order(
            contract_id=37, side=Side.SELL, price=base_price_eur,
            quantity=0.1, order_id=next_oid, agent_id=500,
            timestamp=ts,
        )
        next_oid += 1
        ts += 1
        state, _ = book.submit_order(state, o_sell)
        # aggressive buy (NON) crosses it
        o_buy = make_order(
            contract_id=37, side=Side.BUY, price=base_price_eur,
            quantity=0.1, order_id=next_oid, agent_id=600,
            timestamp=ts,
        )
        next_oid += 1
        ts += 1
        state, _ = book.submit_order(state, o_buy)

    return book, state, next_oid, ts


# P2-PERF-3 Deliverable (3): hoist the saturating replay (~210 submit_order
# calls) into a module-scoped fixture so the four `TestIOCRingBufferInvariant`
# methods share a single saturation cost. The helper logic is unchanged;
# only fixture wiring moves. Safety: OMIEBookState is an immutable NamedTuple
# and ContractBook is documented stateless, so each test's `submit_order`
# returns a new local state and never mutates the fixture in place.
@pytest.fixture(scope="module")
def saturated_book_state_oid_ts():
    """Module-scoped wrapper around `_book_with_saturated_trades(50.0)`.

    Returns the same `(book, state, next_oid, ts)` tuple shape that the
    underlying helper produces. Tests treat `next_oid` and `ts` as starting
    integer counters and increment local copies — they never mutate the
    fixture-owned `book` or `state`.
    """
    return _book_with_saturated_trades(base_price_eur=50.0)


@pytest.mark.slow
class TestIOCRingBufferInvariant:
    """slow: saturates the NTRADE_CAP=100 trades buffer via ~210 submits."""

    def test_ioc_against_saturated_buffer_matches_and_does_not_rest(self, saturated_book_state_oid_ts):
        book, state, next_oid, ts = saturated_book_state_oid_ts

        # Rest a fresh ask at a crossing price for the IOC to consume.
        rest_sell = make_order(
            contract_id=37, side=Side.SELL, price=48.0, quantity=0.3,
            order_id=next_oid, agent_id=700, timestamp=ts,
        )
        next_oid += 1
        ts += 1
        state, _ = book.submit_order(state, rest_sell)

        # IOC buy, willing to pay up to 55.0; should match at 48.0.
        ioc_oid = next_oid
        ioc = make_order(
            contract_id=37, side=Side.BUY, price=55.0, quantity=0.3,
            order_id=ioc_oid, agent_id=800, timestamp=ts + 1,
            exec_restriction=ExecRestriction.IOC,
        )
        state, result = book.submit_order(state, ioc)

        assert result.accepted == True
        assert result.traded == True, (
            "§4.5 violated: IOC did not trade against saturated "
            "trades buffer (ring-buffer fill detection regression)."
        )
        # The IOC must not rest — no order with ioc_oid remains on the
        # BUY side.
        bids, _ = book.get_anonymous_lob(state)
        agent_orders = book.get_agent_orders(state, 800)
        assert all(o.order_id != ioc_oid for o in agent_orders), (
            "§4.5 violated: IOC rested in the book instead of being "
            "cancelled after partial/no match."
        )

    def test_ioc_with_no_liquidity_on_saturated_buffer_does_not_rest(self, saturated_book_state_oid_ts):
        book, state, next_oid, ts = saturated_book_state_oid_ts
        # No fresh liquidity at a crossing price; IOC BUY @ 10 cannot
        # match anything (existing resting book is already consumed).
        ioc_oid = next_oid
        ioc = make_order(
            contract_id=37, side=Side.BUY, price=10.0, quantity=0.2,
            order_id=ioc_oid, agent_id=801, timestamp=ts + 1,
            exec_restriction=ExecRestriction.IOC,
        )
        state, result = book.submit_order(state, ioc)

        # Either trades=False or accepted=False, but in neither case
        # should the IOC rest in the book (§4.5).
        agent_orders = book.get_agent_orders(state, 801)
        assert all(o.order_id != ioc_oid for o in agent_orders), (
            "§4.5 violated: unmatched IOC rests in book on saturated "
            "buffer."
        )

    @pytest.mark.parametrize("qty_mw,limit_eur", [
        (0.1, 55.0),
        (0.3, 60.0),
        (0.5, 100.0),
    ])
    def test_ioc_invariant_parametric_sweep(self, qty_mw, limit_eur, saturated_book_state_oid_ts):
        """Sweep a few (qty, limit) combinations against the saturated
        buffer. In every case the IOC must either (a) trade, or (b)
        be fully cancelled — never rest."""
        book, state, next_oid, ts = saturated_book_state_oid_ts
        # fresh liquidity matching `qty_mw` at a crossing price
        rest_sell = make_order(
            contract_id=37, side=Side.SELL, price=48.0, quantity=qty_mw,
            order_id=next_oid, agent_id=900, timestamp=ts,
        )
        next_oid += 1
        ts += 1
        state, _ = book.submit_order(state, rest_sell)

        ioc_oid = next_oid
        ioc = make_order(
            contract_id=37, side=Side.BUY, price=limit_eur,
            quantity=qty_mw, order_id=ioc_oid,
            agent_id=901, timestamp=ts + 1,
            exec_restriction=ExecRestriction.IOC,
        )
        state, result = book.submit_order(state, ioc)

        # IOC must not rest regardless of trade outcome.
        assert all(
            o.order_id != ioc_oid for o in book.get_agent_orders(state, 901)
        ), "IOC rested — §4.5 violation"
        # On a freshly added crossing ask of same qty, it SHOULD trade.
        assert result.traded == True


# ---------------------------------------------------------------------------
# 5. Consumer parity: loader replay vs submit_order aggregate
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestConsumerParity:
    """slow: runs both the submit_order path and the
    replay_orders_through_book path across a saturating sequence."""

    def test_replay_and_submit_aggregate_agree_on_saturated_sequence(self):
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            replay_orders_through_book,
        )

        contract = Contract(
            contract_id=37,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.OPEN,
        )
        pc = OMIEProductConfig()
        mc = OMIEMarketConfig()

        # Build a paired sequence that saturates the trades buffer
        # through alternate resting-sell / aggressive-buy fills.
        n_pairs = NTRADE_CAP // 2 + 10   # enough to saturate and wrap
        price = 50.0
        qty = 0.1
        rows = []  # list of 13-column int32 rows

        from omiecid_simulator.omiecid.encoding import encode_price, encode_qty

        p_int = encode_price(price, pc)
        q_int = encode_qty(qty, pc)

        oid = 1
        ts = 1000
        for _ in range(n_pairs):
            # rest sell
            rows.append([37, int(Side.SELL), p_int, q_int, oid, 500, ts,
                         int(ExecRestriction.NON), 0, 0, 1, 0, 0])
            oid += 1
            ts += 1
            # aggressive buy NON
            rows.append([37, int(Side.BUY), p_int, q_int, oid, 600, ts,
                         int(ExecRestriction.NON), 0, 0, 1, 0, 0])
            oid += 1
            ts += 1

        orders = jnp.asarray(rows, dtype=jnp.int32)

        # ---- loader replay path ----
        book_a = ContractBook(contract, pc, mc)
        result = replay_orders_through_book(
            orders, book_a, jnp.zeros((0, 6), dtype=jnp.int32),
            heuristic="rest_until_gct",
        )
        replayed_qty_total = float(jnp.sum(result.replayed_trades[:, 1]))

        # ---- direct submit_order path ----
        from omiecid_simulator.jaxomiecid.omie_data_loader import row_to_order
        book_b = ContractBook(contract, pc, mc)
        state_b = book_b.init()
        direct_qty_total = 0.0
        for i in range(orders.shape[0]):
            o = row_to_order(orders[i], pc)
            state_b, r = book_b.submit_order(state_b, o)
            if r.traded:
                # P2-H2: ``r.trade_qty`` is a 0-d ``jnp.float32`` scalar.
                # ``direct_qty_total += r.trade_qty`` would silently
                # promote the accumulator into float32 space and lose
                # precision across the N-iteration sum (~3e-6 drift
                # over 60 fills of 0.1 MW). Coerce to Python float so
                # accumulation stays at float64 precision.
                direct_qty_total += float(r.trade_qty)

        # Both paths should have produced the full volume that the
        # paired sequence implies: n_pairs fills, each `qty` MW.
        expected_volume = n_pairs * qty
        assert direct_qty_total == pytest.approx(expected_volume, abs=1e-4), (
            f"submit_order aggregate: {direct_qty_total} != {expected_volume}"
        )
        # The replay arm sums the float32 ``replayed_trades[:, 1]`` column via
        # ``jnp.sum`` (a float32 reduction whose order is device-specific:
        # CPU and GPU accumulate in different orders), so the tolerance covers
        # float32 ULP for ~60 terms at this scale (worst-case N*eps*sum =
        # 60 * 1.19e-7 * 6.0 ~= 4.3e-5; observed CPU drift ~1.4e-6). The
        # per-fill values are byte-identical across both arms (see n_fills
        # parity); only the reduction order differs.
        assert replayed_qty_total == pytest.approx(expected_volume, abs=1e-4), (
            f"replay aggregate: {replayed_qty_total} != {expected_volume}"
        )
        assert replayed_qty_total == pytest.approx(direct_qty_total, abs=1e-4)


# ---------------------------------------------------------------------------
# 6. Dtype discipline
# ---------------------------------------------------------------------------

class TestDtypeDiscipline:
    def test_aggregates_are_python_int(self):
        # Post-P2-L4 Category B adapter: aggregates are now 0-d JAX scalars
        # with the dtype split (int64, int64, int32). Integer-exact
        # value semantics preserved; representation is JAX, not Python int.
        before = _empty_trades()
        after_np = np.array(before)
        after_np[0] = np.array([100, -4, 1, 11, 0, 0, 0, 0])
        after_np[1] = np.array([200, 3, 2, 12, 0, 0, 0, 0])
        after = jnp.asarray(after_np)

        fills = extract_fills(before, after)
        for field, expected_dtype in (
            (fills.total_qty_int, jnp.int64),
            (fills.weighted_price_int, jnp.int64),
            (fills.num_fills, jnp.int32),
        ):
            assert isinstance(field, jnp.ndarray)
            assert field.dtype == expected_dtype
            assert field.shape == ()

    def test_no_float_promotion_on_moderate_values(self):
        before = _empty_trades()
        after_np = np.array(before)
        after_np[0] = np.array([12345, 6, 1, 11, 0, 0, 0, 0])
        after = jnp.asarray(after_np)

        fills = extract_fills(before, after)
        # Integer exact arithmetic — no float rounding.
        assert int(fills.weighted_price_int) == 12345 * 6
        # Post-P2-L4 Category B adapter: int leaf is now a JAX int64 scalar.
        assert isinstance(fills.weighted_price_int, jnp.ndarray)
        assert fills.weighted_price_int.dtype == jnp.int64


# ---------------------------------------------------------------------------
# 7. Overflow / bounds on saturated buffer with extreme int32 values
# ---------------------------------------------------------------------------

class TestOverflowBounds:
    def test_extreme_price_qty_product_exceeds_int32_without_overflow(self):
        """price * qty at near-int32 magnitudes must not wrap.

        Individual row aggregate (weighted_price_int) needs ~ int64 to
        hold a 2e6-magnitude price times a 1e4-magnitude qty, and the
        sum across many rows can easily exceed 2^31."""
        before = _empty_trades()
        after_np = np.array(before)
        price = 1_999_800
        qty = 9_990
        # populate 50 rows — total = 50 * 1999800 * 9990
        n = 50
        for i in range(n):
            after_np[i] = np.array(
                [price, qty, 1000 + i, 2000 + i, 0, 0, 0, 0]
            )
        after = jnp.asarray(after_np)

        fills = extract_fills(before, after)
        expected_total_qty = n * qty
        expected_weighted = n * price * qty
        # Post-P2-L4 Category B adapter: aggregates are int64 JAX scalars;
        # cast through Python int for comparisons against large Python
        # literals (the test process runs with x64-off, so direct
        # array-vs-Python-int compare on values exceeding int32 range
        # would trip JAX's literal-canonicalisation overflow guard).
        assert int(fills.num_fills) == n
        assert int(fills.total_qty_int) == expected_total_qty
        assert int(fills.weighted_price_int) == expected_weighted
        # Sanity: result exceeds int32 range.
        assert expected_weighted > 2**31

    def test_saturated_buffer_full_replace_counts_all_rows(self):
        """Sanity: every slot of a saturated buffer changing yields
        num_fills == NTRADE_CAP with correct aggregates (no silent
        truncation via int32 intermediates)."""
        before = _saturated_trades(base_oid=1, price=100, qty=5)
        # Build an 'after' where every row differs from 'before'
        after_np = np.array(before)
        after_np[:, P_COL] = after_np[:, P_COL] + 1000  # bump all prices
        after = jnp.asarray(after_np)

        fills = extract_fills(before, after)
        assert fills.num_fills == NTRADE_CAP
        assert fills.total_qty_int == NTRADE_CAP * 5
        # Each row's contribution: (old_price + 1000) * 5; summed.
        expected = sum(int((100 + i + 1000)) * 5 for i in range(NTRADE_CAP))
        assert fills.weighted_price_int == expected
