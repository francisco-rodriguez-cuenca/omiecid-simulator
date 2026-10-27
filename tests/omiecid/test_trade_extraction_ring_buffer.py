"""Locked regression tests for `P2-HOTFIX-ioc-fill-detection`.

Ring-buffer-immune fill detection for `omiecid_simulator.omiecid.trade_extraction`
and the two consumer sites that mirror the same sentinel-transition pattern
(`contract_book.ContractBook.submit_order` IOC loop, and the
`omie_data_loader.replay_orders_through_book` inline fill-extraction).

Root cause (from the diagnostic workbook and agile-plan story card): once the
100-slot jaxob `LobState.trades` ring buffer saturates, new fills overwrite
existing slots. The pre-hotfix mask
``(before_oids == -1) & (after_oids != -1)`` never fires under saturation, so
`extract_fills` reports `num_fills == 0` even though jaxob DID match. This
drops IOC fills (spec §4.5: IOC cancels residual, never rests — hard rule)
and silently fails §6.2 step-3 maker pricing on saturated books.

Fix site (developer's territory — tests do not import these internals):
  * `omiecid_simulator/omiecid/trade_extraction.py`            — `extract_fills`
  * `omiecid_simulator/jaxomiecid/omie_data_loader.py:857-863` — inline copy

Spec references:
  * `docs/omiecid/market-spec.md` §4.5 (IOC must not rest), §6.2 (maker pricing),
    §5.2 (global timestamp queue), §6.1/§11.1 (price-time priority).
  * `docs/omiecid/agile-plan.md` §"Phase 2 HOTFIX" → story P2-HOTFIX-ioc-fill-detection.
  * `docs/omiecid/migration-plan.md` (interfaces).

Test categories (all locked as black-box contracts):
  1. Saturated-buffer regression (extract_fills): fills detected after sat.
  2. Empty-buffer parity with the legacy sentinel-transition detector.
  3. Partial-occupancy (50/100) parity.
  4. No-op negative: non-crossing order produces zero fills under saturation.
  5. Overflow guard on `weighted_price_int`.
  6. IOC #1022 end-to-end on 2026-01-12 contract 81.
  7. Consumer branch coverage: IOC loop takes the fills-detected branch
     under saturation, not the cancel-tail.
  8. Loader replay-harness parity across the saturation boundary.

Expected until the hotfix lands: tests 1, 4, 5, 6, 7, 8 FAIL with the broken
detector; tests 2, 3 pass (the legacy path already handled non-saturated
cases). That FAIL-by-default behaviour is the definition of a locked test.

These tests are pytest black-box: they import only public interfaces.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    SubmitResult,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    make_order,
)
from omiecid_simulator.omiecid.trade_extraction import RawFills, extract_fills


NTRADE_CAP = cst.NTRADE_CAP  # 100 at time of authoring


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_contract(contract_id: int = 37) -> Contract:
    """An OPEN contract (mid-day) usable for submit_order."""
    return Contract(
        contract_id=contract_id,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


def _fresh_book_and_state(
    contract_id: int = 37,
    market_config: OMIEMarketConfig | None = None,
    product_config: OMIEProductConfig | None = None,
):
    market_config = market_config or OMIEMarketConfig()
    product_config = product_config or OMIEProductConfig()
    book = ContractBook(_open_contract(contract_id), product_config, market_config)
    return book, book.init()


def _mk_buy(order_id, price, quantity, *, agent_id, timestamp,
            exec_restriction=ExecRestriction.NON, contract_id=37):
    return make_order(
        contract_id=contract_id, side=Side.BUY,
        price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction,
    )


def _mk_sell(order_id, price, quantity, *, agent_id, timestamp,
             exec_restriction=ExecRestriction.NON, contract_id=37):
    return make_order(
        contract_id=contract_id, side=Side.SELL,
        price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction,
    )


def _saturate_trades_via_replay(book, state, *, agent_a=10, agent_b=11,
                                 base_ts=1000, base_oid=1000) -> OMIEBookState:
    """Saturate `state.lob.trades` to NTRADE_CAP/NTRADE_CAP via realistic replay.

    Each iteration submits a resting SELL then a crossing BUY, producing
    exactly one new trade row. Repeats until the trades ring has wrapped at
    least once (>= NTRADE_CAP fills), so the array is fully saturated.

    This is explicitly NOT a hand-crafted sentinel array — it saturates the
    buffer by actually flowing orders through `submit_order`, exercising the
    real jaxob write pointer. The architecture audit gate requires this.
    """
    oid = base_oid
    ts = base_ts
    # Submit NTRADE_CAP + 5 clean matches to guarantee wrap-around.
    for _ in range(NTRADE_CAP + 5):
        sell = _mk_sell(oid, 50.00, 0.1, agent_id=agent_a, timestamp=ts)
        state, _r1 = book.submit_order(state, sell)
        oid += 1
        ts += 1
        buy = _mk_buy(oid, 50.00, 0.1, agent_id=agent_b, timestamp=ts)
        state, _r2 = book.submit_order(state, buy)
        oid += 1
        ts += 1

    # Invariant the saturation helper must uphold.
    OID = cst.TradesFeat.PASS_OID.value
    occ = int(jnp.sum(state.lob.trades[:, OID] != -1))
    assert occ == NTRADE_CAP, (
        f"saturation helper failed to fully saturate: occ={occ}/{NTRADE_CAP}"
    )
    return state


# ---------------------------------------------------------------------------
# Module-scoped saturated-buffer fixture
# ---------------------------------------------------------------------------
#
# P2-PERF-3 Deliverable (3): the saturation replay (~210 submit_order calls)
# is identical across the ring-buffer-immune fill detection tests below. Each
# test previously rebuilt the saturated state independently — under
# pyproject.toml's `-n 16` xdist pin, this was paying NTRADE_CAP+5 paired
# submits per test, 5x+ redundant across the 8 in-scope ring-buffer tests.
#
# Hoisting this to `scope="module"` is safe because:
#   * `OMIEBookState` is a NamedTuple (immutable; `_replace` is functional);
#   * `ContractBook` is documented stateless;
#   * Consumer tests obtain a new state via `submit_order`'s functional return
#     value — they never mutate the fixture state in place.
#
# Each consumer test still uses a distinct (oid, agent, timestamp) namespace
# (90000+, 91000+, etc.) to avoid collisions when starting from the shared
# saturated state.
#
# xdist workers receive their own module-scope fixture instance, so the
# hoist remains correct under `-n auto`.

@pytest.fixture(scope="module")
def saturated_book_and_state():
    """Saturated `(book, state)` shared across ring-buffer-immune tests.

    Wraps `_saturate_trades_via_replay` 1:1; logic is unchanged. Returns the
    same `(book, state)` tuple shape that consumer tests previously built
    inline via `_fresh_book_and_state()` + `_saturate_trades_via_replay()`.
    """
    book, state = _fresh_book_and_state()
    state = _saturate_trades_via_replay(book, state)
    return book, state


def _legacy_sentinel_extract(trades_before, trades_after) -> RawFills:
    """Pre-hotfix reference detector — the mask being REPLACED.

    Used only as an oracle for the empty-buffer and partial-occupancy
    parity tests (categories 2 and 3), where the legacy mask IS correct.
    Tests that check saturated-buffer behaviour MUST NOT use this oracle
    — the legacy detector is known-broken there.
    """
    P = cst.TradesFeat.P.value
    Q = cst.TradesFeat.Q.value
    OID = cst.TradesFeat.PASS_OID.value
    before_oids = trades_before[:, OID]
    after_oids = trades_after[:, OID]
    new_mask = (before_oids == -1) & (after_oids != -1)
    n_new = int(jnp.sum(new_mask))
    if n_new == 0:
        return RawFills(0, 0, 0)
    new_indices = jnp.where(new_mask, size=n_new)[0]
    new_trades = trades_after[new_indices]
    raw_qtys = jnp.abs(new_trades[:, Q])
    raw_prices = new_trades[:, P]
    total_qty = 0
    weighted = 0
    for i in range(n_new):
        q = int(raw_qtys[i])
        p = int(raw_prices[i])
        total_qty += q
        weighted += p * q
    return RawFills(total_qty, weighted, n_new)


# ---------------------------------------------------------------------------
# Category 1 — Saturated-buffer regression on extract_fills
# ---------------------------------------------------------------------------

class TestSaturatedBufferRegression:
    """extract_fills must detect fills even when the ring buffer is full."""

    def test_single_new_fill_detected_on_saturated_buffer(self, saturated_book_and_state):
        """K=1 expected fill on a saturated ring buffer.

        slow: realistic replay saturation (>= NTRADE_CAP matches) precedes
        the assertion.
        """
        book, state = saturated_book_and_state

        # Park a resting sell at 60.00 so the next buy produces exactly one
        # new fill.
        state, _ = book.submit_order(
            state,
            _mk_sell(90000, 60.00, 0.3, agent_id=20, timestamp=9000),
        )
        trades_before = state.lob.trades

        # Crossing buy consumes the sell at 60.00.
        state, _ = book.submit_order(
            state,
            _mk_buy(90001, 60.00, 0.3, agent_id=21, timestamp=9001),
        )
        trades_after = state.lob.trades

        fills = extract_fills(trades_before, trades_after)
        assert fills.num_fills == 1, (
            f"expected 1 new fill after saturated-buffer cross, got "
            f"{fills.num_fills}"
        )
        assert fills.total_qty_int > 0

    test_single_new_fill_detected_on_saturated_buffer = pytest.mark.slow(
        test_single_new_fill_detected_on_saturated_buffer
    )

    def test_multi_level_new_fills_detected_on_saturated_buffer(self, saturated_book_and_state):
        """Multi-level crossing under saturation — documents known write-side limitation.

        slow: realistic replay saturation.

        KNOWN LIMITATION (scoped OUT of this hotfix):
        The hotfix fixes DETECTION semantics in `trade_extraction.extract_fills`
        and the parallel site in `omie_data_loader.py:857-863`. It does NOT fix
        the upstream jaxob RING-BUFFER WRITE semantics. On a saturated buffer,
        `JaxOrderBookArrays.add_trade` uses `jnp.where(trades == -1, size=1,
        fill_value=-1)`, so every `add_trade` call within a single multi-level
        `submit_order` writes to slot -1, clobbering earlier fills of the same
        submit. The detector cannot recover information that was never persisted.
        Under saturation a two-level walk therefore yields `num_fills == 1`
        (the final surviving row). See the story card "Rejected alternative"
        deferring `LobState.n_trades` write-pointer work, and the tracking item
        "ring-buffer occupancy invariant" under
        `docs/omiecid/agile-plan.md` §P2-HOTFIX-ioc-fill-detection-FOLLOWUP.
        The parity check below proves the limitation is write-side (jaxob
        saturation), not detection-side: the identical input on a
        non-saturated buffer MUST still report `num_fills == 2`.
        """
        # --- Saturated-buffer leg ---
        book, state = saturated_book_and_state

        # Two resting sells at different prices.
        state, _ = book.submit_order(
            state, _mk_sell(91000, 60.00, 0.2, agent_id=30, timestamp=9100))
        state, _ = book.submit_order(
            state, _mk_sell(91001, 60.01, 0.2, agent_id=30, timestamp=9101))
        trades_before_sat = state.lob.trades

        # Buy walks both levels.
        state, _ = book.submit_order(
            state, _mk_buy(91002, 60.02, 0.4, agent_id=31, timestamp=9102))
        trades_after_sat = state.lob.trades

        fills_sat = extract_fills(trades_before_sat, trades_after_sat)
        assert fills_sat.num_fills == 1, (
            f"under saturation, multi-level walk surfaces only the final "
            f"surviving trade row (jaxob write-side limitation, see docstring); "
            f"expected num_fills == 1, got {fills_sat.num_fills}"
        )

        # --- Non-saturated parity leg (proves limitation is write-side) ---
        book_ns, state_ns = _fresh_book_and_state()

        state_ns, _ = book_ns.submit_order(
            state_ns, _mk_sell(81000, 60.00, 0.2, agent_id=30, timestamp=9100))
        state_ns, _ = book_ns.submit_order(
            state_ns, _mk_sell(81001, 60.01, 0.2, agent_id=30, timestamp=9101))
        trades_before_ns = state_ns.lob.trades

        state_ns, _ = book_ns.submit_order(
            state_ns, _mk_buy(81002, 60.02, 0.4, agent_id=31, timestamp=9102))
        trades_after_ns = state_ns.lob.trades

        fills_ns = extract_fills(trades_before_ns, trades_after_ns)
        assert fills_ns.num_fills == 2, (
            f"non-saturated parity: identical input on an empty ring buffer "
            f"MUST detect both fills (proves the limitation above is write-side, "
            f"not detection-side); expected num_fills == 2, got "
            f"{fills_ns.num_fills}"
        )

    test_multi_level_new_fills_detected_on_saturated_buffer = pytest.mark.slow(
        test_multi_level_new_fills_detected_on_saturated_buffer
    )


# ---------------------------------------------------------------------------
# Category 2 — Empty-buffer parity with legacy detector
# ---------------------------------------------------------------------------

class TestEmptyBufferParity:
    """On an empty trades array the new detector must match the legacy one."""

    def test_no_op_empty_returns_zero_fills(self):
        book, state = _fresh_book_and_state()
        before = state.lob.trades
        after = state.lob.trades  # no dispatch — identical
        fills_new = extract_fills(before, after)
        fills_legacy = _legacy_sentinel_extract(before, after)
        assert fills_new.num_fills == fills_legacy.num_fills == 0
        assert fills_new.total_qty_int == fills_legacy.total_qty_int == 0
        assert fills_new.weighted_price_int == fills_legacy.weighted_price_int == 0

    def test_single_fill_from_empty_buffer_parity(self):
        book, state = _fresh_book_and_state()
        state, _ = book.submit_order(
            state, _mk_sell(1, 50.00, 1.0, agent_id=1, timestamp=100))
        before = state.lob.trades
        state, _ = book.submit_order(
            state, _mk_buy(2, 50.00, 1.0, agent_id=2, timestamp=101))
        after = state.lob.trades

        fills_new = extract_fills(before, after)
        fills_legacy = _legacy_sentinel_extract(before, after)
        assert fills_new.num_fills == fills_legacy.num_fills == 1
        assert fills_new.total_qty_int == fills_legacy.total_qty_int
        assert fills_new.weighted_price_int == fills_legacy.weighted_price_int


# ---------------------------------------------------------------------------
# Category 3 — Partial-occupancy (50/100) parity
# ---------------------------------------------------------------------------

class TestPartialOccupancyParity:
    """At ~50/100 occupancy the legacy and new detectors must agree."""

    def test_partial_occupancy_parity_single_fill(self):
        """Drive occ to ~50/100 then cross once; both detectors must agree.

        slow: 50 priming matches + one checked match.
        """
        book, state = _fresh_book_and_state()
        oid = 1
        ts = 100
        target_occ = NTRADE_CAP // 2  # 50
        for _ in range(target_occ):
            state, _ = book.submit_order(
                state, _mk_sell(oid, 50.00, 0.1, agent_id=1, timestamp=ts))
            oid += 1
            ts += 1
            state, _ = book.submit_order(
                state, _mk_buy(oid, 50.00, 0.1, agent_id=2, timestamp=ts))
            oid += 1
            ts += 1

        OID = cst.TradesFeat.PASS_OID.value
        occ_before = int(jnp.sum(state.lob.trades[:, OID] != -1))
        assert occ_before == target_occ, (
            f"priming did not reach target occupancy: {occ_before}/{target_occ}"
        )

        # Next match: one fresh fill.
        state, _ = book.submit_order(
            state, _mk_sell(oid, 60.00, 0.3, agent_id=1, timestamp=ts))
        oid += 1; ts += 1
        before = state.lob.trades
        state, _ = book.submit_order(
            state, _mk_buy(oid, 60.00, 0.3, agent_id=2, timestamp=ts))
        after = state.lob.trades

        fills_new = extract_fills(before, after)
        fills_legacy = _legacy_sentinel_extract(before, after)
        assert fills_new.num_fills == fills_legacy.num_fills == 1
        assert fills_new.total_qty_int == fills_legacy.total_qty_int
        assert fills_new.weighted_price_int == fills_legacy.weighted_price_int

    test_partial_occupancy_parity_single_fill = pytest.mark.slow(
        test_partial_occupancy_parity_single_fill
    )


# ---------------------------------------------------------------------------
# Category 4 — No-op negative: non-crossing under saturation
# ---------------------------------------------------------------------------

class TestNoOpUnderSaturation:
    """A non-crossing order on a saturated buffer must yield zero fills.

    Guards against the whole-row diff over-reporting (e.g. falsely
    detecting row churn that didn't correspond to new trades).
    """

    def test_non_crossing_buy_under_saturation_is_zero_fills(self, saturated_book_and_state):
        """slow: realistic replay saturation."""
        book, state = saturated_book_and_state

        # Park a resting sell at 60.00; then submit a non-crossing buy at 40.00.
        state, _ = book.submit_order(
            state, _mk_sell(92000, 60.00, 0.3, agent_id=40, timestamp=9200))
        before = state.lob.trades
        state, _ = book.submit_order(
            state, _mk_buy(92001, 40.00, 0.3, agent_id=41, timestamp=9201))
        after = state.lob.trades

        fills = extract_fills(before, after)
        assert fills.num_fills == 0, (
            f"non-crossing buy must not be reported as a fill; got "
            f"num_fills={fills.num_fills}"
        )
        assert fills.total_qty_int == 0
        assert fills.weighted_price_int == 0

    test_non_crossing_buy_under_saturation_is_zero_fills = pytest.mark.slow(
        test_non_crossing_buy_under_saturation_is_zero_fills
    )


# ---------------------------------------------------------------------------
# Category 5 — Overflow guard on weighted_price_int
# ---------------------------------------------------------------------------

class TestWeightedPriceOverflowGuard:
    """Document the representable-int upper bound across parity scenarios.

    `weighted_price_int = sum(p_i * q_i)` is accumulated in Python ints
    (see `trade_extraction.py:89-95`), so the final Python scalar can
    exceed int32. This test asserts the value is non-negative and fits
    int64 — a cheap guard that will trip if a future refactor accidentally
    reintroduces int32 accumulation in JAX space.
    """

    INT64_MAX = 2**63 - 1

    def test_weighted_price_non_negative_and_within_int64(self, saturated_book_and_state):
        """slow: saturation replay."""
        book, state = saturated_book_and_state

        state, _ = book.submit_order(
            state, _mk_sell(93000, 60.00, 0.3, agent_id=50, timestamp=9300))
        before = state.lob.trades
        state, _ = book.submit_order(
            state, _mk_buy(93001, 60.00, 0.3, agent_id=51, timestamp=9301))
        after = state.lob.trades

        fills = extract_fills(before, after)
        assert fills.weighted_price_int >= 0
        # P2-L4 adapter: cast int64 leaf to Python int before comparing against int64-max literal — JAX canonicalises the RHS against int32 under x64-off and raises OverflowError on the int64-max literal.
        assert int(fills.weighted_price_int) < self.INT64_MAX, (
            "weighted_price_int must fit int64 — if this trips, something "
            "reintroduced constrained int32 accumulation"
        )
        assert fills.total_qty_int >= 0
        # P2-L4 adapter: cast int64 leaf to Python int before comparing against int64-max literal — JAX canonicalises the RHS against int32 under x64-off and raises OverflowError on the int64-max literal.
        assert int(fills.total_qty_int) < self.INT64_MAX

    test_weighted_price_non_negative_and_within_int64 = pytest.mark.slow(
        test_weighted_price_non_negative_and_within_int64
    )


# ---------------------------------------------------------------------------
# Category 6 — IOC #1022 end-to-end on 2026-01-12 contract 81
# ---------------------------------------------------------------------------

REAL_DATA_PATH = "data/rawOMIE"
REAL_DELIVERY_DATE = "2026-01-12"
REAL_CONTRACT_ID = 81
REAL_ORDERS_FILE = Path(REAL_DATA_PATH) / "orders_20260112.1"


# OMIE-TAPE-CONTRACT-DATE-1 (group R-c re-baseline, 2026-09-30). Under the L1
# loader the 2026-01-12 cid-81 tape is ONE product (session files 20260111 +
# 20260112), so order ids shift. The diagnostic-workbook IOC is re-identified
# by CONTENT: BUY IOC 119.49 EUR/MWh x 0.30 MW by (HC G, HCGC01), sent
# 12/01/2026 11:09:21 (stored 40,161 pre-L1 -> 126,561 = 40,161 + 86,400).
# Its id moves 1022 -> 1689 and, with the corrected resting book, its fill
# moves 114.69 -> 113.24 -> 113.15 -> 113.04 -> 113.09 EUR/MWh (qty 0.30 unchanged). (Pre-L1 facts stand as
# recorded; under L1 the unrelated order #1022 fills at 112.82 x 1.00.)
IOC_TARGET_ORDER_ID = 1689            # pre-L1: 1022
IOC_TARGET_TIMESTAMP = 126_561        # pre-L1 stored: 40_161
IOC_TARGET_AGENT_PAIR = ("HC G", "HCGC01")
IOC_TARGET_LIMIT_EUR = 119.49
# OMIE-ICE-INCOMING-LADDER-1 (licensed, E-C): the E-C row split of incoming
# icebergs saturates the real session ring earlier, so the IOC's derived fill
# price moves 113.04 -> 113.09 (qty 0.30 and traded unchanged); OQ-15,
# certificate saturated-ring gap (agile-plan v5.219, Step 6 amendment).
IOC_TARGET_FILL_PRICE_EUR = 113.09    # pre-L1: 114.69; pre-PRICE-THOUSANDS-SEP-1: 113.24; pre-OMIE-JAXOB-LOOKUP-MISS-1: 113.15; pre-OMIE-ICE-INCOMING-LADDER-1: 113.04
IOC_TARGET_FILL_QTY_MW = 0.30


class TestIOC1022EndToEnd:
    """Reproduce the original defect on real OMIE data.

    Replay 2026-01-12 contract 81 orders through `ContractBook.submit_order`
    in chronological order up to and including the diagnostic-workbook IOC
    (pre-L1 order #1022; post-L1 order #1689, re-identified by content — see
    the IOC_TARGET_* constants). Assert the IOC result is:
        result.traded == True
        trade_qty   == 0.30 MW    (within qty_tick tolerance)
        trade_price == 113.09 EUR/MWh (within price_tick tolerance; 114.69 pre-L1,
        113.24 pre-PRICE-THOUSANDS-SEP-1, 113.15 pre-OMIE-JAXOB-LOOKUP-MISS-1,
        113.04 pre-OMIE-ICE-INCOMING-LADDER-1 (E-C row split saturates the ring
        earlier; OQ-15, certificate saturated-ring gap))

    slow: loads the full 2026-01-12 contract 81 loader output and replays
    ~1000 orders through the book. Appropriate per the test-agent runtime
    policy for integration tests exercising the real loader cache.
    """

    @pytest.mark.slow
    @pytest.mark.skipif(
        not REAL_ORDERS_FILE.exists(),
        reason=f"Real OMIE orders file not present: {REAL_ORDERS_FILE}",
    )
    def test_ioc_1022_matches_at_114_69_eur_0_30_mw(self):
        from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_orders,
            row_to_order,
        )

        config = OMIEDataConfig(
            data_path=REAL_DATA_PATH,
            contract_id=REAL_CONTRACT_ID,
            delivery_date=REAL_DELIVERY_DATE,
        )
        orders = load_omie_orders(config)
        assert orders.shape[0] > 0, "empty order stream — cannot reproduce"

        product_config = OMIEProductConfig()
        market_config = OMIEMarketConfig()
        book, state = _fresh_book_and_state(
            contract_id=REAL_CONTRACT_ID,
            market_config=market_config,
            product_config=product_config,
        )

        # Re-identify the workbook IOC by content (send time, agent pair,
        # side, exec restriction, limit price, quantity) and pin its id.
        from omiecid_simulator.jaxomiecid.omie_data_loader import load_omie_agent_metadata
        from omiecid_simulator.omiecid.encoding import encode_price, encode_qty

        id_to_pair = load_omie_agent_metadata(config)
        rows = [
            i for i in range(orders.shape[0])
            if int(orders[i, 6]) == IOC_TARGET_TIMESTAMP
            and id_to_pair.get(int(orders[i, 5])) == IOC_TARGET_AGENT_PAIR
            and int(orders[i, 1]) == int(Side.BUY)
            and int(orders[i, 7]) == int(ExecRestriction.IOC)
            and int(orders[i, 2]) == encode_price(IOC_TARGET_LIMIT_EUR, product_config)
            and int(orders[i, 3]) == encode_qty(IOC_TARGET_FILL_QTY_MW, product_config)
        ]
        assert len(rows) == 1, f"workbook IOC not uniquely identified by content: rows {rows}"
        assert int(orders[rows[0], 4]) == IOC_TARGET_ORDER_ID

        # Feed orders in order; capture the SubmitResult for the target IOC.
        target_oid = IOC_TARGET_ORDER_ID
        captured: SubmitResult | None = None
        for i in range(orders.shape[0]):
            row = orders[i]
            order = row_to_order(row, product_config)
            state, result = book.submit_order(state, order)
            if order.order_id == target_oid:
                captured = result
                break

        assert captured is not None, (
            f"order #{target_oid} not observed in 2026-01-12 contract "
            f"{REAL_CONTRACT_ID} stream"
        )
        assert captured.traded == True, (
            "IOC #1022 must match (market-spec §4.5: IOC cancels residual, "
            "never rests); pre-hotfix this falsely reported traded=False"
        )
        # Tolerance: one price_tick / qty_tick — we assert the specific
        # domain-level values from the diagnostic workbook.
        assert abs(captured.trade_price - IOC_TARGET_FILL_PRICE_EUR) < product_config.price_tick + 1e-9
        assert abs(captured.trade_qty - IOC_TARGET_FILL_QTY_MW) < product_config.qty_tick + 1e-9


# ---------------------------------------------------------------------------
# Category 7 — IOC loop branch coverage under saturation
# ---------------------------------------------------------------------------

class TestIOCLoopBranchCoverageUnderSaturation:
    """Under saturation, IOC must take the fills-detected branch of the loop.

    This is an observable-behaviour proxy for the branch taken at
    `contract_book.py:402` ("if fills.num_fills > 0"). If the detector
    misreports zero under saturation, the loop falls through to the cancel
    tail at lines 441-446 and the caller sees `traded=False` despite jaxob
    having matched. We assert traded=True with positive qty/price under
    saturation as the externally-visible signal.
    """

    @pytest.mark.slow
    def test_ioc_under_saturation_reports_traded_true(self, saturated_book_and_state):
        book, state = saturated_book_and_state

        # Resting sell @ 60.00, 0.3 MW; then IOC buy crosses.
        state, _ = book.submit_order(
            state, _mk_sell(94000, 60.00, 0.3, agent_id=60, timestamp=9400))

        ioc_buy = _mk_buy(
            94001, 60.00, 0.3,
            agent_id=61, timestamp=9401,
            exec_restriction=ExecRestriction.IOC,
        )
        state, result = book.submit_order(state, ioc_buy)

        assert result.accepted == True
        assert result.traded == True, (
            "IOC under saturated ring-buffer must still report traded=True; "
            "pre-hotfix this falls through to the cancel tail"
        )
        assert result.trade_qty > 0.0
        assert result.trade_price > 0.0


# ---------------------------------------------------------------------------
# Category 8 — Loader replay-harness ring-buffer immunity
# ---------------------------------------------------------------------------

class TestReplayHarnessRingBufferImmunity:
    """`replay_orders_through_book` must be ring-buffer-immune too.

    The second occurrence of the `(before == -1) & (after != -1)` pattern
    lives in `omie_data_loader.replay_orders_through_book` (lines 857-863).
    The hotfix scope expanded to cover this site — after the fix, replaying
    the same input sequence must NOT silently drop fills once the buffer
    saturates.

    Strategy (synthetic, not dependent on real OMIE data): build an
    order stream whose first NTRADE_CAP+10 orders are clean 1:1 matches
    (saturating the buffer partway through), and whose trailing orders
    produce fills that MUST appear in the `replayed_trades` output.
    Compare the count of non-zero rows in the replay output against the
    known number of crossing events in the stream.
    """

    def _build_orders_and_historical(self, product_config, n_matches):
        """Construct (orders_int32_13col, historical_trades_int32_6col).

        Each match = one sell + one crossing buy at 50.00 EUR/MWh, 0.1 MW.
        The historical_trades array declares the expected fills so that
        replay_orders_through_book can count matched.
        """
        from omiecid_simulator.omiecid.encoding import encode_price, encode_qty

        P_int = encode_price(50.00, product_config)
        Q_int = encode_qty(0.1, product_config)
        # Synthetic 13-col rows matching post-P2-F0 loader schema.
        rows = []
        oid = 1
        ts = 100
        for _ in range(n_matches):
            # resting sell
            rows.append([
                37,                                  # contract_id
                int(Side.SELL),
                P_int, Q_int,
                oid, 1, ts,
                int(ExecRestriction.NON),
                0, 0, 0,                             # validity, expiry, status
                0, 0,                                # peak_qty, ppd
            ])
            oid += 1; ts += 1
            # crossing buy
            rows.append([
                37,
                int(Side.BUY),
                P_int, Q_int,
                oid, 2, ts,
                int(ExecRestriction.NON),
                0, 0, 0,
                0, 0,
            ])
            oid += 1; ts += 1
        orders = jnp.array(rows, dtype=jnp.int32)

        # Historical trades: one per match. 6-column int32 (cid, price_int,
        # qty_int, buy_aid, sell_aid, ts) matching the real loader layout
        # referenced in `replay_orders_through_book` decoding path.
        hist = []
        for i in range(n_matches):
            hist.append([37, P_int, Q_int, 2, 1, 100 + 2 * i + 1])
        historical = jnp.array(hist, dtype=jnp.int32)
        return orders, historical

    @pytest.mark.slow
    def test_replay_harness_no_silently_dropped_fills_after_saturation(self):
        """slow: builds > NTRADE_CAP matches and replays through the book."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            replay_orders_through_book,
        )

        product_config = OMIEProductConfig()
        market_config = OMIEMarketConfig()
        n_matches = NTRADE_CAP + 20  # guarantee saturation mid-replay
        orders, historical = self._build_orders_and_historical(
            product_config, n_matches,
        )

        book = ContractBook(_open_contract(37), product_config, market_config)
        result = replay_orders_through_book(
            orders=orders,
            contract_book=book,
            historical_trades=historical,
            heuristic="rest_until_gct",
        )

        # Every match is a clean 1:1 at the same price and qty, so every
        # historical trade must be reproduced. Pre-hotfix: the inline
        # sentinel mask at loader.py:857-863 drops fills post-saturation,
        # so historical_trades_matched < historical_trades_total.
        assert result.historical_trades_total == n_matches
        assert result.historical_trades_matched == n_matches, (
            f"replay dropped fills across the saturation boundary: "
            f"matched={result.historical_trades_matched} of "
            f"{result.historical_trades_total}. Pre-hotfix symptom of the "
            f"duplicated sentinel-transition pattern at "
            f"omie_data_loader.py:857-863."
        )

    @pytest.mark.slow
    def test_replay_harness_parity_saturated_vs_nonsaturated(self):
        """Same input under two sizings must produce identical match counts.

        Running the harness with a small match count (buffer never saturates)
        must give the same match_rate as the saturated-mid-replay scenario
        after the fix — the detector is ring-buffer-immune by construction.
        """
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            replay_orders_through_book,
        )
        product_config = OMIEProductConfig()
        market_config = OMIEMarketConfig()

        # Non-saturated run.
        orders_small, hist_small = self._build_orders_and_historical(
            product_config, NTRADE_CAP // 2,
        )
        book_a = ContractBook(_open_contract(37), product_config, market_config)
        r_small = replay_orders_through_book(
            orders=orders_small, contract_book=book_a,
            historical_trades=hist_small, heuristic="rest_until_gct",
        )

        # Saturated-mid-replay run.
        orders_big, hist_big = self._build_orders_and_historical(
            product_config, NTRADE_CAP + 20,
        )
        book_b = ContractBook(_open_contract(37), product_config, market_config)
        r_big = replay_orders_through_book(
            orders=orders_big, contract_book=book_b,
            historical_trades=hist_big, heuristic="rest_until_gct",
        )

        # Both runs must reach 100% match rate; the ratio is what's invariant.
        assert r_small.historical_trades_matched == r_small.historical_trades_total
        assert r_big.historical_trades_matched == r_big.historical_trades_total
