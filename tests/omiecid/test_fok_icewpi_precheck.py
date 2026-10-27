"""Tests for M6.5 -- FOK ICEwPI Slice-by-Slice Pre-Check.

Behavioral contracts for the corrected FOK pre-check and fill logic when
interacting with ICEwPI icebergs (iceberg orders with non-zero peak price
delta). Per OQ-11 resolution and market-spec SS4.2.1 Rule 1 (updated).

Two changes being tested:
  (1) `_matchable_volume()` must project slice-by-slice drifted prices and
      only count slices whose projected price actually crosses the FOK limit.
      Previously it counted ALL hidden ICEwPI quantity as matchable if the
      base price crossed.
  (2) `_regenerate_iceberg_slices()` must no longer accept `fok_limit_price`
      or `fok_side` parameters. All FOK regeneration uses natural drifted
      prices. No price capping occurs.

Domain rules:
  - SIDC Continuous Trading Matching Algorithm SS3.3: ICEwPI slices have
    distinct limit prices; each new slice is "entered with a new limit price."
  - SS5.2.1: slices "participate with each slice individually... until either
    all slices are fully executed or no further execution is possible anymore."
  - SS5.1.2: "the price is always determined by the iceberg order already in
    the order book." (no price capping)
  - SS3.5.3: FOK "executed immediately and with its full quantity or...deleted."

Acceptance criteria (M6.5 from agile-plan.md):
  AC-1: ICEwPI SELL base=50.00, ppd=+0.50, peak=2, total=10:
        FOK BUY limit=51.00 for 8 MW is rejected (only 6 MW crossing).
  AC-2: Same iceberg: FOK BUY limit=51.00 for 6 MW is accepted,
        fills at natural iceberg prices (50.00, 50.50, 51.00).
  AC-3: `_regenerate_iceberg_slices` no longer accepts `fok_limit_price`
        or `fok_side` parameters.
  AC-4: Standard icebergs (PPD=0) pre-check behavior unchanged.
  AC-5: All existing tests still pass (verified outside this file).

Test categories:
  1. TestICEwPIFOKPreCheckCore -- core acceptance criteria AC-1 and AC-2
  2. TestICEwPIFOKNaturalPrices -- fills use natural drifted prices, no capping
  3. TestICEwPIRegenerateSignature -- AC-3: no fok_limit_price/fok_side params
  4. TestStandardIcebergUnchanged -- AC-4: PPD=0 behavior unchanged
  5. TestICEwPIFOKEdgeCases -- boundary and edge cases
  6. TestICEwPIFOKSellSideSymmetry -- sell-side FOK against buy-side ICEwPI

Sources:
  - docs/omiecid/market-spec.md SS4.2.1 (Rule 1, updated OQ-11)
  - docs/omiecid/open-questions.md OQ-11
  - Continuous Trading Matching Algorithm SS3.3, SS3.5.3, SS5.1.2, SS5.2.1
"""

import inspect

import pytest
from pytest import approx

from omiecid_simulator.omiecid.order_types import (
    Side,
    ExecRestriction,
    make_order,
)
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig
from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTRACT_ID = 42


def _make_contract():
    return Contract(
        contract_id=_CONTRACT_ID,
        delivery_start_minutes=630,   # 10:30
        delivery_end_minutes=645,     # 10:45
        gct_minutes=570,              # 09:30
        status=ContractStatus.OPEN,
    )


def _make_book(book_depth=100):
    return ContractBook(
        contract=_make_contract(),
        product_config=OMIEProductConfig(),
        market_config=OMIEMarketConfig(book_depth=book_depth),
    )


def _order(side, price, quantity, order_id, agent_id, timestamp,
           exec_restriction=ExecRestriction.NON, peak_qty=0.0, ppd=0.0):
    return make_order(
        contract_id=_CONTRACT_ID, side=side, price=price, quantity=quantity,
        order_id=order_id, agent_id=agent_id, timestamp=timestamp,
        exec_restriction=exec_restriction, peak_qty=peak_qty, ppd=ppd,
    )


# ---------------------------------------------------------------------------
# 1. Core acceptance criteria: ICEwPI FOK slice-by-slice pre-check
# ---------------------------------------------------------------------------

class TestICEwPIFOKPreCheckCore:
    """FOK pre-check against ICEwPI icebergs must project slice-by-slice
    drifted prices and only count slices whose projected price crosses.

    Reference scenario from OQ-11 and market-spec SS4.2.1 Rule 1:
      ICEwPI SELL: base=50.00, ppd=+0.50, peak=2 MW, total=10 MW.
      Slices at natural prices:
        Slice 0: 2 MW @ 50.00  (base)
        Slice 1: 2 MW @ 50.50
        Slice 2: 2 MW @ 51.00
        Slice 3: 2 MW @ 51.50  <- does NOT cross FOK BUY limit=51.00
        Slice 4: 2 MW @ 52.00  <- does NOT cross
      Matchable volume for FOK BUY limit=51.00 = 6 MW (slices 0-2).
    """

    def test_fok_buy_8mw_rejected_only_6mw_crossing(self):
        """AC-1: FOK BUY limit=51.00 for 8 MW is rejected.

        Matchable = 6 MW (slices 0-2 at 50.00/50.50/51.00).
        8 MW > 6 MW matchable -> FOK rejected, no trade.
        """
        book = _make_book()
        state = book.init()

        # ICEwPI SELL: base=50.00, ppd=+0.50, peak=2 MW, total=10 MW
        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, r_ice = book.submit_order(state, ice)
        assert r_ice.accepted

        # FOK BUY 8 MW at 51.00 -- only 6 MW crossing -> rejected
        fok = _order(Side.BUY, 51.00, 8.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_fok_buy_6mw_accepted_all_crossing(self):
        """AC-2: FOK BUY limit=51.00 for 6 MW is accepted and fills.

        Matchable = 6 MW (slices 0-2). FOK requests exactly 6 MW -> accepted.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, r_ice = book.submit_order(state, ice)
        assert r_ice.accepted

        fok = _order(Side.BUY, 51.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.accepted == True
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(6.0)


# ---------------------------------------------------------------------------
# 2. FOK fills at natural drifted prices (no price capping)
# ---------------------------------------------------------------------------

class TestICEwPIFOKNaturalPrices:
    """When a FOK is accepted against an ICEwPI iceberg, fills occur at the
    iceberg's natural drifted prices, NOT capped at the FOK limit.

    Per SIDC SS5.1.2: "the price is always determined by the iceberg order
    already in the order book."
    """

    def test_fok_vwap_reflects_natural_drifted_prices(self):
        """AC-2 price check: FOK BUY 6 MW fills at VWAP of natural prices.

        Fills: 2@50.00, 2@50.50, 2@51.00
        VWAP = (2*50.00 + 2*50.50 + 2*51.00) / 6 = 302.00 / 6 = 50.333...

        If price capping were in effect (old behavior), all slices would trade
        at 51.00 and VWAP would be 51.00. This test verifies that capping
        does NOT occur.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 51.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)

        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(6.0)
        # VWAP = (2*50.00 + 2*50.50 + 2*51.00) / 6 = 50.3333...
        expected_vwap = (2 * 50.00 + 2 * 50.50 + 2 * 51.00) / 6.0
        assert r_fok.trade_price == approx(expected_vwap, abs=0.01)
        # Crucially, the VWAP must be LESS than the FOK limit (51.00),
        # proving no price capping occurred.
        assert r_fok.trade_price < 51.00

    def test_fok_fill_prices_not_capped_at_fok_limit(self):
        """Regression: If capping were active, VWAP would equal the FOK limit.

        Use a scenario where capping would be obvious: only the first slice
        is below the FOK limit, rest are at or above.

        ICEwPI SELL: base=50.00, ppd=+1.00, peak=1 MW, total=3 MW.
        Slices: 1@50.00, 1@51.00, 1@52.00.
        FOK BUY limit=51.00 for 2 MW.
        Crossing slices: 1@50.00, 1@51.00 = 2 MW. Accepted.
        VWAP = (50.00 + 51.00) / 2 = 50.50
        If capped, both would be at 51.00, VWAP = 51.00.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 3.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=1.0, ppd=1.00)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 51.00, 2.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)

        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(2.0)
        assert r_fok.trade_price == approx(50.50, abs=0.01)
        assert r_fok.trade_price < 51.00  # no capping


# ---------------------------------------------------------------------------
# 3. _regenerate_iceberg_slices signature: no fok_limit_price/fok_side
# ---------------------------------------------------------------------------

class TestICEwPIRegenerateSignature:
    """AC-3: _regenerate_iceberg_slices no longer accepts fok_limit_price
    or fok_side parameters. All FOK regeneration uses natural drifted prices.

    This is an interface contract test. The method signature should only
    accept (self, state) with no FOK-specific keyword arguments.
    """

    def test_regenerate_has_no_fok_limit_price_param(self):
        """_regenerate_iceberg_slices must not have a fok_limit_price parameter."""
        sig = inspect.signature(ContractBook._regenerate_iceberg_slices)
        param_names = list(sig.parameters.keys())
        assert "fok_limit_price" not in param_names, (
            "_regenerate_iceberg_slices still accepts fok_limit_price; "
            "this parameter must be removed per OQ-11 resolution"
        )

    def test_regenerate_has_no_fok_side_param(self):
        """_regenerate_iceberg_slices must not have a fok_side parameter."""
        sig = inspect.signature(ContractBook._regenerate_iceberg_slices)
        param_names = list(sig.parameters.keys())
        assert "fok_side" not in param_names, (
            "_regenerate_iceberg_slices still accepts fok_side; "
            "this parameter must be removed per OQ-11 resolution"
        )

    def test_regenerate_signature_is_self_and_state_only(self):
        """_regenerate_iceberg_slices should accept exactly (self, state)."""
        sig = inspect.signature(ContractBook._regenerate_iceberg_slices)
        param_names = list(sig.parameters.keys())
        assert param_names == ["self", "state"], (
            f"Expected signature (self, state), got parameters: {param_names}"
        )


# ---------------------------------------------------------------------------
# 4. Standard icebergs (PPD=0) pre-check behavior unchanged
# ---------------------------------------------------------------------------

class TestStandardIcebergUnchanged:
    """AC-4: Standard icebergs (PPD=0) are unaffected by the ICEwPI changes.

    For PPD=0, all slices have the same price. Total remaining quantity is
    the correct matchable volume, same as before M6.5.
    """

    def test_standard_iceberg_fok_accepted_full_quantity(self):
        """FOK BUY for total quantity accepted against standard iceberg.

        Standard iceberg SELL: price=50.00, peak=2, total=10, ppd=0.
        All 10 MW at 50.00 -> matchable = 10 MW.
        FOK BUY 10 MW at 50.00 -> accepted.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 10.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.accepted == True
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(10.0)

    def test_standard_iceberg_fok_rejected_over_total(self):
        """FOK BUY exceeding total rejected against standard iceberg.

        Standard iceberg SELL: price=50.00, peak=2, total=10, ppd=0.
        FOK BUY 10.1 MW at 50.00 -> rejected (10.1 > 10).
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 10.1, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_standard_iceberg_all_fills_at_same_price(self):
        """Standard iceberg VWAP equals the base price (all slices same price).

        Standard iceberg SELL: price=50.00, peak=2, total=6, ppd=0.
        FOK BUY 6 MW at 50.00 -> fills 3 slices all at 50.00. VWAP=50.00.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(6.0)
        assert r_fok.trade_price == approx(50.00)

    def test_standard_iceberg_fok_price_not_crossing_rejected(self):
        """Standard iceberg at price above FOK limit -> 0 matchable -> rejected.

        Standard iceberg SELL: price=52.00, ppd=0, peak=2, total=6.
        FOK BUY limit=51.00 -> 0 MW crossing (52.00 > 51.00) -> rejected.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 52.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.0)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 51.00, 2.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False


# ---------------------------------------------------------------------------
# 5. ICEwPI FOK edge cases
# ---------------------------------------------------------------------------

class TestICEwPIFOKEdgeCases:
    """Edge cases for ICEwPI slice-by-slice pre-check."""

    def test_fok_exact_boundary_slice_included(self):
        """Slice exactly at FOK limit price IS included (price crosses at ==).

        ICEwPI SELL: base=50.00, ppd=+0.50, peak=2 MW, total=10 MW.
        FOK BUY limit=51.00. Slice 2 at 51.00 crosses (51.00 <= 51.00).
        Matchable = 6 MW. FOK for 6 MW -> accepted.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 51.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(6.0)

    def test_fok_one_tick_below_next_slice_rejects_correctly(self):
        """FOK limit just below the next non-crossing slice still rejects.

        ICEwPI SELL: base=50.00, ppd=+0.50, peak=2 MW, total=10 MW.
        FOK BUY limit=51.49 (just below 51.50).
        Slices: 2@50.00, 2@50.50, 2@51.00 = 6 MW crossing.
        Slice at 51.50 does NOT cross (51.50 > 51.49).
        FOK for 7 MW -> rejected (only 6 MW matchable).
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 51.49, 7.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_fok_limit_covers_all_slices(self):
        """FOK limit high enough to cover all ICEwPI slices -> full fill.

        ICEwPI SELL: base=50.00, ppd=+0.50, peak=2 MW, total=10 MW.
        Max slice price = 50.00 + 0.50 * 4 = 52.00 (5th slice, index 4).
        FOK BUY limit=52.00 for 10 MW -> all 10 MW crossing -> accepted.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 52.00, 10.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(10.0)

    def test_fok_limit_covers_all_slices_vwap(self):
        """VWAP for full ICEwPI fill reflects all drifted prices.

        ICEwPI SELL: base=50.00, ppd=+0.50, peak=2 MW, total=10 MW.
        Fills: 2@50.00, 2@50.50, 2@51.00, 2@51.50, 2@52.00
        VWAP = (100+101+102+103+104)/10 = 510/10 = 51.00
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 52.00, 10.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(10.0)
        expected_vwap = (2*50.00 + 2*50.50 + 2*51.00 + 2*51.50 + 2*52.00) / 10.0
        assert r_fok.trade_price == approx(expected_vwap, abs=0.01)

    def test_fok_only_base_slice_crosses(self):
        """FOK limit equals base price -> only first slice (peak) crosses.

        ICEwPI SELL: base=50.00, ppd=+0.50, peak=2 MW, total=10 MW.
        FOK BUY limit=50.00 -> only slice 0 at 50.00 crosses.
        Slice 1 at 50.50 does NOT cross.
        Matchable = 2 MW.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        # FOK for 2 MW at 50.00 -> accepted (exactly the base slice)
        fok = _order(Side.BUY, 50.00, 2.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(2.0)
        assert r_fok.trade_price == approx(50.00)

    def test_fok_only_base_crosses_rejects_for_more(self):
        """FOK requesting more than base-crossing volume is rejected.

        Same scenario: only 2 MW at 50.00 crosses.
        FOK BUY 3 MW at 50.00 -> rejected (3 > 2).
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 50.00, 3.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_icewpi_partial_last_slice(self):
        """ICEwPI with total not evenly divisible by peak.

        ICEwPI SELL: base=50.00, ppd=+1.00, peak=3 MW, total=7 MW.
        Slices: 3@50.00, 3@51.00, 1@52.00 (partial last slice).
        FOK BUY limit=51.00 for 6 MW -> 3@50.00 + 3@51.00 = 6 MW crossing.
        Accepted.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=3.0, ppd=1.00)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 51.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(6.0)

    def test_icewpi_partial_last_slice_rejected(self):
        """Partial last slice is counted when it crosses, but still rejects if over.

        ICEwPI SELL: base=50.00, ppd=+1.00, peak=3 MW, total=7 MW.
        Slices: 3@50.00, 3@51.00, 1@52.00.
        FOK BUY limit=52.00 -> all 7 MW cross. FOK for 8 MW -> rejected.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 7.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=3.0, ppd=1.00)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 52.00, 8.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_icewpi_with_mixed_limit_and_iceberg_liquidity(self):
        """FOK pre-check accounts for both limit orders and ICEwPI crossing slices.

        Limit SELL: 3 MW @ 50.00 (visible, crosses FOK BUY limit=51.00).
        ICEwPI SELL: base=50.50, ppd=+0.50, peak=2 MW, total=6 MW.
          Slices: 2@50.50, 2@51.00, 2@51.50 (last doesn't cross 51.00).
          Crossing slices: 2@50.50 + 2@51.00 = 4 MW.
        Total matchable: 3 (limit) + 4 (ICEwPI crossing) = 7 MW.
        FOK BUY 7 MW at 51.00 -> accepted.
        """
        book = _make_book()
        state = book.init()

        # Limit SELL
        lim = _order(Side.SELL, 50.00, 3.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        # ICEwPI SELL
        ice = _order(Side.SELL, 50.50, 6.0, 2, agent_id=1, timestamp=2,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        # FOK BUY 7 MW
        fok = _order(Side.BUY, 51.00, 7.0, 3, agent_id=2, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(7.0)

    def test_icewpi_mixed_liquidity_fok_rejected_over_crossing(self):
        """FOK exceeding combined limit + ICEwPI crossing volume is rejected.

        Same setup: 3 MW limit + 4 MW ICEwPI crossing = 7 MW.
        FOK BUY 8 MW -> rejected.
        """
        book = _make_book()
        state = book.init()

        lim = _order(Side.SELL, 50.00, 3.0, 1, agent_id=1, timestamp=1)
        state, _ = book.submit_order(state, lim)

        ice = _order(Side.SELL, 50.50, 6.0, 2, agent_id=1, timestamp=2,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.BUY, 51.00, 8.0, 3, agent_id=2, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_icewpi_already_partially_consumed(self):
        """Pre-check counts only REMAINING slices, not already-consumed ones.

        ICEwPI SELL: base=50.00, ppd=+0.50, peak=2 MW, total=10 MW.
        First, a NON BUY consumes some slices. Then check FOK pre-check
        against the residual.

        NON BUY 2 MW at 50.00 -> consumes slice 0 (2@50.00).
        Remaining: total_hidden went from 8 to 6, slice_count=1.
        Next slices: 2@50.50, 2@51.00, 2@51.50, 2@52.00.
        FOK BUY limit=51.00 for 4 MW -> crossing: 2@50.50 + 2@51.00 = 4 MW.
        Accepted.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        # NON BUY consumes the base slice
        non = _order(Side.BUY, 50.00, 2.0, 2, agent_id=2, timestamp=2)
        state, r_non = book.submit_order(state, non)
        assert r_non.traded == True
        assert r_non.trade_qty == approx(2.0)

        # FOK BUY 4 MW at 51.00 -> slices at 50.50, 51.00 cross -> 4 MW
        fok = _order(Side.BUY, 51.00, 4.0, 3, agent_id=2, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(4.0)

    def test_icewpi_already_partially_consumed_rejects_over(self):
        """After partial consumption, FOK exceeding remaining crossing is rejected.

        Same as above: after consuming 2 MW, remaining crossing at 51.00 = 4 MW.
        FOK BUY 5 MW at 51.00 -> rejected (5 > 4).
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.SELL, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=0.50)
        state, _ = book.submit_order(state, ice)

        non = _order(Side.BUY, 50.00, 2.0, 2, agent_id=2, timestamp=2)
        state, _ = book.submit_order(state, non)

        fok = _order(Side.BUY, 51.00, 5.0, 3, agent_id=2, timestamp=3,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False


# ---------------------------------------------------------------------------
# 6. Sell-side FOK against buy-side ICEwPI symmetry
# ---------------------------------------------------------------------------

class TestICEwPIFOKSellSideSymmetry:
    """FOK SELL against ICEwPI BUY: symmetric slice-by-slice pre-check.

    ICEwPI BUY: base=50.00, ppd=-0.50 (price DECREASES per slice).
    Slices: 2@50.00, 2@49.50, 2@49.00, 2@48.50, 2@48.00.
    FOK SELL limit=49.00 -> bid must be >= sell limit.
    Crossing: 2@50.00, 2@49.50, 2@49.00 = 6 MW.
    Slice at 48.50 does NOT cross (48.50 < 49.00).
    """

    def test_fok_sell_against_icewpi_buy_rejected(self):
        """FOK SELL 8 MW at limit=49.00 rejected (only 6 MW crossing)."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=-0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.SELL, 49.00, 8.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False

    def test_fok_sell_against_icewpi_buy_accepted(self):
        """FOK SELL 6 MW at limit=49.00 accepted (exactly 6 MW crossing)."""
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=-0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.SELL, 49.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(6.0)

    def test_fok_sell_against_icewpi_buy_natural_vwap(self):
        """FOK SELL fill VWAP reflects natural drifted bid prices.

        Fills: 2@50.00, 2@49.50, 2@49.00.
        VWAP = (100 + 99 + 98) / 6 = 297 / 6 = 49.50.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 50.00, 10.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=-0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.SELL, 49.00, 6.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == True
        assert r_fok.trade_qty == approx(6.0)
        expected_vwap = (2*50.00 + 2*49.50 + 2*49.00) / 6.0
        assert r_fok.trade_price == approx(expected_vwap, abs=0.01)

    def test_fok_sell_rejects_when_no_bid_crosses(self):
        """ICEwPI BUY base below FOK SELL limit -> 0 crossing -> rejected.

        ICEwPI BUY: base=48.00, ppd=-0.50, peak=2, total=6.
        Slices: 2@48.00, 2@47.50, 2@47.00.
        FOK SELL limit=49.00 -> none cross (all < 49.00). Rejected.
        """
        book = _make_book()
        state = book.init()

        ice = _order(Side.BUY, 48.00, 6.0, 1, agent_id=1, timestamp=1,
                     exec_restriction=ExecRestriction.ICE,
                     peak_qty=2.0, ppd=-0.50)
        state, _ = book.submit_order(state, ice)

        fok = _order(Side.SELL, 49.00, 2.0, 2, agent_id=2, timestamp=2,
                     exec_restriction=ExecRestriction.FOK)
        state, r_fok = book.submit_order(state, fok)
        assert r_fok.traded == False
