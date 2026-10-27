"""Tests for M3.1w -- Cross-Module Wiring Smoke Test.

Lightweight integration test that imports and chains M1 (config, order types,
validator) + M2 (contract) + M3.1 (ContractBook) together.  Verifies the
modules wire up end-to-end: create a config, create a contract, create an
order, validate it, submit it to a ContractBook, and read it back.

Not a matching test -- just a wiring check.

Two acceptance-criteria tests:
  1. test_config_to_contract_to_book_wiring  -- happy path end-to-end
  2. test_validator_rejects_before_book_accepts -- rejected order never enters book

Source: docs/omiecid/agile-plan.md (M3.1w)
Domain: docs/omiecid/market-spec.md
Architecture: docs/omiecid/migration-plan.md
"""

import pytest

# M1.1 -- Config
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig

# M1.2 -- Order types
from omiecid_simulator.omiecid.order_types import (
    OMIEOrder,
    Side,
    ExecRestriction,
    Validity,
    Status,
    make_order,
)

# M1.3 -- Validator
from omiecid_simulator.omiecid.order_validator import validate_order

# M2.1 -- Contract
from omiecid_simulator.omiecid.contract import Contract, ContractStatus

# M3.1 -- ContractBook
from omiecid_simulator.omiecid.contract_book import ContractBook, SubmitResult

# --- P2-H2 Category-B helper: get_order returns sentinel OMIEOrder, not None.
from omiecid_simulator.omiecid.jax_constants import (
    SENTINEL_ORDER_ID as _P2H2_SENTINEL_ORDER_ID,  # noqa: E501
)


def _is_absent(order):
    """Return True iff ``order`` is Phase-1 ``None`` OR the post-P2-H2 sentinel.

    The post-P2-H2 ``get_order`` returns a sentinel ``OMIEOrder`` whose
    ``order_id == SENTINEL_ORDER_ID`` for absent ids. This helper bridges
    both representations so callsites that pre-date P2-H2 keep working
    without per-site mechanical rewrites.
    """
    if order is None:
        return True
    try:
        return int(order.order_id) == _P2H2_SENTINEL_ORDER_ID
    except (AttributeError, TypeError):
        return False




# ---------------------------------------------------------------------------
# Fixtures (product_config, market_config, open_contract, book from conftest)
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_buy_order():
    """A valid BUY limit order on contract 37."""
    return make_order(
        contract_id=37,
        side=Side.BUY,
        price=50.00,
        quantity=1.0,
        order_id=1,
        agent_id=100,
        timestamp=1000,
    )


# ---------------------------------------------------------------------------
# Test 1: Config -> Contract -> Order -> Validate -> Book -> Read back
# ---------------------------------------------------------------------------

class TestConfigToContractToBookWiring:
    """End-to-end wiring: create config, create contract, create order,
    validate, submit to ContractBook, read it back."""

    def test_config_to_contract_to_book_wiring(
        self, product_config, market_config, open_contract, book, valid_buy_order,
    ):
        """Full happy path: config -> contract -> order -> validate -> submit -> read back."""
        # Step 1: Configs exist and have expected types
        assert isinstance(product_config, OMIEProductConfig)
        assert isinstance(market_config, OMIEMarketConfig)

        # Step 2: Contract exists with OPEN status
        assert isinstance(open_contract, Contract)
        assert open_contract.status == ContractStatus.OPEN

        # Step 3: Order exists with expected fields
        assert isinstance(valid_buy_order, OMIEOrder)
        assert valid_buy_order.side == Side.BUY
        assert valid_buy_order.contract_id == 37

        # Step 4: Validator accepts the order (standalone, no book needed)
        contract_is_open = open_contract.status == ContractStatus.OPEN
        assert bool(validate_order(valid_buy_order, product_config, contract_is_open)) is True

        # Step 5: Submit order to ContractBook
        state = book.init()
        new_state, result = book.submit_order(state, valid_buy_order)
        assert isinstance(result, SubmitResult)
        assert result.accepted == True

        # Step 6: Read the order back from the book
        retrieved = book.get_order(new_state, valid_buy_order.order_id, Side.BUY)
        assert not _is_absent(retrieved)
        assert isinstance(retrieved, OMIEOrder)
        assert retrieved.order_id == valid_buy_order.order_id
        assert retrieved.side == Side.BUY
        assert retrieved.price == pytest.approx(valid_buy_order.price)
        assert retrieved.quantity == pytest.approx(valid_buy_order.quantity)
        assert retrieved.agent_id == valid_buy_order.agent_id

    def test_sell_order_wiring(
        self, product_config, book,
    ):
        """Same wiring path but for a SELL order, confirming both sides work."""
        sell_order = make_order(
            contract_id=37,
            side=Side.SELL,
            price=55.00,
            quantity=2.0,
            order_id=2,
            agent_id=200,
            timestamp=2000,
        )

        # Validate standalone
        assert bool(validate_order(sell_order, product_config, contract_open=True)) is True

        # Submit and read back
        state = book.init()
        new_state, result = book.submit_order(state, sell_order)
        assert result.accepted == True

        retrieved = book.get_order(new_state, sell_order.order_id, Side.SELL)
        assert not _is_absent(retrieved)
        assert retrieved.order_id == sell_order.order_id
        assert retrieved.side == Side.SELL
        assert retrieved.price == pytest.approx(55.00)
        assert retrieved.quantity == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 2: Validator rejects -> book never accepts
# ---------------------------------------------------------------------------

class TestValidatorRejectsBeforeBookAccepts:
    """Verify that an order rejected by the validator is also rejected by the
    ContractBook (which delegates to the validator internally).

    As of P2-L1-HOTFIX-tick-precision (Option C), sub-half-tick off-tick
    probes snap onto the nearest tick via the encoder at both the validator
    and the book. The cross-module invariant still holds in the inverse form
    (validator accepts <=> contract_book accepts for the same snapped
    input); the positive-assertion "both reject sub-half-tick" probe has
    therefore been removed. The encoder-layer snap contract lives in
    ``tests/omiecid/test_encoding_snap_semantics.py``. Out-of-bounds, closed-
    contract, and valid-ICE probes below still exercise the wiring seam.
    """

    def test_validator_rejects_before_book_accepts(
        self, product_config, market_config, open_contract, book,
    ):
        """An order with price above max is rejected by validator and by book."""
        bad_order = make_order(
            contract_id=37,
            side=Side.BUY,
            price=99999.99,  # way above price_max (9999.0)
            quantity=1.0,
            order_id=10,
            agent_id=100,
            timestamp=3000,
        )

        # Step 1: Validator rejects standalone
        contract_is_open = open_contract.status == ContractStatus.OPEN
        assert bool(validate_order(bad_order, product_config, contract_is_open)) is False

        # Step 2: ContractBook also rejects (delegates to validator)
        state = book.init()
        new_state, result = book.submit_order(state, bad_order)
        assert result.accepted == False

        # Step 3: Order is NOT in the book
        retrieved = book.get_order(new_state, bad_order.order_id, Side.BUY)
        assert _is_absent(retrieved)

    def test_closed_contract_rejected_by_both(
        self, product_config, market_config,
    ):
        """An order on a CLOSED contract is rejected by both validator and book."""
        closed_contract = Contract(
            contract_id=37,
            delivery_start_minutes=9 * 60 + 15,
            delivery_end_minutes=9 * 60 + 30,
            gct_minutes=9 * 60 + 15 - 60,
            status=ContractStatus.CLOSED,
        )

        order = make_order(
            contract_id=37,
            side=Side.BUY,
            price=50.00,
            quantity=1.0,
            order_id=20,
            agent_id=100,
            timestamp=4000,
        )

        # Validator rejects (contract not open)
        assert bool(validate_order(order, product_config, contract_open=False)) is False

        # Book also rejects
        closed_book = ContractBook(closed_contract, product_config, market_config)
        state = closed_book.init()
        new_state, result = closed_book.submit_order(state, order)
        assert result.accepted == False
        assert _is_absent(closed_book.get_order(new_state, order.order_id, Side.BUY))

    def test_valid_iceberg_accepted_by_both(
        self, product_config, book,
    ):
        """A valid ICE order is accepted by validator and book (M4.1)."""
        ice_order = make_order(
            contract_id=37,
            side=Side.BUY,
            price=50.00,
            quantity=10.0,
            order_id=30,
            agent_id=100,
            timestamp=5000,
            exec_restriction=ExecRestriction.ICE,
            peak_qty=2.0,
        )

        # Validator accepts valid ICE after M4.1
        assert bool(validate_order(ice_order, product_config, contract_open=True)) is True

        # Book also accepts
        state = book.init()
        new_state, result = book.submit_order(state, ice_order)
        assert result.accepted == True
        # Only peak_qty visible in book
        order_in_book = book.get_order(new_state, ice_order.order_id, Side.BUY)
        assert not _is_absent(order_in_book)

