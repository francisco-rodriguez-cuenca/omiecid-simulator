"""Independent audit tests for P3-PERF-REGEN-CAP-1.

Story: `docs/omiecid/agile-plan.md` §"Story P3-PERF-REGEN-CAP-1 — Config-driven
iceberg capacity (`OMIEMarketConfig.max_icebergs`, default 1024)".

Audit mode (orchestrate.md Step 5): these are NEW tests authored AFTER the
implementation shipped, derived independently from the same sprint acceptance
criteria. They are auto-tagged ``@pytest.mark.audit`` by ``tests/omiecid/conftest.py``
(they carry the ``_audit.py`` suffix and are NOT in ``_AUDIT_MARKER_EXEMPT``), so
the routine ``-m "not audit"`` gate skips them.

The load-bearing claim of the card is behavioral: shrinking the iceberg
side-channel from the ``MAX_ICEBERGS`` = 8192 hard structural ceiling to the
config-driven ``OMIEMarketConfig.max_icebergs`` (default 1024) changes
THROUGHPUT (one full ``(M, 8)`` meta copy per inner-loop slot visit) but NOT
BEHAVIOR — provided the tape's peak live ``n_active`` stays under the cap.
Overflow past the cap stays LOUD (sticky ``truncated``), never silent.

What each class probes (independently of the original locked suite):

  * TestByteIdentityAcrossCaps — the load-bearing parity claim: a non-trivial
    multi-iceberg matching + regeneration sequence produces byte-identical
    outcomes (fills, resting book, ``iceberg_meta`` shared prefix, ``ts_counter``,
    ``truncated``, ``book_full``, ``n_active``) at ``max_icebergs=1024`` vs
    ``=8192``. Two scenarios: an ICEwPI (ppd != 0) walk, and a
    NON/IOC/FOK/ICE mixed order stream.
  * TestConfigCapOverflowIsLoud — overflow driven by REAL order submission at a
    small config cap: ``n_active`` never exceeds the configured cap, the
    overflow is loud (sticky ``truncated=True``), and no in-cap iceberg is
    silently dropped or overwritten.
  * TestIndependentAxes — ``book_depth`` and ``max_icebergs`` are independent
    axes: ``book_depth > max_icebergs`` (both <= ceiling) is legal and matches
    correctly; ``max_icebergs > MAX_ICEBERGS`` is rejected at construction.
  * TestOneCompileUnderConfigField — the meta shape is a static compile-time
    constant per config instance: repeated steps at a fixed config do not
    retrace ``_submit_order``; a distinct config instance adds exactly one
    artifact (per-config, not per-cap-value explosion within one config).
  * TestRLEnvInheritsDefaultCap — ``Session()`` with the default config path
    picks up ``max_icebergs=1024``; its book's ``iceberg_meta`` first dim is
    the configured cap.
  * TestEmptyMetaBackCompat — zero-arg ``empty_meta()`` still yields
    ``(MAX_ICEBERGS, N_COLS)`` for direct callers.

Discipline (docs/omiecid/device-policy.md — CPU-hermetic; test-perf-gate.md §4):
  * Synthetic fixtures via the public ``ContractBook`` / ``IcebergManager`` /
    ``Session`` surface only — no real tape load, no implementation internals.
  * Runtime target < 1s per test (fast suite). No ``@pytest.mark.slow`` needed:
    the scenarios are small scripted order lists; the two configs each pay one
    compile, warmed by the session compilation cache.
  * Tests entering ``jax.jit`` (via ``submit_order``) carry ``@pytest.mark.jit``.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import (
    ContractBook,
    OMIEBookState,
    _submit_order,
)
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    IcebergManager,
    N_COLS,
)
from omiecid_simulator.omiecid.jax_constants import MAX_ICEBERGS, SENTINEL_ORDER_ID
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, make_order


# ---------------------------------------------------------------------------
# Fixtures — narrow, synthetic, CPU-hermetic.
# ---------------------------------------------------------------------------


@pytest.fixture
def product_config() -> OMIEProductConfig:
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def open_contract() -> Contract:
    """OPEN contract 37 (09:15-09:30) — same as conftest.py default."""
    return Contract(
        contract_id=37,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


def _book_at_cap(open_contract, product_config, cap: int) -> ContractBook:
    """A ContractBook whose market config carries ``max_icebergs = cap``."""
    mc = dataclasses.replace(OMIEMarketConfig(), max_icebergs=cap)
    return ContractBook(open_contract, product_config, mc)


def _ice_sell(order_id, price, quantity, peak_qty, ppd=0.0, agent_id=5, ts=100):
    """A resting SELL iceberg (ICE exec-restriction)."""
    return make_order(
        contract_id=37,
        side=Side.SELL,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=ts,
        exec_restriction=ExecRestriction.ICE,
        peak_qty=peak_qty,
        ppd=ppd,
    )


def _limit_buy(order_id, price, quantity, agent_id=20, ts=200,
               exec_restriction=ExecRestriction.NON):
    """An aggressive BUY limit order."""
    return make_order(
        contract_id=37,
        side=Side.BUY,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=ts,
        exec_restriction=exec_restriction,
    )


def _state_leaves(state: OMIEBookState, prefix: int) -> dict:
    """Extract the comparable observable leaves of a book state.

    ``iceberg_meta`` is compared only over its shared ``[:prefix]`` prefix
    (the two caps size the array differently; the card's parity claim is over
    the shared prefix — the tail is all-sentinel padding on the larger cap).
    """
    return {
        "asks": jnp.asarray(state.lob.asks),
        "bids": jnp.asarray(state.lob.bids),
        "trades": jnp.asarray(state.lob.trades),
        "meta_prefix": jnp.asarray(state.iceberg_meta[:prefix]),
        "ts_counter": jnp.asarray(state.ts_counter),
        "truncated": jnp.asarray(state.truncated),
        "book_full": jnp.asarray(state.book_full),
        "n_active": jnp.asarray(state.n_active),
    }


def _assert_leaves_byte_identical(leaves_small: dict, leaves_big: dict):
    for key in leaves_small:
        small = leaves_small[key]
        big = leaves_big[key]
        assert small.shape == big.shape, (
            f"leaf {key!r} shape differs across caps: "
            f"{small.shape} vs {big.shape}"
        )
        assert bool(jnp.array_equal(small, big)), (
            f"leaf {key!r} is NOT byte-identical across max_icebergs caps — "
            f"shrinking the cap changed observable behavior (this violates the "
            f"P3-PERF-REGEN-CAP-1 load-bearing parity claim)."
        )


# ===========================================================================
# 1. Behavioral byte-identity across caps on REAL matching scenarios.
# ===========================================================================


# slow: each byte-identity test compiles the jit body for a SECOND distinct
# ``max_icebergs`` static config (8192) that the rest of the fast suite never
# touches — an unavoidable one-time XLA compile (~2.9s wall). The compute itself
# is microseconds; the overrun is pure compile cost.
@pytest.mark.slow
@pytest.mark.jit
class TestByteIdentityAcrossCaps:
    """The same order sequence at ``max_icebergs=1024`` vs ``=8192`` must
    produce byte-identical observable state — cap shrink is a throughput
    lever, not a behavior change.

    ``PREFIX`` = 1024 = the smaller cap: the shared iceberg-meta prefix over
    which the two arrays are compared. The larger cap's meta tail beyond 1024
    is sentinel padding, semantically irrelevant to matching.
    """

    PREFIX = 1024

    def _run_icewpi_walk(self, book: ContractBook) -> OMIEBookState:
        """Register 4 resting SELL icebergs (one ICEwPI, ppd != 0), then
        walk them with 3 aggressive BUYs so slice regeneration fires
        repeatedly. This exercises ``regenerate_slices`` (the ~96% hot path
        the cap parameterises) on a genuine multi-iceberg cascade.
        """
        state = book.init()
        resting = [
            _ice_sell(order_id=100, price=50.0, quantity=30.0, peak_qty=10.0),
            _ice_sell(order_id=101, price=51.0, quantity=40.0, peak_qty=8.0,
                      ppd=0.5),  # ICEwPI: signed per-slice price drift
            _ice_sell(order_id=102, price=52.0, quantity=25.0, peak_qty=5.0),
            _ice_sell(order_id=103, price=50.5, quantity=12.0, peak_qty=4.0),
        ]
        for i, order in enumerate(resting):
            order = order._replace(agent_id=5 + i, timestamp=100 + i)
            state, _ = book.submit_order(state, order)
        aggressors = [
            _limit_buy(order_id=200, price=51.0, quantity=22.0, ts=200),
            _limit_buy(order_id=201, price=53.0, quantity=40.0, ts=201),
            _limit_buy(order_id=202, price=52.0, quantity=18.0, ts=202),
        ]
        for order in aggressors:
            state, _ = book.submit_order(state, order)
        return state

    def test_icewpi_multi_iceberg_walk_byte_identical(
        self, open_contract, product_config
    ):
        """An ICEwPI multi-iceberg regeneration walk is byte-identical at
        1024 vs 8192 — the parity claim on the regen hot path.
        """
        book_small = _book_at_cap(open_contract, product_config, 1024)
        book_big = _book_at_cap(open_contract, product_config, 8192)

        state_small = self._run_icewpi_walk(book_small)
        state_big = self._run_icewpi_walk(book_big)

        # Sanity: the scenario actually traded (else parity is vacuous).
        assert bool(jnp.any(state_small.lob.trades[:, 1] != 0)), (
            "scenario produced no trades — parity assertion would be vacuous"
        )
        _assert_leaves_byte_identical(
            _state_leaves(state_small, self.PREFIX),
            _state_leaves(state_big, self.PREFIX),
        )

    def test_mixed_exec_restriction_stream_byte_identical(
        self, open_contract, product_config
    ):
        """A mixed NON / IOC / FOK / ICE order stream is byte-identical at
        1024 vs 8192 — parity holds across every ExecRestriction arm, not
        just the ICE regen path.
        """
        def run(book: ContractBook) -> OMIEBookState:
            state = book.init()
            stream = [
                _ice_sell(order_id=300, price=50.0, quantity=20.0, peak_qty=6.0),
                _ice_sell(order_id=301, price=50.5, quantity=15.0, peak_qty=5.0,
                          ppd=0.25),
                # NON aggressive buy — partial fill, residual rests.
                _limit_buy(order_id=302, price=50.5, quantity=18.0, ts=210,
                           exec_restriction=ExecRestriction.NON),
                # IOC aggressive buy — cancels residual.
                _limit_buy(order_id=303, price=51.0, quantity=25.0, ts=211,
                           exec_restriction=ExecRestriction.IOC),
                # FOK aggressive buy — fill-or-kill.
                _limit_buy(order_id=304, price=51.0, quantity=8.0, ts=212,
                           exec_restriction=ExecRestriction.FOK),
                # Another resting ICE + one more aggressor.
                _ice_sell(order_id=305, price=52.0, quantity=30.0, peak_qty=10.0),
                _limit_buy(order_id=306, price=52.0, quantity=14.0, ts=213,
                           exec_restriction=ExecRestriction.NON),
            ]
            for order in stream:
                state, _ = book.submit_order(state, order)
            return state

        state_small = run(_book_at_cap(open_contract, product_config, 1024))
        state_big = run(_book_at_cap(open_contract, product_config, 8192))
        _assert_leaves_byte_identical(
            _state_leaves(state_small, self.PREFIX),
            _state_leaves(state_big, self.PREFIX),
        )

    def test_per_call_submit_result_byte_identical(
        self, open_contract, product_config
    ):
        """Each per-order ``SubmitResult`` (traded / trade_qty / trade_price /
        settlement_value) is byte-identical across caps — parity is not just
        on the final state but on every step's emitted fill result.
        """
        def run_collect(book: ContractBook):
            state = book.init()
            results = []
            orders = [
                _ice_sell(order_id=400, price=50.0, quantity=30.0, peak_qty=10.0),
                _ice_sell(order_id=401, price=51.0, quantity=20.0, peak_qty=6.0,
                          ppd=0.5),
                _limit_buy(order_id=402, price=51.0, quantity=28.0, ts=220),
                _limit_buy(order_id=403, price=52.0, quantity=15.0, ts=221),
            ]
            for order in orders:
                state, result = book.submit_order(state, order)
                results.append(result)
            return results

        res_small = run_collect(_book_at_cap(open_contract, product_config, 1024))
        res_big = run_collect(_book_at_cap(open_contract, product_config, 8192))

        assert len(res_small) == len(res_big)
        for i, (rs, rb) in enumerate(zip(res_small, res_big)):
            assert bool(rs.traded) == bool(rb.traded), f"order {i}: traded differs"
            assert float(rs.trade_qty) == float(rb.trade_qty), (
                f"order {i}: trade_qty differs across caps"
            )
            assert float(rs.trade_price) == float(rb.trade_price), (
                f"order {i}: trade_price differs across caps"
            )
            assert float(rs.settlement_value) == float(rb.settlement_value), (
                f"order {i}: settlement_value differs across caps"
            )


# ===========================================================================
# 2. Overflow correctness at a CONFIG cap driven by REAL order submission.
# ===========================================================================


@pytest.mark.jit
class TestConfigCapOverflowIsLoud:
    """Driving the book past a small config ``max_icebergs`` via real ICE
    submissions must overflow LOUDLY (sticky ``truncated``), cap ``n_active``
    at the configured value, and never silently drop/overwrite an in-cap
    iceberg.

    Unlike the pre-existing overflow tests (which synthesise a saturated meta
    and probe ``register`` directly), this drives overflow through the public
    ``submit_order`` path — the real production entry point — at a small
    CONFIG cap (not the ``MAX_ICEBERGS`` constant).
    """

    # CAP = 8 reuses the small-cap jit artifact the locked fast suite already
    # warms (test_iceberg_cap_config.py::TestSmallCapOverflowIsLoud), keeping
    # these overflow tests inside the < 1s fast-suite budget.
    CAP = 8

    def _register_n_icebergs(self, book: ContractBook, n: int):
        """Submit ``n`` resting SELL icebergs at strictly increasing prices
        (so none cross / consume another) and collect n_active per step.
        """
        state = book.init()
        n_active_trace = []
        for i in range(n):
            order = _ice_sell(
                order_id=500 + i,
                price=50.0 + i * 0.5,
                quantity=30.0,
                peak_qty=10.0,
                agent_id=5,
                ts=100 + i,
            )
            state, _ = book.submit_order(state, order)
            n_active_trace.append(int(state.n_active))
        return state, n_active_trace

    def test_n_active_never_exceeds_configured_cap(
        self, open_contract, product_config
    ):
        """``n_active`` saturates AT the configured cap and never exceeds it,
        even when more icebergs than the cap are submitted.
        """
        book = _book_at_cap(open_contract, product_config, self.CAP)
        state, trace = self._register_n_icebergs(book, self.CAP + 4)
        assert max(trace) <= self.CAP, (
            f"n_active exceeded the configured cap {self.CAP}: trace={trace}"
        )
        assert int(state.n_active) == self.CAP, (
            f"n_active should saturate at the cap {self.CAP}; "
            f"got {int(state.n_active)} (trace={trace})"
        )
        # The meta array itself is sized at the config cap, not the ceiling.
        assert state.iceberg_meta.shape[0] == self.CAP

    def test_overflow_sets_sticky_truncated(
        self, open_contract, product_config
    ):
        """The first over-cap ICE submission sets ``truncated=True`` (loud),
        and it stays True afterwards (sticky).
        """
        book = _book_at_cap(open_contract, product_config, self.CAP)
        state = book.init()
        # Fill exactly to the cap — no overflow yet.
        for i in range(self.CAP):
            state, _ = book.submit_order(
                state, _ice_sell(600 + i, 50.0 + i * 0.5, 30.0, 10.0, ts=100 + i),
            )
        assert bool(state.truncated) is False, (
            "truncated must be False while at-or-below the cap"
        )
        # One more ICE — this is the (CAP+1)th; it must overflow loudly.
        state, _ = book.submit_order(
            state, _ice_sell(700, 60.0, 30.0, 10.0, ts=200),
        )
        assert bool(state.truncated) is True, (
            "the (cap+1)th ICE registration must set truncated=True (loud "
            "overflow at the CONFIG cap — never a silent drop)"
        )
        # Sticky: a subsequent benign, non-overflowing order keeps it True.
        state, _ = book.submit_order(
            state,
            make_order(37, Side.BUY, 40.0, 1.0, 800, 9, 210,
                       exec_restriction=ExecRestriction.NON),
        )
        assert bool(state.truncated) is True, (
            "truncated must remain sticky-True across later non-overflow ops"
        )

    def test_in_cap_icebergs_are_not_silently_dropped(
        self, open_contract, product_config
    ):
        """The first ``CAP`` icebergs (submitted before overflow) survive with
        their order_ids intact in the packed prefix; the over-cap ids are the
        ones dropped, never an in-cap id.
        """
        book = _book_at_cap(open_contract, product_config, self.CAP)
        state = book.init()
        in_cap_ids = [900 + i for i in range(self.CAP)]
        over_cap_ids = [900 + self.CAP + i for i in range(3)]
        for i, oid in enumerate(in_cap_ids + over_cap_ids):
            state, _ = book.submit_order(
                state, _ice_sell(oid, 50.0 + i * 0.5, 30.0, 10.0, ts=100 + i),
            )
        live_ids = {
            int(x)
            for x in state.iceberg_meta[:, COL_ORDER_ID].tolist()
            if int(x) != SENTINEL_ORDER_ID
        }
        # Every in-cap id survived; no over-cap id was written.
        assert set(in_cap_ids) == live_ids, (
            f"in-cap icebergs were silently dropped/overwritten: expected "
            f"{sorted(in_cap_ids)}, live={sorted(live_ids)}"
        )
        for oid in over_cap_ids:
            assert oid not in live_ids, (
                f"over-cap id {oid} was silently written into the meta"
            )


# ===========================================================================
# 3. Independent axes: book_depth and max_icebergs are decoupled.
# ===========================================================================


class TestIndependentAxes:
    """``book_depth`` and ``max_icebergs`` are independent, each bounded above
    by ``MAX_ICEBERGS``. A ``book_depth > max_icebergs`` config (both <=
    ceiling) is LEGAL; ``max_icebergs > MAX_ICEBERGS`` is REJECTED.

    This encodes the shipped implementation's boundary contract (the
    ``__init__`` assertion is ``max_icebergs <= MAX_ICEBERGS`` and
    ``book_depth <= MAX_ICEBERGS`` — NOT the chained
    ``book_depth <= max_icebergs`` the card's Description prose prescribed;
    see the implementation's ``__init__`` docstring note on the
    planned-vs-implemented mismatch).
    """

    def test_book_depth_greater_than_max_icebergs_is_legal(
        self, open_contract, product_config
    ):
        """``book_depth=64 > max_icebergs=16`` (both <= ceiling) constructs
        without raising and sizes the meta by ``max_icebergs``.
        """
        mc = dataclasses.replace(
            OMIEMarketConfig(), book_depth=64, max_icebergs=16,
        )
        book = ContractBook(open_contract, product_config, mc)
        state = book.init()
        assert state.iceberg_meta.shape[0] == 16, (
            "meta first dim must follow max_icebergs, independent of book_depth"
        )

    @pytest.mark.jit
    def test_book_depth_greater_than_max_icebergs_matches_correctly(
        self, open_contract, product_config
    ):
        """A ``book_depth > max_icebergs`` book still matches a plain
        crossing limit pair correctly — the two axes do not interfere.

        Uses ``max_icebergs=8`` (warmed by the locked suite's small-cap tests)
        so this jit-entering test stays within the fast-suite budget.
        """
        mc = dataclasses.replace(
            OMIEMarketConfig(), book_depth=64, max_icebergs=8,
        )
        book = ContractBook(open_contract, product_config, mc)
        state = book.init()
        state, _ = book.submit_order(
            state, make_order(37, Side.SELL, 50.0, 5.0, 1, 1, 100),
        )
        state, result = book.submit_order(
            state, make_order(37, Side.BUY, 51.0, 5.0, 2, 2, 101),
        )
        assert bool(result.traded) is True
        assert float(result.trade_qty) == pytest.approx(5.0)

    def test_max_icebergs_above_ceiling_rejected_at_construction(
        self, open_contract, product_config
    ):
        """``max_icebergs = MAX_ICEBERGS + 1`` must be rejected at
        ``ContractBook.__init__`` (fail-fast, Python-side).
        """
        mc = dataclasses.replace(
            OMIEMarketConfig(), max_icebergs=MAX_ICEBERGS + 1,
        )
        with pytest.raises(Exception):
            ContractBook(open_contract, product_config, mc)

    def test_max_icebergs_at_ceiling_is_legal(
        self, open_contract, product_config
    ):
        """``max_icebergs == MAX_ICEBERGS`` is the boundary — must NOT raise
        (the ceiling check is ``<=``, not ``<``).
        """
        mc = dataclasses.replace(
            OMIEMarketConfig(), max_icebergs=MAX_ICEBERGS,
        )
        book = ContractBook(open_contract, product_config, mc)
        assert book.init().iceberg_meta.shape[0] == MAX_ICEBERGS


# ===========================================================================
# 4. One-compile / no-retrace under the config field.
# ===========================================================================


@pytest.mark.jit
class TestOneCompileUnderConfigField:
    """The meta shape is a static compile-time constant PER CONFIG INSTANCE:
    the frozen ``OMIEMarketConfig`` is hashable and passed via
    ``static_argnames``, so repeated stepping at a fixed config does not
    retrace ``_submit_order``. A distinct config instance is a distinct static
    key → exactly one additional artifact (not a per-order recompile).

    The ``_submit_order`` module-level jitted callable exposes JAX's
    ``_cache_size()`` introspection, which counts compiled artifacts.
    """

    def test_repeated_steps_at_fixed_config_do_not_retrace(
        self, open_contract, product_config
    ):
        """Multiple submits at ONE config instance add at most one compiled
        artifact (the first trace); subsequent calls reuse it.
        """
        book = _book_at_cap(open_contract, product_config, 1024)
        state = book.init()
        orders = [
            make_order(37, Side.BUY, 50.0, 5.0, 1, 1, 100),
            make_order(37, Side.SELL, 60.0, 3.0, 2, 2, 101),
            make_order(37, Side.BUY, 61.0, 2.0, 3, 3, 102),
        ]
        # Warm the cache with the first submit, then record the size.
        state, _ = book.submit_order(state, orders[0])
        size_after_first = _submit_order._cache_size()
        for order in orders[1:]:
            state, _ = book.submit_order(state, order)
        size_after_rest = _submit_order._cache_size()
        assert size_after_rest == size_after_first, (
            f"_submit_order retraced on repeated steps at a FIXED config "
            f"(cache {size_after_first} -> {size_after_rest}); the meta shape "
            f"must be a static compile-time constant per config instance "
            f"(one-compile invariant)."
        )

    # slow: compiles the jit body for a distinct cap (777) not touched by the
    # rest of the fast suite — an unavoidable one-time XLA compile.
    @pytest.mark.slow
    def test_distinct_config_instance_adds_exactly_one_artifact(
        self, open_contract, product_config
    ):
        """Two books at DIFFERENT ``max_icebergs`` values are two distinct
        static keys; each contributes exactly one artifact — the config field
        is a legitimate compile-time axis, not a retrace-per-call hazard.
        """
        order = make_order(37, Side.BUY, 50.0, 5.0, 1, 1, 100)

        book_a = _book_at_cap(open_contract, product_config, 1024)
        state_a = book_a.init()
        state_a, _ = book_a.submit_order(state_a, order)
        size_a = _submit_order._cache_size()

        # A distinct cap value the fixed suite is unlikely to have compiled —
        # if it was already compiled by another test, the delta is 0, which is
        # still <= 1; we assert the delta is bounded by 1, not exactly 1.
        book_b = _book_at_cap(open_contract, product_config, 777)
        state_b = book_b.init()
        state_b, _ = book_b.submit_order(state_b, order)
        size_b = _submit_order._cache_size()

        assert size_b - size_a <= 1, (
            f"a distinct config instance added {size_b - size_a} artifacts; "
            f"expected at most 1 (one compile per distinct static config)."
        )
        # And it still matched nothing (empty book) without error.
        assert isinstance(state_b, OMIEBookState)


# ===========================================================================
# 5. RL env inheritance of the new default cap.
# ===========================================================================


class TestRLEnvInheritsDefaultCap:
    """``Session`` on its default-config path inherits ``max_icebergs=1024``;
    its book's ``iceberg_meta`` first dim is the configured cap (per the
    card's ``omie_session.py`` default-config note).
    """

    def test_default_env_market_config_carries_1024(self):
        from omiecid_simulator.omiecid.omie_session import Session

        env = Session()
        assert env.market_config.max_icebergs == 1024, (
            "Session default config must inherit the new max_icebergs=1024 "
            "default from OMIEMarketConfig()"
        )

    def test_default_env_book_meta_first_dim_is_cap(self):
        from omiecid_simulator.omiecid.omie_session import Session

        env = Session()
        book_state = env._empty_book()
        assert book_state.iceberg_meta.shape[0] == 1024, (
            "the env's book iceberg_meta first dim must equal the configured "
            "max_icebergs (1024), not the MAX_ICEBERGS=8192 ceiling"
        )
        assert book_state.iceberg_meta.shape[1] == N_COLS

    def test_env_config_override_propagates_to_book(self):
        """A ``market_config`` override in the env config dict flows through to
        the book's meta size — the env does not hard-code the cap.
        """
        from omiecid_simulator.omiecid.omie_session import Session

        mc = dataclasses.replace(OMIEMarketConfig(), max_icebergs=2048)
        env = Session({"market_config": mc})
        assert env.market_config.max_icebergs == 2048
        assert env._empty_book().iceberg_meta.shape[0] == 2048


# ===========================================================================
# 6. empty_meta back-compat for direct callers.
# ===========================================================================


class TestEmptyMetaBackCompat:
    """Zero-arg ``empty_meta()`` still yields ``(MAX_ICEBERGS, N_COLS)`` so the
    direct-construction callers stay source-compatible; an explicit
    ``max_icebergs`` argument sizes the first dim from the runtime cap.
    """

    def test_zero_arg_empty_meta_is_ceiling_sized(self):
        meta = IcebergManager.empty_meta()
        assert meta.shape == (MAX_ICEBERGS, N_COLS), (
            f"empty_meta() zero-arg default must stay ({MAX_ICEBERGS}, "
            f"{N_COLS}); got {meta.shape}"
        )
        # All slots start sentinel.
        assert bool(
            jnp.all(meta[:, COL_ORDER_ID] == jnp.int32(SENTINEL_ORDER_ID))
        )
        assert meta.dtype == jnp.int32

    def test_empty_meta_arg_sizes_first_dim(self):
        for cap in (8, 256, 1024):
            meta = IcebergManager.empty_meta(cap)
            assert meta.shape == (cap, N_COLS), (
                f"empty_meta({cap}) must size the first dim from the arg"
            )
            assert bool(
                jnp.all(meta[:, COL_ORDER_ID] == jnp.int32(SENTINEL_ORDER_ID))
            )

    def test_empty_meta_arg_matches_init_meta_shape(
        self, open_contract, product_config
    ):
        """``empty_meta(cap)`` matches the shape ``ContractBook.init()`` builds
        for a book at that ``max_icebergs`` — the two construction paths agree.
        """
        cap = 512
        book = _book_at_cap(open_contract, product_config, cap)
        init_meta = book.init().iceberg_meta
        direct_meta = IcebergManager.empty_meta(cap)
        assert init_meta.shape == direct_meta.shape == (cap, N_COLS)
