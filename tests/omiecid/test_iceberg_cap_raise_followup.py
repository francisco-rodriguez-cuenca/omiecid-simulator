"""Locked behavioral-contract tests for story
``P2-PERF-7`` — Re-raise ``MAX_ICEBERGS`` to 8192.

Recovery + update note (2026-05-21)
===================================

This file was originally authored for the retired
``P2-HOTFIX-iceberg-cap-overflow-FOLLOWUP`` story
(commit ``0a89cfa`` on its branch, which was never merged to main; the
parent story was retired on 2026-05-19). The file is recovered here for
P2-PERF-7, which re-targets the cap raise at the value ``8192``
(matching the orchestrator-sanctioned set ``{2048, 4096, 8192}``
inherited from the retired story) after P2-PERF-6 demonstrated that
``regenerate_slices`` / ``matchable_hidden_volume`` / ``truncation_flag``
runtimes are independent of the cap value.

Recovery dispositions vs. the original ``0a89cfa`` file (see test-agent
session note in ``docs/omiecid/sprint-status.md`` §"P2-PERF-7 — Test
Authoring"):
  * **Item (1)** (faithful ``_hidden_total`` on canonical 2026-01-12 cid=81
    warm-up replay) — **DROPPED**. Tangential to P2-PERF-7's narrow
    deliverable (one-line cap bump). Depends on a 988-call replay against
    raw OMIE data and the demo HTML oracle, neither of which is in the
    P2-PERF-7 card's deliverables row.
  * **Item (2)** (demo HTML diverges from pre-FOLLOWUP fingerprint) —
    **DROPPED**. P2-PERF-7 does not commit to regenerating the demo HTML.
  * **Item (3)** (``MAX_ICEBERGS`` >= 2048 + sanctioned set membership)
    — **KEPT & UPDATED**. The original tests stay in spirit; widened to
    ``{100, 2048, 4096, 8192}`` for the in-flight window so that they
    are backwards-compatible with the pre-developer state (cap=100),
    and a separate test pins the exact P2-PERF-7 target value
    ``MAX_ICEBERGS == 8192`` (will FAIL pre-developer, PASS post-developer).
  * **Item (4)** (no sticky ``truncated`` on canonical warm-up replay) —
    **DROPPED**. Same reason as items (1) and (2).
  * **Item (5)** (``jax_constants.py`` + ``contract_book.py`` no longer
    name the retired FOLLOWUP story-ID literal) — **DROPPED**. The
    docstring / assertion-message cleanup is not in the P2-PERF-7 card's
    deliverables row; tests would fail post-developer because the
    developer's deliverable is only the cap bump.
  * **Item (6)** (primary hotfix contract intact: register tuple-return,
    no-eviction, ``book_depth <= MAX_ICEBERGS`` ``__init__`` assertion)
    — **KEPT & UPDATED**. The ``register`` signature changed from
    ``register(meta, order) -> (meta, overflow)`` (2-tuple, pre-P2-PERF-6)
    to ``register(meta, n_active, order) -> (meta, n_active, overflow)``
    (3-tuple, post-P2-PERF-6). All test bodies are updated to the new
    canonical signature. The saturated-meta fixture now also threads
    ``n_active = MAX_ICEBERGS`` through the call.

Test-first contract items retained / introduced (mapping to the P2-PERF-7
v5.31 agile-plan card):

  (3a) **``MAX_ICEBERGS`` >= 2048**. Pre-developer: FAILS (cap=100).
       Post-developer: PASSES (cap=8192).
  (3b) **``MAX_ICEBERGS`` in {2048, 4096, 8192}** — sanctioned set
       inherited from the retired FOLLOWUP story (widened on 2026-05-15
       after the multi-date scan surfaced session-max of 5007).
       Pre-developer: FAILS (cap=100). Post-developer: PASSES.
  (3c) **``MAX_ICEBERGS == 8192``** — the v5.31 P2-PERF-7 pinned target
       value. Pre-developer: FAILS. Post-developer: PASSES.
  (6) **Primary hotfix contract intact post cap raise.** The
      mechanism-locked invariants from
      ``P2-HOTFIX-iceberg-cap-overflow`` (now baked into P2-PERF-6's
      packed-array refactor) remain in force at the new cap. These
      probe symbolically via ``MAX_ICEBERGS``, so they re-saturate at
      whatever value the developer pins (100, 2048, 4096, or 8192).
      GREEN both pre- and post-developer.

Architectural references (read at authoring time):
  - ``docs/omiecid/agile-plan.md`` §"Story P2-PERF-7" (around line 549-559).
  - ``docs/omiecid/sprint-status.md`` §"P2-HOTFIX-iceberg-cap-overflow-FOLLOWUP
    — Implementation (2026-05-15)" — the multi-date scan (105 dates)
    that surfaced the 5007 session-max justifying 8192.
  - ``omiecid_simulator/omiecid/iceberg_manager.py`` — the post-P2-PERF-6
    ``register(meta, n_active, order) -> (meta, n_active, overflow)``
    signature.
  - ``omiecid_simulator/omiecid/jax_constants.py`` §B1 — the
    ``MAX_ICEBERGS`` constant definition (line 308).

Loader-probe oracle figure pin (retired with item 1)
=====================================================

Computed at the original test-authoring time (2026-05-15) via the
loader on ``data/rawOMIE/orders_20260112.1`` filtered to cid=81: the
cumulative ICE-hidden sum over the 988-order warm-up window
``[0, 988)`` is 11893.80 MW (340 ICE rows, ``decode_qty(quantity_int)
- decode_qty(peak_qty_int)``). Retained here as historical context
for the retired items 1 and 4; not used by any retained test.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook, OMIEBookState
from omiecid_simulator.omiecid.iceberg_manager import (
    COL_ORDER_ID,
    COL_PEAK_QTY,
    COL_REMAINING_HIDDEN,
    COL_SIDE,
    IcebergManager,
    N_COLS,
)
from omiecid_simulator.omiecid.jax_constants import (
    MAX_ICEBERGS,
    SENTINEL_ORDER_ID,
)
from omiecid_simulator.omiecid.omie_config import (
    OMIEMarketConfig,
    OMIEProductConfig,
)
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    Side,
    make_order,
)


# ---------------------------------------------------------------------------
# Module-level constants.
# ---------------------------------------------------------------------------

# Canonical demo contract id (20:15-20:30 delivery), reused from the
# retired FOLLOWUP story for fixture continuity. Item (1)/(4)/(2)/(5)
# tests that referenced this cid as the canonical-session pin are
# dropped; this constant is retained only as a contract_id for the
# OPEN contract fixture below.
CANONICAL_CONTRACT_ID = 81


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def product_config() -> OMIEProductConfig:
    return OMIEProductConfig()


@pytest.fixture(scope="module")
def market_config() -> OMIEMarketConfig:
    return OMIEMarketConfig()


def _make_open_contract(cid: int) -> Contract:
    """An OPEN contract at the given cid. Mirrors the helper in the
    retired ``0a89cfa`` recovery source; we construct a minimal
    Contract directly so this test file doesn't import from
    ``demos/`` (the demo script is not on PYTHONPATH conventions)."""
    return Contract(
        contract_id=cid,
        delivery_start_minutes=20 * 60 + 15,  # cid=81 -> 20:15
        delivery_end_minutes=20 * 60 + 30,
        gct_minutes=20 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


@pytest.fixture(scope="module")
def open_contract() -> Contract:
    return _make_open_contract(CANONICAL_CONTRACT_ID)


@pytest.fixture
def book(open_contract, product_config, market_config) -> ContractBook:
    return ContractBook(open_contract, product_config, market_config)


@pytest.fixture
def empty_state(book) -> OMIEBookState:
    return book.init()


@pytest.fixture
def iceberg_mgr(book) -> IcebergManager:
    """The IcebergManager owned by the ContractBook fixture."""
    return getattr(book, "_iceberg_mgr", getattr(book, "iceberg_mgr", None))


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _make_iceberg_buy(
    order_id: int = 1,
    price: float = 50.00,
    quantity: float = 30.0,
    peak_qty: float = 10.0,
    agent_id: int = 0,
    timestamp: int = 100,
    ppd: float = 0.0,
    contract_id: int = CANONICAL_CONTRACT_ID,
):
    return make_order(
        contract_id=contract_id,
        side=Side.BUY,
        price=price,
        quantity=quantity,
        order_id=order_id,
        agent_id=agent_id,
        timestamp=timestamp,
        exec_restriction=ExecRestriction.ICE,
        peak_qty=peak_qty,
        ppd=ppd,
    )


def _saturated_meta(start_oid: int = 1000) -> jnp.ndarray:
    """Build a saturated ``iceberg_meta`` at the current ``MAX_ICEBERGS``.

    Re-saturates automatically at any post-fix value (2048, 4096, 8192)
    because it references ``MAX_ICEBERGS`` symbolically. Used in
    conjunction with ``n_active = MAX_ICEBERGS`` to drive the
    P2-PERF-6 packed-array ``register`` into its overflow path.
    """
    meta = jnp.zeros((MAX_ICEBERGS, N_COLS), dtype=jnp.int32)
    oids = jnp.arange(
        start_oid, start_oid + MAX_ICEBERGS, dtype=jnp.int32
    )
    meta = meta.at[:, COL_ORDER_ID].set(oids)
    meta = meta.at[:, COL_PEAK_QTY].set(jnp.int32(100))
    meta = meta.at[:, COL_REMAINING_HIDDEN].set(jnp.int32(200))
    meta = meta.at[:, COL_SIDE].set(
        (jnp.arange(MAX_ICEBERGS, dtype=jnp.int32) % jnp.int32(2))
    )
    return meta


# ===========================================================================
# Item (3) — MAX_ICEBERGS sizing tests.
# ===========================================================================


class TestMaxIcebergsCapValue:
    """The three test-first contract items pinning the cap value at
    P2-PERF-7. All three FAIL pre-developer (``MAX_ICEBERGS == 100`` on
    branch HEAD) and PASS post-developer once the cap is bumped to 8192.

    Acceptance-criteria mapping:
      * ``test_max_icebergs_at_or_above_2048`` — pins the floor at 2048,
        matching the retired FOLLOWUP story's original criterion (3).
      * ``test_max_icebergs_in_sanctioned_set`` — pins membership in the
        sanctioned set ``{2048, 4096, 8192}``, matching the
        orchestrator-amended set (2026-05-15) recovered from
        commit ``0a89cfa``.
      * ``test_max_icebergs_is_8192`` — pins the exact P2-PERF-7 target
        value of 8192 (v5.31 agile-plan card).
    """

    def test_max_icebergs_at_or_above_2048(self):
        """(3a) ``MAX_ICEBERGS`` is >= 2048 per the P2-PERF-7 cap raise.

        Pre-developer (``MAX_ICEBERGS == 100``): FAILS.
        Post-developer (``MAX_ICEBERGS == 8192``): PASSES.

        Lower bound 2048 is inherited from the retired FOLLOWUP story
        criterion (3); the P2-PERF-7 card narrows the target to 8192
        while keeping the lower-bound contract as the minimum
        post-merge acceptance.
        """
        assert MAX_ICEBERGS >= 2048, (
            f"MAX_ICEBERGS ({MAX_ICEBERGS}) must be >= 2048 per the "
            "P2-PERF-7 v5.31 cap-raise card. Current value indicates "
            "the cap raise has not landed."
        )

    def test_max_icebergs_in_sanctioned_set(self):
        """(3b) ``MAX_ICEBERGS`` is one of the three sanctioned values
        ``{2048, 4096, 8192}``.

        Prevents silent drift to an unsanctioned cap (e.g. 1500, 2000,
        16384, etc.). The orchestrator-sanctioned set was widened on
        2026-05-15 to ``{2048, 4096, 8192}`` (originally ``{2048, 4096}``)
        after the developer's multi-date scan surfaced four dates whose
        session-max ICE-with-hidden submissions exceed 4096
        (top-1: 5007 on 2025-10-18 cid=62).

        State-cost at 8192: ``MAX_ICEBERGS × N_COLS × sizeof(int32)``
        = 256 KB per ``OMIEBookState`` (vs. ~3.2 KB at 100).

        Pre-developer (``MAX_ICEBERGS == 100``): FAILS.
        Post-developer (``MAX_ICEBERGS == 8192``): PASSES.
        """
        assert MAX_ICEBERGS in (2048, 4096, 8192), (
            f"MAX_ICEBERGS ({MAX_ICEBERGS}) must be one of the three "
            "sanctioned values {2048, 4096, 8192} per the "
            "orchestrator's 2026-05-15 amendment recovered from commit "
            "0a89cfa. Any other value indicates an unauthorised cap "
            "choice."
        )

    def test_max_icebergs_is_8192(self):
        """(3c) ``MAX_ICEBERGS == 8192`` — the P2-PERF-7 v5.31 pinned
        target value.

        Justification (from the v5.31 agile-plan card and the
        2026-05-15 multi-date scan):
          * Multi-date scan top-1: 5007 ICE-with-hidden submissions
            (2025-10-18 cid=62).
          * 8192 is the next power of 2 above 5007 (~1.6x headroom).
          * State-cost: 256 KB per ``OMIEBookState`` (acceptable per
            P2-PERF-6 perf gate; ``regenerate_slices`` is now
            cap-independent in cost).

        Pre-developer (``MAX_ICEBERGS == 100``): FAILS.
        Post-developer (``MAX_ICEBERGS == 8192``): PASSES.
        """
        assert MAX_ICEBERGS == 8192, (
            f"MAX_ICEBERGS ({MAX_ICEBERGS}) must equal 8192 per the "
            "P2-PERF-7 v5.31 pinned target value. The developer's "
            "deliverable is exactly one line: "
            "``omiecid_simulator/omiecid/jax_constants.py:308`` "
            "from ``MAX_ICEBERGS = 100`` to ``MAX_ICEBERGS = 8192``."
        )


# ===========================================================================
# Item (6) — Primary hotfix contract intact (updated for P2-PERF-6 surface).
# ===========================================================================


class TestPrimaryHotfixContractIntact:
    """(6) The mechanism-locked invariants from the primary
    P2-HOTFIX-iceberg-cap-overflow story (now baked into P2-PERF-6's
    packed-array refactor) remain in force after the cap raise.

    These tests probe symbolically via ``MAX_ICEBERGS`` so the
    saturated-meta fixture and ``+1``-registration probe re-saturate
    automatically at the new cap. They are GREEN on HEAD (where the
    primary hotfix + P2-PERF-6 are already merged) and STAY GREEN
    post-fix (P2-PERF-7's cap raise does not change any signature; it
    only changes the value of ``MAX_ICEBERGS``).

    Updated for P2-PERF-6 (2026-05-20):
      The original ``0a89cfa`` tests called
      ``register(meta, order) -> (meta, overflow)`` (the pre-P2-PERF-6
      2-tuple signature). P2-PERF-6 introduced the packed-array
      mechanism (2a) and changed the signature to
      ``register(meta, n_active, order) -> (meta, n_active, overflow)``
      (3-tuple). All call sites here use the post-P2-PERF-6 form.

    Coverage spans the three binding invariants of the primary hotfix:
      (a) ``register`` 3-tuple return with ``overflow=False`` on a
          non-saturated meta.
      (b) ``register`` 3-tuple return with ``overflow=True`` on a
          saturated meta — no-eviction property.
      (c) ``ContractBook.__init__`` raises when
          ``market_config.book_depth > MAX_ICEBERGS``, and does NOT
          raise at the boundary ``book_depth == MAX_ICEBERGS``.

    For depth, see ``tests/omiecid/test_iceberg_cap_overflow.py`` — those
    tests are the canonical lock and must also remain green. This
    suite is the cap-raise-side mirror that re-asserts the invariants
    at the new cap.
    """

    def test_register_returns_3_tuple_with_overflow_false_on_empty_meta(
        self, iceberg_mgr, product_config
    ):
        """``register`` returns ``(updated_meta, new_n_active, overflow)``
        and ``overflow=False`` when the meta has free slots (post-
        P2-PERF-6 canonical 3-tuple signature).

        Probes the tuple-return signature post-primary-hotfix +
        P2-PERF-6 packed mechanism. The signature is the binding
        mechanism; the cap raise does not change it.
        """
        empty_meta = IcebergManager.empty_meta()
        n_active = jnp.int32(0)
        order = _make_iceberg_buy(order_id=42)
        out = iceberg_mgr.register(empty_meta, n_active, order)

        assert isinstance(out, tuple) and len(out) == 3, (
            "register must return (updated_meta, new_n_active, overflow); "
            "the post-P2-PERF-6 mechanism-locked signature."
        )
        meta_post, n_active_post, overflow = out
        assert bool(overflow) is False, (
            "overflow must be False when register lands into a meta "
            "with free slots."
        )
        assert int(n_active_post) == 1, (
            "n_active must increment by 1 on a successful (non-overflow) "
            "registration."
        )
        # The new order_id appears in meta_post:
        assert int(42) in meta_post[:, COL_ORDER_ID].tolist(), (
            "Newly-registered order_id must appear in the returned meta."
        )

    def test_register_returns_overflow_true_on_saturated_meta(
        self, iceberg_mgr
    ):
        """At saturation, ``register`` returns ``overflow=True`` and the
        new order_id is NOT written.

        Re-saturates at the current ``MAX_ICEBERGS`` (whatever its
        value: 100 on HEAD, 8192 post-fix). The probe is symbolic — it
        does not hard-code the cap.
        """
        meta_pre = _saturated_meta(start_oid=1000)
        n_active = jnp.int32(MAX_ICEBERGS)
        new_oid = 1000 + MAX_ICEBERGS  # one past the saturation horizon
        order = _make_iceberg_buy(order_id=new_oid)

        out = iceberg_mgr.register(meta_pre, n_active, order)
        assert isinstance(out, tuple) and len(out) == 3
        meta_post, n_active_post, overflow = out

        assert bool(overflow) is True, (
            "overflow must be True when register lands into a "
            "MAX_ICEBERGS-saturated meta."
        )
        assert int(n_active_post) == int(MAX_ICEBERGS), (
            "n_active must NOT increment on overflow — saturated "
            "registers are silent no-ops on n_active."
        )
        assert new_oid not in meta_post[:, COL_ORDER_ID].tolist(), (
            "no-eviction property: the overflowing register call must "
            "NOT write the new order_id into the saturated meta. The "
            "binding contract from the primary hotfix is 'register "
            "drops the MAX_ICEBERGS+1-th iceberg silently' — pre-fix "
            "it overwrote slot 0; the cap raise does not change this."
        )

    def test_no_eviction_cardinality_preserved_at_saturation(
        self, iceberg_mgr
    ):
        """At saturation, the non-sentinel cardinality of ``meta_post``
        equals ``MAX_ICEBERGS``.

        Pins the primary hotfix's no-eviction property symbolically:
        no pre-existing iceberg was overwritten by the saturating
        registration. Threads the post-P2-PERF-6 ``n_active`` argument
        through the 3-tuple-returning ``register`` call.
        """
        meta_pre = _saturated_meta(start_oid=1000)
        n_active = jnp.int32(MAX_ICEBERGS)
        new_oid = 1000 + MAX_ICEBERGS
        order = _make_iceberg_buy(order_id=new_oid)
        meta_post, _n_active_post, _overflow = iceberg_mgr.register(
            meta_pre, n_active, order
        )

        non_sentinel_count = int(
            jnp.sum(meta_post[:, COL_ORDER_ID] != jnp.int32(SENTINEL_ORDER_ID))
        )
        assert non_sentinel_count == MAX_ICEBERGS, (
            f"saturated register must preserve cardinality; got "
            f"{non_sentinel_count}, expected {MAX_ICEBERGS}."
        )

    def test_book_depth_assertion_fires_above_max_icebergs(
        self, open_contract, product_config
    ):
        """``ContractBook.__init__`` raises when ``book_depth >
        MAX_ICEBERGS`` — the assertion from the primary hotfix.

        The cap raise does NOT remove this assertion; it just shifts
        the threshold from 100 to ``MAX_ICEBERGS`` (whatever post-fix
        value lands: 8192 per the v5.31 card).
        """
        bad_market_config = dataclasses.replace(
            OMIEMarketConfig(),
            book_depth=MAX_ICEBERGS + 1,
        )
        with pytest.raises(Exception):
            ContractBook(open_contract, product_config, bad_market_config)

    def test_book_depth_assertion_at_boundary(
        self, open_contract, product_config
    ):
        """``ContractBook.__init__`` does NOT raise at the boundary
        ``book_depth == MAX_ICEBERGS``. Pins the assertion is ``<=``,
        not ``<``.
        """
        ok_market_config = dataclasses.replace(
            OMIEMarketConfig(),
            book_depth=MAX_ICEBERGS,
        )
        # Should construct without raising.
        ContractBook(open_contract, product_config, ok_market_config)
