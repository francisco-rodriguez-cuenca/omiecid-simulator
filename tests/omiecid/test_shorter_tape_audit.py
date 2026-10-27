"""Audit-mode independent verification for P3-PERF-TAPE-1 —
shorter liquid-hour tape / smaller ``n_orders_padded`` (DA-TAPE-2026-07).

This is a POST-implementation audit pass (orchestrate.md Step 5). It is
context-isolated from the original locked suite (``test_shorter_tape.py``):
these tests are derived INDEPENDENTLY from the same sprint requirements
and the authoritative domain ruling in ``docs/omiecid/market-spec.md`` §14,
and probe domain correctness the original suite may not have covered.

Filename ends in ``_audit.py`` so ``tests/omiecid/conftest.py`` auto-tags every
test here with the ``audit`` marker (excluded from the routine merge-gate
surface via ``-m "not audit"``, run on demand).

What the shipped story is (audited, not assumed):
  * NEW additive config ``config/env_configs/omie_short_tape.json`` — the
    D-D reproducibility-anchor config ``omie_single_contract.json``
    (contract_id 16 / 2025-06-01..03) is NOT mutated (§14.5).
  * The shorter window is ``contract_id 0`` (Q1, delivery 00:00–00:15 CET;
    GCT 23:00 D−1) over ``2025-06-01..03`` — the strictly shortest
    GOT→GCT window (§14.2), pure contract-id selection (§14.3(1), primary
    axis), NO loader change, NO tail truncation (§14.3(2) prohibition).
  * On-disk resolution: per-day N = (2201, 1279, 740); n_orders_padded =
    max(N)+pad_extra = 2201+8 = 2209 (vs 5459 baseline); steps_per_episode
    = ceil(2209/4) = 553 (vs 1365 baseline).

Audit dimensions (mapped to §14):
  1. Additivity is real AND complete — baseline D-D block byte-for-byte
     intact (§14.5).
  2. §14.4 floor holds on EVERY shipped delivery date, per-tape (the
     original suite may only probe the first (contract, date) pair).
  3. §14.2 shortest-window property — the shipped contract_id encodes a
     genuinely short GOT→GCT window (small id ⇒ early delivery ⇒ short
     window), monotone in contract_id.
  4. The reduction is genuine on the LOADED tape — n_orders_padded /
     steps_per_episode materially below 5459/1365, AND
     n_orders_padded == max(per-tape N) + pad_extra (the loader's
     documented length rule) — the reduction comes from fewer REAL orders,
     not a mis-set pad_extra.
  5. §14.3 tail-truncation NOT used — pure contract-id selection carries no
     tail-window key; if any cap key were present it must be a head cap.
  6. Env end-to-end on the shorter tape — reset → step to ``done`` at GCT
     on EACH shipped tape (episode terminates via ``time >= GCT``, bounded
     by the reduced ``steps_per_episode``).

Test hygiene / tiering (test-perf-gate + conftest/pyproject markers):
  * Pure-JSON structural probes (dimensions 1, 3, 5) are FAST (no marker) —
    microseconds, no data access.
  * Real-data-path probes (dimensions 2, 4) read ``data/rawOMIE/`` via the
    production loader ⇒ ``@pytest.mark.slow``.
  * Env reset/step probes (dimension 6) compile + step ``Session`` ⇒
    ``@pytest.mark.slow`` AND ``@pytest.mark.jit``; the vmapped/jitted
    ``step_env`` call sites are wrapped in ``jax.experimental.enable_x64()``
    (OQ-18: an in-body ``enable_x64`` no-ops under trace).

All real-data / config-dependent tests SKIP cleanly (never error) when the
shipped config or the OMIE data files are absent, so a data-free host still
collects and runs the pure-JSON audits.

References:
    docs/omiecid/market-spec.md §14 (DA-TAPE-2026-07 ruling), §13.4 (contract_id
        mapping), §3.1 (GCT = delivery_start − 60 min).
    config/env_configs/omie_short_tape.json (shipped shorter-tape config).
    config/env_configs/omie_single_contract.json (baseline D-D anchor).
    omiecid_simulator/jaxomiecid/omie_data_loader.py (build_fixed_tape_set,
        load_omie_orders, load_omie_trades, replay_orders_through_book,
        _PAD_ORDER_ROW, the max(N_i)+pad_extra length rule).
    omiecid_simulator/omiecid/omie_session.py (Session.reset / step_env, done =
        time >= GCT).

RELEASE NOTE (REL-07): the research config files ``omie_short_tape.json`` /
``omie_single_contract.json`` are not shipped. The shorter-tape config's
session keys and data_source block are inlined (``SHORT_TAPE_INLINE``); the
three Dimension-1 tests whose subject was the baseline FILE on disk were
removed. The participant submits nothing (``AgentOrder.none()``).
"""

import copy
import math
import os

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    build_fixed_tape_set,
    load_omie_orders,
    load_omie_trades,
    replay_orders_through_book,
    _PAD_ORDER_ROW,
)
from omiecid_simulator.omiecid.contract import Contract, ContractStatus
from omiecid_simulator.omiecid.contract_book import ContractBook
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig
from omiecid_simulator.omiecid.order_types import Side


# ---------------------------------------------------------------------------
# Frozen anchors (the audit contract — independent of the original suite)
# ---------------------------------------------------------------------------

# The shorter-tape config ``config/env_configs/omie_short_tape.json`` (research,
# not shipped), inlined: its session keys and its data_source block.
SHORT_TAPE_INLINE = {
    "omie_env_config": {
        "orders_per_step": 4,
        "gct_minutes_from_got": 60,
    },
    "data_source": {
        "loader": "omie",
        "data_path": "data/rawOMIE",
        "contract_ids": [0],
        "delivery_dates": ["2025-06-01", "2025-06-02", "2025-06-03"],
        "cache_path": "saved_npz/omie/",
        "pad_extra": 8,
    },
}

# The report §2 / stage0_20260703_585f775 baseline throughput anchors that the
# shorter tape MUST materially beat (§14.5(ii); NOT contract-choice specific).
BASELINE_N_ORDERS_PADDED = 5459
BASELINE_STEPS_PER_EPISODE = 1365

# The D-D anchor values the baseline config's data_source block MUST carry
# byte-for-byte (orchestrator decision D-D). A stronger check than "≠ short".
BASELINE_CONTRACT_IDS = [16]
BASELINE_DELIVERY_DATES = ["2025-06-01", "2025-06-02", "2025-06-03"]

# Tail-window key names §14.3(2) forbids (a "final-N-minutes-before-GCT"
# truncation is NOT domain-defensible). The audit asserts NONE appear in the
# shipped data_source block. Head-cap key names §14.3(2) DOES sanction (if a
# cap is used at all it must be a head cap) are listed separately.
_TAIL_WINDOW_KEYS = frozenset({
    "tail_orders", "tail_orders_per_tape", "last_k_orders", "final_n_minutes",
    "final_minutes_before_gct", "tail_window", "tail_window_minutes",
    "last_minutes", "gct_tail_minutes", "tail_cap",
})
_HEAD_CAP_KEYS = frozenset({
    "max_orders", "max_orders_per_tape", "head_orders", "order_cap",
    "first_k_orders", "n_orders_padded",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_config():
    return copy.deepcopy(SHORT_TAPE_INLINE)


def _data_source(cfg):
    ds = cfg.get("data_source")
    assert ds is not None, "shipped config must carry a `data_source` block"
    return ds


def _orders_file_exists(data_path, delivery_date):
    date_nodash = delivery_date.replace("-", "")
    return os.path.exists(f"{data_path}/orders_{date_nodash}.1")


def _skip_if_data_absent(ds):
    """SKIP cleanly if ANY shipped delivery date's orders file is missing."""
    dp = ds["data_path"]
    for d in ds["delivery_dates"]:
        if not _orders_file_exists(dp, d):
            pytest.skip(
                f"OMIE orders file for {d} absent under {dp!r}; "
                f"real-data audit is skip-on-data-absent."
            )


def _open_contract(contract_id):
    """An OPEN Contract usable in a ContractBook for the H1 replay.

    Only OPEN status + the field presence matter for the replay; the exact
    absolute delivery minutes are irrelevant (mirrors the loader-test
    convention and Session's ``_synthetic_contract``).
    """
    return Contract(
        contract_id=contract_id,
        delivery_start_minutes=9 * 60 + 15,
        delivery_end_minutes=9 * 60 + 30,
        gct_minutes=9 * 60 + 15 - 60,
        status=ContractStatus.OPEN,
    )


def _contract_id_window_minutes(contract_id):
    """§14.2 spec-grounded GOT→GCT window length (minutes) as a pure
    function of contract_id, with NO data access.

    Per §13.4: ``contract_id = start_hour*4 + start_minute//15`` ⇒ the
    delivery-period start on day D is ``contract_id * 15`` minutes after
    00:00 D. Per §3.1 / Hard Rules §11.1: ``GCT = delivery_start − 60 min``.
    GOT is common to all day-D contracts (~15:20 CET on D−1, §2.2). Measuring
    the window relative to that common GOT origin, the window length is
    ``GCT_minutes_on_absolute_timeline − GOT_minutes`` — but since GOT is
    common the RELATIVE ordering of window lengths depends only on GCT, i.e.
    on ``delivery_start − 60 = contract_id*15 − 60`` (minutes-after-00:00 D).
    This is strictly increasing in contract_id, which is the property §14.2
    asserts and this audit verifies. The absolute GOT offset is a common
    additive constant, so we return the GCT-on-day-D offset in minutes (a
    monotone proxy for window length).
    """
    delivery_start_min = contract_id * 15  # minutes after 00:00 on day D
    gct_min = delivery_start_min - 60      # GCT = delivery_start − 1h
    return gct_min


# ===========================================================================
# Dimension 1 — Additivity is real AND complete (§14.5)  [FAST]
# ===========================================================================

class TestAdditivityBaselineDDAnchorIntact:
    """The shorter config selects a different window than the baseline D-D
    anchor (contract_id 16 / 2025-06-01..03)."""

    def test_shorter_config_selects_a_different_contract_than_baseline(self):
        """Additivity implies the shorter config points at a DIFFERENT
        contract-id window than the baseline anchor — otherwise it is not a
        distinct 'shorter' tape."""
        short = _short_config()
        ds = _data_source(short)
        assert ds["contract_ids"] != BASELINE_CONTRACT_IDS, (
            f"shorter config must select a different contract than the baseline "
            f"anchor {BASELINE_CONTRACT_IDS}; got {ds['contract_ids']}"
        )


# ===========================================================================
# Dimension 3 — §14.2 shortest-window property  [FAST, pure JSON + arithmetic]
# ===========================================================================

class TestShortestWindowProperty:
    """The shipped contract_id encodes a genuinely SHORT GOT→GCT window
    (small id ⇒ early day-D delivery ⇒ short window), consistent with the
    §14.2 shortest-window-first search — the story did NOT 'shorten' by
    picking an arbitrary long-window contract that merely happens to have
    fewer orders in this window. The relationship is expressed purely from
    §13.4 / §3.1 arithmetic, no data access."""

    def test_shipped_contract_id_is_smaller_than_baseline(self):
        """§14.2: shortest window = smallest contract_id. The shipped
        contract_id must be STRICTLY below the baseline contract_id 16."""
        short = _short_config()
        ds = _data_source(short)
        cids = ds["contract_ids"]
        assert len(cids) >= 1
        for cid in cids:
            assert cid < BASELINE_CONTRACT_IDS[0], (
                f"§14.2 shortest-window-first: shipped contract_id {cid} must be "
                f"< baseline {BASELINE_CONTRACT_IDS[0]} to have a strictly shorter "
                f"GOT→GCT window"
            )

    def test_shipped_window_length_strictly_below_baseline_window(self):
        """The §14.2 spec-grounded window-length proxy (a pure function of
        contract_id) for the shipped contract must be STRICTLY less than the
        baseline contract_id 16's window — i.e. the shortening is a genuine
        window-length reduction, not an incidental order-count one."""
        short = _short_config()
        ds = _data_source(short)
        base_window = _contract_id_window_minutes(BASELINE_CONTRACT_IDS[0])
        for cid in ds["contract_ids"]:
            w = _contract_id_window_minutes(cid)
            assert w < base_window, (
                f"§14.2: shipped contract_id {cid} window proxy ({w} min) must be "
                f"< baseline contract_id 16 window ({base_window} min)"
            )

    def test_shipped_contract_in_early_delivery_band(self):
        """§14.5 recommended search order starts from the shortest-window
        early-delivery band (small contract_id, ~0–11, the 00:00–03:00
        delivery band). The shipped contract must fall inside that early band,
        confirming the shortest-window-first search rather than a mid/late
        peak-demand hour."""
        short = _short_config()
        ds = _data_source(short)
        for cid in ds["contract_ids"]:
            assert 0 <= cid <= 11, (
                f"§14.5 shortest-window-first: shipped contract_id {cid} must lie "
                f"in the early-delivery band [0, 11] (00:00–03:00 delivery); a "
                f"larger id would be a longer window contrary to the throughput goal"
            )

    def test_window_length_proxy_is_monotone_in_contract_id(self):
        """Encode the §14.2 domain relationship directly: the GOT→GCT window
        length is a STRICTLY INCREASING function of contract_id (common GOT,
        GCT = delivery_start − 1h, delivery_start = contract_id*15 min). This
        is the property that justifies 'smaller id ⇒ shorter tape' and is the
        entire basis for the story's window choice."""
        windows = [_contract_id_window_minutes(cid) for cid in range(0, 96)]
        for i in range(1, len(windows)):
            assert windows[i] > windows[i - 1], (
                f"§14.2 window-length must be strictly increasing in contract_id; "
                f"violated at contract_id {i} ({windows[i]}) vs {i - 1} ({windows[i-1]})"
            )
        # And each quarter-hour step is exactly 15 minutes (the §13.4 grid).
        assert windows[1] - windows[0] == 15


# ===========================================================================
# Dimension 5 — §14.3 tail-truncation NOT used  [FAST, pure JSON]
# ===========================================================================

class TestNoTailTruncation:
    """§14.3(2): a tail 'final-N-minutes-before-GCT' truncation is NOT
    domain-defensible. If the shipped config uses ANY per-tape cap it must be
    a HEAD cap (first-K by submission time). The shipped story used pure
    contract-id selection (no cap), so the data_source block carries NO tail
    key (and, in fact, no cap key at all)."""

    def test_data_source_carries_no_tail_window_key(self):
        """No forbidden tail-window key may appear in the shipped data_source
        block (§14.3(2) prohibition)."""
        short = _short_config()
        ds = _data_source(short)
        present_tail = _TAIL_WINDOW_KEYS.intersection(ds.keys())
        assert not present_tail, (
            f"§14.3(2) forbids tail truncation; shipped data_source carries "
            f"tail-window key(s) {sorted(present_tail)}"
        )

    def test_any_cap_key_present_is_a_head_cap_not_tail(self):
        """If ANY order-cap axis is present, it must be a sanctioned HEAD-cap
        key (§14.3(2) permits only first-K-by-submission-time). This bites a
        future revision that introduces a tail cap under a novel name — and
        passes trivially for the shipped pure-contract-id config (no cap)."""
        short = _short_config()
        ds = _data_source(short)
        # Any key that looks like a cap/limit/truncation but is NOT a
        # sanctioned head-cap key is suspect.
        cap_like = {
            k for k in ds.keys()
            if any(tok in k.lower() for tok in ("cap", "max_orders", "_k_", "first_k",
                                                 "truncat", "head", "tail", "last", "final"))
        }
        illegal = cap_like - _HEAD_CAP_KEYS
        assert not illegal, (
            f"§14.3(2): only head caps are defensible; shipped data_source has "
            f"non-head cap-like key(s) {sorted(illegal)}"
        )

    def test_shipped_config_used_pure_contract_id_selection(self):
        """The as-shipped resolution is pure contract-id selection (§14.3(1)
        primary axis) — the data_source keys are exactly the P3.3 pointer-form
        set plus pad_extra, with NO cap axis of any kind. Documents the
        as-shipped surface so a future cap addition is a visible diff."""
        short = _short_config()
        ds = _data_source(short)
        expected_keys = {
            "loader", "data_path", "contract_ids", "delivery_dates",
            "cache_path", "pad_extra",
        }
        assert set(ds.keys()) == expected_keys, (
            f"shipped data_source keys {sorted(ds.keys())} deviate from the "
            f"pure-contract-id pointer set {sorted(expected_keys)}"
        )


# ===========================================================================
# Dimension 4 — the reduction is genuine on the LOADED tape (§14.5(ii))
# ===========================================================================

class TestReductionIsGenuine:
    """The reduction is materially below 5459/1365 AND obeys the loader's
    documented ``n_orders_padded == max(per-tape N) + pad_extra`` length rule
    — i.e. the reduction comes from fewer REAL orders, not a mis-set
    pad_extra."""

    pytestmark = pytest.mark.slow  # reads real data/rawOMIE/ via the loader

    def test_resolved_length_rule_matches_max_N_plus_pad_extra(self):
        """slow: real-data resolution — n_orders_padded == max(per-tape real
        N) + pad_extra. Loads each shipped (contract, date) tape independently,
        computes max(N_i), and asserts the stacked tape's row count equals
        max(N_i)+pad_extra (the loader's documented length rule). This catches
        a length inflated by a large pad_extra masquerading as real orders."""
        short = _short_config()
        ds = _data_source(short)
        _skip_if_data_absent(ds)

        pad_extra = int(ds["pad_extra"])
        per_tape_N = []
        for cid in ds["contract_ids"]:
            for d in ds["delivery_dates"]:
                cfg = OMIEDataConfig(
                    data_path=ds["data_path"], contract_id=cid, delivery_date=d,
                    cache_path=ds.get("cache_path", "saved_npz/omie/"),
                )
                orders = load_omie_orders(cfg)
                n = int(orders.shape[0])
                if n > 0:  # build_fixed_tape_set drops empty tapes
                    per_tape_N.append(n)

        assert per_tape_N, "no shipped (contract, date) tape loaded any orders"
        max_N = max(per_tape_N)

        tapes = build_fixed_tape_set(
            data_path=ds["data_path"], contract_ids=ds["contract_ids"],
            delivery_dates=ds["delivery_dates"],
            cache_path=ds.get("cache_path", "saved_npz/omie/"),
            pad_extra=pad_extra,
        )
        n_orders_padded = int(tapes.shape[1])
        assert n_orders_padded == max_N + pad_extra, (
            f"loader length rule violated: n_orders_padded={n_orders_padded} != "
            f"max(N_i)={max_N} + pad_extra={pad_extra}"
        )

    def test_reduction_materially_below_baseline_anchors(self):
        """slow: real-data resolution — n_orders_padded < 5459 AND
        steps_per_episode < 1365 (the fixed report-§2 baseline anchors).
        Recomputes steps_per_episode from the shipped orders_per_step so the
        assertion is on the SHIPPED config's own resolution."""
        short = _short_config()
        ds = _data_source(short)
        _skip_if_data_absent(ds)

        orders_per_step = int(short["omie_env_config"]["orders_per_step"])
        tapes = build_fixed_tape_set(
            data_path=ds["data_path"], contract_ids=ds["contract_ids"],
            delivery_dates=ds["delivery_dates"],
            cache_path=ds.get("cache_path", "saved_npz/omie/"),
            pad_extra=int(ds["pad_extra"]),
        )
        n_orders_padded = int(tapes.shape[1])
        steps_per_episode = math.ceil(n_orders_padded / orders_per_step)

        assert n_orders_padded < BASELINE_N_ORDERS_PADDED, (
            f"n_orders_padded={n_orders_padded} is NOT below the baseline "
            f"{BASELINE_N_ORDERS_PADDED}"
        )
        assert steps_per_episode < BASELINE_STEPS_PER_EPISODE, (
            f"steps_per_episode={steps_per_episode} is NOT below the baseline "
            f"{BASELINE_STEPS_PER_EPISODE}"
        )

    def test_real_orders_dominate_the_padded_length(self):
        """slow: real-data resolution — the padded length is overwhelmingly
        REAL orders, not pad. max(N_i) must be >> pad_extra, so the reduction
        is a genuine fewer-real-orders reduction (not achievable by a tiny
        real tape + a small pad). Guards against a degenerate near-empty tape
        that would 'reduce' steps by having almost no real flow."""
        short = _short_config()
        ds = _data_source(short)
        _skip_if_data_absent(ds)

        pad_extra = int(ds["pad_extra"])
        max_N = 0
        for cid in ds["contract_ids"]:
            for d in ds["delivery_dates"]:
                cfg = OMIEDataConfig(
                    data_path=ds["data_path"], contract_id=cid, delivery_date=d,
                    cache_path=ds.get("cache_path", "saved_npz/omie/"),
                )
                max_N = max(max_N, int(load_omie_orders(cfg).shape[0]))
        assert max_N > 100 * max(pad_extra, 1), (
            f"padded length is not dominated by real orders: max(N_i)={max_N} "
            f"vs pad_extra={pad_extra} — the reduction must come from real flow, "
            f"not padding geometry"
        )


# ===========================================================================
# Dimension 2 — §14.4 non-degeneracy floor holds on EVERY shipped tape
# ===========================================================================

class TestNonDegeneracyFloorPerTape:
    """§14.4 non-degeneracy floor, audited on EACH delivery date the shipped
    config lists (the original suite may probe only the first (contract, date)
    pair). For every shipped (contract, date) tape:
      (a) ≥ 1 executed historical trade in trades_YYYYMMDD.1;
      (b) both order sides present (≥1 BUY, ≥1 SELL);
      (c) the deterministic H1 ``rest_until_gct`` production replay produces a
          non-zero simulated fill count (the operational form of §14.4(c),
          the same _replay_orders path Session folds the tape through)."""

    pytestmark = pytest.mark.slow  # reads real data + runs the H1 replay

    def _iter_shipped_pairs(self):
        short = _short_config()
        ds = _data_source(short)
        _skip_if_data_absent(ds)
        for cid in ds["contract_ids"]:
            for d in ds["delivery_dates"]:
                yield ds, cid, d

    def test_every_tape_has_at_least_one_historical_trade(self):
        """§14.4(a): each shipped (contract, date) has ≥1 executed historical
        trade — the historical book demonstrably crossed at least once on
        EVERY date, not just the first."""
        checked = 0
        for ds, cid, d in self._iter_shipped_pairs():
            cfg = OMIEDataConfig(
                data_path=ds["data_path"], contract_id=cid, delivery_date=d,
                cache_path=ds.get("cache_path", "saved_npz/omie/"),
            )
            trades = load_omie_trades(cfg)
            assert int(trades.shape[0]) >= 1, (
                f"§14.4(a) floor: contract_id {cid} / {d} has "
                f"{int(trades.shape[0])} historical trades (< 1); tape is degenerate"
            )
            checked += 1
        assert checked >= 1, "no shipped (contract, date) pair was checked"

    def test_every_tape_has_both_order_sides(self):
        """§14.4(b): each shipped tape carries ≥1 BUY and ≥1 SELL order, so a
        crossing is structurally possible on EVERY date."""
        for ds, cid, d in self._iter_shipped_pairs():
            cfg = OMIEDataConfig(
                data_path=ds["data_path"], contract_id=cid, delivery_date=d,
                cache_path=ds.get("cache_path", "saved_npz/omie/"),
            )
            orders = load_omie_orders(cfg)
            n = int(orders.shape[0])
            if n == 0:
                # build_fixed_tape_set drops empty tapes; a genuinely empty
                # date is not part of the resolved set, so skip it here.
                continue
            n_buy = int(jnp.sum(orders[:, 1] == int(Side.BUY)))
            n_sell = int(jnp.sum(orders[:, 1] == int(Side.SELL)))
            assert n_buy >= 1 and n_sell >= 1, (
                f"§14.4(b) floor: contract_id {cid} / {d} has {n_buy} BUY / "
                f"{n_sell} SELL orders; both sides must be present"
            )

    def test_every_tape_produces_nonzero_simulated_fills(self):
        """§14.4(c): the deterministic H1 ``rest_until_gct`` production replay
        of each shipped tape's REAL orders through a ContractBook produces a
        non-zero simulated fill count (``replayed_trades`` row count ≥ 1) —
        confirming the book crosses in the simulator and presents real,
        agent-reachable resting liquidity on EVERY shipped date."""
        pc, mc = OMIEProductConfig(), OMIEMarketConfig()
        for ds, cid, d in self._iter_shipped_pairs():
            cfg = OMIEDataConfig(
                data_path=ds["data_path"], contract_id=cid, delivery_date=d,
                cache_path=ds.get("cache_path", "saved_npz/omie/"),
            )
            orders = load_omie_orders(cfg)
            if int(orders.shape[0]) == 0:
                continue
            trades = load_omie_trades(cfg)
            book = ContractBook(_open_contract(cid), pc, mc)
            result = replay_orders_through_book(orders, book, trades)
            n_fills = int(result.replayed_trades.shape[0])
            assert n_fills >= 1, (
                f"§14.4(c) floor: contract_id {cid} / {d} H1 replay produced "
                f"{n_fills} simulated fills (< 1); tape is degenerate in the sim"
            )


# ===========================================================================
# Dimension 6 — Env end-to-end on the shorter tape  [slow + jit]
# ===========================================================================

class TestEnvEndToEndOnShorterTape:
    """reset → step to ``done`` at GCT on the shipped config's tapes. Audits
    that (i) the env consumes the shipped tape set at the reduced
    n_orders_padded / steps_per_episode, and (ii) each tape terminates via
    ``done`` at ``time >= GCT``, with the step count BOUNDED by the reduced
    ``steps_per_episode`` (the episode ends at GCT — the interleave scales the
    tape's last-order timestamp onto [0, GCT] — so termination can occur at
    or before steps_per_episode, never beyond it)."""

    # slow: real-data resolution + Session compile/step. jit: compiles &
    # steps the jitted Session; call sites wrapped in enable_x64() (OQ-18).
    pytestmark = [pytest.mark.slow, pytest.mark.jit]

    def _build_env(self):
        from omiecid_simulator.omiecid.omie_session import Session
        short = _short_config()
        ds = _data_source(short)
        _skip_if_data_absent(ds)
        tapes = build_fixed_tape_set(
            data_path=ds["data_path"], contract_ids=ds["contract_ids"],
            delivery_dates=ds["delivery_dates"],
            cache_path=ds.get("cache_path", "saved_npz/omie/"),
            pad_extra=int(ds["pad_extra"]),
        )
        env_cfg = dict(short["omie_env_config"])
        env_cfg["tapes"] = tapes
        return Session(env_cfg), ds

    def test_env_consumes_shorter_tape_at_reduced_dimensions(self):
        """slow+jit: the env built from the shipped config resolves its
        n_orders_padded / steps_per_episode below the 5459/1365 baseline —
        the reduction is realized end-to-end in the env the trainer sees."""
        env, ds = self._build_env()
        assert env.n_orders_padded < BASELINE_N_ORDERS_PADDED, (
            f"env.n_orders_padded={env.n_orders_padded} not below baseline "
            f"{BASELINE_N_ORDERS_PADDED}"
        )
        assert env.steps_per_episode < BASELINE_STEPS_PER_EPISODE, (
            f"env.steps_per_episode={env.steps_per_episode} not below baseline "
            f"{BASELINE_STEPS_PER_EPISODE}"
        )
        # n_tapes matches the resolved outer product of shipped selection.
        assert env.n_tapes >= 1

    def test_each_tape_reaches_done_at_gct_within_reduced_horizon(self):
        """slow+jit: for EACH shipped tape, reset → step (jitted Session,
        under the OQ-18 outer ``enable_x64`` context) submitting no order
        (``AgentOrder.none()``) until ``done``. Assert done fires via ``time >= GCT`` and the
        number of steps to done is in [1, steps_per_episode] — the episode
        terminates at GCT and never runs past the reduced horizon.

        Iterating tapes by pinning ``state.tape_idx`` (rather than searching
        reset keys) makes the per-tape coverage deterministic. The tape
        content is unchanged; only the reset-sampled index is overridden."""
        env, ds = self._build_env()
        gct = env.gct_minutes_from_got
        from omiecid_simulator.omiecid.omie_session import AgentOrder
        with jax.experimental.enable_x64():
            for tidx in range(env.n_tapes):
                key = jax.random.PRNGKey(0)
                state = env.reset(key)
                state = state.replace(tape_idx=jnp.int32(tidx))
                done_all = False
                steps = 0
                # Hard cap a hair past the horizon so a never-terminating
                # regression fails loudly instead of hanging.
                cap = env.steps_per_episode + 2
                while not done_all and steps < cap:
                    key, sk = jax.random.split(key)
                    state, _report, done = env.step_env(
                        sk, state, AgentOrder.none()
                    )
                    steps += 1
                    done_all = bool(done)
                assert done_all, (
                    f"tape {tidx}: episode did not reach done within "
                    f"{cap} steps (steps_per_episode={env.steps_per_episode})"
                )
                assert int(state.time) >= gct, (
                    f"tape {tidx}: done fired but time={int(state.time)} < GCT={gct} "
                    f"(done must be driven by time >= GCT)"
                )
                assert 1 <= steps <= env.steps_per_episode, (
                    f"tape {tidx}: reached done in {steps} steps, outside the "
                    f"reduced horizon [1, {env.steps_per_episode}]"
                )

    def test_reset_produces_clean_initial_state_on_shipped_tape(self):
        """slow+jit: reset on the shipped tape yields a clean GOT-origin
        initial state — time==0 (minutes-from-GOT), tape_cursor==0, OPEN
        status, zero inventory — so the reduced-tape episode starts from a
        faithful gate-open state."""
        env, ds = self._build_env()
        with jax.experimental.enable_x64():
            state = env.reset(jax.random.PRNGKey(0))
        assert int(state.time) == 0, "reset time must be 0 (GOT origin)"
        assert int(state.tape_cursor) == 0, "reset tape_cursor must be 0"
        assert int(state.contract_status) == int(ContractStatus.OPEN)
        assert float(jnp.sum(jnp.abs(state.agent_inventory))) == 0.0
        assert 0 <= int(state.tape_idx) < env.n_tapes


# ===========================================================================
# Pad-row identity guard — the loader's inert sentinel is honored (§14.5)
# ===========================================================================

class TestPadRowSentinelIsInert:
    """The loader's documented pad-row semantics underpin the reduction:
    pad rows are inert (order_id <= 0, timestamp <= 0) so they neither
    produce a fill nor advance simulated time. This guards the ``max(N_i) +
    pad_extra`` length rule's correctness — inflating the length with pad rows
    must not change fills or the GCT-crossing step. FAST (pure constant)."""

    def test_pad_row_is_inert_by_construction(self):
        """The module ``_PAD_ORDER_ROW`` sentinel has order_id <= 0 (column 4)
        and timestamp <= 0 (column 6), the two conditions the env/replay use to
        treat it as a no-op that does not advance time or produce a fill."""
        assert _PAD_ORDER_ROW[4] <= 0, "pad row order_id (col 4) must be <= 0 (inert)"
        assert _PAD_ORDER_ROW[6] <= 0, "pad row timestamp (col 6) must be <= 0 (no time advance)"
        assert len(_PAD_ORDER_ROW) == 13, "pad row must be a 13-column order row"
