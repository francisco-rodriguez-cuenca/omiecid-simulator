#!/usr/bin/env python3
"""Real-data OMIE simulator demo for a fixed contract.

Feeds real OMIE orders from ``data/rawOMIE/orders_<YYYYMMDD>.1`` through
the simulator on a fixed (contract, slice_start) pair and produces an
interactive + print-ready HTML walkthrough of a 35-order slice.

Usage (after ``pip install ".[examples]"``, from the repository root):
    python demos/generate_real_data_demo.py

Output:
    demos/<YYYYMMDD>_omie_real/demo.html
"""

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
from tqdm import tqdm

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    _row_to_order_traced,
    load_omie_orders, load_omie_agent_metadata, row_to_order,
)
from omiecid_simulator.omiecid.contract import (
    Contract, ContractStatus, generate_contracts,
)
from omiecid_simulator.omiecid.contract_book import ContractBook, _submit_order
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig, OMIEMarketConfig
from omiecid_simulator.omiecid.order_types import (
    OMIEOrder, Side, ExecRestriction, Validity, Status, make_order,
)
from omiecid_simulator.omiecid.settlement import compute_trade_value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "data/rawOMIE"
DEFAULT_DELIVERY_DATE = "2026-01-12"
DEFAULT_CONTRACT_ID = 81   # 20:15–20:30 delivery, price-precision regression contract
DEFAULT_SLICE_START = 987  # pre-selected busiest 35-order window on cid 81 (15 matched)
DEFAULT_SLICE_K = 35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    tqdm.write(f"{label} done in {time.perf_counter() - t0:.1f}s")


def _make_open_contract(cid: int) -> Contract:
    base = generate_contracts()[cid]
    return Contract(
        contract_id=base.contract_id,
        delivery_start_minutes=base.delivery_start_minutes,
        delivery_end_minutes=base.delivery_end_minutes,
        gct_minutes=base.gct_minutes,
        status=ContractStatus.OPEN,
    )


def _side_str(side: int) -> str:
    return "BUY" if int(side) == int(Side.BUY) else "SELL"


def _exec_str(er: int) -> str:
    return {
        int(ExecRestriction.NON): "NON",
        int(ExecRestriction.IOC): "IOC",
        int(ExecRestriction.FOK): "FOK",
        int(ExecRestriction.ICE): "ICE",
    }[int(er)]


def _validity_str(v: int) -> str:
    return "GFS" if int(v) == int(Validity.GFS) else "GTD"


def _is_full_fill(order: OMIEOrder, result) -> bool:
    return result.traded and abs(result.trade_qty - order.quantity) < 1e-6


def _rule_text(order: OMIEOrder, result, hidden_before: float,
               hidden_after: float) -> str:
    er = _exec_str(order.exec_restriction)
    side = _side_str(order.side)
    if not result.accepted:
        return f"Rejected (validation failed)."
    if er == "FOK" and not result.traded:
        return ("FOK pre-check: insufficient visible liquidity at the limit "
                f"price ({order.price:.2f} EUR/MWh). Order cancelled entirely "
                "— FOK does not see iceberg hidden quantity.")
    if not result.traded:
        return ("Accepted, resting in book. No resting counterparty crossed "
                f"at {order.price:.2f} EUR/MWh, so the order joins price-time "
                "priority on its own side.")
    full = _is_full_fill(order, result)
    prefix = f"{er} {side} "
    if er == "ICE":
        ice_note = ""
        if hidden_after > 0:
            ice_note = (f" New slice regenerated (peak {order.peak_qty:.2f} MW); "
                        f"remaining hidden: {hidden_after:.2f} MW.")
        elif hidden_before > 0:
            ice_note = " Iceberg exhausted — remaining hidden drained to 0."
        return (prefix + f"matched {result.trade_qty:.2f} MW @ "
                f"{result.trade_price:.2f} EUR/MWh (maker price). "
                f"Settlement: {result.settlement_value:.2f} EUR."
                + ice_note)
    if er == "IOC":
        return (prefix + f"matched {result.trade_qty:.2f} MW @ "
                f"{result.trade_price:.2f} EUR/MWh. "
                + ("Fully filled." if full else
                   f"Partial fill — remainder "
                   f"{order.quantity - result.trade_qty:.2f} MW cancelled (IOC).")
                + f" Settlement: {result.settlement_value:.2f} EUR.")
    if er == "FOK":
        return (prefix + f"filled fully: {result.trade_qty:.2f} MW @ "
                f"{result.trade_price:.2f} EUR/MWh. "
                f"Settlement: {result.settlement_value:.2f} EUR.")
    # NON
    if full:
        tail = "Fully filled at the maker price."
    else:
        tail = (f"Partially filled; remainder "
                f"{order.quantity - result.trade_qty:.2f} MW rests in the book.")
    return (prefix + f"matched {result.trade_qty:.2f} MW @ "
            f"{result.trade_price:.2f} EUR/MWh. " + tail
            + f" Settlement: {result.settlement_value:.2f} EUR.")


def _iter_active_icebergs(state):
    """Yield ``(order_id, side, base_price_eur, drifted_price_eur,
    remaining_hidden_mw)`` for every active iceberg slot.

    ``state.iceberg_meta`` is a ``(MAX_ICEBERGS, 8)`` int32
    array — see ``omiecid_simulator/omiecid/iceberg_manager.py`` for the
    column schema. We decode prices and quantities back into OMIE
    domain units for display.
    """
    from omiecid_simulator.omiecid.iceberg_manager import (
        COL_ORDER_ID, COL_REMAINING_HIDDEN, COL_BASE_PRICE_INT,
        COL_SIDE, COL_PPD, COL_SLICE_COUNT,
    )
    from omiecid_simulator.omiecid.encoding import decode_price, decode_qty
    from omiecid_simulator.omiecid.jax_constants import SENTINEL_ORDER_ID

    meta_arr = state.iceberg_meta
    cfg = OMIEProductConfig()
    n_rows = meta_arr.shape[0]
    for i in range(n_rows):
        oid = int(meta_arr[i, COL_ORDER_ID])
        if oid == SENTINEL_ORDER_ID:
            continue
        remaining_int = int(meta_arr[i, COL_REMAINING_HIDDEN])
        if remaining_int <= 0:
            continue
        side = int(meta_arr[i, COL_SIDE])
        base_price_int = int(meta_arr[i, COL_BASE_PRICE_INT])
        ppd_int = int(meta_arr[i, COL_PPD])
        slice_count = int(meta_arr[i, COL_SLICE_COUNT])
        # Drifted price for the NEXT slice to be regenerated:
        # base + ppd * slice_count (encoded space). Note that
        # slice_count is incremented BEFORE submission inside
        # regenerate_slices; here we mirror the dict-state code which
        # exposed ``meta["price"]`` (the most-recently-regenerated
        # slice price). For an iceberg whose visible peak is still
        # resting, the drifted price equals ``base + ppd*slice_count``.
        drifted_int = base_price_int + ppd_int * slice_count
        base_eur = decode_price(base_price_int, cfg)
        drifted_eur = decode_price(drifted_int, cfg)
        remaining_mw = decode_qty(remaining_int, cfg)
        yield oid, side, base_eur, drifted_eur, remaining_mw


def _hidden_total(state) -> float:
    return sum(remaining_mw
               for _oid, _side, _base, _drift, remaining_mw
               in _iter_active_icebergs(state))


def _lob_with_hidden(book, state):
    """Return (bids, asks) with hidden peak annotation per ask/bid level.

    For each visible level, mark whether any tracked iceberg sits at that
    price (hidden_peak=True). The plan uses this to render "(+ hidden)".
    """
    bids, asks = book.get_anonymous_lob(state)
    ice_bid_prices = set()
    ice_ask_prices = set()
    for _oid, side, _base, drifted_eur, _rem in _iter_active_icebergs(state):
        price = round(drifted_eur, 4)
        if side == int(Side.BUY):
            ice_bid_prices.add(price)
        else:
            ice_ask_prices.add(price)
    bids_out = [
        {"price": p, "qty": q, "has_hidden": round(p, 4) in ice_bid_prices}
        for p, q in bids
    ]
    asks_out = [
        {"price": p, "qty": q, "has_hidden": round(p, 4) in ice_ask_prices}
        for p, q in asks
    ]
    return bids_out, asks_out


# ---------------------------------------------------------------------------
# Full replay with snapshots
# ---------------------------------------------------------------------------

@dataclass
class StepSnapshot:
    step_index: int      # 0-based within slice (1..K for real orders; 0 = opener)
    order: dict          # serializable order dict
    agent_label: str
    bids: list
    asks: list
    hidden_total: float
    result: dict
    rule: str
    cumulative_trades: list  # [{price, qty, value, type, marker}]
    highlight: str           # "trade" | "reject" | "fok-unfilled" | ""
    outcome_kind: str = ""   # "TRADED" | "RESTING" | "FOK_UNFILLED" | "REJECTED"
    reject_reason: str = ""  # populated only when outcome_kind == "REJECTED"


# --- Outcome classification -------------------------------------------------

OUTCOME_TRADED = "TRADED"
OUTCOME_RESTING = "RESTING"
OUTCOME_FOK_UNFILLED = "FOK_UNFILLED"
OUTCOME_REJECTED = "REJECTED"

_OUTCOME_HIGHLIGHT = {
    OUTCOME_TRADED: "trade",
    OUTCOME_RESTING: "",
    OUTCOME_FOK_UNFILLED: "fok-unfilled",
    OUTCOME_REJECTED: "reject",
}


def classify_outcome(order: OMIEOrder, result) -> str:
    """Bucket a submit result into one of the four outcome kinds."""
    if not result.accepted:
        return OUTCOME_REJECTED
    if result.traded:
        return OUTCOME_TRADED
    if int(order.exec_restriction) == int(ExecRestriction.FOK):
        # FOK passed validation but failed the liquidity pre-check.
        return OUTCOME_FOK_UNFILLED
    return OUTCOME_RESTING


def _is_on_tick(value: float, tick: float) -> bool:
    """Mirror of order_validator._is_on_tick."""
    ratio = value / tick
    return abs(ratio - round(ratio)) < 1e-9


def reject_reason(order: OMIEOrder, config: OMIEProductConfig) -> str:
    """Replay validator checks and return a short tag + offending value.

    Mirrors the order in which ``validate_order`` at
    [order_validator.py:24-65](omiecid_simulator/omiecid/order_validator.py#L24-L65)
    runs its checks. ``contract_open`` is not checked here because
    this demo forces contracts to OPEN; if no branch fires, returns
    ``"unknown"`` to flag the surprise.
    """
    if order.order_id <= 0:
        return f"order_id (order_id={order.order_id})"
    if int(order.side) not in (int(Side.BUY), int(Side.SELL)):
        return f"bad_side (side={int(order.side)})"
    if order.price < config.price_min or order.price > config.price_max:
        return (f"price_bounds (price={order.price:.6f}, "
                f"bounds=[{config.price_min}, {config.price_max}])")
    if not _is_on_tick(order.price, config.price_tick):
        return f"price_tick (price={order.price:.6f}, tick={config.price_tick})"
    if order.quantity < config.qty_min or order.quantity > config.qty_max:
        return (f"qty_bounds (quantity={order.quantity:.6f}, "
                f"bounds=[{config.qty_min}, {config.qty_max}])")
    if not _is_on_tick(order.quantity, config.qty_tick):
        return (f"qty_tick (quantity={order.quantity:.6f}, "
                f"tick={config.qty_tick})")
    if int(order.exec_restriction) == int(ExecRestriction.ICE):
        if order.peak_qty <= 0:
            return f"ice_peak_missing (peak_qty={order.peak_qty})"
        if order.peak_qty > order.quantity:
            return (f"ice_peak_too_large (peak_qty={order.peak_qty}, "
                    f"quantity={order.quantity})")
    return "unknown"


def _order_to_dict(order: OMIEOrder) -> dict:
    return {
        "order_id": int(order.order_id),
        "agent_id": int(order.agent_id),
        "side": _side_str(order.side),
        "price": float(order.price),
        "quantity": float(order.quantity),
        "peak_qty": float(order.peak_qty),
        "ppd": float(order.ppd),
        "exec_restriction": _exec_str(order.exec_restriction),
        "validity": _validity_str(order.validity),
        "timestamp": int(order.timestamp),
    }


def _result_to_dict(result) -> dict:
    return {
        "accepted": bool(result.accepted),
        "traded": bool(result.traded),
        "trade_qty": float(result.trade_qty),
        "trade_price": float(result.trade_price),
        "settlement_value": float(result.settlement_value),
    }


def _match_type_label(incoming_er: int, lob_had_ice_at_price: bool) -> str:
    """Very small heuristic label for the trade-log 'Type' column."""
    er = _exec_str(incoming_er)
    other = "ICE" if lob_had_ice_at_price else "NON"
    return f"{er} vs {other}"


# Jitted lax.scan that warms the book state and accumulates per-order
# SubmitResult totals (matched-order count, MW traded, settlement value).
# Modelled on ``_replay_orders_body`` but reads ``SubmitResult`` fields
# directly so the totals match the per-order semantics exactly — the raw
# trade-log buffer cannot be summed for MW/value because iceberg peak
# regeneration writes multiple jaxob trade rows per single order match.
def _replay_with_totals_body(state, orders, product_config, market_config):
    def step(carry, row):
        st, n_matched, mw_sum, val_sum = carry
        order = _row_to_order_traced(row, product_config)
        new_st, result = _submit_order(st, order, product_config, market_config)
        traded = result.accepted & result.traded
        n_matched = n_matched + traded.astype(jnp.int32)
        mw_sum = mw_sum + jnp.where(traded, result.trade_qty, jnp.float32(0.0))
        val_sum = val_sum + jnp.where(
            traded, result.settlement_value, jnp.float32(0.0),
        )
        return (new_st, n_matched, mw_sum, val_sum), None

    init = (state, jnp.int32(0), jnp.float32(0.0), jnp.float32(0.0))
    (final_st, n_matched, mw_sum, val_sum), _ = jax.lax.scan(step, init, orders)
    return final_st, n_matched, mw_sum, val_sum


_replay_with_totals = jax.jit(
    _replay_with_totals_body,
    static_argnames=("product_config", "market_config"),
)


def full_replay(cid: int, orders_array, slice_start: int, agent_md: dict,
                slice_k: int):
    """Pass 2: replay orders [0, slice_start + K), capture snapshots in slice.

    Pre-slice orders are warmed up via the jitted batch ``_replay_orders``
    ``lax.scan`` (one call, decodes rows internally). The slice orders then
    run per-order through ``book.submit_order`` so we can capture detailed
    snapshots. Both paths call the identical ``_submit_order`` body, so the
    warmed state is byte-for-byte what a sequential per-order loop would
    produce.
    """
    contract = _make_open_contract(cid)
    book = ContractBook(contract, OMIEProductConfig(), OMIEMarketConfig())
    state = book.init()
    pc = book._product_config

    end = slice_start + slice_k

    # --- Pre-slice batch warm-up --------------------------------------------
    if slice_start == 0:
        pre_matched = 0
        pre_traded_mw = 0.0
        pre_traded_value = 0.0
    else:
        print(
            f"pass 2 (cid={cid:02d}): batch warm-up over {slice_start} "
            f"pre-slice orders...",
            end=" ", flush=True,
        )
        t0 = time.perf_counter()
        pre_rows = orders_array[:slice_start]
        state, n_matched, mw_sum, val_sum = _replay_with_totals(
            state, pre_rows, book._product_config, book._market_config,
        )
        pre_matched = int(n_matched)
        pre_traded_mw = float(mw_sum)
        pre_traded_value = float(val_sum)
        print(f"done in {time.perf_counter() - t0:.1f}s "
              f"({pre_matched} matched orders, {pre_traded_mw:.1f} MW)")

    cum_trades = []
    snapshots: list = []

    # Step 0: opener snapshot capturing the warmed-up state.
    bids_o, asks_o = _lob_with_hidden(book, state)
    snapshots.append(StepSnapshot(
        step_index=0,
        order=None,  # type: ignore
        agent_label="",
        bids=bids_o,
        asks=asks_o,
        hidden_total=_hidden_total(state),
        result=None,  # type: ignore
        rule=(f"Pre-slice activity: {pre_matched} matched orders "
              f"({pre_traded_mw:.1f} MW, {pre_traded_value:.2f} EUR) "
              f"across the first {slice_start} submissions."),
        cumulative_trades=[],
        highlight="",
    ))

    bar = tqdm(
        range(slice_start, end),
        total=slice_k,
        desc=f"pass 2 (slice on cid={cid:02d})",
        unit="ord",
    )

    for i in bar:
        order = row_to_order(orders_array[i], book._product_config)

        # Snapshot inputs (before submit, to identify iceberg-priced levels)
        ice_prices_before = {
            round(drifted, 4)
            for _oid, _side, _base, drifted, _rem
            in _iter_active_icebergs(state)
        }
        hidden_before = _hidden_total(state)

        state, result = book.submit_order(state, order)

        hidden_after = _hidden_total(state)
        bids, asks = _lob_with_hidden(book, state)

        kind = classify_outcome(order, result)
        highlight = _OUTCOME_HIGHLIGHT[kind]
        for t in cum_trades:
            t["current"] = False
        if kind == OUTCOME_TRADED:
            match_type = _match_type_label(
                int(order.exec_restriction),
                round(order.price, 4) in ice_prices_before,
            )
            cum_trades.append({
                "price": float(result.trade_price),
                "qty": float(result.trade_qty),
                "value": float(result.settlement_value),
                "type": match_type,
                "current": True,
            })
        reason = ""
        if kind == OUTCOME_REJECTED:
            reason = reject_reason(order, OMIEProductConfig())

        pair = agent_md.get(int(order.agent_id), ("?", "?"))
        agent_label = (f"{pair[0]} / {pair[1]}"
                       if pair[0] or pair[1] else "(blank-leg)")

        step_i = i - slice_start + 1
        snapshots.append(StepSnapshot(
            step_index=step_i,
            order=_order_to_dict(order),
            agent_label=agent_label,
            bids=bids,
            asks=asks,
            hidden_total=hidden_after,
            result=_result_to_dict(result),
            rule=_rule_text(order, result, hidden_before, hidden_after),
            cumulative_trades=[dict(t) for t in cum_trades],
            highlight=highlight,
            outcome_kind=kind,
            reject_reason=reason,
        ))

    bar.close()
    return snapshots


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OMIE Real-Data Demo — __DATE__ — Contract __CID__</title>
<style>
  :root {
    --bg: #ffffff;
    --surface: #f6f7fa;
    --border: #d0d0d0;
    --accent: #2c5282;
    --accent2: #276749;
    --text: #1a1a1a;
    --muted: #555;
    --red: #c53030;
    --green: #276749;
    --bid: #2f855a;
    --ask: #c53030;
    --bid-bar: #68d391;
    --ask-bar: #fc8181;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.45;
  }
  .nav {
    position: sticky; top: 0; z-index: 10;
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; background: var(--surface);
    border-bottom: 1px solid var(--border); flex-wrap: wrap;
  }
  .nav button {
    padding: 6px 14px; border: 1px solid var(--border);
    background: white; border-radius: 6px; cursor: pointer;
    font-size: 0.9rem;
  }
  .nav button:hover { border-color: var(--accent); color: var(--accent); }
  .nav .counter { font-size: 0.9rem; color: var(--muted); margin-left: auto; }

  .step-page {
    padding: 28px 48px; max-width: 1100px; margin: 0 auto;
    border-bottom: 1px dashed var(--border);
  }
  .step-page.hidden-screen { display: none; }
  .step-header {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 16px; flex-wrap: wrap; margin-bottom: 18px;
  }
  .step-header h1 {
    font-size: 1.4rem; margin: 0; color: var(--accent);
  }
  .step-header .meta { font-size: 0.9rem; color: var(--muted); }

  .zone {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;
  }
  .zone h3 {
    font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--accent);
    margin: 0 0 10px; font-weight: 600;
  }
  .kv-grid {
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: 4px 24px; font-size: 0.92rem;
  }
  .kv-grid .k { color: var(--muted); }
  .kv-grid .v { font-family: 'Cascadia Code', 'Fira Code', monospace; font-weight: 600; }

  .lob-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .lob-side h4 {
    font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;
    margin: 0 0 8px;
  }
  .lob-side.bids h4 { color: var(--bid); }
  .lob-side.asks h4 { color: var(--ask); }
  .lob-row {
    display: flex; align-items: center; gap: 8px; height: 24px;
    font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.85rem;
  }
  .lob-row .price { width: 78px; text-align: right; font-weight: 600; }
  .lob-row .qty { width: 66px; color: var(--muted); }
  .lob-row .bar { flex: 1; font-family: monospace; font-size: 0.95rem; letter-spacing: -1px; }
  .lob-side.bids .bar { color: var(--bid-bar); text-align: right; }
  .lob-side.asks .bar { color: var(--ask-bar); text-align: left; }
  .lob-row .hidden-note { color: var(--muted); font-style: italic; font-size: 0.8rem; }
  .lob-empty { color: var(--muted); font-style: italic; font-size: 0.88rem; }
  .lob-spread {
    margin-top: 8px; font-size: 0.82rem; color: var(--muted);
    font-family: 'Cascadia Code', 'Fira Code', monospace;
  }

  .outcome.trade { border-left: 4px solid var(--green); }
  .outcome.reject { border-left: 4px solid var(--red); }
  .outcome.fok-unfilled { border-left: 4px solid #b7791f; }

  table.trade-log { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
  table.trade-log th, table.trade-log td {
    padding: 4px 8px; border-bottom: 1px solid var(--border); text-align: right;
  }
  table.trade-log th { background: white; color: var(--accent); font-weight: 600; }
  table.trade-log td.type-col { text-align: left; }
  table.trade-log tr.current td { background: #fef9c3; font-weight: 600; }
  .trade-log-empty { color: var(--muted); font-style: italic; font-size: 0.88rem; }
  .trade-totals {
    margin-top: 8px; font-size: 0.85rem;
    font-family: 'Cascadia Code', 'Fira Code', monospace; color: var(--accent);
  }

  .rule {
    font-size: 0.92rem; line-height: 1.5;
  }

  /* Print layout: one page per step, hide nav */
  @media print {
    .nav { display: none; }
    .step-page { page-break-after: always; border: none; padding: 20px 30px; }
    .step-page.hidden-screen { display: block; }
    html, body { background: white; color: black; }
    body { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
  }
</style>
</head>
<body>

<div class="nav">
  <button onclick="prevStep()">&larr; Prev</button>
  <button onclick="nextStep()">Next &rarr;</button>
  <button onclick="window.print()">Print</button>
  <span class="counter" id="counter">Step 0 of __TOTAL__</span>
</div>

__PAGES__

<script>
const TOTAL = __TOTAL__;
let cur = 0;
function show(i) {
  if (i < 0 || i > TOTAL) return;
  cur = i;
  document.querySelectorAll('.step-page').forEach((el, j) => {
    el.classList.toggle('hidden-screen', j !== cur);
  });
  document.getElementById('counter').textContent =
    cur === 0 ? 'Opening book (Step 0 of ' + TOTAL + ')'
              : 'Step ' + cur + ' of ' + TOTAL;
}
function nextStep() { show(Math.min(cur + 1, TOTAL)); }
function prevStep() { show(Math.max(cur - 1, 0)); }
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight' || e.key === ' ') nextStep();
  if (e.key === 'ArrowLeft') prevStep();
});
show(0);
</script>
</body>
</html>
"""


def _bar_glyphs(qty: float, max_qty: float, width: int = 20) -> str:
    if max_qty <= 0:
        return ""
    n = max(1, round((qty / max_qty) * width))
    return "▓" * n  # ▓


def _render_lob_zone(snap: StepSnapshot) -> str:
    all_qtys = [b["qty"] for b in snap.bids] + [a["qty"] for a in snap.asks]
    max_q = max(all_qtys) if all_qtys else 0.0

    def row_html(side: str, lvl: dict) -> str:
        bar = _bar_glyphs(lvl["qty"], max_q)
        hidden = ' <span class="hidden-note">(+ hidden)</span>' if lvl["has_hidden"] else ""
        if side == "bids":
            # bar | qty | price
            return (f'<div class="lob-row"><span class="bar">{bar}</span>'
                    f'<span class="qty">{lvl["qty"]:.2f} MW</span>'
                    f'<span class="price">{lvl["price"]:.2f}</span>{hidden}</div>')
        else:
            return (f'<div class="lob-row"><span class="price">{lvl["price"]:.2f}</span>'
                    f'<span class="qty">{lvl["qty"]:.2f} MW</span>'
                    f'<span class="bar">{bar}</span>{hidden}</div>')

    bids_html = "".join(row_html("bids", b) for b in snap.bids) \
        or '<div class="lob-empty">(empty)</div>'
    asks_html = "".join(row_html("asks", a) for a in snap.asks) \
        or '<div class="lob-empty">(empty)</div>'

    best_bid = snap.bids[0]["price"] if snap.bids else None
    best_ask = snap.asks[0]["price"] if snap.asks else None
    if best_bid is not None and best_ask is not None:
        spread = (f"Spread: {best_ask - best_bid:.2f} EUR/MWh"
                  f" &nbsp; Best bid: {best_bid:.2f} &nbsp; Best ask: {best_ask:.2f}")
    else:
        spread = "No two-sided market"
    hidden_line = (f" &nbsp;|&nbsp; Hidden (iceberg): {snap.hidden_total:.2f} MW"
                   if snap.hidden_total > 0 else "")

    return f"""
<div class="zone">
  <h3>Limit Order Book &mdash; after this order</h3>
  <div class="lob-grid">
    <div class="lob-side bids">
      <h4>Bids (Buy)</h4>
      {bids_html}
    </div>
    <div class="lob-side asks">
      <h4>Asks (Sell)</h4>
      {asks_html}
    </div>
  </div>
  <div class="lob-spread">{spread}{hidden_line}</div>
</div>
"""


def _render_order_zone(snap: StepSnapshot) -> str:
    if snap.order is None:
        return ""
    o = snap.order
    ppd = f"{o['ppd']:+.2f}" if o["ppd"] else "—"
    peak = f"{o['peak_qty']:.2f} MW" if o["peak_qty"] else "—"
    return f"""
<div class="zone">
  <h3>Incoming order</h3>
  <div class="kv-grid">
    <div class="k">Participant</div><div class="v">{snap.agent_label}</div>
    <div class="k">Side</div><div class="v">{o['side']}</div>
    <div class="k">Order type</div><div class="v">{o['exec_restriction']}</div>
    <div class="k">Price</div><div class="v">{o['price']:.2f} EUR/MWh</div>
    <div class="k">Quantity</div><div class="v">{o['quantity']:.2f} MW</div>
    <div class="k">Peak (iceberg)</div><div class="v">{peak}</div>
    <div class="k">PPD</div><div class="v">{ppd}</div>
    <div class="k">Validity</div><div class="v">{o['validity']}</div>
    <div class="k">Order ID</div><div class="v">#{o['order_id']}</div>
    <div class="k">Timestamp (s)</div><div class="v">{o['timestamp']}</div>
  </div>
</div>
"""


def _render_outcome_zone(snap: StepSnapshot) -> str:
    if snap.result is None:
        # Step 0 opener — render rule text without an outcome header.
        return f"""
<div class="zone">
  <h3>Pre-slice activity</h3>
  <div class="rule">{snap.rule}</div>
</div>
"""
    r = snap.result
    hl = snap.highlight or ""
    kind = snap.outcome_kind
    if kind == OUTCOME_REJECTED:
        reason = snap.reject_reason or "unknown"
        head = f"REJECTED (validation) — {reason}"
    elif kind == OUTCOME_FOK_UNFILLED:
        head = ("UNFILLED (FOK pre-check) — insufficient visible liquidity "
                f"at {snap.order['price']:.2f} EUR/MWh; order cancelled entirely")
    elif kind == OUTCOME_TRADED:
        head = (f"MATCHED: {r['trade_qty']:.2f} MW @ {r['trade_price']:.2f} EUR/MWh "
                f"(settlement: {r['settlement_value']:.2f} EUR)")
    else:
        head = "Accepted — resting in book (no match)"
    return f"""
<div class="zone outcome {hl}">
  <h3>What happened</h3>
  <div style="font-weight:600; margin-bottom:6px;">{head}</div>
  <div class="rule">{snap.rule}</div>
</div>
"""


def _render_trade_log(snap: StepSnapshot) -> str:
    if not snap.cumulative_trades:
        return """
<div class="zone">
  <h3>Trade log (cumulative within slice)</h3>
  <div class="trade-log-empty">No trades yet.</div>
</div>
"""
    rows = []
    total_mw = 0.0
    total_val = 0.0
    for i, t in enumerate(snap.cumulative_trades, start=1):
        cls = ' class="current"' if t.get("current") else ""
        marker = " &larr;" if t.get("current") else ""
        rows.append(
            f'<tr{cls}><td>{i}</td>'
            f'<td>{t["price"]:.2f}</td>'
            f'<td>{t["qty"]:.2f}</td>'
            f'<td>{t["value"]:.2f}</td>'
            f'<td class="type-col">{t["type"]}{marker}</td></tr>'
        )
        total_mw += t["qty"]
        total_val += t["value"]
    return f"""
<div class="zone">
  <h3>Trade log (cumulative within slice)</h3>
  <table class="trade-log">
    <thead><tr><th>#</th><th>Price EUR/MWh</th><th>Qty MW</th>
      <th>Value EUR</th><th class="type-col">Type</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <div class="trade-totals">
    Total matched: {total_mw:.2f} MW &nbsp; Total value: {total_val:.2f} EUR
  </div>
</div>
"""


def _render_step_page(snap: StepSnapshot, cid: int, contract: Contract,
                      date_str: str, total_steps: int,
                      t_minus_min: int) -> str:
    title = (f"Step {snap.step_index} / {total_steps}" if snap.step_index > 0
             else "Step 0 &mdash; Opening book")
    start_h = contract.delivery_start_minutes // 60
    start_m = contract.delivery_start_minutes % 60
    end_h = contract.delivery_end_minutes // 60
    end_m = contract.delivery_end_minutes % 60
    delivery = f"{start_h:02d}:{start_m:02d}–{end_h:02d}:{end_m:02d}"

    meta_txt = f"Contract {cid} ({delivery}) &nbsp; Date: {date_str}"
    if snap.order is not None and t_minus_min is not None:
        meta_txt += f" &nbsp; T{'−' if t_minus_min >= 0 else '+'}{abs(t_minus_min)} min to GCT"

    return f"""
<div class="step-page">
  <div class="step-header">
    <h1>{title}</h1>
    <div class="meta">{meta_txt}</div>
  </div>
  {_render_order_zone(snap)}
  {_render_lob_zone(snap)}
  {_render_outcome_zone(snap)}
  {_render_trade_log(snap)}
</div>
"""


def render_html(snapshots, cid: int, contract: Contract, date_str: str,
                slice_k: int) -> str:
    total_real = slice_k
    pages = []
    for snap in snapshots:
        t_minus = None
        if snap.order is not None:
            ts = snap.order["timestamp"]
            t_minus = (contract.gct_minutes * 60 - ts) // 60
        pages.append(_render_step_page(snap, cid, contract, date_str,
                                        total_real, t_minus))
    html = HTML_TEMPLATE
    html = html.replace("__TOTAL__", str(total_real))
    html = html.replace("__DATE__", date_str)
    html = html.replace("__CID__", str(cid))
    html = html.replace("__PAGES__", "\n".join(pages))
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--delivery-date", default=DEFAULT_DELIVERY_DATE,
                   help=f"YYYY-MM-DD (default: {DEFAULT_DELIVERY_DATE})")
    p.add_argument("--contract-id", type=int, default=DEFAULT_CONTRACT_ID,
                   help=f"OMIE contract id (default: {DEFAULT_CONTRACT_ID})")
    p.add_argument("--slice-start", type=int, default=DEFAULT_SLICE_START,
                   help=f"index of first order in slice (default: {DEFAULT_SLICE_START})")
    p.add_argument("--slice-k", type=int, default=DEFAULT_SLICE_K,
                   help=f"slice length (default: {DEFAULT_SLICE_K})")
    return p.parse_args(argv)


def _output_dir(base: Path, date_str: str, cid: int, slice_start: int) -> Path:
    """The pinned cid=81/slice=987/2026-01-12 demo keeps its historic path
    so that existing references to it by name stay valid.
    All other invocations get a cid+slice-suffixed directory.
    """
    date_nodash = date_str.replace("-", "")
    is_pinned = (cid == DEFAULT_CONTRACT_ID
                 and slice_start == DEFAULT_SLICE_START
                 and date_str == DEFAULT_DELIVERY_DATE)
    name = (f"{date_nodash}_omie_real"
            if is_pinned
            else f"{date_nodash}_omie_real_cid{cid:02d}_s{slice_start:04d}")
    return base / name


def main(argv=None):
    # Warm-cache reuse with the OMIE test suite (shared persistent JAX cache).
    import jax
    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    args = _parse_args(argv)
    cid = args.contract_id
    slice_start = args.slice_start
    slice_k = args.slice_k
    date_str = args.delivery_date

    t_start = time.perf_counter()

    cfg = OMIEDataConfig(
        data_path=DATA_PATH,
        contract_id=cid,
        delivery_date=date_str,
    )

    print("loading agent metadata...", end=" ", flush=True)
    agent_md = load_omie_agent_metadata(cfg)
    print(f"done ({len(agent_md)} participants)")

    print(f"loading orders for cid={cid}...", end=" ", flush=True)
    t0 = time.perf_counter()
    orders = load_omie_orders(cfg)
    print(f"done in {time.perf_counter() - t0:.1f}s  ({orders.shape[0]} orders)")

    if orders.shape[0] < slice_start + slice_k:
        print(f"cid={cid} has only {orders.shape[0]} orders, "
              f"need >= {slice_start + slice_k}. Aborting.")
        return 1

    snapshots = full_replay(cid, orders, slice_start, agent_md, slice_k)

    with timed("rendering HTML..."):
        contract = _make_open_contract(cid)
        html = render_html(snapshots, cid, contract, date_str, slice_k)
        out_dir = _output_dir(Path(__file__).resolve().parent,
                              date_str, cid, slice_start)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "demo.html"
        out_path.write_text(html, encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    mm, ss = divmod(int(elapsed), 60)
    size_kb = out_path.stat().st_size / 1024
    print(f"done: cid={cid}, elapsed {mm:02d}:{ss:02d}")
    print(f"wrote {out_path}  ({len(snapshots)} pages, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
