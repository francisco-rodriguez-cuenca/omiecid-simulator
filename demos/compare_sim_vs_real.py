#!/usr/bin/env python3
"""Compare a simulated OMIE trading session against real OMIE trades.

Replays one contract's real historical orders through the simulator
(``replay_orders_through_book``, H1 ``rest_until_gct`` heuristic) and prints
a side-by-side SIM-vs-REAL "distance" report: trade count, traded volume,
VWAP, and settlement value with deltas, plus the replay match metrics
(``match_rate`` / ``extra_trades`` / ``extra_volume_pct``).

The two will NOT be identical — see the "why they differ" footer (OQ-10
cancellation-blindness, R-CROSSZONA cross-border legs, simplified iceberg /
ATC / IDA modeling). The report surfaces that distance anyway.

Only delivery dates with BOTH ``orders_YYYYMMDD.1`` AND ``trades_YYYYMMDD.1``
on disk can be compared. The default date (2025-06-02) is one such date; the
benchmark default (2026-01-12) has orders but no trades file and will error.

Usage (after ``pip install ".[examples]"``, from the repository root):
    python demos/compare_sim_vs_real.py
    python demos/compare_sim_vs_real.py --contract-id 58 --delivery-date 2025-06-10
    python demos/compare_sim_vs_real.py --backend gpu
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Backend selection must happen before JAX is imported (JAX reads
# JAX_PLATFORMS at import time). Parse --backend against a throwaway
# argparse and pin the platform here.
_pre = argparse.ArgumentParser(add_help=False)
# Device policy: default is CPU (deterministic, no GPU needed).
# ``--backend gpu`` remains a valid opt-in.
_pre.add_argument("--backend", choices=("gpu", "cpu"), default="cpu")
_pre_args, _ = _pre.parse_known_args()
# JAX_PLATFORMS token: NVIDIA GPU is 'cuda', CPU is 'cpu'.
os.environ["JAX_PLATFORMS"] = "cuda" if _pre_args.backend == "gpu" else "cpu"

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

from functools import partial

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    _row_to_order_traced,
    load_omie_orders,
    load_omie_trades,
    replay_orders_through_book,
)
from omiecid_simulator.omiecid.contract import (
    Contract,
    ContractStatus,
    generate_contracts,
)
from omiecid_simulator.omiecid.contract_book import ContractBook, _submit_order
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig


def make_open_contract(cid: int) -> Contract:
    base = generate_contracts()[cid]
    return Contract(
        contract_id=base.contract_id,
        delivery_start_minutes=base.delivery_start_minutes,
        delivery_end_minutes=base.delivery_end_minutes,
        gct_minutes=base.gct_minutes,
        status=ContractStatus.OPEN,
    )


def _trades_file_path(data_path: str, delivery_date: str) -> str:
    """Mirror the loader's trades path: ``{data_path}/trades_{YYYYMMDD}.1``."""
    return f"{data_path}/trades_{delivery_date.replace('-', '')}.1"


@partial(jax.jit, static_argnames=("product_config", "market_config"))
def _peak_resting_scan(state, orders, product_config, market_config):
    """Replay orders and track peak simultaneous resting orders per side.

    ``OMIEMarketConfig.book_depth`` (default 100) sizes the asks / bids
    arrays — it is the number of physical slots per side that hold
    *simultaneously*-resting orders, recycled as orders match or clear.
    It is NOT a cumulative session count. When a side is full,
    ``add_order`` (jaxob) overwrites the last slot (``jnp.where`` returns
    ``fill_value=-1`` ⇒ the final row), so breaching the cap mid-session
    corrupts the book. This scan measures whether that happens and when.

    Returns ``(peak_asks, peak_bids, first_breach_step)`` as 0-d int32.
    ``first_breach_step`` is the first order index where either side
    reached ``book_depth``; ``-1`` if never.
    """
    cap = market_config.book_depth

    def step(carry, xs):
        st, peak_a, peak_b, first_breach = carry
        i, row = xs
        order = _row_to_order_traced(row, product_config)
        st, _ = _submit_order(st, order, product_config, market_config)
        a = jnp.sum(st.lob.asks[:, 1] > 0).astype(jnp.int32)
        b = jnp.sum(st.lob.bids[:, 1] > 0).astype(jnp.int32)
        peak_a = jnp.maximum(peak_a, a)
        peak_b = jnp.maximum(peak_b, b)
        breached_now = (a >= cap) | (b >= cap)
        # Record the first breach step only (first_breach still -1).
        first_breach = jnp.where(
            (first_breach < 0) & breached_now, i, first_breach,
        )
        return (st, peak_a, peak_b, first_breach), None

    n = orders.shape[0]
    idx = jnp.arange(n, dtype=jnp.int32)
    init = (state, jnp.int32(0), jnp.int32(0), jnp.int32(-1))
    (final_state, peak_a, peak_b, first_breach), _ = jax.lax.scan(
        step, init, (idx, orders),
    )
    return peak_a, peak_b, first_breach


def _real_aggregates(historical_trades: jnp.ndarray, pc: OMIEProductConfig) -> dict:
    """Aggregate real historical trades.

    ``historical_trades`` is the loader's (M, 6) int32 array; columns
    [1] = tick-encoded price, [2] = tick-encoded quantity. Decode to
    EUR/MWh and MW, then reduce.
    """
    m = int(historical_trades.shape[0])
    if m == 0:
        return {"count": 0, "volume": 0.0, "vwap": None, "value": 0.0}
    p = historical_trades[:, 1].astype(jnp.float32) * pc.price_tick + pc.price_min
    q = historical_trades[:, 2].astype(jnp.float32) * pc.qty_tick
    return _reduce(p, q, m, pc)


def _sim_aggregates(replayed_trades: jnp.ndarray, pc: OMIEProductConfig) -> dict:
    """Aggregate replayed (simulated) trades.

    ``ReplayResult.replayed_trades`` is the (K, 3) float32 array with
    columns [price (EUR/MWh), qty (MW), aggressor_oid] — already decoded.
    """
    k = int(replayed_trades.shape[0])
    if k == 0:
        return {"count": 0, "volume": 0.0, "vwap": None, "value": 0.0}
    p = replayed_trades[:, 0].astype(jnp.float32)
    q = replayed_trades[:, 1].astype(jnp.float32)
    return _reduce(p, q, k, pc)


def _reduce(p: jnp.ndarray, q: jnp.ndarray, count: int, pc: OMIEProductConfig) -> dict:
    """Shared count / volume / VWAP / settlement-value reduction.

    Settlement value uses V = Q (MW) × p (EUR/MWh) × d (h), d=0.25 for
    15-min contracts (market-spec §2.3).
    """
    volume = float(jnp.sum(q))
    pq = float(jnp.sum(p * q))
    vwap = (pq / volume) if volume > 0 else None
    value = pq * pc.delivery_duration_hours
    return {"count": count, "volume": volume, "vwap": vwap, "value": value}


def _fmt_num(x, width=12, prec=2):
    if x is None:
        return f"{'—':>{width}}"
    return f"{x:>{width},.{prec}f}"


def _fmt_delta_abs(sim, real, width=11, prec=2):
    if sim is None or real is None:
        return f"{'—':>{width}}"
    return f"{sim - real:>+{width},.{prec}f}"


def _fmt_delta_pct(sim, real, width=8):
    if sim is None or real is None or real == 0:
        return f"{'—':>{width}}"
    return f"{(sim - real) / real * 100:>+{width-1}.1f}%"


def _row(label, sim, real, prec=2):
    return (
        f"{label:<22}{_fmt_num(sim, prec=prec)}  {_fmt_num(real, prec=prec)}  "
        f"{_fmt_delta_abs(sim, real, prec=prec)}  {_fmt_delta_pct(sim, real)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract-id", type=int, default=81)
    ap.add_argument("--delivery-date", default="2025-06-02",
                    help="Delivery date YYYY-MM-DD. Must have both orders and "
                         "trades files on disk (2025-06-01..2025-06-30).")
    ap.add_argument("--data-path", default="data/rawOMIE")
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="cpu",
                    help="JAX backend; pinned via JAX_PLATFORMS at import time. "
                         "Default is CPU (simulator runs default to CPU; "
                         "pass --backend gpu to opt in).")
    args = ap.parse_args()

    # Guard: trades file must exist, otherwise sim-vs-real is impossible.
    trades_path = _trades_file_path(args.data_path, args.delivery_date)
    if not os.path.exists(trades_path):
        print(
            f"ERROR: no trades file at {trades_path}\n"
            f"  sim-vs-real comparison needs BOTH orders and trades for the "
            f"delivery date.\n"
            f"  On-disk dates with both files: 2025-06-01 .. 2025-06-30.\n"
            f"  Pick a --delivery-date in that window (default is 2025-06-02).",
            file=sys.stderr,
        )
        sys.exit(1)

    active_backend = jax.default_backend()
    print(f"OMIE sim-vs-real — {args.delivery_date} contract {args.contract_id}"
          f" [backend={active_backend}]")
    print("-" * 60)

    cfg = OMIEDataConfig(
        data_path=args.data_path,
        contract_id=args.contract_id,
        delivery_date=args.delivery_date,
    )
    orders = load_omie_orders(cfg)
    trades = load_omie_trades(cfg)

    contract = make_open_contract(args.contract_id)
    book = ContractBook(contract, OMIEProductConfig(), OMIEMarketConfig())
    pc = book._product_config

    # Single replay: ReplayResult gives us both the simulated trades and
    # the match metrics (match_rate / extra_trades / extra_volume_pct).
    result = replay_orders_through_book(orders, book, trades)

    sim = _sim_aggregates(result.replayed_trades, pc)
    real = _real_aggregates(trades, pc)

    print(f"{'':<22}{'SIM':>12}  {'REAL':>12}  {'Δ (abs)':>11}  {'Δ (%)':>8}")
    print(_row("trade count", sim["count"], real["count"], prec=0))
    print(_row("traded vol (MW)", sim["volume"], real["volume"]))
    print(_row("VWAP (EUR/MWh)", sim["vwap"], real["vwap"]))
    print(_row("settle value (EUR)", sim["value"], real["value"]))

    print("-" * 60)
    print("match metrics (replay vs history):")
    if real["count"] == 0:
        print("  no historical trades for this contract — match metrics N/A")
    else:
        print(f"  matched / total       {result.historical_trades_matched} / "
              f"{result.historical_trades_total}"
              f"       (match_rate {result.match_rate * 100:.1f}%)")
        print(f"  extra trades          {result.extra_trades}")
        print(f"  extra volume          {result.extra_volume_pct * 100:.1f} %")

    # Book-depth cap diagnostic. book_depth (default 100/side) is the number
    # of physical slots for SIMULTANEOUSLY-resting orders, recycled as orders
    # match/clear — NOT a cumulative session count. We measure whether the
    # cap was actually breached DURING matching (peak simultaneous resting
    # reaching the cap), which is the only thing that corrupts the book. A
    # full FINAL book alone is just the H1 no-cancellation artifact and does
    # not prove the cap bit.
    book_depth = book._market_config.book_depth
    if orders.shape[0] > 0:
        pa, pb, fb = _peak_resting_scan(
            book.init(), orders,
            product_config=pc, market_config=book._market_config,
        )
        peak_asks, peak_bids, first_breach = int(pa), int(pb), int(fb)
    else:
        peak_asks = peak_bids = 0
        first_breach = -1
    breached = first_breach >= 0

    print("-" * 60)
    print(f"book cap: {book_depth}/side (simultaneous resting slots)")
    print(f"  peak resting          asks={peak_asks}  bids={peak_bids}")
    if breached:
        print(f"  cap breached          YES — first at order #{first_breach}")
        print("                        ⇒ jaxob overwrites the last slot when a")
        print("                          side is full; the book is corrupted and")
        print("                          the sim can under-fill vs real.")
    else:
        print("  cap breached          no (cap did not bind during matching)")

    print("-" * 60)
    print("why they differ (this distance is expected, not a bug):")
    if breached:
        print(f"  BOOK_DEPTH   simultaneous resting orders exceeded the")
        print(f"               {book_depth}/side cap during matching (see above).")
    print("  OQ-10        cancellations absent from OMIE public data; H1")
    print("               rest_until_gct rests every unmatched order. With no")
    print("               cancels the book diverges from the true historical")
    print("               state, so replayed fills do not line up with reality.")
    print("  R-CROSSZONA  XBID cross-border legs have a blank counterparty ⇒")
    print("               structural ceiling on match_rate; not reproducible.")
    print("  (Also: simplified iceberg semantics; no ATC / IDA / batch modeling.)")


if __name__ == "__main__":
    main()
