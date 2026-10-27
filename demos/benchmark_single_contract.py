#!/usr/bin/env python3
"""Fast OMIE single-contract benchmark.

Replays one full contract through the JIT'd ``lax.scan`` path
(``_replay_orders``) and prints text timings — no HTML, no files written.

Usage (after ``pip install ".[examples]"``, from the repository root):
    python demos/benchmark_single_contract.py
    python demos/benchmark_single_contract.py --contract-id 58
    python demos/benchmark_single_contract.py --backend cpu
"""

import argparse
import os
import sys
import time
from functools import partial
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

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    _replay_orders,
    load_omie_orders,
)
from omiecid_simulator.omiecid.contract import (
    Contract,
    ContractStatus,
    generate_contracts,
)
from omiecid_simulator.omiecid.contract_book import ContractBook
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


@partial(jax.jit, static_argnames=("product_config", "market_config"))
def _replay_and_reduce(state0, orders, product_config, market_config):
    final_state, buf = _replay_orders(state0, orders, product_config, market_config)
    fill_mask = buf[..., 0] != -1
    n_fills = jnp.sum(fill_mask)
    qty_ticks = jnp.sum(jnp.where(fill_mask, buf[..., 2], 0))
    return final_state, n_fills, qty_ticks


def time_pass(state0, orders, pc, mc, label, n):
    t0 = time.perf_counter()
    final_state, n_fills, qty_ticks = _replay_and_reduce(state0, orders, pc, mc)
    for leaf in jax.tree_util.tree_leaves((final_state, n_fills, qty_ticks)):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    elapsed = time.perf_counter() - t0
    print(f"{label:<14}{elapsed:7.2f} s   ({n/elapsed:>10,.0f} orders/s)")
    return n_fills, qty_ticks, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract-id", type=int, default=81)
    ap.add_argument("--delivery-date", default="2026-01-12")
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="cpu",
                    help="JAX backend; pinned via JAX_PLATFORMS at import time. "
                         "Default is CPU (simulator runs default to CPU; "
                         "pass --backend gpu to opt in).")
    args = ap.parse_args()

    active_backend = jax.default_backend()
    print(f"OMIE benchmark — {args.delivery_date} contract {args.contract_id}"
          f" [backend={active_backend}]")
    print("-" * 60)

    t0 = time.perf_counter()
    cfg = OMIEDataConfig(
        data_path="data/rawOMIE",
        contract_id=args.contract_id,
        delivery_date=args.delivery_date,
    )
    orders = load_omie_orders(cfg)
    n = orders.shape[0]
    print(f"data load:    {time.perf_counter()-t0:7.2f} s   ({n:,} orders)")

    contract = make_open_contract(args.contract_id)
    book = ContractBook(contract, OMIEProductConfig(), OMIEMarketConfig())
    pc, mc = book._product_config, book._market_config

    state0 = book.init()
    time_pass(state0, orders, pc, mc, "cold (+JIT):", n)
    n_fills, qty_ticks, _ = time_pass(state0, orders, pc, mc, "warm:", n)
    traded_mw = int(qty_ticks) * pc.qty_tick

    print("-" * 60)
    print(f"fills:        {int(n_fills):>10,}")
    print(f"traded vol:   {traded_mw:>13,.2f} MW")


if __name__ == "__main__":
    main()
