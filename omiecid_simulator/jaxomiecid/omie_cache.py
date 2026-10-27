"""OMIE NPZ Cache Layer.

Serialize and deserialize parsed OMIE market data arrays to/from NPZ files,
following the jaxlobster caching pattern. First load parses raw files and
saves NPZ; subsequent loads read from NPZ cache directly.

Public API:
    save_omie_cache(config, *, orders=None, trades=None, epoch_base=None) -> None
    load_omie_cache(config) -> dict | None

Schema versioning:
    Every NPZ payload written by ``save_omie_cache`` carries a
    ``schema_version`` key (currently ``CACHE_SCHEMA_VERSION = 4``).
    ``load_omie_cache`` refuses to return data written under a different
    schema version and treats such caches as if they were missing
    (returns ``None``), triggering a fresh parse.

    Version history:
      - v1: Agente-only ``agent_id`` (no schema_version field
        stored; rejected by the version check).
      - v2: ``(Agente, Unidad)`` pair-based ``agent_id``. Introduced
        to invalidate any stale v1 cache silently left on disk after the
        mapping algorithm changed.
      - v3: timestamps are now session-relative (seconds from UTC
        midnight of delivery_date). ``epoch_base`` scalar stored so the
        cache is self-describing and absolute timestamps can be recovered.
      - v4: orders/trades arrays are now ``int32`` with price,
        quantity, peak_qty offset-encoded on their tick grids and ``ppd``
        signed int-ticks. See ``omiecid_simulator.omiecid.encoding`` for the
        exact encode/decode functions.

    Per-product tapes (no bump; stays v4): the loaders now return
    one delivery product per tape (session files D-1 and D) with
    timestamps in seconds since D-1 00:00 (no modulo). ``load_omie_cache``
    has no runtime reader, so the version was deliberately left at 4; a
    v4 cache written before this change holds the older per-session tape
    and must not be reused for it.
"""

import os
from typing import Optional

import numpy as np

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig


# Current NPZ schema version. Bump whenever the on-disk array semantics
# change (e.g. agent_id algorithm, new column, timestamp epoch base).
CACHE_SCHEMA_VERSION: int = 4


def _cache_filename(config: OMIEDataConfig) -> str:
    """Build the NPZ filename for a given config, keyed by contract and date.

    Returns an absolute path under ``config.cache_path``.
    """
    cache_dir = os.path.normpath(config.cache_path)
    fname = f"omie_cache_c{config.contract_id}_{config.delivery_date}.npz"
    return os.path.join(cache_dir, fname)


def save_omie_cache(
    config: OMIEDataConfig,
    *,
    orders: Optional[np.ndarray] = None,
    trades: Optional[np.ndarray] = None,
    epoch_base: Optional[int] = None,
) -> None:
    """Save parsed OMIE arrays to a compressed NPZ file.

    Creates the cache directory if it does not already exist. The current
    ``CACHE_SCHEMA_VERSION`` is always embedded in the payload so that
    stale caches from earlier schema versions are rejected on reload.

    Args:
        config: Data configuration identifying the contract and delivery date.
        orders: Optional (N, 13) int32 array of parsed orders
            (tick-encoded).
        trades: Optional (M, 6) int32 array of parsed trades
            (tick-encoded).
        epoch_base: Optional UTC midnight Unix timestamp (int) for the
            delivery date. Stored as a scalar so the cache is self-describing
            and absolute timestamps can be recovered downstream.
    """
    cache_dir = os.path.normpath(config.cache_path)
    os.makedirs(cache_dir, exist_ok=True)

    arrays = {
        "schema_version": np.array(CACHE_SCHEMA_VERSION, dtype=np.int32),
    }
    if orders is not None:
        arrays["orders"] = orders
    if trades is not None:
        arrays["trades"] = trades
    if epoch_base is not None:
        arrays["epoch_base"] = np.array(epoch_base, dtype=np.int64)

    path = _cache_filename(config)
    np.savez_compressed(path, **arrays)


def load_omie_cache(config: OMIEDataConfig) -> Optional[dict]:
    """Load cached OMIE arrays from an NPZ file.

    Returns ``None`` in any of the following cases (treat as cache miss
    and re-parse from raw files):
      - the cache file does not exist on disk;
      - the file exists but has no ``schema_version`` field (v1
        legacy cache);
      - the stored ``schema_version`` does not match
        ``CACHE_SCHEMA_VERSION`` (a future schema, or a schema we have
        explicitly deprecated).

    Returns:
        A dict with keys ``"orders"`` and/or ``"trades"`` (whichever were
        saved), plus ``"schema_version"``, or ``None`` if no compatible
        cache is available for this config.
    """
    path = _cache_filename(config)
    if not os.path.isfile(path):
        return None

    with np.load(path) as data:
        if "schema_version" not in data.files:
            # Legacy (v1) cache; silently reject so the caller
            # re-parses from raw source files with the current mapping.
            return None
        try:
            stored_version = int(np.asarray(data["schema_version"]).item())
        except (ValueError, TypeError):
            return None
        if stored_version != CACHE_SCHEMA_VERSION:
            return None
        return {key: data[key] for key in data.files}
