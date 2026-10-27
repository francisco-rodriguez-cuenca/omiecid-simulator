#!/usr/bin/env python3
"""Download OMIE monthly zip archives and extract daily flat files.

Downloads order and trade data from the OMIE public file download endpoint
and extracts only the ``.1`` daily flat files into a target directory,
matching the path convention expected by ``omie_data_loader.py``.

Usage examples::

    # Download orders for June 2025 into data/rawOMIE/
    python -m scripts.download_omie_data --months 202506 --data-type orders

    # Download trades for multiple months
    python -m scripts.download_omie_data --months 202506 202507 --data-type trades

    # Download the day-ahead marginal hourly price series (Spanish zone)
    python -m scripts.download_omie_data --months 202506 --data-type marginalpdbc

    # Download by date (derives YYYYMM automatically)
    python -m scripts.download_omie_data --date 2025-06-15 --data-type orders

    # Dry-run (no network calls)
    python -m scripts.download_omie_data --months 202506 --dry-run

Source of truth:
    - ``docs/omiecid/market-spec.md`` section 13
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OMIE_BASE_URL = "https://www.omie.es/es/file-download"

DEFAULT_OUTPUT_DIR = "data/rawOMIE"
DEFAULT_DATA_TYPE = "orders"

# ---------------------------------------------------------------------------
# Core download function
# ---------------------------------------------------------------------------


def download_month(
    month: str,
    output_dir: str,
    data_type: str = DEFAULT_DATA_TYPE,
    dry_run: bool = False,
) -> list[str]:
    """Download and extract OMIE data for a single month.

    Args:
        month: Month string in ``YYYYMM`` format (e.g. ``"202506"``).
        output_dir: Directory where ``.1`` files are extracted.
        data_type: ``"orders"`` or ``"trades"``.
        dry_run: If True, log planned actions but perform no downloads.

    Returns:
        List of absolute paths to extracted ``.1`` files.  Empty list if
        the month was skipped (idempotency) or dry-run mode was active.

    Raises:
        HTTPError: On HTTP errors (404, 500, etc.) — not retried.
        URLError: On network errors — retried once, then raised.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Idempotency: skip if .1 files already exist for this month ---------
    existing = glob.glob(str(out / f"{data_type}_{month}*.1"))
    if existing:
        logger.info(
            "Skipping %s %s: %d .1 file(s) already exist",
            data_type,
            month,
            len(existing),
        )
        return []

    # --- Dry-run: report and return -----------------------------------------
    url = f"{OMIE_BASE_URL}?parents={data_type}&filename={data_type}_{month}.zip"
    if dry_run:
        logger.info("[dry-run] Would download %s -> %s", url, output_dir)
        return []

    # --- Download zip -------------------------------------------------------
    zips_dir = out / ".zips"
    zips_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = str(zips_dir / f"{data_type}_{month}.zip")

    _download_with_retry(url, zip_dest)

    # --- Extract .1 files flat into output_dir ------------------------------
    extracted: list[str] = []
    with zipfile.ZipFile(zip_dest, "r") as zf:
        for entry in zf.namelist():
            if entry.endswith(".1"):
                # Flatten: use only the basename, ignore zip-internal paths
                basename = os.path.basename(entry)
                dest_path = out / basename
                with zf.open(entry) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(str(dest_path))
                logger.info("Extracted %s", dest_path)

    logger.info(
        "Extracted %d .1 file(s) for %s %s", len(extracted), data_type, month
    )
    return extracted


# ---------------------------------------------------------------------------
# Download helper with single retry on URLError
# ---------------------------------------------------------------------------


def _download_with_retry(url: str, dest: str) -> None:
    """Download *url* to *dest*.  Retry once on URLError, not on HTTPError."""
    try:
        urlretrieve(url, dest)
    except HTTPError:
        # HTTP errors (404, 500) are not retried — re-raise immediately
        raise
    except URLError:
        # Network errors get a single retry
        logger.warning("URLError on %s — retrying once", url)
        urlretrieve(url, dest)  # if this also fails, let it propagate


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the download CLI."""
    parser = argparse.ArgumentParser(
        description="Download OMIE monthly zip archives and extract .1 files.",
    )

    # Mutually exclusive: --months or --date
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--months",
        nargs="+",
        metavar="YYYYMM",
        help="One or more months in YYYYMM format.",
    )
    group.add_argument(
        "--date",
        type=str,
        metavar="YYYY-MM-DD",
        help="ISO date; the YYYYMM month is derived automatically.",
    )

    parser.add_argument(
        "--data-type",
        choices=["orders", "trades", "marginalpdbc"],
        default=DEFAULT_DATA_TYPE,
        help=(
            "Data type to download (default: orders). ``marginalpdbc`` fetches "
            "the OMIE day-ahead marginal hourly price series (Spanish zone, "
            "EUR/MWh; day-ahead price anchor source)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without downloading.",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep downloaded zip archives after extraction.",
    )

    return parser


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    # Derive months list
    if args.months:
        months = args.months
    else:
        # --date: derive YYYYMM
        date_str = args.date  # "YYYY-MM-DD"
        months = [date_str[:4] + date_str[5:7]]

    for month in months:
        download_month(
            month=month,
            output_dir=args.output_dir,
            data_type=args.data_type,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
