"""Tests for M7.2d — Data Download Utility.

Behavioral contracts for ``scripts/download_omie_data.py``, a CLI tool
that downloads OMIE monthly zip archives and extracts daily flat files
(``orders_YYYYMMDD.1``, ``trades_YYYYMMDD.1``) into an output directory.

All tests run offline.  ``urllib.request.urlretrieve`` is patched so no
real network calls are made.  Zip fixtures are built in ``tmp_path``
with synthetic ``.1`` files and decoy entries.

Source of truth:
  - ``docs/omiecid/market-spec.md`` §13  (URLs, file naming)
  - ``docs/omiecid/agile-plan.md`` M7.2d (story definition)
  - ``docs/omiecid/sprint-status.md``    (architecture review / test plan)
  - ``omiecid_simulator/jaxomiecid/omie_data_loader.py`` (path convention)
"""

import io
import os
import re
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError, URLError

import pytest

# ---------------------------------------------------------------------------
# Guarded import — tests collect cleanly before implementation exists
# ---------------------------------------------------------------------------
try:
    from scripts.download_omie_data import download_month  # noqa: F401

    _MODULE_MISSING = False
except (ImportError, ModuleNotFoundError):
    _MODULE_MISSING = True

_skip_if_missing = pytest.mark.skipif(
    _MODULE_MISSING,
    reason="scripts.download_omie_data not yet implemented",
)

# We also try to import the CLI arg parser so we can test argparse behavior.
try:
    from scripts.download_omie_data import build_parser  # noqa: F401

    _PARSER_MISSING = False
except (ImportError, ModuleNotFoundError, AttributeError):
    _PARSER_MISSING = True

_skip_if_parser_missing = pytest.mark.skipif(
    _PARSER_MISSING,
    reason="scripts.download_omie_data.build_parser not yet implemented",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OMIE_BASE_URL = "https://www.omie.es/es/file-download"

# Expected URL patterns from market-spec.md §13
ORDERS_URL_TEMPLATE = (
    f"{OMIE_BASE_URL}?parents=orders&filename=orders_{{month}}.zip"
)
TRADES_URL_TEMPLATE = (
    f"{OMIE_BASE_URL}?parents=trades&filename=trades_{{month}}.zip"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_zip_bytes(file_entries: dict[str, bytes]) -> bytes:
    """Build an in-memory zip archive.

    Args:
        file_entries: mapping of ``{filename: content}`` to include in the zip.

    Returns:
        Raw bytes of the zip file.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in file_entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def orders_zip_path(tmp_path):
    """Write a zip containing two valid .1 files and one decoy, return path."""
    entries = {
        "orders_20250601.1": b"header\ndata;row;1\n",
        "orders_20250602.1": b"header\ndata;row;2\n",
        "orders_20250601.csv": b"decoy csv content",
        "readme.txt": b"decoy readme",
    }
    p = tmp_path / "orders_202506.zip"
    p.write_bytes(_make_zip_bytes(entries))
    return p


@pytest.fixture
def trades_zip_path(tmp_path):
    """Write a zip containing one valid .1 file and one decoy, return path."""
    entries = {
        "trades_20250601.1": b"header\ntrade;row;1\n",
        "trades_20250601.csv": b"decoy csv content",
    }
    p = tmp_path / "trades_202506.zip"
    p.write_bytes(_make_zip_bytes(entries))
    return p


@pytest.fixture
def nested_zip_path(tmp_path):
    """Zip where .1 files are inside a subdirectory."""
    entries = {
        "subdir/orders_20250601.1": b"nested content\n",
        "subdir/decoy.txt": b"nested decoy",
    }
    p = tmp_path / "orders_nested.zip"
    p.write_bytes(_make_zip_bytes(entries))
    return p


def _patch_urlretrieve(zip_path):
    """Return a side_effect callable that copies *zip_path* to the destination.

    ``urlretrieve(url, dest)`` writes zip content to *dest*.
    """

    def _fake_retrieve(url, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(Path(zip_path).read_bytes())
        return dest, {}

    return _fake_retrieve


# ---------------------------------------------------------------------------
# 1. URL construction
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestURLConstruction:
    """download_month() must build correct URLs per market-spec.md §13."""

    def test_orders_url(self, tmp_path, orders_zip_path):
        """URL for orders matches the OMIE pattern."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(orders_zip_path)
            download_month("202506", str(tmp_path), data_type="orders", dry_run=False)

            # At least one call should use the orders URL pattern
            urls_called = [call.args[0] for call in mock_ret.call_args_list]
            expected = ORDERS_URL_TEMPLATE.format(month="202506")
            assert expected in urls_called, (
                f"Expected URL {expected!r} not found in calls: {urls_called}"
            )

    def test_trades_url(self, tmp_path, trades_zip_path):
        """URL for trades matches the OMIE pattern."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(trades_zip_path)
            download_month("202506", str(tmp_path), data_type="trades", dry_run=False)

            urls_called = [call.args[0] for call in mock_ret.call_args_list]
            expected = TRADES_URL_TEMPLATE.format(month="202506")
            assert expected in urls_called, (
                f"Expected URL {expected!r} not found in calls: {urls_called}"
            )


# ---------------------------------------------------------------------------
# 2. --months argument parsing
# ---------------------------------------------------------------------------

@_skip_if_parser_missing
class TestMonthsArgParsing:
    """CLI accepts one or more YYYYMM values via --months."""

    def test_single_month(self):
        parser = build_parser()
        args = parser.parse_args(["--months", "202506"])
        assert args.months == ["202506"]

    def test_multiple_months(self):
        parser = build_parser()
        args = parser.parse_args(["--months", "202506", "202507", "202508"])
        assert args.months == ["202506", "202507", "202508"]


# ---------------------------------------------------------------------------
# 3. --date argument parsing
# ---------------------------------------------------------------------------

@_skip_if_parser_missing
class TestDateArgParsing:
    """CLI accepts --date YYYY-MM-DD and derives YYYYMM."""

    def test_date_derives_month(self):
        """--date 2025-06-15 should result in month '202506'."""
        parser = build_parser()
        args = parser.parse_args(["--date", "2025-06-15"])
        # The parser should store the derived month or the raw date.
        # We check that either args.months contains the derived month,
        # or args.date holds the ISO date for later derivation.
        if hasattr(args, "months") and args.months:
            assert "202506" in args.months
        elif hasattr(args, "date") and args.date:
            # Date stored as-is; derivation happens later
            assert args.date == "2025-06-15"
        else:
            pytest.fail("--date did not populate args.months or args.date")

    def test_date_first_of_month(self):
        """--date 2025-01-01 derives month '202501'."""
        parser = build_parser()
        args = parser.parse_args(["--date", "2025-01-01"])
        if hasattr(args, "months") and args.months:
            assert "202501" in args.months
        elif hasattr(args, "date") and args.date:
            assert args.date == "2025-01-01"
        else:
            pytest.fail("--date did not populate args.months or args.date")


# ---------------------------------------------------------------------------
# 4. --months / --date mutual exclusivity
# ---------------------------------------------------------------------------

@_skip_if_parser_missing
class TestMutualExclusivity:
    """--months and --date cannot be used together."""

    def test_months_and_date_together_errors(self):
        """Providing both --months and --date must produce an argparse error."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--months", "202506", "--date", "2025-06-15"])
        # argparse exits with code 2 on usage errors
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 5. Idempotency — skip download when files exist
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestIdempotency:
    """download_month() skips download when matching .1 files already exist."""

    def test_skips_when_orders_exist(self, tmp_path):
        """If orders_202506*.1 files exist, download is skipped."""
        # Pre-create files that match the glob pattern
        (tmp_path / "orders_20250601.1").write_text("existing")
        (tmp_path / "orders_20250602.1").write_text("existing")

        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            download_month("202506", str(tmp_path), data_type="orders", dry_run=False)
            mock_ret.assert_not_called()

    def test_skips_when_trades_exist(self, tmp_path):
        """If trades_202506*.1 files exist, download is skipped."""
        (tmp_path / "trades_20250601.1").write_text("existing")

        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            download_month("202506", str(tmp_path), data_type="trades", dry_run=False)
            mock_ret.assert_not_called()

    def test_downloads_when_no_files_exist(self, tmp_path, orders_zip_path):
        """If no matching .1 files exist, download proceeds."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(orders_zip_path)
            download_month("202506", str(tmp_path), data_type="orders", dry_run=False)
            mock_ret.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Zip extraction — only .1 files extracted
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestZipExtractionDot1Only:
    """Only entries ending in .1 are extracted from the zip archive."""

    def test_dot1_files_extracted(self, tmp_path, orders_zip_path):
        """Files ending in .1 should appear in the output directory."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(orders_zip_path)
            result = download_month(
                "202506", str(tmp_path), data_type="orders", dry_run=False
            )

        extracted = {f.name for f in tmp_path.iterdir() if f.suffix == ".1"}
        assert "orders_20250601.1" in extracted
        assert "orders_20250602.1" in extracted

    def test_decoy_files_not_extracted(self, tmp_path, orders_zip_path):
        """Non-.1 entries (csv, txt) must NOT be extracted."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(orders_zip_path)
            download_month("202506", str(tmp_path), data_type="orders", dry_run=False)

        all_names = {f.name for f in tmp_path.iterdir() if f.is_file()}
        assert "orders_20250601.csv" not in all_names
        assert "readme.txt" not in all_names


# ---------------------------------------------------------------------------
# 7. Zip extraction — flat into output dir (no subdirectories)
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestZipExtractionFlat:
    """Extracted .1 files land directly in output_dir, not in subdirectories."""

    def test_no_subdirectory_created(self, tmp_path, nested_zip_path):
        """Even if the zip has subdirs, .1 files end up flat in output_dir."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(nested_zip_path)
            download_month("202506", str(tmp_path), data_type="orders", dry_run=False)

        # The .1 file should be directly in tmp_path, not in a subdir
        dot1_files = list(tmp_path.glob("*.1"))
        assert len(dot1_files) >= 1, "No .1 files found directly in output_dir"

        # Verify no nested directories were created (except .zips which is allowed)
        subdirs = [
            d for d in tmp_path.iterdir()
            if d.is_dir() and d.name != ".zips"
        ]
        assert subdirs == [], f"Unexpected subdirectories: {[d.name for d in subdirs]}"

    def test_flat_file_has_basename_only(self, tmp_path, nested_zip_path):
        """Extracted file name is the basename, not the full zip-internal path."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(nested_zip_path)
            download_month("202506", str(tmp_path), data_type="orders", dry_run=False)

        extracted_names = {f.name for f in tmp_path.glob("*.1")}
        assert "orders_20250601.1" in extracted_names


# ---------------------------------------------------------------------------
# 8. Dry-run mode — no network calls
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestDryRun:
    """--dry-run must not make any network calls."""

    def test_urlretrieve_not_called(self, tmp_path):
        """In dry-run mode, urlretrieve is never invoked."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            download_month(
                "202506", str(tmp_path), data_type="orders", dry_run=True
            )
            mock_ret.assert_not_called()

    def test_no_files_created(self, tmp_path):
        """Dry-run must not create any files in output_dir."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            download_month(
                "202506", str(tmp_path), data_type="orders", dry_run=True
            )

        # Only the directory itself should exist, no new files
        files = [f for f in tmp_path.iterdir() if f.is_file()]
        assert files == [], f"Unexpected files created in dry-run: {files}"


# ---------------------------------------------------------------------------
# 9. download_month() importable API
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestDownloadMonthAPI:
    """download_month is importable and callable with expected signature."""

    def test_importable(self):
        """from scripts.download_omie_data import download_month succeeds."""
        # If we get here, the guarded import at module top succeeded
        assert callable(download_month)

    def test_accepts_expected_arguments(self, tmp_path, orders_zip_path):
        """download_month(month, output_dir, data_type, dry_run) is callable."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(orders_zip_path)
            # Should not raise TypeError for unexpected keyword args
            result = download_month(
                month="202506",
                output_dir=str(tmp_path),
                data_type="orders",
                dry_run=False,
            )

        # Result should be a list (of extracted paths)
        assert isinstance(result, list), (
            f"download_month should return a list, got {type(result).__name__}"
        )

    def test_returns_extracted_paths(self, tmp_path, orders_zip_path):
        """Return value contains paths of extracted .1 files."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(orders_zip_path)
            result = download_month(
                "202506", str(tmp_path), data_type="orders", dry_run=False,
            )

        # Each returned path should be an existing .1 file
        for p in result:
            path = Path(p)
            assert path.exists(), f"Returned path does not exist: {p}"
            assert path.suffix == ".1", f"Returned path is not a .1 file: {p}"


# ---------------------------------------------------------------------------
# 10. HTTPError not retried
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestHTTPErrorNotRetried:
    """HTTP errors (e.g. 404) must raise immediately without retry."""

    def test_httperror_raises_immediately(self, tmp_path):
        """HTTPError is not retried; it propagates on first occurrence."""
        err = HTTPError(
            url="https://www.omie.es/es/file-download?parents=orders&filename=orders_202506.zip",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )
        with patch("scripts.download_omie_data.urlretrieve", side_effect=err) as mock_ret:
            with pytest.raises(HTTPError):
                download_month(
                    "202506", str(tmp_path), data_type="orders", dry_run=False
                )
            # Must have been called exactly once (no retry)
            assert mock_ret.call_count == 1

    def test_httperror_500_also_not_retried(self, tmp_path):
        """Server errors (500) are also not retried."""
        err = HTTPError(
            url="https://www.omie.es/es/file-download?parents=orders&filename=orders_202506.zip",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )
        with patch("scripts.download_omie_data.urlretrieve", side_effect=err) as mock_ret:
            with pytest.raises(HTTPError):
                download_month(
                    "202506", str(tmp_path), data_type="orders", dry_run=False
                )
            assert mock_ret.call_count == 1


# ---------------------------------------------------------------------------
# 11. URLError retried once, then raised
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestURLErrorRetriedOnce:
    """Network errors (URLError) are retried exactly once."""

    def test_urlerror_retried_then_raised(self, tmp_path):
        """URLError triggers one retry; if it fails again, the error propagates."""
        err = URLError(reason="Connection refused")
        with patch("scripts.download_omie_data.urlretrieve", side_effect=err) as mock_ret:
            with pytest.raises(URLError):
                download_month(
                    "202506", str(tmp_path), data_type="orders", dry_run=False
                )
            # First attempt + one retry = 2 calls
            assert mock_ret.call_count == 2

    def test_urlerror_succeeds_on_retry(self, tmp_path, orders_zip_path):
        """If URLError on first attempt but succeeds on retry, no error raised."""
        err = URLError(reason="Connection refused")
        call_count = {"n": 0}

        def _side_effect(url, dest):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise err
            # Succeed on second attempt
            return _patch_urlretrieve(orders_zip_path)(url, dest)

        with patch("scripts.download_omie_data.urlretrieve", side_effect=_side_effect) as mock_ret:
            result = download_month(
                "202506", str(tmp_path), data_type="orders", dry_run=False
            )
            # Should have been called twice (initial + retry)
            assert mock_ret.call_count == 2

        # Files should be extracted successfully
        dot1_files = list(tmp_path.glob("*.1"))
        assert len(dot1_files) >= 1


# ---------------------------------------------------------------------------
# 12. Extracted filenames match loader pattern
# ---------------------------------------------------------------------------

@_skip_if_missing
class TestExtractedFilenamesMatchLoader:
    """Extracted filenames must match the pattern used by omie_data_loader.py.

    The loader constructs paths as:
        f"{data_path}/orders_{date_nodash}.1"
        f"{data_path}/trades_{date_nodash}.1"

    where date_nodash is YYYYMMDD (delivery date with no separators).
    """

    # Patterns derived from omie_data_loader.py path construction
    ORDERS_PATTERN = re.compile(r"^orders_\d{8}\.1$")
    TRADES_PATTERN = re.compile(r"^trades_\d{8}\.1$")

    def test_orders_filenames_match_loader(self, tmp_path, orders_zip_path):
        """All extracted order files match orders_YYYYMMDD.1 pattern."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(orders_zip_path)
            download_month("202506", str(tmp_path), data_type="orders", dry_run=False)

        dot1_files = [f.name for f in tmp_path.glob("*.1")]
        assert len(dot1_files) >= 1, "No .1 files extracted"
        for name in dot1_files:
            assert self.ORDERS_PATTERN.match(name), (
                f"Filename {name!r} does not match loader pattern orders_YYYYMMDD.1"
            )

    def test_trades_filenames_match_loader(self, tmp_path, trades_zip_path):
        """All extracted trade files match trades_YYYYMMDD.1 pattern."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(trades_zip_path)
            download_month("202506", str(tmp_path), data_type="trades", dry_run=False)

        dot1_files = [f.name for f in tmp_path.glob("*.1")]
        assert len(dot1_files) >= 1, "No .1 files extracted"
        for name in dot1_files:
            assert self.TRADES_PATTERN.match(name), (
                f"Filename {name!r} does not match loader pattern trades_YYYYMMDD.1"
            )

    def test_date_digits_are_valid(self, tmp_path, orders_zip_path):
        """The YYYYMMDD portion represents a plausible date within the month."""
        with patch("scripts.download_omie_data.urlretrieve") as mock_ret:
            mock_ret.side_effect = _patch_urlretrieve(orders_zip_path)
            download_month("202506", str(tmp_path), data_type="orders", dry_run=False)

        for f in tmp_path.glob("orders_*.1"):
            # Extract YYYYMMDD from filename
            date_str = f.stem.split("_")[1]  # "20250601"
            assert len(date_str) == 8
            year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
            assert year == "2025"
            assert month == "06"
            assert 1 <= int(day) <= 31
