"""Independent audit tests for WF-14 · OMIE-DAYAHEAD-INGEST-1 — the day-ahead
data utilities.

Post-implementation, black-box verification of the day-ahead ingest
utilities, derived independently from the sprint spec (agile-plan WF-14 card,
dayahead-sourcing-note §3) — NOT from the original locked test file. Focus
areas:

  * the CID->day-ahead alignment ``anchor(c, D) = marginalpdbc(D, (c//4)+1)`` at
    hour-boundary contract_ids (c=0/3 -> h1, c=4 -> h2, c=95 -> h24);
  * ``parse_marginalpdbc_file`` Spanish-price column (col B, field index 5) +
    comma-decimal parsing (re-authored per WF-15-FOLLOWUP-2, see note below);
  * the ``build_day_ahead_anchors`` call-form guards.

All tests are fast-suite (< 1s): synthetic in-test fixtures only, no
``saved_npz/omie/`` cache, no real OMIE replay.

Domain-compliance note (RESOLVED by WF-15-FOLLOWUP-2): the WF-14 audit
originally ASSUMED col A == Spanish zone (per the implementation's docstring;
the sourcing-note §1 flagged it as "verify against a real sample"). The WF-15
owner cross-checked ``marginalpdbc`` against the Portugal-only ``marginalpdbcpt``
file on the 18 divergent June-2025 days and found the columns are BACKWARDS:
**col A (field index 4) = PORTUGAL, col B (field index 5) = SPAIN**. This audit's
``TestParseMarginalpdbc`` is re-authored in lockstep with the parser fix to
assert col B (ES) on a divergent fixture; ``test_spanish_price_is_col_b_and_comma_decimal``
is RED until the developer flips ``parse_marginalpdbc_file`` to field index 5,
and GREEN thereafter. The other marginalpdbc fixtures in this file were made
COUPLED (col A == col B) because those tests (header/terminator skipping,
malformed-row filtering, CID->hour alignment) are column-agnostic and must stay
green across the fix.
"""

from __future__ import annotations

import pytest

from omiecid_simulator.jaxomiecid.omie_data_loader import (
    parse_marginalpdbc_file,
    day_ahead_anchor,
    build_day_ahead_anchors,
)
from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_marginalpdbc(path, hour_to_prices):
    """Write a synthetic marginalpdbc flat file.

    ``hour_to_prices`` maps hour(1..24) -> (price_col_a_pt, price_col_b_es)
    tuples (comma decimals written verbatim). The raw row layout is
    ``YYYY;MM;DD;h;<col A = PORTUGAL>;<col B = SPAIN>;`` — the WF-15-FOLLOWUP-2
    finding cross-checked ``marginalpdbc`` against ``marginalpdbcpt`` on the 18
    divergent June-2025 days and confirmed **col A = PT, col B = ES** (the
    parser reads the Spanish price from col B / field index 5). Emits the
    ``MARGINALPDBC;`` header, one row per hour, and the ``*`` terminator.
    """
    lines = ["MARGINALPDBC;"]
    for h in sorted(hour_to_prices):
        col_a_pt, col_b_es = hour_to_prices[h]
        lines.append(f"2025;06;01;{h};{col_a_pt};{col_b_es};")
    lines.append("*")
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return str(path)


# ===========================================================================
# 1. parse_marginalpdbc_file — Spanish price is col B (index 5) + comma decimals
#
# WF-15-FOLLOWUP-2 correction: col A (field index 4) = PORTUGAL, col B (field
# index 5) = SPAIN. This test formerly pinned the col-A value as ES on a
# divergent fixture (locking the now-confirmed-backwards assumption); it is
# re-authored in lockstep with the parser fix to assert col B (ES). It is RED
# against the current (backwards, col-A) parser and becomes GREEN together with
# the developer's flip to field index 5.
# ===========================================================================

class TestParseMarginalpdbc:
    def test_spanish_price_is_col_b_and_comma_decimal(self, tmp_path):
        """Col B (SECOND price) is the Spanish-zone price; comma is the decimal.

        Divergent fixture (col A = PT != col B = ES) so returning the wrong
        column is observable. RED under the backwards col-A parser; GREEN once
        the parser reads field index 5 (col B = Spanish zone).
        """
        f = _write_marginalpdbc(
            tmp_path / "marginalpdbc_20250601.1",
            # (col A = PT, col B = ES): Spanish price sits in col B.
            {1: ("48,10", "50,25"), 2: ("49,00", "51,00")},
        )
        curve = parse_marginalpdbc_file(f)
        assert curve[1] == pytest.approx(50.25)   # col B = ES
        assert curve[2] == pytest.approx(51.00)   # col B = ES
        # The PT column (col A) must NOT be what is returned.
        assert curve[1] != pytest.approx(48.10)

    def test_header_and_terminator_skipped(self, tmp_path):
        """The MARGINALPDBC header token and the ``*`` terminator are not rows."""
        # Coupled fixture (col A == col B): this test is about row-vs-header
        # structure, not column semantics, so it is robust to the col-A/col-B
        # selection and stays green before AND after the WF-15-FOLLOWUP-2 fix.
        f = _write_marginalpdbc(
            tmp_path / "marginalpdbc_20250601.1",
            {h: (f"{100 + h},0", f"{100 + h},0") for h in range(1, 25)},
        )
        curve = parse_marginalpdbc_file(f)
        assert set(curve.keys()) == set(range(1, 25))
        assert curve[24] == pytest.approx(124.0)

    def test_short_and_nonhour_rows_ignored(self, tmp_path):
        """A malformed / too-short row is skipped, not fatal or mis-keyed."""
        # Valid rows use coupled prices (col A == col B) so this row-filtering
        # test is column-agnostic and stays green across the col-B fix.
        p = tmp_path / "marginalpdbc_20250601.1"
        p.write_text(
            "MARGINALPDBC;\n"
            "2025;06;01;1;50,0;50,0;\n"
            "garbage;row;\n"                      # < 5 fields -> skipped
            "2025;06;01;X;9,9;9,9;\n"             # non-int hour -> skipped
            "2025;06;01;2;51,0;51,0;\n"
            "*\n",
            encoding="latin-1",
        )
        curve = parse_marginalpdbc_file(str(p))
        assert curve == {1: pytest.approx(50.0), 2: pytest.approx(51.0)}


# ===========================================================================
# 2. Alignment anchor(c, D) = marginalpdbc(D, (c//4)+1) at hour boundaries
# ===========================================================================

class TestAlignmentFormula:
    @pytest.fixture
    def curve_file(self, tmp_path):
        # hour h -> ES price 100+h so the hour is recoverable from the value.
        # Coupled columns (col A == col B): this suite tests the CID->hour
        # alignment, not the ES/PT column selection, so it is column-agnostic
        # and stays green before AND after the WF-15-FOLLOWUP-2 col-B fix.
        _write_marginalpdbc(
            tmp_path / "marginalpdbc_20250601.1",
            {h: (f"{100 + h},0", f"{100 + h},0") for h in range(1, 25)},
        )
        return str(tmp_path)

    @pytest.mark.parametrize(
        "contract_id, expected_hour",
        [
            (0, 1),    # 00:00-00:15 -> hour 1
            (3, 1),    # 00:45-01:00 -> still hour 1 (last quarter of h1)
            (4, 2),    # 01:00-01:15 -> hour 2 (boundary crossing)
            (7, 2),    # last quarter of hour 2
            (8, 3),
            (92, 24),  # 23:00-23:15 -> hour 24
            (95, 24),  # 23:45-00:00 -> hour 24 (top of the range)
        ],
    )
    def test_cid_maps_to_correct_hour(self, curve_file, contract_id, expected_hour):
        got = day_ahead_anchor(contract_id, "2025-06-01", curve_file)
        assert got == pytest.approx(100.0 + expected_hour)

    def test_four_consecutive_quarters_share_one_hour(self, curve_file):
        """Contracts 4k..4k+3 broadcast the SAME hourly day-ahead price."""
        vals = {day_ahead_anchor(c, "2025-06-01", curve_file) for c in (8, 9, 10, 11)}
        assert len(vals) == 1  # all hour-3 -> 103.0

    def test_missing_hour_raises(self, tmp_path):
        """A contract whose hour is absent from the curve raises, not silent-0."""
        f_dir = str(tmp_path)
        _write_marginalpdbc(
            tmp_path / "marginalpdbc_20250601.1",
            {1: ("50,0", "48,0"), 2: ("51,0", "49,0")},
        )
        # contract_id 8 -> hour 3, not present.
        with pytest.raises(ValueError):
            day_ahead_anchor(8, "2025-06-01", f_dir)


# ===========================================================================
# 3. build_day_ahead_anchors — call-form guards (cheap, no file access)
# ===========================================================================

class TestBuildAnchorsGuards:
    def test_mixed_call_forms_rejected(self):
        cfg = OMIEDataConfig(
            data_path="/nonexistent", contract_id=0, delivery_date="2025-06-01"
        )
        with pytest.raises(ValueError):
            build_day_ahead_anchors(
                [cfg], data_path="/x", contract_ids=[0], delivery_dates=["2025-06-01"]
            )

    def test_neither_call_form_rejected(self):
        with pytest.raises(ValueError):
            build_day_ahead_anchors()

    def test_empty_config_list_rejected(self):
        with pytest.raises(ValueError):
            build_day_ahead_anchors(configs=[])
