"""Locked tests for WF-15-FOLLOWUP-2 · OMIE-MARGINALPDBC-COLUMN-FIX-1.

A latent data-correctness bug in the merged WF-14 day-ahead loader:
``omiecid_simulator/jaxomiecid/omie_data_loader.py::parse_marginalpdbc_file`` reads
the FIRST price column (``fields[4]``, "col A") as the Spanish (ES) zone price.
The WF-15 owner cross-checked ``marginalpdbc`` against the Portugal-only
``marginalpdbcpt`` file on the 18 divergent June-2025 days and found the columns
are BACKWARDS:

    col A (field index 4) = PORTUGAL (PT)
    col B (field index 5) = SPAIN    (ES)   <- the correct Spanish anchor

The defect was LATENT for the WF-15 run because col A == col B (ES/PT coupled)
on all 30 June days for contract-0 / delivery hour-1, so that run's anchors were
correct; but the parser silently corrupts the anchor for ANY contract whose
delivery hour is a divergent hour.

These are LOCKED tests defining the fix's acceptance bar:
  * ``test_marginalpdbc_reads_spanish_col_b`` is the LOAD-BEARING test. Its
    fixture VARIES the coupling that masked the defect (col A != col B), per the
    lesson that a fixture holding an incidental constant constant masks defects.
    It is RED against the current (backwards, col-A) parser and GREEN after the
    fix selects col B (field index 5).
  * ``test_marginalpdbc_coupled_row_unchanged`` is a regression guard on a
    coupled row (col A == col B): GREEN before AND after the fix (the WF-15
    run's contract-0/hour-1 anchors stay correct).

Return-shape assumption encoded (from ``parse_marginalpdbc_file``,
``omie_data_loader.py:1315-1316``): the parser returns a plain ``dict`` mapping
``hour(1..24) -> price_float`` (the Spanish-zone marginal price, EUR/MWh,
comma-decimal parsed to float). Raw row layout (dayahead-sourcing-note.md §1):

    MARGINALPDBC;                          <- header token line (skipped)
    YYYY;MM;DD;<hour 1..24>;<col A>;<col B>;
    *                                      <- terminator (skipped)

No production code is modified by this file; the parser fix is the developer's.
"""

from __future__ import annotations

import pytest

from omiecid_simulator.jaxomiecid.omie_data_loader import parse_marginalpdbc_file


# ---------------------------------------------------------------------------
# Helper — synthetic marginalpdbc flat file with EXPLICIT (col_a, col_b).
#
# Written with (col A, col B) named exactly by the corrected zone mapping:
#   col A (field index 4) = PORTUGAL, col B (field index 5) = SPAIN.
# The writer keeps the two columns as INDEPENDENT inputs so a caller can make
# them diverge (the load-bearing case) or couple them (the regression guard).
# Prices are written comma-decimal, verbatim, matching the raw OMIE convention.
# ---------------------------------------------------------------------------
def _write_marginalpdbc(path, hour_to_cols):
    """Write a synthetic ``marginalpdbc_YYYYMMDD.1`` file.

    ``hour_to_cols`` maps ``hour(1..24) -> (col_a_pt, col_b_es)`` where each
    element is a comma-decimal string written verbatim into, respectively,
    field index 4 (col A = PT) and field index 5 (col B = ES). Emits the
    ``MARGINALPDBC;`` header, one ``YYYY;MM;DD;h;colA;colB;`` row per hour, and
    the ``*`` terminator.
    """
    lines = ["MARGINALPDBC;"]
    for h in sorted(hour_to_cols):
        col_a_pt, col_b_es = hour_to_cols[h]
        lines.append(f"2025;06;01;{h};{col_a_pt};{col_b_es};")
    lines.append("*")
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return str(path)


# ===========================================================================
# 1. LOAD-BEARING: the Spanish price is col B (field index 5), not col A.
#    Fixture VARIES the coupling that masked the defect (col A != col B).
# ===========================================================================
class TestMarginalpdbcSpanishColumn:
    def test_marginalpdbc_reads_spanish_col_b(self, tmp_path):
        """On a divergent row (col A != col B) the parser returns col B (ES).

        This is the load-bearing test. The fixture deliberately breaks the
        ES/PT coupling that made the bug latent on the WF-15 split: col A (PT)
        and col B (ES) hold DIFFERENT values in every hour, so returning the
        wrong column is observable.

        RED against the current (backwards, col-A) parser; GREEN once the fix
        selects field index 5 (col B = Spanish zone).
        """
        # col A = PORTUGAL (must NOT be returned), col B = SPAIN (must be).
        f = _write_marginalpdbc(
            tmp_path / "marginalpdbc_20250601.1",
            {
                9:  ("70,00", "85,50"),   # divergent hour: PT 70.00, ES 85.50
                12: ("55,25", "99,99"),   # divergent hour: PT 55.25, ES 99.99
                21: ("40,10", "63,00"),   # divergent hour: PT 40.10, ES 63.00
            },
        )
        curve = parse_marginalpdbc_file(f)

        # The Spanish (col B) value is what the anchor must carry.
        assert curve[9] == pytest.approx(85.50)
        assert curve[12] == pytest.approx(99.99)
        assert curve[21] == pytest.approx(63.00)

        # And the Portuguese (col A) value must NOT be what is returned.
        assert curve[9] != pytest.approx(70.00)
        assert curve[12] != pytest.approx(55.25)
        assert curve[21] != pytest.approx(40.10)

    def test_marginalpdbc_col_b_selected_across_full_day(self, tmp_path):
        """Every hour's returned value is the col-B (ES) price, on a fully
        divergent 24-hour curve.

        A per-hour divergent curve where col B = col A + 100 for every hour:
        the parser must return the col-B branch uniformly, not just for one
        cherry-picked hour. RED under the col-A parser (returns the +0 branch).
        """
        f = _write_marginalpdbc(
            tmp_path / "marginalpdbc_20250601.1",
            {h: (f"{10 + h},0", f"{110 + h},0") for h in range(1, 25)},
        )
        curve = parse_marginalpdbc_file(f)

        assert set(curve.keys()) == set(range(1, 25))
        for h in range(1, 25):
            assert curve[h] == pytest.approx(110.0 + h)   # col B (ES)
            assert curve[h] != pytest.approx(10.0 + h)    # col A (PT)


# ===========================================================================
# 2. Regression guard: a coupled row (col A == col B) is unchanged.
#    GREEN before AND after the fix (WF-15 anchors stay correct).
# ===========================================================================
class TestMarginalpdbcCoupledRegression:
    def test_marginalpdbc_coupled_row_unchanged(self, tmp_path):
        """On coupled rows (col A == col B) the returned value is unchanged.

        This mirrors the WF-15 frozen-split condition (ES/PT coupled on
        contract-0 / hour-1 across all 30 June days): the anchor is the shared
        value regardless of which column the parser selects. This guard stays
        GREEN under both the backwards parser and the corrected one, protecting
        the already-shipped WF-15 anchors from regression.
        """
        f = _write_marginalpdbc(
            tmp_path / "marginalpdbc_20250601.1",
            {
                1: ("20,48", "20,48"),   # coupled (the WF-15 contract-0 hour)
                2: ("21,00", "21,00"),   # coupled
            },
        )
        curve = parse_marginalpdbc_file(f)

        assert curve[1] == pytest.approx(20.48)
        assert curve[2] == pytest.approx(21.00)
