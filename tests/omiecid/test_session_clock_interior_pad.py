"""Locked contract for OMIE-SESSION-CLOCK-INTERIOR-PAD-GUARD-1.

Story card: ``docs/omiecid/agile-plan.md`` "#### Story
OMIE-SESSION-CLOCK-INTERIOR-PAD-GUARD-1" (v5.175: decisions D1-D6, revised
Test-first contract (1)-(4)).

Contract under test (constructor only, ``session_clock="got_relative"`` only):

  * A pad row is a row whose column [6] timestamp is ``<= 0`` (D1); order_id
    is ignored.
  * For each tape ``b`` with last real row index ``L >= 0`` and
    ``ops = orders_per_step``, every ALIGNED slice ``k`` with
    ``0 <= k < L // ops`` (rows ``k*ops .. k*ops+ops-1``) must hold at least
    one real row; otherwise ``Session(...)`` raises ``ValueError`` (D2). A tape
    with no real row is exempt. A pad row inside a slice that also holds a real
    row is harmless and must NOT be rejected.
  * Message (D3) names ``session_clock``, the FIRST offending tape (lowest
    ``b``), its FIRST offending slice (lowest ``k``), the row range and
    ``orders_per_step``: ``"session_clock='got_relative': tape {b} has an
    all-pad slice at step {k} (rows {k*ops}..{k*ops+ops-1},
    orders_per_step={ops}) before its last real row {L}; ..."``.
  * The guard is the LAST check (D4): a tape that also has too few trailing
    pads raises the existing trailing-pad message, not the new one.
  * ``span_scaled`` (the default) is unaffected.

MATCH-STRING CAUTION (card): the pre-existing trailing-pad message already
contains "the first all-pad slice never comes", so every RED test here matches
``all-pad slice at step {k}``, never bare ``all-pad slice``.

Runtime: every test is constructor-only (numpy guards, no ``step_env`` /
``reset`` call, no compile) except the real-data class, which loads 30 real
tapes and is ``@pytest.mark.slow``.

RELEASE NOTE (REL-07): the per-tape eval seam check (research ``omie_eval``)
was removed with that module. The two research got_relative training configs
are not shipped; they share the same data source and session keys (they
differ only in RL observation keys), so the real-data check inlines those
keys and dates once.
"""

from __future__ import annotations

import os
import pathlib
from datetime import date, timedelta
import re

import jax
import numpy as np
import pytest

from omiecid_simulator.omiecid.omie_config import OMIEProductConfig
from omiecid_simulator.omiecid.omie_session import Session

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "rawOMIE"
# Session keys and train dates of the research configs
# omie_settle_control_op4_session_clock{,_unitobs}_train.json (not shipped).
TRAIN_ENV_CFG = {"orders_per_step": 4, "gct_minutes_from_got": 460,
                 "session_clock": "got_relative", "got_seconds": 55_200}
TRAIN_CONTRACT_IDS = [0]
TRAIN_DATES = [f"2025-06-{d:02d}" for d in range(1, 25)]   # 2025-06-01..24
HELD_OUT_DATES = ["2025-06-25", "2025-06-26", "2025-06-27",
                  "2025-06-28", "2025-06-29", "2025-06-30"]

G = 55_200            # GOT = 15:20 D-1 in the L1 frame (seconds)
GCT = 460             # contract 0, minutes from GOT
_PC = OMIEProductConfig()

R = "R"               # a real row (ts > 0, in [GOT, GCT))
P = "P"               # a sentinel pad row (ts = -1, order_id = 0, like the loader's)
Z = "Z"               # a ts = 0 row WITH order_id > 0 (D1: still a pad for the clock)
Q = "Q"               # a ts = -1 row WITH order_id > 0 (D1: still a pad for the clock)
O = "O"               # a real-ts row (ts > 0) with order_id = 0 (D1: real for the clock)

_PAD_ROW = [0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Memory hygiene (same pattern as tests/omiecid/test_obs_clock_scale.py).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _bound_inmemory_jit_cache():
    """Teardown-only reset of this process's in-memory JAX caches."""
    yield
    jax.clear_caches()


# ---------------------------------------------------------------------------
# Synthetic tape builders
# ---------------------------------------------------------------------------
def _real_row(i, ts, order_id=None):
    side = i % 2                                  # BUY 40 / SELL 60, never crossing
    price = 40.0 if side == 0 else 60.0
    oid = 1000 + i if order_id is None else order_id
    return [0, side, int(round((price - _PC.price_min) / _PC.price_tick)),
            int(round(1.0 / _PC.qty_tick)), oid, 777, int(ts), 0, 0, 0, 0, 0, 0]


def _rows(pattern):
    """``pattern``: a sequence of R/P/Z/Q/O codes. Real rows get strictly
    increasing in-frame timestamps (G + 60*i + 1 s)."""
    out = []
    for i, code in enumerate(pattern):
        ts = G + 60 * i + 1
        if code == R:
            out.append(_real_row(i, ts))
        elif code == P:
            out.append(list(_PAD_ROW))
        elif code == Z:
            out.append(_real_row(i, 0, order_id=5000 + i))
        elif code == Q:
            out.append(_real_row(i, -1, order_id=5000 + i))
        elif code == O:
            out.append(_real_row(i, ts, order_id=0))
        else:  # pragma: no cover - test-authoring error
            raise AssertionError(code)
    return out


def build_tapes(patterns, n_padded=None):
    """Stack row patterns into a (B, N, 13) int32 tape set, right-padded with
    sentinel pad rows to ``n_padded`` (default: the longest pattern)."""
    n = n_padded or max(len(p) for p in patterns)
    tapes = []
    for p in patterns:
        rows = _rows(p)
        rows += [list(_PAD_ROW) for _ in range(n - len(rows))]
        tapes.append(rows)
    return np.asarray(tapes, dtype=np.int32)


def got_relative_cfg(tapes, ops):
    return {"tapes": tapes, "orders_per_step": ops, "gct_minutes_from_got": GCT,
            "session_clock": "got_relative", "got_seconds": G}


def interior_msg(b, k):
    """D3 match fragment that ONLY the new guard can produce."""
    return rf"session_clock.*tape {b} .*all-pad slice at step {k}\b"


# Canonical shapes from the card --------------------------------------------
T5_SHAPE = [R] * 4 + [P] * 4 + [R] * 4 + [P] * 4            # step-1 all-pad slice
SHORT_GAP = [R, R, P, P, P, P, R] + [P] * 5                 # rejected @ ops 2,3
MISALIGNED_4_GAP = [R, R, P, P, P, P, R, R] + [P] * 4       # accepted @ ops 4
T2_SHAPE = [P, R, P, R, R, P, P, P, P, P, P, R] + [P] * 4   # pads inside real slices
CLEAN_12 = [R] * 12 + [P] * 4


def seven_gap(prefix):
    """``prefix`` real rows, a 7-row pad gap, 4 real rows, 4 trailing pads."""
    return [R] * prefix + [P] * 7 + [R] * 4 + [P] * 4


# ===========================================================================
# (1) RED -> GREEN: an aligned all-pad slice before the last real row raises
# ===========================================================================
class TestInteriorAllPadSliceRejected:
    def test_t5_shape_rejected_at_step_1(self):
        with pytest.raises(ValueError, match=interior_msg(0, 1)):
            Session(got_relative_cfg(build_tapes([T5_SHAPE]), 4))

    def test_t5_shape_message_names_orders_per_step(self):
        with pytest.raises(ValueError, match=interior_msg(0, 1)) as ei:
            Session(got_relative_cfg(build_tapes([T5_SHAPE]), 4))
        assert "orders_per_step" in str(ei.value)

    def test_t5_shape_message_carries_d3_fields(self):
        """D3: rows k*ops..k*ops+ops-1, orders_per_step=ops, last real row L."""
        with pytest.raises(ValueError, match=interior_msg(0, 1)) as ei:
            Session(got_relative_cfg(build_tapes([T5_SHAPE]), 4))
        msg = str(ei.value)
        assert "session_clock='got_relative'" in msg
        assert "rows 4..7" in msg
        assert "orders_per_step=4" in msg
        assert "last real row 11" in msg

    def test_t5_shape_as_second_tape_names_tape_1(self):
        tapes = build_tapes([CLEAN_12, T5_SHAPE])
        with pytest.raises(ValueError, match=interior_msg(1, 1)):
            Session(got_relative_cfg(tapes, 4))

    def test_lowest_offending_tape_is_reported(self):
        """D3: with two offending tapes, the lowest tape index is named."""
        other = [R] * 8 + [P] * 4 + [R] * 4 + [P] * 4           # offends at step 2
        tapes = build_tapes([CLEAN_12, other, T5_SHAPE])
        with pytest.raises(ValueError, match=interior_msg(1, 2)):
            Session(got_relative_cfg(tapes, 4))

    def test_lowest_offending_step_is_reported(self):
        """D3: a tape with all-pad slices at steps 1 and 3 names step 1."""
        tape = [R] * 4 + [P] * 4 + [R] * 4 + [P] * 4 + [R] * 4 + [P] * 4
        with pytest.raises(ValueError, match=interior_msg(0, 1)):
            Session(got_relative_cfg(build_tapes([tape]), 4))

    def test_leading_all_pad_slice_rejected_at_step_0(self):
        """k ranges over 0 <= k < L // ops, so a leading all-pad slice counts."""
        tape = [P] * 4 + [R] * 4 + [P] * 4
        with pytest.raises(ValueError, match=interior_msg(0, 0)):
            Session(got_relative_cfg(build_tapes([tape]), 4))

    @pytest.mark.parametrize("prefix,step", [(1, 1), (2, 1), (3, 1), (4, 1), (5, 2)])
    def test_seven_row_gap_at_ops_4_always_rejected(self, prefix, step):
        """A gap of >= 2*ops - 1 rows always covers an aligned slice (D2)."""
        with pytest.raises(ValueError, match=interior_msg(0, step)):
            Session(got_relative_cfg(build_tapes([seven_gap(prefix)]), 4))

    @pytest.mark.parametrize("ops", [2, 3])
    def test_short_gap_rejected_at_small_ops(self, ops):
        """Card probe: [R,R,P,P,P,P,R] + 5 pads is rejected at ops 2 and 3."""
        with pytest.raises(ValueError, match=interior_msg(0, 1)) as ei:
            Session(got_relative_cfg(build_tapes([SHORT_GAP]), ops))
        assert f"orders_per_step={ops}" in str(ei.value)

    def test_ops_1_single_interior_pad_row_rejected(self):
        """At ops 1 every slice is one row, so ONE interior pad row is an
        all-pad slice before the last real row."""
        tape = [R, R, P, R, P]
        with pytest.raises(ValueError, match=interior_msg(0, 2)):
            Session(got_relative_cfg(build_tapes([tape]), 1))


# ===========================================================================
# (1b) GREEN on both trees: guards against the filed, over-strict rule
# ===========================================================================
class TestHarmlessPadLayoutsAccepted:
    def test_t2_shape_pads_inside_real_slices_constructs(self):
        env = Session(got_relative_cfg(build_tapes([T2_SHAPE]), 4))
        assert env.session_clock == "got_relative"

    def test_misaligned_four_row_gap_at_ops_4_constructs(self):
        Session(got_relative_cfg(build_tapes([MISALIGNED_4_GAP]), 4))

    @pytest.mark.parametrize("ops", [4, 5])
    def test_short_gap_accepted_at_large_ops(self, ops):
        """Card probe: [R,R,P,P,P,P,R] + 5 pads is accepted at ops 4 and 5."""
        Session(got_relative_cfg(build_tapes([SHORT_GAP]), ops))

    @pytest.mark.parametrize("prefix", [1, 2, 3, 4, 5])
    def test_gap_shorter_than_ops_never_rejected(self, prefix):
        """A gap of < ops rows can never cover an aligned slice (D2)."""
        tape = [R] * prefix + [P] * 3 + [R] * 4 + [P] * 4
        Session(got_relative_cfg(build_tapes([tape]), 4))

    def test_all_pad_tape_alone_constructs(self):
        Session(got_relative_cfg(build_tapes([[P] * 8]), 4))

    def test_all_pad_tape_in_a_set_constructs(self):
        """A tape with no real row is exempt (D2), also next to real tapes."""
        tapes = build_tapes([CLEAN_12, [P] * 16])
        env = Session(got_relative_cfg(tapes, 4))
        assert env.n_tapes == 2

    def test_all_pad_slice_after_last_real_row_constructs(self):
        """Trailing all-pad slices (k >= L // ops) are the normal terminator."""
        tape = [R] * 6 + [P] * 10
        Session(got_relative_cfg(build_tapes([tape]), 4))

    def test_partial_last_real_slice_constructs(self):
        """The slice holding L may itself be mostly pad (k = L // ops is exempt)."""
        tape = [R] * 8 + [R, P, P, P] + [P] * 4
        Session(got_relative_cfg(build_tapes([tape]), 4))

    def test_real_ts_rows_with_zero_order_id_count_as_real(self):
        """D1: the mask is ts only; a ts > 0 row with order_id 0 is real."""
        tape = [R] * 4 + [O] * 4 + [R] * 4 + [P] * 4
        Session(got_relative_cfg(build_tapes([tape]), 4))


# ===========================================================================
# (1c) RED -> GREEN: D1 pad = ts <= 0, whatever the order_id
# ===========================================================================
class TestPadMaskIsTimestampOnly:
    @pytest.mark.parametrize("code", [Z, Q], ids=["ts0_oid_pos", "tsneg1_oid_pos"])
    def test_aligned_slice_of_nonpositive_ts_rows_with_order_ids_rejected(self, code):
        tape = [R] * 4 + [code] * 4 + [R] * 4 + [P] * 4
        with pytest.raises(ValueError, match=interior_msg(0, 1)):
            Session(got_relative_cfg(build_tapes([tape]), 4))


# ===========================================================================
# (1d) GREEN on both: D4 precedence — the trailing-pad guard fires first
# ===========================================================================
class TestTrailingPadGuardTakesPrecedence:
    TAPE = [R, R, R, R, P, P, P, P, R, P]   # all-pad slice at step 1 AND 1 trailing pad

    def test_raises_trailing_pad_message(self):
        with pytest.raises(ValueError, match="orders_per_step=4 trailing pad"):
            Session(got_relative_cfg(build_tapes([self.TAPE]), 4))

    def test_message_is_not_the_interior_message(self):
        with pytest.raises(ValueError) as ei:
            Session(got_relative_cfg(build_tapes([self.TAPE]), 4))
        assert not re.search(r"all-pad slice at step", str(ei.value))


# ===========================================================================
# (2) GREEN on both: span_scaled is unaffected (keys absent and explicit)
# ===========================================================================
SPAN_CASES = {
    "t5": ([T5_SHAPE], 4),
    "t5_second_tape": ([CLEAN_12, T5_SHAPE], 4),
    "two_offenders": ([CLEAN_12, [R] * 8 + [P] * 4 + [R] * 4 + [P] * 4, T5_SHAPE], 4),
    "two_offending_steps": ([[R] * 4 + [P] * 4 + [R] * 4 + [P] * 4 + [R] * 4 + [P] * 4], 4),
    "leading_all_pad": ([[P] * 4 + [R] * 4 + [P] * 4], 4),
    "seven_gap_1": ([seven_gap(1)], 4),
    "seven_gap_5": ([seven_gap(5)], 4),
    "short_gap_ops2": ([SHORT_GAP], 2),
    "short_gap_ops3": ([SHORT_GAP], 3),
    "ops1_single_pad": ([[R, R, P, R, P]], 1),
    "ts0_oid_pos": ([[R] * 4 + [Z] * 4 + [R] * 4 + [P] * 4], 4),
    "tsneg1_oid_pos": ([[R] * 4 + [Q] * 4 + [R] * 4 + [P] * 4], 4),
}


class TestSpanScaledUnaffected:
    @pytest.mark.parametrize("name", list(SPAN_CASES))
    def test_constructs_with_clock_keys_absent(self, name):
        patterns, ops = SPAN_CASES[name]
        env = Session({"tapes": build_tapes(patterns), "orders_per_step": ops,
                       "gct_minutes_from_got": GCT})
        assert env.session_clock == "span_scaled"

    @pytest.mark.parametrize("name", list(SPAN_CASES))
    def test_constructs_with_explicit_span_scaled(self, name):
        patterns, ops = SPAN_CASES[name]
        env = Session({"tapes": build_tapes(patterns), "orders_per_step": ops,
                       "gct_minutes_from_got": GCT, "session_clock": "span_scaled"})
        assert env.session_clock == "span_scaled"


# ===========================================================================
# (3) GREEN on both: real contract-0 tapes of the shipped got_relative configs
# ===========================================================================
@pytest.fixture(scope="module")
def real_cache(tmp_path_factory):
    if not DATA_PATH.is_dir():
        pytest.skip(f"real OMIE data not present ({DATA_PATH} missing)")
    return str(tmp_path_factory.mktemp("interior_pad_npz")) + "/"


def _first_missing_product_file(data_dir, dates):
    """First missing session file of the products of ``dates``, else ``None``.

    A product (D, k) reads ``orders_{D-1}.1``, ``orders_{D}.1``,
    ``trades_{D-1}.1`` and ``trades_{D}.1``; all four must be present for every
    delivery date, otherwise the loader refuses the product as incomplete.
    """
    for d_iso in dates:
        d = date.fromisoformat(d_iso)
        for kind in ("orders", "trades"):
            for day in (d - timedelta(days=1), d):
                name = f"{kind}_{day:%Y%m%d}.1"
                if not os.path.isfile(os.path.join(str(data_dir), name)):
                    return name
    return None


def _skip_unless_products_present(data_dir, dates):
    missing = _first_missing_product_file(data_dir, dates)
    if missing is not None:
        pytest.skip(f"real OMIE data incomplete: {missing} missing from {data_dir} "
                    "(partial data/rawOMIE, e.g. after tutorial T03)")


def _load_tapes(cache, contract_ids, dates):
    _skip_unless_products_present(DATA_PATH, dates)
    from omiecid_simulator.jaxomiecid.omie_data_loader import build_fixed_tape_set
    return np.asarray(build_fixed_tape_set(
        data_path=str(DATA_PATH), contract_ids=contract_ids, delivery_dates=dates,
        cache_path=cache, pad_extra=8,
    ))


@pytest.mark.slow
class TestRealTapesConstruct:
    def test_shipped_train_set_constructs(self, real_cache):
        """slow: loads the 24 real contract-0 train tapes (constructor only)."""
        tapes = _load_tapes(real_cache, TRAIN_CONTRACT_IDS, TRAIN_DATES)
        env_cfg = dict(TRAIN_ENV_CFG)
        assert env_cfg["session_clock"] == "got_relative"
        env_cfg["tapes"] = tapes
        env = Session(env_cfg)
        assert env.n_tapes == 24

    def test_held_out_tapes_construct_as_a_set_and_one_by_one(self, real_cache):
        """slow: loads the 6 real contract-0 held-out tapes (constructor only)."""
        tapes = _load_tapes(real_cache, [0], HELD_OUT_DATES)
        assert tapes.shape[0] == 6
        Session(got_relative_cfg(tapes, 4))
        for b in range(tapes.shape[0]):
            Session(got_relative_cfg(tapes[b:b + 1], 4))
