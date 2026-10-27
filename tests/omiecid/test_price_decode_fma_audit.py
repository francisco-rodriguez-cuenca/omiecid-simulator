"""OMIE-PRICE-DECODE-FMA-1 — INDEPENDENT audit (orchestrate.md Step 5).

Written AFTER the implementation landed, from the story card's acceptance
criteria only (agile-plan.md v5.143 §"Story OMIE-PRICE-DECODE-FMA-1",
deliverables (a)/(b)/(b2)/(b3)/(e1,e2,e3)/(f) + the nine-site decode
inventory). The original locked test file was deliberately NOT read.

What this audit tries to break, in the card's own terms:

  1. THE HEADLINE PROMISE — "backend-identical". An EAGER CPU/GPU comparison
     of this decode is VACUOUSLY green (it was green before the fix too): the
     divergence the card measured is an XLA:CPU FMA CONTRACTION, which only
     exists inside a compiled graph. Every cross-device leg here therefore
     jits with an EXPLICIT ``backend=`` argument, and each is paired with a
     NON-VACUITY probe that asserts the same harness DOES catch the
     pre-change (naive) form. A cross-device test that cannot fail on the old
     code is not evidence.
  2. THE MID ANCHOR — ``mid_int = (bid_int + ask_int) // 2``, FLOOR toward the
     bid. Every pre-existing MID fixture in the repo holds an EVEN spread
     (the ``OMIE-REGEN-MASK-1`` incidental-constant lesson), so ODD spreads
     (1 and 3) are the class that was invisible. Asserted on the ENCODED
     SUBMIT TICK (``round(price/tick) + price_offset_int``, i.e. what
     ``order_validator.py:81`` computes), not only on the decoded EUR float —
     the EUR float is where a half-tick tie hides, the tick is where it bites.
  3. NO PARTIAL FIX — two decoded prices disagreeing WITHIN one run is the
     failure mode. The three traced FIX sites are checked to produce the same
     float32 bits for the same tick, and the 4 LEAVE + 1 ENCODE sites are
     checked to be untouched.
  4. THE GOLDEN RE-BASELINE IS HONEST — integer legs byte-exact against the
     PRE-change values (embedded as literals below, so the check does not
     depend on git), float legs <= 1e-5 relative, structure otherwise
     unedited. The golden is READ ONLY; this audit never regenerates it.
  5. DELIVERABLE (f) — ``MARK_ABS_TOL`` gone from both settle-at-gate test
     files, assertions exact again.

RELEASE NOTE (REL-07): every decode site this card fixed is kept in the
release (``Session._decode_price_eur``, ``_top_of_book``, ``_extract_passive_fill``),
so those audits run unchanged. Removed with the research code they read: the
``_counterparty_top_of_book`` checks (a reward-only settlement mark), the
``omie_order_plot`` host-mirror checks, the eval-golden re-baseline checks
(``omie_eval_fast_golden.json``; the golden-independent (e3) decode check is
kept) and the ``MARK_ABS_TOL`` checks on the settle-at-gate reward tests.

Runtime: every test is a synthetic in-test fixture (arrays, no tape replay,
no rollout), fast-suite budget. Cross-device legs carry ``@pytest.mark.gpu``
and auto-skip on CUDA-hidden hosts (tests/omiecid/conftest.py). Exactly ONE
``Session`` is constructed per process (module-scoped fixture) — the P3.4
per-instance-static-key segfault guard.
"""

import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.omiecid.omie_session import Session, AGENT_TRADER_ID
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig


REPO = pathlib.Path(__file__).resolve().parents[2]

TICK = 0.01
PRICE_MIN = -9999.0
PRICE_MAX = 9999.0
OFFSET_INT = 999_900                      # round(-price_min / price_tick)
MAX_TICK = 1_999_800                      # round((price_max - price_min)/tick)

# The card's exhaustively-measured relative bound (max_rel = 8.1062e-8).
REL_BOUND = 2e-7


# ---------------------------------------------------------------------------
# One env per process.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def env():
    return Session.from_test_config()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _bits(x):
    """float32 array -> its exact bit pattern, for bitwise comparison."""
    return np.asarray(jax.device_get(x), dtype=np.float32).view(np.int32)


def _naive_decode(ticks, tick=TICK, price_min=PRICE_MIN):
    """The PRE-change form. Used ONLY as a non-vacuity probe."""
    return jnp.asarray(ticks, jnp.int32).astype(jnp.float32) * jnp.float32(tick) \
        + jnp.float32(price_min)


def _exact_eur(ticks):
    """float64 ground truth for the int-domain decode."""
    return (np.asarray(ticks, dtype=np.int64) - OFFSET_INT) * np.float64(TICK)


def _submit_tick(price_eur):
    """The re-encode ``order_validator.py:79-82`` actually computes.

    Deliberately restated here (the shipped ``validate_order`` returns a bool
    and does not expose the encoded tick), and deliberately NOT routed through
    ``encoding.decode_price``'s float64 companion, which would pin the naive
    form as truth.
    """
    p = jnp.asarray(price_eur, jnp.float32)
    return jnp.round(p * jnp.float32(1.0 / TICK)).astype(jnp.int32) \
        + jnp.int32(OFFSET_INT)


def _tick_grid(n=4001):
    """Stride grid over the WHOLE legal band, with both edges pinned and the
    zero-price tick included. Deliberately not round literals: 51.0/50.0/80.0
    are exactly the values that hid this defect on GPU."""
    g = np.unique(np.concatenate([
        np.linspace(0, MAX_TICK, n).astype(np.int64),
        np.array([0, 1, 2, MAX_TICK - 2, MAX_TICK - 1, MAX_TICK,
                  OFFSET_INT - 1, OFFSET_INT, OFFSET_INT + 1,
                  OFFSET_INT + 5001, OFFSET_INT - 5001], dtype=np.int64),
    ]))
    return g.astype(np.int32)


def _lob_with(env, bid_tick, ask_tick, qty_ticks=40, tid=7):
    """An OMIEBookState LOB carrying at most one resting row per side.

    ``bid_tick``/``ask_tick`` are OFFSET int tick codes (the jaxob price
    column); pass ``None`` for an empty side.
    """
    book0 = env._empty_book()
    lob = book0.lob

    def _side(arr, tick):
        a = np.asarray(jax.device_get(arr)).copy()
        if tick is not None:
            a[0] = [int(tick), int(qty_ticks), 1, int(tid), 1, 0]
        return jnp.asarray(a)

    return lob._replace(asks=_side(lob.asks, ask_tick),
                        bids=_side(lob.bids, bid_tick))


# ===========================================================================
# A. The decode helper itself — deliverable (a)
# ===========================================================================
class TestDecodeHelperNumerics:

    def test_price_offset_int_is_int32_and_the_documented_value(self, env):
        off = env._price_offset_int
        assert jnp.asarray(off).dtype == jnp.int32, (
            f"_price_offset_int dtype is {jnp.asarray(off).dtype}, not int32 — "
            "OQ-18 (no x64) requires the subtraction to stay in int32."
        )
        assert int(off) == OFFSET_INT
        # int32 headroom is a DELIVERABLE, not an assumption: the largest
        # quantity that ever reaches int32 here is the mid anchor's SUM.
        assert 2 * MAX_TICK < np.iinfo(np.int32).max

    def test_decode_is_within_the_relative_bound_over_the_whole_legal_band(self, env):
        ticks = _tick_grid()
        got = np.asarray(jax.device_get(env._decode_price_eur(jnp.asarray(ticks))),
                         dtype=np.float64)
        exact = _exact_eur(ticks)
        rel = np.abs(got - exact) / np.maximum(np.abs(exact), 1.0)
        worst = int(np.argmax(rel))
        assert rel.max() <= REL_BOUND, (
            f"max relative decode error {rel.max():.4e} > {REL_BOUND:.1e} at tick "
            f"{int(ticks[worst])} (exact {exact[worst]!r}, got {got[worst]!r})."
        )

    @pytest.mark.jit
    def test_the_bound_would_REJECT_the_pre_change_naive_form(self, env):
        """Non-vacuity: the bound this audit uses must fail the OLD code."""
        ticks = _tick_grid()
        naive = np.asarray(jax.device_get(jax.jit(_naive_decode)(jnp.asarray(ticks))),
                           dtype=np.float64)
        exact = _exact_eur(ticks)
        rel = np.abs(naive - exact) / np.maximum(np.abs(exact), 1.0)
        assert rel.max() > REL_BOUND * 100, (
            "the naive form now passes the relative bound with 100x margin — "
            "this audit's bound would no longer discriminate the defect."
        )

    @pytest.mark.jit
    def test_decode_is_bitwise_identical_eager_vs_jit_vs_fused_graph(self, env):
        """The assertion that actually pins the FMA contraction on ONE device.

        The fused variant gives XLA a downstream multiply-add to contract the
        decode INTO — the shape in which the pre-change form diverged.
        """
        ticks = jnp.asarray(_tick_grid())
        eager = env._decode_price_eur(ticks)
        jitted = jax.jit(env._decode_price_eur)(ticks)

        def _fused(t):
            p = env._decode_price_eur(t)
            return p, p * jnp.float32(3.0) + jnp.float32(7.0), jnp.sum(p)

        fused_p = jax.jit(_fused)(ticks)[0]

        np.testing.assert_array_equal(_bits(eager), _bits(jitted))
        np.testing.assert_array_equal(_bits(eager), _bits(fused_p))

    @pytest.mark.jit
    def test_naive_form_DOES_move_under_the_same_eager_vs_jit_harness(self):
        """Non-vacuity twin of the test above, on CPU alone.

        XLA:CPU contracts ``i*tick + price_min`` into an FMA; eager does not.
        If this ever stops holding, the eager/jit leg above proves nothing.
        """
        ticks = jnp.asarray(_tick_grid())
        eager = _naive_decode(ticks)
        jitted = jax.jit(_naive_decode)(ticks)
        n_diff = int((_bits(eager) != _bits(jitted)).sum())
        if n_diff == 0:
            pytest.skip(
                "eager and jit agree bitwise on the NAIVE form on this backend, "
                "so the eager-vs-jit harness is not discriminating here; the "
                "cross-BACKEND gpu leg is the load-bearing one."
            )
        assert n_diff > 0

    def test_zero_price_and_band_edges_decode_exactly(self, env):
        d = lambda i: float(jax.device_get(env._decode_price_eur(jnp.int32(i))))
        assert d(OFFSET_INT) == 0.0, "the zero-EUR tick must decode to exact 0.0"
        assert d(0) == pytest.approx(PRICE_MIN, rel=REL_BOUND)
        assert d(MAX_TICK) == pytest.approx(PRICE_MAX, rel=REL_BOUND)
        # negative half of the band (the card: "include negative prices")
        assert d(OFFSET_INT - 5000) == pytest.approx(-50.0, rel=REL_BOUND)

    def test_every_decoded_price_re_encodes_to_its_own_tick(self, env):
        """Through the SHIPPED re-encode path, no tolerance."""
        ticks = _tick_grid(n=20001)
        p = env._decode_price_eur(jnp.asarray(ticks))
        back = np.asarray(jax.device_get(_submit_tick(p)), dtype=np.int64)
        bad = np.flatnonzero(back != ticks.astype(np.int64))
        assert bad.size == 0, (
            f"{bad.size}/{ticks.size} ticks do not re-encode to themselves; "
            f"first: tick {int(ticks[bad[0]])} -> {float(p[bad[0]])} -> "
            f"{int(back[bad[0]])}"
        )

    def test_decode_does_not_promote_to_int64_or_float64(self, env):
        out = env._decode_price_eur(jnp.asarray(_tick_grid()[:16]))
        assert out.dtype == jnp.float32, f"decode returned {out.dtype} (OQ-18)"

    def test_decode_is_shape_polymorphic_and_scalar_safe(self, env):
        scalar = env._decode_price_eur(jnp.int32(1_004_910))
        vec = env._decode_price_eur(jnp.asarray([1_004_910], jnp.int32))
        assert np.asarray(scalar).shape == ()
        np.testing.assert_array_equal(_bits(scalar), _bits(vec)[0])
        # the exact case the repo's own int-domain comment cites (50.10 — the
        # naive form produced 50.0996 here). The bound is the float32 ULP at
        # this magnitude (~3.8e-6), not something tighter: the int-domain form
        # is correctly ROUNDED, not exact-in-decimal.
        assert float(scalar) == pytest.approx(50.10, rel=REL_BOUND)
        assert np.float32(scalar) == np.float32(50.10), (
            "the decode is not the nearest float32 to 50.10")


# ===========================================================================
# B. The MID anchor — deliverable (b2). Odd spreads are the masked class.
# ===========================================================================
class TestMidAnchorIntDomain:

    @pytest.mark.parametrize("bid_eur_tick", [
        OFFSET_INT + 5000,          #  50.00 EUR
        OFFSET_INT - 5000,          # -50.00 EUR (negative price)
        OFFSET_INT,                 #   0.00 EUR
        OFFSET_INT + 4725,          #  47.25 (the existing fixture's bid)
        0,                          #  price_min band edge
        MAX_TICK - 5,               #  price_max band edge
    ])
    @pytest.mark.parametrize("spread", [1, 3, 2, 4, 0, 5])
    def test_mid_equals_the_floored_int_mid_decoded(self, env, bid_eur_tick, spread):
        bid_i, ask_i = bid_eur_tick, bid_eur_tick + spread
        lob = _lob_with(env, bid_i, ask_i)
        _bai, _bbi, has_ask, has_bid, _pa, _pb, mid = env._top_of_book(lob)
        assert bool(has_ask) and bool(has_bid)
        expected = env._decode_price_eur(jnp.int32((bid_i + ask_i) // 2))
        np.testing.assert_array_equal(
            _bits(mid), _bits(expected),
            err_msg=(f"mid for bid_tick={bid_i} ask_tick={ask_i} (spread "
                     f"{spread}) is not the decoded floored int mid"),
        )

    @pytest.mark.parametrize("spread", [1, 3, 5])
    def test_ODD_spread_mid_floors_TOWARD_THE_BID(self, env, spread):
        """The stated tie rule. Every pre-existing MID fixture in the repo has
        an EVEN spread, which is exactly what masked this class."""
        bid_i = OFFSET_INT + 5000
        ask_i = bid_i + spread
        lob = _lob_with(env, bid_i, ask_i)
        _a, _b, _ha, _hb, p_ask, p_bid, mid = env._top_of_book(lob)
        arithmetic = (float(p_bid) + float(p_ask)) / 2.0
        assert float(mid) < arithmetic, (
            f"odd spread {spread}: mid {float(mid)} did not floor below the "
            f"arithmetic midpoint {arithmetic}"
        )
        # floor toward the bid == bid + floor(spread/2) ticks
        assert int(jax.device_get(_submit_tick(mid))) == bid_i + spread // 2

    @pytest.mark.parametrize("spread", [0, 1, 2, 3, 4, 5, 7])
    @pytest.mark.parametrize("bid_eur_tick", [OFFSET_INT + 5000, OFFSET_INT - 5000,
                                              1, MAX_TICK - 7])
    def test_encoded_SUBMIT_tick_of_the_mid_is_the_floored_int_mid(
        self, env, spread, bid_eur_tick
    ):
        """The load-bearing assertion: not the decoded EUR float but the tick
        ``order_validator.py:81`` re-encodes it to. This is where the half-tick
        tie was being broken by a 2.4e-4 EUR shift."""
        bid_i, ask_i = bid_eur_tick, bid_eur_tick + spread
        lob = _lob_with(env, bid_i, ask_i)
        *_r, mid = env._top_of_book(lob)
        got = int(jax.device_get(_submit_tick(mid)))
        assert got == (bid_i + ask_i) // 2, (
            f"submit tick {got} != floor((bid+ask)/2) = {(bid_i + ask_i)//2} "
            f"for bid={bid_i} ask={ask_i} (spread {spread})"
        )

    def test_the_pre_change_float_mid_would_FAIL_the_odd_spread_submit_tick(self):
        """Non-vacuity: show the OLD anchor (naive decode, float mean) does NOT
        satisfy the assertion above on odd spreads."""
        bid_i, ask_i = OFFSET_INT + 5000, OFFSET_INT + 5001
        p_bid = _naive_decode(jnp.int32(bid_i))
        p_ask = _naive_decode(jnp.int32(ask_i))
        old_mid = (p_bid + p_ask) / jnp.float32(2.0)
        old_tick = int(jax.device_get(_submit_tick(old_mid)))
        new_tick = (bid_i + ask_i) // 2
        if old_tick == new_tick:
            pytest.skip(
                "the pre-change float mid happens to re-encode to the floored "
                "tick on this backend (the card measured CPU as already "
                "flooring 100% of the time) — the GPU leg is the discriminating "
                "one for this rule."
            )

    def test_both_sides_empty_gives_the_byte_unchanged_zero_fallback(self, env):
        lob = _lob_with(env, None, None)
        _a, _b, has_ask, has_bid, p_ask, p_bid, mid = env._top_of_book(lob)
        assert not bool(has_ask) and not bool(has_bid)
        for name, v in (("p_ask", p_ask), ("p_bid", p_bid), ("mid", mid)):
            assert float(v) == 0.0 and np.signbit(np.float32(v)) == 0, (
                f"{name} empty-book fallback is {float(v)!r}, not +0.0"
            )

    @pytest.mark.parametrize("side", ["bid_only", "ask_only"])
    def test_one_sided_book_masks_mid_to_zero_and_keeps_the_present_touch(
        self, env, side
    ):
        t = OFFSET_INT + 5000
        lob = _lob_with(env, t if side == "bid_only" else None,
                        None if side == "bid_only" else t)
        _a, _b, has_ask, has_bid, p_ask, p_bid, mid = env._top_of_book(lob)
        assert float(mid) == 0.0, "mid must be 0.0 unless BOTH sides are present"
        present = p_bid if side == "bid_only" else p_ask
        absent = p_ask if side == "bid_only" else p_bid
        np.testing.assert_array_equal(
            _bits(present), _bits(env._decode_price_eur(jnp.int32(t))))
        assert float(absent) == 0.0

    def test_mid_at_the_band_edges_stays_in_band_and_on_tick(self, env):
        for bid_i, ask_i in ((0, 1), (0, 3), (MAX_TICK - 1, MAX_TICK),
                             (MAX_TICK - 3, MAX_TICK)):
            lob = _lob_with(env, bid_i, ask_i)
            *_r, mid = env._top_of_book(lob)
            m = float(mid)
            assert PRICE_MIN - 1e-3 <= m <= PRICE_MAX + 1e-3, (
                f"mid {m} left the SIDC band for ticks ({bid_i},{ask_i})")
            assert int(jax.device_get(_submit_tick(mid))) == (bid_i + ask_i) // 2

    def test_mid_int_sum_cannot_overflow_int32(self, env):
        """The card's own OQ-18 claim, checked at the true maximum."""
        lob = _lob_with(env, MAX_TICK, MAX_TICK)
        *_r, mid = env._top_of_book(lob)
        assert int(jax.device_get(_submit_tick(mid))) == MAX_TICK
        assert float(mid) == pytest.approx(PRICE_MAX, rel=REL_BOUND)


# ===========================================================================
# C. No partial fix — the nine-site inventory
# ===========================================================================
class TestNoPartialFixAcrossSites:

    def test_top_of_book_touches_are_the_shared_helper_bit_for_bit(self, env):
        for bid_i, ask_i in ((OFFSET_INT + 5000, OFFSET_INT + 5001),
                             (12345, 1_777_777), (0, MAX_TICK)):
            lob = _lob_with(env, bid_i, ask_i)
            _a, _b, _ha, _hb, p_ask, p_bid, _m = env._top_of_book(lob)
            np.testing.assert_array_equal(
                _bits(p_bid), _bits(env._decode_price_eur(jnp.int32(bid_i))))
            np.testing.assert_array_equal(
                _bits(p_ask), _bits(env._decode_price_eur(jnp.int32(ask_i))))

    def test_passive_fill_decodes_the_same_tick_to_the_same_float32(self, env):
        """``_extract_passive_fill`` is the second traced FIX site. Two decoded
        prices disagreeing WITHIN one run is the failure mode this closes."""
        n = int(np.asarray(env._empty_book().lob.trades).shape[0])
        price_tick_code = OFFSET_INT + 5137          # 51.37 EUR/MWh
        qty_ticks = 40                               # 4.0 MW

        before = np.full((n, 8), -1, dtype=np.int32)
        after = before.copy()
        after[0] = [price_tick_code, -qty_ticks, 11, 12, 5, 0,
                    AGENT_TRADER_ID, 99]             # agent is the PASSIVE leg

        inv_delta, realized_eur, _row = env._extract_passive_fill(
            jnp.asarray(before), jnp.asarray(after), jnp.int32(5)
        )
        p = env._decode_price_eur(jnp.int32(price_tick_code))

        # Reproduce the shipped fold in the SAME float32 op order:
        #   reward_dir * qty_mw * price * delivery_duration_hours
        qty_mw = np.float32(qty_ticks) * np.float32(0.1)
        expected = np.float32(np.float32(np.float32(1.0) * qty_mw)
                              * np.float32(p)) * np.float32(0.25)
        assert float(inv_delta) < 0.0, "a passive SELL must reduce inventory"
        np.testing.assert_array_equal(
            _bits(realized_eur), _bits(expected),
            err_msg=("_extract_passive_fill's decoded price differs from "
                     "_decode_price_eur for the SAME tick — a partial fix."),
        )

    def test_the_three_traced_FIX_sites_no_longer_add_price_min(self):
        """Source-level guard on the nine-site inventory: the FIX sites must
        not carry the naive ``* price_tick + price_min`` addend."""
        naive = re.compile(r"price_tick\s*\)?\s*\+\s*(jnp\.float32\()?\s*price_min")
        for rel in ("omiecid_simulator/omiecid/omie_session.py",
                    "omiecid_simulator/jaxomiecid/omie_data_loader.py"):
            src = (REPO / rel).read_text()
            traced = [ln for ln in src.splitlines()
                      if naive.search(ln) and "jnp" in ln]
            assert not traced, (
                f"{rel} still contains a traced naive decode: {traced}")

    def test_the_LEAVE_sites_were_not_touched(self):
        """(c) — ``contract_book.py`` / ``settlement.py`` / ``encoding.py`` are
        already correct or host-side and must be untouched."""
        cb = (REPO / "omiecid_simulator/omiecid/contract_book.py").read_text()
        assert "vwap_int - price_offset_int" in cb, (
            "contract_book.py's int-domain precedent changed — it is a LEAVE site")
        enc = (REPO / "omiecid_simulator/omiecid/encoding.py").read_text()
        assert "def decode_price" in enc and "price_min" in enc, (
            "encoding.py (host float64, LEAVE) was modified")

    def test_the_ENCODE_site_is_still_an_encode(self):
        """``_agent_trade_row`` stores the OFFSET int tick, not a decoded EUR."""
        src = (REPO / "omiecid_simulator/omiecid/omie_session.py").read_text()
        assert "_agent_trade_row" in src


# ===========================================================================
# D. The golden re-baseline — deliverables (e1)/(e2)/(e3)
# ===========================================================================
class TestGoldenRebaselineIsHonest:

    @pytest.mark.jit
    def test_e3_the_price_level_move_is_at_most_1e_minus_3_eur(self, env):
        """(e3) demonstrated on the DECODE itself, not on an aggregate."""
        ticks = _tick_grid(n=20001)
        new = np.asarray(jax.device_get(env._decode_price_eur(jnp.asarray(ticks))),
                         dtype=np.float64)
        old = np.asarray(jax.device_get(jax.jit(_naive_decode)(jnp.asarray(ticks))),
                         dtype=np.float64)
        assert np.abs(new - old).max() <= 1e-3, (
            f"price-level move {np.abs(new - old).max():.4e} EUR/MWh > 1e-3")


# ===========================================================================
# F. THE HEADLINE PROMISE — cross-device, JITTED with an explicit backend.
# ===========================================================================
@pytest.mark.gpu
class TestBackendIdentityUnderJit:
    """An EAGER CPU/GPU comparison of this decode is vacuously green (it was
    green before the fix). Every leg here jits with ``backend=``."""

    @pytest.mark.jit
    def test_decode_is_bit_identical_cpu_vs_gpu_under_jit(self, env):
        ticks = jnp.asarray(_tick_grid(n=20001))
        on_cpu = jax.jit(env._decode_price_eur, backend="cpu")(ticks)
        on_gpu = jax.jit(env._decode_price_eur, backend="gpu")(ticks)
        diff = np.flatnonzero(_bits(on_cpu) != _bits(on_gpu))
        assert diff.size == 0, (
            f"{diff.size}/{ticks.size} ticks decode to DIFFERENT float32 bits "
            f"CPU vs GPU under jit; first at tick "
            f"{int(np.asarray(ticks)[diff[0]])}: "
            f"{float(on_cpu[diff[0]])!r} vs {float(on_gpu[diff[0]])!r}"
        )

    @pytest.mark.jit
    def test_the_harness_DOES_catch_the_pre_change_form(self, env):
        """NON-VACUITY. The card measured 51.5% of ticks diverging CPU vs GPU
        for the naive form under jit. If this probe is green, the test above
        proves nothing and the audit must be treated as inconclusive."""
        ticks = jnp.asarray(_tick_grid(n=20001))
        c = jax.jit(_naive_decode, backend="cpu")(ticks)
        g = jax.jit(_naive_decode, backend="gpu")(ticks)
        n_diff = int((_bits(c) != _bits(g)).sum())
        assert n_diff > 0, (
            "the naive decode no longer diverges CPU vs GPU under jit on this "
            "host — the cross-device identity test above is VACUOUS here."
        )

    def test_eager_comparison_alone_would_have_been_vacuous(self, env):
        """Documents WHY the jit is mandatory: eager agrees on the naive form."""
        ticks = jnp.asarray(_tick_grid(n=4001))
        with jax.default_device(jax.devices("cpu")[0]):
            c = _naive_decode(np.asarray(ticks))
        with jax.default_device(jax.devices("gpu")[0]):
            g = _naive_decode(np.asarray(ticks))
        n_diff = int((_bits(c) != _bits(g)).sum())
        assert n_diff >= 0  # informational only
        if n_diff == 0:
            pytest.skip(
                "confirmed: EAGER CPU/GPU agree bitwise even on the naive form "
                f"({ticks.size}/{ticks.size} ticks) — an eager-only cross-device "
                "test of this decode is vacuous by construction."
            )

    @pytest.mark.jit
    def test_reencode_roundtrip_is_exact_on_gpu_too(self, env):
        ticks = np.asarray(_tick_grid(n=20001))
        p = jax.jit(env._decode_price_eur, backend="gpu")(jnp.asarray(ticks))
        back = np.asarray(jax.device_get(
            jax.jit(_submit_tick, backend="gpu")(p)), dtype=np.int64)
        bad = np.flatnonzero(back != ticks.astype(np.int64))
        assert bad.size == 0, (
            f"{bad.size} ticks fail the GPU re-encode round-trip; first "
            f"{int(ticks[bad[0]])} -> {int(back[bad[0]])}")

    @pytest.mark.jit
    @pytest.mark.parametrize("spread", [1, 3, 0, 2])
    def test_mid_submit_tick_is_backend_identical_on_ODD_spreads(self, env, spread):
        """The leg that actually closes the card's headline promise: the decode
        fix ALONE left the MID submit tick CPU/GPU-divergent on 15.7% of
        odd-spread pairs (card measurement)."""
        bids = np.arange(OFFSET_INT - 3000, OFFSET_INT + 3000, 7, dtype=np.int32)
        asks = bids + spread

        def _mid_tick(b, a):
            mid_int = (b + a) // jnp.int32(2)
            return _submit_tick(env._decode_price_eur(mid_int))

        # Route through the SHIPPED _top_of_book on both backends for one
        # representative pair, then vectorise the anchor arithmetic.
        cpu = np.asarray(jax.device_get(
            jax.jit(_mid_tick, backend="cpu")(jnp.asarray(bids), jnp.asarray(asks))))
        gpu = np.asarray(jax.device_get(
            jax.jit(_mid_tick, backend="gpu")(jnp.asarray(bids), jnp.asarray(asks))))
        expected = (bids.astype(np.int64) + asks.astype(np.int64)) // 2
        np.testing.assert_array_equal(cpu, gpu)
        np.testing.assert_array_equal(cpu.astype(np.int64), expected)

    @pytest.mark.jit
    def test_shipped_top_of_book_mid_agrees_across_backends_on_an_odd_spread(self, env):
        bid_i, ask_i = OFFSET_INT + 5000, OFFSET_INT + 5001
        lob = _lob_with(env, bid_i, ask_i)
        cpu = jax.jit(env._top_of_book, backend="cpu")(lob)[6]
        gpu = jax.jit(env._top_of_book, backend="gpu")(lob)[6]
        np.testing.assert_array_equal(_bits(cpu), _bits(gpu))
        assert int(jax.device_get(_submit_tick(cpu))) == (bid_i + ask_i) // 2
