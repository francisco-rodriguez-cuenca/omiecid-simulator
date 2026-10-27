"""LOCKED behavioral contract — Story ``OMIE-PRICE-DECODE-FMA-1``:
make the int-tick -> EUR/MWh price decode cancellation-free and backend-identical.

Authored TEST-FIRST (Step 3) against the agile-plan card as REVISED 2026-09-10
(``agile-plan.md`` v5.143, §"Story OMIE-PRICE-DECODE-FMA-1"), plus the
``sprint-status.md`` §"OMIE-PRICE-DECODE-FMA-1" Step-2 **Architecture readiness
review** and the **Review response (Step-2 card revision, 2026-09-10)** block.
Every numeric bound below is TAKEN FROM those measurement blocks, not re-derived
here.

WHAT IS WRONG TODAY
-------------------
``omie_session.py:1252,:1255`` (``_top_of_book``) and ``:1373``
(``_extract_passive_fill``) decode an int tick as::

    best_int.astype(f32) * price_tick + price_min          # price_min = -9999.0

which is **catastrophic cancellation by construction** — the two operands are
~200x the answer, so the ``float32(0.01)`` representation error is amplified
into the result, and XLA:CPU contracts the multiply-add into an **FMA** while
XLA:GPU (at this shape) does not, so the two backends disagree.

THE FIX THIS FILE PINS (card deliverables (a)/(b)/(b2))
------------------------------------------------------
The repo's OWN existing convention, already used verbatim at
``contract_book.py:354`` and ``settlement.py:100-101``::

    (price_int - price_offset_int).astype(jnp.float32) * jnp.float32(price_tick)

surfaced as deliverable (a): a private ``Session._decode_price_eur(price_int)``
with ``self._price_offset_int = jnp.int32(round(-price_min / price_tick))``
computed once in ``__init__``. **The method NAME and the attribute NAME are part
of the card's deliverable (a) and are therefore pinned here** — this file is the
contract the developer implements against.

Deliverable (b2), THE MID ANCHOR: ``mid = (p_bid + p_ask) / 2`` becomes
``mid_int = (best_bid_int + best_ask_int) // 2`` decoded through the SAME
helper. **Stated tie rule: FLOOR toward the bid.**

MEASUREMENTS THIS FILE HONOURS (do NOT re-derive; from the review-response block)
--------------------------------------------------------------------------------
  * EXHAUSTIVE over all 1 999 801 legal ticks, BOTH backends: the int-domain
    decode re-encodes to its own tick **100.0000%**, is **md5-IDENTICAL CPU vs
    GPU**, and ``eager == jit`` on both. => the re-encode and cross-device tests
    below assert **EQUALITY WITH NO TOLERANCE**.
  * Worst relative error of the int-domain form: ``max_rel = 8.1062e-08``. The
    locked bound is ``<= 2e-7 * max(|p|, 1.0)`` — **2e-7, NOT 1e-7** (only 1.23x
    headroom at 1e-7; 2.47x at 2e-7). The naive form fails this by >1000x
    (2.2354e-4 CPU / 7.0313e-4 GPU relative), which is the RED.
  * (b2) is **BYTE-NEUTRAL ON CPU** *in mid-band*: floor reproduces the tick today's CPU
    pipeline already submits in 100.00% of odd-spread cases (the naive decode's
    uniform negative bias has always floored). It moves only GPU (today's GPU
    agrees with floor on 85.3%/87.5% of odd-spread pairs). So the CPU MID tests
    below are REGRESSION LOCKS that are GREEN today by design; the RED they
    guard is on the anchor's float32 BITS (``test_mid_anchor_eur_bits_equal_the
    _decoded_int_mid``) and on the GPU twin.
  * **QUALIFICATION MEASURED WHILE AUTHORING THIS FILE (CPU, 2026-09-10):** that
    "floor 100.00% on CPU" figure holds MID-BAND only. Re-measured across price
    MAGNITUDE, 2 of the 20 CPU MID cases below are RED today, both at the BAND
    EDGES — near ``price_min`` the cancellation term vanishes so the half-tick
    sits exactly on ``.5`` and ``jnp.round``'s half-to-even takes it UP; near
    ``price_max`` the float32 ulp (~1e-3 EUR at 1e4) exceeds a whole tick. So
    (b2)'s CPU blast radius is not literally zero: it is band-edge-only. See
    ``test_mid_bucket_submit_tick_is_the_floored_int_mid``.

THE INCIDENTAL CONSTANT THAT MASKED THIS CLASS (standing ``OMIE-REGEN-MASK-1`` /
``WF-15-FOLLOWUP-2`` lesson: a fixture that holds an incidental constant masks
the defect)
------------------------------------------------------------------------------
**Every MID-bucket fixture in the repo holds an EVEN spread** —
``test_touch_relative_action.py:196-197`` 47.25/52.75 (550 ticks),
``test_order_plot_module{,_audit}.py`` 50.01/49.99 (2), 50.02/49.98 (4),
50.00/50.00 (0). Spread PARITY is exactly the constant that hid this: on an even
spread the mid is on-tick and ``floor`` is a no-op. **So the MID tests here pin
spreads 1 and 3 (ODD) as well as 2 and 4 (EVEN), at several price levels
including NEGATIVE prices and ticks adjacent to both band edges**, and the same
sweep is repeated on GPU.

CANARY VERDICT (card, verbatim): ``test_order_plot_module_audit.py:542`` is
``p_ask=50.01 / p_bid=49.99`` = **2 ticks, EVEN**, so its mid is exactly on-tick
(50.00) and ``floor`` is a no-op there — it is **EXPECTED NOT TO MOVE**. That
exact fixture is re-asserted here as
``test_even_spread_canary_fixture_does_not_move`` so any movement is caught in
THIS file as a STOP condition rather than as collateral damage over there.

OUT OF SCOPE / DO NOT ASSERT HERE
---------------------------------
  * ``mid_price`` is NOT a policy input — WF-4 dropped it from the flat obs
    (8 channels: ``p_bid, p_ask, q_bid, q_ask, time_to_gct, step_counter,
    inventory, day_ahead_anchor``). **No obs-width / network-shape assertion.**
  * ``tests/omiecid/omie_eval_fast_golden.json`` is regenerated by a LATER step of
    this same story. **Nothing here reads it**, so these tests neither pass nor
    fail on the stale golden.
  * ``encoding.decode_price`` (host, float64) must NOT be used as the float32
    ground truth (review finding B3) — its float64 companion would pin the naive
    form as truth. The re-encode test inverts through the SHIPPED encode path,
    ``jnp.round(price * 100) + price_offset_int`` (``order_validator.py:79-82``).

DEVICE POLICY (``docs/omiecid/device-policy.md``)
-----------------------------------------------
CPU-default; the CPU platform pin is applied by ``tests/omiecid/conftest.py``. The
cross-device twin carries ``@pytest.mark.gpu`` (auto-skipped on CUDA-hidden
hosts by conftest) + ``@pytest.mark.slow`` (real GPU compiles).

RUNTIME / PROCESS BUDGET
------------------------
Fast-suite budget (<1s/test) is met by constructing **exactly TWO** ``Session``
instances for the whole module (module-scoped fixtures) — the documented XLA
single-process segfault ceiling is why the count is kept low — and by driving
``_top_of_book`` / ``Session.top_of_book`` against SYNTHETIC ``LobState`` arrays
built with ``.at[].set()`` on the post-reset book, instead of building one tape
+ env per spread. Tests that trigger a jit compile carry ``@pytest.mark.jit``.

Run (per the story dispatch):
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python3 -m pytest -n0 -q \
        tests/omiecid/test_price_decode_fma.py

RELEASE NOTE (REL-07): the release keeps every decode site this card fixed
(``Session._decode_price_eur``, ``_top_of_book`` / the public
``Session.top_of_book`` touches and mid, ``_extract_passive_fill``); only the
RL action decode (``_decode_action``, the MID price bucket) is gone. A "MID
submit tick" below is therefore the tick an ``AgentOrder`` priced at the
public ``TopOfBook.mid`` re-encodes to through the shipped encode path — the
same price the research MID bucket submitted (it priced at ``mid``, clipped to
the band). The side-independence test of the per-side decode branch was
removed with that branch.
"""

import pytest


# ---------------------------------------------------------------------------
# Interface import — guarded so the file stays fully COLLECTIBLE regardless of
# implementation state (mirrors tests/omiecid/test_touch_relative_action.py:110).
# Every symbol below is already on disk today; the RED is in the ASSERTIONS and
# in the two names deliverable (a) has yet to add.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import jax
    import jax.numpy as jnp

    from omiecid_simulator.omiecid.omie_session import (
        AGENT_TRADER_ID, AgentOrder, Session,
    )
    from omiecid_simulator.omiecid.omie_config import OMIEProductConfig

    _MISSING = False
    _IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - collection-time guard
    _MISSING = True
    _IMPORT_ERROR = repr(exc)

_skip = pytest.mark.skipif(
    _MISSING,
    reason=(
        "omiecid_simulator/omiecid/omie_session.py (Session) / reused interfaces not "
        f"importable ({_IMPORT_ERROR})"
    ),
)


# ===========================================================================
# Documented contract constants (drift-guarded; sourced from the card).
# ===========================================================================

# Default SIDC band (omie_config.py:18-20).
PRICE_TICK = 0.01
PRICE_MIN = -9999.0
PRICE_MAX = 9999.0

# Deliverable (a): the offset and the int32-headroom claim, as NUMBERS.
#   price_offset_int = round(-price_min / price_tick) = 999_900
#   max legal tick   = round((price_max - price_min)/price_tick) = 1_999_800
# Both, and their difference/sum, sit inside int32 => NO x64 scope is opened
# (OQ-18 holds). The sum leg matters for (b2): 2 * 1_999_800 = 3_999_600.
PRICE_OFFSET_INT = 999_900
MAX_LEGAL_TICK = 1_999_800
N_LEGAL_TICKS = MAX_LEGAL_TICK + 1          # 1_999_801

# Locked RELATIVE bound (review finding B3). NOT an absolute bound: the
# int-domain error is 0.5 ulp of a float32 PRODUCT, so it scales with |p|
# (2.5635e-5 absolute over [-500,300]; 7.031e-4 over the full legal band) and NO
# single absolute bound can hold across the band. Exhaustively measured worst
# case over all 1_999_801 legal ticks on BOTH backends: 8.1062e-08.
DECODE_REL_TOL = 2e-7
MEASURED_WORST_REL = 8.1062e-08             # recorded for the headroom assertion

# Stride for the locked grid. 1_999_801 / 7 = 285_686 points — the card's
# "~285k is enough for the locked test" (the exhaustive sweep lives in
# sprint-status.md, not in the fast suite).
GRID_STRIDE = 7

# Order sides (order_types.Side).
BUY, SELL = 0, 1

# ``_extract_passive_fill``'s jaxob TradesFeat columns used below
# (omie_session.py:1335-1338): P=0 maker price offset, Q=1 signed qty ticks,
# PASS_TID=6 passive/maker trader id.
TRADES_P, TRADES_Q, TRADES_PASS_TID = 0, 1, 6

# 40 qty ticks = 4.0 MW EXACTLY in float32, and 4.0 * 0.25 h = 1.0 EXACTLY.
# Chosen so ``passive_realized_eur`` for a 1-row passive BUY collapses to
# exactly ``-price_eur`` with NO rounding of its own — which is what lets test
# (4) compare the two decode sites BITWISE instead of through a tolerance.
PASSIVE_QTY_TICKS = 40


# ===========================================================================
# Helpers.
# ===========================================================================


def _tick_grid():
    """Stride-sampled legal-tick grid + the edges that round literals hide.

    Per the card: "do not sample only round literals. 51.0/50.0/80.0 are exactly
    the values that hid this on GPU." So: a stride-7 sweep of the WHOLE legal
    band (which is ~50% negative prices), UNION the extreme ticks (0 and
    MAX_LEGAL_TICK, i.e. exactly price_min / price_max), their neighbours, and
    the zero-crossing at price_offset_int +/- 1.
    """
    grid = np.arange(0, N_LEGAL_TICKS, GRID_STRIDE, dtype=np.int64)
    edges = np.array(
        [0, 1, 2,
         PRICE_OFFSET_INT - 1, PRICE_OFFSET_INT, PRICE_OFFSET_INT + 1,
         MAX_LEGAL_TICK - 2, MAX_LEGAL_TICK - 1, MAX_LEGAL_TICK],
        dtype=np.int64,
    )
    return np.unique(np.concatenate([grid, edges])).astype(np.int32)


def _truth_eur(ticks, price_tick=PRICE_TICK, offset_int=PRICE_OFFSET_INT):
    """float64 ground truth: (tick - offset) * tick_size, computed in the INT
    domain in float64 (exact for these magnitudes). Deliberately NOT
    ``encoding.decode_price`` — see the module docstring / review finding B3."""
    return (np.asarray(ticks, dtype=np.int64) - offset_int) * np.float64(price_tick)


def _decode(env, ticks):
    """Call deliverable (a)'s helper. Fails LOUDLY (not by AttributeError) while
    the helper is still absent, so the RED reads as a contract statement."""
    fn = getattr(env, "_decode_price_eur", None)
    if fn is None:
        pytest.fail(
            "OMIE-PRICE-DECODE-FMA-1 deliverable (a) missing: Session has no "
            "`_decode_price_eur(price_int)`. Expected "
            "`(price_int - self._price_offset_int).astype(jnp.float32) * "
            "jnp.float32(price_tick)` — the int-domain form ALREADY used at "
            "contract_book.py:354 / settlement.py:100-101."
        )
    return fn(jnp.asarray(ticks, dtype=jnp.int32))


def _reencode(prices_eur, price_tick=PRICE_TICK, offset_int=PRICE_OFFSET_INT):
    """The SHIPPED re-encode path, mirrored exactly: ``order_validator.py:79-82``
    computes ``round(price * round(1/price_tick)) + price_offset_int`` in
    float32 -> int32. This — not ``encoding.encode_price`` (host float64) — is
    what decides which tick a decoded price actually SUBMITS at."""
    scale = jnp.float32(round(1.0 / price_tick))
    p = jnp.asarray(prices_eur, dtype=jnp.float32)
    return jnp.round(p * scale).astype(jnp.int32) + jnp.int32(offset_int)


def _synthetic_lob(base_lob, bid_int, ask_int):
    """One-level two-sided synthetic book at the given int ticks.

    ``get_best_bid = max(bids[:,0])`` and ``get_best_ask = min(asks[:,0] != -1)``
    (JaxOrderBookArrays.py:933-951), so setting row 0 of each side on an
    otherwise all(-1) post-reset LobState pins the top of book exactly. This is
    what keeps the whole MID sweep inside ONE env construction."""
    return base_lob._replace(
        asks=base_lob.asks.at[:, :].set(-1).at[0, 0].set(jnp.int32(ask_int)),
        bids=base_lob.bids.at[:, :].set(-1).at[0, 0].set(jnp.int32(bid_int)),
    )


def _mid_submit_tick(env, state, bid_int, ask_int, side=BUY):
    """The ENCODED submit tick of an order priced at the mid against a
    synthetic book — i.e. the tick the order ACTUALLY rests at, end to end:
    public ``Session.top_of_book`` -> ``AgentOrder`` (float32 price) ->
    ``order_validator``'s re-encode."""
    lob = _synthetic_lob(state.book.lob, bid_int, ask_int)
    st = state.replace(book=state.book._replace(lob=lob))
    tob = env.top_of_book(st)
    order = AgentOrder.create(side=side, price=tob.mid,
                              quantity=OMIEProductConfig().qty_min)
    return int(np.asarray(_reencode(order.price)))


def _top_of_book(env, state, bid_int, ask_int):
    """``(p_ask, p_bid, mid)`` float32 arrays for a synthetic two-sided book."""
    lob = _synthetic_lob(state.book.lob, bid_int, ask_int)
    _bai, _bbi, _ha, _hb, p_ask, p_bid, mid = env._top_of_book(lob)
    return p_ask, p_bid, mid


# ``(bid_int, spread_ticks)`` MID fixtures. **The spread parity is the varied
# constant** — 1 and 3 are ODD (half-tick mid; the previously-uncovered class),
# 2 and 4 are EVEN (on-tick mid; ``floor`` is a no-op). Price levels span the
# band deliberately: a round 50.00 anchor, a NEGATIVE price (-7.77), a
# non-round mid-band level, and ticks adjacent to BOTH band edges.
_MID_LEVELS = [
    ("round_50", 1_004_900),        # 50.00 EUR/MWh (the round literal that hid it)
    ("negative", 999_123),          # -7.77 EUR/MWh
    ("nonround", 1_234_567),        # 2345.67 EUR/MWh
    ("near_price_min", 3),          # -9998.97 EUR/MWh
    ("near_price_max", 1_999_790),  # 9998.90 EUR/MWh
]
_MID_SPREADS = [1, 2, 3, 4]
_MID_CASES = [
    pytest.param(bid, sp, id=f"{name}-spread{sp}{'-ODD' if sp % 2 else '-EVEN'}")
    for (name, bid) in _MID_LEVELS
    for sp in _MID_SPREADS
]


# ===========================================================================
# Fixtures — EXACTLY TWO Session constructions for the whole module.
# ===========================================================================


@pytest.fixture(scope="module")
def env():
    """Default-band env (price_tick=0.01, price_min=-9999.0)."""
    return Session.for_testing()


@pytest.fixture(scope="module")
def state(env):
    """Post-reset state; its ``book.lob`` is the all(-1) canvas every synthetic
    book below is built on."""
    st = env.reset(jax.random.PRNGKey(0))
    return st


@pytest.fixture(scope="module")
def env_alt():
    """SECOND ``(price_tick, price_min)`` pair — the card requires the decode be
    parametrized over at least two bands, because a single band cannot
    distinguish "correct" from "correct at THIS offset". Here price_min=-500.0
    => price_offset_int = 50_000, max legal tick = 1_049_900."""
    return Session({"product_config": OMIEProductConfig(price_min=-500.0)})


ALT_OFFSET_INT = 50_000
ALT_MAX_TICK = 1_049_900


# ===========================================================================
# Contract (a) — the helper exists, and the int32-headroom claim is a
# DELIVERABLE, not an assumption (card deliverable (a), OQ-18).
# ===========================================================================


@_skip
class TestDecodeHelperShape:

    def test_env_exposes_the_int_domain_decode_helper_and_offset(self, env):
        """Deliverable (a): ``_decode_price_eur`` + ``_price_offset_int``,
        computed once in ``__init__`` exactly as contract_book.py:206 does."""
        assert hasattr(env, "_price_offset_int"), (
            "deliverable (a): Session.__init__ must precompute "
            "`_price_offset_int = jnp.int32(round(-price_min / price_tick))`"
        )
        assert int(np.asarray(env._price_offset_int)) == PRICE_OFFSET_INT
        assert np.asarray(env._price_offset_int).dtype == np.int32, (
            "OQ-18: the offset must be int32 — an int64 offset would promote "
            "the subtraction and open x64 scope"
        )
        assert hasattr(env, "_decode_price_eur"), (
            "deliverable (a): Session must expose `_decode_price_eur(price_int)`"
        )

    def test_decode_returns_float32_and_stays_inside_int32(self, env):
        """The subtraction leg must not promote. Max magnitude is
        ``|0 - 999_900| = 999_900`` and ``MAX_LEGAL_TICK - offset = 999_900`` —
        two orders inside int32, so no x64 is opened (OQ-18 holds)."""
        out = _decode(env, np.array([0, PRICE_OFFSET_INT, MAX_LEGAL_TICK], np.int32))
        assert np.asarray(out).dtype == np.float32
        assert abs(MAX_LEGAL_TICK - PRICE_OFFSET_INT) < 2**31 - 1
        assert abs(0 - PRICE_OFFSET_INT) < 2**31 - 1
        # (b2)'s sum leg: 2 * MAX_LEGAL_TICK must also fit (int32 floor-div).
        assert 2 * MAX_LEGAL_TICK < 2**31 - 1

    def test_band_edges_and_zero_crossing_decode_exactly(self, env):
        """No tolerance: ``tick 0 -> price_min``, ``MAX -> price_max``,
        ``offset -> 0.0``. The naive form misses all three."""
        out = np.asarray(_decode(env, np.array([0, PRICE_OFFSET_INT, MAX_LEGAL_TICK],
                                               np.int32)))
        assert float(out[0]) == PRICE_MIN
        assert float(out[1]) == 0.0
        assert float(out[2]) == PRICE_MAX


# ===========================================================================
# Contract (1) — the RELATIVE accuracy bound. THE PRIMARY RED.
# ===========================================================================


@_skip
class TestDecodeAccuracy:

    @pytest.mark.jit
    def test_decode_is_correctly_rounded_across_the_trading_band(self, env):
        """``|decoded - float64_truth| <= 2e-7 * max(|p|, 1.0)`` over a
        stride-7 sweep of the WHOLE legal band (~285k ticks, ~half of them
        negative prices) plus both band edges.

        RED TODAY: the naive form measures 2.2354e-4 (CPU) / 7.0313e-4 (GPU)
        relative — it fails this bound by >1000x. The int-domain form measures
        8.1062e-08 exhaustively on both backends (2.47x headroom).
        """
        ticks = _tick_grid()
        truth = _truth_eur(ticks)
        got = np.asarray(_decode(env, ticks), dtype=np.float64)
        rel = np.abs(got - truth) / np.maximum(np.abs(truth), 1.0)
        worst = float(rel.max())
        assert worst <= DECODE_REL_TOL, (
            f"decode relative error {worst:.6e} exceeds the locked bound "
            f"{DECODE_REL_TOL:.1e}; the int-domain form measures "
            f"{MEASURED_WORST_REL:.4e} exhaustively (worst tick "
            f"{int(ticks[int(rel.argmax())])})"
        )

    @pytest.mark.jit
    def test_measured_headroom_against_the_locked_bound_is_preserved(self, env):
        """Guard the BOUND itself, not just the code: the card's measured worst
        case is 8.1062e-08, i.e. 2.47x under 2e-7. If a future change eats that
        headroom, this fails BEFORE the bound is silently loosened. (1e-7 was
        rejected during review at only 1.23x — do not tighten to it.)"""
        ticks = _tick_grid()
        truth = _truth_eur(ticks)
        got = np.asarray(_decode(env, ticks), dtype=np.float64)
        worst = float((np.abs(got - truth) / np.maximum(np.abs(truth), 1.0)).max())
        assert worst <= 1.25 * MEASURED_WORST_REL, (
            f"worst relative error {worst:.6e} regressed against the recorded "
            f"exhaustive measurement {MEASURED_WORST_REL:.4e}"
        )

    @pytest.mark.jit
    def test_decode_accuracy_holds_on_a_second_price_band(self, env_alt):
        """Second ``(price_tick, price_min)`` pair (price_min=-500.0,
        offset 50_000). A decode that is 'correct' only at the default offset is
        not correct."""
        ticks = np.unique(np.concatenate([
            np.arange(0, ALT_MAX_TICK + 1, GRID_STRIDE, dtype=np.int64),
            np.array([0, 1, ALT_OFFSET_INT, ALT_MAX_TICK], dtype=np.int64),
        ])).astype(np.int32)
        truth = _truth_eur(ticks, offset_int=ALT_OFFSET_INT)
        got = np.asarray(_decode(env_alt, ticks), dtype=np.float64)
        rel = np.abs(got - truth) / np.maximum(np.abs(truth), 1.0)
        assert float(rel.max()) <= DECODE_REL_TOL


# ===========================================================================
# Contract (2) — the FMA lock. This is the assertion that actually pins the
# defect's MECHANISM (XLA:CPU contracting mul+add into an FMA).
# ===========================================================================


@_skip
class TestDecodeIsFmaImmune:

    @pytest.mark.jit
    def test_decode_is_bit_identical_under_jit_and_eager(self, env):
        """``f(x)``, ``jax.jit(f)(x)`` and ``jax.jit`` of ``f`` traced INSIDE a
        larger graph (so XLA has an FMA contraction opportunity) must be
        **bitwise** equal float32.

        RED TODAY on CPU: measured directly on this grid, the naive form's eager
        and jitted results DIFFER — XLA:CPU fuses ``t*tick + price_min`` into an
        FMA while the eager op-by-op path rounds the product first. A form with
        no add to contract cannot differ.
        """
        ticks = _tick_grid()
        eager = np.asarray(_decode(env, ticks))

        jitted = np.asarray(jax.jit(lambda t: env._decode_price_eur(t))(
            jnp.asarray(ticks, jnp.int32)))

        def fused(t):
            # A larger consumer graph: the decode feeds an affine reduction, the
            # shape most likely to invite a contraction.
            p = env._decode_price_eur(t)
            return p, jnp.sum(p * jnp.float32(2.0) - jnp.float32(1.0))

        in_graph = np.asarray(jax.jit(fused)(jnp.asarray(ticks, jnp.int32))[0])

        assert np.array_equal(eager, jitted), (
            "eager vs jit differ bitwise on "
            f"{int((eager != jitted).sum())}/{eager.size} ticks — the decode is "
            "still FMA-contractible"
        )
        assert np.array_equal(eager, in_graph), (
            "standalone vs in-larger-graph differ bitwise on "
            f"{int((eager != in_graph).sum())}/{eager.size} ticks"
        )


# ===========================================================================
# Contract (3) — the invariant that carries the (scoped) economic claim:
# every decoded price re-encodes to its OWN tick. Exhaustively 100.00% on both
# backends => asserted with NO tolerance.
# ===========================================================================


@_skip
class TestReEncodeRoundTrip:

    @pytest.mark.jit
    def test_every_decoded_price_re_encodes_to_its_own_tick(self, env):
        """``round(decode(i) * 100) + price_offset_int == i`` for every sampled
        legal tick — inverted through the SHIPPED encode path
        (``order_validator.py:79-82``), NOT through ``encoding.decode_price``
        (host float64, which would pin the naive form as truth; review B3)."""
        ticks = _tick_grid()
        back = np.asarray(_reencode(_decode(env, ticks)))
        bad = int((back != ticks).sum())
        assert bad == 0, (
            f"{bad}/{ticks.size} decoded prices re-encode to a DIFFERENT tick; "
            "the exhaustive measurement is 100.00% on both backends"
        )

    @pytest.mark.jit
    def test_re_encode_round_trip_holds_on_a_second_price_band(self, env_alt):
        ticks = np.arange(0, ALT_MAX_TICK + 1, GRID_STRIDE, dtype=np.int32)
        back = np.asarray(_reencode(_decode(env_alt, ticks),
                                    offset_int=ALT_OFFSET_INT))
        assert int((back != ticks).sum()) == 0


# ===========================================================================
# Contract (3b) — THE MID ANCHOR, at ODD spreads. Added by review finding
# B2(c): the contract as filed had NO MID-bucket case at all.
# ===========================================================================


@_skip
class TestMidAnchorIsIntDomain:

    @pytest.mark.jit
    @pytest.mark.parametrize("bid_int,spread", _MID_CASES)
    def test_mid_bucket_submit_tick_is_the_floored_int_mid(
        self, env, state, bid_int, spread
    ):
        """The ENCODED submit tick of a MID-bucket order equals
        ``floor((best_bid_int + best_ask_int) / 2)`` EXACTLY — deliverable (b2),
        tie rule **FLOOR TOWARD THE BID**.

        Parametrized over spreads 1 and 3 (**ODD** — the half-tick mid, the
        class that had ZERO coverage) as well as 2 and 4 (EVEN), at five price
        levels including a negative price and ticks adjacent to both band edges.

        RED BASELINE MEASURED WHEN THIS FILE WAS AUTHORED (CPU, 2026-09-10):
        **18 of 20 cases pass today, 2 FAIL** — ``near_price_min-spread1`` (got
        tick 4, floor is 3) and ``near_price_max-spread3`` (got 1999792, floor
        is 1999791). Both are BAND-EDGE cases.

        This QUALIFIES the card's "byte-neutral on CPU / floor 100.00%"
        measurement rather than contradicting it: that sweep held the PRICE
        MAGNITUDE near mid-band, where the naive decode's uniform ~-2.4e-4 EUR
        bias always pushes the half-tick down. Near ``price_min`` the
        cancellation term vanishes, so the half-tick lands EXACTLY on ``.5`` and
        ``jnp.round``'s half-to-even takes it UP; near ``price_max`` the float32
        ulp (~1e-3 EUR at 1e4) exceeds a whole tick, so the half-tick is not
        representable at all. Price MAGNITUDE is therefore a SECOND incidental
        constant of the same family the card already named (standing
        ``OMIE-REGEN-MASK-1`` lesson), and it is varied here deliberately. (b2)
        makes all 20 cases exact by construction — so the developer must expect
        a small, band-edge-only CPU move on top of the card's stated
        GPU-only blast radius.
        """
        ask_int = bid_int + spread
        expected = (bid_int + ask_int) // 2
        assert expected == bid_int + spread // 2      # floor, toward the bid
        got = _mid_submit_tick(env, state, bid_int, ask_int, side=BUY)
        assert got == expected, (
            f"MID submit tick {got} != floor((bid+ask)/2)={expected} at "
            f"bid_int={bid_int}, spread={spread} ticks "
            f"({'ODD' if spread % 2 else 'EVEN'})"
        )

    @pytest.mark.jit
    @pytest.mark.parametrize("bid_int,spread", _MID_CASES)
    def test_mid_anchor_eur_bits_equal_the_decoded_int_mid(
        self, env, state, bid_int, spread
    ):
        """RED TODAY. ``_top_of_book``'s ``mid`` must be **bitwise** the decode
        of ``mid_int = (bid_int + ask_int) // 2`` — i.e. computed in the INT
        domain and pushed through the SAME helper (deliverable (b2)), not as
        ``(p_bid + p_ask) / 2``.

        Today's float mid lands ~2.4e-4 EUR low on CPU (and off by a different,
        backend-specific amount on GPU) and on ODD spreads it is not even
        on-tick, so it cannot equal ``decode(mid_int)`` in any bit.
        """
        ask_int = bid_int + spread
        _p_ask, _p_bid, mid = _top_of_book(env, state, bid_int, ask_int)
        expected = _decode(env, np.int32((bid_int + ask_int) // 2))
        assert np.asarray(mid).dtype == np.float32
        assert float(np.asarray(mid)) == float(np.asarray(expected)), (
            f"mid={float(np.asarray(mid)):.9f} != decode(mid_int)="
            f"{float(np.asarray(expected)):.9f} at bid_int={bid_int}, "
            f"spread={spread}"
        )

    @pytest.mark.jit
    def test_mid_anchor_is_always_on_tick(self, env, state):
        """The floored mid is ON-TICK, so it re-encodes to ``mid_int`` exactly —
        which is what makes the submitted MID limit price genuinely
        backend-identical. (The decode fix ALONE does not achieve this: a
        measured 15.7% of odd-spread pairs still disagree CPU vs GPU.)

        RED TODAY on CPU at the band edges (same two cases as the test above).
        """
        for bid_int, spread in [(b, s) for (_n, b) in _MID_LEVELS for s in _MID_SPREADS]:
            ask_int = bid_int + spread
            _pa, _pb, mid = _top_of_book(env, state, bid_int, ask_int)
            assert int(np.asarray(_reencode(mid))) == (bid_int + ask_int) // 2

    @pytest.mark.jit
    def test_even_spread_canary_fixture_does_not_move(self, env, state):
        """CANARY (card verdict, verbatim): ``test_order_plot_module_audit.py:542``
        is ``p_bid=49.99 / p_ask=50.01`` = **2 ticks, EVEN**, so its mid is
        exactly on-tick (50.00) and ``floor`` is a NO-OP there.

        It is therefore **EXPECTED NOT TO MOVE**. Asserted HERE so movement is a
        STOP condition raised by this story's own file: if this fails, something
        other than (b2) has changed and the developer must stop.
        """
        bid_int = 1_004_899        # 49.99
        ask_int = 1_004_901        # 50.01
        assert (ask_int - bid_int) % 2 == 0, "canary spread must be EVEN"
        assert (bid_int + ask_int) // 2 == 1_004_900          # exactly 50.00
        _pa, _pb, mid = _top_of_book(env, state, bid_int, ask_int)
        assert abs(float(np.asarray(mid)) - 50.00) <= 1e-4
        assert _mid_submit_tick(env, state, bid_int, ask_int) == 1_004_900

    @pytest.mark.jit
    def test_zero_spread_mid_is_the_touch(self, env, state):
        """Degenerate 0-tick spread (the ``50.00/50.00`` fixture in
        ``test_order_plot_module.py``): the int mid IS the touch, floor is
        vacuous. Boundary case of the ``//`` rule."""
        t = 1_004_900
        assert _mid_submit_tick(env, state, t, t) == t

    @pytest.mark.jit
    def test_wide_even_spread_fixture_is_preserved(self, env, state):
        """The 550-tick EVEN spread from ``test_touch_relative_action.py:196``
        (47.25 / 52.75, mid 50.00). Also EVEN, also expected unchanged — the
        second existing MID fixture whose parity masked this class."""
        bid_int = 1_004_725       # 47.25
        ask_int = 1_005_275       # 52.75
        assert (ask_int - bid_int) == 550 and (ask_int - bid_int) % 2 == 0
        assert _mid_submit_tick(env, state, bid_int, ask_int) == 1_005_000


# ===========================================================================
# Contract (4) — the two formerly-naive traced sites decode ONE shared tick to
# the IDENTICAL float32, closing the drift class WF-18 / WF-20 hit.
# ===========================================================================


@_skip
class TestDecodeSitesAgree:

    @pytest.mark.jit
    def test_top_of_book_and_passive_fill_agree_on_the_same_tick(self, env, state):
        """``_top_of_book``'s ``p_bid`` and ``_extract_passive_fill``'s implied
        price must be the SAME float32 bits for the same tick.

        The passive-fill price is recovered EXACTLY (no division rounding):
        with 40 qty ticks = 4.0 MW and d = 0.25 h, ``qty_mw * d == 1.0``
        exactly in float32, so a one-row passive BUY yields
        ``passive_realized_eur == -price_eur`` bit-for-bit.
        """
        n_trades = int(np.asarray(state.book.lob.trades).shape[0])
        n_cols = int(np.asarray(state.book.lob.trades).shape[1])
        before = np.full((n_trades, n_cols), -1, dtype=np.int32)

        for tick in (1_004_900, 999_123, 1_234_567, 3, MAX_LEGAL_TICK - 10):
            after = before.copy()
            after[0, TRADES_P] = tick
            after[0, TRADES_Q] = PASSIVE_QTY_TICKS          # >0 => passive BUY
            after[0, TRADES_PASS_TID] = int(AGENT_TRADER_ID)

            _inv, realized, _row = env._extract_passive_fill(
                jnp.asarray(before), jnp.asarray(after), jnp.int32(0)
            )
            implied = np.asarray(-jnp.asarray(realized, jnp.float32))
            p_ask, p_bid, _mid = _top_of_book(env, state, tick, tick + 2)
            assert float(implied) == float(np.asarray(p_bid)), (
                f"passive-fill decode {float(implied):.9f} != top-of-book "
                f"decode {float(np.asarray(p_bid)):.9f} at tick {tick}"
            )
            # ...and both land on the tick they came from.
            assert int(np.asarray(_reencode(p_bid))) == tick
            assert int(np.asarray(_reencode(implied))) == tick
# ===========================================================================
# Contract (5) — the CROSS-DEVICE twin of (1) + (3) + (3b). This is the lock
# that IS the card's headline promise. @pytest.mark.gpu per the device policy in
# docs/omiecid/device-policy.md (conftest auto-skips on CUDA-hidden hosts).
#
# WHY EVERY LEG HERE IS JITTED WITH AN EXPLICIT ``backend=``, measured while
# authoring this file (2026-09-10, CUDA_VISIBLE_DEVICES=1):
#   * EAGER, op-by-op, CPU and GPU already AGREE bitwise (0/800 disagreements on
#     a random odd-spread mid sweep) — eager gives XLA no fusion opportunity, so
#     an eager cross-device comparison is NOT discriminating and would be a
#     vacuously-green test.
#   * Under ``jax.jit(..., backend=...)`` the naive decode diverges on
#     **147 107 / 285 686 ticks (51.5%)**, and the naive jitted ``mid`` anchor
#     diverges on **144/400 (36%) of odd-spread pairs bitwise / 48/400 (12%) on
#     the SUBMIT TICK** — consistent with the card's measured 15.7%.
# So: jit-with-explicit-backend is the path that carries the defect, and it is
# the path this class pins.
# ===========================================================================


@_skip
@pytest.mark.gpu
@pytest.mark.slow
class TestPriceDecodeCrossDevice:
    """slow: drives real GPU compiles + host<->device transfers of a ~285k-tick
    grid, so it is isolated from the fast suite despite being sub-second of
    actual math.

    Every assertion is EQUALITY WITH NO TOLERANCE, licensed by the exhaustive
    measurement recorded in sprint-status.md: the int-domain decode is
    md5-IDENTICAL CPU vs GPU across all 1 999 801 legal ticks, and the floored
    int mid re-encodes to ``mid_int`` 100.00% on both backends.
    """

    # Module-scope helpers would sit OUTSIDE any marker scope, so the two
    # jit-with-explicit-backend helpers live here as staticmethods, covered
    # by this class-level ``pytest.mark.jit`` (they DO trigger real compiles).
    pytestmark = pytest.mark.jit

    @staticmethod
    def _decode_jit(env, ticks, backend):
        """Deliverable (a)'s helper, jitted onto an explicit backend."""
        if not hasattr(env, "_decode_price_eur"):
            pytest.fail(
                "OMIE-PRICE-DECODE-FMA-1 deliverable (a) missing: Session has no "
                "`_decode_price_eur(price_int)` (see _decode())."
            )
        f = jax.jit(lambda t: env._decode_price_eur(t), backend=backend)
        return np.asarray(jax.device_get(f(jnp.asarray(ticks, jnp.int32))))

    @staticmethod
    def _mid_jit(env, base_lob, bid_int, ask_int, backend):
        """``_top_of_book``'s mid anchor, jitted onto an explicit backend."""
        f = jax.jit(lambda l: env._top_of_book(l)[6], backend=backend)
        return float(np.asarray(jax.device_get(f(_synthetic_lob(base_lob, bid_int, ask_int)))))

    @pytest.mark.jit
    def test_decode_is_bitwise_identical_on_cpu_and_gpu(self, env):
        """RED TODAY: the naive form differs on 147 107/285 686 ticks (51.5%)
        between ``backend="cpu"`` and ``backend="gpu"`` — XLA:CPU contracts the
        multiply-add into an FMA, XLA:GPU (at this shape) does not."""
        ticks = _tick_grid()
        cpu = self._decode_jit(env, ticks, "cpu")
        gpu = self._decode_jit(env, ticks, "gpu")
        assert np.array_equal(cpu, gpu), (
            f"CPU vs GPU decode differ bitwise on {int((cpu != gpu).sum())}/"
            f"{cpu.size} ticks — the int-domain form is md5-identical across "
            "backends by measurement"
        )

    @pytest.mark.jit
    def test_decode_meets_the_relative_bound_on_gpu(self, env):
        """The SAME ``2e-7 * max(|p|,1)`` bound as the CPU leg — the card
        requires the GPU twin assert the SAME numbers, not a looser variant."""
        ticks = _tick_grid()
        truth = _truth_eur(ticks)
        got = self._decode_jit(env, ticks, "gpu").astype(np.float64)
        rel = np.abs(got - truth) / np.maximum(np.abs(truth), 1.0)
        worst = float(rel.max())
        assert worst <= DECODE_REL_TOL, (
            f"GPU decode relative error {worst:.6e} exceeds {DECODE_REL_TOL:.1e}"
        )

    @pytest.mark.jit
    def test_re_encode_round_trip_holds_on_gpu(self, env):
        """100.00% exhaustively on GPU per the measurement => no tolerance."""
        ticks = _tick_grid()
        back = np.asarray(_reencode(self._decode_jit(env, ticks, "gpu")))
        bad = int((back != ticks).sum())
        assert bad == 0, f"{bad}/{ticks.size} GPU-decoded prices miss their tick"

    @pytest.mark.jit
    def test_gpu_decode_is_bit_identical_under_jit_and_eager(self, env):
        """The FMA lock (2), on GPU. Eager and jitted GPU decode must agree
        bitwise — the card records ``eager == jit`` on BOTH backends."""
        ticks = _tick_grid()
        gpu_dev = jax.devices("gpu")[0]
        with jax.default_device(gpu_dev):
            eager = np.asarray(jax.device_get(
                _decode(env, jax.device_put(jnp.asarray(ticks, jnp.int32), gpu_dev))
            ))
        assert np.array_equal(eager, self._decode_jit(env, ticks, "gpu"))

    @pytest.mark.jit
    @pytest.mark.parametrize("bid_int,spread", _MID_CASES)
    def test_mid_anchor_submit_tick_is_the_floored_int_mid_on_gpu(
        self, env, state, bid_int, spread
    ):
        """**THE test that closes the card's headline promise.** The decode fix
        ALONE leaves the MID submit tick CPU/GPU-divergent on a measured 15.7%
        of odd-spread pairs, because the half-tick EUR value is not
        float32-representable and ``mid*100`` straddles ``.5`` differently per
        backend. Only the INT-domain anchor (b2) makes it device-independent.

        RED TODAY on GPU: today's GPU agrees with floor on only 85.3% / 87.5%
        of odd-spread cases (and 48/400 = 12% of jitted odd-spread pairs submit
        a DIFFERENT tick on GPU than on CPU).
        """
        ask_int = bid_int + spread
        mid = self._mid_jit(env, state.book.lob, bid_int, ask_int, "gpu")
        tick = int(np.asarray(_reencode(mid)))
        assert tick == (bid_int + ask_int) // 2, (
            f"GPU MID tick {tick} != floor((bid+ask)/2)="
            f"{(bid_int + ask_int) // 2} at bid_int={bid_int}, spread={spread} "
            f"({'ODD' if spread % 2 else 'EVEN'})"
        )

    @pytest.mark.jit
    @pytest.mark.parametrize("bid_int,spread", _MID_CASES)
    def test_mid_anchor_bits_are_identical_on_cpu_and_gpu(
        self, env, state, bid_int, spread
    ):
        """The anchor itself — not just the tick it rounds to — must be the same
        float32 bits on both backends. This survives even if a future re-encode
        path changes its rounding. RED TODAY: 144/400 (36%) of jitted
        odd-spread pairs disagree bitwise."""
        ask_int = bid_int + spread
        cpu_mid = self._mid_jit(env, state.book.lob, bid_int, ask_int, "cpu")
        gpu_mid = self._mid_jit(env, state.book.lob, bid_int, ask_int, "gpu")
        assert cpu_mid == gpu_mid, (
            f"mid differs across backends at bid_int={bid_int}, spread={spread}"
            f": cpu={cpu_mid:.9f} gpu={gpu_mid:.9f}"
        )
