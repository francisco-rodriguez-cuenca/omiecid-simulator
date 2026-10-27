"""AUDIT tests for OMIE-AGENT-TIME-PRIORITY-1 -- the agent order is stamped
from the post-background book's arrival counter (``book.ts_counter``), not
from the minutes-from-GOT env clock.

Independent post-implementation verification (audit mode, orchestrate.md
Step 5). Derived from the story card in ``docs/omiecid/agile-plan.md`` and
``docs/omiecid/market-spec.md`` §5.2 ("All timestamps ... are assigned from a
single global queue (unique, sequential)"). The locked test file
``test_agent_time_priority.py`` was NOT read.

Contract under test (the global-queue invariant):
  (G1) the agent order's timestamp == ``ts_counter`` of the book AFTER this
       step's background replay (the "pre-submit" counter), independent of
       ``state.time``, ``gct_minutes_from_got`` and ``tape_seconds_span``;
  (G2) that stamp is STRICTLY GREATER than the SEC of every order resting in
       the book at submit (so the agent queues behind every earlier arrival at
       its price, on both the rest-only and the fill-then-rest path);
  (G3) successive agent orders get strictly increasing stamps, so agent
       orders never tie with each other;
  (G4) tape arrivals stamped after the agent queue behind it (a later
       aggressor fills earlier arrivals, then the agent, then later arrivals);
  (G5) a no-order rollout is unchanged: its final book equals the pure
       replay of the consumed tape rows.

The fixtures VARY the incidental constants a single-seed test would hold
fixed: orders_per_step (1..5), side, price level (negative, sub-euro,
near the band maximum, off-round), number of earlier resting orders (1..4),
same-second and non-monotone tape seconds, tape-second offsets up to
~86,000, background agent ids, background OID magnitudes (1 .. ~2e9),
iceberg competitors and multi-step agent/background interleavings.

Out of scope (card "NOT in scope", residual R1): a LATER tape row whose raw
second is <= the agent stamp still rests ahead of the agent (tape-side
stamping is left to a separate user decision). No test here asserts either
direction for that case.

Book rows are read through the jaxob layout ``[P, Q, OID, TID, SEC, NSEC]``
(``OrderSideFeat``); the trade tape through ``TradesFeat``. Empty rows carry
``OID <= 0`` (book) / ``-1`` (trades).

Runtime: every distinct tape is a distinct static jit key for ``Session``
(``reset`` / ``step_env``), so each env-driven test pays a fresh CPU compile
(several seconds) -- the module is therefore marked ``slow``. JAX caches are
cleared after every test to bound the number of live executables.

Device policy: CPU (conftest pins the cpu platform).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omiecid_simulator.jaxob import jaxob_constants as cst
from omiecid_simulator.jaxomiecid.omie_data_loader import _replay_orders
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.omie_session import AgentOrder, Session
from omiecid_simulator.omiecid.order_types import ExecRestriction, Status, Validity

# slow: each test builds at least one fresh Session (new static jit key ->
# full reset/step_env CPU compile, several seconds per distinct tape).
pytestmark = [pytest.mark.jit, pytest.mark.slow]

PC = OMIEProductConfig()
MC = OMIEMarketConfig()

BUY, SELL = 0, 1
NON = int(ExecRestriction.NON)
IOC = int(ExecRestriction.IOC)
ICE = int(ExecRestriction.ICE)

AGENT_OID_BASE = 2_000_000_000
AGENT_TID = 1_000_000

_P = cst.OrderSideFeat.P.value
_Q = cst.OrderSideFeat.Q.value
_OID = cst.OrderSideFeat.OID.value
_SEC = cst.OrderSideFeat.SEC.value
_T_Q = cst.TradesFeat.Q.value
_T_PASS = cst.TradesFeat.PASS_OID.value
_T_PASS_TID = cst.TradesFeat.PASS_TID.value

# The two participant order sizes (MW) the scenarios use.
QTY_BUCKETS = (3.0, 7.0)
# delivery duration of the env's synthetic 15-minute contract (hours): the
# settlement value of a fill is Q * p * d.
D_H = 0.25


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()


def enc_price(p: float) -> int:
    return int(round((p - PC.price_min) / PC.price_tick))


def enc_qty(q: float) -> int:
    return int(round(q / PC.qty_tick))


def row(side, price, qty, oid, ts, er=NON, aid=777, peak=0.0):
    return [
        0, int(side), enc_price(price), enc_qty(qty), int(oid), int(aid),
        int(ts), int(er), int(Validity.GFS), 0, int(Status.ACTIVE),
        enc_qty(peak), 0,
    ]


PAD = [0, 0, 0, 0, -1, 0, -1, NON, int(Validity.GFS), 0, int(Status.ACTIVE), 0, 0]


def order(side: int, price: float, qty: float):
    """A participant order spec (side, price EUR/MWh, quantity MW), resting
    NON GFS. ``None`` in an order list submits nothing."""
    return (int(side), float(price), float(qty))


def agent_order(spec):
    if spec is None:
        return AgentOrder.none()
    side, price, qty = spec
    return AgentOrder.create(side=side, price=price, quantity=qty)


def _span_keeping_session_open(padded, op, gct):
    """``tape_seconds_span`` for a late-second tape, or None.

    The Session ignores calls once ``done`` (time >= GCT) fires. With the
    default span (the tape's max second) a tape whose seconds carry a large
    offset maps an EARLY slice onto GCT, ending the session before its last
    real slice. Only then is a span returned: the midpoint of the spans that
    map the last real slice onto GCT and every earlier slice below it (same
    float32 mapping as the session's span-scaled clock). The tests that use
    it read no clock value.
    """
    ts = np.asarray(padded, dtype=np.int64)[:, 6]
    slice_max = [int(ts[k:k + op].max()) for k in range(0, len(ts), op)]
    real = [k for k, t in enumerate(slice_max) if t > 0]
    last, earlier = real[-1], real[:-1]

    def minute(t, span):
        return int(np.round(np.float32(t) * (np.float32(gct) / np.float32(span))))

    span = float(max(int(ts.max()), 1))
    if all(minute(slice_max[k], span) < gct for k in earlier):
        return None
    lo = max(slice_max[k] for k in earlier) * gct / (gct - 0.5)
    hi = slice_max[last] * gct / (gct - 0.5)
    span = (lo + hi) / 2.0
    assert all(minute(slice_max[k], span) < gct for k in earlier)
    assert minute(slice_max[last], span) == gct
    return span


def make_env(rows, op, open_session=False, **extra):
    """``open_session=True``: on a late-second tape, keep the session open
    until its last real slice (see ``_span_keeping_session_open``)."""
    n = max(8 * op, int(np.ceil(len(rows) / op)) * op)
    padded = list(rows) + [PAD] * (n - len(rows))
    cfg = {
        "tapes": np.asarray([padded], dtype=np.int32),
        "orders_per_step": op,
        "gct_minutes_from_got": 60,
    }
    cfg.update(extra)
    if open_session and "tape_seconds_span" not in extra:
        span = _span_keeping_session_open(padded, op, cfg["gct_minutes_from_got"])
        if span is not None:
            cfg["tape_seconds_span"] = span
    return Session(cfg)


def rollout(env, orders, stop_at_done=False):
    """Step ``env`` through ``orders``; return per-step records.

    Each record carries the pre-step state, the post-step state, the step's
    realized cash (EUR), the done flag, the ``FillReport``, and the
    INDEPENDENT oracle book obtained by replaying this step's tape slice onto
    the pre-step book with the public loader replay (the "post-background,
    pre-submit" book).
    """
    s = env.reset(jax.random.PRNGKey(0))
    tape = np.asarray(env.tapes[int(s.tape_idx)])
    recs = []
    for spec in orders:
        cur = int(s.tape_cursor)
        sl = jnp.asarray(tape[cur:cur + env.orders_per_step])
        bg_book, _ = _replay_orders(s.book, sl, PC, MC)
        s2, report, d = env.step_env(
            jax.random.PRNGKey(0), s, agent_order(spec)
        )
        recs.append(dict(pre=s, post=s2, bg=bg_book,
                         cash=float(report.cash_delta_eur),
                         done=bool(d), report=report))
        s = s2
        if stop_at_done and recs[-1]["done"]:
            break
    return recs


def side_rows(book, side):
    a = np.asarray(book.lob.bids if side == BUY else book.lob.asks)
    return a[a[:, _OID] > 0]


def all_rows(book):
    return np.concatenate([side_rows(book, BUY), side_rows(book, SELL)], axis=0)


def agent_row(book, oid):
    both = all_rows(book)
    r = both[both[:, _OID] == oid]
    assert r.shape[0] == 1, f"expected one resting row for agent oid {oid}, got {r}"
    return r[0]


def new_trades(pre_book, post_book):
    t0 = np.asarray(pre_book.lob.trades)
    t1 = np.asarray(post_book.lob.trades)
    n0 = int(np.sum(t0[:, _T_PASS] != -1))
    n1 = int(np.sum(t1[:, _T_PASS] != -1))
    return t1[n0:n1]


def fills(pre_book, post_book):
    """Ordered list of (passive_oid, |Q| MW) for the step's new trades."""
    return [(int(t[_T_PASS]), abs(int(t[_T_Q])) * PC.qty_tick)
            for t in new_trades(pre_book, post_book)]


def assert_global_queue_invariant(rec, agent_oid):
    """(G1) + (G2) on a submit step whose agent order rests."""
    counter = int(rec["bg"].ts_counter)
    ar = agent_row(rec["post"].book, agent_oid)
    assert int(ar[_SEC]) == counter, (
        f"agent SEC {int(ar[_SEC])} != post-background ts_counter {counter}"
    )
    others = all_rows(rec["post"].book)
    others = others[others[:, _OID] != agent_oid]
    if others.shape[0]:
        assert int(ar[_SEC]) > int(others[:, _SEC].max()), (
            f"agent SEC {int(ar[_SEC])} not strictly greater than every resting "
            f"SEC {sorted(others[:, _SEC].tolist())}"
        )
    # every order resting at submit (the oracle post-background book)
    bg_rows = all_rows(rec["bg"])
    if bg_rows.shape[0]:
        assert int(ar[_SEC]) > int(bg_rows[:, _SEC].max())
    return int(ar[_SEC])


# ---------------------------------------------------------------------------
# (G1)+(G2)+(G4-earlier) -- rest-only PASSIVE join behind an earlier queue,
# over a spread of incidental constants.
# ---------------------------------------------------------------------------

# Each case: (queue side, level P, earlier queue [(qty, raw_ts)], opposite
# (price, qty, raw_ts), orders_per_step, tape-second offset, background agent
# id, background oid base, agent qty bucket, aggressor overshoot x (MW taken
# from the agent after the whole earlier queue), strict_fifo_among_earlier).
QUEUE_CASES = {
    "buy_op1_single": (BUY, 50.00, [(10.0, 600)], (50.40, 5.0, 660), 1, 0,
                       777, 311, 0, 2.0, True),
    "sell_op2_two_offround": (SELL, 61.37, [(8.0, 1210), (1.5, 1250)],
                              (60.90, 4.0, 1300), 2, 0, 5, 2203, 0, 1.5, True),
    "buy_negative_price_op3_offset_bigoid": (
        BUY, -12.34, [(0.1, 100), (2.2, 101), (4.4, 102)], (-11.00, 1.0, 103),
        3, 54_600, 999_999, 1_999_999_777, 1, 6.9, True),
    "sell_near_band_max_op4_late_second": (
        SELL, 9998.99, [(20.0, 86_000)], (9998.00, 1.0, 86_010), 4, 0,
        42, 1, 1, 0.1, True),
    "buy_subeuro_op5_same_second_four": (
        BUY, 0.01, [(1.0, 5000), (1.0, 5000), (1.0, 5000), (1.0, 5000)],
        (0.50, 1.0, 5000), 5, 0, 13, 40_000_001, 0, 2.9, False),
    "sell_nonmonotone_seconds_op3_offset": (
        SELL, 50.00, [(2.0, 700), (3.0, 650)], (49.10, 1.0, 640), 3, 12_345,
        777, 9_000_000, 1, 4.0, False),
}


def _queue_scenario(case):
    (qside, P, queue, opp, op, off, aid, oid0, qb, x, strict) = case
    rows = []
    q_oids = []
    for i, (q, ts) in enumerate(queue):
        # OIDs DESCEND as arrival order advances (so "lowest OID first" and
        # "earliest arrival first" disagree).
        oid = oid0 + 50 - i
        q_oids.append(oid)
        rows.append(row(qside, P, q, oid, off + ts, aid=aid))
    rows.append(row(1 - qside, opp[0], opp[1], oid0 + 100, off + opp[2], aid=aid))
    n_prefix = len(rows)
    submit_step = (n_prefix - 1) // op
    rows += [PAD] * ((submit_step + 1) * op - n_prefix)
    max_prefix_ts = off + max([ts for _, ts in queue] + [opp[2]])
    total_q = sum(q for q, _ in queue)
    rows.append(row(1 - qside, P, total_q + x, oid0 + 200,
                    max_prefix_ts + 400, er=IOC, aid=aid))
    # The agent joins the queue at its own touch P.
    orders = [None] * submit_step + [order(qside, P, QTY_BUCKETS[qb]), None]
    return dict(rows=rows, op=op, orders=orders, submit_step=submit_step,
                q_oids=q_oids, queue=queue, P=P, x=x, qside=qside,
                agent_qty=QTY_BUCKETS[qb], max_prefix_ts=max_prefix_ts,
                strict=strict)


@pytest.mark.parametrize("name", sorted(QUEUE_CASES))
def test_passive_join_rests_behind_every_earlier_same_price_order(name):
    sc = _queue_scenario(QUEUE_CASES[name])
    env = make_env(sc["rows"], sc["op"], open_session=True)
    recs = rollout(env, sc["orders"])
    s = sc["submit_step"]
    agent_oid = AGENT_OID_BASE + s

    # -- submit step: stamp == post-background counter, > every resting SEC
    sub = recs[s]
    stamp = assert_global_queue_invariant(sub, agent_oid)
    # hand-derived: counter = max over replayed rows of (raw_ts + 1)
    assert stamp == sc["max_prefix_ts"] + 1
    ar = agent_row(sub["post"].book, agent_oid)
    assert int(ar[_P]) == enc_price(sc["P"])
    assert int(ar[_Q]) == enc_qty(sc["agent_qty"])
    # no agent fill on a rest-only PASSIVE join
    assert fills(sub["bg"], sub["post"].book) == []
    # stamp does NOT come from the minutes clock
    assert stamp != int(sub["pre"].time)

    # -- aggressor step: the whole earlier queue fills first, then the agent
    agg = recs[s + 1]
    got = fills(agg["pre"].book, agg["post"].book)
    k = len(sc["q_oids"])
    assert len(got) == k + 1, got
    if sc["strict"]:
        assert [o for o, _ in got[:k]] == sc["q_oids"]
    else:
        assert sorted(o for o, _ in got[:k]) == sorted(sc["q_oids"])
    for (o, q) in got[:k]:
        want = dict(zip(sc["q_oids"], [qq for qq, _ in sc["queue"]]))[o]
        assert q == pytest.approx(want, abs=1e-6)
    assert got[k][0] == agent_oid
    assert got[k][1] == pytest.approx(sc["x"], abs=1e-6)

    # agent inventory / realized cash on the passive fill
    sign_inv = 1.0 if sc["qside"] == BUY else -1.0
    assert float(agg["post"].agent_inventory[0]) == pytest.approx(
        sign_inv * sc["x"], abs=1e-5)
    if sc["P"] > 0:
        # BUY pays (negative), SELL receives (positive): cash = -inv*P*d
        assert agg["cash"] == pytest.approx(
            -sign_inv * sc["x"] * sc["P"] * D_H, abs=0.02)

    # post-state: agent keeps (agent_qty - x) at P, earlier queue gone
    post = all_rows(agg["post"].book)
    for o in sc["q_oids"]:
        assert not np.any(post[:, _OID] == o)
    left = agent_row(agg["post"].book, agent_oid)
    assert int(left[_Q]) == enc_qty(sc["agent_qty"]) - enc_qty(sc["x"])
    assert int(left[_SEC]) == stamp  # partial passive fill keeps the stamp


# ---------------------------------------------------------------------------
# (G4) later arrivals queue BEHIND the agent (monotone tape seconds)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "qside,P,n_later,op,off",
    [
        (BUY, 50.00, 1, 1, 0),
        (SELL, 73.21, 3, 2, 40_000),
        (BUY, 0.37, 2, 3, 7),
    ],
    ids=["buy_one_later_op1", "sell_three_later_op2_offset", "buy_two_later_op3"],
)
def test_later_same_price_arrivals_queue_behind_agent(qside, P, n_later, op, off):
    opp_price = P + 0.40 if qside == BUY else P - 0.40
    rows = [
        row(qside, P, 1.0, 71, off + 300),
        row(1 - qside, opp_price, 1.0, 72, off + 310),
    ]
    rows += [PAD] * ((int(np.ceil(len(rows) / op))) * op - len(rows))
    submit_step = len(rows) // op - 1
    later_oids = []
    base_ts = off + 5_000
    for j in range(n_later):
        oid = 500 - j
        later_oids.append(oid)
        rows.append(row(qside, P, 0.5, oid, base_ts + 10 * j))
    rows += [PAD] * ((int(np.ceil(len(rows) / op))) * op - len(rows))
    later_steps = len(rows) // op - (submit_step + 1)
    agent_qty = 3.0
    take = 1.0 + agent_qty + 0.5 * n_later  # sweeps the whole level
    rows.append(row(1 - qside, P, take, 999, base_ts + 1_000, er=IOC))
    orders = ([None] * submit_step + [order(qside, P, 3.0)]
              + [None] * later_steps + [None])
    env = make_env(rows, op)
    recs = rollout(env, orders)
    agent_oid = AGENT_OID_BASE + submit_step
    stamp = assert_global_queue_invariant(recs[submit_step], agent_oid)
    assert stamp == off + 311

    # later arrivals rest at their raw second, strictly after the agent stamp
    before_agg = recs[-1]["pre"].book
    for j, oid in enumerate(later_oids):
        r = side_rows(before_agg, qside)
        r = r[r[:, _OID] == oid]
        assert r.shape[0] == 1
        assert int(r[0][_SEC]) == base_ts + 10 * j
        assert int(r[0][_SEC]) > stamp
    # agent row untouched by the later arrivals
    assert int(agent_row(before_agg, agent_oid)[_SEC]) == stamp

    got = fills(recs[-1]["pre"].book, recs[-1]["post"].book)
    assert [o for o, _ in got] == [71, agent_oid] + later_oids
    assert got[1][1] == pytest.approx(agent_qty, abs=1e-6)


# ---------------------------------------------------------------------------
# (G3)+(G4) strict global FIFO across an interleaving of background and agent
# arrivals at one price.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qside,op", [(BUY, 1), (SELL, 2)], ids=["buy_op1", "sell_op2"])
def test_interleaved_background_and_agent_arrivals_fill_in_arrival_order(qside, op):
    P = 44.44
    opp = P + 1.0 if qside == BUY else P - 1.0

    def slice_(*rs):
        out = list(rs)
        return out + [PAD] * (op - len(out))

    B1 = row(qside, P, 1.0, 1_234_567, 1000, aid=31)
    O = row(1 - qside, opp, 1.0, 1_234_568, 1001, aid=31)
    B2 = row(qside, P, 2.0, 17, 3000, aid=32)
    B3 = row(qside, P, 4.0, 1_999_999_999, 6000, aid=33)
    agg_q = 1.0 + 3.0 + 2.0 + 7.0 + 4.0
    AGG = row(1 - qside, P, agg_q, 88, 9000, er=IOC, aid=34)
    rows = (slice_(B1, O) if op >= 2 else [B1, O])
    # step layout (op=1): 0:B1 1:O(+A1) 2:B2(+A2) 3:B3 4:AGG
    # step layout (op=2): 0:B1,O(+A1) 1:B2(+A2) 2:B3 3:AGG
    rows += slice_(B2) + slice_(B3) + slice_(AGG)
    if op == 1:
        orders = [None, order(qside, P, 3.0), order(qside, P, 7.0), None, None]
        a1_step, a2_step = 1, 2
    else:
        orders = [order(qside, P, 3.0), order(qside, P, 7.0), None, None]
        a1_step, a2_step = 0, 1
    env = make_env(rows, op)
    recs = rollout(env, orders)
    a1, a2 = AGENT_OID_BASE + a1_step, AGENT_OID_BASE + a2_step
    s1 = assert_global_queue_invariant(recs[a1_step], a1)
    s2 = assert_global_queue_invariant(recs[a2_step], a2)
    assert s1 == 1002
    assert s2 == 3001
    assert s1 < 3000 < s2  # B2 sits strictly between the two agent orders

    got = fills(recs[-1]["pre"].book, recs[-1]["post"].book)
    assert [o for o, _ in got] == [1_234_567, a1, 17, a2, 1_999_999_999]
    assert [q for _, q in got] == pytest.approx([1.0, 3.0, 2.0, 7.0, 4.0], abs=1e-6)
    inv = 10.0 if qside == BUY else -10.0
    assert float(recs[-1]["post"].agent_inventory[0]) == pytest.approx(inv, abs=1e-5)


# ---------------------------------------------------------------------------
# (G3) back-to-back agent orders with NO later-stamped background arrival
# between them never tie.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filler", ["same_second_far_rows", "pad_rows"])
def test_back_to_back_agent_orders_get_strictly_increasing_stamps(filler):
    P = 50.00
    rows = [
        row(BUY, P, 10.0, 311, 600),
        row(SELL, 50.40, 5.0, 509, 600),
    ]
    if filler == "same_second_far_rows":
        # far, non-crossing, at an EARLIER raw second than the counter
        rows += [row(SELL, 70.00, 1.0, 9901, 600), row(SELL, 70.10, 1.0, 9903, 600)]
    else:
        rows += [PAD, PAD]
    rows += [row(SELL, P, 10.0 + 3.0 + 3.0 + 3.0, 347, 900, er=IOC)]
    # op1: steps 0..4 ; agent joins the 50.00 bid on steps 1, 2, 3
    b = order(BUY, P, 3.0)
    orders = [None, b, b, b, None]
    env = make_env(rows, 1)
    recs = rollout(env, orders)
    stamps = []
    for st in (1, 2, 3):
        stamps.append(assert_global_queue_invariant(recs[st], AGENT_OID_BASE + st))
    assert stamps == [601, 602, 603]
    got = fills(recs[4]["pre"].book, recs[4]["post"].book)
    assert [o for o, _ in got] == [311] + [AGENT_OID_BASE + st for st in (1, 2, 3)]
    assert [q for _, q in got] == pytest.approx([10.0, 3.0, 3.0, 3.0], abs=1e-6)


# ---------------------------------------------------------------------------
# (G1)+(G2) fill-then-rest (CROSS) and MID placements carry the counter stamp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", [BUY, SELL], ids=["buy", "sell"])
def test_cross_fill_then_rest_residual_stamped_counter_and_ahead_of_later(side):
    # CROSS 7.0 against 2.0 at the opposite touch -> fills 2.0, rests 5.0 at
    # that price on the agent's own side (R3 restamp to the order stamp).
    X = 50.40 if side == BUY else 49.60
    own = 49.60 if side == BUY else 50.40
    rows = [
        row(1 - side, X, 2.0, 60_001, 20_000, aid=4),
        row(1 - side, X + (0.5 if side == BUY else -0.5), 9.0, 60_002, 20_050, aid=4),
        row(side, own, 1.0, 60_003, 20_070, aid=4),
        PAD,  # step 3: agent CROSS (book unchanged by background)
        row(side, X, 1.0, 60_004, 21_000, aid=4),   # later same-price arrival
        row(1 - side, X, 6.0, 60_005, 22_000, er=IOC, aid=4),
    ]
    orders = [None, None, None, order(side, X, 7.0), None, None]
    env = make_env(rows, 1)
    recs = rollout(env, orders)
    agent_oid = AGENT_OID_BASE + 3
    sub = recs[3]
    stamp = assert_global_queue_invariant(sub, agent_oid)
    assert stamp == 20_071
    got = fills(sub["bg"], sub["post"].book)
    assert got == [(60_001, pytest.approx(2.0, abs=1e-6))]
    ar = agent_row(sub["post"].book, agent_oid)
    assert int(ar[_Q]) == enc_qty(5.0)
    assert int(ar[_P]) == enc_price(X)

    # later arrival at X rests behind; aggressor 6.0 takes agent 5.0 then 1.0
    got2 = fills(recs[5]["pre"].book, recs[5]["post"].book)
    assert [o for o, _ in got2] == [agent_oid, 60_004]
    assert [q for _, q in got2] == pytest.approx([5.0, 1.0], abs=1e-6)


def test_mid_placement_new_level_carries_counter_stamp():
    rows = [
        row(BUY, 48.00, 1.0, 5, 30_000, aid=1),
        row(SELL, 52.00, 1.0, 6, 30_001, aid=1),
        row(BUY, 47.00, 1.0, 7, 29_999, aid=1),  # non-monotone raw second
    ]
    env = make_env(rows, 3)
    # SELL at the mid of 48.00 / 52.00.
    recs = rollout(env, [order(SELL, 50.00, 3.0)])
    agent_oid = AGENT_OID_BASE + 0
    stamp = assert_global_queue_invariant(recs[0], agent_oid)
    assert stamp == 30_002
    assert int(agent_row(recs[0]["post"].book, agent_oid)[_P]) == enc_price(50.00)


def test_agent_stamp_exceeds_resting_iceberg_slice_stamp():
    # An iceberg competitor at the agent's price: its visible slice carries a
    # book-assigned stamp; the agent must still be strictly behind it.
    P = 50.00
    rows = [
        row(BUY, P, 6.0, 801, 600, er=ICE, peak=2.0, aid=9),
        row(SELL, 50.40, 1.0, 802, 610, aid=9),
    ]
    env = make_env(rows, 2)
    recs = rollout(env, [order(BUY, P, 3.0)])
    agent_oid = AGENT_OID_BASE + 0
    assert_global_queue_invariant(recs[0], agent_oid)
    bids = side_rows(recs[0]["post"].book, BUY)
    ice = bids[bids[:, _OID] != agent_oid]
    assert ice.shape[0] >= 1
    assert int(agent_row(recs[0]["post"].book, agent_oid)[_SEC]) > int(ice[:, _SEC].max())


# ---------------------------------------------------------------------------
# (G1) the stamp reads the BOOK, not the clock
# ---------------------------------------------------------------------------

_BASE_ROWS = [
    (BUY, 50.00, 10.0, 311, 600, NON),
    (SELL, 50.40, 5.0, 509, 660, NON),
    (SELL, 50.00, 12.0, 347, 900, IOC),
]


@pytest.mark.parametrize(
    "shift,extra",
    [
        (0, {}),
        (54_600, {}),
        (12_345, {}),
        (85_000, {}),
        (0, {"gct_minutes_from_got": 45}),
        (0, {"gct_minutes_from_got": 90}),
        (0, {"tape_seconds_span": 86_400}),
        (3, {"tape_seconds_span": 950, "gct_minutes_from_got": 30}),
    ],
    ids=["shift0", "shift54600", "shift12345", "shift85000", "gct45", "gct90",
         "span86400", "shift3_span950_gct30"],
)
def test_stamp_tracks_counter_under_clock_and_offset_changes(shift, extra):
    rows = [row(s, p, q, o, t + shift, er=e) for (s, p, q, o, t, e) in _BASE_ROWS]
    env = make_env(rows, 1, open_session=True, **extra)
    recs = rollout(env, [None, order(BUY, 50.00, 3.0), None])
    agent_oid = AGENT_OID_BASE + 1
    stamp = assert_global_queue_invariant(recs[1], agent_oid)
    assert stamp == 661 + shift
    got = fills(recs[2]["pre"].book, recs[2]["post"].book)
    assert [o for o, _ in got] == [311, agent_oid]
    assert [q for _, q in got] == pytest.approx([10.0, 2.0], abs=1e-6)
    assert float(recs[2]["post"].agent_inventory[0]) == pytest.approx(2.0, abs=1e-5)
    assert recs[2]["cash"] == pytest.approx(-25.00, abs=0.02)


def test_stamp_timestamp_is_book_counter_for_arbitrary_counter_and_clock():
    rows = [row(s, p, q, o, t, er=e) for (s, p, q, o, t, e) in _BASE_ROWS]
    env = make_env(rows, 1)
    recs = rollout(env, [None])
    book = recs[0]["post"].book
    state = recs[0]["post"]
    ref_spec = order(BUY, 50.00, 3.0)
    ref_order, _, _ = env._stamp_order(agent_order(ref_spec), book, state)
    # join the bid, cross (no ask -> 0.0), join the ask (none -> 0.0), hit
    # the bid
    specs = (ref_spec, order(BUY, 0.0, 7.0), order(SELL, 0.0, 3.0),
             order(SELL, 50.00, 7.0))
    for counter in (0, 1, 661, 123_457, 2**31 - 2):
        for clock in (0, 17, 59, 1_000):
            b = book._replace(ts_counter=jnp.int32(counter))
            st = state.replace(time=jnp.int32(clock))
            for spec in specs:
                stamped, is_noop, _ = env._stamp_order(agent_order(spec), b, st)
                assert not bool(is_noop)
                assert stamped.timestamp.dtype == jnp.int32
                assert int(stamped.timestamp) == counter
            # every other field of the stamped order is independent of the
            # counter and the clock (only the stamp moved)
            stamped, _, _ = env._stamp_order(agent_order(ref_spec), b, st)
            for f in ("side", "price", "quantity", "order_id", "agent_id",
                      "exec_restriction", "validity", "status"):
                assert np.asarray(getattr(stamped, f)) == np.asarray(getattr(ref_order, f)), f


# ---------------------------------------------------------------------------
# background rows are never restamped by the agent path
# ---------------------------------------------------------------------------

def test_agent_submit_leaves_background_stamps_untouched():
    rows = [
        row(BUY, 50.00, 2.0, 1001, 4_000, aid=66),
        row(BUY, 50.00, 2.0, 1002, 3_990, aid=66),  # non-monotone
        row(SELL, 50.20, 2.0, 1003, 4_010, aid=66),
        row(SELL, 50.30, 2.0, 1004, 4_020, aid=66),
    ]
    env = make_env(rows, 4)
    recs = rollout(env, [order(BUY, 50.00, 7.0)])
    bg = all_rows(recs[0]["bg"])
    post = all_rows(recs[0]["post"].book)
    post_bg = post[post[:, _OID] < AGENT_OID_BASE]
    key = lambda a: a[np.argsort(a[:, _OID])]
    np.testing.assert_array_equal(key(bg), key(post_bg))
    assert {int(r[_OID]): int(r[_SEC]) for r in post_bg} == {
        1001: 4_000, 1002: 3_990, 1003: 4_010, 1004: 4_020}
    assert assert_global_queue_invariant(recs[0], AGENT_OID_BASE) == 4_021


# ---------------------------------------------------------------------------
# (G5) no-order rollout == pure replay of the consumed rows
# ---------------------------------------------------------------------------

def _leaves_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("variant", ["op1_basic", "op3_iceberg_nonmonotone", "default_env"])
def test_noop_rollout_book_equals_pure_replay(variant):
    if variant == "default_env":
        env = Session.from_test_config()
    elif variant == "op1_basic":
        env = make_env([row(s, p, q, o, t, er=e) for (s, p, q, o, t, e) in _BASE_ROWS], 1)
    else:
        rows = [
            row(BUY, 50.00, 6.0, 11, 60_000, er=ICE, peak=2.0, aid=3),
            row(SELL, 50.50, 3.0, 12, 59_990, aid=4),
            row(BUY, 49.00, 1.0, 13, 60_010, aid=5),
            row(SELL, 50.00, 3.5, 14, 60_020, er=IOC, aid=6),
            row(BUY, 50.50, 1.0, 15, 60_030, er=IOC, aid=7),
            row(SELL, 51.00, 2.0, 16, 60_040, aid=8),
        ]
        env = make_env(rows, 3)
    recs = rollout(env, [None] * env.steps_per_episode, stop_at_done=True)
    assert recs[-1]["done"]
    consumed = len(recs) * env.orders_per_step
    tape_idx = int(recs[0]["pre"].tape_idx)
    tape = jnp.asarray(np.asarray(env.tapes[tape_idx])[:consumed])
    ref, _ = _replay_orders(env._empty_book(), tape, PC, MC)
    _leaves_equal(recs[-1]["post"].book, ref)
    # no agent row ever enters the book on a no-order rollout
    for r in recs:
        assert not np.any(all_rows(r["post"].book)[:, _OID] >= AGENT_OID_BASE)
