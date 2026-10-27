r"""K4 audit twin for OMIE-CERT-C13-1 (anonymity and information visibility), REL-09.

REL-09 retargets C13 from the removed RL observation to the release ``Session``'s
**participant view** (claim text by the domain analyst, ``REL-09-C13-claim``). This
twin was adapted from the r4 audit (``OMIE-CERT-RERUN-3``, after the 2026-10-20
amendment of market-spec v2.14 §4.2.1 Rule 2 *Incoming iceberg* items (2), (4), (6)).
Fresh-context, adversarial audit (independence rules 3/4). The expectation is
re-derived from the rules and the retargeted claim alone and judged against the
EMITTED results JSON, whatever its status. It never opens or executes
``notebooks/certification/C13_anonymity.ipynb`` and never reads ``contract_book.py``,
``iceberg_manager.py``, ``order_validator.py`` or ``settlement.py`` bodies; the views
are used through their public signatures only (``ContractBook.submit_order /
get_anonymous_lob / get_agent_orders / get_agent_trades``, ``Session.reset / step_env
/ top_of_book``, ``AgentOrder.create / none``, ``FillReport``).

The participant view (claim §1.3)
---------------------------------
* after ``reset`` and after every step: ``Session.top_of_book(state)`` and
  ``get_anonymous_lob(state.book)``;
* after every step: the ``FillReport`` and the ``done`` flag;
* at the end of the episode: ``get_agent_trades(book, AGENT_TRADER_ID)``,
  ``get_agent_orders(book, AGENT_TRADER_ID)``, ``state.agent_cash`` and
  ``state.agent_inventory``.
The returned ``state`` is the simulation carry, not a participant reader (F22).

Rules (market-spec v2.14, as amended 2026-10-20)
------------------------------------------------
* §7.1 / §9.2 "LOB per contract" and "Best bid / best ask": the book is shown
  anonymously, visible quantities only; §4.2 "Only the peak size is shown in the order
  book". -> the participant view is bit-identical between world A (foreign resting
  icebergs with hidden remainders) and its twin B (every foreign hidden remainder
  removed, every identifier permuted) on every draw where the PREMISE holds: no tape
  iceberg fills a resting participant order on its entry. A one-tick change of the
  visible book (CTRL) IS seen by ``top_of_book`` (positive control), and by
  ``get_anonymous_lob`` on the same step.
* PREMISE FAILED. §4.2.1 Rule 2 *Incoming iceberg* (1) total quantity and (4) as
  amended: with $M$ the iceberg's swept quantity on entry (the participant's fill $f$
  included, its own icebergs' hidden part included), $k_e = \lfloor M/P \rfloor$,
  $u = M - k_e P$ and $R = T - M$, the remainder rests as instance $k_e$ at
  $b + k_e d$ showing $v_1 = \min(P - u, R)$ and hiding $R - v_1$. The twin B is a
  plain order of the noop world's fills + the noop display, $q_B$, so it rests
  $q_B - M$. The rule-implied A-B gap is $v_1 - (q_B - M)$. This twin derives $T$,
  $h$, $P$ and $q_B$ ITSELF from the engine record (1) and the twin-validity record
  (7), re-derives $k_e$, $u$, $v_1$, the hidden quantity and the gap, and requires the
  whole participant view to be bit-identical where the gap is 0 (every premise-failed
  draw of this construction: $M = f < P$, $u = f$, $v_1 = P - f = q_B - f$), with
  $f_{\max} = $ ``Q_AGENT_MAX`` $\times K$. The RETIRED display $\min(P, R)$ would give
  a gap of $f \ge 1$ on those same draws: the data discriminate.
* §7.2 / §9.2 "Trade book": "Each agent sees only their own matched trades";
  "Counterparty identity is not disclosed". -> ``get_agent_trades(a)`` holds exactly
  ``a``'s legs, carries no counterparty identifier, and is byte-identical when every
  identifier except ``a``'s is permuted. TRADE SIDE IS NOT CERTIFIED (F16).
* §9.2 "Own orders": "Agent's own active orders (showing full iceberg quantity)" --
  at engine level for tape owners ((6), (6')) and, new in REL-09, on the participant's
  own-orders reader for its own icebergs ((6''), F6, F23): full remaining quantity,
  bit-identical between A and B, while ``top_of_book`` / ``get_anonymous_lob`` show
  the same iceberg only by its visible slice.
* §7.3: identity is kept internally.

Per-draw record layout (``expected`` = world A, ``observed`` = world B, JSON strings)
------------------------------------------------------------------------------------
Unchanged from r4: ``(1)`` engine worlds, ``(3)`` control, ``(4)`` partition, ``(5)``
report twin, ``(6)`` engine ownership, ``(6')`` tape owners' own orders, ``(7)`` twin
validity, ``(2r)`` icebergs / ``detail.rule_implied`` / ``f_max`` /
``first_entry_step`` / ``differs_on`` / ``view_differences``.

What the results JSON MUST carry for the retargeted claim (field names are resolved
through the aliases in ``READER_KEYS`` / ``OWN_KEYS``; the notebook was rewritten
concurrently with this twin, so the exact spellings were not known):

* ``(2)`` ``participant-view twin (noop)`` / ``(random)`` and their ``(eager)``
  copies, and the ``view`` of ``(2r)``: one participant-view episode per world with
  ``n_steps`` (= K), per-step blake2b hex digests ``top_of_book`` (K + 1 entries,
  entry 0 = reset), ``anonymous_lob`` (K + 1), ``fill_report`` (K) and ``done`` (K),
  and the end-of-episode ``own_trades`` (rows ``[price_eur, qty_mw, timestamp]``),
  ``own_orders``, ``agent_cash`` and ``agent_inventory`` in full (top level or under
  ``own``).
* ``(3)`` observed ``{"B": ..., "CTRL": ...}``: each the ``top_of_book`` digest stream
  (or an episode-like dict carrying ``top_of_book`` and ``anonymous_lob``).
* ``(2r)`` ``view_differences`` entries ``{"field", "step", "A", "B"}``; a
  rule-implied difference may only be on a touch / level QUANTITY field (``q_bid``,
  ``q_ask``, a LOB quantity).
* ``(6'') ownership (participant)``: per draw, A's and B's participant own orders.
* Global non-vacuity assertions (seed ``null``): participant's own report non-empty;
  display C3 with ``u > 0``; a participant iceberg resting at the end of a random
  episode; the participant trading as buyer and as seller.
* Market-spec quote of the §9.2 row ``| Best bid / best ask | Top-of-book per
  contract |``.

Results JSON: ``$OMIE_CERT_RESULTS_DIR/C13.json`` else
``notebooks/certification/results/C13.json``; absent -> skip. The slow probes at the
end re-derive the claim through the public API on hand-built tapes and need no JSON.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from notebooks.certification.cert_support import (
    SpecQuote,
    check_quote,
    derive_failure_kinds,
    quote_digest,
    validate_results,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "notebooks" / "certification" / "results"
CERT_ID = "C13"

# --- the specification's constants (§2.3), not the engine's -------------------
PRICE_TICK = 0.01
QTY_TICK = 0.1
O = 999_900                      # price_int of 0.00 EUR/MWh (price_int 0 == -9 999.00)
# The participant's stamped identities: ``omie_session.AGENT_TRADER_ID`` and the base of
# its stamped order ids ``omie_session.AGENT_ORDER_ID_BASE`` (public module constants;
# pinned against the module in ``test_participant_identity_constants``).
PARTICIPANT = 1_000_000
AGENT_ORDER_ID_BASE = 2_000_000_000
Q_AGENT_MAX = 6                  # claim §4: the participant's largest order, in qty ticks
DRAWS = 500
ENVELOPE = {"resting_per_side": 80, "trades": 80, "hops": 160, "concurrent_icebergs": 800}
BATTERY = ("RB", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")
CORPUS = (REPO_ROOT / "docs" / "omiecid" / "corpus"
          / "Mapping_Authoritative_Documents_EU_Continuous_Intraday_Trading_SIDC_XBID_Spain_OMIE"
          / "OMIE_Day-Ahead_and_Intraday_Electricity_Market_Rules_English.md")

SEED_RE = re.compile(r"^seed (\d+) (.*)$")
LEG_RE = re.compile(r"^report_leg \(agent (\d+)\): run raised$")
HEXDIGEST = re.compile(r"^[0-9a-f]{16,128}$")
SPEC_VERSION = "v2.14"

# Participant-view readers and the spellings accepted for them in an episode record.
READER_KEYS = {
    "tob": ("top_of_book", "tob"),
    "lob": ("anonymous_lob", "get_anonymous_lob", "lob"),
    "fill": ("fill_report", "fill_reports", "fill", "fills", "FillReport"),
    "done": ("done", "dones"),
}
# End-of-episode own information: (long top-level names, short names under a container).
# The notebook records the participant's own trades top-level as ``own_report``
# (get_agent_trades on the final book) and its final cash and inventory top-level as
# ``cash`` / ``inventory`` (digests; compared for equality only).
OWN_KEYS = {
    "trades": (("own_trades", "agent_trades_own", "get_agent_trades", "own_report"), ("trades", "own_trades")),
    "orders": (("own_orders", "get_agent_orders"), ("orders", "own_orders")),
    "cash": (("agent_cash", "cash"), ("cash", "agent_cash")),
    "inventory": (("agent_inventory", "inventory"), ("inventory", "agent_inventory")),
}
OWN_CONTAINERS = ("own", "end", "own_end", "end_of_episode")


def _is_view(t: str, policy: str, eager: bool) -> bool:
    """A participant-view twin record: ``(2) ...`` (r4 numbering) or the REL-09 name."""
    if not (t.startswith("(2) ") or t.startswith("participant-view twin")):
        return False
    return policy in t and (("(eager)" in t) == eager)


def _is_participant_own_orders(t: str) -> bool:
    return (t.startswith("(6'') ") or t.startswith("(6″) ") or t.startswith('(6") ')
            or "ownership (participant)" in t)


# family key -> predicate on the name tail (after "seed N "); first match wins
FAMILIES = {
    "own_orders_participant": _is_participant_own_orders,
    "engine": lambda t: t.startswith("(1) "),
    "view_noop": lambda t: _is_view(t, "noop", False),
    "view_random": lambda t: _is_view(t, "random", False),
    "view_noop_eager": lambda t: _is_view(t, "noop", True),
    "view_random_eager": lambda t: _is_view(t, "random", True),
    "surfacing": lambda t: t.startswith("(2r) "),
    "control": lambda t: t.startswith("(3) "),
    "partition": lambda t: t.startswith("(4) "),
    "report_twin": lambda t: t.startswith("(5) "),
    "ownership": lambda t: t.startswith("(6) "),
    "own_orders": lambda t: t.startswith("(6') "),
    "twin_validity": lambda t: t.startswith("(7) "),
}
REQUIRED_PER_DRAW = ("engine", "view_noop", "control", "partition", "report_twin",
                     "ownership", "own_orders", "own_orders_participant", "twin_validity")
# the random-policy episode: exactly one of these two per draw -- the bit-identical
# twin (premise holds) or the rule-implied record (premise failed)
RANDOM_EITHER = ("view_random", "surfacing")


# ---------------------------------------------------------------------------
# Loading and indexing
# ---------------------------------------------------------------------------
def _results_path() -> Path:
    d = os.environ.get("OMIE_CERT_RESULTS_DIR")
    return (Path(d) if d else DEFAULT_RESULTS_DIR) / f"{CERT_ID}.json"


@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    p = _results_path()
    if not p.is_file():
        pytest.skip(f"C13 results JSON absent at {p} (run notebook section 8 first)")
    return json.loads(p.read_text(encoding="utf-8"))


def _load(x):
    return json.loads(x) if isinstance(x, str) else x


def index(payload) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """seed -> family -> [assertion, ...]; report_leg failures under "report_leg"."""
    idx: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for a in payload["assertions"]:
        m = SEED_RE.match(a["name"])
        if not m:
            continue
        seed, tail = int(m.group(1)), m.group(2)
        if LEG_RE.match(tail):
            idx[seed]["report_leg"].append(a)
            continue
        for fam, pred in FAMILIES.items():
            if pred(tail):
                idx[seed][fam].append(a)
                break
        else:
            idx[seed]["unknown"].append(a)
    return idx


@pytest.fixture(scope="module")
def idx(payload):
    return index(payload)


def _one(idx, seed, fam) -> Dict[str, Any]:
    rows = idx[seed][fam]
    assert len(rows) == 1, f"seed {seed}: {fam} present {len(rows)} times (want exactly 1)"
    return rows[0]


def _reload(obj: Dict[str, Any], tmp_path: Path, name: str) -> Dict[str, Any]:
    f = tmp_path / name
    f.write_text(json.dumps(obj), encoding="utf-8")
    return json.loads(f.read_text(encoding="utf-8"))


def _valid_seeds(payload) -> List[int]:
    s = payload["seeds"]
    return list(range(s["first"], s["last"] + 1))


# ---------------------------------------------------------------------------
# The rules, as arithmetic (own implementation)
# ---------------------------------------------------------------------------
def p_int(eur: float) -> int:
    return int(round(eur / PRICE_TICK)) + O


def q_int(mw: float) -> int:
    return int(round(mw / QTY_TICK))


def legs_by_agent(trades) -> Dict[int, Counter]:
    """§7.2: each two-party trade is one leg for the buyer and one for the seller."""
    legs: Dict[int, Counter] = defaultdict(Counter)
    for _bo, _so, ba, sa, p, q, _s in trades:
        legs[int(ba)][(int(p), int(q))] += 1
        legs[int(sa)][(int(p), int(q))] += 1
    return legs


def trade_agents(trades) -> set:
    return {int(t[2]) for t in trades} | {int(t[3]) for t in trades}


# ---------------------------------------------------------------------------
# Judges: each returns a list of problems (empty == the rule holds on the JSON)
# ---------------------------------------------------------------------------
def judge_completeness(payload, idx, seeds=None) -> List[str]:
    """Per-draw values present in full: every family once per valid draw, with values."""
    out = []
    for s in seeds if seeds is not None else _valid_seeds(payload):
        if idx[s].get("unknown"):
            out.append(f"seed {s}: unrecognised per-draw assertion(s)")
        n_rand = [len(idx[s].get(f, [])) for f in RANDOM_EITHER]
        if sorted(n_rand) != [0, 1]:
            out.append(f"seed {s}: random-policy record {dict(zip(RANDOM_EITHER, n_rand))} "
                       f"(want exactly one of the two)")
        for fam in REQUIRED_PER_DRAW + RANDOM_EITHER:
            rows = idx[s].get(fam, [])
            if fam in RANDOM_EITHER and not rows:
                continue
            if len(rows) != 1:
                out.append(f"seed {s}: {fam} present {len(rows)} times")
                continue
            a = rows[0]
            if a["seed"] != s:
                out.append(f"seed {s}: {fam} carries seed {a['seed']}")
            for k in ("expected", "observed"):
                v = a[k]
                if v in (None, "", {}, []):
                    out.append(f"seed {s}: {fam} {k} empty")
    return out


# ---------------------------------------------------------------------------
# The participant-view episode: readers, own information, structure
# ---------------------------------------------------------------------------
def reader_key(e: Dict[str, Any], reader: str) -> str:
    for k in READER_KEYS[reader]:
        if k in e:
            return k
    raise KeyError(f"participant-view episode carries no {reader} stream "
                   f"(results JSON must record one of {READER_KEYS[reader]})")


def stream(e: Dict[str, Any], reader: str) -> List[Any]:
    return e[reader_key(e, reader)]


def own_slot(e: Dict[str, Any], what: str) -> Tuple[Dict[str, Any], str]:
    """(container, key) of an end-of-episode own field; KeyError if not recorded."""
    long_names, short_names = OWN_KEYS[what]
    for c in OWN_CONTAINERS:
        if isinstance(e.get(c), dict):
            for k in short_names:
                if k in e[c]:
                    return e[c], k
    for k in long_names:
        if k in e:
            return e, k
    raise KeyError(f"participant-view episode carries no end-of-episode own {what} "
                   f"(results JSON must record one of {long_names} or {short_names} "
                   f"under {OWN_CONTAINERS})")


def own(e: Dict[str, Any], what: str) -> Any:
    c, k = own_slot(e, what)
    return c[k]


def own_trades(e: Dict[str, Any]) -> List[List[Any]]:
    return own(e, "trades") or []


def _check_episode(e, where) -> List[str]:
    out = []
    try:
        K = e["n_steps"]
        if not (isinstance(K, int) and K >= 1):
            out.append(f"{where}: n_steps {K!r} is not a positive integer")
            return out
        for reader, n in (("tob", K + 1), ("lob", K + 1), ("fill", K), ("done", K)):
            s = stream(e, reader)
            if len(s) != n:
                out.append(f"{where}: {reader} stream has {len(s)} entries, want {n}")
            for h in s:
                if not (isinstance(h, str) and HEXDIGEST.match(h)):
                    out.append(f"{where}: {reader} entry {h!r} is not a hex digest")
                    break
        for r in own_trades(e):
            if not (isinstance(r, list) and len(r) == 3):
                out.append(f"{where}: own trade row {r!r} is not (price, qty, timestamp)")
        for what in ("orders", "cash", "inventory"):
            if own(e, what) is None:
                out.append(f"{where}: own {what} recorded as null")
    except (KeyError, TypeError) as exc:
        out.append(f"{where}: malformed participant-view episode ({exc!r})")
    return out


def _is_participant_oid(oid) -> bool:
    try:
        return int(oid) >= AGENT_ORDER_ID_BASE
    except (TypeError, ValueError):
        return False


def hidden_deficit(idx, seed) -> Dict[str, int]:
    """Per FOREIGN iceberg: engine-A hidden remainder minus the A/random end-of-episode
    one (qty ticks). Positive == the random policy's episode drew on hidden quantity.
    The participant's own icebergs (stamped ids) are not foreign and are ignored."""
    tv = _load(_one(idx, seed, "twin_validity")["observed"])
    hn = {k: int(v) for k, v in tv.get("A/noop", {}).get("hidden", {}).items()
          if not _is_participant_oid(k)}
    hr = {k: int(v) for k, v in tv.get("A/random", {}).get("hidden", {}).items()}
    return {k: hn[k] - hr.get(k, 0) for k in hn if hn[k] != hr.get(k, 0)}


def _episode_of_random(idx, seed) -> Dict[str, Any]:
    """World A's random-policy episode: the (2) episode or the (2r) record's view."""
    rows = idx[seed].get("view_random")
    if rows:
        assert len(rows) == 1, f"seed {seed}: {len(rows)} random-policy records"
        return _load(rows[0]["expected"])
    rows = idx[seed].get("surfacing")
    assert rows and len(rows) == 1, f"seed {seed}: no single random-policy record"
    return _load(rows[0]["expected"])["view"]


def claimed_premise_failed_seeds(payload, idx) -> List[int]:
    """The draws the notebook judged by the rule-implied outcome (a (2r) record)."""
    return sorted(s for s in _valid_seeds(payload) if idx[s].get("surfacing"))


def hidden_signature_seeds(payload, idx) -> List[int]:
    """NECESSARY condition of a premise failure (r3 signature, kept): a draw whose
    random-policy world A drew on a foreign hidden remainder. Under the amended item
    (4) an entry sweep of the agent's order that stays inside instance k_e leaves the
    hidden quantity untouched, so this set may be (and in r4 is) a strict subset of
    the premise-failed draws; nothing but an entry sweep of the agent's order may
    produce it."""
    return [s for s in _valid_seeds(payload) if hidden_deficit(idx, s)]


def premise_evidence_problems(idx, seed) -> List[str]:
    """OWN evidence that a (2r) record's draw really failed the premise: every iceberg
    it names is a FOREIGN iceberg of this very draw (6), the participant traded in the
    random episode, and its own trades cover the claimed entry fills."""
    out: List[str] = []
    rows = idx[seed].get("surfacing", [])
    if len(rows) != 1:
        return [f"rule-implied record present {len(rows)} times"]
    A = _load(rows[0]["expected"])
    own_ice = _load(_one(idx, seed, "ownership")["observed"])
    names = set(A.get("icebergs", {}))
    if not names:
        out.append("names no iceberg")
    for oid in sorted(names - set(own_ice)):
        out.append(f"iceberg {oid} is not a resting foreign iceberg of this draw")
    for oid in sorted(names & set(own_ice)):
        if int(own_ice[oid]["owner"]) == PARTICIPANT:
            out.append(f"iceberg {oid} is the participant's own")
    try:
        rep = own_trades(A.get("view") or {})
    except KeyError:
        rep = []
    if not rep:
        out.append("the participant did not trade in the random episode")
    f_sum = sum(int(r.get("B_entry_fill", 0)) for r in A.get("icebergs", {}).values())
    if sum(q_int(r[1]) for r in rep) < f_sum:
        out.append(f"own trades quantity < claimed entry fills {f_sum}")
    return out


def iceberg_geometry(idx, seed, oid) -> Dict[str, int]:
    """OWN derivation for one iceberg, in qty ticks, from the engine record (1) (the
    world without the participant) and the twin-validity record (7).

    T = what the engine world still holds visible + hidden + what it filled (an
    iceberg's quantity is only ever traded, rested or hidden: no cancel/evict here,
    checked); h = the noop world's hidden remainder; P = T - h (no slice regenerated
    in the noop world, cross-checked below against item (4)); q_B = the noop world's
    fills + its display = the twin's plain-order size; visible = its noop display."""
    eng = _load(_one(idx, seed, "engine")["observed"])["A"]
    tv = _load(_one(idx, seed, "twin_validity")["observed"])
    if oid in eng.get("cancelled", {}) or oid in eng.get("evicted", {}):
        raise ValueError(f"iceberg {oid} cancelled/evicted in engine A")
    vis = int(eng["still_resting"][oid])
    filled = int(eng["filled"].get(oid, 0))
    T = vis + int(eng["hidden_remaining"][oid]) + filled
    h = int(tv["A/noop"]["hidden"][oid])
    return {"T": T, "h": h, "P": T - h, "q_B": filled + vis, "filled": filled, "visible": vis}


def item4_display(T: int, P: int, M: int) -> Dict[str, int]:
    """§4.2.1 Rule 2 *Incoming iceberg* item (4), amended 2026-10-20, as arithmetic:
    R = T - M, k_e = floor(M/P), u = M - k_e P, v1 = min(P - u, R), hidden R - v1."""
    R = T - M
    k_e = M // P
    u = M - k_e * P
    v1 = min(P - u, R)
    return {"R": R, "k_e": k_e, "u": u, "v1": v1, "hidden": R - v1}


def rule_implied(idx, seed, oid, M: int, f: int) -> Dict[str, int]:
    """Own closed form for one premise-failed iceberg: world A (item (4), M including
    the agent's fill f), the twin B (plain order q_B sweeping the same M), the gap, and
    the RETIRED display min(P, R) for discrimination."""
    g = iceberg_geometry(idx, seed, oid)
    T, P, q_B = g["T"], g["P"], g["q_B"]
    a = item4_display(T, P, M)
    noop = item4_display(T, P, M - f)          # the same iceberg's entry without the agent
    rests_B = q_B - M
    retired = min(P, a["R"])
    return dict(g, M=M, f=f, R=a["R"], k_e=a["k_e"], u=a["u"], shown_A=a["v1"],
                hidden_A=a["hidden"], rests_B=rests_B, visible_gap=a["v1"] - rests_B,
                noop_v1=noop["v1"], noop_hidden=noop["hidden"],
                retired_shown_A=retired, retired_gap=retired - rests_B)

QTY_FIELD_TOKENS = ("q_bid", "q_ask")


def _is_qty_field(field: str) -> bool:
    """A rule-implied surfacing can only move a displayed QUANTITY: the touch volume of
    ``top_of_book`` or a level quantity of ``get_anonymous_lob``."""
    f = str(field).lower()
    return any(t in f for t in QTY_FIELD_TOKENS) or ("lob" in f and ("qty" in f or "quantity" in f))


def rule_implied_problems(payload, idx, seed) -> List[str]:
    """Own derivation of §4.2.1 Rule 2 *Incoming iceberg* (1)+(4) (amended 2026-10-20)
    for one premise-failed draw. Empty == the ``(2r)`` record and the participant-view
    episode it carries are exactly what the rule implies, and nothing else differs.

    Conditions (each a consequence of the rule or the claim, none of the notebook):
      * own evidence that the premise failed on this draw (``premise_evidence_problems``);
      * the noop episode is bit-identical (the participant is in no sweep there);
      * f_max = Q_AGENT_MAX x K (claim (2r));
      * the iceberg's entry filled the participant (A and B), f in 1..f_max, M >= f and
        the foreign part M - f no more than the noop world's fills of that iceberg;
      * with T, h, P, q_B derived here: k_e, u, R, v1 = min(P - u, R), hidden R - v1,
        B rests q_B - M, gap v1 - (q_B - M); the noop display of item (4) (M - f swept)
        reproduces the engine's q_B and h (no regeneration, twin construction sound);
      * A's hidden equals the A/random hidden in (7), the deficit h - hidden_A;
      * the retired display recorded is min(P, R) (the record does not misstate it);
      * done, own trades / orders / cash / inventory equal between A and B always;
        ``top_of_book`` / ``get_anonymous_lob`` equal before the entry step and the
        ``FillReport`` before the entry step's report;
      * gap 0 => the whole participant view is bit-identical and no difference is
        reported; gap != 0 => only touch / level QUANTITY fields from the entry step
        on, A above B by at most the gap.
    """
    out: List[str] = list(premise_evidence_problems(idx, seed))
    noop = _one(idx, seed, "view_noop")
    if _load(noop["expected"]) != _load(noop["observed"]):
        out.append("noop episode differs too (no participant order to explain it)")
    rows = idx[seed].get("surfacing", [])
    if len(rows) != 1:
        return out + [f"rule-implied record present {len(rows)} times"]
    a = rows[0]
    A, B = _load(a["expected"]), _load(a["observed"])
    try:
        det = _load(a["detail"])
        ri, f_max = det["rule_implied"], int(det["f_max"])
        diffs = det["view_differences"]
        k_star = int(det["first_entry_step"])
        vA, vB = A["view"], B["view"]
        K = int(vA["n_steps"])
    except (TypeError, ValueError, KeyError) as exc:
        return out + [f"rule-implied record carries no closed form ({exc!r})"]
    out += _check_episode(vA, "(2r) view A") + _check_episode(vB, "(2r) view B")
    if A.get("icebergs") != B.get("icebergs"):
        out.append("per-iceberg entry records differ between A and B")
    if f_max != Q_AGENT_MAX * K:
        out.append(f"f_max = {f_max}, claim gives Q_AGENT_MAX x K = {Q_AGENT_MAX} x {K}")
    if not 1 <= k_star <= K:
        out.append(f"entry step {k_star} outside the episode (K = {K})")
    try:
        noop_A = _load(noop["expected"])
        for reader in ("tob", "lob"):
            if stream(vA, reader)[:1] != stream(noop_A, reader)[:1]:
                out.append(f"reset {reader} differs from the noop episode's")
        if vA.get("n_steps") != vB.get("n_steps"):
            out.append("n_steps differs between A and B")
        if stream(vA, "done") != stream(vB, "done"):
            out.append("done differs between A and B")
        for what in ("trades", "orders", "cash", "inventory"):
            if own(vA, what) != own(vB, what):
                out.append(f"own {what} differs between A and B")
        for reader in ("tob", "lob"):
            if stream(vA, reader)[:k_star] != stream(vB, reader)[:k_star]:
                out.append(f"{reader} before the entry step differs between A and B")
        if stream(vA, "fill")[:k_star - 1] != stream(vB, "fill")[:k_star - 1]:
            out.append("fill report before the entry step differs between A and B")
    except (KeyError, TypeError) as exc:
        return out + [f"participant view not comparable ({exc!r})"]
    differs = sorted(k for k in set(vA) | set(vB) if vA.get(k) != vB.get(k))
    if sorted(det.get("differs_on", [])) != differs:
        out.append(f"differs_on {det.get('differs_on')} != what differs {differs}")
    tv = _load(_one(idx, seed, "twin_validity")["observed"])
    own_ice = _load(_one(idx, seed, "ownership")["observed"])
    deficit = hidden_deficit(idx, seed)
    if not set(deficit) <= set(ri) or set(A.get("icebergs", {})) != set(ri):
        out.append(f"icebergs drawing hidden {sorted(deficit)} / record "
                   f"{sorted(A.get('icebergs', {}))} not covered by rule-implied {sorted(ri)}")
    total_gap = 0
    for oid, r in ri.items():
        rec = A.get("icebergs", {}).get(oid, {})
        try:
            f, M = int(rec["B_entry_fill"]), int(r["M"])
            w = rule_implied(idx, seed, oid, M, f)
        except (KeyError, ValueError, TypeError) as exc:
            out.append(f"iceberg {oid}: geometry not derivable ({exc!r})")
            continue
        for k in ("R", "k_e", "u", "shown_A", "hidden_A", "rests_B", "visible_gap"):
            if int(r.get(k, -10**9)) != w[k]:
                out.append(f"iceberg {oid}: {k} = {r.get(k)}, rule implies {w[k]} "
                           f"(T={w['T']}, P={w['P']}, M={M}, f={f})")
        if r.get("retired_C1") != {"shown_A": w["retired_shown_A"], "visible_gap": w["retired_gap"]}:
            out.append(f"iceberg {oid}: retired display recorded {r.get('retired_C1')}, "
                       f"min(P, R) gives {w['retired_shown_A']} / gap {w['retired_gap']}")
        if r.get("hidden_unchanged") is not (w["hidden_A"] == w["h"]):
            out.append(f"iceberg {oid}: hidden_unchanged {r.get('hidden_unchanged')} misstated")
        if int(r.get("rest_price_A", -1)) != int(own_ice[oid]["price_int"]) or \
                rec.get("A_resting_rows_price") != [int(own_ice[oid]["price_int"])]:
            out.append(f"iceberg {oid}: rest price {r.get('rest_price_A')} / "
                       f"{rec.get('A_resting_rows_price')} != b = {own_ice[oid]['price_int']}")
        if not (1 <= f <= f_max and f <= M and M - f <= w["filled"]):
            out.append(f"iceberg {oid}: f = {f}, M = {M} outside 1 <= f <= {f_max}, "
                       f"f <= M, M - f <= noop fills {w['filled']}")
        # twin construction + no regeneration: the noop entry of item (4) gives q_B, h
        if w["q_B"] - (M - f) != w["noop_v1"] or w["noop_hidden"] != w["h"]:
            out.append(f"iceberg {oid}: noop display {w['noop_v1']} / hidden {w['noop_hidden']} "
                       f"(item (4), M - f = {M - f}) != engine q_B - (M - f) = "
                       f"{w['q_B'] - (M - f)} / h = {w['h']}")
        if int(tv["A/random"]["hidden"].get(oid, -1)) != w["hidden_A"]:
            out.append(f"iceberg {oid}: A/random hidden {tv['A/random']['hidden'].get(oid)} "
                       f"!= R - v1 = {w['hidden_A']}")
        if deficit.get(oid, 0) != w["h"] - w["hidden_A"]:
            out.append(f"iceberg {oid}: hidden deficit {deficit.get(oid, 0)} != {w['h'] - w['hidden_A']}")
        if (rec.get("A_entry_passive_agents") != [PARTICIPANT]
                or rec.get("B_entry_passive_agents") != [PARTICIPANT]):
            out.append(f"iceberg {oid}: entry did not fill the participant "
                       f"(A {rec.get('A_entry_passive_agents')}, B {rec.get('B_entry_passive_agents')})")
        if rec.get("A_entry_fill_in_1_to_f_max_below_peak") is not (1 <= f <= f_max and f < w["P"]):
            out.append(f"iceberg {oid}: entry fill flag {rec.get('A_entry_fill_in_1_to_f_max_below_peak')} "
                       f"misstated (f = {f}, P = {w['P']})")
        if rec.get("hidden_A") != w["hidden_A"] or rec.get("visible_gap_A_minus_B") != w["visible_gap"]:
            out.append(f"iceberg {oid}: per-iceberg summary disagrees with the closed form")
        total_gap += w["visible_gap"]
    if total_gap == 0:
        if vA != vB:
            out.append(f"participant view differs between A and B on {differs}, "
                       f"rule implies a zero gap (LEAK)")
    for d in diffs:
        try:
            st, field = int(d["step"]), str(d["field"])
        except (KeyError, TypeError, ValueError):
            out.append(f"malformed difference {d}")
            continue
        if not _is_qty_field(field):
            out.append(f"difference on non-quantity field {d}")
            continue
        try:
            up = q_int(float(d["A"])) - q_int(float(d["B"]))
        except (KeyError, TypeError, ValueError):
            out.append(f"malformed difference {d}")
            continue
        if st < k_star:
            out.append(f"difference at step {st} before the entry step {k_star}")
        if not 0 < up <= total_gap:
            out.append(f"A-B on {field} = {up} ticks, rule implies 1..{total_gap} (LEAK)")
    if diffs and vA == vB:
        out.append("differences reported but the participant views are identical")
    if not diffs and vA != vB:
        out.append("participant views differ but no difference is reported")
    return out


def judge_view_twin(payload, idx, seeds=None) -> List[str]:
    """§7.1/§9.2: the participant view (top_of_book, get_anonymous_lob, FillReport,
    done, own trades / orders / cash / inventory) bit-identical across A and B on every
    draw whose premise holds; on a premise-failed draw, exactly the rule-implied outcome
    (``rule_implied_problems``). A rule-implied record without own evidence of a sweep
    of the participant's order is a blanket excuse, and fails."""
    out = []
    sig = set(hidden_signature_seeds(payload, idx))
    for s in seeds if seeds is not None else _valid_seeds(payload):
        rows = idx[s].get("view_noop", [])
        if len(rows) != 1:
            out.append(f"seed {s}: view_noop missing")
        else:
            A, B = _load(rows[0]["expected"]), _load(rows[0]["observed"])
            out += _check_episode(A, f"seed {s} view_noop A") + _check_episode(B, f"seed {s} view_noop B")
            if A != B:
                out.append(f"seed {s}: view_noop: participant view differs between A and B (LEAK)")
            elif rows[0]["status"] != "pass":
                out.append(f"seed {s}: view_noop identical but recorded {rows[0]['status']}")
        rnd, sur = idx[s].get("view_random", []), idx[s].get("surfacing", [])
        if len(rnd) + len(sur) != 1:
            out.append(f"seed {s}: {len(rnd)} twin and {len(sur)} rule-implied random records")
            continue
        if sur:
            ev = premise_evidence_problems(idx, s)
            if ev:
                out.append(f"seed {s}: rule-implied record without a sweep of the participant's "
                           f"order: premise holds {ev} (LEAK excused)")
            why = rule_implied_problems(payload, idx, s)
            if why:
                out.append(f"seed {s}: rule-implied record is not what the rule implies (LEAK): {why}")
            if sur[0]["status"] != "pass" and not why:
                out.append(f"seed {s}: rule holds but recorded {sur[0]['status']}")
            continue
        a = rnd[0]
        A, B = _load(a["expected"]), _load(a["observed"])
        out += _check_episode(A, f"seed {s} view_random A") + _check_episode(B, f"seed {s} view_random B")
        if s in sig:
            out.append(f"seed {s}: premise failed (hidden drawn {hidden_deficit(idx, s)}) but "
                       f"judged as a plain twin")
        if A != B:
            out.append(f"seed {s}: view_random: participant view differs between A and B (LEAK)")
        elif a["status"] != "pass":
            out.append(f"seed {s}: view_random identical but recorded {a['status']}")
    return out


def _ctrl_stream(x, reader) -> Optional[List[Any]]:
    """A control arm: a bare top_of_book digest stream, or an episode-like dict."""
    if isinstance(x, dict):
        try:
            return stream(x, reader)
        except KeyError:
            return None
    return x if reader == "tob" else None


def judge_positive_control(payload, idx, seeds=None) -> List[str]:
    """Rule 1 / claim item 3: the twin check is seen failing -- one visible tick changes
    the noop ``top_of_book`` on at least one step, and ``get_anonymous_lob`` on that
    step as well (when the control records the LOB stream)."""
    out = []
    for s in seeds if seeds is not None else _valid_seeds(payload):
        rows = idx[s].get("control", [])
        if len(rows) != 1:
            out.append(f"seed {s}: positive control missing")
            continue
        noop = idx[s].get("view_noop", [])
        if not noop:
            out.append(f"seed {s}: no noop episode to anchor the control")
            continue
        try:
            c = _load(rows[0]["observed"])
            noop_B = _load(noop[0]["observed"])
            b_tob, c_tob = _ctrl_stream(c.get("B"), "tob"), _ctrl_stream(c.get("CTRL"), "tob")
            if b_tob != stream(noop_B, "tob"):
                out.append(f"seed {s}: control's B top_of_book stream is not the twin's B stream")
            if not c_tob or len(c_tob) != len(b_tob or []) or c_tob == b_tob:
                out.append(f"seed {s}: CTRL top_of_book does not differ from B (control blind)")
            b_lob, c_lob = _ctrl_stream(c.get("B"), "lob"), _ctrl_stream(c.get("CTRL"), "lob")
            if b_lob is not None or c_lob is not None:
                if b_lob != stream(noop_B, "lob"):
                    out.append(f"seed {s}: control's B anonymous_lob stream is not the twin's B stream")
                both = [k for k in range(min(len(b_tob or []), len(c_tob or []), len(b_lob or []),
                                             len(c_lob or [])))
                        if b_tob[k] != c_tob[k] and b_lob[k] != c_lob[k]]
                if not both:
                    out.append(f"seed {s}: anonymous_lob does not see the tick on a step "
                               f"top_of_book does (control blind)")
        except (AttributeError, KeyError, TypeError) as exc:
            out.append(f"seed {s}: control record malformed ({exc!r})")
            continue
        if rows[0]["status"] != "pass":
            out.append(f"seed {s}: control recorded {rows[0]['status']}")
    return out


def judge_twin_validity(payload, idx, seeds=None) -> List[str]:
    """The twin is a twin: foreign hidden remainders in A, none in B/CTRL; ids permuted.
    The participant's own icebergs (stamped ids) are excluded from the foreign tally
    (claim item 7, F4): they may regenerate, but identically in A and B."""
    out = []
    for s in seeds if seeds is not None else _valid_seeds(payload):
        eng = _load(_one(idx, s, "engine")["observed"])
        own_ice = _load(_one(idx, s, "ownership")["observed"])
        tv = _load(_one(idx, s, "twin_validity")["observed"])
        A, B = eng["A"], eng["B"]
        hidA = {k: int(v) for k, v in A["hidden_remaining"].items()}
        if not hidA or min(hidA.values()) <= 0:
            out.append(f"seed {s}: world A holds no positive hidden remainder (vacuous twin)")
        if B["hidden_remaining"]:
            out.append(f"seed {s}: world B still holds a hidden remainder")
        part = {}
        for k, v in tv.items():
            if v.get("truncated") or v.get("book_full"):
                out.append(f"seed {s}: capacity flag in {k}")
            want = hidA if k.startswith("A/") else {}
            all_h = {kk: int(vv) for kk, vv in v.get("hidden", {}).items()}
            got = {kk: vv for kk, vv in all_h.items() if not _is_participant_oid(kk)}
            part[k] = {kk: vv for kk, vv in all_h.items() if _is_participant_oid(kk)}
            if got != want:
                # Rule 2 *Incoming iceberg* (1)+(4): an iceberg whose entry swept the
                # participant's order -- legitimate on A/random only, iceberg by iceberg
                # as the closed form says (checked there)
                if k == "A/random" and idx[s].get("surfacing") \
                        and not rule_implied_problems(payload, idx, s):
                    continue
                out.append(f"seed {s}: {k} hidden {v.get('hidden')} != engine A {want}")
        for need in ("A/noop", "A/random", "B/noop", "B/random", "CTRL/noop"):
            if need not in tv:
                out.append(f"seed {s}: twin-validity lacks {need}")
        for k, p in part.items():
            if "/noop" in k and p:
                out.append(f"seed {s}: {k} holds a participant iceberg under the noop policy")
        if part.get("A/random", {}) != part.get("B/random", {}):
            out.append(f"seed {s}: participant icebergs differ between A/random and B/random")
        # engine-world icebergs are FOREIGN to the participant (no participant there)
        for oid, o in own_ice.items():
            if int(o["owner"]) == PARTICIPANT:
                out.append(f"seed {s}: iceberg {oid} owned by the participant (not foreign)")
        if set(own_ice) != set(hidA):
            out.append(f"seed {s}: ownership icebergs {sorted(own_ice)} != A hidden {sorted(hidA)}")
        # B is A with identifiers permuted: a consistent bijection on oids and agents
        if eng["B_is_A_relabelled"] != B["trades"] or len(A["trades"]) != len(B["trades"]):
            out.append(f"seed {s}: B trades are not A's relabelled")
            continue
        omap, amap = {}, {}
        for ta, tb in zip(A["trades"], B["trades"]):
            if ta[4:] != tb[4:]:
                out.append(f"seed {s}: B trade {tb} != A trade {ta} on price/qty/submission")
            for i, m in ((0, omap), (1, omap), (2, amap), (3, amap)):
                if m.setdefault(ta[i], tb[i]) != tb[i]:
                    out.append(f"seed {s}: identifier map not a function at {ta[i]}")
        for m, what in ((omap, "oid"), (amap, "agent")):
            if len(set(m.values())) != len(m):
                out.append(f"seed {s}: {what} map not injective")
        if sorted(A["still_resting"].values()) != sorted(B["still_resting"].values()):
            out.append(f"seed {s}: visible resting quantities differ between A and B")
        if sorted(A["filled"].values()) != sorted(B["filled"].values()):
            out.append(f"seed {s}: fill quantities differ between A and B")
    return out


def judge_participant_own_orders(payload, idx, seeds=None) -> List[str]:
    """(6'') §9.2 "Own orders" on the participant's own-orders reader (F6, F23): one
    record per draw, A's and B's ``get_agent_orders(book, AGENT_TRADER_ID)``
    bit-identical (the foreign twin cannot move the participant's own orders)."""
    out = []
    for s in seeds if seeds is not None else _valid_seeds(payload):
        rows = idx[s].get("own_orders_participant", [])
        if len(rows) != 1:
            out.append(f"seed {s}: ownership (participant) present {len(rows)} times")
            continue
        e, o = _load(rows[0]["expected"]), _load(rows[0]["observed"])
        if e != o:
            out.append(f"seed {s}: participant's own orders differ between A and B (LEAK)")
        elif rows[0]["status"] != "pass":
            out.append(f"seed {s}: ownership (participant) identical but recorded {rows[0]['status']}")
    return out


def global_assertions(payload, *needles_any: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Seed-less assertions whose lower-cased name contains, for every tuple given, at
    least one of its words."""
    out = []
    for a in payload["assertions"]:
        if a["seed"] is not None:
            continue
        n = a["name"].lower()
        if all(any(w in n for w in group) for group in needles_any):
            out.append(a)
    return out


def judge_trade_reports(payload, idx, seeds=None) -> List[str]:
    """§7.2/§9.2: exact per-agent partition, union = every trade once per side, no ids."""
    out = []
    for s in seeds if seeds is not None else _valid_seeds(payload):
        eng = _load(_one(idx, s, "engine")["observed"])
        tw = _one(idx, s, "report_twin")
        rep_a, rep_b = _load(tw["expected"]), _load(tw["observed"])
        if rep_a != rep_b:
            out.append(f"seed {s}: an observer's report differs under permuted foreign ids (LEAK)")
        trades = eng["A"]["trades"]
        legs = legs_by_agent(trades)
        observers = {int(k) for k in rep_a}
        if not trade_agents(trades) <= observers:
            out.append(f"seed {s}: trading agents {sorted(trade_agents(trades) - observers)} have no report")
        union = Counter()
        for ag, rows in rep_a.items():
            got = Counter()
            for r in rows:
                if not isinstance(r, list) or len(r) != 3:
                    out.append(f"seed {s}: agent {ag} row {r!r} is not (price, qty, timestamp)")
                    continue
                if float(r[2]) != int(r[2]):
                    out.append(f"seed {s}: agent {ag} row timestamp {r[2]!r} not integral")
                got[(p_int(r[0]), q_int(r[1]))] += 1
                union[(p_int(r[0]), q_int(r[1]), int(r[2]))] += 1
            if got != legs.get(int(ag), Counter()):
                out.append(f"seed {s}: agent {ag} report {dict(got)} != own legs "
                           f"{dict(legs.get(int(ag), Counter()))}")
        # the union over agents: each two-party trade once per side, same (p, q, timestamp)
        if sum(union.values()) != 2 * len(trades):
            out.append(f"seed {s}: union has {sum(union.values())} rows, want {2 * len(trades)}")
        for key, n in union.items():
            if n % 2:
                out.append(f"seed {s}: leg {key} appears {n} times (unpaired / duplicated)")
        if tw["status"] != "pass":
            out.append(f"seed {s}: report twin recorded {tw['status']}")
    return out


def judge_partition_record(payload, idx, seeds=None) -> List[str]:
    """(4) as emitted: each agent's legs [price_int, qty_int] are exactly its own legs
    (no side, J4), and the union is every trade once per side."""
    out = []
    for s in seeds if seeds is not None else _valid_seeds(payload):
        if idx[s].get("report_leg"):
            out.append(f"seed {s}: report_leg failure(s) present")
        trades = _load(_one(idx, s, "engine")["observed"])["A"]["trades"]
        legs = legs_by_agent(trades)
        a = _one(idx, s, "partition")
        e, o = _load(a["expected"]), _load(a["observed"])
        if e != o or a["status"] != "pass":
            out.append(f"seed {s}: partition record differs / not passing")
        if o["n_report_rows"] != 2 * len(trades):
            out.append(f"seed {s}: n_report_rows {o['n_report_rows']} != {2 * len(trades)}")
        for ag, rows in o["reports"].items():
            if rows is None or any(len(r) != 2 for r in rows):
                out.append(f"seed {s}: agent {ag} legs not [price_int, qty_int]")
                continue
            got = Counter((int(p), int(q)) for p, q in rows)
            if got != legs.get(int(ag), Counter()):
                out.append(f"seed {s}: agent {ag} legs {dict(got)} != own {dict(legs.get(int(ag), {}))}")
        if not trade_agents(trades) <= {int(k) for k in o["reports"]}:
            out.append(f"seed {s}: a trading agent has no (4) report")
        want_union = Counter()
        for c in legs.values():
            want_union += c
        if Counter((int(p), int(q)) for p, q in o["union"]) != want_union:
            out.append(f"seed {s}: union != every trade once per side")
    return out


def judge_ownership(payload, idx, seeds=None) -> List[str]:
    """§7.3: each iceberg held under its owner with visible + hidden = full remaining."""
    out = []
    for s in seeds if seeds is not None else _valid_seeds(payload):
        eng = _load(_one(idx, s, "engine")["observed"])["A"]
        a = _one(idx, s, "ownership")
        e, o = _load(a["expected"]), _load(a["observed"])
        if e != o or a["status"] != "pass":
            out.append(f"seed {s}: engine ownership record differs / not passing")
        for oid, r in o.items():
            if int(r["visible"]) + int(r["hidden"]) != int(r["full_remaining"]):
                out.append(f"seed {s}: iceberg {oid} visible+hidden != full_remaining")
            if int(eng["still_resting"].get(oid, -1)) != int(r["visible"]):
                out.append(f"seed {s}: iceberg {oid} visible != engine still_resting")
            if int(eng["hidden_remaining"].get(oid, -1)) != int(r["hidden"]):
                out.append(f"seed {s}: iceberg {oid} hidden != engine hidden_remaining")
    return out


def own_orders_verdict(payload, idx, seed) -> Dict[str, Any]:
    """Own derivation of §9.2 "Own orders" for one draw, from (1) + (6) only.

    Returns {"leak": [...], "under_disclosed": [...], "other": [...]} where
    ``leak`` = an order shown to a non-owner or a foreign order in a view,
    ``under_disclosed`` = an owner shown less than its full iceberg quantity,
    ``other`` = any other mismatch.
    """
    eng = _load(_one(idx, seed, "engine")["observed"])["A"]
    ice = _load(_one(idx, seed, "ownership")["observed"])
    view = _load(_one(idx, seed, "own_orders")["observed"])
    res = {"leak": [], "under_disclosed": [], "other": []}
    seen: Dict[str, str] = {}
    for ag, orders in view.items():
        for oid, q in orders.items():
            if oid in seen:
                res["leak"].append(f"order {oid} shown to {seen[oid]} and {ag}")
            seen[oid] = ag
            if oid in ice and int(ice[oid]["owner"]) != int(ag):
                res["leak"].append(f"iceberg {oid} of {ice[oid]['owner']} shown to {ag}")
            if oid not in eng["still_resting"]:
                res["other"].append(f"{ag}: {oid} is not resting")
                continue
            if oid in ice:
                full = int(ice[oid]["visible"]) + int(ice[oid]["hidden"])
                if int(q) == full:
                    continue
                if int(q) == int(ice[oid]["visible"]):
                    res["under_disclosed"].append(f"{ag}: iceberg {oid} shown {q}, full {full}")
                else:
                    res["other"].append(f"{ag}: iceberg {oid} shown {q}, full {full}")
            elif int(q) != int(eng["still_resting"][oid]):
                res["other"].append(f"{ag}: order {oid} shown {q} != {eng['still_resting'][oid]}")
    for oid in eng["still_resting"]:
        if oid not in seen:
            res["other"].append(f"resting order {oid} in nobody's view")
    for oid, r in ice.items():
        if oid in view.get(str(r["owner"]), {}) or oid in view.get(r["owner"], {}):
            continue
        res["other"].append(f"iceberg {oid} missing from its owner's view")
    return res


# ---------------------------------------------------------------------------
# 1. The payload: schema, semantic rules, counts
# ---------------------------------------------------------------------------
class TestPayload:
    def test_schema_and_semantics_valid(self, payload):
        assert validate_results(payload) == []

    def test_identity(self, payload):
        assert payload["cert_id"] == CERT_ID
        assert payload["notebook"] == "C13_anonymity.ipynb"
        assert payload["oracle_kind"].startswith("exact-parametric")
        assert "metamorphic twin" in payload["oracle_kind"]

    def test_failure_kinds_derived_from_content(self, payload):
        assert payload["failure_kinds"] == derive_failure_kinds(payload)

    def test_draws_full_count_no_discard(self, payload):
        d = payload["draws"]
        assert d["requested"] == DRAWS
        assert d["valid"] + d["discarded"] >= d["requested"]
        assert d["valid"] == DRAWS, "random scenarios default to 500 valid draws"
        s = payload["seeds"]
        assert s["count"] == d["valid"] == s["last"] - s["first"] + 1

    def test_failing_seed_list_matches_per_seed_failures(self, payload, idx):
        want = sorted(s for s in _valid_seeds(payload)
                      if any(a["status"] == "fail" for fam in idx[s].values() for a in fam))
        assert sorted(payload["seeds"]["failing"]) == want

    def test_capacity_inside_envelope(self, payload):
        assert payload["capacity"]["envelope"] == ENVELOPE
        for k, cap in ENVELOPE.items():
            v = payload["capacity"]["peaks"][k]
            assert v is None or 0 <= v <= cap, (k, v)

    def test_full_battery_ran_on_every_draw(self, payload):
        rows = {r["check_id"]: r for r in payload["invariants"]}
        assert tuple(sorted(rows, key=BATTERY.index)) == BATTERY
        for cid in BATTERY:
            r = rows[cid]
            assert r["n_pass"] + r["n_fail"] + r["n_not_applicable"] == payload["draws"]["valid"], cid
            assert r["n_fail"] == 0 and r["failing_seeds"] == [] and r["status"] == "pass", cid
        assert rows["I8"]["n_pass"] == payload["draws"]["valid"]
        assert rows["I9"]["n_pass"] == payload["draws"]["valid"]



def test_participant_identity_constants():
    """The stamped identities this twin keys on are the Session's public constants."""
    from omiecid_simulator.omiecid import omie_session as M
    assert M.AGENT_TRADER_ID == PARTICIPANT
    assert M.AGENT_ORDER_ID_BASE == AGENT_ORDER_ID_BASE


# ---------------------------------------------------------------------------
# 2. Eager vs jitted: evidence present and identical (absence = failure)
# ---------------------------------------------------------------------------
# Claim §1.10: integer fields are asserted between modes; the float fields of
# ``top_of_book`` and ``FillReport`` (and the euro / MW accounting) are compared bit for
# bit within a mode and only REPORTED between modes.
ASSERTED_READERS = ("lob", "done")
ASSERTED_OWN = ("trades", "orders")
REPORTED_READERS = ("tob", "fill")


def _eager_pairs(payload, idx):
    """Yield (seed, fam, world, eager_episode, jitted_episode) for every eager_jit seed."""
    for a in payload["assertions"]:
        if a["kind"] != "eager_jit" or a["seed"] is None:
            continue
        s = a["seed"]
        for fam in ("view_noop", "view_random"):
            eager = idx[s].get(fam + "_eager", [])
            jit = idx[s].get(fam, [])
            if len(eager) != 1 or len(jit) != 1:
                yield s, fam, None, None, None
                continue
            for world, k in (("A", "expected"), ("B", "observed")):
                yield s, fam, world, _load(eager[0][k]), _load(jit[0][k])


def judge_eager_jit(payload, idx) -> List[str]:
    """Eager vs jitted: present, recorded identical, and -- own cross-check -- the eager
    episodes equal the jitted ones on every non-euro reader (anonymous LOB, done, own
    trades, own orders, n_steps)."""
    out = []
    ej = [a for a in payload["assertions"] if a["kind"] == "eager_jit"]
    if not ej:
        return ["no eager-vs-jitted evidence"]
    if payload["eager_jit_identical"] is not True:
        out.append(f"eager_jit_identical = {payload['eager_jit_identical']!r}")
    for a in ej:
        if a["status"] != "pass" or a["expected"] != a["observed"]:
            out.append(f"{a['name']}: eager != jitted")
        if a["seed"] is None:
            out.append(f"{a['name']}: no seed")
    for s, fam, world, e, j in _eager_pairs(payload, idx):
        if world is None:
            out.append(f"seed {s}: {fam} eager episode missing")
            continue
        out += _check_episode(e, f"seed {s} {fam} world {world} (eager)")
        try:
            if e.get("n_steps") != j.get("n_steps"):
                out.append(f"seed {s}: {fam} world {world}: eager n_steps != jitted n_steps")
            for r in ASSERTED_READERS:
                if stream(e, r) != stream(j, r):
                    out.append(f"seed {s}: {fam} world {world}: eager {r} != jitted {r}")
            for w in ASSERTED_OWN:
                if own(e, w) != own(j, w):
                    out.append(f"seed {s}: {fam} world {world}: eager own {w} != jitted own {w}")
        except KeyError as exc:
            out.append(f"seed {s}: {fam} world {world}: {exc}")
    return out


def eager_jit_drift(payload, idx, reader: str) -> List[tuple]:
    """(seed, fam, world, step, jitted_entry_is_the_empty_one) where the eager digest of
    ``reader`` differs from the jitted one. The "empty" entry is the noop episode's
    first entry (the noop policy never trades)."""
    zero = None
    for s in _valid_seeds(payload):
        r = stream(_load(_one(idx, s, "view_noop")["expected"]), reader)
        if r:
            zero = r[0]
            break
    drift = []
    for s, fam, world, e, j in _eager_pairs(payload, idx):
        if world is None:
            continue
        for t, (x, y) in enumerate(zip(stream(e, reader), stream(j, reader))):
            if x != y:
                drift.append((s, fam, world, t, y == zero))
    return drift


class TestEagerJit:
    def test_eager_jit_evidence_present_and_identical(self, payload, idx):
        assert judge_eager_jit(payload, idx) == []

    def test_eager_jit_fill_report_drift_is_confined_to_a_traded_euro_figure(self, payload, idx):
        """Reported, not a pass/fail criterion of the claim: the float fields of the
        ``FillReport`` may differ eager-vs-jitted. Allowed only on a step whose report
        is not the empty one (a settled euro figure) and only on the random policy (the
        noop policy never submits, so its report carries no float to drift)."""
        for s, fam, world, t, jit_empty in eager_jit_drift(payload, idx, "fill"):
            assert fam == "view_random", (s, fam, world, t)
            assert not jit_empty, (s, fam, world, t)

    def test_eager_jit_top_of_book_drift_is_reported_only(self, payload, idx):
        """``top_of_book`` floats are reported between modes, never asserted (claim
        §1.10); the drift list is computable from the record (no malformed stream)."""
        drift = eager_jit_drift(payload, idx, "tob")
        assert isinstance(drift, list)


# ---------------------------------------------------------------------------
# 3. Per-draw evidence, judged independently
# ---------------------------------------------------------------------------
class TestPerDraw:
    def test_per_draw_values_present_in_full(self, payload, idx):
        assert judge_completeness(payload, idx) == []

    def test_participant_view_twin_bit_identical_on_every_draw(self, payload, idx):
        """§7.1, §9.2 'LOB per contract' / 'Best bid / best ask', §4.2: no foreign
        hidden quantity, no ids, in top_of_book, get_anonymous_lob, FillReport, done or
        the participant's own trades / orders / cash / inventory."""
        assert judge_view_twin(payload, idx) == []

    def test_positive_control_sees_one_visible_tick_on_every_draw(self, payload, idx):
        assert judge_positive_control(payload, idx) == []

    def test_twin_is_a_valid_twin(self, payload, idx):
        assert judge_twin_validity(payload, idx) == []

    def test_identifiers_actually_permuted(self, payload, idx):
        """The permutation must not be the identity, else the id half of the twin is vacuous."""
        moved_agent = moved_oid = 0
        for s in _valid_seeds(payload):
            eng = _load(_one(idx, s, "engine")["observed"])
            for ta, tb in zip(eng["A"]["trades"], eng["B"]["trades"]):
                moved_agent += (ta[2], ta[3]) != (tb[2], tb[3])
                moved_oid += (ta[0], ta[1]) != (tb[0], tb[1])
        assert moved_agent > 0 and moved_oid > 0

    def test_trade_reports_partition_union_no_counterparty(self, payload, idx):
        """§7.2, §9.2 'Trade book': judged on (price, qty, timestamp), side not required."""
        assert judge_trade_reports(payload, idx) == []

    def test_engine_keeps_identity_and_full_quantity(self, payload, idx):
        assert judge_ownership(payload, idx) == []

    def test_participant_own_orders_identical_across_the_twin(self, payload, idx):
        """(6'') on every draw."""
        assert judge_participant_own_orders(payload, idx) == []

    def test_participant_own_trades_carry_no_identifier(self, payload, idx):
        """§7.2: every own-trade row of every participant-view episode is (price, qty,
        timestamp) with an integral timestamp -- no counterparty field."""
        for s in _valid_seeds(payload):
            for row in own_trades(_episode_of_random(idx, s)):
                assert isinstance(row, list) and len(row) == 3, (s, row)
                assert float(row[2]) == int(row[2]), (s, row)

    def test_non_vacuity_participant_trades(self, payload, idx):
        n = sum(1 for s in _valid_seeds(payload) if own_trades(_episode_of_random(idx, s)))
        assert n >= 1
        n_trades = sum(len(_load(_one(idx, s, "engine")["observed"])["A"]["trades"])
                       for s in _valid_seeds(payload))
        assert n_trades >= 1

    def test_non_vacuity_noop_participant_never_trades(self, payload, idx):
        """The noop policy submits ``AgentOrder.none()`` on every step: no own trade, no
        own order, in either world."""
        for s in _valid_seeds(payload):
            e = _load(_one(idx, s, "view_noop")["expected"])
            assert own_trades(e) == [] and not own(e, "orders"), s

    def test_dataset_has_foreign_icebergs_of_two_owners(self, payload, idx):
        two = sum(1 for s in _valid_seeds(payload)
                  if len({o["owner"] for o in _load(_one(idx, s, "ownership")["observed"]).values()}) >= 2)
        assert two >= 1


# ---------------------------------------------------------------------------
# 4. The verdict and the classification of each failure family
# ---------------------------------------------------------------------------
class TestVerdict:
    LEAK_FAMILIES = ("view_noop", "control", "report_twin", "ownership", "engine",
                     "partition", "own_orders", "own_orders_participant", "twin_validity")

    def test_no_anonymity_leak_recorded(self, payload, idx):
        """Every assertion that could expose a leak passed on every draw: (1),
        (2)-noop, (3), (4), (5), (6), (6'), (6''), (7) -- and the random-policy record,
        whichever of (2) / (2r) the draw carries."""
        for s in _valid_seeds(payload):
            for fam in self.LEAK_FAMILIES:
                assert _one(idx, s, fam)["status"] == "pass", (s, fam)
            rows = idx[s].get("view_random") or idx[s].get("surfacing")
            assert rows and len(rows) == 1 and rows[0]["status"] == "pass", s

    def test_premise_check_matches_own_premise_check(self, payload, idx):
        """PREMISE "no tape iceberg fills a resting participant order on entry". Own
        check: every draw whose random world A drew on a foreign hidden remainder carries
        the rule-implied record (necessary condition); every rule-implied record is
        backed by own evidence (a foreign iceberg of this draw, an own trade covering the
        entry fill); the premise-failed draws are few and real (non-vacuous)."""
        theirs = claimed_premise_failed_seeds(payload, idx)
        assert set(hidden_signature_seeds(payload, idx)) <= set(theirs), \
            (hidden_signature_seeds(payload, idx), theirs)
        for s in theirs:
            assert premise_evidence_problems(idx, s) == [], (s, premise_evidence_problems(idx, s))
        assert len(theirs) >= 1, theirs

    def test_premise_failed_draws_show_exactly_the_rule_implied_outcome(self, payload, idx):
        """§4.2.1 Rule 2 *Incoming iceberg* (1) total quantity, (4) as amended
        2026-10-20: the remainder rests as instance $k_e = \\lfloor M/P \\rfloor$ showing
        $v_1 = \\min(P - u, R)$. With T, h, P, q_B derived here from (1)+(7): k_e, u, R,
        v1, hidden R - v1, B rests q_B - M, gap v1 - (q_B - M); the participant view is
        bit-identical where that gap is 0."""
        for s in claimed_premise_failed_seeds(payload, idx):
            assert rule_implied_problems(payload, idx, s) == [], (s, rule_implied_problems(payload, idx, s))

    def test_rule_implied_gap_is_zero_and_the_retired_display_would_differ(self, payload, idx):
        """Claim (2r): on every premise-failed draw my closed form gives a ZERO gap
        (M = f < P, k_e = 0, u = f, v1 = P - f = q_B - f) and the whole emitted
        participant view is bit-identical, with no difference reported. Discrimination:
        the retired display min(P, R) gives a gap of f >= 1 on each of those draws."""
        seen = 0
        for s in claimed_premise_failed_seeds(payload, idx):
            a = _one(idx, s, "surfacing")
            A, B, det = _load(a["expected"]), _load(a["observed"]), _load(a["detail"])
            assert A["view"] == B["view"], s
            assert det["view_differences"] == [] and det["differs_on"] == [], s
            for oid, r in det["rule_implied"].items():
                f = int(A["icebergs"][oid]["B_entry_fill"])
                w = rule_implied(idx, s, oid, int(r["M"]), f)
                assert w["M"] == f, (s, oid, w)
                assert w["visible_gap"] == 0, (s, oid, w)
                assert (w["k_e"], w["u"], w["shown_A"]) == (0, f, w["P"] - f), (s, oid, w)
                assert w["hidden_A"] == w["h"], (s, oid, w)
                assert w["retired_gap"] == f >= 1, (s, oid, w)
                seen += 1
        assert seen >= 1

    def test_no_premise_failure_discarded_silently(self, payload, idx):
        """Fallback clause: discarding a premise-failed draw is allowed only if counted
        and reported. Every premise failure is asserted, and every discard (if any) is
        accounted for in the draw counts."""
        d = payload["draws"]
        assert d["discarded"] == 0 and d["discard_rate"] == 0.0, d
        assert d["valid"] == d["requested"] == DRAWS
        assert set(hidden_signature_seeds(payload, idx)) <= set(claimed_premise_failed_seeds(payload, idx))

    def test_c3_display_exercised_with_u_positive(self, payload, idx):
        """The amended item (4) display with u > 0 is exercised on >= 1 draw, by my own
        arithmetic, and the payload's global non-vacuity assertion says so."""
        n = 0
        for s in claimed_premise_failed_seeds(payload, idx):
            a = _one(idx, s, "surfacing")
            A, det = _load(a["expected"]), _load(a["detail"])
            for oid, r in det["rule_implied"].items():
                w = rule_implied(idx, s, oid, int(r["M"]), int(A["icebergs"][oid]["B_entry_fill"]))
                n += w["u"] > 0 and w["shown_A"] == w["P"] - w["u"] < w["P"]
        assert n >= 1
        rows = [a for a in payload["assertions"] if a["seed"] is None and "display C3" in a["name"]]
        assert len(rows) == 1 and rows[0]["status"] == "pass", rows

    def test_non_vacuity_participant_iceberg_rests(self, payload):
        """Claim §1.10 (new): at least one valid draw whose random episode ends with a
        participant iceberg resting, so (6'') shows a hidden part."""
        rows = global_assertions(payload, ("iceberg",), ("rest", "hidden", "(6''", "6″"))
        rows = [a for a in rows if "display c3" not in a["name"].lower()]
        assert rows, "results JSON must carry the 'participant iceberg resting' non-vacuity assertion"
        assert all(a["status"] == "pass" for a in rows), rows

    def test_non_vacuity_participant_buys_and_sells(self, payload):
        """Claim §1.10 (new): at least one valid draw on which the participant trades as
        buyer, and one on which it trades as seller."""
        buy = global_assertions(payload, ("buyer", "buys", " buy"))
        sell = global_assertions(payload, ("seller", "sells", " sell"))
        assert buy and sell, "results JSON must carry the buyer / seller non-vacuity assertion(s)"
        assert all(a["status"] == "pass" for a in buy + sell), buy + sell

    def test_own_orders_view_shows_full_iceberg_quantity(self, payload, idx):
        """§9.2 'Own orders ... (showing full iceberg quantity)': owner sees
        visible + hidden, nobody sees another's order -- on every draw."""
        for s in _valid_seeds(payload):
            v = own_orders_verdict(payload, idx, s)
            assert v == {"leak": [], "under_disclosed": [], "other": []}, (s, v)
            assert _one(idx, s, "own_orders")["status"] == "pass", s
        # non-vacuity: some owner is shown strictly more than the displayed peak
        assert any(int(r["hidden"]) > 0 and int(_load(_one(idx, s, "own_orders")["observed"])
                   [str(r["owner"])][oid]) == int(r["visible"]) + int(r["hidden"])
                   for s in _valid_seeds(payload)
                   for oid, r in _load(_one(idx, s, "ownership")["observed"]).items())

    def test_partition_holds_without_side(self, payload, idx):
        """§7.2 / §9.2 'Trade book': judged on (price, qty[, timestamp]) only. Trade
        side is NOT certified (F16): no judge here requires a side field."""
        assert judge_trade_reports(payload, idx) == []
        assert judge_partition_record(payload, idx) == []

    def test_trade_side_not_claimed_and_certificate_carries_the_gap_row(self, payload):
        """F16: the payload certifies no side; the certificate records the §9.2
        trade-side conformance gap ("Not implemented", spec only, not certified, C13
        F16)."""
        claim = payload["known_result"].lower().replace("every trade once per side", "")
        assert not re.search(r"\b(side|buy|sell)\b", claim), claim
        cert = (REPO_ROOT / "docs" / "omiecid" / "certificate.md").read_text(encoding="utf-8")
        rows = [ln for ln in cert.splitlines() if ln.startswith("| §9.2 trade side |")]
        assert len(rows) == 1, rows
        assert "Not implemented" in rows[0] and "Not certified (C13 F16)" in rows[0], rows[0]

    def test_known_result_names_the_participant_view_not_the_rl_observation(self, payload):
        """Claim §1.2: the retargeted Known result names the Session readers and no
        longer the RL observation, reward or action."""
        kr = payload["known_result"]
        assert "top_of_book" in kr and "get_anonymous_lob" in kr, kr
        assert not re.search(r"\b(RL agent|observation|reward)\b", kr), kr

    def test_own_verdict_matches_the_emitted_pass(self, payload, idx):
        """Own verdict: every C13 rule holds on all 500 draws (twin equality where the
        premise holds, the rule-implied outcome where it fails); the emitted PASS
        agrees, with no failing seed and no failure explanation."""
        assert judge_completeness(payload, idx) == []
        assert judge_view_twin(payload, idx) == []
        assert judge_twin_validity(payload, idx) == []
        assert judge_positive_control(payload, idx) == []
        assert judge_trade_reports(payload, idx) == []
        assert judge_partition_record(payload, idx) == []
        assert judge_ownership(payload, idx) == []
        assert judge_participant_own_orders(payload, idx) == []
        assert payload["status"] == "PASS"
        assert payload["failure_kinds"] == []
        assert payload["failure_explanation"] is None
        assert payload["seeds"]["failing"] == []
        assert all(a["status"] == "pass" for a in payload["assertions"])



# ---------------------------------------------------------------------------
# 5. Quotes and citations
# ---------------------------------------------------------------------------
MY_QUOTES = (
    SpecQuote("market-spec.md", "§7.1", "identifying solely those operations performed by the agent itself"),
    SpecQuote("market-spec.md", "§7.2", "the result is to be notified to the agent in question"),
    SpecQuote("market-spec.md", "§7.2", "Counterparty identity is not disclosed."),
    SpecQuote("market-spec.md", "§7.3", "Expose the LOB to agents as anonymous (prices and quantities only, no counterparty identity)."),
    SpecQuote("market-spec.md", "§9.2", "Agent's own active orders (showing full iceberg quantity)"),
    SpecQuote("market-spec.md", "§9.2", "Executed trades: price, quantity, timestamp, contract. Own trades only"),
    SpecQuote("market-spec.md", "§4.2", "Only the peak size is shown in the order book."),
    SpecQuote("market-spec.md", "§4.2.1", "It aggresses with its **total** quantity, not only its visible peak."),
    # item (4) as amended 2026-10-20 (the retired "Its visible quantity is
    # $\min(\mathit{peak}, R)$" no longer reads so, see test_retired_display_no_longer_reads_so)
    SpecQuote("market-spec.md", "§4.2.1", r"the remainder keeps its place in the ladder: it continues as instance $k_e$, at limit $b + k_e\,d$."),
    SpecQuote("market-spec.md", "§4.2.1", r"It shows $v_1 = \min(P - u, R)$, the unused part of instance $k_e$, and keeps $R - v_1$ hidden."),
    SpecQuote("market-spec.md", "§4.2.1", "Additionally, if the total quantity does not match, the viewable quantity that will appear in the Bid Book will be at most the reduced part specified when creating the bid, even if the matched quantity was not a multiple of the reduced part."),
    SpecQuote("market-spec.md", "§9.2", "spec only, no engine change filed, by user decision"),
    # REL-09: the participant view and the participant's orders (claim §2, new quotes)
    SpecQuote("market-spec.md", "§9.2", "| Best bid / best ask | Top-of-book per contract |"),
    SpecQuote("market-spec.md", "§2.3", "**Price tick (minimum increment):** 0.01 EUR/MWh (2 decimal places maximum)."),
    SpecQuote("market-spec.md", "§2.3", "**Quantity tick (minimum increment):** 0.1 MW (1 decimal place maximum)."),
    SpecQuote("market-spec.md", "§4.2", "Icebergs behave like NON orders (resting, partial fills allowed) but carry hidden total quantity."),
    SpecQuote("market-spec.md", "§4.5", "| FOK | No (full or cancel) | No (cancelled) | Regular, Linked basket |"),
    SpecQuote("market-spec.md", "§4.6", "| GFS | Valid until contract's GCT | Yes (default) |"),
)
# The §9.2 "Best bid / best ask" row: top_of_book is that observable (claim §2). At least
# one of these byte-forms must be among the payload's quote digests.
BEST_BID_FORMS = (
    "| Best bid / best ask | Top-of-book per contract |",
    "Best bid / best ask | Top-of-book per contract",
    "Best bid / best ask",
)


class TestQuotes:
    def test_rules_this_audit_relies_on_still_read_so(self):
        for q in MY_QUOTES:
            r = check_quote(q)
            assert r.ok, (q.section, q.text, r.reason)

    def test_retired_display_no_longer_reads_so(self):
        """The r3 display rule was deleted by the 2026-10-20 amendment; this audit must
        not rely on it, and the ladder §C13 Pass row names the amended display."""
        assert not check_quote(SpecQuote(
            "market-spec.md", "§4.2.1", r"Its visible quantity is $\min(\mathit{peak}, R)$")).ok
        ladder = (REPO_ROOT / "docs" / "omiecid" / "scenario-ladder.md").read_text(encoding="utf-8")
        sec = ladder[ladder.index("### C13 "):ladder.index("### C14 ")]
        pass_row = next(ln for ln in sec.splitlines() if ln.startswith("| **Pass** |"))
        assert r"$v_1 = \min(P - u, R)$" in pass_row
        assert any(p in pass_row for p in ("including the agent's fill", "including the participant's fill",
                                           "the participant's fill included")), pass_row

    def test_spec_stamp(self, payload):
        assert check_quote(MY_QUOTES[0]).version == SPEC_VERSION
        stamps = {q["version"] for q in payload["spec"]["quotes"] if q["source"] == "market-spec.md"}
        assert stamps == {SPEC_VERSION}, stamps

    def test_classification_rule_is_in_the_regulatory_corpus(self):
        """Annex §4.4 (incoming iceberg display), the source of classification (a)."""
        if not CORPUS.is_file():
            pytest.skip(f"regulatory corpus absent at {CORPUS}")
        corpus = " ".join(CORPUS.read_text(encoding="utf-8").split())
        assert ("the viewable quantity that will appear in the Bid Book will be at most the "
                "reduced part specified when creating the bid, even if the matched quantity "
                "was not a multiple of the reduced part") in corpus
        assert "the quantity to be considered will be the total quantity" in corpus

    def test_payload_quotes_cover_the_certified_sections(self, payload):
        qs = payload["spec"]["quotes"]
        assert qs and all(q["ok"] and q["reason"] == "ok" for q in qs)
        secs = {(q["source"], q["section"]) for q in qs}
        for need in ("§7.1", "§7.2", "§7.3", "§9.2"):
            assert ("market-spec.md", need) in secs, need
        cur = check_quote(MY_QUOTES[0]).version
        for q in qs:
            if q["source"] == "market-spec.md":
                assert q["version"] == cur, (q["section"], q["version"], cur)

    def test_payload_quotes_the_best_bid_best_ask_row(self, payload):
        """Claim §1.2 certifies §9.2 "Best bid / best ask"; its row must be quoted."""
        have = {q["digest"] for q in payload["spec"]["quotes"]}
        hits = [f for f in BEST_BID_FORMS if quote_digest(SpecQuote("market-spec.md", "§9.2", f)) in have]
        assert hits, "results JSON must quote the §9.2 'Best bid / best ask' row"

    def test_citations_match_the_regulatory_corpus(self, payload):
        if not CORPUS.is_file():
            pytest.skip(f"regulatory corpus absent at {CORPUS}")
        norm = lambda t: " ".join(t.split())  # noqa: E731
        corpus = norm(CORPUS.read_text(encoding="utf-8"))
        cits = payload["citations"]
        texts = " ".join(norm(c["text"]) for c in cits)
        assert "displayed on an anonymous basis, identifying solely those operations performed by the agent itself" in texts
        assert "its result is to be notified to the agent in question" in texts
        for c in cits:
            assert norm(c["text"]) in corpus, c["source"]



# ---------------------------------------------------------------------------
# 6. Deliberately broken copies: the judges must reject them
# ---------------------------------------------------------------------------
def _replace(payload, pred, fn) -> Dict[str, Any]:
    """Shallow copy of payload with the assertions matching ``pred`` rewritten by ``fn``
    (``fn`` returns the new assertion, or None to drop it)."""
    out = dict(payload)
    new = []
    for a in payload["assertions"]:
        if pred(a):
            b = fn(dict(a))
            if b is not None:
                new.append(b)
        else:
            new.append(a)
    out["assertions"] = new
    return out


def _is(seed, fam):
    def pred(a):
        m = SEED_RE.match(a["name"])
        if not (m and int(m.group(1)) == seed):
            return False
        for f, p in FAMILIES.items():      # first match wins, as in ``index``
            if p(m.group(2)):
                return f == fam
        return False
    return pred


def _set_entry(e: Dict[str, Any], reader: str, i: int, value: str) -> None:
    stream(e, reader)[i] = value


def _set_own(e: Dict[str, Any], what: str, value) -> None:
    c, k = own_slot(e, what)
    c[k] = value


def _edit_episode(key: str, fn):
    """Rewrite the (2) episode in world ``key`` (expected = A, observed = B)."""
    def go(a):
        e = _load(a[key])
        fn(e)
        a[key] = json.dumps(e)
        return a
    return go


BROKEN_SEED = 0


def _twin_seed(payload, idx) -> int:
    """A draw carrying the plain random-policy twin (premise holds)."""
    s = next((x for x in _valid_seeds(payload) if idx[x].get("view_random")), None)
    if s is None:
        pytest.skip("no premise-holding draw")
    return s


class TestBrokenCopiesRejected:
    def test_leak_top_of_book_differs_between_worlds(self, payload, tmp_path):
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "view_noop"),
                               _edit_episode("observed", lambda e: _set_entry(e, "tob", 1, "0" * 32))),
                      tmp_path, "leak.json")      # status left "pass": a lying record
        probs = judge_view_twin(bad, index(bad), seeds=[BROKEN_SEED])
        assert any("LEAK" in p for p in probs), probs

    def test_leak_anonymous_lob_differs_between_worlds(self, payload, tmp_path):
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "view_noop"),
                               _edit_episode("observed", lambda e: _set_entry(e, "lob", -1, "0" * 32))),
                      tmp_path, "leak_lob.json")
        probs = judge_view_twin(bad, index(bad), seeds=[BROKEN_SEED])
        assert any("LEAK" in p for p in probs), probs

    def test_leak_fill_report_differs_between_worlds(self, payload, idx, tmp_path):
        s = _twin_seed(payload, idx)

        def leak(a):
            a = _edit_episode("observed", lambda e: _set_entry(e, "fill", 0, "f" * 32))(a)
            a["status"] = "fail"
            return a
        bad = _reload(_replace(payload, _is(s, "view_random"), leak), tmp_path, "rew.json")
        assert judge_view_twin(bad, index(bad), seeds=[s]) != []

    def test_leak_own_orders_differ_between_worlds(self, payload, idx, tmp_path):
        s = _twin_seed(payload, idx)
        bad = _reload(_replace(payload, _is(s, "view_random"),
                               _edit_episode("observed", lambda e: _set_own(e, "orders", [["leak"]]))),
                      tmp_path, "oo_leak.json")
        probs = judge_view_twin(bad, index(bad), seeds=[s])
        assert any("view_random" in p and "LEAK" in p for p in probs), probs

    def test_own_trade_row_carrying_a_counterparty_id(self, payload, idx, tmp_path):
        s = next((x for x in _valid_seeds(payload)
                  if idx[x].get("view_random") and own_trades(_load(_one(idx, x, "view_random")["expected"]))),
                 None)
        if s is None:
            pytest.skip("no premise-holding draw with an own trade")

        def tag(e):
            _set_own(e, "trades", [list(r) + [12345] for r in own_trades(e)])
        bad = _reload(_replace(payload, _is(s, "view_random"),
                               lambda a: _edit_episode("observed", tag)(_edit_episode("expected", tag)(a))),
                      tmp_path, "cp_own.json")
        probs = judge_view_twin(bad, index(bad), seeds=[s])
        assert any("not (price, qty, timestamp)" in p for p in probs), probs

    def test_episode_without_a_lob_stream_is_rejected(self, payload, tmp_path):
        def drop(e):
            e.pop(reader_key(e, "lob"))
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "view_noop"),
                               lambda a: _edit_episode("observed", drop)(_edit_episode("expected", drop)(a))),
                      tmp_path, "nolob.json")
        probs = judge_view_twin(bad, index(bad), seeds=[BROKEN_SEED])
        assert any("no lob stream" in p for p in probs), probs

    def test_report_row_carrying_counterparty_id(self, payload, idx, tmp_path):
        trades = _load(_one(idx, BROKEN_SEED, "engine")["observed"])["A"]["trades"]
        bo, so, ba, sa = trades[0][:4]

        def tag(a):
            for k in ("expected", "observed"):
                rep = _load(a[k])
                rep[str(ba)] = [r + [sa] for r in rep[str(ba)]]     # appended counterparty id
                a[k] = json.dumps(rep)
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "report_twin"), tag), tmp_path, "cp.json")
        probs = judge_trade_reports(bad, index(bad), seeds=[BROKEN_SEED])
        assert any("not (price, qty, timestamp)" in p for p in probs), probs

    def test_report_row_with_counterparty_id_in_timestamp_slot(self, payload, idx, tmp_path):
        trades = _load(_one(idx, BROKEN_SEED, "engine")["observed"])["A"]["trades"]
        ba, sa = trades[0][2], trades[0][3]

        def swap(a):
            for k in ("expected", "observed"):
                rep = _load(a[k])
                rep[str(ba)] = [[r[0], r[1], float(sa)] for r in rep[str(ba)]]
                a[k] = json.dumps(rep)
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "report_twin"), swap), tmp_path, "cp2.json")
        assert judge_trade_reports(bad, index(bad), seeds=[BROKEN_SEED]) != []

    def test_report_differs_under_permutation(self, payload, idx, tmp_path):
        trades = _load(_one(idx, BROKEN_SEED, "engine")["observed"])["A"]["trades"]
        ba = trades[0][2]

        def perm(a):
            rep = _load(a["observed"])
            rep[str(ba)] = rep[str(ba)][:-1]
            a["observed"] = json.dumps(rep)
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "report_twin"), perm), tmp_path, "perm.json")
        probs = judge_trade_reports(bad, index(bad), seeds=[BROKEN_SEED])
        assert any("LEAK" in p for p in probs), probs

    def test_report_with_a_foreign_trade(self, payload, idx, tmp_path):
        def foreign(a):
            for k in ("expected", "observed"):
                rep = _load(a[k])
                empty = [ag for ag, rows in rep.items() if not rows]
                full = [ag for ag, rows in rep.items() if rows]
                if empty and full:
                    rep[empty[0]] = list(rep[full[0]])
                else:
                    rep[full[0]] = rep[full[0]] + rep[full[0]][:1]
                a[k] = json.dumps(rep)
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "report_twin"), foreign), tmp_path, "fx.json")
        assert judge_trade_reports(bad, index(bad), seeds=[BROKEN_SEED]) != []

    def test_missing_positive_control(self, payload, tmp_path):
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "control"), lambda a: None), tmp_path, "nc.json")
        bidx = index(bad)
        assert judge_positive_control(bad, bidx, seeds=[BROKEN_SEED]) != []
        assert judge_completeness(bad, bidx, seeds=[BROKEN_SEED]) != []

    def test_blind_positive_control(self, payload, tmp_path):
        def blind(a):
            c = _load(a["observed"])
            c["CTRL"] = json.loads(json.dumps(c["B"]))
            a["observed"] = json.dumps(c)
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "control"), blind), tmp_path, "blind.json")
        probs = judge_positive_control(bad, index(bad), seeds=[BROKEN_SEED])
        assert any("control blind" in p for p in probs), probs

    def test_missing_eager_jit_evidence(self, payload, tmp_path):
        out = dict(payload)
        out["assertions"] = [a for a in payload["assertions"] if a["kind"] != "eager_jit"]
        bad = _reload(out, tmp_path, "noej.json")
        assert judge_eager_jit(bad, index(bad)) == ["no eager-vs-jitted evidence"]

    def test_eager_episode_disagreeing_with_jitted(self, payload, idx, tmp_path):
        s = next(a["seed"] for a in payload["assertions"] if a["kind"] == "eager_jit")
        bad = _reload(_replace(payload, _is(s, "view_noop_eager"),
                               _edit_episode("expected", lambda e: _set_entry(e, "lob", -1, "1" * 32))),
                      tmp_path, "ejd.json")
        assert judge_eager_jit(bad, index(bad)) != []

    def test_participant_own_orders_differ_between_worlds(self, payload, tmp_path):
        def leak(a):
            a["observed"] = json.dumps({"leak": True})
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "own_orders_participant"), leak),
                      tmp_path, "oop.json")
        probs = judge_participant_own_orders(bad, index(bad), seeds=[BROKEN_SEED])
        assert any("LEAK" in p for p in probs), probs

    def test_missing_participant_own_orders_record(self, payload, tmp_path):
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "own_orders_participant"), lambda a: None),
                      tmp_path, "nop.json")
        bidx = index(bad)
        assert judge_participant_own_orders(bad, bidx, seeds=[BROKEN_SEED]) != []
        assert judge_completeness(bad, bidx, seeds=[BROKEN_SEED]) != []

    def test_foreign_order_in_own_orders_view(self, payload, idx, tmp_path):
        ice = _load(_one(idx, BROKEN_SEED, "ownership")["observed"])
        oid, rec = next(iter(ice.items()))

        def show(a):
            v = _load(a["observed"])
            other = next(ag for ag in v if int(ag) != int(rec["owner"]))
            v[other] = dict(v[other], **{oid: rec["visible"]})
            a["observed"] = json.dumps(v)
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "own_orders"), show), tmp_path, "oo.json")
        v = own_orders_verdict(bad, index(bad), BROKEN_SEED)
        assert v["leak"], v

    def test_vacuous_twin_without_hidden_remainder(self, payload, tmp_path):
        def strip(a):
            e = _load(a["observed"])
            e["A"]["hidden_remaining"] = {}
            a["observed"] = json.dumps(e)
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "engine"), strip), tmp_path, "vac.json")
        probs = judge_twin_validity(bad, index(bad), seeds=[BROKEN_SEED])
        assert any("vacuous twin" in p for p in probs), probs

    def test_participant_iceberg_differing_between_worlds_breaks_the_twin(self, payload, tmp_path):
        def own_ice(a):
            tv = _load(a["observed"])
            tv["B/random"].setdefault("hidden", {})[str(AGENT_ORDER_ID_BASE + 3)] = 7
            a["observed"] = json.dumps(tv)
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "twin_validity"), own_ice), tmp_path, "pi.json")
        probs = judge_twin_validity(bad, index(bad), seeds=[BROKEN_SEED])
        assert any("participant icebergs differ" in p for p in probs), probs

    def test_per_draw_values_truncated(self, payload, tmp_path):
        def blank(a):
            a["observed"] = ""
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "ownership"), blank), tmp_path, "trunc.json")
        assert judge_completeness(bad, index(bad), seeds=[BROKEN_SEED]) != []

    def test_pass_with_a_failing_assertion_is_invalid(self, payload, tmp_path):
        """The emitted PASS cannot survive one failing per-draw record."""
        def fail(a):
            a["status"] = "fail"
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "view_noop"), fail), tmp_path, "pf.json")
        assert validate_results(bad) != []

    # --- the rule-implied outcome must not become a blanket excuse ------------------
    @staticmethod
    def _surf_seed(payload, idx) -> int:
        seeds = claimed_premise_failed_seeds(payload, idx)
        if not seeds:
            pytest.skip("no premise-failed draw")
        return seeds[0]

    @staticmethod
    def _edit_detail(fn):
        def go(a):
            d = _load(a["detail"])
            fn(d)
            a["detail"] = json.dumps(d)
            return a
        return go

    @staticmethod
    def _edit_view(fn, key="observed"):
        """Edit the (2r) record's participant view in world ``key`` (expected = A, observed = B)."""
        def go(a):
            e = _load(a[key])
            fn(e["view"])
            a[key] = json.dumps(e)
            return a
        return go

    def _judge(self, payload, s, fn, tmp_path, name, where="detail"):
        if where == "detail":
            fn = self._edit_detail(fn)
        bad = _reload(_replace(payload, _is(s, "surfacing"), fn), tmp_path, name)
        return judge_view_twin(bad, index(bad), seeds=[s])

    @staticmethod
    def _k_star(idx, s) -> int:
        return int(_load(_one(idx, s, "surfacing")["detail"])["first_entry_step"])

    @staticmethod
    def _tob_key(idx, s) -> str:
        return reader_key(_load(_one(idx, s, "surfacing")["expected"])["view"], "tob")

    def test_random_view_difference_on_a_premise_holding_draw_is_a_leak(self, payload, idx, tmp_path):
        s = _twin_seed(payload, idx)
        bad = _reload(_replace(payload, _is(s, "view_random"),
                               _edit_episode("observed", lambda e: _set_entry(e, "tob", 1, "0" * 32))),
                      tmp_path, "rl.json")
        probs = judge_view_twin(bad, index(bad), seeds=[s])
        assert any("view_random" in p and "LEAK" in p for p in probs), probs

    def test_rule_implied_record_on_a_premise_holding_draw_is_rejected(self, payload, idx, tmp_path):
        """Renaming a premise-holding draw's (2) record into a (2r) record (an excuse
        without a sweep of the participant's order: the donor's iceberg is not of this
        draw) must be caught."""
        s0 = self._surf_seed(payload, idx)
        donor = _one(idx, s0, "surfacing")
        target = _twin_seed(payload, idx)

        def relabel(a):
            b = dict(donor)
            b["seed"] = target
            b["name"] = f"seed {target} " + SEED_RE.match(donor["name"]).group(2)
            return b
        bad = _reload(_replace(payload, _is(target, "view_random"), relabel), tmp_path, "ex.json")
        probs = judge_view_twin(bad, index(bad), seeds=[target])
        assert any("premise holds" in p for p in probs), probs

    def test_premise_failed_draw_judged_as_a_plain_twin_is_rejected(self, payload, idx, tmp_path):
        """Dropping the (2r) record of a premise-failed draw (silent discard) is caught."""
        s = self._surf_seed(payload, idx)
        bad = _reload(_replace(payload, _is(s, "surfacing"), lambda a: None), tmp_path, "drop.json")
        bidx = index(bad)
        assert judge_view_twin(bad, bidx, seeds=[s]) != []
        assert judge_completeness(bad, bidx, seeds=[s]) != []

    def test_view_difference_where_the_rule_implies_none_is_a_leak(self, payload, idx, tmp_path):
        """The rule-implied gap is 0 on the premise-failed draws, so ANY difference after
        the entry step (even a touch-quantity one, reported honestly) is a leak."""
        s = self._surf_seed(payload, idx)
        k = self._k_star(idx, s)
        tk = self._tob_key(idx, s)

        def go(a):
            a = self._edit_view(lambda v: _set_entry(v, "tob", k, "0" * 32), "expected")(a)
            d = _load(a["detail"])
            d["differs_on"] = [tk]
            d["view_differences"] = [{"field": f"{tk}.q_bid", "step": k, "A": "1.5", "B": "1.0"}]
            a["detail"] = json.dumps(d)
            return a
        probs = self._judge(payload, s, go, tmp_path, "vd.json", where="record")
        assert any("rule implies a zero gap (LEAK)" in p for p in probs), probs
        assert any("rule implies 1..0" in p for p in probs), probs

    def test_unreported_view_difference_is_a_leak(self, payload, idx, tmp_path):
        s = self._surf_seed(payload, idx)
        probs = self._judge(payload, s,
                            self._edit_view(lambda v: _set_entry(v, "lob", -1, "0" * 32), "expected"),
                            tmp_path, "uv.json", where="record")
        assert any("LEAK" in p for p in probs) and any("no difference is reported" in p for p in probs), probs

    def test_engine_showing_the_retired_full_peak_is_rejected(self, payload, idx, tmp_path):
        """DISCRIMINATION: a copy written by an engine that still shows min(P, R) after
        sweeping the participant's order (the r3 display) -- shown_A = P, gap f, hidden
        R - P, A's touch quantity f above B's from the entry step on -- must fail."""
        s = self._surf_seed(payload, idx)
        k = self._k_star(idx, s)
        tk = self._tob_key(idx, s)
        n = len(stream(_load(_one(idx, s, "surfacing")["expected"])["view"], "tob"))

        def go(a):
            d = _load(a["detail"])
            gap = 0
            for oid, r in d["rule_implied"].items():
                old = r["retired_C1"]
                r["shown_A"], r["visible_gap"] = old["shown_A"], old["visible_gap"]
                r["hidden_A"] = int(r["R"]) - int(old["shown_A"])
                r["hidden_unchanged"] = False
                gap += int(old["visible_gap"])
            d["differs_on"] = [tk]
            d["view_differences"] = [{"field": f"{tk}.q_bid", "step": t, "A": str(1.0 + gap / 10),
                                      "B": "1.0"} for t in range(k, n)]
            a["detail"] = json.dumps(d)
            for key in ("expected", "observed"):
                e = _load(a[key])
                for oid, rec in e["icebergs"].items():
                    rr = d["rule_implied"][oid]
                    rec["hidden_A"], rec["visible_gap_A_minus_B"] = rr["hidden_A"], rr["visible_gap"]
                if key == "expected":
                    t = stream(e["view"], "tob")
                    t[k:] = ["0" * 32] * len(t[k:])
                a[key] = json.dumps(e)
            return a
        probs = self._judge(payload, s, go, tmp_path, "ret.json", where="record")
        assert any("shown_A" in p and "rule implies" in p for p in probs), probs
        assert any("hidden_A" in p for p in probs), probs

    def test_claimed_visible_gap_off_the_closed_form_is_rejected(self, payload, idx, tmp_path):
        s = self._surf_seed(payload, idx)

        def off(d):
            r = next(iter(d["rule_implied"].values()))
            r["visible_gap"] += 1
            r["shown_A"] += 1
        probs = self._judge(payload, s, off, tmp_path, "gap.json")
        assert any("visible_gap" in p for p in probs), probs

    def test_claimed_instance_off_the_closed_form_is_rejected(self, payload, idx, tmp_path):
        """k_e / u not floor(M/P) / M - k_e P (e.g. a remainder parked on a fresh
        instance, u = 0) is caught."""
        s = self._surf_seed(payload, idx)

        def off(d):
            r = next(iter(d["rule_implied"].values()))
            r["k_e"], r["u"] = 1, 0
        probs = self._judge(payload, s, off, tmp_path, "ke.json")
        assert any("k_e" in p for p in probs) and any(": u = " in p for p in probs), probs

    def test_claimed_hidden_off_the_closed_form_is_rejected(self, payload, idx, tmp_path):
        """A hidden remainder that is not R - v1 (e.g. the participant's fill taken out of
        the hidden part, the r3 reading) is caught."""
        s = self._surf_seed(payload, idx)
        f = {oid: int(r["B_entry_fill"]) for oid, r in
             _load(_one(idx, s, "surfacing")["expected"])["icebergs"].items()}

        def off(d):
            oid, r = next(iter(d["rule_implied"].items()))
            r["hidden_A"] -= f[oid]
        probs = self._judge(payload, s, off, tmp_path, "hid.json")
        assert any("hidden_A" in p for p in probs), probs

    def test_f_max_not_from_q_agent_max_is_rejected(self, payload, idx, tmp_path):
        """Claim (2r): f_max = Q_AGENT_MAX x K. The r4 bound (AGENT_QTY_MAX = 2) is caught."""
        s = self._surf_seed(payload, idx)
        K = int(_load(_one(idx, s, "surfacing")["expected"])["view"]["n_steps"])

        def off(d):
            d["f_max"] = 2 * K
        probs = self._judge(payload, s, off, tmp_path, "fmax.json")
        assert any("f_max" in p for p in probs), probs

    def test_hidden_drawn_on_a_premise_failed_draw_off_the_rule_is_rejected(self, payload, idx, tmp_path):
        """(7) A/random hidden reduced by f on a premise-failed draw (the r3 sweep-from-
        hidden reading) disagrees with item (4) and is caught."""
        s = self._surf_seed(payload, idx)
        A = _load(_one(idx, s, "surfacing")["expected"])
        oid, rec = next(iter(A["icebergs"].items()))

        def draw(a):
            tv = _load(a["observed"])
            tv["A/random"]["hidden"][oid] -= int(rec["B_entry_fill"])
            a["observed"] = json.dumps(tv)
            return a
        bad = _reload(_replace(payload, _is(s, "twin_validity"), draw), tmp_path, "hr.json")
        probs = judge_view_twin(bad, index(bad), seeds=[s])
        assert any("A/random hidden" in p for p in probs), probs
        assert judge_twin_validity(bad, index(bad), seeds=[s]) != []

    def test_fill_not_the_participants_is_rejected(self, payload, idx, tmp_path):
        """The premise is about the PARTICIPANT's resting order: an entry that filled only
        a foreign order is no excuse."""
        s = self._surf_seed(payload, idx)

        def foreign(a):
            for k in ("expected", "observed"):
                e = _load(a[k])
                for rec in e["icebergs"].values():
                    rec["A_entry_passive_agents"] = [12345]
                a[k] = json.dumps(e)
            return a
        probs = self._judge(payload, s, foreign, tmp_path, "fa.json", where="record")
        assert any("did not fill the participant" in p for p in probs), probs

    def test_rule_implied_without_an_own_trade_is_rejected(self, payload, idx, tmp_path):
        s = self._surf_seed(payload, idx)

        def notrade(v):
            _set_own(v, "trades", [])

        def go(a):
            a = self._edit_view(notrade, "expected")(a)
            return self._edit_view(notrade, "observed")(a)
        probs = self._judge(payload, s, go, tmp_path, "nt.json", where="record")
        assert any("premise holds" in p and "did not trade" in p for p in probs), probs

    def test_difference_on_a_price_field_is_a_leak(self, payload, idx, tmp_path):
        s = self._surf_seed(payload, idx)
        k = self._k_star(idx, s)

        def price(d):
            d["view_differences"] = [{"field": "top_of_book.p_bid", "step": k, "A": "51.0", "B": "50.0"}]
        assert any("non-quantity field" in p for p in self._judge(payload, s, price, tmp_path, "px.json"))

    def test_difference_before_the_entry_step_is_a_leak(self, payload, idx, tmp_path):
        s = self._surf_seed(payload, idx)

        def early(d):
            d["view_differences"] = [{"field": "top_of_book.q_bid", "step": 0, "A": "1.1", "B": "1.0"}]
        assert any("before the entry step" in p for p in self._judge(payload, s, early, tmp_path, "ea.json"))

    def test_rule_implied_with_different_own_trades_is_a_leak(self, payload, idx, tmp_path):
        s = self._surf_seed(payload, idx)

        def rep(v):
            _set_own(v, "trades", own_trades(v)[:-1])
        probs = self._judge(payload, s, self._edit_view(rep), tmp_path, "or.json", where="record")
        assert any("own trades differs" in p for p in probs), probs

    def test_rule_implied_with_different_own_cash_is_a_leak(self, payload, idx, tmp_path):
        s = self._surf_seed(payload, idx)

        def cash(v):
            _set_own(v, "cash", "tampered")
        probs = self._judge(payload, s, self._edit_view(cash), tmp_path, "cash.json", where="record")
        assert any("own cash differs" in p for p in probs), probs

    def test_rule_implied_with_a_different_pre_entry_top_of_book_is_a_leak(self, payload, idx, tmp_path):
        s = self._surf_seed(payload, idx)
        k = self._k_star(idx, s)
        probs = self._judge(payload, s, self._edit_view(lambda v: _set_entry(v, "tob", k - 1, "0" * 32)),
                            tmp_path, "pre.json", where="record")
        assert any("before the entry step differs" in p for p in probs), probs

    def test_rule_implied_with_a_different_fill_report_is_a_leak(self, payload, idx, tmp_path):
        s = self._surf_seed(payload, idx)
        probs = self._judge(payload, s, self._edit_view(lambda v: _set_entry(v, "fill", -1, "f" * 32)),
                            tmp_path, "rw.json", where="record")
        assert any("zero gap (LEAK)" in p for p in probs), probs

    def test_rule_implied_when_the_noop_twin_also_differs_is_a_leak(self, payload, idx, tmp_path):
        s = self._surf_seed(payload, idx)
        bad = _reload(_replace(payload, _is(s, "view_noop"),
                               _edit_episode("observed", lambda e: _set_entry(e, "tob", -1, "0" * 32))),
                      tmp_path, "nd.json")
        probs = judge_view_twin(bad, index(bad), seeds=[s])
        assert any("noop" in p and "LEAK" in p for p in probs), probs

    def test_hidden_drawn_without_an_own_trade_breaks_the_twin(self, payload, idx, tmp_path):
        def draw(a):
            tv = _load(a["observed"])
            k = next(kk for kk in tv["A/random"]["hidden"] if not _is_participant_oid(kk))
            tv["A/random"]["hidden"][k] -= 1
            a["observed"] = json.dumps(tv)
            return a
        s = next((x for x in _valid_seeds(payload)
                  if not own_trades(_episode_of_random(idx, x))), None)
        if s is None:
            pytest.skip("every draw has a participant trade")
        bad = _reload(_replace(payload, _is(s, "twin_validity"), draw), tmp_path, "hd.json")
        probs = judge_twin_validity(bad, index(bad), seeds=[s])
        assert any("A/random hidden" in p for p in probs), probs

    def test_owner_shown_only_the_peak_is_under_disclosure(self, payload, idx, tmp_path):
        ice = _load(_one(idx, BROKEN_SEED, "ownership")["observed"])
        oid, rec = next((k, r) for k, r in ice.items() if int(r["hidden"]) > 0)

        def peak(a):
            v = _load(a["observed"])
            v[str(rec["owner"])][oid] = int(rec["visible"])
            a["observed"] = json.dumps(v)
            return a
        bad = _reload(_replace(payload, _is(BROKEN_SEED, "own_orders"), peak), tmp_path, "ud.json")
        assert own_orders_verdict(bad, index(bad), BROKEN_SEED)["under_disclosed"]

    def test_partition_leg_moved_to_another_agent(self, payload, idx, tmp_path):
        def move(a):
            for k in ("expected", "observed"):
                r = _load(a[k])
                full = [ag for ag, rows in r["reports"].items() if rows]
                other = [ag for ag in r["reports"] if ag != full[0]]
                r["reports"][other[0]] = r["reports"][other[0]] + [r["reports"][full[0]].pop()]
                a[k] = json.dumps(r)
            return a
        s = next(x for x in _valid_seeds(payload)
                 if _load(_one(idx, x, "engine")["observed"])["A"]["trades"])
        bad = _reload(_replace(payload, _is(s, "partition"), move), tmp_path, "pm.json")
        assert judge_partition_record(bad, index(bad), seeds=[s]) != []


# ---------------------------------------------------------------------------
# 7. Independent book probes (slow): own tapes, ContractBook public entry points only
# ---------------------------------------------------------------------------
def _book_world(world: str):
    """World A: a foreign ICE sell (agent 13, 20 MW, peak 2) at the touch plus a
    crossing IOC; world B: the same with the hidden remainder removed and every
    agent id and order id permuted."""
    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.contract_book import ContractBook
    from omiecid_simulator.omiecid.order_types import ExecRestriction, OMIEOrder, Side, Status, Validity

    BUY, SELL = int(Side.BUY), int(Side.SELL)
    NON, IOC, ICE = int(ExecRestriction.NON), int(ExecRestriction.IOC), int(ExecRestriction.ICE)
    perm = {11: 14, 12: 13, 13: 12, 14: 11} if world == "B" else {k: k for k in (11, 12, 13, 14)}
    rows = [(BUY, 49.0, 10.0, 11, NON, 0.0), (BUY, 48.5, 6.0, 12, NON, 0.0),
            (SELL, 51.0, 20.0, 13, ICE, 2.0), (SELL, 52.0, 5.0, 14, NON, 0.0),
            (SELL, 48.9, 4.0, 14, IOC, 0.0)]
    cb = ContractBook(Contract(contract_id=0, delivery_start_minutes=555, delivery_end_minutes=570,
                               gct_minutes=495, status=ContractStatus.OPEN))
    st = cb.init()
    for i, (side, p, q, ag, ex, pk) in enumerate(rows):
        if ex == ICE and world == "B":
            ex, q, pk = NON, 2.0, 0.0
        oid = i + 1 if world != "B" else 900 - i
        st, res = cb.submit_order(st, OMIEOrder(
            contract_id=0, side=side, price=p, quantity=q, order_id=oid, agent_id=perm[ag],
            timestamp=i + 1, exec_restriction=ex, validity=int(Validity.GFS), expiry_time=0,
            status=int(Status.ACTIVE), peak_qty=pk, ppd=0.0))
        assert bool(res.accepted)
    return cb, st, perm


@pytest.mark.slow
@pytest.mark.jit
def test_probe_book_views_anonymous_and_ownership(payload, idx):
    """slow: two ContractBook worlds (jitted submit_order, ~10 s of compilation).
    Independent reproduction of (a) anonymity of the LOB view, (b) own-trade
    notification without counterparty, (c) the §9.2 own-orders side-condition (owner
    sees visible + hidden) -- whose verdict must agree with (6')."""
    import numpy as np

    views = {}
    for w in ("A", "B"):
        cb, st, perm = _book_world(w)
        lob = [[(float(p), float(q)) for p, q in side] for side in cb.get_anonymous_lob(st)]
        orders = {ag: cb.get_agent_orders(st, perm[ag]) for ag in (11, 12, 13, 14)}
        trades = {ag: [tuple(float(x) for x in r) for r in cb.get_agent_trades(st, perm[ag])]
                  for ag in (11, 12, 13, 14)}
        views[w] = (lob, orders, trades, perm)

    lob_a, orders_a, trades_a, _ = views["A"]
    lob_b, orders_b, trades_b, perm_b = views["B"]
    # (a) §7.1/§4.2: the anonymous book is identical; only the 2 MW peak shows at 51.00
    assert lob_a == lob_b
    assert (51.0, 2.0) in lob_a[1]
    # (b) §7.2: own trades only (the IOC seller 14 and the resting buyer 11), no ids,
    # byte-identical under permuted identifiers
    assert trades_a == trades_b
    assert trades_a[12] == [] and trades_a[13] == []
    assert len(trades_a[11]) == len(trades_a[14]) == 1 and trades_a[11] == trades_a[14]
    for rows in trades_a.values():
        for r in rows:
            # (price, qty, timestamp) only; equality across worlds (above) proves the
            # third column is not an identifier, since every id was permuted in B
            assert len(r) == 3 and r[0] == 49.0 and r[1] == 4.0, r
    # (c) own-orders: nobody sees another agent's order
    for ag, os_ in orders_a.items():
        assert all(int(o.agent_id) == ag for o in os_), ag
    for ag, os_ in orders_b.items():
        assert all(int(o.agent_id) == perm_b[ag] for o in os_), ag
    ice = [o for o in orders_a[13] if np.isclose(float(o.price), 51.0)]
    assert len(ice) == 1
    # §9.2 "Own orders ... (showing full iceberg quantity)": 20 MW, not the 2 MW peak
    owner_sees_full = np.isclose(float(ice[0].quantity), 20.0)
    assert owner_sees_full, float(ice[0].quantity)
    # the emitted (6') verdict must agree with this independent reproduction
    json_6p_pass = all(_one(idx, s, "own_orders")["status"] == "pass" for s in _valid_seeds(payload))
    assert bool(owner_sees_full) == json_6p_pass


def _sweep_world(rows):
    """Submit ``rows`` = (side, price, qty, agent, exec, peak) in order; return views."""
    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.contract_book import ContractBook
    from omiecid_simulator.omiecid.order_types import OMIEOrder, Status, Validity

    cb = ContractBook(Contract(contract_id=0, delivery_start_minutes=555, delivery_end_minutes=570,
                               gct_minutes=495, status=ContractStatus.OPEN))
    st = cb.init()
    for i, (side, p, q, ag, ex, pk) in enumerate(rows):
        st, res = cb.submit_order(st, OMIEOrder(
            contract_id=0, side=side, price=p, quantity=q, order_id=i + 1, agent_id=ag,
            timestamp=i + 1, exec_restriction=ex, validity=int(Validity.GFS), expiry_time=0,
            status=int(Status.ACTIVE), peak_qty=pk, ppd=0.0))
        assert bool(res.accepted)
    lob = tuple(tuple((round(float(p), 2), round(float(q), 1)) for p, q in side)
                for side in cb.get_anonymous_lob(st))
    owner = [(round(float(o.price), 2), round(float(o.quantity), 1)) for o in cb.get_agent_orders(st, 13)]
    rl = [tuple(round(float(x), 2) for x in r) for r in cb.get_agent_trades(st, 77)]
    return lob, owner, rl


@pytest.mark.slow
@pytest.mark.jit
def test_probe_incoming_iceberg_sweeping_own_order_shows_unused_instance_part():
    """slow: 28 ContractBook worlds (jitted submit_order, one compile). The r4
    premise-failed outcome reproduced from the rule at book level, re-derived from
    §4.2.1 Rule 2 *Incoming iceberg* (1) total quantity and (4) as amended 2026-10-20:
    the remainder rests as instance $k_e = \\lfloor M/P \\rfloor$ at $b$ showing
    $v_1 = \\min(P - u, R)$, $u = M - k_e P$, $M$ the swept quantity INCLUDING the
    agent's fill; Annex §4.4 "at most the reduced part".

    Book: agent 77 (the participant) optionally rests SELL 0.1 @ 50.00 first, foreign
    agent 21 SELL q_f @ 50.00, agent 22 SELL 1.0 @ 52.00; then foreign agent 13 enters a
    crossing BUY ICE @ 50.00, peak 2.0 (P = 20 ticks), total T. Twin: the same BUY as a
    plain NON of q_B = noop fills + noop display (the twin's construction).
    DISCRIMINATING: an engine showing the retired min(P, R) fails every case with u > 0
    (it would show 2.0 MW where the rule shows 1.2 / 1.4 / 0.4 ...)."""
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side

    BUY, SELL = int(Side.BUY), int(Side.SELL)
    NON, ICE = int(ExecRestriction.NON), int(ExecRestriction.ICE)
    P = 20

    def rows(qf, agent_rests, last):
        own = [(SELL, 50.0, 0.1, 77, NON, 0.0)] if agent_rests else []
        return own + [(SELL, 50.0, qf, 21, NON, 0.0), (SELL, 52.0, 1.0, 22, NON, 0.0), last]

    def bid_ticks(lob):
        bids, asks = lob
        assert asks == ((52.0, 1.0),), lob
        assert all(p == 50.0 for p, _ in bids), lob
        return sum(q_int(q) for _, q in bids)

    # (q_f, T): k_e = 0 (u = 8 / 7), R < P - u, k_e = 1 (u = 6 / 5), u = 0 with the agent
    cases = ((0.7, 2.0), (0.7, 2.7), (0.7, 10.0), (0.7, 30.0), (2.5, 3.0), (2.5, 10.0), (1.9, 10.0))
    retired_differs = 0
    lobs = {}
    for qf, T in cases:
        Tt, ff = q_int(T), q_int(qf)
        a_noop, a_agent = item4_display(Tt, P, ff), item4_display(Tt, P, ff + 1)
        q_B = ff + a_noop["v1"]
        WA = {ag: _sweep_world(rows(qf, ag, (BUY, 50.0, T, 13, ICE, 2.0))) for ag in (False, True)}
        WB = {ag: _sweep_world(rows(qf, ag, (BUY, 50.0, round(q_B / 10, 1), 13, NON, 0.0)))
              for ag in (False, True)}
        lobs[(qf, T)] = WA[True][0]
        # noop (agent not in the sweep): A shows item (4)'s v1 and the twin is exact
        assert bid_ticks(WA[False][0]) == a_noop["v1"], (qf, T, WA[False][0])
        assert WA[False][0] == WB[False][0], (qf, T)
        # agent swept: A shows v1 = min(P - u, R) with u from M = q_f + 0.1
        shown = bid_ticks(WA[True][0])
        assert shown == a_agent["v1"], (qf, T, a_agent, WA[True][0])
        # §9.2 own orders: the owner sees the full remainder R = T - M, at b (rest price)
        assert [(p, q_int(q)) for p, q in WA[True][1]] == [(50.0, a_agent["R"])], (qf, T, WA[True][1])
        assert q_int(WA[True][1][0][1]) - shown == a_agent["hidden"], (qf, T)
        # (1) + §7.2: the agent's own leg is swept in full, no counterparty field
        assert [r[:2] for r in WA[True][2]] == [(50.0, 0.1)], WA[True][2]
        assert all(len(r) == 3 for r in WA[True][2])
        assert [r[:2] for r in WB[True][2]] == [r[:2] for r in WA[True][2]]
        # the twin rests q_B - M; the A-B gap is v1 - (q_B - M) exactly
        assert bid_ticks(WB[True][0]) == max(q_B - (ff + 1), 0), (qf, T, WB[True][0])
        assert shown - bid_ticks(WB[True][0]) == a_agent["v1"] - (q_B - ff - 1), (qf, T)
        if (ff + 1) // P == ff // P:
            # inside one instance (u0 + f < P): the twin is EXACT, the gap 0
            assert WA[True][0] == WB[True][0], (qf, T)
        retired_differs += min(P, a_agent["R"]) != shown
    # the case with u = 0 after the agent: a fresh instance shows a full peak
    assert lobs[(1.9, 10.0)] == (((50.0, 2.0),), ((52.0, 1.0),))
    # NOT a leak: the display cannot tell a 10 MW iceberg from a 30 MW one
    assert lobs[(0.7, 10.0)] == lobs[(0.7, 30.0)] == lobs[(0.7, 2.7)]
    assert retired_differs == 4, retired_differs


# ---------------------------------------------------------------------------
# 8. Independent Session probes (slow): the participant view, public API only
# ---------------------------------------------------------------------------
def _reader_book():
    """A ContractBook used only for its public readers over ``state.book`` (same
    product / market defaults as the Session)."""
    from omiecid_simulator.omiecid.contract import Contract, ContractStatus
    from omiecid_simulator.omiecid.contract_book import ContractBook
    return ContractBook(Contract(contract_id=0, delivery_start_minutes=555, delivery_end_minutes=570,
                                 gct_minutes=495, status=ContractStatus.OPEN))


def _bytes(tree) -> bytes:
    import jax
    import numpy as np
    return b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(tree))


def _own_orders_rows(cb, book) -> List[tuple]:
    from omiecid_simulator.omiecid.omie_session import AGENT_TRADER_ID
    return sorted((int(o.side), round(float(o.price), 2), round(float(o.quantity), 1),
                   int(o.order_id), int(o.timestamp))
                  for o in cb.get_agent_orders(book, AGENT_TRADER_ID))


def _participant_episode(env, cb, key, orders, tape_idx=None) -> Dict[str, Any]:
    """Run one episode and record the participant view (claim §1.3): top_of_book and
    the anonymous LOB after reset and every step, the FillReport and done of every
    step, and the end-of-episode own trades / orders / cash / inventory. Readers are
    recorded raw (decoded) and as bytes (for bit-identity)."""
    import jax.numpy as jnp
    import numpy as np
    from omiecid_simulator.omiecid.omie_session import AGENT_TRADER_ID

    st = env.reset(key)
    if tape_idx is not None:
        st = st.replace(tape_idx=jnp.int32(tape_idx))
    tob = [env.top_of_book(st)]
    lob = [cb.get_anonymous_lob(st.book)]
    fills, dones = [], []
    for order in orders:
        st, rep, done = env.step_env(key, st, order)
        tob.append(env.top_of_book(st))
        lob.append(cb.get_anonymous_lob(st.book))
        fills.append(rep)
        dones.append(bool(np.asarray(done)))
    view = {
        "tob": [_bytes(t) for t in tob],
        "lob": [repr(x) for x in lob],
        "fill": [_bytes(f) for f in fills],
        "done": dones,
        "own_trades": list(cb.get_agent_trades(st.book, AGENT_TRADER_ID)),
        "own_orders": _own_orders_rows(cb, st.book),
        "cash": _bytes(st.agent_cash),
        "inventory": _bytes(st.agent_inventory),
    }
    return {"view": view, "tob_raw": tob, "lob_raw": lob, "fills_raw": fills, "final": st}


def _env_tape(world: str):
    import numpy as np
    from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side

    BUY, SELL = int(Side.BUY), int(Side.SELL)
    IOC, ICE = int(ExecRestriction.IOC), int(ExecRestriction.ICE)
    perm = {11: 14, 12: 13, 13: 12, 14: 11} if world == "B" else {k: k for k in (11, 12, 13, 14)}
    rows = [
        dict(side=BUY, price_eur=49.0, qty_mw=10.0, agent_id=11),
        dict(side=BUY, price_eur=48.5, qty_mw=6.0, agent_id=12),
        dict(side=SELL, price_eur=51.0, qty_mw=20.0, agent_id=13, exec_restriction=ICE, peak_qty_mw=2.0),
        dict(side=SELL, price_eur=52.0, qty_mw=5.0, agent_id=14),
        dict(side=BUY, price_eur=49.0, qty_mw=3.0, agent_id=12),
        dict(side=SELL, price_eur=48.9, qty_mw=4.0, agent_id=14, exec_restriction=IOC),
        dict(side=BUY, price_eur=47.0, qty_mw=2.0, agent_id=11),
        dict(side=SELL, price_eur=53.0, qty_mw=3.0, agent_id=12),
        dict(side=BUY, price_eur=48.0, qty_mw=4.0, agent_id=13),
        dict(side=SELL, price_eur=52.5, qty_mw=2.0, agent_id=11),
        dict(side=BUY, price_eur=48.8, qty_mw=1.0, agent_id=14),
        dict(side=SELL, price_eur=51.5, qty_mw=1.0, agent_id=12),
    ]
    tb = TapeBuilder()
    n = len(rows)
    for i, r in enumerate(rows):
        r = dict(r, agent_id=perm[r["agent_id"]])
        if r.get("exec_restriction") == ICE and world in ("B", "CTRL"):
            r.pop("exec_restriction")
            r.pop("peak_qty_mw")
            r["qty_mw"] = 2.0 if world == "B" else 2.1     # CTRL: one visible qty tick more
        tb.add(order_id=(i + 1) if world != "B" else 500 + n - i, timestamp=60 + 30 * i, **r)
    return np.asarray(tb.build(pad_extra=8))[None]


def _mixed_orders():
    """Claim §4 shapes, by hand: a crossing IOC BUY that takes 0.3 MW of the foreign
    iceberg's 2 MW peak (never exhausting it), a crossing FOK SELL of 0.2 MW into the
    49.00 bid, a participant ICE SELL (0.6 MW, peak 0.2) resting at 50.50 inside the
    spread, a NON BUY submitted after done (ignored), then nothing. Every quantity <=
    Q_AGENT_MAX ticks."""
    from omiecid_simulator.omiecid.omie_session import AgentOrder
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side
    BUY, SELL = int(Side.BUY), int(Side.SELL)
    ER = ExecRestriction
    return [AgentOrder.create(BUY, 51.0, 0.3, exec_restriction=int(ER.IOC)),
            AgentOrder.create(SELL, 48.0, 0.2, exec_restriction=int(ER.FOK)),
            AgentOrder.create(SELL, 50.5, 0.6, exec_restriction=int(ER.ICE), peak_qty=0.2),
            AgentOrder.create(BUY, 47.5, 0.1),
            AgentOrder.none(), AgentOrder.none()]


@pytest.fixture(scope="module")
def env_twin():
    """Worlds A, B (and CTRL for the noop policy) through three Session instances."""
    import jax
    from omiecid_simulator.omiecid.omie_session import AgentOrder, Session

    cb = _reader_book()
    key = jax.random.PRNGKey(0)
    pol = {"noop": [AgentOrder.none()] * 6, "mixed": _mixed_orders()}
    out = {}
    for world, names in (("A", pol), ("B", pol), ("CTRL", {"noop": pol["noop"]})):
        env = Session({"tapes": _env_tape(world)})
        out[world] = {name: _participant_episode(env, cb, key, orders) for name, orders in names.items()}
    return out


@pytest.mark.slow
@pytest.mark.jit
class TestSessionParticipantViewTwin:
    """slow: three Session instances, one compile each (~60-100 s on CPU). Independent
    two-world probe of the retargeted claim (2), (3), (6'') and F23."""

    def test_participant_view_bit_identical_across_the_twin(self, env_twin):
        for name in ("noop", "mixed"):
            a, b = env_twin["A"][name]["view"], env_twin["B"][name]["view"]
            for k in a:
                assert a[k] == b[k], f"{name}: {k} differs between A and B (LEAK)"

    def test_trade_rows_carry_no_foreign_identifier(self, env_twin):
        """§7.2: neither the FillReport's trade rows nor the carried participant tape
        names a tape agent (11-14 in A, permuted in B)."""
        import numpy as np
        for w in ("A", "B"):
            ep = env_twin[w]["mixed"]
            rows = np.concatenate([np.asarray(f.trade_rows).reshape(-1, 6) for f in ep["fills_raw"]]
                                  + [np.asarray(ep["final"].agent_trades).reshape(-1, 6)])
            rows = rows[rows[:, 2] != 0]
            assert len(rows), w
            assert not (set(rows[:, 3].tolist()) | set(rows[:, 4].tolist())) & {11, 12, 13, 14}, (w, rows)
            for r in ep["view"]["own_trades"]:
                assert len(r) == 3, r

    def test_participant_trades_as_buyer_and_seller(self, env_twin):
        """Non-vacuity: the IOC BUY and the FOK SELL both fill, in both worlds."""
        import numpy as np
        for w in ("A", "B"):
            inv = [float(np.asarray(f.inventory_delta)) for f in env_twin[w]["mixed"]["fills_raw"]]
            assert inv[0] == pytest.approx(0.3) and inv[1] == pytest.approx(-0.2), (w, inv)
            assert [r[:2] for r in env_twin[w]["mixed"]["view"]["own_trades"]] == [(51.0, 0.3), (49.0, 0.2)] \
                or sorted(round(r[1], 1) for r in env_twin[w]["mixed"]["view"]["own_trades"]) == [0.2, 0.3], w

    def test_foreign_iceberg_shown_by_its_peak_only(self, env_twin):
        """§4.2 / §9.2 'Best bid / best ask': at the touch the foreign iceberg reads
        2.0 MW, never its 20 MW total, and the participant's 0.3 MW take leaves 1.7."""
        tobs = env_twin["A"]["noop"]["tob_raw"]
        assert any(bool(t.has_ask) and round(float(t.p_ask), 2) == 51.0 and round(float(t.q_ask), 1) == 2.0
                   for t in tobs)
        assert all(round(float(t.q_ask), 1) <= 2.0 for t in tobs if round(float(t.p_ask), 2) == 51.0)
        lob1 = env_twin["A"]["mixed"]["lob_raw"][1]
        assert [(round(p, 2), round(q, 1)) for p, q in lob1[1] if round(p, 2) == 51.0] == [(51.0, 1.7)], lob1

    def test_own_iceberg_full_on_own_orders_reader_peak_on_the_public_readers(self, env_twin):
        """(6'') and F23: the participant's ICE SELL 0.6 MW (peak 0.2) at 50.50 is listed
        at its full 0.6 MW by ``get_agent_orders(book, AGENT_TRADER_ID)``, identically in
        A and B, while ``top_of_book`` and ``get_anonymous_lob`` show only 0.2 MW."""
        from omiecid_simulator.omiecid.order_types import Side
        for w in ("A", "B"):
            ep = env_twin[w]["mixed"]
            rows = ep["view"]["own_orders"]
            ice = [r for r in rows if r[1] == 50.5]
            assert len(ice) == 1 and ice[0][0] == int(Side.SELL) and ice[0][2] == 0.6, (w, rows)
            # step 3 replays the tape's last real order, so ``done`` fires there and the
            # step-4 NON BUY is ignored (plan §4.4: calls after done are ignored)
            assert [bool(d) for d in ep["view"]["done"]] == [False, False, True, True, True, True], w
            assert not bool(ep["fills_raw"][3].submitted), w
            assert [r for r in rows if r[1] != 50.5] == [], (w, rows)
            for k in range(3, len(ep["tob_raw"])):
                t = ep["tob_raw"][k]
                assert round(float(t.p_ask), 2) == 50.5 and round(float(t.q_ask), 1) == 0.2, (w, k, t)
                asks = [(round(p, 2), round(q, 1)) for p, q in ep["lob_raw"][k][1]]
                assert (50.5, 0.2) in asks and (50.5, 0.6) not in asks, (w, k, asks)
        assert env_twin["A"]["mixed"]["view"]["own_orders"] == env_twin["B"]["mixed"]["view"]["own_orders"]

    def test_noop_participant_has_no_own_information(self, env_twin):
        import numpy as np
        for w in ("A", "B", "CTRL"):
            ep = env_twin[w]["noop"]
            assert ep["view"]["own_trades"] == [] and ep["view"]["own_orders"] == [], w
            assert all(not bool(np.asarray(f.submitted)) for f in ep["fills_raw"]), w

    def test_positive_control_top_of_book_and_lob_see_one_visible_tick(self, env_twin):
        """Claim item 3: CTRL's noop top_of_book differs from B's on at least one step,
        and get_anonymous_lob differs on that step as well."""
        b, c = env_twin["B"]["noop"]["view"], env_twin["CTRL"]["noop"]["view"]
        steps = [k for k in range(len(b["tob"])) if b["tob"][k] != c["tob"][k]]
        assert steps, "CTRL top_of_book does not differ from B (control blind)"
        assert any(b["lob"][k] != c["lob"][k] for k in steps)


def _sweep_env_tape(T: float, twin_qty: Optional[float]):
    """Tape (2 orders per step) on which a foreign BUY ICE @ 51.00 (peak 2 MW, total T)
    enters at step 2 and sweeps the 5 MW foreign ask and then the participant's own
    resting SELL 0.1 @ 51.00 (posted at step 1 as ``AgentOrder.create(SELL, 51.0, 0.1)``).
    ``twin_qty`` given: the twin -- the ICE as a plain NON of that size, every agent and
    order id permuted."""
    import numpy as np
    from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side

    twin = twin_qty is not None
    BUY, SELL, ICE = int(Side.BUY), int(Side.SELL), int(ExecRestriction.ICE)
    perm = {11: 14, 12: 13, 13: 12, 14: 11} if twin else {k: k for k in (11, 12, 13, 14)}
    rows = [dict(side=BUY, price_eur=49.0, qty_mw=10.0, agent_id=11),
            dict(side=SELL, price_eur=51.0, qty_mw=5.0, agent_id=14),
            dict(side=BUY, price_eur=51.0, qty_mw=T, agent_id=13, exec_restriction=ICE, peak_qty_mw=2.0),
            dict(side=BUY, price_eur=48.0, qty_mw=1.0, agent_id=12),
            dict(side=SELL, price_eur=53.0, qty_mw=3.0, agent_id=12),
            dict(side=BUY, price_eur=47.0, qty_mw=2.0, agent_id=11),
            dict(side=SELL, price_eur=52.5, qty_mw=2.0, agent_id=11),
            dict(side=BUY, price_eur=48.8, qty_mw=1.0, agent_id=14)]
    tb = TapeBuilder()
    n = len(rows)
    for i, r in enumerate(rows):
        r = dict(r, agent_id=perm[r["agent_id"]])
        if r.get("exec_restriction") == ICE and twin:
            r.pop("exec_restriction")
            r.pop("peak_qty_mw")
            r["qty_mw"] = twin_qty
        tb.add(order_id=(i + 1) if not twin else 500 + n - i, timestamp=60 + 30 * i, **r)
    return np.asarray(tb.build(pad_extra=8))


def _env_twin_qty(T: float) -> float:
    """OWN twin size: q_B = noop fills + noop display; noop entry sweeps M0 = 5 MW, so by
    item (4) the noop display is min(P - u0, T - M0) with u0 = M0 - floor(M0/P) P."""
    P, M0 = 20, 50
    return round((M0 + item4_display(q_int(T), P, M0)["v1"]) / 10, 1)


@pytest.mark.slow
@pytest.mark.jit
def test_probe_session_entry_sweep_of_own_order_rule_implied_difference():
    """slow: one Session over six stacked tapes, one compile (~60-70 s). The
    premise-failed outcome reproduced END TO END in the participant view
    (top_of_book, get_anonymous_lob, FillReport, done, own trades / orders / cash /
    inventory). §4.2.1 Rule 2 (1), (4) as amended 2026-10-20; Annex §4.4; §7.1, §7.2.

    P = 20 ticks; the noop entry sweeps M0 = 50 (k_e = 2, u0 = 10), the participant's
    0.1 MW makes M = 51 (k_e = 2, u = 11). By item (4): T = 20 -> noop shows 1.0 MW, rest
    shows 0.9; T = 5.5 (R < P - u) -> noop 0.5, rest 0.4. The twin q_B = 5 + noop display
    (6.0, 5.5) rests exactly A's display: A and B bit-identical on both policies (gap 0);
    A(20) and A(40) identical (the display never reveals the hidden size).
    DISCRIMINATING: the retired min(P, R) would show 2.0 MW after the entry (noop and
    rest); the r3 twin (5 + min(2, T - 5) = 7.0) then no longer matches the noop world."""
    import jax
    import numpy as np
    from omiecid_simulator.omiecid.omie_session import AgentOrder, Session
    from omiecid_simulator.omiecid.order_types import Side

    assert _env_twin_qty(20.0) == 6.0 and _env_twin_qty(40.0) == 6.0 and _env_twin_qty(5.5) == 5.5
    tapes = np.stack([_sweep_env_tape(20.0, None), _sweep_env_tape(40.0, None),
                      _sweep_env_tape(20.0, _env_twin_qty(20.0)), _sweep_env_tape(5.5, None),
                      _sweep_env_tape(5.5, _env_twin_qty(5.5)), _sweep_env_tape(20.0, 7.0)])
    env = Session({"tapes": tapes, "orders_per_step": 2})
    cb = _reader_book()
    key = jax.random.PRNGKey(0)
    none = AgentOrder.none()
    pol = {"noop": [none] * 5, "rest": [AgentOrder.create(int(Side.SELL), 51.0, 0.1)] + [none] * 4}
    run = {(ti, name): _participant_episode(env, cb, key, orders, tape_idx=ti)
           for ti in range(len(tapes)) for name, orders in pol.items()}
    A20, A40, B20, A55, B55, B20_r3 = range(6)

    def same(x, y):
        return run[x]["view"] == run[y]["view"] and np.array_equal(
            np.asarray(run[x]["final"].agent_trades), np.asarray(run[y]["final"].agent_trades))

    def best_bid(x):
        """(price, qty ticks) of the best bid after reset and every step (top_of_book)."""
        return [(round(float(t.p_bid), 2), q_int(float(t.q_bid))) for t in run[x]["tob_raw"]]

    # A's own display, re-derived: from the entry step (2) on the best bid is 51.00
    # showing item (4)'s v1 -- not the retired full peak 2.0 MW
    for x, want in (((A20, "noop"), 10), ((A20, "rest"), 9), ((A40, "rest"), 9),
                    ((A55, "noop"), 5), ((A55, "rest"), 4)):
        bb = best_bid(x)
        assert bb[2:] == [(51.0, want)] * (len(bb) - 2), (x, bb)
        assert bb[2][1] != 20, x
        lob_bids = [(round(p, 2), q_int(q)) for p, q in run[x]["lob_raw"][2][0]]
        assert lob_bids[0] == (51.0, want), (x, lob_bids)
    # the twin is EXACT on both policies (gap v1 - (q_B - M) = 0): the whole participant view
    for p in pol:
        assert same((A20, p), (B20, p)), p
        assert same((A55, p), (B55, p)), p
    # the display never reveals the hidden size
    assert same((A20, "rest"), (A40, "rest")) and same((A20, "noop"), (A40, "noop"))
    # the participant was swept (premise failed): one own trade, the swept 0.1 MW, as a
    # passive SELL fill reported on the entry step's FillReport
    ep = run[(A20, "rest")]
    assert [(r[0], r[1]) for r in ep["view"]["own_trades"]] == [(51.0, 0.1)], ep["view"]["own_trades"]
    assert ep["view"]["own_orders"] == []
    assert float(np.asarray(ep["fills_raw"][1].passive_qty_signed)) == pytest.approx(-0.1)
    rows = np.asarray(ep["final"].agent_trades).reshape(-1, 6)
    assert int((rows[:, 2] != 0).sum()) == 1, rows
    # the r3 twin construction (sized to the retired full peak) is no longer a twin
    assert not same((A20, "noop"), (B20_r3, "noop"))
    assert best_bid((B20_r3, "noop"))[2:] == [(51.0, 20)] * (len(run[(A20, "noop")]["tob_raw"]) - 2)
