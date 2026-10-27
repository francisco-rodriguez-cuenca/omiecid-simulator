"""K4 audit twin for OMIE-CERT-C11-1 (expiry at gate closure).

Fresh-context, adversarial audit. The expectation is re-derived from the rules
alone and judged against the EMITTED results JSON, whatever its status. It never
opens or executes ``notebooks/certification/C11_gct_expiry.ipynb`` and never reads
``contract_book.py``, ``iceberg_manager.py``, ``order_validator.py`` or
``settlement.py`` (independence rules 3/4).

Rules (all assertions in integer ticks: price tick 0.01 EUR/MWh, qty tick 0.1 MW):

* §3.1: "Orders on a contract are automatically cancelled at GCT if unmatched."
  §4.6: GFS is valid until the contract's GCT and GTD until a time <= GCT, so at
  the gate **every** resting order is removed, both validities alike.
* §5.1: the fate is ``CANCELLED (... GCT reached ...)``; I2 books it as
  ``gate_expiry``. The whole remaining quantity of an order resting at the gate,
  visible + hidden (an iceberg's hidden remainder included, review V3-16), is
  cancelled as ``gate_expiry``; nothing else is cancelled or evicted; afterwards no
  book entry, no hidden quantity, nothing tradeable remains.
* §3.1 + §6.2 (matching is triggered by a new-timestamp order; an expiry sweep is
  not one): the sweep **creates no trade** -- the trade ring has the same row count
  across the expiry call and the pre-gate trades are unchanged (§8.5: firm).
* I2 per order: ``filled + gate_expiry = declared``; the per-draw record carries
  no declared quantity, so I2 is checked here as closure of the record's own
  buckets (every accepted order is either fully filled or expired with exactly its
  pre-gate remaining) and, independently, against tape-declared quantities in the
  slow probe below, which builds its own books with ``TapeBuilder``.
* §6.2 item 2: the pre-gate book is uncrossed (best bid < best ask), so there was
  nothing a matching pass could have printed at the gate anyway.

Per-draw record layout as emitted (``observed`` / ``expected`` JSON strings):
book entries ``[price, visible_qty, oid]``; trades
``[buyer_oid, seller_oid, price, qty, submission]`` (submission 0-based, oid
``s+1``; the gate is submission ``n``).

Results JSON: ``$OMIE_CERT_RESULTS_DIR/C11.json`` else
``notebooks/certification/results/C11.json``; absent -> skip.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
CERT_ID = "C11"

DRAWS = 500                     # "Random scenarios default to 500 draws"
ENVELOPE = {"resting_per_side": 80, "trades": 80, "hops": 160, "concurrent_icebergs": 800}
SLICE_MARGIN = 64
RECORD_KEYS = frozenset({
    "accepted", "cancelled", "evicted", "expired_at_gct", "filled", "final_book",
    "hidden_remaining", "pre_gate_book", "pre_gate_remaining", "rejected", "still_resting",
    "trade_ring_unchanged", "trade_rows_before_after", "trades",
})
SEED_RE = re.compile(r"^seed (\d+) \[aggressors: (\d+)((?:, maker [^,\]]+, (?:partially|fully) filled)*)\]$")
MAKER_RE = re.compile(r"maker (NON|ICE \(standard\)|ICEwPI), (partially|fully) filled")
EJ_RE = re.compile(r"^eager == jitted \(seed (\d+)\)$")
VALIDITY_CLASSES = [f"resting at the gate: {t} {v} {s}"
                    for t in ("NON", "ICE", "ICEwPI") for v in ("GFS", "GTD") for s in ("BUY", "SELL")]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _results_path() -> Path:
    d = os.environ.get("OMIE_CERT_RESULTS_DIR")
    return (Path(d) if d else DEFAULT_RESULTS_DIR) / f"{CERT_ID}.json"


@pytest.fixture(scope="module")
def payload() -> Dict[str, Any]:
    p = _results_path()
    if not p.is_file():
        pytest.skip(f"C11 results JSON absent at {p} (run notebook section 8 first)")
    return json.loads(p.read_text(encoding="utf-8"))


def _reload(obj: Dict[str, Any], tmp_path: Path, name: str) -> Dict[str, Any]:
    f = tmp_path / name
    f.write_text(json.dumps(obj), encoding="utf-8")
    return json.loads(f.read_text(encoding="utf-8"))


def _load(x):
    return json.loads(x) if isinstance(x, str) else x


def seed_records(payload) -> List[Tuple[int, Dict[str, Any], re.Match]]:
    out = []
    for i, a in enumerate(payload["assertions"]):
        if a["name"].startswith("seed "):
            out.append((i, a, SEED_RE.match(a["name"])))
    return out


# ---------------------------------------------------------------------------
# The independent judgement of one per-draw record
# ---------------------------------------------------------------------------
def _book_index(book) -> Dict[int, Tuple[str, int, int]]:
    """oid -> (side, price, visible); duplicates raise via the caller's check."""
    idx = {}
    for side in ("BUY", "SELL"):
        for price, qty, oid in book[side]:
            idx.setdefault(int(oid), []).append((side, int(price), int(qty)))
    return idx


def hidden_at_gate(rec) -> Dict[int, int]:
    """remaining - visible for every order resting at the gate (iceberg hidden part)."""
    vis = {}
    for side in ("BUY", "SELL"):
        for _, q, oid in rec["pre_gate_book"][side]:
            vis[int(oid)] = vis.get(int(oid), 0) + int(q)
    return {int(o): int(r) - vis.get(int(o), 0) for o, r in rec["pre_gate_remaining"].items()}


def judge_record(rec: Dict[str, Any], tag: str = "") -> List[str]:
    """Every rule of the gate sweep, applied to one record. Empty list = conforming."""
    errs: List[str] = []
    e = errs.append
    missing = RECORD_KEYS - set(rec)
    if missing:
        return [f"{tag}per-draw record not in full: missing {sorted(missing)}"]

    accepted = [int(x) for x in rec["accepted"]]
    n = len(accepted)
    if accepted != list(range(1, n + 1)) or n == 0:
        e(f"{tag}accepted {accepted} is not the full tape 1..n (every order valid, §5.4)")
    if rec["rejected"]:
        e(f"{tag}rejected {rec['rejected']} non-empty: every drawn order is valid before GCT")
    if rec["expired_at_gct"] is not True:
        e(f"{tag}expired_at_gct = {rec['expired_at_gct']!r}: the gate was not applied")

    # --- the gate: nothing left ---------------------------------------------------
    fb = rec["final_book"]
    if fb.get("BUY") != [] or fb.get("SELL") != []:
        e(f"{tag}final_book not empty after GCT (§3.1): {fb}")
    left = {k: v for k, v in rec["still_resting"].items() if int(v) != 0}
    if left:
        e(f"{tag}still_resting after GCT (§3.1, GFS and GTD alike): {left}")
    hid = {k: v for k, v in rec["hidden_remaining"].items() if int(v) != 0}
    if hid:
        e(f"{tag}hidden remainder left after GCT (V3-16, must be expired): {hid}")
    ev = {k: v for k, v in rec["evicted"].items() if int(v) != 0}
    if ev:
        e(f"{tag}evicted {ev}: eviction is not a §5.1 fate and must be 0 outside C14")

    # --- no trade from the sweep ----------------------------------------------------
    trades = rec["trades"]
    rows = rec["trade_rows_before_after"]
    if not (isinstance(rows, list) and len(rows) == 2):
        e(f"{tag}trade_rows_before_after malformed: {rows!r}")
    else:
        if rows[0] != rows[1]:
            e(f"{tag}trade count changed across the expiry call {rows}: the sweep created trades")
        if rows[0] != len(trades) or rows[1] != len(trades):
            e(f"{tag}trade ring rows {rows} != {len(trades)} pre-gate trades listed")
    if rec["trade_ring_unchanged"] is not True:
        e(f"{tag}trade ring changed across the expiry call (pre-gate trades not firm, §8.5)")

    filled_from_trades: Dict[int, int] = {}
    for t in trades:
        buyer, seller, price, qty, sub = (int(x) for x in t)
        if qty <= 0 or price <= 0 or buyer == seller:
            e(f"{tag}degenerate trade {t}")
        if not (0 <= sub < n):
            e(f"{tag}trade {t} written by submission {sub}, not a pre-gate submission "
              f"(the gate is {n}): a trade created by the expiry")
        elif (sub + 1) not in (buyer, seller):
            e(f"{tag}trade {t}: submission {sub}'s own order {sub + 1} is not a party "
              f"(§6.2: matching is triggered by the new-timestamp order)")
        else:
            maker = seller if buyer == sub + 1 else buyer
            if maker > sub + 1:
                e(f"{tag}trade {t}: counterparty {maker} was submitted after the aggressor")
        for o in (buyer, seller):
            if o not in accepted:
                e(f"{tag}trade {t}: party {o} not accepted")
            filled_from_trades[o] = filled_from_trades.get(o, 0) + qty
    filled = {int(k): int(v) for k, v in rec["filled"].items() if int(v) != 0}
    if filled != filled_from_trades:
        e(f"{tag}filled {filled} != the trades' per-order quantities {filled_from_trades}")

    # --- the pre-gate book --------------------------------------------------------
    remaining = {int(k): int(v) for k, v in rec["pre_gate_remaining"].items()}
    if any(v <= 0 for v in remaining.values()):
        e(f"{tag}pre_gate_remaining lists a non-positive remainder: {remaining}")
    idx = _book_index(rec["pre_gate_book"])
    for oid, entries in idx.items():
        if len(entries) != 1:
            e(f"{tag}order {oid} appears {len(entries)} times in the pre-gate book")
        if oid not in remaining:
            e(f"{tag}order {oid} rests pre-gate but has no recorded remainder")
        for side, price, q in entries:
            if q <= 0:
                e(f"{tag}pre-gate book entry {oid} with non-positive visible {q}")
    for oid, h in hidden_at_gate(rec).items():
        if h < 0:
            e(f"{tag}order {oid}: visible exceeds its remaining at the gate")
    bids = [p for p, _, _ in rec["pre_gate_book"]["BUY"]]
    asks = [p for p, _, _ in rec["pre_gate_book"]["SELL"]]
    if bids and asks and max(bids) >= min(asks):
        e(f"{tag}pre-gate book crossed (best bid {max(bids)} >= best ask {min(asks)}, §6.2 item 2)")

    # --- the fates: gate_expiry of exactly the remaining, nothing else --------------
    cancelled = {int(k): v for k, v in rec["cancelled"].items()}
    if set(cancelled) != set(remaining):
        e(f"{tag}gate-expired orders {sorted(cancelled)} != orders resting at the gate "
          f"{sorted(remaining)}")
    for oid, fates in cancelled.items():
        want = [["gate_expiry", remaining.get(oid)]]
        if [[str(c), int(q)] for c, q in fates] != want:
            e(f"{tag}order {oid}: fates {fates} != {want} (whole remaining, visible + hidden, "
              f"cancelled as gate_expiry and nothing else)")

    # --- I2 closure: each accepted order is filled-or-expired, exactly -------------
    for oid in accepted:
        f = filled_from_trades.get(oid, 0)
        g = sum(int(q) for c, q in cancelled.get(oid, []) if c == "gate_expiry")
        if f + g <= 0:
            e(f"{tag}I2: order {oid} accepted but neither filled nor expired (vanished)")
        if g != remaining.get(oid, 0):
            e(f"{tag}I2: order {oid} gate_expiry {g} != pre-gate remaining {remaining.get(oid, 0)}")

    # --- non-vacuity of the gate sweep for this draw --------------------------------
    if not remaining:
        e(f"{tag}nothing rested at the gate: the draw does not exercise the sweep")
    return errs


def judge_label(rec, m: re.Match, tag: str = "") -> List[str]:
    """The seed label's class claims, re-derived from the record."""
    errs = []
    if m is None:
        return [f"{tag}unparsable seed label"]
    k = int(m.group(2))
    makers = MAKER_RE.findall(m.group(3))
    aggressors, maker_oids = set(), set()
    for t in rec["trades"]:
        buyer, seller, _, _, sub = (int(x) for x in t)
        aggressors.add(sub + 1)
        maker_oids.add(seller if buyer == sub + 1 else buyer)
    if len(aggressors) != k:
        errs.append(f"{tag}label says {k} aggressors, trades show {len(aggressors)}")
    # The label lists maker classes, de-duplicated (two makers of one class are listed once).
    if len(set(makers)) != len(makers) or len(makers) > len(maker_oids):
        errs.append(f"{tag}label lists maker classes {makers}, trades show {len(maker_oids)} makers")
    remaining = {int(o) for o in rec["pre_gate_remaining"]}
    how_label = {how for _, how in makers}
    how_seen = {"partially" if o in remaining else "fully" for o in maker_oids}
    if how_label != how_seen:
        errs.append(f"{tag}label fill classes {sorted(how_label)} != record {sorted(how_seen)}")
    return errs


def judge_draws(payload) -> List[str]:
    """Every per-draw record against the rules; record status consistent with them."""
    errs: List[str] = []
    recs = seed_records(payload)
    valid = payload["draws"]["valid"]
    if len(recs) != valid:
        errs.append(f"per-seed records {len(recs)} != valid draws {valid}")
    seeds = [a["seed"] for _, a, _ in recs]
    if len(set(seeds)) != len(seeds):
        errs.append("duplicate seed records")
    for _, a, m in recs:
        tag = f"seed {a['seed']}: "
        if m is None or int(m.group(1)) != a["seed"]:
            errs.append(f"{tag}label {a['name']!r} does not parse or names another seed")
        try:
            obs, exp = _load(a["observed"]), _load(a["expected"])
        except (TypeError, ValueError) as exc:
            errs.append(f"{tag}per-draw evidence not JSON: {exc!r}")
            continue
        if not isinstance(obs, dict) or not isinstance(exp, dict):
            errs.append(f"{tag}per-draw evidence not in full (observed/expected not records)")
            continue
        bad = judge_record(obs, tag)
        errs.extend(bad)
        errs.extend(judge_record(exp, tag + "K2 expected: "))
        if not bad:
            errs.extend(judge_label(obs, m, tag))
        if bad and a["status"] != "fail":
            errs.append(f"{tag}recorded status {a['status']!r} on a rules-violating record")
        if not bad and a["status"] != "pass":
            errs.append(f"{tag}recorded status {a['status']!r} on a rules-conforming record")
    return errs


def judge_eager_jit(payload) -> List[str]:
    errs = []
    ej = [a for a in payload["assertions"] if a["kind"] == "eager_jit"]
    if not ej:
        return ["no eager-vs-jitted evidence"]
    by_seed = {a["seed"]: _load(a["observed"]) for _, a, _ in seed_records(payload)}
    with_trade = with_hidden = False
    for a in ej:
        m = EJ_RE.match(a["name"])
        if not m or int(m.group(1)) != a["seed"]:
            errs.append(f"unparsable eager-jit assertion {a['name']!r}")
            continue
        if a["status"] != "pass" or a["observed"] is not True or a["expected"] is not True:
            errs.append(f"eager != jitted on seed {a['seed']}")
        rec = by_seed.get(a["seed"])
        if rec is None:
            errs.append(f"eager-jit seed {a['seed']} is not a valid draw")
            continue
        with_trade |= bool(rec.get("trades"))
        with_hidden |= any(h > 0 for h in hidden_at_gate(rec).values())
    if not with_trade:
        errs.append("eager-jit evidence covers no draw with a pre-gate trade")
    if not with_hidden:
        errs.append("eager-jit evidence covers no draw with a hidden remainder at the gate")
    if payload.get("eager_jit_identical") is not True:
        errs.append(f"eager_jit_identical = {payload.get('eager_jit_identical')!r}")
    return errs


def judge_battery(payload) -> List[str]:
    errs = []
    rows = payload["invariants"]
    ids = [r["check_id"] for r in rows]
    if ids != ["RB"] + [f"I{i}" for i in range(1, 10)]:
        return [f"battery incomplete or out of order: {ids}"]
    n = payload["draws"]["valid"]
    no_fill = sum(1 for _, a, _ in seed_records(payload) if not _load(a["observed"])["trades"])
    for r in rows:
        cid = r["check_id"]
        if r["n_pass"] + r["n_fail"] + r["n_not_applicable"] != n:
            errs.append(f"{cid}: did not run on every valid draw")
        if r["n_fail"] != 0 or r["failing_seeds"]:
            errs.append(f"{cid}: {r['n_fail']} failing draws {r['failing_seeds']}")
        if cid in ("I1", "I4", "I5", "I6"):
            if r["n_not_applicable"] > no_fill:
                errs.append(f"{cid}: not_applicable on {r['n_not_applicable']} draws, only {no_fill} "
                            f"have no fills")
        elif r["n_not_applicable"] != 0:
            errs.append(f"{cid}: not_applicable on {r['n_not_applicable']} draws (allowed only for "
                        f"fill-dependent checks)")
    return errs


def judge_status(payload) -> List[str]:
    errs = list(validate_results(payload))
    if payload["failure_kinds"] != derive_failure_kinds(payload):
        errs.append("failure_kinds not derived from payload")
    any_fail = any(a["status"] == "fail" for a in payload["assertions"])
    if any_fail != (payload["status"] != "PASS"):
        errs.append(f"status {payload['status']!r} vs any failing assertion = {any_fail}")
    return errs


def judge_all(payload) -> List[str]:
    return judge_status(payload) + judge_draws(payload) + judge_eager_jit(payload) + \
        judge_battery(payload)


# ---------------------------------------------------------------------------
# Tests on the emitted payload
# ---------------------------------------------------------------------------
class TestPayloadShape:
    def test_schema_and_semantic_validation(self, payload):
        assert validate_results(payload) == []

    def test_identity_fields(self, payload):
        assert payload["cert_id"] == "C11"
        assert payload["notebook"] == "C11_gct_expiry.ipynb"
        assert payload["oracle_kind"] == "exact-parametric"
        assert payload["environment"]["backend"] == "cpu"

    def test_status_and_failure_kinds_consistent(self, payload):
        assert judge_status(payload) == []

    def test_draw_counts_and_seeds(self, payload):
        d, s = payload["draws"], payload["seeds"]
        assert d["requested"] == DRAWS and d["valid"] == DRAWS
        assert s["count"] == d["valid"] + d["discarded"]
        assert d["discard_rate"] == pytest.approx(d["discarded"] / (d["valid"] + d["discarded"]))
        recs = seed_records(payload)
        assert len(recs) == d["valid"]
        seeds = sorted(a["seed"] for _, a, _ in recs)
        assert len(set(seeds)) == len(seeds)
        assert s["first"] <= seeds[0] and seeds[-1] <= s["last"]
        assert s["last"] - s["first"] + 1 == s["count"]
        assert sorted(s["failing"]) == sorted(a["seed"] for _, a, _ in recs if a["status"] == "fail")


class TestGateSweepAgainstRules:
    def test_every_draw_conforms_to_the_gate_rules(self, payload):
        errs = judge_draws(payload)
        assert errs == [], "\n".join(errs[:30])

    def test_both_validities_of_every_type_and_side_rest_at_the_gate(self, payload):
        """GFS and GTD alike (§4.6): every validity x type x side class is drawn and swept.

        The per-draw record does not carry validity; the class counts are the
        notebook's aggregate. Cross-check them for internal consistency.
        """
        cov = [a for a in payload["assertions"] if a["name"].startswith("dataset: the valid draws cover")]
        assert len(cov) == 1 and cov[0]["status"] == "pass"
        obs = cov[0]["observed"]
        for cls in VALIDITY_CLASSES:
            assert obs.get(cls, 0) >= 1, cls
        n = payload["draws"]["valid"]
        assert sum(obs[f"aggressors: {k}"] for k in (0, 1, 2)) == n
        # The aggressor counts agree with the per-draw records.
        recs = seed_records(payload)
        for k in (0, 1, 2):
            assert obs[f"aggressors: {k}"] == sum(1 for _, _, m in recs if int(m.group(2)) == k)
        assert obs["an iceberg at the slice margin, ceil(total/peak) = 64"] >= 1

    def test_hidden_remainders_rest_at_the_gate_and_are_expired(self, payload):
        """V3-16: icebergs with positive hidden remainder rest at the gate on both sides."""
        sides_with_hidden = set()
        n_draws_with_hidden = 0
        for _, a, _ in seed_records(payload):
            rec = _load(a["observed"])
            h = hidden_at_gate(rec)
            pos = [o for o, v in h.items() if v > 0]
            n_draws_with_hidden += bool(pos)
            for side in ("BUY", "SELL"):
                if any(int(oid) in pos for _, _, oid in rec["pre_gate_book"][side]):
                    sides_with_hidden.add(side)
            for o in pos:
                assert rec["cancelled"][str(o)] == [["gate_expiry", rec["pre_gate_remaining"][str(o)]]]
                assert int(rec["hidden_remaining"].get(str(o), 0)) == 0
        assert n_draws_with_hidden == payload["draws"]["valid"], n_draws_with_hidden
        assert sides_with_hidden == {"BUY", "SELL"}

    def test_draws_with_pre_gate_trades_exist_and_stay_firm(self, payload):
        with_trades = [a for _, a, _ in seed_records(payload) if _load(a["observed"])["trades"]]
        assert len(with_trades) >= 1
        for a in with_trades:
            obs = _load(a["observed"])
            assert obs["trade_ring_unchanged"] is True
            assert obs["trade_rows_before_after"] == [len(obs["trades"])] * 2

    def test_partially_filled_orders_expire_their_residual(self, payload):
        """A maker hit before the gate expires exactly its residual (filled + expiry = declared)."""
        seen = 0
        for _, a, _ in seed_records(payload):
            rec = _load(a["observed"])
            for o, f in rec["filled"].items():
                if int(f) > 0 and o in rec["pre_gate_remaining"]:
                    seen += 1
                    assert rec["cancelled"][o] == [["gate_expiry", rec["pre_gate_remaining"][o]]]
        assert seen >= 1

    def test_no_iceberg_slot_state_is_asserted(self, payload):
        """OQ-20: slot state is reported, never asserted."""
        for _, a, _ in seed_records(payload):
            assert "live_iceberg_slots" not in a["expected"]
            assert a["detail"] == "" or a["detail"].startswith("reported only")


class TestEvidence:
    def test_eager_vs_jitted_present_and_identical(self, payload):
        assert judge_eager_jit(payload) == []

    def test_full_battery_on_every_valid_draw(self, payload):
        assert judge_battery(payload) == []

    def test_i2_applies_and_passes_on_every_draw(self, payload):
        row = next(r for r in payload["invariants"] if r["check_id"] == "I2")
        assert row["n_pass"] == payload["draws"]["valid"] and row["status"] == "pass"

    def test_capacity_margins_and_peaks(self, payload):
        cap = payload["capacity"]
        assert cap["envelope"] == ENVELOPE
        recs = [_load(a["observed"]) for _, a, _ in seed_records(payload)]
        max_trades = max(len(r["trades"]) for r in recs)
        max_side = max(len(r["pre_gate_book"][s]) for r in recs for s in ("BUY", "SELL"))
        max_hidden = max(sum(1 for v in hidden_at_gate(r).values() if v > 0) for r in recs)
        pk = cap["peaks"]
        assert pk["trades"] == max_trades <= ENVELOPE["trades"]
        assert max_side <= pk["resting_per_side"] <= ENVELOPE["resting_per_side"]
        assert max_hidden <= pk["concurrent_icebergs"] <= ENVELOPE["concurrent_icebergs"]
        sl = [a for a in payload["assertions"] if a["name"].startswith("largest ceil(qty/peak)")]
        assert len(sl) == 1 and sl[0]["status"] == "pass" and sl[0]["observed"] <= SLICE_MARGIN
        hb = [a for a in payload["assertions"] if a["name"].startswith("hop bound <= 160")]
        assert len(hb) == 1 and hb[0]["status"] == "pass" and max(hb[0]["observed"]) <= ENVELOPE["hops"]
        fl = [a for a in payload["assertions"] if a["name"].startswith("no capacity flag")]
        assert len(fl) == 1 and fl[0]["status"] == "pass"

    def test_quotes_present_ok_and_load_bearing_passages_byte_exact(self, payload):
        qs = payload["spec"]["quotes"]
        assert qs and all(q["ok"] for q in qs)
        digests = {q["digest"] for q in qs}
        need = [
            SpecQuote(source="market-spec.md", section="§3.1",
                      text="**Per-contract gate closure (GCT):** 1 hour before the start of the delivery "
                           "period. Orders on a contract are automatically cancelled at GCT if unmatched."),
            SpecQuote(source="market-spec.md", section="§5.1",
                      text="CANCELLED (IOC/FOK unmatched; GTD expired; GCT reached; halt/hibernation)"),
            SpecQuote(source="market-spec.md", section="§8.5",
                      text="The continuous market has no automatic position close-out at GCT. Open orders "
                           "are cancelled at GCT but already-matched trades are firm."),
            SpecQuote(source="market-spec.md", section="§4.6",
                      text="| GTD | Valid until agent-specified time ≤ GCT | No |"),
        ]
        for q in need:
            assert check_quote(q).ok, q.section          # still in the source today
            assert quote_digest(q) in digests, q.section  # and quoted by the notebook
        secs = {(q["source"], q["section"]) for q in qs}
        # §6.2 grounding of "no new trades" (the heading itself is not quotable body text).
        assert any(s == "market-spec.md" and sec.startswith("§6.2") for s, sec in secs)
        assert ("scenario-ladder-review.md", "V3-16") in secs
        assert ("open-questions.md", "OQ-20") in secs
        for c in payload["citations"]:
            assert c["checked"] is False

    def test_assertion_units_never_euros(self, payload):
        for a in payload["assertions"]:
            assert a["unit"] in ("bool", "count", "other", "ticks", "qty_ticks", "price_ticks"), a["name"]

    def test_whole_payload_judgement_is_clean(self, payload):
        errs = judge_all(payload)
        assert errs == [], "\n".join(errs[:30])


# ---------------------------------------------------------------------------
# Deliberately broken copies (each must be rejected)
# ---------------------------------------------------------------------------
def _pick(payload, pred):
    for i, a, m in seed_records(payload):
        rec = _load(a["observed"])
        if pred(rec):
            return i, rec
    raise AssertionError("no draw satisfies the predicate")


def _put(p, i, rec, both=False):
    p["assertions"][i]["observed"] = json.dumps(rec, sort_keys=True)
    if both:
        p["assertions"][i]["expected"] = json.dumps(rec, sort_keys=True)


def _only(p, i):
    q = dict(p)
    q["assertions"] = [p["assertions"][i]]
    return [e for e in judge_draws(q) if "per-seed records" not in e]


def _fully_visible(rec):
    h = hidden_at_gate(rec)
    return [int(oid) for s in ("BUY", "SELL") for _, _, oid in rec["pre_gate_book"][s] if h[int(oid)] == 0]


class TestBrokenCopiesRejected:
    def test_gtd_order_surviving_the_gate_is_rejected(self, payload, tmp_path):
        """A (GTD) order left resting after GCT, its fate dropped."""
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: bool(_fully_visible(r)))
        oid = _fully_visible(rec)[0]
        side, price, q = _book_index(rec["pre_gate_book"])[oid][0]
        rec["final_book"][side] = [[price, q, oid]]
        rec["still_resting"] = {str(oid): q}
        del rec["cancelled"][str(oid)]
        _put(p, i, rec, both=True)
        p = _reload(p, tmp_path, "C11_gtd_survives.json")
        errs = _only(p, i)
        assert any("final_book not empty" in e for e in errs), errs
        assert any("still_resting after GCT" in e for e in errs), errs
        assert any("recorded status 'pass' on a rules-violating record" in e for e in errs), errs

    def test_survivor_hidden_from_final_book_but_unexpired_is_rejected(self, payload, tmp_path):
        """Book cleared but the order's quantity is never booked as expired."""
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: bool(_fully_visible(r)))
        oid = _fully_visible(rec)[0]
        del rec["cancelled"][str(oid)]
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_unexpired.json")
        errs = _only(p, i)
        assert any(f"I2: order {oid} gate_expiry 0" in e for e in errs), errs

    def test_iceberg_hidden_remainder_left_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: any(v > 0 for v in hidden_at_gate(r).values()))
        oid, h = next((o, v) for o, v in hidden_at_gate(rec).items() if v > 0)
        rec["hidden_remaining"] = {str(oid): h}
        rec["cancelled"][str(oid)] = [["gate_expiry", rec["pre_gate_remaining"][str(oid)] - h]]
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_hidden_left.json")
        errs = _only(p, i)
        assert any("hidden remainder left after GCT" in e for e in errs), errs
        assert any(f"order {oid}: fates" in e for e in errs), errs

    def test_hidden_remainder_left_but_fully_booked_as_expired_is_rejected(self, payload, tmp_path):
        """Double counting: hidden still live although the fate claims the whole remaining."""
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: any(v > 0 for v in hidden_at_gate(r).values()))
        oid, h = next((o, v) for o, v in hidden_at_gate(rec).items() if v > 0)
        rec["hidden_remaining"] = {str(oid): h}
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_hidden_double.json")
        assert any("hidden remainder left after GCT" in e for e in _only(p, i))

    def test_trade_created_by_the_expiry_is_rejected(self, payload, tmp_path):
        """The sweep matches best bid against best ask on its way out."""
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: r["pre_gate_book"]["BUY"] and r["pre_gate_book"]["SELL"])
        n = len(rec["accepted"])
        bp, bq, boid = max(rec["pre_gate_book"]["BUY"], key=lambda e: e[0])
        ap, aq, aoid = min(rec["pre_gate_book"]["SELL"], key=lambda e: e[0])
        q = min(bq, aq)
        k = len(rec["trades"])
        rec["trades"].append([boid, aoid, ap, q, n])
        rec["trade_rows_before_after"] = [k, k + 1]
        rec["trade_ring_unchanged"] = False
        for o in (boid, aoid):
            rec["filled"][str(o)] = int(rec["filled"].get(str(o), 0)) + q
            rec["cancelled"][str(o)] = [["gate_expiry", rec["pre_gate_remaining"][str(o)] - q]]
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_expiry_trade.json")
        errs = _only(p, i)
        assert any("trade count changed across the expiry call" in e for e in errs), errs
        assert any("not a pre-gate submission" in e for e in errs), errs
        assert any("trade ring changed" in e for e in errs), errs

    def test_expiry_trade_hidden_from_the_trade_list_is_rejected(self, payload, tmp_path):
        """Ring grows by one row at the gate while the record lists only pre-gate trades."""
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: True)
        k = len(rec["trades"])
        rec["trade_rows_before_after"] = [k, k + 1]
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_ring_grows.json")
        assert any("trade count changed" in e for e in _only(p, i))

    def test_expiry_trade_with_flags_lying_is_rejected(self, payload, tmp_path):
        """A trade listed at the gate submission while the ring flags claim no change."""
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: r["pre_gate_book"]["BUY"] and r["pre_gate_book"]["SELL"])
        n = len(rec["accepted"])
        b = max(rec["pre_gate_book"]["BUY"], key=lambda e: e[0])
        a = min(rec["pre_gate_book"]["SELL"], key=lambda e: e[0])
        rec["trades"].append([b[2], a[2], a[0], 1, n])
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_trade_flag_lie.json")
        errs = _only(p, i)
        assert any("not a pre-gate submission" in e for e in errs), errs
        assert any("pre-gate trades listed" in e for e in errs), errs

    def test_wrong_cancellation_cause_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: bool(r["cancelled"]))
        oid = next(iter(rec["cancelled"]))
        rec["cancelled"][oid] = [["cancellation", rec["cancelled"][oid][0][1]]]
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_wrong_cause.json")
        assert any(f"order {oid}: fates" in e for e in _only(p, i))

    def test_eviction_at_the_gate_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: bool(r["cancelled"]))
        oid = next(iter(rec["cancelled"]))
        q = rec["cancelled"][oid][0][1]
        rec["evicted"] = {oid: q}
        del rec["cancelled"][oid]
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_evicted.json")
        errs = _only(p, i)
        assert any("evicted" in e for e in errs), errs

    def test_order_vanishing_before_the_gate_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: bool(_fully_visible(r)))
        oid = _fully_visible(rec)[0]
        for s in ("BUY", "SELL"):
            rec["pre_gate_book"][s] = [e for e in rec["pre_gate_book"][s] if e[2] != oid]
        del rec["pre_gate_remaining"][str(oid)]
        del rec["cancelled"][str(oid)]
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_vanish.json")
        assert any("vanished" in e for e in _only(p, i))

    def test_pre_gate_trade_rewritten_is_rejected(self, payload, tmp_path):
        """Firmness (§8.5): a pre-gate trade altered by the sweep."""
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: bool(r["trades"]))
        rec["trades"][0][3] += 1
        rec["trade_ring_unchanged"] = False
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_trade_rewritten.json")
        errs = _only(p, i)
        assert any("trade ring changed" in e for e in errs), errs
        assert any("filled" in e for e in errs), errs

    def test_truncated_per_draw_record_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, rec = _pick(p, lambda r: True)
        del rec["pre_gate_remaining"]
        _put(p, i, rec)
        p = _reload(p, tmp_path, "C11_truncated.json")
        assert any("not in full" in e for e in _only(p, i))

    def test_missing_draw_record_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, _ = _pick(p, lambda r: True)
        del p["assertions"][i]
        p = _reload(p, tmp_path, "C11_missing_draw.json")
        assert any("per-seed records" in e for e in judge_draws(p))

    def test_status_lie_on_a_conforming_record_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        i, _ = _pick(p, lambda r: True)
        p["assertions"][i]["status"] = "fail"
        p = _reload(p, tmp_path, "C11_status_lie.json")
        assert any("recorded status 'fail'" in e for e in _only(p, i))

    def test_absent_eager_jit_evidence_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        p["assertions"] = [a for a in p["assertions"] if a["kind"] != "eager_jit"]
        p = _reload(p, tmp_path, "C11_no_ej.json")
        assert judge_eager_jit(p) == ["no eager-vs-jitted evidence"]

    def test_eager_jit_disagreement_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        a = next(a for a in p["assertions"] if a["kind"] == "eager_jit")
        a["observed"] = False
        p = _reload(p, tmp_path, "C11_ej_false.json")
        assert any("eager != jitted" in e for e in judge_eager_jit(p))

    def test_eager_jit_flag_false_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        p["eager_jit_identical"] = False
        p = _reload(p, tmp_path, "C11_ej_flag.json")
        assert any("eager_jit_identical" in e for e in judge_eager_jit(p))

    def test_not_applicable_on_a_draw_with_fills_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        row = next(r for r in p["invariants"] if r["check_id"] == "I5")
        row["n_not_applicable"] += row["n_pass"]
        row["n_pass"] = 0
        row["status"] = "not_applicable"
        p = _reload(p, tmp_path, "C11_na_inflated.json")
        assert any("I5: not_applicable" in e for e in judge_battery(p))

    def test_battery_row_missing_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        p["invariants"] = [r for r in p["invariants"] if r["check_id"] != "I7"]
        p = _reload(p, tmp_path, "C11_no_i7.json")
        assert any("battery incomplete" in e for e in judge_battery(p))

    def test_status_pass_with_a_failure_is_rejected(self, payload, tmp_path):
        p = copy.deepcopy(payload)
        p["assertions"][0]["status"] = "fail"
        p["status"], p["failure_kinds"], p["failure_explanation"] = "PASS", [], None
        p = _reload(p, tmp_path, "C11_status_pass.json")
        assert judge_status(p) != []


# ---------------------------------------------------------------------------
# Independent probe: my own books, swept at the gate (slow)
# ---------------------------------------------------------------------------
def _probe_tapes():
    from omiecid_simulator.jaxomiecid.omie_tape_synth import TapeBuilder
    from omiecid_simulator.omiecid.order_types import ExecRestriction, Side, Validity

    B, S = int(Side.BUY), int(Side.SELL)
    NON, ICE = int(ExecRestriction.NON), int(ExecRestriction.ICE)
    GFS, GTD = int(Validity.GFS), int(Validity.GTD)
    base = 1_005_000
    tapes = []

    # (a) mixed validities and types on both sides, no trade.
    tb = TapeBuilder()
    tb.add(side=B, price_int=base - 100, qty_int=50, validity=GFS)
    tb.add(side=B, price_int=base - 200, qty_int=70, validity=GTD, expiry_time=495)
    tb.add(side=S, price_int=base + 100, qty_int=40, validity=GTD, expiry_time=495)
    tb.add(side=S, price_int=base + 300, qty_int=90, validity=GFS)
    tb.add(side=B, price_int=base - 300, qty_int=200, exec_restriction=ICE, peak_int=30, validity=GTD,
           expiry_time=495)
    tb.add(side=S, price_int=base + 500, qty_int=250, exec_restriction=ICE, peak_int=20, ppd_int=10,
           validity=GFS)
    tapes.append(("no_trade", tb.build(), False))

    # (b) an aggressor partially consumes a resting GTD iceberg, then the gate.
    tb = TapeBuilder()
    tb.add(side=S, price_int=base, qty_int=300, exec_restriction=ICE, peak_int=40, validity=GTD,
           expiry_time=495)
    tb.add(side=S, price_int=base + 50, qty_int=60, validity=GFS)
    tb.add(side=B, price_int=base - 50, qty_int=80, exec_restriction=ICE, peak_int=10, ppd_int=-5,
           validity=GTD, expiry_time=495)
    tb.add(side=B, price_int=base, qty_int=100, validity=GFS)      # takes 100 of the iceberg
    tapes.append(("iceberg_hit", tb.build(), True))

    # (c) a NON maker partially filled; aggressor residual rests (GTD) and is swept too.
    tb = TapeBuilder()
    tb.add(side=B, price_int=base, qty_int=50, validity=GTD, expiry_time=495)
    tb.add(side=S, price_int=base + 10, qty_int=30, validity=GFS)
    tb.add(side=S, price_int=base, qty_int=20, validity=GFS)       # 20 against the bid
    tb.add(side=B, price_int=base + 10, qty_int=45, validity=GTD, expiry_time=495)  # 30 fill, 15 rest
    tapes.append(("residuals", tb.build(), True))
    return tapes


def _declared(tape) -> Dict[int, int]:
    return {int(r[4]): int(r[3]) for r in tape if int(r[4]) >= 1}


@pytest.mark.slow
@pytest.mark.parametrize("case", ["no_trade", "iceberg_hit", "residuals"])
def test_independent_probe_gate_sweep(case):
    """slow: compiles ContractBook submit/expire on CPU, eager and jitted, 3 small books."""
    from tests.omiecid.reference.omie_invariants import build_record, compare_records, run_battery

    name, tape, expects_trade = next(t for t in _probe_tapes() if t[0] == case)
    declared = _declared(tape)
    before = build_record(tape, expire_at_gct=False)
    after = build_record(tape, expire_at_gct=True)
    eager = build_record(tape, expire_at_gct=True, jit=False)
    assert compare_records(after, eager) == (), case

    assert not after.rejections, case
    assert set(after.accepted) == set(declared)
    # Something rests at the gate, hidden quantity included where icebergs rest.
    pre = before.final_snapshot
    resting = {o: pre.visible_qty(o, a.side) + pre.hidden_qty(o) for o, a in before.accepted.items()}
    assert any(v > 0 for v in resting.values())
    # The gate: nothing left, visible or hidden.
    fin = after.final_snapshot
    assert fin.resting(0) == () and fin.resting(1) == (), case
    assert all(fin.hidden_qty(o) == 0 for o in declared), case
    assert all(v == 0 for v in after.still_resting.values())
    assert all(v == 0 for v in after.hidden_remaining.values())
    # No trade from the sweep: identical fills and trade-ring row count.
    key = lambda r: [(f.buyer_oid, f.seller_oid, f.price_int, f.qty_int, f.submission) for f in r.fills]
    assert key(after) == key(before)
    assert fin.trade_rows_used() == pre.trade_rows_used() == len(after.fills)
    assert bool(after.fills) == expects_trade
    # I2 against the tape's declared quantities: filled + gate_expiry = declared.
    for oid, q in declared.items():
        g = sum(c.qty_int for c in after.cancelled_or_expired.get(oid, ()) if c.cause == "gate_expiry")
        others = [c for c in after.cancelled_or_expired.get(oid, ()) if c.cause != "gate_expiry"]
        assert not others, (case, oid, others)
        assert after.filled[oid] + g == q, (case, oid, after.filled[oid], g, q)
        assert g == resting[oid], (case, oid)
    rep = run_battery(after, expects_trade=expects_trade)
    st = rep.statuses()
    assert st["I2"] == "pass" and st["RB"] == "pass" and st["I3"] == "pass", st
    assert all(v != "fail" for v in st.values()), st
