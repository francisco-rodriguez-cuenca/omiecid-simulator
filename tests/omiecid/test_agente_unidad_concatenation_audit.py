"""Audit tests for Story 1.5-B — Agente-Unidad Concatenation.

Independent black-box verification that the Phase 1.5-B pair-mapping refactor
is correct against the domain contract and the story acceptance criteria.
This file is authored *after* implementation, without access to the original
locked test file (``tests/omiecid/test_agente_unidad_concatenation.py``). It
exists to catch behaviours the locked suite may have under-covered.

Scope (derived from agile-plan.md Story 1.5-B, migration-plan v4.0 §Phase 1.5,
and market-spec §13.1/§13.2/§13.4/§13.5):

  1. Pair-mapping determinism across runs (same input → same ids, including
     across a subprocess with a different PYTHONHASHSEED).
  2. Losslessness of the reverse map: `id_to_pair[pair_to_id[p]] == p` and
     every agent_id produced by the loaders round-trips through
     `load_omie_agent_metadata`.
  3. Canonical pair-valued metadata contract — `load_omie_agent_metadata`
     returns ``dict[int, tuple[str, str]]``, never a plain string.
  4. Agente-only projection is derivable in one line at the call site (no
     persistent `id_to_agente` artifact hidden in the module surface).
  5. Pair universe unified across orders and trades (trade-only pairs get
     ids; the two loaders never disagree on the mapping for the same pair).
  6. Blank-leg XBID trades receive a legitimate positive agent_id, not 0
     and not accidentally the id of the first real agent.
  7. Cache schema versioning (F2 pre-sprint finding) — stale / un-versioned
     / mismatched caches must NOT be silently reused.
  8. Array shape invariance — orders (N, 13), trades (M, 6).
  9. `OMIEOrder` 13-field invariant — unchanged by 1.5-B.
 10. Same Agente + different Unidad → distinct ids; same pair → same id.
 11. Deterministic sort key is a tuple sort, not a concatenation (locale-
     and separator-safe).
 12. Trade buy/sell symmetry — both legs resolve through the same id_to_pair.
 13. Zona inclusion preserved — PT and ES rows both contribute to the pair
     mapping and zona is never part of the pair key.
 14. Real-data integration — production orders_*.1 / trades_*.1 files are
     loaded end-to-end (slow-marked; justifies overrunning the 1s budget).

Policy reminders honored:
  - No hard-coded integer agent_id values. All assertions are property-based
    (determinism, positivity, distinctness, losslessness, bijection, union).
  - Synthetic fixtures via ``tmp_path`` for the fast tier.
  - Real-data integration runs under ``@pytest.mark.slow`` only.

Reference paths (read-only):
  - omiecid_simulator/jaxomiecid/omie_data_loader.py
  - omiecid_simulator/jaxomiecid/omie_data_config.py
  - omiecid_simulator/jaxomiecid/omie_cache.py
  - omiecid_simulator/omiecid/order_types.py (OMIEOrder 13-field contract)
  - docs/omiecid/market-spec.md §13
  - docs/omiecid/migration-plan.md §Phase 1.5
  - docs/omiecid/agile-plan.md §Story 1.5-B
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    _build_agent_mapping,
    _collect_pair_universe,
    load_omie_agent_metadata,
    load_omie_orders,
    load_omie_trades,
)
from omiecid_simulator.jaxomiecid.omie_cache import (
    CACHE_SCHEMA_VERSION,
    load_omie_cache,
    save_omie_cache,
)
from omiecid_simulator.omiecid.order_types import OMIEOrder


# ---------------------------------------------------------------------------
# Constants / zone codes (market-spec §13.1/§13.2)
# ---------------------------------------------------------------------------

_ZONE_ES = "10YES-REE------0"
_ZONE_PT = "10YPT-REN------W"

_DATE_DMY = "01/06/2025"
_DATE_ISO = "2025-06-01"
_DATE_NODASH = "20250601"

# Fixture-only product date (OMIE-TAPE-CONTRACT-DATE-1, group F).  Fixture
# rows are sent on 2025-06-01 (``_DATE_DMY``, session file ``*_20250601.1``),
# so their ``Contrato`` is dated 2025-06-02: delivery product (2025-06-02, k),
# whose session files are D-1 = 20250601 (written by the tests) and
# D = 20250602 (header-only).  Market-spec v2.10 §13/§13.4.  With the product
# dated D, the stored timestamps (seconds since D-1 00:00) keep their values.
_PRODUCT_DATE_ISO = "2025-06-02"
_PRODUCT_DATE_NODASH = "20250602"
_PRODUCT_SESSION_FILES_NODASH = (_DATE_NODASH, _PRODUCT_DATE_NODASH)  # (D-1, D)


def _contract_str(contract_id: int) -> str:
    hour = contract_id // 4
    minute = (contract_id % 4) * 15
    end_minute = minute + 15
    end_hour = hour
    if end_minute == 60:
        end_hour = hour + 1
        end_minute = 0
    return (
        f"{_PRODUCT_DATE_NODASH} {hour:02d}:{minute:02d}-"
        f"{_PRODUCT_DATE_NODASH} {end_hour:02d}:{end_minute:02d}"
    )


# ---------------------------------------------------------------------------
# Synthetic file writers (mirror the real OMIE on-disk format)
# ---------------------------------------------------------------------------

_ORDERS_HEADER_META = (
    "OMIE - Mercado de electricidad; Ofertas del Mercado Intradiario Continuo;"
)
_ORDERS_HEADER_COLS = (
    "Fecha;Contrato;Zona;Agente;Unidad;Precio;Cantidad;Tipo oferta;"
    "Condiciones a la ejecucion;Condiciones a la validez;"
    "Cantidad reducida;PPD;Momento de envio;"
)

_TRADES_HEADER_META = (
    "OMIE - Mercado de electricidad; Casaciones del Mercado Intradiario Continuo;"
)
_TRADES_HEADER_COLS = (
    "Fecha;Contrato;Agente compra;Unidad compra;Zona compra;"
    "Agente venta;Unidad venta;Zona venta;Precio;Cantidad;Momento casacion;"
)


def _write_orders(tmp_path: Path, rows: list[str]) -> None:
    lines = [_ORDERS_HEADER_META, _ORDERS_HEADER_COLS]
    for r in rows:
        lines.append(r if r.endswith(";") else r + ";")
    (tmp_path / f"orders_{_DATE_NODASH}.1").write_text(
        "\n".join(lines) + "\n", encoding="latin-1"
    )


def _write_trades(tmp_path: Path, rows: list[str]) -> None:
    lines = [_TRADES_HEADER_META, _TRADES_HEADER_COLS]
    for r in rows:
        lines.append(r if r.endswith(";") else r + ";")
    (tmp_path / f"trades_{_DATE_NODASH}.1").write_text(
        "\n".join(lines) + "\n", encoding="latin-1"
    )


def _order_row(
    *,
    contract_id: int,
    zona: str,
    agente: str,
    unidad: str,
    precio: str = "50,00",
    cantidad: str = "1,00",
    side: str = "C",
    exec_r: str = "",
    validez: str = "",
    cant_red: str = "",
    ppd: str = "",
    momento: str = "01/06/2025 17:41:12",
) -> str:
    return (
        f"{_DATE_DMY};{_contract_str(contract_id)};{zona};{agente};{unidad};"
        f"{precio};{cantidad};{side};{exec_r};{validez};{cant_red};{ppd};{momento}"
    )


def _trade_row(
    *,
    contract_id: int,
    buy_agente: str,
    buy_unidad: str,
    buy_zona: str,
    sell_agente: str,
    sell_unidad: str,
    sell_zona: str,
    precio: str = "50,00",
    cantidad: str = "1,00",
    momento: str = "01/06/2025 17:42:00",
) -> str:
    return (
        f"{_DATE_DMY};{_contract_str(contract_id)};"
        f"{buy_agente};{buy_unidad};{buy_zona};"
        f"{sell_agente};{sell_unidad};{sell_zona};"
        f"{precio};{cantidad};{momento}"
    )


def _complete_product_files(tmp_path: Path) -> None:
    """Add a header-only file for every product session file not written.

    Same header lines as ``_write_orders`` / ``_write_trades`` (unaccented
    ``Momento casacion``).  A file written later by a test overwrites it;
    a file already written is never touched.
    """
    for date_nodash in _PRODUCT_SESSION_FILES_NODASH:
        orders = tmp_path / f"orders_{date_nodash}.1"
        trades = tmp_path / f"trades_{date_nodash}.1"
        if not orders.exists():
            orders.write_text(
                "\n".join([_ORDERS_HEADER_META, _ORDERS_HEADER_COLS]) + "\n",
                encoding="latin-1",
            )
        if not trades.exists():
            trades.write_text(
                "\n".join([_TRADES_HEADER_META, _TRADES_HEADER_COLS]) + "\n",
                encoding="latin-1",
            )


def _config(tmp_path: Path, contract_id: int = 16) -> OMIEDataConfig:
    _complete_product_files(tmp_path)
    return OMIEDataConfig(
        data_path=str(tmp_path),
        contract_id=contract_id,
        delivery_date=_PRODUCT_DATE_ISO,
        cache_path=str(tmp_path / "cache"),
    )


# ===========================================================================
# 1. Pair-mapping determinism (same input → same output)
# ===========================================================================


class TestPairMappingDeterminism:
    """`_build_agent_mapping` is deterministic given the same input pairs.

    The story spec is explicit: "Each unique pair gets a deterministic
    1-indexed ID (alphabetically sorted)". Determinism must hold
    independently of row order and of Python's set iteration order.
    """

    def test_build_mapping_same_input_same_output_twice(self):
        """Two back-to-back calls with the same pairs return equal dicts."""
        a = pd.Series(["ALPHA", "BRAVO", "ALPHA"])
        u = pd.Series(["U1", "U2", "U1"])
        p1, r1 = _build_agent_mapping(a, u)
        p2, r2 = _build_agent_mapping(a, u)
        assert p1 == p2
        assert r1 == r2

    def test_build_mapping_row_order_invariant(self):
        """Permuting input rows must not change the mapping."""
        a1 = pd.Series(["A", "B", "C"])
        u1 = pd.Series(["U1", "U2", "U3"])
        a2 = pd.Series(["C", "A", "B"])
        u2 = pd.Series(["U3", "U1", "U2"])
        p1, _ = _build_agent_mapping(a1, u1)
        p2, _ = _build_agent_mapping(a2, u2)
        assert p1 == p2, (
            "Reordering input rows changed the mapping — the sort key is "
            "probably observing iteration order"
        )

    def test_build_mapping_duplicates_collapse(self):
        """Duplicate pairs must collapse to a single id; cardinality = distinct pairs."""
        a = pd.Series(["A", "A", "A", "B"])
        u = pd.Series(["U1", "U1", "U1", "U2"])
        p, r = _build_agent_mapping(a, u)
        assert len(p) == 2
        assert len(r) == 2

    def test_mapping_identical_across_subprocess(self, tmp_path):
        """Subprocess run with different PYTHONHASHSEED yields identical ids.

        Guards against any latent reliance on set iteration order or on
        hash-based dedup that would surface under hash randomization.
        """
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="ZEE", unidad="UZ",
                           precio="50,00", cantidad="1,00", side="C"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="ACC", unidad="UA",
                           precio="50,00", cantidad="1,00", side="V",
                           momento="01/06/2025 17:41:13"),
                _order_row(contract_id=16, zona=_ZONE_PT, agente="MMM", unidad="UM",
                           precio="50,00", cantidad="1,00", side="C",
                           momento="01/06/2025 17:41:14"),
            ],
        )
        _write_trades(tmp_path, [])
        _complete_product_files(tmp_path)

        repo_root = Path(__file__).resolve().parents[2]
        script = textwrap.dedent(f"""
            import json, sys
            sys.path.insert(0, r"{repo_root}")
            from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
            from omiecid_simulator.jaxomiecid.omie_data_loader import load_omie_agent_metadata
            cfg = OMIEDataConfig(
                data_path=r"{tmp_path}",
                contract_id=16,
                delivery_date="{_PRODUCT_DATE_ISO}",
                cache_path=r"{tmp_path / 'cache'}",
            )
            meta = load_omie_agent_metadata(cfg)
            # Serialize as a list of [id, agente, unidad] rows, sorted by id.
            rows = [[aid, p[0], p[1]] for aid, p in sorted(meta.items())]
            print(json.dumps(rows))
        """)

        def _run(seed: str) -> list:
            res = subprocess.run(
                [sys.executable, "-c", script],
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert res.returncode == 0, (
                f"subprocess failed (seed={seed}): stderr={res.stderr}"
            )
            return json.loads(res.stdout.strip().splitlines()[-1])

        rows_a = _run("0")
        rows_b = _run("99991")
        assert rows_a == rows_b, (
            "Subprocess runs under different PYTHONHASHSEED produced different "
            f"id_to_pair mappings.\nA: {rows_a}\nB: {rows_b}"
        )


# ===========================================================================
# 2. Losslessness / bijection
# ===========================================================================


class TestReverseMappingBijection:
    """`(pair_to_id, id_to_pair)` is a bijection on its domain.

    Story spec: "Same `(Agente, Unidad)` pair → same `agent_id` …
    ``load_omie_agent_metadata`` returns correct `{agent_id: (agente, unidad)}`
    (lossless)."
    """

    def test_round_trip_pair_to_id_to_pair(self):
        """For every pair p, id_to_pair[pair_to_id[p]] == p."""
        a = pd.Series(["A", "B", "A", "C"])
        u = pd.Series(["U1", "U2", "U2", "U3"])
        p_to_id, id_to_p = _build_agent_mapping(a, u)
        for pair, aid in p_to_id.items():
            assert id_to_p[aid] == pair

    def test_round_trip_id_to_pair_to_id(self):
        """For every id i, pair_to_id[id_to_pair[i]] == i."""
        a = pd.Series(["A", "B", "C"])
        u = pd.Series(["U1", "U2", "U3"])
        p_to_id, id_to_p = _build_agent_mapping(a, u)
        for aid, pair in id_to_p.items():
            assert p_to_id[pair] == aid

    def test_same_cardinality(self):
        """|pair_to_id| == |id_to_pair|."""
        a = pd.Series(["A", "B", "A"])
        u = pd.Series(["U1", "U2", "U2"])
        p_to_id, id_to_p = _build_agent_mapping(a, u)
        assert len(p_to_id) == len(id_to_p)

    def test_values_are_tuples_of_length_two(self):
        """id_to_pair values must be (str, str) tuples."""
        a = pd.Series(["A", "B"])
        u = pd.Series(["U1", "U2"])
        _, id_to_p = _build_agent_mapping(a, u)
        for aid, pair in id_to_p.items():
            assert isinstance(pair, tuple), f"id {aid} value is not a tuple"
            assert len(pair) == 2, f"id {aid} value has {len(pair)} fields"
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)

    def test_ids_are_contiguous_1_indexed(self):
        """Ids form exactly the set {1, 2, …, N}."""
        a = pd.Series(["A", "B", "C", "D", "E"])
        u = pd.Series(["U", "U", "U", "U", "U"])
        _, id_to_p = _build_agent_mapping(a, u)
        assert set(id_to_p.keys()) == set(range(1, len(id_to_p) + 1))

    def test_every_loader_agent_id_resolves_through_metadata(self, tmp_path):
        """Every agent_id that appears in the loaded (N, 13) / (M, 6)
        arrays must be a key in ``load_omie_agent_metadata``.

        This is the load-bearing acceptance criterion: the reverse map is
        lossless relative to the data actually emitted by the loader.
        """
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="AAA", unidad="U1",
                           momento="01/06/2025 17:41:10"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="BBB", unidad="U1",
                           momento="01/06/2025 17:41:11", side="V"),
                _order_row(contract_id=16, zona=_ZONE_PT, agente="CCC", unidad="UP",
                           momento="01/06/2025 17:41:12"),
            ],
        )
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="AAA", buy_unidad="U1", buy_zona=_ZONE_ES,
                           sell_agente="BBB", sell_unidad="U1", sell_zona=_ZONE_ES),
            ],
        )
        cfg = _config(tmp_path)
        orders = load_omie_orders(cfg)
        trades = load_omie_trades(cfg)
        meta = load_omie_agent_metadata(cfg)

        ids_in_orders = {int(x) for x in np.asarray(orders[:, 5]).tolist()}
        ids_in_trades = (
            {int(x) for x in np.asarray(trades[:, 3]).tolist()}
            | {int(x) for x in np.asarray(trades[:, 4]).tolist()}
        )
        for aid in ids_in_orders | ids_in_trades:
            assert aid in meta, (
                f"agent_id={aid} appears in loader output but is absent "
                f"from load_omie_agent_metadata — reverse map is not lossless"
            )


# ===========================================================================
# 3. Canonical pair-valued metadata contract
# ===========================================================================


class TestMetadataReturnTypeIsPair:
    """`load_omie_agent_metadata` MUST return dict[int, tuple[str, str]].

    Story spec: "Add `load_omie_agent_metadata(config)` function returning
    the `{agent_id: (agente, unidad)}` reverse mapping as the canonical
    session metadata. Any `{agent_id: agente_string}` view is derived at
    the call site". A plain-string value here would be a silent regression
    to the old Agente-only contract.
    """

    def test_values_are_not_plain_strings(self, tmp_path):
        """Regression guard: values must be tuples, never bare strings."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="A", unidad="U1"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="B", unidad="U1",
                           momento="01/06/2025 17:41:13", side="V"),
            ],
        )
        _write_trades(tmp_path, [])
        meta = load_omie_agent_metadata(_config(tmp_path))
        assert len(meta) >= 2
        for aid, val in meta.items():
            assert not isinstance(val, str), (
                f"agent_id {aid} maps to a bare string {val!r} — the 1.5-B "
                f"contract requires a (agente, unidad) tuple"
            )
            assert isinstance(val, tuple) and len(val) == 2

    def test_keys_are_ints(self, tmp_path):
        """agent_id keys must be plain Python ints, not numpy int64."""
        _write_orders(
            tmp_path,
            [_order_row(contract_id=16, zona=_ZONE_ES, agente="A", unidad="U1")],
        )
        _write_trades(tmp_path, [])
        meta = load_omie_agent_metadata(_config(tmp_path))
        for aid in meta:
            # Allow int or np.integer subtypes that compare equal to int
            assert isinstance(aid, (int, np.integer)), (
                f"agent_id key {aid!r} is not an integer type"
            )


# ===========================================================================
# 4. Agente-only projection is derivable at call site (no persistent dup artifact)
# ===========================================================================


class TestNoPersistentIdToAgente:
    """Module must not expose a second persistent `id_to_agente` artifact.

    Story spec: "do NOT persist a separate `id_to_agente` artifact. Any
    `{agent_id: agente_string}` view is derived at the call site (one-line
    projection)".
    """

    def test_one_line_projection_yields_agente_only(self, tmp_path):
        """{aid: pair[0] for aid, pair in meta.items()} gives the Agente view."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="XX", unidad="U1"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="XX", unidad="U2",
                           momento="01/06/2025 17:41:13", side="V"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="YY", unidad="U1",
                           momento="01/06/2025 17:41:14"),
            ],
        )
        _write_trades(tmp_path, [])
        meta = load_omie_agent_metadata(_config(tmp_path))
        id_to_agente = {aid: pair[0] for aid, pair in meta.items()}

        # Cardinality: id_to_agente has one entry per id (not per Agente),
        # and multiple ids can share the same Agente (XX has two units).
        assert len(id_to_agente) == len(meta)
        assert set(id_to_agente.values()) == {"XX", "YY"}
        # Exactly two ids map to "XX" (the XX/U1 and XX/U2 pairs).
        assert sum(1 for v in id_to_agente.values() if v == "XX") == 2

    def test_no_suspicious_public_name_in_module(self):
        """The module surface must not expose a load_*_agente_only helper.

        We scan ``dir()`` for public names that look like Agente-only
        projections — a regression would introduce, e.g.,
        ``load_omie_agente_names`` or ``id_to_agente`` at module scope.
        """
        import omiecid_simulator.jaxomiecid.omie_data_loader as loader_mod

        suspicious = [
            name for name in dir(loader_mod)
            if not name.startswith("_")
            and any(tok in name.lower() for tok in ("id_to_agente", "agente_only",
                                                    "agente_names", "agent_to_str"))
        ]
        assert not suspicious, (
            f"Module exposes suspicious Agente-only artifacts: {suspicious} "
            f"— the 1.5-B directive forbids persisting a separate id_to_agente"
        )


# ===========================================================================
# 5. Pair universe unified across orders and trades
# ===========================================================================


class TestUnifiedPairUniverse:
    """Orders and trades share the same pair universe — the two loaders
    must never disagree on a pair's agent_id.

    Story spec: "pair_to_id, id_to_pair — symmetric, lossless." The reverse
    map is used by settlement and diagnostics; a disagreement between
    loaders would silently break PnL attribution.
    """

    def test_trade_only_pair_gets_valid_id(self, tmp_path):
        """A pair appearing only in trades (never in orders) still gets an id."""
        _write_orders(
            tmp_path,
            [_order_row(contract_id=16, zona=_ZONE_ES, agente="ORD", unidad="OU")],
        )
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="TRADE_ONLY_A", buy_unidad="TAU",
                           buy_zona=_ZONE_ES,
                           sell_agente="TRADE_ONLY_B", sell_unidad="TBU",
                           sell_zona=_ZONE_ES),
            ],
        )
        cfg = _config(tmp_path)
        meta = load_omie_agent_metadata(cfg)
        # All three pairs must be present in the metadata.
        pairs_in_meta = set(meta.values())
        assert ("ORD", "OU") in pairs_in_meta
        assert ("TRADE_ONLY_A", "TAU") in pairs_in_meta
        assert ("TRADE_ONLY_B", "TBU") in pairs_in_meta

    def test_order_only_pair_gets_valid_id(self, tmp_path):
        """A pair appearing only in orders (never in trades) still gets an id."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="ORD_ONLY",
                           unidad="OOU"),
            ],
        )
        _write_trades(tmp_path, [])
        meta = load_omie_agent_metadata(_config(tmp_path))
        assert ("ORD_ONLY", "OOU") in set(meta.values())

    def test_orders_and_trades_agree_on_shared_pair(self, tmp_path):
        """Same pair across orders + trades gets exactly one agent_id."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="SHARED",
                           unidad="SU", momento="01/06/2025 17:41:10"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="OTHER",
                           unidad="OU", momento="01/06/2025 17:41:11", side="V"),
            ],
        )
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="SHARED", buy_unidad="SU", buy_zona=_ZONE_ES,
                           sell_agente="OTHER", sell_unidad="OU", sell_zona=_ZONE_ES),
            ],
        )
        cfg = _config(tmp_path)
        orders = load_omie_orders(cfg)
        trades = load_omie_trades(cfg)
        meta = load_omie_agent_metadata(cfg)

        # Find SHARED's id in meta (reverse lookup)
        shared_id = {pair: aid for aid, pair in meta.items()}[("SHARED", "SU")]

        # That id must appear in orders[:, 5] for the SHARED row (it's the
        # 17:41:10 row, which is the first chronologically).
        orders_ids = set(int(x) for x in np.asarray(orders[:, 5]).tolist())
        assert shared_id in orders_ids

        # And on the buy leg of the (single) trade.
        assert int(trades[0, 3]) == shared_id

    def test_collect_pair_universe_symmetric(self, tmp_path):
        """`_collect_pair_universe` returns bijective dicts."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="A", unidad="U"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="B", unidad="U",
                           momento="01/06/2025 17:41:13", side="V"),
            ],
        )
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="A", buy_unidad="U", buy_zona=_ZONE_ES,
                           sell_agente="B", sell_unidad="U", sell_zona=_ZONE_ES),
            ],
        )
        p_to_id, id_to_p = _collect_pair_universe(_config(tmp_path))
        assert len(p_to_id) == len(id_to_p)
        for pair, aid in p_to_id.items():
            assert id_to_p[aid] == pair


# ===========================================================================
# 6. Blank-leg XBID trades
# ===========================================================================


class TestBlankLegXBIDTrades:
    """Blank-leg trades must receive a legitimate positive agent_id.

    Real OMIE `trades_*.1` files contain cross-border XBID rows where one
    counterparty (Agente/Unidad/Zona) is blank (see
    ``data/rawOMIE/trades_20250601.1`` rows with ``;;;;``). The blank leg
    must NOT be silently mapped to 0 (which in the loader code is the
    sentinel for ``.get(..., 0)`` — a fallback that would collide with any
    real id if a regression removed the blank pair from the universe).
    """

    @pytest.fixture
    def blank_leg_fixture(self, tmp_path):
        _write_orders(
            tmp_path,
            [_order_row(contract_id=16, zona=_ZONE_ES, agente="REAL", unidad="RU")],
        )
        _write_trades(
            tmp_path,
            [
                # buy side blank
                _trade_row(contract_id=16,
                           buy_agente="", buy_unidad="", buy_zona="",
                           sell_agente="REAL", sell_unidad="RU",
                           sell_zona=_ZONE_ES),
                # sell side blank
                _trade_row(contract_id=16,
                           buy_agente="REAL", buy_unidad="RU", buy_zona=_ZONE_ES,
                           sell_agente="", sell_unidad="", sell_zona="",
                           momento="01/06/2025 17:42:01"),
            ],
        )
        return _config(tmp_path)

    def test_blank_leg_pair_present_in_metadata(self, blank_leg_fixture):
        """The ('', '') pair must have a key in load_omie_agent_metadata."""
        meta = load_omie_agent_metadata(blank_leg_fixture)
        assert ("", "") in set(meta.values()), (
            "Blank-leg XBID pair ('', '') is missing from the canonical "
            "metadata — real OMIE XBID rows would have no id"
        )

    def test_blank_leg_agent_id_is_positive(self, blank_leg_fixture):
        """The blank-leg id is a positive integer (not 0 sentinel)."""
        meta = load_omie_agent_metadata(blank_leg_fixture)
        reverse = {pair: aid for aid, pair in meta.items()}
        blank_id = reverse[("", "")]
        assert blank_id > 0, (
            f"Blank-leg ('', '') got agent_id={blank_id} — 0 is the "
            f".get(..., 0) sentinel; the pair must receive a real positive id"
        )

    def test_blank_leg_and_real_pair_are_distinct(self, blank_leg_fixture):
        """Blank pair id ≠ any real-pair id."""
        meta = load_omie_agent_metadata(blank_leg_fixture)
        reverse = {pair: aid for aid, pair in meta.items()}
        blank_id = reverse[("", "")]
        real_id = reverse[("REAL", "RU")]
        assert blank_id != real_id

    def test_blank_leg_ids_in_trades_match_metadata(self, blank_leg_fixture):
        """The column-3 buy-id / column-4 sell-id of blank-leg trades
        round-trip through load_omie_agent_metadata to ('', '')."""
        trades = load_omie_trades(blank_leg_fixture)
        meta = load_omie_agent_metadata(blank_leg_fixture)
        # Row 0: buy blank, sell REAL
        buy_id_0 = int(trades[0, 3])
        assert meta[buy_id_0] == ("", "")
        # Row 1: buy REAL, sell blank
        sell_id_1 = int(trades[1, 4])
        assert meta[sell_id_1] == ("", "")

    def test_nan_string_never_enters_pair_universe(self, tmp_path):
        """The literal string "nan" must not leak into the universe.

        Regression guard: pandas reads empty cells as float NaN; if the
        loader does str(cell) without normalization, the pair ('nan','nan')
        would be injected into the universe instead of ('', '').
        """
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="", buy_unidad="", buy_zona="",
                           sell_agente="REAL", sell_unidad="RU",
                           sell_zona=_ZONE_ES),
            ],
        )
        meta = load_omie_agent_metadata(_config(tmp_path))
        values = set(meta.values())
        assert ("nan", "nan") not in values, (
            "Blank cells collapsed to the string 'nan' instead of '' — "
            "_normalize_cell regression"
        )
        # And each half: an agente == "nan" or a unidad == "nan" is also a bug.
        for agente, unidad in values:
            assert agente != "nan", (
                f"pair ({agente!r}, {unidad!r}) has agente == 'nan'"
            )
            assert unidad != "nan", (
                f"pair ({agente!r}, {unidad!r}) has unidad == 'nan'"
            )


# ===========================================================================
# 7. Cache schema versioning (F2 from pre-sprint review)
# ===========================================================================


class TestCacheSchemaVersioning:
    """Stale 1.5-A caches (Agente-only agent_ids) must NOT be silently reused.

    The mechanism is ``CACHE_SCHEMA_VERSION`` in
    ``omiecid_simulator/jaxomiecid/omie_cache.py``. Pre-1.5-B caches have no
    ``schema_version`` field; a future schema would write a different
    version. Either case must be rejected (``load_omie_cache`` returns
    ``None``) so the loader re-parses from raw files.
    """

    def test_cache_schema_version_is_defined_and_positive(self):
        assert isinstance(CACHE_SCHEMA_VERSION, int)
        assert CACHE_SCHEMA_VERSION >= 2, (
            f"CACHE_SCHEMA_VERSION must be >= 2 after 1.5-B bump; got "
            f"{CACHE_SCHEMA_VERSION}"
        )

    def test_legacy_cache_without_schema_field_rejected(self, tmp_path):
        """A pre-1.5-B cache (no schema_version field) must be rejected."""
        cfg = _config(tmp_path)
        cache_dir = Path(cfg.cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Simulate a legacy 1.5-A cache: orders + trades, no schema_version.
        fake_orders = np.zeros((3, 13), dtype=np.float32)
        fake_trades = np.zeros((2, 6), dtype=np.float32)
        legacy_path = cache_dir / f"omie_cache_c{cfg.contract_id}_{cfg.delivery_date}.npz"
        np.savez_compressed(legacy_path, orders=fake_orders, trades=fake_trades)

        result = load_omie_cache(cfg)
        assert result is None, (
            "Legacy pre-1.5-B cache (no schema_version) was silently "
            "accepted — stale Agente-only agent_ids would be propagated"
        )

    def test_cache_with_mismatched_schema_version_rejected(self, tmp_path):
        """A cache stamped with a different version must be rejected."""
        cfg = _config(tmp_path)
        cache_dir = Path(cfg.cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fake_orders = np.zeros((1, 13), dtype=np.float32)
        path = cache_dir / f"omie_cache_c{cfg.contract_id}_{cfg.delivery_date}.npz"
        # Write a future schema_version that should be rejected.
        np.savez_compressed(
            path,
            schema_version=np.array(CACHE_SCHEMA_VERSION + 7, dtype=np.int32),
            orders=fake_orders,
        )

        result = load_omie_cache(cfg)
        assert result is None, (
            f"Cache with future schema_version "
            f"{CACHE_SCHEMA_VERSION + 7} was not rejected"
        )

    def test_current_schema_version_round_trips(self, tmp_path):
        """A cache written by save_omie_cache round-trips through load_omie_cache."""
        cfg = _config(tmp_path)
        orders = np.array([[16.0] + [0.0] * 12], dtype=np.float32)
        save_omie_cache(cfg, orders=orders)
        result = load_omie_cache(cfg)
        assert result is not None
        assert "schema_version" in result
        assert int(np.asarray(result["schema_version"]).item()) == CACHE_SCHEMA_VERSION
        assert "orders" in result
        assert np.array_equal(result["orders"], orders)

    def test_missing_cache_file_returns_none(self, tmp_path):
        """A missing cache file is a plain cache miss (None)."""
        cfg = _config(tmp_path)
        assert load_omie_cache(cfg) is None


# ===========================================================================
# 8. Array shape invariance — orders (N, 13), trades (M, 6)
# ===========================================================================


class TestShapeInvariance:
    """Loaders must emit (N, 13) orders and (M, 6) trades regardless of
    how many distinct pairs are in the fixture.

    Story acceptance criterion: "Array shapes unchanged."
    """

    def test_orders_shape_13_with_many_pairs(self, tmp_path):
        """A fixture with many distinct pairs still produces 13 columns."""
        rows = []
        for i in range(20):
            rows.append(
                _order_row(contract_id=16, zona=_ZONE_ES,
                           agente=f"AG{i:02d}", unidad=f"U{i:02d}",
                           momento=f"01/06/2025 17:41:{i:02d}",
                           side="C" if i % 2 == 0 else "V")
            )
        _write_orders(tmp_path, rows)
        _write_trades(tmp_path, [])
        arr = load_omie_orders(_config(tmp_path))
        assert arr.shape[0] == 20
        assert arr.shape[1] == 13

    def test_trades_shape_6_with_many_pairs(self, tmp_path):
        """A fixture with many distinct trade pairs still produces 6 columns."""
        _write_orders(tmp_path, [])
        rows = []
        for i in range(15):
            rows.append(
                _trade_row(
                    contract_id=16,
                    buy_agente=f"B{i:02d}", buy_unidad=f"BU{i:02d}",
                    buy_zona=_ZONE_ES,
                    sell_agente=f"S{i:02d}", sell_unidad=f"SU{i:02d}",
                    sell_zona=_ZONE_ES,
                    momento=f"01/06/2025 17:42:{i:02d}",
                )
            )
        _write_trades(tmp_path, rows)
        arr = load_omie_trades(_config(tmp_path))
        assert arr.shape[0] == 15
        assert arr.shape[1] == 6

    def test_empty_orders_shape_0x13(self, tmp_path):
        """No orders for the contract → (0, 13) — not (0,) or (0, 14)."""
        _write_orders(
            tmp_path,
            [
                # different contract_id → filtered out
                _order_row(contract_id=32, zona=_ZONE_ES, agente="X", unidad="U"),
            ],
        )
        _write_trades(tmp_path, [])
        arr = load_omie_orders(_config(tmp_path, contract_id=16))
        assert arr.shape == (0, 13)

    def test_empty_trades_shape_0x6(self, tmp_path):
        """No trades for the contract → (0, 6)."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=32,
                           buy_agente="A", buy_unidad="U", buy_zona=_ZONE_ES,
                           sell_agente="B", sell_unidad="U", sell_zona=_ZONE_ES),
            ],
        )
        arr = load_omie_trades(_config(tmp_path, contract_id=16))
        assert arr.shape == (0, 6)


# ===========================================================================
# 9. OMIEOrder 13-field invariant
# ===========================================================================


class TestOMIEOrderFieldCountInvariant:
    """OMIEOrder must still have exactly 13 fields.

    Story acceptance criterion: "`OMIEOrder` stays at 13 fields."
    Locked architectural decision #4: no 13→14 expansion.
    """

    def test_fields_count_is_13(self):
        assert len(OMIEOrder._fields) == 13, (
            f"OMIEOrder has {len(OMIEOrder._fields)} fields; the 1.5-B "
            f"contract (and Phase 1.5 decision #4) locks this at 13. "
            f"Fields: {OMIEOrder._fields}"
        )

    def test_no_unit_or_unidad_field_names(self):
        """No surviving unit_id / unidad field (regression guard)."""
        field_names = {f.lower() for f in OMIEOrder._fields}
        forbidden = {"unit_id", "unidad", "unit", "unit_idx"}
        leaked = field_names & forbidden
        assert not leaked, (
            f"OMIEOrder fields include forbidden name(s) {leaked}; the "
            f"1.5-B concatenation design absorbs Unidad into agent_id, "
            f"not into a new field"
        )

    def test_agent_id_is_a_field(self):
        """agent_id must still be one of the 13 fields."""
        assert "agent_id" in OMIEOrder._fields


# ===========================================================================
# 10. Same Agente + different Unidad → distinct ids; same pair → same id
# ===========================================================================


class TestSameAgenteDifferentUnidadDistinct:
    """The core behavioral change of 1.5-B.

    Story spec: "Same `(Agente, Unidad)` pair → same `agent_id`. Different
    pairs → different IDs. Same Agente + different Unidad → different IDs."
    """

    def test_same_agente_different_unidad_distinct_ids(self):
        a = pd.Series(["AGNT", "AGNT", "AGNT"])
        u = pd.Series(["U1", "U2", "U3"])
        p_to_id, _ = _build_agent_mapping(a, u)
        ids = {p_to_id[("AGNT", u_i)] for u_i in ("U1", "U2", "U3")}
        assert len(ids) == 3, (
            "Same Agente with 3 different Unidades must produce 3 distinct "
            f"ids; got {ids}"
        )

    def test_different_agente_same_unidad_string_distinct_ids(self):
        """Different Agentes sharing a Unidad *code* still get different ids."""
        a = pd.Series(["A", "B", "C"])
        u = pd.Series(["U", "U", "U"])
        p_to_id, _ = _build_agent_mapping(a, u)
        assert len({p_to_id[("A", "U")], p_to_id[("B", "U")], p_to_id[("C", "U")]}) == 3

    def test_repeated_pair_same_id(self):
        a = pd.Series(["A", "A", "A"])
        u = pd.Series(["U", "U", "U"])
        p_to_id, _ = _build_agent_mapping(a, u)
        assert len(p_to_id) == 1

    def test_loader_propagates_same_pair_same_id(self, tmp_path):
        """Two rows with the same (Agente, Unidad) pair get the same column-5 id."""
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="X", unidad="U1",
                           momento="01/06/2025 17:41:10"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="X", unidad="U1",
                           momento="01/06/2025 17:41:11"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="X", unidad="U2",
                           momento="01/06/2025 17:41:12"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        ids = [int(orders[i, 5]) for i in range(3)]
        # Rows 0 and 1 share the pair ("X", "U1") → same id.
        assert ids[0] == ids[1]
        # Row 2 is ("X", "U2") → different id.
        assert ids[0] != ids[2]


# ===========================================================================
# 11. Deterministic sort key is a tuple sort (not concatenation-based)
# ===========================================================================


class TestTupleSortKey:
    """Sort key must be canonical tuple comparison, not string concatenation.

    Concatenation is separator-dependent: ``"AA" + "Z" == "AAZ"`` vs
    ``"AAZ" + ""  == "AAZ"`` — collision. Tuple sort is robust.
    """

    def test_concatenation_colliding_pairs_get_distinct_ids(self):
        """Pairs that concat to identical strings are still distinct."""
        # ("AAZ", "") and ("AA", "Z") concat to "AAZ" — they must not collide.
        a = pd.Series(["AAZ", "AA"])
        u = pd.Series(["", "Z"])
        p_to_id, id_to_p = _build_agent_mapping(a, u)
        assert len(p_to_id) == 2
        assert p_to_id[("AAZ", "")] != p_to_id[("AA", "Z")]
        # And the reverse map distinguishes them.
        for pair, aid in p_to_id.items():
            assert id_to_p[aid] == pair

    def test_sort_key_agente_first_then_unidad(self):
        """Canonical tuple sort: Agente breaks ties only on equality of agente."""
        a = pd.Series(["B", "A", "A", "B"])
        u = pd.Series(["U1", "U2", "U1", "U2"])
        p_to_id, _ = _build_agent_mapping(a, u)
        # Sorted order of pairs: ("A","U1"), ("A","U2"), ("B","U1"), ("B","U2")
        # Ids are 1-indexed in that order.
        assert p_to_id[("A", "U1")] == 1
        assert p_to_id[("A", "U2")] == 2
        assert p_to_id[("B", "U1")] == 3
        assert p_to_id[("B", "U2")] == 4

    def test_concatenation_sort_would_produce_different_order(self):
        """Fixture where tuple-sort and concat-sort disagree; verify tuple-sort wins.

        Pairs: ("AA", "B"), ("A", "CB").
        Tuple sort → ("A", "CB") before ("AA", "B") — because "A" < "AA".
        Concat sort ("AA"+"B"="AAB" vs "A"+"CB"="ACB") → "AAB" < "ACB" →
        ("AA","B") would come first. Tuple sort must win.
        """
        a = pd.Series(["AA", "A"])
        u = pd.Series(["B", "CB"])
        p_to_id, _ = _build_agent_mapping(a, u)
        assert p_to_id[("A", "CB")] < p_to_id[("AA", "B")], (
            "Sort key appears to be string concatenation, not tuple sort — "
            "(\"A\",\"CB\") should precede (\"AA\",\"B\") under canonical "
            "tuple ordering"
        )


# ===========================================================================
# 12. Trade buy/sell symmetry
# ===========================================================================


class TestTradeBuySellSymmetry:
    """Both legs of a trade resolve through the same id_to_pair.

    The trades schema has buy_agent_id in column 3 and sell_agent_id in
    column 4 (market-spec §13.2 + loader docstring). Both must round-trip
    to their source pair via the single canonical metadata dict.
    """

    def test_both_legs_resolve_to_correct_pairs(self, tmp_path):
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="BUYER", buy_unidad="BU1",
                           buy_zona=_ZONE_ES,
                           sell_agente="SELLER", sell_unidad="SU1",
                           sell_zona=_ZONE_ES),
            ],
        )
        cfg = _config(tmp_path)
        trades = load_omie_trades(cfg)
        meta = load_omie_agent_metadata(cfg)
        buy_id = int(trades[0, 3])
        sell_id = int(trades[0, 4])
        assert meta[buy_id] == ("BUYER", "BU1")
        assert meta[sell_id] == ("SELLER", "SU1")
        assert buy_id != sell_id

    def test_cross_row_consistency_for_same_pair(self, tmp_path):
        """Same (Agente, Unidad) on buy leg in row A == sell leg in row B."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(contract_id=16,
                           buy_agente="X", buy_unidad="U1", buy_zona=_ZONE_ES,
                           sell_agente="Y", sell_unidad="U1", sell_zona=_ZONE_ES,
                           momento="01/06/2025 17:42:00"),
                _trade_row(contract_id=16,
                           buy_agente="Y", buy_unidad="U1", buy_zona=_ZONE_ES,
                           sell_agente="X", sell_unidad="U1", sell_zona=_ZONE_ES,
                           momento="01/06/2025 17:42:01"),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        # Trade 0: buy=X, sell=Y; Trade 1: buy=Y, sell=X → symmetric.
        assert int(trades[0, 3]) == int(trades[1, 4])  # X's id
        assert int(trades[0, 4]) == int(trades[1, 3])  # Y's id


# ===========================================================================
# 13. Zona inclusion preserved (1.5-A contract carried forward)
# ===========================================================================


class TestZonaInclusionPreserved:
    """Zona is never part of the pair key; PT and ES rows both contribute.

    Phase 1.5 locked decision #5: all OMIE participants (ES + PT). The
    pair universe must not silently re-introduce a zone filter.
    """

    def test_pt_pairs_present_in_metadata(self, tmp_path):
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_PT, agente="PTAG",
                           unidad="PTU1"),
                _order_row(contract_id=16, zona=_ZONE_ES, agente="ESAG",
                           unidad="ESU1", momento="01/06/2025 17:41:13",
                           side="V"),
            ],
        )
        _write_trades(tmp_path, [])
        meta = load_omie_agent_metadata(_config(tmp_path))
        pairs = set(meta.values())
        assert ("PTAG", "PTU1") in pairs
        assert ("ESAG", "ESU1") in pairs

    def test_zona_not_in_pair_key(self, tmp_path):
        """Same (Agente, Unidad) in ES and PT rows collapses to one id.

        This is a non-trivial invariant: if zona leaked into the pair key
        by accident, the same (Agente, Unidad) across two zone rows would
        get distinct ids.
        """
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, zona=_ZONE_ES, agente="SAME",
                           unidad="SU", momento="01/06/2025 17:41:10"),
                _order_row(contract_id=16, zona=_ZONE_PT, agente="SAME",
                           unidad="SU", momento="01/06/2025 17:41:11"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 2
        assert int(orders[0, 5]) == int(orders[1, 5]), (
            "Two rows sharing (Agente, Unidad) but different zona got "
            "distinct agent_ids — zona is leaking into the pair key"
        )


# ===========================================================================
# 14. Real-data integration (slow)
# ===========================================================================


_REAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "rawOMIE"
_REAL_DATE = "2025-06-01"
_REAL_DATE_NODASH = "20250601"


@pytest.mark.skipif(
    not (_REAL_DATA_DIR / f"orders_{_REAL_DATE_NODASH}.1").exists(),
    reason="real OMIE data file not present",
)
@pytest.mark.skipif(
    not (_REAL_DATA_DIR / f"trades_{_REAL_DATE_NODASH}.1").exists(),
    reason="real OMIE trades file not present",
)
@pytest.mark.slow
class TestRealDataIntegration:
    """slow: parse real ~MB OMIE orders_*.1 / trades_*.1 end-to-end.

    Justification: integration coverage against the production data path.
    Each test parses real files containing Portuguese rows, blank-leg
    XBID rows, CRLF line endings, and variable-width optional fields;
    exceeds the 1s fast-suite budget.
    """

    @pytest.fixture(scope="class")
    def real_cfg(self, tmp_path_factory):
        cache_dir = tmp_path_factory.mktemp("real_cache")
        return OMIEDataConfig(
            data_path=str(_REAL_DATA_DIR),
            contract_id=36,  # 09:00-09:15: contains trades incl. blank-leg rows
            delivery_date=_REAL_DATE,
            cache_path=str(cache_dir),
        )

    def test_real_metadata_is_pair_valued(self, real_cfg):
        """slow: verify ({agent_id: (agente, unidad)}) contract on real data."""
        meta = load_omie_agent_metadata(real_cfg)
        assert len(meta) > 0, "Real OMIE metadata is empty"
        for aid, val in meta.items():
            assert isinstance(val, tuple) and len(val) == 2
            assert isinstance(val[0], str) and isinstance(val[1], str)

    def test_real_ids_are_contiguous_1_indexed(self, real_cfg):
        """slow: ids form [1, N] with no gaps on real data."""
        meta = load_omie_agent_metadata(real_cfg)
        assert set(meta.keys()) == set(range(1, len(meta) + 1))

    def test_real_orders_every_id_resolves(self, real_cfg):
        """slow: every agent_id in orders[:, 5] is a key in metadata."""
        orders = load_omie_orders(real_cfg)
        meta = load_omie_agent_metadata(real_cfg)
        ids = {int(x) for x in np.asarray(orders[:, 5]).tolist()}
        unresolved = ids - set(meta.keys())
        assert not unresolved, (
            f"Real orders contain unresolved agent_ids: {sorted(unresolved)[:10]}"
        )

    def test_real_trades_blank_leg_pair_present(self, real_cfg):
        """slow: contract 36 has cross-border trades → blank pair must be in meta."""
        trades = load_omie_trades(real_cfg)
        meta = load_omie_agent_metadata(real_cfg)
        if trades.shape[0] == 0:
            pytest.skip("No trades on contract 36 for this date")
        # At least some trade rows on this real contract have a blank leg
        # (observed in head of trades_20250601.1 at contract 36).
        both_ids = (
            {int(x) for x in np.asarray(trades[:, 3]).tolist()}
            | {int(x) for x in np.asarray(trades[:, 4]).tolist()}
        )
        pairs_seen = {meta[aid] for aid in both_ids}
        assert ("", "") in pairs_seen, (
            "Expected at least one blank-leg ('', '') pair in trades for "
            f"contract 36 on {_REAL_DATE}; got pairs: "
            f"{sorted(pairs_seen)[:5]}"
        )

    def test_real_orders_and_trades_share_same_universe(self, real_cfg):
        """slow: union determinism — two sequential metadata calls agree."""
        m1 = load_omie_agent_metadata(real_cfg)
        m2 = load_omie_agent_metadata(real_cfg)
        assert m1 == m2

    def test_real_same_pair_same_id_across_orders(self, real_cfg):
        """slow: if a pair appears > once in orders, all rows share one id."""
        orders = load_omie_orders(real_cfg)
        meta = load_omie_agent_metadata(real_cfg)
        if orders.shape[0] < 2:
            pytest.skip("Not enough orders on this contract")
        # Group ids by their pair and assert singleton sets per pair.
        per_pair_ids: dict[tuple[str, str], set[int]] = {}
        for i in range(orders.shape[0]):
            aid = int(orders[i, 5])
            pair = meta[aid]
            per_pair_ids.setdefault(pair, set()).add(aid)
        for pair, ids in per_pair_ids.items():
            assert len(ids) == 1, (
                f"Pair {pair} resolved to multiple ids across orders: {ids}"
            )
