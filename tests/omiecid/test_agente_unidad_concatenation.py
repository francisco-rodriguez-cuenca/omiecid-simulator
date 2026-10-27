"""Locked behavioral tests for Story 1.5-B — Agente-Unidad Concatenation.

Scope (derived from agile-plan.md Story 1.5-B, migration-plan.md v4.0
§Phase 1.5, market-spec.md §13.1/§13.2/§13.4/§13.5, and the 1.5-B
pre-sprint architecture review in sprint-status.md):

  The OMIE ``agent_id`` is produced by concatenating each unique
  ``(Agente, Unidad)`` pair into a single deterministic 1-indexed
  integer, via ``_build_agent_mapping(agente_series, unidad_series)``
  which returns two dicts ``(pair_to_id, id_to_pair)`` — symmetric and
  lossless. A new public function ``load_omie_agent_metadata(config)``
  returns the canonical reverse mapping ``{agent_id: (agente, unidad)}``
  as session metadata; any ``{agent_id: agente_string}`` view is
  derived at the call site in one line (no separate artifact).

  Invariants enforced here:
    1. Pair-based mapping: same (Agente, Unidad) pair → same agent_id.
    2. Distinctness: different pairs → distinct agent_ids.
       In particular, same Agente + different Unidad → distinct ids
       (this is the story's core distinguishing test, per TC-2).
    3. Losslessness: ``load_omie_agent_metadata`` round-trips with
       ``_build_agent_mapping``.
    4. Shape invariance: orders (N, 13), trades (M, 6) — unchanged.
    5. ``OMIEOrder`` stays at 13 named fields.
    6. Determinism across runs (seed-independent; no set-based ordering).
    7. No ``id_to_agente`` artifact is exposed from the data-loader module.
    8. 1-indexed; contiguous [1, N] range.
    9. ``load_omie_agent_metadata`` is consistent with trade-side pairs
       (union over orders + trades).
   10. Blank-leg trades (XBID cross-border) get a valid positive
       agent_id under the ``("", "")`` pair convention.

Out of scope (per architect TC-13 — deferred to 1.5-C / 1.5-D / Phase 2):
  - Timestamp session-relative conversion (1.5-C).
  - Settlement sum-over-units invariant (1.5-D).
  - JAX-traceability of the mapping (Phase 2).
  - ``OMIEOrder`` field expansion (forbidden by locked decision #4).

Agent-id policy (carried forward from 1.5-A audit finding F4):
  All tests assert *properties* of the mapping (determinism,
  positivity, distinctness, losslessness, contiguity) — never specific
  integer values — because the pair-universe enumeration can legitimately
  vary by implementation detail as long as the invariants above hold.

Runtime budget: all tests target < 1s wall-clock; fixtures are built
in-process via ``tmp_path``. No ``@pytest.mark.slow`` tests.

Reference paths:
  - omiecid_simulator/jaxomiecid/omie_data_loader.py
  - omiecid_simulator/jaxomiecid/omie_data_config.py
  - docs/omiecid/market-spec.md §13
  - docs/omiecid/migration-plan.md v4.0 §Phase 1.5
  - docs/omiecid/agile-plan.md v4.1 Story 1.5-B
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import pandas as pd
import pytest

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.jaxomiecid.omie_data_loader import (
    _build_agent_mapping,
    load_omie_orders,
    load_omie_trades,
)
from omiecid_simulator.omiecid.order_types import OMIEOrder


# ---------------------------------------------------------------------------
# Synthetic OMIE file writers (same pattern as 1.5-A locked tests)
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

_ZONE_ES = "10YES-REE------0"
_ZONE_PT = "10YPT-REN------W"
_DATE_DMY = "01/06/2025"
_DATE_ISO = "2025-06-01"
_DATE_NODASH = "20250601"


def _contract_str(contract_id: int) -> str:
    hour = contract_id // 4
    minute = (contract_id % 4) * 15
    end_minute = minute + 15
    end_hour = hour
    if end_minute == 60:
        end_hour = hour + 1
        end_minute = 0
    return (
        f"20250602 {hour:02d}:{minute:02d}-20250602 "
        f"{end_hour:02d}:{end_minute:02d}"
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
    agente: str,
    unidad: str,
    zona: str = _ZONE_ES,
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
        f"{precio};{cantidad};{side};{exec_r};{validez};{cant_red};{ppd};"
        f"{momento}"
    )


def _trade_row(
    *,
    contract_id: int,
    buy_agente: str,
    buy_unidad: str,
    sell_agente: str,
    sell_unidad: str,
    buy_zona: str = _ZONE_ES,
    sell_zona: str = _ZONE_ES,
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


# Fixture-only product date (OMIE-TAPE-CONTRACT-DATE-1, group F1).  Every
# fixture row carries a ``Contrato`` dated 2025-06-02 (``_contract_str``),
# i.e. delivery product (2025-06-02, k).  Per market-spec v2.10 §13/§13.4 the
# written session file ``*_20250601.1`` is file D-1 of that product; the
# product also spans session file D (``*_20250602.1``).  ``_DATE_ISO`` /
# ``_DATE_NODASH`` keep naming the session file the tests write.
_PRODUCT_DATE_ISO = "2025-06-02"
_PRODUCT_SESSION_FILES_NODASH = ("20250601", "20250602")  # (D-1, D)


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
    )


# ===========================================================================
# 1. _build_agent_mapping signature and return-shape contract
# ===========================================================================


class TestBuildAgentMappingSignature:
    """The new signature takes two Series and returns (pair_to_id, id_to_pair).

    Replaces the old ``_build_agent_mapping(agent_series)`` signature
    used in Phase 1 / 1.5-A. Tests here exercise the *callable contract*
    in isolation from the data-loader I/O surface.
    """

    def test_accepts_two_series(self):
        """Callable with two pd.Series arguments succeeds without TypeError."""
        ag = pd.Series(["ACC", "BETA"])
        un = pd.Series(["U1", "U2"])
        result = _build_agent_mapping(ag, un)
        assert result is not None

    def test_returns_two_dicts(self):
        """Return value must be a 2-tuple of dicts."""
        ag = pd.Series(["ACC", "BETA"])
        un = pd.Series(["U1", "U2"])
        result = _build_agent_mapping(ag, un)
        assert isinstance(result, tuple)
        assert len(result) == 2
        pair_to_id, id_to_pair = result
        assert isinstance(pair_to_id, dict)
        assert isinstance(id_to_pair, dict)

    def test_forward_map_keys_are_pair_tuples(self):
        """pair_to_id keys are (agente, unidad) 2-tuples of strings."""
        ag = pd.Series(["ACC", "BETA"])
        un = pd.Series(["U1", "U2"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        for key in pair_to_id.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str)
            assert isinstance(key[1], str)

    def test_forward_map_values_are_positive_ints(self):
        """pair_to_id values are positive integers (>= 1, 1-indexed)."""
        ag = pd.Series(["ACC", "BETA"])
        un = pd.Series(["U1", "U2"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        for value in pair_to_id.values():
            assert isinstance(value, int)
            assert value >= 1

    def test_reverse_map_keys_are_ints(self):
        """id_to_pair keys are integers."""
        ag = pd.Series(["ACC", "BETA"])
        un = pd.Series(["U1", "U2"])
        _, id_to_pair = _build_agent_mapping(ag, un)
        for key in id_to_pair.keys():
            assert isinstance(key, int)

    def test_reverse_map_values_are_pair_tuples(self):
        """id_to_pair values are (agente, unidad) 2-tuples of strings."""
        ag = pd.Series(["ACC", "BETA"])
        un = pd.Series(["U1", "U2"])
        _, id_to_pair = _build_agent_mapping(ag, un)
        for value in id_to_pair.values():
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert isinstance(value[0], str)
            assert isinstance(value[1], str)

    def test_series_of_equal_length_required(self):
        """Two series of unequal length is an ill-formed input.

        Not a strict API check, but the natural consequence of zipping the
        two series pairwise — any well-formed mapping must rely on row
        alignment between Agente and Unidad columns. This is a defensive
        assertion that the function does not silently succeed when fed
        obviously-misaligned inputs.
        """
        ag = pd.Series(["ACC", "BETA", "GAMMA"])
        un = pd.Series(["U1", "U2"])
        # Accept either: (a) the function raises (ValueError / mismatch
        # detected), or (b) it returns a mapping built from the shorter
        # common length — but NOT silently creating a 3-pair mapping that
        # pairs "GAMMA" with no unit. The contract is "rows pair 1:1".
        try:
            pair_to_id, _ = _build_agent_mapping(ag, un)
        except (ValueError, IndexError, TypeError, AssertionError):
            return  # acceptable: explicit rejection
        # If no exception, no pair may reference "GAMMA" with an
        # out-of-range unit.
        for (agente, _unidad) in pair_to_id.keys():
            if agente == "GAMMA":
                pytest.fail(
                    "GAMMA paired despite having no aligned Unidad row "
                    "— silent row misalignment."
                )


# ===========================================================================
# 2. Pair determinism — TC-1 (same input data → same dicts)
# ===========================================================================


class TestPairDeterminism:
    """Same input → same mapping (determinism)."""

    def test_same_input_same_forward_map(self):
        """Building the mapping twice on the same Series yields == dicts."""
        ag = pd.Series(["ACCGR", "BETA", "GAMMA"])
        un = pd.Series(["ACCGC01", "U1", "U2"])
        p1, _ = _build_agent_mapping(ag, un)
        p2, _ = _build_agent_mapping(ag, un)
        assert p1 == p2

    def test_same_input_same_reverse_map(self):
        """Building the mapping twice yields equal id_to_pair dicts."""
        ag = pd.Series(["ACCGR", "BETA", "GAMMA"])
        un = pd.Series(["ACCGC01", "U1", "U2"])
        _, r1 = _build_agent_mapping(ag, un)
        _, r2 = _build_agent_mapping(ag, un)
        assert r1 == r2

    def test_input_row_order_independent(self):
        """Shuffling input row order does not change the mapping.

        The sort key is over the unique pair universe, not the row index,
        so ``[(ACC, U1), (BETA, U2)]`` and ``[(BETA, U2), (ACC, U1)]``
        must yield the same ``pair_to_id`` dict.
        """
        ag1 = pd.Series(["ACC", "BETA", "GAMMA"])
        un1 = pd.Series(["U1", "U2", "U3"])
        ag2 = pd.Series(["GAMMA", "ACC", "BETA"])
        un2 = pd.Series(["U3", "U1", "U2"])
        p1, _ = _build_agent_mapping(ag1, un1)
        p2, _ = _build_agent_mapping(ag2, un2)
        assert p1 == p2

    def test_duplicate_rows_collapse_to_single_id(self):
        """Repeated identical pairs get a single entry in the mapping."""
        ag = pd.Series(["ACC", "ACC", "ACC"])
        un = pd.Series(["U1", "U1", "U1"])
        pair_to_id, id_to_pair = _build_agent_mapping(ag, un)
        assert len(pair_to_id) == 1
        assert len(id_to_pair) == 1
        assert ("ACC", "U1") in pair_to_id


# ===========================================================================
# 3. Pair distinctness — TC-2 / TC-3 (the story's core distinguishing tests)
# ===========================================================================


class TestPairDistinctness:
    """Different pairs get different ids; same pair gets same id."""

    def test_same_agente_different_unidad_distinct_ids(self):
        """Same Agente, different Unidad → distinct agent_ids.

        This is the Phase 1.5 distinguishing invariant: under Phase 1 /
        1.5-A these two rows would have collapsed to the same agent_id
        (Agente-only mapping). Under 1.5-B they must differ.
        """
        ag = pd.Series(["ACCGR", "ACCGR"])
        un = pd.Series(["ACCGC01", "ACCGC02"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        id1 = pair_to_id[("ACCGR", "ACCGC01")]
        id2 = pair_to_id[("ACCGR", "ACCGC02")]
        assert id1 != id2, (
            "Same Agente with different Unidad must map to distinct ids"
        )

    def test_same_pair_same_id(self):
        """Two occurrences of the same pair map to the same agent_id."""
        ag = pd.Series(["ACCGR", "BETA", "ACCGR"])
        un = pd.Series(["ACCGC01", "BETA01", "ACCGC01"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        # Only two distinct pairs were provided.
        assert len(pair_to_id) == 2

    def test_different_agente_same_unidad_distinct_ids(self):
        """Different Agente, same Unidad string → distinct ids.

        Unidad codes are agent-local — different agents may happen to
        register units with identical codes. The tuple key must keep
        them distinct.
        """
        ag = pd.Series(["ACC", "BETA"])
        un = pd.Series(["UNIT1", "UNIT1"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        id1 = pair_to_id[("ACC", "UNIT1")]
        id2 = pair_to_id[("BETA", "UNIT1")]
        assert id1 != id2

    def test_different_agente_different_unidad_distinct_ids(self):
        """Fully distinct pairs get fully distinct ids (no collisions)."""
        ag = pd.Series(["ACC", "BETA", "GAMMA", "DELTA"])
        un = pd.Series(["U1", "U2", "U3", "U4"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        ids = set(pair_to_id.values())
        assert len(ids) == 4


# ===========================================================================
# 4. Lossless round-trip — TC-4
# ===========================================================================


class TestRoundTripLosslessness:
    """pair_to_id and id_to_pair are bijective inverses."""

    def test_forward_then_reverse_round_trip(self):
        """For every pair p, id_to_pair[pair_to_id[p]] == p."""
        ag = pd.Series(["ACCGR", "ACCGR", "BETA", "GAMMA"])
        un = pd.Series(["ACCGC01", "ACCGC02", "BETA01", "GAMM01"])
        pair_to_id, id_to_pair = _build_agent_mapping(ag, un)
        for pair in pair_to_id.keys():
            assert id_to_pair[pair_to_id[pair]] == pair

    def test_reverse_then_forward_round_trip(self):
        """For every id i, pair_to_id[id_to_pair[i]] == i."""
        ag = pd.Series(["ACCGR", "ACCGR", "BETA", "GAMMA"])
        un = pd.Series(["ACCGC01", "ACCGC02", "BETA01", "GAMM01"])
        pair_to_id, id_to_pair = _build_agent_mapping(ag, un)
        for aid in id_to_pair.keys():
            assert pair_to_id[id_to_pair[aid]] == aid

    def test_forward_and_reverse_same_cardinality(self):
        """|pair_to_id| == |id_to_pair| (bijective by cardinality)."""
        ag = pd.Series(["A", "A", "B", "C"])
        un = pd.Series(["U1", "U2", "U1", "U1"])
        pair_to_id, id_to_pair = _build_agent_mapping(ag, un)
        assert len(pair_to_id) == len(id_to_pair)

    def test_id_set_matches_pair_image(self):
        """Set of id_to_pair keys equals set of pair_to_id values."""
        ag = pd.Series(["A", "A", "B"])
        un = pd.Series(["U1", "U2", "U1"])
        pair_to_id, id_to_pair = _build_agent_mapping(ag, un)
        assert set(id_to_pair.keys()) == set(pair_to_id.values())

    def test_pair_set_matches_id_image(self):
        """Set of pair_to_id keys equals set of id_to_pair values."""
        ag = pd.Series(["A", "A", "B"])
        un = pd.Series(["U1", "U2", "U1"])
        pair_to_id, id_to_pair = _build_agent_mapping(ag, un)
        assert set(pair_to_id.keys()) == set(id_to_pair.values())


# ===========================================================================
# 5. 1-indexing and contiguity
# ===========================================================================


class TestIndexing:
    """Ids start at 1 and form a contiguous [1, N] range."""

    def test_min_id_is_one(self):
        """Smallest assigned id is 1 (not 0)."""
        ag = pd.Series(["ACC", "BETA", "GAMMA"])
        un = pd.Series(["U1", "U1", "U1"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        assert min(pair_to_id.values()) == 1

    def test_ids_are_contiguous_range(self):
        """Ids form a contiguous range {1, 2, ..., N}."""
        ag = pd.Series(["ACC", "BETA", "GAMMA", "DELTA", "EPSILON"])
        un = pd.Series(["U1", "U2", "U3", "U4", "U5"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        ids = sorted(pair_to_id.values())
        assert ids == list(range(1, len(ids) + 1))

    def test_ids_distinct(self):
        """Each pair gets a unique id."""
        ag = pd.Series(["ACC", "ACC", "BETA", "GAMMA"])
        un = pd.Series(["U1", "U2", "U1", "U1"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        assert len(set(pair_to_id.values())) == len(pair_to_id)


# ===========================================================================
# 6. Alphabetical (tuple) sort ordering
# ===========================================================================


class TestAlphabeticalSortKey:
    """Ids are assigned in alphabetical tuple order on (agente, unidad).

    Canonical sort: ``sorted(unique_pairs)`` (implicit element-wise
    Unicode code-point ordering). Does NOT use concatenated string keys
    (which would collide on separator choice) or locale-dependent
    comparisons.
    """

    def test_sort_respects_agente_first(self):
        """A < B on Agente gives the A pair a smaller id, all else equal."""
        ag = pd.Series(["BBB", "AAA"])
        un = pd.Series(["U1", "U1"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        id_aaa = pair_to_id[("AAA", "U1")]
        id_bbb = pair_to_id[("BBB", "U1")]
        assert id_aaa < id_bbb

    def test_sort_uses_unidad_as_tiebreak(self):
        """For the same Agente, the Unidad breaks the tie alphabetically."""
        ag = pd.Series(["ACC", "ACC"])
        un = pd.Series(["U2", "U1"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        id_u1 = pair_to_id[("ACC", "U1")]
        id_u2 = pair_to_id[("ACC", "U2")]
        assert id_u1 < id_u2

    def test_sort_not_concatenation_based(self):
        """Sort is on (agente, unidad) tuple, not a concatenated string.

        Under a concatenation-with-separator sort (e.g. ``f"{a}_{u}"``)
        the pair ``("ACC", "ZZ")`` would come BEFORE ``("ACCX", "AA")``
        only if ``_`` < ``X`` in the underlying code-point order
        (it is not — ``_`` is 0x5F > ``X`` at 0x58). The canonical
        tuple sort, on the other hand, always orders ``("ACC", ...)``
        before ``("ACCX", ...)`` because ``"ACC" < "ACCX"`` element-wise.
        """
        ag = pd.Series(["ACCX", "ACC"])
        un = pd.Series(["AA", "ZZ"])
        pair_to_id, _ = _build_agent_mapping(ag, un)
        id_short = pair_to_id[("ACC", "ZZ")]
        id_long = pair_to_id[("ACCX", "AA")]
        # Tuple-sort canonical: ("ACC", "ZZ") < ("ACCX", "AA")
        assert id_short < id_long


# ===========================================================================
# 7. load_omie_agent_metadata — public API surface
# ===========================================================================


class TestLoadOmieAgentMetadataSurface:
    """The new ``load_omie_agent_metadata(config)`` public function.

    Per migration-plan v4.0 §Phase 1.5:
      - Signature: ``load_omie_agent_metadata(config: OMIEDataConfig)
        -> dict[int, tuple[str, str]]``
      - Returns the canonical reverse mapping {agent_id: (agente, unidad)}
      - Lossless — round-trips with the forward pair_to_id map
      - Do NOT persist any {agent_id: agente_string} variant artifact
    """

    def test_function_is_importable_from_data_loader(self):
        """``load_omie_agent_metadata`` is exposed from
        ``omiecid_simulator.jaxomiecid.omie_data_loader``."""
        mod = importlib.import_module(
            "omiecid_simulator.jaxomiecid.omie_data_loader"
        )
        assert hasattr(mod, "load_omie_agent_metadata"), (
            "1.5-B must add load_omie_agent_metadata to the loader module"
        )

    def test_function_takes_single_config_argument(self):
        """The function has exactly one required positional parameter
        named ``config`` (or accepting an ``OMIEDataConfig``)."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        sig = inspect.signature(load_omie_agent_metadata)
        params = list(sig.parameters.values())
        required = [
            p for p in params
            if p.default is inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        assert len(required) == 1

    def test_returns_dict(self, tmp_path):
        """Return type is a dict."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(
            tmp_path,
            [_order_row(contract_id=16, agente="ACC", unidad="U1")],
        )
        _write_trades(tmp_path, [])
        result = load_omie_agent_metadata(_config(tmp_path))
        assert isinstance(result, dict)

    def test_returns_int_keys(self, tmp_path):
        """Dict keys are ints (agent_id)."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, agente="ACC", unidad="U1"),
                _order_row(contract_id=16, agente="BETA", unidad="U1",
                           momento="01/06/2025 17:41:13"),
            ],
        )
        _write_trades(tmp_path, [])
        result = load_omie_agent_metadata(_config(tmp_path))
        for k in result.keys():
            assert isinstance(k, int)

    def test_returns_pair_tuple_values(self, tmp_path):
        """Dict values are (agente, unidad) 2-tuples of strings."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, agente="ACC", unidad="U1"),
                _order_row(contract_id=16, agente="BETA", unidad="U1",
                           momento="01/06/2025 17:41:13"),
            ],
        )
        _write_trades(tmp_path, [])
        result = load_omie_agent_metadata(_config(tmp_path))
        for v in result.values():
            assert isinstance(v, tuple)
            assert len(v) == 2
            assert isinstance(v[0], str)
            assert isinstance(v[1], str)

    def test_one_indexed(self, tmp_path):
        """Min key is 1 (1-indexed)."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, agente="ACC", unidad="U1"),
                _order_row(contract_id=16, agente="BETA", unidad="U2",
                           momento="01/06/2025 17:41:13"),
            ],
        )
        _write_trades(tmp_path, [])
        result = load_omie_agent_metadata(_config(tmp_path))
        assert min(result.keys()) == 1

    def test_contiguous_range(self, tmp_path):
        """Keys form a contiguous range [1, N] with no gaps."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, agente="ACC", unidad="U1"),
                _order_row(contract_id=16, agente="BETA", unidad="U2",
                           momento="01/06/2025 17:41:13"),
                _order_row(contract_id=16, agente="GAMMA", unidad="U3",
                           momento="01/06/2025 17:41:14"),
            ],
        )
        _write_trades(tmp_path, [])
        result = load_omie_agent_metadata(_config(tmp_path))
        ids = sorted(result.keys())
        assert ids == list(range(1, len(ids) + 1))


# ===========================================================================
# 8. load_omie_agent_metadata — consistency with loader output
# ===========================================================================


class TestMetadataConsistencyWithLoader:
    """``load_omie_agent_metadata`` round-trips with ``load_omie_orders``
    and ``load_omie_trades``."""

    def test_every_orders_agent_id_appears_in_metadata(self, tmp_path):
        """Every agent_id in column 5 of loaded orders is a key in metadata."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, agente="ACCGR", unidad="ACCGC01"),
                _order_row(contract_id=16, agente="ACCGR", unidad="ACCGC02",
                           momento="01/06/2025 17:41:13"),
                _order_row(contract_id=16, agente="BETA", unidad="BETA01",
                           momento="01/06/2025 17:41:14"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        metadata = load_omie_agent_metadata(_config(tmp_path))
        for i in range(orders.shape[0]):
            aid = int(orders[i, 5])
            assert aid in metadata, (
                f"row {i}: agent_id {aid} absent from metadata"
            )

    def test_metadata_pair_matches_source_row(self, tmp_path):
        """For each order row, metadata[agent_id] matches the
        (Agente, Unidad) of the source row."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, agente="ACCGR", unidad="ACCGC01"),
                _order_row(contract_id=16, agente="BETA", unidad="BETA01",
                           momento="01/06/2025 17:41:13"),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        metadata = load_omie_agent_metadata(_config(tmp_path))
        # Extract (agent_id, expected_pair) assertions by mapping back.
        # Build a set of (Agente, Unidad) -> {agent_ids} per the loader
        # output, then check that metadata agrees.
        seen_pairs = set()
        for i in range(orders.shape[0]):
            aid = int(orders[i, 5])
            pair = metadata[aid]
            assert isinstance(pair, tuple) and len(pair) == 2
            seen_pairs.add(pair)
        # At least two distinct pairs were loaded
        assert len(seen_pairs) == 2
        assert ("ACCGR", "ACCGC01") in seen_pairs
        assert ("BETA", "BETA01") in seen_pairs

    def test_trade_side_pairs_included_in_metadata(self, tmp_path):
        """Pair universe is a union over orders + trades: a pair appearing
        only on the trade side must still be present in
        ``load_omie_agent_metadata``.
        """
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(tmp_path, [
            _order_row(contract_id=16, agente="ONLYO", unidad="OONE"),
        ])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="TRADEONLY_B", buy_unidad="TBU",
                    sell_agente="TRADEONLY_S", sell_unidad="TSU",
                ),
            ],
        )
        metadata = load_omie_agent_metadata(_config(tmp_path))
        pairs = set(metadata.values())
        assert ("TRADEONLY_B", "TBU") in pairs, (
            "Buy-side trade pair must appear in agent metadata"
        )
        assert ("TRADEONLY_S", "TSU") in pairs, (
            "Sell-side trade pair must appear in agent metadata"
        )

    def test_trade_buy_agent_id_matches_metadata_pair(self, tmp_path):
        """For each trade row, column 3 (buy_agent_id) is a metadata key
        whose pair matches the parsed (Agente compra, Unidad compra)."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="BUYER1", buy_unidad="BU1",
                    sell_agente="SELLER1", sell_unidad="SU1",
                ),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        metadata = load_omie_agent_metadata(_config(tmp_path))
        assert trades.shape[0] == 1
        buy_aid = int(trades[0, 3])
        assert buy_aid in metadata
        assert metadata[buy_aid] == ("BUYER1", "BU1")

    def test_trade_sell_agent_id_matches_metadata_pair(self, tmp_path):
        """For each trade row, column 4 (sell_agent_id) is a metadata key
        whose pair matches the parsed (Agente venta, Unidad venta)."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="BUYER1", buy_unidad="BU1",
                    sell_agente="SELLER1", sell_unidad="SU1",
                ),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        metadata = load_omie_agent_metadata(_config(tmp_path))
        assert trades.shape[0] == 1
        sell_aid = int(trades[0, 4])
        assert sell_aid in metadata
        assert metadata[sell_aid] == ("SELLER1", "SU1")

    def test_metadata_is_determinstic_across_calls(self, tmp_path):
        """Two calls with the same config return equal dicts."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(
            tmp_path,
            [
                _order_row(contract_id=16, agente="ACC", unidad="U1"),
                _order_row(contract_id=16, agente="BETA", unidad="U1",
                           momento="01/06/2025 17:41:13"),
            ],
        )
        _write_trades(tmp_path, [])
        m1 = load_omie_agent_metadata(_config(tmp_path))
        m2 = load_omie_agent_metadata(_config(tmp_path))
        assert m1 == m2


# ===========================================================================
# 9. End-to-end pair behaviour through the loader
# ===========================================================================


class TestLoaderPairBehaviour:
    """``load_omie_orders`` and ``load_omie_trades`` reflect the
    (Agente, Unidad) mapping in their agent_id columns."""

    def test_same_agente_different_unidad_distinct_ids_in_orders(
        self, tmp_path
    ):
        """Two orders with same Agente but different Unidad get
        different agent_id values in the (N, 13) loader output."""
        _write_orders(
            tmp_path,
            [
                _order_row(
                    contract_id=16, agente="ACCGR", unidad="ACCGC01",
                    momento="01/06/2025 17:41:12",
                ),
                _order_row(
                    contract_id=16, agente="ACCGR", unidad="ACCGC02",
                    momento="01/06/2025 17:41:13",
                ),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 2
        aid1 = int(orders[0, 5])
        aid2 = int(orders[1, 5])
        assert aid1 != aid2

    def test_same_pair_same_id_in_orders(self, tmp_path):
        """Two orders with the same (Agente, Unidad) pair get the same
        agent_id in the (N, 13) output."""
        _write_orders(
            tmp_path,
            [
                _order_row(
                    contract_id=16, agente="ACCGR", unidad="ACCGC01",
                    momento="01/06/2025 17:41:12",
                ),
                _order_row(
                    contract_id=16, agente="BETA", unidad="B1",
                    momento="01/06/2025 17:41:13",
                ),
                _order_row(
                    contract_id=16, agente="ACCGR", unidad="ACCGC01",
                    momento="01/06/2025 17:41:14",
                ),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 3
        # Rows 0 and 2 share the same pair.
        assert int(orders[0, 5]) == int(orders[2, 5])
        # Row 1 (different Agente) differs.
        assert int(orders[1, 5]) != int(orders[0, 5])

    def test_order_agent_ids_all_positive(self, tmp_path):
        """All agent_id values in the loader output are positive."""
        _write_orders(
            tmp_path,
            [
                _order_row(
                    contract_id=16, agente="A", unidad="U1",
                    momento="01/06/2025 17:41:12",
                ),
                _order_row(
                    contract_id=16, agente="B", unidad="U1",
                    momento="01/06/2025 17:41:13",
                ),
                _order_row(
                    contract_id=16, agente="A", unidad="U2",
                    momento="01/06/2025 17:41:14",
                ),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert jnp.all(orders[:, 5] > 0)

    def test_different_agente_same_unidad_string_distinct_ids_in_orders(
        self, tmp_path
    ):
        """Two different Agentes reusing the same Unidad code still
        get distinct agent_ids in the output."""
        _write_orders(
            tmp_path,
            [
                _order_row(
                    contract_id=16, agente="ACC", unidad="UNIT1",
                    momento="01/06/2025 17:41:12",
                ),
                _order_row(
                    contract_id=16, agente="BETA", unidad="UNIT1",
                    momento="01/06/2025 17:41:13",
                ),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 2
        assert int(orders[0, 5]) != int(orders[1, 5])


# ===========================================================================
# 10. Shape invariance — TC-7 (OMIEOrder stays at 13, arrays unchanged)
# ===========================================================================


class TestShapeInvariance:
    """Array shapes are (N, 13) / (M, 6) — unchanged from Phase 1."""

    def test_orders_have_13_columns(self, tmp_path):
        """Orders output shape[1] == 13."""
        _write_orders(
            tmp_path,
            [
                _order_row(
                    contract_id=16, agente="ACC", unidad="U1",
                    momento="01/06/2025 17:41:12",
                ),
                _order_row(
                    contract_id=16, agente="ACC", unidad="U2",
                    momento="01/06/2025 17:41:13",
                ),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[1] == 13

    def test_trades_have_6_columns(self, tmp_path):
        """Trades output shape[1] == 6."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="B", buy_unidad="BU",
                    sell_agente="S", sell_unidad="SU",
                ),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        assert trades.shape[1] == 6

    def test_orders_shape_preserved_across_many_pairs(self, tmp_path):
        """Even with many distinct (Agente, Unidad) pairs, shape stays
        (N, 13)."""
        rows = []
        for i in range(10):
            rows.append(
                _order_row(
                    contract_id=16,
                    agente=f"AG{i:02d}",
                    unidad=f"U{i:02d}",
                    momento=f"01/06/2025 17:{i:02d}:00",
                )
            )
        _write_orders(tmp_path, rows)
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape == (10, 13)

    def test_orders_dtype_is_float32(self, tmp_path):
        """Orders output dtype is float32 (unchanged)."""
        _write_orders(
            tmp_path,
            [_order_row(contract_id=16, agente="ACC", unidad="U1")],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.dtype == jnp.int32

    def test_trades_dtype_is_float32(self, tmp_path):
        """Trades output dtype is float32 (unchanged)."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="B", buy_unidad="BU",
                    sell_agente="S", sell_unidad="SU",
                ),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        assert trades.dtype == jnp.int32


class TestOMIEOrderFieldCountInvariant:
    """``OMIEOrder`` NamedTuple stays at 13 fields — forbidden to expand
    per migration-plan locked decision #4."""

    def test_omie_order_has_exactly_13_fields(self):
        """``OMIEOrder._fields`` has exactly 13 names."""
        assert len(OMIEOrder._fields) == 13

    def test_omie_order_field_names_unchanged(self):
        """Field names are the canonical 13 from migration-plan §M1.2.

        No new field named ``unit_id`` / ``unidad`` / etc.
        """
        expected = (
            "contract_id",
            "side",
            "price",
            "quantity",
            "order_id",
            "agent_id",
            "timestamp",
            "exec_restriction",
            "validity",
            "expiry_time",
            "status",
            "peak_qty",
            "ppd",
        )
        assert OMIEOrder._fields == expected

    def test_omie_order_has_no_unit_id_field(self):
        """Guard against accidental 13→14 expansion to ``unit_id``."""
        assert "unit_id" not in OMIEOrder._fields
        assert "unidad" not in OMIEOrder._fields


# ===========================================================================
# 11. Deterministic sort across runs — TC-9 (seed independence)
# ===========================================================================


class TestDeterminismAcrossRuns:
    """The mapping is independent of ``PYTHONHASHSEED`` — no set-based
    iteration order leaks into the result.

    This test is an explicit guard against the anti-pattern of
    building the pair universe via ``set(pairs)`` without a follow-up
    ``sorted(...)``, because CPython set iteration order depends on
    the hash seed.
    """

    def test_mapping_identical_across_two_process_runs(self, tmp_path):
        """Running the loader in two subprocesses with different
        PYTHONHASHSEED values yields identical id_to_pair dicts."""
        # Write fixture files once.
        _write_orders(
            tmp_path,
            [
                _order_row(
                    contract_id=16, agente="ACC", unidad="U1",
                    momento="01/06/2025 17:41:12",
                ),
                _order_row(
                    contract_id=16, agente="BETA", unidad="U1",
                    momento="01/06/2025 17:41:13",
                ),
                _order_row(
                    contract_id=16, agente="ACC", unidad="U2",
                    momento="01/06/2025 17:41:14",
                ),
                _order_row(
                    contract_id=16, agente="GAMMA", unidad="G1",
                    momento="01/06/2025 17:41:15",
                ),
            ],
        )
        _write_trades(tmp_path, [])
        _complete_product_files(tmp_path)

        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(Path.cwd())!r})\n"
            "from omiecid_simulator.jaxomiecid.omie_data_config import "
            "OMIEDataConfig\n"
            "from omiecid_simulator.jaxomiecid.omie_data_loader import "
            "load_omie_agent_metadata\n"
            f"cfg = OMIEDataConfig(data_path={str(tmp_path)!r}, "
            f"contract_id=16, delivery_date='{_PRODUCT_DATE_ISO}')\n"
            "m = load_omie_agent_metadata(cfg)\n"
            "for k in sorted(m.keys()):\n"
            "    print(f'{k}\\t{m[k][0]}\\t{m[k][1]}')\n"
        )

        def _run(seed: str) -> str:
            import os
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            out = subprocess.check_output(
                [sys.executable, "-c", script],
                env=env,
            )
            return out.decode("utf-8")

        out_a = _run("0")
        out_b = _run("12345")
        assert out_a == out_b, (
            "Pair mapping depends on PYTHONHASHSEED — likely a set-based "
            "ordering bug in _build_agent_mapping. Use sorted(...)."
        )


# ===========================================================================
# 12. Negative-space: no persistent id_to_agente artifact — TC-11
# ===========================================================================


class TestNoIdToAgenteArtifact:
    """No public ``id_to_agente`` symbol is exposed from the data-loader
    module.

    Per migration-plan §Phase 1.5 directive: "Do NOT persist a separate
    `id_to_agente` artifact — any caller needing just the parent string
    derives it in one line: ``{aid: pair[0] for aid, pair in
    id_to_pair.items()}``".
    """

    def test_no_public_id_to_agente_symbol(self):
        """The data-loader module does not export an ``id_to_agente``
        function, variable, or class."""
        mod = importlib.import_module(
            "omiecid_simulator.jaxomiecid.omie_data_loader"
        )
        forbidden_names = (
            "id_to_agente",
            "agent_id_to_agente",
            "build_agente_view",
            "load_omie_agente_view",
            "agente_view",
        )
        for name in forbidden_names:
            assert not hasattr(mod, name), (
                f"data-loader module must NOT expose {name!r} — "
                "the parent-Agente view is derived at the call site."
            )

    def test_parent_agente_view_is_derivable_at_call_site(self, tmp_path):
        """The one-line projection
        ``{aid: pair[0] for aid, pair in id_to_pair.items()}`` is the
        canonical call-site recipe — documents the intended usage."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        # Fixture: 2 Agentes, 3 total pairs (ACCGR has two Unidades).
        _write_orders(
            tmp_path,
            [
                _order_row(
                    contract_id=16, agente="ACCGR", unidad="ACCGC01",
                    momento="01/06/2025 17:41:12",
                ),
                _order_row(
                    contract_id=16, agente="ACCGR", unidad="ACCGC02",
                    momento="01/06/2025 17:41:13",
                ),
                _order_row(
                    contract_id=16, agente="BETA", unidad="BETA01",
                    momento="01/06/2025 17:41:14",
                ),
            ],
        )
        _write_trades(tmp_path, [])
        id_to_pair = load_omie_agent_metadata(_config(tmp_path))
        # The documented one-line projection:
        id_to_agente = {
            aid: pair[0] for aid, pair in id_to_pair.items()
        }
        # Two distinct Agentes → len(set(values())) == 2
        assert len(set(id_to_agente.values())) == 2


# ===========================================================================
# 13. Blank-leg trade handling (XBID cross-border) — TC-10
# ===========================================================================


class TestBlankLegTradeHandling:
    """Trades with a blank Agente/Unidad leg (XBID cross-border pattern
    observed in real `data/rawOMIE/trades_20250601.1`) must be
    admissible in the pair mapping.

    Per architect recommendation F5-A: treat the blank pair ``("", "")``
    as a legitimate entry of the pair universe — it gets a positive
    agent_id (typically the smallest, since ``"" < anything`` in tuple
    sort) and participates in the round-trip invariant like any other
    pair.

    If the implementer chose policy F5-B (filter blank legs and map to
    agent_id=0), these tests will fail loudly and must be reconciled
    with the architect by updating both the test file and the
    sprint-status record — not silently adapted.
    """

    def test_blank_buy_leg_pair_in_metadata(self, tmp_path):
        """Blank buy-side pair ``("", "")`` is present in agent metadata."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="", buy_unidad="", buy_zona="",
                    sell_agente="AES", sell_unidad="AESU1",
                ),
            ],
        )
        metadata = load_omie_agent_metadata(_config(tmp_path))
        pairs = set(metadata.values())
        assert ("", "") in pairs, (
            "Blank XBID cross-border pair must be represented in metadata"
        )

    def test_blank_buy_leg_agent_id_positive(self, tmp_path):
        """The blank-leg trade still produces a positive buy_agent_id."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="", buy_unidad="", buy_zona="",
                    sell_agente="AES", sell_unidad="AESU1",
                ),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        assert trades.shape[0] == 1
        assert int(trades[0, 3]) > 0

    def test_blank_leg_round_trip(self, tmp_path):
        """Metadata round-trips for the blank pair:
        ``metadata[trades[0, 3]] == ("", "")``."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="", buy_unidad="", buy_zona="",
                    sell_agente="AES", sell_unidad="AESU1",
                ),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        metadata = load_omie_agent_metadata(_config(tmp_path))
        buy_aid = int(trades[0, 3])
        assert metadata[buy_aid] == ("", "")

    def test_blank_sell_leg_pair_in_metadata(self, tmp_path):
        """Blank sell-side pair handled symmetrically to buy-side."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="AES", buy_unidad="AESU1",
                    sell_agente="", sell_unidad="", sell_zona="",
                ),
            ],
        )
        metadata = load_omie_agent_metadata(_config(tmp_path))
        pairs = set(metadata.values())
        assert ("", "") in pairs


# ===========================================================================
# 14. Domain integration — Portuguese pair, real EIC codes
# ===========================================================================


class TestPortugueseAndCrossZonePairs:
    """Portuguese rows and cross-zona trades integrate cleanly with the
    pair mapping."""

    def test_portuguese_pair_gets_agent_id(self, tmp_path):
        """An order row in Portugal zone with a PT Agente/Unidad pair
        gets a valid positive agent_id."""
        _write_orders(
            tmp_path,
            [
                _order_row(
                    contract_id=16, agente="PTAG", unidad="PTAGU01",
                    zona=_ZONE_PT,
                    momento="01/06/2025 17:41:12",
                ),
            ],
        )
        _write_trades(tmp_path, [])
        orders = load_omie_orders(_config(tmp_path))
        assert orders.shape[0] == 1
        assert int(orders[0, 5]) > 0

    def test_cross_zona_agent_ids_distinct(self, tmp_path):
        """In an ES-PT cross-zona trade, the two agent_id columns are
        positive and distinct."""
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="AES", buy_unidad="AESU1",
                    buy_zona=_ZONE_ES,
                    sell_agente="APT", sell_unidad="APTU1",
                    sell_zona=_ZONE_PT,
                ),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        assert trades.shape[0] == 1
        assert int(trades[0, 3]) > 0
        assert int(trades[0, 4]) > 0
        assert int(trades[0, 3]) != int(trades[0, 4])

    def test_trade_ids_consistent_with_metadata(self, tmp_path):
        """For a cross-zona trade, metadata[buy_aid] and metadata[sell_aid]
        round-trip to their source (Agente, Unidad) pairs."""
        from omiecid_simulator.jaxomiecid.omie_data_loader import (
            load_omie_agent_metadata,
        )
        _write_orders(tmp_path, [])
        _write_trades(
            tmp_path,
            [
                _trade_row(
                    contract_id=16,
                    buy_agente="AES", buy_unidad="AESU1",
                    buy_zona=_ZONE_ES,
                    sell_agente="APT", sell_unidad="APTU1",
                    sell_zona=_ZONE_PT,
                ),
            ],
        )
        trades = load_omie_trades(_config(tmp_path))
        metadata = load_omie_agent_metadata(_config(tmp_path))
        assert metadata[int(trades[0, 3])] == ("AES", "AESU1")
        assert metadata[int(trades[0, 4])] == ("APT", "APTU1")
