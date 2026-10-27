"""OMIE Continuous Intraday data loader.

Parses OMIE order files (orders_YYYYMMDD.1, semicolon-delimited,
Spanish comma decimals) and trade files (trades_YYYYMMDD.1) into
JAX arrays matching the 13-field OMIEOrder representation.

Implements the H1 replay heuristic (rest_until_gct): all orders rest
in the book until fully matched or GCT, with no cancellations.

Public API:
    load_omie_orders(config) -> jnp.ndarray of shape (N, 13)
    load_omie_trades(config) -> jnp.ndarray of shape (M, 6)
    load_omie_agent_metadata(config) -> dict[int, tuple[str, str]]
    IncompleteProductError (ValueError) — a product's session file is missing (OQ-29)
    replay_orders_through_book(orders, contract_book, historical_trades, heuristic) -> ReplayResult
    build_fixed_tape_set(configs=..., /, data_path=..., contract_ids=..., delivery_dates=...)
        -> jnp.ndarray of shape (n_tapes, N_orders_padded, 13)  [the Session's tape set]
    build_day_ahead_anchors(configs=..., /, data_path=..., contract_ids=..., delivery_dates=...)
        -> jnp.ndarray of shape (n_tapes,)  [per-tape day-ahead anchors, aligned to build_fixed_tape_set]
    ReplayResult dataclass

Agent identity note:
    Each OMIE ``agent_id`` is the 1-indexed integer assigned to a unique
    ``(Agente, Unidad)`` pair, not to the Agente alone. Two orders from
    the same Agente but different Unidad therefore get distinct
    ``agent_id`` values. The lossless reverse mapping is exposed via
    ``load_omie_agent_metadata(config)``. A Unidad is a bidding unit, so
    the pair is the finest identity the public files expose
    (market-spec §13.5).

Product-tape note:
    A ``(delivery_date D, contract_id k)`` tape is ONE delivery product: the
    rows whose ``Contrato`` start is (D, k), gathered from session files
    D-1 and D (the file date is the SESSION day). Timestamps are seconds
    since D-1 00:00 local wall clock, with no day wrap; real data lies in
    ``[55200, 165600)``. Incomplete products raise ``IncompleteProductError``;
    DST transition days raise unless ``allow_dst_transition=True``.

References:
    docs/omiecid/market-spec.md §13 (file format), §13.5 (Unidad semantics)
"""

import math
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import pandas as pd

from omiecid_simulator.jaxomiecid.omie_data_config import OMIEDataConfig
from omiecid_simulator.omiecid.order_types import (
    ExecRestriction,
    OMIEOrder,
    Side,
    Status,
    Validity,
    make_order,
)
from omiecid_simulator.omiecid.omie_config import OMIEMarketConfig, OMIEProductConfig
from omiecid_simulator.omiecid.encoding import (
    decode_ppd,
    decode_price,
    decode_qty,
    encode_ppd,
    encode_price,
    encode_qty,
)
import omiecid_simulator.jaxob.jaxob_constants as cst


def row_to_order(row, product_config: OMIEProductConfig) -> OMIEOrder:
    """Convert a single int32 13-column loader row to an :class:`OMIEOrder`.

    The loader emits tick-aligned int32 rows directly. This
    helper is a near-identity decode: price / quantity / peak_qty are
    offset-decoded on their respective tick grids; ``ppd`` is the exact
    inverse of :func:`encode_ppd` (sign-preserving, no offset).

    Column decoding:
      * ``price``    (col 2)  -> ``decode_price(int, pc)``  (EUR/MWh)
      * ``quantity`` (col 3)  -> ``decode_qty(int, pc)``    (MW)
      * ``peak_qty`` (col 11) -> ``decode_qty(int, pc)``    (MW)
      * ``ppd``      (col 12) -> ``decode_ppd(int, pc)``    (EUR/MWh)

    All other columns pass through as integers / IntEnum values.

    Args:
        row: A 13-element int32 row from ``load_omie_orders`` output
            (any sequence-like indexable by integer).
        product_config: The :class:`OMIEProductConfig` providing
            ``price_tick`` / ``qty_tick`` / ``price_min`` for decoding.

    Returns:
        An :class:`OMIEOrder` with EUR/MWh prices and MW quantities,
        exact on their respective tick grids.
    """
    return make_order(
        contract_id=int(row[0]),
        side=Side(int(row[1])),
        price=decode_price(int(row[2]), product_config),
        quantity=decode_qty(int(row[3]), product_config),
        order_id=int(row[4]),
        agent_id=int(row[5]),
        timestamp=int(row[6]),
        exec_restriction=ExecRestriction(int(row[7])),
        validity=Validity(int(row[8])),
        expiry_time=int(row[9]),
        status=Status(int(row[10])),
        peak_qty=decode_qty(int(row[11]), product_config),
        ppd=decode_ppd(int(row[12]), product_config),
    )


# ---------------------------------------------------------------------------
# ReplayResult
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    """Result of replaying historical orders through a ContractBook.

    Attributes:
        final_book: ContractBook state after all orders replayed. This is
            the terminal :class:`OMIEBookState` (the ``lax.scan`` carry),
            so it carries every state leaf — including the
            ``book_full`` full-side-eviction diagnostic.
        final_state: alias of ``final_book``. Surfaced explicitly so the
            terminal :class:`OMIEBookState` (and its ``book_full`` leaf) is
            observable end-to-end under the canonical ``final_state`` name
            consumers expect.
        replayed_trades: (K, 3) JAX array of trades produced during replay.
            Columns: [price, quantity, order_id_of_aggressor].
        historical_trades_matched: Count of historical trades reproduced.
        historical_trades_total: Count of historical trades in input data.
        extra_trades: Trades in replay but not in history.
        match_rate: historical_trades_matched / historical_trades_total.
        extra_volume_pct: Extra trade volume / total replayed volume.
    """
    final_book: object  # OMIEBookState — terminal scan carry
    replayed_trades: jnp.ndarray
    historical_trades_matched: int
    historical_trades_total: int
    extra_trades: int
    match_rate: float
    extra_volume_pct: float
    final_state: object = None  # alias of final_book


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_contract_id(contrato_str: str) -> int:
    """Parse OMIE contract string to contract_id (0-95).

    Format: "YYYYMMDD HH:MM-YYYYMMDD HH:MM"
    contract_id = start_hour * 4 + start_minute // 15

    Note: this drops the ``Contrato`` date. The loaders filter on the full
    product key via :func:`_parse_contract_key`.

    Args:
        contrato_str: Raw Contrato column value.

    Returns:
        Integer contract_id in range [0, 95].
    """
    # Extract start portion: "YYYYMMDD HH:MM"
    start_str = contrato_str.strip().split("-")[0].strip()
    # Parse time from "YYYYMMDD HH:MM"
    time_part = start_str.split(" ")[1]  # "HH:MM"
    hour, minute = int(time_part.split(":")[0]), int(time_part.split(":")[1])
    return hour * 4 + minute // 15


# ---------------------------------------------------------------------------
# One delivery product per tape
# ---------------------------------------------------------------------------
# A tape ``(D, k)`` is ONE delivery product: the rows whose ``Contrato`` START
# is (delivery date D, quarter-hour slot k). The file date of
# ``{orders,trades}_YYYYMMDD.1`` is the SESSION day, not the delivery day
# (market-spec v2.10 §13 / §13.4): session file D holds every row sent /
# matched in [22:20 D-1, 22:00 D), so product (D, k) is split across session
# files D-1 and D, and ONLY those two (the Iberian GCT is hour-aligned and at
# the latest 22:00 on D, §3.1; file D+1 opens at 22:20 on D and is never read).


class IncompleteProductError(ValueError):
    """A delivery product (D, k) is missing one of its required session files.

    Raised by :func:`load_omie_orders`, :func:`load_omie_trades` and
    :func:`load_omie_agent_metadata` when any of ``orders_{D-1}.1``,
    ``orders_{D}.1``, ``trades_{D-1}.1`` or ``trades_{D}.1`` is absent
    under ``config.data_path``. The check runs BEFORE any (tolerant) CSV
    parse, so a half-product is never returned silently. Incomplete products
    are excluded from the data set (open-questions OQ-29); callers such as
    :func:`build_fixed_tape_set` propagate this error rather than dropping
    the tape.
    """


class OMIENumberParseError(ValueError):
    """A numeric OMIE cell could not be parsed.

    Raised by :func:`_parse_float_comma` (and therefore by
    :func:`load_omie_orders`, :func:`load_omie_trades` and
    :func:`parse_marginalpdbc_file`) when a REQUIRED numeric cell (orders /
    trades ``Precio`` and ``Cantidad``, ``marginalpdbc`` col B) is empty,
    unparseable or non-finite, or when a NON-empty optional cell
    (``Cantidad reducida``, ``PPD``) is unparseable, or when a cell that
    contains ``,`` is not well-formed Spanish grouping. The message names the
    source, the column and the raw value. Like
    :class:`IncompleteProductError`, it propagates unwrapped through
    :func:`build_fixed_tape_set` and :func:`build_day_ahead_anchors`; no value
    is ever silently defaulted to 0.
    """


class OMIETimestampParseError(ValueError):
    """A trades match-time cell could not be parsed.

    Raised by :func:`load_omie_trades` (via :func:`_parse_match_timestamp`)
    when a ``Momento casacion`` cell is empty (including a file whose header
    lacks the column), unparseable, or a date-only cell whose date is not the
    product's delivery date D. Distinct from
    :class:`OMIENumberParseError`: a timestamp is not a numeric cell. Both
    subclass ``ValueError``. The message names the source, the column and the
    raw value. Orders tapes and agent metadata never parse the match time, so
    they stay loadable when this error would fire.
    """


def _parse_contract_key(contrato_str: str) -> "tuple[str, int]":
    """Parse an OMIE ``Contrato`` string to its product key ``(date_iso, slot)``.

    Format: ``"YYYYMMDD HH:MM-YYYYMMDD HH:MM"``. The key is the START of the
    delivery window: ``date_iso`` is ``"YYYY-MM-DD"`` and
    ``slot = start_hour * 4 + start_minute // 15`` (market-spec §13.4).

    Raises:
        ValueError: if the start time carries a DST ``A`` / ``B`` hour suffix
            (e.g. ``02:00A`` on a fall-back day). No slot mapping is defined
            for those contracts yet (open-questions OQ-30 (a)), so none is
            invented here.
    """
    start_str = contrato_str.strip().split("-")[0].strip()
    date_part, time_part = start_str.split(" ")[:2]
    if time_part[-1:] in ("A", "B"):
        raise ValueError(
            f"Contrato {contrato_str.strip()!r} carries a DST hour suffix "
            f"({time_part!r}); no slot mapping is defined for A/B-suffixed "
            "contracts (open-questions OQ-30 (a), unresolved)"
        )
    hour, minute = (int(x) for x in time_part.split(":")[:2])
    iso = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}"
    return iso, hour * 4 + minute // 15


def _is_dst_transition_day(delivery_date: str) -> bool:
    """True when ``delivery_date`` is a Europe/Madrid DST transition day.

    Day-level test: the UTC offset at D 00:00 differs from the one at
    D 23:59 (e.g. 2025-10-26 fall-back, 2026-03-29 spring-forward).
    """
    tz = ZoneInfo("Europe/Madrid")
    d = date.fromisoformat(delivery_date)
    start = datetime.combine(d, time(0, 0), tz).utcoffset()
    end = datetime.combine(d, time(23, 59), tz).utcoffset()
    return start != end


def _product_session_dates(delivery_date: str) -> "tuple[str, str]":
    """``(YYYYMMDD of D-1, YYYYMMDD of D)``: the two session files of product D."""
    d = date.fromisoformat(delivery_date)
    return (
        (d - timedelta(days=1)).strftime("%Y%m%d"),
        d.strftime("%Y%m%d"),
    )


def _require_complete_product(config: OMIEDataConfig) -> "tuple[list, list]":
    """Validate product (D, k) and return its session-file paths.

    Runs BEFORE any CSV parse (so a half-product is never parsed):

    1. DST guard: a Europe/Madrid transition day raises ``ValueError``
       unless ``config.allow_dst_transition`` is True (OQ-30).
    2. Completeness: all four files ``orders_{D-1}.1``, ``orders_{D}.1``,
       ``trades_{D-1}.1``, ``trades_{D}.1`` must exist, otherwise
       :class:`IncompleteProductError` names the first missing path (OQ-29).

    Returns:
        ``(orders_paths, trades_paths)``, each ``[file D-1, file D]``.
    """
    if _is_dst_transition_day(config.delivery_date) and not getattr(
        config, "allow_dst_transition", False
    ):
        raise ValueError(
            f"delivery date {config.delivery_date} is a Europe/Madrid DST "
            "transition day (open-questions OQ-30); set "
            "OMIEDataConfig.allow_dst_transition=True to opt in"
        )
    orders_paths, trades_paths = [], []
    for nodash in _product_session_dates(config.delivery_date):
        orders_paths.append(f"{config.data_path}/orders_{nodash}.1")
        trades_paths.append(f"{config.data_path}/trades_{nodash}.1")
    for path in orders_paths + trades_paths:
        if not os.path.exists(path):
            raise IncompleteProductError(
                f"incomplete product (delivery_date={config.delivery_date}, "
                f"contract_id={config.contract_id}): missing session file "
                f"{path} (products need files D-1 and D; OQ-29)"
            )
    return orders_paths, trades_paths


def _product_mask(contrato_col: "pd.Series", config: OMIEDataConfig) -> "pd.Series":
    """Boolean row mask of product ``(config.delivery_date, config.contract_id)``.

    Prefix-first: the ``Contrato`` start-date prefix is
    compared FIRST and the slot is parsed only on date-D rows. A session file
    also carries the neighbouring delivery day's rows, which on DST days use
    ``02:00A`` / ``02:00B`` strings the slot parser cannot (and must not) read.
    A date-D row with an A/B suffix raises (see :func:`_parse_contract_key`).
    """
    prefix = config.delivery_date.replace("-", "")
    stripped = contrato_col.astype(str).str.strip()
    on_day = stripped.str[:8] == prefix
    mask = pd.Series(False, index=contrato_col.index, dtype=bool)
    if on_day.any():
        target = (config.delivery_date, int(config.contract_id))
        keys = stripped[on_day].apply(_parse_contract_key)
        mask[on_day] = [key == target for key in keys]
    return mask


def _load_product_frames(
    config: OMIEDataConfig,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Parse the product's session files once; return ``(orders_df, trades_df)``.

    Both frames hold only product-(D, k) rows, file D-1 rows first, then file
    D rows, each in file order (no sort here). Either frame may be empty.
    Raises the DST / :class:`IncompleteProductError` checks of
    :func:`_require_complete_product` before any parse.
    """
    orders_paths, trades_paths = _require_complete_product(config)

    o_frames = []
    for path in orders_paths:
        df = _parse_orders_csv(path)
        if not df.empty and "Contrato" in df.columns:
            o_frames.append(df[_product_mask(df["Contrato"], config)])
    t_frames = []
    for path in trades_paths:
        df = _read_trades_df(path)
        if df is not None and not df.empty and "Contrato" in df.columns:
            t_frames.append(df[_product_mask(df["Contrato"], config)])

    df_o = pd.concat(o_frames, ignore_index=True) if o_frames else pd.DataFrame()
    df_t = pd.concat(t_frames, ignore_index=True) if t_frames else pd.DataFrame()
    return df_o, df_t


def _compute_epoch_base(delivery_date: str) -> int:
    """Compute the UTC midnight Unix timestamp for a delivery date.

    Args:
        delivery_date: ISO-format date string "YYYY-MM-DD".

    Returns:
        Integer Unix timestamp (seconds since 1970-01-01 UTC) for
        UTC midnight of the given delivery date.
    """
    dt = datetime.fromisoformat(delivery_date).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _parse_timestamp(momento_str: str, epoch_base: int) -> int:
    """Parse an OMIE send / match timestamp to signed seconds from D 00:00.

    Format: "DD/MM/YYYY HH:MM:SS" (local wall clock).

    Args:
        momento_str: Raw Momento de envio / Momento casacion column value.
        epoch_base: midnight of the delivery date D, as returned by
            ``_compute_epoch_base``. Both sides are read as naive wall-clock
            times, so the difference is wall-clock seconds.

    Returns:
        Signed integer seconds from D 00:00: negative for sends on D-1
        (a product's sends start at 15:20 D-1, i.e. -31,200), in
        ``[0, 86400)`` for sends on D. The loaders store this value
        ``+ 86400`` (seconds since D-1 00:00, NO modulo).
    """
    dt = datetime.strptime(momento_str.strip(), "%d/%m/%Y %H:%M:%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp()) - epoch_base


def _build_agent_mapping(
    agente_series: "pd.Series",
    unidad_series: "pd.Series",
) -> "tuple[dict, dict]":
    """Build deterministic ``(Agente, Unidad)`` pair -> int mapping.

    Each unique ``(agente, unidad)`` pair is assigned a positive 1-indexed
    integer ``agent_id``. Ids are deterministic and stable across runs
    because the enumeration uses Python's canonical tuple sort on the
    unique pair set (element-wise Unicode code-point order, locale- and
    PYTHONHASHSEED-independent).

    This is the pair identity model:
    every Unidad of the same Agente gets a distinct ``agent_id``, and the
    reverse dict is lossless so settlement / diagnostics can recover the
    source pair from any id.

    Args:
        agente_series: Pandas Series of Agente (participant) codes.
        unidad_series: Pandas Series of Unidad (bidding-unit) codes.
            Must be the same length as ``agente_series``; rows are paired
            positionally.

    Returns:
        ``(pair_to_id, id_to_pair)`` — a symmetric bijection:
        - ``pair_to_id``: ``{(agente, unidad): agent_id}``.
        - ``id_to_pair``: ``{agent_id: (agente, unidad)}``.
        Both dicts have the same cardinality; ids form ``[1, N]``.

    Notes:
        - Sort key: the ``(agente, unidad)`` tuple itself. No string
          concatenation (would be separator-dependent), no locale-aware
          comparison (`locale.strcoll` forbidden), so ids are identical
          on every machine.
        - Blank-leg pairs such as ``("", "")`` (XBID cross-border
          counterparties) are legitimate entries of the universe — they
          sort to the front and receive a positive ``agent_id`` like any
          other pair.
    """
    if len(agente_series) != len(unidad_series):
        raise ValueError(
            "_build_agent_mapping: agente_series and unidad_series must "
            f"have equal length (got {len(agente_series)} vs "
            f"{len(unidad_series)})"
        )

    # Zip positionally; unique pairs via a dict-as-ordered-set keyed on the
    # tuple. We still sort deterministically below; the dict step just
    # collapses duplicates without relying on Python set iteration order.
    unique_pairs = {
        (str(a), str(u))
        for a, u in zip(agente_series, unidad_series)
    }
    sorted_pairs = sorted(unique_pairs)  # canonical element-wise tuple sort
    pair_to_id = {pair: idx + 1 for idx, pair in enumerate(sorted_pairs)}
    id_to_pair = {idx + 1: pair for idx, pair in enumerate(sorted_pairs)}
    return pair_to_id, id_to_pair


def _seconds_since_prev_midnight(momento_str: str, epoch_base: int) -> int:
    """Seconds since D-1 00:00 local wall clock, NO modulo.

    ``_parse_timestamp`` (seconds from D 00:00) shifted by one day.
    """
    return _parse_timestamp(momento_str, epoch_base) + 86400


_DATE_ONLY_RE = re.compile(r"\d{2}/\d{2}/\d{4}")


def _parse_match_timestamp(val, epoch_base: int, *, source: str) -> int:
    """Parse one trades ``Momento casacion`` cell to seconds since D-1 00:00.

    Trades only; the orders send-time parse is separate.

    * An empty cell (``None`` / NaN / ``''`` / ``'nan'``; also a column that
      is absent from the file) raises.
    * A date-only cell ``DD/MM/YYYY`` is midnight of that date (the trades
      exporter drops a ``00:00:00`` time part). It is accepted ONLY when the
      date equals the delivery date D, and then yields exactly 86,400; any
      other date raises.
    * Otherwise the cell is ``DD/MM/YYYY HH:MM:SS`` and goes through
      :func:`_seconds_since_prev_midnight`; a parse failure raises.

    Args:
        val: raw cell value.
        epoch_base: UTC-midnight timestamp of D (``_compute_epoch_base``).
        source: source description (product + data path), for the message.

    Raises:
        OMIETimestampParseError: as described above. There is no default.
    """
    column = "Momento casacion"

    def _fail(reason: str) -> "OMIETimestampParseError":
        return OMIETimestampParseError(
            f"{source}: column {column!r}: {reason} match-time cell {str(val)!r}"
        )

    if _is_missing_cell(val):
        raise _fail("empty")
    s = str(val).strip()
    if _DATE_ONLY_RE.fullmatch(s):
        try:
            cell_date = datetime.strptime(s, "%d/%m/%Y").date()
        except ValueError:
            raise _fail("unparseable") from None
        delivery = datetime.fromtimestamp(epoch_base, tz=timezone.utc).date()
        if cell_date != delivery:
            raise _fail("date-only (not the delivery date D)")
        return 86400
    try:
        return _seconds_since_prev_midnight(s, epoch_base)
    except (ValueError, TypeError):
        raise _fail("unparseable") from None


def _normalize_cell(val) -> str:
    """Normalize an OMIE Agente / Unidad cell to its canonical string form.

    Missing cells arrive as ``None``, ``NaN``, or the literal string
    ``"nan"`` depending on the parsing path; they all collapse to the
    empty string so blank-leg XBID counterparties map onto the single
    ``("", "")`` pair.
    """
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


def _pair_universe_from_frames(
    df_o: "pd.DataFrame", df_t: "pd.DataFrame"
) -> "tuple[dict, dict]":
    """Pair universe over already-filtered product frames (orders + trades)."""
    agente_values: list[str] = []
    unidad_values: list[str] = []

    def _extend(ag_col, un_col):
        agente_values.extend(_normalize_cell(v) for v in ag_col.tolist())
        unidad_values.extend(_normalize_cell(v) for v in un_col.tolist())

    if not df_o.empty and "Agente" in df_o.columns and "Unidad" in df_o.columns:
        _extend(df_o["Agente"], df_o["Unidad"])

    if not df_t.empty:
        if "Agente compra" in df_t.columns and "Unidad compra" in df_t.columns:
            _extend(df_t["Agente compra"], df_t["Unidad compra"])
        if "Agente venta" in df_t.columns and "Unidad venta" in df_t.columns:
            _extend(df_t["Agente venta"], df_t["Unidad venta"])

    return _build_agent_mapping(
        pd.Series(agente_values, dtype=object),
        pd.Series(unidad_values, dtype=object),
    )


def _collect_pair_universe(
    config: OMIEDataConfig,
) -> "tuple[dict, dict]":
    """Build the ``(Agente, Unidad)`` pair universe for a given config.

    The universe is the union over the rows of product
    ``(config.delivery_date, config.contract_id)`` in session files D-1 and
    D:
      - every order row's ``(Agente, Unidad)`` pair, and
      - every trade row's ``(Agente compra, Unidad compra)`` and
        ``(Agente venta, Unidad venta)`` pairs.

    Used by both ``load_omie_orders`` / ``load_omie_trades`` (so the
    two loaders agree on agent_id values) and by
    ``load_omie_agent_metadata`` (the public session-metadata accessor).

    Args:
        config: OMIEDataConfig (contract_id, delivery_date, data_path).

    Returns:
        ``(pair_to_id, id_to_pair)`` — see ``_build_agent_mapping``.

    Raises:
        IncompleteProductError: if any of the four session files is missing.
        ValueError: on a DST transition day without the opt-in (OQ-30).
    """
    df_o, df_t = _load_product_frames(config)
    return _pair_universe_from_frames(df_o, df_t)


def load_omie_agent_metadata(
    config: OMIEDataConfig,
) -> "dict[int, tuple[str, str]]":
    """Return the canonical reverse mapping ``{agent_id: (agente, unidad)}``.

    This is the lossless session metadata for an OMIE delivery / contract
    pair. Every ``agent_id`` that appears in the ``(N, 13)`` orders array
    or in columns 3 / 4 of the ``(M, 6)`` trades array is a key here, and
    the value is the source ``(Agente, Unidad)`` pair.

    The pair universe is the **union** over order and trade rows, so
    trade-only pairs (e.g., pairs appearing only on the trade side due to
    post-hoc reporting) are also included. Blank-leg XBID pairs
    ``("", "")`` are legitimate entries — see ``_build_agent_mapping``.

    Callers that need the parent-Agente-only view derive it in one line
    at the call site::

        id_to_agente = {aid: pair[0] for aid, pair in id_to_pair.items()}

    No separate ``id_to_agente`` artifact is persisted by this module
    (the pair mapping is the single source of truth).

    Args:
        config: OMIEDataConfig pointing at a specific delivery / contract.

    Returns:
        ``{agent_id: (agente, unidad)}`` dict. Keys form ``[1, N]``.

    Raises:
        IncompleteProductError / ValueError: as :func:`load_omie_orders`
            (the universe is taken over product (D, k) in files D-1 and D).
    """
    _, id_to_pair = _collect_pair_universe(config)
    return id_to_pair


def _map_exec_restriction(val) -> int:
    """Map OMIE execution restriction string to IntEnum value.

    Empty/NaN -> NON (0), "IOC" -> 1, "FOK" -> 2, "ICE" -> 3.
    """
    if pd.isna(val) or str(val).strip() == "":
        return int(ExecRestriction.NON)
    s = str(val).strip().upper()
    mapping = {"IOC": int(ExecRestriction.IOC),
               "FOK": int(ExecRestriction.FOK),
               "ICE": int(ExecRestriction.ICE)}
    return mapping.get(s, int(ExecRestriction.NON))


def _map_validity(val) -> int:
    """Map OMIE validity condition string to IntEnum value.

    Empty/NaN -> GFS (0), "GTD" -> 1.
    """
    if pd.isna(val) or str(val).strip() == "":
        return int(Validity.GFS)
    s = str(val).strip().upper()
    if s == "GTD":
        return int(Validity.GTD)
    return int(Validity.GFS)


# Well-formed Spanish number when the cell contains ``,``:
# optional sign, then either dot-grouped thousands or plain digits, then a
# comma decimal part.
_SPANISH_NUMBER_RE = re.compile(r"[+-]?(\d{1,3}(\.\d{3})+|\d+),\d+")


def _is_missing_cell(val) -> bool:
    """True for ``None``, a float NaN, ``''`` and the literal ``'nan'``."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    s = str(val).strip()
    return s == "" or s.lower() == "nan"


def _parse_float_comma(val, *, column: str, source: str) -> float:
    """Strictly parse one numeric OMIE cell to a finite float.

    Number rule:

    * a cell that contains ``,`` is Spanish format (``.`` = thousands
      separator, ``,`` = decimal). It must fully match
      ``[+-]?(\\d{1,3}(\\.\\d{3})+|\\d+),\\d+``; every ``.`` is removed and
      the ``,`` becomes the decimal point (``'-9.999,00'`` -> -9999.0);
    * a cell WITHOUT ``,`` is parsed as-is by ``float`` (``'12.5'`` -> 12.5;
      the dot-decimal ``marginalpdbc`` file, ``'36.07'`` -> 36.07).

    There is no default: an empty (``None`` / NaN / ``''`` / ``'nan'``),
    unparseable or non-finite cell raises.

    Args:
        val: raw cell value.
        column: column name, for the error message.
        source: source description (file / product), for the error message.

    Raises:
        OMIENumberParseError: as described above.
    """
    def _fail(reason: str) -> "OMIENumberParseError":
        return OMIENumberParseError(
            f"{source}: column {column!r}: {reason} numeric cell {str(val)!r}"
        )

    if _is_missing_cell(val):
        raise _fail("empty")
    s = str(val).strip()
    if "," in s:
        if _SPANISH_NUMBER_RE.fullmatch(s) is None:
            raise _fail("malformed Spanish-format")
        s = s.replace(".", "").replace(",", ".")
    try:
        out = float(s)
    except (ValueError, TypeError):
        raise _fail("unparseable") from None
    if not math.isfinite(out):
        raise _fail("non-finite")
    return out


def _parse_optional_float(val, default: float = 0.0, *, column: str, source: str) -> float:
    """Parse an optional numeric cell (peak_qty / ppd) from an OMIE row.

    Missing values arrive as ``None``, ``NaN``, the literal string
    ``"nan"``, or the empty string. All of these collapse to ``default``;
    any other value is delegated to the strict :func:`_parse_float_comma`
    (so non-empty garbage raises :class:`OMIENumberParseError`).
    """
    if _is_missing_cell(val):
        return default
    return _parse_float_comma(val, column=column, source=source)


# ---------------------------------------------------------------------------
# load_omie_orders
# ---------------------------------------------------------------------------

def _parse_orders_csv(filepath: str) -> pd.DataFrame:
    """Parse OMIE orders CSV into a DataFrame with canonical column names.

    Handles variable-width rows (some have extra empty fields) by reading
    without a header and mapping columns from known fixed positions.

    OMIE orders files have:
      - Row 1: metadata boilerplate (skipped)
      - Row 2: column headers (skipped; we assign our own)
      - Rows 3+: data, semicolon-delimited with trailing semicolon

    The first 8 columns are always fixed:
      0:Fecha, 1:Contrato, 2:Zona, 3:Agente, 4:Unidad,
      5:Precio, 6:Cantidad, 7:Tipo oferta

    The remaining columns (exec restriction, validity, peak qty, ppd,
    timestamp) may be shifted by extra empty fields in some rows. We
    find the timestamp (last non-empty field with date pattern) and
    work backwards.

    Args:
        filepath: Path to the orders_YYYYMMDD.1 file.

    Returns:
        DataFrame with canonical column names.
    """
    # Read raw lines and locate the column-header row.  Real OMIE files
    # interleave an empty line between the metadata boilerplate and the
    # column-header row, while synthetic test fixtures write them back-
    # to-back.  Scan for the canonical "Fecha;Contrato;" header and treat
    # every line after it as data.  If no header is found, fall back to
    # the legacy "skip first two lines" heuristic.
    empty_columns = [
        "Fecha", "Contrato", "Zona", "Agente", "Unidad",
        "Precio", "Cantidad", "Tipo oferta",
        "Condiciones a la ejecucion", "Condiciones a la validez",
        "Cantidad reducida", "PPD", "Momento de envio",
    ]

    with open(filepath, encoding="latin-1") as f:
        all_lines = f.readlines()

    header_idx = None
    for idx, line in enumerate(all_lines):
        if line.lstrip().startswith("Fecha;Contrato;"):
            header_idx = idx
            break

    if header_idx is None:
        if len(all_lines) < 3:
            return pd.DataFrame(columns=empty_columns)
        data_lines = all_lines[2:]
    else:
        data_lines = all_lines[header_idx + 1:]

    if not data_lines:
        return pd.DataFrame(columns=empty_columns)

    records = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        fields = line.split(";")
        # Remove trailing empty field from trailing semicolon
        while fields and fields[-1].strip() == "":
            fields.pop()
        if len(fields) < 8:
            continue

        # Fixed columns 0-7
        fecha = fields[0].strip()
        contrato = fields[1].strip()
        zona = fields[2].strip()
        agente = fields[3].strip()
        unidad = fields[4].strip()
        precio = fields[5].strip()
        cantidad = fields[6].strip()
        tipo_oferta = fields[7].strip()

        # The last field is always Momento de envio (timestamp)
        momento = fields[-1].strip() if len(fields) > 8 else ""

        # Between fields[8] and fields[-2] are the optional fields:
        # Condiciones a la ejecucion, Condiciones a la validez,
        # Cantidad reducida, PPD. Strip whitespace and drop empty
        # slots in a single pass; the non-empty values are mapped
        # positionally below.
        middle = fields[8:-1] if len(fields) > 9 else []
        non_empty = [v for v in (f.strip() for f in middle) if v != ""]

        cond_exec = ""
        cond_validez = ""
        cant_reducida = ""
        ppd = ""

        if non_empty:
            # First non-empty middle field: check if it's an exec restriction
            first = non_empty[0].upper()
            if first in ("IOC", "FOK", "ICE"):
                cond_exec = non_empty[0]
                remaining = non_empty[1:]
            elif first == "GTD":
                cond_validez = non_empty[0]
                remaining = non_empty[1:]
            else:
                # Could be a numeric value (peak_qty or ppd)
                remaining = non_empty
        else:
            remaining = []

        # After exec restriction, check for validity
        if remaining and cond_validez == "" and remaining[0].upper() == "GTD":
            cond_validez = remaining[0]
            remaining = remaining[1:]

        # Remaining values are peak_qty and ppd (in order)
        if remaining:
            cant_reducida = remaining[0]
            remaining = remaining[1:]
        if remaining:
            ppd = remaining[0]

        records.append({
            "Fecha": fecha,
            "Contrato": contrato,
            "Zona": zona,
            "Agente": agente,
            "Unidad": unidad,
            "Precio": precio,
            "Cantidad": cantidad,
            "Tipo oferta": tipo_oferta,
            "Condiciones a la ejecucion": cond_exec,
            "Condiciones a la validez": cond_validez,
            "Cantidad reducida": cant_reducida,
            "PPD": ppd,
            "Momento de envio": momento,
        })

    return pd.DataFrame(records)


def _normalize_header(name) -> str:
    """Strip and accent-strip one CSV header name (NFKD, combining marks dropped)."""
    decomposed = unicodedata.normalize("NFKD", str(name).strip())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _read_trades_df(filepath: str) -> "pd.DataFrame | None":
    """Read an OMIE trades CSV into a DataFrame with canonical columns.

    Returns ``None`` if the file does not exist. Returns an empty
    DataFrame (or one missing the ``Contrato`` column) if the file exists
    but is otherwise empty / malformed — callers must guard.
    """
    col_names = [
        "Fecha", "Contrato", "Agente compra", "Unidad compra", "Zona compra",
        "Agente venta", "Unidad venta", "Zona venta",
        "Precio", "Cantidad", "Momento casacion",
    ]

    try:
        df = pd.read_csv(
            filepath,
            sep=";",
            encoding="latin-1",
            skiprows=1,
            decimal=",",
            dtype=str,
            on_bad_lines="skip",
            header=0,
        )
    except FileNotFoundError:
        return None

    # Accent-strip HEADER NAMES only,
    # so the real ``Momento casación`` matches the ASCII canonical name.
    # Cell values (agent / unit codes, Contrato) are never touched.
    df.columns = [_normalize_header(c) for c in df.columns]
    known_cols = [c for c in col_names if c in df.columns]
    df = df[known_cols].copy()
    df = df.dropna(how="all")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    return df


def load_omie_orders(config: OMIEDataConfig) -> jnp.ndarray:
    """Load the orders of ONE delivery product (D, k) as an (N, 13) array.

    Product ``(D, k) = (config.delivery_date, config.contract_id)`` is the
    set of rows whose ``Contrato`` START is (D, slot k). Its rows are read
    from session files ``orders_{D-1}.1`` and ``orders_{D}.1`` ONLY
    (market-spec v2.10 §13.4): the two frames
    are concatenated in file order, filtered prefix-first by the product key,
    then STABLE-sorted by timestamp (same-second ties keep file order, file
    D-1 first); ``order_id`` is ``1..N`` in that order.

    Per market-spec.md §13.4, the ``Zona`` column is parsed from the CSV
    but not used for filtering: every zone (Spain, Portugal, and any other
    published area code) is included.

    ``agent_id`` values are assigned by concatenating each row's
    ``(Agente, Unidad)`` pair into a single 1-indexed integer via
    ``_build_agent_mapping``. The pair universe is shared with
    ``load_omie_trades`` (union over orders + trades — see
    ``_collect_pair_universe``), so the two loaders always agree on
    agent_id assignments for the same config.

    The 13 columns correspond to OMIEOrder fields:
        [0] contract_id, [1] side, [2] price, [3] quantity, [4] order_id,
        [5] agent_id, [6] timestamp, [7] exec_restriction, [8] validity,
        [9] expiry_time, [10] status, [11] peak_qty, [12] ppd

    Args:
        config: OMIEDataConfig with data_path, contract_id, delivery_date.

    Returns:
        JAX int32 array of shape (N, 13) with every numeric column
        tick-encoded (price / quantity / peak_qty offset-encoded; ppd
        signed int-ticks). Timestamp (col 6) = seconds since D-1 00:00
        local wall clock, NO modulo; real data lies in ``[55200, 165600)``
        (15:20 D-1 .. the latest GCT, 22:00 D). The loader does not
        validate or clip that window. Empty result has shape (0, 13).

    Raises:
        IncompleteProductError: if any of ``orders_{D-1}.1``,
            ``orders_{D}.1``, ``trades_{D-1}.1``, ``trades_{D}.1`` is
            missing (OQ-29), checked before any parse.
        ValueError: if D is a Europe/Madrid DST transition day and
            ``config.allow_dst_transition`` is False, or (under the opt-in)
            a date-D ``Contrato`` carries an A/B hour suffix (OQ-30).
    """
    product_config = OMIEProductConfig()

    # Steps 1-2: completeness + DST guard, parse files D-1 and D once,
    # prefix-first product filter (every zone retained per market-spec §13.4).
    df, df_t = _load_product_frames(config)

    if df.empty:
        return jnp.zeros((0, 13), dtype=jnp.int32)
    df = df.copy()
    contract_id = int(config.contract_id)

    # Step 3: seconds since D-1 00:00 (no modulo), stable chronological sort.
    epoch_base = _compute_epoch_base(config.delivery_date)
    df["_timestamp"] = df["Momento de envio"].apply(
        lambda s: _seconds_since_prev_midnight(s, epoch_base)
    )
    df = df.sort_values("_timestamp", kind="stable").reset_index(drop=True)

    # Step 4: Assign sequential order_id (starting at 1)
    df["_order_id"] = range(1, len(df) + 1)

    # Step 5: Build the shared (Agente, Unidad) → agent_id mapping across
    # both orders and trades of the product (union over files D-1 and D).
    pair_to_id, _ = _pair_universe_from_frames(df, df_t)
    df["_agent_id"] = df.apply(
        lambda r: pair_to_id[
            (_normalize_cell(r["Agente"]), _normalize_cell(r["Unidad"]))
        ],
        axis=1,
    )

    # Step 6: Map all fields to the 13-column int32 order representation.
    # Price / quantity / peak_qty are offset-encoded on their tick grids;
    # ppd is signed int-ticks on price_tick (see encoding.encode_ppd).
    src = (
        f"orders of product ({config.delivery_date}, {contract_id}) "
        f"under {config.data_path}"
    )
    rows = []
    for _, row in df.iterrows():
        side = int(Side.BUY) if str(row["Tipo oferta"]).strip().upper() == "C" else int(Side.SELL)
        price_eur = _parse_float_comma(row["Precio"], column="Precio", source=src)
        qty_mw = _parse_float_comma(row["Cantidad"], column="Cantidad", source=src)
        order_id = int(row["_order_id"])
        agent_id = int(row["_agent_id"])
        # Seconds since D-1 00:00 local wall clock, NO modulo.
        # Real values are < 200,000,
        # so the int32 / float32-exact loader contract still holds.
        timestamp = int(row["_timestamp"])
        exec_restriction = _map_exec_restriction(row["Condiciones a la ejecucion"])
        validity = _map_validity(row["Condiciones a la validez"])
        expiry_time = 0
        status = int(Status.ACTIVE)
        peak_qty_mw = _parse_optional_float(
            row["Cantidad reducida"], column="Cantidad reducida", source=src
        )
        ppd_eur = _parse_optional_float(row["PPD"], column="PPD", source=src)

        price_int = encode_price(price_eur, product_config)
        qty_int = encode_qty(qty_mw, product_config)
        peak_qty_int = encode_qty(peak_qty_mw, product_config)
        ppd_int = encode_ppd(ppd_eur, product_config)

        rows.append([
            contract_id, side, price_int, qty_int, order_id, agent_id,
            timestamp, exec_restriction, validity, expiry_time, status,
            peak_qty_int, ppd_int,
        ])

    # `df` is non-empty after the filter above, so `rows` always has at
    # least one entry here.
    return jnp.array(rows, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# load_omie_trades
# ---------------------------------------------------------------------------

def load_omie_trades(config: OMIEDataConfig) -> jnp.ndarray:
    """Load the trades of ONE delivery product (D, k) as an (M, 6) array.

    Same product definition as :func:`load_omie_orders`: rows whose
    ``Contrato`` START is (D, slot k), from ``trades_{D-1}.1`` then
    ``trades_{D}.1``, concatenated in FILE ORDER with NO sort.

    Per market-spec.md §13.4, the ``Zona compra`` / ``Zona venta`` columns
    are parsed from the CSV but not used for filtering: every trade is
    included regardless of buy-/sell-leg zone.

    ``buy_agent_id`` and ``sell_agent_id`` (columns 3 and 4) are the
    1-indexed integers produced by the shared ``(Agente, Unidad)`` pair
    mapping (union over orders + trades, see ``_collect_pair_universe``).
    Blank-leg XBID cross-border counterparties are mapped to the
    ``("", "")`` pair's agent_id (a legitimate, positive value — see
    ``_build_agent_mapping``), not silently to ``0``.

    The 6 columns are:
        [0] contract_id, [1] price, [2] quantity,
        [3] buy_agent_id, [4] sell_agent_id, [5] match_timestamp

    Args:
        config: OMIEDataConfig with data_path, contract_id, delivery_date.

    Returns:
        JAX int32 array of shape (M, 6) with tick-encoded price /
        quantity columns. Match timestamp (col 5) = seconds since D-1
        00:00, NO modulo, in the per-product tape frame:
        header names are accent-stripped, so the real ``Momento casación``
        is read; a date-only cell equal to D is midnight D (86,400). There
        is no default value. Empty result has shape (0, 6).

    Raises:
        IncompleteProductError / ValueError: as :func:`load_omie_orders`.
        OMIETimestampParseError: a match-time cell is empty / missing,
            unparseable, or date-only for a date other than D.
    """
    product_config = OMIEProductConfig()

    df_o, df = _load_product_frames(config)
    if df.empty:
        return jnp.zeros((0, 6), dtype=jnp.int32)
    contract_id = int(config.contract_id)

    # Shared (Agente, Unidad) → agent_id mapping (union over orders +
    # trades of the product).
    pair_to_id, _ = _pair_universe_from_frames(df_o, df)

    # Compute epoch_base once for session-relative timestamp conversion.
    epoch_base = _compute_epoch_base(config.delivery_date)

    src = (
        f"trades of product ({config.delivery_date}, {contract_id}) "
        f"under {config.data_path}"
    )
    rows = []
    for _, row in df.iterrows():
        price_eur = _parse_float_comma(
            row.get("Precio", "0"), column="Precio", source=src
        )
        qty_mw = _parse_float_comma(
            row.get("Cantidad", "0"), column="Cantidad", source=src
        )
        buy_key = (
            _normalize_cell(row.get("Agente compra", "")),
            _normalize_cell(row.get("Unidad compra", "")),
        )
        sell_key = (
            _normalize_cell(row.get("Agente venta", "")),
            _normalize_cell(row.get("Unidad venta", "")),
        )
        buy_agent_id = pair_to_id.get(buy_key, 0)
        sell_agent_id = pair_to_id.get(sell_key, 0)

        ts = _parse_match_timestamp(
            row.get("Momento casacion", ""), epoch_base, source=src
        )

        price_int = encode_price(price_eur, product_config)
        qty_int = encode_qty(qty_mw, product_config)

        rows.append([contract_id, price_int, qty_int, buy_agent_id, sell_agent_id, ts])

    # `df` is non-empty after the filter above, so `rows` always has at
    # least one entry here.
    return jnp.array(rows, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# replay_orders_through_book
# ---------------------------------------------------------------------------
#
# A single ``lax.scan`` over the orders tensor (no per-row Python ``for``
# loop), lifted to the module-level jit shape
# ``_replay_orders = jax.jit(_replay_orders_body,
# static_argnames=("product_config", "market_config"))`` — the same
# idiom as ``_submit_order`` in ``contract_book.py``.
#
# Design notes:
#   - The scan body invokes the lifted ``_submit_order`` (the body, not
#     the trampoline) — the trampoline's Python-side ``contract_open``
#     guard cannot run under trace and is replicated as a closed-over
#     Python-time scalar in ``replay_orders_through_book``.
#   - ``_row_to_order_traced`` converts a traced ``Int32[13]`` row to an
#     ``OMIEOrder`` whose fields are traced 0-d ``jnp.*`` scalars; the
#     existing eager ``row_to_order`` stays in place for non-replay
#     callers (diagnostic tooling, demos).
#   - Per-step new-fill detection uses a ring-buffer-immune
#     whole-row diff mask (an IOC fill leaves no resting order, so it
#     must be read from the trades buffer; market-spec §4.5).
#   - Per-step replayed-trades buffer is a fixed-shape ``(NTRADE_CAP, 3)``
#     int32 slab with column 0 = aggressor order_id (sentinel ``-1`` in
#     unfilled slots), columns 1, 2 = jaxob trade row's int32 price and
#     absolute quantity ticks. Final scan output has shape
#     ``(N_orders, NTRADE_CAP, 3)``.

def _row_to_order_traced(
    row: jnp.ndarray,
    product_config: OMIEProductConfig,
) -> OMIEOrder:
    """Trace-safe sibling of :func:`row_to_order`.

    Consumes a traced ``Int32[13]`` row and returns an ``OMIEOrder``
    whose fields are 0-d ``jnp.*`` scalars. Price / quantity / peak_qty /
    ppd are decoded back to ``float32`` (via the shared ``decode_*``
    helpers) so that ``_submit_order_body`` re-encodes them through its
    JIT-safe round-to-tick path; the integer enum fields are passed through
    as ``int32`` scalars and are dispatched by ``jax.lax.switch`` inside
    the submit body.
    """
    # Row is already int32 (loader output contract: Int32[13]); cast to
    # float32 first so the decode_* helpers' Python-float arithmetic
    # produces a float32 traced scalar under jax_enable_x64=False.
    return OMIEOrder(
        contract_id=row[0],
        side=row[1],
        price=decode_price(row[2].astype(jnp.float32), product_config),
        quantity=decode_qty(row[3].astype(jnp.float32), product_config),
        order_id=row[4],
        agent_id=row[5],
        timestamp=row[6],
        exec_restriction=row[7],
        validity=row[8],
        expiry_time=row[9],
        status=row[10],
        peak_qty=decode_qty(row[11].astype(jnp.float32), product_config),
        ppd=decode_ppd(row[12].astype(jnp.float32), product_config),
    )


def _replay_orders_body(
    state,
    orders: jnp.ndarray,
    product_config: OMIEProductConfig,
    market_config: OMIEMarketConfig,
):
    """Scan body for :data:`_replay_orders`.

    Threads ``OMIEBookState`` as the scan carry and emits a fixed-shape
    ``(NTRADE_CAP, 3) int32`` per-step slab of new-fill rows. Per-step
    layout: column 0 = aggressor ``order_id`` (sentinel ``-1`` in
    unfilled slots), column 1 = jaxob trade row's int32 price tick,
    column 2 = absolute int32 quantity tick.

    Args:
        state: initial :class:`omiecid_simulator.omiecid.contract_book.OMIEBookState`.
        orders: ``(N_orders, 13) int32`` orders tensor (loader output).
        product_config: static :class:`OMIEProductConfig`.
        market_config: static :class:`OMIEMarketConfig`.

    Returns:
        ``(final_state, replayed_trades_buf)`` where
        ``replayed_trades_buf`` has shape ``(N_orders, NTRADE_CAP, 3)``
        and dtype ``int32``.
    """
    # Import inside the body to avoid an import cycle: omie_data_loader is
    # imported very early by some test fixtures, before contract_book has
    # finished initializing its IcebergManager re-export. The cost is one
    # Python-time dict lookup at trace time — not in the hot path.
    from omiecid_simulator.omiecid.contract_book import _submit_order

    P_COL = cst.TradesFeat.P.value
    Q_COL = cst.TradesFeat.Q.value
    NTRADE_CAP = cst.NTRADE_CAP

    def scan_step(carry_state, row):
        order = _row_to_order_traced(row, product_config)
        trades_before = carry_state.lob.trades

        new_state, _result = _submit_order(
            carry_state, order, product_config, market_config,
        )
        trades_after = new_state.lob.trades

        # Ring-buffer-immune new-fill detection (whole-row diff, not a
        # sentinel-transition mask, which misses overwritten slots).
        new_mask = jnp.any(trades_before != trades_after, axis=1)  # (nTrades,) bool
        aggressor_oid = order.order_id.astype(jnp.int32)

        # Per-step slab. Column 0 = aggressor order_id (sentinel -1 in
        # unfilled slots), col 1 = price tick, col 2 = abs(qty) tick.
        # Unfilled slabs are content-deterministic (sentinel rows are
        # (-1, 0, 0)) — the ``jnp.where`` masks zero out price / qty
        # cells in non-fill slots in the same pass.
        oid_col = jnp.where(new_mask, aggressor_oid, jnp.int32(-1))
        price_col = jnp.where(
            new_mask, trades_after[:, P_COL].astype(jnp.int32), jnp.int32(0),
        )
        qty_col = jnp.where(
            new_mask, jnp.abs(trades_after[:, Q_COL]).astype(jnp.int32), jnp.int32(0),
        )
        step_slab = jnp.stack([oid_col, price_col, qty_col], axis=1)  # (NTRADE_CAP, 3)

        return new_state, step_slab

    # If the orders tensor is empty (N_orders=0), lax.scan over a 0-length
    # leading axis still returns the empty stacked output with the right
    # trailing shape. No special-casing required.
    final_state, replayed_trades_buf = jax.lax.scan(
        scan_step, state, orders,
    )
    # replayed_trades_buf has shape (N_orders, NTRADE_CAP, 3) int32.
    return final_state, replayed_trades_buf


# Module-level jitted lift — the same idiom as ``_submit_order``.
_replay_orders = jax.jit(
    _replay_orders_body,
    static_argnames=("product_config", "market_config"),
)


def replay_orders_through_book(
    orders: jnp.ndarray,
    contract_book,
    historical_trades: jnp.ndarray,
    heuristic: str = "rest_until_gct",
) -> ReplayResult:
    """Replay historical orders through a ContractBook and compare to historical trades.

    H1 heuristic (rest_until_gct): every order rests in the book until
    fully matched. No cancellations are issued. This produces an upper
    bound on book depth (OQ-10).

    The replay is a single ``lax.scan`` lifted to the module-level
    :data:`_replay_orders` jit (no per-row Python ``for`` loop). The public signature and ``ReplayResult``
    contract are preserved.

    Args:
        orders: (N, 13) JAX array of orders sorted by timestamp.
        contract_book: ContractBook instance.
        historical_trades: (M, K) JAX array of historical trades.
            Columns [1]=price, [2]=quantity used for matching.
        heuristic: Replay heuristic name. Only "rest_until_gct" implemented.

    Returns:
        ReplayResult with match statistics.

    Raises:
        NotImplementedError: If heuristic is not "rest_until_gct".
    """
    if heuristic != "rest_until_gct":
        raise NotImplementedError(
            f"Heuristic '{heuristic}' is not implemented. "
            f"Only 'rest_until_gct' is supported."
        )

    # Caller-contract invariant (see ``jax_constants.py``, contract.py decision): the
    # ``contract_open`` guard is resolved Python-side via a single
    # ``== ContractStatus.OPEN`` comparison. The result is the closed-
    # over scalar that gates entry into the scan. Mirrors the
    # ``ContractBook.submit_order`` trampoline in ``contract_book.py``.
    from omiecid_simulator.omiecid.contract import ContractStatus
    contract_open = contract_book._contract.status == ContractStatus.OPEN

    product_config = contract_book._product_config
    market_config = contract_book._market_config

    state = contract_book.init()

    n_orders = int(orders.shape[0])
    if contract_open and n_orders > 0:
        state, replayed_trades_buf = _replay_orders(
            state, orders, product_config, market_config,
        )
    else:
        replayed_trades_buf = jnp.full(
            (n_orders, cst.NTRADE_CAP, 3), -1, dtype=jnp.int32,
        ).at[..., 1:].set(jnp.int32(0))

    # Compact the (N_orders, NTRADE_CAP, 3) int32 slab into the legacy
    # (K, 3) float32 replayed-trades array expected by the downstream
    # matching block. Column layout in the slab is
    # (aggressor_order_id, price_tick, qty_tick) with sentinel ``-1`` in
    # the order-id column for unfilled slots. The boundary cast is the
    # natural JIT-to-Python handoff — the
    # matching block below is Python-eager already.
    # The reshape is shape-agnostic: (0, NTRADE_CAP, 3) -> (0, 3) naturally,
    # so no special-casing of the empty-buffer path is required.
    flat = replayed_trades_buf.reshape(-1, 3)
    n_fills = int(jnp.sum(flat[:, 0] != -1))

    if n_fills > 0:
        keep_idx = jnp.where(flat[:, 0] != -1, size=n_fills)[0]
        kept = flat[keep_idx]  # (n_fills, 3) int32
        price_tick = product_config.price_tick
        price_min = product_config.price_min
        qty_tick = product_config.qty_tick
        # Int-domain decode (FMA-immune), the same
        # convention as omie_session._decode_price_eur / contract_book.py.
        # Verdict-neutral here: the comparator's match window below is 0.02 EUR
        # (2 ticks) against a ~2.4e-4 EUR move, but the convention must be
        # uniform across every traced decode in the repo.
        price_offset_int = jnp.int32(round(-price_min / price_tick))
        replayed_trades_arr = jnp.stack([
            (kept[:, 1].astype(jnp.int32) - price_offset_int).astype(jnp.float32)
            * jnp.float32(price_tick),
            kept[:, 2].astype(jnp.float32) * jnp.float32(qty_tick),
            kept[:, 0].astype(jnp.float32),
        ], axis=1)
    else:
        replayed_trades_arr = jnp.zeros((0, 3), dtype=jnp.float32)

    # Compare replayed trades against historical trades. The
    # ``historical_trades`` argument is int32 tick-encoded — decode into
    # EUR/MWh / MW to match the ``replayed_trades_arr`` units. Callers
    # that still hand in decoded float32 trades (older fixtures) are
    # tolerated via ``jnp.issubdtype``.
    historical_trades_total = int(historical_trades.shape[0])
    h_is_int = bool(jnp.issubdtype(historical_trades.dtype, jnp.integer))

    # Greedy 1:1 matching on (price, quantity) with tolerance
    matched_count = 0
    consumed = set()  # indices of replayed trades already matched

    for h_idx in range(historical_trades_total):
        if h_is_int:
            h_price = int(historical_trades[h_idx, 1]) * product_config.price_tick + product_config.price_min
            h_qty = int(historical_trades[h_idx, 2]) * product_config.qty_tick
        else:
            h_price = float(historical_trades[h_idx, 1])
            h_qty = float(historical_trades[h_idx, 2])

        for r_idx in range(replayed_trades_arr.shape[0]):
            if r_idx in consumed:
                continue
            r_price = float(replayed_trades_arr[r_idx, 0])
            r_qty = float(replayed_trades_arr[r_idx, 1])

            if abs(r_price - h_price) < 0.02 and abs(r_qty - h_qty) < 0.02:
                consumed.add(r_idx)
                matched_count += 1
                break

    total_replayed = int(replayed_trades_arr.shape[0])
    extra_trades = total_replayed - matched_count

    if historical_trades_total > 0:
        match_rate = matched_count / historical_trades_total
    else:
        match_rate = 1.0

    # Compute extra_volume_pct
    if total_replayed > 0:
        total_replayed_volume = float(jnp.sum(replayed_trades_arr[:, 1]))
        matched_volume = sum(
            float(replayed_trades_arr[r_idx, 1])
            for r_idx in range(total_replayed)
            if r_idx in consumed
        )
        extra_volume = total_replayed_volume - matched_volume
        extra_volume_pct = extra_volume / total_replayed_volume if total_replayed_volume > 0 else 0.0
    else:
        extra_volume_pct = 0.0

    return ReplayResult(
        final_book=state,
        replayed_trades=replayed_trades_arr,
        historical_trades_matched=matched_count,
        historical_trades_total=historical_trades_total,
        extra_trades=extra_trades,
        match_rate=match_rate,
        extra_volume_pct=extra_volume_pct,
        # Surface the terminal OMIEBookState under
        # the canonical ``final_state`` name so its ``book_full`` leaf is
        # observable end-to-end.
        final_state=state,
    )


# ---------------------------------------------------------------------------
# build_fixed_tape_set — loader output -> Session tape set
# ---------------------------------------------------------------------------
# The 13-column inert PAD sentinel row consumed by ``Session``. It mirrors the
# ``_default_fixed_tape_set`` pad row in ``omiecid_simulator/omiecid/omie_session.py``:
# ``order_id <= 0`` and ``timestamp <= 0`` so ``_replay_orders`` produces no
# fill and ``Session._slice_time_minutes`` masks it out (an inert pad row does
# NOT advance simulated time). The enum-valued columns carry valid sentinels
# (``ExecRestriction.NON`` / ``Validity.GFS`` / ``Status.ACTIVE``) so the
# shared book replay tolerates the row exactly as it tolerates the
# synthetic default set's pad rows.
PAD_ORDER_ROW = [
    0,                          # [0]  contract_id
    0,                          # [1]  side
    0,                          # [2]  price_int
    0,                          # [3]  qty_int
    -1,                         # [4]  order_id  (<= 0 => inert)
    0,                          # [5]  agent_id
    -1,                         # [6]  timestamp (<= 0 => masked, no time advance)
    int(ExecRestriction.NON),   # [7]  exec_restriction
    int(Validity.GFS),          # [8]  validity
    0,                          # [9]  expiry_time
    int(Status.ACTIVE),         # [10] status
    0,                          # [11] peak_qty_int
    0,                          # [12] ppd_int
]

# Backwards-compatible private alias (the same object). The public name is the
# single source of truth for the pad sentinel and is imported by the synthetic
# tape generator (``omie_tape_synth``); the two producers must never drift on
# this row, since a drift silently changes the replay length via
# ``Session._slice_time_minutes``.
_PAD_ORDER_ROW = PAD_ORDER_ROW


def build_fixed_tape_set(
    configs=None,
    *,
    data_path=None,
    contract_ids=None,
    delivery_dates=None,
    cache_path="saved_npz/omie/",
    n_orders_padded=None,
    pad_extra=8,
) -> jnp.ndarray:
    """Stack/pad N per-tape ``load_omie_orders`` outputs into one 3-D array.

    Each
    OMIE ``(contract_id, delivery_date)`` resolves — via the OMIE-native
    :func:`load_omie_orders` (which reads ``data/rawOMIE/`` XBID CSVs, NEVER
    LOBSTER) — to one variable-length ``(N_i, 13)`` int32 tape. This helper
    right-pads every tape to a common ``N_orders_padded`` with the inert
    :data:`_PAD_ORDER_ROW` sentinel and stacks them into the
    ``(n_tapes, N_orders_padded, 13)`` int32 array ``Session`` consumes as its
    ``config["tapes"]`` — with no change to the ``Session``
    signature (the ``jaxomiecid`` layer produces the tapes; the Session consumes them).

    Two call forms are supported:

    * **List of configs** (canonical) — ``build_fixed_tape_set(cfgs)`` /
      ``build_fixed_tape_set(configs=cfgs)`` where ``cfgs`` is a list of
      :class:`OMIEDataConfig`. One tape per config, in list order.
    * **Cross-product pointer** — ``build_fixed_tape_set(data_path=...,
      contract_ids=[...], delivery_dates=[...])``. One tape per
      ``(contract_id, delivery_date)`` pair (outer product), so a single
      contract across several delivery days is
      ``contract_ids=[16], delivery_dates=["2025-06-01", "2025-06-02"]``.

    Args:
        configs: list of :class:`OMIEDataConfig` (one per desired tape). Mutually
            exclusive with the pointer form.
        data_path: OMIE data root (pointer form).
        contract_ids: list of contract ids (pointer form).
        delivery_dates: list of ISO delivery-date strings (pointer form).
        cache_path: :class:`OMIEDataConfig.cache_path` for the pointer form.
        n_orders_padded: explicit common padded length. Defaults to
            ``max(N_i) + pad_extra`` so the final step's slice comfortably
            crosses GCT (mirrors the synthetic default set's ``+ 8`` trailing
            pad). Must be ``>= max(N_i)`` when given.
        pad_extra: trailing inert-pad rows added past the longest real tape when
            ``n_orders_padded`` is not given (default 8).

    Returns:
        A ``(n_tapes, N_orders_padded, 13)`` ``jnp.int32`` array, directly
        consumable as ``Session({"tapes": <this>})``.

    Raises:
        ValueError: if neither call form supplies at least one tape, if both
            forms are mixed, or if an explicit ``n_orders_padded`` is shorter
            than the longest loaded tape.
    """
    # --- resolve the config list from whichever call form was used ----------
    if configs is not None:
        if contract_ids is not None or delivery_dates is not None or data_path is not None:
            raise ValueError(
                "build_fixed_tape_set: pass EITHER a list of OMIEDataConfig "
                "(configs=...) OR the pointer form (data_path/contract_ids/"
                "delivery_dates), not both."
            )
        cfg_list = list(configs)
    else:
        if data_path is None or contract_ids is None or delivery_dates is None:
            raise ValueError(
                "build_fixed_tape_set requires either a list of OMIEDataConfig "
                "or the (data_path, contract_ids, delivery_dates) pointer form."
            )
        cfg_list = [
            OMIEDataConfig(
                data_path=str(data_path),
                contract_id=int(cid),
                delivery_date=str(ddate),
                cache_path=cache_path,
            )
            for cid in contract_ids
            for ddate in delivery_dates
        ]

    if len(cfg_list) == 0:
        raise ValueError("build_fixed_tape_set: no tapes requested (empty config set).")

    # --- load each tape (OMIE-native; NEVER LOBSTER) ------------------------
    # ``load_omie_orders`` returns a chronologically-sorted (N_i, 13) int32
    # tape (empty configs yield (0, 13)). We drop empty tapes so a stray
    # no-data day cannot inject an all-pad tape that terminates at step 0.
    raw_tapes = []
    for cfg in cfg_list:
        orders = jnp.asarray(load_omie_orders(cfg), dtype=jnp.int32)
        if orders.shape[0] > 0:
            raw_tapes.append(orders)

    if len(raw_tapes) == 0:
        raise ValueError(
            "build_fixed_tape_set: every requested (contract_id, delivery_date) "
            "loaded ZERO orders — check the data_path / contract_id / "
            "delivery_date against data/rawOMIE/."
        )

    # --- common padded length ----------------------------------------------
    max_real = max(int(t.shape[0]) for t in raw_tapes)
    if n_orders_padded is None:
        n_padded = max_real + int(pad_extra)
    else:
        n_padded = int(n_orders_padded)
        if n_padded < max_real:
            raise ValueError(
                f"build_fixed_tape_set: n_orders_padded={n_padded} is shorter "
                f"than the longest loaded tape ({max_real} rows)."
            )

    pad_row = jnp.asarray(_PAD_ORDER_ROW, dtype=jnp.int32)

    # --- right-pad each tape to n_padded with the inert sentinel, then stack -
    padded = []
    for t in raw_tapes:
        n_real = int(t.shape[0])
        if n_real < n_padded:
            pad_block = jnp.broadcast_to(pad_row, (n_padded - n_real, 13))
            t = jnp.concatenate([t, pad_block], axis=0)
        padded.append(t)

    tapes = jnp.stack(padded, axis=0).astype(jnp.int32)
    return tapes


# ---------------------------------------------------------------------------
# Per-tape day-ahead anchor builder
# ---------------------------------------------------------------------------
# Materialises a per-tape ``day_ahead_anchors`` array (one reference price
# per tape). The CID->day-ahead alignment is:
#
#     anchor(c, D) = marginalpdbc(date D, hour (c // 4) + 1)          [EUR/MWh]
#
# i.e. the day-ahead (SDAC/MIBEL) marginal clearing price for the Spanish
# bidding zone, on the CID contract's own delivery day ``D``, for the clock hour
# that contains the contract's 15-min delivery window (1-indexed, matching
# marginalpdbc's hours 1..24). The value is held session-constant (OQ-26). Raw
# EUR/MWh — already on the ``p_bid``/``p_ask`` scale, so NO bespoke rescaling
# here (any later standardisation must be a fixed, tape-independent transform).
# It is a read-only reference value: the simulator never uses it for matching
# or settlement (OQ-24).
# ---------------------------------------------------------------------------
def _marginalpdbc_path(data_path: str, delivery_date: str) -> str:
    """The daily marginalpdbc flat-file path for one delivery date.

    Mirrors the ``orders_YYYYMMDD.1`` convention (:func:`load_omie_orders`):
    ``<data_path>/marginalpdbc_YYYYMMDD.1``.
    """
    date_nodash = str(delivery_date).replace("-", "")
    return f"{data_path}/marginalpdbc_{date_nodash}.1"


def parse_marginalpdbc_file(filepath: str) -> dict:
    """Parse an OMIE ``marginalpdbc_YYYYMMDD.1`` day-ahead flat file.

    Structure: a ``MARGINALPDBC;`` header token
    line, then one semicolon-delimited row per hour
    ``YYYY;MM;DD;<hour 1..24>;<price col A>;<price col B>;``, then a ``*``
    terminator. The two price columns are, in order, col A (field index 4) =
    PORTUGAL (PT) and col B (field index 5) = SPAIN (ES). The Spanish-zone
    marginal price is therefore the SECOND price column (col B, field index 5),
    EUR/MWh. Real files are DOT-decimal (``129.8``); a cell containing ``,`` is
    read as Spanish format (see :func:`_parse_float_comma`). (Verified by cross-checking
    ``marginalpdbc`` against the Portugal-only ``marginalpdbcpt`` file on the 18
    divergent June-2025 days: col A = PT, col B = ES.)

    Args:
        filepath: path to the ``marginalpdbc_YYYYMMDD.1`` file.

    Returns:
        ``{hour(1..24): price_es_float}`` for every hour row present.

    Raises:
        FileNotFoundError: if the flat file is absent.
        OMIENumberParseError: if a col-B cell is unparseable or non-finite.
    """
    hour_to_price: dict = {}
    with open(filepath, encoding="latin-1") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("*") or line.upper().startswith("MARGINALPDBC"):
                continue
            fields = [c.strip() for c in line.split(";") if c.strip() != ""]
            # Need at least YYYY, MM, DD, hour, col A (PT), col B (ES).
            if len(fields) < 6:
                continue
            try:
                hour = int(fields[3])
            except ValueError:
                continue
            # col A (field index 4) = PORTUGAL; col B (field index 5) = SPAIN.
            # Dot-decimal on disk ('129.8'); a comma-less cell is parsed as-is,
            # so the '.' is a decimal point here.
            price_es = _parse_float_comma(
                fields[5], column="marginalpdbc col B (ES)", source=str(filepath)
            )
            hour_to_price[hour] = price_es
    return hour_to_price


def day_ahead_anchor(contract_id: int, delivery_date: str, data_path: str) -> float:
    """The single day-ahead anchor scalar for one ``(contract_id, delivery_date)``.

    Applies the alignment ``anchor(c, D) = marginalpdbc(D, (c // 4) + 1)`` — the
    Spanish-zone day-ahead marginal price for delivery day ``D`` at the 1-indexed
    clock hour ``(contract_id // 4) + 1`` (dayahead-sourcing-note.md §3). E.g.
    ``contract_id 0`` -> hour 1 (00:00-01:00 CET). Raw EUR/MWh.
    """
    hour = (int(contract_id) // 4) + 1
    curve = parse_marginalpdbc_file(_marginalpdbc_path(data_path, delivery_date))
    if hour not in curve:
        raise ValueError(
            f"day_ahead_anchor: marginalpdbc for {delivery_date} has no hour "
            f"{hour} (contract_id {contract_id}); available hours "
            f"{sorted(curve.keys())}."
        )
    return float(curve[hour])


def build_day_ahead_anchors(
    configs=None,
    *,
    data_path=None,
    contract_ids=None,
    delivery_dates=None,
) -> jnp.ndarray:
    """Per-tape ``day_ahead_anchors`` array aligned to :func:`build_fixed_tape_set`.

    Accepts the SAME two call forms as :func:`build_fixed_tape_set` (a list of
    :class:`OMIEDataConfig`, or the ``data_path``/``contract_ids``/
    ``delivery_dates`` cross-product pointer) and returns a ``(n_tapes,)``
    float32 array whose ``i``-th entry is the day-ahead anchor for the ``i``-th
    retained tape. To keep index-for-index alignment with
    :func:`build_fixed_tape_set` (which DROPS empty tapes), a config whose
    :func:`load_omie_orders` yields zero rows is skipped here too.

    Feed the result straight into ``Session({"tapes": ..., "day_ahead_anchors":
    <this>})``.

    Raises:
        ValueError: on the same empty-request / mixed-form conditions as
            :func:`build_fixed_tape_set`.
    """
    if configs is not None:
        if contract_ids is not None or delivery_dates is not None or data_path is not None:
            raise ValueError(
                "build_day_ahead_anchors: pass EITHER a list of OMIEDataConfig "
                "(configs=...) OR the pointer form, not both."
            )
        cfg_list = list(configs)
    else:
        if data_path is None or contract_ids is None or delivery_dates is None:
            raise ValueError(
                "build_day_ahead_anchors requires either a list of OMIEDataConfig "
                "or the (data_path, contract_ids, delivery_dates) pointer form."
            )
        cfg_list = [
            OMIEDataConfig(
                data_path=str(data_path),
                contract_id=int(cid),
                delivery_date=str(ddate),
            )
            for cid in contract_ids
            for ddate in delivery_dates
        ]

    if len(cfg_list) == 0:
        raise ValueError(
            "build_day_ahead_anchors: no tapes requested (empty config set)."
        )

    anchors = []
    for cfg in cfg_list:
        # ``load_omie_orders`` already returns an (N, 13) int32 array; we only
        # need N here to mirror build_fixed_tape_set's empty-tape drop, so no
        # re-wrap/dtype-cast of the returned array.
        if load_omie_orders(cfg).shape[0] == 0:
            continue  # mirror build_fixed_tape_set's empty-tape drop
        anchors.append(
            day_ahead_anchor(cfg.contract_id, cfg.delivery_date, cfg.data_path)
        )

    if len(anchors) == 0:
        raise ValueError(
            "build_day_ahead_anchors: every requested (contract_id, "
            "delivery_date) loaded ZERO orders — check data_path/contract_id/"
            "delivery_date against data/rawOMIE/."
        )
    return jnp.asarray(anchors, dtype=jnp.float32)

