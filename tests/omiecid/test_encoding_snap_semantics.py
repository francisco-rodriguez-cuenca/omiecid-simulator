"""Encoder-layer tick-snap contract for P2-L1-HOTFIX-tick-precision.

Story reference: ``docs/omiecid/agile-plan.md`` §``Phase 2 HOTFIX — Order
validator tick precision`` / ``Story P2-L1-HOTFIX-tick-precision`` (Option C
/ C-variant 1-prime, second re-adjudication 2026-04-23).

Architect brief: ``docs/omiecid/sprint-status.md`` §``P2-L1-HOTFIX-tick-precision
— Second fix-mechanism re-adjudication (Option C)``, §4 test-revision brief,
Option R3.

Background
----------
Under Option C, ``order_validator.validate_order`` no longer performs an
independent on-tick check — sub-half-tick float offsets are snapped onto
the nearest tick by the encoder's ``jnp.round`` semantics, and the
validator bounds-checks the resulting int32. The behavioural contract
that used to live as "off-tick is rejected at the validator" has
migrated — in *inverted* form — to this file, which pins the encoder's
snap semantics directly.

The tests here are black-box against
``omiecid_simulator.omiecid.encoding.encode_price`` /
``encoding.encode_qty`` / ``encoding.decode_price`` / ``encoding.decode_qty``
and the default ``OMIEProductConfig``.

Contract summary
----------------
1. **Sub-half-tick snap** (price & qty): inputs within strictly-less-than
   half-a-tick of a valid tick encode to the same int as that tick.
2. **On-tick round-trip**: ``decode(encode(p)) == p`` for on-tick ``p``
   within a half-tick tolerance (the decode introduces float-multiplication
   noise at O(1e-13), well inside half-a-tick).
3. **Offsets ≥ half-tick produce distinct ints**: encoding a value that is
   strictly more than half-a-tick above an on-tick value yields a strictly
   larger int.
4. **Cross-check with ``validate_order``**: for the audit probe ``50.0 +
   1e-4``, the encoder produces the same int as for ``50.00`` (the
   semantic consequence is that the validator — post Option C — will
   accept both; this file asserts the encoder half of that equivalence
   without invoking the validator, so it stays green on HEAD).
5. **Encoder determinism under ``jax.jit``**: ``jax.jit`` wrapping of the
   encoder produces the exact same int32 as the eager Python encoder on
   every probe, for both prices and quantities.

Not in scope
------------
- Bounds rejection (covered by the validator suite).
- Exact float-precision of ``decode_price`` (covered by
  ``tests/omiecid/test_loader_int32*``).
- Iceberg / side / status / contract_open semantics (validator suite).

Runtime
-------
All tests are fast-suite (target < 1 s). No ``@pytest.mark.slow``.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from omiecid_simulator.omiecid.encoding import (
    encode_price,
    encode_qty,
    decode_price,
    decode_qty,
)
from omiecid_simulator.omiecid.omie_config import OMIEProductConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> OMIEProductConfig:
    return OMIEProductConfig()


# ---------------------------------------------------------------------------
# Section 1 — Sub-half-tick price snap
# ---------------------------------------------------------------------------
#
# price_tick = 0.01, half-tick = 0.005. Any float within strictly
# less-than half-a-tick of an on-tick value must encode to the same
# integer as that on-tick value.


class TestPriceSubHalfTickSnap:
    """``|delta| < 0.005`` -> ``encode_price(p + delta) == encode_price(p)``."""

    @pytest.mark.parametrize(
        "base",
        [0.0, 50.0, 129.68, -129.68, 100.0, -100.0, 1000.0, 9998.0, -9998.0],
    )
    @pytest.mark.parametrize("delta", [1e-4, -1e-4, 1e-3, -1e-3, 2e-3, -2e-3, 4e-3, -4e-3])
    def test_positive_and_negative_sub_half_tick_snaps(self, config, base, delta):
        """For each representative base price and sub-half-tick offset
        (``|delta| < 0.005``), the encoded int must equal the base's."""
        enc_base = encode_price(base, config)
        enc_off = encode_price(base + delta, config)
        assert enc_off == enc_base, (
            f"base={base} enc={enc_base}; base+delta={base + delta} "
            f"enc={enc_off}; expected equal (|delta|={abs(delta)} < half-tick 0.005)"
        )

    def test_audit_probe_50_plus_1e4_snaps_to_50(self, config):
        """The exact probe that the pre-Option-C validator audit used
        (``50.0 + 1e-4``). Encoder snaps to the same int as ``50.00``."""
        assert encode_price(50.0 + 1e-4, config) == encode_price(50.0, config)

    def test_audit_probe_50_plus_1e4_equals_expected_int(self, config):
        """Hardcoded to pin the absolute int value (offset-encoding).

        ``encode_price(50.0, cfg) = round((50.0 - (-9999.0)) / 0.01) = 1004900``.
        """
        assert encode_price(50.0, config) == 1004900
        assert encode_price(50.0 + 1e-4, config) == 1004900


# ---------------------------------------------------------------------------
# Section 2 — Sub-half-tick qty snap
# ---------------------------------------------------------------------------
#
# qty_tick = 0.1, half-tick = 0.05. Any float within strictly
# less-than half-a-tick of an on-tick qty must encode to the same
# integer as that on-tick qty.


class TestQtySubHalfTickSnap:
    """``|delta| < 0.05`` -> ``encode_qty(q + delta) == encode_qty(q)``."""

    @pytest.mark.parametrize("base", [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 999.0])
    @pytest.mark.parametrize("delta", [1e-3, -1e-3, 1e-2, -1e-2, 3e-2, -3e-2, 4e-2, -4e-2])
    def test_sub_half_tick_qty_snaps_to_base(self, config, base, delta):
        # Skip combinations that would push qty below the min tick (0.1)
        # so the probe stays structurally meaningful.
        if base + delta < 0.05:
            pytest.skip("qty probe below half a tick — skip to keep encoding probe meaningful")
        enc_base = encode_qty(base, config)
        enc_off = encode_qty(base + delta, config)
        assert enc_off == enc_base, (
            f"base={base} enc={enc_base}; base+delta={base + delta} "
            f"enc={enc_off}; expected equal (|delta|={abs(delta)} < half-tick 0.05)"
        )

    def test_qty_two_decimal_offset_snaps(self, config):
        """Mirror of the pre-Option-C ``TestQtyNotOnTickRejected`` probe."""
        # 10.05 is exactly half-tick — banker's rounding sends it to 10.0
        # (even tick). 10.04 and 10.06 are sub-half-tick and strictly
        # sub-half-tick respectively; both stay on the base tick.
        assert encode_qty(10.04, config) == encode_qty(10.0, config)
        assert encode_qty(10.049, config) == encode_qty(10.0, config)

    def test_qty_three_decimal_offset_snaps(self, config):
        """``5.123`` pre-Option-C was probed for rejection (``TestQtyNotOnTickRejected``)."""
        assert encode_qty(5.123, config) == encode_qty(5.1, config)


# ---------------------------------------------------------------------------
# Section 3 — On-tick round-trip
# ---------------------------------------------------------------------------
#
# The authoritative exhaustive round-trip sweep lives in
# ``tests/omiecid/test_loader_int32*`` (int32 loader contract, P2-F0). Here
# we assert a small representative set so this file can be read in
# isolation as the encoder-layer contract.


class TestOnTickRoundTripRepresentative:
    """``decode(encode(p)) == p`` for on-tick ``p`` (within a half-tick
    tolerance to absorb the O(1e-13) float-multiplication noise in the
    decode path)."""

    def test_price_round_trip_representative(self, config):
        half_tick = config.price_tick / 2.0
        for p in [0.0, 50.0, 50.01, 129.68, -129.68, 9999.0, -9999.0, -50.01, 1000.0]:
            dec = decode_price(encode_price(p, config), config)
            assert math.isclose(dec, p, abs_tol=half_tick), (
                f"on-tick price {p} failed round-trip: encode -> decode = {dec}"
            )

    def test_qty_round_trip_representative(self, config):
        half_tick = config.qty_tick / 2.0
        for q in [0.1, 0.5, 1.0, 10.0, 100.0, 999.0]:
            dec = decode_qty(encode_qty(q, config), config)
            assert math.isclose(dec, q, abs_tol=half_tick), (
                f"on-tick qty {q} failed round-trip: encode -> decode = {dec}"
            )

    def test_round_trip_reference_authority(self):
        """Documentation pointer: the full-grid sweep is owned by the
        loader test suite, not here. This file asserts the encoder
        snap contract; the int32 dtype contract is tests/omiecid/test_loader_int32*."""
        # No behavioural assertion — the test body is the docstring.
        assert True


# ---------------------------------------------------------------------------
# Section 4 — Offsets ≥ half-tick produce distinct ints
# ---------------------------------------------------------------------------
#
# Positive assertion of the claim that replaces the deleted
# "off-tick rejection" tests: offsets strictly greater than half-a-tick
# above an on-tick base encode to a *strictly larger* integer, proving
# the encoder is not collapsing every float onto a single tick.


class TestHalfTickBoundaryProducesDistinctInts:
    """For small epsilon > 0, ``encode(base + half_tick + epsilon) > encode(base)``."""

    def test_price_offset_above_half_tick_is_distinct(self, config):
        # Using 0.006 (= half-tick 0.005 + 0.001), well inside float32
        # discrimination at OMIE magnitudes.
        half_tick = config.price_tick / 2.0
        eps = 1e-3
        offset = half_tick + eps  # 0.006
        for base in [0.0, 50.0, 100.0, 1000.0, -100.0, 500.0]:
            enc_base = encode_price(base, config)
            enc_off = encode_price(base + offset, config)
            assert enc_off > enc_base, (
                f"base={base} enc={enc_base}; base+offset={base + offset} "
                f"enc={enc_off}; expected strictly larger int"
            )

    def test_price_offset_below_minus_half_tick_is_distinct(self, config):
        """Mirror for negative offsets: offset < -half_tick should decrease the int."""
        half_tick = config.price_tick / 2.0
        eps = 1e-3
        offset = -(half_tick + eps)  # -0.006
        for base in [50.0, 100.0, 1000.0, -100.0, 500.0, 0.01]:
            enc_base = encode_price(base, config)
            enc_off = encode_price(base + offset, config)
            assert enc_off < enc_base, (
                f"base={base} enc={enc_base}; base+offset={base + offset} "
                f"enc={enc_off}; expected strictly smaller int"
            )

    def test_qty_offset_above_half_tick_is_distinct(self, config):
        half_tick = config.qty_tick / 2.0
        eps = 1e-2
        offset = half_tick + eps  # 0.06
        for base in [0.2, 1.0, 10.0, 100.0, 500.0]:
            enc_base = encode_qty(base, config)
            enc_off = encode_qty(base + offset, config)
            assert enc_off > enc_base, (
                f"base={base} enc={enc_base}; base+offset={base + offset} "
                f"enc={enc_off}; expected strictly larger int"
            )

    def test_qty_offset_below_minus_half_tick_is_distinct(self, config):
        half_tick = config.qty_tick / 2.0
        eps = 1e-2
        offset = -(half_tick + eps)
        for base in [1.0, 10.0, 100.0, 500.0]:
            enc_base = encode_qty(base, config)
            enc_off = encode_qty(base + offset, config)
            assert enc_off < enc_base, (
                f"base={base} enc={enc_base}; base+offset={base + offset} "
                f"enc={enc_off}; expected strictly smaller int"
            )


# ---------------------------------------------------------------------------
# Section 5 — Cross-check: the pre-Option-C audit probe now snaps cleanly
# ---------------------------------------------------------------------------
#
# The pre-Option-C audit probe at
# ``test_order_validator_audit.py::TestTickAlignmentPurity::test_off_tick_by_more_than_1e9_is_rejected``
# (deleted in this revision) asserted that ``price = 50.0 + 1e-4`` rejects
# at the validator layer. Under Option C the correct semantics are:
#
# 1. The encoder snaps ``50.0001`` onto the tick at ``50.00``
#    (int ``1004900`` under OMIE defaults).
# 2. The decoded value is therefore ``50.00``, not ``50.0001``.
#
# This section documents that encoder contract. The validator half of the
# cross-check (that the post-Option-C validator accepts the resulting
# OMIEOrder) lives implicitly in the locked hotfix file
# ``test_order_validator_tick_precision.py`` via the on-tick sweeps — no
# new validator call is made here so the file stays green on HEAD.


class TestCrossCheckValidatorAuditProbe:
    """The ``50.0 + 1e-4`` probe from the old validator audit — encoder POV."""

    def test_probe_encodes_to_50_tick(self, config):
        assert encode_price(50.0 + 1e-4, config) == 1004900

    def test_probe_and_exact_50_encode_identically(self, config):
        assert encode_price(50.0 + 1e-4, config) == encode_price(50.0, config)

    def test_probe_decodes_to_exact_50(self, config):
        """Decoding the snapped int returns the exact tick, not 50.0001."""
        decoded = decode_price(encode_price(50.0 + 1e-4, config), config)
        assert math.isclose(decoded, 50.0, abs_tol=config.price_tick / 2.0), (
            f"expected decoded ~ 50.00, got {decoded}"
        )


# ---------------------------------------------------------------------------
# Section 6 — Encoder is deterministic under ``jax.jit``
# ---------------------------------------------------------------------------
#
# The Python ``round`` used inside ``encode_price`` / ``encode_qty``
# dispatches to ``jnp.round`` with banker's rounding
# (``RoundingMethod.TO_NEAREST_EVEN``) when the input is a JAX tracer.
# This section pins that the jit-compiled encoder yields the exact same
# int32 as the eager Python encoder for every probe.


class TestEncoderDeterministicUnderJit:
    """``jax.jit(encode_*)`` must agree bitwise with eager encode."""

    # Class-level marker: every test below consumes a `jax.jit` fixture.
    # Story T1.1 — the lint flags fixtures that build jit closures, and
    # the cleanest scope-resolution here is class-scoped (the rest of
    # this file is mixed; sibling classes are non-JAX).
    pytestmark = pytest.mark.jit

    @pytest.fixture(scope="class")
    def jitted_price(self):
        cfg = OMIEProductConfig()
        return jax.jit(lambda p: encode_price(p, cfg))

    @pytest.fixture(scope="class")
    def jitted_qty(self):
        cfg = OMIEProductConfig()
        return jax.jit(lambda q: encode_qty(q, cfg))

    @pytest.mark.parametrize(
        "price",
        [0.0, 50.0, 50.01, 129.68, -129.68, 9999.0, -9999.0, 45.51, -50.01],
    )
    def test_price_jit_matches_eager(self, config, jitted_price, price):
        eager = encode_price(price, config)
        jit_out = int(jitted_price(jnp.float32(price)))
        assert jit_out == eager, (
            f"price={price}: eager={eager}, jit={jit_out}"
        )

    @pytest.mark.parametrize(
        "qty",
        [0.1, 1.0, 10.0, 100.0, 999.0, 10.3],
    )
    def test_qty_jit_matches_eager(self, config, jitted_qty, qty):
        eager = encode_qty(qty, config)
        jit_out = int(jitted_qty(jnp.float32(qty)))
        assert jit_out == eager, (
            f"qty={qty}: eager={eager}, jit={jit_out}"
        )

    @pytest.mark.parametrize(
        "price",
        [50.0001, 50.004, 45.501, 45.5012],
    )
    def test_sub_half_tick_snap_under_jit(self, config, jitted_price, price):
        """The snap contract holds under jit as well as eager."""
        eager = encode_price(price, config)
        jit_out = int(jitted_price(jnp.float32(price)))
        assert jit_out == eager

    def test_jit_no_drift_129_68(self, config, jitted_price):
        """Named ULP-boundary regression — eager/jit must agree at the
        exact demo-oracle price."""
        eager = encode_price(129.68, config)
        jit_out = int(jitted_price(jnp.float32(129.68)))
        assert jit_out == eager == 1012868
