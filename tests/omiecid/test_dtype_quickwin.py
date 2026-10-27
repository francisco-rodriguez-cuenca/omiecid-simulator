"""Locked contract tests for Story P2-CLEANUP-dtype-quickwin.

Two pins:

(a) Anti-duplication pin on ``_build_jaxob_msg``: the helper has a single
    canonical home on ``contract_book``. Any other module under
    ``omiecid_simulator.omiecid`` is allowed to (i) not reference it at all, or
    (ii) re-export the *same* object via import — but it must NEVER define a
    locally-bodied copy. The probe walks every ``omiecid_simulator.omiecid.*``
    submodule and asserts that any module-level attribute named
    ``_build_jaxob_msg`` is ``is``-identical to ``contract_book._build_jaxob_msg``.
    A re-pasted copy would be a distinct function object and fail the ``is``
    check, even if syntactically identical.

    History: prior to P2-H1, ``iceberg_manager`` re-imported the helper and
    the pin asserted cross-module ``is`` identity directly. Under P2-H1,
    ``iceberg_manager`` was rewritten to a JAX-traceable array-backed form
    that builds the jaxob message inline and no longer imports the helper.
    The revised pin generalises across the package while preserving the
    original spirit (no silent re-paste).

(b) Dtype pin on ``compute_trade_value``: returning a float32 0-d scalar must
    survive removal of the redundant final ``.astype(jnp.float32)``.
"""

import importlib
import pkgutil

import jax.numpy as jnp

import omiecid_simulator.omiecid as omie_pkg
import omiecid_simulator.omiecid.contract_book as contract_book
from omiecid_simulator.omiecid.settlement import compute_trade_value


def test_build_jaxob_msg_is_single_canonical_function():
    canonical = contract_book._build_jaxob_msg
    assert callable(canonical), (
        "contract_book._build_jaxob_msg must exist as the canonical helper"
    )

    offenders = []
    for module_info in pkgutil.iter_modules(
        omie_pkg.__path__, prefix=omie_pkg.__name__ + "."
    ):
        module = importlib.import_module(module_info.name)
        attr = getattr(module, "_build_jaxob_msg", None)
        if attr is None:
            # Absent is fine: module simply does not reference the helper.
            continue
        if attr is not canonical:
            offenders.append(module_info.name)

    assert not offenders, (
        "Modules under omiecid_simulator.omiecid defining a non-canonical "
        f"_build_jaxob_msg (re-paste detected): {offenders}. "
        "The helper must live solely on contract_book; other modules may "
        "re-import it but must never define a local copy."
    )


def test_compute_trade_value_returns_float32():
    assert compute_trade_value(10.0, 50.0, 0.25).dtype == jnp.float32
