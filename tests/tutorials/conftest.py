"""Collection hook for the tutorial tests.

Tests carrying the ``tutorials`` marker execute whole notebooks and take minutes, so a
plain ``pytest`` collects them but *deselects* them. They run only when the ``-m``
expression names ``tutorials``; the documented command is ``pytest -m tutorials -n 0``.

This is a hook rather than ``addopts``: any ``-m`` given on the command line would
override an ``-m`` placed in ``addopts``.
"""
from __future__ import annotations

import re

MARKER = "tutorials"


def _names_tutorials(markexpr):
    return re.search(rf"\b{MARKER}\b", markexpr or "") is not None


def pytest_collection_modifyitems(config, items):
    if _names_tutorials(config.getoption("markexpr", "")):
        return
    keep, drop = [], []
    for item in items:
        (drop if item.get_closest_marker(MARKER) else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
