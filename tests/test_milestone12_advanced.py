"""Pytest wrapper of the Milestone 12 synthetic software checks (``m12.validation``). Synthetic data only: a pass says the
code behaves as documented, never anything about athletes. Run by the user: ``pytest tests/test_milestone12_advanced.py``."""

from __future__ import annotations

import pytest

from nflbdb.research.advanced_movement.m12 import validation as v12


@pytest.mark.parametrize("check", v12.ALL_CHECKS, ids=lambda f: f.__name__)
def test_check(check):
    try:
        check()
    except v12.Skip as s:
        pytest.skip(str(s))
