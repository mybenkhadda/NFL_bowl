"""Milestone 10 — final product logic (synthetic data only; no warehouse access, no Streamlit, no matplotlib).

AUTHORED 2026-10-14, NOT EXECUTED by Claude. The checks live in
``nflbdb.research.advanced_movement.milestone10_validation`` so that a notebook cell can run exactly the same code:

    from nflbdb.research.advanced_movement import milestone10_validation as v10
    v10.assert_all_passed(v10.run_all())

This file only exposes them to a test runner, one test per check. Application wiring (that every page imports and reads
the mart) is covered by ``tests/test_cil_app_wiring.py``; the real mart is validated by notebook 10A.
"""

from __future__ import annotations

import pytest

from nflbdb.research.advanced_movement import milestone10_validation as v10


@pytest.mark.parametrize("check", v10.ALL_CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    assert isinstance(check(), str)
