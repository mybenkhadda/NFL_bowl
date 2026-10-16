"""Advanced movement programme — Milestone 08B, football transfer and scouting decision intelligence (synthetic data
only; no warehouse access, no PyTorch).

AUTHORED 2026-10-13, NOT EXECUTED by Claude. The checks live in
``nflbdb.research.advanced_movement.football_validation`` so that a notebook cell can run exactly the same code:

    from nflbdb.research.advanced_movement import football_validation as fv
    fv.assert_all_passed(fv.run_all())

This file only exposes them to a test runner, one test per check. The programme itself is run from the notebooks.
"""

from __future__ import annotations

import pytest

from nflbdb.research.advanced_movement import football_validation as fv


@pytest.mark.parametrize("check", fv.ALL_CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    assert isinstance(check(), str)
