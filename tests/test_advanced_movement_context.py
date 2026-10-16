"""Advanced movement programme — Milestone 07, context decomposition, aligned transfer, detectability and scorecard
(synthetic data only; no warehouse access, no PyTorch).

AUTHORED 2026-10-13, NOT EXECUTED by Claude. The checks live in
``nflbdb.research.advanced_movement.context_validation`` so that a notebook cell can run exactly the same code:

    from nflbdb.research.advanced_movement import context_validation as cv
    cv.assert_all_passed(cv.run_all())

This file only exposes them to a test runner, one test per check. The programme itself is run from the notebooks.
"""

from __future__ import annotations

import pytest

from nflbdb.research.advanced_movement import context_validation as cv


@pytest.mark.parametrize("check", cv.ALL_CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    assert isinstance(check(), str)


def test_check_lists_cover_every_check_once():
    names = [f.__name__ for f in cv.ALL_CHECKS]
    assert len(names) == len(set(names)) == 21
    assert {f.__name__ for g in (cv.CHECKS_07A, cv.CHECKS_07B, cv.CHECKS_07C, cv.CHECKS_07D) for f in g} == set(names)
