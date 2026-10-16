"""Advanced movement programme — Milestone 05, NFL movement outcomes, game-grouped reliability and target freeze
(synthetic data only; no warehouse access, no PyTorch).

AUTHORED 2026-10-13, NOT EXECUTED by Claude. The checks live in
``nflbdb.research.advanced_movement.nfl_outcome_validation`` so that a notebook cell can run exactly the same code:

    from nflbdb.research.advanced_movement import nfl_outcome_validation
    nfl_outcome_validation.assert_all_passed(nfl_outcome_validation.run_all())

This file only exposes them to a test runner, one test per check. The programme itself is run from the notebooks.
"""

from __future__ import annotations

import pytest

from nflbdb.research.advanced_movement import nfl_outcome_validation


@pytest.mark.parametrize("check", nfl_outcome_validation.ALL_CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    assert isinstance(check(), str)


def test_run_all_reports_every_check_and_passes():
    report = nfl_outcome_validation.run_all()
    assert report.height == len(nfl_outcome_validation.ALL_CHECKS)
    nfl_outcome_validation.assert_all_passed(report)
