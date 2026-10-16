"""Advanced movement programme — Milestone 04, feature atlas and robust TCN validation (synthetic data only; no
warehouse access, no PyTorch).

AUTHORED 2026-10-12, NOT EXECUTED by Claude. The checks live in ``nflbdb.research.advanced_movement.atlas_validation``
so that a notebook cell can run exactly the same code:

    from nflbdb.research.advanced_movement import atlas_validation
    atlas_validation.assert_all_passed(atlas_validation.run_all())

This file only exposes them to a test runner, one test per check. The programme itself is run from the notebooks.
"""

from __future__ import annotations

import pytest

from nflbdb.research.advanced_movement import atlas_validation


@pytest.mark.parametrize("check", atlas_validation.ALL_CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    assert isinstance(check(), str)


def test_run_all_reports_every_check_and_passes():
    report = atlas_validation.run_all()
    assert report.height == len(atlas_validation.ALL_CHECKS)
    atlas_validation.assert_all_passed(report)
