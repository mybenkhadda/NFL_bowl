"""Advanced movement programme — Milestone 2, functional PCA (synthetic data only; no warehouse access).

AUTHORED 2026-10-11, NOT EXECUTED by Claude. The checks live in
``nflbdb.research.advanced_movement.fpca_validation`` so that a notebook cell can run exactly the same code:

    from nflbdb.research.advanced_movement import fpca_validation
    fpca_validation.assert_all_passed(fpca_validation.run_all())

This file only exposes them to a test runner, one test per check.
"""

from __future__ import annotations

import pytest

from nflbdb.research.advanced_movement import fpca_validation


@pytest.mark.parametrize("check", fpca_validation.FPCA_CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    assert isinstance(check(), str)


def test_run_all_reports_every_check_and_passes():
    report = fpca_validation.run_all()
    assert report.height == len(fpca_validation.FPCA_CHECKS)
    fpca_validation.assert_all_passed(report)
