"""Advanced movement programme — Milestone 1 (synthetic data only; no warehouse access, no model training).

AUTHORED 2026-10-11, NOT EXECUTED by Claude. The checks live in
``nflbdb.research.advanced_movement.validation`` so that a notebook cell can run exactly the same code:

    from nflbdb.research.advanced_movement import validation
    validation.assert_all_passed(validation.run_all())

This file only exposes them to a test runner, one test per check.
"""

from __future__ import annotations

import pytest

from nflbdb.research.advanced_movement import validation


@pytest.mark.parametrize("check", validation.CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    assert isinstance(check(), str)


def test_run_all_reports_every_check_and_passes():
    report = validation.run_all()
    assert report.height == len(validation.CHECKS)
    validation.assert_all_passed(report)


def test_a_failing_check_is_reported_not_swallowed():
    def check_broken() -> str:
        raise AssertionError("planted failure")

    report = validation.run_all((check_broken,))
    assert report["status"].to_list() == ["FAIL"] and "planted failure" in report["detail"][0]
    with pytest.raises(AssertionError):
        validation.assert_all_passed(report)
