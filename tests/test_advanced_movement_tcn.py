"""Advanced movement programme — Milestone 3, self-supervised encoder (synthetic data only; no warehouse access).

AUTHORED 2026-10-11, NOT EXECUTED by Claude. The checks live in
``nflbdb.research.advanced_movement.tcn_validation`` so that a notebook cell can run exactly the same code:

    from nflbdb.research.advanced_movement import tcn_validation
    tcn_validation.assert_all_passed(tcn_validation.run_all())

This file only exposes them to a test runner, one test per check. It is skipped when PyTorch is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from nflbdb.research.advanced_movement import tcn_validation  # noqa: E402


@pytest.mark.parametrize("check", tcn_validation.TCN_CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    assert isinstance(check(), str)


def test_run_all_reports_every_check_and_passes():
    report = tcn_validation.run_all()
    assert report.height == len(tcn_validation.TCN_CHECKS)
    tcn_validation.assert_all_passed(report)
