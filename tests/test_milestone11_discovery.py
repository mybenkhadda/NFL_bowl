"""Milestone 11 — physics of Combine → NFL movement transfer (synthetic fixtures only; no warehouse access).

AUTHORED 2026-10-14, NOT EXECUTED by Claude. The checks live in ``nflbdb.research.advanced_movement.m11.validation`` so a
notebook cell runs exactly the same code:

    from nflbdb.research.advanced_movement.m11 import validation as v11
    v11.assert_all_passed(v11.run_all())

Covered: physical units, sensor sampling, speed-band extraction, episode segmentation, acceleration estimation, censoring,
coordinate geometry, turn direction, player linkage, athlete-grouped folds, pair overlap, training-only transformations,
outcome construction, resampling, checkpoint compatibility, saved-artifact integrity (incl. the Movement Gap 10.2 fix).
"""

from __future__ import annotations

import pytest

from nflbdb.research.advanced_movement.m11 import validation as v11


@pytest.mark.parametrize("check", v11.ALL_CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    assert isinstance(check(), str)
