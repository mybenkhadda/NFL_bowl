"""Advanced movement programme — Milestone 09, Combine Intelligence Lab (synthetic data only; no warehouse access, no
PyTorch, no Streamlit).

AUTHORED 2026-10-14, NOT EXECUTED by Claude. The checks live in
``nflbdb.research.advanced_movement.intelligence_validation`` so that a notebook cell can run exactly the same code:

    from nflbdb.research.advanced_movement import intelligence_validation as iv
    iv.assert_all_passed(iv.run_all())

This file only exposes them to a test runner, one test per check. The application mart of a real run is validated by
notebook 09E (``milestone09_artifacts.validate_mart``).
"""

from __future__ import annotations

import pytest

from nflbdb.research.advanced_movement import intelligence_validation as iv


@pytest.mark.parametrize("check", iv.ALL_CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    assert isinstance(check(), str)
