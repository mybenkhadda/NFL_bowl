"""Advanced movement programme — Milestone 06, Combine → NFL transfer validation (synthetic data only; no warehouse
access).

AUTHORED 2026-10-13, NOT EXECUTED by Claude. The checks live in
``nflbdb.research.advanced_movement.transfer_validation`` so that a notebook cell can run exactly the same code:

    from nflbdb.research.advanced_movement import transfer_validation as tv
    tv.assert_all_passed(tv.run_all())

This file only exposes them to a test runner, one test per check. The three checks that train or load an encoder
are skipped without PyTorch. The programme itself is run from the notebooks.
"""

from __future__ import annotations

import importlib.util

import pytest

from nflbdb.research.advanced_movement import transfer_validation as tv

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.parametrize("check", tv.ALL_CHECKS, ids=lambda f: f.__name__.removeprefix("check_"))
def test_synthetic_check(check):
    if check.__name__.removeprefix("check_") in tv.TORCH_CHECKS and not HAS_TORCH:
        pytest.skip("PyTorch is not installed")
    assert isinstance(check(), str)


def test_check_lists_cover_every_check_once():
    names = [f.__name__ for f in tv.ALL_CHECKS]
    assert len(names) == len(set(names)) == 21
    assert set(tv.TORCH_CHECKS) <= {n.removeprefix("check_") for n in names}
