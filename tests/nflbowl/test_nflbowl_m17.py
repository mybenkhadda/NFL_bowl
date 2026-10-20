"""M17 — censoring, strata, alignment maps, hierarchical components, signature rules. Synthetic only.
AUTHORED, NOT EXECUTED by the author: ``pytest tests/nflbowl/test_nflbowl_m17.py -q``."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflbowl.lab.m17 import alignment as AL
from nflbowl.lab.m17 import censoring as CS
from nflbowl.lab.m17 import config as C
from nflbowl.lab.m17 import contexts as CX
from nflbowl.lab.m17 import hierarchical as HM


def _cfg(tmp_path, **kw):
    base = {"drive_root": str(tmp_path / "drive"), **{k: str(tmp_path / "ro" / k) for k in ("m12_run_dir", "m13_run_dir", "m14_run_dir", "m15_run_dir", "m16_run_dir")}}
    base.update(kw)
    return C.resolve(base)


def test_config_rules(tmp_path):
    with pytest.raises(ValueError):
        _cfg(tmp_path, run_training=True)
    with pytest.raises(ValueError):
        _cfg(tmp_path, entry_speed_bins=[4.0, 2.0])
    assert _cfg(tmp_path).training_enabled() is False


def test_km_median():
    assert CS.km_median(np.array([1.0, 2.0, 3.0]), np.array([False, False, False])) == pytest.approx(2.0)
    assert np.isnan(CS.km_median(np.array([1.0, 2.0]), np.array([True, True])))                 # never reaches 0.5
    # censoring shifts the median up relative to dropping censored times
    t = np.array([1.0, 1.5, 2.0, 2.5, 2.6, 2.7])
    c = np.array([False, False, True, True, True, True])
    km = CS.km_median(t, c)
    assert np.isnan(km) or km >= np.median(t[~c])


def test_strata_and_coverage(tmp_path):
    cfg = _cfg(tmp_path, min_stratum_athletes=1, min_events_per_athlete=1)
    ev = pl.DataFrame({"nfl_id": [1, 1, 2, 2], "domain": ["combine", "nfl", "combine", "nfl"], "v_entry": [5.0, 5.0, 7.0, 1.0], "speed_drop": [2.0, 2.0, 3.0, 0.5],
                       "turn_angle": [0.1, 0.1, 2.0, np.nan], "turn_heading_coverage": [1.0, 1.0, 1.0, 0.0]})
    st = CX.assign_strata(ev, cfg)
    assert st["s_entry"].to_list()[0] == "4-6yd/s" and st["s_turn"].to_list()[3] == "turn_unknown"
    cov = CX.coverage(st, cfg)
    assert "supported" in cov.columns and cov.filter(pl.col("stratum") == "4-6yd/s | <20deg")["supported"][0]


def test_coral_matches_moments():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(500, 3))
    B = rng.normal(size=(500, 3)) @ np.diag([2.0, 0.5, 1.0]) + 4.0
    f = AL.coral_map(A, B)
    Z = f(A)
    assert np.allclose(Z.mean(0), B.mean(0), atol=0.05)
    assert np.allclose(np.cov(Z, rowvar=False), np.cov(B, rowvar=False), atol=0.1)
    assert AL.mmd2(Z, B) < AL.mmd2(A, B)


def test_ot_map_moves_toward_target():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(100, 2))
    B = A + np.array([3.0, 0.0])
    f = AL.ot_map(A, B, k=3)
    assert np.allclose(f(A).mean(0), B.mean(0), atol=0.3)


def test_hierarchical_components_recover_athlete_variance():
    rng = np.random.default_rng(2)
    n_a, k = 200, 6
    u = rng.normal(0, 1.0, n_a)
    rows = []
    for i in range(n_a):
        for j in range(k):
            rows.append({"nfl_id": i, "position_group": "A", "kind": "three_cone", "stratum": ("s1", "s2")[j % 2], "row": i * 10 + j, "domain": "combine",
                         "m": u[i] + (0.5 if j % 2 else 0.0) + rng.normal(0, 1.0)})
    ev = pl.DataFrame(rows)
    fold_of = {i: i % 5 for i in range(n_a)}
    eff, comps = HM.athlete_effects(ev, "m", "combine", fold_of, use_statsmodels=False, min_athletes=30)
    assert eff.height == n_a and len(comps) == 5
    s = HM.variance_summary(comps)
    assert 0.35 < float(s["athlete_share_residual"][0]) < 0.65                       # true ICC = 1 / (1 + 1) = 0.5
    r = np.corrcoef(eff.sort("nfl_id")["blup"].to_numpy(), u)[0, 1]
    assert r > 0.6
