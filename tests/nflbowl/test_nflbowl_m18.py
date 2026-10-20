"""M18 — configuration, pre-registration, specificity matrix, twin composite, evidence verdicts. Synthetic only.
AUTHORED, NOT EXECUTED by the author: ``pytest tests/nflbowl/test_nflbowl_m18.py -q``."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflbowl.lab.m18 import config as C
from nflbowl.lab.m18 import evidence as EV
from nflbowl.lab.m18 import scouting as SC
from nflbowl.lab.m18 import specificity as SP


def _cfg(tmp_path, **kw):
    base = {"drive_root": str(tmp_path / "drive"),
            **{k: str(tmp_path / "ro" / k) for k in ("m12_run_dir", "m13_run_dir", "m14_run_dir", "m15_run_dir", "m16_run_dir", "m17_run_dir")}}
    base.update(kw)
    return C.resolve(base)


def test_config_and_preregistration(tmp_path):
    with pytest.raises(KeyError):
        C.resolve({"run_training": True})
    cfg = _cfg(tmp_path)
    run = C.M18Run(cfg.run_dir, cfg)
    a = run.preregister({"primary_target": "T0", "primary_family": ["B2_M1+simple"]})
    assert a["conflict"] is False
    b = run.preregister({"primary_target": "T0", "primary_family": ["B2_M1+simple", "B7_M1+signatures"]})
    assert b["conflict"] is True and b["primary_family"] == ["B2_M1+simple"]           # the first registration stands


def test_specificity_diagonal_detects_task_match():
    rng = np.random.default_rng(0)
    n = 300
    f40, fag = rng.normal(size=n), rng.normal(size=n)
    df = pl.DataFrame({"nfl_id": np.arange(n), "position_group": ["A", "B", "C"] * 100, "forty": rng.normal(4.6, 0.1, n), "height": rng.normal(73, 2, n),
                       "trk_forty__max_speed": f40, "trk_agility__min_atan": fag,
                       "t16_high_speed": f40 + 0.5 * rng.normal(size=n), "t16_braking": fag + 0.5 * rng.normal(size=n)})
    long, contrast = SP.matrix(df, {"T1": "t16_high_speed", "T3": "t16_braking"}, ["trk_forty__max_speed", "trk_agility__min_atan"], ["forty", "height"], 100, 0)
    c = {r["drill_family"]: r for r in contrast.iter_rows(named=True)}
    assert c["forty"]["lo"] > 0 and c["agility"]["lo"] > 0


def test_composite_requires_half_the_features():
    df = pl.DataFrame({"nfl_id": [1, 2, 3, 4], "position_group": ["A"] * 4, "a": [1.0, 2.0, 3.0, None], "b": [None, None, 1.0, 2.0]})
    c = SC.composite(df, ["a", "b"])
    v = dict(zip(c["nfl_id"].to_list(), c["braking_composite"].to_list(), strict=True))
    assert v[1] is not None and v[3] is not None


def test_evidence_verdicts():
    rungs = {r: {"status": "blocked", "reason": "x"} for r in ("B2", "B3", "B4", "B5", "B6", "B7")}
    rungs["B3"] = {"status": "eligible", "reason": "", "columns": ["m"]}
    A = {"status": pl.DataFrame({"model": ["B3_M1+M14"], "status": ["inconclusive"]}),
         "paired": pl.DataFrame({"model": ["B3_M1+M14"], "reference": ["M1"], "delta_rmse": [0.0], "delta_lo": [-0.001], "delta_hi": [0.001], "delta_pct": [0.0],
                                 "exceeds_practical_margin": [False], "precise_null": [True]}),
         "folds": pl.DataFrame({"model": ["B3_M1+M14"] * 5, "fold": list(range(5)), "delta_rmse_vs_M1": [0.0] * 5})}

    class Cfg:
        min_position_n = 30

    ev = EV.evidence_table(rungs, A, None, {}, None, None, Cfg())
    v = dict(zip(ev["rung"].to_list(), ev["verdict"].to_list(), strict=True))
    assert v["B3"] == "precise_null" and v["B2"] == "blocked"
    g = EV.g18(ev)
    assert g.filter(pl.col("criterion") == "G18 overall")["status"][0] == "precise_null"
