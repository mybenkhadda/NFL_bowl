"""M16 task T04 — M1 contract, prediction comparison, persisted-record discovery, target registry, G16-B rows.

Synthetic data only. AUTHORED, NOT EXECUTED by the author: run with ``pytest tests/nflbowl/test_nflbowl_m16.py -q``.
The end-to-end ``verify_m1`` needs the research backend (official ladder engine) and runs in the Colab notebook.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from nflbowl.lab import status as S
from nflbowl.lab.m16 import config as C
from nflbowl.lab.m16 import m1_verification as V
from nflbowl.lab.m16 import targets as T


def _cfg(tmp_path, **kw):
    base = {"drive_root": str(tmp_path / "drive"), "m12_run_dir": str(tmp_path / "ro" / "m12"), "m13_run_dir": str(tmp_path / "ro" / "m13"),
            "m14_run_dir": str(tmp_path / "ro" / "m14"), "m15_run_dir": str(tmp_path / "ro" / "m15")}
    base.update(kw)
    return C.resolve(base)


def _oof(n=30, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(100, 100 + n)
    y = rng.normal(5, 0.4, n)
    return pl.DataFrame({"nfl_id": ids, "position_group": ["WR"] * n, "fold": ids % 5, "y": y, "pred_M1": y + rng.normal(0, 0.1, n),
                         "pred_M1+F0": y + rng.normal(0, 0.1, n)})


# ====================================================================== config
def test_resolve_rules(tmp_path):
    with pytest.raises(KeyError):
        C.resolve({"run_training": True})
    with pytest.raises(ValueError):
        _cfg(tmp_path, official_target="nfl_speed_q90_median")
    with pytest.raises(ValueError):
        _cfg(tmp_path, reproduction_tol=1e-3)
    with pytest.raises(ValueError):
        _cfg(tmp_path, experiment_name="m15_x")
    c = _cfg(tmp_path)
    assert c.official_target == "nfl.q90_shared" and c.exploratory_target == "nfl_speed_q90_median"
    assert c.experiment_dirs() == []                                     # none exist under tmp


# ====================================================================== comparisons
def test_compare_identical_passes():
    a = _oof()
    rows, diff = V.compare_predictions(a, a.clone(), V.COMPARED, 1e-8, "self")
    assert all(r["status"] == S.PASS for r in rows)
    assert diff.filter(pl.col("mismatch")).height == 0


def test_compare_detects_athlete_fold_and_coverage_mismatch():
    a = _oof()
    b = a.with_columns(pl.when(pl.col("nfl_id") == 103).then(pl.col("pred_M1") + 1e-6).otherwise(pl.col("pred_M1")).alias("pred_M1"))
    rows, diff = V.compare_predictions(a, b, [("pred_M1", "pred_M1")], 1e-8, "perturbed")
    r = rows[0]
    assert r["status"] == S.FAIL and r["n_mismatched_athletes"] == 1 and json.loads(r["mismatched_athletes"]) == [103]
    assert r["max_abs_pred_diff"] == pytest.approx(1e-6, rel=1e-3)
    c = a.filter(pl.col("nfl_id") != 110).with_columns(pl.when(pl.col("nfl_id") == 101).then(4).otherwise(pl.col("fold")).alias("fold"))
    rows2, _ = V.compare_predictions(a, c, [("pred_M1", "pred_M1")], 1e-8, "coverage")
    assert rows2[0]["status"] == S.FAIL and rows2[0]["athletes_only_reference"] == 1 and rows2[0]["fold_mismatches"] == 1
    rows3, _ = V.compare_predictions(a, a.drop("pred_M1+F0"), V.COMPARED, 1e-8, "missing column")
    assert rows3[1]["status"] == S.NOT_EVALUATED


def test_metric_rows_and_fold_checks():
    a = _oof()
    m = pl.DataFrame(V.metric_rows(a, ["pred_M1"], "x"))
    assert set(m["fold"].to_list()) == {"all", "0", "1", "2", "3", "4"}
    manifest = a.select("nfl_id", "fold")
    exp_folds = dict(zip(a["nfl_id"].to_list(), a["fold"].to_list(), strict=True))
    checks = V.fold_checks(a, exp_folds, manifest, 5, 5)
    assert all(c["status"] == S.PASS for c in checks)
    exp_folds[100] = 3
    assert V.fold_checks(a, exp_folds, manifest, 5, 5)[0]["status"] == S.FAIL
    assert V.fold_checks(a, dict(zip(a["nfl_id"].to_list(), a["fold"].to_list(), strict=True)), manifest, 5, 10)[2]["status"] == S.FAIL


def test_target_alignment():
    a = _oof()
    ath = a.select("nfl_id", pl.col("y").alias("nfl.q90_shared"))
    assert V.target_alignment(a, ath, 1e-8)["status"] == S.PASS
    ath2 = ath.with_columns(pl.when(pl.col("nfl_id") == 100).then(0.0).otherwise(pl.col("nfl.q90_shared")).alias("nfl.q90_shared"))
    assert V.target_alignment(a, ath2, 1e-8)["status"] == S.FAIL


# ====================================================================== persisted records
def test_discover_persisted_oof(tmp_path):
    rec = tmp_path / "exp_a" / "evaluation" / "records"
    rec.mkdir(parents=True)
    _oof().write_parquet(rec / "transfer_oof__aaa.parquet")
    (rec / "transfer_oof__aaa.json").write_text(json.dumps({"target": "nfl.q90_shared", "deps": {"bundle": "B", "science": "S"}, "models": ["M1"]}))
    _oof().write_parquet(rec / "transfer_oof__bbb.parquet")
    (rec / "transfer_oof__bbb.json").write_text(json.dumps({"target": "nfl.q90_shared", "deps": {"bundle": "OTHER", "science": "S"}}))
    _oof().drop("pred_M1").write_parquet(rec / "transfer_oof__ccc.parquet")
    t = V.discover_persisted_oof([tmp_path / "exp_a"], "B", "S")
    by = {r["deps_hash"]: r for r in t.iter_rows(named=True)}
    assert by["aaa"]["compatible"] is True
    assert by["bbb"]["compatible"] is False and "bundle" in by["bbb"]["reason"]
    assert by["ccc"]["compatible"] is False and "no pred_M1" in by["ccc"]["reason"] and "sidecar" in by["ccc"]["reason"]


# ====================================================================== contract
def test_m1_contract(tmp_path, arrays):
    *_x, athletes = arrays
    cfg = _cfg(tmp_path)
    rec, tab = V.m1_contract(athletes, cfg, "/bundle")
    assert rec["m1_cols"] == ["draft_year", "height", "weight", "forty", "ten"] and tab.height == 5
    assert rec["target_present"] is True and rec["n_linked_with_target"] == athletes.height
    with pytest.raises(S.ResearchBlocked) as e:
        V.m1_contract(athletes.drop("ten"), cfg, "/bundle")
    assert e.value.status == S.BLOCKED_MISSING_INPUT and any("ten" in m for m in e.value.missing)
    with pytest.raises(S.ResearchBlocked):
        V.m1_contract(None, cfg, "/bundle")


# ====================================================================== targets
def test_target_registry(tmp_path, arrays):
    *_x, athletes = arrays
    t0 = T.target_frame(T.TARGETS["T0"], athletes)
    assert t0.height == athletes.height
    with pytest.raises(S.ResearchBlocked):
        T.target_frame(T.TARGETS["T0x"], athletes)
    p = tmp_path / "expl.parquet"
    athletes.select("nfl_id", (pl.col("nfl.q90_shared") + 1.0).alias("nfl_speed_q90_median")).head(10).write_parquet(p)
    tx = T.target_frame(T.TARGETS["T0x"], athletes, str(p))
    assert tx.height == 10 and "position_group" in tx.columns
    assert T.target_summary(T.TARGETS["T0"], t0)["fingerprint"] != T.target_summary(T.TARGETS["T0x"], tx)["fingerprint"]
    with pytest.raises(S.ValidationFailed):
        T.assert_same_target("nfl.q90_shared", "nfl_speed_q90_median")


# ====================================================================== gate
def test_gate_g16b_never_passes_without_evidence():
    base = {"tolerance": 1e-8, "max_abs_pred_diff_internal": 0.0, "internal_ok": True, "checks_ok": True, "verdict": V.REPRODUCED_INTERNAL_ONLY}
    g = V.gate_g16b({**base, "persisted_ok": None, "max_abs_pred_diff_persisted": None})
    assert g.filter(pl.col("criterion") == "G16-B overall")["status"][0] == S.NOT_EVALUATED
    g2 = V.gate_g16b({**base, "persisted_ok": True, "max_abs_pred_diff_persisted": 0.0, "verdict": V.REPRODUCED})
    assert g2.filter(pl.col("criterion") == "G16-B overall")["status"][0] == S.PASS
    g3 = V.gate_g16b({**base, "persisted_ok": False, "max_abs_pred_diff_persisted": 1e-3, "verdict": V.MISMATCH})
    assert g3.filter(pl.col("criterion") == "G16-B overall")["status"][0] == S.FAIL
