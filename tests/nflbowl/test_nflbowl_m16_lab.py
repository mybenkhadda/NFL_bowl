"""M16 laboratory (session 5) — scientific invariants on synthetic data.

AUTHORED, NOT EXECUTED by the author: run with ``pytest tests/nflbowl/test_nflbowl_m16_lab.py -q``.
Covered: athlete-disjoint folds, exact target-ID consistency, held-out exclusion from embedding training (provenance),
train-only normalisation, identical-cohort pairing, safe target aggregation, pool-specific chance, cache-signature validity.
Steps that need the research backend (official ladder engine) run in the Colab notebook only.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import polars as pl
import pytest

from nflbowl.lab import status as S
from nflbowl.lab.m16 import cohort as CH
from nflbowl.lab.m16 import config as C
from nflbowl.lab.m16 import controlled_retrieval as CR
from nflbowl.lab.m16 import evaluation as EV
from nflbowl.lab.m16 import features as FT
from nflbowl.lab.m16 import pipeline as PIPE
from nflbowl.lab.m16 import representation_audit as RA
from nflbowl.lab.m16 import target_registry as TRG
from nflbowl.lab.m16 import targets as TG
from nflbowl.lab.m16.baselines import assert_no_nfl_predictors


def _cfg(tmp_path, **kw):
    base = {"drive_root": str(tmp_path / "drive"), **{k: str(tmp_path / "ro" / k) for k in ("m12_run_dir", "m13_run_dir", "m14_run_dir", "m15_run_dir")}}
    base.update(kw)
    return C.resolve(base)


def _index(n_ath=40, n_comb=3, n_nfl=6, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    rows, idx = [], 0
    pos = ["WR", "DB", "OL", "DL"]
    for a in range(n_ath):
        for dom, n, kinds in (("combine", n_comb, ("forty", "three_cone", "short_shuttle")), ("nfl", n_nfl, ("route", "pursuit"))):
            for k in range(n):
                rows.append({"idx": idx, "nfl_id": 1000 + a, "domain": dom, "kind": kinds[k % len(kinds)], "position_group": pos[a % 4], "fold": a % folds,
                             "valid_s": float(rng.uniform(2, 6)), "n_valid": int(rng.integers(20, 64))})
                idx += 1
    return pl.DataFrame(rows)


# ====================================================================== status and configuration
def test_status_vocabulary_has_overlap_and_gate_states():
    assert S.BLOCKED_INSUFFICIENT_OVERLAP in S.STEP_STATUSES and S.BLOCKED_INSUFFICIENT_OVERLAP in S.BLOCKED
    assert {S.EXPLORATORY, S.BLOCKED_GATE} <= set(S.GATE_STATUSES)
    e = S.ResearchBlocked(S.BLOCKED_INSUFFICIENT_OVERLAP, "no support")
    assert e.status == S.BLOCKED_INSUFFICIENT_OVERLAP


def test_config_laboratory_keys(tmp_path):
    c = _cfg(tmp_path, force_steps=["E16.B5"], legacy_m15_dirs=[str(tmp_path / "legacy")])
    assert c.force_steps == ("E16.B5",) and not c.reuse("E16.B5") and c.reuse("E16.A7")
    assert any("legacy" in str(p) for p in c.read_only_dirs())
    with pytest.raises(ValueError):
        _cfg(tmp_path, primary_aggregation="median", sensitivity_aggregations=("median",))
    with pytest.raises(ValueError):
        _cfg(tmp_path, reduced_dims=(8,), embedding_k=8)
    with pytest.raises(ValueError):
        _cfg(tmp_path, retrieval_arms=("procrustes",))
    with pytest.raises(ValueError):
        _cfg(tmp_path, learners=("xgboost",))
    assert "pooled" in C.LEARNERS16


# ====================================================================== athlete-disjoint folds
def test_fold_integrity_and_disjointness():
    idx = _index()
    fold_of = {1000 + a: a % 5 for a in range(40)}
    tab = CH.fold_integrity(idx, fold_of, 5, min_dual_per_fold=5)
    assert (tab["status"] == S.PASS).all()
    bad = idx.with_columns(pl.when((pl.col("nfl_id") == 1003) & (pl.col("domain") == "nfl")).then(4).otherwise(pl.col("fold")).alias("fold"))
    with pytest.raises(S.ValidationFailed):
        CH.fold_integrity(bad, fold_of, 5, min_dual_per_fold=5)
    with pytest.raises(S.ValidationFailed):
        CH.assert_athlete_disjoint([1, 2, 3], [3, 4])
    CH.assert_athlete_disjoint([1, 2], [3, 4])


def test_dual_domain_keeps_both_definitions():
    idx = _index(n_ath=6).filter(~((pl.col("nfl_id") == 1005) & (pl.col("domain") == "nfl")))
    ath = pl.DataFrame({"nfl_id": [1000 + a for a in range(6)] + [2000], "linked": [True] * 6 + [True]})
    per, s = CH.dual_domain(idx, ath)
    assert s["n_observed_dual"] == 5 and s["n_linked_flag"] == 7
    assert set(s["flag_only_ids"]) == {1005, 2000} and s["observed_only_ids"] == []


# ====================================================================== targets
def test_target_registry_consistency_and_versioning():
    TRG.check_consistency()
    assert set(TRG.REGISTRY) == set(TG.TARGETS)
    rec = TRG.definitions_record()
    assert rec["hash"] == TRG.registry_hash() and rec["version"] == TRG.REGISTRY_VERSION
    assert "NOT exact geometric curvature" in TRG.REGISTRY["T5"].eligibility
    with pytest.raises(S.ValidationFailed):
        TG.assert_same_target("T0", "T0x")


def test_safe_target_aggregation_minimum_exposure():
    rng = np.random.default_rng(1)
    rows = []
    for a, n in ((1, 30), (2, 4)):
        for k in range(n):
            rows.append({"idx": a * 100 + k, "nfl_id": a, "position_group": "WR", "kind": "route" if k % 2 else "pursuit", "max_speed": float(rng.uniform(5, 9)),
                         "accel_band_0": float(rng.uniform(1, 3)), "accel_band_1": np.nan, "ends_at_max": False})
    w = TG.window_targets(pl.DataFrame(rows), TG.TargetConfig(min_windows=20, min_windows_kind=5, min_events=5))
    v = dict(zip(w["nfl_id"].to_list(), w[TG.TARGETS["T1"].column].to_list(), strict=True))
    assert v[1] is not None and v[2] is None                               # below the exposure minimum → null, never imputed
    with pytest.raises(S.ValidationFailed):
        assert_no_nfl_predictors(["height", "t16_ctx_forty__T1"])


def test_within_athlete_uncertainty_shrinks_with_exposure():
    rng = np.random.default_rng(2)
    rows = [{"nfl_id": a, "max_speed": float(rng.normal(8, 0.5)), "accel_band_0": np.nan, "accel_band_1": np.nan}
            for a, n in ((1, 8), (2, 200)) for _ in range(n)]
    unc = TRG.athlete_uncertainty(pl.DataFrame(rows), None, pl.DataFrame({"nfl_id": [1, 2], "position_group": ["WR", "WR"]}), 200, 0)
    se = dict(zip(unc.filter(pl.col("target_id") == "T1")["nfl_id"].to_list(), unc.filter(pl.col("target_id") == "T1")["se"].to_list(), strict=True))
    assert se[2] < se[1]


# ====================================================================== provenance: held-out exclusion
def _m15_fixture(root, fold_of, leak_fold=None, folds=range(5), d=4):
    (root / "config").mkdir(parents=True, exist_ok=True)
    from nflbowl.lab.m15.config import CHANNELS, SCALE
    from nflbowl.lab.m15.training import athlete_hash

    (root / "config" / "run_config.json").write_text(json.dumps({"channels": list(CHANNELS), "scale": list(SCALE), "train_domains": ["combine", "nfl"]}))
    ids = sorted(fold_of)
    for f in folds:
        tr = [i for i in ids if fold_of[i] != f]
        if f == leak_fold:
            tr.append(next(i for i in ids if fold_of[i] == f))
        ck = root / "checkpoints" / f"pca_d{d}_s1" / f"fold{f}"
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "pca.npz").write_bytes(b"weights")
        (ck / "train_athletes.json").write_text(json.dumps({"train_and_validation_athletes": tr}))
        (ck / "done.json").write_text(json.dumps({"bundle_fingerprint": "fp", "train_athlete_hash": athlete_hash(tr)}))
        emb = pl.DataFrame({"idx": list(range(len(ids))), "nfl_id": ids, "domain": ["combine"] * len(ids), "kind": ["forty"] * len(ids),
                            "position_group": ["WR"] * len(ids), "fold": [fold_of[i] for i in ids],
                            "role": ["held_out" if fold_of[i] == f else "train" for i in ids], **{f"z{j}": np.random.default_rng(f).normal(size=len(ids)) for j in range(d)}})
        p = root / "embeddings" / "inductive" / f"pca_d{d}_s1__fold{f}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        emb.write_parquet(p)


def test_provenance_detects_heldout_athlete_in_training(tmp_path):
    fold_of = {100 + i: i % 5 for i in range(30)}
    _m15_fixture(tmp_path / "ro" / "m15_run_dir", fold_of, leak_fold=2)
    cfg = _cfg(tmp_path, ladder_dim=4)
    folds, reg, _ = RA.registry(cfg, fold_of, "fp", 5)
    assert folds.filter(pl.col("fold") == 2)["status"][0] == RA.INVALID_LEAKAGE
    assert reg["status"][0] == RA.INVALID_LEAKAGE and not reg["usable_supervised"][0]
    with pytest.raises(S.ResearchBlocked):
        RA.assert_supervised(reg, "pca_d4")


def test_provenance_clean_and_incomplete(tmp_path):
    fold_of = {100 + i: i % 5 for i in range(30)}
    _m15_fixture(tmp_path / "ro" / "m15_run_dir", fold_of)
    cfg = _cfg(tmp_path, ladder_dim=4)
    _, reg, _ = RA.registry(cfg, fold_of, "fp", 5)
    assert reg["status"][0] == RA.VERIFIED_FOLD_SAFE
    rep, why = RA.choose(reg, "pca", 4)
    assert rep is not None and rep["seed"] == 1
    assert RA.choose(reg, "pca", 8)[0] is None                              # never falls back to another d
    _, reg_b, _ = RA.registry(cfg, fold_of, "other_bundle", 5)
    assert reg_b["status"][0] == RA.UNKNOWN_PROVENANCE
    for f in (3, 4):
        (tmp_path / "ro" / "m15_run_dir" / "embeddings" / "inductive" / f"pca_d4_s1__fold{f}.parquet").unlink()
    _, reg_c, _ = RA.registry(cfg, fold_of, "fp", 5)
    assert reg_c["status"][0] == RA.EXPLORATORY_ONLY


# ====================================================================== train-only normalisation
def test_fusion_and_icc_weights_use_training_athletes_only():
    fold_of = {i: i % 5 for i in range(50)}
    rng = np.random.default_rng(3)
    a = {0: pl.DataFrame({"nfl_id": list(range(50)), "e0": rng.normal(size=50)})}
    b = {0: pl.DataFrame({"nfl_id": list(range(50)), "e0": rng.normal(size=50)})}
    f1 = FT.fuse(a, b, fold_of)[0]
    a2 = {0: a[0].with_columns(pl.when(pl.col("nfl_id") % 5 == 0).then(pl.col("e0") * 100).otherwise(pl.col("e0")).alias("e0"))}
    f2 = FT.fuse(a2, b, fold_of)[0]
    tr = pl.col("nfl_id") % 5 != 0
    assert np.allclose(f1.filter(tr)["e0"].to_numpy(), f2.filter(tr)["e0"].to_numpy())
    rows = [{"nfl_id": i, "kind": "forty", "z0": float(i % 7 + rng.normal(0, 0.1))} for i in range(50) for _ in range(3)]
    e = pl.DataFrame(rows)
    w1 = FT.icc_weights(e, ["z0"], {i for i in range(50) if i % 5})
    e2 = e.with_columns(pl.when(pl.col("nfl_id") % 5 == 0).then(pl.lit(999.0)).otherwise(pl.col("z0")).alias("z0"))
    assert np.allclose(w1, FT.icc_weights(e2, ["z0"], {i for i in range(50) if i % 5}))


def test_kinematic_control_standardised_on_training_rows():
    idx = _index()
    feat = idx.select("idx").with_columns(*[pl.Series(c, np.random.default_rng(4).normal(size=idx.height)) for c in FT.KINEMATICS])
    e1 = CR.kinematic_embedding(feat, idx, 0, FT.KINEMATICS)
    held = idx["fold"].to_numpy() == 0
    feat2 = feat.with_columns(pl.Series("max_speed", np.where(held, 1e6, feat["max_speed"].to_numpy())))
    e2 = CR.kinematic_embedding(feat2, idx, 0, FT.KINEMATICS)
    assert np.allclose(e1.filter(~pl.Series(held))["z0"].to_numpy(), e2.filter(~pl.Series(held))["z0"].to_numpy())


def test_aggregations_and_missingness():
    e = pl.DataFrame({"nfl_id": [1, 1, 1, 1, 1, 2], "kind": ["forty"] * 6, "z0": [1.0, 2.0, 3.0, 4.0, 100.0, 5.0]})
    assert FT.aggregate(e, "median").filter(pl.col("nfl_id") == 1)["e0"][0] == 3.0
    assert FT.aggregate(e, "trimmed_mean", trim=0.2).filter(pl.col("nfl_id") == 1)["e0"][0] == pytest.approx(3.0)
    assert FT.aggregate(e, "quantiles").width == 2 + 2                     # nfl_id, n_attempts, q25, q75
    with pytest.raises(ValueError):
        FT.aggregate(e, "icc_weighted")
    t = {0: pl.DataFrame({"nfl_id": [1], "e0": [0.5]})}
    block = FT.block_factory({1: 0, 2: 0, 3: 1}, t, None)(np.array([1, 2, 3]))(np.array([False, False, True]))
    assert np.isnan(block["e0"][1])                                        # absent athlete → NaN (missing indicator downstream)


# ====================================================================== cohort pairing and chance
def test_cohort_audit_flags_extra_athletes():
    long = pl.DataFrame({"model": ["M1"] * 3 + ["M1+PCA"] * 3, "nfl_id": [1, 2, 3] * 2, "pred": [1.0, 2.0, 3.0, 1.0, np.nan, 3.0]})
    a = EV.cohort_audit(long, ["M1", "M1+PCA"])
    assert a.filter(pl.col("model") == "M1")["identical_to_strict"][0] is False
    assert a["n_strict_cohort"][0] == 2


def test_pool_specific_chance_and_perfect_retrieval():
    from nflbowl.lab.stats import chance_mrr, chance_recall_at_k

    rng = np.random.default_rng(5)
    rows, k = [], 0
    for a in range(30):
        z = rng.normal(size=4)
        for dom in ("combine", "nfl"):
            rows.append({"idx": k, "nfl_id": a, "domain": dom, "kind": "forty" if dom == "combine" else "route", "position_group": "WR" if a % 2 else "DB",
                         "fold": 0, "role": "held_out" if a < 20 else "train", **{f"z{j}": float(z[j]) for j in range(4)}})
            k += 1
    emb = pl.DataFrame(rows)
    attrs = {"forty": {a: 4.5 + 0.01 * a for a in range(30)}, "height": {a: 72.0 for a in range(30)}, "weight": {a: 200.0 for a in range(30)}}
    cfg = C.M16Config()
    r, _, meta = CR.score_fold(emb, {"rep": "x", "fold": 0, "scope": "all"}, None, attrs, cfg, ("unaligned",), 0, False)
    s = pl.DataFrame(r)
    assert s.filter(pl.col("arm") == "unaligned")["top1"].mean() == 1.0      # identical domains → perfect retrieval
    allp = s.filter(pl.col("pool") == "all")["n_candidates"].to_numpy()
    assert (allp == 19).all()
    summ = CR.summarise(s, 100, 0)
    row = summ.filter(pl.col("pool") == "same_position").row(0, named=True)
    n = s.filter(pl.col("pool") == "same_position")["n_candidates"].to_numpy()
    assert row["chance_top1"] == pytest.approx(chance_recall_at_k(1, n)) and row["chance_mrr"] == pytest.approx(chance_mrr(n))
    po = CR.position_only_rows(s)
    same = po.filter(pl.col("pool") == "same_position").row(0, named=True)
    m = same["n_candidates"] + 1
    assert same["rr"] == pytest.approx(sum(1 / i for i in range(1, m + 1)) / m)
    a_ = po.filter(pl.col("pool") == "all").row(0, named=True)
    assert a_["top1"] == pytest.approx(1 / 10)                              # 10 held-out athletes per position in the all pool


# ====================================================================== cache signatures
def test_cache_reloads_only_on_matching_signature(tmp_path):
    cfg = _cfg(tmp_path)
    run = C.M16Run(cfg.run_dir, cfg)
    ctx = {"outputs": []}
    inputs = {"bundle": "a", "files": [["x", 1, 2]]}
    PIPE.save_step(run, ctx, "E16.T", "exploration", {"t": pl.DataFrame({"a": [1, 2]})}, inputs)
    assert PIPE.load_step(run, cfg, "E16.T", "exploration", inputs)["t"].height == 2
    assert PIPE.load_step(run, cfg, "E16.T", "exploration", {**inputs, "bundle": "b"}) is None
    forced = _cfg(tmp_path, force_steps=("E16.T",))
    assert PIPE.load_step(run, forced, "E16.T", "exploration", inputs) is None
    (run.root / "tables" / "exploration" / "t.parquet").unlink()
    assert PIPE.load_step(run, cfg, "E16.T", "exploration", inputs) is None  # a missing file is never reconstructed from the signature


def test_file_identity_tracks_content_not_name(tmp_path):
    p = tmp_path / "e.parquet"
    p.write_bytes(b"abc")
    a = PIPE.file_identity([p])
    time.sleep(0.01)
    p.write_bytes(b"abcdef")
    os.utime(p, None)
    b = PIPE.file_identity([p])
    assert a[0][0] == b[0][0] and a != b
