"""Regression tests of the Milestone 13 modules (``nflbowl.lab.m13_*``, handoff, target_audit, transfer_fairness, reliability,
position_models, domain_matching, scouting_value). Written WITHOUT being executed in the authoring session — run with
``pytest tests/nflbowl/test_nflbowl_m13.py -q``. Synthetic data only; no bundle, GPU or Drive needed."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nflbowl.lab import domain_matching as DM
from nflbowl.lab import handoff, m13_config, m13_reporting, position_models, reliability, scouting_value, target_audit
from nflbowl.lab import transfer_fairness as TF


# ---------------------------------------------------------------------- helpers
def _wide(n: int = 60, seed: int = 0, better: float = 0.0) -> pl.DataFrame:
    """Synthetic aligned OOF frame: 5 folds, 3 positions, y, POS / M1 / A predictions (A closer to y by ``better``)."""
    rng = np.random.default_rng(seed)
    y = rng.normal(3.0, 0.4, n)
    m1 = y + rng.normal(0, 0.3, n)
    a = y + (m1 - y) * (1 - better)
    return pl.DataFrame({"nfl_id": np.arange(1000, 1000 + n), "position_group": [("WR", "CB", "DL")[i % 3] for i in range(n)], "fold": np.arange(n) % 5,
                         "y": y, "pred_POS": np.full(n, 3.0), "pred_M1": m1, "pred_A": a})


def _long(w: pl.DataFrame, source: str = "s") -> pl.DataFrame:
    return TF.long_predictions(w, source)


# ---------------------------------------------------------------------- config / store
def test_config_policy():
    c = m13_config.resolve({})
    assert c.precision == "fp32" and not c.training_enabled() and c.run_dir.name == "m13_outcome_transfer_v1"
    with pytest.raises(KeyError):
        m13_config.resolve({"nope": 1})
    with pytest.raises(ValueError):
        m13_config.resolve({"run_training": True})                    # second opt-in missing
    assert m13_config.resolve({"run_training": True, "allow_new_training": True}).training_enabled()
    with pytest.raises(ValueError):
        m13_config.resolve({"precision": "fp16"})
    with pytest.raises(ValueError):
        m13_config.resolve({"encoders": ["F9"]})
    assert "science.seed" not in c.experiment_overrides()


def test_m13run_tables_signature_cache_and_reload(tmp_path):
    run = m13_config.M13Run(tmp_path / "m13", m13_config.resolve({}))
    df = pl.DataFrame({"a": [1, 2, 3]})
    run.save_all({"t": df}, "transfer", signature=["v1", 3])
    assert (tmp_path / "m13" / "tables" / "transfer" / "t.csv").is_file()
    assert run.cached("transfer", ["t"], signature=["v1", 3])["t"].height == 3
    assert run.cached("transfer", ["t"], signature=["v2", 3]) is None          # changed inputs → recompute
    assert run.cached("transfer", ["missing"]) is None
    with pytest.raises(ValueError):
        run.table(df, "x", "dataset")                                          # M12 family is not an M13 family
    run.set_provenance(bundle_fingerprint="abc")
    lines = (tmp_path / "m13" / "manifests" / "table_provenance.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[-1])["table"] == "transfer/t"
    run2 = m13_config.M13Run(tmp_path / "m13", m13_config.resolve({}))       # a new session reloads the saved table
    assert run2.frame("t", "transfer").height == 3


# ---------------------------------------------------------------------- fold isolation / alignment
def test_alignment_rejects_fold_drift_target_mismatch_and_duplicates():
    w = _wide()
    long = _long(w)
    fmap = dict(zip(w["nfl_id"].to_list(), w["fold"].to_list()))
    assert TF.validate_alignment(long, fmap).filter(pl.col("status") == "pass").height >= 4
    bad_fold = pl.concat([long, _long(w.with_columns(((pl.col("fold") + 1) % 5).alias("fold")).select("nfl_id", "position_group", "fold", "y",
                                                                                                          pl.col("pred_A").alias("pred_B")), "t")])
    with pytest.raises(TF.ComparisonError):
        TF.validate_alignment(bad_fold, fmap)
    bad_y = pl.concat([long, _long(w.with_columns((pl.col("y") + 0.1).alias("y")).select("nfl_id", "position_group", "fold", "y", pl.col("pred_A").alias("pred_B")), "t")])
    with pytest.raises(TF.ComparisonError):
        TF.validate_alignment(bad_y, fmap)
    with pytest.raises(TF.ComparisonError):
        TF.validate_alignment(pl.concat([long, long.head(3)]), fmap)
    with pytest.raises(TF.ComparisonError):                                    # fold map of the experiment differs
        TF.validate_alignment(long, {k: (v + 1) % 5 for k, v in fmap.items()})


def test_common_cohort_join_and_native_flags():
    w = _wide(40)
    long = _long(w)
    oof = w.select("nfl_id").with_columns(pl.Series("has_X", [i % 4 != 0 for i in range(40)]), pl.Series("forty", [None if i % 10 == 0 else 4.5 for i in range(40)]),
                                         *[pl.lit(1.0).alias(c) for c in ("draft_year", "height", "weight", "ten")])
    f0 = set(w["nfl_id"].to_list()[:-2])
    coh, desc = TF.cohorts(long, oof, ["POS", "M1", "A"], f0, ["X"])
    strict = coh["strict_common"]
    assert strict <= coh["ladder_all"] and len(coh["ladder_all"]) == 40
    assert all(i % 4 != 0 for i in (k - 1000 for k in strict))                 # missing embedding excluded
    ws = TF.wide(long, strict, ["POS", "M1", "A"])
    assert ws.height == len(strict) and ws["nfl_id"].n_unique() == ws.height
    with pytest.raises(TF.ComparisonError):
        TF.wide(long, strict, ["POS", "M1", "NOT_A_MODEL"])
    nonfin = long.with_columns(pl.when((pl.col("model") == "A") & (pl.col("nfl_id") == 1001)).then(float("nan")).otherwise(pl.col("pred")).alias("pred"))
    with pytest.raises(TF.ComparisonError):
        TF.wide(nonfin, {1001, 1002, 1003}, ["M1", "A"])


# ---------------------------------------------------------------------- paired bootstrap
def test_paired_bootstrap_sign_identity_and_shared_resamples():
    w = _wide(80, better=0.5)
    p = TF.paired_comparisons(w, [("A", "M1"), ("M1", "M1")], 500, 1, "c", primary_family=[("A", "M1")])
    a = p.filter(pl.col("model") == "A").row(0, named=True)
    assert a["delta_rmse"] > 0 and a["delta_lo"] > 0 and a["verdict"] == "model_better" and a["p_holm_primary"] is not None
    same = p.filter((pl.col("model") == "M1") & (pl.col("reference") == "M1")).row(0, named=True)
    assert same["delta_rmse"] == 0 and same["delta_lo"] == 0 and same["delta_hi"] == 0
    idx = TF.bootstrap_index(w["fold"].to_numpy(), 50, 0)
    folds = w["fold"].to_numpy()
    for b in range(5):                                                          # fold-stratified: fold sizes preserved
        assert np.array_equal(np.bincount(folds[idx[b]], minlength=5), np.bincount(folds, minlength=5))
    with pytest.raises(TF.ComparisonError):
        TF.paired_comparisons(w.head(10), [("A", "M1")], 100, 0, "tiny")


def test_cohort_metrics_skill_vs_position():
    w = _wide(60)
    m = TF.cohort_metrics(w, ["POS", "M1", "A"], "c")
    pos = m.filter(pl.col("model") == "POS").row(0, named=True)
    assert abs(pos["skill_vs_position_oof"]) < 1e-12 and m["n_athletes"].unique().to_list() == [60]


# ---------------------------------------------------------------------- reliability
def test_icc_components_known_structure_and_no_clamping():
    rng = np.random.default_rng(0)
    ids = np.repeat(np.arange(200), 3)
    true = np.repeat(rng.normal(0, 1, 200), 3)
    v = true + rng.normal(0, 1, ids.size)                                       # ICC(1,1) = 1 / (1 + 1) = 0.5
    c = reliability.icc_components(v, ids)
    assert 0.38 < c["icc1"] < 0.62 and c["n_athletes"] == 200 and c["k0"] == pytest.approx(3.0)
    neg = np.tile([1.0, -1.0], 200) + rng.normal(0, 0.01, 400)                  # within-athlete variance ≫ between
    cn = reliability.icc_components(neg, np.repeat(np.arange(200), 2))
    assert cn["icc1"] < 0 and cn["icc1"] >= cn["icc1_lower_bound"] - 1e-9 and cn["var_between_raw"] < 0
    assert reliability.spearman_brown_raw(-0.9) == pytest.approx(-18.0)        # the M12 estimator explodes; reported, not clamped
    assert reliability.icc_average(0.5, 2) == pytest.approx(2 / 3)


def test_kind_centring_uses_reference_athletes_only():
    t = pl.DataFrame({"nfl_id": [1, 1, 2, 2], "kind": ["a", "b", "a", "b"], "order": [0, 0, 0, 0], "value": [1.0, 10.0, 3.0, 30.0]})
    full = reliability.kind_centre(t)
    only1 = reliability.kind_centre(t, {1})
    assert full[0] == pytest.approx(-1.0) and only1[0] == pytest.approx(0.0) and only1[2] == pytest.approx(2.0)


def _selection_frame(n: int = 80, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"nfl_id": np.arange(n), "fold": np.arange(n) % 5, "position_group": [("WR", "CB")[i % 2] for i in range(n)],
                         "nfl.q90_shared": rng.normal(size=n), "draft_year": rng.integers(2020, 2023, n).astype(float), "height": rng.normal(73, 2, n),
                         "weight": rng.normal(210, 20, n), "forty": rng.normal(4.6, 0.1, n), "ten": rng.normal(1.6, 0.05, n),
                         "c_forty__peak_speed": rng.normal(9, 0.5, n), "c_forty__peak_accel": rng.normal(5, 0.5, n), "c_agility__peak_decel": rng.normal(6, 0.5, n)})


def test_no_held_out_outcome_enters_train_selection():
    d = _selection_frame()
    cand = ["c_forty__peak_speed", "c_forty__peak_accel", "c_agility__peak_decel"]
    ids = d["nfl_id"].to_numpy()
    train = d["fold"].to_numpy() != 0
    log1, log2 = reliability.SelectionLog(), reliability.SelectionLog()
    reliability.trainsel_block(d, cand, 2, log1)(ids)(train)
    d2 = d.with_columns(pl.when(pl.col("fold") == 0).then(pl.col("nfl.q90_shared") * 100 + 7).otherwise(pl.col("nfl.q90_shared")).alias("nfl.q90_shared"))
    reliability.trainsel_block(d2, cand, 2, log2)(ids)(train)
    assert log1[0]["selected"] == log2[0]["selected"] and log1[0]["fold"] == 0
    assert all(np.isclose(log1[0]["scores"][c], log2[0]["scores"][c], equal_nan=True) for c in cand)


def test_reliability_selection_ignores_held_out_trials():
    d = _selection_frame()
    rng = np.random.default_rng(1)
    rows = []
    for i in d["nfl_id"].to_list():
        base = rng.normal()
        for k in range(2):
            rows.append({"nfl_id": i, "kind": "forty", "order": k, "value": base + rng.normal(0, 0.3)})
    cache = {"c_forty__peak_speed": pl.DataFrame(rows)}
    held = set(d.filter(pl.col("fold") == 0)["nfl_id"].to_list())
    cache2 = {"c_forty__peak_speed": cache["c_forty__peak_speed"].with_columns(
        pl.when(pl.col("nfl_id").is_in(list(held))).then(pl.col("value") * -50).otherwise(pl.col("value")).alias("value"))}
    ids, train = d["nfl_id"].to_numpy(), d["fold"].to_numpy() != 0
    l1, l2 = reliability.SelectionLog(), reliability.SelectionLog()
    reliability.reliable_block(d, ["c_forty__peak_speed"], cache, 0.5, 10, l1)(ids)(train)
    reliability.reliable_block(d, ["c_forty__peak_speed"], cache2, 0.5, 10, l2)(ids)(train)
    assert l1[0]["scores"] == l2[0]["scores"] and l1[0]["selected"] == l2[0]["selected"]


def test_train_fold_standardised_interaction_block():
    d = _selection_frame()
    ids, train = d["nfl_id"].to_numpy(), d["fold"].to_numpy() != 0
    b1 = reliability.interaction_factory(d, "c_forty__peak_speed", "c_forty__peak_accel")(ids)(train)
    d2 = d.with_columns(pl.when(pl.col("fold") == 0).then(pl.col("c_forty__peak_speed") + 100).otherwise(pl.col("c_forty__peak_speed")).alias("c_forty__peak_speed"))
    b2 = reliability.interaction_factory(d2, "c_forty__peak_speed", "c_forty__peak_accel")(ids)(train)
    tr = np.flatnonzero(train)
    assert np.allclose(b1.to_numpy()[tr], b2.to_numpy()[tr])                   # train rows unaffected by held-out values


# ---------------------------------------------------------------------- position standardisation (train-only parameters)
def test_position_params_are_train_only():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    pos = np.array(["A", "B"] * 25)
    train = np.arange(50) >= 10
    mu, sd, gm, gs = position_models.position_params(y, pos, train)
    y2 = y.copy()
    y2[~train] += 1000.0
    mu2, sd2, gm2, gs2 = position_models.position_params(y2, pos, train)
    assert mu == mu2 and sd == sd2 and gm == gm2 and gs == gs2


# ---------------------------------------------------------------------- domain matching
def test_cem_balances_strata_and_uses_covariates_only():
    rng = np.random.default_rng(0)
    dom = np.array(["combine"] * 300 + ["nfl"] * 700)
    strata = np.r_[rng.integers(0, 3, 300), rng.integers(1, 4, 700)]
    strata[5] = -1
    keep = DM.cem(dom, strata, 0)
    for s in np.unique(strata[keep]):
        k = keep[strata[keep] == s]
        assert (dom[k] == "combine").sum() == (dom[k] == "nfl").sum()
    assert 5 not in set(keep.tolist()) and 3 not in set(strata[keep].tolist())
    assert np.array_equal(keep, DM.cem(dom, strata, 0))                        # seeded, reproducible
    b = DM.quantile_bins(np.r_[np.arange(10.0), np.nan], 4)
    assert b[-1] == -1 and b[:-1].min() >= 0


# ---------------------------------------------------------------------- checkpoints (discovery / missing handling)
class _Cfg:
    def __init__(self, sh: str):
        self._sh = sh
        self.science = None

    def science_hash(self) -> str:
        return self._sh


def _fake_exp(tmp_path: Path, fp: str = "fp1", sh: str = "sh1") -> types.SimpleNamespace:
    return types.SimpleNamespace(bundle=types.SimpleNamespace(fingerprint=fp), cfg=_Cfg(sh), root=tmp_path / "m13exp")


def _source(tmp_path: Path, records: list[dict]) -> Path:
    d = tmp_path / "m12exp"
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "manifests").mkdir(parents=True, exist_ok=True)
    (d / "config" / "experiment.json").write_text(json.dumps({"experiment_id": "m12"}), encoding="utf-8")
    (d / "manifests" / "manifest.json").write_text(json.dumps({"artifacts": {r["key"] + str(i): r for i, r in enumerate(records)}}), encoding="utf-8")
    return d


def test_discover_checkpoint_compatibility_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "expected_architecture", lambda exp, name: {"dims": 32})
    best = tmp_path / "m12exp" / "ck" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"x")
    best.with_suffix(".json").write_text(json.dumps({"enc_config": {"dims": 32}}), encoding="utf-8")
    ok = {"kind": "training", "key": "F4_physics_contrastive__inductive_f0", "path": str(best), "fold": 0, "sha256": "s", "created_at": "t",
          "deps": {"bundle": "fp1", "science": "sh1", "regime": "inductive", "tag": "inductive_f0"}}
    recs = [ok, {**ok, "deps": {**ok["deps"], "bundle": "other"}}, {**ok, "deps": {**ok["deps"], "regime": "transductive", "tag": "transductive"},
                                                                     "key": "F4_physics_contrastive__transductive"},
            {**ok, "fold": 1}, {**ok, "path": str(tmp_path / "missing.pt")}]
    d = _source(tmp_path, recs)
    cfg = m13_config.resolve({"m12_run_dir": str(tmp_path / "nowhere"), "extra_source_experiments": [str(d)]})
    t = handoff.discover(cfg, _fake_exp(tmp_path))
    st = t["status"].to_list()
    assert st[0] == "compatible" and all(s.startswith("incompatible") for s in st[1:])
    assert "bundle" in st[1] and "regime" in st[2] and "fold identity" in st[3] and "best.pt missing" in st[4]
    monkeypatch.setattr(handoff, "expected_architecture", lambda exp, name: {"dims": 64})
    assert "architecture" in handoff.discover(cfg, _fake_exp(tmp_path))["status"][0]


def test_ready_models_excludes_partially_available_encoders():
    inv = pl.DataFrame({"model": ["F0_engineered"] * 2 + ["F2"] * 2 + ["F4"] * 2, "fold": [0, 1] * 3,
                        "checkpoint_found": [None, None, True, True, True, False], "ready_for_transfer": [True, True, True, True, True, False]})
    ready, missing = handoff.ready_models(inv, [0, 1])
    assert ready == ["F2"] and missing == {"F4": [1]}


def test_reconcile_reports_differences_without_substitution():
    ir = pl.DataFrame({"representation": ["F4_physics_contrastive", "F4_physics_contrastive"], "held_out_fold": [0, 1], "auc": [0.5126, 0.5126]})
    rec = handoff.reconcile({"experiment/inductive_retrieval": ir})
    f4 = rec.filter((pl.col("model") == "F4_physics_contrastive") & pl.col("quantity").str.contains("retrieval")).row(0, named=True)
    assert f4["status"] == "consistent" and f4["observed_m12_artifact"] == pytest.approx(0.5126)
    assert rec.filter(pl.col("model") == "F2_cnn_physics")["status"][0] == "artifact missing"


def test_f1_validity_detects_leakage(tmp_path):
    fmap = {1: 0, 2: 0, 3: 1, 4: 1}
    tcn = pl.DataFrame({"family": ["TCN"] * 4, "fold": [0, 0, 1, 1], "nfl_id": [1, 3, 3, 1], "dim": [0] * 4, "value": [0.0] * 4, "role": ["test", "train", "test", "train"]})
    exp = types.SimpleNamespace(bundle=types.SimpleNamespace(tcn=tcn), folds_by_athlete=fmap)
    v, _ = handoff.f1_validity(exp)
    assert v["verdict"] == "valid"
    leak = tcn.with_columns(pl.when(pl.col("nfl_id") == 1).then(pl.lit("train")).otherwise(pl.col("role")).alias("role"))
    assert handoff.f1_validity(types.SimpleNamespace(bundle=types.SimpleNamespace(tcn=leak), folds_by_athlete=fmap))[0]["verdict"] == "leaky"
    noroles = tcn.with_columns(pl.lit("all").alias("role"))
    assert handoff.f1_validity(types.SimpleNamespace(bundle=types.SimpleNamespace(tcn=noroles), folds_by_athlete=fmap))[0]["verdict"] == "unverifiable"


# ---------------------------------------------------------------------- target provenance
def test_target_provenance_states_fitting_scope():
    ath = pl.DataFrame({"nfl_id": [1, 2, 3], "position_group": ["WR", "CB", "WR"], "linked": [True, True, False], "nfl.q90_shared": [3.0, None, 2.5]})
    exp = types.SimpleNamespace(bundle=types.SimpleNamespace(info={"transfer_target": "nfl.q90_shared", "source": {"target_freeze": {"config_hash": "c2903a12f8449fa2"}}},
                                                             athletes=ath))
    t = target_audit.target_provenance(exp, {"estimator": "C3_pooled", "freeze_hash": "h"})
    scope = t.filter(pl.col("aspect") == "estimator-fitting scope").row(0, named=True)
    assert "not strictly inductive" in scope["implication"]
    assert "C3_pooled" in t.filter(pl.col("aspect") == "estimator")["finding"][0]
    assert "1 of 3" in t.filter(pl.col("aspect") == "missingness")["finding"][0]


def test_variance_decomposition_and_attrition():
    vd = target_audit.variance_decomposition(np.array([1.0, 1.0, 3.0, 3.0]), np.array(["a", "a", "b", "b"]))
    assert vd["eta2"] == pytest.approx(1.0) and vd["within_var"] == pytest.approx(0.0)
    sets = {"all athletes (bundle)": {1, 2, 3, 4}, "NFL tracking": {1, 2, 3}, "linked": {1, 2}, "frozen target": {2}}
    f = target_audit.attrition_flow(sets)
    assert f["n_remaining"].to_list() == [4, 3, 2, 1] and f["n_removed"].to_list() == [0, 1, 1, 1]


# ---------------------------------------------------------------------- scouting
def test_oof_intervals_use_other_folds_only():
    w = _wide(100)
    out, info = scouting_value.oof_intervals(w, "M1", 0.8, min_position=1000)   # pooled scope
    assert out.height == 100 and 0.0 <= info["coverage"] <= 1.0
    w2 = w.with_columns(pl.when(pl.col("nfl_id") == 1000).then(pl.col("y") + 50).otherwise(pl.col("y")).alias("y"))
    out2, _ = scouting_value.oof_intervals(w2, "M1", 0.8, min_position=1000)
    a, b = out.filter(pl.col("nfl_id") == 1000).row(0, named=True), out2.filter(pl.col("nfl_id") == 1000).row(0, named=True)
    assert a["lo"] == pytest.approx(b["lo"]) and a["hi"] == pytest.approx(b["hi"])  # own residual never calibrates own interval


def test_best_model_requires_validated_interval():
    p = pl.DataFrame({"cohort": ["c"] * 2, "model": ["A", "B"], "reference": ["M1", "M1"], "delta_rmse": [0.05, 0.01], "delta_lo": [-0.01, 0.002], "delta_hi": [0.1, 0.02]})
    b = scouting_value.best_model(p, "c")
    assert b["model"] == "B" and b["validated"]
    nb = scouting_value.best_model(p.with_columns(pl.lit(-0.01).alias("delta_lo")), "c")
    assert nb["model"] == "A" and not nb["validated"]


# ---------------------------------------------------------------------- synthesis / reports
def _paired(rows: list[tuple[str, float, float, float, float]]) -> pl.DataFrame:
    return pl.DataFrame([{"cohort": "strict_common", "model": m, "reference": "M1", "n_athletes": 200, "rmse_ref": 0.30, "rmse_model": 0.30 - d, "delta_rmse": d,
                          "delta_lo": lo, "delta_hi": hi, "p_boot_le0": p} for m, d, lo, hi, p in rows])


def test_outcome_rule():
    enc, bio = ["M1+F4"], ["M1+BIO_reliable"]
    assert m13_reporting.determine_outcome(_paired([("M1+F4", 0.02, 0.01, 0.03, 0.001), ("M1+BIO_reliable", 0.0, -0.002, 0.002, 0.5)]), "strict_common", enc, bio)["outcome"] == "A"
    assert m13_reporting.determine_outcome(_paired([("M1+F4", 0.0, -0.002, 0.002, 0.5), ("M1+BIO_reliable", 0.02, 0.01, 0.03, 0.001)]), "strict_common", enc, bio)["outcome"] == "B"
    assert m13_reporting.determine_outcome(_paired([("M1+F4", 0.02, 0.01, 0.03, 0.001), ("M1+BIO_reliable", 0.02, 0.01, 0.03, 0.001)]), "strict_common", enc, bio)["outcome"] == "C"
    assert m13_reporting.determine_outcome(_paired([("M1+F4", 0.0, -0.002, 0.002, 0.5), ("M1+BIO_reliable", 0.0, -0.002, 0.002, 0.5)]), "strict_common", enc, bio)["outcome"] == "D"
    wide_ci = _paired([("M1+F4", 0.0, -0.05, 0.05, 0.5), ("M1+BIO_reliable", 0.0, -0.05, 0.05, 0.5)])
    assert m13_reporting.determine_outcome(wide_ci, "strict_common", enc, bio)["outcome"] == "INCONCLUSIVE"
    assert m13_reporting.determine_outcome(_paired([("M1+F4", 0.02, 0.01, 0.03, 0.001)]), "strict_common", enc, bio)["outcome"] == "INCONCLUSIVE"


def test_report_generation_from_tables(tmp_path):
    run = m13_config.M13Run(tmp_path / "m13", m13_config.resolve({}))
    paired = _paired([("M1+F4", 0.0123, 0.004, 0.021, 0.002)])
    out = m13_reporting.determine_outcome(paired, "strict_common", ["M1+F4"], [])
    R = {"outcome": out, "paired_strict": paired, "cohorts_desc": pl.DataFrame({"cohort": ["ladder_all", "strict_common"], "n_athletes": [291, 250],
                                                                                  "definition": ["a", "b"]}), "stages": None}
    md = m13_reporting.final_report(run, run.cfg, {"versions": {"nflbowl": "0.5.0"}}, R)
    assert "0.0123" in md and "250" in md and "not evaluated in this run" in md
    paths = m13_reporting.write_reports(run, md)
    html = (run.root / paths["html"]).read_text(encoding="utf-8")
    assert (run.root / paths["markdown"]).is_file() and "<main>" in html and "M13" in html
    txt = m13_reporting.read_paired(paired, "strict_common", "M1")
    assert "interval excludes 0" in txt and "+0.0123" in txt
