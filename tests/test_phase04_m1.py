"""Static tests for Phase 04 · Milestone 1 (synthetic data only — no warehouse access).

Authored 2026-10-11, NOT executed by Claude. Run::

    python -m pytest tests/test_phase04_m1.py tests/test_polars2_compat.py -q
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from nflbdb.features.config import KinematicsConfig
from nflbdb.features.trajectory import COMBINE
from nflbdb.research.phase04 import annotation_design as ad
from nflbdb.research.phase04 import events as c1
from nflbdb.research.phase04 import quality_eligibility as q
from nflbdb.research.phase04 import smoothing as b
from nflbdb.research.phase04 import validation_registry as v
from nflbdb.research.phase04.config import (AnnotationDesignConfig, EligibilityConfig, EventStudyConfig, Phase04Config,
                                            SmoothingStudyConfig, ValidationConfig, threshold_table)
from nflbdb.research.phase04.metrics import (agreement, cohen_kappa, percentile_rank, rank_change, top_fraction_overlap,
                                             variance_share)
from nflbdb.research.studies.store import StudyStore


# ============================================================================== metrics
def test_top_fraction_overlap_identity_and_reversal():
    x = np.arange(100, dtype=float)
    assert top_fraction_overlap(x, x, 0.1) == 1.0
    assert top_fraction_overlap(-x, x, 0.1) == 0.0


def test_top_fraction_overlap_drops_nonfinite_pairs():
    x = np.r_[np.arange(20, dtype=float), np.nan]
    assert top_fraction_overlap(x, np.r_[np.arange(20, dtype=float), 5.0], 0.1) == 1.0


def test_agreement_identical_and_scaled():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 1, 200)
    a = agreement(x, x)
    assert a["spearman"] == pytest.approx(1.0) and a["median_rel_diff"] == 0.0 and a["bias"] == 0.0
    s = agreement(1.1 * x, x)                       # monotone rescale: ranks identical, magnitudes 10 % larger
    assert s["spearman"] == pytest.approx(1.0) and s["top_overlap"] == 1.0
    assert s["median_rel_diff"] == pytest.approx(0.1)
    assert s["rank_change_p90"] == pytest.approx(0.0)


def test_agreement_too_few_pairs_is_nan():
    a = agreement([1.0, 2.0], [1.0, 2.0])
    assert a["n"] == 2 and np.isnan(a["spearman"])


def test_percentile_rank_and_rank_change():
    pr = percentile_rank([3.0, 1.0, 2.0])
    assert np.allclose(pr, [5 / 6, 1 / 6, 3 / 6])
    assert np.allclose(rank_change([1, 2, 3], [3, 2, 1]), [2 / 3, 0, 2 / 3])


def test_variance_share_zero_when_settings_agree_and_drops_incomplete_units():
    m = np.column_stack([np.arange(10.0)] * 4)
    assert variance_share(m)["share_setting"] == 0.0
    m2 = m.copy()
    m2[0, 1] = np.nan
    assert variance_share(m2)["n_units"] == 9
    noisy = m + np.random.default_rng(1).normal(0, 5, m.shape)
    assert variance_share(noisy)["share_setting"] > variance_share(m + 0.01 * np.arange(4))["share_setting"]


def test_cohen_kappa_bounds():
    a = np.array([True, False] * 50)
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    assert cohen_kappa(a, ~a) == pytest.approx(-1.0)


# ============================================================================== Study B: features
def _kin(n=10, boundary=3):
    i = np.arange(n)
    return pl.DataFrame({
        "event_id": ["E1"] * n, "nfl_id": [1] * n, "attempt": [1] * n, "draft_year": [2024] * n,
        "drill_type": ["FORTY_YARD_DASH"] * n, "drill_name": ["FORTY_YARD_DASH"] * n, "drill": ["FORTY_YARD_DASH"] * n,
        "valid_kin": [True] * n, "in_filter_boundary": (i < boundary) | (i >= n - boundary),
        "speed": (i + 1).astype(float), "a_tan": np.r_[np.full(n - 1, 2.0), -4.0], "accel_mag": np.full(n, 3.0),
        "yaw_rate": np.linspace(-10, 10, n), "step_smooth": np.full(n, 0.5)})


def _events(rules=("default", "strict"), types=("braking", "braking")):
    k = len(rules)
    return pl.DataFrame({"event_id": ["E1"] * k, "nfl_id": [1] * k, "attempt": [1] * k, "rule": list(rules),
                         "event_type": list(types)})


def test_trajectory_features_peak_vs_interior_and_rates():
    f = b.trajectory_features(_kin(), _events(), COMBINE, KinematicsConfig()).row(0, named=True)
    assert f["peak_speed"] == 10.0                 # last frame — inside the filter boundary
    assert f["peak_speed_interior"] == 7.0         # frames 3..6 only
    assert f["peak_decel"] == 4.0 and f["peak_decel_interior"] == -2.0
    assert f["path_smooth_yds"] == pytest.approx(5.0)
    assert f["duration_valid_s"] == pytest.approx(1.0)
    assert f["n_braking"] == 1.0 and f["n_turn"] == 0.0        # only the primary (default) rule is counted
    assert f["rate_braking_per_s"] == pytest.approx(1.0)
    assert set(b.FEATURE_NAMES) <= set(f)


def test_trajectory_features_without_events_gives_zero_counts():
    f = b.trajectory_features(_kin(), _events(rules=(), types=()), COMBINE, KinematicsConfig()).row(0, named=True)
    assert f["n_braking"] == 0.0 and f["rate_reversal_per_s"] == 0.0


def test_band_expr_labels():
    lab, low = b.band_expr("s", (0.0, 1.0, 3.0))
    out = pl.DataFrame({"s": [0.5, 1.0, 2.9, 7.0]}).select(lab, low)
    assert out["speed_band"].to_list() == ["[0, 1)", "[1, 3)", "[1, 3)", "≥ 3"]
    assert out["band_lower"].to_list() == [0.0, 1.0, 1.0, 3.0]


# ============================================================================== Study B: analysis
def _long(n_ath=30, attempts=2, scale=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_ath):
        trait = rng.normal(8, 1)
        for k in range(1, attempts + 1):
            val = trait + rng.normal(0, 0.05)
            for sg, prim, fac in (("sg7p2", True, 1.0), ("sg5p2", False, scale)):
                rows.append({"domain": "combine", "traj_id": f"E{k}|{a}|1", "nfl_id": a, "cohort": 2024,
                             "stratum": "FORTY_YARD_DASH", "drill": "FORTY_YARD_DASH", "position": "WR",
                             "attempt_order": k, "smoothing": sg, "is_primary": prim, "feature": "peak_speed",
                             "value": val * fac + (rng.normal(0, noise) if not prim else 0.0)})
    return pl.DataFrame(rows, schema=b.FEATURES_DS.columns)


CFG_B = SmoothingStudyConfig(n_boot=30, min_athletes=5, min_trajectories=5)


def test_agreement_table_levels_and_ci():
    t = b.agreement_table(_long(scale=1.05), CFG_B, seed=1)
    assert set(t["level"].to_list()) == {"trajectory", "athlete"}
    ath = t.filter(pl.col("level") == "athlete").row(0, named=True)
    assert ath["n_units"] == 30 and ath["spearman"] == pytest.approx(1.0)
    assert ath["median_rel_diff"] == pytest.approx(0.05) and ath["spearman_lower"] is not None
    assert ath["top_overlap_lower"] is not None
    trj = t.filter(pl.col("level") == "trajectory").row(0, named=True)
    assert trj["n_units"] == 60 and trj["n_athletes"] == 30 and trj["top_overlap_lower"] is None


def test_agreement_table_insufficient_athletes():
    t = b.agreement_table(_long(n_ath=3), CFG_B, seed=1)
    assert set(t["result_status"].to_list()) == {"insufficient_data"}


def test_variance_table_and_repeatability():
    long = _long(noise=0.01)
    var = b.variance_table(long)
    assert var.filter(pl.col("level") == "athlete")["share_setting"][0] < 0.01
    rep = b.repeatability_table(long, CFG_B, seed=2)
    assert set(rep["smoothing"].to_list()) == {"sg7p2", "sg5p2"}
    assert rep["icc1"].min() > 0.9                  # trait SD 1 vs attempt noise SD 0.05


def test_repeatability_caps_attempts():
    long = _long(attempts=6)
    rep = b.repeatability_table(long, SmoothingStudyConfig(n_boot=10, min_athletes=5, max_attempts=4), seed=0)
    assert rep["n_attempts"].max() == 30 * 4


def test_paired_uses_unit_identity():
    pr = b.paired(_long(n_ath=4))
    assert pr.height == 8 and (pr["traj_id"].n_unique() == 8)


# ============================================================================== Study C1
def _expo():
    return pl.DataFrame({"domain": ["combine"] * 2, "traj_id": ["t1", "t2"], "nfl_id": [1, 2],
                         "drill": ["SHORT_SHUTTLE"] * 2, "smoothing": ["sg7p2"] * 2, "exposure_s": [5.0, 5.0]})


def _c1_events():
    return pl.DataFrame({"domain": ["combine"] * 3, "traj_id": ["t1", "t1", "t1"], "nfl_id": [1, 1, 1],
                         "stratum": ["SHORT_SHUTTLE"] * 3, "drill": ["SHORT_SHUTTLE"] * 3, "smoothing": ["sg7p2"] * 3,
                         "rule": ["default", "default", "strict"], "event_type": ["reversal", "braking", "reversal"],
                         "seg_idx": [0, 0, 0], "event_idx": [0, 0, 0], "t_start_s": [1.0, 0.5, 1.1],
                         "t_end_s": [1.5, 0.9, 1.4], "duration_s": [0.5, 0.4, 0.3], "peak_abs_a_tan": [None] * 3,
                         "peak_abs_yaw_rate": [None] * 3, "heading_change_deg": [None] * 3,
                         "touches_segment_edge": [False, True, False]}, schema=b.EVENTS_DS.columns)


def test_unit_counts_fills_zero_for_event_free_trajectories():
    uc = c1.unit_counts(_c1_events(), _expo())
    assert uc.height == 2 * 2 * 4                      # trajectories × rules present × event types
    t2 = uc.filter(pl.col("traj_id") == "t2")
    assert t2["n_events"].sum() == 0
    assert uc.filter((pl.col("traj_id") == "t1") & (pl.col("rule") == "default"))["n_events"].sum() == 2


def test_rate_table_is_exposure_normalised():
    ev = _c1_events()
    r = c1.rate_table(c1.unit_counts(ev, _expo()), ev, EventStudyConfig(min_athletes=1, n_boot=20), seed=0)
    row = r.filter((pl.col("drill") == "SHORT_SHUTTLE") & (pl.col("rule") == "default")
                   & (pl.col("event_type") == "reversal")).row(0, named=True)
    assert row["rate_per_s"] == pytest.approx(1 / 10.0)    # one event over 2 × 5 s
    assert row["events_per_trajectory"] == pytest.approx(0.5)
    assert set(r["drill"].to_list()) == {"SHORT_SHUTTLE", c1.ALL}


def test_match_configs_onset_shift():
    ev = _c1_events()
    units = _expo().select("traj_id", "nfl_id", "domain", "drill")
    m = c1.match_configs(ev.filter(pl.col("rule") == "default"), ev.filter(pl.col("rule") == "strict"), units, 0.3)
    rev = m.filter(pl.col("event_type") == "reversal").row(0, named=True)
    assert rev["n_ref"] == 1 and rev["n_alt"] == 1 and rev["n_matched"] == 1
    assert rev["onsets"][0] == pytest.approx(0.1)
    brk = m.filter(pl.col("event_type") == "braking").row(0, named=True)
    assert brk["n_alt"] == 0 and brk["n_matched"] == 0


def test_comparisons_exclude_reference():
    comps = c1.comparisons(KinematicsConfig())
    labels = [c for c, *_ in comps]
    assert "rule:default" not in labels and "sg:sg7p2" not in labels
    assert {"rule:lenient", "rule:strict", "sg:sg5p2", "sg:sg9p2", "sg:sg11p3"} == set(labels)


# ============================================================================== Study V
def test_grade_directions():
    assert v.grade(0.95, 0.9, 0.8) == "pass" and v.grade(0.85, 0.9, 0.8) == "marginal"
    assert v.grade(0.7, 0.9, 0.8) == "fail" and v.grade(None, 0.9, 0.8) == "missing"
    assert v.grade(0.02, 0.05, 0.15, higher_better=False) == "pass"
    assert v.grade(0.2, 0.05, 0.15, higher_better=False) == "fail"


SEV = dict(ValidationConfig().severity)


def test_decide_rules():
    ok = {c: "pass" for c in ("smooth_rank", "smooth_overlap", "smooth_magnitude")}
    assert v.decide(ok, SEV)[0] == "VALIDATED"
    assert v.decide({**ok, "boundary": "fail"}, SEV)[0] == "REJECTED"
    assert v.decide({**ok, "repeatability": "fail"}, SEV)[0] == "EXPERIMENTAL"
    assert v.decide({**ok, "smooth_magnitude": "fail"}, SEV)[0] == "PROVISIONAL"
    assert v.decide({**ok, "detector_f1": "missing"}, SEV)[0] == "PROVISIONAL"
    assert v.decide({"smooth_overlap": "pass"}, SEV)[0] == "EXPERIMENTAL"   # core evidence absent


def _ev_row(feature, domain="combine", **vals):
    base = {"domain": domain, "stratum": "FORTY_YARD_DASH", "drill": "FORTY_YARD_DASH", "feature": feature,
            "family": b.FEATURES[feature][1], "unit": b.FEATURES[feature][0], "n_athletes": 300, "_tie": False,
            **{f"value_{c}": None for c in v.CRITERIA}}
    base.update({f"value_{k}": x for k, x in vals.items()})
    return base


GOOD = dict(smooth_rank=0.97, smooth_overlap=0.9, smooth_magnitude=0.01, repeatability=0.9, icc_setting_range=0.01)


def test_event_feature_cannot_be_validated_without_labels():
    reg = v.registry(pl.DataFrame([_ev_row("rate_braking_per_s", **GOOD, event_smoothing=0.95)]), ValidationConfig())
    r = reg.row(0, named=True)
    assert r["state_detector_f1"] == "missing" and r["status"] == "PROVISIONAL"


def test_peak_feature_rejected_when_boundary_dominated_and_interior_not():
    rows = [_ev_row("peak_speed", **GOOD, boundary=11.4, vendor=0.01), _ev_row("peak_speed_interior", **GOOD,
                                                                               vendor=0.01)]
    reg = v.registry(pl.DataFrame(rows), ValidationConfig())
    st = dict(zip(reg["feature"].to_list(), reg["status"].to_list()))
    assert st["peak_speed"] == "REJECTED"
    assert st["peak_speed_interior"] == "VALIDATED"


def test_nfl_features_capped_without_repeatability():
    reg = v.registry(pl.DataFrame([_ev_row("p95_speed", domain="nfl_sample", **GOOD, vendor=0.03)]),
                     ValidationConfig())
    assert reg["state_repeatability"][0] == "missing" and reg["status"][0] == "PROVISIONAL"


def test_summary_takes_worst_primary_status():
    bad = GOOD | {"smooth_rank": 0.5}
    rows = [_ev_row("p95_a_tan", **GOOD), {**_ev_row("p95_a_tan", **bad), "drill": "SHORT_SHUTTLE",
                                           "stratum": "SHORT_SHUTTLE"},
            {**_ev_row("p95_a_tan", **bad), "drill": "LINE_DRILL", "stratum": "SKILL_DRILLS_DB"}]
    reg = v.registry(pl.DataFrame(rows), ValidationConfig())
    s = v.summary(reg, ValidationConfig()).row(0, named=True)
    assert s["status"] == "REJECTED" and s["n_drills"] == 3 and "LINE_DRILL" not in s["primary_strata"]


# ============================================================================== Study Q
def _dims_signals():
    dims = pl.DataFrame({
        "domain": ["combine", "combine", "nfl", "nfl"], "traj_id": ["c1", "c2", "n1", "n2"], "nfl_id": [1, 2, 3, 4],
        "cohort": [2024, 2024, 2023, 2023], "stratum": ["SHORT_SHUTTLE", "SHORT_SHUTTLE", "nfl_play", "nfl_play"],
        "position": ["WR", "DB", "WR", "OL"], "quality_tier": ["A", "reject", "A", "A"],
        "flag_structural_integrity": [False, False, True, False], "flag_temporal_continuity": [False] * 4,
        "flag_coordinate_validity": [False] * 4})
    sig = pl.DataFrame({
        "domain": ["combine", "combine", "nfl_sample"], "traj_id": ["c1", "c2", "n1"],
        "stratum": ["SHORT_SHUTTLE", "SHORT_SHUTTLE", "nfl_play"], "vendor_rmse": [0.1, 0.2, 0.1],
        "jerk_rms_interior": [1.0, 1.0, 1.0], "peak_a_tan_in_boundary": [True, False, None],
        "peak_decel_in_boundary": [False, False, False]})
    drills = pl.DataFrame({"traj_id": ["c1", "c2"], "drill": ["SHORT_SHUTTLE", "SHORT_SHUTTLE"]})
    return dims, sig, drills


def test_build_flags_maps_nfl_sample_and_separates_families():
    t = q.build_flags(*_dims_signals(), EligibilityConfig())
    assert set(t["domain"].to_list()) == {"combine", "nfl_sample"}
    assert t.height == 3                                              # n2 has no A3 signal row → excluded
    r = {x["traj_id"]: x for x in t.iter_rows(named=True)}
    assert r["c2"]["structural_flag"] and "tier_reject" in r["c2"]["structural_reasons"]
    assert r["c1"]["boundary_flag"] and not r["c1"]["structural_flag"]
    assert r["n1"]["structural_flag"] and r["n1"]["structural_reasons"] == "structural_integrity"


def test_eligibility_scenarios_monotone():
    t = q.build_flags(*_dims_signals(), EligibilityConfig())
    e = q.eligibility(t, EligibilityConfig(min_attempts_eligible=1))
    c = e.filter(pl.col("domain") == "combine").sort("scenario_order")
    assert c["n_athletes_any"].to_list()[0] == 2
    assert all(np.diff(c["n_trajectories"].to_numpy()) <= 0)


# ============================================================================== Study C2 v2
def _signals_pool():
    rows = []
    for s in ("FORTY_YARD_DASH", "SHORT_SHUTTLE"):
        for k in range(20):
            rows.append({"domain": "combine", "traj_id": f"{s}|{k}|1", "nfl_id": k, "stratum": s, "position": "WR",
                         "cohort": 2024, "duration_s": 5.0, "jerk_rms_interior": float(k), "quality_tier": "A",
                         "n_valid_kin": 50})
    for k in range(20):
        rows.append({"domain": "nfl_sample", "traj_id": f"9|{k}|{100 + k}", "nfl_id": 100 + k, "stratum": "nfl_play",
                     "position": "WR" if k % 2 else "DB", "cohort": 2023, "duration_s": 8.0,
                     "jerk_rms_interior": float(k), "quality_tier": "A", "n_valid_kin": 80})
    return pl.DataFrame(rows)


def test_select_tasks_deterministic_split_disjoint_and_priority():
    cfg = AnnotationDesignConfig(combine_strata=("FORTY_YARD_DASH", "SHORT_SHUTTLE"), nfl_positions=("WR", "DB"),
                                 per_stratum_random=3, per_stratum_difficult=1)
    t1, t2 = ad.select_tasks(_signals_pool(), cfg, 7), ad.select_tasks(_signals_pool(), cfg, 7)
    assert t1.equals(t2)
    assert t1.height == 4 * (3 + 1)
    assert t1.group_by("nfl_id").agg(pl.col("split").n_unique().alias("k"))["k"].max() == 1
    assert set(t1["stratum"].to_list()) == {"FORTY_YARD_DASH", "SHORT_SHUTTLE", "nfl_play:WR", "nfl_play:DB"}
    dif = t1.filter((pl.col("selection") == "difficult") & (pl.col("stratum") == "FORTY_YARD_DASH"))
    rnd = t1.filter((pl.col("selection") == "random") & (pl.col("stratum") == "FORTY_YARD_DASH"))
    assert dif["traj_id"][0] not in rnd["traj_id"].to_list()
    assert t1["priority"].to_list() == sorted(t1["priority"].to_list())


def test_assign_split_fraction_roughly_respected():
    s = ad.assign_split(list(range(5000)), 0.3, 11)
    share = sum(x == "dev" for x in s.values()) / len(s)
    assert 0.27 < share < 0.33


# ============================================================================== infrastructure
def test_store_phase_namespace(tmp_path: Path):
    cfg = SimpleNamespace(paths=SimpleNamespace(tables=tmp_path / "t", figures=tmp_path / "f", gold=tmp_path / "g"))
    s4 = StudyStore(cfg, phase="phase_04")                  # type: ignore[arg-type]
    s3s = StudyStore(cfg, sampled=True)                      # type: ignore[arg-type]
    assert s4.tables_root == tmp_path / "t" / "phase_04"
    assert s3s.tables_root == tmp_path / "t" / "phase_03" / "_sample"
    assert s4.figure_dir("V").name == "V_feature_validation"


def test_threshold_table_documents_every_field():
    t = threshold_table(Phase04Config())
    assert t.height > 30 and set(t["provenance"].unique().to_list()) <= {"established", "proposed", "execution"}


def test_contracts_have_unique_names():
    names = [c.name for c in (*q.CONTRACTS, *b.CONTRACTS, *c1.CONTRACTS, *v.CONTRACTS, ad.TASKS, ad.MATCHES,
                              ad.PERFORMANCE, ad.TAXONOMY, ad.COVERAGE)]
    assert len(names) == len(set(names))
