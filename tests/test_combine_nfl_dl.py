"""Static tests for the Combine → NFL defensive-line get-off experiment (synthetic data only — no warehouse access).

Authored 2026-10-11, NOT executed by Claude. Run::

    python -m pytest tests/test_combine_nfl_dl.py -q
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from nflbdb.research.combine_nfl_dl import cohort as co
from nflbdb.research.combine_nfl_dl import endpoint as ep
from nflbdb.research.combine_nfl_dl import evaluate as ev
from nflbdb.research.combine_nfl_dl import models as md
from nflbdb.research.combine_nfl_dl import report as rp
from nflbdb.research.combine_nfl_dl import targets as tg
from nflbdb.research.combine_nfl_dl.config import (COHORT, PRIMARY_VARIANT, REFERENCE_MODEL, TRACKING, GetOffConfig,
                                                    model_specs, primary_spec, sensitivity_specs)
from nflbdb.research.combine_nfl_dl.figures import FIGURES
from nflbdb.research.combine_nfl_dl.pipeline import ALL_CONTRACTS, GATED, STAGES, make_store, run_stage
from nflbdb.research.contracts import conform

CFG = GetOffConfig(n_boot=100, n_boot_sensitivity=50, cv_repeats=2, n_game_splits=10, min_athletes_model=20,
                   primary_min_obs=30)
ROOT = Path(__file__).resolve().parents[1]
CONFIRMED = {"status": "confirmed", "source": "data dictionary", "timing_origin": "snap", "timing_end": "x",
             "lower_is_better": True}


# ------------------------------------------------------------------------------ synthetic data
def _ability(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, n)


def _base(n_ath: int = 60, n_games: int = 6, per_game: int = 10, effect: float = 0.08, seed: int = 0) -> pl.DataFrame:
    """Tracked player-plays as ``assemble_base`` returns them. Athlete i: ability a_i lowers his get-off value;
    even athletes play EDGE (−0.1 s), odd athletes INTERIOR_LINE."""
    rng = np.random.default_rng(seed + 1)
    a = _ability(n_ath, seed)
    rows = []
    for i in range(n_ath):
        align = "EDGE" if i % 2 == 0 else "INTERIOR_LINE"
        for g in range(n_games):
            for p in range(per_game):
                v = 1.0 - effect * a[i] - (0.1 if align == "EDGE" else 0.0) + rng.normal(0, 0.15)
                rows.append({"game_id": 2024090000 + g * 100 + i % 3, "play_id": p + 1, "nfl_id": 1000 + i, "season": 2024,
                             "season_type": "REG", "position_group": "DL", "cohort": 2023 + i % 3,
                             "lined_up_position": align, "player_get_off": float(np.clip(v, 0.05, 2.0)),
                             "play_nullified_by_penalty": "N", "blitzing": "FALSE", "pass_result": "C", "down": 1.0,
                             "sack": 0.0, "time_to_pressure": None, "unblocked_pressure": "FALSE",
                             "coverage_assignment": None, "offense_formation": "SHOTGUN"})
    return pl.DataFrame(rows, schema_overrides={"time_to_pressure": pl.Float64, "coverage_assignment": pl.String})


def _plays(**kw) -> pl.DataFrame:
    return ep.valid_plays(ep.dl_population(_base(**kw), CFG), CFG)


def _attempts(n_ath: int = 60, seed: int = 0) -> pl.DataFrame:
    """Two forty attempts per athlete; ``accel_v4_7`` carries the ability, the other features do not."""
    rng = np.random.default_rng(seed + 2)
    a = _ability(n_ath, seed)
    rows = []
    for i in range(n_ath):
        for order in (1, 2):
            rows.append({"nfl_id": 1000 + i, "attempt_order": order, "fit_ok": not (i == 0 and order == 2), "S0_all": True,
                         "S1_structural": True, "S2_signal": order == 1 or i % 4 != 0, "S3_no_boundary": order == 1,
                         "p95_speed": 9.5 + rng.normal(0, 0.3) + 0.1 * order, "accel_v4_7": 4.0 + 0.5 * a[i] + rng.normal(0, 0.05),
                         "model_tau_s": None if (i == 0 and order == 2) else 0.9 + rng.normal(0, 0.05)})
    return pl.DataFrame(rows, schema_overrides={"model_tau_s": pl.Float64})


def _mart(n_ath: int = 60, seed: int = 0, extra: int = 5) -> pl.DataFrame:
    """``extra`` Combine athletes never play in the NFL; they must stay in the research base."""
    rng = np.random.default_rng(seed + 3)
    a = np.r_[_ability(n_ath, seed), np.zeros(extra)]
    rows = [{"nfl_id": 1000 + i, "display_name": f"Athlete {i}", "position_group": "DL", "cohort": 2023 + i % 3,
             "has_combine_results": True, "has_nfl_tracking": i < n_ath, "cmb_combine_weight": 280.0 + rng.normal(0, 15),
             "cmb_combine_height": 76.0 + rng.normal(0, 1), "cmb_forty": 4.9 - 0.01 * a[i] + rng.normal(0, 0.08),
             "cmb_ten_yd_split": 1.7 + rng.normal(0, 0.03), "cmb_vertical": None if i % 7 == 0 else 32.0 + rng.normal(0, 2),
             "cmb_broad_jump": 115.0 + rng.normal(0, 4)} for i in range(n_ath + extra)]
    rows.append({**rows[0], "nfl_id": 9999, "display_name": "A Receiver", "position_group": "WR"})
    return pl.DataFrame(rows, schema_overrides={"cmb_vertical": pl.Float64})


def _inputs(seed: int = 0):
    plays = _plays(seed=seed)
    targets, cells = tg.athlete_targets(plays, CFG), tg.context_cells(plays, CFG)
    variants = co.combine_variants(_attempts(seed=seed), CFG)
    base = co.research_base(_mart(seed=seed), variants, targets, CFG)
    return base, variants, targets, cells


# ------------------------------------------------------------------------------ ridge and preprocessing
def test_ridge_loo_error_equals_explicit_leave_one_out():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(25, 3)), rng.normal(size=25)
    lambdas = np.array([0.5, 5.0])
    fit = md.ridge_loo(X, y, lambdas)
    for k, lam in enumerate(lambdas):
        errs = []
        for i in range(25):
            m = np.arange(25) != i
            b0, beta = md.ridge_fixed(X[m], y[m], lam)
            errs.append(y[i] - b0 - X[i] @ beta)
        assert fit["loo_mse"][k] == pytest.approx(np.mean(np.square(errs)), rel=1e-6)


def test_ridge_loo_recovers_coefficients_and_handles_no_predictor():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 2))
    y = 3.0 + X @ np.array([1.5, -0.5]) + rng.normal(0, 0.01, 200)
    fit = md.ridge_loo(X, y, md.lambda_grid((-4.0, 2.0, 13)))
    assert fit["beta"] == pytest.approx([1.5, -0.5], abs=0.02) and fit["intercept"] == pytest.approx(3.0, abs=0.02)
    empty = md.ridge_loo(np.empty((10, 0)), y[:10], np.array([1.0]))
    assert empty["intercept"] == pytest.approx(y[:10].mean()) and empty["beta"].size == 0


def test_preprocessing_uses_training_rows_only():
    data = {"x": np.array([1.0, 2.0, 3.0, np.nan, 1000.0]), "c": np.array(["a", "b", "a", "b", "z"])}
    train = np.array([True, True, True, True, False])
    prep = md.fit_prep(data, ("x",), ("c",), train)
    assert prep.median[0] == pytest.approx(2.0) and prep.mean[0] == pytest.approx(2.0)   # the outlier is a test row
    assert [lev.tolist() for lev in prep.levels] == [["a", "b"]]
    Xte = md.transform(data, prep, ~train)
    assert Xte.shape == (1, 2) and Xte[0, 1] == 0.0                                       # unseen level → reference


def test_huber_ridge_resists_an_outlier():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 1))
    X[0, 0] = 2.0                                               # the outlier has leverage
    y = 2.0 * X[:, 0] + rng.normal(0, 0.1, 80)
    y[0] += 50.0
    _, plain = md.ridge_fixed(X, y, 0.01)
    _, robust = md.huber_ridge(X, y, 0.01)
    assert abs(robust[0] - 2.0) < abs(plain[0] - 2.0) + 1e-9 and robust[0] == pytest.approx(2.0, abs=0.1)


# ------------------------------------------------------------------------------ target and folds
def test_context_adjusted_target_uses_training_athletes_only():
    t = md.Target(S=np.array([[2.0, 0.0], [0.0, 3.0], [4.0, 4.0]]), K=np.array([[2.0, 0.0], [0.0, 3.0], [2.0, 2.0]]))
    y = t.y(np.array([True, True, False]))
    assert y == pytest.approx([1.0, 1.0, 2.0])
    assert t.y(np.array([True, False, False]))[1] == pytest.approx(1.0)       # unseen cell → grand training mean
    moved = md.Target(S=t.S + np.array([[0, 0], [0, 0], [100.0, 0]]), K=t.K)  # change the TEST athlete's plays only
    assert moved.y(np.array([True, True, False]))[:2] == pytest.approx(y[:2])
    assert md.Target(fixed=np.array([1.0, 2.0])).y(np.array([True, False])) == pytest.approx([1.0, 2.0])


def test_folds_partition_athletes_and_balance_cohorts():
    strata = np.array(["2023"] * 20 + ["2024"] * 20 + ["2025"] * 10)
    f = md.make_folds(strata, 5, 3, seed=7)
    assert f.shape == (3, 50) and (md.make_folds(strata, 5, 3, seed=7) == f).all()
    for r in range(3):
        assert sorted(np.bincount(f[r]).tolist()) == [10] * 5
        for s in ("2023", "2024", "2025"):
            assert np.bincount(f[r][strata == s], minlength=5).min() >= 2
    assert md.group_folds(strata).shape == (1, 50) and np.unique(md.group_folds(strata)).size == 3


# ------------------------------------------------------------------------------ cross-validation and evaluation
def _cv(signal: float, seed: int = 0, n: int = 80):
    rng = np.random.default_rng(seed)
    z, noise = rng.normal(size=n), rng.normal(size=n)
    data = {"z": z, "noise": noise, COHORT: np.array(["2023", "2024"] * (n // 2))}
    y = 1.0 + signal * z + rng.normal(0, 0.1, n)
    models = {"MEAN": ((), ()), "M1": (("noise",), (COHORT,)), "M2": (("noise", "z"), (COHORT,))}
    folds = md.make_folds(data[COHORT], 5, 4, seed)
    return md.cross_validate(data, md.Target(fixed=y), models, folds, md.lambda_grid((-2.0, 4.0, 13)))


def test_noise_predictors_show_no_out_of_fold_skill():
    rows, _ = md.evaluate(_cv(signal=0.0), "M1", 100, 1)
    r2 = {r["model"]: r["estimate"] for r in rows if r["metric"] == "r2_oos"}
    assert r2["MEAN"] == pytest.approx(0.0, abs=1e-9)
    assert r2["M1"] < 0.05 and r2["M2"] < 0.05                 # an optimistic R² here would mean leakage


def test_real_signal_is_detected_as_a_paired_gain():
    rows, diffs = md.evaluate(_cv(signal=0.2), "M1", 200, 1)
    d = next(x for x in diffs if x["model"] == "M2" and x["metric"] == "rmse")
    assert d["estimate"] < 0 and d["ci_upper"] < 0 and d["share_boot_improved"] > 0.95
    assert next(r for r in rows if r["model"] == "M2" and r["metric"] == "r2_oos")["estimate"] > 0.5


def test_identical_models_have_zero_difference():
    cv = _cv(signal=0.1)
    cv["pred"]["COPY"] = cv["pred"]["M1"].copy()
    _, diffs = md.evaluate(cv, "M1", 50, 1)
    d = next(x for x in diffs if x["model"] == "COPY" and x["metric"] == "rmse")
    assert d["estimate"] == 0.0 and d["ci_lower"] == 0.0 and d["ci_upper"] == 0.0


def test_every_athlete_is_predicted_once_per_repeat_by_a_model_that_did_not_see_him():
    cv = _cv(signal=0.2)
    assert np.isfinite(cv["y"]).all() and all(np.isfinite(p).all() for p in cv["pred"].values())
    data = {"z": np.arange(10.0), COHORT: np.array(["a"] * 10)}
    y = np.arange(10.0)
    train = np.arange(10) < 8
    pred, _ = md.fit_predict(data, ("z",), (), y, train, ~train, np.array([1e-6]))
    y2 = y.copy()
    y2[~train] = -999.0                                         # the test athletes' outcome must not matter
    pred2, _ = md.fit_predict(data, ("z",), (), y2, train, ~train, np.array([1e-6]))
    assert pred == pytest.approx(pred2) and pred == pytest.approx([8.0, 9.0], abs=1e-3)


# ------------------------------------------------------------------------------ endpoint gate and valid plays
def test_definition_gate():
    shipped = ep.load_definition(SimpleNamespace(paths=SimpleNamespace(project_root=ROOT)))
    assert shipped.get("status") == "unresolved" and not ep.definition_status(shipped)[0]
    assert "BLOCKER" in ep.blocker_text(shipped)
    assert ep.definition_status(CONFIRMED) == (True, []) and ep.blocker_text(CONFIRMED) == ""
    assert not ep.definition_status({**CONFIRMED, "timing_end": None})[0]         # confirmed needs every field
    assert not ep.definition_status({**CONFIRMED, "status": "unresolved"})[0]


@pytest.mark.parametrize("stage", GATED)
def test_gated_stages_do_not_run_on_an_unresolved_definition(stage):
    inp = SimpleNamespace(definition=lambda: {"status": "unresolved"})
    assert run_stage(stage, inp, store=None) == "blocked"       # returns before touching the store or the data


def test_valid_plays_filter_and_order():
    b = _base(n_ath=2, n_games=2, per_game=3)
    extra = b.head(4).with_columns(pl.Series("play_id", [91, 92, 93, 94]),
                                   pl.Series("lined_up_position", ["SLOT_CB", "EDGE", "EDGE", "EDGE"]),
                                   pl.Series("play_nullified_by_penalty", ["N", "Y", "N", "N"]),
                                   pl.Series("player_get_off", [1.0, 1.0, None, 1.0]),
                                   pl.Series("season_type", ["REG", "REG", "REG", "PRE"]))
    v = ep.valid_plays(ep.dl_population(pl.concat([b, extra]), CFG), CFG)
    assert v.height == b.height                                 # other alignment, nullified, null and PRE all dropped
    conform(v, ep.PLAYS)
    one = v.filter(pl.col("nfl_id") == 1000)
    assert one["obs_idx"].to_list() == list(range(1, one.height + 1))
    assert one.sort("obs_idx")["game_id"].is_sorted() and one["game_idx"].max() == 2
    assert v["at_max"].sum() >= 1 and v.filter(pl.col("at_max"))["value"].min() >= v["value"].max() - CFG.boundary_tol_s


def test_endpoint_audit_reports_the_gate_and_conforms():
    pop = ep.dl_population(_base(n_ath=20), CFG)
    plays = ep.valid_plays(pop, CFG)
    a = conform(ep.endpoint_audit(pop, plays, {"status": "unresolved"}, {"evidence": "inferred"}, CFG), ep.AUDIT)
    row = a.filter((pl.col("section") == "definition") & (pl.col("metric") == "vendor_definition_confirmed"))
    assert row["value"][0] == 0.0 and "inferred" in row["note"][0]
    assert {"availability", "distribution", "boundary", "alignment", "validity"} <= set(a["section"].unique().to_list())
    conform(ep.endpoint_histogram(plays), ep.HIST)


# ------------------------------------------------------------------------------ targets and reliability
def test_targets_windows_and_fold_aware_target_agree_with_the_descriptive_one():
    plays = _plays(n_ath=30)
    t, c = conform(tg.athlete_targets(plays, CFG), tg.TARGET_TABLE), conform(tg.context_cells(plays, CFG), tg.CELLS)
    all_ = t.filter(pl.col("window") == "all").sort("nfl_id")
    assert all_["n_obs"].unique().to_list() == [60]
    assert t.filter(pl.col("window") == "first_20")["n_obs"].unique().to_list() == [20]
    ids = all_["nfl_id"].to_list()
    assert tg.build_target(t, c, ids, "adj_mean", "all").y(None) == pytest.approx(all_["adj_mean"].to_numpy(), abs=1e-9)
    assert tg.build_target(t, c, ids, "median", "all").y(None) == pytest.approx(all_["median"].to_numpy())
    edge = all_.filter(pl.col("alignment_group") == "EDGE")
    raw_gap = all_.filter(pl.col("alignment_group") != "EDGE")["mean"].mean() - edge["mean"].mean()
    adj_gap = all_.filter(pl.col("alignment_group") != "EDGE")["adj_mean"].mean() - edge["adj_mean"].mean()
    assert raw_gap > 0.02 and abs(adj_gap) < 1e-9             # the adjustment removes the alignment offset


def test_outcome_reliability_separates_trait_from_noise():
    cfg = GetOffConfig(n_boot=50, n_game_splits=10, min_obs_grid=(20,), first_n_grid=(20,))

    def rel(effect: float) -> float:
        plays = ep.valid_plays(ep.dl_population(_base(n_ath=40, effect=effect), cfg), cfg)
        r = conform(tg.reliability_table(plays, cfg), tg.RELIABILITY)
        return r.filter((pl.col("target") == "adj_mean") & (pl.col("window") == "all")
                        & (pl.col("method") == "game_random_halves_sb"))["estimate"][0]

    assert rel(0.15) > 0.8 and rel(0.0) < 0.5


# ------------------------------------------------------------------------------ cohort
def test_combine_variants():
    v = conform(co.combine_variants(_attempts(n_ath=8), CFG), co.VARIANTS)
    att = _attempts(n_ath=8).filter(pl.col("nfl_id") == 1001).sort("attempt_order")

    def val(variant: str, col: str, nfl_id: int = 1001):
        return v.filter((pl.col("variant") == variant) & (pl.col("nfl_id") == nfl_id))[col][0]

    assert val(PRIMARY_VARIANT, "p95_speed") == pytest.approx(att["p95_speed"].mean())
    assert val("first|S0_all|all", "p95_speed") == pytest.approx(att["p95_speed"][0])
    assert val("fastest|S0_all|all", "p95_speed") == pytest.approx(att["p95_speed"].max())
    assert val("mean|S3_no_boundary|all", "n_attempts_used") == 1
    assert val("mean|S0_all|fit_ok", "n_attempts_used", 1000) == 1 and val(PRIMARY_VARIANT, "n_attempts_used", 1000) == 2


def test_research_base_keeps_combine_athletes_without_nfl_play():
    base, _, _, _ = _inputs()
    conform(base, co.BASE)
    assert base.height == 65 and base.filter(pl.col("nfl_id") == 9999).height == 0     # DL only
    never = base.filter(pl.col("nfl_id") >= 1060)
    assert never.height == 5 and never["n_obs"].to_list() == [0] * 5 and not never["eligible_primary"].any()
    assert base.filter(pl.col("eligible_primary")).height == 60
    funnel = conform(co.cohort_funnel(_mart(), base, CFG), co.FUNNEL)
    n = dict(funnel.filter(pl.col("cohort") == "ALL").select("stage", "n_athletes").iter_rows())
    assert n["01_combine_dl"] == 65 and n["02_nfl_tracking_any"] == 60
    assert next(v for k, v in n.items() if k.endswith("primary_population")) == 60
    per = funnel.filter(pl.col("cohort") != "ALL").group_by("stage").agg(pl.col("n_athletes").sum())
    assert dict(per.iter_rows()) == n                                                   # cohorts add up to ALL
    conform(co.cohort_missingness(base, CFG), co.MISSING)
    corr = conform(co.predictor_correlations(base), co.CORRELATION)
    assert corr.filter(pl.col("other") == "M1_all").height == len(TRACKING)


# ------------------------------------------------------------------------------ model stage end to end
def test_predictors_are_pre_nfl_and_specs_are_consistent():
    specs = model_specs()
    allowed = {COHORT, *co.OFFICIAL_COLUMNS, *TRACKING}
    for s in specs.values():
        assert set(s.columns) <= allowed and not any(c.startswith("nfl_") or "alignment" in c for c in s.columns)
    assert set(specs["M1"].numeric) < set(specs["M2b"].numeric) and specs["M2bc"].exploratory
    names = [s.name for s in sensitivity_specs(CFG)]
    assert len(names) == len(set(names)) and "primary" not in names
    assert all(m in specs for s in sensitivity_specs(CFG) for m in s.models) and REFERENCE_MODEL in primary_spec(CFG).models


def test_model_stage_outputs_conform_and_models_share_athletes_and_folds():
    out = ev.compute_models(*_inputs(), CFG)
    for c in STAGES["MOD"][0]:
        conform(out[c.name], c)
    comp, oof, folds = out[ev.COMPARISON.name], out[ev.OOF.name], out[ev.FOLDS.name]
    assert comp["n_athletes"].unique().to_list() == [60]
    assert folds.group_by("repeat").agg(pl.col("nfl_id").n_unique())["nfl_id"].unique().to_list() == [60]
    per_model = oof.group_by("model").agg(pl.col("nfl_id").n_unique().alias("n"), pl.len().alias("rows"))
    assert per_model["n"].unique().to_list() == [60] and per_model["rows"].unique().to_list() == [60 * CFG.cv_repeats]
    pivot = oof.pivot(on="model", index=["repeat", "nfl_id"], values="fold")
    assert all((pivot[m] == pivot["M1"]).all() for m in primary_spec(CFG).models)       # identical folds
    y = oof.pivot(on="model", index=["repeat", "nfl_id"], values="y_obs")
    assert (y["M2b"] - y["M1"]).abs().max() < 1e-12                                     # identical targets
    d = out[ev.BOOT.name].filter((pl.col("model") == "M2b") & (pl.col("metric") == "rmse"))
    assert d["estimate"][0] < 0 and d["ci_upper"][0] < 0       # the planted acceleration signal is found …
    a = out[ev.BOOT.name].filter((pl.col("model") == "M2a") & (pl.col("metric") == "rmse"))
    assert a["ci_upper"][0] > 0                                # … and the noise feature is not credited
    assert comp.filter(pl.col("model") == "M2bc")["exploratory"].all()
    assert not comp.filter(pl.col("model") == "M2b")["exploratory"].any()


def test_matched_pairs_respect_calipers_and_use_each_athlete_once():
    frame = ev.analysis_frame(*_inputs(), primary_spec(CFG), model_specs())
    p = conform(ev.matched_pairs(frame, CFG), ev.MATCHED)
    assert p.height >= 5
    ids = p["nfl_id_a"].to_list() + p["nfl_id_b"].to_list()
    assert len(ids) == len(set(ids))
    assert ((p["forty_a"] - p["forty_b"]).abs() <= CFG.match_forty_s + 1e-12).all()
    assert ((p["ten_a"] - p["ten_b"]).abs() <= CFG.match_ten_s + 1e-12).all() and (p["d_feature"] >= 0).all()
    assert set(ev.matched_summary(p)) >= {"n_pairs", "spearman", "share_d_target_negative"}


def test_sensitivity_table_conforms_and_flags_what_cannot_be_estimated():
    s = conform(ev.compute_sensitivity(*_inputs(), CFG), ev.SENSITIVITY)
    names = set(s["analysis"].to_list())
    assert {"primary", "leave_one_cohort_out", "exclude_cohort:2024", "robust_huber", "target:median", "first_50",
            "combine:first|S0_all|all", "own_sample:M2b", "scored_on:alignment=EDGE", "scored_on:cohort=2023"} <= names
    skipped = s.filter(pl.col("model") == "-")
    assert "min_obs:100" in skipped["analysis"].to_list() and skipped["note"].str.contains("not estimable").all()
    loco = s.filter((pl.col("analysis") == "leave_one_cohort_out") & (pl.col("model") == "M2b"))
    assert loco["n_athletes"][0] == 60 and loco["rmse"][0] is not None


# ------------------------------------------------------------------------------ report, figures, registry
def test_template_tokens_are_known_and_an_empty_run_reports_pending_and_the_blocker(tmp_path):
    template = (ROOT / "reports" / rp.TEMPLATE).read_text(encoding="utf-8")
    tokens = set(re.findall(r"\{\{(\w+)\}\}", template))
    assert tokens == set(rp.SECTIONS)
    (tmp_path / "configs").mkdir()
    paths = SimpleNamespace(project_root=tmp_path, tables=tmp_path / "t", figures=tmp_path / "f", gold=tmp_path / "g",
                            rel=str)
    text = rp.fill(template, rp.Tables(make_store(SimpleNamespace(paths=paths))))
    assert "{{" not in text and "Unknown report token" not in text
    assert "BLOCKER" in text and text.count("_Pending:") >= 10
    assert not re.search(r"\d\.\d{3}", text.split("## 7.")[1].split("## 10.")[0])       # no number without a run


def test_stage_registry_and_figures():
    names = [c.name for c in ALL_CONTRACTS]
    assert len(names) == len(set(names)) and len(names) == 18
    assert all(d in STAGES for _, deps in STAGES.values() for d in deps)
    assert len(FIGURES) == 8 and all(n in names for _, needs, _ in FIGURES for n in needs)
    assert all(name.startswith(f"CN_{i + 1:02d}_") for i, (name, _, _) in enumerate(FIGURES))
