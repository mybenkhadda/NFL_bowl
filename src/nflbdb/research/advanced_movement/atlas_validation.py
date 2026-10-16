"""Synthetic validation of Milestone 04 (feature atlas, robust TCN validation, shortlist) — callable from a notebook
cell, no subprocess, no warehouse, no PyTorch.

    from nflbdb.research.advanced_movement import atlas_validation
    report = atlas_validation.run_all(atlas_validation.ATLAS_CHECKS)     # notebook 04A
    report = atlas_validation.run_all(atlas_validation.ROBUST_CHECKS)    # notebook 04B
    atlas_validation.assert_all_passed(report)
    atlas_validation.run_all(atlas_validation.PROJECTION_CHECKS)         # the parts of `embedding_projection`, one row each

Data: the analytic sprint fixture of ``validation`` / ``fpca_validation`` and small simulated cohorts with a planted
answer. The checks test the WIRING on known answers (which athletes an estimate uses, what a statistic returns when
the truth is known, where a file is written); they say nothing about the real measurements.
``tests/test_advanced_movement_atlas.py`` calls the same functions.
"""

from __future__ import annotations

import inspect
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ..contracts import ContractError, TableContract, conform
from ..sprint_rescue.config import feature_specs
from ..sprint_rescue.stats import partial_corr
from . import atlas as at
from . import fpca_eval as fe
from . import fpca_validation as fv
from . import repr_eval as rv
from . import robust as rb
from . import validation as base
from .artifacts import AdvStore, AtlasStore
from .config import NotebookControls
from .fpca_eval import FORTY, HEIGHT, TEN, WEIGHT

CFG = at.AtlasConfig(n_boot=60, n_boot_r2=30, min_repeat_athletes=10, min_athletes=5)
ECFG = rv.EvalConfig(n_boot=60, n_boot_probe=40, min_athletes=5, probe_folds=3)
MC = rb.MatchConfig()
FORBIDDEN = ("nfl_reg_", "nfl_any_", "getoff", "get_off", "play_participation", "player_play", "combine_nfl_dl",
             "nfl_player_play_windows", "athlete_targets")
SCANNED = ("atlas.py", "atlas_figures.py", "robust.py", "robust_figures.py", "robust_train.py")
_raises = base._raises


# ============================================================================== fixtures
def synthetic_registry() -> pl.DataFrame:
    rows = [("combine", "FORTY_YARD_DASH", "FORTY_YARD_DASH", "p95_speed", "speed_quantile", "yd/s", "VALIDATED", "all pass", 100, 0.90),
            ("combine", "FORTY_YARD_DASH", "FORTY_YARD_DASH", "n_turn", "event_count", "events", "REJECTED", "critical fail", 100, -0.02),
            ("combine", "SKILL_DRILLS_LB", "PASS_RUSH_DRILL", "p95_speed", "speed_quantile", "yd/s", "PROVISIONAL", "open", 80, 0.55),
            ("combine", "SKILL_DRILLS_DL", "PASS_RUSH_DRILL", "p95_speed", "speed_quantile", "yd/s", "EXPERIMENTAL", "major fail", 90, 0.30),
            ("nfl_sample", "nfl_play", "nfl_play", "p95_speed", "speed_quantile", "yd/s", "PROVISIONAL", "open", 50, None)]
    return pl.DataFrame(rows, orient="row", schema={"domain": pl.String, "stratum": pl.String, "drill": pl.String, "feature": pl.String,
                                                     "family": pl.String, "unit": pl.String, "status": pl.String, "status_reason": pl.String,
                                                     "n_athletes": pl.Int64, "value_repeatability": pl.Float64})


def synthetic_attempts(b: dict, seed: int = 0) -> pl.DataFrame:
    """A sprint-rescue-like attempt table for the fixture: ``p95_speed`` = the athlete's true top speed + small noise
    (repeatable), ``peak_a_tan`` = pure attempt noise (not repeatable), every other column = a mild athlete trait."""
    rng = np.random.default_rng(seed)
    m = b["manifest"].filter(pl.col("eligible_primary")).select("event_id", "nfl_id", "attempt").sort("nfl_id", "attempt")
    nfl = m["nfl_id"].to_numpy()
    cols = [s.name for s in feature_specs()] + [e[0] for e in at.EXTRA_ATTEMPT]
    data = {}
    for j, c in enumerate(cols):
        trait = np.random.default_rng(1000 + j).normal(size=5000)[nfl - 1000]
        data[c] = 5.0 + trait + 0.5 * rng.normal(size=nfl.size)
    data["p95_speed"] = np.array([base.athlete_params(int(i) - 1000)[0] for i in nfl]) + 0.02 * rng.normal(size=nfl.size)
    data["peak_a_tan"] = 5.0 + rng.normal(size=nfl.size)
    return m.with_columns(*[pl.Series(c, v) for c, v in data.items()], pl.Series("fit_ok", (np.arange(nfl.size) % 7 != 0)),
                          pl.lit(False).alias("band_start_censored"))


@lru_cache(maxsize=2)
def engine(n_athletes: int = 60, seed: int = 0) -> dict:
    """The whole atlas computed on the sprint fixture (cached: several checks read it)."""
    b = fv.synthetic_fpca_bundle(n_athletes, seed)
    reg = synthetic_registry()
    dictionary = at.build_dictionary(CFG, registry=reg, catalog={"p95_speed": "95th percentile of speed (valid frames)"})
    sustained = at.sustained_table(b["ss"], CFG)
    d = at.attempt_data(b["windows"], synthetic_attempts(b, seed), sustained, dictionary)
    a = at.athlete_data(b["athletes"], at.official_table(b["mart"], b["athletes"]), dictionary)
    tables = at.characterise_all(dictionary, d, a, CFG, seed=11, registry=reg)
    status = at.screening_table(tables[at.CHARACTERISTICS.name], dictionary, CFG)
    return {**b, "registry": reg, "dictionary": dictionary, "sustained": sustained, "d": d, "a": a, "tables": tables, "status": status}


def synthetic_meta(n: int = 200, positions: tuple[str, ...] = ("DL", "WR"), cohorts: tuple[int, ...] = (2023, 2024, 2025),
                   cohort_p: tuple[float, ...] | None = None, seed: int = 0) -> dict:
    """Minimal ``windows`` / ``athletes`` frames (two attempts per athlete) and their ``Meta``; official measurements are
    independent normal variables."""
    rng = np.random.default_rng(seed)
    ids = 7000 + np.arange(n)
    pos = np.array(positions)[np.arange(n) % len(positions)]
    coh = rng.choice(np.array(cohorts), size=n, p=cohort_p)
    off = rng.normal(size=(n, 4)) * np.array([0.10, 0.05, 20.0, 1.5]) + np.array([4.7, 1.6, 250.0, 75.0])
    athletes = pl.DataFrame({"nfl_id": ids, "position_group": pos, "draft_year": coh, FORTY: off[:, 0], TEN: off[:, 1],
                             WEIGHT: off[:, 2], HEIGHT: off[:, 3]})
    nfl = np.repeat(ids, 2)
    windows = pl.DataFrame({"seq_id": [f"E|{i}|{k}" for i in ids for k in (1, 2)], "nfl_id": nfl, "position_group": np.repeat(pos, 2),
                            "draft_year": np.repeat(coh, 2), "attempt_order": np.tile([1, 2], n),
                            **{v: rng.normal(size=2 * n) for v in fe.WINDOW_VARS}, **{v: np.zeros(2 * n, dtype=bool) for v in fe.FLAG_VARS},
                            "S2_signal": np.ones(2 * n, dtype=bool), "S3_no_boundary": np.ones(2 * n, dtype=bool)})
    return {"windows": windows, "athletes": athletes, "meta": fe.Meta(windows, athletes), "ids": ids, "nfl": nfl, "position": pos,
            "cohort": coh, "official": off, "seq": windows["seq_id"].to_numpy().astype(str),
            "roles": {int(i): ("design" if k % 2 == 0 else "confirm") for k, i in enumerate(ids)}}


def _emb(method: str, m: dict, Z: np.ndarray, role, view: str = "full") -> pl.DataFrame:
    return rv.embedding_frame(method, 0, 0, m["seq"], m["nfl"], role, view, Z, Z)


def _per_attempt(athlete_values: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    Z = np.repeat(athlete_values, 2, axis=0)
    return Z + noise * rng.normal(size=Z.shape)


# ============================================================================== atlas checks (notebook 04A)
def check_unique_feature_dictionary_keys() -> str:
    d = conform(at.build_dictionary(CFG, registry=synthetic_registry(), catalog={"p95_speed": "catalog text"}), at.DICTIONARY)
    at.assert_unique_keys(d)
    required = ("feature_id", "feature_name", "family", "drill", "position_applicability", "grain", "units", "definition",
                "source_module", "provenance", "smoothing", "time_or_distance_window", "measurement_status", "interpretation_status",
                "quality_flags", "version")
    assert set(required) <= set(d.columns) and d["feature_id"].n_unique() == d.height
    own = d.filter(pl.col("feature_name") == "p95_speed")
    assert own.filter(pl.col("drill") == "FORTY_YARD_DASH").height == 1                     # the forty feature is listed once
    assert "phase04:VALIDATED" in own.filter(pl.col("feature_id") == "F2.forty.p95_speed")["quality_flags"][0]
    assert own.filter(pl.col("drill") == "PASS_RUSH_DRILL").height == 2                     # same name, two strata: two features
    assert own.filter(pl.col("feature_id") == "F2.forty.p95_speed")["definition"][0] == "catalog text"
    event = d.filter(pl.col("feature_name") == "n_turn").row(0, named=True)
    assert event["family"] == "F3" and event["design"] == "event_rate" and event["interpretation_status"].startswith("unvalidated")
    assert not d["feature_id"].str.contains("nfl_play", literal=True).any()                 # only Combine rows of the registry
    assert set(d["family"].unique().to_list()) <= set(at.FAMILIES) and {"F1", "F2", "F4", "F6", "F7"} <= set(d["family"].to_list())
    new = [s[0] for s in at.sustained_specs(CFG)]
    assert len(new) == 4 and d.filter(pl.col("feature_name").is_in(new)).height == 4
    assert _raises(ValueError, lambda: at.assert_unique_keys(pl.concat([d, d.head(1)])))
    assert _raises(ContractError, lambda: conform(pl.concat([d, d.head(1)]), at.DICTIONARY))
    assert _raises(KeyError, lambda: at._entry("x", "x", "F2", "repeated_attempt", "candidate", unknown_field=1))
    return f"{d.height} definitions, one row per feature_id; a drill-specific variant keeps its own row; a duplicate is refused"


def check_aggregation_grain() -> str:
    e = engine()
    d, a, t = e["d"], e["a"], e["tables"]
    att, ath, cov = (conform(t[c.name], c) for c in (at.ATTEMPT_VALUES, at.ATHLETE_VALUES, at.COVERAGE))
    fid = "F2.forty.p95_speed"
    x = att.filter(pl.col("feature_id") == fid)
    assert x.height == d.seq_ids.size and x["seq_id"].n_unique() == x.height               # one row per attempt, missing rows kept
    mean = x.filter(pl.col("value").is_not_null()).group_by("nfl_id").agg(pl.col("value").mean().alias("m"), pl.len().alias("k")).sort("nfl_id")
    got = ath.filter(pl.col("feature_id") == fid).sort("nfl_id")
    assert got.height == mean.height == np.unique(d.nfl_ids).size                           # one row per ATHLETE, not per attempt
    assert np.allclose(got["value"].to_numpy(), mean["m"].to_numpy()) and (got["n_attempts"].to_numpy() == mean["k"].to_numpy()).all()
    c = cov.filter((pl.col("feature_id") == fid) & (pl.col("group_type") == "all")).row(0, named=True)
    assert c["n_athletes"] == a.ids.size and c["n_attempts"] == d.seq_ids.size and c["n_attempts"] > c["n_athletes"]
    assert c["n_athletes_repeat"] < c["n_athletes"]                                          # some athletes have one attempt only
    official = ath.filter(pl.col("feature_id") == f"F1.official.{FORTY}")
    assert official.height == a.ids.size and official["n_attempts"].null_count() == official.height
    ch = t[at.CHARACTERISTICS.name].filter((pl.col("feature_id") == fid) & (pl.col("scope") == "all")).row(0, named=True)
    assert ch["n_athletes"] == a.ids.size and ch["n_attempts"] == d.seq_ids.size
    by_pos = cov.filter((pl.col("feature_id") == fid) & (pl.col("group_type") == "position"))
    assert by_pos["n_athletes"].sum() == a.ids.size
    model = att.filter((pl.col("feature_id") == "F4.forty.model_tau_s") & pl.col("value").is_null())
    assert model.height > 0 and set(model["missing_reason"].unique().to_list()) == {"sprint_fit_not_ok"}
    return f"{d.seq_ids.size} attempts stay attempts; the athlete table holds {a.ids.size} athletes; a missing value keeps its reason"


def check_known_icc_recovery() -> str:
    rng = np.random.default_rng(0)
    g = 360
    ath, pos = np.repeat(np.arange(g), 2), np.repeat(np.array(["a", "b", "c"])[np.arange(g) % 3], 2)
    shift = np.repeat(np.array([0.0, 3.0, 6.0])[np.arange(g) % 3], 2)
    x = shift + np.repeat(rng.normal(size=g), 2) + 0.5 * rng.normal(size=2 * g)              # true within-position ICC = 0.80
    r = at.icc_estimate(x, ath, pos, 300, 1, 10)
    assert abs(r["icc"] - 0.80) < 0.06, r["icc"]
    assert r["ci_lower"] < r["icc"] < r["ci_upper"] and r["ci_upper"] - r["ci_lower"] < 0.15 and r["n_athletes_repeat"] == g
    assert abs(r["typical_error"] - 0.5) < 0.05                                              # SEM = within-athlete SD
    pooled = at.icc_estimate(x, ath, None, 0, 1, 10)["icc"]
    assert pooled > 0.93                                                                     # position differences inflate it
    noise = at.icc_estimate(shift + rng.normal(size=2 * g), ath, pos, 0, 1, 10)["icc"]
    assert abs(noise) < 0.15                                                                 # no athlete effect within position
    assert at.icc_estimate(x[:16], ath[:16], pos[:16], 0, 1, 10)["icc"] is None              # too few athletes: no estimate
    holes = x.copy()
    holes[::7] = np.nan
    assert abs(at.icc_estimate(holes, ath, pos, 0, 1, 10)["icc"] - 0.80) < 0.08              # missing attempts are dropped, not filled
    return f"ICC {r['icc']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] for a true 0.80; pooled over positions {pooled:.3f}"


def check_athlete_level_bootstrap() -> str:
    rng = np.random.default_rng(1)
    g = 200
    trait = rng.normal(size=g)
    attempts = np.repeat(trait, 2) + 0.05 * rng.normal(size=2 * g)                           # two nearly identical attempts each
    _, means, _ = fe.athlete_means(np.repeat(np.arange(g), 2), attempts)
    est, lo, hi = rv.boot_mean(means[:, 0], 800, 5)
    again = rv.boot_mean(means[:, 0], 800, 5)
    assert np.allclose(lo, again[1]) and np.allclose(hi, again[2])                           # reproducible
    se_athlete, se_naive = float(hi[0] - lo[0]) / 3.92, float(attempts.std(ddof=1) / np.sqrt(attempts.size))
    assert 1.2 < se_athlete / se_naive < 1.65, se_athlete / se_naive                         # attempts are not independent athletes
    def width(n: int) -> float:
        a = np.repeat(np.arange(n), 2)
        x = np.repeat(rng.normal(size=n), 2) + 0.7 * rng.normal(size=2 * n)
        r = at.icc_estimate(x, a, None, 400, 3, 10)
        return r["ci_upper"] - r["ci_lower"]
    w_small, w_large = width(60), width(240)
    assert w_small > 1.3 * w_large                                                           # the interval follows the number of ATHLETES
    m = synthetic_meta(120, seed=2)
    Z = _per_attempt(rng.normal(size=(120, 3)), 0.4, rng)
    p = rv.repeat_pairs(next(rv.emb_blocks(_emb("T", m, Z, "heldout"))), m["meta"], "standardised")
    scales = rb.matching_scales(m["meta"], m["roles"], MC)
    M, _, _ = rb.match_matrix(p["ids"], p["position"], m["meta"], scales, MC)
    item = {"D": rv.cdist(p["Z1"], p["Z2"]), "M": M, "confirm": np.arange(p["ids"].size) % 2 == 1}
    two, same = rb.two_way_intervals([item], 200, 9), rb.two_way_intervals([item], 200, 9)
    assert two == same and all(np.isfinite(v).all() and v[0] <= v[1] for v in two.values())
    point = np.nanmean(rb.gallery_stats(item["D"], M)["auc"])
    assert two["all"][0] - 0.02 <= point <= two["all"][1] + 0.02
    return (f"athlete-level SE is {se_athlete / se_naive:.2f} × the attempt-level one; ICC interval width {w_small:.2f} at 60 athletes, "
            f"{w_large:.2f} at 240; the two-way matched interval is reproducible")


def check_position_adjusted_statistics() -> str:
    rng = np.random.default_rng(2)
    n = 300
    pos = np.array(["DB", "DL", "OL"])[np.arange(n) % 3]
    level = np.array([4.4, 4.8, 5.2])[np.arange(n) % 3]
    forty = level + 0.05 * rng.normal(size=n)
    a = at.AthleteData(np.arange(n, dtype=np.int64), pos, np.array(["2023", "2024", "2025"])[rng.integers(0, 3, n)],
                       {FORTY: forty, TEN: 0.35 * forty + 0.02 * rng.normal(size=n), WEIGHT: 250 + 20 * rng.normal(size=n),
                        HEIGHT: 75 + rng.normal(size=n)}, {})
    ar = np.arange(n)
    by_position = -8.0 * level + rng.normal(size=n)                    # differs between positions, unrelated to the forty inside one
    r = at._athlete_level(by_position, ar, a, True, CFG, 1, "x")
    pooled = partial_corr(by_position, forty, (), (), "spearman")["r"]
    assert abs(pooled) > 0.7 and abs(r["rho_forty"]) < 0.2 and r["r2_stopwatch"] < 0.08, (pooled, r["rho_forty"])
    assert r["redundancy_lower"] < CFG.redundancy_reference
    copy = at._athlete_level(-forty + 0.01 * rng.normal(size=n), ar, a, True, CFG, 1, "y")
    assert copy["redundancy_abs"] > 0.9 and copy["redundancy_lower"] > CFG.redundancy_reference and copy["r2_stopwatch"] > 0.8
    assert abs(r["sd_within_position"] - 1.0) < 0.15                    # position means removed from the spread
    one = at._athlete_level(by_position[pos == "DL"], ar[pos == "DL"], a, False, CFG, 1, "x")
    assert abs(one["rho_forty"]) < 0.35 and np.isnan(one["r2_stopwatch_lower"])             # inside a position: no dummy, no bootstrap
    cohort = at._athlete_level(np.where(a.cohort == "2025", 3.0, 0.0) + rng.normal(size=n), ar, a, True, CFG, 1, "z")
    assert cohort["cohort_r2"] > 0.5 and r["cohort_r2"] < 0.1
    skip = at._athlete_level(forty, ar, a, True, CFG, 1, "f", skip=(FORTY,))
    assert "rho_forty" not in skip and "rho_ten" in skip                # a reference is never correlated with itself
    return f"pooled ρ {pooled:+.2f} becomes {r['rho_forty']:+.2f} within position; a copy of the forty is flagged redundant"


def check_sustained_features_use_measured_intervals_only() -> str:
    vmax, tau = 10.0, 0.9
    def run(offset: float, n: int, top: float = vmax) -> dict:
        u = offset + 0.1 * np.arange(n)
        v = top * (1 - np.exp(-u / tau))
        d = top * (u - tau * (1 - np.exp(-u / tau)))
        return at.sustained_attempt(0.1 * np.arange(n), v, d - d[0], CFG)
    names = [s[0] for s in at.sustained_specs(CFG)]
    full = run(0.05, 50)
    u0 = -tau * np.log(1 - CFG.anchor_speed / vmax)                     # analytic moment the anchor speed is reached
    truth = (vmax * (1 - np.exp(-(u0 + 1.0) / tau)) - CFG.anchor_speed) / 1.0
    assert all(why == "" for _, why in full.values()) and abs(full[names[1]][0] - truth) < 0.08, (full[names[1]], truth)
    u1 = -tau * np.log(1 - CFG.upper_speed / vmax)
    assert abs(full[names[3]][0] - (u1 - u0)) < 0.03 and full[names[0]][0] > full[names[1]][0] > 0
    assert 0 < full[names[2]][0] < vmax - CFG.anchor_speed
    longer = run(0.05, 58)
    assert all(abs(full[k][0] - longer[k][0]) < 1e-12 for k in names)   # frames after the window do not change a value
    late = run(1.2, 45)                                                 # recording starts above the anchor speed
    assert all(np.isnan(x) and why == "start_above_anchor" for x, why in late.values())
    short = run(0.05, 14)                                               # stops 1.3 s after the first frame
    assert short[names[1]][1] == "window_exceeds_recording" and np.isnan(short[names[1]][0]) and short[names[2]][1] == "window_exceeds_recording"
    slow = run(0.05, 50, top=7.0)
    assert slow[names[3]][1] == "upper_threshold_not_reached" and slow[names[0]][1] == ""
    assert all(why == "anchor_not_reached" for _, why in run(0.05, 50, top=3.0).values())
    assert all(why == "too_few_frames" for _, why in at.sustained_attempt(np.array([0.0, 0.1]), np.array([1.0, 2.0]), np.array([0.0, 0.1]), CFG).values())
    t = conform(engine()["sustained"], at.SUSTAINED)
    assert t.filter(pl.col("value").is_null() & (pl.col("missing_reason") == "")).height == 0
    assert t.filter(pl.col("value").is_not_null() & (pl.col("missing_reason") != "")).height == 0
    return f"1.0 s mean acceleration {full[names[1]][0]:.3f} against {truth:.3f} analytic; absent intervals stay missing with a reason"


def check_screening_rules() -> str:
    spec = {"role": "candidate", "design": "repeated_attempt", "source_table": at.ATTEMPT_TABLE, "source_column": "x", "family": "F6",
            "interpretation_status": "interpretable", "quality_flags": ""}
    good = {"feature_id": "f", "coverage_athletes": 0.95, "n_athletes_repeat": 300, "reliability": 0.85, "reliability_lower": 0.80,
            "reliability_upper": 0.90, "reliability_of_mean": 0.92, "redundancy_abs": 0.3, "redundancy_lower": 0.2, "redundancy_upper": 0.4,
            "r2_stopwatch": 0.10, "r2_stopwatch_upper": 0.18, "window_r2": 0.05, "window_r2_lower": 0.01, "window_r2_upper": 0.12,
            "cohort_r2": 0.02}
    s = at.screen_feature(good, spec, CFG)
    assert (s["grade"], s["stopwatch_relation"], s["acquisition"], s["transfer_eligibility"]) == (at.GRADES[0], "beyond_stopwatch", "robust", "candidate")
    red = at.screen_feature({**good, "redundancy_abs": 0.9, "redundancy_lower": 0.85, "redundancy_upper": 0.93}, spec, CFG)
    assert red["stopwatch_relation"] == "redundant" and red["transfer_eligibility"] == "control" and red["grade"] == at.GRADES[0]
    weak = at.screen_feature({**good, "reliability": 0.55, "reliability_lower": 0.42, "reliability_upper": 0.66, "reliability_of_mean": 0.20,
                              "redundancy_upper": 0.4}, spec, CFG)
    assert weak["stopwatch_relation"] == "low_correlation_only" and weak["grade"] == at.GRADES[2]      # low |ρ| alone is never 'beyond'
    assert weak["transfer_eligibility"] == "exploratory_only"
    low = at.screen_feature({**good, "reliability": 0.3, "reliability_lower": 0.2, "reliability_upper": 0.4}, spec, CFG)
    assert low["grade"] == at.GRADES[3] and low["transfer_eligibility"] == "not_eligible"
    window = at.screen_feature({**good, "window_r2": 0.6, "window_r2_lower": 0.5, "window_r2_upper": 0.7}, spec, CFG)
    assert window["acquisition"] == "sensitive" and window["grade"] == at.GRADES[2]
    moderate = at.screen_feature({**good, "reliability": 0.65, "reliability_lower": 0.55, "reliability_upper": 0.73}, spec, CFG)
    assert moderate["grade"] == at.GRADES[1]
    thin = at.screen_feature({**good, "coverage_athletes": 0.4}, spec, CFG)
    assert thin["support"] == "inadequate" and thin["grade"] == at.GRADES[3]
    assert at.screen_feature(good, {**spec, "role": "nuisance"}, CFG)["grade"] == at.GRADES[3]
    assert at.screen_feature(good, {**spec, "quality_flags": "phase04:REJECTED"}, CFG)["grade"] == at.GRADES[3]
    ref = at.screen_feature({"feature_id": "o", "coverage_athletes": 0.99}, {**spec, "role": "reference", "design": "static_athlete",
                                                                             "source_table": at.MART_TABLE, "source_column": FORTY}, CFG)
    assert (ref["reliability_label"], ref["stopwatch_relation"], ref["transfer_eligibility"]) == ("not_estimable", "is_reference", "reference_baseline")
    vec = at.screen_feature({"feature_id": "v", "n_athletes_repeat": 300, "reliability": 0.92, "reliability_lower": 0.90,
                             "reliability_upper": 0.94, "retrieval_auc": 0.87, "window_r2": 0.1, "window_r2_lower": 0.05,
                             "window_r2_upper": 0.2, "cohort_probe_gain": 0.02},
                            {**spec, "design": "neural_embedding", "source_table": at.VECTOR_TABLE}, CFG)
    assert vec["grade"] == at.GRADES[1] and vec["stopwatch_relation"] == "pending_04b"       # a vector is capped until notebook 04B
    labels, order = at.cluster_features(np.array([[1, .9, .1, 0], [.9, 1, .05, 0], [.1, .05, 1, np.nan], [0, 0, np.nan, 1]]), 0.7)
    assert labels[0] == labels[1] and len(set(labels.tolist())) == 3 and sorted(order) == [0, 1, 2, 3] and order[:2] == [0, 1]
    return "grades follow the stated rules; a low stopwatch correlation without reliable unexplained variance is not called new"


def check_engine_end_to_end() -> str:
    e = engine()
    t, status, dictionary = e["tables"], conform(e["status"], at.STATUS), e["dictionary"]
    for c in at.CHAR_CONTRACTS:
        conform(t[c.name], c)
    ch = t[at.CHARACTERISTICS.name]
    one = lambda f: ch.filter((pl.col("feature_id") == f) & (pl.col("scope") == "all")).row(0, named=True)      # noqa: E731
    speed, accel = one("F2.forty.p95_speed"), one("F2.forty.peak_a_tan")
    assert speed["reliability"] > 0.9 and accel["reliability"] < 0.4, (speed["reliability"], accel["reliability"])
    assert speed["repeat_spearman"] > 0.8 and speed["sem"] < 0.05 and speed["coverage_athletes"] == 1.0
    assert one(f"F1.official.{FORTY}")["reliability"] is None and "not estimable" in one(f"F1.official.{FORTY}")["reliability_method"]
    assert ch.filter(pl.col("scope").str.starts_with("position:"))["scope"].n_unique() == 3
    grade = dict(status.select("feature_id", "grade").iter_rows())
    assert grade["F2.forty.p95_speed"] in at.GRADES[:2] and grade["F2.forty.peak_a_tan"] not in at.GRADES[:2]
    assert grade["F7.forty.first_speed"] == at.GRADES[3] and grade["F4.forty.model_r2"] == at.GRADES[3]
    assert grade["F1.official.cmb_vertical"] == at.GRADES[3]                                  # absent from the fixture: no support
    assert grade["F3.FORTY_YARD_DASH:FORTY_YARD_DASH.n_turn"] == at.GRADES[3] and grade["F2.SKILL_DRILLS_LB:PASS_RUSH_DRILL.p95_speed"] == at.GRADES[2]
    rel = t[at.RELIABILITY.name].filter(pl.col("feature_id") == "F2.forty.p95_speed")
    assert {"position_adjusted", "scenario:S2_signal", "scenario:S3_no_boundary"} <= set(rel["scope"].to_list())
    nuis = t[at.NUISANCE.name].filter(pl.col("feature_id") == "F2.forty.p95_speed")
    assert {"window_joint", "draft_cohort", "first_speed"} <= set(nuis["variable"].to_list())
    feats = at.matrix_features(dictionary, status)
    M = at.athlete_matrix(feats, e["d"], e["a"])
    R, N = at.redundancy_matrix(M, e["a"].position, min_athletes=10)
    red = at.redundancy_tables(feats, R, N, status, CFG)
    conform(red[at.MATRIX.name], at.MATRIX)
    cl = conform(red[at.CLUSTERS.name], at.CLUSTERS)
    assert cl.height == len(feats) and sorted(cl["display_order"].to_list()) == list(range(len(feats))) and np.allclose(np.diag(R), 1)
    assert set(cl["representative"].to_list()) <= set(feats)
    ret = conform(at.quality_retention(e["windows"]), at.RETENTION)
    assert (ret.filter(pl.col("scenario") == "S0_all")["share_athletes"] == 1.0).all()
    text = at.conclusions_markdown(status, dictionary, CFG, "smoke")
    assert "NOT TESTED" in text and "must not be interpreted" in text and "Demonstrated NFL predictive value" in text
    inv = at.reuse_inventory.__doc__
    assert "reuses" in inv
    return (f"{ch.height} characteristic rows over {status.height} features satisfy their contracts; planted repeatable speed ICC "
            f"{speed['reliability']:.2f}, planted noise ICC {accel['reliability']:.2f}")


# ============================================================================== robustness checks (notebook 04B)
def check_fold_safe_residualisation() -> str:
    b = fv.synthetic_fpca_bundle(60, 0)
    meta, w = fe.Meta(b["windows"], b["athletes"]), b["windows"]
    seq, nfl = w["seq_id"].to_numpy().astype(str), w["nfl_id"].to_numpy()
    rng = np.random.default_rng(0)
    ids = np.unique(nfl)
    trait = dict(zip(ids.tolist(), rng.normal(size=ids.size)))
    forty = meta.official[FORTY][meta.athlete_rows(nfl)]
    Z = np.column_stack([(forty - forty.mean()) / forty.std() + 0.05 * rng.normal(size=nfl.size),
                         np.array([trait[int(i)] for i in nfl]) + 0.05 * rng.normal(size=nfl.size), 0.3 * rng.normal(size=nfl.size)])
    held = np.isin(nfl, ids[::2])
    role = np.where(held, "heldout", "fit")
    block = next(rv.emb_blocks(rv.embedding_frame("SYN", 0, 0, seq, nfl, role, "full", Z, Z)))
    out = {m: rb.residual_variant(block, meta, "official", m, ECFG, 3) for m in rb.RESID_MODES}
    hb = np.flatnonzero(block.held)
    for mode, v in out.items():
        assert np.array_equal(v["nfl"], block.nfl_ids[hb]) and np.isfinite(v["R"]).all(), mode
        two, i1, i2 = fe.first_two(v["nfl"], meta.order[meta.rows(v["seq"])])
        if mode != "crossfit_permuted":                                  # athlete-level covariates: one vector leaves both attempts
            assert np.allclose(v["R"][i1] - v["R"][i2], block.Zs[hb][i1] - block.Zs[hb][i2], atol=1e-9), mode
    assert out["train_fit"]["R"][:, 0].var() < 0.3 * block.Zs[hb][:, 0].var()               # the planted stopwatch dimension is removed
    assert out["train_fit"]["R"][:, 1].var() > 0.5 * block.Zs[hb][:, 1].var()               # the unrelated athlete trait stays
    assert out["crossfit_permuted"]["R"][:, 0].var() > 0.5 * block.Zs[hb][:, 0].var()       # permuted covariates remove nothing real
    assert np.unique(out["crossfit"]["probe_fold"]).size == ECFG.probe_folds and (out["train_fit"]["probe_fold"] == 0).all()
    keep = block.nfl_ids[hb][0]
    Z2 = Z.copy()
    other = held & (nfl != keep)
    Z2[other] = Z2[other] * 3.0 + 1.0                                    # change every OTHER held-out athlete
    block2 = next(rv.emb_blocks(rv.embedding_frame("SYN", 0, 0, seq, nfl, role, "full", Z2, Z2)))
    again = rb.residual_variant(block2, meta, "official", "train_fit", ECFG, 3)
    mine = out["train_fit"]["nfl"] == keep
    assert np.allclose(again["R"][mine], out["train_fit"]["R"][mine], atol=1e-9)             # fitted on training athletes only
    cross2 = rb.residual_variant(block2, meta, "official", "crossfit", ECFG, 3)
    assert not np.allclose(cross2["R"][mine], out["crossfit"]["R"][mine], atol=1e-6)         # cross-fitting does depend on the others
    assert _raises(ValueError, lambda: rb.residual_variant(block, meta, "official", "everything", ECFG, 3))
    assert _raises(ValueError, lambda: rb.residual_variant(block, meta, "official_window", "crossfit_permuted", ECFG, 3))
    roles = {int(i): str(r) for i, r in b["splits"].select("nfl_id", "design_role").unique().iter_rows()}
    audit = conform(rb.residual_audit(rv.embedding_frame("SYN", 0, 0, seq, nfl, role, "full", Z, Z), meta, roles, ECFG, 3), rb.RESID_AUDIT)
    raw = fe._one(audit, method="SYN", covariates="none", mode="raw", metric="euclidean", subset="all")
    tf = fe._one(audit, method="SYN", covariates="official", mode="train_fit", metric="euclidean", subset="all")
    assert raw and tf and abs(tf["d_same_vs_raw"] - 1.0) < 1e-9 and tf["variance_share"] < 1.0 and raw["auc_minus_raw"] is not None
    assert {"euclidean", "euclidean_restandardised", "cosine"} == set(audit["metric"].unique().to_list())
    assert "raw AUC" in rb.audit_reading(audit, "SYN", ECFG, subset="all")
    return ("a training-fitted residual of an athlete does not move when the other held-out athletes change; the own-repeat distance "
            "is unchanged when one vector is removed from both attempts")


def check_cohort_probe_balance() -> str:
    m = synthetic_meta(600, cohort_p=(0.7, 0.2, 0.1), seed=3)
    rng = np.random.default_rng(3)
    role = np.where(np.repeat(np.arange(600) % 2 == 0, 2), "heldout", "fit")
    noise = _emb("NOISE", m, _per_attempt(rng.normal(size=(600, 4)), 0.2, rng), role)
    onehot = np.column_stack([(m["cohort"] == c).astype(float) for c in (2023, 2024, 2025)])
    planted = _emb("PLANTED", m, _per_attempt(np.column_stack([onehot, rng.normal(size=600)]), 0.1, rng), role)
    t = conform(rb.cohort_probe(pl.concat([noise, planted]), m["meta"], ECFG, 4), rb.COHORT_PROBE)
    assert t.height == 4 and set(t["protocol"].to_list()) == {"heldout_crossfit", "train_to_heldout"}
    n, p = fe._one(t, method="NOISE", protocol="heldout_crossfit"), fe._one(t, method="PLANTED", protocol="heldout_crossfit")
    assert n["n_classes"] == 3 and abs(n["chance"] - 1 / 3) < 1e-12 and 0.6 < n["majority_share"] < 0.8 and "2023" in n["class_counts"]
    assert 0.25 < n["balanced_accuracy"] < 0.45 and n["accuracy"] > 0.55 and abs(n["gain_balanced"]) < 0.12, n   # accuracy ≈ majority
    assert p["balanced_accuracy"] > 0.8 and p["gain_balanced"] > 0.3 and p["gain_lower"] > 0, p
    assert n["bal_lower"] - 0.02 <= n["balanced_accuracy"] <= n["bal_upper"] + 0.02 and n["n_athletes"] == 300
    assert abs(rb.balanced_accuracy(np.array(["a", "a", "a", "b"]), np.array(["a", "a", "a", "a"])) - 0.5) < 1e-12
    return (f"an embedding without cohort information scores {n['accuracy']:.2f} plain accuracy but {n['balanced_accuracy']:.2f} balanced "
            f"(chance 0.33); a planted cohort signal scores {p['balanced_accuracy']:.2f}")


def check_retrieval_matching() -> str:
    m = synthetic_meta(400, seed=5)
    meta, roles = m["meta"], m["roles"]
    scales = rb.matching_scales(meta, roles, MC)
    assert all(0.7 * s < scales[v] < 1.3 * s for v, s in zip((FORTY, TEN, WEIGHT, HEIGHT), (0.10, 0.05, 20.0, 1.5))), scales
    confirm = np.array([roles[int(i)] == "confirm" for i in m["ids"]])
    moved = m["athletes"].with_columns(pl.Series(FORTY, np.where(confirm, m["official"][:, 0] * 10.0, m["official"][:, 0])))
    assert rb.matching_scales(fe.Meta(m["windows"], moved), roles, MC) == scales              # read-out athletes do not set the rule
    M, D, diffs = rb.match_matrix(m["ids"], m["position"], meta, scales, MC)
    assert not M.diagonal().any() and (M.sum(1) <= MC.k_nearest).all() and (D[M] <= MC.caliper).all()
    assert (m["position"][:, None] == m["position"][None, :])[M].all() and M.sum() > 0
    cand = (m["position"][:, None] == m["position"][None, :]) & ~np.eye(400, dtype=bool) & (D <= MC.caliper)
    i = int(np.argmax(cand.sum(1)))
    assert cand[i].sum() > MC.k_nearest and M[i].sum() == MC.k_nearest
    assert D[i][M[i]].max() <= D[i][cand[i] & ~M[i]].min() + 1e-12                            # the k NEAREST candidates
    assert np.allclose(diffs[FORTY], -diffs[FORTY].T) and set(diffs) >= {FORTY, TEN, WEIGHT, HEIGHT}
    hole = fe.Meta(m["windows"], m["athletes"].with_columns(pl.when(pl.col("nfl_id") == int(m["ids"][0])).then(None).otherwise(pl.col(TEN)).alias(TEN)))
    M2, _, _ = rb.match_matrix(m["ids"], m["position"], hole, scales, MC)
    assert not M2[0].any() and not M2[:, 0].any()                                             # no required value: never matched
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    off = ~np.eye(4, dtype=bool)
    st = rb.gallery_stats(rv.cdist(centres + 0.1, centres - 0.1), off)
    assert np.allclose(st["auc"], 1) and np.allclose(st["top1"], 1) and st["n"].tolist() == [3] * 4
    tie = rb.gallery_stats(np.zeros((3, 3)), ~np.eye(3, dtype=bool))
    assert np.allclose(tie["auc"], 0.5) and np.allclose(tie["top1"], 0)
    assert np.isnan(rb.gallery_stats(rv.cdist(centres, centres), np.zeros((4, 4), dtype=bool))["auc"]).all()
    w = rb.gallery_stats(rv.cdist(centres + 0.1, centres - 0.1), off, weights=np.array([2.0, 0.0, 1.0, 1.0]))
    assert np.allclose(w["auc"], 1)
    rng = np.random.default_rng(6)
    z = (m["official"] - m["official"].mean(0)) / m["official"].std(0)
    emb = pl.concat([_emb("STOPWATCH", m, _per_attempt(z, 0.5, rng), "heldout"),
                     _emb("TRAIT", m, _per_attempt(rng.normal(size=(400, 4)), 0.5, rng), "heldout")])
    out = rb.matched_retrieval(emb, meta, roles, scales, MC, ECFG, 7, pairs_for=("TRAIT",))
    per, pairs, summary = (conform(out[c.name], c) for c in (rb.MATCHED, rb.MATCH_PAIRS, rb.MATCH_SUMMARY))
    s, t = fe._one(summary, method="STOPWATCH", subset="all"), fe._one(summary, method="TRAIT", subset="all")
    assert s["auc_position"] > 0.85 and s["matching_penalty"] > 0.15 and s["matching_penalty_lower"] > 0, s   # only the stopwatch: falls
    assert t["auc_matched"] > 0.85 and abs(t["matching_penalty"]) < 0.06, t                                  # an own trait: stays
    assert t["auc_matched_lower2"] - 0.02 <= t["auc_matched"] <= t["auc_matched_upper2"] + 0.02
    assert t["n_comparisons"] <= MC.k_nearest * t["n_athletes"]
    assert t["n_distinct_impostors"] <= 400 and t["impostor_reuse_max"] >= 1 and t["n_athletes"] + t["n_unmatched"] == 400
    assert pairs.height == t["n_comparisons"] and set(pairs["method"].unique().to_list()) == {"TRAIT"} and pairs["rank"].min() == 1
    assert (pairs["match_distance"] <= MC.caliper).all() and fe._one(summary, method="TRAIT", subset="role:confirm")["n_athletes"] < t["n_athletes"]
    paired = conform(rb.matched_paired(per, {"TRAIT": "STOPWATCH"}, ECFG, 7), rb.MATCH_PAIRED)
    d = fe._one(paired, method="TRAIT", reference="STOPWATCH", subset="all")
    assert d["difference"] > 0.2 and d["diff_lower"] > 0
    ex = rb.matched_examples(pairs, per, "TRAIT", m["athletes"].with_columns(pl.lit("x").alias("display_name")), k=4)
    assert 1 <= len(ex) <= 4 and all(e["nfl_id"] != e["impostor_id"] for e in ex)
    assert rb.MatchConfig.from_dict(MC.params()) == MC
    return (f"a stopwatch-only embedding drops from {s['auc_position']:.2f} to {s['auc_matched']:.2f} against matched impostors; an athlete "
            f"trait keeps {t['auc_matched']:.2f}; matches respect position, caliper and k")


def check_seed_aggregation() -> str:
    runs = rv._table([{"method": f"M{i}", "training_seed": i, "metric": "auc", "value": v, "lower": v - 0.0392, "upper": v + 0.0392}
                      for i, v in enumerate((0.80, 0.82, 0.84))]
                     + [{"method": "M0", "training_seed": 0, "metric": "single", "value": 0.5, "lower": None, "upper": None}], rb.SEED_RUNS)
    agg = conform(rb.aggregate_seeds(runs), rb.SEED_SUMMARY)
    a = fe._one(agg, metric="auc")
    assert a["n_seeds"] == 3 and abs(a["mean"] - 0.82) < 1e-12 and abs(a["sd_seed"] - 0.02) < 1e-9 and abs(a["se_athlete"] - 0.02) < 1e-9
    assert abs(a["sd_total"] - np.sqrt(0.0008)) < 1e-9 and abs(a["seed_share"] - 0.5) < 1e-9 and (a["min"], a["max"]) == (0.80, 0.84)
    one = fe._one(agg, metric="single")
    assert one["n_seeds"] == 1 and one["sd_seed"] is None and one["se_athlete"] is None      # one seed: no spread is invented
    summary = rv._table([{"method": m, "label": m, "auc": 0.8 + 0.01 * i, "auc_lower": 0.75, "auc_upper": 0.85, "icc_trace": 0.9,
                          "rank_share_min": 0.3, "collapsed": False} for i, m in enumerate(("A", "B"))], rv.SUMMARY)
    trace = rv._table([{"method": m, "space": "raw", "icc_trace": 0.9, "icc_lower": 0.88, "icc_upper": 0.92} for m in ("A", "B")], rv.ICC)
    matched = rv._table([{"method": m, "subset": "role:confirm", "auc_matched": 0.7, "auc_matched_lower2": 0.6, "auc_matched_upper2": 0.8}
                         for m in ("A", "B")], rb.MATCH_SUMMARY)
    long = conform(rb.seed_runs(summary, {"A": 1, "B": 2}, trace=trace, matched=matched), rb.SEED_RUNS)
    assert long.height == 2 * (len(rb.SEED_METRICS) + 1) and set(long["training_seed"].to_list()) == {1, 2}
    icc = fe._one(long, method="A", metric="icc_trace")
    assert (icc["lower"], icc["upper"]) == (0.88, 0.92) and fe._one(long, method="B", metric="matched_auc")["lower"] == 0.6
    assert abs(fe._one(long, method="B", metric="auc")["value"] - 0.81) < 1e-12 and fe._one(long, method="A", metric="rank_share_min")["lower"] is None
    return "mean, SD between seeds and athlete standard error are reported separately; a single seed yields no spread"


def check_effective_rank_calculation() -> str:
    rng = np.random.default_rng(8)
    iso = rv.collapse_metrics(rng.normal(size=(1200, 8)))
    line = rv.collapse_metrics(np.outer(rng.normal(size=1200), np.ones(8)) + 1e-4 * rng.normal(size=(1200, 8)))
    two = rv.collapse_metrics(np.column_stack([rng.normal(size=(1200, 2)), 1e-4 * rng.normal(size=(1200, 6))]))
    assert iso["effective_rank"] > 7.8 and line["effective_rank"] < 1.05 and 1.9 < two["effective_rank"] < 2.05
    assert abs(iso["rank_share"] - iso["effective_rank"] / 8) < 1e-12 and line["top_eigen_share"] > 0.99
    assert rv.is_collapsed(line, ECFG) and not rv.is_collapsed(iso, ECFG)
    X = rng.normal(size=(300, 5))
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert abs(rb.linear_cka(X, 2.5 * X @ Q + 3.0) - 1.0) < 1e-9                              # rotation, scale and shift do not matter
    assert rb.linear_cka(X, rng.normal(size=(300, 5))) < 0.2
    m = synthetic_meta(100, seed=9)
    Z = _per_attempt(rng.normal(size=(100, 5)), 0.3, rng)
    emb = pl.concat([_emb("A", m, Z, "heldout"), _emb("B", m, Z @ Q, "heldout"), _emb("C", m, rng.normal(size=Z.shape), "heldout")])
    g = conform(rb.cross_geometry(emb, ["A", "B", "C"]), rb.GEOMETRY)
    ab, ac = fe._one(g, method_a="A", method_b="B"), fe._one(g, method_a="A", method_b="C")
    assert ab["cka"] > 0.999 and ab["distance_spearman"] > 0.999 and ab["neighbour_overlap"] > 0.999 and ab["n_sequences"] == 200
    assert ac["cka"] < 0.3 and ac["neighbour_overlap"] < 0.3
    return f"effective rank {iso['effective_rank']:.2f} of 8 for an isotropic cloud, {line['effective_rank']:.2f} for a line; geometry agreement ignores rotations"


# ------------------------------------------------------------------------------ projections (notebook 04B)
# Fixture: 150 athletes × 2 attempts in 8 dimensions of decreasing spread, three outer folds (fold f holds out every
# third athlete), a second view on the held-out rows only. Axes of two separate fits are never compared coordinate by
# coordinate — their sign is arbitrary — but through Gram matrices and squared distances, which ignore it.
PROJ_DIMS, PROJ_FOLDS = (2, 4, 8), 3
PROJ_SCALE = np.array([3.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.1])


def _require(ok: Any, what: str, **context: Any) -> None:
    """``assert`` with a message: what was expected and the context (fold, k, shapes, observed and expected values)."""
    if not bool(ok):
        raise AssertionError(what + (" | " + ", ".join(f"{k} = {v}" for k, v in context.items()) if context else ""))


def _close(what: str, observed: Any, expected: Any, atol: float, rtol: float = 0.0, **context: Any) -> None:
    """|observed − expected| ≤ atol + rtol·|expected| everywhere (a non-finite value fails); the message gives the worst entry."""
    o, e = np.atleast_1d(np.asarray(observed, float)), np.atleast_1d(np.asarray(expected, float))
    _require(o.shape == e.shape, f"{what}: shapes differ", observed_shape=o.shape, expected_shape=e.shape, **context)
    err = np.nan_to_num(np.abs(o - e), nan=np.inf, posinf=np.inf)
    bad = ~(err <= atol + rtol * np.abs(e))
    if bad.any():
        i = np.unravel_index(int(np.argmax(np.where(bad, err, -1.0))), err.shape)
        _require(False, f"{what}: {int(bad.sum())} of {o.size} value(s) outside the tolerance", worst_index=tuple(int(j) for j in i),
                 observed=repr(float(o[i])), expected=repr(float(e[i])), abs_difference=f"{err[i]:.3e}", atol=f"{atol:.3e}",
                 rtol=f"{rtol:.1e}", shape=o.shape, **context)


def projection_embedding(n: int = 150, seed: int = 10) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    return synthetic_meta(n, seed=seed), _per_attempt(rng.normal(size=(n, PROJ_SCALE.size)) * PROJ_SCALE, 0.05, rng)


def _fold_held(m: dict, fold: int) -> np.ndarray:
    return np.repeat(np.arange(m["ids"].size) % PROJ_FOLDS == fold, 2)


def projection_frames(m: dict, Z: np.ndarray, folds: Sequence[int] = tuple(range(PROJ_FOLDS)), method: str = "SYN") -> pl.DataFrame:
    """Long-form embeddings of ``Z``: per fold a ``full`` view of every row and a ``head_15`` view of the held-out rows."""
    parts = []
    for f in folds:
        held = _fold_held(m, f)
        parts.append(rv.embedding_frame(method, 0, f, m["seq"], m["nfl"], np.where(held, "heldout", "fit"), "full", Z, Z))
        parts.append(rv.embedding_frame(method, 0, f, m["seq"][held], m["nfl"][held], "heldout", "head_15", Z[held] * 0.9, Z[held] * 0.9))
    return pl.concat(parts)


def _by_fold(emb: pl.DataFrame, view: str = "full") -> dict[tuple[str, int], rv.EmbBlock]:
    return {(b.method, b.fold): b for b in rv.emb_blocks(emb, view)}


def _sq(A: np.ndarray, Bm: np.ndarray | None = None) -> np.ndarray:
    """Squared distances from differences. ``rv.cdist`` expands ‖a‖² + ‖b‖² − 2a·b and takes a root: at a distance of
    zero (an attempt against itself) the rounding of the three terms, ~1e-14, becomes ~1e-7 after the root."""
    return ((A[:, None, :] - (A if Bm is None else Bm)[None, :, :]) ** 2).sum(-1)


def _gram(P: np.ndarray) -> np.ndarray:
    return P @ P.T                                                          # unchanged by a sign flip or rotation of the axes


def _project(emb: pl.DataFrame, method: str, dims: Sequence[int], warns: str | None = None) -> pl.DataFrame:
    """``rb.project_embeddings`` with its ``ProjectionWarning`` checked: none expected, or one containing ``warns``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        P = rb.project_embeddings(emb, method, dims)
    text = " ; ".join(str(w.message) for w in caught if issubclass(w.category, rb.ProjectionWarning))
    if warns is None:
        _require(not text, "project_embeddings warned although every k can be honoured", method=method, dims=tuple(dims), warning=text)
    else:
        _require(warns in text, "project_embeddings did not report what it left out or could not determine", method=method,
                 dims=tuple(dims), expected_in_warning=repr(warns), observed_warning=repr(text))
    return P


def check_projection_dimensionality() -> str:
    m, Z = projection_embedding()
    emb = projection_frames(m, Z)
    P = conform(_project(emb, "SYN", PROJ_DIMS), rv.EMBEDDINGS)
    names = sorted(P["method"].unique().to_list())
    _require(names == sorted(f"SYN|pc{k}" for k in PROJ_DIMS), "projected method names", observed=names, expected_dims=PROJ_DIMS)
    _require(sorted(P["view"].unique().to_list()) == ["full", "head_15"], "every view is projected", observed=P["view"].unique().to_list())
    for view in ("full", "head_15"):
        src, out = _by_fold(emb, view), _by_fold(P, view)
        for f in range(PROJ_FOLDS):
            s = src[("SYN", f)]
            for k in PROJ_DIMS:
                ctx = {"fold": f, "k": k, "view": view}
                _require((f"SYN|pc{k}", f) in out, "a requested projection is missing", **ctx)
                b = out[(f"SYN|pc{k}", f)]
                _require(b.Zs.shape == (s.Zs.shape[0], k), "shape of the projected block", observed=b.Zs.shape,
                         expected=(s.Zs.shape[0], k), **ctx)
                _require(np.array_equal(b.seq_ids, s.seq_ids) and np.array_equal(b.nfl_ids, s.nfl_ids) and np.array_equal(b.role, s.role),
                         "a projection keeps the sequences, athletes and roles of its source", **ctx)
                _close("value and value_z of a projection are the same coordinates", b.Z, b.Zs, 0.0, **ctx)
            for small, large in zip(PROJ_DIMS, PROJ_DIMS[1:]):                 # same fit: the first axes are shared
                _close(f"pc{small} = the first {small} axes of pc{large}", out[(f"SYN|pc{small}", f)].Zs,
                       out[(f"SYN|pc{large}", f)].Zs[:, :small], 1e-12, fold=f, view=view)
    over = _project(emb, "SYN", (4, 16), warns="k = 16")
    _require(over["method"].unique().to_list() == ["SYN|pc4"], "k = 16 > dimension 8 is left out, never replaced by another k",
             observed=sorted(over["method"].unique().to_list()), expected=["SYN|pc4"])
    _require(_project(emb, "SYN", (16,), warns=rb.PROJ_OVER_DIM).height == 0, "k above the dimension alone: no row")
    _require(_project(emb, "OTHER", (2,)).height == 0, "an unknown method yields no row")
    rep = rb.projection_report(emb, "SYN", (*PROJ_DIMS, 16))
    _require(rep.height == PROJ_FOLDS * 4 and (rep["embedding_dim"] == 8).all() and (rep["training_rank"] == 8).all()
             and (rep["n_train_rows"] == 200).all() and (rep["n_heldout_rows"] == 100).all(), "projection report of the fixture",
             observed=rep.to_dicts())
    status = {k: sorted(rep.filter(pl.col("k") == k)["status"].unique().to_list()) for k in (*PROJ_DIMS, 16)}
    _require(status == {2: [rb.PROJ_OK], 4: [rb.PROJ_OK], 8: [rb.PROJ_OK], 16: [rb.PROJ_OVER_DIM]}, "status by k", observed=status)
    return f"k = {PROJ_DIMS} give 2, 4 and 8 columns in {PROJ_FOLDS} folds and both views; k = 16 is reported and left out"


def check_projection_orthogonality() -> str:
    m, Z = projection_embedding()
    emb = projection_frames(m, Z)
    src, out = _by_fold(emb), _by_fold(_project(emb, "SYN", PROJ_DIMS))
    for f in range(PROJ_FOLDS):
        s = src[("SYN", f)]
        train = ~s.held
        X = s.Zs[train]
        ctx = {"fold": f, "n_train_rows": int(train.sum()), "n_heldout_rows": int(s.held.sum())}
        mu, V, w, rank = rb.training_axes(X)
        _close("axes are orthonormal (VᵀV = I)", V.T @ V, np.eye(8), 1e-12, **ctx)
        _require((np.diff(w) <= 0).all() and rank == 8, "training variances decrease and the rank is full",
                 variances=[f"{x:.4e}" for x in w], rank=rank, **ctx)
        _, sv, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)          # independent reference
        var = sv ** 2 / (X.shape[0] - 1)
        _close("training variances = squared singular values / (n − 1)", w, var, 1e-12 * var[0], 1e-9, **ctx)
        scale = float((s.Zs ** 2).sum(1).max())
        for k in PROJ_DIMS:
            b, kc = out[(f"SYN|pc{k}", f)], {**ctx, "k": k}
            _close("projected training coordinates are uncorrelated, with the k largest training variances",
                   np.atleast_2d(np.cov(b.Zs[train].T)), np.diag(var[:k]), 1e-10 * var[0], **kc)
            _close("projection is centred on the TRAINING mean", b.Zs[train].mean(0), np.zeros(k), 1e-9, **kc)
            _close("same subspace as an SVD of the training rows (Gram matrix of every row)", _gram(b.Zs),
                   _gram((s.Zs - mu) @ Vt.T[:, :k]), 1e-9 * scale, **kc)
        full = out[("SYN|pc8", f)].Zs
        _close("k = full dimension reproduces every squared distance", _sq(full), _sq(s.Zs), 1e-9 * scale, k=8, **ctx)
        off = ~np.eye(full.shape[0], dtype=bool)                               # self-distances: see _sq
        _close("k = full dimension reproduces the distances between different attempts (rv.cdist)", rv.cdist(full, full)[off],
               rv.cdist(s.Zs, s.Zs)[off], 1e-8, k=8, **ctx)
    return "axes are orthonormal and ordered by training variance; scores are uncorrelated; k = 8 reproduces every squared distance"


def check_projection_determinism() -> str:
    m, Z = projection_embedding()
    emb = projection_frames(m, Z)
    first, again = _by_fold(_project(emb, "SYN", PROJ_DIMS)), _by_fold(_project(emb, "SYN", PROJ_DIMS))
    shuffled = _by_fold(_project(emb.sample(fraction=1.0, shuffle=True, seed=3), "SYN", PROJ_DIMS))
    Q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(8, 8)))
    rotated = _by_fold(_project(projection_frames(m, Z @ Q), "SYN", PROJ_DIMS))    # another fit: axes and signs may differ
    scale = float((Z ** 2).sum(1).max())
    for (method, f), b in first.items():
        ctx = {"method": method, "fold": f}
        _require(np.array_equal(again[(method, f)].seq_ids, b.seq_ids) and np.array_equal(shuffled[(method, f)].seq_ids, b.seq_ids),
                 "row order of the projected block", **ctx)
        _close("the same input gives the same coordinates", again[(method, f)].Zs, b.Zs, 1e-12, **ctx)
        _close("the row order of the input does not matter", shuffled[(method, f)].Zs, b.Zs, 1e-12, **ctx)
        _close("a rotated embedding gives the same projected geometry (Gram matrix; coordinates are not compared)",
               _gram(rotated[(method, f)].Zs), _gram(b.Zs), 1e-9 * scale, **ctx)
    return "repeated and row-shuffled inputs give identical coordinates; a rotated embedding gives the same Gram matrix for every k"


def check_projection_rank_deficient() -> str:
    rng = np.random.default_rng(12)
    m = synthetic_meta(150, seed=12)
    Zr = _per_attempt(rng.normal(size=(150, 3)) * np.array([3.0, 2.0, 1.0]), 0.05, rng) @ rng.normal(size=(3, 8))   # rank 3 in 8 dimensions
    emb = projection_frames(m, Zr)
    P = conform(_project(emb, "SYN", PROJ_DIMS, warns=rb.PROJ_OVER_RANK), rv.EMBEDDINGS)
    _require(np.isfinite(P["value"].to_numpy()).all() and np.isfinite(P["value_z"].to_numpy()).all(), "every projected value is finite")
    src, out = _by_fold(emb), _by_fold(P)
    for f in range(PROJ_FOLDS):
        s = src[("SYN", f)]
        train = ~s.held
        ctx = {"fold": f, "n_train_rows": int(train.sum()), "n_heldout_rows": int(s.held.sum())}
        _, V, w, rank = rb.training_axes(s.Zs[train])
        _require(rank == 3, "training rank of a rank-3 embedding", observed=rank, expected=3, variances=[f"{x:.3e}" for x in w], **ctx)
        _close("axes stay orthonormal when the embedding is rank-deficient", V.T @ V, np.eye(8), 1e-12, **ctx)
        scale = float((s.Zs ** 2).sum(1).max())
        for k in PROJ_DIMS:
            b = out[(f"SYN|pc{k}", f)]
            _require(b.Zs.shape == (s.Zs.shape[0], k), "a k above the rank keeps its k columns", observed=b.Zs.shape,
                     expected=(s.Zs.shape[0], k), k=k, **ctx)
        for k in (4, 8):                                                       # every row lies in the 3 training directions
            _close("k ≥ rank reproduces every squared distance", _sq(out[(f"SYN|pc{k}", f)].Zs), _sq(s.Zs), 1e-9 * scale, k=k, **ctx)
        _close("axes beyond the rank carry no training variance", out[("SYN|pc8", f)].Zs[train][:, 3:].var(0, ddof=1), np.zeros(5),
               1e-9 * w[0], k=8, **ctx)
        _close("pc2 holds the two largest training variances", out[("SYN|pc2", f)].Zs[train].var(0, ddof=1).sum(), w[:2].sum(),
               0.0, 1e-9, k=2, **ctx)
    rep = rb.projection_report(emb, "SYN", PROJ_DIMS)
    status = {k: sorted(rep.filter(pl.col("k") == k)["status"].unique().to_list()) for k in PROJ_DIMS}
    _require(status == {2: [rb.PROJ_OK], 4: [rb.PROJ_OVER_RANK], 8: [rb.PROJ_OVER_RANK]} and (rep["training_rank"] == 3).all(),
             "projection report of a rank-3 embedding", observed=status, ranks=rep["training_rank"].to_list())

    Z8 = rng.normal(size=(8, 8))
    def tiny(n_fit: int) -> pl.DataFrame:
        return rv.embedding_frame("TINY", 0, 0, m["seq"][:8], m["nfl"][:8], np.where(np.arange(8) < n_fit, "fit", "heldout"), "full", Z8, Z8)
    s = _by_fold(tiny(4))[("TINY", 0)]
    few = _by_fold(_project(tiny(4), "TINY", PROJ_DIMS, warns=rb.PROJ_OVER_RANK))
    rank = rb.training_axes(s.Zs[~s.held])[3]
    _require(rank == 3, "4 training rows in 8 dimensions have rank 3", observed=rank, expected=3, n_train_rows=int((~s.held).sum()))
    _require(sorted(few) == sorted((f"TINY|pc{k}", 0) for k in PROJ_DIMS) and all(np.isfinite(b.Zs).all() for b in few.values()),
             "fewer training rows than dimensions: every k ≤ 8 is still projected, finite", observed=sorted(few))
    _close("fewer training rows than dimensions: k = 8 reproduces every squared distance", _sq(few[("TINY|pc8", 0)].Zs), _sq(s.Zs),
           1e-9 * float((s.Zs ** 2).sum(1).max()), k=8, n_train_rows=4)
    _require(_project(tiny(2), "TINY", PROJ_DIMS, warns=rb.PROJ_FEW_ROWS).height == 0, "2 training rows: no projection, and it is reported")
    return "a rank-3 embedding and a 4-row training set are projected on 2, 4 and 8 finite axes, reported, with distances kept for k ≥ rank"


def check_projection_train_test_isolation() -> str:
    m, Z = projection_embedding()
    scale = float((Z ** 2).sum(1).max())
    def fit(Zx: np.ndarray, f: int, dims: Sequence[int], view: str = "full") -> dict[tuple[str, int], rv.EmbBlock]:
        return _by_fold(_project(projection_frames(m, Zx, (f,)), "SYN", dims), view)
    for f in range(PROJ_FOLDS):
        held = _fold_held(m, f)
        train = ~held
        ctx = {"fold": f, "n_train_rows": int(train.sum()), "n_heldout_rows": int(held.sum())}
        base = fit(Z, f, PROJ_DIMS)
        b = base[("SYN|pc4", f)]
        _require(np.array_equal(b.seq_ids, m["seq"]) and np.array_equal(b.held, held), "block rows follow the fixture rows", **ctx)
        Z2 = Z.copy()
        Z2[held] = Z2[held] * 5.0 - 20.0                                       # change the held-out athletes only
        moved = fit(Z2, f, (4,))[("SYN|pc4", f)]
        _close("training rows do not move when only held-out embeddings change (Gram matrix)", _gram(moved.Zs[train]),
               _gram(b.Zs[train]), 1e-9 * scale, k=4, **ctx)
        _close("the centre stays the training mean", moved.Zs[train].mean(0), np.zeros(4), 1e-9, k=4, **ctx)
        shift = float(np.linalg.norm(moved.Zs[held].mean(0)))
        _require(shift > 10.0, "held-out rows are not re-centred on themselves", observed_norm_of_heldout_mean=f"{shift:.3f}",
                 expected="> 10", k=4, **ctx)
        keep = held & (m["nfl"] == m["nfl"][held][0])
        Z3 = Z.copy()
        Z3[held & ~keep] = Z3[held & ~keep] * 3.0 + 1.0                        # change every OTHER held-out athlete
        alone = fit(Z3, f, (4,))[("SYN|pc4", f)]
        _close("a held-out athlete's projection does not depend on the other held-out athletes (Gram matrix with the training rows)",
               _gram(alone.Zs[train | keep]), _gram(b.Zs[train | keep]), 1e-9 * scale, k=4, **ctx)
        Z4 = Z.copy()
        Z4[train] = Z4[train] * (PROJ_SCALE[::-1] / PROJ_SCALE)                # reverse the spread of the TRAINING rows
        refit = fit(Z4, f, (2,))[("SYN|pc2", f)]
        change = float(np.abs(_gram(refit.Zs[held]) - _gram(base[("SYN|pc2", f)].Zs[held])).max())
        _require(change > 1e-3 * scale, "the check has power: other training embeddings give other axes for the same held-out rows",
                 observed_max_gram_change=f"{change:.3e}", expected=f"> {1e-3 * scale:.3e}", k=2, **ctx)
        head = fit(Z, f, (8,), "head_15")[("SYN|pc8", f)]
        _close("a second view is projected with the centre and axes of the full view's training rows", _sq(head.Zs, base[("SYN|pc8", f)].Zs[train]),
               _sq(Z[held] * 0.9, Z[train]), 1e-9 * scale, k=8, view="head_15", **ctx)
    return ("the centre and axes of a fold come from its training rows only: changing held-out embeddings leaves training rows and "
            "the other held-out athletes where they were; changing training embeddings moves the axes")


def check_scalar_embedding() -> str:
    m, Z = projection_embedding()
    values = pl.DataFrame({"seq_id": m["seq"], "nfl_id": m["nfl"], "a": Z[:, 0],
                           "b": [None if i == 5 else float(x) for i, x in enumerate(Z[:, 1])]})
    folds = pl.DataFrame({"nfl_id": m["ids"], "repeat": np.zeros(150, dtype=np.int64), "outer_fold": np.arange(150) % 3})
    sc = conform(rb.scalar_embedding("S2", values, ["a", "b"], folds, [0]), rv.EMBEDDINGS)
    blocks = list(rv.emb_blocks(sc))
    _require(len(blocks) == 3, "one block per outer fold", observed=len(blocks), expected=3)
    for b in blocks:
        ctx = {"fold": b.fold, "n_train_rows": int((~b.held).sum()), "n_heldout_rows": int(b.held.sum())}
        _close("scalars are centred on the fold's training rows", b.Zs[~b.held].mean(0), np.zeros(2), 1e-9, **ctx)
        _close("scalars have unit SD among the fold's training rows", b.Zs[~b.held].std(0, ddof=1), np.ones(2), 1e-9, **ctx)
    n_held = sum(int(b.held.sum()) for b in blocks)
    _require(n_held == 299, "every attempt with both scalars is held out exactly once", observed=n_held, expected=299)
    return "scalars are standardised with the training rows of each fold; an attempt lacking a column is left out"


PROJECTION_CHECKS: tuple[Callable[[], str], ...] = (
    check_projection_dimensionality, check_projection_orthogonality, check_projection_determinism, check_projection_rank_deficient,
    check_projection_train_test_isolation, check_scalar_embedding)


def check_embedding_projection() -> str:
    """Every projection check as one row of the 04B table; ``run_all(PROJECTION_CHECKS)`` lists them one by one."""
    for fn in PROJECTION_CHECKS:
        fn()
    return ("projection axes come from training rows only; k = 2, 4, 8 keep their dimension, rank-deficient or not; k = full "
            "dimension reproduces every squared distance; scalars are standardised by fold")


def check_shortlist_rules() -> str:
    def diff(method: str, ref: str, lo: float) -> dict:
        return {"method": method, "reference": ref, "space": "standardised", "scope": "within_position", "subset": "role:confirm",
                "metric": "auc", "n_athletes": 200, "estimate": 0.8, "reference_estimate": 0.8, "difference": lo + 0.02, "diff_lower": lo,
                "diff_upper": lo + 0.04}
    rows = [diff("M|pc2", "M|pc8", -0.06), diff("M|pc4", "M|pc8", -0.01), diff("N|pc2", "N|pc8", -0.05), diff("N|pc4", "N|pc8", -0.015)]
    k, why = rb.compact_dimension(rv._table(rows, rv.PAIRED), ["M", "N"], (2, 4, 8), 0.02)
    assert k == 4 and "k = 4" in why
    rows[3] = diff("N|pc4", "N|pc8", -0.04)
    assert rb.compact_dimension(rv._table(rows, rv.PAIRED), ["M", "N"], (2, 4, 8), 0.02)[0] is None          # every seed must agree
    assert rb.compact_dimension(rv._table(rows[:2], rv.PAIRED), ["M", "N"], (2, 4, 8), 0.02)[0] is None      # a missing seed is not a pass

    def q(method: str, **kw) -> dict:
        return {"method": method, "label": method, "kind": "primary", "embedding_dim": 8, "n_confirm": 250, "auc": 0.87, "auc_lower": 0.85,
                "auc_upper": 0.89, "matched_auc": 0.75, "matched_auc_lower": 0.70, "matched_auc_upper": 0.80, "window_r2": 0.10,
                "cohort_balanced_gain": 0.02, "collapsed": False, "forty_partial_r2": 0.9, "icc_trace": 0.92, **kw}
    good, ctrl = rv._table([q("A"), q("B")], rb.QUALITY), {"diff_lower": 0.05}
    assert rb.encoder_readiness(good, ["A", "B"], ctrl, CFG) == ("ready_for_transfer_test", [])
    state, failed = rb.encoder_readiness(rv._table([q("A"), q("B", collapsed=True)], rb.QUALITY), ["A", "B"], ctrl, CFG)
    assert state == "carry_with_caveat" and any("collapsed" in f for f in failed)
    assert rb.encoder_readiness(rv._table([q("A"), q("B", matched_auc_lower=0.45)], rb.QUALITY), ["A", "B"], ctrl, CFG)[0] == "not_ready"
    assert rb.encoder_readiness(good, ["A", "B"], {"diff_lower": -0.01}, CFG)[0] == "carry_with_caveat"
    assert rb.encoder_readiness(good, ["Z"], ctrl, CFG)[0] == "not_ready"
    e = engine()
    quality = rv._table([q("TCNB_A"), q("TCNB_A_s1", kind="seed replicate"), q("FPCA_A_K2", kind="baseline"),
                         q("TCNB_A|pc4", kind="projection", embedding_dim=4)], rb.QUALITY)
    seeds = rb.aggregate_seeds(rv._table([{"method": m, "training_seed": i, "metric": "auc", "value": 0.87, "lower": 0.85, "upper": 0.89}
                                          for i, m in enumerate(("TCNB_A", "TCNB_A_s1"))], rb.SEED_RUNS))
    short, rest = rb.build_shortlist(e["status"], e["tables"][at.CHARACTERISTICS.name], e["dictionary"], quality, seeds, "TCNB_A",
                                     ["TCNB_A", "TCNB_A_s1"], "FPCA_A_K2", (4, "k = 4: rule"), ctrl, CFG, "hash", "v-test", {"tcn": "p"})
    short, rest = conform(short, rb.SHORTLIST), conform(rest, rb.EXPLORATORY)
    ids = short["candidate_id"].to_list()
    assert len(ids) == 7 and len(set(ids)) == 7 and 5 <= short.filter(pl.col("selected")).height <= 8
    row = lambda c: fe._one(short, candidate_id=c)                                                            # noqa: E731
    assert row("C1_traditional_baseline")["selected"] and row("C1_traditional_baseline")["readiness"] == "reference"
    assert row("C2_speed_control")["selected"] and "speed" in row("C2_speed_control")["features"]
    assert row("C6_tcn_b_8d")["features"] == "TCNB_A" and row("C6_tcn_b_8d")["readiness"] == "ready_for_transfer_test"    # never 'the best seed'
    assert row("C7_tcn_compact")["features"] == "TCNB_A|pc4" and row("C5_fpca")["features"] == "FPCA_A_K2"
    assert "F2.forty.peak_a_tan" not in ";".join(short.filter(pl.col("selected"))["features"].to_list())     # a noise feature is not carried
    chosen = {f for x in short.filter(pl.col("selected"))["features"].to_list() for f in x.split(";")}
    assert not chosen & set(rest["feature_id"].to_list()) and (short["config_hash"] == "hash").all()
    empty, _ = rb.build_shortlist(e["status"], e["tables"][at.CHARACTERISTICS.name], e["dictionary"], quality.filter(pl.col("method") == "none"),
                                  seeds, "TCNB_A", ["TCNB_A"], "FPCA_A_K2", (None, "no k"), {}, CFG, "hash", "v-test", {})
    assert not fe._one(empty, candidate_id="C6_tcn_b_8d")["selected"] and not fe._one(empty, candidate_id="C7_tcn_compact")["selected"]
    assert "nfl" not in " ".join(inspect.signature(rb.build_shortlist).parameters).lower()
    return f"{short.filter(pl.col('selected')).height} of 7 slots filled by rule on the fixture; an unmet rule leaves its slot empty and documented"


def check_artifact_integrity() -> str:
    c = TableContract("integrity_table", "synthetic", {"k": pl.Int64(), "v": pl.Float64()}, key=("k",), area="row")
    frame = pl.DataFrame({"k": [1, 2], "v": [0.5, 1.5]})
    with tempfile.TemporaryDirectory() as tmp:
        cfg = base._fake_cfg(Path(tmp))
        smoke, full = AtlasStore(cfg, NotebookControls(run_mode="smoke")), AtlasStore(cfg, NotebookControls(run_mode="full"))
        up = AdvStore(cfg, NotebookControls(run_mode="full"))
        for d in (full.tables_dir, full.figures_dir, full.data_dir, full.models_dir):
            assert d.parts[-1] == "atlas" and "_sample" not in d.parts
        assert all("_sample" in d.parts and d.parts[-1] == "atlas" for d in (smoke.tables_dir, smoke.data_dir, smoke.models_dir))
        assert full.upstream.data_dir == up.data_dir and full.manifest_path != up.manifest_path and full.models_dir != up.models_dir
        up.build("UP", [c], lambda: {c.name: frame}, params={"a": 1})
        before = up.parquet(c).read_bytes()
        assert full.upstream.exists(c) and not full.exists(c)                    # read through upstream, not as an atlas output
        full.build("ATLAS", [c], lambda: {c.name: frame.with_columns(pl.col("v") * 10)}, params={"a": 1})
        assert up.parquet(c).read_bytes() == before and full.parquet(c).is_file() and full.parquet(c) != up.parquet(c)
        assert full.read(c)["v"].to_list() == [5.0, 15.0] and up.read(c)["v"].to_list() == [0.5, 1.5]      # nothing upstream is replaced
        assert not smoke.exists(c) and _raises(Exception, lambda: smoke.require(c, "04a"))
        assert "ATLAS" in full._manifest()["runs"] and "ATLAS" not in up._manifest()["runs"]

        def short(features: str) -> pl.DataFrame:
            return rv._table([{"candidate_id": "C1", "group": "g", "role": "reference", "selected": True, "readiness": "reference",
                               "features": features, "config_hash": "h", "shortlist_version": "v1", "frozen_at": "t0"}], rb.SHORTLIST)
        rest = rb.EXPLORATORY.empty()
        first, state = rb.freeze_shortlist(full, short("a"), rest, {"x": 1})
        assert state.startswith("frozen now") and first["features"][0] == "a"
        kept, state = rb.freeze_shortlist(full, short("b"), rest, {"x": 1})
        assert kept["features"][0] == "a" and "DIFFERENT" in state                # immutable: reported, not applied
        same, state = rb.freeze_shortlist(full, short("a").with_columns(pl.lit("t1").alias("frozen_at")), rest, {"x": 1})
        assert same["frozen_at"][0] == "t0" and "reproduces" in state
        forced, state = rb.freeze_shortlist(full, short("b"), rest, {"x": 1}, refreeze=True)
        assert forced["features"][0] == "b" and "REFREEZE" in state and full.read(rb.SHORTLIST)["features"][0] == "b"
        assert not smoke.exists(rb.SHORTLIST) and (full.data_dir / "transfer_candidates_v1_freeze.json").is_file()
        dry = AtlasStore(cfg, NotebookControls(run_mode="smoke", save_tables=False))
        assert "NOT WRITTEN" in rb.freeze_shortlist(dry, short("c"), rest, {})[1] and not dry.exists(rb.SHORTLIST)
    names = [rv.named(x, "robust").name for x in (rv.METHODS, *rv.EVAL_CONTRACTS, *rv.GATE_CONTRACTS)]
    assert len(set(names)) == len(names) and all(n.startswith("robust_") for n in names)
    tables = {x.name: x.empty() for x in rv.EVAL_CONTRACTS}
    assert rb.from_prefixed("robust", rb.to_prefixed("robust", tables, rv.EVAL_CONTRACTS), rv.EVAL_CONTRACTS).keys() == tables.keys()
    return "atlas outputs live in their own area (smoke under _sample); upstream files are byte-identical after a build; the shortlist is written once"


def check_no_nfl_label_access() -> str:
    here = Path(__file__).parent
    hits = {name: [t for t in FORBIDDEN if t in (here / name).read_text(encoding="utf-8")] for name in SCANNED}
    assert not any(hits.values()), hits                                         # no NFL exposure, endpoint or target identifier
    e = engine()
    allowed = {at.MART_TABLE, at.ATTEMPT_TABLE, at.WINDOW_TABLE, at.SUSTAINED_TABLE, at.GATE_TABLE, at.VECTOR_TABLE, at.REGISTRY_TABLE}
    assert set(e["dictionary"]["source_table"].unique().to_list()) <= allowed
    assert not e["dictionary"]["feature_id"].str.contains("nfl", literal=True).any()
    rich = e["athletes"].with_columns(pl.lit(999).alias("nfl_reg_plays"), pl.lit(5).alias("getoff_obs"))
    a = at.athlete_data(rich, at.official_table(e["mart"], rich), e["dictionary"])
    assert set(a.official) == {FORTY, TEN, WEIGHT, HEIGHT} and all(k.startswith("F1.official.") for k in a.values)
    assert set(at.official_table(e["mart"].with_columns(pl.lit(1.0).alias("nfl_outcome")), rich).columns) == {"nfl_id", *[s[0] for s in at.OFFICIAL_SPECS]}
    for fn in (at.screen_feature, at.screening_table, at.characterise_all, rb.build_shortlist, rb.encoder_readiness, rb.compact_dimension,
               rb.matched_retrieval, rb.matching_scales):
        params = " ".join(inspect.signature(fn).parameters).lower()
        assert not any(w in params for w in ("nfl", "outcome", "target", "label")), fn.__name__
    assert set(at.STATUS.columns) & {"nfl", "outcome"} == set()
    return f"{len(SCANNED)} module files hold no NFL exposure or endpoint identifier; screening and shortlist functions take Combine evidence only"


ATLAS_CHECKS: tuple[Callable[[], str], ...] = (
    check_unique_feature_dictionary_keys, check_aggregation_grain, check_known_icc_recovery, check_athlete_level_bootstrap,
    check_position_adjusted_statistics, check_sustained_features_use_measured_intervals_only, check_screening_rules,
    check_engine_end_to_end, check_artifact_integrity, check_no_nfl_label_access)
ROBUST_CHECKS: tuple[Callable[[], str], ...] = (
    check_fold_safe_residualisation, check_cohort_probe_balance, check_retrieval_matching, check_seed_aggregation,
    check_effective_rank_calculation, check_embedding_projection, check_athlete_level_bootstrap, check_shortlist_rules,
    check_artifact_integrity, check_no_nfl_label_access)
ALL_CHECKS = tuple(dict.fromkeys((*ATLAS_CHECKS, *ROBUST_CHECKS, *PROJECTION_CHECKS)))


def run_all(checks: tuple[Callable[[], str], ...] = ALL_CHECKS) -> pl.DataFrame:
    """Run every check in-process. A failing check is reported, not raised, so one table shows them all."""
    return base.run_all(checks)


assert_all_passed = base.assert_all_passed
