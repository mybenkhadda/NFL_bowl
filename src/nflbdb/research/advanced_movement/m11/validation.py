"""Milestone 11 notebook-callable checks — SYNTHETIC FIXTURES ONLY (software verification; never a scientific finding).

    from nflbdb.research.advanced_movement.m11 import validation as v11
    table = v11.run_all(v11.CHECKS_CORE)          # each notebook runs its own group (or ALL_CHECKS)
    v11.assert_all_passed(table)

Each check returns a one-line description or raises. Tolerances were set by reasoning (not tuned on data); checks marked
[draw] depend on a fixed seed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from .. import movement_demands as md
from .. import movement_gap as mg
from ..validation import assert_all_passed, failure_location
from . import checkpoint as ck
from . import combine_drills as cd
from . import development as dv
from . import drill_selection as ds
from . import evaluation as ev
from . import nfl_physics as nph
from . import physics as ph
from . import segmentation as sg
from . import tournament as tn
from . import twins_transfer as tw
from .config import M11Config
from .sources import drill_family

C = M11Config(n_boot=200, n_perm=10, n_random_projections=5)


def _close(a: float, b: float, tol: float, what: str) -> None:
    if not (np.isfinite(a) and abs(a - b) <= tol):
        raise AssertionError(f"{what}: {a} vs expected {b} (tol {tol})")


# ============================================================================== physics
def check_physical_units_and_speed_bands() -> str:
    t = np.arange(0, 4, 0.1)
    v = 3.0 * t
    m = ph.band_metrics(t, v, None, 2.0, 4.0)
    _close(m["mean_accel"], 3.0, 1e-9, "band acceleration of a 3 yd/s² ramp")
    _close(m["time_s"], 2 / 3, 1e-9, "time through 2→4 yd/s")
    _close(ph.crossing_time(t, v, 5.0), md.crossing_time(t, v, 5.0), 1e-12, "same first-crossing rule as the 09A atlas")
    return "band time, band acceleration (yd/s²) exact on a ramp; first-crossing rule identical to movement_demands"


def check_sensor_sampling_irregular_clock() -> str:
    rng = np.random.default_rng(1)
    t = np.cumsum(np.r_[0, 0.1 + rng.uniform(-0.01, 0.01, 59)])
    a = ph.central_accel(t, 2 * t ** 2)
    _close(float(np.max(np.abs(a[1:-1] - 4 * t[1:-1]))), 0.0, 0.05, "dv/dt on an irregular 10 Hz clock")
    return "acceleration uses the recorded clock: exact enough on an irregular sampling grid"


def check_censoring() -> str:
    t = np.arange(0, 3, 0.1)
    m = ph.band_metrics(t, 5.0 + t, None, 2.0, 4.0)
    assert m["start_censored"] and not np.isfinite(m["mean_accel"]), "a band entered before the recording must give no value"
    m2 = ph.band_metrics(t, 1.0 + t, None, 6.0, 8.0)
    assert m2["end_censored"] and not np.isfinite(m2["time_s"]), "a band never completed must give no value"
    f = cd.forty_features(t, 1.5 + 2 * t, np.full(t.size, 2.0), C)
    assert f["onset_censored"] == 1.0 and not np.isfinite(f["early_accel"]), "onset outside the recording → onset-based features absent"
    return "start / end censoring yields NaN with explicit flags; nothing is extrapolated"


def check_sprint_model_recovery() -> str:
    rng = np.random.default_rng(2)
    t = np.arange(0, 5, 0.1)
    v = ph.sprint_speed(t, 9.5, 1.1, 0.3) + rng.normal(0, 0.02, t.size)
    v[t < 0.3] = rng.uniform(0, 0.05, int((t < 0.3).sum()))
    f = ph.fit_sprint_time(t, v, C.sprint_tau_grid, C.sprint_t0_grid, fit_offset=True)
    _close(f["vmax"], 9.5, 0.15, "Vmax [draw]")
    _close(f["tau"], 1.1, 0.08, "τ [draw]")
    a = ph.central_accel(t, v)
    p = ph.fit_sprint_phase(v, a)
    _close(p["A0"] / (9.5 / 1.1), 1.0, 0.15, "phase-plane A0 relative to Vmax/τ [draw]")
    _close(ph.time_to_distance(40.0, 9.5, 1.1), 40.0 / 9.5 + 1.1, 0.02, "model time to 40 yd")
    return "time-form and offset-free phase-plane sprint fits recover Vmax, τ, A0 = Vmax/τ"


def check_envelope_recovery() -> str:
    rng = np.random.default_rng(3)
    A0, V0 = 5.0, 10.0
    v = rng.uniform(0.5, 9.0, 200_000)
    a = rng.uniform(-1.0, 1.0, v.size) * 0 + rng.uniform(0, 1, v.size) * (A0 * (1 - v / V0) + 1.0) - 1.0
    se = np.arange(0, 11.5 + 0.25, 0.5)
    ae = np.arange(-8, 8.125, 0.25)
    H = np.histogram2d(v, a, bins=[se, ae])[0]
    s = ph.envelope_summary(H, se, ae, 0.95, 40, 5, 1.0)
    expect_v0 = (0.95 * (A0 + 1) - 1) / (0.95 * A0 / V0)
    _close(s["lin_V0"] / expect_v0, 1.0, 0.05, "linear envelope V0 [draw]")
    _close(s["hub_V0"] / expect_v0, 1.0, 0.06, "Huber envelope V0 [draw]")
    iso = ph.isotonic_decreasing(np.array([3.0, 3.2, 2.5, 2.6, 1.0]))
    assert np.all(np.diff(iso) <= 1e-12), "nonparametric envelope must be non-increasing"
    return "upper-quantile envelope estimators recover a planted linear a(v); PAVA curve is monotone"


# ============================================================================== segmentation and geometry
def _shuttle(dt: float = 0.1):
    t = np.arange(0, 4.0, dt)
    v = np.interp(t, [0, 1.0, 1.5, 1.8, 2.8, 4.0], [0.2, 6.0, 0.3, 0.3, 5.5, 6.0])
    sign = np.where(t < 1.65, 1.0, -1.0)
    vx, vy = sign * v, np.zeros_like(v) + 0.01
    return t, v, ph.central_accel(t, v), vx, vy, ph.central_accel(t, vx), ph.central_accel(t, vy)


def check_episode_segmentation() -> str:
    t, v, a, vx, vy, ax, ay = _shuttle()
    tr = sg.transitions(t, v, a, vx, vy, ax, ay)
    assert len(tr) == 1, f"one transition expected, got {len(tr)}"
    d = tr[0]
    _close(abs(d["turn_angle_deg"]), 180.0, 15.0, "reversal angle through a stop")
    assert d["decel_mean"] > 5.0 and np.isfinite(d["brake_to_burst_s"]) and np.isfinite(d["speed_recovery_time_s"]), "transition metrics missing"
    _close(d["transition_cost_index"], (d["entry_speed"] - d["exit_speed"]) / d["entry_speed"], 1e-12, "TCI definition")
    lab = sg.threshold_labels(a)
    assert sg.agreement(lab, sg.changepoint_labels(a)) > 0.6 and sg.agreement(lab, sg.hmm_labels(a)) > 0.6, "segmenters disagree on a clean trace"
    return "threshold segmenter finds the reversal; changepoint and HMM agree on states; TCI follows its definition"


def check_coordinate_geometry_turn_direction() -> str:
    th = np.linspace(0, np.pi / 2, 30)
    vx, vy = -np.sin(th) * 5, np.cos(th) * 5                     # moving +y, turning left (counter-clockwise) to −x
    _close(ph.turn_angle(vx, vy, slice(0, 3), slice(27, 30)), 90.0, 8.0, "left turn")
    _close(ph.turn_angle(-vx, vy, slice(0, 3), slice(27, 30)), -90.0, 8.0, "right turn (mirror)")
    r, w = 5.0, 1.0
    tt = np.linspace(0, 2, 200)
    x, y = r * np.cos(w * tt), r * np.sin(w * tt)
    vx2, vy2, sp = ph.speed_from_xy(tt, x, y)
    an = ph.lateral_accel(vx2, vy2, np.gradient(vx2, tt), np.gradient(vy2, tt))
    _close(float(np.nanmedian(an)), r * w * w, 0.05, "centripetal acceleration v²/r on a ccw circle (sign +)")
    _close(float(np.nanmedian(np.abs(ph.tangential_accel(vx2, vy2, np.gradient(vx2, tt), np.gradient(vy2, tt))))), 0.0, 0.05,
           "tangential acceleration at constant speed")
    return "signed turn angles (+ = ccw), lateral acceleration v²/r and zero tangential acceleration on a circle"


# ============================================================================== linkage, folds, training-only operations
def check_player_linkage() -> str:
    assert drill_family("FORTY_YARD_DASH") == "forty" and drill_family("20_YARD_SHUTTLE") == "short_shuttle" and drill_family("60_YARD_SHUTTLE") == "long_shuttle"
    assert drill_family("THREE_CONE") == "three_cone" and drill_family("GAUNTLET") == "gauntlet" and drill_family("XYZ") == "other"
    assert drill_family("XYZ", {"XYZ": "pass_rush"}) == "pass_rush", "overrides win"
    twins = pl.DataFrame({"pair_id": ["1-2", "1-3"], "position_group": ["WR", "WR"], "nfl_id_a": [1, 1], "nfl_id_b": [2, 3], "d_forty": [0.01, 0.02],
                          "d_ten": [0.0, 0.01], "d_weight": [1.0, 2.0]})
    vals = pl.DataFrame({"nfl_id": [1, 2, 3], "c": [1.0, 2.0, None], "n": [1.0, 3.0, 2.0]})
    p = tw.pair_frame(twins, vals, "c", "n")
    assert p.height == 1 and p["d_combine"][0] == 1.0 and p["d_nfl"][0] == 2.0, "a pair needs both athletes linked with both values"
    return "drill names map to physical families by pattern (+ overrides); twin pairs need both athletes linked"


def check_athlete_grouped_folds() -> str:
    a = pl.DataFrame({"nfl_id": list(range(100)), "position_group": ["WR", "DB"] * 50, "draft_year": [2020 + i % 4 for i in range(100)]})
    m06 = pl.DataFrame({"nfl_id": list(range(40)), "outer_fold": [i % 5 for i in range(40)]})
    f = ev.assign_folds(a, m06, 5, 0)
    assert f.height == 100 and f["fold"].null_count() == 0
    assert f.filter(pl.col("nfl_id") < 40).join(m06, on="nfl_id")["fold"].to_list() == m06["outer_fold"].to_list(), "Milestone 06 folds must be kept"
    try:
        ev.assert_grouped(np.array([1, 1, 2]), np.array([0, 1, 0]))
    except ev.LeakageError:
        pass
    else:
        raise AssertionError("an athlete in two folds must raise")
    return "athlete folds: Milestone 06 folds reused, others stratified; an athlete split across folds raises"


def check_training_only_transformations() -> str:
    rng = np.random.default_rng(4)
    df = pl.DataFrame({"nfl_id": list(range(60)), "position_group": ["WR"] * 30 + ["DB"] * 30, "x": rng.normal(size=60)})
    train = np.arange(60) < 40
    d1 = ev.design(df, ["x"], train)
    df2 = df.with_columns(pl.when(pl.col("nfl_id") >= 40).then(pl.col("x") * 100).otherwise(pl.col("x")).alias("x"))
    d2 = ev.design(df2, ["x"], train)
    assert np.allclose(d1.X[train], d2.X[train]), "training rows must not depend on held-out values"
    X = rng.normal(size=(60, 5))
    X2 = X.copy()
    X2[~train] *= 50
    assert np.allclose(np.abs(ds.pca_scores(X, train, 3)[train]), np.abs(ds.pca_scores(X2, train, 3)[train])), "PCA must be fitted on training rows"
    return "scaling, imputation, dummies and family PCA use training athletes only"


# ============================================================================== outcomes, pairs, resampling
def check_outcome_construction() -> str:
    ep = pl.DataFrame({"game_id": [1, 1, 2, 2, 3], "play_id": [1, 1, 1, 2, 1], "nfl_id": [7] * 5, "episode": [1, 2, 1, 1, 1],
                       "demand_id": ["braking", "reaccel_after_brake", "braking", "accel_low", "braking"], "season": [2023] * 5, "position_group": ["WR"] * 5,
                       "t_start": [1.0, 1.9, 0.5, 0.2, 0.4], "duration_s": [0.5, 0.6, 0.5, 0.8, 0.4], "v_start": [6.0, 2.0, 5.0, 0.5, 6.0],
                       "v_end": [2.0, 5.0, 2.5, 4.0, 3.0], "v_max": [6.0, 5.0, 5.0, 4.0, 6.0], "mean_accel": [-8.0, 5.0, -5.0, 4.4, -7.5],
                       "speed_change": [-4.0, 3.0, -2.5, 3.5, -3.0], "distance_yd": [1.5, 2.0, 1.2, 1.5, 1.0], "peak_rate_05": [-9.0, 6.0, -6.0, 5.0, -8.0],
                       "heading_change_deg": [None] * 5, "acc_2_4": [None, 4.0, None, 4.5, None], "acc_4_6": [None] * 5, "acc_6_8": [None] * 5}
                      ).with_columns(pl.col("heading_change_deg", "acc_2_4", "acc_4_6", "acc_6_8").cast(pl.Float64))
    pairs = nph.brake_reaccel_pairs(ep)
    assert pairs.height == 1 and abs(pairs["latency_s"][0] - 0.4) < 1e-9, "one braking → re-acceleration pair with a 0.4 s gap"
    t = nph.athlete_targets(ep, pairs, None, M11Config(min_episodes_athlete=3), [nph.target("nfl.brake_decel")])
    assert t["value"][0] == 7.5 and t["n_episodes"][0] == 3 and t["supported"][0], "median of −mean_accel over 3 braking episodes"
    return "pairs and athlete-level targets follow their declared rules (median, exposure count, support line)"


def check_pair_overlap_resampling() -> str:
    a, b = np.array([1, 1, 2, 4]), np.array([2, 3, 3, 5])
    comp = tw.components(a, b)
    assert comp[0] == comp[1] == comp[2] != comp[3], "components of the pair graph"
    p = pl.DataFrame({"pair_id": ["a", "b", "c", "d"], "nfl_id_a": a, "nfl_id_b": b, "d_forty": [0.01, 0.0, 0.02, 0.01]})
    dj = tw.disjoint_matching(p)
    used = dj["nfl_id_a"].to_list() + dj["nfl_id_b"].to_list()
    assert len(used) == len(set(used)), "disjoint matching uses every athlete once"
    seen = []
    ev.cluster_bootstrap_pairs(a, b, lambda w: seen.append(w.copy()) or 0.0, 30, 0)
    assert all(w.min() >= 0 for w in seen) and any((w == 0).any() for w in seen), "pairs of unsampled athletes get weight 0"
    return "pair resampling is by athlete (multiplicity weights), components and disjoint matching are correct"


def check_resampling_and_comparison() -> str:
    rng = np.random.default_rng(5)
    y = rng.normal(size=300)
    same = ev.paired_compare(y, y * 0.5, y * 0.5, 300, 0)
    assert same["delta_rmse"] == 0 and same["delta_lo"] <= 0 <= same["delta_hi"], "identical models → Δ 0 inside its interval"
    good = ev.paired_compare(y, np.zeros_like(y), y + rng.normal(0, 0.3, y.size), 300, 0)
    assert good["delta_lo"] > 0 and ev.verdict(good) == "model_better", "a clearly better model must be detected [draw]"
    x = rng.normal(size=200)
    df = pl.DataFrame({"nfl_id": list(range(200)), "position_group": ["WR", "DB"] * 100, "draft_year": [2021] * 200, "height": rng.normal(size=200),
                       "weight": rng.normal(size=200), "forty": rng.normal(size=200), "ten": rng.normal(size=200), "trk": x, "noise": rng.normal(size=200),
                       "y": 0.8 * x + rng.normal(0, 0.3, 200)})
    folds = ev.make_folds(df["nfl_id"].to_numpy(), df["position_group"].to_numpy(), 5, 0)
    rows, _ = ev.run_ladder(df, "y", {"M1": ["draft_year", "height", "weight", "forty", "ten"], "M2": ["draft_year", "height", "weight", "forty", "ten", "trk"]},
                            folds, C, 0, comparisons=(("M2", "M1"),))
    assert rows[0]["verdict"] == "model_better", "planted tracking signal must be found held-out [draw]"
    rows0, _ = ev.run_ladder(df, "y", {"M1": ["draft_year", "height", "weight", "forty", "ten"], "M2": ["draft_year", "height", "weight", "forty", "ten", "noise"]},
                             folds, C, 0, comparisons=(("M2", "M1"),))
    assert rows0[0]["verdict"] != "model_better", "pure-noise feature must not look better [draw]"
    return "paired athlete bootstrap: null → interval covers 0; planted signal detected held-out; noise feature not promoted"


def check_mixed_model_and_twins() -> str:
    rng = np.random.default_rng(6)
    ids = np.repeat(np.arange(150), 4)
    t = np.tile(np.arange(4.0), 150)
    u1 = rng.normal(0, 0.3, 150)
    y = 1.0 + 0.1 * t + rng.normal(0, 0.5, 150)[ids] + u1[ids] * t + rng.normal(0, 0.1, ids.size)
    f = dv.lmm_em(y, t, ids)
    _close(f["sd_slope"], 0.3, 0.08, "random-slope SD [draw]")
    _close(f["slope"], 0.1, 0.08, "fixed slope [draw]")
    a, b = rng.integers(0, 80, 400), rng.integers(80, 160, 400)
    dc = rng.normal(size=400)
    p = pl.DataFrame({"pair_id": [str(i) for i in range(400)], "nfl_id_a": a, "nfl_id_b": b, "d_forty": np.zeros(400), "d_ten": np.zeros(400),
                      "d_combine": dc, "d_nfl": dc + rng.normal(0, 0.3, 400)})
    r = tw.analyse(p, 1, M11Config(n_boot=100), 0)
    assert r["concordance"] > 0.8 and r["concordance_lo"] > 0.5, "planted concordance must be detected with cluster inference [draw]"
    return "EM mixed model recovers slope heterogeneity; twin concordance with athlete-cluster intervals"


# ============================================================================== checkpoints and artifacts
def check_checkpoint_compatibility() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        k1 = ck.ShardCheckpoint(root, "sim", {"a": 1}, "smoke")
        k1.save(("s", 1), pl.DataFrame({"x": [1]}))
        p = k1.save(("s", 1), pl.DataFrame({"x": [999]}))
        assert pl.read_parquet(p)["x"][0] == 1, "an existing shard is never overwritten"
        assert ck.ShardCheckpoint(root, "sim", {"a": 2}, "smoke").dir != k1.dir, "another configuration gets another directory"
        try:
            k1.mark_complete([("s", 1), ("s", 2)])
        except ck.IncompleteRun:
            pass
        else:
            raise AssertionError("a partial set must not be marked complete")
        k1.save(("s", 2), {"y": 2})
        k1.mark_complete([("s", 1), ("s", 2)])
        assert k1.is_complete() and k1.status([("s", 1), ("s", 2)])["state"] == "complete"
        p.write_bytes(p.read_bytes() + b"x")
        assert not k1.has(("s", 1)), "a tampered shard fails verification and is recomputed"
    return "immutable shards, config-hashed directories, verified sidecars, completion gate, tamper detection"


def check_saved_artifact_integrity() -> str:
    e = tn.evidence_frame([tn.evidence_row("forty.phase_A0", "11B", n_athletes=200, icc_lo=0.3, stopwatch_r2=0.2, delta_lo=0.1, beats_controls=True)])
    assert e.schema == pl.Schema(tn.EVIDENCE_SCHEMA), "evidence schema"
    t = tn.tournament(e, M11Config())
    r = t.filter(pl.col("metric_id") == "forty.phase_A0").row(0, named=True)
    assert r["L2_repeatable"] == "fail" and r["L6_predictively_validated"] == "pass" and r["evidence_tier"] == 1, "no automatic promotion past a failed level"
    assert t.filter(pl.col("metric_id") == "geom.adaptability_slope")["evidence_tier"][0] == 0, "a candidate without evidence stays unsupported"
    reg = {"demand_id": "braking", "combine_analogue": "none", "comparability": "none"}
    assert mg.coverage_class(reg, {"supported": True, "n_episodes": 99}, 0)[0] == mg.NOT_MEASURED, "Movement Gap 10.2: braking is not measured by the forty"
    assert mg.coverage_class({"demand_id": "accel_low", "combine_analogue": "acc_2_4, acc_4_6"}, {"supported": True}, 5)[0] == mg.DIRECT
    return "evidence schema, level rules without promotion, corrected Movement Gap classes"


CHECKS_CORE: tuple[Callable[[], str], ...] = (check_physical_units_and_speed_bands, check_sensor_sampling_irregular_clock, check_censoring,
                                              check_player_linkage, check_athlete_grouped_folds, check_training_only_transformations,
                                              check_checkpoint_compatibility, check_saved_artifact_integrity)
CHECKS_PHYSICS: tuple[Callable[[], str], ...] = (check_sprint_model_recovery, check_envelope_recovery, check_outcome_construction)
CHECKS_TRANSITIONS: tuple[Callable[[], str], ...] = (check_episode_segmentation, check_coordinate_geometry_turn_direction)
CHECKS_STATS: tuple[Callable[[], str], ...] = (check_resampling_and_comparison, check_pair_overlap_resampling, check_mixed_model_and_twins)
ALL_CHECKS: tuple[Callable[[], str], ...] = (*CHECKS_CORE, *CHECKS_PHYSICS, *CHECKS_TRANSITIONS, *CHECKS_STATS)


def run_all(checks: tuple[Callable[[], str], ...] = ALL_CHECKS) -> pl.DataFrame:
    rows = []
    for fn in checks:
        try:
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "PASS", "detail": fn()})
        except Exception as e:                                   # noqa: BLE001 — the table lists every failure
            rows.append({"check": fn.__name__.removeprefix("check_"), "status": "FAIL", "detail": f"{type(e).__name__}: {e}{failure_location(e)}"})
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String})


__all__ = ["ALL_CHECKS", "CHECKS_CORE", "CHECKS_PHYSICS", "CHECKS_STATS", "CHECKS_TRANSITIONS", "assert_all_passed", "run_all"]
