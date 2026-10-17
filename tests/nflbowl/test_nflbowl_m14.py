"""Milestone 14 unit tests (synthetic signals; no bundle, no GPU). Authored statically — not executed in the authoring
session. Run: ``pytest tests/nflbowl/test_nflbowl_m14.py -q``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from nflbowl.lab import reliability as REL  # noqa: E402
from nflbowl.lab.m14 import config as M14C  # noqa: E402
from nflbowl.lab.m14 import events as EV  # noqa: E402
from nflbowl.lab.m14 import metrics as MT  # noqa: E402
from nflbowl.lab.m14 import novelty as NV  # noqa: E402
from nflbowl.lab.m14 import registry as RG  # noqa: E402
from nflbowl.lab.m14 import reliability as MR  # noqa: E402
from nflbowl.lab.m14 import reporting as RP  # noqa: E402
from nflbowl.lab.m14 import scouting as SC  # noqa: E402
from nflbowl.lab.m14 import transfer as TR  # noqa: E402

P = EV.EventParams()


# ====================================================================== synthetic signals
def shuttle(n: int = 40) -> np.ndarray:
    """1 → 6 yd/s (frames 0–10), hold to 14, brake to 0.5 at frame 20, re-accelerate to 6 at 28, hold."""
    v = np.empty(n)
    for k in range(n):
        if k <= 10:
            v[k] = 1 + 5 * k / 10
        elif k <= 14:
            v[k] = 6.0
        elif k <= 20:
            v[k] = 6 - 5.5 * (k - 14) / 6
        elif k <= 28:
            v[k] = 0.5 + 5.5 * (k - 20) / 8
        else:
            v[k] = 6.0
    return v


def arrays(v: np.ndarray, valid: np.ndarray | None = None, dt: np.ndarray | None = None, an: np.ndarray | None = None) -> dict[str, np.ndarray]:
    n = v.size
    valid = np.ones(n, bool) if valid is None else valid
    dt = np.full(n, 0.1) if dt is None else dt
    at = np.gradient(v, 0.1)
    yaw = np.zeros(n)
    yaw[14:28] = 1.5                                                   # a turn during the cut
    return {"speed": np.where(valid, v, np.nan), "a_tan": np.where(valid, at, np.nan), "a_n": np.where(valid, np.zeros(n) if an is None else an, np.nan),
            "yaw": np.where(valid, yaw, np.nan), "dt": np.where(valid, dt, np.nan), "heading_valid": np.where(valid, 1.0, np.nan), "valid": valid}


def scaled(v: np.ndarray) -> np.ndarray:
    """(L, 6) bundle-scaled channels of a speed profile (inverse of events.SC)."""
    a = arrays(v)
    x = np.zeros((v.size, 6), np.float32)
    x[:, 0] = a["speed"] / EV.SC["speed"]
    x[:, 1] = a["a_tan"] / EV.SC["a_tan"]
    x[:, 3] = a["yaw"] / EV.SC["yaw"]
    x[:, 4] = 1.0
    x[:, 5] = 1.0
    return x


# ====================================================================== units / masking
def test_acceleration_units_and_scaling():
    x = np.zeros((1, 3, 6))
    x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4] = 0.5, 0.2, -0.4, 1.0, 1.0
    a = EV.physical(x, np.ones((1, 3), bool))
    assert np.allclose(a["speed"], 5.0) and np.allclose(a["a_tan"], 1.0) and np.allclose(a["a_n"], -2.0)
    assert np.allclose(a["yaw"], 3.0) and np.allclose(a["dt"], 0.1)


def test_masked_frames_are_nan_and_invalid_dt_is_nominal():
    x = np.zeros((1, 4, 6))
    x[..., 0] = 0.3
    x[0, 2, 4] = -1.0                                                   # non-positive dt on a valid frame
    m = np.array([[True, False, True, True]])
    a = EV.physical(x, m)
    assert np.isnan(a["speed"][0, 1]) and not a["valid"][0, 1]
    assert a["dt"][0, 2] == pytest.approx(EV.NOMINAL_DT)


def test_segments_break_on_mask_and_dt_gap():
    valid = np.ones(30, bool)
    valid[10] = False
    dt = np.full(30, 0.1)
    dt[20] = 0.4
    segs = EV.segments(valid, dt, 0.15, 5)
    assert segs == [(0, 10), (11, 21), (21, 30)]


# ====================================================================== event timing
def test_event_phases_on_synthetic_shuttle():
    a = arrays(shuttle())
    evs, q, segs = EV.detect_sequence(a, P)
    assert len(evs) == 1 and q["n_events"] == 1 and segs == [(0, 40)]
    e = evs[0]
    assert e["i_min"] == 20 and e["i_entry"] == 14 and e["i_brake"] == 14
    assert e["i_entry"] <= e["i_brake"] <= e["i_min"] < e["i_prop"] <= e["i_exit"]
    assert e["complete"] and not e["exit_censored"] and not e["propulsion_censored"]
    assert e["turn_speed_retention"] == pytest.approx(0.5 / 6)
    assert e["corner_exit_efficiency"] == pytest.approx((0.5 + 5.5 * 5 / 8) / 6, rel=1e-6)
    assert e["braking_duration"] == pytest.approx(0.6)
    assert e["v_entry"] == pytest.approx(6.0) and e["v_min"] == pytest.approx(0.5)
    assert 0 < e["braking_budget_utilization"] <= 1
    assert e["cut_recovery_time"] > 0 and e["turn_dir"] == 1
    assert len(e["trace_speed"]) == 2 * P.trace_half_window + 1
    assert e["trace_speed"][P.trace_half_window] == pytest.approx(0.5)


def test_event_never_crosses_a_mask_gap():
    valid = np.ones(40, bool)
    valid[17] = False                                                   # inside the braking phase
    evs, q, segs = EV.detect_sequence(arrays(shuttle(), valid=valid), P)
    assert len(segs) == 2 and q["n_internal_invalid"] == 1
    for e in evs:
        assert e["seg_start"] <= e["i_entry"] and e["i_exit"] < e["seg_end"]
    assert all(not (e["i_entry"] < 17 < e["i_exit"]) for e in evs)


def test_double_dip_is_merged_into_the_deeper_valley():
    v = shuttle(50).copy()
    v[16:19] = [4.0, 3.6, 4.2]                                          # valley at 17 (drop 2.4) closed by a rise of 0.6 < rise_min
    assert 17 in EV.valleys(v)
    evs, _, _ = EV.detect_sequence(arrays(v), P)
    assert len(evs) == 1 and evs[0]["i_min"] == 20 and evs[0]["i_entry"] == 14


def test_constant_speed_has_no_event_and_safe_denominators():
    a = arrays(np.full(30, 5.0))
    evs, q, segs = EV.detect_sequence(a, P)
    assert evs == [] and q["n_events"] == 0
    assert np.isnan(MT.sparc(np.full(8, 5.0), 10.0))                     # < 10 frames
    assert np.isnan(MT.sparc(np.zeros(30), 10.0))                       # zero spectrum
    assert np.isnan(MT.sparc(np.full(30, 5.0), 0.0))                    # invalid sampling rate
    sp = MT.sprint_profile(np.full(30, 5.0), np.zeros(30), np.full(30, 0.1), P)
    assert np.isnan(sp["accel_envelope_a0"]) and np.isnan(sp["time_to_useful_speed"])


def test_sparc_ranks_a_smooth_profile_above_an_oscillating_one():
    t = np.linspace(0, 1, 40)
    bell = 6 * np.sin(np.pi * t)
    wobbly = bell + 0.8 * np.sin(2 * np.pi * 4 * t)
    s, w = MT.sparc(bell, 10.0), MT.sparc(wobbly, 10.0)
    assert np.isfinite(s) and np.isfinite(w) and s <= 0 and s > w


def test_weak_braking_gives_undefined_braking_metrics():
    weak = EV.EventParams(brake_thresh=50.0)
    evs, _, _ = EV.detect_sequence(arrays(shuttle()), weak)
    assert len(evs) == 1 and evs[0]["weak_braking"]
    assert np.isnan(evs[0]["braking_point_efficiency"]) and np.isnan(evs[0]["max_controlled_decel"])
    assert not evs[0]["complete"]


# ====================================================================== curvature stability
def test_curvature_ignores_near_standstill_frames():
    v = shuttle()
    an = np.zeros(40)
    an[20] = 3.0                                                        # at 0.5 yd/s: |a_n| / v² = 12 would blow up
    an[25] = 2.0                                                        # at ≥ 2 yd/s
    evs, _, _ = EV.detect_sequence(arrays(v, an=an), P)
    k = evs[0]["max_curvature"]
    assert np.isfinite(k) and k == pytest.approx(2.0 / v[25] ** 2)
    env = MT.curvature_speed_envelope(np.array([0.5, 0.5, 0.5, 3, 3, 3, 5, 5, 5]), np.array([9, 9, 9, 1, 1, 1, 2, 2, 2]))
    assert env["curvature_speed_envelope"] == pytest.approx(1.5)


def test_sprint_profile_recovers_linear_av_profile():
    v = [0.5]
    for _ in range(40):
        v.append(v[-1] + (8.0 * (1 - v[-1] / 10.0)) * 0.1)
    v = np.array(v)
    at = 8.0 * (1 - v / 10.0)
    sp = MT.sprint_profile(v, at, np.full(v.size, 0.1), P)
    assert sp["accel_envelope_a0"] == pytest.approx(8.0, rel=1e-6) and sp["accel_envelope_v0"] == pytest.approx(10.0, rel=1e-6)
    assert sp["accel_reserve_at_vref"] == pytest.approx(8.0 * (1 - 7.0 / 10.0), rel=1e-6)
    assert sp["time_to_useful_speed"] > 0


# ====================================================================== extraction → tables → cache
def _bundle(n_ath: int = 6):
    L = 40
    rows, X, M = [], [], []
    for a in range(n_ath):
        for k, kind in enumerate(("three_cone", "short_shuttle", "forty")):
            X.append(scaled(shuttle(L) * (1 + 0.02 * a)))
            M.append(np.ones(L, bool))
            rows.append({"idx": len(rows), "nfl_id": 100 + a, "domain": "combine", "kind": kind, "position_group": "DB" if a % 2 else "WR"})
        X.append(scaled(shuttle(L)))
        M.append(np.ones(L, bool))
        rows.append({"idx": len(rows), "nfl_id": 100 + a, "domain": "nfl", "kind": "braking", "position_group": "DB" if a % 2 else "WR"})
    return np.stack(X), np.stack(M), pl.DataFrame(rows)


def test_extract_sequence_and_athlete_tables():
    X, M, meta = _bundle()
    ev, sq = MT.extract(X, M, meta, P, chunk=5)
    assert sq.height == X.shape[0] and ev.height == X.shape[0]
    st = MT.sequence_table(ev, sq, {100 + a: a % 5 for a in range(6)})
    assert "fold" in st.columns and st["turn_speed_retention"].null_count() == 0
    ath = MT.athlete_table(st, ev)
    assert ath.height == 6
    assert RG.combine_column("turn_speed_retention", "agility") in ath.columns
    assert RG.nfl_column("turn_speed_retention") in ath.columns
    preds = MT.combine_predictors(ath)
    assert preds and not any(c.startswith("m14n__") or "__ctx_" in c for c in preds)


def test_metric_persistence_and_cache_reload(tmp_path):
    X, M, meta = _bundle(3)
    exp = SimpleNamespace(bundle=SimpleNamespace(fingerprint="fp", n_sequences=X.shape[0], X=X, M=M, meta=meta))
    run = M14C.M14Run(tmp_path / "run", None)
    ev1, sq1, how1 = MT.cached_extract(run, exp, P)
    ev2, sq2, how2 = MT.cached_extract(run, exp, P)
    assert how1 == "computed from the bundle" and how2 == "reused cache"
    assert sq2.height == sq1.height and ev2.height == ev1.height
    _, _, how3 = MT.cached_extract(run, exp, EV.EventParams(drop_min=1.2))
    assert how3 == "computed from the bundle"                           # another parameter hash → no stale reuse


def test_run_rejects_unknown_family_and_writes_status(tmp_path):
    run = M14C.M14Run(tmp_path / "run", None)
    with pytest.raises(ValueError):
        run.table(pl.DataFrame({"a": [1]}), "x", "not_a_family")
    run.table(pl.DataFrame({"a": [1]}), "x", "registry")
    assert run.has("x", "registry")
    assert run.write_status().is_file()


def test_config_forbids_training_and_unknown_keys():
    with pytest.raises(KeyError):
        M14C.resolve({"run_training": True})
    with pytest.raises(KeyError):
        M14C.resolve({"nope": 1})
    with pytest.raises(ValueError):
        M14C.resolve({"reliability_k_grid": [1, 2]})
    c = M14C.resolve({"seed": 7})
    assert c.seed == 7 and not c.training_enabled()


# ====================================================================== reliability
def _trials(n_ath: int = 60, k: int = 4, between: float = 1.0, within: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_ath):
        mu = rng.normal(0, between)
        for j in range(k):
            rows.append({"nfl_id": a, "kind": "forty", "order": j, "value": mu + rng.normal(0, within)})
    return pl.DataFrame(rows)


def test_reliability_estimates_high_and_low():
    hi = MR.reliability_table({"m14c_forty__high_speed_accel": _trials()}, n_boot=50, seed=1, min_athletes=30)
    lo = MR.reliability_table({"m14c_forty__high_speed_accel": _trials(between=0.0, within=1.0)}, n_boot=50, seed=1, min_athletes=30)
    assert hi["icc1"][0] > 0.9 and abs(lo["icc1"][0]) < 0.3
    assert hi["split_sb"][0] > 0.9 and hi["icc_k"][0] >= hi["icc1"][0]
    curves = MR.matched_k_curves({"c": _trials()}, (2, 3, 4), repeats=5, seed=0, min_athletes=30)
    assert set(curves["k"].to_list()) == {2, 3, 4} and curves["n_athletes"].n_unique() == 1


def test_measurement_status_rules():
    cfg = M14C.resolve({})
    rel = pl.DataFrame({"column": ["m14c_forty__high_speed_accel", "m14c_forty__accel_envelope_a0"], "icc1": [0.8, 0.2], "icc1_lo": [0.7, 0.1],
                        "n_athletes": [100, 100]})
    rob = pl.DataFrame({"column": ["m14c_forty__high_speed_accel", "m14c_forty__accel_envelope_a0"], "min_spearman": [0.95, 0.95]})
    cov = pl.DataFrame({"column": ["m14c_forty__high_speed_accel", "m14c_forty__accel_envelope_a0"], "coverage_all": [0.9, 0.9], "coverage_transfer": [0.9, 0.9]})
    st = MR.measurement_status(rel, None, rob, cov, cfg)
    s = dict(zip(st["column"].to_list(), st["measurement_status"].to_list()))
    assert s["m14c_forty__high_speed_accel"] == "validated" and s["m14c_forty__accel_envelope_a0"] == "invalid"


# ====================================================================== train-only preprocessing / fold isolation
def _frame(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = np.arange(n)
    d = {"nfl_id": ids, "fold": ids % 5, "position_group": np.where(ids % 2, "DB", "WR"), "nfl.q90_shared": rng.normal(size=n),
         "draft_year": np.full(n, 2020.0), "height": rng.normal(72, 2, n), "weight": rng.normal(210, 20, n), "forty": rng.normal(4.6, 0.1, n),
         "ten": rng.normal(1.6, 0.05, n), "three_cone": rng.normal(7.0, 0.2, n), "m14c_forty__high_speed_accel": rng.normal(size=n)}
    return pl.DataFrame(d)


def test_quality_selection_ignores_held_out_athletes():
    cfg = M14C.resolve({"min_reliability_athletes": 10})
    data = _frame()
    col = "m14c_forty__high_speed_accel"
    trials = _trials(100, 3).with_columns(pl.col("nfl_id").cast(pl.Int64))
    log1, log2 = [], []
    ids = data["nfl_id"].to_numpy()
    train = data["fold"].to_numpy() != 0
    b1 = TR.quality_factory(data, [col], {col: trials}, cfg, log1, "m")(ids)(train)
    # change every held-out athlete's metric, trials and target
    data2 = data.with_columns(pl.when(pl.col("fold") == 0).then(pl.col(col) * 100 + 7).otherwise(pl.col(col)).alias(col),
                              pl.when(pl.col("fold") == 0).then(99.0).otherwise(pl.col("nfl.q90_shared")).alias("nfl.q90_shared"))
    held = set(data.filter(pl.col("fold") == 0)["nfl_id"].to_list())
    trials2 = trials.with_columns(pl.when(pl.col("nfl_id").is_in(list(held))).then(pl.col("value") * -50).otherwise(pl.col("value")).alias("value"))
    TR.quality_factory(data2, [col], {col: trials2}, cfg, log2, "m")(ids)(train)
    assert log1[0]["selected"] == log2[0]["selected"]
    assert log1[0]["scores"][col]["train_icc"] == pytest.approx(log2[0]["scores"][col]["train_icc"])
    assert b1.height == ids.size


def test_train_residual_block_is_fit_on_training_rows_only():
    data = _frame()
    specs = NV.residual_specs(data.columns)
    assert specs and specs[0][0] == "cod_deficit_residual__three_cone"
    ids = data["nfl_id"].to_numpy()
    train = data["fold"].to_numpy() != 0
    log1, log2 = [], []
    r1 = NV.train_residual_factory(data, specs, log1)(ids)(train)
    data2 = data.with_columns(pl.when(pl.col("fold") == 0).then(pl.col("three_cone") + 5).otherwise(pl.col("three_cone")).alias("three_cone"))
    r2 = NV.train_residual_factory(data2, specs, log2)(ids)(train)
    assert np.allclose(log1[0]["coef"], log2[0]["coef"])
    tr_rows = np.flatnonzero(train)
    assert np.allclose(r1[specs[0][0]].to_numpy()[tr_rows], r2[specs[0][0]].to_numpy()[tr_rows])


def test_fold_isolation_rejects_mixed_held_out_rows():
    folds = np.array([0, 0, 1, 1, 2, 2])
    train = np.array([True, False, True, False, True, True])            # held-out rows from folds 0 and 1
    with pytest.raises(ValueError):
        REL._fold_of(train, folds)


def test_combined_blocks_keep_row_alignment():
    data = _frame(20)
    ids = data["nfl_id"].to_numpy()
    train = data["fold"].to_numpy() != 1

    def fa(i):
        return lambda tr: pl.DataFrame({"x": np.arange(i.size, dtype=float)})

    def fb(i):
        return lambda tr: pl.DataFrame({"none": np.zeros(i.size)})

    out = TR.combine_factories(fa, fb, None)(ids)(train)
    assert out.height == ids.size and out.columns == ["p0_x"]


def test_observed_cohort_is_subset_of_strict():
    ath = pl.DataFrame({"nfl_id": [1, 2, 3], "a": [1.0, None, 2.0], "b": [1.0, 1.0, None]})
    assert TR.observed_cohort({1, 2, 3}, ath, ["a", "b"]) == {1}
    assert TR.observed_cohort({2, 3}, ath, ["a"]) == {3}


def test_transfer_status_rules():
    paired = pl.DataFrame({"cohort": ["s"] * 3, "model": ["A", "B", "C"], "reference": ["M1"] * 3, "delta_rmse": [0.01, 0.002, -0.02],
                           "delta_lo": [0.004, -0.001, -0.03], "delta_hi": [0.02, 0.005, -0.01], "p_boot_le0": [0.0, 0.15, 1.0],
                           "primary": [True, False, False], "p_holm_primary": [0.01, None, None], "p_bh_all": [0.01, 0.3, 1.0]})
    stab = pl.DataFrame({"model": ["A", "B", "C"], "folds_better": [5, 3, 0]})
    s = dict(zip(*TR.transfer_status(paired, stab).select("model", "status").to_dict(as_series=False).values()))
    assert s == {"A": "validated", "B": "exploratory", "C": "invalid"}


# ====================================================================== blocked contextual metrics
def test_contextual_metrics_are_blocked_without_synchronised_data():
    av = RG.availability(RG.CHANNELS, athlete_columns=("forty", "weight"))
    s = dict(zip(av["metric"].to_list(), av["status"].to_list()))
    for m in ("pursuit_interception_efficiency", "separation_creation_efficiency", "defensive_recovery_efficiency", "lateral_recovery_efficiency"):
        assert s[m] == "blocked"
    assert s["cod_deficit_residual"] == "blocked"                        # no official agility time column
    assert s["braking_point_efficiency"] == "computable"
    av2 = RG.availability(RG.CHANNELS, athlete_columns=("forty", "weight", "three_cone"))
    assert dict(zip(av2["metric"].to_list(), av2["status"].to_list()))["cod_deficit_residual"] == "computable"


def test_registry_is_complete_and_columns_parse():
    for m in RG.METRICS:
        assert m.formula and m.family and m.level in RG.LEVELS
        assert m.sign in (-1, 0, 1)
    c = RG.combine_column("turn_speed_retention", "agility")
    assert RG.parse_column(c) == ("agility", "turn_speed_retention") and RG.parse_column("nfl__x") is None
    assert set(RG.BRAKING_CORE) <= set(RG.EVENT_METRICS)


# ====================================================================== twins / reports
def test_twins_never_reuse_an_athlete_and_respect_caliper():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"nfl_id": np.arange(40), "position_group": ["DB"] * 40, "forty": rng.normal(4.5, 0.1, 40), "weight": rng.normal(200, 10, 40),
                       "braking_composite": rng.normal(size=40)})
    pairs = SC.stopwatch_twins(df, ["forty", "weight"], caliper=0.5)
    used = pairs["athlete_a"].to_list() + pairs["athlete_b"].to_list()
    assert len(used) == len(set(used)) and (pairs["distance"] <= 0.5).all()
    assert SC._sign_test(5, 10) == pytest.approx(1.0) and SC._sign_test(0, 10) < 0.01


def test_report_generation(tmp_path):
    run = M14C.M14Run(tmp_path / "run", None)
    md = RP.final_report({"primary_status": pl.DataFrame({"model": ["M1+M14_reliable"], "reference": ["M1"], "delta_rmse": [0.001], "delta_lo": [-0.002],
                                                          "delta_hi": [0.004], "p_adjusted": [0.4], "adjustment": ["holm (primary)"], "folds_better": [3],
                                                          "status": ["inconclusive"]})})
    assert "# M14" in md and "inconclusive" in md and "Scientific limitations" in md
    paths = RP.write_report(run, md, "m14_final_report")
    html = paths[1].read_text(encoding="utf-8")
    assert "background: #ffffff" in html and all(p.is_file() for p in paths)
    assert "navy" not in RP.CSS.lower()
