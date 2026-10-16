"""M2.1 Workstream A — duplicate / near-duplicate attempt audit (synthetic, deterministic).

Covers: exact duplicates under different ids, translated copies, re-used subsequences, the
stationary-window false positive, near duplicates, genuinely consistent attempts, time overlap,
truncated duplicate recordings, onset alignment, rotation invariance, cross-athlete copies,
informational metadata, group truncation, exclusion scenarios, zero-variance explanations and the
output contracts (incl. Parquet round trip).

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_m21_duplicates.py -q
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflbdb.research.m21.config import DuplicateAuditConfig
from nflbdb.research.m21.duplicates import (ATTEMPT_AUDIT, FLAGS, PAIRS, SENSITIVITY, SUMMARY, ZERO_VAR, align,
                                            audit_frames, duplicate_sensitivity, finalize, flag_summary, make_track,
                                            onset_index, rigid_rmse, zero_variance_athletes)
from nflbdb.research.m21.io import conform

CFG = DuplicateAuditConfig()
FORTY = "FORTY_YARD_DASH"
DT = 0.1


def run(nfl_id=1, event_id="E1", attempt=1, t0_ms=0, vmax=9.0, tau=1.0, n=60, lead=10, x0=10.0, y0=20.0,
        jitter=0.0, seed=0, year=2024, drill=FORTY) -> pl.DataFrame:
    """Mono-exponential straight-line sprint after a stationary lead-in of ``lead`` frames."""
    t = np.arange(n) * DT
    v = vmax * (1 - np.exp(-np.clip(t - lead * DT, 0, None) / tau))
    d = np.concatenate([[0.0], np.cumsum(v[1:] * DT)])
    x, y = x0 + d, np.full(n, y0)
    if jitter:
        rng = np.random.default_rng(seed)
        x, y = x + rng.normal(0, jitter, n), y + rng.normal(0, jitter, n)
    return pl.DataFrame({
        "event_id": [event_id] * n, "nfl_id": [nfl_id] * n, "attempt": [attempt] * n, "drill": [drill] * n,
        "draft_year": [year] * n, "ts_ms": t0_ms + np.arange(n, dtype=np.int64) * 100, "x": x, "y": y, "s": v,
        "x_s": x, "y_s": y, "speed": v, "a_tan": np.gradient(v, DT), "valid_kin": [True] * n})


def shifted(df: pl.DataFrame, dx=0.0, dy=0.0, dt_ms=0, **ids) -> pl.DataFrame:
    out = df.with_columns(pl.col("x") + dx, pl.col("y") + dy, pl.col("x_s") + dx, pl.col("y_s") + dy,
                          pl.col("ts_ms") + dt_ms)
    return out.with_columns(*[pl.lit(v).alias(k) for k, v in ids.items()])


def only_pair(*frames: pl.DataFrame, cfg: DuplicateAuditConfig = CFG) -> dict:
    _, pairs = audit_frames(pl.concat(frames), cfg)
    assert pairs.height == 1
    return pairs.row(0, named=True)


# ----------------------------------------------------------------------------- pair classes
def test_exact_duplicate_under_different_ids():
    a = run()
    p = only_pair(a, a.with_columns(pl.lit("E2").alias("event_id")))
    assert p["raw_xy_hash_equal"] and p["pair_class"] == "exact_duplicate"
    assert p["integrity_group"] == "integrity_failure" and p["speed_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_translated_copy_in_space_and_time():
    a = run()
    p = only_pair(a, shifted(a, dx=5.0, dy=-3.0, dt_ms=60_000, event_id="E2"))
    assert not p["raw_xy_hash_equal"] and p["rel_xy_hash_equal"]
    assert p["pair_class"] == "translated_copy" and p["ts_overlap_s"] == 0.0


def test_reused_subsequence_after_a_different_lead_in():
    a = run(n=60, lead=10)
    prefix = run(event_id="E2", n=15, lead=100, x0=50.0, y0=50.0, t0_ms=100_000)       # 15 stationary frames
    copy = shifted(a.slice(15, 45), dx=40.0, dy=30.0, dt_ms=100_000, event_id="E2")      # ts continue after prefix
    p = only_pair(a, pl.concat([prefix, copy]))
    assert not p["rel_xy_hash_equal"]
    assert p["shared_kgram_frac"] >= CFG.reuse_kgram_frac and p["pair_class"] == "reused_sequence"


def test_identical_stationary_lead_in_is_not_reuse():
    """False-positive guard: all-zero displacement windows are excluded from the k-gram evidence."""
    a = run(lead=30, n=80)
    b = run(event_id="E2", lead=30, n=80, vmax=7.5, tau=1.4, t0_ms=60_000)               # same standing spot
    p = only_pair(a, b)
    assert not p["is_reused_sequence"] and not p["is_exact_duplicate"]
    assert p["integrity_group"] == "natural"


def test_near_duplicate_with_independent_jitter():
    base = dict(jitter=0.01)
    p = only_pair(run(seed=1, **base), shifted(run(seed=2, **base), dx=5.0, dt_ms=60_000, event_id="E2"))
    assert not p["rel_xy_hash_equal"] and not p["is_reused_sequence"]
    assert p["raw_pos_rmse_rigid_yds"] < CFG.near_dup_pos_rmse_yds
    assert p["pair_class"] == "near_duplicate" and p["integrity_group"] == "suspicious"


def test_genuinely_consistent_attempts_are_natural():
    # speed RMSE ≈ 0.2 yd/s (well above near-dup 0.05, below consistent 0.30); ~1 yd cumulative drift
    p = only_pair(run(), run(event_id="E2", vmax=9.3, tau=1.15, x0=12.0, t0_ms=60_000))
    assert p["comparable"] and p["integrity_group"] == "natural"
    assert p["pair_class"] in ("consistent", "distinct")
    assert p["raw_pos_rmse_rigid_yds"] > CFG.near_dup_pos_rmse_yds


def test_time_overlap_is_an_integrity_failure():
    p = only_pair(run(), run(event_id="E2", vmax=7.0, x0=30.0, t0_ms=3_000))
    assert p["ts_overlap_s"] == pytest.approx(2.9) and p["pair_class"] == "time_overlap"


def test_truncated_duplicate_recording_is_exact():
    a = run()
    p = only_pair(a, a.head(30).with_columns(pl.lit("E2").alias("event_id")))
    assert p["shared_frame_frac"] == 1.0 and p["pair_class"] == "exact_duplicate"


def test_single_frame_attempt_is_not_comparable():
    p = only_pair(run(), run(event_id="E2", n=1, t0_ms=60_000))
    assert not p["comparable"] and p["pair_class"] == "not_comparable" and p["integrity_group"] == "unresolved"


# ----------------------------------------------------------------------------- alignment / geometry
def test_onset_alignment_removes_different_lead_in():
    a, b = make_track(run(lead=5), CFG), make_track(run(event_id="E2", lead=15, n=70, t0_ms=60_000), CFG)
    al = align(a, b, CFG)
    assert al is not None and al.anchor == "onset" and al.lag_s == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("v", [np.zeros(20), np.full(20, 1.0), np.array([np.nan] * 5 + [3.0] * 2)])
def test_onset_index_none_for_zero_constant_or_short_runs(v):
    assert onset_index(v, CFG.align_speed, CFG.align_frames) is None


def test_rigid_rmse_is_rotation_and_translation_invariant():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(50, 2))
    th = np.deg2rad(37.0)
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert rigid_rmse(p, p @ r.T + [3.0, -2.0]) == pytest.approx(0.0, abs=1e-9)
    assert rigid_rmse(p[:2], p[:2]) is None


# ----------------------------------------------------------------------------- finalize / flags
def _finalize(frames: pl.DataFrame, cfg: DuplicateAuditConfig = CFG):
    att, pairs = audit_frames(frames, cfg)
    return finalize(att, pairs, cfg, "test")


def test_dedup_keeps_the_earliest_attempt_and_flags_conflicting_ids():
    a = run()
    audit, _, flags = _finalize(pl.concat([a, a.with_columns(pl.lit("E2").alias("event_id"))]))
    f = flags.sort("event_id")
    assert f["is_dedup_representative"].to_list() == [True, False]
    assert f["exclude_dedup"].to_list() == [False, True] and f["exclude_strict"].to_list() == [False, True]
    assert audit["conflicting_ids_same_start"].all()
    assert set(f["integrity_class"]) == {"exact_duplicate"}


def test_cross_athlete_copy_is_flagged_for_both_athletes_and_excluded_in_strict_only():
    a = run(nfl_id=1)
    b = shifted(a, dx=3.0, dy=3.0, dt_ms=1_000_000, nfl_id=2, event_id="E9")
    _, _, flags = _finalize(pl.concat([a, b]))
    assert flags["flag_cross_athlete_copy"].all() and flags["exclude_strict"].all()
    assert not flags["exclude_dedup"].any() and set(flags["integrity_class"]) == {"cross_athlete_copy"}


def test_short_static_recordings_do_not_trigger_cross_athlete_copy():
    a = run(nfl_id=1, n=8, lead=100)                                  # 8 stationary frames
    b = shifted(a, dt_ms=1_000_000, nfl_id=2, event_id="E9")
    audit, _, flags = _finalize(pl.concat([a, b]))
    assert not flags["flag_cross_athlete_copy"].any() and audit["n_athletes_same_rel_hash"].is_null().all()


def test_shared_event_attempt_id_is_informational_only():
    a = run(nfl_id=1)
    b = run(nfl_id=3, vmax=7.0, x0=50.0, t0_ms=500_000)               # same (event_id, attempt), other athlete
    _, _, flags = _finalize(pl.concat([a, b]))
    assert flags["info_shared_event_attempt_id"].all()
    assert set(flags["integrity_group"]) == {"unpaired"} and not flags["exclude_strict"].any()


def test_group_truncation_bounds_pairwise_work():
    cfg = DuplicateAuditConfig(max_attempts_per_group=2)
    frames = pl.concat([run(event_id=f"E{i}", vmax=8.0 + i, t0_ms=i * 60_000) for i in range(3)])
    att, pairs = audit_frames(frames, cfg)
    assert pairs.height == 1 and att["group_truncated"].all() and att["n_moving_kgrams"].null_count() == 1


def test_missing_required_frame_columns_raise():
    with pytest.raises(KeyError):
        audit_frames(run().drop("s"), CFG)


# ----------------------------------------------------------------------------- sensitivity / zero variance
def _flags_df(rows):
    return pl.DataFrame(rows, schema={"event_id": pl.String, "nfl_id": pl.Int64, "attempt": pl.Int64,
                                      "drill": pl.String, "draft_year": pl.Int64, "integrity_class": pl.String,
                                      "exclude_dedup": pl.Boolean, "exclude_strict": pl.Boolean}, orient="row")


def _values(rows):
    return pl.DataFrame(rows, schema={"event_id": pl.String, "nfl_id": pl.Int64, "attempt": pl.Int64,
                                      "drill": pl.String, "feature": pl.String, "value": pl.Float64}, orient="row")


FLAG_ROWS = [("E1", 1, 1, FORTY, 2024, "exact_duplicate", False, False),
             ("E2", 1, 1, FORTY, 2024, "exact_duplicate", True, True),
             ("E3", 4, 1, FORTY, 2024, "consistent", False, False),
             ("E4", 4, 2, FORTY, 2024, "consistent", False, False),
             ("S1", 2, 1, "SHORT_SHUTTLE", 2024, "consistent", False, False),
             ("S2", 2, 2, "SHORT_SHUTTLE", 2024, "consistent", False, False)]
VALUE_ROWS = [("E1", 1, 1, FORTY, "model_vmax", 9.5), ("E2", 1, 1, FORTY, "model_vmax", 9.5),
              ("E3", 4, 1, FORTY, "model_vmax", 9.0), ("E4", 4, 2, FORTY, "model_vmax", 9.4),
              ("S1", 2, 1, "SHORT_SHUTTLE", "n_cod", 4.0), ("S2", 2, 2, "SHORT_SHUTTLE", "n_cod", 4.0)]


def test_duplicate_sensitivity_scenarios():
    s = conform(duplicate_sensitivity(_values(VALUE_ROWS), _flags_df(FLAG_ROWS), CFG), SENSITIVITY)
    v = {r["scenario"]: r for r in s.filter(pl.col("feature") == "model_vmax").iter_rows(named=True)}
    assert v["all"]["n_athletes_repeated"] == 2 and v["all"]["share_zero_sd"] == pytest.approx(0.5)
    assert v["dedup"]["n_athletes_repeated"] == 1 and v["dedup"]["share_zero_sd"] == pytest.approx(0.0)
    assert v["all"]["pooled_within_sd"] == pytest.approx(0.2)
    assert v["dedup"]["pooled_within_sd_rel_change"] == pytest.approx(np.sqrt(0.08) / 0.2 - 1)


def test_zero_variance_explanations():
    z = conform(zero_variance_athletes(_values(VALUE_ROWS), _flags_df(FLAG_ROWS), CFG), ZERO_VAR)
    e = {(r["nfl_id"], r["feature"]): r["explanation"] for r in z.iter_rows(named=True)}
    assert e == {(1, "model_vmax"): "duplicated / overlapping source data",
                 (2, "n_cod"): "integer-valued feature — ties expected"}


# ----------------------------------------------------------------------------- contracts / reload
def test_outputs_satisfy_contracts_and_round_trip_parquet(tmp_path):
    a = run()
    frames = pl.concat([a, a.with_columns(pl.lit("E2").alias("event_id")),
                        run(event_id="E3", vmax=9.1, tau=1.05, t0_ms=60_000),
                        run(nfl_id=2, event_id="F1", vmax=8.0, t0_ms=900_000, year=2025)])
    audit, pairs, flags = _finalize(frames)
    for df, c in ((audit, ATTEMPT_AUDIT), (pairs, PAIRS), (flags, FLAGS), (flag_summary(flags), SUMMARY)):
        out = conform(df, c)
        p = tmp_path / f"{c.name}.parquet"
        out.write_parquet(p)
        back = pl.read_parquet(p)
        assert back.schema == c.schema and back.height == out.height
    s = flag_summary(flags)
    assert set(s["cohort"]) == {"2024", "2025", "all"}
    assert (s["ci_low"] <= s["share"] + 1e-12).all() and (s["share"] <= s["ci_high"] + 1e-12).all()
