"""M1.1 · snap-anchored NFL windows (features.play_windows, research.nfl_windows).

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_m1_1_play_windows.py -q
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from nflbdb.features.config import KinematicsConfig, PlayWindowConfig, WindowSpec
from nflbdb.features.kinematics import compute_kinematics
from nflbdb.features.play_windows import all_window_features, play_anchors, window_bounds, window_features
from nflbdb.features.trajectory import NFL, prepare_frames

K = KinematicsConfig()
W = PlayWindowConfig()
SNAP = datetime(2024, 9, 8, 13, 0, 0)
TOL = 1e-6


def ts(t: float) -> datetime:
    return SNAP + timedelta(milliseconds=round(t * 1000))


def play_rows(t_rel, events: dict[float, str], play: int = 1, nfl=(10,), football: bool = True) -> pl.DataFrame:
    """Tracking rows (game 1) for one play: events are attached to the FOOTBALL row only (worst case)."""
    rows = []
    for t in t_rel:
        for pid in [*nfl, *([None] if football else [])]:
            rows.append({"game_id": 1, "play_id": play, "nfl_id": pid, "ts": ts(t),
                         "event": events.get(round(t, 3)) if pid is None or not football else None})
    return pl.DataFrame(rows, schema={"game_id": pl.Int64, "play_id": pl.Int64, "nfl_id": pl.Int64,
                                      "ts": pl.Datetime("us"), "event": pl.String})


def player_kin(t_rel, x, play: int = 1, nfl: int = 10, y=None) -> pl.DataFrame:
    n = len(t_rel)
    df = pl.DataFrame({"game_id": [1] * n, "play_id": [play] * n, "nfl_id": [nfl] * n, "season": [2024] * n,
                       "frame_id": list(range(1, n + 1)), "ts": [ts(t) for t in t_rel], "x": np.asarray(x, float),
                       "y": np.zeros(n) if y is None else np.asarray(y, float)})
    frames, _ = prepare_frames(df, NFL, K)
    return compute_kinematics(frames, K).select(
        "game_id", "play_id", "nfl_id", "seg_idx", "pos", "seg_len", "ts", "x", "y", "speed", "a_tan", "accel_mag",
        "step_smooth", "valid_kin")


T = np.round(np.arange(-2.0, 4.0 + 1e-9, 0.1), 3)                   # −2.0 … 4.0 s around the snap


# ----------------------------------------------------------------------------- anchors
def test_anchor_snap_release_and_durations():
    a = play_anchors(play_rows(T, {0.0: "ball_snap", 2.5: "pass_forward", 3.1: "pass_arrived"}), W).row(0, named=True)
    assert a["has_snap"] and a["snap_ts"] == ts(0.0) and a["snap_event"] == "ball_snap"
    assert a["release_ts"] == ts(2.5) and a["release_event"] == "pass_forward"
    assert a["pre_snap_s"] == pytest.approx(2.0) and a["post_snap_s"] == pytest.approx(4.0)
    assert a["snap_to_release_s"] == pytest.approx(2.5)
    assert not a["multiple_snap_tags"] and not a["snap_at_play_start"]


def test_play_without_snap_has_no_windows():
    rows = play_rows(T, {2.5: "pass_forward"})
    a = play_anchors(rows, W)
    assert not a["has_snap"].item() and a["release_ts"].item() is None
    assert window_bounds(a, W.window("post_0_3"), K.dt_s).is_empty()
    feats = all_window_features(player_kin(T, 5 * np.maximum(T, 0)), a, W, K)
    assert feats.is_empty()


def test_multiple_snap_tags_use_the_earliest_and_are_flagged():
    a = play_anchors(play_rows(T, {0.0: "ball_snap", 0.3: "ball_snap"}), W).row(0, named=True)
    assert a["snap_ts"] == ts(0.0) and a["multiple_snap_tags"] and a["n_snap_tag_times"] == 2


def test_snap_event_priority_and_case():
    a = play_anchors(play_rows(T, {-0.5: "autoevent_ballsnap", 0.0: "BALL_SNAP"}), W).row(0, named=True)
    assert a["snap_event"] == "ball_snap" and a["snap_ts"] == ts(0.0)      # higher priority wins, case-insensitive


def test_release_before_snap_is_ignored():
    a = play_anchors(play_rows(T, {-1.0: "handoff", 0.0: "ball_snap", 1.2: "handoff"}), W).row(0, named=True)
    assert a["release_ts"] == ts(1.2)


def test_snap_at_first_frame_means_no_pre_snap_window():
    t = np.round(np.arange(0.0, 3.0 + 1e-9, 0.1), 3)
    a = play_anchors(play_rows(t, {0.0: "ball_snap"}), W)
    assert a["snap_at_play_start"].item()
    assert window_bounds(a, W.window("pre_snap"), K.dt_s).is_empty()       # hi (−0.1) < lo (0.0)


# ----------------------------------------------------------------------------- windows
def _features():
    x = np.where(T < 0, 0.0, 5.0 * T)                                  # still pre-snap, 5 yd/s from the snap
    a = play_anchors(play_rows(T, {0.0: "ball_snap", 2.5: "handoff"}), W)
    return all_window_features(player_kin(T, x), a, W, K)


def test_pre_snap_frames_never_enter_post_snap_windows():
    f = _features()
    post = f.filter(pl.col("phase") == "post_snap")
    assert (post["t_first_rel"] >= -W.frame_tolerance_s).all()
    pre = f.filter(pl.col("window") == "pre_snap").row(0, named=True)
    assert pre["t_last_rel"] == pytest.approx(-0.1) and pre["t_first_rel"] == pytest.approx(-2.0)


def test_window_frame_counts_and_coverage():
    f = {r["window"]: r for r in _features().iter_rows(named=True)}
    assert f["pre_snap"]["n_frames"] == 20 and f["pre_snap"]["expected_frames"] == 20
    assert f["post_0_1p5"]["n_frames"] == 16 and f["post_0_1p5"]["coverage"] == pytest.approx(1.0)
    assert f["post_0_3"]["n_frames"] == 31
    assert f["snap_to_release"]["n_frames"] == 26
    assert f["post_snap_all"]["n_frames"] == 41
    assert not any(r["censored_start"] or r["censored_end"] for r in f.values())


def test_movement_features_respect_the_snap():
    f = {r["window"]: r for r in _features().iter_rows(named=True)}
    # stationary before the snap; the centred SG7 fit at t = −0.1 s sees 2 post-snap frames → ≈ 1.43 yd/s smear
    assert f["pre_snap"]["speed_max"] == pytest.approx(4.0 / 28 / 0.1, abs=0.02)
    assert f["post_0_3"]["speed_max"] == pytest.approx(5.0, abs=0.05)
    assert f["post_0_3"]["disp_x"] == pytest.approx(15.0) and f["post_0_3"]["disp_y"] == pytest.approx(0.0)
    assert f["post_0_1p5"]["time_to_disp_s"] == pytest.approx(0.2, abs=TOL)   # raw displacement 1 yd at 0.2 s
    assert f["pre_snap"]["time_to_disp_s"] is None                     # timing only for snap-anchored windows
    assert f["post_0_3"]["hs_dist_4"] > 0 and f["post_0_3"]["hs_dist_8"] == pytest.approx(0.0)


def test_play_ending_early_is_truncated_and_censored():
    t = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.1), 3)
    a = play_anchors(play_rows(t, {0.0: "ball_snap"}), W)
    r = window_features(player_kin(t, 4 * np.maximum(t, 0)), a, W.window("post_0_3"), W, K).row(0, named=True)
    assert r["truncated_by_play_end"] and r["censored_end"]
    assert r["n_frames"] == 11 and r["expected_frames"] == 31 and not r["coverage_ok"]


def test_player_joining_late_is_censored_at_start():
    a = play_anchors(play_rows(T, {0.0: "ball_snap"}), W)
    late = T[T >= 0.5]
    r = window_features(player_kin(late, 5 * late), a, W.window("post_0_3"), W, K).row(0, named=True)
    assert r["censored_start"] and r["time_to_disp_s"] is None          # no frame at the snap → no first-step time


def test_too_few_valid_frames_nulls_features_but_keeps_row():
    a = play_anchors(play_rows(T, {0.0: "ball_snap"}), W)
    short = T[(T >= 0) & (T <= 0.5)]                                   # 6 frames < SG window → no derivatives
    r = window_features(player_kin(short, 5 * short), a, W.window("post_0_3"), W, K).row(0, named=True)
    assert not r["features_valid"] and r["speed_max"] is None and r["n_frames"] == 6


def test_frames_from_another_play_never_mix():
    a = play_anchors(pl.concat([play_rows(T, {0.0: "ball_snap"}, play=1), play_rows(T, {1.0: "ball_snap"}, play=2)]), W)
    k = pl.concat([player_kin(T, 5 * np.maximum(T, 0), play=1), player_kin(T, np.zeros(T.size), play=2)])
    f = window_features(k, a, W.window("post_0_3"), W, K).sort("play_id")
    assert f["play_id"].to_list() == [1, 2]
    assert f.row(1, named=True)["speed_max"] == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------------- config / versioning
def test_window_config_validation_and_params():
    with pytest.raises(ValueError):
        PlayWindowConfig(windows=(WindowSpec("a", "snap", 0, "snap", 1, "post_snap"),) * 2)
    with pytest.raises(ValueError):
        WindowSpec("bad", "kickoff", 0, "snap", 1, "post_snap")
    changed = replace(W, getoff_disp_yds=1.5)
    assert changed.params() != W.params()
    assert KinematicsConfig().params() == K.params()                  # window changes never touch M1 parameters


# ----------------------------------------------------------------------------- end-to-end (synthetic warehouse)
def _plant_snaps(cfg) -> None:
    """Tag the 3rd tracked timestamp of every play as ball_snap (football row) in the synthetic CSV."""
    p = cfg.paths.data_root / "game_tracking_2023.csv"
    df = pl.read_csv(p, infer_schema_length=0, null_values=["NA"])
    rank = pl.col("time").rank("dense").over(["game_id", "play_id"])
    df = df.with_columns(pl.when((rank == 3) & pl.col("nfl_id").is_null()).then(pl.lit("ball_snap"))
                         .otherwise(pl.col("event")).alias("event"))
    df.write_csv(p, null_value="NA")


def test_build_reuse_and_parameter_invalidation(synth):
    from nflbdb.data.bronze import BronzeIngestor
    from nflbdb.data.gold import GoldBuilder
    from nflbdb.data.silver import SilverBuilder
    from nflbdb.research.nfl_windows import NflWindowBuilder

    _plant_snaps(synth)
    BronzeIngestor(synth).ingest()
    SilverBuilder(synth).build_all()
    GoldBuilder(synth).build_all()
    b = NflWindowBuilder(synth)
    out = b.build()
    anchors = out["anchors"].collect()
    assert anchors.height == 2 and anchors["has_snap"].all()
    w = out["windows"].collect()
    post = w.filter(pl.col("window") == "post_snap_all")
    # 101 moves +x in raw coordinates: play 1 is 'left' (standardised x decreases), play 5 is 'right'
    p1 = post.filter((pl.col("play_id") == 1) & (pl.col("nfl_id") == 101)).row(0, named=True)
    p5 = post.filter((pl.col("play_id") == 5) & (pl.col("nfl_id") == 101)).row(0, named=True)
    assert p1["disp_x"] == pytest.approx(-3.5) and p5["disp_x"] == pytest.approx(3.5)
    assert p5["time_to_disp_s"] == pytest.approx(0.2, abs=TOL)
    pre = w.filter(pl.col("window") == "pre_snap")
    assert not pre["features_valid"].any()                              # 2 pre-snap frames < min_window_frames
    tabs = b.tables(out)
    assert set(tabs) >= {"m11_nfl_anchor_coverage", "m11_nfl_window_coverage", "m11_nfl_presnap_peak_share"}

    reuse = NflWindowBuilder(synth)
    reuse.compute = lambda: (_ for _ in ()).throw(AssertionError("should reuse persisted data"))
    reuse.build()
    changed = NflWindowBuilder(synth, wcfg=replace(PlayWindowConfig(), getoff_disp_yds=1.5))
    current, reason = changed.store.status(changed.specs["windows"], changed.params())
    assert not current and "parameters" in reason


def test_bounded_runs_never_write_to_the_full_namespace(synth):
    from nflbdb.research.nfl_windows import NflWindowBuilder

    assert NflWindowBuilder(synth, limit_games=1).store.data_dir != NflWindowBuilder(synth).store.data_dir
    assert "_sample" in str(NflWindowBuilder(synth, seasons=[2023]).store.data_dir)
