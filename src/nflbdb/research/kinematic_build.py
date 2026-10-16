"""Phase 03 · M1 — build and persist kinematic datasets (heavy step; run once, then reuse).

Outputs (``warehouse/gold/phase_03/``, catalogued as ``p03_<name>`` views):

    combine_frame_kinematics        frame      (event_id, nfl_id, attempt, ts)            partition draft_year
    combine_segment_quality         segment    (event_id, nfl_id, attempt, seg_idx)
    combine_trajectory_quality      trajectory (event_id, nfl_id, attempt)
    combine_kinematic_events        event      (… seg_idx, event_type, rule, event_idx)
    nfl_segment_quality             segment    (game_id, play_id, nfl_id, seg_idx)          partition season
    nfl_trajectory_quality          trajectory (game_id, play_id, nfl_id)                   partition season
    nfl_kinematic_events            event      (… seg_idx, event_type, rule, event_idx)   partition season

NFL frame-level kinematics (~22.7M rows) are deliberately NOT persisted in M1: on a machine with
~2 GB free RAM the per-play summaries are what downstream analyses need, and single plays are
recomputed on demand in milliseconds (:meth:`KinematicFeatureBuilder.nfl_play_kinematics`).

Reuse: ``ArtifactStore`` reuses a dataset only if version, input fingerprint (silver files +
every KinematicsConfig parameter + build scope) and on-disk schema all match. ``--force``
rebuilds; stale versions are archived, never overwritten.

Memory: Combine is processed in one pass (~0.46M frames). NFL is processed per season in
contiguous game_id batches (``--batch-games``, default 25 ≈ 0.6M frames); silver is sorted by
game_id so each batch reads only its row groups.

Commands for the USER (code authored, NOT executed)::

    set NFL_BDB_DUCKDB_MEMORY=3GB
    python -m nflbdb.research.kinematic_build --domain combine
    python -m nflbdb.research.kinematic_build --domain nfl --seasons 2023 --limit-games 10   # pilot
    python -m nflbdb.research.kinematic_build --domain nfl                                   # full
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any, Sequence

import polars as pl

from ..data.config import Phase01Config
from ..data.utils import get_logger
from ..features import FEATURES_VERSION
from ..features.config import KinematicsConfig, SmoothingSpec
from ..features.events import detect_events, empty_events
from ..features.feature_registry import column_docs
from ..features.kinematics import compute_kinematics
from ..features.segment_quality import summarize_segments, trajectory_quality
from ..features.trajectory import COMBINE, NFL, RAW_COLUMNS, TrajectoryDomain, prepare_frames
from .artifacts import ArtifactStore, DatasetSpec

log = get_logger(__name__)

PHASE = "phase_03"
FRAME_FLOAT32 = ("x_s", "y_s", "vx", "vy", "speed", "ax", "ay", "accel_mag", "a_tan", "a_lat", "yaw_rate",
                 "curvature", "heading", "step_smooth", "pos_resid", "step_yds", "t_s", "dt_ms")
FRAME_COLUMNS = ["draft_year", "event_id", "nfl_id", "attempt", "drill_type", "drill_name", "drill", "seg_idx", "pos",
                 "seg_len", "frame_id", "ts", "t_s", "dt_ms", "break_reason", "x", "y", "s", "a", "dir", "step_yds",
                 "x_s", "y_s", "vx", "vy", "speed", "ax", "ay", "accel_mag", "a_tan", "a_lat", "yaw_rate", "curvature",
                 "heading", "step_smooth", "pos_resid", "valid_kin", "heading_valid"]


def _spec(name: str, grain: str, key: tuple[str, ...], domain: TrajectoryDomain, description: str, level: str,
          columns: Sequence[str], partition_by: str | None = None) -> DatasetSpec:
    source = "silver.combine_tracking" if domain.name == "combine" else "silver.game_tracking"
    return DatasetSpec(name=name, grain=grain, key=key, sources=(source,), description=description,
                       columns=column_docs(list(columns), level), version=FEATURES_VERSION, partition_by=partition_by)


_SEG_COLS = ["break_reason", "start_ts", "t_start_s", "n_frames", "duration_s", "smoothable", "n_offgrid", "path_raw_yds",
             "path_smooth_yds", "net_disp_yds", "max_step_raw_yds", "resid_rms_yds", "resid_p95_yds", "speed_vendor_mad",
             "speed_vendor_p95", "max_vendor_s", "max_speed", "max_accel_mag", "n_speed_implausible",
             "n_accel_implausible", "frac_heading_valid"]
_TRAJ_COLS = ["n_raw_frames", "n_null_xy_removed", "n_dup_ts_removed", "n_gap_breaks", "n_jump_breaks", "n_drill_breaks",
              "n_segments", "n_valid_frames", "n_segments_seen", "n_smoothable_segments", "frames_smoothable",
              "longest_segment_frames", "resid_rms_yds", "speed_vendor_mad", "n_offgrid", "max_speed", "max_accel_mag",
              "n_speed_implausible", "n_accel_implausible", "path_smooth_yds", "coverage", "quality_tier", "tier_reasons"]
_EVENT_COLS = ["event_type", "rule", "event_idx", "start_frame_id", "end_frame_id", "t_start_s", "t_end_s", "n_frames",
               "duration_s", "entry_speed", "exit_speed", "min_speed", "max_speed", "speed_change", "peak_abs_a_tan",
               "mean_abs_a_tan", "distance_yds", "time_to_transition_s", "direction", "peak_abs_yaw_rate",
               "heading_change_deg", "median_radius_yds", "max_abs_curvature", "touches_segment_edge"]
_EV_KEY_TAIL = ("seg_idx", "event_type", "rule", "event_idx")

COMBINE_FRAMES = _spec(
    "combine_frame_kinematics", "one row per Combine PLAYER frame (after dedup)", ("event_id", "nfl_id", "attempt", "ts"),
    COMBINE, "Segment-local SG-smoothed kinematics for every Combine frame (raw + derived).", "frame",
    [*FRAME_COLUMNS], partition_by="draft_year")
COMBINE_SEGMENTS = _spec(
    "combine_segment_quality", "one row per Combine attempt segment", (*COMBINE.keys, "seg_idx"), COMBINE,
    "Segment quality metrics (sampling, residual noise, vendor agreement, plausibility).", "segment",
    [*COMBINE.keys, "seg_idx", *COMBINE.context, *_SEG_COLS])
COMBINE_TRAJ = _spec(
    "combine_trajectory_quality", "one row per Combine attempt", COMBINE.keys, COMBINE,
    "Attempt-level integrity, coverage and rule-based quality tier for kinematic features.", "trajectory",
    [*COMBINE.keys, *COMBINE.context, *_TRAJ_COLS])
COMBINE_EVENTS = _spec(
    "combine_kinematic_events", "one row per detected Combine event × threshold set",
    (*COMBINE.keys, *_EV_KEY_TAIL), COMBINE,
    "Braking, acceleration, turn and low-speed reversal events under lenient/default/strict rules.", "event",
    [*COMBINE.keys, *COMBINE.context, *_EVENT_COLS])
NFL_SEGMENTS = _spec(
    "nfl_segment_quality", "one row per NFL player-play segment", (*NFL.keys, "seg_idx"), NFL,
    "Segment quality metrics for NFL player-plays (whole tracked window).", "segment",
    [*NFL.keys, "seg_idx", *NFL.context, *_SEG_COLS], partition_by="season")
NFL_TRAJ = _spec(
    "nfl_trajectory_quality", "one row per NFL player-play", NFL.keys, NFL,
    "Player-play integrity, coverage and rule-based quality tier for kinematic features.", "trajectory",
    [*NFL.keys, *NFL.context, *_TRAJ_COLS], partition_by="season")
NFL_EVENTS = _spec(
    "nfl_kinematic_events", "one row per detected NFL event × threshold set", (*NFL.keys, *_EV_KEY_TAIL), NFL,
    "Braking, acceleration, turn and reversal events over the whole tracked window (no play-phase context yet).",
    "event", [*NFL.keys, *NFL.context, *_EVENT_COLS], partition_by="season")

SPECS = {s.name: s for s in (COMBINE_FRAMES, COMBINE_SEGMENTS, COMBINE_TRAJ, COMBINE_EVENTS,
                             NFL_SEGMENTS, NFL_TRAJ, NFL_EVENTS)}


def compact(df: pl.DataFrame) -> pl.DataFrame:
    """Float64 → Float32 for persisted summaries (halves memory; ample precision for yd, s, yd/s)."""
    return df.with_columns([pl.col(c).cast(pl.Float32) for c, t in df.schema.items() if t == pl.Float64])


def run_pipeline(raw: pl.DataFrame, domain: TrajectoryDomain, kcfg: KinematicsConfig,
                 spec: SmoothingSpec | None = None, with_events: bool = True,
                 event_rules: tuple[str, ...] | None = None) -> dict[str, pl.DataFrame]:
    """Pure in-memory pipeline for one bounded chunk: segment → kinematics → summaries → events.

    ``event_rules`` restricts threshold sets by name (None = all)."""
    frames, stats = prepare_frames(raw, domain, kcfg)
    kin = compute_kinematics(frames, kcfg, spec)
    seg = summarize_segments(kin, domain, kcfg)
    traj = trajectory_quality(seg, stats, domain, kcfg)
    ev = detect_events(kin, domain, kcfg, rules=event_rules) if with_events else empty_events(kin, domain)
    return {"frames": kin, "segments": seg, "trajectories": traj, "events": ev}


class KinematicFeatureBuilder:
    def __init__(self, cfg: Phase01Config, kcfg: KinematicsConfig | None = None, store: ArtifactStore | None = None,
                 seasons: Sequence[int] | None = None, limit_games: int | None = None,
                 nfl_event_rules: tuple[str, ...] | None = None):
        self.cfg = cfg
        self.kcfg = kcfg or KinematicsConfig.from_tracking(cfg.tracking)
        self.store = store or ArtifactStore(cfg, phase=PHASE)
        self.seasons = sorted(seasons) if seasons else None
        self.limit_games = limit_games
        # All threshold sets for NFL would be ~3× the rows (est. several million) — the full build
        # keeps the primary set; lenient/strict sensitivity is computed on a sample in diagnostics.
        self.nfl_event_rules = nfl_event_rules or (self.kcfg.primary_rule,)
        self._cache: dict[str, dict[str, pl.DataFrame]] = {}

    # ------------------------------------------------------------------ parameters
    def params(self, domain: str) -> dict[str, Any]:
        p: dict[str, Any] = {"kinematics": self.kcfg.params(), "domain": domain}
        if domain == "nfl":
            p |= {"seasons": self.seasons, "limit_games": self.limit_games, "event_rules": list(self.nfl_event_rules)}
        return p

    # ------------------------------------------------------------------ Combine
    def load_combine_raw(self) -> pl.DataFrame:
        lf = self.store.refs.scan("silver", "combine_tracking")
        return (lf.filter((pl.col("entity_type") == "PLAYER") & pl.col("nfl_id").is_not_null())
                .select(RAW_COLUMNS["combine"]).collect())

    def _combine(self) -> dict[str, pl.DataFrame]:
        if "combine" not in self._cache:
            t0 = time.perf_counter()
            raw = self.load_combine_raw()
            log.info("[kinematics] combine: %s PLAYER frames loaded", f"{raw.height:,}")
            out = run_pipeline(raw, COMBINE, self.kcfg)
            out["frames"] = out["frames"].select([c for c in FRAME_COLUMNS if c in out["frames"].columns]).with_columns(
                [pl.col(c).cast(pl.Float32) for c in FRAME_FLOAT32 if c in out["frames"].columns])
            for k in ("segments", "trajectories", "events"):
                out[k] = compact(out[k])
            self._cache["combine"] = out
            log.info("[kinematics] combine computed in %.1fs: %s", time.perf_counter() - t0,
                     {k: v.height for k, v in out.items()})
        return self._cache["combine"]

    def build_combine(self, force: bool = False) -> dict[str, pl.LazyFrame]:
        params = self.params("combine")
        parts = {"frames": COMBINE_FRAMES, "segments": COMBINE_SEGMENTS, "trajectories": COMBINE_TRAJ,
                 "events": COMBINE_EVENTS}
        return {part: self.store.build(spec, lambda part=part: self._combine()[part], params=params, force=force)
                for part, spec in parts.items()}

    # ------------------------------------------------------------------ NFL
    def nfl_season_files(self) -> dict[int, Any]:
        files = {}
        for f in self.store.refs.files("silver", "game_tracking"):
            season = int(f.parent.name.split("=")[-1])
            if self.seasons is None or season in self.seasons:
                files[season] = f
        if not files:
            raise FileNotFoundError(f"no silver.game_tracking files for seasons {self.seasons}")
        return dict(sorted(files.items()))

    def game_batches(self, file) -> list[tuple[int, int]]:
        games = (pl.scan_parquet(file).select("game_id").unique().sort("game_id").collect()["game_id"].to_list())
        if self.limit_games:
            games = games[: self.limit_games]
        b = max(1, self.kcfg.nfl_batch_games)
        return [(games[i], games[min(i + b, len(games)) - 1]) for i in range(0, len(games), b)]

    def load_nfl_raw(self, file, season: int, lo: int, hi: int) -> pl.DataFrame:
        cols = [c for c in RAW_COLUMNS["nfl"] if c != "season"]
        return (pl.scan_parquet(file)
                .filter(pl.col("game_id").is_between(lo, hi) & ~pl.col("is_football") & pl.col("nfl_id").is_not_null())
                .select(cols).collect()
                .with_columns(pl.lit(season, dtype=pl.Int64).alias("season")))

    def _nfl(self) -> dict[str, pl.DataFrame]:
        if "nfl" not in self._cache:
            acc: dict[str, list[pl.DataFrame]] = {"segments": [], "trajectories": [], "events": []}
            t0 = time.perf_counter()
            for season, file in self.nfl_season_files().items():
                batches = self.game_batches(file)
                for i, (lo, hi) in enumerate(batches, 1):
                    raw = self.load_nfl_raw(file, season, lo, hi)
                    out = run_pipeline(raw, NFL, self.kcfg, event_rules=self.nfl_event_rules)
                    for k in acc:
                        acc[k].append(compact(out[k]))
                    log.info("[kinematics] nfl season %s batch %d/%d (games %s–%s): %s frames, %s events (%.0fs)",
                             season, i, len(batches), lo, hi, f"{raw.height:,}", f"{out['events'].height:,}",
                             time.perf_counter() - t0)
                    del raw, out
                    gc.collect()
            self._cache["nfl"] = {k: pl.concat(v, how="diagonal_relaxed") for k, v in acc.items()}
        return self._cache["nfl"]

    def build_nfl(self, force: bool = False) -> dict[str, pl.LazyFrame]:
        params = self.params("nfl")
        parts = {"segments": NFL_SEGMENTS, "trajectories": NFL_TRAJ, "events": NFL_EVENTS}
        return {part: self.store.build(spec, lambda part=part: self._nfl()[part], params=params, force=force)
                for part, spec in parts.items()}

    # ------------------------------------------------------------------ on demand
    def nfl_play_kinematics(self, game_id: int, play_id: int, nfl_id: int | Sequence[int] | None = None,
                            spec: SmoothingSpec | None = None) -> pl.DataFrame:
        """Frame-level kinematics for one play (all players or selected ones), recomputed on demand."""
        g = (self.store.refs.scan("silver", "games").filter(pl.col("game_id") == game_id).select("season").collect())
        if g.is_empty():
            raise KeyError(f"game_id {game_id} not in silver.games")
        season = int(g.item())
        files = {int(f.parent.name.split("=")[-1]): f for f in self.store.refs.files("silver", "game_tracking")}
        if season not in files:
            raise KeyError(f"season {season} (game_id {game_id}) not in silver.game_tracking")
        raw = self.load_nfl_raw(files[season], season, game_id, game_id).filter(pl.col("play_id") == play_id)
        if nfl_id is not None:
            ids = [nfl_id] if isinstance(nfl_id, int) else list(nfl_id)
            raw = raw.filter(pl.col("nfl_id").is_in(ids))
        frames, _ = prepare_frames(raw, NFL, self.kcfg)
        return compute_kinematics(frames, self.kcfg, spec)


def main(argv: list[str] | None = None) -> int:
    from ..data.config import configure_runtime

    ap = argparse.ArgumentParser(description="Phase 03 · M1 kinematic datasets")
    ap.add_argument("--domain", choices=("combine", "nfl", "all"), default="combine")
    ap.add_argument("--seasons", type=int, nargs="*", help="NFL seasons (default: all)")
    ap.add_argument("--limit-games", type=int, default=None, help="NFL pilot: first N games per season")
    ap.add_argument("--batch-games", type=int, default=None, help="NFL games per in-memory batch (default 25)")
    ap.add_argument("--force", action="store_true", help="rebuild even if current")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    kcfg = KinematicsConfig.from_tracking(cfg.tracking)
    if a.batch_games:
        from dataclasses import replace
        kcfg = replace(kcfg, nfl_batch_games=a.batch_games)
    b = KinematicFeatureBuilder(cfg, kcfg, seasons=a.seasons, limit_games=a.limit_games)
    if a.domain in ("combine", "all"):
        out = b.build_combine(force=a.force)
        print({k: v.select(pl.len()).collect().item() for k, v in out.items()})
    if a.domain in ("nfl", "all"):
        out = b.build_nfl(force=a.force)
        print({k: v.select(pl.len()).collect().item() for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
