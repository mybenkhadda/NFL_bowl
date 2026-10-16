"""Phase 03 · M1.1 — snap-anchored NFL play windows (Workstream C foundation; heavy step, run once).

Replaces "whole tracked window" NFL summaries with football-defined windows (definitions and caveats:
``features.play_windows``; parameters: ``features.config.PlayWindowConfig``, version-tracked).

Datasets (``ArtifactStore(phase="phase_03")``; bounded runs → ``phase_03/_sample``)::

    nfl_play_anchors           play                       (game_id, play_id)                    partition season
    nfl_player_play_windows    player-play × window       (game_id, play_id, nfl_id, window)    partition season

Tables (``reports/tables/phase_03[/_sample]/``, CSV + Parquet)::

    m11_nfl_anchor_coverage      snap / release availability and timing by season × season_type
    m11_nfl_release_timing       snap → release time quantiles by release event
    m11_nfl_window_coverage      per window: rows, valid features, coverage, censoring, boundary share
    m11_nfl_presnap_peak_share   share of player-plays whose whole-play peak speed is PRE-snap (Wilson CI)
    m11_nfl_snap_speed_profile   mean ± SD speed by position group × side × 0.1 s bin around the snap

Figures (``reports/figures/phase_03[/_sample]/``): m11_nfl_anchor_coverage, m11_nfl_presnap_peak_share,
m11_nfl_snap_speed_profile (PNG + SVG; plotted data saved as ``<figure>__data.csv``).

Leakage / scope: descriptive only. Nothing here uses career outcomes. Pre-snap frames are kept in
their own window and never mixed into post-snap features.

Commands for the USER (code authored, NOT executed)::

    set NFL_BDB_DUCKDB_MEMORY=3GB
    python -m nflbdb.research.nfl_windows --seasons 2023 --limit-games 10      # pilot → phase_03/_sample
    python -m nflbdb.research.nfl_windows                                      # full (REG season)
    python -m nflbdb.research.nfl_windows --check                              # exit 0 = all outputs present
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any, Sequence

import numpy as np
import polars as pl

from ..data.config import Phase01Config
from ..data.reporting import CATEGORICAL, INK
from ..data.utils import get_logger
from ..features.config import PLAY_WINDOWS_VERSION, KinematicsConfig, PlayWindowConfig
from ..features.kinematics import compute_kinematics
from ..features.play_windows import ANCHOR_DOCS, PLAY_KEYS, PP_KEYS, all_window_features, feature_docs, play_anchors
from ..features.trajectory import NFL, prepare_frames
from .artifacts import ArtifactStore, DatasetSpec
from .figkit import empty_panel, figure, finish
from .kinematic_build import KinematicFeatureBuilder
from .studies.stats import wilson_ci

log = get_logger(__name__)

PHASE = "phase_03"
SAMPLE_PHASE = "phase_03/_sample"
SOURCES = ("silver.game_tracking", "silver.games", "silver.plays", "silver.player_play", "gold.dim_athlete")
TRACK_COLS = ["game_id", "play_id", "nfl_id", "is_football", "frame_id", "ts", "x_std", "y_std", "s", "a", "dir_std",
              "o_std", "event"]
CONTEXT_COLS = ("season_type", "week", "possession_team", "team_abbr", "lined_up_position", "play_nullified_by_penalty")
PROFILE_RANGE_S = (-2.0, 5.0)
PROFILE_BIN_S = 0.1


def _specs(wcfg: PlayWindowConfig) -> dict[str, DatasetSpec]:
    anchors = DatasetSpec(
        name="nfl_play_anchors", grain="one row per NFL play (tracked)", key=tuple(PLAY_KEYS), sources=SOURCES,
        description="Snap / release anchors and pre/post-snap durations per play (PlayWindowConfig).",
        columns={"game_id": "game identifier", "play_id": "play identifier", "season": "NFL season",
                 "season_type": "REG / PRE / POST (silver.games)", **ANCHOR_DOCS},
        version=PLAY_WINDOWS_VERSION, partition_by="season")
    windows = DatasetSpec(
        name="nfl_player_play_windows", grain="one row per NFL player-play × window",
        key=(*PP_KEYS, "window"), sources=SOURCES,
        description="Snap-anchored movement features per player-play and window (standardised coordinates).",
        columns={"game_id": "game identifier", "play_id": "play identifier", "nfl_id": "athlete identifier",
                 "season": "NFL season", "season_type": "REG / PRE / POST", "week": "game week",
                 "side": "offense / defense (team_abbr vs possession_team); null if not in player_play",
                 "in_player_play": "player-play present in player_play (play context available)",
                 "position_group": "gold.dim_athlete position_group (Combine position, else NFL position)",
                 "cohort": "gold.dim_athlete cohort (draft year)",
                 "lined_up_position": "player_play lined_up_position (null if unavailable)",
                 "play_nullified_by_penalty": "player_play flag (play-level)", **feature_docs(wcfg)},
        version=PLAY_WINDOWS_VERSION, partition_by="season")
    return {"anchors": anchors, "windows": windows}


class NflWindowBuilder:
    def __init__(self, cfg: Phase01Config, kcfg: KinematicsConfig | None = None, wcfg: PlayWindowConfig | None = None,
                 seasons: Sequence[int] | None = None, limit_games: int | None = None):
        self.cfg = cfg
        self.kcfg = kcfg or KinematicsConfig.from_tracking(cfg.tracking)
        self.wcfg = wcfg or PlayWindowConfig()
        self.seasons = sorted(seasons) if seasons else None
        self.limit_games = limit_games
        self.sampled = limit_games is not None or seasons is not None
        self.store = ArtifactStore(cfg, phase=SAMPLE_PHASE if self.sampled else PHASE)
        self.kin = KinematicFeatureBuilder(cfg, self.kcfg, store=ArtifactStore(cfg, phase=PHASE), seasons=seasons,
                                           limit_games=limit_games)
        self.specs = _specs(self.wcfg)
        self._cache: dict[str, pl.DataFrame] | None = None

    # ------------------------------------------------------------------ parameters
    def params(self) -> dict[str, Any]:
        return {"kinematics": self.kcfg.params(), "windows": self.wcfg.params(), "seasons": self.seasons,
                "limit_games": self.limit_games}

    # ------------------------------------------------------------------ context (small tables, loaded once)
    def _cols(self, layer: str, name: str) -> list[str]:
        return self.store.refs.scan(layer, name).collect_schema().names()

    def games(self) -> pl.DataFrame:
        return self.store.refs.scan("silver", "games").select("game_id", "season", "season_type", "week").collect()

    def context(self) -> pl.DataFrame:
        """(game_id, play_id, nfl_id) → side, lined_up_position, play flags. Silver splits player_play into
        player-level (silver.player_play) and play-level (silver.plays) columns at build time, so each wanted
        column is taken from whichever table holds it. Position comes from dim_athlete (by nfl_id), so
        tracked player-plays absent from player_play still get a position."""
        wanted = ("team_abbr", "lined_up_position", "possession_team", "play_nullified_by_penalty")
        pp_cols, play_cols = self._cols("silver", "player_play"), self._cols("silver", "plays")
        from_pp = [c for c in wanted if c in pp_cols]
        from_play = [c for c in wanted if c in play_cols and c not in from_pp]
        ctx = (self.store.refs.scan("silver", "player_play").select(*PP_KEYS, *from_pp).collect()
               .with_columns([pl.col(c).cast(pl.Int64) for c in PP_KEYS]))
        if from_play:
            plays = (self.store.refs.scan("silver", "plays").select(*PLAY_KEYS, *from_play).collect()
                     .with_columns([pl.col(c).cast(pl.Int64) for c in PLAY_KEYS]))
            ctx = ctx.join(plays, on=PLAY_KEYS, how="left")
        for c in wanted:
            if c not in ctx.columns:
                log.warning("[windows] context column %s unavailable — filled with null", c)
                ctx = ctx.with_columns(pl.lit(None, dtype=pl.String).alias(c))
        return ctx.select(
            *[pl.col(c).cast(pl.Int64) for c in PP_KEYS],
            pl.when(pl.col("team_abbr").is_null() | pl.col("possession_team").is_null()).then(None)
            .when(pl.col("team_abbr") == pl.col("possession_team")).then(pl.lit("offense"))
            .otherwise(pl.lit("defense")).alias("side"),
            pl.col("lined_up_position").cast(pl.String),
            pl.col("play_nullified_by_penalty").cast(pl.String),
            pl.lit(True).alias("in_player_play"))

    def athletes(self) -> pl.DataFrame:
        if not self.store.refs.files("gold", "dim_athlete"):
            log.warning("[windows] gold.dim_athlete missing — position_group = 'UNKNOWN'")
            return pl.DataFrame(schema={"nfl_id": pl.Int64, "position_group": pl.String, "cohort": pl.Int64})
        return (self.store.refs.scan("gold", "dim_athlete")
                .select(pl.col("nfl_id").cast(pl.Int64), pl.col("position_group").cast(pl.String),
                        pl.col("cohort").cast(pl.Int64)).collect())

    # ------------------------------------------------------------------ batches
    def allowed_games(self, games: pl.DataFrame) -> set[int]:
        return set(games.filter(pl.col("season_type").is_in(list(self.wcfg.season_types)))["game_id"].to_list())

    def load_batch(self, file, lo: int, hi: int, allowed: set[int]) -> pl.DataFrame:
        cols = [c for c in TRACK_COLS if c in pl.read_parquet_schema(file)]
        missing = {"x_std", "y_std", "event"} - set(cols)
        if missing:
            raise KeyError(f"silver.game_tracking lacks {sorted(missing)} — rebuild silver (Phase 01)")
        return (pl.scan_parquet(file).filter(pl.col("game_id").is_between(lo, hi) & pl.col("game_id").is_in(list(allowed)))
                .select(cols).collect()
                .with_columns([pl.col(c).cast(pl.Int64) for c in PP_KEYS]))

    def player_kinematics(self, rows: pl.DataFrame, season: int) -> pl.DataFrame:
        """Whole-trajectory kinematics on STANDARDISED coordinates (offense → +x)."""
        p = (rows.filter(~pl.col("is_football") & pl.col("nfl_id").is_not_null())
             .rename({"x_std": "x", "y_std": "y", "dir_std": "dir"})
             .drop([c for c in ("o_std", "event", "is_football") if c in rows.columns])
             .with_columns(pl.lit(season, dtype=pl.Int64).alias("season")))
        frames, _ = prepare_frames(p, NFL, self.kcfg)
        return compute_kinematics(frames, self.kcfg).select(
            *PP_KEYS, "seg_idx", "pos", "seg_len", "ts", "x", "y", "speed", "a_tan", "accel_mag", "step_smooth",
            "valid_kin")

    @staticmethod
    def profile_sums(kin: pl.DataFrame, anchors: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
        """Speed sums by position_group × side × 0.1 s snap-relative bin (mergeable across batches)."""
        lo, hi = PROFILE_RANGE_S
        f = (kin.filter(pl.col("valid_kin")).join(anchors.select(*PLAY_KEYS, "snap_ts"), on=PLAY_KEYS, how="inner")
             .with_columns(((pl.col("ts") - pl.col("snap_ts")).dt.total_microseconds() / 1e6).alias("t_rel"))
             .filter((pl.col("t_rel") >= lo) & (pl.col("t_rel") <= hi))
             .with_columns((pl.col("t_rel") / PROFILE_BIN_S).round(0).cast(pl.Int64).alias("bin"))
             .join(labels, on=PP_KEYS, how="left")
             .with_columns(pl.col("position_group").fill_null("UNKNOWN"), pl.col("side").fill_null("unknown")))
        return f.group_by("position_group", "side", "bin").agg(
            pl.len().cast(pl.Int64).alias("n_frames"), pl.col("speed").sum().alias("sum_speed"),
            pl.col("speed").pow(2).sum().alias("sum_speed2"),
            pl.struct(PP_KEYS).n_unique().cast(pl.Int64).alias("n_player_plays"))

    # ------------------------------------------------------------------ compute
    def compute(self) -> dict[str, pl.DataFrame]:
        if self._cache is not None:
            return self._cache
        t0 = time.perf_counter()
        games, ctx, ath = self.games(), self.context(), self.athletes()
        allowed = self.allowed_games(games)
        sides, positions = ctx.select(*PP_KEYS, "side"), ath.select("nfl_id", "position_group")
        acc: dict[str, list[pl.DataFrame]] = {"anchors": [], "windows": [], "profile": []}
        for season, file in self.kin.nfl_season_files().items():
            batches = self.kin.game_batches(file)
            for i, (lo, hi) in enumerate(batches, 1):
                rows = self.load_batch(file, lo, hi, allowed)
                if rows.is_empty():
                    continue
                anchors = play_anchors(rows.select(*PLAY_KEYS, "ts", "event"), self.wcfg)
                kin = self.player_kinematics(rows, season)
                feats = all_window_features(kin, anchors, self.wcfg, self.kcfg)
                labels = (kin.select(*PP_KEYS).unique().join(sides, on=PP_KEYS, how="left")
                          .join(positions, on="nfl_id", how="left"))
                acc["profile"].append(self.profile_sums(kin, anchors.filter(pl.col("has_snap")), labels))
                acc["anchors"].append(anchors.with_columns(pl.lit(season, dtype=pl.Int64).alias("season")))
                acc["windows"].append(feats.with_columns(pl.lit(season, dtype=pl.Int64).alias("season")))
                log.info("[windows] season %s batch %d/%d (games %s–%s): %s rows, %s plays, %s window rows (%.0fs)",
                         season, i, len(batches), lo, hi, f"{rows.height:,}", f"{anchors.height:,}",
                         f"{feats.height:,}", time.perf_counter() - t0)
                del rows, kin, feats
                gc.collect()
        if not acc["anchors"]:
            raise RuntimeError(f"no NFL tracking rows for season_types {self.wcfg.season_types} — check silver.games")
        g = games.select("game_id", "season_type", "week")
        anchors = (pl.concat(acc["anchors"], how="vertical").join(g.drop("week"), on="game_id", how="left")
                   .sort(PLAY_KEYS))
        windows = (pl.concat(acc["windows"], how="vertical")
                   .join(g, on="game_id", how="left")
                   .join(ctx, on=PP_KEYS, how="left")
                   .with_columns(pl.col("in_player_play").fill_null(False))
                   .join(ath, on="nfl_id", how="left")
                   .with_columns(pl.col("position_group").fill_null("UNKNOWN"))
                   .sort([*PP_KEYS, "window"]))
        profile = (pl.concat(acc["profile"], how="vertical")
                   .group_by("position_group", "side", "bin")
                   .agg(pl.col("n_frames").sum(), pl.col("sum_speed").sum(), pl.col("sum_speed2").sum(),
                        pl.col("n_player_plays").sum()))
        self._cache = {"anchors": anchors, "windows": windows, "profile": profile}
        return self._cache

    # ------------------------------------------------------------------ persist
    def build(self, force: bool = False) -> dict[str, pl.LazyFrame]:
        params = self.params()
        out = {}
        for part in ("anchors", "windows"):
            out[part] = self.store.build(self.specs[part], lambda part=part: self.compute()[part], params=params,
                                         force=force)
        return out

    def tables(self, lf: dict[str, pl.LazyFrame]) -> dict[str, pl.DataFrame]:
        anchors, windows = lf["anchors"].collect(), lf["windows"].collect()
        tabs = {"m11_nfl_anchor_coverage": anchor_coverage(anchors),
                "m11_nfl_release_timing": release_timing(anchors),
                "m11_nfl_window_coverage": window_coverage(windows),
                "m11_nfl_presnap_peak_share": presnap_peak_share(windows)}
        prof = self._cache["profile"] if self._cache is not None else None
        if prof is not None:
            tabs["m11_nfl_snap_speed_profile"] = speed_profile(prof)
        else:
            p = self.store.tables_dir / "m11_nfl_snap_speed_profile.parquet"
            if p.is_file():
                tabs["m11_nfl_snap_speed_profile"] = pl.read_parquet(p)
            else:
                log.warning("[windows] speed profile not in memory and not on disk — rerun with --force to rebuild it")
        for name, df in tabs.items():
            self.store.export_csv(df, name, TABLE_DOCS[name], parquet=True)
        return tabs

    def figures(self, tabs: dict[str, pl.DataFrame]) -> None:
        fig_anchor_coverage(self.store, tabs["m11_nfl_anchor_coverage"], tabs["m11_nfl_release_timing"])
        fig_presnap_peak_share(self.store, tabs["m11_nfl_presnap_peak_share"])
        if "m11_nfl_snap_speed_profile" in tabs:
            fig_speed_profile(self.store, tabs["m11_nfl_snap_speed_profile"])

    def expected_outputs(self) -> list:
        paths = [self.store.dataset_path(s) for s in self.specs.values()]
        paths += [self.store.tables_dir / f"{n}.csv" for n in TABLE_DOCS]
        paths += [self.store.figures_dir / f"{n}.png" for n in FIGURE_NAMES]
        return paths


# ============================================================================== tables (pure)
TABLE_DOCS = {
    "m11_nfl_anchor_coverage": "Snap / release availability and pre/post-snap durations by season × season_type.",
    "m11_nfl_release_timing": "Snap → release time quantiles by release event.",
    "m11_nfl_window_coverage": "Per window × season: player-play rows, valid features, coverage, censoring, boundary.",
    "m11_nfl_presnap_peak_share": "Share of player-plays whose whole-play peak speed occurs before the snap.",
    "m11_nfl_snap_speed_profile": "Mean ± SD derived speed by position group × side × 0.1 s bin around the snap.",
}
FIGURE_NAMES = ("m11_nfl_anchor_coverage", "m11_nfl_presnap_peak_share", "m11_nfl_snap_speed_profile")


def _q(col: str, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> list[pl.Expr]:
    return [pl.col(col).quantile(q, "linear").alias(f"{col}_p{int(q * 100):02d}") for q in qs]


def anchor_coverage(anchors: pl.DataFrame) -> pl.DataFrame:
    return (anchors.group_by("season", "season_type").agg(
        pl.len().cast(pl.Int64).alias("n_plays"),
        pl.col("has_snap").mean().alias("share_has_snap"),
        pl.col("multiple_snap_tags").mean().alias("share_multiple_snap_tags"),
        pl.col("snap_at_play_start").mean().alias("share_snap_at_play_start"),
        pl.col("has_release").mean().alias("share_has_release"),
        *_q("pre_snap_s"), *_q("post_snap_s"))
        .sort("season", "season_type"))


def release_timing(anchors: pl.DataFrame) -> pl.DataFrame:
    return (anchors.filter(pl.col("has_release")).group_by("release_event").agg(
        pl.len().cast(pl.Int64).alias("n_plays"), *_q("snap_to_release_s"))
        .sort("n_plays", descending=True))


def window_coverage(w: pl.DataFrame) -> pl.DataFrame:
    return (w.group_by("window", "season").agg(
        pl.len().cast(pl.Int64).alias("n_player_plays"),
        pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"),
        pl.col("features_valid").mean().alias("share_features_valid"),
        pl.col("coverage_ok").mean().alias("share_coverage_ok"),
        pl.col("censored_start").mean().alias("share_censored_start"),
        pl.col("censored_end").mean().alias("share_censored_end"),
        pl.col("truncated_by_play_end").mean().alias("share_truncated_by_play_end"),
        pl.col("in_player_play").mean().alias("share_in_player_play"),
        pl.col("boundary_frac").mean().alias("mean_boundary_frac"),
        pl.col("speed_max_in_boundary").mean().alias("share_speed_max_in_boundary"),
        pl.col("time_to_disp_s").is_not_null().mean().alias("share_time_to_disp_defined"))
        .sort("window", "season"))


def presnap_peak_share(w: pl.DataFrame) -> pl.DataFrame:
    """Per position group × side: share of player-plays where max speed(pre_snap) > max speed(post_snap_all).

    This quantifies how often an M1-style whole-play peak would have been a PRE-snap value. Wilson CI
    treats player-plays as independent (optimistic; athletes repeat) — n_athletes is reported."""
    keep = ["pre_snap", "post_snap_all"]
    empty = pl.DataFrame(schema={"position_group": pl.String, "side": pl.String, "n_player_plays": pl.Int64,
                                 "n_athletes": pl.Int64, "n_peak_pre_snap": pl.Int64, "share": pl.Float64,
                                 "wilson_lower": pl.Float64, "wilson_upper": pl.Float64})
    long = (w.filter(pl.col("window").is_in(keep) & pl.col("features_valid"))
            .select(*PP_KEYS, "window", "speed_max", "position_group", "side"))
    if long.is_empty():
        return empty
    wide = long.pivot(on="window", index=[*PP_KEYS, "position_group", "side"], values="speed_max")
    for c in keep:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    wide = (wide.filter(pl.col("pre_snap").is_not_null() & pl.col("post_snap_all").is_not_null())
            .with_columns((pl.col("pre_snap") > pl.col("post_snap_all")).alias("peak_pre_snap"),
                          pl.col("side").fill_null("unknown")))
    if wide.is_empty():
        return empty
    parts =[wide.with_columns(pl.lit("all").alias("position_group"), pl.lit("all").alias("side")), wide]
    t = (pl.concat(parts, how="vertical").group_by("position_group", "side").agg(
        pl.len().cast(pl.Int64).alias("n_player_plays"), pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"),
        pl.col("peak_pre_snap").sum().cast(pl.Int64).alias("n_peak_pre_snap"))
        .with_columns((pl.col("n_peak_pre_snap") / pl.col("n_player_plays")).alias("share")))
    lo, hi = wilson_ci(t["n_peak_pre_snap"].to_numpy(), t["n_player_plays"].to_numpy())
    return t.with_columns(pl.Series("wilson_lower", lo, dtype=pl.Float64),
                          pl.Series("wilson_upper", hi, dtype=pl.Float64)).sort("position_group", "side")


def speed_profile(sums: pl.DataFrame) -> pl.DataFrame:
    return (sums.with_columns((pl.col("bin") * PROFILE_BIN_S).round(2).alias("t_rel_s"),
                              (pl.col("sum_speed") / pl.col("n_frames")).alias("mean_speed"))
            .with_columns(((pl.col("sum_speed2") / pl.col("n_frames") - pl.col("mean_speed").pow(2)).clip(0).sqrt())
                          .alias("sd_speed"))
            .select("position_group", "side", "t_rel_s", "n_frames", "n_player_plays", "mean_speed", "sd_speed")
            .sort("position_group", "side", "t_rel_s"))


# ============================================================================== figures
def fig_anchor_coverage(store: ArtifactStore, cov: pl.DataFrame, rel: pl.DataFrame) -> None:
    fig, ax = figure(1, 2)
    a, b = ax[0, 0], ax[0, 1]
    if cov.height:
        lab = [f"{s}\n{t}" for s, t in zip(cov["season"].to_list(), cov["season_type"].to_list())]
        x = np.arange(cov.height)
        for j, (col, name) in enumerate((("share_has_snap", "snap tag"), ("share_has_release", "release event"),
                                         ("share_snap_at_play_start", "snap at first frame"))):
            a.bar(x + (j - 1) * 0.27, cov[col].to_numpy(), width=0.27, color=CATEGORICAL[j], label=name)
        a.set_xticks(x, lab)
        a.set_ylim(0, 1.05)
        a.set_ylabel("share of tracked plays")
        a.legend(frameon=False, fontsize=9)
        a.set_title(f"Anchor availability (n plays = {int(cov['n_plays'].sum()):,})")
    else:
        empty_panel(a, "no plays")
    if rel.height:
        r = rel.head(8)
        y = np.arange(r.height)
        b.hlines(y, r["snap_to_release_s_p05"].to_numpy(), r["snap_to_release_s_p95"].to_numpy(), color=INK["muted"])
        b.hlines(y, r["snap_to_release_s_p25"].to_numpy(), r["snap_to_release_s_p75"].to_numpy(), color=CATEGORICAL[0],
                 lw=6)
        b.plot(r["snap_to_release_s_p50"].to_numpy(), y, "o", color="white", mec=CATEGORICAL[0])
        b.set_yticks(y, [f"{e} (n={n:,})" for e, n in zip(r["release_event"].to_list(), r["n_plays"].to_list())])
        b.invert_yaxis()
        b.set_xlabel("snap → release (s)")
        b.set_title("Release timing: median, IQR, p05–p95")
    else:
        empty_panel(b, "no release events")
    finish(fig, "Snap / release anchors from tracking event tags (PlayWindowConfig, provisional). Plays without a "
                "snap tag get no window. Shares are play-level. PENDING interpretation.",
           "NFL play anchors")
    store.export_csv(cov, "m11_nfl_anchor_coverage__data")
    store.save_fig(fig, "m11_nfl_anchor_coverage", source="nfl_play_anchors", n=int(cov["n_plays"].sum()) if cov.height else 0)


def fig_presnap_peak_share(store: ArtifactStore, t: pl.DataFrame) -> None:
    fig, ax = figure(1, 1)
    a = ax[0, 0]
    d = t.filter(pl.col("side") != "unknown").sort("share")
    if d.height:
        y = np.arange(d.height)
        a.errorbar(d["share"].to_numpy(), y,
                   xerr=np.vstack([d["share"].to_numpy() - d["wilson_lower"].to_numpy(),
                                   d["wilson_upper"].to_numpy() - d["share"].to_numpy()]).clip(0),
                   fmt="o", color=CATEGORICAL[0], capsize=3)
        a.set_yticks(y, [f"{p} · {s} (n={n:,}, ath={k})" for p, s, n, k in
                         zip(d["position_group"].to_list(), d["side"].to_list(), d["n_player_plays"].to_list(),
                             d["n_athletes"].to_list())], fontsize=8)
        a.set_xlabel("share of player-plays whose whole-play peak speed is pre-snap")
        a.set_xlim(0, max(0.05, float(d["wilson_upper"].max()) * 1.1))
    else:
        empty_panel(a, "no player-plays with both pre- and post-snap windows")
    finish(fig, "Pre-snap window max speed > post-snap max speed. Wilson 95 % CI treats player-plays as independent "
                "(optimistic; athletes repeat). Motivates snap-anchored windows; PENDING interpretation.",
           "How often would a whole-play peak be a pre-snap value?")
    store.export_csv(d, "m11_nfl_presnap_peak_share__data")
    store.save_fig(fig, "m11_nfl_presnap_peak_share", source="nfl_player_play_windows",
                   n=int(t.filter(pl.col("position_group") == "all")["n_player_plays"].sum()) if t.height else 0)


def fig_speed_profile(store: ArtifactStore, prof: pl.DataFrame) -> None:
    d = prof.filter(pl.col("side").is_in(["offense", "defense"]))
    groups = (d.group_by("position_group").agg(pl.col("n_player_plays").max().alias("n"))
              .sort("n", descending=True).head(8)["position_group"].to_list())
    fig, ax = figure(1, 2, sharey=True)
    for k, side in enumerate(("offense", "defense")):
        a = ax[0, k]
        s = d.filter(pl.col("side") == side)
        drawn = 0
        for j, g in enumerate(groups):
            r = s.filter(pl.col("position_group") == g).sort("t_rel_s")
            if r.height < 5:
                continue
            a.plot(r["t_rel_s"].to_numpy(), r["mean_speed"].to_numpy(), color=CATEGORICAL[j % len(CATEGORICAL)],
                   label=f"{g} (≤{int(r['n_player_plays'].max()):,} pp)")
            drawn += 1
        a.axvline(0, color=INK["muted"], lw=1, ls="--")
        a.set_xlabel("time relative to snap (s)")
        a.set_title(side)
        if drawn:
            a.legend(frameon=False, fontsize=8)
        else:
            empty_panel(a, f"no {side} profiles")
    ax[0, 0].set_ylabel("mean derived speed (yd/s)")
    finish(fig, "Mean speed of valid frames per 0.1 s bin around the snap (dashed line). Unequal player-play counts "
                "per bin; SD in the data table. Descriptive; PENDING interpretation.",
           "Post-snap movement signatures by position group")
    store.export_csv(d, "m11_nfl_snap_speed_profile__data")
    store.save_fig(fig, "m11_nfl_snap_speed_profile", source="nfl_player_play_windows (frame bins)",
                   n=int(d["n_frames"].sum()) if d.height else 0)


# ============================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    from ..data.config import configure_runtime

    ap = argparse.ArgumentParser(description="Phase 03 · M1.1 snap-anchored NFL windows")
    ap.add_argument("--seasons", type=int, nargs="*", help="NFL seasons (default: all) — a subset writes to _sample")
    ap.add_argument("--limit-games", type=int, default=None, help="pilot: first N games per season (→ _sample)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--check", action="store_true", help="list missing outputs; exit 1 if any")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    b = NflWindowBuilder(cfg, seasons=a.seasons or None, limit_games=a.limit_games)
    if a.check:
        missing = [cfg.paths.rel(p) for p in b.expected_outputs() if not p.exists()]
        print("all outputs present" if not missing else "MISSING:\n  " + "\n  ".join(missing))
        return 1 if missing else 0
    lf = b.build(force=a.force)
    tabs = b.tables(lf)
    if not a.no_figures:
        b.figures(tabs)
    with pl.Config(tbl_rows=20, tbl_cols=12):
        print(tabs["m11_nfl_anchor_coverage"])
        print(tabs["m11_nfl_window_coverage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
