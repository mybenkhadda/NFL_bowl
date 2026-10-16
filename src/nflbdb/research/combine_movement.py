"""Phase 03 · M2 — Combine movement features (heavy step; reads persisted M1 datasets only).

Outputs (``warehouse/gold/phase_03/``, catalog views ``p03_<name>``):

    combine_attempt_features        attempt    (event_id, nfl_id, attempt)        generic families A–E
    combine_cod_events              COD        (event_id, nfl_id, attempt, cod_idx)
    combine_sprint_profiles         attempt    (event_id, nfl_id, attempt)        forty only (model + splits)
    combine_athlete_drill_features  athlete    (nfl_id, drill, feature)           long: median/mean/sd/min/max/best/first

Inputs: ``combine_frame_kinematics``, ``combine_kinematic_events`` (rule ``MovementConfig.event_rule``),
``combine_trajectory_quality`` (tiers in ``usable_tiers``), ``gold.dim_athlete`` (position group, if
built). Raw tracking is never rescanned. Reuse = version + fingerprint (upstream files + every
MovementConfig AND KinematicsConfig parameter) + schema, via ``ArtifactStore``.

Command for the USER (code authored, NOT executed)::

    python -m nflbdb.research.combine_movement            # add --force to rebuild
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import polars as pl

from ..data.config import Phase01Config
from ..data.utils import get_logger
from ..features.agility import cod_events, cod_window_measures, agility_features
from ..features.acceleration import speed_accel_features
from ..features.braking import braking_features
from ..features.config import KinematicsConfig, MovementConfig
from ..features.efficiency import path_features
from ..features.feature_registry import HIGHER_IS_BETTER, column_docs, registered_names
from ..features.sprint import SPRINT_COLUMNS, sprint_profiles
from ..features.temporal import active_frames, movement_window, temporal_features
from ..features.trajectory import COMBINE
from .artifacts import ArtifactStore, DatasetSpec
from .kinematic_build import COMBINE_EVENTS, COMBINE_FRAMES, COMBINE_TRAJ, PHASE, KinematicFeatureBuilder, compact

log = get_logger(__name__)

KEYS = list(COMBINE.keys)
CTX = ["draft_year", "drill_type", "drill_name", "drill"]
ID_COLS = [*KEYS, *CTX, "start_ts", "quality_tier", "n_segments", "attempt_order", "n_attempts_drill", "position_group"]
# attempt columns that are bookkeeping, not athlete features (kept in the attempt table, not in the long table)
NOT_FEATURES = {"t_first_s", "t_last_s", "t_onset_s", "t_onset_fast_s", "t_end_active_s", "n_active_frames",
                "first_speed", "last_speed", "onset_censored", "end_censored", "peak_speed_censored", "axis_anisotropy",
                "last_speed_frac_of_max", "resid_rms_yds"}
SPRINT_FEATURES = ("model_vmax", "model_tau_s", "model_a0", "trk_t10_s", "trk_t20_s", "trk_t40_s", "trk_t10_20_s",
                   "trk_t20_40_s", "speed_at_10", "speed_at_20", "speed_at_40", "dist_to_90pct_vmax_yds", "speed_max_obs")
MODEL_COLUMNS = ("model_vmax", "model_tau_s", "model_a0", "model_t0_rel_s", "dist_to_90pct_vmax_yds")
LONG_COLUMNS = ["nfl_id", "drill", "feature", "draft_year", "position_group", "n_attempts", "median", "mean", "sd",
                "min", "max", "best", "first"]


def make_specs(with_dim_athlete: bool) -> dict[str, DatasetSpec]:
    src = ("phase:combine_frame_kinematics", "phase:combine_kinematic_events", "phase:combine_trajectory_quality",
           *(("gold.dim_athlete",) if with_dim_athlete else ()))
    att_cols = [*ID_COLS, *sorted(registered_names("attempt"))]
    return {
        "attempts": DatasetSpec(
            "combine_attempt_features", "one row per usable Combine attempt", tuple(KEYS), src,
            "Attempt-level speed, acceleration, braking, agility, efficiency and temporal features (active window).",
            column_docs(att_cols, "attempt"), MovementConfig().version),
        "cod": DatasetSpec(
            "combine_cod_events", "one row per change of direction in a usable Combine attempt",
            (*KEYS, "cod_idx"), src, "Merged turn/reversal CODs with approach, exit and recovery measures.",
            column_docs([*KEYS, "seg_idx", *CTX, *sorted(registered_names("cod")), "t_start_s", "t_end_s", "min_speed"],
                        "cod"), MovementConfig().version),
        "sprint": DatasetSpec(
            "combine_sprint_profiles", "one row per usable forty-yard-dash attempt", tuple(KEYS), src,
            "Mono-exponential sprint model and tracked 10/20/40-yd splits (model-start convention).",
            column_docs([*KEYS, *CTX, *SPRINT_COLUMNS, "attempt_order", "quality_tier", "position_group",
                         "onset_censored", "end_censored"], "sprint"), MovementConfig().version),
        "athlete": DatasetSpec(
            "combine_athlete_drill_features", "one row per athlete × canonical drill × feature",
            ("nfl_id", "drill", "feature"), src,
            "Athlete profiles across repeated attempts (long format) — input to reliability and fingerprints.",
            column_docs(LONG_COLUMNS), MovementConfig().version),
    }


def to_long(df: pl.DataFrame, features: list[str], index: list[str]) -> pl.DataFrame:
    return (df.select(*index, *[pl.col(c).cast(pl.Float64) for c in features])
            .unpivot(index=index, on=features, variable_name="feature", value_name="value"))


def athlete_profiles(long: pl.DataFrame) -> pl.DataFrame:
    """Aggregate attempt-level long rows (nfl_id, drill, feature, value, attempt_order, …) per athlete."""
    up = [f for f, v in HIGHER_IS_BETTER.items() if v]
    down = [f for f, v in HIGHER_IS_BETTER.items() if not v]
    v = pl.col("value").filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
    return (long.group_by("nfl_id", "drill", "feature").agg(
                pl.col("draft_year").first(), pl.col("position_group").first(),
                v.len().cast(pl.Int64).alias("n_attempts"), v.median().alias("median"), v.mean().alias("mean"),
                v.std().alias("sd"), v.min().alias("min"), v.max().alias("max"),
                pl.col("value").sort_by("attempt_order").first().alias("first"))
            .with_columns(pl.when(pl.col("feature").is_in(up)).then(pl.col("max"))
                          .when(pl.col("feature").is_in(down)).then(pl.col("min")).alias("best"))
            .filter(pl.col("n_attempts") > 0)
            .select(LONG_COLUMNS).sort("nfl_id", "drill", "feature"))


class CombineMovementBuilder:
    def __init__(self, cfg: Phase01Config, kcfg: KinematicsConfig | None = None, mcfg: MovementConfig | None = None,
                 store: ArtifactStore | None = None):
        self.cfg = cfg
        self.kin = KinematicFeatureBuilder(cfg, kcfg, store or ArtifactStore(cfg, phase=PHASE))
        self.store, self.kcfg = self.kin.store, self.kin.kcfg
        self.mcfg = mcfg or MovementConfig()
        self.has_dim = bool(self.store.refs.files("gold", "dim_athlete"))
        self.specs = make_specs(self.has_dim)
        self._cache: dict[str, pl.DataFrame] | None = None

    def params(self) -> dict[str, Any]:
        return {"movement": self.mcfg.params(), "kinematics": self.kcfg.params(), "dim_athlete": self.has_dim}

    # ------------------------------------------------------------------ inputs
    def _check_upstream(self) -> None:
        p = self.kin.params("combine")
        for spec in (COMBINE_FRAMES, COMBINE_EVENTS, COMBINE_TRAJ):
            if not self.store.dataset_files(spec.name):
                raise FileNotFoundError(f"{spec.name} not built — run `python -m nflbdb.research.kinematic_build "
                                        f"--domain combine` first")
            current, reason = self.store.status(spec, p)
            if not current:
                raise RuntimeError(f"{spec.name} is stale ({reason}) — rebuild M1 before M2")

    def load_inputs(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        self._check_upstream()
        usable = (self.store.load(COMBINE_TRAJ.name)
                  .filter(pl.col("quality_tier").is_in(list(self.mcfg.usable_tiers)))
                  # resid_rms_yds = M1 positional-noise proxy, carried for the cohort processing check
                  .select(*KEYS, "quality_tier", "n_segments", "resid_rms_yds").collect())
        frames = self.store.load(COMBINE_FRAMES.name).join(usable.lazy().select(KEYS), on=KEYS, how="semi").collect()
        events = (self.store.load(COMBINE_EVENTS.name).filter(pl.col("rule") == self.mcfg.event_rule)
                  .join(usable.lazy().select(KEYS), on=KEYS, how="semi").collect())
        return frames, events, usable

    def positions(self) -> pl.DataFrame:
        if not self.has_dim:
            return pl.DataFrame(schema={"nfl_id": pl.Int64, "position_group": pl.String})
        return self.store.refs.scan("gold", "dim_athlete").select("nfl_id", "position_group").collect()

    # ------------------------------------------------------------------ compute
    def compute_from(self, frames: pl.DataFrame, events: pl.DataFrame, usable: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """Pure in-memory M2 pipeline (unit-testable without a warehouse)."""
        m, dt = self.mcfg, self.kcfg.dt_s
        t0 = time.perf_counter()
        win = movement_window(frames, KEYS, m)
        act = active_frames(frames, win, KEYS)
        cod = cod_window_measures(frames, cod_events(events, KEYS, m, CTX), KEYS, m)
        ctx = frames.group_by(KEYS).agg(*[pl.col(c).first() for c in CTX if c in frames.columns],
                                        pl.col("ts").min().alias("start_ts"))
        att = ctx.join(usable, on=KEYS, how="inner")
        for part in (win, temporal_features(act, KEYS, m), speed_accel_features(act, KEYS, m, dt),
                     braking_features(act, events, KEYS, m.event_rule, win), agility_features(act, cod, KEYS, m, dt),
                     path_features(act, KEYS)):
            att = att.join(part, on=KEYS, how="left")
        att = (att.with_columns(
                   pl.col("start_ts").rank("ordinal").over("nfl_id", "drill").cast(pl.Int64).alias("attempt_order"),
                   pl.len().over("nfl_id", "drill").cast(pl.Int64).alias("n_attempts_drill"))
               .join(self.positions(), on="nfl_id", how="left")
               .sort("nfl_id", "drill", "attempt_order"))
        att = att.select(*[c for c in ID_COLS if c in att.columns], pl.exclude(ID_COLS))

        sprint = sprint_profiles(frames, KEYS, m, CTX)
        if sprint.height:
            sprint = (sprint.with_columns([pl.when(pl.col("fit_ok")).then(pl.col(c)).alias(c) for c in MODEL_COLUMNS])
                      .join(att.select(*KEYS, "attempt_order", "quality_tier", "position_group", "onset_censored",
                                       "end_censored"), on=KEYS, how="left"))

        feats = [c for c, t in att.schema.items()
                 if c not in ID_COLS and c not in NOT_FEATURES and t.is_numeric()]
        idx = ["nfl_id", "drill", "draft_year", "position_group", "attempt_order"]
        longs = [to_long(att, feats, idx)]
        if sprint.height:
            longs.append(to_long(sprint, [c for c in SPRINT_FEATURES if c in sprint.columns], idx))
        athlete = athlete_profiles(pl.concat(longs, how="vertical_relaxed"))
        log.info("[m2] combine features: %s attempts, %s CODs, %s sprint attempts, %s athlete×drill×feature rows (%.1fs)",
                 f"{att.height:,}", f"{cod.height:,}", f"{sprint.height:,}", f"{athlete.height:,}", time.perf_counter() - t0)
        return {"attempts": compact(att), "cod": compact(cod), "sprint": compact(sprint), "athlete": compact(athlete)}

    def compute(self) -> dict[str, pl.DataFrame]:
        if self._cache is None:
            self._cache = self.compute_from(*self.load_inputs())
        return self._cache

    def build(self, force: bool = False) -> dict[str, pl.LazyFrame]:
        params = self.params()
        return {part: self.store.build(spec, lambda part=part: self.compute()[part], params=params, force=force)
                for part, spec in self.specs.items()}

    def load(self, part: str, require_current: bool = True) -> pl.LazyFrame:
        spec = self.specs[part]
        return self.store.load(spec.name, spec=spec, require_current=require_current, params=self.params())


def main(argv: list[str] | None = None) -> int:
    from ..data.config import configure_runtime

    ap = argparse.ArgumentParser(description="Phase 03 · M2 Combine movement features")
    ap.add_argument("--force", action="store_true", help="rebuild even if current")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    b = CombineMovementBuilder(cfg, KinematicsConfig.from_tracking(cfg.tracking))
    out = b.build(force=a.force)
    print({k: v.select(pl.len()).collect().item() for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
