"""Data access, census, sequence manifest and the athlete split registry (notebook 00).

Everything here reads PERSISTED datasets; nothing is rebuilt and no NFL tracking frame is loaded. The NFL side is
an aggregate per athlete (DuckDB over the play-participation fact) and, when present, the executed get-off targets.

Units of analysis
    athlete     the independent unit. Splits, bootstrap and every count of "sample size" use athletes.
    sequence    one Combine attempt (event_id, nfl_id, attempt) — several per athlete, never independent.
    frame       one 100 ms sample of a sequence — never a sample size.

Split registry (fixed before any model is fitted)
    outer_fold   repeated K-fold over athletes, stratified by position group × draft cohort. The held-out fold of
                 the PRIMARY (inductive) evaluation: nothing of those athletes — Combine sequences included — may
                 enter encoder fitting, normalisation, selection or target engineering.
    inner_fold   a second, independent fold label. For outer test fold k the development split is taken among the
                 athletes with outer_fold ≠ k, so it never touches the held-out athletes.
    design_role  a fixed stratified subset ("design") on which EDA-driven preprocessing decisions are quantified;
                 the remaining athletes ("confirm") show whether the decision generalises. With repeated K-fold
                 every athlete is held out once, so a decision taken on ALL athletes is transductive by construction;
                 the design subset keeps that exposure explicit and small. No NFL value is involved either way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.config import duckdb_connect
from ...data.frames import query_df
from ...data.tables import LayerRefs
from ...data.utils import get_logger
from ..artifacts import ArtifactStore
from ..combine_nfl_dl.models import make_folds
from ..contracts import TableContract, cast_keys
from ..sprint_rescue import attempts as rescue_attempts
from ..sprint_rescue.config import SCENARIOS
from ..sprint_rescue.stats import seed_for
from ..sprint_rescue.store import RescueStore
from .config import AdvancedConfig

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
KEYS = ["event_id", "nfl_id", "attempt"]
OFFICIAL = ("cmb_forty", "cmb_ten_yd_split", "cmb_combine_weight", "cmb_combine_height")
RESCUE_FLAGS = ("first_speed", "onset_censored", "signal_flag", "structural_flag", "boundary_peak", "fit_ok", *SCENARIOS)

SCHEMAS = TableContract(
    "schema_inventory", "Persisted datasets the programme reads: presence, size and columns.",
    {"dataset": S, "layer": S, "status": S, "n_files": I, "rows": I, "n_columns": I, "columns": S}, key=("dataset",),
    area="dataset")
ATHLETES = TableContract(
    "athlete_registry", "One row per Combine athlete: pre-NFL measurements, tracked sequences and NFL exposure.",
    {"nfl_id": I, "display_name": S, "position_group": S, "draft_year": I, **{c: F for c in OFFICIAL},
     "n_drills_tracked": I, "n_primary_attempts": I, "n_primary_eligible": I, "has_repeat": B, "nfl_any_plays": I,
     "nfl_reg_plays": I, "nfl_reg_games": I, "getoff_obs": I}, key=("nfl_id",), kind="dataset", area="athlete",
    docs={"n_primary_eligible": "forty attempts accepted as sequences (usable tier, long enough)",
          "has_repeat": "n_primary_eligible ≥ 2", "nfl_reg_plays": "tracked regular-season player-plays (0 when none)",
          "getoff_obs": "valid player_get_off observations of the executed DL study (null when not a DL athlete or "
                        "the study outputs are absent)"})
MANIFEST = TableContract(
    "sequence_manifest", "One row per Combine attempt (all drills): identity, length, quality and eligibility.",
    {"seq_id": S, "event_id": S, "nfl_id": I, "attempt": I, "drill": S, "draft_year": I, "position_group": S,
     "attempt_order": I, "n_frames": I, "n_valid": I, "n_segments": I, "longest_valid": I, "duration_s": F,
     "quality_tier": S, "first_speed": F, "onset_censored": B, "signal_flag": B, "structural_flag": B,
     "boundary_peak": B, "fit_ok": B, **{s: B for s in SCENARIOS}, "eligible_primary": B, "exclusion_reason": S},
    key=("seq_id",), kind="dataset", area="Combine attempt (event_id, nfl_id, attempt)",
    docs={"seq_id": "event_id|nfl_id|attempt", "attempt_order": "1 = the athlete's first attempt of the drill in time",
          "longest_valid": "frames of the longest segment with defined derivatives — the sequence the models see",
          "first_speed": "forty only (sprint rescue): derived speed on the first valid frame [yd/s]",
          "eligible_primary": "primary drill, usable quality tier, longest_valid ≥ min_frames",
          "exclusion_reason": "empty when eligible; otherwise other_drill / not_usable_tier / too_short"})
SPLITS = TableContract(
    "split_registry", "Athlete-grouped repeated outer folds, development folds and the design subset.",
    {"nfl_id": I, "repeat": I, "outer_fold": I, "inner_fold": I, "stratum": S, "design_role": S},
    key=("nfl_id", "repeat"), kind="dataset", area="athlete × split repeat",
    docs={"stratum": "position group | draft cohort (the only information the split uses)",
          "design_role": "design = preprocessing decisions are quantified on these athletes · confirm = the rest"})
LEAKAGE = TableContract(
    "leakage_audit", "Checks of the split registry against the sequence manifest.",
    {"check": S, "passed": B, "n_violations": I, "detail": S}, key=("check",), area="check")
AUDIT = TableContract(
    "dataset_audit", "Census of athletes, attempts, frames, quality and NFL outcome availability.",
    {"section": S, "item": S, "metric": S, "value": F, "note": S}, key=("section", "item", "metric"),
    area="section × item × metric")
MISSING = TableContract(
    "missingness_matrix", "Share of Combine athletes lacking each measurement, by position group.",
    {"variable": S, "group": S, "n": I, "share_missing": F}, key=("variable", "group"), area="variable × position group")


class LeakageError(RuntimeError):
    """Athletes are shared between sets that must be disjoint."""


# ============================================================================== inputs
class AdvInputs:
    """Read-only access to upstream datasets. ``athlete_ids`` restricts everything to a subset (smoke mode)."""

    def __init__(self, cfg: Any, acfg: AdvancedConfig, athlete_ids: Sequence[int] | None = None):
        self.cfg, self.acfg = cfg, acfg
        self.refs = LayerRefs(cfg)
        self.p03 = ArtifactStore(cfg, phase="phase_03")
        self.rescue = RescueStore(cfg)
        self.getoff = RescueStore(cfg, namespace="combine_nfl_dl")
        self.athlete_ids = None if athlete_ids is None else sorted(int(i) for i in athlete_ids)

    def _restrict(self, df):
        return df if self.athlete_ids is None else df.filter(pl.col("nfl_id").is_in(self.athlete_ids))

    def frames(self) -> pl.LazyFrame:
        if not self.p03.dataset_files("combine_frame_kinematics"):
            raise FileNotFoundError("gold/phase_03/combine_frame_kinematics is missing — it is built by the Phase 03 "
                                    "kinematic build (see docs/SESSION_STATE.md); this programme does not rebuild it")
        return self._restrict(self.p03.load("combine_frame_kinematics"))

    def frames_sql(self) -> str:
        """DuckDB table expression over the persisted Combine frame kinematics (for ad-hoc census queries)."""
        files = self.p03.dataset_files("combine_frame_kinematics")
        if not files:
            raise FileNotFoundError("gold/phase_03/combine_frame_kinematics is missing")
        return "read_parquet([" + ", ".join(f"'{Path(f).as_posix()}'" for f in files) + "], hive_partitioning = true)"

    def primary_athletes(self) -> list[int]:
        """Athletes with at least one tracked frame of the primary drill (ignores any restriction)."""
        lf = self.p03.load("combine_frame_kinematics").filter(pl.col("drill") == self.acfg.primary_drill)
        return sorted(lf.select(pl.col("nfl_id").cast(pl.Int64)).unique().collect()["nfl_id"].to_list())

    def trajectory_quality(self) -> pl.DataFrame | None:
        if not self.p03.dataset_files("combine_trajectory_quality"):
            return None
        lf = self.p03.load("combine_trajectory_quality")
        return self._restrict(cast_keys(lf.select(*KEYS, "quality_tier").collect()))

    def rescue_attempts(self) -> pl.DataFrame | None:
        c = rescue_attempts.ATTEMPTS
        return self._restrict(self.rescue.read(c)) if self.rescue.exists(c) else None

    def mart(self) -> pl.DataFrame:
        return self._restrict(cast_keys(self.refs.scan("gold", "mart_athlete_research_base").collect()))

    def nfl_exposure(self) -> pl.DataFrame:
        """(nfl_id, season_type, n_plays, n_games): tracked player-plays per athlete, aggregated in DuckDB."""
        con = duckdb_connect(self.cfg)
        try:
            df = query_df(con, f"""
                SELECT CAST(f.nfl_id AS BIGINT) AS nfl_id, coalesce(g.season_type, 'UNKNOWN') AS season_type,
                       count(*) AS n_plays, count(DISTINCT f.game_id) AS n_games
                FROM {self.refs.sql('gold', 'fact_nfl_play_participation')} f
                LEFT JOIN (SELECT game_id, season_type FROM {self.refs.sql('silver', 'games')}) g
                       ON CAST(g.game_id AS BIGINT) = CAST(f.game_id AS BIGINT)
                WHERE f.nfl_id IS NOT NULL
                GROUP BY 1, 2""")
        finally:
            con.close()
        return self._restrict(df.with_columns(pl.col("n_plays").cast(pl.Int64), pl.col("n_games").cast(pl.Int64),
                                              pl.col("season_type").cast(pl.String)))

    def getoff_targets(self) -> pl.DataFrame | None:
        from ..combine_nfl_dl.targets import TARGET_TABLE

        if not self.getoff.exists(TARGET_TABLE):
            return None
        return self._restrict(self.getoff.read(TARGET_TABLE).filter(pl.col("window") == "all").select("nfl_id", "n_obs"))

    def input_files(self) -> list[Path]:
        files = [*self.p03.dataset_files("combine_frame_kinematics"), *self.p03.dataset_files("combine_trajectory_quality"),
                 self.rescue.parquet(rescue_attempts.ATTEMPTS)]
        for layer, name in (("gold", "mart_athlete_research_base"), ("gold", "fact_nfl_play_participation"), ("silver", "games")):
            files += self.refs.files(layer, name)
        return files


def smoke_subset(athlete_ids: Sequence[int], n: int, seed: int) -> list[int]:
    """Deterministic subset of athletes; all attempts of a chosen athlete stay together."""
    ids = np.array(sorted(int(i) for i in athlete_ids))
    if ids.size <= n:
        return ids.tolist()
    return sorted(np.random.default_rng(seed_for(seed, "smoke")).choice(ids, size=n, replace=False).tolist())


# ============================================================================== schema inventory
def schema_inventory(cfg: Any) -> pl.DataFrame:
    """Presence, row count and columns of every dataset the programme reads. A missing dataset is listed as
    MISSING — it is documented, never guessed or rebuilt here."""
    import pyarrow.parquet as pq

    refs, p03 = LayerRefs(cfg), ArtifactStore(cfg, phase="phase_03")
    gold = cfg.paths.gold
    wanted: list[tuple[str, str, list[Path]]] = [(f"{layer}.{name}", layer, refs.files(layer, name)) for layer, name in (
        ("silver", "combine_tracking"), ("silver", "combine_results"), ("silver", "game_tracking"), ("silver", "player_play"),
        ("silver", "plays"), ("silver", "games"), ("gold", "dim_athlete"), ("gold", "fact_combine_attempt"),
        ("gold", "fact_nfl_play_participation"), ("gold", "mart_athlete_research_base"))]
    wanted += [(f"phase_03.{n}", "gold/phase_03", p03.dataset_files(n)) for n in (
        "combine_frame_kinematics", "combine_trajectory_quality", "combine_sprint_profiles", "combine_attempt_features",
        "nfl_player_play_windows", "nfl_trajectory_quality")]
    for area, names in (("phase_04/quality", ("trajectory_quality_flags", "trajectory_features_by_smoothing")),
                        ("sprint_rescue", ("attempt_censoring_diagnostics", "athlete_sprint_features")),
                        ("combine_nfl_dl", ("athlete_targets", "athlete_research_base", "oof_predictions"))):
        wanted += [(f"{area.split('/')[0]}.{n}", f"gold/{area}", [p for p in [gold / area / f"{n}.parquet"] if p.is_file()])
                   for n in names]
    rows = []
    for dataset, layer, files in wanted:
        if not files:
            rows.append({"dataset": dataset, "layer": layer, "status": "MISSING", "n_files": 0, "rows": None,
                         "n_columns": None, "columns": ""})
            continue
        cols = list(pl.read_parquet_schema(files[0]))
        rows.append({"dataset": dataset, "layer": layer, "status": "present", "n_files": len(files),
                     "rows": int(sum(pq.ParquetFile(f).metadata.num_rows for f in files)), "n_columns": len(cols),
                     "columns": ", ".join(cols)})
    return pl.DataFrame(rows, schema=SCHEMAS.columns)


# ============================================================================== sequence manifest
def sequence_manifest(frames: pl.LazyFrame | pl.DataFrame, quality: pl.DataFrame | None, rescue: pl.DataFrame | None,
                      athletes: pl.DataFrame, acfg: AdvancedConfig, rescue_expected: bool = True) -> pl.DataFrame:
    """One row per Combine attempt. ``athletes`` needs (nfl_id, position_group). ``rescue`` = sprint-rescue attempt
    table (forty only); without it the usable tier is taken from ``quality`` (anything but 'reject').
    ``rescue_expected = False`` declares that fallback intended (synthetic fixtures), so it is not warned about;
    production callers keep the default and the warning."""
    f = cast_keys(frames.lazy().select(*KEYS, "drill", "draft_year", "seg_idx", "ts", "t_s", "valid_kin").collect())
    if f.is_empty():
        raise ValueError("no Combine frame for the selected athletes: the sequence manifest would be empty")
    seg = f.group_by(*KEYS, "seg_idx").agg(pl.col("valid_kin").sum().cast(pl.Int64).alias("_v"))
    longest = seg.group_by(KEYS).agg(pl.col("_v").max().alias("longest_valid"), pl.len().cast(pl.Int64).alias("n_segments"))
    m = (f.group_by(KEYS).agg(pl.col("drill").first(), pl.col("draft_year").first().cast(pl.Int64),
                              pl.len().cast(pl.Int64).alias("n_frames"), pl.col("valid_kin").sum().cast(pl.Int64).alias("n_valid"),
                              (pl.col("t_s").max() - pl.col("t_s").min()).cast(pl.Float64).alias("duration_s"),
                              pl.col("ts").min().alias("_start"))
         .join(longest, on=KEYS, how="left")
         .join(athletes.select("nfl_id", pl.col("position_group").cast(pl.String)), on="nfl_id", how="left")
         .with_columns(pl.col("_start").rank("ordinal").over("nfl_id", "drill").cast(pl.Int64).alias("attempt_order"),
                       (pl.col("event_id") + "|" + pl.col("nfl_id").cast(pl.String) + "|"
                        + pl.col("attempt").cast(pl.String)).alias("seq_id")))
    if quality is not None:
        m = m.join(quality.select(*KEYS, pl.col("quality_tier").cast(pl.String)), on=KEYS, how="left")
    else:
        m = m.with_columns(pl.lit(None, dtype=pl.String).alias("quality_tier"))
    if rescue is not None:
        r = rescue.select(*KEYS, pl.lit(True).alias("_usable"), pl.col("attempt_order").alias("_order"),
                          *[c for c in RESCUE_FLAGS if c in rescue.columns])
        m = m.join(r, on=KEYS, how="left").with_columns(pl.coalesce("_order", "attempt_order").alias("attempt_order"),
                                                        pl.col("_usable").fill_null(False))
    else:
        if rescue_expected:
            log.warning("[advanced_movement] sprint-rescue attempt table missing: usable tier taken from quality_tier, "
                        "recording-window flags are null")
        m = m.with_columns((pl.col("quality_tier").fill_null("?") != "reject").alias("_usable"))
    m = m.with_columns([pl.lit(None, dtype=MANIFEST.columns[c]).alias(c) for c in RESCUE_FLAGS if c not in m.columns])
    primary = pl.col("drill") == acfg.primary_drill
    long_enough = pl.col("longest_valid").fill_null(0) >= acfg.min_frames
    m = m.with_columns(
        (primary & pl.col("_usable") & long_enough).alias("eligible_primary"),
        pl.when(~primary).then(pl.lit("other_drill")).when(~pl.col("_usable")).then(pl.lit("not_usable_tier"))
        .when(~long_enough).then(pl.lit("too_short")).otherwise(pl.lit("")).alias("exclusion_reason"))
    out = m.select([pl.col(c).cast(t) for c, t in MANIFEST.columns.items()]).sort("nfl_id", "drill", "attempt_order", "seq_id")
    dup = out.height - out["seq_id"].n_unique()
    if dup:
        raise ValueError(f"{dup} duplicate sequence identifier(s) (event_id|nfl_id|attempt) in the Combine frames")
    return out


# ============================================================================== athlete registry
def athlete_registry(mart: pl.DataFrame, manifest: pl.DataFrame, exposure: pl.DataFrame, getoff: pl.DataFrame | None,
                     acfg: AdvancedConfig) -> pl.DataFrame:
    """Combine athletes (a Combine results row or a tracked sequence), with counts from the other tables."""
    m = mart.filter(pl.col("has_combine_results").fill_null(False) | pl.col("nfl_id").is_in(manifest["nfl_id"].unique().to_list()))
    officials = [(pl.col(c).cast(pl.Float64, strict=False) if c in m.columns else pl.lit(None, dtype=pl.Float64)).alias(c)
                 for c in OFFICIAL]
    base = m.select(pl.col("nfl_id").cast(pl.Int64), pl.col("display_name").cast(pl.String),
                    pl.col("position_group").cast(pl.String), pl.col("cohort").cast(pl.Int64).alias("draft_year"), *officials)
    prim = pl.col("drill") == acfg.primary_drill
    seq = manifest.group_by("nfl_id").agg(pl.col("drill").n_unique().cast(pl.Int64).alias("n_drills_tracked"),
                                          prim.sum().cast(pl.Int64).alias("n_primary_attempts"),
                                          pl.col("eligible_primary").sum().cast(pl.Int64).alias("n_primary_eligible"))
    reg = pl.col("season_type").is_in(list(acfg.season_types))
    nfl = exposure.group_by("nfl_id").agg(pl.col("n_plays").sum().alias("nfl_any_plays"),
                                          pl.col("n_plays").filter(reg).sum().alias("nfl_reg_plays"),
                                          pl.col("n_games").filter(reg).sum().alias("nfl_reg_games"))
    out = base.join(seq, on="nfl_id", how="left").join(nfl, on="nfl_id", how="left")
    out = (out.join(getoff.select("nfl_id", pl.col("n_obs").alias("getoff_obs")), on="nfl_id", how="left")
           if getoff is not None else out.with_columns(pl.lit(None, dtype=pl.Int64).alias("getoff_obs")))
    zero = ["n_drills_tracked", "n_primary_attempts", "n_primary_eligible", "nfl_any_plays", "nfl_reg_plays", "nfl_reg_games"]
    out = out.with_columns([pl.col(c).fill_null(0) for c in zero]).with_columns(
        (pl.col("n_primary_eligible") >= 2).alias("has_repeat"))
    return out.select([pl.col(c).cast(t) for c, t in ATHLETES.columns.items()]).sort("nfl_id")


# ============================================================================== split registry
def design_roles(ids: np.ndarray, strata: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """'design' for about ``fraction`` of the athletes of every stratum (remainders carried across strata)."""
    rng = np.random.default_rng(seed)
    role = np.full(ids.size, "confirm", dtype=object)
    carry = 0.0
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        want = fraction * idx.size + carry
        k = int(np.floor(want + 1e-9))
        carry = want - k
        role[idx[:k]] = "design"
    return role


def build_split_registry(athletes: pl.DataFrame, acfg: AdvancedConfig, seed: int) -> pl.DataFrame:
    """Folds for every athlete with at least one eligible primary sequence. Uses position group and draft cohort
    only — no Combine measurement and no NFL value can influence the split."""
    a = athletes.filter(pl.col("n_primary_eligible") >= 1).sort("nfl_id")
    if a.height < acfg.outer_folds * 2:
        raise ValueError(f"only {a.height} athletes with an eligible sequence: too few for {acfg.outer_folds} outer folds")
    ids = a["nfl_id"].to_numpy()
    strata = (a["position_group"].fill_null("?") + "|" + a["draft_year"].cast(pl.String).fill_null("?")).to_numpy()
    outer = make_folds(strata, acfg.outer_folds, acfg.split_repeats, seed_for(seed, "outer"))
    inner = make_folds(strata, acfg.inner_folds, acfg.split_repeats, seed_for(seed, "inner"))
    role = design_roles(ids, strata, acfg.design_fraction, seed_for(seed, "design"))
    parts = [pl.DataFrame({"nfl_id": ids.astype(np.int64), "repeat": np.full(ids.size, r, dtype=np.int64),
                           "outer_fold": outer[r].astype(np.int64), "inner_fold": inner[r].astype(np.int64),
                           "stratum": strata.astype(str), "design_role": role.astype(str)}, schema=SPLITS.columns)
             for r in range(acfg.split_repeats)]
    return pl.concat(parts).sort("repeat", "nfl_id")


def fold_athletes(splits: pl.DataFrame, repeat: int, outer_fold: int) -> tuple[list[int], list[int]]:
    """(training athletes, held-out athletes) of one outer fold. Raises if they overlap."""
    s = splits.filter(pl.col("repeat") == repeat)
    train = s.filter(pl.col("outer_fold") != outer_fold)["nfl_id"].to_list()
    test = s.filter(pl.col("outer_fold") == outer_fold)["nfl_id"].to_list()
    assert_disjoint(train, test, f"repeat {repeat}, outer fold {outer_fold}")
    return train, test


def inner_athletes(splits: pl.DataFrame, repeat: int, outer_fold: int, inner_fold: int) -> tuple[list[int], list[int]]:
    """(fit athletes, development athletes) INSIDE the training set of an outer fold."""
    s = splits.filter((pl.col("repeat") == repeat) & (pl.col("outer_fold") != outer_fold))
    fit = s.filter(pl.col("inner_fold") != inner_fold)["nfl_id"].to_list()
    dev = s.filter(pl.col("inner_fold") == inner_fold)["nfl_id"].to_list()
    assert_disjoint(fit, dev, f"repeat {repeat}, outer fold {outer_fold}, inner fold {inner_fold}")
    return fit, dev


def assert_disjoint(a: Sequence[int], b: Sequence[int], where: str = "") -> None:
    shared = set(a) & set(b)
    if shared:
        raise LeakageError(f"{len(shared)} athlete(s) appear on both sides of a split ({where}): {sorted(shared)[:5]} …")


def leakage_audit(splits: pl.DataFrame, manifest: pl.DataFrame, acfg: AdvancedConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check: str, violations: int, detail: str) -> None:
        rows.append({"check": check, "passed": violations == 0, "n_violations": int(violations), "detail": detail})

    elig = manifest.filter(pl.col("eligible_primary"))
    add("one_outer_fold_per_athlete_and_repeat", splits.height - splits.select(pl.struct("nfl_id", "repeat").n_unique()).item(),
        "the registry key (nfl_id, repeat) is unique")
    add("every_eligible_sequence_has_a_fold", elig.filter(~pl.col("nfl_id").is_in(splits["nfl_id"].unique().to_list())).height,
        "eligible sequences whose athlete is absent from the registry")
    j = elig.select("seq_id", "nfl_id").join(splits, on="nfl_id", how="inner")
    per = j.group_by("nfl_id", "repeat").agg(pl.col("outer_fold").n_unique().alias("k"))
    add("all_attempts_of_an_athlete_share_a_fold", per.filter(pl.col("k") > 1).height,
        "athlete × repeat combinations whose sequences fall in more than one outer fold")
    sizes = splits.group_by("repeat", "outer_fold").len()
    add("every_outer_fold_is_populated", acfg.split_repeats * acfg.outer_folds - sizes.height,
        f"fold sizes {sizes['len'].min()}–{sizes['len'].max()} athletes")
    empty = 0
    for r in range(acfg.split_repeats):
        for k in range(acfg.outer_folds):
            train = splits.filter((pl.col("repeat") == r) & (pl.col("outer_fold") != k))
            empty += acfg.inner_folds - train["inner_fold"].n_unique()
    add("inner_folds_partition_every_outer_training_set", empty, "missing development folds inside outer training sets")
    roles = splits.group_by("nfl_id").agg(pl.col("design_role").n_unique().alias("k"))
    add("design_role_is_constant_per_athlete", roles.filter(pl.col("k") > 1).height,
        f"{splits.filter(pl.col('design_role') == 'design')['nfl_id'].n_unique()} design athletes of "
        f"{splits['nfl_id'].n_unique()}")
    bad = splits.filter(~pl.col("stratum").str.contains("|", literal=True)).height
    add("split_uses_position_and_cohort_only", bad, "stratum = position group | draft cohort; no measurement, no NFL value")
    return pl.DataFrame(rows, schema=LEAKAGE.columns)


# ============================================================================== census
def dataset_audit(athletes: pl.DataFrame, manifest: pl.DataFrame, acfg: AdvancedConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(section: str, item: Any, metric: str, value: Any, note: str = "") -> None:
        ok = value is not None and np.isfinite(float(value))
        rows.append({"section": section, "item": str(item), "metric": metric, "value": float(value) if ok else None, "note": note})

    prim = manifest.filter(pl.col("drill") == acfg.primary_drill)
    elig = manifest.filter(pl.col("eligible_primary"))
    tracked = athletes.filter(pl.col("n_primary_eligible") >= 1)
    both = tracked.filter(pl.col("cmb_forty").is_not_null() & pl.col("cmb_ten_yd_split").is_not_null())
    # ---- flow: Combine → sequences → NFL (every stage is a subset of the previous one)
    flow = [("01_combine_athletes", athletes, "athletes with a Combine results row or a tracked sequence"),
            ("02_primary_sequence", tracked, f"… with ≥ 1 eligible {acfg.primary_drill} sequence"),
            ("03_official_times", both, "… AND official forty + 10-yard split"),
            ("04_nfl_any", both.filter(pl.col("nfl_any_plays") >= 1), "… AND ≥ 1 tracked NFL play (any season type)"),
            ("05_nfl_reg", both.filter(pl.col("nfl_reg_plays") >= 1), "… AND ≥ 1 tracked regular-season play")]
    flow += [(f"{6 + i:02d}_nfl_reg_ge_{n}", both.filter(pl.col("nfl_reg_plays") >= n), f"… AND ≥ {n} regular-season plays")
             for i, n in enumerate(sorted(acfg.nfl_min_plays_grid))]
    add("repeat", "primary_eligible", "n_athletes_repeat", tracked.filter(pl.col("has_repeat")).height,
        "athletes with ≥ 2 eligible sequences (basis of repeatability and of same-athlete pairs)")
    for name, df, note in flow:
        add("flow", name, "n_athletes", df.height, note)
        for pos, k in df.group_by("position_group").len().sort("position_group").iter_rows():
            add("flow_by_position", f"{name}|{pos}", "n_athletes", k, note)
    # ---- athletes per cohort and position
    for col in ("draft_year", "position_group"):
        for val, k in athletes.group_by(col).len().sort(col).iter_rows():
            add(f"athletes_by_{col}", val, "n_athletes", k)
            add(f"athletes_by_{col}", val, "n_with_primary_sequence", tracked.filter(pl.col(col) == val).height)
            add(f"athletes_by_{col}", val, "n_with_repeat", tracked.filter((pl.col(col) == val) & pl.col("has_repeat")).height)
            add(f"athletes_by_{col}", val, "n_nfl_reg", athletes.filter((pl.col(col) == val) & (pl.col("nfl_reg_plays") >= 1)).height)
    # ---- attempts per athlete, frames per attempt
    for k, n in tracked.group_by("n_primary_eligible").len().sort("n_primary_eligible").iter_rows():
        add("attempts_per_athlete", k, "n_athletes", n, "eligible primary sequences per athlete")
    for label, df in (("primary_all", prim), ("primary_eligible", elig)):
        add("sequences", label, "n_sequences", df.height)
        add("sequences", label, "n_athletes", df["nfl_id"].n_unique() if df.height else 0)
        for col in ("longest_valid", "duration_s"):
            x = df[col].drop_nulls().to_numpy().astype(float)
            for q, name in ((0.05, "p05"), (0.5, "p50"), (0.95, "p95")):
                add("sequences", label, f"{col}_{name}", np.quantile(x, q) if x.size else None)
            add("sequences", label, f"{col}_min", x.min() if x.size else None)
            add("sequences", label, f"{col}_max", x.max() if x.size else None)
    add("sequences", "primary_eligible", "total_valid_frames", elig["longest_valid"].sum() if elig.height else 0,
        "frames are not independent observations")
    # ---- drill coverage, quality tiers, exclusions, flags
    for drill, n_seq, n_ath in (manifest.group_by("drill").agg(pl.len(), pl.col("nfl_id").n_unique()).sort("drill").iter_rows()):
        add("drill_coverage", drill, "n_sequences", n_seq)
        add("drill_coverage", drill, "n_athletes", n_ath)
    for tier, k in prim.group_by("quality_tier").len().sort("quality_tier").iter_rows():
        add("quality_tier_primary", tier, "n_sequences", k)
    for reason, k in prim.group_by("exclusion_reason").len().sort("exclusion_reason").iter_rows():
        add("exclusions_primary", reason or "eligible", "n_sequences", k)
    for flag in ("onset_censored", "signal_flag", "structural_flag", "boundary_peak", "fit_ok", *SCENARIOS):
        x = elig[flag].drop_nulls()
        add("flags_primary_eligible", flag, "share_true", x.mean() if x.len() else None, f"{x.len()} sequences with the flag")
    # ---- NFL outcome availability among athletes with a sequence and both official times
    for n in acfg.nfl_min_plays_grid:
        add("nfl_outcomes", f"reg_plays_ge_{n}", "n_athletes", both.filter(pl.col("nfl_reg_plays") >= n).height,
            "tracked regular-season plays (movement-envelope targets, notebook 07)")
        add("nfl_outcomes", f"getoff_obs_ge_{n}", "n_athletes", both.filter(pl.col("getoff_obs").fill_null(0) >= n).height,
            "valid player_get_off observations (DL only; vendor definition unresolved)")
    return pl.DataFrame(rows, schema=AUDIT.columns)


def missingness_matrix(athletes: pl.DataFrame) -> pl.DataFrame:
    checks = {**{c: pl.col(c).is_null() for c in OFFICIAL}, "primary_sequence": pl.col("n_primary_eligible") < 1,
              "repeat_sequence": ~pl.col("has_repeat"), "nfl_regular_season": pl.col("nfl_reg_plays") < 1,
              "getoff_observations": pl.col("getoff_obs").fill_null(0) < 1}
    rows = []
    for group in ["ALL", *sorted(athletes["position_group"].drop_nulls().unique().to_list())]:
        df = athletes if group == "ALL" else athletes.filter(pl.col("position_group") == group)
        for var, expr in checks.items():
            rows.append({"variable": var, "group": group, "n": df.height,
                         "share_missing": float(df.select(expr.mean()).item()) if df.height else None})
    return pl.DataFrame(rows, schema=MISSING.columns)
