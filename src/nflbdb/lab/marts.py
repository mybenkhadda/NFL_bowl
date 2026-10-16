"""Analytics Lab · dashboard marts (compact, persisted to ``warehouse/gold/phase_lab``).

Design rules
* Inputs are gold dims/facts/marts, silver reference tables and saved report tables. **No
  frame-level scan**: the only frame-derived numbers are the gold facts plus Parquet footer
  metadata (row counts per partition).
* Every mart is a :class:`DatasetSpec` persisted through :class:`LabArtifactStore`
  (manifest, fingerprint reuse, archive-not-overwrite, sidecar ``*.schema.json``). The Phase 01
  catalog picks the folder up automatically as ``plab_<name>`` views (``gold/phase_*`` rule).
* Optional inputs (saved quality results, feasibility matrix, Phase 02 quality flags…) may be
  missing. A missing input yields an EMPTY mart (or NULL columns) plus a row in
  ``availability`` — never invented numbers.
* Key columns never contain NULL (ArtifactStore contract): unknown cohort → ``-1``,
  unknown position → ``'UNKNOWN'``.

Units follow the warehouse: yards, yd/s, yd/s², seconds. ``*_mph`` = yd/s × 2.04545.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import polars as pl

from ..data.config import Phase01Config, duckdb_connect
from ..data.frames import query_column, query_df, query_row
from ..data.tables import LayerRefs
from ..data.utils import get_logger, sql_path, utc_now
from ..research.artifacts import ArtifactStore, DatasetSpec

log = get_logger(__name__)

LAB_VERSION = "1"
LAB_DATASET_DIR = "phase_lab"   # gold/phase_lab → catalogued as plab_* by WarehouseCatalog
YDS_TO_MPH = 2.04545

TRADITIONAL_TESTS = {            # column → unit (combine_results)
    "combine_height": "in", "combine_weight": "lb", "hand_size": "in", "arm_length": "in",
    "wing_span": "in", "ten_yd_split": "s", "forty": "s", "vertical": "in", "broad_jump": "in",
    "three_cone": "s", "short_shuttle": "s", "bench_reps": "reps", "ngs_athleticism_score": "score",
}
REQUIRED_GOLD = ("dim_athlete", "dim_game", "dim_play", "dim_combine_drill", "fact_combine_attempt",
                 "fact_nfl_play_participation", "mart_athlete_nfl_profile")
OPTIONAL_PLAY_CONTEXT = ("quarter", "down", "yards_to_go", "possession_team", "offense_formation",
                         "pass_result", "play_nullified_by_penalty", "play_description")

# Optional saved report tables → (stem, preferred format, empty-table DDL used when missing).
OPTIONAL_REPORTS = {
    "r_quality": ("data_quality_results", "parquet",
                  'check_id VARCHAR, "check" VARCHAR, "table" VARCHAR, columns VARCHAR, result VARCHAR, '
                  "severity VARCHAR, metric VARCHAR, value DOUBLE, denominator DOUBLE, threshold VARCHAR, "
                  "details VARCHAR, recommended_action VARCHAR, mode VARCHAR, seconds DOUBLE"),
    "r_runlog": ("pipeline_run_log", "parquet",
                 "step VARCHAR, started_at VARCHAR, seconds DOUBLE, rss_start_mb DOUBLE, rss_peak_mb DOUBLE, "
                 "rss_end_mb DOUBLE, rows_in BIGINT, rows_out BIGINT, bytes_in BIGINT, bytes_out BIGINT, "
                 "artifacts VARCHAR, warnings VARCHAR, status VARCHAR, error VARCHAR, run_id VARCHAR"),
    "r_intervals": ("tracking_qc_intervals", "parquet",
                    "dt_ms_capped BIGINT, n BIGINT, domain VARCHAR, sample_pct DOUBLE"),
    "r_consistency": ("tracking_qc_consistency", "parquet",
                      "n_frames BIGINT, corr_s DOUBLE, median_abs_diff_s DOUBLE, p95_abs_diff_s DOUBLE, "
                      "corr_dis DOUBLE, median_abs_diff_dis DOUBLE, median_s DOUBLE, median_s_derived DOUBLE, "
                      "domain VARCHAR, sample_pct DOUBLE"),
    # CSV preferred: matching_drills is a ", "-separated string there (Parquet may hold a list).
    "r_feas": ("research_feasibility_matrix", "csv",
               "hypothesis VARCHAR, title VARCHAR, positions VARCHAR, drill_regex VARCHAR, matching_drills VARCHAR, "
               "traditional_baseline VARCHAR, nfl_outcome VARCHAR, n_population BIGINT, n_traditional BIGINT, "
               "n_combine_tracking_drill BIGINT, n_outcome BIGINT, n_matched BIGINT, n_matched_with_baseline BIGINT, "
               "largest_position VARCHAR, largest_position_n_matched BIGINT, feasibility VARCHAR, confounders VARCHAR, "
               "selection_bias VARCHAR, validation VARCHAR, football_interpretation VARCHAR, limitations VARCHAR"),
    "r_feas_pos": ("research_feasibility_by_position", "csv",
                   "position_group VARCHAR, n_population BIGINT, n_traditional BIGINT, n_combine_tracking_drill BIGINT, "
                   "n_outcome BIGINT, n_matched BIGINT, n_matched_with_baseline BIGINT, hypothesis VARCHAR"),
    "r_remediation": ("phase_02/quality_remediation_summary", "csv",
                      'check_id VARCHAR, "check" VARCHAR, "table" VARCHAR, columns VARCHAR, severity VARCHAR, '
                      "result VARCHAR, definition_status VARCHAR, affected_rows_estimate DOUBLE, "
                      "affected_rows_basis VARCHAR, status VARCHAR, root_cause VARCHAR, evidence VARCHAR, "
                      "affected_athletes BIGINT, remediation VARCHAR, exclusion_flag VARCHAR, next_diagnostic VARCHAR"),
}
# Phase 02 flag datasets (gold/phase_02) → (view, empty DDL).
OPTIONAL_FLAGS = {
    "tracking_quality_flags_nfl": ("f_nfl",
                                   "season BIGINT, game_id BIGINT, play_id BIGINT, nfl_id BIGINT, n_frames BIGINT, "
                                   "exclude_kinematics BOOLEAN, exclude_context BOOLEAN, exclusion_reasons VARCHAR"),
    "tracking_quality_flags_combine": ("f_combine",
                                       "event_id VARCHAR, nfl_id BIGINT, attempt BIGINT, n_frames BIGINT, "
                                       "exclude_kinematics BOOLEAN, exclusion_reasons VARCHAR"),
}

CHECK_CATEGORY_SQL = """
    CASE
        WHEN check_id LIKE 'trk:%' AND (metric IN ('n_gaps', 'n_dup_ts', 'n_nonmono', 'n_null_ts'))
            THEN 'Tracking discontinuities'
        WHEN check_id LIKE 'trk:%' THEN 'Kinematic plausibility'
        WHEN check_id LIKE 'null:%' THEN 'Missing identifiers'
        WHEN check_id LIKE 'nullprof:%' THEN 'Completeness'
        WHEN check_id LIKE 'uniq:%' OR check_id LIKE 'dupe:%' THEN 'Uniqueness / duplicates'
        WHEN check_id LIKE 'fk:%' THEN 'Referential integrity'
        WHEN check_id LIKE 'range:%' OR check_id LIKE 'domain:%' THEN 'Value ranges / domains'
        WHEN check_id LIKE 'ts:%' OR check_id LIKE 'date:%' OR check_id LIKE 'season:%' THEN 'Time consistency'
        ELSE 'Other'
    END"""
SEVERITY_RANK_SQL = ("CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 "
                     "WHEN 'LOW' THEN 4 WHEN 'INFO' THEN 5 ELSE 6 END")


def pct_rank(col: str, partition: str, higher_is_better: bool = True) -> str:
    """Percentile in [0, 1] within ``partition`` (1 = best); NULL stays NULL."""
    order = "ASC" if higher_is_better else "DESC"
    return (f"CASE WHEN {col} IS NULL THEN NULL ELSE percent_rank() OVER "
            f"(PARTITION BY {partition}, ({col} IS NULL) ORDER BY {col} {order}) END")


def band(p: str) -> str:
    return (f"CASE WHEN {p} IS NULL THEN 'n/a' WHEN {p} >= 0.9 THEN 'Top 10%' WHEN {p} >= 0.75 THEN '75-90th pct' "
            f"WHEN {p} >= 0.5 THEN '50-75th pct' ELSE 'Below median' END")


def traditional_test_availability_sql(tests: list[str], source: str = "lab_athlete") -> str:
    """Long form (athlete × test) via UNION ALL, aggregated per position × cohort × test.

    Replaces the simplified UNPIVOT with an include-nulls modifier (ParserException in DuckDB 1.5.6). UNION ALL keeps
    NULL results by construction, so ``n_athletes`` counts every athlete and ``n_with_value`` only
    non-null results — identical to INCLUDE NULLS semantics, with no parser-version dependency.
    """
    casts = ", ".join(f"CAST({t} AS DOUBLE) AS {t}" for t in tests)
    long = " UNION ALL ".join(f"SELECT position_group, cohort, '{t}' AS test_name, '{TRADITIONAL_TESTS[t]}' AS unit, "
                              f"{t} AS value FROM a" for t in tests)
    return f"""
        WITH a AS (SELECT position_group, cohort, {casts} FROM {source} WHERE has_combine_results),
             t_long AS ({long})
        SELECT position_group, cohort, test_name, unit, count(*) AS n_athletes,
               count(value) AS n_with_value, count(value) / count(*)::DOUBLE AS share_available,
               median(value) AS median_value
        FROM t_long
        GROUP BY ALL ORDER BY position_group, cohort, test_name
    """


class LabArtifactStore(ArtifactStore):
    """Datasets → ``gold/phase_lab``; CSV → ``reports/tables/analytics_lab``;
    figures → ``reports/figures/analytics_lab``."""

    def __init__(self, cfg: Phase01Config):
        super().__init__(cfg, phase="analytics_lab")
        self.data_dir = cfg.paths.gold / LAB_DATASET_DIR
        self.archive_dir = self.data_dir / "_archive"
        self.manifest_path = self.data_dir / "_manifest.json"


@dataclass
class SourceStatus:
    name: str
    available: bool
    path: str
    note: str
    used_by: str


class LabMartBuilder:
    """Builds every dashboard mart in dependency order. Call :meth:`build_all`."""

    def __init__(self, cfg: Phase01Config, con=None, store: LabArtifactStore | None = None):
        self.cfg = cfg
        self.con = con or duckdb_connect(cfg)
        self.refs = LayerRefs(cfg)
        self.store = store or LabArtifactStore(cfg)
        self.phase02 = ArtifactStore(cfg)          # read-only use: locate Phase 02 flag datasets
        self.sources: list[SourceStatus] = []
        self.optional_sources: dict[str, list[str]] = {}   # view → "file:" source refs
        self.results: list[dict] = []

    # ------------------------------------------------------------------ registration
    def _rel(self, p: Path) -> str:
        return self.cfg.paths.rel(p)

    def _view(self, name: str, expr: str) -> None:
        self.con.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {expr}")

    def columns(self, view: str) -> list[str]:
        return [r[0] for r in self.con.execute(f"DESCRIBE {view}").fetchall()]

    def has_view(self, view: str) -> bool:
        return view in self._registered

    def register_inputs(self) -> None:
        self._registered: set[str] = set()
        missing = [t for t in REQUIRED_GOLD if not self.refs.exists("gold", t)]
        if missing:
            raise FileNotFoundError(f"gold tables missing {missing} — run the Phase 01 pipeline (gold step) first")
        for t in REQUIRED_GOLD:
            self._view(f"g_{t}", self.refs.sql("gold", t))
            self._registered.add(f"g_{t}")
        if not self.refs.exists("silver", "combine_results"):
            raise FileNotFoundError("silver.combine_results missing — run the Phase 01 pipeline (silver step) first")
        self._view("s_combine_results", self.refs.sql("silver", "combine_results"))
        self._registered.add("s_combine_results")
        if self.refs.exists("silver", "player_career_successes"):
            self._view("s_career", self.refs.sql("silver", "player_career_successes"))
            self._registered.add("s_career")
        self._note("silver.player_career_successes", self.refs.exists("silver", "player_career_successes"),
                   self._rel(self.refs.path("silver", "player_career_successes")),
                   "career outcome columns (career_*) in athlete", "Research feasibility")

        for view, (stem, prefer, ddl) in OPTIONAL_REPORTS.items():
            path = self._report_path(stem, prefer)
            if path is not None:
                reader = (f"read_parquet('{sql_path(path)}')" if path.suffix == ".parquet" else
                          f"read_csv('{sql_path(path)}', header = true, auto_detect = true, sample_size = -1)")
                self._view(view, reader)
                self.optional_sources[view] = [f"file:{self._rel(path)}"]
            else:
                self.con.execute(f"CREATE OR REPLACE TEMP TABLE {view} ({ddl})")
                self.optional_sources[view] = []
            self._registered.add(view)
            self._note(f"reports/tables/{stem}", path is not None,
                       self._rel(path) if path else f"reports/tables/{stem}.{prefer}",
                       "saved Phase 01/02 report table", "Data quality / Research feasibility")

        for name, (view, ddl) in OPTIONAL_FLAGS.items():
            files = self.phase02.dataset_files(name)
            if files:
                listing = ", ".join(f"'{sql_path(f)}'" for f in files)
                self._view(view, f"read_parquet([{listing}], hive_partitioning = true, union_by_name = true)")
                self.optional_sources[view] = [f"file:{self._rel(f)}" for f in files]
            else:
                self.con.execute(f"CREATE OR REPLACE TEMP TABLE {view} ({ddl})")
                self.optional_sources[view] = []
            self._registered.add(view)
            self._note(f"gold/phase_02/{name}", bool(files),
                       self._rel(files[0]) if files else f"warehouse/gold/phase_02/{name}",
                       "Phase 02 quality flags — run `python -m nflbdb.research.quality` to build",
                       "Data quality (affected counts), Combine/NFL flag columns")

    def _report_path(self, stem: str, prefer: str) -> Path | None:
        base = self.cfg.paths.tables / stem
        order = (".parquet", ".csv") if prefer == "parquet" else (".csv", ".parquet")
        for suffix in order:
            p = base.with_suffix(suffix)
            if p.is_file():
                return p
        return None

    def _note(self, name: str, available: bool, path: str, note: str, used_by: str) -> None:
        self.sources.append(SourceStatus(name, available, path, note, used_by))

    def flags_available(self, view: str) -> bool:
        return bool(self.optional_sources.get(view))

    # ------------------------------------------------------------------ build helper
    def spec(self, name: str, grain: str, key: tuple[str, ...], sources: tuple[str, ...], description: str,
             columns: dict[str, str] | None = None, optional: tuple[str, ...] = (),
             partition_by: str | None = None) -> DatasetSpec:
        extra = tuple(s for v in optional for s in self.optional_sources.get(v, []))
        return DatasetSpec(name=name, grain=grain, key=key, sources=tuple(sources) + extra,
                           description=description, columns=columns or {}, version=LAB_VERSION,
                           partition_by=partition_by)

    def mart(self, spec: DatasetSpec, sql_or_builder: str | Callable[[], pl.DataFrame],
             force: bool = False) -> None:
        builder = (lambda: query_df(self.con, sql_or_builder)) if isinstance(sql_or_builder, str) else sql_or_builder
        current, reason = self.store.status(spec, {"lab_version": LAB_VERSION})
        self.store.build(spec, builder, params={"lab_version": LAB_VERSION}, force=force)
        files = self.store.dataset_files(spec.name)
        listing = ", ".join(f"'{sql_path(f)}'" for f in files)
        self._view(f"lab_{spec.name}", f"read_parquet([{listing}], hive_partitioning = true)")
        self._registered.add(f"lab_{spec.name}")
        self.results.append({"mart": spec.name, "action": "reused" if current and not force else "built",
                             "reason": "current" if current and not force else ("forced" if force else reason)})

    # ================================================================== athlete grain
    def build_athlete(self, force: bool) -> None:
        cr_cols = self.columns("s_combine_results")
        tests = [t for t in TRADITIONAL_TESTS if t in cr_cols]
        test_sel = "".join(f", arg_max({t}, draft_year) AS {t}" for t in tests)
        career_join, career_sel = "", ""
        if self.has_view("s_career"):
            ccols = [c for c in self.columns("s_career") if c not in ("nfl_id", "_src_file", "_src_row")]
            career_sel = "".join(f', ca."{c}" AS "{c if c.startswith("career_") else "career_" + c}"' for c in ccols)
            career_join = "LEFT JOIN s_career ca USING (nfl_id)"
        forty_pct = pct_rank("forty", "position_group", higher_is_better=False) if "forty" in tests else "NULL::DOUBLE"
        sql = f"""
            WITH cr AS (SELECT nfl_id, max(draft_year) AS combine_year{test_sel}
                        FROM s_combine_results GROUP BY nfl_id),
                 ct AS (SELECT nfl_id, count(DISTINCT drill_name) AS n_combine_drills, count(*) AS n_combine_attempts,
                               max(max_s) AS combine_best_max_s, max(max_a) AS combine_best_max_a
                        FROM g_fact_combine_attempt GROUP BY nfl_id),
                 nf AS (SELECT nfl_id, count(*) AS nfl_n_seasons, sum(n_games) AS nfl_n_games,
                               sum(n_plays_tracked) AS nfl_n_plays_tracked, min(season) AS nfl_first_season,
                               max(p90_play_max_s) AS nfl_best_season_p90_max_s
                        FROM g_mart_athlete_nfl_profile GROUP BY nfl_id),
                 base AS (
                    SELECT d.nfl_id, d.display_name, coalesce(d.position_group, 'UNKNOWN') AS position_group,
                           coalesce(d.cohort, -1) AS cohort, d.nfl_position, d.combine_position, d.college_name,
                           d.draft_round, d.draft_overall_pick,
                           d.has_combine_results, d.has_combine_tracking, d.has_nfl_tracking, d.has_player_play,
                           d.has_career, (d.has_combine_tracking AND d.has_nfl_tracking) AS is_matched,
                           cr.* EXCLUDE (nfl_id),
                           coalesce(ct.n_combine_drills, 0) AS n_combine_drills,
                           coalesce(ct.n_combine_attempts, 0) AS n_combine_attempts,
                           ct.combine_best_max_s, ct.combine_best_max_a,
                           coalesce(nf.nfl_n_seasons, 0) AS nfl_n_seasons, coalesce(nf.nfl_n_games, 0) AS nfl_n_games,
                           coalesce(nf.nfl_n_plays_tracked, 0) AS nfl_n_plays_tracked, nf.nfl_first_season,
                           nf.nfl_best_season_p90_max_s{career_sel}
                    FROM g_dim_athlete d
                    LEFT JOIN cr USING (nfl_id) LEFT JOIN ct USING (nfl_id) LEFT JOIN nf USING (nfl_id)
                    {career_join})
            SELECT *, {band('forty_pct_in_position')} AS forty_band
            FROM (SELECT *, {forty_pct} AS forty_pct_in_position,
                         {pct_rank('combine_best_max_s', 'position_group')} AS combine_max_s_pct_in_position
                  FROM base)
            ORDER BY nfl_id
        """
        cols = {"nfl_id": "athlete id", "display_name": "name (display only, never a join key)",
                "position_group": "Combine position group (fallback: NFL position; 'UNKNOWN')",
                "cohort": "Combine / draft year (-1 = unknown)", "is_matched": "Combine tracking AND NFL tracking",
                "combine_year": "latest Combine year in combine_results",
                **{t: f"traditional Combine result ({u})" for t, u in TRADITIONAL_TESTS.items()},
                "n_combine_drills": "distinct tracked drills", "n_combine_attempts": "tracked Combine attempts",
                "combine_best_max_s": "best tracked attempt max speed, any drill (yd/s)",
                "combine_best_max_a": "best tracked attempt max acceleration, any drill (yd/s²)",
                "nfl_n_seasons": "seasons with NFL tracking", "nfl_n_games": "tracked games",
                "nfl_n_plays_tracked": "tracked player-plays", "nfl_first_season": "first tracked NFL season",
                "nfl_best_season_p90_max_s": "max over seasons of p90 play max speed (yd/s)",
                "forty_pct_in_position": "40-yd percentile within position (1 = fastest)",
                "forty_band": "40-yd percentile band within position",
                "combine_max_s_pct_in_position": "best tracked speed percentile within position (1 = fastest)"}
        self.mart(self.spec("athlete", "one row per athlete", ("nfl_id",),
                            ("gold.dim_athlete", "silver.combine_results", "gold.fact_combine_attempt",
                             "gold.mart_athlete_nfl_profile") + (("silver.player_career_successes",)
                                                                if self.has_view("s_career") else ()),
                            "Athlete profile: biography, source flags, traditional Combine tests, tracked "
                            "Combine/NFL exposure, career outcomes (career_* = OUTCOMES, never predictors).", cols),
                  sql, force)

    # ================================================================== Combine
    def build_combine_attempt(self, force: bool) -> None:
        flags = self.flags_available("f_combine")
        have = set(self.columns("lab_athlete"))
        trad = ", ".join(f"a.{c}" if c in have else f"NULL::DOUBLE AS {c}"
                         for c in ("forty", "ten_yd_split", "combine_height", "combine_weight"))
        sql = f"""
            WITH f AS (
                SELECT f.event_id, f.nfl_id, f.attempt, coalesce(f.draft_year, -1) AS cohort,
                       coalesce(f.drill_type, 'UNKNOWN') AS drill_type, coalesce(f.drill_name, 'UNKNOWN') AS drill_name,
                       a.display_name, coalesce(a.position_group, 'UNKNOWN') AS position_group,
                       row_number() OVER (PARTITION BY f.nfl_id, f.drill_name ORDER BY f.start_ts, f.event_id, f.attempt)
                           AS attempt_seq,
                       count(*) OVER (PARTITION BY f.nfl_id, f.drill_name) AS n_attempts_athlete_drill,
                       f.n_frames, f.start_ts, f.duration_s, f.path_length_yds, f.net_displacement_yds,
                       f.max_s, f.max_s * {YDS_TO_MPH} AS max_s_mph, f.mean_s, f.max_a, f.time_to_max_s_s,
                       f.n_off_grid_frames, f.n_drill_names, {trad},
                       fl.exclude_kinematics, fl.exclusion_reasons, {'TRUE' if flags else 'FALSE'} AS quality_flags_available
                FROM g_fact_combine_attempt f
                LEFT JOIN lab_athlete a USING (nfl_id)
                LEFT JOIN f_combine fl ON fl.event_id = f.event_id AND fl.nfl_id = f.nfl_id AND fl.attempt = f.attempt)
            SELECT *, {band('max_s_pct_in_drill')} AS max_s_band
            FROM (SELECT *, {pct_rank('max_s', 'drill_name')} AS max_s_pct_in_drill FROM f)
            ORDER BY nfl_id, drill_name, attempt_seq
        """
        cols = {"event_id": "Combine event", "attempt": "attempt number (source)",
                "attempt_seq": "chronological attempt index within athlete × drill",
                "n_attempts_athlete_drill": "attempts by this athlete in this drill",
                "duration_s": "attempt duration (s)", "path_length_yds": "sum of per-frame distance (yd)",
                "net_displacement_yds": "start→end straight-line distance (yd)",
                "max_s": "max vendor speed (yd/s)", "max_s_mph": "max speed (mph)", "mean_s": "mean speed (yd/s)",
                "max_a": "max vendor acceleration magnitude (yd/s²)",
                "time_to_max_s_s": "time from attempt start to max speed (s)",
                "max_s_pct_in_drill": "max-speed percentile within drill (1 = fastest)",
                "max_s_band": "max-speed percentile band within drill",
                "exclude_kinematics": "Phase 02 flag (NULL if flags not built)",
                "quality_flags_available": "TRUE if Phase 02 Combine flags were joined"}
        self.mart(self.spec("combine_attempt", "one row per athlete × drill attempt", ("event_id", "nfl_id", "attempt"),
                            ("gold.fact_combine_attempt", "phase:athlete"),
                            "Attempt-level Combine kinematics with athlete context, within-drill percentiles "
                            "and (if built) Phase 02 quality flags.", cols, optional=("f_combine",)), sql, force)

    def build_combine_athlete_drill(self, force: bool) -> None:
        sql = f"""
            WITH g AS (
                SELECT nfl_id, drill_type, drill_name, any_value(display_name) AS display_name,
                       any_value(position_group) AS position_group, any_value(cohort) AS cohort,
                       count(*) AS n_attempts, max(max_s) AS best_max_s, avg(max_s) AS mean_max_s,
                       stddev_samp(max_s) AS sd_max_s, stddev_samp(max_s) / nullif(avg(max_s), 0) AS cv_max_s,
                       max(max_s) - min(max_s) AS range_max_s, max(max_a) AS best_max_a,
                       stddev_samp(max_a) AS sd_max_a, min(duration_s) AS min_duration_s,
                       median(duration_s) AS median_duration_s, min(time_to_max_s_s) AS min_time_to_max_s_s,
                       median(path_length_yds) AS median_path_length_yds,
                       count_if(coalesce(exclude_kinematics, FALSE)) AS n_flagged_attempts
                FROM lab_combine_attempt GROUP BY nfl_id, drill_type, drill_name)
            SELECT *, {band('best_max_s_pct_in_drill')} AS max_s_band
            FROM (SELECT *, {pct_rank('best_max_s', 'drill_name')} AS best_max_s_pct_in_drill FROM g)
            ORDER BY nfl_id, drill_name
        """
        cols = {"n_attempts": "attempts in this drill", "best_max_s": "best attempt max speed (yd/s)",
                "mean_max_s": "mean of attempt max speeds (yd/s)", "sd_max_s": "within-athlete SD of max speed (yd/s)",
                "cv_max_s": "within-athlete coefficient of variation of max speed (NULL if < 2 attempts)",
                "range_max_s": "max − min attempt max speed (yd/s)", "best_max_a": "best attempt max accel (yd/s²)",
                "min_time_to_max_s_s": "fastest time to max speed (s)",
                "n_flagged_attempts": "attempts flagged exclude_kinematics (0 if flags not built)",
                "max_s_band": "best-max-speed percentile band within drill"}
        self.mart(self.spec("combine_athlete_drill", "one row per athlete × drill", ("nfl_id", "drill_type", "drill_name"),
                            ("phase:combine_attempt",),
                            "Per-athlete drill summary incl. within-athlete attempt variability. Flagged attempts are "
                            "counted (n_flagged_attempts) but NOT removed.", cols), sql, force)

    def build_drill_participation(self, force: bool) -> None:
        sql = """
            SELECT drill_type, drill_name, position_group, cohort, count(DISTINCT nfl_id) AS n_athletes,
                   count(*) AS n_attempts, median(max_s) AS median_max_s, median(duration_s) AS median_duration_s
            FROM lab_combine_attempt GROUP BY ALL ORDER BY drill_name, position_group, cohort
        """
        self.mart(self.spec("drill_participation", "one row per drill × position group × cohort",
                            ("drill_type", "drill_name", "position_group", "cohort"), ("phase:combine_attempt",),
                            "Tracked Combine participation per drill.",
                            {"n_athletes": "distinct athletes (additive across position/cohort)",
                             "n_attempts": "attempts"}), sql, force)

    def build_test_availability(self, force: bool) -> None:
        tests = [t for t in TRADITIONAL_TESTS if t in self.columns("lab_athlete")]
        if not tests:
            sql = ("SELECT NULL::VARCHAR AS position_group, NULL::BIGINT AS cohort, NULL::VARCHAR AS test_name, "
                   "NULL::VARCHAR AS unit, 0::BIGINT AS n_athletes, 0::BIGINT AS n_with_value, "
                   "NULL::DOUBLE AS share_available, NULL::DOUBLE AS median_value WHERE FALSE")
        else:
            sql = traditional_test_availability_sql(tests)
        self.mart(self.spec("test_availability", "one row per position group × cohort × traditional test",
                            ("position_group", "cohort", "test_name"), ("phase:athlete",),
                            "Availability of traditional Combine tests among athletes with combine_results.",
                            {"share_available": "share of athletes with a non-null result",
                             "median_value": "median result (see unit)"}), sql, force)

    # ================================================================== coverage / linkage
    def build_coverage_matrix(self, force: bool) -> None:
        sql = """
            -- coalesce: plain UNPIVOT drops NULL values, which would shrink n_athletes per source.
            WITH a AS (SELECT position_group, cohort,
                              coalesce(has_combine_results, FALSE) AS has_combine_results,
                              coalesce(has_combine_tracking, FALSE) AS has_combine_tracking,
                              coalesce(has_nfl_tracking, FALSE) AS has_nfl_tracking,
                              coalesce(has_player_play, FALSE) AS has_player_play,
                              coalesce(has_career, FALSE) AS has_career,
                              coalesce(is_matched, FALSE) AS is_matched FROM lab_athlete)
            SELECT position_group, cohort, regexp_replace(source, '^has_', '') AS source,
                   CASE source WHEN 'has_combine_results' THEN 1 WHEN 'has_combine_tracking' THEN 2
                               WHEN 'has_nfl_tracking' THEN 3 WHEN 'has_player_play' THEN 4 WHEN 'has_career' THEN 5
                               ELSE 6 END AS source_order,
                   count(*) AS n_athletes, count_if(flag) AS n_with_source, avg(flag::INTEGER) AS share_with_source
            FROM (UNPIVOT a ON has_combine_results, has_combine_tracking, has_nfl_tracking, has_player_play,
                               has_career, is_matched INTO NAME source VALUE flag)
            GROUP BY ALL ORDER BY position_group, cohort, source_order
        """
        self.mart(self.spec("coverage_matrix", "one row per position group × cohort × source",
                            ("position_group", "cohort", "source"), ("phase:athlete",),
                            "Dataset coverage heatmap input (source presence per athlete group).",
                            {"source": "combine_results | combine_tracking | nfl_tracking | player_play | career | "
                                       "is_matched", "share_with_source": "n_with_source / n_athletes"}), sql, force)

    def build_linkage_funnel(self, force: bool) -> None:
        stages = [(1, "All athletes", "TRUE"), (2, "Combine results", "has_combine_results"),
                  (3, "Combine tracking", "has_combine_tracking"), (4, "NFL tracking", "has_nfl_tracking"),
                  (5, "Matched (Combine + NFL tracking)", "is_matched"),
                  (6, "Matched + play context", "is_matched AND has_player_play"),
                  (7, "Matched + career outcomes", "is_matched AND has_career")]
        parts = [f"SELECT position_group, cohort, {o} AS stage_order, '{o}. {name}' AS stage, "
                 f"count_if({cond}) AS n_athletes FROM lab_athlete GROUP BY position_group, cohort"
                 for o, name, cond in stages]
        sql = " UNION ALL ".join(parts) + " ORDER BY position_group, cohort, stage_order"
        self.mart(self.spec("linkage_funnel", "one row per position group × cohort × stage",
                            ("position_group", "cohort", "stage_order"), ("phase:athlete",),
                            "Exact nfl_id linkage funnel (stages are nested).",
                            {"stage": "'<order>. <label>' (label sorts in funnel order)"}), sql, force)

    # ================================================================== NFL
    def build_season_volume(self, force: bool) -> None:
        sql = """
            WITH g AS (SELECT season, season_type, count(*) AS n_games FROM g_dim_game GROUP BY ALL),
                 p AS (SELECT g.season, g.season_type, count(DISTINCT f.game_id) AS n_games_tracked,
                              count(DISTINCT (f.game_id, f.play_id)) AS n_plays_tracked, count(*) AS n_player_plays,
                              count(DISTINCT f.nfl_id) AS n_athletes_tracked, sum(f.n_frames) AS n_player_frames,
                              sum(f.distance_yds) AS total_distance_yds
                       FROM g_fact_nfl_play_participation f JOIN g_dim_game g ON g.game_id = f.game_id
                       GROUP BY ALL)
            SELECT g.season, g.season_type, g.n_games, coalesce(p.n_games_tracked, 0) AS n_games_tracked,
                   coalesce(p.n_plays_tracked, 0) AS n_plays_tracked, coalesce(p.n_player_plays, 0) AS n_player_plays,
                   coalesce(p.n_athletes_tracked, 0) AS n_athletes_tracked,
                   coalesce(p.n_player_frames, 0) AS n_player_frames, p.total_distance_yds
            FROM g LEFT JOIN p USING (season, season_type) ORDER BY season, season_type
        """
        self.mart(self.spec("season_volume", "one row per season × season_type", ("season", "season_type"),
                            ("gold.dim_game", "gold.fact_nfl_play_participation"),
                            "NFL tracking volume per season (player frames exclude football rows).",
                            {"n_games": "games in schedule", "n_games_tracked": "games with ≥1 tracked athlete",
                             "n_player_frames": "athlete frames (football excluded)"}), sql, force)

    def build_tracking_partition(self, force: bool) -> None:
        import pyarrow.parquet as pq

        def builder() -> pl.DataFrame:
            rows = []
            for domain, table, key in (("nfl", "game_tracking", "season"), ("combine", "combine_tracking", "draft_year")):
                for f in self.refs.files("silver", table):
                    value = f.parent.name.split("=")[-1] if "=" in f.parent.name else ""
                    num = int(value) if value.lstrip("-").isdigit() else None
                    rows.append({"domain": domain, "partition_key": key, "partition_value": value or "none",
                                 "file": self._rel(f), "season": num if domain == "nfl" else None,
                                 "cohort": num if domain == "combine" else None,
                                 "n_rows": pq.ParquetFile(f).metadata.num_rows, "n_bytes": f.stat().st_size})
            schema = {"domain": pl.String, "partition_key": pl.String, "partition_value": pl.String, "file": pl.String,
                      "season": pl.Int64, "cohort": pl.Int64, "n_rows": pl.Int64, "n_bytes": pl.Int64}
            return pl.DataFrame(rows, schema=schema)

        self.mart(self.spec("tracking_partition", "one row per silver tracking Parquet file", ("file",),
                            ("silver.game_tracking", "silver.combine_tracking"),
                            "Frame-row counts from Parquet footers (no scan). NFL rows include football rows.",
                            {"n_rows": "rows in file (all entities)", "n_bytes": "file size"}), builder, force)

    def _play_context(self) -> list[str]:
        have = set(self.columns("g_dim_play"))
        return [c for c in OPTIONAL_PLAY_CONTEXT if c in have]

    def build_nfl_game(self, force: bool) -> None:
        sql = """
            SELECT g.game_id, g.season, g.season_type, g.week, g.game_date, g.home_team_abbr, g.visitor_team_abbr,
                   concat(g.season, ' W', lpad(CAST(coalesce(g.week, 0) AS VARCHAR), 2, '0'), ' ',
                          coalesce(g.visitor_team_abbr, '?'), '@', coalesce(g.home_team_abbr, '?'), ' · ', g.game_id) AS game_label,
                   coalesce(t.n_plays_tracked, 0) AS n_plays_tracked, coalesce(t.n_athletes_tracked, 0) AS n_athletes_tracked,
                   coalesce(t.n_player_plays, 0) AS n_player_plays, coalesce(t.n_player_frames, 0) AS n_player_frames
            FROM g_dim_game g
            LEFT JOIN (SELECT game_id, count(DISTINCT play_id) AS n_plays_tracked, count(DISTINCT nfl_id) AS n_athletes_tracked,
                              count(*) AS n_player_plays, sum(n_frames) AS n_player_frames
                       FROM g_fact_nfl_play_participation GROUP BY game_id) t USING (game_id)
            ORDER BY g.game_id
        """
        self.mart(self.spec("nfl_game", "one row per game", ("game_id",),
                            ("gold.dim_game", "gold.fact_nfl_play_participation"),
                            "Game list with tracked-athlete volume.", {"game_label": "'<season> W<week> AWAY@HOME · id'"}),
                  sql, force)

    def build_nfl_play(self, force: bool) -> None:
        ctx = self._play_context()
        sel = "".join(f", p.{c}" if c != "play_description" else ", left(p.play_description, 160) AS play_description"
                      for c in ctx)
        q = "coalesce('Q' || CAST(CAST(p.quarter AS INTEGER) AS VARCHAR), '')" if "quarter" in ctx else "''"
        dd = ("coalesce(' ' || CAST(CAST(p.down AS INTEGER) AS VARCHAR) || '&' || CAST(CAST(p.yards_to_go AS INTEGER) AS VARCHAR), '')"
              if {"down", "yards_to_go"} <= set(ctx) else "''")
        sql = f"""
            SELECT t.game_id, t.play_id, g.season, g.week, g.game_label{sel},
                   concat(t.game_id, '-', t.play_id, ' ', {q}, {dd}) AS play_label,
                   t.n_athletes_tracked, t.n_player_frames, t.max_play_max_s
            FROM (SELECT game_id, play_id, count(*) AS n_athletes_tracked, sum(n_frames) AS n_player_frames,
                         max(max_s) AS max_play_max_s
                  FROM g_fact_nfl_play_participation GROUP BY game_id, play_id) t
            LEFT JOIN g_dim_play p ON p.game_id = t.game_id AND p.play_id = t.play_id
            LEFT JOIN lab_nfl_game g ON g.game_id = t.game_id
            ORDER BY t.game_id, t.play_id
        """
        self.mart(self.spec("nfl_play", "one row per tracked play", ("game_id", "play_id"),
                            ("gold.fact_nfl_play_participation", "gold.dim_play", "phase:nfl_game"),
                            "Tracked plays with context (picker source for the trajectory explorer).",
                            {"play_label": "'<game>-<play> Q<quarter> <down>&<distance>'",
                             "n_athletes_tracked": "dataset athletes tracked on the play (not all 22 players)"}),
                  sql, force)

    def build_nfl_player_play(self, force: bool) -> None:
        ctx = [c for c in self._play_context() if c != "play_description"]
        fact_cols = set(self.columns("g_fact_nfl_play_participation"))
        role = [c for c in ("team_abbr", "lined_up_position", "in_player_play") if c in fact_cols]
        offense = ("(f.team_abbr = p.possession_team) AS is_offense, "
                   if {"team_abbr"} <= fact_cols and "possession_team" in ctx else "NULL::BOOLEAN AS is_offense, ")
        flags = self.flags_available("f_nfl")
        sql = f"""
            SELECT f.season, f.game_id, f.play_id, f.nfl_id, a.display_name,
                   coalesce(a.position_group, 'UNKNOWN') AS position_group, a.cohort, a.is_matched,
                   g.week, g.season_type, g.game_date, g.game_label, concat(f.game_id, '-', f.play_id) AS play_label,
                   {''.join(f'f.{c}, ' for c in role)}{''.join(f'p.{c}, ' for c in ctx)}{offense}
                   f.n_frames, f.tracked_duration_s, f.distance_yds, f.max_s, f.max_s * {YDS_TO_MPH} AS max_s_mph,
                   f.mean_s, f.p95_s, f.max_a, f.p95_a, f.x_std_range_yds,
                   fl.exclude_kinematics, fl.exclude_context, fl.exclusion_reasons,
                   {'TRUE' if flags else 'FALSE'} AS quality_flags_available
            FROM g_fact_nfl_play_participation f
            LEFT JOIN lab_athlete a USING (nfl_id)
            LEFT JOIN lab_nfl_game g ON g.game_id = f.game_id
            LEFT JOIN g_dim_play p ON p.game_id = f.game_id AND p.play_id = f.play_id
            LEFT JOIN f_nfl fl ON fl.game_id = f.game_id AND fl.play_id = f.play_id AND fl.nfl_id = f.nfl_id
            ORDER BY f.season, f.game_id, f.play_id, f.nfl_id
        """
        cols = {"season": "NFL season (partition)", "n_frames": "tracked frames", "tracked_duration_s": "tracked window (s)",
                "distance_yds": "distance travelled (yd)", "max_s": "max speed (yd/s)", "p95_s": "p95 speed (yd/s)",
                "max_a": "max accel (yd/s²)", "p95_a": "p95 accel (yd/s²)",
                "x_std_range_yds": "range of standardised x (yd)", "is_offense": "team_abbr = possession_team",
                "exclude_kinematics": "Phase 02 flag (NULL if flags not built)"}
        self.mart(self.spec("nfl_player_play", "one row per tracked athlete × play", ("game_id", "play_id", "nfl_id"),
                            ("gold.fact_nfl_play_participation", "gold.dim_play", "phase:athlete", "phase:nfl_game"),
                            "Player-play movement summaries (whole tracked window, pre-snap included) with context.",
                            cols, optional=("f_nfl",), partition_by="season"), sql, force)

    def build_nfl_athlete_season(self, force: bool) -> None:
        sql = """
            SELECT m.*, a.display_name, coalesce(a.position_group, 'UNKNOWN') AS position_group, a.cohort,
                   a.is_matched, m.season - a.cohort AS seasons_since_draft
            FROM g_mart_athlete_nfl_profile m LEFT JOIN lab_athlete a USING (nfl_id)
            ORDER BY m.nfl_id, m.season
        """
        self.mart(self.spec("nfl_athlete_season", "one row per athlete × season", ("nfl_id", "season"),
                            ("gold.mart_athlete_nfl_profile", "phase:athlete"),
                            "Season exposure + movement distribution per athlete.",
                            {"seasons_since_draft": "season − cohort (0 = rookie season; invalid if cohort = -1)"}),
                  sql, force)

    def build_nfl_speed_distribution(self, force: bool) -> None:
        sql = """
            WITH v AS (SELECT season, position_group, 'max_s' AS metric, floor(max_s / 0.25) * 0.25 AS bin_start
                       FROM lab_nfl_player_play WHERE max_s IS NOT NULL
                       UNION ALL
                       SELECT season, position_group, 'max_a', floor(max_a / 0.25) * 0.25
                       FROM lab_nfl_player_play WHERE max_a IS NOT NULL)
            SELECT season, position_group, metric, bin_start, bin_start + 0.25 AS bin_end, count(*) AS n_player_plays
            FROM v GROUP BY ALL ORDER BY metric, season, position_group, bin_start
        """
        self.mart(self.spec("nfl_speed_distribution", "one row per season × position × metric × 0.25-unit bin",
                            ("season", "position_group", "metric", "bin_start"), ("phase:nfl_player_play",),
                            "Pre-binned histogram of play max speed (yd/s) and max accel (yd/s²) — keeps "
                            "dashboards from pulling 315k rows.", {"metric": "max_s | max_a"}), sql, force)

    # ================================================================== quality
    def build_quality_check(self, force: bool) -> None:
        rem = ("LEFT JOIN (SELECT check_id, any_value(definition_status) AS definition_status, any_value(status) AS remediation_status, "
               "any_value(root_cause) AS root_cause, any_value(next_diagnostic) AS next_diagnostic "
               "FROM r_remediation GROUP BY check_id) rm USING (check_id)")
        mtime = self._mtime_iso("r_quality")
        sql = f"""
            SELECT row_number() OVER (ORDER BY {SEVERITY_RANK_SQL}, q.check_id) AS check_seq,
                   q.check_id, q."check" AS check_name, q."table" AS source_table, q.columns, q.result, q.severity,
                   {SEVERITY_RANK_SQL} AS severity_rank,
                   CAST({SEVERITY_RANK_SQL} AS VARCHAR) || ' ' || q.severity AS severity_label,
                   {CHECK_CATEGORY_SQL} AS check_category,
                   q.metric, TRY_CAST(q.value AS DOUBLE) AS value, TRY_CAST(q.denominator AS DOUBLE) AS denominator,
                   CAST(q.threshold AS VARCHAR) AS threshold, q.details, q.recommended_action, q.mode,
                   (q.mode LIKE 'sample%') AS is_sampled, q.result IN ('FAIL', 'ERROR') AS is_failure,
                   q.result = 'WARN' AS is_warning,
                   rm.definition_status, rm.remediation_status, rm.root_cause, rm.next_diagnostic,
                   {f"'{mtime}'" if mtime else 'NULL'}::VARCHAR AS results_saved_at
            FROM r_quality q {rem}
        """
        self.mart(self.spec("quality_check", "one row per saved data-quality check", ("check_seq",), (),
                            "Latest saved Phase 01 quality results (+ Phase 02 remediation if available). "
                            "Sampled checks find issues but cannot prove absence.",
                            {"value": "measured metric (affected rows/keys; sampled if is_sampled)",
                             "results_saved_at": "file modification time of the saved results (UTC)"},
                            optional=("r_quality", "r_remediation")), sql, force)

    def build_quality_run_history(self, force: bool) -> None:
        sql = """
            SELECT run_id, step, started_at, status, seconds, rss_peak_mb, n_warnings, critical_failure, error
            FROM (SELECT run_id, step,
                         TRY_CAST(replace(replace(CAST(started_at AS VARCHAR), 'T', ' '), 'Z', '') AS TIMESTAMP) AS started_at,
                         status, seconds, rss_peak_mb,
                         CASE WHEN warnings IS NULL OR warnings = '' THEN 0
                              ELSE len(string_split(warnings, '; ')) END AS n_warnings,
                         coalesce(error, '') LIKE '%critical%' AS critical_failure, error
                  FROM r_runlog
                  WHERE step LIKE 'validate%' OR step LIKE 'tracking_qc%' OR step LIKE 'lab_%' OR step LIKE 'quality%')
            WHERE run_id IS NOT NULL AND started_at IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY run_id, step, started_at ORDER BY seconds) = 1
            ORDER BY started_at
        """
        self.mart(self.spec("quality_run_history", "one row per logged quality/validation step execution",
                            ("run_id", "step", "started_at"), (),
                            "Real execution history from reports/tables/pipeline_run_log (no synthetic trend).",
                            {"n_warnings": "warnings recorded by the step"}, optional=("r_runlog",)), sql, force)

    def build_tracking_qc(self, force: bool) -> None:
        self.mart(self.spec("tracking_interval", "one row per domain × capped sampling interval",
                            ("domain", "dt_ms_capped"), (), "Sampling-interval distribution (saved, sampled).",
                            {"n": "frame pairs"}, optional=("r_intervals",)),
                  "SELECT domain, dt_ms_capped, sum(n) AS n, any_value(sample_pct) AS sample_pct FROM r_intervals "
                  "GROUP BY ALL ORDER BY domain, dt_ms_capped", force)
        self.mart(self.spec("tracking_consistency", "one row per domain", ("domain",), (),
                            "Vendor speed vs position-derived speed consistency (saved, sampled).",
                            {"corr_s": "corr(vendor s, derived s)"}, optional=("r_consistency",)),
                  "SELECT * FROM r_consistency QUALIFY row_number() OVER (PARTITION BY domain ORDER BY n_frames DESC) = 1",
                  force)

    def build_quality_affected(self, force: bool) -> None:
        parts = []
        if self.flags_available("f_nfl"):
            parts.append("""
                SELECT 'nfl' AS domain, count(*) AS n_units, count_if(exclude_kinematics) AS n_excluded_kinematics,
                       count_if(exclude_context) AS n_excluded_context,
                       count(DISTINCT nfl_id) FILTER (WHERE exclude_kinematics OR exclude_context) AS n_athletes_affected,
                       count(DISTINCT (game_id, play_id)) FILTER (WHERE exclude_kinematics OR exclude_context) AS n_plays_affected,
                       coalesce(sum(n_frames) FILTER (WHERE exclude_kinematics OR exclude_context), 0) AS n_frames_affected
                FROM f_nfl""")
        if self.flags_available("f_combine"):
            parts.append("""
                SELECT 'combine', count(*), count_if(exclude_kinematics), NULL::BIGINT,
                       count(DISTINCT nfl_id) FILTER (WHERE exclude_kinematics), NULL::BIGINT,
                       coalesce(sum(n_frames) FILTER (WHERE exclude_kinematics), 0)
                FROM f_combine""")
        sql = " UNION ALL ".join(parts) if parts else (
            "SELECT NULL::VARCHAR AS domain, 0::BIGINT AS n_units, 0::BIGINT AS n_excluded_kinematics, "
            "0::BIGINT AS n_excluded_context, 0::BIGINT AS n_athletes_affected, 0::BIGINT AS n_plays_affected, "
            "0::BIGINT AS n_frames_affected WHERE FALSE")
        self.mart(self.spec("quality_affected", "one row per tracking domain (empty until Phase 02 flags exist)",
                            ("domain",), (), "Affected athletes / plays / frames from Phase 02 exclusion flags.",
                            {"n_units": "player-plays (nfl) or attempts (combine)"},
                            optional=("f_nfl", "f_combine")), sql, force)

    # ================================================================== research
    def build_research(self, force: bool) -> None:
        sql = """
            SELECT hypothesis, title, positions, drill_regex, matching_drills, traditional_baseline, nfl_outcome,
                   n_population, n_traditional, n_combine_tracking_drill, n_outcome, n_matched, n_matched_with_baseline,
                   largest_position, largest_position_n_matched, feasibility,
                   1 - n_matched / nullif(n_population, 0)::DOUBLE AS missing_share,
                   n_matched_with_baseline / nullif(n_population, 0)::DOUBLE AS eligible_share,
                   CASE WHEN drill_regex = '.*' THEN 'all drills'
                        ELSE 'regex — needs explicit drill list' END AS drill_matching,
                   CASE WHEN coalesce(n_matched_with_baseline, 0) < 30 THEN 'insufficient sample'
                        WHEN drill_regex <> '.*' THEN 'needs drill-definition review'
                        WHEN feasibility = 'Low' THEN 'exploratory only'
                        ELSE 'data available' END AS readiness_status,
                   CASE WHEN coalesce(n_matched_with_baseline, 0) < 30 THEN 'fewer than 30 matched athletes with baseline'
                        WHEN drill_regex <> '.*' THEN 'drill regex may match unintended drills (see SESSION_STATE)'
                        WHEN feasibility = 'Low' THEN 'Phase 01 feasibility rated Low'
                        ELSE 'counts meet the data threshold; no statistical test performed' END AS readiness_reason,
                   'not assessed' AS statistical_support,
                   confounders, selection_bias, validation, football_interpretation, limitations
            FROM r_feas ORDER BY hypothesis
        """
        self.mart(self.spec("research_hypothesis", "one row per hypothesis", ("hypothesis",), (),
                            "Data readiness per hypothesis (rule-based; NOT statistical support).",
                            {"readiness_status": "rule-based data readiness; see readiness_reason",
                             "statistical_support": "always 'not assessed' in this lab"},
                            optional=("r_feas",)), sql, force)
        self.mart(self.spec("research_by_position", "one row per hypothesis × position group",
                            ("hypothesis", "position_group"), (), "Hypothesis coverage per position.",
                            {"match_share": "n_matched / n_population"}, optional=("r_feas_pos",)),
                  """SELECT hypothesis, position_group, n_population, n_traditional, n_combine_tracking_drill, n_outcome,
                            n_matched, n_matched_with_baseline, n_matched / nullif(n_population, 0)::DOUBLE AS match_share,
                            1 - n_matched / nullif(n_population, 0)::DOUBLE AS missing_share
                     FROM r_feas_pos ORDER BY hypothesis, position_group""", force)
        self.mart(self.spec("research_drill", "one row per hypothesis × matched drill", ("hypothesis", "drill_name"),
                            ("gold.dim_combine_drill",), "Drills matched by each hypothesis (Phase 01 regex).",
                            {"n_athletes": "athletes with ≥1 tracked attempt in the drill"}, optional=("r_feas",)),
                  """WITH u AS (SELECT hypothesis, unnest(string_split(matching_drills, ',')) AS drill
                                FROM r_feas WHERE matching_drills IS NOT NULL),
                          d AS (SELECT DISTINCT hypothesis, trim(drill) AS drill_name FROM u)
                     SELECT d.hypothesis, d.drill_name, sum(c.n_athletes) AS n_athletes, sum(c.n_attempts) AS n_attempts
                     FROM d LEFT JOIN g_dim_combine_drill c ON c.drill_name = d.drill_name
                     WHERE d.drill_name <> '' GROUP BY ALL ORDER BY d.hypothesis, d.drill_name""", force)

    # ================================================================== snapshot tables (always rebuilt)
    def _mtime_iso(self, view: str) -> str | None:
        refs = self.optional_sources.get(view) or []
        if not refs:
            return None
        p = self.store.source_files(refs[0])[0]
        return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def build_kpi_snapshot(self) -> None:
        import pyarrow.parquet as pq

        def builder() -> pl.DataFrame:
            r = query_row(self.con, """
                SELECT (SELECT count(*) FROM lab_athlete) AS n_athletes,
                       (SELECT count_if(has_combine_tracking) FROM lab_athlete) AS n_athletes_combine_tracking,
                       (SELECT count_if(has_nfl_tracking) FROM lab_athlete) AS n_athletes_nfl_tracking,
                       (SELECT count_if(is_matched) FROM lab_athlete) AS n_athletes_matched,
                       (SELECT count(*) FROM g_dim_game) AS n_games,
                       (SELECT count(DISTINCT game_id) FROM g_fact_nfl_play_participation) AS n_games_tracked,
                       (SELECT count(*) FROM g_dim_play) AS n_plays,
                       (SELECT count(DISTINCT (game_id, play_id)) FROM g_fact_nfl_play_participation) AS n_plays_tracked,
                       (SELECT count(*) FROM g_fact_nfl_play_participation) AS n_player_plays,
                       (SELECT sum(n_frames) FROM g_fact_nfl_play_participation) AS n_nfl_player_frames,
                       (SELECT count(*) FROM g_fact_combine_attempt) AS n_combine_attempts,
                       (SELECT count(DISTINCT drill_name) FROM g_fact_combine_attempt) AS n_combine_drills
            """)
            seasons = [str(s) for s in query_column(self.con, "SELECT DISTINCT season FROM g_dim_game ORDER BY 1")]
            gm = self.cfg.paths.catalog / "gold_manifest.parquet"
            gold_built = pl.read_parquet(gm)["built_at"].max() if gm.is_file() else None
            nrows = lambda t: sum(pq.ParquetFile(f).metadata.num_rows for f in self.refs.files("silver", t))  # noqa: E731
            age_h = None
            if gold_built:
                age_h = round((datetime.now(timezone.utc) - datetime.strptime(gold_built, "%Y-%m-%dT%H:%M:%SZ")
                               .replace(tzinfo=timezone.utc)).total_seconds() / 3600, 2)
            return pl.DataFrame([{
                "snapshot": "current", **r, "n_seasons": len(seasons), "seasons_label": ", ".join(seasons),
                "n_silver_nfl_tracking_rows": nrows("game_tracking"),
                "n_silver_combine_tracking_rows": nrows("combine_tracking"),
                "gold_built_at": gold_built, "gold_age_hours_at_lab_build": age_h,
                "quality_results_saved_at": self._mtime_iso("r_quality"), "lab_built_at": utc_now(),
            }])

        self.mart(self.spec("kpi_snapshot", "single row (snapshot = 'current')", ("snapshot",),
                            ("gold.dim_athlete", "gold.dim_game", "gold.fact_nfl_play_participation"),
                            "Unfiltered warehouse KPIs computed at lab build time.",
                            {"n_silver_nfl_tracking_rows": "silver NFL frame rows incl. football (Parquet footers)",
                             "gold_age_hours_at_lab_build": "hours between gold build and this lab build"}),
                  builder, force=True)

    def build_availability(self) -> None:
        def builder() -> pl.DataFrame:
            rows = [{"dataset": s.name, "available": s.available, "path": s.path, "note": s.note, "used_by": s.used_by}
                    for s in self.sources]
            return pl.DataFrame(rows, schema={"dataset": pl.String, "available": pl.Boolean, "path": pl.String,
                                              "note": pl.String, "used_by": pl.String})
        self.mart(self.spec("availability", "one row per optional input", ("dataset",), ("gold.dim_athlete",),
                            "Which optional inputs were present at build time (empty charts ⇒ check here).",
                            {"available": "input found at build time"}), builder, force=True)

    # ================================================================== orchestration
    def build_all(self, force: bool = False) -> pl.DataFrame:
        self.register_inputs()
        for step in (self.build_athlete, self.build_combine_attempt, self.build_combine_athlete_drill,
                     self.build_drill_participation, self.build_test_availability, self.build_coverage_matrix,
                     self.build_linkage_funnel, self.build_season_volume, self.build_tracking_partition,
                     self.build_nfl_game, self.build_nfl_play, self.build_nfl_player_play,
                     self.build_nfl_athlete_season, self.build_nfl_speed_distribution, self.build_quality_check,
                     self.build_quality_run_history, self.build_tracking_qc, self.build_quality_affected,
                     self.build_research):
            step(force)
        self.build_kpi_snapshot()
        self.build_availability()
        df = pl.DataFrame(self.results)
        self.store.export_csv(df, "lab_mart_build_log", "Build/reuse decision per mart for the latest lab build")
        self.store.export_csv(self.store.manifest_frame().filter(pl.col("kind") == "datasets"),
                              "lab_mart_catalog", "Lab marts: grain, key, rows, path, build time")
        return df
