"""Phase 02 · Task 1 — data-quality remediation.

Pipeline (all steps run only when the USER executes this module):

1. **Load** saved Phase 01 results (``reports/tables/data_quality_results.parquet``).
2. **Definition audit** — compare saved key-uniqueness checks with the *current* registry
   (``sources.SOURCES`` grain candidates). A saved failure produced by an outdated definition
   (e.g. a notebook kernel holding a stale module) is marked ``superseded``, not hidden.
3. **Triage** FAIL / WARN / ERROR rows: category, severity, sampled vs exact, affected rows
   (sampled counts are extrapolated and labelled as estimates).
4. **Exact diagnostics** — one targeted DuckDB query per failure type (D1–D4). The same SQL is
   written to ``reports/sql/phase_02/quality/`` with project-relative paths for manual DuckDB CLI use.
5. **Exclusion flags** — reproducible, keyed flag datasets (original data untouched):
   ``tracking_quality_flags_nfl`` (player × play) and ``tracking_quality_flags_combine`` (attempt).
6. **Remediation summary** — status per failure decided by explicit rules on measured numbers;
   anything the rules cannot decide is ``unresolved`` with the diagnostic to run.
7. **Outputs** — CSVs, figure, ``reports/09_quality_remediation.md``.

Statuses
    resolved_superseded_definition  saved failure came from an outdated check definition and the
                                    exact diagnostic shows the current definition holds
    tracking_artifact               values violate physical plausibility rules (speed, accel,
                                    per-frame displacement) for a *human* tracked entity
    expected_behaviour              rule violation belongs to an entity the rule does not apply to
                                    (e.g. human speed limits applied to BALL rows)
    coverage_gap                    rows exist in one source but not the other; cause not
                                    determinable from the data → flagged, cause unresolved
    unresolved                      not decidable from evidence; see ``next_diagnostic``

Command for the user::

    set PYTHONPATH=src & set NFL_BDB_DUCKDB_MEMORY=3GB
    python -m nflbdb.research.quality            # add --force to rebuild flag datasets
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..data.config import Phase01Config, duckdb_connect
from ..data.frames import query_df, query_row
from ..data.reporting import INK, STATUS, MarkdownReport, add_caption, new_figure
from ..data.sources import SOURCES
from ..data.tables import LayerRefs
from ..data.utils import atomic_write_text, get_logger
from .artifacts import ArtifactStore, DatasetSpec

log = get_logger(__name__)

QUALITY_VERSION = "1"
SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
STATUS_COLORS = {
    "resolved_superseded_definition": STATUS["good"],
    "expected_behaviour": "#86b6ef",
    "tracking_artifact": STATUS["warning"],
    "coverage_gap": STATUS["serious"],
    "unresolved": STATUS["critical"],
}

# ======================================================================================
# Dataset contracts
# ======================================================================================
NFL_FLAGS = DatasetSpec(
    name="tracking_quality_flags_nfl",
    grain="one row per tracked player × play (football rows excluded)",
    key=("game_id", "play_id", "nfl_id"),
    sources=("silver.game_tracking", "silver.player_play"),
    description="Frame-integrity and linkage flags per NFL player-play. Join on the key to exclude "
                "or down-weight trajectories; silver data is never modified.",
    version=QUALITY_VERSION,
    columns={
        "season": "NFL season (from partition)", "game_id": "game identifier", "play_id": "play identifier",
        "nfl_id": "player identifier", "n_frames": "tracked frames (count)",
        "n_dup_ts": "frames with Δt = 0 ms vs previous frame", "n_gaps": "frame pairs with Δt > gap_factor × nominal",
        "n_jumps": "frames whose displacement exceeds max_step_yds × max(Δt/nominal, 1)",
        "n_speed_gt_max": "frames with s > max_speed_yds (yd/s)", "n_accel_gt_max": "frames with a > max_accel_yds2 (yd/s²)",
        "n_out_of_bounds": "frames outside field ± coord_tolerance_yds",
        "n_missing_direction": "frames without play_direction (x_std/y_std not standardised)",
        "max_s": "maximum reported speed (yd/s)", "max_step_yds": "largest per-frame displacement (yd)",
        "in_player_play": "player-play key exists in silver.player_play",
        "play_in_player_play": "play key exists in silver.player_play",
        "exclude_kinematics": "TRUE if any frame-integrity flag → exclude from kinematic features",
        "exclude_context": "TRUE if no player_play row → no role/context available",
        "exclusion_reasons": "semicolon-separated reasons",
    },
)
COMBINE_FLAGS = DatasetSpec(
    name="tracking_quality_flags_combine",
    grain="one row per Combine athlete × drill attempt",
    key=("event_id", "nfl_id", "attempt"),
    sources=("silver.combine_tracking",),
    description="Frame-integrity flags per Combine attempt (PLAYER frames; BALL frames counted separately).",
    version=QUALITY_VERSION,
    columns={
        "draft_year": "Combine / draft cohort", "event_id": "Combine event id", "nfl_id": "athlete id",
        "attempt": "attempt number", "drill_type": "drill family", "drill_name": "drill",
        "n_drill_names": "distinct drill names inside the attempt (should be 1)",
        "n_frames": "PLAYER frames", "n_ball_frames": "BALL frames in the same attempt",
        "n_dup_ts": "PLAYER frames with Δt = 0", "n_gaps": "PLAYER frame pairs with Δt > gap_factor × nominal",
        "n_jumps": "PLAYER frames with implausible displacement", "n_speed_gt_max": "PLAYER frames with s > max_speed_yds",
        "n_accel_gt_max": "PLAYER frames with a > max_accel_yds2", "max_s": "max PLAYER speed (yd/s)",
        "max_a": "max PLAYER acceleration (yd/s²)", "ball_max_s": "max BALL speed (yd/s)",
        "exclude_kinematics": "TRUE if any PLAYER integrity flag, < min frames, or mixed drill names",
        "exclusion_reasons": "semicolon-separated reasons",
    },
)
MIN_ATTEMPT_FRAMES = 5
ROW_METRICS = ["duplicate_key_rows", "identical_rows", "rows_with_null_id", "out_of_range_values",
               "unexpected_values", "unparseable", "null_rows"]
SUMMARY_SCHEMA = {
    "check_id": pl.String, "check": pl.String, "table": pl.String, "columns": pl.String, "severity": pl.String,
    "result": pl.String, "definition_status": pl.String, "affected_rows_estimate": pl.Float64,
    "affected_rows_basis": pl.String, "status": pl.String, "root_cause": pl.String, "evidence": pl.String,
    "affected_athletes": pl.Int64, "remediation": pl.String, "exclusion_flag": pl.String, "next_diagnostic": pl.String,
}

# ======================================================================================
# Diagnostic SQL (templates; {placeholders} resolved to read_parquet expressions)
# ======================================================================================
DIAGNOSTICS: dict[str, tuple[str, str]] = {
    "D1_combine_grain": (
        "Do Combine duplicate keys under (event_id, nfl_id, attempt, time) differ only by entity_type?",
        """
WITH k AS (
    SELECT event_id, nfl_id, attempt, time, count(*) AS n, count(DISTINCT entity_type) AS n_entity
    FROM {bronze_combine_tracking} GROUP BY ALL HAVING count(*) > 1)
SELECT (SELECT count(*) FROM {bronze_combine_tracking}) AS total_rows,
       (SELECT count(*) FROM {bronze_combine_tracking} WHERE entity_type = 'BALL') AS ball_rows,
       (SELECT count(*) FROM k) AS dup_keys_old_grain,
       (SELECT coalesce(sum(n - 1), 0) FROM k) AS dup_rows_old_grain,
       (SELECT count_if(n_entity = n) FROM k) AS dup_keys_explained_by_entity_type,
       (SELECT count(*) FROM (SELECT event_id, nfl_id, attempt, entity_type, time FROM {bronze_combine_tracking}
                              GROUP BY ALL HAVING count(*) > 1)) AS dup_keys_new_grain
""",
    ),
    "D2_orphan_plays": (
        "Which tracked plays have no player_play rows at all?",
        """
WITH tp AS (
    SELECT season, game_id, play_id, count(*) AS n_frames, count(DISTINCT nfl_id) AS n_players,
           min(ts) AS t0, max(ts) AS t1
    FROM {silver_game_tracking} WHERE NOT is_football GROUP BY ALL),
     ppp AS (SELECT DISTINCT game_id, play_id FROM {silver_player_play}),
     o AS (SELECT tp.* FROM tp ANTI JOIN ppp ON tp.game_id = ppp.game_id AND tp.play_id = ppp.play_id)
SELECT o.season, o.game_id, o.play_id, g.week, g.season_type, o.n_frames, o.n_players,
       (epoch_ms(o.t1) - epoch_ms(o.t0)) / 1000.0 AS tracked_duration_s
FROM o LEFT JOIN {silver_games} g ON g.game_id = o.game_id
ORDER BY o.season, o.game_id, o.play_id
""",
    ),
    "D3_orphan_player_plays": (
        "Which tracked player-plays have no player_play row, and is the whole play missing or only the player?",
        """
WITH tk AS (SELECT DISTINCT season, game_id, play_id, nfl_id FROM {silver_game_tracking} WHERE NOT is_football),
     ppk AS (SELECT DISTINCT game_id, play_id, nfl_id FROM {silver_player_play}),
     ppp AS (SELECT DISTINCT game_id, play_id FROM {silver_player_play}),
     o AS (SELECT tk.* FROM tk ANTI JOIN ppk
           ON tk.game_id = ppk.game_id AND tk.play_id = ppk.play_id AND tk.nfl_id = ppk.nfl_id)
SELECT o.season, o.game_id, o.play_id, o.nfl_id,
       (p.game_id IS NOT NULL) AS play_in_player_play, a.position_group, a.cohort
FROM o
LEFT JOIN ppp p ON p.game_id = o.game_id AND p.play_id = o.play_id
LEFT JOIN {gold_dim_athlete} a ON a.nfl_id = o.nfl_id
ORDER BY o.season, o.game_id, o.play_id, o.nfl_id
""",
    ),
    "D4_combine_kinematic_outliers": (
        "Which entity types and drills produce Combine speed/acceleration above human plausibility thresholds?",
        """
SELECT entity_type, drill_name, count(*) AS n_rows,
       count_if(s > {max_speed}) AS n_speed_gt_max, count_if(a > {max_accel}) AS n_accel_gt_max,
       count(DISTINCT CASE WHEN s > {max_speed} OR a > {max_accel}
             THEN event_id || '|' || CAST(nfl_id AS VARCHAR) || '|' || CAST(attempt AS VARCHAR) END) AS n_attempts_affected,
       count(DISTINCT CASE WHEN s > {max_speed} OR a > {max_accel} THEN nfl_id END) AS n_athletes_affected,
       max(s) AS max_s, max(a) AS max_a
FROM {silver_combine_tracking}
GROUP BY ALL HAVING count_if(s > {max_speed} OR a > {max_accel}) > 0
ORDER BY entity_type, n_speed_gt_max DESC
""",
    ),
}


@dataclass
class DiagnosticResult:
    name: str
    question: str
    frame: pl.DataFrame
    seconds: float
    error: str = ""


# ======================================================================================
class QualityRemediation:
    def __init__(self, cfg: Phase01Config, store: ArtifactStore | None = None, con=None):
        self.cfg = cfg
        self.store = store or ArtifactStore(cfg)
        self.con = con or duckdb_connect(cfg)
        self.refs = LayerRefs(cfg)
        self.sql_dir = cfg.paths.reports_root / "sql" / "phase_02" / "quality"

    # ------------------------------------------------------------------ 1. load
    def load_results(self) -> pl.DataFrame:
        path = self.cfg.paths.tables / "data_quality_results.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{self.cfg.paths.rel(path)} missing — run the Phase 01 'validate' step first")
        return pl.read_parquet(path)

    # ------------------------------------------------------------------ 2. definition audit
    @staticmethod
    def expected_key_checks() -> dict[str, str]:
        """check_id → severity for key-uniqueness checks under the CURRENT registry
        (mirrors DataQualityValidator.default_suite: last grain candidate = CRITICAL)."""
        out = {}
        for spec in SOURCES.values():
            for key in spec.grain_candidates:
                sev = "CRITICAL" if key == spec.grain_candidates[-1] else "MEDIUM"
                out[f"uniq:{spec.name}:{', '.join(key)}"] = sev
        return out

    def audit_definitions(self, results: pl.DataFrame) -> pl.DataFrame:
        expected = self.expected_key_checks()

        def status(check_id: str, severity: str) -> str:
            if not check_id.startswith("uniq:"):
                return "current"
            if check_id not in expected:
                return "superseded: key no longer a grain candidate"
            if expected[check_id] != severity:
                return f"superseded: severity now {expected[check_id]} (was {severity})"
            return "current"

        return results.with_columns(
            pl.struct("check_id", "severity").map_elements(lambda r: status(r["check_id"], r["severity"]),
                                                           return_dtype=pl.String).alias("definition_status"))

    # ------------------------------------------------------------------ 3. triage
    def triage(self, results: pl.DataFrame) -> pl.DataFrame:
        full_rows = {t: (self.refs.num_rows("bronze", t) if self.refs.exists("bronze", t) else None) for t in SOURCES}
        audited = self.audit_definitions(results)
        fails = audited.filter(pl.col("result").is_in(["FAIL", "WARN", "ERROR"]))
        return (
            fails.with_columns(
                pl.col("check_id").str.split(":").list.first().alias("category"),
                pl.col("mode").str.starts_with("sample").alias("is_sampled"),
                (pl.col("value") / pl.col("denominator")).alias("affected_rate"),
                pl.col("table").replace_strict(full_rows, default=None, return_dtype=pl.Int64).alias("table_rows_full"),
                pl.col("severity").cast(pl.Enum(SEVERITY_ORDER)).alias("severity"),
            )
            .with_columns(
                # Only row-count metrics are extrapolated; distinct-key metrics (orphan keys) are not rows.
                (pl.col("metric").is_in(ROW_METRICS) | pl.col("metric").str.starts_with("n_")).alias("_row_metric"),
            )
            .with_columns(
                pl.when(pl.col("is_sampled") & pl.col("_row_metric"))
                .then((pl.col("affected_rate") * pl.col("table_rows_full")).round(0))
                .otherwise(pl.col("value")).alias("affected_rows_estimate"),
                pl.when(pl.col("is_sampled") & pl.col("_row_metric")).then(pl.lit("extrapolated from sample (rows)"))
                .when(pl.col("is_sampled")).then(pl.lit("observed in sample (distinct keys; not extrapolated)"))
                .otherwise(pl.lit("exact")).alias("affected_rows_basis"),
            )
            .select("check_id", "check", "table", "columns", "category", "severity", "result", "definition_status",
                    "mode", "is_sampled", "metric", "value", "denominator", "affected_rate", "affected_rows_estimate",
                    "affected_rows_basis", "details", "recommended_action")
            .sort("severity", "check_id")
        )

    # ------------------------------------------------------------------ 4. diagnostics
    def _placeholders(self, relative: bool) -> dict[str, str]:
        tables = {
            "bronze_combine_tracking": ("bronze", "combine_tracking"),
            "silver_combine_tracking": ("silver", "combine_tracking"),
            "silver_game_tracking": ("silver", "game_tracking"),
            "silver_player_play": ("silver", "player_play"),
            "silver_games": ("silver", "games"),
            "gold_dim_athlete": ("gold", "dim_athlete"),
        }
        out = {"max_speed": str(self.cfg.tracking.max_speed_yds), "max_accel": str(self.cfg.tracking.max_accel_yds2)}
        for ph, (layer, name) in tables.items():
            if relative:
                base = self.refs.path(layer, name)
                single = base.with_suffix(".parquet")
                glob = self.cfg.paths.rel(single) if single.exists() else self.cfg.paths.rel(base) + "/**/*.parquet"
                out[ph] = f"read_parquet('{glob}', hive_partitioning = true)"
            else:
                out[ph] = self.refs.sql(layer, name) if self.refs.exists(layer, name) else f"<missing {layer}.{name}>"
        return out

    def write_sql_files(self) -> list[Path]:
        """Diagnostic SQL with project-relative paths. Run from the project root, e.g.
        ``duckdb -c ".read reports/sql/phase_02/quality/D2_orphan_plays.sql"``."""
        ph = self._placeholders(relative=True)
        paths = []
        for name, (question, sql) in DIAGNOSTICS.items():
            text = f"-- {name}: {question}\n-- Run from the project root (paths are relative).\n{sql.format(**ph).strip()};\n"
            paths.append(atomic_write_text(text, self.sql_dir / f"{name}.sql"))
        return paths

    def run_diagnostics(self) -> dict[str, DiagnosticResult]:
        ph = self._placeholders(relative=False)
        out = {}
        for name, (question, sql) in DIAGNOSTICS.items():
            t0 = time.perf_counter()
            try:
                df = query_df(self.con, sql.format(**ph))
                err = ""
            except Exception as exc:  # noqa: BLE001 - a failed diagnostic must not hide the others
                df, err = pl.DataFrame(), f"{type(exc).__name__}: {exc}"
                log.error("diagnostic %s failed: %s", name, err)
            out[name] = DiagnosticResult(name, question, df, round(time.perf_counter() - t0, 2), err)
            log.info("diagnostic %s: %s rows in %.1fs", name, df.height, out[name].seconds)
        return out

    # ------------------------------------------------------------------ 5. exclusion flags
    def _nfl_flags_frame(self) -> pl.DataFrame:
        tr = self.cfg.tracking
        nom, tol = tr.nominal_interval_ms, tr.coord_tolerance_yds
        pp = self.refs.sql("silver", "player_play")
        parts = []
        for f in self.refs.files("silver", "game_tracking"):  # one season per pass → bounded memory
            season = int(f.parent.name.split("=")[-1])
            src = f"read_parquet('{f.as_posix()}')"
            parts.append(query_df(self.con, f"""
                WITH t AS (
                    SELECT game_id, play_id, nfl_id, ts, x, y, s, a, play_direction,
                           epoch_ms(ts) - epoch_ms(lag(ts) OVER w) AS dt_ms,
                           sqrt(power(x - lag(x) OVER w, 2) + power(y - lag(y) OVER w, 2)) AS step
                    FROM {src} WHERE NOT is_football
                    WINDOW w AS (PARTITION BY game_id, play_id, nfl_id ORDER BY ts)),
                f AS (
                    SELECT game_id, play_id, nfl_id, count(*) AS n_frames,
                           count_if(dt_ms = 0) AS n_dup_ts,
                           count_if(dt_ms > {nom * tr.gap_factor}) AS n_gaps,
                           count_if(step > {tr.max_step_yds} * greatest(dt_ms / {nom}, 1)) AS n_jumps,
                           count_if(s > {tr.max_speed_yds}) AS n_speed_gt_max,
                           count_if(a > {tr.max_accel_yds2}) AS n_accel_gt_max,
                           count_if(x < {-tol} OR x > {tr.field_length_yds + tol}
                                    OR y < {-tol} OR y > {tr.field_width_yds + tol}) AS n_out_of_bounds,
                           count_if(play_direction IS NULL) AS n_missing_direction,
                           max(s) AS max_s, max(step) AS max_step_yds
                    FROM t GROUP BY ALL),
                ppk AS (SELECT DISTINCT game_id, play_id, nfl_id FROM {pp}),
                ppp AS (SELECT DISTINCT game_id, play_id FROM {pp})
                SELECT {season} AS season, f.*,
                       (k.nfl_id IS NOT NULL) AS in_player_play, (p.play_id IS NOT NULL) AS play_in_player_play
                FROM f
                LEFT JOIN ppk k ON k.game_id = f.game_id AND k.play_id = f.play_id AND k.nfl_id = f.nfl_id
                LEFT JOIN ppp p ON p.game_id = f.game_id AND p.play_id = f.play_id
            """))
            log.info("NFL flags season %s: %s player-plays", season, f"{parts[-1].height:,}")
        df = pl.concat(parts, how="vertical_relaxed")
        reasons = [
            (pl.col("n_jumps") > 0, "jump"), (pl.col("n_speed_gt_max") > 0, "speed_gt_max"),
            (pl.col("n_dup_ts") > 0, "duplicate_timestamp"), (~pl.col("in_player_play"), "no_player_play_row"),
            (pl.col("n_missing_direction") > 0, "no_play_direction"),
        ]
        return df.with_columns(
            ((pl.col("n_jumps") > 0) | (pl.col("n_speed_gt_max") > 0) | (pl.col("n_dup_ts") > 0)).alias("exclude_kinematics"),
            (~pl.col("in_player_play")).alias("exclude_context"),
            pl.concat_str([pl.when(c).then(pl.lit(r)) for c, r in reasons], separator=";", ignore_nulls=True)
            .alias("exclusion_reasons"),
        ).sort("season", "game_id", "play_id", "nfl_id")

    def _combine_flags_frame(self) -> pl.DataFrame:
        tr = self.cfg.tracking
        nom = tr.nominal_interval_ms
        src = self.refs.sql("silver", "combine_tracking")
        player = "entity_type = 'PLAYER'"
        df = query_df(self.con, f"""
            WITH t AS (
                SELECT draft_year, event_id, nfl_id, attempt, entity_type, drill_type, drill_name, ts, x, y, s, a,
                       epoch_ms(ts) - epoch_ms(lag(ts) OVER w) AS dt_ms,
                       sqrt(power(x - lag(x) OVER w, 2) + power(y - lag(y) OVER w, 2)) AS step
                FROM {src} WHERE nfl_id IS NOT NULL
                WINDOW w AS (PARTITION BY event_id, nfl_id, attempt, entity_type ORDER BY ts))
            SELECT draft_year, event_id, nfl_id, attempt,
                   any_value(drill_type) AS drill_type, any_value(drill_name) AS drill_name,
                   count(DISTINCT drill_name) AS n_drill_names,
                   count_if({player}) AS n_frames, count_if(entity_type = 'BALL') AS n_ball_frames,
                   count_if({player} AND dt_ms = 0) AS n_dup_ts,
                   count_if({player} AND dt_ms > {nom * tr.gap_factor}) AS n_gaps,
                   count_if({player} AND step > {tr.max_step_yds} * greatest(dt_ms / {nom}, 1)) AS n_jumps,
                   count_if({player} AND s > {tr.max_speed_yds}) AS n_speed_gt_max,
                   count_if({player} AND a > {tr.max_accel_yds2}) AS n_accel_gt_max,
                   max(s) FILTER (WHERE {player}) AS max_s, max(a) FILTER (WHERE {player}) AS max_a,
                   max(s) FILTER (WHERE entity_type = 'BALL') AS ball_max_s
            FROM t GROUP BY draft_year, event_id, nfl_id, attempt
        """)
        reasons = [
            (pl.col("n_jumps") > 0, "jump"), (pl.col("n_speed_gt_max") > 0, "speed_gt_max"),
            (pl.col("n_accel_gt_max") > 0, "accel_gt_max"), (pl.col("n_dup_ts") > 0, "duplicate_timestamp"),
            (pl.col("n_frames") < MIN_ATTEMPT_FRAMES, f"lt_{MIN_ATTEMPT_FRAMES}_player_frames"),
            (pl.col("n_drill_names") > 1, "mixed_drill_names"),
        ]
        excl = pl.any_horizontal([c for c, _ in reasons])
        return df.with_columns(
            excl.alias("exclude_kinematics"),
            pl.concat_str([pl.when(c).then(pl.lit(r)) for c, r in reasons], separator=";", ignore_nulls=True)
            .alias("exclusion_reasons"),
        ).sort("nfl_id", "drill_name", "event_id", "attempt")

    def build_flags(self, force: bool = False) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        params = {k: getattr(self.cfg.tracking, k) for k in
                  ("nominal_interval_ms", "gap_factor", "max_step_yds", "max_speed_yds", "max_accel_yds2",
                   "coord_tolerance_yds", "field_length_yds", "field_width_yds")} | {"min_frames": MIN_ATTEMPT_FRAMES}
        nfl = self.store.build(NFL_FLAGS, self._nfl_flags_frame, params=params, force=force)
        cmb = self.store.build(COMBINE_FLAGS, self._combine_flags_frame, params=params, force=force)
        return nfl, cmb

    @staticmethod
    def flags_summary(nfl: pl.LazyFrame, cmb: pl.LazyFrame) -> pl.DataFrame:
        flag_cols = ["n_dup_ts", "n_gaps", "n_jumps", "n_speed_gt_max", "n_accel_gt_max"]
        n = (nfl.group_by("season").agg(
                pl.len().alias("units"), pl.col("nfl_id").n_unique().alias("athletes"),
                *[(pl.col(c) > 0).sum().alias(f"units_with_{c[2:]}") for c in flag_cols],
                pl.col("exclude_kinematics").sum().alias("excluded_kinematics"),
                pl.col("exclude_context").sum().alias("excluded_context"),
                pl.col("nfl_id").filter(pl.col("exclude_kinematics")).n_unique().alias("athletes_with_excluded_units"))
             .with_columns(pl.lit("nfl player-play").alias("domain"), pl.col("season").cast(pl.Int64).alias("cohort"))
             .drop("season"))
        c = (cmb.group_by("draft_year").agg(
                pl.len().alias("units"), pl.col("nfl_id").n_unique().alias("athletes"),
                *[(pl.col(col) > 0).sum().alias(f"units_with_{col[2:]}") for col in flag_cols],
                pl.col("exclude_kinematics").sum().alias("excluded_kinematics"),
                pl.lit(0, dtype=pl.UInt32).alias("excluded_context"),
                pl.col("nfl_id").filter(pl.col("exclude_kinematics")).n_unique().alias("athletes_with_excluded_units"))
             .with_columns(pl.lit("combine attempt").alias("domain"), pl.col("draft_year").cast(pl.Int64).alias("cohort"))
             .drop("draft_year"))
        out = pl.concat([n.collect(), c.collect()], how="diagonal_relaxed")
        return out.with_columns((pl.col("excluded_kinematics") / pl.col("units")).round(5).alias("excluded_kinematics_rate")) \
            .select("domain", "cohort", pl.exclude("domain", "cohort")).sort("domain", "cohort")

    # ------------------------------------------------------------------ 6. remediation rules
    def remediate(self, failures: pl.DataFrame, diag: dict[str, DiagnosticResult],
                  nfl: pl.LazyFrame | None, cmb: pl.LazyFrame | None) -> pl.DataFrame:
        rows = [self._rule(f, diag, nfl, cmb) for f in failures.iter_rows(named=True)]
        return pl.DataFrame(rows, schema=SUMMARY_SCHEMA, strict=False)  # stable schema, also when empty

    def _rule(self, f: dict, diag: dict[str, DiagnosticResult], nfl, cmb) -> dict:
        base = {k: f[k] for k in ("check_id", "check", "table", "columns", "severity", "result", "definition_status",
                                  "affected_rows_estimate", "affected_rows_basis")}
        base["severity"] = str(base["severity"])
        out = {**base, "status": "unresolved", "root_cause": "not determinable from available evidence",
               "evidence": "", "affected_athletes": None, "remediation": "", "exclusion_flag": "",
               "next_diagnostic": ""}
        cid, tbl = f["check_id"], f["table"]

        def d(name):
            r = diag.get(name)
            return None if (r is None or r.error or r.frame.is_empty()) else r.frame

        if f["result"] == "ERROR":
            out.update(root_cause="check failed to execute", remediation="fix the check and re-run validate",
                       evidence=f["details"])
        elif cid.startswith("uniq:combine_tracking"):
            g = d("D1_combine_grain")
            if g is None:
                out["next_diagnostic"] = "D1_combine_grain.sql"
            else:
                r = g.row(0, named=True)
                explained = r["dup_keys_old_grain"] > 0 and r["dup_keys_explained_by_entity_type"] == r["dup_keys_old_grain"]
                out["evidence"] = (f"exact: {r['dup_rows_old_grain']:,} duplicate rows under (event_id, nfl_id, attempt, time); "
                                   f"{r['dup_keys_explained_by_entity_type']:,}/{r['dup_keys_old_grain']:,} duplicate keys differ "
                                   f"only by entity_type; BALL rows = {r['ball_rows']:,}; duplicates under grain incl. "
                                   f"entity_type = {r['dup_keys_new_grain']:,}")
                if explained and r["dup_keys_new_grain"] == 0:
                    out.update(status="resolved_superseded_definition" if f["definition_status"] != "current" else "expected_behaviour",
                               root_cause="BALL rows carry the athlete's nfl_id; entity_type is part of the frame grain",
                               remediation="grain = (event_id, nfl_id, attempt, entity_type, time); "
                                           "re-run Phase 01 validate in a fresh kernel to refresh saved results",
                               affected_athletes=0)
                else:
                    out.update(remediation="inspect duplicate_examples() for the residual duplicates",
                               next_diagnostic="D1_combine_grain.sql")
        elif cid == "fk:game_tracking:game_id,play_id->player_play":
            o = d("D2_orphan_plays")
            if o is None:
                out["next_diagnostic"] = "D2_orphan_plays.sql"
            else:
                by = o.group_by("season").len().sort("season").rows()
                out.update(status="coverage_gap",
                           evidence=f"exact: {o.height:,} tracked plays without player_play rows "
                                    f"({o['n_frames'].sum():,} frames; by season {by}; median players/play "
                                    f"{o['n_players'].median()})",
                           root_cause="unknown — tracking exists but play context is absent in player_play",
                           remediation="keep tracking; flag exclude_context; exclude from analyses needing play context",
                           exclusion_flag="tracking_quality_flags_nfl.play_in_player_play = FALSE",
                           next_diagnostic="inspect D2 output for patterns (season_type, week, play_id position in game)")
                if nfl is not None:
                    out["affected_athletes"] = nfl.filter(~pl.col("play_in_player_play")).select(
                        pl.col("nfl_id").n_unique()).collect().item()
        elif cid == "fk:game_tracking:game_id,play_id,nfl_id->player_play":
            o = d("D3_orphan_player_plays")
            if o is None:
                out["next_diagnostic"] = "D3_orphan_player_plays.sql"
            else:
                n_play_missing = int((~o["play_in_player_play"]).sum())
                out.update(status="coverage_gap",
                           evidence=f"exact: {o.height:,} tracked player-plays without player_play row; "
                                    f"{n_play_missing:,} because the whole play is missing, "
                                    f"{o.height - n_play_missing:,} where the play exists but the player row does not",
                           root_cause="unknown — partly explained by missing plays (see D2); remainder unexplained",
                           remediation="flag exclude_context; never impute roles from other players",
                           exclusion_flag="tracking_quality_flags_nfl.in_player_play = FALSE",
                           affected_athletes=o["nfl_id"].n_unique(),
                           next_diagnostic="D3 output grouped by position_group / season")
        elif cid in ("range:combine_tracking:s", "range:combine_tracking:a"):
            o = d("D4_combine_kinematic_outliers")
            if o is None:
                out["next_diagnostic"] = "D4_combine_kinematic_outliers.sql"
            else:
                col = "n_speed_gt_max" if cid.endswith(":s") else "n_accel_gt_max"
                by_ent = o.group_by("entity_type").agg(pl.col(col).sum(), pl.col("n_athletes_affected").sum())
                ent = {r["entity_type"]: r for r in by_ent.iter_rows(named=True)}
                ball, player = (ent.get("BALL", {}).get(col, 0) or 0), (ent.get("PLAYER", {}).get(col, 0) or 0)
                out["evidence"] = f"exact: {col} — BALL rows {ball:,}, PLAYER rows {player:,} (silver, all frames)"
                if player == 0 and ball > 0:
                    out.update(status="expected_behaviour",
                               root_cause="all out-of-range values belong to BALL rows; human plausibility thresholds "
                                          "do not apply to the ball",
                               remediation="restrict human-kinematics checks and features to entity_type = 'PLAYER'",
                               affected_athletes=0)
                elif player > 0:
                    out.update(status="tracking_artifact",
                               root_cause=f"{player:,} PLAYER frames exceed the plausibility threshold "
                                          f"(rule: {self.cfg.tracking.max_speed_yds} yd/s / {self.cfg.tracking.max_accel_yds2} yd/s²)",
                               remediation="exclude flagged attempts from peak features; verify a sample visually (notebook 02 §A)",
                               exclusion_flag=f"tracking_quality_flags_combine.{col} > 0",
                               affected_athletes=int(ent.get("PLAYER", {}).get("n_athletes_affected", 0) or 0))
        elif cid.startswith("trk:game:") or cid.startswith("range:game_tracking"):
            col = {"trk:game:n_jumps": "n_jumps", "trk:game:n_speed_gt_max": "n_speed_gt_max",
                   "range:game_tracking:s": "n_speed_gt_max", "trk:game:n_dup_ts": "n_dup_ts"}.get(cid)
            if col and nfl is not None:
                r = nfl.select((pl.col(col) > 0).sum().alias("units"), pl.col(col).sum().alias("frames"),
                               pl.col("nfl_id").filter(pl.col(col) > 0).n_unique().alias("athletes")).collect().row(0, named=True)
                out.update(status="tracking_artifact" if r["frames"] else "unresolved",
                           evidence=f"exact (flags dataset): {r['frames']:,} frames in {r['units']:,} player-plays, "
                                    f"{r['athletes']:,} athletes",
                           root_cause="per-frame values violate physical plausibility rules" if r["frames"]
                           else "sampled failure not reproduced on the full table",
                           remediation="exclude flagged player-plays from kinematic features (exclude_kinematics)",
                           exclusion_flag=f"tracking_quality_flags_nfl.{col} > 0", affected_athletes=r["athletes"])
        return out

    # ------------------------------------------------------------------ 7. outputs
    def plot_breakdown(self, summary: pl.DataFrame) -> Path | None:
        if summary.is_empty():
            return None
        import numpy as np

        sev = [s for s in SEVERITY_ORDER if s in summary["severity"].to_list()]
        statuses = [s for s in STATUS_COLORS if s in summary["status"].to_list()]
        fig, (ax1, ax2) = new_figure(1, 2, gridspec_kw={"width_ratios": [2, 3]})
        left = np.zeros(len(sev))
        for st in statuses:
            vals = np.array([summary.filter((pl.col("severity") == s) & (pl.col("status") == st)).height for s in sev])
            ax1.barh(sev, vals, left=left, color=STATUS_COLORS[st], label=st.replace("_", " "), height=0.55,
                     edgecolor="white", linewidth=2)
            left += vals
        ax1.invert_yaxis()
        ax1.set_xlabel("Failed / warning checks")
        ax1.set_title("Checks by severity and remediation status")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(axis="y", visible=False)

        d = summary.with_columns(pl.col("affected_rows_estimate").fill_null(0)).sort("affected_rows_estimate")
        labels = [f"{t}: {c[:38]}" for t, c in zip(d["table"], d["columns"])]
        ax2.barh(labels, d["affected_rows_estimate"].to_list(), color=[STATUS_COLORS[s] for s in d["status"]], height=0.6)
        ax2.set_xscale("symlog")
        ax2.set_xlabel("Affected rows (exact or extrapolated; symlog)")
        ax2.set_title("Affected rows per failed check")
        ax2.grid(axis="y", visible=False)
        add_caption(fig, f"Source: data_quality_results + Phase 02 exact diagnostics · {summary.height} failed/warning checks")
        return self.store.save_fig(fig, "quality_failure_breakdown", source="data_quality_results; D1–D4",
                                   n=summary.height)

    def write_report(self, failures: pl.DataFrame, summary: pl.DataFrame, diag: dict[str, DiagnosticResult],
                     flags: pl.DataFrame | None, fig: Path | None, sql_files: list[Path]) -> Path:
        p = self.cfg.paths
        rpt = MarkdownReport("09 — Data-quality remediation", "Phase 02 · Task 1 — triage, exact diagnostics, exclusion flags")
        superseded = failures.filter(pl.col("definition_status") != "current")
        if superseded.height:
            rpt.callout("WARNING", f"{superseded.height} saved failure(s) were produced by an outdated check definition "
                        "(registry has changed since data_quality_results was written). Re-run Phase 01 `validate` in a "
                        "fresh process/kernel to refresh the saved results.")
        rpt.h2("Status overview").table(summary.group_by("severity", "status").len().sort("severity", "status"))
        if fig:
            rpt.image(fig, p.reports_root, "quality failure breakdown")
        rpt.h2("Remediation summary").table(summary.select(
            "severity", "table", "columns", "result", "definition_status", "status", "affected_rows_estimate",
            "affected_athletes", "root_cause", "remediation", "exclusion_flag"), max_rows=100)
        rpt.h2("Evidence per failure")
        rpt.bullets([f"**{r['check_id']}** — {r['evidence'] or '_no diagnostic evidence_'}"
                     + (f" · next: `{r['next_diagnostic']}`" if r["next_diagnostic"] else "")
                     for r in summary.iter_rows(named=True)])
        rpt.h2("Diagnostics")
        for r in diag.values():
            rpt.h3(f"{r.name} — {r.question}")
            rpt.p(f"Runtime {r.seconds}s · rows {r.frame.height:,}" + (f" · **ERROR** {r.error}" if r.error else ""))
            if not r.frame.is_empty():
                rpt.table(r.frame, max_rows=15)
        if flags is not None and not flags.is_empty():
            rpt.h2("Exclusion flags (persisted, keyed — original data untouched)").table(flags)
            rpt.p("Datasets: `warehouse/gold/phase_02/tracking_quality_flags_nfl.parquet` (key game_id, play_id, nfl_id) "
                  "and `tracking_quality_flags_combine.parquet` (key event_id, nfl_id, attempt). Reload with "
                  "`ArtifactStore(cfg).load('tracking_quality_flags_nfl')`.")
        rpt.h2("Unresolved items and manual diagnostics")
        unresolved = summary.filter(pl.col("status").is_in(["unresolved", "coverage_gap"]))
        rpt.bullets([f"{r['check_id']}: {r['root_cause']} → {r['next_diagnostic'] or 'review evidence'}"
                     for r in unresolved.iter_rows(named=True)] or ["_none_"])
        rpt.p("SQL for manual DuckDB CLI runs (from the project root): "
              + ", ".join(f"`{p.rel(s)}`" for s in sql_files))
        return rpt.save(p.reports_root / "09_quality_remediation.md")

    def run(self, force: bool = False) -> dict[str, pl.DataFrame]:
        results = self.load_results()
        failures = self.triage(results)
        sql_files = self.write_sql_files()
        diag = self.run_diagnostics()
        for r in diag.values():
            if not r.frame.is_empty():
                self.store.export_csv(r.frame, f"quality_diag_{r.name}", r.question)
        nfl = cmb = None
        flags = None
        try:
            nfl, cmb = self.build_flags(force=force)
            flags = self.flags_summary(nfl, cmb)
            self.store.export_csv(flags, "quality_flags_summary", "exclusion flags by domain and cohort")
        except FileNotFoundError as exc:
            log.error("flags not built: %s", exc)
        summary = self.remediate(failures, diag, nfl, cmb)
        self.store.export_csv(failures.with_columns(pl.col("severity").cast(pl.String)), "quality_failures",
                              "failed / warning checks with triage columns")
        self.store.export_csv(summary, "quality_remediation_summary", "status, root cause and remediation per failure")
        fig = self.plot_breakdown(summary)
        self.write_report(failures, summary, diag, flags, fig, sql_files)
        return {"failures": failures, "summary": summary, "flags_summary": flags if flags is not None else pl.DataFrame()}


def main(argv: list[str] | None = None) -> int:
    from ..data.config import configure_runtime

    ap = argparse.ArgumentParser(description="Phase 02 · data-quality remediation")
    ap.add_argument("--force", action="store_true", help="rebuild flag datasets even if current")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    out = QualityRemediation(cfg).run(force=a.force)
    print(out["summary"].select("severity", "table", "columns", "status", "affected_rows_estimate"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
