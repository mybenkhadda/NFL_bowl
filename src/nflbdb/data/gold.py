"""Gold layer — analytics-ready dimensions, compact facts and athlete-level marts.

Design rules
* Frame-level tracking is NOT copied into gold: ``fact_nfl_tracking`` / ``fact_combine_tracking``
  are catalog views over silver (natural grain, partitioned, sorted).
* Facts aggregate frames to their natural parent (attempt; player×play) — orders of magnitude
  smaller, so marts and research joins never touch frames.
* Every table has a declared grain + primary key; PK uniqueness is asserted after each build
  and recorded in ``catalog/gold_manifest``.
* The whole tracked window of a play is summarised (pre-snap included). Event-windowed
  features (e.g. snap → +2 s) belong to Phase 02 feature engineering.
* ``career_*`` columns in the research base are OUTCOMES. They are prefixed so they cannot be
  confused with predictors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .config import Phase01Config, duckdb_connect
from .frames import query_row, query_scalar
from .reporting import export_table
from .sources import SOURCES
from .tables import LayerRefs
from .utils import atomic_path, get_logger, sql_path, utc_now

log = get_logger(__name__)


@dataclass(frozen=True)
class GoldTable:
    name: str
    grain: str
    pk: tuple[str, ...]
    sources: str
    description: str
    partition: str = ""


GOLD_TABLES: dict[str, GoldTable] = {t.name: t for t in [
    GoldTable("dim_athlete", "one row per athlete (any source)", ("nfl_id",),
              "silver.players, combine_results, combine_tracking, game_tracking, player_play, career",
              "Biography, draft, positions and source-presence flags."),
    GoldTable("dim_game", "one row per game", ("game_id",), "silver.games", "Game metadata + game_date."),
    GoldTable("dim_play", "one row per play", ("game_id", "play_id"), "silver.plays",
              "Play-level context (columns proven constant within play)."),
    GoldTable("dim_season", "one row per season × season_type", ("season", "season_type"), "silver.games",
              "Season calendar."),
    GoldTable("dim_position", "one row per (source, position code)", ("source", "position"),
              "silver.combine_results, players, player_play", "Position vocabularies with counts."),
    GoldTable("dim_combine_drill", "one row per drill", ("drill_type", "drill_name"), "gold.fact_combine_attempt",
              "Drill catalogue with coverage."),
    GoldTable("fact_combine_attempt", "one row per athlete × drill attempt", ("event_id", "nfl_id", "attempt"),
              "silver.combine_tracking (PLAYER rows)", "Attempt-level kinematic summaries."),
    GoldTable("fact_nfl_play_participation", "one row per player × play (tracked)", ("game_id", "play_id", "nfl_id"),
              "silver.game_tracking ⨝ silver.player_play", "Player-play kinematic summaries + role.", "season"),
    GoldTable("mart_athlete_combine_profile", "one row per athlete × drill", ("nfl_id", "drill_type", "drill_name"),
              "gold.fact_combine_attempt", "Best / median attempt metrics per drill."),
    GoldTable("mart_athlete_nfl_profile", "one row per athlete × season", ("nfl_id", "season"),
              "gold.fact_nfl_play_participation", "Season exposure and movement distribution."),
    GoldTable("mart_combine_nfl_coverage", "one row per position group × cohort", ("position_group", "cohort"),
              "gold.dim_athlete", "Linkage coverage counts and rates."),
    GoldTable("mart_athlete_research_base", "one row per athlete", ("nfl_id",),
              "dim_athlete + combine_results + marts + career",
              "Compact research table: cmb_* traditional, ctrk_* Combine tracking, nfl_* exposure, career_* OUTCOMES."),
]}


class GoldBuilder:
    def __init__(self, cfg: Phase01Config, con=None):
        self.cfg = cfg
        self.con = con or duckdb_connect(cfg)
        self.refs = LayerRefs(cfg)
        self.manifest: list[dict] = []

    def s(self, name: str) -> str:
        return self.refs.sql("silver", name)

    def g(self, name: str) -> str:
        return self.refs.sql("gold", name)

    def has(self, layer: str, name: str) -> bool:
        return self.refs.exists(layer, name)

    # ------------------------------------------------------------------ write + assert
    def _write(self, name: str, sql: str, target: Path | None = None) -> int:
        target = target or self.cfg.paths.gold / f"{name}.parquet"
        t0 = time.perf_counter()
        with atomic_path(target) as tmp:
            self.con.execute(f"COPY ({sql}) TO '{sql_path(tmp)}' ({self.cfg.parquet.duckdb_copy_options()})")
        spec = GOLD_TABLES[name]
        pk = ", ".join(spec.pk)
        chk = query_row(self.con, f"SELECT count(*) AS n, count(DISTINCT ({pk})) AS n_pk FROM read_parquet('{sql_path(target)}')")
        unique = chk["n"] == chk["n_pk"]
        self.manifest.append({
            "table": name if not spec.partition else f"{name}/{target.parent.name}", "grain": spec.grain,
            "pk": pk, "rows": chk["n"], "pk_unique": unique, "bytes": target.stat().st_size,
            "seconds": round(time.perf_counter() - t0, 2), "built_at": utc_now(),
        })
        lvl = log.info if unique else log.error
        lvl("gold %s: %s rows, PK (%s) unique=%s", self.manifest[-1]["table"], f"{chk['n']:,}", pk, unique)
        if not unique:
            raise RuntimeError(f"gold.{name}: primary key ({pk}) is not unique — grain assumption violated")
        return chk["n"]

    # ------------------------------------------------------------------ dimensions
    def build_dim_athlete(self) -> None:
        ids, flags = [], []
        sources = {
            "players": ("players", "TRUE"), "combine_results": ("combine_results", "TRUE"),
            "combine_tracking": ("combine_tracking", "entity_type = 'PLAYER'"),
            "nfl_tracking": ("game_tracking", "NOT is_football"), "player_play": ("player_play", "TRUE"),
            "career": ("player_career_successes", "TRUE"),
        }
        ctes = []
        for alias, (table, pred) in sources.items():
            if self.has("silver", table):
                ctes.append(f"ids_{alias} AS (SELECT DISTINCT nfl_id FROM {self.s(table)} WHERE nfl_id IS NOT NULL AND {pred})")
            else:
                ctes.append(f"ids_{alias} AS (SELECT NULL::BIGINT AS nfl_id WHERE FALSE)")
            ids.append(f"SELECT nfl_id FROM ids_{alias}")
            flags.append(f"a.nfl_id IN (SELECT nfl_id FROM ids_{alias}) AS has_{alias}")
        cr = (f", cr AS (SELECT nfl_id, count(*) AS n_combine_rows, max(draft_year) AS combine_draft_year, "
              f"arg_max(combine_position, draft_year) AS combine_position FROM {self.s('combine_results')} GROUP BY 1)")
        sql = f"""
            WITH {', '.join(ctes)}{cr}, a AS ({' UNION '.join(ids)})
            SELECT a.nfl_id, p.display_name, p.nfl_position, p.birth_date_parsed AS birth_date, p.college_name,
                   p.college_conference, p.draft_year, p.draft_round, p.draft_pick_within_round, p.draft_overall_pick,
                   cr.combine_position, cr.combine_draft_year, coalesce(cr.n_combine_rows, 0) AS n_combine_rows,
                   coalesce(cr.combine_position, p.nfl_position, 'UNKNOWN') AS position_group,
                   coalesce(cr.combine_draft_year, p.draft_year) AS cohort,
                   {', '.join(flags)}
            FROM a LEFT JOIN {self.s('players')} p USING (nfl_id) LEFT JOIN cr USING (nfl_id)
            ORDER BY a.nfl_id
        """
        self._write("dim_athlete", sql)

    def build_dim_game_play_season(self) -> None:
        self._write("dim_game", f"SELECT * EXCLUDE (_src_file, _src_row) FROM {self.s('games')} ORDER BY game_id")
        self._write("dim_play", f"SELECT * FROM {self.s('plays')} ORDER BY game_id, play_id")
        self._write("dim_season", f"""
            SELECT season, season_type, count(*) AS n_games, count(DISTINCT week) AS n_weeks,
                   min(game_date) AS first_game_date, max(game_date) AS last_game_date
            FROM {self.s('games')} GROUP BY ALL ORDER BY season, season_type
        """)

    def build_dim_position(self) -> None:
        parts = [f"SELECT 'combine_position' AS source, combine_position AS position, count(DISTINCT nfl_id) AS n_athletes "
                 f"FROM {self.s('combine_results')} GROUP BY 2",
                 f"SELECT 'nfl_position', nfl_position, count(DISTINCT nfl_id) FROM {self.s('players')} GROUP BY 2"]
        if self.has("silver", "player_play"):
            parts.append(f"SELECT 'lined_up_position', lined_up_position, count(DISTINCT nfl_id) "
                         f"FROM {self.s('player_play')} GROUP BY 2")
        self._write("dim_position", "SELECT * FROM (" + " UNION ALL ".join(parts) +
                    ") WHERE position IS NOT NULL ORDER BY source, n_athletes DESC")

    # ------------------------------------------------------------------ facts
    def build_fact_combine_attempt(self) -> None:
        sql = f"""
            SELECT event_id, nfl_id, attempt, draft_year,
                   any_value(drill_type) AS drill_type, any_value(drill_name) AS drill_name,
                   count(DISTINCT drill_name) AS n_drill_names,
                   count(*) AS n_frames, min(ts) AS start_ts,
                   (epoch_ms(max(ts)) - epoch_ms(min(ts))) / 1000.0 AS duration_s,
                   sum(dis) AS path_length_yds,
                   sqrt(power(arg_max(x, ts) - arg_min(x, ts), 2) + power(arg_max(y, ts) - arg_min(y, ts), 2)) AS net_displacement_yds,
                   max(s) AS max_s, avg(s) AS mean_s, max(a) AS max_a,
                   (epoch_ms(arg_max(ts, s)) - epoch_ms(min(ts))) / 1000.0 AS time_to_max_s_s,
                   count_if(abs(frame_offset_ms - (frame_id - 1) * {self.cfg.tracking.nominal_interval_ms})
                            > {self.cfg.tracking.nominal_interval_ms * 0.25}) AS n_off_grid_frames
            FROM {self.s('combine_tracking')}
            WHERE entity_type = 'PLAYER' AND nfl_id IS NOT NULL
            GROUP BY event_id, nfl_id, attempt, draft_year
            ORDER BY nfl_id, drill_name, event_id, attempt
        """
        self._write("fact_combine_attempt", sql)
        n_multi = query_scalar(self.con, f"SELECT count_if(n_drill_names > 1) FROM {self.g('fact_combine_attempt')}")
        if n_multi:
            log.warning("fact_combine_attempt: %s attempts span several drill_names — attempt grain may need drill_name",
                        f"{n_multi:,}")

    def build_fact_nfl_play_participation(self) -> None:
        pp = self.has("silver", "player_play")
        role = (f"LEFT JOIN (SELECT game_id, play_id, nfl_id, team_abbr, lined_up_position FROM {self.s('player_play')}) r "
                f"ON r.game_id = f.game_id AND r.play_id = f.play_id AND r.nfl_id = f.nfl_id") if pp else ""
        role_cols = ", r.team_abbr, r.lined_up_position, (r.nfl_id IS NOT NULL) AS in_player_play" if pp else ""
        for f in self.refs.files("silver", "game_tracking"):
            season = f.parent.name.split("=")[-1]
            sql = f"""
                WITH f AS (
                    SELECT game_id, play_id, nfl_id, count(*) AS n_frames,
                           (epoch_ms(max(ts)) - epoch_ms(min(ts))) / 1000.0 AS tracked_duration_s,
                           sum(dis) AS distance_yds, max(s) AS max_s, avg(s) AS mean_s,
                           approx_quantile(s, 0.95) AS p95_s, max(a) AS max_a,
                           approx_quantile(a, 0.95) AS p95_a,
                           max(x_std) - min(x_std) AS x_std_range_yds
                    FROM read_parquet('{sql_path(f)}')
                    WHERE NOT is_football
                    GROUP BY game_id, play_id, nfl_id
                )
                SELECT {season} AS season, f.*{role_cols}
                FROM f {role}
                ORDER BY f.game_id, f.play_id, f.nfl_id
            """
            target = self.cfg.paths.gold / "fact_nfl_play_participation" / f"season={season}" / "part-0.parquet"
            self._write("fact_nfl_play_participation", sql, target)

    def build_dim_combine_drill(self) -> None:
        self._write("dim_combine_drill", f"""
            SELECT drill_type, drill_name, count(DISTINCT nfl_id) AS n_athletes, count(*) AS n_attempts,
                   list(DISTINCT draft_year ORDER BY draft_year) AS draft_years,
                   median(n_frames) AS median_frames, median(duration_s) AS median_duration_s,
                   median(path_length_yds) AS median_path_length_yds, median(max_s) AS median_max_s
            FROM {self.g('fact_combine_attempt')} GROUP BY ALL ORDER BY n_athletes DESC
        """)

    # ------------------------------------------------------------------ marts
    def build_marts(self) -> None:
        if self.has("gold", "fact_combine_attempt"):
            self._write("mart_athlete_combine_profile", f"""
                SELECT nfl_id, drill_type, drill_name, count(*) AS n_attempts,
                       max(max_s) AS best_max_s, median(max_s) AS median_max_s,
                       max(max_a) AS best_max_a, min(duration_s) AS min_duration_s,
                       median(path_length_yds) AS median_path_length_yds, min(time_to_max_s_s) AS min_time_to_max_s_s
                FROM {self.g('fact_combine_attempt')} GROUP BY ALL ORDER BY nfl_id, drill_name
            """)
        if self.has("gold", "fact_nfl_play_participation"):
            self._write("mart_athlete_nfl_profile", f"""
                SELECT nfl_id, season, count(DISTINCT game_id) AS n_games, count(*) AS n_plays_tracked,
                       sum(distance_yds) AS total_distance_yds,
                       median(max_s) AS median_play_max_s, quantile_cont(max_s, 0.9) AS p90_play_max_s,
                       max(max_s) AS max_play_max_s, median(max_a) AS median_play_max_a,
                       quantile_cont(max_a, 0.9) AS p90_play_max_a
                       {', mode(lined_up_position) AS modal_lined_up_position, mode(team_abbr) AS modal_team' if self.has('silver', 'player_play') else ''}
                FROM {self.g('fact_nfl_play_participation')} GROUP BY nfl_id, season ORDER BY nfl_id, season
            """)
        self._write("mart_combine_nfl_coverage", f"""
            SELECT position_group, cohort, count(*) AS n_athletes,
                   count_if(has_combine_results) AS n_combine_results, count_if(has_combine_tracking) AS n_combine_tracking,
                   count_if(has_nfl_tracking) AS n_nfl_tracking,
                   count_if(has_combine_tracking AND has_nfl_tracking) AS n_both_tracking,
                   count_if(has_combine_tracking AND has_nfl_tracking) / count_if(has_combine_tracking)::DOUBLE
                       AS match_rate_given_combine_tracking
            FROM {self.g('dim_athlete')} WHERE has_combine_results OR has_combine_tracking
            GROUP BY ALL ORDER BY position_group, cohort
        """)
        self.build_research_base()

    def build_research_base(self) -> None:
        trad = ["combine_height", "combine_weight", "hand_size", "arm_length", "wing_span", "ten_yd_split", "forty",
                "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps", "ngs_athleticism_score"]
        cmb = ", ".join(f"arg_max({c}, draft_year) AS cmb_{c}" for c in trad)
        joins, cols = [], []
        if self.has("gold", "mart_athlete_combine_profile"):
            joins.append(f"""LEFT JOIN (SELECT nfl_id, count(DISTINCT drill_name) AS ctrk_n_drills, sum(n_attempts) AS ctrk_n_attempts,
                                max(best_max_s) AS ctrk_best_max_s, max(best_max_a) AS ctrk_best_max_a
                         FROM {self.g('mart_athlete_combine_profile')} GROUP BY 1) ct USING (nfl_id)""")
            cols.append("ct.* EXCLUDE (nfl_id)")
        if self.has("gold", "mart_athlete_nfl_profile"):
            joins.append(f"""LEFT JOIN (SELECT nfl_id, count(*) AS nfl_n_seasons, sum(n_games) AS nfl_n_games,
                                sum(n_plays_tracked) AS nfl_n_plays_tracked, min(season) AS nfl_first_season,
                                max(p90_play_max_s) AS nfl_best_season_p90_play_max_s
                         FROM {self.g('mart_athlete_nfl_profile')} GROUP BY 1) nf USING (nfl_id)""")
            cols.append("nf.* EXCLUDE (nfl_id)")
        if self.has("silver", "player_career_successes"):
            outcome_cols = [c for c in SOURCES["player_career_successes"].columns if c != "nfl_id"]
            renamed = ", ".join(f'"{c}" AS "career_{c}"' for c in outcome_cols)
            joins.append(f"""LEFT JOIN (SELECT nfl_id, {renamed}
                         FROM {self.s('player_career_successes')}) ca USING (nfl_id)""")
            cols.append("ca.* EXCLUDE (nfl_id)")
        sql = f"""
            SELECT d.nfl_id, d.display_name, d.position_group, d.cohort, d.nfl_position, d.combine_position,
                   d.draft_round, d.draft_overall_pick, d.birth_date,
                   d.has_combine_results, d.has_combine_tracking, d.has_nfl_tracking, d.has_career,
                   c.* EXCLUDE (nfl_id){''.join(', ' + c for c in cols)}
            FROM {self.g('dim_athlete')} d
            LEFT JOIN (SELECT nfl_id, {cmb} FROM {self.s('combine_results')} GROUP BY 1) c USING (nfl_id)
            {' '.join(joins)}
            ORDER BY d.nfl_id
        """
        self._write("mart_athlete_research_base", sql)

    # ------------------------------------------------------------------ orchestration
    def build_all(self) -> pl.DataFrame:
        self.build_dim_athlete()
        self.build_dim_game_play_season()
        self.build_dim_position()
        if self.has("silver", "combine_tracking"):
            self.build_fact_combine_attempt()
            self.build_dim_combine_drill()
        if self.has("silver", "game_tracking"):
            self.build_fact_nfl_play_participation()
        self.build_marts()
        df = pl.DataFrame(self.manifest)
        export_table(df, self.cfg.paths.catalog, "gold_manifest")
        export_table(df, self.cfg.paths.tables, "gold_manifest")
        return df

    @staticmethod
    def table_specs() -> pl.DataFrame:
        return pl.DataFrame([{**t.__dict__, "pk": ", ".join(t.pk)} for t in GOLD_TABLES.values()])
