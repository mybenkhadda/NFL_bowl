"""Silver layer — validated, standardised, analysis-safe tables.

Rules (every rule is data-driven and its evidence is saved under ``warehouse/catalog``):

1. **Types** — :class:`TypePlanner` inspects bronze: Utf8 columns whose distinct values are a
   subset of {TRUE, FALSE} become BOOLEAN; Float64 columns that are integral *and* semantically
   counts/categories become BIGINT. Everything else keeps its bronze type. Plan → JSON.
2. **player_play split** — :class:`PlayerPlaySplitter` measures, per column, the maximum number
   of distinct values within a play (NULL counted as a value). Constant-within-play columns
   move to ``silver.plays`` (one row per play); the rest stay in ``silver.player_play``.
   Exceptions: all-null columns (no evidence → ``no_data``) and player role attributes
   (``PLAYER_ATTRIBUTES``: team_abbr, lined_up_position) always stay in ``player_play``.
   Types come from the actual bronze schema, so columns kept as String by csv_schema
   resolution (e.g. ``recovered_fumble`` = TRUE/FALSE) become BOOLEAN here.
3. **Timestamps** — ``CAST(time AS TIMESTAMP)`` is strict: one unparseable value aborts the
   build (validated beforehand by ``validate.timestamp_parse``).
4. **frame_id** — the raw game tracking has no frame column. ``frame_id`` is derived from
   the time offset to the play's first timestamp: ``round(Δms / nominal_interval) + 1``. It is
   therefore aligned across all players of a play. ``frame_offset_ms`` is kept so residuals
   (irregular sampling) remain auditable. Combine frames use the attempt's first timestamp.
5. **Coordinates** — raw ``x, y, o, dir`` are kept unchanged. ``*_std`` columns rotate plays
   with ``play_direction = 'left'`` by 180° so the offense always moves toward +x
   (x' = 120 − x, y' = 53.3 − y, angle' = (angle + 180) mod 360). Units unchanged (yards, °).
6. **Dedup** — exact duplicate keys are resolved by keeping the lowest ``_src_row``; the dropped
   rows go to ``silver/_quarantine/``. If duplicate keys carry *conflicting* values in more than
   ``max_key_conflict_fraction`` of rows, the build fails instead of choosing silently.
7. **Football rows** (null ``nfl_id`` in game tracking) are kept and flagged ``is_football``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from .config import Phase01Config, duckdb_connect
from .frames import query_column, query_row, query_scalar
from .reporting import export_table
from .schema import semantic_for
from .sources import SOURCES
from .tables import LayerRefs
from .utils import (atomic_path, atomic_write_json, file_fingerprint, get_logger, read_json, sql_path, stable_hash,
                    utc_now)

log = get_logger(__name__)

SILVER_VERSION = "1"
BOOL_TOKENS = {"TRUE", "FALSE"}
INT_ROLES = {"count", "categorical", "outcome", "identifier"}
INT_NAMES = {"draft_round", "draft_pick_within_round", "draft_overall_pick", "yardline_number", "yards_to_go",
             "penalty_yards", "yards_gained", "pre_penalty_yards_gained", "rec_yards", "rush_yards",
             "pre_snap_home_score", "pre_snap_visitor_score", "bench_reps"}
# Dedup / sort keys for reference tables. combine_results uses (draft_year, nfl_id) so an athlete
# attending two Combines is kept twice rather than treated as a conflicting duplicate.
PLAYER_ATTRIBUTES = {"team_abbr", "lined_up_position"}  # never moved to silver.plays
SILVER_KEYS = {"players": ["nfl_id"], "games": ["game_id"], "combine_results": ["draft_year", "nfl_id"],
               "player_career_successes": ["nfl_id"]}


class SilverIntegrityError(RuntimeError):
    pass


# ======================================================================================
class TypePlanner:
    def __init__(self, cfg: Phase01Config, con, refs: LayerRefs):
        self.cfg, self.con, self.refs = cfg, con, refs

    def plan(self, table: str) -> dict[str, str]:
        """Return {column: target DuckDB type} for columns that change type."""
        spec = SOURCES[table]
        ref = self.refs.sql("bronze", table)
        plan: dict[str, str] = {}
        # The bronze file is the truth: schema resolution may have kept a declared-numeric column
        # as String (e.g. recovered_fumble holds TRUE/FALSE), so never plan from the registry alone.
        actual = self.refs.scan("bronze", table).collect_schema()
        str_cols = [c for c in spec.columns if actual.get(c) == pl.String and c != spec.time_col]
        flt_cols = [c for c in spec.columns if actual.get(c) == pl.Float64
                    and (semantic_for(c).role in INT_ROLES or c in INT_NAMES)]
        if str_cols:
            approx = query_row(self.con,
                "SELECT " + ", ".join(f'approx_count_distinct("{c}") AS "{c}"' for c in str_cols) + f" FROM {ref}"
            )
            small = [c for c in str_cols if (approx[c] or 0) <= 3]
            if small:
                vals = query_row(self.con,
                    "SELECT " + ", ".join(f'list(DISTINCT upper(trim("{c}"))) AS "{c}"' for c in small) + f" FROM {ref}"
                )
                for c in small:
                    distinct = {v for v in (vals[c] or []) if v is not None}
                    if distinct and distinct <= BOOL_TOKENS:
                        plan[c] = "BOOLEAN"
        if flt_cols:
            frac = query_row(self.con,
                "SELECT " + ", ".join(f'count_if("{c}" IS NOT NULL AND "{c}" <> floor("{c}")) AS "{c}"' for c in flt_cols)
                + f" FROM {ref}"
            )
            for c in flt_cols:
                if frac[c] == 0:
                    plan[c] = "BIGINT"
        return plan

    @staticmethod
    def select_list(table: str, plan: dict[str, str], exclude: set[str] = frozenset()) -> str:
        exprs = []
        for c in SOURCES[table].columns:
            if c in exclude:
                continue
            t = plan.get(c)
            if t == "BOOLEAN":  # planned only when every non-null value is TRUE/FALSE → no value lost
                exprs.append(f"CASE upper(trim(\"{c}\")) WHEN 'TRUE' THEN true WHEN 'FALSE' THEN false END AS \"{c}\"")
            elif t == "BIGINT":
                exprs.append(f'CAST("{c}" AS BIGINT) AS "{c}"')
            else:
                exprs.append(f'"{c}"')
        return ", ".join(exprs)


# ======================================================================================
class PlayerPlaySplitter:
    def __init__(self, cfg: Phase01Config, con, refs: LayerRefs):
        self.cfg, self.con, self.refs = cfg, con, refs

    def classify(self) -> pl.DataFrame:
        ref = self.refs.sql("bronze", "player_play")
        keys = {"game_id", "play_id", "nfl_id"}
        cols = [c for c in SOURCES["player_play"].columns if c not in keys]

        def nd(c):  # distinct incl. NULL as its own value
            return f"count(DISTINCT coalesce(CAST(\"{c}\" AS VARCHAR), '<NULL>'))"

        per_play = query_row(self.con,
            "SELECT " + ", ".join(f'max(d_{i}) AS "{c}"' for i, c in enumerate(cols))
            + " FROM (SELECT game_id, play_id, " + ", ".join(f"{nd(c)} AS d_{i}" for i, c in enumerate(cols))
            + f" FROM {ref} GROUP BY game_id, play_id)"
        )
        per_team = query_row(self.con,
            "SELECT " + ", ".join(f'max(d_{i}) AS "{c}"' for i, c in enumerate(cols))
            + " FROM (SELECT game_id, play_id, team_abbr, " + ", ".join(f"{nd(c)} AS d_{i}" for i, c in enumerate(cols))
            + f" FROM {ref} GROUP BY game_id, play_id, team_abbr)"
        )
        non_null = query_row(self.con, "SELECT " + ", ".join(f'count("{c}") AS "{c}"' for c in cols) + f" FROM {ref}")
        rows = []
        for c in cols:
            if c in PLAYER_ATTRIBUTES:
                level = "player_play"          # role attributes: player-level by definition
            elif not non_null[c]:
                level = "no_data"              # all null → no evidence; stays in player_play
            else:
                level = "play" if per_play[c] <= 1 else ("team_play" if per_team[c] <= 1 else "player_play")
            rows.append({"column": c, "max_distinct_per_play": per_play[c], "max_distinct_per_team_play": per_team[c],
                         "non_null_rows": non_null[c], "level": level})
        df = pl.DataFrame(rows)
        export_table(df, self.cfg.paths.tables, "player_play_column_levels")
        atomic_write_json({"generated_at": utc_now(), "levels": {r["column"]: r["level"] for r in rows}},
                          self.cfg.paths.catalog / "player_play_levels.json")
        log.info("player_play columns: %s", df.group_by("level").len().sort("level").to_dicts())
        return df


# ======================================================================================
class SilverBuilder:
    def __init__(self, cfg: Phase01Config, con=None):
        self.cfg = cfg
        self.con = con or duckdb_connect(cfg)
        self.refs = LayerRefs(cfg)
        self.types = TypePlanner(cfg, self.con, self.refs)
        self.state_path = cfg.paths.catalog / "silver_state.json"
        self.results: list[dict] = []

    # ------------------------------------------------------------------ infrastructure
    def _copy(self, select_sql: str, target: Path) -> int:
        with atomic_path(target) as tmp:
            self.con.execute(f"COPY ({select_sql}) TO '{sql_path(tmp)}' ({self.cfg.parquet.duckdb_copy_options()})")
        import pyarrow.parquet as pq

        return pq.ParquetFile(target).metadata.num_rows

    def _input_fingerprint(self, files: list[Path]) -> str:
        return stable_hash([[str(f.name), file_fingerprint(f)] for f in files] + [SILVER_VERSION])

    def _skip(self, output: str, fingerprint: str, force: bool) -> bool:
        st = read_json(self.state_path, {})
        e = st.get(output)
        return (not force) and bool(e) and e.get("fingerprint") == fingerprint and Path(e["path"]).exists()

    def _record(self, output: str, path: Path, fingerprint: str, rows_in: int, rows_out: int, seconds: float,
                extra: dict | None = None) -> dict:
        st = read_json(self.state_path, {})
        entry = {"path": str(path), "fingerprint": fingerprint, "rows_in": rows_in, "rows_out": rows_out,
                 "seconds": round(seconds, 2), "written_at": utc_now(), **(extra or {})}
        st[output] = entry
        atomic_write_json(st, self.state_path)
        self.results.append({"output": output, **{k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
                                                  for k, v in entry.items() if k != "path"}})
        log.info("silver %s: %s → %s rows in %.1fs %s", output, f"{rows_in:,}", f"{rows_out:,}", seconds, extra or "")
        return entry

    def _type_plan(self, table: str) -> dict[str, str]:
        path = self.cfg.paths.catalog / "type_plans.json"
        plans = read_json(path, {})
        plan = self.types.plan(table)
        plans[table] = plan
        atomic_write_json(plans, path)
        return plan

    # ------------------------------------------------------------------ dedup
    def duplicate_stats(self, src: str, key: list[str], value_cols: list[str]) -> dict:
        k = ", ".join(key)
        on = " AND ".join(f"s.{c} IS NOT DISTINCT FROM d.{c}" for c in key)
        vals = ", ".join(f"coalesce(CAST(s.\"{c}\" AS VARCHAR), '∅')" for c in value_cols)
        q = f"""
            WITH d AS (SELECT {k}, count(*) AS n FROM {src} GROUP BY ALL HAVING count(*) > 1)
            SELECT (SELECT count(*) FROM d) AS dup_keys,
                   (SELECT CAST(coalesce(sum(n - 1), 0) AS BIGINT) FROM d) AS dup_rows,
                   (SELECT count(*) FROM (
                        SELECT {', '.join('s.' + c for c in key)}
                        FROM {src} s JOIN d ON {on}
                        GROUP BY ALL HAVING count(DISTINCT concat_ws('|', {vals})) > 1)) AS conflicting_keys
        """
        return query_row(self.con, q)

    def _dedup(self, name: str, src: str, key: list[str], value_cols: list[str], n_rows: int) -> tuple[str, dict]:
        """Return (deduplicated relation SQL, stats). Writes dropped rows to quarantine."""
        stats = self.duplicate_stats(src, key, value_cols)
        if stats["dup_rows"] == 0:
            return src, stats
        conflict_frac = stats["conflicting_keys"] / max(n_rows, 1)
        if conflict_frac > self.cfg.silver.max_key_conflict_fraction:
            raise SilverIntegrityError(
                f"{name}: {stats['conflicting_keys']:,} keys have conflicting duplicate values "
                f"({conflict_frac:.4%} > {self.cfg.silver.max_key_conflict_fraction:.4%}). Refusing to pick one silently."
            )
        part = f"PARTITION BY {', '.join(key)} ORDER BY _src_row"
        self._copy(f"SELECT * FROM {src} QUALIFY row_number() OVER ({part}) > 1",
                   self.cfg.paths.silver / "_quarantine" / f"{name}_duplicates.parquet")
        log.warning("%s: dropped %s duplicate-key rows (%s conflicting keys) → quarantine", name,
                    f"{stats['dup_rows']:,}", f"{stats['conflicting_keys']:,}")
        return f"(SELECT * FROM {src} QUALIFY row_number() OVER ({part}) = 1)", stats

    # ------------------------------------------------------------------ reference tables
    def build_reference(self, table: str, force: bool = False) -> dict | None:
        derived = {
            "players": ", try_cast(birth_date AS DATE) AS birth_date_parsed",
            "games": ", CAST(strptime(CAST(game_id // 100 AS VARCHAR), '%Y%m%d') AS DATE) AS game_date",
        }.get(table, "")
        files = self.refs.files("bronze", table)
        if not files:
            log.warning("silver skip %s: bronze missing", table)
            return None
        fp = self._input_fingerprint(files)
        target = self.cfg.paths.silver / f"{table}.parquet"
        if self._skip(table, fp, force):
            log.info("silver skip (up to date): %s", table)
            return None
        t0 = time.perf_counter()
        plan = self._type_plan(table)
        ref = self.refs.sql("bronze", table)
        n_in = self.refs.num_rows("bronze", table)
        key = SILVER_KEYS.get(table, [])
        src = f"(SELECT {self.types.select_list(table, plan)}{derived}, _src_file, _src_row FROM {ref})"
        stats = {}
        if key:
            src, stats = self._dedup(table, src, key, [c for c in SOURCES[table].columns if c not in key], n_in)
        order = f" ORDER BY {', '.join(key)}" if key else ""
        n_out = self._copy(f"SELECT * FROM {src}{order}", target)
        return self._record(table, target, fp, n_in, n_out, time.perf_counter() - t0,
                            {"type_changes": plan, **{f"dup_{k}": v for k, v in stats.items()}})

    # ------------------------------------------------------------------ player_play → plays + player_play
    def build_player_play(self, force: bool = False) -> None:
        files = self.refs.files("bronze", "player_play")
        if not files:
            log.warning("silver skip player_play: bronze missing")
            return
        fp = self._input_fingerprint(files)
        if self._skip("plays", fp, force) and self._skip("player_play", fp, force):
            log.info("silver skip (up to date): plays, player_play")
            return
        t0 = time.perf_counter()
        levels = PlayerPlaySplitter(self.cfg, self.con, self.refs).classify()
        play_cols = levels.filter(pl.col("level") == "play")["column"].to_list()
        plan = self._type_plan("player_play")
        ref = self.refs.sql("bronze", "player_play")
        n_in = self.refs.num_rows("bronze", "player_play")
        typed = f"(SELECT {self.types.select_list('player_play', plan)}, _src_file, _src_row FROM {ref})"
        key = ["game_id", "play_id", "nfl_id"]
        typed, stats = self._dedup("player_play", typed, key,
                                   [c for c in SOURCES["player_play"].columns if c not in key], n_in)

        clock_sel = (", CASE WHEN regexp_matches(q.game_clock, '^\\d+:\\d{2}$') THEN "
                     "CAST(split_part(q.game_clock, ':', 1) AS INTEGER) * 60 + CAST(split_part(q.game_clock, ':', 2) AS INTEGER) "
                     "END AS game_clock_seconds") if "game_clock" in play_cols else ""
        pc = "".join(f', any_value(p."{c}") AS "{c}"' for c in play_cols)
        inner = (f"SELECT p.game_id, p.play_id, count(*) AS n_player_rows{pc} "
                 f"FROM {typed} p GROUP BY p.game_id, p.play_id")
        if self.refs.exists("silver", "games"):
            plays_sql = (f"SELECT q.*{clock_sel}, g.season, g.week, g.season_type FROM ({inner}) q "
                         f"LEFT JOIN {self.refs.sql('silver', 'games')} g ON q.game_id = g.game_id "
                         f"ORDER BY q.game_id, q.play_id")
        else:
            log.warning("silver.games missing: plays will lack season/week")
            plays_sql = f"SELECT q.*{clock_sel} FROM ({inner}) q ORDER BY q.game_id, q.play_id"
        n_plays = self._copy(plays_sql, self.cfg.paths.silver / "plays.parquet")
        self._record("plays", self.cfg.paths.silver / "plays.parquet", fp, n_in, n_plays, time.perf_counter() - t0,
                     {"play_level_columns": len(play_cols)})

        t1 = time.perf_counter()
        keep = [c for c in SOURCES["player_play"].columns if c not in play_cols]
        pp_sql = f"SELECT {', '.join(chr(34) + c + chr(34) for c in keep)}, _src_file, _src_row FROM {typed} ORDER BY game_id, play_id, nfl_id"
        n_pp = self._copy(pp_sql, self.cfg.paths.silver / "player_play.parquet")
        self._record("player_play", self.cfg.paths.silver / "player_play.parquet", fp, n_in, n_pp,
                     time.perf_counter() - t1, {"type_changes": plan, **{f"dup_{k}": v for k, v in stats.items()}})

    # ------------------------------------------------------------------ tracking
    def _std_coords(self, has_o: bool) -> str:
        L, W = self.cfg.tracking.field_length_yds, self.cfg.tracking.field_width_yds
        left = "pl.play_direction = 'left'"
        exprs = [
            f"CASE WHEN {left} THEN {L} - t.x ELSE t.x END AS x_std",
            f"CASE WHEN {left} THEN {W} - t.y ELSE t.y END AS y_std",
            f"CASE WHEN {left} THEN fmod(t.dir + 180, 360) ELSE t.dir END AS dir_std",
        ]
        if has_o:
            exprs.append(f"CASE WHEN {left} THEN fmod(t.o + 180, 360) ELSE t.o END AS o_std")
        return ", ".join(exprs)

    def build_game_tracking(self, force: bool = False) -> None:
        if not self.refs.exists("silver", "plays"):
            raise SilverIntegrityError("build silver.plays before game tracking (play_direction needed)")
        nom = self.cfg.tracking.nominal_interval_ms
        plays_ref = self.refs.sql("silver", "plays")
        has_dir = "play_direction" in self.con.sql(f"SELECT * FROM {plays_ref} LIMIT 0").columns
        if not has_dir:
            log.warning("play_direction is not play-level in player_play: *_std columns will equal raw values")
        for bronze_file in self.refs.files("bronze", "game_tracking"):
            season = bronze_file.parent.name.split("=")[-1]
            output = f"game_tracking/season={season}"
            target = self.cfg.paths.silver / "game_tracking" / f"season={season}" / "part-0.parquet"
            fp = self._input_fingerprint([bronze_file]) + stable_hash(self.cfg.tracking.__dict__)
            if self._skip(output, fp, force):
                log.info("silver skip (up to date): %s", output)
                continue
            t0 = time.perf_counter()
            raw = f"read_parquet('{sql_path(bronze_file)}')"
            n_in = query_scalar(self.con, f"SELECT count(*) FROM {raw}")
            key = ["game_id", "play_id", "nfl_id", "time"]
            src, stats = self._dedup(output.replace("/", "_"), raw, key,
                                     ["x", "y", "s", "a", "dis", "o", "dir", "event"], n_in)
            order = " ORDER BY t.game_id, t.play_id, t.nfl_id, t.ts" if self.cfg.silver.cluster_sort else ""
            pl_join = f"LEFT JOIN {plays_ref} pl ON pl.game_id = t.game_id AND pl.play_id = t.play_id" if has_dir else \
                "LEFT JOIN (SELECT NULL::BIGINT game_id, NULL::BIGINT play_id, NULL::VARCHAR play_direction WHERE FALSE) pl " \
                "ON pl.game_id = t.game_id AND pl.play_id = t.play_id"
            sql = f"""
                WITH t AS (SELECT * EXCLUDE (time), CAST(time AS TIMESTAMP) AS ts FROM {src}),
                     p0 AS (SELECT game_id, play_id, min(ts) AS play_t0 FROM t GROUP BY ALL)
                SELECT t.game_id, t.play_id, t.nfl_id, (t.nfl_id IS NULL) AS is_football,
                       CAST(round((epoch_ms(t.ts) - epoch_ms(p0.play_t0)) / {nom}) AS INTEGER) + 1 AS frame_id,
                       epoch_ms(t.ts) - epoch_ms(p0.play_t0) AS frame_offset_ms,
                       t.ts, t.x, t.y, t.s, t.a, t.dis, t.o, t.dir, t.event,
                       pl.play_direction, {self._std_coords(True)},
                       t._src_file, t._src_row
                FROM t JOIN p0 ON t.game_id = p0.game_id AND t.play_id = p0.play_id
                {pl_join}{order}
            """
            n_out = self._copy(sql, target)
            check = query_row(self.con, f"""
                SELECT count_if(abs(frame_offset_ms - (frame_id - 1) * {nom}) > {nom * 0.25}) AS off_grid,
                       count_if(play_direction IS NULL) AS no_direction,
                       count(*) - count(DISTINCT (game_id, play_id, nfl_id, frame_id)) AS frame_key_collisions
                FROM read_parquet('{sql_path(target)}')
            """)
            self._record(output, target, fp, n_in, n_out, time.perf_counter() - t0,
                         {**{f"dup_{k}": v for k, v in stats.items()}, **check})
            if check["frame_key_collisions"]:
                log.warning("%s: %s rows share (game, play, nfl_id, frame_id) — irregular sampling; use ts as the key",
                            output, f"{check['frame_key_collisions']:,}")

    def build_combine_tracking(self, force: bool = False) -> None:
        files = self.refs.files("bronze", "combine_tracking")
        if not files:
            log.warning("silver skip combine_tracking: bronze missing")
            return
        nom = self.cfg.tracking.nominal_interval_ms
        ref = self.refs.sql("bronze", "combine_tracking")
        fp = self._input_fingerprint(files) + stable_hash(self.cfg.tracking.__dict__)
        years = query_column(self.con, f"SELECT DISTINCT draft_year FROM {ref} ORDER BY 1")
        traj = ["event_id", "nfl_id", "attempt"]
        for year in years:
            label = "null" if year is None else str(year)
            output = f"combine_tracking/draft_year={label}"
            target = self.cfg.paths.silver / "combine_tracking" / f"draft_year={label}" / "part-0.parquet"
            if self._skip(output, fp, force):
                log.info("silver skip (up to date): %s", output)
                continue
            t0 = time.perf_counter()
            pred = "draft_year IS NULL" if year is None else f"draft_year = {year}"
            raw = f"(SELECT * EXCLUDE (draft_year) FROM {ref} WHERE {pred})"
            n_in = query_scalar(self.con, f"SELECT count(*) FROM {raw}")
            src, stats = self._dedup(output.replace("/", "_").replace("=", "_"), raw,
                                     ["event_id", "nfl_id", "attempt", "time", "entity_type"],
                                     ["x", "y", "s", "a", "dis", "dir", "drill_type", "drill_name"], n_in)
            on = " AND ".join(f"t.{c} IS NOT DISTINCT FROM a0.{c}" for c in traj)
            order = " ORDER BY t.nfl_id, t.drill_name, t.event_id, t.attempt, t.ts" if self.cfg.silver.cluster_sort else ""
            sql = f"""
                WITH t AS (SELECT * EXCLUDE (time), CAST(time AS TIMESTAMP) AS ts FROM {src}),
                     a0 AS (SELECT {', '.join(traj)}, min(ts) AS attempt_t0 FROM t GROUP BY ALL)
                SELECT t.event_id, t.nfl_id, t.entity_type, t.drill_type, t.drill_name, t.attempt,
                       CAST(round((epoch_ms(t.ts) - epoch_ms(a0.attempt_t0)) / {nom}) AS INTEGER) + 1 AS frame_id,
                       epoch_ms(t.ts) - epoch_ms(a0.attempt_t0) AS frame_offset_ms,
                       t.ts, t.x, t.y, t.s, t.a, t.dis, t.dir, t._src_file, t._src_row
                FROM t JOIN a0 ON {on}{order}
            """
            n_out = self._copy(sql, target)
            self._record(output, target, fp, n_in, n_out, time.perf_counter() - t0,
                         {f"dup_{k}": v for k, v in stats.items()})

    # ------------------------------------------------------------------ orchestration
    def build_all(self, force: bool = False) -> pl.DataFrame:
        for t in ("players", "games", "combine_results", "player_career_successes"):
            self.build_reference(t, force)
        self.build_player_play(force)
        self.build_game_tracking(force)
        self.build_combine_tracking(force)
        df = pl.DataFrame(self.results, infer_schema_length=None) if self.results else pl.DataFrame()
        if not df.is_empty():
            export_table(df, self.cfg.paths.tables, "silver_build_summary")
        return df
