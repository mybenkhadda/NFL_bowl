"""Bounded data access.

* Pickers (athletes, drills, games, plays) read the compact serving snapshot
  (``lab_db.lab.*``) — never the frame tables.
* Frames are read for ONE trajectory (Combine attempt) or ONE play at a time, from a single
  silver partition file (partition pruning by path) with key predicates that DuckDB pushes into
  the Parquet scan (row-group pruning: silver is sorted by those keys). Results are capped.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

SERVING_DB = Path(os.environ.get("LAB_SERVING_DB", "/srv/nfl-lab/serving/nfl_lab.duckdb"))
SILVER_ROOT = Path(os.environ.get("LAB_SILVER_ROOT", "/data/warehouse/silver"))
DUCKDB_MEMORY = os.environ.get("LAB_DUCKDB_MEMORY", "768MB")
MAX_FRAMES = int(os.environ.get("LAB_MAX_TRAJECTORY_FRAMES", "6000"))
MAX_PLAY_ROWS = MAX_FRAMES * 25          # every tracked entity of one play
PICKER_LIMIT = 5000


class DataUnavailable(RuntimeError):
    """A required input (serving snapshot or silver partition) is missing."""


def _lit(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def serving_mtime() -> float:
    return SERVING_DB.stat().st_mtime if SERVING_DB.exists() else 0.0


@st.cache_resource(show_spinner=False, max_entries=2)
def _connection(mtime: float):
    """One in-memory DuckDB per snapshot version (the mtime key reconnects after a refresh)."""
    if not SERVING_DB.exists():
        raise DataUnavailable(f"{SERVING_DB} not found — run `python -m nflbdb.lab.build` on the host")
    con = duckdb.connect(":memory:", config={"memory_limit": DUCKDB_MEMORY, "threads": 2})
    con.execute(f"ATTACH {_lit(SERVING_DB)} AS lab_db (READ_ONLY)")
    return con


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    cur = _connection(serving_mtime()).cursor()      # cursors are safe across Streamlit threads
    try:
        return cur.execute(sql, params or []).df()
    finally:
        cur.close()


def table_columns(table: str) -> set[str]:
    return set(query(f"SELECT * FROM lab_db.lab.{table} LIMIT 0").columns)


def reset_cache() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


# --------------------------------------------------------------------------- snapshot info
@st.cache_data(show_spinner=False)
def kpi(_mtime: float) -> pd.DataFrame:
    return query("SELECT * FROM lab_db.lab.kpi_snapshot")


@st.cache_data(show_spinner=False)
def availability(_mtime: float) -> pd.DataFrame:
    return query("SELECT dataset, available, note FROM lab_db.lab.availability ORDER BY dataset")


# --------------------------------------------------------------------------- Combine pickers
@st.cache_data(show_spinner=False)
def combine_drills(_mtime: float) -> pd.DataFrame:
    return query("""SELECT drill_name, sum(n_athletes) AS n_athletes, sum(n_attempts) AS n_attempts
                    FROM lab_db.lab.drill_participation GROUP BY drill_name ORDER BY n_athletes DESC""")


@st.cache_data(show_spinner=False)
def combine_athletes(_mtime: float, drill: str, positions: tuple[str, ...], cohorts: tuple[int, ...]) -> pd.DataFrame:
    where, params = ["drill_name = ?"], [drill]
    if positions:
        where.append(f"position_group IN ({', '.join('?' for _ in positions)})")
        params += list(positions)
    if cohorts:
        where.append(f"cohort IN ({', '.join('?' for _ in cohorts)})")
        params += list(cohorts)
    return query(f"""SELECT nfl_id, any_value(display_name) AS display_name, any_value(position_group) AS position_group,
                            any_value(cohort) AS cohort, count(*) AS n_attempts, max(max_s) AS best_max_s
                     FROM lab_db.lab.combine_attempt WHERE {' AND '.join(where)}
                     GROUP BY nfl_id ORDER BY display_name LIMIT {PICKER_LIMIT}""", params)


@st.cache_data(show_spinner=False)
def combine_attempts(_mtime: float, nfl_id: int, drill: str) -> pd.DataFrame:
    return query("""SELECT event_id, attempt, attempt_seq, cohort, n_frames, duration_s, max_s, max_s_mph, max_a,
                           time_to_max_s_s, path_length_yds, max_s_pct_in_drill, exclude_kinematics, exclusion_reasons
                    FROM lab_db.lab.combine_attempt WHERE nfl_id = ? AND drill_name = ? ORDER BY attempt_seq""",
                 [int(nfl_id), drill])


# --------------------------------------------------------------------------- NFL pickers
@st.cache_data(show_spinner=False)
def nfl_seasons(_mtime: float) -> list[int]:
    return query("SELECT DISTINCT season FROM lab_db.lab.nfl_game WHERE n_plays_tracked > 0 ORDER BY 1")["season"].tolist()


@st.cache_data(show_spinner=False)
def nfl_games(_mtime: float, season: int) -> pd.DataFrame:
    return query("""SELECT game_id, game_label, n_plays_tracked, n_athletes_tracked FROM lab_db.lab.nfl_game
                    WHERE season = ? AND n_plays_tracked > 0 ORDER BY game_id""", [int(season)])


@st.cache_data(show_spinner=False)
def nfl_plays(_mtime: float, game_id: int) -> pd.DataFrame:
    return query("SELECT * FROM lab_db.lab.nfl_play WHERE game_id = ? ORDER BY play_id", [int(game_id)])


@st.cache_data(show_spinner=False)
def nfl_play_athletes(_mtime: float, game_id: int, play_id: int) -> pd.DataFrame:
    return query("""SELECT * FROM lab_db.lab.nfl_player_play WHERE game_id = ? AND play_id = ?
                    ORDER BY position_group, display_name""", [int(game_id), int(play_id)])


@st.cache_data(show_spinner=False)
def nfl_athletes(_mtime: float) -> pd.DataFrame:
    return query("""SELECT nfl_id, display_name, position_group, nfl_n_plays_tracked FROM lab_db.lab.athlete
                    WHERE has_nfl_tracking ORDER BY display_name""")


@st.cache_data(show_spinner=False)
def nfl_athlete_plays(_mtime: float, nfl_id: int, season: int | None, limit: int = 200) -> pd.DataFrame:
    where, params = ["nfl_id = ?"], [int(nfl_id)]
    if season is not None:
        where.append("season = ?")
        params.append(int(season))
    return query(f"""SELECT season, game_id, play_id, game_label, play_label, max_s, distance_yds, n_frames
                     FROM lab_db.lab.nfl_player_play WHERE {' AND '.join(where)}
                     ORDER BY max_s DESC NULLS LAST LIMIT {int(limit)}""", params)


# --------------------------------------------------------------------------- frames (bounded)
def partition_files(table: str, key: str, value) -> list[Path]:
    d = SILVER_ROOT / table / f"{key}={value}"
    files = sorted(p for p in d.glob("*.parquet") if not p.name.startswith("."))
    if not files:
        raise DataUnavailable(f"no silver partition {d} (is warehouse/silver mounted?)")
    return files


def _frames(sql: str, params: list) -> pd.DataFrame:
    cur = _connection(serving_mtime()).cursor()
    try:
        return cur.execute(sql, params).df()
    finally:
        cur.close()


@st.cache_data(show_spinner=False, max_entries=64)
def combine_frames(cohort: int, nfl_id: int, event_id: str, attempt: int) -> tuple[pd.DataFrame, bool]:
    """PLAYER frames of one Combine attempt. Returns (frames, truncated)."""
    files = partition_files("combine_tracking", "draft_year", "null" if cohort in (None, -1) else int(cohort))
    listing = "[" + ", ".join(_lit(f) for f in files) + "]"
    df = _frames(f"""SELECT frame_id, frame_offset_ms / 1000.0 AS t_s, ts, x, y, s, a, dis, dir
                     FROM read_parquet({listing})
                     WHERE nfl_id = ? AND event_id = ? AND attempt = ? AND entity_type = 'PLAYER'
                     ORDER BY ts LIMIT {MAX_FRAMES + 1}""", [int(nfl_id), str(event_id), int(attempt)])
    return df.head(MAX_FRAMES), len(df) > MAX_FRAMES


@st.cache_data(show_spinner=False, max_entries=32)
def nfl_play_frames(season: int, game_id: int, play_id: int) -> tuple[pd.DataFrame, bool]:
    """Every tracked entity (dataset athletes + football) of one play. Returns (frames, truncated)."""
    files = partition_files("game_tracking", "season", int(season))
    listing = "[" + ", ".join(_lit(f) for f in files) + "]"
    df = _frames(f"""SELECT nfl_id, is_football, frame_id, frame_offset_ms / 1000.0 AS t_s, ts, x, y, s, a, dis, o, dir,
                            event, play_direction, x_std, y_std, dir_std, o_std
                     FROM read_parquet({listing})
                     WHERE game_id = ? AND play_id = ?
                     ORDER BY is_football, nfl_id, ts LIMIT {MAX_PLAY_ROWS + 1}""", [int(game_id), int(play_id)])
    return df.head(MAX_PLAY_ROWS), len(df) > MAX_PLAY_ROWS
