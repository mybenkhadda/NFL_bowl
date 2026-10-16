"""Synthetic mini-dataset with the exact raw headers, plus planted defects for the tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from nflbdb.data.config import Phase01Config
from nflbdb.data.sources import SOURCES

T0 = datetime(2024, 1, 7, 21, 46, 7, 700000)


def _ts(base: datetime, k: int) -> str:
    return (base + timedelta(milliseconds=100 * k)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def _write(df: pl.DataFrame, path, spec_name: str) -> None:
    cols = SOURCES[spec_name].columns
    for c in cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c))
    df.select(cols).write_csv(path, null_value="NA")


@pytest.fixture()
def synth(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    players = [101, 102, 103, 104]
    _write(pl.DataFrame({"nfl_id": players, "display_name": ["A", "B", "C", "D"], "draft_year": [2023] * 4,
                         "nfl_position": ["WR", "CB", "DE", "WR"], "birth_date": ["2001-01-01"] * 4,
                         "draft_round": [1, 2, None, 3]}), data / "players.csv", "players")
    _write(pl.DataFrame({"draft_year": [2023, 2023, 2023], "nfl_id": [101, 102, 103],
                         "combine_position": ["WR", "DB", "DL"], "forty": [4.4, 4.5, None]}),
           data / "combine_results.csv", "combine_results")
    _write(pl.DataFrame({"game_id": [2023090700, 2023091000], "game_key": [1, 2], "season": [2023, 2023],
                         "season_type": ["REG", "REG"], "week": [1, 1], "home_team_abbr": ["KC", "ATL"],
                         "visitor_team_abbr": ["DET", "CAR"]}), data / "games.csv", "games")
    # player_play: play-level columns constant within play; player-level vary.
    pp = pl.DataFrame({
        "game_id": [2023090700] * 3 + [2023091000] * 2, "play_id": [1, 1, 1, 5, 5],
        "nfl_id": [101, 102, 103, 101, 104], "team_abbr": ["KC", "DET", "DET", "ATL", "ATL"],
        "play_direction": ["left", "left", "left", "right", "right"], "down": [1, 1, 1, 2, 2],
        "game_clock": ["08:42"] * 3 + ["03:48"] * 2, "target": ["TRUE", "FALSE", None, "FALSE", "TRUE"],
        "tackle": [0, 1, 0, None, None],
        # Mirrors the real file: declared Float64 but holds TRUE/FALSE literals (regression).
        "recovered_fumble": ["FALSE", None, "TRUE", None, None],
        # Quoted empty string in a String column must survive as "" (distinct from NA).
        "receiver_alignment": ["3x1", "3x1", "3x1", "", ""],
    })
    _write(pp, data / "player_play.csv", "player_play")
    _write(pl.DataFrame({"nfl_id": [101, 102], "career_games_active": [10, 20]}),
           data / "player_career_successes.csv", "player_career_successes")

    rows = []
    for game, play, pids, base in ((2023090700, 1, [101, 102, 103], T0), (2023091000, 5, [101, 104], T0 + timedelta(days=3))):
        for pid in pids + [None]:  # None = football
            for k in range(10):
                if pid == 103 and k == 5:
                    continue  # planted gap (missing frame)
                rows.append({"game_id": game, "play_id": play, "nfl_id": pid, "time": _ts(base, k),
                             "x": 10.0 + k * 0.5, "y": 20.0, "s": 5.0, "a": 1.0, "dis": 0.5, "o": 90.0, "dir": 90.0})
    gt = pl.DataFrame(rows)
    gt = pl.concat([gt, gt.filter((pl.col("nfl_id") == 102) & (pl.col("x") == 10.0))])  # planted exact duplicate
    _write(gt, data / "game_tracking_2023.csv", "game_tracking")

    crow = []
    for pid in (101, 102):
        for att in (1, 2):
            for k in range(8):
                crow.append({"draft_year": 2023, "event_id": f"E{pid}", "nfl_id": pid, "entity_type": "PLAYER",
                             "time": _ts(T0 + timedelta(minutes=att), k), "drill_type": "SPRINT", "drill_name": "FORTY_YARD_DASH",
                             "attempt": att, "x": k * 0.8, "y": 5.0, "s": 8.0, "a": 3.0, "dis": 0.8, "dir": 90.0})
    _write(pl.DataFrame(crow), data / "combine_tracking.csv", "combine_tracking")

    cfg = Phase01Config.load(project_root=tmp_path, data_root=data, warehouse_root=tmp_path / "wh",
                             reports_root=tmp_path / "reports", tmp_root=tmp_path / ".tmp", threads=2)
    cfg.paths.ensure()
    return cfg
