"""Polars 2.0 compatibility regressions (Phase 03 · M2).

Polars 2.0 raises ``TypeError: not yet implemented: Nested object types`` when a Python list of
expressions is passed to ``with_columns`` / ``select`` / ``agg`` TOGETHER with other positional
expressions (the list is treated as an object literal). A single list as the only argument is
still accepted. Regression: ``features/agility.py::agility_features``.

AUTHORED, NOT EXECUTED. Run manually:  python -m pytest tests/test_polars2_compat.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest

from nflbdb.features.agility import agility_features
from nflbdb.features.config import MovementConfig

SRC = Path(__file__).resolve().parents[1] / "src" / "nflbdb"
CHECKED_METHODS = {"with_columns", "select", "agg", "filter", "group_by", "sort"}
LISTISH = (ast.List, ast.ListComp, ast.GeneratorExp, ast.Tuple)


def _violations(path: Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in CHECKED_METHODS and len(node.args) > 1
                and any(isinstance(a, LISTISH) for a in node.args)):
            out.append(f"{path.relative_to(SRC)}:{node.lineno} .{node.func.attr}(list, …)")
    return out


@pytest.mark.parametrize("path", sorted([*SRC.glob("features/*.py"), *SRC.glob("research/*.py"),
                                         *SRC.glob("research/m21/*.py"), *SRC.glob("research/studies/*.py"),
                                         *SRC.glob("research/phase04/*.py"), *SRC.glob("research/phase05/*.py"),
                                         *SRC.glob("research/sprint_rescue/*.py"),
                                         *SRC.glob("research/combine_nfl_dl/*.py"),
                                         *SRC.glob("research/advanced_movement/*.py")]),
                         ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_list_mixed_with_positional_expressions(path):
    assert not _violations(path), "unpack lists with * — Polars 2.0 rejects list + positional args"


def test_agility_features_fills_counts_for_attempts_without_cod():
    keys = ["event_id", "nfl_id", "attempt"]
    act = pl.DataFrame({"event_id": ["E", "E", "E", "E"], "nfl_id": [1, 1, 2, 2], "attempt": [1, 1, 1, 1],
                        "yaw_rate": [100.0, -50.0, 10.0, 5.0], "a_lat": [2.0, 1.0, 0.1, 0.2]})
    cod = pl.DataFrame({"event_id": ["E", "E"], "nfl_id": [1, 1], "attempt": [1, 1],
                        "cod_angle_deg": [170.0, 90.0], "cod_type": ["reversal", "turn"],
                        "approach_speed": [5.0, 6.0], "approach_decel": [4.0, 2.0], "exit_accel": [4.5, 3.0],
                        "cod_time_s": [1.2, None], "speed_retention": [0.0, 0.7], "turn_radius_yds": [None, 3.0]})
    out = agility_features(act, cod, keys, MovementConfig(), 0.1).sort("nfl_id")
    assert out.height == 2
    a, b = out.row(0, named=True), out.row(1, named=True)
    assert (a["n_cod"], a["n_cod_sharp"], a["n_cod_reversal"]) == (2, 1, 1)
    assert a["cod_angle_total_deg"] == pytest.approx(260.0)
    assert a["turn_speed_retention_median"] == pytest.approx(0.7)
    assert (b["n_cod"], b["n_cod_sharp"], b["n_cod_reversal"], b["cod_angle_total_deg"]) == (0, 0, 0, 0.0)
    assert b["cod_time_median_s"] is None                       # medians stay null, only counts are filled
    assert out.schema["n_cod"] == pl.Int64 and out.schema["cod_angle_total_deg"] == pl.Float64


def test_agility_features_with_empty_cod_table():
    keys = ["event_id", "nfl_id", "attempt"]
    act = pl.DataFrame({"event_id": ["E"], "nfl_id": [1], "attempt": [1], "yaw_rate": [10.0], "a_lat": [0.5]})
    cod = pl.DataFrame(schema={"event_id": pl.String, "nfl_id": pl.Int64, "attempt": pl.Int64,
                               "cod_angle_deg": pl.Float64, "cod_type": pl.String, "approach_speed": pl.Float64,
                               "approach_decel": pl.Float64, "exit_accel": pl.Float64, "cod_time_s": pl.Float64,
                               "speed_retention": pl.Float64, "turn_radius_yds": pl.Float64})
    r = agility_features(act, cod, keys, MovementConfig(), 0.1).row(0, named=True)
    assert (r["n_cod"], r["cod_angle_total_deg"]) == (0, 0.0)
