"""Static tests for Phase 05 · Stage P0 (synthetic data only — no warehouse access).

Authored 2026-10-11, NOT executed by Claude. Run::

    python -m pytest tests/test_phase05_p0_audit.py tests/test_polars2_compat.py -q
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from nflbdb.research.contracts import TableContract, conform
from nflbdb.research.phase05 import audit_figures as af
from nflbdb.research.phase05 import decision_audit as da
from nflbdb.research.phase05.config import (DecisionAuditConfig, Phase05Config, StreamFeasibilityConfig,
                                            threshold_table)
from nflbdb.research.studies.store import StudyStore

FIELDS = ("down", "route_ran", "coverage_assignment", "target", "separation_at_pass_forward", "not_in_warehouse")
ROLES = {"state": ("down",), "action": ("route_ran", "coverage_assignment"),
         "outcome": ("target", "separation_at_pass_forward", "not_in_warehouse")}
CFG = DecisionAuditConfig(min_obs_per_athlete=2, exposure_thresholds=(1, 2), min_level_count=3, max_players_hist=3)


def _rows(contract: TableContract, rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame([{**dict.fromkeys(contract.columns), **r} for r in rows], schema=contract.columns)


# ============================================================================== synthetic warehouse
# athletes 1, 2 = WR (team AAA, offense) · 3, 4 = DB · 5 = DL (team BBB) · 6 = tracked, but in no other table
TRACKED = [(1, 1, 1), (1, 1, 3),             # both sides
           (1, 2, 1), (1, 2, 2),             # offense only
           (1, 3, 3), (1, 3, 5), (1, 3, 6),  # defense + an athlete of unknown side
           (2, 1, 1), (2, 1, 4)]             # preseason game
TEAM = {1: "AAA", 2: "AAA", 3: "BBB", 4: "BBB", 5: "BBB"}
POSITION = {1: "WR", 2: "WR", 3: "DB", 4: "DB", 5: "DL"}


@pytest.fixture()
def tables() -> dict[str, pl.DataFrame]:
    fact = pl.DataFrame([{"game_id": g, "play_id": p, "nfl_id": a, "season": 2023 if g == 1 else 2024, "n_frames": 50}
                         for g, p, a in TRACKED])
    pp = pl.DataFrame(
        [{"game_id": g, "play_id": p, "nfl_id": a, "team_abbr": TEAM[a],
          "route_ran": "GO" if a == 1 else None, "coverage_assignment": "MAN" if a in (3, 4) else None,
          "target": a == 1 and p == 1, "separation_at_pass_forward": 2.5 if a == 1 else None}
         for g, p, a in TRACKED if a != 6],
        schema_overrides={"route_ran": pl.String, "coverage_assignment": pl.String,
                          "separation_at_pass_forward": pl.Float64})
    plays = pl.DataFrame([{"game_id": g, "play_id": p, "possession_team": "AAA", "down": 1}
                          for g, p in sorted({(g, p) for g, p, _ in TRACKED})])
    games = pl.DataFrame({"game_id": [1, 2], "season_type": ["REG", "PRE"]})
    athletes = pl.DataFrame({"nfl_id": list(POSITION), "position_group": list(POSITION.values()),
                             "cohort": [2023] * len(POSITION)})
    return {"fact": fact, "player_play": pp, "plays": plays, "games": games, "athletes": athletes}


@pytest.fixture()
def base(tables) -> tuple[pl.DataFrame, dict[str, str]]:
    return da.assemble_base(tables["fact"], tables["player_play"], tables["plays"], tables["games"],
                            tables["athletes"], FIELDS)


@pytest.fixture()
def pop(base) -> pl.DataFrame:
    return base[0].filter(pl.col("season_type") == "REG")


# ============================================================================== base frame
def test_assemble_base_sides_sources_and_orphans(base):
    b, source = base
    assert b.height == len(TRACKED)
    assert source == {"down": "plays", "route_ran": "player_play", "coverage_assignment": "player_play",
                      "target": "player_play", "separation_at_pass_forward": "player_play",
                      "not_in_warehouse": "absent"}
    assert b["not_in_warehouse"].null_count() == b.height
    side = dict(b.select("nfl_id", "side").unique().iter_rows())
    assert side == {1: "offense", 2: "offense", 3: "defense", 4: "defense", 5: "defense", 6: "unknown"}
    orphan = b.filter(pl.col("nfl_id") == 6).row(0, named=True)
    assert orphan["in_player_play"] is False and orphan["position_group"] == "UNKNOWN"
    assert set(b["season_type"].unique().to_list()) == {"REG", "PRE"}
    assert "team_abbr" not in b.columns and "possession_team" not in b.columns


def test_entity_census_merges_frames_and_population(base):
    frames = pl.DataFrame({"season": [2023, 2024], "season_type": ["REG", "PRE"], "n_rows": [400, 100],
                           "n_football_rows": [40, 0], "n_games": [1, 1], "n_plays": [3, 1],
                           "n_plays_with_football": [1, 0]})
    c = conform(da.entity_census(frames, base[0], ("REG",)), da.CENSUS)
    reg = c.filter(pl.col("season_type") == "REG").row(0, named=True)
    assert reg["in_population"] and reg["n_player_rows"] == 360 and reg["n_tracked_player_plays"] == 7
    assert reg["n_athletes"] == 5 and reg["n_in_player_play"] == 6
    assert reg["share_plays_with_football"] == pytest.approx(1 / 3)
    assert not c.filter(pl.col("season_type") == "PRE")["in_population"].item()


# ============================================================================== census tables
def test_players_per_play_caps_the_last_bin(pop):
    d = conform(da.players_per_play(pop, CFG), da.PER_PLAY)
    tot = {r["n_tracked"]: r for r in d.filter(pl.col("season") == da.ALL).iter_rows(named=True)}
    assert tot[2]["n_plays"] == 2 and not tot[2]["is_capped"]
    assert tot[3]["n_plays"] == 1 and tot[3]["is_capped"]
    assert sum(r["share_plays"] for r in tot.values()) == pytest.approx(1.0)
    assert set(d["season"].unique().to_list()) == {"2023", da.ALL}


def test_side_coverage_counts_each_play_once(pop):
    s = conform(da.side_coverage(pop), da.SIDE).filter(pl.col("season") == da.ALL).row(0, named=True)
    assert (s["n_plays"], s["n_both_sides"], s["n_offense_only"], s["n_defense_only"], s["n_unknown_side"]) == (3, 1, 1, 1, 0)
    assert s["share_both_sides"] == pytest.approx(1 / 3) and s["ci_lower"] < 1 / 3 < s["ci_upper"]
    assert s["median_tracked"] == 2.0


def test_side_coverage_empty_population(pop):
    assert da.side_coverage(pop.head(0)).height == 0 and da.opponent_pairs(pop.head(0), CFG).height == 0


def test_opponent_pairs_need_both_sides_on_the_same_play(pop):
    p = conform(da.opponent_pairs(pop, CFG), da.PAIRS)
    assert p.height == 1                                  # play 2 is offense only, play 3 defense only
    r = p.row(0, named=True)
    assert (r["off_position"], r["def_position"], r["n_plays"], r["n_distinct_pairs"]) == ("WR", "DB", 1, 1)
    assert r["n_pairs_min_obs"] == 0 and r["max_plays_per_pair"] == 1


def test_field_coverage_by_position_and_athlete(pop, base):
    c = conform(da.field_coverage(pop, ROLES, base[1], CFG), da.COVERAGE)
    get = lambda f, pos: c.filter((pl.col("field") == f) & (pl.col("position_group") == pos)).row(0, named=True)  # noqa: E731
    wr = get("route_ran", "WR")
    assert (wr["n_player_plays"], wr["n_non_null"], wr["n_athletes"], wr["n_athletes_any"], wr["n_athletes_min_obs"]) == (3, 2, 2, 1, 1)
    assert get("route_ran", "DB")["n_non_null"] == 0
    assert get("down", da.ALL)["source_table"] == "plays" and get("down", da.ALL)["share_non_null"] == 1.0
    assert get("target", "WR")["n_true"] == 1 and get("route_ran", "WR")["n_true"] is None
    absent = get("not_in_warehouse", da.ALL)
    assert absent["source_table"] == "absent" and absent["n_non_null"] == 0
    assert get("target", "UNKNOWN")["share_non_null"] == 0.0     # no player_play row → no vendor value


def test_exposure_counts_population_plays_only(pop):
    e = conform(da.athlete_exposure(pop), da.EXPOSURE_DS)
    one = e.filter(pl.col("nfl_id") == 1).row(0, named=True)
    assert one["n_tracked_plays"] == 2 and one["n_offense_plays"] == 2 and one["n_seasons"] == 1   # PRE game excluded
    assert 4 not in e["nfl_id"].to_list()                                                         # preseason only
    s = conform(da.exposure_summary(e, CFG), da.EXPOSURE)
    tot = {r["threshold"]: r for r in s.filter(pl.col("position_group") == da.ALL).iter_rows(named=True)}
    assert tot[1]["n_athletes_ge"] == 5 and tot[2]["n_athletes_ge"] == 2 and tot[2]["share_ge"] == pytest.approx(0.4)


def test_action_overlap_requires_repeated_levels():
    rows = ([{"nfl_id": 1, "route_ran": "GO"}] * 3 + [{"nfl_id": 1, "route_ran": "SLANT"}] * 3
            + [{"nfl_id": 2, "route_ran": "GO"}] * 4 + [{"nfl_id": 2, "route_ran": "SLANT"}]   # 1 SLANT < min_level_count
            + [{"nfl_id": 3, "route_ran": None}] * 2)
    pop = pl.DataFrame(rows, schema={"nfl_id": pl.Int64, "route_ran": pl.String}).with_columns(
        pl.lit("WR").alias("position_group"))
    o = conform(da.action_overlap(pop, ("route_ran",), CFG), da.OVERLAP)
    r = o.filter(pl.col("position_group") == "WR").row(0, named=True)
    assert (r["n_player_plays"], r["n_labelled"], r["n_levels"], r["n_levels_supported"]) == (13, 11, 2, 2)
    assert r["n_athletes_labelled"] == 2 and r["n_athletes_varying"] == 1
    assert r["top_level_share"] == pytest.approx(7 / 11) and 0 < r["level_entropy_norm"] < 1
    assert r["median_levels_per_athlete"] == 1.5


# ============================================================================== stream verdicts
def _census(n_ball: int) -> pl.DataFrame:
    return _rows(da.CENSUS, [{"season": 2023, "season_type": "REG", "in_population": True, "n_plays": 100,
                              "n_plays_with_football": n_ball}])


def _scenario() -> dict[str, pl.DataFrame]:
    side = _rows(da.SIDE, [{"season": da.ALL, "n_plays": 100, "share_both_sides": 0.5, "median_tracked": 3.0}])
    pairs = _rows(da.PAIRS, [{"off_position": "WR", "def_position": "DB", "n_plays": 600, "n_distinct_pairs": 40},
                             {"off_position": "OL", "def_position": "DL", "n_plays": 900, "n_distinct_pairs": 5}])
    cov = _rows(da.COVERAGE, [
        {"role": "outcome", "field": "player_get_off", "position_group": "DL", "n_athletes_min_obs": 40},
        {"role": "outcome", "field": "time_to_pressure", "position_group": "DL", "n_athletes_min_obs": 5},
        {"role": "outcome", "field": "separation_at_pass_forward", "position_group": "WR", "n_athletes_min_obs": 40},
        {"role": "state", "field": "cushion", "position_group": "DB", "n_athletes_min_obs": 40},
        {"role": "state", "field": "cushion", "position_group": "WR", "n_athletes_min_obs": 0},
        {"role": "state", "field": "cushion", "position_group": da.ALL, "n_athletes_min_obs": 999}])   # never pooled
    exposure = _rows(da.EXPOSURE_DS, [{"nfl_id": i, "position_group": ("WR", "DB", "DL", "OL")[i % 4],
                                       "n_tracked_plays": 150} for i in range(200)])
    overlap = _rows(da.OVERLAP, [
        {"field": "route_ran", "position_group": "WR", "n_player_plays": 1000, "n_labelled": 600, "n_athletes_varying": 45},
        {"field": "blitzing", "position_group": "DB", "n_player_plays": 1000, "n_labelled": 1000, "n_athletes_varying": 3}])
    return {"side": side, "pairs": pairs, "coverage": cov, "exposure": exposure, "overlap": overlap}


def _verdicts(f: pl.DataFrame) -> dict[str, str]:
    assert f.group_by("stream").agg(pl.col("verdict").n_unique())["verdict"].max() == 1
    return dict(f.select("stream", "verdict").unique().iter_rows())


def test_stream_feasibility_without_ball_is_limited():
    f = conform(da.stream_feasibility(_census(0), **_scenario(), cfg=Phase05Config()), da.FEAS)
    assert _verdicts(f) == {"A": "FEASIBLE", "B": "LIMITED", "C": "LIMITED", "D": "LIMITED", "E": "LIMITED"}
    row = lambda s, c: f.filter((pl.col("stream") == s) & (pl.col("criterion") == c)).row(0, named=True)  # noqa: E731
    assert row("C", "action_label_coverage")["observed"] == pytest.approx(0.6)     # route_ran, not blitzing
    assert "route_ran" in row("C", "action_label_coverage")["basis"]
    assert row("D", "co_tracked_plays_best_pair")["observed"] == 600               # the pair passing BOTH cuts
    assert not row("E", "athletes_with_time_to_pressure")["passed"]
    assert row("E", "athletes_with_cushion")["observed"] == 40                     # DB + WR, 'ALL' row not pooled


def test_stream_feasibility_with_ball_unlocks_interaction_streams():
    v = _verdicts(da.stream_feasibility(_census(95), **_scenario(), cfg=Phase05Config()))
    assert v["B"] == "FEASIBLE" and v["C"] == "FEASIBLE" and v["D"] == "LIMITED"


def test_stream_feasibility_thresholds_are_configurable():
    cfg = dataclasses.replace(Phase05Config(), streams=StreamFeasibilityConfig(min_athletes_twin=500))
    assert _verdicts(da.stream_feasibility(_census(0), **_scenario(), cfg=cfg))["A"] == "LIMITED"


def test_stream_feasibility_on_empty_census_is_infeasible():
    empty = {"side": da.SIDE.empty(), "pairs": da.PAIRS.empty(), "coverage": da.COVERAGE.empty(),
             "exposure": da.EXPOSURE_DS.empty(), "overlap": da.OVERLAP.empty()}
    f = conform(da.stream_feasibility(da.CENSUS.empty(), **empty, cfg=Phase05Config()), da.FEAS)
    assert set(_verdicts(f).values()) == {"INFEASIBLE"} and not f["passed"].any()


# ============================================================================== registry / persistence
def test_audit_fields_cover_the_stream_rules():
    roles = da.audit_fields(Phase05Config())
    audited = {f for fs in roles.values() for f in fs}
    s = Phase05Config().streams
    assert {f for f, _ in s.outcome_positions} | {f for f, _ in s.action_positions} <= audited
    assert "cushion" in roles["state"] and "cushion" not in roles["outcome"]


def test_threshold_table_documents_every_field():
    t = threshold_table(Phase05Config())
    n = sum(len(dataclasses.fields(type(getattr(Phase05Config(), s)))) for s in ("audit", "streams", "frame"))
    assert t.height == n and set(t["provenance"].unique().to_list()) <= {"established", "proposed", "execution"}


def test_store_namespace_and_figure_registry(tmp_path: Path):
    cfg = SimpleNamespace(paths=SimpleNamespace(tables=tmp_path / "t", figures=tmp_path / "f", gold=tmp_path / "g"))
    store = StudyStore(cfg, sampled=True, phase="phase_05")     # type: ignore[arg-type]
    assert store.tables_root == tmp_path / "t" / "phase_05" / "_sample"
    assert store.figure_dir("P0").name == "P0_feasibility"
    assert store.path(da.EXPOSURE_DS).parent.name == "decision_audit"
    names = [rec.name for rec, _ in af.FIGURES]
    assert len(names) == len(set(names)) == 5 and all(rec.study == "P0" for rec, _ in af.FIGURES)
    tables = {c.name for c in da.CONTRACTS}
    assert all(part.strip() in tables for rec, _ in af.FIGURES for part in rec.dataset.split(","))
