"""Static tests for Phase 05 · Stage P0b (synthetic data only — no warehouse access).

Authored 2026-10-11, NOT executed by Claude. Run::

    python -m pytest tests/test_phase05_p0b_sample_frame.py -q
"""

from __future__ import annotations

import polars as pl
import pytest

from nflbdb.research.contracts import conform
from nflbdb.research.phase05 import decision_audit as da
from nflbdb.research.phase05 import sample_frame as sf
from nflbdb.research.phase05.config import DecisionAuditConfig, SampleFrameConfig

CFG = SampleFrameConfig(min_plays_grid=(0, 100), outcome_obs_grid=(2, 5))


def _exposure(rows: list[tuple[int, str, int, int]]) -> pl.DataFrame:
    return pl.DataFrame([{**dict.fromkeys(da.EXPOSURE_DS.columns), "nfl_id": i, "position_group": p, "cohort": c,
                          "n_tracked_plays": n, "n_seasons": 1} for i, p, c, n in rows], schema=da.EXPOSURE_DS.columns)


def test_min_detectable_r_planning_values():
    assert sf.min_detectable_r(303) == pytest.approx(0.160, abs=0.002)
    assert sf.min_detectable_r(70) == pytest.approx(0.330, abs=0.002)
    assert sf.min_detectable_r(3) is None and sf.min_detectable_r(None) is None
    assert sf.min_detectable_r(50) > sf.min_detectable_r(200)
    assert sf.min_detectable_r(70, power=0.9) > sf.min_detectable_r(70, power=0.8)


def test_exposure_by_cohort_counts_and_unknown_cohort():
    e = _exposure([(1, "WR", 2023, 500), (2, "WR", 2023, 50), (3, "WR", 2025, 150)]).with_columns(
        pl.when(pl.col("nfl_id") == 3).then(None).otherwise(pl.col("cohort")).alias("cohort"))
    t = conform(sf.exposure_by_cohort(e, CFG), sf.COHORT)
    get = lambda pos, c, thr: t.filter((pl.col("position_group") == pos) & (pl.col("cohort") == c)  # noqa: E731
                                       & (pl.col("min_plays") == thr)).row(0, named=True)
    assert get("WR", 2023, 100)["n_athletes_ge"] == 1 and get("WR", 2023, 0)["n_athletes_ge"] == 2
    assert get("ALL", -1, 100)["n_athletes"] == 1            # null cohort is reported, never dropped


def test_combine_nfl_sample_intersects_drill_exposure_and_official_result():
    athletes = pl.DataFrame({"nfl_id": [1, 2, 3, 4], "position_group": ["WR", "WR", "WR", "DB"]})
    exposure = _exposure([(1, "WR", 2023, 500), (2, "WR", 2023, 50), (4, "DB", 2023, 300)])     # athlete 3: no NFL play
    attempts = pl.DataFrame({"nfl_id": [1, 2, 3, 4, 1], "drill": ["FORTY_YARD_DASH"] * 4 + ["GAUNTLET"],
                             "n_attempts": [2, 1, 2, 1, 1]})
    results = pl.DataFrame({"nfl_id": [1, 2, 3, 4], "forty": [4.4, 4.5, None, 4.5],
                            "ten_yd_split": [1.5, None, None, 1.6]})
    t = conform(sf.combine_nfl_sample(athletes, exposure, attempts, results, CFG), sf.SAMPLE)
    get = lambda pos, drill, thr: t.filter((pl.col("position_group") == pos) & (pl.col("drill") == drill)  # noqa: E731
                                           & (pl.col("min_plays") == thr)).row(0, named=True)
    any_wr = get("WR", "FORTY_YARD_DASH", 0)
    assert (any_wr["n_combine_athletes"], any_wr["n_nfl_athletes"], any_wr["n_both"], any_wr["n_both_repeat"]) == (3, 2, 2, 1)
    assert any_wr["n_both_official"] == 1                    # athlete 2 lacks the 10-yd split
    assert get("WR", "FORTY_YARD_DASH", 100)["n_both"] == 1
    assert get("ALL", "FORTY_YARD_DASH", 100)["n_both"] == 2 and get("ALL", "FORTY_YARD_DASH", 100)["n_both_official"] == 2
    gauntlet = get("WR", "GAUNTLET", 0)
    assert gauntlet["n_both_official"] is None and gauntlet["official_columns"] == ""
    assert gauntlet["min_detectable_r"] is None              # n = 1
    assert not t.filter((pl.col("position_group") == "DB") & (pl.col("drill") == "GAUNTLET")).height


def test_outcome_counts_and_exposure():
    pop = pl.DataFrame({"nfl_id": [1, 1, 1, 2, 2, 3], "position_group": ["DL"] * 5 + ["WR"],
                        "player_get_off": [0.8, 0.9, None, 1.0, None, None]})
    counts = conform(sf.athlete_outcome_counts(pop, ("player_get_off", "not_a_column")), sf.COUNTS_DS)
    assert dict(counts.select("nfl_id", "n_obs").iter_rows()) == {1: 2, 2: 1, 3: 0}
    t = conform(sf.outcome_exposure(counts, CFG), sf.OUTCOME)
    dl = t.filter((pl.col("position_group") == "DL") & (pl.col("min_obs") == 2)).row(0, named=True)
    assert (dl["n_athletes_any"], dl["n_athletes_ge"], dl["n_obs"], dl["median_obs_per_athlete"]) == (2, 1, 3, 1.5)
    assert not t.filter(pl.col("position_group") == "WR").height     # no observation → no row


def test_pairs_by_season_flags_relevant_pairs():
    rows = [(2024, 1, 1, 10, "OL", "offense"), (2024, 1, 1, 20, "DL", "defense"), (2024, 1, 1, 30, "DB", "defense"),
            (2025, 2, 1, 10, "OL", "offense"), (2025, 2, 1, 20, "DL", "defense")]
    pop = pl.DataFrame([{"season": s, "game_id": g, "play_id": p, "nfl_id": a, "position_group": pos, "side": side}
                        for s, g, p, a, pos, side in rows])
    t = conform(sf.pairs_by_season(pop, DecisionAuditConfig(min_obs_per_athlete=2), CFG), sf.PAIRS_SEASON)
    get = lambda season, d: t.filter((pl.col("season") == season) & (pl.col("def_position") == d)).row(0, named=True)  # noqa: E731
    assert get("ALL", "DL")["n_plays"] == 2 and get("ALL", "DL")["n_pairs_min_obs"] == 1 and get("ALL", "DL")["football_relevant"]
    assert get("2024", "DB")["n_plays"] == 1 and not get("2024", "DB")["football_relevant"]
    assert not t.filter((pl.col("season") == "2025") & (pl.col("def_position") == "DB")).height


def test_play_type_census_shares_and_route_labels():
    pop = pl.DataFrame({"position_group": ["WR", "WR", "WR", "OL"], "pass_result": ["C", "C", None, "C"],
                        "route_ran": ["GO", None, None, None]})
    t = conform(sf.play_type_census(pop, "pass_result"), sf.PLAY_TYPE)
    wr = {r["play_type"]: r for r in t.filter(pl.col("position_group") == "WR").iter_rows(named=True)}
    assert wr["C"]["n_player_plays"] == 2 and wr["C"]["share"] == pytest.approx(2 / 3) and wr["C"]["n_route_labelled"] == 1
    assert wr["<null>"]["n_player_plays"] == 1
    assert t.filter(pl.col("position_group") == "ALL")["n_player_plays"].sum() == 4
    assert sf.play_type_census(pop, "missing_field").height == 0


def test_figure_registry():
    names = [rec.name for rec, _ in sf.FIGURES]
    assert len(names) == len(set(names)) and all(rec.study == "P0B" for rec, _ in sf.FIGURES)
    assert {rec.dataset for rec, _ in sf.FIGURES} <= {c.name for c in sf.CONTRACTS}
