"""Version-controlled definition of every Superset object in the NFL Analytics Lab.

Edit this file, then re-run superset-init (LAB_IMPORT_MODE=on_change re-imports when the bundle
hash changes). Every number shown comes from a ``lab.*`` table in the serving snapshot; nothing
is hard-coded. Units: yards, yd/s, yd/s², seconds.
"""

from __future__ import annotations

from bundle_lib import (COUNT, Dashboard, Dataset, Filter, M, W, bar, big_number, box, bubble, heatmap, histogram,
                        line, pie, table_agg, table_raw)

DATABASE_NAME = "NFL Lab (DuckDB)"
DATABASE_URI = "duckdb:////srv/nfl-lab/serving/nfl_lab.duckdb"

# =========================================================================================
# Datasets (physical tables in schema `lab`). Columns listed = columns charts/filters use;
# post_import.py syncs the full column list from the snapshot.
# =========================================================================================
_D = lambda key, cols, desc, dttm=None: Dataset(key, key, desc, cols.split(), dttm=dttm)  # noqa: E731

athlete = _D("athlete", """nfl_id:BIGINT display_name position_group cohort:BIGINT has_combine_results:BOOLEAN
    has_combine_tracking:BOOLEAN has_nfl_tracking:BOOLEAN has_player_play:BOOLEAN has_career:BOOLEAN
    is_matched:BOOLEAN forty:DOUBLE combine_weight:DOUBLE combine_height:DOUBLE combine_best_max_s:DOUBLE
    forty_band n_combine_attempts:BIGINT nfl_n_plays_tracked:BIGINT draft_round:DOUBLE""",
             "One row per athlete (career_* columns are OUTCOMES).")
combine_attempt = _D("combine_attempt", """event_id nfl_id:BIGINT attempt:BIGINT cohort:BIGINT drill_type drill_name
    display_name position_group attempt_seq:BIGINT n_attempts_athlete_drill:BIGINT duration_s:DOUBLE
    path_length_yds:DOUBLE max_s:DOUBLE max_s_mph:DOUBLE mean_s:DOUBLE max_a:DOUBLE time_to_max_s_s:DOUBLE
    forty:DOUBLE combine_weight:DOUBLE combine_height:DOUBLE max_s_pct_in_drill:DOUBLE max_s_band
    exclude_kinematics:BOOLEAN""", "One row per Combine attempt.")
combine_athlete_drill = _D("combine_athlete_drill", """nfl_id:BIGINT display_name position_group cohort:BIGINT
    drill_type drill_name n_attempts:BIGINT best_max_s:DOUBLE mean_max_s:DOUBLE sd_max_s:DOUBLE cv_max_s:DOUBLE
    range_max_s:DOUBLE best_max_a:DOUBLE min_time_to_max_s_s:DOUBLE min_duration_s:DOUBLE
    n_flagged_attempts:BIGINT max_s_band""", "One row per athlete × drill.")
drill_participation = _D("drill_participation", """drill_type drill_name position_group cohort:BIGINT
    n_athletes:BIGINT n_attempts:BIGINT median_max_s:DOUBLE""", "Drill × position × cohort participation.")
test_availability = _D("test_availability", """position_group cohort:BIGINT test_name unit n_athletes:BIGINT
    n_with_value:BIGINT share_available:DOUBLE""", "Traditional Combine test availability.")
coverage_matrix = _D("coverage_matrix", """position_group cohort:BIGINT source source_order:BIGINT n_athletes:BIGINT
    n_with_source:BIGINT""", "Source presence per position × cohort.")
linkage_funnel = _D("linkage_funnel", """position_group cohort:BIGINT stage_order:BIGINT stage n_athletes:BIGINT""",
                    "Nested linkage stages.")
season_volume = _D("season_volume", """season:BIGINT season_type n_games:BIGINT n_games_tracked:BIGINT
    n_plays_tracked:BIGINT n_player_plays:BIGINT n_athletes_tracked:BIGINT n_player_frames:BIGINT""",
                   "NFL tracking volume per season.")
tracking_partition = _D("tracking_partition", """domain partition_value season:BIGINT cohort:BIGINT n_rows:BIGINT
    n_bytes:BIGINT""", "Silver frame rows per partition (Parquet footers).")
nfl_game = _D("nfl_game", """game_id:BIGINT season:BIGINT season_type week:BIGINT game_label home_team_abbr
    visitor_team_abbr n_plays_tracked:BIGINT n_athletes_tracked:BIGINT n_player_plays:BIGINT""", "Games.")
nfl_player_play = _D("nfl_player_play", """season:BIGINT game_id:BIGINT play_id:BIGINT nfl_id:BIGINT display_name
    position_group cohort:BIGINT week:BIGINT season_type game_label play_label n_frames:BIGINT
    tracked_duration_s:DOUBLE distance_yds:DOUBLE max_s:DOUBLE mean_s:DOUBLE p95_s:DOUBLE max_a:DOUBLE
    in_player_play:BOOLEAN exclude_kinematics:BOOLEAN""", "One row per tracked athlete × play.")
nfl_athlete_season = _D("nfl_athlete_season", """nfl_id:BIGINT season:BIGINT display_name position_group
    cohort:BIGINT n_games:BIGINT n_plays_tracked:BIGINT total_distance_yds:DOUBLE median_play_max_s:DOUBLE
    p90_play_max_s:DOUBLE max_play_max_s:DOUBLE median_play_max_a:DOUBLE""", "Athlete × season.")
nfl_speed_distribution = _D("nfl_speed_distribution", """season:BIGINT position_group metric bin_start:DOUBLE
    n_player_plays:BIGINT""", "Pre-binned play max speed / accel.")
kpi_snapshot = _D("kpi_snapshot", """snapshot gold_built_at quality_results_saved_at lab_built_at
    n_silver_nfl_tracking_rows:BIGINT seasons_label""", "Unfiltered KPIs at lab build time.")
quality_check = _D("quality_check", """check_seq:BIGINT check_id check_name source_table columns result severity
    severity_label check_category metric value:DOUBLE denominator:DOUBLE threshold details recommended_action mode
    is_sampled:BOOLEAN is_failure:BOOLEAN is_warning:BOOLEAN definition_status remediation_status root_cause
    next_diagnostic results_saved_at""", "Saved data-quality checks.")
quality_run_history = _D("quality_run_history", """run_id step started_at:TIMESTAMP status seconds:DOUBLE
    n_warnings:BIGINT critical_failure:BOOLEAN""", "Logged validation runs.", dttm="started_at")
tracking_interval = _D("tracking_interval", "domain dt_ms_capped:BIGINT n:BIGINT", "Sampling intervals.")
tracking_consistency = _D("tracking_consistency", """domain n_frames:BIGINT corr_s:DOUBLE median_abs_diff_s:DOUBLE
    p95_abs_diff_s:DOUBLE corr_dis:DOUBLE median_s:DOUBLE median_s_derived:DOUBLE sample_pct:DOUBLE""",
                          "Speed consistency.")
quality_affected = _D("quality_affected", """domain n_units:BIGINT n_excluded_kinematics:BIGINT
    n_excluded_context:BIGINT n_athletes_affected:BIGINT n_plays_affected:BIGINT n_frames_affected:BIGINT""",
                      "Phase 02 exclusion flags (empty until built).")
research_hypothesis = _D("research_hypothesis", """hypothesis title positions nfl_outcome traditional_baseline
    n_population:BIGINT n_matched:BIGINT n_matched_with_baseline:BIGINT missing_share:DOUBLE eligible_share:DOUBLE
    feasibility readiness_status readiness_reason statistical_support drill_matching confounders selection_bias
    validation limitations""", "Hypothesis data readiness.")
research_by_position = _D("research_by_position", """hypothesis position_group n_population:BIGINT n_matched:BIGINT
    n_matched_with_baseline:BIGINT match_share:DOUBLE n_outcome:BIGINT""", "Hypothesis × position.")
research_drill = _D("research_drill", "hypothesis drill_name n_athletes:BIGINT n_attempts:BIGINT", "Matched drills.")
availability = _D("availability", "dataset available:BOOLEAN path note used_by", "Optional input availability.")

DATASETS = [athlete, combine_attempt, combine_athlete_drill, drill_participation, test_availability, coverage_matrix,
            linkage_funnel, season_volume, tracking_partition, nfl_game, nfl_player_play, nfl_athlete_season,
            nfl_speed_distribution, kpi_snapshot, quality_check, quality_run_history, tracking_interval,
            tracking_consistency, quality_affected, research_hypothesis, research_by_position, research_drill,
            availability]

B = lambda col: f"SUM(CASE WHEN {col} THEN 1 ELSE 0 END)"  # noqa: E731  boolean count
KPI_H, CH_H, TB_H = 18, 50, 60

# =========================================================================================
# Dashboard 1 — NFL Data Overview  (first milestone)
# =========================================================================================
ov_athletes = big_number("ov_athletes", "Athletes", athlete, M("COUNT(*)", "Athletes"), "all sources")
ov_ctrk = big_number("ov_ctrk", "With Combine tracking", athlete, M(B("has_combine_tracking"), "Combine tracked"))
ov_ntrk = big_number("ov_ntrk", "With NFL tracking", athlete, M(B("has_nfl_tracking"), "NFL tracked"))
ov_matched = big_number("ov_matched", "Matched athletes", athlete, M(B("is_matched"), "Matched"),
                        "Combine + NFL tracking")
ov_drills = big_number("ov_drills", "Combine drills", drill_participation,
                       M("COUNT(DISTINCT drill_name)", "Drills"), "tracked drills")
ov_seasons = big_number("ov_seasons", "NFL seasons", season_volume, M("COUNT(DISTINCT season)", "Seasons"))
ov_games = big_number("ov_games", "Games tracked", season_volume, M("SUM(n_games_tracked)", "Games tracked"),
                      "≥1 dataset athlete tracked")
ov_plays = big_number("ov_plays", "Plays tracked", season_volume, M("SUM(n_plays_tracked)", "Plays"))
ov_pp = big_number("ov_pp", "Player-plays", season_volume, M("SUM(n_player_plays)", "Player-plays"))
ov_frames = big_number("ov_frames", "Athlete tracking frames", season_volume, M("SUM(n_player_frames)", "Frames"),
                       "NFL, football rows excluded")
ov_attempts = big_number("ov_attempts", "Combine attempts", combine_attempt, M("COUNT(*)", "Attempts"))
ov_fresh = table_raw("ov_fresh", "Data freshness", kpi_snapshot,
                     ["gold_built_at", "quality_results_saved_at", "lab_built_at"], row_limit=1, page_length=1,
                     description="UTC build times. Refresh: python -m nflbdb.lab.build, then force-refresh the dashboard.")
ov_by_pos = bar("ov_by_pos", "Athletes by position", athlete, "position_group", [M("COUNT(*)", "Athletes")],
                groupby=["cohort"], stack=True, y_title="athletes")
ov_match_rate = bar("ov_match_rate", "Match rate by position", athlete, "position_group",
                    [M(f"{B('is_matched')} * 1.0 / NULLIF({B('has_combine_tracking')}, 0)",
                       "NFL-tracked share of Combine-tracked")], fmt=".0%", y_title="match rate")
ov_season_vol = bar("ov_season_vol", "Tracking volume by season", season_volume, "season",
                    [M("SUM(n_player_frames)", "Athlete frames")], groupby=["season_type"], stack=True,
                    y_title="frames")
ov_coverage = heatmap("ov_coverage", "Dataset coverage", coverage_matrix, "source", "position_group",
                      M("SUM(n_with_source) * 1.0 / NULLIF(SUM(n_athletes), 0)", "Coverage"))
ov_drill_part = bar("ov_drill_part", "Combine drill participation", drill_participation, "drill_name",
                    [M("SUM(n_athletes)", "Athletes")], groupby=["position_group"], stack=True, horizontal=True,
                    sort_by="Athletes", sort_asc=False, row_limit=5000)
ov_tests = heatmap("ov_tests", "Traditional Combine test availability", test_availability, "test_name",
                   "position_group", M("SUM(n_with_value) * 1.0 / NULLIF(SUM(n_athletes), 0)", "Available"))
ov_funnel = bar("ov_funnel", "Athlete linkage", linkage_funnel, "stage", [M("SUM(n_athletes)", "Athletes")],
                y_title="athletes", description="Exact nfl_id linkage; stages are nested.")
ov_obs_cohort = bar("ov_obs_cohort", "Tracked player-plays by cohort", nfl_athlete_season, "cohort",
                    [M("SUM(n_plays_tracked)", "Player-plays")], groupby=["season"], stack=True,
                    y_title="player-plays", description="Cohort −1 = unknown Combine year.")
ov_avail = table_raw("ov_avail", "Optional inputs", availability, ["dataset", "available", "note", "used_by"],
                     row_limit=50, page_length=10)

overview = Dashboard(
    "overview", "NFL Data Overview", "nfl-lab-overview",
    rows=[
        [("chart", ov_athletes, 2, KPI_H), ("chart", ov_ctrk, 2, KPI_H), ("chart", ov_ntrk, 2, KPI_H),
         ("chart", ov_matched, 2, KPI_H), ("chart", ov_drills, 2, KPI_H), ("chart", ov_seasons, 2, KPI_H)],
        [("chart", ov_games, 2, KPI_H), ("chart", ov_plays, 2, KPI_H), ("chart", ov_pp, 2, KPI_H),
         ("chart", ov_frames, 2, KPI_H), ("chart", ov_attempts, 2, KPI_H), ("chart", ov_fresh, 2, KPI_H)],
        [("chart", ov_by_pos, 6, CH_H), ("chart", ov_match_rate, 6, CH_H)],
        [("chart", ov_season_vol, 6, CH_H), ("chart", ov_coverage, 6, CH_H)],
        [("chart", ov_drill_part, 6, 80), ("chart", ov_tests, 6, 80)],
        [("chart", ov_funnel, 6, CH_H), ("chart", ov_obs_cohort, 6, CH_H)],
        [("md", "**Notes.** All values come from `lab.*` marts built by `python -m nflbdb.lab.build` from the "
                "gold layer. *Games tracked* counts games with at least one dataset athlete (tracking covers the "
                "dataset's athletes, not all 22 players). Frames exclude football rows. Empty charts → see "
                "*Optional inputs*.", 6, 30), ("chart", ov_avail, 6, 30)],
    ],
    filters=[Filter("Position", athlete, "position_group"), Filter("Combine year", athlete, "cohort"),
             Filter("NFL season", season_volume, "season"), Filter("Drill", drill_participation, "drill_name"),
             Filter("Athlete", athlete, "display_name")],
    description="Executive overview of warehouse coverage.",
)

# =========================================================================================
# Dashboard 2 — Combine Explorer
# =========================================================================================
FORTY = W("drill_name = 'FORTY_YARD_DASH'")
cx_athletes = big_number("cx_athletes", "Athletes", combine_attempt, M("COUNT(DISTINCT nfl_id)", "Athletes"))
cx_attempts = big_number("cx_attempts", "Attempts", combine_attempt, M("COUNT(*)", "Attempts"))
cx_drills = big_number("cx_drills", "Drills", combine_attempt, M("COUNT(DISTINCT drill_name)", "Drills"))
cx_med_speed = big_number("cx_med_speed", "Median attempt max speed", combine_attempt,
                          M("MEDIAN(max_s)", "Median max speed"), "yd/s", fmt=",.2f")
cx_part = bar("cx_part", "Drill participation", drill_participation, "drill_name", [M("SUM(n_attempts)", "Attempts")],
              groupby=["position_group"], stack=True, horizontal=True, sort_by="Attempts", sort_asc=False,
              row_limit=5000)
cx_forty = histogram("cx_forty", "40-yard dash times (official)", athlete, "forty", ["position_group"], bins=25,
                     x_title="40-yd time (s)")
cx_speed = histogram("cx_speed", "Attempt max speed", combine_attempt, "max_s", ["position_group"], bins=30,
                     x_title="max speed (yd/s)", description="Use the Drill filter: speeds are only comparable within a drill.")
cx_accel = histogram("cx_accel", "Attempt max acceleration", combine_attempt, "max_a", ["position_group"], bins=30,
                     x_title="max accel (yd/s²)", description="Vendor acceleration magnitude (unsigned).")
cx_duration = box("cx_duration", "Attempt duration by position", combine_attempt, ["position_group"],
                  ["event_id", "nfl_id", "attempt"], M("MAX(duration_s)", "Duration (s)"), fmt=",.1f")
cx_speed_vs_40 = bubble("cx_speed_vs_40", "40-yd dash: tracked peak speed vs official time", combine_attempt,
                        "display_name", M("MAX(forty)", "Official 40 (s)"), M("MAX(max_s)", "Best tracked max speed (yd/s)"),
                        M("COUNT(*)", "Attempts"), "position_group", "official 40-yd time (s)", "max speed (yd/s)",
                        filters=[FORTY])
cx_size_vs_speed = bubble("cx_size_vs_speed", "40-yd dash: body weight vs peak speed", combine_attempt,
                          "display_name", M("MAX(combine_weight)", "Weight (lb)"),
                          M("MAX(max_s)", "Best tracked max speed (yd/s)"), M("MAX(combine_height)", "Height (in)"),
                          "position_group", "weight (lb)", "max speed (yd/s)", filters=[FORTY],
                          description="Bubble size = height (in).")
cx_attempts_per = bar("cx_attempts_per", "Attempts per athlete × drill", combine_athlete_drill, "n_attempts",
                      [M("COUNT(*)", "Athlete-drills")], groupby=["position_group"], stack=True,
                      x_title="attempts", y_title="athlete × drill pairs")
cx_variability = box("cx_variability", "Within-athlete variability of max speed (CV)", combine_athlete_drill,
                     ["position_group"], ["nfl_id", "drill_name"], M("MAX(cv_max_s)", "CV of max speed"), fmt=".1%",
                     filters=[W("n_attempts >= 2")], description="Coefficient of variation across ≥ 2 attempts.")
cx_by_pos = table_agg("cx_by_pos", "Movement metrics by position × drill", combine_attempt,
                      ["position_group", "drill_name"],
                      [M("COUNT(*)", "Attempts"), M("MEDIAN(max_s)", "Median max speed (yd/s)"),
                       M("MEDIAN(max_a)", "Median max accel (yd/s²)"), M("MEDIAN(time_to_max_s_s)", "Median time to max speed (s)"),
                       M("MEDIAN(duration_s)", "Median duration (s)")])
cx_compare = table_raw("cx_compare", "Athlete comparison (select athletes in the filter)", combine_athlete_drill,
                       ["display_name", "position_group", "drill_name", "n_attempts", "best_max_s", "mean_max_s",
                        "sd_max_s", "cv_max_s", "best_max_a", "min_time_to_max_s_s", "n_flagged_attempts"],
                       order_by="best_max_s", row_limit=2000)

combine_explorer = Dashboard(
    "combine", "Combine Explorer", "nfl-lab-combine",
    rows=[
        [("chart", cx_athletes, 3, KPI_H), ("chart", cx_attempts, 3, KPI_H), ("chart", cx_drills, 3, KPI_H),
         ("chart", cx_med_speed, 3, KPI_H)],
        [("chart", cx_part, 6, 80), ("chart", cx_forty, 6, 80)],
        [("chart", cx_speed, 6, CH_H), ("chart", cx_accel, 6, CH_H)],
        [("chart", cx_speed_vs_40, 6, CH_H), ("chart", cx_size_vs_speed, 6, CH_H)],
        [("chart", cx_duration, 4, CH_H), ("chart", cx_attempts_per, 4, CH_H), ("chart", cx_variability, 4, CH_H)],
        [("chart", cx_by_pos, 12, TB_H)],
        [("chart", cx_compare, 12, TB_H)],
        [("md", "**Reading this dashboard.** Speeds/accelerations are the vendor's per-frame values (yd/s, yd/s²; "
                "1 yd/s ≈ 2.05 mph). Compare speeds only within one drill. *Performance band* = percentile of the "
                "attempt's (or athlete's best) max speed within the drill. Attempts flagged by Phase 02 quality "
                "checks are counted in `n_flagged_attempts` but not removed. Trajectories, speed-time and "
                "acceleration-time curves: **Streamlit trajectory explorer** (port 8501).", 12, 22)],
    ],
    filters=[Filter("Position", combine_attempt, "position_group"), Filter("Athlete", combine_attempt, "display_name"),
             Filter("Drill", combine_attempt, "drill_name"), Filter("Combine year", combine_attempt, "cohort"),
             Filter("Performance band (within drill)", combine_attempt, "max_s_band")],
)

# =========================================================================================
# Dashboard 3 — NFL Game Tracking Explorer
# =========================================================================================
SPEED = W("metric = 'max_s'")
ACCEL = W("metric = 'max_a'")
nx_pp = big_number("nx_pp", "Player-plays", nfl_player_play, M("COUNT(*)", "Player-plays"))
nx_plays = big_number("nx_plays", "Plays", nfl_player_play, M("COUNT(DISTINCT play_label)", "Plays"))
nx_athletes = big_number("nx_athletes", "Athletes", nfl_player_play, M("COUNT(DISTINCT nfl_id)", "Athletes"))
nx_med_speed = big_number("nx_med_speed", "Median play max speed", nfl_player_play, M("MEDIAN(max_s)", "Median"),
                          "yd/s", fmt=",.2f")
nx_speed_hist = bar("nx_speed_hist", "Play max speed distribution", nfl_speed_distribution, "bin_start",
                    [M("SUM(n_player_plays)", "Player-plays")], groupby=["position_group"], stack=True,
                    filters=[SPEED], x_title="max speed bin start (yd/s, width 0.25)", row_limit=10000)
nx_accel_hist = bar("nx_accel_hist", "Play max acceleration distribution", nfl_speed_distribution, "bin_start",
                    [M("SUM(n_player_plays)", "Player-plays")], groupby=["position_group"], stack=True,
                    filters=[ACCEL], x_title="max accel bin start (yd/s², width 0.25)", row_limit=10000)
nx_p90_box = box("nx_p90_box", "Season p90 play max speed by position", nfl_athlete_season, ["position_group"],
                 ["nfl_id", "season"], M("MAX(p90_play_max_s)", "p90 max speed (yd/s)"), fmt=",.2f")
nx_profile = bubble("nx_profile", "Exposure vs speed profile (athlete × season)", nfl_athlete_season, "display_name",
                    M("SUM(n_plays_tracked)", "Tracked plays"), M("MAX(p90_play_max_s)", "p90 play max speed (yd/s)"),
                    M("SUM(total_distance_yds)", "Distance (yd)"), "position_group", "tracked plays",
                    "p90 play max speed (yd/s)", description="Bubble size = total distance (yd).")
nx_exposure = bar("nx_exposure", "Tracked player-plays by season", nfl_athlete_season, "season",
                  [M("SUM(n_plays_tracked)", "Player-plays")], groupby=["position_group"], stack=True)
nx_context = bar("nx_context", "Player-plays without play context", nfl_player_play, "season",
                 [M("AVG(CASE WHEN in_player_play THEN 0.0 ELSE 1.0 END)", "Share missing from player_play")],
                 groupby=["position_group"], fmt=".2%",
                 description="Tracked player-plays with no player_play row (no role/context) — data-quality signal.")
nx_athlete_seasons = bar("nx_athlete_seasons", "Selected athletes: in-game speed by season", nfl_athlete_season,
                         "season", [M("MAX(median_play_max_s)", "Median play max speed"),
                                    M("MAX(p90_play_max_s)", "p90 play max speed")],
                         groupby=["display_name"], fmt=",.2f", row_limit=500,
                         description="Pick athletes with the Athlete filter.")
nx_games = table_raw("nx_games", "Games", nfl_game, ["game_label", "season_type", "n_plays_tracked",
                                                      "n_athletes_tracked", "n_player_plays"],
                     order_by="n_player_plays", row_limit=1100, page_length=15)
nx_plays_tbl = table_raw("nx_plays_tbl", "Player-play detail (top 500 by max speed)", nfl_player_play,
                         ["game_label", "play_label", "display_name", "position_group", "n_frames", "tracked_duration_s",
                          "distance_yds", "max_s", "p95_s", "max_a", "exclude_kinematics"],
                         order_by="max_s", row_limit=500, page_length=20,
                         description="Bounded at 500 rows. Narrow with Season / Game / Play / Athlete filters.")

nfl_explorer = Dashboard(
    "nfl", "NFL Game Tracking Explorer", "nfl-lab-nfl",
    rows=[
        [("chart", nx_pp, 3, KPI_H), ("chart", nx_plays, 3, KPI_H), ("chart", nx_athletes, 3, KPI_H),
         ("chart", nx_med_speed, 3, KPI_H)],
        [("chart", nx_speed_hist, 6, CH_H), ("chart", nx_accel_hist, 6, CH_H)],
        [("chart", nx_p90_box, 6, CH_H), ("chart", nx_profile, 6, CH_H)],
        [("chart", nx_exposure, 4, CH_H), ("chart", nx_context, 4, CH_H), ("chart", nx_athlete_seasons, 4, CH_H)],
        [("chart", nx_games, 5, TB_H), ("chart", nx_plays_tbl, 7, TB_H)],
        [("md", "**Scope.** Metrics summarise each athlete's whole tracked window (pre-snap included); "
                "event-windowed features belong to Phase 02. Tracking covers dataset athletes only. Speed histograms "
                "are pre-binned in `lab.nfl_speed_distribution` (no 315k-row pulls). Frame-by-frame playback: "
                "**Streamlit trajectory explorer**.", 12, 18)],
    ],
    filters=[Filter("NFL season", nfl_player_play, "season"), Filter("Position", nfl_player_play, "position_group"),
             Filter("Game", nfl_player_play, "game_label", parents=("NFL season",)),
             Filter("Play", nfl_player_play, "play_label", parents=("Game",)),
             Filter("Athlete", nfl_player_play, "display_name")],
)

# =========================================================================================
# Dashboard 5 — Data Quality Monitor
# =========================================================================================
dq_total = big_number("dq_total", "Checks", quality_check, M("COUNT(*)", "Checks"))
dq_pass = big_number("dq_pass", "Passed", quality_check, M("SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END)", "Passed"))
dq_fail = big_number("dq_fail", "Failed", quality_check, M(B("is_failure"), "Failed"))
dq_warn = big_number("dq_warn", "Warnings", quality_check, M(B("is_warning"), "Warnings"))
dq_crit = big_number("dq_crit", "Critical failures", quality_check,
                     M("SUM(CASE WHEN is_failure AND severity = 'CRITICAL' THEN 1 ELSE 0 END)", "Critical"),
                     "see definition_status")
dq_high = big_number("dq_high", "High-severity failures", quality_check,
                     M("SUM(CASE WHEN is_failure AND severity = 'HIGH' THEN 1 ELSE 0 END)", "High"))
dq_ath = big_number("dq_ath", "Affected athletes", quality_affected, M("SUM(n_athletes_affected)", "Athletes"),
                    "Phase 02 flags")
dq_plays = big_number("dq_plays", "Affected plays", quality_affected, M("SUM(n_plays_affected)", "Plays"),
                      "Phase 02 flags")
dq_frames = big_number("dq_frames", "Affected frames", quality_affected, M("SUM(n_frames_affected)", "Frames"),
                       "Phase 02 flags")
dq_sev = bar("dq_sev", "Checks by severity", quality_check, "severity_label", [COUNT], groupby=["result"], stack=True)
dq_status = pie("dq_status", "Checks by status", quality_check, ["result"], COUNT)
dq_table = bar("dq_table", "Failures and warnings by source table", quality_check, "source_table",
               [M(f"{B('is_failure')} + {B('is_warning')}", "Fail + warn")], groupby=["severity_label"], stack=True)
dq_category = bar("dq_category", "Checks by category", quality_check, "check_category", [COUNT], groupby=["result"],
                  stack=True, horizontal=True, description="Tracking discontinuities, missing identifiers, completeness…")
dq_interval = bar("dq_interval", "Sampling interval distribution", tracking_interval, "dt_ms_capped",
                  [M("SUM(n)", "Frame pairs")], groupby=["domain"], x_title="Δt (ms, capped)",
                  description="Saved tracking-QC sample.")
dq_consistency = table_raw("dq_consistency", "Speed consistency (vendor vs derived)", tracking_consistency,
                           ["domain", "n_frames", "corr_s", "median_abs_diff_s", "p95_abs_diff_s", "median_s",
                            "median_s_derived", "sample_pct"], row_limit=10, page_length=5)
dq_history = line("dq_history", "Validation runs: warnings recorded", quality_run_history, "started_at",
                  [M("MAX(n_warnings)", "Warnings")], groupby=["step"], time_grain="PT1M",
                  description="Real runs from pipeline_run_log; one point per run.")
dq_runs = table_raw("dq_runs", "Validation run log", quality_run_history,
                    ["started_at", "step", "status", "n_warnings", "critical_failure", "seconds", "run_id"],
                    order_by="started_at", row_limit=500, page_length=10)
dq_detail = table_raw("dq_detail", "Check details (download via ⋯ → Download → CSV)", quality_check,
                      ["severity", "result", "check_category", "source_table", "check_name", "columns", "value",
                       "denominator", "mode", "details", "recommended_action", "definition_status",
                       "remediation_status", "root_cause", "next_diagnostic"],
                      order_by="check_seq", desc=False, row_limit=1000, page_length=20)

quality_monitor = Dashboard(
    "quality", "Data Quality Monitor", "nfl-lab-quality",
    rows=[
        [("chart", dq_total, 2, KPI_H), ("chart", dq_pass, 2, KPI_H), ("chart", dq_fail, 2, KPI_H),
         ("chart", dq_warn, 2, KPI_H), ("chart", dq_crit, 2, KPI_H), ("chart", dq_high, 2, KPI_H)],
        [("chart", dq_ath, 4, KPI_H), ("chart", dq_plays, 4, KPI_H), ("chart", dq_frames, 4, KPI_H)],
        [("chart", dq_sev, 4, CH_H), ("chart", dq_status, 4, CH_H), ("chart", dq_table, 4, CH_H)],
        [("chart", dq_category, 6, CH_H), ("chart", dq_interval, 6, CH_H)],
        [("chart", dq_consistency, 6, 34), ("md",
          "**Caveats.** Results are the latest *saved* Phase 01 run (`results_saved_at`). Sampled checks find "
          "issues but cannot prove absence. A CRITICAL Combine key-uniqueness failure in the saved file came from "
          "a stale-kernel run with an outdated grain definition (docs/SESSION_STATE.md); once Phase 02 remediation "
          "runs, `definition_status` shows it. Affected athletes/plays/frames stay empty until "
          "`python -m nflbdb.research.quality` has built the flag datasets.", 6, 34)],
        [("chart", dq_history, 6, CH_H), ("chart", dq_runs, 6, CH_H)],
        [("chart", dq_detail, 12, 80)],
    ],
    filters=[Filter("Severity", quality_check, "severity"), Filter("Result", quality_check, "result"),
             Filter("Source table", quality_check, "source_table"), Filter("Category", quality_check, "check_category")],
)

# =========================================================================================
# Dashboard 6 — Research Feasibility
# =========================================================================================
rf_table = table_raw("rf_table", "Hypothesis readiness (data only — no statistical claims)", research_hypothesis,
                     ["hypothesis", "title", "positions", "n_population", "n_matched", "n_matched_with_baseline",
                      "missing_share", "feasibility", "readiness_status", "readiness_reason", "statistical_support",
                      "drill_matching"], order_by="hypothesis", desc=False, row_limit=50, page_length=10)
rf_counts = bar("rf_counts", "Candidate vs eligible athletes", research_hypothesis, "hypothesis",
                [M("SUM(n_population)", "Candidates"), M("SUM(n_matched)", "Matched"),
                 M("SUM(n_matched_with_baseline)", "Eligible (matched + baseline)")], y_title="athletes")
rf_by_pos = bar("rf_by_pos", "Eligible athletes by position", research_by_position, "hypothesis",
                [M("SUM(n_matched_with_baseline)", "Eligible")], groupby=["position_group"], stack=True)
rf_heat = heatmap("rf_heat", "Match share: hypothesis × position", research_by_position, "hypothesis",
                  "position_group", M("SUM(n_matched) * 1.0 / NULLIF(SUM(n_population), 0)", "Match share"))
rf_drills = table_raw("rf_drills", "Drills matched per hypothesis", research_drill,
                      ["hypothesis", "drill_name", "n_athletes", "n_attempts"], order_by="n_athletes",
                      row_limit=500, page_length=15,
                      description="Phase 01 regex matches; H02/H03 include unintended drills — explicit lists pending.")
rf_outcomes = bar("rf_outcomes", "Outcome availability by position", athlete, "position_group",
                  [M(f"{B('has_career')} * 1.0 / COUNT(*)", "Career outcomes"),
                   M(f"{B('has_nfl_tracking')} * 1.0 / COUNT(*)", "NFL tracking"),
                   M(f"{B('has_player_play')} * 1.0 / COUNT(*)", "Play context")], fmt=".0%")
rf_constraints = table_raw("rf_constraints", "Research constraints", research_hypothesis,
                           ["hypothesis", "nfl_outcome", "traditional_baseline", "confounders", "selection_bias",
                            "validation", "limitations"], order_by="hypothesis", desc=False, row_limit=50, page_length=5)

research = Dashboard(
    "research", "Research Feasibility", "nfl-lab-research",
    rows=[
        [("md", "**Readiness ≠ evidence.** `readiness_status` is a rule on *counts* (≥ 30 eligible athletes, explicit "
                "drill definition, Phase 01 feasibility rating). It says nothing about whether a hypothesis is "
                "supported; `statistical_support` stays *not assessed* until the Phase 02 protocol runs.", 12, 16)],
        [("chart", rf_table, 12, 45)],
        [("chart", rf_counts, 6, CH_H), ("chart", rf_by_pos, 6, CH_H)],
        [("chart", rf_heat, 6, CH_H), ("chart", rf_outcomes, 6, CH_H)],
        [("chart", rf_drills, 5, TB_H), ("chart", rf_constraints, 7, TB_H)],
    ],
    filters=[Filter("Hypothesis", research_hypothesis, "hypothesis"),
             Filter("Position", research_by_position, "position_group"),
             Filter("Drill", research_drill, "drill_name"), Filter("NFL outcome", research_hypothesis, "nfl_outcome"),
             Filter("Combine year", athlete, "cohort")],
)

DASHBOARDS = [overview, combine_explorer, nfl_explorer, quality_monitor, research]
