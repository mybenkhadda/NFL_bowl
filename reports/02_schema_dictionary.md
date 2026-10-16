# 02 — Schema dictionary

_Step 2: physical schema, semantics and grain hypotheses_

_Generated 2026-10-08T00:33:02Z by `nflbdb.data` — numbers below come from executed code only._
Samples are deterministic byte-offset samples spanning each file (see `sampling.offset_sample`). Null fractions and distinct counts below are **sample statistics**, not full-table values.


## Grain hypotheses (validated in 03_data_quality.md)

| table | grain |
|---|---|
| players | One row per player (nfl_id). |
| combine_results | One row per athlete per Combine (nfl_id, possibly + draft_year). |
| combine_tracking | One row per tracked entity per 100 ms frame within a Combine drill attempt (event_id, nfl_id, attempt, time). Hierarchy: draft_year → athlete → drill → attempt → frame. |
| game_tracking | One row per tracked entity per 100 ms frame within a play (game_id, play_id, nfl_id, time). Rows with null nfl_id are expected to be the football. Hierarchy: season → game → play → player → frame. |
| games | One row per game. |
| player_play | One row per player per play; play-level columns repeated on every player row. |
| player_career_successes | One row per player: career totals up to the data cut-off (outcome table). |


## Identifiers shared across tables (namespace equality NOT assumed)

| column | tables |
|---|---|
| event_id | combine_tracking |
| game_id | game_tracking, games, player_play |
| game_key | games |
| nfl_id | combine_results, combine_tracking, game_tracking, player_career_successes, player_play, players |
| play_id | game_tracking, player_play |


## `players` — 10 columns

| column | bronze_dtype | semantic_role | unit | evidence | sample_null_frac | sample_n_distinct | sample_examples | meaning |
|---|---|---|---|---|---|---|---|---|
| nfl_id | Int64 | identifier |  | confirmed | 0 | 498 | 55900, 55901, 55903, 55904, 55905 | Player identifier; appears in every player-bearing table |
| display_name | String | text |  | confirmed | 0 | 496 | Michael Mayer, Steve Avila, Matthew Bergeron, Jonathan Mingo, Isaiah Foskey | Player name — never used as a join key |
| draft_year | Int64 | categorical | year | inferred | 0 | 3 | 2023, 2024, 2025 | NFL draft cohort year (= Combine year) |
| nfl_position | String | categorical |  | confirmed | 0 | 17 | TE, G, WR, DE, CB | Listed NFL position |
| birth_date | String | timestamp | date | inferred | 0 | 434 | 2001-07-06, 1999-10-16, 2000-02-26, 2001-04-20, 2000-10-30 | Date of birth |
| college_name | String | categorical |  | inferred | 0 | 112 | Notre Dame, Texas Christian, Syracuse, Mississippi, Kansas State | College |
| college_conference | String | categorical |  | inferred | 0 | 21 | Independent, Big Twelve Conference, Atlantic Coast Conference, Southeastern Conference, Big Ten Conference | College conference |
| draft_round | Float64 | continuous | round | inferred | 0.3077 | 7 | 2.0, 3.0, 4.0, 5.0, 6.0 | Draft round (null = undrafted? unresolved) |
| draft_pick_within_round | Float64 | continuous | pick | inferred | 0.3077 | 42 | 4.0, 5.0, 7.0, 8.0, 9.0 | Pick within round |
| draft_overall_pick | Float64 | continuous | pick | inferred | 0.3077 | 217 | 35.0, 36.0, 38.0, 39.0, 40.0 | Overall pick |


## `combine_results` — 18 columns

| column | bronze_dtype | semantic_role | unit | evidence | sample_null_frac | sample_n_distinct | sample_examples | meaning |
|---|---|---|---|---|---|---|---|---|
| draft_year | Int64 | categorical | year | inferred | 0 | 3 | 2023, 2024, 2025 | NFL draft cohort year (= Combine year) |
| nfl_id | Int64 | identifier |  | confirmed | 0 | 498 | 56541, 55914, 56514, 56371, 56581 | Player identifier; appears in every player-bearing table |
| combine_position | String | categorical |  | confirmed | 0 | 5 | WR, DL, DB, TE, OL | Position group at the Combine (e.g. DB, WR, TE) |
| combine_height | Float64 | continuous | inches | inferred | 0 | 95 | 71.875, 75.75, 75.875, 73.75, 75.625 | Height (77.875 observed → fractional inches) |
| combine_weight | Float64 | continuous | pounds | inferred | 0 | 149 | 192.0, 309.0, 320.0, 210.0, 334.0 | Weight |
| hand_size | Float64 | continuous | inches | inferred | 0 | 27 | 9.375, 9.75, 9.0, 9.5, 9.125 | Hand span |
| arm_length | Float64 | continuous | inches | inferred | 0 | 60 | 31.375, 33.875, 32.75, 32.125, 33.75 | Arm length |
| wing_span | Float64 | continuous | inches | inferred | 0 | 113 | 75.875, 81.5, 79.375, 78.5, 82.0 | Wingspan |
| ten_yd_split | Float64 | continuous | seconds | inferred | 0.1952 | 46 | 1.51, 1.79, 1.88, 1.54, 1.82 | 10-yard split of the 40-yard dash |
| forty | Float64 | continuous | seconds | inferred | 0.1975 | 106 | 4.44, 5.08, 5.41, 4.51, 5.25 | 40-yard dash time |
| vertical | Float64 | continuous | inches | inferred | 0.1644 | 41 | 35.5, 29.5, 27.5, 34.5, 29.0 | Vertical jump |
| broad_jump | Float64 | continuous | inches | inferred | 0.1981 | 42 | 123.0, 111.0, 96.0, 124.0, 102.0 | Broad jump |
| three_cone | Float64 | continuous | seconds | inferred | 0.6464 | 93 | 7.34, 8.01, 7.6, 6.9, 6.57 | 3-cone drill time |
| short_shuttle | Float64 | continuous | seconds | inferred | 0.6105 | 87 | 4.65, 5.03, 4.83, 4.25, 4.27 | 20-yard (short) shuttle time |
| bench_reps | Float64 | count | reps | inferred | 0.6516 | 25 | 13.0, 25.0, 19.0, 23.0, 18.0 | 225-lb bench press repetitions |
| ngs_athleticism_score | Float64 | continuous | score 0-100 | inferred | 0 | 48 | 77.0, 76.0, 51.0, 69.0, 71.0 | NGS athleticism score (pre-draft) |
| ngs_college_production_score | Float64 | continuous | score 0-100 | inferred | 0 | 44 | 61.0, 69.0, 55.0, 63.0, 59.0 | NGS college production score |
| ngs_final_score | Float64 | continuous | score 0-100 | inferred | 0 | 42 | 71.0, 72.0, 52.0, 64.0, 79.0 | NGS overall draft score |


## `combine_tracking` — 14 columns

| column | bronze_dtype | semantic_role | unit | evidence | sample_null_frac | sample_n_distinct | sample_examples | meaning |
|---|---|---|---|---|---|---|---|---|
| draft_year | Int64 | categorical | year | inferred | 0 | 3 | 2025, 2024, 2023 | NFL draft cohort year (= Combine year) |
| event_id | String | identifier |  | confirmed | 0 | 562 | 20250302101000_1740947407678_000, 20250302101000_1740948802060_000, 20250302101000_1740948532038_000, 20250302101000_1740949109325_000, 20250302101000_1740948247686_000 | Combine tracking session/event id; observed format '<yyyymmddhhmmss>_<epoch_ms>_<seq>' |
| nfl_id | Int64 | identifier |  | confirmed | 0 | 67 | 58359, 58242, 58296, 58898, 58395 | Player identifier; appears in every player-bearing table |
| entity_type | String | categorical |  | confirmed | 0 | 2 | PLAYER, BALL | Tracked entity type (PLAYER observed) |
| time | String | timestamp | UTC? (unresolved) | inferred | 0 | 37,444 | 2025-03-02T20:30:04.200, 2025-03-02T20:30:04.300, 2025-03-02T20:30:04.400, 2025-03-02T20:30:04.500, 2025-03-02T20:30:04.600 | Frame timestamp, ISO-8601 with milliseconds; 100 ms steps in sample |
| drill_type | String | categorical |  | confirmed | 0 | 8 | SKILL_DRILLS_OL, SHORT_SHUTTLE, FORTY_YARD_DASH, SKILL_DRILLS_DL, THREE_CONE_DRILL | Combine drill family (e.g. SKILL_DRILLS_OL) |
| drill_name | String | categorical |  | confirmed | 0 | 52 | OL_PULL_DRILL_LONG_PULL_RIGHT, PASS_PRO-MIRROR_DRILL, PASS_RUSH_DROP, SCREEN_DRILL, OL_PULL_DRILL_FOLD_BLOCK_RIGHT | Specific drill (e.g. OL_PULL_DRILL_FOLD_BLOCK_RIGHT) |
| attempt | Int64 | count | count | inferred | 0 | 15 | 2, 7, 6, 8, 5 | Attempt / repetition number within a drill |
| x | Float64 | kinematic | yards | inferred | 0 | 18,195 | 46.0806071, 46.45011902, 46.84228007, 47.24698893, 47.66893514 | Position along the long axis of the field |
| y | Float64 | kinematic | yards | inferred | 0 | 17,674 | 23.91540782, 23.9170812, 23.92012278, 23.90771484, 23.87039185 | Position across the field |
| s | Float64 | kinematic | yards/second | inferred | 0 | 19,255 | 3.519774119, 3.742669741, 3.951399485, 4.128909429, 4.306051572 | Speed (vendor-computed, possibly smoothed) |
| a | Float64 | kinematic | yards/second² | inferred | 0 | 38,845 | 2.33790318, 1.990376887, 2.029948307, 2.416385025, 2.894180731 | Acceleration magnitude (vendor-computed) |
| dis | Float64 | kinematic | yards | inferred | 0 | 16,367 | 0.116839211, 0.136539061, 0.15379029, 0.163789336, 0.178044052 | Distance travelled since previous frame |
| dir | Float64 | kinematic | degrees | inferred | 0 | 39,448 | 92.33764302, 90.94813927, 91.19104238, 93.05413968, 96.01662411 | Direction of motion |


## `game_tracking` — 12 columns

| column | bronze_dtype | semantic_role | unit | evidence | sample_null_frac | sample_n_distinct | sample_examples | meaning |
|---|---|---|---|---|---|---|---|---|
| game_id | Int64 | identifier |  | confirmed | 0 | 787 | 2023110507, 2023081953, 2023102205, 2023112606, 2023092400 | Game identifier; format YYYYMMDDNN (date + sequence) observed |
| play_id | Int64 | identifier |  | inferred | 0 | 1,415 | 1359, 641, 3618, 844, 293 | Play identifier, unique within a game (hypothesis) |
| nfl_id | Int64 | identifier |  | confirmed | 0 | 362 | 55929, 55924, 55903, 56045, 55909 | Player identifier; appears in every player-bearing table |
| time | String | timestamp | UTC? (unresolved) | inferred | 0 | 118,661 | 2023-11-05T18:59:24.600, 2023-11-05T18:59:24.700, 2023-11-05T18:59:24.800, 2023-11-05T18:59:24.900, 2023-11-05T18:59:25.000 | Frame timestamp, ISO-8601 with milliseconds; 100 ms steps in sample |
| x | Float64 | kinematic | yards | inferred | 0 | 11,113 | 88.54, 88.83, 89.14, 89.46, 89.8 | Position along the long axis of the field |
| y | Float64 | kinematic | yards | inferred | 0 | 5,556 | 22.62, 22.9, 23.21, 23.54, 23.89 | Position across the field |
| s | Float64 | kinematic | yards/second | inferred | 0 | 983 | 3.71, 3.95, 4.25, 4.45, 4.66 | Speed (vendor-computed, possibly smoothed) |
| a | Float64 | kinematic | yards/second² | inferred | 0 | 735 | 1.61, 1.03, 0.7, 0.32, 0.28 | Acceleration magnitude (vendor-computed) |
| dis | Float64 | kinematic | yards | inferred | 0 | 108 | 0.4, 0.44, 0.46, 0.48, 0.5 | Distance travelled since previous frame |
| o | Float64 | kinematic | degrees | inferred | 0 | 33,267 | 20.14, 13.25, 18.87, 21.65, 28.9 | Body orientation |
| dir | Float64 | kinematic | degrees | inferred | 0 | 34,656 | 45.24, 45.52, 44.75, 44.54, 43.88 | Direction of motion |
| event | String | categorical |  | inferred | 0.9333 | 39 | pass_outcome_caught, tackle, ball_snap, pass_forward, pass_arrived | Tracking event tag (snap, pass_forward, …); sparse |


## `games` — 7 columns

| column | bronze_dtype | semantic_role | unit | evidence | sample_null_frac | sample_n_distinct | sample_examples | meaning |
|---|---|---|---|---|---|---|---|---|
| game_id | Int64 | identifier |  | confirmed | 0 | 992 | 2023091009, 2023091010, 2023091011, 2023091012, 2023091013 | Game identifier; format YYYYMMDDNN (date + sequence) observed |
| game_key | Int64 | identifier |  | unresolved | 0 | 992 | 59183, 59184, 59185, 59186, 59187 | Alternate game key (league internal) |
| season | Int64 | categorical | year | confirmed | 0 | 3 | 2023, 2024, 2025 | NFL season (season 2023 includes Jan-2024 games) |
| season_type | String | categorical |  | confirmed | 0 | 3 | REG, PRE, POST | REG / POST |
| week | Int64 | categorical | week | inferred | 0 | 23 | 1, 2, 3, 4, 5 | Season week |
| home_team_abbr | String | unknown |  | unresolved | 0 | 33 | DEN, LAC, NE, SEA, NYG | No interpretation yet |
| visitor_team_abbr | String | unknown |  | unresolved | 0 | 33 | LV, MIA, PHI, LA, DAL | No interpretation yet |


## `player_play` — 64 columns

| column | bronze_dtype | semantic_role | unit | evidence | sample_null_frac | sample_n_distinct | sample_examples | meaning |
|---|---|---|---|---|---|---|---|---|
| game_id | Int64 | identifier |  | confirmed | 0 | 996 | 2024010712, 2024110312, 2025090701, 2025122112, 2024122204 | Game identifier; format YYYYMMDDNN (date + sequence) observed |
| play_id | Int64 | identifier |  | inferred | 0 | 4,663 | 3188, 588, 232, 2487, 482 | Play identifier, unique within a game (hypothesis) |
| team_abbr | String | categorical |  | confirmed | 0 | 33 | SEA, JAX, CIN, HOU, ARI | Player's team on the play |
| nfl_id | Int64 | identifier |  | confirmed | 0 | 494 | 55973, 56025, 56071, 57163, 57148 | Player identifier; appears in every player-bearing table |
| lined_up_position | String | categorical |  | confirmed | 0.0053 | 16 | G, HIGH_SAFETY, WR, CB, EDGE | Alignment on the play (T, EDGE, OLB, INTERIOR_LINE observed) |
| play_description | String | text |  | confirmed | 0 | 30,109 | (9:17) K.Walker up the middle to SEA 28 for no gain (N.Jones; K.Barnes)., (4:54) (Shotgun) J.Hurts pass short left to G.Calcaterra to PHI 11 for 6 yards (J.Jones)., (11:07) (Shotgun) J.Burrow pass short middle to T.Higgins to CLV 27 for 13 yards (M.Harden)., (10:32) G.Smith pass short right to B.Bowers to LV 40 for 1 yard (J.Pitre)., (6:49) D.Moore left end to ARZ 5 for no gain (J.Thompson). | Play-by-play text |
| quarter | Float64 | categorical |  | inferred | 0 | 5 | 4.0, 1.0, 3.0, 2.0, 5.0 | Quarter (5 = OT, hypothesis) |
| down | Float64 | categorical |  | inferred | 0 | 5 | 2.0, 3.0, 1.0, 4.0, 0.0 | Down |
| yards_to_go | Float64 | continuous | yards | inferred | 0 | 37 | 7.0, 16.0, 4.0, 1.0, 10.0 | Yards to first down |
| possession_team | String | categorical |  | confirmed | 0 | 33 | SEA, PHI, CIN, LV, CAR | Offensive team |
| yardline_side | String | categorical |  | inferred | 0.0135 | 33 | SEA, PHI, CLE, LV, ARI | Side of field of the line of scrimmage |
| yardline_number | Float64 | continuous | yards | inferred | 0 | 50 | 28.0, 5.0, 40.0, 39.0, 48.0 | Yard line number |
| game_clock | String | text | mm:ss | confirmed | 0 | 901 | 09:17, 04:54, 11:07, 10:32, 06:49 | Game clock at snap, 'MM:SS' |
| pre_snap_home_score | Float64 | continuous | points | inferred | 0 | 50 | 20.0, 7.0, 0.0, 13.0, 52.0 | Score |
| pre_snap_visitor_score | Float64 | continuous | points | inferred | 0 | 46 | 13.0, 0.0, 7.0, 3.0, 14.0 | Score |
| play_direction | String | categorical |  | confirmed | 0.0003 | 2 | left, right | Offense direction of travel (left/right) — drives coordinate standardisation |
| pre_snap_home_team_win_probability | Float64 | continuous | probability 0-1 | inferred | 0 | 30,011 | 0.780780103057623, 0.879653273150325, 0.325441708788276, 0.902164051309228, 0.478122401982546 | Model probability: pre snap home team win probability |
| pre_snap_visitor_team_win_probability | Float64 | continuous | probability 0-1 | inferred | 0 | 30,011 | 0.219219896942377, 0.120346726849675, 0.674558291211724, 0.0978359486907721, 0.521877598017454 | Model probability: pre snap visitor team win probability |
| home_team_win_probability_added | Float64 | continuous | probability 0-1 | inferred | 0 | 29,937 | 0.0347498878836632, -0.0129639264196157, -0.0403737016022205, 0.00493529438972473, -0.0238911807537079 | Model probability: home team win probability added |
| visitor_team_win_probility_added | Float64 | continuous | probability 0-1 | inferred | 0 | 29,937 | -0.0347498878836632, 0.0129639264196157, 0.0403737016022205, -0.00493529438972473, 0.0238911807537079 | Model probability: visitor team win probility added |
| expected_points_added | Float64 | continuous | points | inferred | 0 | 29,694 | -0.795803740853444, -0.821842418052256, 1.85092568700202, -0.825222132960334, -1.04052751732524 | Expected points (added) |
| expected_points | Float64 | continuous | points | inferred | 0 | 29,052 | 1.08551296289079, -1.85991199687123, 2.58146094554104, 1.85013501089998, 5.34462345403153 | Expected points (added) |
| offense_formation | String | categorical |  | inferred | 0.0173 | 7 | SINGLEBACK, EMPTY, SHOTGUN, JUMBO, I_FORM | Offensive formation |
| receiver_alignment | String | categorical |  | inferred | 0.0147 | 13 | 3x1, 3x2, 2x2, 1x1, 4x1 | Receiver alignment (e.g. 3x1) |
| pass_result | String | categorical |  | unresolved | 0 | 6 | , C, I, S, R | Pass result code (I = incomplete, R = run? unresolved) |
| pass_length | Float64 | continuous | yards | inferred | 0.4163 | 73 | 3.0, 11.0, -1.0, 16.0, 0.0 | Distance metric: pass length |
| play_nullified_by_penalty | String | flag |  | confirmed | 0 | 2 | N, Y | Y/N penalty nullification |
| penalty_yards | Float64 | continuous | yards | inferred | 0.9326 | 55 | -5.0, -10.0, 16.0, 0.0, -15.0 | Distance metric: penalty yards |
| pre_penalty_yards_gained | Float64 | continuous | yards | inferred | 0 | 110 | 0.0, 6.0, 13.0, 1.0, 4.0 | Distance metric: pre penalty yards gained |
| yards_gained | Float64 | continuous | yards | inferred | 0 | 119 | 0.0, 6.0, 13.0, 1.0, 4.0 | Distance metric: yards gained |
| team_coverage_man_zone | String | categorical |  | inferred | 0.1241 | 2 | ZONE_COVERAGE, MAN_COVERAGE | Defensive coverage family |
| team_coverage_type | String | categorical |  | inferred | 0.1241 | 8 | COVER_2_ZONE, COVER_1_MAN, COVER_0_MAN, COVER_3_ZONE, COVER_6_ZONE | Coverage shell |
| in_motion_at_ball_snap | String | flag |  | inferred | 0.5013 | 2 | FALSE, TRUE | Player-level flag: in motion at ball snap (TRUE/FALSE literals) |
| target | String | flag |  | inferred | 0 | 2 | FALSE, TRUE | Player-level flag: target (TRUE/FALSE literals) |
| rec_yards | Float64 | continuous | yards | inferred | 0.9795 | 58 | 21.0, 59.0, 24.0, 9.0, 16.0 | Distance metric: rec yards |
| rush_yards | Float64 | continuous | yards | inferred | 0.9999 | 4 | 2.0, 7.0, 5.0, 9.0 | Distance metric: rush yards |
| yards_after_catch | Float64 | continuous | yards | inferred | 0.9795 | 38 | 1.0, 42.0, 19.0, 3.0, 12.0 | Distance metric: yards after catch |
| fumble | Float64 | outcome | count | inferred | 0.9999 | 1 | 1.0 | Player-play event count: fumble |
| fumble_lost | Float64 | outcome | count | inferred | 0.9999 | 2 | 0.0, 1.0 | Player-play event count: fumble lost |
| route_ran | String | categorical |  | inferred | 0.8303 | 12 | CROSS, OUT, GO, FLAT, POST | Route run by the player |
| separation_at_pass_forward | Float64 | continuous | yards | inferred | 0.8519 | 4,763 | 1.61616826057434, 0.614654362201691, 8.03596878051758, 1.95931112766266, 1.93744683265686 | Distance metric: separation at pass forward |
| cushion | Float64 | continuous | yards | inferred | 0.6695 | 1,205 | 2.3199999332428, 2.64000010490417, 11.8100004196167, 3.53999996185303, 2.17000007629395 | Distance metric: cushion |
| expected_yards_after_catch | Float64 | continuous | yards | inferred | 0.9716 | 981 | 2.52247786521912, 2.18117642402649, 3.97415542602539, 6.90656709671021, 9.08130168914795 | Distance metric: expected yards after catch |
| dropback_duration | Float64 | continuous | seconds | inferred | 0.8059 | 3,027 | 5.24200010299683, 1.86600005626678, 2.39100003242493, 5.16200017929077, 2.57299995422363 | Timing metric: dropback duration |
| extended_sack_allowed | Float64 | outcome | count | inferred | 0.8059 | 3 | 0.0, 1.0, 0.5 | Player-play event count: extended sack allowed |
| pass_rushers_encountered | Float64 | outcome | count | inferred | 0.8059 | 3 | 1.0, 2.0, 0.0 | Player-play event count: pass rushers encountered |
| peak_pressure_probability_allowed | Float64 | continuous | probability 0-1 | inferred | 0.815 | 1,154 | 0.194166660308838, 0.177499994635582, 0.218333333730698, 1.0, 0.137500002980232 | Model probability: peak pressure probability allowed |
| pressure_allowed | String | flag |  | inferred | 0.8059 | 2 | FALSE, TRUE | Player-level flag: pressure allowed (TRUE/FALSE literals) |
| sack_allowed | Float64 | outcome | count | inferred | 0.8059 | 3 | 0.0, 1.0, 0.5 | Player-play event count: sack allowed |
| time_to_pressure_allowed | Float64 | continuous | seconds | inferred | 0.985 | 55 | 4.0, 4.09999990463257, 4.19999980926514, 3.09999990463257, 2.90000009536743 | Timing metric: time to pressure allowed |
| quick_pressure | String | flag |  | inferred | 0.988 | 2 | FALSE, TRUE | Player-level flag: quick pressure (TRUE/FALSE literals) |
| unblocked_pressure | String | flag |  | inferred | 0.7258 | 2 | FALSE, TRUE | Player-level flag: unblocked pressure (TRUE/FALSE literals) |
| blitzing | String | flag |  | inferred | 0.3627 | 2 | FALSE, TRUE | Player-level flag: blitzing (TRUE/FALSE literals) |
| player_get_off | Float64 | continuous | seconds | inferred | 0.862 | 1,211 | 1.66600000858307, 1.83299994468689, 0.957000017166138, 0.925000011920929, 1.13900005817413 | Timing metric: player get off |
| sack | Float64 | outcome | count | inferred | 0.7258 | 3 | 0.0, 0.5, 1.0 | Player-play event count: sack |
| tackle | Float64 | outcome | count | inferred | 0.9623 | 2 | 0.0, 1.0 | Player-play event count: tackle |
| assist | Float64 | outcome | count | inferred | 0.9623 | 2 | 1.0, 0.0 | Player-play event count: assist |
| caused_forced_fumble | Float64 | outcome | count | inferred | 0.9623 | 2 | 0.0, 1.0 | Player-play event count: caused forced fumble |
| recovered_fumble | Boolean | outcome | count | inferred | 0.9927 | 2 | false, true | Player-play event count: recovered fumble |
| tackle_for_loss | String | flag |  | inferred | 0.9623 | 2 | FALSE, TRUE | Player-level flag: tackle for loss (TRUE/FALSE literals) |
| time_to_pressure | Float64 | continuous | seconds | inferred | 0.9879 | 51 | 2.79999995231628, 2.29999995231628, 3.20000004768372, 5.19999980926514, 4.69999980926514 | Timing metric: time to pressure |
| time_to_qb_hurry | Float64 | continuous | seconds | inferred | 0.9875 | 51 | 2.79999995231628, 2.29999995231628, 2.90000009536743, 3.20000004768372, 5.19999980926514 | Timing metric: time to qb hurry |
| coverage_assignment | String | categorical |  | inferred | 0.7703 | 20 | DEEP_HALF_LEFT, MAN, NO_ASSIGNMENT, QUARTERS_OUTSIDE_LEFT, FLAT_LEFT | Defender coverage assignment |
| coverage_assignment_at_snap | String | categorical |  | inferred | 0.7723 | 20 | DEEP_HALF_LEFT, MAN, NO_ASSIGNMENT, QUARTERS_OUTSIDE_LEFT, CURL_FLAT_LEFT | Defender assignment at snap |


## `player_career_successes` — 9 columns

| column | bronze_dtype | semantic_role | unit | evidence | sample_null_frac | sample_n_distinct | sample_examples | meaning |
|---|---|---|---|---|---|---|---|---|
| nfl_id | Int64 | identifier |  | confirmed | 0 | 498 | 55900, 55901, 55903, 55904, 55905 | Player identifier; appears in every player-bearing table |
| career_offensive_snaps | Float64 | outcome | count | inferred | 0 | 188 | 1526.0, 2578.0, 3144.0, 1449.0, 0.0 | Career outcome: career offensive snaps |
| career_defensive_snaps | Float64 | outcome | count | inferred | 0 | 179 | 0.0, 181.0, 675.0, 2398.0, 1791.0 | Career outcome: career defensive snaps |
| career_special_teams_snaps | Float64 | outcome | count | inferred | 0 | 218 | 194.0, 198.0, 208.0, 140.0, 574.0 | Career outcome: career special teams snaps |
| career_games_active | Float64 | outcome | count | inferred | 0 | 50 | 46.0, 44.0, 51.0, 43.0, 30.0 | Career outcome: career games active |
| career_games_started | Float64 | outcome | count | inferred | 0 | 43 | 32.0, 40.0, 49.0, 20.0, 0.0 | Career outcome: career games started |
| ap_all_pro_1st_team | Float64 | outcome | count | inferred | 0 | 2 | 0.0, 1.0 | Career outcome: ap all pro 1st team |
| ap_all_pro_2nd_team | Float64 | outcome | count | inferred | 0 | 2 | 0.0, 1.0 | Career outcome: ap all pro 2nd team |
| pro_bowl_original_ballot | Float64 | outcome | count | inferred | 0 | 3 | 0.0, 1.0, 2.0 | Career outcome: pro bowl original ballot |


### ❓ Unresolved questions

- `games.game_key` — Alternate game key (league internal)
- `games.home_team_abbr` — No interpretation yet
- `games.visitor_team_abbr` — No interpretation yet
- `player_play.pass_result` — Pass result code (I = incomplete, R = run? unresolved)
- Timezone of `time` columns (UTC vs stadium local) — compare against kickoff times.
- Does `combine_tracking.event_id` identify one attempt or a whole session? (tested via key uniqueness).
- Coordinate frame of Combine tracking (field-relative like game tracking?).
- Are `s`/`a` smoothed by the vendor? (tested in tracking QC: reported vs. derived speed).
