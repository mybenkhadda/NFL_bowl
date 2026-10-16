_Generated from `feature_dictionary` and `atlas_feature_status` of this run (`RUN_MODE = 'full'`). Intervals are 95 %, athlete-level. Grades are heuristic screening labels for review._

**Seven levels of evidence (forty-yard dash and official measurements; candidates only from level 2)**
1. *Exists*: 1409 definitions in the dictionary (84 characterised on the forty or as official measurements; 1325 from other drills, evidence reused from Phase 04).
2. *Measurable*: 53 of 53 candidates have adequate support (coverage ≥ 0.8, ≥ 50 athletes with a repeat).
3. *Repeatable*: 12 candidates have a reliability interval above 0.75; 5 above 0.5; 19 below it.
4. *Distinct from the stopwatch*: 2 candidates carry reliable variance the official times do not explain (margin 0.1); 8 are redundant; 31 have a low correlation only, which is not evidence of new information.
5. *Plausible physical interpretation*: 20 candidates have a definition readable in physical terms; learned representations have none established.
6. *Ready for an NFL transfer evaluation*: decided by the frozen shortlist at the end of notebook 04B, not here.
7. *Demonstrated NFL predictive value*: **NOT TESTED in this milestone.** No NFL outcome was read.

**Strong candidates (4)**
- `p95_speed` (F2): reliability 0.95 [0.94, 0.96]; |ρ| with the official times 0.89; stopwatch R² 0.87; reliable share not explained 0.10; window R² 0.04; cohort R² 0.00 → redundant, acquisition robust.
- `peak_speed_interior` (F2): reliability 0.95 [0.93, 0.96]; |ρ| with the official times 0.89; stopwatch R² 0.87; reliable share not explained 0.10; window R² 0.02; cohort R² 0.00 → redundant, acquisition robust.
- `peak_speed` (F2): reliability 0.94 [0.93, 0.95]; |ρ| with the official times 0.88; stopwatch R² 0.86; reliable share not explained 0.11; window R² 0.03; cohort R² 0.00 → redundant, acquisition robust.
- `speed_gain_10yd_from_v4` (F6): reliability 0.81 [0.75, 0.86]; |ρ| with the official times 0.82; stopwatch R² 0.70; reliable share not explained 0.19; window R² 0.08; cohort R² 0.04 → redundant, acquisition robust.

**Provisional candidates (8)**
- `A_alt_origin_at_ref_speed component 1` (F5): reliability 0.96 [0.95, 0.97]; |ρ| with the official times 0.94; stopwatch R² 0.92; reliable share not explained 0.06; window R² 0.12; cohort R² 0.00 → redundant, acquisition uncertain.
- `A_speed_by_distance component 1` (F5): reliability 0.95 [0.94, 0.96]; |ρ| with the official times 0.93; stopwatch R² 0.92; reliable share not explained 0.06; window R² 0.32; cohort R² 0.00 → redundant, acquisition uncertain.
- `TCN-B · A (8-d)` (F5): reliability 0.93 [0.90, 0.94]; |ρ| with the official times n/a; stopwatch R² n/a; reliable share not explained n/a; window R² 0.10; cohort R² n/a → pending_04b, acquisition robust.
- `model_vmax` (F4): reliability 0.87 [0.84, 0.90]; |ρ| with the official times 0.83; stopwatch R² 0.77; reliable share not explained 0.16; window R² 0.30; cohort R² 0.01 → redundant, acquisition uncertain.
- `C_alt_origin_at_ref_speed component 1` (F5): reliability 0.83 [0.70, 0.92]; |ρ| with the official times 0.87; stopwatch R² 0.82; reliable share not explained 0.08; window R² 0.26; cohort R² 0.01 → redundant, acquisition uncertain.
- `time_v4_to_v8_s` (F6): reliability 0.82 [0.78, 0.86]; |ρ| with the official times 0.65; stopwatch R² 0.65; reliable share not explained 0.25; window R² 0.09; cohort R² 0.04 → beyond_stopwatch, acquisition uncertain.
- `TCN-B · C (8-d)` (F5): reliability 0.82 [0.71, 0.89]; |ρ| with the official times n/a; stopwatch R² n/a; reliable share not explained n/a; window R² 0.39; cohort R² n/a → pending_04b, acquisition uncertain.
- `accel_1s_from_v4` (F6): reliability 0.68 [0.57, 0.76]; |ρ| with the official times 0.66; stopwatch R² 0.48; reliable share not explained 0.32; window R² 0.11; cohort R² 0.09 → beyond_stopwatch, acquisition uncertain.

**Candidates by family and grade**: F2 Reject / not currently usable: 5; F2 Strong: 3; F4 Exploratory: 3; F4 Provisional: 1; F5 Exploratory: 15; F5 Provisional: 5; F5 Reject / not currently usable: 12; F6 Exploratory: 4; F6 Provisional: 2; F6 Reject / not currently usable: 2; F6 Strong: 1.

A feature with poor repeatability is not a training recommendation, and a grade is not a scouting metric: the football meaning and the usefulness of every candidate remain to be established against NFL movement.