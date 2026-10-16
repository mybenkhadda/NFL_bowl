# Technical Appendix — Combine Intelligence Lab

*NFL Big Data Bowl 2027 · rendered ‹M10:generated_at› · mart ‹M10:mart_version› built ‹M10:mart_built_at› (‹M10:run_mode›).*
*Milestone 09 counts are typed from `reports/milestone09/EXECUTED_NUMBERS.md` and cross-checked by notebook 10E; earlier-milestone values are 08B.6 evidence claims (C-ids). No new audit was performed for this appendix: it records decisions already made (`docs/RESEARCH_DECISIONS.md`).*

---

## A · Datasets and cohorts

| Source | Content | Used for |
|---|---|---|
| NFL Combine tracking (10 Hz) | position, speed, tangential acceleration of each forty attempt | Combine curve features, replays |
| Official Combine results | forty, 10-yard split, weight, height | matching, redundancy |
| NFL regular-season tracking (silver `game_tracking`, 10 Hz) | player trajectories | NFL movement-demand atlas (09A) |
| Milestone 05A eligible player-plays | the exposure denominator | demand rates |

**Cohorts in the application mart:** 510 Combine prospects (positions DB, DL, OL, TE, WR); 418 with ≥ 1 eligible tracked forty; 793 tracked attempts; 416,507 usable NFL movement episodes in 9 demand categories; 3,266 stopwatch-twin pairs; 23 case studies of 10 declared types.

Earlier cohorts (frozen): Milestone 06 paired transfer cohort 284 athletes (D-06-1); Milestone 08B football-transfer cohort ≈ 91 linked athletes (FO01.S0).

## B · Definitions

| Term | Definition |
|---|---|
| Official test result | an observed standardized Combine measurement (stopwatch) |
| Tracking measurement | a value derived from the recorded 10 Hz motion |
| Repeatability | agreement of the same athlete's repeat attempts (within-position ICC) |
| Additional information | repeatability that remains after the official times are accounted for (08B.4 residual repeatability) |
| Movement-demand correspondence | physical similarity of a Combine value to observed NFL episodes of the same position (coverage, RMS robust z) |
| Football outcome prediction | a prospect-level forecast — NOT established as improved by tracking (C08, C14) |

These six are never combined into one score.

## C · Tracking sampling and curve features

10 Hz frames; speed and tangential acceleration as stored by the upstream pipeline (filter-edge frames flagged `boundary` and excluded from acceleration views). Eight curve features per attempt (09B, `movement_matching.combine_features`):

| Feature | Definition | 09B phase | Milestone 10 sprint phase |
|---|---|---|---|
| `acc_2_4` | (4 − 2) / time between first crossings of 2 and 4 yd/s | early | Initial acceleration |
| `acc_4_6` | same, 4 → 6 yd/s | middle | Early speed development |
| `speed_10yd` | tracked speed at 10 yd of path | early | Early speed development |
| `acc_6_8` | same, 6 → 8 yd/s | late | Intermediate speed development |
| `time_4_8` | time from 4 to 8 yd/s (atlas F6) | middle | Intermediate speed development |
| `speed_20yd` | speed at 20 yd | middle | Intermediate speed development |
| `speed_30yd` | speed at 30 yd | late | Late sprint |
| `top_speed` | 95th percentile of tracked speed (atlas F2) | late | Maximum tracked speed |

The five Milestone 10 sprint phases are a **presentation grouping** (`product_kit.SPRINT_PHASES`); the stored 09B phase is kept beside it. A phase without a feature recorded for both athletes is shown as *not identified*; no phase is filled from another.

## D · Measurement stability and redundancy

- Top speed ICC 0.95 [0.94, 0.96]; R² on official tests 0.87 (C01, C04).
- Speed development 4 → 8 yd/s ICC 0.82 [0.78, 0.86]; R² 0.65; repeatable beyond official tests (C02, C17).
- Peak tangential acceleration ICC 0.35 [0.20, 0.49] — rejected (C03).
- FPCA: two components explain 97.7 % of held-out curve variance (C05). TCN-B repeat-attempt AUC 0.873 [0.856, 0.890]; matched-stopwatch AUC 0.796 [0.768, 0.824] (C06, C07).
- Features with status `unstable` or `insufficient_support` are never used to describe an athlete or to lead a phase (`product_kit.usable`).

## E · Stopwatch matching

Stored rule (09C `TwinConfig`, fixed before any pair was examined): same position group; official forty within **0.03 s**; optional sub-conditions recorded per pair (10-yard split within 0.03 s; weight within 15 lb and height within 2 in). Pairs are ranked by the largest |z| of a difference. The application's filters only **narrow** the stored set (tolerance ≤ 0.03 s, split, draft class, recording tier, feature) — they never re-match.

Verdict per feature: difference B − A with standard error from the repeat-attempt variation of both athletes; distinguishable when |z| > 1.96. Distinguishable share by feature over 3,128 tested pairs: top speed 40.0 %, speed at 30 yd 38.0 %, speed at 20 yd 26.0 %, time 4 → 8 20.6 %, speed at 10 yd 16.7 %, acc 4 → 6 16.4 %, acc 2 → 4 14.5 %, acc 6 → 8 13.9 % (≈ 5 % expected by chance). Pairs share athletes: shares are descriptive, not a test. 2,097 of 3,266 pairs have ≥ 1 distinguishable feature.

## F · NFL demand segmentation

09A `DemandConfig` (thresholds declared before distributions were read): acceleration runs ≥ 1.0 yd/s² for ≥ 0.5 s with gain ≥ 1.5 yd/s, classified by start speed (< 2, 2–5, ≥ 5 yd/s reaching 7.5); sustained high speed ≥ 7.5 yd/s for ≥ 1 s; maintenance |a| < 1 yd/s² at ≥ 5 yd/s for ≥ 1 s; braking ≤ −1 yd/s² for ≥ 0.4 s from ≥ 4 yd/s with drop ≥ 2 yd/s; re-acceleration within 1 s of braking; redirection |Δheading| ≥ 45° at ≥ 60 deg/s and ≥ 3 yd/s; repeated change = ≥ 2 accelerations and ≥ 1 braking in a player-play. Edge episodes are excluded from profiles. A position × demand cell is described only with ≥ 30 episodes and ≥ 10 athletes. Rates are per 100 eligible player-plays (05A).

## G · Cross-domain comparison and the Movement Gap

**Shared dimensions** (same `band_accel` rule on both sides): `acc_2_4`, `acc_4_6`, `acc_6_8`, and top speed vs episode peak speed. **Coverage** = share of the position's NFL episodes at or below the prospect's Combine value, with an SE-based interval; **distance** = RMS over shared dimensions of (Combine value − NFL median)/(IQR/1.349). Demands without an analogue abstain.

**Coverage classes (10C, `movement_gap.coverage_class`).** *Direct*: registry analogue + same start state (acceleration from low speed; transition into high speed). *Partial*: analogue, different state (acceleration from intermediate speed — a standing-start band transit is not a rolling-start acceleration; sustained high speed — the forty can end before top speed). *Not measured*: no analogue. *Unavailable*: analogue exists but the cell is below the support rule or no prospect comparison is supported. The start-state table is declared from the 09A registry text, not from data.

‹M10:gap_class_table›

**Exposure index** per position = sum of supported cell rates by class; categories overlap, so it is an index, not a share of distinct movements. The focus position (Story 3) maximises the not-measured share among positions whose supported cells all have ≥ 30 athletes; comparisons flag eligible-play ratios > 2 and cells with < 30 athletes.

‹M10:gap_table›

## H · Statistical uncertainty

- Twin differences: SE of the difference from repeat-attempt variation; features within a run are correlated; multiple features per pair are not corrected (descriptive use only).
- Prospect values: ± 1.96 SE where ≥ 2 attempts exist; single-attempt athletes are flagged and carry no stability statement.
- Demand quantiles are over dependent episodes; athletes are the sample size.
- Percentile context (07D) uses oriented mid-rank percentiles with leave-one-out reference intervals.

## I · NFL transfer (frozen conclusions)

| Study | Comparison | Result | Claim |
|---|---|---|---|
| Milestone 06 | TCN-B (M5) vs official (M1), NFL sustained acceleration | Δ RMSE +0.0003 [−0.0008, +0.0015]; not established (case E) | C08 |
| Milestone 06 power | smallest effect with 80 % power | M2 planted share 0.23, M5 0.19 of residual variance | C10 |
| Milestone 08B | speed development (M2) vs M1, receiver separation FO01.S0 | Δ RMSE −0.0007 [−0.0093, +0.0082], linkage p 0.345; not established (exploratory) | C14 |
| Milestone 08B | TCN-B (M5) vs M1 | Δ RMSE −0.0108 [−0.0244, −0.0001]; tracking model worse | C15 |
| Future study | power by cohort size at planted share 0.05 | n = 284: 0.48 · n = 500: 0.78 · n = 1000: 1.00 | C20 |

## J · Decision simulator rules (10D)

Deterministic (`decision_simulator.simulate`). Inputs: stored profile, metrics, twin row and differences (only for stored twins — a non-twin pair gets no computed difference), similarity matrix, position demand rates, registry, evidence, gap matrix. Each line is one category: measured difference (distinguishable, repeatable feature) · uncertain difference (inside uncertainty, unrepeatable feature, or low recording tier) · missing observation · measurement mismatch (demand without forty analogue) · additional data needed. Phases → demands via `scouting_interpretation.PHASES`; film prompts via `POSITION_FILM`; tests via `PHASES[...]['test']` and `movement_gap.TEST_PROPOSALS`. The wording guard (`assert_language`) rejects grade, ranking and causal phrasing.

## K · Story selection (10B)

WR: six criteria (`signature_stories.WR_CRITERIA`); the pair named in the brief is used only if it passes all. Same forty: the 09D case of that type validated on identical official forty and ≥ 1 distinguishable feature. Gap: rule of section G. No NFL outcome or exposure value is an input (validated in `milestone10_validation.check_wr_selection_rules`).

## L · Limitations

One maximal sprint from rest without pads; 10 Hz resolution of the earliest band; recordings may start after first movement or end before top speed; vendor semantics of football outcomes inferred (08B exploratory); `player_get_off` unusable (C19); heading handedness unverified (redirection magnitude only); thresholds of the atlas are provisional; the movement gap describes exposure, not performance importance; proposed tests are unvalidated.

## M · Reproducibility

Execution order and checks: `REPRODUCIBILITY.md` (written by 10E). Code: `src/nflbdb/research/advanced_movement/{product_kit, movement_gap, decision_simulator, signature_stories, submission_builder, final_figures, milestone10_validation}.py`; app `apps/combine_intelligence_lab/`; tests `tests/test_milestone10_product.py`, `tests/test_cil_app_wiring.py`. Milestone 10 trains nothing and scans no warehouse table.
