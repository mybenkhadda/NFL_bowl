# NFL movement outcomes — results (Milestone 05)

**Status: TEMPLATE — notebooks 05A and 05B AUTHORED 2026-10-13, NOT EXECUTED.** Every `‹…›` is filled from a saved table of the
FULL run; nothing below is a result until then. Smoke-run numbers are never copied here.

Status vocabulary used in this report: Authored · Executed · Validated · Exploratory · Blocked · Not yet tested.

| Item | Status |
|---|---|
| Play measures and candidate outcomes (`nfl_outcomes.py`) | Authored |
| Reliability, exposure, context, sensitivity, screen, freeze (`outcome_reliability.py`) | Authored |
| Synthetic checks (16, `nfl_outcome_validation.py`) | Authored — ‹n passed / 16› |
| Notebook 05A full run | ‹Executed on …› |
| Notebook 05B full run | ‹Executed on …› |
| Freeze decision | ‹FREEZE / STOP / pending› |
| Combine-to-NFL transfer on the frozen target | Not yet tested |
| Vendor `player_get_off` | Blocked (definition unresolved; not used here) |

## 1. Question

Which NFL movement characteristics can be measured reliably enough, across independent games, to serve as targets for
Combine-to-NFL transfer? No Combine value is read in this milestone.

## 2. Data and denominators

Source: `nfl_outcome_manifest.json`, `nfl_window_funnel.csv`.

- Tracked player-plays with a snap tag: ‹…›. Eligible windows: ‹…› (‹…› %).
- Athletes with ≥ 1 eligible window: ‹…›. Athlete-games: ‹…›.
- By position group (athletes / athletes with ≥ 50 windows): DB ‹…/…› · DL ‹…/…› · OL ‹…/…› · TE ‹…/…› · WR ‹…/…›.
- Exclusions by first failing rule: ‹position out of scope … · nullified by penalty … · no quality record … · tier not usable … · short post-snap window …›.

## 3. Outcome definitions

Source: `nfl_outcome_dictionary.csv` (‹…› definitions; outcome definitions v‹…›, definition hash ‹…›).

Play measures come from derived kinematics (Savitzky–Golay window 7 frames, order 2), post-snap frames, fixed windows inside one
tracking segment, filter-edge windows excluded. Families: O1 realized speed · O2 sustained acceleration · O3 acceleration by
starting-speed band · O4 speed gain · O5 envelope · O6 braking and re-acceleration (exploratory).

## 4. Coverage and opportunity

Source: `nfl_outcome_coverage.csv`.

| Outcome | Athletes ≥ 50 windows | Position groups with ≥ 15 | Opportunity rate | ρ(value, log windows) |
|---|---|---|---|---|
| ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |

Starting-speed bands not feasible for a position (fewer than 15 athletes at the threshold): ‹…›.

## 5. Reliability across independent games

Source: `nfl_outcome_reliability.csv` (within position, athletes with ≥ 50 windows, stepped up to all games).

| Outcome | n athletes | Odd / even games [95 %] | Random halves | Context-adjusted | Lowest position |
|---|---|---|---|---|---|
| ‹…› | ‹…› | ‹…› [‹…›, ‹…›] | ‹…› | ‹…› | ‹…› |

Variance of the play measure (athlete / game / window share): ‹…›. Season to season: ‹…›.

## 6. Reliability against exposure

Source: `nfl_reliability_vs_exposure.csv`.

| Outcome | First 20 windows | First 50 | First 100 | First 200 | All (random halves) |
|---|---|---|---|---|---|
| ‹…› | ‹r (n)› | ‹…› | ‹…› | ‹…› | ‹…› |

Game budgets (first 2 / 4 / 8 / 16 games): ‹…›. Reading: ‹what an early exposure supports, in the words of the table›.

## 7. Context

Source: `nfl_outcome_context_audit.csv`. Largest context-cell share of the play-measure variance: ‹…›. Effect of the adjustment
on athlete values: ‹…›. Unobserved: assignment, opponent, scheme.

## 8. Smoothing, boundary and quality sensitivity

Source: `nfl_outcome_quality_sensitivity.csv`. Smallest rank agreement over filters per candidate: ‹…›. Largest level shift:
‹…› SD. Quality filters: ‹…›.

## 9. Screening and decision

Source: `nfl_outcome_screen.csv`, `nfl_movement_targets_v1.json`.

- Eligible: ‹…›. Provisional: ‹…›. Not eligible: ‹…›.
- Rule-based proposal: ‹FREEZE / STOP› — primary ‹…›, secondary ‹…›.
- Recorded decision: ‹…› — primary ‹…›, secondary ‹…›. Follows the proposal: ‹yes / no›. Rationale: ‹…›.
- Version hash of the frozen artifact: ‹…›. Combine data consulted: no.

## 10. Research gate 05

| Criterion | Evidence | Met |
|---|---|---|
| Precisely defined | dictionary row and frozen formula | ‹…› |
| Physically interpretable | unit, window, statistic | ‹…› |
| Adequately observed | coverage table | ‹…› |
| Reproducible across independent games | reliability table | ‹…› |
| Exposure and context sensitivity evaluated | exposure and context tables | ‹…› |
| Frozen without consulting Combine associations | no Combine source in the modules; check `no_combine_access` | ‹…› |

Gate: ‹PASS / STOP / PENDING›.

## 11. Limitations

- Observed movement depends on role, alignment, assignment and usage; a reliable outcome is not thereby a capability.
- Estimates describe athletes who reached the exposure threshold.
- Intervals cover athlete sampling, not the choice of filter, threshold or outcome; the selected outcome's reliability is optimistic to an unknown degree.
- Play class is inferred from the vendor `pass_result` code.
- ‹limitations specific to the executed results›

## 12. Next step

‹FREEZE: author Milestone 06 against `nfl_movement_targets_v1.json`. STOP: revise the outcome definitions as follows: …›
