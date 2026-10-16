# 15 — Movement measurement reliability (Phase 03 · M2, Family G)

_Status: **executed by the user** — `reports/tables/phase_03/feature_reliability.csv` and `reliable_feature_shortlist.csv` are on disk (drill × feature × cohort, including the pooled forty rows with 375 athletes / 750 attempts). The results have not been interpreted in this report. Within-position repeatability of the forty is taken further in `reports/SPRINT_RELIABILITY_RESCUE_RESULTS.md`. (Header corrected 2026-10-11 from those files.)_

## Method
- **Unit:** the athlete. Only athletes with ≥ 2 usable (tier A/B) attempts of the same canonical drill enter. A drill × feature needs ≥ 10 such athletes.
- **ICC(1):** one-way random effects, unbalanced (n0 correction). It measures how well a *single attempt* ranks athletes. **ICC(1,k̄)** applies Spearman–Brown to the athlete mean over k̄ attempts.
- **95 % CI:** percentile bootstrap over athletes (1,000 replicates, seed 20261008), vectorised over per-athlete sufficient statistics.
- **Typical error:** √MSW. CV % = 100·TE / mean.
- **Rank stability:** Spearman ρ between attempts 1 and 2 (chronological).
- **Drift:** mean attempt 2 − attempt 1, a learning or fatigue signal.
- **Best-vs-mean:** median gap between an athlete's best and mean, only where `HIGHER_IS_BETTER` defines "best".
- **Label:** from the **CI lower bound** (Koo & Li 2016 cut-offs: 0.5 / 0.75 / 0.9), so an uncertain ICC is never labelled good.
- **Guards (v2):**
  - Only each athlete's first 4 chronological attempts enter, so position-drill repeaters don't dominate.
  - An ICC needs ≥ 10 athletes with ≥ 2 attempts. A **label** needs ≥ 20 such athletes; otherwise the result is `insufficient_for_label`.
  - If ≥ 80 % of values are identical (`modal_share`), the label is `degenerate_distribution`.
  - CV % is computed only for strictly positive features.
  - Rank stability needs ≥ 10 athletes.
- Results are pooled and per cohort. M1 shows the 2025 Combine data have about ¼ of the positional noise of 2023/24.

Code: `src/nflbdb/features/reliability.py`; orchestration in `research/combine_analytics.py::reliability_all`. Tests: `tests/test_phase03_m2_combine.py` (hand-computed ICC, bootstrap determinism, end-to-end).

## Outputs
- `reports/tables/phase_03/feature_reliability.csv`: drill × feature × cohort.
- `reports/tables/phase_03/reliable_feature_shortlist.csv`: written by notebook 06; pooled CI lower bound ≥ 0.75.
- `reports/figures/phase_03/m2_feature_reliability.png/.svg`

## Results
_Executed; values are in `reports/tables/phase_03/feature_reliability.csv`. Not yet interpreted here._

## Limitations
- Few repeated attempts for the agility drills mean wide CIs.
- Repeating a drill is the athlete's choice (selection effect).
- ICC(1) assumes exchangeable attempts; drift is reported rather than modelled.
- Reliability is drill-specific and does not transfer to NFL measurement.
