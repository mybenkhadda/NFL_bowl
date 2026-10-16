# 12C — Distributional NFL movement capacity

**Status:** authored 2026-10-15 · execution pending. Module `m12/distributional_capacity.py`; notebook `12c_distributional_nfl_movement_capacity.ipynb`.

## Definitions
- Data: `m11_envelope_hist` — athlete × game × 0.5 yd/s speed bin × 0.25 yd/s² acceleration bin frame counts of eligible post-snap NFL frames (no tracking rescan). Games are the replication unit.
- Q_τ(a | v): τ-quantile of tangential acceleration per supported speed bin (≥ 40 frames). τ ∈ {0.50, 0.75, 0.90, 0.95}; primary τ = 0.90 (declared: upper but not extreme).
- Estimators: C0a observed maximum · C0b Milestone 11 linear envelope (refitted per split) · C1 empirical quantile · C2 penalised quantile P-spline (λ tuned once on a dedicated game split) · C3 position partial pooling with game-bootstrap SEs (empirical Bayes) · C4 non-increasing above 2 yd/s for τ ≥ 0.9 · C5 distributional forest declared optional and **not implemented**.
- Trait `nfl.q{τ}_shared`: mean over the 12.0 shared speed bins, defined only when every shared bin is supported. Speed target `nfl.game_top_speed_q{50,75,90}`: quantiles of per-game maximum episode speed.
- **Target freeze:** estimator for `nfl.q90_shared` = highest odd/even game-split reliability among C1–C4 (≥ 0.5, else C1 flagged), written to `m12c_target_freeze.json` before any Combine value is joined; immutable.

## Evaluation
Reliability on odd/even, seeded random and season splits (Spearman–Brown, athlete bootstrap; position-adjusted; shape agreement; per bin); paired reliability difference vs C0a / C0b. Held-out game forecasts: pinball, coverage, CRPS (quantile decomposition), crossing rate. Transfer ladder M0–M6 on identical athletes; primaries Holm-adjusted; permutation; position subgroups; M1+ sensitivity. C7 replication of the two 11B findings under declared variants (alternative estimators, first / later seasons, exposure ≥ 16 / 24 games, positions, target reliability) — all reported, none selected.

## Limitations (known before execution)
No play context (down, alignment, role) in the histograms; an upper quantile mixes capacity with opportunity; an envelope intercept is never a measured maximum; one cohort.

<!-- M12:RESULTS:START -->
_Generated 2026-10-15T00:42:13Z from executed artifacts (run mode full)._

## 12C — conclusion (inconclusive; run mode full; generated 2026-10-15T00:42:13Z)

### Research question
Q3 — Are stable NFL movement distributions better targets than maxima / linear envelopes, and do Combine speed features predict them beyond official tests?

### Declared primary hypothesis
Upper conditional acceleration quantiles (τ = 0.90) over the shared speed range are more game-split reliable than the observed-maximum envelope; fitted sprint Vmax (M3) lowers held-out RMSE of that target vs M1.

### Primary outcome / comparison
nfl.q90_shared (estimator frozen by the declared reliability rule before any Combine value is joined); secondary nfl.game_top_speed_q75 — M3 vs M1 on the primary target; M2 vs M1 on the secondary target (Holm over the two) (evaluation target: `cross_domain_physical_transfer`)

### Sample
316 NFL athletes (≥ 8 games); transfer cohort 267 linked athletes; shared range 1.0–7.0 yd/s.

### Main evidence (from persisted tables)
- frozen estimator C3_pooled (τ = 0.9)
- C1_empirical vs C0a_observed_max: reliability 0.996 vs 0.959, Δ 0.037 [0.023, 0.054] (n=241) — supported
- C1_empirical vs C0b_linear_envelope: reliability 0.996 vs 0.995, Δ 0.001 [-0.000, 0.002] (n=241) — inconclusive
- C2_smooth_qr vs C0a_observed_max: reliability 0.996 vs 0.959, Δ 0.037 [0.023, 0.054] (n=241) — supported
- C2_smooth_qr vs C0b_linear_envelope: reliability 0.996 vs 0.995, Δ 0.001 [-0.000, 0.002] (n=241) — inconclusive
- C3_pooled vs C0a_observed_max: reliability 0.998 vs 0.959, Δ 0.039 [0.025, 0.056] (n=241) — supported
- C3_pooled vs C0b_linear_envelope: reliability 0.998 vs 0.995, Δ 0.003 [0.002, 0.004] (n=241) — supported
- C4_shape vs C0a_observed_max: reliability 0.996 vs 0.959, Δ 0.037 [0.023, 0.054] (n=241) — supported
- C4_shape vs C0b_linear_envelope: reliability 0.996 vs 0.995, Δ 0.001 [-0.000, 0.002] (n=241) — inconclusive
- M3 vs M1 → nfl.q90_shared: Δ RMSE -0.0008 [-0.0018, 0.0002] (n=246, Holm p 0.949) — inconclusive
- M2 vs M1 → nfl.game_top_speed_q75: Δ RMSE 0.0050 [-0.0010, 0.0107] (n=266, Holm p 0.098) — inconclusive
- replication forty.sprint_vmax: 9 variants, share with Δ CI > 0 = 0.00, share Δ > 0 = 0.11
- replication forty.v_max_obs: 8 variants, share with Δ CI > 0 = 0.38, share Δ > 0 = 1.00

### Alternative explanations
- Role and play design decide when athletes accelerate: an upper quantile mixes capacity with opportunity.
- Position and body size are shared causes of Combine and NFL speed; M1 holds them but not every within-position difference.
- The 11B high-speed gains were selected among many comparisons; a replication share near the noise level is expected under no effect.

### Practical interpretation
Defines whether a reproducible, context-aware NFL movement target exists that can replace unstable maxima, and whether Combine speed predicts it beyond the stopwatch.

### Limitations
- One cohort; held-out gains are exploratory until an independent draft class confirms them.
- No play context (down, alignment, route) in the histograms.
- C5 (distributional forest) not implemented.
- Shared range excludes unsupported high-speed bins by design.

### Next research questions
- Re-run the frozen target on a later draft class (independent confirmation).
- Condition quantiles on play context where tracking supplies it.

### Reliability (odd / even games, τ = primary)

| estimator | n_athletes | reliability | rel_lo | rel_hi | shape_agreement |
|---|---|---|---|---|---|
| C0a_observed_max | 241 | 0.959 | 0.942 | 0.972 | 0.954 |
| C0b_linear_envelope | 241 | 0.995 | 0.993 | 0.996 | 1.000 |
| C1_empirical | 241 | 0.996 | 0.995 | 0.997 | 0.981 |
| C2_smooth_qr | 241 | 0.996 | 0.995 | 0.997 | 0.994 |
| C3_pooled | 241 | 0.998 | 0.997 | 0.998 | 0.988 |
| C4_shape | 241 | 0.996 | 0.995 | 0.997 | 0.985 |

### Transfer (identical athletes per target)

| target | model | reference | n_athletes | delta_rmse | delta_lo | delta_hi | p_holm_primary | verdict |
|---|---|---|---|---|---|---|---|---|
| nfl.q90_shared | M1 | M0 | 246 | 0.050 | 0.037 | 0.065 | — | model_better |
| nfl.q90_shared | M2 | M1 | 246 | -0.001 | -0.003 | 0.002 | — | inconclusive |
| nfl.q90_shared | M3 | M1 | 246 | -0.001 | -0.002 | 0.000 | 0.949 | inconclusive |
| nfl.q90_shared | M4 | M1 | 246 | -0.001 | -0.004 | 0.000 | — | inconclusive |
| nfl.q90_shared | M5 | M1 | 246 | -0.000 | -0.002 | 0.001 | — | inconclusive |
| nfl.game_top_speed_q75 | M1 | M0 | 266 | 0.070 | 0.041 | 0.100 | — | model_better |
| nfl.game_top_speed_q75 | M2 | M1 | 266 | 0.005 | -0.001 | 0.011 | 0.098 | inconclusive |
| nfl.game_top_speed_q75 | M3 | M1 | 266 | -0.004 | -0.008 | 0.000 | — | inconclusive |
| nfl.game_top_speed_q75 | M4 | M1 | 266 | 0.000 | -0.003 | 0.004 | — | inconclusive |
| nfl.game_top_speed_q75 | M5 | M1 | 266 | -0.002 | -0.009 | 0.006 | — | inconclusive |
| nfl.q95_shared | M1 | M0 | 246 | 0.065 | 0.048 | 0.083 | — | model_better |
| nfl.q95_shared | M2 | M1 | 246 | 0.000 | -0.002 | 0.003 | — | inconclusive |
| nfl.q95_shared | M3 | M1 | 246 | -0.001 | -0.001 | 0.000 | — | inconclusive |
| nfl.q95_shared | M4 | M1 | 246 | -0.001 | -0.004 | 0.000 | — | inconclusive |
| nfl.q95_shared | M5 | M1 | 246 | 0.000 | -0.002 | 0.002 | — | inconclusive |
| nfl.q50_shared | M1 | M0 | 246 | -0.000 | -0.001 | 0.001 | — | inconclusive |
| nfl.q50_shared | M2 | M1 | 246 | -0.000 | -0.000 | -0.000 | — | model_worse |
| nfl.q50_shared | M3 | M1 | 246 | -0.000 | -0.001 | 0.001 | — | inconclusive |
| nfl.q50_shared | M4 | M1 | 246 | -0.000 | -0.001 | 0.001 | — | inconclusive |
| nfl.q50_shared | M5 | M1 | 246 | -0.001 | -0.003 | 0.000 | — | inconclusive |
| env.lin_V0 | M1 | M0 | 266 | 0.013 | -0.007 | 0.043 | — | inconclusive |
| env.lin_V0 | M2 | M1 | 266 | -0.005 | -0.015 | -0.001 | — | model_worse |
| env.lin_V0 | M3 | M1 | 266 | -0.004 | -0.011 | -0.001 | — | model_worse |
| env.lin_V0 | M4 | M1 | 266 | -0.003 | -0.009 | -0.000 | — | model_worse |
| env.lin_V0 | M5 | M1 | 266 | -0.019 | -0.055 | -0.000 | — | model_worse |
| nfl.vmax_p90 | M1 | M0 | 146 | 0.017 | -0.005 | 0.036 | — | inconclusive |
| nfl.vmax_p90 | M2 | M1 | 146 | 0.013 | 0.003 | 0.023 | — | model_better |
| nfl.vmax_p90 | M3 | M1 | 146 | 0.003 | -0.004 | 0.008 | — | inconclusive |
| nfl.vmax_p90 | M4 | M1 | 146 | -0.002 | -0.005 | 0.000 | — | inconclusive |
| nfl.vmax_p90 | M5 | M1 | 146 | -0.004 | -0.013 | 0.004 | — | inconclusive |
<!-- M12:RESULTS:END -->
