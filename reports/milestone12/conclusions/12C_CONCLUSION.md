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
