# Milestone 12 — Experiment registry

**Status:** authored 2026-10-15 · execution pending. The authoritative registry is code: `src/nflbdb/research/advanced_movement/m12/config.py` (`TRACKS`), fixed before any Milestone 12 number existed.

## Shared design
- **Unit:** the athlete. Attempts, episodes, games and seasons are nested observations; frames are never counted as a sample.
- **Folds:** 12.0 reuses the Milestone 11 athlete folds (Milestone 06 folds inside them); every fitted quantity (imputation, scaling, FPCA, factor model, OT reference, encoder, ridge penalty) comes from training athletes of the fold.
- **Ladder:** M0 position + draft year · M1 + height, weight, official forty, 10-yd split · M1+ adds official three-cone / short shuttle (sensitivity) · M2… advanced blocks. Every model of a comparison is scored on the identical athletes.
- **Inference:** paired athlete bootstrap (1,000 replicates), within-position athlete-label permutation for primaries, Holm over declared primaries, Benjamini–Hochberg for exploratory families. An interval containing 0 is "inconclusive at this n", never "no effect".
- **Regimes:** inductive vs transductive representation learning are reported in separate tables.
- **Statuses:** planned · authored · execution_pending · executed · incomplete · inconclusive · supported · rejected (`m12.config`).

## Declared tracks
| Track | Notebook | Primary outcome | Primary comparison | Depends on |
|---|---|---|---|---|
| 12.0 | `12_00_advanced_movement_research_foundation` | — (inventory) | — | — |
| 12C | `12c_distributional_nfl_movement_capacity` | `nfl.q90_shared` (estimator frozen by a declared reliability rule, NFL data only); secondary `nfl.game_top_speed_q75` | reliability of quantile estimators vs C0a / C0b; M3 vs M1 and M2 vs M1 (Holm) | 12.0 |
| 12B | `12b_measurement_error_and_latent_traits` | corrected vs naive coefficient of 7 preregistered features | B1 (regression calibration) vs B0 held out | 12.0 |
| 12A | `12a_functional_movement_transfer` | athlete-mean squared error of the NFL τ = 0.90 a(v) function | A2 (FPCA) vs A0 (scalar bands) | 12.0, 12C |
| 12D | `12d_multiview_latent_athletic_traits` | attempt-split test–retest of factor scores | selected factor model vs best single feature | 12.0 |
| 12E | `12e_conditional_optimal_transport_matching` | cross-domain retrieval AUC | E5 conditional OT vs E0 feature distance | 12.0 |
| 12F | `12f_physics_informed_shared_movement_encoder` | inductive retrieval AUC; Δ RMSE on the 12C target | F4 vs F0 and vs random encoder | 12.0, 12C |
| 12G | `12g_advanced_movement_research_synthesis` | tournament Δ RMSE vs M1 on identical athletes | each available block vs M1 (exploratory) | all |

## Execution order (user)
12.0 → 12C → 12B → 12A → 12D → 12E → 12F (may be deferred) → 12G. An unfinished 12F never blocks 12G.

<!-- M12:RESULTS:START -->
_Generated 2026-10-15T00:37:51Z from executed artifacts (run mode full)._

### Track registry (declared before execution)

| track | primary_outcome | primary_comparison | evaluation_target |
|---|---|---|---|
| 12.0 | — | — | measurement_reliability |
| 12C | nfl.q90_shared (estimator frozen by the declared reliability rule before any Combine value is joined); secondary nfl.game_top_speed_q75 | M3 vs M1 on the primary target; M2 vs M1 on the secondary target (Holm over the two) | cross_domain_physical_transfer |
| 12B | corrected vs naive standardized coefficient of each preregistered feature on its matched NFL target | regression calibration (B1) vs naive (B0) held-out Δ RMSE; SIMEX / EM coefficients descriptive | cross_domain_physical_transfer |
| 12A | athlete-mean integrated squared error over each athlete's supported shared speed bins | A2 (M1 + FPCA scores) vs A0 (M1 + scalar bands) | cross_domain_physical_transfer |
| 12D | test–retest correlation of factor scores (attempt-split) vs best single feature | D3 structured multi-view FA vs best single feature | measurement_reliability |
| 12E | cross-domain retrieval AUC within position | E5 conditional OT vs E0 Euclidean features (paired athlete bootstrap) | cross_domain_physical_transfer |
| 12F | inductive cross-domain retrieval AUC; Δ RMSE on the 12C primary target | F4 vs F0 engineered and vs random encoder | cross_domain_physical_transfer |
| 12G | Δ RMSE vs M1 on the 12C primary target, identical athletes | each available method vs M1 | cross_domain_physical_transfer |

### Feasibility (12.0, executed)

| track | eligible_athletes | linked_athletes | supported_positions | feasibility | missing_variables |
|---|---|---|---|---|---|
| 12C | 267 | 267 | DB, DL, OL, WR | ready | play context (down, alignment) is not in the histograms |
| 12B | 297 | 297 | DB, DL, OL, WR | ready | only 1–3 attempts per athlete |
| 12A | 310 | 310 | DB, DL, OL, WR | gated_on_upstream |  |
| 12D | 510 | 401 | DB, DL, OL, TE, WR | ready | drill participation depends on position |
| 12E | 267 | 267 | DB, DL, OL, WR | ready | Combine forty has few frames per speed bin |
| 12F | 510 | 401 | DB, DL, OL, TE, WR | ready |  |
<!-- M12:RESULTS:END -->
