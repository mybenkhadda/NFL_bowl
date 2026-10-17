# M13 research plan — NFL outcome transfer, reliability-aware biomechanics and scouting value

Notebook: `notebooks/nflbowl/M13_NFL_OUTCOME_TRANSFER_RESEARCH_COLAB.ipynb` · package 0.5.0 · authored statically (nothing executed).

## Primary question

Can 10 Hz NFL Combine movement tracking improve prediction of future NFL movement capacity (`nfl.q90_shared`) beyond the official Combine measurements (M1), under athlete-disjoint validation?

## Starting point (M12, reported values: reconciliation only)

- Inductive retrieval AUC was near chance for F0 and F2–F6 (≈ 0.50–0.51).
- F4 − F0 ΔAUC ≈ +0.010, with a CI that includes 0.
- F4 domain AUC ≈ 0.997 and MMD² ≈ 0.134.
- The full NFL outcome-transfer ladder was unfinished.

M13 reloads the saved M12 artifacts and never types these numbers into an analysis.

## Hypotheses and how each is read

| ID | Hypothesis | Evidence | Reading |
|---|---|---|---|
| H-A | Some inductive embedding (M1+F2…F6) lowers held-out RMSE vs M1 | Paired Δ RMSE, strict common cohort, Holm across encoders | Supported only if lower bound > 0 and p_holm < 0.05 |
| H-B | Reliable interpretable biomechanics (M1+BIO_*) lower RMSE vs M1 | Same; BH within the BIO family | As above |
| H-C | Both carry complementary signal | H-A and H-B | Recommends a combined model (not fitted in M13) |
| H-D | Neither helps (precise null) | Every interval includes 0 **and** every half-width ≤ 2 % of RMSE(M1) | Otherwise **inconclusive** |
| H-pos | Apparent predictability is mostly position | η², OOF position-only model (POS), skill beyond position | Descriptive |
| H-ret | Retrieval quality tracks transfer | Across-model Spearman (exact permutation) and within-model, two-way demeaned fold association | Association only |
| H-dom | Domain separability is compositional (length, duration, speed, state, position) | Domain AUC / MMD² after CEM on frozen embeddings | Sensitivity, no retraining |

## Design decisions

1. **Own experiment, adopted encoders.** M13 creates `RUN_DIR/checkpoints/experiments/m13_outcome_transfer_v1` under the identical science hash and copies valid M12 inductive `best.pt` files into it (sources are read-only). The validity rules are: same bundle and science, regime `inductive`, the same fold identity, the same architecture, and the file loads. Embedding files are adopted only when their weights checksum matches. A missing (model, fold) excludes that encoder (no transductive fallback). Training needs `run_training` **and** `allow_new_training`.
2. **The ladder is unchanged.** `Experiment.transfer_predictions` provides the official rungs. The POS and MEAN baselines run through the same `ae.oof_ladder` on the same frame. The baseline run re-fits M1 and must reproduce the ladder M1 to 1e-8 (an assertion).
3. **Fairness system (Part D).** All predictions go into one long table, validated for uniqueness, identical target, identical fold and the fold map. There are two cohorts:
   - `ladder_all`: imputation happens inside the design.
   - `strict_common`: complete M1, observed F0, and a real held-out embedding for every primary encoder.

   **Claims use the strict cohort only.** Native-coverage metrics are descriptive.
4. **Uncertainty.** A fold-stratified athlete bootstrap is used, with one resample matrix shared by every comparison of a table, so the comparisons are paired. Holm applies to the primary encoder family and BH to exploratory families.
5. **Reliability.** The M12 split-half Spearman–Brown estimator, 2r/(1+r), is unbounded below for r < 0, which explains the extreme negative values. M13 reports it unclamped, next to the one-way ICC(1,1), its variance components and its bound −1/(k0−1). Agility drills are pooled after kind-centring.
6. **Train-only selection.** Reliability thresholds, train-association top-k selection, position standardisation, interaction z-scores and position-model parameters are all computed from training athletes of each outer fold. Unit tests check that perturbing held-out outcomes or trials leaves every selection unchanged.
7. **Target provenance.** Prediction validation is athlete-disjoint. Target construction is not: the estimator choice, shared bins and C3 pooling were fitted once on all athletes, before any Combine value was joined. M13 never claims strictly inductive end-to-end target construction.
8. **F1 TCN.** It enters the primary ladder only if every held-out athlete lacks a training role in its own Milestone 06 fold (`handoff.f1_validity`). Otherwise it is excluded and reported.

## Limitations known before execution

- The target is a constructed movement-capacity quantity, not NFL success.
- Combine attempts per drill are few (forty ≈ 2), so equal-observation curves may only be estimable for k = 2.
- Subgroup (position) results are small-n and exploratory.
- Partial pooling fixes α at the outer-fold choices when it tunes κ.
- The navy figure palette is not re-validated against the navy surface (no execution allowed).
