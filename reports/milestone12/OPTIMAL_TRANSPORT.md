# 12E — Conditional optimal transport matching

**Status:** authored 2026-10-15 · execution pending. Module `m12/movement_transport.py`; notebook `12e_conditional_optimal_transport_matching.ipynb`.

## Common support
Both domains on the 12.0 shared speed bins × acceleration bins merged to 0.5 yd/s² over −4…6 yd/s² (outside values dropped). Comparable state = accelerating (a_tan ≥ 1 yd/s²). Ground cost: squared Euclidean in physical units, 1 yd/s ≡ 1 yd/s² (declared).

## Distances
E0 Euclidean and E1 Mahalanobis on shared physical features (z-scored within position and domain) · E2 exact W2 (only with POT installed) · E3 sliced W2 · E4 debiased Sinkhorn divergence · E5 conditional W1 per speed bin (PRIMARY) · E6 unbalanced Sinkhorn on frame shares (opportunity mass).

## Evaluation
Within-domain consistency (NFL odd vs even games; Combine attempt 1 vs 2); **cross-domain linked-athlete retrieval AUC** within position (primary: E5 − E0, paired); controls: shuffled identity, uniform-mass query, coarser bins; sensitivity: ε grid, all-frames state; game / attempt bootstrap of each athlete's own distance; rank agreement with the 12C target; stopwatch twins (cluster bootstrap); downstream fold-fitted distance-to-reference features in the ladder. Retrieval is similarity, never football performance.

<!-- M12:RESULTS:START -->
_Generated 2026-10-15T11:02:14Z from executed artifacts (run mode full)._

## 12E — conclusion (inconclusive; run mode full; generated 2026-10-15T11:02:14Z)

### Research question
Q5 — Do distribution-level comparisons recover cross-domain similarity missed by point features?

### Declared primary hypothesis
Conditional OT (speed-bin conditional accelerating-state distributions) gives higher linked-athlete cross-domain retrieval AUC than standardized feature distance.

### Primary outcome / comparison
cross-domain retrieval AUC within position — E5 conditional OT vs E0 Euclidean features (paired athlete bootstrap) (evaluation target: `cross_domain_physical_transfer`)

### Sample
267 linked athletes; 12 speed bins × 20 acceleration bins.

### Main evidence (from persisted tables)
- E0_euclidean: AUC 0.501 [0.466, 0.537] (n=267)
- E5_conditional_w1: AUC 0.506 [0.471, 0.544] (n=267)
- E5_conditional_w1−E0_euclidean: Δ AUC 0.005 [-0.019, 0.031]

### Alternative explanations
- Role, position and movement-state exposure can influence cross-domain similarity.
- Within-position candidate pools do not eliminate differences in assignments and opportunity.

### Practical interpretation
Evaluates whether movement-distribution similarity can identify linked athletes; not football performance prediction.

### Limitations
- E5 pooled-mass weighting requires reconciliation with original implementation.
- E0 uses standardized histogram moments; exact original E0 feature set not verified.
- Attempt/game-split consistency, own-distance bootstrap and downstream transfer were not evaluated.
- Repeated candidate sharing can make simple athlete bootstrap intervals optimistic.

### Next research questions
- Validate original E0/E5 source definitions; rerun game/attempt-split and downstream experiments with original source data.


### Retrieval

| method | auc | auc_lo | auc_hi | n_athletes |
|---|---|---|---|---|
| E0_euclidean | 0.501 | 0.466 | 0.537 | 267 |
| E5_conditional_w1 | 0.506 | 0.471 | 0.544 | 267 |


| method | reference | diff | lo | hi | n_athletes |
|---|---|---|---|---|---|
| E5_conditional_w1 | E0_euclidean | 0.005 | -0.019 | 0.031 | 267 |
<!-- M12:RESULTS:END -->
