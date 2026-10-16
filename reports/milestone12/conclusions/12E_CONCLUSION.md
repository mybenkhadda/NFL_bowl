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
