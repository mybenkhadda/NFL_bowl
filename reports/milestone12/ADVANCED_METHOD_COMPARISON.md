# 12G — Advanced method comparison

**Status:** authored 2026-10-15 · execution pending. Module `m12/synthesis.py`; notebook `12g_advanced_movement_research_synthesis.ipynb`.

## Rules
- Every track persists its comparisons in one schema (`synthesis.COMPARISON_COLUMNS`) and its evidence rows (`advanced_evaluation.EVIDENCE_COLUMNS`); 12G reads them and re-derives nothing.
- Six experiments: (1) scalar vs functional · (2) naive vs measurement-error correction · (3) maximum envelope vs distributional capacity · (4) single drills vs multi-view traits · (5) feature distance vs conditional OT · (6) physical features vs shared neural representations. Each keeps its own metric; nothing is pooled.
- Evidence categories (text statuses, never summed): measurement reliability · physical interpretability · additional information · cross-domain matching · NFL physical prediction · football-outcome transfer (not tested in Milestone 12) · computational practicality.
- One common-cohort tournament: M1 + each track's declared representation on identical held-out athletes (12C frozen target; secondary speed target), Holm and BH, permutation of the best block, cohort-size sensitivity. Exploratory.
- Evidence tiers: E4 held-out physical gain (Δ CI > 0) · E3 cross-domain association · E2 reliable · E1 measurable · E0 not evaluated — never promoted on a point estimate.

<!-- M12:RESULTS:START -->
_execution pending — written by 12G._
<!-- M12:RESULTS:END -->
