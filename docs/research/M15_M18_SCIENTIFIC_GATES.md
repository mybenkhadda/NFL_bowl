# M15–M18 scientific decision gates

**Status:** fixed before execution (2026-10-20).

**Precedence.** Thresholds are heuristics, chosen by reasoning and by continuity with M05/M06/M13/M14. If the user changes a gate, record the change in `docs/RESEARCH_DECISIONS.md` *before* the gate's inputs are produced. A gate changed after its inputs are seen is labelled `post_hoc` in every report.

**Evaluation.** Every gate is computed in code by `src/nflbowl/lab/m1X/gates.py`. The result is written to `tables/synthesis/m1X_gate.csv`, with one row per criterion: `criterion, value, ci_lo, ci_hi, threshold, pass, note`. A gate with any `blocked` input is `blocked`, never `pass`.

## Inherited rules

### Measurement status (M14)

| Status | Rule |
|---|---|
| `validated` | ICC(1,1) ≥ 0.6 **and** ICC lower CI ≥ 0.4 **and** minimum perturbation Spearman ≥ 0.8 **and** coverage ≥ 0.5 **and** ≥ 30 athletes with ≥ 2 trials |
| `invalid` | ICC < 0.4, coverage < 0.5, or < 30 athletes |
| `exploratory` | otherwise |

### Predictive status (M14)

| Status | Rule |
|---|---|
| `validated` | ΔRMSE CI lower > 0 **and** adjusted p ≤ 0.05 (Holm primary / BH secondary) **and** better in ≥ 4/5 folds |
| `exploratory` | Δ > 0 and bootstrap share of Δ ≤ 0 is ≤ 0.2 |
| `invalid` | CI upper < 0 |
| `inconclusive` | otherwise |

### Other inherited rules

- **Practical margin:** Δ ≥ 2 % of the reference RMSE.
- **Precise null (M13 H-D):** every CI includes 0 **and** every half-width ≤ 2 % of RMSE(reference).
- **Novelty:** OOF R² from M1 + position ≤ 0.5.

---

## G15 — representation readiness (end of M15)

Evaluated per representation R ∈ {PCA, STFT-AE, LSTM-AE, GRU-AE, TCN, dilated TCN, optional} at its selected d. d is chosen on design criteria (preservation + repeatability), never on retrieval or outcomes.

| # | Criterion | Rule | Source |
|---|---|---|---|
| G15.a | Physical preservation | Paired Δ (R − PCA) of the physical-unit speed RMSE and a_tan RMSE has CI upper ≤ +5 % of the PCA value (non-inferiority) | E15.C1 |
| G15.b | Event preservation | Event-matching F1 ≥ F1(PCA) − 0.05 and braking-onset timing error ≤ 0.2 s (median) | E15.C1 |
| G15.c | Repeatability | Median per-dimension trial ICC ≥ 0.4 **and** ≥ 25 % of dimensions with ICC ≥ 0.6 (held-out athletes, forty and agility separately) | E15.D2 |
| G15.d | No collapse | Effective rank ≥ 0.25·d, and `GEO.collapse_flags` clear | E15.D1 |
| G15.e | Seed stability | Procrustes-aligned cross-seed embedding r ≥ 0.8 (3 seeds) | E15.D2 |
| G15.f | Robustness | Clean-vs-10 % dropout self-retrieval top-1 ≥ 0.9 (reported; does not block) | E15.C2 |

**Decisions:**
- **GO (R advances to M16-B and M18 B4/B5):** a–e pass.
- **GO-descriptive:** a, b and d pass, but c or e fails. R is used for galleries and similarity displays only. It is never a predictor and never a cross-domain claim.
- **NO-GO:** a, b or d fails.

**Spectral handcrafted features (E15.A3)** advance only if they are estimable (|ρ(feature, length)| ≤ 0.3 within domain × kind, and surrogate AUC ≥ 0.7) **and** M14-`validated`.

**Prototype embeddings** advance only if `adoptable_inductive` (E15.00) **and** they pass a–e. Otherwise they are kept only as a historical reference.

### Fallbacks

| Outcome | Next step |
|---|---|
| No learned representation reaches GO | PCA (if GO) is the sole representation. Neural tracks of M17-C/D are cancelled. |
| PCA also fails c | No representation track at all. M18 B4–B6 are `blocked`. The M17-B event route is the only "beyond stopwatch" route. |
| Spectral features fail the artifact test | They are reported as a negative methods finding and never used as predictors. |

---

## G16 — verified baselines and cross-domain validity (end of M16)

### G16-A — Target reliability (per target)

Pass requires all of:
- reliability ≥ 0.7 with lower CI ≥ 0.5 (game split; season split acceptable; random-half split reported as weaker)
- ≥ 20 athletes at the exposure threshold in ≥ 3 positions
- |ρ(value, log exposure)| ≤ 0.3 within position
- context (`kind`) share ≤ 0.25, **or** reliability after demand-mix adjustment ≥ 0.7

T0 (`nfl.q90_shared`) stays the primary target unless it fails G16-A. In that case the primary moves to the most reliable task target, pre-declared in `manifests/preregistration.json` before M18.

### G16-B — M1 reproducibility

`TF.reproducibility_check` passes at 1e-8 for M1 and M1+F0 on the official frame.

**Hard stop if failed.**

### G16-C — Incremental information

On the primary target, `strict_common` cohort:
- **PASS:** M1+simple **or** M1+M14_reliable is `validated` vs M1.
- **PRECISE-NULL:** the precise-null rule holds for both.
- **INCONCLUSIVE:** otherwise. The M16 MDE (E16.A6) is reported with it.

### G16-D — Cross-domain similarity

For a representation that passed G15, all of:
- same-position pool MRR > F0 MRR (paired CI > 0, Holm on the pre-declared primary)
- top-1 above the analytic position-only expectation (E15.E2) in the all-position pool (CI > 0)
- the gain persists after CEM matching (CI > 0)

### Decisions

| G16 state | M17 | M18 |
|---|---|---|
| A ✓, B ✓, C PASS, D ✓ | full | full ladder |
| A ✓, B ✓, C PASS, D ✗ | Tracks A, B, E only; D optional | B0–B3 (+ B4/B5 if G15 GO, as Combine-only predictors) |
| A ✓, B ✓, C NULL/INCONCL., D any | Tracks A, B, E (aim: better measurement) | runs; primary family unchanged; null is the expected reading |
| A ✗ (all task targets) | Track A event repair also on the NFL side | T0 only |
| B ✗ | — | **blocked** until fixed |

**Revision rule.** If A or C fails, target definitions may be revised **once**, before M18. Each revision is recorded and labelled `post_hoc`. M18 runs on the frozen M16 freeze file whatever the outcome.

---

## G17 — signature validity (end of M17, per signature)

| # | Criterion | Rule |
|---|---|---|
| G17.a | Repeatability after context adjustment | ICC ≥ 0.6, lower CI ≥ 0.4 |
| G17.b | Signal > noise | Athlete variance share ≥ 0.4 (lower CI ≥ 0.25) |
| G17.c | Novelty | OOF R² from M1 + position ≤ 0.5 |
| G17.d | Generalisation | Held-out-athlete split-half r ≥ 0.5 |
| G17.e | NFL relevance | Within-position association with a G16-A target: \|ρ\| CI excludes 0 after BH |
| G17.f | Interpretability (for a *label* only) | \|ρ\| with the named physical quantity ≥ 0.5 (CI-based), stable across 3 seeds |

**Decisions:**
- a–e pass → `validated` signature, enters M18 B7.
- a–d pass, e fails → `measurement-valid, not NFL-relevant`. It is used in descriptive profiles only.
- a or b fails → `invalid`.
- f fails → the signature is kept unlabelled (`latent k`).

**Alignment damage rule (E15.E3 / E17.D1).** An aligned representation is `destructive` if its repeat AUC drops by more than 0.02 vs unaligned, or preservation worsens with CI > 0. A destructive alignment never enters M18.

---

## G18 — scouting relevance (final)

A feature family F earns **scouting relevance** only if all of the following hold:

1. F's measurement gate passed (G15 GO or G17 validated, or M14 `validated` for biomechanics).
2. M1 + F is `validated` vs M1 on the primary target (Holm over the pre-registered family).
3. Δ ≥ practical margin (2 % of RMSE(M1)).
4. Every negative control for F (random embedding, shuffled linkage, wrong drill, random label) has ΔRMSE CI including 0.
5. The refit permutation p (≥ 200 permutations) ≤ 0.05.
6. The gain is positive in ≥ 3 of the positions with n ≥ 30 (point estimate) and in ≥ 4/5 folds.

| Reading | Rule |
|---|---|
| Scouting-relevant | 1–6 |
| Promising, needs replication | 1–3 hold and one of 4–6 is borderline (stated explicitly) |
| Precise null | precise-null rule |
| Inconclusive | otherwise, reported with the MDE |

**Always required:** replication on independent athletes (future draft classes) is listed as necessary even for "scouting-relevant". This is the M14 rule: `validated ≠ scouting guidance`.

---

## Fallback research routes (summary)

| Primary route | Fallback if its gate fails |
|---|---|
| STFT/LSTM learned representations (G15) | PCA → handcrafted time-domain features → none |
| Cross-domain retrieval (G16-D) | Combine-only representations used as predictors without similarity claims |
| Task-specific NFL targets (G16-A) | T0 only |
| Repaired event metrics (G17.a) | Mixed-model athlete effects on simpler speed/acceleration families |
| Context-conditioned alignment (G17 + damage rule) | Procrustes / CORAL → unaligned |
| Learned fusion | Concatenation → single best representation |
| Tree learners | Ridge / Elastic Net |
| Separate position models | Partial pooling → pooled with position dummies |
| Predictive scouting value (G18) | Descriptive, measurement-validated profiles plus a precise null and the MDE ("what a bigger study would need") |
