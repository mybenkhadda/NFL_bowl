# NFL Big Data Bowl 2027 — Phase 03 integrated research report

> **Generated from saved tables only** (`python -m nflbdb.research.studies --report`).
> Numbers appear only after the user has executed the corresponding study. `⟨pending⟩` = not yet executed.
> Narrative fields marked _[interpret after execution]_ are written by the analyst after reviewing the outputs. Nothing in them may be pre-filled.

---

# Part I — Is the tracking data reliable? (Studies A, B, C)

## A1 · Quality-tier calibration
- **Research question:** Do provisional tiers or continuous quality dimensions separate reliable from unreliable trajectories?
- **Data coverage:**
  - Combine: 6,310 trajectories, 510 athletes.
  - NFL: 314,944 player-plays, 497 athletes.
- **Method:**
  - Eight quality dimensions; screening flag above the domain × stratum p99 (proposed).
  - Wilson and athlete-cluster bootstrap CIs.
  - Cliff's δ (non-A vs A) and Spearman ρ vs duration.
- **Main results:**
  - Tier-A share:
    - Combine 99.98% (95 % CI 99.95%–100.00%).
    - NFL 99.06% (95 % CI 98.88%–99.21%).
  - Any screening flag:
    - Combine 4.10%.
    - NFL 2.47%.
  - Retained after "tier A + no screening flag":
    - Combine 100.0% of athletes.
    - NFL 100.0% of athletes.
- **Effect sizes:** Cliff's δ for vendor-speed MAD (NFL, non-A vs A): 0.85 [0.81, 0.88].
- **Main figures:** A1-01, A1-04, A1-07, A1-10 (`reports/figures/phase_03/A_quality/`).
- **Practical interpretation:** _[interpret after execution]_
- **Limitations:**
  - The tiers are provisional, and some separation is circular because tier B is partly defined by vendor disagreement.
  - Heading validity is confounded with movement content.
  - Smoothing instability is pending Study B.
- **Decision & next action:** _[keep / revise / reject tiers — after execution]_

## A2 · Vendor versus derived speed
- **Research question:** Where and how much does coordinate-derived speed disagree with vendor speed?
- **Data coverage:**
  - Combine: 431,094 frames / 510 athletes.
  - NFL sample: 543,964 frames / 173 athletes.
- **Method:**
  - Frame residual r = derived − vendor.
  - MAE, RMSE, bias and Bland–Altman 95 % limits of agreement (LoA).
  - Athlete-cluster bootstrap CIs.
- **Main results:**

| domain | MAE (95 % CI) | bias | LoA | p95 \|r\| |
|---|---|---|---|---|
| Combine | 0.050 (0.049–0.051) | +0.001 | [-0.16, +0.16] | 0.163 |
| NFL sample | 0.088 (0.082–0.094) | +0.035 | [-0.21, +0.28] | 0.269 |

- **Main figures:** A2-01, A2-02, A2-08, A2-10.
- **Practical interpretation:** _[interpret after execution — e.g. is derived speed interchangeable with vendor speed for athlete-level features?]_
- **Limitations:**
  - Frame-level LoA describe frames.
  - NFL is a deterministic game sample.
  - Vendor speed is itself a processed estimate, not ground truth.
- **Decision & next action:** _[after execution]_

## B · Smoothing robustness — _module pending (Milestone 1, next)_
## C · Event detection
- **C1 · Threshold sensitivity:** module pending.
- **C2 · Reference annotation:**
  - Tasks: ⟨pending: annotation_tasks⟩.
  - Detector F1 (default rule, braking, all labels): ⟨pending: detector_performance⟩.
- **C3 · Maneuvers:** Milestone 2.

# Part II — What athletic movement characteristics can we measure? (Studies D, E) — _Milestone 2_
# Part III — Are these stable athlete traits? (Study F) — _Milestone 3_
# Part IV — How do movement profiles differ across athletes and positions? (Study G) — _Milestone 3_
# Part V — What transfers from Combine testing to NFL games? (Study H, exploratory) — _Milestone 4_
# Part VI — Could selection bias explain the results? (Study I) — _Milestone 5_
# Part VII — Prioritized hypotheses for the next phase — _[written after Parts I–VI have real results]_

For each section above, report the following:
- research question
- data coverage
- method
- main statistical results
- main figures
- effect sizes and uncertainty
- practical interpretation
- limitations
- decision and next action


---
## Unresolved placeholders
- `{{rows:annotation_tasks}}`
- `{{tbl:detector_performance|rule=default,event_class=braking,facet=all,level=all,label_set=all_labels|f1|.2f}}`
