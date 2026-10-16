# NFL Big Data Bowl 2027 — Phase 03 integrated research report

> **Generated from saved tables only** (`python -m nflbdb.research.studies --report`).
> Numbers appear only after the user has executed the corresponding study. `⟨pending⟩` = not yet executed.
> Narrative fields marked _[interpret after execution]_ are written by the analyst after reviewing the outputs. Nothing in them may be pre-filled.

---

# Part I — Is the tracking data reliable? (Studies A, B, C)

## A1 · Quality-tier calibration
- **Research question:** Do provisional tiers or continuous quality dimensions separate reliable from unreliable trajectories?
- **Data coverage:**
  - Combine: {{tbl:A1_filter_funnel|domain=combine,stratum=all,scenario=all trajectories|n_trajectories|,}} trajectories, {{tbl:A1_filter_funnel|domain=combine,stratum=all,scenario=all trajectories|n_athletes|,}} athletes.
  - NFL: {{tbl:A1_filter_funnel|domain=nfl,stratum=all,scenario=all trajectories|n_trajectories|,}} player-plays, {{tbl:A1_filter_funnel|domain=nfl,stratum=all,scenario=all trajectories|n_athletes|,}} athletes.
- **Method:**
  - Eight quality dimensions; screening flag above the domain × stratum p99 (proposed).
  - Wilson and athlete-cluster bootstrap CIs.
  - Cliff's δ (non-A vs A) and Spearman ρ vs duration.
- **Main results:**
  - Tier-A share:
    - Combine {{tbl:A1_proportions|domain=combine,facet=all,level=all,indicator=tier_A|share|.2%}} (95 % CI {{tbl:A1_proportions|domain=combine,facet=all,level=all,indicator=tier_A|boot_lower|.2%}}–{{tbl:A1_proportions|domain=combine,facet=all,level=all,indicator=tier_A|boot_upper|.2%}}).
    - NFL {{tbl:A1_proportions|domain=nfl,facet=all,level=all,indicator=tier_A|share|.2%}} (95 % CI {{tbl:A1_proportions|domain=nfl,facet=all,level=all,indicator=tier_A|boot_lower|.2%}}–{{tbl:A1_proportions|domain=nfl,facet=all,level=all,indicator=tier_A|boot_upper|.2%}}).
  - Any screening flag:
    - Combine {{tbl:A1_proportions|domain=combine,facet=all,level=all,indicator=any_flag|share|.2%}}.
    - NFL {{tbl:A1_proportions|domain=nfl,facet=all,level=all,indicator=any_flag|share|.2%}}.
  - Retained after "tier A + no screening flag":
    - Combine {{tbl:A1_filter_funnel|domain=combine,stratum=all,scenario=tier A + no screening flag|share_athletes_retained|.1%}} of athletes.
    - NFL {{tbl:A1_filter_funnel|domain=nfl,stratum=all,scenario=tier A + no screening flag|share_athletes_retained|.1%}} of athletes.
- **Effect sizes:** Cliff's δ for vendor-speed MAD (NFL, non-A vs A): {{tbl:A_quality_results|analysis_id=A1.tier_contrast,domain=nfl,feature=speed_vendor_mad|effect_estimate|.2f}} [{{tbl:A_quality_results|analysis_id=A1.tier_contrast,domain=nfl,feature=speed_vendor_mad|ci_lower|.2f}}, {{tbl:A_quality_results|analysis_id=A1.tier_contrast,domain=nfl,feature=speed_vendor_mad|ci_upper|.2f}}].
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
  - Combine: {{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|n_frames|,}} frames / {{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|n_athletes|,}} athletes.
  - NFL sample: {{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|n_frames|,}} frames / {{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|n_athletes|,}} athletes.
- **Method:**
  - Frame residual r = derived − vendor.
  - MAE, RMSE, bias and Bland–Altman 95 % limits of agreement (LoA).
  - Athlete-cluster bootstrap CIs.
- **Main results:**

| domain | MAE (95 % CI) | bias | LoA | p95 \|r\| |
|---|---|---|---|---|
| Combine | {{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|mae|.3f}} ({{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|mae_lower|.3f}}–{{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|mae_upper|.3f}}) | {{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|bias|+.3f}} | [{{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|loa_lower|+.2f}}, {{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|loa_upper|+.2f}}] | {{tbl:A2_vendor_error_summary|domain=combine,facet=all,level=all|p95_abs|.3f}} |
| NFL sample | {{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|mae|.3f}} ({{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|mae_lower|.3f}}–{{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|mae_upper|.3f}}) | {{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|bias|+.3f}} | [{{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|loa_lower|+.2f}}, {{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|loa_upper|+.2f}}] | {{tbl:A2_vendor_error_summary|domain=nfl_sample,facet=all,level=all|p95_abs|.3f}} |

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
  - Tasks: {{rows:annotation_tasks}}.
  - Detector F1 (default rule, braking, all labels): {{tbl:detector_performance|rule=default,event_class=braking,facet=all,level=all,label_set=all_labels|f1|.2f}}.
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
