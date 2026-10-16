# Combine Intelligence Lab — research narrative (Milestone 09)

> Status 2026-10-14: authored, **not executed**. Results of Milestone 09 are not yet known; this document states the questions, the methods and what each result will mean. After the notebooks run, numbers are read from `EXECUTED_NUMBERS.md` (written by 09G) — never typed by hand.

## 1. Where the project stood (frozen, executed)

| Finding | Evidence (executed artifact) |
|---|---|
| 10 Hz Combine tracking measures top speed repeatably | C01: ICC 0.95 [0.94, 0.96] |
| Speed development (4 → 8 yd/s) is repeatable and partly beyond the stopwatch | C02: ICC 0.82; R² on official times 0.65 |
| A TCN-B fingerprint recognises an athlete's repeat attempt, also among stopwatch twins | C06: AUC 0.873; C07: matched AUC 0.796 |
| No tracked feature improved held-out NFL movement prediction | C08 (n = 284), decision D-06-1 |
| NFL movement windows are mostly context | C09: athlete share 6.3 % |
| Football transfer (receiver separation) not established | C14, C15 (n = 84, exploratory branch) |

Source: `reports/tables/advanced_movement/football_v1/final_evidence_registry.csv`. These conclusions are not revisited.

## 2. The question of Milestone 09

Not "does a fingerprint predict NFL output?" but: **what would a scout learn about a particular prospect that the official forty does not show — and how should that guide film review or further testing?**

## 3. Methods

**09A — NFL movement-demand atlas.** Eligible post-snap player-plays of Milestone 05A are segmented into nine physically defined episode types (accelerations from low and intermediate speed, transition into high speed, sustained high speed, maintenance, braking, re-acceleration after braking, redirection with valid heading, repeated changes). Thresholds are declared in `DemandConfig` before any distribution was read. Per position: frequency with exposure counts, quantiles of 6–9 dimensions, within- and between-athlete variability (one-way method of moments), observed context only, rule-chosen real example episodes.

**09B — cross-domain matching.** A shared physical space: band accelerations 2 → 4, 4 → 6, 6 → 8 yd/s computed with the same first-crossing rule on both sides (10 Hz both sides), plus largest speed. Coverage = share of the position's NFL episodes at or below the prospect's Combine value, with an interval from repeat-attempt error. Demands without a forty analogue abstain. TCN-B is not used across domains (trained on Combine sequences only).

**09C — Stopwatch Twins.** Same-position pairs within 0.03 s on the official forty (pre-declared). For each pair and curve feature: difference, standard error of the difference, verdict. Ranking by the largest standardized difference, restricted to features that are not unstable.

**09D — interpretation and cases.** Ten declared case types, deterministic selection, pre-draft only; one type shows observed NFL movement after selection, with exposure-count eligibility flagged. Three views per case. A wording guard refuses grades, ranking labels and causal promises.

**09E–09G — product.** A versioned application mart validated against its manifest; a Streamlit application; a demonstration notebook; a competition package.

## 4. How each result will be read

| Result | Reading |
|---|---|
| Twins distinguishable on a feature well above 5 % | tracking separates prospects the stopwatch cannot — a measurement statement |
| Near 5 % | the feature does not separate twins beyond noise; the Lab says *not distinguishable* |
| Coverage strongly rank-correlated with the forty | that demand dimension is already described by the stopwatch |
| Wide coverage intervals spanning 50 % | measurement error decides the side; the Lab abstains from a verdict |
| Small athlete share of NFL episodes | game demands are mostly the play, not the athlete — consistent with C09 |
| Any result | no claim of NFL predictive value: none is tested here, and earlier tests did not establish it |

## 5. Results

_Pending execution — see `EXECUTED_NUMBERS.md` and `CASE_STUDY_GALLERY_EXECUTED.md` once 09G has run._
