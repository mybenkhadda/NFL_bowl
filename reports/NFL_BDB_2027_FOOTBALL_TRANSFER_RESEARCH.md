# What does 10 Hz Combine tracking add beyond the stopwatch?

**NFL Big Data Bowl 2027 — football performance transfer, outcome validation and scouting decision intelligence**

> **How to read the numbers.** Every result is a token `‹Cnn: …›`. The token is the value of claim `Cnn` in the evidence registry (`reports/NFL_BDB_2027_EVIDENCE_REGISTRY.md`), read from a persisted table by notebook `08b6_final_research_evidence_package`. A token that says *pending* has no executed artifact yet: it shows no number and is not part of the conclusions. Rerunning 08B.6 refreshes every token in place.

## The question

Among NFL prospects with similar official Combine measurements, does high-frequency movement tracking reveal repeatable information that helps predict legitimate football outcomes — and how should a scouting department use that information responsibly?

We answer in five separate steps, because they are five different claims: what an outcome **means**, whether it is **measured reliably**, whether tracking **predicts** it beyond official tests, what tracking **adds** regardless of prediction, and what a scout can **do** with it.

## 1 · What tracking measures, and how well

Tracked forty-yard dashes give a speed curve for every attempt, not one time.

- Top speed is measured almost without error between attempts: ‹C01: ICC 0.95 [0.94, 0.96]; atlas grade Strong›.
- The time for speed to rise from 4 to 8 yd/s — *speed development* — is repeatable too: ‹C02: ICC 0.82 [0.78, 0.86]; R² on the stopwatch 0.65; atlas grade Provisional›.
- Single-frame extremes are not: ‹C03: ICC 0.35 [0.20, 0.49]; atlas grade Reject / not currently usable›. They describe the recording more than the athlete.

## 2 · How much of it the stopwatch already knows

Most of the top-speed information is already in the official times: ‹C04: R² on the stopwatch 0.87›. Speed development is only partly explained by them (C02). Whether a characteristic stays repeatable **after** position, forty, 10-yard split, weight and height are removed is tested directly, with a cross-fitted adjustment: ‹C17: repeatable beyond official tests: tracked top speed (p95), speed development 4 → 8 yd/s, FPCA speed-curve scores (K = 2), TCN-B movement fingerprint (8-d). Others: acceleration over 1 s from 4 yd/s (unstable); speed gain over 10 yd from 4 yd/s (weak_residual_signal)›.

## 3 · Two representations of the whole curve

- **FPCA.** The speed-by-distance curve is low-dimensional: ‹C05: two components explain 97.7 % of held-out curve variance (mean over 15 fold fits)›.
- **TCN-B.** A self-supervised encoder recognises an athlete's second attempt among his position group — ‹C06: repeat-attempt retrieval AUC 0.873 [0.856, 0.890]› — and still does among prospects with near-identical official results — ‹C07: matched-stopwatch retrieval AUC 0.796 [0.768, 0.824]›.

TCN-B is a repeatable athlete **fingerprint**. That is a statement about measurement. It is not a statement about the NFL.

## 4 · The first NFL test: movement outcomes

We froze NFL movement targets before looking at any Combine association, then asked whether tracking improves athlete-held-out prediction beyond official testing.

It did not: ‹C08: Δ RMSE +0.0003 [-0.0008, +0.0015] (+0.16 % of the official-model error); interval includes 0›. This conclusion is frozen and is not revisited here.

Two diagnoses explain why a null was likely:

- **Context.** How a player moves on an NFL snap is mostly assignment, alignment and opportunity: ‹C09: athlete share of within-position NFL window variance 6.3 % [5.0, 7.2]›.
- **Power.** Only large incremental effects were detectable: ‹C10: smallest effect detected with 80 % power — M2: planted share 0.23 of residual variance (Δ RMSE 6.7 %); M5: planted share 0.19 of residual variance (Δ RMSE 6.4 %)›.

## 5 · The second NFL test: football outcomes

Movement is not the game. We therefore audited which **football** outcomes the data can support.

- Three were observable: receiver separation at pass forward, tackles and sacks. Get-off was not usable: ‹C19: player_get_off: blocked_definition (source semantics unresolved)›.
- Recorded separation is the strongest candidate: ‹C11: odd / even game reliability 0.74 [0.52, 0.87]; semantic status plausible_inferred; screen status provisional›, with 91 linked receivers. Its vendor definition is **inferred**, not confirmed, and it is an interaction outcome — receiver, nearest defender, coverage, route and throw timing together. It is never called receiver ability.
- Tackle and sack counts have no opportunity denominator. Credits per snap are a participation rate, not a success probability, and the sack rate follows playing time. They are secondary at most.

The outcome registry records the decision and its limits: ‹C12: FREEZE → frozen_exploratory: FO01.S0 (primary, PROVISIONAL), FO12.T2 (secondary, PROVISIONAL), FO13.K0 (secondary, PROVISIONAL)›. Game-split reliability on the approved definition: ‹C13: FO01.S0: odd / even Spearman–Brown 0.742 (lower bound 0.516); reference 0.7, floor 0.5 — met›.

### The transfer benchmark

One primary comparison was fixed before any football outcome was related to a Combine value: engineered speed development added to official testing (`M2` versus `M1`), on athlete-held-out folds, with a refit-based permutation test.

- Primary: ‹C14: M2 versus M1 on FO01.S0: Δ RMSE -0.0007 [-0.0093, +0.0082], linkage permutation p 0.345 — not established (interval includes 0) (EXPLORATORY — source semantics inferred; definitions locked)›
- Secondary (TCN-B versus official testing): ‹C15: M5 versus M1 on FO01.S0: Δ RMSE -0.0108 [-0.0244, -0.0001], linkage permutation p 0.886 — the tracking model is worse (interval below 0) (EXPLORATORY — source semantics inferred; definitions locked)›
- What this cohort could have detected: ‹C16: smallest effect detected with 80 % power — M2: 80 % power not reached in the grid (max 0.60); M5: 80 % power not reached in the grid (max 0.29)›

Whatever these tokens become, their wording is capped by the registry. Under inferred source semantics, a positive result is an **exploratory association**, not an established predictive improvement.

## 6 · What a scout can use today

A descriptive scorecard with four sections that are never combined — speed capacity, speed development, movement fingerprint, evidence confidence — with intervals, pairwise tests on differences, and explicit abstention where a measurement is missing: ‹C18: every criterion met›. Details: `reports/NFL_BDB_2027_SCOUTING_VALUE.md`.

## 7 · What cannot be claimed

- No Combine tracking feature has an established incremental predictive value for NFL movement (C08).
- No talent ranking, NFL-potential score, bust or sleeper label is supported.
- A repeatable fingerprint is not a predictive one.
- Reliability of an outcome is not football skill.
- An exploratory association is a hypothesis for an independent cohort.

## 8 · What to test next

The linked cohort is the binding constraint. The cohort size a confirmatory study needs: ‹C20: power to detect a planted share 0.05 of residual variance by cohort size — n = 84: 0.12, n = 150: 0.24, n = 284: 0.48, n = 500: 0.78, n = 1000: 1.00›. Three further requirements follow from the audit: an authoritative definition of separation and of the opportunity fields; all-22 tracking, so that nearest-defender distance and rush participation can be recomputed; and a second, independent linked cohort for any association found here.

## Evidence hierarchy

| Level | Claims |
|---|---|
| Measurement validated | C01, C04, C09, C10, C13, C16 |
| Descriptively repeatable | C02, C05, C06, C07, C18 |
| Incremental information demonstrated | C17 (if its artifact says so) |
| Exploratory association | C11, C12; C14 / C15 if positive under an exploratory registry |
| NFL predictive improvement established | C14 only under a confirmed registry and a held-out improvement |
| Unsupported or unresolved | C03, C08, C19, C20; C14 / C15 if not established |

The level of each executed claim is recomputed from its artifact in the evidence registry; this table lists the default.

Methods and limits: `reports/NFL_BDB_2027_METHODS_AND_LIMITATIONS.md`.
