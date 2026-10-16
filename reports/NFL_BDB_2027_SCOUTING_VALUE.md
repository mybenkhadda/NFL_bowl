# What scouts, coaches and general managers can use

**NFL Big Data Bowl 2027 — scouting value of high-frequency Combine tracking**

> Numbers are tokens `‹Cnn: …›` read from persisted tables by notebook `08b6_final_research_evidence_package`. A *pending* token has no executed artifact yet and carries no conclusion.

## The short version

Combine tracking tells you **how** a prospect ran his forty, reliably, and for some characteristics it tells you something the stopwatch does not. It has **not** been shown to tell you how he will play. Use it to compare prospects who look identical on the official sheet. Do not use it to rank talent.

## For scouts — comparing prospects with the same official results

**What you get.** A scorecard with four sections, shown side by side and never added up:

| Section | What it shows | Evidence |
|---|---|---|
| A · Speed capacity | official forty and 10-yard split, tracked top speed | top speed is measured almost without error: ‹C01: ICC 0.95 [0.94, 0.96]; atlas grade Strong› — and mostly repeats the stopwatch: ‹C04: R² on the stopwatch 0.87› |
| B · Speed development | time for speed to rise from 4 to 8 yd/s | repeatable, only partly explained by the stopwatch: ‹C02: ICC 0.82 [0.78, 0.86]; R² on the stopwatch 0.65; atlas grade Provisional› |
| C · Movement fingerprint | which prospects *move* alike (TCN-B similarity) | a repeat run is recognised among same-stopwatch prospects: ‹C07: matched-stopwatch retrieval AUC 0.796 [0.768, 0.824]› |
| D · Evidence confidence | attempts, recording quality, reference size | about the measurement, never about the athlete |

**How to read a comparison.** Two prospects differ when the **difference** between them exceeds its own measurement error. The scorecard computes that test for you and returns one of four verdicts: *distinguishable*, *not distinguishable*, *insufficient evidence*, *not observed*. Two overlapping percentile bars are not a test.

**What stays repeatable once official tests are accounted for:** ‹C17: repeatable beyond official tests: tracked top speed (p95), speed development 4 → 8 yd/s, FPCA speed-curve scores (K = 2), TCN-B movement fingerprint (8-d). Others: acceleration over 1 s from 4 yd/s (unstable); speed gain over 10 yd from 4 yd/s (weak_residual_signal)›.

**When the scorecard abstains.** If the 4 → 8 yd/s interval was not recorded, or a position has too few repeat attempts to estimate error, the cell says so. A missing recording is never turned into a low score.

## For player-development coaches — where to look

The speed curve shows *where* in the run two athletes with the same time differ: the first steps, the build through 4–8 yd/s, or top speed. A measured difference in one interval is a reason to assess that interval further.

It is not a prescription. Nothing in this study tests a training intervention, and a single maximal sprint at the Combine is one observation of a quality, not a diagnosis.

Extremes of a 10 Hz signal are not coaching information: ‹C03: ICC 0.35 [0.20, 0.49]; atlas grade Reject / not currently usable›.

## For general managers — what the evidence supports

| Kind of information | Status |
|---|---|
| Conventional: official Combine tests | the reference of every comparison |
| Reliable new measured information | speed development (C02); the full list after adjustment for official tests is C17 |
| Duplicated information | tracked top speed (C04): it confirms the stopwatch, it does not add to it |
| Experimental representation | the TCN-B fingerprint: repeatable (‹C06: repeat-attempt retrieval AUC 0.873 [0.856, 0.890]›), a similarity search only |
| Predictive value for NFL **movement** | not established: ‹C08: Δ RMSE +0.0003 [-0.0008, +0.0015] (+0.16 % of the official-model error); interval includes 0› |
| Predictive value for a **football outcome** (receiver separation) | ‹C14: M2 versus M1 on FO01.S0: Δ RMSE -0.0007 [-0.0093, +0.0082], linkage permutation p 0.345 — not established (interval includes 0) (EXPLORATORY — source semantics inferred; definitions locked)› |
| Scorecard integrity | ‹C18: every criterion met› |

The value for a front office is mostly negative knowledge, and it is real: it separates measurements that are new from measurements that repeat what the stopwatch says, and it marks the line no current evidence crosses. That prevents double counting and overconfidence. It has **not** been shown to improve a draft decision.

## What not to do with it

- Do not build an overall score or a draft grade from the four sections.
- Do not read a fingerprint neighbour as a player comparison for NFL production.
- Do not label a prospect a bust or a sleeper from tracking.
- Do not call recorded separation receiver ability, tackles per snap tackling ability, or sacks per snap pass-rush win rate.
- Do not treat an exploratory association as a finding.

## Example gallery

Real comparison cases — similar forty with different speed development, similar top speed with a different curve, similar build with distant fingerprints, high versus low measurement confidence, stable versus unstable repeat attempts — are selected by pre-draft rules fixed in code and never by an NFL label: `reports/NFL_BDB_2027_SCORECARD_GALLERY.md`.
