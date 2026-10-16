# Milestone 12 — Research objectives

**Status:** authored 2026-10-15 · execution pending (no Milestone 12 number exists yet). Numbers appear only inside the generated block at the end, written by the notebooks from persisted tables.

## Central question
Does modelling the full structure, uncertainty and distribution of movement reveal stable, transferable athletic information that conventional feature engineering fails to detect?

The chain under test: **Combine movement → latent physical trait → observed NFL movement**, with uncertainty and position context. Each link is evaluated separately:

| Level | Meaning | Evidence that counts |
|---|---|---|
| Athletic measurement | what is observed (frames, attempts, games) | support counts, quality, censoring |
| Latent characteristic | what can be inferred under explicit assumptions | repeat-attempt / game-split reliability, measurement models |
| NFL movement expression | how the trait appears in games | game-split reliability of the NFL target |
| Predictive information | does pre-draft data improve held-out NFL prediction beyond official tests | athlete-held-out paired Δ RMSE with CI on identical athletes |
| Scouting relevance | how a decision-maker may use it without overstatement | descriptive only unless the predictive level is supported |

## Seven questions
| Q | Track | Question |
|---|---|---|
| Q1 | 12A | Does modelling the whole speed–acceleration curve retain useful information lost by scalar features? |
| Q2 | 12B | Could unreliable Combine features be masking athlete-level relationships? |
| Q3 | 12C | Are stable NFL movement distributions more useful targets than observed maxima or linear envelopes? |
| Q4 | 12D | Do several drills jointly reveal reproducible traits that single drill metrics measure unreliably? |
| Q5 | 12E | Can distribution matching recover cross-domain similarity missed by point estimates? |
| Q6 | 12F | Can an encoder trained on NFL and Combine sequences carry transferable physics beyond the Combine-only TCN? |
| Q7 | 12G | Which advanced methods beat simpler physical models under identical cohorts and protocol? |

A positive NFL predictive result is **not** required. Stable targets, unstable envelopes, measurement-error bounds, null transfers within the detectable range and failed replications are all legitimate findings.

## Starting evidence (from executed Milestones 00–11, not re-derived here)
- Tracked Combine speed is repeatable and largely redundant with the stopwatch; no tracked feature established incremental NFL prediction (D-06-1).
- Milestone 11B: two exploratory held-out gains (`forty.sprint_vmax → env.lin_V0`, `forty.v_max_obs → nfl.vmax_p90`) — candidate discoveries, re-tested in 12C · C7 under declared variants.
- Several turning / transition measures have very low repeatability (11C); envelope estimates depend on exposure (11B).

<!-- M12:RESULTS:START -->
_execution pending — written by 12G._
<!-- M12:RESULTS:END -->
