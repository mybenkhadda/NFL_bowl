# NFL Big Data Bowl 2027 — executive summary

_Generated 2026-10-13T23:44:25Z from persisted artifacts. Raw tracking → Kinematics → Reliability → Stopwatch redundancy → FPCA → TCN fingerprint → NFL movement transfer → Context diagnosis → Football-outcome feasibility → Football transfer → Scouting decision intelligence._

**1. What is new about 10 Hz Combine tracking?**
- Tracked top speed (p95) of the forty is highly repeatable between attempts: ICC 0.95 [0.94, 0.96]; atlas grade Strong (n = 375) — _measurement_validated_ `C01`

**2. Which features are repeatable?**
- Speed development (time from 4 to 8 yd/s) is repeatable and only partly explained by the stopwatch: ICC 0.82 [0.78, 0.86]; R² on the stopwatch 0.65; atlas grade Provisional (n = 375) — _descriptively_repeatable_ `C02`
- Peak tangential acceleration is NOT repeatable enough to describe an athlete: ICC 0.35 [0.20, 0.49]; atlas grade Reject / not currently usable (n = 375) — _unsupported_or_unresolved_ `C03`

**3. Which features are already explained by official testing?**
- Tracked top speed is largely the same information as official testing: R² on the stopwatch 0.87 (n = 375) — _measurement_validated_ `C04`

**4. What did FPCA reveal?**
- The speed-by-distance curve of a forty is low-dimensional: two components explain 97.7 % of held-out curve variance (mean over 15 fold fits) — _descriptively_repeatable_ `C05`

**5. What did TCN-B learn?**
- TCN-B recognises an athlete's repeat attempt: a repeatable movement fingerprint: repeat-attempt retrieval AUC 0.873 [0.856, 0.890] (n = 265) — _descriptively_repeatable_ `C06`
- The fingerprint still recognises the athlete among prospects with near-identical official results: matched-stopwatch retrieval AUC 0.796 [0.768, 0.824] (n = 234) — _descriptively_repeatable_ `C07`

**6. Why did the first NFL prediction attempt fail to establish added value?**
- TCN-B did NOT establish an improvement of held-out NFL sustained-acceleration prediction beyond official testing: Δ RMSE +0.0003 [-0.0008, +0.0015] (+0.16 % of the official-model error); interval includes 0 (n = 284) — _unsupported_or_unresolved_ `C08`

**7. How do context and statistical power affect interpretation?**
- NFL movement windows are dominated by context and opportunity, not by the athlete: athlete share of within-position NFL window variance 6.3 % [5.0, 7.2] (n = 401) — _measurement_validated_ `C09`
- Only large incremental effects were detectable in the Milestone 06 design: smallest effect detected with 80 % power — M2: planted share 0.23 of residual variance (Δ RMSE 6.7 %); M5: planted share 0.19 of residual variance (Δ RMSE 6.4 %) (n = 284) — _measurement_validated_ `C10`

**8. Are football outcomes more promising?**
- Recorded receiver separation is a reliable athlete-level quantity with usable linked support: odd / even game reliability 0.74 [0.52, 0.87]; semantic status plausible_inferred; screen status provisional (n = 91) — _exploratory_association_ `C11`
- Football-outcome registry decision: FREEZE → frozen_exploratory: FO01.S0 (primary, PROVISIONAL), FO12.T2 (secondary, PROVISIONAL), FO13.K0 (secondary, PROVISIONAL) — _exploratory_association_ `C12`
- Gate B of the primary football outcome (game-split reliability and support): FO01.S0: odd / even Spearman–Brown 0.742 (lower bound 0.516); reference 0.7, floor 0.5 — met — _measurement_validated_ `C13`

**9. What did the new football transfer test demonstrate?**
- Primary football transfer comparison (engineered speed development beyond official testing): M2 versus M1 on FO01.S0: Δ RMSE -0.0007 [-0.0093, +0.0082], linkage permutation p 0.345 — not established (interval includes 0) (EXPLORATORY — source semantics inferred; definitions locked) (n = 84) — _unsupported_or_unresolved_ `C14`
- Secondary football transfer comparison (TCN-B beyond official testing): M5 versus M1 on FO01.S0: Δ RMSE -0.0108 [-0.0244, -0.0001], linkage permutation p 0.886 — the tracking model is worse (interval below 0) (EXPLORATORY — source semantics inferred; definitions locked) (n = 84) — _unsupported_or_unresolved_ `C15`
- Detectability of an incremental effect in the football cohort: smallest effect detected with 80 % power — M2: 80 % power not reached in the grid (max 0.60); M5: 80 % power not reached in the grid (max 0.29) (n = 84) — _measurement_validated_ `C16`

**10. What can scouts use today?**
- Combine tracking characteristics that stay repeatable after official tests are accounted for: repeatable beyond official tests: tracked top speed (p95), speed development 4 → 8 yd/s, FPCA speed-curve scores (K = 2), TCN-B movement fingerprint (8-d). Others: acceleration over 1 s from 4 yd/s (unstable); speed gain over 10 yd from 4 yd/s (weak_residual_signal) (n = 418) — _incremental_information_demonstrated_ `C17`
- The scouting scorecard is descriptive, uncertainty-aware and abstains where evidence is missing: every criterion met — _descriptively_repeatable_ `C18`

**11. What cannot yet be claimed?**
- Get-off cannot be used as an outcome: player_get_off: blocked_definition (source semantics unresolved) — _unsupported_or_unresolved_ `C19`

**12. What should be tested with future data?**
- Cohort size a future study needs: power to detect a planted share 0.05 of residual variance by cohort size — n = 84: 0.12, n = 150: 0.24, n = 284: 0.48, n = 500: 0.78, n = 1000: 1.00 — _unsupported_or_unresolved_ `C20`

## Evidence hierarchy

- **measurement_validated** — the quantity is measured repeatably and its relation to official tests is quantified: `C01`, `C04`, `C09`, `C10`, `C13`, `C16`
- **descriptively_repeatable** — a representation or feature is repeatable for an athlete; no claim beyond description: `C02`, `C05`, `C06`, `C07`, `C18`
- **incremental_information_demonstrated** — repeatable information remains after official tests are accounted for: `C17`
- **exploratory_association** — an association seen under exploratory status; a hypothesis for independent data: `C11`, `C12`
- **nfl_predictive_improvement_established** — held-out improvement of an NFL outcome prediction beyond official testing, under a confirmed definition: _none_
- **unsupported_or_unresolved** — tested and not established, or not testable with the current data and definitions: `C03`, `C08`, `C14`, `C15`, `C19`, `C20`