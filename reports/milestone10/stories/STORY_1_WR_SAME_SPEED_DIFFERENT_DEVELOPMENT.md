# Same speed, different development — Ladd McConkey and Isaiah Bond (WR)

## Same speed, different development — Ladd McConkey and Isaiah Bond (WR)

![wr](../../figures/advanced_movement/product_v1/AM10B_wr_01_hero.png)

*Ladd McConkey vs Isaiah Bond: official stopwatch, recorded 10 Hz speed curves, and each difference against its uncertainty.*

**In 100 words.** Ladd McConkey and Isaiah Bond (WR) ran identical official forties of 4.39 s. Their 10 Hz tracking separates them on 3 of 8 measured features beyond measurement uncertainty; the largest is tracked top speed (p95): 10.93 yd/s versus 11.76 yd/s. The difference sits in late acceleration and high speed (above about 6 yd/s), mid-sprint speed development (about 4 to 8 yd/s). That makes it a specific question for film and a targeted test — not a verdict on either player, because no tracked Combine feature has shown NFL predictive value here.

**30-second narration.** Two WR prospects. Identical official forties of 4.39 s. The stopwatch calls them the same. Ten-hertz tracking does not: tracked top speed (p95), 10.93 yd/s against 11.76 yd/s, beyond measurement uncertainty. It sits in late acceleration and high speed (above about 6 yd/s). The Lab turns that into a film question and a targeted test. It does not say who will play better.

**Comparison card**

- position: WR
- names: Ladd McConkey · Isaiah Bond
- official forty: 4.39 · 4.39
- official ten: 1.52 · 1.51
- weight: 186.0 · 180.0
- tracked attempts: 2 · 2
- evidence tier: low · high
- distinguishable: 3 of 8 tested features
- phases: late acceleration and high speed (above about 6 yd/s) · mid-sprint speed development (about 4 to 8 yd/s)
- largest: tracked top speed (p95): 10.93 yd/s vs 11.76 yd/s

**Technical interpretation**

- Matching: stored 09C rule — same position, official forty within 0.03 s (|Δ| = 0.00 s), 10-yard split |Δ| = 0.01 s.
- tracked top speed (p95): 10.93 yd/s vs 11.76 yd/s, difference +0.84 yd/s [+0.65, +1.02] (z = +8.95); evidence status repeatable_beyond_official.
- speed at 30 yd: 10.87 yd/s vs 11.38 yd/s, difference +0.51 yd/s [+0.34, +0.67] (z = +5.88); evidence status repeatable_mostly_stopwatch.
- speed at 20 yd: 10.29 yd/s vs 10.62 yd/s, difference +0.34 yd/s [+0.17, +0.50] (z = +3.98); evidence status repeatable_mostly_stopwatch.
- Initial acceleration: band acceleration 2 → 4 yd/s 6.95 (Ladd McConkey) vs 5.23 (Isaiah Bond) yd/s², difference -1.72 [-3.05, -0.40] — inside its measurement uncertainty — not a difference.
- Early speed development: speed at 10 yd 9.28 (Ladd McConkey) vs 8.92 (Isaiah Bond) yd/s, difference -0.36 [-0.77, +0.06] — inside its measurement uncertainty — not a difference.
- Intermediate speed development: speed at 20 yd 10.29 (Ladd McConkey) vs 10.62 (Isaiah Bond) yd/s, difference +0.34 [+0.17, +0.50] — larger than its measurement uncertainty.
- Late sprint: speed at 30 yd 10.87 (Ladd McConkey) vs 11.38 (Isaiah Bond) yd/s, difference +0.51 [+0.34, +0.67] — larger than its measurement uncertainty.
- Maximum tracked speed: tracked top speed (p95) 10.93 (Ladd McConkey) vs 11.76 (Isaiah Bond) yd/s, difference +0.84 [+0.65, +1.02] — larger than its measurement uncertainty.
- Largely the same information as official testing: speed at 30 yd, speed at 20 yd.
- Uncertainty: each interval is ±1.96 standard errors of the difference built from both athletes' repeat-attempt variation; about 5 % of features would be called distinguishable by chance, and features of one run are correlated.
- C02: Speed development (time from 4 to 8 yd/s) is repeatable and only partly explained by the stopwatch — ICC 0.82 [0.78, 0.86]; R² on the stopwatch 0.65; atlas grade Provisional
- C04: Tracked top speed is largely the same information as official testing — R² on the stopwatch 0.87
- C08: TCN-B did NOT establish an improvement of held-out NFL sustained-acceleration prediction beyond official testing — Δ RMSE +0.0003 [-0.0008, +0.0015] (+0.16 % of the official-model error); interval includes 0
- C14: Primary football transfer comparison (engineered speed development beyond official testing) — M2 versus M1 on FO01.S0: Δ RMSE -0.0007 [-0.0093, +0.0082], linkage permutation p 0.345 — not established (interval includes 0) (EXPLORATORY — source semantics inferred; definitions locked)
- C17: Combine tracking characteristics that stay repeatable after official tests are accounted for — repeatable beyond official tests: tracked top speed (p95), speed development 4 → 8 yd/s, FPCA speed-curve scores (K = 2), TCN-B movement fingerprint (8-d). Others: acceleration over 1 s from 4 yd/s (unstable); speed gain over 10 yd from 4 yd/s (weak_residual_signal)
- The corresponding football advantage has not been established.

**Scouting follow-up checklist**

- [ ] Film — compare both on vertical movements and long runs after the catch.
- [ ] Film — compare both on the stem of intermediate movements before the break.
- [ ] Test — a longer sprint or flying 20-yard segment, because 40 yards can end before top speed is reached.
- [ ] Test — rolling-start (flying) 10-yard segments entered at a controlled speed, so the 4 → 8 yd/s band is measured directly.
- [ ] Confirm both official times and the 10-yard split timing method.
- [ ] Check the recording tiers before quoting a tracked value.

**Provenance.** mart: warehouse/gold/advanced_movement/intelligence_v1/app_mart/v1; mart_version: v1; mart_built_at: 2026-10-14T13:13:05Z; run_mode: full; notebook: 10b_signature_football_stories.ipynb; engine_version: 10.1; selection: the pair named in the brief (Ladd McConkey and Isaiah Bond) passes all six criteria; pair_id: 57155-59278; tables: ['stopwatch_twins', 'twin_differences', 'prospect_profiles', 'nfl_movement_demands']; selected_using_nfl_outcome: False

**If the data are absent.** No stopwatch twin under the stored matching rule (same position, official forty within 0.03 s) and the current filters.

