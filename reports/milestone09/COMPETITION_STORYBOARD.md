# SAME STOPWATCH. DIFFERENT MOVEMENT.

**Combine Intelligence Lab — competition storyboard (Milestone 09)**

> Status 2026-10-14: authored, not executed. Every number in the final version comes from `reports/milestone09/EXECUTED_NUMBERS.md`, which notebook 09G writes from the application mart. Numbers quoted below for Milestones 04–08B are executed results, read from `reports/tables/advanced_movement/football_v1/final_evidence_registry.csv`.

**The 30-second version.** Two prospects ran the same official forty. Tracking shows they got there differently: one built speed earlier, the other later. The difference is larger than the measurement error. NFL games demand that kind of speed change all the time. The Lab tells a scout which phase of the run differs, whether it holds up between attempts, what film to watch next, and what not to conclude.

*The stopwatch tells you how fast. Tracking shows how the athlete got there. The Intelligence Lab helps you decide what to examine next.*

---

## Frame 1 — Opening: one time, two athletes
- **Visual:** `AM09G_01_hero_same_stopwatch_different_movement`, left panel (two equal stopwatch bars).
- **Line:** "Traditional evaluation sees two equivalent speed performances."
- **Numbers:** the hero pair's two forty times (EXECUTED_NUMBERS § Hero pair).

## Frame 2 — Discovery: where the curves part
- **Visual:** same figure, middle panel — every recorded 10 Hz attempt of both athletes over the position's 10–90 % band. Interactive: Stopwatch Twins page, synchronized sprint replay.
- **Line:** "Same finish line, different route through the speed bands."
- **Numbers:** the distinguishable features of the hero pair with their intervals (EXECUTED_NUMBERS § Hero pair).

## Frame 3 — Is it real? Uncertainty first
- **Visual:** same figure, right panel (difference in standard errors, grey band = inside the uncertainty); `AM09G_02_how_often_twins_differ`.
- **Line:** "A difference only counts when it beats the athlete's own attempt-to-attempt variation. About 5 % of pairs would pass by chance; here is how often twins actually differ."
- **Evidence behind it (executed):** speed development 4 → 8 yd/s is repeatable (ICC 0.82 [0.78, 0.86]) and only partly explained by the stopwatch (R² 0.65) — claim C02; tracked top speed is largely the stopwatch (R² 0.87) — claim C04.

## Frame 4 — Football connection: what games demand
- **Visual:** `AM09G_03_nfl_movement_demands` (episodes per 100 player-plays by position); Demand Explorer page with real episode replays.
- **Line:** "NFL games are full of speed changes — from rest, while moving, into high speed, braking and re-accelerating. Some of them the forty can describe; some it cannot."
- **Numbers:** EXECUTED_NUMBERS § NFL movement demands.

## Frame 5 — Matching, honestly labelled
- **Visual:** `AM09G_04_forty_inside_nfl_demands`; Movement Match Lab page.
- **Line:** "We place a maximal-effort Combine value inside the observed game distribution of the same position. That is a physical description — not a forecast."
- **Caveat on screen:** braking, re-acceleration, redirection, maintenance and repeated changes: *not described by the forty* (abstains).

## Frame 6 — Decision: better questions
- **Visual:** a case report (scout / coach / general-manager tabs) and a one-page intelligence card.
- **Line:** "Which phase differs? Is it repeatable? What film? Which targeted test? What is absent from official testing? What cannot be concluded?"

## Frame 7 — Scientific conclusion
- **Line:** "High-frequency tracking expands the description of athletic movement. Repeatability and physical similarity are not proven NFL predictive value."
- **Evidence (executed):** TCN-B did not improve held-out NFL sustained-acceleration prediction (Δ RMSE +0.0003 [−0.0008, +0.0015], n = 284) — C08; NFL movement windows are dominated by context (athlete share 6.3 %) — C09; the football transfer comparison was not established (n = 84) — C14.

## Frame 8 — Practical contribution
- **Line:** "An evidence-aware scouting tool that makes extra tracking information accessible, interpretable and actionable — for further evaluation, not for ranking."
- **Visual:** the application overview page.

---

**Do not say.** "predicts", "projects as", "translates to the NFL", "hidden star", "draft grade", "better receiver". The interpretation engine refuses these phrases in generated text (`scouting_interpretation.assert_language`).
