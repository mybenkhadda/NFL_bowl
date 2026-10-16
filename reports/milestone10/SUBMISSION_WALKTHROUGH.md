# Submission walkthrough

*How to present the Combine Intelligence Lab to judges and scouts. Rendered 2026-10-14T14:25:56Z.*

## 1 · What to show first (the 30-second version)

1. **Two identical forty times** — Bradford and McCormick, 5.08 s each (Overview).
2. **Two different sprints** — the synchronized replay and the phase chart (Stopwatch Twins).
3. **One sentence:** *"Tracking measures how speed is built, repeatably — it is not shown to predict NFL performance, so we use it to ask better scouting questions."*

If you only get 30 seconds, stop there.

## 2 · The best three real examples

| Story | Use it to show | Where |
|---|---|---|
| Same forty, different curve — Anthony Bradford and Mason McCormick (OL) | identical times, different curves, every difference against its uncertainty | Signature Stories → Story 2; Stopwatch Twins |
| Same speed, different development — Ladd McConkey and Isaiah Bond (WR) | a wide-receiver comparison chosen by written criteria (the pair named in the brief (Ladd McConkey and Isaiah Bond) passes all six criteria) | Signature Stories → Story 1 |
| The movement gap — what the forty cannot see for WR | what the forty cannot see for one position, with denominators | NFL Movement Gap; Signature Stories → Story 3 |

## 3 · Which charts matter

1. **Hero stopwatch-twin comparison** (`AM10E_01`) — the whole idea in one figure.
2. **Phase decomposition** (`AM10E_02`) — *where* the sprints differ.
3. **Twin distinguishability** (`AM10E_03`) — it is not one cherry-picked pair: every feature exceeds the 5 % chance line.
4. **Reliability vs redundancy** (`AM10E_04`) — which tracked numbers are new information and which restate the stopwatch.
5. **Movement Gap matrix** (`AM10E_06`) — the football connection and its limit.

Keep `AM10E_05`, `07`, `08`, `09`, `10` for questions and the written report.

## 4 · How to navigate the app (2 minutes)

Sidebar → **Presentation / demo mode** → the demo route links appear:

1. *Same forty — the featured pair* (Overview) → button **Open Stopwatch Twins with this pair**.
2. *Replay and phase differences* — press ▶; switch **time / distance**; point at the phase chart.
3. *What the forty cannot see* — the gap matrix; hover a cell for its denominator.
4. *Generate the scouting action* — question 1, Bradford vs McCormick; scroll the six sections; export the GM summary.
5. *Three stories* — the gallery.
6. *The evidence boundary* — expand "Original NFL transfer".

The athlete you pick on one page stays selected on the others. The data-version box in the sidebar shows which mart is loaded.

## 5 · How to describe the finding to judges

- "High-frequency Combine tracking contains **repeatable** information about how athletes develop speed, and some of it is **not** in the official times."
- "Thousands of same-time pairs differ beyond measurement noise; the differences concentrate in speed development."
- "NFL games ask for braking, redirection and re-acceleration that a straight forty does not contain; we quantify that gap per position, with exposure denominators."
- "We tested whether tracking predicts NFL performance better than the Combine. It did not. So the product is decision support for scouting review — it tells you what to watch and test, and what you cannot conclude."

## 6 · Predictive claims to avoid

| Do not say | Say instead |
|---|---|
| "He's more explosive, so he'll be better." | "He builds speed faster through 2–4 yd/s, beyond measurement uncertainty." |
| "This predicts NFL success." | "Its NFL predictive value was tested and not established." |
| "He matches the NFL demand." | "His Combine value sits at the Nth percentile of observed game episodes — a physical description." |
| "Our new drill finds better players." | "This is a candidate measurement for a demand the forty misses; its value is unproven." |
| Any grade, ranking, probability, "sleeper", "bust". | — |

## 7 · How to reproduce the research

Run in Jupyter, same `RUN_MODE`: Milestones 00–08B (executed) → `09a`…`09g` → `10a` → `10b` → `10c` → `10d` → `10e`. 10E writes `REPRODUCIBILITY.md` (inputs, fingerprints, check results) and `FIGURE_INDEX.md`. Tests: `tests/test_milestone10_product.py` (logic), `tests/test_cil_app_wiring.py` (pages). The app reads only `warehouse/gold/advanced_movement/intelligence_v1/app_mart/v1`.

## 8 · Handling missing data gracefully (live)

| Situation | What the app shows | What to say |
|---|---|---|
| Prospect without a tracked attempt | "No tracked forty attempt… nothing is imputed" | "We only show what was recorded." |
| Missing 10-yard split | the split marked *not recorded*; split filters skip the pair | "No split, no split comparison." |
| One attempt only | "stability cannot be estimated" | "We need a second recording to say it's repeatable." |
| Low recording tier | the caveat beside the value | "Measured, but with a lower-quality recording." |
| No stopwatch twin | the stored matching rule restated | "Nobody in the data ran within 0.03 s at that position." |
| Demand without forty analogue | "no physical analogue… no statement" | "That's the movement gap — film or a new test." |
| Too few NFL episodes | "too few observed episodes to describe it" | "Not enough game data for that position." |
