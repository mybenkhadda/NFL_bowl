# Combine Intelligence Lab

**Same stopwatch. Different movement. Better questions.**

A scouting decision-support application built on Milestone 09 of the NFL Big Data Bowl 2027 project. It lets a scout, position coach or general manager compare prospects with near-identical official Combine tests, see where their 10 Hz sprint curves differ, check whether a difference exceeds its measurement uncertainty, relate it in physical terms to the speed-change demands observed in NFL games, and export a report.

It is **descriptive**. In this project, tracked Combine features did not improve held-out prediction of NFL movement (Milestone 06) or of receiver separation (Milestone 08B) beyond official testing. The application shows no grade, no ranking and no NFL forecast.

> **Status (2026-10-14).** The Milestone 09 mart is built (full mode, 2026-10-14T13:13Z); compiled caches suggest the Milestone 09 app was launched, but no result of that launch was reported. **The Milestone 10 version (10 pages, synchronized replay, NFL Movement Gap, Scouting Decision Simulator, Signature Stories, audience PDF exports) is authored and NOT yet launched.** Run notebook `10a_application_readiness` first, then walk `reports/milestone10/APP_BROWSER_CHECKLIST.md`. Send back any traceback.

## 1 · Prepare the data (Jupyter, in order, same `RUN_MODE`)

| Notebook | Writes |
|---|---|
| `09a_nfl_movement_demand_atlas` | NFL movement-demand episodes, profiles, examples |
| `09b_cross_domain_movement_matching` | Combine curve features, prospect-to-demand similarity |
| `09c_stopwatch_twins_discovery` | stopwatch-twin pairs and their differences |
| `09d_football_interpretation_and_case_studies` | case studies, three views, interpretation registry |
| `09e_application_data_mart` | **the application mart** (the only thing the app reads) |

Notebooks are in `notebooks/advanced_movement/`. Each one needs the earlier milestones' outputs in the same run mode.

## 2 · Launch it (your terminal, from the project root)

```
pip install -r apps/combine_intelligence_lab/requirements.txt
cd apps/combine_intelligence_lab
streamlit run app.py
```

Launching from inside the app folder lets Streamlit pick up `.streamlit/config.toml` (dark theme). Launched from elsewhere, the page CSS still applies the design; only Streamlit's own widget theme falls back to its default.

The app reads `warehouse/gold/advanced_movement/intelligence_v1/app_mart/v1/`. To read another mart (for example the smoke mart under `…/_sample/intelligence_v1/app_mart/v1`), set the environment variable before launching:

```
# PowerShell
$env:CIL_MART_DIR = "C:\path\to\app_mart\v1"; streamlit run apps/combine_intelligence_lab/app.py
```

A smoke mart is labelled on every page.

## 3 · Pages

| Group | Page | What it answers |
|---|---|---|
| Start | Overview | the signature same-forty comparison, the three questions, data coverage, demand categories |
| Prospects | Prospect Intelligence | official tests, anthropometrics, position context, every attempt (speed vs time / distance, acceleration vs speed), replay, measures with ⓘ evidence tooltips and repeat-attempt stability |
| | Stopwatch Twins | stored-rule comparables with narrowing filters (tolerance, split, draft class, tier, feature); synchronized replay (time / distance, speed / acceleration, attempt, playback); phase decomposition; reliability vs redundancy; NFL correspondence; HTML export |
| | Signature Stories | the three rule-selected stories with provenance; the 23-case gallery |
| NFL movement | NFL Movement Demands | rates per 100 eligible player-plays, exposure table, distributions, episode replays, context |
| | NFL Movement Gap | coverage matrix (direct / partial / not measured / unavailable), gap by position, two-position comparison with exposure flags, candidate tests, executive summary |
| | Movement Match Lab | coverage, RMS robust-z distance, abstentions, alignment, examples |
| Decide | Scouting Decision Simulator | question → prospects → evidence → six-part action → scout / coach / GM report (HTML, PDF) |
| | Report Generator | prospect report (HTML, one-page PDF card, full PDF) and three-view case reports |
| Evidence | Research Evidence | nine plain-language explanations with the executed claims; registries |

The athlete selected on one page stays selected on the others. The sidebar holds a position filter, a presentation / demo mode (larger type, demo route links) and the data-version indicator.

## 4 · Architecture

```
app.py                 page config, sidebar, st.navigation
data_access.py         mart reader (cached per mart build); re-orients a twin pair so the selected athlete is A
reporting.py           HTML reports and the PDF card; text from src/nflbdb/research/advanced_movement/scouting_interpretation.py
components/ui.py       theme, stat tiles, evidence badges (icon + label), tables, athlete picker
visualizations/charts.py   Plotly figures (replays, curves, forests, heatmaps, distributions)
pages/1_…10_*.py       the ten pages (8–10 added in Milestone 10)
src/…/product_kit.py, movement_gap.py, decision_simulator.py, signature_stories.py
                       Milestone 10 engine (standard library only, shared with notebooks 10A–10E)
.streamlit/config.toml dark theme
```

The interpretation engine is the same module the notebooks use; `data_access.py` puts `src/` on the import path. It imports only the Python standard library.

## 5 · Acceptance checks you can run

- Data layer: notebook `09e` runs `validate_mart` (version, columns, row counts and fingerprints, every displayed athlete in the profile table, every case a ranked twin pair with three views, no grade column, abstentions without intervals).
- Logic: `tests/test_advanced_movement_intelligence.py` (or `iv.run_all()` in a notebook cell).
- Application: open each page; pick an athlete on Prospect Profile and check it stays selected on Stopwatch Twins and Movement Match Lab; toggle the twin filters; download an HTML report; prepare a PDF card.

## 6 · Reading the numbers

- **Distinguishable** = the 95 % interval of the difference (repeat-attempt variation of both athletes) excludes zero. About 5 % of twin pairs would be called distinguishable on a feature by chance alone.
- **Coverage** = the share of the position's NFL episodes of a demand at or below the prospect's Combine value. A maximal-effort test placed inside observed, usually submaximal game movement. It is not a probability of success.
- **Evidence badges** come from the 08B.4 registry (residual repeatability after official tests) or the 09B local screen (ICC and in-sample R²), as labelled.
