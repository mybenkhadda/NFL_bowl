# Reproducibility record — Milestone 10

_Generated 2026-10-14T14:25:56Z · run mode full · seed 20261011._

## Order of execution (Jupyter, same RUN_MODE)

1. Milestones 00–08B (executed; frozen conclusions in `docs/RESEARCH_DECISIONS.md`).
2. `09a` → `09g` (application mart `warehouse/gold/advanced_movement/intelligence_v1/app_mart/v1`).
3. `10a_application_readiness` → `10b_signature_football_stories` → `10c_nfl_movement_gap` → `10d_decision_simulator_demonstration` → `10e_final_submission_builder`.
4. Optional, from a terminal: `cd apps/combine_intelligence_lab` then `streamlit run app.py`; walk `reports/milestone10/APP_BROWSER_CHECKLIST.md`.

## Inputs

- Mart version v1, built 2026-10-14T13:13:05Z (24 files; fingerprints in `mart_manifest.json`).
- 10B finished 2026-10-14T14:23:23Z · 10C finished 2026-10-14T14:24:12Z · 10D finished 2026-10-14T14:25:10Z.

## Engine versions

- product_kit 10.1 · submission_builder 10.1 · scouting_interpretation 1

## Checks at build time

- Milestone 10 synthetic checks: 11 / 11 PASS
- Document tokens: 35 filled / 35 declared
- Typed Milestone 09 anchors matching the mart: 8 / 8

## What is NOT reproduced here

- No warehouse rebuild, no model training, no new statistical test: Milestone 10 reads persisted artifacts only.
