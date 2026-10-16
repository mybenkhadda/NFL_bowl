# Milestone 10 — final product and competition submission

**Status: AUTHORED 2026-10-14, NOT EXECUTED.** Nothing here was run by the author. Documents carry `‹M10:…›` tokens until notebook 10E renders them.

## Run order (Jupyter, same `RUN_MODE`, after 09A–09G)

| Notebook (`notebooks/advanced_movement/`) | Produces |
|---|---|
| `10a_application_readiness` | mart re-validation, 11 Milestone 10 logic checks, app column contract, no-data census |
| `10b_signature_football_stories` | three stories (`stories/`), hero figures `AM10B_*`, comparison cards (`cards/`) |
| `10c_nfl_movement_gap` | coverage matrix, gap by position, test proposals, `MOVEMENT_GAP_EXECUTIVE_SUMMARY.md`, `AM10C_*` |
| `10d_decision_simulator_demonstration` | five scenarios, scout / coach / GM reports (`decision_reports/`), `AM10D_01` |
| `10e_final_submission_builder` | final figures `AM10E_01…10`, renders the documents below, `FIGURE_INDEX.md`, `REPRODUCIBILITY.md`, `tables/` |

Then launch the app (terminal: `cd apps/combine_intelligence_lab` → `streamlit run app.py`) and walk `APP_BROWSER_CHECKLIST.md`.

## Documents

| File | Audience |
|---|---|
| `NFL_BDB_2027_FINAL_SUBMISSION.md` | judges — the full narrative |
| `EXECUTIVE_SCOUTING_BRIEF.md` | GM / scouting director (two pages) |
| `TECHNICAL_APPENDIX.md` | researchers |
| `SUBMISSION_WALKTHROUGH.md` | presenter |
| `DEMO_VIDEO_STORYBOARD.md` | video recording (110 s) |
| `APP_BROWSER_CHECKLIST.md` | manual application test |

## Code

`src/nflbdb/research/advanced_movement/`: `product_kit.py` (phases, evidence labels, replay sync, no-data states) · `movement_gap.py` · `decision_simulator.py` · `signature_stories.py` (standard library only, shared with the app) · `submission_builder.py` · `final_figures.py` · `milestone10_validation.py` · `artifacts.ProductStore` (area `product_v1`). App pages added: `8_signature_stories`, `9_movement_gap`, `10_decision_simulator`; all others refined. Tests: `tests/test_milestone10_product.py`, `tests/test_cil_app_wiring.py`.

No warehouse scan, no model, no new mart table: Milestone 10 reads the 09E mart only.
