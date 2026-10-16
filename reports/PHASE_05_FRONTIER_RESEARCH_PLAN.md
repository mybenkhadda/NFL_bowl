# Phase 05 · Frontier Research Plan

*Authored 2026-10-11. Code-authoring only — **nothing executed, no result claimed**. Companion files:
`PHASE_05_DECISION_PROBLEM_AUDIT.md` (state/action/reward inventory, horizons), `PHASE_05_NOVELTY_MATRIX.md`,
`PHASE_05_EXPERIMENT_REGISTRY.md`, `PHASE_05_METRIC_REGISTRY.md`, `PHASE_05_RESEARCH_RESULTS.md`.*

**Central idea:** Combine-conditioned player *capability* model → realized NFL movement behaviour →
decision intelligence (what an athlete can do vs. what he does, by situation). Every innovation is a hypothesis.

---

## 1 · Feasibility of the frontier streams

The decisive constraint (audit §1): NFL tracking is **single-cohort, ≈ 3.6 tracked rookies per play, no ball**
[E3, E4 sample; exact census = P0]. Feasibility is rated against that.

| Stream | What it needs | What we have | Verdict (pre-P0) | Re-rated by |
|---|---|---|---|---|
| **A · Capability digital twin** (Combine → athlete movement envelope) | Reliable Combine capacities; reliable per-athlete NFL envelope; athlete-level n | 497 athletes with both; Phase 04 reliability machinery authored | **FEASIBLE — primary** (gated on Phase 04 M1/M2) | P0 exposure census + Phase 04 V/ICC |
| **B · Counterfactual movement engine** | A behaviour model of short-horizon motion conditioned on state + capacity | Own-trajectory state only; context via vendor fields | **LIMITED** — single-agent *kinematic feasibility* counterfactuals ("reachable set under capacity X") only. No interaction counterfactuals (no opponents/ball) → **no causal claims** | Stage S5 calibration gate |
| **C · Offline RL** | Observed actions with overlap, a reward attributable to the actor, a validated behaviour model, OPE with adequate effective sample size | Actions = continuous movement (fully observed) or vendor labels (route/coverage); reward = team EPA (credit-assignment) or vendor individual outcomes; no opponents | **NOT IDENTIFIABLE as policy optimisation.** At most: a conservative contextual-bandit *evaluation* over vendor action labels, only if P0 shows overlap and S5 passes. Default classification: **exploratory simulator, not a recommendation engine** | P0 action overlap + S5 + OPE ESS gate |
| **D · Multi-agent** | Several agents tracked jointly on the same play | Only co-tracked rookies; opponents rarely both tracked [H → P0] | **LIMITED** to rookie-vs-rookie matchups (e.g. WR × DB) if the P0 co-tracking census yields enough plays and distinct pairs; never a 22-player model | P0 co-tracking matrix |
| **E · Metric registry** | Definitions + validation | — | **FEASIBLE** (`PHASE_05_METRIC_REGISTRY.md`) | each metric's validation stage |

## 2 · Comparison against conventional methods

| Question | Conventional method (baseline that must be beaten) | Frontier proposal | What would make the frontier version *worth it* |
|---|---|---|---|
| Does Combine predict NFL movement? | Official Combine results (40, 10-yd split, shuttle, 3-cone) → linear regression on NFL summary stats | Tracking-derived capacity profile → *realized envelope* (speed × acceleration quantile frontier) | Grouped-CV skill gain over the official-result baseline with a CI excluding 0 (M10) |
| How good is a rookie physically in games? | Season max speed / average speed | Context-stratified envelope + utilization (M01, M04) | Higher test–retest reliability per play count than max speed (Phase 04 M2 curves) |
| What if he were faster / quicker? | Scenario arithmetic on a single stat | Reachable-set counterfactual under a swapped envelope (B) | Held-out calibration of the motion model; only feasibility statements |
| Which choices are better? | Descriptive splits (EPA by route, by coverage) | Offline policy evaluation | Only if overlap + ESS gates pass; otherwise report the descriptive split and say why OPE is not identifiable |
| Matchups | Vendor separation / cushion by opponent | Joint rookie–rookie trajectory analysis | Enough co-tracked plays and pairs (P0) |

## 3 · Proposed metrics (full 13-field definitions in the metric registry)

M01 realized movement envelope · M02 Combine-predicted envelope · M03 transfer gap · M04 envelope
utilization · M05 get-off (vendor) · M06 time-to-pressure (vendor) · M07 separation at pass forward (vendor) ·
M08 envelope reliability vs plays · M09 counterfactual reachable-set area · M10 grouped-CV predictive skill vs
baseline · M11 motion-model interval calibration · M12 OPE effective sample size · M13 co-tracking coverage.

## 4 · Recommended primary research question

> **RQ-P (Capability transfer):** Among drafted rookies with both Combine and NFL tracking, does a
> tracking-derived Combine capacity profile predict the athlete's **realized, context-matched NFL movement
> envelope** (upper-quantile acceleration/deceleration at given speed in the first 0–1.5 s and 0–3 s after the
> snap) better than official Combine results alone, under **athlete-grouped** out-of-sample evaluation — and
> is the remaining gap (**capacity utilization**, M04) associated with position-specific vendor outcomes
> (DL `player_get_off` / `time_to_pressure`; WR `separation_at_pass_forward`; DB `cushion`)?

Why this question: it is (i) **actionable** (pregame scouting, development targets: horizons A + D);
(ii) **identifiable** with single-agent tracking (no opponents/ball needed for the envelope);
(iii) **directly tests the required Combine→NFL relationship**; (iv) it is the foundation every frontier
stream needs (B, C, D all condition on the envelope). Outcome associations are **associational**, reported with
athlete-cluster bootstrap CIs; no causal language.

Secondary questions (only after RQ-P gates): RQ-B single-agent counterfactual reachability; RQ-D WR × DB
co-tracked matchups; RQ-C bandit-level OPE feasibility.

## 5 · Staged experiment plan with gates

| Stage | Content | Inputs | Gate to pass before the next stage |
|---|---|---|---|
| **P0** · Data & decision feasibility audit *(authored this session)* | Entity/ball census, tracked-players-per-play, opponent co-tracking matrix, vendor field coverage by position, athlete exposure, action vocabulary & overlap, rule-based stream feasibility matrix | silver tracking + `player_play` + `dim_athlete` (**no Combine features**) | G0: P0 outputs exist + user review; stream verdicts re-rated |
| **(Phase 04 M1/M2)** | V status registry; ICC + NFL reliability vs plays | Phase 04 | **G-P04**: inherited gate — no Combine→NFL stats before it passes |
| **S2** · Realized NFL envelope (outcome definition) | Per-athlete speed-binned upper-quantile accel/decel from snap windows; reliability vs n plays; context strata (down/distance, motion, alignment) | `nfl_player_play_windows` + frame kinematics | G2: envelope split-half reliability ≥ threshold (Phase 04 ICC convention, CI lower ≥ 0.75 "good") at the median exposure; else coarsen |
| **S3** · Capability twin v0 (RQ-P) | Ridge / hierarchical-by-position models: (a) position mean, (b) official Combine results, (c) + tracking capacities; athlete-grouped K-fold, nested tuning | VALIDATED Combine features only (V registry) | G3: (c) − (b) skill gain CI lower bound > 0 on the held-out folds (M10). If not → report a well-powered null; the submission still ships the envelope + null |
| **S4** · Utilization ↔ outcomes | M04 vs M05–M07 by position, cluster bootstrap, BH correction | S2–S3 | G4: pre-registered direction + CI; association only |
| **S5** · Counterfactual movement engine | Short-horizon (≤ 1.5 s) single-agent motion model conditioned on state + envelope; reachable-set counterfactual | S2–S3 | G5: held-out-athlete prediction-interval coverage within ±5 pp of nominal (M11) |
| **S6** · Offline RL / OPE feasibility | Contextual-bandit framing over vendor action labels (route / coverage), behaviour policy estimate, overlap diagnostics, ESS | P0 action overlap, S5 | G6: ESS and overlap above pre-registered minimum; otherwise publish the non-identifiability result and stop |
| **S7** · Multi-agent matchups | WR × DB (or OL × DL) co-tracked plays: relative kinematics, separation vs envelopes | P0 co-tracking matrix | G7: ≥ pre-registered plays and distinct pairs (P0 config) |
| **S8** · Dashboard prototype | Use cases A (pregame card), B (historical replay overlay), C (postgame utilization) in the Analytics Lab Streamlit | S2–S4 tables | G8: every panel reads persisted tables only; no invented values |

**Leakage rules (all stages):** splits by `nfl_id` (athlete-grouped), never by play; Combine features fit on
training athletes only (scalers, imputation, feature selection inside the fold); cohort as a stratification
and a sensitivity facet (2025 noise shift); `career_*` never a predictor; NFL outcomes never used for Combine
feature selection.

## 6 · Minimal code architecture

```
src/nflbdb/research/phase05/
  __init__.py
  config.py            Phase05Config (+ per-stage dataclasses) · DOCS provenance · threshold_table()
  decision_audit.py    Stage P0 — DecisionAuditStudy (SQL census → contracts → StudyStore(phase="phase_05"))
  audit_figures.py     P0 figures (pure plotting from persisted tables; redraw with --figures-only)
  __main__.py          CLI: python -m nflbdb.research.phase05 [--only P0] [--check] [--figures-only]
  (later) envelope.py S2 · twin.py S3 · outcomes.py S4 · counterfactual.py S5 · ope.py S6 · matchups.py S7
tests/test_phase05_p0_audit.py    pure-function tests on synthetic frames (no warehouse access)
```

Reused: `StudyStore` (CSV + Parquet tables, datasets with `.schema.json` reload instructions, figure index,
run fingerprint cache), `TableContract`/`conform`, `figkit`, `studies/stats.py`. Models (S3+) will persist under
`artifacts/models/phase_05/<model_id>/` with `config.json`, `provenance.json` (data fingerprints, fold
assignment, git state) and `RELOAD.md`.

Output roots: `reports/tables/phase_05/`, `reports/figures/phase_05/` (PNG 300 DPI + SVG, + PDF for
publication figures), `warehouse/gold/phase_05/`.

## 7 · Manual execution instructions (for the user)

```powershell
conda activate nfl
$env:NFL_BDB_DUCKDB_MEMORY = "3GB"
cd C:\Users\myben\Downloads\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027

python -m pytest tests/test_phase05_p0_audit.py -q          # 1 · pure tests (no data)
python -m nflbdb.research.phase05 --only P0                 # 2 · P0 audit (DuckDB aggregates over silver)
python -m nflbdb.research.phase05 --check                   # 3 · lists any missing output
```

Then send back (or let me read): `reports/tables/phase_05/descriptive/P0_*.csv`,
`reports/tables/phase_05/descriptive/P0_stream_feasibility.csv`, and the figures in
`reports/figures/phase_05/P0_feasibility/`.
Optional smoke run first: `python -m nflbdb.research.phase05 --only P0 --seasons 2023` writes to the
`phase_05/_sample` namespace, so it can never replace the full-run outputs.

## 8 · Session checkpoint

See the top section of `docs/SESSION_STATE.md` (Phase 05).
