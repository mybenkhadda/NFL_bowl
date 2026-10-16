# 🏈 NFL Big Data Bowl 2027 — Project Tracker

> **Purpose:** A living, evidence-driven roadmap for a competition-quality football analytics submission.  
> **Research theme:** Do pre-NFL Combine movement signatures provide meaningful information about NFL game movement and performance beyond traditional Combine measurements?  
> **Status:** 🟡 Not started · **Start:** 2026-10-07 · **Target submission:** `TBD — verify on official Kaggle rules`  
> **Track:** `[ ] Open` `[ ] Undergraduate / Graduate` `[ ] To confirm`  
> **Owner:** `@your-handle` · **Last updated:** `2026-10-08`

[🏠 Dashboard](#-dashboard) · [🗓️ Weekly roadmap](#️-weekly-roadmap) · [🧪 Hypotheses](#-research-hypotheses) · [🔬 Experiments](#-experiment-register) · [📈 Visualizations](#-visualization-tracker) · [🚦 Gates](#-stage-gates) · [⚠️ Risks](#️-risk-register) · [🏆 Achievements](#-achievement-log) · [📦 Submission](#-submission-checklist)

> [!IMPORTANT]
> The plan contains **proposed research hypotheses**, not established competition findings. Verify the official 2027 submission deadline, permissible external data, data schema, evaluation criteria and deliverable format before committing to the final methodology. The 12-week schedule is a planning template, not a verified Kaggle deadline.

## 📊 Dashboard

### At-a-glance progress

Update these fields at the end of each work session. **GitHub Markdown checkboxes can be checked in the source file, issues, or supported Markdown editors.** Tables and percentages are manually maintained; vanilla Markdown does not automatically count checked tasks.

| Item | Current value | Update rule |
|---|---|---|
| Overall status | 🔵 In progress | 🟡 Not started / 🔵 In progress / 🟢 Complete / 🔴 Blocked |
| Current phase | Week 1 — Phase 01 data foundation (code written, not yet executed) | Set to active week |
| Weeks completed | **0 / 12** | Count fully accepted weekly deliverables |
| Milestone gates passed | **0 / 6** | See [stage gates](#-stage-gates) |
| Hypotheses shortlisted | **0 / 3** | See [hypotheses](#-research-hypotheses) |
| Experiments completed | **0** | See [experiment register](#-experiment-register) |
| Validated findings | **0** | Only independently checked findings count |
| Figures publication-ready | **0 / 5** | See [visualizations](#-visualization-tracker) |
| Latest evidence | [`src/nflbdb/data/`](src/nflbdb/data/), [`notebooks/01_data_discovery_and_warehouse.ipynb`](notebooks/01_data_discovery_and_warehouse.ipynb) (not executed) | Link to concrete artifact |
| Next action | Review 03/04/08 reports; fill in measured numbers in 05/06; start Phase 02 | One actionable task |
| Main blocker | `None recorded` | Explicitly record missing data or dependencies |

### Milestone overview

| Week | Focus | Status | Acceptance evidence |
|---|---|---|---|
| 1 | Rules + dataset audit | ⬜ Not started | [`reports/data_audit.md`](reports/data_audit.md) |
| 2 | Combine EDA | ⬜ Not started | [`notebooks/02_combine_eda.ipynb`](notebooks/02_combine_eda.ipynb) |
| 3 | NFL EDA + hypotheses | ⬜ Not started | [`reports/research_questions.md`](reports/research_questions.md) |
| 4 | Kinematics engineering | ⬜ Not started | [`notebooks/04_kinematic_features.ipynb`](notebooks/04_kinematic_features.ipynb) |
| 5 | Reliability + context | ⬜ Not started | `reports/feature_quality.md` |
| 6 | Outcomes + baselines | ⬜ Not started | `reports/baselines.md` |
| 7 | Statistical experiments | ⬜ Not started | `experiments/results.csv` |
| 8 | Cross-validation + ablations | ⬜ Not started | `reports/validation.md` |
| 9 | Novel metric + case studies | ⬜ Not started | `reports/scouting_metric.md` |
| 10 | Robustness checks | ⬜ Not started | `reports/robustness.md` |
| 11 | Storytelling + report | ⬜ Not started | `reports/final_writeup.md` |
| 12 | QA + final submission | ⬜ Not started | `reports/submission_receipt.md` |

**Status key:** ⬜ Not started · 🔵 In progress · 🟢 Accepted · 🔴 Blocked. A week counts as complete **only** when its acceptance criteria are met and evidence exists.

### Phase 01 — Data foundation (Polars + DuckDB + Parquet lakehouse)

Full pipeline executed end-to-end on 2026-10-08 (discovery mode; conda env `nfl`: Polars 2.0.0, DuckDB 1.5.6, PyArrow 25.0.1). An independent verification script checked row conservation from raw → bronze → silver, value preservation, PK uniqueness and benchmark result hashes: all PASS. "DONE" means the artifact was produced and verified. The findings still need a football-literate review.

| # | Task | Status | Evidence / artifact | Key findings | Problems | Next action |
|---|---|---|---|---|---|---|
| 0 | Environment report | DONE | `reports/00_environment_report.md` | 16 logical cores, 15.4 GB RAM; only ~2 GB free during runs | DuckDB default (50% RAM) too high for this machine | Run with `NFL_BDB_DUCKDB_MEMORY=3GB` |
| 1 | Raw inventory | DONE | `reports/01_data_inventory.md`, `tables/file_inventory.*` | 9 CSVs, 2.1 GB, 23.4M rows | — | — |
| 2 | Schema dictionary + type resolution | DONE | `reports/02_schema_dictionary.md`, `tables/schema_catalog.*`, `tables/schema_resolution.*` | `recovered_fumble` is declared numeric but holds only TRUE/FALSE (2,214 values) → String in bronze, BOOLEAN in silver. Every other declared type verified on a full scan. `play_description` contains quoted newlines | Units inferred, not confirmed | Check against the Kaggle data page |
| 3 | Bronze ingestion | DONE | `warehouse/bronze/`, `catalog/bronze_manifest.*`, `catalog/resolved_schemas.json` | Lossless: bronze rows = raw rows for all 9 files; numeric non-null counts match raw; 2.1 GB CSV → ~470 MB Parquet | — | — |
| 4 | Profiling + tracking QC + validation | DONE | `reports/03_data_quality.md`, `tables/data_quality_results.*` | 0 CRITICAL failures. Combine grain needs `entity_type` (BALL rows carry the athlete's nfl_id). 150 tracked plays and 1,879 tracked player-plays are missing from player_play. Combine `s` max is 25.6 yd/s (outliers) | Tracking key checks in discovery mode were sampled (silver verified exactly) | Decide on an outlier policy for Combine speeds |
| 5 | Relationships + athlete linkage | DONE | `reports/04_relationship_analysis.md`, `diagrams/source_relationships.mmd` (measured) | 510 athletes; exact nfl_id linkage with 99.2–100% match rates | — | Review the coverage-by-position figure |
| 6 | Architecture decision | IN PROGRESS | [`reports/05_architecture_decision.md`](reports/05_architecture_decision.md) | Option C now supported by measurements (see §9) | "Pending" cells not yet filled in | Copy measured numbers into 05 |
| 7 | Warehouse design | IN PROGRESS | [`reports/06_warehouse_design.md`](reports/06_warehouse_design.md), `tables/gold_manifest.*`, `tables/table_catalog.*` | 14 gold tables, all PKs unique | Cardinalities not yet copied into 06 | Fill in from gold_manifest |
| 8 | Silver / gold / catalog | DONE | `warehouse/silver`, `warehouse/gold`, `catalog/bdb27.duckdb` | 22.65M silver tracking rows; key unique; 0 off-grid frames; 114,954 plays | 15,878 tracking rows lack play_direction (plays missing from player_play) | — |
| 9 | Benchmarks | DONE | `reports/07_performance_benchmarks.md`, `figures/07_benchmarks.png` | Results identical across engines. One season from all seasons: CSV 1.95 s / 1.6 GB vs partition-pruned Parquet 0.03 s / 12 MB | OS page cache not cleared | — |
| 10 | Research feasibility | DONE (first pass) | `reports/08_research_feasibility.md` | Counts computed from gold | Drill-name regexes still need a review against `dim_combine_drill` (62 drills) | Review the regexes |
| T | Tests | DONE | `tests/` — 27 passing (incl. regressions for DuckDB→Polars conversion, mixed-type CSV columns, nulls, profiling, multi-line CSV sampling) | — | — | — |

### Phase 02 — Research-ready data, quality remediation, movement EDA

Checkpoint: [`docs/SESSION_STATE.md`](docs/SESSION_STATE.md). Nothing below is checked off until the user has executed and reviewed it.

| # | Task | Status | Evidence / artifact | Key findings | Problems | Next action |
|---|---|---|---|---|---|---|
| 1 | Quality remediation + artifact utilities | IN PROGRESS (code authored, not executed) | `src/nflbdb/research/{artifacts,quality}.py`, `tests/test_phase02_quality.py`, `reports/09_quality_remediation.md` (pre-execution review) | The saved CRITICAL failure comes from a stale-kernel run with the old grain definition | D1–D4 not yet run | User runs pytest, then `validate`, then `python -m nflbdb.research.quality` |
| 2 | Coverage marts | TODO | — | — | — | Author `research/coverage.py` |
| 3 | Movement EDA notebook | TODO | — | — | — | — |
| 4 | Outcome audit | TODO | — | — | — | — |
| 5 | Hypothesis feasibility v2 | TODO | — | The H02/H03 drill regexes match unintended drills | — | Use explicit drill lists |
| 6 | Research protocol | TODO | — | — | Needs executed evidence | — |

### Phase 03 — Advanced analytics, feature engineering & discovery

Checkpoint: [`docs/SESSION_STATE.md`](docs/SESSION_STATE.md). Design record: [`reports/14_feature_engineering.md`](reports/14_feature_engineering.md). Nothing is checked off until the user has executed and reviewed it.

| # | Milestone | Status | Evidence / artifact | Key findings | Problems | Next action |
|---|---|---|---|---|---|---|
| M1 | Kinematic foundation (segmentation, SG kinematics, events, quality tiers, registry, cache) | EXECUTED · reviewed 2026-10-08 (pytest run not recorded) | `src/nflbdb/features/`, `kinematic_{build,diagnostics}.py`, notebook 05 (with outputs), `tables/phase_03/*.csv`, `figures/phase_03/m1_*` | Vendor speed r = 0.9996 / 0.998; vendor `dir` = compass convention; vendor `a` = magnitude. Combine 6,309/6,310 tier A; NFL ≥ 98.9 % A. Shuttle 1.93 reversals per attempt (design 2). Forty recording ends at peak speed. 2025 Combine residual noise ≈ ¼ of 2023/24. Details: report 14 §6 | Strict turn rule non-informative (documented, not changed). Physical handedness unverified | Thresholds kept (v1); see report 14 calibration decision |
| M2 | Combine movement analytics + attempt reliability | IN PROGRESS (code authored, not executed) | `features/{temporal,acceleration,braking,agility,efficiency,sprint,reliability}.py`, `research/combine_{movement,analytics}.py`, `tests/test_phase03_m2_combine.py`, notebooks 03 + 06, reports 13 / 15 | — (no executed results) | Official vs tracked timing conventions unvalidated; Polars 2.0 APIs unverified | User: pytest → `combine_movement` → `combine_analytics` → return CSVs |
| M3 | NFL player-play / athlete-season features with play context | TODO | — | — | Needs M1 + Phase 02 flags | — |
| M4 | Combine ↔ NFL feature alignment | TODO | — | — | — | — |
| M5 | Statistical discovery | TODO | — | — | — | — |
| M6 | Research reporting + dashboard panels | TODO | — | — | — | — |

### Phase 04 — Scientific validation → athlete feature store → discovery → candidate selection

Plan and gates: [`docs/PHASE_04_PLAN.md`](docs/PHASE_04_PLAN.md). Code lives in `src/nflbdb/research/phase04/`. Outputs go to `{reports/tables,reports/figures,warehouse/gold}/phase_04/`. "Code" is never "validated": the validation column changes only after the user has executed and reviewed the outputs.

| # | Milestone / study | Code | Execution | Validation | Main findings | Problems | Artifacts | Next action |
|---|---|---|---|---|---|---|---|---|
| M1·Q | Structural vs signal quality, athlete eligibility | CODE AUTHORED | AWAITING USER EXECUTION | NOT STARTED | — (A3, executed: tiers AUC ≈ 0.50; continuous metrics discriminate) | — | `Q_*` tables, figures Q_01–04 | `--only Q` |
| M1·B | Smoothing robustness (traj / athlete, BA, top-decile, ICC per setting) | CODE AUTHORED | AWAITING USER EXECUTION | NOT STARTED | — | Heavy step (4 SG × 0.46M Combine frames + NFL sample) | `trajectory_features_by_smoothing`, `events_by_smoothing`, `B_*`, B_01–09 | smoke run, then full run |
| M1·C1 | Event-detector sensitivity (rule × SG, per-second rates, event matching) | CODE AUTHORED | AWAITING USER EXECUTION (after B) | NOT STARTED | — | — | `C1_*`, C1_01–05 | `--only C1` |
| M1·C2 | Reference annotation (Combine + NFL, dev/eval) and detector accuracy | CODE AUTHORED | AWAITING USER EXECUTION + manual labels | BLOCKED (labels) | — | Labelling effort ≈ 100 tasks | `C2_*`, protocol doc | `--annotation-tasks`, then label |
| M1·V | Feature status registry (Milestone 1 gate) | CODE AUTHORED | AWAITING USER EXECUTION | NOT STARTED | — | Cut-offs are proposed | `V_*`, V_01–02 | `--only V` after Q/B/C1 |
| M2 | Athlete feature store (maneuver → research base) | NOT STARTED | — | — | — | Gated on M1 | — | after M1 review |
| M3–M6 | Combine / NFL analytics, transfer, discovery registry | NOT STARTED | — | — | — | Gated on M1 + M2 | — | — |

### Phase 05 — Frontier research: capability model → counterfactuals → offline-RL feasibility → decision metrics

Plan and gates: [`reports/PHASE_05_FRONTIER_RESEARCH_PLAN.md`](reports/PHASE_05_FRONTIER_RESEARCH_PLAN.md). Code lives in `src/nflbdb/research/phase05/`. Outputs go to `{reports/tables,reports/figures,warehouse/gold}/phase_05/`. Every innovation is a hypothesis; no Combine→NFL statistic runs before the Phase 04 gates.

| # | Stage | Code | Execution | Validation | Main findings | Problems | Artifacts | Next action |
|---|---|---|---|---|---|---|---|---|
| Audit | Feasibility + novelty audit (6 reports) | WRITTEN | n/a (documents) | NOT REVIEWED | Pre-P0 hypotheses only: twin = primary, RL not identifiable as policy optimisation | Literature search not done (plan only) | `reports/PHASE_05_*.md` | user review |
| P0 | Data & decision feasibility census (E05-00) | CODE AUTHORED | EXECUTED BY USER | REVIEWED 2026-10-11 | No ball tracked; median 2 athletes per play; both sides on 58.9 % of plays; 401 athletes, 303 with ≥ 100 plays; dense outcomes: DL get-off, WR separation, OL pressure | Screening rules for streams C, D, E were misleading (see review §7) | `P0_*` tables, `athlete_exposure`, figures P0_01–05, `reports/PHASE_05_{P0_RESULTS_REVIEW,UPDATED_FEASIBILITY,RESEARCH_DECISION}.md` | — |
| P0b | Analysis sample frame (E05-00b) | CODE AUTHORED | AWAITING USER EXECUTION | NOT STARTED | — | — | `P0b_*` tables, `athlete_outcome_counts`, figures P0b_01–02 | `python -m nflbdb.research.phase05 --only P0B` |
| S2 | Realized NFL envelope + outcome reliability (E05-01, primary) | NOT STARTED | — | — | — | Config waits for P0b | — | author after P0b tables |
| S3–S4 | Capability transfer (E05-02), DL get-off backup (E05-B1), utilization ↔ outcomes | NOT STARTED | — | — | — | BLOCKED on Phase 04 M1 + M2 (Q and B runs incomplete) | — | fix Phase 04 Q / B |
| S5–S8 | Counterfactual engine, OPE feasibility, matchups, dashboard | NOT STARTED | — | — | — | Gated on S2–S4 and P0 verdicts | — | — |

### Side track — Analytics Lab (Docker · Superset · Streamlit · Grafana)

Exploration and decision-support tool, not a submission. Docs: [`infrastructure/analytics_lab/README.md`](infrastructure/analytics_lab/README.md).

| # | Task | Status | Evidence / artifact | Next action |
|---|---|---|---|---|
| L1 | Architecture + connector decision | DONE (written) | `infrastructure/analytics_lab/docs/architecture.md` | Review |
| L2 | Host marts and serving snapshots | IN PROGRESS (code authored, not executed) | `src/nflbdb/lab/`, `tests/test_analytics_lab.py` | User runs the tests, then `python -m nflbdb.lab.build` |
| L3 | Docker stack | IN PROGRESS (not executed) | `infrastructure/analytics_lab/compose.yaml` | `docker compose build` and `up -d` |
| L4 | Superset dashboards (overview, Combine, NFL, quality, research) | IN PROGRESS (not executed) | `superset/bootstrap/dashboards_spec.py` | Check that the charts render; export with `scripts/export_dashboards.py` |
| L5 | Streamlit trajectory explorer | IN PROGRESS (not executed) | `infrastructure/analytics_lab/streamlit/` | Play back one attempt and one play |
| L6 | Grafana engineering monitor | IN PROGRESS (not executed) | `grafana/dashboards/engineering.json` | `--profile monitoring up -d` |

**Architecture decision log (short)**

| Date | Decision | Why |
|---|---|---|
| 2026-10-08 | Bronze/Silver/Gold Parquet + DuckDB catalog + Polars (proposed) | The data has two frame-level domains plus repeated athlete-level queries, so compact gold tables avoid rescanning frames. Runs locally with no services. |
| 2026-10-08 | Partition only by season / draft_year, then sort by key inside each partition | Avoids lots of tiny files. Row-group statistics give pruning for single-play lookups. |
| 2026-10-08 | Link athletes on exact nfl_id only | Matching on names risks silent false links. |
| 2026-10-08 | Derive frame_id from time, aligned to the start of the play | The raw game tracking has no frame column. |
| 2026-10-08 | Split player_play into plays and player_play using a measured constancy test | Removes denormalisation based on evidence, not on assumptions. |
| 2026-10-08 | Read raw CSVs as strings, resolve types from evidence, cast strictly (`csv_schema.py`) | Inference/forced dtypes crashed on `recovered_fumble` = FALSE. Strict casts surface unexpected values instead of silently nulling them. |
| 2026-10-08 | Analytics Lab serves a materialised DuckDB snapshot, not views, and PostgreSQL holds Superset metadata only | Catalog views embed Windows paths, so they can't be used in containers. The marts are tiny, so a self-contained, read-only file is the simplest option. No tracking frames are copied to PostgreSQL. |
| 2026-10-08 | Route every DuckDB→Polars conversion through `frames.py` | In Polars 2.0, `from_arrow(<Arrow stream>)` returns a struct Series. DuckDB HUGEINT/DECIMAL results arrive as Decimal. |
| 2026-10-09 | NFL movement features only inside snap-anchored windows (pre-snap kept separate), on standardised coordinates | Whole-play summaries mix pre-snap motion into "peaks"; direction-dependent features need offense → +x |
| 2026-10-09 | Re-measure every failed Phase 01 check exactly before excluding data | Phase 01 sampled tracking at 2 %; the CRITICAL Combine key failure is likely a grain-rule error (BALL rows) |

### Work allocation targets (guidance, not judging weights)

| Priority | Suggested effort |
|---|---:|
| Data understanding + football context | 25% |
| Research-question selection | 20% |
| Statistical rigor and validation | 20% |
| Kinematic features | 15% |
| Visual storytelling | 15% |
| Advanced modeling | 5% |

---

## 🗓️ Weekly roadmap

Each week includes tasks, **definition of done**, and the required proof. Check tasks only when actually completed, and never mark a gate as passed without linked evidence.

<details open>
<summary><strong>Week 1 — Rules, data inventory and feasibility</strong></summary>

**Outcome:** Know exactly what the dataset contains, who is represented, and which research questions are measurable.

- [ ] Read [2027 overview](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/overview), [rules](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/rules), and [data](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data).
- [ ] Confirm submission deadline, tracks, team constraints, evaluation rubric, permitted outside data, and required format.
- [ ] Inventory all files, row counts, formats, storage sizes, and join keys.
- [ ] Produce a column-level data dictionary including units, coordinate frames, time resolution, nulls.
- [ ] Inspect player identities, position definitions, Combine year, drill IDs, NFL play IDs and frames.
- [ ] Quantify sample sizes by position, Combine year, drill and NFL season.
- [ ] Count athletes with both Combine and NFL tracking; quantify unmatched athletes.
- [ ] Audit duplicate rows, impossible movement, missing frames and bad joins.
- [ ] Document observation windows, sampling bias and limitations.
- [ ] Create a reproducible environment and README with how to run the audit.

**Evidence:** `notebooks/01_data_discovery_and_warehouse.ipynb`, `reports/01_data_inventory.md` … `reports/04_relationship_analysis.md` (see the Phase 01 table above).

**Definition of done:** [ ] Athlete-level linkage is trustworthy; [ ] all primary tables and units are documented; [ ] you can articulate at least two feasible outcomes.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 2 — Combine tracking EDA</strong></summary>

- [ ] Analyze athlete counts per drill, position and cohort.
- [ ] Plot raw and cleaned speed/acceleration distributions.
- [ ] Compare speed-time and acceleration-time profiles on individual drills.
- [ ] Visualize trajectories, heading, turning and braking.
- [ ] Inspect extremes and potential tracking artifacts.
- [ ] Compare repeated drill trials where available; record measurement stability.
- [ ] Identify whether traditional Combine values and measurements are available.
- [ ] Save 6–10 annotated, reproducible diagnostic figures.

**Evidence:** `notebooks/02_combine_eda.ipynb`, `figures/combine/`.

**Definition of done:** [ ] Feature ideas are grounded in available drills; [ ] known measurement artifacts are recorded; [ ] position comparisons are meaningful.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 3 — NFL tracking EDA and hypothesis selection</strong></summary>

- [ ] Characterize NFL tracking by athlete, season, play, position and available context.
- [ ] Identify route/drill phases or game situations that can reasonably be compared.
- [ ] Map offensive/defensive roles, opportunities, exposure and selection bias.
- [ ] Create matched-athlete descriptive summaries without leaking future observations into predictors.
- [ ] Plot preliminary Combine–NFL feature relationships by position.
- [ ] Generate at least three candidate hypotheses and define their outcomes.
- [ ] Score hypotheses by football value, feasibility, novelty, interpretability and visualization potential.
- [ ] Choose one primary and up to two fallback hypotheses.

**Evidence:** `notebooks/03_nfl_eda.ipynb`, `reports/research_questions.md`.

**Definition of done:** [ ] Primary hypothesis states population, predictors, outcome, and comparator; [ ] matched sample supports it; [ ] fallback exists.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 4 — Reliable kinematic features</strong></summary>

- [ ] Implement position-to-velocity conversion using verified sample intervals.
- [ ] Implement vector acceleration and change-of-heading features.
- [ ] Implement robust speed, acceleration, deceleration, jerk and turning summaries where justified.
- [ ] Segment relevant movement phases; avoid mixing incompatible actions.
- [ ] Evaluate smoothing/filter parameters and sensitivity to noise.
- [ ] Test units, timestamps, frame ordering, duplicates and edge cases.
- [ ] Record feature definitions with physical interpretations.
- [ ] Cache efficiently in Parquet; benchmark feature extraction runtime/memory.

**Evidence:** `src/features/`, `tests/`, `notebooks/04_kinematic_features.ipynb`.

**Definition of done:** [ ] Core kinematics pass numerical tests; [ ] features are reproducible; [ ] no impossible values survive unflagged.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 5 — Contextualization, reliability and final features</strong></summary>

- [ ] Compare distributions by position, drill, cohort and game context.
- [ ] Calculate player-level reliable summaries (not naive frame averages only).
- [ ] Estimate within-athlete vs between-athlete variability.
- [ ] Apply training-fold-only normalization and feature selection.
- [ ] Quantify missingness, clipping, imputation and outlier effects.
- [ ] Confirm that features from NFL follow-up are not used to predict that same follow-up outcome.
- [ ] Lock a versioned candidate feature dataset and dictionary.

**Evidence:** `data/processed/`, `reports/feature_quality.md`.

**Definition of done:** [ ] Every retained feature has units, meaning, provenance, and quality tests.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 6 — Primary outcome, baselines and data splits</strong></summary>

- [ ] Formalize an NFL outcome relevant to the chosen position.
- [ ] Define exposure/opportunity adjustments and fixed follow-up horizon when relevant.
- [ ] Specify estimand: physical transfer, NFL football value, or both.
- [ ] Fix grouped-by-athlete train/validation/test partitions.
- [ ] Define cohort-held-out temporal tests if samples allow.
- [ ] Fit B0 (context), B1 (traditional Combine), B2 (B1 + tracking-derived features).
- [ ] Predeclare metrics and minimum meaningful effect; compare B2 versus B1.
- [ ] Inspect residuals, calibration and subgroup failures.

**Evidence:** `notebooks/05_outcomes.ipynb`, `notebooks/06_baselines.ipynb`, `reports/baselines.md`.

**Definition of done:** [ ] Target is well defined; [ ] no athlete leakage; [ ] baselines and comparisons are reproducible.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 7 — Statistical experiments</strong></summary>

- [ ] Fit simple interpretable models (linear, Ridge/Elastic Net; hierarchical models if needed).
- [ ] Add acceleration shape features; compare with traditional Combine baselines.
- [ ] Test braking/agility feature family if supported by data.
- [ ] Evaluate effect sizes and interval estimates, not p-values alone.
- [ ] Check influential athletes and confounding assumptions.
- [ ] Log every model, seed, fold, feature set and output artifact.

**Evidence:** `notebooks/07_ml_experiments.ipynb`, `experiments/results.csv`.

**Definition of done:** [ ] Incremental value assessed with uncertainty; [ ] conclusions not based on a single lucky split.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 8 — Validation, ablations and optional ML</strong></summary>

- [ ] Evaluate strictly on unseen-athlete folds.
- [ ] Evaluate time/cohort holdouts if statistically feasible.
- [ ] Perform feature-family ablations; compare incremental gains.
- [ ] Run permutation/null tests where appropriate.
- [ ] Calculate athlete-cluster bootstrap confidence intervals.
- [ ] Test CatBoost/LightGBM only against established baselines.
- [ ] Reject unstable or non-generalizing claims; record negative results.

**Evidence:** `reports/validation.md`, `experiments/ablation_results.csv`.

**Definition of done:** [ ] A primary finding survives credible splits and sensitivity tests, or an explicit pivot decision is documented.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 9 — New scouting metric and case studies</strong></summary>

- [ ] Define a clear and interpretable metric (optional working name: Combine-to-Game Transfer Score).
- [ ] Explain what the metric adds beyond traditional Combine statistics.
- [ ] Produce out-of-sample athlete comparisons and matched examples.
- [ ] Include an example where the metric is wrong or less informative.
- [ ] Check metric stability by position, context, and cohort.
- [ ] Draft scouting implications as decision support, not causal claims.

**Evidence:** `reports/scouting_metric.md`, `figures/case_studies/`.

**Definition of done:** [ ] A football analyst could explain when the metric is useful and when not to rely on it.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 10 — Robustness and independent checks</strong></summary>

- [ ] Test alternative smoothing and threshold choices.
- [ ] Test alternative outcome definitions and follow-up horizons.
- [ ] Test cohort and position subgroup stability.
- [ ] Inspect unmatched players, survival/selection bias, and opportunity bias.
- [ ] Report clustered bootstrap intervals and model comparison uncertainty.
- [ ] Audit multiple-hypothesis exploration and avoid overstated significance.
- [ ] Rerun final results cleanly from raw inputs.
- [ ] Freeze final result tables and record version/commit hash.

**Evidence:** `reports/robustness.md`, `notebooks/08_final_analysis.ipynb`.

**Definition of done:** [ ] Main claims have no known unresolved validity blocker; [ ] limitations documented honestly.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 11 — Visual storytelling and final report</strong></summary>

- [ ] Create a figure showing same traditional sprint result but different movement signatures.
- [ ] Visualize Combine-to-NFL transfer relationship and uncertainty.
- [ ] Visualize incremental predictive value versus baseline.
- [ ] Show an on-field player/trajectory interpretation.
- [ ] Show robustness by cohort, subgroup or context.
- [ ] Write problem → limitations of current metrics → method → findings → football action → caveats.
- [ ] Ask a football-literate reviewer to challenge interpretations.
- [ ] Ensure charts are accessible, readable and accurately captioned.

**Evidence:** `reports/final_writeup.md`, `figures/final/`.

**Definition of done:** [ ] A nontechnical football evaluator can understand the main insight without opening code.

**Notes / blockers:** `...`

</details>

<details>
<summary><strong>Week 12 — Submission and final quality assurance</strong></summary>

- [ ] Re-read 2027 competition rules and verify all track-specific requirements.
- [ ] Confirm citation, external-data, licensing, originality, and team rules.
- [ ] Check every figure, number, source, and population against frozen results.
- [ ] Run notebooks/pipeline end-to-end with a clean environment.
- [ ] Confirm no sensitive/unlicensed raw data is committed publicly.
- [ ] Proofread the complete submission and verify word/format limits.
- [ ] Submit through Kaggle before its official deadline.
- [ ] Save the submission link, timestamp, and final source commit.

**Evidence:** `reports/submission_receipt.md`.

**Definition of done:** [ ] Official submission confirmation exists; [ ] final report exactly matches reproducible results.

**Notes / blockers:** `...`

</details>

---

## 🚦 Stage gates

**Rule:** A gate is *not* passed just because a week is finished. Record the evidence and decision.

| Gate | Target checkpoint | Go/no-go question | State | Evidence / decision |
|---|---|---|---|---|
| G1 — Feasibility | End Week 1 | Can we match Combine athletes to sufficient NFL observations? | ⬜ | `link` |
| G2 — Hypothesis | End Week 3 | Is there one measurable and football-relevant question? | ⬜ | `link` |
| G3 — Feature integrity | End Week 5 | Are tracking-derived features physically valid and reliable? | ⬜ | `link` |
| G4 — Incremental evidence | End Week 8 | Do features improve on a defensible baseline under valid splits? | ⬜ | `link` |
| G5 — Scientific defensibility | End Week 10 | Does the insight survive robustness checks? | ⬜ | `link` |
| G6 — Submission ready | End Week 12 | Is the narrative accurate, compliant and reproducible? | ⬜ | `link` |

**Pivot rule:** If G1 fails, change the scope or outcome. If G4 fails, do not hide the null result behind more model complexity. If G5 fails, narrow the claim or switch to the strongest supported finding.

---

## 🧪 Research hypotheses

| ID | Hypothesis | Position | Primary outcome | Baseline | Status | Evidence |
|---|---|---|---|---|---|---|
| H01 | Acceleration shape adds information beyond 40-yard dash time | WR | Context-adjusted game acceleration | Traditional Combine | 💡 Candidate | `link` |
| H02 | Braking/reacceleration features relate to relevant game movements | WR/DB | Comparable braking and reacceleration | Context + traditional Combine | 💡 Candidate | `link` |
| H03 | First-step explosiveness relates to in-game rush movement | DL | Initial movement/closing speed in comparable rush contexts | Position-specific baseline | 💡 Candidate | `link` |
| H04 | Tracking features identify overlooked athletic outliers | WR/DB | Validated NFL movement outcome | Traditional scores | 💡 Candidate | `link` |
| H05 | Transfer relationships differ substantially by position | Multiple | Position-specific outcomes | Pooled vs position-aware | 💡 Candidate | `link` |

**For the chosen primary hypothesis, fill this research contract:**

- **Research question:** `...`
- **Population and exclusions:** `...`
- **Combine predictor window:** `...`
- **NFL outcome and observation window:** `...`
- **Comparator / baseline:** `...`
- **Potential confounders:** `...`
- **Selection / survivorship limitations:** `...`
- **Unit of statistical independence:** `Athlete (unless otherwise justified)`
- **Validation protocol:** `...`
- **Minimum practically useful improvement:** `...`
- **What would falsify this hypothesis?** `...`
- **Status:** `candidate / investigating / validated / unsupported / blocked`

---

## 🔬 Experiment register

| ID | Experiment | Question | Metric / baseline | Validation | Status | Result artifact |
|---|---|---|---|---|---|---|
| E00 | Context-only baseline | How much is predictable without Combine data? | `TBD` | Grouped by athlete | ⬜ | `link` |
| E01 | Traditional Combine baseline | How much value do standard tests add? | Δ vs E00 | Grouped by athlete | ⬜ | `link` |
| E02 | Acceleration feature family | What incremental information does acceleration shape add? | Δ vs E01 | Grouped by athlete | ⬜ | `link` |
| E03 | Braking/agility family | Is braking or agility useful? | Δ vs E01/E02 | Grouped by athlete | ⬜ | `link` |
| E04 | Cohort generalization | Does finding transfer across Combine years? | Error + confidence intervals | Temporal cohort holdout | ⬜ | `link` |
| E05 | Subgroup robustness | Does the effect persist by context/position? | Effect sizes | Stratified grouped folds | ⬜ | `link` |
| E06 | Confounder adjustments | Is the effect explained by other athlete attributes? | Coefficient / Δ metric | Pre-specified controls | ⬜ | `link` |
| E07 | Feature ablation | Which feature families actually help? | Δ vs E01 | Identical splits | ⬜ | `link` |
| E08 | Outcome sensitivity | Does result depend on chosen outcome? | Δ across definitions | Identical folds | ⬜ | `link` |
| E09 | Advanced model | Does complexity beat interpretable models? | Δ vs simpler model | Identical folds | ⬜ | `link` |

### New experiment template

Copy this block for each new experiment:

<details>
<summary>📋 Expand experiment template</summary>

```markdown
### EXP-XXX — Descriptive name
- **Date:** YYYY-MM-DD
- **Question:**
- **Hypothesis ID:** H0X
- **Dataset/version:**
- **Population / exclusions:**
- **Predictors / target:**
- **Model / parameters / seed:**
- **Split strategy (athlete/cohort):**
- **Baseline:**
- **Primary metric and uncertainty:**
- **Subgroup / robustness tests:**
- **Result:**
- **Conclusion:** Supported / Inconclusive / Unsupported
- **Decision:** Keep / Improve / Stop
- **Notebook + figure links:**
```

</details>

---

## 📈 Visualization tracker

| Figure | Intended message | Status | Evidence |
|---|---|---|---|
| F01 — Speed curve comparison | Similar traditional times can conceal different signatures | ⬜ | `figures/final/01_speed_profiles.png` |
| F02 — Combine-to-NFL plot | Show estimated transfer with proper uncertainty | ⬜ | `figures/final/02_transfer.png` |
| F03 — Baseline comparison | Quantify added value of tracking features | ⬜ | `figures/final/03_incremental_value.png` |
| F04 — On-field explanation | Connect the metric to a real football movement | ⬜ | `figures/final/04_case_study.png` |
| F05 — Robustness figure | Reveal where the result does or does not hold | ⬜ | `figures/final/05_robustness.png` |

### Figure acceptance checklist

- [ ] Labels, units, sample sizes, and time periods are visible.
- [ ] All values trace to a reproducible calculation.
- [ ] Confidence intervals or uncertainty shown where relevant.
- [ ] Colors and annotations distinguish comparisons clearly.
- [ ] Caption states what is shown without overstating causality.
- [ ] Fonts readable when embedded in the submission.

---

## ⚠️ Risk register

| Risk | Early warning | Mitigation | State |
|---|---|---|---|
| Small matched athlete sample | Very few players with both sources | Narrow position/outcome; avoid overfitting | ⚪ Monitor |
| Survivorship bias | NFL-tracked athletes differ from all Combine athletes | State conditional target population; compare coverage | ⚪ Monitor |
| Outcome confounding | Results heavily affected by scheme/opportunity | Match contexts; use exposure adjustments | ⚪ Monitor |
| Frame leakage | Same athlete appears in train/test | Group by athlete; audit pipelines | ⚪ Monitor |
| Measurement noise | Unrealistic derivative peaks | Quality checks and smoothing sensitivity | ⚪ Monitor |
| Temporal bias | Newer Combine cohorts have shorter follow-up | Fix observation window or model exposure | ⚪ Monitor |
| Weak novel result | Tracking adds little versus baseline | Investigate other defensible outcomes; preserve negative evidence | ⚪ Monitor |
| Time overruns | EDA/modeling expands indefinitely | Enforce gates and freeze dates | ⚪ Monitor |
| Submission mismatch | 2027 rules differ from 2026 | Verify official competition pages | ⚪ Monitor |

---

## 🏆 Achievement log

These achievements should reflect **evidence**, not only activity. Check once proven.

- [ ] 🥉 **Data Detective** — 2027 schema, joins, coverage and units audited.
- [ ] 🥉 **Tracking Quality Guardian** — motion-feature tests and anomaly checks pass.
- [ ] 🥈 **Football Question Defined** — primary hypothesis and target accepted at G2.
- [ ] 🥈 **Baseline Established** — reproducible traditional Combine comparison exists.
- [ ] 🥈 **Leakage-Free Analyst** — grouped split and temporal boundaries verified.
- [ ] 🥇 **Novel Signal Found** — statistically and practically defensible incremental evidence.
- [ ] 🥇 **Robust Researcher** — uncertainty, ablations, cohort and sensitivity checks documented.
- [ ] 🥇 **Scouting Storyteller** — findings understandable to a football audience.
- [ ] 🏆 **Submission Complete** — Kaggle confirmation and reproducible final report saved.

### Achievement / decision journal

| Date | Achievement or decision | Why it matters | Evidence | Next step |
|---|---|---|---|---|
| `YYYY-MM-DD` | `Example: validated athlete joins` | `Reduces ID mismatch risk` | `[Notebook](path)` | `Explore Combine distributions` |

---

## 📝 Session log (update after every work session)

<details open>
<summary><strong>Current session</strong></summary>

- **Date:** `2026-10-11`
- **Current milestone:** `Phase 04 · Milestone 1 — scientific validation (P0)` and `Phase 05 · Stage P0 — decision feasibility audit`
- **Completed:** `AUTHORED (not run): research/phase04 (Studies Q, B, C1, C2 v2, V), StudyStore phase namespace, tests/test_phase04_m1.py, notebook 09, Phase 04 report template, plan + annotation protocol`
- **Completed (Phase 05):** `WRITTEN: six PHASE_05 reports (plan, decision audit, novelty matrix, experiment / metric registries, empty results). AUTHORED (not run): research/phase05 (config, decision_audit, audit_figures, CLI), tests/test_phase05_p0_audit.py`
- **Evidence / commit:** `none yet — awaiting user execution`
- **Main finding:** `none new (code only). First read of executed A0/A3: CRITICAL key failure = rule error (BALL rows); tiers non-informative (AUC ≈ 0.50); Combine raw peak speed concentrated on SG edge frames (2.7–11.4× expected)`
- **Failed attempts / blockers:** `—`
- **Decision made:** `test interior peaks + robust quantiles against raw peaks; event features ≤ PROVISIONAL without eval-split labels; NFL features ≤ PROVISIONAL until NFL repeatability exists`
- **Next single action:** `user runs Phase 04 M1 commands and the Phase 05 P0 commands (docs/SESSION_STATE.md); Claude reviews outputs, freezes the M1 feature list and re-rates the Phase 05 stream verdicts`

Previous sessions: 2026-10-09 M1.1 foundation (A0, A3, NFL windows) authored, since executed by the user; 2026-10-08 Study A, C2 framework.

</details>

### Weekly review template

```markdown
## Review — YYYY-MM-DD (Week N)
- Progress: N / 12 weekly milestones accepted
- Completed deliverables:
- New findings (with evidence):
- Metrics versus baseline (with uncertainty):
- Key blocker / bias risk:
- Scope change or pivot decision:
- Next week's top 3 tasks:
- Links to notebooks, commits, figures:
```

---

## 📦 Submission checklist

- [ ] Official 2027 eligibility/track and submission rules verified.
- [ ] Deadline and time zone verified from official page.
- [ ] Allowed data sources verified; licensing and attribution respected.
- [ ] Final research question clearly motivates football use.
- [ ] Sample definitions, missingness and limitations disclosed.
- [ ] Claims backed by athlete-level validation and uncertainty.
- [ ] Metrics and graphics reconcile with saved results.
- [ ] Final notebook runs end-to-end on documented environment.
- [ ] No restricted datasets or personal data committed publicly.
- [ ] Final writeup proofread; limits and required assets checked.
- [ ] Submission uploaded and Kaggle confirmation captured.
- [ ] Final commit hash and submission URL recorded.

**Official links:** [2027 overview](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/overview) · [2027 data](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data) · [2027 rules](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/rules) · [2026 Analytics reference](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics)

---

## 💡 Project principles

1. **Football insight first.** Define what a scouting analyst would learn or change.
2. **Measure incremental value.** Compare against traditional Combine tests, not just a null mean.
3. **Athlete is the primary independent unit.** Do not randomly split frames across train/test.
4. **Avoid false causal claims.** Association and predictive transfer are not proof of causation.
5. **Report null and negative findings.** Do not cherry-pick one attractive result from dozens of tests.
6. **Make every achievement verifiable.** Link to code, plots, tests or a written decision.
7. **Keep scope narrow.** Prefer one compelling, validated result over five incomplete directions.

> **Next action:** Verify the official 2027 data schema and competition rules, then complete Week 1's athlete coverage audit.
