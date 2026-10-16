# M2.1 progress log — validation & research hardening

_All M2.1 code is AUTHORED, NOT EXECUTED. No results exist yet; no metric below is observed._

| WS | Topic | Status | Module | Tests |
|----|-------|--------|--------|-------|
| — | infrastructure (config, contracts, store, plotting, orchestrator) | authored 2026-10-08 | `research/m21/{config,io,plotting,__main__}.py` | `tests/test_m21_infra.py` |
| A | duplicate / near-duplicate attempt audit | authored 2026-10-08 | `research/m21/duplicates.py` | `tests/test_m21_duplicates.py` |
| B | sprint model robustness | **folded into Study D** (`docs/M1_1_RESEARCH_PLAN.md`) | — | — |
| C | official timing validation | **folded into Study D** | — | — |
| D | COD & braking validity | **folded into Studies C3 / E** | — | — |
| E | reliability & uncertainty | **folded into Study F** | — | — |
| F | cohort harmonization | **folded into Studies A / G / I** | — | — |
| G | redundancy & fingerprint readiness | **folded into Studies F / G** | — | — |
| — | report template, fingerprint contracts, registry update | not started | — | — |

## Interface decisions (binding for B–G)
- Namespace: `reports/{tables,figures}/phase_03/m2_1/`, `warehouse/gold/phase_03/m2_1/` (own `_manifest.json`).
  Sampled runs (`--sample-athletes N`) → `…/m2_1_sample/`. M1/M2 files are read only.
- Every output has a `TableContract` (exact dtypes, key). `conform()` raises; builders cast explicitly.
  Keys are normalised by `io.cast_keys`: `event_id` String, `nfl_id`/`attempt`/`draft_year` Int64, `drill` String.
- Thresholds live in `m21/config.py` (one dataclass per workstream) and are documented in `THRESHOLD_DOCS`
  (`established` = reused M1/M2 value · `proposed` = uncalibrated M2.1 default · `execution`). The test requires full documentation.
- Run cache: `M21Store.fingerprint(workstream, params, upstream_state(...))` + `is_current` / `record_run`.
- Figures: `plotting.figure()` + `plotting.finish(fig, caption, title)` (caption band reserved), PNG 300 DPI + SVG + PDF,
  plot data in `plot_data/<figure>__<part>.csv`, manifest in `m21_figure_manifest.csv`. Figures redraw from persisted tables only.
- Later workstreams consume `attempt_integrity_flags.parquet` (`exclude_dedup`, `exclude_strict`) for sensitivity scenarios;
  they never delete attempts.

## Workstream A — what was authored
- Per attempt: blake2b content hashes (raw and translation-invariant), path length (raw and smoothed), peak speed, moving k-grams.
- Per athlete × drill pair (≤ 8 attempts per group): time overlap, shared timestamps / frames / k-grams; onset-anchored
  lag-searched alignment; speed / vendor-speed / a_tan RMSE; translated and rigid (Kabsch) raw position RMSE; smoothed rigid RMSE.
- Classes: exact_duplicate > translated_copy > reused_sequence > time_overlap > near_duplicate > consistent / distinct / not_comparable.
- Cross-group: cross-athlete identical relative paths, conflicting ids at the same start, shared (event_id, attempt),
  repeated attempt numbers, multi-cohort athletes.
- Clusters: dedup (integrity-failure edges) and strict (plus near-dup edges, plus cross-athlete copies). The representative is the earliest attempt.
- Sensitivity: within-athlete SD per drill × feature under all / dedup / strict.
- Zero-variance athletes, each with an explanation (duplicate data / near-dup / integer-valued / coarse grid / UNEXPLAINED).
- Figures: `m21_A_duplicate_examples`, `m21_A_pair_similarity`, `m21_A_flag_prevalence`.
- M2 fix: `combine_analytics.fig_three_cone` no longer shares x+y with `adjustable="datalim"`.

## Open points for A (verify after the user run)
- `coord_quantum_yds = 0.01` assumes 2-decimal vendor coordinates. If the raw precision is coarser (e.g. in 2025),
  the exact-hash evidence still holds, but check `shared_frame_frac` behaviour.
- `time_overlap` assumes that two attempts of one athlete can never overlap in wall-clock time. If the source pads
  attempts with shared pre-roll, the prevalence will be high in every cohort. Then reclassify it as informational.
- `shared_event_attempt_id` is informational: `event_id` may be a session id shared by athletes.
