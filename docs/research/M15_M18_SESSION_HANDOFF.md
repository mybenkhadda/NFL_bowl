# M15–M18 session handoff (start here in a fresh session)

**Updated:** 2026-10-20, session 4. Every task T01–T17 is authored. Claude has executed nothing.

## Rules (binding)

- **Code-authoring only.** Claude never runs Python, notebooks, tests, linters, shell commands or installs. The user runs everything in Colab.
- **Read-only sources.** M12–M17 run directories and the exploratory M15 folders are read only. `resolve()` and `Run.guard()` enforce this.
- **Context discipline.** Read this file, then only the module named in a traceback. Never open executed `.ipynb` files.
- **Numbers.** Numbers come from saved tables only. Every gate defaults to `NOT_EVALUATED`.

## Objective

Do 10 Hz Combine movement measurements give reliable, independently predictive information about NFL movement beyond official testing? The official baseline is M1 = `draft_year, height, weight, forty, ten`. The official target is `nfl.q90_shared`. A rigorous negative answer is acceptable.

## Status

| Milestone | Tasks | Code | Notebook |
|---|---|---|---|
| M15 | T01, T02, T03, T07, T08 | `src/nflbowl/lab/m15/` | `notebooks/nflbowl/M15_SPECTRAL_TEMPORAL_REPRESENTATION_LAB_COLAB.ipynb` |
| M16 | T04, T06, T09, T10 | `src/nflbowl/lab/m16/` | `notebooks/nflbowl/M16_VERIFIED_BASELINES_CROSS_DOMAIN_VALIDATION_COLAB.ipynb` |
| M17 | T05, T12, T13, T14 | `src/nflbowl/lab/m17/` | `notebooks/nflbowl/M17_CONTEXT_AWARE_ATHLETIC_SIGNATURES_COLAB.ipynb` |
| M18 | T15, T16, T17 | `src/nflbowl/lab/m18/` | `notebooks/nflbowl/M18_NFL_SCOUTING_VALUE_COLAB.ipynb` |
| Shared | — | `src/nflbowl/lab/status.py` (statuses, `run_step`, `gate_row`); `src/nflbowl/lab/learners.py` (ENet / LightGBM / CatBoost, `oof_generic`) | — |

T11, the full encoder grid, needs no code. It is a configuration change in M15 Setup 3: `architectures`, `latent_dims` = (8, 16, 32, 64), and three `train_seeds`.

**Tests (synthetic, unexecuted):** `tests/nflbowl/test_nflbowl_m15.py`, `test_nflbowl_m16.py`, `test_nflbowl_m17.py`, `test_nflbowl_m18.py`.

## Execution order (Colab)

1. **M15.** Run all.
   - With `run_training` and `allow_new_training` False, only PCA is fitted. Learned rows are NOT_RUN, and G15 for them stays NOT_EVALUATED.
   - For a GPU session, set both opt-ins.
2. **M16.** Run all. Main outputs: M1 verification, targets T1–T6 and their freeze, baselines, MDE, retrieval, leakage audit, G16.
3. **M17.** Run all. Tracks C and D are BLOCKED unless a G15-GO representation passed the M16 leakage audit.
4. **M18.** Run all. It pre-registers once. Rungs are blocked or eligible strictly from the gate tables.

Each notebook's last cell lists its SEND BACK files. Always include `manifests/stages.json` and `logs/*_traceback.txt`.

## Cross-milestone data flow (all read only)

- **M15 → M16:** `features/signal/sequence_time_features.parquet`, `embeddings/inductive/*`, `checkpoints/*/fold*/{train_athletes,done}.json`, `tables/synthesis/m15_gate.parquet`.
- **M16 → M17 / M18:**
  - `features/targets/{athlete_targets, nfl_events}.parquet`
  - `manifests/target_freeze_m16.json`
  - `datasets/cached/analysis_frame.parquet`
  - `tables/retrieval/leakage_verdict.parquet`
  - `tables/synthesis/m16_gate.parquet`
  - `tables/detectability/mde.parquet`
- **M17 → M18:** `features/events_v2/athlete_signatures.parquet`, `features/fusion/*__fold*.parquet`, `tables/fusion/fusion_adoption.parquet`, `tables/synthesis/m17_gate.parquet`.
- **M14 → M16 / M17:** `features/athlete/athlete_features.parquet`, `tables/reliability/measurement_status.parquet`.

## Known risks and assumptions (unverified by execution)

**Polars APIs assumed:**
- `replace_strict(return_dtype=…)`
- `pivot(on=…)`, `unpivot`
- `join(how="full", coalesce=True)`, `read_parquet_schema`
- `write_csv(float_precision=…)`
- `int_range().shuffle().over()`
- `group_by(maintain_order=True)`

**Engine and determinism:**
- `ae.oof_ladder` and `se.embedding_block` are reused for every ridge rung.
- Persisted vs recomputed M1 equality assumes deterministic refits. A mismatch is a FAILED_VALIDATION with athlete diffs.

**Assumed external schemas:**
- The M14 measurement-status schema is assumed to hold (`column` | `feature`, `status` | `measurement_status`). Otherwise B3 is BLOCKED with the schema printed.
- The game key is parsed from NFL `seq_key` (10-digit gameId). If that fails, target reliability uses season splits or random halves, labelled.

**Runtime, unmeasured:**
- M15 spectral pass: about 77k × 8 configurations.
- M16: NFL M14 extraction (about 71k windows) and the MDE simulation (`detect_reps` × fractions × targets).
- M17: threshold grid (7 extractions over all Combine rows plus 6k NFL rows).
- M18: permutation null (`n_perm` ladder refits).

Every expensive step caches by provenance and resumes.

**Stated simplifications:**
- MDE uses a fixed-α ridge.
- The KM censor time uses a nominal 0.1 s step.
- M17 BLUPs for training athletes use context means that include held-out athletes' Combine events. No outcome is involved, and this is stated in the M18 ladder docstring.
- B2 simple statistics have no measurement gate, so they can be at most "promising", never "scouting_relevant".
- Attention fusion is NOT_RUN by the adoption rule.

## If something fails

Read the failing step's traceback and the module it names. Fix only that module. The `status.run_step` ledger keeps later independent steps running.

## Next actions

- **User:** run M15 → M16 → M17 → M18 in Colab and send back the listed files.
- **Claude:** fix reported failures. Then, from executed tables only, fill the narrative and decide pivots per `M15_M18_SCIENTIFIC_GATES.md`.
