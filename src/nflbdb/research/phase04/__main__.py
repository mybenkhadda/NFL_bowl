"""Phase 04 · Milestone 1 orchestrator (code authored, NOT executed — the user runs every command).

    python -m nflbdb.research.phase04 --only Q                                   # structural vs signal, eligibility
    python -m nflbdb.research.phase04 --only B --sample-athletes 120             # smoke → */phase_04/_sample
    python -m nflbdb.research.phase04 --only B                                   # smoothing robustness (heavy)
    python -m nflbdb.research.phase04 --only C1                                  # event sensitivity (needs B)
    python -m nflbdb.research.phase04 --annotation-tasks                         # C2 v2 tasks + blind packets
    python -m nflbdb.research.phase04 --evaluate-annotations reports/annotation/phase_04/annotation_labels.csv
    python -m nflbdb.research.phase04 --only V                                   # feature status registry
    python -m nflbdb.research.phase04 --only B,C1 --figures-only                 # redraw from saved tables
    python -m nflbdb.research.phase04 --check                                    # list missing outputs
    python -m nflbdb.research.phase04 --report                                   # fill the Phase 04 report
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ...data.config import Phase01Config, configure_runtime

if __name__ == "__main__":      # `python -m …`: Polars reads POLARS_MAX_THREADS once, at import — set it first
    configure_runtime(Phase01Config.load())

import polars as pl  # noqa: E402

from ...data.utils import get_logger
from ..contracts import TableContract
from ..studies.store import StudyStore
from . import events as c1
from . import quality_eligibility as q
from . import smoothing as b
from . import smoothing_figures as bf
from . import validation_registry as v
from .annotation_design import AnnotationDesign
from .config import Phase04Config, threshold_table
from .report import build_phase04_report

log = get_logger(__name__)
# run order: Q (reads Phase 03 A/A3) → B (heavy recompute) → C1 (reads B) → V (reads everything)
STUDIES = {"Q": (q.QualityEligibilityStudy, q.CONTRACTS, q.FIGURES),
           "B": (b.SmoothingStudy, b.CONTRACTS, bf.FIGURES),
           "C1": (c1.EventSensitivityStudy, c1.CONTRACTS, c1.FIGURES),
           "V": (v.ValidationRegistryStudy, v.CONTRACTS, v.FIGURES)}
THRESHOLDS = TableContract(
    "P04_thresholds", "Every Phase 04 parameter with unit and provenance (established / proposed / execution).",
    {c: pl.String() for c in ("study_section", "name", "value", "unit", "provenance", "rationale")},
    key=("study_section", "name"), area="descriptive")
# study → (prerequisite study, mandatory?). Mandatory: the study refuses to run without it.
UPSTREAM = {"Q": (), "B": (), "C1": (("B", True),), "V": (("B", True), ("C1", False))}


def check_study(store: StudyStore, key: str, contracts, figures, fingerprint=None) -> list[str]:
    """Problems of one study's persisted outputs, one line each (empty = usable):

    missing     an expected file is absent
    invalid     a file is present but violates its contract (schema, null or duplicated key)
    incomplete  no run record — the study never ran or failed before all outputs passed their contracts
    blocked     a mandatory prerequisite study has no completed run
    stale       parameters / upstream changed since the recorded run, or a prerequisite was rebuilt after it
    """
    rel = store.cfg.paths.rel
    issues = []
    for c in contracts:
        issues += [f"missing     {rel(p)}" for p in store.paths(c) if not p.is_file()]
        if store.path(c).is_file():
            try:
                store.read(c)
            except Exception as e:  # noqa: BLE001 — a diagnostic must report every unreadable file, not stop
                issues.append(f"invalid     {rel(store.path(c))}: {e}")
    issues += [f"missing     {rel(store.figure_dir(r.study) / f'{r.name}.png')}" for r, _ in figures
               if not (store.figure_dir(r.study) / f"{r.name}.png").is_file()]
    runs = store._manifest()["runs"]
    entry = runs.get(key)
    if not entry:
        issues.append(f"incomplete  Study {key}: no completed run recorded (never run, or the last run failed) "
                      f"→ --only {key}")
    for up, mandatory in UPSTREAM.get(key, ()):
        u = runs.get(up)
        if not u and mandatory:
            issues.append(f"blocked     Study {key}: prerequisite Study {up} has no completed run → --only {up}")
        elif u and entry and str(entry.get("built_at", "")) < str(u.get("built_at", "")):
            issues.append(f"stale       Study {key}: built before the current Study {up} run → --only {key}")
    if entry and fingerprint is not None:
        try:
            if entry.get("fingerprint") != fingerprint():
                issues.append(f"stale       Study {key}: parameters or upstream datasets changed since the recorded "
                              f"run → --only {key}")
        except FileNotFoundError as e:
            issues.append(f"blocked     Study {key}: {e}")
    return issues


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 04 · Milestone 1 — scientific validation")
    ap.add_argument("--only", default="Q,B,C1,V")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample-athletes", type=int, default=None)
    ap.add_argument("--skip-nfl-sample", action="store_true", help="Study B: Combine only")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--annotation-tasks", action="store_true")
    ap.add_argument("--evaluate-annotations", type=Path, default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    pcfg = Phase04Config(sample_athletes=a.sample_athletes)
    sampled = a.sample_athletes is not None
    store = StudyStore(cfg, sampled=sampled, phase="phase_04")
    store03 = StudyStore(cfg, sampled=sampled)
    keys = [k.strip().upper() for k in a.only.split(",") if k.strip()]
    unknown = [k for k in keys if k not in STUDIES]
    if unknown:
        raise SystemExit(f"study {unknown} not authored; available {list(STUDIES)}")

    if a.check:
        issues = []
        for k in keys:
            cls, contracts, figs = STUDIES[k]
            study = cls(cfg, pcfg, store, store03=store03, skip_nfl_sample=a.skip_nfl_sample)
            issues += check_study(store, k, contracts, figs, getattr(study, "fingerprint", None))
        print("all outputs present, valid and current" if not issues else "PROBLEMS:\n  " + "\n  ".join(issues))
        return 1 if issues else 0
    if a.report:
        unresolved = build_phase04_report(store, store03)
        print(f"report written; {len(unresolved)} unresolved placeholder(s)"
              + (" — sources listed at the end of the report" if unresolved else ""))
        return 0
    if a.annotation_tasks or a.evaluate_annotations:
        if sampled:
            raise SystemExit("annotation always uses the full namespace — drop --sample-athletes")
        ad = AnnotationDesign(cfg, pcfg, store, store03)
        if a.annotation_tasks:
            ad.tasks()
        if a.evaluate_annotations:
            ad.evaluate(a.evaluate_annotations)
        return 0

    store.write(threshold_table(pcfg), THRESHOLDS, "P04")
    for k in sorted(keys, key=list(STUDIES).index):
        study = STUDIES[k][0](cfg, pcfg, store, store03=store03, skip_nfl_sample=a.skip_nfl_sample)
        if a.figures_only:
            study.figures(study.load())
        else:
            study.run(force=a.force, figures=not a.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
