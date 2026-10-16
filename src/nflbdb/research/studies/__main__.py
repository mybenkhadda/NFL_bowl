"""Studies A–I orchestrator (code authored, NOT executed — the user runs every command).

    python -m nflbdb.research.studies --only A --sample-athletes 300 --skip-nfl-sample   # smoke → phase_03/_sample
    python -m nflbdb.research.studies --only A                                           # full Study A
    python -m nflbdb.research.studies --only A0                                          # exact DQ-failure drill-down
    python -m nflbdb.research.studies --only A,A3                                        # A3 after A (uses A1 flags)
    python -m nflbdb.research.studies --only A --figures-only                            # redraw from tables
    python -m nflbdb.research.studies --annotation-tasks                                 # C2 step 1–2
    python -m nflbdb.research.studies --evaluate-annotations reports/annotation/annotation_labels.csv   # C2 step 4–6
    python -m nflbdb.research.studies --report                                           # fill the report template
    python -m nflbdb.research.studies --check                                            # list missing outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ...data.config import Phase01Config, configure_runtime
from ...data.utils import get_logger
from ..contracts import cast_keys
from ..kinematic_build import COMBINE_EVENTS, COMBINE_FRAMES
from . import annotation as ann
from .config import StudiesConfig
from .dq_failures import CONTRACTS as A0_CONTRACTS, FIGURES as A0_FIGURES, DqFailureStudy
from .quality import CONTRACTS as A_CONTRACTS, DIMS, QualityStudy, traj_id_expr
from .quality_figures import FIGURES as A_FIGURES
from .report import build_report
from .signal_reliability import CONTRACTS as A3_CONTRACTS, FIGURES as A3_FIGURES, SignalReliabilityStudy
from .store import StudyStore

log = get_logger(__name__)
# run order matters: A3 reads Study A's dimensions (n_flagged_dims) when they exist
STUDIES = {"A0": (DqFailureStudy, A0_CONTRACTS, A0_FIGURES), "A": (QualityStudy, A_CONTRACTS, A_FIGURES),
           "A3": (SignalReliabilityStudy, A3_CONTRACTS, A3_FIGURES)}
ANNOTATION_DIR = Path("reports") / "annotation"
CKEYS = ["event_id", "nfl_id", "attempt"]


def _combine_frames(qs: QualityStudy, traj_ids: list[str]) -> pl.DataFrame:
    ids = sorted({int(t.split("|")[1]) for t in traj_ids})
    f = (qs.src.load(COMBINE_FRAMES.name).filter(pl.col("nfl_id").is_in(ids) & pl.col("valid_kin"))
         .select(*CKEYS, "t_s", "speed", "accel_mag", "a_tan", "heading", "x_s", "y_s").collect())
    return cast_keys(f).with_columns(traj_id_expr(CKEYS)).filter(pl.col("traj_id").is_in(traj_ids))


def annotation_tasks(cfg: Phase01Config, store: StudyStore, qs: QualityStudy) -> None:
    dims = store.read(DIMS)
    tasks = store.write(ann.select_tasks(dims), ann.TASKS, "C")
    root = cfg.paths.project_root / ANNOTATION_DIR
    root.mkdir(parents=True, exist_ok=True)
    tmpl = root / "annotation_labels_TEMPLATE.csv"
    ann.label_template(tasks).write_csv(tmpl)              # template only — the filled file is never overwritten
    n = len(ann.export_packets(_combine_frames(qs, tasks["traj_id"].to_list()), tasks, root / "packets"))
    print(f"{tasks.height} tasks, {n} packets → {cfg.paths.rel(root)}; copy {tmpl.name} to annotation_labels.csv and fill it")


def evaluate_annotations(cfg: Phase01Config, store: StudyStore, qs: QualityStudy, labels_path: Path) -> None:
    tasks = store.read(ann.TASKS)
    labels = ann.validate_labels(pl.read_csv(cfg.paths.project_root / labels_path,
                                             schema_overrides={c: t for c, t in ann.LABELS.columns.items()}), tasks)
    det = (cast_keys(qs.src.load(COMBINE_EVENTS.name).select(*CKEYS, "rule", "event_type", "t_start_s", "t_end_s").collect())
           .with_columns(traj_id_expr(CKEYS)).filter(pl.col("traj_id").is_in(tasks["traj_id"].to_list())))
    acfg = ann.AnnotationConfig()
    m = store.write(ann.match_all(labels, det, tasks, acfg.tolerance_s), ann.MATCHES, "C")
    perf = store.write(ann.evaluate(m, acfg), ann.PERFORMANCE, "C")
    with pl.Config(tbl_rows=30):
        print(perf.filter((pl.col("facet") == "all") & (pl.col("label_set") == "all_labels")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 03 · studies A–I")
    ap.add_argument("--only", default="A")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample-athletes", type=int, default=None)
    ap.add_argument("--skip-nfl-sample", action="store_true", help="Study A: omit the NFL frame sample (A2 Combine only)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--annotation-tasks", action="store_true")
    ap.add_argument("--evaluate-annotations", type=Path, default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    scfg = StudiesConfig(sample_athletes=a.sample_athletes)
    store = StudyStore(cfg, sampled=a.sample_athletes is not None)
    keys = [k.strip().upper() for k in a.only.split(",") if k.strip()]
    unknown = [k for k in keys if k not in STUDIES]
    if unknown:
        raise SystemExit(f"study {unknown} not authored yet; available {list(STUDIES)}")
    qs = QualityStudy(cfg, scfg, store, skip_nfl_sample=a.skip_nfl_sample)

    if a.check:
        missing = []
        for k in keys:
            _, contracts, figs = STUDIES[k]
            missing += [cfg.paths.rel(p) for c in contracts for p in store.paths(c) if not p.is_file()]
            missing += [cfg.paths.rel(store.figure_dir(r.study) / f"{r.name}.png") for r, _ in figs
                        if not (store.figure_dir(r.study) / f"{r.name}.png").is_file()]
        print("all outputs present" if not missing else "MISSING:\n  " + "\n  ".join(missing))
        return 1 if missing else 0
    if a.annotation_tasks:
        annotation_tasks(cfg, store, qs)
        return 0
    if a.evaluate_annotations:
        evaluate_annotations(cfg, store, qs, a.evaluate_annotations)
        return 0
    if a.report:
        unresolved = build_report(store)
        print(f"report written; {len(unresolved)} unresolved placeholder(s)")
        return 0
    for k in sorted(keys, key=list(STUDIES).index):
        study = STUDIES[k][0](cfg, scfg, store, skip_nfl_sample=a.skip_nfl_sample)
        if a.figures_only:
            study.figures(study.load())
        else:
            study.run(force=a.force, figures=not a.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
