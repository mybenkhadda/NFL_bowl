"""Phase 05 orchestrator (code authored, NOT executed — the user runs every command).

    python -m nflbdb.research.phase05 --only P0                     # data & decision feasibility audit
    python -m nflbdb.research.phase05 --only P0 --seasons 2023      # smoke → */phase_05/_sample
    python -m nflbdb.research.phase05 --only P0B                    # analysis sample frame (after the P0 review)
    python -m nflbdb.research.phase05 --only P0 --figures-only      # redraw from saved tables
    python -m nflbdb.research.phase05 --check                       # list missing outputs
"""

from __future__ import annotations

import argparse
import dataclasses

import polars as pl

from ...data.config import Phase01Config, configure_runtime
from ...data.utils import get_logger
from ..contracts import TableContract
from ..studies.store import StudyStore
from . import audit_figures as af
from . import decision_audit as da
from . import sample_frame as sf
from .config import PHASE, Phase05Config, threshold_table

log = get_logger(__name__)
# later stages (S2 envelope … S8 dashboard) register here once their gate has passed
STUDIES = {"P0": (da.DecisionAuditStudy, da.CONTRACTS, af.FIGURES),
           "P0B": (sf.SampleFrameStudy, sf.CONTRACTS, sf.FIGURES)}
THRESHOLDS = TableContract(
    "P05_thresholds", "Every Phase 05 parameter with unit and provenance (established / proposed / execution).",
    {c: pl.String() for c in ("study_section", "name", "value", "unit", "provenance", "rationale")},
    key=("study_section", "name"), area="descriptive")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 05 — frontier research (Stage P0: feasibility audit)")
    ap.add_argument("--only", default="P0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seasons", default=None, help="comma-separated subset, e.g. 2023 → _sample namespace")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    seasons = tuple(int(s) for s in a.seasons.split(",") if s.strip()) if a.seasons else None
    base = Phase05Config()
    pcfg = dataclasses.replace(base, audit=dataclasses.replace(base.audit, seasons=seasons))
    store = StudyStore(cfg, sampled=pcfg.sampled, phase=PHASE)
    keys = [k.strip().upper() for k in a.only.split(",") if k.strip()]
    unknown = [k for k in keys if k not in STUDIES]
    if unknown:
        raise SystemExit(f"stage {unknown} not authored; available {list(STUDIES)}")

    if a.check:
        missing = []
        for k in keys:
            _, contracts, figs = STUDIES[k]
            missing += [cfg.paths.rel(p) for c in contracts for p in store.paths(c) if not p.is_file()]
            missing += [cfg.paths.rel(store.figure_dir(r.study) / f"{r.name}.png") for r, _ in figs
                        if not (store.figure_dir(r.study) / f"{r.name}.png").is_file()]
        print("all outputs present" if not missing else "MISSING:\n  " + "\n  ".join(missing))
        return 1 if missing else 0

    store.write(threshold_table(pcfg), THRESHOLDS, "P05")
    for k in sorted(keys, key=list(STUDIES).index):
        study = STUDIES[k][0](cfg, pcfg, store)
        if a.figures_only:
            study.figures(study.load())
        else:
            out = study.run(force=a.force, figures=not a.no_figures)
            if k == "P0":
                verdicts = out[da.FEAS.name].unique(subset=["stream"], keep="first").sort("stream")
                for r in verdicts.iter_rows(named=True):
                    log.info("[phase05:P0] stream %s → %s (screening, proposed thresholds)", r["stream"], r["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
