"""Sprint reliability rescue — orchestrator (code authored, NOT executed — the user runs every command).

    python -m nflbdb.research.sprint_rescue --sample-athletes 80     # smoke → */sprint_rescue/_sample
    python -m nflbdb.research.sprint_rescue                          # all stages, figures, report
    python -m nflbdb.research.sprint_rescue --only B,D               # selected stages (dependencies must exist)
    python -m nflbdb.research.sprint_rescue --figures-only           # redraw from saved tables
    python -m nflbdb.research.sprint_rescue --report                 # refill the Markdown report only
    python -m nflbdb.research.sprint_rescue --check                  # list missing outputs
"""

from __future__ import annotations

import argparse

from ...data.config import Phase01Config, configure_runtime
from ...data.utils import get_logger
from .config import RescueConfig
from .figures import FIGURES
from .pipeline import STAGES, RescueInputs, run_stage
from .report import write_report
from .store import RescueStore

log = get_logger(__name__)


def draw_figures(store: RescueStore) -> None:
    by_name = {c.name: c for contracts, _ in STAGES.values() for c in contracts}
    for name, needs, fn in FIGURES:
        missing = [n for n in needs if not store.exists(by_name[n])]
        if missing:
            log.warning("[sprint_rescue] figure %s skipped: %s not built", name, missing)
            continue
        fn(store, {n: store.read(by_name[n]) for n in needs})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sprint reliability rescue: censoring, within-position ICC, redundancy")
    ap.add_argument("--only", default=",".join(STAGES), help=f"comma-separated stages of {list(STAGES)}")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample-athletes", type=int, default=None, help="first N athletes → _sample namespace")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--report", action="store_true", help="only refill the Markdown report from saved tables")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    rcfg = RescueConfig()
    store = RescueStore(cfg, sampled=a.sample_athletes is not None)
    keys = [k.strip().upper() for k in a.only.split(",") if k.strip()]
    unknown = [k for k in keys if k not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage {unknown}; available {list(STAGES)}")

    if a.check:
        missing = [cfg.paths.rel(p) for k in keys for c in STAGES[k][0] for p in store.paths(c) if not p.is_file()]
        missing += [cfg.paths.rel(store.figures_dir / f"{name}.png") for name, _, _ in FIGURES
                    if not (store.figures_dir / f"{name}.png").is_file()]
        print("all outputs present" if not missing else "MISSING:\n  " + "\n  ".join(missing))
        return 1 if missing else 0

    if not (a.figures_only or a.report):
        inp = RescueInputs(cfg, rcfg, a.sample_athletes)
        for k in sorted(keys, key=list(STAGES).index):
            run_stage(k, inp, store, force=a.force)
    if not (a.no_figures or a.report):
        draw_figures(store)
    if not a.sample_athletes:
        log.info("[sprint_rescue] report → %s", write_report(cfg, store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
