"""Combine → NFL defensive-line get-off — orchestrator (code authored, NOT executed — the user runs every command).

    python -m nflbdb.research.combine_nfl_dl --only DEF,TGT,COH            # outcome audit, targets, cohort (never gated)
    python -m nflbdb.research.combine_nfl_dl                               # all stages; MOD and SENS need a confirmed
                                                                           # endpoint definition
    python -m nflbdb.research.combine_nfl_dl --allow-unresolved-definition # run MOD and SENS anyway, labelled unresolved
    python -m nflbdb.research.combine_nfl_dl --smoke --allow-unresolved-definition   # cheap wiring check → */_sample
    python -m nflbdb.research.combine_nfl_dl --figures-only                # redraw from saved tables
    python -m nflbdb.research.combine_nfl_dl --report                      # refill the Markdown report only
    python -m nflbdb.research.combine_nfl_dl --check                       # list missing outputs
"""

from __future__ import annotations

import argparse

from ...data.config import Phase01Config, configure_runtime
from ...data.utils import get_logger
from .config import GetOffConfig
from .figures import FIGURES
from .pipeline import STAGES, Inputs, make_store, run_stage, smoke_config
from .report import write_report

log = get_logger(__name__)


def draw_figures(store) -> None:
    by_name = {c.name: c for contracts, _ in STAGES.values() for c in contracts}
    for name, needs, fn in FIGURES:
        missing = [n for n in needs if not store.exists(by_name[n])]
        if missing:
            log.warning("[combine_nfl_dl] figure %s skipped: %s not built", name, missing)
            continue
        fn(store, {n: store.read(by_name[n]) for n in needs})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Combine tracking vs official tests as predictors of NFL DL get-off")
    ap.add_argument("--only", default=",".join(STAGES), help=f"comma-separated stages of {list(STAGES)}")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-unresolved-definition", action="store_true",
                    help="run MOD and SENS although configs/player_get_off_definition.json is not confirmed")
    ap.add_argument("--smoke", action="store_true", help="cheap settings → _sample namespace (not results)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--report", action="store_true", help="only refill the Markdown report from saved tables")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    gcfg = smoke_config(GetOffConfig()) if a.smoke else GetOffConfig()
    store = make_store(cfg, sampled=a.smoke)
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

    blocked = []
    if not (a.figures_only or a.report):
        inp = Inputs(cfg, gcfg)
        for k in sorted(keys, key=list(STAGES).index):
            if run_stage(k, inp, store, force=a.force, allow_unresolved=a.allow_unresolved_definition) == "blocked":
                blocked.append(k)
    if not (a.no_figures or a.report):
        draw_figures(store)
    if not a.smoke:
        log.info("[combine_nfl_dl] report → %s", write_report(cfg, store))
    if blocked:
        log.warning("[combine_nfl_dl] stages blocked by the endpoint-definition gate: %s", blocked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
