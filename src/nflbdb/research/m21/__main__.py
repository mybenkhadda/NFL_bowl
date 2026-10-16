"""M2.1 orchestrator — runs the authored workstreams in dependency order.

Commands for the USER (code authored, NOT executed)::

    python -m nflbdb.research.m21 --check                      # list missing M2.1 outputs, compute nothing
    python -m nflbdb.research.m21 --only A --sample-athletes 200   # bounded smoke run
    python -m nflbdb.research.m21 --only A                     # full run (reuses current results)
    python -m nflbdb.research.m21 --only A --figures-only      # redraw figures from persisted tables
    python -m nflbdb.research.m21 --force                      # recompute every authored workstream

Workstreams are registered in ``WORKSTREAMS`` as they are authored (A only so far).
"""

from __future__ import annotations

import argparse
import importlib

from ...data.config import Phase01Config, configure_runtime
from .config import M21Config
from .io import M21Store

# key → (module, class); order = execution order (later workstreams read earlier flags)
WORKSTREAMS: dict[str, tuple[str, str]] = {
    "A": ("duplicates", "DuplicateAudit"),
}


def _module(key: str):
    return importlib.import_module(f"{__package__}.{WORKSTREAMS[key][0]}")


def expected_outputs(store: M21Store, keys: list[str]) -> list:
    paths = []
    for k in keys:
        mod = _module(k)
        paths += [p for c in mod.CONTRACTS for p in store.paths(c)]
        paths += [store.figures_dir / f"{f}.{ext}" for f in mod.FIGURES for ext in ("png", "svg", "pdf")]
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 03 · M2.1 validation workflow")
    ap.add_argument("--only", default=",".join(WORKSTREAMS), help="comma-separated workstream keys, e.g. A,B")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample-athletes", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    keys = [k.strip().upper() for k in a.only.split(",") if k.strip()]
    unknown = [k for k in keys if k not in WORKSTREAMS]
    if unknown:
        raise SystemExit(f"unknown / not yet authored workstream(s) {unknown}; available {list(WORKSTREAMS)}")
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    store = M21Store(cfg, sampled=a.sample_athletes is not None)   # sampled runs → phase_03/m2_1_sample
    if a.check:
        missing = [cfg.paths.rel(p) for p in expected_outputs(store, keys) if not p.is_file()]
        print("all M2.1 outputs present" if not missing else "MISSING:\n  " + "\n  ".join(missing))
        return 1 if missing else 0
    m21 = M21Config(sample_athletes=a.sample_athletes)
    for k in keys:
        mod, cls = _module(k), WORKSTREAMS[k][1]
        ws = getattr(mod, cls)(cfg, m21, store=store)
        if a.figures_only:
            ws.figures(ws.load())
        else:
            ws.run(force=a.force, figures=not a.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
