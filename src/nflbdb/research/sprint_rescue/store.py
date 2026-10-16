"""Flat persistence for the sprint-rescue namespace (no study sub-folders, no project-wide indexes).

    reports/tables/sprint_rescue/<name>.csv              tables (human-readable)
    warehouse/gold/sprint_rescue/<name>.parquet          every output, typed (+ <name>.schema.json: grain, key,
                                                         columns, provenance)
    warehouse/gold/sprint_rescue/_manifest.json          stage fingerprints (run cache)
    reports/figures/sprint_rescue/<name>.png|svg|pdf     figures; plot data under tables/…/plot_data/

``sampled=True`` nests everything under ``…/sprint_rescue/_sample`` so a smoke run never replaces full results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import polars as pl

from ...data.reporting import save_figure
from ...data.utils import atomic_path, atomic_write_csv, atomic_write_json, get_logger, read_json, stable_hash, utc_now
from ..contracts import TableContract, conform
from .config import NAMESPACE

log = get_logger(__name__)
FIG_DPI = 300


class RescueStore:
    def __init__(self, cfg: Any, sampled: bool = False, namespace: str = NAMESPACE):
        self.cfg = cfg
        self.sampled = sampled
        self.namespace = namespace          # other experiments reuse this store under their own namespace
        sub = Path(namespace) / ("_sample" if sampled else "")
        self.tables_dir = cfg.paths.tables / sub
        self.figures_dir = cfg.paths.figures / sub
        self.data_dir = cfg.paths.gold / sub
        self.manifest_path = self.data_dir / "_manifest.json"

    # ------------------------------------------------------------------ paths
    def parquet(self, c: TableContract) -> Path:
        return self.data_dir / f"{c.name}.parquet"

    def csv(self, c: TableContract) -> Path:
        return self.tables_dir / f"{c.name}.csv"

    def paths(self, c: TableContract) -> list[Path]:
        return [self.csv(c), self.parquet(c)] if c.kind == "table" else [self.parquet(c)]

    # ------------------------------------------------------------------ tables / datasets
    def write(self, df: pl.DataFrame, c: TableContract, stage: str, provenance: dict[str, Any] | None = None
              ) -> pl.DataFrame:
        out = conform(df, c)
        p = self.parquet(c)
        with atomic_path(p) as tmp:
            out.write_parquet(tmp, **self.cfg.parquet.polars_kwargs())
        if c.kind == "table":
            atomic_write_csv(out, self.csv(c))
        atomic_write_json({"name": c.name, "description": c.description, "grain": c.area, "key": list(c.key),
                           "rows": out.height, "stage": stage, "built_at": utc_now(), "provenance": provenance or {},
                           "reload": f"pl.read_parquet('{self.cfg.paths.rel(p)}')",
                           "columns": [{"name": k, "dtype": str(t), "description": c.docs.get(k, "")}
                                       for k, t in c.columns.items()]}, p.with_suffix(".schema.json"))
        log.info("[%s:%s] %s: %s rows → %s", self.namespace, stage, c.name, f"{out.height:,}", self.cfg.paths.rel(p))
        return out

    def exists(self, c: TableContract) -> bool:
        return self.parquet(c).is_file()

    def read(self, c: TableContract) -> pl.DataFrame:
        p = self.parquet(c)
        if not p.is_file():
            raise FileNotFoundError(f"{self.cfg.paths.rel(p)} missing — run the stage that produces {c.name}")
        return conform(pl.read_parquet(p), c)

    # ------------------------------------------------------------------ run cache
    def _manifest(self) -> dict:
        return read_json(self.manifest_path, {"runs": {}})

    @staticmethod
    def fingerprint(stage: str, params: dict[str, Any], upstream: dict[str, Any]) -> str:
        return stable_hash({"stage": stage, "params": params, "upstream": upstream})

    def recorded(self, stage: str) -> str | None:
        return (self._manifest()["runs"].get(stage) or {}).get("fingerprint")

    def is_current(self, stage: str, fp: str, contracts: Iterable[TableContract]) -> tuple[bool, str]:
        have = self.recorded(stage)
        if have is None:
            return False, "never run"
        if have != fp:
            return False, "parameters or upstream inputs changed"
        missing = [self.cfg.paths.rel(p) for c in contracts for p in self.paths(c) if not p.is_file()]
        return (False, f"outputs missing {missing}") if missing else (True, "current")

    def record(self, stage: str, fp: str, contracts: Iterable[TableContract]) -> None:
        m = self._manifest()
        m["runs"][stage] = {"fingerprint": fp, "built_at": utc_now(), "outputs": [c.name for c in contracts]}
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(m, self.manifest_path)

    # ------------------------------------------------------------------ figures
    def save_figure(self, fig, name: str, plot_data: dict[str, pl.DataFrame], publication: bool = False,
                    source: str = "", n: str = "") -> Path:
        """Plot data first (a figure redraws without recomputation), then PNG 300 DPI + SVG (+ PDF)."""
        for part, df in plot_data.items():
            atomic_write_csv(df, self.tables_dir / "plot_data" / f"{name}__{part}.csv")
        formats = ("png", "svg", "pdf") if publication else ("png", "svg")
        return save_figure(fig, self.figures_dir, name, close=True, dpi=FIG_DPI, formats=formats,
                           metadata={"Source": source, "N": n})
