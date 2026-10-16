"""M2.1 persistence: explicit table contracts, namespaced outputs, run cache and figure manifest.

Namespace (never touches M1/M2 files):

    reports/tables/phase_03/m2_1/<name>.csv (+ .parquet twin)       kind="table"
    reports/tables/phase_03/m2_1/plot_data/<figure>__<part>.csv     data behind every figure
    reports/figures/phase_03/m2_1/<figure>.png|svg|pdf              300 DPI PNG + vector
    warehouse/gold/phase_03/m2_1/<name>.parquet (+ .schema.json)    kind="dataset" (attempt-level flags)
    warehouse/gold/phase_03/m2_1/_manifest.json                     ArtifactStore manifest + M2.1 run cache

Contracts are strict: a missing column, an extra column or a dtype mismatch raises
:class:`ContractError` — builders cast explicitly, nothing is coerced silently.

Run cache: a workstream is recomputed only when its fingerprint (M2.1 parameters + the recorded
build state of every upstream M1/M2 dataset) changed or one of its outputs is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from ...data.config import Phase01Config
from ...data.utils import atomic_path, atomic_write_json, get_logger, read_json, stable_hash, utc_now
from ..artifacts import FIG_DPI, ArtifactStore

log = get_logger(__name__)

M21_PHASE = "phase_03/m2_1"
FIG_FORMATS = ("png", "svg", "pdf")
# contracts are shared with the A–I studies (re-exported for existing M2.1 imports)
from ..contracts import (KEY_DTYPES, ContractError, TableContract, cast_keys, conform,  # noqa: E402,F401
                         upstream_state)


@dataclass(frozen=True)
class FigureSpec:
    """Everything the figure manifest records about one figure."""

    name: str
    description: str
    inputs: tuple[str, ...]
    filters: str
    selection_rule: str = ""
    features: tuple[str, ...] = ()
    workstream: str = ""


class M21Store:
    """``sampled=True`` (bounded runs with ``sample_athletes``) writes to ``phase_03/m2_1_sample`` so a
    smoke run can never replace full-run results."""

    def __init__(self, cfg: Phase01Config, sampled: bool = False):
        self.cfg = cfg
        self.sampled = sampled
        self.art = ArtifactStore(cfg, phase=M21_PHASE + ("_sample" if sampled else ""))
        self.tables_dir, self.figures_dir, self.data_dir = self.art.tables_dir, self.art.figures_dir, self.art.data_dir
        self.plot_data_dir = self.tables_dir / "plot_data"
        self.manifest_path = self.data_dir / "_manifest.json"

    # ------------------------------------------------------------------ tables / datasets
    def path(self, contract: TableContract) -> Path:
        return (self.tables_dir if contract.kind == "table" else self.data_dir) / f"{contract.name}.parquet"

    def paths(self, contract: TableContract) -> list[Path]:
        p = self.path(contract)
        return [p.with_suffix(".csv"), p] if contract.kind == "table" else [p]

    def write(self, df: pl.DataFrame, contract: TableContract) -> pl.DataFrame:
        out = conform(df, contract)
        self.art.ensure_dirs()
        if contract.kind == "table":
            self.art.export_csv(out, contract.name, contract.description, parquet=True)
        else:
            with atomic_path(self.path(contract)) as tmp:
                out.write_parquet(tmp, **self.cfg.parquet.polars_kwargs())
            atomic_write_json({"name": contract.name, "description": contract.description, "key": list(contract.key),
                               "rows": out.height, "built_at": utc_now(),
                               "columns": [{"name": c, "dtype": str(t), "description": contract.docs.get(c, "")}
                                           for c, t in contract.columns.items()],
                               "reload": f"pl.read_parquet('{self.cfg.paths.rel(self.path(contract))}')"},
                              self.data_dir / f"{contract.name}.schema.json")
        log.info("[m21] %s: %s rows → %s", contract.name, f"{out.height:,}", self.cfg.paths.rel(self.path(contract)))
        return out

    def read(self, contract: TableContract) -> pl.DataFrame:
        p = self.path(contract)
        if not p.is_file():
            raise FileNotFoundError(f"{self.cfg.paths.rel(p)} missing — run the M2.1 workstream that produces it")
        return conform(pl.read_parquet(p), contract)

    # ------------------------------------------------------------------ run cache
    def _manifest(self) -> dict:
        return read_json(self.manifest_path, {"datasets": {}, "tables": {}, "figures": {}})

    def fingerprint(self, workstream: str, params: dict[str, Any], upstream: dict[str, Any]) -> str:
        return stable_hash({"workstream": workstream, "params": params, "upstream": upstream})

    def is_current(self, workstream: str, fp: str, contracts: Iterable[TableContract]) -> tuple[bool, str]:
        entry = self._manifest().get("m21_runs", {}).get(workstream)
        if not entry:
            return False, "never run"
        if entry.get("fingerprint") != fp:
            return False, "parameters or upstream M1/M2 datasets changed"
        missing = [self.cfg.paths.rel(p) for c in contracts for p in self.paths(c) if not p.is_file()]
        return (False, f"outputs missing {missing}") if missing else (True, "current")

    def record_run(self, workstream: str, fp: str, contracts: Iterable[TableContract], extra: dict | None = None) -> None:
        m = self._manifest()
        m.setdefault("m21_runs", {})[workstream] = {
            "fingerprint": fp, "built_at": utc_now(), "outputs": [c.name for c in contracts], **(extra or {})}
        self.data_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(m, self.manifest_path)

    # ------------------------------------------------------------------ figures
    def save_figure(self, fig, spec: FigureSpec, plot_data: dict[str, pl.DataFrame], n: int | str | None = None) -> Path:
        """Persist plot data first (figure can be redrawn without raw tracking), then PNG/SVG/PDF."""
        self.plot_data_dir.mkdir(parents=True, exist_ok=True)
        data_files = []
        for part, df in plot_data.items():
            p = self.plot_data_dir / f"{spec.name}__{part}.csv"
            df.write_csv(p)
            data_files.append(self.cfg.paths.rel(p))
        path = self.art.save_fig(fig, spec.name, formats=FIG_FORMATS, dpi=FIG_DPI, source=", ".join(spec.inputs), n=n)
        m = self._manifest()
        m.setdefault("m21_figures", {})[spec.name] = {
            "workstream": spec.workstream, "files": [self.cfg.paths.rel(path.with_suffix(f".{e}")) for e in FIG_FORMATS],
            "inputs": list(spec.inputs), "filters": spec.filters, "selection_rule": spec.selection_rule,
            "features": list(spec.features), "description": spec.description, "n": None if n is None else str(n),
            "plot_data": data_files, "built_at": utc_now()}
        atomic_write_json(m, self.manifest_path)
        self.write_figure_manifest()
        return path

    def figure_manifest(self) -> pl.DataFrame:
        rows = [{"figure": k, "workstream": v.get("workstream", ""), "files": "; ".join(v.get("files", [])),
                 "inputs": "; ".join(v.get("inputs", [])), "filters": v.get("filters", ""),
                 "selection_rule": v.get("selection_rule", ""), "features": "; ".join(v.get("features", [])),
                 "description": v.get("description", ""), "n": v.get("n") or "",
                 "plot_data": "; ".join(v.get("plot_data", [])), "built_at": v.get("built_at", "")}
                for k, v in sorted(self._manifest().get("m21_figures", {}).items())]
        return pl.DataFrame(rows, schema={c: pl.String for c in (
            "figure", "workstream", "files", "inputs", "filters", "selection_rule", "features", "description", "n",
            "plot_data", "built_at")})

    def write_figure_manifest(self) -> Path:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        p = self.tables_dir / "m21_figure_manifest.csv"
        self.figure_manifest().write_csv(p)
        return p
