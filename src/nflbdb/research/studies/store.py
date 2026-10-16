"""Persistence for studies A–I: study folders, strict contracts, run cache, figure & table indexes.

Layout (brief §14/§16; M1/M2 files in the phase_03 roots are never touched)::

    reports/figures/phase_03/<A_quality|B_smoothing|…|final_story>/<name>.png|svg|pdf
    reports/figures/phase_03/figure_index.csv                       one row per figure (question, grain, …)
    reports/tables/phase_03/<descriptive|statistical|reliability|hypothesis_screening>/<name>.csv + .parquet
    reports/tables/phase_03/plot_data/<study dir>/<figure>__<part>.csv
    reports/tables/phase_03/table_index.csv
    warehouse/gold/phase_03/<quality|movement_events|maneuvers|…|research_base>/<name>.parquet (+ .schema.json)
    warehouse/gold/phase_03/research_base/_studies_manifest.json     run cache + indexes (JSON)

``sampled=True`` (bounded runs) nests everything under ``…/phase_03/_sample/`` so a smoke run can
never replace full-run results.

``phase`` (default ``"phase_03"``) selects the root; Phase 04 studies use ``phase="phase_04"`` and get
their own manifest, indexes and ``_sample`` namespace — Phase 03 outputs are never touched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from ...data.config import Phase01Config
from ...data.reporting import save_figure
from ...data.utils import atomic_path, atomic_write_csv, atomic_write_json, get_logger, read_json, stable_hash, utc_now
from ..contracts import TableContract, conform

log = get_logger(__name__)

STUDY_DIRS = {"A0": "A0_integrity", "A": "A_quality", "A3": "A3_signal", "B": "B_smoothing", "C": "C_events", "D": "D_sprint", "E": "E_agility",
              "F": "F_reliability", "G": "G_positions", "H": "H_transfer", "I": "I_selection", "R": "final_story",
              # Phase 04 · Milestone 1 (scientific validation)
              "Q": "Q_quality_eligibility", "C1": "C1_event_sensitivity", "C2": "C2_annotation",
              "V": "V_feature_validation",
              # Phase 05 · frontier research
              "P0": "P0_feasibility", "P0B": "P0b_sample_frame"}
TABLE_AREAS = ("descriptive", "statistical", "reliability", "hypothesis_screening")
DATA_AREAS = ("quality", "movement_events", "maneuvers", "combine_attempts", "nfl_player_plays", "athlete_profiles",
              "research_base", "decision_audit")
FIG_DPI = 300


@dataclass(frozen=True)
class FigureRecord:
    """Figure index entry — every figure states the question it answers before it is drawn."""

    figure_id: str              # e.g. "A1-04"
    name: str                   # file stem, e.g. "A1_04_vendor_residual_by_tier"
    study: str                  # key of STUDY_DIRS
    research_question: str
    expected_insight: str
    statistic: str              # statistical quantity visualised
    dataset: str                # persisted table(s) the figure is drawn from
    grain: str
    filters: str
    method: str
    publication: bool = False   # True → PNG + SVG + PDF, else PNG + SVG
    interpretation_status: str = "pending execution"


class StudyStore:
    def __init__(self, cfg: Phase01Config, sampled: bool = False, phase: str = "phase_03"):
        self.cfg = cfg
        self.sampled = sampled
        self.phase = phase
        sub = Path(phase) / ("_sample" if sampled else "")
        self.tables_root = cfg.paths.tables / sub
        self.figures_root = cfg.paths.figures / sub
        self.data_root = cfg.paths.gold / sub
        self.manifest_path = self.data_root / "research_base" / "_studies_manifest.json"

    # ------------------------------------------------------------------ paths
    def path(self, c: TableContract) -> Path:
        if c.kind == "table":
            if c.area not in TABLE_AREAS:
                raise ValueError(f"{c.name}: table area {c.area!r} not in {TABLE_AREAS}")
            return self.tables_root / c.area / f"{c.name}.parquet"
        if c.area not in DATA_AREAS:
            raise ValueError(f"{c.name}: dataset area {c.area!r} not in {DATA_AREAS}")
        return self.data_root / c.area / f"{c.name}.parquet"

    def paths(self, c: TableContract) -> list[Path]:
        p = self.path(c)
        return [p.with_suffix(".csv"), p] if c.kind == "table" else [p]

    # ------------------------------------------------------------------ manifest
    def _manifest(self) -> dict:
        return read_json(self.manifest_path, {"tables": {}, "figures": {}, "runs": {}})

    def _save_manifest(self, m: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(m, self.manifest_path)

    # ------------------------------------------------------------------ tables / datasets
    def write(self, df: pl.DataFrame, c: TableContract, study: str) -> pl.DataFrame:
        out = conform(df, c)
        p = self.path(c)
        p.parent.mkdir(parents=True, exist_ok=True)
        if c.kind == "table":
            if out.height > 1_000_000:
                raise ValueError(f"{c.name}: {out.height:,} rows is too large for a CSV table — declare a dataset")
            atomic_write_csv(out, p.with_suffix(".csv"))
        with atomic_path(p) as tmp:
            out.write_parquet(tmp, **self.cfg.parquet.polars_kwargs())
        if c.kind == "dataset":
            atomic_write_json({"name": c.name, "description": c.description, "key": list(c.key), "rows": out.height,
                               "built_at": utc_now(), "reload": f"pl.read_parquet('{self.cfg.paths.rel(p)}')",
                               "columns": [{"name": k, "dtype": str(t), "description": c.docs.get(k, "")}
                                           for k, t in c.columns.items()]}, p.with_suffix(".schema.json"))
        m = self._manifest()
        m["tables"][c.name] = {"study": study, "kind": c.kind, "area": c.area, "path": self.cfg.paths.rel(p),
                               "rows": out.height, "description": c.description, "built_at": utc_now()}
        self._save_manifest(m)
        self._write_index("tables", self.tables_root / "table_index.csv")
        log.info("[studies:%s] %s: %s rows → %s", study, c.name, f"{out.height:,}", self.cfg.paths.rel(p))
        return out

    def read(self, c: TableContract) -> pl.DataFrame:
        p = self.path(c)
        if not p.is_file():
            raise FileNotFoundError(f"{self.cfg.paths.rel(p)} missing — run the study that produces {c.name}")
        return conform(pl.read_parquet(p), c)

    # ------------------------------------------------------------------ run cache
    def fingerprint(self, study: str, params: dict[str, Any], upstream: dict[str, Any]) -> str:
        return stable_hash({"study": study, "params": params, "upstream": upstream})

    def is_current(self, study: str, fp: str, contracts: Iterable[TableContract]) -> tuple[bool, str]:
        entry = self._manifest()["runs"].get(study)
        if not entry:
            return False, "never run"
        if entry.get("fingerprint") != fp:
            return False, "parameters or upstream datasets changed"
        missing = [self.cfg.paths.rel(p) for c in contracts for p in self.paths(c) if not p.is_file()]
        return (False, f"outputs missing {missing}") if missing else (True, "current")

    def require_run(self, study: str, needed_by: str, command: str) -> dict:
        """Run record of a prerequisite study. A study records its run only after EVERY output passed its
        contract, so a missing record means it never ran or failed part-way — the caller must not proceed."""
        entry = self._manifest()["runs"].get(study)
        if not entry:
            raise FileNotFoundError(
                f"{needed_by} needs a completed Study {study} run, but none is recorded in "
                f"{self.cfg.paths.rel(self.manifest_path)} (never run, or its last run failed before all outputs "
                f"passed their contracts). Run `{command}` and fix any error it reports first.")
        return entry

    def record_run(self, study: str, fp: str, contracts: Iterable[TableContract], extra: dict | None = None) -> None:
        m = self._manifest()
        m["runs"][study] = {"fingerprint": fp, "built_at": utc_now(), "outputs": [c.name for c in contracts],
                            **(extra or {})}
        self._save_manifest(m)

    # ------------------------------------------------------------------ figures
    def figure_dir(self, study: str) -> Path:
        return self.figures_root / STUDY_DIRS[study]

    def save_figure(self, fig, rec: FigureRecord, plot_data: dict[str, pl.DataFrame], n: str = "") -> Path:
        """Plot data first (figure redraws without recomputation), then PNG (300 DPI) + SVG (+ PDF)."""
        pd_dir = self.tables_root / "plot_data" / STUDY_DIRS[rec.study]
        pd_dir.mkdir(parents=True, exist_ok=True)
        data_files = []
        for part, df in plot_data.items():
            p = pd_dir / f"{rec.name}__{part}.csv"
            atomic_write_csv(df, p)
            data_files.append(self.cfg.paths.rel(p))
        formats = ("png", "svg", "pdf") if rec.publication else ("png", "svg")
        path = save_figure(fig, self.figure_dir(rec.study), rec.name, close=True, dpi=FIG_DPI, formats=formats,
                           metadata={"Source": rec.dataset, "N": n, "Question": rec.research_question})
        m = self._manifest()
        m["figures"][rec.name] = {**asdict(rec), "files": [self.cfg.paths.rel(path.with_suffix(f".{e}")) for e in formats],
                                  "plot_data": data_files, "n": n, "built_at": utc_now()}
        self._save_manifest(m)
        self._write_index("figures", self.figures_root / "figure_index.csv")
        return path

    def _write_index(self, kind: str, target: Path) -> None:
        entries = self._manifest()[kind]
        rows = [{"name": k, **{f: (", ".join(v) if isinstance(v, list) else ("" if v is None else str(v)))
                               for f, v in e.items()}} for k, e in sorted(entries.items())]
        if rows:
            cols = sorted({c for r in rows for c in r})
            target.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame([{c: r.get(c, "") for c in cols} for r in rows],
                         schema={c: pl.String for c in cols}).write_csv(target)

    def figure_index(self) -> pl.DataFrame:
        p = self.figures_root / "figure_index.csv"
        return pl.read_csv(p, infer_schema_length=0) if p.is_file() else pl.DataFrame()
