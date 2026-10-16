"""Phase 02 artifact persistence: versioned Parquet datasets, CSV tables and figures.

Every persisted dataset is described by a :class:`DatasetSpec` (grain, key, sources, column
documentation, version) and recorded in ``warehouse/gold/phase_02/_manifest.json``.

Reuse rule — a dataset is reused only when ALL hold (file existence alone is never enough):

* a manifest entry exists and the Parquet file(s) exist;
* the spec ``version`` equals the recorded version (code/logic change → rebuild);
* the input fingerprint (size + mtime of every source file, plus build parameters) is unchanged;
* the on-disk Parquet schema equals the recorded schema.

When a dataset must be rebuilt and a previous version exists, the old files are moved to
``_archive/`` (never silently overwritten). ``force=True`` rebuilds regardless.

Each dataset also gets a sidecar ``<name>.schema.json`` with grain, key, sources, column
descriptions, dtypes, row count, build metadata and reload instructions.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import polars as pl

from ..data.config import Phase01Config
from ..data.reporting import save_figure
from ..data.tables import LAYERS, LayerRefs
from ..data.utils import (atomic_directory, atomic_path, atomic_write_csv, atomic_write_json, file_fingerprint,
                          get_logger, read_json, stable_hash, utc_now)

log = get_logger(__name__)

PHASE = "phase_02"
FIG_DPI = 300


class ArtifactValidationError(RuntimeError):
    """A built dataset violates its declared contract (key uniqueness, nulls in key, columns)."""


@dataclass(frozen=True)
class DatasetSpec:
    """Contract of a persisted dataset.

    ``sources`` entries: ``"<layer>.<table>"`` (bronze/silver/gold), ``"table:<stem>"``
    (reports/tables/<stem>.parquet), ``"phase:<name>"`` (another Phase 02 dataset) or
    ``"file:<project-relative path>"``.
    """

    name: str
    grain: str
    key: tuple[str, ...]
    sources: tuple[str, ...]
    description: str
    columns: dict[str, str] = field(default_factory=dict)   # column → description (with units)
    version: str = "1"
    partition_by: str | None = None                          # low-cardinality column only

    def reload_snippet(self) -> str:
        return (f"from nflbdb.research.artifacts import ArtifactStore\n"
                f"lf = ArtifactStore(cfg).load('{self.name}')  # Polars LazyFrame")


class ArtifactStore:
    """Single entry point for Phase 02 outputs (datasets, CSV tables, figures)."""

    def __init__(self, cfg: Phase01Config, phase: str = PHASE):
        self.cfg = cfg
        self.phase = phase
        self.refs = LayerRefs(cfg)
        self.figures_dir = cfg.paths.figures / phase
        self.tables_dir = cfg.paths.tables / phase
        self.data_dir = cfg.paths.gold / phase
        self.archive_dir = self.data_dir / "_archive"
        self.manifest_path = self.data_dir / "_manifest.json"

    # ------------------------------------------------------------------ paths
    def ensure_dirs(self) -> None:
        for d in (self.figures_dir, self.tables_dir, self.data_dir):
            d.mkdir(parents=True, exist_ok=True)

    def dataset_path(self, spec_or_name: DatasetSpec | str) -> Path:
        name = spec_or_name if isinstance(spec_or_name, str) else spec_or_name.name
        partitioned = isinstance(spec_or_name, DatasetSpec) and spec_or_name.partition_by
        entry = self._manifest().get("datasets", {}).get(name, {})
        if partitioned or entry.get("partition_by"):
            return self.data_dir / name
        return self.data_dir / f"{name}.parquet"

    def dataset_files(self, name: str) -> list[Path]:
        single = self.data_dir / f"{name}.parquet"
        if single.is_file():
            return [single]
        d = self.data_dir / name
        if not d.is_dir():
            return []
        return sorted(p for p in d.rglob("*.parquet") if not any(x.startswith(".") for x in p.relative_to(d).parts))

    # ------------------------------------------------------------------ manifest
    def _manifest(self) -> dict:
        return read_json(self.manifest_path, {"datasets": {}, "tables": {}, "figures": {}})

    def _write_manifest(self, m: dict) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(m, self.manifest_path)

    def _register(self, kind: str, name: str, entry: dict) -> None:
        m = self._manifest()
        m.setdefault(kind, {})[name] = entry
        self._write_manifest(m)

    def manifest_frame(self) -> pl.DataFrame:
        """All registered artifacts as one table (kind, name, path, built_at, rows, status…)."""
        m, rows = self._manifest(), []
        for kind in ("datasets", "tables", "figures"):
            for name, e in m.get(kind, {}).items():
                rows.append({"kind": kind, "name": name, "path": e.get("path", ""), "built_at": e.get("built_at", ""),
                             "rows": e.get("rows"), "version": e.get("version", ""),
                             "grain": e.get("grain", ""), "key": ", ".join(e.get("key", []))})
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()

    # ------------------------------------------------------------------ fingerprints
    def source_files(self, source: str) -> list[Path]:
        if source.startswith("table:"):
            return [self.cfg.paths.tables / f"{source[6:]}.parquet"]
        if source.startswith("phase:"):
            return self.dataset_files(source[6:])
        if source.startswith("file:"):
            return [self.cfg.paths.project_root / source[5:]]
        layer, _, name = source.partition(".")
        if layer not in LAYERS or not name:
            raise ValueError(f"bad source reference {source!r}")
        return self.refs.files(layer, name)

    def missing_sources(self, spec: DatasetSpec) -> list[str]:
        return [s for s in spec.sources if not any(p.exists() for p in self.source_files(s))]

    def input_fingerprint(self, spec: DatasetSpec, params: dict[str, Any] | None = None) -> str:
        files = {}
        for s in spec.sources:
            files[s] = [[self.cfg.paths.rel(p), *file_fingerprint(p).values()] for p in self.source_files(s) if p.exists()]
        return stable_hash({"sources": files, "params": params or {}, "version": spec.version})

    @staticmethod
    def _schema_of(files: Sequence[Path]) -> dict[str, str]:
        if not files:
            return {}
        return {c: str(t) for c, t in pl.read_parquet_schema(files[0]).items()}

    # ------------------------------------------------------------------ status
    def status(self, spec: DatasetSpec, params: dict[str, Any] | None = None) -> tuple[bool, str]:
        """(is_current, reason). Reasons are human-readable and logged on every decision."""
        entry = self._manifest().get("datasets", {}).get(spec.name)
        files = self.dataset_files(spec.name)
        if not files:
            return False, "missing on disk"
        if not entry:
            return False, "files exist but no manifest entry (unknown provenance)"
        if entry.get("version") != spec.version:
            return False, f"version changed {entry.get('version')} → {spec.version}"
        missing = self.missing_sources(spec)
        if missing:
            return False, f"source(s) missing: {missing}"
        if entry.get("input_fingerprint") != self.input_fingerprint(spec, params):
            return False, "inputs or parameters changed"
        if entry.get("schema") != self._schema_of(files):
            return False, "on-disk schema differs from manifest"
        return True, "current"

    # ------------------------------------------------------------------ build / load
    def build(self, spec: DatasetSpec, builder: Callable[[], pl.DataFrame | pl.LazyFrame],
              params: dict[str, Any] | None = None, force: bool = False,
              validate: Callable[[pl.LazyFrame], list[str]] | None = None) -> pl.LazyFrame:
        """Reuse ``spec`` if current, otherwise run ``builder`` and persist atomically.

        ``builder`` may return a DataFrame (compact outputs) or a LazyFrame (streamed to
        Parquet with ``sink_parquet``; not supported together with ``partition_by``).
        ``validate`` may return a list of contract violations → :class:`ArtifactValidationError`.
        """
        self.ensure_dirs()
        current, reason = self.status(spec, params)
        if current and not force:
            log.info("[artifact] reuse %s (%s)", spec.name, reason)
            return self.load(spec.name)
        missing = self.missing_sources(spec)
        if missing:
            raise FileNotFoundError(f"{spec.name}: upstream source(s) missing {missing} — build them first")
        log.info("[artifact] build %s (%s%s)", spec.name, "forced; " if force else "", reason)

        t0 = time.perf_counter()
        result = builder()
        self._archive_previous(spec, reason if not force else "forced rebuild")
        target = self.dataset_path(spec)
        if spec.partition_by:
            df = result.collect(engine="streaming") if isinstance(result, pl.LazyFrame) else result
            self._write_partitioned(df, target, spec.partition_by)
        elif isinstance(result, pl.LazyFrame):
            with atomic_path(target) as tmp:
                result.sink_parquet(tmp, **self.cfg.parquet.polars_kwargs())
        else:
            with atomic_path(target) as tmp:
                result.write_parquet(tmp, **self.cfg.parquet.polars_kwargs())

        files = self.dataset_files(spec.name)
        lf = self.load(spec.name)
        issues = self.check_contract(spec, lf)
        if validate:
            issues += validate(lf)
        if issues:
            bad = target.with_name(f"{target.stem}.INVALID")
            shutil.rmtree(bad, ignore_errors=True) if bad.is_dir() else bad.unlink(missing_ok=True)
            os.replace(target, bad)
            raise ArtifactValidationError(f"{spec.name}: {issues} (output moved to {self.cfg.paths.rel(bad)})")

        rows = lf.select(pl.len()).collect().item()
        schema = self._schema_of(files)
        undocumented = [c for c in schema if c not in spec.columns]
        if undocumented:
            log.warning("[artifact] %s: undocumented columns %s", spec.name, undocumented)
        entry = {
            "path": self.cfg.paths.rel(target), "version": spec.version, "built_at": utc_now(),
            "seconds": round(time.perf_counter() - t0, 2), "rows": rows, "grain": spec.grain,
            "key": list(spec.key), "sources": list(spec.sources), "partition_by": spec.partition_by,
            "input_fingerprint": self.input_fingerprint(spec, params), "params": params or {}, "schema": schema,
        }
        self._register("datasets", spec.name, entry)
        self._write_sidecar(spec, entry)
        log.info("[artifact] %s: %s rows → %s", spec.name, f"{rows:,}", entry["path"])
        return lf

    def load(self, name: str, spec: DatasetSpec | None = None, require_current: bool = False,
             params: dict[str, Any] | None = None) -> pl.LazyFrame:
        """Lazy scan of a persisted dataset. With ``require_current`` (and ``spec``) a stale
        dataset raises instead of being used silently."""
        files = self.dataset_files(name)
        if not files:
            raise FileNotFoundError(f"dataset {name!r} not built under {self.cfg.paths.rel(self.data_dir)}")
        if spec is not None:
            current, reason = self.status(spec, params)
            if not current:
                if require_current:
                    raise RuntimeError(f"dataset {name!r} is stale: {reason}")
                log.warning("[artifact] %s is stale (%s) — loading anyway", name, reason)
        return pl.scan_parquet(files, hive_partitioning=True)

    # ------------------------------------------------------------------ contract
    @staticmethod
    def check_contract(spec: DatasetSpec, lf: pl.LazyFrame) -> list[str]:
        names = lf.collect_schema().names()
        issues = [f"key column missing: {c}" for c in spec.key if c not in names]
        if issues or not spec.key:
            return issues
        key = list(spec.key)
        stats = lf.select(
            pl.len().alias("n"),
            pl.struct(key).n_unique().alias("n_keys"),
            pl.any_horizontal([pl.col(c).is_null() for c in key]).sum().alias("null_keys"),
        ).collect().row(0, named=True)
        if stats["n"] != stats["n_keys"]:
            issues.append(f"key {key} not unique: {stats['n'] - stats['n_keys']:,} duplicate rows")
        if stats["null_keys"]:
            issues.append(f"{stats['null_keys']:,} rows with null key values")
        return issues

    # ------------------------------------------------------------------ helpers
    def _write_partitioned(self, df: pl.DataFrame, target: Path, col: str) -> None:
        values = df.get_column(col).unique().sort().to_list()
        if len(values) > 200:
            raise ValueError(f"partition column {col!r} has {len(values)} values — too many files")
        if df.is_empty():  # keep an explicit, schema-carrying empty dataset (no partitions to write)
            with atomic_directory(target) as tmp:
                df.write_parquet(tmp / "part-empty.parquet", **self.cfg.parquet.polars_kwargs())
            return
        with atomic_directory(target) as tmp:
            for v in values:
                part = df.filter(pl.col(col).is_null() if v is None else pl.col(col) == v).drop(col)
                d = tmp / f"{col}={'__null__' if v is None else v}"
                d.mkdir(parents=True, exist_ok=True)
                part.write_parquet(d / "part-0.parquet", **self.cfg.parquet.polars_kwargs())

    def _archive_previous(self, spec: DatasetSpec, reason: str) -> None:
        files = self.dataset_files(spec.name)
        if not files:
            return
        old = self._manifest().get("datasets", {}).get(spec.name, {})
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = self.data_dir / spec.name if (self.data_dir / spec.name).is_dir() else files[0]
        dest = self.archive_dir / f"{spec.name}__v{old.get('version', 'unknown')}__{stamp}{'' if src.is_dir() else '.parquet'}"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        log.info("[artifact] archived previous %s → %s (%s)", spec.name, self.cfg.paths.rel(dest), reason)

    def _write_sidecar(self, spec: DatasetSpec, entry: dict) -> None:
        doc = {
            "name": spec.name, "description": spec.description, "grain": spec.grain, "key": list(spec.key),
            "sources": list(spec.sources), "version": spec.version, "partition_by": spec.partition_by,
            "built_at": entry["built_at"], "rows": entry["rows"], "path": entry["path"],
            "columns": [{"name": c, "dtype": t, "description": spec.columns.get(c, "")}
                        for c, t in entry["schema"].items()],
            "reload": spec.reload_snippet(),
        }
        atomic_write_json(doc, self.data_dir / f"{spec.name}.schema.json")

    # ------------------------------------------------------------------ tables & figures
    def export_csv(self, df: pl.DataFrame, name: str, description: str = "", parquet: bool = False) -> Path:
        """Small human-readable result → ``reports/tables/phase_02/<name>.csv`` (+ optional Parquet)."""
        if df.height > 1_000_000:
            raise ValueError(f"{name}: {df.height:,} rows is too large for CSV — persist as a dataset instead")
        self.ensure_dirs()
        path = atomic_write_csv(df, self.tables_dir / f"{name}.csv")
        if parquet:
            df.write_parquet(self.tables_dir / f"{name}.parquet")
        self._register("tables", name, {"path": self.cfg.paths.rel(path), "rows": df.height,
                                        "columns": df.columns, "description": description, "built_at": utc_now()})
        return path

    def save_fig(self, fig, name: str, formats: Sequence[str] = ("png", "svg"), dpi: int = FIG_DPI,
                 source: str = "", n: int | str | None = None, close: bool = True) -> Path:
        """Save a Matplotlib figure (300 DPI PNG + SVG by default) and register it."""
        self.ensure_dirs()
        meta = {k: str(v) for k, v in {"Source": source, "N": n, "Phase": self.phase}.items() if v not in (None, "")}
        path = save_figure(fig, self.figures_dir, name, close=close, dpi=dpi, formats=formats, metadata=meta)
        self._register("figures", name, {"path": self.cfg.paths.rel(path), "formats": list(formats), "dpi": dpi,
                                         "source": source, "n": None if n is None else str(n), "built_at": utc_now()})
        return path
