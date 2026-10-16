"""Configuration for the Phase 01 data layer.

Resolution order (later wins): dataclass defaults -> ``configs/phase01.toml`` -> environment
variables (``NFL_BDB_*``) -> explicit keyword overrides passed to :meth:`Phase01Config.load`.

Nothing in the code base hardcodes machine-specific absolute paths: every path is derived
from the project root (auto-detected) or from the variables above.

IMPORTANT: Polars reads ``POLARS_MAX_THREADS`` once, at import time. Call
:func:`configure_runtime` *before* the first ``import polars`` in a process (the notebook and
the pipeline CLI do this) or the thread setting is silently ignored.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ENV_PREFIX = "NFL_BDB_"


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards until a directory containing ``pyproject.toml`` and ``src/`` is found."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return Path.cwd().resolve()


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            warnings.warn(f"Cannot parse {path}: install 'tomli' on Python 3.10. Using defaults.")
            return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


# --------------------------------------------------------------------------------------
# Settings groups
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PathSettings:
    project_root: Path
    data_root: Path
    warehouse_root: Path
    reports_root: Path
    tmp_root: Path

    # Derived locations -------------------------------------------------------------
    @property
    def tables(self) -> Path:
        return self.reports_root / "tables"

    @property
    def figures(self) -> Path:
        return self.reports_root / "figures"

    @property
    def diagrams(self) -> Path:
        return self.reports_root / "diagrams"

    @property
    def bronze(self) -> Path:
        return self.warehouse_root / "bronze"

    @property
    def silver(self) -> Path:
        return self.warehouse_root / "silver"

    @property
    def gold(self) -> Path:
        return self.warehouse_root / "gold"

    @property
    def catalog(self) -> Path:
        return self.warehouse_root / "catalog"

    @property
    def duckdb_spill(self) -> Path:
        return self.tmp_root / "duckdb_spill"

    @property
    def staging(self) -> Path:
        return self.tmp_root / "staging"

    def ensure(self) -> None:
        """Create every *output* directory. The raw data root is never created or touched."""
        for p in (
            self.tables, self.figures, self.diagrams, self.bronze, self.silver,
            self.gold, self.catalog, self.duckdb_spill, self.staging,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path | str) -> str:
        """Project-relative POSIX path for reports (keeps reports machine-independent)."""
        p = Path(path).resolve()
        try:
            return p.relative_to(self.project_root).as_posix()
        except ValueError:
            return p.as_posix()


@dataclass(frozen=True)
class ResourceSettings:
    threads: int = 0                       # 0 -> auto
    duckdb_memory_fraction: float = 0.5
    duckdb_memory_limit: str = ""          # explicit, e.g. "8GB"

    def resolved_threads(self) -> int:
        if self.threads and self.threads > 0:
            return self.threads
        return max(1, (os.cpu_count() or 2) - 1)

    def resolved_duckdb_memory(self) -> str:
        if self.duckdb_memory_limit:
            return self.duckdb_memory_limit
        import psutil

        total_gb = psutil.virtual_memory().total / 1024**3
        frac = min(max(self.duckdb_memory_fraction, 0.1), 0.8)  # never the whole machine
        return f"{max(1.0, total_gb * frac):.1f}GB"


@dataclass(frozen=True)
class ParquetSettings:
    compression: str = "zstd"
    compression_level: int = 3
    row_group_size: int = 500_000

    def duckdb_copy_options(self) -> str:
        return (
            f"FORMAT parquet, COMPRESSION {self.compression}, "
            f"COMPRESSION_LEVEL {self.compression_level}, ROW_GROUP_SIZE {self.row_group_size}"
        )

    def polars_kwargs(self) -> dict[str, Any]:
        return {
            "compression": self.compression,
            "compression_level": self.compression_level,
            "row_group_size": self.row_group_size,
            "statistics": True,
        }


@dataclass(frozen=True)
class ProfilingSettings:
    schema_sample_rows: int = 50_000
    offset_sample_chunks: int = 20
    offset_sample_lines: int = 2_000
    row_estimate_bytes: int = 8 * 1024**2
    tracking_sample_pct: float = 2.0
    seed: int = 42
    categorical_max_distinct: int = 60
    top_k: int = 15


@dataclass(frozen=True)
class TrackingSettings:
    nominal_interval_ms: int = 100
    field_length_yds: float = 120.0
    field_width_yds: float = 53.3
    max_speed_yds: float = 12.5
    max_accel_yds2: float = 15.0
    max_step_yds: float = 1.5
    coord_tolerance_yds: float = 10.0
    gap_factor: float = 1.5


@dataclass(frozen=True)
class SilverSettings:
    cluster_sort: bool = True
    # Fraction of key-conflicting rows above which the silver build aborts.
    max_key_conflict_fraction: float = 1e-4


# --------------------------------------------------------------------------------------
# Top-level config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase01Config:
    paths: PathSettings
    resources: ResourceSettings = field(default_factory=ResourceSettings)
    parquet: ParquetSettings = field(default_factory=ParquetSettings)
    profiling: ProfilingSettings = field(default_factory=ProfilingSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    silver: SilverSettings = field(default_factory=SilverSettings)

    @classmethod
    def load(
        cls,
        project_root: Path | str | None = None,
        config_file: Path | str | None = None,
        **overrides: Any,
    ) -> "Phase01Config":
        """Build a config. ``overrides`` may set ``data_root``, ``warehouse_root``,
        ``reports_root``, ``tmp_root`` or ``threads`` directly (handy in tests)."""
        root = Path(project_root).resolve() if project_root else find_project_root()
        toml = _load_toml(Path(config_file) if config_file else root / "configs" / "phase01.toml")
        tp, tr = toml.get("paths", {}), toml.get("resources", {})

        env_names = {
            "data_root": "DATA_ROOT", "warehouse_root": "WAREHOUSE",
            "reports_root": "REPORTS", "tmp_root": "TMP",
        }

        def path_value(key: str, default: str) -> Path:
            raw = overrides.get(key) or os.environ.get(ENV_PREFIX + env_names[key]) or tp.get(key) or default
            p = Path(raw).expanduser()
            return p if p.is_absolute() else (root / p)

        paths = PathSettings(
            project_root=root,
            data_root=path_value("data_root", "data"),
            warehouse_root=path_value("warehouse_root", "warehouse"),
            reports_root=path_value("reports_root", "reports"),
            tmp_root=path_value("tmp_root", ".tmp"),
        )
        resources = ResourceSettings(
            threads=int(overrides.get("threads") or os.environ.get(ENV_PREFIX + "THREADS") or tr.get("threads", 0)),
            duckdb_memory_fraction=float(tr.get("duckdb_memory_fraction", 0.5)),
            duckdb_memory_limit=str(os.environ.get(ENV_PREFIX + "DUCKDB_MEMORY") or tr.get("duckdb_memory_limit", "")),
        )

        def group(klass, key):
            base = klass()
            vals = {k: v for k, v in toml.get(key, {}).items() if hasattr(base, k)}
            return replace(base, **vals)

        return cls(
            paths=paths,
            resources=resources,
            parquet=group(ParquetSettings, "parquet"),
            profiling=group(ProfilingSettings, "profiling"),
            tracking=group(TrackingSettings, "tracking"),
            silver=group(SilverSettings, "silver"),
        )


# --------------------------------------------------------------------------------------
# Runtime wiring
# --------------------------------------------------------------------------------------


def configure_runtime(cfg: Phase01Config) -> dict[str, str]:
    """Apply process-level settings. Must run before ``import polars`` to take effect."""
    threads = str(cfg.resources.resolved_threads())
    if "polars" in sys.modules and os.environ.get("POLARS_MAX_THREADS") != threads:
        warnings.warn(
            "polars was imported before configure_runtime(); POLARS_MAX_THREADS will not apply "
            "to this process. Restart the kernel and call configure_runtime() first."
        )
    os.environ.setdefault("POLARS_MAX_THREADS", threads)
    cfg.paths.ensure()
    return {"POLARS_MAX_THREADS": os.environ["POLARS_MAX_THREADS"]}


def duckdb_connect(cfg: Phase01Config, database: str | Path = ":memory:", read_only: bool = False):
    """DuckDB connection with bounded memory, explicit spill directory and thread count."""
    import duckdb

    con = duckdb.connect(str(database), read_only=read_only)
    spill = cfg.paths.duckdb_spill
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET threads = {cfg.resources.resolved_threads()}")
    con.execute(f"SET memory_limit = '{cfg.resources.resolved_duckdb_memory()}'")
    con.execute(f"SET temp_directory = '{spill.as_posix()}'")
    # Large COPY / aggregate jobs do not need input order; dropping it lowers memory use.
    con.execute("SET preserve_insertion_order = false")
    return con
