"""Step 0 — execution environment and resource discovery (read-only)."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil
import polars as pl

from .config import Phase01Config
from .reporting import MarkdownReport, export_table
from .utils import human_bytes

LIBRARIES = ["polars", "duckdb", "pyarrow", "pandas", "numpy", "matplotlib", "psutil", "pytest"]


class EnvironmentInspector:
    def __init__(self, cfg: Phase01Config):
        self.cfg = cfg
        self.facts: dict[str, Any] = {}

    # ------------------------------------------------------------------ collection
    @staticmethod
    def detect_platform() -> str:
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").exists():
            return "kaggle"
        if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
            return "colab"
        return "local"

    @staticmethod
    def library_versions() -> dict[str, str]:
        out = {}
        for lib in LIBRARIES:
            try:
                out[lib] = metadata.version(lib)
            except metadata.PackageNotFoundError:
                out[lib] = "not installed"
        return out

    def _disk(self, path: Path) -> dict[str, Any]:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        return {"path": self.cfg.paths.rel(path), "total": usage.total, "free": usage.free}

    def collect(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        p = self.cfg.paths
        raw_bytes = sum(f.stat().st_size for f in p.data_root.rglob("*") if f.is_file()) if p.data_root.exists() else 0
        self.facts = {
            "platform": self.detect_platform(),
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "python_executable": Path(sys.executable).name,
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "ram_total": vm.total,
            "ram_available": vm.available,
            "data_root": p.rel(p.data_root),
            "data_root_exists": p.data_root.exists(),
            "raw_data_bytes": raw_bytes,
            "disk_data": self._disk(p.data_root),
            "disk_warehouse": self._disk(p.warehouse_root),
            "libraries": self.library_versions(),
            "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS", "unset"),
            "polars_thread_pool_size": pl.thread_pool_size(),
        }
        return self.facts

    # ------------------------------------------------------------------ recommendations
    def recommendations(self) -> dict[str, str]:
        f = self.facts or self.collect()
        threads = self.cfg.resources.resolved_threads()
        raw = f["raw_data_bytes"]
        ram = f["ram_total"]
        free_wh = f["disk_warehouse"]["free"]
        # Rule of thumb: bronze+silver+gold Parquet ≈ 0.6-1.0x raw CSV; spill can reach ~1x the
        # largest table. These are planning estimates, replaced by measured sizes after bronze.
        need = raw * 2.0
        return {
            "polars_threads": f"POLARS_MAX_THREADS={threads} (logical cores - 1; leaves one core for OS/IO)",
            "duckdb_threads": f"SET threads={threads}",
            "duckdb_memory_limit": f"SET memory_limit='{self.cfg.resources.resolved_duckdb_memory()}' "
                                   f"({self.cfg.resources.duckdb_memory_fraction:.0%} of {human_bytes(ram)}; never 100%)",
            "duckdb_temp_directory": f"{self.cfg.paths.rel(self.cfg.paths.duckdb_spill)} (git-ignored; same disk as warehouse)",
            "preserve_insertion_order": "false for COPY/aggregates (lower memory); ordering is made explicit with ORDER BY / _src_row",
            "parquet": f"{self.cfg.parquet.compression} level {self.cfg.parquet.compression_level}, "
                       f"row groups of {self.cfg.parquet.row_group_size:,} rows; one file per partition",
            "streaming": "Polars: scan_csv/scan_parquet + sink_parquet (streaming); collect only aggregates or head(). "
                         "DuckDB: COPY (SELECT …) TO … for out-of-core materialisation.",
            "raw_vs_ram": f"raw data {human_bytes(raw)} vs RAM {human_bytes(ram)} "
                          f"(ratio {raw / ram:.2f}) — {'fits comfortably; out-of-core still used for safety' if raw < 0.3 * ram else 'must stay out-of-core'}",
            "disk_headroom": f"estimated need ≈ {human_bytes(need)} for warehouse + spill; free {human_bytes(free_wh)} "
                             f"— {'OK' if free_wh > need else 'INSUFFICIENT: set NFL_BDB_WAREHOUSE / NFL_BDB_TMP to a larger disk'}",
            "distributed": "Not recommended unless measured peak memory or runtimes exceed limits (see 07_performance_benchmarks.md).",
        }

    # ------------------------------------------------------------------ outputs
    def write_report(self) -> Path:
        f = self.facts or self.collect()
        rec = self.recommendations()
        p = self.cfg.paths
        flat = {k: v for k, v in f.items() if not isinstance(v, dict)}
        flat.update({f"lib_{k}": v for k, v in f["libraries"].items()})
        export_table(pl.DataFrame({"item": list(flat), "value": [str(v) for v in flat.values()]}),
                     p.tables, "environment_facts")

        rpt = MarkdownReport("00 — Environment report", "Step 0: resources, versions and recommended engine settings")
        rpt.h2("Execution environment").kv({
            "Platform": f["platform"], "OS": f["os"], "Python": f["python"],
            "CPU (logical / physical)": f"{f['cpu_logical']} / {f['cpu_physical']}",
            "RAM total / available": f"{human_bytes(f['ram_total'])} / {human_bytes(f['ram_available'])}",
            "DATA_ROOT": f"{f['data_root']} (exists: {f['data_root_exists']})",
            "Raw data on disk": human_bytes(f["raw_data_bytes"]),
            "Disk free (data)": human_bytes(f["disk_data"]["free"]),
            "Disk free (warehouse)": human_bytes(f["disk_warehouse"]["free"]),
            "POLARS_MAX_THREADS / pool": f"{f['POLARS_MAX_THREADS']} / {f['polars_thread_pool_size']}",
        })
        rpt.h2("Library versions").kv(f["libraries"])
        rpt.h2("Recommended settings").kv(rec)
        rpt.h2("Configuration").p(
            "All paths derive from the project root or from `configs/phase01.toml`; override with "
            "`NFL_BDB_DATA_ROOT`, `NFL_BDB_WAREHOUSE`, `NFL_BDB_REPORTS`, `NFL_BDB_TMP`, `NFL_BDB_THREADS`, "
            "`NFL_BDB_DUCKDB_MEMORY`. Call `configure_runtime(cfg)` before importing Polars."
        )
        return rpt.save(p.reports_root / "00_environment_report.md")
