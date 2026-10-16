"""DuckDB catalog — a small ``.duckdb`` file containing only VIEWS over the Parquet layers.

Views embed absolute paths, so the catalog is machine-specific and git-ignored; rebuilding it
is instantaneous (``WarehouseCatalog(cfg).build()``). Naming: ``bronze_<t>``, ``silver_<t>``,
``gold_<t>`` plus the semantic aliases ``fact_nfl_tracking`` and ``fact_combine_tracking``
(silver frame-level tables at their natural grain).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .config import Phase01Config, duckdb_connect
from .gold import GOLD_TABLES
from .reporting import export_table
from .tables import LAYERS, LayerRefs
from .utils import get_logger, human_bytes

log = get_logger(__name__)

ALIASES = {"fact_nfl_tracking": ("silver", "game_tracking"), "fact_combine_tracking": ("silver", "combine_tracking")}


class WarehouseCatalog:
    def __init__(self, cfg: Phase01Config):
        self.cfg = cfg
        self.refs = LayerRefs(cfg)
        self.db_path = cfg.paths.catalog / "bdb27.duckdb"

    def discover_tables(self) -> list[tuple[str, str]]:
        found = []
        for layer in LAYERS:
            base: Path = getattr(self.cfg.paths, layer)
            if not base.exists():
                continue
            for p in sorted(base.iterdir()):
                if p.name.startswith((".", "_", "phase_")):  # phase_NN dirs hold separately catalogued datasets
                    continue
                name = p.stem if p.suffix == ".parquet" else p.name
                if self.refs.exists(layer, name):
                    found.append((layer, name))
        return found

    def build(self) -> pl.DataFrame:
        con = duckdb_connect(self.cfg, self.db_path)
        rows = []
        try:
            for layer, name in self.discover_tables():
                view = f"{layer}_{name}"
                con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM {self.refs.sql(layer, name)}")
                n_cols = len(con.sql(f"SELECT * FROM {view} LIMIT 0").columns)
                spec = GOLD_TABLES.get(name) if layer == "gold" else None
                rows.append({
                    "view": view, "layer": layer, "table": name, "files": len(self.refs.files(layer, name)),
                    "rows": self.refs.num_rows(layer, name), "bytes": self.refs.num_bytes(layer, name),
                    "size": human_bytes(self.refs.num_bytes(layer, name)), "n_columns": n_cols,
                    "grain": spec.grain if spec else "", "pk": ", ".join(spec.pk) if spec else "",
                    "path": self.cfg.paths.rel(self.refs.path(layer, name)),
                })
            for alias, (layer, name) in ALIASES.items():
                if self.refs.exists(layer, name):
                    con.execute(f"CREATE OR REPLACE VIEW {alias} AS SELECT * FROM {layer}_{name}")
            rows += self._phase_views(con)
        finally:
            con.close()
        df = pl.DataFrame(rows)
        if not df.is_empty():
            export_table(df, self.cfg.paths.catalog, "table_catalog")
            export_table(df, self.cfg.paths.tables, "table_catalog")
        log.info("catalog %s: %d views", self.cfg.paths.rel(self.db_path), len(rows))
        return df

    def _phase_views(self, con) -> list[dict]:
        """One view per dataset registered in ``gold/phase_NN/_manifest.json`` → ``pNN_<name>``."""
        from .utils import read_json, sql_path

        rows = []
        for d in sorted(self.cfg.paths.gold.glob("phase_*")):
            manifest = read_json(d / "_manifest.json", {}).get("datasets", {})
            prefix = "p" + d.name.split("_")[-1]
            for name, entry in manifest.items():
                files = sorted(p for p in ([d / f"{name}.parquet"] if (d / f"{name}.parquet").is_file()
                                           else (d / name).rglob("*.parquet")) if not p.name.startswith("."))
                if not files:
                    continue
                listing = ", ".join(f"'{sql_path(f)}'" for f in files)
                view = f"{prefix}_{name}"
                con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM "
                            f"read_parquet([{listing}], hive_partitioning = true)")
                rows.append({"view": view, "layer": f"gold/{d.name}", "table": name, "files": len(files),
                             "rows": entry.get("rows"), "bytes": sum(f.stat().st_size for f in files),
                             "size": human_bytes(sum(f.stat().st_size for f in files)),
                             "n_columns": len(entry.get("schema", {})), "grain": entry.get("grain", ""),
                             "pk": ", ".join(entry.get("key", [])), "path": entry.get("path", "")})
        return rows

    def connect(self, read_only: bool = True):
        if not self.db_path.exists():
            raise FileNotFoundError(f"{self.db_path} missing — run WarehouseCatalog(cfg).build()")
        return duckdb_connect(self.cfg, self.db_path, read_only=read_only)
