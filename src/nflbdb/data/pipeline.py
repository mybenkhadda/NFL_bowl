"""Phase 01 orchestrator + CLI.

Modes
* ``discovery`` (default) — bounded: inventory/schema on byte-offset samples, profiling and
  tracking QC on deterministic samples, validation exact for small tables and sampled for
  tracking tables.
* ``full`` — exact validation / QC over complete tables.

Expensive steps (bronze, silver, gold, benchmark, and any full-mode scan of tracking) print a
cost estimate and are SKIPPED unless ``allow_expensive=True`` (CLI: ``--yes``).

Examples::

    python -m nflbdb.data.pipeline --steps env,inventory,schema
    python -m nflbdb.data.pipeline --steps bronze --yes
    python -m nflbdb.data.pipeline --steps profile,validate,tracking_qc,relationships
    python -m nflbdb.data.pipeline --steps silver,gold,catalog --yes
    python -m nflbdb.data.pipeline --steps benchmark,feasibility --yes
"""

from __future__ import annotations

import argparse
import sys

from .config import Phase01Config, configure_runtime

STEPS = ["env", "inventory", "schema", "bronze", "profile", "tracking_qc", "validate", "relationships",
         "silver", "gold", "catalog", "benchmark", "feasibility"]
ALWAYS_EXPENSIVE = {"bronze", "silver", "gold", "benchmark"}
EXPENSIVE_IN_FULL = {"validate", "tracking_qc"}


class Phase01Pipeline:
    def __init__(self, cfg: Phase01Config, mode: str = "discovery", allow_expensive: bool = False,
                 force: bool = False):
        if mode not in ("discovery", "full"):
            raise ValueError("mode must be 'discovery' or 'full'")
        configure_runtime(cfg)
        from .utils import RunLog, get_logger

        self.cfg, self.mode, self.allow_expensive, self.force = cfg, mode, allow_expensive, force
        self.log = get_logger("nflbdb.pipeline")
        self.runlog = RunLog()
        self._con = None
        self.inventory = None

    @property
    def con(self):
        if self._con is None:
            from .config import duckdb_connect

            self._con = duckdb_connect(self.cfg)
        return self._con

    # ------------------------------------------------------------------ gating
    def _is_expensive(self, step: str) -> bool:
        return step in ALWAYS_EXPENSIVE or (self.mode == "full" and step in EXPENSIVE_IN_FULL)

    def _gate(self, step: str) -> bool:
        if not self._is_expensive(step):
            return True
        self.log.info("[%s] cost estimate:\n%s", step, self.estimate(step))
        if not self.allow_expensive:
            self.log.warning("[%s] SKIPPED: expensive step. Re-run with allow_expensive=True (CLI --yes).", step)
            return False
        return True

    def estimate(self, step: str):
        import polars as pl

        from .bronze import BronzeIngestor
        from .tables import LayerRefs
        from .utils import human_bytes

        if step == "bronze":
            return BronzeIngestor(self.cfg).estimate(self.inventory)
        refs = LayerRefs(self.cfg)
        layer = {"silver": "bronze", "gold": "silver"}.get(step, "bronze")
        rows = [{"input_layer": layer, "table": t, "rows": refs.num_rows(layer, t),
                 "size": human_bytes(refs.num_bytes(layer, t))}
                for t in ("game_tracking", "combine_tracking", "player_play") if refs.exists(layer, t)]
        return pl.DataFrame(rows)

    # ------------------------------------------------------------------ steps
    def step_env(self, m):
        from .environment import EnvironmentInspector

        m.add_artifact(EnvironmentInspector(self.cfg).write_report())

    def step_inventory(self, m):
        from .discover import DataInventory

        inv = DataInventory(self.cfg)
        df = inv.run()
        self.inventory = df
        m.rows_out, m.bytes_in = df.height, int(df["size_bytes"].sum())
        for p in inv.export(df):
            m.add_artifact(p)
        m.add_artifact(inv.write_report(df, inv.plot(df)))
        for r in df.filter(df["schema_drift"] != "").iter_rows(named=True):
            m.warn(f"schema drift {r['rel_path']}: {r['schema_drift']}")

    def step_schema(self, m):
        from .schema import SchemaCatalog

        sc = SchemaCatalog(self.cfg)
        cat, samples = sc.build()
        m.rows_in = sum(s.height for s in samples.values())
        m.rows_out = cat.height
        for p in sc.export(cat):
            m.add_artifact(p)
        m.add_artifact(sc.write_report(cat))

    def step_bronze(self, m):
        from .bronze import BronzeIngestor

        b = BronzeIngestor(self.cfg)
        b.build_manifest(with_sha256=True)
        df = b.ingest(force=self.force)
        if not df.is_empty():
            m.rows_out, m.bytes_in, m.bytes_out = int(df["rows"].sum()), int(df["size_bytes"].sum()), int(df["out_bytes"].sum())
        m.add_artifact(self.cfg.paths.bronze)

    def step_profile(self, m):
        from .profile import TableProfiler

        tp = TableProfiler(self.cfg, self.con)
        summary, top = tp.profile_all()
        part = tp.combine_test_participation() if tp.refs.exists("bronze", "combine_results") else None
        drills = tp.tracking_drill_counts() if tp.refs.exists("bronze", "combine_tracking") else None
        import polars as pl

        tp.export(summary, top, part if part is not None else pl.DataFrame(), drills if drills is not None else pl.DataFrame())
        for f in (tp.plot_participation(part) if part is not None else None,
                  tp.plot_drill_coverage(drills) if drills is not None else None):
            if f:
                m.add_artifact(f)
        m.rows_out = summary.height
        self._profile_summary = summary

    def step_tracking_qc(self, m):
        from .tracking_qc import TrackingQC

        pct = None if self.mode == "discovery" else 0  # 0 → full table
        res = TrackingQC(self.cfg, self.con).run(sample_pct=pct)
        self._qc_checks = res.get("checks")
        for p in res["_figures"]["path"].to_list() if "_figures" in res else []:
            m.add_artifact(p)

    def step_validate(self, m):
        from .reporting import export_table
        from .validate import DataQualityValidator

        pct = self.cfg.profiling.tracking_sample_pct if self.mode == "discovery" else None
        v = DataQualityValidator(self.cfg, self.con, sample_pct=pct)
        df = v.run()
        v.export(df)
        fig = v.plot(df)
        qc = getattr(self, "_qc_checks", None)
        m.add_artifact(v.write_report(df, getattr(self, "_profile_summary", None), qc, [fig] if fig else []))
        if qc is not None and not qc.is_empty():
            import polars as pl

            export_table(pl.concat([df, qc], how="diagonal_relaxed"), self.cfg.paths.tables, "data_quality_results")
        m.rows_out = df.height
        for r in df.filter(df["result"].is_in(["FAIL", "ERROR"])).iter_rows(named=True):
            m.warn(f"{r['severity']} {r['check']} {r['table']}.{r['columns']}: {r['details']}")
        v.raise_on_critical()

    def step_relationships(self, m):
        from .relationships import RelationshipAnalyzer

        ra = RelationshipAnalyzer(self.cfg, self.con)
        res = ra.run()
        m.add_artifact(ra.write_report(res))
        m.rows_out = res["presence"].height

    def step_silver(self, m):
        from .silver import SilverBuilder

        df = SilverBuilder(self.cfg, self.con).build_all(force=self.force)
        if not df.is_empty():
            m.rows_in, m.rows_out = int(df["rows_in"].sum()), int(df["rows_out"].sum())
        m.add_artifact(self.cfg.paths.silver)

    def step_gold(self, m):
        from .gold import GoldBuilder

        df = GoldBuilder(self.cfg, self.con).build_all()
        m.rows_out = int(df["rows"].sum()) if not df.is_empty() else 0
        m.add_artifact(self.cfg.paths.gold)

    def step_catalog(self, m):
        from .catalog import WarehouseCatalog

        df = WarehouseCatalog(self.cfg).build()
        m.rows_out = df.height
        m.add_artifact(self.cfg.paths.catalog / "bdb27.duckdb")

    def storage_by_layer(self):
        import polars as pl

        from .reporting import export_table
        from .utils import human_bytes

        rows = []
        raw = sum(f.stat().st_size for f in self.cfg.paths.data_root.rglob("*") if f.is_file())
        rows.append({"layer": "raw (CSV)", "bytes": raw})
        for layer in ("bronze", "silver", "gold"):
            base = getattr(self.cfg.paths, layer)
            n = sum(f.stat().st_size for f in base.rglob("*.parquet")) if base.exists() else 0
            rows.append({"layer": layer, "bytes": n})
        df = pl.DataFrame(rows).with_columns(
            pl.col("bytes").map_elements(human_bytes, return_dtype=pl.Utf8).alias("size"),
            (pl.col("bytes") / raw).round(3).alias("ratio_to_raw"))
        export_table(df, self.cfg.paths.tables, "storage_by_layer")
        return df

    def step_benchmark(self, m):
        from .benchmark import BenchmarkSuite

        bs = BenchmarkSuite(self.cfg)
        df = bs.run()
        m.add_artifact(bs.write_report(df, self.storage_by_layer()))
        m.rows_out = df.height

    def step_feasibility(self, m):
        from .feasibility import ResearchFeasibilityAssessor

        fa = ResearchFeasibilityAssessor(self.cfg, self.con)
        matrix, detail = fa.run()
        m.add_artifact(fa.write_report(matrix, detail))
        m.rows_out = matrix.height

    # ------------------------------------------------------------------ run
    def run(self, steps: list[str] | None = None, stop_on_error: bool = True):
        steps = steps or STEPS
        unknown = [s for s in steps if s not in STEPS]
        if unknown:
            raise ValueError(f"unknown steps {unknown}; valid: {STEPS}")
        for s in STEPS:  # canonical order regardless of CLI order
            if s not in steps or not self._gate(s):
                continue
            try:
                with self.runlog.step(f"{s} [{self.mode}]") as m:
                    getattr(self, f"step_{s}")(m)
            except Exception:
                self.runlog.save(self.cfg.paths.tables)
                if stop_on_error:
                    raise
        self.runlog.save(self.cfg.paths.tables)
        return self.runlog.to_frame()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NFL BDB 2027 — Phase 01 data pipeline")
    ap.add_argument("--steps", default="env,inventory,schema", help=f"comma list from {STEPS} or 'all'")
    ap.add_argument("--mode", default="discovery", choices=["discovery", "full"])
    ap.add_argument("--yes", action="store_true", help="allow expensive steps (full scans / materialisation)")
    ap.add_argument("--force", action="store_true", help="rebuild even if inputs are unchanged")
    ap.add_argument("--data-root", default=None, help="override DATA_ROOT")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load(**({"data_root": a.data_root} if a.data_root else {}))
    steps = STEPS if a.steps == "all" else [s.strip() for s in a.steps.split(",") if s.strip()]
    Phase01Pipeline(cfg, a.mode, a.yes, a.force).run(steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
