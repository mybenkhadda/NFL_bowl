"""Step 8 — performance benchmarks with identical query semantics across engines/formats.

Method
* Each benchmark runs in a **fresh spawned process**: run 1 = "cold process" (no Polars/DuckDB
  caches, fresh imports), runs 2..k = "warm". The OS page cache is NOT cleared (not possible
  without admin rights on Windows) — CSV benchmarks run first so Parquet does not benefit from
  a cache warmed by its own CSV source. This is stated in the report.
* Peak memory = max RSS of the child process sampled every 20 ms by the parent, minus the
  child's RSS just before the query (import overhead excluded).
* Every engine returns a small result table; a SHA-256 of the normalised, sorted result is
  compared within each query group to prove the engines computed the same thing.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl
import psutil

from .config import Phase01Config
from .frames import query_df
from .reporting import CATEGORICAL, INK, MarkdownReport, add_caption, export_table, new_figure, save_figure
from .utils import get_logger, human_bytes

log = get_logger(__name__)


# ----------------------------------------------------------------------------- helpers
def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    # Polars 2.0 removed collect(streaming=True); the streaming engine is selected via ``engine``.
    return lf.collect(engine="streaming")


def result_hash(df: pl.DataFrame) -> str:
    norm = df.select([
        (pl.col(c).cast(pl.Float64).round(4) if df.schema[c].is_numeric() else pl.col(c).cast(pl.Utf8)).alias(f"c{i}")
        for i, c in enumerate(df.columns)
    ])
    norm = norm.sort(norm.columns, nulls_last=True)
    return hashlib.sha256(norm.write_csv().encode()).hexdigest()[:16]


def _cfg(kw: dict) -> Phase01Config:
    return Phase01Config.load(kw["project_root"], data_root=kw["data_root"], warehouse_root=kw["warehouse_root"],
                              reports_root=kw["reports_root"], tmp_root=kw["tmp_root"])


def _duck(cfg: Phase01Config):
    from .config import duckdb_connect

    return duckdb_connect(cfg)


def _paths(cfg: Phase01Config, season: int) -> dict[str, Path]:
    from .sources import SOURCES
    from .tables import LayerRefs

    refs = LayerRefs(cfg)
    raw = [f for f in SOURCES["game_tracking"].files(cfg.paths.data_root) if str(season) in f.name][0]
    bronze = [f for f in refs.files("bronze", "game_tracking") if f"season={season}" in f.as_posix()][0]
    silver = [f for f in refs.files("silver", "game_tracking") if f"season={season}" in f.as_posix()]
    return {"raw": raw, "bronze": bronze, "silver": silver[0] if silver else None,
            "silver_all": refs.files("silver", "game_tracking"), "raw_all": SOURCES["game_tracking"].files(cfg.paths.data_root),
            "players": refs.files("silver", "players")[0] if refs.exists("silver", "players") else None,
            "games": refs.files("silver", "games")[0] if refs.exists("silver", "games") else None}


def _scan_raw(path: Path, cfg: Phase01Config | None = None) -> pl.LazyFrame:
    """Raw CSV scan with the same string-then-strict-cast typing as bronze (identical semantics)."""
    from .csv_schema import load_resolved_schema, read_csv_strings, strict_cast_exprs
    from .sources import SOURCES

    schema = load_resolved_schema(cfg.paths.catalog, "game_tracking") if cfg else dict(SOURCES["game_tracking"].schema)
    return read_csv_strings(path, null_values=("NA",), lazy=True).with_columns(strict_cast_exprs(schema))


def _sp(p: Path) -> str:
    return p.as_posix().replace("'", "''")


# ----------------------------------------------------------------------------- operations
# Q1: per-player frame count and mean speed for one season (full scan, 2 columns).
def q1_polars_csv(cfg, p):
    return _collect(_scan_raw(p["raw"], cfg).filter(pl.col("nfl_id").is_not_null())
                    .group_by("nfl_id").agg(pl.len().alias("n"), pl.col("s").mean().alias("mean_s")))


def q1_polars_bronze(cfg, p):
    return _collect(pl.scan_parquet(p["bronze"]).filter(pl.col("nfl_id").is_not_null())
                    .group_by("nfl_id").agg(pl.len().alias("n"), pl.col("s").mean().alias("mean_s")))


def q1_duckdb_csv(cfg, p):
    return query_df(_duck(cfg), f"SELECT nfl_id, count(*) AS n, avg(s) AS mean_s FROM read_csv('{_sp(p['raw'])}', "
                          f"header = true, nullstr = 'NA') WHERE nfl_id IS NOT NULL GROUP BY 1")


def q1_duckdb_bronze(cfg, p):
    return query_df(_duck(cfg), f"SELECT nfl_id, count(*) AS n, avg(s) AS mean_s FROM read_parquet('{_sp(p['bronze'])}') "
                          f"WHERE nfl_id IS NOT NULL GROUP BY 1")


def q1_duckdb_silver(cfg, p):
    return query_df(_duck(cfg), f"SELECT nfl_id, count(*) AS n, avg(s) AS mean_s FROM read_parquet('{_sp(p['silver'])}') "
                          f"WHERE NOT is_football GROUP BY 1")


# Q2: tracking ⨝ players, aggregate by nfl_position.
def q2_polars_join(cfg, p):
    players = pl.scan_parquet(p["players"]).select("nfl_id", "nfl_position")
    return _collect(pl.scan_parquet(p["silver"]).filter(~pl.col("is_football")).select("nfl_id", "s")
                    .join(players, on="nfl_id", how="inner")
                    .group_by("nfl_position").agg(pl.len().alias("n"), pl.col("s").mean().alias("mean_s")))


def q2_duckdb_join(cfg, p):
    return query_df(_duck(cfg), f"""
        SELECT pl.nfl_position, count(*) AS n, avg(t.s) AS mean_s
        FROM read_parquet('{_sp(p['silver'])}') t JOIN read_parquet('{_sp(p['players'])}') pl USING (nfl_id)
        WHERE NOT t.is_football GROUP BY 1""")


# Q3: athlete-level two-stage aggregation (play max speed → athlete median / max).
def q3_polars_athlete(cfg, p):
    per_play = (pl.scan_parquet(p["silver"]).filter(~pl.col("is_football"))
                .group_by("game_id", "play_id", "nfl_id").agg(pl.col("s").max().alias("max_s")))
    return _collect(per_play.group_by("nfl_id").agg(pl.len().alias("n_plays"), pl.col("max_s").median().alias("med"),
                                                    pl.col("max_s").max().alias("mx")))


def q3_duckdb_athlete(cfg, p):
    return query_df(_duck(cfg), f"""
        WITH pp AS (SELECT game_id, play_id, nfl_id, max(s) AS max_s FROM read_parquet('{_sp(p['silver'])}')
                    WHERE NOT is_football GROUP BY ALL)
        SELECT nfl_id, count(*) AS n_plays, median(max_s) AS med, max(max_s) AS mx FROM pp GROUP BY 1""")


# Q4: single-play lookup (selective filter) — raw CSV vs bronze vs sorted silver.
def _q4_key(cfg, p) -> tuple[int, int]:
    import duckdb

    return duckdb.sql(f"SELECT game_id, play_id FROM read_parquet('{_sp(p['bronze'])}') "
                      f"WHERE _src_row = (SELECT max(_src_row) // 2 FROM read_parquet('{_sp(p['bronze'])}'))").fetchone()


def q4_polars_csv(cfg, p, key):
    g, pl_ = key
    return _collect(_scan_raw(p["raw"], cfg).filter((pl.col("game_id") == g) & (pl.col("play_id") == pl_))
                    .select("nfl_id", "time", "x", "y"))


def q4_polars_bronze(cfg, p, key):
    g, pl_ = key
    return pl.scan_parquet(p["bronze"]).filter((pl.col("game_id") == g) & (pl.col("play_id") == pl_)) \
        .select("nfl_id", "time", "x", "y").collect()


def q4_polars_silver(cfg, p, key):
    g, pl_ = key
    return pl.scan_parquet(p["silver"]).filter((pl.col("game_id") == g) & (pl.col("play_id") == pl_)) \
        .select("nfl_id", pl.col("ts").dt.strftime("%Y-%m-%dT%H:%M:%S%.3f").alias("time"), "x", "y").collect()


def q4_duckdb_silver(cfg, p, key):
    g, pl_ = key
    return query_df(_duck(cfg), f"SELECT nfl_id, strftime(ts, '%Y-%m-%dT%H:%M:%S.%g') AS time, x, y "
                          f"FROM read_parquet('{_sp(p['silver'])}') WHERE game_id = {g} AND play_id = {pl_}")


# Q5: partition pruning — rows of one season from the all-season dataset.
def q5_polars_raw_all(cfg, p, season):
    games = pl.scan_parquet(p["games"]).filter(pl.col("season") == season).select("game_id")
    lf = pl.concat([_scan_raw(f, cfg).select("game_id") for f in p["raw_all"]])
    return _collect(lf.join(games, on="game_id", how="semi").select(pl.len().alias("n")))


def q5_polars_silver_pruned(cfg, p, season):
    return pl.scan_parquet(p["silver_all"], hive_partitioning=True).filter(pl.col("season") == season) \
        .select(pl.len().alias("n")).collect()


def q5_duckdb_silver_pruned(cfg, p, season):
    files = ", ".join(f"'{_sp(f)}'" for f in p["silver_all"])
    return query_df(_duck(cfg), f"SELECT count(*) AS n FROM read_parquet([{files}], hive_partitioning = true) "
                          f"WHERE season = {season}")


@dataclass(frozen=True)
class Bench:
    name: str
    group: str
    engine: str
    fmt: str
    fn: Callable
    needs: str = ""          # "key" | "season" | ""


BENCHES = [
    Bench("q1_polars_csv", "Q1 per-player aggregate (1 season)", "polars", "csv", q1_polars_csv),
    Bench("q1_duckdb_csv", "Q1 per-player aggregate (1 season)", "duckdb", "csv", q1_duckdb_csv),
    Bench("q1_polars_bronze", "Q1 per-player aggregate (1 season)", "polars", "parquet-bronze", q1_polars_bronze),
    Bench("q1_duckdb_bronze", "Q1 per-player aggregate (1 season)", "duckdb", "parquet-bronze", q1_duckdb_bronze),
    Bench("q1_duckdb_silver", "Q1 per-player aggregate (1 season)", "duckdb", "parquet-silver", q1_duckdb_silver),
    Bench("q2_polars_join", "Q2 tracking JOIN players", "polars", "parquet-silver", q2_polars_join),
    Bench("q2_duckdb_join", "Q2 tracking JOIN players", "duckdb", "parquet-silver", q2_duckdb_join),
    Bench("q3_polars_athlete", "Q3 athlete-level 2-stage aggregate", "polars", "parquet-silver", q3_polars_athlete),
    Bench("q3_duckdb_athlete", "Q3 athlete-level 2-stage aggregate", "duckdb", "parquet-silver", q3_duckdb_athlete),
    Bench("q4_polars_csv", "Q4 single-play lookup", "polars", "csv", q4_polars_csv, "key"),
    Bench("q4_polars_bronze", "Q4 single-play lookup", "polars", "parquet-bronze", q4_polars_bronze, "key"),
    Bench("q4_polars_silver", "Q4 single-play lookup", "polars", "parquet-silver (sorted)", q4_polars_silver, "key"),
    Bench("q4_duckdb_silver", "Q4 single-play lookup", "duckdb", "parquet-silver (sorted)", q4_duckdb_silver, "key"),
    Bench("q5_polars_raw_all", "Q5 one season from all seasons", "polars", "csv (semi-join on games)", q5_polars_raw_all, "season"),
    Bench("q5_polars_silver_pruned", "Q5 one season from all seasons", "polars", "parquet-silver (hive pruned)", q5_polars_silver_pruned, "season"),
    Bench("q5_duckdb_silver_pruned", "Q5 one season from all seasons", "duckdb", "parquet-silver (hive pruned)", q5_duckdb_silver_pruned, "season"),
]
_BY_NAME = {b.name: b for b in BENCHES}


def _child(name: str, cfg_kw: dict, season: int, repeats: int, queue) -> None:
    try:
        cfg = _cfg(cfg_kw)
        p = _paths(cfg, season)
        b = _BY_NAME[name]
        extra = ()
        if b.needs == "key":
            extra = (_q4_key(cfg, p),)
        elif b.needs == "season":
            extra = (season,)
        rss0 = psutil.Process().memory_info().rss
        times, df = [], None
        for _ in range(repeats):
            t0 = time.perf_counter()
            df = b.fn(cfg, p, *extra)
            times.append(time.perf_counter() - t0)
        queue.put({"ok": True, "times": times, "rss_before": rss0, "rows_out": df.height, "hash": result_hash(df)})
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class BenchmarkSuite:
    def __init__(self, cfg: Phase01Config, season: int | None = None, repeats: int = 3, timeout_s: int = 1800):
        self.cfg = cfg
        self.repeats = repeats
        self.timeout_s = timeout_s
        self.season = season
        self.cfg_kw = {k: str(getattr(cfg.paths, k)) for k in
                       ("project_root", "data_root", "warehouse_root", "reports_root", "tmp_root")}

    def _default_season(self) -> int:
        from .tables import LayerRefs

        files = LayerRefs(self.cfg).files("bronze", "game_tracking")
        if not files:
            raise FileNotFoundError("bronze game_tracking required for benchmarks")
        return int(files[0].parent.name.split("=")[-1])  # smallest season = cheapest representative

    def run_one(self, name: str) -> dict:
        season = self.season or self._default_season()
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(target=_child, args=(name, self.cfg_kw, season, self.repeats, q))
        peak = 0
        t0 = time.perf_counter()
        proc.start()
        ps = psutil.Process(proc.pid)
        result = None
        while True:
            try:
                peak = max(peak, ps.memory_info().rss)
            except psutil.Error:
                pass
            if not q.empty():
                result = q.get()
                break
            if not proc.is_alive():
                break
            if time.perf_counter() - t0 > self.timeout_s:
                proc.terminate()
                result = {"ok": False, "error": f"timeout after {self.timeout_s}s"}
                break
            time.sleep(0.02)
        proc.join(timeout=10)
        if result is None:
            result = q.get(timeout=5) if not q.empty() else {"ok": False, "error": f"child exited {proc.exitcode}"}
        b = _BY_NAME[name]
        row = {"bench": name, "group": b.group, "engine": b.engine, "format": b.fmt, "season": season,
               "repeats": self.repeats, "ok": result["ok"], "error": result.get("error", "")}
        if result["ok"]:
            t = result["times"]
            row.update({
                "cold_s": round(t[0], 3), "warm_median_s": round(sorted(t[1:])[len(t[1:]) // 2], 3) if len(t) > 1 else None,
                "peak_rss_mb": round(peak / 1024**2, 1),
                "peak_query_mb": round((peak - result["rss_before"]) / 1024**2, 1),
                "rows_out": result["rows_out"], "result_hash": result["hash"],
            })
        log.info("bench %-26s %s", name, {k: row.get(k) for k in ("cold_s", "warm_median_s", "peak_query_mb", "error")})
        return row

    def run(self, names: list[str] | None = None) -> pl.DataFrame:
        rows = [self.run_one(n) for n in (names or [b.name for b in BENCHES])]
        df = pl.DataFrame(rows, infer_schema_length=None)
        if "result_hash" in df.columns:
            consistent = df.filter(pl.col("ok")).group_by("group").agg(
                (pl.col("result_hash").n_unique() == 1).alias("results_identical"))
            df = df.join(consistent, on="group", how="left")
        export_table(df, self.cfg.paths.tables, "benchmark_results")
        return df

    def plot(self, df: pl.DataFrame) -> Path | None:
        d = df.filter(pl.col("ok"))
        if d.is_empty():
            return None
        groups = d["group"].unique(maintain_order=True).to_list()
        fig, axes = new_figure(len(groups), 1, figsize=(12.8, 2.0 + 1.3 * d.height), squeeze=False,
                               gridspec_kw={"height_ratios": [d.filter(pl.col("group") == g).height for g in groups]})
        eng_color = {"polars": CATEGORICAL[0], "duckdb": CATEGORICAL[1]}
        for ax, g in zip(axes[:, 0], groups):
            s = d.filter(pl.col("group") == g)
            labels = [f"{e} · {f}" for e, f in zip(s["engine"], s["format"])]
            warm = s["warm_median_s"].fill_null(s["cold_s"]).to_list()
            ax.barh(labels, s["cold_s"].to_list(), color="#e1e0d9", height=0.6, label="cold process (run 1)")
            ax.barh(labels, warm, color=[eng_color[e] for e in s["engine"]], height=0.32, label="warm median")
            for i, (c, m) in enumerate(zip(s["cold_s"], s["peak_query_mb"])):
                ax.text(c, i, f"  {c:.2f}s cold · {m:,.0f} MB peak", va="center", fontsize=8, color=INK["secondary"])
            ax.set_title(g, fontsize=12)
            ax.grid(axis="y", visible=False)
            ax.invert_yaxis()
        axes[-1, 0].set_xlabel("Wall-clock seconds")
        axes[0, 0].legend(loc="lower right")
        add_caption(fig, "Fresh process per benchmark; OS page cache not cleared. Bars: grey = cold, colour = warm median "
                         "(blue Polars, orange DuckDB).")
        return save_figure(fig, self.cfg.paths.figures, "07_benchmarks")

    def write_report(self, df: pl.DataFrame, storage: pl.DataFrame | None = None) -> Path:
        p = self.cfg.paths
        fig = self.plot(df)
        rpt = MarkdownReport("07 — Performance benchmarks", "Step 8: measured, identical-semantics comparisons")
        rpt.h2("Method").bullets([
            f"Each benchmark: fresh spawned process, {self.repeats} runs (run 1 = cold process, others = warm).",
            "OS file cache not cleared (Windows, no admin) → 'cold' means cold process, not cold disk. CSV runs first.",
            "Peak memory = max child RSS (20 ms sampling) minus RSS before the query.",
            "`results_identical` compares SHA-256 of normalised sorted results within each query group.",
            f"Machine: {psutil.cpu_count()} logical cores, {human_bytes(psutil.virtual_memory().total)} RAM, "
            f"threads={self.cfg.resources.resolved_threads()}, DuckDB memory_limit={self.cfg.resources.resolved_duckdb_memory()}.",
        ])
        if fig:
            rpt.image(fig, p.reports_root, "benchmarks")
        rpt.h2("Results").table(df.select([c for c in ("group", "engine", "format", "cold_s", "warm_median_s", "peak_query_mb",
                                                       "rows_out", "results_identical", "ok", "error") if c in df.columns]))
        if storage is not None and not storage.is_empty():
            rpt.h2("Storage footprint by layer").table(storage)
        bad = df.filter(pl.col("results_identical") == False) if "results_identical" in df.columns else pl.DataFrame()  # noqa: E712
        if bad.height:
            rpt.callout("WARNING", "Some groups returned different results across engines — investigate before trusting timings.")
        rpt.p("Interpretation must be grounded in the table above; a single query family is not evidence that one engine "
              "is universally faster.")
        return rpt.save(p.reports_root / "07_performance_benchmarks.md")
