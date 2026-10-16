"""Step 3, Pass B — targeted exact validation.

Every check returns a :class:`CheckResult` row (check, table, result, severity, metric,
recommended action, the SQL that produced it, its runtime, and whether it ran on a sample or
exactly). ``DataQualityValidator.raise_on_critical()`` turns CRITICAL failures into an
exception so downstream builds stop instead of propagating bad data.

Cost discipline: :meth:`DataQualityValidator.estimate_cost` reports rows/bytes for each table
from Parquet metadata (free) and is logged before any exact check runs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import polars as pl

from .config import Phase01Config, duckdb_connect
from .frames import query_df, query_row
from .reporting import SEVERITY_COLORS, INK, MarkdownReport, add_caption, export_table, new_figure, save_figure
from .sources import FOREIGN_KEYS, SOURCES, ForeignKey
from .tables import LayerRefs, sample_clause
from .utils import get_logger, human_bytes

log = get_logger(__name__)

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")


class DataIntegrityError(RuntimeError):
    pass


@dataclass
class CheckResult:
    check_id: str
    check: str
    table: str
    columns: str
    result: str              # PASS | FAIL | WARN | ERROR | SKIPPED
    severity: str
    metric: str
    value: float | None
    denominator: float | None
    threshold: str
    details: str
    recommended_action: str
    mode: str                # exact | sample
    seconds: float
    sql: str


class DataQualityValidator:
    def __init__(self, cfg: Phase01Config, con=None, layer: str = "bronze", sample_pct: float | None = None):
        self.cfg = cfg
        self.con = con or duckdb_connect(cfg)
        self.refs = LayerRefs(cfg)
        self.layer = layer
        self.sample_pct = sample_pct            # None = exact
        self.results: list[CheckResult] = []

    # ------------------------------------------------------------------ helpers
    def ref(self, table: str, sampled: bool = False) -> str:
        """Sampling (discovery mode) applies to tracking tables only; small tables are always exact."""
        base = self.refs.sql(self.layer, table)
        if sampled and self.sample_pct and SOURCES[table].is_tracking:
            return f"(SELECT * FROM {base}{sample_clause(self.sample_pct, self.cfg.profiling.seed)})"
        return base

    @property
    def mode(self) -> str:
        return f"tracking sampled {self.sample_pct}%, others exact" if self.sample_pct else "exact"

    def _run(self, check_id: str, check: str, table: str, columns: str, sql: str,
             evaluate: Callable[[dict], tuple[str, float | None, float | None, str]],
             severity: str, metric: str, threshold: str, action: str) -> CheckResult:
        t0 = time.perf_counter()
        try:
            row = query_row(self.con, sql)
            result, value, denom, details = evaluate(row)
        except Exception as exc:  # noqa: BLE001 - a failing check must be reported, not hide others
            result, value, denom, details = "ERROR", None, None, f"{type(exc).__name__}: {exc}"
        sampled = bool(self.sample_pct) and table in SOURCES and SOURCES[table].is_tracking
        mode = f"sample {self.sample_pct}% (finds issues, cannot prove absence)" if sampled else "exact"
        res = CheckResult(check_id, check, table, columns, result, severity, metric, value, denom, threshold,
                          details, action, mode, round(time.perf_counter() - t0, 3), " ".join(sql.split()))
        self.results.append(res)
        lvl = log.warning if result in ("FAIL", "ERROR") else log.info
        lvl("[%s] %s %s.%s → %s (%s)", res.severity, check, table, columns, result, details)
        return res

    # ------------------------------------------------------------------ cost
    def estimate_cost(self) -> pl.DataFrame:
        rows = []
        for name in SOURCES:
            if self.refs.exists(self.layer, name):
                rows.append({"table": name, "rows": self.refs.num_rows(self.layer, name),
                             "parquet_bytes": self.refs.num_bytes(self.layer, name),
                             "parquet_size": human_bytes(self.refs.num_bytes(self.layer, name))})
        df = pl.DataFrame(rows)
        log.info("validation input (%s):\n%s", self.mode, df)
        return df

    # ------------------------------------------------------------------ checks
    def key_uniqueness(self, table: str, key: tuple[str, ...], severity: str = "CRITICAL") -> CheckResult:
        """Exact uniqueness: rows vs distinct keys. On a sample this can only *find*
        duplicates, never prove uniqueness — the mode column makes that explicit."""
        k = ", ".join(key)
        # Single pass (one sample draw): group by key, then compare total rows to group count.
        sql = f"SELECT CAST(coalesce(sum(n), 0) AS BIGINT) AS n_rows, count(*) AS n_keys FROM (SELECT {k}, count(*) AS n FROM {self.ref(table, True)} GROUP BY ALL)"

        def ev(r):
            dup = r["n_rows"] - r["n_keys"]
            return ("PASS" if dup == 0 else "FAIL", float(dup), float(r["n_rows"]),
                    f"{dup:,} duplicate-key rows of {r['n_rows']:,} ({r['n_keys']:,} distinct keys)")

        return self._run(f"uniq:{table}:{k}", "Key uniqueness", table, k, sql, ev, severity,
                         "duplicate_key_rows", "0",
                         "If FAIL: inspect with duplicate_examples(); refine grain or dedup in silver with a documented rule.")

    def exact_duplicate_rows(self, table: str, severity: str = "HIGH") -> CheckResult:
        cols = [c for c in SOURCES[table].columns]
        k = ", ".join(f'"{c}"' for c in cols)
        sql = f"SELECT CAST(coalesce(sum(n), 0) AS BIGINT) AS n_rows, count(*) AS n_distinct FROM (SELECT {k}, count(*) AS n FROM {self.ref(table, True)} GROUP BY ALL)"

        def ev(r):
            dup = r["n_rows"] - r["n_distinct"]
            return ("PASS" if dup == 0 else "WARN", float(dup), float(r["n_rows"]),
                    f"{dup:,} fully identical rows")

        return self._run(f"dupe:{table}", "Exact duplicate rows", table, "*all source columns*", sql, ev, severity,
                         "identical_rows", "0", "Identical rows are safe to drop in silver (keep lowest _src_row).")

    def null_identifiers(self, table: str, cols: tuple[str, ...], severity: str = "HIGH",
                         where: str = "TRUE", allowed_frac: float = 0.0) -> CheckResult:
        nulls = " + ".join(f"CASE WHEN {c} IS NULL THEN 1 ELSE 0 END" for c in cols)
        sql = f"SELECT count(*) AS n, count_if(({nulls}) > 0) AS n_null FROM {self.ref(table, True)} WHERE {where}"

        def ev(r):
            frac = (r["n_null"] or 0) / max(r["n"], 1)
            return ("PASS" if frac <= allowed_frac else "FAIL", float(r["n_null"] or 0), float(r["n"]),
                    f"{frac:.4%} rows with a null in ({', '.join(cols)}) where {where}")

        return self._run(f"null:{table}:{','.join(cols)}:{where}", "Missing identifier", table, ", ".join(cols), sql, ev,
                         severity, "rows_with_null_id", f"<= {allowed_frac:.2%}",
                         "Rows without identifiers cannot be linked; quarantine or document (e.g. football rows).")

    def null_profile(self, table: str, col: str, where: str = "TRUE") -> CheckResult:
        """Informational: share of null values (e.g. null nfl_id in tracking = football?)."""
        sql = f"SELECT count(*) AS n, count(*) - count({col}) AS n_null FROM {self.ref(table, True)} WHERE {where}"

        def ev(r):
            return ("PASS", float(r["n_null"]), float(r["n"]), f"{r['n_null'] / max(r['n'], 1):.4%} null")

        return self._run(f"nullprof:{table}:{col}", "Null share", table, col, sql, ev, "INFO", "null_rows", "info",
                         "Document meaning of nulls.")

    def referential_integrity(self, fk: ForeignKey) -> CheckResult:
        on = " AND ".join(f"c.{a} = p.{b}" for a, b in zip(fk.child_cols, fk.parent_cols))
        ccols = ", ".join(f"c.{a}" for a in fk.child_cols)
        where = fk.child_filter or "TRUE"
        sql = f"""
            WITH c AS (SELECT DISTINCT {', '.join(fk.child_cols)} FROM {self.ref(fk.child, True)} WHERE {where}),
                 p AS (SELECT DISTINCT {', '.join(fk.parent_cols)} FROM {self.ref(fk.parent)})
            SELECT (SELECT count(*) FROM c) AS n_child_keys,
                   (SELECT count(*) FROM c ANTI JOIN p ON {on}) AS n_orphan_keys
        """

        def ev(r):
            frac = r["n_orphan_keys"] / max(r["n_child_keys"], 1)
            return ("PASS" if r["n_orphan_keys"] == 0 else "FAIL", float(r["n_orphan_keys"]), float(r["n_child_keys"]),
                    f"{r['n_orphan_keys']:,} of {r['n_child_keys']:,} distinct child keys have no parent ({frac:.2%})")

        cols = ",".join(fk.child_cols)
        return self._run(f"fk:{fk.child}:{cols}->{fk.parent}", "Referential integrity",
                         fk.child, f"({cols}) → {fk.parent}({','.join(fk.parent_cols)})", sql, ev, fk.severity,
                         "orphan_distinct_keys", "0",
                         fk.note or "Investigate orphans; never repair by name matching. Keep orphans flagged in silver.")

    def value_range(self, table: str, col: str, lo: float | None, hi: float | None,
                    severity: str = "MEDIUM", where: str = "TRUE", allowed_frac: float = 0.0) -> CheckResult:
        conds = []
        if lo is not None:
            conds.append(f"{col} < {lo}")
        if hi is not None:
            conds.append(f"{col} > {hi}")
        sql = (f"SELECT count({col}) AS n, count_if({' OR '.join(conds)}) AS n_out, "
               f"min({col}) AS mn, max({col}) AS mx FROM {self.ref(table, True)} WHERE {where}")

        def ev(r):
            n_out = r["n_out"] or 0
            frac = n_out / max(r["n"], 1)
            return ("PASS" if frac <= allowed_frac else "WARN", float(n_out), float(r["n"]),
                    f"{n_out:,} values outside [{lo}, {hi}] ({frac:.4%}); observed min={r['mn']}, max={r['mx']}")

        return self._run(f"range:{table}:{col}", "Value range", table, col, sql, ev, severity, "out_of_range_values",
                         f"[{lo}, {hi}]", "Flag (do not delete) in silver; inspect extreme rows.")

    def timestamp_parse(self, table: str, col: str = "time", severity: str = "CRITICAL") -> CheckResult:
        sql = (f"SELECT count({col}) AS n, count_if(try_cast({col} AS TIMESTAMP) IS NULL) "
               f"AS n_bad, min({col}) AS mn, max({col}) AS mx FROM {self.ref(table, True)} WHERE {col} IS NOT NULL")

        def ev(r):
            return ("PASS" if (r["n_bad"] or 0) == 0 else "FAIL", float(r["n_bad"] or 0), float(r["n"]),
                    f"{r['n_bad'] or 0:,} unparseable timestamps; range {r['mn']} → {r['mx']}")

        return self._run(f"ts:{table}:{col}", "Timestamp parse", table, col, sql, ev, severity, "unparseable",
                         "0", "Silver aborts on unparseable timestamps; fix the format string.")

    def season_consistency(self) -> CheckResult:
        """Filename season partition of game tracking must equal games.season for every game."""
        sql = f"""
            WITH t AS (SELECT DISTINCT game_id, CAST(season AS BIGINT) AS file_season FROM {self.ref('game_tracking')})
            SELECT count(*) AS n_games, count_if(g.season <> t.file_season) AS n_mismatch
            FROM t JOIN {self.ref('games')} g USING (game_id)
        """

        def ev(r):
            return ("PASS" if (r["n_mismatch"] or 0) == 0 else "FAIL", float(r["n_mismatch"] or 0), float(r["n_games"]),
                    f"{r['n_mismatch'] or 0:,} of {r['n_games']:,} tracked games where filename season ≠ games.season")

        return self._run("season:game_tracking", "Partition consistency", "game_tracking", "season (filename) vs games.season",
                         sql, ev, "HIGH", "mismatched_games", "0", "Use games.season as the authoritative season in silver.")

    def game_date_consistency(self) -> CheckResult:
        """game_id encodes YYYYMMDD; tracking timestamps should fall within ±1 day (UTC offset)."""
        sql = f"""
            WITH t AS (
                SELECT game_id, min(try_cast(time AS TIMESTAMP)) AS t0 FROM {self.ref('game_tracking', True)} GROUP BY 1)
            SELECT count(*) AS n_games,
                   count_if(abs(date_diff('day', strptime(CAST(game_id // 100 AS VARCHAR), '%Y%m%d'), t0)) > 1) AS n_off
            FROM t
        """

        def ev(r):
            return ("PASS" if (r["n_off"] or 0) == 0 else "WARN", float(r["n_off"] or 0), float(r["n_games"]),
                    f"{r['n_off'] or 0:,} games whose first timestamp is > 1 day from the game_id date")

        return self._run("date:game_tracking", "Identifier/time consistency", "game_tracking", "game_id date vs time",
                         sql, ev, "MEDIUM", "games_off_by_gt_1_day", "0",
                         "If WARN: game_id namespace differs from date encoding, or timestamps are shifted.")

    def allowed_values(self, table: str, col: str, allowed: list[str], severity: str = "MEDIUM") -> CheckResult:
        vals = ", ".join(f"'{v}'" for v in allowed)
        sql = (f"SELECT count({col}) AS n, count_if({col} NOT IN ({vals})) AS n_bad, "
               f"list_slice(list(DISTINCT {col}) FILTER (WHERE {col} NOT IN ({vals})), 1, 10) AS examples "
               f"FROM {self.ref(table, True)}")

        def ev(r):
            return ("PASS" if (r["n_bad"] or 0) == 0 else "WARN", float(r["n_bad"] or 0), float(r["n"]),
                    f"{r['n_bad'] or 0:,} values not in {allowed}; examples={r['examples']}")

        return self._run(f"domain:{table}:{col}", "Allowed values", table, col, sql, ev, severity, "unexpected_values",
                         str(allowed), "Extend the domain or map values in silver.")

    def duplicate_examples(self, table: str, key: tuple[str, ...], limit: int = 20) -> pl.DataFrame:
        k = ", ".join(key)
        q = f"""
            WITH d AS (SELECT {k}, count(*) AS n FROM {self.ref(table)} GROUP BY ALL HAVING count(*) > 1 LIMIT {limit})
            SELECT t.* FROM {self.ref(table)} t JOIN d USING ({k}) ORDER BY {k}, _src_row
        """
        return query_df(self.con, q)

    # ------------------------------------------------------------------ suite
    def default_suite(self) -> list[Callable[[], CheckResult]]:
        tr = self.cfg.tracking
        tol = tr.coord_tolerance_yds
        suite: list[Callable[[], CheckResult]] = []
        for spec in SOURCES.values():
            if not self.refs.exists(self.layer, spec.name):
                continue
            for key in spec.grain_candidates:
                # The narrowest key is CRITICAL for tracking; alternatives are informative.
                sev = "CRITICAL" if key == spec.grain_candidates[-1] else "MEDIUM"
                suite.append(lambda s=spec.name, k=key, sv=sev: self.key_uniqueness(s, k, sv))
            if not spec.is_tracking:
                suite.append(lambda s=spec.name: self.exact_duplicate_rows(s))
        if self.refs.exists(self.layer, "game_tracking"):
            suite += [
                lambda: self.null_identifiers("game_tracking", ("game_id", "play_id", "time"), "CRITICAL"),
                lambda: self.null_profile("game_tracking", "nfl_id"),
                lambda: self.timestamp_parse("game_tracking"),
                lambda: self.value_range("game_tracking", "x", -tol, tr.field_length_yds + tol, where="nfl_id IS NOT NULL"),
                lambda: self.value_range("game_tracking", "y", -tol, tr.field_width_yds + tol, where="nfl_id IS NOT NULL"),
                lambda: self.value_range("game_tracking", "s", 0, tr.max_speed_yds, where="nfl_id IS NOT NULL", allowed_frac=1e-4),
                lambda: self.value_range("game_tracking", "a", 0, tr.max_accel_yds2, where="nfl_id IS NOT NULL", allowed_frac=1e-4),
                lambda: self.value_range("game_tracking", "dir", 0, 360, severity="LOW"),
                lambda: self.value_range("game_tracking", "o", 0, 360, severity="LOW"),
                self.season_consistency,
                self.game_date_consistency,
            ]
        if self.refs.exists(self.layer, "combine_tracking"):
            suite += [
                lambda: self.null_identifiers("combine_tracking", ("event_id", "time"), "CRITICAL"),
                lambda: self.null_identifiers("combine_tracking", ("nfl_id",), "HIGH", where="entity_type = 'PLAYER'"),
                lambda: self.allowed_values("combine_tracking", "entity_type", ["PLAYER", "BALL"], "MEDIUM"),
                lambda: self.timestamp_parse("combine_tracking"),
                lambda: self.value_range("combine_tracking", "s", 0, tr.max_speed_yds, allowed_frac=1e-4),
                lambda: self.value_range("combine_tracking", "a", 0, tr.max_accel_yds2, allowed_frac=1e-4),
            ]
        if self.refs.exists(self.layer, "player_play"):
            suite += [
                lambda: self.allowed_values("player_play", "play_direction", ["left", "right"], "HIGH"),
                lambda: self.null_identifiers("player_play", ("game_id", "play_id", "nfl_id"), "CRITICAL"),
            ]
        if self.refs.exists(self.layer, "combine_results"):
            suite += [
                lambda: self.value_range("combine_results", "forty", 4.0, 6.5, severity="LOW"),
                lambda: self.value_range("combine_results", "combine_height", 60, 84, severity="LOW"),
                lambda: self.value_range("combine_results", "combine_weight", 150, 400, severity="LOW"),
            ]
        for fk in FOREIGN_KEYS:
            if self.refs.exists(self.layer, fk.child) and self.refs.exists(self.layer, fk.parent):
                suite.append(lambda f=fk: self.referential_integrity(f))
        return suite

    def run(self, suite: list[Callable[[], CheckResult]] | None = None) -> pl.DataFrame:
        self.estimate_cost()
        for check in suite or self.default_suite():
            check()
        return self.to_frame()

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame([asdict(r) for r in self.results], infer_schema_length=None) if self.results else pl.DataFrame()

    def raise_on_critical(self) -> None:
        bad = [r for r in self.results if r.severity == "CRITICAL" and r.result in ("FAIL", "ERROR")]
        if bad:
            msg = "; ".join(f"{r.check} {r.table}.{r.columns}: {r.details}" for r in bad)
            raise DataIntegrityError(f"{len(bad)} critical data-quality failure(s): {msg}")

    # ------------------------------------------------------------------ outputs
    def export(self, df: pl.DataFrame | None = None) -> list[Path]:
        return export_table(df if df is not None else self.to_frame(), self.cfg.paths.tables, "data_quality_results")

    def plot(self, df: pl.DataFrame) -> Path | None:
        if df.is_empty():
            return None
        agg = (df.group_by("severity", "result").len()
               .pivot(on="result", index="severity", values="len").fill_null(0))
        order = [s for s in SEVERITIES if s in agg["severity"].to_list()]
        agg = agg.with_columns(pl.col("severity").cast(pl.Enum(order))).sort("severity")
        results = [c for c in ("PASS", "WARN", "FAIL", "ERROR") if c in agg.columns]
        colors = {"PASS": "#0ca30c", "WARN": "#fab219", "FAIL": "#d03b3b", "ERROR": INK["secondary"]}
        fig, ax = new_figure(figsize=(12.8, 4.8))
        left = [0] * agg.height
        for r in results:
            vals = agg[r].to_list()
            ax.barh(agg["severity"].cast(pl.Utf8).to_list(), vals, left=left, color=colors[r], label=r, height=0.55,
                    edgecolor="white", linewidth=2)
            left = [a + b for a, b in zip(left, vals)]
        ax.invert_yaxis()
        ax.set_xlabel("Number of checks")
        ax.set_title("Data-quality checks by severity and outcome")
        ax.legend(ncols=len(results), loc="lower right")
        ax.grid(axis="y", visible=False)
        add_caption(fig, f"Source: data_quality_results ({df.height} checks, mode: {self.mode}).")
        return save_figure(fig, self.cfg.paths.figures, "03_data_quality_overview")

    def write_report(self, df: pl.DataFrame, profile_summary: pl.DataFrame | None = None,
                     qc_results: pl.DataFrame | None = None, figures: list[Path] | None = None) -> Path:
        p = self.cfg.paths
        all_checks = pl.concat([d for d in (df, qc_results) if d is not None and not d.is_empty()], how="diagonal_relaxed") \
            if (qc_results is not None and not qc_results.is_empty()) else df
        rpt = MarkdownReport("03 — Data quality", f"Step 3: bounded profiling (Pass A) + targeted validation (Pass B, {self.mode})")
        counts = all_checks.group_by("result").len().sort("result")
        rpt.h2("Summary").table(counts)
        crit = all_checks.filter(pl.col("severity").is_in(["CRITICAL", "HIGH"]) & pl.col("result").is_in(["FAIL", "ERROR"]))
        if crit.height:
            rpt.callout("CAUTION", f"{crit.height} CRITICAL/HIGH check(s) failed — see below before trusting joins.")
        for f in figures or []:
            rpt.image(f, p.reports_root, "figure")
        rpt.h2("Quality results").table(all_checks.select(
            "check", "table", "columns", "result", "severity", "details", "recommended_action", "mode")
            .with_columns(pl.col("severity").cast(pl.Enum(list(SEVERITIES)))).sort("severity", "result"), max_rows=200)
        if profile_summary is not None and not profile_summary.is_empty():
            rpt.h2("Pass A — column profiles")
            rpt.p("Full per-column profile: `reports/tables/profile_summary.csv` (approx_unique = HyperLogLog estimate).")
            rpt.table(profile_summary.select(
                "table", "column_name", "column_type", "min", "max", "approx_unique", "null_percentage", "profile_mode"),
                max_rows=250)
        return rpt.save(p.reports_root / "03_data_quality.md")
