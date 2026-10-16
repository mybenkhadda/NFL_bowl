"""Step 3, Pass A — bounded column profiling over bronze Parquet.

Uses DuckDB ``SUMMARIZE`` (approximate distinct counts via HyperLogLog, approximate
quantiles) over either a deterministic Bernoulli sample (default for tracking tables) or the
full table (small reference tables). The mode is recorded in every output row.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .config import Phase01Config, duckdb_connect
from .frames import query_df
from .reporting import (CATEGORICAL, INK, SEQUENTIAL_BLUE, MarkdownReport, add_caption, export_table, new_figure,
                        save_figure)
from .sources import SOURCES
from .tables import LayerRefs, sample_clause
from .utils import get_logger

log = get_logger(__name__)

# Tables profiled on a sample by default (large). Everything else is profiled exactly.
SAMPLED_TABLES = {"game_tracking", "combine_tracking"}
TRADITIONAL_TESTS = ["forty", "ten_yd_split", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps",
                     "hand_size", "arm_length", "wing_span", "combine_height", "combine_weight"]


class TableProfiler:
    def __init__(self, cfg: Phase01Config, con=None, layer: str = "bronze"):
        self.cfg = cfg
        self.con = con or duckdb_connect(cfg)
        self.refs = LayerRefs(cfg)
        self.layer = layer

    def _from(self, table: str, sample_pct: float | None) -> str:
        return f"{self.refs.sql(self.layer, table)}{sample_clause(sample_pct, self.cfg.profiling.seed)}"

    def summarize(self, table: str, sample_pct: float | None = None) -> pl.DataFrame:
        if sample_pct is None and table in SAMPLED_TABLES:
            sample_pct = self.cfg.profiling.tracking_sample_pct
        q = f"SUMMARIZE SELECT * EXCLUDE (_src_file) FROM {self._from(table, sample_pct)}"
        df = query_df(self.con, q)
        return df.with_columns(
            pl.lit(table).alias("table"),
            pl.lit(f"sample {sample_pct}% bernoulli seed={self.cfg.profiling.seed}" if sample_pct else "exact (full table)")
            .alias("profile_mode"),
        )

    def top_values(self, table: str, column: str, k: int | None = None, sample_pct: float | None = None) -> pl.DataFrame:
        k = k or self.cfg.profiling.top_k
        if sample_pct is None and table in SAMPLED_TABLES:
            sample_pct = self.cfg.profiling.tracking_sample_pct
        # The sample clause must sit in a subquery: "FROM t USING SAMPLE … GROUP BY" is a parser error.
        q = (f'SELECT CAST("{column}" AS VARCHAR) AS value, count(*) AS n '
             f"FROM (SELECT \"{column}\" FROM {self._from(table, sample_pct)}) "
             f"GROUP BY 1 ORDER BY n DESC, value LIMIT {k}")
        return query_df(self.con, q).with_columns(pl.lit(table).alias("table"), pl.lit(column).alias("column"))

    def profile_all(self, tables: list[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
        summaries, tops = [], []
        for name in tables or list(SOURCES):
            if not self.refs.exists(self.layer, name):
                log.warning("skip profile: %s.%s not built", self.layer, name)
                continue
            s = self.summarize(name)
            summaries.append(s)
            low_card = s.filter(
                pl.col("column_type").is_in(["VARCHAR"])
                & (pl.col("approx_unique") <= self.cfg.profiling.categorical_max_distinct)
            )["column_name"].to_list()
            for col in low_card:
                tops.append(self.top_values(name, col))
            log.info("profiled %s: %d columns (%d categorical top-k)", name, s.height, len(low_card))
        summary = pl.concat(summaries, how="diagonal_relaxed") if summaries else pl.DataFrame()
        top = pl.concat(tops, how="diagonal_relaxed") if tops else pl.DataFrame()
        return summary, top

    # ------------------------------------------------------------------ insight tables
    def combine_test_participation(self) -> pl.DataFrame:
        """Share of athletes with a recorded value for each traditional test, by Combine position.
        Missingness here is a selection mechanism (athletes skip tests), not random noise."""
        ref = self.refs.sql(self.layer, "combine_results")
        cols = ", ".join(f'avg(CASE WHEN "{c}" IS NULL THEN 0 ELSE 1 END) AS "{c}"' for c in TRADITIONAL_TESTS)
        q = f"SELECT combine_position, count(*) AS n_athletes, {cols} FROM {ref} GROUP BY 1 ORDER BY n_athletes DESC"
        return query_df(self.con, q)

    def tracking_drill_counts(self) -> pl.DataFrame:
        ref = self.refs.sql(self.layer, "combine_tracking")
        q = f"""
            SELECT draft_year, drill_type, drill_name,
                   count(DISTINCT nfl_id) AS n_athletes,
                   count(DISTINCT event_id || '|' || CAST(nfl_id AS VARCHAR) || '|' || CAST(attempt AS VARCHAR)) AS n_attempts,
                   count(*) AS n_rows
            FROM {ref} WHERE entity_type = 'PLAYER'
            GROUP BY ALL ORDER BY n_athletes DESC
        """
        return query_df(self.con, q)

    # ------------------------------------------------------------------ figures
    def plot_participation(self, part: pl.DataFrame) -> Path | None:
        if part.is_empty():
            return None
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap

        part = part.filter(pl.col("combine_position").is_not_null()).sort("n_athletes", descending=True)
        mat = part.select(TRADITIONAL_TESTS).to_numpy()
        fig, ax = new_figure()
        cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(TRADITIONAL_TESTS)), TRADITIONAL_TESTS, rotation=35, ha="right")
        labels = [f"{p} (n={n})" for p, n in zip(part["combine_position"], part["n_athletes"])]
        ax.set_yticks(range(len(labels)), labels)
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else INK["primary"])
        fig.colorbar(im, ax=ax, label="Share of athletes with a recorded value", shrink=0.8)
        ax.set_title("Traditional Combine test participation by position")
        add_caption(fig, "Source: bronze.combine_results (full table). Missing ≠ random: athletes opt out of tests.")
        return save_figure(fig, self.cfg.paths.figures, "03_combine_test_participation")

    def plot_drill_coverage(self, drills: pl.DataFrame, top_n: int = 25) -> Path | None:
        if drills.is_empty():
            return None
        d = (drills.group_by("drill_name").agg(pl.col("n_athletes").sum(), pl.col("n_attempts").sum())
             .sort("n_athletes", descending=True).head(top_n).sort("n_athletes"))
        fig, ax = new_figure()
        ax.barh(d["drill_name"].to_list(), d["n_athletes"].to_list(), color=CATEGORICAL[0], height=0.6)
        for i, (a, t) in enumerate(zip(d["n_athletes"], d["n_attempts"])):
            ax.text(a, i, f"  {a:,} athletes · {t:,} attempts", va="center", fontsize=8, color=INK["secondary"])
        ax.set_xlabel("Athletes with tracking (summed over draft years)")
        ax.set_title(f"Combine tracking coverage — top {min(top_n, d.height)} drills")
        ax.grid(axis="y", visible=False)
        add_caption(fig, "Source: bronze.combine_tracking, entity_type = PLAYER (exact counts).")
        return save_figure(fig, self.cfg.paths.figures, "03_combine_drill_coverage")

    # ------------------------------------------------------------------ outputs
    def export(self, summary: pl.DataFrame, top: pl.DataFrame, part: pl.DataFrame, drills: pl.DataFrame) -> None:
        t = self.cfg.paths.tables
        if not summary.is_empty():
            export_table(summary, t, "profile_summary")
        if not top.is_empty():
            export_table(top, t, "profile_top_values")
        if not part.is_empty():
            export_table(part, t, "combine_test_participation")
        if not drills.is_empty():
            export_table(drills, t, "combine_drill_counts")
