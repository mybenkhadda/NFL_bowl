"""Tracking integrity for both tracking domains (Pass A on whole-trajectory samples, or exact).

Per trajectory (game: game_id+play_id+nfl_id; combine: event_id+nfl_id+attempt):

* sampling interval distribution (Δt between consecutive timestamps)
* duplicate timestamps, non-monotonic time in original file order (``_src_row``)
* gaps (Δt > gap_factor × nominal) and estimated missing frames
* implausible jumps (displacement per frame above ``max_step_yds``)
* out-of-field coordinates, speed / acceleration above plausibility thresholds
* consistency of vendor ``s`` and ``dis`` with speed/distance derived from x, y, t

Sampling uses a stable hash of the trajectory key so every sampled trajectory is complete.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .config import Phase01Config, duckdb_connect
from .frames import query_df
from .reporting import CATEGORICAL, INK, add_caption, export_table, new_figure, save_figure
from .tables import LayerRefs, hash_sample_predicate
from .utils import get_logger
from .validate import CheckResult

log = get_logger(__name__)


@dataclass(frozen=True)
class TrackingDomain:
    name: str
    table: str
    key: tuple[str, ...]
    entity_filter: str
    check_field_bounds: bool


DOMAINS = {
    "game": TrackingDomain("game", "game_tracking", ("game_id", "play_id", "nfl_id"), "nfl_id IS NOT NULL", True),
    "combine": TrackingDomain("combine", "combine_tracking", ("event_id", "nfl_id", "attempt"),
                              "entity_type = 'PLAYER' AND nfl_id IS NOT NULL", False),
}


class TrackingQC:
    def __init__(self, cfg: Phase01Config, con=None, layer: str = "bronze"):
        self.cfg = cfg
        self.con = con or duckdb_connect(cfg)
        self.refs = LayerRefs(cfg)
        self.layer = layer

    # ------------------------------------------------------------------ SQL building blocks
    def _seq_sql(self, d: TrackingDomain, sample_pct: float | None) -> str:
        ref = self.refs.sql(self.layer, d.table)
        key = ", ".join(d.key)
        ts = "ts" if self.layer != "bronze" else "try_cast(time AS TIMESTAMP)"
        return f"""
            base AS (
                SELECT {key}, _src_row, {ts} AS ts, x, y, s, a, dis
                FROM {ref}
                WHERE {d.entity_filter} AND {hash_sample_predicate(d.key, sample_pct)}
            ),
            seq AS (
                SELECT *,
                    epoch_ms(ts) - epoch_ms(lag(ts) OVER w) AS dt_ms,
                    sqrt(power(x - lag(x) OVER w, 2) + power(y - lag(y) OVER w, 2)) AS step_yds,
                    CASE WHEN ts < lag(ts) OVER wr THEN 1 ELSE 0 END AS nonmono
                FROM base
                WINDOW w AS (PARTITION BY {key} ORDER BY ts, _src_row),
                       wr AS (PARTITION BY {key} ORDER BY _src_row)
            )
        """

    def _pct(self, sample_pct: float | None) -> float | None:
        return self.cfg.profiling.tracking_sample_pct if sample_pct is None else (sample_pct or None)

    # ------------------------------------------------------------------ analyses
    def interval_distribution(self, domain: str, sample_pct: float | None = None) -> pl.DataFrame:
        d, pct = DOMAINS[domain], self._pct(sample_pct)
        q = f"""
            WITH {self._seq_sql(d, pct)}
            SELECT least(dt_ms, 2000) AS dt_ms_capped, count(*) AS n
            FROM seq WHERE dt_ms IS NOT NULL GROUP BY 1 ORDER BY 1
        """
        return query_df(self.con, q).with_columns(pl.lit(domain).alias("domain"), pl.lit(pct or 100.0).alias("sample_pct"))

    def trajectory_metrics(self, domain: str, sample_pct: float | None = None) -> pl.DataFrame:
        d, pct = DOMAINS[domain], self._pct(sample_pct)
        tr = self.cfg.tracking
        nom, gap = tr.nominal_interval_ms, tr.gap_factor
        tol = tr.coord_tolerance_yds
        oob = (f"x < {-tol} OR x > {tr.field_length_yds + tol} OR y < {-tol} OR y > {tr.field_width_yds + tol}"
               if d.check_field_bounds else "FALSE")
        q = f"""
            WITH {self._seq_sql(d, pct)}
            SELECT {', '.join(d.key)},
                count(*) AS n_frames,
                (epoch_ms(max(ts)) - epoch_ms(min(ts))) / 1000.0 AS duration_s,
                count_if(ts IS NULL) AS n_null_ts,
                count_if(dt_ms = 0) AS n_dup_ts,
                count_if(nonmono = 1) AS n_nonmono,
                count_if(dt_ms > {nom * gap}) AS n_gaps,
                CAST(coalesce(sum(CASE WHEN dt_ms > {nom * gap} THEN round(dt_ms / {nom}) - 1 END), 0) AS BIGINT)
                    AS n_missing_frames_est,
                count_if(step_yds > {tr.max_step_yds} * greatest(dt_ms / {nom}, 1)) AS n_jumps,
                count_if({oob}) AS n_out_of_bounds,
                count_if(s > {tr.max_speed_yds}) AS n_speed_gt_max,
                count_if(a > {tr.max_accel_yds2}) AS n_accel_gt_max,
                max(s) AS max_s, max(a) AS max_a, max(step_yds) AS max_step_yds,
                median(dt_ms) AS median_dt_ms
            FROM seq GROUP BY ALL
        """
        return query_df(self.con, q)

    def kinematic_consistency(self, domain: str, sample_pct: float | None = None) -> pl.DataFrame:
        """Vendor ``s`` vs derived speed (step / Δt) and ``dis`` vs derived step, on regular
        frames only (Δt = nominal ± 25 %). Large disagreement implies smoothing, a different
        reference point, or unit differences — it decides whether we may re-derive kinematics."""
        d, pct = DOMAINS[domain], self._pct(sample_pct)
        nom = self.cfg.tracking.nominal_interval_ms
        q = f"""
            WITH {self._seq_sql(d, pct)}, reg AS (
                SELECT s, dis, step_yds, step_yds / (dt_ms / 1000.0) AS s_derived
                FROM seq WHERE dt_ms BETWEEN {nom * 0.75} AND {nom * 1.25}
            )
            SELECT count(*) AS n_frames,
                corr(s, s_derived) AS corr_s,
                median(abs(s - s_derived)) AS median_abs_diff_s,
                quantile_cont(abs(s - s_derived), 0.95) AS p95_abs_diff_s,
                corr(dis, step_yds) AS corr_dis,
                median(abs(dis - step_yds)) AS median_abs_diff_dis,
                median(s) AS median_s, median(s_derived) AS median_s_derived
            FROM reg
        """
        return query_df(self.con, q).with_columns(pl.lit(domain).alias("domain"), pl.lit(pct or 100.0).alias("sample_pct"))

    def speed_pairs_sample(self, domain: str, n: int = 50_000, sample_pct: float | None = None) -> pl.DataFrame:
        d, pct = DOMAINS[domain], self._pct(sample_pct)
        nom = self.cfg.tracking.nominal_interval_ms
        q = f"""
            WITH {self._seq_sql(d, pct)}, reg AS (
                SELECT s, step_yds / (dt_ms / 1000.0) AS s_derived
                FROM seq WHERE dt_ms BETWEEN {nom * 0.75} AND {nom * 1.25}
            )
            SELECT * FROM reg USING SAMPLE reservoir({n} ROWS) REPEATABLE ({self.cfg.profiling.seed})
        """
        return query_df(self.con, q)

    def flagged_examples(self, domain: str, limit: int = 200, sample_pct: float | None = None) -> pl.DataFrame:
        d, pct = DOMAINS[domain], self._pct(sample_pct)
        tr = self.cfg.tracking
        q = f"""
            WITH {self._seq_sql(d, pct)}
            SELECT *, CASE WHEN dt_ms = 0 THEN 'duplicate_ts'
                           WHEN nonmono = 1 THEN 'non_monotonic'
                           WHEN step_yds > {tr.max_step_yds} THEN 'jump'
                           WHEN s > {tr.max_speed_yds} THEN 'speed'
                           ELSE 'gap' END AS flag
            FROM seq
            WHERE dt_ms = 0 OR nonmono = 1 OR step_yds > {tr.max_step_yds} OR s > {tr.max_speed_yds}
               OR dt_ms > {tr.nominal_interval_ms * tr.gap_factor}
            ORDER BY {', '.join(d.key)}, ts LIMIT {limit}
        """
        return query_df(self.con, q)

    # ------------------------------------------------------------------ check results
    def to_checks(self, domain: str, metrics: pl.DataFrame, seconds: float, pct: float | None) -> list[CheckResult]:
        d = DOMAINS[domain]
        n_traj = metrics.height
        n_frames = int(metrics["n_frames"].sum()) if n_traj else 0
        mode = f"sample {pct}% of trajectories" if pct else "exact"
        out = []

        def add(check, col, value, denom, severity, threshold, action, fail_if_positive=True):
            frac = value / max(denom, 1)
            result = "PASS" if value == 0 else ("FAIL" if fail_if_positive and severity in ("CRITICAL", "HIGH") else "WARN")
            out.append(CheckResult(
                f"trk:{domain}:{col}", check, d.table, col, result, severity, col, float(value), float(denom),
                threshold, f"{value:,} of {denom:,} ({frac:.4%}) over {n_traj:,} trajectories", action, mode,
                seconds, "see tracking_qc.trajectory_metrics"))

        s = lambda c: int(metrics[c].sum()) if n_traj else 0  # noqa: E731
        add("Frame consistency: duplicate timestamps", "n_dup_ts", s("n_dup_ts"), n_frames, "HIGH", "0",
            "Dedup exact duplicates in silver; investigate conflicting duplicates.")
        add("Frame consistency: non-monotonic time (file order)", "n_nonmono", s("n_nonmono"), n_frames, "MEDIUM", "0",
            "Always ORDER BY ts in silver; never rely on file order.")
        add("Frame consistency: unparseable timestamps", "n_null_ts", s("n_null_ts"), n_frames, "CRITICAL", "0",
            "Fix timestamp parsing before silver.")
        add("Frame consistency: gaps (missing frames)", "n_gaps", s("n_gaps"), n_frames, "MEDIUM", "0",
            "Keep gaps; derived kinematics must use true Δt, not frame counts.")
        add("Trajectory: implausible jumps", "n_jumps", s("n_jumps"), n_frames, "MEDIUM", f"> {self.cfg.tracking.max_step_yds} yd/frame",
            "Flag frames; exclude from peak-speed features or smooth.")
        add("Trajectory: speed above plausibility", "n_speed_gt_max", s("n_speed_gt_max"), n_frames, "MEDIUM",
            f"> {self.cfg.tracking.max_speed_yds} yd/s", "Flag; use robust peaks (e.g. p99 or sustained 0.5 s max).")
        add("Trajectory: acceleration above plausibility", "n_accel_gt_max", s("n_accel_gt_max"), n_frames, "LOW",
            f"> {self.cfg.tracking.max_accel_yds2} yd/s²", "Flag; prefer smoothed acceleration features.")
        if d.check_field_bounds:
            add("Coordinates: outside field (+tolerance)", "n_out_of_bounds", s("n_out_of_bounds"), n_frames, "LOW",
                f"±{self.cfg.tracking.coord_tolerance_yds} yd", "Flag; players legitimately leave the field on sidelines.")
        return out

    def run(self, domains: tuple[str, ...] = ("game", "combine"), sample_pct: float | None = None) -> dict[str, pl.DataFrame]:
        """Returns dict of result frames; also exports tables and figures."""
        out: dict[str, list[pl.DataFrame]] = {"intervals": [], "consistency": [], "summary": []}
        checks: list[CheckResult] = []
        pct = self._pct(sample_pct)
        figs = []
        for dom in domains:
            if not self.refs.exists(self.layer, DOMAINS[dom].table):
                log.warning("tracking QC skip: %s not built", DOMAINS[dom].table)
                continue
            t0 = time.perf_counter()
            metrics = self.trajectory_metrics(dom, pct)
            seconds = round(time.perf_counter() - t0, 2)
            checks += self.to_checks(dom, metrics, seconds, pct)
            export_table(metrics, self.cfg.paths.tables, f"tracking_qc_trajectories_{dom}")
            out["intervals"].append(self.interval_distribution(dom, pct))
            out["consistency"].append(self.kinematic_consistency(dom, pct))
            out["summary"].append(metrics.select(
                pl.lit(dom).alias("domain"), pl.len().alias("n_trajectories"), pl.col("n_frames").sum(),
                pl.col("n_frames").median().alias("median_frames"), pl.col("duration_s").median().alias("median_duration_s"),
                pl.col("median_dt_ms").median().alias("median_dt_ms"),
                (pl.col("n_gaps") > 0).mean().alias("share_traj_with_gaps"),
                (pl.col("n_dup_ts") > 0).mean().alias("share_traj_with_dup_ts"),
                (pl.col("n_jumps") > 0).mean().alias("share_traj_with_jumps"),
                pl.col("max_s").quantile(0.99).alias("p99_traj_max_s"),
            ))
            examples = self.flagged_examples(dom, sample_pct=pct)
            if not examples.is_empty():
                export_table(examples, self.cfg.paths.tables, f"tracking_qc_flagged_rows_sample_{dom}")
            figs.append(self.plot_speed_consistency(dom, self.speed_pairs_sample(dom, sample_pct=pct)))
        frames = {k: pl.concat(v, how="diagonal_relaxed") for k, v in out.items() if v}
        frames["checks"] = pl.DataFrame([c.__dict__ for c in checks]) if checks else pl.DataFrame()
        for k, df in frames.items():
            if not df.is_empty():
                export_table(df, self.cfg.paths.tables, f"tracking_qc_{k}")
        if "intervals" in frames:
            figs.append(self.plot_intervals(frames["intervals"]))
        frames["_figures"] = pl.DataFrame({"path": [str(f) for f in figs if f]})
        return frames

    # ------------------------------------------------------------------ figures
    def plot_intervals(self, intervals: pl.DataFrame) -> Path | None:
        doms = intervals["domain"].unique(maintain_order=True).to_list()
        fig, axes = new_figure(1, len(doms), squeeze=False)
        nom = self.cfg.tracking.nominal_interval_ms
        for ax, dom, color in zip(axes[0], doms, CATEGORICAL):
            d = intervals.filter(pl.col("domain") == dom).sort("dt_ms_capped")
            share = d["n"] / d["n"].sum()
            ax.bar(d["dt_ms_capped"].to_list(), share.to_list(), width=8, color=color)
            ax.set_yscale("log")
            ax.axvline(nom, color=INK["secondary"], linestyle="--", linewidth=1)
            ax.text(nom, ax.get_ylim()[1], f" nominal {nom} ms", va="top", fontsize=9, color=INK["secondary"])
            ax.set_xlim(-20, min(max(d["dt_ms_capped"].max() or nom, 4 * nom), 2000) + 20)
            ax.set_xlabel("Δt between consecutive frames (ms, capped at 2000)")
            ax.set_ylabel("Share of frame pairs (log)")
            ax.set_title(f"{dom.capitalize()} tracking — sampling interval")
        add_caption(fig, f"Source: {self.layer} tracking, whole-trajectory hash sample ({intervals['sample_pct'][0]}%).")
        return save_figure(fig, self.cfg.paths.figures, "03_tracking_sampling_intervals")

    def plot_speed_consistency(self, domain: str, pairs: pl.DataFrame) -> Path | None:
        if pairs.is_empty():
            return None
        fig, ax = new_figure(figsize=(9.6, 7.2))
        hb = ax.hexbin(pairs["s_derived"].to_numpy(), pairs["s"].to_numpy(), gridsize=70, bins="log", cmap="Blues",
                       extent=(0, 13, 0, 13), mincnt=1)
        ax.plot([0, 13], [0, 13], color=INK["secondary"], linestyle="--", linewidth=1)
        ax.text(12.8, 12.3, "s = derived", ha="right", fontsize=9, color=INK["secondary"])
        ax.set_xlabel("Speed derived from Δposition / Δt (yd/s)")
        ax.set_ylabel("Reported speed s (yd/s)")
        ax.set_title(f"{domain.capitalize()} tracking — reported vs derived speed")
        fig.colorbar(hb, ax=ax, label="Frames (log)")
        add_caption(fig, f"n = {pairs.height:,} regular frames (reservoir sample). Off-diagonal mass ⇒ smoothing or noise.")
        return save_figure(fig, self.cfg.paths.figures, f"03_speed_consistency_{domain}")
