"""Study A3 — derivative-signal reliability and filter-boundary contamination (Phase 03 · M1.1, WS A2/A3).

A3.1 · Signal metrics
    Question     Separately from structural validity, how reliable are the DERIVED signals of each
                 trajectory (sampling regularity, missing frames, vendor residual, jerk roughness,
                 heading definability, filter-boundary share)?
    Population   all persisted Combine attempts (frames persisted) + the deterministic NFL game sample
                 (frames recomputed; same sample as Study A2). Grain: trajectory; unit: athlete.

A3.2 · Filter-boundary contamination
    Question     Do peak features (speed, a_tan, deceleration, |a|, |yaw rate|) land on SG filter-
                 boundary frames more often than the boundary share predicts?
    Method       observed share of trajectories whose peak lies in the boundary vs the expected share
                 (mean ``boundary_frac``) — ratio > 1 means peaks concentrate at segment edges, where
                 the off-centre SG fit has a larger noise gain. Athlete-cluster bootstrap CI on the
                 observed share; distribution of the interior-vs-all peak change (``*_boundary_rel``).
    Caveat       NOT automatically an artefact: the forty recording starts in motion and ends at peak
                 speed (M1 evidence), so true peaks can sit at an edge. Interpretation per drill.

A3.3 · Do tiers / quality metrics discriminate unreliable signals?
    Targets      "unreliable" = top (1 − unreliable_quantile) of vendor_rmse or of jerk_rms_interior
                 within domain × stratum (PROPOSED cut; continuous metrics exported), or a peak a_tan /
                 deceleration located in the filter boundary.
    Predictors   tier ≠ A, Study A1 n_flagged_dims (if Study A ran), and continuous quality metrics.
    Statistic    AUC = P(predictor higher | unreliable) + ½ P(tie) = (Cliff's δ + 1) / 2, with an
                 athlete-cluster bootstrap CI. 0.5 = no discrimination. ``shares_source`` marks
                 predictor/target pairs built from the same quantity (mechanically related — not
                 evidence of validity).

Outputs: see CONTRACTS; figures A3_01…A3_03.

Command for the USER (code authored, NOT executed)::

    python -m nflbdb.research.studies --only A3 [--sample-athletes 300] [--skip-nfl-sample]
"""

from __future__ import annotations

import dataclasses
import zlib
from typing import Any

import numpy as np
import polars as pl

from ...data.config import Phase01Config
from ...data.utils import get_logger
from ...features.config import KinematicsConfig
from ...features.signal_integrity import PEAK_FEATURES, add_signal_columns, trajectory_signal_metrics
from ..contracts import TableContract, cast_keys, upstream_state
from ..kinematic_build import COMBINE_FRAMES, COMBINE_TRAJ, KinematicFeatureBuilder
from .config import SignalStudyConfig, StudiesConfig
from .quality import DIMS, traj_id_expr
from .results import RESULTS, ResultCollector
from .stats import cliffs_delta, cluster_bootstrap, cluster_bootstrap_ratio
from .store import FigureRecord, StudyStore

log = get_logger(__name__)

STUDY = "A3"
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
CKEYS, NKEYS = ["event_id", "nfl_id", "attempt"], ["game_id", "play_id", "nfl_id"]
PEAKS = list(PEAK_FEATURES)
SIGNAL_METRICS = ["usable_frac", "dt_cv", "dt_max_dev_ms", "offgrid_rate", "missing_frame_pct", "vendor_rmse",
                  "vendor_mae", "vendor_bias", "vendor_p95_abs", "jerk_rms_interior", "jerk_rms_all",
                  "jerk_p95_interior", "lowspeed_frac", "heading_null_frac", "boundary_frac"]
INT_METRICS = ["n_frames", "n_valid_kin", "n_segments", "n_gap_breaks", "n_jump_breaks", "n_offgrid",
               "n_missing_frames", "n_vendor_frames"]
# predictor → (column, higher = worse?, source group). Same source group as the target → shares_source.
PREDICTORS: dict[str, tuple[str, str]] = {
    "tier_not_A": ("tier_not_A", "tier"), "n_flagged_dims": ("n_flagged_dims", "a1_flags"),
    "resid_rms_yds": ("resid_rms_yds", "position_noise"), "speed_vendor_mad": ("speed_vendor_mad", "vendor"),
    "offgrid_rate": ("offgrid_rate", "sampling"), "missing_frame_pct": ("missing_frame_pct", "sampling"),
    "dt_cv": ("dt_cv", "sampling"), "boundary_frac": ("boundary_frac", "boundary"),
    "unusable_frac": ("unusable_frac", "structure"),
}
TARGETS: dict[str, str] = {"vendor_rmse_high": "vendor", "jerk_high": "roughness",
                           "peak_a_tan_in_boundary": "boundary", "peak_decel_in_boundary": "boundary"}

# ============================================================================== contracts
SIGNALS = TableContract(
    "trajectory_signal_reliability", "One row per trajectory: derivative-signal reliability metrics (Study A3).",
    {"domain": S, "traj_id": S, "nfl_id": I, "cohort": I, "stratum": S, "position": S, "quality_tier": S,
     "duration_s": F, "resid_rms_yds": F, "speed_vendor_mad": F, "n_flagged_dims": I,
     **{c: I for c in INT_METRICS}, **{c: F for c in SIGNAL_METRICS},
     **{p: F for p in PEAKS}, **{f"{p}_interior": F for p in PEAKS}, **{f"{p}_in_boundary": B for p in PEAKS},
     **{f"{p}_boundary_rel": F for p in PEAKS}},
    key=("domain", "traj_id"), kind="dataset", area="quality",
    docs={"n_flagged_dims": "Study A1 screening flags (null if Study A has not run or for the NFL sample)",
          "resid_rms_yds": "M1 trajectory residual RMS (structural / position noise)",
          "speed_vendor_mad": "M1 median |derived − vendor speed|"})
SUMMARY = TableContract(
    "A3_signal_metric_summary", "Quantiles of every signal metric by domain × stratum × tier.",
    {"domain": S, "stratum": S, "quality_tier": S, "metric": S, "n_trajectories": I, "n_athletes": I, "mean": F,
     "p05": F, "p25": F, "median": F, "p75": F, "p95": F, "max": F},
    key=("domain", "stratum", "quality_tier", "metric"), area="descriptive")
BOUNDARY = TableContract(
    "A3_boundary_peak_share", "Share of trajectories whose peak lies in the SG filter boundary vs the expected share.",
    {"domain": S, "stratum": S, "peak": S, "n_trajectories": I, "n_athletes": I, "share_in_boundary": F,
     "boot_lower": F, "boot_upper": F, "expected_share": F, "ratio_observed_expected": F,
     "median_boundary_rel": F, "p90_boundary_rel": F},
    key=("domain", "stratum", "peak"), area="descriptive",
    docs={"expected_share": "mean boundary_frac (share of valid frames in the boundary) — the share expected if "
                            "peaks were located at random frames",
          "ratio_observed_expected": "> 1: peaks concentrate on boundary frames"})
SAMPLING = TableContract(
    "A3_sampling_irregularity", "Sampling regularity, missing frames and heading consistency by domain × cohort × stratum.",
    {"domain": S, "cohort": I, "stratum": S, "n_trajectories": I, "n_athletes": I, "median_dt_cv": F, "p95_dt_cv": F,
     "share_offgrid_any": F, "mean_missing_frame_pct": F, "p95_missing_frame_pct": F, "share_gap_break": F,
     "share_jump_break": F, "median_lowspeed_frac": F, "n_heading_inconsistent": I, "median_jerk_rms_interior": F,
     "median_vendor_rmse": F},
    key=("domain", "cohort", "stratum"), area="descriptive",
    docs={"n_heading_inconsistent": "trajectories where heading_null_frac ≠ lowspeed_frac (should be 0: heading "
                                    "is null exactly when speed < min_heading_speed)"})
DISCRIM = TableContract(
    "A3_tier_discrimination", "AUC of each quality predictor for each 'unreliable signal' target (athlete-cluster CI).",
    {"domain": S, "predictor": S, "target": S, "shares_source": B, "n_trajectories": I, "n_positive": I,
     "n_athletes": I, "auc": F, "ci_lower": F, "ci_upper": F, "result_status": S},
    key=("domain", "predictor", "target"), area="statistical",
    docs={"auc": "P(predictor higher | unreliable) + ½ P(tie); 0.5 = no discrimination",
          "shares_source": "predictor and target derive from the same quantity (mechanical association)"})
STUDY_RESULTS = dataclasses.replace(RESULTS, name="A3_signal_results")
CONTRACTS = (SIGNALS, SUMMARY, BOUNDARY, SAMPLING, DISCRIM, STUDY_RESULTS)


def _seed(base: int, *parts: Any) -> int:
    return (base + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 32)


# ============================================================================== pure analysis
def normalize(metrics: pl.DataFrame, traj: pl.DataFrame, keys: list[str], domain: str, cohort_col: str,
              stratum_col: str | None, positions: pl.DataFrame, a1: pl.DataFrame | None) -> pl.DataFrame:
    """Signal metrics + M1 trajectory context → SIGNALS schema."""
    ctx_cols = [c for c in (cohort_col, stratum_col, "quality_tier", "resid_rms_yds", "speed_vendor_mad") if c]
    t = cast_keys(metrics).join(cast_keys(traj.select(*keys, *ctx_cols)), on=keys, how="left")
    t = t.with_columns(traj_id_expr(keys)).join(positions, on="nfl_id", how="left")
    if a1 is not None and a1.height:
        t = t.join(a1.filter(pl.col("domain") == domain).select("traj_id", "n_flagged_dims"), on="traj_id", how="left")
    else:
        t = t.with_columns(pl.lit(None, dtype=pl.Int64).alias("n_flagged_dims"))
    return t.select(
        pl.lit(domain).alias("domain"), "traj_id", pl.col("nfl_id").cast(pl.Int64),
        pl.col(cohort_col).cast(pl.Int64).alias("cohort"),
        (pl.col(stratum_col).cast(pl.String) if stratum_col else pl.lit("nfl_play")).alias("stratum"),
        pl.col("position_group").fill_null("UNKNOWN").alias("position"),
        pl.col("quality_tier").cast(pl.String).fill_null("unknown"),
        pl.col("duration_s").cast(pl.Float64), pl.col("resid_rms_yds").cast(pl.Float64),
        pl.col("speed_vendor_mad").cast(pl.Float64), pl.col("n_flagged_dims").cast(pl.Int64),
        *[pl.col(c).cast(pl.Int64) for c in INT_METRICS], *[pl.col(c).cast(pl.Float64) for c in SIGNAL_METRICS],
        *[pl.col(p).cast(pl.Float64) for p in PEAKS], *[pl.col(f"{p}_interior").cast(pl.Float64) for p in PEAKS],
        *[pl.col(f"{p}_in_boundary").cast(pl.Boolean) for p in PEAKS],
        *[pl.col(f"{p}_boundary_rel").cast(pl.Float64) for p in PEAKS])


def metric_summary(t: pl.DataFrame) -> pl.DataFrame:
    metrics = [*SIGNAL_METRICS, *[f"{p}_boundary_rel" for p in PEAKS]]
    long = t.unpivot(index=["domain", "stratum", "quality_tier", "nfl_id"], on=metrics, variable_name="metric",
                     value_name="v").filter(pl.col("v").is_not_null() & pl.col("v").is_finite())
    parts = [long, long.with_columns(pl.lit("all").alias("quality_tier"))]
    q = lambda p: pl.col("v").quantile(p, "linear")  # noqa: E731
    return (pl.concat(parts).group_by("domain", "stratum", "quality_tier", "metric").agg(
        pl.len().cast(pl.Int64).alias("n_trajectories"), pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"),
        pl.col("v").mean().alias("mean"), q(0.05).alias("p05"), q(0.25).alias("p25"), q(0.5).alias("median"),
        q(0.75).alias("p75"), q(0.95).alias("p95"), pl.col("v").max().alias("max"))
        .sort("domain", "stratum", "quality_tier", "metric"))


def boundary_peak_share(t: pl.DataFrame, cfg: SignalStudyConfig, seed: int) -> pl.DataFrame:
    rows = []
    for (domain, stratum), g in t.group_by("domain", "stratum", maintain_order=True):
        for p in PEAKS:
            d = g.filter(pl.col(f"{p}_in_boundary").is_not_null())
            if not d.height:
                continue
            per_ath = d.group_by("nfl_id").agg(pl.col(f"{p}_in_boundary").sum().alias("k"), pl.len().alias("n"))
            ci = (cluster_bootstrap_ratio(per_ath["k"].to_numpy(), per_ath["n"].to_numpy(), cfg.n_boot,
                                          _seed(seed, domain, stratum, p))
                  if per_ath.height >= cfg.min_athletes else {"estimate": float(d[f"{p}_in_boundary"].mean()),
                                                              "ci_lower": None, "ci_upper": None})
            exp = d["boundary_frac"].mean()
            rel = d[f"{p}_boundary_rel"].drop_nulls()
            rows.append({"domain": domain, "stratum": stratum, "peak": p, "n_trajectories": d.height,
                         "n_athletes": per_ath.height, "share_in_boundary": ci["estimate"],
                         "boot_lower": ci["ci_lower"], "boot_upper": ci["ci_upper"], "expected_share": exp,
                         "ratio_observed_expected": (ci["estimate"] / exp) if exp else None,
                         "median_boundary_rel": rel.median() if rel.len() else None,
                         "p90_boundary_rel": rel.quantile(0.9, "linear") if rel.len() else None})
    return pl.DataFrame(rows, schema=BOUNDARY.columns) if rows else BOUNDARY.empty()


def sampling_irregularity(t: pl.DataFrame) -> pl.DataFrame:
    return (t.group_by("domain", "cohort", "stratum").agg(
        pl.len().cast(pl.Int64).alias("n_trajectories"), pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"),
        pl.col("dt_cv").median().alias("median_dt_cv"), pl.col("dt_cv").quantile(0.95, "linear").alias("p95_dt_cv"),
        (pl.col("n_offgrid") > 0).mean().alias("share_offgrid_any"),
        pl.col("missing_frame_pct").mean().alias("mean_missing_frame_pct"),
        pl.col("missing_frame_pct").quantile(0.95, "linear").alias("p95_missing_frame_pct"),
        (pl.col("n_gap_breaks") > 0).mean().alias("share_gap_break"),
        (pl.col("n_jump_breaks") > 0).mean().alias("share_jump_break"),
        pl.col("lowspeed_frac").median().alias("median_lowspeed_frac"),
        ((pl.col("heading_null_frac") - pl.col("lowspeed_frac")).abs() > 1e-9).sum().cast(pl.Int64)
        .alias("n_heading_inconsistent"),
        pl.col("jerk_rms_interior").median().alias("median_jerk_rms_interior"),
        pl.col("vendor_rmse").median().alias("median_vendor_rmse"))
        .sort("domain", "cohort", "stratum"))


def add_targets(t: pl.DataFrame, cfg: SignalStudyConfig) -> pl.DataFrame:
    grp = ["domain", "stratum"]
    hi = lambda c: (pl.col(c) > pl.col(c).quantile(cfg.unreliable_quantile, "linear").over(grp))  # noqa: E731
    return t.with_columns(
        pl.when(pl.col("vendor_rmse").is_not_null()).then(hi("vendor_rmse")).alias("vendor_rmse_high"),
        pl.when(pl.col("jerk_rms_interior").is_not_null()).then(hi("jerk_rms_interior")).alias("jerk_high"),
        pl.col("peak_a_tan_in_boundary").alias("peak_a_tan_in_boundary"),
        pl.col("peak_decel_in_boundary").alias("peak_decel_in_boundary"),
        (pl.col("quality_tier") != "A").cast(pl.Float64).alias("tier_not_A"),
        (1.0 - pl.col("usable_frac")).alias("unusable_frac"),
    )


def auc(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    d = cliffs_delta(x_pos, x_neg)
    return (d + 1.0) / 2.0 if np.isfinite(d) else float("nan")


def tier_discrimination(t: pl.DataFrame, cfg: SignalStudyConfig, seed: int, rc: ResultCollector) -> pl.DataFrame:
    t = add_targets(t, cfg)
    rows = []
    for (domain,), g in t.group_by("domain", maintain_order=True):
        for target, tsrc in TARGETS.items():
            for pred, (col, psrc) in PREDICTORS.items():
                d = g.filter(pl.col(target).is_not_null() & pl.col(col).is_not_null()).select(
                    "nfl_id", pl.col(target).cast(pl.Boolean).alias("y"), pl.col(col).cast(pl.Float64).alias("x"))
                n_pos = int(d["y"].sum()) if d.height else 0
                n_ath = d["nfl_id"].n_unique() if d.height else 0
                status = "exploratory"
                est = lo = hi = None
                if d.height < cfg.min_group_trajectories or n_pos == 0 or n_pos == d.height:
                    status = "insufficient_data"
                else:
                    x, y = d["x"].to_numpy(), d["y"].to_numpy()
                    stat = lambda idx: auc(x[idx][y[idx]], x[idx][~y[idx]])  # noqa: E731
                    if n_ath >= cfg.min_athletes:
                        bs = cluster_bootstrap(stat, d["nfl_id"].to_numpy(), cfg.n_boot, _seed(seed, domain, target, pred))
                        est, lo, hi = bs["estimate"], bs["ci_lower"], bs["ci_upper"]
                    else:
                        est, status = stat(np.arange(d.height)), "insufficient_data"
                shares = psrc == tsrc
                rows.append({"domain": domain, "predictor": pred, "target": target, "shares_source": shares,
                             "n_trajectories": d.height, "n_positive": n_pos, "n_athletes": n_ath, "auc": est,
                             "ci_lower": lo, "ci_upper": hi, "result_status": status})
                if est is not None:
                    rc.add(analysis_id="A3.discrimination", grain="trajectory", n_athletes=n_ath,
                           n_trajectories=d.height, effect_estimate=est, effect_size_type="AUC", method="(Cliff's δ+1)/2",
                           domain=domain, feature=pred, outcome=target, ci_lower=lo, ci_upper=hi,
                           ci_method=f"athlete-cluster bootstrap (B={cfg.n_boot}, percentile)",
                           quality_filter="none (all tiers)", result_status=status,
                           limitations=("predictor and target share a source (mechanical)" if shares else
                                        f"target cut at the domain × stratum q{cfg.unreliable_quantile} (proposed)"))
    return pl.DataFrame(rows, schema=DISCRIM.columns) if rows else DISCRIM.empty()


def boundary_results(b: pl.DataFrame, rc: ResultCollector) -> None:
    for r in b.iter_rows(named=True):
        rc.add(analysis_id="A3.boundary_peak_share", grain="trajectory", n_athletes=r["n_athletes"],
               n_trajectories=r["n_trajectories"], effect_estimate=r["share_in_boundary"], effect_size_type="proportion",
               method="share of trajectories whose peak frame lies in the SG filter boundary", domain=r["domain"],
               stratum=r["stratum"], feature=r["peak"], ci_lower=r["boot_lower"], ci_upper=r["boot_upper"],
               ci_method="athlete-cluster bootstrap (percentile)", quality_filter="none (all tiers)",
               limitations=f"expected share under random location = {r['expected_share']:.3f}"
                           if r["expected_share"] is not None else "")


# ============================================================================== orchestration
class SignalReliabilityStudy:
    STUDY = STUDY

    def __init__(self, cfg: Phase01Config, scfg: StudiesConfig | None = None, store: StudyStore | None = None,
                 kcfg: KinematicsConfig | None = None, skip_nfl_sample: bool = False, **_: Any):
        self.cfg = cfg
        self.scfg = scfg or StudiesConfig()
        self.s = self.scfg.signal
        self.kcfg = kcfg or KinematicsConfig.from_tracking(cfg.tracking)
        self.kin = KinematicFeatureBuilder(cfg, self.kcfg)
        self.src = self.kin.store
        self.store = store or StudyStore(cfg, sampled=self.scfg.sample_athletes is not None)
        self.skip_nfl_sample = skip_nfl_sample

    def positions(self) -> pl.DataFrame:
        if not self.src.refs.files("gold", "dim_athlete"):
            return pl.DataFrame(schema={"nfl_id": pl.Int64, "position_group": pl.String})
        return self.src.refs.scan("gold", "dim_athlete").select(pl.col("nfl_id").cast(pl.Int64),
                                                                 pl.col("position_group").cast(pl.String)).collect()

    def a1(self) -> pl.DataFrame | None:
        try:
            return self.store.read(DIMS)
        except FileNotFoundError:
            log.warning("[studies:A3] Study A dimensions not found — n_flagged_dims predictor is null")
            return None

    def sample_ids(self) -> list[int] | None:
        if self.scfg.sample_athletes is None:
            return None
        ids = np.sort(self.src.load(COMBINE_TRAJ.name).select("nfl_id").unique().collect()["nfl_id"].to_numpy())
        k = min(self.scfg.sample_athletes, ids.size)
        return sorted(np.random.default_rng(self.scfg.seed).choice(ids, size=k, replace=False).tolist())

    def combine(self, pos: pl.DataFrame, a1: pl.DataFrame | None, ids: list[int] | None) -> pl.DataFrame:
        cols = [*CKEYS, "seg_idx", "pos", "seg_len", "t_s", "dt_ms", "break_reason", "s", "speed", "ax", "ay",
                "accel_mag", "a_tan", "yaw_rate", "heading", "valid_kin"]
        lf = self.src.load(COMBINE_FRAMES.name).select(cols)
        fr = (lf.filter(pl.col("nfl_id").is_in(ids)) if ids is not None else lf).collect()
        m = trajectory_signal_metrics(add_signal_columns(cast_keys(fr), CKEYS, self.kcfg), CKEYS, self.kcfg)
        tq = self.src.load(COMBINE_TRAJ.name).collect()
        return normalize(m, tq, CKEYS, "combine", "draft_year", "drill_type", pos, a1)

    def nfl_sample(self, pos: pl.DataFrame, ids: list[int] | None) -> pl.DataFrame | None:
        if self.skip_nfl_sample:
            return None
        from ..kinematic_diagnostics import KinematicDiagnostics

        s = KinematicDiagnostics(self.cfg, self.kcfg, self.src, nfl_games_per_season=self.s.nfl_games_per_season).nfl_sample()
        fr = s["frames"] if ids is None else s["frames"].filter(pl.col("nfl_id").is_in(ids))
        m = trajectory_signal_metrics(add_signal_columns(cast_keys(fr), NKEYS, self.kcfg), NKEYS, self.kcfg)
        return normalize(m, s["trajectories"], NKEYS, "nfl_sample", "season", None, pos, None)

    def compute(self) -> dict[str, pl.DataFrame]:
        seed = self.scfg.seed
        pos, a1, ids = self.positions(), self.a1(), self.sample_ids()
        parts = [self.combine(pos, a1, ids)]
        n = self.nfl_sample(pos, ids)
        if n is not None:
            parts.append(n)
        t = pl.concat(parts, how="vertical")
        log.info("[studies:A3] %s trajectories (%s)", f"{t.height:,}", t.group_by("domain").len().rows())
        rc = ResultCollector(STUDY, self.s.min_athletes)
        b = boundary_peak_share(t, self.s, seed)
        boundary_results(b, rc)
        disc = tier_discrimination(t, self.s, seed, rc)
        return {SIGNALS.name: t, SUMMARY.name: metric_summary(t), BOUNDARY.name: b,
                SAMPLING.name: sampling_irregularity(t), DISCRIM.name: disc, STUDY_RESULTS.name: rc.frame()}

    def fingerprint(self) -> str:
        names = [COMBINE_FRAMES.name, COMBINE_TRAJ.name]
        up = upstream_state(self.src, names)
        a1_run = self.store._manifest()["runs"].get("A", {}).get("fingerprint")
        return self.store.fingerprint(STUDY, {**self.scfg.section("signal"), "kinematics": self.kcfg.params(),
                                              "skip_nfl_sample": self.skip_nfl_sample}, {**up, "study_A": a1_run})

    def run(self, force: bool = False, figures: bool = True) -> dict[str, pl.DataFrame]:
        fp = self.fingerprint()
        current, reason = self.store.is_current(STUDY, fp, CONTRACTS)
        if current and not force:
            log.info("[studies:A3] reuse (%s)", reason)
            out = self.load()
        else:
            log.info("[studies:A3] compute (%s%s)", "forced; " if force else "", reason)
            res = self.compute()
            out = {c.name: self.store.write(res[c.name], c, STUDY) for c in CONTRACTS}
            self.store.record_run(STUDY, fp, CONTRACTS, {"skip_nfl_sample": self.skip_nfl_sample})
        if figures:
            self.figures(out)
        return out

    def load(self) -> dict[str, pl.DataFrame]:
        return {c.name: self.store.read(c) for c in CONTRACTS}

    def figures(self, out: dict[str, pl.DataFrame]) -> None:
        for rec, fn in FIGURES:
            fn(self.store, rec, out)


# ============================================================================== figures
def _colors():
    from ...data.reporting import CATEGORICAL, INK

    return CATEGORICAL, INK


def fig_boundary(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _colors()
    b = out[BOUNDARY.name]
    domains = [d for d in ("combine", "nfl_sample") if d in set(b["domain"].to_list())] or ["combine"]
    fig, ax = figure(1, len(domains), sharex=True)
    for k, dom in enumerate(domains):
        a = ax[0, k]
        d = b.filter(pl.col("domain") == dom).sort("stratum", "peak")
        if not d.height:
            empty_panel(a, f"no {dom} trajectories")
            continue
        y = np.arange(d.height)
        est = d["share_in_boundary"].to_numpy().astype(float)
        lo = d["boot_lower"].fill_null(np.nan).to_numpy().astype(float)
        hi = d["boot_upper"].fill_null(np.nan).to_numpy().astype(float)
        err = np.vstack([np.where(np.isfinite(lo), est - lo, 0), np.where(np.isfinite(hi), hi - est, 0)]).clip(0)
        a.errorbar(est, y, xerr=err, fmt="o", color=cat[0], ms=6, capsize=3, label="observed share (95 % CI)")
        a.plot(d["expected_share"].to_numpy(), y, "|", color=ink["muted"], ms=14, mew=2,
               label="expected share (boundary_frac)")
        a.set_yticks(y, [f"{s} · {p.replace('peak_', '')} (n={n:,})" for s, p, n in
                         zip(d["stratum"].to_list(), d["peak"].to_list(), d["n_trajectories"].to_list())], fontsize=7)
        a.invert_yaxis()
        a.set_xlim(0, 1)
        a.set_xlabel("share of trajectories whose peak lies on a filter-boundary frame")
        a.set_title(dom)
        a.legend(frameon=False, fontsize=8, loc="lower right")
    finish(fig, "Observed share vs the share expected if peaks fell on random valid frames. Observed ≫ expected means "
                "peak values depend on off-centre SG fits at segment edges — or true peaks occur at recording edges "
                "(forty ends at peak speed). Athlete-cluster bootstrap CIs. PENDING interpretation.",
           "Do peak features sit on filter-boundary frames?")
    store.save_figure(fig, rec, {"boundary": b}, n=f"{int(b['n_trajectories'].sum()) if b.height else 0} trajectory-peaks")


def fig_discrimination(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _colors()
    d = out[DISCRIM.name].filter(pl.col("auc").is_not_null())
    domains = [x for x in ("combine", "nfl_sample") if x in set(d["domain"].to_list())] or ["combine"]
    fig, ax = figure(1, len(domains), sharex=True)
    targets = list(TARGETS)
    for k, dom in enumerate(domains):
        a = ax[0, k]
        g = d.filter(pl.col("domain") == dom)
        if not g.height:
            empty_panel(a, f"no estimable AUC ({dom})")
            continue
        preds = [p for p in PREDICTORS if p in set(g["predictor"].to_list())]
        for j, tgt in enumerate(targets):
            r = g.filter(pl.col("target") == tgt)
            if not r.height:
                continue
            ypos = np.array([preds.index(p) for p in r["predictor"].to_list()]) + (j - 1.5) * 0.18
            est = r["auc"].to_numpy().astype(float)
            lo = r["ci_lower"].fill_null(np.nan).to_numpy().astype(float)
            hi = r["ci_upper"].fill_null(np.nan).to_numpy().astype(float)
            err = np.vstack([np.where(np.isfinite(lo), est - lo, 0), np.where(np.isfinite(hi), hi - est, 0)]).clip(0)
            shared = r["shares_source"].to_numpy()
            a.errorbar(est[~shared], ypos[~shared], xerr=err[:, ~shared], fmt="o", color=cat[j], ms=6, capsize=2,
                       label=tgt)
            if shared.any():
                a.errorbar(est[shared], ypos[shared], xerr=err[:, shared], fmt="o", mfc="none", color=cat[j], ms=6,
                           capsize=2)
        a.axvline(0.5, color=ink["muted"], lw=1, ls="--")
        a.set_yticks(np.arange(len(preds)), preds, fontsize=8)
        a.invert_yaxis()
        a.set_xlim(0, 1)
        a.set_xlabel("AUC (0.5 = no discrimination)")
        a.set_title(dom)
        a.legend(frameon=False, fontsize=8, title="target", title_fontsize=8, loc="lower right")
    finish(fig, "AUC of each quality predictor for each 'unreliable signal' target (top 5 % within domain × stratum, "
                "proposed cut; or peak on a boundary frame). Hollow markers: predictor and target share a source "
                "(mechanical association). Athlete-cluster bootstrap 95 % CI. PENDING interpretation.",
           "Do quality tiers and metrics discriminate unreliable derivative signals?")
    store.save_figure(fig, rec, {"discrimination": out[DISCRIM.name]}, n=f"{d.height} AUC estimates")


def fig_roughness(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _colors()
    t = out[SIGNALS.name]
    metrics = (("jerk_rms_interior", "interior jerk RMS (yd/s³)"), ("vendor_rmse", "vendor − derived speed RMSE (yd/s)"))
    fig, ax = figure(1, 2)
    plot = (t.unpivot(index=["domain", "stratum", "cohort", "nfl_id"], on=[m for m, _ in metrics],
                      variable_name="metric", value_name="v").filter(pl.col("v").is_not_null())
            .group_by("domain", "stratum", "cohort", "metric").agg(
                pl.len().alias("n_trajectories"), pl.col("nfl_id").n_unique().alias("n_athletes"),
                pl.col("v").quantile(0.25, "linear").alias("p25"), pl.col("v").median().alias("median"),
                pl.col("v").quantile(0.75, "linear").alias("p75"))
            .with_columns(pl.concat_str([pl.col("domain"), pl.col("stratum")], separator=" · ").alias("group"))
            .sort("group", "cohort"))
    cohorts = sorted(c for c in plot["cohort"].unique().to_list() if c is not None)
    for k, (m, label) in enumerate(metrics):
        a = ax[0, k]
        d = plot.filter(pl.col("metric") == m)
        if not d.height:
            empty_panel(a, f"no {m}")
            continue
        groups = d["group"].unique(maintain_order=True).to_list()
        for j, c in enumerate(cohorts):
            r = d.filter(pl.col("cohort") == c)
            y = np.array([groups.index(g) for g in r["group"].to_list()]) + (j - (len(cohorts) - 1) / 2) * 0.22
            a.hlines(y, r["p25"].to_numpy(), r["p75"].to_numpy(), color=cat[j % len(cat)], lw=3)
            a.plot(r["median"].to_numpy(), y, "o", color=cat[j % len(cat)], ms=6, label=str(c))
        a.set_yticks(np.arange(len(groups)), groups, fontsize=7)
        a.invert_yaxis()
        a.set_xlabel(label)
        a.legend(frameon=False, fontsize=8, title="cohort", title_fontsize=8)
    finish(fig, "Median and IQR per domain · stratum, coloured by cohort (Combine draft year / NFL season). M1 "
                "evidence: 2025 Combine data carry ≈ ¼ of the positional noise of 2023/24 — any cohort shift in "
                "derivative features must be checked against these signal metrics. PENDING interpretation.",
           "Derivative-signal roughness and vendor disagreement by cohort")
    store.save_figure(fig, rec, {"quantiles": plot}, n=f"{t.height:,} trajectories")


def _rec(fid: str, name: str, question: str, insight: str, statistic: str, dataset: str, method: str) -> FigureRecord:
    return FigureRecord(figure_id=fid, name=name, study="A3", research_question=question, expected_insight=insight,
                        statistic=statistic, dataset=dataset, grain="trajectory (athlete-clustered)",
                        filters="none (all tiers)", method=method)


FIGURES = [
    (_rec("A3-01", "A3_01_boundary_peak_share", "Do peak features land on SG filter-boundary frames more often than "
          "chance?", "identify peak features that depend on edge fits", "share with cluster-bootstrap CI vs expected",
          BOUNDARY.name, "trajectory peak location vs boundary_frac"), fig_boundary),
    (_rec("A3-02", "A3_02_tier_discrimination", "Do quality tiers / metrics discriminate unreliable derivative signals?",
          "whether the tiers are informative at all, and which continuous metric is", "AUC with cluster-bootstrap CI",
          DISCRIM.name, "(Cliff's δ + 1) / 2"), fig_discrimination),
    (_rec("A3-03", "A3_03_roughness_by_cohort", "How rough are derived signals and how far from vendor speed, by "
          "drill and cohort?", "cohort-specific noise that could masquerade as athlete differences",
          "median + IQR", SIGNALS.name, "per-trajectory jerk RMS and vendor RMSE"), fig_roughness),
]
