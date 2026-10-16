"""Phase 03 · M1 — diagnostics for the kinematic foundation (CSV tables + figures).

Reads the PERSISTED M1 datasets (``kinematic_build``) plus two bounded recomputations:
the Combine raw frames (~0.46M) for smoothing sensitivity, and a deterministic NFL game sample
(``--nfl-games`` per season, evenly spaced over the sorted game_id list) for vendor agreement
and threshold / smoothing sensitivity. Nothing here rescans all NFL frames.

Outputs
    reports/tables/phase_03/
        feature_dictionary.csv              registry (definition, formula, unit, limitations, status)
        kinematics_vendor_agreement.csv     derived vs vendor speed / accel / direction (2 conventions)
        smoothing_sensitivity.csv           per feature × SG setting: rank stability, relative change
        event_threshold_sensitivity.csv     events per trajectory by event type × threshold set
        segment_quality_summary.csv         quality tiers by domain × cohort (× drill for Combine)
        tier_reason_counts.csv              how often each tier rule fires
        kinematic_example_selection.csv     deterministic example choice + criterion
    reports/figures/phase_03/  (PNG 300 DPI + SVG)
        m1_signal_example_<drill>, m1_trajectory_events_three_cone, m1_vendor_vs_derived_speed,
        m1_smoothing_sensitivity, m1_event_threshold_sensitivity, m1_quality_tiers

Command for the USER (after kinematic_build)::

    python -m nflbdb.research.kinematic_diagnostics                # add --skip-nfl to omit the NFL sample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ..data.config import Phase01Config
from ..data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS, add_caption, new_figure
from ..data.utils import get_logger
from ..features.config import KinematicsConfig
from ..features.events import detect_events
from ..features.feature_registry import registry_frame
from ..features.trajectory import COMBINE, NFL, TrajectoryDomain
from .artifacts import ArtifactStore
from .kinematic_build import (COMBINE_EVENTS, COMBINE_FRAMES, COMBINE_TRAJ, NFL_TRAJ, PHASE, KinematicFeatureBuilder,
                              run_pipeline)

log = get_logger(__name__)

EXAMPLE_DRILLS = ("FORTY_YARD_DASH", "THREE_CONE_DRILL", "SHORT_SHUTTLE")
TIER_ORDER = ["A", "B", "C", "reject"]
TIER_COLORS = {"A": SEQUENTIAL_BLUE[5], "B": SEQUENTIAL_BLUE[3], "C": SEQUENTIAL_BLUE[1], "reject": STATUS["critical"]}
RULE_ORDER = ["lenient", "default", "strict"]
EVENT_PEAK = {"braking": "peak_abs_a_tan", "acceleration": "peak_abs_a_tan", "turn": "peak_abs_yaw_rate",
              "reversal": "heading_change_deg"}


# ======================================================================================
# Pure table functions (unit-testable on synthetic frames)
# ======================================================================================
def vendor_agreement(kin: pl.DataFrame, domain: str, group: str | None = None) -> pl.DataFrame:
    """Derived vs vendor columns. Direction: circular |vendor dir − heading| under the math
    convention (0° = +x, CCW) and under a compass convention (0° = +y, clockwise); the convention
    with the smaller median difference is the one the vendor uses (evidence, not assumption)."""
    df = kin.filter(pl.col("valid_kin"))
    g = [group] if group else []
    compass = (450.0 - pl.col("heading")) % 360.0
    circ = lambda a, b: ((((a - b) + 540.0) % 360.0) - 180.0).abs()  # noqa: E731 — inputs in [0, 360)
    hv = df.filter(pl.col("heading_valid") & pl.col("dir").is_not_null())
    speed = df.group_by(g or pl.lit("all").alias("_all")).agg(
        pl.len().alias("n_frames"),
        pl.corr("speed", "s").alias("corr_speed_vs_s"),
        (pl.col("speed") - pl.col("s")).abs().median().alias("median_abs_diff_speed"),
        (pl.col("speed") - pl.col("s")).abs().quantile(0.95).alias("p95_abs_diff_speed"),
        (pl.col("speed") - pl.col("s")).median().alias("median_signed_diff_speed"),
        pl.corr("accel_mag", "a").alias("corr_accel_mag_vs_a"),
        pl.corr(pl.col("a_tan").abs(), pl.col("a")).alias("corr_abs_a_tan_vs_a"),
        (pl.col("a") < 0).mean().alias("frac_vendor_a_negative"),
    )
    direction = hv.with_columns(circ(pl.col("dir"), pl.col("heading")).alias("_d_math"),
                                circ(pl.col("dir"), compass).alias("_d_compass")).group_by(
        g or pl.lit("all").alias("_all")).agg(
        pl.len().alias("n_frames_heading_valid"),
        pl.col("_d_math").median().alias("median_dir_diff_math_deg"),
        pl.col("_d_compass").median().alias("median_dir_diff_compass_deg"))
    on = g or ["_all"]
    out = speed.join(direction, on=on, how="left").with_columns(
        pl.when(pl.col("median_dir_diff_compass_deg") < pl.col("median_dir_diff_math_deg"))
        .then(pl.lit("compass (0°=+y, clockwise)")).otherwise(pl.lit("math (0°=+x, counter-clockwise)"))
        .alias("vendor_dir_convention_best_fit"),
        pl.lit(domain).alias("domain"))
    if not g:
        out = out.rename({"_all": "group"})
    else:
        out = out.rename({group: "group"}).with_columns(pl.col("group").cast(pl.String))
    return out.select("domain", "group", pl.exclude("domain", "group")).sort("domain", "group")


def trajectory_peaks(kin: pl.DataFrame, domain: TrajectoryDomain, kcfg: KinematicsConfig) -> pl.DataFrame:
    """Per-trajectory summary features used to judge smoothing sensitivity (long format)."""
    keys = list(domain.keys)
    wide = kin.group_by(keys).agg(
        pl.col("speed").max().alias("peak_speed"),
        pl.col("speed").quantile(0.95).alias("p95_speed"),
        pl.col("a_tan").max().alias("peak_a_tan"),
        (-pl.col("a_tan")).max().alias("peak_decel"),
        pl.col("accel_mag").max().alias("peak_accel_mag"),
        pl.col("yaw_rate").abs().max().alias("peak_abs_yaw_rate"),
        pl.col("step_smooth").sum().alias("path_smooth_yds"))
    brk = detect_events(kin, domain, kcfg, event_types=("braking",), rules=(kcfg.primary_rule,))
    nb = brk.group_by(keys).agg(pl.len().cast(pl.Float64).alias("n_braking_events"))
    wide = wide.join(nb, on=keys, how="left").with_columns(pl.col("n_braking_events").fill_null(0.0))
    return wide.unpivot(index=keys, variable_name="feature", value_name="value")


def smoothing_sensitivity(raw: pl.DataFrame, domain: TrajectoryDomain, kcfg: KinematicsConfig) -> pl.DataFrame:
    """Compare each SG setting with the primary one: Spearman rank correlation across trajectories
    (does the athlete ordering survive?) and median relative absolute change."""
    from ..features.kinematics import compute_kinematics
    from ..features.trajectory import prepare_frames

    keys = list(domain.keys)
    rows = []
    for spec in kcfg.sensitivity_smoothing:
        k = kcfg.with_smoothing(spec)
        frames, _ = prepare_frames(raw, domain, k)   # segmentation does not depend on the SG setting
        rows.append(trajectory_peaks(compute_kinematics(frames, k), domain, k).with_columns(pl.lit(spec.label).alias("smoothing")))
    long = pl.concat(rows)
    base = long.filter(pl.col("smoothing") == kcfg.smoothing.label).select(*keys, "feature", pl.col("value").alias("v0"))
    cmp = long.join(base, on=[*keys, "feature"], how="inner").filter(pl.col("value").is_not_null() & pl.col("v0").is_not_null())
    return (cmp.group_by("smoothing", "feature").agg(
                pl.len().alias("n_trajectories"),
                pl.corr("value", "v0", method="spearman").alias("spearman_vs_primary"),
                ((pl.col("value") - pl.col("v0")).abs() / pl.col("v0").abs())
                .filter(pl.col("v0").abs() > 1e-6).median().alias("median_rel_abs_change"),
                pl.col("value").median().alias("median_value"))
            .with_columns(pl.lit(domain.name).alias("domain"),
                          (pl.col("smoothing") == kcfg.smoothing.label).alias("is_primary"))
            .select("domain", "smoothing", "is_primary", "feature", "n_trajectories", "spearman_vs_primary",
                    "median_rel_abs_change", "median_value")
            .sort("domain", "feature", "smoothing"))


def event_threshold_sensitivity(events: pl.DataFrame, traj: pl.DataFrame, domain: str,
                                group: str | None = None) -> pl.DataFrame:
    """Events per usable trajectory (tier ≠ reject) by event type × rule (× group)."""
    g = [group] if group else []
    usable = traj.filter(pl.col("quality_tier") != "reject")
    denom = (usable.group_by(g).agg(pl.len().alias("n_trajectories")) if g
             else pl.DataFrame({"n_trajectories": [usable.height]}))
    peak = pl.coalesce([pl.when(pl.col("event_type") == et).then(pl.col(c).abs()) for et, c in EVENT_PEAK.items()])
    ev = events.with_columns(peak.alias("_peak")).group_by([*g, "event_type", "rule"]).agg(
        pl.len().alias("n_events"), pl.col("duration_s").median().alias("median_duration_s"),
        pl.col("_peak").median().alias("median_peak"),
        pl.col("touches_segment_edge").mean().alias("frac_touching_edge"))
    out = ev.join(denom, on=g, how="left") if g else ev.join(denom, how="cross")
    out = out.with_columns((pl.col("n_events") / pl.col("n_trajectories")).alias("events_per_trajectory"),
                           pl.lit(domain).alias("domain"),
                           pl.lit("|a_tan| yd/s² (braking/accel), |yaw| deg/s (turn), |Δheading| deg (reversal)")
                           .alias("median_peak_unit"))
    if g:
        out = out.rename({group: "group"}).with_columns(pl.col("group").cast(pl.String))
    else:
        out = out.with_columns(pl.lit("all").alias("group"))
    return out.select("domain", "group", "event_type", "rule", "n_events", "n_trajectories", "events_per_trajectory",
                      "median_duration_s", "median_peak", "median_peak_unit", "frac_touching_edge") \
        .sort("domain", "group", "event_type", "rule")


def quality_summary(traj: pl.DataFrame, domain: TrajectoryDomain, extra_group: str | None = None) -> pl.DataFrame:
    g = [domain.partition, *([extra_group] if extra_group else [])]
    return (traj.group_by([*g, "quality_tier"]).agg(pl.len().alias("n_trajectories"),
                                                    pl.col("nfl_id").n_unique().alias("n_athletes"),
                                                    pl.col("coverage").median().alias("median_coverage"),
                                                    pl.col("resid_rms_yds").median().alias("median_resid_rms_yds"))
            .with_columns((pl.col("n_trajectories") / pl.col("n_trajectories").sum().over(g)).alias("share"),
                          pl.lit(domain.name).alias("domain"),
                          pl.col(domain.partition).cast(pl.Int64).alias("cohort"))
            .drop(domain.partition)
            .select("domain", "cohort", *([extra_group] if extra_group else []), "quality_tier", "n_trajectories",
                    "share", "n_athletes", "median_coverage", "median_resid_rms_yds")
            .sort(["domain", "cohort", *([extra_group] if extra_group else []), "quality_tier"]))


def tier_reason_counts(traj: pl.DataFrame, domain: str) -> pl.DataFrame:
    return (traj.select(pl.col("tier_reasons").str.split(";").alias("reason")).explode("reason")
            .filter(pl.col("reason").is_not_null() & (pl.col("reason") != ""))
            .group_by("reason").agg(pl.len().alias("n_trajectories"))
            .with_columns(pl.lit(domain).alias("domain"), (pl.col("n_trajectories") / traj.height).alias("share"))
            .select("domain", "reason", "n_trajectories", "share").sort("domain", "n_trajectories", descending=[False, True]))


def select_examples(traj: pl.DataFrame, drills=EXAMPLE_DRILLS) -> pl.DataFrame:
    """Deterministic, outcome-blind example per drill: the tier-A attempt whose peak derived speed
    is closest to that drill's tier-A median (ties → lowest nfl_id, event_id, attempt)."""
    a = traj.filter((pl.col("quality_tier") == "A") & pl.col("drill").is_in(list(drills)))
    return (a.with_columns((pl.col("max_speed") - pl.col("max_speed").median().over("drill")).abs().alias("_dist"),
                           pl.col("max_speed").median().over("drill").alias("drill_median_max_speed"),
                           pl.len().over("drill").alias("n_tier_a"))
            .sort("drill", "_dist", "nfl_id", "event_id", "attempt")
            .group_by("drill", maintain_order=True).first()
            .with_columns(pl.lit("tier A; |max_speed − drill median| minimal; ties by nfl_id, event_id, attempt")
                          .alias("criterion"))
            .select("drill", "event_id", "nfl_id", "attempt", "draft_year", "max_speed", "drill_median_max_speed",
                    "n_tier_a", "criterion"))


# ======================================================================================
class KinematicDiagnostics:
    def __init__(self, cfg: Phase01Config, kcfg: KinematicsConfig | None = None, store: ArtifactStore | None = None,
                 nfl_games_per_season: int = 8):
        self.cfg = cfg
        self.kcfg = kcfg or KinematicsConfig.from_tracking(cfg.tracking)
        self.store = store or ArtifactStore(cfg, phase=PHASE)
        self.builder = KinematicFeatureBuilder(cfg, self.kcfg, self.store)
        self.nfl_games_per_season = nfl_games_per_season

    # ------------------------------------------------------------------ inputs
    def _load(self, spec) -> pl.DataFrame:
        current, reason = self.store.status(spec, self.builder.params("combine" if spec.name.startswith("combine") else "nfl"))
        if not self.store.dataset_files(spec.name):
            raise FileNotFoundError(f"{spec.name} not built — run `python -m nflbdb.research.kinematic_build` first")
        if not current:
            log.warning("[diagnostics] %s is stale (%s) — results describe the persisted version", spec.name, reason)
        return self.store.load(spec.name).collect()

    def nfl_sample(self) -> dict[str, pl.DataFrame]:
        """Deterministic NFL sample: evenly spaced games per season, all threshold sets."""
        raws = []
        for season, file in self.builder.nfl_season_files().items():
            games = pl.scan_parquet(file).select("game_id").unique().sort("game_id").collect()["game_id"].to_list()
            step = max(1, len(games) // max(1, self.nfl_games_per_season))
            for g in games[::step][: self.nfl_games_per_season]:
                raws.append(self.builder.load_nfl_raw(file, season, g, g))
        raw = pl.concat(raws, how="vertical_relaxed")
        log.info("[diagnostics] NFL sample: %s frames from %d games", f"{raw.height:,}", raw["game_id"].n_unique())
        out = run_pipeline(raw, NFL, self.kcfg)
        out["raw"] = raw
        return out

    # ------------------------------------------------------------------ figures
    def fig_signal_example(self, frames: pl.DataFrame, events: pl.DataFrame, ex: dict) -> Path:
        key = (pl.col("event_id") == ex["event_id"]) & (pl.col("nfl_id") == ex["nfl_id"]) & (pl.col("attempt") == ex["attempt"])
        f = frames.filter(key).sort("ts")
        ev = events.filter(key & (pl.col("rule") == self.kcfg.primary_rule))
        fig, axes = new_figure(3, 1, sharex=True)
        t = f["t_s"].to_numpy()
        axes[0].plot(t, f["s"].to_numpy(), color=INK["muted"], lw=1.2, label="vendor s")
        axes[0].plot(t, f["speed"].to_numpy(), color=CATEGORICAL[0], label=f"derived speed ({self.kcfg.smoothing.label})")
        axes[0].set_ylabel("speed (yd/s)")
        axes[1].axhline(0, color=INK["axis"], lw=1)
        axes[1].plot(t, f["accel_mag"].to_numpy(), color=INK["muted"], lw=1.2, label="|a| vector magnitude")
        axes[1].plot(t, f["a_tan"].to_numpy(), color=CATEGORICAL[0], label="a_tan = d|v|/dt (signed)")
        axes[1].set_ylabel("acceleration (yd/s²)")
        axes[2].axhline(0, color=INK["axis"], lw=1)
        axes[2].plot(t, f["yaw_rate"].to_numpy(), color=CATEGORICAL[0], label="yaw rate (null below "
                     f"{self.kcfg.min_heading_speed} yd/s)")
        axes[2].set_ylabel("yaw rate (deg/s)")
        axes[2].set_xlabel("time since attempt start (s)")
        shade = {"braking": (CATEGORICAL[1], 0), "acceleration": (CATEGORICAL[2], 0), "turn": (CATEGORICAL[1], 2),
                 "reversal": (CATEGORICAL[2], 2)}
        seen = set()
        for r in ev.iter_rows(named=True):
            color, ax_i = shade[r["event_type"]]
            lbl = None if r["event_type"] in seen else f"{r['event_type']} event ({self.kcfg.primary_rule} rule)"
            seen.add(r["event_type"])
            axes[ax_i].axvspan(r["t_start_s"], r["t_end_s"] + self.kcfg.dt_s, color=color, alpha=0.15, lw=0, label=lbl)
        for ax in axes:
            ax.legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"{ex['drill']}: smoothed kinematics and detected events (deterministic example)")
        add_caption(fig, f"Source: p03 combine_frame_kinematics · nfl_id {ex['nfl_id']}, attempt {ex['attempt']} · "
                         f"selection: {ex['criterion']} (n tier-A = {ex['n_tier_a']})")
        return self.store.save_fig(fig, f"m1_signal_example_{ex['drill'].lower()}", source="combine_frame_kinematics",
                                   n=f.height)

    def fig_trajectory(self, frames: pl.DataFrame, events: pl.DataFrame, ex: dict) -> Path:
        from matplotlib.collections import LineCollection
        import numpy as np

        key = (pl.col("event_id") == ex["event_id"]) & (pl.col("nfl_id") == ex["nfl_id"]) & (pl.col("attempt") == ex["attempt"])
        f = frames.filter(key & pl.col("valid_kin")).sort("ts")
        ev = events.filter(key & (pl.col("rule") == self.kcfg.primary_rule))
        fig, ax = new_figure()
        x, y, s = f["x_s"].to_numpy(), f["y_s"].to_numpy(), f["speed"].to_numpy()
        seg = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
        same = (f["seg_idx"].to_numpy()[1:] == f["seg_idx"].to_numpy()[:-1])
        lc = LineCollection(seg[same], cmap="Blues", linewidths=3)
        lc.set_array(s[:-1][same])
        lc.set_clim(0, max(1.0, float(np.nanmax(s))))
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label="derived speed (yd/s)", shrink=0.8)
        markers = {"braking": ("v", CATEGORICAL[1]), "turn": ("o", CATEGORICAL[2]), "reversal": ("D", CATEGORICAL[3])}
        fid = f.select(pl.col("frame_id").cast(pl.Int64), "x_s", "y_s")
        for et, (m, c) in markers.items():
            pts = ev.filter(pl.col("event_type") == et).join(fid, left_on="start_frame_id", right_on="frame_id", how="inner")
            if pts.height:
                ax.scatter(pts["x_s"], pts["y_s"], marker=m, s=90, color=c, edgecolor=INK["surface"], linewidth=2,
                           zorder=3, label=f"{et} onset (n={pts.height})")
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
        ax.set_xlabel("x (yd, Combine frame — orientation unverified)")
        ax.set_ylabel("y (yd)")
        ax.set_title(f"{ex['drill']}: smoothed path coloured by speed, event onsets ({self.kcfg.primary_rule} rules)")
        ax.legend(loc="upper left", fontsize=9)
        add_caption(fig, f"Source: p03 combine_frame_kinematics / combine_kinematic_events · nfl_id {ex['nfl_id']}, "
                         f"attempt {ex['attempt']} · {ex['criterion']}")
        return self.store.save_fig(fig, "m1_trajectory_events_three_cone", source="combine_frame_kinematics", n=f.height)

    def fig_vendor_speed(self, kins: dict[str, pl.DataFrame]) -> Path:
        fig, axes = new_figure(1, len(kins))
        axes = axes if len(kins) > 1 else [axes]
        for ax, (dom, k) in zip(axes, kins.items()):
            d = k.filter(pl.col("valid_kin") & pl.col("s").is_not_null()).select("s", "speed")
            hb = ax.hexbin(d["s"], d["speed"], gridsize=60, cmap="Blues", mincnt=1, bins="log")
            lim = float(max(d["s"].max() or 1, d["speed"].max() or 1))
            ax.plot([0, lim], [0, lim], color=INK["secondary"], lw=1, ls="--", label="y = x")
            ax.set_xlabel("vendor s (yd/s)")
            ax.set_ylabel("derived speed (yd/s)")
            ax.set_title(f"{dom}: derived vs vendor speed (n = {d.height:,} frames)")
            ax.legend(loc="upper left")
            fig.colorbar(hb, ax=ax, label="frames (log)")
        add_caption(fig, "Source: combine = all persisted frames; nfl = deterministic game sample (see kinematics_vendor_agreement.csv)")
        return self.store.save_fig(fig, "m1_vendor_vs_derived_speed", source="phase_03 kinematics")

    def fig_smoothing(self, sens: pl.DataFrame) -> Path:
        doms = sens["domain"].unique().sort().to_list()
        fig, axes = new_figure(1, len(doms), sharey=True)
        axes = axes if len(doms) > 1 else [axes]
        others = [s for s in sens.filter(~pl.col("is_primary"))["smoothing"].unique().sort().to_list()][:3]
        feats = sens["feature"].unique().sort().to_list()
        for ax, dom in zip(axes, doms):
            d = sens.filter((pl.col("domain") == dom) & ~pl.col("is_primary"))
            for i, sm in enumerate(others):
                r = d.filter(pl.col("smoothing") == sm)
                yy = [feats.index(f) + (i - 1) * 0.18 for f in r["feature"]]
                ax.scatter(r["spearman_vs_primary"], yy, s=64, color=CATEGORICAL[i], label=sm, zorder=3)
            ax.set_yticks(range(len(feats)), feats)
            ax.axvline(0.9, color=INK["muted"], lw=1, ls=":")
            ax.set_xlabel("Spearman rank correlation with primary setting")
            n = int(d["n_trajectories"].max() or 0)
            ax.set_title(f"{dom} (n = {n:,} trajectories)")
            ax.legend(loc="lower left", fontsize=9, title="SG setting")
        add_caption(fig, "Dotted line: 0.9 reference (not a validated cut-off). Source: smoothing_sensitivity.csv")
        return self.store.save_fig(fig, "m1_smoothing_sensitivity", source="smoothing_sensitivity.csv")

    def fig_thresholds(self, sens: pl.DataFrame) -> Path:
        import numpy as np

        agg = sens.group_by("domain", "event_type", "rule").agg(
            (pl.col("n_events").sum() / pl.col("n_trajectories").sum()).alias("ept"))
        doms = agg["domain"].unique().sort().to_list()
        types = ["braking", "acceleration", "turn", "reversal"]
        fig, axes = new_figure(1, len(doms), sharey=False)
        axes = axes if len(doms) > 1 else [axes]
        for ax, dom in zip(axes, doms):
            d = agg.filter(pl.col("domain") == dom)
            xs = np.arange(len(types))
            for i, rule in enumerate(RULE_ORDER):
                vals = [d.filter((pl.col("event_type") == t) & (pl.col("rule") == rule))["ept"].sum() for t in types]
                ax.bar(xs + (i - 1) * 0.26, vals, width=0.24, color=CATEGORICAL[i], label=rule,
                       edgecolor=INK["surface"], linewidth=2)
            ax.set_xticks(xs, types)
            ax.set_ylabel("events per usable trajectory")
            ax.set_title(f"{dom}: event counts by threshold set")
            ax.legend(title="rule", fontsize=9)
            ax.grid(axis="x", visible=False)
        add_caption(fig, "Usable = quality tier ≠ reject. NFL = deterministic sample. Source: event_threshold_sensitivity.csv")
        return self.store.save_fig(fig, "m1_event_threshold_sensitivity", source="event_threshold_sensitivity.csv")

    def fig_tiers(self, summary: pl.DataFrame) -> Path:
        import numpy as np

        d = summary.group_by("domain", "cohort", "quality_tier").agg(pl.col("n_trajectories").sum()).with_columns(
            (pl.col("n_trajectories") / pl.col("n_trajectories").sum().over("domain", "cohort")).alias("share"),
            pl.format("{} {}", pl.col("domain"), pl.col("cohort")).alias("label"))
        labels = d["label"].unique().sort().to_list()
        fig, ax = new_figure()
        left = np.zeros(len(labels))
        for tier in TIER_ORDER:
            vals = np.array([d.filter((pl.col("label") == lb) & (pl.col("quality_tier") == tier))["share"].sum()
                             for lb in labels])
            ax.barh(labels, vals, left=left, color=TIER_COLORS[tier], label=f"tier {tier}", height=0.6,
                    edgecolor=INK["surface"], linewidth=2)
            left += vals
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of trajectories")
        ax.set_title("Kinematic quality tiers by domain and cohort (PROVISIONAL rules)")
        ax.legend(loc="lower right", ncols=4, fontsize=9)
        ax.grid(axis="y", visible=False)
        add_caption(fig, f"Source: p03 *_trajectory_quality · n = {int(d['n_trajectories'].sum()):,} trajectories · "
                         "rules in features/config.py TierRules")
        return self.store.save_fig(fig, "m1_quality_tiers", source="trajectory_quality", n=int(d["n_trajectories"].sum()))

    # ------------------------------------------------------------------ orchestration
    def run(self, skip_nfl: bool = False) -> dict[str, pl.DataFrame]:
        out: dict[str, pl.DataFrame] = {}
        exp = self.store.export_csv
        out["feature_dictionary"] = registry_frame()
        exp(out["feature_dictionary"], "feature_dictionary", "M1 feature registry")

        frames = self._load(COMBINE_FRAMES)
        ctraj = self._load(COMBINE_TRAJ)
        cev = self._load(COMBINE_EVENTS)
        agree = [vendor_agreement(frames, "combine"), vendor_agreement(frames, "combine", "drill_type")]
        thr = [event_threshold_sensitivity(cev, ctraj, "combine"),
               event_threshold_sensitivity(cev, ctraj, "combine", "drill_type")]
        qual = [quality_summary(ctraj, COMBINE, "drill_type")]
        reasons = [tier_reason_counts(ctraj, "combine")]
        smooth = [smoothing_sensitivity(self.builder.load_combine_raw(), COMBINE, self.kcfg)]
        kins = {"combine": frames}

        if not skip_nfl:
            s = self.nfl_sample()
            agree.append(vendor_agreement(s["frames"], "nfl_sample"))
            thr.append(event_threshold_sensitivity(s["events"], s["trajectories"], "nfl_sample"))
            smooth.append(smoothing_sensitivity(s["raw"], NFL, self.kcfg))
            kins["nfl_sample"] = s["frames"]
            if self.store.dataset_files(NFL_TRAJ.name):
                ntraj = self._load(NFL_TRAJ)
                qual.append(quality_summary(ntraj, NFL))
                reasons.append(tier_reason_counts(ntraj, "nfl"))
            else:
                qual.append(quality_summary(s["trajectories"], NFL).with_columns(pl.lit("nfl_sample").alias("domain")))
                reasons.append(tier_reason_counts(s["trajectories"], "nfl_sample"))

        out["kinematics_vendor_agreement"] = pl.concat(agree, how="diagonal_relaxed")
        out["event_threshold_sensitivity"] = pl.concat(thr, how="diagonal_relaxed")
        out["segment_quality_summary"] = pl.concat(qual, how="diagonal_relaxed")
        out["tier_reason_counts"] = pl.concat(reasons, how="diagonal_relaxed")
        out["smoothing_sensitivity"] = pl.concat(smooth, how="diagonal_relaxed")
        out["kinematic_example_selection"] = select_examples(ctraj)
        for name, df in out.items():
            if name != "feature_dictionary":
                exp(df, name, f"Phase 03 M1 diagnostic: {name}")

        self.fig_vendor_speed(kins)
        self.fig_smoothing(out["smoothing_sensitivity"])
        self.fig_thresholds(out["event_threshold_sensitivity"].filter(pl.col("group") == "all"))
        self.fig_tiers(out["segment_quality_summary"])
        for ex in out["kinematic_example_selection"].iter_rows(named=True):
            self.fig_signal_example(frames, cev, ex)
            if ex["drill"] == "THREE_CONE_DRILL":
                self.fig_trajectory(frames, cev, ex)
        return out


def main(argv: list[str] | None = None) -> int:
    from ..data.config import configure_runtime

    ap = argparse.ArgumentParser(description="Phase 03 · M1 kinematic diagnostics (tables + figures)")
    ap.add_argument("--skip-nfl", action="store_true", help="omit the NFL game sample")
    ap.add_argument("--nfl-games", type=int, default=8, help="sampled games per season")
    a = ap.parse_args(argv)
    cfg = Phase01Config.load()
    configure_runtime(cfg)
    out = KinematicDiagnostics(cfg, nfl_games_per_season=a.nfl_games).run(skip_nfl=a.skip_nfl)
    print(out["kinematics_vendor_agreement"])
    print(out["segment_quality_summary"].head(20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
