"""11C — braking, turning and re-acceleration mechanics.

    TRANSFER_PAIRS            declared Combine transition metric ↔ NFL target pairs (+ wrong-drill / wrong-quantity controls)
    variance_decomposition    incremental R² of transition-level metrics: drill → turn geometry → direction → athlete;
                              the remainder is sensor noise + within-athlete variation (descriptive, sequential order declared)
    geometry_cells            entry-speed × |turn angle| cells counted in BOTH domains; only shared cells are compared
    combine_geometry_metrics  athlete metrics restricted to a geometry cell (e.g. Transition Cost Index of ≥ 90° turns)
    asymmetry                 left/right difference only where both directions are observed with comparable entry speed
    segmenter_summary         agreement of the threshold, changepoint and HMM segmenters (frames with the same state)

Transition angles are measured from velocity vectors; nothing assumes a drill's nominal angle.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from . import evaluation as ev
from .config import M11Config
from .speed_state import Pair

NAN = float("nan")

TRANSFER_PAIRS: tuple[Pair, ...] = (
    Pair("short_shuttle.decel_mean", "nfl.brake_decel", "matched", "run-mean deceleration with the same −1 yd/s² threshold"),
    Pair("short_shuttle.decel_peak", "nfl.brake_peak05", "matched", "robust peak deceleration vs peak 0.5-s deceleration"),
    Pair("short_shuttle.braking_distance_yd", "nfl.brake_distance_matched", "matched", "stopping path (NFL entry speed 4–7 yd/s)"),
    Pair("short_shuttle.brake_to_burst_s", "nfl.brake_to_burst", "matched", "braking end → burst (thresholds 1.5 vs 1.0 yd/s², declared)"),
    Pair("short_shuttle.exit_accel", "nfl.reaccel_mean", "matched", "acceleration after a stop"),
    Pair("short_shuttle.transition_cost_index", "nfl.pair_tci", "matched", "speed lost through a brake → re-acceleration"),
    Pair("three_cone.transition_cost_index", "nfl.redirect_retention_90", "matched", "speed lost (Combine) vs speed kept (NFL) through ≥ 90° — expected NEGATIVE"),
    Pair("three_cone.curved_speed", "nfl.redirect_retention", "physics", "speed while turning at speed vs speed kept through a redirection"),
    Pair("three_cone.exit_accel", "nfl.reaccel_mean", "matched", "exit acceleration after a reversal"),
    Pair("long_shuttle.decel_mean", "nfl.brake_decel", "matched", "deceleration from a higher entry speed"),
    Pair("long_shuttle.exit_accel", "nfl.reaccel_mean", "matched", "repeated re-acceleration"),
    Pair("backpedal_transition.exit_accel", "nfl.reaccel_mean", "matched", "forward drive after a transition"),
    Pair("backpedal_transition.brake_to_burst_s", "nfl.brake_to_burst", "matched", "plant → drive latency"),
    Pair("geom.tci_turn90", "nfl.pair_tci_turn90", "matched", "GEOMETRY-MATCHED: Transition Cost Index of ≥ 90° turns in both domains"),
    Pair("short_shuttle.decel_mean", "nfl.band_6_8", "noncomparable", "NEGATIVE CONTROL: braking metric vs high-speed acceleration"),
    Pair("forty.band_4_6_mean_accel", "nfl.brake_decel", "noncomparable", "NEGATIVE CONTROL (wrong drill): forty band vs NFL braking"),
)
STOPWATCH = ("forty", "ten", "short_shuttle", "three_cone")


def transition_metric_names(attempt_long: pl.DataFrame, families: Sequence[str]) -> list[str]:
    return sorted(attempt_long.filter(pl.col("family").is_in(list(families)))["feature"].unique().to_list())


def variance_decomposition(trans: pl.DataFrame, metrics: Sequence[str], angle_bins: Sequence[float], min_rows: int = 30) -> pl.DataFrame:
    """Sequential incremental R² (declared order: drill, |angle| bin, direction, athlete) of transition-level metrics."""
    d = trans.with_columns(pl.col("abs_turn_angle_deg").cut(list(angle_bins), left_closed=True).cast(pl.String).fill_null("unobserved").alias("_ang"),
                           pl.when(pl.col("turn_angle_deg") > 0).then(pl.lit("ccw")).when(pl.col("turn_angle_deg") < 0).then(pl.lit("cw"))
                           .otherwise(pl.lit("unobserved")).alias("_dir"))
    rows = []
    for m in metrics:
        if m not in d.columns:
            continue
        g = d.filter(pl.col(m).is_not_null() & pl.col(m).is_finite())
        if g.height < min_rows:
            rows.append({"metric": m, "n_transitions": g.height, "n_athletes": g["nfl_id"].n_unique() if g.height else 0, "status": "too few transitions"})
            continue
        y = g[m].to_numpy().astype(float)
        sst = float(((y - y.mean()) ** 2).sum())
        blocks: list[np.ndarray] = []
        r2_prev, out = 0.0, {"metric": m, "n_transitions": g.height, "n_athletes": g["nfl_id"].n_unique(), "status": "computed"}
        for name, col in (("drill", "drill"), ("geometry", "_ang"), ("direction", "_dir"), ("athlete", "nfl_id")):
            lv = g[col].cast(pl.String).to_numpy()
            levels = sorted(set(lv.tolist()))[1:]
            if levels:
                blocks.append(np.column_stack([(lv == l).astype(float) for l in levels]))
            X = np.column_stack([np.ones(g.height), *blocks]) if blocks else np.ones((g.height, 1))
            res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            r2 = 1 - float((res ** 2).sum()) / sst if sst > 0 else NAN
            out[f"r2_{name}"] = r2 - r2_prev
            r2_prev = r2
        out["residual_share"] = 1 - r2_prev
        rows.append(out)
    return pl.DataFrame(rows, infer_schema_length=None)


def geometry_cells(combine_trans: pl.DataFrame, nfl_pairs: pl.DataFrame, nfl_episodes: pl.DataFrame, c: M11Config) -> pl.DataFrame:
    """Transitions per (entry-speed bin × |angle| bin) in each domain. NFL: brake → re-acceleration pairs with an observed
    turn, and redirection episodes (turns at speed: entry = v_start)."""
    eb, ab = list(c.entry_speed_bins), list(c.angle_bins)
    cut = lambda e, col, b: e.with_columns(pl.col(col).cut(b, left_closed=True).cast(pl.String).alias(f"_{col}"))
    cmb = cut(cut(combine_trans.filter(pl.col("abs_turn_angle_deg").is_not_null()), "entry_speed", eb), "abs_turn_angle_deg", ab)
    cm = cmb.group_by("_entry_speed", "_abs_turn_angle_deg").agg(pl.len().alias("n_combine"), pl.col("nfl_id").n_unique().alias("athletes_combine"))
    p = cut(cut(nfl_pairs.filter(pl.col("angle_observed")).rename({"turn_angle_abs": "abs_turn_angle_deg"}), "entry_speed", eb), "abs_turn_angle_deg", ab)
    r = nfl_episodes.filter((pl.col("demand_id") == "redirection") & pl.col("heading_change_deg").is_not_null()).select(
        "nfl_id", pl.col("v_start").alias("entry_speed"), pl.col("heading_change_deg").abs().alias("abs_turn_angle_deg"))
    r = cut(cut(r, "entry_speed", eb), "abs_turn_angle_deg", ab)
    nf = pl.concat([p.select("nfl_id", "_entry_speed", "_abs_turn_angle_deg", pl.lit("brake_reaccel").alias("kind")),
                    r.select("nfl_id", "_entry_speed", "_abs_turn_angle_deg", pl.lit("turn_at_speed").alias("kind"))])
    nm = nf.group_by("_entry_speed", "_abs_turn_angle_deg").agg(pl.len().alias("n_nfl"), pl.col("nfl_id").n_unique().alias("athletes_nfl"),
                                                                 (pl.col("kind") == "brake_reaccel").sum().alias("n_nfl_brake_reaccel"))
    out = cm.join(nm, on=["_entry_speed", "_abs_turn_angle_deg"], how="full", coalesce=True).fill_null(0)
    return (out.rename({"_entry_speed": "entry_speed_bin", "_abs_turn_angle_deg": "angle_bin"})
            .with_columns(((pl.col("athletes_combine") >= c.min_athletes_limited) & (pl.col("athletes_nfl") >= c.min_athletes_limited)).alias("shared_cell"))
            .sort("entry_speed_bin", "angle_bin"))


def combine_geometry_metrics(trans: pl.DataFrame, min_turns: int = 1) -> pl.DataFrame:
    """Athlete-level geometry-restricted metrics (long: nfl_id, feature, value)."""
    d = trans.filter(pl.col("abs_turn_angle_deg").is_not_null())
    specs = {"geom.tci_turn90": (pl.col("abs_turn_angle_deg") >= 90, "transition_cost_index"),
             "geom.tci_turn45_90": (pl.col("abs_turn_angle_deg").is_between(45, 90, closed="left"), "transition_cost_index"),
             "geom.decel_entry4_6": (pl.col("entry_speed").is_between(4, 6, closed="left"), "decel_mean"),
             "geom.decel_entry6p": (pl.col("entry_speed") >= 6, "decel_mean"),
             "geom.recovery_turn90": (pl.col("abs_turn_angle_deg") >= 90, "speed_recovery_time_s")}
    parts = []
    for name, (cond, col) in specs.items():
        parts.append(d.filter(cond & pl.col(col).is_finite()).group_by("nfl_id").agg(pl.col(col).median().alias("value"), pl.len().alias("n_turns"))
                     .filter(pl.col("n_turns") >= min_turns).with_columns(pl.lit(name).alias("feature")))
    return pl.concat(parts).select("nfl_id", "feature", "value", "n_turns") if parts else pl.DataFrame()


def asymmetry(trans: pl.DataFrame, metric: str = "transition_cost_index", min_per_side: int = 2, max_entry_gap: float = 1.0) -> pl.DataFrame:
    """Athlete left/right difference of ``metric`` (ccw − cw), only with ≥ ``min_per_side`` turns per side AND median entry
    speeds within ``max_entry_gap`` yd/s (comparable conditions). The sign convention of the frame is unverified, so the
    MAGNITUDE is the reported quantity."""
    d = trans.filter(pl.col("turn_angle_deg").is_not_null() & pl.col(metric).is_finite()).with_columns(
        pl.when(pl.col("turn_angle_deg") > 0).then(pl.lit("ccw")).otherwise(pl.lit("cw")).alias("_side"))
    g = d.group_by("nfl_id", "_side").agg(pl.col(metric).median().alias("m"), pl.col("entry_speed").median().alias("e"), pl.len().alias("n"))
    w = g.pivot(on="_side", index="nfl_id", values=["m", "e", "n"])
    need = {"m_ccw", "m_cw", "e_ccw", "e_cw", "n_ccw", "n_cw"}
    if not need <= set(w.columns):
        return pl.DataFrame(schema={"nfl_id": pl.Int64, "asymmetry_abs": pl.Float64, "n_ccw": pl.Int64, "n_cw": pl.Int64, "comparable": pl.Boolean})
    return w.with_columns(((pl.col("n_ccw") >= min_per_side) & (pl.col("n_cw") >= min_per_side) & ((pl.col("e_ccw") - pl.col("e_cw")).abs() <= max_entry_gap))
                          .alias("comparable"), (pl.col("m_ccw") - pl.col("m_cw")).abs().alias("asymmetry_abs")).select(
        "nfl_id", "asymmetry_abs", pl.col("n_ccw").cast(pl.Int64), pl.col("n_cw").cast(pl.Int64), "comparable")


def segmenter_summary(agree: pl.DataFrame) -> pl.DataFrame:
    if agree.is_empty():
        return pl.DataFrame()
    return agree.group_by("family").agg(pl.len().alias("n_attempts"), pl.col("agree_changepoint").median().alias("median_agree_changepoint"),
                                        pl.col("agree_hmm").median().alias("median_agree_hmm"),
                                        pl.col("agree_changepoint").quantile(0.1, "linear").alias("p10_agree_changepoint"),
                                        pl.col("agree_hmm").quantile(0.1, "linear").alias("p10_agree_hmm")).sort("family")


def redundancy_rows(athlete_wide: pl.DataFrame, officials: pl.DataFrame, metrics: Sequence[str]) -> pl.DataFrame:
    """Within-position R² of each transition metric from ALL official times available (incl. shuttle / three-cone)."""
    d = officials.join(athlete_wide, on="nfl_id", how="inner")
    rows = []
    for m in metrics:
        if m in d.columns:
            fam = m.split(".")[0]
            offs = [k for k in ("forty", "ten", fam if fam in ("short_shuttle", "three_cone") else "") if k and k in d.columns]
            r = ev.stopwatch_redundancy(d, m, offs)
            rows.append({"metric": m, "officials": ", ".join(offs), **r})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["STOPWATCH", "TRANSFER_PAIRS", "asymmetry", "combine_geometry_metrics", "geometry_cells", "redundancy_rows", "segmenter_summary",
           "transition_metric_names", "variance_decomposition"]
