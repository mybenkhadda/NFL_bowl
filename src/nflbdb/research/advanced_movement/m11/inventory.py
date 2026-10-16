"""11.0 — scientific setup and data opportunity inventory.

    attempt_summary          one row per Combine attempt of EVERY drill: frames, duration, sampling, coverage, censoring
    drill_inventory          per drill (and family): athletes, attempts, positions, frames, sampling, coverage, censoring, exclusions
    nfl_inventory            per position × demand: episodes, athletes, games, seasons, band opportunities, label availability
    PHYSICAL_OVERLAP         DECLARED physical overlap of a drill family with an NFL demand (confidence + shared quantity).
                             A shared generic column (e.g. "acceleration") is NOT an overlap: the movement state must match.
    feasibility_matrix       drill family × NFL demand × position → SUPPORTED · LIMITED · UNSUPPORTED · UNKNOWN by declared rule
    measurement_registry     the cross-domain quantities with their definition on each side
    movement_gap_registry_v2 corrected Movement Gap definitions (10.1 classification error fixed; 10.1 outputs kept)
    support_funnel           Combine athletes → tracked drill → linked NFL athlete → athletes with enough episodes
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from .. import movement_gap as mg
from .config import LIMITED, M11Config, SUPPORTED, UNKNOWN, UNSUPPORTED
from .sources import FAMILY_ORDER

NAN = float("nan")
DEMANDS = ("accel_low", "accel_mid", "transition_high", "sustained_high", "speed_maintenance", "braking", "reaccel_after_brake", "redirection",
           "repeated_change")
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}

# (family, demand) → (confidence, shared physical quantity, why). Fixed before any Milestone 11 count or value was read.
PHYSICAL_OVERLAP: dict[tuple[str, str], tuple[str, str, str]] = {
    ("forty", "accel_low"): ("high", "band acceleration 2→4 / 4→6 yd/s from rest", "standing start, maximal effort; the NFL episode starts below 2 yd/s"),
    ("forty", "accel_mid"): ("low", "band acceleration 4→6 / 6→8 yd/s", "same band, but the forty passes it from rest, the episode from a rolling start"),
    ("forty", "transition_high"): ("medium", "band acceleration 6→8 yd/s", "both accelerate through 6→8 yd/s while already moving"),
    ("forty", "sustained_high"): ("low", "top speed", "forty recordings can end before the speed peak; game speed is rarely maximal"),
    ("short_shuttle", "braking"): ("high", "deceleration from ~5–7 yd/s to a near-stop", "the shuttle forces two stops; NFL braking from ≥ 4 yd/s"),
    ("short_shuttle", "reaccel_after_brake"): ("high", "acceleration after a stop", "re-acceleration within 1 s of braking in both domains"),
    ("short_shuttle", "redirection"): ("low", "heading change", "180° reversal THROUGH a stop vs NFL turns at ≥ 3 yd/s — different geometry"),
    ("short_shuttle", "accel_low"): ("medium", "acceleration from low speed", "lateral start from a stance"),
    ("short_shuttle", "repeated_change"): ("medium", "brake → accelerate sequence", "two reversals in one effort"),
    ("three_cone", "redirection"): ("medium", "turn at speed (90°) and speed retention", "curved running around a cone at speed"),
    ("three_cone", "braking"): ("medium", "deceleration before a reversal", "one 180° reversal"),
    ("three_cone", "reaccel_after_brake"): ("medium", "acceleration after a reversal", "exit from the reversal"),
    ("three_cone", "repeated_change"): ("medium", "sequence of speed changes", "several changes in one effort"),
    ("three_cone", "accel_low"): ("low", "acceleration from a stance", "short first leg"),
    ("long_shuttle", "braking"): ("medium", "deceleration from higher entry speed", "longer legs, higher entry speed"),
    ("long_shuttle", "reaccel_after_brake"): ("medium", "acceleration after a stop", "repeated reversals"),
    ("long_shuttle", "repeated_change"): ("high", "repeated brake → accelerate", "five legs"),
    ("gauntlet", "sustained_high"): ("medium", "speed held at high speed", "lateral run at speed while catching"),
    ("gauntlet", "speed_maintenance"): ("medium", "steady speed", "speed held across the field"),
    ("backpedal_transition", "reaccel_after_brake"): ("medium", "forward acceleration after a backward / lateral phase", "transition into forward drive"),
    ("backpedal_transition", "braking"): ("medium", "deceleration of the backpedal", "plant before the transition"),
    ("backpedal_transition", "redirection"): ("medium", "change of direction of motion", "hip turn / break; geometry varies by drill"),
    ("pass_rush", "accel_low"): ("high", "launch acceleration from a stance over 2–5 yd", "get-off from a stance"),
    ("pass_rush", "redirection"): ("low", "lateral redirection", "bag / hoop drills; geometry varies"),
    ("pass_rush", "repeated_change"): ("low", "repeated short accelerations", "multiple short phases"),
}


def overlap(family: str, demand: str) -> tuple[str, str, str]:
    if family == "other":
        return "unknown", "", "drill not mapped to a physical family — review the drill-name mapping"
    return PHYSICAL_OVERLAP.get((family, demand), ("none", "", "no shared movement state"))


# ============================================================================== Combine inventory
def attempt_summary(frames: pl.DataFrame, athletes: pl.DataFrame) -> pl.DataFrame:
    """One row per attempt. ``frames``: ``sources.combine_frames(usable_only=False)``; ``athletes``: nfl_id, position_group."""
    has = set(frames.columns)
    cov = lambda k: (pl.col(k).is_not_null() & pl.col(k).cast(pl.Float64).is_finite()).mean() if k in has else pl.lit(None, dtype=pl.Float64)
    f = frames.sort("seq_id", "ts").with_columns(pl.col("t_s").diff().over("seq_id").alias("_dt"))
    s = f.group_by("seq_id").agg(
        pl.col("nfl_id").first(), pl.col("family").first(), pl.col("drill").first(), pl.len().cast(pl.Int64).alias("n_frames"),
        (pl.col("t_s").max() - pl.col("t_s").min()).alias("duration_s"), pl.col("_dt").median().alias("dt_median"),
        ((pl.col("_dt") - 0.1).abs() > 0.011).mean().alias("share_irregular_dt"),
        cov("x_s").alias("coverage_xy"), cov("speed").alias("coverage_speed"), cov("a_tan").alias("coverage_accel"),
        (pl.col("heading_valid").cast(pl.Float64).mean() if "heading_valid" in has else pl.lit(None, dtype=pl.Float64)).alias("coverage_heading"),
        pl.col("speed").first().alias("first_speed"), pl.col("speed").last().alias("last_speed"), pl.col("speed").max().alias("peak_speed"),
        pl.col("seg_idx").n_unique().cast(pl.Int64).alias("n_segments") if "seg_idx" in has else pl.lit(1, dtype=pl.Int64).alias("n_segments"))
    s = s.with_columns((pl.col("first_speed") >= 1.0).alias("start_censored"), (pl.col("last_speed") >= 0.8 * pl.col("peak_speed")).alias("end_censored"))
    return s.join(athletes.select("nfl_id", "position_group"), on="nfl_id", how="left").sort("family", "drill", "seq_id")


def drill_inventory(summary: pl.DataFrame, usable_ids: Sequence[str] | None = None) -> pl.DataFrame:
    """Per drill: counts, positions, sampling, coverage, censoring, quality exclusions (attempts not in ``usable_ids``)."""
    s = summary.with_columns((pl.col("seq_id").is_in(list(usable_ids)) if usable_ids is not None else pl.lit(True)).alias("usable"))
    pos = (s.group_by("drill", "position_group").agg(pl.col("nfl_id").n_unique().alias("k")).sort("drill", "k", descending=[False, True])
           .group_by("drill", maintain_order=True).agg((pl.col("position_group").fill_null("?") + ":" + pl.col("k").cast(pl.String)).str.join(", ").alias("positions")))
    out = s.group_by("family", "drill").agg(
        pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"), pl.len().cast(pl.Int64).alias("n_attempts"),
        pl.col("n_frames").sum().cast(pl.Int64).alias("n_frames"), pl.col("duration_s").median().alias("duration_p50_s"),
        pl.col("duration_s").quantile(0.95, "linear").alias("duration_p95_s"), pl.col("dt_median").median().alias("dt_median_s"),
        pl.col("share_irregular_dt").mean().alias("share_irregular_dt"), pl.col("coverage_xy").mean(), pl.col("coverage_speed").mean(),
        pl.col("coverage_accel").mean(), pl.col("coverage_heading").mean(), pl.col("peak_speed").median().alias("peak_speed_p50"),
        pl.col("start_censored").mean().alias("share_start_censored"), pl.col("end_censored").mean().alias("share_end_censored"),
        (~pl.col("usable")).sum().cast(pl.Int64).alias("n_quality_excluded"), pl.col("usable").sum().cast(pl.Int64).alias("n_usable_attempts"))
    order = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return (out.join(pos, on="drill", how="left").with_columns(pl.col("family").replace_strict(order, default=99, return_dtype=pl.Int64).alias("_o"))
            .sort("_o", "drill").drop("_o"))


def family_position_coverage(summary: pl.DataFrame) -> pl.DataFrame:
    return (summary.group_by("family", "position_group").agg(pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"),
                                                             pl.len().cast(pl.Int64).alias("n_attempts")).sort("family", "position_group"))


# ============================================================================== NFL inventory
def nfl_inventory(episodes: pl.DataFrame, plays: pl.DataFrame, pairs: pl.DataFrame, c: M11Config) -> dict[str, pl.DataFrame]:
    el = plays.filter(pl.col("eligible"))
    pos = el.group_by("position_group").agg(pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"), pl.col("game_id").n_unique().cast(pl.Int64).alias("n_games"),
                                            pl.col("season").n_unique().cast(pl.Int64).alias("n_seasons"), pl.len().cast(pl.Int64).alias("n_eligible_plays"),
                                            pl.col("season").min().alias("first_season"), pl.col("season").max().alias("last_season")).sort("position_group")
    dem = episodes.group_by("position_group", "demand_id").agg(
        pl.len().cast(pl.Int64).alias("n_episodes"), pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes"),
        pl.col("game_id").n_unique().cast(pl.Int64).alias("n_games"), pl.col("season").n_unique().cast(pl.Int64).alias("n_seasons"),
        *[pl.col(b).is_not_null().sum().cast(pl.Int64).alias(f"n_{b}") for b in ("acc_2_4", "acc_4_6", "acc_6_8")],
        pl.col("heading_change_deg").is_not_null().sum().cast(pl.Int64).alias("n_valid_heading"))
    per_ath = episodes.group_by("position_group", "demand_id", "nfl_id").len()
    sup = per_ath.group_by("position_group", "demand_id").agg(*[(pl.col("len") >= k).sum().cast(pl.Int64).alias(f"n_athletes_ge_{k}") for k in c.exposure_grid])
    dem = dem.join(sup, on=["position_group", "demand_id"], how="left").sort("position_group", "demand_id")
    pr = pairs.group_by("position_group").agg(pl.len().cast(pl.Int64).alias("n_brake_reaccel_pairs"), pl.col("angle_observed").sum().cast(pl.Int64).alias("n_pairs_with_turn"),
                                              pl.col("nfl_id").n_unique().cast(pl.Int64).alias("n_athletes_pairs")) if pairs.height else pl.DataFrame()
    lab = pl.DataFrame([{"label": k, "share_episodes_labelled": float(episodes[k].is_not_null().mean()) if k in episodes.columns and episodes.height else 0.0,
                         "n_levels": int(episodes[k].drop_nulls().n_unique()) if k in episodes.columns else 0}
                        for k in ("route_label", "coverage_label", "play_class", "side", "lined_up_position", "down")])
    ctx = pl.DataFrame([
        {"item": "ball tracking", "status": UNKNOWN, "note": "not used by Milestones 05–10; 08A found tracking holds the project athletes only"},
        {"item": "opponent tracking", "status": UNSUPPORTED, "note": "08A: nearest defender not recomputable (tracking holds project athletes only)"},
        {"item": "vendor separation (FO01)", "status": LIMITED, "note": "08B registry frozen_exploratory; semantics inferred"},
        {"item": "body orientation (vendor o)", "status": UNSUPPORTED, "note": "dropped by the 05A/09A reader; backpedal (moving opposite to facing) not identifiable"}])
    return {"m11_nfl_position_inventory": pos.join(pr, on="position_group", how="left") if pr.height else pos, "m11_nfl_demand_inventory": dem,
            "m11_nfl_label_availability": lab, "m11_nfl_context_availability": ctx}


# ============================================================================== feasibility
def feasibility_matrix(attempt_athletes: pl.DataFrame, nfl_athletes: pl.DataFrame, c: M11Config) -> pl.DataFrame:
    """``attempt_athletes``: (nfl_id, family, position_group) of usable attempts · ``nfl_athletes``: (nfl_id, position_group,
    demand_id, n_episodes) from the 09A episodes. A linked athlete counts in a position cell only when the Combine registry
    and the NFL episodes give the same position group (a position change removes the athlete from the cell — reported by
    ``n_combine_athletes`` vs ``n_linked``)."""
    rows = []
    fams = sorted(set(attempt_athletes["family"].to_list()), key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99)
    positions = sorted(set(nfl_athletes["position_group"].drop_nulls().to_list()))
    for fam in fams:
        cmb = attempt_athletes.filter(pl.col("family") == fam)
        for dem in DEMANDS:
            conf, qty, why = overlap(fam, dem)
            for pos in positions:
                c_ids = set(cmb.filter(pl.col("position_group") == pos)["nfl_id"].to_list())
                n_all = nfl_athletes.filter((pl.col("position_group") == pos) & (pl.col("demand_id") == dem))
                n_ids = set(n_all.filter(pl.col("n_episodes") >= c.min_episodes_athlete)["nfl_id"].to_list())
                linked = c_ids & set(n_all["nfl_id"].to_list())
                linked_ok = c_ids & n_ids
                completeness = len(linked_ok) / len(linked) if linked else NAN
                if conf == "unknown":
                    state = UNKNOWN
                elif conf == "none":
                    state = UNSUPPORTED
                elif len(linked_ok) >= c.min_athletes_cell and CONFIDENCE_RANK[conf] >= 2:
                    state = SUPPORTED
                elif len(linked_ok) >= c.min_athletes_limited:
                    state = LIMITED
                else:
                    state = UNSUPPORTED
                reason = ("no shared movement state (never a zero effect)" if conf == "none" else
                          f"{len(linked_ok)} linked athletes with ≥ {c.min_episodes_athlete} episodes; overlap {conf}")
                rows.append({"family": fam, "demand_id": dem, "position_group": pos, "overlap_confidence": conf, "shared_quantity": qty, "overlap_note": why,
                             "n_combine_athletes": len(c_ids), "n_nfl_athletes": len(n_ids), "n_linked": len(linked), "n_linked_supported": len(linked_ok),
                             "data_completeness": completeness, "feasibility": state, "reason": reason})
    return pl.DataFrame(rows)


def pooled_feasibility(fm: pl.DataFrame, c: M11Config) -> pl.DataFrame:
    """Family × demand over positions (the unit of the 11A matrix): linked athletes summed over positions."""
    g = fm.group_by("family", "demand_id").agg(pl.col("overlap_confidence").first(), pl.col("shared_quantity").first(),
                                               pl.col("n_linked_supported").sum().cast(pl.Int64).alias("n_linked_supported"),
                                               pl.col("n_combine_athletes").sum().cast(pl.Int64).alias("n_combine_athletes"))
    rank = pl.col("overlap_confidence").replace_strict(CONFIDENCE_RANK, default=-1, return_dtype=pl.Int64)
    return g.with_columns(
        pl.when(pl.col("overlap_confidence") == "unknown").then(pl.lit(UNKNOWN))
        .when(pl.col("overlap_confidence") == "none").then(pl.lit(UNSUPPORTED))
        .when((pl.col("n_linked_supported") >= c.min_athletes_cell) & (rank >= 2)).then(pl.lit(SUPPORTED))
        .when(pl.col("n_linked_supported") >= c.min_athletes_limited).then(pl.lit(LIMITED)).otherwise(pl.lit(UNSUPPORTED)).alias("feasibility")
    ).sort("family", "demand_id")


def support_funnel(registry: pl.DataFrame, attempt_athletes: pl.DataFrame, nfl_athletes: pl.DataFrame, c: M11Config) -> pl.DataFrame:
    tracked = set(attempt_athletes["nfl_id"].to_list())
    nfl_any = set(nfl_athletes["nfl_id"].to_list())
    nfl_ok = set(nfl_athletes.filter(pl.col("n_episodes") >= c.min_episodes_athlete)["nfl_id"].to_list())
    nonforty = set(attempt_athletes.filter(pl.col("family") != "forty")["nfl_id"].to_list())
    stages = [("01 Combine athletes (registry)", set(registry["nfl_id"].to_list())), ("02 ≥ 1 usable tracked attempt (any drill)", tracked),
              ("03 … and a non-forty tracked drill", tracked & nonforty), ("04 … linked to NFL episodes (any drill)", tracked & nfl_any),
              (f"05 … with ≥ {c.min_episodes_athlete} episodes of some demand", tracked & nfl_ok),
              ("06 non-forty drill AND ≥ min episodes", tracked & nonforty & nfl_ok)]
    return pl.DataFrame([{"stage": s, "n_athletes": len(ids)} for s, ids in stages])


# ============================================================================== registries
def measurement_registry() -> pl.DataFrame:
    """Cross-domain quantities: the same physical rule on both sides, or an explicit statement that it differs."""
    rows = [
        ("band_accel_2_4 / 4_6 / 6_8", "yd/s²", "forty: first crossings of lo and hi in the recorded trace", "accelerating episode: same first-crossing rule",
         "same rule, 10 Hz both sides; Combine passes the band once from rest, NFL from any start inside an accelerating run", "SUPPORTED"),
        ("band_accel_8_10", "yd/s²", "forty: same rule (often censored: recordings end near the peak)", "envelope at 9 yd/s (episodes hold no 8→10 band)",
         "different estimator on the NFL side (envelope) — LIMITED", "LIMITED"),
        ("top speed", "yd/s", "forty: largest tracked speed (may be censored)", "sustained_high: 90th percentile of episode peaks", "game speed is rarely maximal", "LIMITED"),
        ("sprint A0 / Vmax ↔ envelope A0 / V0", "yd/s² · yd/s", "phase-plane fit a = A0 − (A0/V0) v on the rising forty",
         "linear fit to the per-speed-bin 95th percentile of game a_tan", "same functional form; one maximal effort vs an observed upper tail", "SUPPORTED"),
        ("deceleration", "yd/s²", "shuttle / cone: mean −a_tan over the braking run (a_tan ≤ −1)", "braking episode: −(speed change / duration)",
         "Combine: run-mean of frames; NFL: episode mean — same threshold, close definitions", "SUPPORTED"),
        ("Brake-to-Burst latency", "s", "braking end → first a_tan ≥ 1.5", "braking end → start of the following accelerating run (a_tan ≥ 1)",
         "burst threshold differs (1.5 vs 1.0 yd/s²) — declared, kept", "LIMITED"),
        ("Transition Cost Index", "fraction", "(v_entry − v 0.5 s after the speed minimum) / v_entry", "(v_entry − v 0.5 s into the re-acceleration) / v_entry",
         "anchor differs (minimum vs re-acceleration start, ≤ 1 s apart)", "LIMITED"),
        ("turn angle", "deg", "angle between mean velocity before / after (valid through a stop)", "integrated yaw of a redirection episode (≥ 3 yd/s, valid heading)",
         "Combine reversals pass through a stop; NFL turns are at speed", "LIMITED"),
        ("lateral acceleration", "yd/s²", "(vx·ay − vy·ax)/|v| above 1 yd/s", "not persisted in the 09A episodes", "Combine only", "UNSUPPORTED"),
    ]
    return pl.DataFrame([{"quantity": q, "unit": u, "combine_definition": a, "nfl_definition": b, "comparability": n, "state": s} for q, u, a, b, n, s in rows])


def movement_gap_registry_v2(registry: pl.DataFrame) -> pl.DataFrame:
    """Corrected Movement Gap definitions (engine ``movement_gap.VERSION`` ≥ 10.2). ``registry``: 09A demand registry."""
    rows = []
    for r in registry.iter_rows(named=True):
        dims = mg._analogue(r)
        cannot = r["demand_id"] in mg.FORTY_CANNOT_MEASURE
        rows.append({"demand_id": r["demand_id"], "label": r["label"], "registry_text_analogue": r.get("combine_analogue") or "",
                     "forty_analogue_v2": ", ".join(dims) or "none", "forty_can_measure": bool(dims) and not cannot,
                     "class_rule_v2": "not_measured (structural: the timed forty contains no such phase)" if cannot else
                     ("direct / partial by start state" if dims else "not_measured (no shared dimension)"),
                     "class_v1_bug": "10.1 read the placeholder text 'none' as a dimension → 'unavailable (data limitation)'" if cannot else "",
                     "engine_version": mg.VERSION})
    return pl.DataFrame(rows)


def config_manifest(c: M11Config, mapping: pl.DataFrame, overrides: dict[str, str], studies: list[dict[str, Any]]) -> dict[str, Any]:
    return {"config": c.params(), "config_hash": c.hash(), "drill_mapping": mapping.select("drill", "family").to_dicts() if mapping.height else [],
            "drill_overrides": overrides, "studies": studies, "overlap_declarations": [{"family": f, "demand_id": d, "confidence": v[0], "quantity": v[1]}
                                                                                        for (f, d), v in PHYSICAL_OVERLAP.items()]}


__all__ = ["CONFIDENCE_RANK", "DEMANDS", "PHYSICAL_OVERLAP", "attempt_summary", "config_manifest", "drill_inventory", "family_position_coverage",
           "feasibility_matrix", "measurement_registry", "movement_gap_registry_v2", "nfl_inventory", "overlap", "pooled_feasibility", "support_funnel"]
