"""11H — position-specific football-action linkages (feasibility first).

An *action* is a physical episode rule restricted by OBSERVED context (play class, side, recorded vendor route label,
time after the snap) — never a role inferred from movement. Each action declares its semantic status:

    supported     context recorded by the vendor and used as recorded
    proxy         a physical/contextual proxy (e.g. run plays as a blocking context for TEs) — confounded by role; labelled
    unsupported   the event or orientation it needs is not in the warehouse → NOT_FEASIBLE with the reason and an alternative

A predictive experiment (M0 → M3) runs only when: semantics ≠ unsupported, linked athletes ≥ ``min_athletes_cell`` with
≥ ``min_episodes_athlete`` episodes, and game split-half reliability ≥ ``MIN_RELIABILITY``. Otherwise the action is
described (distributions, exposure, context) and prediction is marked unsupported. Game context is NEVER a pre-draft
predictor: it only defines which episodes form the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from . import evaluation as ev
from .combine_drills import PRESELECTED
from .config import LIMITED, M11Config, NOT_FEASIBLE, SUPPORTED

NAN = float("nan")
MIN_RELIABILITY = 0.5


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    position: str
    label: str
    source: str                 # episodes · pairs · football_rows
    demands: tuple[str, ...]
    value: str                  # column / derived name
    context: str                # human-readable restriction
    semantics: str              # supported · proxy · unsupported
    drill_families: tuple[str, ...]
    note: str = ""


ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec("WR.release_launch", "WR", "release acceleration (≤ 1 s after the snap, pass plays)", "episodes", ("accel_low",), "mean_accel",
               "play_class = pass · t_start ≤ 1.0 s", "supported", ("forty",)),
    ActionSpec("WR.route_brake_reaccel", "WR", "brake → re-acceleration on a recorded route (TCI)", "pairs", (), "tci_05",
               "pass plays with a vendor route label", "supported", ("short_shuttle", "three_cone"), "route label = silver.player_play.route_ran"),
    ActionSpec("WR.route_brake_to_burst", "WR", "Brake-to-Burst latency on a recorded route", "pairs", (), "latency_s",
               "pass plays with a vendor route label", "supported", ("short_shuttle", "three_cone")),
    ActionSpec("WR.separation", "WR", "separation at the frozen 08B definition (FO01.S0)", "football_rows", (), "value",
               "08B.0 registry frozen_exploratory; semantics inferred", "proxy", ("forty", "short_shuttle", "three_cone")),
    ActionSpec("WR.after_catch_accel", "WR", "acceleration after the catch", "none", (), "", "needs a catch event per frame", "unsupported", ("forty",),
               "no frame-level catch event in the episode tables → alternative: release launch and route transitions"),
    ActionSpec("DB.backpedal_transition", "DB", "backpedal-to-forward transition", "none", (), "", "needs body orientation (vendor o)", "unsupported",
               ("backpedal_transition",), "orientation dropped by the 05A/09A reader → alternative: DB.brake_drive"),
    ActionSpec("DB.brake_drive", "DB", "brake → drive on pass plays (closing proxy)", "pairs", (), "reaccel_mean", "pass plays, defense", "proxy",
               ("backpedal_transition", "short_shuttle")),
    ActionSpec("DB.recovery_accel", "DB", "re-acceleration after braking on pass plays", "episodes", ("reaccel_after_brake",), "mean_accel", "pass plays",
               "supported", ("short_shuttle", "three_cone", "backpedal_transition")),
    ActionSpec("DB.redirection", "DB", "speed kept through a redirection on pass plays", "episodes", ("redirection",), "_retention", "pass plays · valid heading",
               "supported", ("three_cone",)),
    ActionSpec("DL.launch", "DL", "launch from low speed ≤ 0.8 s after the snap", "episodes", ("accel_low",), "mean_accel", "t_start ≤ 0.8 s", "supported",
               ("forty", "pass_rush")),
    ActionSpec("DL.pass_rush_speed", "DL", "speed developed on pass plays (first acceleration)", "episodes", ("accel_low", "accel_mid"), "v_max",
               "play_class = pass · first accelerating episode", "proxy", ("forty", "pass_rush"), "no rush-assignment label; all pass plays"),
    ActionSpec("OL.short_burst", "OL", "short burst (≤ 5 yd path) from low speed", "episodes", ("accel_low",), "mean_accel", "distance ≤ 5 yd", "supported",
               ("forty", "pass_rush")),
    ActionSpec("OL.redirection", "OL", "lateral redirection (speed kept)", "episodes", ("redirection",), "_retention", "valid heading", "supported", ("three_cone",)),
    ActionSpec("TE.receiving_brake_reaccel", "TE", "brake → re-acceleration on a recorded route", "pairs", (), "tci_05", "pass plays with a vendor route label",
               "supported", ("short_shuttle", "three_cone")),
    ActionSpec("TE.blocking_launch", "TE", "launch on non-pass plays (blocking context proxy)", "episodes", ("accel_low",), "mean_accel",
               "play_class = no_pass · t_start ≤ 1.0 s", "proxy", ("forty",), "run plays ≠ blocking assignment; confounded by role"),
)


def action_values(a: ActionSpec, episodes: pl.DataFrame, pairs: pl.DataFrame, football_rows: pl.DataFrame | None) -> pl.DataFrame:
    """(nfl_id, game_id, value) of every episode / pair / play forming the action target (NFL quantities only)."""
    empty = pl.DataFrame(schema={"nfl_id": pl.Int64, "game_id": pl.Int64, "value": pl.Float64})
    if a.source == "none" or a.semantics == "unsupported":
        return empty
    if a.source == "football_rows":
        if football_rows is None or football_rows.is_empty():
            return empty
        d = football_rows.filter(pl.col("definition_id").str.starts_with("FO01.S0") & (pl.col("position_group") == a.position))
        return d.select("nfl_id", "game_id", pl.col("value").cast(pl.Float64)).drop_nulls()
    if a.source == "pairs":
        ctx = episodes.select("game_id", "play_id", "nfl_id", "play_class", "route_label").unique(subset=["game_id", "play_id", "nfl_id"])
        d = pairs.filter(pl.col("position_group") == a.position).join(ctx, on=["game_id", "play_id", "nfl_id"], how="left")
        d = d.filter(pl.col("play_class") == "pass")
        if "route label" in a.context:
            d = d.filter(pl.col("route_label").is_not_null())
        return d.select("nfl_id", "game_id", pl.col(a.value).cast(pl.Float64).alias("value")).filter(pl.col("value").is_finite())
    e = episodes.filter((pl.col("position_group") == a.position) & pl.col("demand_id").is_in(list(a.demands)))
    if "pass" in a.context and "no_pass" not in a.context:
        e = e.filter(pl.col("play_class") == "pass")
    if "no_pass" in a.context:
        e = e.filter(pl.col("play_class") == "no_pass")
    for lim in (0.8, 1.0):
        if f"t_start ≤ {lim}" in a.context:
            e = e.filter(pl.col("t_start") <= lim)
    if "distance ≤ 5 yd" in a.context:
        e = e.filter(pl.col("distance_yd") <= 5.0)
    if "first accelerating episode" in a.context:
        e = e.sort("t_start").unique(subset=["game_id", "play_id", "nfl_id"], keep="first")
    val = (pl.col("v_start") + pl.col("speed_change")) / pl.col("v_start") if a.value == "_retention" else pl.col(a.value)
    return e.select("nfl_id", "game_id", val.cast(pl.Float64).alias("value")).filter(pl.col("value").is_finite())


def feasibility_table(episodes: pl.DataFrame, pairs: pl.DataFrame, football_rows: pl.DataFrame | None, combine_ids: Sequence[int], c: M11Config
                      ) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    rows, values = [], {}
    ids = set(int(i) for i in combine_ids)
    for a in ACTIONS:
        v = action_values(a, episodes, pairs, football_rows)
        values[a.action_id] = v
        per = v.group_by("nfl_id").len()
        linked = per.filter(pl.col("nfl_id").is_in(list(ids)) & (pl.col("len") >= c.min_episodes_athlete)).height if per.height else 0
        rel = ev.split_half(v, "value", "game_id", min_per_half=max(2, c.min_episodes_athlete // 4)) if v.height else {"reliability": NAN, "n_athletes": 0}
        if a.semantics == "unsupported":
            state, why = NOT_FEASIBLE, a.note
        elif linked >= c.min_athletes_cell and np.isfinite(rel["reliability"]) and rel["reliability"] >= MIN_RELIABILITY:
            state, why = SUPPORTED, f"{linked} linked athletes, split-half reliability {rel['reliability']:.2f}"
        elif linked >= c.min_athletes_limited:
            state, why = LIMITED, f"{linked} linked athletes; reliability {rel['reliability']:.2f} (needs ≥ {MIN_RELIABILITY}) — descriptive only"
        else:
            state, why = NOT_FEASIBLE, f"{linked} linked athletes with ≥ {c.min_episodes_athlete} episodes"
        rows.append({"action_id": a.action_id, "position_group": a.position, "label": a.label, "semantics": a.semantics, "context": a.context,
                     "n_values": v.height, "n_athletes": per.height, "n_linked": linked, "reliability": rel["reliability"], "state": state, "reason": why,
                     "drill_families": ", ".join(a.drill_families), "note": a.note})
    return pl.DataFrame(rows), values


def athlete_action_targets(values: dict[str, pl.DataFrame], c: M11Config) -> pl.DataFrame:
    parts = [v.group_by("nfl_id").agg(pl.col("value").median().alias("value"), pl.len().cast(pl.Int64).alias("n")).filter(pl.col("n") >= c.min_episodes_athlete)
             .with_columns(pl.lit(k).alias("action_id")) for k, v in values.items() if v.height]
    return pl.concat(parts) if parts else pl.DataFrame()


def evaluate_actions(feas: pl.DataFrame, targets: pl.DataFrame, linked: pl.DataFrame, c: M11Config, seed: int) -> pl.DataFrame:
    """M0 context · M1 officials · M2 drill features of the action's first family · M3 all declared families — SUPPORTED actions
    only. ``linked``: one row per athlete with officials, Combine features (wide) and ``fold``."""
    rows = []
    for a in ACTIONS:
        f = feas.filter(pl.col("action_id") == a.action_id).row(0, named=True)
        if f["state"] != SUPPORTED:
            rows.append({"action_id": a.action_id, "position_group": a.position, "state": f["state"], "status": "prediction_unsupported", "reason": f["reason"]})
            continue
        t = targets.filter(pl.col("action_id") == a.action_id).select("nfl_id", pl.col("value").alias("_y"))
        d = linked.filter(pl.col("position_group") == a.position).join(t, on="nfl_id", how="inner")
        first = [k for k in PRESELECTED.get(a.drill_families[0], ()) if k in d.columns]
        allf = [k for fam in a.drill_families for k in PRESELECTED.get(fam, ()) if k in d.columns]
        offs = [k for k in ev.OFFICIAL_CORE if k in d.columns]
        lad = {"M0": list(ev.CONTEXT), "M1": [*ev.CONTEXT, *offs], "M2": [*ev.CONTEXT, *offs, *first]}
        if len(allf) > len(first):
            lad["M3"] = [*ev.CONTEXT, *offs, *allf]
        if d.height < c.min_athletes_limited or not first:
            rows.append({"action_id": a.action_id, "position_group": a.position, "state": f["state"], "status": "too_few_with_drill", "n_athletes": d.height})
            continue
        res, _ = ev.run_ladder(d, "_y", lad, d["fold"].to_numpy(), c, seed, labels={"action_id": a.action_id, "position_group": a.position})
        rows += [{**r, "state": f["state"], "status": "computed", "semantics": a.semantics} for r in res]
    return pl.DataFrame(rows, infer_schema_length=None)


def context_description(episodes: pl.DataFrame, position: str, demand: str) -> pl.DataFrame:
    """Episode counts and median intensity by recorded route label (top levels) for one position × demand (descriptive)."""
    e = episodes.filter((pl.col("position_group") == position) & (pl.col("demand_id") == demand) & pl.col("route_label").is_not_null())
    return (e.group_by("route_label").agg(pl.len().alias("n_episodes"), pl.col("nfl_id").n_unique().alias("n_athletes"),
                                          pl.col("mean_accel").median().alias("median_mean_accel"), pl.col("v_start").median().alias("median_v_start"))
            .sort("n_episodes", descending=True).head(15))


__all__ = ["ACTIONS", "ActionSpec", "MIN_RELIABILITY", "action_values", "athlete_action_targets", "context_description", "evaluate_actions", "feasibility_table"]
