"""11A — drill-to-NFL transfer matrix.

For every drill family × NFL demand with DECLARED physical overlap (``inventory.PHYSICAL_OVERLAP``):

    target      the declared NFL target of the demand (``DEMAND_TARGETS``; first = primary)
    M0          position + draft year + height + weight
    M1          M0 + official forty and 10-yd split (+ the drill's official time when the drill has one)
    M2          M1 + the ≤ 3 PRESELECTED interpretable features of the drill (declared before any NFL value was read)
    M3          M1 + every feature of the drill family (ridge-regularised alternative representation)

Controls (SUPPORTED cells): wrong-drill features of the same size, random Gaussian projections of the same size,
athlete-label permutation within position (declared PRIMARY cells only, bounded budget), noncomparable interval.

Cell class (``config.MATRIX_CLASSES``) — an untested or underpowered cell is NEVER a zero effect:
    not_tested                         no physical overlap (or no declared target)
    insufficient_data                  overlap, but too few linked athletes with enough episodes
    established_physical_overlap       overlap and data, no statistical evidence either way (LIMITED cell, CI includes 0)
    exploratory_statistical_relationship   association interval excludes 0, held-out gain not shown
    supported_held_out_transfer        M2 − M1 interval > 0 AND above every control (and permutation p < 0.05 when run)
    no_established_transfer            SUPPORTED cell tested, held-out interval includes 0 or below a control
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from . import evaluation as ev
from .combine_drills import PRESELECTED
from .config import (ESTABLISHED_OVERLAP, EXPLORATORY_RELATION, INSUFFICIENT_DATA, LIMITED, M11Config, NO_ESTABLISHED_TRANSFER, NOT_TESTED,
                     SUPPORTED, SUPPORTED_TRANSFER)
from .inventory import overlap

NAN = float("nan")
DEMAND_TARGETS: dict[str, tuple[str, ...]] = {
    "accel_low": ("nfl.accel_low_mean", "nfl.band_2_4"), "accel_mid": ("nfl.band_4_6",), "transition_high": ("nfl.band_6_8", "nfl.transition_high_mean"),
    "sustained_high": ("nfl.vmax_p90",), "speed_maintenance": (), "braking": ("nfl.brake_decel", "nfl.brake_peak05"),
    "reaccel_after_brake": ("nfl.reaccel_mean", "nfl.brake_to_burst"), "redirection": ("nfl.redirect_retention", "nfl.redirect_retention_90"),
    "repeated_change": (),
}
PRIMARY_CELLS = (("forty", "accel_low"), ("short_shuttle", "braking"), ("short_shuttle", "reaccel_after_brake"), ("three_cone", "redirection"),
                 ("pass_rush", "accel_low"))
WRONG_DRILL = {"forty": "short_shuttle"}            # every other family uses the forty as its wrong drill
NONCOMPARABLE: dict[str, tuple[str, ...]] = {        # features whose physical interval does not match the demand
    "accel_low": ("forty.band_6_8_mean_accel",), "transition_high": ("forty.band_2_4_mean_accel",), "accel_mid": ("forty.band_2_4_mean_accel",),
    "braking": ("forty.band_6_8_mean_accel",), "reaccel_after_brake": ("forty.v_max_obs",), "redirection": ("forty.band_4_6_mean_accel",),
    "sustained_high": ("forty.early_accel",),
}
DRILL_OFFICIAL = {"short_shuttle": "short_shuttle", "three_cone": "three_cone"}


def officials_for(family: str, data: pl.DataFrame) -> list[str]:
    out = [k for k in ev.OFFICIAL_CORE if k in data.columns]
    if family in DRILL_OFFICIAL and DRILL_OFFICIAL[family] in data.columns and data[DRILL_OFFICIAL[family]].drop_nulls().len() >= 10:
        out.append(DRILL_OFFICIAL[family])
    return out


def family_features(columns: Sequence[str], family: str) -> list[str]:
    return sorted(k for k in columns if k.startswith(f"{family}.") and not k.endswith(("_censored", "_identifiable", "_grid_edge", "censored_share")))


def run_cell(data: pl.DataFrame, family: str, demand: str, target: str, folds: np.ndarray, c: M11Config, seed: int, with_controls: bool,
             with_permutation: bool) -> dict[str, Any]:
    """Ladder + controls of ONE cell on the athletes that have the target and ≥ 1 preselected feature."""
    track = [k for k in PRESELECTED.get(family, ()) if k in data.columns][: c.max_tracking_features]
    out: dict[str, Any] = {"family": family, "demand_id": demand, "target": target, "tracking_features": ", ".join(track)}
    if not track or target not in data.columns:
        return {**out, "status": "not_available"}
    mask = data.select((pl.col(target).is_not_null() & pl.any_horizontal([pl.col(k).is_not_null() for k in track])).alias("_m"))["_m"].to_numpy()
    d, f = data.filter(pl.Series(mask)), folds[mask]
    out["n_athletes"] = d.height
    if d.height < c.min_athletes_limited:
        return {**out, "status": "too_few_athletes"}
    offs = officials_for(family, d)
    alt = family_features(d.columns, family)
    lad = ev.ladder_columns(ev.CONTEXT, offs, track, alt if len(alt) > len(track) else ())
    rows, oof = ev.run_ladder(d, target, lad, f, c, seed, labels={"family": family, "demand_id": demand})
    res = {r["model"]: r for r in rows}
    m2 = res.get("M2", {})
    out |= {"officials": ", ".join(offs), "status": "computed", **{f"m2_{k}": m2.get(k) for k in ("delta_rmse", "delta_lo", "delta_hi", "delta_pct",
                                                                                                 "spearman_model", "spearman_ref", "verdict")},
            **{f"m1_{k}": res.get("M1", {}).get(k) for k in ("delta_rmse", "delta_lo", "delta_hi", "verdict")},
            **{f"m3_{k}": res.get("M3", {}).get(k) for k in ("delta_rmse", "delta_lo", "delta_hi", "verdict")}}
    x = d[track[0]].cast(pl.Float64).to_numpy()
    a = ev.association(x, d[target].cast(pl.Float64).to_numpy(), None, max(200, c.n_boot // 5), seed, positions=d["position_group"].to_numpy())
    out |= {"assoc_feature": track[0], "assoc_spearman": a["spearman"], "assoc_lo": a["spearman_lo"], "assoc_hi": a["spearman_hi"],
            "within_position_spearman": a["within_position_spearman"], "mde_r": a["mde_r"]}
    ctrl_q95 = []
    if with_controls:
        m1 = lad["M1"]
        wrong = WRONG_DRILL.get(family, "forty")
        wfeat = [k for k in PRESELECTED.get(wrong, ()) if k in d.columns][: len(track)]
        nc = [k for k in NONCOMPARABLE.get(demand, ()) if k in d.columns and k not in track]
        crow = ev.control_rows(d, target, m1, {"wrong_drill": wfeat, "noncomparable_interval": nc}, f, c, seed, labels={"family": family, "demand_id": demand})
        for r in crow:
            out[f"ctrl_{r['control']}_delta"] = r["delta_rmse"]
            out[f"ctrl_{r['control']}_hi"] = r["delta_hi"]
            if r["delta_hi"] is not None and np.isfinite(r["delta_hi"]):
                ctrl_q95.append(r["delta_rmse"])
        rp = ev.random_projection_control(d, target, m1, len(track), f, c, seed)
        out["ctrl_random_q95"] = float(np.quantile(rp, 0.95)) if rp.size else NAN
        if rp.size:
            ctrl_q95.append(out["ctrl_random_q95"])
    if with_permutation:
        pn = ev.permutation_null(d, target, lad["M1"], lad["M2"], f, c, seed)
        out |= {"perm_p": pn["p_value"], "perm_null_q95": pn["null_q95"], "perm_n": pn["n_perm"]}
    out["beats_controls"] = bool(ctrl_q95) and m2.get("delta_rmse") is not None and np.isfinite(m2.get("delta_rmse", NAN)) and m2["delta_rmse"] > max(ctrl_q95)
    out["_oof"] = oof
    return out


def classify(cell: dict[str, Any], feasibility: str, overlap_conf: str) -> str:
    if overlap_conf in ("none", "unknown") or not cell.get("target"):
        return NOT_TESTED
    if feasibility not in (SUPPORTED, LIMITED) or cell.get("status") in ("too_few_athletes", "not_available"):
        return INSUFFICIENT_DATA
    assoc = cell.get("assoc_lo") is not None and np.isfinite(cell.get("assoc_lo") or NAN) and (cell["assoc_lo"] > 0 or cell["assoc_hi"] < 0)
    if feasibility == SUPPORTED:
        perm_ok = cell.get("perm_p") is None or (np.isfinite(cell["perm_p"]) and cell["perm_p"] < 0.05)
        if cell.get("m2_verdict") == "model_better" and cell.get("beats_controls") and perm_ok:
            return SUPPORTED_TRANSFER
        return EXPLORATORY_RELATION if assoc else NO_ESTABLISHED_TRANSFER
    return EXPLORATORY_RELATION if assoc else ESTABLISHED_OVERLAP


def run_matrix(data: pl.DataFrame, pooled: pl.DataFrame, folds: np.ndarray, c: M11Config, seed: int, targets_override: dict[str, tuple[str, ...]] | None = None,
               all_targets: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(matrix rows, out-of-fold predictions) over every family × demand of ``pooled`` (``inventory.pooled_feasibility``)."""
    targets = {**DEMAND_TARGETS, **(targets_override or {})}
    rows, oofs = [], []
    for r in pooled.iter_rows(named=True):
        fam, dem = r["family"], r["demand_id"]
        conf = overlap(fam, dem)[0]
        tlist = targets.get(dem, ())
        if conf in ("none", "unknown") or not tlist:
            rows.append({"family": fam, "demand_id": dem, "target": tlist[0] if tlist else "", "feasibility": r["feasibility"], "overlap_confidence": conf,
                         "status": "not_tested", "matrix_class": NOT_TESTED, "primary_target": True})
            continue
        for ti, tgt in enumerate(tlist if all_targets else tlist[:1]):
            primary_cell = (fam, dem) in PRIMARY_CELLS and ti == 0
            cell = run_cell(data, fam, dem, tgt, folds, c, seed, with_controls=r["feasibility"] == SUPPORTED, with_permutation=primary_cell and r["feasibility"] == SUPPORTED)
            oof = cell.pop("_oof", None)
            if oof is not None and oof.height:
                oofs.append(oof.with_columns(pl.lit(tgt).alias("target")))
            rows.append({**cell, "feasibility": r["feasibility"], "overlap_confidence": conf, "primary_cell": primary_cell, "primary_target": ti == 0,
                         "matrix_class": classify(cell, r["feasibility"], conf)})
    return pl.DataFrame(rows, infer_schema_length=None), (pl.concat(oofs, how="diagonal_relaxed") if oofs else pl.DataFrame())


def exposure_sensitivity(data: pl.DataFrame, counts: pl.DataFrame, cells: pl.DataFrame, folds: np.ndarray, c: M11Config, seed: int) -> pl.DataFrame:
    """M2 vs M1 of the computed primary-target cells restricted to athletes with ≥ k target episodes (k in the exposure grid).
    ``counts``: (nfl_id, target_id, n_episodes)."""
    rows = []
    for cell in cells.filter((pl.col("status") == "computed") & pl.col("primary_target")).iter_rows(named=True):
        for k in c.exposure_grid:
            ok_ids = counts.filter((pl.col("target_id") == cell["target"]) & (pl.col("n_episodes") >= k))["nfl_id"]
            m = data["nfl_id"].is_in(ok_ids.to_list()).to_numpy()
            sub = data.filter(pl.Series(m))
            r = run_cell(sub, cell["family"], cell["demand_id"], cell["target"], folds[m], c, seed, False, False)
            rows.append({"family": cell["family"], "demand_id": cell["demand_id"], "target": cell["target"], "min_episodes": k, "n_athletes": r.get("n_athletes", 0),
                         "m2_delta_rmse": r.get("m2_delta_rmse"), "m2_delta_lo": r.get("m2_delta_lo"), "m2_delta_hi": r.get("m2_delta_hi")})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["DEMAND_TARGETS", "NONCOMPARABLE", "PRIMARY_CELLS", "WRONG_DRILL", "classify", "exposure_sensitivity", "family_features", "officials_for",
           "run_cell", "run_matrix"]
