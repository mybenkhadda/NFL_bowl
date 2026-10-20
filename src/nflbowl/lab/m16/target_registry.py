"""Typed, versioned NFL target registry (Track A2) — the complete definition of every target ID in ``targets.TARGETS``.

``targets.TargetSpec`` stays the compact record the pipeline already uses; ``TargetDefinition`` adds what a reader needs to
judge validity: physical quantity, unit, source movement categories, required channels, eligibility, aggregation, minimum
observations, treatment of truncated windows, within-athlete uncertainty, exposure variable, position applicability,
failure conditions and a definition version. ``check_consistency`` guarantees the two registries name the same IDs and
columns (a test invariant). These constructed movement outcomes are NOT football performance; the registry says so.

Within-athlete uncertainty (``athlete_uncertainty``): each athlete's target statistic is bootstrapped over that athlete's
own windows / events (``uncertainty_boot`` replicates) → per-athlete SE. ``measurement_reliability`` = 1 − mean(SE²) /
within-position variance of the athlete values: the share of between-athlete spread that is not sampling noise (a
complement to the split-half reliability of E16.A3, computed on the same cohort).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from ..status import ValidationFailed
from .targets import TARGETS

REGISTRY_VERSION = "t16_registry_v1"
NAN = float("nan")
NOT_PERFORMANCE = "a constructed NFL movement outcome, not a measure of football performance"


@dataclass(frozen=True)
class TargetDefinition:
    target_id: str
    column: str
    name: str
    physical_quantity: str
    unit: str
    role: str
    source_categories: str
    required_channels: tuple[str, ...]
    eligibility: str
    aggregation: str
    min_observations: str
    truncation: str
    within_athlete_uncertainty: str
    exposure_variable: str | None
    position_applicability: str
    failure_conditions: str
    version: str = REGISTRY_VERSION
    note: str = NOT_PERFORMANCE

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["required_channels"] = list(self.required_channels)
        return d


_D = [
    TargetDefinition("T0", TARGETS["T0"].column, "official movement target", "upper-tail NFL speed over shared speed bins (12C estimator)", "yd/s", "official",
                     "all NFL windows of the 12C freeze", ("speed",), "linked athletes with the frozen value", TARGETS["T0"].aggregation,
                     "as frozen in 12C", "as frozen in 12C", "not rebuildable from the bundle (frozen upstream)", None, "all positions (DB, DL, OL, TE, WR)",
                     "absent column in athletes.parquet → BLOCKED_MISSING_INPUT", note=f"{NOT_PERFORMANCE}; primary target unless it fails G16-A"),
    TargetDefinition("T0x", TARGETS["T0x"].column, "exploratory notebook target", "median over NFL windows of a per-window q90 speed", "yd/s", "exploratory",
                     "exploratory notebook selection (definition outside the repository)", ("speed",), "unknown", TARGETS["T0x"].aggregation, "unknown",
                     "unknown", "not computable here", None, "as exported", "no export → BLOCKED_MISSING_INPUT",
                     note="never compared with T0 on RMSE (different definition, cohort and scale)"),
    TargetDefinition("T1", TARGETS["T1"].column, "high-speed capacity", "upper-tail window maximum speed", "yd/s", "task",
                     "every NFL demand kind; per-kind q90 of window max speed, demand-mix weighted", ("speed",),
                     "windows with a finite max speed; kinds with ≥ min_windows_kind windows", TARGETS["T1"].aggregation, "≥ min_windows windows (athlete)",
                     "windows ending at their maximum are right-censored for the max: share reported (share_windows_end_at_max)",
                     "bootstrap SE over the athlete's windows (unweighted q90 approximation)", "n_windows", "all positions",
                     "< 20 athletes in < 3 positions, |ρ(exposure)| > 0.3 or reliability < 0.7 → G16-A FAIL"),
    TargetDefinition("T2", TARGETS["T2"].column, "acceleration capacity (2–4 yd/s entry)", "positive tangential acceleration at matched speed", "yd/s²", "task",
                     "NFL windows reaching the 2–4 yd/s band", ("speed", "a_tan"), "≥ band_min_frames valid frames in the band with a_tan > 0",
                     TARGETS["T2"].aggregation, "≥ min_events band windows", "windows never reaching the band contribute nothing",
                     "bootstrap SE over the athlete's band windows", "n_band_0", "all positions", "band coverage below the minimum → value null"),
    TargetDefinition("T2b", TARGETS["T2b"].column, "acceleration capacity (4–6 yd/s entry)", "positive tangential acceleration at matched speed", "yd/s²", "task",
                     "NFL windows reaching the 4–6 yd/s band", ("speed", "a_tan"), "as T2 for the 4–6 yd/s band", TARGETS["T2b"].aggregation,
                     "≥ min_events band windows", "as T2", "bootstrap SE over the athlete's band windows", "n_band_1", "positions that reach 4–6 yd/s (OL rarely)",
                     "as T2"),
    TargetDefinition("T3", TARGETS["T3"].column, "braking capability", "maximum controlled deceleration of complete braking events", "yd/s²", "task",
                     "M14 braking events detected on NFL windows", ("speed", "a_tan"), "complete events (entry, braking and exit observed)",
                     TARGETS["T3"].aggregation, "≥ min_events complete events", "incomplete (truncated) events excluded; share_incomplete reported",
                     "bootstrap SE over the athlete's complete events", "n_events_t3", "all positions; entry-speed dependence reported (v_entry)",
                     "high incomplete share or entry-speed confounding (selection) → exploratory only"),
    TargetDefinition("T4", TARGETS["T4"].column, "reacceleration", "mean a_tan over 3 frames from propulsion onset after braking", "yd/s²", "task",
                     "M14 events with an observed propulsion phase", ("speed", "a_tan"), "propulsion observed (not censored)", TARGETS["T4"].aggregation,
                     "≥ min_events events", "propulsion-censored events excluded (share reported); recovery time is censored and handled by KM in M17",
                     "bootstrap SE over the athlete's events", "n_events_t4", "all positions", "censored share > max_censored_share → E16-C2 not run"),
    TargetDefinition("T5", TARGETS["T5"].column, "redirection (speed retention)", "v_min / v_entry over a detected turn", "ratio", "task",
                     "M14 events with a turn direction", ("speed", "a_n", "yaw", "heading_valid"),
                     "turn |angle| ≥ 20° with heading coverage ≥ 0.5 (heading-rate derived; NOT exact geometric curvature)", TARGETS["T5"].aggregation,
                     "≥ min_events turn events", "events without a turn direction excluded", "bootstrap SE over the athlete's turn events", "n_events_t5",
                     "all positions; turning demand differs by position (context share reported)",
                     "low heading coverage or context share > 0.25 without mix adjustment → G16-A FAIL"),
    TargetDefinition("T6", TARGETS["T6"].column, "high-speed consistency", "within-athlete SD of window max speed per demand kind", "yd/s", "task",
                     "every NFL demand kind", ("speed",), "kinds with ≥ min_windows_kind windows", TARGETS["T6"].aggregation, "≥ min_windows windows",
                     "as T1", "bootstrap SE over the athlete's windows", "n_windows", "all positions",
                     "exposure-dependent by construction: |ρ(exposure)| > 0.3 → FAIL"),
]
REGISTRY: dict[str, TargetDefinition] = {d.target_id: d for d in _D}

# statistic used for the within-athlete bootstrap: (source, quantity, filter column or None, statistic)
STAT: dict[str, tuple[str, str, str | None, Callable[[np.ndarray], np.ndarray]]] = {
    "T1": ("windows", "max_speed", None, lambda m: np.quantile(m, 0.9, axis=1)),
    "T2": ("windows", "accel_band_0", None, lambda m: np.quantile(m, 0.9, axis=1)),
    "T2b": ("windows", "accel_band_1", None, lambda m: np.quantile(m, 0.9, axis=1)),
    "T6": ("windows", "max_speed", None, lambda m: np.std(m, axis=1, ddof=1)),
    "T3": ("events", "max_controlled_decel", "complete", lambda m: np.quantile(m, 0.9, axis=1)),
    "T4": ("events", "first_step_explosiveness", "not_propulsion_censored", lambda m: np.median(m, axis=1)),
    "T5": ("events", "turn_speed_retention", "turn", lambda m: np.median(m, axis=1)),
}


def check_consistency() -> None:
    """Every ``TARGETS`` ID has exactly one definition with the same column, unit and role (and vice versa)."""
    a, b = set(TARGETS), set(REGISTRY)
    if a != b:
        raise ValidationFailed(f"target registries disagree: only in TARGETS {sorted(a - b)}, only in REGISTRY {sorted(b - a)}")
    bad = [t for t in a if (TARGETS[t].column, TARGETS[t].unit, TARGETS[t].role) != (REGISTRY[t].column, REGISTRY[t].unit, REGISTRY[t].role)]
    if bad:
        raise ValidationFailed(f"column / unit / role differ between TARGETS and REGISTRY for {bad}")
    cols = [d.column for d in REGISTRY.values()]
    if len(cols) != len(set(cols)):
        raise ValidationFailed("two target IDs share one column")


def registry_hash() -> str:
    return hashlib.sha256(json.dumps([REGISTRY[k].to_dict() for k in sorted(REGISTRY)], sort_keys=True).encode()).hexdigest()[:16]


def definitions_record() -> dict[str, Any]:
    """The JSON written to ``manifests/target_definitions_m16.json`` (version + content hash + every definition)."""
    check_consistency()
    return {"version": REGISTRY_VERSION, "hash": registry_hash(), "targets": {k: REGISTRY[k].to_dict() for k in sorted(REGISTRY)},
            "rule": "targets with different IDs are never compared on RMSE (targets.assert_same_target)"}


def definitions_frame() -> Any:
    import polars as pl

    return pl.DataFrame([{**d.to_dict(), "required_channels": ", ".join(d.required_channels)} for d in REGISTRY.values()], infer_schema_length=None)


def _source_frame(tid: str, windows: Any, events: Any) -> Any:
    import polars as pl

    src, qty, filt, _ = STAT[tid]
    d = windows if src == "windows" else events
    if d is None or qty not in d.columns:
        return None
    if filt == "complete" and "complete" in d.columns:
        d = d.filter(pl.col("complete"))
    elif filt == "not_propulsion_censored" and "propulsion_censored" in d.columns:
        d = d.filter(~pl.col("propulsion_censored"))
    elif filt == "turn" and "turn_dir" in d.columns:
        d = d.filter(pl.col("turn_dir") != 0)
    return d.select(pl.col("nfl_id").cast(pl.Int64), pl.col(qty).cast(pl.Float64).alias("v")).filter(pl.col("v").is_finite())


def athlete_uncertainty(windows: Any, events: Any, positions: Any, n_boot: int, seed: int, min_obs: int = 5) -> Any:
    """Per target × athlete: point value of the bootstrap statistic, its within-athlete bootstrap SE and n observations."""
    import polars as pl

    pos = dict(zip(positions["nfl_id"].cast(pl.Int64).to_list(), positions["position_group"].to_list(), strict=True))
    rng = np.random.default_rng(seed)
    rows = []
    for tid in STAT:
        d = _source_frame(tid, windows, events)
        if d is None or not d.height:
            continue
        stat = STAT[tid][3]
        for (i,), g in d.group_by("nfl_id"):
            v = g["v"].to_numpy()
            if v.size < max(min_obs, 3 if tid == "T6" else 2):
                continue
            boot = stat(v[rng.integers(0, v.size, (n_boot, v.size))])
            rows.append({"target_id": tid, "nfl_id": int(i), "position_group": pos.get(int(i), "?"), "n_obs": int(v.size),
                         "value": float(stat(v[None, :])[0]), "se": float(np.std(boot, ddof=1))})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(
        schema={"target_id": pl.String, "nfl_id": pl.Int64, "position_group": pl.String, "n_obs": pl.Int64, "value": pl.Float64, "se": pl.Float64})


def measurement_reliability(unc: Any) -> Any:
    """Per target: 1 − mean(SE²) / pooled within-position variance of the athlete values (≥ 20 athletes)."""
    import polars as pl

    rows = []
    for (tid,), g in unc.group_by("target_id"):
        v, se, p = g["value"].to_numpy(), g["se"].to_numpy(), g["position_group"].to_numpy()
        ok = np.isfinite(v) & np.isfinite(se)
        v, se, p = v[ok], se[ok], p[ok]
        if v.size < 20:
            rows.append({"target_id": tid, "n_athletes": int(v.size), "within_position_var": NAN, "mean_se2": NAN, "measurement_reliability": NAN})
            continue
        resid = np.concatenate([v[p == k] - v[p == k].mean() for k in np.unique(p) if (p == k).sum() >= 3])
        var = float(np.var(resid, ddof=1)) if resid.size > 2 else NAN
        m2 = float(np.mean(se ** 2))
        rows.append({"target_id": tid, "n_athletes": int(v.size), "within_position_var": var, "mean_se2": m2,
                     "measurement_reliability": 1 - m2 / var if var and np.isfinite(var) and var > 0 else NAN,
                     "median_n_obs": float(np.median(g["n_obs"].to_numpy()))})
    return pl.DataFrame(rows, infer_schema_length=None).sort("target_id")


__all__ = ["NOT_PERFORMANCE", "REGISTRY", "REGISTRY_VERSION", "STAT", "TargetDefinition", "athlete_uncertainty", "check_consistency", "definitions_frame",
           "definitions_record", "measurement_reliability", "registry_hash"]
