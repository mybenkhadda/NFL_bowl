"""11I — novel metric tournament and research synthesis.

Every track writes ``m11_metric_evidence_<track>`` rows in ONE schema (``evidence_row``). The tournament joins them with
the declared candidate registry (``CANDIDATES``) and evaluates seven levels SEPARATELY — a level is ``pass``, ``fail`` or
``not_evaluated``; nothing is promoted automatically, and the *evidence tier* is the number of CONSECUTIVE passes from
level 1. A candidate without data stays in the registry as ``not_evaluated`` (unsupported).

Level rules (declared before any Milestone 11 result; thresholds are heuristics stated with the result):
    1 measurable              ≥ ``min_athletes_cell`` athletes with a finite value
    2 repeatable              ICC lower bound ≥ 0.5 (or ICC ≥ 0.6 when no interval); NFL-only metrics: split-half ≥ 0.5
    3 nonredundant            within-position R² from official tests ≤ 0.8
    4 physically transferable cross-domain Spearman interval (matched NFL quantity) excludes 0 in the expected direction
    5 football associated     11H association / held-out interval excludes 0 for a football action
    6 predictively validated  held-out Δ RMSE interval > 0 AND above every negative control (11A / 11B / 11C / 11H)
    7 operationally useful    needs an INDEPENDENT cohort confirmation — not evaluable in this milestone (always not_evaluated)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from .config import EVIDENCE_LEVELS, M11Config

NAN = float("nan")
EVIDENCE_SCHEMA = {"metric_id": pl.String, "track": pl.String, "position_group": pl.String, "n_athletes": pl.Int64, "icc": pl.Float64, "icc_lo": pl.Float64,
                   "nfl_reliability": pl.Float64, "stopwatch_r2": pl.Float64, "filter_sensitivity_rho": pl.Float64, "nfl_comparison": pl.String,
                   "cross_domain_rho": pl.Float64, "cross_domain_lo": pl.Float64, "cross_domain_hi": pl.Float64, "expected_sign": pl.Int64,
                   "football_lo": pl.Float64, "football_hi": pl.Float64, "delta_rmse": pl.Float64, "delta_lo": pl.Float64, "delta_hi": pl.Float64,
                   "beats_controls": pl.Boolean, "execution_mode": pl.String, "config_hash": pl.String, "note": pl.String}


def evidence_row(metric_id: str, track: str, **kw: Any) -> dict[str, Any]:
    """One evidence row with every column present (missing = null). Values must come from executed tables."""
    row = {k: None for k in EVIDENCE_SCHEMA}
    row.update({"metric_id": metric_id, "track": track, "position_group": "ALL", **kw})
    return row


def evidence_frame(rows: Sequence[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=EVIDENCE_SCHEMA)
    clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items() if k in EVIDENCE_SCHEMA} for r in rows]
    return pl.DataFrame(clean, schema=EVIDENCE_SCHEMA)


@dataclass(frozen=True)
class Candidate:
    metric_id: str
    family: str
    definition: str
    unit: str
    source_drill: str
    nfl_comparison: str
    interpretability: str          # high · medium · low
    football_relevance: str        # stated rationale


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("env.lin_A0", "acceleration envelope", "intercept of the linear upper envelope of game a(v)", "yd/s²", "— (NFL)", "forty.phase_A0", "high",
              "how hard an athlete accelerates from low speed in games (observed, not a ceiling)"),
    Candidate("env.lin_V0", "acceleration envelope", "speed intercept of the linear upper envelope", "yd/s", "— (NFL)", "forty.phase_V0", "high", "speed range of game acceleration"),
    Candidate("forty.phase_A0", "sprint physics", "A0 = Vmax/τ of the phase-plane fit on the rising forty", "yd/s²", "forty", "env.lin_A0", "high", "initial acceleration capacity in a sprint"),
    Candidate("forty.phase_V0", "sprint physics", "speed intercept of the phase-plane fit", "yd/s", "forty", "env.lin_V0", "high", "speed capacity"),
    Candidate("forty.sprint_tau", "sprint physics", "time constant τ of v(t) = Vmax(1 − e^(−t/τ))", "s", "forty", "—", "medium", "how quickly speed is built"),
    Candidate("forty.band_2_4_mean_accel", "speed-band acceleration", "(4 − 2) / time between first crossings", "yd/s²", "forty", "nfl.band_2_4", "high", "early acceleration"),
    Candidate("forty.band_4_6_mean_accel", "speed-band acceleration", "(6 − 4) / time between first crossings", "yd/s²", "forty", "nfl.band_4_6", "high", "mid acceleration"),
    Candidate("forty.band_6_8_mean_accel", "speed-band acceleration", "(8 − 6) / time between first crossings", "yd/s²", "forty", "nfl.band_6_8", "high", "late acceleration"),
    Candidate("forty.phase_compensation", "sprint phase compensation", "within-position z(late band) − z(early band)", "z", "forty", "nfl.band_6_8", "medium",
              "a late builder vs an early builder at the same forty"),
    Candidate("short_shuttle.braking_duration_s", "braking", "time with a_tan ≤ −1 yd/s² in a braking run", "s", "short shuttle", "—", "high", "time spent stopping"),
    Candidate("short_shuttle.braking_distance_yd", "braking", "path length of the braking run", "yd", "short shuttle", "nfl.brake_distance_matched", "high", "stopping distance"),
    Candidate("short_shuttle.decel_mean", "braking", "mean −a_tan over the braking run", "yd/s²", "short shuttle", "nfl.brake_decel", "high", "braking intensity"),
    Candidate("short_shuttle.transition_cost_index", "Transition Cost Index", "(v_entry − v_exit)/v_entry, exit 0.5 s after the minimum", "fraction", "short shuttle",
              "nfl.pair_tci", "high", "speed lost through a change of direction"),
    Candidate("three_cone.transition_cost_index", "Transition Cost Index", "same rule, three-cone turns", "fraction", "three-cone", "nfl.redirect_retention_90", "high",
              "speed lost through a turn"),
    Candidate("short_shuttle.brake_to_burst_s", "Brake-to-Burst Latency", "braking end → first a_tan ≥ 1.5 yd/s²", "s", "short shuttle", "nfl.brake_to_burst", "high",
              "how fast an athlete re-starts after stopping"),
    Candidate("short_shuttle.speed_recovery_time_s", "Speed Recovery Time", "minimum speed → 80 % of entry speed", "s", "short shuttle", "—", "high", "recovery of speed"),
    Candidate("short_shuttle.lat_accel_peak", "lateral acceleration", "peak |(vx·ay − vy·ax)/|v|| through the transition", "yd/s²", "short shuttle", "—", "medium",
              "centripetal demand of a turn (Combine only)"),
    Candidate("geom.asymmetry_abs", "athlete asymmetry", "|TCI(ccw) − TCI(cw)| with comparable entry speeds", "fraction", "shuttle / three-cone", "—", "medium",
              "side preference (frame handedness unverified: magnitude only)"),
    Candidate("derived.transfer_efficiency_A0", "Athletic Transfer Efficiency", "NFL envelope A0 / Combine sprint A0", "ratio", "forty + NFL", "—", "medium",
              "share of the measured sprint acceleration expressed in games (post-draft descriptor; NOT pre-draft)"),
    Candidate("geom.adaptability_slope", "Movement Adaptability", "slope of TCI against |turn angle| across an athlete's turns", "1/deg", "shuttle / three-cone", "—", "low",
              "how fast speed cost grows with turn sharpness"),
    Candidate("nfl.movement_diversity", "Movement Vocabulary", "Shannon entropy of the athlete's game demand mix (per season)", "nats", "— (NFL)", "—", "medium",
              "breadth of game movement (NFL descriptor; not a Combine metric)"),
    Candidate("emb.shared_encoder", "shared encoder embeddings", "pooled self-supervised embedding (best 11F configuration)", "—", "all drills + NFL", "cross-domain retrieval",
              "low", "a mathematical similarity; no physical unit"),
    Candidate("family.drill_information", "drill-specific incremental information", "11G marginal Δ RMSE of a drill family (3 training-fold PCs)", "yd/s² (target units)",
              "per drill", "11G targets", "medium", "what one more drill adds beyond official tests"),
)


def _level_pass(level: int, r: dict[str, Any], c: M11Config) -> str:
    f = lambda k: r.get(k) if r.get(k) is not None and np.isfinite(r.get(k)) else None
    if level == 1:
        n = r.get("n_athletes")
        return "not_evaluated" if n is None else ("pass" if n >= c.min_athletes_cell else "fail")
    if level == 2:
        if f("icc_lo") is not None:
            return "pass" if f("icc_lo") >= 0.5 else "fail"
        if f("icc") is not None:
            return "pass" if f("icc") >= 0.6 else "fail"
        if f("nfl_reliability") is not None:
            return "pass" if f("nfl_reliability") >= 0.5 else "fail"
        return "not_evaluated"
    if level == 3:
        return "not_evaluated" if f("stopwatch_r2") is None else ("pass" if f("stopwatch_r2") <= 0.8 else "fail")
    if level == 4:
        lo, hi, s = f("cross_domain_lo"), f("cross_domain_hi"), r.get("expected_sign") or 1
        if lo is None or hi is None:
            return "not_evaluated"
        return "pass" if (s > 0 and lo > 0) or (s < 0 and hi < 0) else "fail"
    if level == 5:
        lo, hi = f("football_lo"), f("football_hi")
        return "not_evaluated" if lo is None else ("pass" if lo > 0 or hi < 0 else "fail")
    if level == 6:
        lo = f("delta_lo")
        if lo is None:
            return "not_evaluated"
        return "pass" if lo > 0 and bool(r.get("beats_controls")) else "fail"
    return "not_evaluated"


def tournament(evidence: pl.DataFrame, c: M11Config) -> pl.DataFrame:
    """One row per candidate (and per position when the evidence is position-specific): level states, evidence tier."""
    ev_by = {}
    for r in evidence.iter_rows(named=True):
        key = (r["metric_id"], r["position_group"] or "ALL")
        cur = ev_by.setdefault(key, {})
        for k, v in r.items():                                   # merge rows of several tracks: keep the first non-null value
            if cur.get(k) is None and v is not None:
                cur[k] = v
    keys = sorted(set(ev_by) | {(cd.metric_id, "ALL") for cd in CANDIDATES})
    cand = {cd.metric_id: cd for cd in CANDIDATES}
    rows = []
    for mid, pos in keys:
        r = ev_by.get((mid, pos), {"metric_id": mid})
        levels = {f"L{i}_{name}": _level_pass(i, r, c) for i, name in EVIDENCE_LEVELS}
        tier = 0
        for i, name in EVIDENCE_LEVELS:
            if levels[f"L{i}_{name}"] == "pass":
                tier = i
            else:
                break
        cd = cand.get(mid)
        rows.append({"metric_id": mid, "position_group": pos, "family": cd.family if cd else "track-specific", "definition": cd.definition if cd else "",
                     "unit": cd.unit if cd else "", "source_drill": cd.source_drill if cd else "", "nfl_comparison": cd.nfl_comparison if cd else (r.get("nfl_comparison") or ""),
                     "interpretability": cd.interpretability if cd else "", "football_relevance": cd.football_relevance if cd else "",
                     "n_athletes": r.get("n_athletes"), "icc": r.get("icc"), "stopwatch_r2": r.get("stopwatch_r2"), "cross_domain_rho": r.get("cross_domain_rho"),
                     "delta_rmse": r.get("delta_rmse"), "delta_lo": r.get("delta_lo"), "delta_hi": r.get("delta_hi"), **levels, "evidence_tier": tier,
                     "has_evidence": (mid, pos) in ev_by, "tracks": r.get("track") or ""})
    return pl.DataFrame(rows, infer_schema_length=None)


def synthesis(t: pl.DataFrame, matrix: pl.DataFrame | None, drill_marginal: pl.DataFrame | None) -> dict[str, list[str]]:
    """Answers to the final questions, assembled ONLY from executed tables (empty answer = nothing qualifies)."""
    a = t.filter(pl.col("position_group") == "ALL")
    names = lambda d: [f"{r['metric_id']} (n={r['n_athletes']})" for r in d.iter_rows(named=True)]
    out = {
        "reliable": names(a.filter(pl.col("L2_repeatable") == "pass")),
        "distinct_from_stopwatch": names(a.filter((pl.col("L2_repeatable") == "pass") & (pl.col("L3_nonredundant") == "pass"))),
        "generalise_combine_to_nfl": names(a.filter(pl.col("L4_physically_transferable") == "pass")),
        "position_specific": names(t.filter((pl.col("position_group") != "ALL") & (pl.col("L4_physically_transferable") == "pass"))),
        "predictively_validated": names(a.filter(pl.col("L6_predictively_validated") == "pass")),
        "failed": names(a.filter((pl.col("L4_physically_transferable") == "fail") | (pl.col("L6_predictively_validated") == "fail"))),
        "unsupported_no_evidence": [r["metric_id"] for r in a.filter(~pl.col("has_evidence")).iter_rows(named=True)],
    }
    if drill_marginal is not None and drill_marginal.height:
        best = drill_marginal.filter(pl.col("family") != "_random").sort("delta_rmse", descending=True, nulls_last=True).head(3)
        out["most_useful_drill_additions"] = [f"{r['family']} → {r['target']}: Δ RMSE {r['delta_rmse']:+.4f} [{r['delta_lo']}, {r['delta_hi']}] ({r['verdict']})"
                                              for r in best.iter_rows(named=True) if r["delta_rmse"] is not None]
    if matrix is not None and matrix.height and "matrix_class" in matrix.columns:
        out["matrix_classes"] = [f"{r['matrix_class']}: {r['len']}" for r in matrix.group_by("matrix_class").len().sort("len", descending=True).iter_rows(named=True)]
    return out


def candidate_table() -> pl.DataFrame:
    return pl.DataFrame([asdict(c) for c in CANDIDATES])


__all__ = ["CANDIDATES", "Candidate", "EVIDENCE_SCHEMA", "candidate_table", "evidence_frame", "evidence_row", "synthesis", "tournament"]
