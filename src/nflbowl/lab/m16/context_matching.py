"""E16.A8 — movement-context common support between each Combine drill and NFL windows, and context-matched NFL outcomes.

Context covariates are label-free kinematics of each window (``m15.signals`` features): entry speed (first 1 s), peak
tangential acceleration, valid duration, braking share and peak |a_n| (turning demand from normal acceleration — heading-
rate information, NOT geometric curvature). Position is an exact stratum.

Common support per drill (``support_table``):
* univariate: overlap coefficient OVL (shared histogram mass on pooled quantile bins) and the standardised mean difference;
* multivariate: coarsened strata (pooled quantile bins of every covariate × position). ``support_combine`` = share of the
  drill's Combine windows in strata that also hold NFL windows; ``support_nfl`` = share of NFL windows in strata that hold
  drill windows; athletes with supported windows in BOTH domains.
* status ``COMPLETED`` when support_combine ≥ ``min_support_share`` and ≥ ``min_support_athletes`` athletes have supported
  windows in both domains; otherwise ``BLOCKED_INSUFFICIENT_OVERLAP``. A blocked drill is never matched by extrapolation.

Context-matched outcomes (``matched_outcomes``): for a supported drill, an athlete's NFL outcome recomputed from the NFL
windows (and the M14 events inside them) that fall in strata supported by that drill — e.g. high speed after a forty-like
entry, redirection retention at three-cone-like turning demand. Column names start with ``t16_ctx_`` so they can never
enter a predictor list (``baselines.assert_no_nfl_predictors``). Binning uses all athletes' windows (outcome-free, no
target value is read) — stated, as for the demand-mix weights of T1.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..status import BLOCKED_INSUFFICIENT_OVERLAP, COMPLETED

NAN = float("nan")
DRILLS = ("forty", "three_cone", "short_shuttle")
# drill → (outcome id, source, quantity, filter, statistic, unit)
MATCHED = {"forty": [("T1", "windows", "max_speed", None, "q90", "yd/s"), ("T2", "windows", "accel_band_0", None, "q90", "yd/s²")],
           "three_cone": [("T5", "events", "turn_speed_retention", "turn", "median", "ratio"), ("T3", "events", "max_controlled_decel", "complete", "q90", "yd/s²")],
           "short_shuttle": [("T3", "events", "max_controlled_decel", "complete", "q90", "yd/s²"),
                             ("T4", "events", "first_step_explosiveness", "not_propulsion_censored", "median", "yd/s²")]}


def ovl(a: np.ndarray, b: np.ndarray, k: int = 20) -> float:
    """Overlap coefficient of two samples on pooled quantile bins (1 = identical histograms, 0 = disjoint)."""
    from ..domain_matching import quantile_bins

    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return NAN
    codes = quantile_bins(np.concatenate([a, b]), k)
    m = int(codes.max()) + 1
    ha = np.bincount(codes[: a.size], minlength=m) / a.size
    hb = np.bincount(codes[a.size:], minlength=m) / b.size
    return float(np.minimum(ha, hb).sum())


def strata(frame: Any, covariates: Sequence[str], bins: int) -> np.ndarray:
    """Pooled quantile strata of the covariates × exact position (−1 when a covariate is missing)."""
    import polars as pl

    from ..domain_matching import strata_codes

    cov = {c: frame[c].cast(pl.Float64).fill_null(NAN).to_numpy() for c in covariates}
    code = strata_codes(cov, tuple(covariates), bins)
    pos = frame["position_group"].fill_null("?").to_numpy()
    levels = {p: k for k, p in enumerate(sorted(set(pos.tolist())))}
    pc = np.array([levels[p] for p in pos], np.int64)
    return np.where(code >= 0, code * (len(levels) + 1) + pc, -1)


def drill_frame(feat: Any, drill: str, covariates: Sequence[str]) -> Any:
    """Drill windows + NFL windows of the same positions, with the covariates (rows with any missing covariate dropped)."""
    import polars as pl

    comb = feat.filter((pl.col("domain") == "combine") & (pl.col("kind") == drill))
    pos = comb["position_group"].drop_nulls().unique().to_list()
    nfl = feat.filter((pl.col("domain") == "nfl") & pl.col("position_group").is_in(pos))
    cols = ["idx", "nfl_id", "domain", "kind", "position_group", *covariates]
    d = pl.concat([comb.select(cols), nfl.select(cols)], how="vertical_relaxed")
    return d.filter(pl.all_horizontal([pl.col(c).cast(pl.Float64).is_finite() for c in covariates]))


def support_table(feat: Any, covariates: Sequence[str], bins: int, min_share: float, min_athletes: int,
                  drills: Sequence[str] = DRILLS) -> tuple[Any, Any, dict[str, np.ndarray]]:
    """(per drill × covariate univariate overlap, per drill multivariate support + status, {drill: supported NFL idx})."""
    import polars as pl

    from ..domain_matching import smd

    uni, multi, keep = [], [], {}
    for dr in drills:
        d = drill_frame(feat, dr, covariates)
        dom = d["domain"].to_numpy()
        n_c, n_n = int((dom == "combine").sum()), int((dom == "nfl").sum())
        if n_c < 10 or n_n < 10:
            multi.append({"drill": dr, "n_combine_windows": n_c, "n_nfl_windows": n_n, "status": BLOCKED_INSUFFICIENT_OVERLAP,
                          "reason": "fewer than 10 windows in a domain"})
            continue
        for c in covariates:
            x = d[c].cast(pl.Float64).to_numpy()
            uni.append({"drill": dr, "covariate": c, "ovl": ovl(x[dom == "combine"], x[dom == "nfl"]), "smd_nfl_minus_combine": smd(x, dom),
                        "combine_median": float(np.median(x[dom == "combine"])), "nfl_median": float(np.median(x[dom == "nfl"])),
                        "combine_q05": float(np.quantile(x[dom == "combine"], 0.05)), "combine_q95": float(np.quantile(x[dom == "combine"], 0.95)),
                        "nfl_q05": float(np.quantile(x[dom == "nfl"], 0.05)), "nfl_q95": float(np.quantile(x[dom == "nfl"], 0.95))})
        s = strata(d, covariates, bins)
        sc = set(s[(dom == "combine") & (s >= 0)].tolist())
        sn = set(s[(dom == "nfl") & (s >= 0)].tolist())
        common = sc & sn
        in_c = np.isin(s, list(common)) & (dom == "combine")
        in_n = np.isin(s, list(common)) & (dom == "nfl")
        ids = d["nfl_id"].to_numpy()
        both = set(ids[in_c].tolist()) & set(ids[in_n].tolist())
        share_c = float(in_c.sum() / max(n_c, 1))
        ok = share_c >= min_share and len(both) >= min_athletes
        keep[dr] = d["idx"].to_numpy()[in_n]
        multi.append({"drill": dr, "covariates": ", ".join(covariates), "bins": bins, "n_combine_windows": n_c, "n_nfl_windows": n_n,
                      "n_strata_combine": len(sc), "n_strata_common": len(common), "support_combine": share_c, "support_nfl": float(in_n.sum() / max(n_n, 1)),
                      "n_athletes_combine_supported": len(set(ids[in_c].tolist())), "n_athletes_nfl_supported": len(set(ids[in_n].tolist())),
                      "n_athletes_both_supported": len(both), "status": COMPLETED if ok else BLOCKED_INSUFFICIENT_OVERLAP,
                      "reason": "" if ok else f"support_combine {share_c:.2f} (need ≥ {min_share}) / athletes in both {len(both)} (need ≥ {min_athletes})"})
    return pl.DataFrame(uni, infer_schema_length=None), pl.DataFrame(multi, infer_schema_length=None), keep


def matched_outcomes(windows: Any, events: Any, keep: dict[str, np.ndarray], support: Any, min_obs: int) -> tuple[Any, Any]:
    """(wide per-athlete matched outcomes ``t16_ctx_<drill>__<id>``, long per drill × outcome summary). Only drills whose
    support status is COMPLETED produce values; the others are listed with their status."""
    import polars as pl

    st = {r["drill"]: r["status"] for r in support.iter_rows(named=True)} if support is not None and support.height else {}
    wide, summ = None, []
    for dr, specs in MATCHED.items():
        for tid, src, qty, filt, stat, unit in specs:
            col = f"t16_ctx_{dr}__{tid}"
            if st.get(dr) != COMPLETED:
                summ.append({"drill": dr, "outcome": col, "target_id": tid, "unit": unit, "status": st.get(dr, "NOT_RUN"), "n_athletes": 0})
                continue
            base = windows if src == "windows" else events
            if base is None or qty not in base.columns:
                summ.append({"drill": dr, "outcome": col, "target_id": tid, "unit": unit, "status": "BLOCKED_MISSING_INPUT", "n_athletes": 0})
                continue
            d = base.filter(pl.col("idx").is_in(keep.get(dr, np.zeros(0, np.int64)).tolist()))
            if filt == "complete" and "complete" in d.columns:
                d = d.filter(pl.col("complete"))
            elif filt == "not_propulsion_censored" and "propulsion_censored" in d.columns:
                d = d.filter(~pl.col("propulsion_censored"))
            elif filt == "turn" and "turn_dir" in d.columns:
                d = d.filter(pl.col("turn_dir") != 0)
            d = d.select(pl.col("nfl_id").cast(pl.Int64), pl.col(qty).cast(pl.Float64).alias("v")).filter(pl.col("v").is_finite())
            agg = pl.col("v").quantile(0.9) if stat == "q90" else pl.col("v").median()
            a = d.group_by("nfl_id").agg(agg.alias(col), pl.len().alias(f"n_{col}"))
            a = a.with_columns(pl.when(pl.col(f"n_{col}") >= min_obs).then(pl.col(col)).otherwise(None).alias(col))
            wide = a if wide is None else wide.join(a, on="nfl_id", how="full", coalesce=True)
            summ.append({"drill": dr, "outcome": col, "target_id": tid, "unit": unit, "status": COMPLETED, "n_athletes": int(a[col].is_not_null().sum()),
                         "n_windows_or_events": d.height})
    return (wide if wide is not None else pl.DataFrame(schema={"nfl_id": pl.Int64})), pl.DataFrame(summ, infer_schema_length=None)


def matched_vs_global(wide: Any, targets: Any, positions: Any) -> Any:
    """Within-position Spearman between each matched outcome and its global target (how much context changes the ranking)."""
    import polars as pl

    from .target_audit import _pos_resid, _spearman
    from .targets import TARGETS

    if wide is None or not wide.height:
        return pl.DataFrame()
    d = wide.join(targets.select(pl.col("nfl_id").cast(pl.Int64), "position_group", *[s.column for s in TARGETS.values() if s.column in targets.columns]),
                  on="nfl_id", how="left")
    pos = d["position_group"].fill_null("?").to_numpy()
    rows = []
    for c in [c for c in wide.columns if c.startswith("t16_ctx_")]:
        tid = c.split("__")[-1]
        g = TARGETS[tid].column
        if g not in d.columns:
            continue
        a, b = d[c].cast(pl.Float64).fill_null(NAN).to_numpy(), d[g].cast(pl.Float64).fill_null(NAN).to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        rows.append({"outcome": c, "global_target": tid, "n_common": int(ok.sum()), "spearman": _spearman(a, b),
                     "spearman_within_position": _spearman(_pos_resid(a, pos), _pos_resid(b, pos))})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["DRILLS", "MATCHED", "drill_frame", "matched_outcomes", "matched_vs_global", "ovl", "strata", "support_table"]
