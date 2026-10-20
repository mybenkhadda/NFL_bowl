"""Study D — scouting decision research with stopwatch twins (E18.D1).

1. Composite (outcome-free): mean of within-position z-scores of the COMBINE features of the best eligible rung (B7
   signatures, else B3 M14 metrics, else B2 simple statistics), requiring ≥ half of them observed.
2. Twins: ``m14.scouting.stopwatch_twins`` — same position, distance on within-position z of the official tests ≤ caliper,
   greedy one-to-one, no outcome used.
3. Pooled evidence: ``m14.scouting.twin_contrast`` (higher-composite twin − lower) on the NFL outcome: pair bootstrap CI,
   sign test, Spearman of Δcomposite with Δoutcome. Only this pooled contrast carries evidence.
4. Case studies: among pairs closer than the median distance, the largest Δcomposite (outcome never consulted); cards show
   official results, tracking features with within-position percentiles, B1 and best-rung OOF predictions with an 80 %
   position-specific residual interval, and the observed outcome REVEALED AFTER the prediction. Athletes are pseudonymised.
No injury, force, reaction-time or causal-development claim is made; every card lists its limitations.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAN = float("nan")
LIMITATIONS = ("Two athletes from one draft cohort; an illustration, not evidence (only the pooled twin contrast is evidence).",
               "10 Hz Combine tracking without positions, body orientation, opponents or ball; movement context differs between Combine and NFL.",
               "The NFL outcome is a constructed movement quantity, not NFL success; replication on new draft classes is required.",
               "No inference about injury risk, force, reaction time or causal development.")


def composite(data: Any, cols: Sequence[str]) -> Any:
    """nfl_id, position_group, braking_composite (the M14 column name required by ``stopwatch_twins``; any feature family)."""
    import polars as pl

    cols = [c for c in cols if c in data.columns]
    d = data.select("nfl_id", "position_group", *[pl.col(c).cast(pl.Float64) for c in cols])
    zs = []
    for c in cols:
        z = (pl.col(c) - pl.col(c).mean().over("position_group")) / pl.col(c).std().over("position_group")
        zs.append(z.alias(f"z_{c}"))
    d = d.with_columns(zs)
    zc = [f"z_{c}" for c in cols]
    need = max(1, len(cols) // 2 + len(cols) % 2)
    return d.with_columns(pl.mean_horizontal(zc).alias("_m"), pl.sum_horizontal([pl.col(z).is_not_null().cast(pl.Int32) for z in zc]).alias("_n")).with_columns(
        pl.when(pl.col("_n") >= need).then(pl.col("_m")).otherwise(None).alias("braking_composite")).select("nfl_id", "position_group", "braking_composite", *zc)


def twins_and_contrast(data: Any, comp: Any, covs: Sequence[str], caliper: float, outcome: Any, n_boot: int, seed: int) -> tuple[Any, Any, Any]:
    import polars as pl

    from ..m14.scouting import stopwatch_twins, twin_contrast

    covs = [c for c in covs if c in data.columns]
    df = data.select("nfl_id", "position_group", *covs).join(comp.select("nfl_id", "braking_composite"), on="nfl_id", how="inner")
    pairs = stopwatch_twins(df, covs, caliper)
    long, summ = twin_contrast(pairs, comp.select("nfl_id", "braking_composite"), outcome, ["y"], n_boot, seed)
    return pairs, long, summ


def select_cases(long: Any, n: int) -> Any:
    import polars as pl

    if long is None or not long.height:
        return pl.DataFrame()
    med = float(long["distance"].median())
    return long.filter(pl.col("distance") <= med).sort("delta_composite", descending=True).head(n)


def _pct(series: Any, value: float) -> float:
    v = np.asarray(series, float)
    v = v[np.isfinite(v)]
    return float(100 * (v < value).mean() + 50 * (v == value).mean()) if v.size and np.isfinite(value) else NAN


def case_markdown(k: int, case: dict[str, Any], data: Any, comp: Any, wide: Any, covs: Sequence[str], feat_cols: Sequence[str], best: str, salt: str,
                  target_label: str) -> str:
    import polars as pl

    from ..stats import pseudonym

    hi, lo = case["athlete_hi"], case["athlete_lo"]
    pos = case["position_group"]
    peers = data.filter(pl.col("position_group") == pos)
    res = wide.filter(pl.col("position_group") == pos)
    lines = [f"# Case {k}: {pseudonym(hi, salt)} vs {pseudonym(lo, salt)} ({pos})", "",
             f"Selected WITHOUT outcomes: stopwatch-twin distance {case['distance']:.2f} (≤ median), largest composite difference "
             f"({case['delta_composite']:.2f} within-position SD).", "", "## Official Combine results", "", "| measure | " + pseudonym(hi, salt) + " | " + pseudonym(lo, salt) + " |",
             "|---|---|---|"]
    row = {i: data.filter(pl.col("nfl_id") == i) for i in (hi, lo)}
    for c in covs:
        if c in data.columns:
            lines.append(f"| {c} | {row[hi][c][0]} | {row[lo][c][0]} |")
    lines += ["", f"## Tracking features (within-{pos} percentile)", "", "| feature | " + pseudonym(hi, salt) + " | " + pseudonym(lo, salt) + " |", "|---|---|---|"]
    for c in feat_cols:
        if c in data.columns:
            vh, vl = row[hi][c][0], row[lo][c][0]
            lines.append(f"| {c} | {vh if vh is None else f'{vh:.3g}'} (p{_pct(peers[c].to_numpy(), vh if vh is not None else np.nan):.0f}) | "
                         f"{vl if vl is None else f'{vl:.3g}'} (p{_pct(peers[c].to_numpy(), vl if vl is not None else np.nan):.0f}) |")
    lines += ["", f"## Model predictions of {target_label} (out of fold)", ""]
    for m in ("M1", best):
        col = f"pred_{m}"
        if col in res.columns:
            r = (res["y"] - res[col]).to_numpy().astype(float)
            q10, q90 = (float(np.quantile(r, 0.1)), float(np.quantile(r, 0.9))) if r.size >= 10 else (NAN, NAN)
            for i in (hi, lo):
                p = wide.filter(pl.col("nfl_id") == i)
                if p.height:
                    v = float(p[col][0])
                    lines.append(f"- {m} · {pseudonym(i, salt)}: {v:.3f} (80 % position interval {v + q10:.3f} – {v + q90:.3f})")
    lines += ["", "## Observed NFL outcome (revealed after the prediction)", ""]
    for i in (hi, lo):
        p = wide.filter(pl.col("nfl_id") == i)
        lines.append(f"- {pseudonym(i, salt)}: {float(p['y'][0]):.3f}" if p.height else f"- {pseudonym(i, salt)}: not in the evaluated cohort")
    lines += ["", "## Reading", "", "Descriptive comparison of two prospects with similar official testing. Whether the tracking difference matters is answered "
              "only by the pooled twin contrast and the ladder status tables, not by this pair.", "", "## Limitations", ""]
    lines += [f"- {x}" for x in LIMITATIONS]
    return "\n".join(lines)


__all__ = ["LIMITATIONS", "case_markdown", "composite", "select_cases", "twins_and_contrast"]
