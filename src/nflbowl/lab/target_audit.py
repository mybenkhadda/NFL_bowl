"""Part B — the NFL prediction problem: frozen-target provenance, sample eligibility (attrition flow + overlap matrix)
and the position structure of the target.

Provenance facts that cannot be read from the bundle are stated with their source (``code inspection`` of
``nflbdb.research.advanced_movement.m12.distributional_capacity`` at the installed revision); facts read from the
bundle / freeze record are marked as such. The key distinction:

* **prediction validation** is athlete-disjoint (outer folds; every fitted step of the ladder uses training athletes);
* **target construction** is NOT fold-aware: the estimator choice (declared NFL-only reliability rule), the shared speed
  bins (support share in both domains) and — when the frozen estimator is ``C3_pooled`` — the position means and
  between-athlete variances used for shrinkage were computed on ALL athletes once, before any Combine value was joined.
  This is a target-definition choice, not predictor leakage, but it means the end-to-end pipeline is not strictly
  inductive and must not be described as such.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.validation import M1_COLS, TRANSFER_TARGET

NAN = float("nan")
CODE = "code inspection (nflbdb m12.distributional_capacity, m12.config)"


# ====================================================================== B1 provenance
def read_freeze(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def target_provenance(exp: Any, freeze: dict[str, Any] | None = None) -> Any:
    """One row per provenance aspect: finding · evidence source · implication for M13 claims."""
    import polars as pl

    b = exp.bundle
    src = (b.info.get("source") or {}) if isinstance(b.info, dict) else {}
    tf = src.get("target_freeze") or {}
    ath = b.athletes
    y = ath[TRANSFER_TARGET].cast(pl.Float64) if ath is not None and TRANSFER_TARGET in ath.columns else None
    est = (freeze or {}).get("estimator")
    rows = []

    def add(aspect: str, finding: str, evidence: str, implication: str) -> None:
        rows.append({"aspect": aspect, "finding": finding, "evidence": evidence, "implication": implication})

    add("target name / version", f"{TRANSFER_TARGET}; bundle transfer_target = {b.info.get('transfer_target')}; freeze config_hash = {tf.get('config_hash')}, "
        f"written {tf.get('written_at')}, run mode {tf.get('run_mode')}" + (f"; freeze_hash = {freeze.get('freeze_hash')}" if freeze else ""),
        "bundle.json source.target_freeze" + (" + m12c_target_freeze.json" if freeze else ""), "every M13 comparison uses this exact column; no re-estimation")
    add("definition", "athlete trait = mean over the frozen SHARED speed bins (equal bin weight; NaN unless every shared bin is supported) of the "
        "speed-conditioned τ = 0.90 acceleration quantile Q_τ(a | v, athlete), from post-snap NFL frame histograms over ALL the athlete's games",
        CODE + " (athlete_targets, trait)", "a constructed movement-capacity quantity, NOT an NFL success / performance outcome")
    add("estimator", f"frozen estimator = {est or 'not recorded in the bundle (freeze JSON not supplied)'}; candidates C1 empirical, C2 smoothed QR, "
        "C3 position-pooled, C4 shape-constrained", "m12c_target_freeze.json" if freeze else CODE,
        "if C3_pooled: per-athlete values are shrunk toward POSITION means computed on all athletes (global fit)")
    add("units / scaling", "yd/s² (acceleration), unscaled; standardization only inside the ladder design (train-fold statistics)", CODE,
        "RMSE / MAE are reported in yd/s²")
    add("population", f"athletes.parquet rows with linked = True and a finite target: "
        f"{int(ath.filter(pl.col('linked').fill_null(False) & pl.col(TRANSFER_TARGET).cast(pl.Float64).is_finite().fill_null(False)).height) if y is not None else 0}",
        "bundle athletes.parquet", "the ladder population; attrition in B2")
    add("NFL observation requirements", "≥ 8 NFL games per athlete (min_games_athlete); a speed bin counts only with ≥ 40 frames (min_frames_bin_nfl); "
        "shared bins = supported by ≥ 60 % of eligible athletes in BOTH domains", CODE + " (M12Config)",
        "athletes with short NFL exposure are excluded → survivorship toward players who stayed on the field")
    add("missingness", f"{int(y.is_null().sum()) if y is not None else 'n/a'} of {ath.height if ath is not None else 'n/a'} athletes.parquet rows lack the target",
        "bundle athletes.parquet", "missing ≠ random (exposure-driven); B2 / B3 report it by position")
    add("temporal validity", "target uses all NFL games available to 12C (post-draft seasons); predictors are pre-draft Combine data → predictors precede the outcome",
        CODE, "no future predictor information; target windows overlap seasons across athletes (draft-year control in M1)")
    add("estimator-fitting scope", "estimator choice, λ (C2), shared bins and C3 pooling statistics were fitted ONCE on all athletes (NFL data + "
        "domain support counts), before any Combine value was joined; NOT refitted per outer fold", CODE + " (freeze_rule, write_freeze, pooled)",
        "athlete-disjoint PREDICTION validation holds; END-TO-END target construction is not strictly inductive — do not claim it")
    add("position dependence", "position enters the target only through C3 pooling (if frozen) and through exposure; position also enters every ladder model "
        "as unpenalized dummies", CODE, "B3 / Part F quantify how much apparent predictability is position alone")
    return pl.DataFrame(rows)


# ====================================================================== B2 eligibility
def athlete_sets(exp: Any, oof: Any | None = None, encoders: Sequence[str] = ()) -> dict[str, set[int]]:
    """Named athlete sets (nfl_id) for the attrition flow and the overlap matrix."""
    import polars as pl

    b = exp.bundle
    meta = b.meta
    ath = b.athletes
    sets: dict[str, set[int]] = {}
    ids = lambda df: set(int(i) for i in df["nfl_id"].unique().to_list())        # noqa: E731
    sets["all athletes (bundle)"] = ids(meta)
    sets["NFL tracking"] = ids(meta.filter(pl.col("domain") == "nfl"))
    sets["Combine tracking"] = ids(meta.filter(pl.col("domain") == "combine"))
    sets["Combine forty tracking (F0 / embeddings)"] = ids(meta.filter((pl.col("domain") == "combine") & (pl.col("kind") == "forty")))
    sets["fold manifest"] = ids(b.folds)
    if ath is not None:
        sets["athletes.parquet"] = ids(ath)
        sets["linked"] = ids(ath.filter(pl.col("linked").fill_null(False)))
        if TRANSFER_TARGET in ath.columns:
            sets["frozen target"] = ids(ath.filter(pl.col(TRANSFER_TARGET).cast(pl.Float64).is_finite().fill_null(False)))
        m1 = [c for c in M1_COLS if c in ath.columns]
        sets["M1 complete"] = ids(ath.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in m1]))) if m1 else set()
    if oof is not None and oof.height:
        sets["transfer ladder rows"] = set(int(i) for i in oof["nfl_id"].to_list())
        for e in encoders:
            c = f"has_{e}"
            if c in oof.columns:
                sets[f"native {e}"] = set(int(i) for i in oof.filter(pl.col(c))["nfl_id"].to_list())
    return sets


FLOW = ("all athletes (bundle)", "NFL tracking", "linked", "frozen target", "fold manifest", "M1 complete", "Combine forty tracking (F0 / embeddings)")


def attrition_flow(sets: dict[str, set[int]], encoders: Sequence[str] = ()) -> Any:
    """Sequential intersection in the declared order (+ every native encoder set, then the strict common cohort)."""
    import polars as pl

    order = [s for s in FLOW if s in sets] + [f"native {e}" for e in encoders if f"native {e}" in sets]
    rows, cur, prev = [], None, None
    for s in order:
        cur = set(sets[s]) if cur is None else cur & sets[s]
        rows.append({"step": s, "n_remaining": len(cur), "n_removed": 0 if prev is None else prev - len(cur)})
        prev = len(cur)
    return pl.DataFrame(rows)


def overlap_matrix(sets: dict[str, set[int]]) -> Any:
    import polars as pl

    names = list(sets)
    return pl.DataFrame([{"set_a": a, "set_b": b, "n_overlap": len(sets[a] & sets[b]), "share_of_a": len(sets[a] & sets[b]) / max(len(sets[a]), 1)}
                         for a in names for b in names])


# ====================================================================== B3 position structure
def analysis_frame(exp: Any) -> Any:
    """The athletes of the transfer ladder (same filters as ``Experiment._ladder_inputs``): nfl_id, position_group, fold, y, M1."""
    import polars as pl

    b = exp.bundle
    return (b.athletes.filter(pl.col("linked").fill_null(False)).filter(pl.col(TRANSFER_TARGET).cast(pl.Float64).is_finite().fill_null(False))
            .join(b.folds.select("nfl_id", "fold"), on="nfl_id", how="inner").sort("nfl_id")
            .select("nfl_id", pl.col("position_group").fill_null("?"), "fold", pl.col(TRANSFER_TARGET).cast(pl.Float64).alias("y"),
                    *[c for c in M1_COLS if c in b.athletes.columns]))


def variance_decomposition(y: np.ndarray, pos: np.ndarray) -> dict[str, float]:
    """Total = between + within (population variances, ddof 0); η² = between / total (in-sample, descriptive)."""
    y = np.asarray(y, float)
    pos = np.asarray(pos)
    ok = np.isfinite(y)
    y, pos = y[ok], pos[ok]
    if y.size < 3:
        return {"n": int(y.size), "total_var": NAN, "between_var": NAN, "within_var": NAN, "eta2": NAN}
    mu = y.mean()
    between = within = 0.0
    for p in np.unique(pos):
        v = y[pos == p]
        between += v.size * (v.mean() - mu) ** 2
        within += ((v - v.mean()) ** 2).sum()
    tot = float(((y - mu) ** 2).sum())
    return {"n": int(y.size), "n_positions": int(np.unique(pos).size), "total_var": tot / y.size, "between_var": between / y.size, "within_var": within / y.size,
            "eta2": between / tot if tot > 0 else NAN}


def position_table(frame: Any, exp: Any, min_n: int = 10) -> Any:
    """Per position: n, mean, SD, median, q05 / q95, within-position |z| > 3 count, Tukey-fence outliers, share of linked
    athletes of that position without a target, and a small-n flag."""
    import polars as pl

    b = exp.bundle
    linked = b.athletes.filter(pl.col("linked").fill_null(False))
    miss = linked.group_by(pl.col("position_group").fill_null("?")).agg(pl.len().alias("n_linked"),
                                                                         pl.col(TRANSFER_TARGET).cast(pl.Float64).is_finite().not_().fill_null(True).sum().alias("n_missing_target"))
    rows = []
    for p in sorted(frame["position_group"].unique().to_list(), key=str):
        v = frame.filter(pl.col("position_group") == p)["y"].to_numpy().astype(float)
        sd = float(v.std(ddof=1)) if v.size > 1 else NAN
        z = (v - v.mean()) / sd if v.size > 1 and sd > 0 else np.zeros(v.size)
        q1, q3 = (np.quantile(v, [0.25, 0.75]) if v.size else (NAN, NAN))
        iqr = q3 - q1
        rows.append({"position_group": p, "n": int(v.size), "mean": float(v.mean()) if v.size else NAN, "sd": sd, "median": float(np.median(v)) if v.size else NAN,
                     "q05": float(np.quantile(v, 0.05)) if v.size else NAN, "q95": float(np.quantile(v, 0.95)) if v.size else NAN,
                     "n_abs_z_gt3": int((np.abs(z) > 3).sum()), "n_tukey_outliers": int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()) if v.size else 0,
                     "small_group": v.size < min_n})
    out = pl.DataFrame(rows)
    return out.join(miss, on="position_group", how="left").with_columns((pl.col("n_missing_target") / pl.col("n_linked")).alias("share_missing_target"))


def outliers(frame: Any, z: float = 3.0) -> Any:
    import polars as pl

    return (frame.with_columns(((pl.col("y") - pl.col("y").mean().over("position_group")) / pl.col("y").std().over("position_group")).alias("z_within_position"))
            .filter(pl.col("z_within_position").abs() > z).select("nfl_id", "position_group", "fold", "y", "z_within_position"))


__all__ = ["FLOW", "analysis_frame", "athlete_sets", "attrition_flow", "outliers", "overlap_matrix", "position_table", "read_freeze", "target_provenance",
           "variance_decomposition"]
