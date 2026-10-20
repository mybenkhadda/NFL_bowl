"""E16.A4 / E16.A5 — baseline ladder per NFL target on one fold system and one cohort.

Models (ridge rungs through the OFFICIAL engine ``ae.oof_ladder``: train-fold design, inner-CV α, position dummies):

    MEAN              training-fold mean
    POS               position dummies only (``ae.Model([])``)
    M1                official Combine M1 (``M1_COLS``)
    M1+F0             M1 + the 17 engineered forty statistics (F0, ``rep.athlete_embedding_table``)
    M1+simple         M1 + simple tracking statistics from M15 E15.A1 (athlete medians per Combine scope), kept per fold
                      when train coverage ≥ ``min_coverage`` (``m14.transfer.coverage_factory``)
    M1+M14_reliable   M1 + M14 Combine metrics whose M14 measurement status is ``validated`` (read from the M14 run;
                      the status was computed outcome-free on all athletes — stated), same coverage rule
    M1_enet / M1_lightgbm / M1_catboost, M1+simple_enet …   alternative learners (``learners.oof_generic``) on the same rows

Claims use the strict cohort (every model finite). Paired fold-stratified bootstrap (``transfer_fairness``), Holm over the
pre-declared primary family, BH otherwise; ``m14.transfer.transfer_status`` gives validated / exploratory / inconclusive /
invalid; ``precise_null`` applies the M13 H-D rule (every CI includes 0 and every half-width ≤ 2 % of the reference RMSE).
No column derived from NFL data (``t16_*``, ``m14n__*``, ``nfl.*``) ever enters a predictor list (``assert_no_nfl_predictors``).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...data.validation import M1_COLS
from ..status import BLOCKED_MISSING_INPUT, ResearchBlocked, ValidationFailed
from .targets import TARGETS

NAN = float("nan")
NFL_PREFIXES = ("t16_", "m14n__", "nfl.", "nfl_", "n_windows", "n_events", "n_band", "share_")
SIMPLE_SCOPES = {"forty": ("forty",), "agility": ("three_cone", "short_shuttle", "long_shuttle")}
SIMPLE_FEATURES = ("max_speed", "q90_speed", "max_atan", "min_atan", "braking_frac", "accel_frac", "jerk_rms", "max_abs_an", "entry_speed")
PRIMARY_T0 = (("M1", "POS"), ("M1+simple", "M1"), ("M1+M14_reliable", "M1"))


def assert_no_nfl_predictors(cols: Sequence[str]) -> None:
    bad = [c for c in cols if str(c).startswith(NFL_PREFIXES)]
    if bad:
        raise ValidationFailed(f"NFL-derived columns in a predictor set: {bad}", {"columns": bad})


def simple_tracking_table(feat: Any) -> Any:
    """Athlete medians of the E15.A1 features per Combine scope: ``trk_<scope>__<feature>``."""
    import polars as pl

    out = None
    for scope, kinds in SIMPLE_SCOPES.items():
        g = feat.filter((pl.col("domain") == "combine") & pl.col("kind").is_in(list(kinds))).group_by("nfl_id").agg(
            [pl.col(c).median().alias(f"trk_{scope}__{c}") for c in SIMPLE_FEATURES if c in feat.columns])
        out = g if out is None else out.join(g, on="nfl_id", how="full", coalesce=True)
    return out.with_columns(pl.col("nfl_id").cast(pl.Int64))


def m14_reliable_table(m14_run_dir: str, statuses: Sequence[str] = ("validated",)) -> tuple[Any, list[str], str]:
    """(M14 athlete features restricted to Combine predictor columns, the columns whose measurement status is in
    ``statuses``, note). Missing M14 outputs → BLOCKED_MISSING_INPUT naming the files."""
    from pathlib import Path

    import polars as pl

    from ..m14 import registry as RG

    root = Path(m14_run_dir)
    fa, ms = root / "features" / "athlete" / "athlete_features.parquet", root / "tables" / "reliability" / "measurement_status.parquet"
    missing = [str(p) for p in (fa, ms) if not p.is_file()]
    if missing:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "M14 athlete features / measurement status not found (run M14 first)", missing=missing)
    ath = pl.read_parquet(fa)
    st = pl.read_parquet(ms)
    col_key = "column" if "column" in st.columns else ("feature" if "feature" in st.columns else None)
    stat_key = "status" if "status" in st.columns else ("measurement_status" if "measurement_status" in st.columns else None)
    if col_key is None or stat_key is None:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"unrecognised measurement-status schema {st.columns}", missing=[str(ms)])
    declared = set(RG.predictor_columns())
    cols = [c for c in st.filter(pl.col(stat_key).is_in(list(statuses)))[col_key].to_list() if c in ath.columns and c in declared]
    keep = ["nfl_id", *[c for c in ath.columns if c in declared]]
    return ath.select(keep).with_columns(pl.col("nfl_id").cast(pl.Int64)), cols, f"{len(cols)} column(s) with status {list(statuses)}"


def analysis_frame(exp: Any, targets: Any, simple: Any | None, m14: Any | None) -> Any:
    """Linked athletes × (M1, position, fold) × F0 × simple tracking × M14 × every target column."""
    import polars as pl

    from ... import _backend

    b = exp.bundle
    if b.athletes is None:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "athletes.parquet absent from the bundle", missing=["athletes.parquet"])
    rep = _backend.rep()
    e0a = rep.athlete_embedding_table(exp.features(), b.meta, "combine", ["forty"], prefix="eng").with_columns(pl.col("nfl_id").cast(pl.Int64))
    fold_of = exp.folds_by_athlete
    a = b.athletes.filter(pl.col("linked").fill_null(False)).select(pl.col("nfl_id").cast(pl.Int64), "position_group", "linked",
                                                                    *[c for c in M1_COLS if c in b.athletes.columns])
    a = a.with_columns(pl.col("nfl_id").replace_strict(fold_of, default=None, return_dtype=pl.Int64).alias("fold")).filter(pl.col("fold").is_not_null())
    tcols = [s.column for s in TARGETS.values() if s.column in targets.columns]
    d = a.join(e0a, on="nfl_id", how="left").join(targets.select(pl.col("nfl_id").cast(pl.Int64), *tcols), on="nfl_id", how="left")
    if simple is not None:
        d = d.join(simple, on="nfl_id", how="left")
    if m14 is not None:
        d = d.join(m14, on="nfl_id", how="left")
    return d.sort("nfl_id")


def model_set(ae: Any, data: Any, m14_cols: Sequence[str], min_coverage: float, log: list) -> tuple[dict[str, Any], dict[str, str]]:
    """(official-engine models, notes on omitted rungs)."""
    from ..m14.transfer import coverage_factory

    m1 = list(ae.M1_COLS)
    eng = [c for c in data.columns if c.startswith("eng")]
    trk = [c for c in data.columns if c.startswith("trk_")]
    for cols in (m1, eng, trk, m14_cols):
        assert_no_nfl_predictors(cols)
    mods: dict[str, Any] = {"POS": ae.Model([]), "M1": ae.Model(m1)}
    notes: dict[str, str] = {}
    if eng:
        mods["M1+F0"] = ae.Model([*m1, *eng])
    else:
        notes["M1+F0"] = "no F0 columns"
    if trk:
        mods["M1+simple"] = ae.Model(m1, factory=coverage_factory(data, trk, min_coverage, log, "M1+simple"))
    else:
        notes["M1+simple"] = "no simple tracking columns (E15.A1 table not supplied)"
    if m14_cols:
        mods["M1+M14_reliable"] = ae.Model(m1, factory=coverage_factory(data, list(m14_cols), min_coverage, log, "M1+M14_reliable"))
    else:
        notes["M1+M14_reliable"] = "no M14 metric with status 'validated' (or M14 outputs missing)"
    return mods, notes


def run_target(exp: Any, data: Any, target_id: str, m14_cols: Sequence[str], learners: Sequence[str], min_coverage: float, seed: int) -> dict[str, Any]:
    """Wide OOF frame (pred_<model>) of every model for one target + ladder / selection / learner logs."""
    import polars as pl

    from ... import _backend
    from .. import learners as LR

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    col = TARGETS[target_id].column
    if col not in data.columns:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"target {target_id} ({col}) absent from the analysis frame", missing=[col])
    log: list = []
    mods, notes = model_set(ae, data, m14_cols, min_coverage, log)
    res = ae.oof_ladder(data, col, mods, C, C.seed)
    wide = res.oof_frame()
    y, folds = wide["y"].to_numpy(), wide["fold"].to_numpy()
    mean = np.full(wide.height, NAN)
    for f in np.unique(folds):
        mean[folds == f] = float(y[folds != f].mean())
    wide = wide.with_columns(pl.Series("pred_MEAN", mean))
    lrn_log = []
    for name in learners:
        if name not in LR.LEARNERS:
            continue
        for base in ("M1", "M1+simple"):
            if base not in mods:
                continue
            try:
                oof, lg = LR.oof_generic(data, col, list(mods[base].cols), LR.LEARNERS[name](), seed, factory=mods[base].factory)
            except ResearchBlocked as e:
                notes[f"{base}_{name}"] = e.reason
                continue
            wide = wide.join(oof.select("nfl_id", pl.col("pred").alias(f"pred_{base}_{name}")), on="nfl_id", how="left")
            lrn_log += [{"model": f"{base}_{name}", **r} for r in lg]
    return {"wide": wide, "ladder_log": pl.DataFrame(res.log, infer_schema_length=None), "selection_log": log, "learner_log": lrn_log, "notes": notes,
            "target_id": target_id}


def precise_null(paired: Any, margin: float = 0.02) -> Any:
    """M13 H-D per comparison: CI includes 0 AND half-width ≤ margin × RMSE(reference)."""
    import polars as pl

    if paired is None or not paired.height:
        return pl.DataFrame()
    return paired.with_columns((((pl.col("delta_lo") <= 0) & (pl.col("delta_hi") >= 0)) &
                                (((pl.col("delta_hi") - pl.col("delta_lo")) / 2) <= margin * pl.col("rmse_ref"))).alias("precise_null"))


def within_position_ranking(w: Any, models: Sequence[str], n_boot: int, seed: int, min_n: int = 20) -> Any:
    """Per position × model: Spearman(prediction, y) and pairwise concordance with athlete-bootstrap CIs."""
    import polars as pl

    from ..stats import spearman

    rng = np.random.default_rng(seed)
    rows = []
    for p in sorted(set(w["position_group"].to_list())):
        g = w.filter(pl.col("position_group") == p)
        n = g.height
        if n < min_n:
            continue
        y = g["y"].to_numpy().astype(float)
        idx = rng.integers(0, n, (n_boot, n))
        for m in models:
            pr = g[f"pred_{m}"].to_numpy().astype(float)
            iu = np.triu_indices(n, 1)
            conc = float(np.mean(np.sign(pr[iu[0]] - pr[iu[1]]) == np.sign(y[iu[0]] - y[iu[1]])))
            bs = np.array([spearman(pr[i], y[i]) for i in idx])
            bs = bs[np.isfinite(bs)]
            rows.append({"position_group": p, "model": m, "n_athletes": n, "spearman": spearman(pr, y), "spearman_lo": float(np.quantile(bs, 0.025)) if bs.size else NAN,
                         "spearman_hi": float(np.quantile(bs, 0.975)) if bs.size else NAN, "pairwise_concordance": conc})
    return pl.DataFrame(rows, infer_schema_length=None)


def evaluate_target(result: dict[str, Any], folds_by_athlete: dict[int, int], n_boot: int, seed: int, primary: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    """Alignment validation → strict cohort → metrics, paired Δ, intervals, fold table, status, precise null, ranking."""
    import polars as pl

    from .. import position_models as PM
    from .. import transfer_fairness as TF
    from ..m14.transfer import fold_stability, transfer_status

    tid = result["target_id"]
    wide = result["wide"]
    models = [c.removeprefix("pred_") for c in wide.columns if c.startswith("pred_")]
    long = TF.long_predictions(wide, f"m16_{tid}", models)
    audit = TF.validate_alignment(long, folds_by_athlete)
    fin = long.filter(pl.col("pred").is_finite()).group_by("nfl_id").agg(pl.col("model").n_unique().alias("k"))
    strict = set(int(i) for i in fin.filter(pl.col("k") == len(models))["nfl_id"].to_list())
    w = TF.wide(long, strict, models)
    others = [m for m in models if m not in ("M1", "POS", "MEAN")]
    pairs = [(m, "M1") for m in others] + [("M1", "POS"), ("POS", "MEAN")]
    prim = [p for p in primary if p[0] in models and p[1] in models]
    paired = precise_null(TF.paired_comparisons(w, pairs, n_boot, seed, "strict_common", primary_family=prim))
    folds = TF.fold_table(w, models, "M1")
    status = transfer_status(paired, fold_stability(folds))
    within = PM.within_position_metrics(w, models)
    rank = within_position_ranking(w, [m for m in ("M1", "M1+simple", "M1+M14_reliable", "POS") if m in models], max(200, n_boot // 5), seed)
    pos_eff = pl.concat([PM.position_effects(w, m, "M1", max(200, n_boot // 5), seed) for m in others if not m.startswith("M1_")], how="diagonal_relaxed") \
        if others else pl.DataFrame()
    tag = pl.lit(tid).alias("target_id")
    return {"long": long.with_columns(tag), "audit": audit.with_columns(tag), "strict_ids": strict,
            "metrics": TF.cohort_metrics(w, models, "strict_common").with_columns(tag), "paired": paired.with_columns(tag) if paired.height else paired,
            "intervals": TF.rmse_intervals(w, models, n_boot, seed, "strict_common").with_columns(tag), "folds": folds.with_columns(tag),
            "status": status.with_columns(tag) if status.height else status, "within": within.with_columns(tag), "ranking": rank.with_columns(tag) if rank.height else rank,
            "position_effects": pos_eff.with_columns(tag) if pos_eff.height else pos_eff}


__all__ = ["NFL_PREFIXES", "PRIMARY_T0", "SIMPLE_FEATURES", "analysis_frame", "assert_no_nfl_predictors", "evaluate_target", "m14_reliable_table", "model_set",
           "precise_null", "run_target", "simple_tracking_table", "within_position_ranking"]
