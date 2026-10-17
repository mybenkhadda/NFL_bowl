"""Phase 5 — NFL outcome transfer of the M14 metrics, on the SAME engine, analysis frame, folds and seed as the official
ladder (``Experiment._ladder_inputs`` + ``ae.oof_ladder``), and the M13 fairness system (``lab.transfer_fairness``).

Models (all with position as unpenalised dummies and M1, ridge α chosen by inner athlete-grouped CV):

    M1, M1+F0, M1+<combo encoder>         re-fitted here; MUST reproduce the official ladder (asserted, then dropped)
    M1+M14_<family>                       every declared column of the family with TRAIN-fold coverage ≥ min_coverage
                                          (rugby_soccer: + train-fold-only residual metrics)
    M1+M14_all                            every declared column (train coverage filter)
    M1+M14_reliable          [primary]    outcome-free train-fold quality selection (below)
    M1+M14_braking           [primary]    the same selection restricted to the pre-declared braking signature
    M1+M14_reliable+<combo>  [primary]    quality selection + the fold's inductive embedding block of the combo encoder
    M1+<column>              [secondary]  one metric at a time, for measurement-eligible metrics (BH family)

Train-fold quality selection (NO outcome — neither held-out nor training targets — enters it):
    coverage on training athletes ≥ min_coverage → train-athlete ICC(1,1) ≥ reliability_threshold (kind-centred, ≥ min
    athletes) → in-sample R² of the metric from M1 on training athletes ≤ novelty_max_r2 → greedy redundancy pruning in
    decreasing train ICC (|Spearman_train| < redundancy_rho with every kept metric).

The official ladder (M1, M1+F0, M1+<encoder>) and the POS / MEAN baselines enter the long table from their own sources;
claims are made on the ``strict_common`` cohort of M13 (secondary: ``m14_observed`` = strict ∩ every braking-signature
column observed, and ``ladder_all``). Δ RMSE = RMSE(reference) − RMSE(model), positive = model better.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ...data.validation import M1_COLS, TRANSFER_TARGET
from .. import reliability as REL
from .. import transfer_fairness as TF
from . import novelty as NV
from . import registry as RG

NAN = float("nan")
PRIMARY_STEMS = ("M1+M14_reliable", "M1+M14_braking")


def combo_name(combo: str) -> str:
    from ...models import registry as reg

    return reg.resolve(combo)


def primary_models(combo: str) -> list[str]:
    return [*PRIMARY_STEMS, f"M1+M14_reliable+{combo_name(combo)}"]


# ====================================================================== analysis frame
def analysis_frame(exp: Any, ath: Any, ready: Sequence[str]) -> tuple[Any, dict[str, Any], list[str]]:
    """(analysis frame of the official ladder + M14 Combine columns, official ladder model specs, encoder names)."""
    names, _byf, _deps, data, mods = exp._ladder_inputs(list(ready), allow_empty=True)
    feats = ath.select("nfl_id", *[c for c in ath.columns if c.startswith("m14c_")])
    bad = [c for c in feats.columns if c.startswith(("m14n__", "nfl__"))]
    if bad:
        raise TF.ComparisonError(f"outcome-side columns in the predictor frame: {bad}")
    return data.join(feats, on="nfl_id", how="left"), mods, list(names)


# ====================================================================== block factories
def _spearman_train(a: np.ndarray, b: np.ndarray) -> float:
    from ..stats import spearman

    ok = np.isfinite(a) & np.isfinite(b)
    return spearman(a[ok], b[ok]) if ok.sum() >= 20 else NAN


def _train_r2(x: np.ndarray, Z: np.ndarray, train: np.ndarray) -> float:
    """In-sample R² of x from Z on TRAINING rows with finite x (train-median imputation of Z)."""
    tr = train & np.isfinite(x)
    if tr.sum() < 30 or Z.shape[1] == 0:
        return NAN
    Xs = NV._standardise(Z, tr)
    b0, beta = NV._ridge(Xs[tr], x[tr], 1.0)
    p = b0 + Xs[tr] @ beta
    sst = float(((x[tr] - x[tr].mean()) ** 2).sum())
    return 1 - float(((x[tr] - p) ** 2).sum()) / sst if sst > 0 else NAN


def quality_factory(data: Any, cand: Sequence[str], cache: dict[str, Any], cfg: Any, log: list, model: str) -> Callable:
    """Outcome-free train-fold quality selection (module docstring); logs every fold's scores and selection."""
    cand = list(cand)

    def factory(ids: np.ndarray) -> Callable:
        X = REL._aligned(data, ids, cand)
        Z = REL._aligned(data, ids, [c for c in M1_COLS if c in data.columns])
        folds = REL._aligned(data, ids, ["fold"])[:, 0]

        def block(train: np.ndarray) -> Any:
            import polars as pl

            f = REL._fold_of(train, folds)
            tr_ids = set(int(i) for i in ids[train])
            sc: dict[str, dict[str, float]] = {}
            for j, c in enumerate(cand):
                x = X[:, j]
                cov = float(np.isfinite(x[train]).mean()) if train.any() else 0.0
                icc, n = REL.train_icc(cache, c, tr_ids, cfg.min_reliability_athletes) if c in cache else (NAN, 0)
                sc[c] = {"coverage": cov, "train_icc": icc, "n_icc": n, "train_r2_m1": _train_r2(x, Z, train)}
            elig = [c for c in cand if sc[c]["coverage"] >= cfg.min_coverage and np.isfinite(sc[c]["train_icc"]) and sc[c]["train_icc"] >= cfg.reliability_threshold
                    and not (np.isfinite(sc[c]["train_r2_m1"]) and sc[c]["train_r2_m1"] > cfg.novelty_max_r2)]
            sel: list[str] = []
            for c in sorted(elig, key=lambda c: -sc[c]["train_icc"]):
                xj = X[train, cand.index(c)]
                redundant = False
                for s in sel:
                    r = _spearman_train(xj, X[train, cand.index(s)])
                    if np.isfinite(r) and abs(r) >= cfg.redundancy_rho:
                        redundant = True
                        break
                if not redundant:
                    sel.append(c)
            log.append({"model": model, "fold": f, "n_candidates": len(cand), "n_eligible": len(elig), "selected": sel,
                        "scores": {c: sc[c] for c in cand}})
            if not sel:
                return pl.DataFrame({"none": np.zeros(ids.size)})
            return pl.DataFrame({c: X[:, cand.index(c)] for c in sel})
        return block
    return factory


def coverage_factory(data: Any, cols: Sequence[str], min_cov: float, log: list, model: str) -> Callable:
    """Static family columns, kept when their TRAIN-athlete coverage ≥ min_cov (fold-wise, outcome-free)."""
    cols = list(cols)

    def factory(ids: np.ndarray) -> Callable:
        X = REL._aligned(data, ids, cols)
        folds = REL._aligned(data, ids, ["fold"])[:, 0]

        def block(train: np.ndarray) -> Any:
            import polars as pl

            f = REL._fold_of(train, folds)
            keep = [c for j, c in enumerate(cols) if float(np.isfinite(X[train, j]).mean()) >= min_cov]
            log.append({"model": model, "fold": f, "n_candidates": len(cols), "n_eligible": len(keep), "selected": keep, "scores": {}})
            return pl.DataFrame({c: X[:, cols.index(c)] for c in keep}) if keep else pl.DataFrame({"none": np.zeros(ids.size)})
        return block
    return factory


def combine_factories(*factories: Callable | None) -> Callable:
    """One block = the horizontal union of several fold-fitted blocks (columns prefixed per part; empty parts dropped)."""
    fs = [f for f in factories if f is not None]

    def factory(ids: np.ndarray) -> Callable:
        blocks = [f(ids) for f in fs]

        def block(train: np.ndarray) -> Any:
            import polars as pl

            parts = []
            for k, b in enumerate(blocks):
                d = b(train) if b is not None else None
                if d is None or not d.width:
                    continue
                cols = [c for c in d.columns if c != "none"]
                if cols:
                    parts.append(d.select(cols).rename({c: f"p{k}_{c}" for c in cols}))
            return pl.concat(parts, how="horizontal") if parts else pl.DataFrame({"none": np.zeros(ids.size)})
        return block
    return factory


# ====================================================================== model set
def build_models(ae: Any, data: Any, official: dict[str, Any], cand: Sequence[str], cache: dict[str, Any], cfg: Any, eligible_single: Sequence[str],
                 log: list) -> dict[str, Any]:
    """The M14 model dictionary (see module docstring)."""
    m1 = list(ae.M1_COLS)
    combo = combo_name(cfg.combo_encoder)
    mods: dict[str, Any] = {"M1": ae.Model(m1)}
    if "M1+F0" in official:
        mods["M1+F0"] = official["M1+F0"]
    if f"M1+{combo}" in official:
        mods[f"M1+{combo}"] = official[f"M1+{combo}"]
    fams = RG.family_columns(cand)
    res_specs = NV.residual_specs(list(data.columns))
    for fam, cols in fams.items():
        fac = coverage_factory(data, cols, cfg.min_coverage, log, f"M1+M14_{fam}")
        if fam == "rugby_soccer" and res_specs:
            fac = combine_factories(fac, NV.train_residual_factory(data, res_specs, log, model=f"M1+M14_{fam}"))
        mods[f"M1+M14_{fam}"] = ae.Model(m1, factory=fac)
    if "rugby_soccer" not in fams and res_specs:
        mods["M1+M14_rugby_soccer"] = ae.Model(m1, factory=NV.train_residual_factory(data, res_specs, log, model="M1+M14_rugby_soccer"))
    mods["M1+M14_all"] = ae.Model(m1, factory=coverage_factory(data, cand, cfg.min_coverage, log, "M1+M14_all"))
    mods["M1+M14_reliable"] = ae.Model(m1, factory=quality_factory(data, cand, cache, cfg, log, "M1+M14_reliable"))
    braking = [c for c in cand if (p := RG.parse_column(c)) and p[1] in RG.BRAKING_CORE]
    if braking:
        mods["M1+M14_braking"] = ae.Model(m1, factory=quality_factory(data, braking, cache, cfg, log, "M1+M14_braking"))
    emb = official.get(f"M1+{combo}")
    if emb is not None and emb.factory is not None:
        mods[f"M1+M14_reliable+{combo}"] = ae.Model(m1, factory=combine_factories(quality_factory(data, cand, cache, cfg, log, f"M1+M14_reliable+{combo}"),
                                                                                 emb.factory))
    if cfg.single_metric_models:
        for c in eligible_single:
            if c in data.columns:
                mods[f"M1+{c}"] = ae.Model([*m1, c])
    return mods


def run_ladder(exp: Any, data: Any, mods: dict[str, Any], log: list) -> tuple[Any, Any, Any]:
    """(wide OOF, selection log frame, ladder log frame) — the official engine, folds and seed."""
    import json

    import polars as pl

    from ... import _backend

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    res = ae.oof_ladder(data, TRANSFER_TARGET, mods, C, C.seed)
    sel = pl.DataFrame([{"model": r["model"], "fold": r["fold"], "n_candidates": r.get("n_candidates"), "n_eligible": r.get("n_eligible"),
                         "n_selected": len(r.get("selected", [])) if "selected" in r else None, "selected": ", ".join(r.get("selected", [])),
                         "feature": r.get("feature"), "n_train_fit": r.get("n_train_fit"), "scores_json": json.dumps(r.get("scores", {}), default=str)}
                        for r in log], infer_schema_length=None) if log else pl.DataFrame()
    return res.oof_frame(), sel, pl.DataFrame(res.log, infer_schema_length=None)


def selection_frequency(sel: Any) -> Any:
    """How often each metric is selected across folds, per selection model (fold stability of the selection)."""
    import polars as pl

    if sel is None or not sel.height or "selected" not in sel.columns:
        return pl.DataFrame()
    d = sel.filter(pl.col("selected").is_not_null() & (pl.col("selected") != "")).with_columns(pl.col("selected").str.split(", ").alias("_s")).explode("_s")
    nf = sel.group_by("model").agg(pl.col("fold").n_unique().alias("n_folds"))
    return (d.group_by("model", "_s").agg(pl.col("fold").n_unique().alias("folds_selected")).rename({"_s": "column"}).join(nf, on="model")
            .with_columns((pl.col("folds_selected") / pl.col("n_folds")).alias("selection_rate")).sort("model", "selection_rate", descending=[False, True]))


# ====================================================================== integration with the official ladder
def integrate(official_oof: Any, baselines: Any, m14_oof: Any, ready_models: Sequence[str], combo: str, folds_by_athlete: dict[int, int],
              f0_ids: set[int], ready: Sequence[str]) -> dict[str, Any]:
    """Reproducibility checks → one long table → alignment audit → cohorts (strict_common, m14_observed, ladder_all)."""
    import polars as pl

    cname = combo_name(combo)
    checks = [TF.reproducibility_check(official_oof, m14_oof.rename({"pred_M1": "pred_M1_check"}))]
    if f"pred_M1+{cname}" in m14_oof.columns and f"pred_M1+{cname}" in official_oof.columns:
        r = TF.reproducibility_check(official_oof.drop("pred_M1").rename({f"pred_M1+{cname}": "pred_M1"}),
                                     m14_oof.select("nfl_id", "y", pl.col(f"pred_M1+{cname}").alias("pred_M1_check")))
        checks.append({**r, "check": f"M14-run M1+{cname} reproduces the ladder"})
    if "pred_M1+F0" in m14_oof.columns and "pred_M1+F0" in official_oof.columns:
        r = TF.reproducibility_check(official_oof.drop("pred_M1").rename({"pred_M1+F0": "pred_M1"}),
                                     m14_oof.select("nfl_id", "y", pl.col("pred_M1+F0").alias("pred_M1_check")))
        checks.append({**r, "check": "M14-run M1+F0 reproduces the ladder"})
    repro = pl.DataFrame(checks, infer_schema_length=None)
    official_models = [m for m in ready_models if f"pred_{m}" in official_oof.columns]
    m14_models = [c.removeprefix("pred_") for c in m14_oof.columns if c.startswith("pred_") and c.removeprefix("pred_") not in ("M1", "M1+F0", f"M1+{cname}")]
    long = pl.concat([TF.long_predictions(official_oof, "ladder", models=official_models),
                      TF.long_predictions(baselines, "baselines", models=list(TF.BASELINES)),
                      TF.long_predictions(m14_oof, "m14", models=m14_models)], how="vertical")
    audit = TF.validate_alignment(long, folds_by_athlete)
    primary = [*official_models, *[m for m in m14_models if not m.startswith("M1+m14c_")]]
    coh, desc = TF.cohorts(long, official_oof, primary, f0_ids, list(ready))
    return {"repro": repro, "long": long, "audit": audit, "cohorts": coh, "cohort_desc": desc, "official_models": official_models, "m14_models": m14_models}


def observed_cohort(strict: set[int], ath: Any, cols: Sequence[str]) -> set[int]:
    """strict_common ∩ athletes with EVERY column in ``cols`` observed (secondary coverage-explicit cohort)."""
    import polars as pl

    cols = [c for c in cols if c in ath.columns]
    if not cols:
        return set()
    ok = set(int(i) for i in ath.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in cols]))["nfl_id"].to_list())
    return strict & ok


# ====================================================================== comparisons
def comparison_pairs(models: Sequence[str], combo: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    cname = combo_name(combo)
    ms = set(models)
    pairs = [(m, "M1") for m in models if m not in ("M1", *TF.BASELINES)]
    pairs += [(m, "M1+F0") for m in models if m.startswith("M1+M14") and "M1+F0" in ms]
    if f"M1+M14_reliable+{cname}" in ms and f"M1+{cname}" in ms:
        pairs.append((f"M1+M14_reliable+{cname}", f"M1+{cname}"))
    pairs += [("M1", "POS")] if "POS" in ms else []
    prim = [(m, "M1") for m in primary_models(combo) if m in ms]
    return list(dict.fromkeys(pairs)), prim


def evaluate(long: Any, ids: set[int], models: Sequence[str], cohort: str, combo: str, n_boot: int, seed: int) -> dict[str, Any]:
    """Metrics, paired Δ (Holm on the primary family, BH on all), RMSE intervals and fold table on ONE cohort."""
    w = TF.wide(long, ids, list(models))
    pairs, prim = comparison_pairs(models, combo)
    return {"wide": w, "metrics": TF.cohort_metrics(w, list(models), cohort), "paired": TF.paired_comparisons(w, pairs, n_boot, seed, cohort, primary_family=prim),
            "intervals": TF.rmse_intervals(w, list(models), n_boot, seed, cohort), "folds": TF.fold_table(w, list(models), "M1")}


def fold_stability(folds: Any) -> Any:
    import polars as pl

    col = "delta_rmse_vs_M1"
    if folds is None or not folds.height or col not in folds.columns:
        return pl.DataFrame()
    return folds.group_by("model").agg((pl.col(col) > 0).sum().alias("folds_better"), pl.len().alias("n_folds"), pl.col(col).mean().alias("mean_fold_delta"),
                                       pl.col(col).min().alias("worst_fold_delta")).sort("model")


def by_position(w: Any, models: Sequence[str], min_n: int, n_boot: int, seed: int) -> tuple[Any, Any]:
    """(metrics per position, paired Δ vs M1 per position with BH inside each position) for positions with ≥ min_n."""
    import polars as pl

    mets, pairs_out = [], []
    for p in sorted(set(w["position_group"].to_list())):
        sub = w.filter(pl.col("position_group") == p)
        if sub.height < max(min_n, 20):
            continue
        mets.append(TF.cohort_metrics(sub, list(models), f"position:{p}").with_columns(pl.lit(p).alias("position_group")))
        pp = [(m, "M1") for m in models if m not in ("M1", *TF.BASELINES)]
        try:
            pairs_out.append(TF.paired_comparisons(sub, pp, n_boot, seed, f"position:{p}").with_columns(pl.lit(p).alias("position_group")))
        except TF.ComparisonError:
            continue
    m = pl.concat(mets, how="diagonal_relaxed") if mets else pl.DataFrame()
    q = pl.concat(pairs_out, how="diagonal_relaxed") if pairs_out else pl.DataFrame()
    return m, q


def transfer_status(paired: Any, stab: Any, alpha: float = 0.05, min_folds: int = 4) -> Any:
    """Declared predictive status per (model, reference):
    validated    Δ CI lower > 0 AND adjusted p ≤ alpha (Holm for the primary family, BH otherwise) AND better in ≥ min_folds folds
    exploratory  Δ > 0 and bootstrap share Δ ≤ 0 ≤ 0.2, not validated
    invalid      Δ CI upper < 0 (the model is reliably WORSE: the incremental-value hypothesis is refuted on this cohort)
    inconclusive otherwise."""
    import polars as pl

    if paired is None or not paired.height:
        return pl.DataFrame()
    st = {r["model"]: r for r in stab.iter_rows(named=True)} if stab is not None and stab.height else {}
    rows = []
    for r in paired.iter_rows(named=True):
        padj = r.get("p_holm_primary") if r.get("primary") else r.get("p_bh_all")
        fb = st.get(r["model"], {}).get("folds_better") if r["reference"] == "M1" else None
        if r["delta_lo"] > 0 and padj is not None and padj <= alpha and (fb is None or fb >= min_folds):
            s = "validated"
        elif r["delta_hi"] < 0:
            s = "invalid"
        elif r["delta_rmse"] > 0 and r["p_boot_le0"] <= 0.2:
            s = "exploratory"
        else:
            s = "inconclusive"
        rows.append({"cohort": r["cohort"], "model": r["model"], "reference": r["reference"], "delta_rmse": r["delta_rmse"], "delta_lo": r["delta_lo"],
                     "delta_hi": r["delta_hi"], "p_adjusted": padj, "adjustment": "holm (primary)" if r.get("primary") else "bh (all)", "folds_better": fb,
                     "status": s})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== permutation null
def permutation_null(exp: Any, data: Any, cand: Sequence[str], cache: dict[str, Any], cfg: Any, observed_w: Any, n_perm: int, seed: int) -> Any:
    """Within-position permutation of the M14 columns (jointly, rows shuffled) → M1+M14_reliable refitted → Δ RMSE vs
    the observed M1 on the same cohort. Selection depends only on trial ICC (athlete ids), so the selected set is
    unchanged and only the athlete ↔ value link is broken."""
    import polars as pl

    from ... import _backend

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    rng = np.random.default_rng(seed)
    ids = set(int(i) for i in observed_w["nfl_id"].to_list())
    y_obs = observed_w.select("nfl_id", "y", "pred_M1")
    rows = []
    cand = [c for c in cand if c in data.columns]
    pos = data["position_group"].fill_null("?").to_numpy()
    for k in range(n_perm):
        perm = np.arange(data.height)
        for p in np.unique(pos):
            ix = np.flatnonzero(pos == p)
            perm[ix] = ix[rng.permutation(ix.size)]
        shuffled = data.with_columns([pl.Series(c, data[c].cast(pl.Float64).to_numpy()[perm], dtype=pl.Float64).fill_nan(None) for c in cand])
        log: list = []
        mods = {"M1+M14_reliable_perm": ae.Model(list(ae.M1_COLS), factory=quality_factory(shuffled, cand, cache, cfg, log, "perm"))}
        oof = ae.oof_ladder(shuffled, TRANSFER_TARGET, mods, C, C.seed).oof_frame()
        j = y_obs.join(oof.select("nfl_id", "pred_M1+M14_reliable_perm"), on="nfl_id", how="inner").filter(pl.col("nfl_id").is_in(list(ids)))
        y, a, b = j["y"].to_numpy(), j["pred_M1"].to_numpy(), j["pred_M1+M14_reliable_perm"].to_numpy()
        rows.append({"permutation": k, "n": j.height, "delta_rmse": float(np.sqrt(np.mean((y - a) ** 2)) - np.sqrt(np.mean((y - b) ** 2)))})
    return pl.DataFrame(rows)


def permutation_p(null: Any, observed_delta: float) -> dict[str, float]:
    d = null["delta_rmse"].to_numpy() if null is not None and null.height else np.array([])
    if not d.size or not np.isfinite(observed_delta):
        return {"n_permutations": int(d.size), "observed_delta": observed_delta, "p_perm": NAN}
    return {"n_permutations": int(d.size), "observed_delta": float(observed_delta), "null_mean": float(d.mean()), "null_q95": float(np.quantile(d, 0.95)),
            "p_perm": float((1 + (d >= observed_delta).sum()) / (1 + d.size))}


__all__ = ["PRIMARY_STEMS", "analysis_frame", "build_models", "by_position", "combine_factories", "combo_name", "comparison_pairs", "coverage_factory",
           "evaluate", "fold_stability", "integrate", "observed_cohort", "permutation_null", "permutation_p", "primary_models", "quality_factory", "run_ladder",
           "selection_frequency", "transfer_status"]
