"""E16.C4–C5 — incremental NFL prediction ladder B0–B7 with fold-safe Combine-only features, and its negative controls.

M16 numbering (this module; M18 numbers its confirmatory ladder differently and owns the confirmatory claim):

    B0  POS               position only                                            ``ae.Model([])``
    B1  M1                official Combine M1 (``M1_COLS``)                        ``ae.Model(M1_COLS)``
    B2  M1+simple         M1 + simple kinematic statistics (E15.A1 athlete medians, ``baselines.simple_tracking_table``)
    B3  M1+PCA            M1 + PCA embedding block      ┐ fold-f encoder, Combine attempts only, primary aggregation,
    B4  M1+STFT           M1 + STFT-AE embedding block  │ train-fold PCA to ≤ ``embedding_k`` dims (``se.embedding_block``);
    B5  M1+LSTM           M1 + LSTM-AE embedding block  ┘ representation = the VERIFIED_FOLD_SAFE one at ``ladder_dim``
    B6  M1+STFT+LSTM      M1 + fused STFT ⊕ LSTM block (train-standardised, ``features.fuse``)
    B7  M1+M14_reliable   M1 + M14 ``validated`` metrics; BLOCKED when none passed (never relaxed)

Every rung is fitted by the official engine (``ae.oof_ladder``: athlete outer folds, train-fold medians / missing indicators /
standardisation, unpenalised position dummies, inner-CV ridge α on training athletes only). Optional learners on the same
rows: ENet / LightGBM / CatBoost (``learners``) and ``pooled`` — a partially pooled position model (global ridge + per-
position slope deviations shrunk towards it, both penalties chosen by inner CV on training athletes).

Representation rungs with provenance other than VERIFIED_FOLD_SAFE are BLOCKED_PROVENANCE; representations without G15
GO are fitted (fold-safe) but labelled ``exploratory_screen`` — they cannot pass G16 / G18 in M16.

Negative controls (``controls``): per embedding rung, ``n_shuffle`` refits with vectors permuted WITHIN POSITION in every
fold table (athlete link broken, position structure kept) → null distribution of ΔRMSE vs M1; random Gaussian blocks;
reduced dimensions (k ∈ ``reduced_dims``). A rung whose gain is not above its shuffled null is never called athlete-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ...data.validation import M1_COLS
from ..status import BLOCKED_MISSING_INPUT, BLOCKED_PROVENANCE, ResearchBlocked

NAN = float("nan")
RUNGS = {"B0": "POS", "B1": "M1", "B2": "M1+simple", "B3": "M1+PCA", "B4": "M1+STFT", "B5": "M1+LSTM", "B6": "M1+STFT+LSTM", "B7": "M1+M14_reliable"}
ARCH = {"B3": "pca", "B4": "stft_ae", "B5": "lstm_ae"}
EMBEDDING_RUNGS = ("B3", "B4", "B5", "B6")


@dataclass
class RungPlan:
    rung: str
    model: str
    status: str                                       # eligible · BLOCKED_* (never fitted)
    reason: str
    rep: dict[str, Any] | None = None
    columns: list[str] = field(default_factory=list)
    label: str = "primary"                            # primary · exploratory_screen (representation without G15 GO)

    def row(self) -> dict[str, Any]:
        return {"rung": self.rung, "model": self.model, "status": self.status, "reason": self.reason, "rep": self.rep["rep"] if self.rep else None,
                "seed": int(self.rep["seed"]) if self.rep else None, "g15_decision": self.rep.get("g15_decision") if self.rep else None,
                "label": self.label, "n_columns": len(self.columns)}


def plan_rungs(cfg: Any, reg: Any, analysis: Any, m14_cols: Sequence[str]) -> dict[str, RungPlan]:
    """Eligibility of every rung from saved evidence only (provenance registry, analysis-frame columns, M14 status)."""
    from .representation_audit import choose

    plan = {"B0": RungPlan("B0", RUNGS["B0"], "eligible", "always"), "B1": RungPlan("B1", RUNGS["B1"], "eligible", "official M1", columns=list(M1_COLS))}
    trk = [c for c in analysis.columns if c.startswith("trk_")]
    plan["B2"] = RungPlan("B2", RUNGS["B2"], "eligible" if trk else BLOCKED_MISSING_INPUT,
                          f"{len(trk)} simple kinematic column(s)" if trk else "no trk_* columns (E15.A1 feature table missing)", columns=trk)
    for r in ("B3", "B4", "B5"):
        rep, why = choose(reg, ARCH[r], cfg.ladder_dim)
        if rep is None:
            st = BLOCKED_PROVENANCE if "not VERIFIED_FOLD_SAFE" in why else BLOCKED_MISSING_INPUT
            plan[r] = RungPlan(r, RUNGS[r], st, why)
        else:
            go = str(rep.get("g15_decision")) == "GO"
            plan[r] = RungPlan(r, RUNGS[r], "eligible", why, rep=rep, label="primary" if go else "exploratory_screen")
    if plan["B4"].status == "eligible" and plan["B5"].status == "eligible":
        lab = "primary" if plan["B4"].label == plan["B5"].label == "primary" else "exploratory_screen"
        plan["B6"] = RungPlan("B6", RUNGS["B6"], "eligible", "STFT ⊕ LSTM, train-standardised blocks", label=lab)
    else:
        plan["B6"] = RungPlan("B6", RUNGS["B6"], plan["B4"].status if plan["B4"].status != "eligible" else plan["B5"].status, "needs eligible B4 and B5")
    plan["B7"] = RungPlan("B7", RUNGS["B7"], "eligible" if m14_cols else BLOCKED_MISSING_INPUT,
                          f"{len(m14_cols)} M14 validated metric(s)" if m14_cols else "no M14 metric passed the measurement gate (criteria never relaxed)",
                          columns=list(m14_cols))
    return plan


def plan_frame(plan: dict[str, RungPlan]) -> Any:
    import polars as pl

    return pl.DataFrame([plan[k].row() for k in sorted(plan)], infer_schema_length=None)


def embedding_tables(cfg: Any, plan: dict[str, RungPlan], fold_of: dict[int, int], outer_folds: int, agg: str | None = None, scope: str = "all") -> tuple[dict[str, Any], Any]:
    """({rung: {fold: table}}, inventory) for the eligible embedding rungs (B6 = fusion of B4 and B5)."""
    import polars as pl

    from . import features as FT

    agg = agg or cfg.primary_aggregation
    out, inv = {}, []
    for r in ("B3", "B4", "B5"):
        if plan[r].status == "eligible":
            t, i = FT.by_fold(cfg.m15_run_dir, plan[r].rep, scope, agg, fold_of, outer_folds, cfg.trim_share)
            out[r] = t
            inv.append(i.with_columns(pl.lit(r).alias("rung")))
    if plan["B6"].status == "eligible" and "B4" in out and "B5" in out:
        out["B6"] = FT.fuse(out["B4"], out["B5"], fold_of)
    return out, (pl.concat(inv, how="diagonal_relaxed") if inv else pl.DataFrame())


def build_models(cfg: Any, plan: dict[str, RungPlan], analysis: Any, fold_of: dict[int, int], tables: dict[str, Any], log: list, k: int | None = None) -> dict[str, Any]:
    """Official-engine models keyed by model name (eligible rungs only)."""
    from ... import _backend
    from ..m14.transfer import coverage_factory
    from .baselines import assert_no_nfl_predictors
    from .features import block_factory

    ae = _backend.ae()
    m1 = list(ae.M1_COLS)
    mods: dict[str, Any] = {RUNGS["B0"]: ae.Model([]), RUNGS["B1"]: ae.Model(m1)}
    for r in ("B2", "B7"):
        if plan[r].status == "eligible":
            assert_no_nfl_predictors(plan[r].columns)
            mods[RUNGS[r]] = ae.Model(m1, factory=coverage_factory(analysis, plan[r].columns, cfg.min_coverage, log, RUNGS[r]))
    for r in EMBEDDING_RUNGS:
        if plan[r].status == "eligible" and r in tables:
            mods[RUNGS[r]] = ae.Model(m1, factory=block_factory(fold_of, tables[r], cfg.embedding_k if k is None else k))
    return mods


# ====================================================================== learners
def pooled_learner(alphas: Sequence[float], lams: Sequence[float] = (0.3, 3.0, 30.0, 300.0), inner: int = 5) -> Callable:
    """Partially pooled position model: ridge on [position intercepts, features] (penalty λ on features), then per-position
    slope deviations δ_p = argmin ||r_p − X_p δ||² + α||δ||² on the global residuals. α → ∞ = pooled, α → 0 = separate
    slopes. (λ, α) by inner athlete CV on TRAINING athletes; the design (medians, scaling) uses outer-training rows only."""
    from .. import learners as LR

    def fit_predict(df: Any, y: np.ndarray, cols: Sequence[str], extra: Any, train: np.ndarray, test: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        d = LR.design_train(df, cols, extra, train)
        X, P = d.X[:, d.pen], d.X[:, ~d.pen]
        pos = df["position_group"].fill_null("?").to_numpy()

        def fit(rows: np.ndarray, lam: float, alpha: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
            A = np.column_stack([np.ones(int(rows.sum())), P[rows], X[rows]])
            pen = np.r_[0.0, np.zeros(P.shape[1]), np.ones(X.shape[1])]
            beta = np.linalg.solve(A.T @ A + lam * np.diag(pen) + 1e-9 * np.eye(A.shape[1]), A.T @ y[rows])
            res = y[rows] - A @ beta
            dev = {}
            for p in np.unique(pos[rows]):
                m = pos[rows] == p
                Xp = X[rows][m]
                dev[p] = np.linalg.solve(Xp.T @ Xp + alpha * np.eye(X.shape[1]), Xp.T @ res[m]) if X.shape[1] else np.zeros(0)
            return beta, dev

        def predict(rows: np.ndarray, beta: np.ndarray, dev: dict[str, np.ndarray]) -> np.ndarray:
            A = np.column_stack([np.ones(int(rows.sum())), P[rows], X[rows]])
            out = A @ beta
            pr, Xr = pos[rows], X[rows]
            for p, dv in dev.items():
                m = pr == p
                if m.any() and dv.size:
                    out[m] += Xr[m] @ dv
            return out

        grid = [(lam, a) for lam in lams for a in alphas]
        idx = np.flatnonzero(train)
        fo = LR._inner_folds(idx.size, inner, seed)
        sse = np.zeros(len(grid))
        for j in range(inner):
            tr, va = np.zeros(df.height, bool), np.zeros(df.height, bool)
            tr[idx[fo != j]], va[idx[fo == j]] = True, True
            for g, (lam, a) in enumerate(grid):
                b, dv = fit(tr, lam, a)
                sse[g] += float(((y[va] - predict(va, b, dv)) ** 2).sum())
        lam, a = grid[int(np.argmin(sse))]
        b, dv = fit(train, lam, a)
        return predict(test, b, dv), {"lambda": lam, "alpha": a, "n_positions": len(dv)}
    return fit_predict


def learner_fns(cfg: Any) -> dict[str, Callable]:
    from .. import learners as LR

    out = {}
    for name in cfg.learners:
        if name == "pooled":
            out[name] = pooled_learner(cfg.pooling_alphas)
        elif name in LR.LEARNERS:
            out[name] = LR.LEARNERS[name]()
    return out


# ====================================================================== fitting and evaluation
def fit(exp: Any, analysis: Any, target_id: str, target_col: str, mods: dict[str, Any], learners: dict[str, Callable], learner_models: Sequence[str],
        seed: int) -> dict[str, Any]:
    """Wide OOF frame (pred_<model>; + MEAN; + <model>_<learner>) on the official row filter of ``target_col``."""
    import polars as pl

    from ... import _backend
    from .. import learners as LR

    ae = _backend.ae()
    if target_col not in analysis.columns:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"target {target_id} ({target_col}) absent from the analysis frame", missing=[target_col])
    C = exp.cfg.science.to_m12config()
    res = ae.oof_ladder(analysis, target_col, mods, C, C.seed)
    wide = res.oof_frame()
    y, folds = wide["y"].to_numpy(), wide["fold"].to_numpy()
    mean = np.full(wide.height, NAN)
    for f in np.unique(folds):
        mean[folds == f] = float(y[folds != f].mean())
    wide = wide.with_columns(pl.Series("pred_MEAN", mean))
    notes, lrn_log = {}, []
    for lname, lfn in learners.items():
        for base in learner_models:
            if base not in mods:
                continue
            try:
                oof, lg = LR.oof_generic(analysis, target_col, list(mods[base].cols), lfn, seed, factory=mods[base].factory)
            except ResearchBlocked as e:
                notes[f"{base}_{lname}"] = e.reason
                continue
            wide = wide.join(oof.select(pl.col("nfl_id").cast(wide["nfl_id"].dtype), pl.col("pred").alias(f"pred_{base}_{lname}")), on="nfl_id", how="left")
            lrn_log += [{"model": f"{base}_{lname}", **r} for r in lg]
    return {"wide": wide, "ladder_log": pl.DataFrame(res.log, infer_schema_length=None), "learner_log": lrn_log, "notes": notes, "target_id": target_id,
            "selection_log": []}


def evaluate(result: dict[str, Any], fold_of: dict[int, int], cfg: Any, primary: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """``baselines.evaluate_target`` (strict cohort, paired Δ, Holm on ``primary``, status, within-position metrics) plus
    within-position ranking of every base model and the identical-cohort audit."""
    import polars as pl

    from .baselines import evaluate_target, within_position_ranking
    from .evaluation import cohort_audit

    ev = evaluate_target(result, fold_of, cfg.n_boot, cfg.seed, primary)
    models = [c.removeprefix("pred_") for c in result["wide"].columns if c.startswith("pred_")]
    base_models = [m for m in models if m in RUNGS.values()]
    w = _strict_wide(ev, models)
    tag = pl.lit(result["target_id"]).alias("target_id")
    ev["ranking"] = within_position_ranking(w, base_models, max(200, cfg.n_boot // 5), cfg.seed).with_columns(tag) if w is not None else pl.DataFrame()
    ev["cohort_audit"] = cohort_audit(ev["long"], models).with_columns(tag)
    ev["wide_strict"] = w
    ev["target_id"] = result["target_id"]
    return ev


def _strict_wide(ev: dict[str, Any], models: Sequence[str]) -> Any:
    from .. import transfer_fairness as TF

    try:
        return TF.wide(ev["long"], ev["strict_ids"], list(models))
    except TF.ComparisonError:
        return None


def primary_family(plan: dict[str, RungPlan]) -> list[tuple[str, str]]:
    """Pre-declared M16 screen family: every eligible rung B2…B7 vs M1 (Holm inside the family)."""
    return [(plan[r].model, "M1") for r in ("B2", "B3", "B4", "B5", "B6", "B7") if plan[r].status == "eligible"]


# ====================================================================== negative controls
def shuffled_tables(tables: dict[int, Any], positions: dict[int, str], seed: int) -> dict[int, Any]:
    """Every fold table with its vectors permuted among athletes of the same position (nfl_id column kept)."""
    import polars as pl

    rng = np.random.default_rng(seed)
    out = {}
    for f, t in tables.items():
        cols = [c for c in t.columns if c != "nfl_id"]
        ids = t["nfl_id"].to_numpy()
        pos = np.array([positions.get(int(i), "?") for i in ids])
        perm = np.arange(ids.size)
        for p in np.unique(pos):
            m = np.flatnonzero(pos == p)
            perm[m] = m[rng.permutation(m.size)]
        X = t.select(cols).to_numpy()[perm]
        out[f] = pl.DataFrame({"nfl_id": ids, **{c: X[:, j] for j, c in enumerate(cols)}})
    return out


def random_tables(tables: dict[int, Any], seed: int) -> dict[int, Any]:
    import polars as pl

    rng = np.random.default_rng(seed)
    return {f: pl.DataFrame({"nfl_id": t["nfl_id"].to_numpy(), **{c: rng.normal(size=t.height) for c in t.columns if c != "nfl_id"}}) for f, t in tables.items()}


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2))) if ok.any() else NAN


def control_refit(exp: Any, analysis: Any, target_col: str, fold_of: dict[int, int], tables: dict[int, Any], k: int, ref: Any) -> float:
    """ΔRMSE (M1 − control) of M1 + a control block on the reference rows (``ref``: nfl_id, y, pred_M1 of the main ladder)."""
    import polars as pl

    from ... import _backend
    from .features import block_factory

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    res = ae.oof_ladder(analysis, target_col, {"ctrl": ae.Model(list(ae.M1_COLS), factory=block_factory(fold_of, tables, k))}, C, C.seed)
    w = res.oof_frame().select(pl.col("nfl_id").cast(pl.Int64), "pred_ctrl").join(ref, on="nfl_id", how="inner")
    y, a, b = w["y"].to_numpy(), w["pred_M1"].to_numpy(), w["pred_ctrl"].to_numpy()
    return _rmse(y, a) - _rmse(y, b)


def controls(exp: Any, analysis: Any, target_id: str, target_col: str, fold_of: dict[int, int], tables: dict[str, Any], ev: dict[str, Any], cfg: Any,
             positions: dict[int, str]) -> tuple[Any, Any]:
    """(null-distribution rows, summary per rung): shuffled-within-position, random and reduced-dimension controls vs the
    observed ΔRMSE (M1 − rung) on the same strict cohort."""
    import polars as pl

    w = ev.get("wide_strict")
    if w is None:
        return pl.DataFrame(), pl.DataFrame()
    ref = w.select(pl.col("nfl_id").cast(pl.Int64), "y", "pred_M1")
    rows, summ = [], []
    for r, t in tables.items():
        model = RUNGS[r]
        if f"pred_{model}" not in w.columns:
            continue
        obs = _rmse(w["y"].to_numpy(), w["pred_M1"].to_numpy()) - _rmse(w["y"].to_numpy(), w[f"pred_{model}"].to_numpy())
        null = []
        for s in range(cfg.n_shuffle):
            d = control_refit(exp, analysis, target_col, fold_of, shuffled_tables(t, positions, cfg.seed + 7919 * s), cfg.embedding_k, ref)
            null.append(d)
            rows.append({"target_id": target_id, "rung": r, "model": model, "control": "shuffled_within_position", "replicate": s, "delta_rmse_vs_M1": d})
        rnd = [control_refit(exp, analysis, target_col, fold_of, random_tables(t, cfg.seed + 104729 * s), cfg.embedding_k, ref) for s in range(5)]
        rows += [{"target_id": target_id, "rung": r, "model": model, "control": "random_embedding", "replicate": s, "delta_rmse_vs_M1": d} for s, d in enumerate(rnd)]
        for kk in cfg.reduced_dims:
            d = control_refit(exp, analysis, target_col, fold_of, t, kk, ref)
            rows.append({"target_id": target_id, "rung": r, "model": model, "control": f"reduced_k{kk}", "replicate": 0, "delta_rmse_vs_M1": d})
        nv = np.asarray([x for x in null if np.isfinite(x)])
        p_perm = float((1 + (nv >= obs).sum()) / (1 + nv.size)) if nv.size else NAN
        summ.append({"target_id": target_id, "rung": r, "model": model, "observed_delta_rmse": obs, "null_mean": float(nv.mean()) if nv.size else NAN,
                     "null_q95": float(np.quantile(nv, 0.95)) if nv.size else NAN, "n_null": int(nv.size), "p_perm": p_perm,
                     "random_mean": float(np.nanmean(rnd)) if rnd else NAN, "above_shuffled_null": bool(nv.size and obs > np.quantile(nv, 0.95)),
                     "reading": ("gain exceeds the within-position shuffled null" if nv.size and obs > np.quantile(nv, 0.95)
                                 else "indistinguishable from within-position shuffled embeddings: NOT athlete-specific")})
    return pl.DataFrame(rows, infer_schema_length=None), pl.DataFrame(summ, infer_schema_length=None)


def sensitivity_models(cfg: Any, plan: dict[str, RungPlan], analysis: Any, fold_of: dict[int, int], outer_folds: int, log: list) -> tuple[dict[str, Any], Any]:
    """M1 + each eligible single-representation rung under every sensitivity aggregation and drill scope (forty, agility)."""
    import polars as pl

    from ... import _backend
    from . import features as FT

    ae = _backend.ae()
    m1 = list(ae.M1_COLS)
    mods: dict[str, Any] = {"M1": ae.Model(m1)}
    inv = []
    for r in ("B3", "B4", "B5"):
        if plan[r].status != "eligible":
            continue
        variants = [(a, "all") for a in cfg.sensitivity_aggregations] + [(cfg.primary_aggregation, s) for s in ("forty", "agility")]
        for agg, scope in variants:
            t, i = FT.by_fold(cfg.m15_run_dir, plan[r].rep, scope, agg, fold_of, outer_folds, cfg.trim_share)
            inv.append(i.with_columns(pl.lit(r).alias("rung")))
            mods[f"{RUNGS[r]}[{agg}|{scope}]"] = ae.Model(m1, factory=FT.block_factory(fold_of, t, cfg.embedding_k))
    return mods, (pl.concat(inv, how="diagonal_relaxed") if inv else pl.DataFrame())


__all__ = ["ARCH", "EMBEDDING_RUNGS", "RUNGS", "RungPlan", "build_models", "control_refit", "controls", "embedding_tables", "evaluate", "fit", "learner_fns",
           "plan_frame", "plan_rungs", "pooled_learner", "primary_family", "random_tables", "sensitivity_models", "shuffled_tables"]
