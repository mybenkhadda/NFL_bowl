"""E16.B6 — does alignment damage the representation? Repeatability-preservation audit on independent held-out folds.

Per verified representation × outer fold (held-out athletes only; anything fitted uses training athletes of the fold):

    repeat AUC       within-position, within-drill attempt identification on held-out Combine attempts (``m15.latent.
                     repeat_auc``) — unaligned, and after the Procrustes map / the InfoNCE Combine head fitted on training
                     athletes' (Combine, NFL) means
    paired change    aligned − unaligned per fold and drill; across the five independent held-out folds: mean, t-interval
                     (df = n_folds − 1), folds with a drop > ``damage_threshold``. ``destructive`` when the mean drop exceeds
                     the threshold or ≥ 3 folds drop by more than it (G17 damage rule)
    geometry         effective rank / d, collapsed-dimension share (``m15.latent.geometry``)
    ICC              median per-dimension ICC(1,1) of repeated attempts and the share of dimensions ≥ 0.6 (``dim_icc``)
    probes           domain AUC and position balanced accuracy (``m15.latent.probes``): how much of the space is domain /
                     position
    duration         held-out R² of log(valid frames) from the latent vector (ridge fitted on training rows) and the repeat
                     AUC within short vs long attempts (median split per drill): sensitivity to observation duration

An alignment is never chosen because it raises unrestricted top-1 retrieval: a destructive arm is reported as such.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAN = float("nan")
KINDS = ("forty", "three_cone", "short_shuttle")


def _t_interval(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return NAN, NAN
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(x.size - 1, 2.0)
    h = t * x.std(ddof=1) / np.sqrt(x.size)
    return float(x.mean() - h), float(x.mean() + h)


def mapped_table(emb: Any, fmap: Any) -> Any:
    """Apply a Combine-side map to every Combine row (NFL rows unchanged — only Combine attempts are scored)."""
    import polars as pl

    from ..m15.latent import zcols

    zc = zcols(emb)
    comb = emb.filter(pl.col("domain") == "combine")
    Z = np.asarray(fmap(comb.select(zc).to_numpy().astype(float)), float)
    return comb.drop(zc).with_columns(*[pl.Series(f"z{k}", Z[:, k]) for k in range(Z.shape[1])])


def fold_rows(emb: Any, label: dict[str, Any], arms: Sequence[str], seed: int) -> list[dict[str, Any]]:
    """Repeat AUC per drill: unaligned and each fittable alignment arm (Combine map from training athletes)."""
    from ..m15.latent import repeat_auc
    from .controlled_retrieval import TRAIN_ROLES, arm_transform, means

    base = repeat_auc(emb, label, KINDS)
    out = [{**r, "arm": "unaligned"} for r in base.to_dicts()]
    _, _, Ct, Nt = means(emb, TRAIN_ROLES, None)
    for arm in arms:
        if arm == "unaligned":
            continue
        maps = arm_transform(arm, Ct, Nt, Ct.shape[1] if Ct.size else 0, seed)
        if maps is None:
            out.append({**label, "arm": arm, "kind": None, "repeat_auc": NAN, "note": "arm not fittable (too few training athletes / no torch)"})
            continue
        out += [{**r, "arm": arm} for r in repeat_auc(mapped_table(emb, maps[0]), label, KINDS).to_dicts()]
    return out


def damage_summary(rows: Any, threshold: float) -> Any:
    """Per rep × arm × drill: paired change vs unaligned over folds (mean, t-interval, folds dropping > threshold)."""
    import polars as pl

    if rows is None or not rows.height:
        return pl.DataFrame()
    base = rows.filter(pl.col("arm") == "unaligned").select("rep", "fold", "kind", pl.col("repeat_auc").alias("auc_unaligned"))
    al = rows.filter(pl.col("arm") != "unaligned").join(base, on=["rep", "fold", "kind"], how="inner").with_columns(
        (pl.col("repeat_auc") - pl.col("auc_unaligned")).alias("delta"))
    out = []
    for (rep, arm, kind), g in al.group_by("rep", "arm", "kind"):
        d = g["delta"].to_numpy().astype(float)
        lo, hi = _t_interval(d)
        n_drop = int((d < -threshold).sum())
        mean = float(np.nanmean(d)) if np.isfinite(d).any() else NAN
        out.append({"rep": rep, "arm": arm, "kind": kind, "n_folds": int(np.isfinite(d).sum()), "auc_unaligned_mean": float(g["auc_unaligned"].mean()),
                    "auc_aligned_mean": float(g["repeat_auc"].mean()), "delta_mean": mean, "delta_lo": lo, "delta_hi": hi, "folds_dropping": n_drop,
                    "destructive": bool((np.isfinite(mean) and mean < -threshold) or n_drop >= 3)})
    return pl.DataFrame(out, infer_schema_length=None).sort("rep", "arm", "kind")


def duration_sensitivity(emb: Any, label: dict[str, Any], index: Any, alpha: float = 1.0) -> dict[str, Any]:
    """Held-out R² of log(valid frames) from z (ridge on training rows) and repeat AUC in short vs long attempts."""
    import polars as pl

    from ..m15.latent import repeat_auc, zcols

    zc = zcols(emb)
    d = emb.join(index.select("idx", "n_valid"), on="idx", how="left")
    y = np.log(np.maximum(d["n_valid"].cast(pl.Float64).fill_null(NAN).to_numpy(), 1.0))
    X = d.select(zc).to_numpy().astype(float)
    tr = (d["role"] != "held_out").to_numpy() & np.isfinite(y)
    te = (d["role"] == "held_out").to_numpy() & np.isfinite(y)
    out = {**label, "duration_r2_heldout": NAN}
    if tr.sum() > len(zc) + 20 and te.sum() > 20:
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xt, Xe = (X[tr] - mu) / sd, (X[te] - mu) / sd
        beta = np.linalg.solve(Xt.T @ Xt + alpha * np.eye(Xt.shape[1]), Xt.T @ (y[tr] - y[tr].mean()))
        p = y[tr].mean() + Xe @ beta
        ss = float(((y[te] - y[te].mean()) ** 2).sum())
        out["duration_r2_heldout"] = 1 - float(((y[te] - p) ** 2).sum()) / ss if ss > 0 else NAN
    comb = d.filter((pl.col("domain") == "combine") & (pl.col("role") == "held_out"))
    for kind in KINDS:
        g = comb.filter(pl.col("kind") == kind)
        if g.height < 20:
            continue
        med = float(g["n_valid"].median())
        for half, sub in (("short", g.filter(pl.col("n_valid") <= med)), ("long", g.filter(pl.col("n_valid") > med))):
            r = repeat_auc(sub.drop("n_valid"), label, (kind,))
            out[f"repeat_auc_{kind}_{half}"] = float(r["repeat_auc"][0]) if r.height else NAN
    return out


def representation_rows(emb: Any, label: dict[str, Any], seed: int, index: Any) -> dict[str, Any]:
    """Geometry, probes, ICC summary and duration sensitivity of one fold table (held-out rows)."""
    from ..m15 import latent as LT

    g = LT.geometry(emb, label)
    p = LT.probes(emb, label)
    icc = LT.icc_summary(LT.dim_icc(emb, label, KINDS))
    rec = {**g, **{k: v for k, v in p.items() if k not in label}}
    if icc.height:
        for r in icc.iter_rows(named=True):
            rec[f"median_dim_icc_{r['kind']}"] = r["median_dim_icc"]
            rec[f"share_dims_icc_ge_0_6_{r['kind']}"] = r["share_dims_ge_0_6"]
    rec.update({k: v for k, v in duration_sensitivity(emb, label, index).items() if k not in label})
    return rec


__all__ = ["KINDS", "damage_summary", "duration_sensitivity", "fold_rows", "mapped_table", "representation_rows"]
