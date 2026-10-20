"""E15.D1–D3 — latent-space analytics of the exported fold embeddings (T08).

All read-outs use the HELD-OUT rows of each fold model (role == held_out), so every number describes athletes the model
never saw. Training-role rows are used only to FIT a probe / an alignment, never to score it.

    geometry          effective rank, participation ratio, PC1 share (``stats.effective_rank``), collapsed-dimension share,
                      norm distribution, within- / between-athlete variance ratio
    probes            domain AUC (ridge discriminant fitted on training rows) and position balanced accuracy (nearest
                      training centroid) — high values mean the latent space encodes domain / position
    dim_icc           ICC(1,1) of every latent dimension across repeated Combine attempts of one drill (held-out athletes,
                      ≥ ``min_athletes`` with ≥ 2 attempts), with the G15.c summary (median ICC, share ≥ 0.6)
    repeat_auc        within-position, within-drill attempt identification: P(distance between two attempts of the same
                      athlete < distance to another athlete's attempt) — reference value of Milestone 04 TCN-B (0.873) is
                      historical context only
    seed_stability    athlete-mean embeddings of two seeds aligned by orthogonal Procrustes (fitted on TRAINING athletes,
                      scored on held-out athletes): mean per-dimension Pearson r (G15.e)
    interpretation    partial Spearman of each held-out athlete-mean dimension with simple tracking features controlling for
                      position (``stats.partial_spearman``), BH-adjusted; a dimension may be described in words only when
                      |ρ| ≥ 0.5 with a CI excluding 0.3 AND the relation is stable across seeds (checked in ``labels``)
    projection        2-D PCA (+ UMAP when installed) of athlete-weighted held-out rows (descriptive only)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAN = float("nan")


def zcols(emb: Any) -> list[str]:
    return [c for c in emb.columns if c.startswith("z") and c[1:].isdigit()]


def athlete_means(emb: Any, domain: str | None = None, kinds: Sequence[str] | None = None, roles: Sequence[str] | None = ("held_out",)) -> Any:
    import polars as pl

    d = emb
    if domain is not None:
        d = d.filter(pl.col("domain") == domain)
    if kinds is not None:
        d = d.filter(pl.col("kind").is_in(list(kinds)))
    if roles is not None:
        d = d.filter(pl.col("role").is_in(list(roles)))
    zc = zcols(emb)
    return d.group_by("nfl_id").agg(pl.col("position_group").first(), pl.len().alias("n_seq"), *[pl.col(c).mean() for c in zc]).sort("nfl_id")


# ====================================================================== D1
def geometry(emb: Any, label: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    from ..stats import effective_rank

    h = emb.filter(pl.col("role") == "held_out")
    zc = zcols(emb)
    Z = h.select(zc).to_numpy().astype(float)
    if Z.shape[0] < 10:
        return {**label, "n_rows": Z.shape[0]}
    sd = Z.std(0)
    er = effective_rank(Z)
    nrm = np.linalg.norm(Z, axis=1)
    ids = h["nfl_id"].to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    means = np.stack([Z[inv == k].mean(0) for k in range(u.size)])
    within = float(np.mean([Z[inv == k].var(0).mean() for k in range(u.size) if (inv == k).sum() > 1])) if u.size else NAN
    between = float(means.var(0).mean())
    d = len(zc)
    return {**label, "n_rows": int(Z.shape[0]), "n_athletes": int(u.size), **er, "effective_rank_over_d": er["effective_rank"] / d if d else NAN,
            "collapsed_dim_share": float((sd < 1e-4 * max(float(sd.max()), 1e-12)).mean()), "norm_median": float(np.median(nrm)),
            "norm_iqr": float(np.subtract(*np.quantile(nrm, [0.75, 0.25]))), "within_athlete_var": within, "between_athlete_var": between,
            "between_over_within": between / within if within and np.isfinite(within) and within > 0 else NAN}


def _auc(scores: np.ndarray, pos: np.ndarray) -> float:
    s1, s0 = scores[pos], scores[~pos]
    if s1.size < 5 or s0.size < 5:
        return NAN
    allv = np.concatenate([s1, s0])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((ranks[: s1.size].sum() - s1.size * (s1.size + 1) / 2) / (s1.size * s0.size))


def probes(emb: Any, label: dict[str, Any], alpha: float = 1.0) -> dict[str, Any]:
    """Domain AUC (ridge discriminant) and position balanced accuracy (nearest centroid), fitted on training rows."""
    import polars as pl

    zc = zcols(emb)
    tr = emb.filter(pl.col("role") == "train")
    te = emb.filter(pl.col("role") == "held_out")
    if tr.height < 50 or te.height < 20:
        return {**label, "domain_auc": NAN, "position_bal_acc": NAN}
    Xt, Xe = tr.select(zc).to_numpy().astype(float), te.select(zc).to_numpy().astype(float)
    mu, sd = Xt.mean(0), Xt.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Xt, Xe = (Xt - mu) / sd, (Xe - mu) / sd
    yt = np.where(tr["domain"].to_numpy() == "nfl", 1.0, -1.0)
    beta = np.linalg.solve(Xt.T @ Xt + alpha * np.eye(Xt.shape[1]), Xt.T @ (yt - yt.mean()))
    auc = _auc(Xe @ beta, te["domain"].to_numpy() == "nfl")
    pt, pe = tr["position_group"].fill_null("?").to_numpy(), te["position_group"].fill_null("?").to_numpy()
    cents = {p: Xt[pt == p].mean(0) for p in np.unique(pt)}
    keys = list(cents)
    C = np.stack([cents[k] for k in keys])
    pred = np.array(keys)[np.argmin(((Xe[:, None, :] - C[None]) ** 2).sum(-1), axis=1)]
    accs = [float(np.mean(pred[pe == p] == p)) for p in np.unique(pe) if (pe == p).sum() >= 5]
    return {**label, "domain_auc": auc, "position_bal_acc": float(np.mean(accs)) if accs else NAN, "position_chance": 1.0 / max(len(keys), 1)}


# ====================================================================== D2
def dim_icc(emb: Any, label: dict[str, Any], kinds: Sequence[str] = ("forty", "three_cone", "short_shuttle"), min_athletes: int = 30,
            n_boot: int = 0, seed: int = 0) -> Any:
    import polars as pl

    from ..reliability import icc_bootstrap, icc_components

    h = emb.filter((pl.col("role") == "held_out") & (pl.col("domain") == "combine"))
    rows = []
    for kd in kinds:
        g = h.filter(pl.col("kind") == kd)
        if not g.height:
            continue
        ids = g["nfl_id"].to_numpy()
        for c in zcols(emb):
            comp = icc_components(g[c].to_numpy().astype(float), ids)
            lo = hi = NAN
            if n_boot and comp["n_athletes"] >= min_athletes:
                lo, hi = icc_bootstrap(g[c].to_numpy().astype(float), ids, n_boot, seed)
            rows.append({**label, "kind": kd, "dim": c, "n_athletes": comp["n_athletes"], "icc1": comp["icc1"], "icc_lo": lo, "icc_hi": hi,
                         "enough": comp["n_athletes"] >= min_athletes})
    return pl.DataFrame(rows, infer_schema_length=None)


def icc_summary(icc: Any) -> Any:
    import polars as pl

    if icc is None or not icc.height:
        return pl.DataFrame()
    keys = [c for c in ("arch", "d", "seed", "fold", "kind") if c in icc.columns]
    return icc.filter(pl.col("enough")).group_by(keys).agg(pl.col("icc1").median().alias("median_dim_icc"), (pl.col("icc1") >= 0.6).mean().alias("share_dims_ge_0_6"),
                                                            pl.len().alias("n_dims"), pl.col("n_athletes").first())


def repeat_auc(emb: Any, label: dict[str, Any], kinds: Sequence[str] = ("forty", "three_cone", "short_shuttle")) -> Any:
    """Within-position, within-drill attempt identification AUC on held-out Combine attempts (cosine distance)."""
    import polars as pl

    h = emb.filter((pl.col("role") == "held_out") & (pl.col("domain") == "combine"))
    zc = zcols(emb)
    rows = []
    for kd in kinds:
        g = h.filter(pl.col("kind") == kd)
        pos_d, neg_d = [], []
        for _, gp in g.group_by("position_group"):
            Z = gp.select(zc).to_numpy().astype(float)
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            ids = gp["nfl_id"].to_numpy()
            if Z.shape[0] < 4:
                continue
            D = 1 - Z @ Z.T
            iu = np.triu_indices(Z.shape[0], 1)
            same = ids[iu[0]] == ids[iu[1]]
            pos_d.append(D[iu][same])
            neg_d.append(D[iu][~same])
        pd_, nd_ = (np.concatenate(pos_d) if pos_d else np.zeros(0)), (np.concatenate(neg_d) if neg_d else np.zeros(0))
        auc = _auc(-np.concatenate([pd_, nd_]), np.concatenate([np.ones(pd_.size, bool), np.zeros(nd_.size, bool)])) if pd_.size >= 5 else NAN
        rows.append({**label, "kind": kd, "n_same_pairs": int(pd_.size), "n_diff_pairs": int(nd_.size), "repeat_auc": auc})
    return pl.DataFrame(rows, infer_schema_length=None)


def procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||B R − A|| (both centred)."""
    u, _, vt = np.linalg.svd(B.T @ A)
    return u @ vt


def seed_stability(emb_a: Any, emb_b: Any, label: dict[str, Any]) -> dict[str, Any]:
    """Procrustes fitted on TRAINING athletes' means, scored on held-out athletes' means (mean per-dimension r)."""
    import polars as pl

    zc = zcols(emb_a)
    out = {}
    for role in ("train", "held_out"):
        a = athlete_means(emb_a, roles=(role,))
        b = athlete_means(emb_b, roles=(role,))
        j = a.select("nfl_id", *zc).join(b.select("nfl_id", *zc), on="nfl_id", suffix="_b")
        out[role] = (j.select(zc).to_numpy().astype(float), j.select([f"{c}_b" for c in zc]).to_numpy().astype(float))
    (At, Bt), (Ah, Bh) = out["train"], out["held_out"]
    if At.shape[0] < 10 or Ah.shape[0] < 10:
        return {**label, "seed_r": NAN, "n_athletes": int(Ah.shape[0])}
    mA, mB = At.mean(0), Bt.mean(0)
    R = procrustes(At - mA, Bt - mB)
    Bh2 = (Bh - mB) @ R + mA
    rs = [np.corrcoef(Ah[:, k], Bh2[:, k])[0, 1] for k in range(Ah.shape[1]) if Ah[:, k].std() > 0 and Bh2[:, k].std() > 0]
    return {**label, "seed_r": float(np.mean(rs)) if rs else NAN, "seed_r_min": float(np.min(rs)) if rs else NAN, "n_athletes": int(Ah.shape[0])}


# ====================================================================== D3
def interpretation(emb: Any, athlete_features: Any, label: dict[str, Any], kinds: Sequence[str] = ("forty",), n_boot: int = 200, seed: int = 0) -> Any:
    """Partial Spearman (position-controlled) of held-out athlete-mean dimensions with athlete features; BH; bootstrap CI."""
    import polars as pl

    from ..stats import bh, partial_spearman

    a = athlete_means(emb, domain="combine", kinds=kinds)
    j = a.join(athlete_features, on="nfl_id", how="inner")
    if j.height < 30:
        return pl.DataFrame()
    pos = j["position_group"].fill_null("?").to_numpy()
    P = np.stack([(pos == p).astype(float) for p in np.unique(pos)[1:]], 1) if np.unique(pos).size > 1 else np.zeros((j.height, 0))
    feats = [c for c in athlete_features.columns if c != "nfl_id"]
    rng = np.random.default_rng(seed)
    rows = []
    for c in zcols(emb):
        x = j[c].to_numpy().astype(float)
        for f in feats:
            y = j[f].cast(pl.Float64).to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 30:
                continue
            r = partial_spearman(x[ok], y[ok], P[ok])
            n = int(ok.sum())
            bs = []
            for _ in range(n_boot):
                i = np.flatnonzero(ok)[rng.integers(0, n, n)]
                bs.append(partial_spearman(x[i], y[i], P[i]))
            bs = np.asarray([b for b in bs if np.isfinite(b)])
            p = float(np.mean(np.sign(bs) != np.sign(r))) * 2 if bs.size else NAN
            rows.append({**label, "dim": c, "feature": f, "n": n, "partial_spearman": r, "lo": float(np.quantile(bs, 0.025)) if bs.size else NAN,
                         "hi": float(np.quantile(bs, 0.975)) if bs.size else NAN, "p_boot": min(1.0, p) if np.isfinite(p) else NAN})
    out = pl.DataFrame(rows, infer_schema_length=None)
    if out.height:
        out = out.with_columns(pl.Series("p_bh", bh([1.0 if not np.isfinite(v) else v for v in out["p_boot"].to_list()])))
    return out


def labels(interp_by_seed: Any) -> Any:
    """A (dim, feature) relation may name a dimension only if |ρ| ≥ 0.5, the CI excludes 0.3 in magnitude and the sign
    agrees in every seed. Dimensions are not aligned across seeds, so seed agreement is checked on the best-matching
    feature per dimension and reported, never assumed."""
    import polars as pl

    if interp_by_seed is None or not interp_by_seed.height:
        return pl.DataFrame()
    t = interp_by_seed.with_columns(((pl.col("partial_spearman").abs() >= 0.5) & ((pl.col("lo") > 0.3) | (pl.col("hi") < -0.3))).alias("strong"))
    best = t.sort(pl.col("partial_spearman").abs(), descending=True).group_by("arch", "d", "seed", "dim", maintain_order=True).first()
    stab = best.filter(pl.col("strong")).group_by("arch", "d", "feature").agg(pl.col("seed").n_unique().alias("seeds_with_strong_dim"),
                                                                              pl.col("partial_spearman").sign().n_unique().alias("n_signs"))
    return stab.with_columns(((pl.col("n_signs") == 1)).alias("sign_consistent"))


def projection(emb: Any, label: dict[str, Any], per_athlete: int = 5, max_points: int = 6000, seed: int = 0, use_umap: bool = True) -> Any:
    """2-D PCA (+ UMAP) coordinates of held-out rows, at most ``per_athlete`` rows per athlete and domain (descriptive)."""
    import polars as pl

    h = emb.filter(pl.col("role") == "held_out").with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over("nfl_id", "domain").alias("_r")).filter(
        pl.col("_r") < per_athlete).drop("_r")
    if h.height > max_points:
        h = h.sample(max_points, seed=seed)
    Z = h.select(zcols(emb)).to_numpy().astype(float)
    Zc = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    xy = Zc @ vt[:2].T
    out = h.select("idx", "nfl_id", "domain", "kind", "position_group").with_columns(pl.Series("pc1", xy[:, 0]), pl.Series("pc2", xy[:, 1] if xy.shape[1] > 1 else 0.0))
    if use_umap:
        try:
            import umap

            U = umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
            out = out.with_columns(pl.Series("umap1", U[:, 0]), pl.Series("umap2", U[:, 1]))
        except ImportError:
            pass
    return out.with_columns(*[pl.lit(v).alias(k) for k, v in label.items()])


__all__ = ["athlete_means", "dim_icc", "geometry", "icc_summary", "interpretation", "labels", "probes", "procrustes", "projection", "repeat_auc", "seed_stability",
           "zcols"]
