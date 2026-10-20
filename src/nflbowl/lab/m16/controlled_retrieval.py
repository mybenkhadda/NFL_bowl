"""E16.B5 — controlled five-fold cross-domain athlete retrieval for EVERY verified fold-safe representation, its fusion and
the controls, on identical folds and candidate pools.

Unit = held-out athlete of fold f with both a Combine and an NFL athlete-mean embedding from the fold-f model (dual
domain observed). Query = Combine mean over the attempts of a drill SCOPE (all / forty / three_cone / short_shuttle);
gallery = NFL means (all NFL windows) of held-out athletes in the query's POOL:

    all             every held-out dual-domain athlete of the fold
    same_position   same position group
    twins           same position and |Δforty| ≤ ``twin_forty_s`` (stopwatch twins)
    physical_twins  same position, |Δheight| ≤ ``twin_height_in`` and |Δweight| ≤ ``twin_weight_lb`` (similar physical profile)

Representations: each VERIFIED_FOLD_SAFE M15 representation (E16.B4), STFT ⊕ LSTM fusion (train-standardised blocks,
``m15.transfer.concat_embeddings``), and the controls F0 (engineered forty statistics), ``kinematic`` (simple per-window
speed / acceleration summaries, z-scored with training rows of the fold), ``position_only`` (analytic: position recovered
perfectly, uniform within position) and the ``random`` arm (Gaussian vectors). Anthropometrics are never used as a
similarity (they identify the athlete in both domains by construction); they only define the physical_twins pool.

Arms: unaligned, Procrustes and InfoNCE — anything fitted uses TRAINING athletes of the fold (``m15.transfer``).

Per query: rank of the true NFL mean (cosine), reciprocal rank, top-1, top-5, AUC, pool size, the top-1 retrieved athlete
(error gallery: same position? |Δforty|? |Δweight|?) and similarity distributions (true vs sampled impostors). Chance is
pool-specific (``stats.chance_*`` from each query's candidate count). The decisive comparison is each representation vs the
STRONGEST relevant control on the SAME queries (paired athlete bootstrap; Holm across representations × arms per pool /
scope), not vs uniform global chance.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..status import BLOCKED_INSUFFICIENT_SAMPLE, ResearchBlocked

NAN = float("nan")
POOLS16 = ("all", "same_position", "twins", "physical_twins")
CONTROLS = ("F0", "kinematic", "position_only")
TRAIN_ROLES = ("train", "validation")


# ====================================================================== inputs
def attributes(athletes: Any) -> dict[str, dict[int, float]]:
    import polars as pl

    out = {}
    for c in ("forty", "height", "weight"):
        if athletes is not None and c in athletes.columns:
            a = athletes.select(pl.col("nfl_id").cast(pl.Int64), pl.col(c).cast(pl.Float64)).drop_nulls()
            out[c] = dict(zip(a["nfl_id"].to_list(), a[c].to_list(), strict=True))
        else:
            out[c] = {}
    return out


def kinematic_embedding(feat: Any, index: Any, fold: int, cols: Sequence[str]) -> Any:
    """A fold 'embedding table' of per-window kinematics (index order): NaN → training median, z-score with training rows."""
    import polars as pl

    cols = [c for c in cols if c in feat.columns]
    d = index.select("idx", "nfl_id", "domain", "kind", "position_group", "fold").join(feat.select("idx", *cols), on="idx", how="left")
    X = d.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy().astype(float)
    tr = d["fold"].to_numpy() != fold
    for j in range(X.shape[1]):
        x = X[:, j]
        med = np.nanmedian(x[tr]) if np.isfinite(x[tr]).any() else 0.0
        x = np.where(np.isfinite(x), x, med)
        mu, sd = x[tr].mean(), x[tr].std()
        X[:, j] = (x - mu) / (sd if sd > 1e-12 else 1.0)
    role = np.where(~tr, "held_out", "train")
    return d.select("idx", "nfl_id", "domain", "kind", "position_group", "fold").with_columns(pl.Series("role", role),
                                                                                            *[pl.Series(f"z{j}", X[:, j]) for j in range(X.shape[1])])


def means(emb: Any, roles: Sequence[str], kinds: Sequence[str] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(ids, positions, Combine means over ``kinds``, NFL means over all windows) of dual athletes in ``roles``; athletes
    with a non-finite mean are dropped."""
    from ..m15.latent import athlete_means, zcols

    zc = zcols(emb)
    c = athlete_means(emb, "combine", kinds, roles)
    n = athlete_means(emb, "nfl", None, roles)
    j = c.select("nfl_id", "position_group", *zc).join(n.select("nfl_id", *zc), on="nfl_id", suffix="_n").sort("nfl_id")
    C = j.select(zc).to_numpy().astype(float)
    N = j.select([f"{x}_n" for x in zc]).to_numpy().astype(float)
    ok = np.isfinite(C).all(1) & np.isfinite(N).all(1)
    return j["nfl_id"].to_numpy()[ok], j["position_group"].fill_null("?").to_numpy()[ok], C[ok], N[ok]


def pools(ids: np.ndarray, pos: np.ndarray, attrs: dict[str, dict[int, float]], cfg: Any) -> dict[str, list[np.ndarray]]:
    from ..m15.transfer import pool_members

    out = {p: pool_members(ids, pos, p, attrs["forty"], cfg.twin_forty_s) for p in ("all", "same_position", "twins")}
    h = np.array([attrs["height"].get(int(i), np.nan) for i in ids])
    w = np.array([attrs["weight"].get(int(i), np.nan) for i in ids])
    phys = []
    for i in range(ids.size):
        if not (np.isfinite(h[i]) and np.isfinite(w[i])):
            phys.append(np.zeros(0, np.int64))
            continue
        m = (pos == pos[i]) & np.isfinite(h) & np.isfinite(w) & (np.abs(h - h[i]) <= cfg.twin_height_in) & (np.abs(w - w[i]) <= cfg.twin_weight_lb)
        m[i] = False
        phys.append(np.flatnonzero(m))
    out["physical_twins"] = phys
    return out


# ====================================================================== scoring
def _unit(Z: np.ndarray) -> np.ndarray:
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def arm_transform(arm: str, Ct: np.ndarray, Nt: np.ndarray, d: int, seed: int) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]] | None:
    """(map for Combine queries, map for NFL gallery) fitted on TRAINING athletes, or None when the arm cannot be fitted."""
    from ..m15 import transfer as TR

    ident = (lambda X: X)
    if arm == "unaligned":
        return ident, ident
    if arm == "procrustes":
        return (TR.procrustes_map(Ct, Nt), ident) if Ct.shape[0] >= d + 5 else None
    if arm == "infonce":
        if Ct.shape[0] < 30:
            return None
        try:
            return TR.infonce_heads(Ct, Nt, d, seed)
        except ResearchBlocked:
            return None
    if arm == "random":
        rng = np.random.default_rng(seed + 17)
        return (lambda X: rng.normal(size=X.shape)), (lambda X: rng.normal(size=X.shape))
    raise ValueError(arm)


def score_fold(emb: Any, label: dict[str, Any], kinds: Sequence[str] | None, attrs: dict[str, dict[int, float]], cfg: Any, arms: Sequence[str], seed: int,
               keep_sims: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(per-query rows, similarity-distribution rows, pool-metadata rows) of one embedding table and scope."""
    from ..m15 import transfer as TR

    ids, pos, C, N = means(emb, ("held_out",), kinds)
    if ids.size < 10:
        raise ResearchBlocked(BLOCKED_INSUFFICIENT_SAMPLE, f"{ids.size} held-out dual-domain athletes ({label})")
    _, _, Ct, Nt = means(emb, TRAIN_ROLES, kinds)
    pl_ = pools(ids, pos, attrs, cfg)
    meta = [{**{k: v for k, v in label.items() if k != "arm"}, "pool": p, "nfl_id": int(ids[i]), "position_group": pos[i], "n_candidates": int(m[i].size)}
            for p, m in pl_.items() for i in range(ids.size)]
    rows, sims = [], []
    rng = np.random.default_rng(seed + 991)
    forty = np.array([attrs["forty"].get(int(i), np.nan) for i in ids])
    weight = np.array([attrs["weight"].get(int(i), np.nan) for i in ids])
    for arm in arms:
        maps = arm_transform(arm, Ct, Nt, C.shape[1], seed)
        if maps is None:
            continue
        Q, G = maps[0](C), maps[1](N)
        lab = {**label, "arm": arm}
        recs = TR.score_queries(Q, G, ids, pos, pl_, lab)
        S = _unit(np.asarray(Q, float)) @ _unit(np.asarray(G, float)).T
        k = 0
        for p, members in pl_.items():
            for i in range(ids.size):
                cand = members[i]
                if cand.size == 0:
                    continue
                allc = np.concatenate([[i], cand])
                top = int(allc[int(np.argmax(S[i, allc]))])
                r = recs[k]
                k += 1
                r.update({"top1_id": int(ids[top]), "top1_same_position": bool(pos[top] == pos[i]),
                          "top1_abs_dforty": float(abs(forty[top] - forty[i])) if top != i else 0.0,
                          "top1_abs_dweight": float(abs(weight[top] - weight[i])) if top != i else 0.0, "true_sim": float(S[i, i]),
                          "best_impostor_sim": float(S[i, cand].max())})
                if keep_sims and p == "same_position":
                    imp = S[i, cand]
                    pick = imp if imp.size <= cfg.impostors_per_query else imp[rng.choice(imp.size, cfg.impostors_per_query, replace=False)]
                    sims.append({**lab, "pool": p, "nfl_id": int(ids[i]), "pair": "match", "similarity": float(S[i, i])})
                    sims += [{**lab, "pool": p, "nfl_id": int(ids[i]), "pair": "impostor", "similarity": float(s)} for s in pick]
        rows += recs
    return rows, sims, meta


def position_only_rows(scores: Any) -> Any:
    """Analytic position-only control on the same queries: all pool → E[1/rank] for a uniform pick among the m same-position
    athletes (m = same-position candidates + 1); within-position pools → uniform chance of that pool."""
    import polars as pl

    base = scores.filter(pl.col("arm") == "unaligned").select("rep", "fold", "scope", "pool", "nfl_id", "position_group", "n_candidates").unique()
    same = base.filter(pl.col("pool") == "same_position").select("rep", "fold", "scope", "nfl_id", pl.col("n_candidates").alias("n_same"))
    b = base.join(same, on=["rep", "fold", "scope", "nfl_id"], how="left")
    m = np.where(b["pool"].to_numpy() == "all", b["n_same"].fill_null(0).to_numpy() + 1, b["n_candidates"].to_numpy() + 1).astype(int)
    h = np.array([np.sum(1.0 / np.arange(1, k + 1)) for k in m])
    n_all = b["n_candidates"].to_numpy() + 1
    return b.select("rep", "fold", "scope", "pool", "nfl_id", "position_group", "n_candidates").with_columns(
        pl.lit("analytic").alias("arm"), pl.Series("rr", h / m), pl.Series("top1", 1.0 / m), pl.Series("top5", np.minimum(5, m) / m),
        pl.Series("auc", np.where(b["pool"].to_numpy() == "all", 1 - (m - 1) / (2 * np.maximum(n_all - 1, 1)), 0.5)),
        pl.Series("rank", (m + 1) / 2.0)).with_columns(pl.lit("position_only").alias("rep_control"))


# ====================================================================== summaries and tests
def summarise(scores: Any, n_boot: int, seed: int) -> Any:
    """Per (rep, arm, scope, pool): metrics with athlete-bootstrap CIs and the pool-specific chance references."""
    import polars as pl

    from ..stats import chance_median_rank, chance_mrr, chance_recall_at_k

    rng = np.random.default_rng(seed)
    out = []
    for key, g in scores.group_by("rep", "arm", "scope", "pool"):
        n = g["n_candidates"].to_numpy()
        rr, t1, auc, rk = (g[c].to_numpy().astype(float) for c in ("rr", "top1", "auc", "rank"))
        idx = rng.integers(0, g.height, (n_boot, g.height)) if g.height >= 10 else None
        ci = (lambda v: (float(np.quantile(v[idx].mean(1), 0.025)), float(np.quantile(v[idx].mean(1), 0.975)))) if idx is not None else (lambda v: (NAN, NAN))
        mlo, mhi = ci(rr)
        tlo, thi = ci(t1)
        alo, ahi = ci(auc)
        med_pool = float(np.median(n)) if n.size else NAN
        out.append({**dict(zip(("rep", "arm", "scope", "pool"), key, strict=True)), "n_queries": g.height, "median_pool": med_pool,
                    "top1": float(t1.mean()), "top1_lo": tlo, "top1_hi": thi, "chance_top1": chance_recall_at_k(1, n),
                    "top5": float(g["top5"].mean()) if med_pool >= 10 else None, "chance_top5": chance_recall_at_k(5, n) if med_pool >= 10 else None,
                    "mrr": float(rr.mean()), "mrr_lo": mlo, "mrr_hi": mhi, "chance_mrr": chance_mrr(n), "median_rank": float(np.median(rk)),
                    "chance_median_rank": chance_median_rank(n), "auc": float(auc.mean()), "auc_lo": alo, "auc_hi": ahi,
                    "mrr_lift_over_chance": float(rr.mean()) / chance_mrr(n) if n.size and chance_mrr(n) > 0 else NAN})
    return pl.DataFrame(out, infer_schema_length=None).sort("scope", "pool", "rep", "arm")


def _paired(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    d = a - b
    bs = d[rng.integers(0, d.size, (n_boot, d.size))].mean(1)
    return {"delta_mrr": float(d.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "p_boot_le0": float((bs <= 0).mean())}


def paired_vs_controls(scores: Any, pos_only: Any, n_boot: int, seed: int, min_n: int = 10) -> Any:
    """Each non-control (rep, arm) vs every control on the SAME queries (per scope × pool): ΔMRR with an athlete bootstrap.
    ``strongest`` marks the control with the highest MRR on those queries; Holm (over rep × arm within scope × pool) is
    applied to the strongest-control tests only."""
    import polars as pl

    from ... import _backend

    ae = _backend.ae()
    rng = np.random.default_rng(seed)
    keys = ["fold", "scope", "pool", "nfl_id"]
    ctrl_frames = {f"{r}·{a}": scores.filter((pl.col("rep") == r) & (pl.col("arm") == a)).select(*keys, pl.col("rr").alias("rr_c"))
                   for r, a in scores.filter(pl.col("rep").is_in(["F0", "kinematic"])).select("rep", "arm").unique().rows()}
    if pos_only is not None and pos_only.height:
        ctrl_frames["position_only·analytic"] = pos_only.select(*keys, pl.col("rr").alias("rr_c")).unique(subset=keys)
    subjects = scores.filter(~pl.col("rep").is_in(["F0", "kinematic"]) & (pl.col("arm") != "random")).select("rep", "arm").unique().rows()
    rows = []
    for rep, arm in subjects:
        s = scores.filter((pl.col("rep") == rep) & (pl.col("arm") == arm)).select(*keys, "rr")
        for (scope, pool), g in s.group_by("scope", "pool"):
            cand = []
            for cname, cf in ctrl_frames.items():
                j = g.join(cf, on=keys, how="inner")
                if j.height < min_n:
                    continue
                r = _paired(j["rr"].to_numpy(), j["rr_c"].to_numpy(), n_boot, rng)
                cand.append({"rep": rep, "arm": arm, "scope": scope, "pool": pool, "control": cname, "n": j.height, "mrr": float(j["rr"].mean()),
                             "mrr_control": float(j["rr_c"].mean()), **r})
            if cand:
                best = max(cand, key=lambda c: c["mrr_control"])
                for c in cand:
                    c["strongest"] = c is best
                rows += cand
    out = pl.DataFrame(rows, infer_schema_length=None)
    if not out.height:
        return out
    parts = []
    for _, g in out.group_by("scope", "pool"):
        st = g.filter(pl.col("strongest"))
        holm = dict(zip(zip(st["rep"].to_list(), st["arm"].to_list()), ae.holm(st["p_boot_le0"].to_list()))) if st.height else {}
        parts.append(g.with_columns(pl.Series("p_holm_strongest", [holm.get((r, a)) if s else None for r, a, s in
                                                                   zip(g["rep"].to_list(), g["arm"].to_list(), g["strongest"].to_list())], dtype=pl.Float64)))
    return pl.concat(parts, how="vertical_relaxed").sort("scope", "pool", "rep", "arm", "control")


def foldwise(scores: Any) -> Any:
    import polars as pl

    return scores.group_by("rep", "arm", "scope", "pool", "fold").agg(pl.len().alias("n_queries"), pl.col("rr").mean().alias("mrr"),
                                                                     pl.col("top1").mean().alias("top1"), pl.col("auc").mean().alias("auc"),
                                                                     pl.col("n_candidates").median().alias("median_pool")).sort("rep", "arm", "scope", "pool", "fold")


def within_position(scores: Any) -> Any:
    import polars as pl

    return scores.group_by("rep", "arm", "scope", "pool", "position_group").agg(pl.len().alias("n_queries"), pl.col("rr").mean().alias("mrr"),
                                                                               pl.col("top1").mean().alias("top1"), pl.col("auc").mean().alias("auc"),
                                                                               pl.col("n_candidates").median().alias("median_pool"))


def error_summary(scores: Any) -> Any:
    """Top-1 errors: share retrieving another position, and the mistaken athlete's |Δforty| / |Δweight| (stopwatch / size
    driven errors look like small deltas)."""
    import polars as pl

    e = scores.filter(pl.col("top1") < 1)
    if not e.height:
        return pl.DataFrame()
    return e.group_by("rep", "arm", "scope", "pool").agg(pl.len().alias("n_errors"), (~pl.col("top1_same_position")).mean().alias("share_other_position"),
                                                        pl.col("top1_abs_dforty").median().alias("median_abs_dforty_error"),
                                                        pl.col("top1_abs_dweight").median().alias("median_abs_dweight_error"),
                                                        (pl.col("best_impostor_sim") - pl.col("true_sim")).median().alias("median_margin_lost"))


def gallery(scores: Any, rep: str, arm: str = "unaligned", pool: str = "same_position", scope: str = "all", k: int = 12) -> Any:
    """Best matched and worst mismatched queries of one representation (rank, true vs best-impostor similarity)."""
    import polars as pl

    s = scores.filter((pl.col("rep") == rep) & (pl.col("arm") == arm) & (pl.col("pool") == pool) & (pl.col("scope") == scope))
    cols = ["fold", "nfl_id", "position_group", "n_candidates", "rank", "true_sim", "best_impostor_sim", "top1_id", "top1_abs_dforty", "top1_abs_dweight"]
    return pl.concat([s.sort("rank").head(k).select(cols).with_columns(pl.lit("matched").alias("case")),
                      s.sort("rank", descending=True).head(k).select(cols).with_columns(pl.lit("mismatched").alias("case"))], how="vertical_relaxed")


def similarity_matrix(emb: Any, kinds: Sequence[str] | None, position: str) -> tuple[np.ndarray, np.ndarray]:
    """Held-out Combine-mean × NFL-mean cosine matrix for one position (diagonal = true pairs), ids sorted."""
    ids, pos, C, N = means(emb, ("held_out",), kinds)
    m = pos == position
    return ids[m], _unit(C[m]) @ _unit(N[m]).T


__all__ = ["CONTROLS", "POOLS16", "arm_transform", "attributes", "error_summary", "foldwise", "gallery", "kinematic_embedding", "means", "paired_vs_controls",
           "pools", "position_only_rows", "score_fold", "similarity_matrix", "summarise", "within_position"]
