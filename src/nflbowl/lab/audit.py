"""Part B — dataset characterization, sample attrition and the leakage audit.

Everything is computed from the bundle (exact counts, no sampling) and the per-sequence metric table
(``biomech.sequence_table``). Statistics over sequences use the athlete-cluster bootstrap. The leakage audit separates
checks executed at runtime from properties verified by code inspection, and never declares the evaluation
leakage-free from athlete ids alone.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from ..data.validation import M1_COLS, TRANSFER_TARGET

NAN = float("nan")
COMPOSITION_METRICS = ("peak_speed", "mean_speed", "peak_accel", "peak_decel", "mean_a_tan", "max_abs_a_n", "heading_rate_sd", "duration_s",
                       "n_valid", "heading_valid_share")


# ====================================================================== B1 dataset audit
def athlete_table(exp: Any, seq: Any) -> Any:
    """One row per athlete: position, fold (and whether it is a fallback fold), sequence counts per domain, drill kinds,
    linked / target / M1 availability."""
    import polars as pl

    b = exp.bundle
    manifest = set(b.folds["nfl_id"].to_list())
    per = seq.group_by("nfl_id").agg(pl.col("position_group").first(), pl.col("fold").first(),
                                     (pl.col("domain") == "combine").sum().alias("n_combine_seq"), (pl.col("domain") == "nfl").sum().alias("n_nfl_seq"),
                                     pl.col("kind").filter(pl.col("domain") == "combine").n_unique().alias("n_combine_kinds"),
                                     (pl.col("kind") == "forty").sum().alias("n_forty_seq"))
    per = per.with_columns(pl.col("nfl_id").is_in(list(manifest)).alias("fold_from_manifest"), (pl.col("n_combine_seq") > 0).alias("has_combine"),
                           (pl.col("n_nfl_seq") > 0).alias("has_nfl"))
    per = per.with_columns((pl.col("has_combine") & pl.col("has_nfl")).alias("both_domains"))
    if b.athletes is not None:
        a = b.athletes.select("nfl_id", "linked", *[c for c in M1_COLS if c in b.athletes.columns],
                              *([TRANSFER_TARGET] if TRANSFER_TARGET in b.athletes.columns else []))
        per = per.join(a, on="nfl_id", how="left").with_columns(pl.col("linked").fill_null(False))
        per = per.with_columns(pl.all_horizontal([pl.col(c).is_not_null() for c in M1_COLS if c in a.columns]).alias("m1_complete"),
                               (pl.col(TRANSFER_TARGET).is_not_null() if TRANSFER_TARGET in a.columns else pl.lit(False)).alias("has_target"))
    return per.sort("nfl_id")


def dataset_overview(exp: Any, seq: Any, ath: Any) -> dict[str, Any]:
    """Tables of Part B1. Keys: overview, position_domain, kinds, valid_frames, per_athlete_counts, missingness, fold_balance, target_by_position."""
    import polars as pl

    s = exp.bundle.summary()
    ov = pl.DataFrame({"quantity": ["sequences", "Combine sequences", "NFL sequences", "athletes", "athletes with Combine", "athletes with NFL",
                                    "athletes with both domains", "frames per sequence", "channels", "athletes in fold manifest",
                                    "athletes with fallback fold (nfl_id % k)", "athletes with frozen target", "athletes with complete M1"],
                       "value": [s["n_sequences"], s["n_combine"], s["n_nfl"], s["n_athletes"], s["n_athletes_combine"], s["n_athletes_nfl"],
                                 s["n_athletes_both_domains"], s["frames"], s["channels"], s["folds"]["athletes_with_manifest_fold"],
                                 s["folds"]["athletes_fallback_fold"], int(ath["has_target"].sum()) if "has_target" in ath.columns else 0,
                                 int(ath["m1_complete"].sum()) if "m1_complete" in ath.columns else 0]})
    pos_dom = seq.group_by("position_group", "domain").agg(pl.len().alias("n_sequences"), pl.col("nfl_id").n_unique().alias("n_athletes")).sort("position_group", "domain")
    kinds = seq.group_by("domain", "kind", "category").agg(pl.len().alias("n_sequences"), pl.col("nfl_id").n_unique().alias("n_athletes")).sort("domain", "kind")
    vf = seq.group_by("domain").agg(pl.col("n_valid").min().alias("min"), pl.col("n_valid").quantile(0.05).alias("q05"), pl.col("n_valid").median().alias("median"),
                                    pl.col("n_valid").quantile(0.95).alias("q95"), pl.col("n_valid").max().alias("max"),
                                    (pl.col("n_valid") == 0).sum().alias("n_zero"), (pl.col("n_valid") < 10).sum().alias("n_below_10"))
    pac = ath.select("nfl_id", "position_group", "n_combine_seq", "n_nfl_seq").unpivot(index=["nfl_id", "position_group"], variable_name="domain",
                                                                                      value_name="n_seq")
    pac_sum = pac.group_by("domain").agg(pl.col("n_seq").min().alias("min"), pl.col("n_seq").quantile(0.25).alias("q25"), pl.col("n_seq").median().alias("median"),
                                         pl.col("n_seq").quantile(0.75).alias("q75"), pl.col("n_seq").max().alias("max"), (pl.col("n_seq") == 0).sum().alias("n_zero"))
    miss_cols = [c for c in (*M1_COLS, TRANSFER_TARGET) if c in ath.columns]
    miss = (ath.filter(pl.col("linked")) if "linked" in ath.columns else ath).group_by("position_group").agg(
        pl.len().alias("n_linked"), *[pl.col(c).is_null().mean().alias(c) for c in miss_cols]).sort("position_group")
    seq_miss = seq.group_by("domain").agg(*[pl.col(m).is_nan().or_(pl.col(m).is_null()).mean().alias(m) for m in
                                            ("accel_rate_0_90", "turn_at_speed", "curvature_p90", "heading_rate_sd", "turn_asymmetry")],
                                          (pl.col("heading_valid_share") < 0.5).mean().alias("heading_valid_lt_half"))
    fb = ath.group_by("fold", "position_group").agg(pl.len().alias("n_athletes"), pl.col("both_domains").sum().alias("n_both_domains"),
                                                    *([pl.col("has_target").sum().alias("n_target")] if "has_target" in ath.columns else [])).sort("fold", "position_group")
    fseq = seq.group_by("fold", "domain").agg(pl.len().alias("n_sequences")).sort("fold", "domain")
    tgt = pl.DataFrame()
    if TRANSFER_TARGET in ath.columns:
        tgt = ath.filter(pl.col(TRANSFER_TARGET).is_not_null()).group_by("position_group").agg(
            pl.len().alias("n"), pl.col(TRANSFER_TARGET).mean().alias("mean"), pl.col(TRANSFER_TARGET).std().alias("sd"),
            pl.col(TRANSFER_TARGET).quantile(0.1).alias("q10"), pl.col(TRANSFER_TARGET).median().alias("median"),
            pl.col(TRANSFER_TARGET).quantile(0.9).alias("q90")).sort("position_group")
    return {"overview": ov, "position_domain": pos_dom, "kinds": kinds, "valid_frames": vf, "per_athlete_counts": pac_sum, "per_athlete_long": pac,
            "missingness_athletes": miss, "missingness_sequences": seq_miss, "fold_balance": fb, "fold_sequences": fseq, "target_by_position": tgt}


def m1_correlations(ath: Any) -> tuple[np.ndarray, list[str]]:
    """Spearman matrix of M1 officials, the frozen target and athlete-level sequence counts (pairwise complete)."""
    from .stats import spearman

    cols = [c for c in (*M1_COLS, TRANSFER_TARGET, "n_combine_seq", "n_nfl_seq") if c in ath.columns]
    X = ath.select(cols).to_numpy().astype(float)
    k = len(cols)
    C = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            C[i, j] = 1.0 if i == j else spearman(X[:, i], X[:, j])
    return C, cols


# ====================================================================== B2 domain composition
def domain_composition(seq: Any, n_boot: int, seed: int, metrics: Sequence[str] = COMPOSITION_METRICS) -> tuple[Any, Any]:
    """(overall, by_position). Overall: per metric Combine vs NFL median, IQR, standardized mean difference and the
    athlete-cluster bootstrap interval of the median difference (NFL − Combine). By position: the same median
    difference within each position and a position-adjusted (athlete-weighted) average difference."""
    import polars as pl

    from .stats import cluster_bootstrap_diff

    rows, prow = [], []
    c = seq.filter(pl.col("domain") == "combine")
    n = seq.filter(pl.col("domain") == "nfl")
    for m in metrics:
        a, ca = n[m].cast(float).to_numpy(), n["nfl_id"].to_numpy()
        b, cb = c[m].cast(float).to_numpy(), c["nfl_id"].to_numpy()
        r = cluster_bootstrap_diff(a, ca, b, cb, np.median, n_boot, seed)
        fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]
        sd = np.sqrt((np.var(fa) + np.var(fb)) / 2) if fa.size and fb.size else NAN
        rows.append({"metric": m, "median_nfl": r["a"], "median_combine": r["b"], "diff_nfl_minus_combine": r["diff"], "lo": r["lo"], "hi": r["hi"],
                     "iqr_nfl": float(np.subtract(*np.quantile(fa, [0.75, 0.25]))) if fa.size else NAN,
                     "iqr_combine": float(np.subtract(*np.quantile(fb, [0.75, 0.25]))) if fb.size else NAN,
                     "smd": float((fa.mean() - fb.mean()) / sd) if sd and np.isfinite(sd) and sd > 0 else NAN, "n_nfl": r["n_a"], "n_combine": r["n_b"],
                     "athletes_nfl": r["clusters_a"], "athletes_combine": r["clusters_b"]})
        diffs, weights = [], []
        for p in sorted(seq["position_group"].drop_nulls().unique().to_list()):
            pa = n.filter(pl.col("position_group") == p)[m].cast(float).to_numpy()
            pb = c.filter(pl.col("position_group") == p)[m].cast(float).to_numpy()
            pa, pb = pa[np.isfinite(pa)], pb[np.isfinite(pb)]
            if pa.size >= 20 and pb.size >= 20:
                d = float(np.median(pa) - np.median(pb))
                w = int(seq.filter(pl.col("position_group") == p)["nfl_id"].n_unique())
                diffs.append(d)
                weights.append(w)
                prow.append({"metric": m, "position_group": p, "median_nfl": float(np.median(pa)), "median_combine": float(np.median(pb)), "diff": d,
                             "n_nfl": int(pa.size), "n_combine": int(pb.size), "athletes": w})
        rows[-1]["position_adjusted_diff"] = float(np.average(diffs, weights=weights)) if diffs else NAN
    return pl.DataFrame(rows, infer_schema_length=None), pl.DataFrame(prow, infer_schema_length=None)


def state_category_composition(seq: Any) -> Any:
    """Share of sequences per movement-state id (start-speed bin × accelerating) and per category, by domain — the
    labels the matched-state / category alignment terms use."""
    import polars as pl

    st = (seq.with_columns(pl.when(pl.col("state_id") >= 0).then((pl.col("state_id") // 2).cast(pl.String) + " yd/s · "
                                                                  + pl.when(pl.col("state_id") % 2 == 1).then(pl.lit("accel")).otherwise(pl.lit("steady")))
                           .otherwise(pl.lit("undefined")).alias("state"))
          .group_by("domain", "state").agg(pl.len().alias("n")).with_columns((pl.col("n") / pl.col("n").sum().over("domain")).alias("share")).sort("domain", "state"))
    cat = (seq.with_columns(pl.col("category").fill_null("unmapped")).group_by("domain", "category").agg(pl.len().alias("n"))
           .with_columns((pl.col("n") / pl.col("n").sum().over("domain")).alias("share")).sort("domain", "category"))
    return st, cat


# ====================================================================== B3 attrition
def attrition(ath: Any) -> tuple[Any, Any]:
    """(steps, by_position). Sequential eligibility for the inductive retrieval and the transfer ladder."""
    import polars as pl

    steps = [("athletes in bundle", pl.lit(True)), ("with Combine sequences", pl.col("has_combine")), ("with NFL sequences", pl.col("has_nfl")),
             ("in fold manifest", pl.col("fold_from_manifest"))]
    if "linked" in ath.columns:
        steps += [("linked (12.0 registry)", pl.col("linked")), ("frozen target present", pl.col("has_target")), ("complete M1 officials", pl.col("m1_complete"))]
    rows, prow, keep = [], [], pl.lit(True)
    prev = ath.height
    for name, cond in steps:
        keep = keep & cond.fill_null(False)
        d = ath.filter(keep)
        rows.append({"step": name, "n_remaining": d.height, "n_removed": prev - d.height})
        prev = d.height
        for p, g in d.group_by("position_group"):
            prow.append({"step": name, "position_group": p[0] if isinstance(p, tuple) else p, "n_remaining": g.height})
    return pl.DataFrame(rows), pl.DataFrame(prow, infer_schema_length=None)


# ====================================================================== B4 leakage audit
def _row(check: str, status: str, detail: str, method: str = "runtime", scope: str = "all") -> dict[str, Any]:
    return {"check": check, "status": status, "detail": detail, "method": method, "scope": scope}


def sequence_hashes(X: np.ndarray, M: np.ndarray, chunk: int = 4096) -> list[str]:
    out = []
    for s in range(0, X.shape[0], chunk):
        x, m = np.asarray(X[s: s + chunk], np.float32), np.asarray(M[s: s + chunk], bool)
        for i in range(x.shape[0]):
            out.append(hashlib.sha1(np.ascontiguousarray(x[i][m[i]]).tobytes()).hexdigest()[:20] if m[i].any() else "empty")
    return out


def duplicate_audit(exp: Any, seq: Any) -> tuple[Any, Any]:
    """Exact duplicate sequences (identical valid frames): groups, and whether a group spans athletes / domains / folds."""
    import polars as pl

    b = exp.bundle
    h = sequence_hashes(b.X, b.M)
    d = seq.select("idx", "nfl_id", "domain", "fold", "kind").with_columns(pl.Series("hash", h))
    g = (d.filter(pl.col("hash") != "empty").group_by("hash").agg(pl.len().alias("n"), pl.col("nfl_id").n_unique().alias("athletes"),
                                                                   pl.col("domain").n_unique().alias("domains"), pl.col("fold").n_unique().alias("folds"),
                                                                   pl.col("idx").alias("rows")).filter(pl.col("n") > 1))
    summ = pl.DataFrame({"quantity": ["duplicate groups", "sequences in duplicate groups", "groups spanning athletes", "groups spanning domains",
                                      "groups spanning folds", "empty sequences"],
                         "value": [g.height, int(g["n"].sum()) if g.height else 0, int((g["athletes"] > 1).sum()) if g.height else 0,
                                   int((g["domains"] > 1).sum()) if g.height else 0, int((g["folds"] > 1).sum()) if g.height else 0,
                                   int(sum(1 for x in h if x == "empty"))]})
    return summ, g


def shortcut_domain_auc(exp: Any, seq: Any, seed: int) -> Any:
    """Domain AUC (athlete-grouped ridge, ``domain_diagnostics.grouped_domain_auc``) of single nuisance features. A
    value near 1 means the domain can be read off a non-movement property — any representation that encodes it (F0
    includes the valid length) separates domains trivially."""
    import polars as pl

    from ..evaluation.domain_diagnostics import grouped_domain_auc

    meta = exp.bundle.meta
    rows = np.arange(meta.height)
    out = []
    for f in ("n_valid", "duration_s", "heading_valid_share", "mean_speed", "peak_speed"):
        E = seq[f].cast(float).fill_nan(None).fill_null(strategy="mean").to_numpy().reshape(-1, 1)
        r = grouped_domain_auc(E, meta, rows, seed=seed)
        auc = r["domain_auc"]
        sep = max(auc, 1 - auc) if np.isfinite(auc) else NAN
        out.append({"feature": f, "domain_auc": auc, "separability": sep, "n_combine": r["n_combine"], "n_nfl": r["n_nfl"],
                    "reading": "trivial shortcut" if sep >= 0.9 else ("partial shortcut" if sep >= 0.7 else "weak")})
    return pl.DataFrame(out)


PREPROCESSING_SCOPE = (
    ("F0 engineered features", "none — fixed per-sequence statistics (rep.engineered_features)", "code inspection"),
    ("physics-head targets", "standardised with TRAINING-sequence mean / SD only (trainer.train_encoder)", "code inspection"),
    ("encoder fitting", "training rows exclude every held-out athlete (both domains); validation = other athletes (se.split_indices)", "runtime (split checks)"),
    ("embedding reduction in the ladder", "PCA fitted on the training athletes of the outer fold (se.embedding_block, ≤ 8 dims)", "code inspection"),
    ("ladder design matrix", "imputation medians, missing indicators, mean / SD and position levels from training rows (m11 ev.design)", "code inspection"),
    ("ridge penalty", "inner athlete CV on training athletes; the fold-fitted block is not refitted per inner fold (mildly optimistic α)", "code inspection"),
    ("retrieval", "cosine similarity, no fitted or tuned parameter; candidate pools from metadata only", "code inspection"),
    ("domain AUC", "standardisation fitted on the inner training athletes (grouped_domain_auc)", "code inspection"),
    ("frozen target nfl.q90_shared", "estimator chosen by a declared NFL-only reliability rule on all athletes BEFORE Combine data were joined (12C); "
     "a target-definition choice, not a predictor — documented, not verifiable from the bundle", "provenance (12C freeze)"),
    ("biomechanical metrics", "deterministic per sequence; Combine-side only in predictor frames (nfl__* columns excluded)", "code inspection"),
)


def leakage_audit(exp: Any, seq: Any, ath: Any, models_folds: dict[str, list[int]], seed: int) -> dict[str, Any]:
    """Runtime checks + documented preprocessing scope. Returns {'checks', 'duplicates', 'duplicate_groups', 'shortcuts', 'scope'}."""
    import polars as pl

    from ..data.splits import SplitLeakError
    from ..evaluation import paired_inductive as pi
    from ..models import registry as reg

    b = exp.bundle
    rows = []
    fmap = exp.folds_by_athlete
    folds = sorted(set(fmap.values()))
    # 1. fold manifest
    rows.append(_row("folds.parquet: one fold per athlete", "pass", f"{b.folds.height} athletes, folds {sorted(b.folds['fold'].drop_nulls().unique().to_list())}"))
    fb = int((~ath["fold_from_manifest"]).sum())
    rows.append(_row("athletes using the nfl_id % k fallback fold", "warn" if fb else "pass", f"{fb} athlete(s) not in the fold manifest",
                     scope="encoder splits"))
    # 2. split integrity per fold
    for f in folds:
        try:
            tr, va, ho = exp.splits(f)
            info = pi.assert_fold_isolation(b.meta, fmap, ho, f, tr, va)
            rows.append(_row("athlete-disjoint train / validation / held-out", "pass",
                             f"held-out athletes {info['held_out_athletes']}, rows {info['held_out_rows']}; train {tr.size}, val {va.size}", scope=f"fold {f}"))
        except SplitLeakError as e:
            rows.append(_row("athlete-disjoint train / validation / held-out", "fail", str(e), scope=f"fold {f}"))
    # 3. ladder folds = encoder folds
    if b.athletes is not None:
        j = b.athletes.select("nfl_id").join(b.folds.select("nfl_id", "fold"), on="nfl_id", how="inner")
        bad = [int(i) for i, f in zip(j["nfl_id"].to_list(), j["fold"].to_list()) if f is not None and fmap.get(int(i), f) != f]
        rows.append(_row("transfer-ladder folds equal encoder folds", "fail" if bad else "pass", f"{len(bad)} mismatching athlete(s)", scope="transfer"))
    # 4. embedding tables / training records
    for name, fl in models_folds.items():
        s = reg.spec(name)
        for f in fl:
            tag = f"inductive_f{f}"
            emb = exp.manifest.find("embeddings", f"{name}__{tag}")
            if emb is not None:
                try:
                    roles = pl.read_parquet(exp.manifest.resolve(emb), columns=["split_role"])["split_role"].to_list()
                    pi.verify_held_out_roles(roles, exp.splits(f)[2], f)
                    rows.append(_row("embedding table marks exactly the held-out rows", "pass", emb["id"], scope=f"{name} fold {f}"))
                except SplitLeakError as e:
                    rows.append(_row("embedding table marks exactly the held-out rows", "fail", str(e), scope=f"{name} fold {f}"))
            if s.trainable:
                rec = exp.manifest.find("training", f"{name}__{tag}")
                if rec is not None:
                    d = rec["deps"]
                    ok = d.get("regime") == "inductive" and d.get("tag") == tag and d.get("bundle") == b.fingerprint and d.get("science") == exp.cfg.science_hash()
                    rows.append(_row("training record = inductive encoder of this fold, same bundle + science", "pass" if ok else "fail",
                                     f"regime {d.get('regime')}, tag {d.get('tag')}" + (f", imported from {rec.get('imported_from')}" if rec.get("imported_from") else ""),
                                     scope=f"{name} fold {f}"))
    # 5. target
    if b.athletes is not None and TRANSFER_TARGET in b.athletes.columns:
        no_nfl = ath.filter(pl.col("has_target") & ~pl.col("has_nfl")).height if "has_target" in ath.columns else 0
        rows.append(_row("frozen target never used as a predictor", "pass", "ladder predictors = M1 officials + Combine-side features + fold-fitted embedding PCs",
                         method="code inspection", scope="transfer"))
        rows.append(_row("athletes with a target but no NFL sequence in the bundle", "info", f"{no_nfl}", scope="transfer"))
        rows.append(_row("encoders never read athletes.parquet", "pass", "training uses X, M, derived physics / state / category / domain only",
                         method="code inspection", scope="training"))
    # 6. duplicates
    dsum, dgroups = duplicate_audit(exp, seq)
    vals = dict(zip(dsum["quantity"].to_list(), dsum["value"].to_list()))
    rows.append(_row("exact duplicate sequences", "pass" if vals["duplicate groups"] == 0 else ("fail" if vals["groups spanning folds"] else "warn"),
                     f"{vals['duplicate groups']} groups; spanning athletes {vals['groups spanning athletes']}, folds {vals['groups spanning folds']}"))
    # 7. shortcuts
    sc = shortcut_domain_auc(exp, seq, seed)
    triv = sc.filter(pl.col("reading") == "trivial shortcut")["feature"].to_list()
    rows.append(_row("domain readable from a non-movement property", "warn" if triv else "pass",
                     f"trivial shortcut features: {triv or 'none'} — domain AUC of learned embeddings must be read with this in mind"))
    scope = pl.DataFrame([{"component": a, "fit_scope": b_, "verification": c} for a, b_, c in PREPROCESSING_SCOPE])
    return {"checks": pl.DataFrame(rows), "duplicates": dsum, "duplicate_groups": dgroups, "shortcuts": sc, "scope": scope}


__all__ = ["COMPOSITION_METRICS", "PREPROCESSING_SCOPE", "athlete_table", "attrition", "dataset_overview", "domain_composition", "duplicate_audit",
           "leakage_audit", "m1_correlations", "sequence_hashes", "shortcut_domain_auc", "state_category_composition"]
