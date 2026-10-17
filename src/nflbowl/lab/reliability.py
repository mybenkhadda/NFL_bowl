"""Part E — reliability-aware biomechanical features.

E1 audit. M12 reported split-half reliability as Spearman–Brown(r) = 2r / (1 + r) of attempt 1 vs attempt 2 (store
order). For r < 0 this transform is not a reliability: it is unbounded below (r = −0.5 → −2, r → −1 → −∞), which is
where the extreme negative M12 values come from. M13 keeps that number (column ``sb_raw``, never clamped) and adds the
one-way random-effects ICC(1,1) over ALL attempts with its variance components:

    MSB, MSW from the one-way ANOVA (athletes = groups, unbalanced), k0 = (N − Σ nᵢ² / N) / (a − 1)
    ICC(1,1) = (MSB − MSW) / (MSB + (k0 − 1) MSW)      bounded below by −1 / (k0 − 1); negative = within > between
    σ²_between (raw, may be negative) = (MSB − MSW) / k0 ; σ²_within = MSW
    ICC(1,k) = k ICC / (1 + (k − 1) ICC)               reliability of an athlete mean of k trials

Agility columns (``c_agility__*``) pool three drills: values are centred on each drill's mean first ("kind-centred").
Intervals: athlete-cluster bootstrap (athletes resampled; a resampled athlete keeps all its trials).

E2 equal-observation reliability: the SAME athletes, the SAME number k of trials in both domains, k trials drawn
without replacement per athlete, repeated across seeds → curves of ICC(1,1) and ICC(1,k) versus k.

E3 selection ladders (``ae.oof_ladder`` blocks, fitted per outer fold on TRAINING athletes only):
    M1+BIO_all       every candidate metric (static columns)
    M1+BIO_reliable  candidates whose TRAIN-fold kind-centred ICC(1,1) ≥ threshold (≥ min athletes)
    M1+BIO_trainsel  top-k candidates by |partial Spearman(metric, target | M1)| among TRAIN athletes
    M1+BIO_poscond   all candidates standardised within position with TRAIN-athlete position mean / SD
No held-out outcome or held-out trial enters any selection (unit-tested).

E4 stability and E5 interactions are exploratory and labelled so; interactions use train-fold standardisation inside
the block (never all-row z-scores).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..data.validation import M1_COLS, TRANSFER_TARGET

NAN = float("nan")
AGILITY_KINDS = ("three_cone", "short_shuttle", "long_shuttle")
INTERACTIONS = (("speed × braking", "c_forty__peak_speed", "c_agility__peak_decel"),
                ("acceleration × turning", "c_forty__peak_accel", "c_agility__turn_at_speed"),
                ("peak speed × acceleration rate", "c_forty__peak_speed", "c_forty__accel_rate_0_90"),
                ("consistency × explosiveness", "c_forty__sd_peak_speed", "c_forty__peak_accel"))
POSITION_INTERACTION = ("position × acceleration rate", "c_forty__accel_rate_0_90")


# ====================================================================== candidate columns ↔ trials
def feature_map(columns: Sequence[str]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Athlete column → (sequence metric, Combine drill kinds whose attempts it summarises). Consistency (SD) and count
    columns have no per-trial value and are not mapped."""
    out = {}
    for c in columns:
        if c.startswith("c_forty__") and not c.endswith(("__n_seq", "__sd_peak_speed")):
            out[c] = (c.removeprefix("c_forty__"), ("forty",))
        elif c.startswith("c_agility__") and not c.endswith(("__n_seq", "__sd_peak_speed")):
            out[c] = (c.removeprefix("c_agility__"), AGILITY_KINDS)
    return out


def candidate_columns(data: Any, min_coverage: float = 0.5) -> list[str]:
    return [c for c in data.columns if c.startswith("c_") and not c.endswith("__n_seq") and float(data[c].is_not_null().mean()) >= min_coverage]


def trials(seq: Any, metric: str, kinds: Sequence[str] | None, domain: str = "combine") -> Any:
    """Long trials (nfl_id, kind, order, value) of one metric — finite values only, store order within athlete × kind."""
    import polars as pl

    d = seq.filter(pl.col("domain") == domain)
    if kinds is not None:
        d = d.filter(pl.col("kind").is_in(list(kinds)))
    d = d.filter(pl.col(metric).is_not_null() & pl.col(metric).cast(pl.Float64).is_finite()).sort("idx")
    return d.select("nfl_id", "kind", pl.int_range(pl.len()).over("nfl_id", "kind").alias("order"), pl.col(metric).cast(pl.Float64).alias("value"))


def kind_centre(t: Any, ref_ids: set[int] | None = None) -> np.ndarray:
    """Values centred on each drill kind's mean (means from ``ref_ids`` athletes only when given)."""
    import polars as pl

    ref = t if ref_ids is None else t.filter(pl.col("nfl_id").is_in(list(ref_ids)))
    mu = dict(zip(*ref.group_by("kind").agg(pl.col("value").mean()).to_dict(as_series=False).values()))
    v = t["value"].to_numpy().astype(float)
    k = t["kind"].to_list()
    return np.array([x - mu.get(kk, np.nan) for x, kk in zip(v, k)])


# ====================================================================== ICC
def icc_components(values: np.ndarray, ids: np.ndarray) -> dict[str, float]:
    """One-way random-effects ANOVA on athletes with ≥ 2 finite trials. Nothing is clamped."""
    v = np.asarray(values, float)
    g = np.asarray(ids)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2
    v, inv = v[keep], inv[keep]
    out = {"n_athletes": 0, "n_obs": 0, "k0": NAN, "msb": NAN, "msw": NAN, "var_between_raw": NAN, "var_within": NAN, "icc1": NAN, "icc1_lower_bound": NAN}
    if v.size == 0:
        return out
    u2, inv2, n_i = np.unique(inv, return_inverse=True, return_counts=True)
    a, N = u2.size, v.size
    out.update({"n_athletes": int(a), "n_obs": int(N)})
    if a < 2:
        return out
    means = np.bincount(inv2, weights=v) / n_i
    grand = v.mean()
    ssb = float((n_i * (means - grand) ** 2).sum())
    ssw = float(((v - means[inv2]) ** 2).sum())
    msb, msw = ssb / (a - 1), ssw / (N - a) if N > a else NAN
    k0 = (N - (n_i ** 2).sum() / N) / (a - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw) if np.isfinite(msw) and (msb + (k0 - 1) * msw) > 0 else NAN
    out.update({"k0": float(k0), "msb": msb, "msw": float(msw), "var_between_raw": float((msb - msw) / k0) if np.isfinite(msw) else NAN,
                "var_within": float(msw), "icc1": float(icc), "icc1_lower_bound": float(-1.0 / (k0 - 1)) if k0 > 1 else NAN})
    return out


def icc_average(icc1: float, k: float) -> float:
    """ICC(1,k): reliability of a k-trial mean (Spearman–Brown applied to ICC(1,1); defined for ICC > −1/(k−1))."""
    if not np.isfinite(icc1) or k < 1:
        return NAN
    den = 1 + (k - 1) * icc1
    return float(k * icc1 / den) if den > 0 else NAN


def icc_bootstrap(values: np.ndarray, ids: np.ndarray, n_boot: int, seed: int, ci: float = 0.95) -> tuple[float, float]:
    """Athlete-cluster bootstrap percentile interval of ICC(1,1) (resampled athletes get distinct labels)."""
    v, g = np.asarray(values, float), np.asarray(ids)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    u, inv = np.unique(g, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(u.size)]
    groups = [x for x in groups if x.size >= 2]
    if len(groups) < 10 or n_boot <= 0:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vv = np.concatenate([v[groups[j]] for j in pick])
        gg = np.concatenate([np.full(groups[j].size, r) for r, j in enumerate(pick)])
        bs.append(icc_components(vv, gg)["icc1"])
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    if bs.size < 20:
        return NAN, NAN
    a = (1 - ci) / 2
    return float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def spearman_brown_raw(r: float) -> float:
    """The M12 split-half estimator 2r/(1+r), UNCLAMPED (diagnostic: unbounded below for r < 0)."""
    return float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else NAN


# ====================================================================== E1 audit
def _shape(v: np.ndarray) -> dict[str, Any]:
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n_unique_values": 0, "share_zero": NAN, "cv": NAN}
    m = float(np.mean(v))
    return {"n_unique_values": int(np.unique(v).size), "share_zero": float((v == 0).mean()), "cv": float(np.std(v) / abs(m)) if m != 0 else NAN}


def _pairs(t: Any) -> tuple[np.ndarray, np.ndarray]:
    import polars as pl

    w = (t.filter(pl.col("order") < 2).group_by("nfl_id", "kind").agg(pl.col("value").sort_by("order").alias("_v"))
         .filter(pl.col("_v").list.len() == 2))
    vals = w["_v"].to_list()
    return np.array([x[0] for x in vals], float), np.array([x[1] for x in vals], float)


def audit_row(t: Any, feature: str, metric: str, domain: str, scope: str, n_boot: int, seed: int, min_athletes: int, centre: bool) -> dict[str, Any]:
    v = kind_centre(t) if centre else t["value"].to_numpy().astype(float)
    ids = t["nfl_id"].to_numpy()
    comp = icc_components(v, ids)
    lo, hi = icc_bootstrap(v, ids, n_boot, seed) if comp["n_athletes"] >= min_athletes else (NAN, NAN)
    a, b = _pairs(t) if domain == "combine" else _odd_even(t)
    ok = np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 5 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else NAN
    same = int((a[ok] == b[ok]).sum())
    per = np.unique(ids, return_counts=True)[1]
    flags = []
    if np.isfinite(r) and r < 0:
        flags.append("negative split-half r: Spearman–Brown is unbounded below (not a reliability)")
    if comp["n_athletes"] < min_athletes:
        flags.append(f"fewer than {min_athletes} athletes with ≥ 2 trials")
    if np.isfinite(comp["var_between_raw"]) and comp["var_between_raw"] < comp["var_within"]:
        flags.append("between-athlete variance < within-athlete variance")
    shp = _shape(t["value"].to_numpy().astype(float))
    if shp["n_unique_values"] and shp["n_unique_values"] < 10:
        flags.append("discrete / coarse metric")
    if same:
        flags.append(f"{same} identical repeat pair(s): possible duplicate sequences")
    if domain == "combine":
        flags.append("trial pairing by store order (chronology not verified)")
    return {"feature": feature, "metric": metric, "domain": domain, "scope": scope, **comp, "icc1_lo": lo, "icc1_hi": hi,
            "icc_mean_k0": icc_average(comp["icc1"], comp["k0"]) if np.isfinite(comp["k0"]) else NAN,
            "trials_per_athlete_median": float(np.median(per)) if per.size else NAN, "trials_per_athlete_max": int(per.max()) if per.size else 0,
            "n_pairs": int(ok.sum()), "r_half": r, "sb_raw": spearman_brown_raw(r), "n_identical_pairs": same,
            "share_identical_pairs": same / max(int(ok.sum()), 1), **shp, "kind_centred": centre, "flags": "; ".join(flags),
            "ci_method": "athlete-cluster bootstrap percentile" if np.isfinite(lo) else "none (too few athletes)"}


def _odd_even(t: Any) -> tuple[np.ndarray, np.ndarray]:
    import polars as pl

    h = t.with_columns(pl.int_range(pl.len()).over("nfl_id").alias("_k")).group_by("nfl_id", (pl.col("_k") % 2).alias("_h")).agg(pl.col("value").median())
    w = h.pivot(on="_h", index="nfl_id", values="value").drop_nulls()
    cols = [c for c in w.columns if c != "nfl_id"]
    if len(cols) != 2:
        return np.array([]), np.array([])
    return w[cols[0]].to_numpy().astype(float), w[cols[1]].to_numpy().astype(float)


def reliability_audit(seq: Any, columns: Sequence[str], n_boot: int, seed: int, min_athletes: int = 30) -> Any:
    """One row per (candidate column, domain, scope): Combine per drill + kind-centred pooled column; NFL windows."""
    import polars as pl

    fm = feature_map(columns)
    rows = []
    for col, (metric, kinds) in fm.items():
        if metric not in seq.columns:
            continue
        t = trials(seq, metric, kinds, "combine")
        for k in kinds:
            tk = t.filter(pl.col("kind") == k)
            if tk.height:
                rows.append(audit_row(tk, col, metric, "combine", k, n_boot, seed, min_athletes, centre=False))
        if len(kinds) > 1 and t.height:
            rows.append(audit_row(t, col, metric, "combine", "pooled drills (kind-centred)", n_boot, seed, min_athletes, centre=True))
        tn = trials(seq, metric, None, "nfl")
        if tn.height:
            rows.append(audit_row(tn, col, metric, "nfl", "all windows", n_boot, seed, min_athletes, centre=False))
    return pl.DataFrame(rows, infer_schema_length=None)


def column_reliability(audit: Any) -> dict[str, float]:
    """Full-sample Combine ICC(1,1) per column (forty: the forty row; agility: the kind-centred pooled row)."""
    import polars as pl

    out = {}
    c = audit.filter(pl.col("domain") == "combine")
    for col in c["feature"].unique().to_list():
        g = c.filter(pl.col("feature") == col)
        pick = g.filter(pl.col("scope") == "pooled drills (kind-centred)")
        pick = pick if pick.height else g.filter(pl.col("scope") == "forty")
        out[col] = float(pick["icc1"][0]) if pick.height and pick["icc1"][0] is not None else NAN
    return out


def explain_m12(m12_rep: Any, audit: Any) -> Any:
    """M12 repeatability rows with reliability < 0 next to the M13 raw r, SB and ICC of the same metric / domain / scope."""
    import polars as pl

    if m12_rep is None or not m12_rep.height or "reliability" not in m12_rep.columns:
        return pl.DataFrame()
    bad = m12_rep.filter(pl.col("reliability").cast(pl.Float64) < 0).select(
        "metric", "domain", "scope", pl.col("reliability").alias("m12_reliability"), *[c for c in ("r_half", "n_athletes") if c in m12_rep.columns])
    a = audit.select("metric", "domain", "scope", pl.col("r_half").alias("m13_r_half"), "sb_raw", "icc1", "icc1_lo", "icc1_hi", pl.col("n_athletes").alias("m13_n"),
                     "flags")
    return bad.join(a, on=["metric", "domain", "scope"], how="left").sort("m12_reliability")


# ====================================================================== E2 equal-observation curves
def _sample_k(t: Any, ids: Sequence[int], k: int, rng: np.random.Generator, centre: bool) -> tuple[np.ndarray, np.ndarray]:
    v = kind_centre(t) if centre else t["value"].to_numpy().astype(float)
    g = t["nfl_id"].to_numpy()
    order = np.argsort(g, kind="stable")
    gs, vs = g[order], v[order]
    cut = np.flatnonzero(np.diff(gs)) + 1
    groups = dict(zip(gs[np.r_[0, cut]].tolist() if gs.size else [], np.split(vs, cut) if gs.size else []))
    vv, gg = [], []
    for i in ids:
        x = groups[int(i)]
        pick = rng.choice(x.size, k, replace=False)
        vv.append(x[pick])
        gg.append(np.full(k, int(i)))
    return np.concatenate(vv), np.concatenate(gg)


def equal_observation_curves(seq: Any, columns: Sequence[str], k_grid: Sequence[int], n_repeats: int, seed: int, min_athletes: int = 30) -> tuple[Any, Any]:
    """(curve rows: column, domain, k, n_athletes, icc1 mean / q025 / q975, icc_k mean / q025 / q975 ;
    differences nfl − combine per repeat). Same athletes and same k in both domains."""
    import polars as pl

    fm = feature_map(columns)
    rows, diffs = [], []
    for col, (metric, kinds) in fm.items():
        if metric not in seq.columns:
            continue
        tc, tn = trials(seq, metric, kinds, "combine"), trials(seq, metric, None, "nfl")
        nc = dict(zip(*tc.group_by("nfl_id").len().to_dict(as_series=False).values())) if tc.height else {}
        nn = dict(zip(*tn.group_by("nfl_id").len().to_dict(as_series=False).values())) if tn.height else {}
        for k in k_grid:
            if k < 2:
                continue
            ids = sorted(i for i in nc if nc[i] >= k and nn.get(i, 0) >= k)
            if len(ids) < min_athletes:
                rows.append({"feature": col, "metric": metric, "k": int(k), "domain": "both", "n_athletes": len(ids), "status": "insufficient athletes"})
                continue
            res = {"combine": [], "nfl": []}
            for r in range(n_repeats):
                rng = np.random.default_rng(seed + 1000 * r + k)
                for dom, t, centre in (("combine", tc, len(kinds) > 1), ("nfl", tn, False)):
                    vv, gg = _sample_k(t, ids, k, rng, centre)
                    res[dom].append(icc_components(vv, gg)["icc1"])
                diffs.append({"feature": col, "k": int(k), "repeat": r, "icc1_combine": res["combine"][-1], "icc1_nfl": res["nfl"][-1],
                              "diff_nfl_minus_combine": res["nfl"][-1] - res["combine"][-1]})
            for dom in ("combine", "nfl"):
                a = np.asarray(res[dom], float)
                a = a[np.isfinite(a)]
                ak = np.array([icc_average(x, k) for x in a])
                rows.append({"feature": col, "metric": metric, "k": int(k), "domain": dom, "n_athletes": len(ids), "status": "evaluated",
                             "icc1_mean": float(a.mean()) if a.size else NAN, "icc1_q025": float(np.quantile(a, 0.025)) if a.size else NAN,
                             "icc1_q975": float(np.quantile(a, 0.975)) if a.size else NAN, "icck_mean": float(np.nanmean(ak)) if ak.size else NAN,
                             "icck_q025": float(np.nanquantile(ak, 0.025)) if ak.size else NAN, "icck_q975": float(np.nanquantile(ak, 0.975)) if ak.size else NAN,
                             "n_repeats": int(n_repeats)})
    return pl.DataFrame(rows, infer_schema_length=None), pl.DataFrame(diffs, infer_schema_length=None)


def projected_curve(audit: Any, k_max: int = 10) -> Any:
    """Spearman–Brown projection ICC(1,k), k = 1…k_max, of the full-sample ICC(1,1) (dashed reference in E2)."""
    import polars as pl

    rows = []
    for r in audit.filter(pl.col("icc1").is_not_null()).iter_rows(named=True):
        for k in range(1, k_max + 1):
            rows.append({"feature": r["feature"], "domain": r["domain"], "scope": r["scope"], "k": k, "icc_k_projected": icc_average(r["icc1"], k)})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== E3 train-fold-only selection blocks
class SelectionLog(list):
    """Rows {model, fold, selected, scores} appended by the blocks while the ladder runs."""


def _fold_of(train: np.ndarray, folds: np.ndarray) -> int:
    f = np.unique(folds[~train])
    if f.size != 1:
        raise ValueError(f"the held-out rows of a block span folds {f.tolist()}")
    return int(f[0])


def _aligned(data: Any, ids: np.ndarray, cols: Sequence[str]) -> np.ndarray:
    """Rows of ``data`` (float columns ``cols``) in the order of ``ids`` (explicit id → row map; NaN when absent)."""
    import polars as pl

    if not cols:
        return np.zeros((ids.size, 0))
    M = data.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy().astype(float)
    row = {int(i): r for r, i in enumerate(data["nfl_id"].to_list())}
    out = np.full((ids.size, len(cols)), np.nan)
    for k, i in enumerate(ids.tolist()):
        r = row.get(int(i))
        if r is not None:
            out[k] = M[r]
    return out


def _aligned_positions(data: Any, ids: np.ndarray) -> np.ndarray:
    pos = dict(zip(data["nfl_id"].to_list(), data["position_group"].fill_null("?").to_list()))
    return np.array([pos.get(int(i), "?") for i in ids.tolist()], dtype=object)


def train_icc(trial_cache: dict[str, Any], col: str, train_ids: set[int], min_athletes: int) -> tuple[float, int]:
    import polars as pl

    t = trial_cache.get(col)
    if t is None or not t.height:
        return NAN, 0
    tt = t.filter(pl.col("nfl_id").is_in(list(train_ids)))
    centre = tt["kind"].n_unique() > 1
    v = kind_centre(tt, train_ids) if centre else tt["value"].to_numpy().astype(float)
    c = icc_components(v, tt["nfl_id"].to_numpy())
    return (c["icc1"], c["n_athletes"]) if c["n_athletes"] >= min_athletes else (NAN, c["n_athletes"])


def reliable_block(data: Any, cand: Sequence[str], trial_cache: dict[str, Any], threshold: float, min_athletes: int, log: SelectionLog) -> Callable:
    def factory(ids: np.ndarray) -> Callable:
        X = _aligned(data, ids, cand)
        folds = _aligned(data, ids, ["fold"])[:, 0]

        def block(train: np.ndarray) -> Any:
            import polars as pl

            f = _fold_of(train, folds)
            tr_ids = set(int(i) for i in ids[train])
            scores = {c: train_icc(trial_cache, c, tr_ids, min_athletes) for c in cand}
            sel = [c for c in cand if np.isfinite(scores[c][0]) and scores[c][0] >= threshold]
            log.append({"model": "M1+BIO_reliable", "fold": f, "selected": sel, "scores": {c: s[0] for c, s in scores.items()}})
            if not sel:
                return pl.DataFrame({"none": np.zeros(ids.size)})
            return pl.DataFrame({c: X[:, cand.index(c)] for c in sel})
        return block
    return factory


def _partial_spearman_train(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    from .stats import partial_spearman

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return NAN
    Zi = Z[ok].copy()
    for j in range(Zi.shape[1]):
        col = Zi[:, j]
        med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        Zi[:, j] = np.where(np.isfinite(col), col, med)
    return partial_spearman(x[ok], y[ok], Zi)


def trainsel_block(data: Any, cand: Sequence[str], k: int, log: SelectionLog) -> Callable:
    """Top-k by |partial Spearman(metric, target | M1)| on TRAINING athletes only."""
    def factory(ids: np.ndarray) -> Callable:
        X = _aligned(data, ids, cand)
        y = _aligned(data, ids, [TRANSFER_TARGET])[:, 0]
        Z = _aligned(data, ids, [c for c in M1_COLS if c in data.columns])
        folds = _aligned(data, ids, ["fold"])[:, 0]

        def block(train: np.ndarray) -> Any:
            import polars as pl

            f = _fold_of(train, folds)
            sc = {c: _partial_spearman_train(X[train, j], y[train], Z[train]) for j, c in enumerate(cand)}
            ranked = sorted((c for c in cand if np.isfinite(sc[c])), key=lambda c: -abs(sc[c]))
            sel = ranked[:k]
            log.append({"model": "M1+BIO_trainsel", "fold": f, "selected": sel, "scores": sc})
            if not sel:
                return pl.DataFrame({"none": np.zeros(ids.size)})
            return pl.DataFrame({c: X[:, cand.index(c)] for c in sel})
        return block
    return factory


def poscond_block(data: Any, cand: Sequence[str], log: SelectionLog, min_group: int = 5) -> Callable:
    """Candidates standardised within position with TRAINING-athlete position mean / SD (pooled train stats for small groups)."""
    def factory(ids: np.ndarray) -> Callable:
        import polars as pl

        X = _aligned(data, ids, cand)
        folds = _aligned(data, ids, ["fold"])[:, 0]
        pos = _aligned_positions(data, ids)

        def block(train: np.ndarray) -> Any:
            f = _fold_of(train, folds)
            out = {}
            for j, c in enumerate(cand):
                x = X[:, j]
                z = np.full(x.size, np.nan)
                tr_all = train & np.isfinite(x)
                mu0 = float(x[tr_all].mean()) if tr_all.any() else NAN
                sd0 = float(x[tr_all].std()) if tr_all.sum() > 1 else NAN
                for p in np.unique(pos):
                    m = pos == p
                    trp = m & tr_all
                    mu, sd = (float(x[trp].mean()), float(x[trp].std())) if trp.sum() >= min_group else (mu0, sd0)
                    sd = sd if np.isfinite(sd) and sd > 1e-12 else (sd0 if np.isfinite(sd0) and sd0 > 1e-12 else 1.0)
                    z[m] = (x[m] - mu) / sd
                out[f"{c}__poscond"] = z
            log.append({"model": "M1+BIO_poscond", "fold": f, "selected": list(cand), "scores": {}})
            return pl.DataFrame(out)
        return block
    return factory


def selection_ladder(exp: Any, data: Any, cand: Sequence[str], trial_cache: dict[str, Any], threshold: float, k: int, min_athletes: int) -> tuple[Any, Any, Any]:
    """(wide OOF frame, selection log frame, ladder log). Same engine, folds, seed and analysis rows as the official ladder."""
    import polars as pl

    from .. import _backend

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    log = SelectionLog()
    cand = list(cand)
    mods = {"M1": ae.Model(list(ae.M1_COLS)), "M1+BIO_all": ae.Model([*ae.M1_COLS, *cand]),
            "M1+BIO_reliable": ae.Model(list(ae.M1_COLS), factory=reliable_block(data, cand, trial_cache, threshold, min_athletes, log)),
            "M1+BIO_trainsel": ae.Model(list(ae.M1_COLS), factory=trainsel_block(data, cand, k, log)),
            "M1+BIO_poscond": ae.Model(list(ae.M1_COLS), factory=poscond_block(data, cand, log))}
    res = ae.oof_ladder(data, TRANSFER_TARGET, mods, C, C.seed)
    oof = res.oof_frame()
    sel = pl.DataFrame([{"model": r["model"], "fold": r["fold"], "n_selected": len(r["selected"]), "selected": ", ".join(r["selected"]),
                         **{f"score:{c}": (None if not np.isfinite(v) else float(v)) for c, v in r["scores"].items()}} for r in log], infer_schema_length=None) if log else pl.DataFrame()
    return oof, sel, pl.DataFrame(res.log, infer_schema_length=None)


def trial_cache(seq: Any, cand: Sequence[str]) -> dict[str, Any]:
    return {c: trials(seq, m, kinds, "combine") for c, (m, kinds) in feature_map(cand).items() if m in seq.columns}


# ====================================================================== E4 stability
def feature_stability(data: Any, cache: dict[str, Any], cand: Sequence[str], min_athletes: int, min_position_n: int) -> Any:
    """Per candidate: full-sample ICC, train-fold ICC spread, train-fold partial association (sign consistency), ICC range
    over positions, Spearman with the officials, in-sample partial Spearman with the target | M1 (descriptive only)."""
    import polars as pl

    from .stats import spearman

    folds = sorted(int(f) for f in data["fold"].unique().to_list())
    ids_all = set(int(i) for i in data["nfl_id"].to_list())
    y = data[TRANSFER_TARGET].cast(pl.Float64).to_numpy()
    Z = data.select([pl.col(c).cast(pl.Float64) for c in M1_COLS if c in data.columns]).to_numpy().astype(float)
    pos_of = dict(zip(data["nfl_id"].to_list(), data["position_group"].fill_null("?").to_list()))
    rows = []
    for c in cand:
        x = data[c].cast(pl.Float64).to_numpy()
        icc_all, n_all = train_icc(cache, c, ids_all, min_athletes) if c in cache else (NAN, 0)
        fold_icc, fold_ps = [], []
        for f in folds:
            tr = data["fold"].to_numpy() != f
            tr_ids = set(int(i) for i in data["nfl_id"].to_numpy()[tr])
            if c in cache:
                fold_icc.append(train_icc(cache, c, tr_ids, min_athletes)[0])
            fold_ps.append(_partial_spearman_train(x[tr], y[tr], Z[tr]))
        fi, fp = np.asarray(fold_icc, float), np.asarray(fold_ps, float)
        pos_icc = []
        if c in cache:
            for p in sorted(set(pos_of.values()), key=str):
                pid = {i for i, q in pos_of.items() if q == p}
                v, n = train_icc(cache, c, pid, max(min_position_n, 10))
                if np.isfinite(v):
                    pos_icc.append(v)
        offi = {f"spearman_{o}": spearman(x, data[o].cast(pl.Float64).to_numpy()) for o in ("forty", "ten", "weight", "height") if o in data.columns}
        sign = np.sign(fp[np.isfinite(fp)])
        rows.append({"feature": c, "coverage": float(np.isfinite(x).mean()), "icc1_all": icc_all, "n_icc": n_all,
                     "icc1_train_mean": float(np.nanmean(fi)) if np.isfinite(fi).any() else NAN, "icc1_train_sd": float(np.nanstd(fi)) if np.isfinite(fi).sum() > 1 else NAN,
                     "partial_assoc_train_mean": float(np.nanmean(fp)) if np.isfinite(fp).any() else NAN,
                     "partial_assoc_sign_consistency": float(max((sign > 0).mean(), (sign < 0).mean())) if sign.size else NAN,
                     "icc1_position_min": float(min(pos_icc)) if pos_icc else NAN, "icc1_position_max": float(max(pos_icc)) if pos_icc else NAN,
                     "n_positions_icc": len(pos_icc), **offi, "partial_spearman_target_given_M1_insample": _partial_spearman_train(x, y, Z)})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== E5 interactions (exploratory)
def interaction_factory(data: Any, a: str, b: str) -> Callable:
    """Block = z(a) · z(b) with TRAINING-fold mean / SD (missing stays missing → design-internal imputation)."""
    def factory(ids: np.ndarray) -> Callable:
        X = _aligned(data, ids, [a, b])

        def block(train: np.ndarray) -> Any:
            import polars as pl

            z = []
            for j in range(2):
                x = X[:, j]
                tr = train & np.isfinite(x)
                mu, sd = (float(x[tr].mean()), float(x[tr].std())) if tr.sum() > 2 else (0.0, 1.0)
                z.append((x - mu) / (sd if sd > 1e-12 else 1.0))
            return pl.DataFrame({f"{a}__x__{b}": z[0] * z[1]})
        return block
    return factory


def position_interaction_factory(data: Any, a: str) -> Callable:
    """Block = z(a) × 1[position = p] for every TRAINING position level except the first (within-position slopes)."""
    def factory(ids: np.ndarray) -> Callable:
        x = _aligned(data, ids, [a])[:, 0]
        pos = _aligned_positions(data, ids)

        def block(train: np.ndarray) -> Any:
            import polars as pl

            tr = train & np.isfinite(x)
            mu, sd = (float(x[tr].mean()), float(x[tr].std())) if tr.sum() > 2 else (0.0, 1.0)
            z = (x - mu) / (sd if sd > 1e-12 else 1.0)
            levels = sorted(set(pos[train].tolist()))
            return pl.DataFrame({f"{a}__x__pos={lv}": z * (pos == lv) for lv in levels[1:]}) if len(levels) > 1 else pl.DataFrame({"none": np.zeros(ids.size)})
        return block
    return factory


def interaction_tests(exp: Any, data: Any, n_boot: int, seed: int) -> tuple[Any, Any]:
    """(paired comparisons interaction vs additive per declared pair — cohort = athletes with both metrics observed;
    long OOF of all interaction ladders). BH across the family. Exploratory, non-causal."""
    import polars as pl

    from .. import _backend
    from . import transfer_fairness as tf

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    rows, oofs = [], []
    specs = [(lbl, a, b) for lbl, a, b in INTERACTIONS if a in data.columns and b in data.columns]
    for lbl, a, b in specs:
        mods = {"additive": ae.Model([*ae.M1_COLS, a, b]), "interaction": ae.Model([*ae.M1_COLS, a, b], factory=interaction_factory(data, a, b))}
        res = ae.oof_ladder(data, TRANSFER_TARGET, mods, C, C.seed)
        o = res.oof_frame()
        both = set(int(i) for i in data.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())["nfl_id"].to_list())
        w = o.filter(pl.col("nfl_id").is_in(list(both)))
        if w.height >= 20:
            rows.append(tf.paired_comparisons(w, [("interaction", "additive")], n_boot, seed, cohort=f"{lbl} (both observed)").with_columns(
                pl.lit(lbl).alias("interaction"), pl.lit(a).alias("metric_a"), pl.lit(b).alias("metric_b")))
        oofs.append(o.with_columns(pl.lit(lbl).alias("interaction")))
    lbl, a = POSITION_INTERACTION
    if a in data.columns:
        mods = {"additive": ae.Model([*ae.M1_COLS, a]), "interaction": ae.Model([*ae.M1_COLS, a], factory=position_interaction_factory(data, a))}
        res = ae.oof_ladder(data, TRANSFER_TARGET, mods, C, C.seed)
        o = res.oof_frame()
        obs = set(int(i) for i in data.filter(pl.col(a).is_not_null())["nfl_id"].to_list())
        w = o.filter(pl.col("nfl_id").is_in(list(obs)))
        if w.height >= 20:
            rows.append(tf.paired_comparisons(w, [("interaction", "additive")], n_boot, seed, cohort=f"{lbl} (observed)").with_columns(
                pl.lit(lbl).alias("interaction"), pl.lit(a).alias("metric_a"), pl.lit("position_group").alias("metric_b")))
        oofs.append(o.with_columns(pl.lit(lbl).alias("interaction")))
    out = pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame()
    if out.height:
        out = out.with_columns(pl.Series("p_bh_family", ae.bh(out["p_boot_le0"].to_list())), pl.lit("exploratory, non-causal").alias("status"))
    return out, (pl.concat(oofs, how="diagonal_relaxed") if oofs else pl.DataFrame())


__all__ = ["AGILITY_KINDS", "INTERACTIONS", "POSITION_INTERACTION", "SelectionLog", "audit_row", "candidate_columns", "column_reliability",
           "equal_observation_curves", "explain_m12", "feature_map", "feature_stability", "icc_average", "icc_bootstrap", "icc_components",
           "interaction_factory", "interaction_tests", "kind_centre", "poscond_block", "position_interaction_factory", "projected_curve", "reliability_audit",
           "reliable_block", "selection_ladder", "spearman_brown_raw", "train_icc", "trainsel_block", "trial_cache", "trials"]
