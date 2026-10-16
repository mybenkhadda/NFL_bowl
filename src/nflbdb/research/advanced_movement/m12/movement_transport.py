"""12E — conditional optimal transport between Combine and NFL movement distributions. NumPy (POT optional for exact W).

Distributions live on ONE physical grid shared by both domains: the 12.0 shared speed bins (0.5 yd/s) × acceleration
bins of ``m11_envelope_hist`` merged to 0.5 yd/s² over the declared range ``ACCEL_RANGE`` (values outside are dropped,
not clipped). NFL: an athlete's eligible post-snap frames (games pooled or split). Combine: the forty attempt frames.
States: ``all`` = every frame on the grid; ``accelerating`` = frames with a_tan ≥ ``ot_accel_on`` (the comparable state —
a standing-start sprint is compared with game movement only where both are accelerating at the same speed).

Ground cost: squared Euclidean distance in physical units (1 yd/s ≡ 1 yd/s², declared; sensitivity in the notebook).

    E0  Euclidean on standardized physical features (same quantities both sides, z-scored within position and domain)
    E1  Mahalanobis on the same features (shrunk pooled covariance of training athletes)
    E2  exact W2 (POT ``ot.emd2`` when installed; else not_available)
    E3  sliced W2 (``ot_projections`` fixed directions)
    E4  debiased Sinkhorn divergence S_ε = OT_ε(P,Q) − ½OT_ε(P,P) − ½OT_ε(Q,Q), ε = ``ot_eps`` × median cost
    E5  conditional OT: mean over shared speed bins (both supported) of the 1-D W1 between acceleration distributions
    E6  unbalanced Sinkhorn (KL marginal relaxation ``unbalanced_rho``) on UNNORMALISED frame shares (opportunity mass)

Retrieval is a statement about cross-domain similarity, never about football performance.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ..m11 import evaluation as ev
from . import advanced_evaluation as ae
from .config import M12Config

NAN = float("nan")
ACCEL_RANGE = (-4.0, 6.0)        # proposed: acceleration support of the OT grid [yd/s²]
MERGE = 2                        # proposed: 0.25 → 0.5 yd/s² acceleration bins for transport
METHODS = ("E0_euclidean", "E1_mahalanobis", "E2_exact_w2", "E3_sliced_w2", "E4_sinkhorn", "E5_conditional_w1", "E6_unbalanced")
FEATURE_PAIRS = (("forty.band_2_4_mean_accel", "nfl.band_2_4"), ("forty.band_4_6_mean_accel", "nfl.band_4_6"),
                 ("forty.band_6_8_mean_accel", "nfl.band_6_8"), ("forty.v_max_obs", "nfl.vmax_p90"))


def pot_available() -> bool:
    try:
        import ot  # noqa: F401
        return True
    except Exception:                                                      # noqa: BLE001
        return False


# ============================================================================== grid
class Grid:
    """Restricted transport grid: rows = shared speed bins, columns = merged acceleration bins inside ACCEL_RANGE."""

    def __init__(self, c: M12Config, speed_bins: Sequence[int], accel_range: tuple[float, float] = ACCEL_RANGE, merge: int = MERGE):
        self.c, self.speed_bins = c, list(speed_bins)
        ab = ae.accel_edges(c)
        keep = np.flatnonzero((ab[:-1] >= accel_range[0] - 1e-9) & (ab[1:] <= accel_range[1] + 1e-9))
        keep = keep[: (keep.size // merge) * merge]
        self.accel_idx = keep.reshape(-1, merge)
        self.amid = np.array([0.5 * (ab[g[0]] + ab[g[-1] + 1]) for g in self.accel_idx])
        self.vmid = ae.mids(ae.speed_edges(c))[self.speed_bins]
        self.da = float(ab[self.accel_idx[0][-1] + 1] - ab[self.accel_idx[0][0]])
        V, A = np.meshgrid(self.vmid, self.amid, indexing="ij")
        self.points = np.c_[V.ravel(), A.ravel()]

    def restrict(self, H: np.ndarray, state: str = "all") -> np.ndarray:
        """(speed × acceleration) full histogram → restricted mass matrix (unnormalised frame counts)."""
        M = np.stack([H[self.speed_bins][:, g].sum(1) for g in self.accel_idx], 1).astype(float)
        if state == "accelerating":
            M[:, self.amid < self.c.ot_accel_on] = 0.0
        return M

    def coarsen(self) -> "Grid":
        """Binning sensitivity: acceleration bins twice as wide (1 yd/s²), same speed bins, same data."""
        return Grid(self.c, self.speed_bins, ACCEL_RANGE, MERGE * 2)


def normalise(M: np.ndarray) -> np.ndarray:
    s = M.sum()
    return M / s if s > 0 else M


# ============================================================================== distances on one pair
def cost_matrix(points: np.ndarray, mask_a: np.ndarray | None = None, mask_b: np.ndarray | None = None) -> np.ndarray:
    pa = points if mask_a is None else points[mask_a]
    pb = points if mask_b is None else points[mask_b]
    return ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)


def _lse(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray, eps: float, iters: int, tol: float = 1e-7, rho: float | None = None) -> tuple[float, np.ndarray]:
    """Log-domain Sinkhorn. ``rho`` None = balanced; else KL-relaxed marginals (unbalanced, exponent ρ/(ρ+ε)).
    Returns (transport cost ⟨P, C⟩, plan P)."""
    la, lb = np.log(np.maximum(a, 1e-300)), np.log(np.maximum(b, 1e-300))
    f, g = np.zeros(a.size), np.zeros(b.size)
    k = 1.0 if rho is None else rho / (rho + eps)
    for _ in range(iters):
        f_new = k * (-eps * _lse((g[None, :] - C) / eps + lb[None, :], 1))
        g_new = k * (-eps * _lse((f_new[:, None] - C) / eps + la[:, None], 0))
        if np.max(np.abs(f_new - f)) < tol and np.max(np.abs(g_new - g)) < tol:
            f, g = f_new, g_new
            break
        f, g = f_new, g_new
    P = np.exp((f[:, None] + g[None, :] - C) / eps + la[:, None] + lb[None, :])
    return float((P * C).sum()), P


def sinkhorn_divergence(P: np.ndarray, Q: np.ndarray, grid: Grid, eps_rel: float, iters: int) -> float:
    a, b = normalise(P).ravel(), normalise(Q).ravel()
    ma, mb = a > 0, b > 0
    if ma.sum() == 0 or mb.sum() == 0:
        return NAN
    Cab = cost_matrix(grid.points, ma, mb)
    eps = eps_rel * float(np.median(cost_matrix(grid.points)))
    ab_, _ = sinkhorn(a[ma], b[mb], Cab, eps, iters)
    aa, _ = sinkhorn(a[ma], a[ma], cost_matrix(grid.points, ma, ma), eps, iters)
    bb, _ = sinkhorn(b[mb], b[mb], cost_matrix(grid.points, mb, mb), eps, iters)
    return float(ab_ - 0.5 * aa - 0.5 * bb)


def unbalanced(P: np.ndarray, Q: np.ndarray, grid: Grid, eps_rel: float, rho: float, iters: int) -> float:
    """Unbalanced Sinkhorn cost between frame SHARES (P, Q each divided by the athlete's total frames on the full grid,
    so mass differences reflect how often the state was observed)."""
    a, b = P.ravel(), Q.ravel()
    ma, mb = a > 0, b > 0
    if ma.sum() == 0 or mb.sum() == 0:
        return NAN
    eps = eps_rel * float(np.median(cost_matrix(grid.points)))
    cost, plan = sinkhorn(a[ma], b[mb], cost_matrix(grid.points, ma, mb), eps, iters, rho=rho)
    kl = lambda p, q: float((p * np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300)) - p + q).sum())
    return cost + rho * (kl(plan.sum(1), a[ma]) + kl(plan.sum(0), b[mb]))


def exact_w2(P: np.ndarray, Q: np.ndarray, grid: Grid) -> float:
    if not pot_available():
        return NAN
    import ot

    a, b = normalise(P).ravel(), normalise(Q).ravel()
    ma, mb = a > 0, b > 0
    if ma.sum() == 0 or mb.sum() == 0:
        return NAN
    return float(math.sqrt(max(ot.emd2(a[ma], b[mb], cost_matrix(grid.points, ma, mb)), 0.0)))


def _w1d_quantile(x: np.ndarray, w: np.ndarray, levels: np.ndarray) -> np.ndarray:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw = cw / cw[-1]
    return x[o][np.minimum(np.searchsorted(cw, levels, side="left"), x.size - 1)]


def sliced_w2(P: np.ndarray, Q: np.ndarray, grid: Grid, n_proj: int, seed: int = 0) -> float:
    a, b = normalise(P).ravel(), normalise(Q).ravel()
    if a.sum() == 0 or b.sum() == 0:
        return NAN
    th = np.linspace(0, np.pi, n_proj, endpoint=False) + np.random.default_rng(seed).uniform(0, np.pi / n_proj)
    D = np.c_[np.cos(th), np.sin(th)]
    lv = (np.arange(200) + 0.5) / 200
    tot = 0.0
    for d in D:
        x = grid.points @ d
        qa, qb = _w1d_quantile(x[a > 0], a[a > 0], lv), _w1d_quantile(x[b > 0], b[b > 0], lv)
        tot += float(np.mean((qa - qb) ** 2))
    return float(math.sqrt(tot / len(D)))


def conditional_w1(P: np.ndarray, Q: np.ndarray, grid: Grid, min_mass: float = 1.0) -> float:
    """E5: per speed bin (row) where both have ≥ ``min_mass`` frames, the 1-D W1 between the normalised acceleration
    distributions (Σ |CDF_P − CDF_Q| Δa); mean over those bins (equal bin weight). NaN when no bin is shared."""
    vals = []
    for k in range(P.shape[0]):
        if P[k].sum() >= min_mass and Q[k].sum() >= min_mass:
            vals.append(float(np.abs(np.cumsum(P[k] / P[k].sum()) - np.cumsum(Q[k] / Q[k].sum())).sum() * grid.da))
    return float(np.mean(vals)) if vals else NAN


def distance(method: str, P: np.ndarray, Q: np.ndarray, grid: Grid, c: M12Config, eps_rel: float | None = None, seed: int = 0) -> float:
    if method == "E2_exact_w2":
        return exact_w2(P, Q, grid)
    if method == "E3_sliced_w2":
        return sliced_w2(P, Q, grid, c.ot_projections, seed)
    if method == "E4_sinkhorn":
        return sinkhorn_divergence(P, Q, grid, c.ot_eps if eps_rel is None else eps_rel, c.ot_iters)
    if method == "E5_conditional_w1":
        return conditional_w1(P, Q, grid, max(1.0, float(c.min_frames_bin_combine)))   # a speed bin needs ≥ 1 frame in BOTH domains
    raise ValueError(method)


# ============================================================================== athlete distributions
def combine_hists(ch: pl.DataFrame, grid: Grid, c: M12Config, family: str = "forty", state: str = "all", by: str = "nfl_id") -> dict[Any, np.ndarray]:
    """{athlete (or seq_id): restricted Combine mass matrix} from ``m12_combine_hist``."""
    ns, na = ae.speed_edges(c).size - 1, ae.accel_edges(c).size - 1
    out = {}
    d = ch.filter(pl.col("family") == family)
    for (k,), g in d.partition_by([by], as_dict=True).items():
        H = np.zeros((ns, na))
        np.add.at(H, (g["speed_bin"].to_numpy(), g["accel_bin"].to_numpy()), g["n"].to_numpy().astype(float))
        out[k] = grid.restrict(H, state)
    return out


def nfl_hists(cubes: dict[int, Any], grid: Grid, state: str = "all", sel: dict[int, np.ndarray] | None = None, shares: bool = False) -> dict[int, np.ndarray]:
    """{athlete: restricted NFL mass matrix} from 12C cubes; ``shares`` divides by the athlete's total frames on the FULL grid."""
    out = {}
    for i, cb in cubes.items():
        s = None if sel is None else sel.get(i)
        if sel is not None and s is None:
            continue
        H = cb.total(s)
        M = grid.restrict(H, state)
        out[i] = M / max(H.sum(), 1.0) if shares else M
    return out


# ============================================================================== feature distances (E0 / E1)
def feature_vectors(data: pl.DataFrame, positions: dict[int, str]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Per athlete (Combine vector, NFL vector) of the shared physical quantities, each z-scored within position and
    domain (relative standing; removes the domain level shift)."""
    comb, nfl = {}, {}
    cc, nc = [a for a, _ in FEATURE_PAIRS], [b for _, b in FEATURE_PAIRS]
    d = data.select("nfl_id", "position_group", *[k for k in cc + nc if k in data.columns])
    for k in cc + nc:
        if k in d.columns:
            d = d.with_columns(((pl.col(k) - pl.col(k).mean().over("position_group")) / pl.col(k).std().over("position_group")).alias(k))
    for r in d.iter_rows(named=True):
        cv = np.array([r.get(k, np.nan) if r.get(k) is not None else np.nan for k in cc], float)
        nv = np.array([r.get(k, np.nan) if r.get(k) is not None else np.nan for k in nc], float)
        comb[int(r["nfl_id"])], nfl[int(r["nfl_id"])] = cv, nv
    return comb, nfl


def mahalanobis_metric(vectors: Sequence[np.ndarray], shrink: float = 0.2) -> np.ndarray:
    X = np.array([v for v in vectors if np.isfinite(v).all()])
    S = np.cov(X.T) if X.shape[0] > X.shape[1] + 2 else np.eye(X.shape[1])
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
    return np.linalg.inv(S + 1e-6 * np.eye(S.shape[0]))


def fdist(a: np.ndarray, b: np.ndarray, Minv: np.ndarray | None = None) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return NAN
    d = a[m] - b[m]
    return float(math.sqrt(d @ (Minv[np.ix_(m, m)] if Minv is not None else np.eye(m.sum())) @ d))


# ============================================================================== retrieval
def retrieval_auc(D: dict[tuple[int, int], float], queries: Sequence[int], candidates: Sequence[int], positions: dict[int, str]) -> pl.DataFrame:
    """Per query athlete i: AUC = share of same-position candidates j ≠ i with d(i, j) > d(i, i) (ties ½). Chance 0.5."""
    rows = []
    for i in queries:
        own = D.get((i, i))
        if own is None or not np.isfinite(own):
            continue
        others = [D.get((i, j)) for j in candidates if j != i and positions.get(j) == positions.get(i)]
        others = np.array([x for x in others if x is not None and np.isfinite(x)])
        if others.size < 3:
            continue
        rows.append({"nfl_id": i, "position_group": positions.get(i), "auc": float(((others > own) + 0.5 * (others == own)).mean()), "n_candidates": int(others.size),
                     "rank": int((others < own).sum()) + 1})
    return pl.DataFrame(rows)


def pairwise(queries: dict[int, np.ndarray], cands: dict[int, np.ndarray], positions: dict[int, str], fn: Callable[[np.ndarray, np.ndarray], float],
             progress: Callable[[], None] | None = None) -> dict[tuple[int, int], float]:
    """Distances from each query to every SAME-POSITION candidate (the only comparisons retrieval uses)."""
    D = {}
    for i, P in queries.items():
        for j, Q in cands.items():
            if positions.get(i) == positions.get(j):
                D[(i, j)] = fn(P, Q)
        if progress:
            progress()
    return D


def auc_summary(tab: pl.DataFrame, method: str, c: M12Config, seed: int) -> dict[str, Any]:
    a = tab["auc"].to_numpy() if tab.height else np.array([])
    out = {"method": method, "n_athletes": int(a.size), "auc": float(a.mean()) if a.size else NAN, "auc_lo": NAN, "auc_hi": NAN}
    if a.size >= ev.MIN_CI_ATHLETES:
        bs = np.random.default_rng(seed).integers(0, a.size, (max(200, c.n_boot // 5), a.size))
        m = a[bs].mean(1)
        out["auc_lo"], out["auc_hi"] = float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
    return out


def paired_auc_diff(t1: pl.DataFrame, t2: pl.DataFrame, c: M12Config, seed: int) -> dict[str, Any]:
    j = t1.select("nfl_id", pl.col("auc").alias("a")).join(t2.select("nfl_id", pl.col("auc").alias("b")), on="nfl_id")
    return ae.paired_metric_diff(j["a"].to_numpy(), j["b"].to_numpy(), max(200, c.n_boot // 5), seed)


def shuffled_null(D: dict[tuple[int, int], float], queries: Sequence[int], candidates: Sequence[int], positions: dict[int, str], n_perm: int, seed: int) -> np.ndarray:
    """Mean AUC when the candidate identities are permuted within position (the query's 'own' NFL distribution becomes a
    random same-position athlete's) — the retrieval null."""
    rng = np.random.default_rng(seed)
    cand = list(candidates)
    out = []
    for _ in range(n_perm):
        perm = {}
        for p in set(positions.get(i) for i in cand):
            mem = [i for i in cand if positions.get(i) == p]
            for a, b in zip(mem, rng.permutation(mem)):
                perm[a] = int(b)
        Dp = {(i, j): D[(i, perm.get(j, j))] for (i, j) in D if (i, perm.get(j, j)) in D}
        t = retrieval_auc(Dp, queries, cand, positions)
        out.append(float(t["auc"].mean()) if t.height else NAN)
    return np.asarray(out)


# ============================================================================== bootstrap uncertainty of an athlete's own distance
def own_distance_ci(i: int, comb_att: dict[str, np.ndarray], att_of: dict[int, list[str]], cube: Any, grid: Grid, method: str, c: M12Config, seed: int,
                    state: str = "all") -> dict[str, Any]:
    """Resample the athlete's Combine attempts and NFL games (with replacement) and recompute d(Combine_i, NFL_i)."""
    rng = np.random.default_rng(seed + i)
    atts = att_of.get(i, [])
    if not atts or cube is None or cube.games.size < 3:
        return {"nfl_id": i, "n_boot": 0}
    vals = []
    for _ in range(c.ot_boot):
        P = sum(comb_att[a] for a in rng.choice(atts, len(atts)))
        Hn = cube.H[rng.integers(0, cube.games.size, cube.games.size)].sum(0).astype(float)
        vals.append(distance(method, P, grid.restrict(Hn, state), grid, c, seed=seed))
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return {"nfl_id": i, "n_boot": int(v.size), "d_lo": float(np.quantile(v, 0.025)) if v.size >= 10 else NAN, "d_hi": float(np.quantile(v, 0.975)) if v.size >= 10 else NAN,
            "d_median": float(np.median(v)) if v.size else NAN, "n_attempts": len(atts), "n_games": int(cube.games.size)}


# ============================================================================== fold-fitted distributional features (ladder block)
def ot_block(analysis_ids: np.ndarray, comb: dict[int, np.ndarray], nfl: dict[int, np.ndarray], positions: dict[int, str], grid: Grid, c: M12Config,
             methods: Sequence[str] = ("E5_conditional_w1", "E3_sliced_w2")) -> ae.BlockFn:
    """Per athlete: distance of its Combine distribution to the mean NFL distribution of TRAINING athletes of its position
    (the reference is refitted per fold; the athlete's own NFL data never enter its feature)."""
    def block(train: np.ndarray) -> pl.DataFrame:
        tr_ids = [int(i) for i in analysis_ids[train]]
        ref = {}
        for p in set(positions.get(i) for i in tr_ids):
            mem = [nfl[i] for i in tr_ids if positions.get(i) == p and i in nfl]
            if mem:
                ref[p] = sum(normalise(m) for m in mem) / len(mem)
        cols = {m: [] for m in methods}
        for i in analysis_ids:
            P, R = comb.get(int(i)), ref.get(positions.get(int(i)))
            for m in methods:
                cols[m].append(distance(m, P, R, grid, c) if P is not None and R is not None and P.sum() > 0 else None)
        return pl.DataFrame({f"ot_{m.split('_')[0]}": pl.Series(v, dtype=pl.Float64) for m, v in cols.items()})
    return block


__all__ = ["ACCEL_RANGE", "FEATURE_PAIRS", "Grid", "MERGE", "METHODS", "auc_summary", "combine_hists", "conditional_w1", "cost_matrix", "distance", "exact_w2",
           "fdist", "feature_vectors", "mahalanobis_metric", "nfl_hists", "normalise", "own_distance_ci", "pairwise", "paired_auc_diff", "pot_available",
           "retrieval_auc", "shuffled_null", "sinkhorn", "sinkhorn_divergence", "sliced_w2", "ot_block", "unbalanced"]
