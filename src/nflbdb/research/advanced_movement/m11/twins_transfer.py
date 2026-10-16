"""11E — stopwatch-twin discordance: within pairs of same-position prospects with near-identical official forty times
(the 09C rule, reused), do Combine tracking differences point the same way as NFL movement differences?

    ΔCombine = Combine_b − Combine_a · ΔNFL = NFL_b − NFL_a   (a < b by athlete id, the 09C orientation)

Statistics (all DESCRIPTIVE associations, never causal)
    concordance        share of pairs where sign(ΔCombine) = expected_sign × sign(ΔNFL) (ties excluded)
    delta_spearman     rank correlation of ΔCombine and ΔNFL (oriented by the expected sign)
    conditional_slope  ΔNFL ~ ΔCombine + Δforty + Δten through the origin (pair orientation is arbitrary)

Dependence (one athlete is in many pairs)
    cluster bootstrap  athletes resampled; a pair enters with weight m(a)·m(b) (``evaluation.cluster_bootstrap_pairs``)
    components         union-find connected components of the pair graph; component bootstrap as a sensitivity
    disjoint matching  greedy matching in which every athlete appears at most once (closest forty first) — independent pairs

Controls: random same-position pairs (same number of pairs), wrong feature, NFL values reshuffled within position
(pair structure kept), reliability-weighted pairs (weights 1 / (SE_a² + SE_b²) of the Combine value).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...studies.stats import rankdata
from . import evaluation as ev
from .config import M11Config

NAN = float("nan")


@dataclass(frozen=True)
class TwinPair:
    combine: str
    nfl: str
    expected_sign: int      # +1: a larger Combine value should go with a larger NFL value; −1: the opposite
    kind: str               # matched · control
    note: str


TWIN_PAIRS: tuple[TwinPair, ...] = (
    TwinPair("forty.band_4_6_mean_accel", "nfl.band_4_6", 1, "matched", "same band"),
    TwinPair("forty.band_6_8_mean_accel", "nfl.band_6_8", 1, "matched", "same band"),
    TwinPair("forty.band_2_4_mean_accel", "nfl.band_2_4", 1, "matched", "same band"),
    TwinPair("forty.phase_A0", "env.lin_A0", 1, "matched", "sprint intercept vs envelope intercept"),
    TwinPair("forty.phase_V0", "env.lin_V0", 1, "matched", "sprint speed intercept vs envelope speed intercept"),
    TwinPair("forty.v_max_obs", "nfl.vmax_p90", 1, "matched", "tracked top speed vs observed game peak"),
    TwinPair("forty.early_accel", "nfl.accel_low_mean", 1, "matched", "early acceleration vs acceleration from low speed"),
    TwinPair("short_shuttle.decel_mean", "nfl.brake_decel", 1, "matched", "non-forty drill (supported pairs only)"),
    TwinPair("forty.band_2_4_mean_accel", "nfl.band_6_8", 1, "control", "WRONG FEATURE: early band vs late NFL band"),
    TwinPair("forty.band_6_8_mean_accel", "nfl.brake_decel", 1, "control", "WRONG FEATURE: Combine band vs NFL braking"),
)


# ============================================================================== pair tables
def pair_frame(twins: pl.DataFrame, values: pl.DataFrame, combine: str, nfl: str, se: pl.DataFrame | None = None) -> pl.DataFrame:
    """Pairs where BOTH athletes have the Combine feature and the NFL target. ``values``: one row per athlete (wide)."""
    v = values.select("nfl_id", pl.col(combine).cast(pl.Float64).alias("c"), pl.col(nfl).cast(pl.Float64).alias("n"))
    p = (twins.select("pair_id", "position_group", "nfl_id_a", "nfl_id_b", "d_forty", "d_ten", "d_weight")
         .join(v.rename({"nfl_id": "nfl_id_a", "c": "c_a", "n": "n_a"}), on="nfl_id_a", how="inner")
         .join(v.rename({"nfl_id": "nfl_id_b", "c": "c_b", "n": "n_b"}), on="nfl_id_b", how="inner")
         .drop_nulls(["c_a", "c_b", "n_a", "n_b"])
         .with_columns((pl.col("c_b") - pl.col("c_a")).alias("d_combine"), (pl.col("n_b") - pl.col("n_a")).alias("d_nfl")))
    if se is not None and combine in se.columns:
        s = se.select("nfl_id", pl.col(combine).alias("se"))
        p = (p.join(s.rename({"nfl_id": "nfl_id_a", "se": "se_a"}), on="nfl_id_a", how="left").join(s.rename({"nfl_id": "nfl_id_b", "se": "se_b"}), on="nfl_id_b", how="left"))
    return p


def _wconcord(dc: np.ndarray, dn: np.ndarray, sign: int, w: np.ndarray) -> float:
    s = np.sign(dc) * np.sign(dn) * sign
    m = s != 0
    return float((w[m] * (s[m] > 0)).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else NAN


def _wspearman(dc: np.ndarray, dn: np.ndarray, sign: int, w: np.ndarray) -> float:
    a, b = rankdata(dc), rankdata(sign * dn)
    W = w / w.sum()
    ma, mb = (W * a).sum(), (W * b).sum()
    cov = (W * (a - ma) * (b - mb)).sum()
    den = np.sqrt((W * (a - ma) ** 2).sum() * (W * (b - mb) ** 2).sum())
    return float(cov / den) if den > 0 else NAN


def _wslope(dc: np.ndarray, dn: np.ndarray, dz: np.ndarray, w: np.ndarray) -> float:
    """Coefficient of ΔCombine in a weighted regression of ΔNFL on [ΔCombine, Δofficials] through the origin, with ΔCombine
    and ΔNFL standardised by their weighted SD (a standardised conditional slope)."""
    X = np.column_stack([dc, dz]) if dz.size else dc[:, None]
    ok = np.isfinite(X).all(1) & np.isfinite(dn) & (w > 0)
    if ok.sum() < X.shape[1] + 3:
        return NAN
    sw = np.sqrt(w[ok])
    sx, sy = np.sqrt(np.average(dc[ok] ** 2, weights=w[ok])), np.sqrt(np.average(dn[ok] ** 2, weights=w[ok]))
    if sx <= 0 or sy <= 0:
        return NAN
    Xs = X[ok].copy()
    Xs[:, 0] /= sx
    beta = np.linalg.lstsq(Xs * sw[:, None], dn[ok] / sy * sw, rcond=None)[0]
    return float(beta[0])


def components(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected-component id of every pair (union-find over athletes)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in zip(a.tolist(), b.tolist()):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
    roots = [find(x) for x in a.tolist()]
    _, comp = np.unique(roots, return_inverse=True)
    return comp


def disjoint_matching(p: pl.DataFrame) -> pl.DataFrame:
    """Greedy matching: pairs in order of |Δforty| (then pair id); a pair is kept only if neither athlete is used."""
    used: set[int] = set()
    keep = []
    for r in p.with_columns(pl.col("d_forty").abs().alias("_d")).sort("_d", "pair_id").iter_rows(named=True):
        if r["nfl_id_a"] in used or r["nfl_id_b"] in used:
            continue
        used |= {r["nfl_id_a"], r["nfl_id_b"]}
        keep.append(r["pair_id"])
    return p.filter(pl.col("pair_id").is_in(keep))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return NAN, NAN
    z, ph = 1.959964, k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(mid - half), float(mid + half)


# ============================================================================== analysis of one feature × target
def analyse(p: pl.DataFrame, sign: int, c: M11Config, seed: int) -> dict[str, Any]:
    n = p.height
    out: dict[str, Any] = {"n_pairs": n, "n_athletes": int(len(set(p["nfl_id_a"].to_list()) | set(p["nfl_id_b"].to_list()))) if n else 0}
    if n < 10:
        return {**out, "status": "too_few_pairs"}
    dc, dn = p["d_combine"].to_numpy(), p["d_nfl"].to_numpy()
    dz = np.column_stack([p[k].cast(pl.Float64).fill_null(0.0).to_numpy() for k in ("d_forty", "d_ten")])
    w1 = np.ones(n)
    a, b = p["nfl_id_a"].to_numpy(), p["nfl_id_b"].to_numpy()
    out |= {"concordance": _wconcord(dc, dn, sign, w1), "delta_spearman": _wspearman(dc, dn, sign, w1), "conditional_slope": _wslope(dc, dn, dz, w1)}
    for name, fn in (("concordance", lambda w: _wconcord(dc, dn, sign, w)), ("delta_spearman", lambda w: _wspearman(dc, dn, sign, w)),
                     ("conditional_slope", lambda w: _wslope(dc, dn, dz, w))):
        lo, hi, _ = ev.cluster_bootstrap_pairs(a, b, fn, c.n_boot, seed)
        out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
    comp = components(a, b)
    out["n_components"] = int(comp.max() + 1)
    rng = np.random.default_rng(seed + 3)
    reps = []
    for _ in range(c.n_boot):
        m = np.bincount(rng.integers(0, comp.max() + 1, comp.max() + 1), minlength=comp.max() + 1).astype(float)
        reps.append(_wconcord(dc, dn, sign, m[comp]))
    reps = np.asarray(reps)
    reps = reps[np.isfinite(reps)]
    out["concordance_comp_lo"], out["concordance_comp_hi"] = (float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))) if reps.size >= 20 else (NAN, NAN)
    dj = disjoint_matching(p)
    s = np.sign(dj["d_combine"].to_numpy()) * np.sign(dj["d_nfl"].to_numpy()) * sign
    k, m_ = int((s > 0).sum()), int((s != 0).sum())
    out |= {"disjoint_pairs": dj.height, "disjoint_concordance": k / m_ if m_ else NAN}
    out["disjoint_lo"], out["disjoint_hi"] = wilson(k, m_)
    if "se_a" in p.columns and "se_b" in p.columns:
        var = p["se_a"].fill_null(np.nan).to_numpy() ** 2 + p["se_b"].fill_null(np.nan).to_numpy() ** 2
        wr = np.where(np.isfinite(var) & (var > 0), 1.0 / var, 0.0)
        out["concordance_reliability_weighted"] = _wconcord(dc, dn, sign, wr) if wr.sum() > 0 else NAN
    lo = out.get("concordance_lo")
    out["reading"] = ("concordant beyond chance (cluster interval > 0.5)" if lo is not None and np.isfinite(lo) and lo > 0.5 else
                      "discordant (cluster interval < 0.5)" if np.isfinite(out.get("concordance_hi", NAN)) and out["concordance_hi"] < 0.5 else "inconclusive")
    out["status"] = "computed"
    return out


def random_pair_control(values: pl.DataFrame, positions: pl.DataFrame, combine: str, nfl: str, n_pairs_by_pos: dict[str, int], sign: int,
                        n_rep: int, seed: int) -> np.ndarray:
    """Concordance of random same-position pairs (as many per position as the twin analysis used)."""
    v = values.join(positions.select("nfl_id", "position_group"), on="nfl_id", how="inner").drop_nulls([combine, nfl])
    rng = np.random.default_rng(seed)
    by = {k[0]: g for k, g in v.partition_by(["position_group"], as_dict=True).items()}
    out = []
    for _ in range(n_rep):
        dc, dn = [], []
        for pos, k in n_pairs_by_pos.items():
            g = by.get(pos)
            if g is None or g.height < 2:
                continue
            i, j = rng.integers(0, g.height, k), rng.integers(0, g.height, k)
            ok = i != j
            cc, nn = g[combine].to_numpy().astype(float), g[nfl].to_numpy().astype(float)
            dc.append(cc[j[ok]] - cc[i[ok]])
            dn.append(nn[j[ok]] - nn[i[ok]])
        if dc:
            a, b = np.concatenate(dc), np.concatenate(dn)
            out.append(_wconcord(a, b, sign, np.ones(a.size)))
    return np.asarray(out, float)


def reshuffle_control(p: pl.DataFrame, values: pl.DataFrame, positions: pl.DataFrame, nfl: str, sign: int, n_rep: int, seed: int) -> np.ndarray:
    """NFL values permuted among athletes of the same position; pairs and Combine differences kept."""
    v = values.join(positions.select("nfl_id", "position_group"), on="nfl_id", how="inner").drop_nulls([nfl])
    ids, pos, y = v["nfl_id"].to_numpy(), v["position_group"].to_numpy(), v[nfl].cast(pl.Float64).to_numpy()
    rng = np.random.default_rng(seed)
    a, b, dc = p["nfl_id_a"].to_numpy(), p["nfl_id_b"].to_numpy(), p["d_combine"].to_numpy()
    out = []
    for _ in range(n_rep):
        yy = y.copy()
        for g in np.unique(pos):
            ix = np.flatnonzero(pos == g)
            yy[ix] = y[rng.permutation(ix)]
        lut = dict(zip(ids.tolist(), yy.tolist()))
        dn = np.array([lut.get(j, np.nan) - lut.get(i, np.nan) for i, j in zip(a.tolist(), b.tolist())])
        ok = np.isfinite(dn)
        out.append(_wconcord(dc[ok], dn[ok], sign, np.ones(int(ok.sum()))))
    return np.asarray(out, float)


def overlap_network(twins: pl.DataFrame) -> dict[str, pl.DataFrame]:
    a, b = twins["nfl_id_a"].to_numpy(), twins["nfl_id_b"].to_numpy()
    comp = components(a, b)
    deg = pl.concat([twins.select(pl.col("nfl_id_a").alias("nfl_id")), twins.select(pl.col("nfl_id_b").alias("nfl_id"))]).group_by("nfl_id").len()
    comp_df = pl.DataFrame({"pair_id": twins["pair_id"], "component": comp}).group_by("component").len().rename({"len": "n_pairs"})
    return {"degree": deg.rename({"len": "n_pairs"}).sort("n_pairs", descending=True), "components": comp_df.sort("n_pairs", descending=True)}


__all__ = ["TWIN_PAIRS", "TwinPair", "analyse", "components", "disjoint_matching", "overlap_network", "pair_frame", "random_pair_control",
           "reshuffle_control", "wilson"]
