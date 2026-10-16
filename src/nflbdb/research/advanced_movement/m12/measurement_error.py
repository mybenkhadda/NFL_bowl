"""12B — measurement-error recovery. NumPy / Polars (PyMC optional for B4).

Measurement model (classical, additive, nondifferential unless a synthetic scenario says otherwise)
    x_ij = θ_i + ε_ij,   ε_ij ~ N(0, σ²_ε)          attempt j of athlete i (Combine)
    θ_i  = μ_pos(i) + γ_xᵀ z_i + u_i,  u_i ~ N(0, σ²_θ)
    y_i  = α + β θ_i + γ_yᵀ z_i + e_i                  NFL physical target (athlete level), z = M1 covariates + position

Estimands, reported SEPARATELY (never one for another)
    observed association       standardized partial slope / partial r of the athlete MEAN x̄_i given z
    corrected (latent) effect  β by regression calibration (RC), SIMEX, joint EM; disattenuated partial r
    out-of-sample value        held-out Δ RMSE of M1 + x̄ (naive, B0) and M1 + E[θ | x̄] (RC, B1) vs M1
A corrected coefficient is NOT a predictive improvement: with equal attempt counts RC is a linear rescaling of x̄ and
cannot change a ridge prediction beyond the penalty; it can only help through unequal attempts and position pooling.

Reliability of the athlete mean with n_i attempts: λ_i = σ²_θ / (σ²_θ + σ²_ε / n_i). With one attempt per athlete σ²_ε
is NOT identifiable — every correction is then marked ``not_identifiable`` (never imputed from another feature).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from ..m11 import evaluation as ev
from . import advanced_evaluation as ae
from .config import M12Config

NAN = float("nan")


@dataclass(frozen=True)
class Prereg:
    feature: str
    target: str
    sign: int
    note: str


PREREGISTERED: tuple[Prereg, ...] = (
    Prereg("forty.band_2_4_mean_accel", "nfl.band_2_4", 1, "acceleration 2→4 yd/s, same band rule both domains"),
    Prereg("forty.band_4_6_mean_accel", "nfl.band_4_6", 1, "acceleration 4→6 yd/s"),
    Prereg("forty.band_6_8_mean_accel", "nfl.band_6_8", 1, "acceleration 6→8 yd/s"),
    Prereg("forty.time_4_8", "nfl.band_4_6", -1, "time from 4 to 8 yd/s (derived per attempt: t_to_8 − t_to_4); shorter = faster build-up"),
    Prereg("forty.v_max_obs", "nfl.vmax_p90", 1, "observed Combine top speed vs NFL observed high-speed peak"),
    Prereg("short_shuttle.decel_mean", "nfl.brake_decel", 1, "turning / braking: shuttle deceleration vs NFL braking deceleration"),
    Prereg("three_cone.transition_cost_index", "nfl.redirect_retention", -1, "turning: speed lost through a cone turn vs NFL speed retained"),
)
PREREG_FEATURES = tuple(p.feature for p in PREREGISTERED)


# ============================================================================== attempt table
def attempt_long(attempts: pl.DataFrame, features: Sequence[str] = PREREG_FEATURES) -> pl.DataFrame:
    """(seq_id, nfl_id, feature, value, onset_censored) for the preregistered features, with ``forty.time_4_8`` derived
    per attempt. ``onset_censored`` is the attempt's forty flag (False for other families)."""
    a = attempts.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
    forty = a.filter(pl.col("family") == "forty").pivot(on="feature", index=["seq_id", "nfl_id"], values="value")
    parts = [a.filter(pl.col("feature").is_in([f for f in features if f != "forty.time_4_8"])).select("seq_id", "nfl_id", "feature", "value")]
    if {"forty.t_to_4", "forty.t_to_8"} <= set(forty.columns) and "forty.time_4_8" in features:
        parts.append(forty.select("seq_id", "nfl_id", pl.lit("forty.time_4_8").alias("feature"), (pl.col("forty.t_to_8") - pl.col("forty.t_to_4")).alias("value"))
                     .filter(pl.col("value").is_not_null() & pl.col("value").is_finite()))
    long = pl.concat(parts, how="diagonal_relaxed")
    cens = forty.select("seq_id", (pl.col("forty.onset_censored") > 0.5).alias("onset_censored")) if "forty.onset_censored" in forty.columns else None
    long = long.join(cens, on="seq_id", how="left") if cens is not None else long.with_columns(pl.lit(None, dtype=pl.Boolean).alias("onset_censored"))
    return long.with_columns(pl.col("onset_censored").fill_null(False))


def athlete_stats(long: pl.DataFrame) -> pl.DataFrame:
    """Per athlete × feature: mean x̄_i, n_i, within SD."""
    return long.group_by("nfl_id", "feature").agg(pl.col("value").mean().alias("xbar"), pl.len().alias("n_att"), pl.col("value").var().alias("s2_within"))


# ============================================================================== variance components
def components(x: np.ndarray, ids: np.ndarray) -> dict[str, float]:
    """Unbalanced one-way method of moments: σ²_ε = pooled within variance (athletes with ≥ 2 attempts);
    σ²_θ = (MSB − MSW) / n0, n0 = (N − Σn²/N)/(k − 1). Not identifiable when no athlete has 2 attempts."""
    m = np.isfinite(x)
    x, ids = x[m], ids[m]
    u, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    k, N = u.size, x.size
    out = {"n_athletes": int(k), "n_repeat": int((cnt >= 2).sum()), "n_obs": int(N), "mean": float(x.mean()) if N else NAN,
           "sigma2_theta": NAN, "sigma2_eps": NAN, "icc": NAN, "identifiable": bool((cnt >= 2).sum() >= 5)}
    if k < 3 or not out["identifiable"]:
        return out
    means = np.bincount(inv, weights=x) / cnt
    ssw = float(((x - means[inv]) ** 2).sum())
    dfw = N - k
    msw = ssw / dfw if dfw > 0 else NAN
    grand = x.mean()
    msb = float((cnt * (means - grand) ** 2).sum() / (k - 1))
    n0 = (N - (cnt ** 2).sum() / N) / (k - 1)
    s2t = max((msb - msw) / n0, 0.0)
    out.update({"sigma2_theta": s2t, "sigma2_eps": msw, "icc": s2t / (s2t + msw) if s2t + msw > 0 else NAN, "mean_attempts": float(cnt.mean())})
    return out


def reliability_table(long: pl.DataFrame, positions: dict[int, str], c: M12Config, seed: int) -> pl.DataFrame:
    """Per feature (pooled and within position): variance components, ICC with athlete-bootstrap CI, reliability of the
    athlete mean at the observed attempt counts (mean λ_i), Spearman–Brown for 1–4 attempts, onset-recorded sensitivity."""
    rows = []
    for (f,), g in sorted(long.partition_by(["feature"], as_dict=True).items()):
        groups = {"ALL": g}
        pos = g.with_columns(pl.col("nfl_id").replace_strict(positions, default="?", return_dtype=pl.String).alias("_p"))
        for (p,), gp in pos.partition_by(["_p"], as_dict=True).items():
            groups[p] = gp.drop("_p")
        if f.startswith("forty."):
            groups["ALL|onset_recorded"] = g.filter(~pl.col("onset_censored"))
        for name, d in groups.items():
            x, ids = d["value"].to_numpy().astype(float), d["nfl_id"].to_numpy()
            comp = components(x, ids)
            if comp["n_athletes"] < 10:
                continue
            ci = ev.repeatability_table(d.select("nfl_id", pl.lit(f).alias("feature"), "value"), n_boot=max(100, c.n_boot // 10), seed=seed)
            lam = NAN
            if comp["identifiable"] and np.isfinite(comp["sigma2_theta"]):
                n_i = d.group_by("nfl_id").len()["len"].to_numpy().astype(float)
                lam = float(np.mean(comp["sigma2_theta"] / (comp["sigma2_theta"] + comp["sigma2_eps"] / n_i)))
            rows.append({"feature": f, "group": name, **comp, "icc_lo": ci["icc_lo"][0] if ci.height else None, "icc_hi": ci["icc_hi"][0] if ci.height else None,
                         "reliability_mean": lam, **{f"sb_{k}": ae.spearman_brown(comp["icc"], k) for k in (1, 2, 3, 4)}})
    return pl.DataFrame(rows, infer_schema_length=None)


# ============================================================================== regression calibration (fold-fitted block)
def rc_values(stats: pl.DataFrame, feature: str, positions: dict[int, str], ref_ids: Sequence[int], long: pl.DataFrame, min_pos: int = 20) -> dict[int, float]:
    """E[θ_i | x̄_i] = μ + λ_i (x̄_i − μ) with μ, σ²_θ, σ²_ε from ``ref_ids`` (training athletes) — within position when the
    position has ≥ ``min_pos`` reference athletes with an identifiable model, else pooled."""
    s = stats.filter(pl.col("feature") == feature)
    lf = long.filter(pl.col("feature") == feature)
    ref = set(int(i) for i in ref_ids)
    pooled = components(lf.filter(pl.col("nfl_id").is_in(list(ref)))["value"].to_numpy().astype(float), lf.filter(pl.col("nfl_id").is_in(list(ref)))["nfl_id"].to_numpy())
    par = {}
    for p in set(positions.values()):
        mem = [i for i in ref if positions.get(i) == p]
        sub = lf.filter(pl.col("nfl_id").is_in(mem))
        comp = components(sub["value"].to_numpy().astype(float), sub["nfl_id"].to_numpy()) if len(mem) >= min_pos else None
        par[p] = comp if comp and comp["identifiable"] and np.isfinite(comp["sigma2_theta"]) else pooled
    out = {}
    for r in s.iter_rows(named=True):
        comp = par.get(positions.get(int(r["nfl_id"])), pooled)
        if not comp["identifiable"] or not np.isfinite(comp["sigma2_theta"]):
            continue
        lam = comp["sigma2_theta"] / (comp["sigma2_theta"] + comp["sigma2_eps"] / r["n_att"])
        out[int(r["nfl_id"])] = comp["mean"] + lam * (r["xbar"] - comp["mean"])
    return out


def rc_block(analysis_ids: np.ndarray, stats: pl.DataFrame, long: pl.DataFrame, feature: str, positions: dict[int, str]) -> ae.BlockFn:
    def block(train: np.ndarray) -> pl.DataFrame:
        v = rc_values(stats, feature, positions, analysis_ids[train].tolist(), long)
        return pl.DataFrame({"rc": [v.get(int(i), None) for i in analysis_ids]}, schema={"rc": pl.Float64})
    return block


# ============================================================================== coefficient estimands (full cohort, descriptive)
def _resid(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X = np.c_[np.ones(len(v)), Z]
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def _covariates(d: pl.DataFrame, cols: Sequence[str]) -> np.ndarray:
    blocks = []
    for k in cols:
        x = d[k].cast(pl.Float64).fill_null(np.nan).to_numpy()
        med = np.nanmedian(x) if np.isfinite(x).any() else 0.0
        blocks.append(np.where(np.isfinite(x), x, med))
        if not np.isfinite(x).all():
            blocks.append((~np.isfinite(x)).astype(float))
    pos = d["position_group"].fill_null("?").to_numpy()
    blocks += [(pos == p).astype(float) for p in sorted(set(pos.tolist()))[1:]]
    return np.column_stack(blocks) if blocks else np.zeros((d.height, 0))


def estimands(d: pl.DataFrame, x_col: str, y_col: str, n_col: str, s2e: float, cov_cols: Sequence[str], c: M12Config, seed: int,
              rel_y: float = NAN, simex: bool = True) -> dict[str, Any]:
    """Standardised estimands (after partialling z) with athlete-bootstrap CIs: beta_obs per SD of the observed mean x̄;
    beta_rc and beta_simex per SD of the LATENT θ (SD(x̄)·√reliability), so they are comparable with the planted / EM
    standardized β; r_partial; r_disattenuated."""
    d = d.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
    n = d.height
    out = {"n_athletes": n}
    if n < 20 or not np.isfinite(s2e):
        return out | {"status": "insufficient_or_not_identifiable"}
    x, y = d[x_col].to_numpy().astype(float), d[y_col].to_numpy().astype(float)
    ni = d[n_col].to_numpy().astype(float)
    Z = _covariates(d, cov_cols)
    rng = np.random.default_rng(seed)

    def stats(ix: np.ndarray, with_simex: bool) -> dict[str, float]:
        rx, ry = _resid(x[ix], Z[ix]), _resid(y[ix], Z[ix])
        sx, sy = rx.std(), ry.std()
        if sx < 1e-12 or sy < 1e-12:
            return {}
        b = float((rx * ry).mean() / (rx * rx).mean()) * sx / sy
        r = float(np.corrcoef(rx, ry)[0, 1])
        err = float(np.mean(s2e / ni[ix]))
        relp = max((rx.var() - err) / rx.var(), 1e-3)
        # latent standardisation: raw RC slope b_raw / relp, scaled by SD(θ | z) = SD(x̄ | z)·√relp → b_obs_std / √relp
        s = {"beta_obs": b, "r_partial": r, "rel_x_partial": relp, "beta_rc": b / math.sqrt(relp)}
        s["r_disattenuated"] = r / math.sqrt(relp * rel_y) if np.isfinite(rel_y) and rel_y > 0 else NAN
        if with_simex:
            lams = np.r_[0.0, np.asarray(c.simex_lambdas)]
            bl = [b]
            for lam in c.simex_lambdas:
                noise = rng.standard_normal((c.simex_reps, ix.size)) * np.sqrt(lam * s2e / ni[ix])
                vals = []
                for e in noise:
                    rxe = _resid(x[ix] + e, Z[ix])
                    vals.append(float((rxe * ry).mean() / (rxe * rxe).mean()) * sx / sy)
                bl.append(float(np.mean(vals)))
            co = np.polyfit(lams, np.asarray(bl), 2)
            s["beta_simex"] = float(np.polyval(co, -1.0)) * math.sqrt(relp)      # same latent standardisation as RC
        return s

    base = stats(np.arange(n), simex)
    reps = [stats(rng.integers(0, n, n), simex and k < max(30, c.n_boot // 20)) for k in range(max(100, c.n_boot // 5))]
    for k in base:
        arr = np.array([r.get(k, np.nan) for r in reps], float)
        arr = arr[np.isfinite(arr)]
        out[k] = base[k]
        out[f"{k}_lo"], out[f"{k}_hi"] = (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))) if arr.size >= 20 else (NAN, NAN)
    out["status"] = "computed"
    return out


# ============================================================================== joint latent-variable regression (EM)
def em_latent(x_long: Sequence[np.ndarray], y: np.ndarray, Z: np.ndarray, iters: int = 300, tol: float = 1e-8) -> dict[str, Any]:
    """EM for x_ij = θ_i + ε, θ_i = Zγx + u_i, y_i = Zγy + β θ_i + e_i (Gaussian). Returns β, σ²_θ, σ²_ε, σ²_e,
    log-likelihood trace length and an identification flag (needs ≥ 5 athletes with 2+ attempts)."""
    n = len(x_long)
    ni = np.array([len(v) for v in x_long], float)
    sx = np.array([np.sum(v) for v in x_long])
    X1 = np.c_[np.ones(n), Z]
    xbar = sx / ni
    gx = np.linalg.lstsq(X1, xbar, rcond=None)[0]
    within = [np.var(v, ddof=1) for v in x_long if len(v) >= 2]
    if len(within) < 5:
        return {"identifiable": False, "beta": NAN}
    s2e = float(np.mean(within))
    s2t = max(float(np.var(xbar - X1 @ gx)) - s2e * float(np.mean(1 / ni)), 1e-4)
    D = np.c_[X1, xbar]
    co = np.linalg.lstsq(D, y, rcond=None)[0]
    gy, beta = co[:-1], float(co[-1])
    s2y = float(np.var(y - D @ co))
    it = 0
    for it in range(iters):
        mx = X1 @ gx
        prec = 1 / s2t + ni / s2e + beta ** 2 / s2y
        V = 1 / prec
        m = V * (mx / s2t + sx / s2e + beta * (y - X1 @ gy) / s2y)
        Et2 = V + m ** 2
        gx_new = np.linalg.lstsq(X1, m, rcond=None)[0]
        s2t_new = float(np.mean(Et2 - 2 * m * (X1 @ gx_new) + (X1 @ gx_new) ** 2))
        ss = sum(float(np.sum(v ** 2) - 2 * m[i] * np.sum(v) + len(v) * Et2[i]) for i, v in enumerate(x_long))
        s2e_new = ss / float(ni.sum())
        A = np.block([[X1.T @ X1, (X1.T @ m)[:, None]], [(m @ X1)[None, :], np.array([[Et2.sum()]])]])
        rhs = np.r_[X1.T @ y, m @ y]
        sol = np.linalg.solve(A + 1e-9 * np.eye(A.shape[0]), rhs)
        gy_new, beta_new = sol[:-1], float(sol[-1])
        r = y - X1 @ gy_new
        s2y_new = float(np.mean(r ** 2 - 2 * beta_new * r * m + beta_new ** 2 * Et2))
        delta = abs(beta_new - beta) + abs(s2t_new - s2t) + abs(s2e_new - s2e)
        gx, s2t, s2e, gy, beta, s2y = gx_new, max(s2t_new, 1e-8), max(s2e_new, 1e-8), gy_new, beta_new, max(s2y_new, 1e-8)
        if delta < tol:
            break
    return {"identifiable": True, "beta": beta, "sigma2_theta": s2t, "sigma2_eps": s2e, "sigma2_y": s2y, "iterations": it + 1,
            "beta_std": beta * math.sqrt(s2t) / math.sqrt(float(np.var(y))) if np.var(y) > 0 else NAN}


def em_estimand(d: pl.DataFrame, long: pl.DataFrame, feature: str, y_col: str, cov_cols: Sequence[str], c: M12Config, seed: int) -> dict[str, Any]:
    """Standardised EM β (per SD of θ and y) with an athlete-bootstrap CI (bounded replicates)."""
    d = d.filter(pl.col(y_col).is_not_null())
    lf = long.filter(pl.col("feature") == feature)
    groups = {int(i): g["value"].to_numpy().astype(float) for (i,), g in lf.partition_by(["nfl_id"], as_dict=True).items()}
    d = d.filter(pl.col("nfl_id").is_in(list(groups)))
    if d.height < 20:
        return {"status": "insufficient", "n_athletes": d.height}
    xs = [groups[int(i)] for i in d["nfl_id"].to_list()]
    y = d[y_col].to_numpy().astype(float)
    Z = _covariates(d, cov_cols)
    base = em_latent(xs, y, Z, c.em_iters)
    if not base["identifiable"]:
        return {"status": "not_identifiable", "n_athletes": d.height}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(max(50, c.n_boot // 10)):
        ix = rng.integers(0, d.height, d.height)
        r = em_latent([xs[i] for i in ix], y[ix], Z[ix], max(60, c.em_iters // 3))
        if r["identifiable"] and np.isfinite(r["beta_std"]):
            bs.append(r["beta_std"])
    bs = np.asarray(bs)
    return {"status": "computed", "n_athletes": d.height, "beta_em": base["beta_std"], "beta_em_lo": float(np.quantile(bs, 0.025)) if bs.size >= 20 else NAN,
            "beta_em_hi": float(np.quantile(bs, 0.975)) if bs.size >= 20 else NAN, "em_iterations": base["iterations"],
            "em_sigma2_theta": base["sigma2_theta"], "em_sigma2_eps": base["sigma2_eps"]}


def compatible_latent(r_lo: float, r_hi: float, rel_x: float, rel_y: float, n: int, k: int) -> dict[str, float]:
    """Latent partial correlations compatible with the observed interval (÷ √(rel_x rel_y), clipped) and the smallest latent
    correlation detectable at this n (minimum detectable observed r ÷ √(rel_x rel_y))."""
    den = math.sqrt(rel_x * rel_y) if np.isfinite(rel_x) and np.isfinite(rel_y) and rel_x > 0 and rel_y > 0 else NAN
    clip = lambda v: float(np.clip(v, -1, 1)) if np.isfinite(v) else NAN
    mde = ev.mde_correlation(n, k)
    return {"latent_lo": clip(r_lo / den), "latent_hi": clip(r_hi / den), "mde_observed": mde, "mde_latent": clip(mde / den)}


# ============================================================================== synthetic recovery experiments
@dataclass(frozen=True)
class Scenario:
    name: str
    icc: float
    attempts: int
    beta: float
    n: int = 300
    corr_err: float = 0.0          # correlation of the athlete's mean attempt error with the NFL error (differential error)
    mnar: float = 0.0              # strength of θ-dependent loss of the 2nd+ attempt (low θ → fewer repeats)
    positions: int = 1             # number of positions with shifted means
    rel_y: float = 0.8


def scenario_grid() -> list[Scenario]:
    out = [Scenario(f"icc{icc}_k{k}_b{b}", icc, k, b) for icc in (0.2, 0.4, 0.6, 0.8) for k in (1, 2, 3) for b in (0.0, 0.3)]
    for b in (0.0, 0.3):
        out += [Scenario(f"correlated_err_b{b}", 0.5, 2, b, corr_err=0.3), Scenario(f"mnar_b{b}", 0.5, 2, b, mnar=1.5),
                Scenario(f"positions_b{b}", 0.5, 2, b, positions=4), Scenario(f"small_n_b{b}", 0.5, 2, b, n=120)]
    return out


def simulate(s: Scenario, rng: np.random.Generator) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    """(long attempts, athlete frame with y and covariates, true standardized β). θ ~ N(μ_pos, 1); ε variance from ICC;
    y = β θ + e with Var(e) such that the NFL target has reliability ``rel_y`` around its true part."""
    pos = rng.integers(0, s.positions, s.n)
    mu = np.linspace(-1, 1, s.positions)[pos] if s.positions > 1 else np.zeros(s.n)
    theta = mu + rng.standard_normal(s.n)
    s2e = (1 - s.icc) / s.icc
    k = np.full(s.n, s.attempts)
    if s.mnar > 0 and s.attempts > 1:
        p_keep = 1 / (1 + np.exp(-s.mnar * (theta - mu)))
        k = np.where(rng.random(s.n) < p_keep, s.attempts, 1)
    eps_athlete = []
    rows = []
    for i in range(s.n):
        e = rng.standard_normal(k[i]) * math.sqrt(s2e)
        eps_athlete.append(e.mean())
        rows += [{"seq_id": f"{i}-{j}", "nfl_id": i, "feature": "x", "value": float(theta[i] + e[j]), "onset_censored": False} for j in range(k[i])]
    eps_m = np.asarray(eps_athlete)
    true_part = s.beta * (theta - mu) + 0.5 * mu
    var_true = max(np.var(true_part), 1e-6)
    s2y = var_true * (1 - s.rel_y) / s.rel_y if s.beta > 0 else 1.0
    ey = rng.standard_normal(s.n) * math.sqrt(s2y)
    if s.corr_err > 0:
        ey = ey * math.sqrt(1 - s.corr_err ** 2) + s.corr_err * math.sqrt(s2y) * eps_m / max(eps_m.std(), 1e-9)
    y = true_part + ey
    ath = pl.DataFrame({"nfl_id": np.arange(s.n), "position_group": [f"P{p}" for p in pos], "y": y, "draft_year": np.zeros(s.n)})
    sd_th = math.sqrt(np.var(theta - mu))
    beta_std = s.beta * sd_th / math.sqrt(np.var(_resid(y, np.column_stack([(pos == p).astype(float) for p in range(1, s.positions)]) if s.positions > 1 else np.zeros((s.n, 0)))))
    return pl.DataFrame(rows), ath, float(beta_std)


def run_scenario(s: Scenario, c: M12Config, seed: int, reps: int | None = None) -> pl.DataFrame:
    """Per replicate: naive / RC / SIMEX / EM standardized β with CIs (RC and naive by analytic SE: the λ uncertainty is
    ignored — undercoverage, if any, is a reported finding), interval coverage of the true β, rejection of β = 0."""
    rows = []
    small = M12Config(n_boot=200, simex_reps=20, em_iters=120)
    for r in range(c.sim_reps if reps is None else reps):
        rng = np.random.default_rng(seed + 10_007 * r)
        long, ath, beta_true = simulate(s, rng)
        st = athlete_stats(long).filter(pl.col("feature") == "x")
        d = ath.join(st.select("nfl_id", "xbar", "n_att"), on="nfl_id")
        comp = components(long["value"].to_numpy(), long["nfl_id"].to_numpy())
        row = {"scenario": s.name, "rep": r, "beta_true": beta_true, "identifiable": comp["identifiable"]}
        if comp["identifiable"]:
            e = estimands(d, "xbar", "y", "n_att", comp["sigma2_eps"], ["draft_year"], small, seed + r, simex=True)
            for k in ("beta_obs", "beta_rc", "beta_simex"):
                row[k], row[f"{k}_lo"], row[f"{k}_hi"] = e.get(k, NAN), e.get(f"{k}_lo", NAN), e.get(f"{k}_hi", NAN)
            em = em_latent([g["value"].to_numpy() for _, g in sorted(long.partition_by(["nfl_id"], as_dict=True).items())], d.sort("nfl_id")["y"].to_numpy(),
                           np.column_stack([(d.sort("nfl_id")["position_group"] == p).to_numpy().astype(float) for p in sorted(d["position_group"].unique().to_list())[1:]])
                           if d["position_group"].n_unique() > 1 else np.zeros((d.height, 0)), small.em_iters)
            row["beta_em"] = em.get("beta_std", NAN)
        else:
            rx = d["xbar"].to_numpy()
            row["beta_obs"] = float(np.corrcoef(rx, d["y"].to_numpy())[0, 1])
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def scenario_summary(sim: pl.DataFrame) -> pl.DataFrame:
    """Bias, RMSE, coverage of the true β, and the false-positive rate (β_true = 0 & CI excludes 0) per estimator."""
    out = []
    for (sc,), g in sorted(sim.partition_by(["scenario"], as_dict=True).items()):
        bt = g["beta_true"].to_numpy()
        for est in ("beta_obs", "beta_rc", "beta_simex", "beta_em"):
            if est not in g.columns:
                continue
            b = g[est].cast(pl.Float64).to_numpy()
            m = np.isfinite(b)
            row = {"scenario": sc, "estimator": est, "n_reps": int(m.sum()), "identifiable_share": float(g["identifiable"].mean()),
                   "bias": float(np.mean(b[m] - bt[m])) if m.any() else NAN, "rmse": float(np.sqrt(np.mean((b[m] - bt[m]) ** 2))) if m.any() else NAN}
            if f"{est}_lo" in g.columns:
                lo, hi = g[f"{est}_lo"].cast(pl.Float64).to_numpy(), g[f"{est}_hi"].cast(pl.Float64).to_numpy()
                mm = m & np.isfinite(lo) & np.isfinite(hi)
                row["coverage"] = float(np.mean((lo[mm] <= bt[mm]) & (bt[mm] <= hi[mm]))) if mm.any() else NAN
                null = mm & (np.abs(bt) < 1e-12)
                row["false_positive_rate"] = float(np.mean((lo[null] > 0) | (hi[null] < 0))) if null.any() else NAN
            out.append(row)
    return pl.DataFrame(out, infer_schema_length=None)


# ============================================================================== optional Bayesian measurement-error model (B4)
def bayes_available() -> bool:
    try:
        import pymc  # noqa: F401
        return True
    except Exception:                                                    # noqa: BLE001
        return False


def fit_bayes(x_long: Sequence[np.ndarray], y: np.ndarray, Z: np.ndarray, draws: int = 1000, tune: int = 1000, chains: int = 2, seed: int = 0) -> dict[str, Any]:
    """Optional PyMC version of the joint model with weakly informative priors on standardized data:
    γ ~ N(0, 1), β ~ N(0, 1), σ_θ, σ_ε, σ_y ~ HalfNormal(1). Returns the posterior summary of β with R-hat and ESS.
    NOT run by default (cost); the API calls follow PyMC 5 and are not verified in this environment."""
    import arviz as az
    import pymc as pm

    n = len(x_long)
    idx = np.concatenate([np.full(len(v), i) for i, v in enumerate(x_long)])
    xv = np.concatenate(x_long)
    xm, xs = xv.mean(), xv.std() or 1.0
    ym, ys = y.mean(), y.std() or 1.0
    with pm.Model():
        gx = pm.Normal("gx", 0, 1, shape=Z.shape[1]) if Z.shape[1] else 0.0
        gy = pm.Normal("gy", 0, 1, shape=Z.shape[1]) if Z.shape[1] else 0.0
        s_t, s_e, s_y = pm.HalfNormal("sigma_theta", 1), pm.HalfNormal("sigma_eps", 1), pm.HalfNormal("sigma_y", 1)
        beta = pm.Normal("beta", 0, 1)
        theta = pm.Normal("theta", (Z @ gx) if Z.shape[1] else 0.0, s_t, shape=n)
        pm.Normal("x_obs", theta[idx], s_e, observed=(xv - xm) / xs)
        pm.Normal("y_obs", ((Z @ gy) if Z.shape[1] else 0.0) + beta * theta, s_y, observed=(y - ym) / ys)
        tr = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False)
    s = az.summary(tr, var_names=["beta", "sigma_theta", "sigma_eps", "sigma_y"])
    return {"summary": s.reset_index().rename(columns={"index": "parameter"}).to_dict(orient="records"), "r_hat_max": float(s["r_hat"].max()),
            "ess_bulk_min": float(s["ess_bulk"].min())}


__all__ = ["PREREGISTERED", "PREREG_FEATURES", "Prereg", "Scenario", "athlete_stats", "attempt_long", "bayes_available", "compatible_latent", "components",
           "em_estimand", "em_latent", "estimands", "fit_bayes", "rc_block", "rc_values", "reliability_table", "run_scenario", "scenario_grid",
           "scenario_summary", "simulate"]
