"""11D — NFL athletic development trajectories (descriptive; an association with time in the league is not a training effect).

    season_panel          athlete × season × trait rows: value, episodes (exposure), games, eligible plays, years since draft
    feasibility           athletes with ≥ ``min_seasons`` supported seasons per trait → SUPPORTED / LIMITED / NOT_FEASIBLE
    within_change         first → last supported season: Δ, OLS slope per year, Theil–Sen slope (≥ 3 seasons)
    position_adjust       value minus the position × season mean of ALL athletes (league-wide drift and tracking changes)
    mixed_model           random intercept + random slope linear mixed model by EM (NumPy); slope SD with athlete-bootstrap CI
    selection_table       athletes observed in ≥ 2 seasons vs 1: first-season value, Combine values, draft year
    within_season_panel   NOT_FEASIBLE alternative: first vs second half of each season's games

Observation horizons differ by draft cohort (a 2024 draftee cannot have four seasons); every model carries
years-since-draft and the panel reports who is missing from later seasons. Age is not in the warehouse: years since
the draft is the time axis.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from . import evaluation as ev
from .config import LIMITED, M11Config, NOT_FEASIBLE, SUPPORTED
from .nfl_physics import TargetSpec, athlete_targets, target

NAN = float("nan")
PANEL_TARGETS = ("nfl.band_4_6", "nfl.band_6_8", "nfl.accel_low_mean", "nfl.vmax_p90", "nfl.brake_decel", "nfl.reaccel_mean")
DEMANDS_FOR_DIVERSITY = ("accel_low", "accel_mid", "transition_high", "sustained_high", "speed_maintenance", "braking", "reaccel_after_brake", "redirection")


def season_panel(episodes: pl.DataFrame, pairs: pl.DataFrame, plays: pl.DataFrame, registry: pl.DataFrame, env_season: pl.DataFrame | None,
                 c: M11Config, targets: Sequence[str] = PANEL_TARGETS) -> pl.DataFrame:
    """Long panel: nfl_id, season, trait, value, se, n_episodes, n_games, n_plays, position_group, draft_year, years_since_draft."""
    specs: list[TargetSpec] = [target(t) for t in targets]
    t = athlete_targets(episodes, pairs, None, c, specs, by_season=True)
    parts = []
    if t.height:
        parts.append(t.select("nfl_id", "season", pl.col("target_id").alias("trait"), "value", "n_episodes", "n_games", "supported"))
    plays_s = plays.filter(pl.col("eligible")).group_by("nfl_id", "season").agg(pl.len().cast(pl.Int64).alias("n_plays"),
                                                                                 pl.col("game_id").n_unique().cast(pl.Int64).alias("n_games_all"))
    acc = episodes.filter(pl.col("demand_id").is_in(["accel_low", "accel_mid", "transition_high", "reaccel_after_brake"])).group_by("nfl_id", "season").len()
    hs = episodes.filter(pl.col("demand_id") == "sustained_high").group_by("nfl_id", "season").len()
    rate = (plays_s.join(acc.rename({"len": "_acc"}), on=["nfl_id", "season"], how="left").join(hs.rename({"len": "_hs"}), on=["nfl_id", "season"], how="left")
            .with_columns((100.0 * pl.col("_acc").fill_null(0) / pl.col("n_plays")).alias("burst_per_100"),
                          (100.0 * pl.col("_hs").fill_null(0) / pl.col("n_plays")).alias("high_speed_per_100")))
    for col, name in (("burst_per_100", "nfl.burst_per_100_plays"), ("high_speed_per_100", "nfl.high_speed_per_100_plays")):
        parts.append(rate.select("nfl_id", "season", pl.lit(name).alias("trait"), pl.col(col).alias("value"), pl.col("n_plays").alias("n_episodes"),
                                 pl.col("n_games_all").alias("n_games"), (pl.col("n_plays") >= 5 * c.min_episodes_season).alias("supported")))
    mix = (episodes.filter(pl.col("demand_id").is_in(list(DEMANDS_FOR_DIVERSITY))).group_by("nfl_id", "season", "demand_id").len()
           .with_columns((pl.col("len") / pl.col("len").sum().over("nfl_id", "season")).alias("_p"))
           .group_by("nfl_id", "season").agg((-(pl.col("_p") * pl.col("_p").log())).sum().alias("value"), pl.col("len").sum().alias("n_episodes")))
    parts.append(mix.select("nfl_id", "season", pl.lit("nfl.movement_diversity").alias("trait"), "value", pl.col("n_episodes").cast(pl.Int64),
                            pl.lit(None, dtype=pl.Int64).alias("n_games"), (pl.col("n_episodes") >= 5 * c.min_episodes_season).alias("supported")))
    if env_season is not None and env_season.height:
        for col in ("lin_A0", "lin_V0"):
            parts.append(env_season.select("nfl_id", "season", pl.lit(f"env.{col}").alias("trait"), pl.col(col).alias("value"),
                                           pl.col("n_frames").cast(pl.Int64).alias("n_episodes"), pl.col("n_games").cast(pl.Int64), pl.col("supported")))
    panel = pl.concat([p.with_columns(pl.col("n_episodes").cast(pl.Int64), pl.col("n_games").cast(pl.Int64)) for p in parts], how="diagonal_relaxed")
    reg = registry.select("nfl_id", "position_group", "draft_year")
    return (panel.join(reg, on="nfl_id", how="left").join(plays_s.select("nfl_id", "season", "n_plays"), on=["nfl_id", "season"], how="left")
            .with_columns((pl.col("season") - pl.col("draft_year")).alias("years_since_draft"))
            .filter(pl.col("value").is_finite()).sort("trait", "nfl_id", "season"))


def feasibility(panel: pl.DataFrame, c: M11Config) -> pl.DataFrame:
    s = panel.filter(pl.col("supported")).group_by("trait", "nfl_id").agg(pl.col("season").n_unique().alias("k"))
    g = s.group_by("trait").agg((pl.col("k") >= c.min_seasons).sum().cast(pl.Int64).alias("n_athletes_multi_season"),
                                (pl.col("k") >= 3).sum().cast(pl.Int64).alias("n_athletes_ge3"), pl.len().cast(pl.Int64).alias("n_athletes_any"))
    return g.with_columns(pl.when(pl.col("n_athletes_multi_season") >= c.min_athletes_cell).then(pl.lit(SUPPORTED))
                          .when(pl.col("n_athletes_multi_season") >= c.min_athletes_limited).then(pl.lit(LIMITED)).otherwise(pl.lit(NOT_FEASIBLE)).alias("state")).sort("trait")


def position_adjust(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.with_columns((pl.col("value") - pl.col("value").mean().over("trait", "position_group", "season")).alias("value_adj"))


def theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    s = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(x.size) for j in range(i + 1, x.size) if x[j] != x[i]]
    return float(np.median(s)) if s else NAN


def within_change(panel: pl.DataFrame, value: str = "value") -> pl.DataFrame:
    rows = []
    for (trait, i), g in panel.filter(pl.col("supported")).sort("season").partition_by(["trait", "nfl_id"], as_dict=True).items():
        if g.height < 2:
            continue
        x, y = g["years_since_draft"].cast(pl.Float64).fill_null(np.nan).to_numpy(), g[value].to_numpy().astype(float)
        if not np.isfinite(x).all():
            x = g["season"].to_numpy().astype(float)
        rows.append({"trait": trait, "nfl_id": i, "position_group": g["position_group"][0], "n_seasons": g.height, "first_season": int(g["season"][0]),
                     "first_value": float(y[0]), "last_value": float(y[-1]), "delta": float(y[-1] - y[0]),
                     "slope_ols": float(np.polyfit(x, y, 1)[0]) if np.unique(x).size >= 2 else NAN, "slope_theil_sen": theil_sen(x, y) if g.height >= 3 else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


# ============================================================================== mixed model (random intercept + slope, EM)
def lmm_em(y: np.ndarray, t: np.ndarray, groups: np.ndarray, iters: int = 200, tol: float = 1e-7) -> dict[str, Any]:
    """y_ij = b0 + b1 t_ij + u0_i + u1_i t_ij + e_ij, (u0, u1) ~ N(0, D), e ~ N(0, σ²). ML by EM with closed-form updates."""
    y, t = np.asarray(y, float), np.asarray(t, float)
    ids = np.unique(groups)
    idx = [np.flatnonzero(groups == g) for g in ids]
    X = np.c_[np.ones_like(t), t]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    s2 = float(r.var()) or 1e-3
    D = np.diag([max(float(r.var()) / 2, 1e-4), 1e-3])
    ll_old = -np.inf
    for _ in range(iters):
        sum_uu, sum_res, XtX, Xty = np.zeros((2, 2)), 0.0, np.zeros((2, 2)), np.zeros(2)
        ll = 0.0
        Dinv = np.linalg.inv(D)
        U = []
        for ix in idx:
            Z = X[ix]
            ri = y[ix] - Z @ beta
            P = np.linalg.inv(Z.T @ Z / s2 + Dinv)
            u = P @ Z.T @ ri / s2
            U.append((u, P))
            sum_uu += np.outer(u, u) + P
            e = ri - Z @ u
            sum_res += float(e @ e + np.trace(Z @ P @ Z.T))
            V = Z @ D @ Z.T + s2 * np.eye(ix.size)
            sign, logdet = np.linalg.slogdet(V)
            ll += -0.5 * (logdet + ri @ np.linalg.solve(V, ri) + ix.size * np.log(2 * np.pi))
        D = sum_uu / len(idx) + 1e-9 * np.eye(2)
        s2 = sum_res / y.size
        for ix, (u, _) in zip(idx, U):
            Z = X[ix]
            XtX += Z.T @ Z
            Xty += Z.T @ (y[ix] - Z @ u)
        beta = np.linalg.solve(XtX, Xty)
        if abs(ll - ll_old) < tol * (1 + abs(ll)):
            break
        ll_old = ll
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "sd_intercept": float(np.sqrt(max(D[0, 0], 0))), "sd_slope": float(np.sqrt(max(D[1, 1], 0))),
            "corr_int_slope": float(D[0, 1] / np.sqrt(D[0, 0] * D[1, 1])) if D[0, 0] > 0 and D[1, 1] > 0 else NAN, "sd_residual": float(np.sqrt(s2)),
            "loglik": float(ll_old), "n_athletes": len(idx), "n_obs": int(y.size)}


def mixed_model_table(panel: pl.DataFrame, c: M11Config, value: str = "value_adj", n_boot: int = 200, seed: int = 0, by_position: bool = False) -> pl.DataFrame:
    rows = []
    keys = ["trait", "position_group"] if by_position else ["trait"]
    rng = np.random.default_rng(seed)
    for key, g in sorted(panel.filter(pl.col("supported") & pl.col("years_since_draft").is_not_null()).partition_by(keys, as_dict=True).items()):
        multi = g.group_by("nfl_id").len().filter(pl.col("len") >= c.min_seasons)["nfl_id"]
        g = g.filter(pl.col("nfl_id").is_in(multi.to_list()))
        n_ath = g["nfl_id"].n_unique()
        if n_ath < c.min_athletes_limited:
            rows.append({**dict(zip(keys, key)), "n_athletes": n_ath, "status": NOT_FEASIBLE})
            continue
        y, t, ids = g[value].to_numpy().astype(float), g["years_since_draft"].to_numpy().astype(float), g["nfl_id"].to_numpy()
        fit = lmm_em(y, t, ids)
        bs = []
        uniq = np.unique(ids)
        pos = {u: np.flatnonzero(ids == u) for u in uniq}
        for _ in range(n_boot):
            pick = rng.choice(uniq, uniq.size)
            ix = np.concatenate([pos[u] for u in pick])
            lab = np.concatenate([np.full(pos[u].size, j) for j, u in enumerate(pick)])
            bs.append(lmm_em(y[ix], t[ix], lab, iters=60)["sd_slope"])
        bs = np.asarray(bs)
        rows.append({**dict(zip(keys, key)), **fit, "sd_slope_lo": float(np.quantile(bs, 0.025)), "sd_slope_hi": float(np.quantile(bs, 0.975)),
                     "status": "computed"})
    return pl.DataFrame(rows, infer_schema_length=None)


def selection_table(panel: pl.DataFrame, combine_wide: pl.DataFrame, features: Sequence[str]) -> pl.DataFrame:
    """Athletes with ≥ 2 supported seasons vs 1, per trait: first-season value and Combine values (selection into later seasons)."""
    first = panel.filter(pl.col("supported")).sort("season").group_by("trait", "nfl_id").agg(pl.col("value").first().alias("first_value"),
                                                                                           pl.col("season").n_unique().alias("k"), pl.col("draft_year").first())
    d = first.join(combine_wide.select("nfl_id", *[f for f in features if f in combine_wide.columns]), on="nfl_id", how="left").with_columns(
        (pl.col("k") >= 2).alias("multi_season"))
    aggs = [pl.len().alias("n_athletes"), pl.col("first_value").median().alias("first_value_median"), pl.col("draft_year").median().alias("draft_year_median")]
    aggs += [pl.col(f).median().alias(f"{f}_median") for f in features if f in d.columns]
    return d.group_by("trait", "multi_season").agg(aggs).sort("trait", "multi_season")


def combine_relations(change: pl.DataFrame, panel: pl.DataFrame, combine_wide: pl.DataFrame, features: Sequence[str], c: M11Config, seed: int) -> pl.DataFrame:
    """Spearman (bootstrap CI) of Combine features with first-season level, last-season level, slope and |Δ| (stability)."""
    first = panel.filter(pl.col("supported")).sort("season").group_by("trait", "nfl_id").agg(pl.col("value").first().alias("first_level"),
                                                                                           pl.col("value").last().alias("last_level"))
    d = first.join(change.select("trait", "nfl_id", "slope_ols", "delta"), on=["trait", "nfl_id"], how="left").join(combine_wide, on="nfl_id", how="inner")
    rows = []
    for (trait,), g in sorted(d.partition_by(["trait"], as_dict=True).items()):
        for f in features:
            if f not in g.columns:
                continue
            x = g[f].cast(pl.Float64).to_numpy()
            for what, col in (("first_season_level", "first_level"), ("last_season_level", "last_level"), ("slope_per_year", "slope_ols"), ("abs_change", "delta")):
                y = g[col].cast(pl.Float64).to_numpy()
                y = np.abs(y) if what == "abs_change" else y
                r = ev.association(x, y, None, max(200, c.n_boot // 5), seed)
                rows.append({"trait": trait, "combine": f, "relation": what, "n_athletes": r["n_athletes"], "spearman": r["spearman"],
                             "spearman_lo": r["spearman_lo"], "spearman_hi": r["spearman_hi"], "mde_r": r["mde_r"], "reading": ev.association_reading(r)})
    return pl.DataFrame(rows, infer_schema_length=None)


def early_to_later(panel: pl.DataFrame, c: M11Config, seed: int) -> pl.DataFrame:
    """NFL-to-NFL transfer: first supported season vs the mean of later supported seasons (same trait)."""
    rows = []
    for (trait,), g in sorted(panel.filter(pl.col("supported")).sort("season").partition_by(["trait"], as_dict=True).items()):
        a = g.group_by("nfl_id").agg(pl.col("value").first().alias("first"), pl.col("value").slice(1).mean().alias("later"), pl.len().alias("k")).filter(pl.col("k") >= 2)
        r = ev.association(a["first"].to_numpy().astype(float), a["later"].to_numpy().astype(float), None, max(200, c.n_boot // 5), seed)
        rows.append({"trait": trait, "n_athletes": r["n_athletes"], "spearman": r["spearman"], "spearman_lo": r["spearman_lo"], "spearman_hi": r["spearman_hi"]})
    return pl.DataFrame(rows, infer_schema_length=None)


def within_season_panel(episodes: pl.DataFrame, pairs: pl.DataFrame, c: M11Config, targets: Sequence[str] = PANEL_TARGETS) -> pl.DataFrame:
    """Alternative when multi-season data are NOT feasible: each athlete-season split into its first and second half of games
    (by game order); the same targets are computed per half."""
    order = episodes.select("nfl_id", "season", "game_id").unique().with_columns(pl.col("game_id").rank("dense").over("nfl_id", "season").alias("_r"),
                                                                                  pl.col("game_id").n_unique().over("nfl_id", "season").alias("_n"))
    half = order.with_columns(pl.when(pl.col("_r") <= pl.col("_n") / 2).then(pl.lit(1)).otherwise(pl.lit(2)).alias("half")).select("nfl_id", "season", "game_id", "half")
    e = episodes.join(half, on=["nfl_id", "season", "game_id"], how="inner").with_columns((pl.col("season") * 10 + pl.col("half")).alias("season"))
    p = pairs.join(half, on=["nfl_id", "season", "game_id"], how="inner").with_columns((pl.col("season") * 10 + pl.col("half")).alias("season")) if pairs.height else pairs
    t = athlete_targets(e, p, None, c, [target(x) for x in targets], by_season=True)
    return t.with_columns((pl.col("season") // 10).alias("season_year"), (pl.col("season") % 10).alias("half")) if t.height else t


__all__ = ["PANEL_TARGETS", "combine_relations", "early_to_later", "feasibility", "lmm_em", "mixed_model_table", "position_adjust", "season_panel",
           "selection_table", "theil_sen", "within_change", "within_season_panel"]
