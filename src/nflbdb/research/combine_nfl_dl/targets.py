"""Stage TGT — athlete-level NFL targets from repeated get-off observations, and how reliable they are.

Targets (one value per athlete × observation window)
    mean, median, p25, p75     of the raw endpoint [s]
    adj_mean                   mean after removing alignment × season means, plus the grand mean (still seconds:
                               "get-off at the average context"). The value stored here uses the context means of
                               ALL athletes and is DESCRIPTIVE. Inside cross-validation the same quantity is rebuilt
                               from ``athlete_context_cells`` with context means of the training athletes only.
Windows
    all            every valid observation
    first_<n>      the athlete's first n valid observations in time
    no_boundary    observations at the observed maximum or below the low bound removed
Reliability (independent GAMES, never random plays)
    game_odd_even          Pearson between the target computed on an athlete's odd and even games (athlete bootstrap)
    game_odd_even_sb       the same, stepped up to the full number of games (Spearman–Brown)
    game_random_halves_sb  mean over random half-splits of each athlete's games, stepped up; the interval is the
                           spread over splits, NOT a sampling interval
    game_icc1              ICC(1) of athlete-game means (games with ≥ game_min_obs observations)
    play_icc1              ICC(1) of single observations; sb_projected:<n> = Spearman–Brown projection to n plays
                           (assumes exchangeable plays — an upper bound when plays of one game are correlated)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from ..sprint_rescue.stats import bootstrap_ci, seed_for, spearman_brown, stratified_icc
from ..studies.stats import pearson
from .config import TARGETS, GetOffConfig
from .models import Target

S, I, F = pl.String(), pl.Int64(), pl.Float64()
TARGET_TABLE = TableContract(
    "athlete_targets", "Athlete-level get-off targets per observation window (adj_mean here is descriptive).",
    {"nfl_id": I, "window": S, "n_obs": I, "n_games": I, **{t: F for t in TARGETS}, "share_first_alignment": F,
     "alignment_group": S}, key=("nfl_id", "window"), kind="dataset", area="athlete × observation window",
    docs={"adj_mean": "alignment × season means of ALL athletes removed; cross-validation rebuilds it per fold",
          "share_first_alignment": "share of the window's observations in the first configured alignment (EDGE)",
          "alignment_group": "modal alignment over ALL valid observations — an NFL observation, never a predictor"})
CELLS = TableContract(
    "athlete_context_cells", "Sum and count of the endpoint per athlete × window × (alignment, season) cell.",
    {"nfl_id": I, "window": S, "cell": S, "alignment": S, "season": I, "n": I, "total": F},
    key=("nfl_id", "window", "cell"), kind="dataset", area="athlete × window × context cell",
    docs={"total": "sum of player_get_off over the cell [s]; lets a fold recompute context means on training athletes"})
RELIABILITY = TableContract(
    "outcome_reliability", "Reliability of athlete-level get-off targets across independent games.",
    {"target": S, "window": S, "min_obs": I, "method": S, "n_athletes": I, "estimate": F, "ci_lower": F, "ci_upper": F,
     "note": S}, key=("target", "window", "min_obs", "method"), area="target × window × eligibility threshold × method")


def windows(gcfg: GetOffConfig) -> tuple[str, ...]:
    return ("all", *[f"first_{n}" for n in gcfg.first_n_grid], "no_boundary")


def window_plays(plays: pl.DataFrame, window: str) -> pl.DataFrame:
    if window == "all":
        return plays
    if window == "no_boundary":
        return plays.filter(~pl.col("at_max") & ~pl.col("below_low"))
    if window.startswith("first_"):
        return plays.filter(pl.col("obs_idx") <= int(window.split("_")[1]))
    raise ValueError(f"unknown observation window {window!r}")


def add_residual(plays: pl.DataFrame) -> pl.DataFrame:
    """``resid`` = value − mean(alignment, season) + grand mean, all means taken over the rows given."""
    return plays.with_columns((pl.col("value") - pl.col("value").mean().over("alignment", "season")
                               + pl.col("value").mean()).alias("resid"))


def _target_aggs() -> list[pl.Expr]:
    v = pl.col("value")
    return [v.mean().alias("mean"), v.median().alias("median"), v.quantile(0.25, "linear").alias("p25"),
            v.quantile(0.75, "linear").alias("p75"), pl.col("resid").mean().alias("adj_mean")]


# ============================================================================== athlete targets
def athlete_targets(plays: pl.DataFrame, gcfg: GetOffConfig) -> pl.DataFrame:
    if not plays.height:
        return TARGET_TABLE.empty()
    a0 = gcfg.alignments[0]
    other = gcfg.alignments[1] if len(gcfg.alignments) > 1 else a0
    group = plays.group_by("nfl_id").agg((pl.col("alignment") == a0).mean().alias("_s")).select(
        "nfl_id", pl.when(pl.col("_s") >= 0.5).then(pl.lit(a0)).otherwise(pl.lit(other)).alias("alignment_group"))
    parts = []
    for w in windows(gcfg):
        g = add_residual(window_plays(plays, w))
        if not g.height:
            continue
        aggs = [pl.len().cast(pl.Int64).alias("n_obs"), pl.col("game_id").n_unique().cast(pl.Int64).alias("n_games"),
                *_target_aggs(), (pl.col("alignment") == a0).mean().alias("share_first_alignment")]
        parts.append(g.group_by("nfl_id").agg(aggs).with_columns(pl.lit(w).alias("window")))
    out = pl.concat(parts, how="vertical_relaxed").join(group, on="nfl_id", how="left")
    return out.select([pl.col(c).cast(t) for c, t in TARGET_TABLE.columns.items()]).sort("nfl_id", "window")


def context_cells(plays: pl.DataFrame, gcfg: GetOffConfig) -> pl.DataFrame:
    if not plays.height:
        return CELLS.empty()
    parts = []
    for w in windows(gcfg):
        g = window_plays(plays, w)
        if not g.height:
            continue
        parts.append(g.group_by("nfl_id", "alignment", "season")
                     .agg(pl.len().cast(pl.Int64).alias("n"), pl.col("value").sum().alias("total"))
                     .with_columns(pl.lit(w).alias("window"),
                                   (pl.col("alignment") + "|" + pl.col("season").cast(pl.String)).alias("cell")))
    out = pl.concat(parts, how="vertical_relaxed")
    return out.select([pl.col(c).cast(t) for c, t in CELLS.columns.items()]).sort("nfl_id", "window", "cell")


def build_target(targets: pl.DataFrame, cells: pl.DataFrame, ids: Sequence[int], target: str, window: str) -> Target:
    """Outcome of the athletes ``ids`` (in that order). ``adj_mean`` → per-cell sums and counts (fold-aware);
    every other target → fixed values. An athlete without observations in the window gets NaN / zero counts."""
    ids = [int(i) for i in ids]
    if target != "adj_mean":
        vals = dict(targets.filter(pl.col("window") == window).select("nfl_id", target).iter_rows())
        return Target(fixed=np.array([vals.get(i, np.nan) if vals.get(i) is not None else np.nan for i in ids], float))
    c = cells.filter(pl.col("window") == window)
    names = sorted(c["cell"].unique().to_list())
    row, col = {i: k for k, i in enumerate(ids)}, {n: k for k, n in enumerate(names)}
    Sm, Km = np.zeros((len(ids), len(names))), np.zeros((len(ids), len(names)))
    for nfl_id, cell, n, total in c.select("nfl_id", "cell", "n", "total").iter_rows():
        if nfl_id in row:
            Sm[row[nfl_id], col[cell]], Km[row[nfl_id], col[cell]] = total, n
    return Target(S=Sm, K=Km)


# ============================================================================== reliability
def _halves(g: pl.DataFrame, half: np.ndarray) -> pl.DataFrame:
    """Targets on the two halves of every athlete: columns ``<target>_0`` and ``<target>_1``."""
    h = (add_residual(g).with_columns(pl.Series("_h", half.astype(np.int64)))
         .group_by("nfl_id", "_h").agg(_target_aggs()))
    a = h.filter(pl.col("_h") == 0).drop("_h").rename({t: f"{t}_0" for t in TARGETS})
    b = h.filter(pl.col("_h") == 1).drop("_h").rename({t: f"{t}_1" for t in TARGETS})
    return a.join(b, on="nfl_id", how="inner")


def _game_layout(g: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(athlete code per athlete-game, start offset of each athlete, athlete-game row of every play)."""
    games = g.select("nfl_id", "game_id").unique().sort("nfl_id", "game_id").with_row_index("_g")
    row = g.join(games, on=["nfl_id", "game_id"], how="left", maintain_order="left")["_g"].to_numpy().astype(int)
    ath = games["nfl_id"].to_numpy()
    _, start, inv = np.unique(ath, return_index=True, return_inverse=True)
    return inv, start, row


def reliability_table(plays: pl.DataFrame, gcfg: GetOffConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(target: str, window: str, min_obs: int, method: str, n: int, est: Any, lo: Any = None, hi: Any = None,
            note: str = "") -> None:
        f = lambda v: float(v) if v is not None and np.isfinite(float(v)) else None    # noqa: E731
        rows.append({"target": target, "window": window, "min_obs": int(min_obs), "method": method, "n_athletes": int(n),
                     "estimate": f(est), "ci_lower": f(lo), "ci_upper": f(hi), "note": note})

    def sb2(r: float) -> float:
        return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")

    cases = [("all", thr) for thr in gcfg.min_obs_grid] + [(f"first_{n}", n) for n in gcfg.first_n_grid]
    for window, min_obs in cases:
        g = window_plays(plays, window)
        g = g.filter(pl.len().over("nfl_id") >= min_obs).sort("nfl_id", "game_id", "play_id")
        n_ath = g["nfl_id"].n_unique() if g.height else 0
        if n_ath < 10:
            continue
        inv, start, row = _game_layout(g)
        pos = np.arange(inv.size) - start[inv]                       # chronological game position within the athlete
        seed = seed_for(gcfg.seed, "TGT", window, min_obs)
        # ---- odd / even games (deterministic), athlete bootstrap
        pair = _halves(g, (pos % 2)[row])
        for t in TARGETS:
            a, b = pair[f"{t}_0"].to_numpy().astype(float), pair[f"{t}_1"].to_numpy().astype(float)
            r = pearson(a, b)
            lo, hi = bootstrap_ci(lambda i: pearson(a[i], b[i]), a.size, gcfg.n_boot, seed)
            add(t, window, min_obs, "game_odd_even", pair.height, r, lo, hi, "Pearson, odd vs even games; athlete bootstrap")
            add(t, window, min_obs, "game_odd_even_sb", pair.height, sb2(r), sb2(lo), sb2(hi),
                "odd/even stepped up to all games (Spearman–Brown)")
        # ---- random halves of each athlete's games
        rng = np.random.default_rng(seed)
        reps = {t: [] for t in TARGETS}
        n_pairs = 0
        for _ in range(gcfg.n_game_splits):
            order = np.lexsort((rng.random(inv.size), inv))
            rank = np.empty(inv.size, dtype=int)
            rank[order] = np.arange(inv.size) - start[inv[order]]
            pair = _halves(g, (rank % 2)[row])
            n_pairs = pair.height
            for t in TARGETS:
                reps[t].append(sb2(pearson(pair[f"{t}_0"].to_numpy().astype(float), pair[f"{t}_1"].to_numpy().astype(float))))
        for t in TARGETS:
            v = np.array(reps[t], float)
            v = v[np.isfinite(v)]
            if v.size:
                add(t, window, min_obs, "game_random_halves_sb", n_pairs, v.mean(), np.quantile(v, 0.025), np.quantile(v, 0.975),
                    f"mean over {gcfg.n_game_splits} random game half-splits; interval = spread over splits")
        if window != "all":
            continue
        # ---- ICC of athlete-game means and of single observations (mean-type targets only)
        gr = add_residual(g)
        for t, col in (("mean", "value"), ("adj_mean", "resid")):
            gm = (gr.group_by("nfl_id", "game_id").agg(pl.col(col).mean().alias("_m"), pl.len().alias("_n"))
                  .filter(pl.col("_n") >= gcfg.game_min_obs))
            gm = gm.filter(pl.len().over("nfl_id") >= 2)
            if gm.height:
                ic = stratified_icc(gm["_m"].to_numpy(), gm["nfl_id"].to_numpy(), None, gcfg.n_boot, seed)
                add(t, window, min_obs, "game_icc1", ic["n_athletes"], ic["icc"], ic["ci_lower"], ic["ci_upper"],
                    f"athlete-game means, games with ≥ {gcfg.game_min_obs} observations")
            ic = stratified_icc(gr[col].to_numpy(), gr["nfl_id"].to_numpy(), None, gcfg.n_boot, seed)
            add(t, window, min_obs, "play_icc1", ic["n_athletes"], ic["icc"], ic["ci_lower"], ic["ci_upper"], "single observations")
            for n in sorted({*gcfg.min_obs_grid, *gcfg.first_n_grid}):
                add(t, window, min_obs, f"sb_projected:{n}", ic["n_athletes"], spearman_brown(ic["icc"], n),
                    spearman_brown(ic["ci_lower"], n), spearman_brown(ic["ci_upper"], n),
                    "Spearman–Brown from the single-observation ICC (assumes exchangeable plays)")
    return pl.DataFrame(rows, schema=RELIABILITY.columns) if rows else RELIABILITY.empty()


def primary_reliability(reliability: pl.DataFrame, gcfg: GetOffConfig, target: str | None = None) -> float | None:
    """Reliability of the primary target at the primary threshold (random game halves, stepped up): the ceiling
    of the variance any predictor can explain."""
    r = reliability.filter((pl.col("target") == (target or gcfg.primary_target)) & (pl.col("window") == "all")
                           & (pl.col("min_obs") == gcfg.primary_min_obs) & (pl.col("method") == "game_random_halves_sb"))
    return r["estimate"][0] if r.height else None
