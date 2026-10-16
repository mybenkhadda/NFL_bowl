"""Reliability of the candidate NFL movement outcomes across INDEPENDENT GAMES, and the target freeze (Milestone 05).

Every split is a split of an athlete's GAMES, never of frames or plays; every interval resamples ATHLETES.
No Combine value is read: the selection of the targets cannot depend on a Combine association.

Reliability (table ``nfl_outcome_reliability``; group WITHIN_POSITION = position-group means removed in each half)
    game_odd_even            Pearson between the outcome on an athlete's odd and even games (athlete bootstrap)
    game_odd_even_sb         the same, stepped up to all games (Spearman–Brown)
    game_random_halves_sb    mean over random half-splits of each athlete's games, stepped up; the interval is the
                             spread over splits, NOT a sampling interval
    game_icc1 · play_icc1    ICC(1) of athlete-game means / of single windows of the PLAY MEASURE (stratified by position)
    var_athlete · var_game · var_window    variance of the play measure between athletes, between games of an athlete,
                             between windows of a game (one-way moment estimators; approximate for unbalanced data)
    season_<a>_to_<b>        Pearson between the outcome on two consecutive seasons
Exposure (table ``nfl_reliability_vs_exposure``; time order preserved)
    first_vs_later_games     Pearson between the outcome on the athlete's FIRST N windows (or G games) and on the
                             games played strictly AFTER them. Attenuated by the noise of both sides: a lower bound of
                             the reliability of the first-N value, and the quantity a scout would actually face.
    within_budget_game_split_sb    odd / even games inside the first N windows, stepped up
    model_projection         σ²_a / (σ²_a + σ²_g / G + σ²_w / N) from the variance components (mean of the play measure)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import atomic_write_json, get_logger, read_json, stable_hash, utc_now
from ..contracts import TableContract
from ..sprint_rescue.stats import bootstrap_ci, seed_for, stratified_icc
from ..studies.stats import pearson, spearman
from . import nfl_outcomes as no
from .nfl_outcomes import ALL, OutcomeConfig, OutcomeEngine, OutcomeSpec

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
WITHIN = "WITHIN_POSITION"
ADJ = {"raw": "value", "context": "value_adj"}
TARGETS_FILE = "nfl_movement_targets_v1"
DECISIONS = ("FREEZE", "STOP")

RELIABILITY = TableContract(
    "nfl_outcome_reliability", "Reliability of athlete-level NFL movement outcomes across independent games.",
    {"outcome_id": S, "adjustment": S, "group": S, "min_windows": I, "method": S, "n_athletes": I, "estimate": F,
     "ci_lower": F, "ci_upper": F, "median_windows": F, "median_games": F, "note": S},
    key=("outcome_id", "adjustment", "group", "min_windows", "method"), area="outcome × adjustment × group × threshold × method")
EXPOSURE = TableContract(
    "nfl_reliability_vs_exposure", "Reliability of the outcomes at fixed observation budgets (time order preserved).",
    {"outcome_id": S, "budget_type": S, "budget": I, "cohort": S, "method": S, "n_athletes": I, "estimate": F,
     "ci_lower": F, "ci_upper": F, "note": S}, key=("outcome_id", "budget_type", "budget", "cohort", "method"),
    area="outcome × budget × athlete cohort × method",
    docs={"budget": "first N windows or first G games; 0 = all available",
          "cohort": "available = every athlete with the budget · common = the same athletes at every budget"})
CONTEXT = TableContract(
    "nfl_outcome_context_audit", "Play-measure level by observed context, and the share of its variance a context explains.",
    {"outcome_id": S, "position_group": S, "variable": S, "level": S, "n_windows": I, "n_athletes": I, "mean": F, "sd": F,
     "eta2": F, "note": S}, key=("outcome_id", "position_group", "variable", "level"),
    area="outcome × position group × context variable × level",
    docs={"eta2": "level '__ALL__' only: between-level share of the play-measure variance inside the position group",
          "variable": "context_cell = alignment × play class · athlete_adjustment: mean = Pearson(raw, adjusted athlete "
                      "value), sd = SD(adjusted − raw) / SD(raw)"})
SENSITIVITY = TableContract(
    "nfl_outcome_quality_sensitivity", "Athlete-level outcomes under alternative smoothing, boundary handling and quality filters.",
    {"outcome_id": S, "variant": S, "kind": S, "n_athletes": I, "pearson": F, "spearman": F, "mean_shift": F,
     "shift_sd_units": F, "sd_ratio": F, "reliability_primary": F, "reliability_variant": F, "windows_retained": F, "note": S},
    key=("outcome_id", "variant"), area="outcome × variant",
    docs={"mean_shift": "variant − primary, in the unit of the outcome", "shift_sd_units": "mean shift / SD of the primary values",
          "reliability_primary": "odd / even games, stepped up, on the SAME windows as the variant"})
REDUNDANCY = TableContract(
    "nfl_outcome_redundancy", "Within-position Spearman correlation between athlete-level NFL outcomes.",
    {"outcome_a": S, "outcome_b": S, "n_athletes": I, "spearman": F}, key=("outcome_a", "outcome_b"), area="outcome pair")
SCREEN = TableContract(
    "nfl_outcome_screen", "Rule-based screening of the candidate outcomes (heuristic reference lines, for review).",
    {"outcome_id": S, "family": S, "role": S, "n_athletes": I, "n_positions": I, "reliability": F, "reliability_lower": F,
     "reliability_upper": F, "reliability_adj": F, "reliability_min_position": F, "first_n_r": F, "smoothing_min_spearman": F,
     "max_shift_sd": F, "exposure_rho": F, "context_eta2": F, "pass_coverage": B, "pass_reliability": B, "pass_smoothing": B,
     "pass_exposure": B, "pass_context": B, "n_passed": I, "status": S, "reasons": S}, key=("outcome_id",), area="outcome")


# ============================================================================== small statistics
def sb2(r: float) -> float:
    """Spearman–Brown step-up of a half-split correlation to the full length."""
    return float(2 * r / (1 + r)) if r is not None and np.isfinite(r) and r > -1 else float("nan")


def _codes(x: Sequence[Any]) -> np.ndarray:
    return np.unique(np.asarray(x).astype(str), return_inverse=True)[1]


def _centre(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.bincount(codes).astype(float)
        return x - (np.bincount(codes, weights=x) / k)[codes]


def within_r(a: np.ndarray, b: np.ndarray, codes: np.ndarray | None, rank: bool = False) -> float:
    """Correlation after removing group means from both variables (pooled within-group association)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok], b[ok]
    if codes is not None:
        cd = np.asarray(codes)[ok]
        a, b = _centre(a, cd), _centre(b, cd)
    return spearman(a, b) if rank else pearson(a, b)


def _f(v: Any) -> float | None:
    return float(v) if v is not None and np.isfinite(float(v)) else None


def _corr_ci(a: np.ndarray, b: np.ndarray, codes: np.ndarray | None, n_boot: int, seed: int) -> tuple[float, float, float]:
    r = within_r(a, b, codes)
    lo, hi = bootstrap_ci(lambda i: within_r(a[i], b[i], None if codes is None else codes[i]), a.size, n_boot, seed)
    return r, lo, hi


def random_assign(games: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """(nfl_id, game_id, _h): each athlete's games in a random order, alternately assigned to halves 0 and 1."""
    ath = games["nfl_id"].to_numpy()
    _, start, inv = np.unique(ath, return_index=True, return_inverse=True)
    order = np.lexsort((rng.random(inv.size), inv))
    rank = np.empty(inv.size, dtype=np.int64)
    rank[order] = np.arange(inv.size) - start[inv[order]]
    return games.select("nfl_id", "game_id").with_columns(pl.Series("_h", rank % 2, dtype=pl.Int64))


def _eligible_pairs(pair: pl.DataFrame, thr: int, c: OutcomeConfig) -> pl.DataFrame:
    half = max(c.min_half_windows, thr // 4)
    return pair.filter(((pl.col("n_0") + pl.col("n_1")) >= thr) & (pl.min_horizontal("n_0", "n_1") >= half))


def _groups(pair: pl.DataFrame, c: OutcomeConfig) -> list[tuple[str, pl.DataFrame, bool]]:
    out = [(WITHIN, pair, True)] if pair.height >= c.min_athletes else []
    for p in sorted(pair["position_group"].drop_nulls().unique().to_list()):
        sub = pair.filter(pl.col("position_group") == p)
        if sub.height >= c.min_athletes_position:
            out.append((p, sub, False))
    return out


# ============================================================================== reliability across games
def reliability_table(eng: OutcomeEngine, c: OutcomeConfig, specs: Sequence[OutcomeSpec] | None = None) -> pl.DataFrame:
    specs = list(specs) if specs is not None else eng.specs
    rows: list[dict[str, Any]] = []

    def add(oid, adj, group, thr, method, n, est, lo=None, hi=None, mw=None, mg=None, note="") -> None:
        rows.append({"outcome_id": oid, "adjustment": adj, "group": group, "min_windows": int(thr), "method": method,
                     "n_athletes": int(n), "estimate": _f(est), "ci_lower": _f(lo), "ci_upper": _f(hi),
                     "median_windows": _f(mw), "median_games": _f(mg), "note": note})

    # ---- odd / even games (deterministic), athlete bootstrap
    for s in specs:
        pair = eng.pairs(s)
        for thr in c.min_windows_grid:
            for group, sub, centred in _groups(_eligible_pairs(pair, thr, c), c):
                codes = _codes(sub["position_group"].to_list()) if centred else None
                mw = float((sub["n_0"] + sub["n_1"]).median())
                mg = float((sub["n_games_0"] + sub["n_games_1"]).median())
                for adj, col in ADJ.items():
                    a, b = sub[f"{col}_0"].to_numpy().astype(float), sub[f"{col}_1"].to_numpy().astype(float)
                    r, lo, hi = _corr_ci(a, b, codes, c.n_boot, seed_for(c.seed, "REL", s.outcome_id, thr, group))
                    add(s.outcome_id, adj, group, thr, "game_odd_even", sub.height, r, lo, hi, mw, mg,
                        "Pearson, odd vs even games; athlete bootstrap")
                    add(s.outcome_id, adj, group, thr, "game_odd_even_sb", sub.height, sb2(r), sb2(lo), sb2(hi), mw, mg,
                        "odd / even stepped up to all games (Spearman–Brown)")
    # ---- random halves of each athlete's games (the same assignment for every outcome of a split)
    games = eng.games()
    rng = np.random.default_rng(seed_for(c.seed, "REL", "random_halves"))
    reps: dict[tuple, list[float]] = {}
    sizes: dict[tuple, int] = {}
    for _ in range(c.n_game_splits):
        assign = random_assign(games, rng)
        for s in specs:
            pair = eng.pairs(s, half="_h", assign=assign)
            for thr in c.min_windows_grid:
                for group, sub, centred in _groups(_eligible_pairs(pair, thr, c), c):
                    codes = _codes(sub["position_group"].to_list()) if centred else None
                    for adj, col in ADJ.items():
                        k = (s.outcome_id, adj, group, thr)
                        r = within_r(sub[f"{col}_0"].to_numpy().astype(float), sub[f"{col}_1"].to_numpy().astype(float), codes)
                        reps.setdefault(k, []).append(sb2(r))
                        sizes[k] = sub.height
    for (oid, adj, group, thr), v in reps.items():
        v = np.array(v, float)
        v = v[np.isfinite(v)]
        if v.size:
            add(oid, adj, group, thr, "game_random_halves_sb", sizes[(oid, adj, group, thr)], v.mean(), np.quantile(v, 0.025),
                np.quantile(v, 0.975), note=f"mean over {c.n_game_splits} random game half-splits; interval = spread over splits")
    # ---- variance of the play measure: athlete, game, window (base outcomes, primary threshold)
    thr = c.primary_min_windows
    for s in (x for x in specs if not x.components):
        d = eng.frame(s).filter(pl.len().over("nfl_id") >= thr)
        if d["nfl_id"].n_unique() < c.min_athletes:
            continue
        seed = seed_for(c.seed, "VAR", s.outcome_id)
        for adj, col in (("raw", "y"), ("context", "r")):
            comp = variance_components(d, col, c, seed)
            for method, key, note in (("play_icc1", "play", "single windows of the play measure, stratified by position"),
                                      ("game_icc1", "game", f"athlete-game means, games with ≥ {c.min_game_windows} windows")):
                ic = comp[key]
                add(s.outcome_id, adj, WITHIN, thr, method, ic["n_athletes"], ic["icc"], ic["ci_lower"], ic["ci_upper"], note=note)
            for method in ("var_athlete", "var_game", "var_window", "windows_per_game"):
                add(s.outcome_id, adj, WITHIN, thr, method, comp["play"]["n_athletes"], comp[method],
                    note="play measure, squared unit of the measure" if method != "windows_per_game" else "median windows per athlete-game")
    # ---- consecutive seasons
    for s in specs:
        v = eng.values(s, by=("nfl_id", "season")).join(eng.athletes, on="nfl_id", how="left")
        seasons = sorted(v["season"].drop_nulls().unique().to_list())
        need = max(c.min_half_windows, c.primary_min_windows // 2)
        for a_, b_ in zip(seasons, seasons[1:]):
            j = (v.filter((pl.col("season") == a_) & (pl.col("n") >= need))
                 .join(v.filter((pl.col("season") == b_) & (pl.col("n") >= need)), on="nfl_id", how="inner", suffix="_b"))
            if j.height < c.min_athletes:
                continue
            codes = _codes(j["position_group"].to_list())
            for adj, col in ADJ.items():
                r, lo, hi = _corr_ci(j[col].to_numpy().astype(float), j[f"{col}_b"].to_numpy().astype(float), codes, c.n_boot,
                                     seed_for(c.seed, "SEASON", s.outcome_id, a_))
                add(s.outcome_id, adj, WITHIN, need, f"season_{a_}_to_{b_}", j.height, r, lo, hi, float(j["n"].median()),
                    float(j["n_games"].median()), f"Pearson between seasons, ≥ {need} windows in each; not stepped up")
    return pl.DataFrame(rows, schema=RELIABILITY.columns) if rows else RELIABILITY.empty()


def variance_components(d: pl.DataFrame, col: str, c: OutcomeConfig, seed: int = 0) -> dict[str, Any]:
    """Moment estimators on one outcome frame (columns nfl_id, game_id, position_group, ``col``)."""
    y, ath = d[col].to_numpy().astype(float), d["nfl_id"].to_numpy()
    play = stratified_icc(y, ath, d["position_group"].to_numpy(), c.n_boot, seed)
    ag = ath.astype(np.int64) * 10_000_000_000 + d["game_id"].to_numpy().astype(np.int64)     # one code per athlete-game
    nested = stratified_icc(y, ag, ath, 0, seed)                       # athlete-games nested in athletes
    gm = (d.group_by("nfl_id", "game_id").agg(pl.col(col).mean().alias("_m"), pl.len().alias("_n"),
                                              pl.col("position_group").first())
          .filter(pl.col("_n") >= c.min_game_windows).filter(pl.len().over("nfl_id") >= 2))
    game = (stratified_icc(gm["_m"].to_numpy(), gm["nfl_id"].to_numpy(), gm["position_group"].to_numpy(), c.n_boot, seed)
            if gm.height else {"icc": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n_athletes": 0})
    per_game = d.group_by("nfl_id", "game_id").len()["len"].median()
    return {"play": play, "game": game, "var_athlete": play["var_between"], "var_game": nested["var_between"],
            "var_window": nested["var_within"], "windows_per_game": float(per_game) if per_game is not None else float("nan")}


def projected_reliability(var_a: float, var_g: float, var_w: float, n_windows: float, per_game: float) -> float:
    """Reliability of the athlete MEAN of the play measure from n windows spread over n / per_game games."""
    if not all(np.isfinite(x) for x in (var_a, var_g, var_w, n_windows, per_game)) or n_windows <= 0 or per_game <= 0:
        return float("nan")
    den = var_a + var_g / max(n_windows / per_game, 1.0) + var_w / n_windows
    return float(var_a / den) if den > 0 else float("nan")


# ============================================================================== exposure curves
def _cut(eng: OutcomeEngine, spec: OutcomeSpec, kind: str, budget: int) -> pl.DataFrame:
    d = eng.frame(spec)
    if kind == "windows":
        return d.filter(pl.col("w_order") == budget).select("nfl_id", pl.col("game_order").alias("_g"))
    return (d.group_by("nfl_id").agg(pl.col("game_order").max().alias("_mx")).filter(pl.col("_mx") > budget)
            .select("nfl_id", pl.lit(budget, dtype=pl.Int64).alias("_g")))


def budget_pairs(eng: OutcomeEngine, spec: OutcomeSpec, kind: str, budget: int) -> pl.DataFrame:
    """Per athlete: the outcome on the FIRST ``budget`` windows (or games) and on the games strictly after them.

    half 0 = windows with w_order ≤ budget (kind 'windows') or game_order ≤ budget (kind 'games');
    half 1 = windows of games AFTER the game that holds the last window of half 0. The rest of that game is in
    neither half, so the two halves never share a game. A composite uses the latest cut of its components."""
    base = [eng.by_id[k] for k in spec.components] or [spec]
    cut = None
    for s in base:
        cu = _cut(eng, s, kind, budget)
        cut = cu if cut is None else (cut.join(cu, on="nfl_id", how="inner", suffix="_r")
                                      .select("nfl_id", pl.max_horizontal("_g", "_g_r").alias("_g")))
    first = (pl.col("w_order") <= budget) if kind == "windows" else (pl.col("game_order") <= budget)
    out = None
    for i, s in enumerate(base):
        h = (eng.frame(s).join(cut, on="nfl_id", how="inner")
             .with_columns(pl.when(first & (pl.col("game_order") <= pl.col("_g"))).then(pl.lit(0))
                           .when(pl.col("game_order") > pl.col("_g")).then(pl.lit(1)).otherwise(None).alias("_h"))
             .filter(pl.col("_h").is_not_null())
             .group_by("nfl_id", "_h").agg(pl.len().cast(pl.Int64).alias("n"), no._stat(pl.col("y"), s.stat).alias("value")))
        a = h.filter(pl.col("_h") == 0).select("nfl_id", pl.col("n").alias(f"n0_{i}"), pl.col("value").alias(f"v0_{i}"))
        b = h.filter(pl.col("_h") == 1).select("nfl_id", pl.col("n").alias(f"n1_{i}"), pl.col("value").alias(f"v1_{i}"))
        p = a.join(b, on="nfl_id", how="inner")
        out = p if out is None else out.join(p, on="nfl_id", how="inner")
    k = range(len(base))
    return (out.select("nfl_id", pl.min_horizontal([pl.col(f"n0_{i}") for i in k]).alias("n_0"),
                       pl.mean_horizontal([pl.col(f"v0_{i}") for i in k]).alias("value_0"),
                       pl.min_horizontal([pl.col(f"n1_{i}") for i in k]).alias("n_1"),
                       pl.mean_horizontal([pl.col(f"v1_{i}") for i in k]).alias("value_1"))
            .join(eng.athletes, on="nfl_id", how="left").sort("nfl_id"))


def exposure_table(eng: OutcomeEngine, c: OutcomeConfig, reliability: pl.DataFrame,
                   specs: Sequence[OutcomeSpec] | None = None) -> pl.DataFrame:
    specs = list(specs) if specs is not None else eng.specs
    rows: list[dict[str, Any]] = []

    def add(oid, kind, budget, cohort, method, n, est, lo=None, hi=None, note="") -> None:
        rows.append({"outcome_id": oid, "budget_type": kind, "budget": int(budget), "cohort": cohort, "method": method,
                     "n_athletes": int(n), "estimate": _f(est), "ci_lower": _f(lo), "ci_upper": _f(hi), "note": note})

    def comp(oid: str, method: str) -> float:
        r = reliability.filter((pl.col("outcome_id") == oid) & (pl.col("adjustment") == "raw") & (pl.col("method") == method)
                               & (pl.col("min_windows") == c.primary_min_windows) & (pl.col("group") == WITHIN))
        return float(r["estimate"][0]) if r.height and r["estimate"][0] is not None else float("nan")

    for s in specs:
        for kind, grid in (("windows", c.first_n_grid), ("games", c.first_games_grid)):
            later = {}
            for n in grid:
                p = budget_pairs(eng, s, kind, n).filter(pl.col("n_1") >= c.holdout_min_windows)
                if p.height >= c.min_athletes:
                    later[n] = p
            common = set(later[max(later)]["nfl_id"].to_list()) if later else set()
            for n, p in later.items():
                for cohort, sub in (("available", p), ("common", p.filter(pl.col("nfl_id").is_in(list(common))))):
                    if sub.height < c.min_athletes:
                        continue
                    codes = _codes(sub["position_group"].to_list())
                    a, b = sub["value_0"].to_numpy().astype(float), sub["value_1"].to_numpy().astype(float)
                    r, lo, hi = _corr_ci(a, b, codes, c.n_boot, seed_for(c.seed, "EXP", s.outcome_id, kind, n, cohort))
                    add(s.outcome_id, kind, n, cohort, "first_vs_later_games", sub.height, r, lo, hi,
                        f"Pearson within position, first {n} {kind} vs games played after them (≥ {c.holdout_min_windows} windows); "
                        "attenuated by the noise of both sides")
                    add(s.outcome_id, kind, n, cohort, "first_vs_later_games_spearman", sub.height, within_r(a, b, codes, rank=True),
                        note="rank version of first_vs_later_games")
            for n in grid:
                where = (pl.col("w_order") <= n) if kind == "windows" else (pl.col("game_order") <= n)
                full = eng.values(s, where=where).filter(pl.col("n") >= n) if kind == "windows" else None
                p = eng.pairs(s, where=where).filter(pl.min_horizontal("n_0", "n_1") >= c.min_half_windows)
                if full is not None:
                    p = p.join(full.select("nfl_id"), on="nfl_id", how="semi")       # athletes who HAVE n windows
                if p.height < c.min_athletes:
                    continue
                codes = _codes(p["position_group"].to_list())
                r, lo, hi = _corr_ci(p["value_0"].to_numpy().astype(float), p["value_1"].to_numpy().astype(float), codes, c.n_boot,
                                     seed_for(c.seed, "EXPW", s.outcome_id, kind, n))
                add(s.outcome_id, kind, n, "available", "within_budget_game_split_sb", p.height, sb2(r), sb2(lo), sb2(hi),
                    f"odd / even games inside the first {n} {kind}, stepped up; athletes whose first {n} {kind} span ≥ 2 games")
        full = reliability.filter((pl.col("outcome_id") == s.outcome_id) & (pl.col("adjustment") == "raw") & (pl.col("group") == WITHIN)
                                  & (pl.col("min_windows") == c.primary_min_windows) & (pl.col("method") == "game_random_halves_sb"))
        if full.height:
            f0 = full.row(0, named=True)
            add(s.outcome_id, "windows", 0, "available", "game_random_halves_sb", f0["n_athletes"], f0["estimate"], f0["ci_lower"],
                f0["ci_upper"], f"all available windows, athletes with ≥ {c.primary_min_windows}; interval = spread over splits")
        if not s.components:
            va, vg, vw, m = (comp(s.outcome_id, k) for k in ("var_athlete", "var_game", "var_window", "windows_per_game"))
            for n in c.first_n_grid:
                add(s.outcome_id, "windows", n, "model", "model_projection", 0, projected_reliability(va, vg, vw, n, m),
                    note="variance components of the play measure; describes the athlete MEAN, indicative for quantile statistics")
    return pl.DataFrame(rows, schema=EXPOSURE.columns) if rows else EXPOSURE.empty()


# ============================================================================== context
def context_audit(eng: OutcomeEngine, outcomes: pl.DataFrame, c: OutcomeConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for s in eng.base_specs():
        d = eng.frame(s).with_columns(pl.col("down").cast(pl.String).fill_null("NA").alias("down"),
                                      pl.col("lined_up_position").fill_null("NA"), pl.col("cell").alias("context_cell"))
        for p in sorted(d["position_group"].unique().to_list()):
            g = d.filter(pl.col("position_group") == p)
            tot = float(g["y"].var()) if g.height > 2 else float("nan")
            for var in (*no.CONTEXT_VARS, "context_cell"):
                lv = (g.group_by(var).agg(pl.len().cast(pl.Int64).alias("n"), pl.col("nfl_id").n_unique().cast(pl.Int64).alias("a"),
                                          pl.col("y").mean().alias("m"), pl.col("y").std().alias("s")).sort("n", descending=True))
                ssb = float(((lv["m"] - g["y"].mean()) ** 2 * lv["n"]).sum())
                eta = ssb / (tot * (g.height - 1)) if np.isfinite(tot) and tot > 0 else float("nan")
                rows.append({"outcome_id": s.outcome_id, "position_group": p, "variable": var, "level": "__ALL__", "n_windows": g.height,
                             "n_athletes": g["nfl_id"].n_unique(), "mean": _f(g["y"].mean()), "sd": _f(g["y"].std()), "eta2": _f(eta),
                             "note": f"{lv.height} levels"})
                if var == "context_cell":
                    continue
                for r in lv.iter_rows(named=True):
                    rows.append({"outcome_id": s.outcome_id, "position_group": p, "variable": var, "level": str(r[var]),
                                 "n_windows": r["n"], "n_athletes": r["a"], "mean": _f(r["m"]), "sd": _f(r["s"]), "eta2": None,
                                 "note": "" if r["n"] >= c.min_cell_windows else "small level"})
            o = outcomes.filter((pl.col("outcome_id") == s.outcome_id) & (pl.col("position_group") == p)
                                & (pl.col("n_windows") >= c.primary_min_windows))
            if o.height >= c.min_athletes_position:
                raw, adj = o["value"].to_numpy().astype(float), o["value_adj"].to_numpy().astype(float)
                sd = float(np.std(raw, ddof=1))
                rows.append({"outcome_id": s.outcome_id, "position_group": p, "variable": "athlete_adjustment", "level": "__ALL__",
                             "n_windows": int(o["n_windows"].sum()), "n_athletes": o.height, "mean": _f(pearson(raw, adj)),
                             "sd": _f(np.std(adj - raw, ddof=1) / sd if sd > 0 else float("nan")), "eta2": None,
                             "note": "mean = Pearson(raw, adjusted); sd = SD(adjusted − raw) / SD(raw)"})
    return pl.DataFrame(rows, schema=CONTEXT.columns) if rows else CONTEXT.empty()


# ============================================================================== sensitivity
def quality_filters() -> dict[str, pl.Expr]:
    return {"tier_A_only": pl.col("quality_tier") == "A", "in_player_play_only": pl.col("in_player_play"),
            "post_snap_ge_3s": pl.col("n_post_frames") >= 30, "boundary_frac_le_0p2": pl.col("boundary_frac") <= 0.2}


def _compare(vp: pl.DataFrame, vv: pl.DataFrame, pp: pl.DataFrame, pv: pl.DataFrame, athletes: pl.DataFrame, thr: int,
             c: OutcomeConfig) -> dict[str, Any] | None:
    j = (vp.filter(pl.col("n") >= thr).join(vv.filter(pl.col("n") >= max(c.min_half_windows, thr // 2)), on="nfl_id", how="inner",
                                            suffix="_v").join(athletes, on="nfl_id", how="left"))
    if j.height < c.min_athletes:
        return None
    codes = _codes(j["position_group"].to_list())
    a, b = j["value"].to_numpy().astype(float), j["value_v"].to_numpy().astype(float)
    sd = float(np.nanstd(a, ddof=1))

    def rel(pair: pl.DataFrame) -> float:
        pair = pair.filter(pl.min_horizontal("n_0", "n_1") >= c.min_half_windows).join(j.select("nfl_id"), on="nfl_id", how="semi")
        if pair.height < c.min_athletes:
            return float("nan")
        return sb2(within_r(pair["value_0"].to_numpy().astype(float), pair["value_1"].to_numpy().astype(float),
                            _codes(pair["position_group"].to_list())))

    return {"n_athletes": j.height, "pearson": _f(within_r(a, b, codes)), "spearman": _f(within_r(a, b, codes, rank=True)),
            "mean_shift": _f(np.nanmean(b - a)), "shift_sd_units": _f(np.nanmean(b - a) / sd if sd > 0 else float("nan")),
            "sd_ratio": _f(np.nanstd(b, ddof=1) / sd if sd > 0 else float("nan")), "reliability_primary": _f(rel(pp)),
            "reliability_variant": _f(rel(pv)), "windows_retained": _f(j["n_v"].sum() / max(int(j["n"].sum()), 1))}


def sensitivity_table(eng: OutcomeEngine, plays: pl.DataFrame, variants: pl.DataFrame, c: OutcomeConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    names = sorted(variants["variant"].unique().to_list()) if variants.height else []
    stride = max(1, c.sensitivity_game_stride)
    for v in names:
        ep, ev = no.variant_engine(plays, variants, v, c)
        kind = "boundary" if v.startswith("boundary") else "smoothing"
        thr = max(2 * c.min_half_windows, c.primary_min_windows // stride)
        for s in eng.specs:
            sp, sv = ep.by_id[s.outcome_id], ev.by_id[s.outcome_id]
            m = _compare(ep.values(sp), ev.values(sv), ep.pairs(sp), ev.pairs(sv), ep.athletes, thr, c)
            if m:
                rows.append({"outcome_id": s.outcome_id, "variant": v, "kind": kind, **m,
                             "note": f"same windows in both (every {stride}th game of a batch); ≥ {thr} windows"})
    for name, where in quality_filters().items():
        for s in eng.specs:
            m = _compare(eng.values(s), eng.values(s, where=where), eng.pairs(s), eng.pairs(s, where=where), eng.athletes,
                         c.primary_min_windows, c)
            if m:
                rows.append({"outcome_id": s.outcome_id, "variant": name, "kind": "quality_filter", **m,
                             "note": "primary = all eligible windows; variant = windows passing the filter"})
    return pl.DataFrame(rows, schema=SENSITIVITY.columns) if rows else SENSITIVITY.empty()


def redundancy_table(outcomes: pl.DataFrame, c: OutcomeConfig) -> pl.DataFrame:
    o = outcomes.filter(pl.col("n_windows") >= c.primary_min_windows)
    ids = sorted(o["outcome_id"].unique().to_list())
    rows = []
    for i, a in enumerate(ids):
        da = o.filter(pl.col("outcome_id") == a).select("nfl_id", "position_group", pl.col("value").alias("a"))
        for b in ids[i + 1:]:
            j = da.join(o.filter(pl.col("outcome_id") == b).select("nfl_id", pl.col("value").alias("b")), on="nfl_id", how="inner")
            if j.height >= c.min_athletes:
                rows.append({"outcome_a": a, "outcome_b": b, "n_athletes": j.height,
                             "spearman": _f(within_r(j["a"].to_numpy().astype(float), j["b"].to_numpy().astype(float),
                                                     _codes(j["position_group"].to_list()), rank=True))})
    return pl.DataFrame(rows, schema=REDUNDANCY.columns) if rows else REDUNDANCY.empty()


# ============================================================================== screening and freeze
def _one(df: pl.DataFrame, col: str = "estimate") -> float | None:
    return df[col][0] if df.height else None


def screen_table(specs: Sequence[OutcomeSpec], coverage: pl.DataFrame, reliability: pl.DataFrame, exposure: pl.DataFrame,
                 sensitivity: pl.DataFrame, context: pl.DataFrame, c: OutcomeConfig) -> pl.DataFrame:
    """One row per outcome: the evidence and the reference lines it meets. Labels are heuristic and for review —
    the reliability line is one criterion among five, and the final choice is recorded by the analyst."""
    thr, ge = c.primary_min_windows, f"n_athletes_ge_{c.primary_min_windows}"
    rows = []
    for s in specs:
        cov = coverage.filter(pl.col("outcome_id") == s.outcome_id)
        n_ath = int(_one(cov.filter(pl.col("position_group") == ALL), ge) or 0)
        n_pos = cov.filter((pl.col("position_group") != ALL) & (pl.col(ge) >= c.min_athletes_position)).height
        rel = reliability.filter((pl.col("outcome_id") == s.outcome_id) & (pl.col("min_windows") == thr))
        pick = lambda adj, grp, m: rel.filter((pl.col("adjustment") == adj) & (pl.col("group") == grp) & (pl.col("method") == m))   # noqa: E731
        r = _one(pick("raw", WITHIN, "game_random_halves_sb"))
        oe = pick("raw", WITHIN, "game_odd_even_sb")
        r_adj = _one(pick("context", WITHIN, "game_random_halves_sb"))
        by_pos = rel.filter((pl.col("adjustment") == "raw") & (pl.col("group") != WITHIN) & (pl.col("method") == "game_random_halves_sb"))
        first = exposure.filter((pl.col("outcome_id") == s.outcome_id) & (pl.col("budget_type") == "windows") & (pl.col("budget") == thr)
                                & (pl.col("cohort") == "available") & (pl.col("method") == "first_vs_later_games"))
        sm = sensitivity.filter((pl.col("outcome_id") == s.outcome_id) & pl.col("kind").is_in(["smoothing", "boundary"]))
        rho = _one(cov.filter(pl.col("position_group") == ALL), "rho_value_log_windows")
        ctx = context.filter((pl.col("outcome_id").is_in([s.outcome_id, *s.components])) & (pl.col("variable") == "context_cell")
                             & (pl.col("level") == "__ALL__"))
        eta = ctx["eta2"].max() if ctx.height else None
        sm_min = sm["spearman"].min() if sm.height else None
        lower = _one(oe, "ci_lower")
        p_cov = n_ath >= c.min_athletes and n_pos >= c.ref_positions
        p_rel = r is not None and r >= c.ref_reliability and lower is not None and lower >= c.ref_reliability_lower
        p_sm = sm_min is not None and sm_min >= c.ref_smoothing_rank
        p_exp = rho is not None and abs(rho) <= c.ref_exposure_rho
        p_ctx = eta is not None and (eta <= c.ref_context_share or (r_adj is not None and r_adj >= c.ref_reliability))
        why = [t for ok, t in ((p_cov, f"coverage: {n_ath} athletes, {n_pos} position groups (reference ≥ {c.min_athletes}, ≥ {c.ref_positions})"),
                               (p_rel, f"reliability {r} / lower bound {lower} (reference ≥ {c.ref_reliability:g} / ≥ {c.ref_reliability_lower:g})"),
                               (p_sm, f"smoothing rank agreement {sm_min} (reference ≥ {c.ref_smoothing_rank:g})"),
                               (p_exp, f"|Spearman with log windows| {rho} (reference ≤ {c.ref_exposure_rho:g})"),
                               (p_ctx, f"context share {eta}, adjusted reliability {r_adj} (reference ≤ {c.ref_context_share:g} or ≥ {c.ref_reliability:g})"))
               if not ok]
        n_ok = sum(map(bool, (p_cov, p_rel, p_sm, p_exp, p_ctx)))
        status = ({"reference": "reference_only", "exploratory": "exploratory", "component": "component"}.get(s.role)
                  or ("eligible" if n_ok == 5 else "provisional" if p_cov and p_rel else "not_eligible"))
        rows.append({"outcome_id": s.outcome_id, "family": s.family, "role": s.role, "n_athletes": n_ath, "n_positions": n_pos,
                     "reliability": _f(r), "reliability_lower": _f(lower), "reliability_upper": _f(_one(oe, "ci_upper")),
                     "reliability_adj": _f(r_adj), "reliability_min_position": _f(by_pos["estimate"].min()) if by_pos.height else None,
                     "first_n_r": _f(_one(first)), "smoothing_min_spearman": _f(sm_min),
                     "max_shift_sd": _f(sm["shift_sd_units"].abs().max()) if sm.height else None, "exposure_rho": _f(rho),
                     "context_eta2": _f(eta), "pass_coverage": bool(p_cov), "pass_reliability": bool(p_rel), "pass_smoothing": bool(p_sm),
                     "pass_exposure": bool(p_exp), "pass_context": bool(p_ctx), "n_passed": n_ok, "status": status,
                     "reasons": "; ".join(why)})
    return pl.DataFrame(rows, schema=SCREEN.columns) if rows else SCREEN.empty()


def propose_targets(screen: pl.DataFrame, redundancy: pl.DataFrame, c: OutcomeConfig) -> dict[str, Any]:
    """Rule-based PROPOSAL: primary = eligible candidate with the highest lower reliability bound; up to two
    secondaries from other families whose athlete ranking is not a near copy of a chosen target."""
    el = screen.filter(pl.col("status") == "eligible").sort(["reliability_lower", "reliability", "outcome_id"],
                                                            descending=[True, True, False], nulls_last=True)
    if not el.height:
        prov = screen.filter(pl.col("status") == "provisional")["outcome_id"].to_list()
        return {"decision": "STOP", "primary": None, "secondary": [],
                "reason": "no candidate outcome meets all five reference lines"
                          + (f"; provisional (coverage and reliability only): {prov}" if prov else "")}

    def rho(a: str, b: str) -> float:
        r = redundancy.filter(((pl.col("outcome_a") == a) & (pl.col("outcome_b") == b)) | ((pl.col("outcome_a") == b) & (pl.col("outcome_b") == a)))
        return abs(float(r["spearman"][0])) if r.height and r["spearman"][0] is not None else 0.0

    chosen = [el.row(0, named=True)]
    for r in el.iter_rows(named=True):
        if len(chosen) == 3:
            break
        if r["family"] in {x["family"] for x in chosen} or any(rho(r["outcome_id"], x["outcome_id"]) >= c.ref_redundancy for x in chosen):
            continue
        chosen.append(r)
    return {"decision": "FREEZE", "primary": chosen[0]["outcome_id"], "secondary": [x["outcome_id"] for x in chosen[1:]],
            "reason": "highest lower reliability bound among eligible candidates; secondaries from other families with "
                      f"|Spearman| < {c.ref_redundancy:g} to every chosen target"}


def target_definition(oid: str, rank: str, dictionary: pl.DataFrame, screen: pl.DataFrame, coverage: pl.DataFrame,
                      c: OutcomeConfig) -> dict[str, Any]:
    d = dictionary.filter(pl.col("outcome_id") == oid).row(0, named=True)
    s = screen.filter(pl.col("outcome_id") == oid).row(0, named=True)
    eta = s["context_eta2"]
    definition = {"outcome_id": oid, "formula": d["formula"], "units": d["unit"], "play_measure": d["play_measure"],
                  "athlete_statistic": d["athlete_statistic"], "window": d["window_definition"],
                  "starting_speed_range": d["starting_speed_range"], "context_definition": d["applicable_context"],
                  "context_adjustment": "cell_residual" if eta is not None and eta > c.ref_context_share else "none",
                  "exposure_requirement": {"min_windows": c.primary_min_windows, "min_half_windows": c.min_half_windows},
                  "inclusion_exclusion": d["quality_requirements"], "smoothing": d["smoothing"],
                  "grouping_grain": "athlete, over eligible player-play windows of regular-season games; compared within position group",
                  "direction_note": d["direction_note"], "interpretation_limit": d["interpretation_limit"]}
    cov = coverage.filter(pl.col("outcome_id") == oid)
    return {"rank": rank, **definition, "definition_hash": stable_hash(definition),
            "reliability": {"method": "random game half-splits, Spearman–Brown, within position", "estimate": s["reliability"],
                            "odd_even_ci_lower": s["reliability_lower"], "odd_even_ci_upper": s["reliability_upper"],
                            "context_adjusted": s["reliability_adj"], "min_position": s["reliability_min_position"],
                            "first_windows_vs_later_games_r": s["first_n_r"]},
            "eligibility_counts": cov.select("position_group", "n_athletes_tracked", f"n_athletes_ge_{c.primary_min_windows}",
                                             "n_windows", "n_athlete_games").to_dicts(),
            "screen": {k: s[k] for k in ("status", "n_passed", "smoothing_min_spearman", "exposure_rho", "context_eta2", "reasons")}}


def freeze_targets(store: Any, decision: str | None, primary: str | None, secondary: Sequence[str], rationale: str,
                   proposal: dict[str, Any], screen: pl.DataFrame, dictionary: pl.DataFrame, coverage: pl.DataFrame,
                   c: OutcomeConfig, provenance: dict[str, Any], refreeze: bool = False) -> tuple[dict[str, Any] | None, str]:
    """Write ``nfl_movement_targets_v1.json`` ONCE per run mode. ``decision`` None = pending review (nothing written).

    A target must be a candidate outcome labelled eligible, or provisional with a written rationale; reference,
    component, exploratory and not-eligible outcomes are refused. No Combine table is an argument of this function."""
    for t in (screen, dictionary, coverage):
        no.assert_no_combine_columns(t, "freeze input")
    path = store.data_dir / f"{TARGETS_FILE}.json"
    if decision is None:
        return (read_json(path, None) if path.is_file() else None), "PENDING REVIEW — set DECISION to 'FREEZE' or 'STOP' and rerun this cell"
    if decision not in DECISIONS:
        raise ValueError(f"DECISION must be one of {DECISIONS} or None, got {decision!r}")
    secondary = list(secondary or [])
    targets: list[dict[str, Any]] = []
    if decision == "FREEZE":
        if primary is None or len(secondary) > 2 or len({primary, *secondary}) != 1 + len(secondary):
            raise ValueError("a FREEZE needs one primary and at most two distinct secondary outcomes")
        status = dict(screen.select("outcome_id", "status").iter_rows())
        for oid in (primary, *secondary):
            st = status.get(oid)
            if st not in ("eligible", "provisional") or (st == "provisional" and not rationale.strip()):
                raise ValueError(f"{oid!r} cannot be frozen (status {st!r}): targets are eligible candidates, or provisional "
                                 "ones with a written RATIONALE")
        targets = [target_definition(primary, "primary", dictionary, screen, coverage, c),
                   *[target_definition(o, "secondary", dictionary, screen, coverage, c) for o in secondary]]
    doc = {"artifact": "nfl_movement_targets", "version": "v1", "decision": decision, "frozen_at": utc_now(),
           "outcomes_version": c.version, "config": c.params(), "config_hash": stable_hash(c.params()), "targets": targets,
           "selection": {"rule_based_proposal": proposal, "follows_proposal": decision == proposal.get("decision")
                         and primary == proposal.get("primary") and secondary == list(proposal.get("secondary", [])),
                         "rationale": rationale, "combine_data_consulted": False,
                         "criteria": "physical interpretability, reliability across games, coverage, exposure dependence, "
                                     "context share, smoothing sensitivity; reference lines are heuristics",
                         "screen": screen.to_dicts()},
           "provenance": provenance,
           "version_hash": stable_hash({"decision": decision, "targets": [t["definition_hash"] for t in targets],
                                        "config": c.params()})}
    if not store.controls.save_tables:
        return doc, "NOT WRITTEN (SAVE_TABLES is off)"
    if path.is_file() and not refreeze:
        old = read_json(path, {})
        same = old.get("version_hash") == doc["version_hash"]
        return old, (f"already frozen on {old.get('frozen_at')} (decision {old.get('decision')})"
                     + ("; the current run reproduces it" if same else "; THE CURRENT RUN WOULD PRODUCE A DIFFERENT FREEZE — kept "
                        "as frozen. Set REFREEZE_TARGETS = True only if no transfer analysis has used it, and record why"))
    store.data_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json({**doc, "run_mode": store.controls.run_mode, "refrozen": bool(refreeze)}, path)
    log.info("[nfl_outcomes] %s written: decision %s, targets %s", path.name, decision, [t["outcome_id"] for t in targets])
    return doc, "frozen now" + (" (REFREEZE)" if refreeze else "")


def gate_05(doc: dict[str, Any] | None) -> tuple[str, str]:
    """(status, sentence) of research gate 05 from the frozen artifact."""
    if not doc:
        return "PENDING", "no freeze decision is recorded yet"
    if doc.get("decision") == "STOP":
        return "STOP", "no NFL movement outcome was accepted: revise the outcome definitions before any transfer model"
    ids = [t["outcome_id"] for t in doc.get("targets", [])]
    return "PASS", f"frozen without consulting Combine associations: {', '.join(ids)}"


def conclusions_markdown(screen: pl.DataFrame, proposal: dict[str, Any], doc: dict[str, Any] | None, c: OutcomeConfig) -> str:
    """Generated ONLY from the tables of this run; every number is read from them."""
    lines = ["### What the tables show", ""]
    for st in ("eligible", "provisional", "not_eligible", "component", "reference_only", "exploratory"):
        sub = screen.filter(pl.col("status") == st).sort("reliability", descending=True, nulls_last=True)
        if sub.height:
            lines.append(f"- **{st}** ({sub.height}): " + "; ".join(
                f"`{r['outcome_id']}` reliability {r['reliability'] if r['reliability'] is None else round(r['reliability'], 3)} "
                f"(n = {r['n_athletes']})" for r in sub.iter_rows(named=True)))
    lines += ["", f"- Rule-based proposal: **{proposal['decision']}** — primary `{proposal['primary']}`, secondary "
                  f"{proposal['secondary']}. {proposal['reason']}.",
              f"- Recorded decision: **{gate_05(doc)[0]}** — {gate_05(doc)[1]}.", "",
              f"Reference lines (heuristics): reliability ≥ {c.ref_reliability:g} with lower bound ≥ {c.ref_reliability_lower:g}, "
              f"≥ {c.min_athletes} athletes at ≥ {c.primary_min_windows} windows in ≥ {c.ref_positions} position groups, smoothing rank "
              f"agreement ≥ {c.ref_smoothing_rank:g}, |Spearman with log windows| ≤ {c.ref_exposure_rho:g}, context share ≤ "
              f"{c.ref_context_share:g} (or adjusted reliability above the line).", "",
              "Reliability says an outcome is repeatable for an athlete across games. It does not say the outcome is an "
              "athletic capability: role, scheme and usage are stable too, and they are part of what repeats."]
    return "\n".join(lines)
