"""Milestone 08B.1 — reliability and context of the approved football outcomes (NumPy / Polars; NFL side only).

    outcome_rows          one row per player-play of a definition, with the recorded context labels (target, route, alignment)
    reliability_suite     odd / even games, random game halves, first-N versus later games (the 08A.2 estimators, by position),
                          reliability against the observation threshold, ranking stability between game halves
    adjust / crossfit_adjust      context conditioning with parameters from OTHER athletes (or from the training athletes of a fold)
    context_table         raw versus context-conditioned athlete values: reliability, rank agreement, variance removed
    group_table           value and support by NFL season / draft cohort / games observed
    attribution_table · sack_exposure_table      diagnostics of the secondary outcomes
    gate_b                Gate B — outcome measurement, per approved target

Rules. Halves are WHOLE GAMES: plays of one game never sit in both halves. A context parameter used for an athlete is
never estimated from that athlete's rows. Conditioning on route, alignment or target status CHANGES THE ESTIMAND and is
reported as a sensitivity; a missing label is never inferred (the row falls back to the coarser mean). Reliability
measures stable athlete-level differences in the RECORDED outcome — it is not evidence of football skill.
No Combine value is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from . import opportunity_validation as ov
from . import outcome_semantics as osm
from . import scouting_feasibility as sf
from .nfl_outcomes import assert_no_combine_columns

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
ALL = ov.ALL
ROW_SCHEMA = {**ov.ROW_SCHEMA, "route": S, "alignment": S, "tgt_label": S}
# (context id, label, cell keys, what conditioning on it does to the estimand)
CONTEXTS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("R0", "raw (no conditioning)", (), "the approved estimand"),
    ("R1", "route", ("route",), "removes route mix — partly a property of the player and of his NFL usage"),
    ("R2", "route × alignment", ("route", "alignment"), "also removes the NFL alignment the team chose"),
    ("R3", "target status", ("tgt_label",), "conditions on the quarterback's choice, which depends on the outcome: SENSITIVITY ONLY"),
)


@dataclass(frozen=True)
class ReliabilityConfig:
    """Extra knobs of 08B.1 (the support rules and reference lines are those of ``ov.ValidationConfig``)."""

    obs_grid: tuple[int, ...] = (4, 8, 12, 16, 24, 32)     # observations required in EACH game half
    crossfit_folds: int = 5                                 # athlete folds of the context cross-fit
    top_share: float = 0.25                                 # "top group" of the ranking-stability table
    min_group_athletes: int = 10

    def for_mode(self, smoke: bool) -> "ReliabilityConfig":
        return ReliabilityConfig(obs_grid=(2, 4), crossfit_folds=3, min_group_athletes=3) if smoke else self


# ============================================================================== contracts
_D = {"definition_id": S}
ROWS = TableContract("football_outcome_rows", "Player-plays of every analysed definition with their recorded context labels (input of 08B.2 / 08B.3).",
                     {**_D, "row": I, **ROW_SCHEMA}, key=("definition_id", "row"), kind="dataset", area="definition × player-play",
                     docs={"value": "RAW recorded value (a conditioned definition is always residualised later, on training athletes)",
                           "tgt": "1 targeted · 0 not targeted · null unknown", "route": "recorded route label; null = no label (never inferred)"})


def rows_frame(rows: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The row sets of several definitions in one persisted frame."""
    parts = [r.with_row_index("row").with_columns(pl.lit(k).alias("definition_id"), pl.col("row").cast(pl.Int64)).select(list(ROWS.columns))
             for k, r in rows.items() if r is not None]
    return pl.concat(parts).cast(ROWS.columns) if parts else ROWS.empty()


def rows_of(frame: pl.DataFrame, definition_id: str) -> pl.DataFrame:
    return frame.filter(pl.col("definition_id") == definition_id).select(list(ROW_SCHEMA))


REL = TableContract("football_reliability_game_splits", "Game-split reliability of each approved definition, pooled and by position.",
                    ov.REL_SCHEMA, key=("denominator_id", "split", "position_group"), area="definition × split × position group",
                    docs={"split": "odd_even = chronological parity of the athlete's games · random = random game halves (whole games)"})
FIRST_N = TableContract("football_reliability_exposure", "First-N games versus later games, and reliability by minimum games.",
                        ov.EXPO_SCHEMA, key=("denominator_id", "mode", "level"), area="definition × mode × level")
THRESH = TableContract("football_reliability_vs_opportunities", "Odd / even reliability against the observations required in each game half.",
                       {**_D, "position_group": S, "min_half_obs": I, "n_athletes": I, "r_half": F, "rel_sb": F, "rel_lo": F, "rel_hi": F},
                       key=("definition_id", "position_group", "min_half_obs"), area="definition × position group × threshold")
RANKS = TableContract("football_ranking_stability", "Stability of the within-position ranking between odd and even games.",
                      {**_D, "position_group": S, "n_athletes": I, "spearman": F, "mean_abs_pct_shift": F, "p90_abs_pct_shift": F,
                       "top_group_retention": F, "top_share": F}, key=("definition_id", "position_group"), area="definition × position group",
                      docs={"top_group_retention": "share of the top group of the odd games that is also in the top group of the even games"})
ATHLETES = TableContract("football_athlete_outcomes", "Athlete value of each approved definition with its game-bootstrap interval.",
                         ov.ATHLETE_SCHEMA, key=("denominator_id", "nfl_id"), kind="dataset", area="definition × athlete")
CONTEXT = TableContract("football_context_conditioning", "Raw versus context-conditioned athlete values (cross-fitted between athletes).",
                        {**_D, "context_id": S, "context": S, "estimand_change": S, "n_rows": I, "share_rows_supported_cell": F,
                         "n_athletes": I, "row_variance_removed": F, "rel_sb": F, "rel_lo": F, "rel_hi": F, "spearman_vs_raw": F,
                         "mean_abs_pct_shift_vs_raw": F}, key=("definition_id", "context_id"), area="definition × context",
                        docs={"row_variance_removed": "1 − var(conditioned rows) / var(raw rows), parameters from other athletes",
                              "rel_sb": "odd / even Spearman–Brown of the conditioned athlete value (athlete bootstrap)"})
CONTEXT_VALUES = TableContract("football_context_athlete_values", "Athlete value under each context definition (descriptive).",
                               {**_D, "context_id": S, "nfl_id": I, "position_group": S, "n_obs": I, "value": F, "pct_position": F},
                               key=("definition_id", "context_id", "nfl_id"), kind="dataset", area="definition × context × athlete")
CELLS = TableContract("football_context_cells", "Mean recorded value by context cell (all rows; descriptive).",
                      {**_D, "context_id": S, "cell": S, "n_rows": I, "n_athletes": I, "mean_value": F, "sd_value": F, "supported": B},
                      key=("definition_id", "context_id", "cell"), area="definition × context × cell")
GROUPS = TableContract("football_outcome_groups", "Athlete values and support by NFL season, draft cohort and games observed.",
                       {**_D, "group_kind": S, "group": S, "n_athletes": I, "n_rows": I, "mean_value": F, "sd_athlete": F, "median_games": F},
                       key=("definition_id", "group_kind", "group"), area="definition × group")
TARGET_SPLIT = TableContract("football_targeted_versus_not", "Targeted versus non-targeted separation of the same athletes.",
                             {"position_group": S, "n_athletes": I, "mean_targeted": F, "mean_not_targeted": F, "mean_difference": F,
                              "r_between": F, "rel_targeted": F, "rel_not_targeted": F, "share_rows_targeted": F},
                             key=("position_group",), area="position group")
ATTRIBUTION = TableContract("football_tackle_attribution", "Tackle and assist credits per eligible defensive snap, by position.",
                            {"position_group": S, "n_snaps": I, "n_statistic_rows": I, "n_solo": I, "n_assist": I, "n_both": I,
                             "credits_per_snap": F, "assist_share_of_credits": F, "share_snaps_with_statistic": F},
                            key=("position_group",), area="position group")
SACK = TableContract("football_sack_exposure", "Sack definitions against their own exposure.",
                     {**_D, "n_athletes": I, "mean_rate": F, "median_rows": F, "rho_value_log_obs": F, "rho_value_games": F,
                      "share_athletes_zero": F}, key=("definition_id",), area="definition")
GATE_B = TableContract("football_gate_b_measurement", "Gate B — outcome measurement, per approved target.",
                       {"target_id": S, "role": S, "criterion": S, "value": S, "passed": B, "blocking": B}, key=("target_id", "criterion"),
                       area="target × criterion")


# ============================================================================== rows
def outcome_rows(base: pl.DataFrame, definition_id: str, cfg: ov.ValidationConfig) -> pl.DataFrame | None:
    """Player-plays of a definition with their context labels. FO01 definitions carry target / route / alignment;
    the conditioned definition FO01.S2 returns its RAW rows (the residual is always cross-fitted, never taken here).
    Other definitions are the rows of ``ov.denominator_rows`` with empty context labels."""
    d = ov.BY_ID[definition_id]
    if not ov.available(base, d):
        return None
    none_s = pl.lit(None, dtype=pl.String)
    if d.outcome_id != osm.FO01:
        r = ov.denominator_rows(base, d, cfg)
        return r.with_columns(none_s.alias("route"), none_s.alias("alignment"), none_s.alias("tgt_label")).cast(ROW_SCHEMA)
    has_route, has_tgt = "route_ran" in base.columns, "target" in base.columns
    tgt = osm.flag(base, "target") if has_tgt else pl.lit(None, dtype=pl.Float64)
    route = (pl.when(osm.observed(base, "route_ran")).then(pl.col("route_ran").cast(pl.String).str.strip_chars()).otherwise(none_s)
             if has_route else none_s)
    el = (base.filter(pl.col("eligible") & pl.col("position_group").is_in(list(d.positions)))
          .with_columns(osm.numeric(d.needs[0]).alias("value"), tgt.alias("tgt"), route.alias("route"),
                        pl.col("lined_up_position").cast(pl.String).alias("alignment"))
          .filter(pl.col("value").is_not_null()))
    if definition_id in ("FO01.S1", "FO01.S2"):
        el = el.filter(pl.col("route").is_not_null())
    if definition_id == "FO01.S0T":
        el = el.filter(pl.col("tgt") == 1.0)
    if definition_id == "FO01.S0N":
        el = el.filter(pl.col("tgt") == 0.0)
    el = el.with_columns(pl.when(pl.col("tgt") == 1.0).then(pl.lit("targeted")).when(pl.col("tgt") == 0.0).then(pl.lit("not_targeted"))
                         .otherwise(none_s).alias("tgt_label"))
    out = el.select([pl.col(c).cast(t) for c, t in ROW_SCHEMA.items()])
    assert_no_combine_columns(out, f"rows of {definition_id}")
    return out


# ============================================================================== split statistics
def split_stat(ga: ov.GameArrays, half: np.ndarray, min_obs: int, n_boot: int, rng: np.random.Generator, scope: str = ALL,
               keep: np.ndarray | None = None) -> dict[str, Any]:
    """Position-centred half-versus-half correlation, Spearman–Brown step-up and its athlete-bootstrap interval.
    ``keep``: athlete mask applied on top of the support rule (never chosen from an outcome value)."""
    k, ma, mb = ov.half_stats(ga, half, min_obs, min_obs)
    if scope != ALL:
        k = k & (ga.pos == scope)
    if keep is not None:
        k = k & keep
    idx = np.flatnonzero(k)
    out = {"n": int(idx.size), "r": None, "rel_sb": None, "lo": None, "hi": None, "idx": idx, "a": ma[idx], "b": mb[idx]}
    if idx.size < 3:
        return out
    ra, rb = ov._centred(ma[idx], ga.pos[idx]), ov._centred(mb[idx], ga.pos[idx])
    r = ov._r(ra, rb)
    if not np.isfinite(r):
        return out
    out.update(r=float(r), rel_sb=ov._fn(sf._sb(r)))
    if n_boot > 0:
        bi = rng.integers(0, idx.size, (n_boot, idx.size))
        boot = sf._sb(sf._corr(ra[bi], rb[bi]))
        out.update(lo=ov._fn(np.nanpercentile(boot, 2.5)), hi=ov._fn(np.nanpercentile(boot, 97.5)))
    return out


def _pct(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Mid-rank percentile within position."""
    out = np.full(x.size, np.nan)
    for p in np.unique(pos):
        m = pos == p
        out[m] = 100.0 * (ov._rank(x[m]) - 0.5) / m.sum()
    return out


def threshold_rows(d: ov.Denominator, ga: ov.GameArrays, cfg: ov.ValidationConfig, rc: ReliabilityConfig, salt: int) -> list[dict[str, Any]]:
    rows = []
    for j, scope in enumerate((ALL, *d.positions) if len(d.positions) > 1 else (ALL,)):
        for m in rc.obs_grid:
            s = split_stat(ga, ga.rk % 2, int(m), cfg.n_boot, np.random.default_rng([cfg.seed, salt, j, int(m), 11]), scope)
            ok = s["n"] >= (3 if scope == ALL else cfg.min_scope_athletes)
            rows.append({"definition_id": d.denominator_id, "position_group": scope, "min_half_obs": int(m), "n_athletes": s["n"],
                         "r_half": s["r"] if ok else None, "rel_sb": s["rel_sb"] if ok else None, "rel_lo": s["lo"] if ok else None,
                         "rel_hi": s["hi"] if ok else None})
    return rows


def rank_rows(d: ov.Denominator, ga: ov.GameArrays, cfg: ov.ValidationConfig, rc: ReliabilityConfig) -> list[dict[str, Any]]:
    """Would the ranking of the athletes be the same with the other half of their games?"""
    s = split_stat(ga, ga.rk % 2, cfg.min_half_obs, 0, np.random.default_rng(0))
    idx, rows = s["idx"], []
    if idx.size < 3:
        return rows
    pos = ga.pos[idx]
    pa, pb = _pct(s["a"], pos), _pct(s["b"], pos)
    for scope in (ALL, *d.positions) if len(d.positions) > 1 else (ALL,):
        m = np.ones(idx.size, bool) if scope == ALL else pos == scope
        if m.sum() < 3:
            continue
        shift, cut = np.abs(pa[m] - pb[m]), 100.0 * (1.0 - rc.top_share)
        top_a = pa[m] >= cut
        rows.append({"definition_id": d.denominator_id, "position_group": scope, "n_athletes": int(m.sum()),
                     "spearman": ov._fn(ov.spearman(pa[m], pb[m])), "mean_abs_pct_shift": float(shift.mean()),
                     "p90_abs_pct_shift": float(np.percentile(shift, 90)),
                     "top_group_retention": float((pb[m][top_a] >= cut).mean()) if top_a.any() else None, "top_share": rc.top_share})
    return rows


def reliability_suite(rows: dict[str, pl.DataFrame], cfg: ov.ValidationConfig, rc: ReliabilityConfig, progress: Any = None
                      ) -> dict[str, pl.DataFrame]:
    """{REL, FIRST_N, THRESH, RANKS, ATHLETES} for the given definitions (``rows``: definition → ``outcome_rows``)."""
    rel, expo, thr, rank, ath = [], [], [], [], []
    for salt, (k, r) in enumerate(rows.items()):
        d = ov.BY_ID[k]
        gt = ov.game_table(r.select(list(ov.ROW_SCHEMA)))
        ga = ov.game_arrays(gt)
        rel += ov.reliability_rows(d, ga, cfg, salt)
        expo += ov.exposure_rows(d, ga, cfg)
        thr += threshold_rows(d, ga, cfg, rc, salt)
        rank += rank_rows(d, ga, cfg, rc)
        ath.append(ov.athlete_table(gt, d, cfg, salt))
        if progress:
            progress(f"{k}: {r.height:,} rows · {len(ga.ids)} athletes · {gt.height:,} athlete-games")
    athletes = pl.concat(ath).cast(ov.ATHLETE_SCHEMA) if ath else pl.DataFrame(schema=ov.ATHLETE_SCHEMA)
    assert_no_combine_columns(athletes, "football athlete outcomes")
    return {REL.name: pl.DataFrame(rel, schema=ov.REL_SCHEMA), FIRST_N.name: pl.DataFrame(expo, schema=ov.EXPO_SCHEMA),
            THRESH.name: pl.DataFrame(thr, schema=THRESH.columns), RANKS.name: pl.DataFrame(rank, schema=RANKS.columns),
            ATHLETES.name: athletes}


# ============================================================================== context conditioning
def adjust(rows: pl.DataFrame, fit: pl.DataFrame, cells: Sequence[str], cfg: ov.ValidationConfig) -> pl.DataFrame:
    """``rows`` with ``value`` replaced by value − expected value of its context cell, every expectation estimated on
    ``fit`` (rows of OTHER athletes). A cell is used when it holds ≥ ``min_cell_obs`` rows of ≥ ``min_cell_athletes``
    athletes in ``fit`` — a COUNT rule; otherwise the row falls back to the coarser cell (first key), then to the mean
    of ``fit``. A row with a missing label is never assigned a label: it falls back. Adds ``cell_supported``."""
    if not cells:
        return rows.with_columns(pl.lit(True).alias("cell_supported"))
    g = fit["value"].mean()
    out, names = rows, []
    for depth in range(len(cells), 0, -1):
        keys, name = list(cells[:depth]), f"_m{depth}"
        st = (fit.drop_nulls(keys).group_by(keys).agg(pl.len().alias("_n"), pl.col("nfl_id").n_unique().alias("_a"), pl.col("value").mean().alias(name))
              .filter((pl.col("_n") >= cfg.min_cell_obs) & (pl.col("_a") >= cfg.min_cell_athletes)).select(*keys, name))
        out = out.join(st, on=keys, how="left")
        names.append(name)
    expected = pl.coalesce([pl.col(n) for n in names] + [pl.lit(float(g) if g is not None else 0.0)])
    return (out.with_columns((pl.col("value") - expected).alias("value"), pl.col(names[0]).is_not_null().alias("cell_supported"))
            .drop(names))


def athlete_folds(ids: np.ndarray, k: int, seed: int) -> dict[int, int]:
    """Deterministic athlete folds for the context cross-fit (no outcome value is involved)."""
    u = np.unique(ids)
    perm = np.random.default_rng([seed, 808]).permutation(u.size)
    return {int(i): int(f) for i, f in zip(u[perm], np.arange(u.size) % max(2, int(k)))}


def crossfit_adjust(rows: pl.DataFrame, cells: Sequence[str], cfg: ov.ValidationConfig, k: int = 5) -> pl.DataFrame:
    """Context-conditioned rows where the parameters applied to an athlete come from the athletes of the other folds."""
    if not cells:
        return adjust(rows, rows, cells, cfg)
    fold = athlete_folds(rows["nfl_id"].to_numpy(), k, cfg.seed)
    r = rows.with_columns(pl.col("nfl_id").replace_strict(fold, return_dtype=pl.Int64).alias("_fold"))
    parts = [adjust(r.filter(pl.col("_fold") == f), r.filter(pl.col("_fold") != f), cells, cfg) for f in sorted(set(fold.values()))]
    return pl.concat(parts).drop("_fold")


def _athlete_means(rows: pl.DataFrame) -> pl.DataFrame:
    return (rows.group_by("nfl_id", "position_group").agg(pl.len().cast(pl.Int64).alias("n_obs"), pl.col("game_id").n_unique().alias("n_games"),
                                                          pl.col("value").mean().alias("value")).sort("nfl_id"))


def cell_table(definition_id: str, rows: pl.DataFrame, cfg: ov.ValidationConfig) -> pl.DataFrame:
    out = []
    for cid, _, cells, _ in CONTEXTS[1:]:
        keys = list(cells)
        t = (rows.drop_nulls(keys).group_by(keys).agg(pl.len().alias("n_rows"), pl.col("nfl_id").n_unique().alias("n_athletes"),
                                                       pl.col("value").mean().alias("mean_value"), pl.col("value").std().alias("sd_value")))
        for r in t.iter_rows(named=True):
            out.append({"definition_id": definition_id, "context_id": cid, "cell": " × ".join(str(r[c]) for c in keys), "n_rows": r["n_rows"],
                        "n_athletes": r["n_athletes"], "mean_value": r["mean_value"], "sd_value": r["sd_value"],
                        "supported": r["n_rows"] >= cfg.min_cell_obs and r["n_athletes"] >= cfg.min_cell_athletes})
    return pl.DataFrame(out, schema=CELLS.columns).sort("context_id", "n_rows", descending=[False, True])


def context_table(definition_id: str, rows: pl.DataFrame, cfg: ov.ValidationConfig, rc: ReliabilityConfig) -> dict[str, pl.DataFrame]:
    """{CONTEXT, CONTEXT_VALUES, CELLS}: for each context, the athlete values conditioned with parameters from other
    athletes, their odd / even reliability and their agreement with the raw ranking. DESCRIPTIVE: no context
    definition replaces the approved one because it looks better."""
    d = ov.BY_ID[definition_id]
    raw_var = float(rows["value"].var()) if rows.height > 1 else float("nan")
    summary, values, raw_pct = [], [], None
    for j, (cid, label, cells, change) in enumerate(CONTEXTS):
        if cells and any(rows[c].null_count() == rows.height for c in cells):
            continue
        adj = crossfit_adjust(rows, cells, cfg, rc.crossfit_folds)
        ga = ov.game_arrays(ov.game_table(adj.select(list(ov.ROW_SCHEMA))))
        s = split_stat(ga, ga.rk % 2, cfg.min_half_obs, cfg.n_boot, np.random.default_rng([cfg.seed, j, 21]))
        m = _athlete_means(adj).filter((pl.col("n_obs") >= cfg.min_obs) & (pl.col("n_games") >= cfg.min_games))
        pct = _pct(m["value"].to_numpy(), np.asarray(m["position_group"].to_list(), dtype=str))
        m = m.with_columns(pl.Series("pct_position", pct))
        if cid == "R0":
            raw_pct = m.select("nfl_id", pl.col("pct_position").alias("_raw"))
        j = m.join(raw_pct, on="nfl_id", how="inner") if raw_pct is not None else m.head(0).with_columns(pl.lit(None, dtype=pl.Float64).alias("_raw"))
        a, b = j["_raw"].to_numpy(), j["pct_position"].to_numpy()
        summary.append({"definition_id": definition_id, "context_id": cid, "context": label, "estimand_change": change, "n_rows": adj.height,
                        "share_rows_supported_cell": float(adj["cell_supported"].mean()) if adj.height else None, "n_athletes": m.height,
                        "row_variance_removed": (1.0 - float(adj["value"].var()) / raw_var) if cells and raw_var > 0 else 0.0,
                        "rel_sb": s["rel_sb"], "rel_lo": s["lo"], "rel_hi": s["hi"],
                        "spearman_vs_raw": ov._fn(ov.spearman(a, b)) if a.size >= 3 else None,
                        "mean_abs_pct_shift_vs_raw": float(np.abs(a - b).mean()) if a.size else None})
        values.append(m.select(pl.lit(definition_id).alias("definition_id"), pl.lit(cid).alias("context_id"), "nfl_id", "position_group",
                               "n_obs", "value", "pct_position"))
    return {CONTEXT.name: pl.DataFrame(summary, schema=CONTEXT.columns),
            CONTEXT_VALUES.name: pl.concat(values).cast(CONTEXT_VALUES.columns) if values else CONTEXT_VALUES.empty(),
            CELLS.name: cell_table(definition_id, rows, cfg)}


# ============================================================================== descriptive groups
def group_table(definition_id: str, rows: pl.DataFrame, cfg: ov.ValidationConfig, draft_year: dict[int, int] | None = None) -> pl.DataFrame:
    """Support and level by NFL season (rows of that season), draft cohort and number of games (athlete level)."""
    out = []

    def add(kind: str, name: str, r: pl.DataFrame) -> None:
        m = _athlete_means(r)
        out.append({"definition_id": definition_id, "group_kind": kind, "group": str(name), "n_athletes": m.height, "n_rows": r.height,
                    "mean_value": ov._fn(r["value"].mean()), "sd_athlete": ov._fn(m["value"].std()) if m.height > 1 else None,
                    "median_games": ov._fn(m["n_games"].median())})

    for (s,), r in sorted(rows.partition_by(["season"], as_dict=True).items()):
        add("nfl_season", s, r)
    for (p,), r in sorted(rows.partition_by(["position_group"], as_dict=True).items()):
        add("position_group", p, r)
    if draft_year:
        r2 = rows.with_columns(pl.col("nfl_id").replace_strict(draft_year, default=None, return_dtype=pl.Int64).alias("_dy")).drop_nulls("_dy")
        for (y,), r in sorted(r2.partition_by(["_dy"], as_dict=True).items()):
            add("draft_cohort", y, r.drop("_dy"))
    g = rows.group_by("nfl_id").agg(pl.col("game_id").n_unique().alias("_g"))
    band = g.with_columns(pl.when(pl.col("_g") < cfg.min_games).then(pl.lit(f"< {cfg.min_games} games"))
                          .when(pl.col("_g") < 2 * cfg.min_games).then(pl.lit(f"{cfg.min_games}–{2 * cfg.min_games - 1} games"))
                          .when(pl.col("_g") < 4 * cfg.min_games).then(pl.lit(f"{2 * cfg.min_games}–{4 * cfg.min_games - 1} games"))
                          .otherwise(pl.lit(f"≥ {4 * cfg.min_games} games")).alias("_band"))
    r3 = rows.join(band.select("nfl_id", "_band"), on="nfl_id", how="left")
    for (b,), r in sorted(r3.partition_by(["_band"], as_dict=True).items()):
        add("games_observed", b, r.drop("_band"))
    return pl.DataFrame(out, schema=GROUPS.columns)


def targeted_split(rows_t: pl.DataFrame | None, rows_n: pl.DataFrame | None, rows_all: pl.DataFrame, cfg: ov.ValidationConfig) -> pl.DataFrame:
    """Do the same athletes separate on targeted and on non-targeted plays? (Selection made visible, not corrected.)"""
    if rows_t is None or rows_n is None or rows_t.is_empty() or rows_n.is_empty():
        return TARGET_SPLIT.empty()
    half = max(1, cfg.min_obs // 2)
    a = _athlete_means(rows_t).filter(pl.col("n_obs") >= half).select("nfl_id", "position_group", pl.col("value").alias("t"))
    b = _athlete_means(rows_n).filter(pl.col("n_obs") >= half).select("nfl_id", pl.col("value").alias("n"))
    j = a.join(b, on="nfl_id", how="inner")
    out = []
    for scope in (ALL, *sorted(j["position_group"].unique().to_list())):
        c = j if scope == ALL else j.filter(pl.col("position_group") == scope)
        if c.height < 3:
            continue
        pos = np.asarray(c["position_group"].to_list(), dtype=str)
        t, n = c["t"].to_numpy(), c["n"].to_numpy()
        rel = {}
        for name, r in (("t", rows_t), ("n", rows_n)):
            ga = ov.game_arrays(ov.game_table(r.select(list(ov.ROW_SCHEMA))))
            rel[name] = split_stat(ga, ga.rk % 2, max(2, cfg.min_half_obs // 2), 0, np.random.default_rng(0), scope)["rel_sb"]
        ra = rows_all if scope == ALL else rows_all.filter(pl.col("position_group") == scope)
        out.append({"position_group": scope, "n_athletes": c.height, "mean_targeted": float(t.mean()), "mean_not_targeted": float(n.mean()),
                    "mean_difference": float((t - n).mean()), "r_between": ov._fn(ov._r(ov._centred(t, pos), ov._centred(n, pos))),
                    "rel_targeted": rel["t"], "rel_not_targeted": rel["n"], "share_rows_targeted": ov._fn(ra["tgt"].mean())})
    return pl.DataFrame(out, schema=TARGET_SPLIT.columns)


# ============================================================================== secondary outcomes
def attribution_table(base: pl.DataFrame) -> pl.DataFrame:
    """Solo and assisted tackle credits per eligible defensive snap. Credits are scorer attributions; a snap is
    exposure, not a tackle opportunity — the table shows how much of the 'rate' is assists and playing time."""
    if "tackle" not in base.columns:
        return ATTRIBUTION.empty()
    d = ov.BY_ID["FO12.T2"]
    tk = osm.numeric("tackle")
    asst = osm.numeric("assist") if "assist" in base.columns else pl.lit(None, dtype=pl.Float64)
    el = base.filter(pl.col("eligible") & pl.col("position_group").is_in(list(d.positions))).select("position_group", tk.alias("t"), asst.alias("a"))
    grid = pl.concat([el, el.with_columns(pl.lit(ALL).alias("position_group"))])
    solo, assist = pl.col("t").fill_null(0.0) > 0, pl.col("a").fill_null(0.0) > 0
    t = grid.group_by("position_group").agg(pl.len().alias("n_snaps"), (pl.col("t").is_not_null() | pl.col("a").is_not_null()).sum().alias("n_statistic_rows"),
                                             solo.sum().alias("n_solo"), assist.sum().alias("n_assist"), (solo & assist).sum().alias("n_both"),
                                             (solo | assist).sum().alias("_c"))
    return (t.with_columns((pl.col("_c") / pl.col("n_snaps")).alias("credits_per_snap"),
                           pl.when(pl.col("_c") > 0).then(pl.col("n_assist") / pl.col("_c")).otherwise(None).alias("assist_share_of_credits"),
                           (pl.col("n_statistic_rows") / pl.col("n_snaps")).alias("share_snaps_with_statistic"))
            .select([pl.col(c).cast(tp) for c, tp in ATTRIBUTION.columns.items()]).sort("position_group"))


def sack_exposure_table(athletes: pl.DataFrame) -> pl.DataFrame:
    """Sack rates against their own exposure: a rate that follows playing time ranks usage, not rushing."""
    a = athletes.filter(pl.col("denominator_id").str.starts_with("FO13.") & pl.col("supported"))
    if a.is_empty():
        return SACK.empty()
    dep = {r["denominator_id"]: r for r in ov.dependence_table(a).iter_rows(named=True)}
    rows = []
    for (k,), t in a.partition_by(["denominator_id"], as_dict=True).items():
        rows.append({"definition_id": k, "n_athletes": t.height, "mean_rate": ov._fn(t["value"].mean()), "median_rows": ov._fn(t["n_obs"].median()),
                     "rho_value_log_obs": dep.get(k, {}).get("rho_value_log_obs"), "rho_value_games": dep.get(k, {}).get("rho_value_games"),
                     "share_athletes_zero": float((t["value"] == 0).mean())})
    return pl.DataFrame(rows, schema=SACK.columns).sort("definition_id")


# ============================================================================== Gate B
def gate_b(targets: Sequence[dict[str, Any]], rel: pl.DataFrame, athletes: pl.DataFrame, links: pl.DataFrame, ranks: pl.DataFrame,
           cfg: ov.ValidationConfig) -> tuple[pl.DataFrame, dict[str, str]]:
    """Gate B — is the target reproducible with sufficient athlete and game support? Returns (criteria, {target: status}).

    PASS  odd / even reliability ≥ the reference line with a lower bound ≥ the floor, and linked support.
    WEAK  reliability ≥ the floor with a positive lower bound: secondary / sensitivity use only.
    FAIL  otherwise: no predictive analysis of this target.
    The exposure criterion is blocking for a RATE (a rate that follows playing time is not an athlete characteristic)."""
    dep = {r["denominator_id"]: r for r in ov.dependence_table(athletes).iter_rows(named=True)}
    rows, status = [], {}
    for t in targets:
        k, d = t["target_id"], ov.BY_ID[t["target_id"]]
        oe = rel.filter((pl.col("denominator_id") == k) & (pl.col("split") == "odd_even") & (pl.col("position_group") == ALL))
        sb, lo = (oe["rel_sb"][0], oe["rel_lo"][0]) if oe.height else (None, None)
        lk = links.filter((pl.col("denominator_id") == k) & pl.col("all_models"))
        n_link = lk.height
        by_pos = dict(lk.group_by("position_group").len().iter_rows())
        n_pos = sum(v >= cfg.min_linked_position for v in by_pos.values())
        rho = dep.get(k, {}).get("rho_value_log_obs")
        rk = ranks.filter((pl.col("definition_id") == k) & (pl.col("position_group") == ALL))
        strong = sb is not None and lo is not None and sb >= cfg.reliability_ref and lo >= cfg.reliability_floor
        weak = sb is not None and lo is not None and sb >= cfg.reliability_floor and lo > 0
        support = n_link >= cfg.min_linked and n_pos >= 1
        expo_ok = rho is not None and abs(rho) <= cfg.exposure_ref
        crit = [("player_level_stability", f"odd / even Spearman–Brown {sb if sb is None else round(sb, 3)} (lower bound {lo if lo is None else round(lo, 3)}); "
                                           f"reference {cfg.reliability_ref}, floor {cfg.reliability_floor}", strong, True),
                ("opportunity_support", f"{n_link} linked athletes supporting every model family (needed {cfg.min_linked})", n_link >= cfg.min_linked, True),
                ("position_coverage", ", ".join(f"{p} {n}" for p, n in sorted(by_pos.items())) or "none", n_pos >= 1, True),
                ("exposure_independence", f"ρ(value, log observations) = {rho if rho is None else round(rho, 3)} (|ρ| ≤ {cfg.exposure_ref})", expo_ok, d.kind == "rate"),
                ("ranking_sensitivity", f"mean |percentile shift| between game halves {rk['mean_abs_pct_shift'][0]:.1f}" if rk.height else "not assessable",
                 rk.height > 0, False),
                ("definition_validity", f"source class {t['source_confidence']} — {t['interpretation_label']}", t["source_confidence"] != "BLOCKED", True)]
        rows += [{"target_id": k, "role": t["role"], "criterion": c, "value": v, "passed": bool(p), "blocking": b} for c, v, p, b in crit]
        blocked = not support or t["source_confidence"] == "BLOCKED" or (d.kind == "rate" and not expo_ok)
        status[k] = "FAIL" if blocked or not weak else ("PASS" if strong else "WEAK")
    return pl.DataFrame(rows, schema=GATE_B.columns), status


def conclusions_markdown(gate: pl.DataFrame, status: dict[str, str], rel: pl.DataFrame, context: pl.DataFrame, cfg: ov.ValidationConfig) -> str:
    f = lambda x: "—" if x is None else f"{x:.2f}"
    lines = ["### 08B.1 — outcome reliability and context", "", "| Target | Gate B | Position | Odd / even reliability [95 %] | Random halves | n |",
             "|---|---|---|---|---|---|"]
    for k, s in status.items():
        for r in rel.filter((pl.col("denominator_id") == k) & (pl.col("split") == "odd_even")).iter_rows(named=True):
            rnd = rel.filter((pl.col("denominator_id") == k) & (pl.col("split") == "random") & (pl.col("position_group") == r["position_group"]))
            lines.append(f"| `{k}` | {s} | {r['position_group']} | {f(r['rel_sb'])} [{f(r['rel_lo'])}, {f(r['rel_hi'])}] | "
                         f"{f(rnd['rel_sb'][0]) if rnd.height else '—'} | {r['n_athletes']} |")
    lines += ["", "| Definition | Context | Reliability | Rank agreement with raw | Row variance removed | Estimand |", "|---|---|---|---|---|---|"]
    lines += [f"| `{r['definition_id']}` | {r['context']} | {f(r['rel_sb'])} | {f(r['spearman_vs_raw'])} | {f(r['row_variance_removed'])} | {r['estimand_change']} |"
              for r in context.iter_rows(named=True)]
    failed = gate.filter(pl.col("blocking") & ~pl.col("passed"))
    lines += [""] + [f"- `{r['target_id']}` fails **{r['criterion']}**: {r['value']}" for r in failed.iter_rows(named=True)]
    lines += ["", f"Reference lines (heuristics declared in 08A.2): reliability ≥ {cfg.reliability_ref:.2f} with a lower bound ≥ {cfg.reliability_floor:.2f}; "
                  f"≥ {cfg.min_linked} linked athletes. Reliability says athletes differ stably in the RECORDED outcome; it does not say the "
                  "outcome is a skill of the athlete alone. Position-level rows (about 24 linked TE) are descriptive, not independent evidence."]
    return "\n".join(lines)


__all__ = ["ATHLETES", "ATTRIBUTION", "CELLS", "CONTEXT", "CONTEXTS", "CONTEXT_VALUES", "FIRST_N", "GATE_B", "GROUPS", "RANKS", "REL", "ROWS",
           "ROW_SCHEMA", "ReliabilityConfig", "rows_frame", "rows_of", "SACK", "TARGET_SPLIT", "THRESH", "adjust", "athlete_folds", "attribution_table",
           "cell_table", "conclusions_markdown", "context_table", "crossfit_adjust", "gate_b", "group_table", "outcome_rows",
           "rank_rows", "reliability_suite", "sack_exposure_table", "split_stat", "targeted_split", "threshold_rows"]
