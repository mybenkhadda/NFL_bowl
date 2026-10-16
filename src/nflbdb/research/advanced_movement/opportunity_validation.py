"""Milestone 08A.2 — opportunity denominators, game-split reliability, linked Combine cohort, explicit target freeze.

    DENOMINATORS          pre-declared opportunity populations of each candidate outcome (static, written before reading)
    denominator_rows      one row per player-play of a denominator: athlete, game, value
    denominator_table     athletes, games, rows, targeted share per denominator × position
    athlete_table         athlete value, game-bootstrap interval, odd / even halves, support, within-position percentile
    reliability_table     odd / even games (athlete bootstrap) and random game halves — halves are WHOLE GAMES
    exposure_table        first N games versus later games; reliability among athletes with at least G games
    ranking_table         does the opportunity rule change the athlete ranking?
    link_athletes / link_funnel / cohort_support      Combine linkage by COUNTS (no Combine value is read)
    screen_table / propose                            rule on semantics, denominator, support, reliability, exposure
    freeze_football_targets                           explicit analyst decision → immutable, versioned registry
    require_frozen                                    the gate notebook 08B calls before any transfer model

Legitimacy of a denominator: ``recorded`` (the vendor recorded a value on exactly these rows) · ``structural``
(tracked snaps on the field; a null is read as "no credit", which is an INFERENCE and caps the status at provisional) ·
``inferred`` / ``selected`` / ``credited_only`` (sensitivity only, never frozen).

Nothing here reads a Combine measurement: the linked cohort is built from the linkage FLAGS of the Milestone 06
manifest, and the screen uses declarations, counts and NFL-side reliability only. The Milestone 05 registry
(``nfl_movement_targets_v1``) is never written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import atomic_write_json, get_logger, read_json, utc_now
from ..contracts import TableContract
from . import outcome_semantics as osm
from . import scouting_feasibility as sf
from .config import NotebookControls
from .nfl_outcomes import assert_no_combine_columns

log = get_logger(__name__)
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
VALIDATION_VERSION = "1"
ALL = "ALL"
REGISTRY_FILE = "football_outcome_targets"
PROPOSAL_FILE = "football_outcome_freeze_proposal"
DECISIONS = ("FREEZE", "PROVISIONAL", "STOP")
FREEZABLE = ("recorded", "structural")
PRIMARY_METRIC = {"continuous": "paired ΔRMSE of M5 versus M1 on athlete-held-out predictions (athlete bootstrap)",
                  "rate": "paired Δ mean Poisson deviance of M5 versus M1 on athlete-held-out predictions, exposure = denominator rows"}


class FootballTargetFreezeError(RuntimeError):
    """No valid, approved freeze of the football-outcome registry: transfer models must not run."""


class FreezeConflictError(RuntimeError):
    """A registry of this version exists with another content: a changed definition needs a NEW version."""


# ============================================================================== configuration
@dataclass(frozen=True)
class ValidationConfig:
    """Support counts and reference lines. All are heuristics declared before running; none is tuned on a result."""

    min_obs: int = 20                # observations for one athlete value
    min_games: int = 4               # distinct games behind them
    min_half_obs: int = 8            # observations in EACH game half
    min_cell_obs: int = 200          # S2: observations in a route × alignment cell
    min_cell_athletes: int = 15      # S2: athletes in the cell
    min_scope_athletes: int = 10     # athletes for a reliability estimate of one position
    n_boot: int = 1000               # athlete bootstrap of the odd / even reliability
    n_random_splits: int = 100       # random game-half splits
    n_boot_games: int = 200          # game bootstrap of one athlete's value
    first_games_grid: tuple[int, ...] = (2, 4, 6, 8)
    min_games_grid: tuple[int, ...] = (4, 8, 12, 16)
    reliability_ref: float = 0.70    # Spearman–Brown, odd / even games
    reliability_floor: float = 0.50  # below it (or lower bound ≤ 0) the outcome is rejected
    exposure_ref: float = 0.30       # |ρ(value, log observations)|
    min_linked: int = 60             # linked athletes supporting every model family
    min_linked_position: int = 15
    extreme_yd: float = 15.0         # separation above it is reported as extreme (reported, not removed)
    seed: int = 20261011

    def for_mode(self, controls: NotebookControls) -> "ValidationConfig":
        if not controls.smoke:
            return replace(self, seed=controls.seed)
        return replace(self, min_obs=5, min_games=2, min_half_obs=2, min_cell_obs=10, min_cell_athletes=3, min_scope_athletes=4,
                       n_boot=100, n_random_splits=10, n_boot_games=50, first_games_grid=(1, 2), min_games_grid=(2, 3),
                       min_linked=6, min_linked_position=2, seed=controls.seed)

    def params(self) -> dict[str, Any]:
        return {**asdict(self), "validation_version": VALIDATION_VERSION}


@dataclass(frozen=True)
class Denominator:
    denominator_id: str
    outcome_id: str
    label: str
    positions: tuple[str, ...]
    needs: tuple[str, ...]            # vendor fields without which the denominator does not exist
    legitimacy: str                   # recorded · structural · inferred · selected · credited_only
    kind: str                         # continuous · rate
    unit: str
    formula: str
    assumption: str = ""              # what is NOT recorded and had to be assumed
    statistic: str = "mean"
    context_adjustment: str = "none"


_SEP, _TKL, _SCK = "separation_at_pass_forward", "tackle", "sack"
DENOMINATORS: tuple[Denominator, ...] = (
    Denominator("FO01.S0", osm.FO01, "all recorded separations", ("WR", "TE"), (_SEP,), "recorded", "continuous", "yd",
                "mean of separation_at_pass_forward over eligible player-plays with a recorded value"),
    Denominator("FO01.S1", osm.FO01, "recorded separations with a route label", ("WR", "TE"), (_SEP, "route_ran"), "recorded", "continuous",
                "yd", "mean of separation_at_pass_forward over eligible player-plays with a recorded value AND a recorded route_ran"),
    Denominator("FO01.S2", osm.FO01, "route × alignment adjusted separations (supported cells)", ("WR", "TE"),
                (_SEP, "route_ran"), "recorded", "continuous", "yd (residual)",
                "mean over S1 rows in supported route × alignment cells of (separation − cell mean)",
                assumption="cell means are estimated on all athletes here; a transfer model must refit them on training athletes",
                context_adjustment="cell_residual(route_ran × lined_up_position)"),
    Denominator("FO01.S0T", osm.FO01, "targeted only (sensitivity)", ("WR", "TE"), (_SEP, "target"), "selected", "continuous", "yd",
                "mean of separation_at_pass_forward over S0 rows with target = true",
                assumption="the quarterback chooses the target: the plays are selected on the outcome"),
    Denominator("FO01.S0N", osm.FO01, "not targeted only (sensitivity)", ("WR", "TE"), (_SEP, "target"), "selected", "continuous", "yd",
                "mean of separation_at_pass_forward over S0 rows with target = false",
                assumption="complement of a selection on the outcome"),
    Denominator("FO12.T0", osm.FO12, "rows with a recorded tackle statistic", ("DB", "DL"), (_TKL,), "credited_only", "rate",
                "share of recorded rows", "mean of tackle over eligible player-plays with a recorded tackle value",
                assumption="the rows exist only when a tackle statistic was credited: non-successes are not observed"),
    Denominator("FO12.T1", osm.FO12, "credited tackles per tracked snap", ("DB", "DL"), (_TKL,), "structural", "rate",
                "tackles per eligible snap", "sum of (tackle > 0) over ALL eligible player-plays of the athlete / number of those player-plays",
                assumption="a null is read as 'no tackle credited'; snaps on the field are exposure, not tackle opportunities"),
    Denominator("FO12.T2", osm.FO12, "credited tackles + assists per tracked snap", ("DB", "DL"), (_TKL, "assist"), "structural", "rate",
                "tackles or assists per eligible snap",
                "sum of (tackle > 0 or assist > 0) over ALL eligible player-plays of the athlete / number of those player-plays",
                assumption="a null is read as 'no credit'; snaps on the field are exposure, not tackle opportunities"),
    Denominator("FO13.K0", osm.FO13, "sack credit per recorded row", ("DL",), (_SCK,), "recorded", "rate", "sacks per recorded row",
                "mean of sack (0, 0.5, 1) over eligible player-plays with a recorded sack value",
                assumption="a recorded row is a defensive pass-play row; that the player rushed the passer on it is not recorded"),
    Denominator("FO13.K1", osm.FO13, "sack credit per recorded row without a coverage assignment", ("DL",), (_SCK, "coverage_assignment"),
                "inferred", "rate", "sacks per inferred rush row",
                "mean of sack over K0 rows with coverage_assignment = NO_ASSIGNMENT",
                assumption="NO_ASSIGNMENT is read as 'rushed the passer': an inference about the assignment"),
    Denominator("FO13.K2", osm.FO13, "sack credit per tracked snap", ("DL",), (_SCK,), "structural", "rate", "sacks per eligible snap",
                "sum of sack over ALL eligible player-plays of the athlete / number of those player-plays",
                assumption="a null is read as 'no sack credited'; run plays are in the denominator"),
)
BY_ID = {d.denominator_id: d for d in DENOMINATORS}
PRIORITY = ("FO01.S0", "FO01.S1", "FO01.S2", "FO13.K0", "FO12.T1", "FO12.T2", "FO13.K2")   # order of the rule-based proposal


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def denominators_hash(dens: Sequence[Denominator] = DENOMINATORS) -> str:
    return _hash([asdict(d) for d in dens])


# ============================================================================== contracts
DENOM_SCHEMA = {"denominator_id": S, "outcome_id": S, "label": S, "formula": S, "unit": S, "statistic": S, "kind": S, "legitimacy": S,
                "assumption": S, "available": B, "position_group": S, "n_rows": I, "n_athletes_any": I, "n_athletes_supported": I,
                "n_athlete_games": I, "median_obs": F, "median_games": F, "mean_value": F, "targeted_share": F}
SEASON_SCHEMA = {"denominator_id": S, "season": I, "n_rows": I, "n_athletes": I, "n_games": I, "mean_value": F}
ATHLETE_SCHEMA = {"denominator_id": S, "outcome_id": S, "nfl_id": I, "position_group": S, "n_obs": I, "n_games": I, "n_seasons": I,
                  "first_season": I, "value": F, "se_game_boot": F, "lo": F, "hi": F, "half_a": F, "half_b": F, "supported": B,
                  "pct_position": F}
REL_SCHEMA = {"denominator_id": S, "outcome_id": S, "split": S, "position_group": S, "n_athletes": I, "r_half": F, "rel_sb": F,
              "rel_lo": F, "rel_hi": F, "interval_kind": S, "note": S}
EXPO_SCHEMA = {"denominator_id": S, "mode": S, "level": I, "n_athletes": I, "r": F, "rel_sb": F}
RANK_SCHEMA = {"outcome_id": S, "definition_a": S, "definition_b": S, "position_group": S, "n_common": I, "spearman": F,
               "mean_abs_pct_shift": F, "max_abs_pct_shift": F}
DEP_SCHEMA = {"denominator_id": S, "n_athletes": I, "rho_value_log_obs": F, "rho_value_games": F}
LINK_FLAGS = ("linked", "has_official", "has_tracking", "has_repeat", "has_engineered", "has_speed", "has_fpca", "has_tcn")
ELIG = ("elig_M1", "elig_M2", "elig_M3", "elig_M4", "elig_M5", "all_models")
LINKS_SCHEMA = {"denominator_id": S, "nfl_id": I, "position_group": S, "draft_year": I, "supported": B,
                **{k: B for k in (*LINK_FLAGS, *ELIG)}}
FUNNEL_SCHEMA = {"denominator_id": S, "position_group": S, "stage": S, "stage_order": I, "cumulative": B, "n_athletes": I}
SUPPORT_SCHEMA = {"denominator_id": S, "position_group": S, "draft_year": I, "n_supported_linked": I, "n_all_models": I}
SCREEN_SCHEMA = {"denominator_id": S, "outcome_id": S, "legitimacy": S, "semantic_status": S, "unit": S, "available": B, "n_supported": I,
                 "n_linked_all_models": I, "n_positions_linked": I, "rel_sb": F, "rel_lo": F, "rel_hi": F, "rel_random": F,
                 "exposure_rho": F, "c_semantics": B, "c_denominator": B, "c_support": B, "c_reliability": B, "c_exposure": B,
                 "status": S, "reason": S}

DENOMS = TableContract("football_opportunity_denominators", "Candidate opportunity denominators: declaration and measured support by position.",
                       DENOM_SCHEMA, key=("denominator_id", "position_group"), area="denominator × position group",
                       docs={"legitimacy": "recorded · structural · inferred · selected · credited_only",
                             "n_athletes_supported": "athletes with ≥ min_obs rows in ≥ min_games games"})
SEASONS = TableContract("football_opportunity_seasons", "Rows, athletes, games and mean value per denominator and season.", SEASON_SCHEMA,
                        key=("denominator_id", "season"), area="denominator × season")
ATHLETES = TableContract("football_outcome_athlete_values", "Athlete value of each denominator with a game-bootstrap interval and game halves.",
                         ATHLETE_SCHEMA, key=("denominator_id", "nfl_id"), kind="dataset", area="denominator × athlete",
                         docs={"lo": "percentile 2.5 of the athlete's value when his games are resampled",
                               "pct_position": "mid-rank percentile among supported athletes of the same position (descriptive)"})
RELIABILITY = TableContract("football_outcome_reliability", "Game-split reliability of the athlete value of each denominator.", REL_SCHEMA,
                            key=("denominator_id", "split", "position_group"), area="denominator × split × position group",
                            docs={"split": "odd_even = chronological parity of the athlete's games · random = random game halves",
                                  "interval_kind": "athlete_bootstrap · split_distribution (percentiles over the random splits)"})
EXPOSURE = TableContract("football_outcome_reliability_exposure", "Reliability against game exposure.", EXPO_SCHEMA,
                         key=("denominator_id", "mode", "level"), area="denominator × mode × level",
                         docs={"mode": "first_n_vs_later = value of the first N games against all later games (correlation, no step-up) · "
                                       "min_games = odd / even reliability among athletes with at least N games"})
RANKING = TableContract("football_outcome_ranking_sensitivity", "Agreement of athlete rankings between opportunity definitions of one outcome.",
                        RANK_SCHEMA, key=("outcome_id", "definition_a", "definition_b", "position_group"), area="definition pair × position group")
DEPENDENCE = TableContract("football_outcome_exposure_dependence", "Association of the athlete value with its own exposure.", DEP_SCHEMA,
                           key=("denominator_id",), area="denominator")
LINKS = TableContract("football_outcome_athlete_links", "Combine linkage flags of every athlete with an outcome value.", LINKS_SCHEMA,
                      key=("denominator_id", "nfl_id"), kind="dataset", area="denominator × athlete")
FUNNEL = TableContract("football_outcome_linked_cohort", "Linked-cohort funnel of each denominator: NFL support → Combine link → model families.",
                       FUNNEL_SCHEMA, key=("denominator_id", "position_group", "stage"), area="denominator × position group × stage",
                       docs={"cumulative": "true = each stage is a subset of the previous one; false = a side count"})
SUPPORT = TableContract("football_outcome_cohort_support", "Linked athletes by position group and draft cohort.", SUPPORT_SCHEMA,
                        key=("denominator_id", "position_group", "draft_year"), area="denominator × position group × draft cohort")
SCREEN = TableContract("football_outcome_screen", "Readiness screen of each denominator (rule on declarations, counts and NFL-side reliability).",
                       SCREEN_SCHEMA, key=("denominator_id",), area="denominator",
                       docs={"status": "eligible · provisional · rejected · sensitivity_only · unavailable"})
OPPORTUNITY_CONTRACTS = (DENOMS, SEASONS, ATHLETES, RELIABILITY, EXPOSURE, RANKING, DEPENDENCE)
LINK_CONTRACTS = (LINKS, FUNNEL, SUPPORT)


# ============================================================================== denominator rows
ROW_SCHEMA = {"nfl_id": I, "position_group": S, "season": I, "week": I, "game_id": I, "value": F, "tgt": F}


def available(base: pl.DataFrame, d: Denominator) -> bool:
    return all(f in base.columns for f in d.needs)


def denominator_rows(base: pl.DataFrame, d: Denominator, cfg: ValidationConfig) -> pl.DataFrame | None:
    """One row per player-play of the denominator, or None when a needed field is absent (nothing is created)."""
    if not available(base, d):
        return None
    el = base.filter(pl.col("eligible") & pl.col("position_group").is_in(list(d.positions)))
    tgt = osm.flag(base, "target") if "target" in base.columns else pl.lit(None, dtype=pl.Float64)
    k = d.denominator_id
    if d.outcome_id == osm.FO01:
        el = el.with_columns(osm.numeric(_SEP).alias("value"), tgt.alias("tgt")).filter(pl.col("value").is_not_null())
        if k in ("FO01.S1", "FO01.S2"):
            el = el.filter(osm.observed(base, "route_ran"))
        if k == "FO01.S2":
            cell = ["route_ran", "lined_up_position"]
            stats = el.group_by(cell).agg(pl.len().alias("_n"), pl.col("nfl_id").n_unique().alias("_a"), pl.col("value").mean().alias("_m"))
            el = (el.join(stats, on=cell, how="left")
                  .filter((pl.col("_n") >= cfg.min_cell_obs) & (pl.col("_a") >= cfg.min_cell_athletes))
                  .with_columns((pl.col("value") - pl.col("_m")).alias("value")))
        if k == "FO01.S0T":
            el = el.filter(pl.col("tgt") == 1.0)
        if k == "FO01.S0N":
            el = el.filter(pl.col("tgt") == 0.0)
    elif k == "FO12.T0":
        el = el.with_columns(osm.numeric(_TKL).alias("value")).filter(pl.col("value").is_not_null())
    elif k == "FO12.T1":
        el = el.with_columns((osm.numeric(_TKL).fill_null(0.0) > 0).cast(pl.Float64).alias("value"))
    elif k == "FO12.T2":
        el = el.with_columns(((osm.numeric(_TKL).fill_null(0.0) > 0) | (osm.numeric("assist").fill_null(0.0) > 0)).cast(pl.Float64).alias("value"))
    elif k == "FO13.K0":
        el = el.with_columns(osm.numeric(_SCK).alias("value")).filter(pl.col("value").is_not_null())
    elif k == "FO13.K1":
        el = (el.with_columns(osm.numeric(_SCK).alias("value"))
              .filter(pl.col("value").is_not_null() & (pl.col("coverage_assignment").cast(pl.String).str.strip_chars() == "NO_ASSIGNMENT")))
    elif k == "FO13.K2":
        el = el.with_columns(osm.numeric(_SCK).fill_null(0.0).alias("value"))
    else:
        raise KeyError(f"no row rule for denominator {k!r}")
    if "tgt" not in el.columns:
        el = el.with_columns(pl.lit(None, dtype=pl.Float64).alias("tgt"))
    return el.select([pl.col(c).cast(t) for c, t in ROW_SCHEMA.items()])


def all_rows(base: pl.DataFrame, cfg: ValidationConfig, dens: Sequence[Denominator] = DENOMINATORS) -> dict[str, pl.DataFrame]:
    out = {}
    for d in dens:
        r = denominator_rows(base, d, cfg)
        if r is not None:
            assert_no_combine_columns(r, f"rows of {d.denominator_id}")
            out[d.denominator_id] = r
    return out


def game_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Athlete × game sufficient statistics (count, sum), in the athlete's chronological game order."""
    return (rows.group_by("nfl_id", "position_group", "season", "week", "game_id")
            .agg(pl.len().alias("n"), pl.col("value").sum().alias("s")).sort("nfl_id", "season", "week", "game_id"))


@dataclass(frozen=True)
class GameArrays:
    ids: np.ndarray     # athlete ids (sorted)
    pos: np.ndarray     # position group per athlete
    ai: np.ndarray      # athlete index per athlete-game row
    rk: np.ndarray      # 0-based chronological rank of the game within the athlete
    start: np.ndarray   # first row of each athlete
    n: np.ndarray       # observations of the athlete-game
    s: np.ndarray       # sum of the values of the athlete-game


def game_arrays(gt: pl.DataFrame) -> GameArrays:
    ids, ai = np.unique(gt["nfl_id"].to_numpy(), return_inverse=True)
    start = np.searchsorted(ai, np.arange(len(ids)))
    pos = np.asarray(gt["position_group"].to_list(), dtype=object)[start].astype(str) if len(ids) else np.array([], dtype=str)
    return GameArrays(ids, pos, ai, np.arange(len(ai)) - start[ai] if len(ai) else np.array([], dtype=int), start,
                      gt["n"].to_numpy().astype(float), gt["s"].to_numpy().astype(float))


# ============================================================================== reliability (halves are whole games)
def _fn(x: Any) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def half_stats(ga: GameArrays, half: np.ndarray, min_a: int, min_b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(keep, mean in half 0, mean in half 1) per athlete; a game with another label belongs to neither half."""
    k = len(ga.ids)
    a, b = half == 0, half == 1
    na, nb = np.bincount(ga.ai, weights=ga.n * a, minlength=k), np.bincount(ga.ai, weights=ga.n * b, minlength=k)
    sa, sb = np.bincount(ga.ai, weights=ga.s * a, minlength=k), np.bincount(ga.ai, weights=ga.s * b, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (na >= min_a) & (nb >= min_b), np.where(na > 0, sa / na, np.nan), np.where(nb > 0, sb / nb, np.nan)


def _centred(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = x.astype(float).copy()
    for p in np.unique(pos):
        m = pos == p
        out[m] -= out[m].mean()
    return out


def pair(ga: GameArrays, half: np.ndarray, min_a: int, min_b: int, scope: str = ALL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Athletes kept (index), position-centred value in each half."""
    keep, ma, mb = half_stats(ga, half, min_a, min_b)
    if scope != ALL:
        keep = keep & (ga.pos == scope)
    idx = np.flatnonzero(keep)
    return idx, _centred(ma[idx], ga.pos[idx]), _centred(mb[idx], ga.pos[idx])


def _r(a: np.ndarray, b: np.ndarray) -> float:
    return float(sf._corr(a[None, :], b[None, :])[0]) if len(a) >= 3 else float("nan")


def random_half(ga: GameArrays, rng: np.random.Generator) -> np.ndarray:
    """Random split of each athlete's GAMES into two halves (sizes differ by at most one)."""
    order = np.lexsort((rng.random(len(ga.ai)), ga.ai))
    half = np.empty(len(ga.ai), dtype=int)
    half[order] = (np.arange(len(ga.ai)) - ga.start[ga.ai[order]]) % 2
    return half


def reliability_rows(d: Denominator, ga: GameArrays, cfg: ValidationConfig, salt: int) -> list[dict[str, Any]]:
    rows = []
    for j, scope in enumerate((ALL, *d.positions) if len(d.positions) > 1 else (ALL,)):
        base = {"denominator_id": d.denominator_id, "outcome_id": d.outcome_id, "position_group": scope}
        need = 3 if scope == ALL else cfg.min_scope_athletes
        idx, ra, rb = pair(ga, ga.rk % 2, cfg.min_half_obs, cfg.min_half_obs, scope)
        row = {**base, "split": "odd_even", "n_athletes": len(idx), "r_half": None, "rel_sb": None, "rel_lo": None, "rel_hi": None,
               "interval_kind": "athlete_bootstrap", "note": ""}
        r = _r(ra, rb)
        if len(idx) < need or not np.isfinite(r):
            row["note"] = f"fewer than {need} athletes with ≥ {cfg.min_half_obs} observations in each game half, or no variation"
        else:
            rng = np.random.default_rng([cfg.seed, salt, j, 0])
            bi = rng.integers(0, len(idx), (cfg.n_boot, len(idx)))
            boot = sf._sb(sf._corr(ra[bi], rb[bi]))
            row |= {"r_half": r, "rel_sb": _fn(sf._sb(r)), "rel_lo": _fn(np.nanpercentile(boot, 2.5)), "rel_hi": _fn(np.nanpercentile(boot, 97.5))}
        rows.append(row)
        rng = np.random.default_rng([cfg.seed, salt, j, 1])
        reps, ns = [], []
        for _ in range(cfg.n_random_splits):
            idx, ra, rb = pair(ga, random_half(ga, rng), cfg.min_half_obs, cfg.min_half_obs, scope)
            if len(idx) >= need:
                reps.append(_r(ra, rb))
                ns.append(len(idx))
        reps_a = np.asarray([x for x in reps if np.isfinite(x)], float)
        row = {**base, "split": "random", "n_athletes": int(np.median(ns)) if ns else 0, "r_half": None, "rel_sb": None, "rel_lo": None,
               "rel_hi": None, "interval_kind": "split_distribution", "note": ""}
        if len(reps_a) < max(2, cfg.n_random_splits // 2):
            row["note"] = "too few random splits with enough athletes"
        else:
            sb = sf._sb(reps_a)
            row |= {"r_half": float(reps_a.mean()), "rel_sb": _fn(np.nanmean(sb)), "rel_lo": _fn(np.nanpercentile(sb, 2.5)),
                    "rel_hi": _fn(np.nanpercentile(sb, 97.5))}
        rows.append(row)
    return rows


def exposure_rows(d: Denominator, ga: GameArrays, cfg: ValidationConfig) -> list[dict[str, Any]]:
    rows = []
    n_games = np.bincount(ga.ai, minlength=len(ga.ids))
    for n in cfg.first_games_grid:
        idx, ra, rb = pair(ga, np.where(ga.rk < n, 0, 1), 1, cfg.min_half_obs)
        idx_ok = n_games[idx] > n
        rows.append({"denominator_id": d.denominator_id, "mode": "first_n_vs_later", "level": int(n), "n_athletes": int(idx_ok.sum()),
                     "r": _fn(_r(_recentre(ra, idx_ok, ga.pos[idx]), _recentre(rb, idx_ok, ga.pos[idx]))), "rel_sb": None})
    for g in cfg.min_games_grid:
        half = np.where(n_games[ga.ai] >= g, ga.rk % 2, -1)
        idx, ra, rb = pair(ga, half, cfg.min_half_obs, cfg.min_half_obs)
        r = _r(ra, rb)
        rows.append({"denominator_id": d.denominator_id, "mode": "min_games", "level": int(g), "n_athletes": len(idx), "r": _fn(r),
                     "rel_sb": _fn(sf._sb(r)) if np.isfinite(r) else None})
    return rows


def _recentre(x: np.ndarray, keep: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return _centred(x[keep], pos[keep])


# ============================================================================== athlete values
def _percentile(df: pl.DataFrame) -> pl.DataFrame:
    """Mid-rank percentile among SUPPORTED athletes of the same position; null for the others."""
    sup = (df.filter(pl.col("supported"))
           .with_columns((100.0 * (pl.col("value").rank("average") - 0.5) / pl.len()).over("position_group").alias("pct_position"))
           .select("nfl_id", "pct_position"))
    return df.drop("pct_position", strict=False).join(sup, on="nfl_id", how="left")


def athlete_table(gt: pl.DataFrame, d: Denominator, cfg: ValidationConfig, salt: int) -> pl.DataFrame:
    ga = game_arrays(gt)
    k = len(ga.ids)
    if not k:
        return pl.DataFrame(schema=ATHLETE_SCHEMA)
    n, s, g = np.bincount(ga.ai, weights=ga.n, minlength=k), np.bincount(ga.ai, weights=ga.s, minlength=k), np.bincount(ga.ai, minlength=k)
    rng = np.random.default_rng([cfg.seed, salt, 2])
    se, lo, hi = np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan)
    for a in range(k):                                           # game bootstrap: the athlete's games are resampled whole
        if g[a] < 2:
            continue
        sl = slice(ga.start[a], ga.start[a] + g[a])
        bi = rng.integers(0, g[a], (cfg.n_boot_games, g[a]))
        m = ga.s[sl][bi].sum(1) / ga.n[sl][bi].sum(1)
        se[a], (lo[a], hi[a]) = m.std(ddof=1), np.percentile(m, [2.5, 97.5])
    _, ha, hb = half_stats(ga, ga.rk % 2, 1, 1)
    seasons = gt.group_by("nfl_id").agg(pl.col("season").n_unique().alias("n_seasons"), pl.col("season").min().alias("first_season"))
    out = pl.DataFrame({"denominator_id": [d.denominator_id] * k, "outcome_id": [d.outcome_id] * k, "nfl_id": ga.ids.astype(np.int64),
                        "position_group": ga.pos.tolist(), "n_obs": n.astype(np.int64), "n_games": g.astype(np.int64), "value": s / n,
                        "se_game_boot": se, "lo": lo, "hi": hi, "half_a": ha, "half_b": hb,
                        "supported": (n >= cfg.min_obs) & (g >= cfg.min_games)})
    out = out.with_columns([pl.col(c).fill_nan(None) for c in ("se_game_boot", "lo", "hi", "half_a", "half_b")])
    out = _percentile(out.join(seasons, on="nfl_id", how="left"))
    return out.select([pl.col(c).cast(t) for c, t in ATHLETE_SCHEMA.items()]).sort("nfl_id")


def athlete_median(rows: pl.DataFrame, cfg: ValidationConfig) -> pl.DataFrame:
    """Median instead of mean (statistic sensitivity of a continuous outcome): nfl_id, position_group, value, supported, pct_position."""
    t = (rows.group_by("nfl_id", "position_group")
         .agg(pl.len().alias("n_obs"), pl.col("game_id").n_unique().alias("n_games"), pl.col("value").median().alias("value"))
         .with_columns(((pl.col("n_obs") >= cfg.min_obs) & (pl.col("n_games") >= cfg.min_games)).alias("supported")))
    return _percentile(t).select("nfl_id", "position_group", "value", "supported", "pct_position")


def _rank(x: np.ndarray) -> np.ndarray:
    return pl.Series(np.asarray(x, float)).rank("average").to_numpy()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _r(_rank(a), _rank(b)) if len(a) >= 3 else float("nan")


def ranking_table(defs: dict[str, pl.DataFrame], outcome_of: dict[str, str]) -> pl.DataFrame:
    """Rank agreement of every pair of definitions of ONE outcome, on the athletes supported by both. ``defs`` maps a
    definition name to a frame with nfl_id, position_group, supported, pct_position."""
    rows, names = [], list(defs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if outcome_of[a] != outcome_of[b]:
                continue
            j = (defs[a].filter(pl.col("supported")).select("nfl_id", "position_group", pl.col("pct_position").alias("pa"))
                 .join(defs[b].filter(pl.col("supported")).select("nfl_id", pl.col("pct_position").alias("pb")), on="nfl_id", how="inner"))
            for scope in (ALL, *sorted(j["position_group"].unique().to_list())):
                c = j if scope == ALL else j.filter(pl.col("position_group") == scope)
                pa, pb = c["pa"].to_numpy(), c["pb"].to_numpy()
                shift = np.abs(pa - pb)
                rows.append({"outcome_id": outcome_of[a], "definition_a": a, "definition_b": b, "position_group": scope, "n_common": c.height,
                             "spearman": _fn(spearman(pa, pb)), "mean_abs_pct_shift": _fn(shift.mean()) if c.height else None,
                             "max_abs_pct_shift": _fn(shift.max()) if c.height else None})
    return pl.DataFrame(rows, schema=RANK_SCHEMA)


def dependence_table(athletes: pl.DataFrame) -> pl.DataFrame:
    """Does the athlete value follow its own exposure? Spearman within position (centred), supported athletes."""
    rows = []
    for (den,), t in athletes.filter(pl.col("supported")).group_by(["denominator_id"], maintain_order=True):
        pos = np.asarray(t["position_group"].to_list(), dtype=str)
        v = _centred(t["value"].to_numpy(), pos)
        rows.append({"denominator_id": den, "n_athletes": t.height,
                     "rho_value_log_obs": _fn(spearman(v, _centred(np.log(t["n_obs"].to_numpy().astype(float)), pos))),
                     "rho_value_games": _fn(spearman(v, _centred(t["n_games"].to_numpy().astype(float), pos)))})
    return pl.DataFrame(rows, schema=DEP_SCHEMA)


# ============================================================================== denominators: measured support
def denominator_table(rows: dict[str, pl.DataFrame], athletes: pl.DataFrame, dens: Sequence[Denominator] = DENOMINATORS) -> pl.DataFrame:
    out = []
    for d in dens:
        static = {"denominator_id": d.denominator_id, "outcome_id": d.outcome_id, "label": d.label, "formula": d.formula, "unit": d.unit,
                  "statistic": d.statistic, "kind": d.kind, "legitimacy": d.legitimacy, "assumption": d.assumption}
        r = rows.get(d.denominator_id)
        if r is None:
            out.append({**static, "available": False, "position_group": ALL, "n_rows": 0, "n_athletes_any": 0, "n_athletes_supported": 0,
                        "n_athlete_games": 0, "median_obs": None, "median_games": None, "mean_value": None, "targeted_share": None})
            continue
        a = athletes.filter(pl.col("denominator_id") == d.denominator_id)
        for scope in (ALL, *d.positions):
            rs = r if scope == ALL else r.filter(pl.col("position_group") == scope)
            at = a if scope == ALL else a.filter(pl.col("position_group") == scope)
            out.append({**static, "available": True, "position_group": scope, "n_rows": rs.height, "n_athletes_any": at.height,
                        "n_athletes_supported": int(at["supported"].sum()), "n_athlete_games": int(at["n_games"].sum()),
                        "median_obs": _fn(at["n_obs"].median()), "median_games": _fn(at["n_games"].median()),
                        "mean_value": _fn(rs["value"].mean()), "targeted_share": _fn(rs["tgt"].mean())})
    return pl.DataFrame(out, schema=DENOM_SCHEMA)


def season_table(rows: dict[str, pl.DataFrame]) -> pl.DataFrame:
    parts = [r.group_by("season").agg(pl.len().alias("n_rows"), pl.col("nfl_id").n_unique().alias("n_athletes"),
                                      pl.col("game_id").n_unique().alias("n_games"), pl.col("value").mean().alias("mean_value"))
             .with_columns(pl.lit(k).alias("denominator_id")).select(list(SEASON_SCHEMA)) for k, r in rows.items() if r.height]
    return pl.concat(parts).cast(SEASON_SCHEMA).sort("denominator_id", "season") if parts else pl.DataFrame(schema=SEASON_SCHEMA)


def run_opportunity(base: pl.DataFrame, cfg: ValidationConfig, dens: Sequence[Denominator] = DENOMINATORS,
                    progress: Any = None) -> dict[str, pl.DataFrame]:
    """The seven opportunity / reliability tables, keyed by contract name."""
    rows = all_rows(base, cfg, dens)
    ath, rel, expo, defs, outcome_of = [], [], [], {}, {}
    for salt, d in enumerate(dens):
        if d.denominator_id not in rows:
            continue
        gt = game_table(rows[d.denominator_id])
        at = athlete_table(gt, d, cfg, salt)
        ga = game_arrays(gt)
        ath.append(at)
        rel += reliability_rows(d, ga, cfg, salt)
        expo += exposure_rows(d, ga, cfg)
        defs[d.denominator_id], outcome_of[d.denominator_id] = at, d.outcome_id
        if d.kind == "continuous" and d.legitimacy == "recorded" and d.context_adjustment == "none":
            name = f"{d.denominator_id}|median"
            defs[name], outcome_of[name] = athlete_median(rows[d.denominator_id], cfg), d.outcome_id
        if progress:
            progress(f"{d.denominator_id}: {rows[d.denominator_id].height:,} rows · {at.height} athletes · {int(at['supported'].sum())} supported")
    athletes = pl.concat(ath).cast(ATHLETE_SCHEMA) if ath else pl.DataFrame(schema=ATHLETE_SCHEMA)
    assert_no_combine_columns(athletes, "football athlete values")
    return {DENOMS.name: denominator_table(rows, athletes, dens), SEASONS.name: season_table(rows), ATHLETES.name: athletes,
            RELIABILITY.name: pl.DataFrame(rel, schema=REL_SCHEMA), EXPOSURE.name: pl.DataFrame(expo, schema=EXPO_SCHEMA),
            RANKING.name: ranking_table(defs, outcome_of), DEPENDENCE.name: dependence_table(athletes)}


# ============================================================================== linked Combine cohort (flags and counts only)
MANIFEST_FLAGS = {"has_official": "st_official", "has_tracking": "st_tracked", "has_engineered": "st_engineered", "has_fpca": "st_fpca",
                  "has_tcn": "st_tcn"}
STAGES = (("nfl_observed", True, pl.col("nfl_id").is_not_null()),("nfl_supported", True, pl.col("supported")),
          ("combine_linked", True, pl.col("supported") & pl.col("linked")), ("official_tests", True, pl.col("elig_M1")),
          ("valid_tracking", True, pl.col("elig_M1") & pl.col("has_tracking")), ("all_model_families", True, pl.col("all_models")),
          ("repeat_attempts", False, pl.col("elig_M1") & pl.col("has_tracking") & pl.col("has_repeat")),
          ("eligible_M2_engineered", False, pl.col("elig_M2")), ("eligible_M3_speed", False, pl.col("elig_M3")),
          ("eligible_M4_fpca", False, pl.col("elig_M4")), ("eligible_M5_tcn", False, pl.col("elig_M5")))


def link_athletes(athletes: pl.DataFrame, manifest: pl.DataFrame) -> pl.DataFrame:
    """Join the athlete values with the Combine cohort manifest of Milestone 06 — its linkage FLAGS only. An athlete
    absent from the manifest is not Combine-linked. Each denominator keeps its own population."""
    m = (manifest.select(pl.col("nfl_id").cast(pl.Int64), pl.col("draft_year").cast(pl.Int64), pl.lit(True).alias("linked"),
                         *[pl.col(src).cast(pl.Boolean).alias(dst) for dst, src in MANIFEST_FLAGS.items()],
                         (pl.col("n_attempts") >= 2).alias("has_repeat"), (pl.col("n_speed_attempts") > 0).alias("has_speed"))
         .unique(subset="nfl_id", keep="first", maintain_order=True))
    j = (athletes.select("denominator_id", "nfl_id", "position_group", "supported").join(m, on="nfl_id", how="left")
         .with_columns([pl.col(f).fill_null(False) for f in LINK_FLAGS]))
    m1 = pl.col("supported") & pl.col("linked") & pl.col("has_official")
    j = (j.with_columns(m1.alias("elig_M1"))
         .with_columns((pl.col("elig_M1") & pl.col("has_engineered")).alias("elig_M2"), (pl.col("elig_M1") & pl.col("has_speed")).alias("elig_M3"),
                       (pl.col("elig_M1") & pl.col("has_fpca")).alias("elig_M4"), (pl.col("elig_M1") & pl.col("has_tcn")).alias("elig_M5"))
         .with_columns((pl.col("elig_M2") & pl.col("elig_M3") & pl.col("elig_M4") & pl.col("elig_M5") & pl.col("has_tracking")).alias("all_models")))
    return j.select([pl.col(c).cast(t) for c, t in LINKS_SCHEMA.items()]).sort("denominator_id", "nfl_id")


def link_funnel(links: pl.DataFrame) -> pl.DataFrame:
    rows = []
    both = pl.concat([links, links.with_columns(pl.lit(ALL).alias("position_group"))])
    for order, (stage, cumulative, cond) in enumerate(STAGES):
        t = both.group_by("denominator_id", "position_group").agg(cond.sum().alias("n"))
        rows += [{"denominator_id": r["denominator_id"], "position_group": r["position_group"], "stage": stage, "stage_order": order,
                  "cumulative": cumulative, "n_athletes": int(r["n"])} for r in t.iter_rows(named=True)]
    return pl.DataFrame(rows, schema=FUNNEL_SCHEMA).sort("denominator_id", "position_group", "stage_order")


def cohort_support(links: pl.DataFrame) -> pl.DataFrame:
    t = links.filter(pl.col("supported") & pl.col("linked") & pl.col("draft_year").is_not_null())
    if t.is_empty():
        return pl.DataFrame(schema=SUPPORT_SCHEMA)
    return (t.group_by("denominator_id", "position_group", "draft_year")
            .agg(pl.len().alias("n_supported_linked"), pl.col("all_models").sum().alias("n_all_models"))
            .cast(SUPPORT_SCHEMA).sort("denominator_id", "position_group", "draft_year"))


def run_links(athletes: pl.DataFrame, manifest: pl.DataFrame) -> dict[str, pl.DataFrame]:
    links = link_athletes(athletes, manifest)
    return {LINKS.name: links, FUNNEL.name: link_funnel(links), SUPPORT.name: cohort_support(links)}


# ============================================================================== screen, proposal
def screen_table(denoms: pl.DataFrame, semantic: dict[str, dict[str, str]], reliability: pl.DataFrame, dependence: pl.DataFrame,
                 funnel: pl.DataFrame, cfg: ValidationConfig, dens: Sequence[Denominator] = DENOMINATORS) -> pl.DataFrame:
    """Ordered rule on declarations, counts and NFL-side reliability. No Combine value and no Combine association.

    eligible = confirmed semantics · recorded denominator · linked support · reliability at the reference line · low
    exposure dependence. provisional = usable only with an explicit, written acknowledgement of the unmet criteria."""
    for t in (denoms, reliability, dependence):
        assert_no_combine_columns(t, "screen input")
    rel = {(r["denominator_id"], r["split"]): r for r in reliability.filter(pl.col("position_group") == ALL).iter_rows(named=True)}
    dep = {r["denominator_id"]: r for r in dependence.iter_rows(named=True)}
    fun = {(r["denominator_id"], r["position_group"]): r["n_athletes"]
           for r in funnel.filter(pl.col("stage") == "all_model_families").iter_rows(named=True)}
    sup = {r["denominator_id"]: r for r in denoms.filter(pl.col("position_group") == ALL).iter_rows(named=True)}
    rows = []
    for d in dens:
        s, oe, rnd = sup.get(d.denominator_id, {}), rel.get((d.denominator_id, "odd_even"), {}), rel.get((d.denominator_id, "random"), {})
        sem = semantic.get(d.outcome_id, {}).get("status", "unresolved")
        n_link = int(fun.get((d.denominator_id, ALL), 0))
        n_pos = sum(int(fun.get((d.denominator_id, p), 0)) >= cfg.min_linked_position for p in d.positions)
        sb, lo, rho = oe.get("rel_sb"), oe.get("rel_lo"), dep.get(d.denominator_id, {}).get("rho_value_log_obs")
        c_sem, c_den = sem == "confirmed_source", d.legitimacy == "recorded"
        c_sup = n_link >= cfg.min_linked and n_pos >= 1
        c_rel = sb is not None and lo is not None and sb >= cfg.reliability_ref and lo >= cfg.reliability_floor
        c_exp = rho is not None and abs(rho) <= cfg.exposure_ref
        unmet = [name for name, ok in (("vendor definition not confirmed", c_sem), (f"denominator is {d.legitimacy}", c_den),
                                       (f"reliability below {cfg.reliability_ref:.2f} / lower bound below {cfg.reliability_floor:.2f}", c_rel),
                                       (f"|ρ(value, log exposure)| above {cfg.exposure_ref:.2f}", c_exp)) if not ok]
        if not s.get("available", False):
            status, reason = "unavailable", "a needed vendor field is absent: " + ", ".join(d.needs)
        elif d.legitimacy not in FREEZABLE:
            status, reason = "sensitivity_only", f"{d.legitimacy} denominator: {d.assumption}"
        elif sem == "unresolved":
            status, reason = "rejected", "the definition of the outcome is unresolved"
        elif not c_sup:
            status, reason = "rejected", (f"{n_link} linked athletes support every model family (needed {cfg.min_linked}); "
                                          f"{n_pos} position(s) reach {cfg.min_linked_position}")
        elif sb is None or lo is None or sb < cfg.reliability_floor or lo <= 0:
            status, reason = "rejected", "game-split reliability is not assessable, below the floor, or its lower bound is not positive"
        elif not unmet:
            status, reason = "eligible", "every criterion met"
        else:
            status, reason = "provisional", "unmet: " + "; ".join(unmet)
        rows.append({"denominator_id": d.denominator_id, "outcome_id": d.outcome_id, "legitimacy": d.legitimacy, "semantic_status": sem,
                     "unit": d.unit, "available": bool(s.get("available", False)), "n_supported": int(s.get("n_athletes_supported") or 0),
                     "n_linked_all_models": n_link, "n_positions_linked": int(n_pos), "rel_sb": sb, "rel_lo": lo, "rel_hi": oe.get("rel_hi"),
                     "rel_random": rnd.get("rel_sb"), "exposure_rho": rho, "c_semantics": c_sem, "c_denominator": c_den,
                     "c_support": bool(c_sup), "c_reliability": bool(c_rel), "c_exposure": bool(c_exp), "status": status, "reason": reason})
    return pl.DataFrame(rows, schema=SCREEN_SCHEMA)


def propose(screen: pl.DataFrame, priority: Sequence[str] = PRIORITY) -> dict[str, Any]:
    """Rule-based PROPOSAL in the pre-declared priority order — a recommendation, never a freeze."""
    st = {r["denominator_id"]: r for r in screen.iter_rows(named=True)}
    ranked = [k for want in ("eligible", "provisional") for k in priority if st.get(k, {}).get("status") == want]
    if not ranked:
        return {"decision": "STOP", "primary": None, "secondary": [], "basis": "no denominator is eligible or provisional"}
    primary, secondary = ranked[0], []
    for k in ranked[1:]:
        if st[k]["outcome_id"] != st[primary]["outcome_id"] and st[k]["outcome_id"] not in {st[x]["outcome_id"] for x in secondary}:
            secondary.append(k)
    decision = "FREEZE" if st[primary]["status"] == "eligible" else "PROVISIONAL"
    return {"decision": decision, "primary": primary, "secondary": secondary[:2],
            "basis": f"first {st[primary]['status']} denominator in the pre-declared order {list(priority)}; one denominator per other outcome",
            "statuses": {k: st[k]["status"] for k in st}}


# ============================================================================== explicit freeze
def registry_path(store: Any, version: str):
    return store.data_dir / f"{REGISTRY_FILE}_{version}.json"


def target_definition(k: str, role: str, screen_row: dict[str, Any], semantic: dict[str, dict[str, str]], athletes: pl.DataFrame,
                      cfg: ValidationConfig) -> dict[str, Any]:
    d = BY_ID[k]
    sem = osm.BY_ID[d.outcome_id]
    ids = sorted(athletes.filter((pl.col("denominator_id") == k) & pl.col("supported"))["nfl_id"].to_list())
    rule = {"min_obs": cfg.min_obs, "min_games": cfg.min_games, "min_cell_obs": cfg.min_cell_obs, "min_cell_athletes": cfg.min_cell_athletes}
    return {"target_id": k, "outcome_id": d.outcome_id, "role": role, "value_field": sem.value_field, "source_table": sem.source_table,
            "formula": d.formula, "units": d.unit, "kind": d.kind, "athlete_statistic": d.statistic,
            "opportunity_denominator": d.label, "denominator_legitimacy": d.legitimacy, "assumption": d.assumption,
            "context_adjustment": d.context_adjustment, "positions": list(d.positions), "eligibility": sem.eligibility, "support_rule": rule,
            "semantic_status": semantic[d.outcome_id]["status"], "semantic_basis": semantic[d.outcome_id]["basis"],
            "screen_status": screen_row["status"], "unmet_criteria": screen_row["reason"],
            "reliability": {x: screen_row[x] for x in ("rel_sb", "rel_lo", "rel_hi", "rel_random", "exposure_rho")},
            "support": {x: screen_row[x] for x in ("n_supported", "n_linked_all_models", "n_positions_linked")},
            "primary_metric": PRIMARY_METRIC[d.kind], "interpretation_limit": sem.limits,
            "definition_hash": _hash({"denominator": asdict(d), "semantics": asdict(sem), "rule": rule}),
            "cohort_hash": _hash(ids), "n_cohort": len(ids)}


def version_hash(doc: dict[str, Any]) -> str:
    return _hash({"targets": [(t["target_id"], t["role"], t["definition_hash"], t["cohort_hash"]) for t in doc.get("targets", [])],
                  "config_hash": doc.get("config_hash"), "version": doc.get("version")})


def freeze_football_targets(store: Any, decision: str | None, primary: str | None, secondary: Sequence[str], rationale: str,
                            acknowledge_provisional: bool, screen: pl.DataFrame, proposal: dict[str, Any], athletes: pl.DataFrame,
                            semantic: dict[str, dict[str, str]], cfg: ValidationConfig, provenance: dict[str, Any],
                            version: str = "v1") -> tuple[dict[str, Any] | None, str]:
    """Record the analyst's decision. Only ``FREEZE`` writes ``football_outcome_targets_<version>.json``, once: an
    existing registry is never overwritten (another content needs another version). ``PROVISIONAL`` and ``STOP`` are
    validated and reported but write no registry, so notebook 08B stays blocked.

    A target must be ``eligible`` in the screen, or ``provisional`` with ``acknowledge_provisional = True`` and a
    written rationale. ``rejected``, ``sensitivity_only`` and ``unavailable`` denominators are refused."""
    assert_no_combine_columns(screen, "freeze screen")
    path = registry_path(store, version)
    if decision is None:
        return (read_json(path, None) if path.is_file() else None), "PENDING REVIEW — set DECISION to 'FREEZE', 'PROVISIONAL' or 'STOP' and rerun this cell"
    if decision not in DECISIONS:
        raise ValueError(f"DECISION must be one of {DECISIONS} or None, got {decision!r}")
    if not str(rationale).strip():
        raise ValueError("a decision needs a written RATIONALE")
    if decision == "STOP":
        return None, "STOP recorded: no football outcome is accepted; notebook 08B must not run"
    secondary = list(secondary or [])
    if primary is None or len(secondary) > 2 or len({primary, *secondary}) != 1 + len(secondary):
        raise ValueError(f"a {decision} needs one PRIMARY and at most two distinct SECONDARY denominators")
    st = {r["denominator_id"]: r for r in screen.iter_rows(named=True)}
    for k in (primary, *secondary):
        status = st.get(k, {}).get("status")
        if status not in ("eligible", "provisional"):
            raise ValueError(f"{k!r} cannot be a target (screen status {status!r}): {st.get(k, {}).get('reason', 'unknown denominator')}")
        if decision == "FREEZE" and status == "provisional" and not acknowledge_provisional:
            raise ValueError(f"{k!r} is provisional ({st[k]['reason']}). Freezing it needs ACKNOWLEDGE_PROVISIONAL = True and a RATIONALE "
                             "that addresses each unmet criterion; otherwise record 'PROVISIONAL' or 'STOP'")
    if len({st[k]["outcome_id"] for k in (primary, *secondary)}) != 1 + len(secondary):
        raise ValueError("one denominator per outcome: two targets share an outcome")
    if decision == "PROVISIONAL":
        return None, "PROVISIONAL recorded: definitions kept for review, NOT frozen; notebook 08B must not run"
    targets = [target_definition(primary, "primary", st[primary], semantic, athletes, cfg),
               *[target_definition(k, "secondary", st[k], semantic, athletes, cfg) for k in secondary]]
    doc = {"artifact": REGISTRY_FILE, "version": version, "decision": "FREEZE", "frozen_at": utc_now(), "targets": targets,
           "config": cfg.params(), "config_hash": _hash(cfg.params()), "denominators_hash": denominators_hash(),
           "semantics_hash": osm.definition_hash(),
           "selection": {"rule_based_proposal": proposal, "rationale": rationale, "acknowledge_provisional": bool(acknowledge_provisional),
                         "follows_proposal": primary == proposal.get("primary") and secondary == list(proposal.get("secondary", [])),
                         "combine_values_consulted": False, "combine_link_counts_consulted": True, "screen": screen.to_dicts()},
           "evidence_family": "NEW exploratory family (Milestone 08); not a confirmation of the Milestone 06 hypothesis",
           "provenance": provenance, "run_mode": store.controls.run_mode, "seed": cfg.seed}
    doc["version_hash"] = version_hash(doc)
    if path.is_file():
        old = read_json(path, {})
        if old.get("version_hash") == doc["version_hash"]:
            return old, f"already frozen on {old.get('frozen_at')}; the current run reproduces it"
        raise FreezeConflictError(f"{path.name} exists with another content (frozen {old.get('frozen_at')}, hash {old.get('version_hash')} ≠ "
                                  f"{doc['version_hash']}). A frozen registry is immutable: set REGISTRY_VERSION to a new version.")
    if not store.controls.save_tables:
        return doc, "NOT WRITTEN (SAVE_TABLES is off)"
    store.data_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(doc, path)
    log.info("[football_outcomes] %s written: %s", path.name, [t["target_id"] for t in targets])
    return doc, "frozen now"


def gate_08a2(doc: dict[str, Any] | None, decision: str | None) -> tuple[str, str]:
    """(status, sentence) of gate 08A.2. PASS only for a written, approved freeze."""
    if doc and doc.get("decision") == "FREEZE" and doc.get("targets"):
        return "PASS", "frozen football-outcome targets: " + ", ".join(f"{t['target_id']} ({t['role']})" for t in doc["targets"])
    if decision == "STOP":
        return "STOP", "no football outcome accepted: notebook 08B does not run; Milestone 08 continues with 08C / 08D"
    if decision == "PROVISIONAL":
        return "PROVISIONAL", "definitions recorded for review, not frozen: notebook 08B stays blocked"
    return "PENDING", "no decision recorded yet"


def require_frozen(store: Any, version: str = "v1") -> dict[str, Any]:
    """The gate of notebook 08B: the registry exists, was approved (FREEZE), belongs to this run mode and is intact."""
    path = registry_path(store, version)
    if not path.is_file():
        raise FootballTargetFreezeError(f"{path.name} is missing for RUN_MODE = {store.controls.run_mode!r}: record DECISION = 'FREEZE' in "
                                        "notebook 08a2 (section 10) first. Transfer models do not run on a proposal.")
    doc = read_json(path, {})
    problems = []
    if doc.get("decision") != "FREEZE":
        problems.append(f"decision is {doc.get('decision')!r}")
    if doc.get("run_mode") != store.controls.run_mode:
        problems.append(f"frozen in run mode {doc.get('run_mode')!r}")
    if sum(t.get("role") == "primary" for t in doc.get("targets", [])) != 1:
        problems.append("not exactly one primary target")
    if doc.get("version_hash") != version_hash(doc):
        problems.append("the registry was edited after the freeze (hash mismatch)")
    if problems:
        raise FootballTargetFreezeError(f"{path.name} is not a valid freeze: " + "; ".join(problems))
    return doc


def conclusions_markdown(candidates: pl.DataFrame, screen: pl.DataFrame, proposal: dict[str, Any], gate: tuple[str, str],
                         cfg: ValidationConfig) -> str:
    def f(x: Any) -> str:
        return "—" if x is None else f"{x:.2f}"

    lines = ["### 08A.2 — outcome semantics and opportunity validation", "", f"Gate 08A.2: **{gate[0]}** — {gate[1]}", "",
             "| Outcome | Definition status | Basis |", "|---|---|---|"]
    lines += [f"| `{r['outcome_id']}` | {r['definition_status']} | {r['status_basis']} |" for r in candidates.iter_rows(named=True)]
    lines += ["", "| Denominator | Legitimacy | Linked (all models) | Odd / even reliability [95 %] | Random halves | ρ exposure | Status | Reason |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| `{r['denominator_id']}` | {r['legitimacy']} | {r['n_linked_all_models']} | {f(r['rel_sb'])} [{f(r['rel_lo'])}, {f(r['rel_hi'])}] | "
              f"{f(r['rel_random'])} | {f(r['exposure_rho'])} | {r['status']} | {r['reason']} |" for r in screen.iter_rows(named=True)]
    lines += ["", f"Rule-based proposal (a recommendation, not a freeze): **{proposal['decision']}** · primary `{proposal['primary']}` · "
                  f"secondary {proposal['secondary'] or 'none'} — {proposal['basis']}.", "",
              f"Reference lines (heuristics declared before running): ≥ {cfg.min_obs} rows in ≥ {cfg.min_games} games per athlete; "
              f"≥ {cfg.min_linked} linked athletes supporting every model family; odd / even Spearman–Brown ≥ {cfg.reliability_ref:.2f} with a "
              f"lower bound ≥ {cfg.reliability_floor:.2f}; |ρ(value, log exposure)| ≤ {cfg.exposure_ref:.2f}.", "",
              "Limits: no vendor data dictionary in the repository (a definition is confirmed only by evidence the analyst records); the "
              "delivered tracking holds the project athletes only, so nearest-defender distance cannot be recomputed; assignment, opponent "
              "and scheme are unobserved; no Combine measurement was read and no outcome was related to one."]
    return "\n".join(lines)


__all__ = ["ALL", "ATHLETES", "BY_ID", "DECISIONS", "DENOMINATORS", "DENOMS", "DEPENDENCE", "Denominator", "EXPOSURE", "FUNNEL",
           "FootballTargetFreezeError", "FreezeConflictError", "GameArrays", "LINKS", "LINK_CONTRACTS", "OPPORTUNITY_CONTRACTS", "PRIORITY",
           "PROPOSAL_FILE", "RANKING", "REGISTRY_FILE", "RELIABILITY", "SCREEN", "SEASONS", "SUPPORT", "ValidationConfig", "all_rows",
           "athlete_median", "athlete_table", "available", "cohort_support", "conclusions_markdown", "denominator_rows",
           "denominator_table", "denominators_hash", "dependence_table", "exposure_rows", "freeze_football_targets", "game_arrays",
           "game_table", "gate_08a2", "half_stats", "link_athletes", "link_funnel", "pair", "propose", "random_half", "ranking_table",
           "registry_path", "reliability_rows", "require_frozen", "run_links", "run_opportunity", "screen_table",
           "season_table", "spearman", "target_definition", "version_hash"]
