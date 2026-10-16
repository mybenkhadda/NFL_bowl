"""Stage DEF — what is ``player_get_off``, and where is it observed? (the first gate of the experiment)

Definition      The repository holds no vendor data dictionary; ``schema_catalog.csv`` lists the column with
                evidence = "inferred" (a guess from the column name). The definition is therefore read from
                ``configs/player_get_off_definition.json``, which the user fills from the competition data
                dictionary. Status ``confirmed`` needs a source, a timing origin, a timing end and the direction.
                Anything else is ``unresolved``: the audit below is still produced, the models run only on an
                explicit flag, and the report replaces the football interpretation with a blocker.
Audit           descriptive only. No Combine value is used. It reports where the value exists (season, alignment,
                play type, blitz label, rush-role proxies), its distribution, boundary values (the observed
                maximum and very small values), its numeric resolution, the EDGE − INTERIOR difference within
                athletes who played both, and how it co-varies with sacks and pressures.
                The last block is EVIDENCE ABOUT DIRECTION, not a definition: it cannot tell what the clock measures.
Valid plays     regular season, defensive-line athletes, endpoint present, alignment EDGE or INTERIOR_LINE, play
                not nullified by penalty. One row per player-play, ordered in time within the athlete.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ...data.utils import read_json
from ..contracts import TableContract
from ..sprint_rescue.stats import bootstrap_ci, partial_corr, seed_for
from .config import GetOffConfig

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
DEFINITION_FILE = "player_get_off_definition.json"
REQUIRED = ("source", "timing_origin", "timing_end", "lower_is_better")
PLAYER_FIELDS = ("lined_up_position", "blitzing", "sack", "unblocked_pressure", "tackle", "time_to_pressure",
                 "coverage_assignment")
PLAY_FIELDS = ("down", "yards_to_go", "offense_formation", "pass_result", "quarter", "team_coverage_man_zone",
               "play_nullified_by_penalty")
STRATA_FIELDS = ("lined_up_position", "pass_result", "blitzing", "down", "offense_formation")
NULL = "<null>"

AUDIT = TableContract(
    "endpoint_audit", "Descriptive audit of player_get_off among defensive-line athletes (no Combine value used).",
    {"section": S, "stratum": S, "metric": S, "value": F, "n": I, "note": S},
    key=("section", "stratum", "metric"), area="section × stratum × metric",
    docs={"section": "definition · availability · distribution · boundary · alignment · direction_evidence · validity",
          "n": "denominator of the metric (player-plays or athletes, see note)"})
HIST = TableContract(
    "endpoint_distribution", "Histogram of valid player_get_off observations by alignment.",
    {"alignment": S, "bin_lower": F, "bin_upper": F, "n": I}, key=("alignment", "bin_lower"),
    area="alignment × 0.05 s bin")
PLAYS = TableContract(
    "dl_getoff_plays", "Valid player_get_off observations of defensive-line athletes (repeated-play data).",
    {"game_id": I, "play_id": I, "nfl_id": I, "season": I, "draft_year": I, "alignment": S, "value": F,
     "obs_idx": I, "game_idx": I, "at_max": B, "below_low": B, "blitzing": S, "pass_result": S, "down": I},
    key=("game_id", "play_id", "nfl_id"), kind="dataset", area="player-play (game_id, play_id, nfl_id)",
    docs={"value": "player_get_off as delivered [s]", "obs_idx": "1 = the athlete's first valid observation in time",
          "game_idx": "1 = the athlete's first game with a valid observation",
          "at_max": "within boundary_tol_s of the largest value observed", "below_low": "value < low_value_s"})


# ============================================================================== definition
def load_definition(cfg: Any) -> dict[str, Any]:
    d = read_json(cfg.paths.project_root / "configs" / DEFINITION_FILE, {}) or {}
    return {k: v for k, v in d.items() if not k.startswith("_")}


def definition_status(d: dict[str, Any]) -> tuple[bool, list[str]]:
    """(confirmed, missing fields). Confirmed needs status == 'confirmed' AND every required field filled."""
    missing = [k for k in REQUIRED if d.get(k) is None or d.get(k) == ""]
    return (d.get("status") == "confirmed" and not missing), missing


def blocker_text(d: dict[str, Any]) -> str:
    ok, missing = definition_status(d)
    if ok:
        return ""
    return ("BLOCKER — the vendor definition of `player_get_off` is unresolved (status = "
            f"{d.get('status', 'file missing')!r}; missing: {', '.join(missing) or 'status not confirmed'}). "
            f"Fill `configs/{DEFINITION_FILE}` from the competition data dictionary. Until then the endpoint is "
            "reported as an opaque vendor number: no direction ('faster', 'better'), no physical meaning and no "
            "coaching conclusion may be attached to any estimate.")


# ============================================================================== valid plays
def _lev(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.String).fill_null(NULL)


def dl_population(base: pl.DataFrame, gcfg: GetOffConfig) -> pl.DataFrame:
    """Tracked regular-season player-plays of the position group, endpoint as Float64."""
    return (base.filter(pl.col("season_type").is_in(list(gcfg.season_types)) & (pl.col("position_group") == gcfg.position))
            .with_columns(pl.col(gcfg.endpoint).cast(pl.Float64, strict=False).alias("value"),
                          _lev("play_nullified_by_penalty").str.to_uppercase().is_in(["Y", "TRUE"]).alias("nullified")))


def valid_plays(pop: pl.DataFrame, gcfg: GetOffConfig) -> pl.DataFrame:
    v = pop.filter(pl.col("value").is_not_null() & pl.col("lined_up_position").is_in(list(gcfg.alignments))
                   & ~pl.col("nullified"))
    if not v.height:
        return PLAYS.empty()
    top = float(v["value"].max())
    v = v.sort("nfl_id", "game_id", "play_id").with_columns(
        pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("nfl_id").alias("obs_idx"),
        pl.col("game_id").rank("dense").over("nfl_id").cast(pl.Int64).alias("game_idx"),
        (pl.col("value") >= top - gcfg.boundary_tol_s).alias("at_max"),
        (pl.col("value") < gcfg.low_value_s).alias("below_low"),
        pl.col("lined_up_position").cast(pl.String).alias("alignment"), pl.col("cohort").alias("draft_year"),
        _lev("blitzing").alias("blitzing"), _lev("pass_result").alias("pass_result"),
        pl.col("down").cast(pl.Float64, strict=False).cast(pl.Int64).alias("down"))
    return v.select([pl.col(c).cast(t) for c, t in PLAYS.columns.items()])


# ============================================================================== audit
def _dist(x: np.ndarray) -> list[tuple[str, float]]:
    if not x.size:
        return []
    q = np.quantile(x, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return [("mean", float(x.mean())), ("sd", float(x.std(ddof=1)) if x.size > 1 else float("nan")),
            ("min", float(x.min())), *zip(("p01", "p05", "p25", "p50", "p75", "p95", "p99"), map(float, q)),
            ("max", float(x.max()))]


def endpoint_audit(pop: pl.DataFrame, plays: pl.DataFrame, definition: dict[str, Any], catalog: dict[str, str],
                   gcfg: GetOffConfig) -> pl.DataFrame:
    """``pop``: :func:`dl_population`. ``plays``: :func:`valid_plays`. ``catalog``: schema_catalog row of the column."""
    rows: list[dict[str, Any]] = []

    def add(section: str, stratum: str, metric: str, value: Any, n: int | None = None, note: str = "") -> None:
        ok = value is not None and np.isfinite(float(value))
        rows.append({"section": section, "stratum": stratum, "metric": metric, "value": float(value) if ok else None,
                     "n": None if n is None else int(n), "note": note})

    # ---- definition
    confirmed, missing = definition_status(definition)
    add("definition", "all", "vendor_definition_confirmed", 1.0 if confirmed else 0.0, note=(
        f"configs/{DEFINITION_FILE}: status = {definition.get('status', 'file missing')!r}; schema_catalog: meaning = "
        f"{catalog.get('meaning', 'not listed')!r}, unit = {catalog.get('unit', '')!r}, evidence = "
        f"{catalog.get('evidence', 'not listed')!r}"))
    for k in (*REQUIRED, "valid_play_contexts", "censoring_or_cap"):
        v = definition.get(k)
        add("definition", "field", k, 0.0 if v is None or v == "" else 1.0, note="unknown" if v is None or v == "" else str(v))

    # ---- availability (player-plays of the position group)
    def avail(stratum: str, g: pl.DataFrame) -> None:
        n, k = g.height, g.filter(pl.col("value").is_not_null()).height
        add("availability", stratum, "n_non_null", k, n, "player-plays")
        add("availability", stratum, "share_non_null", k / n if n else None, n, "player-plays")

    avail("all", pop)
    for season in sorted(pop["season"].drop_nulls().unique().to_list()):
        avail(f"season:{season}", pop.filter(pl.col("season") == season))
    for field in STRATA_FIELDS:
        if field not in pop.columns:
            continue
        lev = pop.with_columns(_lev(field).alias("_l"))
        for name in lev.group_by("_l").len().sort(["len", "_l"], descending=[True, False])["_l"].head(8).to_list():
            avail(f"{field}:{name}", lev.filter(pl.col("_l") == name))
    for field in ("sack", "coverage_assignment", "time_to_pressure"):         # rush / coverage role proxies
        if field in pop.columns:
            avail(f"{field}:recorded", pop.filter(pl.col(field).is_not_null()))
            avail(f"{field}:not_recorded", pop.filter(pl.col(field).is_null()))
    avail("nullified_by_penalty", pop.filter(pl.col("nullified")))

    # ---- validity funnel (player-plays)
    non_null = pop.filter(pl.col("value").is_not_null())
    add("validity", "all", "n_endpoint_non_null", non_null.height, pop.height, "player-plays")
    add("validity", "all", "n_other_alignment", non_null.filter(~pl.col("lined_up_position").is_in(list(gcfg.alignments))
                                                                | pl.col("lined_up_position").is_null()).height,
        non_null.height, "endpoint present, alignment not EDGE / INTERIOR_LINE (excluded)")
    add("validity", "all", "n_nullified", non_null.filter(pl.col("nullified")).height, non_null.height,
        "endpoint present, play nullified by penalty (excluded)")
    add("validity", "all", "n_valid", plays.height, non_null.height, "valid observations kept")
    add("validity", "all", "n_athletes_valid", plays["nfl_id"].n_unique() if plays.height else 0, None, "athletes")

    # ---- distribution of valid observations
    def dist(stratum: str, g: pl.DataFrame) -> None:
        x = g["value"].to_numpy().astype(float)
        for name, val in _dist(x):
            add("distribution", stratum, name, val, x.size, "valid player-plays")

    dist("all", plays)
    for col in ("alignment", "season", "pass_result", "blitzing", "down"):
        for name in sorted(plays[col].drop_nulls().unique().to_list()) if plays.height else []:
            g = plays.filter(pl.col(col) == name)
            if g.height >= 30:
                dist(f"{col}:{name}", g)

    # ---- boundary values and resolution
    x = plays["value"].to_numpy().astype(float) if plays.height else np.empty(0)
    if x.size:
        add("boundary", "all", "max_value", float(x.max()), x.size, "largest value observed [s]")
        add("boundary", "all", "share_at_max", float(plays["at_max"].mean()), x.size,
            f"within {gcfg.boundary_tol_s:g} s of the maximum; a spike here means a cap or a censoring window")
        add("boundary", "all", "share_top_5pct_of_range", float((x >= 0.95 * x.max()).mean()), x.size, "value ≥ 0.95 · max")
        add("boundary", "all", "share_below_low", float(plays["below_low"].mean()), x.size, f"value < {gcfg.low_value_s:g} s")
        add("boundary", "all", "n_distinct_values", int(np.unique(np.round(x, 6)).size), x.size, "")
        for dec in (1, 2, 3):
            add("boundary", "all", f"share_on_{dec}_decimal_grid", float((np.abs(x - np.round(x, dec)) < 1e-6).mean()),
                x.size, "numeric resolution of the delivered value")
        for a in gcfg.alignments:
            g = plays.filter(pl.col("alignment") == a)
            if g.height:
                add("boundary", f"alignment:{a}", "share_at_max", float(g["at_max"].mean()), g.height, "")
                add("boundary", f"alignment:{a}", "share_below_low", float(g["below_low"].mean()), g.height, "")

    # ---- alignment: athletes' role mix and the within-athlete EDGE − INTERIOR difference
    if plays.height and len(gcfg.alignments) == 2:
        a0, a1 = gcfg.alignments
        per = plays.group_by("nfl_id").agg(
            pl.len().alias("n"), (pl.col("alignment") == a0).mean().alias("share0"),
            pl.col("value").filter(pl.col("alignment") == a0).mean().alias("m0"),
            pl.col("value").filter(pl.col("alignment") == a1).mean().alias("m1"),
            (pl.col("alignment") == a0).sum().alias("n0"), (pl.col("alignment") == a1).sum().alias("n1"))
        add("alignment", "athletes", f"n_mostly_{a0}", per.filter(pl.col("share0") >= 0.9).height, per.height, "≥ 90 % of observations")
        add("alignment", "athletes", f"n_mostly_{a1}", per.filter(pl.col("share0") <= 0.1).height, per.height, "≥ 90 % of observations")
        add("alignment", "athletes", "n_mixed", per.filter((pl.col("share0") > 0.1) & (pl.col("share0") < 0.9)).height, per.height, "")
        both = per.filter((pl.col("n0") >= 10) & (pl.col("n1") >= 10))
        d = (both["m0"] - both["m1"]).to_numpy().astype(float)
        if d.size >= 5:
            lo, hi = bootstrap_ci(lambda i: float(d[i].mean()), d.size, gcfg.n_boot, seed_for(gcfg.seed, "DEF", "align"))
            note = f"{a0} − {a1}, athletes with ≥ 10 observations in each alignment; athlete bootstrap"
            add("alignment", "within_athlete", "mean_difference_s", float(d.mean()), d.size, note)
            add("alignment", "within_athlete", "mean_difference_ci_lower", lo, d.size, note)
            add("alignment", "within_athlete", "mean_difference_ci_upper", hi, d.size, note)

    # ---- direction evidence (NOT a definition): co-variation with rush outcomes
    v = pop.filter(pl.col("value").is_not_null() & pl.col("lined_up_position").is_in(list(gcfg.alignments)) & ~pl.col("nullified"))
    for field, label in (("sack", "sack > 0"), ("time_to_pressure", "pressure timed"), ("unblocked_pressure", "unblocked pressure")):
        if field not in v.columns or not v.height:
            continue
        col = v[field]
        hit = (col.cast(pl.Float64, strict=False) > 0) if field == "sack" else (
            col.is_not_null() if field == "time_to_pressure" else col.cast(pl.String).str.to_uppercase() == "TRUE")
        g = v.with_columns(hit.fill_null(False).alias("_hit"))
        yes, no = g.filter(pl.col("_hit"))["value"], g.filter(~pl.col("_hit"))["value"]
        add("direction_evidence", field, "mean_value_when_true", yes.mean(), yes.len(), f"plays with {label}")
        add("direction_evidence", field, "mean_value_when_false", no.mean(), no.len(), f"plays without {label}")
        ath = g.group_by("nfl_id").agg(pl.len().alias("n"), pl.col("value").mean().alias("m"), pl.col("_hit").mean().alias("rate"),
                                       (pl.col("lined_up_position") == gcfg.alignments[0]).mean().alias("share0"))
        ath = ath.filter(pl.col("n") >= gcfg.primary_min_obs)
        if ath.height >= 10:
            r = partial_corr(ath["m"].to_numpy(), ath["rate"].to_numpy(), continuous=[ath["share0"].to_numpy()])
            note = (f"athlete mean get-off vs athlete rate of '{label}', given the athlete's alignment share; athletes with "
                    f"≥ {gcfg.primary_min_obs} observations; Fisher interval. Evidence about direction only")
            add("direction_evidence", field, "athlete_partial_spearman", r["r"], r["n"], note)
            add("direction_evidence", field, "athlete_partial_spearman_ci_lower", r["ci_lower"], r["n"], note)
            add("direction_evidence", field, "athlete_partial_spearman_ci_upper", r["ci_upper"], r["n"], note)
    return pl.DataFrame(rows, schema=AUDIT.columns) if rows else AUDIT.empty()


def endpoint_histogram(plays: pl.DataFrame, width: float = 0.05) -> pl.DataFrame:
    if not plays.height:
        return HIST.empty()
    h = (plays.with_columns((pl.col("value") / width).floor().cast(pl.Int64).alias("_b"))
         .group_by("alignment", "_b").agg(pl.len().cast(pl.Int64).alias("n"))
         .with_columns((pl.col("_b") * width).alias("bin_lower"), ((pl.col("_b") + 1) * width).alias("bin_upper")))
    return h.select([pl.col(c).cast(t) for c, t in HIST.columns.items()]).sort("alignment", "bin_lower")
