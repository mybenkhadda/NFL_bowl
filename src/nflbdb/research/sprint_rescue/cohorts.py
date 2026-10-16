"""Studies E and F — who is in the data, and can the next experiment's NFL endpoint support it?

Study E   One table reconciling every athlete count quoted in earlier reports with the counts this experiment
          uses. Athletes without regular-season tracking are listed by the season types they DO have (PRE, POST,
          PRE+POST); nothing is assumed about them.
Study F   Feasibility of an NFL endpoint (primary: defensive-line ``player_get_off``; secondary: receiver
          ``separation_at_pass_forward``): definition status, missingness, volume per athlete, overlap with the
          Combine measurements, play-context availability, repeatability over plays and games, and first-N-play
          eligibility. It is a census of the ENDPOINT. No Combine value is related to any NFL value here, and
          no model is fitted.

Vendor definition: the repository holds no vendor data dictionary. ``schema_catalog.csv`` lists both endpoints
with evidence = "inferred" (meaning guessed from the column name). The definition row therefore reports
``vendor_definition_established = 0`` unless the catalog says "confirmed"; the timing convention of
``player_get_off`` (start event, end event, sign) is NOT known and must not be invented.

Repeatability: ICC(1) of single plays with athletes as groups (athlete bootstrap). The reliability of an athlete
mean over N plays follows from Spearman–Brown, which gives the plays needed for a target reliability. It assumes
exchangeable plays; the context-centred variant removes alignment × season means first.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from ..studies.stats import spearman
from .config import SCENARIOS, RescueConfig
from .stats import observations_needed, seed_for, spearman_brown, stratified_icc

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
RECON = TableContract(
    "athlete_cohort_reconciliation", "Athlete counts from Combine results to the cohorts used by this experiment.",
    {"stage": S, "definition": S, "n_athletes": I, "expected": I, "matches_expected": B, "source": S},
    key=("stage",), area="cohort definition",
    docs={"expected": "count quoted in earlier project reports, when one exists"})
ENDPOINT = TableContract(
    "dl_getoff_feasibility", "Feasibility census of NFL endpoints (primary DL get-off, secondary WR separation).",
    {"endpoint": S, "position_group": S, "section": S, "stratum": S, "metric": S, "value": F, "n": I, "note": S},
    key=("endpoint", "section", "stratum", "metric"), area="endpoint × section × stratum × metric",
    docs={"section": "definition · missingness · distribution · athletes · intersection · context · reliability · "
                     "first_n", "n": "denominator of the metric (player-plays or athletes, see note)"})


# ============================================================================== Study E
def cohort_reconciliation(combine_ids: Sequence[int], nfl: pl.DataFrame, att: pl.DataFrame, official: pl.DataFrame,
                          rcfg: RescueConfig) -> pl.DataFrame:
    """``nfl``: (nfl_id, season_type, n_plays), one row per athlete × season type with tracked plays.
    ``official``: (nfl_id, cmb_forty, cmb_ten_yd_split), one row per athlete."""
    expected = dict(rcfg.expected_counts)
    rows: list[dict[str, Any]] = []

    def add(stage: str, definition: str, ids: Any, source: str, key: str | None = None) -> None:
        n = len(set(ids))
        exp = expected.get(key) if key else None
        rows.append({"stage": f"{len(rows) + 1:02d}_{stage}", "definition": definition, "n_athletes": n,
                     "expected": exp, "matches_expected": (n == exp) if exp is not None else None, "source": source})

    combine = set(combine_ids)
    types = nfl.group_by("nfl_id").agg(pl.col("season_type").unique().sort().str.join("+").alias("types"))
    reg_plays = dict(nfl.filter(pl.col("season_type").is_in(list(rcfg.season_types)))
                     .group_by("nfl_id").agg(pl.col("n_plays").sum()).iter_rows())
    any_nfl = set(types["nfl_id"].to_list())
    add("combine_results", "athletes with a Combine results row", combine, "silver.combine_results", "combine_results")
    add("nfl_tracking_any", "… with ≥ 1 tracked NFL play in any season type", combine & any_nfl,
        "gold.fact_nfl_play_participation", "nfl_tracking_any")
    add("nfl_tracking_reg", "… with ≥ 1 tracked regular-season play", combine & set(reg_plays),
        "gold.fact_nfl_play_participation × silver.games", "nfl_tracking_reg")
    no_reg = types.filter(pl.col("nfl_id").is_in(list(combine)) & ~pl.col("nfl_id").is_in(list(reg_plays)))
    for t in sorted(no_reg["types"].unique().to_list()):
        add(f"nfl_no_reg:{t}", f"NFL tracking but no regular-season play; season types present: {t}",
            no_reg.filter(pl.col("types") == t)["nfl_id"].to_list(), "silver.games.season_type")
    for n in rcfg.nfl_min_plays_grid:
        add(f"nfl_reg_ge_{n}", f"≥ {n} tracked regular-season plays", {i for i, k in reg_plays.items() if k >= n} & combine,
            "gold.fact_nfl_play_participation")
    tracked = set(att["nfl_id"].to_list())
    repeat = set(att.filter(pl.col("n_attempts") >= rcfg.min_attempts)["nfl_id"].to_list())
    forty = set(official.filter(pl.col("cmb_forty").is_not_null())["nfl_id"].to_list())
    ten = set(official.filter(pl.col("cmb_ten_yd_split").is_not_null())["nfl_id"].to_list())
    add("forty_tracked", "≥ 1 tracked forty attempt (usable tier)", tracked, "phase_03 combine_sprint_profiles")
    add("forty_repeat", f"≥ {rcfg.min_attempts} tracked forty attempts", repeat, "phase_03 combine_sprint_profiles")
    add("official_forty", "official forty time present", forty & combine, "gold.mart_athlete_research_base")
    add("official_ten_split", "official 10-yard split present", ten & combine, "gold.mart_athlete_research_base")
    add("official_both", "official forty AND 10-yard split present", forty & ten & combine, "gold.mart_athlete_research_base")
    add("forty_tracked_and_official_both", "tracked forty AND both official times", tracked & forty & ten, "join")
    for sc in SCENARIOS:
        a = att.filter(pl.col(sc))
        k = a.group_by("nfl_id").len()
        add(f"scenario_{sc}_any", f"≥ 1 forty attempt retained by {sc}", a["nfl_id"].to_list(), "phase_04 Q flags")
        add(f"scenario_{sc}_repeat", f"≥ {rcfg.min_attempts} forty attempts retained by {sc}",
            k.filter(pl.col("len") >= rcfg.min_attempts)["nfl_id"].to_list(), "phase_04 Q flags")
    fit = att.filter(pl.col("fit_ok").fill_null(False)).group_by("nfl_id").len()
    add("model_fit_ok_any", "≥ 1 forty attempt with a usable sprint-model fit", fit["nfl_id"].to_list(), "phase_03 fit_ok")
    add("model_fit_ok_repeat", f"≥ {rcfg.min_attempts} forty attempts with a usable fit",
        fit.filter(pl.col("len") >= rcfg.min_attempts)["nfl_id"].to_list(), "phase_03 fit_ok")
    for n in rcfg.nfl_min_plays_grid:
        enough = {i for i, k in reg_plays.items() if k >= n}
        add(f"forty_repeat_and_reg_ge_{n}", f"repeated forty AND ≥ {n} regular-season plays", repeat & enough, "join")
        add(f"forty_official_both_and_reg_ge_{n}", f"tracked forty, both official times AND ≥ {n} regular-season plays",
            tracked & forty & ten & enough, "join")
    return pl.DataFrame(rows, schema=RECON.columns)


def combine_flags(att: pl.DataFrame, base: pl.DataFrame, rcfg: RescueConfig) -> pl.DataFrame:
    """One row per athlete of ``base`` with boolean Combine availability flags (participation only, no values)."""
    k = att.group_by("nfl_id").agg(pl.len().alias("_n"), pl.col("fit_ok").fill_null(False).sum().alias("_fit"))
    cols = [c for c in (*rcfg.official, *rcfg.traditional) if c in base.columns]
    return (base.select("nfl_id", *[pl.col(c).is_not_null().alias(f"has_{c}") for c in cols])
            .join(k, on="nfl_id", how="left")
            .with_columns((pl.col("_n").fill_null(0) >= 1).alias("forty_tracked"),
                          (pl.col("_n").fill_null(0) >= rcfg.min_attempts).alias("forty_repeat"),
                          (pl.col("_fit").fill_null(0) >= rcfg.min_attempts).alias("model_fit_ok_repeat"))
            .drop("_n", "_fit"))


# ============================================================================== Study F
def endpoint_feasibility(pop: pl.DataFrame, endpoint: str, position: str, flags: pl.DataFrame,
                         definition: dict[str, str], rcfg: RescueConfig) -> list[dict[str, Any]]:
    """``pop``: tracked player-plays of the analysis population (game_id, play_id, nfl_id, season, cohort,
    position_group, the endpoint and the context fields). ``flags``: :func:`combine_flags`."""
    rows: list[dict[str, Any]] = []

    def add(section: str, stratum: str, metric: str, value: float | None, n: int | None = None, note: str = "") -> None:
        ok = value is not None and np.isfinite(value)
        rows.append({"endpoint": endpoint, "position_group": position, "section": section, "stratum": stratum,
                     "metric": metric, "value": float(value) if ok else None, "n": None if n is None else int(n),
                     "note": note})

    established = definition.get("evidence", "") == "confirmed"
    add("definition", "all", "vendor_definition_established", 1.0 if established else 0.0, note=(
        f"schema_catalog: meaning = {definition.get('meaning', 'not listed')!r}, unit = {definition.get('unit', '')!r}, "
        f"evidence = {definition.get('evidence', 'not listed')!r}. No vendor data dictionary in the repository: "
        "start / end events and sign convention are unknown."))
    if endpoint not in pop.columns:
        add("missingness", "all", "share_non_null", None, 0, "endpoint column absent from the population")
        return rows
    p = pop.filter(pl.col("position_group") == position).with_columns(pl.col(endpoint).cast(pl.Float64))
    v = p.filter(pl.col(endpoint).is_not_null()).sort("nfl_id", "game_id", "play_id").with_columns(
        pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("nfl_id").alias("_idx"))
    x, ath = v[endpoint].to_numpy().astype(float), v["nfl_id"].to_numpy()

    # ---- missingness (player-plays)
    def miss(stratum: str, g: pl.DataFrame) -> None:
        n, k = g.height, g.filter(pl.col(endpoint).is_not_null()).height
        add("missingness", stratum, "n_non_null", k, n, "player-plays")
        add("missingness", stratum, "share_non_null", k / n if n else None, n, "player-plays")

    miss("all", p)
    for season in sorted(p["season"].drop_nulls().unique().to_list()):
        miss(f"season:{season}", p.filter(pl.col("season") == season))
    for field, top in (("pass_result", 0), ("lined_up_position", 6), ("blitzing", 0)):
        if field not in p.columns:
            continue
        lev = p.with_columns(pl.col(field).cast(pl.String))
        if not top:                                   # labelled vs unlabelled plays
            miss(f"{field}:labelled", lev.filter(pl.col(field).is_not_null()))
            miss(f"{field}:null", lev.filter(pl.col(field).is_null()))
            continue
        for name in lev.group_by(field).len().sort(["len", field], descending=[True, False])[field].head(top).to_list():
            if name is not None:
                miss(f"{field}:{name}", lev.filter(pl.col(field) == name))

    # ---- distribution (non-null player-plays)
    if x.size:
        q = np.quantile(x, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        for name, val in (("mean", x.mean()), ("sd", x.std(ddof=1) if x.size > 1 else None), ("min", x.min()),
                          ("p01", q[0]), ("p05", q[1]), ("p25", q[2]), ("p50", q[3]), ("p75", q[4]), ("p95", q[5]),
                          ("p99", q[6]), ("max", x.max()), ("share_le_zero", float((x <= 0).mean()))):
            add("distribution", "all", name, val, x.size, "non-null player-plays")

    # ---- athletes and observations per athlete
    counts = v.group_by("nfl_id").len()
    k = counts["len"].to_numpy().astype(float)
    n_pos = p["nfl_id"].n_unique()
    add("athletes", "all", "n_athletes_position", n_pos, n_pos, "athletes of the position group with tracked plays")
    add("athletes", "all", "n_athletes_any_obs", k.size, n_pos, "athletes")
    for qn, qv in (("p10", 0.1), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9)):
        add("athletes", "all", f"obs_per_athlete_{qn}", float(np.quantile(k, qv)) if k.size else None, k.size, "athletes")
    fl = [c for c in flags.columns if c != "nfl_id"]
    for thr in rcfg.outcome_obs_grid:
        ids = counts.filter(pl.col("len") >= thr)["nfl_id"].to_list()
        add("athletes", f"min_obs={thr}", "n_athletes", len(ids), k.size, "athletes with at least this many observations")
        got = flags.filter(pl.col("nfl_id").is_in(ids)) if ids else flags.head(0)
        for c in fl:
            add("intersection", f"min_obs={thr}", c, int(got[c].sum() or 0), len(ids), "athletes also meeting the Combine flag")
        if {"forty_tracked", "has_cmb_forty", "has_cmb_ten_yd_split"} <= set(fl):
            both = got.filter(pl.col("forty_tracked") & pl.col("has_cmb_forty") & pl.col("has_cmb_ten_yd_split")).height
            add("intersection", f"min_obs={thr}", "forty_tracked_and_official_both", both, len(ids), "athletes")

    # ---- play context among non-null endpoint plays
    for field in rcfg.context_fields:
        if field in v.columns:
            add("context", "all", f"share_non_null:{field}", v[field].is_not_null().mean() if v.height else None, v.height,
                "non-null endpoint player-plays")
    if "lined_up_position" in v.columns and v.height:
        for name, cnt in v.group_by("lined_up_position").len().sort(["len", "lined_up_position"], descending=[True, False]
                                                                    ).head(8).iter_rows():
            add("context", f"lined_up_position:{name}", "share_of_endpoint_plays", cnt / v.height, v.height, "player-plays")

    # ---- repeatability over plays and games
    lo = min(rcfg.outcome_obs_grid)
    keep = np.isin(ath, counts.filter(pl.col("len") >= lo)["nfl_id"].to_numpy())
    seed = seed_for(rcfg.seed, "F", endpoint, position)
    if keep.sum() > 0:
        r = stratified_icc(x[keep], ath[keep], None, rcfg.n_boot, seed)
        note = f"single plays, athletes with ≥ {lo} observations; athlete bootstrap"
        add("reliability", "play", "icc1", r["icc"], r["n_athletes"], note)
        add("reliability", "play", "icc1_ci_lower", r["ci_lower"], r["n_athletes"], note)
        add("reliability", "play", "icc1_ci_upper", r["ci_upper"], r["n_athletes"], note)
        for n in rcfg.outcome_obs_grid:
            add("reliability", "play", f"projected_reliability_n={n}", spearman_brown(r["icc"], n), r["n_athletes"],
                "Spearman–Brown from the single-play ICC")
        for target in (0.7, 0.8):
            add("reliability", "play", f"plays_needed_for_{target:g}", observations_needed(r["icc"], target),
                r["n_athletes"], "Spearman–Brown inverse")
        if "lined_up_position" in v.columns:
            centred = v.with_columns((pl.col(endpoint) - pl.col(endpoint).mean().over("lined_up_position", "season"))
                                     .alias("_c"))["_c"].to_numpy().astype(float)
            c = stratified_icc(centred[keep], ath[keep], None, rcfg.n_boot, seed)
            add("reliability", "play_context_centred", "icc1", c["icc"], c["n_athletes"],
                "alignment × season means removed before the ICC")
            add("reliability", "play_context_centred", "icc1_ci_lower", c["ci_lower"], c["n_athletes"])
            add("reliability", "play_context_centred", "icc1_ci_upper", c["ci_upper"], c["n_athletes"])
    games = (v.group_by("nfl_id", "game_id").agg(pl.col(endpoint).mean().alias("_m"), pl.len().alias("_n"))
             .filter(pl.col("_n") >= rcfg.game_min_obs))
    games = games.filter(pl.len().over("nfl_id") >= 2)
    if games.height:
        g = stratified_icc(games["_m"].to_numpy(), games["nfl_id"].to_numpy(), None, rcfg.n_boot, seed)
        note = f"athlete-game means, games with ≥ {rcfg.game_min_obs} observations, athletes with ≥ 2 such games"
        add("reliability", "game", "icc1", g["icc"], g["n_athletes"], note)
        add("reliability", "game", "icc1_ci_lower", g["ci_lower"], g["n_athletes"], note)
        add("reliability", "game", "icc1_ci_upper", g["ci_upper"], g["n_athletes"], note)
        add("reliability", "game", "games_per_athlete_median", float(games.group_by("nfl_id").len()["len"].median()),
            g["n_athletes"], "athletes")
    half = (v.filter(pl.len().over("nfl_id") >= lo).group_by("nfl_id")
            .agg(pl.col(endpoint).filter(pl.col("_idx") % 2 == 1).mean().alias("_odd"),
                 pl.col(endpoint).filter(pl.col("_idx") % 2 == 0).mean().alias("_even")))
    if half.height >= rcfg.min_athletes:
        rho = spearman(half["_odd"].to_numpy(), half["_even"].to_numpy())
        add("reliability", "split_half", "spearman_odd_even", rho, half.height, "odd vs even chronological observations")
        add("reliability", "split_half", "spearman_brown_full_length", 2 * rho / (1 + rho) if np.isfinite(rho) and rho > -1
            else None, half.height, "reliability of the athlete mean over all own observations")

    # ---- first-N-play eligibility
    rookie = v.filter(pl.col("season") == pl.col("cohort")) if "cohort" in v.columns else v.head(0)
    rk = rookie.group_by("nfl_id").len()
    for n in rcfg.first_n_grid:
        add("first_n", f"n={n}", "n_athletes_ge_n", int((k >= n).sum()), k.size, "athletes with ≥ n observations overall")
        add("first_n", f"n={n}", "n_athletes_ge_n_rookie_season", rk.filter(pl.col("len") >= n).height, k.size,
            "observations in the season equal to the draft cohort year")
        first = v.filter(pl.col("_idx") <= n).group_by("nfl_id").agg(pl.len().alias("_n"), pl.col("game_id").n_unique()
                                                                    .alias("_g"), pl.col(endpoint).mean().alias("_a"))
        first = first.filter(pl.col("_n") == n)
        add("first_n", f"n={n}", "median_games_to_reach_n", float(first["_g"].median()) if first.height else None,
            first.height, "athletes reaching n observations")
        nxt = (v.filter((pl.col("_idx") > n) & (pl.col("_idx") <= 2 * n)).group_by("nfl_id")
               .agg(pl.len().alias("_n"), pl.col(endpoint).mean().alias("_b")).filter(pl.col("_n") == n))
        pair = first.join(nxt, on="nfl_id", how="inner")
        add("first_n", f"n={n}", "spearman_first_n_vs_next_n",
            spearman(pair["_a"].to_numpy(), pair["_b"].to_numpy()) if pair.height >= rcfg.min_athletes else None,
            pair.height, "mean of the first n observations vs mean of the next n (athletes with ≥ 2n)")
    return rows


def endpoint_table(pop: pl.DataFrame, flags: pl.DataFrame, definitions: dict[str, dict[str, str]], rcfg: RescueConfig
                   ) -> pl.DataFrame:
    rows = [r for endpoint, position in rcfg.endpoints
            for r in endpoint_feasibility(pop, endpoint, position, flags, definitions.get(endpoint, {}), rcfg)]
    return pl.DataFrame(rows, schema=ENDPOINT.columns) if rows else ENDPOINT.empty()
