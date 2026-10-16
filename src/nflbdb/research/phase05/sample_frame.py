"""Stage P0b — analysis sample frame (Phase 05 · experiment E05-00b).

Why          The executed P0 census (reports/PHASE_05_P0_RESULTS_REVIEW.md) left five questions open that
             decide whether the primary experiment is supported:
             1. effective sample: how many athletes have enough NFL plays AND the Combine drill (and ≥ 2
                attempts, and an official result)? P0 counted the NFL side only;
             2. cohort: exposure accrues by draft cohort (1–3 seasons), so cohort confounds exposure;
             3. outcome volume: P0 used one cut (≥ 20 observations) per vendor outcome;
             4. co-tracking by season, restricted to position pairs that interact by role (the P0 rule
                picked OL × DB by count alone);
             5. play type: for WR / TE, ≥ 92 % of tracked player-plays carry a route label — are the tracked
                plays mostly pass plays?
Population   as Stage P0 (tracked player-plays, ``season_types`` of DecisionAuditConfig) plus every Combine
             athlete of gold.dim_athlete. Unit: athlete.
Method       A census. Combine PARTICIPATION only (attempt counts, presence of an official result): no
             Combine feature value is read and no Combine-to-NFL statistic is computed, so the Phase 04 gate
             is not touched.
Planning     ``min_detectable_r`` = smallest |correlation| detectable with the configured power (Fisher z,
             two-sided α). It is a planning number for a SIMPLE correlation with n athletes; an incremental
             skill gain (model c − model b, E05-02) is harder to detect than this.

Command for the USER (code authored, NOT executed)::

    python -m nflbdb.research.phase05 --only P0B
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.config import Phase01Config
from ...data.tables import LayerRefs
from ...data.utils import get_logger
from ...features.trajectory import canonical_drill_expr
from ..contracts import TableContract, cast_keys
from ..studies.store import FigureRecord, StudyStore
from . import decision_audit as da
from .config import PHASE, DecisionAuditConfig, Phase05Config, SampleFrameConfig

log = get_logger(__name__)
STUDY = "P0B"
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
ALL = da.ALL

COHORT = TableContract(
    "P0b_exposure_by_cohort", "Athletes with at least N tracked plays by position group × draft cohort.",
    {"position_group": S, "cohort": I, "min_plays": I, "n_athletes": I, "n_athletes_ge": I, "share_ge": F,
     "median_plays": F, "median_seasons": F},
    key=("position_group", "cohort", "min_plays"), area="descriptive",
    docs={"cohort": "gold.dim_athlete cohort; -1 = unknown", "min_plays": "0 = any play in the population"})
SAMPLE = TableContract(
    "P0b_combine_nfl_sample", "Athletes with both a Combine drill and NFL exposure (participation counts only).",
    {"position_group": S, "drill": S, "min_plays": I, "n_combine_athletes": I, "n_nfl_athletes": I, "n_both": I,
     "n_both_repeat": I, "n_both_official": I, "official_columns": S, "min_detectable_r": F},
    key=("position_group", "drill", "min_plays"), area="descriptive",
    docs={"n_combine_athletes": "athletes with ≥ 1 tracked attempt of the drill",
          "n_nfl_athletes": "athletes of the position group with ≥ max(min_plays, 1) tracked plays",
          "n_both_repeat": "… and ≥ min_attempts_repeat attempts (repeatability / ICC possible)",
          "n_both_official": "… and every listed official result non-null (baseline model available); null = "
                             "the drill has no official measurement",
          "min_detectable_r": "planning value for n_both athletes (Fisher z, configured α and power)"})
COUNTS_DS = TableContract(
    "athlete_outcome_counts", "Non-null vendor outcome observations per athlete × field (P0 population).",
    {"nfl_id": I, "position_group": S, "field": S, "n_obs": I},
    key=("nfl_id", "field"), kind="dataset", area="decision_audit",
    docs={"n_obs": "non-null player-plays; for boolean fields this is NOT the number of TRUE values"})
OUTCOME = TableContract(
    "P0b_outcome_exposure", "Athletes with at least N non-null observations of a vendor outcome, by position group.",
    {"field": S, "position_group": S, "min_obs": I, "n_athletes_any": I, "n_athletes_ge": I, "n_obs": I,
     "median_obs_per_athlete": F, "min_detectable_r": F},
    key=("field", "position_group", "min_obs"), area="descriptive",
    docs={"median_obs_per_athlete": "among athletes with ≥ 1 observation",
          "min_detectable_r": "planning value for n_athletes_ge athletes"})
PAIRS_SEASON = TableContract(
    "P0b_pairs_by_season", "Co-tracked offense × defense position pairs by season (availability, not matchups).",
    {"season": S, **da.PAIRS.columns, "football_relevant": B},
    key=("season", "off_position", "def_position"), area="descriptive",
    docs={"football_relevant": "pair listed in SampleFrameConfig.relevant_pairs (roles that interact)"})
PLAY_TYPE = TableContract(
    "P0b_play_type", "Tracked player-plays by play type (vendor play-level field) and position group.",
    {"position_group": S, "play_type": S, "n_player_plays": I, "share": F, "n_route_labelled": I},
    key=("position_group", "play_type"), area="descriptive",
    docs={"play_type": "raw level of play_type_field; '<null>' = missing", "share": "within the position group"})
CONTRACTS = (COHORT, SAMPLE, COUNTS_DS, OUTCOME, PAIRS_SEASON, PLAY_TYPE)


# ============================================================================== pure functions
def min_detectable_r(n: int | None, alpha: float = 0.05, power: float = 0.80) -> float | None:
    """Smallest |r| detectable with ``power`` at two-sided ``alpha`` for n independent athletes (Fisher z)."""
    if n is None or n <= 3:
        return None
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return float(math.tanh(z / math.sqrt(n - 3)))


def exposure_by_cohort(exposure: pl.DataFrame, cfg: SampleFrameConfig) -> pl.DataFrame:
    e = exposure.with_columns(pl.col("cohort").fill_null(-1))
    rows = []
    for pos in (da._positions(e) if e.height else []):
        g = e if pos == ALL else e.filter(pl.col("position_group") == pos)
        for cohort in sorted(g["cohort"].unique().to_list()):
            c = g.filter(pl.col("cohort") == cohort)
            n = c["n_tracked_plays"].to_numpy()
            for thr in cfg.min_plays_grid:
                ge = int((n >= max(thr, 1)).sum())
                rows.append({"position_group": pos, "cohort": int(cohort), "min_plays": int(thr), "n_athletes": int(n.size),
                             "n_athletes_ge": ge, "share_ge": ge / n.size, "median_plays": float(np.median(n)),
                             "median_seasons": float(c["n_seasons"].median())})
    return da._frame(rows, COHORT)


def combine_nfl_sample(athletes: pl.DataFrame, exposure: pl.DataFrame, attempts: pl.DataFrame, results: pl.DataFrame,
                       cfg: SampleFrameConfig) -> pl.DataFrame:
    """``attempts``: (nfl_id, drill, n_attempts) · ``results``: silver.combine_results rows (presence only)."""
    a = (athletes.select("nfl_id", pl.col("position_group").fill_null("UNKNOWN"))
         .join(exposure.select("nfl_id", "n_tracked_plays"), on="nfl_id", how="left")
         .with_columns(pl.col("n_tracked_plays").fill_null(0)))
    official = dict(cfg.official_results)
    rows = []
    for drill in (sorted(attempts["drill"].unique().to_list()) if attempts.height and a.height else []):
        d = a.join(attempts.filter(pl.col("drill") == drill).select("nfl_id", "n_attempts"), on="nfl_id",
                   how="left").with_columns(pl.col("n_attempts").fill_null(0))
        cols = [c for c in official.get(drill, ()) if c in results.columns]
        if cols:
            has = results.group_by("nfl_id").agg(pl.all_horizontal([pl.col(c).is_not_null() for c in cols]).any()
                                                 .alias("_official"))
            d = d.join(has, on="nfl_id", how="left").with_columns(pl.col("_official").fill_null(False))
        for pos in da._positions(d):
            g = d if pos == ALL else d.filter(pl.col("position_group") == pos)
            plays, att = g["n_tracked_plays"].to_numpy(), g["n_attempts"].to_numpy()
            if not (att >= 1).any():
                continue
            off = g["_official"].to_numpy() if cols else None
            for thr in cfg.min_plays_grid:
                nfl = plays >= max(thr, 1)
                both = nfl & (att >= 1)
                rows.append({"position_group": pos, "drill": drill, "min_plays": int(thr),
                             "n_combine_athletes": int((att >= 1).sum()), "n_nfl_athletes": int(nfl.sum()),
                             "n_both": int(both.sum()),
                             "n_both_repeat": int((nfl & (att >= cfg.min_attempts_repeat)).sum()),
                             "n_both_official": int((both & off).sum()) if cols else None,
                             "official_columns": ", ".join(cols),
                             "min_detectable_r": min_detectable_r(int(both.sum()), cfg.alpha, cfg.power)})
    return da._frame(rows, SAMPLE).sort("position_group", "drill", "min_plays")


def athlete_outcome_counts(pop: pl.DataFrame, fields: Sequence[str]) -> pl.DataFrame:
    fields = [f for f in dict.fromkeys(fields) if f in pop.columns]
    if not pop.height or not fields:
        return COUNTS_DS.empty()
    wide = pop.group_by("nfl_id").agg(pl.col("position_group").first(),
                                      *[pl.col(f).is_not_null().sum().alias(f) for f in fields])
    return da._as(wide.unpivot(on=fields, index=["nfl_id", "position_group"], variable_name="field",
                               value_name="n_obs"), COUNTS_DS).sort("nfl_id", "field")


def outcome_exposure(counts: pl.DataFrame, cfg: SampleFrameConfig) -> pl.DataFrame:
    rows = []
    for field in (sorted(counts["field"].unique().to_list()) if counts.height else []):
        c = counts.filter(pl.col("field") == field)
        for pos in da._positions(c):
            k = (c if pos == ALL else c.filter(pl.col("position_group") == pos))["n_obs"].to_numpy()
            if not k.sum():
                continue
            for thr in cfg.outcome_obs_grid:
                ge = int((k >= thr).sum())
                rows.append({"field": field, "position_group": pos, "min_obs": int(thr), "n_athletes_any": int((k > 0).sum()),
                             "n_athletes_ge": ge, "n_obs": int(k.sum()), "median_obs_per_athlete": float(np.median(k[k > 0])),
                             "min_detectable_r": min_detectable_r(ge, cfg.alpha, cfg.power)})
    return da._frame(rows, OUTCOME)


def pairs_by_season(pop: pl.DataFrame, audit: DecisionAuditConfig, cfg: SampleFrameConfig) -> pl.DataFrame:
    parts = []
    for season in ([*sorted(pop["season"].unique().to_list()), ALL] if pop.height else []):
        sub = pop if season == ALL else pop.filter(pl.col("season") == season)
        parts.append(da.opponent_pairs(sub, audit).with_columns(pl.lit(str(season)).alias("season")))
    if not parts:
        return PAIRS_SEASON.empty()
    relevant = [f"{o}|{d}" for o, d in cfg.relevant_pairs]
    out = pl.concat(parts).with_columns(pl.concat_str("off_position", "def_position", separator="|").is_in(relevant)
                                        .alias("football_relevant"))
    return da._as(out, PAIRS_SEASON).sort("season", "off_position", "def_position")


def play_type_census(pop: pl.DataFrame, field: str) -> pl.DataFrame:
    if not pop.height or field not in pop.columns:
        return PLAY_TYPE.empty()
    route = (pl.col("route_ran").is_not_null().sum() if "route_ran" in pop.columns
             else pl.lit(None, dtype=pl.Int64).first())
    t = pop.with_columns(pl.col(field).cast(pl.String).fill_null("<null>").alias("play_type"))
    out = (pl.concat([t, t.with_columns(pl.lit(ALL).alias("position_group"))])
           .group_by("position_group", "play_type").agg(pl.len().alias("n_player_plays"), route.alias("n_route_labelled"))
           .with_columns((pl.col("n_player_plays") / pl.col("n_player_plays").sum().over("position_group")).alias("share")))
    return da._as(out, PLAY_TYPE).sort("position_group", "play_type")


# ============================================================================== orchestration
class SampleFrameStudy:
    STUDY = STUDY

    def __init__(self, cfg: Phase01Config, pcfg: Phase05Config | None = None, store: StudyStore | None = None,
                 **_: Any):
        self.cfg = cfg
        self.pcfg = pcfg or Phase05Config()
        self.f = self.pcfg.frame
        self.store = store or StudyStore(cfg, sampled=self.pcfg.sampled, phase=PHASE)
        self.refs = LayerRefs(cfg)

    def population(self) -> pl.DataFrame:
        """The Stage P0 base frame (same joins, same population) with the play-type field added."""
        aud = da.DecisionAuditStudy(self.cfg, self.pcfg, self.store)
        aud.roles = {**aud.roles, "state": (*aud.roles["state"], self.f.play_type_field)}
        inp = aud.inputs()
        fields = [f for r in da.ROLES for f in aud.roles[r]]
        base, _ = da.assemble_base(inp["fact"], inp["player_play"], inp["plays"], inp["games"], inp["athletes"], fields)
        return base.filter(pl.col("season_type").is_in(list(self.pcfg.audit.season_types)))

    def combine_inputs(self) -> dict[str, pl.DataFrame]:
        attempts = (cast_keys(self.refs.scan("gold", "fact_combine_attempt").select("nfl_id", "drill_name").collect())
                    .with_columns(canonical_drill_expr()).group_by("nfl_id", "drill")
                    .agg(pl.len().cast(pl.Int64).alias("n_attempts")))
        res = self.refs.scan("silver", "combine_results")
        cols = [c for _, cs in self.f.official_results for c in cs if c in res.collect_schema().names()]
        athletes = self.refs.scan("gold", "dim_athlete").select("nfl_id", "position_group").collect()
        return {"attempts": attempts, "results": cast_keys(res.select(["nfl_id", *dict.fromkeys(cols)]).collect()),
                "athletes": cast_keys(athletes)}

    def compute(self) -> dict[str, pl.DataFrame]:
        pop = self.population()
        exposure = da.athlete_exposure(pop)
        cmb = self.combine_inputs()
        outcome_fields = [*da.audit_fields(self.pcfg)["outcome"], *[f for f, _ in self.pcfg.streams.outcome_positions]]
        counts = athlete_outcome_counts(pop, outcome_fields)
        log.info("[phase05:P0B] %s player-plays, %s athletes, %s athlete-drills", f"{pop.height:,}",
                 f"{exposure.height:,}", f"{cmb['attempts'].height:,}")
        return {COHORT.name: exposure_by_cohort(exposure, self.f),
                SAMPLE.name: combine_nfl_sample(cmb["athletes"], exposure, cmb["attempts"], cmb["results"], self.f),
                COUNTS_DS.name: counts, OUTCOME.name: outcome_exposure(counts, self.f),
                PAIRS_SEASON.name: pairs_by_season(pop, self.pcfg.audit, self.f),
                PLAY_TYPE.name: play_type_census(pop, self.f.play_type_field)}

    def fingerprint(self) -> str:
        upstream = {f"{layer}.{name}": self.refs.num_rows(layer, name)
                    for layer, name in (("gold", "fact_nfl_play_participation"), ("silver", "player_play"),
                                        ("silver", "plays"), ("gold", "dim_athlete"), ("gold", "fact_combine_attempt"),
                                        ("silver", "combine_results"))}
        return self.store.fingerprint(STUDY, self.pcfg.section("audit", "frame"), upstream)

    def run(self, force: bool = False, figures: bool = True) -> dict[str, pl.DataFrame]:
        fp = self.fingerprint()
        current, reason = self.store.is_current(STUDY, fp, CONTRACTS)
        if current and not force:
            log.info("[phase05:P0B] reuse (%s)", reason)
            out = self.load()
        else:
            log.info("[phase05:P0B] compute (%s%s)", "forced; " if force else "", reason)
            res = self.compute()
            out = {c.name: self.store.write(res[c.name], c, STUDY) for c in CONTRACTS}
            self.store.record_run(STUDY, fp, CONTRACTS)
        if figures:
            self.figures(out)
        return out

    def load(self) -> dict[str, pl.DataFrame]:
        return {c.name: self.store.read(c) for c in CONTRACTS}

    def figures(self, out: dict[str, pl.DataFrame]) -> None:
        for rec, fn in FIGURES:
            fn(self.store, rec, out)


# ============================================================================== figures
def fig_sample(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from .audit_figures import _heatmap, _kit

    _, ink, seq, _, empty_panel, figure, finish = _kit()
    show = SampleFrameConfig()
    s = out[SAMPLE.name].filter(pl.col("min_plays") == show.figure_min_plays)
    fig, ax = figure(1, 2, size=(12.8, 8.0))
    top = s.filter(pl.col("position_group") == ALL).sort("n_both", descending=True).head(show.figure_max_drills)
    drills = top["drill"].to_list()
    cols = [p for p in sorted(s["position_group"].unique().to_list()) if p != ALL] + [ALL] if s.height else []
    for a, col, title in ((ax[0, 0], "n_both", "athletes with the drill and the NFL exposure"),
                          (ax[0, 1], "n_both_repeat", "… and ≥ 2 Combine attempts (repeatability)")):
        if not drills:
            empty_panel(a, "no athlete has both")
            continue
        m = np.full((len(drills), len(cols)), np.nan)
        lab = np.full(m.shape, "–", dtype=object)
        for r in s.filter(pl.col("drill").is_in(drills)).iter_rows(named=True):
            i, j = drills.index(r["drill"]), cols.index(r["position_group"])
            m[i, j], lab[i, j] = r[col], f"{r[col]:,}"
        _heatmap(a, m, lab, [d.replace("_", " ").lower() for d in drills], cols, seq, ink, fontsize=7)
        a.set_title(title)
        a.set_xlabel("position group")
    finish(fig, f"Athletes with ≥ {show.figure_min_plays} tracked regular-season plays AND the Combine drill; the "
                f"{show.figure_max_drills} drills with the largest overall sample. Participation counts only: no Combine "
                "value is used. The play minimum is a display choice, the table holds the full grid. "
                "PENDING interpretation.", "P0b-1 · How many athletes can a Combine → NFL analysis actually use?")
    store.save_figure(fig, rec, {"sample": s}, n=f"{int(top['n_both'].max() or 0) if top.height else 0:,} athletes (largest drill)")


def fig_pairs(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from .audit_figures import _kit

    cat, ink, _, _, empty_panel, figure, finish = _kit()
    p = out[PAIRS_SEASON.name].filter(pl.col("football_relevant"))
    fig, ax = figure(1, 2)
    a, b = ax[0, 0], ax[0, 1]
    by = p.filter(pl.col("season") != ALL)
    tot = p.filter(pl.col("season") == ALL).sort("off_position", "def_position")
    if not by.height:
        empty_panel(a, "no relevant co-tracked pair")
        empty_panel(b, "no relevant co-tracked pair")
    else:
        seasons = sorted(by["season"].unique().to_list())
        names = [f"{o} × {d}" for o, d in tot.select("off_position", "def_position").iter_rows()]
        w = 0.8 / max(len(names), 1)
        for k, (o, d) in enumerate(tot.select("off_position", "def_position").iter_rows()):
            g = {r["season"]: r["n_plays"] for r in by.filter((pl.col("off_position") == o) & (pl.col("def_position") == d))
                 .iter_rows(named=True)}
            a.bar(np.arange(len(seasons)) + (k - (len(names) - 1) / 2) * w, [g.get(x, 0) for x in seasons], width=w,
                  color=cat[k % 8], label=names[k])
        a.set_xticks(np.arange(len(seasons)), seasons)
        a.set_ylabel("plays with both position groups tracked")
        a.legend(frameon=False, fontsize=8)
        y = np.arange(tot.height)
        b.barh(y - 0.2, tot["n_distinct_pairs"].to_numpy(), height=0.4, color=cat[0], label="distinct athlete pairs")
        b.barh(y + 0.2, tot["n_pairs_min_obs"].to_numpy(), height=0.4, color=cat[1], label="pairs with ≥ min_obs plays")
        for i, r in enumerate(tot.iter_rows(named=True)):
            b.text(r["n_distinct_pairs"], i - 0.2, f" {r['n_distinct_pairs']:,}", va="center", fontsize=8, color=ink["secondary"])
            b.text(r["n_pairs_min_obs"], i + 0.2, f" {r['n_pairs_min_obs']:,} (median {r['median_plays_per_pair']:.0f} plays/pair)",
                   va="center", fontsize=8, color=ink["secondary"])
        b.set_yticks(y, names)
        b.invert_yaxis()
        b.set_xlabel("athlete pairs, all seasons")
        b.legend(frameon=False, fontsize=8, loc="lower right")
    finish(fig, "Only position pairs whose roles interact. Co-tracking is availability: being on the same play does not "
                "mean the two athletes faced each other, which Stage S7 must establish from geometry. Cohort accrual "
                "drives the growth by season. PENDING interpretation.",
           "P0b-2 · Where could a matchup analysis come from?")
    store.save_figure(fig, rec, {"pairs": p}, n=f"{int(tot['n_plays'].sum()) if tot.height else 0:,} co-tracked plays")


def _rec(fid: str, name: str, q: str, insight: str, stat: str, ds: str, grain: str) -> FigureRecord:
    return FigureRecord(figure_id=fid, name=name, study=STUDY, research_question=q, expected_insight=insight,
                        statistic=stat, dataset=ds, grain=grain, filters="season types of DecisionAuditConfig",
                        method="exact census")


FIGURES = [
    (_rec("P0b-01", "P0b_01_combine_nfl_sample", "How many athletes have both the Combine drill and NFL exposure?",
          "effective athlete sample per drill × position for E05-02", "athlete counts", SAMPLE.name, "athlete"),
     fig_sample),
    (_rec("P0b-02", "P0b_02_relevant_pairs_by_season", "Where is role-relevant co-tracking concentrated?",
          "seasons and position pairs that could support a matchup analysis", "plays and athlete pairs",
          PAIRS_SEASON.name, "play × athlete pair"), fig_pairs),
]
