"""Study Q — structural vs signal quality, and the impact of quality filters on ATHLETE eligibility
(Phase 04 · Milestone 1 · Study 1.1).

Question     Executed evidence (Study A1/A3) shows the provisional tiers do not discriminate (A3: AUC of
             tier ≠ A ≈ 0.50 for every 'unreliable signal' target) while continuous metrics do (resid_rms_yds,
             speed_vendor_mad). (1) Are STRUCTURAL defects (breaks, off-grid sampling, positional noise) and
             SIGNAL defects (vendor disagreement, derivative roughness) the same trajectories? (2) How many
             athletes stay eligible — ≥ 1 and ≥ 2 attempts per drill — under each filter scenario?
Population   every Combine attempt, plus the deterministic NFL game sample ('nfl_sample', 8 games/season)
             — the trajectories characterised by BOTH Study A (``trajectory_quality_dimensions``) and Study A3
             (``trajectory_signal_reliability``). Read-only: nothing is recomputed from frames.
             Grain: trajectory; unit: athlete.
Definitions  structural_flag = tier 'reject' OR an A1 p99 screening flag on structural_integrity /
                               temporal_continuity / coordinate_validity
             signal_flag     = vendor RMSE OR interior jerk RMS above the domain × stratum q95 (= A3 targets)
             boundary_flag   = peak a_tan or peak deceleration on an SG filter-boundary frame (A3)
             All three thresholds are PROPOSED screening cuts (continuous metrics stay in Phase 03 tables).
Scenarios    S0 all → S1 structural pass → S2 + signal pass → S3 + no boundary peak (cumulative).
Statistics   flag shares with athlete-cluster bootstrap CIs by domain × facet (stratum, cohort, position);
             Cohen's κ between structural and signal flags (κ ≈ 0 → distinct dimensions) with cluster CI;
             eligibility counts per domain × stratum (drill_type) × drill and scenario.
Caveat       Quantile flags flag ≈ 5 % by construction within each stratum — their RATE is not evidence;
             their OVERLAP, their concentration in athletes / positions / cohorts, and the eligibility
             they remove are.

Command for the USER (code authored, NOT executed)::

    python -m nflbdb.research.phase04 --only Q
"""

from __future__ import annotations

import dataclasses
import zlib
from typing import Any

import numpy as np
import polars as pl

from ...data.config import Phase01Config
from ...data.utils import get_logger
from ..contracts import TableContract, cast_keys
from ..kinematic_build import COMBINE_TRAJ, KinematicFeatureBuilder
from ..studies.quality import DIMS, traj_id_expr
from ..studies.results import RESULTS, ResultCollector
from ..studies.signal_reliability import SIGNALS
from ..studies.stats import cluster_bootstrap, cluster_bootstrap_ratio
from ..studies.store import FigureRecord, StudyStore
from .config import EligibilityConfig, Phase04Config
from .metrics import cohen_kappa

log = get_logger(__name__)
STUDY = "Q"
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
STRUCTURAL_DIMS = ("structural_integrity", "temporal_continuity", "coordinate_validity")
SCENARIOS = (("S0 all", None), ("S1 structural pass", ["structural_flag"]),
             ("S2 + signal pass", ["structural_flag", "signal_flag"]),
             ("S3 + no boundary peak", ["structural_flag", "signal_flag", "boundary_flag"]))
FLAGS = ("structural_flag", "signal_flag", "boundary_flag", "any_flag")

FLAGS_DS = TableContract(
    "trajectory_quality_flags", "One row per trajectory: structural / signal / boundary screening flags (Study Q).",
    {"domain": S, "traj_id": S, "nfl_id": I, "cohort": I, "stratum": S, "drill": S, "position": S, "quality_tier": S,
     "structural_flag": B, "signal_flag": B, "boundary_flag": B, "any_flag": B, "structural_reasons": S,
     "signal_reasons": S},
    key=("domain", "traj_id"), kind="dataset", area="quality",
    docs={"structural_flag": "tier reject or A1 p99 flag on structural_integrity / temporal_continuity / "
                             "coordinate_validity", "signal_flag": "vendor RMSE or interior jerk > domain × stratum "
                                                                   "signal_quantile (A3 metrics)",
          "boundary_flag": "peak a_tan or peak deceleration on an SG filter-boundary frame (A3)"})
RATES = TableContract(
    "Q_flag_rates", "Share of trajectories flagged, by domain × facet level (athlete-cluster bootstrap CI).",
    {"domain": S, "facet": S, "level": S, "flag": S, "n_trajectories": I, "n_athletes": I, "n_flagged": I,
     "share": F, "ci_lower": F, "ci_upper": F, "result_status": S},
    key=("domain", "facet", "level", "flag"), area="descriptive")
AGREE = TableContract(
    "Q_flag_agreement", "Structural vs signal flags: 2×2 counts and Cohen's κ (athlete-cluster bootstrap CI).",
    {"domain": S, "stratum": S, "n_trajectories": I, "n_athletes": I, "n_both": I, "n_structural_only": I,
     "n_signal_only": I, "n_neither": I, "kappa": F, "ci_lower": F, "ci_upper": F, "result_status": S},
    key=("domain", "stratum"), area="statistical",
    docs={"kappa": "≈ 0: the two flag families pick different trajectories (distinct quality dimensions)"})
ELIG = TableContract(
    "Q_athlete_eligibility", "Athletes retained per stratum × drill under cumulative quality-filter scenarios.",
    {"domain": S, "stratum": S, "drill": S, "scenario": S, "scenario_order": I, "n_trajectories": I,
     "n_athletes_any": I, "n_athletes_repeat": I, "share_trajectories": F, "share_athletes_any": F,
     "share_athletes_repeat": F},
    key=("domain", "stratum", "drill", "scenario"), area="descriptive",
    docs={"stratum": "Combine drill_type · NFL 'nfl_play'. Part of the grain: a canonical drill run by several "
                     "position groups (e.g. GAUNTLET_DRILL in SKILL_DRILLS_DB / _WR / _TE) is one row per drill_type",
          "n_athletes_repeat": "athletes with ≥ min_attempts_eligible retained trajectories in this stratum × drill "
                               "(needed for repeatability / ICC)",
          "share_athletes_any": "relative to S0 (all trajectories) in the same stratum × drill"})
STUDY_RESULTS = dataclasses.replace(RESULTS, name="Q_results")
CONTRACTS = (FLAGS_DS, RATES, AGREE, ELIG, STUDY_RESULTS)


def _seed(base: int, *parts: Any) -> int:
    return (base + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 32)


# ============================================================================== pure analysis
def build_flags(dims: pl.DataFrame, signals: pl.DataFrame, drills: pl.DataFrame, cfg: EligibilityConfig) -> pl.DataFrame:
    """``dims``: Study A DIMS · ``signals``: Study A3 SIGNALS · ``drills``: traj_id → canonical drill (Combine)."""
    grp = ["domain", "stratum"]
    hi = lambda c: (pl.col(c) > pl.col(c).quantile(cfg.signal_quantile, "linear").over(grp)).fill_null(False)  # noqa: E731
    sig = signals.select(
        "domain", "traj_id", "stratum", "vendor_rmse", "jerk_rms_interior", "peak_a_tan_in_boundary",
        "peak_decel_in_boundary").with_columns(
        hi("vendor_rmse").alias("_v"), hi("jerk_rms_interior").alias("_j"),
        (pl.col("peak_a_tan_in_boundary").fill_null(False) | pl.col("peak_decel_in_boundary").fill_null(False))
        .alias("boundary_flag"))
    sig = sig.with_columns(
        (pl.col("_v") | pl.col("_j")).alias("signal_flag"),
        pl.concat_str([pl.when(pl.col("_v")).then(pl.lit("vendor_rmse")), pl.when(pl.col("_j")).then(pl.lit("jerk"))],
                      separator=";", ignore_nulls=True).alias("signal_reasons")).drop("stratum")
    struct_cols = [f"flag_{d}" for d in STRUCTURAL_DIMS]
    d = dims.with_columns(
        ((pl.col("quality_tier") == "reject") | pl.any_horizontal([pl.col(c).fill_null(False) for c in struct_cols]))
        .alias("structural_flag"),
        pl.concat_str([pl.when(pl.col("quality_tier") == "reject").then(pl.lit("tier_reject")),
                       *[pl.when(pl.col(f"flag_{x}").fill_null(False)).then(pl.lit(x)) for x in STRUCTURAL_DIMS]],
                      separator=";", ignore_nulls=True).alias("structural_reasons"))
    # Study A labels the full NFL population 'nfl'; A3 measured signals on the deterministic 'nfl_sample'
    # (same traj_id format). Inner join → NFL rows are the sample, where BOTH flag families are defined.
    # Full-NFL structural retention is already in A1_filter_funnel.
    sig = sig.with_columns(pl.col("domain").replace({"nfl_sample": "nfl"}).alias("_dims_domain"))
    t = (d.rename({"domain": "_dims_domain"})
         .join(sig.select("domain", "_dims_domain", "traj_id", "signal_flag", "boundary_flag", "signal_reasons"),
               on=["_dims_domain", "traj_id"], how="inner")
         .join(drills, on="traj_id", how="left"))
    return t.select(
        "domain", "traj_id", pl.col("nfl_id").cast(pl.Int64), pl.col("cohort").cast(pl.Int64), "stratum",
        pl.coalesce(pl.col("drill"), pl.col("stratum")).cast(pl.String).alias("drill"), "position", "quality_tier",
        pl.col("structural_flag").fill_null(False), pl.col("signal_flag").fill_null(False),
        pl.col("boundary_flag").fill_null(False),
        (pl.col("structural_flag").fill_null(False) | pl.col("signal_flag").fill_null(False)
         | pl.col("boundary_flag").fill_null(False)).alias("any_flag"),
        pl.col("structural_reasons").fill_null(""), pl.col("signal_reasons").fill_null(""))


def flag_rates(t: pl.DataFrame, cfg: EligibilityConfig, seed: int) -> pl.DataFrame:
    rows = []
    for facet in ("all", "stratum", "cohort", "position"):
        d = t.with_columns(pl.lit("all").alias("all")) if facet == "all" else t
        for (domain, level), g in d.group_by("domain", facet, maintain_order=True):
            for fl in FLAGS:
                per = g.group_by("nfl_id").agg(pl.col(fl).sum().alias("k"), pl.len().alias("n"))
                ok = per.height >= cfg.min_athletes
                ci = (cluster_bootstrap_ratio(per["k"].to_numpy(), per["n"].to_numpy(), cfg.n_boot,
                                              _seed(seed, domain, facet, level, fl)) if ok else
                      {"estimate": float(g[fl].mean()), "ci_lower": None, "ci_upper": None})
                rows.append({"domain": domain, "facet": facet, "level": str(level), "flag": fl,
                             "n_trajectories": g.height, "n_athletes": per.height, "n_flagged": int(g[fl].sum()),
                             "share": ci["estimate"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
                             "result_status": "descriptive" if ok else "insufficient_data"})
    return pl.DataFrame(rows, schema=RATES.columns) if rows else RATES.empty()


def flag_agreement(t: pl.DataFrame, cfg: EligibilityConfig, seed: int) -> pl.DataFrame:
    rows = []
    parts = [t, t.with_columns(pl.lit("all").alias("stratum"))]
    for (domain, stratum), g in pl.concat(parts).group_by("domain", "stratum", maintain_order=True):
        a, b = g["structural_flag"].to_numpy(), g["signal_flag"].to_numpy()
        n_ath = g["nfl_id"].n_unique()
        est = lo = hi = None
        status = "insufficient_data"
        if n_ath >= cfg.min_athletes and a.any() and b.any():
            bs = cluster_bootstrap(lambda idx: cohen_kappa(a[idx], b[idx]), g["nfl_id"].to_numpy(), cfg.n_boot,
                                   _seed(seed, "kappa", domain, stratum))
            est, lo, hi, status = bs["estimate"], bs["ci_lower"], bs["ci_upper"], "exploratory"
        rows.append({"domain": domain, "stratum": stratum, "n_trajectories": g.height, "n_athletes": n_ath,
                     "n_both": int((a & b).sum()), "n_structural_only": int((a & ~b).sum()),
                     "n_signal_only": int((~a & b).sum()), "n_neither": int((~a & ~b).sum()),
                     "kappa": est, "ci_lower": lo, "ci_upper": hi, "result_status": status})
    return pl.DataFrame(rows, schema=AGREE.columns) if rows else AGREE.empty()


def eligibility(t: pl.DataFrame, cfg: EligibilityConfig) -> pl.DataFrame:
    base = t.group_by("domain", "stratum", "drill").agg(pl.len().alias("_n0"), pl.col("nfl_id").n_unique().alias("_a0"))
    rows = []
    for i, (name, excl) in enumerate(SCENARIOS):
        keep = pl.lit(True) if not excl else ~pl.any_horizontal([pl.col(c) for c in excl])
        per = (t.filter(keep).group_by("domain", "stratum", "drill", "nfl_id").agg(pl.len().alias("k"))
               .group_by("domain", "stratum", "drill").agg(
                   pl.col("k").sum().cast(pl.Int64).alias("n_trajectories"),
                   pl.len().cast(pl.Int64).alias("n_athletes_any"),
                   (pl.col("k") >= cfg.min_attempts_eligible).sum().cast(pl.Int64).alias("n_athletes_repeat")))
        r = base.join(per, on=["domain", "stratum", "drill"], how="left").with_columns(
            pl.col("n_trajectories").fill_null(0), pl.col("n_athletes_any").fill_null(0),
            pl.col("n_athletes_repeat").fill_null(0), pl.lit(name).alias("scenario"),
            pl.lit(i, dtype=pl.Int64).alias("scenario_order"))
        rows.append(r)
    out = pl.concat(rows, how="vertical_relaxed")
    # S0 reference at the SAME grain as ``base`` (domain × stratum × drill): joining on (domain, drill) alone
    # multiplies the rows of every drill that exists under several drill_types.
    rep0 = out.filter(pl.col("scenario_order") == 0).select("domain", "stratum", "drill",
                                                            pl.col("n_athletes_repeat").alias("_r0"))
    out = out.join(rep0, on=["domain", "stratum", "drill"], how="left").with_columns(
        (pl.col("n_trajectories") / pl.col("_n0")).cast(pl.Float64).alias("share_trajectories"),
        (pl.col("n_athletes_any") / pl.col("_a0")).cast(pl.Float64).alias("share_athletes_any"),
        pl.when(pl.col("_r0") > 0).then(pl.col("n_athletes_repeat") / pl.col("_r0")).cast(pl.Float64)
        .alias("share_athletes_repeat"))
    return out.select(list(ELIG.columns)).sort("domain", "stratum", "drill", "scenario_order")


def q_results(rates: pl.DataFrame, agree: pl.DataFrame, rc: ResultCollector) -> None:
    for r in rates.filter(pl.col("facet") != "all").iter_rows(named=True):
        rc.add(analysis_id="Q.flag_rate", grain="trajectory", n_athletes=r["n_athletes"],
               n_trajectories=r["n_trajectories"], effect_estimate=r["share"], effect_size_type="proportion",
               method="flagged trajectories / trajectories", domain=r["domain"], feature=r["flag"],
               stratum=r["level"] if r["facet"] == "stratum" else "all",
               cohort=r["level"] if r["facet"] == "cohort" else "all",
               position=r["level"] if r["facet"] == "position" else "all", ci_lower=r["ci_lower"],
               ci_upper=r["ci_upper"], ci_method="athlete-cluster bootstrap (percentile)",
               limitations="quantile flags are ≈ 5 % per stratum by construction; compare across levels")
    for r in agree.iter_rows(named=True):
        if r["kappa"] is not None:
            rc.add(analysis_id="Q.structural_vs_signal", grain="trajectory", n_athletes=r["n_athletes"],
                   n_trajectories=r["n_trajectories"], effect_estimate=r["kappa"], effect_size_type="Cohen kappa",
                   method="structural_flag vs signal_flag", domain=r["domain"], stratum=r["stratum"],
                   ci_lower=r["ci_lower"], ci_upper=r["ci_upper"], ci_method="athlete-cluster bootstrap (percentile)",
                   result_status="exploratory")


# ============================================================================== orchestration
class QualityEligibilityStudy:
    STUDY = STUDY

    def __init__(self, cfg: Phase01Config, pcfg: Phase04Config | None = None, store: StudyStore | None = None,
                 store03: StudyStore | None = None, **_: Any):
        self.cfg = cfg
        self.pcfg = pcfg or Phase04Config()
        self.e = self.pcfg.eligibility
        sampled = self.pcfg.sample_athletes is not None
        self.store = store or StudyStore(cfg, sampled=sampled, phase="phase_04")
        self.store03 = store03 or StudyStore(cfg, sampled=sampled)
        self.kin = KinematicFeatureBuilder(cfg)

    def inputs(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        dims, signals = self.store03.read(DIMS), self.store03.read(SIGNALS)
        keys = ["event_id", "nfl_id", "attempt"]
        drills = (cast_keys(self.kin.store.load(COMBINE_TRAJ.name).select(*keys, "drill").collect())
                  .select(traj_id_expr(keys), pl.col("drill").cast(pl.String)))
        return dims, signals, drills

    def compute(self) -> dict[str, pl.DataFrame]:
        dims, signals, drills = self.inputs()
        t = build_flags(dims, signals, drills, self.e)
        log.info("[phase04:Q] %s trajectories", f"{t.height:,}")
        rc = ResultCollector(STUDY, self.e.min_athletes)
        rates, agree = flag_rates(t, self.e, self.pcfg.seed), flag_agreement(t, self.e, self.pcfg.seed)
        q_results(rates, agree, rc)
        return {FLAGS_DS.name: t, RATES.name: rates, AGREE.name: agree, ELIG.name: eligibility(t, self.e),
                STUDY_RESULTS.name: rc.frame()}

    def fingerprint(self) -> str:
        runs = self.store03._manifest()["runs"]
        return self.store.fingerprint(STUDY, self.pcfg.section("eligibility"),
                                      {"study_A": runs.get("A", {}).get("fingerprint"),
                                       "study_A3": runs.get("A3", {}).get("fingerprint")})

    def run(self, force: bool = False, figures: bool = True) -> dict[str, pl.DataFrame]:
        fp = self.fingerprint()
        current, reason = self.store.is_current(STUDY, fp, CONTRACTS)
        if current and not force:
            log.info("[phase04:Q] reuse (%s)", reason)
            out = self.load()
        else:
            log.info("[phase04:Q] compute (%s%s)", "forced; " if force else "", reason)
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
def _style():
    from ...data.reporting import CATEGORICAL, INK

    return CATEGORICAL, INK


def _hbar_ci(a, labels, est, lo, hi, color, label=None, offset=0.0):
    y = np.arange(len(labels)) + offset
    est = np.asarray(est, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    err = np.vstack([np.where(np.isfinite(lo), est - lo, 0), np.where(np.isfinite(hi), hi - est, 0)]).clip(0)
    a.errorbar(est, y, xerr=err, fmt="o", color=color, ms=5, capsize=2, label=label)


def fig_flag_rates(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    r = out[RATES.name].filter(pl.col("facet") == "stratum")
    domains = [d for d in ("combine", "nfl_sample") if d in set(r["domain"].to_list())] or ["combine"]
    fig, ax = figure(1, len(domains), sharex=True)
    for k, dom in enumerate(domains):
        a = ax[0, k]
        d = r.filter(pl.col("domain") == dom)
        if not d.height:
            empty_panel(a, f"no {dom} rows")
            continue
        levels = sorted(d["level"].unique().to_list())
        for j, fl in enumerate(FLAGS):
            g = d.filter(pl.col("flag") == fl).sort("level")
            g = pl.DataFrame({"level": levels}).join(g, on="level", how="left")
            _hbar_ci(a, levels, g["share"].fill_null(np.nan), g["ci_lower"].fill_null(np.nan),
                     g["ci_upper"].fill_null(np.nan), cat[j], fl, (j - 1.5) * 0.18)
        n = d.filter(pl.col("flag") == "any_flag").sort("level")
        a.set_yticks(np.arange(len(levels)), [f"{lv} (n={m:,})" for lv, m in zip(levels, n["n_trajectories"].to_list())],
                     fontsize=7)
        a.invert_yaxis()
        a.set_xlabel("share of trajectories flagged")
        a.set_title(dom)
        a.legend(frameon=False, fontsize=8, loc="lower right")
    finish(fig, "Structural flag = tier reject or A1 p99 flag on structural / temporal / coordinate dimensions; signal "
                "flag = vendor RMSE or interior jerk above the stratum q95 (≈ 5 % by construction); boundary flag = peak "
                "a_tan or deceleration on an SG edge frame. Athlete-cluster bootstrap 95 % CI. PENDING interpretation.",
           "Q1 · Which quality defects occur where?")
    store.save_figure(fig, rec, {"rates": out[RATES.name]}, n=f"{int(r.filter(pl.col('flag') == 'any_flag')['n_trajectories'].sum()):,} trajectories")


def fig_agreement(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    d = out[AGREE.name].sort("domain", "stratum")
    fig, ax = figure(1, 2)
    a = ax[0, 0]
    if not d.height:
        empty_panel(a, "no agreement rows")
    else:
        labels = [f"{x} · {s}" for x, s in zip(d["domain"].to_list(), d["stratum"].to_list())]
        _hbar_ci(a, labels, d["kappa"].fill_null(np.nan), d["ci_lower"].fill_null(np.nan),
                 d["ci_upper"].fill_null(np.nan), cat[0])
        a.axvline(0, color=ink["muted"], ls="--", lw=1)
        a.set_yticks(np.arange(len(labels)), labels, fontsize=7)
        a.invert_yaxis()
        a.set_xlabel("Cohen's κ (structural flag vs signal flag)")
    b = ax[0, 1]
    tot = d.filter(pl.col("stratum") == "all")
    if not tot.height:
        empty_panel(b, "no pooled rows")
    else:
        cols = ["n_both", "n_structural_only", "n_signal_only"]
        x = np.arange(tot.height)
        bottom = np.zeros(tot.height)
        for j, c in enumerate(cols):
            v = tot[c].to_numpy().astype(float)
            b.bar(x, v, bottom=bottom, color=cat[j], label=c.replace("n_", ""))
            bottom += v
        b.set_xticks(x, tot["domain"].to_list())
        b.set_ylabel("flagged trajectories")
        b.legend(frameon=False, fontsize=8)
    finish(fig, "κ ≈ 0 means structural integrity and signal reliability flag DIFFERENT trajectories — they must be "
                "reported as separate dimensions, not one tier. Right: pooled composition of flagged trajectories. "
                "PENDING interpretation.", "Q2 · Are structural and signal quality the same thing?")
    store.save_figure(fig, rec, {"agreement": d}, n=f"{int(tot['n_trajectories'].sum()) if tot.height else 0:,} trajectories")


def fig_eligibility(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    e = out[ELIG.name]
    fig, ax = figure(1, 2, sharey=True)
    for k, (col, lab) in enumerate((("share_athletes_any", "athletes with ≥ 1 retained attempt"),
                                    ("share_athletes_repeat", "athletes with ≥ 2 retained attempts"))):
        a = ax[0, k]
        if not e.height:
            empty_panel(a, "no eligibility rows")
            continue
        base = e.filter(pl.col("scenario_order") == 0).sort("domain", "stratum", "drill")
        labels = [f"{d if d == s else f'{s} · {d}'} (A={n})" for s, d, n in zip(
            base["stratum"].to_list(), base["drill"].to_list(), base["n_athletes_any"].to_list())]
        for i, (name, _) in enumerate(SCENARIOS[1:], start=1):
            g = base.select("domain", "stratum", "drill").join(e.filter(pl.col("scenario_order") == i),
                                                               on=["domain", "stratum", "drill"], how="left")
            a.plot(g[col].fill_null(np.nan).to_numpy(), np.arange(g.height), "o", color=cat[i - 1], ms=5, label=name)
        a.set_yticks(np.arange(base.height), labels, fontsize=6)
        a.invert_yaxis()
        a.set_xlim(0, 1.02)
        a.set_xlabel(f"share of {lab} (vs S0)")
        a.legend(frameon=False, fontsize=8, loc="lower left")
    finish(fig, "Cumulative filters: S1 structural pass, S2 + signal pass, S3 + peak not on a filter-boundary frame. "
                "Right panel = athletes who still support a repeatability (ICC) estimate. A = athletes in S0. "
                "PENDING interpretation.", "Q3 · How many athletes survive each quality filter?")
    store.save_figure(fig, rec, {"eligibility": e}, n=f"{e.filter(pl.col('scenario_order') == 0)['n_athletes_any'].sum():,} athlete-drills")


def fig_by_group(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    r = out[RATES.name].filter(pl.col("facet").is_in(["cohort", "position"]) & (pl.col("flag") == "any_flag"))
    fig, ax = figure(1, 2)
    for k, facet in enumerate(("cohort", "position")):
        a = ax[0, k]
        d = r.filter(pl.col("facet") == facet).sort("domain", "level")
        if not d.height:
            empty_panel(a, f"no {facet} rows")
            continue
        labels = [f"{x} · {lv} (n={n:,})" for x, lv, n in zip(d["domain"].to_list(), d["level"].to_list(),
                                                              d["n_trajectories"].to_list())]
        _hbar_ci(a, labels, d["share"], d["ci_lower"].fill_null(np.nan), d["ci_upper"].fill_null(np.nan), cat[k])
        a.set_yticks(np.arange(len(labels)), labels, fontsize=7)
        a.invert_yaxis()
        a.set_xlabel("share of trajectories with any flag")
        a.set_title(facet)
    finish(fig, "Concentration of quality flags in cohorts and positions — a filter that removes one position or cohort "
                "more than others is a selection mechanism (Study I). Athlete-cluster bootstrap 95 % CI. "
                "PENDING interpretation.", "Q4 · Do quality filters remove some positions or cohorts more than others?")
    store.save_figure(fig, rec, {"rates": r}, n=f"{int(r['n_trajectories'].sum()):,} trajectory-facet rows")


def _rec(fid: str, name: str, q: str, insight: str, stat: str, ds: str, method: str) -> FigureRecord:
    return FigureRecord(figure_id=fid, name=name, study=STUDY, research_question=q, expected_insight=insight,
                        statistic=stat, dataset=ds, grain="trajectory (athlete-clustered)", filters="none", method=method)


FIGURES = [
    (_rec("Q-01", "Q_01_flag_rates_by_stratum", "Which quality defects occur in which drills?",
          "drills where derived signals are least trustworthy", "share + cluster CI", RATES.name,
          "flag shares"), fig_flag_rates),
    (_rec("Q-02", "Q_02_structural_vs_signal", "Are structural and signal quality distinct dimensions?",
          "whether a single tier can summarise quality", "Cohen's κ + cluster CI", AGREE.name, "2×2 agreement"),
     fig_agreement),
    (_rec("Q-03", "Q_03_athlete_eligibility", "How many athletes remain eligible under each filter?",
          "research sample cost of each filter", "athlete counts / shares", ELIG.name, "cumulative filters"),
     fig_eligibility),
    (_rec("Q-04", "Q_04_flags_by_cohort_position", "Do filters remove some cohorts / positions more?",
          "selection introduced by quality filtering", "share + cluster CI", RATES.name, "flag shares"), fig_by_group),
]
