"""Study C1 — event-detector sensitivity to threshold rule and smoothing (Phase 04 · M1 · Study 1.3, part 1).

Question     How much do event counts, rates and individual events depend on the detector's threshold rule
             (lenient / default / strict) and on the SG setting? M1 evidence (executed): turn counts collapse
             under the strict rule (Combine 1.07 → 0.34 → ≈ 0.02 per trajectory) — a detector property, not
             evidence that athletes rarely turn.
Scope        CONSISTENCY, not validity. Agreement between two detector configurations says nothing about
             agreement with the truth; that needs the human reference labels of Study C2.
Population   Study B's recomputation (all Combine attempts + deterministic NFL sample), every rule × setting
             (``events_by_smoothing``), exposure = valid tracked seconds per trajectory and setting.
Statistics   (1) exposure-normalised rates: events per valid second with athlete-cluster bootstrap CI
                 (ratio Σevents / Σseconds over athletes), events per trajectory, share of trajectories with
                 ≥ 1 event, median duration, share touching a segment edge;
             (2) count stability vs the reference (default rule, primary SG): Spearman of per-trajectory and
                 per-athlete rates, exact-count agreement, mean |count difference|, total-count ratio;
             (3) event-level agreement: one-to-one onset matching (|Δonset| ≤ tolerance, C2 matcher) →
                 share of reference events retained, share of alternative events that are new, onset / offset
                 shift. Comparisons: rule ∈ {lenient, strict} at primary SG; SG ∈ {sg5p2, sg9p2, sg11p3} at
                 default rule.

Command for the USER (code authored, NOT executed; needs Study B)::

    python -m nflbdb.research.phase04 --only C1
"""

from __future__ import annotations

import dataclasses
import zlib
from typing import Any

import numpy as np
import polars as pl

from ...data.config import Phase01Config
from ...data.utils import get_logger
from ...features.config import KinematicsConfig
from ...features.events import EVENT_TYPES
from ..contracts import TableContract
from ..studies.annotation import match_events
from ..studies.results import RESULTS, ResultCollector
from ..studies.stats import cluster_bootstrap_ratio, spearman
from ..studies.store import FigureRecord, StudyStore
from .config import EventStudyConfig, Phase04Config
from .smoothing import EVENTS_DS, FEATURES_DS

log = get_logger(__name__)
STUDY = "C1"
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
ALL = "ALL"

RATES = TableContract(
    "C1_event_rates", "Exposure-normalised event rates by domain × drill × event type × rule × SG setting.",
    {"domain": S, "drill": S, "event_type": S, "rule": S, "smoothing": S, "n_events": I, "n_trajectories": I,
     "n_athletes": I, "exposure_s": F, "rate_per_s": F, "rate_lower": F, "rate_upper": F, "events_per_trajectory": F,
     "share_traj_with_event": F, "median_duration_s": F, "share_touching_edge": F, "result_status": S},
    key=("domain", "drill", "event_type", "rule", "smoothing"), area="descriptive",
    docs={"drill": f"'{ALL}' = all drills of the domain pooled (exposure-weighted)",
          "rate_per_s": "Σ events / Σ valid seconds; CI = athlete-cluster bootstrap of the ratio"})
STABILITY = TableContract(
    "C1_count_stability", "Per-unit count / rate stability of each configuration vs the reference configuration.",
    {"domain": S, "drill": S, "event_type": S, "comparison": S, "factor": S, "level": S, "n_units": I,
     "n_athletes": I, "spearman_rate": F, "share_exact_count": F, "mean_abs_count_diff": F, "total_ratio": F,
     "result_status": S},
    key=("domain", "drill", "event_type", "comparison", "level"), area="statistical",
    docs={"comparison": "rule:<name> (primary SG) or sg:<label> (default rule) vs reference default/primary",
          "total_ratio": "Σ events (alternative) / Σ events (reference)"})
EVENT_AGREE = TableContract(
    "C1_event_agreement", "Event-level matching of each configuration against the reference configuration.",
    {"domain": S, "drill": S, "event_type": S, "comparison": S, "factor": S, "n_trajectories": I, "n_athletes": I,
     "n_ref": I, "n_alt": I, "n_matched": I, "share_ref_retained": F, "retained_lower": F, "retained_upper": F,
     "share_alt_new": F, "onset_shift_median_s": F, "onset_shift_mae_s": F, "offset_shift_median_s": F,
     "tolerance_s": F, "result_status": S},
    key=("domain", "drill", "event_type", "comparison"), area="statistical",
    docs={"share_ref_retained": "reference events with a same-type alternative event within ± tolerance (onset)",
          "share_alt_new": "alternative events without a reference partner"})
STUDY_RESULTS = dataclasses.replace(RESULTS, name="C1_results")
CONTRACTS = (RATES, STABILITY, EVENT_AGREE, STUDY_RESULTS)


def _seed(base: int, *parts: Any) -> int:
    return (base + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 32)


def comparisons(kcfg: KinematicsConfig) -> list[tuple[str, str, str, str]]:
    """(comparison label, factor, smoothing, rule) of every alternative configuration."""
    prim_s, prim_r = kcfg.smoothing.label, kcfg.primary_rule
    rules = sorted({r.name for et in EVENT_TYPES for r in kcfg.rules(et)} - {prim_r})
    sgs = [s.label for s in kcfg.sensitivity_smoothing if s.label != prim_s]
    return [(f"rule:{r}", "rule", prim_s, r) for r in rules] + [(f"sg:{s}", "smoothing", s, prim_r) for s in sgs]


# ============================================================================== pure analysis
def exposure(features: pl.DataFrame) -> pl.DataFrame:
    """Per trajectory × setting exposure (valid seconds) — every trajectory, including event-free ones."""
    return (features.filter(pl.col("feature") == "duration_valid_s")
            .select("domain", "traj_id", "nfl_id", "drill", "smoothing", pl.col("value").alias("exposure_s"))
            .filter(pl.col("exposure_s") > 0))


def unit_counts(events: pl.DataFrame, expo: pl.DataFrame) -> pl.DataFrame:
    """Per trajectory × setting × rule × event type: count (0 when none). Rules = rules present in ``events``."""
    rules = events.select("rule").unique()
    types = pl.DataFrame({"event_type": list(EVENT_TYPES)})
    grid = expo.join(rules, how="cross").join(types, how="cross")
    cnt = events.group_by("domain", "traj_id", "smoothing", "rule", "event_type").agg(
        pl.len().cast(pl.Int64).alias("n_events"), pl.col("duration_s").median().alias("_dur"),
        pl.col("touches_segment_edge").sum().cast(pl.Int64).alias("_edge"))
    return grid.join(cnt, on=["domain", "traj_id", "smoothing", "rule", "event_type"], how="left").with_columns(
        pl.col("n_events").fill_null(0), pl.col("_edge").fill_null(0))


def _with_all(df: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([df, df.with_columns(pl.lit(ALL).alias("drill"))])


def rate_table(uc: pl.DataFrame, events: pl.DataFrame, cfg: EventStudyConfig, seed: int) -> pl.DataFrame:
    keys = ["domain", "drill", "event_type", "rule", "smoothing"]
    dur = _with_all(events).group_by(keys).agg(pl.col("duration_s").median().alias("median_duration_s"))
    rows = []
    for k, g in _with_all(uc).group_by(keys, maintain_order=True):
        per = g.group_by("nfl_id").agg(pl.col("n_events").sum().alias("k"), pl.col("exposure_s").sum().alias("e"))
        ok = per.height >= cfg.min_athletes
        ci = (cluster_bootstrap_ratio(per["k"].to_numpy(), per["e"].to_numpy(), cfg.n_boot, _seed(seed, *k)) if ok
              else {"estimate": float(per["k"].sum() / per["e"].sum()) if per["e"].sum() > 0 else None,
                    "ci_lower": None, "ci_upper": None})
        n_ev = int(g["n_events"].sum())
        rows.append({**dict(zip(keys, k)), "n_events": n_ev, "n_trajectories": g.height, "n_athletes": per.height,
                     "exposure_s": float(g["exposure_s"].sum()), "rate_per_s": ci["estimate"],
                     "rate_lower": ci["ci_lower"], "rate_upper": ci["ci_upper"],
                     "events_per_trajectory": n_ev / g.height if g.height else None,
                     "share_traj_with_event": float((g["n_events"] > 0).mean()),
                     "share_touching_edge": float(g["_edge"].sum() / n_ev) if n_ev else None,
                     "result_status": "descriptive" if ok else "insufficient_data"})
    out = pl.DataFrame(rows, schema={c: t for c, t in RATES.columns.items() if c != "median_duration_s"}, strict=False)
    return out.join(dur, on=keys, how="left").select(list(RATES.columns)).sort(keys)


def stability_table(uc: pl.DataFrame, kcfg: KinematicsConfig, cfg: EventStudyConfig) -> pl.DataFrame:
    prim_s, prim_r = kcfg.smoothing.label, kcfg.primary_rule
    ref = uc.filter((pl.col("smoothing") == prim_s) & (pl.col("rule") == prim_r)).select(
        "domain", "traj_id", "nfl_id", "drill", "event_type", pl.col("n_events").alias("k0"),
        pl.col("exposure_s").alias("e0"))
    rows = []
    for label, factor, s, r in comparisons(kcfg):
        alt = uc.filter((pl.col("smoothing") == s) & (pl.col("rule") == r)).select(
            "domain", "traj_id", "event_type", "n_events", "exposure_s")
        pr = _with_all(ref.join(alt, on=["domain", "traj_id", "event_type"], how="inner"))
        for (domain, drill, et), g in pr.group_by("domain", "drill", "event_type", maintain_order=True):
            n_ath = g["nfl_id"].n_unique()
            k, k0 = g["n_events"].to_numpy().astype(float), g["k0"].to_numpy().astype(float)
            status = "exploratory" if n_ath >= cfg.min_athletes else "insufficient_data"
            rows.append({"domain": domain, "drill": drill, "event_type": et, "comparison": label, "factor": factor,
                         "level": "trajectory", "n_units": g.height, "n_athletes": n_ath,
                         "spearman_rate": spearman(k / g["exposure_s"].to_numpy(), k0 / g["e0"].to_numpy()),
                         "share_exact_count": float((k == k0).mean()), "mean_abs_count_diff": float(np.abs(k - k0).mean()),
                         "total_ratio": float(k.sum() / k0.sum()) if k0.sum() > 0 else None, "result_status": status})
            a = g.group_by("nfl_id").agg(pl.col("n_events").sum(), pl.col("exposure_s").sum(), pl.col("k0").sum(),
                                         pl.col("e0").sum())
            rows.append({"domain": domain, "drill": drill, "event_type": et, "comparison": label, "factor": factor,
                         "level": "athlete", "n_units": a.height, "n_athletes": a.height,
                         "spearman_rate": spearman((a["n_events"] / a["exposure_s"]).to_numpy(),
                                                   (a["k0"] / a["e0"]).to_numpy()),
                         "share_exact_count": None, "mean_abs_count_diff": None,
                         "total_ratio": float(a["n_events"].sum() / a["k0"].sum()) if a["k0"].sum() > 0 else None,
                         "result_status": status})
    return (pl.DataFrame(rows, schema=STABILITY.columns, strict=False) if rows else STABILITY.empty()).with_columns(
        pl.col("spearman_rate").fill_nan(None))


def _event_lists(ev: pl.DataFrame) -> dict[tuple[str, str], list[tuple[float, float]]]:
    out: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for tid, et, on, off in ev.sort("traj_id", "event_type", "t_start_s").select(
            "traj_id", "event_type", "t_start_s", "t_end_s").iter_rows():
        out.setdefault((tid, et), []).append((on, off))
    return out


def match_configs(ref_ev: pl.DataFrame, alt_ev: pl.DataFrame, units: pl.DataFrame, tol: float) -> pl.DataFrame:
    """Per trajectory × event type: n_ref, n_alt, n_matched and matched onset / offset shifts (alt − ref).
    ``units``: traj_id, nfl_id, domain, drill of every trajectory present in BOTH configurations."""
    R, A = _event_lists(ref_ev), _event_lists(alt_ev)
    info = {t: (n, d, dr) for t, n, d, dr in units.select("traj_id", "nfl_id", "domain", "drill").iter_rows()}
    rows = []
    for (tid, et) in sorted(set(R) | set(A)):
        if tid not in info:
            continue
        ref, alt = R.get((tid, et), []), A.get((tid, et), [])
        pairs, _, _ = match_events([(a, b, False) for a, b in ref], alt, tol)
        on = [alt[j][0] - ref[i][0] for i, j in pairs]
        off = [alt[j][1] - ref[i][1] for i, j in pairs]
        n, d, dr = info[tid]
        rows.append({"traj_id": tid, "nfl_id": n, "domain": d, "drill": dr, "event_type": et, "n_ref": len(ref),
                     "n_alt": len(alt), "n_matched": len(pairs), "onsets": on, "offsets": off})
    schema = {"traj_id": pl.String, "nfl_id": pl.Int64, "domain": pl.String, "drill": pl.String,
              "event_type": pl.String, "n_ref": pl.Int64, "n_alt": pl.Int64, "n_matched": pl.Int64,
              "onsets": pl.List(pl.Float64), "offsets": pl.List(pl.Float64)}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def event_agreement_table(events: pl.DataFrame, expo: pl.DataFrame, kcfg: KinematicsConfig, cfg: EventStudyConfig,
                          seed: int) -> pl.DataFrame:
    prim_s, prim_r = kcfg.smoothing.label, kcfg.primary_rule
    ref_ev = events.filter((pl.col("smoothing") == prim_s) & (pl.col("rule") == prim_r))
    rows = []
    for label, factor, s, r in comparisons(kcfg):
        alt_ev = events.filter((pl.col("smoothing") == s) & (pl.col("rule") == r))
        units = (expo.filter(pl.col("smoothing") == prim_s).select("traj_id", "nfl_id", "domain", "drill")
                 .join(expo.filter(pl.col("smoothing") == s).select("traj_id"), on="traj_id", how="inner"))
        m = _with_all(match_configs(ref_ev, alt_ev, units, cfg.match_tolerance_s))
        for (domain, drill, et), g in m.group_by("domain", "drill", "event_type", maintain_order=True):
            per = g.group_by("nfl_id").agg(pl.col("n_matched").sum(), pl.col("n_ref").sum())
            n_ref, n_alt, n_m = int(g["n_ref"].sum()), int(g["n_alt"].sum()), int(g["n_matched"].sum())
            ok = per.height >= cfg.min_athletes and n_ref > 0
            ci = (cluster_bootstrap_ratio(per["n_matched"].to_numpy(), per["n_ref"].to_numpy(), cfg.n_boot,
                                          _seed(seed, label, domain, drill, et)) if ok else
                  {"ci_lower": None, "ci_upper": None})
            on = np.array([x for lst in g["onsets"].to_list() for x in lst], float)
            off = np.array([x for lst in g["offsets"].to_list() for x in lst], float)
            rows.append({"domain": domain, "drill": drill, "event_type": et, "comparison": label, "factor": factor,
                         "n_trajectories": g["traj_id"].n_unique(), "n_athletes": per.height, "n_ref": n_ref,
                         "n_alt": n_alt, "n_matched": n_m, "share_ref_retained": n_m / n_ref if n_ref else None,
                         "retained_lower": ci["ci_lower"], "retained_upper": ci["ci_upper"],
                         "share_alt_new": (n_alt - n_m) / n_alt if n_alt else None,
                         "onset_shift_median_s": float(np.median(on)) if on.size else None,
                         "onset_shift_mae_s": float(np.abs(on).mean()) if on.size else None,
                         "offset_shift_median_s": float(np.median(off)) if off.size else None,
                         "tolerance_s": cfg.match_tolerance_s,
                         "result_status": "exploratory" if ok else "insufficient_data"})
    return pl.DataFrame(rows, schema=EVENT_AGREE.columns, strict=False) if rows else EVENT_AGREE.empty()


def c1_results(rates: pl.DataFrame, stab: pl.DataFrame, agree: pl.DataFrame, rc: ResultCollector) -> None:
    for r in rates.iter_rows(named=True):
        rc.add(analysis_id="C1.rate", grain="trajectory", n_athletes=r["n_athletes"], n_trajectories=r["n_trajectories"],
               effect_estimate=r["rate_per_s"], effect_size_type="events per valid second", unit="1/s",
               method="Σ events / Σ valid seconds", domain=r["domain"], stratum=r["drill"], feature=r["event_type"],
               outcome=f"{r['rule']}|{r['smoothing']}", ci_lower=r["rate_lower"], ci_upper=r["rate_upper"],
               ci_method="athlete-cluster bootstrap of the ratio", result_status=r["result_status"])
    for r in stab.filter(pl.col("level") == "athlete").iter_rows(named=True):
        rc.add(analysis_id="C1.rate_stability.athlete", grain="athlete", n_athletes=r["n_athletes"],
               n_trajectories=None, effect_estimate=r["spearman_rate"], effect_size_type="Spearman rho",
               method="per-athlete rate vs reference configuration", domain=r["domain"], stratum=r["drill"],
               feature=r["event_type"], outcome=r["comparison"], result_status=r["result_status"],
               limitations="detector consistency, not validity")
    for r in agree.iter_rows(named=True):
        rc.add(analysis_id="C1.event_agreement", grain="event", n_athletes=r["n_athletes"],
               n_trajectories=r["n_trajectories"], effect_estimate=r["share_ref_retained"],
               effect_size_type="share of reference events retained", method="one-to-one onset matching",
               domain=r["domain"], stratum=r["drill"], feature=r["event_type"], outcome=r["comparison"],
               ci_lower=r["retained_lower"], ci_upper=r["retained_upper"],
               ci_method="athlete-cluster bootstrap of the ratio", result_status=r["result_status"],
               limitations=f"tolerance ± {r['tolerance_s']} s; consistency, not validity")


# ============================================================================== orchestration
class EventSensitivityStudy:
    STUDY = STUDY

    def __init__(self, cfg: Phase01Config, pcfg: Phase04Config | None = None, store: StudyStore | None = None,
                 kcfg: KinematicsConfig | None = None, **_: Any):
        self.cfg = cfg
        self.pcfg = pcfg or Phase04Config()
        self.e = self.pcfg.events
        self.kcfg = kcfg or KinematicsConfig.from_tracking(cfg.tracking)
        self.store = store or StudyStore(cfg, sampled=self.pcfg.sample_athletes is not None, phase="phase_04")

    def compute(self) -> dict[str, pl.DataFrame]:
        events, feats = self.store.read(EVENTS_DS), self.store.read(FEATURES_DS)
        expo = exposure(feats)
        del feats
        uc = unit_counts(events, expo)
        seed = self.pcfg.seed
        rates = rate_table(uc, events, self.e, seed)
        stab = stability_table(uc, self.kcfg, self.e)
        agree = event_agreement_table(events, expo, self.kcfg, self.e, seed)
        rc = ResultCollector(STUDY, self.e.min_athletes)
        c1_results(rates, stab, agree, rc)
        return {RATES.name: rates, STABILITY.name: stab, EVENT_AGREE.name: agree, STUDY_RESULTS.name: rc.frame()}

    def fingerprint(self) -> str:
        sample = f" --sample-athletes {self.pcfg.sample_athletes}" if self.pcfg.sample_athletes is not None else ""
        b = self.store.require_run("B", "Study C1", f"python -m nflbdb.research.phase04 --only B{sample}")
        return self.store.fingerprint(STUDY, {**self.pcfg.section("events"), "kinematics": self.kcfg.params()},
                                      {"study_B": b.get("fingerprint")})

    def run(self, force: bool = False, figures: bool = True) -> dict[str, pl.DataFrame]:
        fp = self.fingerprint()
        current, reason = self.store.is_current(STUDY, fp, CONTRACTS)
        if current and not force:
            log.info("[phase04:C1] reuse (%s)", reason)
            out = self.load()
        else:
            log.info("[phase04:C1] compute (%s%s)", "forced; " if force else "", reason)
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
            fn(self.store, rec, out, self.kcfg)


# ============================================================================== figures
def _style():
    from ...data.reporting import CATEGORICAL, INK

    return CATEGORICAL, INK


PRIMARY_DRILLS = ("FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL", "nfl_play")


def _rate_panels(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame], kcfg: KinematicsConfig,
                 factor: str) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    r = out[RATES.name]
    if factor == "rule":
        r = r.filter(pl.col("smoothing") == kcfg.smoothing.label)
        levels, col = ["lenient", kcfg.primary_rule, "strict"], "rule"
    else:
        r = r.filter(pl.col("rule") == kcfg.primary_rule)
        levels, col = [s.label for s in kcfg.sensitivity_smoothing], "smoothing"
    drills = [d for d in PRIMARY_DRILLS if d in set(r["drill"].to_list())]
    fig, ax = figure(1, len(EVENT_TYPES), sharey=True)
    for k, et in enumerate(EVENT_TYPES):
        a = ax[0, k]
        g = r.filter(pl.col("event_type") == et)
        if not g.height or not drills:
            empty_panel(a, f"no {et} rows")
            continue
        for j, lv in enumerate(levels):
            h = pl.DataFrame({"drill": drills}).join(g.filter(pl.col(col) == lv), on="drill", how="left")
            y = np.arange(len(drills)) + (j - (len(levels) - 1) / 2) * 0.18
            est = h["rate_per_s"].fill_null(np.nan).to_numpy()
            lo, hi = h["rate_lower"].fill_null(np.nan).to_numpy(), h["rate_upper"].fill_null(np.nan).to_numpy()
            err = np.vstack([np.where(np.isfinite(lo), est - lo, 0), np.where(np.isfinite(hi), hi - est, 0)]).clip(0)
            a.errorbar(est, y, xerr=err, fmt="o", ms=4, capsize=2, color=cat[j], label=lv)
        a.set_yticks(np.arange(len(drills)), drills, fontsize=7)
        a.invert_yaxis()
        a.set_title(et, fontsize=10)
        a.set_xlabel("events per valid second")
    ax[0, -1].legend(frameon=False, fontsize=7, title=col, title_fontsize=7)
    what = "threshold rule (primary SG setting)" if factor == "rule" else "SG setting (default rule)"
    finish(fig, f"Exposure-normalised event rates under each {what}; athlete-cluster bootstrap 95 % CI of Σevents / "
                "Σseconds. Rates differ between configurations by design; the question is whether the SAME athletes "
                "remain high or low (C1_03) and whether events correspond to human-labelled events (C2). "
                "PENDING interpretation.", f"C1 · Event rates by {col}")
    store.save_figure(fig, rec, {"rates": r}, n=f"{int(r['n_trajectories'].max() or 0):,} trajectories (largest group)")


def fig_rates_rule(store, rec, out, kcfg):
    _rate_panels(store, rec, out, kcfg, "rule")


def fig_rates_smoothing(store, rec, out, kcfg):
    _rate_panels(store, rec, out, kcfg, "smoothing")


def fig_stability(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame], kcfg: KinematicsConfig) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    from ...data.reporting import SEQUENTIAL_BLUE
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    s = out[STABILITY.name].filter(pl.col("level") == "athlete")
    drills = [d for d in PRIMARY_DRILLS if d in set(s["drill"].to_list())]
    comps = [c for c, *_ in comparisons(kcfg)]
    fig, ax = figure(1, max(1, len(drills)), sharey=True)
    cmap = LinearSegmentedColormap.from_list("seqblue", list(SEQUENTIAL_BLUE))
    im = None
    for k, dr in enumerate(drills or ["none"]):
        a = ax[0, k]
        g = s.filter(pl.col("drill") == dr)
        if not g.height:
            empty_panel(a, f"no {dr}")
            continue
        m = np.full((len(EVENT_TYPES), len(comps)), np.nan)
        for r in g.iter_rows(named=True):
            if r["comparison"] in comps and r["spearman_rate"] is not None:
                m[EVENT_TYPES.index(r["event_type"]), comps.index(r["comparison"])] = r["spearman_rate"]
        im = a.imshow(np.clip(m, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    a.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                           color=ink["surface"] if m[i, j] > 0.6 else ink["primary"])
        a.set_xticks(np.arange(len(comps)), comps, rotation=45, ha="right", fontsize=7)
        a.set_yticks(np.arange(len(EVENT_TYPES)), EVENT_TYPES, fontsize=8)
        a.set_title(f"{dr} (A={int(g['n_athletes'].max())})", fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=ax[0, -1], shrink=0.7)
    finish(fig, "Spearman ρ between each athlete's event rate under an alternative configuration and under the reference "
                "(default rule, sg7p2). Low ρ: the athlete ordering by that event rate depends on the detector "
                "configuration, so the rate cannot be used as an athlete trait without validation. PENDING "
                "interpretation.", "C1 · Is the athlete ordering by event rate detector-dependent?")
    store.save_figure(fig, rec, {"stability": s}, n=f"{s.height} athlete-level comparisons")


def fig_event_agreement(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame],
                        kcfg: KinematicsConfig) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    e = out[EVENT_AGREE.name].filter(pl.col("drill") == ALL)
    domains = [d for d in ("combine", "nfl_sample") if d in set(e["domain"].to_list())] or ["combine"]
    comps = [c for c, *_ in comparisons(kcfg)]
    fig, ax = figure(len(domains), 2, sharey=True)
    for i, dom in enumerate(domains):
        g = e.filter(pl.col("domain") == dom)
        for k, (col, lab) in enumerate((("share_ref_retained", "reference events retained"),
                                        ("share_alt_new", "alternative events that are new"))):
            a = ax[i, k]
            if not g.height:
                empty_panel(a, f"no {dom}")
                continue
            for j, et in enumerate(EVENT_TYPES):
                h = pl.DataFrame({"comparison": comps}).join(g.filter(pl.col("event_type") == et), on="comparison",
                                                             how="left")
                y = np.arange(len(comps)) + (j - 1.5) * 0.18
                a.barh(y, h[col].fill_null(np.nan).to_numpy(), height=0.18, color=cat[j], label=et)
            a.set_yticks(np.arange(len(comps)), comps, fontsize=8)
            a.invert_yaxis()
            a.set_xlim(0, 1)
            a.set_xlabel(f"share of {lab}")
            a.set_title(dom, fontsize=9)
    ax[0, -1].legend(frameon=False, fontsize=7)
    finish(fig, "Event-level consistency with the reference configuration (default rule, sg7p2), all drills pooled: an "
                "event is retained when an alternative event of the same type starts within ± 0.3 s. Strict rules drop "
                "events; lenient rules add them; smoothing mainly shifts onsets. Consistency is NOT accuracy (see C2). "
                "PENDING interpretation.", "C1 · Do individual events survive a change of rule or smoothing?")
    store.save_figure(fig, rec, {"agreement": out[EVENT_AGREE.name]}, n=f"{int(e['n_ref'].sum()):,} reference events")


def fig_combine_vs_nfl(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame],
                       kcfg: KinematicsConfig) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    r = out[RATES.name].filter((pl.col("rule") == kcfg.primary_rule) & (pl.col("smoothing") == kcfg.smoothing.label))
    drills = [d for d in PRIMARY_DRILLS if d in set(r["drill"].to_list())]
    fig, ax = figure(1, 2)
    for k, (col, lab) in enumerate((("events_per_trajectory", "events per trajectory"),
                                    ("rate_per_s", "events per valid second"))):
        a = ax[0, k]
        if not drills:
            empty_panel(a, "no rows")
            continue
        for j, et in enumerate(EVENT_TYPES):
            h = pl.DataFrame({"drill": drills}).join(r.filter(pl.col("event_type") == et), on="drill", how="left")
            a.plot(h[col].fill_null(np.nan).to_numpy(), np.arange(len(drills)), "o", color=cat[j], ms=6, label=et)
        a.set_yticks(np.arange(len(drills)), drills, fontsize=8)
        a.invert_yaxis()
        a.set_xlabel(lab)
    ax[0, 0].legend(frameon=False, fontsize=7)
    finish(fig, "Same events, two denominators (default rule, sg7p2). Combine drills last 3–8 s and NFL player-plays "
                "longer, so per-trajectory counts mix exposure with movement content; per-second rates remove exposure "
                "but still mix task content (a shuttle REQUIRES reversals). Neither is a like-for-like Combine-vs-NFL "
                "comparison. PENDING interpretation.", "C1 · Why exposure normalisation matters")
    store.save_figure(fig, rec, {"rates": r}, n=f"{len(drills)} strata")


def _rec(fid: str, name: str, q: str, insight: str, stat: str, ds: str, grain: str) -> FigureRecord:
    return FigureRecord(figure_id=fid, name=name, study=STUDY, research_question=q, expected_insight=insight,
                        statistic=stat, dataset=ds, grain=grain, filters="none (all tiers)",
                        method="Study B recomputation, all rules × SG settings")


FIGURES = [
    (_rec("C1-01", "C1_01_rates_by_rule", "How do event rates depend on the threshold rule?",
          "detector sensitivity per event type", "rate + cluster CI", RATES.name, "trajectory"), fig_rates_rule),
    (_rec("C1-02", "C1_02_rates_by_smoothing", "How do event rates depend on smoothing?",
          "smoothing sensitivity of event detection", "rate + cluster CI", RATES.name, "trajectory"),
     fig_rates_smoothing),
    (_rec("C1-03", "C1_03_rate_stability", "Is the athlete ordering by event rate configuration-dependent?",
          "event rates usable as athlete traits", "Spearman ρ", STABILITY.name, "athlete"), fig_stability),
    (_rec("C1-04", "C1_04_event_agreement", "Do individual events survive configuration changes?",
          "event-level consistency", "retained / new shares", EVENT_AGREE.name, "event"), fig_event_agreement),
    (_rec("C1-05", "C1_05_exposure_normalisation", "Why compare rates per second, not per trajectory?",
          "exposure confounding of Combine vs NFL counts", "rates", RATES.name, "trajectory"), fig_combine_vs_nfl),
]
