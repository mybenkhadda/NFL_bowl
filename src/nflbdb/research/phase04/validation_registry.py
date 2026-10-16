"""Study V — Milestone 1 decision: evidence-based feature status registry (Phase 04 · M1 gate).

Every feature × domain × drill receives VALIDATED / PROVISIONAL / EXPERIMENTAL / REJECTED from SAVED
evidence only. Nothing is computed from frames here; a missing evidence table makes the corresponding
criterion ``missing`` (never ``pass``).

Criteria (value → state with the PROPOSED cut-offs of ``ValidationConfig``; the values are exported):

    smooth_rank        min over SG settings of the athlete-level Spearman CI lower bound      (Study B)
    smooth_overlap     min athlete-level top-decile overlap (n/a when ties dominate)            (Study B)
    smooth_magnitude   max trajectory-level median relative change                              (Study B)
    boundary           A3 observed / expected share of the peak on SG edge frames (raw peaks)   (Study A3)
    repeatability      min over settings of the ICC CI lower bound (Combine attempts)           (Study B)
    icc_setting_range  max − min ICC across settings                                             (Study B)
    event_smoothing    min athlete-level Spearman of event rates across SG settings             (Study C1)
    detector_f1        detector F1 CI lower bound vs human labels, EVAL split, random tasks     (Study C2)
    vendor             A2 relative MAE vs vendor speed (speed features only)                    (Study A2)

States: pass · marginal (between the cuts) · fail · missing (evidence absent) · not_applicable.
Status:  any critical fail → REJECTED; any major fail → EXPERIMENTAL; core smoothing evidence missing →
         EXPERIMENTAL; every applicable criterion pass → VALIDATED; otherwise PROVISIONAL.
Consequences by construction: event features cannot be VALIDATED before human labels exist (detector_f1
missing); NFL features cannot be VALIDATED before an NFL repeatability study (Milestone 2/P1) exists.
The registry is a rule-based PROPOSAL for scientific review, not an automatic verdict.

Command for the USER (code authored, NOT executed; after Q, B, C1 [and C2 evaluation])::

    python -m nflbdb.research.phase04 --only V
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import polars as pl

from ...data.config import Phase01Config
from ...data.utils import get_logger
from ..contracts import TableContract
from ..studies.quality import VENDOR as A2_VENDOR
from ..studies.results import RESULTS, ResultCollector
from ..studies.signal_reliability import BOUNDARY as A3_BOUNDARY
from ..studies.store import FigureRecord, StudyStore
from .annotation_design import PERFORMANCE as C2_PERF
from .config import STATUSES, Phase04Config, ValidationConfig
from .events import STABILITY as C1_STABILITY
from .smoothing import AGREE, BOUNDARY_PEAK, FEATURES, REPEAT, stratum_label

log = get_logger(__name__)
STUDY = "V"
S, I, F = pl.String(), pl.Int64(), pl.Float64()
CRITERIA = ("smooth_rank", "smooth_overlap", "smooth_magnitude", "boundary", "repeatability", "icc_setting_range",
            "event_smoothing", "detector_f1", "vendor")
STATES = ("pass", "marginal", "fail", "missing", "not_applicable")
RANK = {s: i for i, s in enumerate(("REJECTED", "EXPERIMENTAL", "PROVISIONAL", "VALIDATED"))}
SPEED_FEATURES = ("peak_speed", "peak_speed_interior", "p95_speed")
EVENT_OF = {f: f.split("_")[1] for f, (_, fam) in FEATURES.items() if fam in ("event_count", "event_rate")}

REGISTRY = TableContract(
    "V_feature_validation_registry", "Feature × domain × drill: criterion values, states and proposed status.",
    {"domain": S, "stratum": S, "drill": S, "feature": S, "family": S, "unit": S, "n_athletes": I,
     **{f"value_{c}": F for c in CRITERIA}, **{f"state_{c}": S for c in CRITERIA}, "status": S, "status_reason": S,
     "decision_basis": S},
    key=("domain", "stratum", "drill", "feature"), area="reliability",
    docs={"status": f"one of {STATUSES}", "decision_basis": "rule-based proposal (ValidationConfig) — requires review",
          "stratum": "Combine drill_type · NFL 'nfl_play' — part of the grain (same grain as B_smoothing_agreement)",
          "value_event_smoothing": "Study C1 pools drill_types within a canonical drill: for a drill shared by "
                                   "several position groups this value is the pooled-drill estimate"})
SUMMARY = TableContract(
    "V_feature_status_summary", "Roll-up per domain × feature over the primary strata (worst status).",
    {"domain": S, "feature": S, "family": S, "status": S, "primary_strata": S, "n_drills": I, "n_validated": I,
     "n_provisional": I, "n_experimental": I, "n_rejected": I, "limiting_criteria": S},
    key=("domain", "feature"), area="reliability")
STUDY_RESULTS = dataclasses.replace(RESULTS, name="V_results")
CONTRACTS = (REGISTRY, SUMMARY, STUDY_RESULTS)


# ============================================================================== pure decision logic
def grade(value: float | None, pass_at: float, fail_at: float, higher_better: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "missing"
    if higher_better:
        return "pass" if value >= pass_at else "fail" if value < fail_at else "marginal"
    return "pass" if value <= pass_at else "fail" if value > fail_at else "marginal"


def decide(states: dict[str, str], severity: dict[str, str]) -> tuple[str, str]:
    """(status, reason) from criterion states. See module docstring for the rule."""
    fails = {c: severity.get(c, "minor") for c, s in states.items() if s == "fail"}
    if any(v == "critical" for v in fails.values()):
        return "REJECTED", "critical fail: " + ", ".join(sorted(c for c, v in fails.items() if v == "critical"))
    if any(v == "major" for v in fails.values()):
        return "EXPERIMENTAL", "major fail: " + ", ".join(sorted(c for c, v in fails.items() if v == "major"))
    if states.get("smooth_rank") in (None, "missing"):
        return "EXPERIMENTAL", "core smoothing evidence missing"
    open_ = sorted(c for c, s in states.items() if s in ("marginal", "missing", "fail"))
    if not open_:
        return "VALIDATED", "all applicable criteria pass"
    return "PROVISIONAL", "open: " + ", ".join(f"{c}={states[c]}" for c in open_)


def _worst(values: list[float]) -> float | None:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return min(v) if v else None


def _max(values: list[float]) -> float | None:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return max(v) if v else None


def evidence(agree: pl.DataFrame, repeat: pl.DataFrame | None, boundary: pl.DataFrame | None,
             stability: pl.DataFrame | None, perf: pl.DataFrame | None, vendor: pl.DataFrame | None,
             cfg: ValidationConfig, primary_sg: str, primary_rule: str, tol: float) -> pl.DataFrame:
    """One row per domain × stratum × drill × feature with criterion values (None = evidence absent)."""
    units = (agree.filter(pl.col("feature").is_in([f for f, (_, fam) in FEATURES.items() if fam != "exposure"]))
             .select("domain", "stratum", "drill", "feature").unique().sort("domain", "stratum", "drill", "feature"))
    ath = agree.filter(pl.col("level") == "athlete")
    trj = agree.filter(pl.col("level") == "trajectory")
    rows = []
    for u in units.iter_rows(named=True):
        # the unit is the Study B grain; without the stratum term a drill shared by several drill_types would
        # take the worst value across position groups
        key = ((pl.col("domain") == u["domain"]) & (pl.col("stratum") == u["stratum"]) & (pl.col("drill") == u["drill"])
               & (pl.col("feature") == u["feature"]))
        a, t = ath.filter(key), trj.filter(key)
        tie = bool(a.height and (a["modal_share"].fill_null(0).max() or 0) >= 0.5)
        r: dict[str, Any] = {**u, "family": FEATURES[u["feature"]][1], "unit": FEATURES[u["feature"]][0],
                             "n_athletes": int(a["n_units"].max()) if a.height else 0,
                             "value_smooth_rank": _worst(a["spearman_lower"].to_list()) if a.height else None,
                             "value_smooth_overlap": None if tie else (_worst(a["top_overlap"].to_list()) if a.height
                                                                       else None),
                             "_tie": tie,
                             "value_smooth_magnitude": _max(t["median_rel_diff"].to_list()) if t.height else None}
        # boundary (raw peak features only; A3 stratum = Combine drill_type / NFL 'nfl_play', domain names equal)
        r["value_boundary"] = None
        if boundary is not None and u["feature"] in BOUNDARY_PEAK:
            b = boundary.filter((pl.col("domain") == u["domain"]) & (pl.col("stratum") == u["stratum"])
                                & (pl.col("peak") == BOUNDARY_PEAK[u["feature"]]))
            r["value_boundary"] = b["ratio_observed_expected"][0] if b.height else None
        # repeatability (Combine)
        r["value_repeatability"] = r["value_icc_setting_range"] = None
        if repeat is not None and u["domain"] == "combine":
            rp = repeat.filter((pl.col("stratum") == u["stratum"]) & (pl.col("drill") == u["drill"])
                               & (pl.col("feature") == u["feature"]) & pl.col("icc1").is_not_null())
            if rp.height:
                r["value_repeatability"] = _worst(rp["ci_lower"].to_list())
                r["value_icc_setting_range"] = float(rp["icc1"].max() - rp["icc1"].min())
        # event features: smoothing stability of the rate (C1) and detector accuracy (C2)
        r["value_event_smoothing"] = r["value_detector_f1"] = None
        if u["feature"] in EVENT_OF:
            et = EVENT_OF[u["feature"]]
            if stability is not None:
                s = stability.filter((pl.col("domain") == u["domain"]) & (pl.col("drill") == u["drill"])
                                     & (pl.col("event_type") == et) & (pl.col("factor") == "smoothing")
                                     & (pl.col("level") == "athlete"))
                r["value_event_smoothing"] = _worst(s["spearman_rate"].to_list()) if s.height else None
            if perf is not None and perf.height:
                p = perf.filter((pl.col("split") == "eval") & (pl.col("selection_set") == "random")
                                & (pl.col("label_set") == "all_labels") & (pl.col("smoothing") == primary_sg)
                                & (pl.col("detector_rule") == primary_rule) & (pl.col("tolerance_s") == tol)
                                & (pl.col("event_class") == et) & (pl.col("facet") == "drill"))
                p = (p.filter(pl.col("level") == u["stratum"]) if u["domain"] == "combine"
                     else p.filter(pl.col("level").str.starts_with("nfl_play:")))
                r["value_detector_f1"] = _worst(p["f1_lower"].to_list()) if p.height else None
        # vendor agreement (speed features)
        r["value_vendor"] = None
        if vendor is not None and u["feature"] in SPEED_FEATURES:
            v = vendor.filter((pl.col("domain") == u["domain"]) & (
                ((pl.col("facet") == "stratum") & (pl.col("level") == u["stratum"])) if u["domain"] == "combine"
                else (pl.col("facet") == "all")))
            r["value_vendor"] = v["rel_mae"][0] if v.height else None
        rows.append(r)
    schema = {"domain": S, "stratum": S, "drill": S, "feature": S, "family": S, "unit": S, "n_athletes": I,
              **{f"value_{c}": F for c in CRITERIA}, "_tie": pl.Boolean()}
    # explicit schema: an evidence column can be None for the first rows (absent table) and numeric later
    return pl.DataFrame([{k: r.get(k) for k in schema} for r in rows], schema=schema)


def states_for(r: dict[str, Any], cfg: ValidationConfig) -> dict[str, str]:
    fam, dom = r["family"], r["domain"]
    st = {"smooth_rank": grade(r["value_smooth_rank"], cfg.rho_pass, cfg.rho_fail),
          "smooth_overlap": "not_applicable" if r.get("_tie") else grade(r["value_smooth_overlap"], cfg.overlap_pass,
                                                                         cfg.overlap_fail),
          "smooth_magnitude": grade(r["value_smooth_magnitude"], cfg.rel_diff_pass, cfg.rel_diff_fail, False)}
    st["boundary"] = (grade(r["value_boundary"], cfg.boundary_ratio_pass, cfg.boundary_ratio_fail, False)
                      if r["feature"] in BOUNDARY_PEAK else "not_applicable")
    if dom == "combine":
        st["repeatability"] = grade(r["value_repeatability"], cfg.icc_pass, cfg.icc_fail)
        st["icc_setting_range"] = grade(r["value_icc_setting_range"], cfg.icc_range_pass, 2 * cfg.icc_range_pass, False)
    else:   # NFL within-athlete reliability needs context-matched plays (Milestone 2 / P1) — evidence absent
        st["repeatability"], st["icc_setting_range"] = "missing", "not_applicable"
    if fam in ("event_count", "event_rate"):
        st["event_smoothing"] = grade(r["value_event_smoothing"], cfg.rho_pass, cfg.rho_fail)
        st["detector_f1"] = grade(r["value_detector_f1"], cfg.f1_pass, cfg.f1_fail)
    else:
        st["event_smoothing"] = st["detector_f1"] = "not_applicable"
    st["vendor"] = (("pass" if r["value_vendor"] <= cfg.vendor_rel_mae_pass else "marginal")
                    if r["feature"] in SPEED_FEATURES and r["value_vendor"] is not None
                    else "missing" if r["feature"] in SPEED_FEATURES else "not_applicable")
    return st


def registry(ev: pl.DataFrame, cfg: ValidationConfig) -> pl.DataFrame:
    sev = dict(cfg.severity)
    rows = []
    for r in ev.iter_rows(named=True):
        st = states_for(r, cfg)
        status, reason = decide({k: v for k, v in st.items() if v != "not_applicable"}, sev)
        rows.append({**{k: r[k] for k in ("domain", "stratum", "drill", "feature", "family", "unit", "n_athletes")},
                     **{f"value_{c}": r.get(f"value_{c}") for c in CRITERIA}, **{f"state_{c}": st[c] for c in CRITERIA},
                     "status": status, "status_reason": reason,
                     "decision_basis": "rule-based proposal (ValidationConfig, proposed cut-offs) — requires review"})
    return pl.DataFrame(rows, schema=REGISTRY.columns, strict=False) if rows else REGISTRY.empty()


def summary(reg: pl.DataFrame, cfg: ValidationConfig) -> pl.DataFrame:
    rows = []
    for (domain, feature), g in reg.group_by("domain", "feature", maintain_order=True):
        prim = g.filter(pl.col("drill").is_in(list(cfg.primary_strata)))
        basis = prim if prim.height else g
        worst = min(basis["status"].to_list(), key=lambda s: RANK[s])
        lim = sorted({c for r in basis.filter(pl.col("status") == worst).iter_rows(named=True)
                      for c in CRITERIA if r[f"state_{c}"] in ("fail", "marginal", "missing")})
        cnt = {s: int((g["status"] == s).sum()) for s in STATUSES}
        rows.append({"domain": domain, "feature": feature, "family": g["family"][0], "status": worst,
                     "primary_strata": ", ".join(sorted(basis["drill"].to_list())), "n_drills": g.height,
                     "n_validated": cnt["VALIDATED"], "n_provisional": cnt["PROVISIONAL"],
                     "n_experimental": cnt["EXPERIMENTAL"], "n_rejected": cnt["REJECTED"],
                     "limiting_criteria": ", ".join(lim)})
    return (pl.DataFrame(rows, schema=SUMMARY.columns) if rows else SUMMARY.empty()).sort("domain", "feature")


# ============================================================================== orchestration
class ValidationRegistryStudy:
    STUDY = STUDY

    def __init__(self, cfg: Phase01Config, pcfg: Phase04Config | None = None, store: StudyStore | None = None,
                 store03: StudyStore | None = None, kcfg=None, **_: Any):
        from ...features.config import KinematicsConfig

        self.cfg = cfg
        self.pcfg = pcfg or Phase04Config()
        self.v = self.pcfg.validation
        self.kcfg = kcfg or KinematicsConfig.from_tracking(cfg.tracking)
        sampled = self.pcfg.sample_athletes is not None
        self.store = store or StudyStore(cfg, sampled=sampled, phase="phase_04")
        self.store03 = store03 or StudyStore(cfg, sampled=sampled)
        # C2 labels are never sampled (annotation lives in the full namespace)
        self.store_c2 = StudyStore(cfg, phase="phase_04")

    @staticmethod
    def _opt(store: StudyStore, c: TableContract) -> pl.DataFrame | None:
        try:
            return store.read(c)
        except FileNotFoundError:
            log.warning("[phase04:V] evidence %s absent — criteria using it are 'missing'", c.name)
            return None

    def compute(self) -> dict[str, pl.DataFrame]:
        # Study B is mandatory (core smoothing evidence): never read tables left behind by a failed B run.
        # C1 / C2 / A2 / A3 stay optional by design — their criteria become 'missing', never 'pass'.
        sample = f" --sample-athletes {self.pcfg.sample_athletes}" if self.pcfg.sample_athletes is not None else ""
        self.store.require_run("B", "Study V", f"python -m nflbdb.research.phase04 --only B{sample}")
        ev = evidence(self.store.read(AGREE), self._opt(self.store, REPEAT), self._opt(self.store03, A3_BOUNDARY),
                      self._opt(self.store, C1_STABILITY), self._opt(self.store_c2, C2_PERF),
                      self._opt(self.store03, A2_VENDOR), self.v, self.kcfg.smoothing.label, self.kcfg.primary_rule,
                      self.pcfg.annotation.primary_tolerance_s)
        reg = registry(ev, self.v)
        rc = ResultCollector(STUDY, 1)
        for r in reg.iter_rows(named=True):
            rc.add(analysis_id="V.status", grain="athlete", n_athletes=r["n_athletes"], n_trajectories=None,
                   effect_estimate=float(RANK[r["status"]]), effect_size_type="status rank (0 = REJECTED … 3 = VALIDATED)",
                   method="rule-based criteria (ValidationConfig)", domain=r["domain"],
                   stratum=stratum_label(r["stratum"], r["drill"]), feature=r["feature"],
                   result_status="descriptive", limitations=r["status_reason"])
        return {REGISTRY.name: reg, SUMMARY.name: summary(reg, self.v), STUDY_RESULTS.name: rc.frame()}

    def run(self, force: bool = True, figures: bool = True) -> dict[str, pl.DataFrame]:
        """Always recomputed (cheap; reads saved tables) so a new evidence table is never ignored."""
        res = self.compute()
        out = {c.name: self.store.write(res[c.name], c, STUDY) for c in CONTRACTS}
        self.store.record_run(STUDY, self.store.fingerprint(STUDY, self.pcfg.section("validation"), {}), CONTRACTS)
        if figures:
            self.figures(out)
        return out

    def load(self) -> dict[str, pl.DataFrame]:
        return {c.name: self.store.read(c) for c in CONTRACTS}

    def figures(self, out: dict[str, pl.DataFrame]) -> None:
        for rec, fn in FIGURES:
            fn(self.store, rec, out, self.v)


# ============================================================================== figures
STATUS_COLOR_KEYS = {"VALIDATED": "good", "PROVISIONAL": "warning", "EXPERIMENTAL": "serious", "REJECTED": "critical"}
STATE_SHADE = {"pass": 0, "marginal": 1, "missing": 2, "fail": 3, "not_applicable": 4}


def fig_status_matrix(store: StudyStore, rec: FigureRecord, out: dict, cfg: ValidationConfig) -> None:
    from matplotlib.colors import ListedColormap

    from ...data.reporting import INK, STATUS
    from ..figkit import empty_panel, figure, finish

    reg = out[REGISTRY.name].filter(pl.col("drill").is_in(list(cfg.primary_strata)))
    fig, ax = figure(1, 1, size="tall")
    a = ax[0, 0]
    if not reg.height:
        empty_panel(a, "no registry rows")
    else:
        cols = [(d, s) for d, s in reg.select("domain", "drill").unique().sort("domain", "drill").rows()]
        feats = sorted(reg["feature"].unique().to_list(), key=lambda f: (list(FEATURES).index(f), f))
        m = np.full((len(feats), len(cols)), np.nan)
        for r in reg.iter_rows(named=True):
            m[feats.index(r["feature"]), cols.index((r["domain"], r["drill"]))] = RANK[r["status"]]
        cmap = ListedColormap([STATUS[STATUS_COLOR_KEYS[s]] for s in ("REJECTED", "EXPERIMENTAL", "PROVISIONAL",
                                                                        "VALIDATED")])
        a.imshow(m, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    a.text(j, i, ("REJ", "EXP", "PROV", "VAL")[int(m[i, j])], ha="center", va="center", fontsize=6,
                           color=INK["primary"])
        a.set_yticks(np.arange(len(feats)), feats, fontsize=7)
        a.set_xticks(np.arange(len(cols)), [f"{d}\n{s}" for d, s in cols], fontsize=7)
    finish(fig, "Proposed status per feature and primary stratum from saved evidence (Studies A2, A3, B, C1, C2). "
                "Rule: critical fail → REJECTED; major fail or missing core evidence → EXPERIMENTAL; every applicable "
                "criterion pass → VALIDATED; otherwise PROVISIONAL. Cut-offs are proposed (ValidationConfig). Event "
                "features cannot be VALIDATED without human labels. PENDING review.",
           "V1 · Milestone 1 gate — which measurements can be trusted?")
    store.save_figure(fig, rec, {"registry": reg}, n=f"{reg.height} feature-strata")


def fig_criteria(store: StudyStore, rec: FigureRecord, out: dict, cfg: ValidationConfig) -> None:
    from matplotlib.colors import ListedColormap

    from ...data.reporting import INK, STATUS
    from ..figkit import empty_panel, figure, finish

    reg = out[REGISTRY.name].filter(pl.col("drill").is_in(list(cfg.primary_strata)))
    drills = [d for d in cfg.primary_strata if d in set(reg["drill"].to_list())]
    fig, ax = figure(1, max(1, len(drills)), sharey=True, size="wide")
    cmap = ListedColormap([STATUS["good"], STATUS["warning"], INK["axis"], STATUS["critical"], INK["surface"]])
    feats = sorted(reg["feature"].unique().to_list(), key=lambda f: (list(FEATURES).index(f), f)) if reg.height else []
    for k, dr in enumerate(drills or ["none"]):
        a = ax[0, k]
        g = reg.filter(pl.col("drill") == dr)
        if not g.height:
            empty_panel(a, "no rows")
            continue
        m = np.full((len(feats), len(CRITERIA)), np.nan)
        for r in g.iter_rows(named=True):
            for j, c in enumerate(CRITERIA):
                m[feats.index(r["feature"]), j] = STATE_SHADE[r[f"state_{c}"]]
        a.imshow(m, cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto")
        a.set_xticks(np.arange(len(CRITERIA)), CRITERIA, rotation=60, ha="right", fontsize=6)
        a.set_yticks(np.arange(len(feats)), feats, fontsize=6)
        a.set_title(dr, fontsize=9)
    finish(fig, "Criterion states per feature (green pass · amber marginal · grey missing evidence · red fail · white not "
                "applicable). Values behind each state are in V_feature_validation_registry. PENDING review.",
           "V2 · Which evidence limits each feature?")
    store.save_figure(fig, rec, {"registry": reg}, n=f"{len(feats)} features")


def _rec(fid: str, name: str, q: str, insight: str, stat: str, publication: bool = False) -> FigureRecord:
    return FigureRecord(figure_id=fid, name=name, study=STUDY, research_question=q, expected_insight=insight,
                        statistic=stat, dataset=REGISTRY.name, grain="feature × drill", filters="primary strata",
                        method="rule-based criteria", publication=publication)


FIGURES = [
    (_rec("V-01", "V_01_feature_status_matrix", "Which features pass the Milestone 1 measurement-validity gate?",
          "trusted feature shortlist", "status", publication=True), fig_status_matrix),
    (_rec("V-02", "V_02_criterion_states", "Which evidence limits each feature?", "next validation work",
          "criterion states"), fig_criteria),
]
