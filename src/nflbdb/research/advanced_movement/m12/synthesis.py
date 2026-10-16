"""12G — advanced method comparison and evidence synthesis. Reads persisted artifacts only (plus ONE common-cohort
tournament computed in the notebook with the shared ladder).

Every track writes ``m12<x>_comparisons`` in ONE schema (``COMPARISON_COLUMNS``) through ``comparison_row`` and
``m12<x>_evidence`` in the ``advanced_evaluation.EVIDENCE_COLUMNS`` schema. 12G concatenates them; it never re-derives a
number a track already persisted and never pools unrelated outcomes into one score (no naive meta-analysis).

The six declared experiments (brief §13 G3)
    1 scalar vs functional features              (12A)    4 individual drills vs multi-view latent traits   (12D)
    2 naive vs measurement-error correction      (12B)    5 feature distance vs conditional OT              (12E)
    3 maximum envelope vs distributional target  (12C)    6 physical features vs shared neural encoders     (12F)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from . import advanced_evaluation as ae
from .config import EXECUTED, INCOMPLETE, INCONCLUSIVE, PENDING, REJECTED, SUPPORTED, TRACKS

EXPERIMENTS = {1: ("12A", "Scalar features versus functional features"), 2: ("12B", "Naive features versus measurement-error correction"),
               3: ("12C", "Maximum envelope versus distributional capacity"), 4: ("12D", "Individual drills versus multi-view latent traits"),
               5: ("12E", "Feature distance versus conditional optimal transport"), 6: ("12F", "Physical features versus shared neural representations")}
COMPARISON_COLUMNS = ("track", "experiment", "comparison", "metric", "better", "estimate", "lo", "hi", "n_athletes", "cohort", "evaluation_target",
                      "training_regime", "primary", "verdict", "source_table", "run_mode")
CATEGORIES = ("measurement_reliability", "physical_interpretability", "additional_information", "cross_domain_matching", "nfl_physical_prediction",
              "football_outcome_transfer", "computational_practicality")
TRACK_TABLES = {"12C": "m12c", "12B": "m12b", "12A": "m12a", "12D": "m12d", "12E": "m12e", "12F": "m12f"}


def comparison_row(track: str, comparison: str, metric: str, estimate: Any, lo: Any, hi: Any, n: Any, evaluation_target: str, source_table: str,
                   run_mode: str, experiment: int | None = None, better: str = "positive", cohort: str = "", training_regime: str = "",
                   primary: bool = False) -> dict[str, Any]:
    f = lambda v: float(v) if v is not None and np.isfinite(float(v)) else None
    est, l, h = f(estimate), f(lo), f(hi)
    return {"track": track, "experiment": experiment, "comparison": comparison, "metric": metric, "better": better, "estimate": est, "lo": l, "hi": h,
            "n_athletes": int(n) if n is not None else None, "cohort": cohort, "evaluation_target": evaluation_target, "training_regime": training_regime,
            "primary": bool(primary), "verdict": ae.reading(l if l is not None else np.nan, h if h is not None else np.nan, better), "source_table": source_table,
            "run_mode": run_mode}


def comparison_frame(rows: Sequence[dict[str, Any]]) -> pl.DataFrame:
    schema = {"track": pl.String, "experiment": pl.Int64, "comparison": pl.String, "metric": pl.String, "better": pl.String, "estimate": pl.Float64,
              "lo": pl.Float64, "hi": pl.Float64, "n_athletes": pl.Int64, "cohort": pl.String, "evaluation_target": pl.String, "training_regime": pl.String,
              "primary": pl.Boolean, "verdict": pl.String, "source_table": pl.String, "run_mode": pl.String}
    return pl.DataFrame([{k: r.get(k) for k in COMPARISON_COLUMNS} for r in rows], schema=schema)


# ============================================================================== collection
def _read(store: Any, name: str) -> pl.DataFrame | None:
    return store.read_frame(name, "12G") if store.has(name) else None


def collect(store: Any, manifest_of: Any) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(track status, all comparisons, all evidence rows) from whatever is executed in this run mode."""
    st, comps, evid = [], [], []
    for t in TRACKS:
        if t.track in ("12.0", "12G"):
            continue
        m = manifest_of(store, t.track)
        pre = TRACK_TABLES[t.track]
        c_, e_ = _read(store, f"{pre}_comparisons"), _read(store, f"{pre}_evidence")
        st.append({"track": t.track, "status": (m or {}).get("status", PENDING), "finished_at": (m or {}).get("finished_at"),
                   "n_comparisons": 0 if c_ is None else c_.height, "n_evidence": 0 if e_ is None else e_.height})
        if c_ is not None and c_.height:
            comps.append(c_)
        if e_ is not None and e_.height:
            evid.append(e_)
    return (pl.DataFrame(st), pl.concat(comps, how="diagonal_relaxed") if comps else comparison_frame([]),
            pl.concat(evid, how="diagonal_relaxed") if evid else ae.evidence_frame([]))


def experiment_table(comps: pl.DataFrame, status: pl.DataFrame) -> pl.DataFrame:
    """One block per declared experiment: its comparisons (identical-cohort rows only, as each track persisted them), or an
    explicit ``execution_pending`` row when the track has not run."""
    rows = []
    for k, (tr, title) in EXPERIMENTS.items():
        sub = comps.filter(pl.col("experiment") == k) if comps.height else comps
        if sub.is_empty():
            s = status.filter(pl.col("track") == tr)["status"]
            rows.append({"experiment": k, "title": title, "track": tr, "comparison": "—", "metric": "—", "estimate": None, "lo": None, "hi": None,
                         "n_athletes": None, "verdict": s[0] if s.len() else PENDING, "training_regime": "", "primary": False})
            continue
        rows += [{"experiment": k, "title": title, **{c: r[c] for c in ("track", "comparison", "metric", "estimate", "lo", "hi", "n_athletes", "verdict",
                                                                          "training_regime", "primary")}} for r in sub.iter_rows(named=True)]
    return pl.DataFrame(rows, infer_schema_length=None)


def evidence_matrix(evid: pl.DataFrame, comps: pl.DataFrame) -> pl.DataFrame:
    """Method × category status (text labels; never a combined score). Categories without evidence are ``not_tested``."""
    rows = []
    for r in evid.iter_rows(named=True):
        m = r["method_id"]
        cat = {c: "not_tested" for c in CATEGORIES}
        if r.get("reliability") is not None:
            lo = r.get("reliability_lo")
            cat["measurement_reliability"] = "supported" if r["reliability"] >= 0.7 and (lo is None or lo >= 0.5) else "inconclusive" if r["reliability"] >= 0.4 else "rejected"
        if r.get("physical_interpretation"):
            cat["physical_interpretability"] = "descriptive"
        if r.get("stopwatch_r2") is not None:
            cat["additional_information"] = "supported" if r["stopwatch_r2"] < 0.5 else "inconclusive" if r["stopwatch_r2"] < 0.8 else "rejected"
        if r.get("cross_domain_lo") is not None and r.get("cross_domain_hi") is not None:
            cat["cross_domain_matching"] = ae.reading(r["cross_domain_lo"] - (0.5 if "auc" in str(r.get("output", "")).lower() else 0.0),
                                                      r["cross_domain_hi"] - (0.5 if "auc" in str(r.get("output", "")).lower() else 0.0))
        if r.get("delta_lo") is not None and r.get("delta_hi") is not None:
            cat["nfl_physical_prediction"] = ae.reading(r["delta_lo"], r["delta_hi"])
        cost = str(r.get("compute_cost") or "")
        cat["computational_practicality"] = "supported" if cost in ("low", "seconds", "minutes") else "inconclusive" if cost in ("medium", "tens of minutes") else (
            "rejected" if cost in ("high", "hours") else "not_tested")
        rows += [{"method_id": m, "track": r["track"], "category": k, "status": v} for k, v in cat.items()]
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(schema={"method_id": pl.String, "track": pl.String, "category": pl.String, "status": pl.String})


# ============================================================================== cohort-size sensitivity of the tournament
def cohort_size_sensitivity(df: pl.DataFrame, target: str, ref: ae.Model, mod: ae.Model, fracs: Sequence[float], reps: int, c: Any, seed: int) -> pl.DataFrame:
    """Δ RMSE (mod vs ref) on random athlete subsamples of the common cohort (folds kept), ``reps`` per fraction."""
    rows = []
    d = df.filter(pl.col(target).is_not_null())
    rng = np.random.default_rng(seed)
    for f in fracs:
        for r in range(reps):
            ids = rng.choice(d["nfl_id"].to_numpy(), max(20, int(f * d.height)), replace=False)
            sub = d.filter(pl.col("nfl_id").is_in(ids.tolist()))
            res = ae.oof_ladder(sub, target, {"ref": ref, "mod": mod}, c, seed + r)
            rows.append({"fraction": f, "rep": r, "n_athletes": int(res.y.size), "delta_rmse": float(np.sqrt(np.nanmean((res.y - res.preds["ref"]) ** 2))
                                                                                                     - np.sqrt(np.nanmean((res.y - res.preds["mod"]) ** 2)))})
    return pl.DataFrame(rows)


# ============================================================================== research decision (G6)
def recommendations(status: pl.DataFrame, comps: pl.DataFrame, tournament: pl.DataFrame | None) -> pl.DataFrame:
    """Declared rules mapping persisted verdicts to candidate next directions. A rule fires only on its evidence; several
    may fire; none is preselected. Each row names the comparisons that triggered it."""
    out = []

    def fired(direction: str, why: str, refs: Sequence[str], strength: str) -> None:
        out.append({"direction": direction, "triggered_by": why, "evidence": "; ".join(refs), "strength": strength})

    get = lambda exp: comps.filter(pl.col("experiment") == exp) if comps.height else comps
    for k, (tr, title) in EXPERIMENTS.items():
        s = status.filter(pl.col("track") == tr)["status"]
        if not s.len() or s[0] in (PENDING, INCOMPLETE):
            fired(f"complete {tr} before interpreting experiment {k}", f"{tr} status {s[0] if s.len() else PENDING}", [title], "blocking")
    e3 = get(3)
    rel_win = e3.filter(pl.col("metric").str.contains("reliability") & (pl.col("verdict") == "supported"))
    if rel_win.height:
        fired("prioritise distributional NFL capacity targets (replace maxima / linear envelopes)", "a distributional estimator is more game-split reliable than the envelope",
              rel_win["comparison"].to_list(), "strong" if rel_win.height >= 2 else "moderate")
    hs = comps.filter(pl.col("comparison").str.contains("replication")) if comps.height else comps
    if hs.height:
        share = float((hs["verdict"] == "supported").mean())
        fired("prioritise high-speed transfer confirmation on an independent draft class" if share >= 0.5 else
              "do not build on the 11B high-speed findings until replicated", f"{share:.0%} of declared replication variants have Δ CI > 0", hs["comparison"].to_list()[:6],
              "moderate" if share >= 0.5 else "cautionary")
    if tournament is not None and tournament.height:
        win = tournament.filter((pl.col("reference") == "M1") & (pl.col("delta_lo") > 0))
        if win.height:
            fired("confirm the winning representation(s) on an independent cohort before any product use", "tournament Δ RMSE CI > 0 vs M1 on identical athletes",
                  win["model"].to_list(), "moderate")
        else:
            fired("treat advanced representations as descriptive; NFL physical prediction beyond official tests is not established", "no tournament Δ CI > 0 vs M1",
                  tournament["model"].to_list(), "strong")
    e2 = get(2)
    if e2.height and not e2.filter(pl.col("verdict") == "supported").height:
        fired("stop pursuing measurement-error correction as a source of predictive gain", "corrected estimands / held-out RC show no supported gain",
              e2["comparison"].to_list()[:4], "moderate")
    e5 = get(5)
    if e5.filter(pl.col("verdict") == "supported").height:
        fired("use distribution-level matching for descriptive cross-domain comparison (not prediction)", "conditional OT retrieval beats feature distance",
              e5["comparison"].to_list(), "moderate")
    e6 = get(6)
    if e6.height and not e6.filter((pl.col("training_regime") == "inductive") & (pl.col("verdict") == "supported")).height:
        fired("pause shared neural representation learning; physics-based features are not outperformed", "no inductive encoder comparison supported",
              e6["comparison"].to_list()[:4], "moderate")
    e4 = get(4)
    if e4.filter(pl.col("verdict") == "supported").height:
        fired("focus on multi-view traits as more reliable measurements", "latent factor test–retest exceeds the best single feature", e4["comparison"].to_list(), "moderate")
    if not out:
        fired("no rule fired: report all tracks as inconclusive at the available sample size", "—", [], "—")
    return pl.DataFrame(out)


__all__ = ["CATEGORIES", "COMPARISON_COLUMNS", "EXPERIMENTS", "TRACK_TABLES", "cohort_size_sensitivity", "collect", "comparison_frame", "comparison_row",
           "evidence_matrix", "experiment_table", "recommendations"]
