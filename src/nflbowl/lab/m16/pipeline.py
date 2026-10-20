"""Notebook-level orchestration of M16 E16.A2–A6 and E16.B1–B3 + G16 (T06, T09, T10). Each ``run_*`` = one guarded step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..status import BLOCKED_MISSING_INPUT, ResearchBlocked
from . import baselines as BL
from . import detectability as DT
from . import gates as GT
from . import leakage as LK
from . import registry as RG
from . import retrieval as RV
from . import target_reliability as TRL
from . import targets as TG


def m15_config(cfg: Any) -> Any:
    """The M15 configuration matching this M16 run (same bundle; used for the shared data adapter and feature code)."""
    from ..m15.config import resolve

    return resolve({"bundle_dir": cfg.bundle_dir, "drive_root": cfg.drive_root, "stage_dir": cfg.stage_dir, "expected_fingerprint": cfg.expected_fingerprint,
                    "expected_science_hash": cfg.expected_science_hash, "m12_run_dir": cfg.m12_run_dir, "m13_run_dir": cfg.m13_run_dir,
                    "m14_run_dir": cfg.m14_run_dir, "experiment_name": "m15_view_for_m16", "seed": cfg.seed})


def data_from_experiment(exp: Any, cfg15: Any) -> Any:
    """``M15Data`` built on the already opened experiment bundle (no second open, nothing written)."""
    from ...data.splits import fold_map
    from ..m15.data import M15Data, audit_bundle, fold_assignments, sequence_index

    b = exp.bundle
    fo = fold_map(b.meta, b.folds, cfg15.outer_folds)
    fa = fold_assignments(b, cfg15)
    return M15Data(bundle=b, audit=audit_bundle(b, cfg15, fa), folds=fa, index=sequence_index(b, fo, cfg15), fold_of=fo, fingerprint=b.fingerprint)


def feature_table(cfg: Any, data: Any, cfg15: Any, run: Any) -> Any:
    """E15.A1 sequence features: read from the M15 run when present (read only), else computed and cached in M16."""
    import polars as pl

    from ..m15.signals import feature_table as ft

    p = Path(cfg.m15_run_dir) / "features" / "signal" / "sequence_time_features.parquet"
    if p.is_file():
        return pl.read_parquet(p)
    prov = {"bundle_fingerprint": data.fingerprint, "features": "signals_v1"}
    f = run.load_data("datasets/cached/sequence_time_features", prov)
    if f is None:
        f = ft(data, cfg15)
        run.data_file(f, "datasets/cached/sequence_time_features", prov)
    return f


def tconfig(cfg: Any) -> TG.TargetConfig:
    return TG.TargetConfig(min_windows=cfg.target_min_windows, min_windows_kind=cfg.target_min_windows_kind, min_events=cfg.target_min_events)


# ====================================================================== A2
def run_targets(cfg: Any, data: Any, feat: Any, cfg15: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    tc = tconfig(cfg)
    prov = {"bundle_fingerprint": data.fingerprint, "targets": "t16_v1", "tconfig": str(tc)}
    win = run.load_data("features/targets/window_table", prov) if cfg.reuse_artifacts else None
    if win is None:
        win = TG.nfl_window_table(data, feat, cfg15, tc)
        run.data_file(win, "features/targets/window_table", prov)
    else:
        ctx["cache"] = "hit"
    ev = TG.nfl_events(data, run, reuse=cfg.reuse_artifacts)
    ath = TG.athlete_targets(win, ev, data.bundle.athletes, data.fold_of, tc)
    run.data_file(ath, "features/targets/athlete_targets", prov)
    defs = pl.DataFrame([s.to_dict() for s in TG.TARGETS.values()])
    for name, df in (("definitions", defs), ("coverage", TG.coverage_table(ath)), ("redundancy", TG.redundancy(ath))):
        ctx["outputs"] += run.table(df, name, "target")
    return {"windows": win, "events": ev, "targets": ath}


# ====================================================================== A3
def run_reliability(cfg: Any, data: Any, T: dict[str, Any], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    from ..m15.data import game_key

    tc = tconfig(cfg)
    gk, _ = game_key(data.index.filter(data.index["domain"] == "nfl"))
    gkf = None
    if gk is not None:
        import polars as pl

        gkf = data.index.filter(pl.col("domain") == "nfl").select("idx").with_columns(gk.alias("game_key"))
    rel = TRL.reliability_table(T["windows"], T["events"], data.bundle.athletes, cfg.reliability_split, cfg.seed, max(200, cfg.n_boot // 4), tc, gkf)
    expo = TRL.exposure_dependence(T["targets"])
    curve = TRL.exposure_curve(T["windows"], T["events"], data.bundle.athletes, cfg.seed)
    cs = TRL.context_share(T["windows"], T["events"])
    gate = TRL.gate_a(rel, expo, cs)
    frz = TRL.freeze(run, gate, {k: v.to_dict() for k, v in TG.TARGETS.items()})
    for name, df in (("reliability", rel), ("exposure_dependence", expo), ("exposure_curve", curve), ("context_share", cs), ("gate_a", gate)):
        ctx["outputs"] += run.table(df, name, "target")
    return {"reliability": rel, "exposure": expo, "curve": curve, "context": cs, "gate_a": gate, "freeze": frz}


# ====================================================================== A6
def run_detectability(cfg: Any, data_frame: Any, target_ids: list[str], run: Any, ctx: dict[str, Any]) -> Any:
    import polars as pl

    from ...data.validation import M1_COLS

    parts = []
    for tid in target_ids:
        col = TG.TARGETS[tid].column
        if col not in data_frame.columns:
            continue
        prov = {"target": tid, "fractions": list(cfg.detect_fractions), "reps": cfg.detect_reps, "rel": cfg.detect_reliability, "seed": cfg.seed,
                "n": int(data_frame[col].is_not_null().sum())}
        p = run.load_data(f"datasets/cached/mde_{tid}", prov) if cfg.reuse_artifacts else None
        if p is None:
            p = DT.simulate(data_frame, col, [c for c in M1_COLS if c in data_frame.columns], cfg.detect_fractions, cfg.detect_reliability, cfg.detect_reps,
                            200, 1.0, cfg.seed).with_columns(pl.lit(tid).alias("target_id"))
            run.data_file(p, f"datasets/cached/mde_{tid}", prov)
        parts.append(p)
    if not parts:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no target column available for the simulation", missing=target_ids)
    power = pl.concat(parts, how="diagonal_relaxed")
    m = DT.mde(power)
    ctx["outputs"] += run.table(power, "power", "detectability") + run.table(m, "mde", "detectability")
    return {"power": power, "mde": m}


# ====================================================================== A4 / A5
def ladder_targets(cfg: Any, freeze: dict[str, Any] | None) -> list[str]:
    if cfg.task_targets_in_ladder == "all":
        return ["T0", *TG.TASK_IDS]
    return ["T0", *((freeze or {}).get("eligible_task_targets") or [])]


def run_baselines(cfg: Any, exp: Any, feat: Any, targets: Any, freeze: dict[str, Any] | None, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    simple = BL.simple_tracking_table(feat)
    try:
        m14, m14_cols, note = BL.m14_reliable_table(cfg.m14_run_dir)
    except ResearchBlocked as e:
        m14, m14_cols, note = None, [], e.reason
    data = BL.analysis_frame(exp, targets, simple, m14)
    run.data_file(data, "datasets/cached/analysis_frame", {"bundle_fingerprint": exp.bundle.fingerprint})
    results, notes = [], [{"item": "M14 reliable columns", "note": note}]
    for tid in ladder_targets(cfg, freeze):
        try:
            res = BL.run_target(exp, data, tid, m14_cols, cfg.learners, cfg.min_coverage, cfg.seed)
        except ResearchBlocked as e:
            notes.append({"item": f"target {tid}", "note": e.reason})
            continue
        ev = BL.evaluate_target(res, exp.folds_by_athlete, cfg.n_boot, cfg.seed, BL.PRIMARY_T0 if tid == "T0" else ())
        ev["target_id"] = tid
        notes += [{"item": f"{tid}: {k}", "note": v} for k, v in res["notes"].items()]
        results.append(ev)
    if not results:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no target could be evaluated", missing=["targets"])
    reg, oof = RG.evaluation_registry("m16", results)
    run.data_file(oof, "predictions/oof_long", {"bundle_fingerprint": exp.bundle.fingerprint})
    def cat_frames(k: str) -> Any:
        fs = [r[k] for r in results if r.get(k) is not None and r[k].height]
        return pl.concat(fs, how="diagonal_relaxed") if fs else pl.DataFrame()

    cat = {k: cat_frames(k) for k in ("metrics", "paired", "intervals", "folds", "status", "within", "ranking", "position_effects", "audit")}
    for k, df in cat.items():
        ctx["outputs"] += run.table(df, k, "baselines")
    ctx["outputs"] += run.table(reg, "evaluation_registry", "registry") + run.table(pl.DataFrame(notes), "baseline_notes", "baselines")
    return {"data": data, "results": results, **cat, "registry": reg, "oof": oof}


# ====================================================================== B1–B3
def run_retrieval(cfg: Any, exp: Any, data: Any, feat: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    reps, note = RV.eligible_representations(cfg.m15_run_dir, cfg.allow_descriptive)
    folds = sorted(set(exp.folds_by_athlete.values()))
    scores = RV.run_retrieval(cfg.m15_run_dir, reps, data.index, exp.bundle.athletes, exp.features(), folds, cfg.seed)
    from ..m15.transfer import summarise

    summ = summarise(scores) if scores.height else scores
    tests = RV.primary_tests(scores, cfg.n_boot, cfg.seed) if scores.height else scores
    cem = RV.cem_sensitivity(cfg.m15_run_dir, reps, feat, exp.bundle.athletes, folds, cfg.seed) if reps else None
    run.data_file(scores, "datasets/cached/m16_retrieval_scores", {"reps": [f"{a}_d{d}" for a, d in reps]})
    for name, df in (("m16_summary", summ), ("m16_tests", tests), ("cem", cem)):
        if df is not None:
            ctx["outputs"] += run.table(df, name, "retrieval")
    return {"reps": reps, "note": note, "scores": scores, "summary": summ, "tests": tests, "cem": cem}


def run_leakage(cfg: Any, exp: Any, reps: list[tuple[str, int]], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    tab, verdict = LK.audit(cfg.m15_run_dir, reps, exp.folds_by_athlete, exp.bundle.fingerprint)
    run.data_file(tab, "diagnostics/leakage_audit", {"reps": [f"{a}_d{d}" for a, d in reps]})
    ctx["outputs"] += run.table(verdict, "leakage_verdict", "retrieval")
    return {"audit": tab, "verdict": verdict}


def run_gate(run: Any, ctx: dict[str, Any]) -> Any:
    import polars as pl

    def fr(name: str, fam: str) -> Any:
        return run.frame(name, fam) if run.has(name, fam) else None

    status = fr("status", "baselines")
    paired = fr("paired", "baselines")
    st0 = status.filter(pl.col("target_id") == "T0") if status is not None and "target_id" in status.columns else status
    pa0 = paired.filter(pl.col("target_id") == "T0") if paired is not None and "target_id" in paired.columns else paired
    gate = GT.g16(fr("gate_a", "target"), fr("g16b_gate", "synthesis"), st0, pa0, fr("m16_tests", "retrieval"), fr("cem", "retrieval"))
    ctx["outputs"] += run.table(gate, "m16_gate", "synthesis")
    return gate


__all__ = ["data_from_experiment", "feature_table", "ladder_targets", "m15_config", "run_baselines", "run_detectability", "run_gate", "run_leakage",
           "run_reliability", "run_retrieval", "run_targets", "tconfig"]
