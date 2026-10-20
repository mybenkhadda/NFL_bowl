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


# ====================================================================== M16 laboratory steps (E16.A7–A9, B4–B6, C0–C7, G16 extended)
# Cache contract: a step's tables are reloaded only when the input signature stored with them (``manifests/
# stage_signatures.json`` → {inputs, tables}) equals the current one — data / config fingerprints and the size + mtime of
# every upstream file. A matching file name is never enough. ``cfg.force_steps`` forces a recomputation.
def file_identity(paths: Any) -> list[list[Any]]:
    out = []
    for p in sorted(str(x) for x in paths):
        q = Path(p)
        st = q.stat() if q.is_file() else None
        out.append([p, st.st_size if st else None, st.st_mtime_ns if st else None])
    return out


def load_step(run: Any, cfg: Any, step: str, family: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
    import json

    if not cfg.reuse(step):
        return None
    rec = run._sigs().get(f"{family}/{step}", {}).get("signature")
    if not isinstance(rec, dict) or json.dumps(rec.get("inputs"), sort_keys=True, default=str) != json.dumps(inputs, sort_keys=True, default=str):
        return None
    names = rec.get("tables") or []
    if not names or not all(run.has(n, family) for n in names):
        return None
    return {n: run.frame(n, family) for n in names}


def save_step(run: Any, ctx: dict[str, Any], step: str, family: str, frames: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    written = []
    for name, df in frames.items():
        if df is None or not getattr(df, "width", 0):
            continue
        out = run.table(df, name, family)
        if out:
            ctx["outputs"] += out
            written.append(name)
    run.set_signature(f"{family}/{step}", {"inputs": inputs, "tables": written})
    return written


def _frame(run: Any, name: str, fam: str) -> Any:
    return run.frame(name, fam) if run.has(name, fam) else None


# ---------------------------------------------------------------------- A9 cohort
def run_cohort(cfg: Any, exp: Any, data: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import json

    import polars as pl

    from ...artifacts.store import atomic_json
    from ...data.validation import M1_COLS, TRANSFER_TARGET
    from ..status import ValidationFailed
    from . import cohort as CH

    per, summ = CH.dual_domain(data.index, exp.bundle.athletes)
    comp = CH.composition(data.index)
    max_frames = int(getattr(exp.bundle.X, "shape", (0, 64))[1])
    frames = {"dual_domain": per.with_columns(pl.col("combine_kinds").list.join(", ")),
              "dual_domain_summary": pl.DataFrame([{"key": k, "value": json.dumps(v)} for k, v in summ.items()]),
              "documented_checks": CH.documented_checks(data.index, summ, max_frames), "funnel": CH.funnel(per, exp.bundle.athletes, TRANSFER_TARGET, M1_COLS),
              "artifact_inventory": CH.artifact_inventory(cfg), **{k: v for k, v in comp.items() if k != "exposure_per_athlete"}}
    for name, df in frames.items():
        ctx["outputs"] += run.table(df, name, "cohort")
    run.data_file(comp["exposure_per_athlete"], "datasets/cached/exposure_per_athlete", {"bundle_fingerprint": data.fingerprint})
    atomic_json(summ, run.root / "manifests" / "dual_domain.json", overwrite=True)
    try:
        fi = CH.fold_integrity(data.index, exp.folds_by_athlete, exp.cfg.science.outer_folds)
    except ValidationFailed as e:
        run.table(pl.DataFrame(e.detail.get("table", [])), "fold_integrity", "cohort")
        raise
    ctx["outputs"] += run.table(fi, "fold_integrity", "cohort")
    return {**frames, "fold_integrity": fi, "summary": summ, "exposure": comp["exposure_per_athlete"]}


# ---------------------------------------------------------------------- A7 target laboratory
def run_target_lab(cfg: Any, data: Any, T: dict[str, Any], rel: dict[str, Any] | None, athletes: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    from ...artifacts.store import atomic_json
    from ...data.validation import M1_COLS
    from . import target_audit as TA
    from . import target_registry as TRG

    TRG.check_consistency()
    atomic_json(TRG.definitions_record(), run.root / "manifests" / "target_definitions_m16.json", overwrite=True)
    inputs = {"bundle": data.fingerprint, "config": cfg.fingerprint(), "targets": TG.frame_fingerprint(T["targets"]), "registry": TRG.registry_hash(),
              "gate_a": TG.frame_fingerprint(rel["gate_a"], "target_id") if rel else None}
    hit = load_step(run, cfg, "E16.A7", "exploration", inputs)
    long = run.load_data("features/targets/target_long", inputs)
    if hit is not None and long is not None:
        ctx["cache"] = "hit"
        return {**hit, "long": long}
    ath = T["targets"]
    long = TA.long_values(ath)
    positions = ath.select("nfl_id", "position_group")
    unc = TRG.athlete_uncertainty(T["windows"], T["events"], positions, cfg.uncertainty_boot, cfg.seed)
    meas = TRG.measurement_reliability(unc)
    nb = max(200, cfg.n_boot // 5)
    rels = TA.relations(ath, TA.athlete_context(data.index), athletes, nb, cfg.seed)
    tc = tconfig(cfg)
    frames = {"definitions": TRG.definitions_frame(), "distributions": TA.distributions(long), "variance": TA.variance_decomposition(long, nb, cfg.seed),
              "position_means": TA.position_means(long, nb, cfg.seed), "relations": rels,
              "min_obs_sensitivity": TA.min_obs_sensitivity(T["windows"], T["events"], ath, tc, cfg.min_windows_grid, cfg.min_events_grid),
              "aggregation_sensitivity": TA.aggregation_sensitivity(T["windows"], T["events"], ath, positions),
              "subsample_stability": TA.subsample_stability(T["windows"], T["events"], cfg.subsample_fractions, cfg.subsample_reps, cfg.seed),
              "correlations": TA.correlations(ath), "selection_bias": TA.selection_bias(ath, athletes, M1_COLS), "censoring": TA.censoring(ath),
              "measurement_reliability": meas}
    gate_a = rel["gate_a"] if rel else _frame(run, "gate_a", "target")
    rel_tab = rel["reliability"] if rel else _frame(run, "reliability", "target")
    frames["scorecard"] = TA.scorecard(gate_a, _frame(run, "coverage", "target"), rels, meas, rel_tab, frames["min_obs_sensitivity"],
                                       frames["aggregation_sensitivity"], frames["selection_bias"], frames["censoring"], cfg.max_censored_share)
    run.data_file(long, "features/targets/target_long", inputs)
    run.data_file(unc, "features/targets/athlete_uncertainty", inputs)
    save_step(run, ctx, "E16.A7", "exploration", frames, inputs)
    return {**frames, "long": long, "uncertainty": unc}


# ---------------------------------------------------------------------- A8 context common support
def run_context(cfg: Any, data: Any, feat: Any, T: dict[str, Any], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    from ..status import BLOCKED_INSUFFICIENT_OVERLAP, COMPLETED
    from . import context_matching as CM

    inputs = {"bundle": data.fingerprint, "config": cfg.fingerprint(), "targets": TG.frame_fingerprint(T["targets"]), "feat_rows": feat.height}
    hit = load_step(run, cfg, "E16.A8", "context", inputs)
    wide = run.load_data("features/context/matched_outcomes", inputs)
    if hit is None or wide is None:
        uni, multi, keep = CM.support_table(feat, cfg.context_covariates, cfg.context_bins, cfg.min_support_share, cfg.min_support_athletes)
        wide, summ = CM.matched_outcomes(T["windows"], T["events"], keep, multi, cfg.target_min_events)
        hit = {"overlap": uni, "support": multi, "matched_summary": summ, "matched_vs_global": CM.matched_vs_global(wide, T["targets"], None)}
        run.data_file(wide, "features/context/matched_outcomes", inputs)
        save_step(run, ctx, "E16.A8", "context", hit, inputs)
    else:
        ctx["cache"] = "hit"
    out = {**hit, "matched": wide}
    sup = hit.get("support")
    if sup is None or not sup.filter(pl.col("status") == COMPLETED).height:
        raise ResearchBlocked(BLOCKED_INSUFFICIENT_OVERLAP, "no Combine drill has sufficient common support with NFL windows (tables written; no extrapolated match)",
                              detail={"support": sup.to_dicts() if sup is not None else []})
    return out


# ---------------------------------------------------------------------- B4 provenance registry
def run_provenance(cfg: Any, exp: Any, data: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    from ...artifacts.store import atomic_json
    from . import representation_audit as RA

    K = exp.cfg.science.outer_folds
    folds, reg, pres = RA.registry(cfg, data.fold_of, data.fingerprint, K)
    legacy = RA.legacy_rows(cfg, exp.bundle, data.fold_of, K) if cfg.legacy_m15_dirs else None
    atomic_json(pres, run.root / "manifests" / "m15_preservation_evidence.json", overwrite=True)
    for name, df in (("registry_folds", folds), ("registry", reg), ("registry_counts", RA.summary_counts(reg)), ("legacy", legacy)):
        if df is not None:
            ctx["outputs"] += run.table(df, name, "representation")
    if not reg.height:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no M15 inductive embedding found (run M15 Track B)", missing=[str(Path(cfg.m15_run_dir) / "embeddings" / "inductive")])
    return {"folds": folds, "registry": reg, "preservation": pres, "legacy": legacy}


def chosen_reps(cfg: Any, reg: Any) -> list[dict[str, Any]]:
    """The pre-declared representation per architecture (``ladder_dim``, smallest verified seed) — shared by B5, B6, C."""
    from . import representation_audit as RA

    out = []
    for arch in cfg.representation_archs:
        r, _ = RA.choose(reg, arch, cfg.ladder_dim)
        if r is not None:
            out.append(r)
    return out


def _emb_path(cfg: Any, rep: dict[str, Any], f: int) -> Path:
    return Path(cfg.m15_run_dir) / "embeddings" / "inductive" / f"{rep['arch']}_d{int(rep['d'])}_s{int(rep['seed'])}__fold{f}.parquet"


# ---------------------------------------------------------------------- B5 controlled retrieval
def run_controlled_retrieval(cfg: Any, exp: Any, data: Any, feat: Any, reg: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import gc

    import polars as pl

    from ..m15 import transfer as TR
    from ..status import BLOCKED_PROVENANCE
    from . import controlled_retrieval as CR
    from . import features as FT

    K = exp.cfg.science.outer_folds
    reps = chosen_reps(cfg, reg)
    inputs = {"bundle": data.fingerprint, "config": cfg.fingerprint(), "reps": [r["rep"] for r in reps],
              "files": file_identity(_emb_path(cfg, r, f) for r in reps for f in range(K))}
    hit = load_step(run, cfg, "E16.B5", "retrieval", inputs)
    scores = run.load_data("retrieval/controlled_scores", inputs)
    if hit is not None and scores is not None:
        ctx["cache"] = "hit"
        return {**hit, "scores": scores, "sims": run.load_data("retrieval/controlled_similarities", inputs), "reps": reps}
    attrs = CR.attributes(exp.bundle.athletes)
    try:
        e0 = exp.features()
    except Exception as e:                                           # noqa: BLE001 — F0 is a control; its absence is recorded
        e0, notes = None, [{"item": "F0", "note": f"engineered features unavailable: {type(e).__name__}: {e}"}]
    else:
        notes = []
    rows, sims, meta = [], [], []
    by_arch = {r["arch"]: r for r in reps}
    for f in range(K):
        tabs: dict[str, Any] = {}
        for r in reps:
            p = _emb_path(cfg, r, f)
            if p.is_file():
                tabs[r["rep"]] = pl.read_parquet(p)
            else:
                notes.append({"item": f"{r['rep']} fold {f}", "note": "embedding file missing"})
        if "stft_ae" in by_arch and "lstm_ae" in by_arch and by_arch["stft_ae"]["rep"] in tabs and by_arch["lstm_ae"]["rep"] in tabs:
            tabs["stft+lstm"] = TR.concat_embeddings(tabs[by_arch["stft_ae"]["rep"]], tabs[by_arch["lstm_ae"]["rep"]])
        if e0 is not None:
            tabs["F0"] = TR.sequence_matrix_embedding(e0, data.index, f)
        tabs["kinematic"] = CR.kinematic_embedding(feat, data.index, f, FT.KINEMATICS)
        for name, emb in tabs.items():
            arms = ("unaligned", "procrustes") if name in ("F0", "kinematic") else cfg.retrieval_arms
            for scope in cfg.retrieval_scopes:
                try:
                    r_, s_, m_ = CR.score_fold(emb, {"rep": name, "fold": f, "scope": scope}, FT.SCOPES[scope], attrs, cfg, arms, cfg.seed + f, scope == "all")
                    if name == "kinematic":
                        r2, _, _ = CR.score_fold(emb, {"rep": "random", "fold": f, "scope": scope}, FT.SCOPES[scope], attrs, cfg, ("random",), cfg.seed + f, False)
                        r_ += r2
                        meta += m_
                except ResearchBlocked as e:
                    notes.append({"item": f"{name} fold {f} scope {scope}", "note": e.reason})
                    continue
                rows += r_
                sims += s_
        del tabs
        gc.collect()
    scores = pl.DataFrame(rows, infer_schema_length=None)
    sim = pl.DataFrame(sims, infer_schema_length=None) if sims else pl.DataFrame()
    pmeta = pl.DataFrame(meta, infer_schema_length=None) if meta else pl.DataFrame()
    if not scores.height:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no retrieval could be scored (no dual-domain held-out athletes?)", detail={"notes": notes})
    po = CR.position_only_rows(scores.filter(pl.col("rep") == "kinematic") if scores.filter(pl.col("rep") == "kinematic").height else scores)
    po = po.unique(subset=["fold", "scope", "pool", "nfl_id"])
    summ = CR.summarise(pl.concat([scores, po.drop("rep_control").with_columns(pl.lit("position_only").alias("rep"))], how="diagonal_relaxed"),
                        max(500, cfg.n_boot // 2), cfg.seed)
    vs = CR.paired_vs_controls(scores, po, max(500, cfg.n_boot // 2), cfg.seed)
    frames = {"controlled_summary": summ, "controlled_vs_controls": vs, "controlled_foldwise": CR.foldwise(scores), "controlled_within_position": CR.within_position(scores),
              "controlled_errors": CR.error_summary(scores), "controlled_notes": pl.DataFrame(notes, infer_schema_length=None) if notes else None,
              "controlled_gallery": pl.concat([CR.gallery(scores, r) for r in scores["rep"].unique().to_list() if r not in ("random",)], how="diagonal_relaxed")}
    run.data_file(scores, "retrieval/controlled_scores", inputs)
    if sim.height:
        run.data_file(sim, "retrieval/controlled_similarities", inputs)
    if pmeta.height:
        run.data_file(pmeta, "retrieval/pool_meta", inputs)
    save_step(run, ctx, "E16.B5", "retrieval", frames, inputs)
    out = {**frames, "scores": scores, "sims": sim, "pool_meta": pmeta, "reps": reps}
    if not reps:
        raise ResearchBlocked(BLOCKED_PROVENANCE, "no VERIFIED_FOLD_SAFE representation at the pre-declared d: only the controls were scored (tables written)")
    return out


# ---------------------------------------------------------------------- B6 repeatability / alignment damage
def run_repeatability(cfg: Any, exp: Any, data: Any, reg: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import gc

    import polars as pl

    from ...data.validation import M1_COLS
    from ..status import BLOCKED_PROVENANCE
    from . import features as FT
    from . import repeatability as RP

    K = exp.cfg.science.outer_folds
    reps = chosen_reps(cfg, reg)
    if not reps:
        raise ResearchBlocked(BLOCKED_PROVENANCE, "no VERIFIED_FOLD_SAFE representation at the pre-declared d")
    inputs = {"bundle": data.fingerprint, "config": cfg.fingerprint(), "files": file_identity(_emb_path(cfg, r, f) for r in reps for f in range(K))}
    hit = load_step(run, cfg, "E16.B6", "retrieval", inputs)
    if hit is not None:
        ctx["cache"] = "hit"
        return hit
    rows, geo, m1r = [], [], []
    for r in reps:
        for f in range(K):
            p = _emb_path(cfg, r, f)
            if not p.is_file():
                continue
            emb = pl.read_parquet(p)
            lab = {"rep": r["rep"], "fold": f}
            rows += RP.fold_rows(emb, lab, cfg.retrieval_arms, cfg.seed + f)
            geo.append(RP.representation_rows(emb, lab, cfg.seed, data.index))
            del emb
            gc.collect()
        tabs, _ = FT.by_fold(cfg.m15_run_dir, r, "all", cfg.primary_aggregation, exp.folds_by_athlete, K, cfg.trim_share)
        rel = FT.m1_relation(tabs, exp.bundle.athletes, M1_COLS, exp.folds_by_athlete)
        if rel.height:
            m1r.append(rel.with_columns(pl.lit(r["rep"]).alias("rep")))
    rr = pl.DataFrame(rows, infer_schema_length=None)
    frames = {"repeat_rows": rr, "repeat_damage": RP.damage_summary(rr, cfg.damage_threshold), "representation_geometry": pl.DataFrame(geo, infer_schema_length=None),
              "representation_m1_relation": pl.concat(m1r, how="diagonal_relaxed") if m1r else None}
    save_step(run, ctx, "E16.B6", "retrieval", frames, inputs)
    return frames


# ---------------------------------------------------------------------- C0 Combine-only features + analysis frame
def analysis16(cfg: Any, exp: Any, feat: Any, targets: Any, matched: Any | None) -> tuple[Any, list[str], str]:
    """The official analysis frame (``baselines.analysis_frame``) + drill kinematics (``kin_*``) + context-matched NFL
    outcomes (``t16_ctx_*``, outcomes only — never predictors)."""
    import polars as pl

    from . import features as FT

    simple = BL.simple_tracking_table(feat)
    try:
        m14, m14_cols, note = BL.m14_reliable_table(cfg.m14_run_dir)
    except ResearchBlocked as e:
        m14, m14_cols, note = None, [], e.reason
    a = BL.analysis_frame(exp, targets, simple, m14).join(FT.drill_kinematics(feat), on="nfl_id", how="left")
    if matched is not None and matched.height:
        a = a.join(matched.select(pl.col("nfl_id").cast(pl.Int64), *[c for c in matched.columns if c.startswith("t16_ctx_")]), on="nfl_id", how="left")
    return a, list(m14_cols), note


def run_features(cfg: Any, exp: Any, feat: Any, targets: Any, matched: Any | None, reg: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    from ...data.validation import M1_COLS
    from . import features as FT
    from . import ladder as LD

    K = exp.cfg.science.outer_folds
    fold_of = exp.folds_by_athlete
    analysis, m14_cols, note = analysis16(cfg, exp, feat, targets, matched)
    plan = LD.plan_rungs(cfg, reg, analysis, m14_cols)
    tables, inv = LD.embedding_tables(cfg, plan, fold_of, K)
    linked = analysis["nfl_id"].to_list()
    cov, m1r = [], []
    for r, tb in tables.items():
        for f, t in tb.items():
            run.data_file(t, f"features/athlete/{r}__{cfg.primary_aggregation}__all__fold{f}", {"bundle_fingerprint": exp.bundle.fingerprint, "rung": r,
                                                                                               "rep": plan[r].rep["rep"] if plan[r].rep else "fusion", "fold": f})
        cov.append({"rung": r, "model": LD.RUNGS[r], **FT.coverage(tb, linked, fold_of)})
        rel = FT.m1_relation(tb, exp.bundle.athletes, M1_COLS, fold_of)
        if rel.height:
            m1r.append(rel.with_columns(pl.lit(r).alias("rung")))
    run.data_file(analysis, "datasets/cached/analysis_frame_lab", {"bundle_fingerprint": exp.bundle.fingerprint, "config": cfg.fingerprint()})
    frames = {"rung_plan": LD.plan_frame(plan)}
    ctx["outputs"] += run.table(frames["rung_plan"], "rung_plan", "ladder")
    for name, df in (("inventory", inv), ("coverage", pl.DataFrame(cov, infer_schema_length=None) if cov else None),
                     ("m1_relation", pl.concat(m1r, how="diagonal_relaxed") if m1r else None),
                     ("notes", pl.DataFrame([{"item": "M14 reliable columns", "note": note}]))):
        if df is not None and df.width:
            ctx["outputs"] += run.table(df, name, "features")
    return {"analysis": analysis, "plan": plan, "tables": tables, "m14_cols": m14_cols, "note": note}


# ---------------------------------------------------------------------- C4 ladder
def wides_from_long(long: Any) -> dict[str, Any]:
    """{target: strict-cohort wide frame} rebuilt from the saved OOF long table (every model finite)."""
    import polars as pl

    from .. import transfer_fairness as TF

    out = {}
    for (tid,), g in long.group_by("target_id"):
        models = sorted(set(g["model"].to_list()))
        fin = g.filter(pl.col("pred").is_finite()).group_by("nfl_id").agg(pl.col("model").n_unique().alias("k"))
        strict = set(int(i) for i in fin.filter(pl.col("k") == len(models))["nfl_id"].to_list())
        try:
            out[str(tid)] = TF.wide(g, strict, models)
        except TF.ComparisonError:
            continue
    return out


def run_ladder16(cfg: Any, exp: Any, F: dict[str, Any], freeze: dict[str, Any] | None, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import json

    import polars as pl

    from ...artifacts.store import atomic_json
    from .. import transfer_fairness as TF
    from . import evaluation as EV
    from . import ladder as LD

    fold_of = exp.folds_by_athlete
    plan, analysis = F["plan"], F["analysis"]
    tids = ladder_targets(cfg, freeze)
    inputs = {"bundle": exp.bundle.fingerprint, "config": cfg.fingerprint(), "targets": tids, "plan": LD.plan_frame(plan).to_dicts(),
              "analysis": TG.frame_fingerprint(analysis), "files": file_identity(_emb_path(cfg, p.rep, f) for p in plan.values() if p.rep
                                                                                for f in range(exp.cfg.science.outer_folds))}
    hit = load_step(run, cfg, "E16.C4", "ladder", inputs)
    long = run.load_data("predictions/m16_ladder_oof_long", inputs)
    if hit is not None and long is not None:
        ctx["cache"] = "hit"
        return {**hit, "long": long, "wides": wides_from_long(long)}
    from .target_audit import EXPOSURE

    tgt_frame = _targets_frame(run)
    log: list = []
    mods = LD.build_models(cfg, plan, analysis, fold_of, F["tables"], log)
    learners = LD.learner_fns(cfg)
    lmodels = [m for m in mods if m != "POS"]
    prim = LD.primary_family(plan)
    evs, notes, extra = [], [], {k: [] for k in ("calibration", "calibration_bins", "stratified_errors", "paired_spearman", "improvement")}
    for tid in tids:
        col = TG.TARGETS[tid].column
        try:
            res = LD.fit(exp, analysis, tid, col, mods, learners, lmodels, cfg.seed)
        except ResearchBlocked as e:
            notes.append({"item": f"target {tid}", "note": e.reason})
            continue
        ev = LD.evaluate(res, fold_of, cfg, prim)
        notes += [{"item": f"{tid}: {k}", "note": v} for k, v in res["notes"].items()]
        ev["ladder_log"] = res["ladder_log"].with_columns(pl.lit(tid).alias("target_id"))
        ev["learner_log"] = pl.DataFrame(res["learner_log"], infer_schema_length=None).with_columns(pl.lit(tid).alias("target_id")) if res["learner_log"] else None
        w = ev["wide_strict"]
        if w is not None:
            base = [m for m in LD.RUNGS.values() if f"pred_{m}" in w.columns]
            tag = pl.lit(tid).alias("target_id")
            cal, bins = EV.calibration(w, base, max(200, cfg.n_boot // 5), cfg.seed)
            extra["calibration"].append(cal.with_columns(tag))
            extra["calibration_bins"].append(bins.with_columns(tag))
            strata = [EV.stratified_errors(w, base, EV.strata_exposure(tgt_frame, EXPOSURE.get(tid, "n_windows")), "NFL exposure"),
                      EV.stratified_errors(w, base, EV.strata_coverage(analysis, [c for c in analysis.columns if c.startswith("trk_")], "Combine tracking"),
                                           "Combine tracking coverage")]
            extra["stratified_errors"] += [s.with_columns(tag) for s in strata]
            extra["paired_spearman"].append(EV.paired_spearman(w, [(m, "M1") for m in base if m not in ("M1", "POS")] + [("M1", "POS")],
                                                               max(200, cfg.n_boot // 5), cfg.seed).with_columns(tag))
            extra["improvement"].append(EV.improvement_table(ev["paired"], ev["status"], None, cfg.practical_margin_pct))
        evs.append(ev)
    if not evs:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no target could be evaluated", missing=["targets"], detail={"notes": notes})

    def cat(k: str, src: Any = None) -> Any:
        fs = [e[k] for e in (src or evs) if e.get(k) is not None and getattr(e[k], "height", 0)]
        return pl.concat(fs, how="diagonal_relaxed") if fs else None

    long = pl.concat([e["long"] for e in evs], how="diagonal_relaxed")
    frames = {k: cat(k) for k in ("metrics", "paired", "intervals", "folds", "status", "within", "ranking", "position_effects", "cohort_audit", "ladder_log",
                                  "learner_log", "audit")}
    for k, v in extra.items():
        fs = [x for x in v if x is not None and x.height]
        frames[k] = pl.concat(fs, how="diagonal_relaxed") if fs else None
    frames["rung_plan"] = LD.plan_frame(plan)
    frames["notes"] = pl.DataFrame(notes, infer_schema_length=None) if notes else None
    # sensitivity: aggregation / drill scope variants of the single-representation rungs on T0
    smods, sinv = LD.sensitivity_models(cfg, plan, analysis, fold_of, exp.cfg.science.outer_folds, log)
    if len(smods) > 1 and "T0" in tids:
        try:
            sres = LD.fit(exp, analysis, "T0", TG.TARGETS["T0"].column, smods, {}, [], cfg.seed)
            sm = [c.removeprefix("pred_") for c in sres["wide"].columns if c.startswith("pred_") and c != "pred_MEAN"]
            sl = TF.long_predictions(sres["wide"], "m16_sensitivity_T0", sm)
            sw = wides_from_long(sl.with_columns(pl.lit("T0").alias("target_id"))).get("T0")
            if sw is not None:
                frames["sensitivity_paired"] = TF.paired_comparisons(sw, [(m, "M1") for m in sm if m != "M1"], cfg.n_boot, cfg.seed, "strict_common")
                frames["sensitivity_inventory"] = sinv
        except (ResearchBlocked, TF.ComparisonError) as e:
            notes.append({"item": "sensitivity ladder", "note": str(e)})
    run.data_file(long, "predictions/m16_ladder_oof_long", inputs)
    atomic_json({"rungs": LD.plan_frame(plan).to_dicts(), "learners": list(cfg.learners), "embedding_k": cfg.embedding_k, "ladder_dim": cfg.ladder_dim,
                 "primary_aggregation": cfg.primary_aggregation, "primary_family": [list(p) for p in prim], "seed": cfg.seed,
                 "engine": "nflbdb m12.advanced_evaluation.oof_ladder (ridge, inner-CV alpha, train-fold design, unpenalised position)",
                 "selection_log": json.loads(json.dumps(log, default=str))[:500]}, run.root / "models" / "ladder_models.json", overwrite=True)
    save_step(run, ctx, "E16.C4", "ladder", frames, inputs)
    return {**frames, "long": long, "wides": {e["target_id"]: e["wide_strict"] for e in evs if e.get("wide_strict") is not None}}


def _targets_frame(run: Any) -> Any:
    """The athlete target table written by E16.A2 (exposure columns for the stratified errors)."""
    import polars as pl

    p = run.root / "features" / "targets" / "athlete_targets.parquet"
    return pl.read_parquet(p).with_columns(pl.col("nfl_id").cast(pl.Int64)) if p.is_file() else pl.DataFrame(schema={"nfl_id": pl.Int64})


# ---------------------------------------------------------------------- C5 negative controls
def run_controls16(cfg: Any, exp: Any, F: dict[str, Any], L: dict[str, Any], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    from . import ladder as LD

    emb = {r: t for r, t in F["tables"].items() if r in LD.EMBEDDING_RUNGS}
    if not emb:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no eligible embedding rung: nothing to control", missing=["B3–B6"])
    inputs = {"ladder": run._sigs().get("ladder/E16.C4", {}).get("signature"), "n_shuffle": cfg.n_shuffle, "reduced": list(cfg.reduced_dims)}
    hit = load_step(run, cfg, "E16.C5", "controls", inputs)
    if hit is not None:
        ctx["cache"] = "hit"
        return hit
    a = F["analysis"]
    positions = dict(zip(a["nfl_id"].to_list(), a["position_group"].fill_null("?").to_list(), strict=True))
    nulls, summs = [], []
    for tid, w in L["wides"].items():
        if tid not in TG.TARGETS:
            continue
        n, s = LD.controls(exp, a, tid, TG.TARGETS[tid].column, exp.folds_by_athlete, emb, {"wide_strict": w}, cfg, positions)
        nulls.append(n)
        summs.append(s)
    frames = {"null": pl.concat([x for x in nulls if x.height], how="diagonal_relaxed") if any(x.height for x in nulls) else None,
              "summary": pl.concat([x for x in summs if x.height], how="diagonal_relaxed") if any(x.height for x in summs) else None}
    save_step(run, ctx, "E16.C5", "controls", frames, inputs)
    return frames


# ---------------------------------------------------------------------- C6 positions
def run_positions16(cfg: Any, exp: Any, F: dict[str, Any], L: dict[str, Any], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    from . import ladder as LD
    from . import positions as PS

    mods = LD.build_models(cfg, F["plan"], F["analysis"], exp.folds_by_athlete, F["tables"], [])
    nb = max(200, cfg.n_boot // 5)
    parts: dict[str, list] = {k: [] for k in ("metrics", "paired", "sample_sizes", "eligibility", "strategies", "effects")}
    for tid, w in L["wides"].items():
        if tid not in TG.TARGETS:
            continue
        tag = pl.lit(tid).alias("target_id")
        base = [m for m in mods if f"pred_{m}" in w.columns]
        met, pq = PS.global_by_position(w, base, nb, cfg.seed)
        sep, elig = PS.separate_models(exp, F["analysis"], TG.TARGETS[tid].column, {m: mods[m] for m in base if m != "POS"},
                                       sorted(set(w["position_group"].to_list())), cfg.position_min_n)
        for k, df in (("metrics", met), ("paired", pq), ("sample_sizes", PS.sample_sizes(w)), ("eligibility", elig),
                      ("strategies", PS.compare_strategies(w, sep, [m for m in base if m != "POS"], nb, cfg.seed)),
                      ("effects", PS.position_effects_all(w, base, nb, cfg.seed))):
            if df is not None and df.height:
                parts[k].append(df.with_columns(tag))
    frames = {k: (pl.concat(v, how="diagonal_relaxed") if v else None) for k, v in parts.items()}
    for k, df in frames.items():
        if df is not None:
            ctx["outputs"] += run.table(df, k, "positions")
    return frames


# ---------------------------------------------------------------------- C7 hypotheses
def run_hypotheses(cfg: Any, exp: Any, F: dict[str, Any], T: dict[str, Any], gate_a: Any, support: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    from ..status import COMPLETED
    from . import hypotheses as HY
    from . import ladder as LD

    status, paired, metrics, inv, declared = [], [], [], [], []
    log: list = []
    for hid, h in HY.HYPOTHESES.items():
        for t in h["targets"]:
            st, why = HY.prerequisites(hid, t, F["analysis"], gate_a, T["targets"] if T else None, support, cfg)
            status.append({"hypothesis": hid, "question": h["question"], "target": t, "status": st, "reason": why})
            if st != COMPLETED:
                continue
            mods, i = HY.drill_models(cfg, F["plan"], F["analysis"], exp.folds_by_athlete, exp.cfg.science.outer_folds, hid, log)
            if i.height:
                inv.append(i)
            tid, col = HY.target_column(t)
            try:
                res = LD.fit(exp, F["analysis"], tid, col, mods, {}, [], cfg.seed)
                ev = BL.evaluate_target(res, exp.folds_by_athlete, cfg.n_boot, cfg.seed, HY.family(mods))
            except ResearchBlocked as e:
                status[-1].update({"status": e.status, "reason": e.reason})
                continue
            tag = [pl.lit(hid).alias("hypothesis"), pl.lit(tid).alias("target_id")]
            for coll, k in ((paired, "paired"), (metrics, "metrics"), (declared, "status")):
                if ev.get(k) is not None and ev[k].height:
                    coll.append(ev[k].with_columns(*tag))
    frames = {"hypotheses_status": pl.DataFrame(status, infer_schema_length=None),
              "hypotheses_paired": pl.concat(paired, how="diagonal_relaxed") if paired else None,
              "hypotheses_metrics": pl.concat(metrics, how="diagonal_relaxed") if metrics else None,
              "hypotheses_declared": pl.concat(declared, how="diagonal_relaxed") if declared else None,
              "hypotheses_inventory": pl.concat(inv, how="diagonal_relaxed") if inv else None}
    for k, df in frames.items():
        if df is not None and df.width:
            ctx["outputs"] += run.table(df, k, "hypotheses")
    return frames


# ---------------------------------------------------------------------- G16 extended + report
def run_gate_extended(cfg: Any, run: Any, ctx: dict[str, Any]) -> Any:
    t = {"gate_a": _frame(run, "gate_a", "target"), "scorecard": _frame(run, "scorecard", "exploration"), "g16b": _frame(run, "g16b_gate", "synthesis"),
         "cohort_audit": _frame(run, "cohort_audit", "ladder"), "plan": _frame(run, "rung_plan", "ladder"), "status": _frame(run, "status", "ladder"),
         "paired": _frame(run, "paired", "ladder"), "controls_summary": _frame(run, "summary", "controls"), "registry": _frame(run, "registry", "representation"),
         "vs_controls": _frame(run, "controlled_vs_controls", "retrieval"), "damage": _frame(run, "repeat_damage", "retrieval"), "cem": _frame(run, "cem", "retrieval"),
         "mde": _frame(run, "mde", "detectability")}
    ext = GT.g16_extended(t, cfg)
    ctx["outputs"] += run.table(ext, "m16_gate_extended", "synthesis")
    return ext


def run_report(cfg: Any, run: Any, ctx: dict[str, Any]) -> str:
    from . import reporting as REP

    text = REP.build(run, cfg)
    ctx["outputs"] += [str(p) for p in run.save_markdown(text, "m16_research_report")]
    return text


__all__ = ["analysis16", "chosen_reps", "data_from_experiment", "feature_table", "file_identity", "ladder_targets", "load_step", "m15_config", "run_baselines",
           "run_cohort", "run_context", "run_controlled_retrieval", "run_controls16", "run_detectability", "run_features", "run_gate", "run_gate_extended",
           "run_hypotheses", "run_ladder16", "run_leakage", "run_positions16", "run_provenance", "run_reliability", "run_repeatability", "run_report",
           "run_retrieval", "run_target_lab", "run_targets", "save_step", "tconfig", "wides_from_long"]
